(* Software pipelining, gh-ocannl-487 phase 1: the double-buffered cooperative Stage over the
   staged+tensorized matmul composition (the lane-aware staging leg of schedule_mma_matmul), pinned
   structurally and by executed parity.

   [mc = ma * mb] (32x32 times 32x32) under the staged schedule: expand + annotate the zeroing,
   split i by 16 into Grid x Serial, split k by 16 into Serial x Serial, swap j and i_i inside k_o,
   cooperatively Stage ma over [i_i; k_i] and mb over [k_i; j] with [pipeline_depth = d], then
   Tensorize the i_i x j x k_i micro-kernel. Both stages anchor at the Serial k_o — the rotor.

   THE INVARIANT (the whole point of the transform being a pure schedule transform): the pipelined
   rendering is bitwise identical to depth 1. The compute subtree is remapped identically — only the
   copy for k_o+1 is issued before the compute of k_o (prologue before the loop, one shared in-loop
   barrier with both stages' prefetches grouped behind it, a trailing barrier after), and the
   renderer's buffer rotation makes every read resolve to the copy holding exactly the values the
   unpipelined form reads. So depth 2 must match depth 1 BITWISE on the GPU backends (while both
   only approximate the serial twin — the tile reduction reassociates, which is Tensorize's license,
   not pipelining's). Depth 2 only: both the portable form (phase 1) and the CUDA cp.async arm
   (phase 2) have single-step lookahead, and depth 3 is pinned as a typed rejection.

   THE OTHER INVARIANT (gh-ocannl-567): the same-anchor load-phase grouping and the Tile_mma-bracket
   barrier elision, both of which PR #303 landed [opt.pipelined]-gated and gh-567 widened to depth
   1, change synchronization structure and never values. The reference is the un-elided leg below:
   the applied depth-1 schedule with the pre-gh-567 barrier structure rebuilt on top of it (a
   barrier after every load-phase statement and one closing the k-block — adding barriers is always
   conservative). Depth 1 must match it BITWISE.

   Structural pins (backend-independent, on the applied schedule): the pipelined map records both
   tiles at the requested depth with k_o as rotor; the k_o body holds ONE barrier for both
   same-anchor stages with both If-guarded prefetches grouped behind it; both prologue loads sit
   outside the loop. Depth 1 groups the same way and keeps exactly the one phase barrier the
   intrinsic does not provide (its bracket ends each k-block, but only some rendering forms open one
   per iteration — the fragment-scope form opens it once, outside the loop). Canonical digests
   distinguish the depths, and the pipelined companion section alone must separate byte-identical IR
   whose side-table depths differ (the phase-2 aliasing hazard, pinned by record surgery).
   Validation legs pin the illegal placements. The C backends cannot express shared placement and
   must reject the compile cleanly (the same pinning as the SMEM matmul test); the schedule-level
   validation errors are the typed Illegal_schedule declines autotune candidates see. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Sched = Ir.Schedule
module SC = Ir.Schedule_cache
module SO = Ir.Schedule_outcome
module Asgns = Ir.Assignments
module Numerics = Ir.Numerics

let () = Utils.settings.output_debug_files_in_build_directory <- true
let () = Numerics.set_policy { (Numerics.get ()) with tf32_matmuls = false }

open Verdict.Claims

(* Zeros compare equal to zeros. A fragment mapping that reads outside the staged block, a kernel
   that never ran, or a reference whose own setup silently collapsed all yield all-zeros, and a
   parity check between two zero arrays passes while covering nothing (gh-ocannl-481 item 3). Every
   reference array is pinned nonzero where it is produced, so the parity claims below have
   content. *)
let nonzero name (a : float array) =
  if not (Array.exists a ~f:(fun x -> Float.(x <> 0.))) then
    failwith (name ^ ": the reference is all zeros — the parity checks against it are vacuous");
  a

let approx a b = Float.(abs (a - b) < 1e-2)
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let skipped = Verdict.skipped ~backend:backend_name

(* Intentional dialect identity: these branches count the exact MSL versus CUDA/HIP array and
   barrier tokens. Async-copy availability is queried separately through codegen capabilities. *)
let on_metal = String.is_substring backend_name ~substring:"metal"
let on_gpu = Sched.backend_is_gpu backend_name

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name

(* The maximal single-child chains of statement-level loops: one symbol list per top-level nest. *)
let nest_paths (llc : LL.t) : Ir.Indexing.symbol list list =
  let strip stmts = List.filter stmts ~f:(function LL.Noop | LL.Comment _ -> false | _ -> true) in
  let rec path (llc : LL.t) : Ir.Indexing.symbol list =
    match llc with
    | LL.For_loop { index; body; _ } ->
        index :: (match strip (LL.flat_lines [ body ]) with [ single ] -> path single | _ -> [])
    | LL.If { body; _ } -> path body
    | _ -> []
  in
  List.filter_map (LL.flat_lines [ llc ]) ~f:(fun stmt ->
      match path stmt with [] -> None | p -> Some p)

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let n = 32
let bm = 16
let simd_width = 32

(* Statement-level walks for the structural pins. *)
let rec find_loop sym (llc : LL.t) : LL.t option =
  match llc with
  | LL.For_loop { index; _ } when Ir.Indexing.equal_symbol index sym -> Some llc
  | LL.For_loop { body; _ } -> find_loop sym body
  | LL.Seq (a, b) -> ( match find_loop sym a with Some _ as r -> r | None -> find_loop sym b)
  | LL.If { body; _ } -> find_loop sym body
  | _ -> None

let rec count_stmts ~f (llc : LL.t) : int =
  (if f llc then 1 else 0)
  +
  match llc with
  | LL.Seq (a, b) -> count_stmts ~f a + count_stmts ~f b
  | LL.For_loop { body; _ } | LL.If { body; _ } -> count_stmts ~f body
  | _ -> 0

(* Per-element zero-fringe guards ([cond ? src : 0]) surviving in the statement tree. *)
let rec where_guards (llc : LL.t) : int =
  match llc with
  | LL.Seq (a, b) -> where_guards a + where_guards b
  | LL.For_loop { body; _ } | LL.If { body; _ } -> where_guards body
  | LL.Tile_mma { fallback; _ } -> where_guards fallback
  | LL.Set { llsc; _ } | LL.Set_local (_, llsc) -> where_guards_s llsc
  | _ -> 0

and where_guards_s (llsc : LL.scalar_t) : int =
  match llsc with
  | LL.Ternop (Ir.Ops.Where, (a, _), (b, _), (c, _)) ->
      1 + where_guards_s a + where_guards_s b + where_guards_s c
  | LL.Ternop (_, (a, _), (b, _), (c, _)) -> where_guards_s a + where_guards_s b + where_guards_s c
  | LL.Binop (_, (a, _), (b, _)) -> where_guards_s a + where_guards_s b
  | LL.Unop (_, (a, _)) -> where_guards_s a
  | LL.Local_scope { body; _ } -> where_guards body
  | _ -> 0

let rec writes_tile ~is_tile (llc : LL.t) : bool =
  match llc with
  | LL.Set { tn; _ } -> is_tile tn
  | LL.Seq (a, b) -> writes_tile ~is_tile a || writes_tile ~is_tile b
  | LL.For_loop { body; _ } | LL.If { body; _ } -> writes_tile ~is_tile body
  | _ -> false

let () =
  let mav = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 13) *. 0.25) in
  let mbv = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 17) -. 8.) in
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in

  (* --- Serial twin (approximate reference; the bitwise references are the staged depths). --- *)
  let%op mc0 = ma * mb in
  let serial_comp = named "pipe_mm_serial" (Train.forward mc0) in
  let ctx_s = Context.auto () in
  let ctx_s, routine_s =
    Context.compile ~lowered_transform:(fun opt -> [ opt ]) ctx_s serial_comp Ir.Indexing.Empty
  in
  let ctx_s = Context.run ctx_s routine_s in
  let got_serial = nonzero "pd_serial" (Context.get_values ctx_s mc0.Tensor.value) in

  (* --- The staged + tensorized schedule with a pipeline depth knob. Returns the rotor. --- *)
  let staged_schedule ~pipeline_depth ~out (opt : LL.optimized) :
      Sched.schedule * Ir.Indexing.symbol =
    let paths = nest_paths opt.LL.llc in
    let i, j, k =
      match List.find_exn paths ~f:(fun path -> List.length path = 3) with
      | [ i; j; k ] -> (i, j, k)
      | _ -> assert false
    in
    let ez, zsyms = Sched.expand_zero ~tn:out in
    let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
    let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let rz = Sched.Retype { axis = zj; ty = LL.Workgroup } in
    let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bm ~outer:LL.Serial ~inner:LL.Serial in
    let tz, _lane = Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width () in
    let stage source tile_loops =
      Sched.Stage
        {
          source;
          tile_loops;
          shared = true;
          cooperative = Some simd_width;
          hoisted = false;
          swizzle = None;
          pad_stride = None;
          pipeline_depth;
          tile_prec = None;
        }
    in
    ( [
        ez;
        sp_zi;
        rz;
        sp_i;
        sp_k;
        Sched.Swap { outer = j; inner = k_o };
        Sched.Swap { outer = i_i; inner = k_o };
        stage ma.Tensor.value [ i_i; k_i ];
        stage mb.Tensor.value [ k_i; j ];
        tz;
      ],
      k_o )
  in

  (* The gh-567 reference: rebuild the pre-gh-567 barrier structure on top of the applied depth-1
     schedule — drop the surviving barriers, put one after every load-phase statement, and close the
     k-block with one. Adding barriers can only remove races, so this leg is a sound reference for
     the invariant regardless of what the elision does. *)
  let reinsert_barriers ~is_tile ~k_o (llc : LL.t) : LL.t =
    let rec go (llc : LL.t) : LL.t =
      match llc with
      | LL.For_loop { index; from_; to_; body; axis } when Ir.Indexing.equal_symbol index k_o ->
          let stmts =
            LL.flat_lines [ body ]
            |> List.filter ~f:(function LL.Workgroup_barrier -> false | _ -> true)
            |> List.concat_map ~f:(fun s ->
                if writes_tile ~is_tile s then [ s; LL.Workgroup_barrier ] else [ s ])
          in
          LL.For_loop
            { index; from_; to_; axis; body = LL.unflat_lines (stmts @ [ LL.Workgroup_barrier ]) }
      | LL.For_loop { index; from_; to_; body; axis } ->
          LL.For_loop { index; from_; to_; axis; body = go body }
      | LL.Seq (a, b) -> LL.Seq (go a, go b)
      | LL.If { cond; body } -> LL.If { cond; body = go body }
      | other -> other
    in
    go llc
  in
  (* One compile per leg; the transform captures the applied optimized code and the rotor for the
     backend-independent structural pins (on the C backends the capture still happens — the
     rejection below comes later, from the backend's compile of the transformed code). Key 0 is the
     un-elided depth-1 reference. *)
  let captured : (int, LL.optimized * Ir.Indexing.symbol) Hashtbl.t = Hashtbl.create (module Int) in
  let compile_depth ?post ~key ~pipeline_depth ~out comp =
    let transform opt =
      let schedule, k_o = staged_schedule ~pipeline_depth ~out opt in
      let applied = Sched.apply schedule opt in
      let applied =
        match post with
        | None -> applied
        | Some f ->
            let is_tile tn = Set.mem applied.LL.workgroup_shared tn in
            { applied with LL.llc = f ~is_tile ~k_o applied.LL.llc }
      in
      Hashtbl.set captured ~key ~data:(applied, k_o);
      applied
    in
    let ctx = Context.auto () in
    Context.compile ~lowered_transform:(fun o -> [ transform o ]) ctx comp Ir.Indexing.Empty
  in
  let%op mc0b = ma * mb in
  let%op mc1 = ma * mb in
  let%op mc2 = ma * mb in
  let legs =
    [
      (0, mc0b, named "pipe_mm_d1_barriers" (Train.forward mc0b));
      (1, mc1, named "pipe_mm_d1" (Train.forward mc1));
      (2, mc2, named "pipe_mm_d2" (Train.forward mc2));
    ]
  in
  let compile_leg (key, mc, comp) =
    let post = if key = 0 then Some reinsert_barriers else None in
    compile_depth ?post ~key ~pipeline_depth:(max key 1) ~out:mc.Tensor.value comp
  in
  (if on_gpu then (
     let results =
       List.map legs ~f:(fun ((key, mc, _) as leg) ->
           let ctx, routine = compile_leg leg in
           let ctx = Context.run ctx routine in
           (key, Context.get_values ctx mc.Tensor.value))
     in
     let got_ref = List.Assoc.find_exn results 0 ~equal:Int.equal in
     let got_d1 = List.Assoc.find_exn results 1 ~equal:Int.equal in
     let got_d2 = List.Assoc.find_exn results 2 ~equal:Int.equal in
     p_all "staged depths approximate the serial twin (GPU) or clean rejection (CPU)" results
       ~f:(fun (_, got) -> Array.for_all2_exn got got_serial ~f:approx);
     (* The invariant: pipelining is a pure prefetch-timing transform. *)
     p_all2 "depth 2 matches depth 1 BITWISE" got_d2 got_d1 ~f:Float.equal;
     (* The gh-567 invariant: grouping and eliding barriers is a pure synchronization transform. *)
     p_all2 "depth 1 matches the un-elided barrier structure BITWISE" got_d1 got_ref ~f:Float.equal;
     let count_sub src sub =
       String.substr_index_all src ~may_overlap:false ~pattern:sub |> List.length
     in
     (let src1 = Generated.read "pipe_mm_d1" in
      let src2 = Generated.read "pipe_mm_d2" in
      (* The depth-2 kernel rotates both tile buffers ([% 2] terms on reads and writes) and
         allocates each tile at twice the depth-1 size (16x16 -> [512] and 16x32 -> [1024] floats);
         the depth-1 kernel has no rotation and single-size tiles. *)
      let decl_ok src d =
        if on_metal then
          count_sub src ("[" ^ Int.to_string (256 * d) ^ "]") >= 1
          && count_sub src ("[" ^ Int.to_string (512 * d) ^ "]") >= 1
        else true
      in
      p "depth 2 rotates the buffers the depth-1 kernel does not"
        (count_sub src2 "% 2" >= 4 && count_sub src1 "% 2" = 0 && decl_ok src1 1 && decl_ok src2 2));
     (* gh-487 phase 2: where the backend has an async-copy arm, the depth-2 staging renders as
        async copies — both prologues and both prefetches, 4 call sites — with one wait-then-barrier
        opening each rotor iteration, while depth 1 stays fully synchronous: the pd1/pd2 pair keeps
        comparing prefetch timing only. The arm's presence is probed through the capability it gates
        identically (CUDA advertises [mma_pipeline_depths] exactly where [async_copy] is provided,
        sm_80+), so a pre-Ampere CUDA device expects the portable form rather than failing (Codex P2
        on PR #317). Backends without the arm (Metal — which advertises depths for the portable form
        — and HIP) keep the synchronous rendering at both depths: a real check that the hook does
        not leak, not a skip. *)
     (let src1 = Generated.read "pipe_mm_d1" in
      let src2 = Generated.read "pipe_mm_d2" in
      let asynchronous_staging =
        (Context.codegen_capabilities (Context.auto ())).asynchronous_staging_copy
        &&
        match (Context.hardware_limits (Context.auto ())).Ir.Backend_intf.mma with
        | Some m -> not (List.is_empty m.Ir.Backend_intf.mma_pipeline_depths)
        | None -> false
      in
      p "async copies appear exactly on the depth-2 async arm"
        (if asynchronous_staging then
           count_sub src2 "ocannl_cp_async4(&" = 4
           && count_sub src2 "ocannl_cp_async_wait_all();" = 1
           && count_sub src1 "ocannl_cp_async" = 0
         else count_sub src2 "ocannl_cp_async" = 0 && count_sub src1 "ocannl_cp_async" = 0));
     (* The count in the EMITTED kernel, where the intrinsic's own brackets are visible too: the
        reference and depth 1 differ by exactly the two barrier statements gh-567 dropped (the
        k-block has one loop body in the source, so this counts statements, not executions). *)
     let src1 = Generated.read "pipe_mm_d1" in
     let src_ref = Generated.read "pipe_mm_d1_barriers" in
     let barrier_kw = if on_metal then "threadgroup_barrier" else "__syncthreads" in
     p "the emitted depth-1 kernel sheds exactly the two elided barriers"
       (count_sub src_ref barrier_kw - count_sub src1 barrier_kw = 2))
   else
     (* The C backends reject the shared staging composition at compile (workgroup-shared placement
        is not renderable there) — same clean rejection as the SMEM matmul test, at any depth. *)
     let rejects leg =
       try
         ignore (compile_leg leg : Context.t * Context.routine);
         false
       with Invalid_argument msg -> String.is_substring msg ~substring:"not supported"
     in
     p_all "staged depths approximate the serial twin (GPU) or clean rejection (CPU)" legs
       ~f:rejects;
     skipped "depth 2 matches depth 1 BITWISE";
     skipped "depth 1 matches the un-elided barrier structure BITWISE";
     skipped "depth 2 rotates the buffers the depth-1 kernel does not";
     skipped "async copies appear exactly on the depth-2 async arm";
     skipped "the emitted depth-1 kernel sheds exactly the two elided barriers");

  (* --- Structural pins on the applied schedules (all backends; captured by the transforms above).
     --- *)
  let opt_ref, k_o_ref = Hashtbl.find_exn captured 0 in
  let opt_d1, k_o1 = Hashtbl.find_exn captured 1 in
  let opt_d2, k_o2 = Hashtbl.find_exn captured 2 in
  (* Both stages' tiles, at either depth: the workgroup-shared set. *)
  let is_tile opt tn = Set.mem opt.LL.workgroup_shared tn in
  p "depth 1 records no pipelined tiles" (Map.is_empty opt_d1.LL.pipelined);
  p "depth 2 records both tiles at depth 2 with k_o as rotor"
    (Map.length opt_d2.LL.pipelined = 2
    && Map.for_all opt_d2.LL.pipelined ~f:(fun { LL.pt_depth; pt_rotor } ->
        pt_depth = 2 && Ir.Indexing.equal_symbol pt_rotor k_o2));
  let barriers llc = count_stmts llc ~f:(function LL.Workgroup_barrier -> true | _ -> false) in
  let body_of sym llc =
    match find_loop sym llc with Some (LL.For_loop { body; _ }) -> body | _ -> LL.Noop
  in
  let ref_body = body_of k_o_ref opt_ref.LL.llc in
  let d1_body = body_of k_o1 opt_d1.LL.llc in
  let d2_body = body_of k_o2 opt_d2.LL.llc in
  (* Depth 1 (gh-567): the two same-anchor stages share ONE barrier phase per k-block — both load
     nests back to back, then the barrier the compute's reads need — and the phase CLOSER is left to
     [Tile_mma]'s own trailing bracket ("Tile_mma is a barrier"). The k-block's opener stays
     explicit: only some intrinsic rendering forms bracket the block on both sides per iteration
     (the fragment-scope form opens once, outside the anchor loop). Without gh-567 that k-block
     carried FOUR barriers — the reference leg rebuilds three of them (the duplicated closer is
     semantically the same barrier). Depth 2 additionally elides the opener, against the previous
     iteration's bracket: both If-guarded prefetches (counted statement-level, so the lane
     restriction Ifs nested within load nests are not miscounted) sit behind no explicit barrier at
     all. Either way the one explicit barrier left in the routine outside the k-block is the
     trailing one after the contraction's store-back (its elision would need the region contract
     across the store — kept conservative). *)
  p "the un-elided reference carries a barrier per staged phase"
    (barriers ref_body = 3 && barriers opt_ref.LL.llc = 3);
  p "depth 1 groups both stages into one phase and leaves the closer to Tile_mma's bracket"
    (barriers d1_body = 1 && barriers opt_d1.LL.llc = 1);
  p "depth 2 leaves the per-iteration phase to Tile_mma's bracket, one trailing barrier"
    (barriers d2_body = 0 && barriers opt_d2.LL.llc = 1);
  let stmt_prefetches opt llc =
    List.count (LL.flat_lines [ llc ]) ~f:(function
      | LL.If { body; _ } -> writes_tile ~is_tile:(is_tile opt) body
      | _ -> false)
  in
  p "depth 2 issues one If-guarded prefetch per stage inside k_o"
    (stmt_prefetches opt_d2 d2_body = 2 && stmt_prefetches opt_d1 d1_body = 0);
  (* Each stage's copy is a single guarded [Set] of its tile: depth 1 has both inside k_o; depth 2
     has two inside (the prefetches) and two outside (the prologues). *)
  let tile_sets opt llc =
    count_stmts llc ~f:(function LL.Set { tn; _ } -> is_tile opt tn | _ -> false)
  in
  let d1_total = tile_sets opt_d1 opt_d1.LL.llc and d1_in = tile_sets opt_d1 d1_body in
  let d2_total = tile_sets opt_d2 opt_d2.LL.llc and d2_in = tile_sets opt_d2 d2_body in
  p "depth 2 places both prologue loads outside k_o"
    (d1_total = 2 && d1_in = 2 && d2_total = 4 && d2_in = 2);
  (* gh-ocannl-566: the prefetch's zero-fringe edge guard is provably in range — the [k := k+1]
     substitution widens its index interval past the extent, but the [If (k < to_)] the prefetch
     sits under excludes exactly the block that would overrun. Before [simplify_llc] narrowed the
     interval environment from enclosing [If] conditions, the guard survived as a per-element
     ternary in every prefetch while the depth-1 form's identical guard folded on the loop range
     alone — pure overhead, and a measurement bias against pipelining. Neither depth carries one
     now. *)
  p "the depth-2 prefetch's edge guard folds under its If"
    (where_guards opt_d2.LL.llc = 0 && where_guards opt_d1.LL.llc = 0);

  (* --- Cache identity: canonical digests distinguish the depths. In phase 1 (depth <= 2) the
     depths already differ in the IR; the pipelined companion section is what keeps depths >= 2
     distinct once the phase-2 async-copy arms widen the legal range (their IR is identical, only
     the rotation modulus differs) — pinned here by digesting depth 2's optimized value against a
     copy whose side-table alone records depth 3 (byte-identical IR by construction), rather than
     discovered as an alias later. --- *)
  let dg opt = SC.digest (SC.canonicalize opt) in
  p "canonical digests distinguish depths 1 and 2" (not (String.equal (dg opt_d1) (dg opt_d2)));
  let opt_d2_as_d3 =
    {
      opt_d2 with
      LL.pipelined = Map.map opt_d2.LL.pipelined ~f:(fun pt -> { pt with LL.pt_depth = 3 });
    }
  in
  p "the pipelined companion section alone distinguishes same-IR depths"
    (not (String.equal (dg opt_d2) (dg opt_d2_as_d3)));

  (* --- Validation legs: the illegal placements are typed schedule errors, not crashes. --- *)
  let expect_invalid name ~substring:sub (f : unit -> unit) =
    match f () with
    | () -> p name false
    | exception Invalid_argument msg -> p name (String.is_substring msg ~substring:sub)
  in
  let reapply ~probe ~f () =
    (* Re-lower a fresh forward to probe schedule application hermetically. *)
    let%op mcv = ma * mb in
    let comp = named probe (Train.forward mcv) in
    let transform opt =
      f ~out:mcv.Tensor.value opt;
      opt
    in
    ignore
      (Context.compile
         ~lowered_transform:(fun o -> [ transform o ])
         (Context.auto ()) comp Ir.Indexing.Empty
        : Context.t * Context.routine)
  in
  expect_invalid "pipeline_depth 0 is rejected" ~substring:"pipeline_depth must be >= 1"
    (reapply ~probe:"pipe_mm_probe0" ~f:(fun ~out opt ->
         let schedule, _ = staged_schedule ~pipeline_depth:0 ~out opt in
         ignore (Sched.apply schedule opt : LL.optimized)));
  expect_invalid "pipeline_depth 3 is rejected (single-step lookahead)"
    ~substring:"not implemented in gh-ocannl-487"
    (reapply ~probe:"pipe_mm_probe3" ~f:(fun ~out opt ->
         let schedule, _ = staged_schedule ~pipeline_depth:3 ~out opt in
         ignore (Sched.apply schedule opt : LL.optimized)));
  expect_invalid "pipelining a non-cooperative (packing) Stage is rejected"
    ~substring:"requires cooperative staging"
    (reapply ~probe:"pipe_mm_probe1" ~f:(fun ~out:_ opt ->
         let paths = nest_paths opt.LL.llc in
         let k =
           match List.find_exn paths ~f:(fun path -> List.length path = 3) with
           | [ _; _; k ] -> k
           | _ -> assert false
         in
         let op =
           Sched.Stage
             {
               source = ma.Tensor.value;
               tile_loops = [ k ];
               shared = false;
               cooperative = None;
               hoisted = false;
               swizzle = None;
               pad_stride = None;
               pipeline_depth = 2;
               tile_prec = None;
             }
         in
         ignore (Sched.apply [ op ] opt : LL.optimized)));
  expect_invalid "pipelining at a non-Serial anchor is rejected" ~substring:"to be Serial"
    (reapply ~probe:"pipe_mm_probe2" ~f:(fun ~out opt ->
         (* Same composition, but the anchor k_o retyped to Grid before the stages: the rotor has no
            serial iteration order to rotate with. (Racy as a real schedule — irrelevant, the
            application must already reject it.) *)
         let paths = nest_paths opt.LL.llc in
         let i, j, k =
           match List.find_exn paths ~f:(fun path -> List.length path = 3) with
           | [ i; j; k ] -> (i, j, k)
           | _ -> assert false
         in
         let ez, zsyms = Sched.expand_zero ~tn:out in
         let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
         let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
         let rz = Sched.Retype { axis = zj; ty = LL.Workgroup } in
         let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
         let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bm ~outer:LL.Serial ~inner:LL.Serial in
         let schedule =
           [
             ez;
             sp_zi;
             rz;
             sp_i;
             sp_k;
             Sched.Swap { outer = j; inner = k_o };
             Sched.Swap { outer = i_i; inner = k_o };
             Sched.Retype { axis = k_o; ty = LL.Grid };
             Sched.Stage
               {
                 source = ma.Tensor.value;
                 tile_loops = [ i_i; k_i ];
                 shared = true;
                 cooperative = Some simd_width;
                 hoisted = false;
                 swizzle = None;
                 pad_stride = None;
                 pipeline_depth = 2;
                 tile_prec = None;
               };
           ]
         in
         ignore (Sched.apply schedule opt : LL.optimized)));

  (* --- The RENDERER's own precondition, one level below schedule application (gh-ocannl-569). The
     rotation term selects the copy for the current rotor iteration, so a read of a pipelined tile
     reached outside its rotor loop has no copy to name and the renderer must refuse it. Schedule
     application does not catch this — it validates the anchor, not every later read's position — so
     a candidate can construct and only fail at render, which is how it reached a Metal search as a
     hard [Invalid_argument] out of [C_syntax] and, unclassified at [Backend_codegen] under
     [strict_failure_classification], ended the whole search. Typed, it is one candidate's decline.

     Probed by pointing the applied depth-2 schedule's rotors at a symbol that encloses nothing —
     which is what "outside its rotor loop" is, without needing the particular composition that
     produced one. The public boundary is unchanged: [Context.compile] still renders the same
     [Invalid_argument] for a hand-written schedule. --- *)
  let rotorless_transform ~out opt =
    let schedule, _ = staged_schedule ~pipeline_depth:2 ~out opt in
    let applied = Sched.apply schedule opt in
    let stray = Ir.Indexing.get_symbol () in
    {
      applied with
      LL.pipelined = Map.map applied.LL.pipelined ~f:(fun pt -> { pt with LL.pt_rotor = stray });
    }
  in
  (* The sibling precondition, checked up front in [compile_proc]: the rotation counter is the rotor
     loop's SERIAL counter, so a rotor that is no longer Serial would silently freeze the buffer
     selection at copy 0. A composition that retypes the anchor after the stages were applied
     reaches it — which is what ended a Metal [autotune_candidate_release] search. *)
  let unserial_rotor_transform ~out opt =
    let schedule, _ = staged_schedule ~pipeline_depth:2 ~out opt in
    let applied = Sched.apply schedule opt in
    (* Re-pointed rather than retyped in place: retyping the anchor loop itself makes the
       accumulation race and [Low_level.validate_parallel] — which runs first — declines it for that
       instead, never reaching the rotor check. Pointing the rotor at a loop the composition already
       parallelized leaves the IR valid and isolates the check under test. *)
    let rec first_parallel (llc : LL.t) : Ir.Indexing.symbol option =
      match llc with
      | LL.For_loop { index; axis; body; _ } -> (
          match axis with LL.Serial -> first_parallel body | _ -> Some index)
      | LL.Seq (a, b) -> (
          match first_parallel a with Some _ as r -> r | None -> first_parallel b)
      | LL.If { body; _ } -> first_parallel body
      | _ -> None
    in
    let rotor = Option.value_exn (first_parallel applied.LL.llc) in
    {
      applied with
      LL.pipelined = Map.map applied.LL.pipelined ~f:(fun pt -> { pt with LL.pt_rotor = rotor });
    }
  in
  let declines_at_codegen ~feature ~substring transform =
    let%op mcr = ma * mb in
    match
      Context.compile_outcome
        ~lowered_transform:(fun o -> [ (transform ~out:mcr.Tensor.value) o ])
        ~provenance:SO.Candidate ~candidate:feature (Context.auto ())
        (named ("pipe_mm_" ^ feature) (Train.forward mcr))
        Ir.Indexing.Empty
    with
    | Ok _ | Error (SO.Fatal _) -> false
    | Error (SO.Classified { phase; cause; execution_effect }) -> (
        SO.equal_phase phase SO.Backend_codegen
        && SO.equal_execution_effect execution_effect SO.No_device_writes
        &&
        match cause with
        | SO.Unsupported { feature = f; detail } ->
            String.equal f feature && String.is_substring detail ~substring
        | _ -> false)
  in
  if on_gpu then (
    p "a read outside the rotor loop is a candidate's typed decline, not a fatal"
      (declines_at_codegen ~feature:"pipelined tile read outside its rotor loop"
         ~substring:"outside its rotor loop" rotorless_transform);
    p "so is a pipelined tile whose rotor loop is not Serial"
      (declines_at_codegen ~feature:"pipelined tile whose rotor loop is not Serial"
         ~substring:"no longer Serial" unserial_rotor_transform);
    let%op mcr2 = ma * mb in
    expect_invalid "and the same schedule written by hand still raises at Context.compile"
      ~substring:"outside its rotor loop" (fun () ->
        ignore
          (Context.compile
             ~lowered_transform:(fun o -> [ (rotorless_transform ~out:mcr2.Tensor.value) o ])
             (Context.auto ())
             (named "pipe_mm_rotorless_user" (Train.forward mcr2))
             Ir.Indexing.Empty
            : Context.t * Context.routine)))
  else (
    (* The C backends reject the workgroup-shared composition before either check is reached, so the
       renderer's preconditions are unreachable here. *)
    skipped "a read outside the rotor loop is a candidate's typed decline, not a fatal";
    skipped "so is a pipelined tile whose rotor loop is not Serial";
    skipped "and the same schedule written by hand still raises at Context.compile");

  (* --- Seeding pin (gh-ocannl-487): the depth twins follow the backend's advertised
     [mma_pipeline_depths] — one twin per staged seed per advertised depth, identical to its plain
     sibling except [sk_depth]; unstaged seeds are never twinned; an empty advertisement (the HIP
     state until its LDS arm lands, and cc) seeds no twins. Probed against synthetic capabilities so
     the pin is backend-independent, like the schedule_pad seed checks. --- *)
  let limits_with depths =
    {
      Ir.Backend_intf.no_hardware_limits with
      max_threads_per_workgroup = Some 1024;
      max_workgroup_memory_bytes = Some 32768;
      mma =
        Some
          {
            Ir.Backend_intf.mma_simd_width = 32;
            mma_tile = (8, 8, 8);
            mma_format_tiles =
              [
                ( (Ir.Backend_intf.Mma_f32, Ir.Backend_intf.Mma_f32, Ir.Backend_intf.Mma_f32),
                  (8, 8, 8) );
              ];
            mma_f16_wide_acc_scopes = [];
            mma_staged_layouts = [];
            mma_pipeline_depths = depths;
          };
    }
  in
  let seed_pin = ref None in
  let seed_transform opt =
    let seeds depths =
      Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits:(limits_with depths) opt
      |> List.filter ~f:(fun sp -> sp.Autotune.sk_mma && sp.Autotune.sk_gpu)
    in
    let advertised = seeds [ 2 ] and unadvertised = seeds [] in
    let staged = List.filter advertised ~f:(fun sp -> sp.Autotune.sk_bk > 0) in
    let plain_staged = List.filter staged ~f:(fun sp -> sp.Autotune.sk_depth = 1) in
    let twins = List.filter staged ~f:(fun sp -> sp.Autotune.sk_depth > 1) in
    let twin_of sp =
      List.exists twins ~f:(fun tw ->
          tw.Autotune.sk_depth = 2 && Poly.equal { tw with Autotune.sk_depth = 1 } sp)
    in
    seed_pin :=
      Some
        ( (not (List.is_empty plain_staged))
          && List.length twins = List.length plain_staged
          && List.for_all plain_staged ~f:twin_of
          && List.for_all advertised ~f:(fun sp ->
              sp.Autotune.sk_bk > 0 || sp.Autotune.sk_depth = 1),
          List.for_all unadvertised ~f:(fun sp -> sp.Autotune.sk_depth = 1) );
    opt
  in
  let%op mc4 = ma * mb in
  ignore
    (Context.compile
       ~lowered_transform:(fun o -> [ seed_transform o ])
       (Context.auto ())
       (named "pipe_mm_seeds" (Train.forward mc4))
       Ir.Indexing.Empty
      : Context.t * Context.routine);
  match !seed_pin with
  | Some (twinned, none_without) ->
      p "advertised depth seeds one pd2 twin per plain staged seed, staged only" twinned;
      p "no depth twins without the capability advertisement" none_without
  | None ->
      p "advertised depth seeds one pd2 twin per plain staged seed, staged only" false;
      p "no depth twins without the capability advertisement" false
