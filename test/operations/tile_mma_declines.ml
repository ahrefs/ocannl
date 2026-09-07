(* Decline visibility for Tile_mma renderings and pool-parallel Grid privatization (gh-ocannl-474 /
   gh-ocannl-479).

   Declines are silent by design — the scalar fallback and the serial loop are correct — but
   indistinguishable from the intended rendering in timings, which already cost one wrong conclusion
   (docs/proposals/tensorize-mma.md: parity passed while tensor cores never ran). This test pins the
   three visibility mechanisms:

   - The [Ir.C_syntax.mma_census]: each compiled [Tile_mma] statement records how it actually
   rendered (intrinsics / register-tiled / scalar fallback), and since gh-ocannl-626 the summary --
   including the three-way [tensorization] label -- is a field of the compiled routine
   ([Context.routine.mma]) rather than a global a call site brackets. Autotune reads it to annotate
   candidate timings; here we assert directly that a whole-triple [Tensorize] over a standard layout
   renders register-tiled on the C backends while a transposed-B layout (the gradient-GEMM shape)
   falls back to scalar, and that each routine's label reports which happened.

   - The per-chunk privatization byte cap (config [cc_grid_private_bytes_cap], default 256KB): a
   grid-outermost packed GEMM that re-packs the B~ panel per chunk (gh-ocannl-475's alternative
   shape) privatizes both tiles per chunk — under the default cap it fits at [n = 256] (80KB) and
   declines at [n = 1024] (272KB), rendering serial. Compile-only; the stderr diagnostics themselves
   are opt-in via [schedule_log_declines].

   - The autotune seeding pre-filter ([Autotune.sketch_seed_params], gh-ocannl-479): tensorized
   candidates that statically must render the scalar fallback are not seeded — transposed-B sites
   lose the whole-triple family (their packed flavors normalize the layout and stay), a
   non-hoistable transposed B additionally loses the hoisted-only Grid flavor (it would read B in
   place), and a column extent below one vector of lanes loses everything. Exactly one hoistable
   operand additionally seeds the mixed grid-outermost shape ([sk_pack_rest] with [sk_hoist],
   gh-ocannl-473); no hoistable operand seeds the grid-outermost per-chunk B~ re-packing shape
   ([sk_pack_rest] alone, gh-ocannl-475) when the tiles fit the privatization cap. Seed enumeration
   runs on the pre-schedule lowering with synthetic limits, so the printed counts are
   backend-independent. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments

let () = Utils.settings.output_debug_files_in_build_directory <- true

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

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_cpu = Sched.backend_is_cpu backend_name

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name

(* The zeroing nest is its own Grid loop and parallelizes independently of the accumulation loop:
   count constructs rather than testing presence. *)
let count_parallel_constructs src =
  List.length (String.substr_index_all src ~may_overlap:false ~pattern:"dispatch_apply")
  + List.length (String.substr_index_all src ~may_overlap:false ~pattern:"#pragma omp parallel for")

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

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

let accum_syms (opt : LL.optimized) =
  let paths = nest_paths opt.LL.llc in
  match List.find_exn paths ~f:(fun p -> List.length p = 3) with
  | [ i; j; k ] -> (i, j, k)
  | _ -> assert false

let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner })

(* Compile [comp] under [transform] and return the compiled routine's Tile_mma rendering summary. No
   bracket around the census global: since gh-ocannl-626 [Context.compile] collects it and the
   summary is a field of the routine, so a caller cannot forget to ask. *)
let compile_with_census ~transform comp =
  let ctx = Context.auto () in
  let _ctx, routine =
    Context.compile ~lowered_transform:(fun o -> [ transform o ]) ctx comp Ir.Indexing.Empty
  in
  routine.Context.mma

let renderings summary = List.map summary.Ir.C_syntax.renderings ~f:snd
let n = 64

(* === Census: whole-triple Tensorize, standard vs. transposed-B layout (C backends). === *)
let () =
  let mav = Array.init (n * n) ~f:(fun x -> Float.of_int (x % 13) *. 0.25) in
  let mbv = Array.init (n * n) ~f:(fun x -> Float.of_int (x % 17) -. 8.) in
  let ma = TDSL.ndarray mav ~label:[ "tmd_a" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "tmd_b" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let tensorize_schedule ~out (opt : LL.optimized) : Sched.schedule =
    let i, j, k = accum_syms opt in
    let ez, zsyms = Sched.expand_zero ~tn:out in
    let zj = match zsyms with [ _; zj ] -> zj | _ -> assert false in
    let rz = Sched.Retype { axis = zj; ty = LL.Workgroup } in
    let tz, _lane = Sched.tensorize ~i ~j ~k ~simd_width:n () in
    [ ez; rz; tz ]
  in
  let%op mc = ma * mb in
  let census =
    compile_with_census
      ~transform:(fun opt ->
        if on_cpu then Sched.apply (tensorize_schedule ~out:mc.Tensor.value opt) opt else opt)
      (named "tmd_regtile" (Train.forward mc))
  in
  p "census: standard layout renders register-tiled"
    ((not on_cpu)
    || List.equal Ir.C_syntax.equal_mma_rendering (renderings census)
         [ Ir.C_syntax.Mma_register_tiled ]);
  p "census: the honoured schedule's routine is labeled tensorized"
    ((not on_cpu)
    || Ir.C_syntax.equal_tensorization census.Ir.C_syntax.tensorization Ir.C_syntax.Tensorized);
  let mta = TDSL.ndarray mav ~label:[ "tmd_ta" ] ~output_dims:[ n; n ] () in
  let mtb = TDSL.ndarray mbv ~label:[ "tmd_tb" ] ~output_dims:[ n; n ] () in
  let%op td = mta +* "ik;jk=>ij" mtb in
  let census_tb =
    compile_with_census
      ~transform:(fun opt ->
        if on_cpu then Sched.apply (tensorize_schedule ~out:td.Tensor.value opt) opt else opt)
      (named "tmd_scalar_tb" (Train.forward td))
  in
  p "census: transposed-B layout falls back to the lane-0 scalar rendering"
    ((not on_cpu)
    || List.equal Ir.C_syntax.equal_mma_rendering (renderings census_tb)
         [ Ir.C_syntax.Mma_scalar_fallback ]);
  p "census: the declined schedule's routine is labeled scalar-fallback, not tensorized"
    ((not on_cpu)
    || Ir.C_syntax.equal_tensorization census_tb.Ir.C_syntax.tensorization
         Ir.C_syntax.Scalar_fallback)

(* === The per-chunk privatization cap: grid-outermost packed GEMM re-packing B~ per chunk
   (gh-ocannl-475's shape). Both packing Stages land inside the Grid body, so both tiles must
   privatize per chunk: at [n = 256] the footprint (bk*n + bm*bk floats = 80KB) fits under the
   default 256KB cap and the loop pool-parallelizes; at [n = 1024] (272KB) the cap trips and the
   loop renders serial. Compile-only. === *)
let () =
  let bm, bk = (64, 64) in
  let bpack_schedule ~a ~b ~out (opt : LL.optimized) : Sched.schedule =
    let i, j, k = accum_syms opt in
    let ez, zsyms = Sched.expand_zero ~tn:out in
    let zi = match zsyms with [ zi; _ ] -> zi | _ -> assert false in
    let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    let stage source tile_loops =
      Sched.Stage
        {
          source;
          tile_loops;
          shared = false;
          cooperative = None;
          hoisted = false;
          swizzle = None;
          pad_stride = None;
          pipeline_depth = 1;
          tile_prec = None;
        }
    in
    let tz, _lane = Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width:1 () in
    (* No [sink i_o [k_o]]: the Grid loop stays outermost, so the B~ pack at [k_o] lands inside the
       Grid body and each chunk re-packs its own panel. *)
    [ ez; sp_zi; sp_i; sp_k ] @ sink j [ k_o ] @ sink i_i [ k_o ]
    @ [ stage b [ k_i; j ]; stage a [ i_i; k_i ]; tz ]
  in
  let compile_bpack ?(run_parity = false) ~name ~dim () =
    let av = Array.init (dim * dim) ~f:(fun x -> Float.of_int (x % 13) *. 0.25) in
    let bv = Array.init (dim * dim) ~f:(fun x -> Float.of_int (x % 17) -. 8.) in
    let a = TDSL.ndarray av ~label:[ "tmd_bp_a" ] ~input_dims:[ dim ] ~output_dims:[ dim ] () in
    let b = TDSL.ndarray bv ~label:[ "tmd_bp_b" ] ~input_dims:[ dim ] ~output_dims:[ dim ] () in
    let%op c = a * b in
    let transform opt =
      if on_cpu then
        Sched.apply (bpack_schedule ~a:a.Tensor.value ~b:b.Tensor.value ~out:c.Tensor.value opt) opt
      else opt
    in
    let ctx = Context.auto () in
    let ctx, routine =
      Context.compile
        ~lowered_transform:(fun o -> [ transform o ])
        ctx
        (named name (Train.forward c))
        Ir.Indexing.Empty
    in
    (* Executed parity, not just structure: per-chunk privatization rewrites where the packed values
       live, so compare against a serial twin of the same computation. *)
    (if run_parity then
       let ctx = Context.run ctx routine in
       let got = Context.get_values ctx c.Tensor.value in
       let a2 =
         TDSL.ndarray av ~label:[ "tmd_bp_a2" ] ~input_dims:[ dim ] ~output_dims:[ dim ] ()
       in
       let b2 =
         TDSL.ndarray bv ~label:[ "tmd_bp_b2" ] ~input_dims:[ dim ] ~output_dims:[ dim ] ()
       in
       let%op c2 = a2 * b2 in
       let sctx = Context.auto () in
       let sctx, sroutine =
         Context.compile
           ~lowered_transform:(fun opt -> [ opt ])
           sctx
           (named (name ^ "_serial") (Train.forward c2))
           Ir.Indexing.Empty
       in
       let sctx = Context.run sctx sroutine in
       let want = nonzero "decl_serial" (Context.get_values sctx c2.Tensor.value) in
       p_all2 "per-chunk B~ re-pack matches the serial twin bitwise" got want ~f:Float.equal);
    Generated.read name
  in
  (let src = compile_bpack ~run_parity:true ~name:"tmd_bpack_fits" ~dim:256 () in
   (* Two parallel loops: the expanded zeroing nest and the packed accumulation loop. *)
   p "per-chunk B~ re-pack under the cap pool-parallelizes"
     ((not on_cpu)
     || (count_parallel_constructs src = 2 && String.is_substring src ~substring:"tile_")));
  let src = compile_bpack ~name:"tmd_bpack_over" ~dim:1024 () in
  (* Only the zeroing nest parallelizes; the accumulation loop's per-chunk tiles trip the cap and it
     renders serial. *)
  p "per-chunk B~ re-pack over the cap declines to serial"
    ((not on_cpu) || count_parallel_constructs src = 1)

(* === The seeding pre-filter (gh-ocannl-479) and the mixed grid-outermost seed (gh-ocannl-473).
   Synthetic limits pin the vector width (32 bytes = 8 f32 lanes), and seeds are enumerated on the
   pre-schedule lowering, so the counts hold on every backend. === *)
let () =
  let limits = { Ir.Backend_intf.no_hardware_limits with simd_vector_bytes = 32 } in
  let seeds_of ~name comp =
    let captured = ref None in
    let ctx = Context.auto () in
    let (_ : Context.t * Context.routine) =
      Context.compile
        ~lowered_transform:(fun opt ->
          captured := Some (Autotune.sketch_seed_params ~is_gpu:false ~is_cpu:true ~limits opt);
          [ opt ])
        ctx (named name comp) Ir.Indexing.Empty
    in
    Option.value_exn !captured
  in
  let summarize label seeds =
    let count f = List.count seeds ~f in
    Stdio.printf "%s: total=%d whole=%d packed=%d hoist=%d grid=%d packrest=%d\n" label
      (List.length seeds)
      (count (fun p -> p.Autotune.sk_mma && p.Autotune.sk_bk = 0))
      (count (fun p -> p.Autotune.sk_mma && p.Autotune.sk_bk > 0))
      (count (fun p -> p.Autotune.sk_hoist))
      (count (fun p -> p.Autotune.sk_grid))
      (count (fun p -> p.Autotune.sk_pack_rest))
  in
  let mav = Array.init (n * n) ~f:(fun x -> Float.of_int (x % 13) *. 0.25) in
  let mbv = Array.init (n * n) ~f:(fun x -> Float.of_int (x % 17) -. 8.) in
  (* Both operands hoistable, standard layout: every family seeds, no mixed shape (nothing left to
     pack in-kernel under the hoisted grid-outermost flavor). *)
  let ma = TDSL.ndarray mav ~label:[ "tmd_s_a" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "tmd_s_b" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op sc = ma * mb in
  summarize "seeds: standard, both hoistable" (seeds_of ~name:"tmd_seeds_std" (Train.forward sc));
  (* Transposed B, both hoistable: the whole-triple family (which reads B in place) is filtered out;
     the packed flavors normalize the layout and stay, including the hoisted Grid flavor (hoisted B~
     packs at link time, so nothing reads B in place). *)
  let mta = TDSL.ndarray mav ~label:[ "tmd_s_ta" ] ~output_dims:[ n; n ] () in
  let mtb = TDSL.ndarray mbv ~label:[ "tmd_s_tb" ] ~output_dims:[ n; n ] () in
  let%op st = mta +* "ik;jk=>ij" mtb in
  summarize "seeds: transposed B, hoistable" (seeds_of ~name:"tmd_seeds_tb" (Train.forward st));
  (* Exactly one hoistable operand (computed A x constant B), standard layout: the mixed
     grid-outermost shape ([sk_pack_rest]) joins the pool (gh-ocannl-473). *)
  let ma2 = TDSL.ndarray mav ~label:[ "tmd_s_a2" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb2 = TDSL.ndarray mbv ~label:[ "tmd_s_b2" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op xa = relu ma2 in
  let%op yc = xa * mb2 in
  summarize "seeds: computed A x hoistable B" (seeds_of ~name:"tmd_seeds_mixed" (Train.forward yc));
  (* Non-hoistable transposed B: the hoisted-only Grid flavor would read B in place and statically
     decline, so it is filtered; the mixed shape packs B in-kernel (normalizing it) and stays. *)
  let ma3 = TDSL.ndarray mav ~label:[ "tmd_s_a3" ] ~output_dims:[ n; n ] () in
  let mtb3 = TDSL.ndarray mbv ~label:[ "tmd_s_tb3" ] ~output_dims:[ n; n ] () in
  let%op xb = relu mtb3 in
  let%op yt = ma3 +* "ik;jk=>ij" xb in
  summarize "seeds: hoistable A x computed transposed B"
    (seeds_of ~name:"tmd_seeds_tb_mixed" (Train.forward yt));
  (* No hoistable operand (both trainable params): the grid-outermost per-chunk B~ re-packing shape
     ([sk_pack_rest] without [sk_hoist], gh-ocannl-475) is the only one-dispatch flavor, and is
     seeded whenever the packed tiles statically fit the per-chunk privatization cap. *)
  let wa = TDSL.param ~values:mav "tmd_wa" ~input_dims:[ n ] ~output_dims:[ n ] () in
  let wb = TDSL.param ~values:mbv "tmd_wb" ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op yw = wa * wb in
  summarize "seeds: no hoistable operand" (seeds_of ~name:"tmd_seeds_params" (Train.forward yw));
  (* Column extent below one vector of lanes (nj = 4 < 8): the whole-triple form statically declines
     and is not seeded. The packed compositions with a split column panel ([bn > 0]) now survive via
     pad-composition seeding (gh-ocannl-485): the padded micro-kernel's column extent is the panel
     width, so the register tiling genuinely fires — the (large) padding waste is the tuner's call.
     Unsplit-panel flavors ([bn = 0], panel width = nj) stay filtered. *)
  let mbn = Array.init (n * 4) ~f:(fun x -> Float.of_int (x % 17) -. 8.) in
  let ma4 = TDSL.ndarray mav ~label:[ "tmd_s_a4" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb4 = TDSL.ndarray mbn ~label:[ "tmd_s_b4" ] ~input_dims:[ 4 ] ~output_dims:[ n ] () in
  let%op yn = ma4 * mb4 in
  summarize "seeds: narrow columns (nj < lanes)"
    (seeds_of ~name:"tmd_seeds_narrow" (Train.forward yn))

(* === Accumulator-aware GPU seeding (gh-ocannl-545). A backend's tensor-core support is a property
   of the whole (a, b, accumulator) triple, not of the operand pair: CUDA's [nvcuda::wmma] declares
   bf16 multiplicands against a [float] accumulator only. Keyed on the pair alone, a uniformly-bf16
   network seeded tensorized candidates that every emission then rendered as the lane-0 scalar
   fallback — the tuner spent its measurement budget timing, ranking and crowning scalar schedules
   under `mma-gpu` labels, which is indistinguishable in a report from genuine tensor-core adoption.
   The capability key carries the accumulator format, so seeding follows emission.

   The synthetic table below deliberately advertises bf16 against f32 ONLY (and f16 against both),
   so it pins the gating mechanism rather than any one backend's current arm set. Seeds are
   enumerated on the pre-schedule lowering, so the counts hold on every backend. === *)
let () =
  let mma_limits =
    {
      Ir.Backend_intf.no_hardware_limits with
      mma =
        Some
          {
            Ir.Backend_intf.mma_simd_width = 32;
            mma_tile = (16, 16, 16);
            mma_format_tiles =
              [
                ( (Ir.Backend_intf.Mma_f16, Ir.Backend_intf.Mma_f16, Ir.Backend_intf.Mma_f32),
                  (16, 16, 16) );
                ( (Ir.Backend_intf.Mma_f16, Ir.Backend_intf.Mma_f16, Ir.Backend_intf.Mma_f16),
                  (16, 16, 16) );
                ( (Ir.Backend_intf.Mma_bf16, Ir.Backend_intf.Mma_bf16, Ir.Backend_intf.Mma_f32),
                  (16, 16, 16) );
              ];
            mma_f16_wide_acc_scopes = [];
            mma_staged_layouts = [];
            mma_pipeline_depths = [];
          };
    }
  in
  let mma_seeds_of ~name comp =
    let captured = ref None in
    let ctx = Context.auto () in
    let (_ : Context.t * Context.routine) =
      Context.compile
        ~lowered_transform:(fun opt ->
          captured :=
            Some (Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits:mma_limits opt);
          [ opt ])
        ctx (named name comp) Ir.Indexing.Empty
    in
    Option.value_exn !captured
  in
  let count_mma label ~name ~operand_prec ~acc_prec =
    let a =
      NTDSL.init ~l:(name ^ "_a") ~prec:operand_prec ~i:[ n ] ~o:[ n ]
        ~f:(fun idcs -> Float.of_int (((idcs.(0) * n) + idcs.(1)) % 3) *. 0.25)
        ()
    in
    let b =
      NTDSL.init ~l:(name ^ "_b") ~prec:operand_prec ~i:[ n ] ~o:[ n ]
        ~f:(fun idcs -> (Float.of_int (((idcs.(0) * n) + idcs.(1)) % 5) -. 2.) *. 0.5)
        ()
    in
    let%op c = a * b in
    Ir.Tnode.update_prec c.Tensor.value acc_prec;
    let seeds = mma_seeds_of ~name (Train.forward c) in
    Stdio.printf "%s: mma seeds=%d\n" label (List.count seeds ~f:(fun p -> p.Autotune.sk_mma))
  in
  count_mma "seeds: f16 operands, f16 accumulator (advertised)" ~name:"tmd_acc_f16_f16"
    ~operand_prec:Ir.Ops.half ~acc_prec:Ir.Ops.half;
  count_mma "seeds: bf16 operands, f32 accumulator (advertised)" ~name:"tmd_acc_bf16_f32"
    ~operand_prec:Ir.Ops.bfloat16 ~acc_prec:Ir.Ops.single;
  count_mma "seeds: bf16 operands, bf16 accumulator (not advertised)" ~name:"tmd_acc_bf16_bf16"
    ~operand_prec:Ir.Ops.bfloat16 ~acc_prec:Ir.Ops.bfloat16

(* === Swizzled staged twins are capability-gated (gh-ocannl-481 item 3, D3) ===

   A swizzled staged tile is only worth proposing where the emission can actually read it: a twin
   the arms decline renders the scalar fallback, and the tuner would time and rank it under an
   `mma-gpu` label — the same failure the accumulator key above was introduced to stop.
   [mma_staged_layouts] is therefore keyed by format triple, and the twins follow it.

   Two more properties this pins, both mechanism rather than any backend's arm set:

   - Only STAGED seeds ([sk_bk > 0]) are twinned. An unstaged seed reads its operands in place;
   there is no shared tile to lay out. - The twin must satisfy [Swizzle_b128]'s extent rule (a
   power-of-two count > 1 of whole 16-byte units per staged tile row) at SEEDING time. The f32 table
   below advertises the layout for a format whose 32-element tile rows are 128 bytes — 8 units —
   while the bf16 table's are 64 bytes; a format whose rows missed the rule would simply not be
   twinned, rather than raise inside [Schedule.apply]. === *)
let () =
  let staged_limits advertised =
    {
      Ir.Backend_intf.no_hardware_limits with
      mma =
        Some
          {
            Ir.Backend_intf.mma_simd_width = 32;
            mma_tile = (16, 16, 16);
            mma_format_tiles =
              [
                ( (Ir.Backend_intf.Mma_f32, Ir.Backend_intf.Mma_f32, Ir.Backend_intf.Mma_f32),
                  (16, 16, 16) );
              ];
            mma_f16_wide_acc_scopes = [];
            mma_staged_layouts =
              (if advertised then
                 [
                   ( (Ir.Backend_intf.Mma_f32, Ir.Backend_intf.Mma_f32, Ir.Backend_intf.Mma_f32),
                     Ir.Backend_intf.Mma_swizzled_b128 );
                 ]
               else []);
            mma_pipeline_depths = [];
          };
    }
  in
  let av = Array.init (n * n) ~f:(fun x -> Float.of_int (x % 13) *. 0.25) in
  let bv = Array.init (n * n) ~f:(fun x -> Float.of_int (x % 17) -. 8.) in
  let a = TDSL.ndarray av ~label:[ "tmd_swz_a" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let b = TDSL.ndarray bv ~label:[ "tmd_swz_b" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op c = a * b in
  let captured = ref None in
  let ctx = Context.auto () in
  let (_ : Context.t * Context.routine) =
    Context.compile
      ~lowered_transform:(fun opt ->
        captured := Some opt;
        [ opt ])
      ctx
      (named "tmd_swizzled_seeds" (Train.forward c))
      Ir.Indexing.Empty
  in
  let opt = Option.value_exn !captured in
  let report label advertised =
    let seeds =
      Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits:(staged_limits advertised) opt
      |> List.filter ~f:(fun p -> p.Autotune.sk_mma)
    in
    let swizzled = List.filter seeds ~f:(fun p -> Option.is_some p.Autotune.sk_swizzle) in
    Stdio.printf "%s: mma seeds=%d swizzled=%d unstaged swizzled=%d\n" label (List.length seeds)
      (List.length swizzled)
      (List.count swizzled ~f:(fun p -> p.Autotune.sk_bk = 0))
  in
  report "seeds: staged layout advertised" true;
  report "seeds: staged layout not advertised" false
