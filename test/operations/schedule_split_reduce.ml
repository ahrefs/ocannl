(* Deterministic two-pass split reductions, gh-ocannl-484 (tasks 1, 2, 4): parallel reductions
   without atomics. [Sched.Split_reduce] splits a Serial reduction loop into [num_blocks] chunks
   accumulating into per-block partials (a fresh [num_blocks x target-dims] scratch node), plus a
   synthesized combine statement folding the partials into the target in a fixed balanced-tree
   order. The partials producer/consumer pair is a materialized cross-nest edge, so
   [Sched.fission_scheduled] compiles the passes as separate kernels and the default presets
   annotate them.

   The parity discipline (task 4): the combine tree is a function of the schedule, not of thread
   timing — so for a FIXED schedule, the parallel (fissioned + annotated) execution must match the
   all-serial execution of the same schedule BITWISE, on every backend. Against the unsplit serial
   twin the values only agree approximately: the split reassociates the reduction (same license as
   [Swap] of accumulations) — schedule identity pins numerics.

   The gh-466 scatter unbail (task 2): the embedding-backward [Set_dynamic] previously made the
   whole kernel bail out of parallelization. Now (a) plain-Iterator components of the scatter's
   index vector (e.g. the embedding-dim loop) may parallelize directly — the dynamic row is masked
   from the affine queries, per-component pinning decides — and (b) [Split_reduce] on the position
   loop gives per-block partial gradient rows, parallelizing the position dimension too. Both run
   bitwise-deterministically. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Cache = Ir.Schedule_cache
module Idx = Ir.Indexing
module Asgns = Ir.Assignments
module IDX = Train.IDX

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

let approx a b = Float.(abs (a - b) < 1e-3 *. (1. +. abs b))
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_cpu = Sched.backend_is_cpu backend_name

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* The first [For_loop] (in preorder) with the given iteration count. *)
let find_loop_with_extent ~n (llc : LL.t) : Idx.symbol option =
  let found = ref None in
  let rec go (llc : LL.t) =
    if Option.is_none !found then
      match llc with
      | LL.For_loop { index; from_; to_; body; _ } ->
          if to_ - from_ + 1 = n then found := Some index else go body
      | LL.Seq (a, b) ->
          go a;
          go b
      | LL.If { body; _ } -> go body
      | _ -> ()
  in
  go llc;
  !found

(* Statement-level census: If guards, For_loops, and accesses of nodes labeled with [label]. *)
let census ?label (llc : LL.t) : int * int * int =
  let ifs = ref 0 and loops = ref 0 and labeled = ref 0 in
  let tagged tn =
    match label with None -> false | Some l -> List.mem tn.Tn.label l ~equal:String.equal
  in
  let rec go (llc : LL.t) =
    match llc with
    | LL.Noop | LL.Comment _ | LL.Staged_compilation _ | LL.Declare_local _ | LL.Workgroup_barrier
      ->
        ()
    | LL.Zero_out tn -> if tagged tn then Int.incr labeled
    | LL.Seq (a, b) ->
        go a;
        go b
    | LL.For_loop { body; _ } ->
        Int.incr loops;
        go body
    | LL.Scan_loop { carried; body; _ } ->
        List.iter carried ~f:(fun c -> scan c.LL.init);
        go body
    | LL.If { cond = c, _; body } ->
        Int.incr ifs;
        scan c;
        go body
    | LL.Tile_mma { fallback; _ } -> go fallback
    | LL.Set { tn; llsc; _ } ->
        if tagged tn then Int.incr labeled;
        scan llsc
    | LL.Set_local (_, llsc) -> scan llsc
    | LL.Set_dynamic { tn; dyn_value = v, _; llsc; _ } ->
        if tagged tn then Int.incr labeled;
        scan v;
        scan llsc
    | LL.Set_from_vec { tn; arg = a, _; _ } ->
        if tagged tn then Int.incr labeled;
        scan a
  and scan (sc : LL.scalar_t) =
    match sc with
    | LL.Ternop (_, (a, _), (b, _), (c, _)) ->
        scan a;
        scan b;
        scan c
    | LL.Binop (_, (a, _), (b, _)) ->
        scan a;
        scan b
    | LL.Unop (_, (a, _)) -> scan a
    | LL.Local_scope { body; _ } -> go body
    | LL.Get (tn, _) -> if tagged tn then Int.incr labeled
    | LL.Get_dynamic { tn; dyn_value = v, _; _ } ->
        if tagged tn then Int.incr labeled;
        scan v
    | LL.Get_local _ | LL.Get_merge_buffer _ | LL.Constant _ | LL.Constant_bits _ | LL.Embed_index _
      ->
        ()
  in
  go llc;
  (!ifs, !loops, !labeled)

let preset opt =
  if on_cpu then Sched.default_cpu ~min_parallel:1 opt else Sched.default_gpu ~min_parallel:1 opt

(* Fission the (already Split_reduce-transformed) routine and annotate each segment with the default
   preset — the plural-transform flow of gh-484: the partials edge cuts, each pass gets its own
   launch geometry, and the runtime event chain provides the grid-wide synchronization the combine
   needs. *)
let fission_annotate opt =
  Sched.fission_scheduled ~preset ~zero_sched:(fun _ -> []) ~static_indices:[] opt
  |> List.map ~f:(fun (_, _, _, scheduled) -> scheduled)

(* Non-empty by construction (gh-ocannl-746): [Array.for_all2_exn] answers [true] on two empty
   arrays, and a readback and the reference it is compared against go empty TOGETHER -- the
   reference went through the same path. {!Verdict.p_all2} is the claim-shaped form; the sites
   reached through this helper keep a boolean, so the guard lives here. *)
let bitwise a b = (not (Array.is_empty a)) && Array.for_all2_exn a b ~f:Float.equal

(* === Leg 1: static reduction — matvec y[i] = Σ_k m[i,k] * v[k], awkward extents. === *)

let m_ext = 13
let k_ext = 70

(* Mixed magnitudes so reassociation is numerically observable: the split-schedule values may differ
   from the unsplit serial twin in low bits, while staying bitwise-stable per schedule. *)
let mav =
  Array.init (m_ext * k_ext) ~f:(fun x ->
      Float.sin (Float.of_int x) *. (10. **. Float.of_int (x % 5)))

let mvv =
  Array.init k_ext ~f:(fun x -> Float.cos (Float.of_int x) *. (10. **. Float.of_int (x % 3)))

let make_matvec label =
  let m =
    TDSL.ndarray mav ~label:[ "sr_m" ^ label ] ~input_dims:[ k_ext ] ~output_dims:[ m_ext ] ()
  in
  let v = TDSL.ndarray mvv ~label:[ "sr_v" ^ label ] ~output_dims:[ k_ext ] () in
  let%op y = m * v in
  (y, m)

let run_forward ?(transform = fun (opt : LL.optimized) -> opt) ?transforms ~name (t : Tensor.t) =
  let ctx = Context.auto () in
  let ctx, routine =
    match transforms with
    | None ->
        Context.compile
          ~lowered_transform:(fun o -> [ transform o ])
          ctx
          (named name (Train.forward t))
          Idx.Empty
    | Some transforms ->
        Context.compile ~lowered_transform:transforms ctx (named name (Train.forward t)) Idx.Empty
  in
  let ctx = Context.run ctx routine in
  Context.get_values ctx t.Tensor.value

let () =
  Tensor.unsafe_reinitialize ();
  let y0, _ = make_matvec "0" in
  let want = nonzero "sr_serial" (run_forward ~name:"sr_serial" y0) in

  (* -- Dividing split (B = 7, chunk = 10): guard folds away; structural pins. -- *)
  let stats = ref (-1, -1, -1) in
  let n_nests = ref (-1) in
  let legal = ref None and illegal_target = ref None and cache_ok = ref false in
  let y1, m1 = make_matvec "1" in
  let split_transform ?(num_blocks = 7) ~target ?(observe = fun (_ : LL.optimized) -> ()) () =
   fun (opt : LL.optimized) ->
    let axis = Option.value_exn ~here:[%here] (find_loop_with_extent ~n:k_ext opt.LL.llc) in
    let op, _b, _i, _c = Sched.split_reduce ~axis ~target ~num_blocks in
    let opt' = Sched.apply [ op ] opt in
    observe opt';
    opt'
  in
  let observe (opt' : LL.optimized) =
    stats := census ~label:"partials" opt'.LL.llc;
    n_nests :=
      List.count (LL.flat_lines [ opt'.LL.llc ]) ~f:(function LL.For_loop _ -> true | _ -> false)
  in
  let twin_transform (opt : LL.optimized) =
    (* Legality oracle and cache round-trip, probed against the pre-transform code. *)
    let axis = Option.value_exn ~here:[%here] (find_loop_with_extent ~n:k_ext opt.LL.llc) in
    let op, _, _, _ = Sched.split_reduce ~axis ~target:y1.Tensor.value ~num_blocks:7 in
    legal := Some (Sched.op_legality opt op);
    let bad, _, _, _ = Sched.split_reduce ~axis ~target:m1.Tensor.value ~num_blocks:7 in
    illegal_target := Some (Sched.op_legality opt bad);
    (let canonical = Cache.canonicalize opt in
     let saved1, _ = Cache.to_saved (Cache.base_registry canonical) [ op ] in
     let sched2, _ = Cache.of_saved canonical saved1 in
     let saved2, _ = Cache.to_saved (Cache.base_registry canonical) sched2 in
     (* Applicability of the re-minted schedule, probed hermetically (a raw [Sched.apply] here would
        pollute the compile's traced store with a stray partials node). *)
     let verdicts = Sched.schedule_legality opt sched2 in
     let reapplicable = match verdicts with [ (_, Sched.Op_legal) ] -> true | _ -> false in
     cache_ok := Cache.equal_saved_schedule saved1 saved2 && reapplicable);
    split_transform ~target:y1.Tensor.value ~observe () opt
  in
  let twin = run_forward ~transform:twin_transform ~name:"sr_twin7" y1 in
  let ifs, _, partials_accs = !stats in
  p "dividing split-reduce is guard-free" (ifs = 0);
  (* Partials accesses: pass-1 init + rmw write + rmw read, combine reads = num_blocks. *)
  p "pass 1 accumulates into partials and the combine folds all blocks" (partials_accs = 3 + 7);
  p "the routine gains the combine as a separate top-level nest" (!n_nests >= 2);
  p "split-reduce on an accumulation is Op_legal"
    (match !legal with Some Sched.Op_legal -> true | _ -> false);
  p "split-reduce on a read-only operand is Op_illegal"
    (match !illegal_target with Some (Sched.Op_illegal _) -> true | _ -> false);
  p "schedule-cache round-trip re-mints an applicable Split_reduce" !cache_ok;
  p_all2 "split-schedule serial twin approximates the unsplit reference" twin want ~f:approx;

  (* -- The same schedule, fissioned and annotated: must match the serial twin BITWISE. -- *)
  let seg_pins = ref (false, false) in
  let y2, _ = make_matvec "2" in
  let par_transforms (opt : LL.optimized) =
    let axis = Option.value_exn ~here:[%here] (find_loop_with_extent ~n:k_ext opt.LL.llc) in
    let op, _, _, _ = Sched.split_reduce ~axis ~target:y2.Tensor.value ~num_blocks:7 in
    let opt' = Sched.apply [ op ] opt in
    let segs = Sched.fission_scheduled ~preset ~zero_sched:(fun _ -> []) ~static_indices:[] opt' in
    let pass1_annotated =
      List.exists segs ~f:(fun (_, pre, sched, _) ->
          let _, _, l = census ~label:"partials" pre.LL.llc in
          l > 0 && l < 7 && not (List.is_empty sched))
    in
    let combine_separate =
      List.exists segs ~f:(fun (_, pre, _, _) ->
          let _, _, l = census ~label:"partials" pre.LL.llc in
          (* The combine reads all blocks and nothing else of partials. *)
          l = 7)
    in
    seg_pins := (pass1_annotated, combine_separate);
    List.map segs ~f:(fun (_, _, _, scheduled) -> scheduled)
  in
  let par = run_forward ~transforms:par_transforms ~name:"sr_par7" y2 in
  let pass1_annotated, combine_separate = !seg_pins in
  p "fission annotates the pass-1 segment" pass1_annotated;
  p "fission gives the combine its own segment" combine_separate;
  p "parallel split-reduce matches its serial twin bitwise (fixed schedule pins numerics)"
    (bitwise par twin);

  (* -- Non-dividing split (B = 6, chunk = 12): the remainder guard survives, parity holds. -- *)
  let y3, _ = make_matvec "3" in
  let twin6 =
    run_forward
      ~transform:(split_transform ~num_blocks:6 ~target:y3.Tensor.value ~observe ())
      ~name:"sr_twin6" y3
  in
  let ifs6, _, _ = !stats in
  p "non-dividing split-reduce keeps the remainder guard" (ifs6 > 0);
  p_all2 "non-dividing serial twin approximates the reference" twin6 want ~f:approx;
  let y4, _ = make_matvec "4" in
  let par6 =
    run_forward
      ~transforms:(fun opt ->
        let axis = Option.value_exn ~here:[%here] (find_loop_with_extent ~n:k_ext opt.LL.llc) in
        let op, _, _, _ = Sched.split_reduce ~axis ~target:y4.Tensor.value ~num_blocks:6 in
        fission_annotate (Sched.apply [ op ] opt))
      ~name:"sr_par6" y4
  in
  p "non-dividing parallel split-reduce matches its serial twin bitwise" (bitwise par6 twin6)

(* === Leg 2: max-reduce (identity = -inf) — y = max_i x[i]. === *)

let () =
  let n = 50 in
  let xv = Array.init n ~f:(fun i -> Float.sin (Float.of_int (7 * i)) *. 100.) in
  let make () =
    let x = TDSL.ndarray xv ~label:[ "sr_x" ] ~output_dims:[ n ] () in
    let%op ymax = x @^^ "i => 0" in
    ymax
  in
  let y0 = make () in
  let want = nonzero "sr_max_serial" (run_forward ~name:"sr_max_serial" y0) in
  let y1 = make () in
  let twin =
    run_forward
      ~transform:(fun opt ->
        let axis = Option.value_exn ~here:[%here] (find_loop_with_extent ~n opt.LL.llc) in
        let op, _, _, _ = Sched.split_reduce ~axis ~target:y1.Tensor.value ~num_blocks:5 in
        Sched.apply [ op ] opt)
      ~name:"sr_max_twin" y1
  in
  (* Max is insensitive to association: the split twin matches the reference bitwise. *)
  p "split max-reduce matches the reference bitwise" (bitwise twin want);
  let y2 = make () in
  let par =
    run_forward
      ~transforms:(fun opt ->
        let axis = Option.value_exn ~here:[%here] (find_loop_with_extent ~n opt.LL.llc) in
        let op, _, _, _ = Sched.split_reduce ~axis ~target:y2.Tensor.value ~num_blocks:5 in
        fission_annotate (Sched.apply [ op ] opt))
      ~name:"sr_max_par" y2
  in
  p "parallel split max-reduce matches the serial twin bitwise" (bitwise par twin)

(* === Leg 3: the gh-466 embedding-backward scatter, unbailed (task 2). === *)

let vocab = 5
let embed = 4
let positions = 10
let grad_of t = (Option.value_exn t.Tensor.diff).Tensor.grad

(* Integer-valued coefficients: every accumulation order is exact, so the split/parallel gradients
   must equal the analytic dense gradient BITWISE — the strongest determinism pin. *)
let coeff_values =
  Array.init (positions * embed)
    ~f:(Ll_test.cycle_flat ~dims:[| positions; embed |] ~modulus:7 ~offset:1. ~stride:1.)

let id_values = [| 1.; 3.; 0.; 1.; 4.; 2.; 0.; 3.; 1.; 2. |]

let expected_grads () =
  let g = Array.create ~len:(embed * vocab) 0. in
  Array.iteri id_values ~f:(fun pos idf ->
      let v = Int.of_float idf in
      for o = 0 to embed - 1 do
        g.((o * vocab) + v) <- g.((o * vocab) + v) +. coeff_values.((pos * embed) + o)
      done);
  g

(* Build the embedding + loss; return the update comp and the table-gradient node. *)
let make_embedding label =
  let classes = TDSL.range vocab in
  let ids =
    TDSL.ndarray id_values ~label:[ "sr_ids" ^ label ] ~batch_dims:[ positions ] ~output_dims:[] ()
  in
  let%op one_hot = classes = ids in
  let c =
    TDSL.param
      ~values:(Array.init (embed * vocab) ~f:Float.of_int)
      ("sr_c" ^ label) ~input_dims:[ vocab ] ~output_dims:[ embed ] ()
  in
  let%op embedded = c * one_hot in
  let coeff =
    TDSL.ndarray coeff_values
      ~label:[ "sr_coeff" ^ label ]
      ~batch_dims:[ positions ] ~output_dims:[ embed ] ()
  in
  let%op loss = (embedded *. coeff) ++ "...|... => |->0" in
  (Train.grad_update loss, loss, grad_of c)

let run_update ?(transform = fun (opt : LL.optimized) -> opt) ?transforms ~name (update, loss, grad)
    =
  let ctx = Context.auto () in
  let ctx = Train.init_params ctx IDX.empty loss in
  let ctx, routine =
    match transforms with
    | None ->
        Context.compile
          ~lowered_transform:(fun o -> [ transform o ])
          ctx (named name update) Idx.Empty
    | Some transforms ->
        Context.compile ~lowered_transform:transforms ctx (named name update) Idx.Empty
  in
  let ctx = Context.run ctx routine in
  Context.get_values ctx grad

(* The statement containing the scatter into [grad], and the position loop inside it. *)
let scatter_stmt ~grad (llc : LL.t) : LL.t =
  let rec has_scatter (llc : LL.t) =
    match llc with
    | LL.Set_dynamic { tn; _ } -> Tn.equal tn grad
    | LL.Seq (a, b) -> has_scatter a || has_scatter b
    | LL.For_loop { body; _ } | LL.If { body; _ } -> has_scatter body
    | _ -> false
  in
  List.find_exn (LL.flat_lines [ llc ]) ~f:has_scatter

let () =
  let expected = expected_grads () in
  let g0 = make_embedding "0" in
  let want = nonzero "sr_emb_serial" (run_update ~name:"sr_emb_serial" g0) in
  p "serial scatter gradient equals the analytic dense gradient bitwise" (bitwise want expected);

  (* -- Unbail (a): with no split at all, the default preset now annotates the scatter segment — the
     embedding-dim loop parallelizes (its component pins the written row's column), while the
     position loop driving the dynamic index stays serial. Bitwise-equal to the serial run: per
     cell, the accumulation order is unchanged. -- *)
  let scatter_annotated = ref false in
  let g1 = make_embedding "1" in
  let _, _, grad1 = g1 in
  let unbailed =
    run_update
      ~transforms:(fun opt ->
        let segs =
          Sched.fission_scheduled ~preset ~zero_sched:(fun _ -> []) ~static_indices:[] opt
        in
        List.iter segs ~f:(fun (_, pre, sched, _) ->
            let rec has_dyn (llc : LL.t) =
              match llc with
              | LL.Set_dynamic { tn; _ } -> Tn.equal tn grad1
              | LL.Seq (a, b) -> has_dyn a || has_dyn b
              | LL.For_loop { body; _ } | LL.If { body; _ } -> has_dyn body
              | _ -> false
            in
            if has_dyn pre.LL.llc && not (List.is_empty sched) then scatter_annotated := true);
        List.map segs ~f:(fun (_, _, _, scheduled) -> scheduled))
      ~name:"sr_emb_unbailed" g1
  in
  p "default preset annotates the un-split scatter segment (gh-466 unbail)" !scatter_annotated;
  p "annotated un-split scatter matches the serial gradient bitwise" (bitwise unbailed expected);

  (* -- Split_reduce on the position loop: per-block partial gradient rows + tree combine. -- *)
  let struct_pins = ref (-1, -1) in
  let g2 = make_embedding "2" in
  let _, _, grad2 = g2 in
  let split_pos ~observe (opt : LL.optimized) =
    let stmt = scatter_stmt ~grad:grad2 opt.LL.llc in
    let axis = Option.value_exn ~here:[%here] (find_loop_with_extent ~n:positions stmt) in
    let op, _, _, _ = Sched.split_reduce ~axis ~target:grad2 ~num_blocks:4 in
    let opt' = Sched.apply [ op ] opt in
    if observe then begin
      let _, _, partials_zeros = census ~label:"partials" opt'.LL.llc in
      let dyn_to_partials =
        let count = ref 0 in
        let rec go (llc : LL.t) =
          match llc with
          | LL.Set_dynamic { tn; _ } when List.mem tn.Tn.label "partials" ~equal:String.equal ->
              Int.incr count
          | LL.Seq (a, b) ->
              go a;
              go b
          | LL.For_loop { body; _ } | LL.If { body; _ } -> go body
          | _ -> ()
        in
        go opt'.LL.llc;
        !count
      in
      struct_pins := (partials_zeros, dyn_to_partials)
    end;
    opt'
  in
  let twin = run_update ~transform:(split_pos ~observe:true) ~name:"sr_emb_twin" g2 in
  let partials_accs, dyn_to_partials = !struct_pins in
  p "the scatter is redirected into the per-block partials" (dyn_to_partials = 1);
  (* Zero_out + Set_dynamic write + Get_dynamic read + 4 combine reads. *)
  p "partials are zeroed, scattered into, and tree-combined" (partials_accs = 3 + 4);
  p "split-scatter serial twin equals the analytic gradient bitwise" (bitwise twin expected);
  let g3 = make_embedding "3" in
  let _, _, grad3 = g3 in
  let pass1_annotated = ref false in
  let par =
    run_update
      ~transforms:(fun opt ->
        let stmt = scatter_stmt ~grad:grad3 opt.LL.llc in
        let axis = Option.value_exn ~here:[%here] (find_loop_with_extent ~n:positions stmt) in
        let op, _, _, _ = Sched.split_reduce ~axis ~target:grad3 ~num_blocks:4 in
        let opt' = Sched.apply [ op ] opt in
        let segs =
          Sched.fission_scheduled ~preset ~zero_sched:(fun _ -> []) ~static_indices:[] opt'
        in
        List.iter segs ~f:(fun (_, pre, sched, _) ->
            let rec dyn_partials (llc : LL.t) =
              match llc with
              | LL.Set_dynamic { tn; _ } -> List.mem tn.Tn.label "partials" ~equal:String.equal
              | LL.Seq (a, b) -> dyn_partials a || dyn_partials b
              | LL.For_loop { body; _ } | LL.If { body; _ } -> dyn_partials body
              | _ -> false
            in
            if dyn_partials pre.LL.llc && not (List.is_empty sched) then pass1_annotated := true);
        List.map segs ~f:(fun (_, _, _, scheduled) -> scheduled))
      ~name:"sr_emb_par" g3
  in
  p "default preset annotates the split-scatter pass-1 segment" !pass1_annotated;
  p "parallel split-scatter equals the analytic gradient bitwise" (bitwise par expected)

(* === Leg 4: pattern discipline — targeted errors. === *)

let () =
  let expect_error name transform t =
    match
      try
        ignore
          (Context.compile
             ~lowered_transform:(fun o -> [ transform o ])
             (Context.auto ())
             (named name (Train.forward t))
             Idx.Empty
            : Context.t * Context.routine);
        None
      with Invalid_argument msg -> Some msg
    with
    | Some msg ->
        p
          (name ^ " rejected with a targeted error")
          (String.is_substring msg ~substring:"Schedule.Split_reduce")
    | None -> p (name ^ " rejected with a targeted error") false
  in
  (* [Train.forward] consumes the root, so each case gets a fresh graph. *)
  let y1, _ = make_matvec "err1" in
  expect_error "sr_err_no_such_loop"
    (fun (opt : LL.optimized) ->
      let op, _, _, _ =
        Sched.split_reduce ~axis:(Idx.get_symbol ()) ~target:y1.Tensor.value ~num_blocks:4
      in
      Sched.apply [ op ] opt)
    y1;
  let y2, m2 = make_matvec "err2" in
  expect_error "sr_err_not_accumulated"
    (fun (opt : LL.optimized) ->
      let axis = Option.value_exn ~here:[%here] (find_loop_with_extent ~n:k_ext opt.LL.llc) in
      let op, _, _, _ = Sched.split_reduce ~axis ~target:m2.Tensor.value ~num_blocks:4 in
      Sched.apply [ op ] opt)
    y2;
  let y3, _ = make_matvec "err3" in
  expect_error "sr_err_num_blocks"
    (fun (opt : LL.optimized) ->
      let axis = Option.value_exn ~here:[%here] (find_loop_with_extent ~n:k_ext opt.LL.llc) in
      let op, _, _, _ = Sched.split_reduce ~axis ~target:y3.Tensor.value ~num_blocks:1 in
      Sched.apply [ op ] opt)
    y3;
  let y4, _ = make_matvec "err4" in
  expect_error "sr_err_output_axis"
    (fun (opt : LL.optimized) ->
      (* Splitting the output loop: the accumulation cell mentions the split axis. *)
      let axis = Option.value_exn ~here:[%here] (find_loop_with_extent ~n:m_ext opt.LL.llc) in
      let op, _, _, _ = Sched.split_reduce ~axis ~target:y4.Tensor.value ~num_blocks:4 in
      Sched.apply [ op ] opt)
    y4;
  Stdio.printf "\nDone.\n%!"
