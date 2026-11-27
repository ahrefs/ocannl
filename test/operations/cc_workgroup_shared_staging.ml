(* gh-ocannl-678: serial barrier-fission rendering of workgroup-shared staging on backends without
   hardware workgroups/shared memory (cc), with parity against the serial oracle on every backend
   (including Metal) and deterministic negative controls for missing/misplaced barriers and
   cross-nest grid misalignment on cc.

   What is verified: - Valid shared-staging schedules execute bitwise-identically to the unstaged
   serial oracle on both cc and GPU backends: 1. 2-D workgroup SMEM matmul (multi-k_o cooperative
   staging + Privatize + two barriers). 2. Register-blocktiled batched matmul (4-D Grid with .z
   folding + 2-D Workgroup + Unroll). 3. Thread-private [Declare_local] scalar live across a
   [Workgroup_barrier] beside a shared tile. 4. [Scan_loop] (both [Forward] and [Backward])
   containing a [Workgroup_barrier] inside a [Workgroup] loop. 5. Two sibling [Workgroup] loops
   communicating through a shared tile across a barrier. - Deterministic negative controls on cc
   (off-GPU): 1. Removing the load->compute barrier (Barrier 1) inside k_o fails parity (and
   produces NaNs from per-workgroup 0xFF shared-tile poisoning). 2. Removing ONLY the trailing
   compute->next-load barrier (Barrier 2) inside k_o fails parity (proving no phantom barrier is
   inserted at the k_o loop back-edge). 3. Misplacing Barrier 1 before the cooperative loads (same
   barrier count, wrong position) fails parity. 4. Removing the barrier between two sibling
   [Workgroup] loops fails parity (proving sibling workgroup loops merge into one phase when no
   barrier separates them). 5. Swapping two [Grid] slots between a zeroing nest and an accumulation
   nest (gh-ocannl-643's cross-nest alignment trap) fails parity against a prefilled output. -
   Validation gates remain enforced on every backend (cc and GPU alike): 1. A [Workgroup_barrier]
   under an [If] guard is rejected by [validate_parallel]. 2. Mismatched [Workgroup] extents across
   a barrier are rejected by [validate_parallel]. *)

(* cc-fp-contract: exact-operands -- dyadic small operands whose products and sums stay exact in
   f32 *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
module L = Ll_test
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let on_gpu = Ir.Backend_intf.binds_hardware_grid (Context.backend_facts_of_name backend_name)
let cc_leg = Verdict.leg ~when_:(not on_gpu) ~on:backend_name ()
let accum_prec = Ll_test.backend_accum_prec

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let nonzero name (a : float array) =
  if not (Array.exists a ~f:(fun x -> Float.(x <> 0.))) then
    failwith (name ^ ": reference is all zeros");
  a

let make_node = L.node_factory ~first_id:678000 ~dims:[| 8 |] ()

let make_shared ~dims label =
  let tn = make_node ~dims label in
  Tn.update_memory_mode tn Tn.Local (Site "991:test-setup");
  tn

(* Compile and execute a hand-built [LL.optimized] with optional workgroup_shared tiles, seeding
   inputs and reading back [out]. *)
let exec_handbuilt ~name ~shared ~inputs ~out ?(prefill = true) (llc : LL.t) : float array =
  let traced_store = Hashtbl.create (module Tn) in
  let reg ~read_only tn =
    let node = LL.get_node traced_store tn in
    node.read_only <- read_only
  in
  List.iter inputs ~f:(fun (tn, _) -> reg ~read_only:true tn);
  List.iter shared ~f:(reg ~read_only:false);
  reg ~read_only:false out;
  let opt =
    {
      LL.traced_store;
      optimize_ctx = LL.empty_optimize_ctx ();
      llc;
      merge_node = None;
      workgroup_shared = Set.of_list (module Tn) shared;
      simdgroup_fragments = Set.empty (module Tn);
      masked_fragments = Set.empty (module Tn);
      swizzled = Map.empty (module Tn);
      pipelined = Map.empty (module Tn);
      zero_fringe = Set.empty (module Tn);
      flip_candidates = [];
      spliced_rbw = Set.empty (module Tn);
      source = llc;
    }
  in
  let ctx = Context.auto () in
  let ctx, routine =
    Context.compile ~name ~prelowered:opt
      ~lowered_transform:(fun x -> [ x ])
      ctx Asgns.empty_comp Ir.Indexing.Empty
  in
  let ctx = List.fold inputs ~init:ctx ~f:(fun ctx (tn, vals) -> Context.set_values ctx tn vals) in
  let ctx =
    if prefill then
      let pf = Array.init (Tn.num_elems out) ~f:(fun i -> Float.of_int (9000 + i)) in
      Context.set_values ctx out pf
    else ctx
  in
  let ctx = Context.run ctx routine in
  Context.get_values ctx out

(* --- Build a 16x16 x 16x16 matmul with BM=BN=BK=8 (2x2 grid of 4x4 workgroups, 2 k_o steps, exact
   f32 arithmetic) and its valid SMEM schedule --- *)
let build_smem_matmul () =
  let n = 16 and bm = 8 and bn = 8 and bk = 8 in
  let ma_r =
    Ll_test.operand ~b:[] ~o:[ n ] ~i:[ n ]
      (Ll_test.cycle ~dims:[| n; n |] ~modulus:7 ~offset:0. ~stride:0.25)
  in
  let mb_r =
    Ll_test.operand ~b:[] ~o:[ n ] ~i:[ n ]
      (Ll_test.cycle ~dims:[| n; n |] ~modulus:5 ~offset:(-2.) ~stride:0.5)
  in
  let products = Ll_test.matmul_products ~m:n ~n ~k:n ma_r mb_r in
  p_all "smem matmul operands have f32-exact partial sums" products
    ~f:(Ll_test.exact_partials ~any_sign:true ~sig_bits:(Ir.Ops.significand_bits Ir.Ops.single));
  let mav = Array.init (n * n) ~f:(fun i -> ma_r (Ll_test.unflat ~dims:[| n; n |] i)) in
  let mbv = Array.init (n * n) ~f:(fun i -> mb_r (Ll_test.unflat ~dims:[| n; n |] i)) in
  let ma = TDSL.ndarray mav ~label:[ "s678_a" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "s678_b" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op mc0 = ma * mb in
  let ctx, routine =
    Context.compile ~name:"s678_mm_serial"
      ~lowered_transform:(fun o -> [ o ])
      (Context.auto ()) (Train.forward mc0) Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let want = Context.get_values ctx mc0.Tensor.value in
  p_all2 "smem matmul serial oracle matches the premise's host sums bitwise" want
    (Array.of_list (List.map products ~f:(List.sum (module Float) ~f:Fn.id)))
    ~f:Float.equal;
  let want = nonzero "s678_mm_serial" want in
  let make_schedule (mc : Tensor.t) (opt : LL.optimized) : LL.optimized =
    let paths = L.nest_paths opt.LL.llc in
    let accum = List.find_exn paths ~f:(fun p -> List.length p = 3) in
    let i, j, k = match accum with [ i; j; k ] -> (i, j, k) | _ -> assert false in
    let ez, zsyms = Sched.expand_zero ~tn:mc.Tensor.value in
    let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
    let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_zj, _, _ = Sched.split ~axis:zj ~factor:bn ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_j, _, j_i = Sched.split ~axis:j ~factor:bn ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    Sched.apply
      [
        ez;
        sp_zi;
        sp_zj;
        sp_i;
        sp_j;
        sp_k;
        Sched.Stage
          {
            source = ma.Tensor.value;
            tile_loops = [ i_i; k_i ];
            shared = true;
            cooperative = None;
            hoisted = false;
            swizzle = None;
            pad_stride = None;
            pipeline_depth = 1;
            tile_prec = None;
          };
        Sched.Stage
          {
            source = mb.Tensor.value;
            tile_loops = [ k_i; j_i ];
            shared = true;
            cooperative = None;
            hoisted = false;
            swizzle = None;
            pad_stride = None;
            pipeline_depth = 1;
            tile_prec = None;
          };
        Sched.privatize ~accum_prec ~target:mc.Tensor.value ~over:k_o;
      ]
      opt
  in
  let run_variant ~name ~(mutate : LL.optimized -> LL.optimized) =
    let%op mc = ma * mb in
    let ctx, routine =
      Context.compile
        ~lowered_transform:(fun o -> [ mutate (make_schedule mc o) ])
        (Context.auto ())
        (named name (Train.forward mc))
        Ir.Indexing.Empty
    in
    let prefill = Array.init (n * n) ~f:(fun c -> Float.of_int (1000 + c)) in
    let ctx = Context.set_values ctx mc.Tensor.value prefill in
    let ctx = Context.run ctx routine in
    Context.get_values ctx mc.Tensor.value
  in
  (want, run_variant)

(* Mutate the barrier sequence inside the innermost [Serial] loop that contains two
   [Workgroup_barrier]s (the [k_o] loop of [Stage ~shared:true]). *)
let mutate_ko_barriers ~mode (opt : LL.optimized) : LL.optimized =
  let mutated = ref false in
  let rec go = function
    | LL.For_loop ({ axis = LL.Serial; body; _ } as fc) ->
        let stmts = LL.flat_lines [ body ] in
        let barrier_positions =
          List.filter_mapi stmts ~f:(fun idx s ->
              match s with LL.Workgroup_barrier -> Some idx | _ -> None)
        in
        if List.length barrier_positions >= 2 && not !mutated then (
          mutated := true;
          let first_b = List.hd_exn barrier_positions in
          let last_b = List.last_exn barrier_positions in
          let stmts' =
            match mode with
            | `Drop_first -> List.filteri stmts ~f:(fun idx _ -> idx <> first_b)
            | `Drop_last -> List.filteri stmts ~f:(fun idx _ -> idx <> last_b)
            | `Move_first_before_loads ->
                LL.Workgroup_barrier :: List.filteri stmts ~f:(fun idx _ -> idx <> first_b)
          in
          LL.For_loop { fc with body = LL.unflat_lines stmts' })
        else LL.For_loop { fc with body = go body }
    | LL.For_loop fc -> LL.For_loop { fc with body = go fc.body }
    | LL.Seq (a, b) -> LL.Seq (go a, go b)
    | LL.If i -> LL.If { i with body = go i.body }
    | other -> other
  in
  let llc = go opt.LL.llc in
  if not !mutated then failwith "mutate_ko_barriers: did not find k_o loop with >= 2 barriers";
  { opt with llc }

let () =
  (* === 1. SMEM matmul valid leg + 3 barrier-mutation negative controls === *)
  let want_mm, run_mm = build_smem_matmul () in
  let got_valid = run_mm ~name:"s678_mm_valid" ~mutate:Fn.id in
  p_all2 "smem matmul: valid shared-staging schedule matches serial oracle bitwise" got_valid
    want_mm ~f:Float.equal;

  in_leg ~leg:cc_leg
    "negative control: removing load->compute barrier fails parity and surfaces NaN poison"
    (fun label ->
      let got = run_mm ~name:"s678_mm_drop_b1" ~mutate:(mutate_ko_barriers ~mode:`Drop_first) in
      let mismatches = Array.counti got ~f:(fun i v -> not (Float.equal v want_mm.(i))) in
      let nans = Array.count got ~f:Float.is_nan in
      Stdio.eprintf
        "cc_workgroup_shared_staging: drop_b1 mismatches=%d nans=%d (not part of the golden)\n"
        mismatches nans;
      p label (mismatches > 0 && nans > 0));

  in_leg ~leg:cc_leg
    "negative control: removing trailing compute->next-load barrier fails parity (no back-edge \
     phantom barrier)" (fun label ->
      let got = run_mm ~name:"s678_mm_drop_b2" ~mutate:(mutate_ko_barriers ~mode:`Drop_last) in
      let mismatches = Array.counti got ~f:(fun i v -> not (Float.equal v want_mm.(i))) in
      Stdio.eprintf "cc_workgroup_shared_staging: drop_b2 mismatches=%d (not part of the golden)\n"
        mismatches;
      p label (mismatches > 0));

  in_leg ~leg:cc_leg
    "negative control: moving load->compute barrier before loads fails parity (same barrier count)"
    (fun label ->
      let got =
        run_mm ~name:"s678_mm_misplaced_b1"
          ~mutate:(mutate_ko_barriers ~mode:`Move_first_before_loads)
      in
      let mismatches = Array.counti got ~f:(fun i v -> not (Float.equal v want_mm.(i))) in
      Stdio.eprintf
        "cc_workgroup_shared_staging: misplaced_b1 mismatches=%d (not part of the golden)\n"
        mismatches;
      p label (mismatches > 0));

  (* === 2. Thread-private Declare_local crossing a Workgroup_barrier === *)
  (let n = 8 in
   let x_vals = Array.init n ~f:(fun i -> Float.of_int ((i * 3) + 1)) in
   let x = make_node ~dims:[| n |] "dl_x" in
   let out = make_node ~dims:[| n |] "dl_out" in
   L.materialize x;
   L.materialize out;
   let tile = make_shared ~dims:[| n |] "dl_tile" in
   let loc_tn = make_node ~dims:[| 1 |] "dl_loc" in
   L.virtualize loc_tn;
   let loc_id = LL.get_scope loc_tn in
   let w = L.sym () in
   let shifted =
     LL.Binop
       ( Ir.Ops.Mod,
         ( LL.Binop (Ir.Ops.Add, (L.embed w, Ir.Ops.index_prec ()), (L.ic 3, Ir.Ops.index_prec ())),
           Ir.Ops.index_prec () ),
         (L.ic n, Ir.Ops.index_prec ()) )
   in
   (* Each thread w stores a distinct private value (100 + w) in [loc_id], writes [tile[w] = x[w]],
      synchronizes, then writes [out[w] = loc_id + tile[(w + 3) % n]]. *)
   let llc =
     L.loop_n ~axis:LL.Workgroup w n
       (LL.unflat_lines
          [
            L.declare_local loc_id;
            LL.Set_local (loc_id, L.add (L.c 100.) (L.embed w));
            L.set_at tile (L.iter w) (L.get x [| L.iter w |]);
            LL.Workgroup_barrier;
            L.set_at out (L.iter w)
              (L.add (LL.Get_local loc_id)
                 (LL.Get_dynamic
                    {
                      tn = tile;
                      idcs = [| L.fixed 0 |];
                      dyn_axis = 0;
                      dyn_value = (shifted, Ir.Ops.index_prec ());
                    }));
          ])
   in
   let got =
     exec_handbuilt ~name:"s678_declare_local" ~shared:[ tile ] ~inputs:[ (x, x_vals) ] ~out llc
   in
   let want = Array.init n ~f:(fun i -> 100. +. Float.of_int i +. x_vals.((i + 3) % n)) in
   p_all2 "thread-private Declare_local survives across Workgroup_barrier" got want ~f:Float.equal);

  (* === 3. Scan_loop (Forward and Backward) with Workgroup_barrier inside Workgroup loop === *)
  let run_scan_leg ~direction ~name =
    let nw = 4 and nt = 4 in
    let out = make_node ~dims:[| nw |] (name ^ "_out") in
    L.materialize out;
    let tile = make_shared ~dims:[| nw |] (name ^ "_tile") in
    let acc_tn = make_node ~dims:[| 1 |] (name ^ "_acc") in
    L.virtualize acc_tn;
    let cr = L.carry ~init:(L.c 0.) acc_tn in
    let w = L.sym () and t = L.sym () in
    let next_w =
      LL.Binop
        ( Ir.Ops.Mod,
          ( LL.Binop (Ir.Ops.Add, (L.embed w, Ir.Ops.index_prec ()), (L.ic 1, Ir.Ops.index_prec ())),
            Ir.Ops.index_prec () ),
          (L.ic nw, Ir.Ops.index_prec ()) )
    in
    let step_val = L.add (L.mul (L.c 10.) (L.embed t)) (L.add (L.embed w) (L.c 1.)) in
    let llc =
      L.loop_n ~axis:LL.Workgroup w nw
        (L.scan ~direction ~upto:(nt - 1) t ~carried:[ cr ]
           (LL.unflat_lines
              [
                L.set_at tile (L.iter w) (L.add (L.prev cr) step_val);
                LL.Workgroup_barrier;
                L.set_next cr
                  (L.add (L.prev cr)
                     (LL.Get_dynamic
                        {
                          tn = tile;
                          idcs = [| L.fixed 0 |];
                          dyn_axis = 0;
                          dyn_value = (next_w, Ir.Ops.index_prec ());
                        }));
                LL.Workgroup_barrier;
                L.set_at out (L.iter w) (L.next cr);
              ]))
    in
    let got = exec_handbuilt ~name ~shared:[ tile ] ~inputs:[] ~out llc in
    (* Host oracle: step-by-step lockstep simulation over all nw threads. *)
    let acc = Array.create ~len:nw 0. in
    let steps =
      match direction with
      | LL.Forward -> List.init nt ~f:Fn.id
      | LL.Backward -> List.rev (List.init nt ~f:Fn.id)
    in
    List.iter steps ~f:(fun tv ->
        let published =
          Array.init nw ~f:(fun wv -> acc.(wv) +. (10. *. Float.of_int tv) +. Float.of_int (wv + 1))
        in
        for wv = 0 to nw - 1 do
          acc.(wv) <- acc.(wv) +. published.((wv + 1) % nw)
        done);
    (got, acc)
  in
  (let got_fwd, want_fwd = run_scan_leg ~direction:LL.Forward ~name:"s678_scan_fwd" in
   p_all2 "Forward Scan_loop enclosing Workgroup_barrier matches lockstep oracle bitwise" got_fwd
     want_fwd ~f:Float.equal;
   let got_bwd, want_bwd = run_scan_leg ~direction:LL.Backward ~name:"s678_scan_bwd" in
   p_all2 "Backward Scan_loop enclosing Workgroup_barrier matches lockstep oracle bitwise" got_bwd
     want_bwd ~f:Float.equal);

  (* === 4. Sibling Workgroup loops: with barrier (passes) vs without barrier (negative control)
     === *)
  (let n = 8 in
   let x_vals = Array.init n ~f:(fun i -> Float.of_int ((i * 5) + 2)) in
   let build_sibling ~with_barrier ~name =
     let x = make_node ~dims:[| n |] (name ^ "_x") in
     let out = make_node ~dims:[| n |] (name ^ "_out") in
     L.materialize x;
     L.materialize out;
     let tile = make_shared ~dims:[| n |] (name ^ "_tile") in
     let w1 = L.sym () and w2 = L.sym () in
     let shifted =
       LL.Binop
         ( Ir.Ops.Mod,
           ( LL.Binop
               (Ir.Ops.Add, (L.embed w2, Ir.Ops.index_prec ()), (L.ic 1, Ir.Ops.index_prec ())),
             Ir.Ops.index_prec () ),
           (L.ic n, Ir.Ops.index_prec ()) )
     in
     let stmts =
       [ L.loop_n ~axis:LL.Workgroup w1 n (L.set_at tile (L.iter w1) (L.get x [| L.iter w1 |])) ]
       @ (if with_barrier then [ LL.Workgroup_barrier ] else [])
       @ [
           L.loop_n ~axis:LL.Workgroup w2 n
             (L.set_at out (L.iter w2)
                (LL.Get_dynamic
                   {
                     tn = tile;
                     idcs = [| L.fixed 0 |];
                     dyn_axis = 0;
                     dyn_value = (shifted, Ir.Ops.index_prec ());
                   }));
         ]
     in
     exec_handbuilt ~name ~shared:[ tile ] ~inputs:[ (x, x_vals) ] ~out (LL.unflat_lines stmts)
   in
   let want = Array.init n ~f:(fun i -> x_vals.((i + 1) % n)) in
   let got_ok = build_sibling ~with_barrier:true ~name:"s678_sib_ok" in
   p_all2 "sibling Workgroup loops separated by Workgroup_barrier match oracle bitwise" got_ok want
     ~f:Float.equal;
   in_leg ~leg:cc_leg
     "negative control: removing barrier between sibling Workgroup loops fails parity" (fun label ->
       let got_bad = build_sibling ~with_barrier:false ~name:"s678_sib_bad" in
       let mismatches = Array.counti got_bad ~f:(fun i v -> not (Float.equal v want.(i))) in
       let nans = Array.count got_bad ~f:Float.is_nan in
       Stdio.eprintf
         "cc_workgroup_shared_staging: sibling_no_barrier mismatches=%d nans=%d (not part of the \
          golden)\n"
         mismatches nans;
       p label (mismatches > 0 && nans > 0)));

  (* === 5. Cross-nest Grid misalignment (gh-ocannl-643 trap): aligned passes, swapped fails === *)
  (let g0 = 2 and g1 = 2 and wg = 4 in
   let build_grid_align ~swap_zero ~name =
     let out = make_node ~dims:[| g1; g0; wg |] (name ^ "_out") in
     L.materialize out;
     let tile = make_shared ~dims:[| wg |] (name ^ "_tile") in
     let z1 = L.sym () and z0 = L.sym () and zw = L.sym () in
     let a1 = L.sym () and a0 = L.sym () and aw = L.sym () in
     let zero_nest =
       if swap_zero then
         (* Deliberately swap the two Grid axes relative to the accumulation nest: block (g1, g0)
            zeroes out[g0, g1, zw] while the accumulation updates out[g1, g0, aw]. *)
         L.loop_n ~axis:LL.Grid z1 g1
           (L.loop_n ~axis:LL.Grid z0 g0
              (L.loop_n ~axis:LL.Workgroup zw wg
                 (L.set out [| L.iter z0; L.iter z1; L.iter zw |] (L.c 0.))))
       else
         L.loop_n ~axis:LL.Grid z1 g1
           (L.loop_n ~axis:LL.Grid z0 g0
              (L.loop_n ~axis:LL.Workgroup zw wg
                 (L.set out [| L.iter z1; L.iter z0; L.iter zw |] (L.c 0.))))
     in
     let accum_nest =
       L.loop_n ~axis:LL.Grid a1 g1
         (L.loop_n ~axis:LL.Grid a0 g0
            (L.loop_n ~axis:LL.Workgroup aw wg
               (LL.unflat_lines
                  [
                    L.set_at tile (L.iter aw)
                      (L.add
                         (L.mul (L.c 10.) (L.embed a1))
                         (L.add (L.mul (L.c 2.) (L.embed a0)) (L.add (L.embed aw) (L.c 1.))));
                    LL.Workgroup_barrier;
                    L.set out
                      [| L.iter a1; L.iter a0; L.iter aw |]
                      (L.add
                         (L.get out [| L.iter a1; L.iter a0; L.iter aw |])
                         (L.get tile [| L.iter aw |]));
                  ])))
     in
     exec_handbuilt ~name ~shared:[ tile ] ~inputs:[] ~out (LL.Seq (zero_nest, accum_nest))
   in
   let want =
     Array.init
       (g1 * g0 * wg)
       ~f:(fun flat ->
         match L.unflat ~dims:[| g1; g0; wg |] flat with
         | [| b1; b0; wv |] -> Float.of_int ((10 * b1) + (2 * b0) + wv + 1)
         | _ -> assert false)
   in
   let got_aligned = build_grid_align ~swap_zero:false ~name:"s678_grid_aligned" in
   p_all2 "cross-nest aligned Grid zeroing + staged accumulation matches oracle bitwise" got_aligned
     want ~f:Float.equal;
   in_leg ~leg:cc_leg
     "negative control: swapped Grid slots between zeroing and accumulation fails parity"
     (fun label ->
       let got_misaligned = build_grid_align ~swap_zero:true ~name:"s678_grid_misaligned" in
       let mismatches = Array.counti got_misaligned ~f:(fun i v -> not (Float.equal v want.(i))) in
       Stdio.eprintf
         "cc_workgroup_shared_staging: grid_misaligned mismatches=%d (not part of the golden)\n"
         mismatches;
       p label (mismatches > 0)));

  (* === 6. Validation gates still reject illegal barrier shapes on every backend === *)
  let expect_validate_rejected ~name ~shared ~out llc =
    try
      ignore (exec_handbuilt ~name ~shared ~inputs:[] ~out llc : float array);
      false
    with Invalid_argument msg -> String.is_substring msg ~substring:"validate_parallel"
  in
  let out1 = make_node ~dims:[| 4 |] "vg_out1" in
  L.materialize out1;
  let tile1 = make_shared ~dims:[| 4 |] "vg_tile1" in
  let w1 = L.sym () in
  let guarded_barrier =
    L.loop_n ~axis:LL.Workgroup w1 4
      (LL.unflat_lines
         [
           L.set_at tile1 (L.iter w1) (L.tick w1);
           L.if_idx (L.lt (L.embed w1) (L.ic 2)) LL.Workgroup_barrier;
           L.set_at out1 (L.iter w1) (L.get tile1 [| L.iter w1 |]);
         ])
  in
  p "validation gate: Workgroup_barrier under an If guard is still rejected"
    (expect_validate_rejected ~name:"s678_vg_if" ~shared:[ tile1 ] ~out:out1 guarded_barrier);

  let out2 = make_node ~dims:[| 8 |] "vg_out2" in
  L.materialize out2;
  let tile2 = make_shared ~dims:[| 8 |] "vg_tile2" in
  let wa = L.sym () and wb = L.sym () in
  let mismatched_wg =
    LL.unflat_lines
      [
        L.loop_n ~axis:LL.Workgroup wa 8 (L.set_at tile2 (L.iter wa) (L.tick wa));
        LL.Workgroup_barrier;
        L.loop_n ~axis:LL.Workgroup wb 4 (L.set_at out2 (L.iter wb) (L.get tile2 [| L.iter wb |]));
      ]
  in
  p "validation gate: mismatched Workgroup extents across a barrier are still rejected"
    (expect_validate_rejected ~name:"s678_vg_ext" ~shared:[ tile2 ] ~out:out2 mismatched_wg)
