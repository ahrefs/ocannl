(* The autotuner's config-threshold fission candidate must reproduce the untuned default pipeline
   exactly (PR #140: "tuned >= untuned by construction" rests on it). Three layers of the property,
   on a computation replicating the benchmark suite's mlp workload — batch slicing over a static
   index, an MLP with relu, the softmax cross-entropy loss (whose intermediate chain was implicated
   by the CUDA round-5 segment diff), gradient update + SGD:

   1. Base lowerings agree across sibling root contexts: the tuned cell captures its base from a
   scratch context (the [timing_ctx] of [Train.tune_placements]) while the untuned cell compiles
   from its own context — both roots initialized the same way must lower the step identically. 2.
   Given one captured lowering, [Sched.maybe_default_schedules] (the untuned pipeline) and the
   autotuner's candidate pipeline ([Sched.fission_scheduled] with config-default presets, mirroring
   [Autotune.compile_candidate]'s [F_preset { block_size = None; config_thresholds = true; privatize
   = false }]) produce the same segments. 3. End to end: the candidate pipeline on the scratch
   capture equals the untuned pipeline on the target-context capture — the actual arm-A-candidate vs
   untuned-cell comparison. *)

open Base
open Ocannl
module IDX = Train.IDX
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module SC = Ir.Schedule_cache
module Sched = Ir.Schedule
module Asgns = Ir.Assignments

let cross_entropy_loss = Nn_blocks.cross_entropy_loss

open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

let () =
  let n_batches = 4 and batch = 16 and d_in = 8 and d_hid = 16 and classes = 4 in
  let xs =
    NTDSL.init ~l:"xs" ~prec:Ir.Ops.single ~b:[ n_batches; batch ] ~i:[] ~o:[ d_in ]
      ~f:(Ll_test.weighted ~weights:[| 1; 2; 1 |] ~modulus:5 ~offset:0. ~stride:0.1)
      ()
  in
  let ys =
    NTDSL.init ~l:"ys" ~prec:Ir.Ops.single ~b:[ n_batches; batch ] ~i:[] ~o:[ classes ]
      ~f:(fun idcs -> if (idcs.(0) + idcs.(1)) % classes = idcs.(2) then 1. else 0.)
      ()
  in
  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let%op batch_x = xs @| batch_n in
  let%op batch_y = ys @| batch_n in
  let w1v =
    Array.init (d_hid * d_in)
      ~f:(Ll_test.cycle_flat ~dims:[| d_hid; d_in |] ~modulus:7 ~offset:0. ~stride:0.05)
  in
  let w2v =
    Array.init (classes * d_hid)
      ~f:(Ll_test.cycle_flat ~dims:[| classes; d_hid |] ~modulus:5 ~offset:0. ~stride:0.04)
  in
  let w1 =
    Operation.init ~l:"w1" ~prec:Ir.Ops.single ~b:[] ~i:[ d_in ] ~o:[ d_hid ]
      ~f:(fun idcs -> w1v.((idcs.(0) * d_in) + idcs.(1)))
      ~grad_spec:Tensor.Require_grad ()
  in
  let w2 =
    Operation.init ~l:"w2" ~prec:Ir.Ops.single ~b:[] ~i:[ d_hid ] ~o:[ classes ]
      ~f:(fun idcs -> w2v.((idcs.(0) * d_hid) + idcs.(1)))
      ~grad_spec:Tensor.Require_grad ()
  in
  let%op logits = w2 * relu (w1 * batch_x) in
  let%op batch_loss =
    cross_entropy_loss ~spec:"...|v" ~normalize_by:!..batch () ~logits ~targets:batch_y
  in
  let update = Train.grad_update batch_loss in
  (* [w1]/[w2] are deliberately unregistered ([Operation.init ~grad_spec:Require_grad] leaves do not
     join [loss.params]), so the derived trainable set is empty — pass the parameter set explicitly
     so the workload really contains its SGD step (gh-ocannl-670: before the params-driven helpers
     grew this guard, this [sgd_update] silently compiled an empty block). *)
  let sgd =
    Train.sgd_update ~learning_rate:(TDSL.O.( !. ) 0.01)
      ~params:(Set.of_list (module Tensor) [ w1; w2 ])
      batch_loss
  in
  let step = Asgns.sequence [ update; sgd ] in
  let static_indices = Ir.Indexing.bound_symbols bindings in
  let capture cctx =
    let r = ref None in
    let (_ : Context.t), (_ : Context.routine) =
      Context.compile
        ~lowered_transform:(fun o ->
          r := Some (o, SC.canonicalize ~static_indices o);
          [ o ])
        cctx step bindings
    in
    Option.value_exn !r
  in
  (* The "untuned cell": its own root context. *)
  let ctx = Context.auto () in
  let ctx = Train.init_params ctx bindings batch_loss in
  let opt_ctx, canon_ctx = capture ctx in
  (* The "tuned cell"'s arm-A base: a scratch sibling root, initialized the same way (the
     [timing_ctx] recipe of the benchmark runners). *)
  let scratch = Train.init_params (Context.auto ()) bindings batch_loss in
  let opt_scratch, canon_scratch = capture scratch in
  p "base lowerings agree across sibling roots"
    (String.equal (SC.digest canon_ctx) (SC.digest canon_scratch));
  let limits = Context.hardware_limits ctx in
  let copy (opt : LL.optimized) =
    {
      opt with
      LL.traced_store = Hashtbl.copy opt.LL.traced_store;
      LL.optimize_ctx = LL.copy_optimize_ctx opt.LL.optimize_ctx;
    }
  in
  let untuned opt =
    Sched.maybe_default_schedules ~backend_name ~limits ~static_indices (copy opt)
  in
  let is_gpu = Sched.backend_is_gpu backend_name in
  let is_cpu = Sched.backend_is_cpu backend_name in
  let candidate opt =
    (* Mirrors Autotune.compile_candidate's preset for [F_preset { block_size = None; privatize =
       false; config_thresholds = true }]. *)
    let preset seg =
      if is_gpu then Sched.default_gpu ~limits seg else if is_cpu then Sched.default_cpu seg else []
    in
    let zero_sched tns = if is_gpu then Sched.zero_expansion ~limits tns else [] in
    List.map
      (Sched.fission_scheduled ~promote_locals:is_gpu ~preset ~zero_sched ~static_indices (copy opt))
      ~f:(fun (_, _, _, post) -> post)
  in
  let digests posts =
    List.map posts ~f:(fun post -> SC.digest (SC.canonicalize ~static_indices post))
  in
  let report name a b =
    (* Counts are backend-dependent (cc: single serial segment; GPU backends fission), so print them
       only on divergence to keep the expected output backend-stable. *)
    if List.length a <> List.length b then
      Stdio.printf "DIVERGENCE at %s — %d segments vs %d segments\n" name (List.length a)
        (List.length b);
    p (name ^ ": segment counts agree") (List.length a = List.length b);
    p (name ^ ": per-segment digests agree") (List.equal String.equal (digests a) (digests b))
  in
  report "same capture" (untuned opt_ctx) (candidate opt_ctx);
  report "cross capture (arm-A candidate vs untuned cell)" (untuned opt_ctx) (candidate opt_scratch)
