(* gh-ocannl-1493: [Context.link_delta] plans, without codegen or link, the device allocation that
   linking a compile would make -- the seam [Autotune.tune]'s GPU baseline probe and a placement
   arm's pre-compile check read. Pinned here against the thing it predicts: the allocation census's
   delta over the real compile of the same lowering from the same context.

   Each estimate is taken INSIDE the compile's [lowered_transform], on the lowering as the transform
   receives it and the segments it returns -- before codegen settles any open placement, which is
   where the probe takes it (the GPU baseline never reaches codegen). Covered: a forward routine and
   a whole training step (forward, backprop, SGD); the unscheduled form ([fun o -> [ o ]], the
   search's baseline) and the default schedules (fission, where the backend fissions); a fresh
   context, and one already holding the parameters, whose delta must leave them out; constants the
   device cached in an earlier compile (every compile after a routine's first); [buffer_aliasing]
   off and then on, where the working group is one liveness-planned arena.

   And [Context.probe_link_delta], the trial allocation: it allocates exactly the planned pools and
   frees them, leaving nothing live, and under a simulated capacity it is refused, typed, with
   nothing left live either. *)

open Base
open Ocannl
module IDX = Train.IDX
open Operation.DSL_modules
module Asgns = Ir.Assignments
module FI = Ir.Resource_fault_injection
module AC = Ir.Alloc_census
module SO = Ir.Schedule_outcome
open Verdict.Claims

let delta_between (a : AC.t) (b : AC.t) : Context.link_delta =
  {
    ld_working_bytes = b.live_working_bytes - a.live_working_bytes;
    ld_constant_bytes = b.live_constant_bytes - a.live_constant_bytes;
    ld_pools =
      b.live_working_pools + b.live_constant_pools - a.live_working_pools - a.live_constant_pools;
  }

let show (d : Context.link_delta) =
  Printf.sprintf "%d working + %d constant bytes in %d pool(s)" d.ld_working_bytes
    d.ld_constant_bytes d.ld_pools

let count_hits f =
  let hits = ref 0 in
  let result = FI.with_callback (fun _ -> Int.incr hits) ~f in
  (result, !hits)

(* A device with no room for the probe's first pool: the shared seam refuses the allocation after
   the census recorded it, so the probe's own unwind has to give it back. *)
let refuse_first f =
  let refused = ref 0 in
  let refuse = function
    | FI.Delta_pool_allocated when !refused = 0 ->
        Int.incr refused;
        raise
          (SO.Device_memory_exhausted
             {
               backend = "simulated";
               requested = 1;
               free = Some 0;
               detail = "gh-ocannl-1493 simulated capacity";
             })
    | _ -> ()
  in
  let result = FI.with_callback refuse ~f:(fun () -> Result.try_with f) in
  (result, !refused)

type form = Unscheduled | Default

let form_name = function Unscheduled -> "unscheduled" | Default -> "default schedules"

(* Compiles [comp] from [ctx] in [form], estimating its link delta inside the transform, and returns
   the child context, the estimate and the census's delta over the compile. With [~probe], the
   transform also runs the trial allocation and its refused twin, and the claims about them are made
   there. *)
let compile_and_estimate ?(probe = false) ~what ~form ctx comp =
  let estimate = ref None in
  let lowered_transform (o : Ir.Low_level.optimized) =
    let segments =
      match form with
      | Unscheduled -> [ o ]
      | Default ->
          Ir.Schedule.maybe_default_schedules ~backend_name:(Context.backend_name ctx)
            ~limits:(Context.hardware_limits ctx) ~static_indices:[] o
    in
    let before = AC.snapshot () in
    let d, hits = count_hits (fun () -> Context.link_delta ctx o segments) in
    let after = AC.snapshot () in
    p
      (what ^ ": the estimate allocates nothing and reaches no allocation seam")
      (hits = 0 && Context.equal_link_delta (delta_between before after) (delta_between after after));
    estimate := Some d;
    if probe then (
      let before = AC.snapshot () in
      let probed, hits = count_hits (fun () -> Context.probe_link_delta ctx o segments) in
      let after = AC.snapshot () in
      p (what ^ ": the probe plans what the estimate plans") (Context.equal_link_delta probed d);
      p
        (what ^ ": the probe allocates exactly the planned pools, and frees them all")
        (hits = d.ld_pools
        && after.working_pools_allocated + after.constant_pools_allocated
           - before.working_pools_allocated - before.constant_pools_allocated
           = d.ld_pools
        && after.pools_freed - before.pools_freed = d.ld_pools
        && Context.equal_link_delta (delta_between before after) (delta_between after after));
      let refused, n_refused = refuse_first (fun () -> Context.probe_link_delta ctx o segments) in
      let after_refusal = AC.snapshot () in
      p
        (what ^ ": a device refusing the probe's allocation refuses the probe, typed")
        (n_refused = 1
        &&
        match refused with
        | Error (SO.Device_memory_exhausted _) -> true
        | Ok _ | Error _ -> false);
      p
        (what ^ ": the refused probe leaves nothing live")
        (Context.equal_link_delta (delta_between after after_refusal) (delta_between after after)));
    segments
  in
  let before = AC.snapshot () in
  let ctx', routine = Context.compile ~lowered_transform ctx comp IDX.empty in
  let measured = delta_between before (AC.snapshot ()) in
  let estimate = Option.value_exn !estimate in
  Stdio.eprintf "%s (not part of the golden): estimated %s, linked %s\n%!" what (show estimate)
    (show measured);
  p (what ^ ": the estimate is the link's allocation") (Context.equal_link_delta estimate measured);
  (ctx', routine, estimate)

let scenario ~aliasing =
  Unix.putenv "OCANNL_BUFFER_ALIASING" (Bool.to_string aliasing);
  let tag s = Printf.sprintf "[aliasing %s] %s" (if aliasing then "on" else "off") s in
  Tensor.unsafe_reinitialize ();
  Utils.settings.fixed_state_for_init <- Some 5;
  let n = 16 in
  let ma =
    NTDSL.init ~l:"ld_ma" ~prec:Ir.Ops.single ~o:[ n ] ~i:[ n ]
      ~f:(Ll_test.cycle ~dims:[| n; n |] ~modulus:7 ~offset:(-3.) ~stride:1.)
      ()
  in
  let mb =
    NTDSL.init ~l:"ld_mb" ~prec:Ir.Ops.single ~o:[ n ] ~i:[ n ]
      ~f:(Ll_test.cycle ~dims:[| n; n |] ~modulus:5 ~offset:(-2.) ~stride:1.)
      ()
  in
  let%op mc = ma * mb in
  let%op fwd = relu mc in
  ignore mc;
  let forward = Train.forward fwd in
  List.iter [ Unscheduled; Default ] ~f:(fun form ->
      let what = tag ("forward, fresh context, " ^ form_name form) in
      let ctx, _, d = compile_and_estimate ~probe:true ~what ~form (Context.auto ()) forward in
      p (what ^ ": the delta holds working bytes") (d.ld_working_bytes > 0);
      Context.release ctx);
  (* A training step from a context already holding its parameters: the delta leaves them out, as
     the link's allocator does -- an estimate that counted them would exceed the link's census. *)
  let batch = 32 and d_in = 64 and d_hid = 256 in
  let xs =
    NTDSL.init ~l:"ld_xs" ~prec:Ir.Ops.single ~b:[ batch ] ~o:[ d_in ]
      ~f:(Ll_test.cycle ~dims:[| batch; d_in |] ~modulus:9 ~offset:0. ~stride:0.125)
      ()
  in
  let ys =
    NTDSL.init ~l:"ld_ys" ~prec:Ir.Ops.single ~b:[ batch ] ~o:[ 1 ]
      ~f:(fun idcs -> Float.of_int (idcs.(0) % 3) /. 2.)
      ()
  in
  (* Three hidden layers: in backprop the layer-k activation gradient dies once the layer-(k-1) one
     is computed, so the gradient chain has liveness-disjoint links for the arena to overlap (the
     [buffer_aliasing] test's reasoning). *)
  let%op mlp x =
    { w4 }
    * relu
        ({ b3; o = [ d_hid ] }
        + { w3 }
          * relu ({ b2; o = [ d_hid ] } + ({ w2 } * relu ({ b1; o = [ d_hid ] } + ({ w1 } * x)))))
  in
  let%op err = mlp xs - ys in
  let%op loss = ((err *. err) ++ "...|... => |->0") /. !..batch in
  let step =
    Asgns.sequence
      [ Train.grad_update loss; Train.sgd_update ~learning_rate:(TDSL.O.( !. ) 0.0625) loss ]
  in
  let with_params = Train.init_params (Context.auto ()) IDX.empty loss in
  let step_deltas =
    List.map [ Unscheduled; Default ] ~f:(fun form ->
        let what = tag ("training step, parameters held, " ^ form_name form) in
        let ctx, _, d = compile_and_estimate ~probe:true ~what ~form with_params step in
        Context.release ctx;
        (form, d))
  in
  Context.release with_params;
  List.Assoc.find_exn step_deltas ~equal:Poly.equal Default

let () =
  Stdio.eprintf "backend: %s (not part of the golden)\n%!" (Context.backend_name (Context.auto ()));
  let off = scenario ~aliasing:false in
  let on = scenario ~aliasing:true in
  (* The arena leg is reached: the default schedules' training step shares bytes under the liveness
     planner, on the statement-granularity (CPU) and segment-granularity (GPU) planners alike (the
     [buffer_aliasing] test's model), so its delta -- estimated and linked alike, above --
     shrinks. *)
  p "the training step's delta is a liveness-planned arena under buffer_aliasing"
    (on.ld_working_bytes < off.ld_working_bytes)
