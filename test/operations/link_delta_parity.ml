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
   nothing left live either -- at its first pool and at each later one.

   gh-ocannl-1554: [Context.link_floor], the floor under every schedule of the lowering that
   [Autotune.tune]'s search-wide device-memory refusal asks about, taken on the same lowering in the
   same place. Pinned: planning it touches no seam and leaves the lowering's placements and store as
   they were; its probe allocates exactly its pools and frees them, and a refusal at any of its
   pools is typed and leaves nothing live; and it is at most what each compiled form actually
   allocated -- the unscheduled form, the default schedules (under [buffer_aliasing] the fissioned
   training step allocates LESS than the unscheduled delta on a GPU backend, which is why that delta
   bounds no candidate from below), and a shared-fused epilogue candidate (which keeps its reduction
   target off the device on a GPU backend, below the unscheduled delta too). *)

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
    ld_liveness_planned = false (* Not the census's to say: compared apart. *);
  }

let show (d : Context.link_delta) =
  Printf.sprintf "%d working + %d constant bytes in %d pool(s)" d.ld_working_bytes
    d.ld_constant_bytes d.ld_pools

let count_hits f =
  let hits = ref 0 in
  let result = FI.with_callback (fun _ -> Int.incr hits) ~f in
  (result, !hits)

(* A device with no room for the probe's [nth] pool (1-based): the shared seam refuses that
   allocation after the census recorded it, so the probe's own unwind has to give it back, and the
   pools before it as well. *)
let refuse_nth nth f =
  let refused = ref 0 and seen = ref 0 in
  let refuse = function
    | FI.Delta_pool_allocated
      when !refused = 0
           &&
           (Int.incr seen;
            !seen = nth) ->
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

let refused_typed = function
  | Error (SO.Device_memory_exhausted _) -> true
  | Ok _ | Error _ -> false

(* Every pool position of a probe of [pools] pools, refused in turn: typed, and nothing left
   live. *)
let refusals_unwind ~what ~pools probe =
  p_all
    (what
   ^ ": a device refusing any of the probe's pools refuses the probe, typed, and leaves nothing \
      live")
    (List.init pools ~f:(fun i -> i + 1))
    ~f:(fun nth ->
      let before = AC.snapshot () in
      let refused, n_refused = refuse_nth nth probe in
      let after = AC.snapshot () in
      n_refused = 1 && refused_typed refused
      && Context.equal_link_delta (delta_between before after) (delta_between after after))

let show_floor (f : Context.link_floor) =
  Printf.sprintf
    "%d working + %d constant bytes in %d pool(s); left out %d off-device, %d replaceable, %d \
     overlappable"
    f.lf_working_bytes f.lf_constant_bytes f.lf_pools f.lf_off_device_bytes f.lf_replaceable_bytes
    f.lf_overlap_bytes

(* gh-ocannl-1554: the floor of [o] on [ctx], with its planning and probe claims. *)
let floor_of ~what ctx (o : Ir.Low_level.optimized) =
  let plc = o.Ir.Low_level.optimize_ctx.Ir.Low_level.placements in
  let modes () =
    List.map (Hashtbl.to_alist o.Ir.Low_level.traced_store) ~f:(fun (tn, _) ->
        (tn, Option.map (Ir.Tnode.Placements.get plc tn) ~f:fst))
  in
  let modes_before = modes () in
  let before = AC.snapshot () in
  let f, hits = count_hits (fun () -> Context.link_floor ctx o) in
  let after = AC.snapshot () in
  p
    (what
   ^ ": the floor allocates nothing, reaches no allocation seam, and leaves the lowering's \
      placements and store as they were")
    (hits = 0
    && Context.equal_link_delta (delta_between before after) (delta_between after after)
    && List.equal
         (fun (a, ma) (b, mb) ->
           Ir.Tnode.equal a b && Option.equal Ir.Tnode.equal_memory_mode ma mb)
         modes_before (modes ()));
  p (what ^ ": the floor asks for at least one pool") (f.lf_pools >= 1);
  let before = AC.snapshot () in
  let probed, hits = count_hits (fun () -> Context.probe_link_floor ctx o) in
  let after = AC.snapshot () in
  p
    (what ^ ": the floor's probe allocates exactly its pools, and frees them all")
    (Context.equal_link_floor probed f && hits = f.lf_pools
    && after.working_pools_allocated + after.constant_pools_allocated
       - before.working_pools_allocated - before.constant_pools_allocated
       = f.lf_pools
    && after.pools_freed - before.pools_freed = f.lf_pools
    && Context.equal_link_delta (delta_between before after) (delta_between after after));
  refusals_unwind ~what:(what ^ " (floor)") ~pools:f.lf_pools (fun () ->
      Context.probe_link_floor ctx o);
  f

(* Whether [f] is at most the allocation [measured]: a floor under it. *)
let below (f : Context.link_floor) (measured : Context.link_delta) =
  f.lf_working_bytes <= measured.ld_working_bytes
  && f.lf_constant_bytes <= measured.ld_constant_bytes

type form = Unscheduled | Default

let form_name = function Unscheduled -> "unscheduled" | Default -> "default schedules"

(* Compiles [comp] from [ctx] in [form], estimating its link delta inside the transform, and returns
   the child context, the estimate and the census's delta over the compile. With [~probe], the
   transform also runs the trial allocation and its refused twin, and the claims about them are made
   there. *)
let compile_and_estimate ?(probe = false) ~what ~form ctx comp =
  let estimate = ref None and floor = ref None in
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
      refusals_unwind ~what ~pools:d.ld_pools (fun () -> Context.probe_link_delta ctx o segments);
      floor := Some (floor_of ~what ctx o));
    segments
  in
  let before = AC.snapshot () in
  let ctx', routine = Context.compile ~lowered_transform ctx comp IDX.empty in
  let measured = delta_between before (AC.snapshot ()) in
  let estimate = Option.value_exn !estimate in
  Stdio.eprintf "%s (not part of the golden): estimated %s, linked %s\n%!" what (show estimate)
    (show measured);
  p
    (what ^ ": the estimate is the link's allocation")
    (Context.equal_link_delta { estimate with ld_liveness_planned = false } measured);
  Option.iter !floor ~f:(fun f ->
      Stdio.eprintf "%s (not part of the golden): floor %s\n%!" what (show_floor f);
      p (what ^ ": the floor is at most the link's allocation") (below f measured));
  (ctx', routine, estimate, measured)

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
      let ctx, _, d, _ = compile_and_estimate ~probe:true ~what ~form (Context.auto ()) forward in
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
        let ctx, _, d, measured = compile_and_estimate ~probe:true ~what ~form with_params step in
        Context.release ctx;
        (form, (d, measured)))
  in
  Context.release with_params;
  (* gh-ocannl-1554: why the search's refusal asks the floor and not the unscheduled delta. On a GPU
     backend the liveness planner's position is the kernel, so the one-kernel unscheduled step
     overlaps nothing, while the default schedules' fissioned step overlaps what its segments
     separate and allocates less: the unscheduled delta is no lower bound there, which the floor
     claims above are (the default form's floor claim is this very comparison's other side). *)
  let backend_name = Context.backend_name (Context.auto ()) in
  (let unscheduled, _ = List.Assoc.find_exn step_deltas ~equal:Poly.equal Unscheduled in
   let _, default_linked = List.Assoc.find_exn step_deltas ~equal:Poly.equal Default in
   Stdio.eprintf
     "%s (not part of the golden): unscheduled delta %d working bytes, default schedules linked %d\n\
      %!"
     (tag "training step") unscheduled.ld_working_bytes default_linked.ld_working_bytes;
   if aliasing then
     Verdict.gated
       ~when_:(Ir.Schedule.backend_is_gpu backend_name)
       ~on:backend_name
       (tag
          "training step: on a GPU backend under buffer_aliasing the default schedules allocate \
           less than the unscheduled delta")
       (default_linked.ld_working_bytes < unscheduled.ld_working_bytes));
  let step_deltas = List.map step_deltas ~f:(fun (form, (d, _)) -> (form, d)) in
  (* The flag [Autotune.tune]'s probe abstains on: the training step has nodes the liveness planner
     may overlap, in either form, exactly when the planner is on. *)
  p_all (tag "training step: the plan says it is liveness-planned exactly under buffer_aliasing")
    step_deltas ~f:(fun (_, (d : Context.link_delta)) -> Bool.equal d.ld_liveness_planned aliasing);
  List.Assoc.find_exn step_deltas ~equal:Poly.equal Default

(* gh-ocannl-1554: on-chip fusion. A matmul whose result only an elementwise tail reads, sized past
   the default [stack_threshold_in_bytes] (so the unscheduled compile settles it on the device) and
   within every GPU backend's workgroup memory (so a shared epilogue placement may hold it): the
   staged, tensorized [Fuse_epilogue] twin of [schedule_epilogue_fusion] with [shared] on a GPU
   backend keeps it off the device. The floor leaves exactly such a node out there, and nothing on a
   CPU backend, where shared placement does not exist. *)
let fusion () =
  Unix.putenv "OCANNL_BUFFER_ALIASING" "false";
  Tensor.unsafe_reinitialize ();
  let backend_name = Context.backend_name (Context.auto ()) in
  let on_gpu = Ir.Schedule.backend_is_gpu backend_name in
  let m = 192 and k = 32 and nn = 32 and bm = 16 and simd_width = 32 in
  let make () =
    let ma =
      NTDSL.init ~l:"lf_ma" ~prec:Ir.Ops.single ~o:[ m ] ~i:[ k ]
        ~f:(Ll_test.weighted ~weights:[| 3; 1 |] ~modulus:7 ~offset:(-3.) ~stride:1.)
        ()
    in
    let mb =
      NTDSL.init ~l:"lf_mb" ~prec:Ir.Ops.single ~o:[ k ] ~i:[ nn ]
        ~f:(Ll_test.weighted ~weights:[| 1; 2 |] ~modulus:5 ~offset:(-2.) ~stride:1.)
        ()
    in
    let%op prod = ma * mb in
    let%op out = relu prod in
    (prod, out)
  in
  let module LL = Ir.Low_level in
  let module Sched = Ir.Schedule in
  let fused (prod : Tensor.t) (opt : LL.optimized) =
    let i, j, k =
      match List.find_exn (Ll_test.nest_paths opt.LL.llc) ~f:(fun p -> List.length p = 3) with
      | [ i; j; k ] -> (i, j, k)
      | _ -> assert false
    in
    let target = prod.Tensor.value in
    let ez, zsyms = Sched.expand_zero ~tn:target in
    let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
    let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let rz = Sched.Retype { axis = zj; ty = LL.Workgroup } in
    let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bm ~outer:LL.Serial ~inner:LL.Serial in
    let tz, _lane = Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width () in
    Sched.apply
      [
        ez;
        sp_zi;
        rz;
        sp_i;
        sp_k;
        Sched.Swap { outer = j; inner = k_o };
        Sched.Swap { outer = i_i; inner = k_o };
        tz;
        Sched.Fuse_epilogue { target; shared = on_gpu };
      ]
      opt
  in
  let _, reference = make () in
  let ctx, routine = Context.compile (Context.auto ()) (Train.forward reference) IDX.empty in
  let want = Context.get_values (Context.run ctx routine) reference.Tensor.value in
  Context.release ctx;
  let prod, out = make () in
  let ctx = Context.auto () in
  let floor = ref None and unscheduled = ref None in
  let lowered_transform o =
    unscheduled := Some (Context.link_delta ctx o [ o ]);
    floor := Some (floor_of ~what:"fusion" ctx o);
    [ fused prod o ]
  in
  let before = AC.snapshot () in
  let ctx', routine = Context.compile ~lowered_transform ctx (Train.forward out) IDX.empty in
  let measured = delta_between before (AC.snapshot ()) in
  let got = Context.get_values (Context.run ctx' routine) out.Tensor.value in
  Context.release ctx';
  let f = Option.value_exn !floor and unscheduled = Option.value_exn !unscheduled in
  let prod_bytes = m * nn * 4 in
  Stdio.eprintf
    "fusion (not part of the golden): unscheduled %s, fused linked %s, floor %s, the target %d B\n\
     %!"
    (show unscheduled) (show measured) (show_floor f) prod_bytes;
  p "fusion: the fused candidate computes the right values"
    (Array.equal (fun a b -> Float.(abs (a - b) <= 1e-3)) got want);
  p "fusion: the floor is at most the fused candidate's allocation" (below f measured);
  p "fusion: the floor leaves the fusion target out on a GPU backend, and nothing on a CPU one"
    (if on_gpu then f.lf_off_device_bytes >= prod_bytes else f.lf_off_device_bytes = 0);
  Verdict.gated ~when_:on_gpu ~on:backend_name
    "fusion: on a GPU backend the shared-fused candidate allocates less than the unscheduled delta"
    (measured.ld_working_bytes < unscheduled.ld_working_bytes)

let () =
  Stdio.eprintf "backend: %s (not part of the golden)\n%!" (Context.backend_name (Context.auto ()));
  let off = scenario ~aliasing:false in
  let on = scenario ~aliasing:true in
  (* The arena leg is reached: the default schedules' training step shares bytes under the liveness
     planner, on the statement-granularity (CPU) and segment-granularity (GPU) planners alike (the
     [buffer_aliasing] test's model), so its delta -- estimated and linked alike, above --
     shrinks. *)
  p "the training step's delta is a liveness-planned arena under buffer_aliasing"
    (on.ld_working_bytes < off.ld_working_bytes);
  fusion ()
