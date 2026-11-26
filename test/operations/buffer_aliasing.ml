(* gh-ocannl-489 liveness-based buffer aliasing (the memory planner):

   1. Unit checks of the pure arena planner [Backends.plan_arena_offsets]: liveness-disjoint
   same-precision items share bytes; overlapping spans, cross-precision pairs and always-live items
   never do; layouts over the cap fall back (return [None]).

   1b. Unit checks of [Low_level.sink_zero_outs] (the planner-gated pass that un-pins gradient live
   spans from [Train.grad_update]'s up-front zero-grads block): a [Zero_out] sinks to just before
   the first later statement accessing its node, stops at a [Workgroup_barrier], and stays in place
   when its node is never re-accessed in-routine.

   2. A forward chain of materialized unobservable intermediates: under [buffer_aliasing] the
   liveness-disjoint links share bytes (footprint shrinks), post-hoc host reads of the aliased
   intermediates raise the buffer-aliased [User_error] (the read guard), while the observable result
   stays readable. Golden parity: the result value is identical with the planner on and off.

   3. A training step (forward + backprop + SGD compiled as ONE routine, the whole-step program the
   planner targets): the loss trajectory is bitwise identical with the planner on and off (a
   miscolored interval corrupts values — this catches it), the loss stays host-readable (the
   observation intent [Train.grad_update] declares), and the step's working footprint shrinks.

   4. An observable virtual result replays a non-observable buffer after the routine runs
   (gh-ocannl-2461). A separately stored virtual component extends that buffer's lifetime, while a
   materialized component shields it (gh-ocannl-2493).

   The [buffer_aliasing] gate is flipped via the environment between sections; the config is read
   afresh at each compile. Printed facts are booleans/PASS lines so the expected output stays
   backend-stable (byte sizes and which exact pairs share differ between statement-granularity CPU
   and segment-granularity GPU liveness). *)

open Base
open Stdio
open Ocannl
module IDX = Train.IDX
open Nn_blocks.DSL_modules
module Tn = Ir.Tnode
module Asgns = Ir.Assignments
module Backends = Context.Backends
open Verdict.Claims

let () =
  let cap = 0x1_0000_0000 in
  let f = Backends.plan_arena_offsets ~cap in
  (match f [ (64, 8, "single", Some (0, 1)); (64, 8, "single", Some (2, 3)) ] with
  | Some ([ o1; o2 ], total) -> p "planner: disjoint same-prec share bytes" (o1 = o2 && total = 64)
  | _ -> p "planner: disjoint same-prec share bytes" false);
  (match f [ (64, 8, "single", Some (0, 2)); (64, 8, "single", Some (2, 3)) ] with
  | Some ([ o1; o2 ], total) ->
      p "planner: overlapping spans get disjoint bytes" (o1 <> o2 && total = 128)
  | _ -> p "planner: overlapping spans get disjoint bytes" false);
  (match f [ (64, 8, "single", Some (0, 1)); (64, 8, "int32", Some (2, 3)) ] with
  | Some ([ o1; o2 ], total) -> p "planner: cross-precision never shares" (o1 <> o2 && total = 128)
  | _ -> p "planner: cross-precision never shares" false);
  (match f [ (64, 8, "single", None); (64, 8, "single", Some (2, 3)) ] with
  | Some ([ o1; o2 ], total) -> p "planner: always-live conflicts with all" (o1 <> o2 && total = 128)
  | _ -> p "planner: always-live conflicts with all" false);
  (* Greedy by size: the large always-live pair claims low offsets; the two small disjoint items
     tuck into the same bytes. *)
  (match
     f
       [
         (16, 8, "single", Some (0, 1));
         (128, 8, "single", None);
         (16, 8, "single", Some (2, 3));
         (128, 8, "single", None);
       ]
   with
  | Some ([ s1; b1; s2; b2 ], total) ->
      p "planner: greedy-by-size packs small pair into one gap"
        (s1 = s2 && b1 <> b2 && total = 128 + 128 + 16)
  | _ -> p "planner: greedy-by-size packs small pair into one gap" false);
  match
    Backends.plan_arena_offsets ~cap:100 [ (64, 8, "single", None); (64, 8, "single", None) ]
  with
  | None -> p "planner: over-cap layout falls back" true
  | Some _ -> p "planner: over-cap layout falls back" false

(* Layer 1b: [Low_level.sink_zero_outs] on hand-built statement lists. Executed coverage of the sunk
   code is the train phase below: with the planner on, the whole-step routine runs with its
   zero-grads block sunk, and the loss trajectory must match the planner-off run bitwise. *)
let () =
  let module LL = Ir.Low_level in
  let tnode l = (NTDSL.init ~l ~prec:Ir.Ops.single ~o:[ 1 ] ~f:(fun _ -> 0.) ()).Tensor.value in
  let a = tnode "sink_a" and b = tnode "sink_b" and c = tnode "sink_c" in
  let idcs = [| Ir.Indexing.Fixed_idx 0 |] in
  let set t v = Ll_test.set t idcs v in
  let get t = Ll_test.get t idcs in
  let sunk lines = LL.flat_lines [ LL.sink_zero_outs (LL.unflat_lines lines) ] in
  (let result =
     sunk [ LL.Zero_out a; LL.Zero_out b; set c (LL.Constant 1.); set a (get c); set b (get a) ]
   in
   p "sink: zeros sink to their first access"
     (match result with
     | [
      LL.Set { tn = c1; _ };
      LL.Zero_out a1;
      LL.Set { tn = a2; _ };
      LL.Zero_out b1;
      LL.Set { tn = b2; _ };
     ] ->
         Tn.equal c1 c && Tn.equal a1 a && Tn.equal a2 a && Tn.equal b1 b && Tn.equal b2 b
     | _ -> false));
  (let result =
     sunk [ LL.Zero_out a; set c (LL.Constant 1.); LL.Workgroup_barrier; set a (get c) ]
   in
   p "sink: barrier blocks sinking"
     (match result with
     | [ LL.Set _; LL.Zero_out a1; LL.Workgroup_barrier; LL.Set _ ] -> Tn.equal a1 a
     | _ -> false));
  let result = sunk [ LL.Zero_out a; set c (LL.Constant 1.); set b (get c) ] in
  p "sink: never-reaccessed zero stays in place"
    (match result with LL.Zero_out a1 :: _ -> Tn.equal a1 a | _ -> false)

(* A chain of materialized unobservable intermediates: h1 dies once h2 is computed, so h1/h3 (and
   h2/h4) can share bytes under the planner. The links are matmuls, not elementwise maps, on
   purpose: an elementwise chain gets merged into ONE parallel kernel by the GPU backends' aligned
   cross-nest fusion, where interleaving threads make sharing genuinely unsafe (segment-granularity
   liveness correctly refuses) — a reduction chain cannot merge, so it fissions into per-link
   kernels and both the statement-granularity (CPU) and segment-granularity (GPU) planners find the
   same disjoint spans, keeping this output backend-stable. *)
let chain_phase ~label () =
  Utils.settings.fixed_state_for_init <- Some 7;
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let mem0 = Context.get_used_memory ctx in
  let dim = 2048 in
  let x =
    TDSL.init ~l:"x" ~prec:Ir.Ops.single ~o:[ dim ]
      ~f:(fun idcs -> (Float.of_int (idcs.(0) % 7) *. 0.5) -. 1.)
      ()
  in
  let a_mat =
    TDSL.init ~l:"a_mat" ~prec:Ir.Ops.single ~i:[ dim ] ~o:[ dim ]
      ~f:(fun idcs ->
        if idcs.(0) = idcs.(1) then 0.5
        else Ll_test.weighted ~weights:[| 1; 1 |] ~modulus:3 ~offset:0. ~stride:0.001 idcs)
      ()
  in
  let%op h1 = relu (a_mat * x) in
  let%op h2 = relu (a_mat * h1) in
  let%op h3 = relu (a_mat * h2) in
  let%op h4 = relu (a_mat * h3) in
  let%op out = h4 ++ "i=>0" in
  List.iter [ h1; h2; h3; h4 ] ~f:(fun t -> Train.set_materialized t.Tensor.value);
  Tn.set_observable out.Tensor.value;
  let ctx = Train.forward_once ctx out in
  let mem_delta = Context.get_used_memory ctx - mem0 in
  let read name (t : Tensor.t) =
    match Context.get_values ctx t.Tensor.value with
    | (_ : float array) -> printf "%s: read %s: ok\n" label name
    | exception Utils.User_error msg when String.is_substring msg ~substring:"buffer-aliased" ->
        printf "%s: read %s: raises buffer-aliased\n" label name;
        (* gh-ocannl-777: the query answers the same without attempting the read. *)
        pf "%s: %s is classified aliased up front" label name
          (match Context.observability ctx t.Tensor.value with
          | Unobservable_aliased [ v ] -> Tn.equal v t.Tensor.value
          | _ -> false)
  in
  read "h1" h1;
  read "h4" h4;
  read "out" out;
  let out_v = Context.get_values ctx out.Tensor.value in
  (out_v.(0), mem_delta)

(* A whole training step compiled as one routine: forward + zero-grads + backprop + SGD. Hidden
   activations (batch*d_hid floats, over the stack threshold) materialize because backprop revisits
   them; they and the SGD scratch are the aliasing candidates. Params and their grads (observation
   intent) keep dedicated buffers. *)
let train_phase () =
  Utils.settings.fixed_state_for_init <- Some 3;
  Tensor.unsafe_reinitialize ();
  (* Fan-scaled init: the default centered uniform diverges on a 4-layer 256-wide net. *)
  TDSL.default_param_init := NTDSL.xavier ~scale_sq:2.0 TDSL.O.uniform1;
  let ctx = Context.auto () in
  let mem0 = Context.get_used_memory ctx in
  let batch = 32 and d_in = 64 and d_hid = 256 in
  let xs =
    NTDSL.init ~l:"xs" ~prec:Ir.Ops.single ~b:[ batch ] ~o:[ d_in ]
      ~f:(fun idcs -> Float.sin (Float.of_int ((idcs.(0) * d_in) + idcs.(1))))
      ()
  in
  let ys =
    NTDSL.init ~l:"ys" ~prec:Ir.Ops.single ~b:[ batch ] ~o:[ 1 ]
      ~f:(fun idcs -> Float.cos (Float.of_int idcs.(0)))
      ()
  in
  (* Depth matters for the footprint assertion: in backprop the layer-k activation-gradient dies
     once the layer-(k-1) gradient is computed, so with 3+ hidden layers the gradient chain has
     liveness-disjoint links (whereas forward activations all stay live into the backward pass). *)
  let%op mlp x =
    { w4 }
    * relu
        ({ b3; o = [ d_hid ] }
        + { w3 }
          * relu ({ b2; o = [ d_hid ] } + ({ w2 } * relu ({ b1; o = [ d_hid ] } + ({ w1 } * x)))))
  in
  let%op err = mlp xs - ys in
  let%op scalar_loss = ((err *. err) ++ "...|... => |->0") /. !..batch in
  let update = Train.grad_update scalar_loss in
  let sgd = Train.sgd_update ~learning_rate:(TDSL.O.( !. ) 1e-6) scalar_loss in
  let ctx = Train.init_params ctx IDX.empty scalar_loss in
  let ctx, routine = Train.to_routine ctx IDX.empty (Asgns.sequence [ update; sgd ]) in
  let open Operation.At in
  let losses = ref [] in
  for _ = 1 to 6 do
    Train.run ctx routine;
    losses := (ctx, scalar_loss).@[0] :: !losses
  done;
  let losses = List.rev !losses in
  let mem_delta = Context.get_used_memory ctx - mem0 in
  (* Stderr only (not part of the compared stdout): concrete values for local debugging. *)
  eprintf "train losses: %s; step footprint: %d bytes\n%!"
    (String.concat ~sep:", " (List.map losses ~f:(Printf.sprintf "%.6f")))
    mem_delta;
  (losses, mem_delta)

let () =
  (* Section 1: planner off (the default). *)
  let out_off, chain_mem_off = chain_phase ~label:"gate-off" () in
  let losses_off, train_mem_off = train_phase () in
  (* Section 2: planner on. The config is re-read at each compile, so flipping the environment
     between sections works within one process. *)
  Unix.putenv "OCANNL_BUFFER_ALIASING" "true";
  let out_on, chain_mem_on = chain_phase ~label:"gate-on" () in
  let losses_on, train_mem_on = train_phase () in
  p "chain: result parity across buffer_aliasing" (Float.equal out_off out_on);
  p "chain: footprint reduced" (chain_mem_on < chain_mem_off);
  Verdict.pass_fail "train: loss trajectory parity across buffer_aliasing"
    (List.equal Float.equal losses_off losses_on);
  p "train: loss decreased" Float.(List.last_exn losses_off < List.hd_exn losses_off);
  (* Under the planner, [Low_level.sink_zero_outs] moves each gradient's Zero_out from the up-front
     zero-grads block to its first accumulation, so the backprop chain's live spans stagger: chain
     links two apart are liveness-disjoint and share bytes on both the statement-granularity (CPU)
     and segment-granularity (GPU) planners. *)
  p "train: footprint reduced" (train_mem_on < train_mem_off)

let () =
  let open Ll_test in
  let width = 16 in
  let mk = node_factory ~first_id:246100 ~dims:[| width |] () in
  let run_replay ?launch_offset ~name ~aliasing ~stored () =
    Unix.putenv "OCANNL_BUFFER_ALIASING" (Bool.to_string aliasing);
    let early = mk "early" and tap = mk "tap" and middle = mk "middle" in
    let late = mk "late" and last = mk "last" and out = mk "out" in
    Tn.set_observable tap;
    materialize out;
    let bindings, launch =
      match launch_offset with
      | None -> (Idx.Empty, None)
      | Some offset ->
          let k, bindings = Idx.get_static_symbol Idx.Empty in
          (bindings, Some (k, offset))
    in
    let write tn value =
      let i = sym () in
      loop_n i width (set_at tn (iter i) (value i))
    in
    let plus source bias i = add (get source [| iter i |]) (c bias) in
    let tap_value i =
      let value = plus early 10. i in
      match launch with None -> value | Some (k, _) -> add value (embed k.Idx.static_symbol)
    in
    let llc =
      List.reduce_exn ~f:seq
        [
          write early tick;
          write tap tap_value;
          write middle (plus tap 20.);
          write late (plus middle 40.);
          write last (plus late 80.);
          write out (plus last 160.);
        ]
    in
    let materialized = [ early; middle; late; last ] @ if stored then [ tap ] else [] in
    let lowered = optimize ~name ~materialized ~static_indices:(Idx.bound_symbols bindings) llc in
    (* A cut at each producer gives the GPU planner the same disjoint lifetimes as cc. *)
    let lowered_transform o =
      List.map (LL.flat_lines [ o.LL.llc ]) ~f:(fun llc -> { o with LL.llc })
    in
    let ctx, routine =
      Context.compile ~name ~prelowered:lowered ~lowered_transform (Context.auto ())
        Asgns.empty_comp bindings
    in
    Option.iter launch ~f:(fun (k, offset) -> Idx.find_exn routine.Context.bindings k := offset);
    (* The test config prefills fresh outputs, including those that share bytes. *)
    (Context.run ctx routine, early, tap, middle, last, out)
  in
  let ref_ctx, _, ref_tap, _, _, ref_out =
    run_replay ~name:"buffer_alias_replay_reference" ~aliasing:false ~stored:true ()
  in
  let ctx, early, tap, middle, last, out =
    run_replay ~name:"buffer_alias_replay" ~aliasing:true ~stored:false ()
  in
  let expected bias = Array.init width ~f:(fun i -> Float.of_int i +. bias) in
  let exact = Array.equal Float.equal in
  let ref_values = Context.get_values ref_ctx ref_tap in
  p "replay: the materialized reader matches the closed form" (exact ref_values (expected 11.));
  p "replay: the forward result matches its materialized run and the closed form"
    (let values = Context.get_values ctx out in
     exact values (expected 311.) && exact values (Context.get_values ref_ctx ref_out));
  let aliased ctx tn =
    match Context.observability ctx tn with
    | Unobservable_aliased [ v ] -> Tn.equal v tn
    | _ -> false
  in
  p "replay: unrelated intermediates still share storage" (aliased ctx middle && aliased ctx last);
  p "replay: the observable virtual result stays recomputable"
    (match Context.observability ctx tap with
    | Recomputable -> true
    | Unobservable_aliased [ root; leaf ] when Tn.equal root tap && Tn.equal leaf early ->
        eprintf "replay: Unobservable_aliased through the early buffer\n%!";
        false
    | status ->
        eprintf "replay: unexpected classification %s\n%!"
          (Option.value (Context.observability_tag status) ~default:"readable");
        false);
  p "replay: the host read matches the materialized reader"
    (match Context.get_values ctx tap with
    | values -> exact values ref_values
    | exception Utils.User_error msg ->
        eprintf "replay: %s\n%!" msg;
        false);
  let launch_ref, _, launch_tap, _, _, launch_ref_out =
    run_replay ~name:"buffer_alias_launch_reference" ~aliasing:false ~stored:false ~launch_offset:3
      ()
  in
  let launch_ctx, launch_early, _, _, _, launch_out =
    run_replay ~name:"buffer_alias_launch" ~aliasing:true ~stored:false ~launch_offset:3 ()
  in
  p "launch-bound replay: the virtual tap needs a launch index"
    (match Context.observability launch_ref launch_tap with
    | Launch_bound ([ root ], [ _ ]) -> Tn.equal root launch_tap
    | _ -> false);
  p "launch-bound replay: the early buffer can still alias" (aliased launch_ctx launch_early);
  p "launch-bound replay: the forward result matches the closed form with aliasing on and off"
    (exact (Context.get_values launch_ctx launch_out) (expected 314.)
    && exact (Context.get_values launch_ref launch_ref_out) (expected 314.))

(* The shared loop captures [tap] before its sibling [dep], so the stored tap retains a read of
   [dep]. The first loop already defines the dependency; the sibling store writes the same values. A
   late gather makes [dep] a materialized shield while leaving its stored component in the
   lineage. *)
let () =
  let open Ll_test in
  (* A late-refused dependency must be larger than the stack threshold to own a buffer. *)
  let width = 8192 in
  let mk = node_factory ~first_id:246200 ~dims:[| width |] () in
  let run_case ~name ~aliasing ~shield ~stored =
    Unix.putenv "OCANNL_BUFFER_ALIASING" (Bool.to_string aliasing);
    let early = mk "early" and dep = mk "dep" and tap = mk "tap" in
    let forward = mk "forward" and late = mk "late" and out = mk "out" in
    Tn.set_observable tap;
    materialize out;
    let at tn i = get tn [| iter i |] in
    let plus tn bias i = add (at tn i) (c bias) in
    let set tn value i = set_at tn (iter i) (value i) in
    let write tn value =
      let i = sym () in
      loop_n i width (set tn value i)
    in
    let forward_value i =
      let dep_value =
        if shield then gather ~tn:dep ~idcs:[| iter i |] ~dyn_axis:0 ~dyn_value:(embed i, iprec ())
        else at dep i
      in
      add (plus early 20. i) dep_value
    in
    let i = sym () in
    let llc =
      LL.unflat_lines
        [
          write early tick;
          write dep (plus early 5.);
          loop_n i width (seq (set tap (plus dep 10.) i) (set dep (plus early 5.) i));
          write forward forward_value;
          write late (plus forward 100.);
          write out (plus late 40.);
        ]
    in
    let materialized = [ early; forward; late ] @ if stored then [ dep; tap ] else [] in
    let lowered = optimize ~name ~materialized llc in
    let placements = lowered.LL.optimize_ctx.placements in
    let reads self source =
      p_exists
        (name ^ ": stored replay reads " ^ Tn.debug_name source)
        (Hashtbl.find_multi lowered.LL.optimize_ctx.computations self)
        ~f:(fun (_, code) -> Set.mem (LL.replay_facts ~placements ~self code).replay_reads source)
    in
    if not stored then (
      p (name ^ ": the tap is virtual") (known_virtual lowered tap);
      reads tap dep;
      if not shield then p (name ^ ": the dependency is virtual") (known_virtual lowered dep);
      reads dep early);
    let lowered_transform o =
      List.map (LL.flat_lines [ o.LL.llc ]) ~f:(fun llc -> { o with LL.llc })
    in
    let ctx, routine =
      Context.compile ~name ~prelowered:lowered ~lowered_transform (Context.auto ())
        Asgns.empty_comp Idx.Empty
    in
    (Context.run ctx routine, early, dep, tap, out)
  in
  let ref_ctx, _, _, ref_tap, ref_out =
    run_case ~name:"buffer_alias_components_reference" ~aliasing:false ~shield:false ~stored:true
  in
  let ref_tap = Context.get_values ref_ctx ref_tap
  and ref_out = Context.get_values ref_ctx ref_out in
  let exact = Array.equal Float.equal in
  let expected bias = Array.init width ~f:(fun i -> Float.of_int i +. bias) in
  p "components: the materialized taps and outputs match their closed forms"
    (exact ref_tap (expected 16.)
    && exact ref_out (Array.init width ~f:(fun i -> Float.of_int ((2 * i) + 167))));
  List.iter
    [
      ("buffer_alias_components", true, false);
      ("buffer_alias_components_no_alias", false, false);
      ("buffer_alias_shield", true, true);
      ("buffer_alias_shield_no_alias", false, true);
    ]
    ~f:(fun (name, aliasing, shield) ->
      let ctx, early, dep, tap, out = run_case ~name ~aliasing ~shield ~stored:false in
      p
        (name ^ ": the upstream buffer has the expected aliasing")
        (match Context.observability ctx early with
        | Unobservable_aliased [ v ] -> shield && aliasing && Tn.equal v early
        | Stored -> not (shield && aliasing)
        | _ -> false);
      if shield then
        p (name ^ ": the shield is stored")
          (match Context.observability ctx dep with Stored -> true | _ -> false);
      p
        (name ^ ": the forward output matches the materialized run")
        (exact (Context.get_values ctx out) ref_out);
      p
        (name ^ ": the virtual tap is recomputable")
        (match Context.observability ctx tap with
        | Recomputable -> true
        | status ->
            eprintf "%s: %s\n%!" name
              (Option.value (Context.explain_observability ctx tap status) ~default:"readable");
            false);
      p
        (name ^ ": the host replay matches the materialized tap")
        (match Context.get_values ctx tap with
        | values -> exact values ref_tap
        | exception Utils.User_error msg ->
            eprintf "%s: %s\n%!" name msg;
            false))
