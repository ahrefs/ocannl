(* gh-ocannl-489 liveness-based buffer aliasing (the memory planner). Four layers:

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
  let set t v = LL.Set { tn = t; idcs; llsc = v; debug = "" } in
  let get t = LL.Get (t, idcs) in
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
        printf "%s: read %s: raises buffer-aliased\n" label name
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
