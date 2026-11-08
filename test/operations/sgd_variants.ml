(* Executed oracle for SGD's option matrix and every core optimizer construction entry point
   (gh-ocannl-772, gh-ocannl-811).

   Every other training test runs sgd on its defaults, so [momentum], [nesterov], [grad_scale] and a
   gate that actually CLOSES are reachable only from here. This test drives each of them for several
   steps and compares the parameter against a host simulation of the documented update rule — a
   multi-step trajectory rather than a structural check on the emitted tree, because the properties
   at stake are about values carried BETWEEN invocations of the routine: the momentum buffer must
   survive the step that wrote it (it is optimizer state, not a scratch temporary), and on a gated-
   shut step both the parameter and the buffer must keep their previous values EXACTLY — selection,
   not multiplication, since the gate exists to skip inf/nan gradients and [0 * inf] is [nan]. The
   "gate closed mid-run" case is where those two meet: step 4 must resume from step 1's buffer.

   The entry-point cases send the same non-default [momentum] through [Train.sgd_one],
   [Train.sgd_update], [Mixed_prec.scaled_sgd_update] and [Mixed_prec.gated_scaled_update]. Each
   runs both zero- and nonzero-momentum trajectories on the cc backend, checks the latter against
   the host rule, AND requires the two device results to differ. The last condition is the
   forwarding oracle: a wrapper that accepts but drops the option still takes ordinary SGD steps,
   but it cannot pass by matching its own default.

   [Parallel.data_parallel], whose execution harness lives in [test/training/data_parallel.ml],
   separately runs its public momentum and weight-decay forwarders against the same kind of host
   recurrence and nondefault-vs-default discriminator.

   The oracle discriminates: perturbing the momentum the device is given (0.9 -> 0.45) fails all
   four momentum-bearing cases with errors of 0.09 to 1.3, against a 1e-4 tolerance. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module IDX = Train.IDX
module MP = Mixed_prec

let dims = 3

(* d(loss)/d(p_i) = coeffs_i on every step: [loss] is linear in [p], so the gradient is a constant
   the oracle knows exactly, and the trajectory tests the optimizer rather than backprop. *)
let coeffs = [| 0.5; -1.5; 2.0 |]
let p_init = [| 1.0; -2.0; 0.25 |]
let lr = 0.1
let steps = 4

type entry_point = Sgd_one | Sgd_update | Scaled_sgd_update | Gated_scaled_update

let entry_point_name = function
  | Sgd_one -> "Train.sgd_one"
  | Sgd_update -> "Train.sgd_update"
  | Scaled_sgd_update -> "Mixed_prec.scaled_sgd_update"
  | Gated_scaled_update -> "Mixed_prec.gated_scaled_update"

(** Host simulation of {!Train.sgd_one}. [gate_of_step] is [None] for the ungated arm. *)
let oracle ~momentum ~nesterov ~weight_decay ~grad_scale ~gate_of_step =
  let p = Array.copy p_init in
  let buf = Array.create ~len:dims 0.0 in
  for step = 1 to steps do
    let open_step = match gate_of_step with None -> true | Some f -> f step in
    for i = 0 to dims - 1 do
      let g = match grad_scale with None -> coeffs.(i) | Some s -> coeffs.(i) *. s in
      let delta = ref (g +. (weight_decay *. p.(i))) in
      if Float.(momentum > 0.0) then (
        let advanced = (momentum *. buf.(i)) +. !delta in
        if open_step then buf.(i) <- advanced;
        delta := if nesterov then !delta +. (momentum *. buf.(i)) else buf.(i));
      if not open_step then delta := 0.0;
      p.(i) <- p.(i) -. (lr *. !delta)
    done
  done;
  p

let device ~entry_point ~momentum ~nesterov ~weight_decay ~grad_scale ~gate_of_step =
  Tensor.unsafe_reinitialize ();
  let p = TDSL.param ~values:p_init "p" ~output_dims:[ dims ] () in
  let c =
    NTDSL.init ~l:"c" ~prec:Ir.Ops.single ~o:[ dims ]
      ~f:(function [| i |] -> coeffs.(i) | _ -> assert false)
      ()
  in
  let%op loss = (p *. c) ++ "...|... => |->0" in
  let%op learning_rate = 0.1 in
  Train.set_materialized p.Tensor.value;
  Train.set_materialized learning_rate.Tensor.value;
  let gate = Option.map gate_of_step ~f:(fun _ -> Train.host_scalar ~l:"gate" 1.0) in
  let scale = Option.map grad_scale ~f:(Train.host_scalar ~l:"grad_scale") in
  let comp =
    match entry_point with
    | Sgd_one ->
        let update = Train.grad_update loss in
        let sgd =
          Train.sgd_one ~learning_rate ~momentum ~weight_decay ~nesterov ?grad_scale:scale
            ?update_gate:gate p
        in
        Ir.Assignments.sequence [ update; sgd ]
    | Sgd_update ->
        let update = Train.grad_update loss in
        let sgd =
          Train.sgd_update ~learning_rate ~momentum ~weight_decay ~nesterov ?grad_scale:scale
            ?update_gate:gate loss
        in
        Ir.Assignments.sequence [ update; sgd ]
    | Scaled_sgd_update ->
        let scaler = MP.Loss_scaler.create ~init_scale:8. ~growth_interval:100 () in
        let _checksum, update = MP.scaled_grad_update scaler loss in
        let sgd =
          MP.scaled_sgd_update scaler ~learning_rate ~momentum ~weight_decay ~nesterov loss
        in
        Ir.Assignments.sequence [ update; sgd ]
    | Gated_scaled_update ->
        let scaler = MP.Loss_scaler.create ~init_scale:8. ~growth_interval:100 () in
        snd (MP.gated_scaled_update scaler ~learning_rate ~momentum ~weight_decay ~nesterov loss)
  in
  let ctx = Train.init_params (Context.cpu ()) IDX.empty loss in
  let ctx, routine = Train.to_routine ctx IDX.empty comp in
  let ctx = ref ctx in
  for step = 1 to steps do
    (match (gate, gate_of_step) with
    | Some g, Some f ->
        let v = if f step then 1.0 else 0.0 in
        ctx := Context.set_values !ctx g.Tensor.value [| v |]
    | _ -> ());
    ctx := Context.run !ctx routine
  done;
  let open Operation.At in
  Array.init dims ~f:(fun i -> (!ctx, p).@[i])

let max_abs_diff a b =
  Array.foldi a ~init:0.0 ~f:(fun i acc v -> Float.max acc (Float.abs (v -. b.(i))))

let case label ~momentum ?(nesterov = false) ?(weight_decay = 0.0) ?grad_scale ?gate_of_step () =
  let expected = oracle ~momentum ~nesterov ~weight_decay ~grad_scale ~gate_of_step in
  let actual =
    device ~entry_point:Sgd_one ~momentum ~nesterov ~weight_decay ~grad_scale ~gate_of_step
  in
  let err = max_abs_diff actual expected in
  (* Device-produced floats: the digits stay off stdout, the claim about them goes on it. *)
  Stdio.eprintf "%s (not part of the golden): expected %s; got %s; max abs err %.3e\n%!" label
    (String.concat ~sep:", " (Array.to_list (Array.map expected ~f:(Printf.sprintf "%.6f"))))
    (String.concat ~sep:", " (Array.to_list (Array.map actual ~f:(Printf.sprintf "%.6f"))))
    err;
  (* Two-sided: the parameters must both match the oracle and have actually moved, so an optimizer
     that emitted nothing at all cannot pass by leaving [p] where a wrong oracle also put it. *)
  let moved = Array.existsi actual ~f:(fun i v -> Float.(abs (v -. p_init.(i)) > 1e-4)) in
  Verdict.pass_fail_close ~difference_only:true (label ^ ": matches the host oracle")
    actual expected ~tol:(fun _ -> Stdlib.Float.pred 1e-4);
  Verdict.p (label ^ ": the step moved the parameter") moved

let entry_point_case entry_point =
  let label = entry_point_name entry_point ^ " forwards momentum" in
  let actual =
    device ~entry_point ~momentum:0.9 ~nesterov:false ~weight_decay:0.0 ~grad_scale:None
      ~gate_of_step:None
  in
  let baseline =
    device ~entry_point ~momentum:0.0 ~nesterov:false ~weight_decay:0.0 ~grad_scale:None
      ~gate_of_step:None
  in
  let expected =
    oracle ~momentum:0.9 ~nesterov:false ~weight_decay:0.0 ~grad_scale:None ~gate_of_step:None
  in
  let oracle_err = max_abs_diff actual expected in
  let effect_size = max_abs_diff actual baseline in
  Stdio.eprintf "%s (not part of the golden): oracle err %.3e; option effect %.3e\n%!" label
    oracle_err effect_size;
  Verdict.pass_fail_close ~difference_only:true (label ^ ": matches the host oracle")
    actual expected ~tol:(fun _ -> Stdlib.Float.pred 1e-4);
  Verdict.pass_fail
    (label ^ ": non-default changes the device result")
    Float.(effect_size > 1e-3)
    ~detail:(fun () -> Printf.sprintf "max abs difference %.3e" effect_size)

let () =
  case "plain" ~momentum:0.0 ();
  case "weight decay" ~momentum:0.0 ~weight_decay:0.05 ();
  case "momentum" ~momentum:0.9 ();
  case "momentum + nesterov + weight decay" ~momentum:0.9 ~nesterov:true ~weight_decay:0.05 ();
  case "grad scale + momentum" ~momentum:0.9 ~grad_scale:2.0 ();
  case "gate open throughout" ~momentum:0.0 ~gate_of_step:(fun _ -> true) ();
  (* The gate closes on steps 2 and 3: both the parameter and the momentum buffer must be held, so
     step 4 resumes from step 1's buffer rather than from an advanced one. *)
  case "gate closed from the first step, with momentum" ~momentum:0.9
    ~gate_of_step:(fun step -> step >= 2)
    ();
  case "gate closed mid-run, with momentum" ~momentum:0.9
    ~gate_of_step:(fun step -> step = 1 || step = 4)
    ();
  List.iter [ Sgd_one; Sgd_update; Scaled_sgd_update; Gated_scaled_update ] ~f:entry_point_case
