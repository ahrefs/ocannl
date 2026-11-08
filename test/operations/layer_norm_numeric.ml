(* Numeric regression test for Nn_blocks.layer_norm: checks the output and the input gradient for
   small fixed vectors against hand-computed double-precision LayerNorm oracles. Guards against the
   historical bug where [++] add-reduction results were used as if they were means (mean = sum(x),
   centered = (x - sum)/d instead of x - sum/d, variance never divided by d) — a shape-only test
   cannot catch that. gamma/beta initialize to 1/0, so the forward output must equal the plain
   normalization. *)

open! Base
open Ocannl.Nn_blocks.DSL_modules
module IDX = Ocannl.Train.IDX
module At = Ocannl_tensor.Operation.At

let batch = 2
let d = 4
let epsilon = 1e-5

(* Row 0 is deliberately asymmetric around its mean; for the buggy centering (x - sum)/d the result
   is not even proportional to the true one, so any per-row affine (gamma, beta) cannot mask the
   error. *)
let x_val = [| [| 1.; 2.; 3.; 4. |]; [| -1.; 0.; 2.; 7. |] |]

(* Upstream gradients: loss = sum (layer_norm x *. w). *)
let w_val = [| [| 0.5; -1.; 2.; 0.25 |]; [| 1.; 1.; -0.5; 3. |] |]
let mean row = Array.fold row ~init:0. ~f:( +. ) /. Float.of_int d

let sigma row =
  let m = mean row in
  let variance =
    Array.fold row ~init:0. ~f:(fun acc v -> acc +. ((v -. m) *. (v -. m))) /. Float.of_int d
  in
  Float.sqrt (variance +. epsilon)

let y_hat b i =
  let row = x_val.(b) in
  (row.(i) -. mean row) /. sigma row

(* Exact LayerNorm input gradient (gamma = 1): dx = (g - mean(g) - y_hat * mean(g * y_hat)) / sigma,
   with y_hat and sigma including epsilon. *)
let grad_oracle b i =
  let g = w_val.(b) in
  let mean_g = mean g in
  let mean_gy =
    Array.foldi g ~init:0. ~f:(fun j acc gj -> acc +. (gj *. y_hat b j)) /. Float.of_int d
  in
  (g.(i) -. mean_g -. (y_hat b i *. mean_gy)) /. sigma x_val.(b)

(* [tol_label] instead of printf %g: Windows printf renders 1e-05 as "1e-005". *)
let report ~what ~tol ~tol_label actual expected =
  Verdict.pf_close ~difference_only:true
    "layer_norm %s matches hand-computed LayerNorm (max abs err < %s)" what tol_label actual
    expected ~tol:(fun _ -> Stdlib.Float.pred tol)

let () =
  let ctx = Context.auto () in
  let x =
    NTDSL.init ~l:"x" ~prec:Ir.Ops.single ~b:[ batch ] ~o:[ d ]
      ~f:(function [| b; i |] -> x_val.(b).(i) | _ -> assert false)
      ()
  in
  let ln = Ocannl.Nn_blocks.layer_norm ~label:[ "ln" ] ~epsilon () in
  let y = ln x in
  let ctx = Ocannl.Train.forward_once ctx y in
  let actual_values = Array.create ~len:(batch * d) 0. in
  for b = 0 to batch - 1 do
    Stdio.printf "row %d:" b;
    for i = 0 to d - 1 do
      let actual = At.((ctx, y).@{[| b; i |]}) in
      Stdio.printf " %8.5f" actual;
      actual_values.((b * d) + i) <- actual
    done;
    Stdio.printf "\n"
  done;
  let expected = Array.init (batch * d) ~f:(fun i -> y_hat (i / d) (i % d)) in
  report ~what:"forward" ~tol:1e-5 ~tol_label:"1e-5" actual_values expected

let () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let x =
    Ocannl_tensor.Operation.init ~l:"x" ~prec:Ir.Ops.single ~b:[ batch ] ~o:[ d ]
      ~f:(function [| b; i |] -> x_val.(b).(i) | _ -> assert false)
      ~grad_spec:Tensor.Require_grad ()
  in
  let w =
    NTDSL.init ~l:"w" ~prec:Ir.Ops.single ~b:[ batch ] ~o:[ d ]
      ~f:(function [| b; i |] -> w_val.(b).(i) | _ -> assert false)
      ()
  in
  let ln = Ocannl.Nn_blocks.layer_norm ~label:[ "ln" ] ~epsilon () in
  let y = ln x in
  let%op loss = (y *. w) ++ " ... | ... => | ->0 " in
  Ocannl.Train.set_materialized (Option.value_exn ~here:[%here] x.Tensor.diff).grad;
  let update = Ocannl.Train.grad_update loss in
  let ctx = Ocannl.Train.init_params ctx IDX.empty loss in
  let ctx, routine = Context.compile ctx update IDX.empty in
  let ctx = Context.run ctx routine in
  let actual =
    Array.init (batch * d) ~f:(fun j ->
        let b = j / d and i = j % d in
        At.((ctx, x).@%{[| b; i |]}))
  in
  let expected = Array.init (batch * d) ~f:(fun j -> grad_oracle (j / d) (j % d)) in
  report ~what:"input gradient" ~tol:1e-4 ~tol_label:"1e-4" actual expected
