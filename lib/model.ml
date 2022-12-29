(** Losses and the training loop. *)
open Base

type params = (Formula.t, Formula.comparator_witness) Set.t

type content = {
  input: Formula.t;
  output: Formula.t;
  loss: Formula.t;
  params: params;
}

type loss_fun = params -> output:Formula.t -> target:Formula.t -> Formula.t
type t = input:Formula.t -> target:Formula.t -> content

let make ?(clear_session=true) network (loss_fun:loss_fun): t =
  fun ~input ~target ->
  let nn = network() in
  let output = Network.O.(nn @@ input) in
  let loss = loss_fun nn.params ~output ~target in
  (* Reset the session. *)
  if clear_session then (
    Formula.first_session_id := Node.global.unique_id;
    if (Map.existsi !Formula.global_roots ~f:(fun ~key ~data:_ -> key <> loss.node_id)) then (
      let _, other_root = Map.min_elt_exn @@ Map.filteri !Formula.global_roots
          ~f:(fun ~key ~data:_ -> key <> loss.node_id) in
          raise @@ Formula.Session_error (
            "Model.make expects the loss to be the only global root", Some other_root.formula));
    Formula.global_roots := Map.empty (module Int)
  );
  { input; output; loss; params=nn.params }

let hinge_loss ~output:y ~target:y' = Network.O.(!/(!.1.0 - y * y'))

let sum_over_params (params: params) ~f = Set.sum (module Operation.Summable) params ~f

let l2_reg_loss ~alpha (loss:loss_fun): loss_fun =
    fun params ~output ~target ->
     Network.O.(!.alpha * sum_over_params params ~f:(fun p -> p * p) + loss params ~output ~target)


(* 
~/ocannl$ dune utop

open Base
#load "_build/default/lib/ocannl.cma"
open Ocannl
module F = Formula
let d = [|3; 3|]
let o = [|3; 1|]
let res_mlp3 = let open Network in O.(
    nonlinear ~w:(!~"w1" d) ~b:(!~"b1" d) %+> nonlinear ~w:(!~"w2" d) ~b:(!~"b2" d) %+>
    linear ~w:(!~"w3" o) ~b:(!~"b3" o))
let loss = Network.O.(l2_reg_loss ~alpha:1e-4 res_mlp3 + (Train.hinge_loss res_mlp3 !.3.0 !.7.0))
let () = Stdio.print_endline @@ fst @@ F.sprint loss.toplevel_forward
let () = Stdio.print_endline @@ fst @@ F.sprint loss.toplevel_backprop
*)