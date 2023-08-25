open Base
module NTDSL = Operation.NTDSL

let is_param t =
  match t with
  | { Tensor.children = []; value = { literal = false; _ }; diff = Some _; _ } -> true
  | _ -> false

let params t =
  let rec loop accu { Tensor.subtensor = t; _ } =
    List.fold t.children ~init:(if is_param t then Set.add accu t else accu) ~f:loop
  in
  loop (Set.empty (module Tensor)) { subtensor = t; embedded = true }

(** See: {!https://github.com/tinygrad/tinygrad/blob/master/tinygrad/nn/optim.py}. *)
let sgd_one ?(lr = 0.001) ?(momentum = 0.0) ?(weight_decay = 0.0) ?(nesterov = false) p =
  if not @@ is_param p then raise @@ Tensor.Session_error ("Train.sgd_one: not a parameter", Some p);
  let module TDSL = NTDSL in
  let pg = NTDSL.term ~label:(p.value.label ^ " sgd delta") () in
  let b = NTDSL.term ~label:(p.value.label ^ " sgd momentum") () in
  [%cd
    pg =: p.grad + (!.weight_decay *. p);
    if Float.(momentum > 0.0) then (
      b =: (!.momentum *. b) + pg;
      if nesterov then pg =+ !.momentum *. b else pg =: b);
    p =- !.lr *. pg]
