open Base
open Arrayjit
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

let update_loss l =
  match l.Tensor.diff with
  | Some diff ->
      let%cd init_grad = l.grad =: 1 in
      High_level.sequential [ l.forward; diff.zero_grads; init_grad; diff.backprop ]
  | None -> raise @@ Tensor.Session_error ("Train.backprop: tensor is not differentiable", Some l)

(** See: {!https://github.com/tinygrad/tinygrad/blob/master/tinygrad/nn/optim.py}. *)
let sgd_one ?(lr = 0.001) ?(momentum = 0.0) ?(weight_decay = 0.0) ?(nesterov = false) p =
  if not @@ is_param p then raise @@ Tensor.Session_error ("Train.sgd_one: not a parameter", Some p);
  let pg = NTDSL.term ~label:(p.value.label ^ " sgd delta") () in
  let b = NTDSL.term ~label:(p.value.label ^ " sgd momentum") () in
  [%cd
    pg =: p.grad + (!.weight_decay *. p);
    if Float.(momentum > 0.0) then (
      b =: (!.momentum *. b) + pg;
      if nesterov then pg =+ !.momentum *. b else pg =: b);
    p =- !.lr *. pg]

let sgd_update ?lr ?momentum ?weight_decay ?nesterov t =
  params t |> Set.to_list
  |> List.map ~f:(sgd_one ?lr ?momentum ?weight_decay ?nesterov)
  |> High_level.sequential

(* FIXME: figure out handling backend "session contexts". *)

let run_on_cpu ?(name = "main") ?verbose code =
  High_level.compile_proc ~name ?verbose code |> Exec_as_gccjit.jit ~name ?verbose

let run_on_gpu ?(name = "main") ?verbose code =
  High_level.compile_proc ~name ?verbose code |> Exec_as_cuda.jit ~name ?verbose
