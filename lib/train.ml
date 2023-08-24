open Base

let is_params t =
  match t with
  | { Tensor.children = []; value = { literal = false; array = (lazy (Some _)); _ }; diff = Some _; _ } ->
      true
  | _ -> false

let params t =
  let rec loop accu { Tensor.subtensor = t; _ } =
    List.fold t.children ~init:(if is_params t then Set.add accu t else accu) ~f:loop
  in
  loop (Set.empty (module Tensor)) { subtensor = t; embedded = true }
