open Base
module LA = Arrayjit.Lazy_array
module NTDSL = Operation.NTDSL
open Arrayjit

(** Reinitializes a backend selected via a global [backend] flag. *)
let fresh_backend ?(verbose = true) () =
  let open Arrayjit.Backends in
  let backend =
    match Utils.get_global_arg ~verbose ~arg_name:"backend" ~default:"gccjit" |> String.lowercase with
    | "gccjit" -> (module Gccjit_backend : Backend)
    | "cuda" -> (module Cuda_backend : Backend)
    | backend -> invalid_arg [%string "Train.fresh_backend: unknown backend %{backend}"]
  in
  reinitialize backend;
  backend

let is_param t =
  match t with
  | { Tensor.children = []; value = { literal = false; _ }; diff = Some _; _ } -> true
  | _ -> false

let params t =
  let rec loop accu { Tensor.subtensor = t; _ } =
    List.fold t.children ~init:(if is_param t then Set.add accu t else accu) ~f:loop
  in
  loop (Set.empty (module Tensor)) { subtensor = t; embedded = true }

let desc_label_suffix s =
  let pos = Option.value ~default:(-1) (String.rindex s '#') + 1 in
  String.sub s ~pos ~len:(String.length s - pos)

let grad_update l =
  match l.Tensor.diff with
  | Some diff ->
      let%cd init_grad = l.grad =: 1 in
      let label = desc_label_suffix l.value.label in
      Assignments.(
        Block_comment
          ( label ^ " grad update",
            sequential
              [
                Block_comment (label ^ " fwd", l.forward);
                Block_comment (label ^ " zero grads", diff.zero_grads);
                init_grad;
                Block_comment (label ^ " bprop", diff.backprop);
              ] ))
  | None -> raise @@ Tensor.Session_error ("Train.backprop: tensor is not differentiable", Some l)

(** See: {!https://github.com/tinygrad/tinygrad/blob/master/tinygrad/nn/optim.py}. *)
let sgd_one ?(lr = 0.001) ?(momentum = 0.0) ?(weight_decay = 0.0) ?(nesterov = false) p =
  if not @@ is_param p then raise @@ Tensor.Session_error ("Train.sgd_one: not a parameter", Some p);
  let pg = NTDSL.term ~label:(p.value.label ^ " sgd delta") () in
  let b = NTDSL.term ~label:(p.value.label ^ " sgd momentum") () in
  Assignments.Block_comment
    ( desc_label_suffix p.value.label ^ " param sgd step",
      [%cd
        pg =: p.grad + (!.weight_decay *. p);
        if Float.(momentum > 0.0) then (
          b =: (!.momentum *. b) + pg;
          if nesterov then pg =+ !.momentum *. b else pg =: b);
        p =- !.lr *. pg] )

let sgd_update ?lr ?momentum ?weight_decay ?nesterov t =
  let code =
    params t |> Set.to_list
    |> List.map ~f:(sgd_one ?lr ?momentum ?weight_decay ?nesterov)
    |> Assignments.sequential
  in
  Assignments.Block_comment (desc_label_suffix t.value.label ^ " sgd update", code)

let for_loop ~f bindings =
  let rec loop = function
    | [] -> f ()
    | ({ Indexing.static_range; static_symbol }, idx) :: more -> (
        match static_range with
        | None ->
            raise
            @@ Tensor.Session_error
                 ( [%string
                     "Train.for_loop: missing range for static symbol %{Indexing.symbol_ident static_symbol}"],
                   None )
        | Some range ->
            let old_idx = !idx in
            for i = 0 to range - 1 do
              idx := i;
              loop more
            done;
            idx := old_idx)
  in
  loop @@ Indexing.assoc_of_bindings bindings

let set_fully_on_host (a : LA.t) =
  if LA.is_true a.virtual_ then
    raise
    @@ Ndarray.User_error
         [%string "Train.set_fully_on_host: array #%{a.id#Int} %{a.label} is already virtual"];
  a.virtual_ <- Some (false, 27);
  if LA.is_true a.device_only then
    raise
    @@ Ndarray.User_error
         [%string "Train.set_fully_on_host: array #%{a.id#Int} %{a.label} is already device-only"];
  a.device_only <- Some (false, 28)

let everything_fully_on_host = Tensor.iter_embedded_arrays ~f:set_fully_on_host
