open Base
open Arrayjit
module NTDSL = Operation.NTDSL

module type Backend_with_context = sig
  include Backends.Backend

  val active_context : context
end

(** Reinitializes a backend selected via a global [backend] flag, and creates an initial context on
    the indicated device. *)
let fresh_backend ?(verbose = true) ?(reinit = true) ?(on_device_num = 0) () =
  let open Backends in
  let backend =
    match Utils.get_global_arg ~verbose ~arg_name:"backend" ~default:"gccjit" |> String.lowercase with
    | "gccjit" -> (module Gccjit_backend : Backend)
    | "cuda" -> (module Cuda_backend : Backend)
    | backend -> invalid_arg [%string "Train.fresh_backend: unknown backend %{backend}"]
  in
  if reinit then reinitialize backend;
  let module Backend = (val backend) in
  (module struct
    include Backend

    let active_context = Backend.init @@ Backend.get_device ~ordinal:on_device_num
  end : Backend_with_context)

let is_param t =
  match t with
  | { Tensor.children = []; value = { literal = false; _ }; diff = Some _; _ } -> true
  | _ -> false

let params t =
  let rec loop accu { Tensor.subtensor = t; _ } =
    List.fold t.children ~init:(if is_param t then Set.add accu t else accu) ~f:loop
  in
  loop (Set.empty (module Tensor)) { subtensor = t; embedded = true }

let grad_update l =
  match l.Tensor.diff with
  | Some diff ->
      let%cd init_grad = l.grad =: 1 in
      Assignments.(
        Block_comment
          (l.value.label ^ " grad update", sequential [ l.forward; diff.zero_grads; init_grad; diff.backprop ]))
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
  let code =
    params t |> Set.to_list
    |> List.map ~f:(sgd_one ?lr ?momentum ?weight_decay ?nesterov)
    |> Assignments.sequential
  in
  Assignments.(Block_comment (t.value.label ^ " sgd update", code))

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

(** A small helper that automates [Train.grad_update], [Backend.jit] and [Train.for_loop]. *)
let jit_and_run (type context) (backend : (module Backend_with_context with type context = context)) bindings
    t : context =
  let module Backend = (val backend) in
  let code = grad_update t in
  let jitted = Backend.jit Backend.active_context bindings code in
  for_loop ~f:jitted.run jitted.bindings;
  jitted.context
