open Base
module LA = Arrayjit.Lazy_array
module NTDSL = Operation.NTDSL
module Asgns = Arrayjit.Assignments
module Idx = Arrayjit.Indexing
module type Backend_type = Arrayjit.Backends.Backend

(** Reinitializes a backend selected via a global [backend] flag. *)
let fresh_backend ?backend_name ?(verbose = true) () =
  let open Arrayjit.Backends in
  let backend =
    match
      Option.value_or_thunk backend_name ~default:(fun () ->
          Arrayjit.Utils.get_global_arg ~verbose ~arg_name:"backend" ~default:"gccjit")
      |> String.lowercase
    with
    | "gccjit" -> (module Gccjit_backend : Backend)
    | "cuda" -> (module Cuda_backend : Backend)
    | backend -> invalid_arg [%string "Train.fresh_backend: unknown backend %{backend}"]
  in
  reinitialize backend;
  backend

let literal_heuristic (a : LA.t) =
  try
    ignore (Float.of_string (List.hd_exn a.label) : float);
    true
  with _ -> false

let is_param t =
  match t with { Tensor.children = []; diff = Some _; _ } -> not @@ literal_heuristic t.value | _ -> false

let params t =
  let rec loop accu { Tensor.subtensor = t; _ } =
    List.fold t.children ~init:(if is_param t then Set.add accu t else accu) ~f:loop
  in
  loop (Set.empty (module Tensor)) { subtensor = t; embedded = true }

let set_on_host (a : LA.t) =
  if LA.is_true a.virtual_ then
    raise
    @@ Arrayjit.Ndarray.User_error
         [%string "Train.set_on_host: array #%{a.id#Int} %{LA.label a} is already virtual"];
  if Option.is_none a.virtual_ then a.virtual_ <- Some (false, 27);
  if LA.is_true a.device_only then
    raise
    @@ Arrayjit.Ndarray.User_error
         [%string "Train.set_on_host: array #%{a.id#Int} %{LA.label a} is already device-only"];
  a.device_only <- Some (false, 28)

(** Sets the tensor's value as "fully on host",
    returns the tensor's forward code with a label-derived comment. *)
let forward t =
  set_on_host t.Tensor.value;
  let label = Option.value ~default:"tensor" @@ List.last t.Tensor.value.label in
  Asgns.Block_comment (label ^ " fwd", t.forward)

(** Sets the tensor's value as "fully on host", returns the tensor's forward, zeroing gradients, and
    backprop code wrapped with label-derived comments. *)
let grad_update l =
  set_on_host l.Tensor.value;
  match l.Tensor.diff with
  | Some diff ->
      let%cd init_grad = l.grad =: 1 in
      let label = Option.value ~default:"tensor" @@ List.last l.value.label in
      Asgns.(
        Block_comment
          ( label ^ " gradient update",
            sequential
              [
                Block_comment (label ^ " fwd", l.forward);
                Block_comment (label ^ " zero grads", diff.zero_grads);
                init_grad;
                Block_comment (label ^ " bprop", diff.backprop);
              ] ))
  | None -> raise @@ Tensor.Session_error ("Train.backprop: tensor is not differentiable", Some l)

let label_suffix label = Option.value ~default:"unknown" @@ List.last label

(** See: {!https://github.com/tinygrad/tinygrad/blob/master/tinygrad/nn/optim.py}. *)
let sgd_one ~learning_rate ?(momentum = 0.0) ?(weight_decay = 0.0) ?(nesterov = false) p =
  if not @@ is_param p then raise @@ Tensor.Session_error ("Train.sgd_one: not a parameter", Some p);
  let pg = NTDSL.term ~label:("sgd_delta" :: p.value.label) () in
  let b = NTDSL.term ~label:("sgd_momentum" :: p.value.label) () in
  Asgns.Block_comment
    ( label_suffix p.value.label ^ " param sgd step",
      [%cd
        pg =: p.grad + (!.weight_decay *. p);
        if Float.(momentum > 0.0) then (
          b =: (!.momentum *. b) + pg;
          if nesterov then pg =+ !.momentum *. b else pg =: b);
        p =- learning_rate *. pg] )

let sgd_update ~learning_rate ?momentum ?weight_decay ?nesterov t =
  let code =
    params t |> Set.to_list
    |> List.map ~f:(sgd_one ~learning_rate ?momentum ?weight_decay ?nesterov)
    |> Asgns.sequential
  in
  Asgns.Block_comment (label_suffix t.value.label ^ " sgd update", code)

let for_loop ~f bindings =
  let rec loop = function
    | [] -> f ()
    | ({ Idx.static_range; static_symbol }, idx) :: more -> (
        match static_range with
        | None ->
            raise
            @@ Tensor.Session_error
                 ( [%string
                     "Train.for_loop: missing range for static symbol %{Idx.symbol_ident static_symbol}"],
                   None )
        | Some range ->
            let old_idx = !idx in
            for i = 0 to range - 1 do
              idx := i;
              loop more
            done;
            idx := old_idx)
  in
  loop @@ Idx.assoc_of_bindings bindings

let set_virtual (a : LA.t) =
  if LA.is_false a.virtual_ then
    raise
    @@ Arrayjit.Ndarray.User_error
         [%string "Train.set_virtual: array #%{a.id#Int} %{LA.label a} is already non-virtual"];
  if Option.is_none a.virtual_ then a.virtual_ <- Some (true, 29)

let every_non_literal_on_host =
  Tensor.iter_embedded_arrays ~f:(fun a -> if not @@ literal_heuristic a then set_on_host a)

let all_host_to_device ?(verbose = false) (type context)
    (module Backend : Backend_type with type context = context) (context : context) =
  Tensor.iter_embedded_arrays ~f:(fun a ->
      let b = Backend.from_host context a in
      if verbose && b then
        Stdio.printf "Train.all_device_to_host: copied array %s (%s) from host to device %d.\n%!" (LA.name a)
          (LA.label a)
          (Backend.get_ctx_device context |> Backend.to_ordinal))

let all_device_to_host ?(verbose = false) (type context)
    (module Backend : Backend_type with type context = context) (context : context) =
  Tensor.iter_embedded_arrays ~f:(fun a ->
      let b = Backend.to_host context a in
      if verbose && b then
        Stdio.printf "Train.all_device_to_host: copied array %s (%s) from device %d to host.\n%!" (LA.name a)
          (LA.label a)
          (Backend.get_ctx_device context |> Backend.to_ordinal))

(* Executes the jitted code and copies arrays embedded in the given tenosor from and to host,
   synchronizes before copying to host. If [looping] is provided, loops over bindings and executes
   the given function inside the loop after a run. *)
let sync_run ?verbose ?looping (type context) (module Backend : Backend_type with type context = context)
    (jitted : Backend.jitted) t =
  all_host_to_device ?verbose (module Backend) jitted.context t;
  (match looping with
  | None -> jitted.run ()
  | Some then_ ->
      let f () =
        jitted.run ();
        then_ ()
      in
      for_loop ~f jitted.bindings);
  Backend.await @@ Backend.get_ctx_device jitted.context;
  all_device_to_host ?verbose (module Backend) jitted.context t
