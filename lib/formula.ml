(** Reverse-mode autodiff. *)
open Base

type data = Ndarray.t

(** Stages of life:
    1. Constructing a model out of formulas.
    2. Call `finalize` once all the model(s) with parameter tying are constructed. This will
      force `fwd_code` and `bwd_code` all / recursively, and store the compiled code as
      `forward` and `backprop`.
    3.  *)
type final = {
  fwd_code: (unit -> unit) Codelib.code Lazy.t;
  bwd_code: (unit -> unit) Codelib.code Lazy.t;
  forward: (unit -> unit) ref;
  backprop: (unit -> unit) ref;
  value: data;
  grad: data;
  label: string;
  id: int;
}

type computation_queue = {
  mutable front: (unit -> unit) list;
  mutable back: (unit -> unit) list;
}

type state = {
  forward_queue: computation_queue;
  backprop_queue: computation_queue;
  mutable unique_id: int;
  visited: (int, final) Hashtbl.t;
}

let global = {
  forward_queue = {front=[]; back=[]};
  backprop_queue = {front=[]; back=[]};
  unique_id = 0;
  visited = Hashtbl.create (module Int);
}

let noop_final = {
  fwd_code = Lazy.from_val ( .< fun () -> () >. );
  bwd_code = Lazy.from_val ( .< fun () -> () >. );
  forward = ref (fun () -> ());
  backprop = ref (fun () -> ());
  value = Ndarray.empty;
  grad = Ndarray.empty;
  label = "noop";
  id = let id = global.unique_id in global.unique_id <- global.unique_id + 1; id
}

type initial =
| Add of initial list
| Mul of initial * initial
| Relu of initial
| Param of string * data
| Custom of final

let rec flatten ?(aux=[]) = function
| Add (Add elems1::elems2) -> flatten ~aux (Add (elems1 @ elems2))
| Add (el::elems) -> flatten ~aux:(el::aux) (Add elems)
| Add [] -> Add (List.rev_map ~f:(flatten ~aux:[]) aux)
| Mul (i1, i2) -> assert (List.is_empty aux); Mul (flatten ~aux:[] i1, flatten ~aux:[] i2)
| Relu i -> assert (List.is_empty aux); Relu (flatten ~aux:[] i)
| (Param _ | Custom _) as c -> c

let add = function
| [] -> noop_final
| elem::_ as elems ->
  let id = global.unique_id in
  global.unique_id <- global.unique_id + 1;
  let label = "(" ^ String.concat ~sep:" + " (List.map ~f:(fun e -> e.label) elems) ^ ")" in
  let dims = Ndarray.dims elem.value in
  let value = Ndarray.create dims in
  let grad = Ndarray.create dims in
  (* TODO: FIXME: NOT IMPLEMENTED *)
  (* let fwd_code =
    lazy ( .< let p1 = .~ (Lazy.force f1.fwd_code) and p2 = .~ (Lazy.force f2.fwd_code) in
      fun () -> (* p1(); p2();*) Ndarray.assign_binop value ( +. ) f1.value f2.value >. ) in
  let forward = ref (fun () -> ()) in
  let bwd_code = ( .< let p1 = .~f1.fwd_code and p2 = .~f2.fwd_code in
  fun () ->
    Ndarray.assign_binop f1.grad ( +. ) f1.grad grad;
    Ndarray.assign_binop f2.grad ( +. ) f2.grad grad;
    (* FIXME: this does not work with non-tree DAGs, when prefix != topological sort. *)
    p1(); p2() >. ) in
  let backprop = !.bwd_code in
   *)
  {fwd_code=noop_final.fwd_code; bwd_code=noop_final.bwd_code;
   forward=noop_final.forward; backprop=noop_final.backprop; value; grad; label; id}

let mul f1 f2 =
  (* TODO: FIXME: NOT IMPLEMENTED *)
  ignore f1;
  ignore f2;
  noop_final

let relu f =
  (* TODO: FIXME: NOT IMPLEMENTED *)
  ignore f;
  noop_final

let param label value =
  let id = global.unique_id in
  global.unique_id <- global.unique_id + 1;
  let dims = Ndarray.dims value in
  let grad = Ndarray.create_ones dims in
  (* TODO: FIXME: NOT IMPLEMENTED *)
  {fwd_code=noop_final.fwd_code; bwd_code=noop_final.bwd_code;
   forward=noop_final.forward; backprop=noop_final.backprop; value; grad; label; id}

let rec compile source =
  compile_aux (flatten source)

and compile_aux = function
| Add elems -> add (List.map ~f:compile_aux elems)
| Mul (e1, e2) -> mul (compile_aux e1) (compile_aux e2)
| Relu e -> relu (compile_aux e)
| Param (n, d) -> param n d
| Custom f -> f
