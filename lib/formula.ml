(** Reverse-mode autodiff. *)

type data = Ndarray.t

(** Stages of life:
    1. Constructing a model out of formulas.
    2. Call `finalize` once all the model(s) with parameter tying are constructed. This will
      force `fwd_code` and `bwd_code` all / recursively, and store the compiled code as
      `forward` and `backprop`.
    3.  *)
type t = {
  fwd_code: (unit -> unit) code Lazy.t;
  bwd_code: unit -> (unit -> unit) code Lazy.t;
  forward: (unit -> unit) ref;
  backprop: (unit -> unit) ref;
  value: data;
  grad: data;
  label: string;
  id: int;
}

let unique_id = ref 0

type computation_queue = {
  mutable front: (unit -> unit) list;
  mutable back: (unit -> unit) list;
}

type state = {
  forward_queue: computation_queue;
  backprop_queue: computation_queue;
  mutable unique_id: int;
  visited: (int, t) Hashtbl.t;
}

let global = {
  forward_queue = {front=[]; back=[]};
  backprop_queue = {front=[]; back=[]};
  unique_id = 0;
  visited = Hashtbl.create 256;
}

let add f1 f2 =
  let id = global.unique_id in
  global.unique_id <- global.unique_id + 1;
  let label = "(" ^ f1.label ^ " + " ^ f2.label ^ ")" in
  let dims = Ndarray.dims f1.value in
  let value = Ndarray.create dims in
  let fwd_code () =
     .< let p1 = ~.f1.fwd_code and p2 = ~.f2.fwd_code in
    fun () -> p1(); p2(); Ndarray.assign_binop value ( +. ) f1.value f2.value >. in
  let forward = !.fwd_code in
  let grad = Ndarray.create dims in
  let bwd_code = .< let p1 = ~.f1.fwd_code and p2 = ~.f2.fwd_code in
  fun () ->
    Ndarray.assign_binop f1.grad ( +. ) f1.grad grad;
    Ndarray.assign_binop f2.grad ( +. ) f2.grad grad;
    (* FIXME: this does not work with non-tree DAGs, when prefix != topological sort. *)
    p1(); p2() >. in
  let backprop = !.bwd_code in
  {fwd_code; bwd_code; forward; backprop; value; grad; label; id}

