(** `Node`: the computation type, global state and utils which the `Formula` staged code uses. *)
open Base

module A = Bigarray.Genarray
type elt = Bigarray.float32_elt
type data = (float, elt, Bigarray.c_layout) A.t

let dims (arr: data) = A.dims arr

(** Initializes or resets a tensor by filling in the corresponding numbers, at the appropriate precision. *)
type init_op =
  [ `Unspecified
  (** Uninitialized. On reset, values may remain unchanged, but are not guaranteed to. *)
  | `ConstantOfValue of float
  (** Puts the value in all cells. *)
  | `FixedConstant of float array
  (** Fills in the numbers where the rightmost axis is contiguous. *)
  | `StandardUniform
  (** Draws the values from U(0,1). *)
  | `StandardGaussian
  (** Draws the values from N(0,1). *)
  ]

 let create_array dims: init_op -> data = function
   | `Unspecified -> A.create Bigarray.Float32 Bigarray.C_layout dims
   | `ConstantOfValue c -> A.init Bigarray.Float32 Bigarray.C_layout dims (fun _ -> c)
   | `FixedConstant cs ->
     A.init Bigarray.Float32 Bigarray.C_layout dims
       (fun indcs -> cs.(Array.foldi indcs ~init:0 ~f:(fun d pos i -> dims.(d) * pos + i)))
   | `StandardUniform ->
     A.init Bigarray.Float32 Bigarray.C_layout dims (fun _ -> Random.float_range 0.0 1.0)
   | `StandardGaussian ->
     (* FIXME: *) failwith "NOT IMPLEMENTED YET"

 let reset_array arr (reset_op: init_op) =
  let _dims = A.dims arr in
   match reset_op with
   | `Unspecified -> ()
   | `ConstantOfValue c -> A.fill arr c
   | `FixedConstant _cs ->
     (* FIXME: *) failwith "NOT IMPLEMENTED YET"
     | `StandardUniform ->
     (* FIXME: *) failwith "NOT IMPLEMENTED YET"
     | `StandardGaussian ->
     (* FIXME: *) failwith "NOT IMPLEMENTED YET"

 let empty = create_array [||] `Unspecified

type t = {
  mutable value: data;
  mutable grad: data;
  mutable forward: (unit -> unit) option;
  mutable backprop: (unit -> unit) option;
  label: string;
  id: int;
}

type state = {
  mutable unique_id: int;
  node_store: (int, t) Hashtbl.t;
}

let global = {
  unique_id = 1;
  node_store = Hashtbl.create (module Int);
}
let get uid = Hashtbl.find_exn global.node_store uid

let create ~label =
  let node = {
    value=empty; grad=empty;
    forward=None; backprop=None;
    label;
    id=let uid = global.unique_id in global.unique_id <- global.unique_id + 1; uid
  } in
  Hashtbl.add_exn global.node_store ~key:node.id ~data:node;
  node
