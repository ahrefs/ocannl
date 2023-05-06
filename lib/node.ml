open Base
(** `Node`: the computation type, global state and utils which the `Formula` staged code uses. *)

module A = Bigarray.Genarray

type ('a, 'b, 'c) bigarray = ('a, 'b, 'c) A.t

let sexp_of_bigarray (arr : ('a, 'b, 'c) bigarray) =
  let dims = A.dims arr in
  Sexp.Atom ("<bigarray_dims_" ^ String.concat_array ~sep:"x" (Array.map dims ~f:Int.to_string))

(* type bit_as_bool_nd = (bool, Bigarray.bool_elt, Bigarray.c_layout) bigarray *)
type byte_as_int_nd = (int, Bigarray.int8_signed_elt, Bigarray.c_layout) bigarray

let sexp_of_byte_as_int_nd (arr : byte_as_int_nd) =
  let dims = A.dims arr in
  Sexp.Atom ("<byte_as_int_nd_dims_" ^ String.concat_array ~sep:"x" (Array.map dims ~f:Int.to_string))

type half_as_int_nd = (int, Bigarray.int16_signed_elt, Bigarray.c_layout) bigarray

let sexp_of_half_as_int_nd (arr : half_as_int_nd) =
  let dims = A.dims arr in
  Sexp.Atom ("<half_as_int_nd_dims_" ^ String.concat_array ~sep:"x" (Array.map dims ~f:Int.to_string))

type single_nd = (float, Bigarray.float32_elt, Bigarray.c_layout) bigarray

let sexp_of_single_nd (arr : single_nd) =
  let dims = A.dims arr in
  Sexp.Atom ("<single_nd_dims_" ^ String.concat_array ~sep:"x" (Array.map dims ~f:Int.to_string))

type double_nd = (float, Bigarray.float64_elt, Bigarray.c_layout) bigarray

let sexp_of_double_nd (arr : double_nd) =
  let dims = A.dims arr in
  Sexp.Atom ("<double_nd_dims_" ^ String.concat_array ~sep:"x" (Array.map dims ~f:Int.to_string))

type ndarray =
  (* | Bit_as_bool_nd of bit_as_bool_nd *)
  | Byte_as_int_nd of byte_as_int_nd
  | Half_as_int_nd of half_as_int_nd
  | Single_nd of single_nd
  | Double_nd of double_nd
[@@deriving sexp_of]

type ('a, 'b) precision =
  (* | Bit_as_bool: (bool, bit_as_bool_nd) precision *)
  | Byte_as_int : (int, byte_as_int_nd) precision
  | Half_as_int : (int, half_as_int_nd) precision
  (* | Bit: (float, (bool, Bigarray.bool_elt, Bigarray.c_layout) bigarray) precision *)
  (* | Byte: (float, (float, Bigarray.float8_elt, Bigarray.c_layout) bigarray) precision *)
  (* | Half: (float, (float, Bigarray.float16_elt, Bigarray.c_layout) bigarray) precision *)
  | Single : (float, single_nd) precision
  | Double : (float, double_nd) precision

let as_ndarray (type val_t arr_t) (prec : (val_t, arr_t) precision) (arr : arr_t) =
  match prec with
  | Byte_as_int -> Byte_as_int_nd arr
  | Half_as_int -> Half_as_int_nd arr
  | Single -> Single_nd arr
  | Double -> Double_nd arr

let precision_to_bigarray_kind (type val_t elt_t)
    (prec : (val_t, (val_t, elt_t, Bigarray.c_layout) bigarray) precision) : (val_t, elt_t) Bigarray.kind =
  match prec with
  (* | Bit -> Bigarray.Bool *)
  | Byte_as_int -> Bigarray.Int8_signed
  | Half_as_int -> Bigarray.Int16_signed
  (* | Half -> Bigarray.Float16 *)
  | Single -> Bigarray.Float32
  | Double -> Bigarray.Float64
  | _ -> . (* invalid_arg "precision_to_bigarray_kind: not a Bigarray precision" *)

let precision_to_string (type val_t arr_t) (prec : (val_t, arr_t) precision) =
  match prec with Byte_as_int -> "byte" | Half_as_int -> "half" | Single -> "single" | Double -> "double"

let ndarray_precision_to_string = function
  | Byte_as_int_nd _ -> "byte"
  | Half_as_int_nd _ -> "half"
  | Single_nd _ -> "single"
  | Double_nd _ -> "double"

let default_kind = Single

type 'c map_as_bigarray = { f : 'a 'b. ('a, 'b, Bigarray.c_layout) bigarray -> 'c }

let map_as_bigarray { f } = function
  | Byte_as_int_nd arr -> f arr
  | Half_as_int_nd arr -> f arr
  | Single_nd arr -> f arr
  | Double_nd arr -> f arr

let dims = map_as_bigarray { f = A.dims }

let get_as_float arr idx =
  match arr with
  | Byte_as_int_nd arr -> Float.of_int (A.get arr idx)
  | Half_as_int_nd arr -> Float.of_int (A.get arr idx)
  | Single_nd arr -> A.get arr idx
  | Double_nd arr -> A.get arr idx

let get_as_int arr idx =
  match arr with
  | Byte_as_int_nd arr -> A.get arr idx
  | Half_as_int_nd arr -> A.get arr idx
  | Single_nd arr -> (
      let v = A.get arr idx in
      try Float.to_int v
      with Invalid_argument _ ->
        Stdio.eprintf "\nOCANNL Runtime error: get_as_int invalid float: %f\n%!" v;
        0)
  | Double_nd arr -> (
      let v = A.get arr idx in
      try Float.to_int v
      with Invalid_argument _ ->
        Stdio.eprintf "\nOCANNL Runtime error: get_as_int invalid float: %f\n%!" v;
        0)

let set_from_float arr idx v =
  match arr with
  | Byte_as_int_nd arr -> A.set arr idx (Int.of_float v)
  | Half_as_int_nd arr -> A.set arr idx (Int.of_float v)
  | Single_nd arr -> A.set arr idx v
  | Double_nd arr -> A.set arr idx v

let fill_from_float arr v =
  match arr with
  | Byte_as_int_nd arr -> A.fill arr (Int.of_float v)
  | Half_as_int_nd arr -> A.fill arr (Int.of_float v)
  | Single_nd arr -> A.fill arr v
  | Double_nd arr -> A.fill arr v

(** Initializes or resets a tensor by filling in the corresponding numbers, at the appropriate precision. *)
type init_op =
  | Constant_fill of float array
      (** Fills in the numbers where the rightmost axis is contiguous, looping over the provided values. *)
  | Range_over_offsets
      (** Fills in the offset number of each cell (i.e. how many cells away it is from the beginning). *)
  | Standard_uniform  (** Draws the values from U(0,1). *)
[@@deriving sexp]

let create_array_of_prec (type val_t arr_t) (prec : (val_t, arr_t) precision) dims : arr_t =
  match prec with
  | Byte_as_int -> A.create Bigarray.Int8_signed Bigarray.C_layout dims
  | Half_as_int -> A.create Bigarray.Int16_signed Bigarray.C_layout dims
  | Single -> A.create Bigarray.Float32 Bigarray.C_layout dims
  | Double -> A.create Bigarray.Float64 Bigarray.C_layout dims

let init_array_of_prec (type val_t arr_t) (prec : (val_t, arr_t) precision) dims ~(f : int array -> val_t) =
  match prec with
  | Byte_as_int -> (A.init Bigarray.Int8_signed Bigarray.C_layout dims f : arr_t)
  | Half_as_int -> A.init Bigarray.Int16_signed Bigarray.C_layout dims f
  | Single -> A.init Bigarray.Float32 Bigarray.C_layout dims f
  | Double -> A.init Bigarray.Float64 Bigarray.C_layout dims f

let indices_to_offset ~dims ~idcs =
  Array.fold2_exn dims idcs ~init:0 ~f:(fun accu dim idx -> (accu * dim) + idx)

let create_array (type arr_t) (prec : (float, arr_t) precision) dims (init_op : init_op) : arr_t =
  match init_op with
  | Constant_fill cs ->
      let size = Array.length cs in
      init_array_of_prec prec dims ~f:(fun idcs -> cs.(indices_to_offset ~dims ~idcs % size))
  | Range_over_offsets ->
      init_array_of_prec prec dims ~f:(fun idcs -> Float.of_int @@ indices_to_offset ~dims ~idcs)
  | Standard_uniform -> init_array_of_prec prec dims ~f:(fun _ -> Random.float_range 0.0 1.0)

let create_ndarray prec dims init_op = as_ndarray prec @@ create_array prec dims init_op

type 'c cast_map_as_bigarray = {
  ff :
    'a 'b.
    (float -> 'a) ->
    ('a, ('a, 'b, Bigarray.c_layout) bigarray) precision ->
    ('a, 'b, Bigarray.c_layout) bigarray ->
    'c;
}

let cast_map_as_bigarray { ff } = function
  | Byte_as_int_nd arr -> ff Int.of_float Byte_as_int arr
  | Half_as_int_nd arr -> ff Int.of_float Half_as_int arr
  | Single_nd arr -> ff Fn.id Single arr
  | Double_nd arr -> ff Fn.id Double arr

let loop_bigarray arr ~f =
  let dims = A.dims arr in
  let rec cloop idx f col =
    if col = Array.length idx then A.set arr idx (f idx)
    else
      for j = 0 to Int.pred dims.(col) do
        idx.(col) <- j;
        cloop idx f (Int.succ col)
      done
  in
  let len = Array.length dims in
  cloop (Array.create ~len 0) f 0

let empty prec = create_array prec [||] (Constant_fill [| 0.0 |])

type t = { mutable value : ndarray; mutable grad : ndarray option; id : int } [@@deriving sexp_of]

let size_in_bytes n =
  (* Cheating here because 1 number Bigarray is same size as empty Bigarray:
     it's more informative to report the cases differently. *)
  let f arr = if Array.is_empty @@ A.dims arr then 0 else A.size_in_bytes arr in
  let size = map_as_bigarray { f } in
  size n.value + Option.value_map ~f:size n.grad ~default:0

exception Runtime_error of string * t option

let most_recent_suspension : (unit -> unit) option ref = ref None

type state = {
  mutable unique_id : int;
  node_store : (int, t) Hashtbl.t;
  session_step_update : (unit -> unit) option ref;
}

let global =
  {
    unique_id = 1;
    node_store = Hashtbl.create (module Int);
    session_step_update = ref @@ Some (fun () -> ());
  }

let global_size_in_bytes () =
  Hashtbl.fold global.node_store ~init:0 ~f:(fun ~key:_ ~data sum -> sum + size_in_bytes data)

let get uid = Hashtbl.find_exn global.node_store uid

let get_value (type val_t arr_t) (prec : (val_t, arr_t) precision) uid : arr_t =
  let n = Hashtbl.find_exn global.node_store uid in
  match (prec, n.value) with
  | Byte_as_int, Byte_as_int_nd arr -> arr
  | Half_as_int, Half_as_int_nd arr -> arr
  | Single, Single_nd arr -> arr
  | Double, Double_nd arr -> arr
  | _, arr ->
      raise
      @@ Runtime_error
           ( "Precision mismatch: expected " ^ precision_to_string prec ^ ", got "
             ^ ndarray_precision_to_string arr,
             Some n )

let get_grad (type val_t arr_t) (prec : (val_t, arr_t) precision) uid : arr_t =
  let n = Hashtbl.find_exn global.node_store uid in
  match (prec, n.grad) with
  | Byte_as_int, Some (Byte_as_int_nd arr) -> arr
  | Half_as_int, Some (Half_as_int_nd arr) -> arr
  | Single, Some (Single_nd arr) -> arr
  | Double, Some (Double_nd arr) -> arr
  | _, Some arr ->
      raise
      @@ Runtime_error
           ( "Precision mismatch: expected " ^ precision_to_string prec ^ ", got "
             ^ ndarray_precision_to_string arr,
             Some n )
  | _, None -> raise @@ Runtime_error ("get_grad: non-form node", Some n)

(** Constructs a node with empty tensors of the specified precision and registers it in the global store.
    Note that the precision for gradients should not be lower than the precision for values. *)
let create (type grad_arr_t value_arr_t) ~(value_prec : ('a, value_arr_t) precision)
    ?(grad_prec : ('a, grad_arr_t) precision option) ~needs_gradient () =
  let id =
    let uid = global.unique_id in
    global.unique_id <- global.unique_id + 1;
    uid
  in
  let grad =
    match (grad_prec, needs_gradient) with
    | Some grad_prec, true -> Some (as_ndarray grad_prec @@ empty grad_prec)
    | None, true -> invalid_arg "Node.create: ~needs_gradient:true requires providing ~grad_prec"
    | _, false -> None
  in
  let node = { value = as_ndarray value_prec @@ empty value_prec; grad; id } in
  Hashtbl.add_exn global.node_store ~key:node.id ~data:node;
  node

let init_bigarray (init_op : init_op) (type val_t b) (cast : float -> val_t)
    (_prec : (val_t, (val_t, b, Bigarray.c_layout) bigarray) precision)
    (arr : (val_t, b, Bigarray.c_layout) bigarray) =
  let dims = A.dims arr in
  match init_op with
  | Constant_fill cs ->
      let size = Array.length cs in
      loop_bigarray arr ~f:(fun idcs -> cast cs.(indices_to_offset ~dims ~idcs % size))
  | Range_over_offsets ->
      loop_bigarray arr ~f:(fun idcs -> cast @@ Float.of_int @@ indices_to_offset ~dims ~idcs)
  | Standard_uniform -> loop_bigarray arr ~f:(fun _ -> cast @@ Random.float_range 0.0 1.0)

let init_ndarray init_op arr =
  let ff arr = init_bigarray init_op arr in
  cast_map_as_bigarray { ff } arr
