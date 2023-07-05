open Base
(** `Node`: the computation type, global state and utils which the `Formula` staged code uses. *)

module A = Bigarray.Genarray

type 'a bigarray = (float, 'a, Bigarray.c_layout) A.t

let sexp_of_bigarray (arr : 'a bigarray) =
  let dims = A.dims arr in
  Sexp.Atom ("bigarray_dims_" ^ String.concat_array ~sep:"x" (Array.map dims ~f:Int.to_string))

(* FIXME: Upcoming in OCaml 5.2.0. See:
   https://github.com/ocaml/ocaml/pull/10775/commits/ba6a2c378056c8669fb1bb99bf07b12d69bd4a12 *)
type float16_elt = Bigarray.float32_elt
type float32_elt = Bigarray.float32_elt

let float16 : (float, float16_elt) Bigarray.kind = Bigarray.float32

type half_nd = float16_elt bigarray

let sexp_of_half_nd (arr : half_nd) =
  let dims = A.dims arr in
  Sexp.Atom ("half_nd_dims_" ^ String.concat_array ~sep:"x" (Array.map dims ~f:Int.to_string))

type single_nd = Bigarray.float32_elt bigarray

let sexp_of_single_nd (arr : single_nd) =
  let dims = A.dims arr in
  Sexp.Atom ("single_nd_dims_" ^ String.concat_array ~sep:"x" (Array.map dims ~f:Int.to_string))

type double_nd = Bigarray.float64_elt bigarray

let sexp_of_double_nd (arr : double_nd) =
  let dims = A.dims arr in
  Sexp.Atom ("double_nd_dims_" ^ String.concat_array ~sep:"x" (Array.map dims ~f:Int.to_string))

type ndarray = Half_nd of half_nd | Single_nd of single_nd | Double_nd of double_nd [@@deriving sexp_of]

type 'a precision = Half : half_nd precision | Single : single_nd precision | Double : double_nd precision
[@@deriving sexp_of]

let as_ndarray (type arr_t) (prec : arr_t precision) (arr : arr_t) =
  match prec with Half -> Half_nd arr | Single -> Single_nd arr | Double -> Double_nd arr

let precision_to_bigarray_kind (type elt_t) (prec : elt_t bigarray precision) : (float, elt_t) Bigarray.kind =
  match prec with
  | Half -> float16
  | Single -> Bigarray.Float32
  | Double -> Bigarray.Float64
  | _ -> . (* invalid_arg "precision_to_bigarray_kind: not a Bigarray precision" *)

let precision_to_string (type arr_t) (prec : arr_t precision) =
  match prec with Half -> "half" | Single -> "single" | Double -> "double"

let ndarray_precision_to_string = function
  | Half_nd _ -> "half"
  | Single_nd _ -> "single"
  | Double_nd _ -> "double"

let default_kind = Single

type 'r map_as_bigarray = { f : 'a. 'a bigarray -> 'r }

let map_as_bigarray { f } = function Half_nd arr -> f arr | Single_nd arr -> f arr | Double_nd arr -> f arr
let dims = map_as_bigarray { f = A.dims }

let get_as_float arr idx =
  let f x = A.get x idx in
  map_as_bigarray { f } arr

let get_as_int arr idx =
  let f x =
    let v = A.get x idx in
    try Float.to_int v
    with Invalid_argument _ ->
      Stdio.eprintf "\nOCANNL Runtime error: get_as_int invalid float: %f\n%!" v;
      0
  in
  map_as_bigarray { f } arr

let set_from_float arr idx v =
  let f x = A.set x idx in
  map_as_bigarray { f } arr v

let fill_from_float arr v = map_as_bigarray { f = A.fill } arr v

(** Initializes or resets a tensor by filling in the corresponding numbers, at the appropriate precision. *)
type init_op =
  | Constant_fill of float array
      (** Fills in the numbers where the rightmost axis is contiguous, looping over the provided values. *)
  | Range_over_offsets
      (** Fills in the offset number of each cell (i.e. how many cells away it is from the beginning). *)
  | Standard_uniform  (** Draws the values from U(0,1). *)
[@@deriving sexp]

let create_array_of_prec (type elt_t) (prec : elt_t bigarray precision) dims : elt_t bigarray =
  A.create (precision_to_bigarray_kind prec) Bigarray.C_layout dims

let init_array_of_prec (type elt_t) (prec : elt_t bigarray precision) dims ~(f : int array -> float) :
    elt_t bigarray =
  A.init (precision_to_bigarray_kind prec) Bigarray.C_layout dims f

let indices_to_offset ~dims ~idcs =
  Array.fold2_exn dims idcs ~init:0 ~f:(fun accu dim idx -> (accu * dim) + idx)

let fixed_state_for_init = ref None

let create_array (type elt_t) (prec : elt_t bigarray precision) dims (init_op : init_op) : elt_t bigarray =
  Option.iter !fixed_state_for_init ~f:(fun seed -> Random.init seed);
  match init_op with
  | Constant_fill cs ->
      let size = Array.length cs in
      init_array_of_prec prec dims ~f:(fun idcs -> cs.(indices_to_offset ~dims ~idcs % size))
  | Range_over_offsets ->
      init_array_of_prec prec dims ~f:(fun idcs -> Float.of_int @@ indices_to_offset ~dims ~idcs)
  | Standard_uniform -> init_array_of_prec prec dims ~f:(fun _ -> Random.float_range 0.0 1.0)

let create_ndarray prec dims init_op = as_ndarray prec @@ create_array prec dims init_op
let precision_in_bytes = function Half_nd _ -> 2 | Single_nd _ -> 4 | Double_nd _ -> 8

let reset_bigarray arr ~f =
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

let fold_bigarray arr ~init ~f =
  let dims = A.dims arr in
  let accu = ref init in
  let rec cloop idx col =
    if col = Array.length idx then accu := f !accu idx @@ A.get arr idx
    else
      for j = 0 to Int.pred dims.(col) do
        idx.(col) <- j;
        cloop idx (Int.succ col)
      done
  in
  let len = Array.length dims in
  cloop (Array.create ~len 0) 0;
  !accu

let empty prec = create_array prec [||] (Constant_fill [| 0.0 |])

type t = { mutable value : ndarray; mutable grad : ndarray option; id : int } [@@deriving sexp_of]

let host_size n =
  let dims = map_as_bigarray { f = A.dims } n.value in
  if Array.is_empty dims then 0 else Array.fold dims ~init:1 ~f:( * )

let host_size_in_bytes n =
  (* Cheating here because 1 number Bigarray is same size as empty Bigarray:
     it's more informative to report the cases differently. *)
  let f arr = if Array.is_empty @@ A.dims arr then 0 else A.size_in_bytes arr in
  let size = map_as_bigarray { f } in
  size n.value + Option.value_map ~f:size n.grad ~default:0

type state = { mutable unique_id : int; node_store : (int, t) Hashtbl.t }

let global = { unique_id = 1; node_store = Hashtbl.create (module Int) }

let global_host_size_in_bytes () =
  Hashtbl.fold global.node_store ~init:0 ~f:(fun ~key:_ ~data sum -> sum + host_size_in_bytes data)

let get uid = Hashtbl.find_exn global.node_store uid

(** Constructs a node with empty tensors of the specified precision and registers it in the global store.
    Note that the precision for gradients should not be lower than the precision for values. *)
let create (type grad_elt_t value_elt_t) ~(value_prec : value_elt_t bigarray precision)
    ?(grad_prec : grad_elt_t bigarray precision option) ~needs_gradient () =
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

let init_bigarray (init_op : init_op) (type b) (arr : b bigarray) =
  let dims = A.dims arr in
  match init_op with
  | Constant_fill cs ->
      let size = Array.length cs in
      reset_bigarray arr ~f:(fun idcs -> cs.(indices_to_offset ~dims ~idcs % size))
  | Range_over_offsets -> reset_bigarray arr ~f:(fun idcs -> Float.of_int @@ indices_to_offset ~dims ~idcs)
  | Standard_uniform -> reset_bigarray arr ~f:(fun _ -> Random.float_range 0.0 1.0)

let init_ndarray init_op arr =
  let f arr = init_bigarray init_op arr in
  map_as_bigarray { f } arr

let fold_ndarray ~init ~f arr =
  let f arr = fold_bigarray ~init ~f arr in
  map_as_bigarray { f } arr
