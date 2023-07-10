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

type t = Half_nd of half_nd | Single_nd of single_nd | Double_nd of double_nd [@@deriving sexp_of]

type 'a precision = Half : half_nd precision | Single : single_nd precision | Double : double_nd precision
[@@deriving sexp_of]

let as_t (type arr_t) (prec : arr_t precision) (arr : arr_t) =
  match prec with Half -> Half_nd arr | Single -> Single_nd arr | Double -> Double_nd arr

let precision_to_bigarray_kind (type elt_t) (prec : elt_t bigarray precision) : (float, elt_t) Bigarray.kind =
  match prec with Half -> float16 | Single -> Bigarray.Float32 | Double -> Bigarray.Float64

let precision_to_string (type arr_t) (prec : arr_t precision) =
  match prec with Half -> "half" | Single -> "single" | Double -> "double"

let precision_string = function Half_nd _ -> "half" | Single_nd _ -> "single" | Double_nd _ -> "double"
let default_kind = Single

type prec =
  | Void_prec : prec
  | Half_prec : half_nd precision -> prec
  | Single_prec : single_nd precision -> prec
  | Double_prec : double_nd precision -> prec

let half = Half_prec Half
let single = Single_prec Single
let double = Double_prec Double
let is_double_prec_t = function Double_nd _ -> true | _ -> false
let is_double (type arr_t) (prec : arr_t precision) = match prec with Double -> true | _ -> false
let is_double_prec = function Double_prec _ -> true | _ -> false

let sexp_of_prec = function
  | Void_prec -> Sexp.Atom "Void_prec"
  | Half_prec _ -> Sexp.Atom "Half_prec"
  | Single_prec _ -> Sexp.Atom "Single_prec"
  | Double_prec _ -> Sexp.Atom "Double_prec"

let prec_of_sexp = function
  | Sexp.Atom "Void_prec" -> Void_prec
  | Sexp.Atom "Half_prec" -> half
  | Sexp.Atom "Single_prec" -> single
  | Sexp.Atom "Double_prec" -> double
  | Sexp.List _ -> invalid_arg "prec_of_sexp: expected atom, found list"
  | Sexp.Atom s -> invalid_arg @@ "prec_of_sexp: unknown precision " ^ s

let pack_prec (type a) (prec : a precision) =
  match prec with Half -> half | Single -> single | Double -> double

let get_prec = function Half_nd _ -> half | Single_nd _ -> single | Double_nd _ -> double

type 'r map_as_bigarray = { f : 'a. 'a bigarray -> 'r }

let map { f } = function Half_nd arr -> f arr | Single_nd arr -> f arr | Double_nd arr -> f arr
let dims = map { f = A.dims }

let get_as_float arr idx =
  let f x = A.get x idx in
  map { f } arr

let get_as_int arr idx =
  let f x =
    let v = A.get x idx in
    try Float.to_int v
    with Invalid_argument _ ->
      Stdio.eprintf "\nOCANNL Runtime error: get_as_int invalid float: %f\n%!" v;
      0
  in
  map { f } arr

let set_from_float arr idx v =
  let f x = A.set x idx in
  map { f } arr v

let fill_from_float arr v = map { f = A.fill } arr v

type 'r map_prec = { f : 'a. 'a bigarray precision -> 'r }

let iter_prec { f } = function
  | Void_prec -> ()
  | Half_prec (Half | Single) -> f Half
  | Single_prec (Single | Half) -> f Single
  | Double_prec Double -> f Double
  | _ -> .

let map_prec ?default { f } = function
  | Void_prec -> Option.value_or_thunk default ~default:(fun () -> invalid_arg "map_prec: Void_prec")
  | Half_prec (Half | Single) -> f Half
  | Single_prec (Single | Half) -> f Single
  | Double_prec Double -> f Double
  | _ -> .

type 'r map_with_prec = { f : 'a. 'a bigarray precision -> 'a bigarray -> 'r }

let map_with_prec { f } = function
  | Half_nd arr -> f Half arr
  | Single_nd arr -> f Single arr
  | Double_nd arr -> f Double arr

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

let create prec dims init_op =
  let f prec = as_t prec @@ create_array prec dims init_op in
  map_prec { f } prec

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

let init_bigarray (init_op : init_op) (type b) (arr : b bigarray) =
  let dims = A.dims arr in
  match init_op with
  | Constant_fill cs ->
      let size = Array.length cs in
      reset_bigarray arr ~f:(fun idcs -> cs.(indices_to_offset ~dims ~idcs % size))
  | Range_over_offsets -> reset_bigarray arr ~f:(fun idcs -> Float.of_int @@ indices_to_offset ~dims ~idcs)
  | Standard_uniform -> reset_bigarray arr ~f:(fun _ -> Random.float_range 0.0 1.0)

let init init_op arr =
  let f arr = init_bigarray init_op arr in
  map { f } arr

let fold ~init ~f arr =
  let f arr = fold_bigarray ~init ~f arr in
  map { f } arr

let retrieve_2d_points ?from_axis ~xdim ~ydim arr =
  let dims = dims arr in
  if Array.is_empty dims then [||]
  else
    let n_axes = Array.length dims in
    let from_axis = Option.value from_axis ~default:(n_axes - 1) in
    let result = ref [] in
    let idx = Array.create ~len:n_axes 0 in
    let rec iter axis =
      if axis = n_axes then
        let x =
          idx.(from_axis) <- xdim;
          get_as_float arr idx
        in
        let y =
          idx.(from_axis) <- ydim;
          get_as_float arr idx
        in
        result := (x, y) :: !result
      else if axis = from_axis then iter (axis + 1)
      else
        for p = 0 to dims.(axis) - 1 do
          idx.(axis) <- p;
          iter (axis + 1)
        done
    in
    iter 0;
    Array.of_list_rev !result

let retrieve_1d_points ?from_axis ~xdim arr =
  let dims = dims arr in
  if Array.is_empty dims then [||]
  else
    let n_axes = Array.length dims in
    let from_axis = Option.value from_axis ~default:(n_axes - 1) in
    let result = ref [] in
    let idx = Array.create ~len:n_axes 0 in
    let rec iter axis =
      if axis = n_axes then
        let x =
          idx.(from_axis) <- xdim;
          get_as_float arr idx
        in
        result := x :: !result
      else if axis = from_axis then iter (axis + 1)
      else
        for p = 0 to dims.(axis) - 1 do
          idx.(axis) <- p;
          iter (axis + 1)
        done
    in
    iter 0;
    Array.of_list_rev !result