open Base
module Lazy = Utils.Lazy

(** N-dimensional arrays: a precision-handling wrapper for [Bigarray.Genarray] and its utilities. *)

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

module A = Bigarray.Genarray

(** {2 *** Handling of precisions ***} *)

type ('ocaml, 'elt_t) bigarray = ('ocaml, 'elt_t, Bigarray.c_layout) A.t

let bigarray_start_not_managed (arr : ('ocaml, 'elt_t) bigarray) =
  Ctypes_bigarray.unsafe_address arr

let big_ptr_to_string arr = "@" ^ Nativeint.Hex.to_string (bigarray_start_not_managed arr)

let sexp_of_bigarray (arr : ('a, 'b) bigarray) =
  let dims = A.dims arr in
  Sexp.Atom ("bigarray_dims_" ^ String.concat_array ~sep:"x" (Array.map dims ~f:Int.to_string))

type byte_nd = (char, Ops.uint8_elt) bigarray
type uint16_nd = (int, Ops.uint16_elt) bigarray
type int32_nd = (int32, Ops.int32_elt) bigarray
type uint4x32_nd = (Stdlib.Complex.t, Bigarray.complex64_elt) bigarray
type half_nd = (float, Ops.float16_elt) bigarray
type bfloat16_nd = (int, Ops.uint16_elt) bigarray (* Using uint16 representation *)
type fp8_nd = (char, Ops.uint8_elt) bigarray (* Using uint8 representation *)
type single_nd = (float, Ops.float32_elt) bigarray
type double_nd = (float, Ops.float64_elt) bigarray

let sexp_of_byte_nd (arr : byte_nd) = Sexp.Atom (big_ptr_to_string arr)
let sexp_of_uint16_nd (arr : uint16_nd) = Sexp.Atom (big_ptr_to_string arr)
let sexp_of_int32_nd (arr : int32_nd) = Sexp.Atom (big_ptr_to_string arr)
let sexp_of_uint4x32_nd (arr : uint4x32_nd) = Sexp.Atom (big_ptr_to_string arr)
let sexp_of_half_nd (arr : half_nd) = Sexp.Atom (big_ptr_to_string arr)
let sexp_of_bfloat16_nd (arr : bfloat16_nd) = Sexp.Atom (big_ptr_to_string arr)
let sexp_of_fp8_nd (arr : fp8_nd) = Sexp.Atom (big_ptr_to_string arr)
let sexp_of_single_nd (arr : single_nd) = Sexp.Atom (big_ptr_to_string arr)
let sexp_of_double_nd (arr : double_nd) = Sexp.Atom (big_ptr_to_string arr)

type t =
  | Byte_nd of byte_nd
  | Uint16_nd of uint16_nd
  | Int32_nd of int32_nd
  | Uint4x32_nd of uint4x32_nd
  | Half_nd of half_nd
  | Bfloat16_nd of bfloat16_nd
  | Fp8_nd of fp8_nd
  | Single_nd of single_nd
  | Double_nd of double_nd
[@@deriving sexp_of]

let as_array (type ocaml elt_t) (prec : (ocaml, elt_t) Ops.precision)
    (arr : (ocaml, elt_t) bigarray) =
  match prec with
  | Ops.Byte -> Byte_nd arr
  | Ops.Uint16 -> Uint16_nd arr
  | Ops.Int32 -> Int32_nd arr
  | Ops.Uint4x32 -> Uint4x32_nd arr
  | Ops.Half -> Half_nd arr
  | Ops.Bfloat16 -> Bfloat16_nd arr
  | Ops.Fp8 -> Fp8_nd arr
  | Ops.Single -> Single_nd arr
  | Ops.Double -> Double_nd arr

let precision_to_bigarray_kind (type ocaml elt_t) (prec : (ocaml, elt_t) Ops.precision) :
    (ocaml, elt_t) Bigarray.kind =
  match prec with
  | Ops.Byte -> Bigarray.Char
  | Ops.Uint16 -> Bigarray.Int16_unsigned
  | Ops.Int32 -> Bigarray.Int32
  | Ops.Uint4x32 -> Bigarray.Complex64
  | Ops.Half -> Bigarray.Float16
  | Ops.Bfloat16 -> Bigarray.Int16_unsigned (* Using uint16 representation *)
  | Ops.Fp8 -> Bigarray.Char (* Using uint8 representation *)
  | Ops.Single -> Bigarray.Float32
  | Ops.Double -> Bigarray.Float64

let precision_string = function
  | Byte_nd _ -> "byte"
  | Uint16_nd _ -> "uint16"
  | Int32_nd _ -> "int32"
  | Uint4x32_nd _ -> "uint4x32"
  | Half_nd _ -> "half"
  | Bfloat16_nd _ -> "bfloat16"
  | Fp8_nd _ -> "fp8"
  | Single_nd _ -> "single"
  | Double_nd _ -> "double"

let default_kind = Ops.Single

let get_prec = function
  | Byte_nd _ -> Ops.byte
  | Uint16_nd _ -> Ops.uint16
  | Int32_nd _ -> Ops.int32
  | Uint4x32_nd _ -> Ops.uint4x32
  | Half_nd _ -> Ops.half
  | Bfloat16_nd _ -> Ops.bfloat16
  | Fp8_nd _ -> Ops.fp8
  | Single_nd _ -> Ops.single
  | Double_nd _ -> Ops.double

type 'r apply_with_prec = {
  f : 'ocaml 'elt_t. ('ocaml, 'elt_t) Ops.precision -> ('ocaml, 'elt_t) bigarray -> 'r;
}

let apply_with_prec { f } = function
  | Byte_nd arr -> f Ops.Byte arr
  | Uint16_nd arr -> f Ops.Uint16 arr
  | Int32_nd arr -> f Ops.Int32 arr
  | Uint4x32_nd arr -> f Ops.Uint4x32 arr
  | Half_nd arr -> f Ops.Half arr
  | Bfloat16_nd arr -> f Ops.Bfloat16 arr
  | Fp8_nd arr -> f Ops.Fp8 arr
  | Single_nd arr -> f Ops.Single arr
  | Double_nd arr -> f Ops.Double arr

let create_bigarray_of_prec (type ocaml elt_t) (prec : (ocaml, elt_t) Ops.precision) dims :
    (ocaml, elt_t) bigarray =
  A.create (precision_to_bigarray_kind prec) Bigarray.C_layout dims

(** {2 *** Initialization ***} *)

let create_bigarray (type ocaml elt_t) (prec : (ocaml, elt_t) Ops.precision) ~dims ~padding :
    (ocaml, elt_t) bigarray =
  let arr = create_bigarray_of_prec prec dims in
  (* Fill with padding value if padding is specified *)
  (match padding with
  | None -> ()
  | Some (_, pad_value) -> (
      (* Fill the entire array with padding value using precision-specific fill *)
      match prec with
      | Ops.Byte -> A.fill arr (Char.of_int_exn @@ Int.of_float pad_value)
      | Ops.Uint16 -> A.fill arr (Int.of_float pad_value)
      | Ops.Int32 -> A.fill arr (Int32.of_float pad_value)
      | Ops.Uint4x32 -> A.fill arr Stdlib.Complex.{ re = pad_value; im = 0.0 }
      | Ops.Half -> A.fill arr pad_value
      | Ops.Bfloat16 -> A.fill arr (Ops.single_to_bfloat16 pad_value)
      | Ops.Fp8 -> A.fill arr (Char.of_int_exn @@ Ops.single_to_fp8 pad_value)
      | Ops.Single -> A.fill arr pad_value
      | Ops.Double -> A.fill arr pad_value));
  arr

(** {2 *** Accessing ***} *)

type 'r apply_as_bigarray = { f : 'ocaml 'elt_t. ('ocaml, 'elt_t) bigarray -> 'r }

let apply { f } = function
  | Byte_nd arr -> f arr
  | Uint16_nd arr -> f arr
  | Int32_nd arr -> f arr
  | Uint4x32_nd arr -> f arr
  | Half_nd arr -> f arr
  | Bfloat16_nd arr -> f arr
  | Fp8_nd arr -> f arr
  | Single_nd arr -> f arr
  | Double_nd arr -> f arr

type 'r apply2_as_bigarray = {
  f2 : 'ocaml 'elt_t. ('ocaml, 'elt_t) bigarray -> ('ocaml, 'elt_t) bigarray -> 'r;
}

let apply2 { f2 } x1 x2 =
  match (x1, x2) with
  | Byte_nd arr1, Byte_nd arr2 -> f2 arr1 arr2
  | Uint16_nd arr1, Uint16_nd arr2 -> f2 arr1 arr2
  | Int32_nd arr1, Int32_nd arr2 -> f2 arr1 arr2
  | Uint4x32_nd arr1, Uint4x32_nd arr2 -> f2 arr1 arr2
  | Half_nd arr1, Half_nd arr2 -> f2 arr1 arr2
  | Bfloat16_nd arr1, Bfloat16_nd arr2 -> f2 arr1 arr2
  | Fp8_nd arr1, Fp8_nd arr2 -> f2 arr1 arr2
  | Single_nd arr1, Single_nd arr2 -> f2 arr1 arr2
  | Double_nd arr1, Double_nd arr2 -> f2 arr1 arr2
  | _ -> invalid_arg "Ndarray.apply2: precision mismatch"

let dims = apply { f = A.dims }

let get_fatptr_not_managed nd =
  let f arr =
    Ctypes_memory.make_unmanaged ~reftyp:Ctypes_static.void @@ bigarray_start_not_managed arr
  in
  apply { f } nd

let get_voidptr_not_managed nd : unit Ctypes.ptr =
  Ctypes_static.CPointer (get_fatptr_not_managed nd)
(* This doesn't work because Ctypes.bigarray_start doesn't support half precision: *)
(* let open Ctypes in coerce (ptr @@ typ_of_bigarray_kind @@ Bigarray.Genarray.kind arr) (ptr void)
   (bigarray_start genarray arr) *)

(** Helper function to adjust indices by adding left padding when padding is specified *)
let adjust_idx_for_padding ?padding idx =
  match padding with
  | None -> idx
  | Some padding_arr ->
      Array.mapi idx ~f:(fun i dim_idx ->
          if i < Array.length padding_arr then dim_idx + padding_arr.(i).Ops.left else dim_idx)

(** Helper function to compute end index for iteration, respecting padding margins *)
let compute_end_idx ?padding dims axis =
  match padding with
  | None -> dims.(axis) - 1
  | Some padding_arr when axis < Array.length padding_arr ->
      dims.(axis) - padding_arr.(axis).Ops.left - padding_arr.(axis).Ops.right - 1
  | Some _ -> dims.(axis) - 1

let set_from_float ?padding arr idx v =
  let adjusted_idx = adjust_idx_for_padding ?padding idx in
  match arr with
  | Byte_nd arr -> A.set arr adjusted_idx @@ Char.of_int_exn @@ Int.of_float v
  | Uint16_nd arr -> A.set arr adjusted_idx @@ Int.of_float v
  | Int32_nd arr -> A.set arr adjusted_idx @@ Int32.of_float v
  | Uint4x32_nd arr -> A.set arr adjusted_idx @@ Stdlib.Complex.{ re = v; im = 0.0 }
  | Half_nd arr -> A.set arr adjusted_idx v
  | Bfloat16_nd arr -> A.set arr adjusted_idx @@ Ops.single_to_bfloat16 v
  | Fp8_nd arr -> A.set arr adjusted_idx @@ Char.of_int_exn @@ Ops.single_to_fp8 v
  | Single_nd arr -> A.set arr adjusted_idx v
  | Double_nd arr -> A.set arr adjusted_idx v

let fill_from_float arr v =
  match arr with
  | Byte_nd arr -> A.fill arr @@ Char.of_int_exn @@ Int.of_float v
  | Uint16_nd arr -> A.fill arr @@ Int.of_float v
  | Int32_nd arr -> A.fill arr @@ Int32.of_float v
  | Uint4x32_nd arr -> A.fill arr @@ Stdlib.Complex.{ re = v; im = 0.0 }
  | Half_nd arr -> A.fill arr v
  | Bfloat16_nd arr -> A.fill arr @@ Ops.single_to_bfloat16 v
  | Fp8_nd arr -> A.fill arr @@ Char.of_int_exn @@ Ops.single_to_fp8 v
  | Single_nd arr -> A.fill arr v
  | Double_nd arr -> A.fill arr v

let fold_bigarray ?padding arr ~init ~f =
  let dims = A.dims arr in
  let accu = ref init in
  let rec cloop idx col =
    if col = Array.length idx then
      let adjusted_idx = adjust_idx_for_padding ?padding idx in
      accu := f !accu idx @@ A.get arr adjusted_idx
    else
      let end_idx = compute_end_idx ?padding dims col in
      for j = 0 to end_idx do
        idx.(col) <- j;
        cloop idx (Int.succ col)
      done
  in
  let len = Array.length dims in
  cloop (Array.create ~len 0) 0;
  !accu

let fold_as_float ?padding ~init ~f arr =
  match arr with
  | Byte_nd arr ->
      fold_bigarray ?padding ~init
        ~f:(fun accu idx c -> f accu idx @@ Float.of_int @@ Char.to_int c)
        arr
  | Uint16_nd arr ->
      fold_bigarray ?padding ~init ~f:(fun accu idx v -> f accu idx @@ Float.of_int v) arr
  | Int32_nd arr ->
      fold_bigarray ?padding ~init ~f:(fun accu idx v -> f accu idx @@ Int32.to_float v) arr
  | Uint4x32_nd arr ->
      fold_bigarray ?padding ~init ~f:(fun accu idx c -> f accu idx c.Stdlib.Complex.re) arr
  | Half_nd arr -> fold_bigarray ?padding ~init ~f arr
  | Bfloat16_nd arr ->
      fold_bigarray ?padding ~init ~f:(fun accu idx v -> f accu idx @@ Ops.bfloat16_to_single v) arr
  | Fp8_nd arr ->
      fold_bigarray ?padding ~init
        ~f:(fun accu idx c -> f accu idx @@ Ops.fp8_to_single @@ Char.to_int c)
        arr
  | Single_nd arr -> fold_bigarray ?padding ~init ~f arr
  | Double_nd arr -> fold_bigarray ?padding ~init ~f arr

let size_in_bytes v =
  (* Use Bigarray's natural behavior: empty dimensions have the same size as 1-element arrays *)
  let f arr = A.size_in_bytes arr in
  apply { f } v

let get_as_float ?padding arr idx =
  let adjusted_idx = adjust_idx_for_padding ?padding idx in
  match arr with
  | Byte_nd arr -> Float.of_int @@ Char.to_int @@ A.get arr adjusted_idx
  | Uint16_nd arr -> Float.of_int @@ A.get arr adjusted_idx
  | Int32_nd arr -> Int32.to_float @@ A.get arr adjusted_idx
  | Uint4x32_nd arr -> (A.get arr adjusted_idx).Stdlib.Complex.re
  | Half_nd arr -> A.get arr adjusted_idx
  | Bfloat16_nd arr -> Ops.bfloat16_to_single @@ A.get arr adjusted_idx
  | Fp8_nd arr -> Ops.fp8_to_single @@ Char.to_int @@ A.get arr adjusted_idx
  | Single_nd arr -> A.get arr adjusted_idx
  | Double_nd arr -> A.get arr adjusted_idx

let retrieve_2d_points ?from_axis ?padding ~xdim ~ydim arr =
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
          get_as_float ?padding arr idx
        in
        let y =
          idx.(from_axis) <- ydim;
          get_as_float ?padding arr idx
        in
        result := (x, y) :: !result
      else if axis = from_axis then iter (axis + 1)
      else
        let end_idx = compute_end_idx ?padding dims axis in
        for p = 0 to end_idx do
          idx.(axis) <- p;
          iter (axis + 1)
        done
    in
    iter 0;
    Array.of_list_rev !result

let retrieve_1d_points ?from_axis ?padding ~xdim arr =
  let dims = dims arr in
  if Array.is_empty dims then [| get_as_float ?padding arr [||] |]
  else
    let n_axes = Array.length dims in
    let from_axis = Option.value from_axis ~default:(n_axes - 1) in
    let result = ref [] in
    let idx = Array.create ~len:n_axes 0 in
    let rec iter axis =
      if axis = n_axes then
        let x =
          idx.(from_axis) <- xdim;
          get_as_float ?padding arr idx
        in
        result := x :: !result
      else if axis = from_axis then iter (axis + 1)
      else
        let end_idx = compute_end_idx ?padding dims axis in
        for p = 0 to end_idx do
          idx.(axis) <- p;
          iter (axis + 1)
        done
    in
    iter 0;
    Array.of_list_rev !result

let retrieve_flat_values ?padding arr =
  let dims = dims arr in
  if Array.is_empty dims then [| get_as_float ?padding arr [||] |]
  else
    let n_axes = Array.length dims in
    let result = ref [] in
    let idx = Array.create ~len:n_axes 0 in
    let rec iter axis =
      if axis = n_axes then
        let x = get_as_float ?padding arr idx in
        result := x :: !result
      else
        let end_idx = compute_end_idx ?padding dims axis in
        for p = 0 to end_idx do
          idx.(axis) <- p;
          iter (axis + 1)
        done
    in
    iter 0;
    Array.of_list_rev !result

let set_flat_values ?padding arr values =
  let dims = dims arr in
  if Array.is_empty dims then set_from_float ?padding arr [| 0 |] values.(0)
  else
    let n_axes = Array.length dims in
    let idx = Array.create ~len:n_axes 0 in
    let values_idx = ref 0 in
    let rec iter axis =
      if axis = n_axes then (
        if !values_idx < Array.length values then (
          set_from_float ?padding arr idx values.(!values_idx);
          Int.incr values_idx))
      else
        let end_idx = compute_end_idx ?padding dims axis in
        for p = 0 to end_idx do
          idx.(axis) <- p;
          iter (axis + 1)
        done
    in
    iter 0

let c_ptr_to_string nd =
  let prec = get_prec nd in
  let f arr = Ops.c_rawptr_to_string (bigarray_start_not_managed arr) prec in
  apply { f } nd

let ptr_to_string_hum nd =
  let prec = get_prec nd in
  let f arr = Ops.rawptr_to_string_hum (bigarray_start_not_managed arr) prec in
  apply { f } nd

let to_native = apply { f = bigarray_start_not_managed }
let equal a1 a2 = equal_nativeint (to_native a1) (to_native a2)
let compare a1 a2 = compare_nativeint (to_native a1) (to_native a2)
let hash nd = Nativeint.hash (to_native nd)
let hash_fold_t acc nd = hash_fold_nativeint acc (to_native nd)
let hash_t nd = Nativeint.hash @@ to_native nd

(** Copies the whole of [source] onto the parts of [target] skipping over padding margins --
    requires that source dimensions + padding = target dimensions. *)
let copy_with_padding ~source ~target ~padding =
  let copy_impl source_arr target_arr = Ops.copy_with_padding_c source_arr target_arr padding in
  apply2 { f2 = copy_impl } source target

(** {2 *** Creating ***} *)

let used_memory = Atomic.make 0

let%track7_sexp create_array ~debug:(_debug : string) (prec : Ops.prec) ~(dims : int array) ~padding
    =
  (* dims already includes padding if padding is specified *)
  let size_in_bytes : int = Array.fold dims ~init:1 ~f:( * ) * Ops.prec_in_bytes prec in
  let%track7_sexp finalizer (_result : t) =
    let _ : int = Atomic.fetch_and_add used_memory size_in_bytes in
    [%log3 "Deleting", _debug, ptr_to_string_hum _result]
  in
  let f prec = as_array prec @@ create_bigarray prec ~dims ~padding in
  let result = Ops.apply_prec { f } prec in
  Stdlib.Gc.finalise finalizer result;
  let _ : int = Atomic.fetch_and_add used_memory size_in_bytes in
  [%debug3_sexp
    [%log_block
      "create_array";
      [%log _debug, ptr_to_string_hum result]]];
  result

(** See {!Bigarray.reshape}. *)
let reshape nd dims =
  let f prec arr = as_array prec @@ Bigarray.reshape arr dims in
  apply_with_prec { f } nd

let get_used_memory () = Atomic.get used_memory

(** {2 *** Printing ***} *)

(** Dimensions to string, ["x"]-separated, e.g. 1x2x3 for batch dims 1, input dims 3, output dims 2.
    Outputs ["-"] for empty dimensions. *)
let int_dims_to_string ?(with_axis_numbers = false) ?padding dims =
  if Array.is_empty dims then "-"
  else if with_axis_numbers then
    String.concat_array ~sep:" x "
    @@ Array.mapi dims ~f:(fun d s -> Int.to_string d ^ ":" ^ Int.to_string s)
  else
    let dim_strings =
      Array.mapi dims ~f:(fun i dim ->
          match padding with
          | None -> Int.to_string dim
          | Some padding_arr when i < Array.length padding_arr ->
              let unpadded_dim = dim - padding_arr.(i).Ops.left - padding_arr.(i).right in
              let total_padding = padding_arr.(i).left + padding_arr.(i).right in
              if total_padding > 0 then
                Int.to_string unpadded_dim ^ "+" ^ Int.to_string total_padding
              else Int.to_string dim
          | Some _ -> Int.to_string dim)
    in
    String.concat_array ~sep:"x" dim_strings

(** Logs information about the array on the default ppx_minidebug runtime, if
    [from_log_level > Utlis.settings.with_log_level]. *)
let log_debug_info ~from_log_level:_level _nd =
  [%debug_sexp
    [%at_log_level
      _level;
      [%log_block
        "Ndarray " ^ Sexp.to_string_hum (sexp_of_t _nd);
        [%log
          "value-at-0:",
          (get_as_float _nd (Array.map (dims _nd) ~f:(fun _ -> 0)) : float),
          "has nan:",
          (fold_as_float _nd ~init:false ~f:(fun has_nan _ v -> has_nan || Float.is_nan v) : bool),
          "has +inf:",
          (fold_as_float _nd ~init:false ~f:(fun has_inf _ v -> has_inf || Float.(v = infinity))
            : bool),
          "has -inf:",
          (fold_as_float _nd ~init:false ~f:(fun has_neg_inf _ v ->
               has_neg_inf || Float.(v = neg_infinity))
            : bool)]]]]

let concise_float ~prec v =
  (* The C99 standard requires at least two digits for the exponent, but the leading zero is a waste
     of space. Also handles e+0. String-based approach to avoid rounding issues I noticed on
     Windows. *)
  let s = Printf.sprintf "%.*e" (prec + 3) v in
  let s = Str.global_replace (Str.regexp "[0-9][0-9][0-9]e") "e" s in
  let s = Str.global_replace (Str.regexp "e[+-]0+$") "" s in
  let s = Str.global_replace (Str.regexp "e\\([+-]\\)0+\\([1-9]\\)") "e\\1\\2" s in
  s

(** Prints 0-based [indices] entries out of [arr], where a number between [-5] and [-1] in an axis
    means to print out the axis, and a non-negative number means to print out only the indexed
    dimension of the axis. Prints up to [entries_per_axis] or [entries_per_axis+1] entries per axis,
    possibly with ellipsis in the middle. [labels] provides the axis labels for all axes (use [""]
    or ["_"] for no label). The last label corresponds to axis [-1] etc. The printed out axes are
    arranged as:
    - [-1]: a horizontal segment in an inner rectangle (i.e. column numbers of the inner rectangle),
    - [-2]: a sequence of segments in a line of text (i.e. column numbers of an outer rectangle),
    - [-3]: a vertical segment in an inner rectangle (i.e. row numbers of the inner rectangle),
    - [-4]: a vertical sequence of segments (i.e. column numbers of an outer rectangle),
    - [-5]: a sequence of screens of text (i.e. stack numbers of outer rectangles). *)
let render_array ?(brief = false) ?(prefix = "") ?(entries_per_axis = 4) ?(labels = [||]) ~indices
    arr =
  let module B = PrintBox in
  let dims = dims arr in
  let has_nan = fold_as_float ~init:false ~f:(fun has_nan _ v -> has_nan || Float.is_nan v) arr in
  let has_inf =
    fold_as_float ~init:false ~f:(fun has_inf _ v -> has_inf || Float.(v = infinity)) arr
  in
  let has_neg_inf =
    fold_as_float ~init:false
      ~f:(fun has_neg_inf _ v -> has_neg_inf || Float.(v = neg_infinity))
      arr
  in
  let header =
    prefix
    ^ (if has_nan || has_inf || has_neg_inf then " includes" else "")
    ^ (if has_nan then " NaN" else "")
    ^ (if has_inf then " pos. inf." else "")
    ^ if has_neg_inf then " neg. inf." else ""
  in
  if Array.is_empty dims then
    B.vlist ~bars:false
      [
        B.text header;
        B.line
          ("scalar "
          ^ concise_float ~prec:Utils.settings.print_decimals_precision (get_as_float arr [||]));
      ]
  else
    let indices = Array.copy indices in
    let entries_per_axis =
      if entries_per_axis % 2 = 0 then entries_per_axis + 1 else entries_per_axis
    in
    let var_indices =
      Array.filter_mapi indices ~f:(fun i d -> if d <= -1 then Some (5 + d, i) else None)
    in
    let extra_indices =
      [| (0, -1); (1, -1); (2, -1); (3, -1); (4, -1) |]
      |> Array.filter
           ~f:(Fn.non @@ Array.mem var_indices ~equal:(fun (a, _) (b, _) -> Int.equal a b))
    in
    let var_indices = Array.append extra_indices var_indices in
    Array.sort ~compare:(fun (a, _) (b, _) -> Int.compare a b) var_indices;
    let var_indices = Array.map ~f:snd @@ var_indices in
    let ind0, ind1, ind2, ind3, ind4 =
      match var_indices with
      | [| ind0; ind1; ind2; ind3; ind4 |] -> (ind0, ind1, ind2, ind3, ind4)
      | _ -> raise @@ Utils.User_error "render: indices should contain at most 5 negative numbers"
    in
    let labels = Array.map labels ~f:(fun l -> if String.is_empty l then "" else l ^ "=") in
    let entries_per_axis = (entries_per_axis / 2 * 2) + 1 in
    let size0 = if ind0 = -1 then 1 else min dims.(ind0) entries_per_axis in
    let size1 = if ind1 = -1 then 1 else min dims.(ind1) entries_per_axis in
    let size2 = if ind2 = -1 then 1 else min dims.(ind2) entries_per_axis in
    let size3 = if ind3 = -1 then 1 else min dims.(ind3) entries_per_axis in
    let size4 = if ind4 = -1 then 1 else min dims.(ind4) entries_per_axis in
    let no_label ind = Array.length labels <= ind in
    let label0 = if ind0 = -1 || no_label ind0 then "" else labels.(ind0) in
    let label1 = if ind1 = -1 || no_label ind1 then "" else labels.(ind1) in
    let label2 = if ind2 = -1 || no_label ind2 then "" else labels.(ind2) in
    let label3 = if ind3 = -1 || no_label ind3 then "" else labels.(ind3) in
    let label4 = if ind4 = -1 || no_label ind4 then "" else labels.(ind4) in
    (* FIXME: handle ellipsis. *)
    let halfpoint = (entries_per_axis / 2) + 1 in
    let expand i ~ind =
      if dims.(ind) <= entries_per_axis then i
      else if i < halfpoint then i
      else dims.(ind) - entries_per_axis + i
    in
    let update_indices v i j k l =
      if ind0 <> -1 then indices.(ind0) <- expand v ~ind:ind0;
      if ind1 <> -1 then indices.(ind1) <- expand i ~ind:ind1;
      if ind2 <> -1 then indices.(ind2) <- expand j ~ind:ind2;
      if ind3 <> -1 then indices.(ind3) <- expand k ~ind:ind3;
      if ind4 <> -1 then indices.(ind4) <- expand l ~ind:ind4
    in
    let elide_for i ~ind = ind >= 0 && dims.(ind) > entries_per_axis && i + 1 = halfpoint in
    let is_ellipsis () = Array.existsi indices ~f:(fun ind i -> elide_for i ~ind) in
    let inner_grid v i j =
      B.init_grid ~bars:false ~line:size3 ~col:size4 (fun ~line ~col ->
          update_indices v i j line col;
          try
            B.hpad 1 @@ B.line
            @@
            if is_ellipsis () then "..."
            else
              concise_float ~prec:Utils.settings.print_decimals_precision (get_as_float arr indices)
          with Invalid_argument _ ->
            raise
            @@ Utils.User_error
                 [%string
                   "Invalid indices: %{int_dims_to_string indices} into array: \
                    %{(int_dims_to_string dims)}"])
    in
    let tag ?pos label ind =
      if ind = -1 then ""
      else
        match pos with
        | Some pos when elide_for pos ~ind -> "~~~~~"
        | Some pos when pos >= 0 ->
            Int.to_string (expand pos ~ind) ^ " @ " ^ label ^ Int.to_string ind
        | _ -> "axis " ^ label ^ Int.to_string ind
    in
    let nlines = if brief then size1 else size1 + 1 in
    let ncols = if brief then size2 else size2 + 1 in
    let outer_grid v =
      (if brief then Fn.id else B.frame ~stretch:false)
      @@ B.init_grid ~bars:true ~line:nlines ~col:ncols (fun ~line ~col ->
             if (not brief) && line = 0 && col = 0 then
               B.lines @@ List.filter ~f:(Fn.non String.is_empty) @@ [ tag ~pos:v label0 ind0 ]
             else if (not brief) && line = 0 then
               B.lines
               @@ List.filter ~f:(Fn.non String.is_empty)
               @@ [ tag ~pos:(col - 1) label2 ind2; tag label4 ind4 ]
             else if (not brief) && col = 0 then
               B.lines
               @@ List.filter ~f:(Fn.non String.is_empty)
               @@ [ tag ~pos:(line - 1) label1 ind1; tag label3 ind3 ]
             else
               let nline = if brief then line else line - 1 in
               let ncol = if brief then col else col - 1 in
               if elide_for ncol ~ind:ind2 || elide_for nline ~ind:ind1 then
                 B.hpad 1 @@ B.line "..."
               else inner_grid v nline ncol)
    in
    let screens =
      B.init_grid ~bars:true ~line:size0 ~col:1 (fun ~line ~col:_ ->
          if elide_for line ~ind:ind0 then B.hpad 1 @@ B.line "..." else outer_grid line)
    in
    (if brief then Fn.id else B.frame ~stretch:false)
    @@ B.vlist ~bars:false [ B.text header; screens ]

let to_doc ?prefix ?entries_per_axis ?labels ~indices arr =
  let box = render_array ?prefix ?entries_per_axis ?labels ~indices arr in
  PPrint.(separate hardline @@ lines (PrintBox_text.to_string box))

(** Prints the whole array in an inline syntax. *)
let to_doc_inline ~num_batch_axes ~num_output_axes ~num_input_axes ?axes_spec arr =
  let dims = dims arr in
  let num_all_axes = num_batch_axes + num_output_axes + num_input_axes in
  let open PPrint in
  let ind = Array.copy dims in
  let spec_doc =
    match axes_spec with None -> empty | Some spec -> dquotes (string spec) ^^ space
  in
  let rec loop axis =
    let sep =
      if axis < num_batch_axes then string ";"
      else if axis < num_batch_axes + num_output_axes then string ";"
      else string ","
    in
    let open_delim =
      if axis < num_batch_axes then string "[|"
      else if axis < num_batch_axes + num_output_axes then string "["
      else if axis = num_batch_axes + num_output_axes then empty
      else string "("
    in
    let close_delim =
      if axis < num_batch_axes then string "|]"
      else if axis < num_batch_axes + num_output_axes then string "]"
      else if axis = num_batch_axes + num_output_axes then empty
      else string ")"
    in
    if axis = num_all_axes then
      string (Printf.sprintf "%.*f" Utils.settings.print_decimals_precision (get_as_float arr ind))
    else
      group
        (open_delim
        ^^ nest 2
             (break 1
             ^^ separate_map
                  (break 1 ^^ sep ^^ space)
                  (fun i ->
                    ind.(axis) <- i;
                    loop (axis + 1))
                  (List.init dims.(axis) ~f:(fun i -> i)))
        ^^ break 1 ^^ close_delim)
  in
  spec_doc ^^ loop 0

(* TODO: restore npy support or alternative. *)

(* let save ~file_name t = let f arr = Npy.write arr file_name in map { f } t *)

(* let restore ~file_name t = let local = Npy.read_mmap file_name ~shared:false in let f prec arr =
   let local = Npy.to_bigarray Bigarray.c_layout (precision_to_bigarray_kind prec) local in A.blit
   (Option.value_exn ~here:[%here] local) arr in map_with_prec { f } t *)
