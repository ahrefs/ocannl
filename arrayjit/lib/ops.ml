open Base
(** Operation types shared by all backends; and precision types. *)

module Lazy = Utils.Lazy

(** {2 *** Precision ***} *)

type uint8_elt = Bigarray.int8_unsigned_elt
type float16_elt = Bigarray.float16_elt
type float32_elt = Bigarray.float32_elt
type float64_elt = Bigarray.float64_elt

type ('ocaml, 'impl) precision =
  | Byte : (char, uint8_elt) precision
  | Half : (float, float16_elt) precision
  | Single : (float, float32_elt) precision
  | Double : (float, float64_elt) precision
[@@deriving sexp_of]

type prec =
  | Void_prec
  | Byte_prec of (char, uint8_elt) precision
  | Half_prec of (float, float16_elt) precision
  | Single_prec of (float, float32_elt) precision
  | Double_prec of (float, float64_elt) precision

let byte = Byte_prec Byte
let half = Half_prec Half
let single = Single_prec Single
let double = Double_prec Double
let is_up_to_fp16 = function Half_prec _ | Byte_prec _ -> true | _ -> false

let sexp_of_prec = function
  | Void_prec -> Sexp.Atom "Void_prec"
  | Byte_prec _ -> Sexp.Atom "Byte_prec"
  | Half_prec _ -> Sexp.Atom "Half_prec"
  | Single_prec _ -> Sexp.Atom "Single_prec"
  | Double_prec _ -> Sexp.Atom "Double_prec"

let prec_of_sexp = function
  | Sexp.Atom "Void_prec" -> Void_prec
  | Sexp.Atom "Byte_prec" -> byte
  | Sexp.Atom "Half_prec" -> half
  | Sexp.Atom "Single_prec" -> single
  | Sexp.Atom "Double_prec" -> double
  | Sexp.List _ -> invalid_arg "prec_of_sexp: expected atom, found list"
  | Sexp.Atom s -> invalid_arg @@ "prec_of_sexp: unknown precision " ^ s

let precision_to_string (type ocaml elt_t) (prec : (ocaml, elt_t) precision) =
  match prec with Byte -> "byte" | Half -> "half" | Single -> "single" | Double -> "double"

let prec_string = function
  | Void_prec -> "void"
  | Byte_prec _ -> "byte"
  | Half_prec _ -> "half"
  | Single_prec _ -> "single"
  | Double_prec _ -> "double"

let equal_prec p1 p2 =
  match (p1, p2) with
  | Void_prec, Void_prec -> true
  | Byte_prec _, Byte_prec _ -> true
  | Half_prec _, Half_prec _ -> true
  | Single_prec _, Single_prec _ -> true
  | Double_prec _, Double_prec _ -> true
  | Void_prec, _ | Byte_prec _, _ | Half_prec _, _ | Single_prec _, _ | Double_prec _, _ -> false

let prec_in_bytes = function
  | Void_prec -> 0
  | Byte_prec _ -> 1
  | Half_prec _ -> 2
  | Single_prec _ -> 4
  | Double_prec _ -> 8

let promote_prec p1 p2 =
  match (p1, p2) with
  | Double_prec _, _ -> p1
  | _, Double_prec _ -> p2
  | Single_prec _, _ -> p1
  | _, Single_prec _ -> p2
  | Half_prec _, _ -> p1
  | _, Half_prec _ -> p2
  | Byte_prec _, _ -> p1
  | _, Byte_prec _ -> p2
  | Void_prec, Void_prec -> Void_prec

let pack_prec (type ocaml elt_t) (prec : (ocaml, elt_t) precision) =
  match prec with Byte -> byte | Half -> half | Single -> single | Double -> double

type 'r map_prec = { f : 'ocaml 'elt_t. ('ocaml, 'elt_t) precision -> 'r }

let map_prec ?default { f } = function
  | Void_prec ->
      Option.value_or_thunk default ~default:(fun () -> invalid_arg "map_prec: Void_prec")
  | Byte_prec Byte -> f Byte
  | Half_prec Half -> f Half
  | Single_prec Single -> f Single
  | Double_prec Double -> f Double
  | _ -> .

let c_typ_of_prec = function
  | Byte_prec _ -> "unsigned char"
  | Half_prec _ -> "_Float16"
  | Single_prec _ -> "float"
  | Double_prec _ -> "double"
  | Void_prec -> "void"

let hum_typ_of_prec = function
  | Byte_prec _ -> "byte"
  | Half_prec _ -> "half"
  | Single_prec _ -> "float"
  | Double_prec _ -> "double"
  | Void_prec -> "void"

(** {2 *** Operations ***} *)

(** Initializes or resets a array by filling in the corresponding numbers, at the appropriate
    precision. *)
type init_op =
  | Constant_fill of { values : float array; strict : bool }
      (** Fills in the numbers where the rightmost axis is contiguous. If [strict=true], loops over
          the provided values. *)
  | Range_over_offsets
      (** Fills in the offset number of each cell (i.e. how many cells away it is from the
          beginning). *)
  | Standard_uniform  (** Draws the values from U(0,1). *)
  | File_mapped of string * prec  (** Reads the data using [Unix.openfile] and [Unix.map_file]. *)
[@@deriving equal, sexp]

type binop = Add | Sub | Mul | Div | ToPowOf | Relu_gate | Arg2 | Arg1
[@@deriving sexp, compare, equal]

type unop = Identity | Relu [@@deriving sexp, compare, equal]

(** Either the left-neutral or right-neutral element of the operation. Unspecified if the operation
    does not have a neutral element. *)
let neutral_elem = function
  | Add | Sub -> 0.
  | Mul | Div -> 1.
  | ToPowOf -> 1.
  | Relu_gate -> 1.
  | Arg2 -> 0.
  | Arg1 -> 0.

let interpret_binop op v1 v2 =
  let open Float in
  match op with
  | Arg1 -> v1
  | Arg2 -> v2
  | Add -> v1 + v2
  | Sub -> v1 - v2
  | Mul -> v1 * v2
  | Div -> v1 / v2
  | ToPowOf -> if is_integer v2 then int_pow v1 @@ to_int v2 else v1 ** v2
  | Relu_gate -> if v1 > 0.0 then v2 else 0.0

let interpret_unop op v =
  let open Float in
  match op with Identity -> v | Relu when v >= 0. -> v | Relu -> 0.

let binop_cd_syntax = function
  | Arg1 -> "-@>"
  | Arg2 -> "-/>"
  | Add -> "+"
  | Sub -> "-"
  | Mul -> "*"
  | Div -> "/"
  | ToPowOf -> "**"
  | Relu_gate -> "-?/"

let binop_c_syntax prec v =
  match (v, prec) with
  | Arg1, _ -> invalid_arg "Ops.binop_c_syntax: Arg1 is not an operator"
  | Arg2, _ -> invalid_arg "Ops.binop_c_syntax: Arg2 is not an operator"
  | _, Void_prec -> invalid_arg "Ops.binop_c_syntax: Void precision"
  | Add, _ -> ("(", " +", ")")
  | Sub, _ -> ("(", " -", ")")
  | Mul, _ -> ("(", " *", ")")
  | Div, _ -> ("(", " /", ")")
  | ToPowOf, Double_prec _ -> ("pow(", ",", ")")
  | ToPowOf, Single_prec _ -> ("powf(", ",", ")")
  | ToPowOf, Half_prec _ -> ("powf(", ",", ")")
  | ToPowOf, Byte_prec _ ->
      invalid_arg "Ops.binop_c_syntax: ToPowOf not supported for byte/integer precisions"
  | Relu_gate, Byte_prec _ -> ("(", " > 0 ?", " : 0)")
  | Relu_gate, _ -> ("(", " > 0.0 ?", " : 0.0)")

let assign_op_cd_syntax ~initialize_neutral = function
  | Arg1 -> invalid_arg "Ops.assign_op_cd_syntax: Arg1 is not a %cd assignment operator"
  | Arg2 -> "=:"
  | Add when initialize_neutral -> "=:+"
  | Sub when initialize_neutral -> "=:-"
  | Mul when initialize_neutral -> "=:*"
  | Div when initialize_neutral -> "=:/"
  | ToPowOf when initialize_neutral -> "=:**"
  | Relu_gate when initialize_neutral -> "=:?/"
  | Add -> "=+"
  | Sub -> "=-"
  | Mul -> "=*"
  | Div -> "=/"
  | ToPowOf -> "=**"
  | Relu_gate -> "=?/"

let assign_op_c_syntax = function
  | Arg1 -> invalid_arg "Ops.assign_op_c_syntax: Arg1 is not a C assignment operator"
  | Arg2 -> "="
  | Add -> "+="
  | Sub -> "-="
  | Mul -> "*="
  | Div -> "/="
  | ToPowOf -> invalid_arg "Ops.assign_op_c_syntax: ToPowOf function is not a C assignment operator"
  | Relu_gate -> invalid_arg "Ops.assign_op_c_syntax: Relu_gate is not a C assignment operator"

let unop_cd_syntax = function Identity -> "~=" | Relu -> "?/"

let unop_c_syntax prec v =
  match (v, prec) with
  | Identity, _ -> ("", "")
  | Relu, Single_prec _ -> ("fmaxf(0.0, ", ")")
  | Relu, Byte_prec _ -> ("fmax(0, ", ")")
  | Relu, _ -> ("fmax(0.0, ", ")")

let c_convert_precision ~from ~to_ =
  match (from, to_) with
  | Double_prec _, Double_prec _
  | Single_prec _, Single_prec _
  | Half_prec _, Half_prec _
  | Byte_prec _, Byte_prec _
  | Void_prec, Void_prec ->
      ("", "")
  | _ -> ("(" ^ c_typ_of_prec to_ ^ ")(", ")")

(** {2 *** Global references ***} *)

type voidptr = unit Ctypes.ptr

let sexp_of_voidptr p = Sexp.Atom Ctypes.(string_of (ptr void) p)
let compare_voidptr = Ctypes.ptr_compare
let equal_voidptr : voidptr -> voidptr -> bool = phys_equal

let c_rawptr_to_string (ptr : nativeint) prec =
  "(" ^ c_typ_of_prec prec ^ "*)" ^ Nativeint.Hex.to_string ptr

let rawptr_to_string_hum (ptr : nativeint) prec =
  "(" ^ hum_typ_of_prec prec ^ "*)" ^ Nativeint.Hex.to_string ptr

let c_ptr_to_string (type elem) (ptr : elem Ctypes.ptr) prec =
  c_rawptr_to_string (Ctypes.raw_address_of_ptr @@ Ctypes.to_voidp ptr) prec

let ptr_to_string_hum (type elem) (ptr : elem Ctypes.ptr) prec =
  rawptr_to_string_hum (Ctypes.raw_address_of_ptr @@ Ctypes.to_voidp ptr) prec

type global_identifier =
  | C_function of string  (** Calls a no-argument or indices-arguments C function. *)
  | External_unsafe of {
      ptr : voidptr;
      prec : (prec[@equal.ignore] [@compare.ignore]);
      dims : int array Lazy.t;
    }
  | Merge_buffer of { source_node_id : int }
      (** Each device has at most one merge buffer, which is re-used, and re-allocated as needed, by
          merge operations. The merge buffer is associated with the source node of the device's most
          recent [device_to_device ~into_merge_buffer:true] operation. *)
[@@deriving sexp_of, equal, compare]
