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

(** {2 *** Operations ***}

    See: {{https://github.com/tinygrad/tinygrad/blob/master/tinygrad/ops.py#L123} tinygrad ops},
    {{https://docs.nvidia.com/cuda/cuda-math-api/index.html} CUDA Math API} (intrinsics).

    This is a redundant set of operations, aiming to expose hardware-supported "intrinsics",
    to reduce the need for backends to pattern-match and optimize. Also for convenience.
*)

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

type binop =
  | Arg1
  | Arg2
  | Add
  | Sub
  | Mul
  | Div
  | ToPowOf
  | Relu_gate
  | Satur01_gate
  | Max
  | Min
  | Mod
  | Cmplt
  | Cmpeq
  (* Waiting till we have a use-case to see how to sensibly introduce bitwise operations. *)
  (* | Shl *)
  (* | Shr *)
  | Or
  | And
[@@deriving sexp, compare, equal]

type unop =
  | Identity
  | Relu
  | Satur01  (** Saturate (truncate) to within the interval [[0; 1]]. *)
  | Exp
  | Log
  | Exp2
  | Log2
  | Sin
  | Cos
  | Sqrt
  | Recip
  | Recip_sqrt
  | Neg
  | Tanh_approx
  | Not  (** 0. -> 1. | _ -> 0. *)
[@@deriving sexp, compare, equal]

type ternop = Where  (** Where(a,b,c): if a then b else c *) | FMA  (** FMA(a,b,c): (a * b) + c *)
[@@deriving sexp, compare, equal]

type op = Ternop of ternop | Binop of binop | Unop of unop [@@deriving sexp, compare, equal]

(** Either the left-neutral or right-neutral element of the operation. Unspecified if the operation
    does not have a neutral element. *)
let neutral_elem = function
  | Add | Sub -> 0.
  | Mul | Div -> 1.
  | ToPowOf -> 1.
  | Relu_gate -> 1.
  | Satur01_gate -> 0.5
  | Max -> Float.neg_infinity
  | Min -> Float.infinity
  | And -> 1.
  | Or -> 0.
  | Arg2 | Arg1 | Mod | Cmplt | Cmpeq (* | Shl | Shr *) -> 0.

let interpret_binop op v1 v2 =
  let open Float in
  match op with
  | Arg1 -> v1
  | Arg2 -> v2
  | Add -> v1 + v2
  | Sub -> v1 - v2
  | Mul -> v1 * v2
  | Div -> v1 / v2
  | ToPowOf when is_integer v2 -> int_pow v1 @@ to_int v2
  | ToPowOf -> v1 ** v2
  | Relu_gate -> if v1 > 0.0 then v2 else 0.0
  | Satur01_gate -> if v1 > 0.0 && v1 < 1.0 then v2 else 0.0
  | Max -> max v1 v2
  | Min -> min v1 v2
  | Mod -> v1 % v2
  | Cmplt -> if v1 < v2 then 1. else 0.
  | Cmpeq -> if v1 = v2 then 1. else 0.
  (* | Shl -> v1 * (int_pow 2. @@ to_int v2) *)
  (* | Shr -> v1 / (int_pow 2. @@ to_int v2) *)
  | Or -> if v1 <> 0. || v2 <> 0. then 1. else 0.
  | And -> if v1 <> 0. && v2 <> 0. then 1. else 0.

let interpret_unop op v =
  let open Float in
  match op with
  | Identity -> v
  | Relu when v >= 0. -> v
  | Relu -> 0.
  | Satur01 when v <= 0. -> 0.
  | Satur01 when v >= 1. -> 1.
  | Satur01 -> v
  | Exp -> exp v
  | Log -> log v
  | Exp2 -> 2. ** v
  | Log2 -> log v / log 2.
  | Sin -> sin v
  | Cos -> cos v
  | Sqrt -> sqrt v
  | Recip -> 1. / v
  | Recip_sqrt -> 1. / sqrt v
  | Neg -> ~-.v
  | Tanh_approx -> tanh v
  | Not -> if v = 0. then 1. else 0.

let interpret_ternop op v1 v2 v3 =
  let open Float in
  match op with Where -> if v1 <> 0. then v2 else v3 | FMA -> (v1 * v2) + v3

(** Note: currently the %cd syntax only supports infix binops as assignment ops. *)
let is_binop_infix _ = true

let is_binop_nice_infix = function
  | Arg1 | Arg2 | Relu_gate | Satur01_gate | Max | Min -> false
  | _ -> true

let binop_cd_syntax = function
  | Arg1 -> "-@>"
  | Arg2 -> "-/>"
  | Add -> "+"
  | Sub -> "-"
  | Mul -> "*"
  | Div -> "/"
  | ToPowOf -> "**"
  | Relu_gate -> "-?/"
  | Satur01_gate -> "-?^"
  | Cmplt -> "<"
  | Cmpeq -> "="
  | Or -> "||"
  | And -> "&&"
  | Mod -> "%"
  | Max -> "@^"
  | Min -> "^^"
(* | Shl -> "lsl" *)
(* | Shr -> "lsr" *)

(** In the %cd syntax, we support uncurried notation for binary ops in addition to the infix
    notation. *)
let binop_cd_fallback_syntax = function
  | Arg1 -> "fst"
  | Arg2 -> "snd"
  | Add -> "add"
  | Sub -> "sub"
  | Mul -> "mul"
  | Div -> "div"
  | ToPowOf -> "pow"
  | Relu_gate -> "relu_gate"
  | Satur01_gate -> "sat01_gate"
  | Cmplt -> "lt"
  | Cmpeq -> "eq"
  | Or -> "or_"
  | And -> "and_"
  | Mod -> "mod_"
  | Max -> "max"
  | Min -> "min"
(* | Shl -> "shlf" *)
(* | Shr -> "shrf" *)

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
  | ToPowOf, Byte_prec _ ->
      invalid_arg "Ops.binop_c_syntax: ToPowOf not supported for byte/integer precisions"
  | ToPowOf, _ -> ("powf(", ",", ")")
  | Relu_gate, Byte_prec _ -> ("(", " > 0 ?", " : 0)")
  | Relu_gate, _ -> ("(", " > 0.0 ?", " : 0.0)")
  | Satur01_gate, Byte_prec _ -> ("(abs(", " ) > 0 ? 0 : (", "))")
  | Satur01_gate, Single_prec _ -> ("(fabsf(truncf(", ")) > 0.0 ? 0.0 : (", "))")
  | Satur01_gate, _ -> ("(fabs(trunc(", ")) > 0.0 ? 0.0 : (", "))")
  | Max, (Double_prec _ | Byte_prec _) -> ("fmax(", ",", ")")
  | Max, _ -> ("fmaxf(", ",", ")")
  | Min, (Double_prec _ | Byte_prec _) -> ("fmin(", ",", ")")
  | Min, _ -> ("fminf(", ",", ")")
  | Mod, _ -> ("(", " %", ")")
  | Cmplt, _ -> ("(", " <", ")")
  | Cmpeq, _ -> ("(", " ==", ")")
  (* | Shl, Byte_prec _ -> ("(", " <<", ")") *)
  (* | Shl, _ -> ("((", ") * exp2(", "))") *)
  (* | Shr, Byte_prec _ -> ("(", " >>", ")") *)
  (* | Shr, _ -> ("((", ") / exp2(", "))") *)
  | Or, _ -> ("(", " ||", ")")
  | And, _ -> ("(", " &&", ")")

let is_assign_op = function
  | Arg1 | Mod (* | Shl | Shr *) | Cmplt | Cmpeq -> false
  | Add | Sub | Mul | Div | ToPowOf | Relu_gate | Satur01_gate | Arg2 | Max | Min | Or | And -> true

let assign_op_cd_syntax ~initialize_neutral = function
  | Arg2 -> "=:"
  | Add when initialize_neutral -> "=:+"
  | Sub when initialize_neutral -> "=:-"
  | Mul when initialize_neutral -> "=:*"
  | Div when initialize_neutral -> "=:/"
  | ToPowOf when initialize_neutral -> "=:**"
  | Relu_gate when initialize_neutral -> "=:?/"
  | Satur01_gate when initialize_neutral -> "=:?^"
  | Or when initialize_neutral -> "=:||"
  | And when initialize_neutral -> "=:&&"
  | Max when initialize_neutral -> "=:@^"
  | Min when initialize_neutral -> "=:^^"
  | Add -> "=+"
  | Sub -> "=-"
  | Mul -> "=*"
  | Div -> "=/"
  | ToPowOf -> "=**"
  | Relu_gate -> "=?/"
  | Satur01_gate -> "=?^"
  | Max -> "=@^"
  | Min -> "=^^"
  | Or -> "=||"
  | And -> "=&&"
  | Arg1 | Mod (* | Shl | Shr *) | Cmplt | Cmpeq ->
      invalid_arg "Ops.assign_op_cd_syntax: not an assignment op"

(** Note: currently we do not support unary prefix symbols. *)
let unop_cd_syntax = function
  | Identity -> "id"
  | Relu -> "relu"
  | Satur01 -> "sat01"
  | Exp -> "exp"
  | Log -> "log"
  | Exp2 -> "exp2"
  | Log2 -> "log2"
  | Sin -> "sin"
  | Cos -> "cos"
  | Sqrt -> "sqrt"
  | Recip -> "recip"
  | Recip_sqrt -> "recip_sqrt"
  | Neg -> "neg"
  | Tanh_approx -> "tanh"
  | Not -> "not"

let unop_c_syntax prec op =
  let fmax () =
    (* See: https://en.cppreference.com/w/c/numeric/math/fmax option (4) *)
    match prec with
    | Double_prec _ | Byte_prec _ -> "fmax"
    | _ -> "fmaxf"
  in
  let fmin () =
    match prec with
    | Double_prec _ | Byte_prec _ -> "fmin"
    | _ -> "fminf"
  in
  match (op, prec) with
  | Identity, _ -> ("", "")
  | Relu, Byte_prec _ -> ("fmax(0, ", ")")
  | Relu, _ -> (fmax () ^ "(0.0, ", ")")
  | Satur01, Byte_prec _ -> ("fmax(0, fmin(1, ", "))")
  | Satur01, _ -> (fmax () ^ "(0.0, " ^ fmin () ^ "(1.0, ", "))")
  | Exp, (Double_prec _ | Byte_prec _) -> ("exp(", ")")
  | Exp, _ -> ("expf(", ")")
  | Log, (Double_prec _ | Byte_prec _) -> ("log(", ")")
  | Log, _ -> ("logf(", ")")
  | Exp2, (Double_prec _ | Byte_prec _) -> ("exp2(", ")")
  | Exp2, _ -> ("exp2f(", ")")
  | Log2, (Double_prec _ | Byte_prec _) -> ("log2(", ")")
  | Log2, _ -> ("log2f(", ")")
  | Sin, (Double_prec _ | Byte_prec _) -> ("sin(", ")")
  | Sin, _ -> ("sinf(", ")")
  | Cos, (Double_prec _ | Byte_prec _) -> ("cos(", ")")
  | Cos, _ -> ("cosf(", ")")
  | Sqrt, (Double_prec _ | Byte_prec _) -> ("sqrt(", ")")
  | Sqrt, _ -> ("sqrtf(", ")")
  | Recip, Byte_prec _ ->
      invalid_arg "Ops.unop_c_syntax: Recip not supported for byte/integer precisions"
  | Recip, _ -> ("(1.0 / (", "))")
  | Recip_sqrt, Byte_prec _ ->
      invalid_arg "Ops.unop_c_syntax: Recip_sqrt not supported for byte/integer precisions"
  | Recip_sqrt, Double_prec _ -> ("(1.0 / sqrt(", "))")
  | Recip_sqrt, _ -> ("(1.0 / sqrtf(", "))")
  | Neg, _ -> ("(-(", "))")
  | Tanh_approx, Byte_prec _ ->
      invalid_arg "Ops.unop_c_syntax: Tanh_approx not supported for byte/integer precisions"
  | Tanh_approx, _ -> ("tanhf(", ")")
  | Not, _ -> ("(", " == 0.0 ? 1.0 : 0.0)")

(** In the %cd syntax, we use uncurried notation for ternary ops. *)
let ternop_cd_syntax = function Where -> "where" | FMA -> "fma"

let ternop_c_syntax prec op =
  match (op, prec) with
  | Where, Byte_prec _ -> ("((", ") != 0 ? (", ") : (", "))")
  | Where, _ -> ("((", ") != 0.0 ? (", ") : (", "))")
  | FMA, (Double_prec _ | Byte_prec _) -> ("fma(", ",", ",", ")")
  | FMA, _ -> ("fmaf(", ",", ",", ")")

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
