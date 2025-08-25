open Base
(** Operation types shared by all backends; and precision types. *)

module Lazy = Utils.Lazy

(** {2 *** Precision ***} *)

type uint8_elt = Bigarray.int8_unsigned_elt
type uint16_elt = Bigarray.int16_unsigned_elt
type int32_elt = Bigarray.int32_elt
type float16_elt = Bigarray.float16_elt
type float32_elt = Bigarray.float32_elt
type float64_elt = Bigarray.float64_elt
type int64_elt = Bigarray.int64_elt

type ('ocaml, 'impl) precision =
  | Byte : (char, uint8_elt) precision
  | Uint16 : (int, uint16_elt) precision
  | Int32 : (int32, int32_elt) precision
  | Int64 : (int64, int64_elt) precision
  | Uint4x32 : (Stdlib.Complex.t, Bigarray.complex64_elt) precision
      (** A 128-bit value that corresponds to e.g. CUDA's uint4 type. Luckily, the OCaml Bigarray
          library supports complex64_elt which is a 128-bit value, so we avoid dims conversions. *)
  | Half : (float, float16_elt) precision
  | Bfloat16 : (int, uint16_elt) precision  (** Using uint16 representation for now *)
  | Fp8 : (char, uint8_elt) precision  (** Using uint8 representation for now *)
  | Single : (float, float32_elt) precision
  | Double : (float, float64_elt) precision
[@@deriving sexp_of]

type prec =
  | Void_prec
  | Byte_prec of (char, uint8_elt) precision
  | Uint16_prec of (int, uint16_elt) precision
  | Int32_prec of (int32, int32_elt) precision
  | Int64_prec of (int64, int64_elt) precision
  | Uint4x32_prec of (Stdlib.Complex.t, Bigarray.complex64_elt) precision
  | Half_prec of (float, float16_elt) precision
  | Bfloat16_prec of (int, uint16_elt) precision
  | Fp8_prec of (char, uint8_elt) precision
  | Single_prec of (float, float32_elt) precision
  | Double_prec of (float, float64_elt) precision

let byte = Byte_prec Byte
let uint16 = Uint16_prec Uint16
let int32 = Int32_prec Int32
let int64 = Int64_prec Int64
let uint4x32 = Uint4x32_prec Uint4x32
let half = Half_prec Half
let bfloat16 = Bfloat16_prec Bfloat16
let fp8 = Fp8_prec Fp8
let single = Single_prec Single
let double = Double_prec Double

let is_up_to_fp16 = function
  | Half_prec _ | Byte_prec _ | Fp8_prec _ -> true
  | _ (* includes Bfloat16_prec *) -> false

let exceeds_fp16_cutoff c =
  match Utils.settings.check_half_prec_constants_cutoff with
  | None -> false
  | Some cutoff -> Float.(abs c >= cutoff)

let sexp_of_prec = function
  | Void_prec -> Sexp.Atom "Void_prec"
  | Byte_prec _ -> Sexp.Atom "Byte_prec"
  | Uint16_prec _ -> Sexp.Atom "Uint16_prec"
  | Int32_prec _ -> Sexp.Atom "Int32_prec"
  | Int64_prec _ -> Sexp.Atom "Int64_prec"
  | Uint4x32_prec _ -> Sexp.Atom "Uint4x32_prec"
  | Half_prec _ -> Sexp.Atom "Half_prec"
  | Bfloat16_prec _ -> Sexp.Atom "Bfloat16_prec"
  | Fp8_prec _ -> Sexp.Atom "Fp8_prec"
  | Single_prec _ -> Sexp.Atom "Single_prec"
  | Double_prec _ -> Sexp.Atom "Double_prec"

let prec_of_sexp = function
  | Sexp.Atom "Void_prec" -> Void_prec
  | Sexp.Atom "Byte_prec" -> byte
  | Sexp.Atom "Uint16_prec" -> uint16
  | Sexp.Atom "Int32_prec" -> int32
  | Sexp.Atom "Int64_prec" -> int64
  | Sexp.Atom "Uint4x32_prec" -> uint4x32
  | Sexp.Atom "Half_prec" -> half
  | Sexp.Atom "Bfloat16_prec" -> bfloat16
  | Sexp.Atom "Fp8_prec" -> fp8
  | Sexp.Atom "Single_prec" -> single
  | Sexp.Atom "Double_prec" -> double
  | Sexp.List _ -> invalid_arg "prec_of_sexp: expected atom, found list"
  | Sexp.Atom s -> invalid_arg @@ "prec_of_sexp: unknown precision " ^ s

let precision_to_string (type ocaml elt_t) (prec : (ocaml, elt_t) precision) =
  match prec with
  | Byte -> "byte"
  | Uint16 -> "uint16"
  | Int32 -> "int32"
  | Int64 -> "int64"
  | Uint4x32 -> "uint4x32"
  | Half -> "half"
  | Bfloat16 -> "bfloat16"
  | Fp8 -> "fp8"
  | Single -> "single"
  | Double -> "double"

let prec_string = function
  | Void_prec -> "void"
  | Byte_prec _ -> "byte"
  | Uint16_prec _ -> "uint16"
  | Int32_prec _ -> "int32"
  | Int64_prec _ -> "int64"
  | Uint4x32_prec _ -> "uint4x32"
  | Half_prec _ -> "half"
  | Bfloat16_prec _ -> "bfloat16"
  | Fp8_prec _ -> "fp8"
  | Single_prec _ -> "single"
  | Double_prec _ -> "double"

let prec_of_string s = prec_of_sexp (Sexp.Atom (String.(capitalize @@ lowercase s) ^ "_prec"))

let equal_prec p1 p2 =
  match (p1, p2) with
  | Void_prec, Void_prec -> true
  | Byte_prec _, Byte_prec _ -> true
  | Uint16_prec _, Uint16_prec _ -> true
  | Int32_prec _, Int32_prec _ -> true
  | Int64_prec _, Int64_prec _ -> true
  | Uint4x32_prec _, Uint4x32_prec _ -> true
  | Half_prec _, Half_prec _ -> true
  | Bfloat16_prec _, Bfloat16_prec _ -> true
  | Fp8_prec _, Fp8_prec _ -> true
  | Single_prec _, Single_prec _ -> true
  | Double_prec _, Double_prec _ -> true
  | Void_prec, _
  | Byte_prec _, _
  | Uint16_prec _, _
  | Int32_prec _, _
  | Int64_prec _, _
  | Uint4x32_prec _, _
  | Half_prec _, _
  | Bfloat16_prec _, _
  | Fp8_prec _, _
  | Single_prec _, _
  | Double_prec _, _ ->
      false

let compare_prec p1 p2 =
  match (p1, p2) with
  | Void_prec, Void_prec -> 0
  | Byte_prec _, Byte_prec _ -> 0
  | Uint16_prec _, Uint16_prec _ -> 0
  | Int32_prec _, Int32_prec _ -> 0
  | Int64_prec _, Int64_prec _ -> 0
  | Uint4x32_prec _, Uint4x32_prec _ -> 0
  | Half_prec _, Half_prec _ -> 0
  | Bfloat16_prec _, Bfloat16_prec _ -> 0
  | Fp8_prec _, Fp8_prec _ -> 0
  | Single_prec _, Single_prec _ -> 0
  | Double_prec _, Double_prec _ -> 0
  | Void_prec, _ -> -1
  | _, Void_prec -> 1
  | Byte_prec _, _ -> -1
  | _, Byte_prec _ -> 1
  | Uint16_prec _, _ -> -1
  | _, Uint16_prec _ -> 1
  | Int32_prec _, _ -> -1
  | _, Int32_prec _ -> 1
  | Int64_prec _, _ -> -1
  | _, Int64_prec _ -> 1
  | Uint4x32_prec _, _ -> -1
  | _, Uint4x32_prec _ -> 1
  | Half_prec _, _ -> -1
  | _, Half_prec _ -> 1
  | Bfloat16_prec _, _ -> -1
  | _, Bfloat16_prec _ -> 1
  | Fp8_prec _, _ -> -1
  | _, Fp8_prec _ -> 1
  | Single_prec _, _ -> -1
  | _, Single_prec _ -> 1

let prec_in_bytes = function
  | Void_prec -> 0
  | Byte_prec _ -> 1
  | Uint16_prec _ -> 2
  | Int32_prec _ -> 4
  | Int64_prec _ -> 8
  | Uint4x32_prec _ -> 16
  | Half_prec _ -> 2
  | Bfloat16_prec _ -> 2
  | Fp8_prec _ -> 1
  | Single_prec _ -> 4
  | Double_prec _ -> 8

(** Prefer precision which is more likely to remain functional in the resulting computations.
    uint4x32 always dominates, because operations that work on uint4x32 do not support other
    precisions. Otherwise, fractional number precisions dominate; within them, larger dynamic range
    precisions dominate. *)
let promote_prec p1 p2 =
  match (p1, p2) with
  | Uint4x32_prec _, _ -> p1
  | _, Uint4x32_prec _ -> p2
  | Double_prec _, _ -> p1
  | _, Double_prec _ -> p2
  | Single_prec _, _ -> p1
  | _, Single_prec _ -> p2
  | Bfloat16_prec _, _ -> p1
  | _, Bfloat16_prec _ -> p2
  | Half_prec _, _ -> p1
  | _, Half_prec _ -> p2
  | Fp8_prec _, _ -> p1
  | _, Fp8_prec _ -> p2
  | Int64_prec _, _ -> p1
  | _, Int64_prec _ -> p2
  | Int32_prec _, _ -> p1
  | _, Int32_prec _ -> p2
  | Uint16_prec _, _ -> p1
  | _, Uint16_prec _ -> p2
  | Byte_prec _, _ -> p1
  | _, Byte_prec _ -> p2
  | Void_prec, Void_prec -> Void_prec

let pack_prec (type ocaml elt_t) (prec : (ocaml, elt_t) precision) =
  match prec with
  | Byte -> byte
  | Uint16 -> uint16
  | Int32 -> int32
  | Int64 -> int64
  | Uint4x32 -> uint4x32
  | Half -> half
  | Bfloat16 -> bfloat16
  | Fp8 -> fp8
  | Single -> single
  | Double -> double

type 'r apply_prec = { f : 'ocaml 'elt_t. ('ocaml, 'elt_t) precision -> 'r }

let apply_prec ?default { f } = function
  | Void_prec ->
      Option.value_or_thunk default ~default:(fun () -> invalid_arg "apply_prec: Void_prec")
  | Byte_prec Byte -> f Byte
  | Byte_prec Fp8 -> invalid_arg "apply_prec: Fp8 is not a valid Byte precision"
  | Byte_prec _ -> .
  | Uint16_prec Uint16 -> f Uint16
  | Uint16_prec Bfloat16 -> invalid_arg "apply_prec: Bfloat16 is not a valid Uint16 precision"
  | Uint16_prec _ -> .
  | Int32_prec Int32 -> f Int32
  | Int32_prec _ -> .
  | Int64_prec Int64 -> f Int64
  | Int64_prec _ -> .
  | Half_prec Half -> f Half
  | Half_prec _ -> .
  | Bfloat16_prec Bfloat16 -> f Bfloat16
  | Bfloat16_prec Uint16 -> invalid_arg "apply_prec: Uint16 is not a valid Bfloat16 precision"
  | Bfloat16_prec _ -> .
  | Fp8_prec Fp8 -> f Fp8
  | Fp8_prec Byte -> invalid_arg "apply_prec: Byte is not a valid Fp8 precision"
  | Fp8_prec _ -> .
  | Single_prec Single -> f Single
  | Single_prec _ -> .
  | Double_prec Double -> f Double
  | Double_prec _ -> .
  | Uint4x32_prec Uint4x32 -> f Uint4x32
  | Uint4x32_prec _ -> .

let c_typ_of_prec = function
  | Byte_prec _ -> "unsigned char"
  | Uint16_prec _ -> "unsigned short"
  | Int32_prec _ -> "int"
  | Int64_prec _ -> "long long"
  | Uint4x32_prec _ -> "uint4x32_t" (* Note that both CUDA and Metal usa a native type uint4 here *)
  | Half_prec _ -> "HALF_T"
  | Bfloat16_prec _ -> "unsigned short" (* Bfloat16 represented as uint16 *)
  | Fp8_prec _ -> "unsigned char" (* FP8 represented as uint8 *)
  | Single_prec _ -> "float"
  | Double_prec _ -> "double"
  | Void_prec -> "void"

let c_vec_typ_of_prec ~length prec =
  match (prec, length) with
  | Single_prec _, 4 -> "float4_t"
  | Double_prec _, 2 -> "double2_t"
  | Int32_prec _, 4 -> "int32x4_t"
  | Int64_prec _, 2 -> "int64x2_t"
  | (Byte_prec _ | Fp8_prec _), 16 -> "int8x16_t"
  | (Uint16_prec _ | Bfloat16_prec _), 8 -> "uint16x8_t"
  | Half_prec _, 8 -> "half8_t"
  | _, 1 -> c_typ_of_prec prec
  | _ -> invalid_arg "Ops.c_vec_typ_of_prec: invalid combination"

let hum_typ_of_prec = function
  | Byte_prec _ -> "byte"
  | Uint16_prec _ -> "uint16"
  | Int32_prec _ -> "int32"
  | Int64_prec _ -> "int64"
  | Uint4x32_prec _ -> "uint4x32"
  | Half_prec _ -> "half"
  | Bfloat16_prec _ -> "bfloat16"
  | Fp8_prec _ -> "fp8"
  | Single_prec _ -> "float"
  | Double_prec _ -> "double"
  | Void_prec -> "void"

(** {2 *** Operations ***}

    See: {{:https://github.com/tinygrad/tinygrad/blob/master/tinygrad/ops.py#L123} tinygrad ops},
    {{:https://docs.nvidia.com/cuda/cuda-math-api/index.html} CUDA Math API} (intrinsics).

    This is a redundant set of operations, aiming to expose hardware-supported "intrinsics", to
    reduce the need for backends to pattern-match and optimize. Also for convenience. *)

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
  | Cmpne
  (* Waiting till we have a use-case to see how to sensibly introduce bitwise operations. *)
  (* | Shl *)
  (* | Shr *)
  | Or
  | And
  | Threefry4x32_crypto
      (** 4x32-bit Threefry PRNG, 20-round cryptographic version. Requires a 128-bit key and a
          128-bit counter and outputs a 128-bit value (precision [Uint4x32]). *)
  | Threefry4x32_light
      (** 4x32-bit Threefry PRNG, 2-round light version (as in JAX/XLA). Requires a 128-bit key and
          a 128-bit counter and outputs a 128-bit value (precision [Uint4x32]). *)
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
  | Uint4x32_to_prec_uniform1
      (** Non-vectorized variant of [Uint4x32_to_prec_uniform] that converts the given Uint4x32 to a
          single value of the output precision. Less bit-efficient but operates poitwise. For random
          bits, the result is uniform over the range of the precision for integer precisions, and
          over the range \[0.0, 1.0) for floating point precisions. *)
[@@deriving sexp, compare, equal]

type vec_unop =
  | Uint4x32_to_prec_uniform
      (** Converts the given Uint4x32 to the precision of the output in a bit-efficient manner. For
          random bits, the result is uniform over the range of the precision for integer precisions,
          and over the range \[0.0, 1.0) for floating point precisions. When used in an access
          pattern, the indices are converted to a byte offset depending on the given precision.
          NOTE: this operation, unlike any others, impacts projections and shape inference (one
          input cell corresponds to a few output cells). *)
[@@deriving sexp, compare, equal]

type ternop =
  | Where  (** Where(a,b,c): if a then b else c *)
  | FMA  (** FMA(a,b,c): (a * b) + c, non-accumulating *)
[@@deriving sexp, compare, equal]

type op = Ternop of ternop | Binop of binop | Unop of unop [@@deriving sexp, compare, equal]

(** Either the left-neutral or right-neutral element of the operation. Unspecified if the operation
    does not have a neutral element. *)
let neutral_elem = function
  | Add -> 0.
  | Sub -> 0.
  | Mul -> 1.
  | Div -> 1.
  | ToPowOf -> 1.
  | Relu_gate -> 1.
  | Satur01_gate -> 0.5
  | Max -> Float.neg_infinity
  | Min -> Float.infinity
  | And -> 1.
  | Or -> 0.
  | Arg2 | Arg1 | Mod | Cmplt | Cmpeq | Cmpne | Threefry4x32_crypto
  | Threefry4x32_light (* | Shl | Shr *) ->
      0.

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
  | Cmpne -> if v1 <> v2 then 1. else 0.
  (* | Shl -> v1 * (int_pow 2. @@ to_int v2) *)
  (* | Shr -> v1 / (int_pow 2. @@ to_int v2) *)
  | Or -> if v1 <> 0. || v2 <> 0. then 1. else 0.
  | And -> if v1 <> 0. && v2 <> 0. then 1. else 0.
  | Threefry4x32_crypto | Threefry4x32_light ->
      invalid_arg "Ops.interpret_binop: Threefry4x32 operations are outside the domain of float"

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
  | Uint4x32_to_prec_uniform1 ->
      invalid_arg
        "Ops.interpret_unop: Uint4x32_to_prec_uniform1 argument outside the domain of float"

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
  | Cmpne -> "<>"
  | Or -> "||"
  | And -> "&&"
  | Mod -> "%"
  | Max -> "@^"
  | Min -> "@-"
  | Threefry4x32_crypto -> "^^^^"
  | Threefry4x32_light -> "^^"
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
  | Cmpne -> "ne"
  | Or -> "or_"
  | And -> "and_"
  | Mod -> "mod_"
  | Max -> "max"
  | Min -> "min"
  | Threefry4x32_crypto -> "threefry4x32_crypto"
  | Threefry4x32_light -> "threefry4x32_light"
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
  | ToPowOf, (Byte_prec _ | Uint16_prec _ | Int32_prec _ | Fp8_prec _) ->
      invalid_arg "Ops.binop_c_syntax: ToPowOf not supported for integer precisions"
  | ToPowOf, _ -> ("powf(", ",", ")")
  | Relu_gate, (Byte_prec _ | Uint16_prec _ | Int32_prec _ | Fp8_prec _) -> ("(", " > 0 ?", " : 0)")
  | Relu_gate, _ -> ("(", " > 0.0 ?", " : 0.0)")
  | Satur01_gate, (Byte_prec _ | Uint16_prec _ | Int32_prec _ | Fp8_prec _) ->
      ("(abs(", " ) > 0 ? 0 : (", "))")
  | Satur01_gate, Single_prec _ ->
      (* This disagrees at 0 with the semantics. *)
      ("(fabsf(floorf(", ")) > 0.0 ? 0.0 : (", "))")
  | Satur01_gate, _ -> ("(fabs(floor(", ")) > 0.0 ? 0.0 : (", "))")
  | Max, (Double_prec _ | Byte_prec _ | Uint16_prec _ | Int32_prec _ | Fp8_prec _) ->
      ("fmax(", ",", ")")
  | Max, _ -> ("fmaxf(", ",", ")")
  | Min, (Double_prec _ | Byte_prec _ | Uint16_prec _ | Int32_prec _ | Fp8_prec _) ->
      ("fmin(", ",", ")")
  | Min, _ -> ("fminf(", ",", ")")
  | Mod, _ -> ("(", " %", ")")
  | Cmplt, _ -> ("(", " <", ")")
  | Cmpeq, _ -> ("(", " ==", ")")
  | Cmpne, _ -> ("(", " !=", ")")
  (* | Shl, Byte_prec _ -> ("(", " <<", ")") *)
  (* | Shl, _ -> ("((", ") * exp2(", "))") *)
  (* | Shr, Byte_prec _ -> ("(", " >>", ")") *)
  (* | Shr, _ -> ("((", ") / exp2(", "))") *)
  | Or, _ -> ("(", " ||", ")")
  | And, _ -> ("(", " &&", ")")
  | Threefry4x32_crypto, _ ->
      (* This corresponds to the pure C implementation in builtins.c. *)
      ("arrayjit_threefry4x32_crypto(", ",", ")")
  | Threefry4x32_light, _ -> ("arrayjit_threefry4x32_light(", ",", ")")

let is_assign_op = function
  | Arg1 | Mod | Threefry4x32_crypto | Threefry4x32_light (* | Shl | Shr *) | Cmplt | Cmpeq | Cmpne
    ->
      false
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
  | Min when initialize_neutral -> "=:@-"
  | Add -> "=+"
  | Sub -> "=-"
  | Mul -> "=*"
  | Div -> "=/"
  | ToPowOf -> "=**"
  | Relu_gate -> "=?/"
  | Satur01_gate -> "=?^"
  | Max -> "=@^"
  | Min -> "=@-"
  | Or -> "=||"
  | And -> "=&&"
  | Arg1 | Mod | Threefry4x32_crypto | Threefry4x32_light (* | Shl | Shr *) | Cmplt | Cmpeq | Cmpne
    ->
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
  | Uint4x32_to_prec_uniform1 -> "uint4x32_to_prec_uniform1"

let vec_unop_cd_syntax = function Uint4x32_to_prec_uniform -> "uint4x32_to_prec_uniform"

let unop_c_syntax prec op =
  let fmax () =
    (* See: https://en.cppreference.com/w/c/numeric/math/fmax option (4) *)
    match prec with
    | Double_prec _ | Byte_prec _ | Uint16_prec _ | Int32_prec _ | Fp8_prec _ -> "fmax"
    | _ -> "fmaxf"
  in
  let fmin () =
    match prec with
    | Double_prec _ | Byte_prec _ | Uint16_prec _ | Int32_prec _ | Fp8_prec _ -> "fmin"
    | _ -> "fminf"
  in
  match (op, prec) with
  | Identity, _ -> ("", "")
  | Relu, (Byte_prec _ | Uint16_prec _ | Int32_prec _ | Fp8_prec _) -> ("fmax(0, ", ")")
  | Relu, _ -> (fmax () ^ "(0.0, ", ")")
  | Satur01, (Byte_prec _ | Uint16_prec _ | Int32_prec _ | Fp8_prec _) -> ("fmax(0, fmin(1, ", "))")
  | Satur01, _ -> (fmax () ^ "(0.0, " ^ fmin () ^ "(1.0, ", "))")
  | Exp, (Double_prec _ | Byte_prec _ | Uint16_prec _ | Int32_prec _ | Fp8_prec _) -> ("exp(", ")")
  | Exp, _ -> ("expf(", ")")
  | Log, (Double_prec _ | Byte_prec _ | Uint16_prec _ | Int32_prec _ | Fp8_prec _) -> ("log(", ")")
  | Log, _ -> ("logf(", ")")
  | Exp2, (Double_prec _ | Byte_prec _ | Uint16_prec _ | Int32_prec _ | Fp8_prec _) -> ("exp2(", ")")
  | Exp2, _ -> ("exp2f(", ")")
  | Log2, (Double_prec _ | Byte_prec _ | Uint16_prec _ | Int32_prec _ | Fp8_prec _) -> ("log2(", ")")
  | Log2, _ -> ("log2f(", ")")
  | Sin, (Double_prec _ | Byte_prec _ | Uint16_prec _ | Int32_prec _ | Fp8_prec _) -> ("sin(", ")")
  | Sin, _ -> ("sinf(", ")")
  | Cos, (Double_prec _ | Byte_prec _ | Uint16_prec _ | Int32_prec _ | Fp8_prec _) -> ("cos(", ")")
  | Cos, _ -> ("cosf(", ")")
  | Sqrt, (Double_prec _ | Byte_prec _ | Uint16_prec _ | Int32_prec _ | Fp8_prec _) -> ("sqrt(", ")")
  | Sqrt, _ -> ("sqrtf(", ")")
  | Recip, (Byte_prec _ | Uint16_prec _ | Int32_prec _ | Fp8_prec _) ->
      invalid_arg "Ops.unop_c_syntax: Recip not supported for integer precisions"
  | Recip, _ -> ("(1.0 / (", "))")
  | Recip_sqrt, (Byte_prec _ | Uint16_prec _ | Int32_prec _ | Fp8_prec _) ->
      invalid_arg "Ops.unop_c_syntax: Recip_sqrt not supported for integer precisions"
  | Recip_sqrt, Double_prec _ -> ("(1.0 / sqrt(", "))")
  | Recip_sqrt, _ -> ("(1.0 / sqrtf(", "))")
  | Neg, _ -> ("(-(", "))")
  | Tanh_approx, (Byte_prec _ | Uint16_prec _ | Int32_prec _ | Fp8_prec _) ->
      invalid_arg "Ops.unop_c_syntax: Tanh_approx not supported for integer precisions"
  | Tanh_approx, _ -> ("tanhf(", ")")
  | Not, _ -> ("(", " == 0.0 ? 1.0 : 0.0)")
  | Uint4x32_to_prec_uniform1, Uint4x32_prec _ ->
      invalid_arg "Ops.vec_unop_c_syntax: Uint4x32_to_prec_uniform1 not supported for Uint4x32"
  | Uint4x32_to_prec_uniform1, _ -> ("uint4x32_to_" ^ prec_string prec ^ "_uniform(", ")")

let vec_unop_c_syntax prec op =
  match (op, prec) with
  | Uint4x32_to_prec_uniform, Uint4x32_prec _ ->
      invalid_arg "Ops.vec_unop_c_syntax: Uint4x32_to_prec_uniform not supported for Uint4x32"
  | Uint4x32_to_prec_uniform, _ -> ("uint4x32_to_" ^ prec_string prec ^ "_uniform_vec(", ")")

(** In the %cd syntax, we use uncurried notation for ternary ops. *)
let ternop_cd_syntax = function Where -> "where" | FMA -> "fma"

let ternop_c_syntax prec op =
  match (op, prec) with
  | Where, (Byte_prec _ | Uint16_prec _ | Int32_prec _ | Fp8_prec _) ->
      ("((", ") != 0 ? (", ") : (", "))")
  | Where, _ -> ("((", ") != 0.0 ? (", ") : (", "))")
  | FMA, (Double_prec _ | Byte_prec _ | Uint16_prec _ | Int32_prec _ | Fp8_prec _) ->
      ("fma(", ",", ",", ")")
  | FMA, _ -> ("fmaf(", ",", ",", ")")

let c_convert_precision ~from ~to_ =
  match (from, to_) with
  | Double_prec _, Double_prec _
  | Single_prec _, Single_prec _
  | Half_prec _, Half_prec _
  | Byte_prec _, Byte_prec _
  | Uint16_prec _, Uint16_prec _
  | Int32_prec _, Int32_prec _
  | Uint4x32_prec _, Uint4x32_prec _
  | Bfloat16_prec _, Bfloat16_prec _
  | Fp8_prec _, Fp8_prec _
  | Void_prec, Void_prec ->
      ("", "")
  (* BFloat16 conversions *)
  | Bfloat16_prec _, Single_prec _ -> ("bfloat16_to_single(", ")")
  | Single_prec _, Bfloat16_prec _ -> ("single_to_bfloat16(", ")")
  | Bfloat16_prec _, Double_prec _ -> ("(double)bfloat16_to_single(", ")")
  | Double_prec _, Bfloat16_prec _ -> ("single_to_bfloat16((float)", ")")
  (* FP8 conversions *)
  | Fp8_prec _, Single_prec _ -> ("fp8_to_single(", ")")
  | Single_prec _, Fp8_prec _ -> ("single_to_fp8(", ")")
  | Fp8_prec _, Double_prec _ -> ("(double)fp8_to_single(", ")")
  | Double_prec _, Fp8_prec _ -> ("single_to_fp8((float)", ")")
  (* Conversions involving BFloat16 and other types *)
  | Bfloat16_prec _, Half_prec _ -> ("FLOAT_TO_HALF(bfloat16_to_single(", "))")
  | Half_prec _, Bfloat16_prec _ -> ("single_to_bfloat16(HALF_TO_FLOAT(", "))")
  | Bfloat16_prec _, (Byte_prec _ | Uint16_prec _ | Int32_prec _) ->
      ("(" ^ c_typ_of_prec to_ ^ ")bfloat16_to_single(", ")")
  | (Byte_prec _ | Uint16_prec _ | Int32_prec _), Bfloat16_prec _ ->
      ("single_to_bfloat16((float)", ")")
  (* Conversions involving FP8 and other types *)
  | Fp8_prec _, Half_prec _ -> ("FLOAT_TO_HALF(fp8_to_single(", "))")
  | Half_prec _, Fp8_prec _ -> ("single_to_fp8(HALF_TO_FLOAT(", "))")
  | Fp8_prec _, (Byte_prec _ | Uint16_prec _ | Int32_prec _ | Int64_prec _) ->
      ("(" ^ c_typ_of_prec to_ ^ ")fp8_to_single(", ")")
  | (Byte_prec _ | Uint16_prec _ | Int32_prec _ | Int64_prec _), Fp8_prec _ ->
      ("single_to_fp8((float)", ")")
  (* BFloat16 <-> FP8 conversions *)
  | Bfloat16_prec _, Fp8_prec _ -> ("single_to_fp8(bfloat16_to_single(", "))")
  | Fp8_prec _, Bfloat16_prec _ -> ("single_to_bfloat16(fp8_to_single(", "))")
  (* Half precision conversions - use macros for zero overhead on native systems *)
  | Half_prec _, Single_prec _ -> ("HALF_TO_FLOAT(", ")")
  | Single_prec _, Half_prec _ -> ("FLOAT_TO_HALF(", ")")
  | Half_prec _, Double_prec _ -> ("(double)HALF_TO_FLOAT(", ")")
  | Double_prec _, Half_prec _ -> ("FLOAT_TO_HALF((float)", ")")
  | Half_prec _, (Byte_prec _ | Uint16_prec _ | Int32_prec _ | Int64_prec _) ->
      ("(" ^ c_typ_of_prec to_ ^ ")HALF_TO_FLOAT(", ")")
  | (Byte_prec _ | Uint16_prec _ | Int32_prec _ | Int64_prec _), Half_prec _ ->
      ("FLOAT_TO_HALF((float)", ")")
  (* Uint4x32 conversions - special handling *)
  | Uint4x32_prec _, _ -> ("uint4x32_to_" ^ prec_string to_ ^ "(", ")")
  | _, Uint4x32_prec _ -> (prec_string from ^ "_to_uint4x32(", ")")
  (* Default case for all other conversions *)
  | _ -> ("(" ^ c_typ_of_prec to_ ^ ")(", ")")

(** {2 *** Pointer representation ***} *)

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

(** {2 *** External FFI declarations ***} *)

type axis_padding = { left : int; right : int } [@@deriving sexp, equal]

external bfloat16_to_single : int -> float = "arrayjit_bfloat16_to_single"
(** Original conversion functions *)

external single_to_bfloat16 : float -> int = "arrayjit_single_to_bfloat16"
external half_to_single : int -> float = "arrayjit_half_to_single"
external single_to_half : float -> int = "arrayjit_single_to_half"
external fp8_to_single : int -> float = "arrayjit_fp8_to_single"
external single_to_fp8 : float -> int = "arrayjit_single_to_fp8"

external copy_with_padding_c :
  ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t ->
  ('a, 'b, Bigarray.c_layout) Bigarray.Genarray.t ->
  axis_padding array ->
  unit = "arrayjit_copy_with_padding"

external threefry4x32_crypto : int array -> int array -> int array
  = "arrayjit_threefry4x32_crypto_ocaml"
(** Threefry4x32 PRNG - 20 round cryptographic version *)

external threefry4x32_light : int array -> int array -> int array
  = "arrayjit_threefry4x32_light_ocaml"
(** Threefry4x32 PRNG - 2 round light version *)

external threefry4x32 : int array -> int array -> int array = "arrayjit_threefry4x32_ocaml"
(** Threefry4x32 PRNG - default version *)

external uint4x32_to_single_uniform : int array -> float
  = "arrayjit_uint4x32_to_single_uniform_ocaml"
(** Conversion from uint4x32 to various uniform distributions *)

external uint4x32_to_double_uniform : int array -> float
  = "arrayjit_uint4x32_to_double_uniform_ocaml"

external uint4x32_to_int32_uniform : int array -> int = "arrayjit_uint4x32_to_int32_uniform_ocaml"
external uint4x32_to_int64_uniform : int array -> int64 = "arrayjit_uint4x32_to_int64_uniform_ocaml"
external uint4x32_to_uint32_uniform : int array -> int = "arrayjit_uint4x32_to_uint32_uniform_ocaml"

external uint4x32_to_uint64_uniform : int array -> int64
  = "arrayjit_uint4x32_to_uint64_uniform_ocaml"

external uint4x32_to_byte_uniform : int array -> int = "arrayjit_uint4x32_to_byte_uniform_ocaml"
external uint4x32_to_uint16_uniform : int array -> int = "arrayjit_uint4x32_to_uint16_uniform_ocaml"

external uint4x32_to_bfloat16_uniform : int array -> int
  = "arrayjit_uint4x32_to_bfloat16_uniform_ocaml"

external uint4x32_to_half_uniform : int array -> int = "arrayjit_uint4x32_to_half_uniform_ocaml"
external uint4x32_to_fp8_uniform : int array -> int = "arrayjit_uint4x32_to_fp8_uniform_ocaml"

external single_to_uint4x32 : float -> int array = "arrayjit_single_to_uint4x32_ocaml"
(** Conversion to uint4x32 from various types *)

external double_to_uint4x32 : float -> int array = "arrayjit_double_to_uint4x32_ocaml"
external int32_to_uint4x32 : int -> int array = "arrayjit_int32_to_uint4x32_ocaml"
external int64_to_uint4x32 : int64 -> int array = "arrayjit_int64_to_uint4x32_ocaml"
external uint32_to_uint4x32 : int -> int array = "arrayjit_uint32_to_uint4x32_ocaml"
external uint64_to_uint4x32 : int64 -> int array = "arrayjit_uint64_to_uint4x32_ocaml"
external byte_to_uint4x32 : int -> int array = "arrayjit_byte_to_uint4x32_ocaml"
external uint16_to_uint4x32 : int -> int array = "arrayjit_uint16_to_uint4x32_ocaml"
external bfloat16_to_uint4x32 : int -> int array = "arrayjit_bfloat16_to_uint4x32_ocaml"
external half_to_uint4x32 : int -> int array = "arrayjit_half_to_uint4x32_ocaml"
external fp8_to_uint4x32 : int -> int array = "arrayjit_fp8_to_uint4x32_ocaml"

(** {2 *** Precision homogeneity classification ***} *)

(** Returns true if the unary operation is homogeneous in precision, meaning
    its argument should be converted to the result precision. *)
let is_homogeneous_prec_unop = function
  | Uint4x32_to_prec_uniform1 -> false  (* Heterogeneous: argument must be uint4x32 *)
  | _ -> true  (* All other unary operations are homogeneous *)

(** Returns true if the vec_unop operation is homogeneous in precision, meaning
    its argument should be converted to the result precision. *)
let is_homogeneous_prec_vec_unop = function
  | Uint4x32_to_prec_uniform -> false  (* Heterogeneous: argument must be uint4x32 *)

(** Returns true if the binary operation is homogeneous in precision, meaning
    its arguments should be converted to the result precision. *)
let is_homogeneous_prec_binop = function
  | _ -> true  (* All binary operations are currently homogeneous *)

(** Returns true if the ternary operation is homogeneous in precision, meaning
    its arguments should be converted to the result precision. *)
let is_homogeneous_prec_ternop = function
  | Where -> false  (* Heterogeneous: condition can have different precision *)
  | FMA -> true     (* FMA is homogeneous *)

let () =
  (* Ensure that the functions are linked in *)
  let _ = bfloat16_to_single 0 in
  let _ = single_to_bfloat16 0.0 in
  let _ = fp8_to_single 0 in
  let _ = single_to_fp8 0.0 in
  let _ =
    copy_with_padding_c
      (Bigarray.Genarray.create Bigarray.Float32 Bigarray.c_layout [| 1; 1 |])
      (Bigarray.Genarray.create Bigarray.Float32 Bigarray.c_layout [| 1; 1 |])
      [| { left = 0; right = 0 }; { left = 0; right = 0 } |]
  in
  let _ = threefry4x32 [| 0 |] [| 0 |] in
  let _ = threefry4x32_crypto [| 0 |] [| 0 |] in
  let _ = threefry4x32_light [| 0 |] [| 0 |] in
  let _ = uint4x32_to_single_uniform [| 0 |] in
  let _ = uint4x32_to_double_uniform [| 0 |] in
  let _ = uint4x32_to_int32_uniform [| 0 |] in
  let _ = uint4x32_to_int64_uniform [| 0 |] in
  let _ = uint4x32_to_uint32_uniform [| 0 |] in
  let _ = uint4x32_to_uint64_uniform [| 0 |] in
  let _ = uint4x32_to_byte_uniform [| 0 |] in
  let _ = uint4x32_to_uint16_uniform [| 0 |] in
  let _ = uint4x32_to_bfloat16_uniform [| 0 |] in
  let _ = uint4x32_to_half_uniform [| 0 |] in
  let _ = uint4x32_to_fp8_uniform [| 0 |] in
  let _ = single_to_uint4x32 0.0 in
  let _ = double_to_uint4x32 0.0 in
  let _ = int32_to_uint4x32 0 in
  let _ = int64_to_uint4x32 0L in
  let _ = uint32_to_uint4x32 0 in
  let _ = uint64_to_uint4x32 0L in
  let _ = byte_to_uint4x32 0 in
  let _ = uint16_to_uint4x32 0 in
  let _ = bfloat16_to_uint4x32 0 in
  let _ = half_to_uint4x32 0 in
  let _ = fp8_to_uint4x32 0 in
  ()
