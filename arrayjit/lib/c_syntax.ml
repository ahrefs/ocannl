open Base
module Lazy = Utils.Lazy
open Backend_intf

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

module Tn = Tnode

type t = PPrint.document

module type C_syntax_config = sig
  val procs : Low_level.optimized array
  (** The low-level prcedure to compile, and the arrays of the context it will be linked to if not
      shared and already known. *)

  type buffer_ptr

  val use_host_memory : (size_in_bytes:int -> unit Ctypes.ptr -> buffer_ptr) option
  val main_kernel_prefix : string
  val kernel_prep_line : string
  val buffer_prefix : string
  val buffer_suffix : pos:int -> string
  val arg_int_prefix : string
  val extra_args : string list
  val includes : string list
  val extra_declarations : string list
  val typ_of_prec : Ops.prec -> string
  val vec_typ_of_prec : length:int -> Ops.prec -> string
  val ident_blacklist : string list

  val float_log_style : string
  (** Format specifier for printing floating point numbers in debug logs. *)

  val styled_log_arg : PPrint.document -> PPrint.document
  (** Function to convert potentially floating-point numeric values for logging. *)

  val ternop_syntax :
    Ops.prec ->
    Ops.ternop ->
    PPrint.document ->
    PPrint.document ->
    PPrint.document ->
    PPrint.document

  val binop_syntax : Ops.prec -> Ops.binop -> PPrint.document -> PPrint.document -> PPrint.document
  val unop_syntax : Ops.prec -> Ops.unop -> PPrint.document -> PPrint.document
  val vec_unop_syntax : Ops.prec -> Ops.vec_unop -> PPrint.document -> PPrint.document
  val convert_precision : from:Ops.prec -> to_:Ops.prec -> string * string

  val kernel_log_param : (string * string) option
  (** Kernel parameter for logging, if any. E.g., (Some ("int", "log_id")) or (Some ("const char*",
      "log_file_name")). *)

  val log_involves_file_management : bool
  (** Whether the logging setup involves opening/closing a FILE* (e.g., for fprintf). *)

  val pp_log_statement :
    log_param_c_expr_doc:PPrint.document option ->
    base_message_literal:string ->
    args_docs:PPrint.document list ->
    PPrint.document
  (** Generates a C log statement.
      - [log_param_c_expr_doc]: Document for the C expression of the log parameter (e.g.,
        [string "log_id"] or [string "log_file_name"]), if [kernel_log_param] is Some).
      - [base_message_literal]: The raw, unescaped, unquoted base printf-style format string (e.g.,
        "index %s = %d\n").
      - [args_docs]: Documents for the C expressions of the arguments to the format string. The
        implementation should handle quoting [base_message_literal], choosing the log function
        (printf, fprintf, os_log), and prepending any necessary prefixes (like a log_id or
        captured_log_prefix) to the format string and arguments. *)
end

module Pure_C_config (Input : sig
  type buffer_ptr

  val use_host_memory : (size_in_bytes:int -> unit Ctypes.ptr -> buffer_ptr) option
  val procs : Low_level.optimized array
  val full_printf_support : bool
end) =
struct
  let procs = Input.procs

  type nonrec buffer_ptr = Input.buffer_ptr

  let use_host_memory = Input.use_host_memory
  let main_kernel_prefix = ""
  let kernel_prep_line = ""
  let buffer_prefix = ""
  let buffer_suffix = fun ~pos:_ -> ""
  let arg_int_prefix = "const int "
  let extra_args = []
  let includes = [ "<stdio.h>"; "<stdlib.h>"; "<string.h>"; "<math.h>" ]

  let extra_declarations =
    [
      (* BFloat16 conversion functions *)
      "static inline float bfloat16_to_single(unsigned short bf16) {";
      "  unsigned int f32 = ((unsigned int)bf16) << 16;";
      "  return *((float*)&f32);";
      "}";
      "";
      "static inline unsigned short single_to_bfloat16(float f) {";
      "  unsigned int f32 = *((unsigned int*)&f);";
      "  unsigned int rounded = f32 + 0x7FFF + ((f32 >> 16) & 1);";
      "  return (unsigned short)(rounded >> 16);";
      "}";
      "";
      (* FP8 E5M2 conversion functions *)
      "static inline float fp8_to_single(unsigned char fp8) {";
      "  if (fp8 == 0) return 0.0f;";
      "  unsigned int sign = (fp8 >> 7) & 1;";
      "  unsigned int exp = (fp8 >> 2) & 0x1F;";
      "  unsigned int mant = fp8 & 0x3;";
      "  if (exp == 0x1F) {";
      "    if (mant == 0) return sign ? -INFINITY : INFINITY;";
      "    else return NAN;";
      "  }";
      "  if (exp == 0) {";
      "    float result = ldexpf((float)mant / 4.0f, -14);";
      "    if (sign) result = -result;";
      "    return result;";
      "  }";
      "  float result = (1.0f + (float)mant * 0.25f) * ldexpf(1.0f, (int)exp - 15);";
      "  if (sign) result = -result;";
      "  return result;";
      "}";
      "";
      "static inline unsigned char single_to_fp8(float f) {";
      "  if (f == 0.0f) return 0;";
      "  unsigned int sign = (f < 0) ? 1 : 0;";
      "  f = fabsf(f);";
      "  if (isinf(f)) return (sign << 7) | 0x7C;";
      "  if (isnan(f)) return (sign << 7) | 0x7F;";
      "  int exp_val;";
      "  float mant_f = frexpf(f, &exp_val);";
      "  int exp = exp_val + 14;";
      "  if (exp < 0) return sign << 7;";
      "  if (exp > 30) return (sign << 7) | 0x7C;";
      "  if (exp == 0) {";
      "    float denorm_mant = f * ldexpf(1.0f, 14) * 4.0f;";
      "    unsigned int mant_bits = (unsigned int)(denorm_mant + 0.5f);";
      "    if (mant_bits > 3) mant_bits = 3;";
      "    return (sign << 7) | mant_bits;";
      "  }";
      "  mant_f = (mant_f - 0.5f) * 4.0f;";
      "  unsigned int mant_bits = (unsigned int)(mant_f + 0.5f);";
      "  if (mant_bits > 3) mant_bits = 3;";
      "  return (unsigned char)((sign << 7) | ((exp & 0x1F) << 2) | (mant_bits & 0x3));";
      "}";
    ]

  let typ_of_prec = Ops.c_typ_of_prec
  let vec_typ_of_prec = Ops.c_vec_typ_of_prec
  let float_log_style = if Input.full_printf_support then "%g" else "%de-3"

  let styled_log_arg doc =
    if Input.full_printf_support then doc
    else
      let open PPrint in
      string "(int)(" ^^ doc ^^ string " * 1000.0)"

  let ident_blacklist =
    let remove_paren s = String.substr_replace_all s ~pattern:"(" ~with_:"" in
    let functions = ref (Set.empty (module String)) in
    let precs = Ops.[ byte; half; single; double ] in
    List.iter precs ~f:(fun prec ->
        List.iter
          Ops.[ Where; FMA ]
          ~f:(fun op ->
            let p, _, _, _ =
              try Ops.ternop_c_syntax prec op with Invalid_argument _ -> ("", "", "", "")
            in
            if String.is_suffix p ~suffix:"(" then functions := Set.add !functions (remove_paren p));
        List.iter
          Ops.
            [
              Add;
              Sub;
              Mul;
              Div;
              ToPowOf;
              Relu_gate;
              Satur01_gate;
              Max;
              Min;
              Mod;
              Cmplt;
              Cmpeq;
              Cmpne;
              Or;
              And;
              Threefry4x32;
            ]
          ~f:(fun op ->
            let p, _, _ =
              try Ops.binop_c_syntax prec op with Invalid_argument _ -> ("", "", "")
            in
            if String.is_suffix p ~suffix:"(" then functions := Set.add !functions (remove_paren p));
        List.iter
          Ops.
            [
              Identity;
              Relu;
              Satur01;
              Exp;
              Log;
              Exp2;
              Log2;
              Sin;
              Cos;
              Sqrt;
              Recip;
              Recip_sqrt;
              Neg;
              Tanh_approx;
              Not;
            ]
          ~f:(fun op ->
            let p, _ = try Ops.unop_c_syntax prec op with Invalid_argument _ -> ("", "") in
            if String.is_suffix p ~suffix:"(" then functions := Set.add !functions (remove_paren p));
        List.iter
          Ops.[ Uint4x32_to_prec_uniform ]
          ~f:(fun op ->
            let p, _ = try Ops.vec_unop_c_syntax prec op with Invalid_argument _ -> ("", "") in
            if String.is_suffix p ~suffix:"(" then functions := Set.add !functions (remove_paren p)));
    Set.to_list !functions

  let ternop_syntax prec op v1 v2 v3 =
    match prec with
    | Ops.Bfloat16_prec _ ->
        (* For BFloat16, perform operations in float precision *)
        let float_v1 = PPrint.(string "bfloat16_to_single(" ^^ v1 ^^ string ")") in
        let float_v2 = PPrint.(string "bfloat16_to_single(" ^^ v2 ^^ string ")") in
        let float_v3 = PPrint.(string "bfloat16_to_single(" ^^ v3 ^^ string ")") in
        let op_prefix, op_infix1, op_infix2, op_suffix = Ops.ternop_c_syntax Ops.single op in
        let float_result =
          PPrint.(
            group
              (string op_prefix ^^ float_v1 ^^ string op_infix1
              ^^ ifflat (space ^^ float_v2) (nest 2 (break 1 ^^ float_v2))
              ^^ string op_infix2
              ^^ ifflat (space ^^ float_v3) (nest 2 (break 1 ^^ float_v3))
              ^^ string op_suffix))
        in
        PPrint.(string "single_to_bfloat16(" ^^ float_result ^^ string ")")
    | Ops.Fp8_prec _ ->
        (* For FP8, perform operations in float precision *)
        let float_v1 = PPrint.(string "fp8_to_single(" ^^ v1 ^^ string ")") in
        let float_v2 = PPrint.(string "fp8_to_single(" ^^ v2 ^^ string ")") in
        let float_v3 = PPrint.(string "fp8_to_single(" ^^ v3 ^^ string ")") in
        let op_prefix, op_infix1, op_infix2, op_suffix = Ops.ternop_c_syntax Ops.single op in
        let float_result =
          PPrint.(
            group
              (string op_prefix ^^ float_v1 ^^ string op_infix1
              ^^ ifflat (space ^^ float_v2) (nest 2 (break 1 ^^ float_v2))
              ^^ string op_infix2
              ^^ ifflat (space ^^ float_v3) (nest 2 (break 1 ^^ float_v3))
              ^^ string op_suffix))
        in
        PPrint.(string "single_to_fp8(" ^^ float_result ^^ string ")")
    | _ ->
        let op_prefix, op_infix1, op_infix2, op_suffix = Ops.ternop_c_syntax prec op in
        let open PPrint in
        group
          (string op_prefix ^^ v1 ^^ string op_infix1
          ^^ ifflat (space ^^ v2) (nest 2 (break 1 ^^ v2))
          ^^ string op_infix2
          ^^ ifflat (space ^^ v3) (nest 2 (break 1 ^^ v3))
          ^^ string op_suffix)

  let binop_syntax prec op v1 v2 =
    match op with
    | Ops.Satur01_gate -> (
        match prec with
        | Ops.Byte_prec _ | Ops.Uint16_prec _ | Ops.Int32_prec _ | Ops.Uint4x32_prec _ ->
            let open PPrint in
            group
              (parens
                 (group
                    (parens
                       (string "(float)" ^^ v1 ^^ string " > 0.0f && (float)" ^^ v1
                      ^^ string " < 1.0f"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "("
                      ^^ string (typ_of_prec prec)
                      ^^ string ")0")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "("
                         ^^ string (typ_of_prec prec)
                         ^^ string ")0"))))
        | Ops.Fp8_prec _ ->
            let open PPrint in
            group
              (parens
                 (group
                    (parens
                       (string "fp8_to_single(" ^^ v1
                       ^^ string ") > 0.0f && fp8_to_single("
                       ^^ v1 ^^ string ") < 1.0f"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "single_to_fp8(0.0f)")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "single_to_fp8(0.0f)"))))
        | Ops.Bfloat16_prec _ ->
            let open PPrint in
            group
              (parens
                 (group
                    (parens
                       (string "bfloat16_to_single(" ^^ v1
                       ^^ string ") > 0.0f && bfloat16_to_single("
                       ^^ v1 ^^ string ") < 1.0f"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "single_to_bfloat16(0.0f)")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "single_to_bfloat16(0.0f)"))))
        | Ops.Half_prec _ ->
            let open PPrint in
            group
              (parens
                 (group (parens (v1 ^^ string " > 0.0f16 && " ^^ v1 ^^ string " < 1.0f16"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "0.0f16")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "0.0f16"))))
        | Ops.Single_prec _ ->
            let open PPrint in
            group
              (parens
                 (group (parens (v1 ^^ string " > 0.0f && " ^^ v1 ^^ string " < 1.0f"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "0.0f")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "0.0f"))))
        | Ops.Double_prec _ ->
            let open PPrint in
            group
              (parens
                 (group (parens (v1 ^^ string " > 0.0 && " ^^ v1 ^^ string " < 1.0"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "0.0")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "0.0"))))
        | Ops.Void_prec -> invalid_arg "Pure_C_config.binop_syntax: Satur01_gate on Void_prec")
    | _ -> (
        match prec with
        | Ops.Bfloat16_prec _ -> (
            (* For BFloat16, perform all operations in float precision *)
            let float_v1 = PPrint.(string "bfloat16_to_single(" ^^ v1 ^^ string ")") in
            let float_v2 = PPrint.(string "bfloat16_to_single(" ^^ v2 ^^ string ")") in
            let op_prefix, op_infix, op_suffix = Ops.binop_c_syntax Ops.single op in
            let float_result =
              PPrint.(
                group
                  (string op_prefix ^^ float_v1 ^^ string op_infix
                  ^^ ifflat (space ^^ float_v2) (nest 2 (break 1 ^^ float_v2))
                  ^^ string op_suffix))
            in
            (* For comparison operations, return float result (0.0 or 1.0) converted to BFloat16 *)
            match op with
            | Ops.Cmplt | Ops.Cmpeq | Ops.Cmpne | Ops.Or | Ops.And ->
                PPrint.(string "single_to_bfloat16(" ^^ float_result ^^ string ")")
            | _ -> PPrint.(string "single_to_bfloat16(" ^^ float_result ^^ string ")"))
        | Ops.Fp8_prec _ ->
            (* For FP8, perform all operations in float precision *)
            let float_v1 = PPrint.(string "fp8_to_single(" ^^ v1 ^^ string ")") in
            let float_v2 = PPrint.(string "fp8_to_single(" ^^ v2 ^^ string ")") in
            let op_prefix, op_infix, op_suffix = Ops.binop_c_syntax Ops.single op in
            let float_result =
              PPrint.(
                group
                  (string op_prefix ^^ float_v1 ^^ string op_infix
                  ^^ ifflat (space ^^ float_v2) (nest 2 (break 1 ^^ float_v2))
                  ^^ string op_suffix))
            in
            PPrint.(string "single_to_fp8(" ^^ float_result ^^ string ")")
        | _ ->
            let op_prefix, op_infix, op_suffix = Ops.binop_c_syntax prec op in
            let open PPrint in
            group
              (string op_prefix ^^ v1 ^^ string op_infix
              ^^ ifflat (space ^^ v2) (nest 2 (break 1 ^^ v2))
              ^^ string op_suffix))

  let unop_syntax prec op v =
    match prec with
    | Ops.Bfloat16_prec _ ->
        (* For BFloat16, perform operations in float precision *)
        let float_v = PPrint.(string "bfloat16_to_single(" ^^ v ^^ string ")") in
        let op_prefix, op_suffix = Ops.unop_c_syntax Ops.single op in
        let float_result = PPrint.(group (string op_prefix ^^ float_v ^^ string op_suffix)) in
        PPrint.(string "single_to_bfloat16(" ^^ float_result ^^ string ")")
    | Ops.Fp8_prec _ ->
        (* For FP8, perform operations in float precision *)
        let float_v = PPrint.(string "fp8_to_single(" ^^ v ^^ string ")") in
        let op_prefix, op_suffix = Ops.unop_c_syntax Ops.single op in
        let float_result = PPrint.(group (string op_prefix ^^ float_v ^^ string op_suffix)) in
        PPrint.(string "single_to_fp8(" ^^ float_result ^^ string ")")
    | _ ->
        let op_prefix, op_suffix = Ops.unop_c_syntax prec op in
        let open PPrint in
        group (string op_prefix ^^ v ^^ string op_suffix)

  let vec_unop_syntax prec op v =
    let op_prefix, op_suffix = Ops.vec_unop_c_syntax prec op in
    let open PPrint in
    group (string op_prefix ^^ v ^^ string op_suffix)

  let convert_precision = Ops.c_convert_precision
  let kernel_log_param = Some ("const char*", "log_file_name")
  let log_involves_file_management = true

  let for_log_trace_tree =
    Utils.get_global_flag ~default:false ~arg_name:"debug_log_to_stream_files"

  let pp_log_statement ~log_param_c_expr_doc:_ ~base_message_literal ~args_docs =
    let open PPrint in
    let log_file_check =
      match kernel_log_param with
      | Some (_, lname) -> string ("if (" ^ lname ^ " && log_file) ")
      | None ->
          string "if (log_file) " (* Should not happen if log_involves_file_management is true *)
    in
    let base_message_literal =
      let with_ = if for_log_trace_tree then "$" else "\\n" in
      let res = String.substr_replace_all base_message_literal ~pattern:"\n" ~with_ in
      if for_log_trace_tree && String.is_suffix res ~suffix:"$" then
        String.drop_suffix res 1 ^ "\\n"
      else res
    in
    log_file_check ^^ string "fprintf(log_file, "
    ^^ dquotes (string base_message_literal)
    ^^ (if List.is_empty args_docs then empty else comma ^^ space)
    ^^ separate (comma ^^ space) args_docs
    ^^ rparen ^^ semi
end

module C_syntax (B : C_syntax_config) = struct
  let get_ident =
    Low_level.get_ident_within_code ~no_dots:true ~blacklist:B.ident_blacklist
    @@ Array.map B.procs ~f:(fun l -> l.llc)

  let in_ctx tn = B.(Tn.is_in_context_force ~use_host_memory tn 46)
  let pp_include s = PPrint.(string "#include " ^^ string s)

  open Indexing
  open Doc_helpers

  let pp_array_offset (idcs, dims) =
    let open PPrint in
    if Array.is_empty idcs then string "0"
    else
      let doc = ref (pp_axis_index idcs.(0)) in
      for i = 1 to Array.length idcs - 1 do
        doc :=
          parens !doc ^^ string (" * " ^ Int.to_string dims.(i) ^ " + ") ^^ pp_axis_index idcs.(i)
      done;
      !doc

  let doc_to_string doc =
    let buf = Buffer.create 128 in
    PPrint.ToBuffer.compact buf doc;
    Buffer.contents buf

  let array_offset_to_string (idcs, dims) = doc_to_string @@ pp_array_offset (idcs, dims)

  let print_declarations () =
    let open PPrint in
    let includes = separate hardline (List.map B.includes ~f:pp_include) in
    let extras = separate hardline (List.map B.extra_declarations ~f:string) in
    includes ^^ hardline ^^ extras ^^ hardline

  let rec pp_ll (c : Low_level.t) : PPrint.document =
    let open PPrint in
    match c with
    | Low_level.Noop -> empty
    | Seq (c1, c2) ->
        let d1 = pp_ll c1 in
        let d2 = pp_ll c2 in
        (* Avoid extra hardlines if one side is empty *)
        if PPrint.is_empty d1 then d2 else if PPrint.is_empty d2 then d1 else d1 ^^ hardline ^^ d2
    | For_loop { index = i; from_; to_; body; trace_it = _ } ->
        let header =
          string "for (int " ^^ pp_symbol i ^^ string " = " ^^ PPrint.OCaml.int from_ ^^ semi
          ^^ space ^^ pp_symbol i ^^ string " <= " ^^ PPrint.OCaml.int to_ ^^ semi ^^ space
          ^^ string "++" ^^ pp_symbol i ^^ string ")"
        in
        let body_doc = ref (pp_ll body) in
        (if Utils.debug_log_from_routines () then
           let log_doc =
             let base_message = Printf.sprintf "index %s = %%d\n" (symbol_ident i) in
             let log_param_doc = Option.map B.kernel_log_param ~f:(fun (_, name) -> string name) in
             B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
               ~base_message_literal:base_message
               ~args_docs:[ pp_symbol i ]
           in
           body_doc := log_doc ^^ hardline ^^ !body_doc);
        group (header ^^ space ^^ lbrace ^^ nest 2 (hardline ^^ !body_doc) ^^ hardline ^^ rbrace)
    | Zero_out tn ->
        pp_ll
          (Low_level.loop_over_dims (Lazy.force tn.dims) ~body:(fun idcs ->
               Set { tn; idcs; llsc = Constant 0.0; debug = get_ident tn ^ " := 0" }))
    | Set { tn; idcs; llsc; debug } ->
        let ident_doc = string (get_ident tn) in
        let dims = Lazy.force tn.dims in
        let prec = Lazy.force tn.prec in
        let local_defs, val_doc = pp_float prec llsc in
        let offset_doc = pp_array_offset (idcs, dims) in
        let assignment =
          group
            (ident_doc ^^ brackets offset_doc ^^ string " ="
            ^^ ifflat (space ^^ val_doc) (nest 4 (hardline ^^ val_doc))
            ^^ semi)
        in
        if Utils.debug_log_from_routines () then
          let num_typ = string (B.typ_of_prec prec) in
          let new_var = string "new_set_v" in
          let decl = num_typ ^^ space ^^ new_var ^^ string " = " ^^ val_doc ^^ semi in
          let debug_val_doc, debug_args_docs = debug_float prec llsc in
          let debug_val_str = doc_to_string debug_val_doc in
          let pp_args_docs =
            List.map debug_args_docs ~f:(function
              | `Accessor idx -> pp_array_offset idx
              | `Value v_doc -> B.styled_log_arg v_doc)
          in
          let log_args_for_printf =
            offset_doc
            :: B.styled_log_arg (ident_doc ^^ brackets offset_doc)
            :: B.styled_log_arg new_var :: pp_args_docs
          in
          let log_doc =
            let log_param_doc = Option.map B.kernel_log_param ~f:(fun (_, name) -> string name) in
            let comment_base_msg = "# " ^ debug ^ "\n" in
            let value_base_msg =
              Printf.sprintf "%s[%%u]{=%s} = %s = %s\n" (get_ident tn) B.float_log_style
                B.float_log_style debug_val_str
            in
            let comment_log =
              B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
                ~base_message_literal:comment_base_msg ~args_docs:[]
            in
            let value_log =
              B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
                ~base_message_literal:value_base_msg ~args_docs:log_args_for_printf
            in
            let flush_log =
              if B.log_involves_file_management then string "fflush(log_file);" else empty
            in
            comment_log ^^ hardline ^^ value_log ^^ hardline ^^ flush_log
          in
          let assignment' = ident_doc ^^ brackets offset_doc ^^ string " = " ^^ new_var ^^ semi in
          let block_content =
            if PPrint.is_empty local_defs then
              decl ^^ hardline ^^ log_doc ^^ hardline ^^ assignment'
            else local_defs ^^ hardline ^^ decl ^^ hardline ^^ log_doc ^^ hardline ^^ assignment'
          in
          lbrace ^^ nest 2 (hardline ^^ block_content) ^^ hardline ^^ rbrace
        else if PPrint.is_empty local_defs then assignment
        else local_defs ^^ hardline ^^ assignment
    | Comment message ->
        if Utils.debug_log_from_routines () then
          let base_message = "COMMENT: " ^ message ^ "\n" in
          let log_param_doc = Option.map B.kernel_log_param ~f:(fun (_, name) -> string name) in
          B.pp_log_statement ~log_param_c_expr_doc:log_param_doc ~base_message_literal:base_message
            ~args_docs:[]
        else string "/* " ^^ string message ^^ string " */"
    | Staged_compilation callback -> callback ()
    | Set_from_vec { tn; idcs; length; vec_unop; arg; debug } ->
        let ident_doc = string (get_ident tn) in
        let dims = Lazy.force tn.dims in
        let prec = Lazy.force tn.prec in
        let arg_prec = Ops.uint4x32 in
        let local_defs, arg_doc = pp_float arg_prec arg in
        (* Generate the function call *)
        let result_doc = B.vec_unop_syntax prec vec_unop arg_doc in
        (* Generate assignments for each output element *)
        let assignments =
          let open PPrint in
          let vec_var = string "vec_result" in
          let vec_typ = string (B.vec_typ_of_prec ~length prec) in
          let vec_decl = vec_typ ^^ space ^^ vec_var ^^ string " = " ^^ result_doc ^^ semi in
          let elem_assigns =
            List.init length ~f:(fun i ->
                let offset_doc =
                  match idcs.(Array.length idcs - 1) with
                  | Fixed_idx idx ->
                      (* For Fixed_idx, update the index and compute offset normally *)
                      let elem_idcs = Array.copy idcs in
                      elem_idcs.(Array.length elem_idcs - 1) <- Fixed_idx (idx + i);
                      pp_array_offset (elem_idcs, dims)
                  | _ ->
                      (* For non-Fixed_idx (Iterator, etc), add i to the computed offset *)
                      pp_array_offset (idcs, dims) ^^ string (" + " ^ Int.to_string i)
                in
                ident_doc ^^ brackets offset_doc ^^ string " = " ^^ vec_var
                ^^ string (".v[" ^ Int.to_string i ^ "]")
                ^^ semi)
          in
          vec_decl ^^ hardline ^^ separate hardline elem_assigns
        in
        if Utils.debug_log_from_routines () then
          let open PPrint in
          let log_param_doc = Option.map B.kernel_log_param ~f:(fun (_, name) -> string name) in
          let comment_base_msg = "# " ^ debug ^ "\n" in
          let comment_log =
            B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
              ~base_message_literal:comment_base_msg ~args_docs:[]
          in
          let value_logs =
            List.init length ~f:(fun i ->
                let elem_idcs = Array.copy idcs in
                (match elem_idcs.(Array.length elem_idcs - 1) with
                | Fixed_idx idx -> elem_idcs.(Array.length elem_idcs - 1) <- Fixed_idx (idx + i)
                | _ -> ());
                let offset_doc =
                  let base_offset = pp_array_offset (elem_idcs, dims) in
                  match elem_idcs.(Array.length elem_idcs - 1) with
                  | Fixed_idx _ -> base_offset
                  | _ -> base_offset ^^ string (" + " ^ Int.to_string i)
                in
                let value_base_msg =
                  Printf.sprintf "%s[%%u]{=%s} = vec_result.v[%d] = %s\n" (get_ident tn)
                    B.float_log_style i B.float_log_style
                in
                let log_args =
                  [
                    offset_doc;
                    B.styled_log_arg (ident_doc ^^ brackets offset_doc);
                    B.styled_log_arg (string ("vec_result.v[" ^ Int.to_string i ^ "]"));
                  ]
                in
                B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
                  ~base_message_literal:value_base_msg ~args_docs:log_args)
          in
          let flush_log =
            if B.log_involves_file_management then string "fflush(log_file);" else empty
          in
          let log_docs =
            comment_log ^^ hardline ^^ separate hardline value_logs ^^ hardline ^^ flush_log
          in
          let block_content =
            if PPrint.is_empty local_defs then assignments ^^ hardline ^^ log_docs
            else local_defs ^^ hardline ^^ assignments ^^ hardline ^^ log_docs
          in
          lbrace ^^ nest 2 (hardline ^^ block_content) ^^ hardline ^^ rbrace
        else if PPrint.is_empty local_defs then assignments
        else local_defs ^^ hardline ^^ assignments
    | Set_local ({ scope_id; tn = { prec; _ } }, value) ->
        let local_defs, value_doc = pp_float (Lazy.force prec) value in
        let assignment =
          string ("v" ^ Int.to_string scope_id) ^^ string " = " ^^ value_doc ^^ semi
        in
        if PPrint.is_empty local_defs then assignment else local_defs ^^ hardline ^^ assignment

  and pp_float (prec : Ops.prec) (vcomp : Low_level.scalar_t) : PPrint.document * PPrint.document =
    (* Returns (local definitions, value expression) *)
    let open PPrint in
    match vcomp with
    | Local_scope { id = { scope_id; tn = { prec = scope_prec; _ } }; body; orig_indices = _ } ->
        let num_typ = string (B.typ_of_prec @@ Lazy.force scope_prec) in
        let decl =
          num_typ ^^ space ^^ string ("v" ^ Int.to_string scope_id) ^^ string " = 0" ^^ semi
        in
        let body_doc = pp_ll body in
        let defs = decl ^^ hardline ^^ body_doc in
        let prefix, postfix = B.convert_precision ~from:(Lazy.force scope_prec) ~to_:prec in
        let expr = string prefix ^^ string ("v" ^ Int.to_string scope_id) ^^ string postfix in
        (defs, expr)
    | Get_local id ->
        let scope_prec = Lazy.force id.tn.prec in
        let prefix, postfix = B.convert_precision ~from:scope_prec ~to_:prec in
        let expr = string prefix ^^ string ("v" ^ Int.to_string id.scope_id) ^^ string postfix in
        (empty, expr)
    | Get_merge_buffer (source, idcs) ->
        let tn = source in
        let from_prec = Lazy.force tn.prec in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        let offset_doc = pp_array_offset (idcs, Lazy.force tn.dims) in
        let expr =
          string prefix ^^ string "merge_buffer" ^^ brackets offset_doc ^^ string postfix
        in
        (empty, expr)
    | Get (tn, idcs) ->
        let ident_doc = string (get_ident tn) in
        let from_prec = Lazy.force tn.prec in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        let offset_doc = pp_array_offset (idcs, Lazy.force tn.dims) in
        let expr = string prefix ^^ ident_doc ^^ brackets offset_doc ^^ string postfix in
        (empty, expr)
    | Constant c ->
        let from_prec = Ops.double in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        let c_str = Printf.sprintf "%.16g" c in
        let expr =
          if String.is_empty prefix && Float.(c < 0.0) then
            string "(" ^^ string c_str ^^ string ")" ^^ string postfix
          else string prefix ^^ string c_str ^^ string postfix
        in
        (empty, expr)
    | Embed_index idx ->
        let from_prec = Ops.double in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        let expr = string prefix ^^ pp_axis_index idx ^^ string postfix in
        (empty, expr)
    | Binop (Arg1, v1, _v2) -> pp_float prec v1
    | Binop (Arg2, _v1, v2) -> pp_float prec v2
    | Ternop (op, v1, v2, v3) ->
        let d1, e1 = pp_float prec v1 in
        let d2, e2 = pp_float prec v2 in
        let d3, e3 = pp_float prec v3 in
        let defs =
          List.filter_map [ d1; d2; d3 ] ~f:(fun d -> if PPrint.is_empty d then None else Some d)
          |> separate hardline
        in
        let expr = group (B.ternop_syntax prec op e1 e2 e3) in
        (defs, expr)
    | Binop (op, v1, v2) ->
        let d1, e1 = pp_float prec v1 in
        let d2, e2 = pp_float prec v2 in
        let defs =
          List.filter_map [ d1; d2 ] ~f:(fun d -> if PPrint.is_empty d then None else Some d)
          |> separate hardline
        in
        let expr = group (B.binop_syntax prec op e1 e2) in
        (defs, expr)
    | Unop (op, v) ->
        let defs, expr_v = pp_float prec v in
        let expr = group (B.unop_syntax prec op expr_v) in
        (defs, expr)

  and debug_float (prec : Ops.prec) (value : Low_level.scalar_t) :
      PPrint.document
      * [ `Accessor of Indexing.axis_index array * int array | `Value of PPrint.document ] list =
    (* Returns (value expression doc, list of arguments for printf) *)
    let open PPrint in
    match value with
    | Local_scope { id; _ } ->
        (* Not printing the inlined definition: (1) code complexity; (2) don't overload the debug
           logs. *)
        debug_float prec @@ Get_local id
    | Get_local id ->
        let scope_prec = Lazy.force id.tn.prec in
        let prefix, postfix = B.convert_precision ~from:scope_prec ~to_:prec in
        let v_doc = string prefix ^^ string ("v" ^ Int.to_string id.scope_id) ^^ string postfix in
        (v_doc ^^ braces (string ("=" ^ B.float_log_style)), [ `Value v_doc ])
    | Get_merge_buffer (source, idcs) ->
        let tn = source in
        let from_prec = Lazy.force tn.prec in
        let dims = Lazy.force tn.dims in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        let offset_doc = pp_array_offset (idcs, dims) in
        let access_doc =
          string prefix ^^ string "merge_buffer" ^^ brackets offset_doc ^^ string postfix
        in
        let expr_doc =
          string prefix ^^ string "merge_buffer"
          ^^ brackets (string "%u")
          ^^ string postfix
          ^^ braces (string ("=" ^ B.float_log_style))
        in
        (expr_doc, [ `Accessor (idcs, dims); `Value access_doc ])
    | Get (tn, idcs) ->
        let ident_doc = string (get_ident tn) in
        let from_prec = Lazy.force tn.prec in
        let dims = Lazy.force tn.dims in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        let offset_doc = pp_array_offset (idcs, dims) in
        let access_doc = string prefix ^^ ident_doc ^^ brackets offset_doc ^^ string postfix in
        let expr_doc =
          string prefix ^^ ident_doc
          ^^ brackets (string "%u")
          ^^ string postfix
          ^^ braces (string ("=" ^ B.float_log_style))
        in
        (expr_doc, [ `Accessor (idcs, dims); `Value access_doc ])
    | Constant c ->
        let from_prec = Ops.double in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        let c_str = Printf.sprintf "%.16g" c in
        (string prefix ^^ string c_str ^^ string postfix, [])
    | Embed_index idx -> (pp_axis_index idx, [])
    | Binop (Arg1, v1, _v2) -> debug_float prec v1
    | Binop (Arg2, _v1, v2) -> debug_float prec v2
    | Ternop (op, v1, v2, v3) ->
        let v1_doc, idcs1 = debug_float prec v1 in
        let v2_doc, idcs2 = debug_float prec v2 in
        let v3_doc, idcs3 = debug_float prec v3 in
        (B.ternop_syntax prec op v1_doc v2_doc v3_doc, idcs1 @ idcs2 @ idcs3)
    | Binop (op, v1, v2) ->
        let v1_doc, idcs1 = debug_float prec v1 in
        let v2_doc, idcs2 = debug_float prec v2 in
        (B.binop_syntax prec op v1_doc v2_doc, idcs1 @ idcs2)
    | Unop (op, v) ->
        let v_doc, idcs = debug_float prec v in
        (B.unop_syntax prec op v_doc, idcs)

  let compile_main llc : PPrint.document = pp_ll llc

  let compile_proc ~name idx_params Low_level.{ traced_store; llc; merge_node; optimize_ctx = _ } :
      (string * param_source) list * PPrint.document =
    let open PPrint in
    let params : (string * param_source) list =
      List.rev
      @@ Hashtbl.fold traced_store ~init:[] ~f:(fun ~key:tn ~data:_ params ->
             let backend_info, is_param =
               if Tn.is_virtual_force tn 334 then ("Virt", false)
               else if in_ctx tn then ("Ctx", true)
               else if Tn.is_materialized_force tn 335 then ("Global", true)
               else if Tn.known_not_materialized tn then ("Local", false)
               else assert false
             in
             let backend_info = Sexp.Atom backend_info in
             if not @@ Utils.sexp_mem ~elem:backend_info tn.backend_info then
               tn.backend_info <- Utils.sexp_append ~elem:backend_info tn.backend_info;
             if is_param then
               (B.typ_of_prec (Lazy.force tn.Tn.prec) ^ " *" ^ get_ident tn, Param_ptr tn) :: params
             else params)
    in
    let idx_params =
      List.map idx_params ~f:(fun s ->
          (B.arg_int_prefix ^ Indexing.symbol_ident s.Indexing.static_symbol, Static_idx s))
    in
    let log_file_param =
      if Utils.debug_log_from_routines () then
        match B.kernel_log_param with
        | Some (typ, name) -> [ (typ ^ " " ^ name, Log_file_name) ]
        | None -> []
      else []
    in
    let merge_param =
      Option.(
        to_list
        @@ map merge_node ~f:(fun tn ->
               ("const " ^ B.typ_of_prec (Lazy.force tn.prec) ^ " *merge_buffer", Merge_buffer)))
    in
    let all_params = log_file_param @ merge_param @ idx_params @ params in
    let sorted_params =
      List.sort all_params ~compare:(fun (p1_name, _) (p2_name, _) ->
          compare_string p1_name p2_name)
    in
    let args_docs =
      List.mapi sorted_params ~f:(fun pos (name, _) ->
          string (B.buffer_prefix ^ name ^ B.buffer_suffix ~pos))
      @ List.map B.extra_args ~f:string
    in
    let func_header =
      string B.main_kernel_prefix ^^ space ^^ string "void" ^^ space ^^ string name
      ^^ nest 4 (lparen ^^ hardline ^^ separate (comma ^^ hardline) args_docs ^^ rparen)
    in
    let body = ref empty in
    if not (String.is_empty B.kernel_prep_line) then
      body := !body ^^ string B.kernel_prep_line ^^ semi ^^ hardline;

    if Utils.debug_log_from_routines () && B.log_involves_file_management then
      let log_file_var_name =
        match B.kernel_log_param with
        | Some (_, name) -> name
        | None -> "log_file_name" (* Should ideally not be reached if management is true *)
      in
      body :=
        !body ^^ string "FILE* log_file = NULL;" ^^ hardline
        ^^ string ("if (" ^ log_file_var_name ^ ") ")
        ^^ lbrace
        ^^ nest 2 (hardline ^^ string ("log_file = fopen(" ^ log_file_var_name ^ ", \"w\");"))
        ^^ hardline ^^ rbrace ^^ hardline
    else body := !body ^^ hardline;

    (if Utils.debug_log_from_routines () then
       let debug_init_doc =
         string "/* Debug initial parameter state. */"
         ^^ hardline
         ^^ separate_map hardline
              (fun (p_name_and_type, source) ->
                let log_param_doc =
                  Option.map B.kernel_log_param ~f:(fun (_, name) -> string name)
                in
                match source with
                | Merge_buffer ->
                    let merge_tn = Option.value_exn merge_node in
                    let base_msg =
                      Printf.sprintf "%s &[%d] = %%p\n" p_name_and_type (Tnode.num_elems merge_tn)
                    in
                    B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
                      ~base_message_literal:base_msg
                      ~args_docs:[ string @@ "(" ^ B.buffer_prefix ^ "void*)merge_buffer" ]
                | Log_file_name -> empty (* Already handled by fopen or if it's just an ID *)
                | Param_ptr tn ->
                    let base_msg =
                      Printf.sprintf "%s &[%d] = %%p\n" p_name_and_type (Tnode.num_elems tn)
                    in
                    let ident_doc = string (get_ident tn) in
                    B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
                      ~base_message_literal:base_msg
                      ~args_docs:[ string ("(" ^ B.buffer_prefix ^ "void*)") ^^ ident_doc ]
                | Static_idx s ->
                    let base_msg = Printf.sprintf "%s = %%d\n" p_name_and_type in
                    let ident_doc = pp_symbol s.static_symbol in
                    B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
                      ~base_message_literal:base_msg ~args_docs:[ ident_doc ])
              sorted_params
       in
       body := !body ^^ debug_init_doc ^^ hardline);

    let local_decls =
      string "/* Local declarations and initialization. */"
      ^^ hardline
      ^^ separate_map empty
           (fun (tn, node) ->
             if not (Tn.is_virtual_force tn 333 || Tn.is_materialized_force tn 336) then
               let typ_doc = string (B.typ_of_prec @@ Lazy.force tn.prec) in
               let ident_doc = string (get_ident tn) in
               let size_doc = OCaml.int (Tn.num_elems tn) in
               let init_doc = if node.Low_level.zero_initialized then string " = {0}" else empty in
               typ_doc ^^ space ^^ ident_doc ^^ brackets size_doc ^^ init_doc ^^ semi ^^ hardline
             else empty)
           (Hashtbl.to_alist traced_store)
    in
    body := !body ^^ local_decls ^^ hardline;

    let main_logic = string "/* Main logic. */" ^^ hardline ^^ compile_main llc in
    body := !body ^^ main_logic;

    if Utils.debug_log_from_routines () && B.log_involves_file_management then
      body :=
        !body ^^ hardline
        ^^ string "if (log_file) { fclose(log_file); log_file = NULL; }"
        ^^ hardline;

    let func_doc =
      func_header ^^ space ^^ lbrace ^^ nest 2 (hardline ^^ !body) ^^ hardline ^^ rbrace
    in
    (sorted_params, func_doc)
end
