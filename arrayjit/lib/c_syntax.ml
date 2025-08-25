open Base
module Lazy = Utils.Lazy
open Backend_intf

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_C_SYNTAX=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_C_SYNTAX"]

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
              Threefry4x32_crypto;
              Threefry4x32_light;
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
    let op_prefix, op_infix1, op_infix2, op_suffix = Ops.ternop_c_syntax prec op in
    let open PPrint in
    group
      (string op_prefix ^^ v1 ^^ string op_infix1
      ^^ ifflat (space ^^ v2) (nest 2 (break 1 ^^ v2))
      ^^ string op_infix2
      ^^ ifflat (space ^^ v3) (nest 2 (break 1 ^^ v3))
      ^^ string op_suffix)

  let binop_syntax prec op v1 v2 =
    let op_prefix, op_infix, op_suffix = Ops.binop_c_syntax prec op in
    let open PPrint in
    group
      (string op_prefix ^^ v1 ^^ string op_infix
      ^^ ifflat (space ^^ v2) (nest 2 (break 1 ^^ v2))
      ^^ string op_suffix)

  let unop_syntax prec op v =
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

  let filter_and_prepend_builtins ~includes ~builtins ~proc_doc =
    let doc_buffer = Buffer.create 4096 in
    PPrint.ToBuffer.pretty 1.0 110 doc_buffer proc_doc;
    let doc_string = Buffer.contents doc_buffer in
    let result_buffer = Buffer.create 4096 in
    Buffer.add_string result_buffer includes;
    Buffer.add_string result_buffer "\n";

    (* Collect all needed keys, including dependencies *)
    let needed_keys = ref (Set.empty (module String)) in
    List.iter builtins ~f:(fun (key, _, _) ->
        if String.is_substring doc_string ~substring:key then
          needed_keys := Set.add !needed_keys key);

    (* Add dependencies recursively *)
    let processed_keys = ref (Set.empty (module String)) in
    let rec add_dependencies key =
      if not (Set.mem !processed_keys key) then (
        processed_keys := Set.add !processed_keys key;
        needed_keys := Set.add !needed_keys key;
        match List.find builtins ~f:(fun (k, _, _) -> String.equal k key) with
        | Some (_, _, deps) -> List.iter deps ~f:add_dependencies
        | None -> ())
    in
    Set.iter !needed_keys ~f:add_dependencies;

    (* Add the builtins in order *)
    List.iter builtins ~f:(fun (key, definition, _) ->
        if Set.mem !needed_keys key then (
          Buffer.add_string result_buffer definition;
          Buffer.add_string result_buffer "\n"));
    Buffer.add_string result_buffer doc_string;
    Buffer.contents result_buffer

  open Indexing
  open Doc_helpers

  let pp_array_offset (idcs, dims) =
    let open PPrint in
    if Array.is_empty idcs then string "0"
    else
      let doc = ref (pp_axis_index idcs.(0)) in
      for i = 1 to Array.length idcs - 1 do
        let idx_doc = pp_axis_index idcs.(i) in
        if PPrint.is_empty !doc then doc := idx_doc
        else if PPrint.is_empty idx_doc then
          doc := parens !doc ^^ string (" * " ^ Int.to_string dims.(i))
        else doc := parens !doc ^^ string (" * " ^ Int.to_string dims.(i) ^ " + ") ^^ idx_doc
      done;
      !doc

  let doc_to_string doc =
    let buf = Buffer.create 128 in
    PPrint.ToBuffer.compact buf doc;
    Buffer.contents buf

  let array_offset_to_string (idcs, dims) = doc_to_string @@ pp_array_offset (idcs, dims)

  let pp_local_defs (local_defs : (int * PPrint.document) list) =
    let open PPrint in
    List.dedup_and_sort local_defs ~compare:(fun (a, _) (b, _) -> Int.compare a b)
    |> List.map ~f:snd |> separate hardline

  let pp_scope_id Low_level.{ scope_id; tn } =
    let open PPrint in
    string ("v" ^ Int.to_string scope_id ^ "_" ^ get_ident tn)

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
        let local_defs, val_doc = pp_scalar prec llsc in
        let local_defs = pp_local_defs local_defs in
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
        else
          let block_content = local_defs ^^ hardline ^^ assignment in
          lbrace ^^ nest 2 (hardline ^^ block_content) ^^ hardline ^^ rbrace
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
        (* FIXME: this precision is hardcoded, bad, bad practice. *)
        let arg_prec = Ops.uint4x32 in
        let local_defs, arg_doc = pp_scalar arg_prec arg in
        let local_defs = pp_local_defs local_defs in
        (* Generate the function call *)
        let result_doc = B.vec_unop_syntax prec vec_unop arg_doc in
        (* Generate assignments for each output element *)
        let open PPrint in
        let vec_var = string "vec_result" in
        let vec_typ = string (B.vec_typ_of_prec ~length prec) in
        let vec_decl = vec_typ ^^ space ^^ vec_var ^^ string " = " ^^ result_doc ^^ semi in
        let assignments =
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
                let value_doc =
                  if length = 1 then
                    (* When length=1, vec_typ_of_prec returns a scalar type, so no .v[] access *)
                    vec_var
                  else
                    (* When length>1, access the vector element *)
                    vec_var ^^ string (".v[" ^ Int.to_string i ^ "]")
                in
                ident_doc ^^ brackets offset_doc ^^ string " = " ^^ value_doc ^^ semi)
          in
          separate hardline elem_assigns
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
            if PPrint.is_empty local_defs then
              vec_decl ^^ hardline ^^ log_docs ^^ hardline ^^ assignments
            else
              local_defs ^^ hardline ^^ vec_decl ^^ hardline ^^ log_docs ^^ hardline ^^ assignments
          in
          lbrace ^^ nest 2 (hardline ^^ block_content) ^^ hardline ^^ rbrace
        else if PPrint.is_empty local_defs then vec_decl ^^ hardline ^^ assignments
        else
          let block_content = local_defs ^^ hardline ^^ vec_decl ^^ hardline ^^ assignments in
          lbrace ^^ nest 2 (hardline ^^ block_content) ^^ hardline ^^ rbrace
    | Set_local (({ tn = { prec; _ }; _ } as id), value) ->
        let local_defs, value_doc = pp_scalar (Lazy.force prec) value in
        let local_defs = pp_local_defs local_defs in
        let assignment = pp_scope_id id ^^ string " = " ^^ value_doc ^^ semi in
        if PPrint.is_empty local_defs then assignment
        else
          let block_content = local_defs ^^ hardline ^^ assignment in
          lbrace ^^ nest 2 (hardline ^^ block_content) ^^ hardline ^^ rbrace

  and pp_scalar (prec : Ops.prec) (vcomp : Low_level.scalar_t) :
      (int * PPrint.document) list * PPrint.document =
    (* Returns (local definitions, value expression) *)
    let open PPrint in
    match vcomp with
    | Local_scope { id = { tn = { prec = scope_prec; _ }; scope_id } as id; body; orig_indices = _ }
      ->
        let scope_prec = Lazy.force scope_prec in
        let num_typ = string (B.typ_of_prec scope_prec) in
        let init_zero =
          (* TODO(#340): only do this in the rare cases where the computation is accumulating *)
          let prefix, postfix = B.convert_precision ~from:Ops.int32 ~to_:scope_prec in
          string " = " ^^ string prefix ^^ string "0" ^^ string postfix
        in
        let decl = num_typ ^^ space ^^ pp_scope_id id ^^ init_zero ^^ semi in
        let body_doc = pp_ll body in
        let def_doc = decl ^^ hardline ^^ body_doc in
        let prefix, postfix = B.convert_precision ~from:scope_prec ~to_:prec in
        let expr = string prefix ^^ pp_scope_id id ^^ string postfix in
        ([ (scope_id, def_doc) ], expr)
    | Get_local id ->
        let scope_prec = Lazy.force id.tn.prec in
        let prefix, postfix = B.convert_precision ~from:scope_prec ~to_:prec in
        let expr = string prefix ^^ pp_scope_id id ^^ string postfix in
        ([], expr)
    | Get_merge_buffer (source, idcs) ->
        let tn = source in
        let dims = Lazy.force tn.dims in
        let from_prec = Lazy.force tn.prec in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        let offset_doc = pp_array_offset (idcs, dims) in
        let expr =
          string prefix ^^ string "merge_buffer" ^^ brackets offset_doc ^^ string postfix
        in
        ([], expr)
    | Get (tn, idcs) ->
        let ident_doc = string (get_ident tn) in
        let dims = Lazy.force tn.dims in
        let from_prec = Lazy.force tn.prec in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        let offset_doc = pp_array_offset (idcs, dims) in
        let expr = string prefix ^^ ident_doc ^^ brackets offset_doc ^^ string postfix in
        ([], expr)
    | Constant c ->
        let from_prec = Ops.double in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        let c_str = Printf.sprintf "%.16g" c in
        let expr =
          if String.is_empty prefix && Float.(c < 0.0) then
            string "(" ^^ string c_str ^^ string ")" ^^ string postfix
          else string prefix ^^ string c_str ^^ string postfix
        in
        ([], expr)
    | Constant_bits i ->
        let from_prec = Ops.int64 in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        let expr = string prefix ^^ string (Printf.sprintf "%LdLL" i) ^^ string postfix in
        ([], expr)
    | Embed_index idx ->
        let from_prec = Ops.int32 in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        let idx_doc = pp_axis_index idx in
        let idx_doc = if PPrint.is_empty idx_doc then string "0" else idx_doc in
        let expr = string prefix ^^ idx_doc ^^ string postfix in
        ([], expr)
    | Binop (Arg1, v1, _v2) -> pp_scalar prec v1
    | Binop (Arg2, _v1, v2) -> pp_scalar prec v2
    | Ternop (op, v1, v2, v3) ->
        let d1, e1 = pp_scalar prec v1 in
        let d2, e2 = pp_scalar prec v2 in
        let d3, e3 = pp_scalar prec v3 in
        let defs = List.concat [ d1; d2; d3 ] in
        let expr = group (B.ternop_syntax prec op e1 e2 e3) in
        (defs, expr)
    | Binop (op, v1, v2) ->
        let d1, e1 = pp_scalar prec v1 in
        let d2, e2 = pp_scalar prec v2 in
        let defs = List.concat [ d1; d2 ] in
        let expr = group (B.binop_syntax prec op e1 e2) in
        (defs, expr)
    | Unop (op, v) ->
        let arg_prec =
          match op with
          | Ops.Uint4x32_to_prec_uniform1 ->
              (* The argument to Uint4x32_to_prec_uniform1 must be evaluated with uint4x32
                 precision, regardless of the target precision. This handles the case where the
                 operation is inlined as part of a scalar expression. *)
              Ops.uint4x32
          | _ -> prec
        in
        let defs, expr_v = pp_scalar arg_prec v in
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
        let v_doc = string prefix ^^ pp_scope_id id ^^ string postfix in
        (v_doc ^^ braces (string ("=" ^ B.float_log_style)), [ `Value v_doc ])
    | Get_merge_buffer (source, idcs) ->
        let tn = source in
        let dims = Lazy.force tn.dims in
        let from_prec = Lazy.force tn.prec in
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
        let dims = Lazy.force tn.dims in
        let from_prec = Lazy.force tn.prec in
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
    | Constant_bits i ->
        let from_prec = Ops.int64 in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        let expr = string prefix ^^ string (Printf.sprintf "%LdLL" i) ^^ string postfix in
        (expr, [])
    | Embed_index idx ->
        let idx_doc = pp_axis_index idx in
        ((if PPrint.is_empty idx_doc then string "0" else idx_doc), [])
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
                    let merge_tn = Option.value_exn ~here:[%here] merge_node in
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
               let num_elems = Tn.num_elems tn in
               let size_doc = OCaml.int num_elems in
               let init_doc =
                 if node.Low_level.zero_initialized_by_code then string " = {0}" else empty
               in
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
