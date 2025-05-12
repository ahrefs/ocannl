open Base
module Lazy = Utils.Lazy
open Backend_intf
open PPrint (* Open PPrint for direct access to combinators *)

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

module Tn = Tnode

type t = document

(* Refactored adapters to return PPrint documents instead of using Format *)
let ternop_adapter (prefix, infix1, infix2, suffix) v1 v2 v3 =
  group
    (string prefix ^^ v1 ^^ string infix1 ^^ space ^^ v2 ^^ string infix2 ^^ space ^^ v3
   ^^ string suffix)

let binop_adapter (prefix, infix, suffix) v1 v2 =
  group (string prefix ^^ v1 ^^ string infix ^^ space ^^ v2 ^^ string suffix)

let unop_adapter (prefix, suffix) v = group (string prefix ^^ v ^^ string suffix)

(* Updated config interface to use PPrint documents *)
module type C_syntax_config = sig
  val procs : Low_level.optimized array
  (** The low-level prcedure to compile, and the arrays of the context it will be linked to if not
      shared and already known. *)

  type buffer_ptr

  val use_host_memory : (size_in_bytes:int -> unit Ctypes.ptr -> buffer_ptr) option
  val logs_to_stdout : bool
  val main_kernel_prefix : string
  val kernel_prep_line : string
  val buffer_prefix : string
  val buffer_suffix : pos:int -> string
  val arg_int_prefix : string
  val extra_args : string list
  val includes : string list
  val extra_declarations : string list
  val typ_of_prec : Ops.prec -> string
  val ident_blacklist : string list
  val ternop_syntax : Ops.prec -> Ops.ternop -> document -> document -> document -> document
  val binop_syntax : Ops.prec -> Ops.binop -> document -> document -> document
  val unop_syntax : Ops.prec -> Ops.unop -> document -> document
  val convert_precision : from:Ops.prec -> to_:Ops.prec -> string * string
end

module Pure_C_config (Input : sig
  type buffer_ptr

  val use_host_memory : (size_in_bytes:int -> unit Ctypes.ptr -> buffer_ptr) option
  val procs : Low_level.optimized array
end) =
struct
  let procs = Input.procs

  type nonrec buffer_ptr = Input.buffer_ptr

  let use_host_memory = Input.use_host_memory
  let logs_to_stdout = false
  let main_kernel_prefix = ""
  let kernel_prep_line = ""
  let buffer_prefix = ""
  let buffer_suffix = fun ~pos:_ -> ""
  let arg_int_prefix = "const int "
  let extra_args = []
  let includes = [ "<stdio.h>"; "<stdlib.h>"; "<string.h>"; "<math.h>" ]
  let extra_declarations = []
  let typ_of_prec = Ops.c_typ_of_prec

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
            if String.is_suffix p ~suffix:"(" then functions := Set.add !functions (remove_paren p)));
    Set.to_list !functions

  let ternop_syntax prec op v1 v2 v3 =
    let prefix, infix1, infix2, suffix = Ops.ternop_c_syntax prec op in
    ternop_adapter (prefix, infix1, infix2, suffix) v1 v2 v3

  let binop_syntax prec op =
    match op with
    | Ops.Satur01_gate -> (
        match prec with
        | Ops.Byte_prec _ ->
            fun v1 v2 ->
              group
                (parens
                   (parens
                      (parens (string "(float)" ^^ v1)
                      ^^ string " > 0.0f && (float)" ^^ v1 ^^ string " < 1.0f")
                   ^^ string " ? " ^^ v2 ^^ string " : (unsigned char)0"))
        | Ops.Half_prec _ ->
            fun v1 v2 ->
              group
                (parens
                   (parens (v1 ^^ string " > 0.0f16 && " ^^ v1 ^^ string " < 1.0f16")
                   ^^ string " ? " ^^ v2 ^^ string " : 0.0f16"))
        | Ops.Single_prec _ ->
            fun v1 v2 ->
              group
                (parens
                   (parens (v1 ^^ string " > 0.0f && " ^^ v1 ^^ string " < 1.0f")
                   ^^ string " ? " ^^ v2 ^^ string " : 0.0f"))
        | Ops.Double_prec _ ->
            fun v1 v2 ->
              group
                (parens
                   (parens (v1 ^^ string " > 0.0 && " ^^ v1 ^^ string " < 1.0")
                   ^^ string " ? " ^^ v2 ^^ string " : 0.0"))
        | Ops.Void_prec -> invalid_arg "Pure_C_config.binop_syntax: Satur01_gate on Void_prec")
    | _ ->
        let prefix, infix, suffix = Ops.binop_c_syntax prec op in
        fun v1 v2 -> binop_adapter (prefix, infix, suffix) v1 v2

  let unop_syntax prec op v =
    let prefix, suffix = Ops.unop_c_syntax prec op in
    unop_adapter (prefix, suffix) v

  let convert_precision = Ops.c_convert_precision
end

module C_syntax (B : C_syntax_config) = struct
  let get_ident =
    Low_level.get_ident_within_code ~no_dots:true ~blacklist:B.ident_blacklist
    @@ Array.map B.procs ~f:(fun l -> l.llc)

  let in_ctx tn = B.(Tn.is_in_context_force ~use_host_memory tn 46)

  (* Helper to convert from Indexing types to PPrint documents *)
  let axis_index_to_doc idx =
    match idx with
    | Indexing.Iterator sym -> string (Indexing.symbol_ident sym)
    | Indexing.Fixed_idx i -> string (Int.to_string i)

  (* Create a document for array offsets *)
  let array_offset_to_doc (idcs, dims) =
    assert (not @@ Array.is_empty idcs);
    let rec build_nested i =
      if i >= Array.length idcs then empty
      else
        let dim = dims.(i) in
        let idx_doc = axis_index_to_doc idcs.(i) in
        if i = 0 then idx_doc
        else if i = Array.length idcs - 1 then
          string " * " ^^ string (Int.to_string dim) ^^ string " + " ^^ idx_doc
        else
          let inner =
            string " * " ^^ string (Int.to_string dim) ^^ string " + " ^^ build_nested (i + 1)
          in
          if i <= Array.length idcs - 3 then parens inner else inner
    in
    group (build_nested 0)

  (* Helpers to render a document to a string *)
  let doc_to_string doc =
    let buf = Buffer.create 32 in
    ToBuffer.pretty 0.9 80 buf doc;
    Buffer.contents buf

  let vsep docs = separate hardline docs

  (* Print declarations and imports *)
  let print_declarations out =
    let include_docs = List.map B.includes ~f:(fun s -> string "#include " ^^ string s) in
    let declaration_docs = List.map B.extra_declarations ~f:string in
    let doc =
      align (vsep include_docs) ^^ hardline ^^ hardline ^^ align (vsep declaration_docs) ^^ hardline
    in
    ToChannel.pretty 0.9 80 out doc

  (* Main code generation function *)
  let compile_main ~traced_store:_ llc =
    let visited = Hash_set.create (module Tn) in

    (* Recursive functions to convert Low_level constructs to PPrint documents *)
    let rec ll_to_doc c =
      match c with
      | Low_level.Noop -> empty
      | Seq (c1, c2) ->
          let docs =
            List.filter_map [ c1; c2 ] ~f:(function Noop -> None | c -> Some (ll_to_doc c))
          in
          vsep docs
      | For_loop { index = i; from_; to_; body; trace_it = _ } ->
          let idx_doc = string (Indexing.symbol_ident i) in
          let for_header =
            group
              (string "for (int " ^^ idx_doc ^^ string " = "
              ^^ string (Int.to_string from_)
              ^^ string "; " ^^ idx_doc ^^ string " <= "
              ^^ string (Int.to_string to_)
              ^^ string "; ++" ^^ idx_doc ^^ string ") {")
          in
          let body_doc = ll_to_doc body in
          group (for_header ^^ nest 2 (break 1 ^^ body_doc) ^^ break 1 ^^ string "}")
      | Zero_out tn ->
          ll_to_doc
            (Low_level.loop_over_dims (Lazy.force tn.dims) ~body:(fun idcs ->
                 Low_level.Set
                   { tn; idcs; llv = Low_level.Constant 0.0; debug = get_ident tn ^ " := 0" }))
      | Set { tn; idcs; llv; debug } ->
          Hash_set.add visited tn;
          let ident = get_ident tn in
          let dims = Lazy.force tn.dims in
          let value_doc = float_to_doc (Lazy.force tn.prec) llv in
          let offset_doc = array_offset_to_doc (idcs, dims) in

          let num_closing_braces, with_locals_doc = top_locals_to_doc llv in
          let num_typ = B.typ_of_prec @@ Lazy.force tn.prec in

          if Utils.debug_log_from_routines () then
            (* ... debug logging code ... *)
            empty (* Simplified for now *)
          else
            (* Regular assignment *)
            let assign_doc =
              group (string ident ^^ brackets offset_doc ^^ string " = " ^^ value_doc ^^ string ";")
            in
            with_locals_doc ^^ assign_doc
            ^^
            if num_closing_braces > 0 then break 1 ^^ string (String.make num_closing_braces '}')
            else empty
      | Comment message ->
          if Utils.debug_log_from_routines () then
            (* ... debug logging code ... *)
            empty (* Simplified for now *)
          else string "/* " ^^ string message ^^ string " */"
      | Staged_compilation callback ->
          callback ();
          empty
      | Set_local (Low_level.{ scope_id; tn = { prec; _ } }, value) ->
          let num_closing_braces, with_locals_doc = top_locals_to_doc value in
          let value_doc = float_to_doc (Lazy.force prec) value in

          with_locals_doc
          ^^ group
               (string "v"
               ^^ string (Int.to_string scope_id)
               ^^ string " = " ^^ value_doc ^^ string ";")
          ^^
          if num_closing_braces > 0 then break 1 ^^ string (String.make num_closing_braces '}')
          else empty
    (* Handle local scope declarations *)
    and top_locals_to_doc (vcomp : Low_level.float_t) =
      match vcomp with
      | Local_scope { id = { scope_id = i; tn = { prec; _ } }; body; orig_indices = _ } ->
          let num_typ = B.typ_of_prec @@ Lazy.force prec in
          let body_doc = ll_to_doc body in

          ( 1,
            group
              (string "{ " ^^ string num_typ ^^ string " v"
              ^^ string (Int.to_string i)
              ^^ string " = 0; "
              ^^ nest 2 (break 1 ^^ body_doc)
              ^^ space) )
      | Get_local _ | Get_global _ | Get _ | Constant _ | Embed_index _ -> (0, empty)
      | Binop (Arg1, v1, _v2) -> top_locals_to_doc v1
      | Binop (Arg2, _v1, v2) -> top_locals_to_doc v2
      | Ternop (_, v1, v2, v3) ->
          let n1, d1 = top_locals_to_doc v1 in
          let n2, d2 = top_locals_to_doc v2 in
          let n3, d3 = top_locals_to_doc v3 in
          (n1 + n2 + n3, d1 ^^ d2 ^^ d3)
      | Binop (_, v1, v2) ->
          let n1, d1 = top_locals_to_doc v1 in
          let n2, d2 = top_locals_to_doc v2 in
          (n1 + n2, d1 ^^ d2)
      | Unop (_, v) -> top_locals_to_doc v
    (* Convert floating point expressions to documents *)
    and float_to_doc (prec : Ops.prec) value =
      match value with
      | Local_scope { id; _ } ->
          (* Embedding of Local_scope is done by top_locals_to_doc *)
          float_to_doc prec @@ Low_level.Get_local id
      | Get_local id ->
          let prefix, postfix = B.convert_precision ~from:(Lazy.force id.tn.prec) ~to_:prec in
          string prefix ^^ string "v" ^^ string (Int.to_string id.scope_id) ^^ string postfix
      | Get_global (Ops.Merge_buffer { source_node_id }, Some idcs) ->
          let tn = Option.value_exn ~here:[%here] @@ Tn.find ~id:source_node_id in
          let prefix, postfix = B.convert_precision ~from:(Lazy.force tn.prec) ~to_:prec in
          let dims = Lazy.force tn.dims in
          group
            (string prefix ^^ string "merge_buffer"
            ^^ brackets (array_offset_to_doc (idcs, dims))
            ^^ string postfix)
      | Get_global _ -> failwith "C_syntax: Get_global / FFI NOT IMPLEMENTED YET"
      | Get (tn, idcs) ->
          Hash_set.add visited tn;
          let ident = get_ident tn in
          let prefix, postfix = B.convert_precision ~from:(Lazy.force tn.prec) ~to_:prec in
          group
            (string prefix ^^ string ident
            ^^ brackets (array_offset_to_doc (idcs, Lazy.force tn.dims))
            ^^ string postfix)
      | Constant c ->
          let prefix, postfix = B.convert_precision ~from:Ops.double ~to_:prec in
          let prefix, postfix =
            if String.is_empty prefix && Float.(c < 0.0) then ("(", ")" ^ postfix)
            else (prefix, postfix)
          in
          string prefix ^^ string (Float.to_string_hum ~decimals:16 c) ^^ string postfix
      | Embed_index idx ->
          let prefix, postfix = B.convert_precision ~from:Ops.double ~to_:prec in
          string prefix ^^ axis_index_to_doc idx ^^ string postfix
      | Binop (Arg1, v1, _v2) -> float_to_doc prec v1
      | Binop (Arg2, _v1, v2) -> float_to_doc prec v2
      | Ternop (op, v1, v2, v3) ->
          B.ternop_syntax prec op (float_to_doc prec v1) (float_to_doc prec v2)
            (float_to_doc prec v3)
      | Binop (op, v1, v2) -> B.binop_syntax prec op (float_to_doc prec v1) (float_to_doc prec v2)
      | Unop (op, v) -> B.unop_syntax prec op (float_to_doc prec v)
    in

    ll_to_doc llc

  (* Similar to debug_float, now using PPrint *)
  let debug_float_to_doc prec value =
    (* Implementation simplified for brevity *)
    let msg = "debug_float_to_doc not yet implemented" in
    (string msg, [])

  (* Main entry point for compilation *)
  let compile_proc ~name out idx_params lowered =
    let params = (* ... parameter collection logic ... *) [] in

    (* Create function header *)
    let arg_docs =
      List.map params ~f:(fun (pname, _) ->
          string (B.buffer_prefix ^ pname ^ B.buffer_suffix ~pos:0))
    in
    let extra_arg_docs = List.map B.extra_args ~f:string in
    let all_args = arg_docs @ extra_arg_docs in

    (* Create full function document *)
    let body_doc = compile_main ~traced_store:lowered.Low_level.traced_store lowered in
    let func_doc =
      group
        (string B.main_kernel_prefix
        ^^ (if String.is_empty B.main_kernel_prefix then empty else space)
        ^^ string "void " ^^ string name
        ^^ group (parens (align (separate (string ", ") all_args)))
        ^^ space
        ^^ braces
             (nest 2
                (break 1
                ^^ (if String.is_empty B.kernel_prep_line then empty
                    else string B.kernel_prep_line ^^ break 1)
                ^^ body_doc)
             ^^ break 1))
    in

    (* Output the document *)
    ToChannel.pretty 0.9 80 out func_doc;
    hardline |> ToChannel.pretty 0.9 80 out;

    params
end
