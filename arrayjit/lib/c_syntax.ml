open Base
module Lazy = Utils.Lazy
open Backend_intf

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

module Tn = Tnode

type t = PPrint.document

let ternop_adapter (pref, infix1, infix2, suf) v1 v2 v3 =
  let open PPrint in
  group (string pref ^^ v1 ^^ string infix1 ^^ space ^^ v2 ^^ string infix2 ^^ space ^^ v3 ^^ string suf)

let binop_adapter (pref, inf, suf) v1 v2 =
  let open PPrint in
  group (string pref ^^ v1 ^^ string inf ^^ space ^^ v2 ^^ string suf)

let unop_adapter (pref, suf) v1 =
  let open PPrint in
  group (string pref ^^ v1 ^^ string suf)

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

  val ternop_syntax :
    Ops.prec ->
    Ops.ternop ->
    t ->
    t ->
    t ->
    t

  val binop_syntax :
    Ops.prec -> Ops.binop -> t -> t -> t

  val unop_syntax : Ops.prec -> Ops.unop -> t -> t
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
    let (prefix, infix1, infix2, suffix) = Ops.ternop_c_syntax prec op in
    ternop_adapter (prefix, infix1, infix2, suffix) v1 v2 v3

  let binop_syntax prec op =
    match op with
    | Ops.Satur01_gate -> (
        match prec with
        | Ops.Byte_prec _ ->
            fun v1 v2 ->
              let open PPrint in
              group (string "(((float)" ^^ v1 ^^ string " > 0.0f && (float)" ^^ v1 ^^ string " < 1.0f) ? " ^^ v2 ^^ string " : (unsigned char)0)")
        | Ops.Half_prec _ ->
            fun v1 v2 ->
              let open PPrint in
              group (string "((" ^^ v1 ^^ string " > 0.0f16 && " ^^ v1 ^^ string " < 1.0f16) ? " ^^ v2 ^^ string " : 0.0f16)")
        | Ops.Single_prec _ ->
            fun v1 v2 ->
              let open PPrint in
              group (string "((" ^^ v1 ^^ string " > 0.0f && " ^^ v1 ^^ string " < 1.0f) ? " ^^ v2 ^^ string " : 0.0f)")
        | Ops.Double_prec _ ->
            fun v1 v2 ->
              let open PPrint in
              group (string "((" ^^ v1 ^^ string " > 0.0 && " ^^ v1 ^^ string " < 1.0) ? " ^^ v2 ^^ string " : 0.0)")
        | Ops.Void_prec -> invalid_arg "Pure_C_config.binop_syntax: Satur01_gate on Void_prec")
    | _ -> 
      let (prefix, infix, suffix) = Ops.binop_c_syntax prec op in
      fun v1 v2 -> binop_adapter (prefix, infix, suffix) v1 v2

  let unop_syntax prec op v =
    let (prefix, suffix) = Ops.unop_c_syntax prec op in
    unop_adapter (prefix, suffix) v

  let convert_precision = Ops.c_convert_precision
end

module C_syntax (B : C_syntax_config) = struct
  let get_ident =
    Low_level.get_ident_within_code ~no_dots:true ~blacklist:B.ident_blacklist
    @@ Array.map B.procs ~f:(fun l -> l.llc)

  let in_ctx tn = B.(Tn.is_in_context_force ~use_host_memory tn 46)

  open Indexing.Pp_helpers

  let with_formatter f =
    let b = Buffer.create 32 in
    let ppf = Stdlib.Format.formatter_of_buffer b in
    f ppf;
    Stdlib.Format.pp_print_flush ppf ();
    Buffer.contents b

  let doc_of_axis_index idx =
    let open PPrint in
    match idx with
    | Indexing.Fixed_idx i -> string (Int.to_string i)
    | Iterator s -> string (Indexing.symbol_ident s)

  let doc_array_offset (idcs, dims) =
    let open PPrint in
    assert (not @@ Array.is_empty idcs);
    let build_expr i =
      if i = 0 then
        doc_of_axis_index idcs.(i)
      else if i = Array.length idcs - 1 then
        string " * " ^^ string (Int.to_string dims.(i)) ^^ string " + " ^^ doc_of_axis_index idcs.(i)
      else
        group (string " * " ^^ string (Int.to_string dims.(i)) ^^ string " + " ^^ doc_of_axis_index idcs.(i))
    in
    let rec nest_parens i result =
      if i >= Array.length idcs - 2 then result
      else nest_parens (i + 1) (string "(" ^^ result ^^ string ")")
    in
    nest_parens 0 (Array.fold (Array.init (Array.length idcs) ~f:Fn.id) ~init:PPrint.empty ~f:(fun acc i -> 
      if i = 0 then build_expr i else acc ^^ build_expr i))

  let array_offset_to_string (idcs, dims) =
    let doc = doc_array_offset (idcs, dims) in
    let buf = Buffer.create 32 in
    PPrint.ToBuffer.pretty 0.8 80 buf doc;
    Buffer.contents buf

  (** Toplevel declarations, comprised of [includes] and [extra_declarations]. *)
  let print_declarations : t =
    let open PPrint in
    let include_docs = List.map B.includes ~f:(fun incl -> string "#include " ^^ string incl) in
    let decl_docs = List.map B.extra_declarations ~f:string in
    let vbox = group in
    vbox (separate_map hardline (fun x -> x) include_docs) ^^ hardline ^^ hardline ^^
    vbox (separate_map hardline (fun x -> x) decl_docs) ^^ hardline ^^ hardline
      
  let compile_main ~traced_store llc : t =
    let open PPrint in
    let vbox = group in
    let visited = Hash_set.create (module Tn) in
    let rec pp_ll c : t =
      match c with
      | Low_level.Noop -> empty
      | Seq (c1, c2) ->
          (* Note: no separator. Filter out some entries known to not generate code to avoid
             whitespace. *)
          vbox (
            List.filter_map [ c1; c2 ] ~f:(function 
              | Noop -> None 
              | c -> Some (pp_ll c))
            |> separate_map empty (fun x -> x)
          )
      | For_loop { index = i; from_; to_; body; trace_it = _ } ->
          let sym_i = string (Indexing.symbol_ident i) in
          group (string "for (int " ^^ sym_i ^^ string " = " ^^ string (Int.to_string from_) ^^ 
                 string "; " ^^ sym_i ^^ string " <= " ^^ string (Int.to_string to_) ^^ 
                 string "; ++" ^^ sym_i ^^ string ") {" ^^ hardline ^^
                 (if Utils.debug_log_from_routines () then
                   if B.logs_to_stdout then
                     let msg = with_formatter (fun fmt -> 
                       let prefix = !Utils.captured_log_prefix in
                       Stdlib.Format.fprintf fmt "%s%%d: index %s = %%d" prefix (Indexing.symbol_ident i)) in
                     let msg = String.substr_replace_all msg ~pattern:"\n" ~with_:"$" in
                     group (string (Printf.sprintf "printf(\"%s\\n\", log_id, %s);" msg (Indexing.symbol_ident i)) ^^ space)
                   else
                     let msg = with_formatter (fun fmt -> 
                       Stdlib.Format.fprintf fmt "index %s = %%d" (Indexing.symbol_ident i)) in
                     let msg = String.substr_replace_all msg ~pattern:"\n" ~with_:"$" in
                     group (string (Printf.sprintf "fprintf(log_file, \"%s\\n\", %s);" msg (Indexing.symbol_ident i)) ^^ space)
                 else empty) ^^
                 nest 2 (pp_ll body) ^^ hardline ^^ string "}")
      | Zero_out tn ->
          pp_ll
            (Low_level.loop_over_dims (Lazy.force tn.dims) ~body:(fun idcs ->
                 Set { tn; idcs; llv = Constant 0.0; debug = get_ident tn ^ " := 0" }))
      | Set { tn; idcs; llv; debug } ->
          Hash_set.add visited tn;
          let ident = get_ident tn in
          let dims = Lazy.force tn.dims in
          let num_typ = B.typ_of_prec @@ Lazy.force tn.prec in
          let closing_braces, doc_locals = pp_top_locals llv in
          let doc = pp_float (Lazy.force tn.prec) llv in
          let doc_debug =
            if Utils.debug_log_from_routines () then
              let v_code, v_idcs = debug_float (Lazy.force tn.prec) llv in
              let args_text = String.concat (List.map v_idcs ~f:(function
                | `Accessor idx -> ", " ^ array_offset_to_string idx
                | `Value v -> ", " ^ v)) in
              let offset = array_offset_to_string (idcs, dims) in
              if B.logs_to_stdout then
                let comment_msg = with_formatter (fun fmt -> Stdlib.Format.fprintf fmt "%s%%d: # %s" !Utils.captured_log_prefix debug) in
                let comment_msg = String.substr_replace_all comment_msg ~pattern:"\n" ~with_:"$" in
                let printf_cmd = Printf.sprintf "printf(\"%s\\n\", log_id);" comment_msg in
                
                let value_msg = with_formatter (fun fmt -> Stdlib.Format.fprintf fmt "%s%%d: %s[%%u]{=%%g} = %%g = %s" !Utils.captured_log_prefix ident v_code) in
                let value_msg = String.substr_replace_all value_msg ~pattern:"\n" ~with_:"$" in
                let printf_val_cmd = Printf.sprintf "printf(\"%s\\n\", log_id, %s, %s[%s], new_set_v%s);" 
                  value_msg offset ident offset args_text in
                
                group (string "{ " ^^ 
                       string (num_typ ^ " new_set_v = ") ^^ doc ^^ string ";" ^^ hardline ^^
                       string printf_cmd ^^ hardline ^^
                       string printf_val_cmd ^^ hardline ^^
                       string (ident ^ "[") ^^ doc_array_offset (idcs, dims) ^^ string "] = new_set_v;" ^^ hardline ^^
                       string "}")
              else
                let comment_msg = with_formatter (fun fmt -> Stdlib.Format.fprintf fmt "# %s" debug) in
                let comment_msg = String.substr_replace_all comment_msg ~pattern:"\n" ~with_:"$" in
                let fprintf_cmd = Printf.sprintf "fprintf(log_file, \"%s\\n\");" comment_msg in
                
                let value_msg = with_formatter (fun fmt -> Stdlib.Format.fprintf fmt "%s[%%u]{=%%g} = %%g = %s" ident v_code) in
                let value_msg = String.substr_replace_all value_msg ~pattern:"\n" ~with_:"$" in
                let fprintf_val_cmd = Printf.sprintf "fprintf(log_file, \"%s\\n\", %s, %s[%s], new_set_v%s);" 
                  value_msg offset ident offset args_text in
                
                group (string "{ " ^^ 
                       string (num_typ ^ " new_set_v = ") ^^ doc ^^ string ";" ^^ hardline ^^
                       string fprintf_cmd ^^ hardline ^^
                       string fprintf_val_cmd ^^ hardline ^^
                       string "fflush(log_file);" ^^ hardline ^^
                       string (ident ^ "[") ^^ doc_array_offset (idcs, dims) ^^ string "] = new_set_v;" ^^ hardline ^^
                       string "}")
            else
              group (string (ident ^ "[") ^^ doc_array_offset (idcs, dims) ^^ string "] = " ^^ doc ^^ string ";")
          in
          let closing_doc = List.fold ~init:empty ~f:(fun acc _ -> acc ^^ string "}" ^^ hardline) (List.init closing_braces ~f:ignore) in
          (doc_locals ^^ doc_debug ^^ closing_doc)
      | Comment message ->
          if Utils.debug_log_from_routines () then
            if B.logs_to_stdout then
              let msg = with_formatter (fun fmt -> Stdlib.Format.fprintf fmt "%s%%d: COMMENT: %s" !Utils.captured_log_prefix message) in
              let msg = String.substr_replace_all msg ~pattern:"\n" ~with_:"$" in
              string (Printf.sprintf "printf(\"%s\\n\", log_id);" msg)
            else
              let msg = with_formatter (fun fmt -> Stdlib.Format.fprintf fmt "COMMENT: %s" message) in
              let msg = String.substr_replace_all msg ~pattern:"\n" ~with_:"$" in
              string (Printf.sprintf "fprintf(log_file, \"%s\\n\");" msg)
          else
            string ("/* " ^ message ^ " */")
      | Staged_compilation callback -> 
          callback ();
          empty
      | Set_local (Low_level.{ scope_id; tn = { prec; _ } }, value) ->
          let closing_braces, doc_locals = pp_top_locals value in
          let prec_val = Lazy.force prec in 
          let value_doc = pp_float prec_val value in
          let assignment_doc = 
            group (string (Printf.sprintf "v%d = " scope_id) ^^ value_doc ^^ string ";")
          in
          let closing_doc = List.fold ~init:empty ~f:(fun acc _ -> acc ^^ string "}" ^^ hardline) (List.init closing_braces ~f:ignore) in
          doc_locals ^^ assignment_doc ^^ closing_doc
    
    and pp_top_locals (vcomp : Low_level.float_t) : int * t =
      match vcomp with
      | Local_scope { id = { scope_id = i; tn = { prec; _ } }; body; orig_indices = _ } ->
          let num_typ = B.typ_of_prec @@ Lazy.force prec in
          (* Arrays are initialized to 0 by default. However, there is typically an explicit
             initialization for virtual nodes. *)
          (1, 
           group (string ("{ " ^ num_typ ^ " v" ^ Int.to_string i ^ " = 0;") ^^ hardline ^^ 
                  pp_ll body ^^ space))
      | Get_local _ | Get_global _ | Get _ | Constant _ | Embed_index _ -> (0, empty)
      | Binop (Arg1, v1, _v2) -> pp_top_locals v1
      | Binop (Arg2, _v1, v2) -> pp_top_locals v2
      | Ternop (_, v1, v2, v3) -> 
          let n1, d1 = pp_top_locals v1 in
          let n2, d2 = pp_top_locals v2 in
          let n3, d3 = pp_top_locals v3 in
          (n1 + n2 + n3, d1 ^^ d2 ^^ d3)
      | Binop (_, v1, v2) -> 
          let n1, d1 = pp_top_locals v1 in
          let n2, d2 = pp_top_locals v2 in
          (n1 + n2, d1 ^^ d2)
      | Unop (_, v) -> pp_top_locals v
    
    and pp_float (prec : Ops.prec) (value : Low_level.float_t) : t =
      match value with
      | Local_scope { id; _ } ->
          (* Embedding of Local_scope is done by pp_top_locals. *)
          pp_float prec @@ Get_local id
      | Get_local id ->
          let prefix, postfix = B.convert_precision ~from:(Lazy.force id.tn.prec) ~to_:prec in
          string (prefix ^ "v" ^ Int.to_string id.scope_id ^ postfix)
      | Get_global (Ops.Merge_buffer { source_node_id }, Some idcs) ->
          let tn = Option.value_exn ~here:[%here] @@ Tn.find ~id:source_node_id in
          let prefix, postfix = B.convert_precision ~from:(Lazy.force tn.prec) ~to_:prec in
          group (string (prefix ^ "merge_buffer[") ^^ doc_array_offset (idcs, Lazy.force tn.dims) ^^ string ("]" ^ postfix))
      | Get_global _ -> failwith "C_syntax: Get_global / FFI NOT IMPLEMENTED YET"
      | Get (tn, idcs) ->
          Hash_set.add visited tn;
          let ident = get_ident tn in
          let prefix, postfix = B.convert_precision ~from:(Lazy.force tn.prec) ~to_:prec in
          group (string (prefix ^ ident ^ "[") ^^ doc_array_offset (idcs, Lazy.force tn.dims) ^^ string ("]" ^ postfix))
      | Constant c ->
          let prefix, postfix = B.convert_precision ~from:Ops.double ~to_:prec in
          let prefix, postfix =
            if String.is_empty prefix && Float.(c < 0.0) then ("(", ")" ^ postfix)
            else (prefix, postfix)
          in
          string (prefix ^ Printf.sprintf "%.16g" c ^ postfix)
      | Embed_index idx ->
          let prefix, postfix = B.convert_precision ~from:Ops.double ~to_:prec in
          string prefix ^^ doc_of_axis_index idx ^^ string postfix
      | Binop (Arg1, v1, _v2) -> pp_float prec v1
      | Binop (Arg2, _v1, v2) -> pp_float prec v2
      | Ternop (op, v1, v2, v3) -> 
          B.ternop_syntax prec op (pp_float prec v1) (pp_float prec v2) (pp_float prec v3)
      | Binop (op, v1, v2) -> 
          B.binop_syntax prec op (pp_float prec v1) (pp_float prec v2)
      | Unop (op, v) -> 
          B.unop_syntax prec op (pp_float prec v)
    
    and debug_float (prec : Ops.prec) (value : Low_level.float_t) : string * 'a list =
      let loop = debug_float prec in
      match value with
      | Local_scope { id; _ } ->
          (* Not printing the inlined definition: (1) code complexity; (2) don't overload the debug
             logs. *)
          loop @@ Get_local id
      | Get_local id ->
          let prefix, postfix = B.convert_precision ~from:(Lazy.force id.tn.prec) ~to_:prec in
          let v = String.concat [ prefix; "v"; Int.to_string id.scope_id; postfix ] in
          (v ^ "{=%g}", [ `Value v ])
      | Get_global (Ops.Merge_buffer { source_node_id }, Some idcs) ->
          let tn = Option.value_exn ~here:[%here] @@ Tn.find ~id:source_node_id in
          let prefix, postfix = B.convert_precision ~from:(Lazy.force tn.prec) ~to_:prec in
          let dims = Lazy.force tn.dims in
          let v =
            Printf.sprintf "%smerge_buffer[%s]%s" prefix
              (array_offset_to_string (idcs, dims))
              postfix
          in
          ( String.concat [ prefix; "merge_buffer[%u]"; postfix; "{=%g}" ],
            [ `Accessor (idcs, dims); `Value v ] )
      | Get_global _ -> failwith "Exec_as_cuda: Get_global / FFI NOT IMPLEMENTED YET"
      | Get (tn, idcs) ->
          let dims = Lazy.force tn.dims in
          let ident = get_ident tn in
          let prefix, postfix = B.convert_precision ~from:(Lazy.force tn.prec) ~to_:prec in
          let v =
            Printf.sprintf "%s%s[%s]%s" prefix ident
              (array_offset_to_string (idcs, dims))
              postfix
          in
          ( String.concat [ prefix; ident; "[%u]"; postfix; "{=%g}" ],
            [ `Accessor (idcs, dims); `Value v ] )
      | Constant c ->
          let prefix, postfix = B.convert_precision ~from:Ops.double ~to_:prec in
          (prefix ^ Float.to_string c ^ postfix, [])
      | Embed_index (Fixed_idx i) -> (Int.to_string i, [])
      | Embed_index (Iterator s) -> (Indexing.symbol_ident s, [])
      | Binop (Arg1, v1, _v2) -> loop v1
      | Binop (Arg2, _v1, v2) -> loop v2
      | Ternop (op, v1, v2, v3) ->
          let v1, idcs1 = loop v1 in
          let v2, idcs2 = loop v2 in
          let v3, idcs3 = loop v3 in
          let doc1 = PPrint.string v1 in
          let doc2 = PPrint.string v2 in
          let doc3 = PPrint.string v3 in
          let doc = B.ternop_syntax prec op doc1 doc2 doc3 in
          let result = 
            let buf = Buffer.create 128 in
            PPrint.ToBuffer.pretty 0.8 80 buf doc;
            Buffer.contents buf
          in
          (result, idcs1 @ idcs2 @ idcs3)
      | Binop (op, v1, v2) ->
          let v1, idcs1 = loop v1 in
          let v2, idcs2 = loop v2 in
          let doc1 = PPrint.string v1 in
          let doc2 = PPrint.string v2 in
          let doc = B.binop_syntax prec op doc1 doc2 in
          let result = 
            let buf = Buffer.create 128 in
            PPrint.ToBuffer.pretty 0.8 80 buf doc;
            Buffer.contents buf
          in
          (result, idcs1 @ idcs2)
      | Unop (op, v) ->
          let v, idcs = loop v in
          let doc1 = PPrint.string v in
          let doc = B.unop_syntax prec op doc1 in
          let result = 
            let buf = Buffer.create 128 in
            PPrint.ToBuffer.pretty 0.8 80 buf doc;
            Buffer.contents buf
          in
          (result, idcs)
    in
    pp_ll llc

  let compile_proc ~name (ppf : Stdlib.Format.formatter) idx_params Low_level.{ traced_store; llc; merge_node } =
    let open PPrint in
    let vbox = group in
    let params : (string * param_source) list =
      (* Preserve the order in the hashtable. *)
      List.rev
      @@ Hashtbl.fold traced_store ~init:[] ~f:(fun ~key:tn ~data:_ params ->
             (* A rough approximation to the type Gccjit_backend.mem_properties. *)
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
             (* We often don't know ahead of linking with relevant contexts what the stream sharing
                mode of the node will become. Conservatively, use passing as argument. *)
             if is_param then
               (B.typ_of_prec (Lazy.force tn.Tn.prec) ^ " *" ^ get_ident tn, Param_ptr tn) :: params
             else params)
    in
    let idx_params =
      List.map idx_params ~f:(fun s ->
          (B.arg_int_prefix ^ Indexing.symbol_ident s.Indexing.static_symbol, Static_idx s))
    in
    let log_file =
      (* FIXME: this is a hack that should be fixed by the backends. *)
      if Utils.debug_log_from_routines () then
        [
          ((if B.logs_to_stdout then "int log_id" else "const char* log_file_name"), Log_file_name);
        ]
      else []
    in
    let merge_param =
      Option.(
        to_list
        @@ map merge_node ~f:(fun tn ->
               ("const " ^ B.typ_of_prec (Lazy.force tn.prec) ^ " *merge_buffer", Merge_buffer)))
    in
    let params = log_file @ merge_param @ idx_params @ params in
    let params =
      List.sort params ~compare:(fun (p1_name, _) (p2_name, _) -> compare_string p1_name p2_name)
    in
    let args =
      List.mapi ~f:(fun pos (name, _) -> B.buffer_prefix ^ name ^ B.buffer_suffix ~pos) params
      @ B.extra_args
    in
    
    (* Construct the document *)
    let args_doc = separate_map (string "," ^^ break 1) string args in
    let func_decl = 
      group (string (B.main_kernel_prefix ^ (if String.is_empty B.main_kernel_prefix then "" else " ") ^ "void " ^ name ^ "(") ^^ 
             nest 4 (group args_doc) ^^ string ")") 
    in
    
    let body = 
      vbox (
        string "{" ^^ hardline ^^
        (if not (String.is_empty B.kernel_prep_line) then string B.kernel_prep_line ^^ hardline else empty) ^^
        (if (not (List.is_empty log_file)) && not B.logs_to_stdout then 
           string "FILE* log_file = fopen(log_file_name, \"w\");" ^^ hardline 
         else empty) ^^
        
        (if Utils.debug_log_from_routines () then
           string "/* Debug initial parameter state. */" ^^ hardline ^^
           concat_map params ~f:(function
             | p_name, Merge_buffer ->
                 if B.logs_to_stdout then
                   string (Printf.sprintf "printf(\"%s%%d: %s &[%d] = %%p\\n\", log_id, (void*)merge_buffer);" 
                            !Utils.captured_log_prefix p_name (Tnode.num_elems @@ Option.value_exn merge_node)) ^^ hardline
                 else
                   string (Printf.sprintf "fprintf(log_file, \"%s &[%d] = %%p\\n\", (void*)merge_buffer);" 
                            p_name (Tnode.num_elems @@ Option.value_exn merge_node)) ^^ hardline
             | _, Log_file_name -> empty
             | p_name, Param_ptr tn ->
                 if B.logs_to_stdout then
                   string (Printf.sprintf "printf(\"%s%%d: %s &[%d] = %%p\\n\", log_id, (void*)%s);" 
                            !Utils.captured_log_prefix p_name (Tnode.num_elems tn) (get_ident tn)) ^^ hardline
                 else
                   string (Printf.sprintf "fprintf(log_file, \"%s &[%d] = %%p\\n\", (void*)%s);" 
                            p_name (Tnode.num_elems tn) (get_ident tn)) ^^ hardline
             | p_name, Static_idx s ->
                 if B.logs_to_stdout then
                   string (Printf.sprintf "printf(\"%s%%d: %s = %%d\\n\", log_id, %s);" 
                            !Utils.captured_log_prefix p_name (Indexing.symbol_ident s.Indexing.static_symbol)) ^^ hardline
                 else
                   string (Printf.sprintf "fprintf(log_file, \"%s = %%d\\n\", %s);" 
                            p_name (Indexing.symbol_ident s.Indexing.static_symbol)) ^^ hardline)
         else empty) ^^
        
        string "/* Local declarations and initialization. */" ^^ hardline ^^
        concat_map (Hashtbl.to_alist traced_store) ~f:(fun (tn, node) ->
            if not (Tn.is_virtual_force tn 333 || Tn.is_materialized_force tn 336) then
              string (Printf.sprintf "%s %s[%d]%s;" 
                       (B.typ_of_prec @@ Lazy.force tn.prec)
                       (get_ident tn) (Tn.num_elems tn)
                       (if node.zero_initialized then " = {0}" else "")) ^^ hardline
            else empty) ^^
        
        hardline ^^ string "/* Main logic. */" ^^ hardline ^^
        compile_main ~traced_store llc ^^ hardline ^^
        string "}"
      )
    in
    
    let doc = vbox (func_decl ^^ hardline ^^ body ^^ hardline) in
    ToBuffer.pretty 0.8 80 ppf doc;
    params
end
