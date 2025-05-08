open Base
module Lazy = Utils.Lazy
open Backend_intf

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

module Tn = Tnode

type fmt = Stdlib.Format.formatter

let ternop_adapter (prefix, infix1, infix2, suffix) ppf pp1 v1 pp2 v2 pp3 v3 =
  Stdlib.Format.fprintf ppf "@[<1>%s%a%s@ %a%s@ %a@]%s" prefix pp1 v1 infix1 pp2 v2 infix2 pp3 v3
    suffix

let binop_adapter (prefix, infix, suffix) ppf pp1 v1 pp2 v2 =
  Stdlib.Format.fprintf ppf "@[<1>%s%a%s@ %a@]%s" prefix pp1 v1 infix pp2 v2 suffix

let unop_adapter (prefix, suffix) ppf pp1 v1 =
  Stdlib.Format.fprintf ppf "@[<1>%s%a@]%s" prefix pp1 v1 suffix

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
    fmt ->
    (fmt -> 'a -> unit) ->
    'a ->
    (fmt -> 'b -> unit) ->
    'b ->
    (fmt -> 'c -> unit) ->
    'c ->
    unit

  val binop_syntax :
    Ops.prec -> Ops.binop -> fmt -> (fmt -> 'a -> unit) -> 'a -> (fmt -> 'b -> unit) -> 'b -> unit

  val unop_syntax : Ops.prec -> Ops.unop -> fmt -> (fmt -> 'a -> unit) -> 'a -> unit
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

  let ternop_syntax prec op = ternop_adapter (Ops.ternop_c_syntax prec op)
  let binop_syntax prec op = binop_adapter (Ops.binop_c_syntax prec op)
  let unop_syntax prec op = unop_adapter (Ops.unop_c_syntax prec op)
  let convert_precision = Ops.c_convert_precision
end

module C_syntax (B : C_syntax_config) = struct
  let get_ident =
    Low_level.get_ident_within_code ~no_dots:true ~blacklist:B.ident_blacklist
    @@ Array.map B.procs ~f:(fun l -> l.llc)

  let in_ctx tn = B.(Tn.is_in_context_force ~use_host_memory tn 46)
  let pp_include ppf s = Stdlib.Format.fprintf ppf "#include %s" s

  open Indexing.Pp_helpers

  let pp_array_offset ppf (idcs, dims) =
    let open Stdlib.Format in
    assert (not @@ Array.is_empty idcs);
    for _ = 0 to Array.length idcs - 3 do
      fprintf ppf "@[<1>("
    done;
    for i = 0 to Array.length idcs - 1 do
      let dim = dims.(i) in
      if i = 0 then fprintf ppf "%a" pp_axis_index idcs.(i)
      else if i = Array.length idcs - 1 then fprintf ppf " * %d + %a" dim pp_axis_index idcs.(i)
      else fprintf ppf " * %d +@ %a@;<0 -1>)@]" dim pp_axis_index idcs.(i)
    done

  let array_offset_to_string (idcs, dims) =
    let b = Buffer.create 32 in
    let ppf = Stdlib.Format.formatter_of_buffer b in
    pp_array_offset ppf (idcs, dims);
    Stdlib.Format.pp_print_flush ppf ();
    Buffer.contents b

  (* let compute_array_offset ~idcs ~dims = Array.fold2_exn idcs dims ~init:0 ~f:(fun offset idx dim
     -> idx + (offset * dim)) *)

  (** Toplevel declarations, comprised of [includes] and [extra_declarations]. *)
  let print_declarations ppf =
    Stdlib.Format.(fprintf ppf {|@[<v 0>%a@,@,|} (pp_print_list pp_include) B.includes);
    Stdlib.Format.(
      fprintf ppf {|@[<v 0>%a@,@,|} (pp_print_list pp_print_string) B.extra_declarations)

  let compile_main ~traced_store:_ ppf llc : unit =
    let open Stdlib.Format in
    let visited = Hash_set.create (module Tn) in
    let rec pp_ll ppf c : unit =
      match c with
      | Low_level.Noop -> ()
      | Seq (c1, c2) ->
          (* Note: no separator. Filter out some entries known to not generate code to avoid
             whitespace. *)
          fprintf ppf "@[<v 0>%a@]" (pp_print_list pp_ll)
            (List.filter [ c1; c2 ] ~f:(function Noop -> false | _ -> true))
      | For_loop { index = i; from_; to_; body; trace_it = _ } ->
          fprintf ppf "@[<2>for (int@ %a = %d;@ %a <= %d;@ ++%a) {@ " pp_symbol i from_ pp_symbol i
            to_ pp_symbol i;
          if Utils.debug_log_from_routines () then
            if B.logs_to_stdout then
              fprintf ppf {|printf(@[<h>"%s%%d: index %a = %%d\n",@] log_id, %a);@ |}
                !Utils.captured_log_prefix pp_symbol i pp_symbol i
            else
              fprintf ppf {|fprintf(log_file,@ @[<h>"index %a = %%d\n",@] %a);@ |} pp_symbol i
                pp_symbol i;
          fprintf ppf "%a@;<1 -2>}@]@," pp_ll body
      | Zero_out tn ->
          pp_ll ppf
            (Low_level.loop_over_dims (Lazy.force tn.dims) ~body:(fun idcs ->
                 Set { tn; idcs; llv = Constant 0.0; debug = get_ident tn ^ " := 0" }))
      | Set { tn; idcs; llv; debug } ->
          Hash_set.add visited tn;
          let ident = get_ident tn in
          let dims = Lazy.force tn.dims in
          let loop_f = pp_float @@ Lazy.force tn.prec in
          let loop_debug_f = debug_float @@ Lazy.force tn.prec in
          let num_closing_braces = pp_top_locals ppf llv in
          let num_typ = B.typ_of_prec @@ Lazy.force tn.prec in
          if Utils.debug_log_from_routines () then (
            fprintf ppf "@[<2>{@ @[<2>%s new_set_v =@ %a;@]@ " num_typ loop_f llv;
            let v_code, v_idcs = loop_debug_f llv in
            let pp_args =
              pp_print_list @@ fun ppf -> function
              | `Accessor idx ->
                  pp_comma ppf ();
                  pp_array_offset ppf idx
              | `Value v ->
                  pp_comma ppf ();
                  pp_print_string ppf v
            in
            let offset = (idcs, dims) in
            if B.logs_to_stdout then (
              fprintf ppf {|@[<7>printf(@[<h>"%s%%d: # %s\n", log_id@]);@]@ |}
                !Utils.captured_log_prefix
                (String.substr_replace_all debug ~pattern:"\n" ~with_:"$");
              fprintf ppf
                {|@[<7>printf(@[<h>"%s%%d: %s[%%u]{=%%g} = %%g = %s\n",@]@ log_id,@ %a,@ %s[%a],@ new_set_v%a);@]@ |}
                !Utils.captured_log_prefix ident v_code pp_array_offset offset ident pp_array_offset
                offset pp_args v_idcs)
            else (
              fprintf ppf {|@[<7>fprintf(log_file,@ @[<h>"# %s\n"@]);@]@ |}
                (String.substr_replace_all debug ~pattern:"\n" ~with_:"$");
              fprintf ppf
                {|@[<7>fprintf(log_file,@ @[<h>"%s[%%u]{=%%g} = %%g = %s\n",@]@ %a,@ %s[%a],@ new_set_v%a);@]@ |}
                ident v_code pp_array_offset offset ident pp_array_offset offset pp_args v_idcs);
            if not B.logs_to_stdout then fprintf ppf "fflush(log_file);@ ";
            fprintf ppf "@[<2>%s[@,%a] =@ new_set_v;@]@;<1 -2>}@]@ " ident pp_array_offset
              (idcs, dims))
          else
            (* No idea why adding any cut hint at the end of the assign line breaks formatting! *)
            fprintf ppf "@[<2>%s[@,%a] =@ %a;@]@ " ident pp_array_offset (idcs, dims) loop_f llv;
          for _ = 1 to num_closing_braces do
            fprintf ppf "@]@ }@,"
          done
      | Comment message ->
          if Utils.debug_log_from_routines () then
            if B.logs_to_stdout then
              fprintf ppf {|printf(@[<h>"%s%%d: COMMENT: %s\n",@] log_id);@ |}
                !Utils.captured_log_prefix
                (String.substr_replace_all ~pattern:"%" ~with_:"%%" message)
            else
              fprintf ppf {|fprintf(log_file,@ @[<h>"COMMENT: %s\n"@]);@ |}
                (String.substr_replace_all ~pattern:"%" ~with_:"%%" message)
          else fprintf ppf "/* %s */@ " message
      | Staged_compilation callback -> callback ()
      | Set_local (Low_level.{ scope_id; tn = { prec; _ } }, value) ->
          let num_closing_braces = pp_top_locals ppf value in
          fprintf ppf "@[<2>v%d =@ %a;@]" scope_id (pp_float @@ Lazy.force prec) value;
          for _ = 1 to num_closing_braces do
            fprintf ppf "@]@ }@,"
          done
    and pp_top_locals ppf (vcomp : Low_level.float_t) : int =
      match vcomp with
      | Local_scope { id = { scope_id = i; tn = { prec; _ } }; body; orig_indices = _ } ->
          let num_typ = B.typ_of_prec @@ Lazy.force prec in
          (* Arrays are initialized to 0 by default. However, there is typically an explicit
             initialization for virtual nodes. *)
          fprintf ppf "@[<2>{@ %s v%d = 0;@ " num_typ i;
          pp_ll ppf body;
          pp_print_space ppf ();
          1
      | Get_local _ | Get_global _ | Get _ | Constant _ | Embed_index _ -> 0
      | Binop (Arg1, v1, _v2) -> pp_top_locals ppf v1
      | Binop (Arg2, _v1, v2) -> pp_top_locals ppf v2
      | Ternop (_, v1, v2, v3) -> pp_top_locals ppf v1 + pp_top_locals ppf v2 + pp_top_locals ppf v3
      | Binop (_, v1, v2) -> pp_top_locals ppf v1 + pp_top_locals ppf v2
      | Unop (_, v) -> pp_top_locals ppf v
    and pp_float (prec : Ops.prec) ppf value =
      let loop = pp_float prec in
      match value with
      | Local_scope { id; _ } ->
          (* Embedding of Local_scope is done by pp_top_locals. *)
          loop ppf @@ Get_local id
      | Get_local id ->
          let prefix, postfix = B.convert_precision ~from:(Lazy.force id.tn.prec) ~to_:prec in
          fprintf ppf "%sv%d%s" prefix id.scope_id postfix
      | Get_global (Ops.Merge_buffer { source_node_id }, Some idcs) ->
          let tn = Option.value_exn ~here:[%here] @@ Tn.find ~id:source_node_id in
          let prefix, postfix = B.convert_precision ~from:(Lazy.force tn.prec) ~to_:prec in
          fprintf ppf "@[<2>%smerge_buffer[%a@;<0 -2>]%s@]" prefix pp_array_offset
            (idcs, Lazy.force tn.dims)
            postfix
      | Get_global _ -> failwith "C_syntax: Get_global / FFI NOT IMPLEMENTED YET"
      | Get (tn, idcs) ->
          Hash_set.add visited tn;
          let ident = get_ident tn in
          let prefix, postfix = B.convert_precision ~from:(Lazy.force tn.prec) ~to_:prec in
          fprintf ppf "@[<2>%s%s[%a@;<0 -2>]%s@]" prefix ident pp_array_offset
            (idcs, Lazy.force tn.dims)
            postfix
      | Constant c ->
          let prefix, postfix = B.convert_precision ~from:Ops.double ~to_:prec in
          let prefix, postfix =
            if String.is_empty prefix && Float.(c < 0.0) then ("(", ")" ^ postfix)
            else (prefix, postfix)
          in
          fprintf ppf "%s%.16g%s" prefix c postfix
      | Embed_index idx ->
          let prefix, postfix = B.convert_precision ~from:Ops.double ~to_:prec in
          fprintf ppf "%s%a%s" prefix pp_axis_index idx postfix
      | Binop (Arg1, v1, _v2) -> loop ppf v1
      | Binop (Arg2, _v1, v2) -> loop ppf v2
      | Ternop (op, v1, v2, v3) -> B.ternop_syntax prec op ppf loop v1 loop v2 loop v3
      | Binop (op, v1, v2) -> B.binop_syntax prec op ppf loop v1 loop v2
      | Unop (op, v) -> B.unop_syntax prec op ppf loop v
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
            sprintf "@[<2>%smerge_buffer[%s@;<0 -2>]%s@]" prefix
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
            sprintf "@[<2>%s%s[%s@;<0 -2>]%s@]" prefix ident
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
          let open Stdlib.Format in
          B.ternop_syntax prec op (get_str_formatter ()) pp_print_string v1 pp_print_string v2
            pp_print_string v3;
          (flush_str_formatter (), idcs1 @ idcs2 @ idcs3)
      | Binop (op, v1, v2) ->
          let v1, idcs1 = loop v1 in
          let v2, idcs2 = loop v2 in
          let open Stdlib.Format in
          B.binop_syntax prec op (get_str_formatter ()) pp_print_string v1 pp_print_string v2;
          (flush_str_formatter (), idcs1 @ idcs2)
      | Unop (op, v) ->
          let v, idcs = loop v in
          let open Stdlib.Format in
          B.unop_syntax prec op (get_str_formatter ()) pp_print_string v;
          (flush_str_formatter (), idcs)
    in
    pp_ll ppf llc

  let%track3_sexp compile_proc ~name ppf idx_params Low_level.{ traced_store; llc; merge_node } =
    let open Stdlib.Format in
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
    fprintf ppf "@[<v 2>@[<hv 4>%s%svoid %s(@,@[<hov 0>%a@]@;<0 -4>)@] {@ " B.main_kernel_prefix
      (if String.is_empty B.main_kernel_prefix then "" else " ")
      name
      (pp_print_list ~pp_sep:pp_comma pp_print_string)
      args;
    if not (String.is_empty B.kernel_prep_line) then fprintf ppf "%s@ " B.kernel_prep_line;
    (* FIXME: we should also close the file. *)
    if (not (List.is_empty log_file)) && not B.logs_to_stdout then
      fprintf ppf {|FILE* log_file = fopen(log_file_name, "w");@ |};
    if Utils.debug_log_from_routines () then (
      fprintf ppf "/* Debug initial parameter state. */@ ";
      List.iter
        ~f:(function
          | p_name, Merge_buffer ->
              if B.logs_to_stdout then
                fprintf ppf
                  {|@[<7>printf(@[<h>"%s%%d: %s &[%d] = %%p\n",@] log_id, (void*)merge_buffer);@]@ |}
                  !Utils.captured_log_prefix p_name
                  (Tnode.num_elems @@ Option.value_exn merge_node)
              else
                fprintf ppf
                  {|@[<7>fprintf(log_file,@ @[<h>"%s &[%d] = %%p\n",@] (void*)merge_buffer);@]@ |}
                  p_name
                  (Tnode.num_elems @@ Option.value_exn merge_node)
          | _, Log_file_name -> ()
          | p_name, Param_ptr tn ->
              if B.logs_to_stdout then
                fprintf ppf
                  {|@[<7>printf(@[<h>"%s%%d: %s &[%d] = %%p\n",@] log_id, (void*)%s);@]@ |}
                  !Utils.captured_log_prefix p_name (Tnode.num_elems tn)
                @@ get_ident tn
              else
                fprintf ppf {|@[<7>fprintf(log_file,@ @[<h>"%s &[%d] = %%p\n",@] (void*)%s);@]@ |}
                  p_name (Tnode.num_elems tn) (get_ident tn)
          | p_name, Static_idx s ->
              if B.logs_to_stdout then
                fprintf ppf {|@[<7>printf(@[<h>"%s%%d: %s = %%d\n",@] log_id, %s);@]@ |}
                  !Utils.captured_log_prefix p_name
                @@ Indexing.symbol_ident s.Indexing.static_symbol
              else
                fprintf ppf {|@[<7>fprintf(log_file,@ @[<h>"%s = %%d\n",@] %s);@]@ |} p_name
                @@ Indexing.symbol_ident s.Indexing.static_symbol)
        params);
    fprintf ppf "/* Local declarations and initialization. */@ ";
    Hashtbl.iteri traced_store ~f:(fun ~key:tn ~data:node ->
        if not (Tn.is_virtual_force tn 333 || Tn.is_materialized_force tn 336) then
          fprintf ppf "%s %s[%d]%s;@ "
            (B.typ_of_prec @@ Lazy.force tn.prec)
            (get_ident tn) (Tn.num_elems tn)
            (if node.zero_initialized then " = {0}" else ""));
    fprintf ppf "@,/* Main logic. */@ ";
    compile_main ~traced_store ppf llc;
    fprintf ppf "@;<0 -2>}@]@.";
    params
end
