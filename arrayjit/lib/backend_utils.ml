open Base
module Lazy = Utils.Lazy
module Debug_runtime = Utils.Debug_runtime

[%%global_debug_log_level Nothing]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

module Types = struct
  type 'context routine = {
    context : 'context;
    schedule : Tnode.task;
    bindings : Indexing.lowered_bindings;
    name : string;
  }
  [@@deriving sexp_of]

  type config = Physical_devices_only | For_parallel_copying | Most_parallel_devices
  [@@deriving equal, sexp, variants]

  type merge_buffer_use = No | Streaming | Copy [@@deriving equal, sexp]

  type param_source =
    | Log_file_name
    | Merge_buffer
    | Param_ptr of Tnode.t
    | Static_idx of Indexing.static_symbol
  [@@deriving sexp_of]
end

module Tn = Tnode

module C_syntax (B : sig
  val for_lowereds : Low_level.optimized array

  type ctx_array

  val opt_ctx_arrays : ctx_array Map.M(Tnode).t option
  val hardcoded_context_ptr : (ctx_array -> string) option
  val is_in_context : Low_level.traced_array -> bool
  val host_ptrs_for_readonly : bool
  val logs_to_stdout : bool
  val main_kernel_prefix : string
  val kernel_prep_line : string
end) =
struct
  open Types

  let get_ident =
    Low_level.get_ident_within_code ~no_dots:true @@ Array.map B.for_lowereds ~f:(fun l -> l.llc)

  let pp_zero_out ppf tn =
    Stdlib.Format.fprintf ppf "@[<2>memset(%s, 0, %d);@]@ " (get_ident tn) @@ Tn.size_in_bytes tn

  (* let pp_semi ppf () = Stdlib.Format.fprintf ppf ";@ " *)
  let pp_comma ppf () = Stdlib.Format.fprintf ppf ",@ "

  (* let pp_symbol ppf sym = Stdlib.Format.fprintf ppf "%s" @@ Indexing.symbol_ident sym *)
  let pp_index ppf sym = Stdlib.Format.fprintf ppf "%s" @@ Indexing.symbol_ident sym

  let pp_index_axis ppf = function
    | Indexing.Iterator it -> pp_index ppf it
    | Fixed_idx i -> Stdlib.Format.fprintf ppf "%d" i

  let pp_array_offset ppf (idcs, dims) =
    let open Stdlib.Format in
    assert (not @@ Array.is_empty idcs);
    for _ = 0 to Array.length idcs - 3 do
      fprintf ppf "@[<1>("
    done;
    for i = 0 to Array.length idcs - 1 do
      let dim = dims.(i) in
      if i = 0 then fprintf ppf "%a" pp_index_axis idcs.(i)
      else if i = Array.length idcs - 1 then fprintf ppf " * %d + %a" dim pp_index_axis idcs.(i)
      else fprintf ppf " * %d +@ %a@;<0 -1>)@]" dim pp_index_axis idcs.(i)
    done

  let array_offset_to_string (idcs, dims) =
    let b = Buffer.create 32 in
    let ppf = Stdlib.Format.formatter_of_buffer b in
    pp_array_offset ppf (idcs, dims);
    Stdlib.Format.pp_print_flush ppf ();
    Buffer.contents b

  (* let compute_array_offset ~idcs ~dims = Array.fold2_exn idcs dims ~init:0 ~f:(fun offset idx dim
     -> idx + (offset * dim)) *)
  let%track_sexp compile_globals ppf =
    let open Stdlib.Format in
    let is_global = Hash_set.create (module Tn) in
    fprintf ppf {|@[<v 0>#include <stdio.h>@,#include <stdlib.h>@,/* Global declarations. */@,|};
    Array.iter B.for_lowereds ~f:(fun l ->
        Hashtbl.iter l.Low_level.traced_store ~f:(fun (node : Low_level.traced_array) ->
            if not @@ Hash_set.mem is_global node.tn then
              let in_ctx : bool = B.is_in_context node in
              let ctx_ptr = B.hardcoded_context_ptr in
              let mem : (Tn.memory_mode * int) option = node.tn.memory_mode in
              match
                (in_ctx, ctx_ptr, B.opt_ctx_arrays, B.host_ptrs_for_readonly, mem, node.read_only)
              with
              | true, Some get_ptr, Some ctx_arrays, _, _, _ ->
                  let ident = get_ident node.tn in
                  let ctx_array =
                    Option.value_exn ~here:[%here] ~message:ident @@ Map.find ctx_arrays node.tn
                  in
                  fprintf ppf "#define %s (%s)@," ident @@ get_ptr ctx_array;
                  Hash_set.add is_global node.tn
              | false, _, _, true, Some (Hosted _, _), true ->
                  (* In-context nodes to read directly from host would be error prone. *)
                  let nd = Option.value_exn ~here:[%here] @@ Lazy.force node.tn.array in
                  fprintf ppf "#define %s (%s)@," (get_ident node.tn) (Ndarray.c_ptr_to_string nd);
                  Hash_set.add is_global node.tn
              | _ -> ()));
    fprintf ppf "@,@]";
    is_global

  let compile_main ~traced_store ppf llc : unit =
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
          fprintf ppf "@[<2>for (int@ %a = %d;@ %a <= %d;@ ++%a) {@ %a@;<1 -2>}@]@," pp_index i
            from_ pp_index i to_ pp_index i pp_ll body
      | Zero_out tn ->
          let traced = Low_level.(get_node traced_store tn) in
          if Hash_set.mem visited tn then pp_zero_out ppf tn else assert traced.zero_initialized
          (* The initialization will be emitted by get_array. *)
      | Set { tn; idcs; llv; debug } ->
          Hash_set.add visited tn;
          let ident = get_ident tn in
          let dims = Lazy.force tn.dims in
          let loop_f = pp_float tn.prec in
          let loop_debug_f = debug_float tn.prec in
          let num_closing_braces = pp_top_locals ppf llv in
          let num_typ = Ops.cuda_typ_of_prec tn.prec in
          if Utils.settings.debug_log_from_routines then (
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
                {|@[<7>printf(@[<h>"%s%%d: %s[%%u] = %%f = %s\n",@]@ log_id,@ %a,@ new_set_v%a);@]@ |}
                !Utils.captured_log_prefix ident v_code pp_array_offset offset pp_args v_idcs)
            else (
              fprintf ppf {|@[<7>fprintf(log_file,@ @[<h>"# %s\n"@]);@]@ |}
                (String.substr_replace_all debug ~pattern:"\n" ~with_:"$");
              fprintf ppf
                {|@[<7>fprintf(log_file,@ @[<h>"%s[%%u] = %%f = %s\n",@]@ %a,@ new_set_v%a);@]@ |}
                ident v_code pp_array_offset offset pp_args v_idcs);
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
          if Utils.settings.debug_log_from_routines then
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
          fprintf ppf "@[<2>v%d =@ %a;@]" scope_id (pp_float prec) value;
          for _ = 1 to num_closing_braces do
            fprintf ppf "@]@ }@,"
          done
    and pp_top_locals ppf (vcomp : Low_level.float_t) : int =
      match vcomp with
      | Local_scope { id = { scope_id = i; tn = { prec; _ } }; body; orig_indices = _ } ->
          let num_typ = Ops.cuda_typ_of_prec prec in
          (* Arrays are initialized to 0 by default. However, there is typically an explicit
             initialization for virtual nodes. *)
          fprintf ppf "@[<2>{@ %s v%d = 0;@ " num_typ i;
          pp_ll ppf body;
          pp_print_space ppf ();
          1
      | Get_local _ | Get_global _ | Get _ | Constant _ | Embed_index _ -> 0
      | Binop (Arg1, v1, _v2) -> pp_top_locals ppf v1
      | Binop (Arg2, _v1, v2) -> pp_top_locals ppf v2
      | Binop (_, v1, v2) -> pp_top_locals ppf v1 + pp_top_locals ppf v2
      | Unop (_, v) -> pp_top_locals ppf v
    and pp_float prec ppf value =
      let num_typ = Ops.cuda_typ_of_prec prec in
      let loop = pp_float prec in
      match value with
      | Local_scope { id; _ } ->
          (* Embedding of Local_scope is done by pp_top_locals. *)
          loop ppf @@ Get_local id
      | Get_local id ->
          let get_typ = Ops.cuda_typ_of_prec id.tn.prec in
          if not @@ String.equal num_typ get_typ then fprintf ppf "(%s)" num_typ;
          fprintf ppf "v%d" id.scope_id
      | Get_global (Ops.Merge_buffer { source_node_id }, Some idcs) ->
          let tn = Option.value_exn ~here:[%here] @@ Tn.find ~id:source_node_id in
          fprintf ppf "@[<2>((%s*)merge_buffer)[%a@;<0 -2>]@]" (Ops.cuda_typ_of_prec prec)
            pp_array_offset
            (idcs, Lazy.force tn.dims)
      | Get_global _ -> failwith "C_syntax: Get_global / FFI NOT IMPLEMENTED YET"
      | Get (tn, idcs) ->
          Hash_set.add visited tn;
          let ident = get_ident tn in
          fprintf ppf "@[<2>%s[%a@;<0 -2>]@]" ident pp_array_offset (idcs, Lazy.force tn.dims)
      | Constant c -> fprintf ppf "(%f)" c
      | Embed_index idx ->
          if not @@ List.exists ~f:(String.equal num_typ) [ "int"; "size_t" ] then
            fprintf ppf "(%s)" num_typ;
          pp_index_axis ppf idx
      | Binop (Arg1, v1, _v2) -> loop ppf v1
      | Binop (Arg2, _v1, v2) -> loop ppf v2
      | Binop (op, v1, v2) ->
          let prefix, infix, postfix = Ops.binop_C_syntax prec op in
          fprintf ppf "@[<1>%s%a%s@ %a@]%s" prefix loop v1 infix loop v2 postfix
      | Unop (Identity, v) -> loop ppf v
      | Unop (Relu, v) ->
          (* FIXME: don't recompute v *)
          fprintf ppf "@[<1>(%a > 0.0 ?@ %a : 0.0@;<0 -1>)@]" loop v loop v
    and debug_float prec (value : Low_level.float_t) : string * 'a list =
      let num_typ = Ops.cuda_typ_of_prec prec in
      let loop = debug_float prec in
      match value with
      | Local_scope { id; _ } ->
          (* Not printing the inlined definition: (1) code complexity; (2) don't overload the debug
             logs. *)
          loop @@ Get_local id
      | Get_local id ->
          let get_typ = Ops.cuda_typ_of_prec id.tn.prec in
          let v =
            (if not @@ String.equal num_typ get_typ then "(" ^ num_typ ^ ")" else "")
            ^ "v" ^ Int.to_string id.scope_id
          in
          (v ^ "{=%f}", [ `Value v ])
      | Get_global (Ops.Merge_buffer { source_node_id }, Some idcs) ->
          let tn = Option.value_exn ~here:[%here] @@ Tn.find ~id:source_node_id in
          let dims = Lazy.force tn.dims in
          let v = sprintf "@[<2>merge_buffer[%s@;<0 -2>]@]" (array_offset_to_string (idcs, dims)) in
          ("merge_buffer[%u]{=%f}", [ `Accessor (idcs, dims); `Value v ])
      | Get_global _ -> failwith "Exec_as_cuda: Get_global / FFI NOT IMPLEMENTED YET"
      | Get (tn, idcs) ->
          let dims = Lazy.force tn.dims in
          let ident = get_ident tn in
          let v = sprintf "@[<2>%s[%s@;<0 -2>]@]" ident (array_offset_to_string (idcs, dims)) in
          (ident ^ "[%u]{=%f}", [ `Accessor (idcs, dims); `Value v ])
      | Constant c -> (Float.to_string c, [])
      | Embed_index (Fixed_idx i) -> (Int.to_string i, [])
      | Embed_index (Iterator s) -> (Indexing.symbol_ident s, [])
      | Binop (Arg1, v1, _v2) -> loop v1
      | Binop (Arg2, _v1, v2) -> loop v2
      | Binop (op, v1, v2) ->
          let prefix, infix, postfix = Ops.binop_C_syntax prec op in
          let v1, idcs1 = loop v1 in
          let v2, idcs2 = loop v2 in
          (String.concat [ prefix; v1; infix; " "; v2; postfix ], idcs1 @ idcs2)
      | Unop (Identity, v) -> loop v
      | Unop (Relu, v) ->
          let v, idcs = loop v in
          (String.concat [ "("; v; " > 0.0 ? "; v; " : 0.0)" ], idcs @ idcs)
    in
    pp_ll ppf llc

  let%track_sexp compile_proc ~name ppf idx_params ~is_global
      Low_level.{ traced_store; llc; merge_node } =
    let open Stdlib.Format in
    let params : (string * param_source) list =
      Hashtbl.fold traced_store ~init:[] ~f:(fun ~key:tn ~data:node params ->
          if Utils.settings.with_debug_level > 0 then
            [%log "array-used:", (tn : Tn.t), get_ident tn];
          if B.is_in_context node && not (Hash_set.mem is_global tn) then
            (Ops.cuda_typ_of_prec tn.Tn.prec ^ " *" ^ get_ident tn, Param_ptr tn) :: params
          else params)
    in
    let idx_params =
      List.map idx_params ~f:(fun s ->
          ("int " ^ Indexing.symbol_ident s.Indexing.static_symbol, Static_idx s))
    in
    let log_file =
      if Utils.settings.debug_log_from_routines then
        [
          ((if B.logs_to_stdout then "int log_id" else "const char* log_file_name"), Log_file_name);
        ]
      else []
    in
    let merge_param =
      Option.(
        to_list
        @@ map merge_node ~f:(fun tn ->
               ("const " ^ Ops.cuda_typ_of_prec tn.prec ^ " *merge_buffer", Merge_buffer)))
    in
    let params = log_file @ merge_param @ idx_params @ params in
    fprintf ppf "@[<v 2>@[<hv 4>%s%svoid %s(@,@[<hov 0>%a@]@;<0 -4>)@] {@ " B.main_kernel_prefix
      (if String.is_empty B.main_kernel_prefix then "" else " ")
      name
      (pp_print_list ~pp_sep:pp_comma pp_print_string)
    @@ List.map ~f:fst params;
    if not (String.is_empty B.kernel_prep_line) then fprintf ppf "%s@ " B.kernel_prep_line;
    (* FIXME: we should also close the file. *)
    if (not (List.is_empty log_file)) && not B.logs_to_stdout then
      fprintf ppf {|FILE* log_file = fopen(log_file_name, "w");@ |};
    if Utils.settings.debug_log_from_routines && Utils.settings.with_debug_level > 1 then (
      fprintf ppf "/* Debug initial parameter state. */@ ";
      List.iter
        ~f:(function
          | p_name, Merge_buffer ->
              if B.logs_to_stdout then
                fprintf ppf {|@[<7>printf(@[<h>"%s%%d: %s = %%p\n",@] log_id, (void*)merge_buffer);@]@ |}
                  !Utils.captured_log_prefix p_name
              else
                fprintf ppf {|@[<7>fprintf(log_file,@ @[<h>"%s = %%p\n",@] (void*)merge_buffer);@]@ |}
                  p_name
          | _, Log_file_name -> ()
          | p_name, Param_ptr tn ->
              if B.logs_to_stdout then
                fprintf ppf {|@[<7>printf(@[<h>"%s%%d: %s = %%p\n",@] log_id, (void*)%s);@]@ |}
                  !Utils.captured_log_prefix p_name
                @@ get_ident tn
              else
                fprintf ppf {|@[<7>fprintf(log_file,@ @[<h>"%s = %%p\n",@] (void*)%s);@]@ |} p_name
                @@ get_ident tn
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
        if not (Tn.is_virtual_force tn 333 || B.is_in_context node || Hash_set.mem is_global tn)
        then
          fprintf ppf "%s %s[%d]%s;@ " (Ops.cuda_typ_of_prec tn.prec) (get_ident tn)
            (Tn.num_elems tn)
            (if node.zero_initialized then " = {0}" else "")
        else if (not (Tn.is_virtual_force tn 333)) && node.zero_initialized then pp_zero_out ppf tn);
    fprintf ppf "@,/* Main logic. */@ ";
    compile_main ~traced_store ppf llc;
    fprintf ppf "@;<0 -2>}@]@.";
    params
end

let check_merge_buffer ~merge_buffer ~code_node =
  let device_node = Option.map !merge_buffer ~f:snd in
  let name = function Some tn -> Tn.debug_name tn | None -> "none" in
  match (device_node, code_node) with
  | _, None -> ()
  | Some actual, Some expected when Tn.equal actual expected -> ()
  | _ ->
      raise
      @@ Utils.User_error
           ("Merge buffer mismatch, on device: " ^ name device_node ^ ", expected by code: "
          ^ name code_node)
