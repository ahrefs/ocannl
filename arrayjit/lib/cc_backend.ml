open Base
module Lazy = Utils.Lazy
open Ir

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

include Backend_impl.No_device_buffer_and_copying ()
open Backend_intf

let name = "cc"

let optimization_level () =
  Int.of_string @@ Utils.get_global_arg ~default:"3" ~arg_name:"cc_backend_optimization_level"

let compiler_command =
  let default =
    lazy
      (let ic = Unix.open_process_in "ocamlc -config" in
       let rec find_compiler () =
         match In_channel.input_line ic with
         | None -> "cc" (* Default fallback *)
         | Some line ->
             if String.is_prefix line ~prefix:"c_compiler: " then
               String.drop_prefix line 12 (* Length of "c_compiler: " *)
             else find_compiler ()
       in
       let compiler = find_compiler () in
       ignore (Unix.close_process_in ic);
       compiler)
  in
  fun () ->
    Utils.get_global_arg ~default:(Lazy.force default) ~arg_name:"cc_backend_compiler_command"

module Tn = Tnode

type library = { lib : (Dl.library[@sexp.opaque]); libname : string } [@@deriving sexp_of]

type procedure = {
  bindings : Indexing.unit_bindings;
  name : string;
  result : library;
  params : (string * param_source) list;
}
[@@deriving sexp_of]

let get_global_run_id =
  let next_id = ref 0 in
  fun () ->
    Int.incr next_id;
    if !next_id < 0 then next_id := 0;
    !next_id

let%track7_sexp c_compile_and_load ~f_name =
  let base_name : string = Stdlib.Filename.chop_extension f_name in
  (* There can be only one library with a given name, the object gets cached. Moreover, [Dl.dlclose]
     is not required to unload the library, although ideally it should. *)
  let run_id = Int.to_string @@ get_global_run_id () in
  let log_fname = base_name ^ "_run_id_" ^ run_id ^ ".log" in
  let libname = base_name ^ "_run_id_" ^ run_id ^ if Sys.win32 then ".dll" else ".so" in
  (try Stdlib.Sys.remove log_fname with _ -> ());
  (try Stdlib.Sys.remove libname with _ -> ());
  let cmdline : string =
    Printf.sprintf "%s %s -O%d -o %s -shared -fPIC >> %s 2>&1" (compiler_command ()) f_name
      (optimization_level ()) libname log_fname
  in
  let rc : int = Stdlib.Sys.command cmdline in
  while rc = 0 && (not @@ (Stdlib.Sys.file_exists libname && Stdlib.Sys.file_exists log_fname)) do
    Unix.sleepf 0.001
  done;
  if rc <> 0 then (
    let errors =
      "Cc_backend.c_compile_and_load: compilation failed with errors:\n"
      ^ Stdio.In_channel.read_all log_fname
    in
    Stdio.prerr_endline errors;
    invalid_arg errors);
  (* Note: RTLD_DEEPBIND not available on MacOS. *)
  let result = { lib = Dl.dlopen ~filename:libname ~flags:[ RTLD_NOW ]; libname } in
  let%track7_sexp finalize (lib : library) : unit = Dl.dlclose ~handle:lib.lib in
  Stdlib.Gc.finalise finalize result;
  result

module C_syntax_config (Input : sig
  val procs : Low_level.optimized array
end) =
struct
  let procs = Input.procs

  type nonrec buffer_ptr = buffer_ptr

  let use_host_memory = use_host_memory
  let logs_to_stdout = false
  let main_kernel_prefix = ""
  let kernel_prep_line = ""
  let includes = [ "<stdio.h>"; "<stdlib.h>"; "<string.h>"; "<math.h>" ]
  let typ_of_prec = Ops.c_typ_of_prec
  let ternop_syntax = Ops.ternop_c_syntax
  let binop_syntax = Ops.binop_c_syntax
  let unop_syntax = Ops.unop_c_syntax
  let convert_precision = Ops.c_convert_precision
end

let%diagn_sexp compile ~(name : string) bindings (lowered : Low_level.optimized) =
  let module Syntax = C_syntax.C_syntax (C_syntax_config (struct
    let procs = [| lowered |]
  end)) in
  (* FIXME: do we really want all of them, or only the used ones? *)
  let idx_params = Indexing.bound_symbols bindings in
  let pp_file = Utils.pp_file ~base_name:name ~extension:".c" in
  Syntax.print_includes pp_file.ppf;
  let params = Syntax.compile_proc ~name pp_file.ppf idx_params lowered in
  pp_file.finalize ();
  let result = c_compile_and_load ~f_name:pp_file.f_name in
  { result; params; bindings; name }

let%diagn_sexp compile_batch ~names bindings (lowereds : Low_level.optimized option array) =
  let module Syntax = C_syntax.C_syntax (C_syntax_config (struct
    let procs = Array.filter_opt lowereds
  end)) in
  (* FIXME: do we really want all of them, or only the used ones? *)
  let idx_params = Indexing.bound_symbols bindings in
  let base_name =
    String.(
      strip ~drop:(equal_char '_')
      @@ common_prefix (Array.to_list @@ Array.concat_map ~f:Option.to_array names))
  in
  let pp_file = Utils.pp_file ~base_name ~extension:".c" in
  Syntax.print_includes pp_file.ppf;
  let params =
    Array.mapi lowereds ~f:(fun i lowered ->
        Option.map2 names.(i) lowered ~f:(fun name lowered ->
            Syntax.compile_proc ~name pp_file.ppf idx_params lowered))
  in
  pp_file.finalize ();
  let result = c_compile_and_load ~f_name:pp_file.f_name in
  (* Note: for simplicity, we share ctx_arrays across all contexts. *)
  Array.mapi params ~f:(fun i params ->
      Option.map names.(i) ~f:(fun name ->
          { result; params = Option.value_exn ~here:[%here] params; bindings; name }))

let%track3_sexp link_compiled ~merge_buffer ~runner_label ctx_arrays (code : procedure) =
  let name : string = code.name in
  let log_file_name = Utils.diagn_log_file [%string "debug-%{runner_label}-%{code.name}.log"] in
  let run_variadic =
    [%log_level
      0;
      let rec link :
          'a 'b 'idcs.
          'idcs Indexing.bindings ->
          param_source list ->
          ('a -> 'b) Ctypes.fn ->
          ('a -> 'b, 'idcs, 'p1, 'p2) Indexing.variadic =
       fun (type a b idcs) (binds : idcs Indexing.bindings) params (cs : (a -> b) Ctypes.fn) ->
        match (binds, params) with
        | Empty, [] -> Indexing.Result (Foreign.foreign ~from:code.result.lib name cs)
        | Bind _, [] -> invalid_arg "Cc_backend.link: too few static index params"
        | Bind (_, bs), Static_idx _ :: ps -> Param_idx (ref 0, link bs ps Ctypes.(int @-> cs))
        | Empty, Static_idx _ :: _ -> invalid_arg "Cc_backend.link: too many static index params"
        | bs, Log_file_name :: ps ->
            Param_1 (ref (Some log_file_name), link bs ps Ctypes.(string @-> cs))
        | bs, Merge_buffer :: ps ->
            let get_ptr buf = buf.ptr in
            Param_2f (get_ptr, merge_buffer, link bs ps Ctypes.(ptr void @-> cs))
        | bs, Param_ptr tn :: ps ->
            let c_ptr =
              match Map.find ctx_arrays tn with
              | None ->
                  Ndarray.get_voidptr_not_managed
                  @@ Option.value_exn ~here:[%here]
                       ~message:
                         [%string
                           "Cc_backend.link_compiled: node %{Tn.debug_name tn} missing from \
                            context: %{Tn.debug_memory_mode tn.Tn.memory_mode}"]
                  @@ Lazy.force tn.array
              | Some arr -> arr
            in
            Param_2 (ref (Some c_ptr), link bs ps Ctypes.(ptr void @-> cs))
      in
      (* Reverse the input order because [Indexing.apply] will reverse it again. Important:
         [code.bindings] are traversed in the wrong order but that's OK because [link] only uses
         them to check the number of indices. *)
      let params = List.rev_map code.params ~f:(fun (_, p) -> p) in
      link code.bindings params Ctypes.(void @-> returning void)]
  in
  let%diagn_sexp work () : unit =
    [%log_result name];
    (* Stdio.printf "launching %s\n" name; *)
    Indexing.apply run_variadic ();
    if Utils.debug_log_from_routines () then (
      Utils.log_trace_tree (Stdio.In_channel.read_lines log_file_name);
      Stdlib.Sys.remove log_file_name)
  in
  ( Indexing.lowered_bindings code.bindings run_variadic,
    Task.Task
      {
        (* In particular, keep code alive so it doesn't get unloaded. *)
        context_lifetime = (ctx_arrays, code);
        description = "executes " ^ code.name ^ " on " ^ runner_label;
        work;
      } )
