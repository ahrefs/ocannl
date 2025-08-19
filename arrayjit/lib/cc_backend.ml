open Base
module Lazy = Utils.Lazy
open Ir

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_CC_BACKEND=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_CC_BACKEND"]

include Backend_impl.No_device_buffer_and_copying ()
open Backend_intf

let name = "cc"

let optimization_level () =
  Int.of_string @@ Utils.get_global_arg ~default:"3" ~arg_name:"cc_backend_optimization_level"

let compiler_command =
  let default =
    (* TODO: there's a direct way to get the compiler command from the OCaml compiler. *)
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

let%track7_sexp c_compile_and_load ~f_path =
  let base_name : string = Stdlib.Filename.chop_extension f_path in
  (* There can be only one library with a given name, the object gets cached. Moreover, [Dl.dlclose]
     is not required to unload the library, although ideally it should. *)
  let run_id = Int.to_string @@ get_global_run_id () in
  let log_fname = base_name ^ "_run_id_" ^ run_id ^ ".log" in
  let libname =
    let file_stem = Stdlib.Filename.chop_extension @@ Stdlib.Filename.basename f_path in
    if Utils.get_global_flag ~default:false ~arg_name:"output_dlls_in_build_directory" then
      (* Use only the path from f_path for the linked library libname *)
      base_name ^ "_run_id_" ^ run_id ^ if Sys.win32 then ".dll" else ".so"
    else
      (* Use temp_file without the run_id component *)
      Stdlib.Filename.temp_file file_stem (if Sys.win32 then ".dll" else ".so")
  in
  (try Stdlib.Sys.remove log_fname with _ -> ());
  (try Stdlib.Sys.remove libname with _ -> ());
  let kernel_link_flags =
    match Sys.os_type with
    | "Unix" ->
        if Stdlib.Sys.command "uname -s | grep -q Darwin" = 0 then
          "-bundle -undefined dynamic_lookup"
        else "-shared -fPIC"
    | "Win32" | "Cygwin" -> "-shared"
    | _ -> "-shared -fPIC"
  in
  (* On Windows, we need to link with the builtins library *)
  let builtins_lib = 
    if String.equal Sys.os_type "Win32" || String.equal Sys.os_type "Cygwin" then
      (* Try to find the builtins object file in the build directory *)
      let paths = [
        "_build/default/arrayjit/lib/builtins.o";
        "_build/default/arrayjit/lib/libir_stubs.a";
        "arrayjit/lib/builtins.o";
      ] in
      match List.find ~f:Stdlib.Sys.file_exists paths with
      | Some path -> " " ^ path
      | None -> ""
    else ""
  in
  let cmdline : string =
    Printf.sprintf "%s %s%s -O%d -o %s %s >> %s 2>&1" (compiler_command ()) f_path
      builtins_lib (optimization_level ()) libname kernel_link_flags log_fname
  in
  (* Debug: write the command to the log file *)
  let () = 
    let oc = Stdio.Out_channel.create ~append:false log_fname in
    Stdio.Out_channel.fprintf oc "Command: %s\n" cmdline;
    Stdio.Out_channel.fprintf oc "Builtins lib: '%s'\n" builtins_lib;
    Stdio.Out_channel.close oc
  in
  let rc : int = Stdlib.Sys.command cmdline in
  (* Note: it seems waiting for the file to exist is necessary here and below regardless of needing
     the logs. *)
  let start_time = Unix.gettimeofday () in
  let timeout = 1.0 in
  while rc = 0 && (not @@ (Stdlib.Sys.file_exists libname && Stdlib.Sys.file_exists log_fname)) do
    let elapsed = Unix.gettimeofday () -. start_time in
    if Float.(elapsed > timeout) then
      failwith "Cc_backend.c_compile_and_load: timeout waiting for compilation files to appear";
    Unix.sleepf 0.001
  done;
  if rc <> 0 then (
    let errors =
      "Cc_backend.c_compile_and_load: compilation failed with errors:\n"
      ^ Stdio.In_channel.read_all log_fname
    in
    Stdio.prerr_endline errors;
    invalid_arg errors);
  (* Expected to succeed on MacOS only. *)
  let sign_log_fname = base_name ^ "_run_id_" ^ run_id ^ "-sign.log" in
  let rc =
    Stdlib.Sys.command @@ Printf.sprintf "codesign -s - %s >> %s 2>&1" libname sign_log_fname
  in
  let start_time_sign = Unix.gettimeofday () in
  let timeout_sign = 1.0 in
  while
    rc = 0 && (not @@ (Stdlib.Sys.file_exists libname && Stdlib.Sys.file_exists sign_log_fname))
  do
    let elapsed_sign = Unix.gettimeofday () -. start_time_sign in
    if Float.(elapsed_sign > timeout_sign) then
      failwith "Cc_backend.c_compile_and_load: timeout waiting for codesign files to appear";
    Unix.sleepf 0.001
  done;
  let verify_codesign =
    Utils.get_global_flag ~default:false ~arg_name:"cc_backend_verify_codesign"
  in
  if verify_codesign && rc <> 0 then (
    let errors =
      "Cc_backend.c_compile_and_load: codesign failed with errors:\n"
      ^ Stdio.In_channel.read_all sign_log_fname
    in
    Stdio.prerr_endline errors;
    invalid_arg errors);
  (* Note: RTLD_DEEPBIND not available on MacOS. *)
  let result = { lib = Dl.dlopen ~filename:libname ~flags:[ RTLD_NOW ]; libname } in
  let%track7_sexp finalize (lib : library) : unit = Dl.dlclose ~handle:lib.lib in
  Stdlib.Gc.finalise finalize result;
  result

module CC_syntax_config (Procs : sig
  val procs : Low_level.optimized array
end) =
struct
  include C_syntax.Pure_C_config (struct
    type nonrec buffer_ptr = buffer_ptr

    let use_host_memory = use_host_memory
    let procs = Procs.procs

    let full_printf_support =
      not @@ Utils.get_global_flag ~default:false ~arg_name:"prefer_backend_uniformity"
  end)

  (* Override operation syntax to handle special precision types *)
  let ternop_syntax prec op v1 v2 v3 =
    match prec with
    | Ops.Bfloat16_prec _ ->
        (* For BFloat16, perform operations in float precision *)
        let open PPrint in
        let float_v1 = string "bfloat16_to_single(" ^^ v1 ^^ string ")" in
        let float_v2 = string "bfloat16_to_single(" ^^ v2 ^^ string ")" in
        let float_v3 = string "bfloat16_to_single(" ^^ v3 ^^ string ")" in
        let op_prefix, op_infix1, op_infix2, op_suffix = Ops.ternop_c_syntax Ops.single op in
        let float_result =
          group
            (string op_prefix ^^ float_v1 ^^ string op_infix1
            ^^ ifflat (space ^^ float_v2) (nest 2 (break 1 ^^ float_v2))
            ^^ string op_infix2
            ^^ ifflat (space ^^ float_v3) (nest 2 (break 1 ^^ float_v3))
            ^^ string op_suffix)
        in
        string "single_to_bfloat16(" ^^ float_result ^^ string ")"
    | Ops.Half_prec _ ->
        (* For Half, perform operations in float precision on non-native systems *)
        let open PPrint in
        let float_v1 = string "HALF_TO_FP(" ^^ v1 ^^ string ")" in
        let float_v2 = string "HALF_TO_FP(" ^^ v2 ^^ string ")" in
        let float_v3 = string "HALF_TO_FP(" ^^ v3 ^^ string ")" in
        let op_prefix, op_infix1, op_infix2, op_suffix = Ops.ternop_c_syntax Ops.single op in
        let float_result =
          group
            (string op_prefix ^^ float_v1 ^^ string op_infix1
            ^^ ifflat (space ^^ float_v2) (nest 2 (break 1 ^^ float_v2))
            ^^ string op_infix2
            ^^ ifflat (space ^^ float_v3) (nest 2 (break 1 ^^ float_v3))
            ^^ string op_suffix)
        in
        string "FP_TO_HALF(" ^^ float_result ^^ string ")"
    | Ops.Fp8_prec _ ->
        (* For FP8, perform operations in float precision *)
        let open PPrint in
        let float_v1 = string "fp8_to_single(" ^^ v1 ^^ string ")" in
        let float_v2 = string "fp8_to_single(" ^^ v2 ^^ string ")" in
        let float_v3 = string "fp8_to_single(" ^^ v3 ^^ string ")" in
        let op_prefix, op_infix1, op_infix2, op_suffix = Ops.ternop_c_syntax Ops.single op in
        let float_result =
          group
            (string op_prefix ^^ float_v1 ^^ string op_infix1
            ^^ ifflat (space ^^ float_v2) (nest 2 (break 1 ^^ float_v2))
            ^^ string op_infix2
            ^^ ifflat (space ^^ float_v3) (nest 2 (break 1 ^^ float_v3))
            ^^ string op_suffix)
        in
        string "single_to_fp8(" ^^ float_result ^^ string ")"
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
    | Ops.Threefry4x32 -> (
        match prec with
        | Ops.Uint4x32_prec _ ->
            let open PPrint in
            group (string "arrayjit_threefry4x32(" ^^ v1 ^^ string ", " ^^ v2 ^^ string ")")
        | _ -> invalid_arg "CC_syntax_config.binop_syntax: Threefry4x32 on non-uint4x32 precision")
    | _ -> (
        match prec with
        | Ops.Bfloat16_prec _ ->
            (* For BFloat16, perform all operations in float precision *)
            let open PPrint in
            let float_v1 = string "bfloat16_to_single(" ^^ v1 ^^ string ")" in
            let float_v2 = string "bfloat16_to_single(" ^^ v2 ^^ string ")" in
            let op_prefix, op_infix, op_suffix = Ops.binop_c_syntax Ops.single op in
            let float_result =
              group
                (string op_prefix ^^ float_v1 ^^ string op_infix
                ^^ ifflat (space ^^ float_v2) (nest 2 (break 1 ^^ float_v2))
                ^^ string op_suffix)
            in
            string "single_to_bfloat16(" ^^ float_result ^^ string ")"
        | Ops.Fp8_prec _ ->
            (* For FP8, perform all operations in float precision *)
            let open PPrint in
            let float_v1 = string "fp8_to_single(" ^^ v1 ^^ string ")" in
            let float_v2 = string "fp8_to_single(" ^^ v2 ^^ string ")" in
            let op_prefix, op_infix, op_suffix = Ops.binop_c_syntax Ops.single op in
            let float_result =
              group
                (string op_prefix ^^ float_v1 ^^ string op_infix
                ^^ ifflat (space ^^ float_v2) (nest 2 (break 1 ^^ float_v2))
                ^^ string op_suffix)
            in
            string "single_to_fp8(" ^^ float_result ^^ string ")"
        | Ops.Half_prec _ ->
            (* For Half, perform all operations in float precision on non-native systems *)
            let open PPrint in
            let float_v1 = string "HALF_TO_FP(" ^^ v1 ^^ string ")" in
            let float_v2 = string "HALF_TO_FP(" ^^ v2 ^^ string ")" in
            let op_prefix, op_infix, op_suffix = Ops.binop_c_syntax Ops.single op in
            let float_result =
              group
                (string op_prefix ^^ float_v1 ^^ string op_infix
                ^^ ifflat (space ^^ float_v2) (nest 2 (break 1 ^^ float_v2))
                ^^ string op_suffix)
            in
            string "FP_TO_HALF(" ^^ float_result ^^ string ")"
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
        let open PPrint in
        let float_v = string "bfloat16_to_single(" ^^ v ^^ string ")" in
        let op_prefix, op_suffix = Ops.unop_c_syntax Ops.single op in
        let float_result = group (string op_prefix ^^ float_v ^^ string op_suffix) in
        string "single_to_bfloat16(" ^^ float_result ^^ string ")"
    | Ops.Fp8_prec _ ->
        (* For FP8, perform operations in float precision *)
        let open PPrint in
        let float_v = string "fp8_to_single(" ^^ v ^^ string ")" in
        let op_prefix, op_suffix = Ops.unop_c_syntax Ops.single op in
        let float_result = group (string op_prefix ^^ float_v ^^ string op_suffix) in
        string "single_to_fp8(" ^^ float_result ^^ string ")"
    | Ops.Half_prec _ ->
        (* For Half, perform operations in float precision on non-native systems *)
        let open PPrint in
        let float_v = string "HALF_TO_FP(" ^^ v ^^ string ")" in
        let op_prefix, op_suffix = Ops.unop_c_syntax Ops.single op in
        let float_result = group (string op_prefix ^^ float_v ^^ string op_suffix) in
        string "FP_TO_HALF(" ^^ float_result ^^ string ")"
    | _ ->
        let op_prefix, op_suffix = Ops.unop_c_syntax prec op in
        let open PPrint in
        group (string op_prefix ^^ v ^^ string op_suffix)
end

let%diagn_sexp compile ~(name : string) bindings (lowered : Low_level.optimized) : procedure =
  let module Syntax = C_syntax.C_syntax (CC_syntax_config (struct
    let procs = [| lowered |]
  end)) in
  (* FIXME: do we really want all of them, or only the used ones? *)
  let idx_params = Indexing.bound_symbols bindings in
  let build_file = Utils.open_build_file ~base_name:name ~extension:".c" in
  let params, proc_doc = Syntax.compile_proc ~name idx_params lowered in
  let builtins_doc = PPrint.string Builtins_cc.source in
  let final_doc = PPrint.(builtins_doc ^^ proc_doc) in
  (* Use ribbon = 1.0 for usual code formatting, width 110 *)
  PPrint.ToChannel.pretty 1.0 110 build_file.oc final_doc;
  build_file.finalize ();

  (* let result = c_compile_and_load ~f_name:pp_file.f_name in *)
  let result_library = c_compile_and_load ~f_path:build_file.f_path in
  { result = result_library; params; bindings; name }

let%diagn_sexp compile_batch ~names bindings (lowereds : Low_level.optimized option array) :
    procedure option array =
  let module Syntax = C_syntax.C_syntax (CC_syntax_config (struct
    let procs = Array.filter_opt lowereds
  end)) in
  (* FIXME: do we really want all of them, or only the used ones? *)
  let idx_params = Indexing.bound_symbols bindings in
  let base_name =
    String.(
      strip ~drop:(equal_char '_')
      @@ common_prefix (Array.to_list @@ Array.concat_map ~f:Option.to_array names))
  in
  let build_file = Utils.open_build_file ~base_name ~extension:".c" in
  let params_and_docs =
    Array.map2_exn names lowereds ~f:(fun name_opt lowered_opt ->
        Option.map2 name_opt lowered_opt ~f:(fun name lowered ->
            Syntax.compile_proc ~name idx_params lowered))
  in
  let all_proc_docs = List.filter_map (Array.to_list params_and_docs) ~f:(Option.map ~f:snd) in
  let header_doc = PPrint.string Builtins_cc.source in
  let final_doc = PPrint.(header_doc ^^ separate hardline all_proc_docs) in
  PPrint.ToChannel.pretty 1.0 110 build_file.oc final_doc;
  build_file.finalize ();
  let result_library = c_compile_and_load ~f_path:build_file.f_path in
  (* Note: for simplicity, we share ctx_arrays across all contexts. *)
  Array.mapi params_and_docs ~f:(fun i opt_params_and_doc ->
      Option.bind opt_params_and_doc ~f:(fun (params, _doc) ->
          Option.map names.(i) ~f:(fun name -> { result = result_library; params; bindings; name })))

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
    if Utils.debug_log_from_routines () then
      Utils.log_debug_routine_file ~log_file_name ~stream_name:runner_label
  in
  ( Indexing.lowered_bindings code.bindings run_variadic,
    Task.Task
      {
        (* In particular, keep code alive so it doesn't get unloaded. *)
        context_lifetime = (ctx_arrays, code);
        description = "executes " ^ code.name ^ " on " ^ runner_label;
        work;
      } )
