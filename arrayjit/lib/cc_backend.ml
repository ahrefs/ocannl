open Base
module Lazy = Utils.Lazy
open Ir

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

include Backend_impl.No_device_buffer_and_copying ()
open Backend_intf

let name = "cc"

(* Header declarations for arrayjit builtins *)
let builtins_header = {|
/* ArrayJIT builtins declarations */
#include <stdint.h>

typedef struct {
    uint32_t v[4];
} uint4x32_t;

/* Threefry4x32 random number generator */
extern uint4x32_t arrayjit_threefry4x32(uint4x32_t key, uint4x32_t counter);

/* Vector types for efficient extraction of multiple values */
typedef struct { float v[4]; } float4_t;
typedef struct { double v[2]; } double2_t;
typedef struct { int32_t v[4]; } int32x4_t;
typedef struct { int64_t v[2]; } int64x2_t;
typedef struct { int8_t v[16]; } int8x16_t;
typedef struct { uint16_t v[8]; } uint16x8_t;
typedef struct { uint8_t v[16]; } uint8x16_t;
typedef struct { _Float16 v[8]; } half8_t;

/* Conversion functions from uint4x32 to various precisions uniformly */
extern float4_t uint4x32_to_single_uniform_vec(uint4x32_t x);
extern double2_t uint4x32_to_double_uniform_vec(uint4x32_t x);
extern int32x4_t uint4x32_to_int32_uniform_vec(uint4x32_t x);
extern int64x2_t uint4x32_to_int64_uniform_vec(uint4x32_t x);
extern int8x16_t uint4x32_to_byte_uniform_vec(uint4x32_t x);
extern uint16x8_t uint4x32_to_uint16_uniform_vec(uint4x32_t x);
extern uint16x8_t uint4x32_to_bfloat16_uniform_vec(uint4x32_t x);
extern half8_t uint4x32_to_half_uniform_vec(uint4x32_t x);
extern uint8x16_t uint4x32_to_fp8_uniform_vec(uint4x32_t x);

/* Conversion functions from various precisions to uint4x32_t */
extern uint4x32_t single_to_uint4x32(float x);
extern uint4x32_t double_to_uint4x32(double x);
extern uint4x32_t int32_to_uint4x32(int32_t x);
extern uint4x32_t int64_to_uint4x32(int64_t x);
extern uint4x32_t uint32_to_uint4x32(uint32_t x);
extern uint4x32_t uint64_to_uint4x32(uint64_t x);
extern uint4x32_t byte_to_uint4x32(unsigned char x);
extern uint4x32_t uint16_to_uint4x32(uint16_t x);
extern uint4x32_t bfloat16_to_uint4x32(uint16_t x);
extern uint4x32_t half_to_uint4x32(uint16_t x);
extern uint4x32_t fp8_to_uint4x32(uint8_t x);

|}

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

let%track7_sexp c_compile_and_load ~f_name =
  let base_name : string = Stdlib.Filename.chop_extension f_name in
  (* There can be only one library with a given name, the object gets cached. Moreover, [Dl.dlclose]
     is not required to unload the library, although ideally it should. *)
  let run_id = Int.to_string @@ get_global_run_id () in
  let log_fname = base_name ^ "_run_id_" ^ run_id ^ ".log" in
  let libname = base_name ^ "_run_id_" ^ run_id ^ if Sys.win32 then ".dll" else ".so" in
  (try Stdlib.Sys.remove log_fname with _ -> ());
  (try Stdlib.Sys.remove libname with _ -> ());
  let kernel_link_flags = 
    match Sys.os_type with
    | "Unix" -> 
        if Stdlib.Sys.command "uname -s | grep -q Darwin" = 0 then
          "-bundle -undefined dynamic_lookup"
        else
          "-shared -fPIC"
    | "Win32" | "Cygwin" -> "-shared"
    | _ -> "-shared -fPIC" in
  let cmdline : string =
    Printf.sprintf "%s %s -O%d -o %s %s >> %s 2>&1" (compiler_command ()) f_name
      (optimization_level ()) libname kernel_link_flags log_fname
  in
  let rc : int = Stdlib.Sys.command cmdline in
  (* Note: it seems waiting for the file to exist is necessary here and below regardless of needing
     the logs. *)
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
  (* Expected to succeed on MacOS only. *)
  let sign_log_fname = base_name ^ "_run_id_" ^ run_id ^ "-sign.log" in
  let rc =
    Stdlib.Sys.command @@ Printf.sprintf "codesign -s - %s >> %s 2>&1" libname sign_log_fname
  in
  while
    rc = 0 && (not @@ (Stdlib.Sys.file_exists libname && Stdlib.Sys.file_exists sign_log_fname))
  do
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

  (* Override to add our custom type and conversion support *)
  let typ_of_prec = typ_of_prec
  let vec_typ_of_prec = vec_typ_of_prec
  let extra_declarations = extra_declarations (* Our bfloat16/fp8 conversion functions *)
  let convert_precision = convert_precision
end

let%diagn_sexp compile ~(name : string) bindings (lowered : Low_level.optimized) : procedure =
  let module Syntax = C_syntax.C_syntax (CC_syntax_config (struct
    let procs = [| lowered |]
  end)) in
  (* FIXME: do we really want all of them, or only the used ones? *)
  let idx_params = Indexing.bound_symbols bindings in
  let build_file = Utils.open_build_file ~base_name:name ~extension:".c" in
  let declarations_doc = Syntax.print_declarations () in
  let params, proc_doc = Syntax.compile_proc ~name idx_params lowered in
  let header_doc = PPrint.string builtins_header in
  let final_doc = PPrint.(header_doc ^^ declarations_doc ^^ proc_doc) in
  (* Use ribbon = 1.0 for usual code formatting, width 110 *)
  PPrint.ToChannel.pretty 1.0 110 build_file.oc final_doc;
  build_file.finalize ();

  (* let result = c_compile_and_load ~f_name:pp_file.f_name in *)
  let result_library = c_compile_and_load ~f_name:build_file.f_name in
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
  let declarations_doc = Syntax.print_declarations () in
  let params_and_docs =
    Array.map2_exn names lowereds ~f:(fun name_opt lowered_opt ->
        Option.map2 name_opt lowered_opt ~f:(fun name lowered ->
            Syntax.compile_proc ~name idx_params lowered))
  in
  let all_proc_docs = List.filter_map (Array.to_list params_and_docs) ~f:(Option.map ~f:snd) in
  let header_doc = PPrint.string builtins_header in
  let final_doc = PPrint.(header_doc ^^ declarations_doc ^^ separate hardline all_proc_docs) in
  PPrint.ToChannel.pretty 1.0 110 build_file.oc final_doc;
  build_file.finalize ();
  let result_library = c_compile_and_load ~f_name:build_file.f_name in
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
