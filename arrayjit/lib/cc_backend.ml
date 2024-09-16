open Base
module Lazy = Utils.Lazy
module Debug_runtime = Utils.Debug_runtime

let _get_local_debug_runtime = Utils._get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

open Backend_utils.Types

let name = "cc"

let optimization_level () =
  Int.of_string @@ Utils.get_global_arg ~default:"3" ~arg_name:"cc_backend_optimization_level"

let compiler_command () = Utils.get_global_arg ~default:"cc" ~arg_name:"cc_backend_compiler_command"

module Tn = Tnode

type ctx_array = Ndarray.t [@@deriving sexp_of]
type ctx_arrays = ctx_array Map.M(Tn).t [@@deriving sexp_of]
type context = { label : string; arrays : ctx_arrays } [@@deriving sexp_of]

let ctx_arrays context = context.arrays

type buffer_ptr = ctx_array [@@deriving sexp_of]

let buffer_ptr ctx_array = ctx_array

let alloc_buffer ?old_buffer ~size_in_bytes () =
  (* FIXME: NOT IMPLEMENTED YET but should not be needed for the streaming case. *)
  match old_buffer with
  | Some (old_ptr, old_size) when size_in_bytes <= old_size -> old_ptr
  | Some (_old_ptr, _old_size) -> assert false
  | None -> assert false

let to_buffer tn ~dst ~src =
  let src = Map.find_exn src.arrays tn in
  Ndarray.map2 { f2 = Ndarray.A.blit } src dst

let host_to_buffer src ~dst = Ndarray.map2 { f2 = Ndarray.A.blit } src dst
let buffer_to_host dst ~src = Ndarray.map2 { f2 = Ndarray.A.blit } src dst
let unsafe_cleanup () = Stdlib.Gc.compact ()

let is_initialized, initialize =
  let initialized = ref false in
  ((fun () -> !initialized), fun () -> initialized := true)

let finalize _ctx = ()

let init ~label =
  let result = { label; arrays = Map.empty (module Tn) } in
  Stdlib.Gc.finalise finalize result;
  result

type library = { lib : (Dl.library[@sexp.opaque]); libname : string } [@@deriving sexp_of]

type procedure = {
  lowered : Low_level.optimized;
  bindings : Indexing.unit_bindings;
  name : string;
  result : library;
  params : (string * param_source) list;
  opt_ctx_arrays : Ndarray.t Map.M(Tn).t option;
}
[@@deriving sexp_of]

let expected_merge_node proc = proc.lowered.merge_node

let is_in_context node =
  Tnode.default_to_most_local node.Low_level.tn 33;
  match node.tn.memory_mode with
  | Some (Hosted (Constant | Volatile), _) -> false
  | Some ((Virtual | Local), _) -> false
  | _ -> true

let header_sep =
  let open Re in
  compile (seq [ str " "; opt any; str "="; str " " ])

let get_global_run_id =
  let next_id = ref 0 in
  fun () ->
    Int.incr next_id;
    if !next_id < 0 then next_id := 0;
    !next_id

let c_compile_and_load ~f_name =
  let base_name = Filename_base.chop_extension f_name in
  (* There can be only one library with a given name, the object gets cached. Moreover, [Dl.dlclose]
     is not required to unload the library, although ideally it should. *)
  let run_id = Int.to_string @@ get_global_run_id () in
  let log_fname = base_name ^ "_run_id_" ^ run_id ^ ".log" in
  let libname = base_name ^ "_run_id_" ^ run_id ^ ".so" in
  (try Stdlib.Sys.remove log_fname with _ -> ());
  (try Stdlib.Sys.remove libname with _ -> ());
  let cmdline =
    Printf.sprintf "%s %s -O%d -o %s --shared >> %s 2>&1" (compiler_command ()) f_name
      (optimization_level ()) libname log_fname
  in
  let rc = Stdlib.Sys.command cmdline in
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
  Stdlib.Gc.finalise (fun lib -> Dl.dlclose ~handle:lib.lib) result;
  result

module C_syntax_config (Input : sig
  val for_lowereds : Low_level.optimized array
  val opt_ctx_arrays : (Tn.t, buffer_ptr, Tn.comparator_witness) Base.Map.t option
end) =
struct
  let for_lowereds = Input.for_lowereds

  type nonrec ctx_array = ctx_array

  let opt_ctx_arrays = Input.opt_ctx_arrays
  let hardcoded_context_ptr = Some Ndarray.c_ptr_to_string
  let is_in_context = is_in_context
  let host_ptrs_for_readonly = true
  let logs_to_stdout = false
  let main_kernel_prefix = ""
  let kernel_prep_line = ""

  let include_lines =
    [ "#include <stdio.h>"; "#include <stdlib.h>"; "#include <string.h>"; "#include <math.h>" ]

  let typ_of_prec = Ops.c_typ_of_prec
  let binop_syntax = Ops.binop_c_syntax
  let unop_syntax = Ops.unop_c_syntax
  let convert_precision = Ops.c_convert_precision
end

let%diagn_sexp compile ~(name : string) ~opt_ctx_arrays bindings (lowered : Low_level.optimized) =
  let opt_ctx_arrays =
    Option.map opt_ctx_arrays ~f:(fun ctx_arrays ->
        Hashtbl.fold lowered.traced_store ~init:ctx_arrays ~f:(fun ~key:tn ~data:node ctx_arrays ->
            match Map.find ctx_arrays tn with
            | None ->
                if is_in_context node then
                  let debug = "CC compile-time ctx array for " ^ Tn.debug_name tn in
                  let data =
                    Ndarray.create_array ~debug (Lazy.force tn.Tn.prec) ~dims:(Lazy.force tn.dims)
                    @@ Constant_fill { values = [| 0. |]; strict = false }
                  in
                  Map.add_exn ctx_arrays ~key:tn ~data
                else ctx_arrays
            | Some _ -> ctx_arrays))
  in
  let module Syntax = Backend_utils.C_syntax (C_syntax_config (struct
    let for_lowereds = [| lowered |]
    let opt_ctx_arrays = opt_ctx_arrays
  end)) in
  (* FIXME: do we really want all of them, or only the used ones? *)
  let idx_params = Indexing.bound_symbols bindings in
  let pp_file = Utils.pp_file ~base_name:name ~extension:".c" in
  let is_global = Syntax.compile_globals pp_file.ppf in
  let params = Syntax.compile_proc ~name pp_file.ppf idx_params ~is_global lowered in
  pp_file.finalize ();
  let result = c_compile_and_load ~f_name:pp_file.f_name in
  { lowered; result; params; bindings; name; opt_ctx_arrays }

let%diagn_sexp compile_batch ~names ~opt_ctx_arrays bindings
    (lowereds : Low_level.optimized option array) =
  let for_lowereds = Array.filter_map ~f:Fn.id lowereds in
  let opt_ctx_arrays =
    Option.map opt_ctx_arrays ~f:(fun ctx_arrays ->
        Array.fold for_lowereds ~init:ctx_arrays ~f:(fun ctx_arrays lowered ->
            Hashtbl.fold lowered.traced_store ~init:ctx_arrays
              ~f:(fun ~key:tn ~data:node ctx_arrays ->
                match Map.find ctx_arrays tn with
                | None ->
                    if is_in_context node then
                      let debug = "CC compile-time ctx array for " ^ Tn.debug_name tn in
                      let data =
                        Ndarray.create_array ~debug (Lazy.force tn.Tn.prec)
                          ~dims:(Lazy.force tn.dims)
                        @@ Constant_fill { values = [| 0. |]; strict = false }
                      in
                      Map.add_exn ctx_arrays ~key:tn ~data
                    else ctx_arrays
                | Some _ -> ctx_arrays)))
  in
  let module Syntax = Backend_utils.C_syntax (C_syntax_config (struct
    let for_lowereds = for_lowereds
    let opt_ctx_arrays = opt_ctx_arrays
  end)) in
  (* FIXME: do we really want all of them, or only the used ones? *)
  let idx_params = Indexing.bound_symbols bindings in
  let global_ctx_arrays =
    ref (match opt_ctx_arrays with Some ctx_arrays -> ctx_arrays | None -> Map.empty (module Tn))
  in
  let base_name =
    String.(
      strip ~drop:(equal_char '_')
      @@ common_prefix (Array.to_list @@ Array.concat_map ~f:Option.to_array names))
  in
  let pp_file = Utils.pp_file ~base_name ~extension:".c" in
  let is_global = Syntax.compile_globals pp_file.ppf in
  let params =
    Array.mapi lowereds ~f:(fun i lowered ->
        Option.map2 names.(i) lowered ~f:(fun name lowered ->
            Syntax.compile_proc ~name pp_file.ppf idx_params ~is_global lowered))
  in
  pp_file.finalize ();
  let result = c_compile_and_load ~f_name:pp_file.f_name in
  (* Note: for simplicity, we share ctx_arrays across all contexts. *)
  let opt_ctx_arrays = Option.map opt_ctx_arrays ~f:(fun _ -> !global_ctx_arrays) in
  ( opt_ctx_arrays,
    Array.mapi params ~f:(fun i params ->
        Option.map2 names.(i) lowereds.(i) ~f:(fun name lowered ->
            {
              lowered;
              result;
              params = Option.value_exn ~here:[%here] params;
              bindings;
              name;
              opt_ctx_arrays;
            })) )

let%diagn_sexp link_compiled ~merge_buffer (prior_context : context) (code : procedure) :
    context * _ * _ * string =
  let label : string = prior_context.label in
  let name : string = code.name in
  let arrays : Ndarray.t Base.Map.M(Tn).t =
    match code with
    | { opt_ctx_arrays = Some arrays; _ } -> arrays
    | { params; _ } ->
        List.fold params ~init:prior_context.arrays ~f:(fun ctx_arrays -> function
          | _, Param_ptr tn ->
              let f = function
                | Some arr -> arr
                | None ->
                    let debug = "CC link-time ctx array for " ^ Tn.debug_name tn in
                    Ndarray.create_array ~debug (Lazy.force tn.Tn.prec) ~dims:(Lazy.force tn.dims)
                    @@ Constant_fill { values = [| 0. |]; strict = false }
              in
              Map.update ctx_arrays tn ~f
          | _ -> ctx_arrays)
  in
  let context = { label; arrays } in
  let log_file_name = Utils.diagn_log_file [%string "debug-%{label}-%{code.name}.log"] in
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
            let get_ptr (buffer, _) = Ndarray.get_voidptr_not_managed buffer in
            Param_2f (get_ptr, merge_buffer, link bs ps Ctypes.(ptr void @-> cs))
        | bs, Param_ptr tn :: ps ->
            let nd = match Map.find arrays tn with Some nd -> nd | None -> assert false in
            let c_ptr = Ndarray.get_voidptr_not_managed nd in
            Param_2 (ref (Some c_ptr), link bs ps Ctypes.(ptr void @-> cs))
      in
      (* Reverse the input order because [Indexing.apply] will reverse it again. Important:
         [code.bindings] are traversed in the wrong order but that's OK because [link] only uses
         them to check the number of indices. *)
      let params = List.rev_map code.params ~f:(fun (_, p) -> p) in
      link code.bindings params Ctypes.(void @-> returning void)]
  in
  let%diagn_l_sexp work () : unit =
    [%log_result name];
    Backend_utils.check_merge_buffer ~merge_buffer ~code_node:code.lowered.merge_node;
    Indexing.apply run_variadic ();
    if Utils.debug_log_from_routines () then (
      Utils.log_trace_tree (Stdio.In_channel.read_lines log_file_name);
      Stdlib.Sys.remove log_file_name)
  in
  ( context,
    Indexing.lowered_bindings code.bindings run_variadic,
    Tn.Task
      {
        (* In particular, keep code alive so it doesn't get unloaded. *)
        context_lifetime = (context, code);
        description = "executes " ^ code.name ^ " on " ^ context.label;
        work;
      },
    name )
