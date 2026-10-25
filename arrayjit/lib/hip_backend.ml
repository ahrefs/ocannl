open Base
open Ir
module Tn = Tnode
module Lazy = Utils.Lazy
module H = Hip
open Backend_intf

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_HIP_BACKEND=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_HIP_BACKEND"]

let () =
  H.hip_call_hook :=
    Some
      (fun ~message:_message ~status:_status ->
        [%debug_sexp
          [%log5_block
            _message;
            if not @@ H.is_success _status then [%log (_status : H.result)]]])

let _suspended () =
  H.hip_call_hook := Some (fun ~message ~status:_ -> Stdlib.Printf.printf "HIP %s\n" message)

module Backend_buffer = struct
  type buffer_ptr = H.Deviceptr.t

  let sexp_of_buffer_ptr ptr = Sexp.Atom (H.Deviceptr.string_of ptr)
end

module Device_config = struct
  include Backend_buffer

  type dev = {
    dev : H.Device.t;
    primary_context : H.Context.t;
    set_builtins_in : H.Module.t -> unit;
  }
  [@@deriving sexp_of]

  type runner = H.Stream.t [@@deriving sexp_of]
  type event = H.Delimited_event.t [@@deriving sexp_of]

  let name = "hip"
end

module Device_stream = Backend_impl.Device_types_ll (Device_config)
open Device_config

let set_ctx ctx = H.Context.set_current ctx

(* The HIP slab allocator: a private [(device_id, pool_id) -> hipDeviceptr_t] table backing the
   shared {!Backend_intf.Slab_alloc}. *)
module Slab = struct
  open Backend_intf

  type device = Device_stream.device
  type buffer_ptr = H.Deviceptr.t

  (* Requested sizes are tracked alongside the pointers so [get_used_memory] can report the bytes
     OCANNL allocated on the device (gh-ocannl-289, mirroring the CUDA backend): the driver's
     [get_free_and_total_mem] moves in allocation granules, which hides sub-granule effects such as
     the liveness planner's arena savings (gh-ocannl-489) and counts other processes' memory. On an
     APU sharing memory with the display that second effect dominates outright — it is what made
     test/operations/buffer_aliasing report the planner INCREASING the training footprint here while
     decreasing it everywhere else (gh-ocannl-542). *)
  let pools : (int * int, buffer_ptr * int) Hashtbl.Poly.t = Hashtbl.Poly.create ()

  let alloc_pool ?mode:_ (device : device) ~pool_id ~size_in_bytes ~alignment:_ =
    set_ctx device.dev.primary_context;
    let key = (device.device_id, pool_id) in
    let size_in_bytes = max 1 size_in_bytes in
    (* Free-first replacement avoids requiring both merge slabs to fit. Invalidate the complete
       ownership transaction before the fallible free/allocation sequence. *)
    Option.iter (Hashtbl.find pools key) ~f:(fun (old, _) ->
        if Int.equal pool_id merge_buffer_pool_id then
          invalidate_merge_slab device ~remove_slab_claim:(fun () -> Hashtbl.remove pools key)
        else Hashtbl.remove pools key;
        H.Deviceptr.mem_free old);
    let ptr =
      try H.Deviceptr.mem_alloc ~size_in_bytes
      with
      | H.Hip_error { status; message }
      when match H.sexp_of_result status with
           | Sexp.Atom "HIP_ERROR_OUT_OF_MEMORY" -> true
           | _ -> false
      ->
        (* gh-ocannl-1326: typed here, where the refused size is known, so that every allocation
           site reports one backend-independent shape ([Schedule_outcome.protect] classifies it). *)
        let free = try Some (fst (H.Device.get_free_and_total_mem ())) with _ -> None in
        raise
          (Schedule_outcome.Device_memory_exhausted
             {
               backend = name;
               requested = size_in_bytes;
               free;
               detail = "HIP_ERROR_OUT_OF_MEMORY raised by " ^ message;
             })
    in
    if Int.equal pool_id merge_buffer_pool_id then
      commit_merge_slab device ~size_in_bytes ~install_slab_claim:(fun () ->
          Hashtbl.set pools ~key ~data:(ptr, size_in_bytes))
    else Hashtbl.set pools ~key ~data:(ptr, size_in_bytes)

  let free_pool =
    Some
      (fun (device : device) ~pool_id ->
        let key = (device.device_id, pool_id) in
        Option.iter (Hashtbl.find pools key) ~f:(fun (ptr, _) -> H.Deviceptr.mem_free ptr);
        Hashtbl.remove pools key)

  let resolve_pool (device : device) { pool_id; offset = _ } : buffer_ptr =
    (* Return the slab base. The byte offset is NOT folded into the handle here; callers apply it
       via the hipjit ?offset / ?dst_offset / ?src_offset params or via H.Deviceptr.offset. *)
    fst (Hashtbl.find_exn pools (device.device_id, pool_id))

  let used_memory (device : device) =
    Hashtbl.fold pools ~init:0 ~f:(fun ~key:(dev_id, _) ~data:(_, size) acc ->
        if dev_id = device.device_id then acc + size else acc)

  let memset_zero (device : device) ~pool_id ~offset ~size_in_bytes =
    let base = resolve_pool device { pool_id; offset } in
    if size_in_bytes > 0 then
      H.Stream.memset_d8 ~offset base Unsigned.UChar.zero ~length:size_in_bytes device.runner
end

(* The HIP SDK include dir (no-spaces junction on Windows / HIP_PATH / /opt/rocm / /usr),
   forward-slashed for the clang command line. [None] when no SDK is found (the Linux
   built-in-headers path). [/usr] is the distro ROCm stack (Ubuntu 26.04's packages, the native
   fleet's AMD boxes), which has no /opt/rocm and is only reached through [HIP_PATH=/usr] from a
   login shell: without it a bare non-login environment -- tools/machine-verify.sh's hermetic leg --
   found no rocWMMA headers either, since [rocwmma_include_dir] searches this tree, and every WMMA
   test silently ran the scalar fallback while staying green (gh-ocannl-1070). It comes last so an
   explicit [HIP_PATH] or a vendor /opt/rocm install still wins over distro headers. Lifted out of
   [Impl] for the same reason [Cuda_backend.cuda_include_options] was: it is a policy every hiprtc
   caller has to agree with, and tools/fp8_soak.ml is one -- a soak that guessed its own include
   path would report the HIP arm ready and then fail to compile its kernel wherever the SDK does not
   sit where the guess looked (Codex P2 on PR #463, for the CUDA side of the same program). *)
let hip_sdk_include_dir =
  lazy
    (let candidates =
       (match Sys.getenv "LOCALAPPDATA" with Some l -> [ l ^ "/hip_path_link" ] | None -> [])
       @ (match Sys.getenv "HIP_PATH" with Some p -> [ p ] | None -> [])
       @ [ "/opt/rocm"; "/usr" ]
     in
     List.find_map candidates ~f:(fun p ->
         if Stdlib.Sys.file_exists (p ^ "/include/hip/hip_fp16.h") then
           Some (String.map ~f:(fun c -> if Char.(c = '\\') then '/' else c) (p ^ "/include"))
         else None))

let hip_include_options () =
  match Lazy.force hip_sdk_include_dir with Some d -> [ "-I" ^ d ] | None -> []

(* The two guarded narrowing helpers HIP emits in place of a bare cast (gh-ocannl-647), as SOURCE
   TEXT, so a caller sweeping them sweeps the shipped definitions rather than a transcription of
   them -- tools/fp8_soak.ml's HIP arm, whose whole subject is what the guard does and does not
   change. Raises rather than silently returning less if either helper is renamed: a soak that
   quietly stopped sweeping the guard would keep printing the same passing line. *)
let fp8_guard_helper_names = [ "ocannl_single_to_fp8_uniform"; "ocannl_double_to_fp8_uniform" ]

let fp8_guard_source () =
  List.map fp8_guard_helper_names ~f:(fun wanted ->
      match List.find Builtins_hip.builtins ~f:(fun (name, _, _) -> String.equal name wanted) with
      | Some (_, code, _) -> code
      | None ->
          raise
          @@ Utils.User_error
               ("Hip_backend.fp8_guard_source: no builtin named " ^ wanted
              ^ " -- Builtins_hip and this list have drifted apart"))
  |> String.concat ~sep:"\n"

(* The bf16-residency predicate behind [accum_prec] and the mma combination table, device-free in
   [Hip_mma] beside them (gh-ocannl-1246). *)
let bf16_accum_wide = Hip_mma.bf16_accum_wide

(* A directory containing a COMPLETE rocWMMA header tree, if any: [ROCWMMA_PATH] variants, a clone
   under [%LOCALAPPDATA%/rocwmma], or the HIP include tree (rocWMMA installs there on Linux).
   rocWMMA is header-only and is NOT in the ROCm Windows SDK, hence the extra search paths.

   "Complete" is load-bearing, not pedantry (gh-ocannl-1032): Ubuntu 26.04's librocwmma-dev 7.1.0
   installs the five umbrella headers into /usr/include/rocwmma WITHOUT the rocwmma/internal/
   directory that rocwmma.hpp's very first include needs, and that is the tree an [apt install
   rocwmma] leaves behind on the distro ROCm stack. Probing rocwmma.hpp alone accepts it, and then
   every tensorized kernel fails at hiprtc with "'internal/accessors.hpp' file not found" -- turning
   a clean capability decline into a compile error on the one path that is supposed to be
   unreachable when the headers are absent. So require an internal header too; [types.hpp] is one
   rocwmma.hpp pulls in unconditionally. *)
let rocwmma_include_dir =
  lazy
    (let candidates =
       (match Sys.getenv "ROCWMMA_PATH" with
         | Some p -> [ p; p ^ "/include"; p ^ "/library/include" ]
         | None -> [])
       @ (match Sys.getenv "LOCALAPPDATA" with
         | Some l -> [ l ^ "/rocwmma/library/include" ]
         | None -> [])
       @ match Lazy.force hip_sdk_include_dir with Some d -> [ d ] | None -> []
     in
     List.find candidates ~f:(fun p ->
         Stdlib.Sys.file_exists (p ^ "/rocwmma/rocwmma.hpp")
         && Stdlib.Sys.file_exists (p ^ "/rocwmma/internal/types.hpp"))
     |> Option.map ~f:(String.map ~f:(fun c -> if Char.(c = '\\') then '/' else c)))

(* hiprtc ships built-in HIP headers, and device-side printf needs no declaration on ROCm. Top level
   rather than in [Impl] so [host_capabilities] fingerprints exactly the headers it names. *)
let hip_includes =
  {|#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
/* hip_fp8.h ships with ROCm >= 6.2 (hiprtc is clang, so __has_include is available); guarding
   keeps non-fp8 kernels compiling on older SDKs, where fp8 kernels still fail with unknown type
   __hip_fp8_e5m2. */
#if __has_include(<hip/hip_fp8.h>)
#include <hip/hip_fp8.h>
#endif

/* Define math constants that would normally come from <math.h> */
#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif
#ifndef NAN
#define NAN __builtin_nanf("")
#endif|}

(* gh-ocannl-1038: what this backend probes on the host without opening a device -- the SDK include
   tree hiprtc compiles against, the rocWMMA header tree [Impl.mma_supported] requires, and the
   hiprtc library's version -- in the form {!Context.host_capabilities} records, so a tree appearing
   at an already-searched path reruns the tests whose verdicts read it. *)
let host_capabilities () =
  [
    (* The headers fingerprinted are the ones kernels include, read off the text that includes them,
       so a header added to the prelude is tracked without a second list to keep in step. *)
    ( "hip_include",
      Utils.capability_dir
        ~requires:(Utils.included_headers hip_includes)
        (Lazy.force hip_sdk_include_dir) );
    ( "rocwmma_include",
      Utils.capability_dir
        ~requires:
          (List.concat_map Cuda_like_config.Hip.conditional_includes ~f:(fun (_, directive) ->
               Utils.included_headers directive)
          (* The completeness half of [rocwmma_include_dir]'s own probe. *)
          @ [ "rocwmma/internal/types.hpp" ])
        (Lazy.force rocwmma_include_dir) );
    ( "hiprtc",
      try
        let major, minor = Hiprtc.version () in
        Printf.sprintf "%d.%d" major minor
      with _ -> "unavailable" );
  ]

(* [initialized_devices] never forgets its entries. *)
let initialized_devices = Hash_set.create (module Int)
let initialized = ref false

module Impl : sig
  include Ir.Backend_impl.Lowered_backend

  val hip_to_code : name:string -> string -> Hiprtc.compile_to_code_result
end = struct
  include Backend_impl.Device (Device_stream) (Slab)

  (* The concrete [buffer_ptr]/[buffer] + sexps for the impl-facing interface (no longer carried by
     the shared [Device_config_common]). *)
  include Backend_buffer

  let ctx_of (context : context) = context.device.dev.primary_context
  let is_done event = H.Delimited_event.query event
  let will_wait_for context event = H.Delimited_event.wait context.device.runner event
  let sync event = H.Delimited_event.synchronize event
  let all_work device = H.Delimited_event.record device.runner

  (* The inline-test harness ([ppx_inline_test]'s runner, which hosts ppx_expect tests) is detected
     by its command line, the same way [Ppx_inline_test_lib] switches to test mode. *)
  let am_running_inline_tests =
    Array.length Stdlib.Sys.argv > 1 && String.equal Stdlib.Sys.argv.(1) "inline-test-runner"

  (* On Windows, amdhip64_N.dll prints a "HIP Library Path: <dll>" banner to stdout when the runtime
     first initializes (rocclr os_win32.cpp; unconditional — not gated by AMD_LOG_LEVEL), which
     would pollute expect-test and .expected-test outputs. Silence OS-level stdout (fd 1) across the
     first HIP call; the redirect is harmless on platforms without the banner. Inside a ppx_expect
     capture the descriptor games corrupt the harness's bookkeeping (removing its capture file fails
     with a sharing violation), so skip there — the inline-test case is instead handled by the eager
     module-init forcing below, which runs before any capture starts. *)
  let quiet_first_call f =
    if am_running_inline_tests then f ()
    else (
      Stdlib.flush Stdlib.stdout;
      let original = Unix.dup Unix.stdout in
      let devnull =
        Unix.openfile (if Stdlib.Sys.win32 then "NUL" else "/dev/null") [ Unix.O_WRONLY ] 0o666
      in
      Unix.dup2 devnull Unix.stdout;
      Exn.protect ~f ~finally:(fun () ->
          Stdlib.flush Stdlib.stdout;
          Unix.dup2 original Unix.stdout;
          Unix.close original;
          Unix.close devnull))

  (* Driver initialization and device discovery are lazy: the singleton [Impl] module initializes at
     program startup (Backends instantiates it eagerly for nameable types), and hipjit is a depopt
     -- the library being installed does not imply a usable driver/GPU. Forcing here, at first
     device use, keeps CPU-only runs from touching the driver and lets [Context.auto] catch
     unusable-HIP failures per call. *)
  let ensure_initialized =
    lazy
      (if not !initialized then (
         quiet_first_call (fun () -> H.init ());
         initialized := true))

  (* Under the inline-test harness, when the session's backend is [hip], force runtime
     initialization at module init: the "HIP Library Path" banner then goes to the runner's own
     stdout instead of the first [%expect] block (no ppx_expect capture is active yet, so the banner
     cannot corrupt or pollute test output). Failures are swallowed — an unusable driver/GPU
     surfaces at [get_device], where [Context.auto]'s fallback can catch it. This is the one driver
     entry that runs before [Backends.With_device_init_deadline] wraps the module, so it takes the
     same deadline itself (gh-ocannl-1281): a wedged driver would otherwise hang every inline-test
     runner in [H.init] with no bound. *)
  let () =
    if
      am_running_inline_tests
      && String.equal "hip"
           (String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:""))
    then
      try
        Utils.with_device_init_deadline ~backend_name:name (fun () -> Lazy.force ensure_initialized)
      with _ -> ()

  let num_devices () =
    Lazy.force ensure_initialized;
    H.Device.get_count ()

  (* [devices] is mutable to support plugging in new devices. *)
  let devices = lazy (ref @@ Array.create ~len:(num_devices ()) None)

  (* Bytes OCANNL has allocated on this device via [Slab], exact rather than the driver's
     granule-quantized [total - free] (gh-ocannl-289). Device-wide across contexts, matching the
     [Context.get_used_memory] contract. *)
  let get_used_memory (device : device) = Slab.used_memory device

  let get_free_and_total_memory (device : device) =
    set_ctx device.dev.primary_context;
    Some (H.Device.get_free_and_total_mem ())

  (* The merge buffer is the device's reserved single-tenant pool (id [merge_buffer_pool_id]); grow
     it in place when a larger node arrives ([Slab.alloc_pool] overwrites the reserved entry). *)
  let opt_alloc_merge_buffer ~size_in_bytes (device : device) : unit =
    if device.merge_buffer_capacity < size_in_bytes then
      Slab.alloc_pool device ~pool_id:merge_buffer_pool_id ~size_in_bytes ~alignment:1

  let%track4_sexp finalize_device (device : device) =
    H.Context.set_current device.dev.primary_context;
    H.Context.synchronize ();
    (* gh-ocannl-344: constants are bump-packed, so several cache entries share one constant pool
       slab; free each distinct [pool_id] exactly once (freeing per-entry would double-free / free a
       sub-region pointer). [Slab.free_pool] frees the slab and drops its table entry. *)
    Hashtbl.data device.constant_buffer_cache
    |> List.map ~f:(fun (loc : Backend_intf.buffer_loc) -> loc.pool_id)
    |> List.dedup_and_sort ~compare:Int.compare
    |> List.iter ~f:(fun pool_id ->
        Option.iter Slab.free_pool ~f:(fun free -> free device ~pool_id))

  (* --- Cooperative tile-MMA (rocWMMA) capability, shared by [hardware_limits] and [mma_syntax].
     Requires BOTH the RDNA3/RDNA3.5+ (gfx11/gfx12) wave32 architecture AND discoverable rocWMMA
     headers. Gating both sites matters: [hardware_limits] keeps autotune/scheduling from selecting
     [Tile_mma] where it cannot work, and [mma_syntax] makes a manual [Sched.tensorize] on an
     unsupported device (CDNA gfx9 wave64, a mixed fleet) or a host without rocWMMA decline to the
     scalar fallback rather than emit an uncompilable kernel. Memoized behind [lazy]: device
     enumeration and filesystem probes must not run at module init. *)
  let tile_mma_eligible (a : H.Device.attributes) =
    (String.is_prefix a.gcn_arch_name ~prefix:"gfx11"
    || String.is_prefix a.gcn_arch_name ~prefix:"gfx12")
    && a.warp_size = 32

  (* Device discovery a compile can force first: under the device-init deadline like the guarded
     entry points (gh-ocannl-1281, [Backends.With_device_init_deadline]). *)
  let all_rdna_wave32 =
    lazy
      (Utils.with_device_init_deadline ~backend_name:name (fun () ->
           let n = num_devices () in
           n > 0
           && Array.for_all
                (Array.init n ~f:(fun ordinal -> H.Device.get_attributes (H.Device.get ~ordinal)))
                ~f:tile_mma_eligible))

  let mma_supported () =
    Lazy.force all_rdna_wave32 && Option.is_some (Lazy.force rocwmma_include_dir)

  (* Compilation is backend-wide, not per-context. A compiled artifact may link on any device, so
     the guard must cover the linkable set even when HIPRTC's implicit current-device target is a
     different ordinal. Querying attributes does not switch that current target. The cache uses this
     same set and option selector. *)
  let hiprtc_target_archs () =
    (* Device discovery on every compile: under the device-init deadline (gh-ocannl-1281). *)
    Utils.with_device_init_deadline ~backend_name:name (fun () ->
        Array.init (num_devices ()) ~f:(fun ordinal ->
            (H.Device.get_attributes (H.Device.get ~ordinal)).gcn_arch_name)
        |> Array.to_list)

  (* The options hiprtc compiles [hip_src] with -- the source as assembled, before
     [prepend_conditional_includes]: whether it uses rocWMMA adds that header tree's include path
     and the language level it needs. A function of the source and the device set alone, so the
     kernel code cache keys on it (gh-ocannl-1314). *)
  let hiprtc_options ~target_archs hip_src =
    let uses_rocwmma = C_syntax.source_mentions ~marker:"rocwmma::" hip_src in
    (* hiprtc targets the architecture of the current default device when no [--offload-arch] is
       given. On Linux hiprtc ships built-in HIP headers; on Windows (observed with ROCm 7.1)
       [#include <hip/hip_fp16.h>] is not found without an include path, so point at the SDK's
       include directory ([hip_sdk_include_dir]: the no-spaces junction created by ocaml-hipjit,
       falling back to HIP_PATH, /opt/rocm or /usr). The -I is only added when the directory exists,
       so a Linux box with no SDK tree keeps hiprtc's built-in headers. *)
    let hip_include_opt = hip_include_options () in
    (* rocWMMA include dir, only for tensor-core kernels ([rocwmma_include_dir] finds the dir
       holding [rocwmma/rocwmma.hpp]). *)
    let rocwmma_include_opt =
      if not uses_rocwmma then []
      else match Lazy.force rocwmma_include_dir with Some d -> [ "-I" ^ d ] | None -> []
    in
    (* gh-ocannl-735: hiprtc otherwise reassociates ordinary scalar bf16/f16 recurrences differently
       across loop, repeated-statement and scope-local spellings. [Compiler_options.hiprtc] keeps
       the narrow override -- and the infinity sentinels the masked-softmax path emits as values --
       after fast math at compiler scope, covering operators parsed in the HIP headers; its complete
       option matrix is tested without hipjit in arrayjit/test. *)
    Compiler_options.hiprtc ~target_archs ~hip_include_options:hip_include_opt
      ~rocwmma_include_options:rocwmma_include_opt ~uses_rocwmma
      ~with_debug:(Utils.with_runtime_debug ())

  let with_conditional_includes hip_src =
    C_syntax.prepend_conditional_includes
      ~conditional_includes:Cuda_like_config.Hip.conditional_includes hip_src

  let write_source_artifact ~name hip_src =
    if Utils.settings.output_debug_files_in_build_directory then (
      let build_file = Utils.open_build_file ~base_name:name ~extension:".hip" in
      Stdio.Out_channel.output_string build_file.oc hip_src;
      build_file.finalize ())

  (* One hiprtc program. [log_header] opens the [.hip_log] artifact: the kernel cache's account of
     which kernels this program compiled. *)
  let%diagn2_sexp compile_program ~name ?(log_header = "") ~options hip_src =
    let name_hip = name ^ ".hip" in
    [%log "compiling to a code object"];
    (* [with_debug] only asks hiprtc to keep the compilation log on a SUCCESSFUL compile (a failing
       one carries its log in the exception either way), and the only reader of that log is the
       [.hip_log] build file written just below -- so it wants exactly the flag that writes it. The
       [|| log_level > 0] disjunct it used to carry made the log's retention hostage to a verbosity
       knob that nothing here consults (gh-ocannl-595). *)
    let with_debug = Utils.settings.output_debug_files_in_build_directory in
    let code =
      C_syntax.with_compiler_options ~compiler:"hiprtc" ~options
        ~enrich:(fun exn ~suffix ->
          match exn with
          | Hiprtc.Hiprtc_error { status; message } ->
              Some (Hiprtc.Hiprtc_error { status; message = message ^ suffix })
          | _ -> None)
        (fun () -> Hiprtc.compile_to_code ~hip_src ~name:name_hip ~options ~with_debug)
    in
    if Utils.settings.output_debug_files_in_build_directory then (
      let oc = Stdio.Out_channel.create ~binary:true @@ Utils.build_file @@ name ^ ".hsaco" in
      Stdio.Out_channel.output_string oc @@ Hiprtc.string_from_code code;
      Stdio.Out_channel.flush oc;
      Stdio.Out_channel.close oc;
      let oc = Out_channel.open_text @@ Utils.build_file @@ name ^ ".hip_log" in
      Stdio.Out_channel.output_string oc log_header;
      Stdio.Out_channel.output_string oc @@ Option.value ~default:"" (Hiprtc.compilation_log code);
      Stdio.Out_channel.flush oc;
      Stdio.Out_channel.close oc);
    code

  let hip_to_code ~name hip_src =
    let options =
      Lazy.force ensure_initialized;
      hiprtc_options ~target_archs:(hiprtc_target_archs ()) hip_src
    in
    let hip_src = with_conditional_includes hip_src in
    write_source_artifact ~name hip_src;
    compile_program ~name ~options hip_src

  let run_options () =
    (* NOTE: on the AMD platform these are accepted for CUDA-driver-API compatibility but mostly
       ignored by [hipModuleLoadDataEx]. *)
    if Utils.with_runtime_debug () then
      H.Module.[ GENERATE_DEBUG_INFO true; GENERATE_LINE_INFO true ]
    else []

  (* No runtime linking needed since Threefry is included directly in each kernel *)
  let set_builtins_for_device ~primary_context:_ _kernel_module = assert !initialized

  (* gh-ocannl-1445: a graph-captured routine's instantiated graphs, by capture key, in FIFO order
     for eviction (see [sequence_segments]). Destroyed when the routine's context is released
     ([destroy_graphs], one of its [release_hooks]) and at process exit, before the device's stream
     (below), not only on eviction: an undestroyed [hipGraphExec] at exit is what kept ROCr's
     [SharedSignalPool] leak report coming back with graph capture on (gh-ocannl-1370). *)
  type graph_cache = {
    graphs_device : device;
    cache : (string, H.Graph.exec) Hashtbl.t;
    order : string Queue.t;
    lock : Stdlib.Mutex.t;
        (** Held by every run of the routine and by [destroy_graphs]: upload forks share their
            context's release hooks and can be released from different domains. *)
  }

  let graph_caches : graph_cache Backend_impl.Weak_registry.t = Backend_impl.Weak_registry.create ()

  let live_graph_execs () =
    List.sum
      (module Int)
      (Backend_impl.Weak_registry.live graph_caches)
      ~f:(fun g -> Stdlib.Mutex.protect g.lock (fun () -> Hashtbl.length g.cache))

  (* Idempotent, and the routine stays runnable: its next run captures afresh. The stream is
     synchronized first, since a cached exec may still have a launch pending on it; the cache is
     emptied under its lock, and every exec taken out of it is attempted, the first failure
     re-raised. *)
  let destroy_graphs g =
    let execs =
      Stdlib.Mutex.protect g.lock (fun () ->
          if Hashtbl.is_empty g.cache then []
          else (
            set_ctx g.graphs_device.dev.primary_context;
            H.Stream.synchronize g.graphs_device.runner;
            let execs = Hashtbl.data g.cache in
            Hashtbl.clear g.cache;
            Queue.clear g.order;
            execs))
    in
    Backend_impl.attempt_each (List.map execs ~f:(fun exec () -> H.Graph.exec_destroy exec))

  (* gh-ocannl-1036: destroy each device's stream at process exit. A HIP stream that is never
     destroyed makes the ROCm runtime report its 77 signals as leaked on stderr when the process
     ends; the GC finalizer above does not run at exit, and would not destroy the stream if it did.
     [H.Stream.destroy] synchronizes unboundedly, so it is called only once [is_ready] says there is
     nothing to wait for (see [Utils.bounded_exit_teardown]). Registered at module initialization,
     not when the first device opens: [at_exit] handlers run most-recent first, and every module
     that can reach this backend initializes after this one, so every handler that might still use a
     stream runs before the streams go. The device records stay in [devices], so no GC finalizer
     reaches a destroyed stream, and [H.Stream.destroy] is idempotent anyway. A process that never
     opened a device finds [devices] unforced and does nothing. One bound covers every device. *)
  let () =
    Stdlib.at_exit (fun () ->
        if Lazy.is_val devices then
          ignore
            (Utils.bounded_exit_teardown
               (Array.to_list !(Lazy.force devices)
               |> List.filter_opt
               |> List.map ~f:(fun (device : device) : Utils.exit_teardown_resource ->
                   {
                     what = [%string "the HIP stream of device %{device.ordinal#Int}"];
                     is_idle =
                       (fun () ->
                         set_ctx device.dev.primary_context;
                         H.Stream.is_ready device.runner);
                     teardown =
                       (fun () ->
                         (* gh-ocannl-1445: the device's cached graphs first, then its stream. *)
                         (* Every cache is attempted, and the stream is destroyed whatever
                            they raise. *)
                         Exn.protect
                           ~f:(fun () ->
                             Backend_impl.Weak_registry.live graph_caches
                             |> List.filter ~f:(fun g -> phys_equal g.graphs_device device)
                             |> List.map ~f:(fun g () -> destroy_graphs g)
                             |> Backend_impl.attempt_each)
                           ~finally:(fun () -> H.Stream.destroy device.runner));
                   }))
              : Utils.exit_teardown_outcome list))

  let%track3_sexp get_device ~(ordinal : int) : device =
    let n = num_devices () in
    (* See the corresponding note in [Cuda_backend.get_device]. *)
    if n = 0 then
      raise
      @@ Backend_intf.Backend_unavailable
           { backend = name; detail = "the driver reports no HIP devices" };
    if n <= ordinal then
      invalid_arg [%string "Exec_as_hip.get_device %{ordinal#Int}: not enough devices"];
    let devices = Lazy.force devices in
    (if Array.length !devices <= ordinal then
       let old, len = (!devices, Array.length !devices) in
       devices := Array.init (ordinal + 1) ~f:(fun i -> if i < len then old.(i) else None));
    let default () =
      let dev = H.Device.get ~ordinal in
      let primary_context : H.Context.t = H.Context.get_primary dev in
      let set_builtins_in = set_builtins_for_device ~primary_context in
      let dev = { dev; primary_context; set_builtins_in } in
      set_ctx primary_context;
      if Utils.debug_log_from_routines () && not (Hash_set.mem initialized_devices ordinal) then
        Int.of_string_opt @@ Utils.get_global_arg ~arg_name:"hip_printf_fifo_size" ~default:""
        |> Option.iter ~f:H.Context.(set_limit PRINTF_FIFO_SIZE);
      Hash_set.add initialized_devices ordinal;
      (* gh-ocannl-1370: an assertions-enabled ROCr prints one [time stamps may be invalid.] per
         recorded event, and events need a device and its stream, so the C stderr is replaced by the
         collapsing stream here, once a HIP device is confirmed -- never in a process that probed
         HIP and fell back to another backend ([Context.auto] with no HIP device). *)
      ignore (Lazy.force Utils.c_stderr_noise_collapsed : bool);
      (* With one compute stream per device, the runner (HIP stream) is created with the device. *)
      let hip_stream = H.Stream.create ~non_blocking:true () in
      let result = make_device dev hip_stream ~ordinal in
      Stdlib.Gc.finalise finalize_device result;
      !devices.(ordinal) <- Some result;
      result
    in
    Option.value_or_thunk !devices.(ordinal) ~default

  let _hip_properties =
    let cache =
      let%debug2_sexp f (ordinal : int) =
        let dev = get_device ~ordinal in
        lazy (H.Device.get_attributes dev.dev.dev)
      in
      lazy (Array.init (num_devices ()) ~f)
    in
    let%debug2_sexp get_props (device : device) : H.Device.attributes =
      let cache = Lazy.force cache in
      Lazy.force cache.(device.ordinal)
    in
    get_props

  let await (device : device) : unit =
    set_ctx device.dev.primary_context;
    (* Device-side [printf] is buffered outside the stream. On ROCm, stream synchronization can
       return while the printf FIFO is still being copied to host stdout; use device synchronization
       while routine logging is enabled so callers may safely close or restore stdout afterward. *)
    if Utils.debug_log_from_routines () then H.Context.synchronize ()
    else H.Stream.synchronize device.runner

  let is_idle (device : device) = H.Stream.is_ready device.runner

  (* Transfers take {!Backend_intf.buffer_loc} and resolve to the concrete device pointer here,
     against the device's private pool table. We pass [~length] explicitly to [memcpy_H_to_D] /
     [memcpy_D_to_H]: without it the hipjit impl computes [size_in_bytes = full_size - offset],
     which would reduce the copy to 0 bytes when the tensor is placed at an offset equal to its own
     size (the common bump-packed case). *)
  let from_host ~dst ~dst_loc hosted =
    set_ctx @@ ctx_of dst;
    let base = Slab.resolve_pool dst.device dst_loc in
    let f src =
      let full_bytes = Bigarray.Genarray.size_in_bytes src in
      let elem_bytes = Bigarray.kind_size_in_bytes (Bigarray.Genarray.kind src) in
      H.Stream.memcpy_H_to_D ~length:(full_bytes / elem_bytes) ~dst_offset:dst_loc.offset ~dst:base
        ~src dst.device.runner
    in
    Ndarray.apply { f } hosted

  let to_host ~src ~src_loc hosted =
    set_ctx @@ ctx_of src;
    let base = Slab.resolve_pool src.device src_loc in
    let f dst =
      let full_bytes = Bigarray.Genarray.size_in_bytes dst in
      let elem_bytes = Bigarray.kind_size_in_bytes (Bigarray.Genarray.kind dst) in
      H.Stream.memcpy_D_to_H ~length:(full_bytes / elem_bytes) ~src_offset:src_loc.offset ~dst
        ~src:base src.device.runner
    in
    Ndarray.apply { f } hosted

  let device_to_device tn ~into_merge_buffer ~dst_loc ~dst ~src_loc ~src =
    let dev = dst.device in
    let same_device = dev.ordinal = src.device.ordinal in
    let size_in_bytes = Lazy.force tn.Tn.size_in_bytes in
    let src_base = Slab.resolve_pool src.device src_loc in
    let src_offset = src_loc.offset in
    let memcpy ~dst_base ~dst_offset =
      if same_device then
        H.Stream.memcpy_D_to_D ~size_in_bytes ~dst_offset ~src_offset ~dst:dst_base ~src:src_base
          dst.device.runner
      else
        (* Note: unlike CUDA, HIP identifies peers by device rather than by context. *)
        H.Stream.memcpy_peer ~size_in_bytes ~dst_offset ~src_offset ~dst:dst_base
          ~dst_device:dst.device.dev.dev ~src:src_base ~src_device:src.device.dev.dev
          dst.device.runner
    in
    match (into_merge_buffer, dst_loc) with
    | No, None -> invalid_arg "Hip_backend.device_to_device: missing dst_loc"
    | No, Some dst_loc ->
        set_ctx @@ ctx_of dst;
        let dst_base = Slab.resolve_pool dst.device dst_loc in
        memcpy ~dst_base ~dst_offset:dst_loc.offset
    | Copy, _ ->
        set_ctx @@ ctx_of dst;
        opt_alloc_merge_buffer ~size_in_bytes dst.device;
        let loc = Option.value_exn ~here:[%here] !(dst.device.merge_buffer) in
        let dst_base = Slab.resolve_pool dst.device loc in
        memcpy ~dst_base ~dst_offset:loc.offset

  (* One hiprtc program's code object. [program_id] identifies it, so a batch whose kernels come
     from the same program loads it once ([link_batch]). With the kernel code cache (gh-ocannl-1314)
     the kernels of one batch can come from several programs: the ones it reused from earlier
     compiles, and the one program it compiled its cache misses into. *)
  type program = { program_id : int; code : Hiprtc.compile_to_code_result } [@@deriving sexp_of]

  type code = {
    code : program;
    symbol : string;
        (** What [code] holds this kernel under: [name] unless the kernel code cache answered with
            code compiled under another name (gh-ocannl-1525). *)
    kparams : (string * kparam_source) list;
    bindings : Indexing.unit_bindings;
    name : string;
    launch : Low_level.launch_dims;
  }
  [@@deriving sexp_of]

  type code_batch = {
    programs : (program * string) array;
        (** Per kernel, in batch order: the program holding it, and the symbol it holds it under
            (gh-ocannl-1525: not necessarily the kernel's current name, which diagnostics use). *)
    bindings : Indexing.unit_bindings;
    kparams_and_names : ((string * kparam_source) list * string * Low_level.launch_dims) array;
  }
  [@@deriving sexp_of]

  module Hip_syntax_config (Input : sig
    val procs : Low_level.t array
  end) =
  struct
    include
      Cuda_like_config.Make
        (struct
          include Cuda_like_config.Hip

          let builtins = Builtins_hip.builtins
          let extra_blacklist = []
        end)
        (Input)

    (* gh-ocannl-663: serial-rendered reduction accumulators mirror the mma legs' residency. Unlike
       CUDA, RDNA WMMA has genuine bf16 (and f16) accumulator variants and the uniform 16-bit
       triples are seeded, so wherever the policy keeps the narrow ARM a 16-bit accumulator keeps
       its storage residency too — widening only the serial legs would re-introduce the
       serial-vs-mma width dependence gh-ocannl-639 removes. Which arm the policy picks is the
       residency: f16 is narrow unless [Fp16_wide]; bf16 is WIDE unless [Bf16_narrow]
       (gh-ocannl-1051, below). fp8 has an accumulator format on no backend (its serial arithmetic
       already bridges through float per operator), so it follows the CPU policy: f32 residency, one
       narrowing per nest, governed by the same [narrow_compute_f32] knob.

       Under [Numerics.Fp16_wide] (gh-ocannl-680) f16 accumulators reside in f32 here too, and the
       mma legs follow — not by being withheld, but by swapping arms: [mma_combo] renders the
       uniform-f16 combination against a [float] accumulator fragment, converting at the [d]
       boundary (gh-ocannl-789), so width stays schedule-uniform WITH the tensor unit rather than
       against it, and [mma_f16_wide_acc_scopes] advertises both emission scopes.
       [Numerics.Bf16_wide] does the same for bf16 (gh-ocannl-838) — the uniform-bf16 arm swaps to
       an f32 accumulator fragment — which is what takes the uniform-bf16 legs off gfx11's
       bf16-accumulate WMMA, whose result is not exactly rounded. Since gh-ocannl-1051 [Bf16_auto]
       resolves wide here too ([bf16_accum_wide] above); only [Bf16_narrow] keeps bf16 storage
       residency. *)
    let accum_prec prec =
      match prec with
      | Ops.Half_prec _ when Numerics.fp16_accum_wide () -> Ops.single
      | Ops.Bfloat16_prec _ when bf16_accum_wide () -> Ops.single
      | Ops.Fp8_prec _ when (Numerics.get ()).Numerics.narrow_compute_f32 -> Ops.single
      | _ -> prec

    (* --- Shared rocWMMA tile-MMA vocabulary, used by both [mma_syntax] and [mma_fragment_syntax].
       The combination table lived in both hooks verbatim before gh-ocannl-789 added an arm to it;
       one copy is what keeps a new arm from reaching only half the emission (the two hooks' guards
       are required to accept together). *)
    let mma_tile = Hip_mma.tile
    let mma_frag_typ = Hip_mma.frag_typ

    (* The combination table both hooks consult, device-free in [Hip_mma] beside the descriptor it
       must correspond to (gh-ocannl-1246): per storage triple, the accumulator fragment's precision
       and the wmma leading-dimension multiples for a/b and for d. *)
    let mma_accumulator = Hip_mma.mma_accumulator

    (* rocWMMA's element type for a precision. It need not be textually identical to the node's own
       C type ([__half] / [__hip_bfloat16]), so the operand pointers are [reinterpret_cast] to it at
       each call site. *)
    let rocwmma_elt = function
      | Ops.Single_prec _ -> "float"
      | Ops.Half_prec _ -> "rocwmma::float16_t"
      | Ops.Bfloat16_prec _ -> "rocwmma::bfloat16_t"
      | prec -> invalid_arg ("Hip_backend.rocwmma_elt: " ^ Ops.prec_string prec)

    (* (a/b fragment element type, accumulator fragment element type, [d] STORAGE element type) of
       [mma_accumulator]'s arm. Whether a call renders at all is [rocwmma_accepts]'. *)
    let mma_combo ~a_prec ~b_prec ~d_prec =
      Option.map (mma_accumulator ~a_prec ~b_prec ~d_prec) ~f:(fun (acc_prec, _, _) ->
          (rocwmma_elt a_prec, rocwmma_elt acc_prec, rocwmma_elt d_prec))

    (* The one acceptance predicate of the three rocWMMA sites below ([Hip_mma.accepts],
       gh-ocannl-1456), behind the device gate. A [`Fragment] operand is never an addressed one. *)
    let rocwmma_accepts ~a_prec ~b_prec ~d_prec ~m ~n ~k ~d ~a:(lda, a_space, a_layout)
        ~b:(ldb, b_space, b_layout) =
      let place space layout =
        match space with
        | `Fragment _ -> None
        | (`Device | `Shared | `Thread) as space -> Some (space, layout)
      in
      mma_supported ()
      &&
      match (place a_space a_layout, place b_space b_layout) with
      | Some a, Some b -> Hip_mma.accepts ~a_prec ~b_prec ~d_prec ~m ~n ~k ~d ~a:(lda, a) ~b:(ldb, b)
      | _ -> false

    (* A destination as [Hip_mma.accepts] takes it: a [`Fragment] one is the resident fragment array
       of an update-only statement, any other an addressed operand. *)
    let rocwmma_d ~ldd space layout =
      match space with
      | `Fragment _ -> `Resident layout
      | (`Device | `Shared | `Thread) as space -> `Addressed (ldd, (space, layout))

    let mma_arm = Hip_mma.mma_arm

    (* gh-ocannl-1456: how [Hip_mma.k_loop_lines], which both hooks render, loads its operands. *)
    let mma_operand_loads = Hip_mma.operand_loads
    let mma_operand_fragments = Hip_mma.operand_fragments

    (* The [d] boundary of the [mt] x [nt] accumulator-fragment array [acc]: [`Load] brings the
       destination's 16x16 blocks at [__mma_dp] into [acc[__mi][__ni]], [`Store] writes them back.
       Where the accumulator element type is the destination's storage type this is rocWMMA's own
       load/store per block, exactly as before gh-ocannl-789.

       Where they differ -- the wide-f16 and wide-bf16 arms, a [float] accumulator over a 16-bit
       storage destination -- neither rocWMMA call is type-correct, so each element crosses the
       boundary with a scalar conversion at the coordinate the ACCUMULATOR FRAGMENT TYPE ITSELF
       names (gh-ocannl-1064, the port of CUDA's gh-ocannl-925 [wmma_d_boundary_lines]): each lane
       [load_matrix_sync]s the row-major table [ocannl_wmma_rc16] (entry [16 * row + col] holds that
       very number, exact in f32) into a fragment of exactly [acc]'s type, after which [x[i]] of
       that fragment names the (row, col) that [x[i]] of every fragment of that type holds. This
       relies only on [load_matrix_sync]'s contract and on the position being a property of the type
       and lane, not of the data -- which [mma_sync] itself requires.

       NOT the element copy through a destination-typed staging fragment that gh-ocannl-789 shipped.
       That copy assumed the [float] and 16-bit accumulator fragments place [x[i]] at the same
       coordinate, and rocWMMA says the opposite: its [fragment] class is documented with "vector
       elements have no guaranteed order or locality", and the accumulator's register layout is a
       template of the element type ([MmaDimSelector<BlockDim, DataT>] and
       [RegisterLayout::MmaAcc<MmaDim, DataT, ...>] in [rocwmma/internal/io_layout.hpp]). The copy
       happened to be right on gfx1151 and gfx1102 (gh-ocannl-838's k-split discriminators read the
       wide values bitwise), which is evidence about that rocWMMA release on those devices only. The
       table costs one cached fragment load per boundary, twice per fragment scope. *)
    let mma_d_boundary_lines ~dir ~acc_typ ~d_typ ~acc ~ldd ~mt ~nt =
      let nest body =
        [
          Printf.sprintf "for (int __mi = 0; __mi < %d; ++__mi) {" mt;
          Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
        ]
        @ List.map ~f:(fun l -> "    " ^ l) body
        @ [ "  }"; "}" ]
      in
      if String.equal acc_typ d_typ then
        let ptr = Printf.sprintf "__mma_dp + __mi * %d * %d + __ni * %d" mma_tile ldd mma_tile in
        let blk = acc ^ "[__mi][__ni]" in
        nest
          [
            (match dir with
            | `Load ->
                Printf.sprintf "rocwmma::load_matrix_sync(%s, %s, %d, rocwmma::mem_row_major);" blk
                  ptr ldd
            | `Store ->
                Printf.sprintf "rocwmma::store_matrix_sync(%s, %s, %d, rocwmma::mem_row_major);" ptr
                  blk ldd);
          ]
      else
        let cell =
          Printf.sprintf "__mma_dp[(__mi * %d + (__rc >> 4)) * %d + __ni * %d + (__rc & 15)]"
            mma_tile ldd mma_tile
        in
        let elt = Printf.sprintf "%s[__mi][__ni].x[__ei]" acc in
        [
          "{ /* rocwmma converted d boundary: coordinates from ocannl_wmma_rc16 */";
          Printf.sprintf "  %s __mma_rc;" (mma_frag_typ "accumulator" acc_typ None);
          "  rocwmma::load_matrix_sync(__mma_rc, ocannl_wmma_rc16, 16, rocwmma::mem_row_major);";
        ]
        @ List.map
            ~f:(fun l -> "  " ^ l)
            (nest
               [
                 "for (int __ei = 0; __ei < (int)__mma_rc.num_elements; ++__ei) {";
                 "  const int __rc = (int)__mma_rc.x[__ei];";
                 (match dir with
                 | `Load -> Printf.sprintf "  %s = (%s)%s;" elt acc_typ cell
                 | `Store -> Printf.sprintf "  %s = (%s)%s;" cell d_typ elt);
                 "}";
               ])
        @ [ "}" ]

    (* DRAFT (tensorize-mma T3, HIP counterpart of the CUDA wmma draft): cooperative tile-MMA
       emission for [Low_level.Tile_mma] via rocWMMA -- ROCm's header-compatible analogue of
       nvcuda::wmma, so the fragment/load/mma/store shape mirrors cuda_backend.ml almost verbatim.
       The extent-32 lane loop binds threadIdx.x, so the 32 consecutive .x threads reaching the
       statement form the cooperating RDNA wavefront (wave32); 16x16x16 fragment blocks stay
       resident across the whole [k] extent. Supported combinations on RDNA3 / RDNA3.5 WMMA: f16 x
       f16 -> f32 (flagship), f16 x f16 -> f16, bf16 x bf16 -> f32, bf16 x bf16 -> bf16. RDNA WMMA
       has no f32-input (tf32-like) shape, so uniform f32 stays on the scalar path -- unlike Metal
       [simdgroup_matrix], which does f32. Declines (the barrier-bracketed lane-0 fallback renders
       instead) on: other precision combinations, extents not multiples of 16, leading dimensions
       violating the 16-element-tile stride constraint, and thread-space operands (per-thread stacks
       are not a jointly-owned tile). Also declines (via [mma_supported] in the guard below) when
       the target is not RDNA3+/wave32 or rocWMMA headers are absent: a manual [Sched.tensorize]
       reaches this hook even where [hardware_limits.mma] is [None], so the capability check cannot
       live in [hardware_limits] alone. Verified on gfx1151 (Radeon 8060S, RDNA3.5) under hiprtc via
       schedule_mma_matmul: the f16 -> f16 combination compiles and executes and matches the serial
       twin bitwise (on same-sign products only: gfx11's WMMA instruction itself leaves a residue of
       about 2^-24 where products cancel, so a signed exact-zero cell comes back as a subnormal,
       gh-ocannl-1420, [tools/hip_wmma_sign_probe.hip]); the bf16 and f16 -> f32 combinations take
       the same rocWMMA template path, differing only in fragment element type. rocWMMA
       (header-only) is cloned under %LOCALAPPDATA%/rocwmma since it is not in the ROCm 7.1 Windows
       SDK. [hip_to_code] injects the header and -std=c++17 only when a kernel actually uses it, so
       non-tensor-core kernels are unaffected and do not require rocWMMA to be present. *)
    let mma_syntax =
      Some
        (fun ~d_prec
          ~a_prec
          ~b_prec
          ~ta
          ~tb
          ~m
          ~n
          ~k
          ~d:(d_ptr, ldd, d_space, d_layout)
          ~a:(lda, a_space, a_layout)
          ~b:(ldb, b_space, b_layout)
        ->
          let tile = mma_tile in
          let combo = mma_combo ~a_prec ~b_prec ~d_prec in
          (* Asked only where the triple has an arm, as the device gate always was. *)
          let accepted () =
            rocwmma_accepts ~a_prec ~b_prec ~d_prec ~m ~n ~k ~d:(rocwmma_d ~ldd d_space d_layout)
              ~a:(lda, a_space, a_layout) ~b:(ldb, b_space, b_layout)
          in
          match d_space with
          | `Fragment fragment -> (
              let (* gh-ocannl-480 (cross-[k_o] accumulator residency): the accumulator fragment
                     array [fragment] was declared and loaded once by [mma_fragment_syntax]. Here
                     each [Tile_mma] at a serial [k_o] emits update-only mma steps into it -- no
                     per-[k_o] load or store of [d]. The trailing barrier keeps the next k-block's
                     cooperative staging from overwriting the shared tiles still being read. The
                     acceptance guard matches [mma_fragment_syntax]'s (both see the same
                     [lda]/[ldb]), so whenever the fragment scope accepts this branch does too. *)
                open
                PPrint
              in
              match combo with
              | Some (ab_typ, _acc_typ, _d_typ) when accepted () ->
                  let ptr_decl name typ ptr =
                    string (Printf.sprintf "%s *%s = reinterpret_cast<%s *>(" typ name typ)
                    ^^ ptr ^^ string ");"
                  in
                  let barrier = "__syncthreads();" in
                  let body_lines =
                    Hip_mma.k_loop_lines ~ab_typ ~acc:fragment ~ta ~tb ~lda ~ldb ~m ~n ~k
                    @ [ barrier ]
                  in
                  let body ~a_ptr ~b_ptr =
                    ptr_decl "__mma_ap" ("const " ^ ab_typ) a_ptr
                    ^^ hardline
                    ^^ ptr_decl "__mma_bp" ("const " ^ ab_typ) b_ptr
                    ^^ hardline
                    ^^ separate_map hardline string body_lines
                  in
                  Some
                    (fun ~a_ptr ~b_ptr ->
                      group
                        (string
                           (Printf.sprintf "{ /* tile_mma fragment update %dx%dx%d (rocwmma) */" m n
                              k)
                        ^^ nest 2 (hardline ^^ body ~a_ptr ~b_ptr)
                        ^^ hardline ^^ rbrace))
              | _ -> None)
          | `Device | `Shared | `Thread -> (
              match combo with
              | Some (ab_typ, acc_typ, d_typ) when accepted () ->
                  let open PPrint in
                  let mt = m / tile and nt = n / tile in
                  let frag = mma_frag_typ in
                  (* [reinterpret_cast] bridges the node's C element type to the rocWMMA fragment
                     type. *)
                  let ptr_decl name typ ptr =
                    string (Printf.sprintf "%s *%s = reinterpret_cast<%s *>(" typ name typ)
                    ^^ ptr ^^ string ");"
                  in
                  let barrier = "__syncthreads();" in
                  let body_lines =
                    [
                      barrier;
                      Printf.sprintf "%s __mma_acc[%d][%d];" (frag "accumulator" acc_typ None) mt nt;
                    ]
                    @ mma_d_boundary_lines ~dir:`Load ~acc_typ ~d_typ ~acc:"__mma_acc" ~ldd ~mt ~nt
                    @ Hip_mma.k_loop_lines ~ab_typ ~acc:"__mma_acc" ~ta ~tb ~lda ~ldb ~m ~n ~k
                    @ mma_d_boundary_lines ~dir:`Store ~acc_typ ~d_typ ~acc:"__mma_acc" ~ldd ~mt ~nt
                    @ [ barrier ]
                  in
                  let body ~a_ptr ~b_ptr =
                    ptr_decl "__mma_dp" d_typ d_ptr ^^ hardline
                    ^^ ptr_decl "__mma_ap" ("const " ^ ab_typ) a_ptr
                    ^^ hardline
                    ^^ ptr_decl "__mma_bp" ("const " ^ ab_typ) b_ptr
                    ^^ hardline
                    ^^ separate_map hardline string body_lines
                  in
                  Some
                    (fun ~a_ptr ~b_ptr ->
                      group
                        (string (Printf.sprintf "{ /* tile_mma %dx%dx%d (rocwmma) */" m n k)
                        ^^ nest 2 (hardline ^^ body ~a_ptr ~b_ptr)
                        ^^ hardline ^^ rbrace))
              | _ -> None))

    (* Cross-[k_o] accumulator residency (gh-ocannl-480): the marked local accumulator tile becomes
       a persistent rocWMMA accumulator-fragment array whose load/store bracket the whole serial
       reduction. Loaded once from [target] before the [k_o] loop, updated in place by the nested
       [Tile_mma]s (which see [`Fragment fragment] and take the update-only branch of [mma_syntax]),
       stored once after. Mirrors the Metal [simdgroup_matrix] rendering; the guard matches the
       [`Fragment] branch of [mma_syntax] so both accept together. *)
    let mma_fragment_syntax =
      Some
        (fun ~d_prec
          ~a_prec
          ~b_prec
          ~m
          ~n
          ~k
          ~fragment
          ~target:(d_ptr, ldd, d_space, d_layout)
          ~a:(lda, a_space, a_layout)
          ~b:(ldb, b_space, b_layout)
          ~body
        ->
          let tile = mma_tile in
          match mma_combo ~a_prec ~b_prec ~d_prec with
          | Some (_ab_typ, acc_typ, d_typ)
            when (match d_space with `Fragment _ -> false | `Device | `Shared | `Thread -> true)
                 && rocwmma_accepts ~a_prec ~b_prec ~d_prec ~m ~n ~k
                      ~d:(rocwmma_d ~ldd d_space d_layout) ~a:(lda, a_space, a_layout)
                      ~b:(ldb, b_space, b_layout) ->
              let open PPrint in
              let mt = m / tile and nt = n / tile in
              let frag = mma_frag_typ in
              let ptr_decl name typ ptr =
                string (Printf.sprintf "%s *%s = reinterpret_cast<%s *>(" typ name typ)
                ^^ ptr ^^ string ");"
              in
              let barrier = "__syncthreads();" in
              let lines_before =
                [
                  barrier;
                  Printf.sprintf "%s %s[%d][%d];" (frag "accumulator" acc_typ None) fragment mt nt;
                ]
                @ mma_d_boundary_lines ~dir:`Load ~acc_typ ~d_typ ~acc:fragment ~ldd ~mt ~nt
                @ [ "/* rocwmma fragment reduction body begins */" ]
              in
              let lines_after =
                "/* rocwmma fragment reduction body ends */"
                :: mma_d_boundary_lines ~dir:`Store ~acc_typ ~d_typ ~acc:fragment ~ldd ~mt ~nt
                @ [ barrier ]
              in
              let d_decl = ptr_decl "__mma_dp" d_typ d_ptr in
              Some
                (group
                   (string (Printf.sprintf "{ /* rocwmma fragment %dx%d across k_o */" m n)
                   ^^ nest 2
                        (hardline ^^ d_decl ^^ hardline
                        ^^ separate_map hardline string lines_before
                        ^^ hardline ^^ body () ^^ hardline
                        ^^ separate_map hardline string lines_after)
                   ^^ hardline ^^ rbrace))
          | _ -> None)

    (* THE float-to-fp8 narrowing this backend emits — the three operator bridges below, which
       compute in float and narrow the result, and [convert_precision]'s conversions. It is ALWAYS
       the guarded helper, never the platform's own cast: ROCm's software float-to-e5m2 conversion
       shifts out of range for magnitudes around 4e-25 to 3.3e-24 and returns codes as large as
       2^-14 where the answer is a signed zero (gh-ocannl-647, filed upstream as
       https://github.com/ROCm/rocm-systems/issues/10591 — that fix landing is what retires this
       helper). The guard pre-rounds only that range, which rounds to zero anyway, so it is exact
       everywhere, and this backend's fp8 output matches CUDA, cc, Metal and the host codec on every
       input.

       One funnel rather than four call sites, because the first version of this guarded
       [convert_precision] alone and left fp8 ARITHMETIC results narrowing through bare casts — an
       fp8 [**.] whose f32 result lands in ROCm's broken window still produced a spurious value
       (Codex P2 on PR #372). A fifth narrowing site cannot repeat that without going through here.

       Which spelling narrows a value of precision [from]: the helpers are per-source-width on
       purpose, since a double handed to the float helper would narrow at the call and round twice
       (gh-ocannl-648), which the platform's own cast does not do. *)
    let fp8_from_prec_fn = Cuda_like_config.Hip.fp8_from_prec_fn

    let convert_precision ~from ~to_ =
      match (from, to_) with
      | Ops.Double_prec _, Ops.Double_prec _
      | Single_prec _, Single_prec _
      | Half_prec _, Half_prec _
      | Byte_prec _, Byte_prec _
      | Uint16_prec _, Uint16_prec _
      | Int32_prec _, Int32_prec _
      | Int64_prec _, Int64_prec _
      | Uint4x32_prec _, Uint4x32_prec _
      | Bfloat16_prec _, Bfloat16_prec _
      | Fp8_prec _, Fp8_prec _
      | Void_prec, Void_prec ->
          ("", "")
      (* hip_fp16.h has no [__double2half]; route through float. *)
      | Double_prec _, Half_prec _ -> ("__float2half((float)(", "))")
      | Single_prec _, Half_prec _ -> ("__float2half(", ")")
      | Byte_prec _, Half_prec _ -> ("__ushort2half_rn((unsigned short int)", ")")
      | Double_prec _, Uint4x32_prec _ -> ("double_to_uint4x32(", ")")
      | Single_prec _, Uint4x32_prec _ -> ("single_to_uint4x32(", ")")
      | Uint4x32_prec _, _ -> ("", ".v[0]")
      | Byte_prec _, Uint4x32_prec _ -> ("byte_to_uint4x32(", ")")
      | Uint16_prec _, Uint4x32_prec _ -> ("uint16_to_uint4x32(", ")")
      | Bfloat16_prec _, Uint4x32_prec _ -> ("bfloat16_to_uint4x32(", ")")
      | Half_prec _, Uint4x32_prec _ -> ("half_to_uint4x32(", ")")
      | Fp8_prec _, Uint4x32_prec _ -> ("fp8_to_uint4x32(", ")")
      (* The integer counter conversions MUST call the builtins, which spread the bits across all
         four uint4x32 lanes (golden-ratio / MMIX / rotation mixing). The raw struct literal below
         only fills lane 0, leaving lanes 1-3 zero; with the 2-round light threefry used for
         parameter init that produces near-identical outputs for consecutive counters (periodicity),
         so random inits diverge from CC/Metal. [Ops.index_prec] is signed [int32] (or [int64] under
         [large_models]), so the signed arms are the conversion hit by every PRNG init loop, e.g.
         centered [uniform1] parameter initialization (task-04f97340). *)
      | Int32_prec _, Uint4x32_prec _ -> ("int32_to_uint4x32(", ")")
      | Int64_prec _, Uint4x32_prec _ -> ("int64_to_uint4x32(", ")")
      | Uint32_prec _, Uint4x32_prec _ -> ("uint32_to_uint4x32(", ")")
      | Uint64_prec _, Uint4x32_prec _ -> ("uint64_to_uint4x32(", ")")
      | _, Uint4x32_prec _ -> ("{(unsigned int)(", "), 0, 0, 0}")
      (* [__hip_bfloat16] has constructors from float, double, short, unsigned short, ... and
         [__half] has several conversion operators, so C-style casts between them are ambiguous
         (observed with ROCm 7.1 hiprtc); route through float explicitly. Same precaution for the
         half <-> fp8 pairs. *)
      | Half_prec _, Bfloat16_prec _ -> ("__float2bfloat16(__half2float(", "))")
      | Bfloat16_prec _, Half_prec _ -> ("__float2half(__bfloat162float(", "))")
      | _, Bfloat16_prec _ -> ("__float2bfloat16((float)(", "))")
      | Bfloat16_prec _, _ -> ("(" ^ typ_of_prec to_ ^ ")(__bfloat162float(", "))")
      (* Through the same funnel as the operator bridges. A helper and not a ternary wrapped around
         the operand: a [convert_precision] pair brackets ONE occurrence of its expression, and a
         ternary would name it twice. With the guard off both arms are spelled exactly as before. *)
      | _, Fp8_prec _ -> (
          let fn = fp8_from_prec_fn from in
          match from with Ops.Half_prec _ -> (fn ^ "(__half2float(", "))") | _ -> (fn ^ "(", ")"))
      | Fp8_prec _, Half_prec _ -> ("__float2half((float)(", "))")
      | ( Fp8_prec _,
          (Byte_prec _ | Uint16_prec _ | Int32_prec _ | Uint32_prec _ | Int64_prec _ | Uint64_prec _)
        ) ->
          (* __hip_fp8_e5m2's integer conversion operators saturate (wrong for negative values into
             unsigned types) and overlap enough to make direct casts ambiguity-prone; convert via
             float, like the CC backend. *)
          ("(" ^ typ_of_prec to_ ^ ")((float)(", "))")
      | _ -> ("(" ^ typ_of_prec to_ ^ ")(", ")")
  end

  let codegen_capabilities () =
    let module Config = Hip_syntax_config (struct
      let procs = [||]
    end) in
    C_syntax.codegen_capabilities (module Config)

  module Compile = C_syntax.Compile_driver (Hip_syntax_config)

  let next_program_id = Atomic.make 0
  let new_program code = { program_id = Atomic.fetch_and_add next_program_id 1; code }

  (* {2 The kernel code cache (gh-ocannl-1314)}

     hiprtc's cost is mostly per PROGRAM, and it compiles one program at a time per process:
     measured on the dumped 261-kernel source of a [gpt2_mini_train] step, one joint program took
     5.3 s on gfx1102 (3.0 s on gfx1151), the same kernels as one program each 87.7 s (37.1 s), and
     threads changed neither, however many. A search candidate re-renders the whole routine and
     typically changes a handful of its kernels, so the batch path compiles only the kernels no
     earlier compile in this process produced -- its cache misses -- as ONE program, and reuses the
     rest.

     A kernel's key is its canonical standalone source ([render_batch]'s [key_text i]: the includes
     and the builtins it uses verbatim, its definition with the names it owns -- loop and
     static-index symbols, scope locals, its nodes' identifiers, the emitter's derivations of them
     -- renamed by first occurrence and its segment index dropped,
     [Kernel_code_cache.canonical_kernel], gh-ocannl-1525), and the complete hiprtc option list its
     standalone source compiles under ([hiprtc_options]: the device set's target architectures, the
     include paths, the numerics flags, the debug flag). Misses with different option lists compile
     as separate programs, so a kernel's code always came from its own option list; misses with
     equal keys compile once ([Kernel_code_cache.resolve]). The reused code was compiled from the
     same kernel text up to that renaming -- a bijection on names the kernel binds, which an emitted
     kernel's meaning and its machine code do not depend on -- possibly beside other kernels and
     possibly under another name: the builtins are [__forceinline__] helpers and read-only
     constants, so a companion changes nothing it computes, and the function is looked up by the
     symbol the cache stored with it ([code.symbol]), while diagnostics keep the kernel's current
     name. Process-local and bounded by [hip_kernel_cache_mib]; 0 disables it, and every batch
     compiles as one program of all its kernels, as before. Parameters, bindings and launch geometry
     always come from the current rendering, and every kernel still goes through [link_proc]'s
     checks. *)
  let kernel_cache_capacity_bytes =
    lazy
      (let mib = Utils.get_global_arg ~default:"256" ~arg_name:"hip_kernel_cache_mib" in
       match Int.of_string_opt (String.strip mib) with
       | Some m when m >= 0 -> m * 1024 * 1024
       | _ ->
           raise
           @@ Utils.User_error
                ("hip_kernel_cache_mib must be a non-negative integer (MiB); found: " ^ mib))

  let kernel_cache : (program * string) Kernel_code_cache.t =
    Kernel_code_cache.create ~capacity_bytes:(fun () -> Lazy.force kernel_cache_capacity_bytes)

  (* Held from the cache lookups to the publication of what they missed, so two domains compiling
     the same kernels cannot both miss and both compile them: the second finds the first's code.
     hiprtc compiles one program at a time per process anyway, so serializing here costs nothing it
     was not already paying. *)
  let kernel_compile_lock = Stdlib.Mutex.create ()

  let compile_kernels ~batch ~names bindings lowereds =
    Compile_cost.record_reuse_capable ();
    let r =
      Compile.render_batch ~names bindings lowereds ~includes:hip_includes
        ~builtins:Builtins_hip.builtins
        ~conditional_includes:Cuda_like_config.Hip.conditional_includes ()
    in
    let n = Array.length r.entries in
    Compile_cost.record_kernels ~rendered:n ~compiled:0;
    let all = List.init n ~f:Fn.id in
    let kernel_names = Array.map r.entries ~f:(fun (_, name, _) -> name) in
    (* The artifacts' name, as [Compile_driver.drive] derives it. *)
    let name =
      if batch then
        String.(strip ~drop:(equal_char '_') @@ common_prefix (Array.to_list kernel_names))
      else kernel_names.(0)
    in
    Lazy.force ensure_initialized;
    let target_archs = hiprtc_target_archs () in
    (* The routine's full source, as codegen emitted it, is its [.hip] artifact whatever was
       compiled: the debug view of what codegen emitted (and what tests read) does not depend on
       what the cache held. The renaming is the key's alone; hiprtc compiles the emitted text. *)
    let full_source = r.assemble all in
    write_source_artifact ~name (with_conditional_includes full_source);
    (* Not under device debugging: a code object's debug info then names the source file it was
       compiled from, which a later compile of the same routine rewrites, so reuse would point a
       debugger at text that is not the code's. Every kernel compiles, as with the cache off. *)
    let enabled = Kernel_code_cache.enabled kernel_cache && not (Utils.with_runtime_debug ()) in
    (* Clears what an earlier compile of this name left, which is not this compile's; with
       [log_header], writes the [.hip_log] an all-hit compile has no hiprtc call to write. *)
    let clear_artifacts ?log_header () =
      if Utils.settings.output_debug_files_in_build_directory then (
        let dir = Stdlib.Filename.dirname (Utils.build_file (name ^ ".hsaco")) in
        let ours file =
          List.exists [ ".hsaco"; ".hip_log" ] ~f:(fun ext -> String.equal file (name ^ ext))
          || String.is_prefix file ~prefix:(name ^ ".part")
        in
        (try
           Array.iter (Stdlib.Sys.readdir dir) ~f:(fun file ->
               if ours file then
                 try Stdlib.Sys.remove (Stdlib.Filename.concat dir file) with Sys_error _ -> ())
         with Sys_error _ -> ());
        Option.iter log_header ~f:(fun data ->
            Stdio.Out_channel.write_all (Utils.build_file (name ^ ".hip_log")) ~data))
    in
    let log_header n_compiled =
      Printf.sprintf "/* kernel code cache: compiled %d of %d kernels, reused %d */\n" n_compiled n
        (n - n_compiled)
    in
    (* One program of [idxs]. A program of every kernel in batch order is the routine's own source,
       already written as [<name>.hip]; any other program is written as [<name>.part<k>.hip] beside
       its code object and log, under the name hiprtc was given, so its diagnostics and debug info
       point at the text it compiled. [.part<k>] is a spelling no routine name takes. *)
    let compile_group ~log_header ~group ~n_groups options idxs =
      let whole = n_groups = 1 && List.equal Int.equal idxs all in
      let artifact = if whole then name else name ^ ".part" ^ Int.to_string group in
      let source = with_conditional_includes (r.assemble idxs) in
      if not whole then write_source_artifact ~name:artifact source;
      (* Before the call: a program hiprtc refuses was still given these kernels. *)
      Compile_cost.record_kernels ~rendered:0 ~compiled:(List.length idxs);
      let code =
        try
          Compile_cost.time_device (fun () ->
              compile_program ~name:artifact ~log_header ~options source)
        with exn ->
          let backtrace = Stdlib.Printexc.get_raw_backtrace () in
          r.attribute idxs exn backtrace
      in
      (new_program code, code.Hiprtc.code_length)
    in
    if not enabled then
      (* As before the cache: one program of every kernel, each looked up by its own name. *)
      let log_header = log_header n in
      Stdlib.Mutex.protect kernel_compile_lock (fun () ->
          clear_artifacts ();
          let program, _ =
            compile_group ~log_header ~group:0 ~n_groups:1
              (hiprtc_options ~target_archs full_source)
              all
          in
          (Array.map kernel_names ~f:(fun symbol -> (program, symbol)), r.entries))
    else
      (* Each kernel keyed by its canonical standalone source ([render_batch]'s [key_text]: the
         names it owns renamed, its segment index dropped, gh-ocannl-1525) plus the complete hiprtc
         option list its standalone source compiles under. *)
      let options = Array.init n ~f:(fun i -> hiprtc_options ~target_archs (r.assemble [ i ])) in
      let keys =
        Array.init n ~f:(fun i -> Kernel_code_cache.key ~parts:(options.(i) @ [ r.key_text i ]))
      in
      let header = ref "" in
      let resolved =
        Kernel_code_cache.resolve kernel_cache ~lock:kernel_compile_lock ~keys ~options
          ~symbols:kernel_names ~always_lock:Utils.settings.output_debug_files_in_build_directory
          ~before_compile:(fun ~n_compiled ~n_groups ->
            header := log_header n_compiled;
            clear_artifacts ?log_header:(if n_groups = 0 then Some !header else None) ())
          ~compile_group:(fun ~group ~n_groups options idxs ->
            compile_group ~log_header:!header ~group ~n_groups options idxs)
          ()
      in
      (resolved, r.entries)

  let%diagn2_sexp compile ~name bindings lowered =
    let programs, entries = compile_kernels ~batch:false ~names:[| name |] bindings [| lowered |] in
    let kparams, name, launch = entries.(0) in
    let code, symbol = programs.(0) in
    { code; symbol; kparams; bindings; name; launch }

  let%diagn2_sexp compile_batch ~names bindings lowereds =
    let programs, kparams_and_names = compile_kernels ~batch:true ~names bindings lowereds in
    { programs; kparams_and_names; bindings }

  (* {2 Post-link scratch validation (gh-ocannl-533)}

     A kernel's private (scratch) segment is sized by the compiler and budgeted by the runtime only
     at dispatch. When the dispatch asks for more scratch than the device can back, ROCm aborts the
     QUEUE -- "[UpdateScratch] scratch_size overflow!" / [HSA_STATUS_ERROR_INVALID_ARGUMENT] -- and
     what reaches OCaml out of synchronize is a bare [hipErrorInvalidValue], the same code an
     uninitialized input yields. There is nothing to classify on, and the stream is already dead:
     gh-ocannl-533 saw one autotune candidate take the whole benchmark process down with it.

     So this is prediction, not recovery (see docs/proposals/gh-ocannl-536.md): read the linked
     kernel's private segment size and decline an over-budget kernel BEFORE it is ever launched, as
     a typed [Resource_exceeded Thread_scratch]. To a tuner candidate that is an ordinary decline
     the blocker census tabulates; a hand-written schedule gets the usual [Utils.User_error]
     rendering at the public [Context.compile] boundary.

     The budget model, established experimentally on gfx1151 (Radeon 8060S, ROCm 7.14, WSL2) -- see
     the gh-ocannl-533 writeup:

     - The rejection is a function of the per-work-item size ALONE. A kernel at 98320 B/work-item
     launches at 204800 work-items; one at 114704 B is rejected at a SINGLE work-item. So the
     runtime backs the worst-case fully-occupied device, not the requested grid, and the check needs
     no launch geometry. - The cutoff sits where [private_seg_size] rounded up to a 64-byte granule,
     times the device's maximum resident work-items ([max_threads_per_multiprocessor *
     multiprocessor_count]), crosses 4 GiB. Measured boundary: 104832 B accepted, 104848 B rejected;
     the model reproduces every one of the ~70 sampled points, including both sides of that 16-byte
     step. - The compiler separately refuses a stack frame over 262136 B, so it never emits a kernel
     far above this; #533's 163856 B is comfortably inside what compiles and outside what launches.

     The 4 GiB cap is not a value any HIP or HSA query exposes -- the abort comes from the WSL WDDM
     thunk ([wsl::thunk::ComputeQueue::UpdateScratch]) -- so it is a documented constant from this
     experiment, while the multiplier is genuinely queried. Where the model is unverified the right
     answer is silence, not a guess: [ocannl_hip_scratch_validation=false] disables the check
     entirely, and a device that reports no usable occupancy figures is never rejected. *)

  let hip_scratch_validation =
    lazy (Utils.get_global_flag ~default:true ~arg_name:"hip_scratch_validation")

  (* The compile-side half (gh-ocannl-1313): LLVM AMDGPU refuses a kernel whose stack frame exceeds
     this many bytes, and hiprtc reports it only after compiling the whole kernel, as an untyped
     [HIPRTC_ERROR_COMPILATION] — "stack frame size (360544) exceeds limit (262136)", measured on
     gfx1102 after 52 s; gh-ocannl-533 saw the same limit on gfx1151. Published as
     [hardware_limits.max_thread_private_bytes], so [Schedule.check_hardware_limits_classified]
     declines such a kernel from its declared [Local] scratch before codegen. A compiler limit, not
     a budget model like the post-link check below, so [hip_scratch_validation] does not lift it:
     past it there is no kernel to validate.

     What is compared is the DECLARED arrays, and the frame holds more than them: 92 bytes more on
     that kernel, and a kernel that spills registers puts its spill slots there too. So the
     published budget keeps a reserve below the compiler's limit, for the same reason
     [hip_scratch_budget]'s near-limit fixture sizes itself about 2 KB under it: a kernel whose
     arrays land just under 262136 bytes would otherwise pass the gate and still die in hiprtc,
     untyped (Codex P1 on PR #1115). A kernel whose non-array frame exceeds the reserve can still
     reach that failure; the reserve narrows the band, it cannot close it before codegen. *)
  let compiler_frame_limit_bytes = 262136
  let frame_reserve_bytes = 1024

  (* Total scratch the runtime backs = per-work-item size, rounded up to the allocation granule,
     times the device's maximum resident work-items. *)
  let scratch_granule_bytes = 64
  let scratch_total_cap_bytes = 4 * 1024 * 1024 * 1024

  let scratch_limit_per_work_item (attrs : H.Device.attributes) =
    let resident = attrs.max_threads_per_multiprocessor * attrs.multiprocessor_count in
    if resident <= 0 then None
    else
      (* Largest granule-aligned size whose full-occupancy total still fits the cap. *)
      let granules_per_work_item = scratch_total_cap_bytes / resident / scratch_granule_bytes in
      if granules_per_work_item <= 0 then None
      else Some (granules_per_work_item * scratch_granule_bytes)

  (* Memoized per ordinal, like [_hip_properties]: this runs on EVERY link, and re-entering the
     driver for static device properties once per routine is pure overhead — it showed up as
     contention with several HIP processes sharing one iGPU. *)
  let scratch_budget_of_device =
    let cache =
      lazy
        (Array.init (num_devices ()) ~f:(fun ordinal ->
             lazy
               (let attrs = H.Device.get_attributes (H.Device.get ~ordinal) in
                (attrs, scratch_limit_per_work_item attrs))))
    in
    fun (device : device) ->
      (* Discovery a link can force first: under the device-init deadline (gh-ocannl-1281). *)
      Utils.with_device_init_deadline ~backend_name:name (fun () ->
          Lazy.force (Lazy.force cache).(device.ordinal))

  let validate_scratch_budget ~(device : device) ~name func =
    if Lazy.force hip_scratch_validation then
      let attrs, limit = scratch_budget_of_device device in
      Option.iter limit ~f:(fun limit ->
          let requested =
            H.Module.get_function_attribute func H.Module.HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
          in
          let rounded =
            (requested + scratch_granule_bytes - 1) / scratch_granule_bytes * scratch_granule_bytes
          in
          if rounded > limit then
            raise
            @@ Schedule_outcome.Cause_at
                 ( Schedule_outcome.Backend_link,
                   Schedule_outcome.Resource_exceeded
                     {
                       resource = Schedule_outcome.Thread_scratch;
                       requested;
                       limit = Some limit;
                       detail =
                         [%string
                           "HIP: kernel %{name} needs %{requested#Int} bytes of private (scratch) \
                            memory per work-item, above the %{limit#Int} bytes this device can \
                            back at full occupancy (%{attrs.max_threads_per_multiprocessor#Int} \
                            work-items x %{attrs.multiprocessor_count#Int} CUs against a \
                            %{scratch_total_cap_bytes#Int}-byte scratch allocation). Launching it \
                            would abort the queue rather than fail cleanly (gh-ocannl-533)"];
                     } ))

  let link_proc ~prior_context ~name ~symbol ~(kparams : (string * kparam_source) list)
      ~(launch : Low_level.launch_dims) ~ctx_buffers lowered_bindings run_module =
    let func = H.Module.get_function run_module ~name:symbol in
    let device = prior_context.device in
    validate_scratch_budget ~device ~name func;
    let stream_name = get_name device in
    (* Pre-resolve slab bases to keep the owning [Deviceptr.t]'s alive for the lifetime of the task
       closure; region views ([Tensor_at]) are non-owning and must not outlive the slab base. *)
    let ctx_bases = Map.map ctx_buffers ~f:(Slab.resolve_pool device) in
    let%diagn3_sexp work () : unit =
      let log_id = Utils.get_global_run_id () in
      let log_id_prefix = Int.to_string log_id ^ ": " in
      [%log_result
        "Launching",
        name,
        "on",
        stream_name,
        (log_id : int),
        (kparams : (string * kparam_source) list)];
      let module S = H.Stream in
      let args : S.kernel_param list =
        (* TODO: should we prohibit or warn about local-only tensors that are in
           prior_context.ctx_buffers? *)
        List.map kparams ~f:(function
          | _name, Kparam_ptr tn ->
              let loc = Option.value_exn ~here:[%here] @@ Map.find ctx_buffers tn in
              let base = Map.find_exn ctx_bases tn in
              S.Tensor_at (H.Deviceptr.offset base ~bytes:loc.offset)
          | _name, Log_file_name -> S.Int log_id
          | _name, Merge_buffer ->
              let loc = Option.value_exn ~here:[%here] !(device.merge_buffer) in
              let base = Slab.resolve_pool device loc in
              S.Tensor_at (H.Deviceptr.offset base ~bytes:loc.offset)
          | _name, Static_idx s ->
              let i = Indexing.find_exn lowered_bindings s in
              (* Shared bind-time validation: negativity, range -- inclusive [0, range] for symbolic
                 extents (gh-490), strict [0, range) for indices -- and index width. *)
              Indexing.validate_bound_value ~width64:Utils.settings.large_models s !i;
              S.Int !i
          | _name, (Kparam_pool_slab _ | Kparam_pool_slots _) ->
              Backend_intf.unexpected_pooled_kparam ~backend:"Hip_backend")
      in
      set_ctx @@ ctx_of prior_context;
      [%log "launching the kernel"];
      (if Utils.debug_log_from_routines () then
         Utils.add_log_processor ~prefix:log_id_prefix @@ fun log_contents ->
         Utils.log_debug_routine_logs ~log_contents ~stream_name);
      (* Launch dimensions derived from hardware-annotated loops (axis-types proposal §4);
         all-Serial kernels launch 1x1x1, as before. Static [__shared__] declarations do not use the
         dynamic pool, so [shared_mem_bytes] stays 0. *)
      S.launch_kernel func ~grid_dim_x:launch.Low_level.grid.(0)
        ~grid_dim_y:launch.Low_level.grid.(1) ~grid_dim_z:launch.Low_level.grid.(2)
        ~block_dim_x:launch.Low_level.block.(0) ~block_dim_y:launch.Low_level.block.(1)
        ~block_dim_z:launch.Low_level.block.(2) ~shared_mem_bytes:0 device.runner args;
      [%log "kernel launched"]
    in
    Task.Task
      {
        context_lifetime = (run_module, ctx_bases);
        description = "launches " ^ name ^ " on " ^ stream_name;
        work;
      }

  let%track3_sexp link prior_context (code : code) ctx_buffers =
    let ctx = ctx_of prior_context in
    set_ctx ctx;
    let run_module = H.Module.load_data_ex code.code.code (run_options ()) in
    prior_context.device.dev.set_builtins_in run_module;
    let idx_params = Indexing.bound_symbols code.bindings in
    let lowered_bindings : Indexing.lowered_bindings =
      List.map idx_params ~f:(fun s -> (s, Indexing.fresh_launch_param ()))
    in
    let task =
      link_proc ~prior_context ~name:code.name ~symbol:code.symbol ~kparams:code.kparams
        ~launch:code.launch ~ctx_buffers lowered_bindings run_module
    in
    (lowered_bindings, task)

  let%track3_sexp link_batch prior_context (code_batch : code_batch) ctx_buffers =
    let idx_params = Indexing.bound_symbols code_batch.bindings in
    let lowered_bindings : Indexing.lowered_bindings =
      List.map idx_params ~f:(fun s -> (s, Indexing.fresh_launch_param ()))
    in
    let ctx = ctx_of prior_context in
    set_ctx ctx;
    (* Each distinct program is loaded once, its module shared by the kernels it holds; every task
       keeps its own kernel's module alive through its [context_lifetime], as before. *)
    let modules = Hashtbl.create (module Int) in
    let module_of (program : program) =
      Hashtbl.find_or_add modules program.program_id ~default:(fun () ->
          let run_module = H.Module.load_data_ex program.code (run_options ()) in
          prior_context.device.dev.set_builtins_in run_module;
          run_module)
    in
    let procs =
      Array.mapi code_batch.kparams_and_names ~f:(fun i (kparams, name, launch) ->
          let program, symbol = code_batch.programs.(i) in
          link_proc ~prior_context ~name ~symbol ~kparams ~launch ~ctx_buffers lowered_bindings
            (module_of program))
    in
    (lowered_bindings, procs)

  (* One HIP graph for a fissioned routine's whole segment batch (gh-ocannl-488), mirroring the CUDA
     backend: stream-capture the segment launch loop once per distinct set of launch-time-varying
     arguments — static-index binding values, plus the merge-buffer position when the routine reads
     it — instantiate, and replay with a single hipGraphLaunch per step instead of one launch per
     segment. Baking kernel arguments into the graph is sound because context buffer bases are
     pre-resolved at link time (tnode pools are never reallocated in place while their routines are
     live; the merge pool, which can be, is part of the key by pointer identity) and every other
     varying argument is part of the key. Instantiated graphs are retained in a bounded FIFO cache,
     so a training loop cycling through batch-index bindings replays cached graphs from the second
     epoch on. Transparent fallback to per-segment launches: when kernel logging is on (the log id
     is a fresh kernel argument every run), when disabled via [gpu_graph_capture=false], or
     permanently for this routine if the runtime rejects capture. *)
  let sequence_segments (context : context) ~name ~(bindings : Indexing.lowered_bindings)
      ~uses_merge_buffer (tasks : Task.t list) : (Task.t * (unit -> unit)) option =
    let use_capture =
      List.length tasks > 1
      && Utils.get_global_flag ~default:true ~arg_name:"gpu_graph_capture"
      && not (Utils.debug_log_from_routines ())
    in
    if not use_capture then None
    else
      let device = context.device in
      let max_cached_graphs = 128 in
      let graphs =
        {
          graphs_device = device;
          cache = Hashtbl.create (module String);
          order = Queue.create ();
          lock = Stdlib.Mutex.create ();
        }
      in
      Backend_impl.Weak_registry.register graph_caches graphs;
      let cache = graphs.cache and order = graphs.order in
      let broken = ref false in
      let run_plain () = List.iter tasks ~f:Task.run in
      let current_key () =
        let idx = List.map bindings ~f:(fun (_, r) -> Int.to_string !r) in
        let merge =
          if not uses_merge_buffer then []
          else
            match !(device.merge_buffer) with
            | Some loc ->
                [ H.Deviceptr.string_of (Slab.resolve_pool device loc); Int.to_string loc.offset ]
            | None -> [ "no-merge" ]
        in
        String.concat ~sep:";" (idx @ merge)
      in
      let capture () =
        (* RELAXED, not THREAD_LOCAL: GC finalizers (module unloads, buffer frees of dead handles)
           can fire at any allocation point on the capturing thread, and stricter modes make the
           runtime reject such "potentially unsafe" calls mid-capture — with the exception then
           escaping [Gc.finalise] at an arbitrary program point. The finalizers only release dead
           handles, so they are genuinely safe to run concurrently with capture. *)
        H.Graph.begin_capture ~mode:H.Graph.RELAXED device.runner;
        let graph =
          try
            run_plain ();
            H.Graph.end_capture device.runner
          with exn ->
            (* Terminate the capture before propagating, else the stream stays in capture mode. *)
            (try H.Graph.destroy (H.Graph.end_capture device.runner) with _ -> ());
            raise exn
        in
        let exec = H.Graph.instantiate graph in
        H.Graph.destroy graph;
        exec
      in
      Some
        ( Task.Task
            {
              (* Holding [graphs] here keeps the registry's weak pointer alive exactly as long as
                 the routine's schedule. *)
              context_lifetime = (tasks, graphs);
              description = "graph-captured segments of " ^ name ^ " on " ^ get_name device;
              work =
                (fun () ->
                  if !broken then run_plain ()
                  else
                    Stdlib.Mutex.protect graphs.lock (fun () ->
                        set_ctx @@ ctx_of context;
                        let key = current_key () in
                        match Hashtbl.find cache key with
                        | Some exec -> H.Graph.launch exec device.runner
                        | None -> (
                            match capture () with
                            | exec ->
                                if Queue.length order >= max_cached_graphs then (
                                  let victim = Queue.dequeue_exn order in
                                  (* The evicted exec may still have a pending launch on the
                                     stream. *)
                                  H.Stream.synchronize device.runner;
                                  H.Graph.exec_destroy (Hashtbl.find_exn cache victim);
                                  Hashtbl.remove cache victim);
                                Hashtbl.set cache ~key ~data:exec;
                                Queue.enqueue order key;
                                H.Graph.launch exec device.runner
                            | exception H.Hip_error { status; message } ->
                                (* E.g. capture unsupported on this runtime: fall back to
                                   per-segment launches for this routine (same-stream FIFO supplies
                                   the segment ordering), and re-run outside capture so a genuine
                                   launch failure surfaces on the plain path. *)
                                broken := true;
                                Stdio.eprintf
                                  "ocannl: disabling HIP graph capture for routine %s (%s: %s)\n%!"
                                  name message
                                  (Sexp.to_string_hum @@ H.sexp_of_result status);
                                run_plain ())));
            },
          fun () -> destroy_graphs graphs )

  let get_global_debug_info () =
    Sexp.message "hip_global_debug"
      [
        ("live_streams", [%sexp_of: int] @@ H.Stream.get_total_live_streams ());
        ("live_graph_execs", [%sexp_of: int] @@ live_graph_execs ());
      ]

  let static_properties () =
    let device_properties =
      Array.init (num_devices ()) ~f:(fun ordinal ->
          let dev = H.Device.get ~ordinal in
          let attributes = H.Device.get_attributes dev in
          let props =
            [
              ("device_name", Sexp.Atom attributes.name);
              ("device_ordinal", [%sexp_of: int] ordinal);
              ("gcn_arch_name", Sexp.Atom attributes.gcn_arch_name);
              ("multiprocessor_count", [%sexp_of: int] attributes.multiprocessor_count);
              ("clock_rate", [%sexp_of: int] attributes.clock_rate);
              ("warp_size", [%sexp_of: int] attributes.warp_size);
              ("async_engine_count", [%sexp_of: int] attributes.async_engine_count);
              ("compute_capability_major", [%sexp_of: int] attributes.compute_capability_major);
              ("compute_capability_minor", [%sexp_of: int] attributes.compute_capability_minor);
              ("max_threads_per_block", [%sexp_of: int] attributes.max_threads_per_block);
              (* The launch-dimension limits the schedule layer gates against: [max_grid_size] feeds
                 [hardware_limits.max_grid_yz], and [max_threads_dim] bounds a workgroup
                 per-dimension (beyond the [max_threads_per_block] product). Surfaced so a run on
                 hardware can read back what the gates compare against -- otherwise the only
                 evidence a query is not degenerate is that nothing got rejected. *)
              ("max_grid_size", [%sexp_of: int * int * int] attributes.max_grid_size);
              ("max_threads_dim", [%sexp_of: int * int * int] attributes.max_threads_dim);
              ("unified_addressing", [%sexp_of: bool] attributes.unified_addressing);
              (* The memory model, as [hardware_limits] reads it: an integrated part shares host
                 DRAM (minix's gfx1151), a discrete one has its own (tuf's gfx1102) --
                 [unified_addressing] reads false on both. gh1133_cells.sh's HIP kprof step states
                 it from here (gh-ocannl-1392). *)
              ("integrated", [%sexp_of: bool] attributes.integrated);
              (* The DEVICE half of [mma_supported], read through the predicate the backend gates
                 on, so a verifier can tell a device that can never tensorize from a host missing
                 the rocWMMA headers -- [hardware_limits.mma] reports only the conjunction.
                 tools/machine-verify.sh pairs it with its own header probe to assert the capability
                 where both halves hold (gh-ocannl-1070). *)
              ("tile_mma_eligible", [%sexp_of: bool] (tile_mma_eligible attributes));
            ]
          in
          Sexp.message "device" props)
    in
    Sexp.List (Sexp.Atom "hip_devices" :: Array.to_list device_properties)

  (* Conservative per-workgroup device limits for the schedule layer (schedule-ir-optops §6):
     minimum across devices, so code compiled once is valid wherever it links. *)
  (* Memoized behind [lazy]: driver init and device enumeration must not run at backend-module
     initialization ([num_devices] forces [ensure_initialized]). *)
  (* Concrete static capabilities separate mixed devices without changing the conservative
     construction limits. Driver/runtime/header provenance remains a documented separate concern. *)
  let timing_identity (device : device) =
    try
      let attributes = H.Device.get_attributes device.dev.dev in
      Some
        {
          Backend_intf.device_signature =
            Sexp.to_string
              (Sexp.message "device_capabilities"
                 [
                   ("name", Sexp.Atom attributes.name);
                   ("gcn_arch_name", Sexp.Atom attributes.gcn_arch_name);
                   ("multiprocessor_count", [%sexp_of: int] attributes.multiprocessor_count);
                   ("clock_rate", [%sexp_of: int] attributes.clock_rate);
                   ("memory_clock_rate", [%sexp_of: int] attributes.memory_clock_rate);
                   ("memory_bus_width", [%sexp_of: int] attributes.memory_bus_width);
                   ("total_global_mem", [%sexp_of: int] attributes.total_global_mem);
                   ("l2_cache_size", [%sexp_of: int] attributes.l2_cache_size);
                   ("max_threads_per_block", [%sexp_of: int] attributes.max_threads_per_block);
                   ( "max_threads_per_multiprocessor",
                     [%sexp_of: int] attributes.max_threads_per_multiprocessor );
                   ("shared_mem_per_block", [%sexp_of: int] attributes.shared_mem_per_block);
                   ( "shared_mem_per_multiprocessor",
                     [%sexp_of: int] attributes.shared_mem_per_multiprocessor );
                   ("regs_per_block", [%sexp_of: int] attributes.regs_per_block);
                   ("warp_size", [%sexp_of: int] attributes.warp_size);
                 ]);
          toolchain_signature =
            (try
               let major, minor = Hiprtc.version () in
               Some
                 (Sexp.to_string
                    (Sexp.message "hip_runtime_hiprtc"
                       [
                         ("runtime", [%sexp_of: int] (H.runtime_get_version ()));
                         ("hiprtc", [%sexp_of: int * int] (major, minor));
                       ]))
             with H.Hip_error _ | Hiprtc.Hiprtc_error _ -> None);
        }
    with H.Hip_error _ -> None

  let hardware_limits =
    let limits =
      lazy
        (let attrs =
           Array.init (num_devices ()) ~f:(fun ordinal ->
               H.Device.get_attributes (H.Device.get ~ordinal))
         in
         let min_over f = Array.map attrs ~f |> Array.min_elt ~compare:Int.compare in
         let all_integrated =
           (not (Array.is_empty attrs))
           && Array.for_all attrs ~f:(fun (a : H.Device.attributes) -> a.integrated)
         in
         {
           Backend_intf.max_threads_per_workgroup =
             min_over (fun (a : H.Device.attributes) -> a.max_threads_per_block);
           max_workgroup_memory_bytes =
             min_over (fun (a : H.Device.attributes) -> a.shared_mem_per_block);
           max_thread_private_bytes = Some (compiler_frame_limit_bytes - frame_reserve_bytes);
           (* Per-dimension workgroup caps (gh-ocannl-679), from the same queried [max_threads_dim]
              the dump above surfaces. On the AMD parts seen so far it reads (1024, 1024, 1024) --
              equal to [max_threads_per_block], so on those devices every per-dimension violation is
              also a product violation and this row never fires alone. It is CUDA, whose [.z] is 64,
              that the row exists for; filling it here keeps the two backends' gates identical
              rather than making the caller ask which backend it is on. *)
           max_workgroup_dims =
             (match
                ( min_over (fun (a : H.Device.attributes) ->
                      let x, _, _ = a.max_threads_dim in
                      x),
                  min_over (fun (a : H.Device.attributes) ->
                      let _, y, _ = a.max_threads_dim in
                      y),
                  min_over (fun (a : H.Device.attributes) ->
                      let _, _, z = a.max_threads_dim in
                      z) )
              with
             | Some x, Some y, Some z -> Some (x, y, z)
             | _ -> None);
           (* One cap for both gated dimensions (see [Backend_intf.max_grid_yz]): the smaller of the
              queried .y and .z components, so the gate is never looser than the device on either.
              On the AMD devices seen so far they coincide. *)
           max_bound_pools = None;
           max_grid_yz =
             min_over (fun (a : H.Device.attributes) ->
                 let _, y, z = a.max_grid_size in
                 Int.min y z);
           (* Cooperative tile-MMA via rocWMMA, gated on [mma_supported]: RDNA3/RDNA3.5+
              (gfx11/gfx12) wave32 across ALL devices AND discoverable rocWMMA headers. CDNA (gfx9,
              wave64, MFMA) and header-less hosts stay on the scalar path -- reporting [Some] there
              would let autotune pick [Tile_mma] and then fail to compile. [None] unless EVERY
              device qualifies: limits are min-over-devices, so code compiled once must be valid
              wherever it links. Precision combinations are decided per call by [mma_syntax]. *)
           mma = (if mma_supported () then Some Hip_mma.capability else None);
           simd_vector = Ir.Vector_file.none;
           native_fp16_arithmetic = false;
           simdgroup_width = Some 32;
           (* gh-ocannl-1124: measured false on gfx1151; see the field. *)
           lane_scalar_recompute_cheap = false;
           (* gh-ocannl-1168: unmeasured beyond one simdgroup; see the field. *)
           lane_all_reduce_simdgroups = 1;
           (* gh-ocannl-1184: clean cached confirmation loses with block 16 on both gfx1151 unified
              and gfx1102 discrete. Keep the two-pass rewrite under auto; other HIP targets are
              unmeasured. *)
           online_softmax_auto_block = 0;
           worker_pool_tag = None;
           (* Filled fresh by the accessor below, not here: both inputs are process-mutable
              ([Train.CDSL.enable_all_debugs] flips the debug settings at any point), and this
              record is memoized (Codex P1 on PR #337). *)
           codegen_tag = None;
           (* Advisory roofline envelope (gh-ocannl-491): documented rough constants for the targets
              this backend runs on. Flops: RDNA3-class dGPU/APU ~10 fp32 TFLOP/s — the model only
              ranks, so a class-typical number suffices. Bandwidth is a class CEILING per the
              [hardware_limits] contract (gh-ocannl-578): the previous 2.5e11 (Strix-Halo-class
              LPDDR5X) sat below what CDNA parts sustain — MI300-family HBM3/3e reaches ~5.3-6 TB/s.
              Per-device queries remain calibration follow-up work. *)
           peak_flops = Some 1.0e13;
           peak_memory_bandwidth = Some 6.0e12;
           (* gh-ocannl-1253: the ranking score's two tiers and residency, read only by the untimed
              choosers (Untimed, outside the schedule-cache key). Residency is queried --
              multiprocessors times resident threads per multiprocessor, minimum over devices. The
              capacity and the cache bandwidth are per class, from benchmarks/gh1253_cells.sh
              (2026-10-18, d818474a): on a discrete part (tuf, RX 7700S, gfx1102) the requested rate
              of the untiled weight gradient falls to DRAM between working sets of 1.5 and 3.0 GB
              across two runs, on a 1.3 TB/s cache plateau against 261 GB/s DRAM; on a unified one
              (minix, gfx1151) it never reaches DRAM up to the 10.8 GB swept, at 1.7 TB/s against
              235 GB/s -- so an integrated part has no capacity (no DRAM tier), rather than an
              invented one. Mixed machines take the discrete class. *)
           cache_capacity_bytes = (if all_integrated then None else Some (2 * 1024 * 1024 * 1024));
           cache_bandwidth = Some (if all_integrated then 1.7e12 else 1.3e12);
           (* gh-ocannl-1442: the ranking score's own rates, below the envelope's ceilings, per
              class from benchmarks/gh1253_cells.sh (2026-10-21, 46d84231): STREAM triad and the
              seeded register-blocktiled sketch's scalar operation rate (its 512 x 512 sweep) -- 235
              GB/s and 2.81e12 on the unified gfx1151 (minix), 261 GB/s and 1.93e12 on the discrete
              gfx1102 (tuf). They predicted the seeded weight gradient at b8 and b256 within
              1.14-1.35x on both; at b1, 0.69x on minix and 0.28x on tuf, whose small kernels sit on
              an erratic 0.1-0.4 ms floor the score does not price. Mixed machines take the discrete
              class. Override per machine with model_dram_bandwidth / model_issue_rate. *)
           issue_rate = Some (if all_integrated then 2.8e12 else 1.9e12);
           dram_bandwidth = Some (if all_integrated then 2.35e11 else 2.6e11);
           max_resident_threads =
             min_over (fun (a : H.Device.attributes) ->
                 a.multiprocessor_count * a.max_threads_per_multiprocessor);
           (* gh-ocannl-1479: the launch fill -- the multiprocessor count, queried, and the threads
              per multiprocessor at which scalar issue saturates, fitted by
              tools/calibrate_issue_rate.exe's launch sweep on the unified gfx1151 (minix, 20
              multiprocessors; 2026-10-25, ffeac5f8), whose 512 x 512 calibration kernel is full
              under it, so issue_rate above needs no normalization. The discrete class is unmeasured
              (no fill). Overridable by model_fill_threads. *)
           multiprocessors = min_over (fun (a : H.Device.attributes) -> a.multiprocessor_count);
           fill_threads = (if all_integrated then Some 384 else None);
         })
    in
    (* gh-ocannl-572: the kernel source is a function of the lowered code, the device capabilities
       and the numerics policy, all covered by the record itself — but two dispatch- and
       compilation-mechanics regimes are not. Graph capture fires only for multi-segment routines
       and only without routine logging, so it changes a fissioned candidate's launch overhead
       relative to a whole-routine one, and a crown from one regime is not evidence about the other;
       runtime debug switches this compiler to debug compilation, which no other backend's does.
       Both are the EFFECTIVE predicates and both are recomputed per call, since the settings behind
       them are mutable within a process (Codex P1/P2 on PR #337). *)
    let codegen_tag () =
      (if
         Utils.get_global_flag ~default:true ~arg_name:"gpu_graph_capture"
         && not (Utils.debug_log_from_routines ())
       then "graph-capture"
       else "no-graph-capture")
      (* No fp8 component here any more (gh-ocannl-647): the float-to-fp8 guard used to be
         configurable, so the two regimes needed distinct cache entries; it is now unconditional,
         and a constant contributes nothing to a tag. Restore a component here if the guard ever
         becomes conditional again — say on a ROCm version predicate, once upstream fixes it. *)
      ^ (if Utils.with_runtime_debug () then "/device-debug" else "/no-device-debug")
      (* Derive the cache regime from the same option selector the compiler uses. A pre-workaround
         crown is not evidence for the target-workaround regime. Limits describe this backend's
         device set, so include every device's policy. *)
      ^
      let target_archs = hiprtc_target_archs () in
      match Compiler_options.hip_target_options ~target_archs with
      | [] -> ""
      | options -> "/hiprtc-target:" ^ Compiler_options.render options
      (* No [bf16_accum_wide] component (gh-ocannl-1117): gh-ocannl-1051 added one by hand when
         [Bf16_auto] went wide here with the configured mode unchanged, but what a mode resolves to
         is now cache identity by derivation — [Schedule_cache.codegen_tag] tabulates this backend's
         [accum_prec] (which this predicate drives, as it does [mma_combo]) over every precision. *)
    in
    fun () -> { (Lazy.force limits) with Backend_intf.codegen_tag = Some (codegen_tag ()) }

  (* hiprtc's verdict on generated code, classified as CUDA's [classify_failure] classifies nvrtc's
     (ahrefs/ocannl#1387): [HIPRTC_ERROR_COMPILATION] means the compiler read our HIP C++ and
     rejected it, before anything was allocated or launched, so it is a typed decline of that kernel
     at the cc backend's ["compiler"] stage. Left to the common policy it was [Fatal] under
     [strict_failure_classification] for [Candidate] and [User_schedule] provenance alike, so one
     rejected kernel ended a tuner search, or the isolated per-kernel table partway (41 of 100
     kernels on cifar_resnet).

     This status only, for the reason CUDA gives: every other hiprtc status (an option it does not
     accept, an internal error) is the toolchain's and fails every candidate alike, so it must
     propagate rather than become a silent "nothing worked". So must a header hiprtc could not open
     ([hip/hip_fp16.h] without an SDK include path, an incomplete rocWMMA tree), which it reports
     under this very status: [C_syntax.missing_include] reads it off the log. The gh-ocannl-1313
     frame-size refusal also arrives as [HIPRTC_ERROR_COMPILATION] and so lands here too, inside the
     reserve band [compiler_frame_limit_bytes] leaves; it is reported as the compiler's rejection
     rather than as a [Resource_exceeded] with numbers parsed out of a log. HIP driver errors stay
     unclassified: no HIP fault has been characterized the way CUDA's are. *)
  let classify_failure phase exn =
    match exn with
    | Hiprtc.Hiprtc_error { status; message } -> (
        let status =
          match Hiprtc.sexp_of_result status with
          | Sexp.Atom atom -> atom
          | sexp -> Sexp.to_string sexp
        in
        let claim = C_syntax.codegen_bug_claim ~artifact:"the generated .hip file" in
        match status with
        | "HIPRTC_ERROR_COMPILATION" when not (C_syntax.missing_include message) ->
            Some
              {
                Schedule_outcome.phase;
                cause =
                  Schedule_outcome.Backend_rejected
                    {
                      backend = name;
                      stage = "compiler";
                      severity = Schedule_outcome.Compiler_bug;
                      detail =
                        [%string
                          "OCANNL hip backend: generated code failed to compile (%{status}).\n\
                           %{claim}\n\
                           hiprtc output:\n\
                           %{message}"];
                    };
                execution_effect = Schedule_outcome.No_device_writes;
              }
        | _ -> None)
    | _ -> None

  let get_debug_info (device : device) =
    let tot, unr, unf = H.Stream.total_unreleased_unfinished_delimited_events device.runner in
    let i2s = [%sexp_of: int] in
    Sexp.message "hip_stream_debug"
      [ ("total_events", i2s tot); ("unreleased_events", i2s unr); ("unfinished_events", i2s unf) ]
end
