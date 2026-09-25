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
    let ptr = H.Deviceptr.mem_alloc ~size_in_bytes in
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

(* The HIP SDK include dir (no-spaces junction on Windows / HIP_PATH / /opt/rocm), forward-slashed
   for the clang command line. [None] when no SDK is found (the Linux built-in-headers path). Lifted
   out of [Impl] for the same reason [Cuda_backend.cuda_include_options] was: it is a policy every
   hiprtc caller has to agree with, and tools/fp8_soak.ml is one -- a soak that guessed its own
   include path would report the HIP arm ready and then fail to compile its kernel wherever the SDK
   does not sit where the guess looked (Codex P2 on PR #463, for the CUDA side of the same
   program). *)
let hip_sdk_include_dir =
  lazy
    (let candidates =
       (match Sys.getenv "LOCALAPPDATA" with Some l -> [ l ^ "/hip_path_link" ] | None -> [])
       @ (match Sys.getenv "HIP_PATH" with Some p -> [ p ] | None -> [])
       @ [ "/opt/rocm" ]
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
     surfaces at [get_device], where [Context.auto]'s fallback can catch it. *)
  let () =
    if
      am_running_inline_tests
      && String.equal "hip"
           (String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:""))
    then try Lazy.force ensure_initialized with _ -> ()

  let num_devices () =
    Lazy.force ensure_initialized;
    H.Device.get_count ()

  (* [devices] is mutable to support plugging in new devices. *)
  let devices = lazy (ref @@ Array.create ~len:(num_devices ()) None)

  (* Bytes OCANNL has allocated on this device via [Slab], exact rather than the driver's
     granule-quantized [total - free] (gh-ocannl-289). Device-wide across contexts, matching the
     [Context.get_used_memory] contract. *)
  let get_used_memory (device : device) = Slab.used_memory device

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
  let all_rdna_wave32 =
    lazy
      (let n = num_devices () in
       n > 0
       && Array.for_all
            (Array.init n ~f:(fun ordinal -> H.Device.get_attributes (H.Device.get ~ordinal)))
            ~f:(fun (a : H.Device.attributes) ->
              (String.is_prefix a.gcn_arch_name ~prefix:"gfx11"
              || String.is_prefix a.gcn_arch_name ~prefix:"gfx12")
              && a.warp_size = 32))

  (* A directory containing a COMPLETE rocWMMA header tree, if any: [ROCWMMA_PATH] variants, a clone
     under [%LOCALAPPDATA%/rocwmma], or the HIP include tree (rocWMMA installs there on Linux).
     rocWMMA is header-only and is NOT in the ROCm Windows SDK, hence the extra search paths.

     "Complete" is load-bearing, not pedantry (gh-ocannl-1032): Ubuntu 26.04's librocwmma-dev 7.1.0
     installs the five umbrella headers into /usr/include/rocwmma WITHOUT the rocwmma/internal/
     directory that rocwmma.hpp's very first include needs, and that is the tree an [apt install
     rocwmma] leaves behind on the distro ROCm stack. Probing rocwmma.hpp alone accepts it, and then
     every tensorized kernel fails at hiprtc with "'internal/accessors.hpp' file not found" --
     turning a clean capability decline into a compile error on the one path that is supposed to be
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

  let mma_supported () =
    Lazy.force all_rdna_wave32 && Option.is_some (Lazy.force rocwmma_include_dir)

  let%diagn2_sexp hip_to_code ~name hip_src =
    let name_hip = name ^ ".hip" in
    let uses_rocwmma = C_syntax.source_mentions ~marker:"rocwmma::" hip_src in
    let hip_src =
      C_syntax.prepend_conditional_includes
        ~conditional_includes:Cuda_like_config.Hip.conditional_includes hip_src
    in
    if Utils.settings.output_debug_files_in_build_directory then (
      let build_file = Utils.open_build_file ~base_name:name ~extension:".hip" in
      Stdio.Out_channel.output_string build_file.oc hip_src;
      build_file.finalize ());
    [%log "compiling to a code object"];
    (* [with_debug] only asks hiprtc to keep the compilation log on a SUCCESSFUL compile (a failing
       one carries its log in the exception either way), and the only reader of that log is the
       [.hip_log] build file written just below -- so it wants exactly the flag that writes it. The
       [|| log_level > 0] disjunct it used to carry made the log's retention hostage to a verbosity
       knob that nothing here consults (gh-ocannl-595). *)
    let with_debug = Utils.settings.output_debug_files_in_build_directory in
    (* hiprtc targets the architecture of the current default device when no [--offload-arch] is
       given. On Linux hiprtc ships built-in HIP headers; on Windows (observed with ROCm 7.1)
       [#include <hip/hip_fp16.h>] is not found without an include path, so point at the SDK's
       include directory ([hip_sdk_include_dir]: the no-spaces junction created by ocaml-hipjit,
       falling back to HIP_PATH or /opt/rocm). The -I is only added when the directory exists, so
       the Linux built-in-headers path is unaffected. *)
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
    let options =
      Compiler_options.hiprtc ~hip_include_options:hip_include_opt
        ~rocwmma_include_options:rocwmma_include_opt ~uses_rocwmma
        ~with_debug:(Utils.with_runtime_debug ())
    in
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
      Stdio.Out_channel.output_string oc @@ Option.value ~default:"" (Hiprtc.compilation_log code);
      Stdio.Out_channel.flush oc;
      Stdio.Out_channel.close oc);
    code

  let run_options () =
    (* NOTE: on the AMD platform these are accepted for CUDA-driver-API compatibility but mostly
       ignored by [hipModuleLoadDataEx]. *)
    if Utils.with_runtime_debug () then
      H.Module.[ GENERATE_DEBUG_INFO true; GENERATE_LINE_INFO true ]
    else []

  (* No runtime linking needed since Threefry is included directly in each kernel *)
  let set_builtins_for_device ~primary_context:_ _kernel_module = assert !initialized

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

  type code = {
    code : Hiprtc.compile_to_code_result;
    kparams : (string * kparam_source) list;
    bindings : Indexing.unit_bindings;
    name : string;
    launch : Low_level.launch_dims;
  }
  [@@deriving sexp_of]

  type code_batch = {
    code : Hiprtc.compile_to_code_result;
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
       triples are seeded, so narrow 16-bit accumulators keep their storage residency — widening the
       serial legs here would re-introduce the serial-vs-mma width dependence gh-ocannl-639 removes.
       fp8 has an accumulator format on no backend (its serial arithmetic already bridges through
       float per operator), so it follows the CPU policy: f32 residency, one narrowing per nest,
       governed by the same [narrow_compute_f32] knob.

       Under [Numerics.Fp16_wide] (gh-ocannl-680) f16 accumulators reside in f32 here too, and the
       mma legs follow — not by being withheld, but by swapping arms: [mma_combo] renders the
       uniform-f16 combination against a [float] accumulator fragment, converting at the [d]
       boundary (gh-ocannl-789), so width stays schedule-uniform WITH the tensor unit rather than
       against it, and [mma_f16_wide_acc_scopes] advertises both emission scopes.
       [Numerics.Bf16_wide] does the same for bf16 (gh-ocannl-838) — the uniform-bf16 arm swaps to
       an f32 accumulator fragment — which is what takes the uniform-bf16 legs off gfx11's
       bf16-accumulate WMMA, whose result is not exactly rounded. *)
    let accum_prec prec =
      match prec with
      | Ops.Half_prec _ when Numerics.fp16_accum_wide () -> Ops.single
      | Ops.Bfloat16_prec _ when Numerics.bf16_accum_wide () -> Ops.single
      | Ops.Fp8_prec _ when (Numerics.get ()).Numerics.narrow_compute_f32 -> Ops.single
      | _ -> prec

    (* --- Shared rocWMMA tile-MMA vocabulary, used by both [mma_syntax] and [mma_fragment_syntax].
       The combination table lived in both hooks verbatim before gh-ocannl-789 added an arm to it;
       one copy is what keeps a new arm from reaching only half the emission (the two hooks' guards
       are required to accept together). *)
    let mma_tile = 16

    let mma_frag_typ kind typ layout =
      Printf.sprintf "rocwmma::fragment<rocwmma::%s, %d, %d, %d, %s%s>" kind mma_tile mma_tile
        mma_tile typ
        (match layout with Some l -> ", rocwmma::" ^ l | None -> "")

    (* (a/b fragment element type, accumulator fragment element type, [d] STORAGE element type, ld
       multiple for a/b, ld multiple for d). rocWMMA element types [rocwmma::float16_t] /
       [rocwmma::bfloat16_t] / [float] need not be textually identical to the node's own C type
       ([__half] / [__hip_bfloat16]), so the operand pointers are [reinterpret_cast] to them at each
       call site. The accumulator and the destination storage types coincide on every arm but the
       wide-f16 and wide-bf16 ones; where they differ, [mma_d_boundary] carries the conversion. *)
    let mma_combo ~a_prec ~b_prec ~d_prec ~d_layout ~a_layout ~b_layout =
      (* rocWMMA fragments are opaque like [nvcuda::wmma]'s: there is no swizzle-aware fragment load
         here, so a swizzled operand layout declines to the caller's scalar fallback (gh-ocannl-481
         item 3, D2). *)
      let plain = function `Plain -> true | `Swizzled_b128 -> false in
      if not (plain d_layout && plain a_layout && plain b_layout) then None
      else
        match (a_prec, b_prec, d_prec) with
        | Ops.Half_prec _, Ops.Half_prec _, Ops.Single_prec _ ->
            Some ("rocwmma::float16_t", "float", "float", 8, 4)
        (* The uniform-f16 arm's accumulator follows the [Numerics] policy, and the two arms are
           mutually exclusive by construction. Under [Fp16_auto]/[Fp16_narrow] the accumulator
           fragment is itself f16, so the [d] boundary is rocWMMA's own load/store. Under
           [Fp16_wide] (gh-ocannl-789) the accumulator is [float] against the same f16 STORAGE, and
           [mma_d_boundary] converts once at each end -- which is what lets
           [mma_f16_wide_acc_scopes] advertise both scopes and the uniform-f16 seeds survive the
           wide policy on this backend. *)
        | Ops.Half_prec _, Ops.Half_prec _, Ops.Half_prec _ when Numerics.fp16_accum_wide () ->
            Some ("rocwmma::float16_t", "float", "rocwmma::float16_t", 8, 8)
        | Ops.Half_prec _, Ops.Half_prec _, Ops.Half_prec _ ->
            Some ("rocwmma::float16_t", "rocwmma::float16_t", "rocwmma::float16_t", 8, 8)
        | Ops.Bfloat16_prec _, Ops.Bfloat16_prec _, Ops.Single_prec _ ->
            Some ("rocwmma::bfloat16_t", "float", "float", 8, 4)
        (* The uniform-bf16 twin of the wide-f16 arm (gh-ocannl-838): under [Bf16_wide] a [float]
           accumulator against the bf16 STORAGE destination, converted by [mma_d_boundary]. gfx11's
           bf16-accumulate WMMA is not exactly rounded (about a bf16 ulp at the partial-sum scale,
           see schedule_mma_matmul's table), so this leaves only the narrowing rounding. *)
        | Ops.Bfloat16_prec _, Ops.Bfloat16_prec _, Ops.Bfloat16_prec _
          when Numerics.bf16_accum_wide () ->
            Some ("rocwmma::bfloat16_t", "float", "rocwmma::bfloat16_t", 8, 8)
        | Ops.Bfloat16_prec _, Ops.Bfloat16_prec _, Ops.Bfloat16_prec _ ->
            Some ("rocwmma::bfloat16_t", "rocwmma::bfloat16_t", "rocwmma::bfloat16_t", 8, 8)
        | _ -> None

    (* One 16x16 block of the [d] boundary: [`Load] brings the destination block into the
       accumulator fragment [acc], [`Store] writes it back. Where the accumulator element type is
       the destination's storage type this is rocWMMA's own load/store, exactly as before
       gh-ocannl-789.

       Where they differ -- the wide-f16 and wide-bf16 arms, a [float] accumulator over a 16-bit
       storage destination -- neither rocWMMA call is type-correct, so the conversion stages through
       a DESTINATION-TYPED accumulator fragment that rocWMMA does load and store, and copies
       element-for-element with [num_elements] / [x[]], the surface rocWMMA documents as
       "compatibility with nvcuda::wmma". This is legitimate despite fragments being opaque, because
       it never assumes WHICH matrix cell an element index names: it only assumes that two
       accumulator fragments of the same 16x16x16 shape name the SAME cell at the same index,
       whatever that cell is. That holds by rocWMMA's construction (the accumulator's IO layout is
       derived from the fragment shape and the wave size, not from its element type) and is verified
       on gfx1151: loading a 16x16 tile of distinct values through a [float] and a [float16_t]
       accumulator fragment and dumping every lane's elements gives identical per-(lane, index)
       values, 8 elements each; the [bfloat16_t] staging of the wide-bf16 arm is pinned end to end
       by schedule_mma_matmul's [Bf16_wide] legs on the same device (gh-ocannl-838). Deliberately
       not the warp-staged float tile in LDS that was the fallback design (gh-ocannl-789): this adds
       no memory traffic at all.

       [acc] and [ptr] are C expressions for one block ([__mma_acc[__mi][__ni]] and the block's base
       pointer), so the caller keeps ownership of the block indexing. *)
    let mma_d_boundary ~dir ~acc_typ ~d_typ ~acc ~ptr ~ldd =
      let load frag =
        Printf.sprintf "rocwmma::load_matrix_sync(%s, %s, %d, rocwmma::mem_row_major);" frag ptr ldd
      in
      let store frag =
        Printf.sprintf "rocwmma::store_matrix_sync(%s, %s, %d, rocwmma::mem_row_major);" ptr frag
          ldd
      in
      if String.equal acc_typ d_typ then
        [ (match dir with `Load -> load acc | `Store -> store acc) ]
      else
        let stage = mma_frag_typ "accumulator" d_typ None in
        (* The element-count agreement the copy relies on, checked by the C++ compiler rather than
           assumed: a rocWMMA release that packed the two accumulator types differently would fail
           the kernel compile here instead of silently converting a prefix. *)
        let guard =
          Printf.sprintf
            "  static_assert(%s::num_elements == %s::num_elements, \"rocwmma accumulator element \
             counts must agree at the d boundary\");"
            stage
            (mma_frag_typ "accumulator" acc_typ None)
        in
        let copy ~src ~dst ~cast =
          Printf.sprintf
            "  for (int __ei = 0; __ei < (int)%s.num_elements; ++__ei) %s.x[__ei] = (%s)%s.x[__ei];"
            src dst cast src
        in
        "{"
        :: Printf.sprintf "  %s __mma_dstage;" stage
        :: guard
        ::
        (match dir with
        | `Load -> [ "  " ^ load "__mma_dstage"; copy ~src:"__mma_dstage" ~dst:acc ~cast:acc_typ ]
        | `Store -> [ copy ~src:acc ~dst:"__mma_dstage" ~cast:d_typ; "  " ^ store "__mma_dstage" ])
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
       twin bitwise; the bf16 and f16 -> f32 combinations take the same rocWMMA template path,
       differing only in fragment element type. rocWMMA (header-only) is cloned under
       %LOCALAPPDATA%/rocwmma since it is not in the ROCm 7.1 Windows SDK. [hip_to_code] injects the
       header and -std=c++17 only when a kernel actually uses it, so non-tensor-core kernels are
       unaffected and do not require rocWMMA to be present. *)
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
          let combo = mma_combo ~a_prec ~b_prec ~d_prec ~d_layout ~a_layout ~b_layout in
          let loadable = function
            | `Device | `Shared -> true (* generic-address loads cover both *)
            | `Thread | `Fragment _ -> false
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
              | Some (ab_typ, _acc_typ, _d_typ, ab_ld_mult, _d_ld_mult)
                when mma_supported ()
                     && m % tile = 0
                     && n % tile = 0
                     && k % tile = 0
                     && lda % ab_ld_mult = 0
                     && ldb % ab_ld_mult = 0
                     && loadable a_space && loadable b_space ->
                  let mt = m / tile and nt = n / tile and kt = k / tile in
                  let frag = mma_frag_typ in
                  let ptr_decl name typ ptr =
                    string (Printf.sprintf "%s *%s = reinterpret_cast<%s *>(" typ name typ)
                    ^^ ptr ^^ string ");"
                  in
                  let a_layout = if ta then "col_major" else "row_major" in
                  let b_layout = if tb then "col_major" else "row_major" in
                  let barrier = "__syncthreads();" in
                  let body_lines =
                    [
                      Printf.sprintf "for (int __ki = 0; __ki < %d; ++__ki) {" kt;
                      Printf.sprintf "  %s __mma_bf[%d];"
                        (frag "matrix_b" ab_typ (Some b_layout))
                        nt;
                      Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                      (if tb then
                         Printf.sprintf
                           "    rocwmma::load_matrix_sync(__mma_bf[__ni], __mma_bp + __ni * %d * \
                            %d + __ki * %d, %d);"
                           tile ldb tile ldb
                       else
                         Printf.sprintf
                           "    rocwmma::load_matrix_sync(__mma_bf[__ni], __mma_bp + __ki * %d * \
                            %d + __ni * %d, %d);"
                           tile ldb tile ldb);
                      "  }";
                      Printf.sprintf "  for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                      Printf.sprintf "    %s __mma_af;" (frag "matrix_a" ab_typ (Some a_layout));
                      (if ta then
                         Printf.sprintf
                           "    rocwmma::load_matrix_sync(__mma_af, __mma_ap + __ki * %d * %d + \
                            __mi * %d, %d);"
                           tile lda tile lda
                       else
                         Printf.sprintf
                           "    rocwmma::load_matrix_sync(__mma_af, __mma_ap + __mi * %d * %d + \
                            __ki * %d, %d);"
                           tile lda tile lda);
                      Printf.sprintf "    for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                      Printf.sprintf
                        "      rocwmma::mma_sync(%s[__mi][__ni], __mma_af, __mma_bf[__ni], \
                         %s[__mi][__ni]);"
                        fragment fragment;
                      "    }";
                      "  }";
                      "}";
                      barrier;
                    ]
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
              | Some (ab_typ, acc_typ, d_typ, ab_ld_mult, d_ld_mult)
                when mma_supported ()
                     && m % tile = 0
                     && n % tile = 0
                     && k % tile = 0
                     && lda % ab_ld_mult = 0
                     && ldb % ab_ld_mult = 0
                     && ldd % d_ld_mult = 0
                     && loadable d_space && loadable a_space && loadable b_space ->
                  let open PPrint in
                  let mt = m / tile and nt = n / tile and kt = k / tile in
                  let frag = mma_frag_typ in
                  (* [reinterpret_cast] bridges the node's C element type to the rocWMMA fragment
                     type. *)
                  let ptr_decl name typ ptr =
                    string (Printf.sprintf "%s *%s = reinterpret_cast<%s *>(" typ name typ)
                    ^^ ptr ^^ string ");"
                  in
                  let a_layout = if ta then "col_major" else "row_major" in
                  let b_layout = if tb then "col_major" else "row_major" in
                  let barrier = "__syncthreads();" in
                  let body_lines =
                    [
                      barrier;
                      Printf.sprintf "%s __mma_acc[%d][%d];" (frag "accumulator" acc_typ None) mt nt;
                      Printf.sprintf "for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                      Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                    ]
                    @ List.map
                        ~f:(fun l -> "  " ^ l)
                        (mma_d_boundary ~dir:`Load ~acc_typ ~d_typ ~acc:"__mma_acc[__mi][__ni]" ~ldd
                           ~ptr:
                             (Printf.sprintf "__mma_dp + __mi * %d * %d + __ni * %d" tile ldd tile))
                    @ [
                        "  }";
                        "}";
                        Printf.sprintf "for (int __ki = 0; __ki < %d; ++__ki) {" kt;
                        Printf.sprintf "  %s __mma_bf[%d];"
                          (frag "matrix_b" ab_typ (Some b_layout))
                          nt;
                        Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                        (* Transposed storage ([tb]): the stored matrix is the role's transpose --
                           index it at (col, row) and declare the fragment [col_major]; the leading
                           dimension stays the operand's own. Same for [ta] below. *)
                        (if tb then
                           Printf.sprintf
                             "    rocwmma::load_matrix_sync(__mma_bf[__ni], __mma_bp + __ni * %d * \
                              %d + __ki * %d, %d);"
                             tile ldb tile ldb
                         else
                           Printf.sprintf
                             "    rocwmma::load_matrix_sync(__mma_bf[__ni], __mma_bp + __ki * %d * \
                              %d + __ni * %d, %d);"
                             tile ldb tile ldb);
                        "  }";
                        Printf.sprintf "  for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                        Printf.sprintf "    %s __mma_af;" (frag "matrix_a" ab_typ (Some a_layout));
                        (if ta then
                           Printf.sprintf
                             "    rocwmma::load_matrix_sync(__mma_af, __mma_ap + __ki * %d * %d + \
                              __mi * %d, %d);"
                             tile lda tile lda
                         else
                           Printf.sprintf
                             "    rocwmma::load_matrix_sync(__mma_af, __mma_ap + __mi * %d * %d + \
                              __ki * %d, %d);"
                             tile lda tile lda);
                        Printf.sprintf "    for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                        "      rocwmma::mma_sync(__mma_acc[__mi][__ni], __mma_af, __mma_bf[__ni], \
                         __mma_acc[__mi][__ni]);";
                        "    }";
                        "  }";
                        "}";
                        Printf.sprintf "for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                        Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                      ]
                    @ List.map
                        ~f:(fun l -> "  " ^ l)
                        (mma_d_boundary ~dir:`Store ~acc_typ ~d_typ ~acc:"__mma_acc[__mi][__ni]"
                           ~ldd
                           ~ptr:
                             (Printf.sprintf "__mma_dp + __mi * %d * %d + __ni * %d" tile ldd tile))
                    @ [ "  }"; "}"; barrier ]
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
          let combo = mma_combo ~a_prec ~b_prec ~d_prec ~d_layout ~a_layout ~b_layout in
          let loadable = function `Device | `Shared -> true | `Thread | `Fragment _ -> false in
          match combo with
          | Some (_ab_typ, acc_typ, d_typ, ab_ld_mult, d_ld_mult)
            when mma_supported ()
                 && m % tile = 0
                 && n % tile = 0
                 && k % tile = 0
                 && lda % ab_ld_mult = 0
                 && ldb % ab_ld_mult = 0
                 && ldd % d_ld_mult = 0
                 && loadable d_space && loadable a_space && loadable b_space ->
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
                  Printf.sprintf "for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                  Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                ]
                @ List.map
                    ~f:(fun l -> "  " ^ l)
                    (mma_d_boundary ~dir:`Load ~acc_typ ~d_typ ~acc:(fragment ^ "[__mi][__ni]") ~ldd
                       ~ptr:(Printf.sprintf "__mma_dp + __mi * %d * %d + __ni * %d" tile ldd tile))
                @ [ "  }"; "}"; "/* rocwmma fragment reduction body begins */" ]
              in
              let lines_after =
                [
                  "/* rocwmma fragment reduction body ends */";
                  Printf.sprintf "for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                  Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                ]
                @ List.map
                    ~f:(fun l -> "  " ^ l)
                    (mma_d_boundary ~dir:`Store ~acc_typ ~d_typ ~acc:(fragment ^ "[__mi][__ni]")
                       ~ldd
                       ~ptr:(Printf.sprintf "__mma_dp + __mi * %d * %d + __ni * %d" tile ldd tile))
                @ [ "  }"; "}"; barrier ]
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

  (* hiprtc ships built-in HIP headers, and device-side printf needs no declaration on ROCm. *)
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

  module Compile = C_syntax.Compile_driver (Hip_syntax_config)

  let%diagn2_sexp compile ~name bindings lowered =
    let code, kparams, name, launch =
      Compile.compile ~name bindings lowered ~includes:hip_includes ~builtins:Builtins_hip.builtins
        ~conditional_includes:Cuda_like_config.Hip.conditional_includes ~compile_source:hip_to_code
        ()
    in
    { code; kparams; bindings; name; launch }

  let%diagn2_sexp compile_batch ~names bindings lowereds =
    let code, kparams_and_names =
      Compile.compile_batch ~names bindings lowereds ~includes:hip_includes
        ~builtins:Builtins_hip.builtins
        ~conditional_includes:Cuda_like_config.Hip.conditional_includes ~compile_source:hip_to_code
        ()
    in
    { code; kparams_and_names; bindings }

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
    fun (device : device) -> Lazy.force (Lazy.force cache).(device.ordinal)

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

  let link_proc ~prior_context ~name ~(kparams : (string * kparam_source) list)
      ~(launch : Low_level.launch_dims) ~ctx_buffers lowered_bindings run_module =
    let func = H.Module.get_function run_module ~name in
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
    let run_module = H.Module.load_data_ex code.code (run_options ()) in
    prior_context.device.dev.set_builtins_in run_module;
    let idx_params = Indexing.bound_symbols code.bindings in
    let lowered_bindings : Indexing.lowered_bindings =
      List.map idx_params ~f:(fun s -> (s, ref 0))
    in
    let task =
      link_proc ~prior_context ~name:code.name ~kparams:code.kparams ~launch:code.launch
        ~ctx_buffers lowered_bindings run_module
    in
    (lowered_bindings, task)

  let%track3_sexp link_batch prior_context (code_batch : code_batch) ctx_buffers =
    let idx_params = Indexing.bound_symbols code_batch.bindings in
    let lowered_bindings : Indexing.lowered_bindings =
      List.map idx_params ~f:(fun s -> (s, ref 0))
    in
    let ctx = ctx_of prior_context in
    set_ctx ctx;
    let run_module = H.Module.load_data_ex code_batch.code (run_options ()) in
    prior_context.device.dev.set_builtins_in run_module;
    let procs =
      Array.map code_batch.kparams_and_names ~f:(fun (kparams, name, launch) ->
          link_proc ~prior_context ~name ~kparams ~launch ~ctx_buffers lowered_bindings run_module)
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
      ~uses_merge_buffer (tasks : Task.t list) : Task.t option =
    let use_capture =
      List.length tasks > 1
      && Utils.get_global_flag ~default:true ~arg_name:"gpu_graph_capture"
      && not (Utils.debug_log_from_routines ())
    in
    if not use_capture then None
    else
      let device = context.device in
      let max_cached_graphs = 128 in
      let cache : (string, H.Graph.exec) Hashtbl.t = Hashtbl.create (module String) in
      let order : string Queue.t = Queue.create () in
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
        (Task.Task
           {
             context_lifetime = tasks;
             description = "graph-captured segments of " ^ name ^ " on " ^ get_name device;
             work =
               (fun () ->
                 if !broken then run_plain ()
                 else (
                   set_ctx @@ ctx_of context;
                   let key = current_key () in
                   match Hashtbl.find cache key with
                   | Some exec -> H.Graph.launch exec device.runner
                   | None -> (
                       match capture () with
                       | exec ->
                           if Queue.length order >= max_cached_graphs then (
                             let victim = Queue.dequeue_exn order in
                             (* The evicted exec may still have a pending launch on the stream. *)
                             H.Stream.synchronize device.runner;
                             H.Graph.exec_destroy (Hashtbl.find_exn cache victim);
                             Hashtbl.remove cache victim);
                           Hashtbl.set cache ~key ~data:exec;
                           Queue.enqueue order key;
                           H.Graph.launch exec device.runner
                       | exception H.Hip_error { status; message } ->
                           (* E.g. capture unsupported on this runtime: fall back to per-segment
                              launches for this routine (same-stream FIFO supplies the segment
                              ordering), and re-run outside capture so a genuine launch failure
                              surfaces on the plain path. *)
                           broken := true;
                           Stdio.eprintf
                             "ocannl: disabling HIP graph capture for routine %s (%s: %s)\n%!" name
                             message
                             (Sexp.to_string_hum @@ H.sexp_of_result status);
                           run_plain ())));
           })

  let get_global_debug_info () =
    Sexp.message "hip_global_debug"
      [ ("live_streams", [%sexp_of: int] @@ H.Stream.get_total_live_streams ()) ]

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
         {
           Backend_intf.max_threads_per_workgroup =
             min_over (fun (a : H.Device.attributes) -> a.max_threads_per_block);
           max_workgroup_memory_bytes =
             min_over (fun (a : H.Device.attributes) -> a.shared_mem_per_block);
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
           mma =
             (if mma_supported () then
                Some
                  {
                    Backend_intf.mma_simd_width = 32;
                    mma_tile = (16, 16, 16);
                    (* rocWMMA has both accumulator widths for both operand pairs — the CUDA
                       asymmetry that motivated the accumulator key (gh-ocannl-545) does not exist
                       here. *)
                    mma_format_tiles =
                      [
                        ( (Backend_intf.Mma_f16, Backend_intf.Mma_f16, Backend_intf.Mma_f32),
                          (16, 16, 16) );
                        ( (Backend_intf.Mma_f16, Backend_intf.Mma_f16, Backend_intf.Mma_f16),
                          (16, 16, 16) );
                        ( (Backend_intf.Mma_bf16, Backend_intf.Mma_bf16, Backend_intf.Mma_f32),
                          (16, 16, 16) );
                        ( (Backend_intf.Mma_bf16, Backend_intf.Mma_bf16, Backend_intf.Mma_bf16),
                          (16, 16, 16) );
                      ];
                    (* gh-ocannl-789: rocWMMA's [(f16, f16, f32)] fragments now carry the
                       uniform-f16 arm under [Numerics.Fp16_wide] too — [mma_combo] pairs a [float]
                       accumulator with the f16 STORAGE destination and [mma_d_boundary] converts
                       elementwise at each end — so the wide policy no longer costs this backend its
                       f16 tensor-unit legs (gh-ocannl-680's stated remainder). *)
                    mma_f16_wide_acc_scopes =
                      [ Backend_intf.Mma_per_statement; Backend_intf.Mma_fragment_scope ];
                    (* gh-ocannl-838: the uniform-bf16 arm swaps the same way under
                       [Numerics.Bf16_wide], through rocWMMA's [(bf16, bf16, f32)] fragments and the
                       same converted boundary. *)
                    mma_bf16_wide_acc_scopes =
                      [ Backend_intf.Mma_per_statement; Backend_intf.Mma_fragment_scope ];
                    (* rocWMMA fragments are opaque like wmma's: no swizzle-aware fragment load here
                       (gh-ocannl-481 item 3, D3). *)
                    mma_staged_layouts = [];
                    mma_pipeline_depths = [];
                  }
              else None);
           simd_vector_bytes = 0;
           native_fp16_arithmetic = false;
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
      ^ if Utils.with_runtime_debug () then "/device-debug" else "/no-device-debug"
    in
    fun () -> { (Lazy.force limits) with Backend_intf.codegen_tag = Some (codegen_tag ()) }

  let get_debug_info (device : device) =
    let tot, unr, unf = H.Stream.total_unreleased_unfinished_delimited_events device.runner in
    let i2s = [%sexp_of: int] in
    Sexp.message "hip_stream_debug"
      [ ("total_events", i2s tot); ("unreleased_events", i2s unr); ("unfinished_events", i2s unf) ]
end
