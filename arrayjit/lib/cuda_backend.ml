open Base
open Ir
module Tn = Tnode
module Lazy = Utils.Lazy
module Cu = Cuda
open Backend_intf

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_CUDA_BACKEND=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_CUDA_BACKEND"]

let () =
  Cu.cuda_call_hook :=
    Some
      (fun ~message:_message ~status:_status ->
        [%debug_sexp
          [%log5_block
            _message;
            if not @@ Cu.is_success _status then [%log (_status : Cu.result)]]])

let _suspended () =
  Cu.cuda_call_hook := Some (fun ~message ~status:_ -> Stdlib.Printf.printf "CUDA %s\n" message)

module Backend_buffer = struct
  type buffer_ptr = Cu.Deviceptr.t

  let sexp_of_buffer_ptr ptr = Sexp.Atom (Cu.Deviceptr.string_of ptr)
end

module Device_config = struct
  include Backend_buffer

  type dev = {
    dev : Cu.Device.t;
    primary_context : Cu.Context.t;
    set_builtins_in : Cu.Module.t -> unit;
  }
  [@@deriving sexp_of]

  type runner = Cu.Stream.t [@@deriving sexp_of]
  type event = Cu.Delimited_event.t [@@deriving sexp_of]

  let name = "cuda"
end

module Device_stream = Backend_impl.Device_types_ll (Device_config)
open Device_config

let set_ctx ctx = Cu.Context.set_current ctx

(* The CUDA slab allocator: a private [(device_id, pool_id) -> CUdeviceptr] table backing the shared
   {!Backend_intf.Slab_alloc}. *)
module Slab = struct
  open Backend_intf

  type device = Device_stream.device
  type buffer_ptr = Cu.Deviceptr.t

  (* Requested sizes are tracked alongside the pointers so [get_used_memory] can report the bytes
     OCANNL allocated on the device (gh-ocannl-289): the driver's [get_free_and_total_mem] moves in
     allocation granules (~2 MiB), which hides sub-granule effects such as the liveness planner's
     arena savings (gh-ocannl-489) and counts other processes' memory. *)
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
        Cu.Deviceptr.mem_free old);
    let ptr =
      try Cu.Deviceptr.mem_alloc ~size_in_bytes
      with
      | Cu.Cuda_error { status; message }
      when match Cu.sexp_of_result status with
           | Sexp.Atom "CUDA_ERROR_OUT_OF_MEMORY" -> true
           | _ -> false
      ->
        (* gh-ocannl-1326: typed here, where the refused size is known, so that every allocation
           site reports one backend-independent shape ([Schedule_outcome.protect] classifies it). *)
        let free = try Some (fst (Cu.Device.get_free_and_total_mem ())) with _ -> None in
        raise
          (Schedule_outcome.Device_memory_exhausted
             {
               backend = name;
               requested = size_in_bytes;
               free;
               detail = "CUDA_ERROR_OUT_OF_MEMORY raised by " ^ message;
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
        Option.iter (Hashtbl.find pools key) ~f:(fun (ptr, _) -> Cu.Deviceptr.mem_free ptr);
        Hashtbl.remove pools key)

  let resolve_pool (device : device) { pool_id; offset = _ } : buffer_ptr =
    (* Return the slab base. The byte offset is NOT folded into the handle here; callers apply it
       via the cudajit ?offset / ?dst_offset / ?src_offset params or via Cu.Deviceptr.offset. *)
    fst (Hashtbl.find_exn pools (device.device_id, pool_id))

  let used_memory (device : device) =
    Hashtbl.fold pools ~init:0 ~f:(fun ~key:(dev_id, _) ~data:(_, size) acc ->
        if dev_id = device.device_id then acc + size else acc)

  let memset_zero (device : device) ~pool_id ~offset ~size_in_bytes =
    let base = resolve_pool device { pool_id; offset } in
    if size_in_bytes > 0 then
      Cu.Stream.memset_d8 ~offset base Unsigned.UChar.zero ~length:size_in_bytes device.runner
end

(* [initialized_devices] never forgets its entries. *)
let initialized_devices = Hash_set.create (module Int)
let initialized = ref false

(* The [-I] nvrtc needs to find <cuda_fp8.h> and friends, discovered from the environment: CUDA_PATH
   where it is set, the no-spaces junction ocaml-cudajit creates on Windows (whose SDK path
   otherwise contains spaces), and /usr/local/cuda as the Linux fallback. Separated out of the
   compile path for the same reason [Cuda_mma.gpu_arch_options] is: it is a policy other nvrtc
   callers need to agree with, and tools/fp8_soak.ml is one -- a soak that looked only in
   /usr/local/cuda would report the CUDA arm ready and then fail to compile its kernel on every
   Windows installation and every Linux one outside that prefix (Codex P2 on PR #463). *)
let cuda_include_options () =
  (* On Windows, check for the no-spaces junction created by ocaml-cudajit *)
  let cuda_path =
    if String.(Stdlib.Sys.os_type = "Win32" || Stdlib.Sys.os_type = "Cygwin") then
      let junction_path =
        match Sys.getenv "LOCALAPPDATA" with
        | Some local_appdata -> local_appdata ^ "/cuda_path_link"
        | None -> ( match Sys.getenv "CUDA_PATH" with Some p -> p | None -> "")
      in
      if Stdlib.Sys.file_exists (junction_path ^ "/include") then Some junction_path
      else Sys.getenv "CUDA_PATH"
    else Sys.getenv "CUDA_PATH"
  in
  match cuda_path with
  | Some cuda_path ->
      (* Normalize path separators for Windows *)
      let include_path =
        if String.(Stdlib.Sys.os_type = "Win32" || Stdlib.Sys.os_type = "Cygwin") then
          String.map ~f:(fun c -> if Char.(c = '\\') then '/' else c) (cuda_path ^ "/include")
        else cuda_path ^ "/include"
      in
      [ "-I" ^ include_path ]
  | None ->
      if
        (* Fallback to common location if CUDA_PATH is not set *)
        Stdlib.Sys.file_exists "/usr/local/cuda/include"
      then [ "-I/usr/local/cuda/include" ]
      else []

(* The prelude every kernel starts with. Top level rather than in [Impl] so [host_capabilities]
   fingerprints exactly the headers it names. *)
let cuda_header_prelude =
  {|#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

/* Define math constants that would normally come from <math.h> */
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif
#ifndef NAN
#define NAN __int_as_float(0x7fffffff)
#endif|}

module Impl : Ir.Backend_impl.Lowered_backend = struct
  include Backend_impl.Device (Device_stream) (Slab)

  (* The concrete [buffer_ptr]/[buffer] + sexps for the impl-facing interface (no longer carried by
     the shared [Device_config_common]). *)
  include Backend_buffer

  let ctx_of (context : context) = context.device.dev.primary_context
  let is_done event = Cu.Delimited_event.query event
  let will_wait_for context event = Cu.Delimited_event.wait context.device.runner event
  let sync event = Cu.Delimited_event.synchronize event
  let all_work device = Cu.Delimited_event.record device.runner

  (* Driver initialization and device discovery are lazy: the singleton [Impl] module initializes at
     program startup (Backends instantiates it eagerly for nameable types), and cudajit is a depopt
     -- the library being installed does not imply a usable driver/GPU. Forcing here, at first
     device use, keeps CPU-only runs from touching the driver and lets [Context.auto] catch
     unusable-CUDA failures per call, as the retired per-call [fresh_backend] did. *)
  let ensure_initialized =
    lazy
      (if not !initialized then (
         Cu.init ();
         initialized := true))

  let num_devices () =
    Lazy.force ensure_initialized;
    Cu.Device.get_count ()

  (* [devices] is mutable to support plugging in new devices. *)
  let devices = lazy (ref @@ Array.create ~len:(num_devices ()) None)

  (* Minimum compute capability across devices as [major * 10 + minor] (e.g. 89 for sm_89, 120 for
     sm_120), memoized behind [lazy] (driver init and device enumeration must not run at
     backend-module initialization). Tensor cores (wmma) need sm_70+, bf16 wmma sm_80+, fp8 mma.sync
     sm_89+ (docs/proposals/tensorize-mma.md T3). *)
  let min_compute_capability =
    let cc =
      lazy
        ((* Device discovery a compile can force first: under the device-init deadline
            (gh-ocannl-1281, [Backends.With_device_init_deadline]). *)
         Utils.with_device_init_deadline ~backend_name:name (fun () ->
             let n = num_devices () in
             if n = 0 then 0
             else
               Array.init n ~f:(fun ordinal ->
                   let a : Cu.Device.attributes =
                     Cu.Device.get_attributes (Cu.Device.get ~ordinal)
                   in
                   (a.compute_capability_major * 10) + a.compute_capability_minor)
               |> Array.min_elt ~compare:Int.compare
               |> Option.value ~default:0))
    in
    fun () -> Lazy.force cc

  (* Bytes OCANNL has allocated on this device via [Slab], exact rather than the driver's
     granule-quantized [total - free] (gh-ocannl-289). Device-wide across contexts, matching the
     [Context.get_used_memory] contract. *)
  let get_used_memory (device : device) = Slab.used_memory device

  let get_free_and_total_memory (device : device) =
    set_ctx device.dev.primary_context;
    Some (Cu.Device.get_free_and_total_mem ())

  (* The merge buffer is the device's reserved single-tenant pool (id [merge_buffer_pool_id]); grow
     it in place when a larger node arrives ([Slab.alloc_pool] overwrites the reserved entry). *)
  let opt_alloc_merge_buffer ~size_in_bytes (device : device) : unit =
    if device.merge_buffer_capacity < size_in_bytes then
      Slab.alloc_pool device ~pool_id:merge_buffer_pool_id ~size_in_bytes ~alignment:1

  let%track4_sexp finalize_device (device : device) =
    Cu.Context.set_current device.dev.primary_context;
    Cu.Context.synchronize ();
    (* Note: this is not necessary as releasing the primary context by GC will reset the context.
       gh-ocannl-344: constants are bump-packed, so several cache entries share one constant pool
       slab; free each distinct [pool_id] exactly once (freeing per-entry would double-free / free a
       sub-region pointer). [Slab.free_pool] frees the slab and drops its table entry. *)
    Hashtbl.data device.constant_buffer_cache
    |> List.map ~f:(fun (loc : Backend_intf.buffer_loc) -> loc.pool_id)
    |> List.dedup_and_sort ~compare:Int.compare
    |> List.iter ~f:(fun pool_id ->
        Option.iter Slab.free_pool ~f:(fun free -> free device ~pool_id))

  let with_conditional_includes cu_src =
    C_syntax.prepend_conditional_includes
      ~conditional_includes:Cuda_like_config.Cuda.conditional_includes cu_src

  (* The options nvrtc compiles [cu_src] -- the source with its conditional includes applied --
     with: the target arch its markers floor (over the device set), the include paths, the numerics
     flags, the debug flag. A function of the source and the device set alone, so the kernel code
     cache keys on it (gh-ocannl-1529). *)
  let nvrtc_options cu_src =
    let arch_opts = Cuda_mma.gpu_arch_options ~device_cc:(min_compute_capability ()) cu_src in
    (* gh-ocannl-784: assembled by [Compiler_options.nvrtc], the pure module the HIP backend already
       shares, so the exact vector -- semantic state for fast-math numerics -- is pinned by a
       GPU-free test (arrayjit/test/test_cuda_compile_options.ml) instead of only by whichever box
       happens to have a device. Its header carries the measured reassociation boundary. *)
    Compiler_options.nvrtc ~cuda_include_options:(cuda_include_options ()) ~arch_options:arch_opts
      ~with_device_debug:(Utils.with_runtime_debug ())

  let write_source_artifact ~name cu_src =
    if Utils.settings.output_debug_files_in_build_directory then (
      let build_file = Utils.open_build_file ~base_name:name ~extension:".cu" in
      Stdio.Out_channel.output_string build_file.oc cu_src;
      build_file.finalize ())

  (* One nvrtc program, [cu_src] with its conditional includes applied. [log_header] opens the
     [.cu_log] artifact: the kernel cache's account of which kernels this program compiled. *)
  let%diagn2_sexp compile_program ~name ?(log_header = "") ~options cu_src =
    let name_cu = name ^ ".cu" in
    [%log "compiling to PTX"];
    (* [with_debug] only asks NVRTC to keep the compilation log on a SUCCESSFUL compile (a failing
       one carries its log in the exception either way), and the only reader of that log is the
       [.cu_log] build file written just below -- so it wants exactly the flag that writes it, and
       that file's [Option.value_exn] is what makes the implication load-bearing. The [|| log_level
       > 0] disjunct it used to carry made the log's retention hostage to a verbosity knob that
       nothing here consults (gh-ocannl-595). Kernel debug codegen is separate: it comes from
       [Utils.with_runtime_debug ()] via [--device-debug] in [nvrtc_options]. *)
    let with_debug = Utils.settings.output_debug_files_in_build_directory in
    [%log "nvrtc options", (options : string list)];
    let ptx =
      C_syntax.with_compiler_options ~compiler:"nvrtc" ~options
        ~enrich:(fun exn ~suffix ->
          match exn with
          | Nvrtc.Nvrtc_error { status; message } ->
              Some (Nvrtc.Nvrtc_error { status; message = message ^ suffix })
          | _ -> None)
        (fun () -> Nvrtc.compile_to_ptx ~cu_src ~name:name_cu ~options ~with_debug)
    in
    if Utils.settings.output_debug_files_in_build_directory then (
      let oc = Out_channel.open_text @@ Utils.build_file @@ name ^ ".ptx" in
      Stdio.Out_channel.output_string oc @@ Nvrtc.string_from_ptx ptx;
      Stdio.Out_channel.flush oc;
      Stdio.Out_channel.close oc;
      let oc = Out_channel.open_text @@ Utils.build_file @@ name ^ ".cu_log" in
      Stdio.Out_channel.output_string oc log_header;
      Stdio.Out_channel.output_string oc
      @@ Option.value_exn ~here:[%here] (Nvrtc.compilation_log ptx);
      Stdio.Out_channel.flush oc;
      Stdio.Out_channel.close oc);
    ptx

  let run_options () =
    if Utils.with_runtime_debug () then
      Cu.Module.[ GENERATE_DEBUG_INFO true; GENERATE_LINE_INFO true ]
    else []

  (* gh-ocannl-550: [Cu.Module] exposes no explicit unload — a module is unloaded by cudajit's own
     GC finalizer — so loads counted against unloads is the only way to see whether a schedule
     search's per-candidate modules are being reclaimed at all. The added finalizer only counts; the
     unload stays cudajit's, and the two finalizers on one module are independent. *)
  let load_module ptx =
    let m = Cu.Module.load_data_ex ptx (run_options ()) in
    Alloc_census.count_module_loaded ();
    Stdlib.Gc.finalise (fun _ -> Alloc_census.count_module_unloaded ()) m;
    m

  (* No longer need runtime linking since Threefry is included directly in each kernel *)
  let set_builtins_for_device ~primary_context:_ _kernel_module = assert !initialized

  let%track3_sexp get_device ~(ordinal : int) : device =
    let n = num_devices () in
    (* No devices at all is "this backend is not available here" ([Context.auto] moves on); asking
       for an ordinal past an existing device is an ordinary caller error. A driver that fails to
       initialize propagates out of [num_devices] unchanged — see
       {!Backend_intf.Backend_unavailable}. *)
    if n = 0 then
      raise
      @@ Backend_intf.Backend_unavailable
           { backend = name; detail = "the driver reports no CUDA devices" };
    if n <= ordinal then
      invalid_arg [%string "Exec_as_cuda.get_device %{ordinal#Int}: not enough devices"];
    let devices = Lazy.force devices in
    (if Array.length !devices <= ordinal then
       let old, len = (!devices, Array.length !devices) in
       devices := Array.init (ordinal + 1) ~f:(fun i -> if i < len then old.(i) else None));
    let default () =
      let dev = Cu.Device.get ~ordinal in
      let primary_context : Cu.Context.t = Cu.Context.get_primary dev in
      let set_builtins_in = set_builtins_for_device ~primary_context in
      let dev = { dev; primary_context; set_builtins_in } in
      set_ctx primary_context;
      if Utils.debug_log_from_routines () && not (Hash_set.mem initialized_devices ordinal) then
        Int.of_string_opt @@ Utils.get_global_arg ~arg_name:"cuda_printf_fifo_size" ~default:""
        |> Option.iter ~f:Cu.Context.(set_limit PRINTF_FIFO_SIZE);
      Hash_set.add initialized_devices ordinal;
      (* With one compute stream per device, the runner (CUDA stream) is created with the device. *)
      let cu_stream = Cu.Stream.create ~non_blocking:true () in
      let result = make_device dev cu_stream ~ordinal in
      Stdlib.Gc.finalise finalize_device result;
      !devices.(ordinal) <- Some result;
      result
    in
    Option.value_or_thunk !devices.(ordinal) ~default

  let _cuda_properties =
    let cache =
      let%debug2_sexp f (ordinal : int) =
        let dev = get_device ~ordinal in
        lazy (Cu.Device.get_attributes dev.dev.dev)
      in
      lazy (Array.init (num_devices ()) ~f)
    in
    let%debug2_sexp get_props (device : device) : Cu.Device.attributes =
      let cache = Lazy.force cache in
      Lazy.force cache.(device.ordinal)
    in
    get_props

  let await (device : device) : unit =
    set_ctx device.dev.primary_context;
    (* Match the HIP logging contract: device-side [printf] must be fully drained before a caller
       closes or restores captured stdout. The stronger synchronization is debug-only. *)
    if Utils.debug_log_from_routines () then Cu.Context.synchronize ()
    else Cu.Stream.synchronize device.runner

  let is_idle (device : device) = Cu.Stream.is_ready device.runner

  (* Transfers take {!Backend_intf.buffer_loc} and resolve to the concrete [CUdeviceptr] here,
     against the device's private pool table. We pass [~length] explicitly to [memcpy_H_to_D] /
     [memcpy_D_to_H]: without it the cudajit impl computes [size_in_bytes = full_size - offset],
     which would reduce the copy to 0 bytes when the tensor is placed at an offset equal to its own
     size (the common bump-packed case). *)
  let from_host ~dst ~dst_loc hosted =
    set_ctx @@ ctx_of dst;
    let base = Slab.resolve_pool dst.device dst_loc in
    let f src =
      let full_bytes = Bigarray.Genarray.size_in_bytes src in
      let elem_bytes = Bigarray.kind_size_in_bytes (Bigarray.Genarray.kind src) in
      Cu.Stream.memcpy_H_to_D ~length:(full_bytes / elem_bytes) ~dst_offset:dst_loc.offset ~dst:base
        ~src dst.device.runner
    in
    Ndarray.apply { f } hosted

  let to_host ~src ~src_loc hosted =
    set_ctx @@ ctx_of src;
    let base = Slab.resolve_pool src.device src_loc in
    let f dst =
      let full_bytes = Bigarray.Genarray.size_in_bytes dst in
      let elem_bytes = Bigarray.kind_size_in_bytes (Bigarray.Genarray.kind dst) in
      Cu.Stream.memcpy_D_to_H ~length:(full_bytes / elem_bytes) ~src_offset:src_loc.offset ~dst
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
        Cu.Stream.memcpy_D_to_D ~size_in_bytes ~dst_offset ~src_offset ~dst:dst_base ~src:src_base
          dst.device.runner
      else
        Cu.Stream.memcpy_peer ~size_in_bytes ~dst_offset ~src_offset ~dst:dst_base
          ~dst_ctx:(ctx_of dst) ~src:src_base ~src_ctx:(ctx_of src) dst.device.runner
    in
    match (into_merge_buffer, dst_loc) with
    | No, None -> invalid_arg "Cuda_backend.device_to_device: missing dst_loc"
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

  (* One nvrtc program's PTX. [program_id] identifies it. With the kernel code cache
     (gh-ocannl-1529) the kernels of one batch can come from several programs: the ones it reused
     from earlier compiles, and the one program it compiled its cache misses into.

     [modules] holds the program's loaded module per CUDA context, loaded on the first link into
     that context and kept with the program ([module_of]): the driver's PTX JIT costs about as much
     per kernel as nvrtc does (measured for gh-ocannl-1529: 4.6 ms against 12.9 ms per kernel on
     sm_120), so a reused program that was loaded again on every link would pay it again for every
     kernel it holds, the unchanged ones included. A loaded module holds device code memory for as
     long as the program lives -- the cache's retention ([cuda_kernel_cache_mib]) or a compiled
     routine holding it -- and is unloaded by cudajit's finalizer once neither does. Sharing one
     module between routines is sound because OCANNL's kernels keep no mutable module-scope state:
     the only module-scope data are the builtins' [__constant__] tables. *)
  type program = {
    program_id : int;
    ptx : Nvrtc.compile_to_ptx_result;
    mutable modules : ((Cu.Context.t * Cu.Module.t) list[@sexp.opaque]);
    modules_lock : (Stdlib.Mutex.t[@sexp.opaque]);
  }
  [@@deriving sexp_of]

  type code = {
    code : program;
    kparams : (string * kparam_source) list;
    bindings : Indexing.unit_bindings;
    name : string;
    launch : Low_level.launch_dims;
  }
  [@@deriving sexp_of]

  type code_batch = {
    programs : program array;  (** Per kernel, in batch order: the program holding it. *)
    bindings : Indexing.unit_bindings;
    kparams_and_names : ((string * kparam_source) list * string * Low_level.launch_dims) array;
  }
  [@@deriving sexp_of]

  module Cuda_syntax_config (Input : sig
    val procs : Low_level.t array
  end) =
  struct
    include
      Cuda_like_config.Make
        (struct
          include Cuda_like_config.Cuda

          let builtins = Builtins_cuda.builtins
          let extra_blacklist = []
        end)
        (Input)

    (* gh-ocannl-487 phase 2: [cp.async] staging for software-pipelined tiles (the
       [ocannl_cp_async*] builtins; their name doubles as the [Cuda_mma.gpu_arch_options] floor
       marker). Gated on the attached devices' minimum compute capability, not the emission call:
       pre-Ampere devices keep the portable synchronous rendering, which is correct at any depth —
       the same posture as a hand-built pipelined schedule on a backend without the hook. *)
    let async_copy =
      if min_compute_capability () >= Cuda_mma.async_copy_floor then
        Some
          {
            C_syntax.ac_copy =
              (fun ~dst ~src ~bytes ->
                PPrint.(
                  string (Printf.sprintf "ocannl_cp_async%d(" bytes)
                  ^^ dst ^^ string ", " ^^ src ^^ string ");"));
            ac_wait_all = "ocannl_cp_async_wait_all();";
          }
      else None

    (* gh-ocannl-663: serial-rendered reduction accumulators mirror the mma legs' residency, so a
       narrow reduction's width does not depend on whether a schedule tensorized it. bf16 has no
       accumulator format on NVIDIA hardware — the uniform-bf16 [mma.sync] arm holds f32 per-lane
       registers across the whole [k] extent and narrows once at the [d] boundary — so serial bf16
       accumulation resides in f32 and narrows once per nest. fp8 likewise accumulates f32-only (and
       its serial arithmetic already bridges through float per operator); f16 accumulates natively
       at f16 in its seeded wmma triple and stays at storage width.

       The bf16 residency is STRUCTURAL — not gated on [narrow_compute_f32] — for the same reason
       Metal's fp8 one is: the seeded uniform-bf16 [mma.sync] arm accumulates f32 in hardware
       unconditionally, so honoring the off setting on the serial legs would re-introduce the
       schedule-dependent width this hook exists to remove (Codex P1 round 3 on PR #396). fp8 DOES
       honor the policy: no backend seeds an fp8-destination mma triple, so per-step fp8 semantics
       under policy-off are schedule-uniform. Compute precision stays the identity either way, so
       pointwise narrow arithmetic OUTSIDE recognized accumulations remains native. Within one, the
       whole update — contribution included — renders at the accumulator's residency, deliberately
       (Codex P1 on PR #396): operand widenings are exact, a bf16xbf16 product is exact in f32 — the
       same full-precision-product-into-f32 semantics the tensor cores apply per element — and the
       FMA form stays one [fmaf]. Rounding the contribution to bf16 first would mint a third
       numerics matching neither the mma legs nor the CPU serial rendering, whose f32 chain
       accum_width.ml pins bitwise across cc and CUDA. *)
    let accum_prec prec =
      match prec with
      | Ops.Bfloat16_prec _ -> Ops.single
      (* gh-ocannl-680: [Fp16_wide] gives f16 accumulators f32 residency on every backend; here the
         uniform-f16 mma legs keep width-uniform with it through the f32-accumulate inline-PTX
         m16n8k16 arm (the fragment layouts are shared by .f16 and .bf16), seeded via the
         per-statement entry in [mma_f16_wide_acc_scopes]. Under [Fp16_auto]/[Fp16_narrow] f16 keeps
         storage residency, matching the f16-accumulate wmma triple — the pre-gh-680 behavior. *)
      | Ops.Half_prec _ when Numerics.fp16_accum_wide () -> Ops.single
      | Ops.Fp8_prec _ when (Numerics.get ()).Numerics.narrow_compute_f32 -> Ops.single
      | _ -> prec

    (* The arm variant both mma hooks dispatch on, projected: [Cuda_mma.resolve_arm] and its floors,
       markers and tiles are pure functions of precisions, policy and scope, kept outside this
       [cudajit]-gated library so they are tested wherever OCANNL builds (gh-ocannl-1214,
       gh-ocannl-1232). *)
    let mma_arm = Cuda_mma.mma_arm

    (* The converted [d] boundary of a wmma accumulator-fragment array whose element type is not the
       destination's storage type (gh-ocannl-925, [wc_d_cvt]): [`Load] fills [frag][__mi][__ni] from
       the [mt] x [nt] 16x16 tiles of [__mma_dp] widened, [`Store] writes them back narrowed.

       NOT the HIP/Metal element copy through a same-shaped fragment of the destination type. That
       copy assumes the [float] and [__half] accumulator fragments place element [x[i]] at the same
       matrix coordinate, and for [nvcuda::wmma] nothing says so: the CUDA programming guide calls
       the mapping of matrix elements to fragment storage unspecified and subject to change across
       architectures, and the PTX ISA leaves the [wmma] fragment layouts opaque too. An element copy
       that happens to validate on one device is evidence about that device's compiler only. So the
       coordinates are read from the ACCUMULATOR FRAGMENT TYPE ITSELF: each lane [load_matrix_sync]s
       the row-major table [ocannl_wmma_rc16] (entry [16 * row + col] holds that very number, exact
       in f32) into a fragment of exactly [frag]'s type, after which [x[i]] of that fragment names
       the (row, col) that [x[i]] of every fragment of that type holds. What this relies on is only
       [load_matrix_sync]'s documented contract (element (row, col) of the described matrix lands at
       the fragment position representing (row, col)) plus the fact that the position is a property
       of the type and lane, not of the data or of the call — which [mma_sync] itself requires,
       being one instruction sequence however its operands were loaded. If an element were held by
       several lanes, [`Store] would have each write the same value to the same cell. The table
       costs one extra cached fragment load per boundary, twice per fragment scope. *)
    let wmma_d_boundary_lines ~dir ~acc_frag ~frag ~widen ~narrow ~ldd ~mt ~nt =
      let cell =
        Printf.sprintf "__mma_dp[(__mi * 16 + (__rc >> 4)) * %d + __ni * 16 + (__rc & 15)]" ldd
      in
      let elt = Printf.sprintf "%s[__mi][__ni].x[__t]" frag in
      [
        "{ /* wmma converted d boundary: coordinates from ocannl_wmma_rc16 */";
        Printf.sprintf "  %s __mma_rc;" acc_frag;
        "  nvcuda::wmma::load_matrix_sync(__mma_rc, ocannl_wmma_rc16, 16, \
         nvcuda::wmma::mem_row_major);";
        Printf.sprintf "  for (int __mi = 0; __mi < %d; ++__mi) {" mt;
        Printf.sprintf "    for (int __ni = 0; __ni < %d; ++__ni) {" nt;
        "      for (int __t = 0; __t < __mma_rc.num_elements; ++__t) {";
        "        const int __rc = (int)__mma_rc.x[__t];";
        (match dir with
        | `Load -> Printf.sprintf "        %s = %s(%s);" elt widen cell
        | `Store -> Printf.sprintf "        %s = %s(%s);" cell narrow elt);
        "      }";
        "    }";
        "  }";
        "}";
      ]

    (* Operand predicates shared by [mma_syntax] and [mma_fragment_syntax], so a fragment scope
       accepts exactly the operands its nested update-only calls do. *)
    let mma_loadable = function
      | `Device | `Shared -> true (* generic-address loads cover both *)
      | `Thread | `Fragment _ -> false

    let mma_plain = function `Plain -> true | `Swizzled_b128 -> false

    (* The wmma arms' one acceptance predicate ([Cuda_mma.wmma_accepts], gh-ocannl-1456) at this
       device's compute capability, for the operands as the hooks receive them: a [`Fragment] [a] or
       [b] is never an addressed one. *)
    let mma_wmma_accepts kind ~m ~n ~k ~d ~a:(lda, a_space, a_layout) ~b:(ldb, b_space, b_layout) =
      let place space layout =
        match space with
        | `Fragment _ -> None
        | (`Device | `Shared | `Thread) as space -> Some (space, layout)
      in
      match (place a_space a_layout, place b_space b_layout) with
      | Some a, Some b ->
          Cuda_mma.wmma_accepts kind ~cc:(min_compute_capability ()) ~m ~n ~k ~d ~a:(lda, a)
            ~b:(ldb, b)
      | _ -> false

    (* An operand eligible for the warp-cooperative [ldmatrix] load (gh-ocannl-481 item 3):
       [Cuda_mma.ldmatrix_eligible] (shared memory in the swizzled 16-byte-unit layout); a fragment
       is never one. Everything else keeps the per-lane gathers, which stay correct for plain shared
       tiles and device pointers alike. Note the asymmetry the arms rely on: eligibility is [space
       AND layout], but the DECLINE is on the layout alone — a swizzled operand an arm cannot
       [ldmatrix] must not silently fall through to row-major gathers, whatever space it sits in. *)
    let mma_ldm = function
      | ((`Shared | `Device | `Thread) as space), layout ->
          Cuda_mma.ldmatrix_eligible (space, layout)
      | `Fragment _, (`Plain | `Swizzled_b128) -> false

    (* The same per-operand load choice drives emission, its census and the cost model's load
       tie-break: the arm's own [Cuda_mma.arm_ldmatrix_operands] over each operand's eligibility.
       The census and the capability read [Cuda_mma.operand_loads], its per-statement-arm form. *)
    let mma_ldmatrix_operands arm ~ta ~tb ~a ~b =
      Cuda_mma.arm_ldmatrix_operands arm ~ta ~tb ~a:(mma_ldm a) ~b:(mma_ldm b)

    let mma_operand_loads = Cuda_mma.operand_loads
    let mma_operand_fragments = Cuda_mma.operand_fragments

    (* Shared by statement and scope for the inline-PTX arms: the intrinsic divides the block, the
       device meets the arm's floor, and every operand is plain or shared-swizzled. The 16-bit arms
       read swizzled operands with [ldmatrix]; fp8 uses it in its contiguous orientations and
       swizzle-aware byte gathers otherwise. Destination conditions remain with the callers. *)
    let mma16_operands_ok inline ~m ~n ~k ~a:(a_space, a_layout) ~b:(b_space, b_layout) =
      let arm = Cuda_mma.Inline_ptx inline in
      let tm, tn, tk = Cuda_mma.arm_tile arm in
      (mma_plain a_layout || mma_ldm (a_space, a_layout))
      && (mma_plain b_layout || mma_ldm (b_space, b_layout))
      && m % tm = 0
      && n % tn = 0
      && k % tk = 0
      && mma_loadable a_space && mma_loadable b_space
      && min_compute_capability () >= Cuda_mma.arm_floor arm

    (* Both inline-PTX shapes address and move the same four f32 accumulators. The statement
       declares scalar registers on load; a persistent boundary moves to/from its existing fragment
       array. [None] is the f32 storage identity; [Some] names the boundary conversions. *)
    let mma16_d_rows ~elt_typ ~ldd =
      [
        Printf.sprintf
          "%s *__mma_dr0 = __mma_dp + (__mi * 16 + __mma_g) * %d + __ni * 8 + 2 * __mma_t;" elt_typ
          ldd;
        Printf.sprintf "%s *__mma_dr1 = __mma_dr0 + 8 * %d;" elt_typ ldd;
      ]

    let mma16_d_moves ~dir ~reg ~declare ~d_cvt =
      let convert value =
        match d_cvt with
        | None -> value
        | Some (widen, narrow) ->
            let conversion = match dir with `Load -> widen | `Store -> narrow in
            Printf.sprintf "%s(%s)" conversion value
      in
      let move i row col =
        let cell = Printf.sprintf "__mma_dr%d[%d]" row col in
        match dir with
        | `Load ->
            Printf.sprintf "%s%s = %s;" (if declare then "float " else "") (reg i) (convert cell)
        | `Store -> Printf.sprintf "%s = %s;" cell (convert (reg i))
      in
      [ move 0 0 0 ^ " " ^ move 1 0 1; move 2 1 0 ^ " " ^ move 3 1 1 ]

    let mma16_acc resident i =
      match resident with
      | None -> Printf.sprintf "__mma_d%d" i
      | Some frag -> Printf.sprintf "%s[__mi][__ni][%d]" frag i

    let mma16_either resident ~fragment ~statement =
      match resident with Some _ -> fragment | None -> statement

    let mma16_d_statement_lines ~dir ~elt_typ ~d_cvt ~ldd =
      let rows = match dir with `Load -> mma16_d_rows ~elt_typ ~ldd | `Store -> [] in
      List.map
        (rows @ mma16_d_moves ~dir ~reg:(mma16_acc None) ~declare:true ~d_cvt)
        ~f:(fun line -> "    " ^ line)

    (* The [d] boundary of an m16n8 register fragment [frag] ([float frag[mt][nt][4]], one m16n8
       accumulator tile per [(__mi, __ni)]; gh-ocannl-1063): [`Load] fills it from the [elt] storage
       at [__mma_dp], applying [d_cvt] on load/store when storage is narrower than f32. Unlike
       [wmma_d_boundary_lines] no coordinate table is needed: the PTX ISA fixes the accumulator
       layout — with groupID g = lane>>2 and threadID-in-group t = lane&3, registers 0..3 hold rows
       {g, g+8} x columns {2t, 2t+1} — which is the same mapping the per-statement rendering in
       [mma_syntax] loads and stores through. *)
    let mma16_d_boundary_lines ~dir ~frag ~elt_typ ~d_cvt ~ldd ~mt ~nt =
      let reg i = Printf.sprintf "%s[__mi][__ni][%d]" frag i in
      [
        "{ /* mma.sync register fragment d boundary: the m16n8 f32 accumulator layout */";
        "  unsigned __mma_lid;";
        "  asm(\"mov.u32 %0, %%laneid;\" : \"=r\"(__mma_lid));";
        "  const int __mma_g = (int)(__mma_lid >> 2);";
        "  const int __mma_t = (int)(__mma_lid & 3);";
        "#pragma unroll";
        Printf.sprintf "  for (int __mi = 0; __mi < %d; ++__mi) {" mt;
        "#pragma unroll";
        Printf.sprintf "    for (int __ni = 0; __ni < %d; ++__ni) {" nt;
      ]
      @ List.map
          (mma16_d_rows ~elt_typ ~ldd @ mma16_d_moves ~dir ~reg ~declare:false ~d_cvt)
          ~f:(fun line -> "      " ^ line)
      @ [ "    }"; "  }"; "}" ]

    (* Tensorize-mma T3: cooperative tile-MMA emission for [Low_level.Tile_mma]. Two renderings:

       - The CUDA wmma C++ API for the 16-bit combinations: f16 x f16 -> f32 (the flagship), f16 x
       f16 -> f16 (under [Numerics.Fp16_auto]/[Fp16_narrow] only — the wide policy routes the
       uniform-f16 combination to the f32-accumulate inline-PTX m16n8k16 arm instead,
       gh-ocannl-680), and bf16 x bf16 -> f32 (sm_80+; the [(wmma-bf16)] marker makes
       [nvrtc_options] select the arch). 16x16x16 fragment blocks are resident across the whole [k]
       extent, mirroring the Metal emission. - Inline-PTX
       [mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32] for fp8 x fp8 -> f32 (OCANNL's fp8 is
       e5m2), which wmma cannot express. sm_89+ (Ada); the [(mma-fp8)] marker makes [nvrtc_options]
       target compute_89, whose PTX the driver forward-JIT-compiles on newer GPUs (e.g. sm_120
       Blackwell GeForce) — deliberately NOT the device arch, where the plain e5m2 encoding gives
       way to the family-specific [kind::f8f6f4] forms.

       The extent-32 lane loop binds threadIdx.x, so the 32 consecutive .x threads reaching the
       statement form the cooperating warp. Transposed operand storage ([ta]/[tb]) loads wmma
       fragments as [col_major] with swapped offset arithmetic; both inline-PTX arms index every
       gathered element at the transposed (col, row) address instead (gh-ocannl-481 item 1). Uniform
       f32 targets the wmma tf32 shape m16n16k8 on sm_80+ (the [(wmma-tf32)] marker selects the
       arch) when the numerics policy opts in ([Numerics.t.tf32_matmuls], gh-ocannl-478): tf32
       truncates the mantissa to 10 bits, so with the policy off — the default — uniform f32 stays
       on the scalar path with full f32 numerics. Declines (the barrier-bracketed lane-0 fallback
       renders instead) on: other precision combinations, extents not multiples of the intrinsic
       tile, leading dimensions violating wmma's stride constraint (a multiple of 8 elements for
       16-bit types, 4 for f32; the fp8 path loads per-lane bytes and has no stride constraint),
       thread-space operands (per-thread stacks are not a jointly-owned tile), and devices below the
       arch floor. *)
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
          (* The arm, resolved once (gh-ocannl-1232). An update-only call against a fragment array
             takes the arm of the scope that declared the array (gh-ocannl-1063, gh-ocannl-925);
             every other call is a per-statement one. Each arm's conditions below decline to the
             caller's scalar fallback, never to another arm. *)
          let arm =
            Cuda_mma.resolve_arm ~a_prec ~b_prec ~d_prec
              ~scope:
                (match d_space with
                | `Fragment _ -> Backend_intf.Mma_fragment_scope
                | `Device | `Shared | `Thread -> Backend_intf.Mma_per_statement)
          in
          let loadable = mma_loadable and plain = mma_plain in
          let a_swz = mma_ldm (a_space, a_layout) and b_swz = mma_ldm (b_space, b_layout) in
          (* Element offset in a [Swizzle_b128] tile: XOR the column's 16-byte-unit index with the
             low row bits, preserving the within-unit remainder. Both [ldmatrix] addresses and fp8
             byte gathers use this map. *)
          let b128_offset ~ld ~u ~row ~col =
            let shift = Int.floor_log2 u in
            Printf.sprintf "(%s) * %d + (((((%s) >> %d) ^ ((%s) & %d)) << %d) + ((%s) & %d))" row ld
              col shift row
              ((ld / u) - 1)
              shift col (u - 1)
          in
          (* [ldmatrix] takes a [.shared] address, converted from the generic window. Its fragment
             entry points have columns aligned to 16-byte units. *)
          let swz_saddr ~ptr ~ld ~prec ~row ~col =
            let offset = b128_offset ~ld ~u:(16 / Ops.prec_in_bytes prec) ~row ~col in
            Printf.sprintf "(unsigned)__cvta_generic_to_shared(%s + %s)" ptr offset
          in
          (* [ldmatrix.sync.aligned.m8n8.xN[.trans].shared.b16]: the first 8N lanes each supply one
             16-byte row address of the N 8x8 tiles of 16-bit elements, and the results land in
             exactly the per-lane fragment layout [mma.sync] consumes — replacing 4 (resp. 2)
             per-lane gather chains with one instruction. The remaining lanes' addresses are
             ignored; they are still computed in range. *)
          let ldmatrix_lines ~indent ~regs ~trans ~addr =
            let num = List.length regs in
            let outs =
              String.concat ~sep:", " (List.map regs ~f:(fun r -> Printf.sprintf "\"=r\"(%s)" r))
            in
            let slots =
              String.concat ~sep:"," (List.mapi regs ~f:(fun i _ -> Printf.sprintf "%%%d" i))
            in
            [
              Printf.sprintf "%sunsigned %s;" indent (String.concat ~sep:", " regs);
              Printf.sprintf "%s{ unsigned __mma_sa = %s;" indent addr;
              Printf.sprintf
                "%s  asm volatile(\"ldmatrix.sync.aligned.m8n8.x%d%s.shared.b16 {%s}, [%%%d];\" : \
                 %s : \"r\"(__mma_sa)); }"
                indent num
                (if trans then ".trans" else "")
                slots num outs;
            ]
          in
          (* Names the load path in the emitted block comment, so a reader (and the structural test
             pins) can tell which operands came in through [ldmatrix]. *)
          let ldm_tag ~a ~b =
            match (a, b) with
            | false, false -> ""
            | true, true -> " ldmatrix a,b"
            | true, false -> " ldmatrix a"
            | false, true -> " ldmatrix b"
          in
          (* gh-ocannl-1063: an update-only call against the register fragment array the inline-PTX
             arms' [mma_fragment_syntax] scope declared. *)
          let resident =
            match (arm, d_space) with
            | Some (Cuda_mma.Inline_ptx _), `Fragment frag -> Some frag
            | _, (`Fragment _ | `Device | `Shared | `Thread) -> None
          in
          (* fp8's [ldmatrix] eligibility is one-sided per operand, and the sides are opposite
             (gh-ocannl-481 item 3, D2). [ldmatrix.b16] moves 16-bit units, so it can build a
             fragment register only when the 4 fp8 bytes that register holds are CONTIGUOUS in
             storage. For A those 4 bytes are 4 consecutive [k] at fixed [m] — contiguous exactly
             when A is stored row-major ([ta = false]); for B they are 4 consecutive [k] at fixed
             [n] — contiguous exactly when B is stored transposed ([tb = true]). The other
             orientations keep byte gathers through the tile's XOR-swizzled address map. This lets
             BOTH row-major staged operands be swizzled: A uses [ldmatrix], B gathers its strided
             bytes, and both twins share the accumulator scope. *)
          let a_ldm, b_ldm =
            match arm with
            | Some arm ->
                mma_ldmatrix_operands arm ~ta ~tb ~a:(a_space, a_layout) ~b:(b_space, b_layout)
            | None -> (false, false)
          in
          match arm with
          | None -> None
          | Some (Cuda_mma.Inline_ptx (Cuda_mma.Fp8_m16n8k32 as inline) as arm) ->
              if
                plain d_layout
                && (loadable d_space || Option.is_some resident)
                && mma16_operands_ok inline ~m ~n ~k ~a:(a_space, a_layout) ~b:(b_space, b_layout)
              then
                (* Raw [mma.sync] with the architecture-defined per-lane fragment layouts of
                   m16n8k32 (PTX ISA "Matrix Fragments for mma.m16n8k32"; fp8 shares the .s8/.u8
                   layouts). Thread [lane], with groupID g = lane>>2 and threadID-in-group t =
                   lane&3, holds: A (16x32, row-major source) regs a0..a3 = 4 consecutive bytes at
                   rows {g, g+8} x column groups {4t, 4t+16}; B (32x8, column-major fragment from
                   our row-major source) regs b0,b1 = rows {4t..4t+3, 16+4t..16+4t+3} down column g,
                   gathered bytewise (stride ldb); accumulator D (16x8 f32) regs d0..d3 = rows {g,
                   g+8} x columns {2t, 2t+1}. Low-indexed elements sit in low-order bytes of each
                   .b32 register. Byte loads impose no alignment or stride constraints and read
                   generic addresses, covering both __shared__ tiles and device pointers.

                   Transposed operand storage ([ta]/[tb], gh-ocannl-481 item 1) is bookkeeping here,
                   not a layout constraint: every fragment byte is addressed by its logical (row,
                   col) through [a_at]/[b_at], which index the STORED matrix at (col, row) under the
                   flag while keeping the operand's own leading dimension. The fragment content is
                   unchanged, only its addresses are — so the gradient GEMMs ([dA = g.B^T], [dB =
                   A^T.g]), whose layouts are exactly these, tensorize instead of silently
                   scalar-falling-back. *)
                let open PPrint in
                let mt = m / 16 and nt = n / 8 and kt = k / 32 in
                let pack4 name base offs =
                  Printf.sprintf
                    "unsigned %s = (unsigned)%s[%s] | ((unsigned)%s[%s] << 8) | ((unsigned)%s[%s] \
                     << 16) | ((unsigned)%s[%s] << 24);"
                    name base offs.(0) base offs.(1) base offs.(2) base offs.(3)
                in
                (* Byte offset of logical element (row, col) in an operand stored under its own
                   leading dimension, transposed or not, then through the b128 map when swizzled.
                   fp8 is one byte per element, so element index = byte offset from the [unsigned
                   char *] base. *)
                let elem_at ~ld ~transposed ~swizzled ~row ~col =
                  let row, col = if transposed then (col, row) else (row, col) in
                  if swizzled then b128_offset ~ld ~u:16 ~row ~col
                  else Printf.sprintf "(%s) * %d + (%s)" row ld col
                in
                let a_at ~row ~col = elem_at ~ld:lda ~transposed:ta ~swizzled:a_swz ~row ~col in
                let b_at ~row ~col = elem_at ~ld:ldb ~transposed:tb ~swizzled:b_swz ~row ~col in
                let a_row lo = if lo then "__mi * 16 + __mma_g" else "__mi * 16 + __mma_g + 8" in
                let a_col c = Printf.sprintf "__ki * 32 + 4 * __mma_t + %d" c in
                let b_row r = Printf.sprintf "__ki * 32 + 4 * __mma_t + %d" r in
                let b_col = "__ni * 8 + __mma_g" in
                (* A regs: 4 consecutive columns at rows {g, g+8} of column groups {4t, 4t+16}. *)
                let a_reg name ~lo ~c =
                  "      "
                  ^ pack4 name "__mma_ap"
                      (Array.init 4 ~f:(fun i -> a_at ~row:(a_row lo) ~col:(a_col (c + i))))
                in
                (* B regs: 4 consecutive rows of the column-major fragment column [g]. *)
                let b_reg name ~r =
                  "      "
                  ^ pack4 name "__mma_bp"
                      (Array.init 4 ~f:(fun i -> b_at ~row:(b_row (r + i)) ~col:b_col))
                in
                (* The [ldmatrix] entry points, in the 16-bit view of the fp8 bytes: A's four 8x8
                   matrices are (m rows {0-7, 8-15}) x (byte columns {0-15, 16-31}), which is the
                   [m16n8k16] A arrangement at byte granularity; B stored transposed is (n rows 0-7)
                   x (byte columns {0-15, 16-31}), giving b0/b1 in that order. With [q = lane >> 3]
                   naming the matrix, lane [r = lane & 7] supplies row [r] of matrix [q]. *)
                let a_ldm_lines =
                  ldmatrix_lines ~indent:"      "
                    ~regs:[ "__mma_a0"; "__mma_a1"; "__mma_a2"; "__mma_a3" ]
                    ~trans:false
                    ~addr:
                      (swz_saddr ~ptr:"__mma_ap" ~ld:lda ~prec:a_prec
                         ~row:"__mi * 16 + (__mma_lid & 7) + 8 * ((__mma_lid >> 3) & 1)"
                         ~col:"__ki * 32 + 16 * (__mma_lid >> 4)")
                in
                let b_ldm_lines =
                  ldmatrix_lines ~indent:"      " ~regs:[ "__mma_b0"; "__mma_b1" ] ~trans:false
                    ~addr:
                      (swz_saddr ~ptr:"__mma_bp" ~ld:ldb ~prec:b_prec
                         ~row:"__ni * 8 + (__mma_lid & 7)"
                         ~col:"__ki * 32 + 16 * ((__mma_lid >> 3) & 1)")
                in
                let barrier = "__syncthreads();" in
                let either ~fragment ~statement = mma16_either resident ~fragment ~statement in
                let acc = mma16_acc resident in
                let unroll = either ~fragment:[ "#pragma unroll" ] ~statement:[] in
                let body_lines =
                  either ~fragment:[] ~statement:[ barrier ]
                  @ [
                      "unsigned __mma_lid;";
                      "asm(\"mov.u32 %0, %%laneid;\" : \"=r\"(__mma_lid));";
                      "const int __mma_g = (int)(__mma_lid >> 2);";
                      "const int __mma_t = (int)(__mma_lid & 3);";
                    ]
                  @ unroll
                  @ [ Printf.sprintf "for (int __mi = 0; __mi < %d; ++__mi) {" mt ]
                  @ unroll
                  @ [ Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt ]
                  @ either ~fragment:[]
                      ~statement:
                        (mma16_d_statement_lines ~dir:`Load ~elt_typ:"float" ~d_cvt:None ~ldd)
                  @ [ Printf.sprintf "    for (int __ki = 0; __ki < %d; ++__ki) {" kt ]
                  @ (if a_ldm then a_ldm_lines
                     else
                       [
                         a_reg "__mma_a0" ~lo:true ~c:0;
                         a_reg "__mma_a1" ~lo:false ~c:0;
                         a_reg "__mma_a2" ~lo:true ~c:16;
                         a_reg "__mma_a3" ~lo:false ~c:16;
                       ])
                  @ (if b_ldm then b_ldm_lines
                     else [ b_reg "__mma_b0" ~r:0; b_reg "__mma_b1" ~r:16 ])
                  @ [
                      "      asm(\"mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 \"";
                      "          \"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"";
                      Printf.sprintf "          : \"+f\"(%s), \"+f\"(%s), \"+f\"(%s), \"+f\"(%s)"
                        (acc 0) (acc 1) (acc 2) (acc 3);
                      "          : \"r\"(__mma_a0), \"r\"(__mma_a1), \"r\"(__mma_a2), \
                       \"r\"(__mma_a3), \"r\"(__mma_b0), \"r\"(__mma_b1));";
                      "    }";
                    ]
                  @ either ~fragment:[]
                      ~statement:
                        (mma16_d_statement_lines ~dir:`Store ~elt_typ:"float" ~d_cvt:None ~ldd)
                  @ [ "  }"; "}"; barrier ]
                in
                let cast_ptr name ptr =
                  string (Printf.sprintf "const unsigned char *%s = (const unsigned char *)(" name)
                  ^^ ptr ^^ string ");"
                in
                let body ~a_ptr ~b_ptr =
                  either ~fragment:empty
                    ~statement:(string "float *__mma_dp = " ^^ d_ptr ^^ semi ^^ hardline)
                  ^^ cast_ptr "__mma_ap" a_ptr ^^ hardline ^^ cast_ptr "__mma_bp" b_ptr ^^ hardline
                  ^^ separate_map hardline string body_lines
                in
                Some
                  (fun ~a_ptr ~b_ptr ->
                    group
                      (string
                         (Printf.sprintf "{ /* tile_mma %s%dx%dx%d (%s) e5m2%s */"
                            (either ~fragment:"fragment update " ~statement:"")
                            m n k (Cuda_mma.arm_marker arm) (ldm_tag ~a:a_ldm ~b:b_ldm))
                      ^^ nest 2 (hardline ^^ body ~a_ptr ~b_ptr)
                      ^^ hardline ^^ rbrace))
              else None
          | Some (Cuda_mma.Inline_ptx (Cuda_mma.M16n8k16 kind as inline) as arm) ->
              if
                (* Unlike fp8, both 16-bit operands can come in through [ldmatrix] in either storage
                   orientation: the fragment registers hold 16-bit element PAIRS, and the pair a
                   lane needs is contiguous under one of the two forms — [.trans] transposes each
                   8x8 tile on distribution, which is exactly the difference between the two
                   orientations. *)
                plain d_layout
                && (loadable d_space || Option.is_some resident)
                && mma16_operands_ok inline ~m ~n ~k ~a:(a_space, a_layout) ~b:(b_space, b_layout)
              then
                (* Raw [mma.sync] with the architecturally-defined per-lane fragment layouts of
                   m16n8k16 (PTX ISA "Matrix Fragments for mma.m16n8k16", shared by .f16 and .bf16 —
                   which is what lets [Cuda_mma.m16n8k16_spelling] parameterize this one body over
                   both element types). Thread [lane], with groupID g = lane>>2 and
                   threadID-in-group t = lane&3, holds: A (16x16) regs a0..a3 = the element pairs at
                   rows {g, g+8} x column pairs {2t, 2t+8}; B (16x8, column-major fragment) regs
                   b0,b1 = row pairs {2t, 2t+8} down column g; accumulator D (16x8 f32) regs d0..d3
                   = rows {g, g+8} x columns {2t, 2t+1}. The lower-indexed element of each pair sits
                   in the low half of its .b32 register.

                   Unlike the fp8 arm, transposed operand storage ([ta]/[tb]) is supported: every
                   gather goes through [a_at]/[b_at], which index the STORED matrix at (col, row)
                   under the flag while keeping the operand's own leading dimension — the fragment
                   content is unchanged, only its addresses are. Element-wise 16-bit loads impose no
                   alignment or stride constraint and read generic addresses, so both __shared__
                   tiles and device pointers are covered.

                   The accumulator is read from and written back to the 16-bit [d] once per
                   statement, with the whole [k] extent accumulated in f32 registers in between. For
                   bf16 that is strictly better rounding than the scalar fallback this replaces,
                   which rounds to bf16 per term; for f16 it is the [Numerics.Fp16_wide] residency
                   [accum_prec] gives every serial rendering (gh-ocannl-680).

                   Against a [resident] register fragment (gh-ocannl-1063) the call is update-only:
                   the f32 registers are [frag[__mi][__ni][0..3]], loaded and stored once by the
                   enclosing [mma_fragment_syntax] scope, so no [d] crosses a 16-bit boundary per
                   outer [k] block. The leading barrier goes too — no sibling zeroing of [d] is read
                   here, and the staged tiles' visibility is the staging's own barrier, as for the
                   wmma update form — while the trailing one stays: it releases the staged tiles for
                   the next serial iteration's cooperative loads. *)
                let open PPrint in
                let {
                  Cuda_mma.ms_elt_typ = elt_typ;
                  ms_as_ushort = as_ushort;
                  ms_widen = to_float;
                  ms_narrow = from_float;
                  ms_infix = instr_elt;
                } =
                  Cuda_mma.m16n8k16_spelling kind
                in
                let marker = Cuda_mma.arm_marker arm in
                let mt = m / 16 and nt = n / 8 and kt = k / 16 in
                (* Address of logical element (row, col) in an operand stored under its own leading
                   dimension, transposed or not. *)
                let elem_at ~ld ~transposed ~row ~col =
                  if transposed then Printf.sprintf "(%s) * %d + (%s)" col ld row
                  else Printf.sprintf "(%s) * %d + (%s)" row ld col
                in
                let a_at ~row ~col = elem_at ~ld:lda ~transposed:ta ~row ~col in
                let b_at ~row ~col = elem_at ~ld:ldb ~transposed:tb ~row ~col in
                let pack2 name base i0 i1 =
                  Printf.sprintf
                    "unsigned %s = (unsigned)%s(%s[%s]) | ((unsigned)%s(%s[%s]) << 16);" name
                    as_ushort base i0 as_ushort base i1
                in
                let a_row lo = if lo then "__mi * 16 + __mma_g" else "__mi * 16 + __mma_g + 8" in
                let a_col c = Printf.sprintf "__ki * 16 + 2 * __mma_t + %d" c in
                let b_row r = Printf.sprintf "__ki * 16 + 2 * __mma_t + %d" r in
                let b_col = "__ni * 8 + __mma_g" in
                let a_reg name ~lo ~c =
                  "      "
                  ^ pack2 name "__mma_ap"
                      (a_at ~row:(a_row lo) ~col:(a_col c))
                      (a_at ~row:(a_row lo) ~col:(a_col (c + 1)))
                in
                let b_reg name ~r =
                  "      "
                  ^ pack2 name "__mma_bp"
                      (b_at ~row:(b_row r) ~col:b_col)
                      (b_at ~row:(b_row (r + 1)) ~col:b_col)
                in
                (* The [ldmatrix] entry points. Naming the matrix [q = lane >> 3] and the row within
                   it [r = lane & 7], the four A matrices are (m rows {0-7, 8-15}) x (k columns
                   {0-7, 8-15}) in register order a0..a3, and the two B matrices are (k rows {0-7,
                   8-15}) x (n columns 0-7) in order b0, b1. Each is read as 8 rows of 8 contiguous
                   16-bit elements from the operand's OWN storage, so the transposed orientations
                   swap which index walks the rows and add [.trans] exactly when the fragment's
                   minor direction is the stored major one. *)
                let a_ldm_lines =
                  ldmatrix_lines ~indent:"      "
                    ~regs:[ "__mma_a0"; "__mma_a1"; "__mma_a2"; "__mma_a3" ]
                    ~trans:ta
                    ~addr:
                      (if ta then
                         swz_saddr ~ptr:"__mma_ap" ~ld:lda ~prec:a_prec
                           ~row:"__ki * 16 + (__mma_lid & 7) + 8 * (__mma_lid >> 4)"
                           ~col:"__mi * 16 + 8 * ((__mma_lid >> 3) & 1)"
                       else
                         swz_saddr ~ptr:"__mma_ap" ~ld:lda ~prec:a_prec
                           ~row:"__mi * 16 + (__mma_lid & 7) + 8 * ((__mma_lid >> 3) & 1)"
                           ~col:"__ki * 16 + 8 * (__mma_lid >> 4)")
                in
                let b_ldm_lines =
                  ldmatrix_lines ~indent:"      " ~regs:[ "__mma_b0"; "__mma_b1" ] ~trans:(not tb)
                    ~addr:
                      (if tb then
                         swz_saddr ~ptr:"__mma_bp" ~ld:ldb ~prec:b_prec
                           ~row:"__ni * 8 + (__mma_lid & 7)"
                           ~col:"__ki * 16 + 8 * ((__mma_lid >> 3) & 1)"
                       else
                         swz_saddr ~ptr:"__mma_bp" ~ld:ldb ~prec:b_prec
                           ~row:"__ki * 16 + (__mma_lid & 15)" ~col:"__ni * 8")
                in
                let barrier = "__syncthreads();" in
                (* Which of two renderings: against the scope's register [fragment], or the
                   self-contained [statement] that loads and stores [d] itself. *)
                let either ~fragment ~statement = mma16_either resident ~fragment ~statement in
                let acc = mma16_acc resident in
                let unroll = either ~fragment:[ "#pragma unroll" ] ~statement:[] in
                let body_lines =
                  either ~fragment:[] ~statement:[ barrier ]
                  @ [
                      "unsigned __mma_lid;";
                      "asm(\"mov.u32 %0, %%laneid;\" : \"=r\"(__mma_lid));";
                      "const int __mma_g = (int)(__mma_lid >> 2);";
                      "const int __mma_t = (int)(__mma_lid & 3);";
                    ]
                  @ unroll
                  @ [ Printf.sprintf "for (int __mi = 0; __mi < %d; ++__mi) {" mt ]
                  @ unroll
                  @ [ Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt ]
                  @ either ~fragment:[]
                      ~statement:
                        (mma16_d_statement_lines ~dir:`Load ~elt_typ
                           ~d_cvt:(Some (to_float, from_float))
                           ~ldd)
                  @ [ Printf.sprintf "    for (int __ki = 0; __ki < %d; ++__ki) {" kt ]
                  @ (if a_ldm then a_ldm_lines
                     else
                       [
                         a_reg "__mma_a0" ~lo:true ~c:0;
                         a_reg "__mma_a1" ~lo:false ~c:0;
                         a_reg "__mma_a2" ~lo:true ~c:8;
                         a_reg "__mma_a3" ~lo:false ~c:8;
                       ])
                  @ (if b_ldm then b_ldm_lines else [ b_reg "__mma_b0" ~r:0; b_reg "__mma_b1" ~r:8 ])
                  @ [
                      Printf.sprintf
                        "      asm(\"mma.sync.aligned.m16n8k16.row.col.f32.%s.%s.f32 \"" instr_elt
                        instr_elt;
                      "          \"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"";
                      Printf.sprintf "          : \"+f\"(%s), \"+f\"(%s), \"+f\"(%s), \"+f\"(%s)"
                        (acc 0) (acc 1) (acc 2) (acc 3);
                      "          : \"r\"(__mma_a0), \"r\"(__mma_a1), \"r\"(__mma_a2), \
                       \"r\"(__mma_a3), \"r\"(__mma_b0), \"r\"(__mma_b1));";
                      "    }";
                    ]
                  @ either ~fragment:[]
                      ~statement:
                        (mma16_d_statement_lines ~dir:`Store ~elt_typ
                           ~d_cvt:(Some (to_float, from_float))
                           ~ldd)
                  @ [ "  }"; "}"; barrier ]
                in
                let ptr_decl name typ ptr =
                  string (Printf.sprintf "%s *%s = " typ name) ^^ ptr ^^ semi
                in
                let body ~a_ptr ~b_ptr =
                  either ~fragment:empty ~statement:(ptr_decl "__mma_dp" elt_typ d_ptr ^^ hardline)
                  ^^ ptr_decl "__mma_ap" ("const " ^ elt_typ) a_ptr
                  ^^ hardline
                  ^^ ptr_decl "__mma_bp" ("const " ^ elt_typ) b_ptr
                  ^^ hardline
                  ^^ separate_map hardline string body_lines
                in
                Some
                  (fun ~a_ptr ~b_ptr ->
                    group
                      (string
                         (Printf.sprintf "{ /* tile_mma %s%dx%dx%d (%s)%s */"
                            (either ~fragment:"fragment update " ~statement:"")
                            m n k marker (ldm_tag ~a:a_ldm ~b:b_ldm))
                      ^^ nest 2 (hardline ^^ body ~a_ptr ~b_ptr)
                      ^^ hardline ^^ rbrace))
              else None
          | Some (Cuda_mma.Wmma kind as arm) -> (
              (* wmma fragments are opaque: [load_matrix_sync] assumes row-major (or column-major)
                 pointer+stride storage and there is no supported way to feed one from [ldmatrix]
                 destination registers. So the whole template path declines swizzled operands
                 (gh-ocannl-481 item 3, D2) — the caller then reaches for the scalar fallback. That
                 and the arm's extent, stride, space and floor conditions are
                 [Cuda_mma.wmma_accepts], which [mma_fragment_syntax] decides by too. *)
              let accepts d =
                mma_wmma_accepts kind ~m ~n ~k ~d ~a:(lda, a_space, a_layout)
                  ~b:(ldb, b_space, b_layout)
              in
              let { Cuda_mma.wc_acc_prec = acc_prec; wc_d_cvt = d_cvt; _ } =
                Cuda_mma.wmma_info kind
              in
              let wc_tm, wc_tn, _ = Cuda_mma.arm_tile arm in
              let marker = Cuda_mma.arm_marker arm in
              let open PPrint in
              let mt = m / wc_tm and nt = n / wc_tn in
              let frag = Cuda_mma.wmma_frag_typ kind in
              let ptr_decl name typ ptr =
                string (Printf.sprintf "%s *%s = " typ name) ^^ ptr ^^ semi
              in
              let barrier = "__syncthreads();" in
              let k_loop_lines ~acc =
                Cuda_mma.wmma_k_loop_lines kind ~acc ~ta ~tb ~lda ~ldb ~m ~n ~k
              in
              match d_space with
              | `Fragment fragment when accepts (`Resident d_layout) ->
                  (* Update-only steps against the accumulator-fragment array [fragment] declared by
                     the enclosing [mma_fragment_syntax] scope (gh-ocannl-480): no per-statement
                     load/store of [d]. The trailing barrier releases the staged shared tiles for
                     the next serial iteration's cooperative loads. *)
                  let body ~a_ptr ~b_ptr =
                    ptr_decl "__mma_ap" ("const " ^ typ_of_prec a_prec) a_ptr
                    ^^ hardline
                    ^^ ptr_decl "__mma_bp" ("const " ^ typ_of_prec b_prec) b_ptr
                    ^^ hardline
                    ^^ separate_map hardline string (k_loop_lines ~acc:fragment @ [ barrier ])
                  in
                  Some
                    (fun ~a_ptr ~b_ptr ->
                      group
                        (string
                           (Printf.sprintf "{ /* tile_mma fragment update %dx%dx%d (%s) */" m n k
                              marker)
                        ^^ nest 2 (hardline ^^ body ~a_ptr ~b_ptr)
                        ^^ hardline ^^ rbrace))
              (* A converted destination (gh-ocannl-925) renders only against the fragment array
                 above: the per-statement scope never resolves to such an arm (the wide-f16
                 statement is the inline-PTX m16n8k16 arm's), so the guard only keeps a
                 self-contained rendering from ever being minted for one. *)
              | (`Device | `Shared | `Thread) as d_space
                when Option.is_none d_cvt && accepts (`Addressed (ldd, (d_space, d_layout))) ->
                  let body_lines =
                    [
                      barrier;
                      Printf.sprintf "%s __mma_acc[%d][%d];"
                        (frag "accumulator" (typ_of_prec acc_prec) None)
                        mt nt;
                      Printf.sprintf "for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                      Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                      Printf.sprintf
                        "    nvcuda::wmma::load_matrix_sync(__mma_acc[__mi][__ni], __mma_dp + __mi \
                         * %d * %d + __ni * %d, %d, nvcuda::wmma::mem_row_major);"
                        wc_tm ldd wc_tn ldd;
                      "  }";
                      "}";
                    ]
                    @ k_loop_lines ~acc:"__mma_acc"
                    @ [
                        Printf.sprintf "for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                        Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                        Printf.sprintf
                          "    nvcuda::wmma::store_matrix_sync(__mma_dp + __mi * %d * %d + __ni * \
                           %d, __mma_acc[__mi][__ni], %d, nvcuda::wmma::mem_row_major);"
                          wc_tm ldd wc_tn ldd;
                        "  }";
                        "}";
                        barrier;
                      ]
                  in
                  let body ~a_ptr ~b_ptr =
                    ptr_decl "__mma_dp" (typ_of_prec d_prec) d_ptr
                    ^^ hardline
                    ^^ ptr_decl "__mma_ap" ("const " ^ typ_of_prec a_prec) a_ptr
                    ^^ hardline
                    ^^ ptr_decl "__mma_bp" ("const " ^ typ_of_prec b_prec) b_ptr
                    ^^ hardline
                    ^^ separate_map hardline string body_lines
                  in
                  Some
                    (fun ~a_ptr ~b_ptr ->
                      group
                        (string (Printf.sprintf "{ /* tile_mma %dx%dx%d (%s) */" m n k marker)
                        ^^ nest 2 (hardline ^^ body ~a_ptr ~b_ptr)
                        ^^ hardline ^^ rbrace))
              | _ -> None))

    (* Cross-[k_o] accumulator residency (gh-ocannl-480), following the Metal emission: the marked
       local tile becomes a wmma accumulator-fragment array declared once, loaded from the backing
       target before the serial reduction body and stored after it; the nested [Tile_mma] sees
       [`Fragment] and emits update-only mma steps. Both hooks resolve that inner call to this
       scope's arm ([Cuda_mma.resolve_arm] at [Mma_fragment_scope]), and the acceptance conditions
       mirror [mma_syntax]'s wmma arm exactly (the same extent/stride/space/arch checks), so an
       accepted scope never strands its inner call. Swizzled operands decline the wmma rendering
       because opaque fragments cannot be fed from [ldmatrix].

       Uniform bf16 and fp8 instead render as the inline-PTX arm's own per-lane registers held
       across the reduction ([Cuda_mma.inline_ptx_d_storage]): [float frag[mt][nt][4]] in the
       architected m16n8 f32 accumulator layout shared by m16n8k16 and m16n8k32. The destination
       boundary converts bf16 or directly moves f32 once at each end, with nested calls rendering
       update-only. Plain and swizzled staged twins share this scope regardless of how their operand
       registers arrive. *)
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
          let loadable = mma_loadable and plain = mma_plain in
          let barrier = "__syncthreads();" in
          let d_decl =
            PPrint.(string (Printf.sprintf "%s *__mma_dp = " (typ_of_prec d_prec)) ^^ d_ptr ^^ semi)
          in
          (* Bracketing barriers: sibling statements (the zeroing of [d], cooperative staging)
             execute lane-partitioned, so the fragment loads must observe the other lanes' writes,
             and later statements the stores. *)
          let scope_doc ~title ~lines_before ~lines_after =
            let open PPrint in
            group
              (string title
              ^^ nest 2
                   (hardline ^^ d_decl ^^ hardline
                   ^^ separate_map hardline string lines_before
                   ^^ hardline ^^ body () ^^ hardline
                   ^^ separate_map hardline string lines_after)
              ^^ hardline ^^ rbrace)
          in
          match
            Cuda_mma.resolve_arm ~a_prec ~b_prec ~d_prec ~scope:Backend_intf.Mma_fragment_scope
          with
          | Some (Cuda_mma.Inline_ptx inline as arm)
            when plain d_layout && loadable d_space
                 && mma16_operands_ok inline ~m ~n ~k ~a:(a_space, a_layout) ~b:(b_space, b_layout)
            ->
              let elt_typ, d_cvt = Cuda_mma.inline_ptx_d_storage inline in
              let marker = Cuda_mma.arm_marker arm in
              let mt = m / 16 and nt = n / 8 in
              let boundary dir =
                mma16_d_boundary_lines ~dir ~frag:fragment ~elt_typ ~d_cvt ~ldd ~mt ~nt
              in
              Some
                (scope_doc
                   ~title:
                     (Printf.sprintf "{ /* mma.sync register fragment %dx%d across k_o (%s) */" m n
                        marker)
                   ~lines_before:
                     ([ barrier; Printf.sprintf "float %s[%d][%d][4];" fragment mt nt ]
                     @ boundary `Load
                     @ [ "/* mma.sync fragment reduction body begins */" ])
                   ~lines_after:
                     (("/* mma.sync fragment reduction body ends */" :: boundary `Store)
                     @ [ barrier ]))
          | Some (Cuda_mma.Wmma kind as arm)
            when match d_space with
                 | `Fragment _ -> false
                 | (`Device | `Shared | `Thread) as d_space ->
                     mma_wmma_accepts kind ~m ~n ~k
                       ~d:(`Addressed (ldd, (d_space, d_layout)))
                       ~a:(lda, a_space, a_layout) ~b:(ldb, b_space, b_layout) ->
              let { Cuda_mma.wc_acc_prec = acc_prec; wc_d_cvt = d_cvt; _ } =
                Cuda_mma.wmma_info kind
              in
              let wc_tm, wc_tn, wc_tk = Cuda_mma.arm_tile arm in
              let mt = m / wc_tm and nt = n / wc_tn in
              let acc_frag =
                Printf.sprintf "nvcuda::wmma::fragment<nvcuda::wmma::accumulator, %d, %d, %d, %s>"
                  wc_tm wc_tn wc_tk (typ_of_prec acc_prec)
              in
              (* The [d] boundary: the fragment array's own loads and stores, or — for a destination
                 whose storage type is not the accumulator's (gh-ocannl-925) — the element-wise
                 conversion through the layout-independent coordinate table. *)
              let boundary dir =
                match d_cvt with
                | Some (widen, narrow) ->
                    wmma_d_boundary_lines ~dir ~acc_frag ~frag:fragment ~widen ~narrow ~ldd ~mt ~nt
                | None ->
                    [
                      Printf.sprintf "for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                      Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                      (match dir with
                      | `Load ->
                          Printf.sprintf
                            "    nvcuda::wmma::load_matrix_sync(%s[__mi][__ni], __mma_dp + __mi * \
                             %d * %d + __ni * %d, %d, nvcuda::wmma::mem_row_major);"
                            fragment wc_tm ldd wc_tn ldd
                      | `Store ->
                          Printf.sprintf
                            "    nvcuda::wmma::store_matrix_sync(__mma_dp + __mi * %d * %d + __ni \
                             * %d, %s[__mi][__ni], %d, nvcuda::wmma::mem_row_major);"
                            wc_tm ldd wc_tn fragment ldd);
                      "  }";
                      "}";
                    ]
              in
              let lines_before =
                [ barrier; Printf.sprintf "%s %s[%d][%d];" acc_frag fragment mt nt ]
                @ boundary `Load
                @ [ "/* wmma fragment reduction body begins */" ]
              in
              let lines_after =
                ("/* wmma fragment reduction body ends */" :: boundary `Store) @ [ barrier ]
              in
              Some
                (scope_doc
                   ~title:(Printf.sprintf "{ /* wmma fragment %dx%d across k_o */" m n)
                   ~lines_before ~lines_after)
          | _ -> None)

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
      | Double_prec _, Half_prec _ -> ("__double2half(", ")")
      | Single_prec _, Half_prec _ -> ("__float2half(", ")")
      | Byte_prec _, Half_prec _ -> ("__ushort2half_rn((unsigned short int)", ")")
      | Double_prec _, Uint4x32_prec _ -> ("double_to_uint4x32(", ")")
      | Single_prec _, Uint4x32_prec _ -> ("single_to_uint4x32(", ")")
      (* __nv_fp8_e5m2's constructors are all explicit, so the [.v[0]] arm below would not
         implicitly convert on assignment; spell out the (numeric) constructor call. *)
      | Uint4x32_prec _, Bfloat16_prec _ -> ("__ushort_as_bfloat16((unsigned short)((", ").v[0]))")
      | Uint4x32_prec _, Fp8_prec _ -> ("(__nv_fp8_e5m2)((", ").v[0])")
      | Uint4x32_prec _, _ -> ("", ".v[0]")
      | Byte_prec _, Uint4x32_prec _ -> ("byte_to_uint4x32(", ")")
      | Uint16_prec _, Uint4x32_prec _ -> ("uint16_to_uint4x32(", ")")
      | Bfloat16_prec _, Uint4x32_prec _ -> ("bfloat16_to_uint4x32(", ")")
      | Half_prec _, Uint4x32_prec _ -> ("half_to_uint4x32(", ")")
      | Fp8_prec _, Uint4x32_prec _ -> ("fp8_to_uint4x32(", ")")
      | ( Fp8_prec _,
          (Byte_prec _ | Uint16_prec _ | Int32_prec _ | Uint32_prec _ | Int64_prec _ | Uint64_prec _)
        ) ->
          (* __nv_fp8_e5m2 has no integer conversion operators: a C-style cast to an integer type
             would resolve through the saturating [operator unsigned char] (wrong for negative
             values). Convert via float, like the CC backend. *)
          ("(" ^ typ_of_prec to_ ^ ")((float)(", "))")
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
      | _ -> ("(" ^ typ_of_prec to_ ^ ")(", ")")
  end

  let codegen_capabilities () =
    let module Config = Cuda_syntax_config (struct
      let procs = [||]
    end) in
    C_syntax.codegen_capabilities (module Config)

  let cuda_includes () =
    cuda_header_prelude
    ^
    if Utils.debug_log_from_routines () then "\n__device__ int printf (const char * format, ... );"
    else ""

  module Compile = C_syntax.Compile_driver (Cuda_syntax_config)

  let next_program_id = Atomic.make 0

  let new_program ptx =
    {
      program_id = Atomic.fetch_and_add next_program_id 1;
      ptx;
      modules = [];
      modules_lock = Stdlib.Mutex.create ();
    }

  (* {2 The kernel code cache (gh-ocannl-1529)}

     The HIP backend's cache (gh-ocannl-1314), ported: a search candidate re-renders the whole
     routine and typically changes a handful of its kernels, so the batch path compiles only the
     kernels no earlier compile in this process produced -- its cache misses -- as ONE program, and
     reuses the rest. On CUDA the case rests on per-KERNEL cost rather than hiprtc's per-program
     overhead. Measured on rog (sm_120, nvrtc 13.4, driver cache disabled) on a [gpt2_mini_train]
     step's 269 kernels: nvrtc costs 18 ms per program plus 12.9 ms per kernel, the driver's PTX JIT
     4.6 ms per kernel. A cold search's candidates spent a median 1.42 s in nvrtc and 0.95 s in the
     JIT of a 2.77 s compile, compiling all 261 kernels where a sketch changes about 12. Every
     kernel as its own program costs 2.6x the joint program, and nvrtc's threads recover only 1.7x
     of that, so compiling FEWER kernels in one program is the lever, as on HIP.

     A kernel's key is its standalone source ([render_batch]'s [assemble [i]] with the conditional
     includes applied: the includes, the builtins it uses, its definition, including its name) with
     its minted names renamed by first occurrence ([Kernel_code_cache.canonical_symbols]), and the
     complete nvrtc option list that source compiles under ([nvrtc_options]: the arch the device set
     and the source's markers floor, the include paths, the numerics flags, the debug flag). Misses
     with different option lists compile as separate programs, so a kernel's PTX always came from
     its own option list. The reused PTX was compiled from the same kernel text up to that renaming
     -- a bijection on names the kernel owns -- possibly beside other kernels: the builtins are
     [__forceinline__] helpers and [__constant__] tables, so a companion changes nothing it
     computes. Process-local and bounded by [cuda_kernel_cache_mib] (PTX bytes; each retained
     program also keeps its loaded modules, see [program]); 0 disables it, and every batch compiles
     as one program of all its kernels, as before. Parameters, bindings and launch geometry always
     come from the current rendering. *)
  let kernel_cache_capacity_bytes =
    lazy
      (let mib = Utils.get_global_arg ~default:"64" ~arg_name:"cuda_kernel_cache_mib" in
       match Int.of_string_opt (String.strip mib) with
       | Some m when m >= 0 -> m * 1024 * 1024
       | _ ->
           raise
           @@ Utils.User_error
                ("cuda_kernel_cache_mib must be a non-negative integer (MiB); found: " ^ mib))

  let kernel_cache : program Kernel_code_cache.t =
    Kernel_code_cache.create ~capacity_bytes:(fun () -> Lazy.force kernel_cache_capacity_bytes)

  (* Held from the cache lookups to the publication of what they missed, so two domains compiling
     the same kernels cannot both miss and both compile them: the second finds the first's PTX.
     nvrtc's threads overlap little (8 threads recovered 1.7x on separate programs), so serializing
     here gives up little. *)
  let kernel_compile_lock = Stdlib.Mutex.create ()

  let compile_kernels ~batch ~names bindings lowereds =
    Compile_cost.record_reuse_capable ();
    let r =
      Compile.render_batch ~names bindings lowereds ~includes:(cuda_includes ())
        ~builtins:Builtins_cuda.builtins
        ~conditional_includes:Cuda_like_config.Cuda.conditional_includes ()
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
    (* The routine's full source, as codegen emitted it, is its [.cu] artifact whatever was
       compiled: the debug view of what codegen emitted (and what tests read) does not depend on
       what the cache held. The renaming is the key's alone; nvrtc compiles the emitted text. *)
    let full_source = with_conditional_includes (r.assemble all) in
    write_source_artifact ~name full_source;
    let programs = Array.create ~len:n None in
    (* Not under device debugging: PTX debug info names the source file it was compiled from, which
       a later compile of the same routine rewrites, so reuse would point a debugger at text that is
       not the code's. Every kernel compiles, as with the cache off. *)
    let enabled = Kernel_code_cache.enabled kernel_cache && not (Utils.with_runtime_debug ()) in
    let keyed =
      if not enabled then [||]
      else
        Array.init n ~f:(fun i ->
            let source = with_conditional_includes (r.assemble [ i ]) in
            let options = nvrtc_options source in
            (* Keyed on the text with its minted names renamed, so the next candidate's unchanged
               kernel -- rendered with other symbol numbers -- finds this one. *)
            let canonical = Kernel_code_cache.canonical_symbols ~keep:[ kernel_names.(i) ] source in
            (options, Kernel_code_cache.key ~parts:(options @ [ canonical ])))
    in
    let lookup idxs =
      List.iter idxs ~f:(fun i ->
          if Option.is_none programs.(i) then
            programs.(i) <- Kernel_code_cache.find kernel_cache (snd keyed.(i)))
    in
    let misses () = List.filter all ~f:(fun i -> Option.is_none programs.(i)) in
    (* The lookups run without the lock, so an all-hit compile never waits on another domain's
       nvrtc; the misses are looked up again under it, where another domain may just have published
       them. *)
    if enabled then lookup all;
    let compile_misses () =
      if enabled then lookup (misses ());
      let misses = misses () in
      let groups =
        if not enabled then
          (* As before the cache: one program of every kernel. *)
          [ (nvrtc_options full_source, all) ]
        else
          (* Misses grouped by option list, in batch order: one program per distinct list, so a
             kernel's PTX always comes from its own option list. *)
          let same = List.equal String.equal in
          let distinct =
            List.fold misses ~init:[] ~f:(fun acc i ->
                let options = fst keyed.(i) in
                if List.mem acc options ~equal:same then acc else acc @ [ options ])
          in
          List.map distinct ~f:(fun options ->
              (options, List.filter misses ~f:(fun i -> same (fst keyed.(i)) options)))
      in
      let n_compiled = List.length misses in
      let log_header =
        Printf.sprintf "/* kernel code cache: compiled %d of %d kernels, reused %d */\n" n_compiled
          n (n - n_compiled)
      in
      (* A program of every kernel in batch order is the routine's own source, already written as
         [<name>.cu]; any other program is written as [<name>.part<k>.cu] beside its PTX and log,
         under the name nvrtc was given, so its diagnostics point at the text it compiled.
         [.part<k>] is a spelling no routine name takes. *)
      let whole = match groups with [ (_, idxs) ] -> List.equal Int.equal idxs all | _ -> false in
      if Utils.settings.output_debug_files_in_build_directory then (
        (* What an earlier compile of this name left is not this compile's: remove it, so none is
           left that would read as one. *)
        let dir = Stdlib.Filename.dirname (Utils.build_file (name ^ ".ptx")) in
        let ours file =
          List.exists [ ".ptx"; ".cu_log" ] ~f:(fun ext -> String.equal file (name ^ ext))
          || String.is_prefix file ~prefix:(name ^ ".part")
        in
        (try
           Array.iter (Stdlib.Sys.readdir dir) ~f:(fun file ->
               if ours file then
                 try Stdlib.Sys.remove (Stdlib.Filename.concat dir file) with Sys_error _ -> ())
         with Sys_error _ -> ());
        if List.is_empty groups then
          Stdio.Out_channel.write_all (Utils.build_file (name ^ ".cu_log")) ~data:log_header);
      List.iteri groups ~f:(fun k (options, idxs) ->
          let artifact = if whole then name else name ^ ".part" ^ Int.to_string k in
          let source = with_conditional_includes (r.assemble idxs) in
          if not whole then write_source_artifact ~name:artifact source;
          (* Before the call: a program nvrtc refuses was still given these kernels. *)
          Compile_cost.record_kernels ~rendered:0 ~compiled:(List.length idxs);
          let ptx =
            try
              Compile_cost.time_device (fun () ->
                  compile_program ~name:artifact ~log_header ~options source)
            with exn ->
              let backtrace = Stdlib.Printexc.get_raw_backtrace () in
              r.attribute idxs exn backtrace
          in
          let program = new_program ptx in
          (* Published only now, after nvrtc returned: a failed program leaves no entry. *)
          if enabled then
            Kernel_code_cache.publish kernel_cache
              ~keys:(List.map idxs ~f:(fun i -> snd keyed.(i)))
              ~bytes:(String.length (Nvrtc.string_from_ptx ptx))
              program;
          List.iter idxs ~f:(fun i -> programs.(i) <- Some program))
    in
    if List.is_empty (misses ()) then (
      if Utils.settings.output_debug_files_in_build_directory then
        Stdlib.Mutex.protect kernel_compile_lock compile_misses)
    else Stdlib.Mutex.protect kernel_compile_lock compile_misses;
    (Array.map programs ~f:(fun p -> Option.value_exn ~here:[%here] p), r.entries)

  let%diagn2_sexp compile ~name bindings lowered =
    let programs, entries = compile_kernels ~batch:false ~names:[| name |] bindings [| lowered |] in
    let kparams, name, launch = entries.(0) in
    { code = programs.(0); kparams; bindings; name; launch }

  let%diagn2_sexp compile_batch ~names bindings lowereds =
    let programs, kparams_and_names = compile_kernels ~batch:true ~names bindings lowereds in
    { programs; kparams_and_names; bindings }

  (* [program]'s module in [prior_context]'s CUDA context: loaded on the program's first link into
     that context, then shared by every kernel and every routine the program holds there. *)
  let module_of prior_context (program : program) =
    let ctx = ctx_of prior_context in
    Stdlib.Mutex.protect program.modules_lock (fun () ->
        match List.Assoc.find program.modules ~equal:phys_equal ctx with
        | Some run_module -> run_module
        | None ->
            set_ctx ctx;
            let run_module = load_module program.ptx in
            prior_context.device.dev.set_builtins_in run_module;
            program.modules <- (ctx, run_module) :: program.modules;
            run_module)

  let link_proc ~prior_context ~name ~(kparams : (string * kparam_source) list)
      ~(launch : Low_level.launch_dims) ~ctx_buffers lowered_bindings run_module =
    let func = Cu.Module.get_function run_module ~name in
    let device = prior_context.device in
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
      let module S = Cu.Stream in
      let args : S.kernel_param list =
        (* TODO: should we prohibit or warn about local-only tensors that are in
           prior_context.ctx_buffers? *)
        List.map kparams ~f:(function
          | _name, Kparam_ptr tn ->
              let loc = Option.value_exn ~here:[%here] @@ Map.find ctx_buffers tn in
              let base = Map.find_exn ctx_bases tn in
              S.Tensor_at (Cu.Deviceptr.offset base ~bytes:loc.offset)
          | _name, Log_file_name -> S.Int log_id
          | _name, Merge_buffer ->
              let loc = Option.value_exn ~here:[%here] !(device.merge_buffer) in
              let base = Slab.resolve_pool device loc in
              S.Tensor_at (Cu.Deviceptr.offset base ~bytes:loc.offset)
          | _name, Static_idx s ->
              let i = Indexing.find_exn lowered_bindings s in
              (* Shared bind-time validation: negativity, range -- inclusive [0, range] for symbolic
                 extents (gh-490), strict [0, range) for indices -- and index width. *)
              Indexing.validate_bound_value ~width64:Utils.settings.large_models s !i;
              S.Int !i
          | _name, (Kparam_pool_slab _ | Kparam_pool_slots _) ->
              Backend_intf.unexpected_pooled_kparam ~backend:"Cuda_backend")
      in
      set_ctx @@ ctx_of prior_context;
      [%log "launching the kernel"];
      (* Stdio.printf "launching %s\n" name; *)
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
    let run_module = module_of prior_context code.code in
    set_ctx ctx;
    let idx_params = Indexing.bound_symbols code.bindings in
    let lowered_bindings : Indexing.lowered_bindings =
      List.map idx_params ~f:(fun s -> (s, Indexing.fresh_launch_param ()))
    in
    let task =
      link_proc ~prior_context ~name:code.name ~kparams:code.kparams ~launch:code.launch
        ~ctx_buffers lowered_bindings run_module
    in
    (lowered_bindings, task)

  let%track3_sexp link_batch prior_context (code_batch : code_batch) ctx_buffers =
    let idx_params = Indexing.bound_symbols code_batch.bindings in
    let lowered_bindings : Indexing.lowered_bindings =
      List.map idx_params ~f:(fun s -> (s, Indexing.fresh_launch_param ()))
    in
    let ctx = ctx_of prior_context in
    (* Each distinct program's module is loaded once per context ([module_of]); every task keeps its
       own kernel's module alive through its [context_lifetime], as before. *)
    let procs =
      Array.mapi code_batch.kparams_and_names ~f:(fun i (kparams, name, launch) ->
          let run_module = module_of prior_context code_batch.programs.(i) in
          set_ctx ctx;
          link_proc ~prior_context ~name ~kparams ~launch ~ctx_buffers lowered_bindings run_module)
    in
    (lowered_bindings, procs)

  (* gh-ocannl-1445: a graph-captured routine's instantiated graphs, by capture key, in FIFO order
     for eviction. Destroyed when the routine's context is released ([destroy_graphs], one of its
     [release_hooks]) and at process exit (below), not only on eviction: a cached exec otherwise
     lived as long as its routine's task, and to the end of the process for a routine nobody
     dropped. *)
  type graph_cache = {
    graphs_device : device;
    cache : (string, Cu.Graph.exec) Hashtbl.t;
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
            Cu.Stream.synchronize g.graphs_device.runner;
            let execs = Hashtbl.data g.cache in
            Hashtbl.clear g.cache;
            Queue.clear g.order;
            execs))
    in
    Backend_impl.attempt_each (List.map execs ~f:(fun exec () -> Cu.Graph.exec_destroy exec))

  (* gh-ocannl-1445: destroy every cached exec still alive at process exit, per device once its
     stream is idle, under the one [exit_stream_teardown_timeout] bound HIP's stream teardown uses
     ([Utils.bounded_exit_teardown]). Registered at module initialization for the same reason as
     HIP's: [at_exit] handlers run most-recent first, so every program handler that might still run
     a routine runs before this. The CUDA stream itself is not destroyed here (gh-ocannl-1048). *)
  let () =
    Stdlib.at_exit (fun () ->
        (* Only caches holding graphs: an emptied one (a released routine still reachable, or a
           capture that never succeeded) has nothing to wait for the stream for. *)
        match
          Backend_impl.Weak_registry.live graph_caches
          |> List.filter ~f:(fun g ->
              Stdlib.Mutex.protect g.lock (fun () -> not (Hashtbl.is_empty g.cache)))
        with
        | [] -> ()
        | live ->
            let by_device =
              List.fold live
                ~init:(Map.empty (module Int))
                ~f:(fun acc g -> Map.add_multi acc ~key:g.graphs_device.ordinal ~data:g)
            in
            ignore
              (Utils.bounded_exit_teardown
                 (Map.to_alist by_device
                 |> List.map ~f:(fun (ordinal, gs) : Utils.exit_teardown_resource ->
                     let device = (List.hd_exn gs).graphs_device in
                     {
                       what = [%string "the cached CUDA graphs of device %{ordinal#Int}"];
                       is_idle =
                         (fun () ->
                           set_ctx device.dev.primary_context;
                           Cu.Stream.is_ready device.runner);
                       teardown =
                         (fun () ->
                           Backend_impl.attempt_each (List.map gs ~f:(fun g () -> destroy_graphs g)));
                     }))
                : Utils.exit_teardown_outcome list))

  (* One CUDA graph for a fissioned routine's whole segment batch (gh-ocannl-488): stream-capture
     the segment launch loop once per distinct set of launch-time-varying arguments — static-index
     binding values, plus the merge-buffer position when the routine reads it — instantiate, and
     replay with a single cuGraphLaunch per step instead of one cuLaunchKernel per segment. Baking
     kernel arguments into the graph is sound because context buffer bases are pre-resolved at link
     time (tnode pools are never reallocated in place while their routines are live; the merge pool,
     which can be, is part of the key by pointer identity) and every other varying argument is part
     of the key. The same-stream linear dependency chain the capture records is exactly the FIFO
     ordering the per-segment launch loop relies on. Instantiated graphs are retained in a bounded
     FIFO cache, so a training loop cycling through batch-index bindings replays cached graphs from
     the second epoch on. Transparent fallback to per-segment launches: when kernel logging is on
     (the log id is a fresh kernel argument every run), when disabled via [gpu_graph_capture=false],
     or permanently for this routine if the driver rejects capture. *)
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
                [ Cu.Deviceptr.string_of (Slab.resolve_pool device loc); Int.to_string loc.offset ]
            | None -> [ "no-merge" ]
        in
        String.concat ~sep:";" (idx @ merge)
      in
      let capture () =
        (* RELAXED, not THREAD_LOCAL: GC finalizers (module unloads, buffer frees of dead handles)
           can fire at any allocation point on the capturing thread, and stricter modes make the
           driver reject such "potentially unsafe" calls mid-capture — with the exception then
           escaping [Gc.finalise] at an arbitrary program point. The finalizers only release dead
           handles, so they are genuinely safe to run concurrently with capture. *)
        Cu.Graph.begin_capture ~mode:Cu.Graph.RELAXED device.runner;
        let graph =
          try
            run_plain ();
            Cu.Graph.end_capture device.runner
          with exn ->
            (* Terminate the capture before propagating, else the stream stays in capture mode. *)
            (try Cu.Graph.destroy (Cu.Graph.end_capture device.runner) with _ -> ());
            raise exn
        in
        let exec = Cu.Graph.instantiate graph in
        Cu.Graph.destroy graph;
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
                        | Some exec -> Cu.Graph.launch exec device.runner
                        | None -> (
                            match capture () with
                            | exec ->
                                if Queue.length order >= max_cached_graphs then (
                                  let victim = Queue.dequeue_exn order in
                                  (* The evicted exec may still have a pending launch on the
                                     stream. *)
                                  Cu.Stream.synchronize device.runner;
                                  Cu.Graph.exec_destroy (Hashtbl.find_exn cache victim);
                                  Hashtbl.remove cache victim);
                                Hashtbl.set cache ~key ~data:exec;
                                Queue.enqueue order key;
                                Cu.Graph.launch exec device.runner
                            | exception Cu.Cuda_error { status; message } ->
                                (* E.g. capture unsupported on this driver: fall back to per-segment
                                   launches for this routine (same-stream FIFO supplies the segment
                                   ordering), and re-run outside capture so a genuine launch failure
                                   surfaces on the plain path. *)
                                broken := true;
                                Stdio.eprintf
                                  "ocannl: disabling CUDA graph capture for routine %s (%s: %s)\n%!"
                                  name message
                                  (Sexp.to_string_hum @@ Cu.sexp_of_result status);
                                run_plain ())));
            },
          fun () -> destroy_graphs graphs )

  let get_global_debug_info () =
    Sexp.message "cuda_global_debug"
      [
        ("live_streams", [%sexp_of: int] @@ Cu.Stream.get_total_live_streams ());
        ("live_graph_execs", [%sexp_of: int] @@ live_graph_execs ());
      ]

  let static_properties () =
    let device_properties =
      Array.init (num_devices ()) ~f:(fun ordinal ->
          let dev = Cu.Device.get ~ordinal in
          let attributes = Cu.Device.get_attributes dev in
          let props =
            [
              ("device_name", Sexp.Atom attributes.name);
              ("device_ordinal", [%sexp_of: int] ordinal);
              ("multiprocessor_count", [%sexp_of: int] attributes.multiprocessor_count);
              ("clock_rate", [%sexp_of: int] attributes.clock_rate);
              ("async_engine_count", [%sexp_of: int] attributes.async_engine_count);
              ("compute_capability_major", [%sexp_of: int] attributes.compute_capability_major);
              ("compute_capability_minor", [%sexp_of: int] attributes.compute_capability_minor);
              ("max_threads_per_block", [%sexp_of: int] attributes.max_threads_per_block);
              (* The launch-dimension limits the schedule layer gates against, mirroring the HIP
                 dump: [max_block_dim] bounds a workgroup per-dimension beyond the
                 [max_threads_per_block] product (gh-ocannl-679), and [max_grid_dim] is the device's
                 reading of what [hardware_limits.max_grid_yz] asserts architecturally. Surfaced so
                 a run on hardware can read back what the gates compare against --
                 [bin/device_props.ml] prints exactly this (gh-ocannl-684). *)
              ( "max_block_dim",
                [%sexp_of: int * int * int]
                  ( attributes.max_block_dim_x,
                    attributes.max_block_dim_y,
                    attributes.max_block_dim_z ) );
              ( "max_grid_dim",
                [%sexp_of: int * int * int]
                  (attributes.max_grid_dim_x, attributes.max_grid_dim_y, attributes.max_grid_dim_z)
              );
              ("unified_addressing", [%sexp_of: bool] attributes.unified_addressing);
              (* The memory model, as HIP's dump states it: an integrated part (a Jetson, a GB10)
                 shares host DRAM, a discrete one has its own. gh1133_cells.sh's kprof step states
                 it for every traced cell from here (gh-ocannl-1444). *)
              ("integrated", [%sexp_of: bool] attributes.integrated);
            ]
          in
          Sexp.message "device" props)
    in
    Sexp.List (Sexp.Atom "cuda_devices" :: Array.to_list device_properties)

  (* Conservative per-workgroup device limits for the schedule layer (schedule-ir-optops §6):
     minimum across devices, so code compiled once is valid wherever it links. *)
  (* Memoized behind [lazy]: driver init and device enumeration must not run at backend-module
     initialization ([num_devices] forces [ensure_initialized]). *)
  (* Concrete static capabilities separate mixed devices without changing the conservative
     construction limits. Driver/runtime/header provenance remains a documented separate concern. *)
  let timing_identity (device : device) =
    try
      let attributes = Cu.Device.get_attributes device.dev.dev in
      Some
        {
          Backend_intf.device_signature =
            Sexp.to_string
              (Sexp.message "device_capabilities"
                 [
                   ("name", Sexp.Atom attributes.name);
                   ( "compute_capability",
                     [%sexp_of: int * int]
                       (attributes.compute_capability_major, attributes.compute_capability_minor) );
                   ("multiprocessor_count", [%sexp_of: int] attributes.multiprocessor_count);
                   ("clock_rate", [%sexp_of: int] attributes.clock_rate);
                   ("memory_clock_rate", [%sexp_of: int] attributes.memory_clock_rate);
                   ("global_memory_bus_width", [%sexp_of: int] attributes.global_memory_bus_width);
                   ("l2_cache_size", [%sexp_of: int] attributes.l2_cache_size);
                   ("max_threads_per_block", [%sexp_of: int] attributes.max_threads_per_block);
                   ( "max_threads_per_multiprocessor",
                     [%sexp_of: int] attributes.max_threads_per_multiprocessor );
                   ( "max_shared_memory_per_block",
                     [%sexp_of: int] attributes.max_shared_memory_per_block );
                   ( "max_shared_memory_per_multiprocessor",
                     [%sexp_of: int] attributes.max_shared_memory_per_multiprocessor );
                   ("max_registers_per_block", [%sexp_of: int] attributes.max_registers_per_block);
                   ( "max_registers_per_multiprocessor",
                     [%sexp_of: int] attributes.max_registers_per_multiprocessor );
                   ("warp_size", [%sexp_of: int] attributes.warp_size);
                 ]);
          toolchain_signature = None (* Driver/compiler provenance is tracked in gh-ocannl-1026. *);
        }
    with Cu.Cuda_error _ -> None

  let hardware_limits =
    let limits =
      lazy
        (let attrs =
           Array.init (num_devices ()) ~f:(fun ordinal ->
               Cu.Device.get_attributes (Cu.Device.get ~ordinal))
         in
         let min_over f = Array.map attrs ~f |> Array.min_elt ~compare:Int.compare in
         {
           Backend_intf.max_threads_per_workgroup =
             min_over (fun (a : Cu.Device.attributes) -> a.max_threads_per_block);
           max_workgroup_memory_bytes =
             min_over (fun (a : Cu.Device.attributes) -> a.max_shared_memory_per_block);
           (* Unmeasured (gh-ocannl-1313): CUDA documents a 512 KiB local-memory ceiling per thread,
              but none of our kernels has been checked against it, so the gate stays off rather than
              encode an untested figure. *)
           max_thread_private_bytes = None;
           (* Per-dimension workgroup caps (gh-ocannl-679). Queried rather than asserted from the
              Compute Capability tables (where they are (1024, 1024, 64) from cc 2.0 up) because the
              driver answers per device and the fold below shows the query costs nothing extra. The
              [.z] entry is the point of the field: at 64 it sits 16x below [max_threads_per_block],
              so a 2 x 2 x 128 workgroup has a legal 512-thread product and is still an invalid
              launch configuration. *)
           max_workgroup_dims =
             (match
                ( min_over (fun (a : Cu.Device.attributes) -> a.max_block_dim_x),
                  min_over (fun (a : Cu.Device.attributes) -> a.max_block_dim_y),
                  min_over (fun (a : Cu.Device.attributes) -> a.max_block_dim_z) )
              with
             | Some x, Some y, Some z -> Some (x, y, z)
             (* No devices: [min_over] is [None] on all three, and an empty machine advertises no
                cap rather than a spurious one. *)
             | _ -> None);
           (* CUDA's gridDim.y and gridDim.z cap is 65535 on every architecture (the Compute
              Capability tables), unlike the per-device limits queried above — a constant, not an
              attribute. gridDim.x is 2^31-1 and needs no gate. *)
           max_bound_pools = None;
           max_grid_yz = Some 65535;
           (* Tensor cores: the descriptor of the attached devices' minimum compute capability, from
              [Cuda_mma.capability], whose cutoffs are tested against the arm floors of the hooks
              below on both sides of each one (gh-ocannl-1214). *)
           mma = Cuda_mma.capability ~cc:(min_compute_capability ());
           simd_vector = Ir.Vector_file.none;
           native_fp16_arithmetic = false;
           simdgroup_width = Some 32;
           (* gh-ocannl-1124: measured true on the RTX of rog-nv; see the field. *)
           lane_scalar_recompute_cheap = true;
           (* gh-ocannl-1168, gh-ocannl-1256: measured on the RTX of rog-nv -- heads of 64 and 128
              on the cross-simdgroup all-reduce train 16-63% faster; see the field. *)
           lane_all_reduce_simdgroups = 4;
           (* gh-ocannl-1194: the tuned f32 transformer loses with the block fold on the measured
              RTX, while the two-pass rewrite keeps the tf32 gain. Keep unmeasured CUDA devices
              conservative too; an explicit block size still requests the fold. *)
           online_softmax_auto_block = 0;
           worker_pool_tag = None;
           (* Filled fresh by the accessor below, not here: both inputs are process-mutable
              ([Train.CDSL.enable_all_debugs] flips the debug settings at any point), and this
              record is memoized (Codex P1 on PR #337). *)
           codegen_tag = None;
           (* Advisory roofline envelope (gh-ocannl-491): documented rough constants for the sm_70+
              class. Flops: RTX-30/40 mid-range ~15 fp32 TFLOP/s — the model only ranks, so a
              class-typical number suffices. Bandwidth is a class CEILING per the [hardware_limits]
              contract (gh-ocannl-578): an RTX 4090 already sustains ~1 TB/s (above the previous
              4.5e11) and HBM3e datacenter parts reach ~8 TB/s. Per-device queries (SM count x
              clock, memory clock x bus width) remain calibration follow-up work. *)
           peak_flops = Some 1.5e13;
           peak_memory_bandwidth = Some 8.0e12;
           (* gh-ocannl-1253: the ranking score's two tiers and residency, read only by the untimed
              choosers (Untimed, outside the schedule-cache key). Residency is queried --
              multiprocessors times resident threads per multiprocessor, minimum over devices. The
              capacity and the cache bandwidth come from benchmarks/gh1253_cells.sh on the RTX 5070
              Ti of rog-nv (2026-10-19, 84b727ea): the untiled weight gradient's requested rate
              stays on a 1.8 TB/s cache plateau up to a 4.6 GB working set and is still at 1.26 TB/s
              at 9.3 GB, falling towards the 582 GB/s DRAM rate by 18.6 GB (1.04 TB/s there, at the
              split threshold); the fit put the capacity near 13 GB. *)
           cache_capacity_bytes = Some (12 * 1024 * 1024 * 1024);
           cache_bandwidth = Some 1.8e12;
           (* gh-ocannl-1442: the ranking score's own rates, below the envelope's ceilings, from
              benchmarks/gh1253_cells.sh on the RTX 5070 Ti of rog-nv (2026-10-21, 46d84231): the
              STREAM triad's 603 GB/s, and the seeded register-blocktiled sketch's 3.44e12 scalar
              operations/s (its 512 x 512 sweep), which predicted the seeded weight gradient at b1,
              b8 and b256 within 0.87-1.17x. One device's rates for the class, like the cache tier's
              above: override per machine with model_dram_bandwidth / model_issue_rate. *)
           issue_rate = Some 3.4e12;
           dram_bandwidth = Some 6.0e11;
           max_resident_threads =
             min_over (fun (a : Cu.Device.attributes) ->
                 a.multiprocessor_count * a.max_threads_per_multiprocessor);
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
      ^ if Utils.with_runtime_debug () then "/device-debug" else "/no-device-debug"
    in
    fun () -> { (Lazy.force limits) with Backend_intf.codegen_tag = Some (codegen_tag ()) }

  (* {2 Failure classification (gh-ocannl-536)}

     A driver error code is the only evidence a launch or sync failure leaves, and only this backend
     can read it: [Cu.Cuda_error] and [Nvrtc.Nvrtc_error] are opaque to the common policy, which
     therefore treats every one of them as unclassified — fatal at [Launch]/[Sync] regardless of
     [strict_failure_classification], since {!Ir.Schedule_outcome.classify_raw} only softens
     compile-side phases. One candidate the driver refuses ends the whole search, and takes the
     measurements already collected with it.

     Two axes, per {!Ir.Schedule_outcome}: attribution (is this candidate at fault) and damage (what
     the device may have written). CUDA splits cleanly on the second, and that split is why this is
     worth writing:

     - Statuses the driver returns {e before any thread runs} — a launch configuration refused, an
     allocation denied, a module that will not load — are [No_device_writes]. The tuner withdraws
     the routine's execution claim and keeps searching. This is the containment win, and on this
     backend it is the whole of it: [CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES], the status a
     register-heavy or over-wide candidate actually gets, is non-sticky and write-free. - The fault
     family (illegal address, misaligned address, device-side assert, watchdog timeout) is
     asynchronous: the kernel was running, whatever it wrote before faulting stays written, and the
     CUDA context is left sticky — every later call in it fails with the same code. These are
     [Writes_may_have_occurred]: the tuner counts the decline, then escalates and poisons the
     lineage. Classifying them does not make them survivable — nothing does until
     [recover_after_launch_failure] lands — but it puts the driver's own verdict in the report
     instead of an opaque [Cuda_error], and states the damage rather than leaving it to the phase
     default.

     Everything else stays [None], deliberately. An environment or toolchain fault (no JIT compiler,
     PTX newer than the driver, no binary for this GPU, an unavailable device) fails every candidate
     identically; absorbing it as a decline would turn a broken installation into a silent "nothing
     worked" report instead of an error. Same for a bare [CUDA_ERROR_INVALID_VALUE] outside
     [Launch], where it means API misuse rather than a rejected launch geometry.

     {3 What is missing: the pre-launch launchability validator}

     The other half of gh-ocannl-536 for this backend is a post-link check in the shape of Metal's
     (metal_backend.ml, [link_proc]): ask the {e compiled} kernel what it can be launched with,
     before launching it. CUDA's answer is [cuFuncGetAttribute] on the linked [CUfunction] —
     [CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK] (which drops below the device width under register
     pressure, exactly as Metal's [maxTotalThreadsPerThreadgroup] does),
     [CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES], [CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES] — and cudajit binds
     none of them: {!Cu.Module.get_function} returns an opaque handle and [Cu.Module] exposes no
     attribute query, so there is no API-supported launchability condition to reject against here.
     Extending the bindings is the work, and note what it does and does not buy on CUDA: a better
     message and a [Resource_exceeded] cause with populated [requested]/[limit], not more
     containment, because the condition it would predict is already contained above. (Per the
     proposal's rule, a cause that cannot populate those fields is reported as [Backend_rejected]
     rather than as a [Resource_exceeded] with invented numbers — which is why every case below is
     [Backend_rejected].) *)

  let status_name sexp_of status =
    match sexp_of status with Sexp.Atom name -> name | sexp -> Sexp.to_string sexp

  let classify_failure phase exn =
    let reject ~stage ~severity ~execution_effect detail =
      Some
        {
          Schedule_outcome.phase;
          cause = Schedule_outcome.Backend_rejected { backend = name; stage; severity; detail };
          execution_effect;
        }
    in
    match exn with
    | Cu.Cuda_error { status; message } -> (
        let stage = status_name Cu.sexp_of_result status in
        let detail = [%string "CUDA driver: %{stage} raised by %{message}"] in
        match stage with
        (* Refused before execution, context left usable: the block's register x width or static
           shared-memory demand exceeds what an SM can give this kernel; the cluster or cooperative
           launch shape is out of range; a device allocation was denied. *)
        | "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES" | "CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE"
        | "CUDA_ERROR_INVALID_CLUSTER_SIZE" | "CUDA_ERROR_OUT_OF_MEMORY" ->
            reject ~stage ~severity:Schedule_outcome.Expected
              ~execution_effect:Schedule_outcome.No_device_writes detail
        (* Only informative where the driver is validating a launch geometry we chose. *)
        | "CUDA_ERROR_INVALID_VALUE" when Schedule_outcome.equal_phase phase Schedule_outcome.Launch
          ->
            reject ~stage ~severity:Schedule_outcome.Expected
              ~execution_effect:Schedule_outcome.No_device_writes detail
        (* Our PTX, rejected by the driver's JIT before anything ran: a codegen bug, not the user's
           and not the environment's. Counted and logged even though the search survives it. *)
        | "CUDA_ERROR_INVALID_PTX" | "CUDA_ERROR_INVALID_SOURCE" | "CUDA_ERROR_INVALID_IMAGE" ->
            reject ~stage ~severity:Schedule_outcome.Compiler_bug
              ~execution_effect:Schedule_outcome.No_device_writes detail
        (* Asynchronous device faults: partial writes, sticky context. *)
        | "CUDA_ERROR_ILLEGAL_ADDRESS" | "CUDA_ERROR_MISALIGNED_ADDRESS"
        | "CUDA_ERROR_INVALID_ADDRESS_SPACE" | "CUDA_ERROR_INVALID_PC"
        | "CUDA_ERROR_ILLEGAL_INSTRUCTION" | "CUDA_ERROR_HARDWARE_STACK_ERROR" | "CUDA_ERROR_ASSERT"
        | "CUDA_ERROR_LAUNCH_FAILED" ->
            reject ~stage ~severity:Schedule_outcome.Compiler_bug
              ~execution_effect:Schedule_outcome.Writes_may_have_occurred detail
        (* Same damage, but the kernel was merely slower than the display driver's watchdog allows —
           that is the candidate's shape, not a codegen bug. *)
        | "CUDA_ERROR_LAUNCH_TIMEOUT" ->
            reject ~stage ~severity:Schedule_outcome.Expected
              ~execution_effect:Schedule_outcome.Writes_may_have_occurred detail
        | _ -> None)
    | Nvrtc.Nvrtc_error { status; message } -> (
        let status = status_name Nvrtc.sexp_of_result status in
        match status with
        (* The generated CUDA C did not compile. Nothing was allocated or launched, and cudajit puts
           nvrtc's compilation log in [message]. [stage] matches the cc backend's, so the blocker
           census groups "our codegen does not compile" across backends.

           Except for a header nvrtc could not open ("cannot open source file", a missing or
           mismatched CUDA include tree), which it reports under this same status but which fails
           every candidate alike ([C_syntax.missing_include], ahrefs/ocannl#1387).

           This status only. [NVRTC_ERROR_BUILTIN_OPERATION_FAILURE] in particular stays
           unclassified: it reports nvrtc failing on its own builtins, which an incomplete or
           mismatched CUDA installation reproduces for every candidate — precisely the
           fails-identically class this classifier must let propagate. *)
        | "NVRTC_ERROR_COMPILATION" when not (C_syntax.missing_include message) ->
            reject ~stage:"compiler" ~severity:Schedule_outcome.Compiler_bug
              ~execution_effect:Schedule_outcome.No_device_writes
              [%string
                "OCANNL cuda backend: generated code failed to compile (%{status}).\n\
                 This is a bug in OCANNL. Please file an issue with the generated .cu file.\n\
                 nvrtc output:\n\
                 %{message}"]
        | _ -> None)
    | _ -> None

  let get_debug_info (device : device) =
    let tot, unr, unf = Cu.Stream.total_unreleased_unfinished_delimited_events device.runner in
    let i2s = [%sexp_of: int] in
    Sexp.message "cuda_stream_debug"
      [ ("total_events", i2s tot); ("unreleased_events", i2s unr); ("unfinished_events", i2s unf) ]
end

(* gh-ocannl-1038: what this backend probes on the host without opening a device, in the form
   {!Context.host_capabilities} records. Only the toolkit include tree: everything else this backend
   decides on -- compute capability, the device limits -- is a device query. *)
let host_capabilities () =
  [
    ( "cuda_include",
      (* The headers kernels include, read off the text that includes them. *)
      Utils.capability_dir
        ~requires:
          (Utils.included_headers cuda_header_prelude
          @ List.concat_map Cuda_like_config.Cuda.conditional_includes ~f:(fun (_, directive) ->
              Utils.included_headers directive))
        (List.find_map (cuda_include_options ()) ~f:(String.chop_prefix ~prefix:"-I")) );
  ]
