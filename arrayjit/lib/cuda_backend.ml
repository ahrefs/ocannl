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
    let ptr = Cu.Deviceptr.mem_alloc ~size_in_bytes in
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

(* The compute capability, as [major * 10 + minor], of the GeForce-Blackwell family whose
   block-scaled [mma.sync ... kind::mxf8f6f4] forms exist only under an architecture-specific target
   (gh-ocannl-481 item 3, D4). *)
let mxfp8_family_cc = 120

let%diagn_sexp gpu_arch_options ~device_cc cu_src : string list =
  (* The [--gpu-architecture] options for one nvrtc compile, as a pure function of the source's arch
     markers and the attached devices' minimum compute capability — separated from [cuda_to_ptx] so
     the policy is testable without a device (arrayjit/test/test_cuda_arch_flags.ml).

     Two kinds of target, and the difference is the whole point of this function:

     - A FLOOR (every marker but one): the lowest arch whose PTX contains the instruction. PTX
     targeted at a floor is forward-JIT-compiled by the driver on every later GPU, so one compile
     covers the entire range above it. The [(wmma-bf16)] and [(wmma-tf32)] markers are emitted by
     [mma_syntax] for bf16 resp. tf32 fragments (both sm_80+); [(mma-fp8)], [(mma-bf16)] and
     [(mma-f16)] for the inline-PTX [mma.sync] paths (sm_89+ resp. sm_80+ for both 16-bit forms, no
     header needed). - A FAMILY target ([(mma-mxfp8)]): a [compute_120a]-style architecture-specific
     arch, which only the device family it names can load. Blackwell's block-scaled [kind::mxf8f6f4]
     forms exist ONLY under such a target, so this is the one case where forward-JIT portability has
     to be given up — and therefore the one marker gated on the attached devices' own family. Family
     PTX is never produced for a device that could not load it; a marked kernel reaching a
     mismatched device falls back to floor targeting, which is defense in depth rather than a
     recovery path (an arm emitting the marker is gated on the same family, so it should have
     declined already). No arm emits it yet: block scaling itself is blocked on OCANNL having
     microscaling storage at all (the e8m0 per-32-element scale factors are extra mma OPERANDS with
     their own layout, and [Tile_mma] has no slot for them), and a unit-scale arm would be
     numerically identical to the plain fp8 path while forfeiting forward-JIT. *)
  let has s = String.is_substring cu_src ~substring:s in
  if has "(mma-mxfp8)" && device_cc / 10 = mxfp8_family_cc / 10 then (
    [%log "family-arch target", (device_cc : int)];
    [ Printf.sprintf "--gpu-architecture=compute_%da" device_cc ])
  else
    let uses_wmma = has "nvcuda::wmma" in
    (* Half/bf16 ARITHMETIC intrinsics (unlike the conversions, which cuda_fp16.h/cuda_bf16.h
       emulate on any arch) are only declared for __CUDA_ARCH__ >= 530 (halfs) resp. >= 800
       (bfloat16s), while nvrtc's default target is compute_52 — e.g. [__hfma] in a serial half
       matmul fails with "identifier undefined" unless we raise the floor. The bf16 overloads share
       the half intrinsics' names, so a bf16 kernel is recognized by the type name appearing
       alongside the arithmetic tokens; a kernel mixing half arithmetic with bf16 storage-only is
       conservatively floored at compute_80 too. *)
    let uses_h_arith =
      (* Every half-arith token [unop_syntax]/[binop_syntax]/[ternop_syntax] can emit; [hexp]/[hlog]
         also cover their [2]-suffixed variants and [__hmax]/[__hmin] the [_nan] variants as
         substrings. *)
      List.exists ~f:has
        [
          "__hadd";
          "__hsub";
          "__hmul";
          "__hdiv";
          "__hmax";
          "__hmin";
          "__hgt";
          "__hfma";
          "hexp";
          "hlog";
          "hsin";
          "hcos";
          "hsqrt";
          "hrcp";
          "hrsqrt";
          "htanh_approx";
        ]
    in
    (* [compile_batch] concatenates several kernels into one source, so the floors can trigger
       independently (e.g. a half-wmma kernel batched with scalar bf16 arithmetic needs compute_80
       even without the [(wmma-bf16)] marker): take the max, not the first match. *)
    let arch_floor =
      List.filter_opt
        [
          (if uses_wmma then Some (if has "(wmma-bf16)" || has "(wmma-tf32)" then 80 else 70)
           else None);
          (if has "(mma-fp8)" then Some 89 else None);
          (if has "(mma-bf16)" then Some 80 else None);
          (* The f32-accumulate uniform-f16 [mma.sync] arm (gh-ocannl-680): m16n8k16 with .f16
             operands needs sm_80, same as the bf16 form it shares its body with. *)
          (if has "(mma-f16)" then Some 80 else None);
          (* The [cp.async] staging builtins (gh-ocannl-487 phase 2); emission is gated on the
             devices' own capability, so the floor only ever fires where it can also load. The
             marker is the PTX instruction text, which appears exactly in the prepended helper
             definitions' asm literals — never in an identifier — so a routine or label merely NAMED
             like a helper (which the token-scoped builtin filter declines to activate) cannot raise
             the floor past a pre-Ampere device (Codex P2 on PR #317, round 5). *)
          (if has "cp.async." then Some 80 else None);
          (if uses_h_arith then Some (if has "__nv_bfloat16" then 80 else 53) else None);
        ]
      |> List.max_elt ~compare:Int.compare
    in
    match arch_floor with
    | Some floor ->
        (* CUDA 13 dropped offline compilation below compute_75 (Maxwell through Volta), so nvrtc 13
           rejects the compute_53/compute_70 floors outright: raise a triggered floor to compute_75
           whenever every attached device can load such PTX (a device below sm_75 keeps the literal
           floor — it must be paired with an nvrtc 12.x that still accepts it). We deliberately do
           NOT raise all the way to the device arch (e.g. compute_120 on Blackwell GeForce): the
           sm_89 fp8 [mma.sync] encoding remains valid under a compute_89 target where a compute_120
           target would demand the family-specific [kind::] forms. *)
        let arch = max floor (min 75 device_cc) in
        [ Printf.sprintf "--gpu-architecture=compute_%d" arch ]
    | None -> []

(* The [-I] nvrtc needs to find <cuda_fp8.h> and friends, discovered from the environment: CUDA_PATH
   where it is set, the no-spaces junction ocaml-cudajit creates on Windows (whose SDK path
   otherwise contains spaces), and /usr/local/cuda as the Linux fallback. Separated out of the
   compile path for the same reason [gpu_arch_options] is: it is a policy other nvrtc callers need
   to agree with, and tools/fp8_soak.ml is one -- a soak that looked only in /usr/local/cuda would
   report the CUDA arm ready and then fail to compile its kernel on every Windows installation and
   every Linux one outside that prefix (Codex P2 on PR #463). *)
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
        (let n = num_devices () in
         if n = 0 then 0
         else
           Array.init n ~f:(fun ordinal ->
               let a : Cu.Device.attributes = Cu.Device.get_attributes (Cu.Device.get ~ordinal) in
               (a.compute_capability_major * 10) + a.compute_capability_minor)
           |> Array.min_elt ~compare:Int.compare
           |> Option.value ~default:0)
    in
    fun () -> Lazy.force cc

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

  let%diagn2_sexp cuda_to_ptx ~name cu_src =
    let cu_src =
      C_syntax.prepend_conditional_includes
        ~conditional_includes:Cuda_like_config.Cuda.conditional_includes cu_src
    in
    let arch_opts = gpu_arch_options ~device_cc:(min_compute_capability ()) cu_src in
    let name_cu = name ^ ".cu" in
    if Utils.settings.output_debug_files_in_build_directory then (
      let build_file = Utils.open_build_file ~base_name:name ~extension:".cu" in
      Stdio.Out_channel.output_string build_file.oc cu_src;
      build_file.finalize ());
    [%log "compiling to PTX"];
    (* [with_debug] only asks NVRTC to keep the compilation log on a SUCCESSFUL compile (a failing
       one carries its log in the exception either way), and the only reader of that log is the
       [.cu_log] build file written just below -- so it wants exactly the flag that writes it, and
       that file's [Option.value_exn] is what makes the implication load-bearing. The [|| log_level
       > 0] disjunct it used to carry made the log's retention hostage to a verbosity knob that
       nothing here consults (gh-ocannl-595). Kernel debug codegen is separate: it comes from
       [Utils.with_runtime_debug ()] via [--device-debug] below. *)
    let with_debug = Utils.settings.output_debug_files_in_build_directory in
    let cuda_include_opt = cuda_include_options () in
    (* gh-ocannl-784: assembled by [Compiler_options.nvrtc], the pure module the HIP backend already
       shares, so the exact vector -- semantic state for fast-math numerics -- is pinned by a
       GPU-free test (arrayjit/test/test_cuda_compile_options.ml) instead of only by whichever box
       happens to have a device. Its header carries the measured reassociation boundary. *)
    let options =
      Compiler_options.nvrtc ~cuda_include_options:cuda_include_opt ~arch_options:arch_opts
        ~with_device_debug:(Utils.with_runtime_debug ())
    in
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

  type code = {
    ptx : Nvrtc.compile_to_ptx_result;
    kparams : (string * kparam_source) list;
    bindings : Indexing.unit_bindings;
    name : string;
    launch : Low_level.launch_dims;
  }
  [@@deriving sexp_of]

  type code_batch = {
    ptx : Nvrtc.compile_to_ptx_result;
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
       [ocannl_cp_async*] builtins; their name doubles as the [gpu_arch_options] sm_80 floor
       marker). Gated on the attached devices' minimum compute capability, not the emission call:
       pre-Ampere devices keep the portable synchronous rendering, which is correct at any depth —
       the same posture as a hand-built pipelined schedule on a backend without the hook. *)
    let async_copy =
      if min_compute_capability () >= 80 then
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

    (* The wmma-supported precision combinations (tensorize-mma T3). Shared by [mma_syntax] and
       [mma_fragment_syntax] so a fragment scope accepts exactly when its nested update-only MMA
       calls would — including the numerics-policy gate on the tf32 arm, which lives here for the
       same reason. *)
    type wmma_combo_info = {
      wc_ab_typ : string;
          (** The fragment element type for [matrix_a]/[matrix_b] — a C++ type for the 16-bit
              combinations, the tag type [nvcuda::wmma::precision::tf32] for tf32 (storage stays
              [float]; only the fragments are tagged). *)
      wc_acc_typ : string;  (** The accumulator fragment element type. *)
      wc_tm : int;
      wc_tn : int;
      wc_tk : int;
          (** The intrinsic tile shape: 16×16×16 for the 16-bit combinations, 16×16×8 for tf32
              (mirrors [mma_format_tiles] in the capability descriptor). *)
      wc_ab_ld_mult : int;  (** wmma stride constraint: a/b leading-dim multiple, in elements. *)
      wc_d_ld_mult : int;  (** wmma stride constraint: d leading-dim multiple, in elements. *)
      wc_min_cc : int;
      wc_marker : string;
          (** Marker suffix for the rendering comment (["" | "-bf16" | "-tf32"]); [cuda_to_ptx]
              greps it to select the arch floor. *)
      wc_cvt_tf32 : bool;
          (** Convert loaded a/b fragment elements with [__float_to_tf32]: tf32 fragments load raw
              f32 bits, the explicit conversion performs the mantissa truncation (per the CUDA
              programming guide; the intrinsic requires already-converted inputs). *)
    }

    let wmma_combo ~a_prec ~b_prec ~d_prec =
      let mk ?(tile = (16, 16, 16)) ?(marker = "") ?(cvt_tf32 = false) ab_typ acc_typ ab_ld d_ld cc
          =
        let wc_tm, wc_tn, wc_tk = tile in
        Some
          {
            wc_ab_typ = ab_typ;
            wc_acc_typ = acc_typ;
            wc_tm;
            wc_tn;
            wc_tk;
            wc_ab_ld_mult = ab_ld;
            wc_d_ld_mult = d_ld;
            wc_min_cc = cc;
            wc_marker = marker;
            wc_cvt_tf32 = cvt_tf32;
          }
      in
      match (a_prec, b_prec, d_prec) with
      | Ops.Half_prec _, Ops.Half_prec _, Ops.Single_prec _ -> mk "__half" "float" 8 4 70
      (* The f16-accumulate wmma triple must not render under [Numerics.Fp16_wide] (gh-ocannl-680):
         the uniform-f16 combination then goes through the f32-accumulate inline-PTX m16n8k16 arm
         instead, or declines to the scalar fallback, whose accumulator follows [accum_prec] —
         width-uniform either way. *)
      | Ops.Half_prec _, Ops.Half_prec _, Ops.Half_prec _ when not (Numerics.fp16_accum_wide ()) ->
          mk "__half" "__half" 8 8 70
      | Ops.Bfloat16_prec _, Ops.Bfloat16_prec _, Ops.Single_prec _ ->
          mk ~marker:"-bf16" "__nv_bfloat16" "float" 8 4 80
      | Ops.Single_prec _, Ops.Single_prec _, Ops.Single_prec _
        when (Numerics.get ()).Numerics.tf32_matmuls ->
          (* gh-ocannl-478: uniform-f32 GEMMs compute in tf32 (m16n16k8, sm_80+) when the numerics
             policy opts in; with the policy off this arm is [None] and the scalar fallback keeps
             full f32 numerics. f32's wmma stride constraint is 4 elements. *)
          mk ~tile:(16, 16, 8) ~marker:"-tf32" ~cvt_tf32:true "nvcuda::wmma::precision::tf32"
            "float" 4 4 80
      | _ -> None

    (* Tensorize-mma T3: cooperative tile-MMA emission for [Low_level.Tile_mma]. Two renderings:

       - The CUDA wmma C++ API for the 16-bit combinations: f16 x f16 -> f32 (the flagship), f16 x
       f16 -> f16 (under [Numerics.Fp16_auto]/[Fp16_narrow] only — the wide policy routes the
       uniform-f16 combination to the f32-accumulate inline-PTX m16n8k16 arm instead,
       gh-ocannl-680), and bf16 x bf16 -> f32 (sm_80+; the [(wmma-bf16)] marker makes [cuda_to_ptx]
       select the arch). 16x16x16 fragment blocks are resident across the whole [k] extent,
       mirroring the Metal emission. - Inline-PTX
       [mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32] for fp8 x fp8 -> f32 (OCANNL's fp8 is
       e5m2), which wmma cannot express. sm_89+ (Ada); the [(mma-fp8)] marker makes [cuda_to_ptx]
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
          (* Pointer declarations use [typ_of_prec] of the operand's own precision, which coincides
             with [wmma_combo]'s fragment element types (tf32 fragments load from plain [float]
             pointers). *)
          let combo = wmma_combo ~a_prec ~b_prec ~d_prec in
          let loadable = function
            | `Device | `Shared -> true (* generic-address loads cover both *)
            | `Thread | `Fragment _ -> false
          in
          let plain = function `Plain -> true | `Swizzled_b128 -> false in
          (* An operand eligible for the warp-cooperative [ldmatrix] load (gh-ocannl-481 item 3):
             the swizzled 16-byte-unit layout, in shared memory, which is the only combination whose
             per-lane row addresses [ldmatrix] can both reach and de-conflict. Everything else keeps
             the per-lane gathers, which stay correct for plain shared tiles and device pointers
             alike. Note the asymmetry the arms below rely on: eligibility is [space AND layout],
             but the DECLINE is on the layout alone — a swizzled operand this arm cannot [ldmatrix]
             must not silently fall through to row-major gathers, whatever space it sits in. *)
          let ldm = function
            | `Shared, `Swizzled_b128 -> true
            | (`Shared | `Device | `Thread | `Fragment _), (`Plain | `Swizzled_b128) -> false
          in
          let a_swz = ldm (a_space, a_layout) and b_swz = ldm (b_space, b_layout) in
          (* The shared-window address of the element at (row, col) of a [Swizzle_b128] tile whose
             minor dim is [ld]: the column's 16-byte-unit index is XORed with the low bits of the
             row (see [Low_level.Swizzle_b128]), everything within a unit left alone. Every fragment
             entry point below has [col] a multiple of [u], so the within-unit remainder is zero and
             drops out. [ldmatrix] is a [.shared] instruction, hence the conversion out of the
             generic window. *)
          let swz_saddr ~ptr ~ld ~prec ~row ~col =
            let u = 16 / Ops.prec_in_bytes prec in
            let s = Int.floor_log2 u in
            let units = ld / u in
            Printf.sprintf
              "(unsigned)__cvta_generic_to_shared(%s + (%s) * %d + (((((%s) >> %d) ^ ((%s) & %d)) \
               << %d)))"
              ptr row ld col s row (units - 1) s
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
          let is_fp8_combo =
            match (a_prec, b_prec, d_prec) with
            | Ops.Fp8_prec _, Ops.Fp8_prec _, Ops.Single_prec _ -> true
            | _ -> false
          in
          (* gh-ocannl-545: [nvcuda::wmma] pairs [__nv_bfloat16] operands with a [float] accumulator
             only — [crt/mma.hpp] declares no bf16 accumulator fragment — so a uniformly-bf16
             network, where the GEMM's destination node is itself bf16, has no wmma combination and
             used to render the lane-0 scalar fallback under a tensorized label. The hardware is not
             the limit: [mma.sync] accumulates bf16 operands in per-lane f32 registers, which we can
             convert at the [d] boundary because that layout is architecturally defined (unlike wmma
             fragments, whose element mapping is opaque and forces a warp-uniform [float] staging
             buffer). Rendered by the inline-PTX arm below. *)
          let is_bf16_uniform =
            match (a_prec, b_prec, d_prec) with
            | Ops.Bfloat16_prec _, Ops.Bfloat16_prec _, Ops.Bfloat16_prec _ -> true
            | _ -> false
          in
          (* gh-ocannl-680: under [Numerics.Fp16_wide] the uniform-f16 combination renders through
             the same m16n8k16 inline-PTX arm — the PTX ISA's "Matrix Fragments for mma.m16n8k16"
             layouts are shared by .f16 and .bf16 — holding f32 per-lane registers across the whole
             k extent and converting once at the [d] boundary: exactly the residency [accum_prec]
             gives the serial legs under that policy. Under [Fp16_auto]/[Fp16_narrow] the wmma
             f16-accumulate combo above renders instead and this stays false. *)
          let is_f16_uniform_wide =
            match (a_prec, b_prec, d_prec) with
            | Ops.Half_prec _, Ops.Half_prec _, Ops.Half_prec _ -> Numerics.fp16_accum_wide ()
            | _ -> false
          in
          (* The element-type spellings of the m16n8k16 arm, shared by its two 16-bit forms: the C
             type, the bits-as-ushort intrinsic, the widening and narrowing conversions, the
             instruction's element infix, and the marker [gpu_arch_options] greps for the arch floor
             (sm_80 for both). *)
          let mma16_spellings =
            if is_bf16_uniform then
              Some
                ( "__nv_bfloat16",
                  "__bfloat16_as_ushort",
                  "__bfloat162float",
                  "__float2bfloat16",
                  "bf16",
                  "mma-bf16" )
            else if is_f16_uniform_wide then
              Some ("__half", "__half_as_ushort", "__half2float", "__float2half", "f16", "mma-f16")
            else None
          in
          (* fp8's [ldmatrix] eligibility is one-sided per operand, and the sides are opposite
             (gh-ocannl-481 item 3, D2). [ldmatrix.b16] moves 16-bit units, so it can build a
             fragment register only when the 4 fp8 bytes that register holds are CONTIGUOUS in
             storage. For A those 4 bytes are 4 consecutive [k] at fixed [m] — contiguous exactly
             when A is stored row-major ([ta = false]); for B they are 4 consecutive [k] at fixed
             [n] — contiguous exactly when B is stored transposed ([tb = true]). The other
             orientations need the 8-bit [ldmatrix] forms (Blackwell-only), so they keep their byte
             gathers; per-operand choice, one statement. *)
          let a_ldm = is_fp8_combo && a_swz && not ta in
          let b_ldm = is_fp8_combo && b_swz && tb in
          if
            is_fp8_combo && plain d_layout
            && (plain a_layout || a_ldm)
            && (plain b_layout || b_ldm)
            && m % 16 = 0
            && n % 8 = 0
            && k % 32 = 0
            && loadable d_space && loadable a_space && loadable b_space
            && min_compute_capability () >= 89
          then
            (* Raw [mma.sync] with the architecture-defined per-lane fragment layouts of m16n8k32
               (PTX ISA "Matrix Fragments for mma.m16n8k32"; fp8 shares the .s8/.u8 layouts). Thread
               [lane], with groupID g = lane>>2 and threadID-in-group t = lane&3, holds: A (16x32,
               row-major source) regs a0..a3 = 4 consecutive bytes at rows {g, g+8} x column groups
               {4t, 4t+16}; B (32x8, column-major fragment from our row-major source) regs b0,b1 =
               rows {4t..4t+3, 16+4t..16+4t+3} down column g, gathered bytewise (stride ldb);
               accumulator D (16x8 f32) regs d0..d3 = rows {g, g+8} x columns {2t, 2t+1}.
               Low-indexed elements sit in low-order bytes of each .b32 register. Byte loads impose
               no alignment or stride constraints and read generic addresses, covering both
               __shared__ tiles and device pointers.

               Transposed operand storage ([ta]/[tb], gh-ocannl-481 item 1) is bookkeeping here, not
               a layout constraint: every fragment byte is addressed by its logical (row, col)
               through [a_at]/[b_at], which index the STORED matrix at (col, row) under the flag
               while keeping the operand's own leading dimension. The fragment content is unchanged,
               only its addresses are — so the gradient GEMMs ([dA = g.B^T], [dB = A^T.g]), whose
               layouts are exactly these, tensorize instead of silently scalar-falling-back. *)
            let open PPrint in
            let mt = m / 16 and nt = n / 8 and kt = k / 32 in
            let pack4 name base offs =
              Printf.sprintf
                "unsigned %s = (unsigned)%s[%s] | ((unsigned)%s[%s] << 8) | ((unsigned)%s[%s] << \
                 16) | ((unsigned)%s[%s] << 24);"
                name base offs.(0) base offs.(1) base offs.(2) base offs.(3)
            in
            (* Byte offset of logical element (row, col) in an operand stored under its own leading
               dimension, transposed or not (fp8 is one byte per element, so element index = byte
               offset from the [unsigned char *] base). *)
            let elem_at ~ld ~transposed ~row ~col =
              if transposed then Printf.sprintf "(%s) * %d + (%s)" col ld row
              else Printf.sprintf "(%s) * %d + (%s)" row ld col
            in
            let a_at ~row ~col = elem_at ~ld:lda ~transposed:ta ~row ~col in
            let b_at ~row ~col = elem_at ~ld:ldb ~transposed:tb ~row ~col in
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
               [m16n8k16] A arrangement at byte granularity; B stored transposed is (n rows 0-7) x
               (byte columns {0-15, 16-31}), giving b0/b1 in that order. With [q = lane >> 3] naming
               the matrix, lane [r = lane & 7] supplies row [r] of matrix [q]. *)
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
                  (swz_saddr ~ptr:"__mma_bp" ~ld:ldb ~prec:b_prec ~row:"__ni * 8 + (__mma_lid & 7)"
                     ~col:"__ki * 32 + 16 * ((__mma_lid >> 3) & 1)")
            in
            let barrier = "__syncthreads();" in
            let body_lines =
              [
                barrier;
                "unsigned __mma_lid;";
                "asm(\"mov.u32 %0, %%laneid;\" : \"=r\"(__mma_lid));";
                "const int __mma_g = (int)(__mma_lid >> 2);";
                "const int __mma_t = (int)(__mma_lid & 3);";
                Printf.sprintf "for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                Printf.sprintf
                  "    float *__mma_dr0 = __mma_dp + (__mi * 16 + __mma_g) * %d + __ni * 8 + 2 * \
                   __mma_t;"
                  ldd;
                Printf.sprintf "    float *__mma_dr1 = __mma_dr0 + 8 * %d;" ldd;
                "    float __mma_d0 = __mma_dr0[0], __mma_d1 = __mma_dr0[1];";
                "    float __mma_d2 = __mma_dr1[0], __mma_d3 = __mma_dr1[1];";
                Printf.sprintf "    for (int __ki = 0; __ki < %d; ++__ki) {" kt;
              ]
              @ (if a_ldm then a_ldm_lines
                 else
                   [
                     a_reg "__mma_a0" ~lo:true ~c:0;
                     a_reg "__mma_a1" ~lo:false ~c:0;
                     a_reg "__mma_a2" ~lo:true ~c:16;
                     a_reg "__mma_a3" ~lo:false ~c:16;
                   ])
              @ (if b_ldm then b_ldm_lines else [ b_reg "__mma_b0" ~r:0; b_reg "__mma_b1" ~r:16 ])
              @ [
                  "      asm(\"mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 \"";
                  "          \"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"";
                  "          : \"+f\"(__mma_d0), \"+f\"(__mma_d1), \"+f\"(__mma_d2), \
                   \"+f\"(__mma_d3)";
                  "          : \"r\"(__mma_a0), \"r\"(__mma_a1), \"r\"(__mma_a2), \"r\"(__mma_a3), \
                   \"r\"(__mma_b0), \"r\"(__mma_b1));";
                  "    }";
                  "    __mma_dr0[0] = __mma_d0; __mma_dr0[1] = __mma_d1;";
                  "    __mma_dr1[0] = __mma_d2; __mma_dr1[1] = __mma_d3;";
                  "  }";
                  "}";
                  barrier;
                ]
            in
            let cast_ptr name ptr =
              string (Printf.sprintf "const unsigned char *%s = (const unsigned char *)(" name)
              ^^ ptr ^^ string ");"
            in
            let body ~a_ptr ~b_ptr =
              string "float *__mma_dp = " ^^ d_ptr ^^ semi ^^ hardline ^^ cast_ptr "__mma_ap" a_ptr
              ^^ hardline ^^ cast_ptr "__mma_bp" b_ptr ^^ hardline
              ^^ separate_map hardline string body_lines
            in
            Some
              (fun ~a_ptr ~b_ptr ->
                group
                  (string
                     (Printf.sprintf "{ /* tile_mma %dx%dx%d (mma-fp8) e5m2%s */" m n k
                        (ldm_tag ~a:a_ldm ~b:b_ldm))
                  ^^ nest 2 (hardline ^^ body ~a_ptr ~b_ptr)
                  ^^ hardline ^^ rbrace))
          else if
            (* Unlike fp8, both 16-bit operands can come in through [ldmatrix] in either storage
               orientation: the fragment registers hold 16-bit element PAIRS, and the pair a lane
               needs is contiguous under one of the two forms — [.trans] transposes each 8x8 tile on
               distribution, which is exactly the difference between the two orientations. *)
            Option.is_some mma16_spellings && plain d_layout
            && (plain a_layout || a_swz)
            && (plain b_layout || b_swz)
            && m % 16 = 0
            && n % 8 = 0
            && k % 16 = 0
            && loadable d_space && loadable a_space && loadable b_space
            && min_compute_capability () >= 80
          then
            (* Raw [mma.sync] with the architecturally-defined per-lane fragment layouts of m16n8k16
               (PTX ISA "Matrix Fragments for mma.m16n8k16", shared by .f16 and .bf16 — which is
               what lets [mma16_spellings] parameterize this one body over both element types).
               Thread [lane], with groupID g = lane>>2 and threadID-in-group t = lane&3, holds: A
               (16x16) regs a0..a3 = the element pairs at rows {g, g+8} x column pairs {2t, 2t+8}; B
               (16x8, column-major fragment) regs b0,b1 = row pairs {2t, 2t+8} down column g;
               accumulator D (16x8 f32) regs d0..d3 = rows {g, g+8} x columns {2t, 2t+1}. The
               lower-indexed element of each pair sits in the low half of its .b32 register.

               Unlike the fp8 arm, transposed operand storage ([ta]/[tb]) is supported: every gather
               goes through [a_at]/[b_at], which index the STORED matrix at (col, row) under the
               flag while keeping the operand's own leading dimension — the fragment content is
               unchanged, only its addresses are. Element-wise 16-bit loads impose no alignment or
               stride constraint and read generic addresses, so both __shared__ tiles and device
               pointers are covered.

               The accumulator is read from and written back to the 16-bit [d] once per statement,
               with the whole [k] extent accumulated in f32 registers in between. For bf16 that is
               strictly better rounding than the scalar fallback this replaces, which rounds to bf16
               per term; for f16 it is the [Numerics.Fp16_wide] residency [accum_prec] gives every
               serial rendering (gh-ocannl-680). *)
            let open PPrint in
            let elt_typ, as_ushort, to_float, from_float, instr_elt, marker =
              Option.value_exn mma16_spellings
            in
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
              Printf.sprintf "unsigned %s = (unsigned)%s(%s[%s]) | ((unsigned)%s(%s[%s]) << 16);"
                name as_ushort base i0 as_ushort base i1
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
            (* The [ldmatrix] entry points. Naming the matrix [q = lane >> 3] and the row within it
               [r = lane & 7], the four A matrices are (m rows {0-7, 8-15}) x (k columns {0-7,
               8-15}) in register order a0..a3, and the two B matrices are (k rows {0-7, 8-15}) x (n
               columns 0-7) in order b0, b1. Each is read as 8 rows of 8 contiguous 16-bit elements
               from the operand's OWN storage, so the transposed orientations swap which index walks
               the rows and add [.trans] exactly when the fragment's minor direction is the stored
               major one. *)
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
            let body_lines =
              [
                barrier;
                "unsigned __mma_lid;";
                "asm(\"mov.u32 %0, %%laneid;\" : \"=r\"(__mma_lid));";
                "const int __mma_g = (int)(__mma_lid >> 2);";
                "const int __mma_t = (int)(__mma_lid & 3);";
                Printf.sprintf "for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                Printf.sprintf
                  "    %s *__mma_dr0 = __mma_dp + (__mi * 16 + __mma_g) * %d + __ni * 8 + 2 * \
                   __mma_t;"
                  elt_typ ldd;
                Printf.sprintf "    %s *__mma_dr1 = __mma_dr0 + 8 * %d;" elt_typ ldd;
                Printf.sprintf "    float __mma_d0 = %s(__mma_dr0[0]), __mma_d1 = %s(__mma_dr0[1]);"
                  to_float to_float;
                Printf.sprintf "    float __mma_d2 = %s(__mma_dr1[0]), __mma_d3 = %s(__mma_dr1[1]);"
                  to_float to_float;
                Printf.sprintf "    for (int __ki = 0; __ki < %d; ++__ki) {" kt;
              ]
              @ (if a_swz then a_ldm_lines
                 else
                   [
                     a_reg "__mma_a0" ~lo:true ~c:0;
                     a_reg "__mma_a1" ~lo:false ~c:0;
                     a_reg "__mma_a2" ~lo:true ~c:8;
                     a_reg "__mma_a3" ~lo:false ~c:8;
                   ])
              @ (if b_swz then b_ldm_lines else [ b_reg "__mma_b0" ~r:0; b_reg "__mma_b1" ~r:8 ])
              @ [
                  Printf.sprintf "      asm(\"mma.sync.aligned.m16n8k16.row.col.f32.%s.%s.f32 \""
                    instr_elt instr_elt;
                  "          \"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"";
                  "          : \"+f\"(__mma_d0), \"+f\"(__mma_d1), \"+f\"(__mma_d2), \
                   \"+f\"(__mma_d3)";
                  "          : \"r\"(__mma_a0), \"r\"(__mma_a1), \"r\"(__mma_a2), \"r\"(__mma_a3), \
                   \"r\"(__mma_b0), \"r\"(__mma_b1));";
                  "    }";
                  Printf.sprintf "    __mma_dr0[0] = %s(__mma_d0); __mma_dr0[1] = %s(__mma_d1);"
                    from_float from_float;
                  Printf.sprintf "    __mma_dr1[0] = %s(__mma_d2); __mma_dr1[1] = %s(__mma_d3);"
                    from_float from_float;
                  "  }";
                  "}";
                  barrier;
                ]
            in
            let ptr_decl name typ ptr =
              string (Printf.sprintf "%s *%s = " typ name) ^^ ptr ^^ semi
            in
            let body ~a_ptr ~b_ptr =
              ptr_decl "__mma_dp" elt_typ d_ptr ^^ hardline
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
                     (Printf.sprintf "{ /* tile_mma %dx%dx%d (%s)%s */" m n k marker
                        (ldm_tag ~a:a_swz ~b:b_swz))
                  ^^ nest 2 (hardline ^^ body ~a_ptr ~b_ptr)
                  ^^ hardline ^^ rbrace))
          else
            (* wmma fragments are opaque: [load_matrix_sync] assumes row-major (or column-major)
               pointer+stride storage and there is no supported way to feed one from [ldmatrix]
               destination registers. So the whole template path declines swizzled operands
               (gh-ocannl-481 item 3, D2) — the caller then reaches for the scalar fallback, or, in
               an accumulator-resident scope, for the inline-PTX arms above. *)
            match if plain d_layout && plain a_layout && plain b_layout then combo else None with
            | None -> None
            | Some
                {
                  wc_ab_typ = ab_typ;
                  wc_acc_typ = acc_typ;
                  wc_tm;
                  wc_tn;
                  wc_tk;
                  wc_ab_ld_mult = ab_ld_mult;
                  wc_d_ld_mult = d_ld_mult;
                  wc_min_cc = min_cc;
                  wc_marker = marker;
                  wc_cvt_tf32 = cvt_tf32;
                } -> (
                let open PPrint in
                let mt = m / wc_tm and nt = n / wc_tn and kt = k / wc_tk in
                let frag kind typ layout =
                  Printf.sprintf "nvcuda::wmma::fragment<nvcuda::wmma::%s, %d, %d, %d, %s%s>" kind
                    wc_tm wc_tn wc_tk typ
                    (match layout with Some l -> ", nvcuda::wmma::" ^ l | None -> "")
                in
                let ptr_decl name typ ptr =
                  string (Printf.sprintf "%s *%s = " typ name) ^^ ptr ^^ semi
                in
                let a_frag_layout = if ta then "col_major" else "row_major" in
                let b_frag_layout = if tb then "col_major" else "row_major" in
                let barrier = "__syncthreads();" in
                (* tf32 fragments load raw f32 bits; the explicit elementwise [__float_to_tf32]
                   performs the mantissa truncation the intrinsic expects (CUDA programming guide
                   requirement — without it the multiplicand bits are implementation-defined). *)
                let cvt_lines frag_expr =
                  if cvt_tf32 then
                    [
                      Printf.sprintf
                        "    for (int __t = 0; __t < %s.num_elements; ++__t) %s.x[__t] = \
                         nvcuda::wmma::__float_to_tf32(%s.x[__t]);"
                        frag_expr frag_expr frag_expr;
                    ]
                  else []
                in
                (* The serial [k] extent of one block statement: per step, the B-row fragment loads,
                   the A fragment loads, and the mma updates of the accumulator array [acc]. Shared
                   between the self-contained rendering ([acc] = the block-local [__mma_acc]) and
                   the update-only rendering against a resident fragment array (gh-ocannl-480). *)
                let k_loop_lines ~ab_typ ~acc =
                  [
                    Printf.sprintf "for (int __ki = 0; __ki < %d; ++__ki) {" kt;
                    Printf.sprintf "  %s __mma_bf[%d];"
                      (frag "matrix_b" ab_typ (Some b_frag_layout))
                      nt;
                    Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                    (* Transposed storage ([tb]): the stored matrix is the role's transpose — index
                       it at (col, row) and declare the fragment [col_major]; the leading dimension
                       stays the operand's own. Same for [ta] below. *)
                    (if tb then
                       Printf.sprintf
                         "    nvcuda::wmma::load_matrix_sync(__mma_bf[__ni], __mma_bp + __ni * %d \
                          * %d + __ki * %d, %d);"
                         wc_tn ldb wc_tk ldb
                     else
                       Printf.sprintf
                         "    nvcuda::wmma::load_matrix_sync(__mma_bf[__ni], __mma_bp + __ki * %d \
                          * %d + __ni * %d, %d);"
                         wc_tk ldb wc_tn ldb);
                  ]
                  @ cvt_lines "__mma_bf[__ni]"
                  @ [
                      "  }";
                      Printf.sprintf "  for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                      Printf.sprintf "    %s __mma_af;"
                        (frag "matrix_a" ab_typ (Some a_frag_layout));
                      (if ta then
                         Printf.sprintf
                           "    nvcuda::wmma::load_matrix_sync(__mma_af, __mma_ap + __ki * %d * %d \
                            + __mi * %d, %d);"
                           wc_tk lda wc_tm lda
                       else
                         Printf.sprintf
                           "    nvcuda::wmma::load_matrix_sync(__mma_af, __mma_ap + __mi * %d * %d \
                            + __ki * %d, %d);"
                           wc_tm lda wc_tk lda);
                    ]
                  @ cvt_lines "__mma_af"
                  @ [
                      Printf.sprintf "    for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                      Printf.sprintf
                        "      nvcuda::wmma::mma_sync(%s[__mi][__ni], __mma_af, __mma_bf[__ni], \
                         %s[__mi][__ni]);"
                        acc acc;
                      "    }";
                      "  }";
                      "}";
                    ]
                in
                let ab_ok =
                  m % wc_tm = 0
                  && n % wc_tn = 0
                  && k % wc_tk = 0
                  && lda % ab_ld_mult = 0
                  && ldb % ab_ld_mult = 0
                  && loadable a_space && loadable b_space
                  && min_compute_capability () >= min_cc
                in
                match d_space with
                | `Fragment fragment when ab_ok ->
                    (* Update-only steps against the accumulator-fragment array [fragment] declared
                       by the enclosing [mma_fragment_syntax] scope (gh-ocannl-480): no
                       per-statement load/store of [d]. The trailing barrier releases the staged
                       shared tiles for the next serial iteration's cooperative loads. *)
                    let body ~a_ptr ~b_ptr =
                      ptr_decl "__mma_ap" ("const " ^ typ_of_prec a_prec) a_ptr
                      ^^ hardline
                      ^^ ptr_decl "__mma_bp" ("const " ^ typ_of_prec b_prec) b_ptr
                      ^^ hardline
                      ^^ separate_map hardline string
                           (k_loop_lines ~ab_typ ~acc:fragment @ [ barrier ])
                    in
                    Some
                      (fun ~a_ptr ~b_ptr ->
                        group
                          (string
                             (Printf.sprintf "{ /* tile_mma fragment update %dx%dx%d (wmma%s) */" m
                                n k marker)
                          ^^ nest 2 (hardline ^^ body ~a_ptr ~b_ptr)
                          ^^ hardline ^^ rbrace))
                | _ when ab_ok && ldd % d_ld_mult = 0 && loadable d_space ->
                    let body_lines =
                      [
                        barrier;
                        Printf.sprintf "%s __mma_acc[%d][%d];" (frag "accumulator" acc_typ None) mt
                          nt;
                        Printf.sprintf "for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                        Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                        Printf.sprintf
                          "    nvcuda::wmma::load_matrix_sync(__mma_acc[__mi][__ni], __mma_dp + \
                           __mi * %d * %d + __ni * %d, %d, nvcuda::wmma::mem_row_major);"
                          wc_tm ldd wc_tn ldd;
                        "  }";
                        "}";
                      ]
                      @ k_loop_lines ~ab_typ ~acc:"__mma_acc"
                      @ [
                          Printf.sprintf "for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                          Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                          Printf.sprintf
                            "    nvcuda::wmma::store_matrix_sync(__mma_dp + __mi * %d * %d + __ni \
                             * %d, __mma_acc[__mi][__ni], %d, nvcuda::wmma::mem_row_major);"
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
                          (string (Printf.sprintf "{ /* tile_mma %dx%dx%d (wmma%s) */" m n k marker)
                          ^^ nest 2 (hardline ^^ body ~a_ptr ~b_ptr)
                          ^^ hardline ^^ rbrace))
                | _ -> None))

    (* Cross-[k_o] accumulator residency (gh-ocannl-480), following the Metal emission: the marked
       local tile becomes a wmma accumulator-fragment array declared once, loaded from the backing
       target before the serial reduction body and stored after it; the nested [Tile_mma] sees
       [`Fragment] and emits update-only mma steps. The acceptance conditions mirror [mma_syntax]'s
       wmma arm exactly (via [wmma_combo] and the same extent/stride/space/arch checks), so an
       accepted scope never strands its inner call. The fp8 inline-PTX combination declines: its
       accumulator lives in per-lane f32 registers with the m16n8k32 layout, not wmma fragments, so
       it keeps the per-[k_o] rendering through the caller's target-aliasing path. Swizzled operands
       decline for the same reason the wmma arm of [mma_syntax] does — opaque fragments cannot be
       fed from [ldmatrix] — which is precisely what routes a swizzled staged bf16 leg to the
       inline-PTX arm via the caller's target aliasing (gh-ocannl-481 item 3, D3). *)
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
          let loadable = function
            | `Device | `Shared -> true (* generic-address loads cover both *)
            | `Thread | `Fragment _ -> false
          in
          let plain = function `Plain -> true | `Swizzled_b128 -> false in
          match
            if plain d_layout && plain a_layout && plain b_layout then
              wmma_combo ~a_prec ~b_prec ~d_prec
            else None
          with
          | Some
              {
                wc_acc_typ = acc_typ;
                wc_tm;
                wc_tn;
                wc_tk;
                wc_ab_ld_mult = ab_ld_mult;
                wc_d_ld_mult = d_ld_mult;
                wc_min_cc = min_cc;
                _;
              }
            when m % wc_tm = 0
                 && n % wc_tn = 0
                 && k % wc_tk = 0
                 && lda % ab_ld_mult = 0
                 && ldb % ab_ld_mult = 0
                 && ldd % d_ld_mult = 0
                 && loadable d_space && loadable a_space && loadable b_space
                 && min_compute_capability () >= min_cc ->
              let open PPrint in
              let mt = m / wc_tm and nt = n / wc_tn in
              let barrier = "__syncthreads();" in
              let acc_frag =
                Printf.sprintf "nvcuda::wmma::fragment<nvcuda::wmma::accumulator, %d, %d, %d, %s>"
                  wc_tm wc_tn wc_tk acc_typ
              in
              (* Bracketing barriers: sibling statements (the zeroing of [d], cooperative staging)
                 execute lane-partitioned, so the fragment loads must observe the other lanes'
                 writes, and later statements the stores. *)
              let lines_before =
                [
                  barrier;
                  Printf.sprintf "%s %s[%d][%d];" acc_frag fragment mt nt;
                  Printf.sprintf "for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                  Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                  Printf.sprintf
                    "    nvcuda::wmma::load_matrix_sync(%s[__mi][__ni], __mma_dp + __mi * %d * %d \
                     + __ni * %d, %d, nvcuda::wmma::mem_row_major);"
                    fragment wc_tm ldd wc_tn ldd;
                  "  }";
                  "}";
                  "/* wmma fragment reduction body begins */";
                ]
              in
              let lines_after =
                [
                  "/* wmma fragment reduction body ends */";
                  Printf.sprintf "for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                  Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                  Printf.sprintf
                    "    nvcuda::wmma::store_matrix_sync(__mma_dp + __mi * %d * %d + __ni * %d, \
                     %s[__mi][__ni], %d, nvcuda::wmma::mem_row_major);"
                    wc_tm ldd wc_tn fragment ldd;
                  "  }";
                  "}";
                  barrier;
                ]
              in
              let d_decl =
                string (Printf.sprintf "%s *__mma_dp = " (typ_of_prec d_prec)) ^^ d_ptr ^^ semi
              in
              Some
                (group
                   (string (Printf.sprintf "{ /* wmma fragment %dx%d across k_o */" m n)
                   ^^ nest 2
                        (hardline ^^ d_decl ^^ hardline
                        ^^ separate_map hardline string lines_before
                        ^^ hardline ^^ body () ^^ hardline
                        ^^ separate_map hardline string lines_after)
                   ^^ hardline ^^ rbrace))
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
    ^
    if Utils.debug_log_from_routines () then "\n__device__ int printf (const char * format, ... );"
    else ""

  module Compile = C_syntax.Compile_driver (Cuda_syntax_config)

  let%diagn2_sexp compile ~name bindings lowered =
    let ptx, kparams, name, launch =
      Compile.compile ~name bindings lowered ~includes:(cuda_includes ())
        ~builtins:Builtins_cuda.builtins
        ~conditional_includes:Cuda_like_config.Cuda.conditional_includes ~compile_source:cuda_to_ptx
        ()
    in
    { ptx; kparams; bindings; name; launch }

  let%diagn2_sexp compile_batch ~names bindings lowereds =
    let ptx, kparams_and_names =
      Compile.compile_batch ~names bindings lowereds ~includes:(cuda_includes ())
        ~builtins:Builtins_cuda.builtins
        ~conditional_includes:Cuda_like_config.Cuda.conditional_includes ~compile_source:cuda_to_ptx
        ()
    in
    { ptx; kparams_and_names; bindings }

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
    set_ctx ctx;
    let run_module = load_module code.ptx in
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
    let run_module = load_module code_batch.ptx in
    prior_context.device.dev.set_builtins_in run_module;
    let procs =
      Array.map code_batch.kparams_and_names ~f:(fun (kparams, name, launch) ->
          link_proc ~prior_context ~name ~kparams ~launch ~ctx_buffers lowered_bindings run_module)
    in
    (lowered_bindings, procs)

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
      let cache : (string, Cu.Graph.exec) Hashtbl.t = Hashtbl.create (module String) in
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
                   | Some exec -> Cu.Graph.launch exec device.runner
                   | None -> (
                       match capture () with
                       | exec ->
                           if Queue.length order >= max_cached_graphs then (
                             let victim = Queue.dequeue_exn order in
                             (* The evicted exec may still have a pending launch on the stream. *)
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
                             "ocannl: disabling CUDA graph capture for routine %s (%s: %s)\n%!" name
                             message
                             (Sexp.to_string_hum @@ Cu.sexp_of_result status);
                           run_plain ())));
           })

  let get_global_debug_info () =
    Sexp.message "cuda_global_debug"
      [ ("live_streams", [%sexp_of: int] @@ Cu.Stream.get_total_live_streams ()) ]

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
           max_grid_yz = Some 65535;
           (* Tensor cores (tensorize-mma T3): the 32-thread warp cooperates on 16x16x16 wmma tiles
              from sm_70 up; [mma_format_tiles] advertises the divergent fp8 16x8x32, tf32 16x16x8
              and uniform-bf16 16x8x16 shapes to typed autotune seeds. Precision combinations are
              ultimately decided per call by [mma_syntax] — but each entry here mirrors an arm of
              that hook, INCLUDING its accumulator format and arch floor (gh-ocannl-545), because a
              seed the hook will decline is a candidate the tuner times as scalar code under a
              tensorized label. *)
           mma =
             (let cc = min_compute_capability () in
              let entry ~min_cc key tile = if cc >= min_cc then Some (key, tile) else None in
              if cc >= 70 then
                Some
                  {
                    Backend_intf.mma_simd_width = 32;
                    mma_tile = (16, 16, 16);
                    mma_format_tiles =
                      List.filter_opt
                        [
                          (* wmma, sm_70+: f16 operands against either accumulator width. *)
                          entry ~min_cc:70
                            (Backend_intf.Mma_f16, Backend_intf.Mma_f16, Backend_intf.Mma_f32)
                            (16, 16, 16);
                          entry ~min_cc:70
                            (Backend_intf.Mma_f16, Backend_intf.Mma_f16, Backend_intf.Mma_f16)
                            (16, 16, 16);
                          (* wmma, sm_80+: bf16 operands accumulate in f32 only. *)
                          entry ~min_cc:80
                            (Backend_intf.Mma_bf16, Backend_intf.Mma_bf16, Backend_intf.Mma_f32)
                            (16, 16, 16);
                          (* Inline-PTX [mma.sync] m16n8k16, sm_80+: the uniform-bf16 combination
                             wmma cannot express. *)
                          entry ~min_cc:80
                            (Backend_intf.Mma_bf16, Backend_intf.Mma_bf16, Backend_intf.Mma_bf16)
                            (16, 8, 16);
                          (* Inline-PTX [mma.sync] m16n8k32, sm_89+. *)
                          entry ~min_cc:89
                            ( Backend_intf.Mma_fp8_e5m2,
                              Backend_intf.Mma_fp8_e5m2,
                              Backend_intf.Mma_f32 )
                            (16, 8, 32);
                          (* wmma tf32, sm_80+; [mma_input_formats_of_prec] additionally gates this
                             on the numerics policy. *)
                          entry ~min_cc:80
                            (Backend_intf.Mma_tf32, Backend_intf.Mma_tf32, Backend_intf.Mma_f32)
                            (16, 16, 8);
                        ];
                    (* gh-ocannl-680: under [Numerics.Fp16_wide] the uniform-f16 key above renders
                       through the f32-accumulate inline-PTX m16n8k16 arm (sm_80+, sharing the bf16
                       form's body). The advertised (16, 16, 16) tile stays valid for it — its
                       divisibility constraints (m%16, n%8, k%16) are implied — merely conservative
                       about n. The scope list contains only [Mma_per_statement]: CUDA's wmma
                       combination table has no uniform-f16 wide fragment arm, so a staged outer-k
                       split must not inherit this inner tile's capability (gh-ocannl-836). Below
                       sm_80 the arm cannot render and the list is empty, withholding every
                       uniform-f16 candidate under the wide policy instead of timing scalar
                       fallbacks under a tensorized label (gh-ocannl-545). *)
                    mma_f16_wide_acc_scopes =
                      (if cc >= 80 then [ Backend_intf.Mma_per_statement ] else []);
                    (* gh-ocannl-838: the uniform-bf16 key is the same inline-PTX arm, f32 in
                       hardware whatever the policy, so [Numerics.Bf16_wide] changes no rendering
                       here — but, exactly as for f16, a staged outer-k split would store bf16 at
                       every block boundary, so the fragment scope is absent. *)
                    mma_bf16_wide_acc_scopes =
                      (if cc >= 80 then [ Backend_intf.Mma_per_statement ] else []);
                    (* Swizzled staged tiles (gh-ocannl-481 item 3, D3): only the inline-PTX arms
                       can read them, and only in the orientations the staged sketches mint. That is
                       the uniform-bf16 combination — both its operands' fragment registers hold
                       16-bit pairs that are contiguous in the roles' own orientations. fp8 is
                       deliberately absent: its A side qualifies but its B side does not (4 fp8
                       bytes of a B register are strided at that orientation), so a swizzled fp8
                       twin would render the scalar fallback under a tensorized label. *)
                    mma_staged_layouts =
                      List.filter_opt
                        [
                          entry ~min_cc:80
                            (Backend_intf.Mma_bf16, Backend_intf.Mma_bf16, Backend_intf.Mma_bf16)
                            Backend_intf.Mma_swizzled_b128;
                        ];
                    (* gh-ocannl-487 phase 2: the depth-2 twins are worth proposing exactly where
                       the [cp.async] arm renders them (sm_80+; [Cuda_syntax_config.async_copy] has
                       the matching gate) — pre-Ampere the twin would be the portable synchronous
                       form, whose occupancy cost was measured, not hypothesized (phase 1: ~1.4-1.5x
                       on Metal). Depth 2 only: the wait-all emission has single-step lookahead;
                       deeper pipelines need commit_group/wait_group N. *)
                    mma_pipeline_depths = (if cc >= 80 then [ 2 ] else []);
                  }
              else None);
           simd_vector_bytes = 0;
           native_fp16_arithmetic = false;
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

           This status only. [NVRTC_ERROR_BUILTIN_OPERATION_FAILURE] in particular stays
           unclassified: it reports nvrtc failing on its own builtins, which an incomplete or
           mismatched CUDA installation reproduces for every candidate — precisely the
           fails-identically class this classifier must let propagate. *)
        | "NVRTC_ERROR_COMPILATION" ->
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
