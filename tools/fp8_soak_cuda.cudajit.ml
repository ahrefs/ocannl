(* The CUDA arm of the fp8 soak (gh-ocannl-657): narrow every input with [__nv_fp8_e5m2] and hand
   the codes back. Nothing here knows what the answers are compared against, what the vendor type is
   called in a claim, or which narrowing spellings this platform has -- fp8_soak.ml owns all of that
   (its [ARM] signature and [cuda_vendor] record), because THIS FILE IS COMPILED ONLY ON A BOX WITH
   cudajit and every edit to it is made blind everywhere else (gh-ocannl-758). What belongs here is
   the kernel source and the calls into cudajit; anything else belongs in fp8_soak.ml.

   The kernel casts to the vendor type exactly as [Cuda_backend]'s [convert_precision] emits it:
   [(__nv_fp8_e5m2)x] for a float and for a double, which is a saturating round-to-nearest-even
   conversion done by the hardware. Compiled WITHOUT [--use_fast_math] (which the backend does pass)
   because fast math is about arithmetic and this program does none: the conversion is the whole
   kernel, and the soak should not have to reason about what a relaxed floating-point mode does to
   the thing being measured. *)

open Base
module Cu = Cuda

(* Update this whenever you compile this file on a box that has cudajit -- the run header prints it,
   and it is what tells the next editor whether they are editing blind (gh-ocannl-758). *)
let last_compiled = "on rog-nv-wsl (RTX 5070 Ti Laptop, CUDA 13.3), 2026-08-27, staging PR #490"
let built = true

type bytes_buf =
  (int, Stdlib.Bigarray.int8_unsigned_elt, Stdlib.Bigarray.c_layout) Stdlib.Bigarray.Array1.t

let source =
  {|
#include <cuda_fp8.h>

/* What __CUDA_ARCH__ the kernels above were actually built with, asked of the DEVICE rather than
   inferred from the options we think we passed -- nvrtc's default target when no
   --gpu-architecture is given is a toolkit detail, and this program's whole subject is not
   assuming things about vendor conversions. cuda_fp8.hpp's conversions are hardware at >= 890 and
   the header's own software emulation below, so this value is what says which was swept. */
extern "C" __global__ void ocannl_report_arch(unsigned int *out) {
#if defined(__CUDA_ARCH__)
  out[0] = (unsigned int)__CUDA_ARCH__;
#else
  out[0] = 0u;
#endif
}

extern "C" __global__ void ocannl_vendor_narrow_f32(unsigned long long base,
                                                    unsigned long long count,
                                                    unsigned char *out) {
  unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
  for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
       i < count; i += stride) {
    unsigned int u = (unsigned int)(base + i);
    float x = __uint_as_float(u);
    __nv_fp8_e5m2 v = (__nv_fp8_e5m2)x;
    out[i] = v.__x;
  }
}

extern "C" __global__ void ocannl_vendor_narrow_f64(unsigned long long base,
                                                    unsigned long long count,
                                                    unsigned int l0, unsigned int l1,
                                                    unsigned int l2, unsigned int l3,
                                                    unsigned char *out) {
  unsigned int lows[4] = {l0, l1, l2, l3};
  unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
  for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
       i < count; i += stride) {
    unsigned long long hi = base + i;
    for (int k = 0; k < 4; ++k) {
      unsigned long long u = (hi << 32) | (unsigned long long)lows[k];
      double d = __longlong_as_double((long long)u);
      __nv_fp8_e5m2 v = (__nv_fp8_e5m2)d;
      out[(4 * i) + k] = v.__x;
    }
  }
}
|}

(* Everything the sweep needs, held for the whole of it. The two lifetime fields are not decoration:
   [Cu.Context.get_primary] finalizes with [cuDevicePrimaryCtxRelease], and the underlying context
   is RESET once the last reference goes -- so a context dropped after [init] returns can be
   collected mid-sweep and take the device allocation and the loaded code with it. [Cuda_backend]
   keeps its [primary_context] in the device record for exactly this reason. [kernel_module] is belt
   and braces: cudajit's [get_function] already documents that the returned [func] retains its
   module, so this one is about saying the lifetime rather than inferring it from a binding's
   internals -- and about the HIP arm, where the same reasoning has to hold. *)
type state = {
  context : Cu.Context.t;
  kernel_module : Cu.Module.t;
  attrs : Cu.Device.attributes;
  stream : Cu.Stream.t;
  narrow_f32 : Cu.Module.func;
  narrow_f64 : Cu.Module.func;
  cuda_arch : int; (* __CUDA_ARCH__ as the compiled kernel itself reports it *)
  options : string list; (* what nvrtc was given, so a run record says what it compiled *)
  mutable buffer : (Cu.Deviceptr.t * int) option; (* pointer and its size in bytes *)
}

(* Which [--gpu-architecture] the kernel is built for. WHAT the two settings mean, and why the
   default is [`Device], is documented on [Fp8_soak.ARM.set_arch_policy]; here they are two option
   lists. Must be set before the first sweep: the module is compiled once, on first use. *)
let arch_policy : [ `Device | `Backend ] ref = ref `Device
let set_arch_policy p = arch_policy := p
let state = ref None

let device_count () : (int, string) Result.t =
  match
    Cu.init ();
    Cu.Device.get_count ()
  with
  | n -> Ok n
  | exception e -> Error (Exn.to_string e)

let init () =
  match !state with
  | Some st -> st
  | None ->
      Cu.init ();
      let device = Cu.Device.get ~ordinal:0 in
      let ctx = Cu.Context.get_primary device in
      Cu.Context.set_current ctx;
      let attrs = Cu.Device.get_attributes device in
      (* Both option groups come from the backend's own policies ([Cuda_backend]'s include
         discovery, [Ir.Cuda_mma]'s arch targets), not from a local guess. The include discovery
         especially: this kernel `#include`s <cuda_fp8.h>, and a soak that looked only in
         /usr/local/cuda would probe "ready" and then fail to compile on every Windows installation
         and every Linux one outside that prefix (Codex P2 on PR #463). The arch policy is shared
         for the same reason -- one nvrtc caller disagreeing with the backend about which
         [--gpu-architecture] a source needs is how a soak comes to measure something the backend
         never emits. *)
      let device_cc = (attrs.compute_capability_major * 10) + attrs.compute_capability_minor in
      let options =
        Cuda_backend.cuda_include_options ()
        @
        match !arch_policy with
        | `Device -> [ Printf.sprintf "--gpu-architecture=compute_%d" device_cc ]
        | `Backend -> Ir.Cuda_mma.gpu_arch_options ~device_cc source
      in
      let ptx =
        Nvrtc.compile_to_ptx ~cu_src:source ~name:"ocannl_fp8_soak.cu" ~options ~with_debug:false
      in
      let kernel_module = Cu.Module.load_data_ex ptx [] in
      (* Ask the compiled module which arch it got, before anything is swept. *)
      let arch_out = Cu.Deviceptr.mem_alloc ~size_in_bytes:4 in
      let stream = Cu.Stream.create () in
      Cu.Stream.launch_kernel
        (Cu.Module.get_function kernel_module ~name:"ocannl_report_arch")
        ~grid_dim_x:1 ~block_dim_x:1 ~shared_mem_bytes:0 stream [ Cu.Stream.Tensor arch_out ];
      let arch_host =
        Stdlib.Bigarray.Array1.create Stdlib.Bigarray.int32 Stdlib.Bigarray.c_layout 1
      in
      Cu.Stream.memcpy_D_to_H ~length:1
        ~dst:(Stdlib.Bigarray.genarray_of_array1 arch_host)
        ~src:arch_out stream;
      Cu.Stream.synchronize stream;
      Cu.Deviceptr.mem_free arch_out;
      let cuda_arch = Int32.to_int_exn arch_host.{0} in
      let st =
        {
          context = ctx;
          kernel_module;
          attrs;
          cuda_arch;
          options;
          stream;
          narrow_f32 = Cu.Module.get_function kernel_module ~name:"ocannl_vendor_narrow_f32";
          narrow_f64 = Cu.Module.get_function kernel_module ~name:"ocannl_vendor_narrow_f64";
          buffer = None;
        }
      in
      state := Some st;
      st

(* ["device"] is the one entry fp8_soak.ml looks up by name; the rest is printed in the run header
   in order, under labels saying what they are facts about. The device's capability is NOT the
   architecture the kernel compiled for -- under [--arch=backend] this marker-free source gets
   nvrtc's default target instead -- so that one is derived in fp8_soak.ml from the compiled
   kernel's own [__CUDA_ARCH__]. *)
let device_report () =
  let st = init () in
  [
    ("device", st.attrs.name);
    ( "device capability",
      Printf.sprintf "%d.%d" st.attrs.compute_capability_major st.attrs.compute_capability_minor );
    ("multiprocessors", Int.to_string st.attrs.multiprocessor_count);
  ]

let compile_options () = (init ()).options

(* Keyed by the macro's own C spelling, which is how fp8_soak.ml looks it up. *)
let kernel_macros () = [ ("__CUDA_ARCH__", (init ()).cuda_arch) ]

(* One allocation, grown on demand and reused across chunks: 64 chunk-sized [cuMemAlloc]s in a row
   is a way to run into fragmentation on a GPU another process is also using. *)
let device_buffer st bytes =
  match st.buffer with
  | Some (ptr, size) when size >= bytes -> ptr
  | prev ->
      Option.iter prev ~f:(fun (ptr, _) -> Cu.Deviceptr.mem_free ptr);
      let ptr = Cu.Deviceptr.mem_alloc ~size_in_bytes:bytes in
      st.buffer <- Some (ptr, bytes);
      ptr

let block_dim = 256

(* Enough blocks to fill the device several times over, but not one per element: the kernels are
   grid-stride loops, so the launch shape is a scheduling choice rather than a correctness one. *)
let grid_dim st = Int.max 1 (st.attrs.multiprocessor_count * 32)

let fetch st ptr (out : bytes_buf) count =
  Cu.Stream.memcpy_D_to_H ~length:count
    ~dst:(Stdlib.Bigarray.genarray_of_array1 out)
    ~src:ptr st.stream;
  Cu.Stream.synchronize st.stream

(* fp8_soak.ml never asks an arm for a spelling outside the vendor record's own list, CUDA's being
   [`Raw] alone; this is the vendor boundary's own check on that, so a mistake there cannot sweep
   the bare cast while a claim says the guarded helpers were swept. *)
let reject_guarded (spelling : [ `Raw | `Guarded ]) =
  match spelling with
  | `Raw -> ()
  | `Guarded -> invalid_arg "fp8_soak: the cuda arm has no guarded narrowing spelling"

let narrow_f32 ~(spelling : [ `Raw | `Guarded ]) ~base ~count (out : bytes_buf) =
  reject_guarded spelling;
  let st = init () in
  let ptr = device_buffer st count in
  Cu.Stream.launch_kernel st.narrow_f32 ~grid_dim_x:(grid_dim st) ~block_dim_x:block_dim
    ~shared_mem_bytes:0 st.stream
    [
      Cu.Stream.Size_t (Unsigned.Size_t.of_int base);
      Cu.Stream.Size_t (Unsigned.Size_t.of_int count);
      Cu.Stream.Tensor ptr;
    ];
  fetch st ptr out count

let narrow_f64 ~(spelling : [ `Raw | `Guarded ]) ~base ~count ~lows (out : bytes_buf) =
  reject_guarded spelling;
  let st = init () in
  let bytes = 4 * count in
  let ptr = device_buffer st bytes in
  Cu.Stream.launch_kernel st.narrow_f64 ~grid_dim_x:(grid_dim st) ~block_dim_x:block_dim
    ~shared_mem_bytes:0 st.stream
    [
      Cu.Stream.Size_t (Unsigned.Size_t.of_int base);
      Cu.Stream.Size_t (Unsigned.Size_t.of_int count);
      Cu.Stream.Int lows.(0);
      Cu.Stream.Int lows.(1);
      Cu.Stream.Int lows.(2);
      Cu.Stream.Int lows.(3);
      Cu.Stream.Tensor ptr;
    ];
  fetch st ptr out bytes
