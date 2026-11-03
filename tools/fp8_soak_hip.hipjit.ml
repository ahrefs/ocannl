(* The HIP arm of the fp8 soak (gh-ocannl-657, brought to real hardware by gh-ocannl-757): the same
   sweep against [__hip_fp8_e5m2], plus the one thing the CUDA arm has no counterpart for — a second
   pair of kernels calling the GUARDED helpers the HIP backend actually emits.

   Like the CUDA arm, this file holds the kernel source and the calls into hipjit and nothing else:
   which spellings this platform has, what they are called in a claim, and what the reported macros
   MEAN all live in fp8_soak.ml ([ARM] and [hip_vendor]), because THIS FILE IS LINKED AND RUN ONLY
   ON A ROCm BOX (gh-ocannl-758): every other box type-checks it against arrayjit/vendor_stubs
   (gh-ocannl-1688), but only a ROCm box learns whether an edit still drives the vendor correctly.

   The two spellings are not the same conversion on ROCm:

   - [`Raw] is [(__hip_fp8_e5m2)x], the platform's own cast. It is BROKEN for tiny magnitudes
   (gh-ocannl-647): an out-of-range shift in [hip/amd_detail/amd_hip_fp8.h]'s [cast_to_f8] returns
   values as large as 2^-14 where every other implementation returns a signed zero. Sweeping it is
   what localizes the defect to an exponent window, and a count of 0 there is the trigger to remove
   the guard (https://github.com/ROCm/rocm-systems/issues/10591). - [`Guarded] is
   [ocannl_single_to_fp8_uniform] / [ocannl_double_to_fp8_uniform] from {!Builtins_hip} — the source
   text, fetched from the backend rather than transcribed, exactly as the host side of this soak is
   the shipped [builtins.c] object code reached by [extern]. This is what an OCANNL HIP kernel
   narrows with, unconditionally since gh-ocannl-647, so it is what "does our codec still agree with
   what our kernels produce" really asks here.

   Compiled WITHOUT [-ffast-math], which the backend does pass: fast math is about arithmetic and
   this program does none — the conversion is the whole kernel. The guard's own [fabsf]/[copysignf]
   comparison is fast-math-invariant on finite inputs, which is the class the claims are about. *)

open Base
module H = Hip

(* Update this whenever you run this file on a box that has hipjit -- the run header prints it, and
   it is what tells the next editor how long the arm has gone without a real run (gh-ocannl-758);
   type-checking happens everywhere through arrayjit/vendor_stubs. *)
let last_compiled =
  "on minix-amd-wsl (Radeon 8060S, gfx1151, ROCm 7.14), 2026-08-27, staging PR #490"

let built = true

type bytes_buf =
  (int, Stdlib.Bigarray.int8_unsigned_elt, Stdlib.Bigarray.c_layout) Stdlib.Bigarray.Array1.t

(* hiprtc ships HIP's own headers, and targets the current default device when given no
   [--offload-arch] -- the same reliance hip_backend.ml's [compile] has. <hip/hip_fp8.h> is included
   unconditionally, where [Hip_backend.hip_includes] guards it with [__has_include] because a
   non-fp8 kernel must still compile on a pre-6.2 SDK: here the fp8 type IS the program, so a
   missing header should be a compile error naming the header rather than one naming the type. *)
let source () =
  {|
#include <hip/hip_fp8.h>

|} ^ Hip_backend.fp8_guard_source ()
  ^ {|

/* Which side of amd_hip_fp8.h's compile-time split the conversions above were built on, asked of
   the COMPILED KERNEL rather than inferred from the device name we happen to know. What the values
   mean is fp8_soak.ml's business (its [hip_vendor] record); this kernel only reports them, and 2 is
   "the header did not define it". HIP_FP8_TYPE_OCP rides along because it selects which fp8
   INTERPRETATION the type has; e5m2 is the OCP one. */
extern "C" __global__ void ocannl_report_fp8_path(unsigned int *out) {
#if defined(HIP_FP8_CVT_FAST_PATH)
  out[0] = (unsigned int)HIP_FP8_CVT_FAST_PATH;
  out[1] = (unsigned int)HIP_FP8_TYPE_OCP;
#else
  out[0] = 2u;
  out[1] = 2u;
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
    __hip_fp8_e5m2 v = (__hip_fp8_e5m2)x;
    out[i] = (unsigned char)v.__x;
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
      __hip_fp8_e5m2 v = (__hip_fp8_e5m2)d;
      out[(4 * i) + k] = (unsigned char)v.__x;
    }
  }
}

/* The same two sweeps through the guarded helpers above -- i.e. through what Hip_backend's
   [fp8_from_prec_fn] funnel emits for every narrowing site, conversions and operator bridges
   alike. Separate kernels rather than a runtime flag so that neither spelling can perturb the
   other's code generation. */
extern "C" __global__ void ocannl_guarded_narrow_f32(unsigned long long base,
                                                     unsigned long long count,
                                                     unsigned char *out) {
  unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
  for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
       i < count; i += stride) {
    unsigned int u = (unsigned int)(base + i);
    float x = __uint_as_float(u);
    __hip_fp8_e5m2 v = ocannl_single_to_fp8_uniform(x);
    out[i] = (unsigned char)v.__x;
  }
}

extern "C" __global__ void ocannl_guarded_narrow_f64(unsigned long long base,
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
      __hip_fp8_e5m2 v = ocannl_double_to_fp8_uniform(d);
      out[(4 * i) + k] = (unsigned char)v.__x;
    }
  }
}
|}

(* [context] and [kernel_module] are held for the same reason the CUDA arm holds them: a primary
   context released by its finalizer resets the device under the sweep. See that arm's comment. *)
type state = {
  context : H.Context.t;
  kernel_module : H.Module.t;
  attrs : H.Device.attributes;
  stream : H.Stream.t;
  narrow_f32 : H.Module.func;
  narrow_f64 : H.Module.func;
  guarded_f32 : H.Module.func;
  guarded_f64 : H.Module.func;
  fast_path : int; (* HIP_FP8_CVT_FAST_PATH as the compiled kernel itself reports it; 2 = absent *)
  ocp : int; (* HIP_FP8_TYPE_OCP likewise *)
  options : string list; (* what hiprtc was given, so a run record says what it compiled *)
  mutable buffer : (H.Deviceptr.t * int) option;
}

(* Present so the arms share one signature, and it genuinely has nothing to select between here.
   ROCm's split IS compile-time -- [HIP_FP8_CVT_FAST_PATH] -- but it is keyed off the TARGET
   ARCHITECTURE MACRO (__gfx942__ and four others), not off a numeric threshold an option could be
   dialled below the device's own capability the way [--gpu-architecture=compute_XX] can be on CUDA.
   hiprtc compiles for the current default device, so [`Device] and [`Backend] are the same
   compilation, and which side of the split it landed on is REPORTED by [kernel_macros] rather than
   chosen here. *)
let set_arch_policy (_ : [ `Device | `Backend ]) = ()
let state = ref None

let device_count () : (int, string) Result.t =
  match
    H.init ();
    H.Device.get_count ()
  with
  | n -> Ok n
  | exception e -> Error (Exn.to_string e)

let init () =
  match !state with
  | Some st -> st
  | None ->
      H.init ();
      let device = H.Device.get ~ordinal:0 in
      let ctx = H.Context.get_primary device in
      H.Context.set_current ctx;
      (* From [Hip_backend], not from a local guess: on Windows hiprtc does not find
         <hip/hip_fp16.h> without an include path, and a soak disagreeing with the backend about
         where the SDK is would probe "ready" and then fail to compile. *)
      let options = Hip_backend.hip_include_options () in
      let code =
        Hiprtc.compile_to_code ~hip_src:(source ()) ~name:"ocannl_fp8_soak.hip" ~options
          ~with_debug:false
      in
      let kernel_module = H.Module.load_data_ex code [] in
      let stream = H.Stream.create () in
      (* Ask the compiled module which side of the header's split it got, before anything is
         swept. *)
      let path_out = H.Deviceptr.mem_alloc ~size_in_bytes:8 in
      H.Stream.launch_kernel
        (H.Module.get_function kernel_module ~name:"ocannl_report_fp8_path")
        ~grid_dim_x:1 ~block_dim_x:1 ~shared_mem_bytes:0 stream [ H.Stream.Tensor path_out ];
      let path_host =
        Stdlib.Bigarray.Array1.create Stdlib.Bigarray.int32 Stdlib.Bigarray.c_layout 2
      in
      H.Stream.memcpy_D_to_H ~length:2
        ~dst:(Stdlib.Bigarray.genarray_of_array1 path_host)
        ~src:path_out stream;
      H.Stream.synchronize stream;
      H.Deviceptr.mem_free path_out;
      let st =
        {
          context = ctx;
          kernel_module;
          attrs = H.Device.get_attributes device;
          stream;
          narrow_f32 = H.Module.get_function kernel_module ~name:"ocannl_vendor_narrow_f32";
          narrow_f64 = H.Module.get_function kernel_module ~name:"ocannl_vendor_narrow_f64";
          guarded_f32 = H.Module.get_function kernel_module ~name:"ocannl_guarded_narrow_f32";
          guarded_f64 = H.Module.get_function kernel_module ~name:"ocannl_guarded_narrow_f64";
          fast_path = Int32.to_int_exn path_host.{0};
          ocp = Int32.to_int_exn path_host.{1};
          options;
          buffer = None;
        }
      in
      state := Some st;
      st

(* ["device"] is the one entry fp8_soak.ml looks up by name. ["gcn arch"] is the other one it reads:
   hiprtc given no [--offload-arch] compiles for the current default device, so this is also the
   architecture the kernels were built for, which is what selects the header's conversion path. *)
let device_report () =
  let st = init () in
  [ ("device", st.attrs.name); ("gcn arch", st.attrs.gcn_arch_name) ]

let compile_options () = (init ()).options

(* Keyed by the macros' own C spellings, which is how fp8_soak.ml looks them up. *)
let kernel_macros () =
  let st = init () in
  [ ("HIP_FP8_CVT_FAST_PATH", st.fast_path); ("HIP_FP8_TYPE_OCP", st.ocp) ]

let device_buffer st bytes =
  match st.buffer with
  | Some (ptr, size) when size >= bytes -> ptr
  | prev ->
      Option.iter prev ~f:(fun (ptr, _) -> H.Deviceptr.mem_free ptr);
      let ptr = H.Deviceptr.mem_alloc ~size_in_bytes:bytes in
      st.buffer <- Some (ptr, bytes);
      ptr

let block_dim = 256
let grid_dim = 4096

let fetch st ptr (out : bytes_buf) count =
  H.Stream.memcpy_D_to_H ~length:count
    ~dst:(Stdlib.Bigarray.genarray_of_array1 out)
    ~src:ptr st.stream;
  H.Stream.synchronize st.stream

let narrow_f32 ~(spelling : [ `Raw | `Guarded ]) ~base ~count (out : bytes_buf) =
  let st = init () in
  let ptr = device_buffer st count in
  H.Stream.launch_kernel
    (match spelling with `Raw -> st.narrow_f32 | `Guarded -> st.guarded_f32)
    ~grid_dim_x:grid_dim ~block_dim_x:block_dim ~shared_mem_bytes:0 st.stream
    [
      H.Stream.Size_t (Unsigned.Size_t.of_int base);
      H.Stream.Size_t (Unsigned.Size_t.of_int count);
      H.Stream.Tensor ptr;
    ];
  fetch st ptr out count

let narrow_f64 ~(spelling : [ `Raw | `Guarded ]) ~base ~count ~lows (out : bytes_buf) =
  let st = init () in
  let bytes = 4 * count in
  let ptr = device_buffer st bytes in
  H.Stream.launch_kernel
    (match spelling with `Raw -> st.narrow_f64 | `Guarded -> st.guarded_f64)
    ~grid_dim_x:grid_dim ~block_dim_x:block_dim ~shared_mem_bytes:0 st.stream
    [
      H.Stream.Size_t (Unsigned.Size_t.of_int base);
      H.Stream.Size_t (Unsigned.Size_t.of_int count);
      H.Stream.Int lows.(0);
      H.Stream.Int lows.(1);
      H.Stream.Int lows.(2);
      H.Stream.Int lows.(3);
      H.Stream.Tensor ptr;
    ];
  fetch st ptr out bytes
