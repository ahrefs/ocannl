(* CUDA builtin code split into (key, definition, dependencies) triples for filtering *)
let builtins =
  [
    ("uint4x32_t", {|typedef struct {
    unsigned int v[4];
} uint4x32_t;|}, []);
    (* The 16-byte alignment lets the [Vectorized] packed rendering (gh-ocannl-463) load/store these
       through [reinterpret_cast] as single 128-bit transactions (llm.c's Packed128); it is harmless
       for the value-typed [Set_from_vec] uses. *)
    ("float4_t", {|typedef struct __align__(16) { float v[4]; } float4_t;|}, []);
    ("double2_t", {|typedef struct __align__(16) { double v[2]; } double2_t;|}, []);
    ( "ocannl_shfl_xor",
      (* Butterfly warp shuffle for the [Workgroup_reduce] warp-shuffle rendering (gh-ocannl-462).
         The full mask is sound because the rendering requires the reduce axis to cover whole warps
         of the block's .x dimension, so every lane reaches the call. *)
      {|__device__ __forceinline__ float ocannl_shfl_xor(float v, int lane_mask) {
  return __shfl_xor_sync(0xffffffffu, v, lane_mask);
}
__device__ __forceinline__ double ocannl_shfl_xor(double v, int lane_mask) {
  return __shfl_xor_sync(0xffffffffu, v, lane_mask);
}|},
      [] );
    (* Asynchronous global->shared staging copies for software-pipelined tiles (gh-ocannl-487 phase
       2; [C_syntax_config.async_copy]). sm_80+; the [cp.async.] PTX instruction text in the
       definitions serves as [gpu_arch_options]' arch-floor marker — it exists in the source exactly
       when a definition was actually prepended. Inline PTX rather than <cuda_pipeline.h> so nvrtc
       needs no extra headers; [cvta.to.shared] inside the asm avoids relying on
       [__cvta_generic_to_shared]'s availability. 4- and 8-byte element copies only — the alignment
       plain shared declarations guarantee; a 16-byte [.cg] variant returns when a rendering
       guarantees 16-byte-aligned destinations (Codex P2 on PR #317). The "memory" clobbers pin the
       copies and the wait against compiler reordering of shared/global accesses. One entry PER
       emitted name — the entry key is both the usage filter and the identifier blacklist
       ([C_syntax.builtin_idents]), so a multi-name entry would leave the unlisted names open to
       shadowing by label-derived identifiers (Codex P2 on PR #317, round 2). *)
    ( "ocannl_cp_async4",
      {|__device__ __forceinline__ void ocannl_cp_async4(void *dst_shared, const void *src_global) {
  asm volatile(
      "{\n\t"
      ".reg .u64 ocannl_cpa_d;\n\t"
      "cvta.to.shared.u64 ocannl_cpa_d, %0;\n\t"
      "cp.async.ca.shared.global [ocannl_cpa_d], [%1], 4;\n\t"
      "}"
      :
      : "l"(dst_shared), "l"(src_global)
      : "memory");
}|},
      [] );
    ( "ocannl_cp_async8",
      {|__device__ __forceinline__ void ocannl_cp_async8(void *dst_shared, const void *src_global) {
  asm volatile(
      "{\n\t"
      ".reg .u64 ocannl_cpa_d;\n\t"
      "cvta.to.shared.u64 ocannl_cpa_d, %0;\n\t"
      "cp.async.ca.shared.global [ocannl_cpa_d], [%1], 8;\n\t"
      "}"
      :
      : "l"(dst_shared), "l"(src_global)
      : "memory");
}|},
      [] );
    (* Completes ALL prior copies of the calling thread, committed or not (PTX: wait_all =
       commit_group + wait_group 0), so the emission needs no commit bookkeeping at the depth-2
       single-step lookahead; deeper pipelines would switch to commit_group/wait_group N. *)
    ( "ocannl_cp_async_wait_all",
      {|__device__ __forceinline__ void ocannl_cp_async_wait_all() {
  asm volatile("cp.async.wait_all;" : : : "memory");
}|},
      [] );
    (* The coordinate table of the wmma converted [d] boundary (gh-ocannl-925,
       [Cuda_backend.wmma_d_boundary_lines]): entry [16 * row + col] holds that number, so a 16x16
       float accumulator fragment [load_matrix_sync]ed from it names, per lane and element, the
       matrix coordinate the fragment type keeps there. 32-byte alignment is [load_matrix_sync]'s
       pointer requirement. *)
    ( "ocannl_wmma_rc16",
      "__device__ __align__(32) float ocannl_wmma_rc16[256] = {"
      ^ String.concat ", " (List.init 256 (Printf.sprintf "%d.f"))
      ^ "};",
      [] );
    ("int32x4_t", {|typedef struct { int v[4]; } int32x4_t;|}, []);
    ("int64x2_t", {|typedef struct { long long v[2]; } int64x2_t;|}, []);
    ("int8x16_t", {|typedef struct { signed char v[16]; } int8x16_t;|}, []);
    ("uint16x8_t", {|typedef struct { unsigned short v[8]; } uint16x8_t;|}, []);
    ("bfloat16x8_t", {|typedef struct { __nv_bfloat16 v[8]; } bfloat16x8_t;|}, []);
    ("uint8x16_t", {|typedef struct { unsigned char v[16]; } uint8x16_t;|}, []);
    ("uint32x4_t", {|typedef struct { unsigned int v[4]; } uint32x4_t;|}, []);
    ("uint64x2_t", {|typedef struct { unsigned long long v[2]; } uint64x2_t;|}, []);
    (* Elements are the class type [__nv_fp8_e5m2] (not a plain integer): the [Set_from_vec]
       emission assigns vector elements to the fp8 array cells without a cast, and [__nv_fp8_e5m2]
       has no assignment from integer types. The 16-byte alignment supports the [Vectorized] packed
       [reinterpret_cast] loads/stores, like float4_t above. *)
    ("fp8x16_t", {|typedef struct __align__(16) { __nv_fp8_e5m2 v[16]; } fp8x16_t;|}, []);
    ("half8_t", {|typedef struct { __half v[8]; } half8_t;|}, []);
    ( "uint32_to_single_uniform",
      {|__device__ __forceinline__ float uint32_to_single_uniform(unsigned int x) {
  /* Use __uint2float_rn for correct rounding */
  return __uint2float_rn(x >> 8) * (1.0f / 16777216.0f);
}|},
      [] );
    ( "uint32_to_double_uniform",
      {|__device__ __forceinline__ double uint32_to_double_uniform(unsigned int x) {
  return __uint2double_rn(x) * (1.0 / 4294967296.0);
}|},
      [] );
    ( "uint4x32_to_single_uniform",
      {|__device__ float uint4x32_to_single_uniform(uint4x32_t x) {
  return uint32_to_single_uniform(x.v[0]);
}|},
      [ "uint4x32_t"; "uint32_to_single_uniform" ] );
    ( "uint4x32_to_double_uniform",
      (* Combine the lanes as an integer and convert NUMERICALLY before scaling (mirroring
         builtins.c); bit-casting the lanes to a double (via [__longlong_as_double]) would yield
         NaN/Inf/huge values instead of [0, 1). Only the top 53 bits are used so the conversion is
         exact and the result stays below 1.0 (all 64 bits could round up to 2^64, yielding exactly
         1.0). *)
      {|__device__ double uint4x32_to_double_uniform(uint4x32_t x) {
  unsigned long long combined = ((unsigned long long)x.v[1] << 32) | x.v[0];
  return (double)(combined >> 11) * (1.0 / 9007199254740992.0);
}|},
      [ "uint4x32_t" ] );
    ( "uint4x32_to_int32_uniform",
      {|__device__ int uint4x32_to_int32_uniform(uint4x32_t x) {
  return (int)x.v[0];
}|},
      [ "uint4x32_t" ] );
    ( "uint4x32_to_i64_uniform",
      {|__device__ long long uint4x32_to_i64_uniform(uint4x32_t x) {
  return __double_as_longlong(__hiloint2double(x.v[1], x.v[0]));
}|},
      [ "uint4x32_t" ] );
    ( "uint4x32_to_u32_uniform",
      {|__device__ unsigned int uint4x32_to_u32_uniform(uint4x32_t x) {
  return x.v[0];
}|},
      [ "uint4x32_t" ] );
    ( "uint4x32_to_u64_uniform",
      {|__device__ unsigned long long uint4x32_to_u64_uniform(uint4x32_t x) {
  return (unsigned long long)__double_as_longlong(__hiloint2double(x.v[1], x.v[0]));
}|},
      [ "uint4x32_t" ] );
    ( "uint4x32_to_uint32_uniform",
      {|__device__ unsigned int uint4x32_to_uint32_uniform(uint4x32_t x) {
  return uint4x32_to_u32_uniform(x);
}|},
      [ "uint4x32_t"; "uint4x32_to_u32_uniform" ] );
    ( "uint4x32_to_uint64_uniform",
      {|__device__ unsigned long long uint4x32_to_uint64_uniform(uint4x32_t x) {
  return uint4x32_to_u64_uniform(x);
}|},
      [ "uint4x32_t"; "uint4x32_to_u64_uniform" ] );
    ( "uint4x32_to_i8_uniform",
      {|__device__ signed char uint4x32_to_i8_uniform(uint4x32_t x) {
  return (signed char)(x.v[0] & 0xFF);
}|},
      [ "uint4x32_t" ] );
    ( "uint4x32_to_u8_uniform",
      {|__device__ unsigned char uint4x32_to_u8_uniform(uint4x32_t x) {
  return (unsigned char)(x.v[0] & 0xFF);
}|},
      [ "uint4x32_t" ] );
    ( "uint4x32_to_byte_uniform",
      {|__device__ unsigned char uint4x32_to_byte_uniform(uint4x32_t x) {
  return uint4x32_to_u8_uniform(x);
}|},
      [ "uint4x32_t"; "uint4x32_to_u8_uniform" ] );
    ( "uint4x32_to_uint16_uniform",
      {|__device__ unsigned short uint4x32_to_uint16_uniform(uint4x32_t x) {
  return (unsigned short)(x.v[0] & 0xFFFF);
}|},
      [ "uint4x32_t" ] );
    ( "uint4x32_to_int64_uniform",
      {|__device__ long long uint4x32_to_int64_uniform(uint4x32_t x) {
  return uint4x32_to_i64_uniform(x);
}|},
      [ "uint4x32_t"; "uint4x32_to_i64_uniform" ] );
    ( "uint4x32_to_bfloat16_uniform",
      {|__device__ __nv_bfloat16 uint4x32_to_bfloat16_uniform(uint4x32_t x) {
  float f = uint32_to_single_uniform(x.v[0]);
  return __float2bfloat16(f);
}|},
      [ "uint4x32_t"; "uint32_to_single_uniform" ] );
    ( "uint4x32_to_fp8_uniform",
      {|__device__ __nv_fp8_e5m2 uint4x32_to_fp8_uniform(uint4x32_t x) {
  /* Random FP8 E5M2 bit pattern from the low 8 bits (matches the CC backend, which stores fp8
     as a raw byte). Assign through the storage member: constructing __nv_fp8_e5m2 from the
     byte would be a numeric conversion, not a bit reinterpretation. */
  __nv_fp8_e5m2 result;
  result.__x = (__nv_fp8_storage_t)(x.v[0] & 0xFF);
  return result;
}|},
      [ "uint4x32_t" ] );
    ( "uint4x32_to_half_uniform",
      {|__device__ __half uint4x32_to_half_uniform(uint4x32_t x) {
  float f = uint32_to_single_uniform(x.v[0]);
  return __float2half(f);
}|},
      [ "uint4x32_t"; "uint32_to_single_uniform" ] );
    ( "uint4x32_to_single_uniform_vec",
      {|__device__ float4_t uint4x32_to_single_uniform_vec(uint4x32_t x) {
  float4_t result;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    result.v[i] = uint32_to_single_uniform(x.v[i]);
  }
  return result;
}|},
      [ "uint4x32_t"; "float4_t"; "uint32_to_single_uniform" ] );
    ( "uint4x32_to_double_uniform_vec",
      (* Numeric top-53-bits integer-to-double conversion, like the scalar variant (see the note
         there). *)
      {|__device__ double2_t uint4x32_to_double_uniform_vec(uint4x32_t x) {
  double2_t result;
  result.v[0] = (double)((((unsigned long long)x.v[1] << 32) | x.v[0]) >> 11) * (1.0 / 9007199254740992.0);
  result.v[1] = (double)((((unsigned long long)x.v[3] << 32) | x.v[2]) >> 11) * (1.0 / 9007199254740992.0);
  return result;
}|},
      [ "uint4x32_t"; "double2_t" ] );
    ( "uint4x32_to_int32_uniform_vec",
      {|__device__ int32x4_t uint4x32_to_int32_uniform_vec(uint4x32_t x) {
  int32x4_t result;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    result.v[i] = (int)x.v[i];
  }
  return result;
}|},
      [ "uint4x32_t"; "int32x4_t" ] );
    ( "uint4x32_to_i64_uniform_vec",
      {|__device__ int64x2_t uint4x32_to_i64_uniform_vec(uint4x32_t x) {
  int64x2_t result;
  result.v[0] = __double_as_longlong(__hiloint2double(x.v[1], x.v[0]));
  result.v[1] = __double_as_longlong(__hiloint2double(x.v[3], x.v[2]));
  return result;
}|},
      [ "uint4x32_t"; "int64x2_t" ] );
    ( "uint4x32_to_i8_uniform_vec",
      {|__device__ int8x16_t uint4x32_to_i8_uniform_vec(uint4x32_t x) {
  int8x16_t result;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    result.v[i*4 + 0] = (signed char)(x.v[i] & 0xFF);
    result.v[i*4 + 1] = (signed char)((x.v[i] >> 8) & 0xFF);
    result.v[i*4 + 2] = (signed char)((x.v[i] >> 16) & 0xFF);
    result.v[i*4 + 3] = (signed char)((x.v[i] >> 24) & 0xFF);
  }
  return result;
}|},
      [ "uint4x32_t"; "int8x16_t" ] );
    ( "uint4x32_to_u16_uniform_vec",
      {|__device__ uint16x8_t uint4x32_to_u16_uniform_vec(uint4x32_t x) {
  uint16x8_t result;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    result.v[i*2 + 0] = (unsigned short)(x.v[i] & 0xFFFF);
    result.v[i*2 + 1] = (unsigned short)((x.v[i] >> 16) & 0xFFFF);
  }
  return result;
}|},
      [ "uint4x32_t"; "uint16x8_t" ] );
    ( "uint4x32_to_bfloat16_uniform_vec",
      {|__device__ bfloat16x8_t uint4x32_to_bfloat16_uniform_vec(uint4x32_t x) {
  bfloat16x8_t result;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    // Convert each uint32 to two bfloat16 values
    float f1 = __uint2float_rn((x.v[i] & 0xFFFF) >> 0) * (1.0f / 65536.0f);
    float f2 = __uint2float_rn((x.v[i] >> 16) & 0xFFFF) * (1.0f / 65536.0f);
    result.v[i*2 + 0] = __float2bfloat16(f1);
    result.v[i*2 + 1] = __float2bfloat16(f2);
  }
  return result;
}|},
      [ "uint4x32_t"; "bfloat16x8_t" ] );
    ( "uint4x32_to_half_uniform_vec",
      {|__device__ half8_t uint4x32_to_half_uniform_vec(uint4x32_t x) {
  half8_t result;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    float f1 = __uint2float_rn((x.v[i] & 0xFFFF) >> 0) * (1.0f / 65536.0f);
    float f2 = __uint2float_rn((x.v[i] >> 16) & 0xFFFF) * (1.0f / 65536.0f);
    result.v[i*2 + 0] = __float2half(f1);
    result.v[i*2 + 1] = __float2half(f2);
  }
  return result;
}|},
      [ "uint4x32_t"; "half8_t" ] );
    (* The [Uint4x32_to_prec_uniform] emission names helpers by [Ops.prec_string]
       ("uint4x32_to_int64_uniform_vec", ...), while the workhorse definitions above use short
       names; provide the emitted names as wrappers (mirroring builtins.c, which defines the long
       names directly). Return types match [vec_typ_of_prec]; fp8's wrapper is defined further below
       with the fp8-typed pack. *)
    ( "uint4x32_to_int64_uniform_vec",
      {|__device__ int64x2_t uint4x32_to_int64_uniform_vec(uint4x32_t x) {
  return uint4x32_to_i64_uniform_vec(x);
}|},
      [ "uint4x32_t"; "int64x2_t"; "uint4x32_to_i64_uniform_vec" ] );
    ( "uint4x32_to_uint16_uniform_vec",
      {|__device__ uint16x8_t uint4x32_to_uint16_uniform_vec(uint4x32_t x) {
  return uint4x32_to_u16_uniform_vec(x);
}|},
      [ "uint4x32_t"; "uint16x8_t"; "uint4x32_to_u16_uniform_vec" ] );
    ( "uint4x32_to_byte_uniform_vec",
      {|__device__ int8x16_t uint4x32_to_byte_uniform_vec(uint4x32_t x) {
  return uint4x32_to_i8_uniform_vec(x);
}|},
      [ "uint4x32_t"; "int8x16_t"; "uint4x32_to_i8_uniform_vec" ] );
    ( "uint4x32_to_u8_uniform_vec",
      {|__device__ uint8x16_t uint4x32_to_u8_uniform_vec(uint4x32_t x) {
  uint8x16_t result;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    result.v[i*4 + 0] = (unsigned char)(x.v[i] & 0xFF);
    result.v[i*4 + 1] = (unsigned char)((x.v[i] >> 8) & 0xFF);
    result.v[i*4 + 2] = (unsigned char)((x.v[i] >> 16) & 0xFF);
    result.v[i*4 + 3] = (unsigned char)((x.v[i] >> 24) & 0xFF);
  }
  return result;
}|},
      [ "uint4x32_t"; "uint8x16_t" ] );
    ( "uint4x32_to_fp8_uniform_vec",
      {|__device__ fp8x16_t uint4x32_to_fp8_uniform_vec(uint4x32_t x) {
  /* 16 random FP8 E5M2 bit patterns; see uint4x32_to_fp8_uniform for the __x rationale. */
  fp8x16_t result;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    result.v[i*4 + 0].__x = (__nv_fp8_storage_t)(x.v[i] & 0xFF);
    result.v[i*4 + 1].__x = (__nv_fp8_storage_t)((x.v[i] >> 8) & 0xFF);
    result.v[i*4 + 2].__x = (__nv_fp8_storage_t)((x.v[i] >> 16) & 0xFF);
    result.v[i*4 + 3].__x = (__nv_fp8_storage_t)((x.v[i] >> 24) & 0xFF);
  }
  return result;
}|},
      [ "uint4x32_t"; "fp8x16_t" ] );
    ( "uint4x32_to_uint32_uniform_vec",
      {|__device__ uint32x4_t uint4x32_to_uint32_uniform_vec(uint4x32_t x) {
  uint32x4_t result;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    result.v[i] = x.v[i];
  }
  return result;
}|},
      [ "uint4x32_t"; "uint32x4_t" ] );
    ( "uint4x32_to_uint64_uniform_vec",
      {|__device__ uint64x2_t uint4x32_to_uint64_uniform_vec(uint4x32_t x) {
  uint64x2_t result;
  result.v[0] = ((unsigned long long)x.v[1] << 32) | x.v[0];
  result.v[1] = ((unsigned long long)x.v[3] << 32) | x.v[2];
  return result;
}|},
      [ "uint4x32_t"; "uint64x2_t" ] );
    (* Lane extraction from the packed uniform conversion (gh-509 task 4): minted by the virtualizer
       to inline packed-uniform results per cell. Implemented via the _vec builtins so the value
       stream is bitwise-identical to the vectorized stores by construction. *)
    ( "uint4x32_to_single_uniform_lane",
      {|__device__ float uint4x32_to_single_uniform_lane(uint4x32_t x, int lane) {
  return uint4x32_to_single_uniform_vec(x).v[lane];
}|},
      [ "uint4x32_t"; "uint4x32_to_single_uniform_vec" ] );
    ( "uint4x32_to_double_uniform_lane",
      {|__device__ double uint4x32_to_double_uniform_lane(uint4x32_t x, int lane) {
  return uint4x32_to_double_uniform_vec(x).v[lane];
}|},
      [ "uint4x32_t"; "uint4x32_to_double_uniform_vec" ] );
    ( "uint4x32_to_int32_uniform_lane",
      {|__device__ int uint4x32_to_int32_uniform_lane(uint4x32_t x, int lane) {
  return uint4x32_to_int32_uniform_vec(x).v[lane];
}|},
      [ "uint4x32_t"; "uint4x32_to_int32_uniform_vec" ] );
    ( "uint4x32_to_int64_uniform_lane",
      {|__device__ long long uint4x32_to_int64_uniform_lane(uint4x32_t x, int lane) {
  return uint4x32_to_int64_uniform_vec(x).v[lane];
}|},
      [ "uint4x32_t"; "uint4x32_to_int64_uniform_vec" ] );
    ( "uint4x32_to_byte_uniform_lane",
      {|__device__ signed char uint4x32_to_byte_uniform_lane(uint4x32_t x, int lane) {
  return uint4x32_to_byte_uniform_vec(x).v[lane];
}|},
      [ "uint4x32_t"; "uint4x32_to_byte_uniform_vec" ] );
    ( "uint4x32_to_uint16_uniform_lane",
      {|__device__ unsigned short uint4x32_to_uint16_uniform_lane(uint4x32_t x, int lane) {
  return uint4x32_to_uint16_uniform_vec(x).v[lane];
}|},
      [ "uint4x32_t"; "uint4x32_to_uint16_uniform_vec" ] );
    ( "uint4x32_to_bfloat16_uniform_lane",
      {|__device__ __nv_bfloat16 uint4x32_to_bfloat16_uniform_lane(uint4x32_t x, int lane) {
  return uint4x32_to_bfloat16_uniform_vec(x).v[lane];
}|},
      [ "uint4x32_t"; "uint4x32_to_bfloat16_uniform_vec" ] );
    ( "uint4x32_to_half_uniform_lane",
      {|__device__ __half uint4x32_to_half_uniform_lane(uint4x32_t x, int lane) {
  return uint4x32_to_half_uniform_vec(x).v[lane];
}|},
      [ "uint4x32_t"; "uint4x32_to_half_uniform_vec" ] );
    ( "uint4x32_to_fp8_uniform_lane",
      {|__device__ __nv_fp8_e5m2 uint4x32_to_fp8_uniform_lane(uint4x32_t x, int lane) {
  return uint4x32_to_fp8_uniform_vec(x).v[lane];
}|},
      [ "uint4x32_t"; "uint4x32_to_fp8_uniform_vec" ] );
    ( "uint4x32_to_uint32_uniform_lane",
      {|__device__ unsigned int uint4x32_to_uint32_uniform_lane(uint4x32_t x, int lane) {
  return uint4x32_to_uint32_uniform_vec(x).v[lane];
}|},
      [ "uint4x32_t"; "uint4x32_to_uint32_uniform_vec" ] );
    ( "uint4x32_to_uint64_uniform_lane",
      {|__device__ unsigned long long uint4x32_to_uint64_uniform_lane(uint4x32_t x, int lane) {
  return uint4x32_to_uint64_uniform_vec(x).v[lane];
}|},
      [ "uint4x32_t"; "uint4x32_to_uint64_uniform_vec" ] );
    ( "single_to_uint4x32",
      {|__device__ uint4x32_t single_to_uint4x32(float x) {
  unsigned int bits = __float_as_uint(x);
  uint4x32_t result = {{bits, 0, 0, 0}};
  return result;
}|},
      [ "uint4x32_t" ] );
    ( "double_to_uint4x32",
      {|__device__ uint4x32_t double_to_uint4x32(double x) {
  unsigned long long bits = __double_as_longlong(x);
  uint4x32_t result = {{(unsigned int)(bits & 0xFFFFFFFF), (unsigned int)(bits >> 32), 0, 0}};
  return result;
}|},
      [ "uint4x32_t" ] );
    ( "int32_to_uint4x32",
      {|__device__ uint4x32_t int32_to_uint4x32(int x) {
  /* Spread bits across all 4 components for better entropy with light threefry.
     Without this, consecutive counter values produce nearly identical v[0] outputs
     from 2-round threefry, causing periodicity in random number generation. */
  unsigned int u = (unsigned int)x;
  uint4x32_t result = {{
      u,
      u ^ 0x9E3779B9,              /* golden ratio constant */
      u ^ 0x6C078965,              /* Knuth's MMIX constant */
      u ^ ((u << 16) | (u >> 16))  /* bit rotation */
  }};
  return result;
}|},
      [ "uint4x32_t" ] );
    ( "int64_to_uint4x32",
      {|__device__ uint4x32_t int64_to_uint4x32(long long x) {
  unsigned long long bits = (unsigned long long)x;
  uint4x32_t result = {{(unsigned int)(bits & 0xFFFFFFFF), (unsigned int)(bits >> 32), 0, 0}};
  return result;
}|},
      [ "uint4x32_t" ] );
    ( "uint32_to_uint4x32",
      {|__device__ uint4x32_t uint32_to_uint4x32(unsigned int x) {
  /* Spread bits across all 4 components for better entropy with light threefry.
     Without this, consecutive counter values produce nearly identical v[0] outputs
     from 2-round threefry, causing periodicity in random number generation. */
  uint4x32_t result = {{
      x,
      x ^ 0x9E3779B9,              /* golden ratio constant */
      x ^ 0x6C078965,              /* Knuth's MMIX constant */
      x ^ ((x << 16) | (x >> 16))  /* bit rotation */
  }};
  return result;
}|},
      [ "uint4x32_t" ] );
    ( "uint64_to_uint4x32",
      {|__device__ uint4x32_t uint64_to_uint4x32(unsigned long long x) {
  uint4x32_t result = {{(unsigned int)(x & 0xFFFFFFFF), (unsigned int)(x >> 32), 0, 0}};
  return result;
}|},
      [ "uint4x32_t" ] );
    ( "byte_to_uint4x32",
      {|__device__ uint4x32_t byte_to_uint4x32(unsigned char x) {
  uint4x32_t result = {{(unsigned int)x, 0, 0, 0}};
  return result;
}|},
      [ "uint4x32_t" ] );
    ( "uint16_to_uint4x32",
      {|__device__ uint4x32_t uint16_to_uint4x32(unsigned short x) {
  uint4x32_t result = {{(unsigned int)x, 0, 0, 0}};
  return result;
}|},
      [ "uint4x32_t" ] );
    ( "bfloat16_to_uint4x32",
      {|__device__ uint4x32_t bfloat16_to_uint4x32(__nv_bfloat16 x) {
  uint4x32_t result = {{(unsigned int)__bfloat16_as_ushort(x), 0, 0, 0}};
  return result;
}|},
      [ "uint4x32_t" ] );
    ( "half_to_uint4x32",
      {|__device__ uint4x32_t half_to_uint4x32(__half x) {
  unsigned short bits = __half_as_ushort(x);
  uint4x32_t result = {{(unsigned int)bits, 0, 0, 0}};
  return result;
}|},
      [ "uint4x32_t" ] );
    ( "fp8_to_uint4x32",
      {|__device__ uint4x32_t fp8_to_uint4x32(__nv_fp8_e5m2 x) {
  /* Spread the raw bit pattern, matching the CC backend's byte-typed fp8. */
  uint4x32_t result = {{(unsigned int)x.__x, 0, 0, 0}};
  return result;
}|},
      [ "uint4x32_t" ] );
    ("THREEFRY_C240", {|__device__ __constant__ unsigned int THREEFRY_C240 = 0x1BD11BDA;|}, []);
    ( "THREEFRY_ROTATION",
      {|__device__ __constant__ unsigned int THREEFRY_ROTATION[8][4] = {
    {13, 15, 26, 6},
    {17, 29, 16, 24},
    {13, 15, 26, 6},
    {17, 29, 16, 24},
    {13, 15, 26, 6},
    {17, 29, 16, 24},
    {13, 15, 26, 6},
    {17, 29, 16, 24}
};|},
      [] );
    ( "rotl32",
      {|__device__ __forceinline__ unsigned int rotl32(unsigned int x, unsigned int n) {
    return __funnelshift_l(x, x, n);
}|},
      [] );
    ( "threefry_round",
      {|__device__ __forceinline__ void threefry_round(uint4 &x, unsigned int r0, unsigned int r1, unsigned int r2, unsigned int r3) {
    x.x += x.y; x.y = rotl32(x.y, r0); x.y ^= x.x;
    x.z += x.w; x.w = rotl32(x.w, r1); x.w ^= x.z;
    
    unsigned int tmp = x.y;
    x.y = x.w;
    x.w = tmp;
    
    x.x += x.y; x.y = rotl32(x.y, r2); x.y ^= x.x;
    x.z += x.w; x.w = rotl32(x.w, r3); x.w ^= x.z;
    
    tmp = x.y;
    x.y = x.w;
    x.w = tmp;
}|},
      [ "rotl32" ] );
    ( "arrayjit_threefry4x32_crypto",
      {|__device__ uint4x32_t arrayjit_threefry4x32_crypto(uint4x32_t key, uint4x32_t counter) {
    uint4 x = make_uint4(counter.v[0], counter.v[1], counter.v[2], counter.v[3]);
    uint4 k = make_uint4(key.v[0], key.v[1], key.v[2], key.v[3]);
    
    /* Compute ks[4] */
    unsigned int ks4 = k.x ^ k.y ^ k.z ^ k.w ^ THREEFRY_C240;
    
    /* Initial key injection */
    x.x += k.x;
    x.y += k.y;
    x.z += k.z;
    x.w += k.w;
    
    /* Unrolled 20 rounds with key injections */
    #pragma unroll
    for (int round = 0; round < 20; round += 4) {
        threefry_round(x, THREEFRY_ROTATION[0][0], THREEFRY_ROTATION[0][1], 
                          THREEFRY_ROTATION[0][2], THREEFRY_ROTATION[0][3]);
        threefry_round(x, THREEFRY_ROTATION[1][0], THREEFRY_ROTATION[1][1], 
                          THREEFRY_ROTATION[1][2], THREEFRY_ROTATION[1][3]);
        threefry_round(x, THREEFRY_ROTATION[0][0], THREEFRY_ROTATION[0][1], 
                          THREEFRY_ROTATION[0][2], THREEFRY_ROTATION[0][3]);
        threefry_round(x, THREEFRY_ROTATION[1][0], THREEFRY_ROTATION[1][1], 
                          THREEFRY_ROTATION[1][2], THREEFRY_ROTATION[1][3]);
        
        /* Key injection */
        unsigned int inj_round = (round / 4) + 1;
        if (inj_round == 1) {
            x.x += k.y;
            x.y += k.z;
            x.z += k.w;
            x.w += ks4 + inj_round;
        } else if (inj_round == 2) {
            x.x += k.z;
            x.y += k.w;
            x.z += ks4;
            x.w += k.x + inj_round;
        } else if (inj_round == 3) {
            x.x += k.w;
            x.y += ks4;
            x.z += k.x;
            x.w += k.y + inj_round;
        } else if (inj_round == 4) {
            x.x += ks4;
            x.y += k.x;
            x.z += k.y;
            x.w += k.z + inj_round;
        }
    }
    
    /* Final key injection */
    x.x += k.x;
    x.y += k.y;
    x.z += k.z;
    x.w += k.w + 5;
    
    uint4x32_t result;
    result.v[0] = x.x;
    result.v[1] = x.y;
    result.v[2] = x.z;
    result.v[3] = x.w;
    return result;
}|},
      [ "uint4x32_t"; "THREEFRY_C240"; "threefry_round"; "THREEFRY_ROTATION" ] );
    ( "arrayjit_threefry4x32_light",
      {|__device__ uint4x32_t arrayjit_threefry4x32_light(uint4x32_t key, uint4x32_t counter) {
    uint4 x = make_uint4(counter.v[0], counter.v[1], counter.v[2], counter.v[3]);
    uint4 k = make_uint4(key.v[0], key.v[1], key.v[2], key.v[3]);
    
    /* Compute ks[4] */
    unsigned int ks4 = k.x ^ k.y ^ k.z ^ k.w ^ THREEFRY_C240;
    
    /* Initial key injection */
    x.x += k.x;
    x.y += k.y;
    x.z += k.z;
    x.w += k.w;
    
    /* Only 2 rounds for light version */
    threefry_round(x, THREEFRY_ROTATION[0][0], THREEFRY_ROTATION[0][1], 
                      THREEFRY_ROTATION[0][2], THREEFRY_ROTATION[0][3]);
    threefry_round(x, THREEFRY_ROTATION[1][0], THREEFRY_ROTATION[1][1], 
                      THREEFRY_ROTATION[1][2], THREEFRY_ROTATION[1][3]);
    
    /* Final key injection after round 2 */
    x.x += k.y;
    x.y += k.z;
    x.z += k.w;
    x.w += ks4 + 1;
    
    uint4x32_t result;
    result.v[0] = x.x;
    result.v[1] = x.y;
    result.v[2] = x.z;
    result.v[3] = x.w;
    return result;
}|},
      [ "uint4x32_t"; "THREEFRY_C240"; "threefry_round"; "THREEFRY_ROTATION" ] );
    ( "arrayjit_threefry4x32",
      {|__device__ uint4x32_t arrayjit_threefry4x32(uint4x32_t key, uint4x32_t counter) {
    /* Default to light version */
    return arrayjit_threefry4x32_light(key, counter);
}|},
      [ "uint4x32_t"; "arrayjit_threefry4x32_light" ] );
  ]

let builtins =
  Builtins_cc.integer_power_builtins ~prefix:"__device__ __forceinline__" ~supports_double:true
  @ builtins
