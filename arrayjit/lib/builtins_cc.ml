(* Canonical C builtin definitions for kernels and host stubs (gh-ocannl-656). gen_builtins emits
   builtins_shared.h from this table, excluding the cc-only OCANNL_ SIMD namespace. Keep platform
   includes here; host headers are owned by the generator. *)

let includes =
  {|
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* SIMD platform detection (gh-ocannl-164): compile-time macros for conditional SIMD code paths
   (explicit intrinsics are emitted by follow-up work; auto-vectorization needs no guards). */
#ifdef __AVX2__
  #define OCANNL_HAS_AVX2 1
  #include <immintrin.h>
#else
  #define OCANNL_HAS_AVX2 0
#endif
#ifdef __ARM_NEON
  #define OCANNL_HAS_NEON 1
  /* Do NOT include <arm_neon.h> here: it defines int8x16_t, int64x2_t, uint16x8_t, etc. as
     native vector types, colliding with the pack-struct typedefs emitted by the builtins below
     (e.g. the uint4x32_to_*_uniform_vec result types). Nothing emits NEON intrinsics yet; when
     that changes, the emitter must rename the pack types or include the header per-kernel. */
#else
  #define OCANNL_HAS_NEON 0
#endif

/* Fused elementwise FMA for explicit SIMD rendering of Vectorized loops: clang's builtin where
   available, else the codegen's per-lane fmaf/fma loop. __has_builtin needs a shim on compilers
   that lack it (function-like use of an undefined macro is a preprocessor error). */
#ifndef __has_builtin
  #define __has_builtin(x) 0
#endif
#if __has_builtin(__builtin_elementwise_fma)
  #define OCANNL_HAS_ELEMENTWISE_FMA 1
#else
  #define OCANNL_HAS_ELEMENTWISE_FMA 0
#endif

/* Parallel Grid loops (gh-ocannl-164) use libdispatch on Apple platforms; OpenMP needs no
   header. Guarded unconditionally so generated sources are byte-identical across platforms. */
#ifdef __APPLE__
  #include <dispatch/dispatch.h>
#endif

/* No longer need export macros since we're using textual prepending */
|}

(* Each entry is (key, definition, dependencies) *)
(* One multiplication algorithm across dialects. The exponent is encoded exactly as two
   unsigned 32-bit words times 2^shift; even the largest integral OCaml float fits without a
   vendor integer-width assumption or narrowing the exponent to the base's precision. Bitwise
   exceptional-value handling and final sign restoration survive fast math: HIP otherwise loses
   (-0)^3 at half precision and turns double 1/0 into NaN (gh-ocannl-1185). *)
let integer_power_builtins ~prefix ~supports_double ~dialect =
  let make typ name int_typ sign_mask infinity_mask quiet_mask to_bits from_bits =
    let read_bits var value =
      match dialect with
      | `C -> Printf.sprintf "%s %s; memcpy(&%s, &%s, sizeof(%s));" int_typ var var value var
      | `Cuda_like | `Metal -> Printf.sprintf "%s %s = %s(%s);" int_typ var to_bits value
    in
    let return_bits bits =
      match dialect with
      | `C -> Printf.sprintf "%s value; memcpy(&value, &%s, sizeof(value)); return value;" typ bits
      | `Cuda_like | `Metal -> Printf.sprintf "return %s(%s);" from_bits bits
    in
    ( name,
      Printf.sprintf
        {|%s %s %s(%s x, unsigned int lo, unsigned int hi, int shift, int negative) {
  if (lo == 0u && hi == 0u) return (%s)1;
  %s
  %s magnitude_bits = bits & ~%s;
  %s sign = (shift == 0 && (lo & 1u) != 0u) ? (bits & %s) : 0;
  if (magnitude_bits > %s) {
    %s result_bits = bits | %s;
    %s
  }
  if (magnitude_bits == 0 || magnitude_bits == %s) {
    %s result_bits = ((magnitude_bits == 0) == (negative != 0) ? %s : 0) | sign;
    %s
  }
  %s result = (%s)1;
  for (int i = 0; i < shift; ++i) x = x * x;
  while (lo != 0u || hi != 0u) {
    if ((lo & 1u) != 0u) result = result * x;
    lo = (lo >> 1) | (hi << 31);
    hi >>= 1;
    if (lo != 0u || hi != 0u) x = x * x;
  }
  %s
  magnitude_bits = result_bits & ~%s;
  if (negative) {
    if (magnitude_bits == 0 || magnitude_bits == %s) {
      result_bits = (magnitude_bits == 0 ? %s : 0) | sign;
      %s
    }
    result = (%s)1 / result;
    %s
    magnitude_bits = inverse_bits & ~%s;
  }
  result_bits = magnitude_bits | sign;
  %s
}|}
        prefix typ name typ typ (read_bits "bits" "x") int_typ sign_mask int_typ sign_mask
        infinity_mask int_typ quiet_mask (return_bits "result_bits") infinity_mask int_typ
        infinity_mask (return_bits "result_bits") typ typ
        (read_bits "result_bits" "result")
        sign_mask infinity_mask infinity_mask (return_bits "result_bits") typ
        (read_bits "inverse_bits" "result")
        sign_mask (return_bits "result_bits"),
      [] )
  in
  let f32_to_bits, f32_from_bits, f64_to_bits, f64_from_bits =
    match dialect with
    | `C -> ("", "", "", "")
    | `Cuda_like ->
        ("__float_as_uint", "__uint_as_float", "__double_as_longlong", "__longlong_as_double")
    | `Metal -> ("as_type<uint>", "as_type<float>", "", "")
  in
  let single =
    make "float" "ocannl_powi_f32" "unsigned int" "0x80000000u" "0x7f800000u" "0x00400000u"
      f32_to_bits f32_from_bits
  in
  if supports_double then
    [
      single;
      make "double" "ocannl_powi_f64" "unsigned long long" "0x8000000000000000ull"
        "0x7ff0000000000000ull" "0x0008000000000000ull" f64_to_bits f64_from_bits;
    ]
  else [ single ]

let builtins =
  [
    (* Float16 feature detection and type definitions *)
    ( "HAS_NATIVE_FLOAT16",
      {|
#ifdef __FLT16_MAX__
  #define HAS_NATIVE_FLOAT16 1
#else
  #define HAS_NATIVE_FLOAT16 0
#endif
|},
      [] );
    ( "HALF_T",
      {|
#if HAS_NATIVE_FLOAT16
  #define HALF_T _Float16
#else
  #define HALF_T uint16_t
#endif
|},
      [ "HAS_NATIVE_FLOAT16" ] );
    ( "HALF_TO_FP",
      {|
#if HAS_NATIVE_FLOAT16
  #define HALF_TO_FP(x) (x)  /* Identity - already in floating point */
#else
  #define HALF_TO_FP(x) half_to_float_emulated(x)  /* Convert to float for computation */
#endif
|},
      [ "HAS_NATIVE_FLOAT16"; "half_to_float_emulated" ] );
    ( "FP_TO_HALF",
      {|
#if HAS_NATIVE_FLOAT16
  #define FP_TO_HALF(x) (x)  /* Identity - already half precision */
#else
  #define FP_TO_HALF(x) float_to_half_emulated(x)  /* Convert back from float */
#endif
|},
      [ "HAS_NATIVE_FLOAT16"; "float_to_half_emulated" ] );
    ( "HALF_TO_FLOAT",
      {|
#if HAS_NATIVE_FLOAT16
  #define HALF_TO_FLOAT(x) ((float)(x))
#else
  #define HALF_TO_FLOAT(x) half_to_float_emulated(x)
#endif
|},
      [ "HAS_NATIVE_FLOAT16"; "half_to_float_emulated" ] );
    ( "FLOAT_TO_HALF",
      {|
#if HAS_NATIVE_FLOAT16
  #define FLOAT_TO_HALF(x) ((_Float16)(x))
#else
  #define FLOAT_TO_HALF(x) float_to_half_emulated(x)
#endif
|},
      [ "HAS_NATIVE_FLOAT16"; "float_to_half_emulated" ] );
    ( "HALF_TO_UINT16",
      {|
#if HAS_NATIVE_FLOAT16
  #define HALF_TO_UINT16(x) ({ _Float16 _h = (x); uint16_t _r; memcpy(&_r, &_h, 2); _r; })
#else
  #define HALF_TO_UINT16(x) (x)
#endif
|},
      [ "HAS_NATIVE_FLOAT16" ] );
    ( "UINT16_TO_HALF",
      {|
#if HAS_NATIVE_FLOAT16
  #define UINT16_TO_HALF(x) ({ uint16_t _u = (x); _Float16 _h; memcpy(&_h, &_u, 2); _h; })
#else
  #define UINT16_TO_HALF(x) (x)
#endif
|},
      [ "HAS_NATIVE_FLOAT16" ] );
    (* Float16 emulation functions *)
    ( "half_to_float_emulated",
      {|
#if !HAS_NATIVE_FLOAT16
/* Convert IEEE 754 half precision (stored as uint16_t) to float */
static float half_to_float_emulated(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    
    if (exponent == 0) {
        if (mantissa == 0) {
            /* Zero */
            return sign ? -0.0f : 0.0f;
        } else {
            /* Subnormal */
            float result = ldexpf(mantissa / 1024.0f, -14);
            return sign ? -result : result;
        }
    } else if (exponent == 31) {
        if (mantissa == 0) {
            /* Infinity */
            return sign ? -INFINITY : INFINITY;
        } else {
            /* NaN */
            return NAN;
        }
    } else {
        /* Normal number */
        float result = ldexpf(1.0f + mantissa / 1024.0f, exponent - 15);
        return sign ? -result : result;
    }
}
#endif
|},
      [ "HAS_NATIVE_FLOAT16" ] );
    ( "float_to_half_emulated",
      {|
#if !HAS_NATIVE_FLOAT16
/* Convert float to IEEE 754 half precision (stored as uint16_t) */
static uint16_t float_to_half_emulated(float f) {
    uint32_t f32;
    memcpy(&f32, &f, sizeof(float));
    
    uint32_t sign = (f32 >> 31) & 0x1;
    uint32_t exponent = (f32 >> 23) & 0xFF;
    uint32_t mantissa = f32 & 0x7FFFFF;
    
    /* Convert exponent from float bias (127) to half bias (15) */
    int32_t new_exp = (int32_t)exponent - 127 + 15;
    
    if (exponent == 0xFF) {
        /* Infinity or NaN */
        if (mantissa == 0) {
            /* Infinity */
            return (sign << 15) | (0x1F << 10);
        } else {
            /* NaN - preserve sign and set mantissa bit */
            return (sign << 15) | (0x1F << 10) | 0x200;
        }
    } else if (new_exp <= 0) {
        /* Underflow to zero or subnormal */
        if (new_exp < -10) {
            /* Too small - flush to zero */
            return sign << 15;
        }
        /* Subnormal - with round-to-nearest-even */
        uint32_t shift = -new_exp + 1;
        mantissa = (mantissa | 0x800000);
        
        /* For subnormal, we need to shift right by (shift + 13) total bits */
        uint32_t total_shift = shift + 13;
        
        /* new_exp is in [-10, 0] here, so total_shift is in [14, 24] and every shift below stays
           in range. At total_shift == 24 the quotient is zero but the guard bit is the mantissa's
           implicit leading one, so round-to-nearest-even still yields the smallest subnormal for
           anything strictly above the midpoint 2^-25: 0x1.8p-25 narrows to half bits 0x0001, and
           only the exact midpoint ties down to zero (gh-ocannl-981). Zero is unconditional only
           once the guard bit itself is shifted away, total_shift > 24 -- which the new_exp < -10
           arm above has already returned, so this is the boundary and not a flush. */
        if (total_shift > 24) {
            /* Would shift away all bits, guard bit included */
            return sign << 15;
        }
        
        /* Extract guard, round, and sticky bits before shifting */
        uint32_t guard_bit = (mantissa >> (total_shift - 1)) & 1;
        uint32_t round_bit = total_shift > 1 ? ((mantissa >> (total_shift - 2)) & 1) : 0;
        uint32_t sticky_mask = (1U << (total_shift - 2)) - 1;
        uint32_t sticky_bits = total_shift > 1 ? (mantissa & sticky_mask) : 0;
        
        mantissa = mantissa >> total_shift;
        
        /* Round to nearest even */
        if (guard_bit && (round_bit || sticky_bits || (mantissa & 1))) {
            mantissa++;
        }
        
        return (sign << 15) | mantissa;
    } else if (new_exp >= 0x1F) {
        /* Overflow to infinity */
        return (sign << 15) | (0x1F << 10);
    } else {
        /* Normal number - with round-to-nearest-even (banker's rounding) */
        uint32_t rounded_mantissa;
        uint32_t guard_bit = (mantissa >> 12) & 1;
        uint32_t round_bit = (mantissa >> 11) & 1;
        uint32_t sticky_bits = mantissa & 0x7FF;
        
        rounded_mantissa = mantissa >> 13;
        
        /* Round to nearest even: round up if we have:
         * - guard bit set and (round bit set OR sticky bits non-zero OR mantissa LSB set)
         */
        if (guard_bit && (round_bit || sticky_bits || (rounded_mantissa & 1))) {
            rounded_mantissa++;
        }
        
        if (rounded_mantissa > 0x3FF) {
            /* Rounding caused overflow in mantissa */
            new_exp++;
            rounded_mantissa = 0;
            if (new_exp >= 0x1F) {
                /* Overflow to infinity */
                return (sign << 15) | (0x1F << 10);
            }
        }
        return (sign << 15) | (new_exp << 10) | rounded_mantissa;
    }
}
#endif
|},
      [ "HAS_NATIVE_FLOAT16" ] );
    (* Threefry4x32 types and complete implementation *)
    ("uint4x32_t", {|
typedef struct {
    uint32_t v[4];
} uint4x32_t;
|}, []);
    ( "threefry_common",
      {|
/* Threefry4x32 constants */
const uint32_t THREEFRY_C240 = 0x1BD11BDA;

/* Rotation constants for Threefry4x32 */
const unsigned int THREEFRY_ROTATION_0_0 = 13;
const unsigned int THREEFRY_ROTATION_0_1 = 15;
const unsigned int THREEFRY_ROTATION_0_2 = 26;
const unsigned int THREEFRY_ROTATION_0_3 = 6;
const unsigned int THREEFRY_ROTATION_1_0 = 17;
const unsigned int THREEFRY_ROTATION_1_1 = 29;
const unsigned int THREEFRY_ROTATION_1_2 = 16;
const unsigned int THREEFRY_ROTATION_1_3 = 24;

/* Rotate left function */
uint32_t rotl32(uint32_t x, unsigned int n) {
    return (x << n) | (x >> (32 - n));
}

/* Threefry4x32 round function */
void threefry_round(uint32_t x[4], unsigned int r0, unsigned int r1, unsigned int r2, unsigned int r3) {
    x[0] += x[1]; x[1] = rotl32(x[1], r0); x[1] ^= x[0];
    x[2] += x[3]; x[3] = rotl32(x[3], r1); x[3] ^= x[2];
    
    uint32_t tmp = x[1];
    x[1] = x[3];
    x[3] = tmp;
    
    x[0] += x[1]; x[1] = rotl32(x[1], r2); x[1] ^= x[0];
    x[2] += x[3]; x[3] = rotl32(x[3], r3); x[3] ^= x[2];
    
    tmp = x[1];
    x[1] = x[3];
    x[3] = tmp;
}
|},
      [ "uint4x32_t" ] );
    ( "arrayjit_threefry4x32_crypto",
      {|

/* Threefry4x32 implementation - 20 rounds (cryptographic version) */
uint4x32_t arrayjit_threefry4x32_crypto(uint4x32_t key, uint4x32_t counter) {
    uint32_t x[4];
    uint32_t ks[5];
    
    /* Initialize key schedule */
    ks[0] = key.v[0];
    ks[1] = key.v[1];
    ks[2] = key.v[2];
    ks[3] = key.v[3];
    ks[4] = ks[0] ^ ks[1] ^ ks[2] ^ ks[3] ^ THREEFRY_C240;
    
    /* Initialize state with counter */
    x[0] = counter.v[0];
    x[1] = counter.v[1];
    x[2] = counter.v[2];
    x[3] = counter.v[3];
    
    /* Initial key injection */
    x[0] += ks[0];
    x[1] += ks[1];
    x[2] += ks[2];
    x[3] += ks[3];
    
    /* 20 rounds */
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    
    /* Key injection after round 4 */
    x[0] += ks[1];
    x[1] += ks[2];
    x[2] += ks[3];
    x[3] += ks[4] + 1;
    
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    
    /* Key injection after round 8 */
    x[0] += ks[2];
    x[1] += ks[3];
    x[2] += ks[4];
    x[3] += ks[0] + 2;
    
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    
    /* Key injection after round 12 */
    x[0] += ks[3];
    x[1] += ks[4];
    x[2] += ks[0];
    x[3] += ks[1] + 3;
    
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    
    /* Key injection after round 16 */
    x[0] += ks[4];
    x[1] += ks[0];
    x[2] += ks[1];
    x[3] += ks[2] + 4;
    
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    
    /* Final key injection after round 20 */
    x[0] += ks[0];
    x[1] += ks[1];
    x[2] += ks[2];
    x[3] += ks[3] + 5;
    
    uint4x32_t result;
    result.v[0] = x[0];
    result.v[1] = x[1];
    result.v[2] = x[2];
    result.v[3] = x[3];
    return result;
}
|},
      [ "uint4x32_t"; "threefry_common" ] );
    ( "arrayjit_threefry4x32_light",
      {|
/* Threefry4x32 implementation - 2 rounds (light version, as in JAX/XLA) */
uint4x32_t arrayjit_threefry4x32_light(uint4x32_t key, uint4x32_t counter) {
    uint32_t x[4];
    uint32_t ks[5];
    
    /* Initialize key schedule */
    ks[0] = key.v[0];
    ks[1] = key.v[1];
    ks[2] = key.v[2];
    ks[3] = key.v[3];
    ks[4] = ks[0] ^ ks[1] ^ ks[2] ^ ks[3] ^ THREEFRY_C240;
    
    /* Initialize state with counter */
    x[0] = counter.v[0];
    x[1] = counter.v[1];
    x[2] = counter.v[2];
    x[3] = counter.v[3];
    
    /* Initial key injection */
    x[0] += ks[0];
    x[1] += ks[1];
    x[2] += ks[2];
    x[3] += ks[3];
    
    /* Only 2 rounds for light version */
    threefry_round(x, THREEFRY_ROTATION_0_0, THREEFRY_ROTATION_0_1, THREEFRY_ROTATION_0_2, THREEFRY_ROTATION_0_3);
    threefry_round(x, THREEFRY_ROTATION_1_0, THREEFRY_ROTATION_1_1, THREEFRY_ROTATION_1_2, THREEFRY_ROTATION_1_3);
    
    /* Final key injection after round 2 */
    x[0] += ks[1];
    x[1] += ks[2];
    x[2] += ks[3];
    x[3] += ks[4] + 1;
    
    uint4x32_t result;
    result.v[0] = x[0];
    result.v[1] = x[1];
    result.v[2] = x[2];
    result.v[3] = x[3];
    return result;
}
|},
      [ "uint4x32_t"; "threefry_common" ] );
    ( "arrayjit_threefry4x32",
      {|
/* Default threefry4x32 function - will be configured at runtime */
uint4x32_t arrayjit_threefry4x32(uint4x32_t key, uint4x32_t counter) {
    /* Default to light version */
    return arrayjit_threefry4x32_light(key, counter);
}
|},
      [ "uint4x32_t"; "arrayjit_threefry4x32_light" ] );
    (* Vector types with half precision *)
    ("half8_t", {|
typedef struct { HALF_T v[8]; } half8_t;
|}, [ "HALF_T" ]);
    ("float4_t", {|
typedef struct { float v[4]; } float4_t;
|}, []);
    ("double2_t", {|
typedef struct { double v[2]; } double2_t;
|}, []);
    ("int32x4_t", {|
typedef struct { int32_t v[4]; } int32x4_t;
|}, []);
    ("int64x2_t", {|
typedef struct { int64_t v[2]; } int64x2_t;
|}, []);
    ("int8x16_t", {|
typedef struct { int8_t v[16]; } int8x16_t;
|}, []);
    ("uint16x8_t", {|
typedef struct { uint16_t v[8]; } uint16x8_t;
|}, []);
    ("uint8x16_t", {|
typedef struct { uint8_t v[16]; } uint8x16_t;
|}, []);
    ("uint32x4_t", {|
typedef struct { uint32_t v[4]; } uint32x4_t;
|}, []);
    ("uint64x2_t", {|
typedef struct { uint64_t v[2]; } uint64x2_t;
|}, []);
    (* Basic conversion functions *)
    ( "uint32_to_single_uniform",
      {|
/* Convert to float in [0, 1) */
float uint32_to_single_uniform(uint32_t x) {
    /* Use upper 24 bits for float mantissa (23 bits + implicit 1) */
    return (x >> 8) * (1.0f / 16777216.0f);
}
|},
      [] );
    ( "uint32_to_double_uniform",
      {|
/* Convert to double in [0, 1) */
double uint32_to_double_uniform(uint32_t x) {
    return x * (1.0 / 4294967296.0);
}
|},
      [] );
    (* Conversion functions with dependencies *)
    ( "uint4x32_to_single_uniform",
      {|
/* Uint4x32 to float32 uniform - uses first 32 bits */
float uint4x32_to_single_uniform(uint4x32_t x) {
    return uint32_to_single_uniform(x.v[0]);
}
|},
      [ "uint4x32_t"; "uint32_to_single_uniform" ] );
    ( "uint4x32_to_half_uniform",
      {|
/* Uint4x32 to float16 uniform - the single draw of the first 32 bits, narrowed: the route every
   backend's scalar bf16 conversion takes and the one CUDA, HIP and Metal take for half, so one key
   draws the same half on every backend (gh-ocannl-951; the _vec/_lane forms consume 16 bits per
   lane on every backend). Returns the half STORAGE type: under native fp16 that is _Float16, and a
   uint16_t return type would convert the value through an integer (truncating every draw in [0, 1)
   to 0) and hand an integer to the half FMA builtin. */
HALF_T uint4x32_to_half_uniform(uint4x32_t x) {
    float f = uint32_to_single_uniform(x.v[0]);
    return FLOAT_TO_HALF(f);
}
|},
      [ "uint4x32_t"; "HALF_T"; "FLOAT_TO_HALF"; "uint32_to_single_uniform" ] );
    ( "uint4x32_to_half_uniform_vec",
      {|
/* Convert uint4x32 to 8 float16s uniform */
half8_t uint4x32_to_half_uniform_vec(uint4x32_t x) {
    half8_t result;
    for (int i = 0; i < 4; i++) {
        // Extract two 16-bit values and convert to float in [0, 1)
        float f1 = (x.v[i] & 0xFFFF) * (1.0f / 65536.0f);
        float f2 = ((x.v[i] >> 16) & 0xFFFF) * (1.0f / 65536.0f);
        
        // Convert to half precision - macros handle both native and emulated cases
        result.v[i*2 + 0] = FLOAT_TO_HALF(f1);
        result.v[i*2 + 1] = FLOAT_TO_HALF(f2);
    }
    return result;
}
|},
      [ "uint4x32_t"; "half8_t"; "FLOAT_TO_HALF" ] );
    (* Pure C conversion functions *)
    ( "bfloat16_to_single",
      {|
/* BFloat16 to Float conversion (C function) */
float bfloat16_to_single(uint16_t bf16)
{
  /* BFloat16 format: 1 sign bit, 8 exponent bits, 7 mantissa bits
     To convert to float32, we shift left by 16 bits */
  uint32_t f32 = ((uint32_t)bf16) << 16;
  float f;
  memcpy(&f, &f32, sizeof(f));
  return f;
}
|},
      [] );
    ( "single_to_bfloat16",
      {|
/* Float to BFloat16 conversion (C function) */
uint16_t single_to_bfloat16(float f)
{
  uint32_t f32;
  uint32_t rounded;
  memcpy(&f32, &f, sizeof(f32));

  /* A NaN is quieted and truncated, keeping its sign and the top of its payload (gh-ocannl-1069).
     The rounding add below would carry a NaN's payload out of the NaN class: 0x7F800001 truncates
     to 0x7F80 (+inf), and 0x7FFF8000 carries into the sign bit (-0) -- 131072 of the f32 NaNs. */
  if ((f32 & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t)((f32 >> 16) | 0x0040u);

  /* Round to nearest even */
  rounded = f32 + 0x7FFFu + ((f32 >> 16) & 1u);
  return (uint16_t)(rounded >> 16);
}
|},
      [] );
    (* Whole-vector conversion between 16-bit storage and f32 compute registers (gh-ocannl-517). The
       vector typedefs are minted by the codegen (their lane count follows cc_vector_bytes), so they
       are macro parameters; [LANES] is only used by the fallback arms. Each fallback goes through
       the very scalar conversion the serial path uses, which is what makes the vectorized rendering
       bitwise identical to its serial remainder loop on any compiler. *)
    ( "OCANNL_HAS_CONVERTVECTOR",
      {|
#if __has_builtin(__builtin_convertvector)
  #define OCANNL_HAS_CONVERTVECTOR 1
#else
  #define OCANNL_HAS_CONVERTVECTOR 0
#endif
|},
      [] );
    ( "OCANNL_VEC_WIDEN_BFLOAT16",
      {|
/* bfloat16 is the top 16 bits of a float: zero-extend and shift. LANES is the number of VALID
   lanes -- the vector types' width, or fewer for a partial vector (gh-ocannl-620), whose lanes past
   LANES read as zero and are never written back: only LANES elements cross the memory boundary. */
#if OCANNL_HAS_CONVERTVECTOR
  #define OCANNL_VEC_WIDEN_BFLOAT16(U16V, U32V, LANES, dst, src) do { \
    U16V ocannl_nb__ = {0}; __builtin_memcpy(&ocannl_nb__, (src), (LANES) * 2); \
    U32V ocannl_nw__ = __builtin_convertvector(ocannl_nb__, U32V) << 16; \
    __builtin_memcpy(&(dst), &ocannl_nw__, sizeof(dst)); \
  } while (0)
#else
  #define OCANNL_VEC_WIDEN_BFLOAT16(U16V, U32V, LANES, dst, src) do { \
    const unsigned short *ocannl_ps__ = (src); \
    for (int ocannl_l__ = 0; ocannl_l__ < (LANES); ++ocannl_l__) \
      (dst)[ocannl_l__] = bfloat16_to_single(ocannl_ps__[ocannl_l__]); \
  } while (0)
#endif
|},
      [ "OCANNL_HAS_CONVERTVECTOR"; "bfloat16_to_single" ] );
    ( "OCANNL_VEC_NARROW_BFLOAT16",
      {|
/* single_to_bfloat16's round-to-nearest-even, lane-wise, NaN guard included: a NaN lane takes the
   quieted, truncated form instead of the rounding add. The comparison yields 0 or all-ones per lane
   (as a signed vector, hence the cast), which selects between the two. */
#if OCANNL_HAS_CONVERTVECTOR
  #define OCANNL_VEC_NARROW_BFLOAT16(U16V, U32V, LANES, dst, src) do { \
    U32V ocannl_nb__; __builtin_memcpy(&ocannl_nb__, &(src), sizeof(ocannl_nb__)); \
    U32V ocannl_nq__ = (U32V)((ocannl_nb__ & 0x7FFFFFFFu) > 0x7F800000u); \
    U32V ocannl_nr__ = (ocannl_nq__ & (ocannl_nb__ | 0x00400000u)) \
      | (~ocannl_nq__ & (ocannl_nb__ + 0x7FFFu + ((ocannl_nb__ >> 16) & 1u))); \
    U16V ocannl_nn__ = __builtin_convertvector(ocannl_nr__ >> 16, U16V); \
    __builtin_memcpy((dst), &ocannl_nn__, (LANES) * 2); \
  } while (0)
#else
  #define OCANNL_VEC_NARROW_BFLOAT16(U16V, U32V, LANES, dst, src) do { \
    unsigned short *ocannl_pd__ = (dst); \
    for (int ocannl_l__ = 0; ocannl_l__ < (LANES); ++ocannl_l__) \
      ocannl_pd__[ocannl_l__] = single_to_bfloat16((src)[ocannl_l__]); \
  } while (0)
#endif
|},
      [ "OCANNL_HAS_CONVERTVECTOR"; "single_to_bfloat16" ] );
    ( "OCANNL_HALF_FMA",
      {|
/* The fused multiply-add of fp16 arithmetic (gh-ocannl-516), shared by the scalar rendering and
   the per-lane fallback of the vector rendering so the two cannot round differently: a native fp16
   FMA rounds once, while promoting to fmaf rounds at float and then again at fp16.

   Which arm is taken is the SAME target question that decides whether C_syntax.vec_acc_fma has a
   single-rounding whole-vector arm to offer, and the guards are kept in step deliberately
   (gh-ocannl-621): the second arm's condition is the one cc_backend's fp16 probe calls `Native`,
   which is also what the AVX512-FP16 and NEON rows of vec_fma_builtin key off. Were the vector
   body to round once while this macro rounded twice, the vector rendering would stop matching the
   scalar peel and the serial fallback it promises to equal bit for bit -- and the two spellings do
   NOT agree: over 4.1e8 fp16 triples they differ on roughly one in ten thousand. Not only in the
   corners, either -- restricted to triples whose three operands are all NORMAL fp16 values, they
   still differ once in ~29000 (3393 of 9.9e7), so no input-range argument retires the question.
   float's 24 bits are 2p+2 for fp16, which makes double rounding innocuous for a single
   multiplication or addition, but an FMA's exact a*b+c can need far more than 24 bits and the
   guarantee does not extend to it.

   So on a native target the second arm changes gcc's fp16 results -- toward clang's, and toward
   the GPU backends' single-rounding __hfma / fma(half,...): before it, gcc alone rounded fp16 FMAs
   twice. It is also the larger of the two speedups this seam had left, on both AVX512-FP16 and
   ARMv8.2-FP16 taking the emitted micro-kernel from 5-10 instructions per FMA to under 2, because
   the promoting arm widens and narrows every lane inside the k-loop. On a promoted target nothing
   changes: the guard is false there.

   __builtin_fmaf16 is guarded on the ISA feature rather than on __has_builtin, which always
   answers yes for it: where the hardware instruction is absent gcc emits a call to fmaf16(), which
   a glibc need not export at all -- on the machine this was written on, linking one fails.

   The first arm additionally requires the native type, and the fallback goes through
   HALF_TO_FLOAT / FLOAT_TO_HALF rather than plain casts: under narrow_compute_f32=false on a
   target without _Float16, HALF_T is uint16_t, where a cast would compute on the raw half bit
   pattern (0x3c00 instead of 1.0) and the builtins would not compile at all.

   Only the first arm accepts vectors as well as scalars; every caller passes scalars. */
#if HAS_NATIVE_FLOAT16 && OCANNL_HAS_ELEMENTWISE_FMA
  #define OCANNL_HALF_FMA(a, b, c) __builtin_elementwise_fma((a), (b), (c))
#elif HAS_NATIVE_FLOAT16 && \
    (defined(__AVX512FP16__) || defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC))
  #define OCANNL_HALF_FMA(a, b, c) __builtin_fmaf16((a), (b), (c))
#else
  #define OCANNL_HALF_FMA(a, b, c) \
    FLOAT_TO_HALF(fmaf(HALF_TO_FLOAT(a), HALF_TO_FLOAT(b), HALF_TO_FLOAT(c)))
#endif
|},
      [ "HALF_T"; "HAS_NATIVE_FLOAT16"; "HALF_TO_FLOAT"; "FLOAT_TO_HALF" ] );
    ( "OCANNL_VEC_WIDEN_HALF",
      {|
/* LANES: the valid lanes, as for the bfloat16 bridges above. */
#if HAS_NATIVE_FLOAT16 && OCANNL_HAS_CONVERTVECTOR
  #define OCANNL_VEC_WIDEN_HALF(FV, HV, LANES, dst, src) do { \
    HV ocannl_nh__ = {0}; __builtin_memcpy(&ocannl_nh__, (src), (LANES) * 2); \
    (dst) = __builtin_convertvector(ocannl_nh__, FV); \
  } while (0)
#else
  #define OCANNL_VEC_WIDEN_HALF(FV, HV, LANES, dst, src) do { \
    const HALF_T *ocannl_ps__ = (src); \
    for (int ocannl_l__ = 0; ocannl_l__ < (LANES); ++ocannl_l__) \
      (dst)[ocannl_l__] = HALF_TO_FLOAT(ocannl_ps__[ocannl_l__]); \
  } while (0)
#endif
|},
      [ "OCANNL_HAS_CONVERTVECTOR"; "HAS_NATIVE_FLOAT16"; "HALF_T"; "HALF_TO_FLOAT" ] );
    ( "OCANNL_VEC_NARROW_HALF",
      {|
#if HAS_NATIVE_FLOAT16 && OCANNL_HAS_CONVERTVECTOR
  #define OCANNL_VEC_NARROW_HALF(HV, LANES, dst, src) do { \
    HV ocannl_nh__ = __builtin_convertvector((src), HV); \
    __builtin_memcpy((dst), &ocannl_nh__, (LANES) * 2); \
  } while (0)
#else
  #define OCANNL_VEC_NARROW_HALF(HV, LANES, dst, src) do { \
    HALF_T *ocannl_pd__ = (dst); \
    for (int ocannl_l__ = 0; ocannl_l__ < (LANES); ++ocannl_l__) \
      ocannl_pd__[ocannl_l__] = FLOAT_TO_HALF((src)[ocannl_l__]); \
  } while (0)
#endif
|},
      [ "OCANNL_HAS_CONVERTVECTOR"; "HAS_NATIVE_FLOAT16"; "HALF_T"; "FLOAT_TO_HALF" ] );
    (* The attribute [C_syntax.compile_proc] puts on a kernel whose rendering declared a vector
       wider than 32 bytes (gh-ocannl-1103). clang's x86 tuning prefers 256-bit vectors on
       [x86-64-v4] and the Intel AVX-512 parts, and under that preference it legalizes a 64-byte GNU
       C vector the function body declares as two [ymm] halves unless the function says it needs the
       width: a register tile's 24 accumulators become 48 and spill (66 stack references per k step
       at [x86-64-v4], where the whole-width tile holds none). [min_vector_width] changes only which
       vector types are legal in the function, OpenMP-outlined bodies included; what width clang's
       own auto-vectorizer picks is still its tuning's. gcc has no such attribute and keeps a
       64-byte vector whole where the target has it. *)
    ( "OCANNL_WIDE_VECTOR_KERNEL",
      {|
#ifndef __has_attribute
  #define __has_attribute(x) 0
#endif
#if __has_attribute(min_vector_width)
  #define OCANNL_WIDE_VECTOR_KERNEL __attribute__((min_vector_width(512)))
#else
  #define OCANNL_WIDE_VECTOR_KERNEL
#endif
|},
      [] );
    (* The widening bridges at the three f32 lane counts an x86 register holds, each ONE packed
       instruction where the target has it (gh-ocannl-1072), and otherwise the portable bridge above
       under another name. [C_syntax.vec_widen_macro] picks the name; the arguments are the portable
       bridge's, so the fallback is a rename.

       The portable spellings are whole-vector C, but gcc lowers two of them badly inside the
       register tile's k-loop. [__builtin_convertvector] from a [_Float16] vector to a float one has
       no packed pattern without AVX512-FP16, so gcc widens lane by lane -- each half moved to a
       general-purpose register ([movq], [shrq]), reinserted ([vpinsrw]) and converted by a scalar
       [vcvtph2ps] -- and the temporaries spill: 12 to 35 stack references per k step at [x86-64-v3]
       and [x86-64-v4] on gcc 15. And gcc 13 lowers the zero-extension of an 8-byte [unsigned short]
       vector as a split, two [vpmovzxwd] and a [vpunpcklqdq], whose two extra registers spill the
       31-register bf16 tile at [-march=sapphirerapids]. The builtins below are [<immintrin.h>]'s
       [_mm*_cvtph_ps] and [_mm*_cvtepu16_epi32] as gcc spells them (the 512-bit forms with an
       all-ones mask and [_MM_FROUND_CUR_DIRECTION] = 4), so their semantics are the ISA's. A
       compiler that spells one differently -- clang implements [_mm*_cvtepu16_epi32] with
       [__builtin_convertvector], and lowers that well -- fails [__has_builtin] and takes the
       portable bridge, as in [C_syntax.vec_fma_builtin].

       Parity: the bf16 arms are the portable arm's zero-extend and shift, bit for bit. The fp16
       arms keep the portable arm's [HAS_NATIVE_FLOAT16] guard, so they replace only its
       [__builtin_convertvector] from [_Float16] -- the conversion gcc itself emits for the scalar
       path's [(float)] cast of a [_Float16] on an F16C target, [vcvtph2ps] (quieting a signalling
       NaN exactly as the scalar path does). Without [_Float16], the scalar path is the software
       codec, and so is the fallback. A partial vector (LANES below the width) is zeroed and only
       LANES elements are read, as in the portable arms. *)
    ( "OCANNL_VEC_WIDEN_BFLOAT16_X4",
      {|
#if defined(__SSE4_1__) && __has_builtin(__builtin_ia32_pmovzxwd128)
  #define OCANNL_VEC_WIDEN_BFLOAT16_X4(U16V, U32V, LANES, dst, src) do { \
    typedef short ocannl_w8hi__ __attribute__((vector_size(16))); \
    typedef long long ocannl_w2di__ __attribute__((vector_size(16))); \
    long long ocannl_nq__ = 0; __builtin_memcpy(&ocannl_nq__, (src), (LANES) * 2); \
    (dst) = (__typeof__(dst))((U32V)__builtin_ia32_pmovzxwd128( \
      (ocannl_w8hi__)(ocannl_w2di__){ocannl_nq__, 0}) << 16); \
  } while (0)
#else
  #define OCANNL_VEC_WIDEN_BFLOAT16_X4 OCANNL_VEC_WIDEN_BFLOAT16
#endif
|},
      [ "OCANNL_VEC_WIDEN_BFLOAT16" ] );
    ( "OCANNL_VEC_WIDEN_BFLOAT16_X8",
      {|
#if defined(__AVX2__) && __has_builtin(__builtin_ia32_pmovzxwd256)
  #define OCANNL_VEC_WIDEN_BFLOAT16_X8(U16V, U32V, LANES, dst, src) do { \
    typedef short ocannl_w8hi__ __attribute__((vector_size(16))); \
    ocannl_w8hi__ ocannl_nb__ = {0}; __builtin_memcpy(&ocannl_nb__, (src), (LANES) * 2); \
    (dst) = (__typeof__(dst))((U32V)__builtin_ia32_pmovzxwd256(ocannl_nb__) << 16); \
  } while (0)
#else
  #define OCANNL_VEC_WIDEN_BFLOAT16_X8 OCANNL_VEC_WIDEN_BFLOAT16
#endif
|},
      [ "OCANNL_VEC_WIDEN_BFLOAT16" ] );
    ( "OCANNL_VEC_WIDEN_BFLOAT16_X16",
      {|
#if defined(__AVX512F__) && __has_builtin(__builtin_ia32_pmovzxwd512_mask)
  #define OCANNL_VEC_WIDEN_BFLOAT16_X16(U16V, U32V, LANES, dst, src) do { \
    typedef short ocannl_w16hi__ __attribute__((vector_size(32))); \
    typedef int ocannl_w16si__ __attribute__((vector_size(64))); \
    ocannl_w16hi__ ocannl_nb__ = {0}; __builtin_memcpy(&ocannl_nb__, (src), (LANES) * 2); \
    (dst) = (__typeof__(dst))((U32V)__builtin_ia32_pmovzxwd512_mask( \
      ocannl_nb__, (ocannl_w16si__){0}, (unsigned short)-1) << 16); \
  } while (0)
#else
  #define OCANNL_VEC_WIDEN_BFLOAT16_X16 OCANNL_VEC_WIDEN_BFLOAT16
#endif
|},
      [ "OCANNL_VEC_WIDEN_BFLOAT16" ] );
    ( "OCANNL_VEC_WIDEN_HALF_X4",
      {|
#if HAS_NATIVE_FLOAT16 && defined(__F16C__) && __has_builtin(__builtin_ia32_vcvtph2ps)
  #define OCANNL_VEC_WIDEN_HALF_X4(FV, HV, LANES, dst, src) do { \
    typedef short ocannl_w8hi__ __attribute__((vector_size(16))); \
    typedef long long ocannl_w2di__ __attribute__((vector_size(16))); \
    long long ocannl_nq__ = 0; __builtin_memcpy(&ocannl_nq__, (src), (LANES) * 2); \
    (dst) = (FV)__builtin_ia32_vcvtph2ps((ocannl_w8hi__)(ocannl_w2di__){ocannl_nq__, 0}); \
  } while (0)
#else
  #define OCANNL_VEC_WIDEN_HALF_X4 OCANNL_VEC_WIDEN_HALF
#endif
|},
      [ "HAS_NATIVE_FLOAT16"; "OCANNL_VEC_WIDEN_HALF" ] );
    ( "OCANNL_VEC_WIDEN_HALF_X8",
      {|
#if HAS_NATIVE_FLOAT16 && defined(__F16C__) && __has_builtin(__builtin_ia32_vcvtph2ps256)
  #define OCANNL_VEC_WIDEN_HALF_X8(FV, HV, LANES, dst, src) do { \
    typedef short ocannl_w8hi__ __attribute__((vector_size(16))); \
    ocannl_w8hi__ ocannl_nh__ = {0}; __builtin_memcpy(&ocannl_nh__, (src), (LANES) * 2); \
    (dst) = (FV)__builtin_ia32_vcvtph2ps256(ocannl_nh__); \
  } while (0)
#else
  #define OCANNL_VEC_WIDEN_HALF_X8 OCANNL_VEC_WIDEN_HALF
#endif
|},
      [ "HAS_NATIVE_FLOAT16"; "OCANNL_VEC_WIDEN_HALF" ] );
    ( "OCANNL_VEC_WIDEN_HALF_X16",
      {|
#if HAS_NATIVE_FLOAT16 && defined(__AVX512F__) && __has_builtin(__builtin_ia32_vcvtph2ps512_mask)
  #define OCANNL_VEC_WIDEN_HALF_X16(FV, HV, LANES, dst, src) do { \
    typedef short ocannl_w16hi__ __attribute__((vector_size(32))); \
    typedef float ocannl_w16sf__ __attribute__((vector_size(64))); \
    ocannl_w16hi__ ocannl_nh__ = {0}; __builtin_memcpy(&ocannl_nh__, (src), (LANES) * 2); \
    (dst) = (FV)__builtin_ia32_vcvtph2ps512_mask( \
      ocannl_nh__, (ocannl_w16sf__){0}, (unsigned short)-1, 4); \
  } while (0)
#else
  #define OCANNL_VEC_WIDEN_HALF_X16 OCANNL_VEC_WIDEN_HALF
#endif
|},
      [ "HAS_NATIVE_FLOAT16"; "OCANNL_VEC_WIDEN_HALF" ] );
    (* The fp16 narrowing bridges at the same three widths (gh-ocannl-1101), the store-side twins of
       the widenings above, named by [C_syntax.vec_narrow_macro]; same arguments as the portable
       [OCANNL_VEC_NARROW_HALF], so the fallback is a rename.

       [__builtin_convertvector] from a float vector to a [_Float16] one has no packed pattern in
       gcc without AVX512-FP16 either, so gcc narrows lane by lane: each lane shuffled to the bottom
       of its own register, converted by its own [vcvtps2ph] and the halves reassembled with
       [vpunpck*] -- 4, 8 and 16 conversions at the three widths (gcc 13.4 and 15.2, at [x86-64-v3]
       and [x86-64-v4]), in every register-tile C-tile store and every vectorized fp16-storage
       write. F16C converts a whole register at once. The builtins are [<immintrin.h>]'s
       [_mm_cvtps_ph], [_mm256_cvtps_ph] and [_mm512_cvtps_ph] as gcc and clang both spell them,
       with [_MM_FROUND_CUR_DIRECTION] = 4: round as MXCSR says, as the scalar path's [(_Float16)]
       cast does.

       Parity: on an F16C target without AVX512-FP16 the scalar path's cast IS [vcvtps2ph], one lane
       at a time. With AVX512-FP16 it is [vcvtss2sh] (and gcc's own vector lowering [vcvtps2phx]):
       the same IEEE round-to-nearest-even conversion with the same NaN rule -- quieted, payload
       truncated to its top bits -- so under the default MXCSR (no DAZ/FTZ) they agree on every
       input. The arms are taken there too, as the widening ones are, so a width has one instruction
       choice per x86 target on either compiler. Every f32 input through the X4 and X8 arms, and the
       scalar cast, matches an integer round-to-nearest-even reference bit for bit (all 2^32, F16C
       host, gh-ocannl-1101); [test/operations/narrow_storage_compute.ml] pins the arm its host
       takes against the serial twin at every fp16 rounding boundary. Only LANES elements are
       written, as in the portable arm. *)
    ( "OCANNL_VEC_NARROW_HALF_X4",
      {|
#if HAS_NATIVE_FLOAT16 && defined(__F16C__) && __has_builtin(__builtin_ia32_vcvtps2ph)
  #define OCANNL_VEC_NARROW_HALF_X4(HV, LANES, dst, src) do { \
    typedef float ocannl_n4sf__ __attribute__((vector_size(16))); \
    typedef short ocannl_n8hi__ __attribute__((vector_size(16))); \
    ocannl_n8hi__ ocannl_nh__ = __builtin_ia32_vcvtps2ph((ocannl_n4sf__)(src), 4); \
    __builtin_memcpy((dst), &ocannl_nh__, (LANES) * 2); \
  } while (0)
#else
  #define OCANNL_VEC_NARROW_HALF_X4 OCANNL_VEC_NARROW_HALF
#endif
|},
      [ "HAS_NATIVE_FLOAT16"; "OCANNL_VEC_NARROW_HALF" ] );
    ( "OCANNL_VEC_NARROW_HALF_X8",
      {|
#if HAS_NATIVE_FLOAT16 && defined(__F16C__) && __has_builtin(__builtin_ia32_vcvtps2ph256)
  #define OCANNL_VEC_NARROW_HALF_X8(HV, LANES, dst, src) do { \
    typedef float ocannl_n8sf__ __attribute__((vector_size(32))); \
    typedef short ocannl_n8hi__ __attribute__((vector_size(16))); \
    ocannl_n8hi__ ocannl_nh__ = __builtin_ia32_vcvtps2ph256((ocannl_n8sf__)(src), 4); \
    __builtin_memcpy((dst), &ocannl_nh__, (LANES) * 2); \
  } while (0)
#else
  #define OCANNL_VEC_NARROW_HALF_X8 OCANNL_VEC_NARROW_HALF
#endif
|},
      [ "HAS_NATIVE_FLOAT16"; "OCANNL_VEC_NARROW_HALF" ] );
    ( "OCANNL_VEC_NARROW_HALF_X16",
      {|
#if HAS_NATIVE_FLOAT16 && defined(__AVX512F__) && __has_builtin(__builtin_ia32_vcvtps2ph512_mask)
  #define OCANNL_VEC_NARROW_HALF_X16(HV, LANES, dst, src) do { \
    typedef float ocannl_n16sf__ __attribute__((vector_size(64))); \
    typedef short ocannl_n16hi__ __attribute__((vector_size(32))); \
    ocannl_n16hi__ ocannl_nh__ = __builtin_ia32_vcvtps2ph512_mask( \
      (ocannl_n16sf__)(src), 4, (ocannl_n16hi__){0}, (unsigned short)-1); \
    __builtin_memcpy((dst), &ocannl_nh__, (LANES) * 2); \
  } while (0)
#else
  #define OCANNL_VEC_NARROW_HALF_X16 OCANNL_VEC_NARROW_HALF
#endif
|},
      [ "HAS_NATIVE_FLOAT16"; "OCANNL_VEC_NARROW_HALF" ] );
    (* A register tile's bf16 A column, widened a group of four rows at a time (named by
       [C_syntax.vec_widen_rows_macros], which says why only there). [..._ROWS_X4] declares [dst]
       and widens the four rows' storage elements [a0]..[a3] into it; [OCANNL_VEC_WIDENED_ROW_X4] is
       row [lane]'s widened scalar, and [widened] is that row's per-element conversion.

       On aarch64 the rows are packed: one [shll] per k step for the whole group, and each row's FMA
       takes its multiplier from a lane of that one register ([fmla ..., v.s[lane]]), where four
       per-row widenings would hold four registers. Elsewhere [..._ROWS_X4] is empty and a row is
       its own [widened] conversion, so after preprocessing the tile is the per-row rendering, byte
       for byte.

       Parity: the zero-extend and shift is [bfloat16_to_single]'s, lane for lane, as in
       [OCANNL_VEC_WIDEN_BFLOAT16]. *)
    ( "OCANNL_VEC_WIDEN_BFLOAT16_ROWS_X4",
      {|
#if defined(__aarch64__) && OCANNL_HAS_CONVERTVECTOR
  #define OCANNL_VEC_WIDEN_BFLOAT16_ROWS_X4(U16V, U32V, FV, dst, a0, a1, a2, a3) \
    FV dst; do { \
      U16V ocannl_rh__ = {(a0), (a1), (a2), (a3)}; \
      U32V ocannl_rw__ = __builtin_convertvector(ocannl_rh__, U32V) << 16; \
      __builtin_memcpy(&(dst), &ocannl_rw__, sizeof(dst)); \
    } while (0)
  #define OCANNL_VEC_WIDENED_ROW_X4(packed, lane, widened) ((packed)[lane])
#else
  #define OCANNL_VEC_WIDEN_BFLOAT16_ROWS_X4(U16V, U32V, FV, dst, a0, a1, a2, a3) do { } while (0)
  #define OCANNL_VEC_WIDENED_ROW_X4(packed, lane, widened) (widened)
#endif
|},
      [ "OCANNL_HAS_CONVERTVECTOR" ] );
    ( "half_to_single",
      {|
/* Half (Float16) to Float conversion (C function) */
float half_to_single(uint16_t h)
{
  HALF_T half_val = UINT16_TO_HALF(h);
  return HALF_TO_FLOAT(half_val);
}
|},
      [ "HALF_T"; "UINT16_TO_HALF"; "HALF_TO_FLOAT" ] );
    ( "single_to_half",
      {|
/* Float to Half (Float16) conversion (C function) */
uint16_t single_to_half(float f)
{
  HALF_T half_val = FLOAT_TO_HALF(f);
  return HALF_TO_UINT16(half_val);
}
|},
      [ "HALF_T"; "FLOAT_TO_HALF"; "HALF_TO_UINT16" ] );
    ( "fp8_to_single",
      {|
/* FP8 E5M2 format to Float conversion (C function)
   Format: 1 sign bit, 5 exponent bits, 2 mantissa bits */
float fp8_to_single(uint8_t fp8)
{
  /* Handle zero */
  if (fp8 == 0)
  {
    return 0.0f;
  }

  uint32_t sign = (fp8 >> 7) & 1;
  uint32_t exp = (fp8 >> 2) & 0x1F;
  uint32_t mant = fp8 & 0x3;

  /* Handle special cases */
  if (exp == 0x1F)
  { /* Infinity or NaN */
    if (mant == 0)
    {
      return sign ? -INFINITY : INFINITY;
    }
    else
    {
      return NAN;
    }
  }

  /* Denormalized numbers */
  if (exp == 0)
  {
    float result = ldexpf((float)mant / 4.0f, -14);
    if (sign)
      result = -result;
    return result;
  }

  /* Normalized numbers */
  float result = (1.0f + (float)mant * 0.25f) * ldexpf(1.0f, (int)exp - 15);
  if (sign)
    result = -result;

  return result;
}
|},
      [] );
    ( "single_to_fp8",
      {|
/* Float to FP8 E5M2 conversion (C function).

   IEEE round-to-nearest-even, subnormals rounded rather than flushed, signed zero preserved,
   and finite overflow saturating to the max finite magnitude. Every one of those is the behavior
   of the native GPU fp8 types (CUDA __nv_fp8_e5m2, HIP __hip_fp8_e5m2), verified against both
   over all 2^32 float bit patterns, so a value narrowed on the host, on cc or on Metal lands on
   the same code a CUDA or HIP kernel would produce. See
   docs/agent-notes/backend-precision-and-simd.md for the two inputs where the vendors themselves
   disagree (an already-infinite input, and the sign of a NaN). */
uint8_t single_to_fp8(float f)
{
  uint32_t bits;
  memcpy(&bits, &f, sizeof(bits));
  uint32_t sign = (bits >> 24) & 0x80u;
  uint32_t e32 = (bits >> 23) & 0xFFu;
  uint32_t m32 = bits & 0x7FFFFFu;

  /* Infinity and NaN keep their sign; a NaN takes the all-mantissa code. */
  if (e32 == 0xFFu)
  {
    return (uint8_t)(sign | (m32 != 0u ? 0x7Fu : 0x7Cu));
  }
  /* Zero, and f32 subnormals, which are far below e5m2's smallest subnormal. */
  if (e32 == 0u)
  {
    return (uint8_t)sign;
  }

  int exp = (int)e32 - 112; /* the e5m2 exponent field: rebias 127 -> 15 */
  if (exp >= 31)
  {
    return (uint8_t)(sign | 0x7Bu); /* saturate to the largest finite, 57344 */
  }
  if (exp <= 0)
  {
    /* Subnormal target: the value is q * 2^-16 with q in [0, 4), rounded to nearest even. */
    uint32_t sig = 0x800000u | m32; /* make the implicit leading bit explicit */
    int shift = 22 - exp;           /* >= 22 */
    if (shift > 24)
    {
      return (uint8_t)sign; /* below half the smallest subnormal: signed zero */
    }
    uint32_t q = sig >> shift;
    uint32_t rest = sig & ((1u << shift) - 1u);
    uint32_t tie = 1u << (shift - 1);
    if (rest > tie || (rest == tie && (q & 1u)))
    {
      q++;
    }
    if (q > 3u)
    {
      return (uint8_t)(sign | 0x04u); /* rounded up into the smallest normal, 2^-14 */
    }
    return (uint8_t)(sign | q);
  }

  uint32_t mant = m32 >> 21;
  uint32_t rest = m32 & 0x1FFFFFu;
  uint32_t tie = 0x100000u;
  if (rest > tie || (rest == tie && (mant & 1u)))
  {
    mant++;
  }
  if (mant > 3u)
  {
    /* The mantissa rounded past the top: carry into the exponent. */
    mant = 0u;
    exp++;
    if (exp >= 31)
    {
      return (uint8_t)(sign | 0x7Bu);
    }
  }
  return (uint8_t)(sign | ((uint32_t)exp << 2) | mant);
}
|},
      [] );
    ( "double_to_fp8",
      {|
/* Double to FP8 E5M2 conversion (C function).

   One step, not double(f64 -> f32 -> e5m2): rounding twice moves a value that is just off an f32
   tie onto it, and the second rounding then breaks that tie by a rule the first rounding has
   already made wrong. The CUDA and HIP fp8 types convert straight from the double, so a
   two-step host or cc conversion disagreed with them for exactly those inputs (gh-ocannl-648).
   Same rules as [single_to_fp8], read off f64's fields. */
uint8_t double_to_fp8(double f)
{
  uint64_t bits;
  memcpy(&bits, &f, sizeof(bits));
  uint32_t sign = (uint32_t)((bits >> 56) & 0x80u);
  uint32_t e64 = (uint32_t)((bits >> 52) & 0x7FFu);
  uint64_t m64 = bits & 0xFFFFFFFFFFFFFULL;

  /* Infinity and NaN keep their sign, exactly as [single_to_fp8] does — the two codecs must not
     differ from each other, whatever the vendors do (CUDA drops a NaN's sign, HIP keeps it). */
  if (e64 == 0x7FFu)
  {
    return (uint8_t)(sign | (m64 != 0ULL ? 0x7Fu : 0x7Cu));
  }
  if (e64 == 0u)
  {
    return (uint8_t)sign; /* zero, and f64 subnormals, far below e5m2's smallest */
  }

  int exp = (int)e64 - 1008; /* the e5m2 exponent field: rebias 1023 -> 15 */
  if (exp >= 31)
  {
    return (uint8_t)(sign | 0x7Bu); /* saturate to the largest finite, 57344 */
  }
  if (exp <= 0)
  {
    uint64_t sig = 0x10000000000000ULL | m64; /* the implicit leading bit, made explicit */
    int shift = 51 - exp;                     /* >= 51 */
    if (shift > 53)
    {
      return (uint8_t)sign; /* below half the smallest subnormal: signed zero */
    }
    uint64_t q = sig >> shift;
    uint64_t rest = sig & ((1ULL << shift) - 1ULL);
    uint64_t tie = 1ULL << (shift - 1);
    if (rest > tie || (rest == tie && (q & 1ULL)))
    {
      q++;
    }
    if (q > 3ULL)
    {
      return (uint8_t)(sign | 0x04u); /* rounded up into the smallest normal, 2^-14 */
    }
    return (uint8_t)(sign | (uint32_t)q);
  }

  uint32_t mant = (uint32_t)(m64 >> 50);
  uint64_t rest = m64 & 0x3FFFFFFFFFFFFULL;
  uint64_t tie = 0x2000000000000ULL;
  if (rest > tie || (rest == tie && (mant & 1u)))
  {
    mant++;
  }
  if (mant > 3u)
  {
    mant = 0u;
    exp++;
    if (exp >= 31)
    {
      return (uint8_t)(sign | 0x7Bu);
    }
  }
  return (uint8_t)(sign | ((uint32_t)exp << 2) | mant);
}
|},
      [] );
    (* Conversion functions from various precisions to uint4x32_t *)
    ( "int32_to_uint4x32",
      {|
uint4x32_t int32_to_uint4x32(int32_t x) {
    /* Spread bits across all 4 components for better entropy with light threefry.
       Without this, consecutive counter values produce nearly identical v[0] outputs
       from 2-round threefry, causing periodicity in random number generation. */
    uint32_t u = (uint32_t)x;
    uint4x32_t result = {{
        u,
        u ^ 0x9E3779B9,              /* golden ratio constant */
        u ^ 0x6C078965,              /* Knuth's MMIX constant */
        u ^ ((u << 16) | (u >> 16))  /* bit rotation */
    }};
    return result;
}
|},
      [ "uint4x32_t" ] );
    ( "int64_to_uint4x32",
      {|
uint4x32_t int64_to_uint4x32(int64_t x) {
    uint64_t bits = (uint64_t)x;
    uint4x32_t result = {{(uint32_t)(bits & 0xFFFFFFFF), (uint32_t)(bits >> 32), 0, 0}};
    return result;
}
|},
      [ "uint4x32_t" ] );
    ( "uint32_to_uint4x32",
      {|
uint4x32_t uint32_to_uint4x32(uint32_t x) {
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
}
|},
      [ "uint4x32_t" ] );
    ( "uint64_to_uint4x32",
      {|
uint4x32_t uint64_to_uint4x32(uint64_t x) {
    uint4x32_t result = {{(uint32_t)(x & 0xFFFFFFFF), (uint32_t)(x >> 32), 0, 0}};
    return result;
}
|},
      [ "uint4x32_t" ] );
    ( "single_to_uint4x32",
      {|
uint4x32_t single_to_uint4x32(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof(float));
    uint4x32_t result = {{bits, 0, 0, 0}};
    return result;
}
|},
      [ "uint4x32_t" ] );
    ( "double_to_uint4x32",
      {|
uint4x32_t double_to_uint4x32(double x) {
    uint64_t bits;
    memcpy(&bits, &x, sizeof(double));
    uint4x32_t result = {{(uint32_t)(bits & 0xFFFFFFFF), (uint32_t)(bits >> 32), 0, 0}};
    return result;
}
|},
      [ "uint4x32_t" ] );
    ( "byte_to_uint4x32",
      {|
uint4x32_t byte_to_uint4x32(unsigned char x) {
    uint4x32_t result = {{(uint32_t)x, 0, 0, 0}};
    return result;
}
|},
      [ "uint4x32_t" ] );
    ( "uint16_to_uint4x32",
      {|
uint4x32_t uint16_to_uint4x32(uint16_t x) {
    uint4x32_t result = {{(uint32_t)x, 0, 0, 0}};
    return result;
}
|},
      [ "uint4x32_t" ] );
    ( "bfloat16_to_uint4x32",
      {|
uint4x32_t bfloat16_to_uint4x32(uint16_t x) {
    uint4x32_t result = {{(uint32_t)x, 0, 0, 0}};
    return result;
}
|},
      [ "uint4x32_t" ] );
    ( "half_to_uint4x32",
      {|
uint4x32_t half_to_uint4x32(uint16_t x) {
    uint4x32_t result = {{(uint32_t)x, 0, 0, 0}};
    return result;
}
|},
      [ "uint4x32_t" ] );
    ( "fp8_to_uint4x32",
      {|
uint4x32_t fp8_to_uint4x32(uint8_t x) {
    uint4x32_t result = {{(uint32_t)x, 0, 0, 0}};
    return result;
}
|},
      [ "uint4x32_t" ] );
    (* More uint4x32 to various precision conversion functions *)
    ( "uint4x32_to_double_uniform",
      {|
/* Uint4x32 to float64 uniform - top 53 of the first 64 bits, so the int-to-double conversion
   is exact and the result stays below 1.0 (all 64 bits could round up to 2^64, yielding 1.0) */
double uint4x32_to_double_uniform(uint4x32_t x) {
    uint64_t combined = ((uint64_t)x.v[1] << 32) | x.v[0];
    return (combined >> 11) * (1.0 / 9007199254740992.0);
}
|},
      [ "uint4x32_t" ] );
    ( "uint4x32_to_int32_uniform",
      {|
/* Uint4x32 to int32 uniform - full range */
int32_t uint4x32_to_int32_uniform(uint4x32_t x) {
    return (int32_t)x.v[0];
}
|},
      [ "uint4x32_t" ] );
    ( "uint4x32_to_int64_uniform",
      {|
/* Uint4x32 to int64 uniform - full range */
int64_t uint4x32_to_int64_uniform(uint4x32_t x) {
    return (int64_t)(((uint64_t)x.v[1] << 32) | x.v[0]);
}
|},
      [ "uint4x32_t" ] );
    ( "uint4x32_to_uint32_uniform",
      {|
/* Uint4x32 to uint32 uniform - full range */
uint32_t uint4x32_to_uint32_uniform(uint4x32_t x) {
    return x.v[0];
}
|},
      [ "uint4x32_t" ] );
    ( "uint4x32_to_uint64_uniform",
      {|
/* Uint4x32 to uint64 uniform - full range */
uint64_t uint4x32_to_uint64_uniform(uint4x32_t x) {
    return ((uint64_t)x.v[1] << 32) | x.v[0];
}
|},
      [ "uint4x32_t" ] );
    ( "uint4x32_to_byte_uniform",
      {|
/* Uint4x32 to int8 uniform - full range */
int8_t uint4x32_to_byte_uniform(uint4x32_t x) {
    return (int8_t)(x.v[0] & 0xFF);
}
|},
      [ "uint4x32_t" ] );
    ( "uint4x32_to_uint16_uniform",
      {|
/* Uint4x32 to uint16 uniform - full range */
uint16_t uint4x32_to_uint16_uniform(uint4x32_t x) {
    return (uint16_t)(x.v[0] & 0xFFFF);
}
|},
      [ "uint4x32_t" ] );
    ( "uint4x32_to_bfloat16_uniform",
      {|
/* Uint4x32 to bfloat16 uniform - uses first 16 bits */
uint16_t uint4x32_to_bfloat16_uniform(uint4x32_t x) {
    /* Convert to float first, then to bfloat16 */
    float f = uint32_to_single_uniform(x.v[0]);
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    /* Round to nearest even for bfloat16 */
    uint16_t bf = bits >> 16;
    if ((bits & 0x8000) && ((bits & 0x7FFF) || (bf & 1))) bf++;
    return bf;
}
|},
      [ "uint4x32_t"; "uint32_to_single_uniform" ] );
    ( "uint4x32_to_fp8_uniform",
      {|
/* Uint4x32 to fp8 uniform - uses first 8 bits */
uint8_t uint4x32_to_fp8_uniform(uint4x32_t x) {
    return (uint8_t)(x.v[0] & 0xFF);
}
|},
      [ "uint4x32_t" ] );
    (* Vectorized conversion functions *)
    ( "uint4x32_to_single_uniform_vec",
      {|
/* Convert uint4x32 to 4 floats in [0, 1) */
float4_t uint4x32_to_single_uniform_vec(uint4x32_t x) {
    float4_t result;
    for (int i = 0; i < 4; i++) {
        result.v[i] = uint32_to_single_uniform(x.v[i]);
    }
    return result;
}
|},
      [ "uint4x32_t"; "float4_t"; "uint32_to_single_uniform" ] );
    ( "uint4x32_to_double_uniform_vec",
      {|
/* Convert uint4x32 to 2 doubles in [0, 1) - top 53 bits per lane pair, see
   uint4x32_to_double_uniform */
double2_t uint4x32_to_double_uniform_vec(uint4x32_t x) {
    double2_t result;
    uint64_t combined1 = ((uint64_t)x.v[1] << 32) | x.v[0];
    uint64_t combined2 = ((uint64_t)x.v[3] << 32) | x.v[2];
    result.v[0] = (combined1 >> 11) * (1.0 / 9007199254740992.0);
    result.v[1] = (combined2 >> 11) * (1.0 / 9007199254740992.0);
    return result;
}
|},
      [ "uint4x32_t"; "double2_t" ] );
    ( "uint4x32_to_int32_uniform_vec",
      {|
/* Convert uint4x32 to 4 int32s - full range */
int32x4_t uint4x32_to_int32_uniform_vec(uint4x32_t x) {
    int32x4_t result;
    for (int i = 0; i < 4; i++) {
        result.v[i] = (int32_t)x.v[i];
    }
    return result;
}
|},
      [ "uint4x32_t"; "int32x4_t" ] );
    ( "uint4x32_to_int64_uniform_vec",
      {|
/* Convert uint4x32 to 2 int64s - full range */
int64x2_t uint4x32_to_int64_uniform_vec(uint4x32_t x) {
    int64x2_t result;
    result.v[0] = (int64_t)(((uint64_t)x.v[1] << 32) | x.v[0]);
    result.v[1] = (int64_t)(((uint64_t)x.v[3] << 32) | x.v[2]);
    return result;
}
|},
      [ "uint4x32_t"; "int64x2_t" ] );
    ( "uint4x32_to_byte_uniform_vec",
      {|
/* Convert uint4x32 to 16 int8s - full range */
int8x16_t uint4x32_to_byte_uniform_vec(uint4x32_t x) {
    int8x16_t result;
    for (int i = 0; i < 4; i++) {
        result.v[i*4 + 0] = (int8_t)(x.v[i] & 0xFF);
        result.v[i*4 + 1] = (int8_t)((x.v[i] >> 8) & 0xFF);
        result.v[i*4 + 2] = (int8_t)((x.v[i] >> 16) & 0xFF);
        result.v[i*4 + 3] = (int8_t)((x.v[i] >> 24) & 0xFF);
    }
    return result;
}
|},
      [ "uint4x32_t"; "int8x16_t" ] );
    ( "uint4x32_to_uint16_uniform_vec",
      {|
/* Convert uint4x32 to 8 uint16s - full range */
uint16x8_t uint4x32_to_uint16_uniform_vec(uint4x32_t x) {
    uint16x8_t result;
    for (int i = 0; i < 4; i++) {
        result.v[i*2 + 0] = (uint16_t)(x.v[i] & 0xFFFF);
        result.v[i*2 + 1] = (uint16_t)((x.v[i] >> 16) & 0xFFFF);
    }
    return result;
}
|},
      [ "uint4x32_t"; "uint16x8_t" ] );
    ( "uint4x32_to_bfloat16_uniform_vec",
      {|
/* Convert uint4x32 to 8 bfloat16s uniform */
uint16x8_t uint4x32_to_bfloat16_uniform_vec(uint4x32_t x) {
    uint16x8_t result;
    for (int i = 0; i < 4; i++) {
        // Convert each uint32 to two bfloat16 values
        float f1 = ((x.v[i] & 0xFFFF) >> 0) * (1.0f / 65536.0f);
        float f2 = ((x.v[i] >> 16) & 0xFFFF) * (1.0f / 65536.0f);
        uint32_t bits1, bits2;
        memcpy(&bits1, &f1, sizeof(float));
        memcpy(&bits2, &f2, sizeof(float));
        // Round to nearest even for bfloat16
        uint16_t bf1 = bits1 >> 16;
        uint16_t bf2 = bits2 >> 16;
        // Check if we need to round up (guard bit set and round/sticky or LSB)
        if ((bits1 & 0x8000) && ((bits1 & 0x7FFF) || (bf1 & 1))) bf1++;
        if ((bits2 & 0x8000) && ((bits2 & 0x7FFF) || (bf2 & 1))) bf2++;
        result.v[i*2 + 0] = bf1;
        result.v[i*2 + 1] = bf2;
    }
    return result;
}
|},
      [ "uint4x32_t"; "uint16x8_t" ] );
    ( "uint4x32_to_fp8_uniform_vec",
      {|
/* Convert uint4x32 to 16 fp8s uniform (raw bit patterns). Returns int8x16_t to match
   [Ops.c_vec_typ_of_prec], which maps both byte and fp8 to int8x16_t; the casts preserve
   the bit patterns. */
int8x16_t uint4x32_to_fp8_uniform_vec(uint4x32_t x) {
    int8x16_t result;
    for (int i = 0; i < 4; i++) {
        result.v[i*4 + 0] = (int8_t)(x.v[i] & 0xFF);
        result.v[i*4 + 1] = (int8_t)((x.v[i] >> 8) & 0xFF);
        result.v[i*4 + 2] = (int8_t)((x.v[i] >> 16) & 0xFF);
        result.v[i*4 + 3] = (int8_t)((x.v[i] >> 24) & 0xFF);
    }
    return result;
}
|},
      [ "uint4x32_t"; "int8x16_t" ] );
    ( "uint4x32_to_uint32_uniform_vec",
      {|
/* Convert uint4x32 to 4 uint32s - full range */
uint32x4_t uint4x32_to_uint32_uniform_vec(uint4x32_t x) {
    uint32x4_t result;
    for (int i = 0; i < 4; i++) {
        result.v[i] = x.v[i];
    }
    return result;
}
|},
      [ "uint4x32_t"; "uint32x4_t" ] );
    ( "uint4x32_to_uint64_uniform_vec",
      {|
/* Convert uint4x32 to 2 uint64s - full range */
uint64x2_t uint4x32_to_uint64_uniform_vec(uint4x32_t x) {
    uint64x2_t result;
    result.v[0] = ((uint64_t)x.v[1] << 32) | x.v[0];
    result.v[1] = ((uint64_t)x.v[3] << 32) | x.v[2];
    return result;
}
|},
      [ "uint4x32_t"; "uint64x2_t" ] );
    (* Lane extraction from the packed uniform conversion (gh-509 task 4): minted by the virtualizer
       to inline packed-uniform results per cell. Implemented via the _vec builtins so the value
       stream is bitwise-identical to the vectorized stores by construction. *)
    ( "uint4x32_to_single_uniform_lane",
      {|
/* Lane of the packed single uniform conversion. */
float uint4x32_to_single_uniform_lane(uint4x32_t x, int32_t lane) {
    return uint4x32_to_single_uniform_vec(x).v[lane];
}
|},
      [ "uint4x32_t"; "uint4x32_to_single_uniform_vec" ] );
    ( "uint4x32_to_double_uniform_lane",
      {|
/* Lane of the packed double uniform conversion. */
double uint4x32_to_double_uniform_lane(uint4x32_t x, int32_t lane) {
    return uint4x32_to_double_uniform_vec(x).v[lane];
}
|},
      [ "uint4x32_t"; "uint4x32_to_double_uniform_vec" ] );
    ( "uint4x32_to_int32_uniform_lane",
      {|
/* Lane of the packed int32 uniform conversion. */
int32_t uint4x32_to_int32_uniform_lane(uint4x32_t x, int32_t lane) {
    return uint4x32_to_int32_uniform_vec(x).v[lane];
}
|},
      [ "uint4x32_t"; "uint4x32_to_int32_uniform_vec" ] );
    ( "uint4x32_to_int64_uniform_lane",
      {|
/* Lane of the packed int64 uniform conversion. */
int64_t uint4x32_to_int64_uniform_lane(uint4x32_t x, int32_t lane) {
    return uint4x32_to_int64_uniform_vec(x).v[lane];
}
|},
      [ "uint4x32_t"; "uint4x32_to_int64_uniform_vec" ] );
    ( "uint4x32_to_byte_uniform_lane",
      {|
/* Lane of the packed byte uniform conversion. */
int8_t uint4x32_to_byte_uniform_lane(uint4x32_t x, int32_t lane) {
    return uint4x32_to_byte_uniform_vec(x).v[lane];
}
|},
      [ "uint4x32_t"; "uint4x32_to_byte_uniform_vec" ] );
    ( "uint4x32_to_uint16_uniform_lane",
      {|
/* Lane of the packed uint16 uniform conversion. */
uint16_t uint4x32_to_uint16_uniform_lane(uint4x32_t x, int32_t lane) {
    return uint4x32_to_uint16_uniform_vec(x).v[lane];
}
|},
      [ "uint4x32_t"; "uint4x32_to_uint16_uniform_vec" ] );
    ( "uint4x32_to_bfloat16_uniform_lane",
      {|
/* Lane of the packed bfloat16 uniform conversion (bfloat16 bits as uint16). */
uint16_t uint4x32_to_bfloat16_uniform_lane(uint4x32_t x, int32_t lane) {
    return uint4x32_to_bfloat16_uniform_vec(x).v[lane];
}
|},
      [ "uint4x32_t"; "uint4x32_to_bfloat16_uniform_vec" ] );
    ( "uint4x32_to_half_uniform_lane",
      {|
/* Lane of the packed half uniform conversion. */
HALF_T uint4x32_to_half_uniform_lane(uint4x32_t x, int32_t lane) {
    return uint4x32_to_half_uniform_vec(x).v[lane];
}
|},
      [ "uint4x32_t"; "HALF_T"; "uint4x32_to_half_uniform_vec" ] );
    ( "uint4x32_to_fp8_uniform_lane",
      {|
/* Lane of the packed fp8 uniform conversion (raw bit pattern as int8). */
int8_t uint4x32_to_fp8_uniform_lane(uint4x32_t x, int32_t lane) {
    return uint4x32_to_fp8_uniform_vec(x).v[lane];
}
|},
      [ "uint4x32_t"; "uint4x32_to_fp8_uniform_vec" ] );
    ( "uint4x32_to_uint32_uniform_lane",
      {|
/* Lane of the packed uint32 uniform conversion. */
uint32_t uint4x32_to_uint32_uniform_lane(uint4x32_t x, int32_t lane) {
    return uint4x32_to_uint32_uniform_vec(x).v[lane];
}
|},
      [ "uint4x32_t"; "uint4x32_to_uint32_uniform_vec" ] );
    ( "uint4x32_to_uint64_uniform_lane",
      {|
/* Lane of the packed uint64 uniform conversion. */
uint64_t uint4x32_to_uint64_uniform_lane(uint4x32_t x, int32_t lane) {
    return uint4x32_to_uint64_uniform_vec(x).v[lane];
}
|},
      [ "uint4x32_t"; "uint4x32_to_uint64_uniform_vec" ] );
  ]

let builtins =
  integer_power_builtins ~prefix:"static inline" ~supports_double:true ~dialect:`C @ builtins

let source = includes ^ String.concat "" (List.map (fun (_, def, _) -> def) builtins)
