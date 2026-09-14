#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/bigarray.h>
/* The C implementations are shared with generated cc kernels. */
#include "builtins_shared.h"

/* OCaml wrapper functions */

/* Helper functions to convert between OCaml and C uint4x32_t */
/* The OCaml side of every uint4x32 stub is typed [int array], which carries no length, so the
   four lanes have to be checked rather than assumed: reading Field(v_array, 1..3) off a shorter
   array walks past the block, and a one-word array sitting at the top of the minor heap puts
   those reads on the PROT_NONE guard beyond young_end -- a SIGBUS, not a silently wrong lane. */
uint4x32_t ocaml_array_to_uint4x32(value v_array) {
    uint4x32_t result;
    if (Wosize_val(v_array) != 4)
    {
        caml_invalid_argument("uint4x32 argument must be an int array of length 4");
    }
    result.v[0] = (uint32_t)Long_val(Field(v_array, 0));
    result.v[1] = (uint32_t)Long_val(Field(v_array, 1));
    result.v[2] = (uint32_t)Long_val(Field(v_array, 2));
    result.v[3] = (uint32_t)Long_val(Field(v_array, 3));
    return result;
}

value uint4x32_to_ocaml_array(uint4x32_t x) {
    CAMLparam0();
    CAMLlocal1(result);
    result = caml_alloc(4, 0);
    Store_field(result, 0, Val_long(x.v[0]));
    Store_field(result, 1, Val_long(x.v[1]));
    Store_field(result, 2, Val_long(x.v[2]));
    Store_field(result, 3, Val_long(x.v[3]));
    CAMLreturn(result);
}

/* Threefry4x32 OCaml wrapper */
CAMLprim value arrayjit_threefry4x32_crypto_ocaml(value v_key, value v_counter)
{
  CAMLparam2(v_key, v_counter);
  uint4x32_t key = ocaml_array_to_uint4x32(v_key);
  uint4x32_t counter = ocaml_array_to_uint4x32(v_counter);
  uint4x32_t result = arrayjit_threefry4x32_crypto(key, counter);
  CAMLreturn(uint4x32_to_ocaml_array(result));
}

CAMLprim value arrayjit_threefry4x32_light_ocaml(value v_key, value v_counter)
{
  CAMLparam2(v_key, v_counter);
  uint4x32_t key = ocaml_array_to_uint4x32(v_key);
  uint4x32_t counter = ocaml_array_to_uint4x32(v_counter);
  uint4x32_t result = arrayjit_threefry4x32_light(key, counter);
  CAMLreturn(uint4x32_to_ocaml_array(result));
}

CAMLprim value arrayjit_threefry4x32_ocaml(value v_key, value v_counter)
{
  CAMLparam2(v_key, v_counter);
  uint4x32_t key = ocaml_array_to_uint4x32(v_key);
  uint4x32_t counter = ocaml_array_to_uint4x32(v_counter);
  uint4x32_t result = arrayjit_threefry4x32(key, counter);
  CAMLreturn(uint4x32_to_ocaml_array(result));
}

/* Conversion from uint4x32 to various types - OCaml wrappers */
CAMLprim value arrayjit_uint4x32_to_single_uniform_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint4x32_t x = ocaml_array_to_uint4x32(v_x);
  float result = uint4x32_to_single_uniform(x);
  CAMLreturn(caml_copy_double((double)result));
}

CAMLprim value arrayjit_uint4x32_to_double_uniform_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint4x32_t x = ocaml_array_to_uint4x32(v_x);
  double result = uint4x32_to_double_uniform(x);
  CAMLreturn(caml_copy_double(result));
}

CAMLprim value arrayjit_uint4x32_to_int32_uniform_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint4x32_t x = ocaml_array_to_uint4x32(v_x);
  int32_t result = uint4x32_to_int32_uniform(x);
  CAMLreturn(Val_long(result));
}

CAMLprim value arrayjit_uint4x32_to_int64_uniform_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint4x32_t x = ocaml_array_to_uint4x32(v_x);
  int64_t result = uint4x32_to_int64_uniform(x);
  CAMLreturn(caml_copy_int64(result));
}

CAMLprim value arrayjit_uint4x32_to_uint32_uniform_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint4x32_t x = ocaml_array_to_uint4x32(v_x);
  uint32_t result = uint4x32_to_uint32_uniform(x);
  CAMLreturn(Val_long(result));
}

CAMLprim value arrayjit_uint4x32_to_uint64_uniform_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint4x32_t x = ocaml_array_to_uint4x32(v_x);
  uint64_t result = uint4x32_to_uint64_uniform(x);
  CAMLreturn(caml_copy_int64((int64_t)result));
}

CAMLprim value arrayjit_uint4x32_to_byte_uniform_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint4x32_t x = ocaml_array_to_uint4x32(v_x);
  int8_t result = uint4x32_to_byte_uniform(x);
  CAMLreturn(Val_int(result));
}

CAMLprim value arrayjit_uint4x32_to_uint16_uniform_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint4x32_t x = ocaml_array_to_uint4x32(v_x);
  uint16_t result = uint4x32_to_uint16_uniform(x);
  CAMLreturn(Val_int(result));
}

CAMLprim value arrayjit_uint4x32_to_bfloat16_uniform_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint4x32_t x = ocaml_array_to_uint4x32(v_x);
  uint16_t result = uint4x32_to_bfloat16_uniform(x);
  CAMLreturn(Val_int(result));
}

CAMLprim value arrayjit_uint4x32_to_half_uniform_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint4x32_t x = ocaml_array_to_uint4x32(v_x);
  /* The shared C ABI returns HALF_T; OCaml exposes its uint16 bits. */
  uint16_t result = HALF_TO_UINT16(uint4x32_to_half_uniform(x));
  CAMLreturn(Val_int(result));
}

CAMLprim value arrayjit_uint4x32_to_fp8_uniform_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint4x32_t x = ocaml_array_to_uint4x32(v_x);
  uint8_t result = uint4x32_to_fp8_uniform(x);
  CAMLreturn(Val_int(result));
}

/* Conversion to uint4x32 from various types - OCaml wrappers */
CAMLprim value arrayjit_single_to_uint4x32_ocaml(value v_x)
{
  CAMLparam1(v_x);
  float x = (float)Double_val(v_x);
  uint4x32_t result = single_to_uint4x32(x);
  CAMLreturn(uint4x32_to_ocaml_array(result));
}

CAMLprim value arrayjit_double_to_uint4x32_ocaml(value v_x)
{
  CAMLparam1(v_x);
  double x = Double_val(v_x);
  uint4x32_t result = double_to_uint4x32(x);
  CAMLreturn(uint4x32_to_ocaml_array(result));
}

CAMLprim value arrayjit_int32_to_uint4x32_ocaml(value v_x)
{
  CAMLparam1(v_x);
  int32_t x = (int32_t)Long_val(v_x);
  uint4x32_t result = int32_to_uint4x32(x);
  CAMLreturn(uint4x32_to_ocaml_array(result));
}

CAMLprim value arrayjit_int64_to_uint4x32_ocaml(value v_x)
{
  CAMLparam1(v_x);
  int64_t x = Int64_val(v_x);
  uint4x32_t result = int64_to_uint4x32(x);
  CAMLreturn(uint4x32_to_ocaml_array(result));
}

CAMLprim value arrayjit_uint32_to_uint4x32_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint32_t x = (uint32_t)Long_val(v_x);
  uint4x32_t result = uint32_to_uint4x32(x);
  CAMLreturn(uint4x32_to_ocaml_array(result));
}

CAMLprim value arrayjit_uint64_to_uint4x32_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint64_t x = (uint64_t)Int64_val(v_x);
  uint4x32_t result = uint64_to_uint4x32(x);
  CAMLreturn(uint4x32_to_ocaml_array(result));
}

CAMLprim value arrayjit_byte_to_uint4x32_ocaml(value v_x)
{
  CAMLparam1(v_x);
  unsigned char x = (unsigned char)Int_val(v_x);
  uint4x32_t result = byte_to_uint4x32(x);
  CAMLreturn(uint4x32_to_ocaml_array(result));
}

CAMLprim value arrayjit_uint16_to_uint4x32_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint16_t x = (uint16_t)Int_val(v_x);
  uint4x32_t result = uint16_to_uint4x32(x);
  CAMLreturn(uint4x32_to_ocaml_array(result));
}

CAMLprim value arrayjit_bfloat16_to_uint4x32_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint16_t x = (uint16_t)Int_val(v_x);
  uint4x32_t result = bfloat16_to_uint4x32(x);
  CAMLreturn(uint4x32_to_ocaml_array(result));
}

CAMLprim value arrayjit_half_to_uint4x32_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint16_t x = (uint16_t)Int_val(v_x);
  uint4x32_t result = half_to_uint4x32(x);
  CAMLreturn(uint4x32_to_ocaml_array(result));
}

CAMLprim value arrayjit_fp8_to_uint4x32_ocaml(value v_x)
{
  CAMLparam1(v_x);
  uint8_t x = (uint8_t)Int_val(v_x);
  uint4x32_t result = fp8_to_uint4x32(x);
  CAMLreturn(uint4x32_to_ocaml_array(result));
}

/* BFloat16 to Float conversion (OCaml wrapper) */
CAMLprim value arrayjit_bfloat16_to_single(value v_bf16)
{
  CAMLparam1(v_bf16);
  uint16_t bf16 = (uint16_t)Int_val(v_bf16);
  float result = bfloat16_to_single(bf16);
  CAMLreturn(caml_copy_double((double)result));
}

/* Float to BFloat16 conversion (OCaml wrapper) */
CAMLprim value arrayjit_single_to_bfloat16(value v_float)
{
  CAMLparam1(v_float);
  float f = (float)Double_val(v_float);
  uint16_t bf16 = single_to_bfloat16(f);
  CAMLreturn(Val_int(bf16));
}

/* Half (Float16) to Float conversion (OCaml wrapper) */
CAMLprim value arrayjit_half_to_single(value v_half)
{
  CAMLparam1(v_half);
  uint16_t half = (uint16_t)Int_val(v_half);
  float result = half_to_single(half);
  CAMLreturn(caml_copy_double((double)result));
}

/* Float to Half (Float16) conversion (OCaml wrapper) */
CAMLprim value arrayjit_single_to_half(value v_float)
{
  CAMLparam1(v_float);
  float f = (float)Double_val(v_float);
  uint16_t half = single_to_half(f);
  CAMLreturn(Val_int(half));
}

/* FP8 E5M2 format to Float conversion (OCaml wrapper) */
CAMLprim value arrayjit_fp8_to_single(value v_fp8)
{
  CAMLparam1(v_fp8);
  uint8_t fp8 = (uint8_t)Int_val(v_fp8);
  float result = fp8_to_single(fp8);
  CAMLreturn(caml_copy_double((double)result));
}

/* Float to FP8 E5M2 conversion (OCaml wrapper) */
CAMLprim value arrayjit_single_to_fp8(value v_float)
{
  CAMLparam1(v_float);
  float f = (float)Double_val(v_float);
  uint8_t fp8 = single_to_fp8(f);
  CAMLreturn(Val_int(fp8));
}

/* An OCaml float IS a double, so the host side narrows from f64 — through the one-step codec, or
   it would double-round where the GPU backends do not (gh-ocannl-648). */
CAMLprim value arrayjit_double_to_fp8(value v_float)
{
  CAMLparam1(v_float);
  double f = Double_val(v_float);
  uint8_t fp8 = double_to_fp8(f);
  CAMLreturn(Val_int(fp8));
}

// TODO: a more efficient approach would involve computing strides once and using memcpy
// for contiguous inner slices, but that adds complexity.
CAMLprim value arrayjit_copy_with_padding(value v_source, value v_target, value v_padding)
{
  CAMLparam3(v_source, v_target, v_padding);

  struct caml_ba_array *source_ba = Caml_ba_array_val(v_source);
  struct caml_ba_array *target_ba = Caml_ba_array_val(v_target);
  int ndim = source_ba->num_dims;

  if (ndim != target_ba->num_dims)
  {
    caml_failwith("Source and target must have the same number of dimensions");
  }

  if (ndim == 0)
  {
    CAMLreturn(Val_unit);
  }

  void *source_data = Caml_ba_data_val(v_source);
  void *target_data = Caml_ba_data_val(v_target);

  if ((source_ba->flags & CAML_BA_KIND_MASK) != (target_ba->flags & CAML_BA_KIND_MASK))
  {
    caml_failwith("Source and target must have the same element kind");
  }
  /* Per-element byte size: caml_ba_byte_size is the WHOLE array's size, so divide by the element
     count. */
  uintnat source_elems = 1;
  for (int d = 0; d < ndim; d++)
  {
    source_elems *= source_ba->dim[d];
  }
  if (source_elems == 0)
  {
    CAMLreturn(Val_unit);
  }
  size_t elem_size = caml_ba_byte_size(source_ba) / source_elems;

  // Use source dimensions directly from bigarray
  intnat *source_shape = source_ba->dim;

  // Extract paddings
  if (Wosize_val(v_padding) != (uintnat)ndim)
  {
    caml_failwith("Padding array length mismatch");
  }
  intnat *left = malloc(ndim * sizeof(intnat));
  if (left == NULL)
  {
    caml_failwith("Malloc failed");
  }
  intnat *right = malloc(ndim * sizeof(intnat));
  if (right == NULL)
  {
    free(left);
    caml_failwith("Malloc failed");
  }
  for (int d = 0; d < ndim; d++)
  {
    value pad = Field(v_padding, d);
    left[d] = Long_val(Field(pad, 0));
    right[d] = Long_val(Field(pad, 1));
    if (left[d] < 0 || right[d] < 0)
    {
      free(left);
      free(right);
      caml_failwith("Negative padding");
    }
  }

  // Verify target dimensions match source + padding
  for (int d = 0; d < ndim; d++)
  {
    if (target_ba->dim[d] != source_shape[d] + left[d] + right[d])
    {
      free(left);
      free(right);
      caml_failwith("Target dimensions do not match source + padding");
    }
  }

  // Multi-dimensional index loop
  intnat *indices = calloc(ndim, sizeof(intnat));
  if (indices == NULL)
  {
    free(left);
    free(right);
    caml_failwith("Calloc failed");
  }

  while (1)
  {
    // Compute source flat offset
    intnat source_offset = 0;
    intnat s_stride = 1;
    for (int d = ndim - 1; d >= 0; d--)
    {
      source_offset += indices[d] * s_stride;
      s_stride *= source_shape[d];
    }

    // Compute target flat offset with padding offset
    intnat target_offset = 0;
    intnat t_stride = 1;
    for (int d = ndim - 1; d >= 0; d--)
    {
      target_offset += (indices[d] + left[d]) * t_stride;
      t_stride *= target_ba->dim[d];
    }

    // Copy the element
    memcpy((char *)target_data + target_offset * elem_size,
           (char *)source_data + source_offset * elem_size,
           elem_size);

    // Increment indices (odometer-style)
    int carry = 1;
    for (int d = ndim - 1; d >= 0; d--)
    {
      if (carry == 0)
        break;
      indices[d] += carry;
      if (indices[d] < source_shape[d])
      {
        carry = 0;
      }
      else
      {
        indices[d] = 0;
        carry = 1;
      }
    }
    if (carry == 1)
      break; // Done
  }

  free(indices);
  free(left);
  free(right);

  CAMLreturn(Val_unit);
}