/* A foreign C library's write to the C runtime's [stderr], reduced to its essentials
   (gh-ocannl-1370): the bytes as given -- no newline added, so a line can be split across calls the
   way a library's piecewise [fprintf]s split it -- then a flush. It stands in for ROCr's
   `Signal 0x... time stamps may be invalid.`, a plain `fprintf(stderr, ...)` inside
   libhsa-runtime64. */

#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <stdio.h>

CAMLprim value ocannl_test_noise_c_stderr(value s) {
  CAMLparam1(s);
  fputs(String_val(s), stderr);
  fflush(stderr);
  CAMLreturn(Val_unit);
}
