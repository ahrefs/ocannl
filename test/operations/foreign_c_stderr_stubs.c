/* A foreign C library's diagnostic, reduced to its essentials: one line written to the C
   runtime's [stderr] and flushed (gh-ocannl-1031). It stands in for ROCr's
   `Signal 0x... time stamps may be invalid.`, which is a plain `fprintf(stderr, ...)` from inside
   libhsa-runtime64 — a stream the OCaml program never names. */

#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <stdio.h>

CAMLprim value ocannl_test_write_c_stderr(value line) {
  CAMLparam1(line);
  fprintf(stderr, "%s\n", String_val(line));
  fflush(stderr);
  CAMLreturn(Val_unit);
}
