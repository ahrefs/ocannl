/* A foreign C library's diagnostic, reduced to its essentials: one line written to the C
   runtime's [stderr] and flushed (gh-ocannl-1031). It stands in for ROCr's
   `Signal 0x... time stamps may be invalid.`, which is a plain `fprintf(stderr, ...)` from inside
   libhsa-runtime64 -- a stream the OCaml program never names.

   The second stub is the control the claims are gated on. It writes the same way but to the
   descriptor NUMBER 2 rather than through the C `stderr` stream, so it answers, on the host
   actually running, the question no `#ifdef` here should be answering: does redirecting fd 2 from
   OCaml reach a C-level write at all? Where it does not, a claim about what a capture of fd 2
   caught from C is vacuous rather than false, and the test says so instead of failing. */

#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <stdio.h>
#include <string.h>

#if defined(_WIN32)
#include <io.h>
#define ocannl_write _write
#else
#include <unistd.h>
#define ocannl_write write
#endif

CAMLprim value ocannl_test_write_c_stderr(value line) {
  CAMLparam1(line);
  fprintf(stderr, "%s\n", String_val(line));
  fflush(stderr);
  CAMLreturn(Val_unit);
}

CAMLprim value ocannl_test_write_fd2(value line) {
  CAMLparam1(line);
  {
    /* Unbuffered and straight at the descriptor: nothing between this and fd 2 that could carry
       the line somewhere else and make the control lie about the host. */
    const char *s = String_val(line);
    (void)ocannl_write(2, s, (unsigned int)strlen(s));
    (void)ocannl_write(2, "\n", 1);
  }
  CAMLreturn(Val_unit);
}
