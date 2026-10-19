/* gh-ocannl-1281: a SIGALRM raised by a thread of the process itself -- [raise], which is
   thread-directed and which Linux reports as si_code SI_TKILL rather than the SI_USER of a [kill].
   OCaml has no binding for it. A no-op on Windows, where the test that calls it does not run. */

#include <caml/mlvalues.h>
#include <signal.h>

CAMLprim value ocannl_test_raise_sigalrm(value unit) {
  (void)unit;
#if !defined(_WIN32)
  raise(SIGALRM);
#endif
  return Val_unit;
}
