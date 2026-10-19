/* gh-ocannl-1281: a SIGALRM raised by a thread of the process itself -- [raise], which is
   thread-directed and which Linux reports as si_code SI_TKILL rather than the SI_USER of a [kill].
   OCaml has no binding for it. A no-op on Windows, where the test that calls it does not run. */

#include <caml/mlvalues.h>
#include <signal.h>
#include <string.h>

CAMLprim value ocannl_test_raise_sigalrm(value unit) {
  (void)unit;
#if !defined(_WIN32)
  raise(SIGALRM);
#endif
  return Val_unit;
}

#if !defined(_WIN32)
static void ocannl_test_siginfo_handler(int signo, siginfo_t *info, void *context) {
  (void)signo;
  (void)info;
  (void)context;
}
#endif

/* Catches SIGALRM with an [SA_SIGINFO] handler, as GitHub's macOS runner agent does: an [exec] after
   it resets the handler to [SIG_DFL], and on macOS leaves [SA_SIGINFO] in the flags the new image
   reads (Linux clears them). */
CAMLprim value ocannl_test_catch_sigalrm_siginfo(value unit) {
  (void)unit;
#if !defined(_WIN32)
  struct sigaction sa;
  memset(&sa, 0, sizeof sa);
  sa.sa_sigaction = ocannl_test_siginfo_handler;
  sa.sa_flags = SA_SIGINFO;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGALRM, &sa, NULL);
#endif
  return Val_unit;
}

/* Whether SIGALRM reads as [SIG_DFL] with [SA_SIGINFO] set: the state the case above leaves an
   [exec]ed image in on macOS. */
CAMLprim value ocannl_test_sigalrm_default_with_siginfo(value unit) {
  (void)unit;
#if !defined(_WIN32)
  struct sigaction cur;
  if (sigaction(SIGALRM, NULL, &cur) == 0 && cur.sa_handler == SIG_DFL &&
      (cur.sa_flags & SA_SIGINFO))
    return Val_true;
#endif
  return Val_false;
}
