#if defined(__linux__) && !defined(_GNU_SOURCE)
/* sched_getaffinity / sched_setaffinity / CPU_* macros. */
#define _GNU_SOURCE
#endif

#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#if !defined(_WIN32)
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#endif

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(__linux__)
#include <sched.h>
#endif

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#endif

CAMLprim value ocannl_flush_c_streams(value unit) {
  (void)unit;
  fflush(NULL);
  return Val_unit;
}

/* The operating system's id of this process (gh-ocannl-1247). Not [Unix.getpid]: on Windows that
   answers a handle to the current process, duplicated at startup, which is local to the process
   and need not differ between two of them. */
CAMLprim value ocannl_os_process_id(value unit) {
  (void)unit;
#if defined(_WIN32)
  return Val_long((long)GetCurrentProcessId());
#else
  return Val_long((long)getpid());
#endif
}

/* The OS id of a process holding a lock on [v_fd]'s file that would block an exclusive one
   (gh-ocannl-1462): 0 when none does, -1 when this cannot tell. [fcntl(F_GETLK)] reports one
   conflicting lock, so with several shared holders it names one of them. OCaml's [Unix.lockf] is
   built on the same fcntl record locks on POSIX. Windows has no such query. */
CAMLprim value ocannl_lock_holder_pid(value v_fd) {
#if defined(_WIN32)
  (void)v_fd;
  return Val_long(-1);
#else
  struct flock fl;
  memset(&fl, 0, sizeof fl);
  fl.l_type = F_WRLCK;
  fl.l_whence = SEEK_SET;
  fl.l_start = 0;
  fl.l_len = 0;
  if (fcntl(Int_val(v_fd), F_GETLK, &fl) == -1) return Val_long(-1);
  if (fl.l_type == F_UNLCK) return Val_long(0);
  return Val_long((long)fl.l_pid);
#endif
}

/* Whether a process with the OS id [pid] is running now. A process that exists but that this one
   may not signal or open still counts as running: the answer protects its files from deletion. */
CAMLprim value ocannl_process_alive(value v_pid) {
#if defined(_WIN32)
  {
    DWORD pid = (DWORD)Long_val(v_pid);
    HANDLE h;
    DWORD wait;
    if (Long_val(v_pid) <= 0) return Val_false;
    h = OpenProcess(SYNCHRONIZE, FALSE, pid);
    if (h == NULL) return Val_bool(GetLastError() == ERROR_ACCESS_DENIED);
    wait = WaitForSingleObject(h, 0);
    CloseHandle(h);
    return Val_bool(wait == WAIT_TIMEOUT);
  }
#else
  {
    long pid = Long_val(v_pid);
    if (pid <= 0 || pid != (long)(pid_t)pid) return Val_false;
    if (kill((pid_t)pid, 0) == 0) return Val_true;
    return Val_bool(errno == EPERM);
  }
#endif
}

/* Where the C runtime's [stderr] is a modifiable lvalue, so the two stubs below can replace the
   stream (glibc, Darwin). One spelling for both, and published through
   [ocannl_c_stderr_replaceable] so a test can gate on the very predicate the stubs use rather than
   on their own report of success, which a regression could turn into a silent skip. */
#if defined(__GLIBC__) || defined(__APPLE__)
#define OCANNL_C_STDERR_REPLACEABLE 1
#endif

CAMLprim value ocannl_c_stderr_replaceable(value unit) {
  (void)unit;
#ifdef OCANNL_C_STDERR_REPLACEABLE
  return Val_true;
#else
  return Val_false;
#endif
}

/* Give the C runtime's [stderr] its own file descriptor, duplicated from the process's real
   stderr at startup, so that a later redirection of fd 2 does not capture what foreign C
   libraries print (gh-ocannl-1031).

   The motivating writer is the ROCm stack: Ubuntu 26.04 ships libhsa-runtime64 7.1.0 built with
   assertions on, so ROCr's internal `debug_print` is live and every `hipEventRecord` costs one
   `Signal 0x... time stamps may be invalid.` on stderr -- ROCclr's `hip::EventMarker` profiles
   every event marker whatever `hipEventDisableTiming` says, and ROCr then reads back timestamps
   the hardware never wrote for a barrier packet. Nothing OCANNL does avoids it and no runtime
   switch turns it off.

   That is only chatter, and the project's contract already puts chatter on stderr and keeps
   stdout as the data channel -- but ppx_expect redirects BOTH fd 1 and fd 2 into the file it
   diffs against the `%expect` block, so foreign chatter lands in a golden. Moving the C stream
   to its own descriptor restores the contract for it: the chatter still reaches the real stderr
   and the run log, nothing is dropped or rewritten anywhere, and OCaml's own [Stdlib.stderr]
   (fd 2) is untouched, so the library's diagnostics keep behaving exactly as before.

   Unbuffered like the stream it replaces: a fatal driver message that precedes an abort must
   already be out when the process dies. The replaced FILE* is deliberately not closed -- that
   would close fd 2. Only where [stderr] is a modifiable lvalue (glibc, Darwin); elsewhere the
   stub reports that it did nothing. */
CAMLprim value ocannl_detach_c_stderr(value unit) {
  (void)unit;
#ifdef OCANNL_C_STDERR_REPLACEABLE
  {
    static int detached = 0;
    int fd;
    FILE *replacement;
    if (detached) return Val_true;
    /* Close-on-exec: this descriptor is the PARENT's stderr, and a child that inherited it would
       hold the write end of the parent's stderr pipe open -- a log collector waiting for EOF then
       waits for a descriptor nobody writes to, and a caller redirecting the child's streams has
       lost the isolation it asked for. F_DUPFD_CLOEXEC sets it in the dup itself, leaving no
       window where a concurrent fork/exec inherits it; where the constant is missing the two-step
       fallback still closes the leak for every exec but a racing one. */
#ifdef F_DUPFD_CLOEXEC
    fd = fcntl(fileno(stderr), F_DUPFD_CLOEXEC, 3);
#else
    fd = -1;
#endif
    if (fd == -1) {
      fd = dup(fileno(stderr));
      if (fd != -1) (void)fcntl(fd, F_SETFD, FD_CLOEXEC);
    }
    if (fd == -1) return Val_false;
    replacement = fdopen(fd, "w");
    if (replacement == NULL) {
      close(fd);
      return Val_false;
    }
    setvbuf(replacement, NULL, _IONBF, 0);
    stderr = replacement;
    detached = 1;
    return Val_true;
  }
#else
  return Val_false;
#endif
}

/* Collapse the ROCm runtime's per-event `Signal 0x... time stamps may be invalid.` lines on the C
   runtime's [stderr] into a count (gh-ocannl-1370).

   [ocannl_detach_c_stderr] keeps that chatter out of a ppx_expect golden, but it still reaches the
   process's real stderr -- 268,172 lines for one autotuning test on an assertions-enabled ROCr
   (gfx1151; a discrete card writes valid timestamps and prints none) -- and Dune truncates a long
   action output from the middle, so the test's own diagnostics (SKIPPED lines, per-claim output,
   a `cold search` line) were the part that went missing. No switch reaches the print (see
   [ocannl_detach_c_stderr]), so the C stream itself is replaced by one that writes through to the
   same descriptor and recognizes exactly that line:

   - the FIRST such line is written verbatim, followed by one note saying that later ones are
     counted, so the message and the fact of collapsing are both on the log even if the process
     dies before it exits;
   - every later one is counted, and an exit-time destructor reports the count in one line;
   - every other byte is written through at once, in order, unbuffered like the stream it
     replaces. Only a partial line that is still a prefix of the recognized one is held back, and
     only until its newline (or the exit summary) decides it.

   The replacement wraps whatever descriptor the C stream wrote to: the detached one, or fd 2 when
   [detach_c_stderr] is off -- in which case it keeps following fd 2, as that setting promises.
   Its FILE has no descriptor of its own, so [fileno(stderr)] reads -1 afterwards; nothing in the
   ROCm stack writes to [fileno(stderr)] directly, and the stream is installed only by the HIP
   backend, on opening a device (see [Utils.c_stderr_noise_collapsed]). A cookie stream cannot be
   wide-oriented, so a C stream already wide-oriented is not replaced; an [fclose(stderr)] reports
   the count at the close. Writes through the FILE are serialized by its
   lock, which is what makes the filter state safe from the runtime's own threads. */

#ifdef OCANNL_C_STDERR_REPLACEABLE
#define OCANNL_NOISE_HOLD 256

static const char ocannl_noise_head[] = "Signal 0x";
static const char ocannl_noise_tail[] = " time stamps may be invalid.\n";

static struct {
  int installed;
  int fd;
  int passthrough;  /* the current line is known not to be the recognized one */
  int summarized;   /* the exit summary is out: later lines are written through, uncounted */
  size_t held_len;
  char held[OCANNL_NOISE_HOLD];
  unsigned long seen;
  FILE *stream;
} ocannl_noise;

enum { NOISE_NO, NOISE_PREFIX, NOISE_FULL };

/* Whether [s] (a line start, [n] bytes, no newline before its last byte) is the recognized line,
   a prefix of it, or neither. */
static int ocannl_noise_match(const char *s, size_t n) {
  size_t head = sizeof ocannl_noise_head - 1, tail = sizeof ocannl_noise_tail - 1;
  size_t i = 0, j = 0, hex;
  for (; i < n && i < head; i++)
    if (s[i] != ocannl_noise_head[i]) return NOISE_NO;
  if (i < head) return NOISE_PREFIX;
  hex = i;
  while (i < n && ((s[i] >= '0' && s[i] <= '9') || (s[i] >= 'a' && s[i] <= 'f') ||
                   (s[i] >= 'A' && s[i] <= 'F')))
    i++;
  if (i == n) return NOISE_PREFIX;
  if (i == hex) return NOISE_NO;
  for (; i < n && j < tail; i++, j++)
    if (s[i] != ocannl_noise_tail[j]) return NOISE_NO;
  if (j < tail) return NOISE_PREFIX;
  return i == n ? NOISE_FULL : NOISE_NO;
}

static void ocannl_noise_write_all(const char *p, size_t n) {
  while (n > 0) {
    ssize_t w = write(ocannl_noise.fd, p, n);
    if (w < 0) {
      if (errno == EINTR) continue;
      return; /* Like the stream it replaces: a write error loses the bytes, not the process. */
    }
    p += w;
    n -= (size_t)w;
  }
}

static void ocannl_noise_puts(const char *s) { ocannl_noise_write_all(s, strlen(s)); }

static void ocannl_noise_filter(const char *buf, size_t n) {
  size_t i = 0;
  while (i < n) {
    if (ocannl_noise.passthrough) {
      const char *nl = memchr(buf + i, '\n', n - i);
      size_t end = nl ? (size_t)(nl - buf) + 1 : n;
      ocannl_noise_write_all(buf + i, end - i);
      if (nl) ocannl_noise.passthrough = 0;
      i = end;
    } else {
      char c = buf[i++];
      int m;
      ocannl_noise.held[ocannl_noise.held_len++] = c;
      m = ocannl_noise_match(ocannl_noise.held, ocannl_noise.held_len);
      if (m == NOISE_FULL && !ocannl_noise.summarized) {
        if (ocannl_noise.seen++ == 0) {
          ocannl_noise_write_all(ocannl_noise.held, ocannl_noise.held_len);
          ocannl_noise_puts(
              "OCANNL: further `Signal 0x... time stamps may be invalid.` lines from the ROCm "
              "runtime are counted, not printed; the count is reported at exit "
              "(collapse_foreign_stderr_noise=false prints each, gh-ocannl-1370)\n");
        }
        ocannl_noise.held_len = 0;
      } else if (m != NOISE_PREFIX || ocannl_noise.held_len == OCANNL_NOISE_HOLD) {
        ocannl_noise_write_all(ocannl_noise.held, ocannl_noise.held_len);
        ocannl_noise.held_len = 0;
        ocannl_noise.passthrough = (c != '\n');
      }
    }
  }
}

/* Write out a held prefix and the count, and stop collapsing: anything written afterwards passes
   through, uncounted. The caller holds the stream's lock. */
static void ocannl_noise_finish(void) {
  char line[256];
  if (ocannl_noise.held_len > 0) {
    ocannl_noise_write_all(ocannl_noise.held, ocannl_noise.held_len);
    ocannl_noise.held_len = 0;
    ocannl_noise.passthrough = 1;
  }
  if (ocannl_noise.seen > 1 && !ocannl_noise.summarized) {
    snprintf(line, sizeof line,
             "OCANNL: collapsed %lu further `Signal 0x... time stamps may be invalid.` lines from "
             "the ROCm runtime (gh-ocannl-1370)\n",
             ocannl_noise.seen - 1);
    ocannl_noise_puts(line);
  }
  ocannl_noise.summarized = 1;
}

/* After every [atexit] handler -- among them OCANNL's own exit teardown of the HIP streams, which
   is when ROCclr prints most of these -- since glibc runs the executable's destructors from
   [_dl_fini], the earliest-registered exit handler. The shared libraries' own destructors run after
   the executable's, so a line one of them prints is written through, uncounted, rather than lost.
   A stream some code has [fclose]d is not touched: its close callback already summarized and
   cleared [installed], and its FILE is gone. */
__attribute__((destructor)) static void ocannl_noise_summary(void) {
  if (!ocannl_noise.installed) return;
  flockfile(ocannl_noise.stream);
  ocannl_noise_finish();
  funlockfile(ocannl_noise.stream);
}

/* [fclose(stderr)] is valid foreign code (daemonizing, silencing C diagnostics). The descriptor is
   not closed -- the replaced FILE owns it -- but the count is reported now, since the exit hook
   must not reach the freed FILE. Called by the C library with the stream's lock held. */
static int ocannl_noise_cookie_close(void *cookie) {
  (void)cookie;
  ocannl_noise_finish();
  ocannl_noise.installed = 0;
  ocannl_noise.stream = NULL;
  return 0;
}

#if defined(__GLIBC__)
static ssize_t ocannl_noise_cookie_write(void *cookie, const char *buf, size_t n) {
  (void)cookie;
  ocannl_noise_filter(buf, n);
  return (ssize_t)n;
}
#else
static int ocannl_noise_cookie_write(void *cookie, const char *buf, int n) {
  (void)cookie;
  if (n > 0) ocannl_noise_filter(buf, (size_t)n);
  return n;
}
#endif
#endif

CAMLprim value ocannl_collapse_c_stderr_noise(value unit) {
  (void)unit;
#ifdef OCANNL_C_STDERR_REPLACEABLE
  {
    FILE *replacement;
    int fd;
    if (ocannl_noise.installed) return Val_true;
    /* A cookie stream is byte-oriented only (glibc gives it no wide-character buffer), so a C
       stream some code has already made wide-oriented is left alone rather than replaced by one
       that would make its next [fwprintf] fail. An unoriented stream is the normal case: the first
       write orients it, and ROCr's own [fprintf] makes it byte-oriented either way. */
    if (fwide(stderr, 0) > 0) return Val_false;
    fflush(stderr);
    fd = fileno(stderr);
    if (fd < 0) return Val_false;
#if defined(__GLIBC__)
    {
      cookie_io_functions_t io = {NULL, ocannl_noise_cookie_write, NULL, ocannl_noise_cookie_close};
      replacement = fopencookie(NULL, "w", io);
    }
#else
    replacement = funopen(NULL, NULL, ocannl_noise_cookie_write, NULL, ocannl_noise_cookie_close);
#endif
    if (replacement == NULL) return Val_false;
    setvbuf(replacement, NULL, _IONBF, 0);
    ocannl_noise.fd = fd;
    ocannl_noise.stream = replacement;
    ocannl_noise.installed = 1;
    /* The replaced FILE stays open: it owns [fd], which the replacement now writes to. */
    stderr = replacement;
    return Val_true;
  }
#else
  return Val_false;
#endif
}

/* --- A deadline on GPU device initialization (gh-ocannl-1281) ---

   A faulted GPU leaves its driver wedged: every later process that touches the runtime spins in
   device initialization and never returns (gh-ocannl-1211's tuf fault). Nothing on the stuck
   thread can bound that -- it is inside a C runtime call and reaches no OCaml poll point -- so the
   bound is a WATCHDOG THREAD of this library's own. [arm] gives it a due time on the monotonic
   clock; [disarm] takes it back. When a due time passes while still armed, the watchdog writes the
   diagnostic prepared before arming to fd 2 and kills the process by SIGALRM (status 142 in a
   shell), so harnesses can classify it.

   Being a thread rather than a signal handler is what keeps the host program's own signal state
   out of it: the real-time timer is never touched (an [alarm] or [setitimer] the program set, or
   sets meanwhile, fires exactly when it would have), no handler is installed, and no signal mask
   is changed outside the watchdog itself. A SIGALRM anything else sends during a region meets the
   program's own disposition, as it always would. The line is written only by the watchdog and
   only after its own due time passed while armed, so it always means the deadline. [arm] refuses
   when SIGALRM does not have its default disposition -- a program with its own handler, or an
   ignore, gets no deadline, since killing by SIGALRM would have to override it -- and on Windows.
   One region is armed at a time: an [arm] while another thread's region is armed WAITS for it
   (outside the OCaml runtime lock), so it is bounded by that region's deadline and then by its
   own; an [arm] from inside the thread's own armed region returns false, and the outer deadline
   covers the nested region. The serialization lives here rather than in an OCaml lock so that a
   fork child, whose other threads did not come along, can reset all of it (an OCaml mutex held by
   a vanished thread would stay locked). Once the watchdog has committed to the death, no [arm]
   proceeds: the diagnostic it is about to write is never overwritten.

   Each arm is a GENERATION, and [disarm] takes back only the one it names (gh-ocannl-1446): a
   region armed before a [fork] is still on the child's stack, and its disarm must not take back a
   deadline a new domain of the child armed meanwhile. The fork reset keeps the counter, so no
   generation the child arms equals one it inherited.

   Boundary: whether SIGALRM has its default disposition is read when a region arms (and again
   after waiting for another region), not afterwards. A handler a thread installs while the region
   runs is overridden only if the region outlives its bound: the deadline is then a death, not a
   signal delivery, and it puts the default disposition back to die by it. Until then the handler
   is untouched.

   The death never waits on the diagnostic for more than 2 s: the write runs on a thread of its
   own, so a blocking fd 2 that is full, or that another writer fills meanwhile, cannot hold it.
   The watchdog then puts
   SIGALRM's default disposition back, unblocks SIGALRM in its own thread (it blocks every signal
   otherwise, so it never takes the program's) and raises it there. */

#if !defined(_WIN32)
#include <pthread.h>
#include <time.h>

#define OCANNL_DEADLINE_MSG_MAX 1024
static pthread_mutex_t ocannl_deadline_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t ocannl_deadline_cond = PTHREAD_COND_INITIALIZER; /* wakes the watchdog */
static pthread_cond_t ocannl_deadline_free = PTHREAD_COND_INITIALIZER; /* wakes waiting [arm]s */
static int ocannl_deadline_thread_started; /* under the lock */
static int ocannl_deadline_writer_started; /* under the lock */
static int ocannl_deadline_armed;          /* under the lock */
static int ocannl_deadline_dying;          /* under the lock: the watchdog committed to the death */
static long ocannl_deadline_generation;    /* under the lock: the latest arm's, positive once armed */
static pthread_t ocannl_deadline_owner;    /* under the lock: the armed region's thread */
static int64_t ocannl_deadline_due_ns;     /* under the lock; monotonic */
static char ocannl_deadline_msg[OCANNL_DEADLINE_MSG_MAX];
static size_t ocannl_deadline_msg_len;
static pthread_mutex_t ocannl_deadline_write_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t ocannl_deadline_write_go = PTHREAD_COND_INITIALIZER;
static int ocannl_deadline_write_requested; /* under the write lock */
static int ocannl_deadline_written;         /* under the write lock */

static int64_t ocannl_monotonic_ns(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return -1;
  return (int64_t)ts.tv_sec * 1000000000 + (int64_t)ts.tv_nsec;
}

/* The process this one forked off is not this one: its watchdog thread did not come along. */
static void ocannl_deadline_atfork_child(void) {
  pthread_mutex_t fresh_lock = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t fresh_cond = PTHREAD_COND_INITIALIZER;
  pthread_cond_t fresh_free = PTHREAD_COND_INITIALIZER;
  ocannl_deadline_lock = fresh_lock;
  ocannl_deadline_cond = fresh_cond;
  ocannl_deadline_free = fresh_free;
  ocannl_deadline_thread_started = 0;
  ocannl_deadline_writer_started = 0;
  ocannl_deadline_armed = 0;
  ocannl_deadline_dying = 0;
  /* [ocannl_deadline_generation] is kept: a region the fork copied disarms by its own generation,
     which no arm in this process will reuse. */
  {
    pthread_mutex_t fresh_write_lock = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t fresh_write_go = PTHREAD_COND_INITIALIZER;
    ocannl_deadline_write_lock = fresh_write_lock;
    ocannl_deadline_write_go = fresh_write_go;
    ocannl_deadline_write_requested = 0;
    ocannl_deadline_written = 0;
  }
}

/* The diagnostic's write runs on a thread of its own, so the death never waits on it: a blocking
   fd 2 that is full, or that another writer fills between a readiness check and the write, can
   hold that thread forever, and the process dies regardless. The writer is started together with
   the watchdog and waits for the death, so the death needs no thread creation, which could fail
   under the very resource pressure that comes with a wedged process. */

static void *ocannl_deadline_writer(void *arg) {
  const char *p = ocannl_deadline_msg;
  size_t n;
  (void)arg;
  pthread_mutex_lock(&ocannl_deadline_write_lock);
  while (!ocannl_deadline_write_requested)
    pthread_cond_wait(&ocannl_deadline_write_go, &ocannl_deadline_write_lock);
  pthread_mutex_unlock(&ocannl_deadline_write_lock);
  /* The diagnostic is final: the watchdog committed to the death before requesting the write. */
  n = ocannl_deadline_msg_len;
  while (n > 0) {
    ssize_t w = write(2, p, n);
    if (w < 0) {
      if (errno == EINTR) continue;
      break;
    }
    p += w;
    n -= (size_t)w;
  }
  pthread_mutex_lock(&ocannl_deadline_write_lock);
  ocannl_deadline_written = 1;
  pthread_mutex_unlock(&ocannl_deadline_write_lock);
  return NULL;
}

static void ocannl_deadline_die(void) {
  struct sigaction dfl;
  sigset_t alrm;
  struct timespec nap = {0, 10000000};
  int i, done = 0;
  int64_t until = ocannl_monotonic_ns();
  if (until >= 0) until += 2000000000;
  pthread_mutex_lock(&ocannl_deadline_write_lock);
  ocannl_deadline_write_requested = 1;
  pthread_cond_signal(&ocannl_deadline_write_go);
  pthread_mutex_unlock(&ocannl_deadline_write_lock);
  /* At most 2 s, measured on the monotonic clock, in relative 10 ms naps: no wall-clock wait. A nap
     may oversleep by far (timer slack on a loaded macOS VM stretched 200 counted naps to 10 s), so
     the naps are not what is counted; the count bounds the wait only if the clock cannot be read. */
  for (i = 0; !done; i++) {
    pthread_mutex_lock(&ocannl_deadline_write_lock);
    done = ocannl_deadline_written;
    pthread_mutex_unlock(&ocannl_deadline_write_lock);
    if (done) break;
    if (until >= 0 ? ocannl_monotonic_ns() >= until : i >= 200) break;
    nanosleep(&nap, NULL);
  }
  memset(&dfl, 0, sizeof dfl);
  dfl.sa_handler = SIG_DFL;
  sigemptyset(&dfl.sa_mask);
  sigaction(SIGALRM, &dfl, NULL);
  sigemptyset(&alrm);
  sigaddset(&alrm, SIGALRM);
  pthread_sigmask(SIG_UNBLOCK, &alrm, NULL);
  raise(SIGALRM);
  /* Not reached: the default action of SIGALRM terminates the process. */
  _exit(128 + SIGALRM);
}

static void *ocannl_deadline_watchdog(void *arg) {
  (void)arg;
  pthread_mutex_lock(&ocannl_deadline_lock);
  for (;;) {
    if (!ocannl_deadline_armed) {
      pthread_cond_wait(&ocannl_deadline_cond, &ocannl_deadline_lock);
      continue;
    }
    {
      int64_t now = ocannl_monotonic_ns();
      int64_t left = ocannl_deadline_due_ns - now;
      struct timespec nap;
      int64_t step;
      if (now >= 0 && left <= 0) {
        /* Still armed past the due time: the deadline. Committed under the lock, so no [arm]
           can proceed, nor rewrite the diagnostic, from here on. */
        ocannl_deadline_dying = 1;
        break;
      }
      /* Sleeps relative steps of at most 100 ms with the lock released, re-reading the monotonic
         clock after each: no wait anywhere is pinned to the wall clock, so stepping it moves no
         deadline. An arm or disarm meanwhile is seen at the next step. */
      step = (now < 0 || left > 100000000) ? 100000000 : left;
      nap.tv_sec = 0;
      nap.tv_nsec = (long)step;
      pthread_mutex_unlock(&ocannl_deadline_lock);
      nanosleep(&nap, NULL);
      pthread_mutex_lock(&ocannl_deadline_lock);
    }
  }
  pthread_mutex_unlock(&ocannl_deadline_lock);
  ocannl_deadline_die();
  return NULL;
}

/* Registered when the library is loaded, before the deadline's lock can ever be taken: a fork
   racing the first arm must already run the child reset. */
static int ocannl_deadline_atfork_registered;
__attribute__((constructor)) static void ocannl_deadline_register_atfork(void) {
  ocannl_deadline_atfork_registered =
      pthread_atfork(NULL, NULL, ocannl_deadline_atfork_child) == 0;
}

/* Whether SIGALRM has its default disposition. The handler slot alone decides it: [sa_handler] and
   [sa_sigaction] share it, so an [SA_SIGINFO] handler is never [SIG_DFL]. [sa_flags] must not be
   consulted: macOS keeps a caught signal's [SA_SIGINFO] across [exec], which resets the handler
   to [SIG_DFL], so every descendant of a process that once caught SIGALRM that way -- GitHub's
   macOS runner agent is one -- reads [SIG_DFL] with [SA_SIGINFO] set, and a flag check refused
   the deadline in all of them. */
static int ocannl_sigalrm_is_default(void) {
  struct sigaction cur;
  return sigaction(SIGALRM, NULL, &cur) == 0 && cur.sa_handler == SIG_DFL;
}

/* Under the lock. */
static int ocannl_deadline_start_thread(void) {
  pthread_t thread;
  pthread_attr_t attr;
  sigset_t all, saved;
  int rc;
  if (ocannl_deadline_thread_started) return 1;
  if (!ocannl_deadline_atfork_registered) return 0;
  /* Created with every signal blocked, which it inherits: it must never take the program's. */
  sigfillset(&all);
  pthread_sigmask(SIG_SETMASK, &all, &saved);
  /* Neither call can fail on glibc or macOS; a failure elsewhere is "could not start", never an
     uninitialized attribute handed to [pthread_create]. */
  if (pthread_attr_init(&attr) != 0) {
    pthread_sigmask(SIG_SETMASK, &saved, NULL);
    return 0;
  }
  rc = pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
  /* The writer first: a watchdog without it would have nothing to write the diagnostic with. A
     writer whose watchdog could not be created is kept, and reused by the retry. */
  if (rc == 0 && !ocannl_deadline_writer_started) {
    rc = pthread_create(&thread, &attr, ocannl_deadline_writer, NULL);
    if (rc == 0) ocannl_deadline_writer_started = 1;
  }
  if (rc == 0) rc = pthread_create(&thread, &attr, ocannl_deadline_watchdog, NULL);
  pthread_attr_destroy(&attr);
  pthread_sigmask(SIG_SETMASK, &saved, NULL);
  if (rc != 0) return 0;
  ocannl_deadline_thread_started = 1;
  return 1;
}
#endif

/* Whether this build can arm the deadline at all: the predicate [arm] is compiled under, published
   so a test gates on it rather than on [arm]'s report of success. */
CAMLprim value ocannl_device_init_deadline_supported(value unit) {
  (void)unit;
#if defined(_WIN32)
  return Val_false;
#else
  return Val_true;
#endif
}

/* [ocannl_device_init_deadline_arm seconds message]: a positive generation when the deadline is now
   armed (to be passed to [disarm]); 0 when disabled, refused, nested in this thread's own armed
   region, or on Windows; -1 when it should have been armed and the watchdog thread could not be
   started (or the monotonic clock read), which the caller must not take as "no deadline". Waits,
   outside the OCaml runtime lock, while another thread's region is armed. */
CAMLprim value ocannl_device_init_deadline_arm(value v_seconds, value v_msg) {
#if defined(_WIN32)
  (void)v_seconds;
  (void)v_msg;
  return Val_int(0);
#else
  double seconds = Double_val(v_seconds);
  char msg[OCANNL_DEADLINE_MSG_MAX];
  size_t len;
  long armed = -1;
  pthread_t self = pthread_self();
  if (!(seconds > 0.0)) return Val_int(0);
  /* A bound past three years is a bound all the same, never "disabled". */
  if (seconds > 1e8) seconds = 1e8;
  if (!ocannl_sigalrm_is_default())
    return Val_int(0);
  /* Copied before the runtime lock is released: the OCaml string may move meanwhile. */
  len = caml_string_length(v_msg);
  if (len > OCANNL_DEADLINE_MSG_MAX) len = OCANNL_DEADLINE_MSG_MAX;
  memcpy(msg, String_val(v_msg), len);
  caml_enter_blocking_section();
  pthread_mutex_lock(&ocannl_deadline_lock);
  if (ocannl_deadline_armed && pthread_equal(ocannl_deadline_owner, self))
    armed = 0;
  else {
    while (ocannl_deadline_armed || ocannl_deadline_dying)
      pthread_cond_wait(&ocannl_deadline_free, &ocannl_deadline_lock);
    /* Checked again after the wait, right before committing: another thread may have installed
       its own SIGALRM handler meanwhile, and a program with its own handler gets no deadline. */
    if (!ocannl_sigalrm_is_default())
      armed = 0;
    else if (ocannl_deadline_start_thread()) {
      int64_t now = ocannl_monotonic_ns();
      if (now >= 0) {
        memcpy(ocannl_deadline_msg, msg, len);
        ocannl_deadline_msg_len = len;
        ocannl_deadline_due_ns = now + (int64_t)(seconds * 1e9);
        ocannl_deadline_owner = self;
        ocannl_deadline_armed = 1;
        if (++ocannl_deadline_generation <= 0) ocannl_deadline_generation = 1;
        armed = ocannl_deadline_generation;
        pthread_cond_signal(&ocannl_deadline_cond);
      }
    }
  }
  pthread_mutex_unlock(&ocannl_deadline_lock);
  caml_leave_blocking_section();
  return Val_long(armed);
#endif
}

/* [ocannl_device_init_deadline_disarm generation]: take back the deadline the [arm] that returned
   [generation] gave. A no-op when nothing is armed, or when the armed deadline is another arm's --
   one a fork child armed after inheriting this region (gh-ocannl-1446). */
CAMLprim value ocannl_device_init_deadline_disarm(value v_generation) {
#if defined(_WIN32)
  (void)v_generation;
#else
  long generation = Long_val(v_generation);
  pthread_mutex_lock(&ocannl_deadline_lock);
  if (ocannl_deadline_armed && ocannl_deadline_generation == generation) {
    ocannl_deadline_armed = 0;
    pthread_cond_signal(&ocannl_deadline_cond);
    pthread_cond_broadcast(&ocannl_deadline_free);
  }
  pthread_mutex_unlock(&ocannl_deadline_lock);
#endif
  return Val_unit;
}

/* --- CPU topology facts (gh-ocannl-530) ---

   These stubs report facts only; the pool-restriction policy that consumes them lives in
   Cc_backend. Each degrades to an "unknown" answer rather than failing: the callers treat
   unknown conservatively (no restriction). */

#if defined(_WIN32)
static int popcount_ull(unsigned long long m) {
  int count = 0;
  while (m) {
    count += (int)(m & 1);
    m >>= 1;
  }
  return count;
}
#endif

/* Perf-ranked core classes as a string "rank:count:maskhex;..." with higher rank = higher
   performance, or "" when the platform reports no class structure this way (Linux answers via
   sysfs, parsed on the OCaml side). */
CAMLprim value ocannl_core_classes_str(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(result);
#if defined(_WIN32)
  /* GetLogicalProcessorInformationEx(RelationProcessorCore): one entry per physical core,
     carrying EfficiencyClass (higher value = higher performance; P=1, E=0 on Intel hybrid) and
     the logical-processor group mask. Only processor group 0 is supported: hybrid client parts
     fit in one 64-CPU group, and the policy no-ops rather than guessing on larger machines. */
  DWORD len = 0;
  char *buf = NULL;
  char out[2048];
  unsigned long long masks[256];
  memset(masks, 0, sizeof(masks));
  out[0] = '\0';
  GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
  if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || len == 0)
    CAMLreturn(caml_copy_string(""));
  buf = (char *)malloc(len);
  if (buf == NULL) CAMLreturn(caml_copy_string(""));
  if (!GetLogicalProcessorInformationEx(RelationProcessorCore,
                                        (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf, &len)) {
    free(buf);
    CAMLreturn(caml_copy_string(""));
  }
  {
    char *p = buf;
    int ok = 1;
    while (p < buf + len) {
      PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info =
          (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)p;
      if (info->Relationship == RelationProcessorCore) {
        int cls = (int)info->Processor.EfficiencyClass;
        WORD i;
        for (i = 0; i < info->Processor.GroupCount; i++) {
          if (info->Processor.GroupMask[i].Group != 0) {
            ok = 0; /* Multi-group machine: report nothing rather than a partial map. */
          } else {
            masks[cls] |= (unsigned long long)info->Processor.GroupMask[i].Mask;
          }
        }
      }
      p += info->Size;
    }
    if (ok) {
      int cls;
      size_t pos = 0;
      for (cls = 255; cls >= 0; cls--) {
        if (masks[cls] != 0ULL) {
          pos += (size_t)snprintf(out + pos, sizeof(out) - pos, "%s%d:%d:%" PRIx64,
                                  pos > 0 ? ";" : "", cls, popcount_ull(masks[cls]),
                                  (uint64_t)masks[cls]);
          if (pos >= sizeof(out)) {
            out[0] = '\0';
            break;
          }
        }
      }
    }
  }
  free(buf);
  result = caml_copy_string(out);
#elif defined(__APPLE__)
  /* hw.nperflevels / hw.perflevelN.logicalcpu: perflevel0 is the highest-performance level.
     No affinity masks exist on Darwin, so the mask field is 0 (informational classes only; the
     cc pool there is libdispatch and the restriction policy never fires). */
  char out[512];
  int nlevels = 0;
  size_t sz = sizeof(nlevels);
  out[0] = '\0';
  if (sysctlbyname("hw.nperflevels", &nlevels, &sz, NULL, 0) != 0 || nlevels <= 0 ||
      nlevels > 16)
    CAMLreturn(caml_copy_string(""));
  {
    int i;
    size_t pos = 0;
    for (i = 0; i < nlevels; i++) {
      char name[64];
      int count = 0;
      sz = sizeof(count);
      snprintf(name, sizeof(name), "hw.perflevel%d.logicalcpu", i);
      if (sysctlbyname(name, &count, &sz, NULL, 0) != 0 || count <= 0) {
        out[0] = '\0';
        break;
      }
      pos += (size_t)snprintf(out + pos, sizeof(out) - pos, "%s%d:%d:0", pos > 0 ? ";" : "",
                              nlevels - 1 - i, count);
      if (pos >= sizeof(out)) {
        out[0] = '\0';
        break;
      }
    }
  }
  result = caml_copy_string(out);
#else
  result = caml_copy_string("");
#endif
  CAMLreturn(result);
}

/* Logical CPUs available to THIS process (affinity-respecting, unlike
   Domain.recommended_domain_count, which is affinity-blind on Windows). 0 = unknown. */
CAMLprim value ocannl_effective_cpu_count(value unit) {
  (void)unit;
#if defined(_WIN32)
  {
    DWORD_PTR pmask = 0, smask = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &pmask, &smask)) return Val_int(0);
    return Val_int(popcount_ull((unsigned long long)pmask));
  }
#elif defined(__linux__)
  {
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) != 0) return Val_int(0);
    return Val_int(CPU_COUNT(&set));
  }
#elif defined(__APPLE__)
  {
    int count = 0;
    size_t sz = sizeof(count);
    if (sysctlbyname("hw.logicalcpu", &count, &sz, NULL, 0) != 0) return Val_int(0);
    return Val_int(count);
  }
#else
  return Val_int(0);
#endif
}

/* The process's current affinity mask over logical CPUs (bit i = CPU i); 0 = no such notion on
   this platform (macOS), the mask involves CPUs beyond bit 63, or the query failed. Two
   same-width pinnings over different cores are different worker pools, so the mask (not just
   its popcount) enters the autotune pool tag. */
CAMLprim value ocannl_affinity_mask(value unit) {
  CAMLparam1(unit);
#if defined(_WIN32)
  {
    DWORD_PTR pmask = 0, smask = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &pmask, &smask))
      CAMLreturn(caml_copy_int64(0));
    CAMLreturn(caml_copy_int64((int64_t)pmask));
  }
#elif defined(__linux__)
  {
    cpu_set_t set;
    unsigned long long m = 0;
    int i;
    if (sched_getaffinity(0, sizeof(set), &set) != 0) CAMLreturn(caml_copy_int64(0));
    for (i = 0; i < CPU_SETSIZE; i++)
      if (CPU_ISSET(i, &set)) {
        if (i >= 64) CAMLreturn(caml_copy_int64(0)); /* Beyond the 64-bit mask: unknown. */
        m |= 1ULL << i;
      }
    CAMLreturn(caml_copy_int64((int64_t)m));
  }
#else
  CAMLreturn(caml_copy_int64(0));
#endif
}

/* 1 = running as a hypervisor GUEST (fabricated topology), 0 = the physical machine, -1 =
   cannot tell. The x86 answer starts from CPUID leaf 1 ECX bit 31, which WSL2, Hyper-V, KVM,
   VMware etc. all set — but which is ALSO set on the native Windows host once the Hyper-V role
   (or WSL2, or VBS) is enabled, because the host then runs atop the hypervisor. The host sees
   real topology (its processor APIs report the true core classes), so it must read as "physical
   machine". Distinguishing it by the root partition's CreatePartitions privilege (CPUID
   0x40000003, EBX bit 0) FAILS empirically: on a Windows 11 host with VBS the privilege is not
   visible to the OS partition (verified on the gh-530 rog box, which the plain check classified
   as a guest). The rule that holds instead is OS-based: on WINDOWS, a "Microsoft Hv" hypervisor
   is the host platform itself — Hyper-V guests are still told "Microsoft Hv" but see fabricated
   UNIFORM topology (no EfficiencyClass differentiation), so the core-classes check is the
   operative gate there and this probe answers "physical". Windows under a non-Microsoft
   hypervisor, and every hypervisor on other OSes (WSL2's Linux included), reads as guest. */
CAMLprim value ocannl_hypervisor_present(value unit) {
  (void)unit;
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
  {
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return Val_int(-1);
    if (!((ecx >> 31) & 1)) return Val_int(0);
#if defined(_WIN32)
    /* NOT __get_cpuid: it validates against the BASIC max leaf, which hypervisor leaves
       (0x4000_00xx) always exceed, so it refuses them; the raw __cpuid macro has no check. */
    __cpuid(0x40000000u, eax, ebx, ecx, edx);
    {
      char vendor[13];
      memcpy(vendor, &ebx, 4);
      memcpy(vendor + 4, &ecx, 4);
      memcpy(vendor + 8, &edx, 4);
      vendor[12] = '\0';
      if (strcmp(vendor, "Microsoft Hv") != 0) return Val_int(1);
    }
    /* KVM/QEMU Windows guests with Hyper-V enlightenments also present "Microsoft Hv" at
       0x40000000, but QEMU then moves the KVM signature to leaf 0x40000100 — a real Windows
       host has nothing there. Without this, such a guest with topology passthrough could pass
       both this gate and the >= 2-classes gate on fabricated masks. */
    __cpuid(0x40000100u, eax, ebx, ecx, edx);
    {
      char vendor[13];
      memcpy(vendor, &ebx, 4);
      memcpy(vendor + 4, &ecx, 4);
      memcpy(vendor + 8, &edx, 4);
      vendor[12] = '\0';
      return Val_int(strncmp(vendor, "KVMKVMKVM", 9) == 0 ? 1 : 0);
    }
#else
    return Val_int(1);
#endif
  }
#elif defined(__APPLE__)
  {
    int present = 0;
    size_t sz = sizeof(present);
    if (sysctlbyname("kern.hv_vmm_present", &present, &sz, NULL, 0) != 0) return Val_int(-1);
    return Val_int(present ? 1 : 0);
  }
#else
  return Val_int(-1);
#endif
}

/* Restrict the whole process to the logical CPUs set in [mask] (bit i = logical CPU i).
   0 = success; nonzero = failed or unsupported (macOS has no process affinity). */
CAMLprim value ocannl_set_process_affinity(value mask) {
#if defined(_WIN32)
  {
    DWORD_PTR m = (DWORD_PTR)Int64_val(mask);
    if (m == 0) return Val_int(1);
    return Val_int(SetProcessAffinityMask(GetCurrentProcess(), m) ? 0 : 1);
  }
#elif defined(__linux__)
  {
    unsigned long long m = (unsigned long long)Int64_val(mask);
    cpu_set_t set;
    int i;
    if (m == 0) return Val_int(1);
    CPU_ZERO(&set);
    for (i = 0; i < 64; i++)
      if ((m >> i) & 1) CPU_SET(i, &set);
    return Val_int(sched_setaffinity(0, sizeof(set), &set) == 0 ? 0 : 1);
  }
#else
  (void)mask;
  return Val_int(-1);
#endif
}

/* Cumulative CPU-seconds the whole machine has spent busy since boot, summed over every logical
   CPU, or NaN where this platform gives no such counter (gh-ocannl-1558: [Host_load], the
   co-tenant signal the autotuner's dispersion attribution reads). Linux: the aggregate [cpu] line
   of /proc/stat -- user, nice, system, irq, softirq and steal; idle and iowait are not busy, and
   guest time is already inside user, so the guest columns are not added again -- over
   sysconf(_SC_CLK_TCK). macOS: the mach host's CPU load ticks (user, system, nice) over the same
   tick rate; those tick counters are 32-bit and wrap after weeks of uptime, which a reader sees as
   a counter that went backwards. Windows: not provided. */
CAMLprim value ocannl_host_cpu_busy_seconds(value unit) {
  CAMLparam1(unit);
  double busy = NAN;
#if defined(__linux__)
  {
    FILE *f = fopen("/proc/stat", "r");
    long hz = sysconf(_SC_CLK_TCK);
    if (f != NULL) {
      unsigned long long user = 0, nice = 0, system = 0, idle = 0, iowait = 0, irq = 0,
                         softirq = 0, steal = 0;
      int n = fscanf(f, "cpu %llu %llu %llu %llu %llu %llu %llu %llu", &user, &nice, &system,
                     &idle, &iowait, &irq, &softirq, &steal);
      fclose(f);
      /* Kernels before 2.6.11 lack steal; anything shorter than the first four is malformed. */
      if (n >= 4 && hz > 0)
        busy = (double)(user + nice + system + irq + softirq + steal) / (double)hz;
    }
  }
#elif defined(__APPLE__)
  {
    host_cpu_load_info_data_t info;
    mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;
    mach_port_t host = mach_host_self();
    long hz = sysconf(_SC_CLK_TCK);
    kern_return_t kr =
        host_statistics(host, HOST_CPU_LOAD_INFO, (host_info_t)&info, &count);
    mach_port_deallocate(mach_task_self(), host);
    if (kr == KERN_SUCCESS && hz > 0)
      busy = ((double)info.cpu_ticks[CPU_STATE_USER] + (double)info.cpu_ticks[CPU_STATE_SYSTEM] +
              (double)info.cpu_ticks[CPU_STATE_NICE]) /
             (double)hz;
  }
#endif
  CAMLreturn(caml_copy_double(busy));
}
