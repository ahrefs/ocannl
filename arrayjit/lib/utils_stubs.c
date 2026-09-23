#if defined(__linux__) && !defined(_GNU_SOURCE)
/* sched_getaffinity / sched_setaffinity / CPU_* macros. */
#define _GNU_SOURCE
#endif

#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
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
#if defined(__GLIBC__) || defined(__APPLE__)
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
