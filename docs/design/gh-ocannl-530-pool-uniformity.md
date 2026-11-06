# gh-ocannl-530: cc worker-pool uniformity on hybrid CPUs

## Context

gh-ocannl-530 established (benchmarks/report-gh530-rog-pinning.md, 36 parity-gated runs on a
Core Ultra 9 275HX, 8 P-cores + 16 E-cores, no SMT) why conv-sketch tuning wins do not port to
hybrid x86:

- **Mixing core classes in one pool is causal**: at width 8 the mixed pool tunes below *both*
  uniform endpoints (1.667x vs 1.752x for 8P and 2.097x for 8E), which no monotone dependence on
  the P:E proportion can produce. Mechanism: pool-parallel Grid chunks end at a barrier, so the
  step is set by the slowest worker; the more a schedule depends on balanced workers, the more a
  two-speed pool costs it — which is why the effect falls on the tuned side while default
  baselines barely move.
- **Pool width is separately causal**: 1.678x -> 1.337x over width 8 -> 16 with composition and
  chunk count both fixed.
- The two compound; the full machine (24 wide, 67% E) is the worst case. There the search stops
  crowning the split-reduce family that wins on every narrower pool, and the crown collapses to
  materialize-all.
- **Restricting the pool to a uniform core class buys 32–34% (`cifar_conv`) and 25–26%
  (`cifar_stride`)** — full-machine tuned against subset tuned — while the *default* (untuned)
  baselines are within noise of the full machine (P-only was even slightly faster on
  `cifar_conv`: 1346.96 vs 1377.18 ms).

Separately, the issue recorded a design constraint: **WSL2 fabricates CPU topology** (rog's 24
vCPUs present as uniform; minix interleaves SMT siblings), so any machine-derived geometry that
trusts guest-visible topology gets garbage under virtualization and must detect it and fall back.

## Decision

**The `cc` backend restricts its worker pool to the highest-performance core class, by default,
on hybrid, native (non-virtualized) topologies.** Everything else is a conservative no-op.

This chooses *pool uniformity* over the alternative the issue left open — core-type-aware seed
parameterization or schedules. Rationale:

- Restriction captures most of the measured win (25–34%) with a policy whose mechanism —
  process-affinity masks sized down to one core class — is exactly what the campaign measured.
  Core-type-aware schedules were never constructed; their value is unknown, their complexity
  (per-class work division inside one kernel, straggler-aware chunking) is not.
- With a uniform pool, the existing seed families need no composition input: the crown family
  that mixed pools lose (`F_split_saved`) is recovered "for free" on every uniform subset
  measured.
- The untuned baselines show restriction does not sacrifice default performance on the measured
  machine, so defaulting it on does not trade default-path speed for tuned-path speed.

### Policy

New config key `cc_pool_core_class = auto | all | performance | efficiency` (default `auto`):

- `auto`: restrict to the performance class iff **all** of: the pool backend is OpenMP (not
  libdispatch, not `none`); the topology probe reports ≥ 2 core classes with usable CPU masks;
  the machine is not virtualized (hypervisor bit / guest markers); and the process is not already
  externally pinned below the full machine (an external mask is the user's decision — respect
  it). Otherwise no-op.
- `all`: never restrict (pre-gh-530 behavior).
- `performance` / `efficiency`: force that class on a hybrid topology, overriding an external
  pinning and skipping the hypervisor gate (an explicit setting is the user's judgment — this
  also keeps big.LITTLE ARM Linux forceable, where hypervisor detection is `Unknown`); no-op
  when there is no restrictable class structure or the pool is libdispatch. `efficiency` picks
  the class *just below* the performance class, not the slowest: on three-class parts
  (P / E / LP-E) the low-power island is a near-serial pool nobody means by "efficiency
  cores".

The performance class is chosen over efficiency for `auto` because on the measured machine it
had slightly better tuned absolute times (768.71 vs 793.12 ms on `cifar_conv`), a faster
*default* baseline (E-only default was 10% worse), and narrower pools search faster.

### Mechanism

Process affinity (`SetProcessAffinityMask` on Windows, `sched_setaffinity` on Linux), applied
once, lazily, before the first OpenMP kernel is loaded. This is the campaign-verified lever:
mingw libgomp sizes its pool from the process affinity mask, and Linux libgomp does the same.
It is deliberately **not** done with `OMP_NUM_THREADS`/`OMP_PLACES` environment variables: on
Windows, `putenv` from the OCaml runtime (Win32 env) is not reliably visible to the getenv of
the CRT instance a mingw-built libgomp reads, whereas the affinity mask is process state every
runtime observes.

Timing constraint recorded here because it is load-bearing: libgomp computes its default team
size in its ELF/PE constructor, i.e. at `dlopen` of the first kernel linked with `-fopenmp` —
not at the first parallel region. The restriction must therefore be applied before the first
`c_compile_and_load` dlopens anything, which is where the lazy is forced.

Scope of the affinity call differs by OS, and the difference is load-bearing: Windows
`SetProcessAffinityMask` restricts the whole process including already-running threads, while
Linux `sched_setaffinity(0)` restricts the *calling thread* and whatever it spawns afterwards.
The Sync scheduler runs kernels on the calling thread, so the pre-dlopen force suffices there;
Multidev forces the policy before spawning its worker domains (which it also sizes by the
effective pool width, keeping a restricted or pinned run from oversubscribing its CPU subset).
Child processes (candidate compiles during a search) inherit on both platforms — that matches
how every gh-530 arm ran. The restriction is process-scoped in effect: a mixed-backend program
that wants full-width host threads alongside occasional cc kernels should set
`cc_pool_core_class=all`. Machines with more than 64 logical CPUs in one class/group are out of
scope for v1 (the mask is 64-bit, Windows processor groups are not crossed); the policy no-ops
there.

### Topology probe

A new `Cpu_topology` module (utils library, C stubs in `utils_stubs.c`) reports facts only;
policy stays in `cc_backend`:

- `core_classes ()`: perf-ranked classes with logical-CPU counts and affinity masks. Sources:
  Windows `GetLogicalProcessorInformationEx` (`RelationProcessorCore`, `EfficiencyClass` —
  higher value = higher performance, P=1/E=0 on Intel hybrid); Linux
  `/sys/devices/cpu_core/cpus` + `/sys/devices/cpu_atom/cpus` (Intel hybrid), falling back to
  grouping by `cpu_capacity` (ARM big.LITTLE); macOS `hw.nperflevels` /
  `hw.perflevelN.logicalcpu` (counts only, no masks — informational, since the pool there is
  libdispatch).
- `effective_cpu_count ()`: affinity-respecting width (`GetProcessAffinityMask` /
  `sched_getaffinity`), fixing the recorded trap that `Domain.recommended_domain_count` is
  affinity-blind on Windows (and `_SC_NPROCESSORS_ONLN`-based elsewhere).
- `hypervisor_present ()`: x86 CPUID leaf 1 ECX bit 31 — refined for Hyper-V, because the
  native Windows host also sets that bit once the Hyper-V role (or WSL2, or VBS) is enabled,
  yet sees real topology. The refinement is OS-based: on Windows, a "Microsoft Hv" hypervisor
  is the host platform itself and reads as physical — Hyper-V guests are also told
  "Microsoft Hv" but see fabricated *uniform* topology, so the ≥2-classes check is the
  operative gate for them. (Recognizing the root partition by its CreatePartitions privilege,
  CPUID 0x40000003 EBX bit 0, fails empirically: on the gh-530 rog box, a VBS-era Windows 11,
  the privilege is not visible to the OS partition.) Windows under a non-Microsoft hypervisor,
  and every hypervisor on other OSes (WSL2's Linux included), reads as guest.
  `kern.hv_vmm_present` on macOS.
  `Unknown` (non-x86 Linux) blocks restriction — "cannot rule virtualization out" is treated
  like "virtualized". The one hybrid family this leaves unrestricted is ARM big.LITTLE Linux,
  which no measurement covers yet.

These facts are also the input surface the narrow-fp16 seed work (re-scoped out of gh-530) will
consume: element-size/lane-count seeds keyed on `native_fp16_arithmetic` plus the effective
width reported here.

### Consumers wired in this change

1. **`cc_parallel_chunks` auto** becomes `4 x effective pool width` (the restricted class width
   when the policy fired, else `effective_cpu_count ()`), replacing the affinity-blind
   `4 x Domain.recommended_domain_count ()`. This closes the recorded footgun where a pinned
   Windows run silently got a grid decomposition sized for the full machine.
2. **Autotune cache identity.** Crowned schedules do not transfer across pools (the campaign's
   cross-arm replay: the P-only crown is worse than materialize-all on the full machine), so the
   pool must enter the disk-cache key the way the numerics policy already does.
   `hardware_limits` gains `worker_pool_tag : string option`; the CPU backends fill it with a
   compact signature, GPU backends leave it `None` (their keys do not change).
   `Schedule_cache.cache_key` appends it when present. The tag grammar: `w8P`/`w16E` for a
   policy-restricted class; `w24` for the unrestricted full machine; `w8xc03c03` for an
   externally pinned pool, carrying the mask itself — two same-width pinnings over different
   cores (8 P-cores vs 8 E-cores) are different pools whose crowns must not replay onto each
   other. Flipping the policy, or running under a different pinning, then re-tunes instead of
   replaying a crown measured on a different pool.

## Alternatives considered

- **Core-type-aware seeds/schedules**: unmeasured, higher complexity, and the uniform-pool data
  says most of the win does not need them. Remains the open follow-up recorded on the issue.
- **Restricting only during the search**: unsound — a crown is only meaningful on the pool it
  was timed on; the shipped schedule must run on the same pool.
- **OMP environment variables**: rejected for the Windows CRT-environment reliability reason
  above, and because affinity is the mechanism the measurements validated.
- **Making `auto` opt-in (default `all`)**: rejected; on the one hybrid machine measured the
  default-path cost is ~zero and the tuned-path win is 25–34%. The key exists precisely so a
  machine where this ever regresses has a one-line escape hatch.

## Risks and limitations

- Evidence is n=1 hybrid machine (plus the mechanism argument); the policy is a default with an
  off switch, not a hard-wired behavior.
- Hybrid parts with SMT on P-cores (e.g. 8P×2 + 8E) get the full P class including siblings;
  minix's SMT-loaded pools tuned well (2.48x), so siblings are kept in v1.
- Process-wide affinity confines candidate compiles during a search to the restricted class;
  the campaign ran entire searches under exactly this condition.
- A restricted test-suite process on a hybrid native box leaves E-cores idle; test kernels are
  tiny, and suite throughput is dominated by builds (which dune, not the test process, spawns).

## Out of scope, re-scoped on close

- Narrow (16-bit) `Tile_mma` register tiling (the #516/#517 remainder parked on gh-530) moves
  to its own issue; it consumes this change's probe facts and gating rather than blocking on
  them.
- Core-type-aware schedules on hybrid pools: open question, recorded on the issue.
