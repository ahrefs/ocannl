(** Numerics policy (gh-ocannl-478's "option 3" knob): compute-precision decisions that change
    results, so they are chosen by the user — via the global config or {!set_policy} — never by the
    optimizer. Storage precisions live on tensor nodes ({!Tnode.t.storage_prec}); this record
    governs how computations over those storages are carried out. It must be identical across
    sibling autotune candidates: candidate schedules compete on speed, never on numerics (the
    bitwise-parity discipline of the tensorized twins depends on it).

    The record is deliberately open-ended — later compute-precision questions (fast-math
    transcendentals, accumulation widths, fp8 format selection per tensor class, gh-ocannl-492) land
    here rather than growing ad-hoc booleans elsewhere. *)

(** The fp16 compute/accumulator mode (gh-ocannl-680, refining gh-ocannl-516's boolean).

    - [Fp16_auto] (the default): each backend's structural residency. On the CPU backends f16
      computes in f32 (under {!field-narrow_compute_f32}); on the GPU backends f16 arithmetic is
      native and reduction accumulators keep storage residency, mirroring the tensor-unit triples
      every backend seeds at f16 accumulate. The auto resolution is per BACKEND and deliberately
      retains latitude: on hardware where a wide f16 accumulate costs nothing (datacenter-class
      NVIDIA runs f32-accumulate f16 mma at full rate) a later refinement may resolve it wide, so do
      not write code that assumes [Fp16_auto] equals [Fp16_narrow] — ask the backend's
      [accum_prec]/seeding instead.
    - [Fp16_wide] (config [false]): f16 reduction accumulators reside in f32 on every backend,
      narrowing once per nest — the strict cross-backend-uniform semantics. Backends whose
      tensor-unit f16 legs cannot accumulate f32 in the required emission scope
      ({!Backend_intf.mma_capability.mma_f16_wide_acc_scopes} omits it) have those uniform-f16 mma
      seeds withheld, per the gh-ocannl-545 seeding-vs-emission discipline — widening only the
      serial legs would restore the schedule-dependent width gh-ocannl-663 removed. Metal advertises
      both scopes since gh-ocannl-837 through mixed [simdgroup_matrix] accumulation and boundary
      conversion.
    - [Fp16_narrow] (config [true]): compute fp16 in fp16 on CPU targets that have native 16-bit
      arithmetic (ARMv8.2-FP16, AVX512-FP16) — gh-ocannl-516's opt-in, trading fp16's 10-bit
      mantissa and 65504 range for a doubled lane count. On targets that merely promote to float it
      is ignored (costs accuracy for no speed), and the GPU backends behave as under [Fp16_auto]
      (their f16 arithmetic is native narrow already). *)
type fp16_mode = Fp16_auto | Fp16_narrow | Fp16_wide [@@deriving sexp, compare, equal]

(** The bf16 accumulator mode (gh-ocannl-838), the same three-way shape as {!fp16_mode} so the
    policy surface stays one shape for both 16-bit float formats.

    - [Bf16_auto] (the default): each backend's own resolution. The CPU backends compute bf16 in f32
      (under {!field-narrow_compute_f32}), CUDA's bf16 accumulators are f32 structurally (NVIDIA has
      no bf16 accumulate), HIP resolves WIDE since gh-ocannl-1051 (as [Bf16_wide]: gfx11's
      bf16-accumulate WMMA is not exactly rounded and drifts grossly over long reductions, and the
      f32 arm was measured at most ~8% slower on tensorized GEMM cells;
      [Hip_backend.bf16_accum_wide]), and Metal keeps storage-width bf16 accumulators mirroring its
      uniform-bf16 tensor-unit triple. Like [Fp16_auto] this retains latitude per backend: do not
      write code that assumes it equals [Bf16_narrow] — ask the backend's [accum_prec]/seeding
      instead.
    - [Bf16_wide] (config [false]): bf16 reduction accumulators reside in f32 on every backend,
      narrowing once per nest — the rounding of the narrowing alone, where gfx11's bf16-accumulate
      WMMA loses about a bf16 ulp at the partial-sum scale. As for [Fp16_wide], backends whose
      uniform-bf16 tensor-unit arm cannot accumulate f32 in the required emission scope
      ({!Backend_intf.mma_capability.mma_bf16_wide_acc_scopes} omits it) have those seeds withheld
      (gh-ocannl-545's seeding-vs-emission discipline). HIP swaps its rocWMMA arm to an f32
      accumulator fragment with a converted [d] boundary (both scopes); CUDA's arm is already wide
      in both (the fragment scope since gh-ocannl-1063, under every policy); Metal swaps its
      [simdgroup_matrix] arm to a float accumulator over bfloat operands with a converted [d]
      boundary (both scopes, gh-ocannl-923).
    - [Bf16_narrow] (config [true]): the narrow side of the trade wherever a backend offers one. No
      target has native general bf16 arithmetic, so it resolves as [Bf16_auto] everywhere except
      HIP, where it keeps the bf16-accumulate WMMA arm and storage-width serial accumulators that
      [Bf16_auto] resolved to before gh-ocannl-1051. It is what the [approximate] profile names,
      which is how that profile kept the narrow side when auto moved. *)
type bf16_mode = Bf16_auto | Bf16_narrow | Bf16_wide [@@deriving sexp, compare, equal]

type t = {
  tf32_matmuls : bool;
      (** Allow tensor-core matmuls over uniform-f32 operands to compute in tf32 on backends with a
          tf32 tile shape (CUDA sm_80+): f32's exponent range with a 10-bit mantissa, accumulation
          in f32. Off by default — opt-in like PyTorch's [allow_tf32], because enabling it silently
          changes numerics. Metal ([simdgroup_float8x8] is genuine f32) and HIP (RDNA WMMA has no
          tf32-like shape) are unaffected. *)
  narrow_compute_f32 : bool;
      (** Run the arithmetic over narrow-float storage (bf16, fp16, fp8) in f32 on backends that
          have no native narrow arithmetic — the CPU backends, where every narrow operator is an
          explicit widen/op/narrow round-trip anyway (gh-ocannl-517). Storage stays narrow: reads
          widen once at the load, the result narrows once at the store, and the intermediates of an
          assignment keep f32 mantissa instead of being rounded per operator. That is both the
          faithful reading of "16-bit storage with f32 compute" and what makes the vectorized
          renderings — which are f32/f64 shaped — reachable for narrow-storage kernels.

          A reduction accumulator is such an intermediate (gh-ocannl-639): every rendering of an
          accumulation nest — the plain serial fallback included — holds the accumulator at the
          resolved compute precision across the whole nest and narrows it once at the store, so the
          effective accumulation width is this policy's, never a property of which schedule happened
          to place the accumulator in a register. (Narrowing POINTS beyond that single one remain a
          property of a schedule's reduction structure: a k-blocked schedule stores
          storage-precision partials at its block boundaries by construction.)

          On by default: it strictly increases accuracy relative to per-operator rounding and is the
          precondition for narrow storage being a speedup rather than a pessimization on CPU. Turn
          it off to recover the pre-gh-517 semantics, where every operator rounds to the target
          node's storage precision.

          The GPU backends' {e compute} precision is unaffected either way — they have native 16-bit
          types and arithmetic, so pointwise narrow arithmetic computes where it stores. Their
          reduction-{e accumulator} residency follows the tensor-unit formats (gh-ocannl-663):
          CUDA's bf16 mma legs hold f32 per-lane registers, so its serial bf16 legs widen to match,
          and fp8 — which has an accumulator format on no backend — takes f32 residency everywhere;
          bf16 on Metal (whose tiles accumulate in storage-width fragments) keeps storage residency
          so serial and tensorized legs stay width-uniform (under {!Bf16_auto}), as HIP's does only
          under {!Bf16_narrow} since gh-ocannl-1051. f16 residency is {!field-fp16_arithmetic}'s
          question and wide bf16 residency {!field-bf16_arithmetic}'s, not this knob's
          (gh-ocannl-680, gh-ocannl-838). This knob reaches the GPU accumulators only where per-step
          narrowing can be restored SCHEDULE-UNIFORMLY: fp8 on CUDA and HIP (nothing tensorizes fp8
          destinations). CUDA's bf16 residency is structural — the mma accumulate is hardware-f32,
          so narrowing only the serial legs would resurrect the schedule-dependent width — and so is
          Metal's fp8 one: MSL has no fp8 type, every fp8 computation there runs in f32
          ([Metal_backend]'s [compute_prec]). *)
  fp16_arithmetic : fp16_mode;
      (** How f16 computes and accumulates, per {!fp16_mode} (gh-ocannl-680). The narrow request is
          fp16-specific because fp16 is the one narrow format a CPU can execute natively — bf16 has
          no C type and no general ARM/x86 arithmetic, and stays emulated by design. The asymmetry
          with {!narrow_compute_f32} is deliberate: computing in fp16 trades accuracy for
          throughput, while widening to f32 trades nothing — which is also why [Fp16_auto] rather
          than [Fp16_narrow] is the default, and why the narrow request only takes effect where the
          target's arithmetic is genuinely 16-bit
          ({!Ir.Backend_intf.hardware_limits.native_fp16_arithmetic}). *)
  bf16_arithmetic : bf16_mode;
      (** How bf16 accumulates, per {!bf16_mode} (gh-ocannl-838). bf16 COMPUTE needs no knob: no
          target has native bf16 arithmetic, so {!field-narrow_compute_f32} already decides it. *)
}
[@@deriving sexp, compare, equal]

(** Serialization and structural comparison are deliberately public: callers can persist and compare
    complete policies and individual modes without consulting the process-wide policy. The [sexp_of]
    converters also provide debugging and observability entry points. *)

val default : unit -> t
(** Read a policy from the global configuration without changing the cached current policy. *)

val fingerprint : t -> string
(** Exhaustive, stable policy rendering for cache keys and digests. *)

val policy : t option ref
(** The process-wide cached policy. Retained for compatibility with direct overrides and resets;
    prefer {!get} and {!set_policy} for ordinary use. [None] makes the next {!get} reload the global
    configuration. Changing this reference does not affect already compiled routines. *)

val get : unit -> t
(** Current policy; reads and caches the global configuration on first use. *)

val set_policy : t -> unit
(** Override the current policy before compilation. Already compiled routines retain their
    compilation policy. *)

val fp16_accum_wide : unit -> bool
(** Whether the current policy explicitly requests {!Fp16_wide}. *)

val bf16_accum_wide : unit -> bool
(** Whether the current policy explicitly requests {!Bf16_wide}. Backend resolution of {!Bf16_auto}
    can also use wide accumulators. *)

val cpu_compute_prec : native_fp16_arithmetic:bool -> Ops.prec -> Ops.prec
(** Resolve storage precision to CPU compute precision under the current policy. *)

val cpu_accum_prec : native_fp16_arithmetic:bool -> Ops.prec -> Ops.prec
(** Resolve CPU accumulator precision, including unconditional widening for {!Fp16_wide} and
    {!Bf16_wide}. *)
