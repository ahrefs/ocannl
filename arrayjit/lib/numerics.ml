open Base

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

    - [Bf16_auto] (the default): each backend's structural residency, unchanged from before this
      knob. The CPU backends compute bf16 in f32 (under {!field-narrow_compute_f32}), CUDA's bf16
      accumulators are f32 structurally (NVIDIA has no bf16 accumulate), and HIP and Metal keep
      storage-width bf16 accumulators mirroring their uniform-bf16 tensor-unit triples. Like
      [Fp16_auto] this retains latitude per backend: do not write code that assumes it equals
      [Bf16_narrow] — ask the backend's [accum_prec]/seeding instead.
    - [Bf16_wide] (config [false]): bf16 reduction accumulators reside in f32 on every backend,
      narrowing once per nest — the rounding of the narrowing alone, where gfx11's bf16-accumulate
      WMMA loses about a bf16 ulp at the partial-sum scale. As for [Fp16_wide], backends whose
      uniform-bf16 tensor-unit arm cannot accumulate f32 in the required emission scope
      ({!Backend_intf.mma_capability.mma_bf16_wide_acc_scopes} omits it) have those seeds withheld
      (gh-ocannl-545's seeding-vs-emission discipline). HIP swaps its rocWMMA arm to an f32
      accumulator fragment with a converted [d] boundary (both scopes); CUDA's arm is already wide
      in the per-statement scope; Metal's uniform-bf16 [simdgroup_matrix] arm declines.
    - [Bf16_narrow] (config [true]): the narrow side of the trade wherever a backend offers one. No
      target has native general bf16 arithmetic, so today it resolves exactly as [Bf16_auto] on
      every backend; it exists so a profile can name the narrow side without depending on how
      [Bf16_auto] resolves. *)
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
          bf16 on HIP/Metal (whose tiles accumulate in storage-width fragments) keeps storage
          residency so serial and tensorized legs stay width-uniform per backend (under
          {!Bf16_auto}). f16 residency is {!field-fp16_arithmetic}'s question and wide bf16
          residency {!field-bf16_arithmetic}'s, not this knob's (gh-ocannl-680, gh-ocannl-838). This
          knob reaches the GPU accumulators only where per-step narrowing can be restored
          SCHEDULE-UNIFORMLY: fp8 on CUDA and HIP (nothing tensorizes fp8 destinations). CUDA's bf16
          residency is structural — the mma accumulate is hardware-f32, so narrowing only the serial
          legs would resurrect the schedule-dependent width — and so is Metal's fp8 one: MSL has no
          fp8 type, every fp8 computation there runs in f32 ([Metal_backend]'s [compute_prec]). *)
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

let default () =
  {
    tf32_matmuls = Utils.get_global_flag ~default:false ~arg_name:"tf32_matmuls";
    narrow_compute_f32 = Utils.get_global_flag ~default:true ~arg_name:"narrow_compute_f32";
    fp16_arithmetic =
      (let s =
         String.lowercase
           (String.strip (Utils.get_global_arg ~default:"auto" ~arg_name:"fp16_arithmetic"))
       in
       if String.equal s "auto" then Fp16_auto
       else if Utils.bool_of_config_string ~arg_name:"fp16_arithmetic" s then Fp16_narrow
       else Fp16_wide);
    bf16_arithmetic =
      (let s =
         String.lowercase
           (String.strip (Utils.get_global_arg ~default:"auto" ~arg_name:"bf16_arithmetic"))
       in
       if String.equal s "auto" then Bf16_auto
       else if Utils.bool_of_config_string ~arg_name:"bf16_arithmetic" s then Bf16_narrow
       else Bf16_wide);
  }

(** A stable, exhaustive rendering of a policy, for cache keys and digests (gh-ocannl-568). Derived
    from the sexp rather than spelled out field by field, so a knob added to this deliberately
    open-ended record enters every fingerprint by construction — the failure mode being guarded
    against is precisely a policy field that a cache key forgot about. *)
let fingerprint p = Sexp.to_string (sexp_of_t p)

let policy : t option ref = ref None

(** The current policy; reads the global config on first use. *)
let get () =
  match !policy with
  | Some p -> p
  | None ->
      let p = default () in
      policy := Some p;
      p

(** Programmatic override, e.g. per-experiment toggles from training scripts. Call before
    compilation; routines already compiled keep the numerics they were compiled with. *)
let set_policy p = policy := Some p

(** Whether the current policy is {!Fp16_wide}: f16 reduction accumulators reside in f32 on every
    backend (gh-ocannl-680). Consulted by every backend's [accum_prec] and by the mma seeding gate
    in [Sketch_families.mma_tile_for_precisions] — one predicate on both sides of the gh-ocannl-545
    seam, so seeding and emission cannot drift apart on which f16 sites tensorize. *)
let fp16_accum_wide () =
  match (get ()).fp16_arithmetic with Fp16_wide -> true | Fp16_auto | Fp16_narrow -> false

(** Whether the current policy is {!Bf16_wide}: bf16 reduction accumulators reside in f32 on every
    backend (gh-ocannl-838). The bf16 twin of {!fp16_accum_wide}, consulted on both sides of the
    same seam: every backend's [accum_prec] and [mma] arm table, and the seeding gate in
    [Sketch_families]. *)
let bf16_accum_wide () =
  match (get ()).bf16_arithmetic with Bf16_wide -> true | Bf16_auto | Bf16_narrow -> false

(** The compute precision the CPU backends resolve a storage precision to under the current policy:
    fp16 stays fp16 only where {!field-fp16_arithmetic} requests it ([Fp16_narrow]) AND the target's
    arithmetic is genuinely 16-bit (gh-ocannl-516); every other narrow float computes in f32 under
    {!field-narrow_compute_f32} (gh-ocannl-517); everything else is itself. The single source of
    truth shared by [Cc_backend.compute_prec] (emission) and autotune's sketch seeding (the
    candidate pre-filter and the packed-[Stage] tile precisions, gh-ocannl-575) — the gh-ocannl-545
    lesson: when seeding and emission can disagree about a gate, candidates get timed under labels
    their rendering does not honor. *)
let cpu_compute_prec ~native_fp16_arithmetic (prec : Ops.prec) : Ops.prec =
  match prec with
  | Ops.Half_prec _
    when (match (get ()).fp16_arithmetic with Fp16_narrow -> true | _ -> false)
         && native_fp16_arithmetic ->
      prec
  | _ when Ops.is_narrow_float prec && (get ()).narrow_compute_f32 -> Ops.single
  | _ -> prec

(** The accumulator residency the CPU backends resolve a storage precision to: {!cpu_compute_prec},
    except that {!Fp16_wide}'s contract — f32 f16-accumulators on every backend, unconditionally —
    holds even where [narrow_compute_f32 = false] leaves the f16 {e compute} at storage width
    (gh-ocannl-680), and {!Bf16_wide}'s likewise for bf16 (gh-ocannl-838). Like {!cpu_compute_prec}
    this is the single source of truth shared by [Cc_backend.accum_prec] (emission) and autotune's
    sketch seeding: where the two diverge, a C-tile rendering cannot honor the residency and both
    the emission ([C_syntax.try_register_tile]) and the seeding pre-filter must decline (Codex P1
    round 1 on staging PR #477). *)
let cpu_accum_prec ~native_fp16_arithmetic (prec : Ops.prec) : Ops.prec =
  match prec with
  | Ops.Half_prec _ when fp16_accum_wide () -> Ops.single
  | Ops.Bfloat16_prec _ when bf16_accum_wide () -> Ops.single
  | _ -> cpu_compute_prec ~native_fp16_arithmetic prec
