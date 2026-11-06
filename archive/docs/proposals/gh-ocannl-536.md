# Candidate-failure containment (gh-ocannl-536)

## The problem, restated

The compile path has one channel for "this did not work" — `exn`, overwhelmingly
`Invalid_argument` — and every site that observes it holds a private theory of what it means. The
inventory is larger than the three sites the issue names:

| layer | site | today |
|---|---|---|
| candidate loop | `compile_candidate`'s `with exn -> Error (Exn.to_string exn)` (autotune.ml:2719) | catches everything, collapses to a string |
| candidate loop | `try_spec` (autotune.ml:3367): the `FAILED` arm at :3370 and the `RUN FAILED` arm at :3406, the latter wrapping `time_routine` — launch *and* sync | catch everything |
| cache replay | the same `compile_spec_real` `Result` consumed at autotune.ml:3325, `cache entry replay FAILED, re-searching` | catches everything (and should) |
| advisory pick | `compile_advisory` (autotune.ml:2966), `model_default`'s counting arm at :3033, and four bare `exception _ -> None` arms (autotune.ml:3065, 3082, 3097, 3118) | catch everything, then recompile |
| tuner internals | `exception _` fallbacks at autotune.ml:2054 (→ `256 * 1024`), 2166, 2172, 2282, 2316, 3471 | catch everything, substitute a default |
| backend selection | `Context.auto`'s `try … with _ -> try_backends rest` (context.ml:103, :109) | catches everything, silently picks another backend |
| hand schedules | a user's own `?lowered_transform` through `Context.compile` (context.ml:111) | catches nothing |

Two corrections to the issue's reading of its own three instances, because they change what the
design has to do:

- **#519 is already fixed, and its fix is a hand-rolled instance of what this note proposes.** The
  landed fix has two halves: `validate_segments` (autotune.ml:2963) runs `validate_parallel`
  *eagerly* at the transform seam so the seam's handler fires (`validate_parallel` signals purely
  through `invalid_arg`, low_level.ml:3093-3190), and `compile_advisory` adds a
  backstop catch around `Context.compile` with a `fallback_if` guard against recompiling a failure
  the default pipeline shares. That is a phase boundary, a classification, and a fallback policy —
  written once, by hand, for one caller. The design's job is to make it the general case, not to
  fix #519 again.
- **#533 is not a "too narrow guard".** `try_spec`'s arm is `| exception exn ->` and wraps
  `time_routine`, which contains both `Context.run` and `Context.sync` (autotune.ml:68-95). A HIP
  driver abort raised out of `H.Stream.synchronize` (hip_backend.ml:349) *is* that shape and *is*
  caught. So either the fatal escape came from one of the two timing sites that are deliberately
  uncaught — the baseline at autotune.ml:3340, documented as the user's bug — or from the
  benchmark harness above `tune`, in both cases *after* an earlier candidate's abort had already
  killed the queue. Either way the catch was not the problem: the search continued on dead state.

That second correction is the whole reason this design needs two axes rather than one. It also
means **the escape site in #533 must be established before implementation** — the traceback in the
issue is truncated above `time_routine`, and the two readings imply different fixes. This is a
one-line instrumentation ask on the HIP machine (print the caller, or bracket the baseline timing),
not a design question.

So the three instances sort as: #519 = attribution, scope of the guard (fixed); the too-wide
catches above = attribution, wrong direction, unfiled; #533 = **damage**, orthogonal to how wide
any catch is.

## Where the current behaviour is actually wrong

Stripped of the "three sites guessing" narrative — post-#519 they all guess the same way, *catch
everything* — three defects remain, and they are the deliverables:

1. **Over-containment.** `with exn -> Error …` also absorbs `Out_of_memory`, `Assert_failure` and
   `Stack_overflow`. There are 20 `assert`s in low_level.ml and 13 in schedule.ml; a middle end that
   trips one during a tuning run reports a `FAILED` line and crowns some other candidate. That is
   the failure mode the issue names as worse than the crash. The current code has it.
2. **No recovery.** Nothing in the search asks whether the device still works after a candidate
   failed. #533 is this.
3. **No census.** `Exn.to_string exn` is where the structured decline data is thrown away.
   `Autotune.report` carries `candidates_failed : int` (autotune.ml:10) and nothing else — a single
   counter incremented at :3371 and :3407, with the message only logged, and hardcoded to `0` on the
   cache-hit path (:3308). #521's blocker table was assembled by hand-grepping `autotune_log`, and
   gh-479's "did the tensorized candidate lose, or never run?" is unanswerable from the returned
   value. `model_default` keeps its own parallel counter (`n_rejected`, autotune.ml:3033) — a second
   hand-rolled census of the same event.

## Exhibit: the same constructor, opposite meanings

`invalid_arg` is called 108 times in `schedule.ml`, all op preconditions — "this candidate is not
applicable", expected, skippable. (The `grep -c` figure of 113 that circulates includes one doc
comment and four *catch* sites at schedule.ml:3121, 3515, 3617, 3643; `raise (Invalid_argument …)`
appears zero times.) It is also what the cc backend raises when the generated C fails to compile
(cc_backend.ml:228), where the message itself reads *"This is a bug in OCANNL.
Please file an issue with the generated .c file"*. One constructor, two meanings at opposite ends
of the containment decision.

That second one is a trap this design must not walk into. Containing a cc codegen failure is
*correct* for the search — the candidate is candidate-specific and the run is better off
continuing — and it institutionalizes hiding an OCANNL bug behind a decline. The taxonomy below
therefore carries a severity distinction: `Backend_rejected` from a C backend is always a bug, and
its count is reported and logged even when the search survives it. Contained must not mean quiet.

`Utils.User_error` (utils.ml:935) is the closest thing to an existing attribution channel, and it
is ambiguous at exactly the interesting place: it carries the pool-capacity errors that are genuine
user misuse ("set `large_models=true`", backends.ml:33-38) *and* the hardware-limit rejections that
are ordinary declines (`Schedule.check_hardware_limits`, schedule.ml:5120, raising at 5126 and
5137; metal_backend.ml:1250). It is also raised across some fifteen modules for unrelated reasons,
so "is a `User_error`" carries no containment information on its own.

And on CUDA the question cannot be asked at all: `Nvrtc.compile_to_ptx` (cuda_backend.ml:292) is
called bare — the file contains no `try` at all — and no site in the tree names a cudajit exception
constructor. Whatever those bindings raise, we classify it by not looking.

## The conceptual core: two axes, not one

The issue's `Valid | Declined of reason | Fatal` is right about the shape and one axis short.

1. **Attribution** — *this schedule cannot be built or run here* versus *this process is in
   trouble*. Decides whether the search may continue at all.
2. **Damage** — what state the failure may have destroyed: candidate-visible buffers, the
   stream/queue, or the device context. Decides what must be restored before the search *can*
   continue.

#533 is attribution=candidate, damage≠nothing, which is exactly the cell no amount of widening or
narrowing a catch addresses.

**Runner damage is measured; memory effects are conservative.** The note's earlier form had the
classifier report a single `damage` field, which no classifier can honestly fill: the exception
that surfaces from
`H.Stream.synchronize` is `HIP_ERROR_INVALID_VALUE`, indistinguishable from a dozen benign launch
errors, and the queue's state is not in it. Replace the declared field with a state-transition hook:

```ocaml
type recovery =
  | Healthy
  | Recovered
  | Lost of exn * Printexc.raw_backtrace

val recover_after_launch_failure : device -> recovery
```

This is deliberately not a read-only `device_health` query. A backend implements it as "run a
trivial no-op and synchronize; if that fails, retire the runner, create a new one, run the no-op
again; if that also fails, mark the device lost". Returning `Recovered` commits to a usable state,
not merely to having allocated a new stream. The no-op is a backend-owned, allocation-free driver
probe (compiled/lazily cached independently of the candidate), so the hook needs only `device` and
does not depend on a possibly damaged candidate routine or OCANNL buffer state.

That commitment requires one explicit backend-interface change. `device.runner` is immutable today
(`backend_intf.ml:179`), while the device record also owns event tables whose values belong to that
runner. Recovery therefore mutates the shared device in place: make `runner` mutable, replace it
only after the new runner passes the probe, and clear `updating_for` plus
`updating_for_merge_buffer`. Buffers, pool ids, and constant caches survive because they are
device/context resources, not stream resources. All `Context.t` descendants refer to the same
device record, so subsequent candidate compiles and already-linked task closures observe the new
runner without rebuilding the scratch lineage. If a backend cannot make those guarantees, it
returns `Lost`; manufacturing a fresh device record would strand the existing pool table under a
new `device_id` and is not recovery.

The old runner must be destroyed or safely retired after all driver calls against it are finished.
The implementation must also specify what happens to driver-managed delimited events; merely
clearing OCANNL's maps while their callbacks can still fire is not enough.

**The recovery hook answers runner health and only runner health.** `Recovered` establishes that the
runner was rebuildable; it says nothing about whose fault the failure was or whether buffers were
written, and the axes must not leak into each other. A `Fatal` launch outcome stays `Fatal` whatever
recovery reports. The hook is still invoked best-effort before re-raising a `Fatal`, so a caller that
catches it cannot unknowingly reuse a poisoned device, but a failure in the hook never masks the
original exception and backtrace.

Runner recovery does **not** prove that candidate-visible buffers are intact. A failure reported
from `sync` may follow partial kernel execution; a no-op cannot detect or undo those writes.
Therefore every classified launch/sync rejection also carries a conservative execution effect:

```ocaml
type execution_effect = No_device_writes | Writes_may_have_occurred
```

Compile/link rejections and a backend-proven synchronous pre-dispatch launch rejection use
`No_device_writes`. An asynchronous failure defaults to `Writes_may_have_occurred`. With the current
`timing_ctx : Context.t` API there is no factory/checkpoint from which to restore a damaged scratch
lineage, so `Writes_may_have_occurred` is fatal even when the runner returns `Healthy` or
`Recovered`. A future `timing_ctx_factory` could relax that rule by rebuilding inputs and parameters,
but silently continuing on possibly modified data is out of scope. For a `Classified
{ execution_effect = No_device_writes; _ }` candidate failure, `Healthy` or `Recovered` lets the
search continue and `Lost` escalates to "candidate rejected, device lost during recovery".

Letting `Recovered` turn an unattributed launch failure—or a possibly-writing one—into a decline
would reintroduce exactly the silent degradation the design exists to remove.

Runner recovery is also not execution-ledger recovery. `Context.run` marks a routine executed before
the later `Context.sync` can report an asynchronous failure (context.ml:256-259). Extend the shared
`execution_ledger` with a poisoned marker for launch/sync failures. A classified
`No_device_writes` failure rolls back that candidate routine id and discards the returned candidate
context before continuing. A fatal or possibly-writing failure poisons the lineage before it
unwinds, so a caller that catches the exception cannot reuse a ledger that claims the failed routine
completed. When `timing_ctx` is a separate scratch lineage this poison stays there; the target
context remains usable. Context entrypoints check the marker and raise the stored contextual failure
until a future explicit restore API says otherwise.

Re-creating the runner is plausible on CUDA — `Cu.Stream.create` is called exactly once in the tree
(cuda_backend.ml:331, one compute stream per device) — but the record/event changes above make it
real work rather than a one-line probe. Whether an HSA-aborted HIP queue and its surrounding primary
context survive runner replacement is an empirical question for the HIP machine.

## Provenance: the verdict is not a property of the failure

`Schedule.check_hardware_limits` raising on an oversized workgroup is the right answer for a
hand-written schedule — the user asked for something the device cannot do — and the wrong answer
for a tuner candidate, where it is an ordinary decline. Nothing about the exception distinguishes
them; the difference is entirely *who proposed the schedule*.

This rules out the tempting design where the exception constructor carries the verdict (a
`Candidate_declined` exception that catch sites match on), and with it the reflex of rewriting 108
`invalid_arg` sites: those sites do not know their verdict. The vocabulary describes the *failure*;
the *caller* supplies the policy:

- the candidate loop: rejection ⇒ decline, continue;
- cache replay: any non-catastrophic failure ⇒ discard the entry, re-search (the entry is untrusted
  persisted data, possibly from another machine or an older tree);
- `model_default` (documented advisory): rejection ⇒ fall back to the default pipeline;
- a user's own `lowered_transform`: rejection ⇒ raise, reason rendered — today's behaviour,
  unchanged, better worded.

## Vocabulary

Two thirds of this vocabulary already exist, one layer down and one layer up:

- `Schedule.op_verdict = Op_legal | Op_illegal of string | Op_unknown of string` (schedule.ml:3170)
  is the *pre-compile* three-valued oracle. Its `Op_illegal` is built by catching `Invalid_argument`
  out of a hermetic apply (schedule.ml:3515 and 3643 out of `apply_opt_op`, 3617 out of
  `tensorize_llc`) — the legality layer already treats
  "`Invalid_argument` escaping `apply`" as a decline verdict. The new type is the *post-hoc*
  counterpart of that judgement and should read like it.
- `C_syntax.mma_rendering = Mma_intrinsics | Mma_register_tiled | Mma_scalar_fallback`
  (c_syntax.ml:41) with its census is the *rendering-level* outcome record; declines below it are
  log-only strings (`declinef`, c_syntax.ml:845).

What is missing is the boundary in between. The vocabulary separates a typed *cause* from the
caller's verdict:

```ocaml
(* arrayjit/lib/schedule_outcome.ml — depends on Base only; below Backend_intf and Schedule. *)

type phase =
  | Transform
  | Hardware_limits
  | Backend_codegen
  | Backend_compile
  | Backend_link
  | Launch
  | Sync

type resource =
  | Workgroup_threads
  | Workgroup_memory        (* threadgroup / __shared__ *)
  | Thread_scratch          (* private segment, local memory, spills — #533 *)

type severity = Expected | Compiler_bug
type execution_effect = No_device_writes | Writes_may_have_occurred

type cause =
  | Illegal_schedule of { check : string; detail : string }
      (* op preconditions in [Schedule.apply]; [Low_level.validate_parallel]. *)
  | Unsupported of { feature : string; detail : string }
      (* no mma for this precision/extent/arch; thread-space operands. *)
  | Resource_exceeded of {
      resource : resource; requested : int; limit : int option; detail : string }
  | Backend_rejected of {
      backend : string; stage : string; severity : severity; detail : string }
      (* nvrtc / hiprtc / MSL / cc compiler diagnostics. [cc] uses [Compiler_bug]. *)
  | Unclassified of {
      phase : phase; exn_constructor : string; detail : string }
      (* Temporary migration bucket; fatal in strict mode. *)

type rejection_key =
  | Illegal_schedule_key of string             (* [check] *)
  | Unsupported_key of string                  (* [feature] *)
  | Resource_exceeded_key of resource
  | Backend_rejected_key of string * string * severity  (* backend, stage, severity *)
  | Unclassified_key of phase * string         (* exception constructor, not its message *)

val key_of_cause : cause -> rejection_key

type fatal = {
  exn : exn;
  backtrace : Printexc.raw_backtrace;
  phase : phase;
  candidate : string option;
}

type classified_cause = { cause : cause; execution_effect : execution_effect }
type failure = Classified of classified_cause | Fatal of fatal
type 'a outcome = ('a, failure) Result.t

(* Internal transport only; never escapes the public Context/Schedule APIs. *)
exception Cause_at of phase * cause
exception Raised_at of phase * exn * Printexc.raw_backtrace
```

`cause` is deliberately coarse, but it is not itself the census key. `detail`, `requested`, and
`limit` retain the diagnostic evidence and are expected to vary between candidates;
`key_of_cause` drops those fields so #521's blocker table is a stable fold. The report keeps a
bounded sample of details per key. Keys are sorted by the derived constructor order before they
leave `Autotune`, so logs, JSON, and `.expected` files are deterministic. `Unclassified` obtains
`exn_constructor` from `Printexc.exn_slot_name`, not by parsing `Exn.to_string`.

`Fatal` stores the raw backtrace at the catch site. Re-raising later uses
`Printexc.raise_with_backtrace`; `raise fatal.exn` would replace the evidence this design is meant
to preserve. Candidate and phase context are data rather than a preceding log line, so a fatal
result can name the candidate even when logging is disabled.

Two variants from the earlier draft are deliberately absent. `Registers` has no reporting site
anywhere. `Grid_private_bytes` — the cc pool-worker stack cap — looked like a resource rejection
and is not one: c_syntax.ml:1185 does not raise, it *declines the parallel rendering and keeps the
loop serial*, logging through `declinef`. Folding it in would merge the renderer census with the
candidate census, which is precisely the distinction gh-479 exists to preserve. Renderer-level
degradations stay in `declinef` and `mma_census`.

Also out of scope, and worth naming so they are not swept in: the internal bail-outs
(`exception Bail`, `Opaque_stmt`, `Unfissionable` in schedule.ml, `Fail` in affine.ml) are analyses
declining inside a phase, never crossing a boundary. They stay local.

## Policy: preserve the cause at the narrow seam; decide at the caller boundary

Provenance determines the verdict, but a broad compile catch arrives too late to recover the cause.
`Backends.compile` currently invokes the user/candidate transform, `check_hardware_limits`, and the
backend compiler at distinct sites (backends.ml:620-627 and :701-715). Preserve causes before those
operations merge:

- add internal typed wrappers for `Schedule.apply` and `Low_level.validate_parallel`; they catch an
  `Invalid_argument` from inside the operation and raise/return `Illegal_schedule`. Autotune's
  generated transforms use those wrappers. Do **not** classify an arbitrary `Invalid_argument`
  escaping the whole user-supplied transform as a schedule decline — in strict mode it remains an
  unclassified fatal. The generic transform call only tags such an exception as
  `Raised_at (Transform, ...)`;
- make `Schedule.check_hardware_limits` raise/return a typed `Resource_exceeded` while it still has
  `requested`, `limit`, and `resource` in hand, then render the same `Utils.User_error` at a public
  hand-schedule boundary;
- translate backend compiler and linker diagnostics inside the backend call that knows its stage.

This does not rewrite 108 schedule preconditions. It adds typed entrypoints around the schedule
operation/validation functions, changes the two hardware-limit raises, teaches Autotune's transform
closures to use the typed entrypoints, and adds backend-local translations. `Cause_at` preserves a
known cause and `Raised_at` preserves only phase, exception, and raw backtrace; neither carries the
verdict. The caller still supplies that.

The common catcher has an explicit signature. The backend classifier is a parameter rather than a
global lookup: `schedule_outcome.ml` is below `Backend_intf`, so having it dispatch back upward
would introduce a dependency cycle.

```ocaml
type provenance = Candidate | Cache_replay | Advisory | User_schedule

val protect :
  classify_backend:(phase -> exn -> classified_cause option) ->
  provenance:provenance ->
  phase:phase ->
  ?candidate:string ->
  (unit -> 'a) ->
  'a outcome
```

Rules are applied in order:

1. **Fatal deny-list, except untrusted cache replay.** `Out_of_memory`, `Stack_overflow`, `Sys.Break`,
   `Assert_failure`. Host exhaustion, user interrupt, and "the compiler is confused" are never a
   candidate's fault, at any boundary. This is the part of the change that *removes* containment
   rather than adding it. `Utils.User_error` deliberately does **not** join the deny-list, but it is
   no longer treated as a rejection merely by constructor: the hardware-limit sites now preserve a
   typed cause, while an unrelated raw `User_error` is unclassified and therefore fatal in strict
   mode.
   *Exception, and it is not cosmetic:* cache replay (autotune.ml:3325) opts out of the deny-list
   for everything except `Out_of_memory` and `Sys.Break`. A cache entry is untrusted data; an
   `Assert_failure` while replaying one means the entry is incompatible with this tree, not that
   the compiler is confused, and the correct response is to discard it and re-search.
2. **Typed cause, then backend translation.** A cause preserved at a narrow seam is used directly.
   Otherwise a new `Backend_intf` hook,
   `classify_failure : phase -> exn -> classified_cause option`, lets a backend recognize its own
   driver
   errors; `None` means "not mine". Passing the phase matters because the same driver error can mean
   a compiler refusal at link and an asynchronous program failure at sync, and it lets the backend
   conservatively distinguish a pre-dispatch refusal from a possibly-writing asynchronous failure.
   Returning `classified_cause option` also prevents the nonsensical `Valid`-from-an-exception result
   allowed by the earlier
   `exn -> outcome option` shape. The hook belongs in
   `Backend_device_common` (backend_intf.ml:269), which `Backend` includes, and every implementor
   must supply it — the four real backends, `backend_impl.ml`'s functors, and the `*_missing.ml`
   stubs, with `backends.ml`/`context.ml` doing the first-class-module dispatch. Budget for that
   breadth; the default (`fun _phase _exn -> None`) keeps each one a one-liner until it has something
   to say.
3. **Phase default.**
   - a typed schedule, resource, unsupported-feature, or backend-compiler cause under
     any provenance → `Classified` with `No_device_writes`. The consumer policy then differs:
     candidates decline, cache entries are discarded, advisory picks fall back, and user schedules
     raise the rendered cause;
   - an *unclassified* transform/codegen/compile/link exception → `Fatal` in strict mode. A base
     compile succeeding does not prove a later disk-full error, compiler crash, context loss, or
     internal `Failure` is candidate-specific.
   - launch / sync (`Context.run`, `Context.sync`) → **`Fatal`**. A run failure is normally the
     program's bug (uninitialized inputs — the property `tune` already documents for its uncaught
     baseline timing, autotune.ml:3338), and only a typed/backend-recognized cause makes a launch
     failure `Classified`.

`Cache_replay` is the one provenance override: a typed failure or an `Assert_failure` /
`Stack_overflow` produced while applying persisted schedule data discards the entry and re-searches.
`Out_of_memory` and `Sys.Break` still propagate. Replay must thread `Cache_replay` through the whole
reconstruction/compile path rather than first classifying the failure as an ordinary live candidate.

Causes must not leak through the public API as a new exception contract. Add an internal
outcome-returning path:

```ocaml
val compile_outcome :
  provenance:provenance ->
  ?candidate:string ->
  ... ->
  t ->
  Assignments.comp ->
  Indexing.unit_bindings ->
  (t * routine) outcome
```

It performs backend dispatch, passes `Backend.classify_failure` to `protect`, and uses the
`Cause_at` / `Raised_at` tags installed around the existing backend pipeline calls to keep
transform, hardware-limit, compile, and link phases distinct. Existing `Context.compile` is the
`provenance:User_schedule` wrapper: it returns the valid value or renders a typed cause back into the
same `Invalid_argument` / `Utils.User_error` shape users see today, and re-raises fatal exceptions
with their stored backtraces. `Autotune` uses `compile_outcome` directly. The analogous internal
launch/sync wrappers take a `Context.t`, obtain the backend classifier through the same dispatch,
and return typed outcomes; the public `Context.run` / `Context.sync` contracts remain raising APIs.

**Three behavioural tightenings.** Rule 1 is the first. Unclassified compile failures becoming
fatal by default is the second. Rule 3's launch arm is the third:
`try_spec`'s blanket `RUN FAILED` catch becomes "decline only what the backend claims, else fatal".
That is deliberate — a candidate that produces garbage should not be quietly scored — and it is the
arm #533 populates.

## This design does not, by itself, fix #533

Worth stating plainly, because the earlier draft let the reader assume otherwise. Under rule 3 a
launch/sync failure is `Fatal` unless HIP's `classify_failure` attributes it, and what HIP delivers
at that point is `HIP_ERROR_INVALID_VALUE` out of `synchronize` — the same code an uninitialized
input yields. There is no signal to classify on. So after this design lands, #533's abort is still
fatal; what changes is that it is *contextualized*: the crash names the candidate, phase, original
exception, and preserved backtrace instead of surfacing as a bare `hip_stream_synchronize`
backtrace. It has no rejection reason because it was not classified as a rejection. That is a real
improvement over today and it is not containment.

Containment of #533 therefore comes from prediction, not from recovery.
`recover_after_launch_failure` only supplies the device to continue *on* once something else has
established that the candidate was at fault; on its own it cannot make an unattributed HIP abort a
decline, for the reason given above.
Attribution at launch has exactly two possible sources: a HIP `classify_failure` that finds a
narrower signal than `HIP_ERROR_INVALID_VALUE` — worth one look at whether the runtime exposes the
aborted-queue status distinctly, but do not plan on it — or moving the failure to compile time,
where a post-link validator can supply a typed resource cause before any state is damaged. The
precedent for the latter exists: Metal already re-checks the workgroup size against the compiled
pipeline state at link time (metal_backend.ml:1243-1253) on top of the pre-compile check.

Do **not** add the earlier draft's generic `max_thread_scratch_bytes` field yet. CUDA exposes a
function's per-thread local-memory use, but no directly corresponding `CU_DEVICE_ATTRIBUTE` is a
portable maximum for that value. HIP exposes `localSizeBytes`, a per-thread stack limit, and on
newer ROCm/hardware a device scratch-allocation threshold; those are related but not interchangeable.
**RESOLVED (gfx1151 / Radeon 8060S / ROCm 7.14 / WSL2).** The experiment below was run and the
validator landed; this paragraph's caution about which quantity to compare is now answered, and the
answer is *none of the three candidates it lists*.

- The rejecting limit is **not** the per-thread stack limit. `hipDeviceGetLimit(hipLimitStackSize)`
  reports 1024 on this device, while kernels at 104832 B/work-item launch happily — two orders of
  magnitude above it. It is also not the compiler's ceiling: hipcc refuses a stack frame over
  262136 B (`stack frame size (262160) exceeds limit (262136)`), and #533's 163856 B compiles fine.
- The limit is a **total scratch allocation cap of 4 GiB**, and it is a function of the per-work-item
  size ALONE — the runtime backs a fully-occupied device, not the requested grid. Confirmed by
  geometry sweeps: 98320 B/work-item launches at 204800 work-items, while 114704 B is rejected at a
  *single* work-item. So the check needs no launch dimensions, which is why it can live at link.
- The predicate that reproduces every one of ~70 sampled points is
  `ceil(pss / 64) * 64 * max_threads_per_multiprocessor * multiprocessor_count <= 4 GiB`.
  Measured boundary on this device: **104832 B accepted, 104848 B rejected** (the 16-byte step is
  the compiler's own frame granularity, so the cutoff cannot be pinned finer than that).
- The 4 GiB cap is **not** exposed by any HIP or HSA query. The abort text
  (`[UpdateScratch] scratch_size overflow!`) lives in `librocdxg.so`, in
  `wsl::thunk::ComputeQueue::UpdateScratch`, next to `Device::MaxScratchSlotsPerCu` — i.e. it is
  enforced by the WSL WDDM thunk. The multiplier is genuinely queried; the cap is a documented
  constant from this experiment, and `ocannl_hip_scratch_validation=false` disables the check on a
  platform where the model does not hold.

One correction to the premise above, worth recording because it changes what containment costs: on
the **driver-API path this backend actually uses** (`hipModuleLoadData` → `hipModuleGetFunction` →
`hipModuleLaunchKernel`), the scratch abort surfaces as a returned `hipErrorInvalidValue` from
`hipDeviceSynchronize` and the process **survives**; only the runtime-API `hipLaunchKernelGGL` path
takes SIGSEGV. The queue is still aborted either way, so this does not make recovery possible — but
"a driver-level process abort no OCaml handler can contain" is too strong for the path we are on.
Whether the surrounding context is reusable after such an abort remains the open question that
`recover_after_launch_failure` would have to answer, and it stays unbuilt.

Historical statement of the task, retained for context:

The `private_seg_size=163856` in #533 must first be matched experimentally to the limit that rejected
it.

The implementation seam is instead a backend-private post-link validator, run after obtaining the
compiled function/pipeline and before returning a routine:

```ocaml
(* Conceptual; the concrete function/kernel handle stays backend-private. *)
val validate_linked_kernel : device -> linked_kernel -> (unit, cause) Result.t
```

For HIP, extend hipjit as needed to expose the compiled function attributes and the relevant runtime
limit. If the #533 experiment establishes that `private_seg_size` is bounded by the per-thread
stack limit, compare those values. If it is the newer device scratch threshold, validate with its
actual total/per-dispatch semantics instead of pretending it is per-thread. CUDA may query local
size for diagnostics, but rejects only against an API-supported launchability condition. The
backend returns `Resource_exceeded Thread_scratch` with requested and limit populated whenever it
can prove the launch would fail; absence of such a proof is not a made-up `None` hardware limit.

If the OCaml bindings do not expose those queries, extending the bindings is the work. This remains
the design's centre of gravity: **turn failures whose damage exceeds nothing into failures that
happen before launch.** It is separable and should land as soon as the HIP experiment identifies
the right quantity; classification without it only improves the message.

Two notes on how far the precedent actually reaches. Metal's link-time check is the *only* pipeline
query the backend makes: shared memory is checked device-wide and ahead of time, from
`limits.max_workgroup_memory_bytes` (metal_backend.ml:347) inside `check_hardware_limits`, against
the schedule's declared usage rather than the compiled kernel's. The binding for the per-kernel
answer already exists and is simply unused
(`Metal.ComputePipelineState.get_static_threadgroup_memory_length`), so a separate small patch can
cross-check codegen's accounting. Do not couple that cleanup to #533's acceptance criteria. And on
this machine cudajit is not installed — `cuda_backend_impl.missing.ml` is selected — so the CUDA
arm of both the query and `classify_failure` can be written but not exercised here; it needs a
session on CUDA hardware, as the HIP arm needs one on minix.

## Fatal must still emit the report

The issue's complaint is that uncontained failures *delete measurements*. A `Fatal` that unwinds
out of `tune` today takes the whole report with it, including the candidates that did time
successfully. Once the baseline has completed, initialize best-so-far to that baseline and keep the
report state live around the candidate search. On a fatal candidate, build and emit a partial report
before re-raising with `Printexc.raise_with_backtrace`.

This is not quite "one `Exn.protect`": the callback is user code and can itself raise. The report
callback is invoked at most once. If it raises while handling an existing fatal candidate, log the
callback failure and re-raise the original fatal/backtrace; otherwise its exception propagates
normally. Base compile or baseline timing failures emit no autotune report because no candidate
measurements exist yet.

The report gains:

```ocaml
type decline_summary = {
  key : rejection_key;
  count : int;
  sample_details : string list;  (* bounded, e.g. first three distinct details *)
}

type terminal_failure = {
  phase : phase;
  candidate : string option;
  detail : string;
}

(* Added fields. *)
partial : bool;
declines : decline_summary list;
terminal_failure : terminal_failure option;
```

For a completed non-cache search, `partial = false`, `terminal_failure = None`, and
`candidates_failed = sum declines.count`. For a fatal partial report, the terminal failure is not a
decline and is therefore not included in `candidates_failed`. If an already-classified rejection
escalates because recovery is `Lost` or writes may have occurred, count its cause once in
`declines` and record the escalation separately as `terminal_failure`. A cache hit has an empty
decline list; a stale cache entry discarded before a fresh search is logged separately and does not
pretend to be a proposed live candidate.

## What the sites become

- **`compile_candidate`** (autotune.ml:2719): replace the outer string catch with
  a provenance parameter and `Context.compile_outcome`, retaining the cause produced by the narrow
  transform, hardware-limit, compiler, or linker seam. Live candidates pass `Candidate`; replay
  passes `Cache_replay`. It returns a typed `'compiled outcome`; no site reparses `Exn.to_string`.
- **`try_spec`'s timing arm** (autotune.ml:3406): protect launch and sync separately so the report
  names the phase. On `Classified`, record a candidate decline and run
  `recover_after_launch_failure`; continue only when recovery is
  not `Lost` and the classified effect is `No_device_writes`. A possibly-writing rejection becomes
  fatal because there is no scratch-lineage restore API. On `Fatal`, run recovery best-effort
  without changing the verdict, emit the partial report, then re-raise the stored
  exception/backtrace.
- **Cache replay** (autotune.ml:3325): put
  `protect ~provenance:Cache_replay` around winner reconstruction and call
  `compile_candidate ~provenance:Cache_replay`, so persisted-data failures are never first
  classified under ordinary live-candidate policy.
- **`compile_advisory`** (autotune.ml:2966) and `model_default`'s bare `exception _` arms: keep the
  fallback for `Classified` causes under `Advisory`, propagate `Fatal` immediately instead of paying
  for a recompile that will fail the same way. `fallback_if` narrows to its real job.
- **`validate_segments`** (autotune.ml:2963) can be **retired**. It exists because `validate_parallel`
  runs past the seam and the seam's handler could not see it; with the compile phase classifying
  that failure as `Classified`, `compile_advisory`'s backstop reaches the same fallback without
  duplicating the check. Deleting it is the concrete simplification that shows the abstraction pays
  for itself. (Keep it if profiling shows the eager check meaningfully cheaper than a failed
  compile — but say so, rather than leaving both.)
- **`Context.auto`** (context.ml:103 and :109): backend selection is not candidate compilation and
  does not use the compile-phase default. Introduce/recognize a narrow `Backend_unavailable` cause
  from backend discovery and catch only that while trying the next backend. The configured-backend
  arm preserves the original failure unless the backend name itself is unknown; it no longer turns
  driver initialization failures into `Invalid_argument "Unknown backend"`.

One `protect` with provenance and phase, typed translations at the existing narrow seams, one
backend classifier, one post-link validator, and one recovery hook.

## Consumers

`Autotune.report` gains the fields above alongside `candidates_failed`. #521's hand-assembled census
becomes a returned value, and gh-479's "did the tensorized candidate lose, or never run?" is
answered from the same fold. Exactly one non-test consumer needs updating today
(benchmarks/runners/ocannl/bench_mlp.ml:225), plus the tests that construct or print reports.
`C_syntax.mma_census` stays: *declined rendering* and *declined candidate* are different events, and
telling them apart is the point of that census. `autotune_log` keeps its per-candidate line; the
census is what a benchmark runner tabulates without parsing prose.

## Migration and compatibility

- **Unclassified failures are fatal by default.** `strict_failure_classification=true` is the
  default and is pinned explicitly in `test/config/ocannl_config`. Otherwise `Match_failure`,
  `Division_by_zero`, arbitrary internal `Failure`, and new compiler invariants would remain
  silently containable — the exact failure mode this design exists to remove.
- **Compatibility mode is a temporary escape hatch, not the ratchet.** Setting
  `strict_failure_classification=false` converts an unclassified transform/codegen/compile/link
  failure into `Classified { cause = Unclassified ...; execution_effect = No_device_writes }`, logs a
  prominent warning per key, and includes it in the census. It never weakens launch/sync
  classification. Benchmark sweeps may use it while the initial taxonomy is being populated, but
  CI and checked-in test configs do not. Once the census is empty across supported backends, remove
  the key rather than preserving a permanent permissive mode. (Named for the classifier rather than
  `autotune_*`, because `Context.compile` and `model_default` share it.)
- **User-facing behaviour is unchanged**: a hand-written schedule that violates a precondition still
  raises out of `Context.compile`, with a message built from the same `detail`. Nine test files
  pattern-match or print `Utils.User_error` as the public contract (test/operations/schedule_ops.ml,
  buffer_aliasing.ml, test_slice_alias.ml, context_copy.ml, merge_buffer_static_verification.ml,
  test_threefry_precision.ml, test_bounds_folded_gather.ml, symbolic_extent_launch.ml,
  test/einsum/test_symbolic_extents.ml); they keep passing.
- One new config key ⇒ the three-place update (`ocannl_config.reference`, `Utils.known_config_keys`,
  the consistency test's scan list). The revised design adds no speculative `hardware_limits`
  field. If a proven portable limit is added later, audit its `deriving sexp, compare, equal` users
  and whether it reaches the schedule-cache key before assuming existing caches survive.

### Test plan

Tests pinning the boundary in *both* directions — the second matters as much as the first, since
the failure this design most wants to prevent is a silently degraded tuning result.

These run on `cc`, so CI enforces the boundary itself on every machine:

- **Contained**: seed a candidate through `tune`'s transform seam that violates an op precondition
  (`Schedule.apply` raises `Invalid_argument`, device-independent). Assert the search completes, the
  report counts one decline, its key is `Illegal_schedule_key`, the detail sample is retained, and
  `candidates_failed = sum declines.count`.
- **Not contained**: a transform raising `Assert_failure` makes `tune` propagate rather than log a
  `FAILED` line and crown a different candidate — *and* the partial report is emitted before it
  propagates, names the candidate/phase, and the re-raised exception retains the original backtrace.
- **Unclassified is strict**: a transform raising `Failure "compiler bug"` is fatal under the
  checked-in config. A focused compatibility-mode test verifies that only compile-side
  unclassified failures become `Unclassified_key`; launch/sync remains fatal.
- **Advisory provenance**: `model_default` falls back for a typed illegal-schedule cause but
  propagates `Assert_failure` and unclassified `Failure`. This covers the original #519 consumer,
  not only `tune`.

Add a unit test of `protect` over synthesized exceptions/typed causes at each phase and provenance,
including the cache replay override, fatal deny-list, strict/permissive modes, and every `cause`
constructor. Test `key_of_cause` separately: changes to `detail`, `requested`, or `limit` must not
change the key; backend/stage/severity must. This is where `Resource_exceeded` gets portable
coverage: the policy is pure, so its cause-to-verdict mapping is testable on `cc`.

Add a fake-backend or backend-implementation unit test for the recovery state machine:

- the first no-op succeeds → `Healthy`, no state replacement;
- the old runner fails and a new runner succeeds → `Recovered`, runner replaced, stale event maps
  empty, pool/buffer identity preserved, and the next candidate uses the new runner;
- both fail → `Lost`, the search emits a partial report and stops;
- runner recovery succeeds but the classification says `Writes_may_have_occurred` → the current
  API poisons the scratch ledger, emits a partial report, and stops rather than timing the next
  candidate on suspect data;
- a `No_device_writes` sync refusal rolls back the candidate's executed id and discards its returned
  context before the next candidate;
- the original outcome is `Fatal` → recovery never upgrades it and a recovery exception never
  masks its backtrace; the affected lineage is poisoned even if the physical runner recovers.

`Context.auto` gets a focused test with injectable/fake backend discovery: only
`Backend_unavailable` advances to the next backend; `Out_of_memory`, `Sys.Break`, assertion failure,
and driver initialization failure propagate. The configured-backend arm must not relabel the latter
as an unknown name.

Two tests are genuinely GPU-gated and must be marked as such rather than counted as portable CI
coverage:

- **Contained, resource, end to end**: on a backend actually reporting `max_threads_per_workgroup`,
  an over-budget workgroup declines with key `Resource_exceeded_key Workgroup_threads`. `cc` cannot
  stand in — `cpu_mma_limits` builds on `no_hardware_limits` (schedulers.ml:20-27), so
  `max_threads_per_workgroup` is `None` and `check_hardware_limits` has nothing to reject on;
  hardware-axis annotations render serially there by design. Metal locally; the other GPU backends
  in later sessions on their machines.
- **HIP scratch, before launch**: build a kernel whose compiled `private_seg_size` crosses the
  experimentally established limit. Assert the post-link validator returns
  `Resource_exceeded_key Thread_scratch`, no launch occurs, and a following reduction on the same
  device is correct. This is the regression for #533; the workgroup-threads test is not a substitute.

Injecting a synthetic workgroup limit to make the first GPU test portable is possible but not worth
it: it would need a seam into `hardware_limits`, which backends supply, and what it would then test
is the policy — already covered above, without the seam. The GPU test's value is confirming that
the *real* rejection path lands in the right bucket, and only real hardware limits do that. The HIP
scratch test is different: its value is validating the binding query and the experimentally
established limit semantics, so it cannot be replaced by a synthetic limit.

Placement: `test/operations/autotune_containment.ml` with an `.expected` file, alongside the
existing `autotune_smoke` family; pure policy/key tests under `arrayjit/test`; backend recovery tests
next to the existing backend implementation tests. The HIP test is a GPU-gated standalone test with
the normal training lock if its setup exercises the process-local pool.

## Non-goals

- Making every failure recoverable. Device-scope damage ends the run; the deliverable there is
  attribution in the error message, not survival.
- Retrying candidates. A rejection is final for that candidate on that device.
- **Hangs.** #532 — the middle-end wedge that cost the same HIP cells — raises nothing, so no
  classification reaches it. Containing a hang needs a timeout, a different mechanism, and its own
  decision about what a partially-timed candidate means. Explicitly out of scope here.
- Classifying *wrong values*. A candidate that compiles, runs, and computes garbage belongs to
  gh-ocannl-484's numerics pinning, not to containment.

## Suggested landing order

0. Establish where #533's exception actually escaped (HIP machine, one instrumented run). If it was
   the baseline timing, the scratch validator still prevents the damage; the partial-report path is
   then not involved in that reproduction.
   **Established afterwards, and it was neither.** With the validator in place the abort became a
   typed `Resource_exceeded Thread_scratch` at `Backend_link` — and the run still died, because the
   compile it fired on was the tuner's own *base* compile (`Autotune.tune`'s identity-transform
   capture), the one compile in `tune` that went through raising `Context.compile` rather than
   `compile_outcome`. Its escape route bypassed `try_spec`, the census, and the partial report
   entirely, so a green rejection test and a dead search were fully compatible — exactly the "clean
   typed diagnosis that still kills the run" the gh-538 sweep reported. Two things made it invisible
   to the design: the baseline is the one candidate not compiled *as* a candidate, and supplying a
   `?lowered_transform` bypasses the default annotator, so the form being validated is the
   unscheduled serial one, which is the worst case for per-work-item scratch and is never even
   dispatched on GPU. Fixed by protecting that compile too (provenance `Candidate`); the capture
   happens inside the transform, so the base lowering survives the rejection and the scheduled
   candidates — whose fission and placement promotion are what bring the head back within budget —
   still derive from it. The general lesson matches the seam note under step 1: audit for compiles
   the tuner performs *outside* its candidate loop.
1. `schedule_outcome.ml`: phases, causes/keys, provenance policy, fatal-with-backtrace, strict mode,
   `Context.compile_outcome`, the default backend hook, and narrow translations at
   schedule-apply/`validate_parallel`/hardware-limit/compiler/link seams. Land the portable
   policy/key tests while public `Context.compile` remains behavior-compatible.
   **Follow-up seam found afterwards:** `Autotune.sketch_schedule` is a phase boundary of its own,
   and the inventory above missed it. Its `invalid_arg`s are applicability preconditions ("no
   matmul site detected"; a companion nest the family cannot cover), so untyped they were
   unclassified and fatal, and one inapplicable GPU sketch family ended the whole search —
   reproducible on Metal with `test/operations/autotune_fission_sketch`, invisible on `cc`. Typed at
   the sketch construction, not around the transform closure: an arbitrary exception escaping a
   transform must stay fatal. Worth expecting more of these wherever the tuner's own code signals
   applicability with `invalid_arg`; the checked-in test configs only exercise `cc`.
2. Typed candidate/advisory/cache consumers, the decline census and partial report,
   `bench_mlp`, and retirement of `validate_segments` if failed-compile cost is acceptable. Land the
   `cc` containment/advisory/backtrace tests.
   **Implementation note:** the public/general `validate_segments` helper was retired, but a
   model-ranking-only eager validator remains. Removing that check made an invalid tensorized
   argmin displace a viable model candidate and then fall back all the way to the default (the
   `cost_model_selection` regression demonstrates that the cost is semantic, not merely compile
   time). Attribution and advisory fallback still use the typed codegen boundary; the retained
   check exists only to rank the best viable uncompiled contender.
3. On HIP, identify the actual private-segment limit, extend hipjit, add the backend-private
   post-link validator, and land the gated "rejected before launch, following reduction correct"
   test. Add CUDA diagnostics/validation only for API-supported limits. Keep Metal's unused static
   threadgroup-memory cross-check as a separate cleanup commit.
   **Landed, Metal half only.** Metal's post-link `maxTotalThreadsPerThreadgroup` check raised a
   bare `Utils.User_error`, so under strict classification one register-heavy candidate was fatal
   rather than declined; it is now a typed `Resource_exceeded Workgroup_threads` at `Backend_link`.
   `get_static_threadgroup_memory_length` is now called, checked against the same device limit the
   schedule layer estimated against. The HIP private-segment quantity still needs its machine.
   **CUDA arm landed, classification only — the validator half is blocked on the bindings.**
   `Cuda_backend.Impl.classify_failure` maps the driver's and nvrtc's own status codes to typed
   causes, split on the damage axis: statuses returned before any thread runs
   (`CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`, `..._OUT_OF_MEMORY`, `..._COOPERATIVE_LAUNCH_TOO_LARGE`,
   `..._INVALID_CLUSTER_SIZE`, `..._INVALID_VALUE` at `Launch`, the PTX/image rejections, and
   `NVRTC_ERROR_COMPILATION`) are `No_device_writes` contained declines; the asynchronous fault
   family (`..._ILLEGAL_ADDRESS`, `..._MISALIGNED_ADDRESS`, `..._INVALID_ADDRESS_SPACE`,
   `..._INVALID_PC`, `..._ILLEGAL_INSTRUCTION`, `..._HARDWARE_STACK_ERROR`, `..._ASSERT`,
   `..._LAUNCH_FAILED`, `..._LAUNCH_TIMEOUT`) is `Writes_may_have_occurred` — counted as a decline,
   then escalated, since those leave the CUDA context sticky. Environment and toolchain faults stay
   unclassified on purpose: they fail every candidate identically, and absorbing them would turn a
   broken installation into a silent "nothing worked" report. Every case is `Backend_rejected`
   rather than `Resource_exceeded`, per this note's own rule — the classifier sees the status, not
   the launch geometry, so it cannot populate `requested`/`limit`.
   The pre-launch validator has **no API-supported condition to reject against on CUDA today**.
   Its query would be `cuFuncGetAttribute` on the linked `CUfunction`
   (`CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK`, the direct analogue of Metal's
   `maxTotalThreadsPerThreadgroup`; `..._SHARED_SIZE_BYTES`; `..._LOCAL_SIZE_BYTES`), and cudajit
   binds none of them — `Cu.Module.get_function` returns an opaque handle and `Cu.Module` exposes no
   attribute query. Extending the bindings (`lukstafi/ocaml-cudajit`, already a pin) is the work.
   Note what it buys here and what it does not: a better message and a `Resource_exceeded` with
   populated fields, **not** more containment, because the condition it would predict —
   `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES` — is non-sticky and write-free and is already contained by
   classification. CUDA's uncontainable failures are the memory-fault family, which no function
   attribute predicts. That asymmetry with HIP (where #533's abort is uncontainable *and*
   predictable) is why the validator is lower priority on this backend than on that one.
4. Make runner replacement an explicit backend operation, implement
   `recover_after_launch_failure`, add ledger rollback/poisoning and the fake recovery-state tests,
   then tighten the launch/sync arm. Exercise HIP and CUDA recovery on their machines.
   **Partly landed: everything except the recovery hook.** The launch/sync arm now passes the real
   `Context.failure_classifier` instead of an always-`None` classifier — without that a backend
   could never attribute a launch failure, and the HIP arm had nowhere to plug in.
   `classified_cause` carries the phase it was raised at, so a report separates a link rejection
   from a launch refusal from an asynchronous failure at sync. A contained `No_device_writes`
   rejection rolls back `Context.run`'s optimistic execution marking; a possibly-writing rejection
   or an unattributed fatal poisons the execution lineage, and every entrypoint on it then refuses,
   naming the routine and the original failure.
   `recover_after_launch_failure` is deliberately **not** implemented yet. Nothing would consult
   it under the current policy — a contained rejection continues without needing it, and everything
   else is fatal — and its central claim ("`Recovered` commits to a usable state") is exactly the
   empirical question the HIP machine has to answer. Adding mutable runners and per-backend stubs
   before then would be untested infrastructure. It becomes worth doing once step 3's HIP validator
   exists and a launch failure can actually be attributed to a candidate.
5. Replace `Context.auto`'s blanket catches with `Backend_unavailable` and its focused tests.
   Remove compatibility mode once supported-backend censuses contain no `Unclassified_key`.
   **Landed.** Device discovery raises `Ir.Backend_intf.Backend_unavailable` when the backend's
   library is not linked in or the driver reports no devices; only that advances to the next
   backend. A bad ordinal stays a caller error, a driver-initialization failure propagates, and the
   configured arm no longer relabels an unusable backend as an unknown name. Compatibility mode
   stays for now.

Steps 3 and 4 need hardware this machine does not have for their HIP/CUDA acceptance tests. Steps 1,
2, and 5 are portable; Metal and `cc` cover the generic compiler/link and report paths.

**Where a wall-clock budget would attach (ahrefs/ocannl#532).** #532 asks whether these typed
outcomes should also carry a time budget on candidate execution — the same containment problem in a
different currency. They should not carry it as a `cause`: a timeout is not evidence about the
candidate's validity, and #532's own preferred fix is cheaper and earlier (do not time an
unparallelized candidate on a GPU backend at all). But the seam is the one this design already
installed: the `protect` around `time_routine` in `try_spec` is the single place where a launch is
bounded, and a budget overrun would arrive there as a `Classified` rejection with
`Writes_may_have_occurred` — a dispatch that was abandoned rather than completed has, by
construction, written whatever it wrote. That already escalates rather than silently scoring the
next candidate on the same lineage, so the two mechanisms compose without a second policy.
