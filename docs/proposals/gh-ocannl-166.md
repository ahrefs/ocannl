# Diagnose RTX 3050 compute_mode = 1 (uncategorized)

## Status update (2026-06-12)

- Issue [ahrefs/ocannl#166](https://github.com/ahrefs/ocannl/issues/166) is still OPEN, milestone v0.8 (ROADMAP target: mid-June 2026 — the GitHub milestone due date of Feb 2026 lags ROADMAP.md, which is authoritative).
- No fix has landed in `lukstafi/ocaml-cudajit` (HEAD `16c61b6`, post-0.7.2 release): `computemode_of_cu` still raises `invalid_arg` on `CU_COMPUTEMODE_UNCATEGORIZED` (`src/cuda.ml:234`), so a driver-returned `1` would crash `properties.exe` rather than print.
- All cited locations re-verified against the current ocaml-cudajit checkout: `type computemode` at `src/cuda.ml:179`, `type cu_computemode` at `cuda_ffi/bindings_types.ml:489`, ctypes constants at `cuda_ffi/bindings_types.ml:1555-1557`. Nothing in the proposal's code analysis is stale.
- No work in the ocannl repo since April 2026 touches this task; it remains blocked on physical access to the RTX 3050 desktop.
- Everything in the Approach section remains to do.

## Goal

Determine why `dune exec cudajit/bin/properties.exe` on the RTX 3050 desktop reports
`compute_mode = 1`, a value that does not match any constant in CUDA's `CUcomputemode`
enum (`DEFAULT=0`, `PROHIBITED=2`, `EXCLUSIVE_PROCESS=3`).

Source: https://github.com/ahrefs/ocannl/issues/166

This is a hardware-dependent diagnostic — it requires the RTX 3050 desktop. The fix
(if any) lands in [`lukstafi/ocaml-cudajit`](https://github.com/lukstafi/ocaml-cudajit),
not in `ahrefs/ocannl`. The proposal lives in ocannl-staging because the issue is
filed there.

## Acceptance Criteria

- [ ] The cause is identified as either (A) system configuration, (B) an
  ocaml-cudajit binding/printing issue, or (C) a CUDA driver oddity / legacy value.
- [ ] If (A) — system config: `nvidia-smi -q | grep -A1 "Compute Mode"` is recorded
  in the issue, the appropriate `nvidia-smi --compute-mode=DEFAULT` (or equivalent)
  step is documented, and the issue is closed with the resolution.
- [ ] If (B) — binding/printing issue: a fix PR is opened against
  `lukstafi/ocaml-cudajit` adjusting the affected enum mapping, sexp printer, or
  raw-int printing path; the fix is verified on the RTX 3050.
- [ ] If (C) — driver oddity (no fix possible from our side): the finding is
  documented on the issue and the issue is closed.

## Context

### Where compute_mode is reported

`bin/properties.ml` prints `Cu.Device.sexp_of_attributes props`, which in turn
relies on `sexp_of_computemode` defined in `src/cuda.ml`. The high-level type is:

```ocaml
type computemode = DEFAULT | PROHIBITED | EXCLUSIVE_PROCESS
```

with `sexp_of_computemode` mapping each constructor to a symbolic atom — it does
*not* print an integer. So a printed `1` cannot come out of this code path
verbatim; the value `1` would only appear if either:

1. The ctypes view in `cuda_ffi/bindings_types.ml` reports the raw value via the
   `unexpected` callback `CU_COMPUTEMODE_UNCATEGORIZED 1`, *and* `computemode_of_cu`
   in `src/cuda.ml` raises `invalid_arg` on it — which would crash, not print "1".
2. An older revision of the binding printed the raw int, or the user copied the
   attribute integer from a different output line (e.g. if attribute lookup was
   invoked directly without enum conversion at the time the issue was filed).

This needs to be reproduced first before assuming a code-side bug.

### Key files

- ocaml-cudajit `bin/properties.ml` — entry point; uses `Cu.Device.get_attributes`
  and prints via `sexp_of_attributes`.
- ocaml-cudajit `src/cuda.ml` — high-level OCaml API:
  - `type computemode` (line ~179), `sexp_of_computemode`,
  - `computemode_of_cu` which raises `invalid_arg "Unknown computemode: <i>"` on
    `CU_COMPUTEMODE_UNCATEGORIZED i`,
  - the `compute_mode` field of `Device.attributes` populated in
    `Cu.Device.get_attributes` via `cu_device_get_attribute … COMPUTE_MODE` then
    `cu_computemode_of_int`.
- ocaml-cudajit `cuda_ffi/bindings.ml` — `cu_computemode_of_int` matches the raw
  int against `E.cu_computemode_default / _exclusive_process / _prohibited`,
  falling back to `CU_COMPUTEMODE_UNCATEGORIZED`.
- ocaml-cudajit `cuda_ffi/bindings_types.ml` —
  - `type cu_computemode` constructors (~line 489): `DEFAULT | PROHIBITED |
    EXCLUSIVE_PROCESS | UNCATEGORIZED of int64`,
  - the ctypes constants `cu_computemode_default / _prohibited /
    _exclusive_process` (~line 1555) are pulled from the CUDA headers at C-stub
    generation time, so source-level reordering of OCaml constructors cannot
    misalign them.

### CUDA reference

Per CUDA driver API: `CU_COMPUTEMODE_DEFAULT = 0`, `CU_COMPUTEMODE_PROHIBITED = 2`,
`CU_COMPUTEMODE_EXCLUSIVE_PROCESS = 3`. Value `1` was historically
`CU_COMPUTEMODE_EXCLUSIVE` (single-context per-device), removed in CUDA 8.0.
Modern drivers should not return 1 for any device.

## Approach

*Suggested approach — agents may deviate if they find a better path.*

Run on the RTX 3050 desktop in this order, stopping when the cause is clear:

1. **Capture system state**:
   ```bash
   nvidia-smi -q | grep -A1 "Compute Mode"
   nvidia-smi --query-gpu=compute_mode --format=csv
   nvidia-smi --version
   ```
   If `Compute Mode` is `Default`, the kernel sees 0 — anything else is the
   smoking gun and points at branch (A).

2. **Reproduce** with the current ocaml-cudajit:
   ```bash
   cd ~/ocaml-cudajit && dune exec bin/properties.exe
   ```
   - If it crashes with `Unknown computemode: 1`, the binding *did* receive 1
     from the driver — branch (C). Document and consider whether to soften
     `computemode_of_cu` (replace `invalid_arg` with a non-fatal sexp like
     `(UNCATEGORIZED 1)`) so `properties.exe` prints something useful instead of
     crashing.
   - If it prints `(compute_mode DEFAULT)` and no "1" appears anywhere, the
     original report was against an older binding revision and is stale —
     close the issue.
   - If "1" still appears but is not a crash, locate the exact print site to
     decide between (B) and (C).

3. **Cross-check raw attribute** with a tiny C program (or by editing
   `properties.ml` to print the raw int alongside the enum) to confirm whether
   the driver itself is returning 1 or whether it's a printing artifact.

4. **Decision branch**:
   - **(A) System config**: run `sudo nvidia-smi --compute-mode=DEFAULT`, rerun
     `properties.exe`, document on the issue, close.
   - **(B) Printing/binding issue in ocaml-cudajit**: open a PR against
     `lukstafi/ocaml-cudajit` with a minimal fix (most likely: turn
     `invalid_arg` in `computemode_of_cu` into a non-fatal `UNCATEGORIZED i`
     surface variant + sexp printer, so legacy or unknown driver values do not
     break `properties.exe`). Verify on the RTX 3050.
   - **(C) Driver oddity**: document and close.

The (B) softening is worth doing regardless once we have the data, because
silently crashing on an unrecognised compute mode is bad UX for a diagnostic
binary — but that is a follow-up, not the focus of this task.

## Scope

**In scope**:
- Diagnose the `compute_mode = 1` report on the RTX 3050.
- If a binding fix is warranted, open a PR against `lukstafi/ocaml-cudajit`.
- Update issue ahrefs/ocannl#166 with findings and close it.

**Out of scope**:
- Broader cleanup of other CUDA attribute enums (separate task if desired).
- Any work that does not require the RTX 3050 hardware.

**Dependencies**: requires physical access to the RTX 3050 desktop with a
working CUDA driver and the ocaml-cudajit checkout buildable there.

**PR destination**: any fix lands in `lukstafi/ocaml-cudajit`, not
`ahrefs/ocannl` and not `ocannl-staging`.
