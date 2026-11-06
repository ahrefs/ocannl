# Disposition: JS/WASM target + WebGPU backend (gh-ocannl-123)

## Goal

Resolve the long-standing v1.1 issue [#123](https://github.com/ahrefs/ocannl/issues/123)
("Work out a JS or WASM compilation target with a WebGPU-based backend") which
the user has already personally deprioritized on the GitHub issue itself. The
purpose of this proposal is *disposition*, not design: pick a path that stops
the issue from drifting in the v1.1 backlog as a contradictory signal.

## Acceptance Criteria

This proposal is a memo recommending closure (or another disposition the user
prefers). It is satisfied when one of the following has happened:

- [ ] **(A — recommended)** Issue #123 is closed as `not-planned`, with a
  closing comment that quotes the user's 2025-09-23 deprioritization for the
  audit trail. Task `gh-ocannl-123` is marked abandoned in the harness.
- [ ] **(B)** Issue #123 stays open with a `defer` / `v2.x` label and a
  brief doc memo (`docs/proposals/gh-ocannl-123-deferred.md`) summarizing
  the prerequisites enumerated in *Context* below.
- [ ] **(C)** Issue #123 is split into two narrower tracking issues — one
  for js_of_ocaml host support (which entails ctypes removal from the core
  `ir`/`context` libraries) and one for a WebGPU backend — and #123 is
  closed pointing at them.
- [ ] **(D)** Status quo: issue stays open as v1.1-explore. Selecting this
  re-opens the question of why the user's own 2025 comment did not close it.

The user picks. The recommendation is (A).

## Context

### Why this proposal is short

Per the orchestrator brief, this is the cleanest stale-task case in the
current OCANNL v1.1 backlog. The exploration here deliberately stops at the
disposition level: enumerating WGSL shader-gen patterns or ctypes-removal
architecture is out of scope until the user signals interest in revisiting.

### Historical signal on the issue itself

- 2024-12-05 — lukstafi: *"Adding this to priority backends: cc, cuda, metal,
  webgpu."*
- 2025-09-23 — lukstafi: *"Removing webgpu from priority backends (currently:
  cc, cuda, metal, hip-amd)."*

Net: one user-driven full reversal already landed on the issue. The task body
in `tasks/gh-ocannl-123.md` mirrors this with a `Status: DEPRIORITIZED`
subsection.

### Verified state of the codebase (as of this proposal)

1. **Priority-backend list reality check.** The 2025-09-23 comment names
   `cc, cuda, metal, hip-amd`. Of those:
   - `cc_backend.ml`, `cuda_backend.ml`, `metal_backend.ml` exist in
     `arrayjit/lib/`. CUDA and Metal libraries are `(optional)` selects on
     `cudajit` / `metal` packages in `arrayjit/lib/dune`.
   - **HIP is aspirational, not implemented.** `ROADMAP.md` line 33 lists
     "HIP backend (#411) — parallel / background task, can slip to v0.6.4"
     with "Standalone bindings package for HIP (AMD hardware)". No
     `hip_backend.ml` or HIP-related modules exist in `arrayjit/lib/`. Worth
     flagging in passing — the user's 2025-09-23 comment treats `hip-amd`
     as a current priority backend, but it is currently a roadmap entry only.
     Not in scope here; mention only so the user can decide whether to file
     a follow-up to clarify the priority list.
   - No `webgpu`, `wgsl`, `js_of_ocaml`, or `wasm_of_ocaml` references
     exist in `arrayjit/`, `lib/`, `tensor/`, `bin/`, or `docs/`. No partial
     work to preserve.

2. **ctypes is a hard dependency of the core, not just non-priority
   backends.** `arrayjit/lib/dune` declares `ctypes` and `ctypes.foreign`
   as direct (non-optional, non-select) dependencies of two core libraries:
   - `ir` library (lines 31–32) — modules: `ops`, `ndarray`, `indexing`,
     `tnode`, `low_level`, `assignments`, `task`, `backend_intf`,
     `backend_impl`, `c_syntax`.
   - `context` library (lines 106–107) — the runtime composition layer that
     `select`s the cuda/metal backend implementations.

   `dune-project` declares `(using ctypes 0.3)` and lists `ctypes` /
   `ctypes-foreign` as opam dependencies (lines 44–46). ctypes is also
   transitively required via `cudajit`, `metal`, and `ppx_minidebug`.

   **Implication:** js_of_ocaml host support is not a backend-selection
   problem. It requires removing `ctypes` from the core `ir` and `context`
   libraries before any browser target is feasible. That refactor is the
   real prerequisite, and it is not motivated by browser deployment alone —
   it would also affect how every existing backend interacts with the IR.

### Strategic context

- Active milestones (per `ROADMAP.md` and harness memory):
  - v0.7: streams (#68), hosted tensor (#70), RoPE (#65), transformer (#66).
  - v0.7.1: tokenizers (#73), GPT-2 inference (#74).
  - v0.8: loop hoisting (#76), CPU tiling (#79), CUDA perf (#80), llm.c (#81).
  - v0.8 GPU performance milestone starts from a `grid_dim=1, block_dim=1`
    baseline — non-trivial perf work ahead before any new-backend story.
- Conference targets: ICFP 2026 OCaml Workshop / FProPer paper deadlines
  May–June 2026.
- Browser deployment is not on any near-term roadmap.
- User has paused autonomous OCANNL work pending a hands-on quality audit.

### Why (A) is recommended

- The strongest possible historical signal exists: the user's own comment
  on the issue, removing webgpu from priority backends.
- No partial work would be lost (zero references in code/docs).
- The real prerequisite (ctypes removal from core `ir`/`context`) is
  unmotivated by any current goal and would be a major refactor justified
  only by browser deployment, which the user has explicitly deprioritized.
- The OCANNL v1.1 backlog has multiple stale entries (3 of the 5 prior
  v1.1 explore tasks this Mag session ended in close-as-not-planned
  recommendations); leaving #123 open contradicts the user's own comment.

## Scope

**In scope (this memo):**
- Recommend a disposition for issue #123.
- Surface one ancillary observation: HIP is roadmapped, not implemented,
  while the user's 2025 comment names it as a priority backend.

**Out of scope:**
- Any js_of_ocaml or WebGPU design work.
- ctypes-removal architecture for `ir` / `context`.
- WGSL shader-generation patterns.
- Filing the HIP-priority-clarification follow-up (mention only).
