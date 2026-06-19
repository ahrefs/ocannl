# Fix ocannl_config.reference: inline_complex_computations default is true, not false

## Goal

`ocannl_config.reference` documents `inline_complex_computations=false`, but the
authoritative runtime default in `arrayjit/lib/low_level.ml` is `~default:true`.
The reference file therefore misleads readers about actual default behaviour.

This is stale documentation left behind by the CSE work (gh-ocannl-351, done
2026-03-25). The flag was originally gated off (`~default:false`, with a
`TODO(#351): change to true once CSE is implemented` comment) until Common
Subexpression Elimination existed. Commit `bf2f1daa` (CSE, #351) flipped the
code default to `~default:true` and removed the code-side TODO, but missed the
documentation and adjacent FIXME comments that still describe the pre-CSE world.

Relates to gh-ocannl-134 (whose coder retrospective surfaced the drift) and
gh-ocannl-351 (the CSE work that caused it).

## Acceptance Criteria

- [ ] `ocannl_config.reference` documents `inline_complex_computations=true`,
  matching the `~default:true` in `low_level.ml`'s `virtualize_settings`
  binding. Verifiable by:
  `grep -n inline_complex_computations ocannl_config.reference` → shows
  `inline_complex_computations=true`.
- [ ] The stale `# FIXME(#351): avoid excessive inlining while CSE is not
  implemented` line immediately preceding the flag in
  `ocannl_config.reference` is removed (CSE is implemented). The remaining
  comment block describing the `max_visits` accounting behaviour is preserved.
  Verifiable by:
  `grep -n 'FIXME(#351)' ocannl_config.reference` → no match.
- [ ] The obsolete `(* FIXME(#351): avoid excessive inlining while CSE is not
  implemented *)` comment in `arrayjit/lib/low_level.ml` (in `visit_llc`, near
  the `is_too_many` helper) is removed. Verifiable by:
  `grep -n 'FIXME(#351)' arrayjit/lib/low_level.ml` → no match.
- [ ] The authoritative code default
  (`Utils.get_global_flag ~default:true ~arg_name:"inline_complex_computations"`
  in `virtualize_settings`) is **unchanged** — this is the source of truth the
  doc is being synced to, not the thing being edited.

## Context

How things work now:

- **`ocannl_config.reference`** (repo root) is the canonical, documented
  configuration reference (renamed from `ocannl_config.example` by
  gh-ocannl-409). Its `inline_complex_computations` stanza currently reads:
  ```
  # If true, virtualize_max_visits only counts accesses that are not used for assignment of
  # the same cell (typically accumulation). Otherwise, all accesses are counted, so computations
  # that reduce an axis will rarely be inlined.
  # FIXME(#351): avoid excessive inlining while CSE is not implemented
  inline_complex_computations=false
  ```
  Both the `FIXME(#351)` line and the `=false` value are stale. The preceding
  comment block (the `max_visits` accounting description) is accurate and should
  remain.

- **`arrayjit/lib/low_level.ml`**, `virtualize_settings` let-binding — the
  authoritative default:
  `Utils.get_global_flag ~default:true ~arg_name:"inline_complex_computations"`.
  This is correct and must not change. A separate obsolete
  `(* FIXME(#351): avoid excessive inlining while CSE is not implemented *)`
  comment lives in `visit_llc` (just above the `is_too_many` helper) and is the
  same drift — CSE shipped, so it should be removed.

- gh-ocannl-351 (CSE) is complete; its acceptance criteria explicitly included
  "Enable `inline_complex_computations = true` as the default." So syncing the
  doc to `true` reflects the intended steady state, not a behaviour change.

## Approach

*Suggested approach — agents may deviate if they find a better path.*

A pure doc-and-comment sync, no behaviour change:

1. In `ocannl_config.reference`: delete the `# FIXME(#351): ...` line and change
   `inline_complex_computations=false` to `inline_complex_computations=true`.
   Keep the preceding `max_visits` accounting comment block.
2. In `arrayjit/lib/low_level.ml`: delete the obsolete
   `(* FIXME(#351): avoid excessive inlining while CSE is not implemented *)`
   comment in `visit_llc`. Leave the `~default:true` binding untouched.

## Scope

In scope:
- `ocannl_config.reference` — value fix + stale FIXME removal.
- `arrayjit/lib/low_level.ml` — removal of the obsolete `FIXME(#351)` comment
  only (no logic change).

Out of scope:
- `arrayjit/lib/low_level.ml` `virtualize_settings` default — must stay
  `~default:true`.
- `test/einsum/inline_permuted_view.ml` — sets
  `inline_complex_computations <- false` as a deliberate per-test override;
  removing it would change test behaviour. Leave it alone.
- `test/operations/micrograd_demo_logging.ml` `(* FIXME(#351): this is a good
  test for common subexpression elimination. *)` — a prose note about a
  potential future test, not stale config drift; not part of this doc-sync and
  left untouched.

Dependencies: none. gh-ocannl-351 (the CSE work this re-syncs to) is already
complete.
