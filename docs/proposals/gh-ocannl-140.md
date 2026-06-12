# Partition the Benchmark Table by `result_label`

## Status update (2026-06-12)

- Issue [#140](https://github.com/ahrefs/ocannl/issues/140) is still OPEN, milestone v1.0 (due 2026-06-30).
- Not yet started: re-verified at HEAD `d9de22f0` — `tensor/PrintBox_utils.ml` still has the `TODO(#140)` comment at line 119 and the `List.hd_exn result_labels` bug at line 129; `let table` still starts at line 102.
- `bin/compilation_speed.ml:69` remains the single call site, unchanged.
- No commits touched `tensor/PrintBox_utils.ml` or `bin/compilation_speed.ml` since April 2026; none of the proposal's assumptions are invalidated.
- The repo-wide broadcast-order reversal (LUB→GLB) and basis-rename refactors did not affect this file.
- Everything in the Approach section remains to do: grouping, per-group `Speedup`/`Mem gain`, stacking, tests.

## Goal

Refactor `PrintBox_utils.table` so that benchmark rows with different `result_label` values are grouped, with each group rendered as its own record (and its own results-column header) stacked vertically. Today the function uses `List.hd_exn result_labels` as the results-column header for every row regardless of the row's actual label, and computes `Speedup` / `Mem gain` against a single global maximum that mixes incomparable measurements when labels disagree.

Tracked by: https://github.com/ahrefs/ocannl/issues/140

## Acceptance Criteria

- **Multi-label grouping.** Calling `table` on a row list containing two distinct `result_label` values (e.g., a row with `result_label = "ms"` followed by a row with `result_label = "MB"`) renders output that contains *both* labels as separate column headers in separate sub-tables. The rendered text (e.g., via `PrintBox_text.to_string`) must contain both label strings as headers, not just the first one. *Falsifier:* reverting to `List.hd_exn` would put a single label as the header for every row; the second label would never appear, and the assertion would fail.
- **Single-label backward compatibility.** Calling `table` on a row list where every row shares one `result_label` produces output structurally equivalent to the current behavior: a single `frame`-wrapped `record` with the six existing columns (`Benchmarks`, `Time in sec`, `Memory in bytes`, `Speedup`, `Mem gain`, `<result_label>`). The number of grouped sub-records is exactly one. *Falsifier:* a refactor that always emits `vlist [single_record]` (an extra layer) or that strips the outer `frame` would fail this comparison; a refactor that re-runs the global aggregation when only one group exists is fine.
- **Per-group `Speedup` and `Mem gain`.** With two groups -- group A of three rows with times `[1.0; 2.0; 4.0]` and group B of three rows with times `[10.0; 20.0; 40.0]` -- the `Speedup` column for A's rows is computed against `max A.times = 4.0` (yielding `[4.0; 2.0; 1.0]`), and B's against `max B.times = 40.0` (yielding `[4.0; 2.0; 1.0]`). The same applies to `Mem gain` against per-group `max mem_in_bytes`. *Falsifier:* a global-max implementation would compute B's speedups against `40.0` (correct for B) but A's against `40.0` too (giving `[40.0; 20.0; 10.0]`), failing the A-side check.
- **Empty input still returns `PrintBox.empty`.** Unchanged from current behavior. *Falsifier:* a refactor that always runs `List.reduce_exn` on `times` would crash on empty input.
- **Call-site compatibility.** `bin/compilation_speed.ml` continues to compile and produce output. No change to the type signature `table : table_row_spec list -> PrintBox.t` is required; if one is made, the call-site must be updated in the same commit. *Falsifier:* a signature change that breaks `dune build bin/compilation_speed.exe` would be caught by CI.
- **Unit tests.** A new test file (`test/operations/printbox_utils_test.ml` or analogous, plus a `dune` stanza) exercises the four behavioral ACs above by constructing `Benchmark` records directly and inspecting the rendered string output of `PrintBox_text.to_string`.

## Context

### OCANNL audit pause

Per harness memory, autonomous OCANNL work is paused while the user does a hands-on quality audit. This proposal will be authored and committed but the task will defer launch -- no slot will start work without explicit approval.

### Current state

`tensor/PrintBox_utils.ml`, lines 102--130 (verified at HEAD `03116b23`; re-verified unchanged at `d9de22f0`, 2026-06-12):

```ocaml
let table rows =
  if List.is_empty rows then PrintBox.empty
  else
    let titles = List.map rows ~f:(fun (Benchmark { bench_title; _ }) -> nolines bench_title) in
    let times = List.map rows ~f:(fun (Benchmark { time_in_sec; _ }) -> time_in_sec) in
    let sizes = List.map rows ~f:(fun (Benchmark { mem_in_bytes; _ }) -> mem_in_bytes) in
    let max_time = List.reduce_exn ~f:Float.max times in
    let max_size = List.reduce_exn ~f:Int.max sizes in
    let speedups = List.map times ~f:(fun x -> max_time /. x) in
    let mem_gains = List.map sizes ~f:Float.(fun x -> of_int max_size / of_int x) in
    ...
    let result_labels =
      List.map rows ~f:(fun (Benchmark { result_label; _ }) -> nolines result_label)
    in
    (* TODO(#140): partition by unique result_label and output a vlist of records. *)
    PrintBox.(
      frame
      @@ record
           [
             ("Benchmarks", vlist_map ~bars:false line titles);
             ...
             (List.hd_exn result_labels, vlist_map ~bars:false line results);
           ])
```

The TODO at line 119 and the `List.hd_exn` bug at line 129 are still present.

### Call sites

`grep -rn 'PrintBox_utils.table'` finds exactly one call site:

```
bin/compilation_speed.ml:69:  |> PrintBox_utils.table |> PrintBox_text.output Stdio.stdout
```

`bin/compilation_speed.ml` builds a list of `Benchmark` records (currently with `result_label = "x, f(x)"` for the only enabled benchmark) and pipes the resulting box to `PrintBox_text.output`. The signature of `table` is `table_row_spec list -> PrintBox.t` and the call-site treats it as such; no surrounding code inspects the box's structure.

### Available primitives

`printbox` is pinned to `>= 0.12` in `dune-project`, and the installed version is `0.12`. The relevant combinators exist:

- `PrintBox.vlist : ?bars:bool -> t list -> t` -- vertical stack (with optional separators).
- `PrintBox.record : (string * t) list -> t` -- already used.
- `PrintBox.frame : t -> t` -- already used.

`PrintBox.vlist ~bars:true` yields visible separators between groups; `~bars:false` is a clean stack. Both render correctly under `PrintBox_text`, `PrintBox_html`, and `PrintBox_md`.

### Grouping primitive

`Base.List.group ~break:(fun a b -> not (String.equal (label a) (label b)))` preserves the input order and chunks runs of consecutive equal-labeled rows. `Base.List.Assoc.group_by` does not exist, but `List.fold` accumulating into a `Map.Poly` of `string -> row list` (then `Map.to_alist` to recover the alist) is a standard alternative that *re-orders* groups alphabetically. See "Ambiguities" below.

### Related work

- **gh-ocannl-103** (plot legends and axis tick marks): touches the same file but a different function; merge conflict is unlikely but worth flagging if both ship in parallel.

## Approach

1. **Group rows by label** using `List.group` with the consecutive-break comparator (preserves the order in which the user passed rows). The result is `(string * Benchmark list) list` after extracting the shared `result_label` from each chunk's head.

2. **Per-group rendering.** For each `(label, group_rows)` pair, replicate the existing column construction but scoped to the group: per-group `max_time`, `max_size`, `speedups`, `mem_gains`. Build one `PrintBox.record` per group with `(label, vlist_map ~bars:false line group_results)` as the sixth column; the first five columns are unchanged.

3. **Stack groups.** If there is exactly one group, return `frame (record [...])` -- byte-identical to current single-label output. If there are multiple groups, return `frame (vlist ~bars:true [record_g1; record_g2; ...])` (or `~bars:false` -- see Ambiguities). The outer `frame` is preserved so external consumers see a single bordered box.

4. **Extract a helper.** A private `let render_group : string -> table_row_spec list -> PrintBox.t = fun label rows -> ...` keeps the per-group logic separate and makes the test surface cleaner.

5. **Tests.** Add `test/operations/printbox_utils_test.ml` (and the corresponding `dune` stanza). Construct `Benchmark` records inline, render via `PrintBox_text.to_string`, and assert on substring presence and absence (label headers, expected `Speedup` numerics for the per-group AC). Tests should not depend on exact whitespace -- match on the column headers and the formatted speedup values (`%.3f`).

6. **Verify call-site.** Run `dune build bin/compilation_speed.exe` and visually inspect `dune exec bin/compilation_speed.exe` output if a backend is available. The single-row, single-label case must look unchanged.

### Risk notes

- `List.group` from `Base` chunks *consecutive* equal elements only. If a caller interleaves rows by label, the user will get multiple sub-tables for the same label. This is arguably what the user wanted -- they chose the row order -- but if it becomes a problem, switching to `Map`-based grouping (alphabetic order, one chunk per label) is a one-line change.
- `frame (record [...])` and `frame (vlist [record [...]])` may render differently in `PrintBox_text` (the latter could add a blank line). The single-label backward-compatibility AC pins the cheaper branch (`frame (record [...])`).
- No interaction with `printbox-ext-plot` -- `table` is independent of plotting.

## Ambiguities

1. **Group ordering.** Preserve the row-insertion order via `List.group` (cheap, matches user intent), or sort groups alphabetically by label via a `Map`-based grouping (deterministic for golden tests, but reorders the user's data)? Recommendation: insertion order. Defer to user if golden-test stability is preferred.

2. **Visual separation between groups.** `vlist ~bars:true` draws a horizontal divider between sub-records; `~bars:false` stacks them with no visible boundary. Recommendation: `~bars:true` for clarity since the groups have semantically different result columns. Single-element vlist falls back to "just the record" in either case.

3. **Empty-group / single-row-group handling.** A `result_label` value with zero rows is impossible by construction (groups come from non-empty row lists), but a single-row group will produce `Speedup = 1.000` and `Mem gain = 1.000` (the row is its own max). This is arguably noisy; an alternative is to suppress those columns when a group has exactly one row. Recommendation: leave the columns in -- removing them would break the column-set invariant the AC depends on. Defer to user if undesirable.
