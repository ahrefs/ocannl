(** The anti-degeneracy machinery the matmul-shaped benches share (gh-ocannl-711): an aperiodic mix
    of a (row, column) pair, the operand values minted from it, and the position-weighted
    whole-output checksum that is each bench's correctness guard.

    Two benches ([bin/schedule_bench.ml], [bin/narrow_gebp_bench.ml]) time several schedules of the
    same matmul against each other and compare a checksum of the whole output, because a single spot
    cell cannot see a remainder region and a mishandled edge peel is exactly what these schedules
    risk. A guard that cannot see a row permutation is worse than none: the bench then reports a
    fast WRONG schedule as a win, and the autotuner learns from those reports. Three ways for it to
    go blind, all of which have been live in one copy or the other:

    - {b A residue of the FLATTENED offset.} [t = i * row_stride + j] loses its row dependence
      precisely when the modulus divides the row stride: [1 + (t mod 251)] gives every row the
      identical weight vector [1 + j] at n = 251, 502, 753, …, so a swapped pair of rows leaves the
      checksum unchanged — and the spot cell at [1][1] is blind to other rows at the same extent, so
      both halves of the guard fail together. Operand data collapses the same way: an mb of
      [(t mod 17) - 8] over a k x n array has every row identical whenever 17 divides n, and a
      schedule substituting the wrong row of a collapsed operand computes the right answer. {!mix}
      keys on the (row, column) PAIR, so the row index enters the value in its own right and no
      divisibility relation between a modulus and a stride can erase it. Mixing rather than a
      per-axis residue also removes the shift symmetry — any value drawn from [index mod p] repeats
      under [k -> k + p], so if both operands share that period every packed K panel is identical
      and a staging bug that repeats the wrong panel is invisible; the packing factors are user
      arguments, so no fixed period can be assumed coprime to them.

    - {b A weighted sum that cancels.} A checksum is a linear functional of the output, so a row
      permutation is missed whenever the value difference is orthogonal to the weight difference —
      by the weights colliding (a weight capped at [weight_cap] puts a row's weight vector in
      [weight_cap ^ row_stride] values, and at row_stride 2 one stream gives rows 9 and 363 the same
      weights), or by plain cancellation (at m = 719, row_stride 2, rows 240 and 718 differ by
      [-14; 14] against weight differences that are constant across both columns, so BOTH streams
      cancel). No bounded-weight scalar escapes that class — which is why the guard the benches
      assert on is {!first_difference}, an elementwise comparison against the reference variant, and
      the checksum is what they PRINT: a compact fingerprint for reading a table and comparing runs,
      not the thing that decides. The two streams of {!weight_salts} are what make the printed
      fingerprint worth reading.

    - {b A producer value that IS the accumulator's init.} An operand row of all zeros makes the
      corresponding output row all zeros, which is indistinguishable from a schedule that dropped
      that row entirely against a zero-initialized destination. A row of [row_stride] independent
      residues is all-zero with probability [levels ^ -row_stride] — likely at the narrow extents,
      and it is not the checksum's job to catch what the data hid. {!positive_level} mints the
      multiplicand strictly positive, so no row of it can be all-zero however narrow the row is.

    Both benches had their own copy of this, which is how the fixed version and the degenerate one
    came to sit one file apart. It lives here once instead;
    [test/operations/bench_checksum_discrimination] pins what it must discriminate, with the
    single-stream and flat-offset forms as the negative controls. *)

open Base

(** An aperiodic mix of two indices: no shift of either index is a symmetry, at any lag. Every
    intermediate is masked below 2^24, so nothing overflows a 63-bit int, the result is non-negative
    (so [%] on it is a true residue), and the sequence is reproducible by an external oracle. The
    salt lets one bench mint several independent streams — its operands and its checksum weights —
    from the same function.

    The mask sits on the FIRST line, and that placement is the whole of a defect this inherited from
    the copy it was lifted from. With it further down, the fold [x lxor (x lsr 13)] compressed a
    40-bit product into 24 bits, and it is GF(2)-linear: two rows' outputs then differed by
    [L (a*P lxor a'*P)], a value depending on neither the column nor the salt — so a row pair in
    [L]'s kernel was identical at EVERY column of EVERY stream, and no number of streams could tell
    those two rows apart. Eight such pairs sit below row 20000, the first being 5977 and 10232.
    Masking to 24 bits first removes the class rather than shrinking it: [a * 73856093] is injective
    mod 2^24 (the multiplier is odd), and everything after it is a bijection on 24 bits — an
    xor-shift, a multiply by an odd constant, an xor-shift — so distinct rows below 2^24 differ at
    every column, and distinct columns likewise. Provable, not swept.

    That is a guarantee about FULL values, not their low bits: bits 0--6 can alias whole rows and
    columns (gh-ocannl-1118). Do not mask them or take an even remainder directly; use {!residue},
    which draws even-modulus values from bit 7 upwards. Reduced values have a finite space, so their
    row/column distinctness still needs checking at the fixture's extent. *)
let mix ~salt a b =
  let x = a * 73856093 lxor (b * 19349663) lxor salt land 0xFFFFFF in
  let x = x lxor (x lsr 13) land 0xFFFFFF in
  let x = x * 1274126177 land 0xFFFFFF in
  x lxor (x lsr 7)

(** [residue ~salt ~row_stride ~modulus t] is [mix] of the (row, column) pair that the flat offset
    [t] denotes in a row-major array of the given row stride, reduced mod [modulus]. This is the
    call site to prefer over a hand-written [t % modulus]: same shape, no divisibility collapse.
    Non-negative, and below [modulus]. Odd moduli retain the full mix's residue. Even moduli use
    [mix lsr 7]: the raw low bits alias whole rows and columns, and an even remainder retains that
    parity defect. This keeps {!mix} and odd-modulus checksum weights unchanged; it does not promise
    distinct vectors at arbitrary extents in a bounded value space. *)
let residue ~salt ~row_stride ~modulus t =
  if row_stride <= 0 then
    invalid_arg
      (Printf.sprintf "Bench_checksum.residue: row_stride = %d must be positive" row_stride);
  if modulus <= 0 then
    invalid_arg (Printf.sprintf "Bench_checksum.residue: modulus = %d must be positive" modulus);
  let key = mix ~salt (t / row_stride) (t % row_stride) in
  (if modulus % 2 = 0 then key lsr 7 else key) % modulus

(** [positive_level ~salt ~row_stride ~levels ~scale t] is a producer value drawn from the [levels]
    multiples of [scale] starting at [scale] — STRICTLY POSITIVE, never the zero an accumulator is
    initialized to. Use it for the multiplicand whose row spans the reduction (a matmul's left
    operand): a zero row there zeroes the whole output row, which reads exactly like a schedule that
    dropped the row. The other operand may keep zero in its set — with this one positive, no output
    row is systematically zero. *)
let positive_level ~salt ~row_stride ~levels ~scale t =
  Float.of_int (1 + residue ~salt ~row_stride ~modulus:levels t) *. scale

(** The cap on the checksum weights. Small on purpose: the benches' operands are exact in binary and
    their products are small multiples of a negative power of two, so a weight below 256 keeps the
    whole weighted reduction exact in the double accumulator — which is what lets variants that sum
    in different orders be compared for BITWISE equality. *)
let weight_cap = 251

(** The salts the checksum's weight streams are drawn from, distinct from the salts the benches mint
    operands with so that the weights are independent of the data they weigh.

    TWO of them, because one is not enough at a narrow row stride: a row's weight vector then lives
    in [weight_cap ^ row_stride] values, and two rows landing on the same vector are a pair whose
    swap NO weighting of that stream can see. A second independent stream squares the space — at the
    narrowest legal stride, 2, from 63001 to about 4e9 — which puts the birthday threshold far
    beyond any row count a bench is run at, and costs one more pass outside the timed region.
    Neither sum's exactness argument changes: each is the same bounded-weight reduction it was. *)
let weight_salts = [ 0x7E51; 0x2F3B ]

(** One weight stream of {!whole_output}. Exposed for the discrimination test's negative control — a
    bench should call {!whole_output}, which is the guard. *)
let weighted ~salt ~row_stride values =
  Array.foldi values ~init:0.0 ~f:(fun t acc v ->
      acc +. (v *. Float.of_int (1 + residue ~salt ~row_stride ~modulus:weight_cap t)))

(** [whole_output ~row_stride values] is the position-weighted checksum of a row-major
    [m x row_stride] output, one sum per stream in {!weight_salts}. Weighted rather than plain
    because a plain sum reads only the multiset, so a tail written with the right values at the
    wrong offsets — a permutation, which is what a misplaced edge peel produces — leaves it
    unchanged. Every correct variant of a computation produces the identical list.

    This is the printed fingerprint, not the assertion — {!first_difference} is what a bench decides
    on. Both are correctness checks rather than measurements: call them OUTSIDE the timed region. *)
let whole_output ~row_stride values =
  List.map weight_salts ~f:(fun salt -> weighted ~salt ~row_stride values)

(** The weights {!whole_output} applies to one row, every stream concatenated: the vector two rows
    must differ in for a swap of them to be visible at all. What the discrimination test sweeps. *)
let row_weights ~row_stride ~row =
  Array.of_list
    (List.concat_map weight_salts ~f:(fun salt ->
         List.init row_stride ~f:(fun j ->
             1 + residue ~salt ~row_stride ~modulus:weight_cap ((row * row_stride) + j))))

(** A checksum rendered for a bench's timing line, LOSSLESSLY: [%.17g] round-trips a double, so a
    reader comparing two lines compares the checksums themselves, not a rounding of them — the
    gh-ocannl-1051 A/B validator's equality checks rest on this (%.10g dropped deltas below the
    tenth digit). *)
let render checksums =
  String.concat ~sep:"/" (List.map checksums ~f:(fun c -> Printf.sprintf "%.17g" c))

(** How one variant's output differs from the reference variant's. *)
type disagreement =
  | Length of { reference : int; got : int }
  | Cell of { at : int; reference : float; got : float }

(** [first_difference ~reference values] is the first cell at which [values] departs from
    [reference], if any. THIS is the guard a bench asserts on, and it is where the checksum's whole
    collision class goes away: it compares what the variants computed rather than a digest of it, so
    a permutation, a dropped edge peel and a repeated tail are all simply differences, with nothing
    to cancel and no weights to collide. The digest stays worth printing — one number per line
    fingerprints a run and travels into a report — but it is not what decides.

    Exact equality is a verdict only where the bench has established that every partial sum AND the
    output are exact at its precision and extents — whatever order a variant sums in. Exact operands
    are not enough: whether the reduction rounds depends on k and on the widths the partials and the
    output are held at (at f16, [schedule_bench 256 2 256 256 1] rounds its output at 1/8 above 128,
    so a reordered sum differs in its last bit without being wrong). Each bench derives this for
    itself and reports a difference outside it as expected rather than wrong: [schedule_bench]'s
    [exact_reduction] bounds k times the operands' magnitudes against the operand precision's
    significand, and [narrow_gebp_bench] splits its comparisons into required ones and the
    naive-vs-packed one its [naive_comparable] gates. A new bench calling this owes the same
    derivation.

    Outside the timed region, like the checksum. *)
let first_difference ~reference values =
  if Array.length reference <> Array.length values then
    Some (Length { reference = Array.length reference; got = Array.length values })
  else
    Array.findi values ~f:(fun t v -> not (Float.equal reference.(t) v))
    |> Option.map ~f:(fun (at, got) -> Cell { at; reference = reference.(at); got })

(** How a bench renders {!first_difference} on its timing line: short enough to sit beside the
    timings, specific enough to start a diagnosis from. [name] is the variant compared against — a
    bench with more than one legitimate comparand must say which one a line is about, or a
    difference cannot be read. *)
let render_agreement ~name = function
  | None -> Printf.sprintf "= %s" name
  | Some (Length { reference; got }) -> Printf.sprintf "SIZE %d vs %s %d" got name reference
  | Some (Cell { at; reference; got }) ->
      Printf.sprintf "DIFFERS from %s at [%d]: %.10g vs %.10g" name at got reference

(** The degenerate flat-offset weight this module exists to replace, kept so the discrimination test
    can carry it as a NEGATIVE CONTROL: a check that the new form passes is evidence only once the
    old form is shown to fail the same check. Not for use in a bench. *)
let flat_offset_weighted values =
  Array.foldi values ~init:0.0 ~f:(fun t acc v -> acc +. (v *. Float.of_int (1 + (t % weight_cap))))
