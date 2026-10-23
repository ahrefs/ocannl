(** Balanced timing-position designs for interleaved benchmark arms (gh-ocannl-743).

    A harness that times [n] arms against each other visits them in batches: every batch times every
    arm once, in some order, and the order is the design. Two position effects confound an arm with
    where it was visited. A LINEAR one (monotone session drift: clocks settling, a queue filling) is
    cancelled by mirroring: each batch is followed by its exact reverse, so every pair of arms is
    visited in both orders equally often and every arm's mean position is the same. A NONLINEAR one
    (a first-launch warm-up, the edges of a round) is not -- an arm visited at the edges more often
    than another still carries more of it. That needs every arm's position HISTOGRAM to be the same,
    which mirroring alone does not give.

    The design here is a mirrored rotation whose per-pair offsets are spread over the arm count
    rather than advanced by one: pair [p] of [P = batches / 2] rotates the arm order by
    [2 * floor (p * (n / 2) / P)] for an even [n] (the even offsets, spread), by [floor (p * n / P)]
    for an odd one. Its properties, each pinned by [test/operations/bench_design]:

    - every pair of arms is exchanged equally often, at every even batch count (the mirror);
    - the position histograms are EQUAL across arms as soon as [n] divides [batches] for an even
      [n], and [2n] does for an odd one. That is the least batch count any design can do it at: each
      position holds [batches] visits to share among [n] arms, and mirrored pairs make [batches]
      even. The rotation advanced by one per pair, which [projection_shape_bench] used before, needs
      [2n] for every [n];
    - below that, for an even [n], no arm visits the same position CLASS (a position and its mirror,
      [q] and [n - 1 - q]) twice before [batches] reaches [n]: the even offsets keep each arm's
      visits on one parity, and a position and its mirror have opposite parities. So an edge
      position falls to an arm at most once per mirrored pair, where the advanced-by-one rotation
      hands the first arm both edges in its first two pairs.

    Below the exact count no design balances every histogram, so {!describe} says which regime a run
    is in and the least batch count that would balance it. A seeded random order -- the other design
    the issue named -- balances only in expectation, never exactly, and would need its generator
    kept in step across the two languages that share this design; it was not taken.

    The design is shared with the process-level driver [benchmarks/bin_cells.py], which visits whole
    bench processes as arms in the same order; that driver's [design_orders] is held equal to
    {!orders} over the grid [test/operations/bench_design] prints. *)

open Base

let check ~arms ~batches =
  if arms < 1 then invalid_arg (Printf.sprintf "Bench_design: %d arms; at least one is needed" arms);
  if batches < 2 || batches % 2 = 1 then
    invalid_arg
      (Printf.sprintf
         "Bench_design: %d batches; the count must be even and positive -- every batch is mirrored \
          by its partner"
         batches)

(** The rotation offset of mirrored pair [p] of [pairs]. *)
let offset ~arms ~pairs p = if arms % 2 = 0 then 2 * (p * (arms / 2) / pairs) else p * arms / pairs

(** [order ~arms ~batches b]: the arm indices in the order batch [b] (0-based) visits them. *)
let order ~arms ~batches b =
  check ~arms ~batches;
  if b < 0 || b >= batches then
    invalid_arg (Printf.sprintf "Bench_design.order: batch %d outside 0..%d" b (batches - 1));
  let s = offset ~arms ~pairs:(batches / 2) (b / 2) in
  let idx = Array.init arms ~f:(fun i -> (i + s) % arms) in
  if b % 2 = 1 then Array.rev_inplace idx;
  idx

let orders ~arms ~batches =
  check ~arms ~batches;
  Array.init batches ~f:(order ~arms ~batches)

type balance = {
  pairs_exchanged : bool;
      (** Every pair of arms visited in each order equally often -- which also equalizes every arm's
          mean position. *)
  histograms_equal : bool;  (** Every arm visited each position equally often. *)
  edge_visits : int * int;
      (** The fewest and the most visits any arm made to the two edge positions, first and last. *)
}

(** The balance [orders] achieves over [arms] arms, read off the orders themselves. *)
let balance ~arms (orders : int array array) =
  let hist = Array.make_matrix ~dimx:arms ~dimy:arms 0 in
  let before = Array.make_matrix ~dimx:arms ~dimy:arms 0 in
  Array.iter orders ~f:(fun ord ->
      Array.iteri ord ~f:(fun pos a ->
          hist.(a).(pos) <- hist.(a).(pos) + 1;
          for later = pos + 1 to arms - 1 do
            before.(a).(ord.(later)) <- before.(a).(ord.(later)) + 1
          done));
  let pairs_exchanged =
    List.for_all (List.range 0 arms) ~f:(fun a ->
        List.for_all (List.range 0 arms) ~f:(fun b -> before.(a).(b) = before.(b).(a)))
  in
  let histograms_equal = Array.for_all hist ~f:(fun h -> Array.equal Int.equal h hist.(0)) in
  let edges = Array.map hist ~f:(fun h -> if arms = 1 then h.(0) else h.(0) + h.(arms - 1)) in
  let lo = Array.fold edges ~init:Int.max_value ~f:min and hi = Array.fold edges ~init:0 ~f:max in
  { pairs_exchanged; histograms_equal; edge_visits = (lo, hi) }

(** The least batch count at which every arm's position histogram is equal. *)
let exact_batches ~arms = if arms % 2 = 0 then arms else 2 * arms

(** One line naming the design and the regime [batches] puts it in, for a harness's header. *)
let describe ~arms ~batches =
  let b = balance ~arms (orders ~arms ~batches) in
  let lo, hi = b.edge_visits in
  let exact = exact_batches ~arms in
  Printf.sprintf
    "visiting order: spread mirrored rotation, %d arm(s) x %d batches -- every pair exchanged: %b; \
     position histograms equal: %b (exact at multiples of %d batches); edge visits per arm %d-%d"
    arms batches b.pairs_exchanged b.histograms_equal exact lo hi
