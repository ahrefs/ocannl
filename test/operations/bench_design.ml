(* The balanced visiting order interleaved timing shares (gh-ocannl-743, [bin/bench_design.ml]).

   What the design promises, claimed over every arm count up to 24 and every even batch count up to
   four times it: each batch is a permutation, mirrored pairs exchange every pair of arms, the
   position histograms are equal at every multiple of [Bench_design.exact_batches] and at no smaller
   count, and -- for an even arm count -- no arm visits a position class twice before the batch
   count reaches the arm count. The advanced-by-one rotation it replaced in [projection_shape_bench]
   is computed beside it, so the improvement is a claim too.

   The ORDERS table at the end is the relationship [benchmarks/bin_cells.py] is held to: its own
   [design_orders] must reproduce every row of it (the Python control in
   [test/operations/bin_cells.py] reads this golden), so a change to either side that the other does
   not make fails there rather than as two designs quietly diverging. *)

open Base
open Stdio
open Verdict.Claims
module D = Bench_design

let is_permutation n ord =
  Array.length ord = n
  && List.equal Int.equal (List.sort ~compare:Int.compare (Array.to_list ord)) (List.range 0 n)

(* The design [projection_shape_bench] used before: rotate by the pair index, mirror the odd
   member. *)
let advanced_by_one ~arms ~batches =
  Array.init batches ~f:(fun b ->
      let idx = Array.init arms ~f:(fun i -> (i + (b / 2)) % arms) in
      if b % 2 = 1 then Array.rev_inplace idx;
      idx)

(* The most visits any arm made to one position class: a position and its mirror. *)
let max_class_visits ~arms orders =
  let hist = Array.make_matrix ~dimx:arms ~dimy:arms 0 in
  Array.iter orders ~f:(fun ord ->
      Array.iteri ord ~f:(fun pos a ->
          let c = Int.min pos (arms - 1 - pos) in
          hist.(a).(c) <- hist.(a).(c) + 1));
  Array.fold hist ~init:0 ~f:(fun m h -> Array.fold h ~init:m ~f:Int.max)

let grid =
  List.concat_map (List.range 1 25) ~f:(fun arms ->
      List.map
        (List.filter (List.range 2 ((4 * arms) + 1)) ~f:(fun b -> b % 2 = 0))
        ~f:(fun batches -> (arms, batches)))

let () =
  let designs =
    List.map grid ~f:(fun (arms, batches) -> (arms, batches, D.orders ~arms ~batches))
  in
  p_all "every batch visits every arm exactly once" designs ~f:(fun (arms, _, orders) ->
      Array.for_all orders ~f:(is_permutation arms));
  p_all "every pair of arms is visited in both orders equally often, at every even batch count"
    designs ~f:(fun (arms, _, orders) -> (D.balance ~arms orders).pairs_exchanged);
  p_all "position histograms are equal at every multiple of the exact batch count"
    (List.filter designs ~f:(fun (arms, batches, _) -> batches % D.exact_batches ~arms = 0))
    ~f:(fun (arms, _, orders) -> (D.balance ~arms orders).histograms_equal);
  p_all "and at no batch count below the first multiple"
    (List.filter designs ~f:(fun (arms, batches, _) -> arms > 1 && batches < D.exact_batches ~arms))
    ~f:(fun (arms, _, orders) -> not (D.balance ~arms orders).histograms_equal);
  p_all "an even arm count: no arm visits a position class twice below the arm count's batches"
    (List.filter designs ~f:(fun (arms, batches, _) -> arms % 2 = 0 && batches <= arms))
    ~f:(fun (arms, _, orders) -> max_class_visits ~arms orders <= 2);
  p_all "the advanced-by-one rotation balances no histogram the spread design leaves unbalanced"
    grid ~f:(fun (arms, batches) ->
      (not (D.balance ~arms (advanced_by_one ~arms ~batches)).histograms_equal)
      || (D.balance ~arms (D.orders ~arms ~batches)).histograms_equal);
  p_all "nor gives any arm more edge visits than the advanced-by-one rotation's worst arm" grid
    ~f:(fun (arms, batches) ->
      snd (D.balance ~arms (D.orders ~arms ~batches)).edge_visits
      <= snd (D.balance ~arms (advanced_by_one ~arms ~batches)).edge_visits);
  p_exists "and an arm count where the spread design is exact at half the batches" grid
    ~f:(fun (arms, batches) ->
      (D.balance ~arms (D.orders ~arms ~batches)).histograms_equal
      && not (D.balance ~arms (advanced_by_one ~arms ~batches)).histograms_equal);
  (* The case the issue named: projection_shape_bench's rounds of 20 arms at 8 batches. *)
  printf "\nThe gh-ocannl-743 case, 20 arms:\n";
  List.iter [ 8; 20; 40 ] ~f:(fun batches ->
      let show orders =
        let b = D.balance ~arms:20 orders in
        let lo, hi = b.edge_visits in
        Printf.sprintf "histograms equal %b, edge visits %d-%d, most visits to one class %d"
          b.histograms_equal lo hi
          (max_class_visits ~arms:20 orders)
      in
      printf "  %2d batches: spread: %s\n              advanced-by-one: %s\n" batches
        (show (D.orders ~arms:20 ~batches))
        (show (advanced_by_one ~arms:20 ~batches)));
  printf "\n%s\n" (D.describe ~arms:20 ~batches:8);
  printf "%s\n" (D.describe ~arms:3 ~batches:6);
  (* Refusals: an odd or empty batch count, and no arms. *)
  let refused f = match f () with _ -> false | exception Invalid_argument _ -> true in
  p "an odd batch count is refused" (refused (fun () -> D.orders ~arms:3 ~batches:5));
  p "a zero batch count is refused" (refused (fun () -> D.orders ~arms:3 ~batches:0));
  p "no arms is refused" (refused (fun () -> D.orders ~arms:0 ~batches:2));
  (* The table benchmarks/bin_cells.py is held to: one row per batch, arm indices in visiting order.
     The grid covers both parities, counts below, at and above the exact one, and a pair count that
     does not divide the arm count. *)
  printf "\nORDERS\n";
  List.iter
    [ (1, 2); (2, 2); (2, 6); (3, 2); (3, 4); (3, 6); (4, 2); (4, 6); (5, 6); (6, 4); (7, 10) ]
    ~f:(fun (arms, batches) ->
      Array.iteri (D.orders ~arms ~batches) ~f:(fun b ord ->
          printf "%d %d %d: %s\n" arms batches b
            (String.concat ~sep:" " (Array.to_list (Array.map ord ~f:Int.to_string)))));
  printf "END ORDERS\n"
