(* What the benches' whole-output checksum has to discriminate (gh-ocannl-711).

   [bin/schedule_bench.ml] and [bin/narrow_gebp_bench.ml] compare hand-written schedules of the same
   matmul against each other, and the only thing standing between "this schedule is fast" and "this
   schedule is fast and correct" is a checksum of the whole output plus one interior spot cell. A
   checksum that cannot see a row permutation is worse than none: the bench then reports a fast
   WRONG schedule as a win, and the autotuner learns from those reports.

   The natural way to write such a weight — a residue of the FLATTENED offset [t = i*n + j] — is
   degenerate exactly where these benches run. [1 + (t mod 251)] gives every row the identical
   weight vector whenever 251 divides n, so a swapped pair of rows leaves the checksum unchanged;
   and at the same extent the spot cell at [1][1] sees nothing outside its own row, so both halves
   of the guard fail together. The same collapse hits operand data drawn as [(t mod p)] over a
   row-major array: at [p | row_stride] every row is identical, so a schedule substituting the wrong
   row computes the right answer and no whole-output check can tell.

   [Bench_checksum] keys both on the (row, column) pair through an aperiodic mix instead. Three more
   ways for the same guard to go blind are pinned here beside that one, because keying on the pair
   answers none of them:

   - a checksum is a LINEAR functional of the output, so a row swap survives it whenever the value
   difference is orthogonal to the weight difference — by the weights colliding, or by plain
   cancellation. No bounded-weight scalar escapes that, so what a bench asserts on is
   [Bench_checksum.first_difference], an elementwise comparison against the reference variant; the
   checksum is what it PRINTS; - the mix's own fold used to compress a 40-bit product into 24 bits
   and was GF(2)-linear, so two rows' outputs differed by a value depending on neither column nor
   salt: row pairs existed that were identical in EVERY stream at EVERY salt, which no number of
   streams can repair; - an operand row of all zeros makes its output row all zeros,
   indistinguishable from a schedule that dropped the row against a zero-initialized destination
   (hence [Bench_checksum.positive_level]).

   Every claim below is paired with the pre-fix form as a NEGATIVE CONTROL: a check the new form
   passes is evidence only once the old form is shown to fail the same check. The controls are the
   reason [Bench_checksum.flat_offset_weighted] and [Bench_checksum.weighted] are exposed. *)

open Base
module Bc = Bench_checksum

let p = Stdio.printf

(* A synthetic m x n output whose value varies with BOTH indices, which is what a matmul result
   does. A value depending on the row alone (or an affine [i*n + j]) would make every row difference
   constant along the row, and a row swap would then be visible only through the SUM of each row's
   weights — a much weaker thing to ask of a weight, and one that collides by accident at a handful
   of extents for reasons that have nothing to do with the degeneracy under test. *)
let output ~m ~n =
  Array.init (m * n) ~f:(fun t -> Float.of_int (1 + (Bc.mix ~salt:0x00A5 (t / n) (t % n) % 97)))

let swap_rows ~n values ~r1 ~r2 =
  let v = Array.copy values in
  for j = 0 to n - 1 do
    let keep = v.((r1 * n) + j) in
    v.((r1 * n) + j) <- v.((r2 * n) + j);
    v.((r2 * n) + j) <- keep
  done;
  v

let plain_sum = Array.fold ~init:0.0 ~f:( +. )

(* Rows of an operand minted the way [schedule_bench] mints ma and mb: a residue of the flat offset
   over a row-major [rows x row_stride] array, either through the mix or the flat form. *)
let mixed_rows ~salt ~modulus ~row_stride ~rows =
  List.init rows ~f:(fun r ->
      Array.init row_stride ~f:(fun c ->
          Bc.residue ~salt ~row_stride ~modulus ((r * row_stride) + c)))

let flat_rows ~modulus ~row_stride ~rows =
  List.init rows ~f:(fun r -> Array.init row_stride ~f:(fun c -> ((r * row_stride) + c) % modulus))

let pairwise_distinct rows =
  let n = List.length rows in
  n = List.length (List.dedup_and_sort rows ~compare:(fun a b -> Array.compare Int.compare a b))

let all_identical = function
  | [] | [ _ ] -> true
  | first :: rest -> List.for_all rest ~f:(fun r -> Array.equal Int.equal first r)

(* The extents swept for the row-permutation claims: everything from the smallest two-row output up
   past the third multiple of 251, so the degenerate extents 251, 502 and 753 are reached by a sweep
   rather than named into it. *)
let extents = List.range 2 801
let row_pairs = [ (0, 1); (1, 3); (0, 3); (2, 3) ]
let rows = 4

let () =
  (* Which extents each form of the guard FAILED to notice a row swap at. *)
  let missed_mixed = ref [] and missed_flat = ref [] and missed_plain = ref [] in
  List.iter extents ~f:(fun n ->
      let v = output ~m:rows ~n in
      let mixed = Bc.whole_output ~row_stride:n v in
      let flat = Bc.flat_offset_weighted v in
      let plain = plain_sum v in
      List.iter row_pairs ~f:(fun (r1, r2) ->
          let w = swap_rows ~n v ~r1 ~r2 in
          if List.equal Float.equal (Bc.whole_output ~row_stride:n w) mixed then
            missed_mixed := n :: !missed_mixed;
          if Float.equal (Bc.flat_offset_weighted w) flat then missed_flat := n :: !missed_flat;
          if Float.equal (plain_sum w) plain then missed_plain := n :: !missed_plain));
  let uniq ns = List.dedup_and_sort ~compare:Int.compare ns in
  let flat_missed = uniq !missed_flat in
  p "extents swept: %d..%d, row pairs per extent: %d\n" (List.hd_exn extents)
    (List.last_exn extents) (List.length row_pairs);
  p "extents where the flat-offset weight missed a row swap: %s\n"
    (String.concat ~sep:", " (List.map flat_missed ~f:Int.to_string));
  (* Named with its row count: over four rows this sweep cannot reach the weight-collision class
     below, and a claim of "every row swap at every extent" would be an over-reading of it. *)
  Verdict.p_empty "the shared checksum sees every 4-row swap at every extent swept" ~over:extents
    !missed_mixed;
  (* The multiples of 251 in range are where the flat form is degenerate BY CONSTRUCTION (rather
     than by an accidental weight collision), and it must miss the swap at every row pair there. *)
  let degenerate = List.filter extents ~f:(fun n -> n % Bc.weight_cap = 0) in
  Verdict.p_all
    "the flat-offset weight misses a row swap at every multiple of 251 (negative control)"
    degenerate ~f:(fun n ->
      List.length (List.filter !missed_flat ~f:(fun m -> m = n)) = List.length row_pairs);
  Verdict.p_all "the shared checksum sees a 4-row swap at those same multiples of 251" degenerate
    ~f:(fun n -> not (List.mem !missed_mixed n ~equal:Int.equal));
  (* Why the guard is WEIGHTED at all: a permutation preserves the multiset, and a plain sum reads
     nothing else. *)
  Verdict.p "an unweighted sum misses every 4-row swap at every extent (negative control)"
    (List.length !missed_plain = List.length extents * List.length row_pairs);

  (* Operands. [schedule_bench] draws ma over a row-major m x k array and mb as a residue mod 17
     over a k x n one, so the collapse lands at 48 | k and 17 | n respectively. Sweep the multiples
     of each modulus (ma's is 48 since gh-ocannl-738 took its granularity to 1/16: see the
     repeat-distance table below, and the zero-sentinel claims). *)
  let operand_multiples = List.range 1 25 in
  List.iter
    [ ("ma", 0x5A17, 48); ("mb", 0x3C6E, 17) ]
    ~f:(fun (name, salt, modulus) ->
      let strides = List.map operand_multiples ~f:(fun mult -> modulus * mult) in
      Verdict.p_all
        (Printf.sprintf "mixed %s rows stay pairwise distinct at every row stride %d divides" name
           modulus) strides ~f:(fun row_stride ->
          pairwise_distinct (mixed_rows ~salt ~modulus ~row_stride ~rows:16));
      Verdict.p_all
        (Printf.sprintf
           "flat-offset %s rows are all identical at every row stride %d divides (negative control)"
           name modulus) strides ~f:(fun row_stride ->
          all_identical (flat_rows ~modulus ~row_stride ~rows:16)));

  (* Enough distinct weights to tell the rows apart. A row's weight vector is what a swap of two
     rows has to differ in; where two rows share one, their swap is invisible to any weighting of
     those streams, whatever the data. The space is [251 ^ row_stride] per stream, so the risk is
     concentrated at the NARROW strides, and a sweep over four rows (above) is far too small to
     reach it: it takes hundreds of rows at row_stride 2. Sweep row counts instead of row pairs, and
     ask the property directly — pairwise distinct weight vectors — which is what makes a 512-row
     sweep affordable. *)
  let colliding ~streams ~row_stride ~rows =
    let seen = Hashtbl.create (module String) in
    List.find_map (List.range 0 rows) ~f:(fun row ->
        let w =
          match streams with
          | `Both -> Bc.row_weights ~row_stride ~row
          | `Primary ->
              Array.init row_stride ~f:(fun j ->
                  1
                  + Bc.residue ~salt:(List.hd_exn Bc.weight_salts) ~row_stride
                      ~modulus:Bc.weight_cap
                      ((row * row_stride) + j))
        in
        let key = String.concat ~sep:"," (Array.to_list (Array.map w ~f:Int.to_string)) in
        match Hashtbl.find seen key with
        | Some earlier -> Some (row_stride, earlier, row)
        | None ->
            Hashtbl.set seen ~key ~data:row;
            None)
  in
  let sweep_strides = List.range 2 65 @ [ 251; 502; 753 ] in
  let collisions ~streams ~rows =
    List.filter_map sweep_strides ~f:(fun row_stride -> colliding ~streams ~row_stride ~rows)
  in
  let both_collisions = collisions ~streams:`Both ~rows:512 in
  let primary_collisions = collisions ~streams:`Primary ~rows:512 in
  p "row strides swept for weight collisions: %d..%d plus 251, 502, 753, over %d rows\n" 2 64 512;
  p "single-stream weight collisions found: %s\n"
    (String.concat ~sep:", "
       (List.map primary_collisions ~f:(fun (stride, a, b) ->
            Printf.sprintf "row stride %d rows %d/%d" stride a b)));
  Verdict.p_empty "no two rows share a weight vector across both streams, at any row stride swept"
    ~over:sweep_strides both_collisions;
  Verdict.p "a single weight stream does collide there, so the sweep can fail (negative control)"
    (not (List.is_empty primary_collisions));
  (* And the collision the single stream has is a swap it really cannot see: same weights, different
     data. This is the shape the review named — [schedule_bench 2 <r> 364 2]. *)
  let single_stream_blind =
    match primary_collisions with
    | (row_stride, a, b) :: _ ->
        let v = output ~m:(b + 1) ~n:row_stride in
        let w = swap_rows ~n:row_stride v ~r1:a ~r2:b in
        let salt = List.hd_exn Bc.weight_salts in
        (not (Array.equal Float.equal v w))
        && Float.equal (Bc.weighted ~salt ~row_stride v) (Bc.weighted ~salt ~row_stride w)
        && not
             (List.equal Float.equal (Bc.whole_output ~row_stride v) (Bc.whole_output ~row_stride w))
    | [] -> false
  in
  Verdict.p
    "the swap that single stream is blind to changes the data and is seen by both streams together"
    single_stream_blind;

  (* No producer value is the accumulator's init. ma's row spans the reduction, so an all-zero ma
     row zeroes the whole output row — indistinguishable from a schedule that dropped the row
     against a zero-initialized destination. Under a residue that admits zero, a row of k
     independent draws is all-zero with probability levels^-k, which at k = 2 is one row in 169. *)
  let ma_value ~row_stride t =
    Bc.positive_level ~salt:0x5A17 ~row_stride ~levels:48 ~scale:0.0625 t
  in
  let ma_rows_with_zero ~row_stride ~rows =
    List.count (List.range 0 rows) ~f:(fun r ->
        List.for_all (List.range 0 row_stride) ~f:(fun c ->
            Float.equal (ma_value ~row_stride ((r * row_stride) + c)) 0.0))
  in
  let zeroing_rows ~row_stride ~rows =
    List.count (List.range 0 rows) ~f:(fun r ->
        List.for_all (List.range 0 row_stride) ~f:(fun c ->
            Bc.residue ~salt:0x5A17 ~row_stride ~modulus:13 ((r * row_stride) + c) = 0))
  in
  let ma_strides = List.range 1 9 in
  Verdict.p_all "every ma value is strictly positive, so no ma row can be all-zero" ma_strides
    ~f:(fun row_stride ->
      List.for_all
        (List.range 0 (600 * row_stride))
        ~f:(fun t -> Float.( > ) (ma_value ~row_stride t) 0.0));
  Verdict.p_all "no ma row is all-zero at any row stride swept" ma_strides ~f:(fun row_stride ->
      ma_rows_with_zero ~row_stride ~rows:600 = 0);
  (* The control keeps the pre-fix modulus, 13, rather than following ma's level count to 48: what
     it demonstrates is that a ZERO-ADMITTING residue produces the all-zero row, and raising the
     level count only pushes the probability down ((levels + 1)^-k) without ever reaching zero. The
     fix is [positive_level]'s strict positivity, so the control has to be a form that admits zero
     at a rate a 600-row sweep can see. *)
  Verdict.p
    "a residue admitting zero does produce an all-zero ma row at row stride 2 (negative control)"
    (zeroing_rows ~row_stride:2 ~rows:600 > 0);

  (* The elementwise guard. Everything above is about how far a WEIGHTED SUM can be pushed; this is
     the claim that the benches do not rest on that. The reviewer's case is [schedule_bench 2 <r>
     719 2]: output rows 240 and 718 differ by [-14; 14], and both streams' weight differences are
     constant across the two columns, so both sums cancel and the swap is invisible to the printed
     fingerprint. It is not invisible to a comparison of the outputs. *)
  let sb_ma ~m ~k =
    Array.init (m * k) ~f:(Bc.positive_level ~salt:0x5A17 ~row_stride:k ~levels:48 ~scale:0.0625)
  in
  let sb_mb ~k ~n =
    Array.init (k * n) ~f:(fun t ->
        Float.of_int (Bc.residue ~salt:0x3C6E ~row_stride:n ~modulus:17 t) -. 8.)
  in
  let sb_output ~m ~n ~k =
    let ma = sb_ma ~m ~k and mb = sb_mb ~k ~n in
    Array.init (m * n) ~f:(fun t ->
        let i = t / n and j = t % n in
        List.fold (List.range 0 k) ~init:0.0 ~f:(fun acc kk ->
            acc +. (ma.((i * k) + kk) *. mb.((kk * n) + j))))
  in
  (* Search for a swap of GENERATED rows that both streams miss, rather than asserting the one pair
     the review happened to find: the claim is about the class, so a pair found by sweep is one the
     sweep still finds if the constants move. Done on the difference form — a swap of rows a and b
     shifts stream s by [sum_j (v_aj - v_bj) (w_aj - w_bj)] — which is O(n) per pair instead of a
     checksum of the whole output per pair. *)
  let cancelling ~m ~n ~k =
    let v = sb_output ~m ~n ~k in
    let w ~salt t = 1 + Bc.residue ~salt ~row_stride:n ~modulus:Bc.weight_cap t in
    let shift ~salt a b =
      List.fold (List.range 0 n) ~init:0.0 ~f:(fun acc j ->
          let ta = (a * n) + j and tb = (b * n) + j in
          acc +. ((v.(ta) -. v.(tb)) *. Float.of_int (w ~salt ta - w ~salt tb)))
    in
    let rows_differ a b =
      List.exists (List.range 0 n) ~f:(fun j -> not (Float.equal v.((a * n) + j) v.((b * n) + j)))
    in
    List.find_map (List.range 0 m) ~f:(fun a ->
        List.find_map
          (List.range (a + 1) m)
          ~f:(fun b ->
            if
              rows_differ a b
              && List.for_all Bc.weight_salts ~f:(fun salt -> Float.equal (shift ~salt a b) 0.0)
            then Some (a, b, v)
            else None))
  in
  (* Two geometries, so the class is not read as a property of the narrowest one. The n = 2 one
     sweeps twice as many rows as the n = 3 one because ma's finer levels moved its first cancelling
     pair out past 2000 (gh-ocannl-738): a richer operand set makes an accidental cancellation
     rarer, which is the same improvement the repeat-distance table below measures — it does not
     remove the class, and this is where that is shown. *)
  let cancel_cases =
    List.filter_map
      [ (4000, 2, 2); (2000, 3, 2) ]
      ~f:(fun (m, n, k) ->
        Option.map (cancelling ~m ~n ~k) ~f:(fun (a, b, v) -> (m, n, k, a, b, v)))
  in
  List.iter cancel_cases ~f:(fun (m, n, k, a, b, _) ->
      p "both weight streams miss the swap of generated rows %d and %d at m=%d n=%d k=%d\n" a b m n
        k);
  Verdict.p
    "the printed checksum can miss a swap of generated output rows, so it is not the guard \
     (negative control)"
    (List.length cancel_cases = 2);
  (* Confirm through the guard itself, not only through the difference form it was found with. *)
  Verdict.p_all "those swaps really do leave every printed checksum stream unchanged" cancel_cases
    ~f:(fun (_, n, _, a, b, v) ->
      let w = swap_rows ~n v ~r1:a ~r2:b in
      (not (Array.equal Float.equal v w))
      && List.equal Float.equal (Bc.whole_output ~row_stride:n v) (Bc.whole_output ~row_stride:n w));
  Verdict.p_all "the elementwise guard sees those swaps" cancel_cases ~f:(fun (_, n, _, a, b, v) ->
      let w = swap_rows ~n v ~r1:a ~r2:b in
      Option.is_some (Bc.first_difference ~reference:v w));
  (* And it sees the whole class the weighted sums were swept for, at every extent, without a
     collision argument: it compares what was computed. *)
  let elementwise_missed =
    List.concat_map extents ~f:(fun n ->
        let v = output ~m:rows ~n in
        List.filter_map row_pairs ~f:(fun (r1, r2) ->
            let w = swap_rows ~n v ~r1 ~r2 in
            if Option.is_none (Bc.first_difference ~reference:v w) then Some n else None))
  in
  Verdict.p_empty "the elementwise guard sees every row swap at every extent swept" ~over:extents
    elementwise_missed;
  Verdict.p "the elementwise guard reports a length mismatch too"
    (Option.is_some (Bc.first_difference ~reference:[| 1.0; 2.0 |] [| 1.0 |]));

  (* The mix's own row identity. Under the pre-fix fold two rows could be identical at EVERY column
     of EVERY stream, so no number of weight streams and no operand value set could tell them apart
     — the first such pair is 5977 and 10232. The fix is structural rather than statistical: masking
     the pre-state to 24 bits before the fold makes the whole finalizer a bijection of it, and [a *
     73856093] is injective mod 2^24, so distinct rows below 2^24 differ at every column. *)
  let pre_fix_mix ~salt a b =
    let x = a * 73856093 lxor (b * 19349663) lxor salt in
    let x = x lxor (x lsr 13) land 0xFFFFFF in
    let x = x * 1274126177 land 0xFFFFFF in
    x lxor (x lsr 7)
  in
  let mix_salts = [ 0x5A17; 0x3C6E ] @ Bc.weight_salts in
  let rows_identical mix ~salt a b ~columns =
    List.for_all (List.range 0 columns) ~f:(fun c -> mix ~salt a c = mix ~salt b c)
  in
  Verdict.p_all
    "the pre-fix fold made rows 5977 and 10232 identical at every column of every salt (negative \
     control)"
    mix_salts ~f:(fun salt -> rows_identical pre_fix_mix ~salt 5977 10232 ~columns:64);
  Verdict.p_all "the mix keeps those two rows apart" mix_salts ~f:(fun salt ->
      not (rows_identical Bc.mix ~salt 5977 10232 ~columns:64));
  let first_value_collision mix ~salt ~column ~rows =
    let seen = Hashtbl.create (module Int) in
    List.find_map (List.range 0 rows) ~f:(fun r ->
        let v = mix ~salt r column in
        match Hashtbl.find seen v with
        | Some _ -> Some r
        | None ->
            Hashtbl.set seen ~key:v ~data:r;
            None)
  in
  Verdict.p_all
    "no two rows below 100000 share a mix value in a column, at any salt or column swept" mix_salts
    ~f:(fun salt ->
      List.for_all [ 0; 1; 7; 63 ] ~f:(fun column ->
          Option.is_none (first_value_collision Bc.mix ~salt ~column ~rows:100_000)));
  Verdict.p "the pre-fix fold did collide there, so that sweep can fail (negative control)"
    (List.exists mix_salts ~f:(fun salt ->
         List.exists [ 0; 1; 7; 63 ] ~f:(fun column ->
             Option.is_some (first_value_collision pre_fix_mix ~salt ~column ~rows:100_000))));

  (* Narrow reductions, which the four-row and 12-multiple sweeps above cannot reach. How many rows
     an operand keeps distinct is bounded by [levels ^ row_stride] whatever the generator — 2304 at
     the narrowest reduction this bench accepts, since gh-ocannl-738 took ma to 48 levels of 1/16
     from 12 of 1/4 and the bound at k = 2 from 144 — so the honest claim is a COMPARISON against
     the form this replaced, not distinctness. The flat residue's bound is not a birthday at all:
     rows repeat with the modulus's period, 13, at EVERY stride. That is what the granularity bump
     bought and what this table measures: ma's first repeat at k = 2 moved from row 11 to row 33,
     which is the first stride at which the mixed form overtakes the flat one at all, so the
     crossover below is k = 1 rather than the k = 2 it was. *)
  let first_row_collision rowf =
    let seen = Hashtbl.create (module String) in
    List.find_map (List.range 0 20_000) ~f:(fun r ->
        let key = String.concat ~sep:"," (List.map (rowf r) ~f:Int.to_string) in
        match Hashtbl.find seen key with
        | Some _ -> Some r
        | None ->
            Hashtbl.set seen ~key ~data:r;
            None)
  in
  let narrow_strides = [ 1; 2; 3; 4; 6; 8; 12; 16 ] in
  (* Both of [schedule_bench]'s operands, because the class is the generator's, not one operand's:
     ma is m x k so its rows repeat along the OUTPUT extent m, and mb is k x n so its rows repeat
     along the REDUCTION extent k — a wrong-row read there is a staging or tensorization bug, and
     equally invisible. Each is swept against the form it replaced. *)
  let mixed_at ~salt ~levels stride =
    first_row_collision (fun r ->
        List.init stride ~f:(fun c ->
            Bc.residue ~salt ~row_stride:stride ~modulus:levels ((r * stride) + c)))
  in
  let flat_at ~modulus stride =
    first_row_collision (fun r -> List.init stride ~f:(fun c -> ((r * stride) + c) % modulus))
  in
  let render_row = function None -> "none below 20000" | Some r -> Int.to_string r in
  let operands = [ ("ma", 0x5A17, 48, 13, "k", 1); ("mb", 0x3C6E, 17, 17, "n", 2) ] in
  List.iter operands ~f:(fun (name, salt, levels, modulus, axis, crossover) ->
      p "row at which %s rows first repeat, by %s (mixed over %d levels vs flat mod %d):\n" name
        axis levels modulus;
      List.iter narrow_strides ~f:(fun stride ->
          p "  %s = %-3d mixed %-18s flat %s\n" axis stride
            (render_row (mixed_at ~salt ~levels stride))
            (render_row (flat_at ~modulus stride)));
      (* The flat form's bound is not a birthday: its rows repeat with the modulus's period at EVERY
         stride, 13 or 17 rows in. The mixed form is birthday-limited in a space of levels^stride,
         so it is behind only where that space is itself too small to matter. *)
      Verdict.p_all
        (Printf.sprintf
           "the mixed %s keeps rows distinct at least as far as the flat form, at every %s above %d"
           name axis crossover)
        (List.filter narrow_strides ~f:(fun stride -> stride > crossover))
        ~f:(fun stride ->
          match (mixed_at ~salt ~levels stride, flat_at ~modulus stride) with
          | None, _ -> true
          | Some _, None -> false
          | Some m, Some f -> m >= f);
      Verdict.p_all
        (Printf.sprintf
           "at %s <= %d both forms are exhausted within twenty rows, against a levels^%s space \
            bound of %d that leaves no room to improve"
           axis crossover axis (Int.pow levels crossover))
        (List.range 1 (crossover + 1))
        ~f:(fun stride ->
          match (mixed_at ~salt ~levels stride, flat_at ~modulus stride) with
          | Some m, Some f -> m <= 20 && f <= 20
          | _ -> false));

  (* The arithmetic the weights' exactness argument rests on: a residue is a residue (non-negative,
     below its modulus), so a weight is in [1, 251] and products of the benches' exact-in-binary
     operands stay exact in the double accumulator. *)
  let moduli = [ 3; 5; 13; 17; Bc.weight_cap ] in
  let row_strides = List.range 1 40 in
  Verdict.p_all "every residue lands in [0, modulus)" moduli ~f:(fun modulus ->
      List.for_all row_strides ~f:(fun row_stride ->
          let indices = List.range 0 (7 * row_stride) in
          (not (List.is_empty indices))
          && List.for_all indices ~f:(fun t ->
              let r = Bc.residue ~salt:0x7E51 ~row_stride ~modulus t in
              r >= 0 && r < modulus)));
  let refuses f = match f () with exception Invalid_argument _ -> true | _ -> false in
  Verdict.p "residue refuses a non-positive row stride"
    (refuses (fun () -> Bc.residue ~salt:0 ~row_stride:0 ~modulus:13 1));
  Verdict.p "residue refuses a non-positive modulus"
    (refuses (fun () -> Bc.residue ~salt:0 ~row_stride:8 ~modulus:0 1))
