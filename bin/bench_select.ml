(** Two-stage selection of a class's best arm from noisy timings (gh-ocannl-1531).

    A harness that times many candidates, one round per geometry, and then wants "the best candidate
    of a class" faces the winner's curse: the arm that reads best in its own round is, more often
    than its true rank warrants, the one whose batches drew the least contention, and the round it
    was read in is not the round its rivals were read in. [projection_shape_bench] used to send
    exactly that arm -- the round-best under each statistic -- to its finalists round, so a true
    best that read second in its geometry round never reached the round that decides (staging#1331:
    a b8x128 coal-k finalist slower than the one staging#1209 had crowned).

    The rule here is two stages:

    - {!shortlist}: from the first readings, keep the top [k] under the PRIMARY statistic and the
      best under each other statistic. It contains every arm the single-shot rule would have sent,
      so the change can only add contenders.
    - the harness re-times the whole shortlist in ONE interleaved round, in one process, so every
      contender of the class is read under the same drift;
    - {!best}: the finalists are the best under each statistic by the RE-TIMED readings alone. The
      first readings only decided who is re-timed; mixing them back in would reintroduce the
      cross-round drift the re-timing exists to remove.

    The harness then times the finalists once more, so the number it reports is not the draw that
    selected them. Scores are "higher is better" (throughput); a NaN ranks below every number, and
    ties go to the earlier arm, so a selection is a function of its inputs. The property test
    [test/operations/bench_select] drives both rules with scripted noise. *)

open Base

(* Higher first; NaN last. *)
let compare_desc a b =
  match (Float.is_nan a, Float.is_nan b) with
  | true, true -> 0
  | true, false -> 1
  | false, true -> -1
  | false, false -> Float.compare b a

(** [ranked ~score xs]: the indices of [xs] from best to worst under [score], ties to the earlier
    index. *)
let ranked ~score xs =
  List.mapi xs ~f:(fun i x -> (i, score x))
  |> List.stable_sort ~compare:(fun (_, a) (_, b) -> compare_desc a b)
  |> List.map ~f:fst

let keep xs idx =
  let idx = Set.of_list (module Int) idx in
  List.filteri xs ~f:(fun i _ -> Set.mem idx i)

(** [best ~scores xs]: the best of [xs] under each of [scores], deduplicated, in [xs]'s order. Empty
    for an empty [xs]. *)
let best ~scores xs = keep xs (List.filter_map scores ~f:(fun score -> List.hd (ranked ~score xs)))

(** [shortlist ~k ~primary ~others xs]: the top [k] of [xs] under [primary] and the best under each
    of [others], deduplicated, in [xs]'s order. A superset of [best ~scores:(primary :: others) xs].
*)
let shortlist ~k ~primary ~others xs =
  if k < 1 then invalid_arg (Printf.sprintf "Bench_select.shortlist: k = %d; at least 1" k);
  keep xs
    (List.take (ranked ~score:primary xs) k
    @ List.filter_map others ~f:(fun score -> List.hd (ranked ~score xs)))
