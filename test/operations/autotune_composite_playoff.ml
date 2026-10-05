(* gh-ocannl-1166: the composite playoff. A per-segment single is timed as the whole routine with
   every other segment on its untuned preset, so near-tie singles are ranked through that backdrop's
   noise; the playoff re-times them inside the recombined composite, one keyed segment at a time,
   and a faster alternate becomes the incumbent the later segments build on.

   The ranking is the machine's unless the test decides it, so every admitted window is scripted
   through [Autotune.on_candidate_measured] (gh-ocannl-1027) and the windows are opted out of host
   contention ([with_uncontended_test_windows], gh-ocannl-1156). Everything that is not a
   per-segment sketch is scripted far slower than any of them, so the crown is a composite's by
   construction. Four searches over one two-matmul chain.

   In [ties] every single measures the same, so each segment has near-tie contenders, and each later
   composite is scripted faster than the one before: every admitted playoff window swaps, so each
   window challenges the previous one, differs from it in one segment, and the crown is the last
   window.

   In [slower] the singles tie the same way but each later composite is slower: nothing swaps, every
   window differs from the recombined composite in one segment, and that composite keeps the crown.

   In [near] the singles are 0.3% apart in attempt order, inside the playoff margin, so each
   segment's runner-up single plays and the one behind it (0.6%) does not; in [spread] they are 1%
   apart, wider than the margin, so the playoff times nothing.

   Pinned to cc: the seeding and the scripted ranking are backend-independent, and cc is always
   available. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let is_single label = String.is_prefix label ~prefix:"F_sketch[" && not (String.mem label ',')
let is_composite label = String.is_prefix label ~prefix:"F_sketch[" && String.mem label ','
let q = 32

let matrix ~label ~modulus ~offset ~stride =
  TDSL.ndarray
    (Array.init (q * q) ~f:(Ll_test.cycle_flat ~dims:[| q; q |] ~modulus ~offset ~stride))
    ~label:[ label ] ~input_dims:[ q ] ~output_dims:[ q ] ()

(* Two matmul segments of different shapes (the multi-site chain of [autotune_fission_sketch]): the
   materialized sums cut the chain so each product is a keyed segment of its own. *)
let chain () =
  let qa = matrix ~label:"qa" ~modulus:11 ~offset:0. ~stride:0.125 in
  let qb = matrix ~label:"qb" ~modulus:7 ~offset:(-3.) ~stride:1. in
  let qc = matrix ~label:"qc" ~modulus:5 ~offset:(-2.) ~stride:0.5 in
  let qc16 =
    TDSL.ndarray
      (Array.init (16 * q)
         ~f:(Ll_test.cycle_flat ~dims:[| 16; q |] ~modulus:9 ~offset:(-4.) ~stride:0.25))
      ~label:[ "qc16" ] ~input_dims:[ q ] ~output_dims:[ 16 ] ()
  in
  let%op qd = qa + qb in
  Train.set_materialized qd.Tensor.value;
  let%op qe = qd * qc in
  Train.set_materialized qe.Tensor.value;
  let%op qg = qc16 * qe in
  Train.forward qg

type observed = {
  report : Autotune.report;
  composite_ms : (string * float) list;  (** Every admitted composite window, in attempt order. *)
}

(* One search under a scripted ranking: [single n] is the n-th single's time (0-based, attempt
   order), [composite n] the n-th composite's. *)
let search ~tag ~single ~composite =
  let report = ref None and singles = ref 0 and composites = ref [] in
  let old_measured = !Autotune.on_candidate_measured in
  Exn.protect
    ~finally:(fun () -> Autotune.on_candidate_measured := old_measured)
    ~f:(fun () ->
      (Autotune.on_candidate_measured :=
         fun ~label ~digest:_ _ms ->
           if is_single label then (
             let ms = single !singles in
             Int.incr singles;
             ms)
           else if is_composite label then (
             let ms = composite (List.length !composites) in
             composites := (label, ms) :: !composites;
             ms)
           else 1000.);
      let ctx, _routine =
        Autotune.with_uncontended_test_windows (fun () ->
            Autotune.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:""
              ~report:(fun r -> report := Some r)
              (Context.cpu ()) (chain ()) Ir.Indexing.Empty)
      in
      Context.release ctx);
  match !report with
  | Some report ->
      let r = report in
      Stdio.eprintf
        "%s (not part of the golden): singles %d, composite %s, playoff timed %d swaps %d, crown \
         %s %.2f ms\n\
         %!"
        tag !singles
        (match r.Autotune.fiss_sketch_composite with
        | `Timed -> "timed"
        | `Refused -> "refused"
        | `Proposed -> "proposed"
        | `Singles_refused -> "singles refused"
        | `Ineligible -> "ineligible")
        r.Autotune.fiss_sketch_playoff_timed r.Autotune.fiss_sketch_playoff_swaps
        r.Autotune.best_label r.Autotune.best_ms;
      List.iter (List.rev !composites) ~f:(fun (label, ms) ->
          Stdio.eprintf "  %s window (not part of the golden): %s %.2f ms\n%!" tag label ms);
      { report; composite_ms = List.rev !composites }
  | None -> failwith (tag ^ ": the search delivered no report")

(* A composite's per-segment entries: its label lists one per keyed segment, in key order. *)
let entries label =
  String.chop_prefix_exn label ~prefix:"F_sketch["
  |> String.chop_suffix_exn ~suffix:"]"
  |> String.split ~on:','

(* The segment positions where two composites differ. *)
let changed a b =
  let a = entries a and b = entries b in
  if List.length a <> List.length b then [ -1 ]
  else
    List.filter_mapi (List.zip_exn a b) ~f:(fun i (x, y) ->
        if String.equal x y then None else Some i)

(* Per-segment width: no segment position is challenged by more than [width] windows, given each
   window's challenged positions. *)
let width_claim name ~width positions =
  let all = List.concat positions in
  p_all name (List.dedup_and_sort all ~compare:Int.compare) ~f:(fun pos ->
      List.count all ~f:(Int.equal pos) <= width)

let faster n = 10. -. (0.1 *. Float.of_int n)

let () =
  let ties = search ~tag:"ties" ~single:(fun _ -> 100.) ~composite:faster in
  let r = ties.report in
  p "ties: the coarse composite was timed" (Poly.equal r.Autotune.fiss_sketch_composite `Timed);
  p "ties: the playoff timed alternates" (r.Autotune.fiss_sketch_playoff_timed >= 1);
  (* The coarse composite plus every playoff window, and nothing else: playoff windows are counted
     apart from [fiss_sketch_timed], and the scripted composites are the only multi-entry
     candidates. *)
  p "ties: every composite window is the recombination's or the playoff's"
    (List.length ties.composite_ms = 1 + r.Autotune.fiss_sketch_playoff_timed);
  p "ties: each faster alternate replaced the incumbent"
    (r.Autotune.fiss_sketch_playoff_swaps = r.Autotune.fiss_sketch_playoff_timed);
  (* Every window swapped, so each window's incumbent is the window before it: an alternate built
     from the recombined composite instead of the swapped incumbent differs from it in two segments.
     "At most one" rather than "exactly one": a label does not render every parameter (two register
     tile geometries of one tile print alike), so a genuine one-segment change can read as none. *)
  let labels = List.map ties.composite_ms ~f:fst in
  let challenges =
    List.zip_exn (List.drop_last_exn labels) (List.tl_exn labels)
    |> List.map ~f:(fun (incumbent, window) -> changed incumbent window)
  in
  p_all "ties: each window differs from the incumbent it challenged in at most one segment"
    challenges ~f:(fun c -> List.length c <= 1);
  p_exists "ties: some window visibly changes a segment" challenges ~f:(fun c -> List.length c = 1);
  width_claim "ties: no segment is challenged by more than two windows" ~width:2 challenges;
  (match List.last ties.composite_ms with
  | Some (label, ms) ->
      p "ties: the crown is the last alternate" (String.equal r.Autotune.best_label label);
      p "ties: the crown's time is the last alternate's" (Float.equal r.Autotune.best_ms ms)
  | None ->
      p "ties: the crown is the last alternate" false;
      p "ties: the crown's time is the last alternate's" false);
  let slower =
    search ~tag:"slower"
      ~single:(fun _ -> 100.)
      ~composite:(fun n -> 10. +. (0.1 *. Float.of_int n))
  in
  let r = slower.report in
  p "slower: the playoff timed alternates" (r.Autotune.fiss_sketch_playoff_timed >= 1);
  p "slower: no slower alternate replaced the incumbent" (r.Autotune.fiss_sketch_playoff_swaps = 0);
  (match slower.composite_ms with
  | (first, _) :: windows ->
      p "slower: the crown is the recombined composite" (String.equal r.Autotune.best_label first);
      let challenges = List.map windows ~f:(fun (window, _) -> changed first window) in
      p_all "slower: each window differs from the recombined composite in at most one segment"
        challenges ~f:(fun c -> List.length c <= 1);
      p_exists "slower: some window visibly changes a segment" challenges ~f:(fun c ->
          List.length c = 1);
      width_claim "slower: no segment is challenged by more than two windows" ~width:2 challenges
  | [] ->
      p "slower: the crown is the recombined composite" false;
      p "slower: each window differs from the recombined composite in at most one segment" false;
      p "slower: some window visibly changes a segment" false;
      p "slower: no segment is challenged by more than two windows" false);
  let near =
    search ~tag:"near" ~single:(fun n -> 100. *. Float.int_pow 1.003 n) ~composite:faster
  in
  let r = near.report in
  p "near: singles 0.3% apart reach the playoff" (r.Autotune.fiss_sketch_playoff_timed >= 1);
  let labels = List.map near.composite_ms ~f:fst in
  let challenges =
    match labels with
    | [] -> []
    | _ ->
        List.zip_exn (List.drop_last_exn labels) (List.tl_exn labels)
        |> List.map ~f:(fun (incumbent, window) -> changed incumbent window)
  in
  width_claim "near: only the single within the margin plays, one per segment" ~width:1 challenges;
  let spread =
    search ~tag:"spread" ~single:(fun n -> 100. *. Float.int_pow 1.01 n) ~composite:faster
  in
  let r = spread.report in
  p "spread: the coarse composite was timed" (Poly.equal r.Autotune.fiss_sketch_composite `Timed);
  p "spread: singles 1% apart leave the playoff nothing to time"
    (r.Autotune.fiss_sketch_playoff_timed = 0 && List.length spread.composite_ms = 1)
