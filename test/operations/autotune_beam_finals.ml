(* gh-ocannl-1635: the beam finals of [Autotune.tune] -- the beam's contenders re-timed together
   before the crown, driven by a SCRIPTED clock (no reading of the machine's decides anything).

   The question the issue asked: does the beam compare readings taken in different windows? It does.
   Every candidate is timed in a window of its own, the beam keeps the entries whose windows read
   fastest, and the crown was the beam's head -- so a window's offset (drift, or contention that
   only ever slows a window, admitted when it is uniform) decided the crown as much as the
   schedule's speed did: the winner's curse [Bench_select] fixed for [projection_shape_bench]
   (gh-ocannl-1531). The finals recompile the near-tie contenders, re-time them together in adjacent
   windows (each twice, in [Autotune.finals_order]) and crown by those readings alone.

   Claimed, first over the search's own selection functions under a scripted clock (a seeded
   population, like [bench_select]'s): each candidate read in its own window of a clock whose
   offsets LAST -- a slow drift and co-tenant episodes spanning several windows, which is what
   survives a window's own least-of-its-batches -- the finals keep the true best more often than the
   head of the first readings does, and whenever the contenders held it, the finals crowned it in
   all but a fiftieth of the searches. The boundary: contention independent per window is not shared
   by adjacent windows, so re-timing whole windows cannot remove it (interleaving batch by batch
   could, with every contender resident at once, which gh-ocannl-1326 rules out); there the finals
   lose at most a fiftieth of the searches. With no window offsets, reading each contender twice
   keeps the true best no less often. Then the shapes the search relies on: which entries contend,
   the order of the windows, and the crown's ties and inconclusive case.

   Then in a real search ([Scripted_search], cc, [Fissioned_chain]'s small menu): the first new
   candidate is scripted the true best and the second the runner-up, every other candidate far
   slower, and the true best's FIRST window reads 8% slow -- a contended window. With the finals off
   ([autotune_finals_margin=0]) the search crowns the runner-up, the flaw the issue asked about;
   with them on (the default margin) it times exactly the same windows and then the two contenders,
   runner-up first, as the palindrome [runner-up; best; best; runner-up], and crowns the true best,
   whose routine computes the untuned output bit for bit. A search whose finals windows of the true
   best are all refused is inconclusive and keeps the runner-up, and those refusals count neither as
   candidate refusals nor against the search's cacheability. *)

open Base
open Stdio
open Verdict.Claims

(* --- the selection functions under a scripted clock --- *)

(* A 31-bit linear congruential generator: the population is scripted, identical on every platform
   and OCaml version, unlike [Random]. *)
let lcg = ref 20261026

let uniform () =
  lcg := ((1103515245 * !lcg) + 12345) land 0x7fffffff;
  Float.of_int !lcg /. 2147483648.

(* The clock. A window's reading is already the least of its batches, so a stall shorter than a
   window is filtered out of it; what offsets a whole window is what lasts longer than one -- a slow
   drift ([drift], a random walk advancing up to +-[drift]/2 per window) and co-tenant EPISODES,
   which start in a quiet window with probability [episode_start], end with probability 0.1 per
   window (ten windows on average), and slow every window they span by one amplitude drawn up to
   15%. [independent] instead contends each window apart, with probability 0.3 and up to 15%: the
   regime a window-granular re-timing cannot share. Every reading carries a jitter of
   +-[jitter]/2. *)
type clock = {
  mutable level : float;
  mutable episode : float option;
  drift : float;
  episode_start : float;
  independent : bool;
  jitter : float;
}

let clock ?(drift = 0.) ?(episode_start = 0.) ?(independent = false) ~jitter () =
  { level = 0.; episode = None; drift; episode_start; independent; jitter }

let window clock truth =
  clock.level <- clock.level +. ((uniform () -. 0.5) *. clock.drift);
  clock.episode <-
    (match clock.episode with
    | Some _ when Float.(uniform () < 0.1) -> None
    | Some a -> Some a
    | None when Float.(uniform () < clock.episode_start) -> Some (uniform () *. 0.15)
    | None -> None);
  let contended =
    Option.value clock.episode ~default:0.
    +. if clock.independent && Float.(uniform () < 0.3) then uniform () *. 0.15 else 0.
  in
  truth *. (1. +. clock.level) *. (1. +. contended) *. (1. +. ((uniform () -. 0.5) *. clock.jitter))

type outcome = { head_right : bool; held : bool; finals_right : bool }

(* One search's worth: 40 windows, eight of them the near-tie candidates, 1.5% apart, at random
   places in random order (the true best is index 0), the rest far slower ones; the beam keeps the
   two fastest readings, as the default [autotune_beam_width] does, and the finals re-time its
   contenders in the windows that follow, at the default margin and the width a beam of two gets. *)
let trial clock =
  clock.level <- 0.;
  clock.episode <- None;
  let truth = Array.init 8 ~f:(fun i -> 1. +. (0.015 *. Float.of_int i)) in
  let shuffled n =
    List.init n ~f:(fun i -> (uniform (), i))
    |> List.sort ~compare:(fun (a, _) (b, _) -> Float.compare a b)
    |> List.map ~f:snd
  in
  let at = Hashtbl.create (module Int) in
  List.iter2_exn
    (List.sort (List.take (shuffled 40) 8) ~compare:Int.compare)
    (shuffled 8)
    ~f:(fun slot i -> Hashtbl.set at ~key:slot ~data:i);
  let first =
    List.filter_map (List.init 40 ~f:Fn.id) ~f:(fun slot ->
        match Hashtbl.find at slot with
        | Some i -> Some (i, window clock truth.(i))
        | None ->
            ignore (window clock 2. : float);
            None)
  in
  let beam =
    List.take (List.stable_sort first ~compare:(fun (_, a) (_, b) -> Float.compare a b)) 2
  in
  let head = fst (List.hd_exn beam) in
  let contenders = Autotune.finals_contenders ~margin:0.1 ~width:2 beam in
  let crown =
    if List.is_empty contenders then head
    else
      let readings = Hashtbl.create (module Int) in
      List.iter (Autotune.finals_order contenders) ~f:(fun (i, _) ->
          Hashtbl.add_multi readings ~key:i ~data:(window clock truth.(i)));
      match
        Autotune.finals_crown
          (List.map contenders ~f:(fun (i, ms) -> ((i, ms), Hashtbl.find_multi readings i)))
      with
      | Some ((i, _), _) -> i
      | None -> head
  in
  {
    head_right = head = 0;
    held = List.exists contenders ~f:(fun (i, _) -> i = 0);
    finals_right = crown = 0;
  }

let () =
  let n = 2000 in
  let population clock = List.init n ~f:(fun _ -> trial clock) in
  let count l f = List.count l ~f in
  let tally l = (count l (fun o -> o.head_right), count l (fun o -> o.finals_right)) in
  (* A search's regime: drift and co-tenant episodes dominate a small per-reading jitter. *)
  let lasting = population (clock ~drift:0.004 ~episode_start:0.05 ~jitter:0.004 ()) in
  let h_lasting, f_lasting = tally lasting in
  let held = List.filter lasting ~f:(fun o -> o.held) in
  (* The boundary: contention independent per window. *)
  let independent = population (clock ~independent:true ~jitter:0.004 ()) in
  let h_independent, f_independent = tally independent in
  (* No window offsets at all, and a jitter wide enough that the head does err. *)
  let iid = population (clock ~jitter:0.05 ()) in
  let h_iid, f_iid = tally iid in
  printf
    "seeded population, %d searches of 40 windows, 8 near-tie candidates 1.5%% apart, beam of 2:\n"
    n;
  printf
    "  drift and co-tenant episodes, +-0.2%% jitter: the head %d, the finals %d kept the true best \
     (contenders held it %d times)\n"
    h_lasting f_lasting (List.length held);
  printf "  contention independent per window:          the head %d, the finals %d\n" h_independent
    f_independent;
  printf "  no window offsets, +-2.5%% jitter:            the head %d, the finals %d\n" h_iid f_iid;
  p "with lasting window offsets, the head misses the true best in over a fifth of the searches"
    (h_lasting * 5 < n * 4);
  p "the finals keep it in at least a fortieth more of them" (f_lasting * 40 >= (h_lasting * 40) + n);
  p_exists "the population holds searches whose contenders held the true best" lasting ~f:(fun o ->
      o.held);
  p "and whenever the contenders held the true best, the finals crowned it in all but a fiftieth"
    (count held (fun o -> not o.finals_right) * 50 <= List.length held);
  p "with contention independent per window, the head misses the true best sometimes"
    (h_independent < n);
  (* The rule's boundary: a contended window the neighbours do not share is not removed by re-timing
     whole windows -- only interleaving the contenders batch by batch shares it, which needs every
     contender resident at once (gh-ocannl-1326). The finals do not make that regime worse. *)
  p "and the finals lose at most a fiftieth of the searches there"
    ((h_independent - f_independent) * 50 <= n);
  p "with no window offsets, the head misses the true best sometimes" (h_iid < n);
  p "and the finals, reading each contender twice, keep it no less often" (f_iid >= h_iid)

(* The shapes the search relies on. *)
let () =
  let pool = [ ("a", 1.05); ("b", 1.0); ("c", Float.infinity); ("d", 1.2); ("e", 1.05) ] in
  let names l = List.map l ~f:fst in
  let contenders ~margin ~width = names (Autotune.finals_contenders ~margin ~width pool) in
  p "the contenders: within the margin of the fastest, by reading, ties in pool order"
    (List.equal String.equal (contenders ~margin:0.1 ~width:4) [ "b"; "a"; "e" ]);
  p "at most [width] of them"
    (List.equal String.equal (contenders ~margin:0.1 ~width:2) [ "b"; "a" ]);
  p "a wide margin admits every finite reading, never an unmeasured one"
    (List.equal String.equal (contenders ~margin:10. ~width:9) [ "b"; "a"; "e"; "d" ]);
  p_all "a margin of 0 or less is off" [ 0.; -0.5 ] ~f:(fun margin ->
      List.is_empty (contenders ~margin ~width:4));
  p_empty "a margin admitting the fastest alone holds no finals" ~over:pool
    (contenders ~margin:0.01 ~width:4);
  p_empty "nor does a width of one" ~over:pool (contenders ~margin:0.1 ~width:1);
  p "the windows: the contenders, then the same in reverse"
    (List.equal String.equal
       (Autotune.finals_order [ "x"; "y"; "z" ])
       [ "x"; "y"; "z"; "z"; "y"; "x" ]);
  let crown readings = Option.map (Autotune.finals_crown readings) ~f:(fun (c, ms) -> (c, ms)) in
  p "the crown: the least of each contender's readings, the least of those"
    (Poly.equal (crown [ ("x", [ 1.2; 1.1 ]); ("y", [ 1.15; 1.3 ]) ]) (Some ("x", 1.1)));
  p "a tie goes to the earlier contender"
    (Poly.equal (crown [ ("x", [ 1.1 ]); ("y", [ 1.1; 1.4 ]) ]) (Some ("x", 1.1)));
  p "a contender with no reading makes the finals inconclusive"
    (Option.is_none (crown [ ("x", [ 1.1 ]); ("y", []) ]))

(* --- a real search --- *)

let () = Fissioned_chain.small_menu ()
let margin_key = "autotune_finals_margin"
let backend_name = Context.backend_name (Context.cpu ())
let default_label = Autotune.default_seed_label ~backend_name
let is_floor label = Option.exists default_label ~f:(String.equal label)
let chain = Fissioned_chain.chain_with_output ()
let reference = Scripted_search.untuned (fst chain) (snd chain)

(* The script, per search: the candidates in the order their first windows came, the baseline and
   the untuned-default floor left out. The first is the true best (2.0 ms), the second the runner-up
   (2.1 ms), every later one far slower; the true best's first window reads 8% slow, within the
   default margin of the runner-up; [~refuse_best_finals] refuses every later window of the true
   best (a reading that is not finite refuses it, as a degenerate clock reading would). *)
let search ?(refuse_best_finals = false) ~settings ~tag () =
  let seen = ref [] in
  let windows_of = Hashtbl.create (module String) in
  let rank label =
    match List.findi (List.rev !seen) ~f:(fun _ l -> String.equal l label) with
    | Some (j, _) -> j
    | None ->
        seen := label :: !seen;
        List.length !seen - 1
  in
  let measure (w : Scripted_search.window) =
    if String.equal w.label "baseline" then 1000.
    else if is_floor w.label then 100.
    else
      let j = rank w.label in
      let k = Hashtbl.find windows_of w.label |> Option.value ~default:0 in
      Hashtbl.set windows_of ~key:w.label ~data:(k + 1);
      match j with
      | 0 when k = 0 -> 2.0 *. 1.08
      | 0 when refuse_best_finals -> Float.nan
      | 0 -> 2.0
      | 1 -> 2.1
      | j -> 2.5 +. (0.01 *. Float.of_int j)
  in
  let qg, comp = chain in
  let (o : Scripted_search.observed) =
    Scripted_search.search ~settings ~output:qg ~tag ~measure comp
  in
  let order = List.rev !seen in
  let best = List.nth order 0 and runner_up = List.nth order 1 in
  eprintf
    "%s (not part of the golden): crown %s %.2f ms, finals %s (%d contenders, %d timed, %d \
     admitted), true best %s, runner-up %s\n\
     measured: %s\n\
     %!"
    tag o.report.Autotune.best_label o.report.Autotune.best_ms
    (Autotune.finals_outcome_name o.report.Autotune.finals_outcome)
    (List.length o.report.Autotune.finals)
    o.report.Autotune.finals_timed o.report.Autotune.finals_admitted
    (Option.value best ~default:"none")
    (Option.value runner_up ~default:"none")
    (String.concat ~sep:"; " o.measured);
  (o, best, runner_up)

let () =
  let off, best, runner_up = search ~settings:[ (margin_key, "0") ] ~tag:"off" () in
  let on, best_on, runner_up_on = search ~settings:[] ~tag:"on" () in
  let refused, _, _ = search ~refuse_best_finals:true ~settings:[] ~tag:"refused" () in
  let best = Option.value_exn best and runner_up = Option.value_exn runner_up in
  let crown (o : Scripted_search.observed) = o.report.Autotune.best_label in
  p "setup: the scripted searches met the same candidates in the same order"
    (Option.equal String.equal best_on (Some best)
    && Option.equal String.equal runner_up_on (Some runner_up));
  p "setup: the true best is a candidate, not the baseline or the floor"
    ((not (String.equal best "baseline")) && not (is_floor best));
  (* The flaw: the first readings' head is the runner-up, read in a cleaner window. *)
  p "off: the search crowns the runner-up, whose window read faster than the true best's"
    (String.equal (crown off) runner_up);
  p "off: no finals ran"
    (Poly.equal off.report.Autotune.finals_outcome Autotune.Finals_not_run
    && off.report.Autotune.finals_timed = 0);
  (* The fix. *)
  p
    "on: the search's own windows are the finals-off search's, then the palindrome of the two \
     contenders, runner-up first"
    (List.equal String.equal on.measured (off.measured @ [ runner_up; best; best; runner_up ]));
  p "on: the finals crown the true best, swapping the head"
    (String.equal (crown on) best
    && Poly.equal on.report.Autotune.finals_outcome Autotune.Finals_swapped);
  p "on: two contenders, four windows, all admitted"
    (List.length on.report.Autotune.finals = 2
    && on.report.Autotune.finals_timed = 4
    && on.report.Autotune.finals_admitted = 4);
  p "on: the crown's best_ms is its search reading, the contended one"
    (Float.equal on.report.Autotune.best_ms (2.0 *. 1.08));
  p "on: the report holds each contender's search and finals readings, runner-up first"
    (Poly.equal on.report.Autotune.finals
       [
         {
           Autotune.contender_label = runner_up;
           contender_search_ms = 2.1;
           contender_finals_ms = Some 2.1;
         };
         {
           contender_label = best;
           contender_search_ms = 2.0 *. 1.08;
           contender_finals_ms = Some 2.0;
         };
       ]);
  p "on: the crown's search reading exceeds the runner-up's, which the finals outranked"
    (Float.(on.report.Autotune.best_ms > 2.1) && Autotune.finals_outranked on.report ~search_ms:2.1);
  p_all "on: every search reading is no faster than the crown's, or was outranked in the finals"
    (List.take on.windows (List.length off.windows))
    ~f:(fun (_, ms) ->
      Float.(on.report.Autotune.best_ms <= ms) || Autotune.finals_outranked on.report ~search_ms:ms);
  p "off: nothing is outranked" (not (Autotune.finals_outranked off.report ~search_ms:2.1));
  p "on: the finals windows are not candidate timings"
    (on.report.Autotune.candidates_timed = off.report.Autotune.candidates_timed);
  p_all2 "on: the crowned routine computes the untuned output bit for bit" on.output reference
    ~f:Float.equal;
  (* A refused finals window. *)
  p "refused: the true best's finals windows were refused, so the finals are inconclusive"
    (Poly.equal refused.report.Autotune.finals_outcome Autotune.Finals_inconclusive
    && refused.report.Autotune.finals_timed = 4
    && refused.report.Autotune.finals_admitted = 2);
  p "refused: the crown stays the first readings' head" (String.equal (crown refused) runner_up);
  p "refused: nothing is outranked" (not (Autotune.finals_outranked refused.report ~search_ms:2.1));
  p "refused: the finals' refusals are not the search's"
    (refused.report.Autotune.timings_contended = off.report.Autotune.timings_contended
    && refused.report.Autotune.timings_unreadable = off.report.Autotune.timings_unreadable)
