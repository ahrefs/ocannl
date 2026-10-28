(* gh-ocannl-1229: which units of each beam entry a beam-expansion round expands
   ([autotune_round_expansion]).

   After the composite playoff (gh-ocannl-1166) the beam is the recombined composite and its
   near-copies: each runner-up is the head with one segment's sketch swapped. A fissioned routine's
   time is the sum of its segments' times, so on a segment a runner-up shares with the head its move
   is the head's same move over a backdrop already measured slower: that child can never beat the
   head's own. The default rule, [differing], expands the head in full and every other entry only on
   the segments where it differs from the head; [full] is the rule before gh-ocannl-1229, and
   [head], expanding the head alone, is the single-entry control.

   The ranking is the test's: every admitted window is scripted through
   [Autotune.on_candidate_measured] (gh-ocannl-1027), with the windows opted out of host contention
   ([with_uncontended_test_windows], gh-ocannl-1156). The seeds are scripted as the playoff test's
   [slower] scenario: singles tie, so every keyed segment plays alternates, and every alternate is
   slower than the recombined composite, so nothing swaps; one alternate is scripted faster than the
   others, so the beam (width 2) enters round 1 as the recombined composite plus that alternate
   (which one, below). A round child is scripted additively from its parent's time: a move on the
   near-copy's differing segment gains a full unit (it starts from the alternate geometry the head
   lacks); every other move loses a little. That landscape is the one the rule is built for, and on
   it a rule that expanded the head alone would lose the round's best, while one that skipped a
   differing segment would too.

   One search per rule, then the claims comparing them: [differing] times exactly [full]'s round
   candidates minus the near-copy's moves on the segments it shares with the head, and crowns the
   same round best; [head] times exactly the head's moves and misses that best. A malformed setting
   is refused before the search compiles anything.

   Every search runs on [Fissioned_chain]'s small menu: the roofline pre-filter keeps a fifth of
   each segment's sketches, eight and seven singles per segment instead of 21, and those compiles
   were most of the test's cost. That each segment still offers three or more is claimed ([menu]),
   as the playoff test claims it: the beam's second entry is a playoff alternate, and a segment
   offers one only beside its best single. Unlike the playoff's, these claims also depend on WHICH
   alternate seats: its differing segment must offer round moves, or the rule has nothing to tell
   apart. Some alternates' do not (tensorized or fully register-tiled geometries leave the round
   nothing to move), and which ones offer moves is the menu's, which the cc vector width reshapes:
   at 16 bytes (NEON), 32 (AVX2) and 64 (AVX-512) the pre-filter keeps different singles, and the
   alternate the scripting once seated by position -- the last one timed -- offered moves at 16 and
   64 bytes but not at 32, so the test failed on exactly the CI runners without AVX-512.

   So the seat is chosen from the report (gh-ocannl-1360): a probe search holds every alternate in a
   beam wide enough for all of them and stops at round 1's first attempt, after the round has
   recorded each entry's moves per segment ([Autotune.report.round_moves]) and before it has timed
   any; the seat is the latest-timed alternate whose differing segment offers moves, and every rule
   then seats it. The small menu pins the width to 32 bytes, so the menu is the same on every
   machine and the positional choice offers no move there: claimed off the probe as a setup control,
   so the selection is exercised on every run. A menu change under which the last alternate offers
   moves at 32 bytes fails that control, and the remedy is to pin a width where it does not (the
   small menu's, which the other small-menu tests share); one under which no alternate offers any
   fails [probe: some alternate offers a round move on the segment it differs in]. Neither is to be
   weakened. The report's counts are themselves claimed against the round: every unit's [unit_moves]
   is the number of round attempts naming its entry and segment, and a unit the rule leaves
   unexpanded is marked so.

   Pinned to cc: the seeding and the scripted ranking are backend-independent, and cc is always
   available. *)
open Base
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let is_single = Fissioned_chain.is_single
let is_composite = Fissioned_chain.is_composite

(* The small menu (see the header), for every search here: stated before the first search. It pins
   the cc vector file too ([Fissioned_chain.vector_file]). *)
let () = Fissioned_chain.small_menu ()
let expansion_key = "autotune_round_expansion"

(* A beam-round candidate's parent tag and the segment its move extends: [F_saved[<n> segs from
   <tag>]+seg<i>:<move>] (gh-ocannl-1230). *)
let round_move label =
  let ( let* ) = Option.( >>= ) in
  let* rest = String.chop_prefix label ~prefix:"F_saved[" in
  let* _, rest = String.lsplit2 rest ~on:' ' in
  let* rest = String.chop_prefix rest ~prefix:"segs from " in
  let* tag, ext = String.lsplit2 rest ~on:']' in
  let* ext = String.chop_prefix ext ~prefix:"+seg" in
  let* i, _move = String.lsplit2 ext ~on:':' in
  let* i = Int.of_string_opt i in
  Some (tag, i)

(* A fissioned candidate's per-segment digests, in the positions [+seg<i>] names. *)
let segment_digests digest = String.split digest ~on:'+'

type child = { label : string; parent : string; segment : int; ms : float }

type observed = {
  report : Autotune.report;
  measured : string list;  (** Every admitted window's label, in attempt order. *)
  head : string;  (** The beam head's digest tag as round 1 starts. *)
  near_copy : string;  (** The other beam entry's. *)
  differing : int list;  (** The segments where the near-copy differs from the head. *)
  children : child list;  (** Round 1's timed candidates, in attempt order. *)
  attempts : (string * int) list;
      (** Every round candidate's parent tag and segment, as the round attempted them, timed or not.
      *)
  composites : (string * string) list;
      (** Every timed composite's label and digest tag, in the order of their first windows: the
          recombined composite, then its alternates. *)
}

let gain = 1.0
let loss = 0.05

(* The [n]th timed composite's scripted time: the recombined composite, then its alternates, all
   slower than it however many play. Without [seat] each alternate is faster than the one before, so
   the last seats; with it, [seat] is faster than every other. *)
let composite_ms ?seat ~n label =
  if n = 0 then 10.
  else
    match seat with
    | None -> 10. +. (1. /. Float.of_int (n + 1))
    | Some seat when String.equal label seat -> 10.01
    | Some _ -> 10.5 +. (1. /. Float.of_int (n + 1))

(* A composite window's scripted time, recording a first window in [composites] (newest first). A
   composite timed already is being re-timed by a finals -- the playoff's near-tie finals
   (gh-ocannl-1665) or the beam finals -- and reads as its first window did. *)
let composite_window ?seat composites (w : Scripted_search.window) =
  match List.findi (List.rev !composites) ~f:(fun _ (label, _) -> String.equal label w.label) with
  | Some (n, _) -> composite_ms ?seat ~n w.label
  | None ->
      let n = List.length !composites in
      composites := (w.label, Autotune.digest_tag w.digest) :: !composites;
      composite_ms ?seat ~n w.label

(* One search under [rule] (stated at the config-file level for this search only), seating as the
   beam's near-copy the alternate labelled [seat], or without one the last alternate timed. *)
let search ?seat rule =
  let composites = ref [] and children = ref [] in
  (* Every timed candidate's digest tag -> (full digest, scripted time). *)
  let timed = Hashtbl.create (module String) in
  (* The two fastest composites: the beam round 1 expands (its singles, presets and baseline are
     scripted far slower). Fixed at the first round candidate. *)
  let beam = ref None in
  let measure (w : Scripted_search.window) =
    let label = w.label and digest = w.digest in
    let ms =
      if is_single label then 100.
      else if is_composite label then composite_window ?seat composites w
      else
        match round_move label with
        | None -> 1000.
        | Some (parent, segment) ->
            let head, near_copy, differing =
              match !beam with
              | Some b -> b
              | None ->
                  let ranked =
                    Hashtbl.to_alist timed
                    |> List.sort ~compare:(fun (_, (_, a)) (_, (_, b)) -> Float.compare a b)
                  in
                  let b =
                    match ranked with
                    | (h, (hd, _)) :: (n, (nd, _)) :: _ ->
                        let hs = segment_digests hd and ns = segment_digests nd in
                        let differing =
                          if List.length hs <> List.length ns then [ -1 ]
                          else
                            List.filter_mapi (List.zip_exn hs ns) ~f:(fun i (x, y) ->
                                if String.equal x y then None else Some i)
                        in
                        (h, n, differing)
                    | _ -> ("", "", [])
                  in
                  beam := Some b;
                  b
            in
            let parent_ms = Option.value_map (Hashtbl.find timed parent) ~default:1000. ~f:snd in
            let ms =
              if
                String.equal parent near_copy
                && (not (String.equal near_copy head))
                && List.mem differing segment ~equal:Int.equal
              then parent_ms -. gain
              else parent_ms +. loss
            in
            children := { label; parent; segment; ms } :: !children;
            ms
    in
    Hashtbl.set timed ~key:(Autotune.digest_tag digest) ~data:(digest, ms);
    ms
  in
  let (o : Scripted_search.observed) =
    Scripted_search.search
      ~settings:[ (expansion_key, rule) ]
      ~rounds:1 ~log:false ~progress:false ~tag:rule ~measure (Fissioned_chain.chain ())
  in
  match !beam with
  | Some (head, near_copy, differing) ->
      let report = o.report in
      let children = List.rev !children in
      Stdio.eprintf
        "%s, seating %s (not part of the golden): beam %s + %s (differing in segment(s) %s), %d \
         round candidate(s), crown %s %.2f ms\n\
         %!"
        rule
        (Option.value seat ~default:"the last alternate")
        head near_copy
        (String.concat ~sep:"," (List.map differing ~f:Int.to_string))
        (List.length children) report.Autotune.best_label report.Autotune.best_ms;
      {
        report;
        measured = o.measured;
        head;
        near_copy;
        differing;
        children;
        attempts = List.filter_map o.attempts ~f:round_move;
        composites = List.rev !composites;
      }
  | None -> failwith (rule ^ ": the search timed no round candidate")

let labels children = List.map children ~f:(fun c -> c.label)
let from parent (o : observed) = List.filter o.children ~f:(fun c -> String.equal c.parent parent)

let round_best (o : observed) =
  List.min_elt o.children ~compare:(fun a b -> Float.compare a.ms b.ms)

let same_labels a b = List.equal String.equal (labels a) (labels b)

(* Round 1's record in a report, if the search started that round. *)
let round1 (r : Autotune.report) =
  List.find r.Autotune.round_moves ~f:(fun m -> m.Autotune.round_index = 1)

let entry_of (r : Autotune.report) tag =
  Option.bind (round1 r) ~f:(fun m ->
      List.find m.Autotune.round_entries ~f:(fun e -> String.equal e.Autotune.entry_digest_tag tag))

(* An entry's units on the segments where it differs from the head, as the round recorded them; none
   when it does not differ segment by segment. *)
let differing_units (e : Autotune.round_entry_moves) =
  match e.entry_segments_differing with
  | Some segs ->
      List.filter e.entry_units ~f:(fun u ->
          Option.exists u.Autotune.unit_segment ~f:(List.mem segs ~equal:Int.equal))
  | None -> []

(* Whether an entry's menu, as the round recorded it, proposed a move on a segment where the entry
   differs from the head: what every claim below needs of the seated alternate. *)
let offers_on_differing e = List.exists (differing_units e) ~f:(fun u -> u.Autotune.unit_moves > 0)

(* The units of [tag]'s entry in round 1 of [r], none when it was not in that round's beam. *)
let units_of r tag =
  Option.value_map (entry_of r tag) ~default:[] ~f:(fun e -> e.Autotune.entry_units)

exception Probe_stop

(* Wide enough for the recombined composite and every alternate: the playoff's default width (2) on
   each of the chain's two keyed segments, plus the composite. Claimed, not assumed. *)
let probe_beam_width = 5

(* The probe (see the header): a [full] search scripted like the positional one, whose beam holds
   every alternate, stopped at round 1's first attempt -- after the round recorded its moves, before
   it compiled any. Returns the report delivered on the way out and the timed composites. *)
let probe () =
  let report = ref None and composites = ref [] in
  ignore
    ( Autotune.Seam.with_value Autotune.on_candidate_attempt (fun label ->
          if Option.is_some (round_move label) then raise Probe_stop)
    @@ fun () ->
      Scripted_search.scripted
        ~settings:[ (expansion_key, "full") ]
        ~measure:(fun w ->
          if is_single w.label then 100.
          else if is_composite w.label then composite_window composites w
          else 1000.)
      @@ fun () ->
      match
        Autotune.tune ~beam_width:probe_beam_width ~rounds:1 ~repeats:1 ~cache_dir:"" ~log:false
          ~progress:false
          ~report:(fun r -> report := Some r)
          (Context.cpu ()) (Fissioned_chain.chain ()) Ir.Indexing.Empty
      with
      | ctx, _ -> Context.release ctx
      | exception Probe_stop -> () );
  match !report with
  | Some report -> (report, List.rev !composites)
  | None -> failwith "probe: the search delivered no report"

(* Every unit of every entry round 1 recorded, with its entry's tag. *)
let recorded_units (o : observed) =
  Option.value_map (round1 o.report) ~default:[] ~f:(fun m ->
      List.concat_map m.Autotune.round_entries ~f:(fun e ->
          List.map e.Autotune.entry_units ~f:(fun u -> (o, e.entry_digest_tag, u))))

(* The round attempts naming [parent] and the unit's segment. *)
let attempts_on (o : observed) parent (u : Autotune.round_unit_moves) =
  List.count o.attempts ~f:(fun (p, segment) ->
      String.equal p parent && Option.equal Int.equal u.unit_segment (Some segment))

let () =
  let probe_report, probe_composites = probe () in
  let alternates = List.drop probe_composites 1 in
  let offers (_, tag) = Option.exists (entry_of probe_report tag) ~f:offers_on_differing in
  (* The latest-timed alternate whose differing segment offers moves. *)
  let seat = Option.map (List.find (List.rev alternates) ~f:offers) ~f:fst in
  Stdio.eprintf "probe (not part of the golden): %d alternate(s), seating %s\n%!"
    (List.length alternates)
    (Option.value seat ~default:"none");
  let full = search ?seat "full" in
  let differing = search ?seat "differing" in
  let head = search ?seat "head" in
  (* The beam's near-copy is a playoff alternate, so the claims below can fail only where each
     segment offers alternates to its best single. *)
  Fissioned_chain.menu_claim full.measured;
  (* The probe the seat is chosen from. *)
  p "probe: round 1 recorded the recombined composite as its head"
    (match (round1 probe_report, probe_composites) with
    | Some { Autotune.round_entries = e :: _; _ }, (_, tag) :: _ ->
        String.equal e.Autotune.entry_digest_tag tag
    | _ -> false);
  p_all "probe: its beam holds every alternate" alternates ~f:(fun (_, tag) ->
      Option.is_some (entry_of probe_report tag));
  p "probe: some alternate offers a round move on the segment it differs in" (Option.is_some seat);
  (* The control: at the pinned width, the alternate seated by position offers no move on its
     differing segment, so the selection above is exercised on every machine. *)
  p_none "setup: the last alternate, seated by position, offers no round move where it differs"
    (Option.value_map (List.last alternates) ~default:[] ~f:(fun (_, tag) ->
         Option.value_map (entry_of probe_report tag) ~default:[] ~f:differing_units))
    ~f:(fun u -> u.Autotune.unit_moves > 0);
  (* The setup the rule is about, read off the [full] search. *)
  p_all "setup: every rule ran its round" [ full; differing; head ] ~f:(fun o ->
      o.report.Autotune.rounds_run = 1);
  p_all "setup: every rule entered round 1 with the same beam" [ differing; head ] ~f:(fun o ->
      String.equal o.head full.head && String.equal o.near_copy full.near_copy);
  p "setup: the seated alternate is the near-copy"
    (List.exists full.composites ~f:(fun (label, tag) ->
         Option.exists seat ~f:(String.equal label) && String.equal tag full.near_copy));
  p "setup: the near-copy differs from the head in exactly one segment"
    (match full.differing with [ i ] -> i >= 0 | _ -> false);
  p_all "full: every round candidate extends a beam entry" full.children ~f:(fun c ->
      String.equal c.parent full.head || String.equal c.parent full.near_copy);
  p_exists "full: the round extends the near-copy on a segment it shares with the head"
    (from full.near_copy full) ~f:(fun c ->
      not (List.mem full.differing c.segment ~equal:Int.equal));
  p_exists "full: the round extends the near-copy on its differing segment"
    (from full.near_copy full) ~f:(fun c -> List.mem full.differing c.segment ~equal:Int.equal);
  (* The report's record of the round (gh-ocannl-1360), against what the round did. *)
  p_all "report: every unit's move count is the round's attempts on it"
    (List.concat_map [ full; differing; head ] ~f:recorded_units)
    ~f:(fun (o, parent, u) -> u.Autotune.unit_moves = attempts_on o parent u);
  p_all "report: the round attempted nothing the units do not count" [ full; differing; head ]
    ~f:(fun o ->
      List.length o.attempts
      = List.sum (module Int) (recorded_units o) ~f:(fun (_, _, u) -> u.Autotune.unit_moves));
  p_all "report: round 1 is recorded under its rule, over the beam it extended"
    [ ("full", full); ("differing", differing); ("head", head) ]
    ~f:(fun (rule, o) ->
      match round1 o.report with
      | Some m ->
          String.equal m.Autotune.round_expansion_rule rule
          && List.equal String.equal
               (List.map m.round_entries ~f:(fun e -> e.Autotune.entry_digest_tag))
               [ o.head; o.near_copy ]
      | None -> false);
  p "report: the near-copy's differing segments are the ones its digest differs in"
    (Option.exists (entry_of full.report full.near_copy) ~f:(fun e ->
         Option.equal (List.equal Int.equal) e.Autotune.entry_segments_differing
           (Some full.differing)));
  p "report: full's near-copy offers a move on its differing segment"
    (Option.exists (entry_of full.report full.near_copy) ~f:offers_on_differing);
  p_all "report: differing expands the near-copy on its differing segment alone"
    (units_of differing.report differing.near_copy) ~f:(fun u ->
      Bool.equal u.Autotune.unit_expanded
        (Option.exists u.unit_segment ~f:(List.mem differing.differing ~equal:Int.equal)));
  p_none "report: head expands no unit of the near-copy" (units_of head.report head.near_copy)
    ~f:(fun u -> u.Autotune.unit_expanded);
  (* [differing]: the head in full, the near-copy on its differing segment only. *)
  p_exists "differing: the round extends the head" (from differing.head differing) ~f:(fun _ ->
      true);
  p "differing: the head's moves are exactly full's"
    (same_labels (from differing.head differing) (from full.head full));
  p_all "differing: the near-copy's moves are on its differing segment"
    (from differing.near_copy differing) ~f:(fun c ->
      List.mem differing.differing c.segment ~equal:Int.equal);
  p "differing: the near-copy's moves are exactly full's on that segment"
    (same_labels
       (from differing.near_copy differing)
       (List.filter (from full.near_copy full) ~f:(fun c ->
            List.mem full.differing c.segment ~equal:Int.equal)));
  p "differing: the round is smaller than full's"
    (List.length differing.children < List.length full.children);
  (match (round_best full, round_best differing) with
  | Some f, Some d ->
      p "differing: the round's best is full's"
        (String.equal f.label d.label && Float.equal f.ms d.ms);
      p "differing: the round's best gained on the near-copy's differing segment"
        (String.equal d.parent differing.near_copy
        && List.mem differing.differing d.segment ~equal:Int.equal);
      p "differing: it crowns what full crowns"
        (String.equal differing.report.Autotune.best_label full.report.Autotune.best_label
        && Float.equal differing.report.Autotune.best_ms full.report.Autotune.best_ms);
      p "differing: the crown is the round's best"
        (Float.equal differing.report.Autotune.best_ms d.ms)
  | _ ->
      p "differing: the round's best is full's" false;
      p "differing: the round's best gained on the near-copy's differing segment" false;
      p "differing: it crowns what full crowns" false;
      p "differing: the crown is the round's best" false);
  (* [head], the single-entry control. *)
  p_all "head: every round candidate extends the head" head.children ~f:(fun c ->
      String.equal c.parent head.head);
  p "head: the head's moves are exactly full's" (same_labels head.children (from full.head full));
  (match (round_best full, round_best head) with
  | Some f, Some h -> p "head: the control misses the round's best" Float.(h.ms > f.ms)
  | _ -> p "head: the control misses the round's best" false);
  p "head: the control's crown is the seed winner, the round falling short"
    Float.(head.report.Autotune.best_ms > differing.report.Autotune.best_ms)

(* A malformed rule refuses the search before it compiles anything. *)
let () =
  let outcome, (record : Scripted_search.record) =
    Scripted_search.scripted
      ~settings:[ (expansion_key, "widest") ]
      ~measure:(fun w -> w.reading)
      (fun () ->
        match
          Autotune.tune ~beam_width:2 ~rounds:1 ~repeats:1 ~cache_dir:"" ~log:false ~progress:false
            (Context.cpu ()) (Fissioned_chain.chain ()) Ir.Indexing.Empty
        with
        | ctx, _ ->
            Context.release ctx;
            `Accepted
        | exception Utils.Invalid_config_value { key; _ } -> `Refused key)
  in
  (match outcome with
  | `Refused key ->
      p "malformed: the refusal names the key" (String.equal key "autotune_round_expansion")
  | `Accepted -> p "malformed: the refusal names the key" false);
  p "malformed: nothing was compiled" (List.length record.attempts = 0)
