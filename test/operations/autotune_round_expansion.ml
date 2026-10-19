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
   64 bytes but not at 32, so the test failed on exactly the CI runners without AVX-512. So the seat
   is chosen by that property instead: a [full] search seats the last alternate, and if its
   differing segment offered no round move, the alternates are tried latest first, one [full] search
   each, until one does; the other rules then seat the same one. The small menu pins the width to 32
   bytes, so the menu is the same on every machine and the positional choice fails there: claimed as
   a setup control, so the selection is exercised on every run. A menu change under which the last
   alternate offers moves at 32 bytes fails that control, and the remedy is to pin a width where it
   does not (the small menu's, which the other small-menu tests share); one under which no alternate
   offers any fails [full: the round extends the near-copy on its differing segment]. Neither is to
   be weakened.

   Pinned to cc: the seeding and the scripted ranking are backend-independent, and cc is always
   available. *)
open Base
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let is_single = Fissioned_chain.is_single
let is_composite = Fissioned_chain.is_composite

(* The small menu (see the header), for every search here: stated before the first search. It pins
   the cc vector width too ([Fissioned_chain.vector_bytes]). *)
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
  composites : (string * string) list;
      (** Every timed composite's label and digest tag, in attempt order: the recombined composite,
          then its alternates. *)
}

let gain = 1.0
let loss = 0.05

(* One search under [rule] (stated at the config-file level for this search only), seating as the
   beam's near-copy the alternate labelled [seat], or without one the last alternate timed. *)
let search ?seat rule =
  let report = ref None and composites = ref [] and children = ref [] and measured = ref [] in
  (* Every timed candidate's digest tag -> (full digest, scripted time). *)
  let timed = Hashtbl.create (module String) in
  (* The two fastest composites: the beam round 1 expands (its singles, presets and baseline are
     scripted far slower). Fixed at the first round candidate. *)
  let beam = ref None in
  let old_measured = Autotune.Seam.get Autotune.on_candidate_measured in
  Exn.protect
    ~finally:(fun () ->
      Autotune.Seam.set Autotune.on_candidate_measured old_measured;
      Hashtbl.remove Utils.config_file_args expansion_key)
    ~f:(fun () ->
      Hashtbl.set Utils.config_file_args ~key:expansion_key ~data:rule;
      ( Autotune.Seam.set Autotune.on_candidate_measured @@ fun ~label ~digest _ms ->
        measured := label :: !measured;
        let ms =
          if is_single label then 100.
          else if is_composite label then (
            (* The recombined composite, then its alternates, all slower than it however many play.
               Without [seat] each alternate is faster than the one before, so the last seats; with
               it, [seat] is faster than every other. *)
            let n = List.length !composites in
            composites := (label, Autotune.digest_tag digest) :: !composites;
            if n = 0 then 10.
            else
              match seat with
              | None -> 10. +. (1. /. Float.of_int (n + 1))
              | Some seat when String.equal label seat -> 10.01
              | Some _ -> 10.5 +. (1. /. Float.of_int (n + 1)))
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
                let parent_ms =
                  Option.value_map (Hashtbl.find timed parent) ~default:1000. ~f:snd
                in
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
        ms );
      let ctx, _routine =
        Autotune.with_uncontended_test_windows (fun () ->
            Autotune.tune ~beam_width:2 ~rounds:1 ~repeats:1 ~cache_dir:"" ~log:false
              ~progress:false
              ~report:(fun r -> report := Some r)
              (Context.cpu ()) (Fissioned_chain.chain ()) Ir.Indexing.Empty)
      in
      Context.release ctx);
  match (!report, !beam) with
  | Some report, Some (head, near_copy, differing) ->
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
        measured = List.rev !measured;
        head;
        near_copy;
        differing;
        children;
        composites = List.rev !composites;
      }
  | Some _, None -> failwith (rule ^ ": the search timed no round candidate")
  | None, _ -> failwith (rule ^ ": the search delivered no report")

let labels children = List.map children ~f:(fun c -> c.label)
let from parent (o : observed) = List.filter o.children ~f:(fun c -> String.equal c.parent parent)

let round_best (o : observed) =
  List.min_elt o.children ~compare:(fun a b -> Float.compare a.ms b.ms)

let same_labels a b = List.equal String.equal (labels a) (labels b)

(* Whether a [full] search extended its near-copy on the segment where it differs from the head:
   what every claim below needs of the seated alternate. *)
let moves_on_differing (o : observed) =
  List.exists (from o.near_copy o) ~f:(fun c -> List.mem o.differing c.segment ~equal:Int.equal)

let seated_label (o : observed) =
  List.find_map o.composites ~f:(fun (label, tag) ->
      Option.some_if (String.equal tag o.near_copy) label)

(* The seat (see the header): the last alternate if its differing segment offers round moves,
   otherwise the latest-timed alternate whose does, tried one [full] search each. Returns the
   positional search, the [full] search at the seat, and the seat to script the other rules with:
   [None] where the positional search is the one kept, so every rule scripts the same times. *)
let select_seat () =
  let positional = search "full" in
  let tried = seated_label positional in
  let rec next = function
    | _ when moves_on_differing positional -> (positional, None)
    | [] -> (positional, None)
    | seat :: rest when Option.equal String.equal (Some seat) tried -> next rest
    | seat :: rest ->
        let o = search ~seat "full" in
        if moves_on_differing o then (o, Some seat) else next rest
  in
  let full, seat = next (List.rev_map (List.drop positional.composites 1) ~f:fst) in
  (positional, full, seat)

let () =
  let positional, full, seat = select_seat () in
  let differing = search ?seat "differing" in
  let head = search ?seat "head" in
  (* The beam's near-copy is a playoff alternate, so the claims below can fail only where each
     segment offers alternates to its best single. *)
  Fissioned_chain.menu_claim full.measured;
  (* The setup the rule is about, read off the [full] search. *)
  p_all "setup: every rule ran its round" [ full; differing; head ] ~f:(fun o ->
      o.report.Autotune.rounds_run = 1);
  p_all "setup: every rule entered round 1 with the same beam" [ differing; head ] ~f:(fun o ->
      String.equal o.head full.head && String.equal o.near_copy full.near_copy);
  (* The control: at the pinned width, seating by position seats an alternate the claims cannot use,
     so the selection above is exercised on every machine. *)
  p_none "setup: the last alternate, seated by position, is extended only on segments it shares"
    (from positional.near_copy positional) ~f:(fun c ->
      List.mem positional.differing c.segment ~equal:Int.equal);
  p "setup: the near-copy differs from the head in exactly one segment"
    (match full.differing with [ i ] -> i >= 0 | _ -> false);
  p_all "full: every round candidate extends a beam entry" full.children ~f:(fun c ->
      String.equal c.parent full.head || String.equal c.parent full.near_copy);
  p_exists "full: the round extends the near-copy on a segment it shares with the head"
    (from full.near_copy full) ~f:(fun c ->
      not (List.mem full.differing c.segment ~equal:Int.equal));
  p_exists "full: the round extends the near-copy on its differing segment"
    (from full.near_copy full) ~f:(fun c -> List.mem full.differing c.segment ~equal:Int.equal);
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
  let attempts = ref 0 in
  let old_attempt = Autotune.Seam.get Autotune.on_candidate_attempt in
  let outcome =
    Exn.protect
      ~finally:(fun () ->
        Autotune.Seam.set Autotune.on_candidate_attempt old_attempt;
        Hashtbl.remove Utils.config_file_args expansion_key)
      ~f:(fun () ->
        Hashtbl.set Utils.config_file_args ~key:expansion_key ~data:"widest";
        (Autotune.Seam.set Autotune.on_candidate_attempt @@ fun _ -> Int.incr attempts);
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
  p "malformed: nothing was compiled" (!attempts = 0)
