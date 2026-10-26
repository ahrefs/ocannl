(* gh-ocannl-1138: staffed singles. A per-segment sketch single used to be timed as the whole
   routine with every other segment on its untuned preset, a backdrop several times the composite
   the single would join, so a search's best stayed flat until the recombination composites it timed
   last. With config [autotune_staffed_singles] the search sweeps the keyed segments one at a time,
   each single staffed with the running composite's choice for every other key, and after a key's
   sweep its fastest single joins the composite.

   The ranking is the test's: every admitted window is scripted through
   [Autotune.on_candidate_measured] (gh-ocannl-1027), the windows opted out of host contention
   ([with_uncontended_test_windows], gh-ocannl-1156). The script is ADDITIVE, the model the sweep is
   built for: a per-segment candidate measures the sum of its scheduled segments' costs plus a
   preset's cost for each keyed segment it leaves alone, a segment's cost a function of its label
   entry ([params @key]); every other candidate measures far slower, so the crown is a per-segment
   one by construction. The searches run over [Fissioned_chain]'s two-matmul chain on its small
   menu, so the chain keys two segments with three or more singles each ([menu_claim]); on cc no
   seeded untuned plan exists, so the running composite starts empty unless [on_staffed_start] gives
   it a start.

   [off] is the reference search; [on] the same search staffed. From the script alone the test
   derives the composite each key's sweep must have been staffed with, and claims every staffed
   single carried exactly that staff; that the staffed search times the reference's singles in the
   reference's order, only their backdrops differing, and one window fewer (the reference's coarse
   composite, which the last key's winning staffed single already timed), with the playoff off in
   both so the count is the sweep's alone; that its best reaches the crown inside the seed pass,
   where the reference's stays short of it until the composite; and that the crowned routine
   computes what the untuned compile computes, bit for bit (the chain's inputs are dyadic and its
   sums exact, so no reassociation can move a bit). The crown then replays from a schedule cache,
   computing the same.

   [start] gives the composite a start: the first key's last single, scripted the fastest of its
   key, and the second key's last single, scripted the slowest. The first key's own choice is swept
   first, staffed with the second's; every slower replacement leaves it as it was; the second key's
   start choice, by then the timed composite itself, is not compiled again; and a faster single
   replaces it.

   [refused] fault-injects a refusal of the first staffed single ([on_candidate_preflight]): it is
   retried bare at once, and that bare twin is timed. [refused_all] refuses every attempt of the
   second key, staffed and bare: none is admitted, so every staffed refusal is retried bare, and the
   progress record never counts past an announced total; once a staffed single of the key is
   admitted ([refused_after]), no later refusal is retried. [slower] keeps a key whose sketches all
   lose to its preset out of the composite, and [shrunk] keeps a timed composite's outcome after a
   bare replacement. [refused_staffed] refuses only the second key's staffed attempts: their bare
   retries are timed, and the composite takes the fastest of them. [partial] cuts a search at its
   first admitted staffed window: the partial report already says the composite was timed.

   [ties] runs the playoff over a staffed search whose singles tie within each key: the last key's
   sweep ran inside the crowned composite already, so only the first key's alternates play.

   Pinned to cc: the scripted ranking is backend-independent, and cc is always available. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let staffed_key = "autotune_staffed_singles"
let width_key = "autotune_composite_playoff_width"
let () = Fissioned_chain.small_menu ()

(* A per-segment label's entries ([params @key]), its focus and its staff apart. *)
let parts label =
  let body =
    String.chop_prefix_exn label ~prefix:"F_sketch["
    |> String.chop_suffix_exn ~suffix:"]"
    |> String.chop_prefix_if_exists ~prefix:"fine "
  in
  let entries s = if String.is_empty s then [] else String.split s ~on:',' in
  let sep = " | staff " in
  match String.substr_index body ~pattern:sep with
  | Some i ->
      (entries (String.prefix body i), entries (String.drop_prefix body (i + String.length sep)))
  | None -> (entries body, [])

let focus label = fst (parts label)
let staff label = snd (parts label)
let key_of entry = List.last_exn (String.split entry ~on:'@')
let is_sketch label = String.is_prefix label ~prefix:"F_sketch["
let is_single = Fissioned_chain.is_single
let is_composite = Fissioned_chain.is_composite
let is_staffed = Fissioned_chain.is_staffed

(* A bare single's label from a staffed one's: its focus alone. *)
let bare label = Fissioned_chain.focus label ^ if is_staffed label then "]" else ""

(* A segment's scripted cost when scheduled by [entry]: a spread of distinct values, so each key has
   one fastest single. *)
let hashed entry = 2. +. (Float.of_int (Stdlib.Hashtbl.hash entry % 1000) /. 100.)
let preset_ms = 50.
let keyed_segments = 2

let routine_ms ?(keyed = keyed_segments) ~cost label =
  if is_sketch label then
    let f, s = parts label in
    let entries = f @ s in
    List.sum (module Float) entries ~f:cost
    +. (preset_ms *. Float.of_int (keyed - List.length entries))
  else 1000.

(* An entry's label text, as [parts] reads it off a candidate label. *)
let entry_of key p =
  Autotune.fiss_sketch_label ~fine:false [ (key, p) ]
  |> String.chop_prefix_exn ~prefix:"F_sketch["
  |> String.chop_suffix_exn ~suffix:"]"

type observed = Scripted_search.observed

(* The beam finals (gh-ocannl-1635) off in every search here: the scripts tie composites, which the
   finals would re-time after the search's own windows, and the claims count the composite windows.
   The finals are [autotune_beam_finals]' subject. *)
let finals_off = ("autotune_finals_margin", "0")

(* One search under the additive script [cost]. [?staffed] and [?width] state
   [autotune_staffed_singles] and [autotune_composite_playoff_width] for this search only; [?start]
   gives the running composite its start; [?refuse] refuses (at pre-dispatch validation, a contained
   decline) every attempt whose label it holds for. *)
let search ?(staffed = true) ?(log = false) ?(progress = false) ?keyed ?width ?start ?refuse
    ?(cache_dir = "") ?chain ~tag ~cost () =
  let qg, comp = match chain with Some c -> c | None -> Fissioned_chain.chain_with_output () in
  let current = ref "" in
  let settings =
    (staffed_key, Bool.to_string staffed)
    :: finals_off
    :: Option.value_map width ~default:[] ~f:(fun w -> [ (width_key, Int.to_string w) ])
  in
  let (o : observed) =
    Autotune.Seam.with_value Autotune.on_candidate_attempt (fun label -> current := label)
    @@ fun () ->
    Autotune.Seam.with_value Autotune.on_candidate_preflight (fun _routine ->
        match refuse with
        | Some refused when refused !current -> failwith ("injected refusal of " ^ !current)
        | Some _ | None -> ())
    @@ fun () ->
    Autotune.Seam.with_value Autotune.on_staffed_start (fun ~menu seeded ->
        match start with Some f -> f ~menu | None -> seeded)
    @@ fun () ->
    Scripted_search.search ~settings ~output:qg ~cache_dir ~log ~progress ~tag
      ~measure:(fun w -> routine_ms ?keyed ~cost w.label)
      comp
  in
  let report = o.report in
  Stdio.eprintf
    "%s (not part of the golden): timed %d, staffed %d, fallbacks %d, playoff %d, crown %s %.2f ms\n\
     %!"
    tag report.Autotune.candidates_timed report.Autotune.fiss_sketch_staffed_timed
    report.Autotune.fiss_sketch_backdrop_fallbacks report.Autotune.fiss_sketch_playoff_timed
    report.Autotune.best_label report.Autotune.best_ms;
  o

(* The keys in sweep order, by their singles' first appearance among [measured]. *)
let keys measured =
  List.filter measured ~f:is_single
  |> List.map ~f:(fun l -> key_of (List.hd_exn (focus l)))
  |> List.fold ~init:[] ~f:(fun seen k ->
      if List.mem seen k ~equal:String.equal then seen else k :: seen)
  |> List.rev

let singles_of key measured =
  List.filter measured ~f:(fun l ->
      is_single l && String.equal (key_of (List.hd_exn (focus l))) key)

(* The fastest single of [key] under [cost], by its focus entry: within one key's sweep the backdrop
   is common to every single, so the ranking is the entries' own. *)
let fastest ~cost key measured =
  List.map (singles_of key measured) ~f:(fun l -> List.hd_exn (focus l))
  |> List.min_elt ~compare:(fun a b -> Float.compare (cost a) (cost b))

(* 1-based position of the first admitted window labeled [label]. *)
let position measured label =
  Option.map (List.findi measured ~f:(fun _ l -> String.equal l label)) ~f:(fun (i, _) -> i + 1)

let last_step (r : Autotune.report) = List.last r.Autotune.best_steps

(* --- off and on: the same search, bare and staffed --- *)

let cost = hashed
let cache_dir = "autotune_cache_staffed"

(* An earlier run's entries would turn the staffed search into a replay. *)
let () = Scripted_search.clear_cache_dir cache_dir
let chain = Fissioned_chain.chain_with_output ()
let reference = Scripted_search.untuned (fst chain) (snd chain)
let off = search ~staffed:false ~width:0 ~tag:"off" ~cost ()
let on = search ~width:0 ~cache_dir ~chain ~tag:"on" ~cost ()

let () =
  Fissioned_chain.menu_claim off.measured;
  let r = on.report in
  p_none "off: no candidate carries a staff" off.attempts ~f:is_staffed;
  p "off: the report counts no staffed window or fallback"
    (off.report.Autotune.fiss_sketch_staffed_timed = 0
    && off.report.Autotune.fiss_sketch_backdrop_fallbacks = 0);
  (match keys on.measured with
  | [ first; second ] ->
      p_all "on: the first key is swept bare (the composite starts empty)"
        (singles_of first on.measured) ~f:(fun l -> List.is_empty (staff l));
      (match fastest ~cost first on.measured with
      | Some best ->
          p_all "on: every single of the second key is staffed with the first key's fastest"
            (singles_of second on.measured) ~f:(fun l -> List.equal String.equal (staff l) [ best ])
      | None -> p "on: every single of the second key is staffed with the first key's fastest" false);
      p "on: every staffed window is counted"
        (r.Autotune.fiss_sketch_staffed_timed = List.length (singles_of second on.measured));
      (* The crown is the composite of each key's fastest. *)
      p "on: the crown is a staffed single holding each key's fastest"
        (is_staffed r.Autotune.best_label
        && List.equal String.equal
             (List.sort ~compare:String.compare
                (focus r.Autotune.best_label @ staff r.Autotune.best_label))
             (List.sort ~compare:String.compare
                (List.filter_opt
                   [ fastest ~cost first on.measured; fastest ~cost second on.measured ])))
  | ks ->
      Stdio.eprintf "on: swept keys %s\n%!" (String.concat ~sep:" " ks);
      p "on: the search swept the chain's two keys" false);
  p "on: no bare retry without a refusal" (r.Autotune.fiss_sketch_backdrop_fallbacks = 0);
  (* Candidate coverage: staffing changes a single's backdrop, never which singles are timed or in
     what order. *)
  p "on: the reference's singles, in the reference's order, only their backdrops differing"
    (List.equal String.equal
       (List.filter off.measured ~f:is_single)
       (List.map (List.filter on.measured ~f:is_single) ~f:bare));
  p "on: one window fewer than the reference, its coarse composite timed already"
    (r.Autotune.candidates_timed = off.report.Autotune.candidates_timed - 1);
  p_none "on: no composite window was timed" on.measured ~f:is_composite;
  p "on and off: the same crown time" (Float.equal r.Autotune.best_ms off.report.Autotune.best_ms);
  (* The best improves during the seeds: it reaches the crown at the crown's own window, a staffed
     single, where the reference's reaches it only at its coarse composite. *)
  let reached_at (o : observed) =
    match last_step o.report with
    | Some (n, ms) ->
        Float.equal ms o.report.Autotune.best_ms
        && Option.equal Int.equal (Some n) (position o.measured o.report.Autotune.best_label)
    | None -> false
  in
  p "on: the best reaches the crown at the crown's own window" (reached_at on);
  p "off: the best reaches the crown at the crown's own window" (reached_at off);
  p "on: the crown's window is a single of the seed pass" (is_single r.Autotune.best_label);
  p "off: the reference's crown is its coarse composite (the flat best this fixes)"
    (is_composite off.report.Autotune.best_label);
  (* Executed: the crown computes what the untuned compile does. *)
  p_all2 "on: the crowned routine computes the untuned output bit for bit" on.output reference
    ~f:Float.equal;
  let replay = search ~width:0 ~cache_dir ~chain ~tag:"replay" ~cost () in
  p "replay: the staffed crown replays from the schedule cache"
    (match replay.report.Autotune.outcome with Autotune.Cache_replay -> true | _ -> false);
  (* A replay is labeled by its saved form, so it is the schedule and the stored time that match. *)
  p "replay: the crown's schedule, at the time the search stored"
    (Poly.equal replay.report.Autotune.best_schedule r.Autotune.best_schedule
    && Float.equal replay.report.Autotune.best_ms r.Autotune.best_ms);
  p_all2 "replay: the replayed routine computes the untuned output bit for bit" replay.output
    reference ~f:Float.equal

(* --- start: a composite with a choice for each key before its sweep --- *)

let () =
  let cheap = ref "" and dear = ref "" in
  let start ~menu =
    match menu with
    | [ (k1, ps1); (k2, ps2) ] ->
        let p1 = List.last_exn ps1 and p2 = List.last_exn ps2 in
        cheap := entry_of k1 p1;
        dear := entry_of k2 p2;
        [ (k1, p1); (k2, p2) ]
    | _ -> []
  in
  let cost entry =
    if String.equal entry !cheap then 1. else if String.equal entry !dear then 20. else hashed entry
  in
  let o = search ~width:0 ~start ~tag:"start" ~cost () in
  let r = o.report in
  p "start: the seam gave both keys a choice"
    ((not (String.is_empty !cheap)) && not (String.is_empty !dear));
  match keys o.measured with
  | [ first; second ] -> (
      let firsts = singles_of first o.measured and seconds = singles_of second o.measured in
      (match firsts with
      | head :: _ ->
          p "start: the first key's own choice is swept first, staffed with the second's"
            (List.equal String.equal (focus head) [ !cheap ]
            && List.equal String.equal (staff head) [ !dear ])
      | [] -> p "start: the first key's own choice is swept first, staffed with the second's" false);
      p_all "start: every slower replacement left the first key's choice as it was" seconds
        ~f:(fun l -> List.equal String.equal (staff l) [ !cheap ]);
      p_none "start: the second key's start choice, the timed composite by then, is not timed again"
        seconds ~f:(fun l -> List.equal String.equal (focus l) [ !dear ]);
      match fastest ~cost second o.measured with
      | Some best ->
          p "start: a faster single replaced the second key's start choice"
            ((not (String.equal best !dear))
            && List.equal String.equal
                 (List.sort ~compare:String.compare
                    (focus r.Autotune.best_label @ staff r.Autotune.best_label))
                 (List.sort ~compare:String.compare [ !cheap; best ]))
      | None -> p "start: a faster single replaced the second key's start choice" false)
  | _ -> p "start: the search swept the chain's two keys" false

(* --- start_refused: the timed composite proves its staff compatible --- *)

(* As in [start], both keys start with a choice. After the first key's sweep the composite is timed,
   and the second key's own choice is that composite: reused without a compile, it proves the second
   key's staff compatible (Codex P2 on PR #1215), so its later staffed singles, all refused here,
   are not retried bare. *)
let () =
  let second = ref "" in
  let start ~menu =
    match menu with
    | [ (k1, ps1); (k2, ps2) ] ->
        second := key_of (entry_of k2 (List.last_exn ps2));
        [ (k1, List.last_exn ps1); (k2, List.last_exn ps2) ]
    | _ -> []
  in
  let refuse label =
    is_staffed label
    && String.equal (key_of (List.hd_exn (focus label))) !second
    && not (String.is_empty !second)
  in
  let o = search ~width:0 ~start ~refuse ~tag:"start_refused" ~cost () in
  let refused_attempts =
    List.count o.attempts ~f:(fun l ->
        is_staffed l && String.equal (key_of (List.hd_exn (focus l))) !second)
  in
  p "start_refused: no refusal after the reused composite is retried bare"
    (refused_attempts >= 3 && o.report.Autotune.fiss_sketch_backdrop_fallbacks = 0)

(* --- refused: a staffed refusal is retried bare --- *)

let () =
  let refused = ref None in
  let refuse label =
    match !refused with
    | None when is_staffed label ->
        refused := Some label;
        true
    | Some l -> String.equal l label
    | None -> false
  in
  let o = search ~width:0 ~refuse ~tag:"refused" ~cost () in
  let r = o.report in
  match !refused with
  | None -> p "refused: a staffed single was refused" false
  | Some label ->
      p "refused: one staffed single refused, one bare retry"
        (r.Autotune.fiss_sketch_backdrop_fallbacks = 1);
      (match List.findi o.attempts ~f:(fun _ l -> String.equal l label) with
      | Some (i, _) ->
          p "refused: the attempt after the refusal is its bare twin"
            (Option.equal String.equal (List.nth o.attempts (i + 1)) (Some (bare label)))
      | None -> p "refused: the attempt after the refusal is its bare twin" false);
      p "refused: the bare twin was timed" (Option.is_some (position o.measured (bare label)));
      p_none "refused: the refused staffed single was never timed" o.measured
        ~f:(String.equal label);
      p "refused: the refusal is a contained pre-dispatch decline"
        (List.exists r.Autotune.declines ~f:(fun d ->
             match d.Autotune.key with
             | Ir.Schedule_outcome.Unclassified_key (Ir.Schedule_outcome.Preflight, _) -> true
             | _ -> false))

(* --- refused_all: until a staffed single of the key is admitted, every refusal is retried --- *)

(* Every attempt of the second key is refused, staffed and bare: no staffed single of it is ever
   admitted, so nothing shows its staff compatible, and each staffed refusal is retried bare (Codex
   P1 on PR #1215). The search's progress record, captured, never counts an attempt past its phase's
   announced total (Codex P2 on PR #1215: the retries are attempts the seed total did not count). *)
let () =
  let second = ref None in
  (* The second key is the first one a staffed attempt focuses on. *)
  let refuse label =
    if is_staffed label && Option.is_none !second then
      second := Some (key_of (List.hd_exn (focus label)));
    is_sketch label
    && (not (is_composite label))
    && Option.exists !second ~f:(fun k -> String.equal k (key_of (List.hd_exn (focus label))))
  in
  let o, lines =
    Stderr_capture.lines (fun () ->
        search ~progress:true ~width:0 ~refuse ~tag:"refused_all" ~cost ())
  in
  let r = o.report in
  (match !second with
  | None -> p "refused_all: a staffed single was refused" false
  | Some k ->
      let ofk =
        List.filter o.attempts ~f:(fun l ->
            is_single l && String.equal (key_of (List.hd_exn (focus l))) k)
      in
      let staffed_n = List.count ofk ~f:is_staffed in
      p "refused_all: every staffed refusal was retried bare, none of them admitted"
        (staffed_n >= 3
        && r.Autotune.fiss_sketch_backdrop_fallbacks = staffed_n
        && List.count ofk ~f:(fun l -> not (is_staffed l)) = staffed_n));
  (* [tried=<k>/<total>], with [?] for a phase announced without a total. *)
  let tried =
    List.filter_map (Stderr_capture.Progress.of_lines lines) ~f:(fun line ->
        Option.bind (Stderr_capture.Progress.field line "tried") ~f:(fun v ->
            match String.split v ~on:'/' with [ k; t ] -> Some (k, t) | _ -> None))
  in
  p_all "progress: no attempt is counted past its phase's announced total" tried ~f:(fun (k, t) ->
      String.equal t "?" || Int.of_string k <= Int.of_string t)

(* --- refused_after: once a staffed single of the key is admitted, refusals are its seeds' own
   --- *)

(* The second key's first staffed single is admitted, and every later attempt of that key, staffed
   or bare, is refused: its staff is shown compatible, so no refusal is retried bare. *)
let () =
  let second = ref None and admitted = ref false in
  let refuse label =
    if not (is_staffed label) then false
    else
      let k = key_of (List.hd_exn (focus label)) in
      if Option.is_none !second then second := Some k;
      if not (Option.exists !second ~f:(String.equal k)) then false
      else if !admitted then true
      else (
        admitted := true;
        false)
  in
  let o = search ~width:0 ~refuse ~tag:"refused_after" ~cost () in
  let r = o.report in
  match !second with
  | Some k ->
      let staffed_of_k =
        List.count o.attempts ~f:(fun l ->
            is_staffed l && String.equal (key_of (List.hd_exn (focus l))) k)
      in
      p "refused_after: after an admitted staffed single, no refusal of the key is retried bare"
        (staffed_of_k >= 3 && r.Autotune.fiss_sketch_backdrop_fallbacks = 0)
  | None -> p "refused_after: the second key was swept staffed" false

(* --- slower: a key whose every single is slower than its preset stays on it --- *)

(* The second key's sketches all cost more than its preset: the running composite, which holds the
   key on its preset and was timed whole, is that key's incumbent, so none of its singles joins
   (Codex P1 on PR #1215), by the search's own log lines. *)
let () =
  let second = ref None in
  let cost entry =
    if Option.exists !second ~f:(String.equal (key_of entry)) then preset_ms +. hashed entry
    else hashed entry
  in
  let refuse label =
    if is_staffed label && Option.is_none !second then
      second := Some (key_of (List.hd_exn (focus label)));
    false
  in
  let o, lines =
    Stderr_capture.lines (fun () -> search ~log:true ~width:0 ~refuse ~tag:"slower" ~cost ())
  in
  let staffed_lines = List.filter lines ~f:(String.is_prefix ~prefix:"autotune: staffed: ") in
  let joins k l =
    String.is_prefix l ~prefix:("autotune: staffed: segment " ^ k ^ ": ")
    && String.is_substring l ~substring:"joins the running composite"
  in
  match (keys o.measured, !second) with
  | [ first; k ], Some k' when String.equal k k' ->
      p_exists "slower: the first key joined the composite" staffed_lines ~f:(joins first);
      p_none "slower: no single of the slower key joined it" staffed_lines ~f:(joins k);
      p "slower: the crown is the first key's single, the second on its preset"
        (List.length (focus o.report.Autotune.best_label) = 1
        && (not (is_staffed o.report.Autotune.best_label))
        && String.equal (key_of (List.hd_exn (focus o.report.Autotune.best_label))) first)
  | _ -> p "slower: the search swept the chain's two keys" false

(* --- shrunk: a composite window timed stays the outcome after a bare replacement --- *)

(* Three keys: the first two are swept into a timed composite, then the third key's staffed singles
   are all refused and its bare single, the faster whole routine, replaces the composite. The
   composite window completed, so the report still says Timed (Codex P2 on PR #1215). *)
let () =
  let seen = ref [] in
  let early k = List.mem (List.take (List.rev !seen) 2) k ~equal:String.equal in
  let refuse label =
    if is_single label then (
      let k = key_of (List.hd_exn (focus label)) in
      if not (List.mem !seen k ~equal:String.equal) then seen := k :: !seen;
      is_staffed label && not (early k))
    else false
  in
  let cost entry =
    if early (key_of entry) then 40. +. (hashed entry /. 10.) else hashed entry /. 10.
  in
  let o, lines =
    Stderr_capture.lines (fun () ->
        search ~log:true ~width:0 ~keyed:3 ~refuse
          ~chain:(Fissioned_chain.chain_with_output ~third:true ())
          ~tag:"shrunk" ~cost ())
  in
  p "shrunk: three keys were swept" (List.length (keys o.measured) = 3);
  p "shrunk: the third key's bare single replaced the composite"
    (List.exists lines ~f:(fun l ->
         String.is_prefix l ~prefix:"autotune: staffed: segment "
         && String.is_substring l ~substring:"replaces the running composite"));
  p "shrunk: the report keeps the composite window it timed"
    (Poly.equal o.report.Autotune.fiss_sketch_composite `Timed)

(* --- refused_staffed: a key whose staffed singles all decline competes as a whole routine --- *)

(* Every staffed attempt of the second key is refused, none of its bare ones, so each is retried
   bare and timed. The key cannot join the running composite -- the composite holding both is the
   very code refused -- but a bare single is the composite of its key alone, a whole routine like
   the running composite, so the faster of the two carries on (Codex P1 on PR #1215). The script
   makes the second key's bare singles the faster: they replace the composite, by the search's own
   log line. *)
let () =
  let second = ref None in
  let key_of_label l = key_of (List.hd_exn (focus l)) in
  let refuse label =
    is_staffed label
    &&
    let k = key_of_label label in
    if Option.is_none !second then second := Some k;
    Option.exists !second ~f:(String.equal k)
  in
  let cost entry =
    if Option.exists !second ~f:(String.equal (key_of entry)) then hashed entry /. 10.
    else hashed entry
  in
  let o, lines =
    Stderr_capture.lines (fun () ->
        search ~log:true ~width:0 ~refuse ~tag:"refused_staffed" ~cost ())
  in
  let r = o.report in
  match !second with
  | Some k ->
      let bare_of_k =
        List.filter o.measured ~f:(fun l ->
            is_single l && (not (is_staffed l)) && String.equal (key_of_label l) k)
      in
      p "refused_staffed: every refused staffed single was retried bare and timed"
        (r.Autotune.fiss_sketch_backdrop_fallbacks = List.length bare_of_k
        && List.length bare_of_k >= 3);
      let replaced =
        List.filter lines ~f:(fun l ->
            String.is_prefix l ~prefix:"autotune: staffed: segment "
            && String.is_substring l ~substring:"replaces the running composite")
      in
      (match fastest ~cost k o.measured with
      | Some best ->
          (* The entry is [<params> @<key>]; the log names the key and [W_sketch[<params>]]. *)
          let params = String.prefix best (String.rindex_exn best '@' - 1) in
          p
            "refused_staffed: the key's fastest bare retry, the faster routine, replaced the \
             composite"
            (match replaced with
            | [ line ] ->
                String.is_substring line ~substring:("segment " ^ k ^ ": ")
                && String.is_substring line ~substring:("W_sketch[" ^ params ^ "]")
            | _ -> false)
      | None ->
          p
            "refused_staffed: the key's fastest bare retry, the faster routine, replaced the \
             composite"
            false);
      p "refused_staffed: the crown is that bare single"
        (Option.equal String.equal (fastest ~cost k o.measured)
           (match focus r.Autotune.best_label with
           | [ e ] when not (is_staffed r.Autotune.best_label) -> Some e
           | _ -> None))
  | None -> p "refused_staffed: the second key's staffed singles were refused" false

(* --- partial: a cut search reports the staffed composite it timed --- *)

(* The search is cut at the first admitted staffed window, by a callback raising after admission:
   the partial report must already say a coarse composite was timed (Codex P2 on PR #1215), as it
   does for an explicit composite's window. *)
let () =
  let report = ref None and staffed_attempt = ref false and injected = ref false in
  ignore
    ( Autotune.Seam.with_value Autotune.on_candidate_attempt (fun label ->
          staffed_attempt := is_staffed label)
    @@ fun () ->
      Scripted_search.scripted
        ~settings:[ (staffed_key, "true"); finals_off ]
        ~measure:(fun w -> routine_ms ~cost w.label)
      @@ fun () ->
      Autotune.Seam.with_value Autotune.on_candidate_timed (fun _ ~timed_so_far:_ ->
          if !staffed_attempt then (
            injected := true;
            raise Stdlib.Exit))
      @@ fun () ->
      match
        Autotune.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:"" ~log:false ~progress:false
          ~report:(fun r -> report := Some r)
          (Context.cpu ()) (Fissioned_chain.chain ()) Ir.Indexing.Empty
      with
      | ctx, _ -> Context.release ctx
      | exception Stdlib.Exit when !injected -> () );
  match !report with
  | Some r when !injected ->
      p "partial: the cut search's report says the staffed composite was timed"
        (Poly.equal r.Autotune.fiss_sketch_composite `Timed
        && r.Autotune.fiss_sketch_staffed_timed = 1
        && match r.Autotune.outcome with Autotune.Search_died _ -> true | _ -> false)
  | _ -> p "partial: the cut search's report says the staffed composite was timed" false

(* --- ties: the playoff over a staffed composite --- *)

let () =
  let cost _ = 3. in
  let o = search ~tag:"ties" ~cost () in
  let r = o.report in
  match keys o.measured with
  | [ first; second ] ->
      let playoff = List.filter o.measured ~f:is_composite in
      p "ties: the playoff timed alternates" (r.Autotune.fiss_sketch_playoff_timed >= 1);
      p "ties: every composite window is the playoff's"
        (List.length playoff = r.Autotune.fiss_sketch_playoff_timed);
      let crowned_second =
        List.find
          (focus r.Autotune.best_label @ staff r.Autotune.best_label)
          ~f:(fun e -> String.equal (key_of e) second)
      in
      p_all "ties: only the first key plays; the last key's sweep ran inside the composite" playoff
        ~f:(fun l ->
          Option.exists crowned_second ~f:(fun e -> List.mem (focus l) e ~equal:String.equal)
          && List.exists (focus l) ~f:(fun e -> String.equal (key_of e) first))
  | _ -> p "ties: the search swept the chain's two keys" false

(* --- heavy: the keys are swept heaviest first --- *)

(* The chain with its second product four times the rows: by the model the heavier of the two, yet
   the second enumerated, so sweeping in enumeration order would fail the order claim. The weights
   are the search's own, read off its [autotune_log] lines. *)
let () =
  let o, lines =
    Stderr_capture.lines (fun () ->
        search ~log:true ~width:0
          ~chain:(Fissioned_chain.chain_with_output ~rows:64 ())
          ~tag:"heavy" ~cost ())
  in
  let weights =
    List.filter_map lines ~f:(fun line ->
        match String.chop_prefix line ~prefix:"autotune: staffed: segment " with
        | None -> None
        | Some rest -> (
            match String.split rest ~on:' ' with
            | [ tag; "weighs"; w ] -> Option.map (Float.of_string_opt w) ~f:(fun w -> (tag, w))
            | _ -> None))
  in
  Stdio.eprintf "heavy weights (not part of the golden): %s\n%!"
    (String.concat ~sep:" " (List.map weights ~f:(fun (tag, w) -> Printf.sprintf "%s=%.3e" tag w)));
  match weights with
  | [ (_, first); (_, second) ] ->
      p "heavy: the model weighs the second enumerated key the heavier" Float.(second > first);
      let by_weight =
        List.map ~f:fst (List.stable_sort weights ~compare:(fun (_, a) (_, b) -> Float.compare b a))
      in
      p "heavy: the keys are swept heaviest first"
        (List.equal String.equal (keys o.measured) by_weight)
  | _ -> p "heavy: the search logged a weight for each of the two keys" false

(* --- split: the split-reduce sites keep a running composite too --- *)

(* Two skinny matvec accumulations, each a split-reduce site (the multi-site workload of
   [autotune_split_reduce], over dyadic values whose sums are exact in any order, so the split's
   reassociation moves no bit): the sites are swept in their cost order, each single's prelude
   staffed with the other site's block count. The script is additive over the sites a split
   candidate splits, every other candidate far slower. *)
let () =
  let out = 8 and red1 = 128 and red2 = 256 in
  let matrix label red =
    TDSL.ndarray
      (Array.init (out * red)
         ~f:(Ll_test.cycle_flat ~dims:[| out; red |] ~modulus:5 ~offset:0. ~stride:0.25))
      ~label:[ label ] ~input_dims:[ red ] ~output_dims:[ out ] ()
  in
  let vector label red =
    TDSL.ndarray
      (Array.init red ~f:(Ll_test.cycle_flat ~dims:[| red |] ~modulus:3 ~offset:(-0.5) ~stride:0.5))
      ~label:[ label ] ~output_dims:[ red ] ()
  in
  let m1 = matrix "ss_m1" red1 and v1 = vector "ss_v1" red1 in
  let m2 = matrix "ss_m2" red2 and v2 = vector "ss_v2" red2 in
  let%op y1 = m1 * v1 in
  Train.set_materialized y1.Tensor.value;
  let%op y2 = m2 * v2 in
  Train.set_materialized y2.Tensor.value;
  let%op z = y1 + y2 in
  let comp = Train.forward z in
  let comp =
    { comp with Ir.Assignments.asgns = Ir.Assignments.Block_comment ("ss_pair", comp.asgns) }
  in
  let reference = Scripted_search.untuned z comp in
  let is_split label = String.is_prefix label ~prefix:"F_split[" in
  let split_parts label =
    let body =
      String.chop_prefix_exn label ~prefix:"F_split[" |> String.chop_suffix_exn ~suffix:"]"
    in
    let entries s = if String.is_empty s then [] else String.split s ~on:',' in
    match String.substr_index body ~pattern:" | staff " with
    | Some i -> (entries (String.prefix body i), entries (String.drop_prefix body (i + 9)))
    | None -> (entries body, [])
  in
  (* A site entry without its block count: which site it splits. *)
  let site entry =
    String.split entry ~on:' '
    |> List.filter ~f:(fun w ->
        not
          (String.length w > 1
          && Char.equal w.[0] 'b'
          && String.for_all (String.drop_prefix w 1) ~f:Char.is_digit))
    |> String.concat ~sep:" "
  in
  let cost entry = hashed entry in
  let split_ms label =
    if is_split label then
      let f, s = split_parts label in
      List.sum (module Float) (f @ s) ~f:cost
      +. (preset_ms *. Float.of_int (2 - List.length (f @ s)))
    else 1000.
  in
  let (o : observed) =
    Scripted_search.search
      ~settings:[ (staffed_key, "true"); finals_off ]
      ~output:z ~log:false ~progress:false ~tag:"split"
      ~measure:(fun w -> split_ms w.label)
      comp
  in
  let measured = o.measured and output = o.output in
  let singles =
    List.filter measured ~f:(fun l -> is_split l && List.length (fst (split_parts l)) = 1)
  in
  let sites =
    List.fold singles ~init:[] ~f:(fun seen l ->
        let s = site (List.hd_exn (fst (split_parts l))) in
        if List.mem seen s ~equal:String.equal then seen else s :: seen)
    |> List.rev
  in
  let of_site s =
    List.filter singles ~f:(fun l -> String.equal (site (List.hd_exn (fst (split_parts l)))) s)
  in
  let fastest s =
    List.map (of_site s) ~f:(fun l -> List.hd_exn (fst (split_parts l)))
    |> List.min_elt ~compare:(fun a b -> Float.compare (cost a) (cost b))
  in
  match sites with
  | [ first; second ] ->
      let r = o.report in
      p_all "split: the first site is swept bare" (of_site first) ~f:(fun l ->
          List.is_empty (snd (split_parts l)));
      p_all "split: every single of the second site is staffed with the first site's fastest"
        (of_site second) ~f:(fun l ->
          Option.exists (fastest first) ~f:(fun e ->
              List.equal String.equal (snd (split_parts l)) [ e ]));
      p "split: the composite was eligible and its window counted"
        (r.Autotune.split_reduce_composite_eligible && r.Autotune.split_reduce_composite_timed);
      p_none "split: no window re-timed the composite, the last accepted staffed single"
        (List.filter measured ~f:is_split) ~f:(fun l -> List.length (fst (split_parts l)) >= 2);
      p "split: the crown is a staffed single holding each site's fastest"
        (is_split r.Autotune.best_label
        && List.equal String.equal
             (List.sort ~compare:String.compare
                (fst (split_parts r.Autotune.best_label) @ snd (split_parts r.Autotune.best_label)))
             (List.sort ~compare:String.compare (List.filter_opt [ fastest first; fastest second ]))
        );
      p_all2 "split: the crowned routine computes the untuned output bit for bit" output reference
        ~f:Float.equal
  | _ ->
      Stdio.eprintf "split: sites %s\n%!" (String.concat ~sep:"; " sites);
      p "split: the search swept the workload's two sites" false
