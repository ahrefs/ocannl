(* gh-ocannl-1166: the composite playoff. A per-segment single is timed as the whole routine with
   every other segment on its untuned preset, so near-tie singles are ranked through that backdrop's
   noise; the playoff re-times them inside the recombined composite, one keyed segment at a time,
   and a faster alternate becomes the incumbent the later segments build on.

   The ranking is the machine's unless the test decides it, so every admitted window is scripted
   through [Autotune.on_candidate_measured] (gh-ocannl-1027) and the windows are opted out of host
   contention ([with_uncontended_test_windows], gh-ocannl-1156). Everything that is not a
   per-segment sketch is scripted far slower than any of them, so the crown is a composite's by
   construction. Every search is over one two-matmul chain.

   Each search compiles every single it times, and those compiles are the test's cost: unpruned, the
   chain offers 21 singles per segment, and the test took about 430 s on mac-studio. So every search
   runs on [Fissioned_chain]'s small menu (the roofline pre-filter keeping a fifth of each segment's
   sketches, [Train.tune_placements]' arms included): eight and seven singles per segment at the
   vector width the small menu pins, which is all the playoff's claims need. Which ones is the
   model's choice and no claim depends on it; that each segment still offers three or more, without
   which the width claims could not fail, is claimed ([menu]).

   In [ties] every single measures the same, so each segment has near-tie contenders, and each later
   composite is scripted faster than the one before: every admitted playoff window swaps, so each
   window challenges the previous one, differs from it in one segment, and the crown is the last
   window.

   In [slower] the singles tie the same way but each later composite is slower: nothing swaps, every
   window differs from the recombined composite in one segment, and that composite keeps the crown.

   In [near] the singles are 0.3% apart in attempt order, inside the playoff margin, so each
   segment's runner-up single plays and the one behind it (0.6%) does not; in [spread] they are 1%
   apart, wider than the margin, so the playoff times nothing. [wide] is [spread] with
   [autotune_composite_playoff_margin=0.015]: over the same singles, the runner-up 1% behind its
   segment's best now plays and the one 2% behind does not, so the margin, not the width, decides
   which singles enter.

   [off] is [ties] with [autotune_composite_playoff_width=0] (gh-ocannl-1259): it must time exactly
   the candidates [ties] timed outside its playoff, in the same order, and crown the recombined
   composite; [one] is [ties] at width 1, which plays one alternate per segment and so fewer windows
   than [ties].

   Every search runs with the timing trace observed ([Autotune.with_timing_call_observer], whose
   records carry the decision of gh-ocannl-1199), which must attribute each playoff window to the
   [playoff] phase; [ties] is searched once more with the trace unobserved, and must crown and time
   exactly the same. The report's settle-by-admission counts (gh-ocannl-1263) must be the traced
   calls' own, the untuned control's left out, whether or not the trace observes them.

   Every search passes [tune]'s [?log] explicitly, so none depends on the ambient [autotune_log].
   [ties] is searched once more with [~log:true], which adds the untuned-default in-process control
   after the crown: exactly one more timing decision, the last, attributed to no phase. That search
   also passes [~progress:true] with stderr captured, so both per-call stream overrides are claimed
   on what they write -- the [autotune:] diagnostics and the [autotune-progress:] record, including
   the control's own [untuned_control] stage line -- and on ending with the call. A last leg runs
   [Train.tune_placements] with both overrides on one matmul: its own arm lines and both nested arm
   searches must see them.

   Pinned to cc: the seeding and the scripted ranking are backend-independent, and cc is always
   available. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let is_single = Fissioned_chain.is_single
let is_composite = Fissioned_chain.is_composite

type observed = {
  report : Autotune.report;
  measured : string list;  (** Every admitted window's label, in attempt order. *)
  composite_ms : (string * float) list;  (** Every admitted composite window, in attempt order. *)
  calls : Autotune.timing_call list;
      (** Every timing call's record, in order; empty when the trace was unobserved. *)
  phases : string option list;  (** The search phase of each of [calls]. *)
}

let playoff_width_key = "autotune_composite_playoff_width"
let playoff_margin_key = "autotune_composite_playoff_margin"

(* The small menu (see the header), for every search here: stated before the first search. *)
let () = Fissioned_chain.small_menu ()

(* A list of records as a report counts them: per (settle, admission), in constructor order. *)
let tally (calls : Autotune.timing_call list) =
  List.map calls ~f:(fun c -> (c.decision.settle, c.decision.admission))
  |> List.sort_and_group ~compare:Poly.compare
  |> List.map ~f:(fun group ->
      let settle, admission = List.hd_exn group in
      (settle, admission, List.length group))

let counted (r : Autotune.report) =
  List.sum (module Int) r.timing_decisions ~f:(fun (_, _, calls) -> calls)

(* One search under a scripted ranking: [single n] is the n-th single's time (0-based, attempt
   order), [composite n] the n-th composite's. [?width] states [autotune_composite_playoff_width]
   and [?margin] [autotune_composite_playoff_margin] for this search only (at the config-file level;
   the stanza declares both keys' environment variables, so an ambient override reruns the test,
   whose width and margin claims then fail). *)
let search ?(trace = true) ?(log = false) ?progress ?width ?margin ~tag ~single ~composite () =
  let singles = ref 0 and composites = ref 0 and calls = ref [] in
  let observed body =
    if trace then Autotune.with_timing_call_observer (fun c -> calls := c :: !calls) body
    else body ()
  in
  (* The beam finals (gh-ocannl-1635) off: these searches script near-tie composites, which the
     finals would re-time after the playoff, and every claim here counts composite windows. The
     finals are [autotune_beam_finals]' subject. *)
  let settings =
    [ ("autotune_finals_margin", "0") ]
    @ Option.value_map width ~default:[] ~f:(fun w -> [ (playoff_width_key, Int.to_string w) ])
    @ Option.value_map margin ~default:[] ~f:(fun m -> [ (playoff_margin_key, Float.to_string m) ])
  in
  let (o : Scripted_search.observed) =
    observed @@ fun () ->
    Scripted_search.search ~settings ~log ?progress ~tag
      ~measure:(fun w ->
        if is_single w.label then (
          let ms = single !singles in
          Int.incr singles;
          ms)
        else if is_composite w.label then (
          let ms = composite !composites in
          Int.incr composites;
          ms)
        else 1000.)
      (Fissioned_chain.chain ())
  in
  let composite_ms =
    List.filter_map o.windows ~f:(fun (w, ms) ->
        Option.some_if (is_composite w.label) (w.label, ms))
  in
  let r = o.report in
  Stdio.eprintf
    "%s (not part of the golden): singles %d, composite %s, playoff timed %d swaps %d, crown %s \
     %.2f ms\n\
     %!"
    tag !singles
    (match r.Autotune.fiss_sketch_composite with
    | `Timed -> "timed"
    | `Refused -> "refused"
    | `Proposed -> "proposed"
    | `Singles_refused -> "singles refused"
    | `Ineligible -> "ineligible")
    r.Autotune.fiss_sketch_playoff_timed r.Autotune.fiss_sketch_playoff_swaps r.Autotune.best_label
    r.Autotune.best_ms;
  List.iter composite_ms ~f:(fun (label, ms) ->
      Stdio.eprintf "  %s window (not part of the golden): %s %.2f ms\n%!" tag label ms);
  let calls = List.rev !calls in
  {
    report = r;
    measured = o.measured;
    composite_ms;
    calls;
    phases = List.map calls ~f:(fun c -> c.Autotune.decision.phase);
  }

(* The [autotune-progress:] lines among [lines] that carry every one of [fields] ([key=value], the
   value unquoted). *)
let progress_lines lines fields =
  Stderr_capture.Progress.count lines
    (List.map fields ~f:(fun field -> String.lsplit2_exn field ~on:'='))

let streams_off () = (not (Autotune.log_enabled ())) && not (Autotune.progress_enabled ())

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

(* Bound at the top level: the trace claims at the end compare it with an unobserved rerun. *)
let ties = search ~tag:"ties" ~single:(fun _ -> 100.) ~composite:faster ()

let () =
  (* The width claims below can fail only where a segment offers more singles than the width. *)
  Fissioned_chain.menu_claim ties.measured;
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
     Exactly one: a sketch's label renders every parameter (gh-ocannl-1230), so two register-tile
     geometries of one tile print apart and a one-segment change cannot read as none. *)
  let labels = List.map ties.composite_ms ~f:fst in
  let challenges =
    List.zip_exn (List.drop_last_exn labels) (List.tl_exn labels)
    |> List.map ~f:(fun (incumbent, window) -> changed incumbent window)
  in
  p_all "ties: each window differs from the incumbent it challenged in exactly one segment"
    challenges ~f:(fun c -> List.length c = 1);
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
      ()
  in
  let r = slower.report in
  p "slower: the playoff timed alternates" (r.Autotune.fiss_sketch_playoff_timed >= 1);
  p "slower: no slower alternate replaced the incumbent" (r.Autotune.fiss_sketch_playoff_swaps = 0);
  (match slower.composite_ms with
  | (first, _) :: windows ->
      p "slower: the crown is the recombined composite" (String.equal r.Autotune.best_label first);
      let challenges = List.map windows ~f:(fun (window, _) -> changed first window) in
      p_all "slower: each window differs from the recombined composite in exactly one segment"
        challenges ~f:(fun c -> List.length c = 1);
      width_claim "slower: no segment is challenged by more than two windows" ~width:2 challenges
  | [] ->
      p "slower: the crown is the recombined composite" false;
      p "slower: each window differs from the recombined composite in exactly one segment" false;
      p "slower: no segment is challenged by more than two windows" false);
  let near =
    search ~tag:"near" ~single:(fun n -> 100. *. Float.int_pow 1.003 n) ~composite:faster ()
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
    search ~tag:"spread" ~single:(fun n -> 100. *. Float.int_pow 1.01 n) ~composite:faster ()
  in
  let r = spread.report in
  p "spread: the coarse composite was timed" (Poly.equal r.Autotune.fiss_sketch_composite `Timed);
  p "spread: singles 1% apart leave the playoff nothing to time"
    (r.Autotune.fiss_sketch_playoff_timed = 0 && List.length spread.composite_ms = 1);
  let wide =
    search ~margin:0.015 ~tag:"wide"
      ~single:(fun n -> 100. *. Float.int_pow 1.01 n)
      ~composite:faster ()
  in
  let r = wide.report in
  p "wide: the same singles as spread, so only the margin differs"
    (List.equal String.equal
       (List.filter wide.measured ~f:is_single)
       (List.filter spread.measured ~f:is_single));
  (* The recombined composite holds each segment's best single, so its runner-up is the one
     contender a segment has. *)
  p "wide: a 1.5% margin lets each segment's runner-up 1% behind reach the playoff"
    (r.Autotune.fiss_sketch_playoff_timed
    = List.length (Fissioned_chain.singles_per_segment wide.measured));
  let labels = List.map wide.composite_ms ~f:fst in
  let challenges =
    match labels with
    | [] -> []
    | _ ->
        List.zip_exn (List.drop_last_exn labels) (List.tl_exn labels)
        |> List.map ~f:(fun (incumbent, window) -> changed incumbent window)
  in
  width_claim "wide: only the single within the wider margin plays, one per segment" ~width:1
    challenges

(* gh-ocannl-1259: the playoff's width is a setting, and 0 is off. [ties] is the reference: its
   composite windows are the recombined composite and then exactly its playoff windows (claimed
   above), so dropping every composite after the first leaves what a search without the playoff
   times. *)
let () =
  let without_playoff (o : observed) =
    match o.composite_ms with
    | [] -> o.measured
    | _ :: playoff_windows ->
        let windows = List.map playoff_windows ~f:fst in
        List.filter o.measured ~f:(fun label -> not (List.mem windows label ~equal:String.equal))
  in
  let off = search ~width:0 ~tag:"off" ~single:(fun _ -> 100.) ~composite:faster () in
  let r = off.report in
  p "off: the reference search played windows a width-0 search must drop"
    (ties.report.Autotune.fiss_sketch_playoff_timed >= 1);
  p "off: width 0 times no playoff window"
    (r.Autotune.fiss_sketch_playoff_timed = 0 && r.Autotune.fiss_sketch_playoff_swaps = 0);
  p "off: width 0 times exactly the reference's candidates outside its playoff, in order"
    (List.equal String.equal off.measured (without_playoff ties));
  p_none "off: no timing decision is in the playoff phase" off.phases
    ~f:(Option.equal String.equal (Some "playoff"));
  (match ties.composite_ms with
  | (recombined, _) :: _ ->
      p "off: the crown is the recombined composite" (String.equal r.Autotune.best_label recombined)
  | [] -> p "off: the crown is the recombined composite" false);
  let one = search ~width:1 ~tag:"one" ~single:(fun _ -> 100.) ~composite:faster () in
  let r = one.report in
  p "one: width 1 still plays" (r.Autotune.fiss_sketch_playoff_timed >= 1);
  p "one: width 1 plays fewer windows than width 2"
    (r.Autotune.fiss_sketch_playoff_timed < ties.report.Autotune.fiss_sketch_playoff_timed);
  let labels = List.map one.composite_ms ~f:fst in
  let challenges =
    match labels with
    | [] -> []
    | _ ->
        List.zip_exn (List.drop_last_exn labels) (List.tl_exn labels)
        |> List.map ~f:(fun (incumbent, window) -> changed incumbent window)
  in
  width_claim "one: no segment is challenged by more than one window" ~width:1 challenges

(* gh-ocannl-1199: the decision trace carries the playoff phase, and observing it moves nothing. *)
let () =
  let traced = ties in
  let quiet = search ~trace:false ~tag:"quiet" ~single:(fun _ -> 100.) ~composite:faster () in
  let in_phase name = List.count traced.phases ~f:(Option.equal String.equal (Some name)) in
  Stdio.eprintf "trace (not part of the golden): %s\n%!"
    (String.concat ~sep:" " (List.map traced.phases ~f:(Option.value ~default:"-")));
  p "trace: one decision per playoff window says phase playoff"
    (traced.report.Autotune.fiss_sketch_playoff_timed >= 1
    && in_phase "playoff" = traced.report.Autotune.fiss_sketch_playoff_timed);
  p_empty "trace: an unobserved seam reports nothing" ~over:traced.phases quiet.phases;
  p "trace: the same windows, crown and time with the seam observed and unobserved"
    (List.equal Poly.equal traced.composite_ms quiet.composite_ms
    && String.equal traced.report.Autotune.best_label quiet.report.Autotune.best_label
    && Float.equal traced.report.Autotune.best_ms quiet.report.Autotune.best_ms);
  (* gh-ocannl-1263: the search counts its own calls through [time_routine]'s [on_call], apart from
     the domain observer the trace uses -- so the two agree, and the count is there unobserved. *)
  p_all "trace: every traced call is one of the search's own, tagged" traced.calls ~f:(fun c ->
      c.Autotune.tagged);
  p "trace: the report counts exactly the traced calls by settle and admission"
    (Poly.equal traced.report.Autotune.timing_decisions (tally traced.calls));
  p "trace: an unobserved search counts as many calls"
    (counted quiet.report = List.length traced.calls)

(* The untuned control under a scoped [~log:true]: the logged search times everything the unlogged
   [ties] did, in the same phases, then the control once more with no phase. Its window is real (the
   measurement seam scripts candidates only), so it moves no crown. *)
let () =
  (* Without this, a line below could come from the ambient config rather than the override. *)
  p "streams: the ambient config leaves both diagnostic streams off" (streams_off ());
  let logged, lines =
    Stderr_capture.lines (fun () ->
        search ~log:true ~progress:true ~tag:"logged" ~single:(fun _ -> 100.) ~composite:faster ())
  in
  Stdio.eprintf "logged trace (not part of the golden): %s\n%!"
    (String.concat ~sep:" " (List.map logged.phases ~f:(Option.value ~default:"-")));
  p "log: the logged search adds exactly one decision, last and phase-less (the untuned control)"
    (List.equal (Option.equal String.equal) logged.phases (ties.phases @ [ None ]));
  p "log: the report's counts leave out the untuned control, the one untagged call"
    (List.count logged.calls ~f:(fun c -> not c.Autotune.tagged) = 1
    && (not (List.last_exn logged.calls).Autotune.tagged)
    && Poly.equal logged.report.Autotune.timing_decisions
         (tally (List.filter logged.calls ~f:(fun c -> c.Autotune.tagged))));
  p "log: the same windows, crown and time with the control on and off"
    (List.equal Poly.equal logged.composite_ms ties.composite_ms
    && String.equal logged.report.Autotune.best_label ties.report.Autotune.best_label
    && Float.equal logged.report.Autotune.best_ms ties.report.Autotune.best_ms);
  p "log: the scoped search wrote autotune: diagnostics"
    (List.exists lines ~f:(String.is_prefix ~prefix:"autotune: "));
  p "progress: the scoped search wrote one search_start and one search_done"
    (progress_lines lines [ "event=search_start" ] = 1
    && progress_lines lines [ "event=search_done" ] = 1);
  p "progress: the untuned control, under both overrides, wrote its stage line once"
    (progress_lines lines [ "event=stage"; "stage=untuned_control" ] = 1);
  p "streams: both overrides end with the search" (streams_off ())

(* [Train.tune_placements]' [?log] and [?progress] cover the whole call: its own arm lines (no
   longer a config read of their own) and, by inheritance, both arms' searches. Every window is
   scripted alike: nothing here is about the ranking. *)
let () =
  let pa = Fissioned_chain.matrix ~label:"pa" ~modulus:11 ~offset:0. ~stride:0.125 in
  let pb = Fissioned_chain.matrix ~label:"pb" ~modulus:7 ~offset:(-3.) ~stride:1. in
  let%op pc = pa * pb in
  let comp = Train.forward pc in
  let ((), _), lines =
    Stderr_capture.lines (fun () ->
        Scripted_search.scripted
          ~measure:(fun _ -> 1.)
          (fun () ->
            let ctx, _routine =
              Train.tune_placements ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:"" ~log:true
                ~progress:true (Context.cpu ()) pc comp Ir.Indexing.Empty
            in
            Context.release ctx))
  in
  let arm_line arm =
    List.exists lines ~f:(String.is_prefix ~prefix:("tune_placements: arm " ^ arm ^ " "))
  in
  p "tune_placements: ~log reached its own arm lines, for both arms" (arm_line "A" && arm_line "B");
  p "tune_placements: ~progress framed both arms" (progress_lines lines [ "event=arm_start" ] = 2);
  p "tune_placements: both arm searches inherited both overrides (an untuned-control stage each)"
    (progress_lines lines [ "event=search_done" ] = 2
    && progress_lines lines [ "event=stage"; "stage=untuned_control" ] = 2);
  p "streams: both overrides end with the tune_placements call" (streams_off ())
