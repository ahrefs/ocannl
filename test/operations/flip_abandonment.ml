(* gh-ocannl-1110: the flip chain abandons a hopeless flip early, at EQUAL search depth.

   On gh-719's approximate CUDA gpt2_mini cell, [Train.tune_placements]' two inline flips took 4747
   s of a 9145 s search and both ended 9-15x behind the arm they refined. The rule that cuts them
   compares a flip's best after its first [beam_width] timed candidates with the INCUMBENT'S best
   after the incumbent's own first as many ([report.best_steps]), against [tune_flip_profit_margin]
   squared. Against the incumbent's final best instead it would abandon every flip: that cell's arm
   A sat at 80.6 ms for 207 of its 209 timed candidates and finished at 6.862 ms.

   Three layers, all on cc with the timings supplied by a synthetic clock
   ([Autotune.on_candidate_measured] replaces every admitted reading, indexed by its admission
   ordinal), so no verdict depends on the machine. Each search runs one beam round: this
   computation's seeds dedup to two timed candidates on cc, too few to show what an abandonment cuts
   off, or to survive one window refused as contended on a loaded box: - the verdict itself, on
   gh-719's recorded steps and on its margin boundary; - [Autotune.tune ?abandon] executed: a
   hopeless search stops after exactly [beam_width] timed candidates and reports the abandonment;
   the negative controls (a flip within the ratio that then wins, and one far behind the incumbent's
   FINAL best but not at equal depth) run to completion; and the record survives a schedule-cache
   replay; - [Train.tune_placements] end to end, with [autotune_progress] on: a hopeless flip is
   abandoned and reported in the progress record, and the A/B winner ships; a flip that becomes
   profitable is searched in full and ships. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let approx a b = Float.(abs (a -. b) < 1e-4)
let n = 8
let k = 2
let ratio = Autotune.flip_abandon_ratio ~margin:1.25 ()
let rule incumbent_steps = { Autotune.incumbent_steps; trailing_ratio = ratio }
let abandons incumbent steps = Option.is_some (Autotune.abandon_verdict (rule incumbent) ~k ~steps)

(* gh-719's search trajectories (autotune-progress lines of the approximate cuda cell), as the best
   after each admitted timing where it changed. *)
let arm_a = [ (1, 80.6267); (208, 23.0120); (209, 6.8624) ]
let arm_b = [ (1, 86.6251); (264, 26.6355); (265, 9.1411) ]
let flip_max_logits = [ (1, 161.4671); (207, 103.2327); (208, 101.5021) ]
let flip_n242_k = [ (1, 136.6943); (4, 136.5825); (208, 81.9251); (209, 61.6913) ]

let verdicts () =
  p "the default trailing ratio is the default margin squared"
    (Float.equal ratio 1.5625 && Float.equal (Autotune.flip_abandon_ratio ()) 1.5625);
  p "gh-719's flips trail arm A at equal depth by more than the ratio, and are abandoned"
    (abandons arm_a flip_max_logits && abandons arm_a flip_n242_k);
  p "arm B, the other direction a flip can take, does not trail arm A by the ratio"
    (not (abandons arm_a arm_b));
  p "against arm A's FINAL best, arm A itself would have looked hopeless at that depth"
    Float.(Autotune.best_after arm_a k > ratio *. Autotune.best_after arm_a 209);
  let inc = [ (1, 4.0) ] in
  p "trailing by exactly the ratio keeps searching" (not (abandons inc [ (1, 6.25) ]));
  p "trailing by one ulp more is abandoned" (abandons inc [ (1, Float.one_ulp `Up 6.25) ]);
  p "the verdict reads the best after k timings, not the first one"
    (not (abandons inc [ (1, 7.0); (2, 6.0) ]));
  p "an incumbent that timed fewer than k compares at its final best"
    (Float.equal (Autotune.best_after [ (1, 4.0) ] k) 4.0);
  p "an incumbent with no timed record abandons nothing" (not (abandons [] [ (1, 1e9) ]))

(* One [Autotune.on_candidate_measured] script: the [i]-th admitted window (1-based) measures
   [script i]. Counts the admitted windows through [on_candidate_timed], the tuner's own count. *)
let with_clock script f =
  let measured = ref 0 and timed = ref 0 in
  let old_measured = !Autotune.on_candidate_measured and old_timed = !Autotune.on_candidate_timed in
  Exn.protect
    ~finally:(fun () ->
      Autotune.on_candidate_measured := old_measured;
      Autotune.on_candidate_timed := old_timed)
    ~f:(fun () ->
      (Autotune.on_candidate_measured :=
         fun ~label:_ ~digest:_ _ms ->
           Int.incr measured;
           script !measured);
      (Autotune.on_candidate_timed := fun _ ~timed_so_far:_ -> Int.incr timed);
      let r = f () in
      (r, !timed))

type run = Completed of Autotune.report | Abandoned of Autotune.report * Autotune.abandonment

let run_tune comp ~incumbent script =
  let report = ref None in
  let outcome, timed =
    with_clock script (fun () ->
        match
          Autotune.tune ~search:true ~beam_width:k ~rounds:1 ~repeats:1 ~timing:Autotune.Isolated
            ~cache_dir:"" ~abandon:(rule incumbent)
            ~report:(fun r -> report := Some r)
            (Context.auto ()) comp Ir.Indexing.Empty
        with
        | _ -> `Completed
        | exception Autotune.Search_abandoned ab -> `Abandoned ab)
  in
  let r = Option.value_exn !report in
  ((match outcome with `Completed -> Completed r | `Abandoned ab -> Abandoned (r, ab)), timed)

let executed_tune comp =
  (* Hopeless: every window 100 ms against an incumbent at 4 ms. *)
  let hopeless, hopeless_timed = run_tune comp ~incumbent:[ (1, 4.0) ] (fun _ -> 100.0) in
  (match hopeless with
  | Abandoned (r, ab) ->
      p "a hopeless search is abandoned after exactly beam_width timed candidates"
        (ab.Autotune.ab_timed = k && r.Autotune.candidates_timed = k && hopeless_timed = k);
      p "its report is the abandonment, carrying the verdict's numbers"
        (match r.Autotune.outcome with
        | Autotune.Abandoned ab' ->
            Poly.equal ab ab'
            && Float.equal ab.Autotune.ab_best_ms 100.0
            && Float.equal ab.Autotune.ab_incumbent_ms 4.0
            && Float.equal ab.Autotune.ab_ratio ratio
        | _ -> false);
      p "an abandonment is not a terminal failure" (Option.is_none (Autotune.terminal_failure r))
  | Completed _ -> fail "a hopeless search was not abandoned");
  (* Negative control: within the ratio at depth k, then faster than the incumbent. *)
  let winning, winning_timed =
    run_tune comp ~incumbent:[ (1, 4.0) ] (fun i -> if i <= k then 6.0 else 1.0)
  in
  (match winning with
  | Completed r ->
      p "a flip within the ratio at depth k is searched in full and finds its better time"
        (Poly.equal r.Autotune.outcome Autotune.Searched
        && Float.equal r.Autotune.best_ms 1.0
        && winning_timed > k);
      p "the search the hopeless one was cut from had more to time"
        (r.Autotune.candidates_timed > hopeless_timed);
      p "the completed search records its steps"
        (List.equal
           (fun (a, x) (b, y) -> a = b && Float.equal x y)
           r.Autotune.best_steps
           [ (1, 6.0); (k + 1, 1.0) ])
  | Abandoned _ -> fail "a flip within the ratio was abandoned");
  (* Negative control, gh-719's shape: 16.7x behind the incumbent's final best, yet within the ratio
     of where the incumbent stood at the same depth. *)
  let late, _ =
    run_tune comp ~incumbent:[ (1, 80.0); (9, 6.0) ] (fun i -> if i <= k then 100.0 else 5.0)
  in
  p "a flip far behind the incumbent's final best but not at equal depth is not abandoned"
    (match late with Completed r -> Float.equal r.Autotune.best_ms 5.0 | Abandoned _ -> false)

let clean_cache dir =
  if Stdlib.Sys.file_exists dir && Stdlib.Sys.is_directory dir then
    Array.iter (Stdlib.Sys.readdir dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat dir f))

(* The warm path: an incumbent whose search replays from the schedule cache still carries the timed
   record a flip is abandoned against. Nothing is cached after a refused window, so on a box whose
   load refused one the claim is skipped rather than decided. *)
let cache_replay comp =
  let tune_once ?(beam_width = k) ?(repeats = 1) () =
    let report = ref None in
    let _ =
      with_clock
        (fun i -> if i <= 1 then 3.0 else 2.0)
        (fun () ->
          Autotune.tune ~search:true ~beam_width ~rounds:0 ~repeats ~timing:Autotune.Isolated
            ~cache_dir:"autotune_cache_flip_abandonment"
            ~report:(fun r -> report := Some r)
            (Context.auto ()) comp Ir.Indexing.Empty)
    in
    Option.value_exn !report
  in
  (* The dune rule sets [autotune_keep_fraction] (Search_shaping) and [autotune_progress]
     (Execution_neutral) on the command line. *)
  let shape = Utils.config_class_fingerprint Utils.Search_shaping in
  p "the trajectory's identity carries a set Search_shaping key and no other class's"
    (String.is_substring shape ~substring:"autotune_keep_fraction=1;"
    && not (String.is_substring shape ~substring:"autotune_progress"));
  clean_cache "autotune_cache_flip_abandonment";
  let first = tune_once () in
  let second = tune_once () in
  (* The cache key carries neither the beam width nor the repeats, so these replay the same entry;
     its trajectory was timed in another candidate order, or under another sampling, so it must not
     stand as an equal-depth record. *)
  let other_beam = tune_once ~beam_width:(k + 1) () in
  let other_repeats = tune_once ~repeats:2 () in
  clean_cache "autotune_cache_flip_abandonment";
  gated ~aggregation:`Environment
    ~when_:(first.Autotune.timings_contended = 0)
    ~on:"a contended timing window (nothing was cached)"
    "a replayed incumbent carries the storing search's timed record"
    (Poly.equal second.Autotune.outcome Autotune.Cache_replay
    && (not (List.is_empty first.Autotune.best_steps))
    && List.equal
         (fun (a, x) (b, y) -> a = b && Float.equal x y)
         second.Autotune.best_steps first.Autotune.best_steps);
  let replays_no_record (r : Autotune.report) =
    Poly.equal r.Autotune.outcome Autotune.Cache_replay
    && abandons second.Autotune.best_steps [ (1, 1e9) ]
    && not (abandons r.Autotune.best_steps [ (1, 1e9) ])
  in
  gated ~aggregation:`Environment
    ~when_:(first.Autotune.timings_contended = 0)
    ~on:"a contended timing window (nothing was cached)"
    "a replay under another beam width abandons nothing, where the same shape's replay would"
    (replays_no_record other_beam);
  gated ~aggregation:`Environment
    ~when_:(first.Autotune.timings_contended = 0)
    ~on:"a contended timing window (nothing was cached)"
    "a replay under other repeats abandons nothing, where the same shape's replay would"
    (replays_no_record other_repeats)

(* An [autotune-progress:] line's [key=value] fields; values are unquoted words here. *)
let progress_fields line =
  Option.map (String.chop_prefix line ~prefix:"autotune-progress: ") ~f:(fun rest ->
      String.split rest ~on:' '
      |> List.filter_map ~f:(fun w ->
          Option.map (String.lsplit2 w ~on:'=') ~f:(fun (k, v) ->
              (k, String.strip ~drop:(Char.equal '"') v))))

let field fields key = List.Assoc.find fields key ~equal:String.equal

let with_stderr_captured f =
  let file = Stdlib.Filename.temp_file "flip_abandonment" ".stderr" in
  Stdio.Out_channel.flush Stdio.stderr;
  let saved = Unix.dup Unix.stderr in
  let fd = Unix.openfile file [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
  Unix.dup2 fd Unix.stderr;
  Unix.close fd;
  let restore () =
    Stdio.Out_channel.flush Stdio.stderr;
    Unix.dup2 saved Unix.stderr;
    Unix.close saved
  in
  let result = Exn.protect ~f ~finally:restore in
  let text = Stdio.In_channel.read_all file in
  Stdlib.Sys.remove file;
  (* Echoed, so nothing the run wrote is hidden. *)
  Stdio.eprintf "%s%!" text;
  (result, List.filter_map (String.split_lines text) ~f:progress_fields)

(* [Train.tune_placements] with one flip: arm A's windows measure 4 ms, arm B's 50 ms, and the
   flip's per [flip_script] (1-based within the flip's search). The searches are told apart by how
   many reports have been delivered when a window is admitted: A's search delivers the first, B's
   the second. *)
exception Arm_a_dies

let executed_chain ?fail_arm_a_at comp t2 expected ~flip_script =
  let arm_reports = ref [] and flip_reports = ref [] and shipped = ref [] in
  let delivered () = List.length !arm_reports + List.length !flip_reports in
  (* [fail_arm_a_at]: arm A's search dies as its attempt of that ordinal starts, through the
     containment tests' fault-injection seam — after it has timed something, so its report carries a
     partial record. *)
  let arm_a_attempts = ref 0 in
  let old_attempt = !Autotune.on_candidate_attempt in
  (Autotune.on_candidate_attempt :=
     fun _ ->
       if delivered () = 0 then (
         Int.incr arm_a_attempts;
         if Option.exists fail_arm_a_at ~f:(fun n -> !arm_a_attempts = n) then raise Arm_a_dies));
  let flip_window = ref 0 in
  let script _ =
    match delivered () with
    | 0 -> 4.0
    | 1 -> 50.0
    | _ ->
        Int.incr flip_window;
        flip_script !flip_window
  in
  let (got, _), lines =
    with_stderr_captured (fun () ->
        Exn.protect
          ~finally:(fun () -> Autotune.on_candidate_attempt := old_attempt)
          ~f:(fun () ->
            with_clock script (fun () ->
                let ctx, routine =
                  Train.tune_placements ~beam_width:k ~rounds:1 ~repeats:1 ~cache_dir:""
                    ~report:(fun r -> arm_reports := r :: !arm_reports)
                    ~flip_report:(fun r -> flip_reports := r :: !flip_reports)
                    ~on_ship:(fun what -> shipped := what :: !shipped)
                    ~inline_flips:1 (Context.auto ()) t2 comp Ir.Indexing.Empty
                in
                let ctx = Context.run ctx routine in
                Context.get_values ctx t2.Tensor.value)))
  in
  p_all2 "the shipped routine computes the plain compile's values" got expected ~f:approx;
  let events name =
    List.filter lines ~f:(fun f -> Option.equal String.equal (field f "event") (Some name))
  in
  (List.rev !arm_reports, List.rev !flip_reports, !shipped, events)

let () =
  verdicts ();
  let mav =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:7 ~offset:0. ~stride:0.5)
  in
  let mbv =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:5 ~offset:(-2.) ~stride:1.)
  in
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  (* A pointwise product feeding a relu: its product is a policy-virtual intermediate, so the chain
     has a [Materialize] flip to try (inline_flip_tune's computation, for the same reason). *)
  let%op mc = ma *. mb in
  let%op t2 = relu mc in
  ignore mc;
  let comp = Train.forward t2 in
  let ctx_ref, routine_ref = Context.compile (Context.auto ()) comp Ir.Indexing.Empty in
  let ctx_ref = Context.run ctx_ref routine_ref in
  let expected = Context.get_values ctx_ref t2.Tensor.value in
  executed_tune comp;
  cache_replay comp;
  p "autotune_progress is on for this run" (Autotune.progress_enabled ());
  (* A hopeless flip: abandoned, reported, and the A/B winner ships. *)
  let _, flips, shipped, events = executed_chain comp t2 expected ~flip_script:(fun _ -> 100.0) in
  p "the hopeless flip's report is an abandonment after beam_width timed candidates"
    (match flips with
    | [ r ] -> (
        match r.Autotune.outcome with
        | Autotune.Abandoned ab -> ab.Autotune.ab_timed = k && r.Autotune.candidates_timed = k
        | _ -> false)
    | _ -> false);
  p "the A/B winner ships" (List.equal String.equal shipped [ "A" ]);
  p "the progress record closes the flip's search as abandoned"
    (List.exists (events "search_done") ~f:(fun f ->
         Option.equal String.equal (field f "outcome") (Some "abandoned")));
  p "the flip's arm_done says abandoned, after beam_width timings, against the incumbent's 4 ms"
    (match List.filter (events "arm_done") ~f:(fun f -> Option.is_some (field f "flip")) with
    | [ f ] ->
        Option.equal String.equal (field f "result") (Some "abandoned")
        && Option.equal String.equal (field f "after") (Some (Int.to_string k))
        && Option.equal String.equal (field f "incumbent_ms") (Some "4.0000")
    | _ -> false);
  p "flips_done counts the abandonment among the measured flips"
    (match events "flips_done" with
    | [ f ] ->
        Option.equal String.equal (field f "measured") (Some "1")
        && Option.equal String.equal (field f "abandoned") (Some "1")
        && Option.equal String.equal (field f "improved") (Some "false")
    | _ -> false);
  (* The negative control: within the ratio at depth k, then faster than arm A — searched in full,
     and it ships. *)
  let _, flips, shipped, events =
    executed_chain comp t2 expected ~flip_script:(fun i -> if i <= k then 5.0 else 1.0)
  in
  p "a flip that becomes profitable is searched in full"
    (match flips with
    | [ r ] -> Poly.equal r.Autotune.outcome Autotune.Searched && r.Autotune.candidates_timed > k
    | _ -> false);
  p "and the refined placement ships" (List.equal String.equal shipped [ "flip" ]);
  p "flips_done reports no abandonment and an improvement"
    (match events "flips_done" with
    | [ f ] ->
        Option.equal String.equal (field f "abandoned") (Some "0")
        && Option.equal String.equal (field f "improved") (Some "true")
    | _ -> false);
  (* A failed arm A abandons nothing: its partial record is no incumbent, and a flip from its
     context may be what ships. The same hopeless flip is searched in full. *)
  let arms, flips, shipped, _ =
    executed_chain ~fail_arm_a_at:3 comp t2 expected ~flip_script:(fun _ -> 100.0)
  in
  p "arm A died after timing something"
    (match arms with
    | a :: _ ->
        Option.is_some (Autotune.terminal_failure a) && not (List.is_empty a.Autotune.best_steps)
    | [] -> false);
  p "against a failed arm A the hopeless flip is searched in full"
    (match flips with
    | [ r ] -> Poly.equal r.Autotune.outcome Autotune.Searched && r.Autotune.candidates_timed > k
    | _ -> false);
  p "and arm B, the only better shippable result, ships" (List.equal String.equal shipped [ "B" ])
