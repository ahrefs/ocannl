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

   Every search runs with the timing trace's decision seam observed ([Autotune.on_batch_decision],
   gh-ocannl-1199), which must attribute each playoff window to the [playoff] phase; [ties] is
   searched once more with the seam unobserved, and must crown and time exactly the same.

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
  phases : string option list;
      (** The search phase of every timing decision, in order; empty when the seam was unobserved.
      *)
}

(* One search under a scripted ranking: [single n] is the n-th single's time (0-based, attempt
   order), [composite n] the n-th composite's. *)
let search ?(trace = true) ?(log = false) ?progress ~tag ~single ~composite () =
  let report = ref None and singles = ref 0 and composites = ref [] and phases = ref [] in
  let old_measured = !Autotune.on_candidate_measured
  and old_decision = !Autotune.on_batch_decision in
  Exn.protect
    ~finally:(fun () ->
      Autotune.on_candidate_measured := old_measured;
      Autotune.on_batch_decision := old_decision)
    ~f:(fun () ->
      (if trace then Autotune.on_batch_decision := fun d -> phases := d.Autotune.phase :: !phases);
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
            Autotune.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:"" ~log ?progress
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
      { report; composite_ms = List.rev !composites; phases = List.rev !phases }
  | None -> failwith (tag ^ ": the search delivered no report")

(* Runs [f] with stderr routed into a file and returns its result with the lines it wrote. The
   captured text is echoed back to stderr once stderr is restored, whether [f] returned or raised
   (an exception is re-raised with its original backtrace after the echo), so nothing the run wrote
   is hidden; the file is removed on every path. *)
let with_stderr_captured f =
  let file = Stdlib.Filename.temp_file "autotune_composite_playoff" ".stderr" in
  Exn.protect
    ~finally:(fun () -> try Stdlib.Sys.remove file with Sys_error _ -> ())
    ~f:(fun () ->
      Stdio.Out_channel.flush Stdio.stderr;
      let saved = Unix.dup Unix.stderr in
      let outcome =
        Exn.protect
          ~finally:(fun () -> Unix.close saved)
          ~f:(fun () ->
            let fd = Unix.openfile file [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
            Exn.protect ~finally:(fun () -> Unix.close fd) ~f:(fun () -> Unix.dup2 fd Unix.stderr);
            let outcome =
              match f () with
              | result -> Ok result
              | exception exn -> Error (exn, Stdlib.Printexc.get_raw_backtrace ())
            in
            Stdio.Out_channel.flush Stdio.stderr;
            Unix.dup2 saved Unix.stderr;
            outcome)
      in
      let text = Stdio.In_channel.read_all file in
      Stdio.eprintf "%s%!" text;
      match outcome with
      | Ok result -> (result, String.split_lines text)
      | Error (exn, backtrace) -> Stdlib.Printexc.raise_with_backtrace exn backtrace)

(* The [autotune-progress:] lines among [lines] that carry every one of [fields] ([key=value] words,
   values unquoted as the stage and event names print). *)
let progress_lines lines fields =
  List.count lines ~f:(fun line ->
      match String.chop_prefix line ~prefix:"autotune-progress: " with
      | None -> false
      | Some rest ->
          let words = String.split rest ~on:' ' in
          List.for_all fields ~f:(fun field -> List.mem words field ~equal:String.equal))

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
      ()
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
    (r.Autotune.fiss_sketch_playoff_timed = 0 && List.length spread.composite_ms = 1)

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
    && Float.equal traced.report.Autotune.best_ms quiet.report.Autotune.best_ms)

(* The untuned control under a scoped [~log:true]: the logged search times everything the unlogged
   [ties] did, in the same phases, then the control once more with no phase. Its window is real (the
   measurement seam scripts candidates only), so it moves no crown. *)
let () =
  (* Without this, a line below could come from the ambient config rather than the override. *)
  p "streams: the ambient config leaves both diagnostic streams off" (streams_off ());
  let logged, lines =
    with_stderr_captured (fun () ->
        search ~log:true ~progress:true ~tag:"logged" ~single:(fun _ -> 100.) ~composite:faster ())
  in
  Stdio.eprintf "logged trace (not part of the golden): %s\n%!"
    (String.concat ~sep:" " (List.map logged.phases ~f:(Option.value ~default:"-")));
  p "log: the logged search adds exactly one decision, last and phase-less (the untuned control)"
    (List.equal (Option.equal String.equal) logged.phases (ties.phases @ [ None ]));
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
  let pa = matrix ~label:"pa" ~modulus:11 ~offset:0. ~stride:0.125 in
  let pb = matrix ~label:"pb" ~modulus:7 ~offset:(-3.) ~stride:1. in
  let%op pc = pa * pb in
  let comp = Train.forward pc in
  let old_measured = !Autotune.on_candidate_measured in
  let (), lines =
    with_stderr_captured (fun () ->
        Exn.protect
          ~finally:(fun () -> Autotune.on_candidate_measured := old_measured)
          ~f:(fun () ->
            (Autotune.on_candidate_measured := fun ~label:_ ~digest:_ _ms -> 1.);
            let ctx, _routine =
              Autotune.with_uncontended_test_windows (fun () ->
                  Train.tune_placements ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:"" ~log:true
                    ~progress:true (Context.cpu ()) pc comp Ir.Indexing.Empty)
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
