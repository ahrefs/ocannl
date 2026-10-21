(* gh-ocannl-1277: the splice-segments A/B's cache splicing, on synthetic cache directories.

   Both arms replay the control's schedule outside the target segments: the control arm is the
   control's cache, the treatment arm the control's cache with the targets' schedules taken from the
   treatment's. Pinned here: what each arm's directory holds after [write_arm] (read back through
   the cache's own lookup), the targets' two spellings, and every refusal -- a splice by position
   into a different cut, a target that selects nothing, an ambiguous index, an older entry version
   that was not adopted. No backend: the entries are planted by hand and nothing is compiled. *)

open Base
open Verdict.Claims
module SC = Ir.Schedule_cache
module S = Bench_splice
module Cache_plant = Test_utils.Cache_plant

(* Every directory is a literal carrying the schedule caches' ignored prefix, in the test's working
   directory, and removed at both ends of the run. *)
let control_cache_dir = "autotune_cache_splice_control"
let treatment_cache_dir = "autotune_cache_splice_treatment"

let rec remove_tree path =
  if Stdlib.Sys.file_exists path then
    if Stdlib.Sys.is_directory path then (
      Array.iter (Stdlib.Sys.readdir path) ~f:(fun f -> remove_tree (Stdlib.Filename.concat path f));
      Stdlib.Sys.rmdir path)
    else Stdlib.Sys.remove path

let sched s : SC.saved_schedule = SC.saved_schedule_of_sexp (Sexplib.Sexp.of_string s)

let split f =
  sched (Printf.sprintf "((Split (axis (Base 0)) (factor %d) (outer Grid) (inner Workgroup)))" f)

let coal =
  sched
    "((Coalesce (outer (Base 0)) (inner (Base 1))) (Split (axis (Base 0)) (factor 64) (outer Grid) \
     (inner Workgroup)))"

(* Four distinct segment codes: A and B elsewhere in the step, Q the target (twice in the step, once
   in arm B's routine), S a solo segment. *)
let d_a = String.make 32 'a'
let d_b = String.make 32 'b'
let d_q = "c0ffee" ^ String.make 26 '0'
let d_s = String.make 32 'e'
let src_step = "1111" ^ String.make 28 '1'
let src_arm_b = "2222" ^ String.make 28 '2'
let src_small = "3333" ^ String.make 28 '3'
let src_ctrl_only = "4444" ^ String.make 28 '4'
let src_treat_only = "5555" ^ String.make 28 '5'

let seg ?(kind = `Normal) ?(units = 1) digest saved : SC.saved_segment =
  { seg_kind = kind; seg_units = units; seg_digest = digest; seg_saved = saved }

let entry ?(version = SC.entry_version) ?saved ?segments ~source_digest () =
  { (Cache_plant.entry ?saved ?segments ~backend:"synthetic" ~source_digest ()) with SC.version }

let placement ~outcome decision : SC.placement_entry =
  {
    version = SC.placement_entry_version;
    backend = "synthetic";
    numerics = "n";
    codegen = "c";
    objective = "queued";
    problem_digest = "p";
    decision;
    outcome_digest = outcome;
    shipped_ms = 1.;
    arm_a_ms = 1.;
    arm_b_ms = 2.;
  }

let step_segments ~a ~q1 ~q2 =
  [ seg d_a a; seg ~kind:`Solo ~units:2 d_s []; seg d_q q1; seg d_q q2; seg d_b (split 32) ]

let control_step = step_segments ~a:(split 256) ~q1:(split 8) ~q2:(split 8)

(* The treatment's own search also crowned something else for segment 0: the variance the splice
   exists to remove. *)
let treatment_step = step_segments ~a:(split 128) ~q1:coal ~q2:coal

let store_all ~cache_dir entries =
  let (), io =
    SC.recording_cache_io (fun () ->
        List.iter entries ~f:(fun (key, e) -> SC.store ~dir:cache_dir ~key:(Some key) e))
  in
  match List.find io ~f:(fun (r : SC.cache_io) -> Option.is_some r.SC.refusal) with
  | Some r -> failwith (Printf.sprintf "planting %s was refused" r.SC.key)
  | None -> ()

let plant_control ?(version = SC.entry_version) ?(placements = true) ~cache_dir () =
  store_all ~cache_dir
    [
      ("k-step", entry ~version ~segments:control_step ~source_digest:src_step ());
      ( "k-arm-b",
        entry ~version
          ~segments:[ seg d_q (split 8); seg d_b (split 16) ]
          ~source_digest:src_arm_b () );
      ("k-small", entry ~version ~saved:(split 64) ~source_digest:src_small ());
      ("k-ctrl-only", entry ~version ~saved:(split 4) ~source_digest:src_ctrl_only ());
    ];
  (* Arm A shipped: the step routine. *)
  if placements then
    SC.store_placements ~dir:cache_dir ~key:(Some "placements-k")
      (placement ~outcome:src_step SC.Default);
  SC.store_abandonment ~dir:cache_dir ~key:(Some "k-step")
    { SC.version; source_digest = src_step; trajectory = { search_shape = ""; steps = [] } }

let plant_treatment ?(step = treatment_step) ?(step_saved = []) ?step_entry ?arm_b ~cache_dir () =
  store_all ~cache_dir
    [
      ( "k-step",
        Option.value step_entry
          ~default:(entry ~saved:step_saved ~segments:step ~source_digest:src_step ()) );
      ( "k-arm-b",
        Option.value arm_b
          ~default:
            (entry ~segments:[ seg d_q coal; seg d_b (split 16) ] ~source_digest:src_arm_b ()) );
      ("k-small", entry ~saved:(split 128) ~source_digest:src_small ());
      ("k-treat-only", entry ~saved:(split 2) ~source_digest:src_treat_only ());
    ];
  SC.store_placements ~dir:cache_dir ~key:(Some "placements-k")
    (placement ~outcome:src_arm_b SC.Materialize_all)

let targets s =
  match S.parse_targets s with Ok t -> t | Error e -> failwith ("parse_targets: " ^ e)

let plan ?adopt_version ?(control = control_cache_dir) ?(treatment = treatment_cache_dir) t =
  S.make_plan ?adopt_version ~control ~treatment ~targets:(targets t) ()

let ok_plan ?adopt_version ?control ?treatment t =
  match plan ?adopt_version ?control ?treatment t with
  | Ok p -> p
  | Error e -> failwith ("expected a plan, got the refusal: " ^ e)

let refusal label result ~mentions =
  let msg = match result with Ok _ -> "(no refusal)" | Error e -> e in
  Stdio.printf "%s: %s\n" label msg;
  pf "%s is refused, naming %S" label mentions
    (Result.is_error result && String.is_substring msg ~substring:mentions)

let segs_of ~cache_dir key =
  match SC.lookup ~dir:cache_dir ~key:(Some key) with
  | Some e -> Option.value e.SC.segments ~default:[]
  | None -> []

let saved_of ~cache_dir key =
  Option.map (SC.lookup ~dir:cache_dir ~key:(Some key)) ~f:(fun e -> e.SC.saved)

let schedules ~cache_dir key =
  List.map (segs_of ~cache_dir key) ~f:(fun (s : SC.saved_segment) -> s.SC.seg_saved)

let placement_of ~cache_dir key =
  Option.map (SC.lookup_placements ~dir:cache_dir ~key:(Some key)) ~f:(fun e -> e.SC.decision)

let all_dirs =
  [
    control_cache_dir;
    treatment_cache_dir;
    "autotune_cache_splice_arm_t";
    "autotune_cache_splice_arm_c";
    "autotune_cache_splice_unused";
    "autotune_cache_splice_t_units";
    "autotune_cache_splice_t_digest";
    "autotune_cache_splice_t_count";
    "autotune_cache_splice_t_prelude";
    "autotune_cache_splice_t_whole";
    "autotune_cache_splice_t_source";
    "autotune_cache_splice_t_arm_b";
    "autotune_cache_splice_arm_left";
    "autotune_cache_splice_no_decision";
    "autotune_cache_splice_old";
    "autotune_cache_splice_arm_adopted";
    "autotune_cache_splice_stampless";
  ]

let () =
  List.iter all_dirs ~f:remove_tree;
  plant_control ~cache_dir:control_cache_dir ();
  plant_treatment ~cache_dir:treatment_cache_dir ();
  (* The listing a person reads to pick the targets: Q's rows differ between the arms, and so does
     segment 0, which no target selects. *)
  let p = ok_plan "c0ffee" in
  Stdio.print_endline "== listing, target c0ffee";
  List.iter (S.describe p) ~f:Stdio.print_endline;
  pf "the digest prefix selects the three Q segments across both routines"
    (List.equal Poly.equal (List.map p.S.targets ~f:snd) [ 3 ]);
  let arm_t_cache_dir = "autotune_cache_splice_arm_t" in
  let arm_c_cache_dir = "autotune_cache_splice_arm_c" in
  let written = function Ok n -> n | Error e -> failwith ("write_arm: " ^ e) in
  let t_n = written (S.write_arm ~cache_dir:arm_t_cache_dir p ~arm:S.Treatment ()) in
  let c_n = written (S.write_arm ~cache_dir:arm_c_cache_dir p ~arm:S.Control ()) in
  pf "each arm holds the control's four schedule entries and its placement decision"
    (t_n = 5 && c_n = 5);
  pf
    "the treatment arm takes the targets' schedules from the treatment and the rest from the \
     control"
    (List.equal SC.equal_saved_schedule
       (schedules ~cache_dir:arm_t_cache_dir "k-step")
       [ split 256; []; coal; coal; split 32 ]
    && List.equal SC.equal_saved_schedule
         (schedules ~cache_dir:arm_t_cache_dir "k-arm-b")
         [ coal; split 16 ]);
  let entry_of ~cache_dir key = SC.lookup ~dir:cache_dir ~key:(Some key) in
  pf "a spliced entry carries no timing: the hybrid was never measured"
    (Option.value_map (entry_of ~cache_dir:arm_t_cache_dir "k-step") ~default:false ~f:(fun e ->
         Float.equal e.SC.best_ms Float.infinity
         && Option.is_none e.SC.mma_best_ms && Option.is_none e.SC.best_steps
         && Option.is_none e.SC.default_ms));
  pf "an entry replayed unchanged keeps its measurements"
    (Option.value_map (entry_of ~cache_dir:arm_c_cache_dir "k-step") ~default:false ~f:(fun e ->
         Float.equal e.SC.best_ms 0.));
  pf "the control arm is the control's cache"
    (List.equal SC.equal_saved_schedule
       (schedules ~cache_dir:arm_c_cache_dir "k-step")
       (List.map control_step ~f:(fun (s : SC.saved_segment) -> s.SC.seg_saved))
    && List.equal SC.equal_saved_schedule
         (schedules ~cache_dir:arm_c_cache_dir "k-arm-b")
         [ split 8; split 16 ]);
  let both f = f ~cache_dir:arm_t_cache_dir && f ~cache_dir:arm_c_cache_dir in
  pf "an untargeted whole-routine entry replays the control's schedule in both arms"
    (both (fun ~cache_dir ->
         Option.equal SC.equal_saved_schedule (saved_of ~cache_dir "k-small") (Some (split 64))));
  pf "a control-only entry is in both arms, a treatment-only entry in neither"
    (both (fun ~cache_dir ->
         Option.is_some (saved_of ~cache_dir "k-ctrl-only")
         && Option.is_none (saved_of ~cache_dir "k-treat-only")));
  pf "both arms ship the control's placement decision"
    (both (fun ~cache_dir ->
         Option.equal SC.equal_placement_decision
           (placement_of ~cache_dir "placements-k")
           (Some SC.Default)));
  pf "reading the two caches left them unchanged"
    (Option.equal SC.equal_placement_decision
       (placement_of ~cache_dir:treatment_cache_dir "placements-k")
       (Some SC.Materialize_all)
    && List.equal SC.equal_saved_schedule
         (schedules ~cache_dir:treatment_cache_dir "k-step")
         (List.map treatment_step ~f:(fun (s : SC.saved_segment) -> s.SC.seg_saved)));
  (* The inputs are another binary's caches. A lookup reads a directory whose regime stamp is older
     or missing as all misses (gh-ocannl-1439); the splice reads entry files, so it still finds the
     schedules there, and neither may delete them. *)
  let stampless_cache_dir = "autotune_cache_splice_stampless" in
  plant_control ~cache_dir:stampless_cache_dir ();
  Stdlib.Sys.remove (Stdlib.Filename.concat stampless_cache_dir SC.regime_stamp_filename);
  let entry_files () =
    Array.count (Stdlib.Sys.readdir stampless_cache_dir) ~f:(String.is_suffix ~suffix:".sexp")
  in
  let before = entry_files () in
  let p_stampless = plan ~control:stampless_cache_dir "c0ffee" in
  pf "a cache stamped by no regime is spliced from, and left whole"
    (Result.is_ok p_stampless && before = 6 && entry_files () = before);
  pf "where a lookup reads the same cache as a miss, and leaves it whole too"
    (Option.is_none (SC.lookup ~dir:stampless_cache_dir ~key:(Some "k-step"))
    && entry_files () = before);
  refusal "writing an arm into a non-empty directory"
    (S.write_arm ~cache_dir:arm_t_cache_dir p ~arm:S.Treatment ())
    ~mentions:"absent or empty";
  (* An index names one entry: ambiguous while two routines are fissioned in both caches. *)
  refusal "a bare index with two fissioned entries" (plan "#2") ~mentions:"ambiguous";
  let p_idx = ok_plan "1111#2" in
  pf "an entry-prefixed index selects that one segment"
    (List.equal Poly.equal (List.map p_idx.S.targets ~f:snd) [ 1 ]);
  refusal "an index past the entry's segments" (plan "1111#9") ~mentions:"has 5 segments";
  refusal "a target that selects nothing" (plan "dddddd") ~mentions:"selects no segment";
  refusal "a too-short digest prefix"
    (Result.map_error (S.parse_targets "c0f") ~f:Fn.id)
    ~mentions:"at least 6 hex digits";
  refusal "writing an arm with nothing targeted"
    (Result.bind
       (S.make_plan ~control:control_cache_dir ~treatment:treatment_cache_dir ~targets:[] ())
       ~f:(fun p -> S.write_arm ~cache_dir:"autotune_cache_splice_unused" p ~arm:S.Treatment ()))
    ~mentions:"nothing is targeted";
  (* Splicing by position into a different cut is refused, never repaired. *)
  let refused_treatment ~cache_dir ?step ?step_saved ?step_entry ?arm_b () =
    plant_treatment ?step ?step_saved ?step_entry ?arm_b ~cache_dir ();
    plan ~treatment:cache_dir "c0ffee"
  in
  refusal "a segmentation that differs outside the target"
    (refused_treatment ~cache_dir:"autotune_cache_splice_t_units"
       ~step:
         [
           seg d_a ~units:2 (split 128);
           seg ~kind:`Solo ~units:1 d_s [];
           seg d_q coal;
           seg d_q coal;
           seg d_b (split 32);
         ]
       ())
    ~mentions:"segmentations differ at segment 0";
  refusal "a different code cut at a non-target position"
    (refused_treatment ~cache_dir:"autotune_cache_splice_t_digest"
       ~step:
         (List.mapi treatment_step ~f:(fun i s ->
              if i = 4 then { s with SC.seg_digest = d_a } else s))
       ())
    ~mentions:"segmentations differ at segment 4";
  refusal "a different number of segments"
    (refused_treatment ~cache_dir:"autotune_cache_splice_t_count" ~step:(List.take treatment_step 4)
       ())
    ~mentions:"5 control segments, 4 treatment segments";
  refusal "a whole-routine prelude that differs"
    (refused_treatment ~cache_dir:"autotune_cache_splice_t_prelude" ~step_saved:(split 2) ())
    ~mentions:"preludes differ";
  refusal "a target in a shipping routine the treatment won whole"
    (refused_treatment ~cache_dir:"autotune_cache_splice_t_whole"
       ~step_entry:(entry ~saved:coal ~source_digest:src_step ())
       ())
    ~mentions:"whole-routine";
  refusal "one key held for two source digests"
    (refused_treatment ~cache_dir:"autotune_cache_splice_t_source"
       ~arm_b:(entry ~segments:[ seg d_q coal; seg d_b (split 16) ] ~source_digest:src_small ())
       ())
    ~mentions:"different source digests";
  (* A routine the control's placement decision does not ship (arm B's) replays only when the
     placement store is bypassed: a target there that cannot be spliced is left as the control's and
     reported, not refused -- unless the control recorded no decision, when every routine counts. *)
  let t_arm_b_cache_dir = "autotune_cache_splice_t_arm_b" in
  plant_treatment
    ~arm_b:(entry ~saved:coal ~source_digest:src_arm_b ())
    ~cache_dir:t_arm_b_cache_dir ();
  let p_left = ok_plan ~treatment:t_arm_b_cache_dir "c0ffee" in
  Stdio.print_endline "== listing, arm B's winner whole-routine in the treatment";
  List.iter (S.describe p_left) ~f:Stdio.print_endline;
  let left_cache_dir = "autotune_cache_splice_arm_left" in
  ignore (written (S.write_arm ~cache_dir:left_cache_dir p_left ~arm:S.Treatment ()) : int);
  pf "an unspliceable target in a routine that does not ship leaves it as the control's"
    (List.equal Poly.equal (List.map p_left.S.targets ~f:snd) [ 2 ]
    && List.equal SC.equal_saved_schedule
         (schedules ~cache_dir:left_cache_dir "k-arm-b")
         [ split 8; split 16 ]
    && List.equal SC.equal_saved_schedule
         (schedules ~cache_dir:left_cache_dir "k-step")
         [ split 256; []; coal; coal; split 32 ]);
  (* What a run replayed decides whether it measured the splice: the cache lookups of its compile
     must reach a spliced routine, and no routine a target was left unspliced in. *)
  pf "a compile that replayed the spliced routine measured the splice"
    (Result.is_ok
       (S.replay_verdict p ~arm_reports:1 ~looked_up:[ "placements-k"; "k-step"; "k-small" ]));
  refusal "a compile that replayed no placement decision"
    (S.replay_verdict p ~arm_reports:2 ~looked_up:[ "k-step"; "k-arm-b" ])
    ~mentions:"did not replay the control's placement decision";
  refusal "a compile whose looked-up decision was rejected, both candidates compared"
    (S.replay_verdict p ~arm_reports:2 ~looked_up:[ "placements-k"; "k-step"; "k-arm-b" ])
    ~mentions:"did not replay the control's placement decision";
  refusal "a compile that replayed only untargeted routines"
    (S.replay_verdict p ~arm_reports:1 ~looked_up:[ "placements-k"; "k-small"; "k-ctrl-only" ])
    ~mentions:"replayed no spliced routine";
  refusal "a compile that replayed a routine left unspliced"
    (S.replay_verdict p_left ~arm_reports:1 ~looked_up:[ "placements-k"; "k-step"; "k-arm-b" ])
    ~mentions:"left unspliced";
  let no_decision_cache_dir = "autotune_cache_splice_no_decision" in
  plant_control ~placements:false ~cache_dir:no_decision_cache_dir ();
  refusal "the same target with no placement decision recorded"
    (plan ~control:no_decision_cache_dir ~treatment:t_arm_b_cache_dir "c0ffee")
    ~mentions:"whole-routine";
  refusal "targets that splice only a routine that does not ship" (plan "2222#0")
    ~mentions:"no routine the control's placement decisions ship";
  (* A treatment that adds a search branch bumps the entry version: the control's older entries are
     undecodable until adopted, and re-stamped current once they are. *)
  let old = SC.entry_version - 1 in
  let old_cache_dir = "autotune_cache_splice_old" in
  plant_control ~version:old ~cache_dir:old_cache_dir ();
  refusal "an older control version, not adopted"
    (plan ~control:old_cache_dir "c0ffee")
    ~mentions:"selects no segment";
  let p_old = ok_plan ~adopt_version:old ~control:old_cache_dir "c0ffee" in
  let adopted_cache_dir = "autotune_cache_splice_arm_adopted" in
  ignore (written (S.write_arm ~cache_dir:adopted_cache_dir p_old ~arm:S.Treatment ()) : int);
  pf "an adopted control replays re-stamped at the current version, spliced alike"
    (Option.value_map (SC.lookup ~dir:adopted_cache_dir ~key:(Some "k-step")) ~default:false
       ~f:(fun e -> e.SC.version = SC.entry_version)
    && List.equal SC.equal_saved_schedule
         (schedules ~cache_dir:adopted_cache_dir "k-step")
         [ split 256; []; coal; coal; split 32 ]);
  List.iter all_dirs ~f:remove_tree
