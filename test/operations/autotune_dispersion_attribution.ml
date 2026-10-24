(* gh-ocannl-1558: a dispersed timing window on a CPU backend is refused as host contention only
   when an independent host-load signal shows a co-tenant.

   The 2x-majority rule ([Autotune.window_result]) refused a window when at least half its samples
   exceeded twice its fastest. A 32-thread cc routine's own run-to-run spread meets that on a quiet
   host (step p90 1.7-4.9x p10 on minix, gh-ocannl-1482), and any refused window makes a search
   uncacheable -- so no search there stored an entry. Now the rule only says the window is
   [dispersed]; on a CPU backend the timing call reads [Utils.Host_load] around the window, admits a
   dispersed window when the foreign CPU load over it is below a core ([Quiet]), and keeps refusing
   it under a co-tenant or with no signal. GPU backends keep the dispersion-only rule.

   Every leg drives the load evidence through [Autotune.host_load_reading] and the clock through an
   injected [batch] or [Autotune.on_batch_wall], never the machine: a scripted high-variance clock
   under a scripted quiet host is admitted and the executed search caches; the same clock under a
   scripted co-tenant is refused and the search does not. One leg reads the real counters, and
   claims only that they resolve. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Host_load = Utils.Host_load
module SC = Ir.Schedule_cache
module Cache_plant = Test_utils.Cache_plant
open Verdict.Claims

let sample ms : Autotune.timing_sample = { per_launch_ms = ms; contention_ms = ms }

(* --- The window and its attribution, pure. --- *)

let () =
  Stdio.printf "== the attribution, pure ==\n";
  (* One clean 16 ms sample and fifteen at 64 ms: the window's median is four times its floor. *)
  let noisy = Autotune.window_result (sample 16. :: List.init 15 ~f:(fun _ -> sample 64.)) in
  p "a mostly slow window is dispersed, and the dispersion-only rule refuses it"
    (noisy.dispersed && noisy.contended
    && Poly.equal (Autotune.timing_admission noisy) Autotune.Refused_contended);
  p "its spread is its median over its floor" (Float.equal noisy.spread 4.);
  let quiet = Autotune.attribute_dispersion (Host_load.Quiet 0.125) noisy in
  p "a quiet host admits it at its minimum"
    (Option.equal Float.equal (Autotune.admitted_timing_ms quiet) (Some 16.));
  p "the admitted window keeps its dispersion and spread"
    (quiet.dispersed && Float.equal quiet.spread noisy.spread);
  p "a co-tenant keeps the refusal"
    (Poly.equal
       (Autotune.timing_admission (Autotune.attribute_dispersion (Host_load.Cotenant 3.) noisy))
       Autotune.Refused_contended);
  p "no signal keeps the refusal"
    (Poly.equal
       (Autotune.timing_admission (Autotune.attribute_dispersion Host_load.Unavailable noisy))
       Autotune.Refused_contended);
  let tight = Autotune.window_result (List.init 16 ~f:(fun _ -> sample 20.)) in
  p "a uniformly delayed window under a co-tenant is admitted as before"
    ((not tight.dispersed)
    && Option.is_some
         (Autotune.admitted_timing_ms (Autotune.attribute_dispersion (Host_load.Cotenant 3.) tight))
    );
  let unreadable = Autotune.window_result (List.init 16 ~f:(fun _ -> sample 0.)) in
  p "a quiet host does not make an unresolved clock usable"
    (Poly.equal
       (Autotune.timing_admission (Autotune.attribute_dispersion (Host_load.Quiet 0.) unreadable))
       Autotune.Refused_unreadable);
  (* The policy follows the tuner's own CPU predicate rather than a restated backend list. *)
  p_all "the attribution policy is the CPU backends' and only theirs"
    [ "cc"; "multidev_cc"; "cuda"; "hip"; "metal" ] ~f:(fun b ->
      Bool.equal
        (Poly.equal (Autotune.dispersion_policy_for_backend b) Autotune.Attribute_dispersion)
        (Ir.Schedule.backend_is_cpu b))

(* --- The host-load classification, pure. --- *)

let reading ~wall_s ~busy_s ~own_s : Host_load.reading = { wall_s; busy_s; own_s }
let start = reading ~wall_s:10. ~busy_s:1000. ~own_s:500.

(* A span of [span] seconds in which this process used [own] CPU-seconds and the machine [own +
   foreign * span]. *)
let after ~span ~own ~foreign =
  reading ~wall_s:(10. +. span) ~busy_s:(1000. +. own +. (foreign *. span)) ~own_s:(500. +. own)

let is_quiet = function Host_load.Quiet _ -> true | _ -> false
let is_cotenant = function Host_load.Cotenant _ -> true | _ -> false

let () =
  Stdio.printf "== the host-load classification ==\n";
  let classify after = Host_load.classify ~before:start ~after () in
  (* A 32-thread routine using nearly all of the machine's busy time is this process, not a
     co-tenant. *)
  p "a many-thread routine consuming the machine's busy time is quiet"
    (is_quiet (classify (after ~span:1. ~own:31.5 ~foreign:0.125)));
  p "residual load of several cores is a co-tenant"
    (is_cotenant (classify (after ~span:1. ~own:16. ~foreign:3.)));
  (* The bar, from the module's own constants: [cotenant_cores] plus the counter allowance over the
     span. *)
  let bar span = Host_load.cotenant_cores +. (Host_load.counter_allowance_s /. span) in
  p_all "just under the bar is quiet and just over it a co-tenant" [ 0.25; 1.; 4. ] ~f:(fun span ->
      is_quiet (classify (after ~span ~own:8. ~foreign:(bar span -. 0.01)))
      && is_cotenant (classify (after ~span ~own:8. ~foreign:(bar span +. 0.01))));
  p "the allowance asks a short span for more evidence than a long one"
    (let foreign = (bar 0.25 +. bar 4.) /. 2. in
     is_quiet (classify (after ~span:0.25 ~own:2. ~foreign))
     && is_cotenant (classify (after ~span:4. ~own:2. ~foreign)));
  p "the precise own time running ahead of the tick-sampled busy count is quiet"
    (is_quiet (classify (after ~span:1. ~own:8. ~foreign:(-0.5))));
  p_all "a counter that went backwards, a non-finite reading or an empty span is no signal"
    [
      reading ~wall_s:11. ~busy_s:999. ~own_s:501.;
      reading ~wall_s:11. ~busy_s:1001. ~own_s:499.;
      reading ~wall_s:11. ~busy_s:Float.nan ~own_s:501.;
      reading ~wall_s:10. ~busy_s:1001. ~own_s:501.;
    ]
    ~f:(fun after -> Host_load.equal_verdict (classify after) Host_load.Unavailable);
  (* A short window padded to [min_span_s]: two cores for the window's 50 ms and nothing after. Over
     the padded span that averages under a core; charged to the window, as the timing call charges
     it, it is the co-tenant it was (local Codex review). *)
  let padded =
    after ~span:Host_load.min_span_s ~own:0.4 ~foreign:(2. *. 0.05 /. Host_load.min_span_s)
  in
  p "padding a short window does not dilute a co-tenant that ran only during it"
    (is_quiet (classify padded)
    && is_cotenant (Host_load.classify ~window_s:0.05 ~before:start ~after:padded ()));
  p_all "a window longer than its span, or not positive, is no signal"
    [ 0.; -1.; 2. *. Host_load.min_span_s ]
    ~f:(fun window_s ->
      Host_load.equal_verdict
        (Host_load.classify ~window_s ~before:start ~after:padded ())
        Host_load.Unavailable);
  (* The real counters: where the platform has them, two readings around an idle wait resolve to a
     verdict, and this process's CPU time did not run backwards. Whether the host is quiet is the
     machine's business, not a claim. *)
  let r0 = Host_load.read () in
  Unix.sleepf 0.3;
  let r1 = Host_load.read () in
  (match (r0, r1) with
  | Some before, Some after ->
      Stdio.eprintf "real host-load verdict over an idle 0.3 s (not part of the golden): %s\n%!"
        (Sexp.to_string_hum (Host_load.sexp_of_verdict (Host_load.classify ~before ~after ())))
  | _ -> ());
  gated ~aggregation:`Environment ~when_:(Option.is_some r0) ~on:"a platform without a busy counter"
    "the real counters resolve to a verdict over a 0.3 s span"
    (match (r0, r1) with
    | Some before, Some after ->
        Float.(after.wall_s -. before.wall_s >= 0.3)
        && Float.(after.own_s >= before.own_s)
        && not
             (Host_load.equal_verdict (Host_load.classify ~before ~after ()) Host_load.Unavailable)
    | _ -> false)

(* --- One timing call on a scripted clock and a scripted host. --- *)

(* Readings one second apart, each span carrying [foreign] cores beside 8 cores of this process: a
   span long enough that no reading waits. Each reading is logged, so a claim can place them around
   the batches. *)
let scripted_host ~foreign events =
  let k = ref 0 in
  fun () ->
    let t = Float.of_int !k in
    Int.incr k;
    events := `Load :: !events;
    Some (reading ~wall_s:t ~busy_s:((8. +. foreign) *. t) ~own_s:(8. *. t))

(* Every fourth batch at [floor_ms] per launch, the rest at three times it: a majority beyond twice
   the floor. *)
let scripted_clock ~floor_ms events =
  let ordinal = ref 0 in
  fun depth ->
    let per_launch = if !ordinal % 4 = 0 then floor_ms else 3. *. floor_ms in
    Int.incr ordinal;
    events := `Batch :: !events;
    per_launch *. Float.of_int depth

type call = {
  result : Autotune.timing_result;
  c : Autotune.timing_call;
  events : [ `Load | `Batch ] list;
}

let call ?host ~dispersion ~foreign ~timing ~floor_ms () =
  let events = ref [] and record = ref None in
  let host = match host with Some h -> h events | None -> scripted_host ~foreign events in
  let result =
    Autotune.Seam.with_value Autotune.host_load_reading host (fun () ->
        Autotune.calibrate_and_time_with_retry_observer
          ~on_retry:(fun () -> ())
          ~on_call:(fun c -> record := Some c)
          ~phase:None ~retry_contended:false ~dispersion ~timing ~repeats:3
          ~queue_depth_cap:(Autotune.queue_depth_cap_for_backend "cc")
          ~batch:(scripted_clock ~floor_ms events))
  in
  { result; c = Option.value_exn ~here:[%here] !record; events = List.rev !events }

let () =
  Stdio.printf "== one timing call ==\n";
  (* Isolated; queued at a fast floor (a batch deeper than one); queued at a floor above the batch
     target, which settles at depth 1 and resumes the calibration's singles as its window. *)
  let shapes =
    [
      ("isolated", Autotune.Isolated, 1.);
      ("queued", Autotune.Queued, 0.25);
      ("queued depth-1 reuse", Autotune.Queued, 20.);
    ]
  in
  List.iter shapes ~f:(fun (name, timing, floor_ms) ->
      let quiet =
        call ~dispersion:Autotune.Attribute_dispersion ~foreign:0.125 ~timing ~floor_ms ()
      in
      let cotenant =
        call ~dispersion:Autotune.Attribute_dispersion ~foreign:4. ~timing ~floor_ms ()
      in
      let refuse = call ~dispersion:Autotune.Refuse_dispersed ~foreign:0.125 ~timing ~floor_ms () in
      Stdio.eprintf "%s (not part of the golden): depth %d, %d samples (%d reused), spread %.3f\n%!"
        name quiet.c.decision.depth quiet.c.window.samples quiet.c.window.reused quiet.result.spread;
      p
        (name ^ ": the scripted window is dispersed with spread 3")
        (quiet.result.dispersed
        && Float.equal quiet.result.spread 3.
        && Float.equal quiet.c.window.spread 3.);
      p
        (name ^ ": a quiet host admits it at the floor")
        (Option.equal Float.equal (Autotune.admitted_timing_ms quiet.result) (Some floor_ms)
        && Poly.equal quiet.c.decision.admission Autotune.Admitted
        && Option.exists quiet.c.decision.host_load ~f:is_quiet);
      p
        (name ^ ": a co-tenant refuses it as contention")
        (Poly.equal (Autotune.timing_admission cotenant.result) Autotune.Refused_contended
        && Poly.equal cotenant.c.decision.admission Autotune.Refused_contended
        && Option.exists cotenant.c.decision.host_load ~f:is_cotenant);
      p
        (name ^ ": the dispersion-only policy refuses it and reads no load")
        (Poly.equal (Autotune.timing_admission refuse.result) Autotune.Refused_contended
        && Option.is_none refuse.c.decision.host_load
        && not (List.mem refuse.events `Load ~equal:Poly.equal));
      (* The span covers exactly the window's samples: after the reading that opens it come the
         window's batches and nothing else -- reused calibration singles counted, unused calibration
         batches not (GitHub Codex review, round 1) -- and the closing reading is the call's last
         event. *)
      p
        (name ^ ": the load span covers exactly the window's batches")
        (let rev = List.rev quiet.events in
         match rev with
         | `Load :: rest -> (
             match List.findi rest ~f:(fun _ e -> Poly.equal e `Load) with
             (* Everything between the two last readings is a batch: events are only those. *)
             | Some (i, _) -> i = quiet.c.window.samples
             | None -> false)
         | _ -> false));
  p "the depth-1 shape reuses the calibration singles"
    (let reuse =
       call ~dispersion:Autotune.Attribute_dispersion ~foreign:0.125 ~timing:Autotune.Queued
         ~floor_ms:20. ()
     in
     reuse.c.decision.depth = 1 && reuse.c.window.reused > 0);
  p "the fast queued shape batches deeper than one"
    ((call ~dispersion:Autotune.Attribute_dispersion ~foreign:0.125 ~timing:Autotune.Queued
        ~floor_ms:0.25 ())
       .c
       .decision
       .depth > 1)

(* A window too short for the counters, padded: the co-tenant's 0.1 CPU-s all fall in the window's
   0.05 s, and nothing after it. The first reading is the call's opening one, which an isolated
   window retakes before its first batch; the last is the one the call takes after its wait. *)
let short_window_host events =
  let readings =
    ref
      [
        reading ~wall_s:(-1.) ~busy_s:0. ~own_s:0.;
        reading ~wall_s:0. ~busy_s:0. ~own_s:0.;
        reading ~wall_s:0.05 ~busy_s:0.5 ~own_s:0.4;
        reading ~wall_s:Host_load.min_span_s ~busy_s:0.5 ~own_s:0.4;
      ]
  in
  fun () ->
    events := `Load :: !events;
    match !readings with
    | r :: rest ->
        readings := rest;
        Some r
    | [] -> None

let () =
  let short =
    call ~host:short_window_host ~dispersion:Autotune.Attribute_dispersion ~foreign:0.
      ~timing:Autotune.Isolated ~floor_ms:1. ()
  in
  p
    "a short dispersed window reads once more after its wait and charges the span's foreign load \
     to it"
    ((match List.rev short.events with `Load :: `Load :: `Batch :: _ -> true | _ -> false)
    && Option.exists short.c.decision.host_load ~f:is_cotenant
    && Poly.equal (Autotune.timing_admission short.result) Autotune.Refused_contended)

(* --- An executed cc search on the scripted clock. --- *)

let rows = 8
let cols = 16

let av =
  Array.init (rows * cols)
    ~f:(Ll_test.cycle_flat ~dims:[| rows; cols |] ~modulus:11 ~offset:(-2.5) ~stride:0.5)

let fresh_dir name =
  if Stdlib.Sys.file_exists name && Stdlib.Sys.is_directory name then
    Array.iter (Stdlib.Sys.readdir name) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat name f))

(* The executed clock: every fourth batch reads 1 ms and the rest 4 ms, whatever the routine took,
   so every window of the search is dispersed with spread 4. *)
let executed_clock ~depth ~ordinal _wall =
  (if ordinal % 4 = 0 then 1. else 4.) *. Float.of_int depth

(* One directory for every leg, cleared before each leg that must start cold. *)
let cache_dir = "autotune_cache_dispersion_attribution"

let search ~tag ~foreign comp =
  let report = ref None in
  let events = ref [] in
  let ctx = Context.cpu () in
  let identity = Option.is_some (Context.timing_identity ctx) in
  let (ctx, ios), lines =
    Stderr_capture.lines (fun () ->
        SC.recording_cache_io (fun () ->
            Autotune.Seam.with_value Autotune.host_load_reading (scripted_host ~foreign events)
              (fun () ->
                Autotune.Seam.with_value Autotune.on_batch_wall executed_clock (fun () ->
                    fst
                      (Autotune.tune ~search:true ~rounds:0 ~repeats:1 ~timing:Autotune.Isolated
                         ~cache_dir ~log:true
                         ~report:(fun r -> report := Some r)
                         ctx comp Ir.Indexing.Empty)))))
  in
  Context.release ctx;
  let r = Option.value_exn ~here:[%here] !report in
  let admitted_lines =
    List.count lines ~f:(fun l -> String.is_substring l ~substring:"admitted a dispersed window")
  in
  Stdio.eprintf
    "%s (not part of the golden): %s, timed %d, refused %d (contention %d), dispersed %d, best \
     spread %s, %d cache operations, %d admitted-dispersed log lines\n\
     %!"
    tag (Autotune.outcome_name r.outcome) r.candidates_timed r.timings_contended
    (Autotune.contention_refusals r) r.timings_dispersed
    (Option.value_map r.best_spread ~default:"none" ~f:Float.to_string)
    (List.length ios) admitted_lines;
  (r, ios, identity, admitted_lines)

let schedule_stores ios =
  List.filter ios ~f:(fun (io : SC.cache_io) ->
      (match io.op with SC.Store -> true | SC.Lookup -> false)
      && String.equal (Cache_plant.store_kind io) "schedule")

let () =
  Stdio.printf "== an executed cc search ==\n";
  let a = TDSL.ndarray av ~label:[ "da_a" ] ~output_dims:[ rows; cols ] () in
  let%op s = a ++ "ij => i" in
  let comp = Train.forward s in
  (* A quiet host: every window is admitted as the routine's own variance, and the search caches. *)
  fresh_dir cache_dir;
  let r1, ios1, identity, admitted_lines = search ~tag:"quiet, cold" ~foreign:0.125 comp in
  p "quiet: the search timed candidates and refused no window"
    (Poly.equal r1.outcome Autotune.Searched && r1.candidates_timed > 0 && r1.timings_contended = 0);
  p "quiet: every admitted window was dispersed" (r1.timings_dispersed = r1.candidates_timed);
  (* At least: under [~log:true] the search also times the untuned-default control, whose window is
     attributed and logged like any other but is not one of the report's timings. *)
  p "quiet: each one has a log line naming it" (admitted_lines >= r1.timings_dispersed);
  p "quiet: the winner's spread is recorded" (Option.equal Float.equal r1.best_spread (Some 4.));
  Cache_plant.unless_unkeyed_or_refused ~identity ios1
    "quiet: the search committed a schedule entry"
    (List.exists (schedule_stores ios1) ~f:(Fn.non Cache_plant.refused));
  let r2, ios2, _, _ = search ~tag:"quiet, warm" ~foreign:0.125 comp in
  Cache_plant.unless_unkeyed_or_refused ~identity (ios1 @ ios2)
    "quiet: the second call replays it, with the stored spread"
    (Poly.equal r2.outcome Autotune.Cache_replay
    && Option.equal Float.equal r2.best_spread (Some 4.));
  (* A co-tenant: the same windows are refused, and nothing is stored. *)
  fresh_dir cache_dir;
  let r3, ios3, _, _ = search ~tag:"co-tenant" ~foreign:4. comp in
  p "co-tenant: the windows are refused as contention"
    (r3.timings_contended > 0
    && Autotune.contention_refusals r3 = r3.timings_contended
    && r3.candidates_timed = 0 && r3.timings_dispersed = 0);
  (* Over the search's cache operations, its lookup at least: none of them stored a schedule. *)
  p_none ~min:1 "co-tenant: none of the search's cache operations stores a schedule" ios3
    ~f:(fun io -> not (List.is_empty (schedule_stores [ io ])))
