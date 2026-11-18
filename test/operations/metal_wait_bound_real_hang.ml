(* gh-ocannl-2134, gh-ocannl-2139: the tuner's host-side wait bound against a kernel that really
   never ends and that no driver ever reports.

   Every other test of the bound injects its expiry (test/operations/wait_bound_declines) or expires
   a finite kernel against a zero bound (test/operations/wait_bound_device_wedged). Here the search
   is over a reduction of 10^18 additions -- [y[o] = sum over i, j, k of x[o, k]], with two outer
   extents of 10^6 added to the product space of [x ++ "ok=>o"], so no schedule of it ends within
   days, while its 1024 outputs give the GPU seeds a hardware dimension to bind -- and the tuner's
   own bounded wait ([Autotune.wait_bound_s], the 60 s floor: on Metal the serial baseline is not
   dispatched, so the first candidate's warmup has no reference) is what ends it. The driver
   wrapping [Autotune.Context_driver] only records each bounded wait; it answers nothing itself.

   Whether macOS kills the kernel first is the platform's call (gh-ocannl-2082): where the watchdog
   fires, Metal reports a hang and the search ends on a [gpu_hang] decline instead, a few seconds
   in. Which of the two happened is read off the IORegistry's GPU [recoveryCount], never off the
   backend under test, and each outcome's claims are a leg open only where the count says it
   happened. The claims that the search ended [Search_died] on one candidate's [Sync] are written to
   hold either way, but the [gpu_hang] leg has not run: no fleet Mac's watchdog has fired since the
   test was written (2026-11-18), and gh-ocannl-2139 holds it until one does. The control is the
   same reduction with outer extents of 2, searched to completion: its value, four times each row's
   sum, says the patched product space is iterated as built.

   Gated behind `slow`, Metal only: each run stalls the GPU for the bound's 60 s where the OS does
   not kill, then exits, which frees the GPU at once. With the argument [control] it runs the finite
   search and only compiles the never-ending one, for a dry run on a shared box. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Asgns = Ir.Assignments
module Idx = Ir.Indexing
module SO = Ir.Schedule_outcome
module MF = Ir.Metal_failure
open Verdict.Claims

(* The mode word: [control] anywhere among the arguments, beside any [--ocannl_*] flags. Any other
   word is refused, so that a mistyped dry run never falls through to the stall. *)
let control_only =
  let words =
    List.filter
      (List.tl_exn (Array.to_list (Sys.get_argv ())))
      ~f:(fun a -> not (String.is_prefix a ~prefix:"--"))
  in
  match words with
  | [] -> false
  | [ "control" ] -> true
  | _ ->
      failwith
        ("metal_wait_bound_real_hang: the only mode argument is [control], got: "
       ^ String.concat ~sep:" " words)

(* [x]'s two axes, the outputs and the summed one, and the never-ending search's two outer
   extents. *)
let m = 1024
let inner = 1024
let outer_hang = 1_000_000
let ctx0 = Context.auto ()
let backend_name = Context.backend_name ctx0
let leg = Verdict.leg ~when_:(String.equal backend_name "metal") ~on:backend_name ()
let recovery_count = Gpu_recovery.recovery_count
let arm_watchdog = Gpu_recovery.arm_watchdog

(* [x[o, k]] is [o mod 8] plus 0, 0.25, 0.5 or 0.75 by [k], so the rows differ. *)
let x_value o k = (Float.of_int (k % 4) *. 0.25) +. Float.of_int (o % 8)
let row_sum o = Array.sum (module Float) (Array.init inner ~f:(x_value o)) ~f:Fn.id

(* [y = x ++ "ok=>o"], with two more product components of extent [outer] whose iterators neither
   side reads: [y] sums each row of [x] [outer * outer] times. [Train.forward] gives that reduction
   as one [Accum_op] leaf, which is rebuilt alone. *)
let reduction ~label ~outer =
  let x =
    TDSL.ndarray
      (Array.init (m * inner) ~f:(fun f -> x_value (f / inner) (f % inner)))
      ~label:[ label ^ "_x" ]
      ~output_dims:[ m; inner ] ()
  in
  let%op y = x ++ "ok=>o" in
  let fwd = Train.forward y in
  let leaves = Asgns.fold_leaves fwd.Asgns.asgns ~init:[] ~f:(fun acc leaf -> leaf :: acc) in
  let patched =
    match leaves with
    | [ Asgns.Accum_op a ] ->
        let p = Lazy.force a.projections in
        let extra = Array.init 2 ~f:(fun _ -> [ (outer, Idx.get_symbol ()) ]) in
        let p = { p with Idx.components = Array.append extra p.Idx.components } in
        Asgns.Accum_op { a with projections = Lazy.from_val p }
    | _ -> failwith "expected the reduction's forward code to be one Accum_op"
  in
  (Asgns.comp ~embedded:fwd.Asgns.embedded_nodes (Asgns.Block_comment (label, patched)), y)

(* Every bounded wait of the current search: its bound, the production wait's answer and how long it
   took. *)
type wait = { bound : float; completed : bool; took : float }

let waits : wait list ref = ref []
let search_start = ref 0.

module Recording_driver = struct
  include Autotune.Context_driver

  let await_within ctx ~seconds =
    let t = Unix.gettimeofday () and c = Mtime_clock.counter () in
    let completed = Autotune.Context_driver.await_within ctx ~seconds in
    let took = Mtime.Span.to_float_ns (Mtime_clock.count c) /. 1e9 in
    waits := { bound = seconds; completed; took } :: !waits;
    if (not completed) || List.length !waits = 1 then
      Stdio.eprintf "wait at %.1f s bounded %g s: %s after %.1f s (not part of the golden)\n%!"
        (t -. !search_start) seconds
        (if completed then "completed" else "EXPIRED")
        took;
    completed
end

module Recorded = Autotune.Make (Recording_driver)

type run = {
  result : (float array, exn) Result.t;
  report : Autotune.report option;
  run_waits : wait list;
  took : float;
}

(* With [~run_routine:false] a routine the search returns is never run: over the never-ending
   reduction that run would be a wait no bound covers. *)
let search ~run_routine ~comp ~(output : Tensor.t) =
  waits := [];
  let report = ref None in
  let ctx = Context.auto () in
  search_start := Unix.gettimeofday ();
  let result =
    match
      Recorded.tune ~search:true ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:""
        ~report:(fun r -> report := Some r)
        ctx comp Idx.Empty
    with
    | _, _ when not run_routine -> Ok [||]
    | ctx, routine ->
        let ctx = Context.run ctx routine in
        let values = Context.get_values ctx output.Tensor.value in
        Context.release ctx;
        Ok values
    | exception exn -> Error exn
  in
  {
    result;
    report = !report;
    run_waits = List.rev !waits;
    took = Unix.gettimeofday () -. !search_start;
  }

let describe (r : run) =
  Stdio.eprintf "search ended after %.1f s, %d bounded waits: %s (not part of the golden)\n%!"
    r.took (List.length r.run_waits)
    (match r.result with Ok _ -> "returned a routine" | Error exn -> Exn.to_string exn);
  Option.iter r.report ~f:(fun rep ->
      List.iter rep.Autotune.declines ~f:(fun d ->
          Stdio.eprintf "decline %s x%d (not part of the golden)\n%!"
            (Sexp.to_string (SO.sexp_of_rejection_key d.Autotune.key))
            d.Autotune.count))

let died_at_a_candidate_sync (r : run) =
  match Option.map r.report ~f:(fun rep -> rep.Autotune.outcome) with
  | Some (Autotune.Search_died { phase = SO.Sync; candidate = Some c; _ }) ->
      not (String.equal c "baseline")
  | _ -> false

(* Whether [key] is the one candidate decline, counted once. On a GPU backend the serial baseline's
   [Not_dispatched_key] refusal (gh-ocannl-532) sits beside it in every report, so refusals before
   dispatch are not counted: the claim is about what a dispatched candidate failed with. *)
let one_dispatched_decline (declines : Autotune.decline_summary list) key =
  match
    List.filter declines ~f:(fun d ->
        match d.Autotune.key with SO.Not_dispatched_key _ -> false | _ -> true)
  with
  | [ d ] -> SO.equal_rejection_key d.Autotune.key key && d.Autotune.count = 1
  | _ -> false

let the_one_decline (r : run) key =
  one_dispatched_decline
    (Option.value_map r.report ~default:[] ~f:(fun rep -> rep.Autotune.declines))
    key

type hang = { run : run; killed : bool }

let hang =
  lazy
    (let comp, output = reduction ~label:"hang" ~outer:outer_hang in
     let recoveries = recovery_count () in
     Stdio.eprintf "GPU recoveries before the search: %d (not part of the golden)\n%!" recoveries;
     let run = search ~run_routine:false ~comp ~output in
     (* The count can trail a reported failure, as in metal_gpu_hang_failure: poll it briefly. *)
     let rec settle tries =
       let after = recovery_count () in
       if after > recoveries || tries = 0 then after
       else (
         Unix.sleepf 0.1;
         settle (tries - 1))
     in
     let after = settle 20 in
     Stdio.eprintf "GPU recoveries after the search: %d (not part of the golden)\n%!" after;
     describe run;
     { run; killed = after > recoveries })

let not_killed_leg =
  Verdict.sub_leg ~aggregation:`Environment leg
    ~when_:(lazy (not (Lazy.force hang).killed))
    ~on:"a macOS whose watchdog killed the never-ending kernel (gh-ocannl-2082)"

let killed_leg =
  Verdict.sub_leg ~aggregation:`Environment leg
    ~when_:(lazy (Lazy.force hang).killed)
    ~on:"a macOS whose watchdog did not kill the never-ending kernel (gh-ocannl-2082)"

let claim ?(leg = leg) label read = Verdict.in_leg ~leg label (fun label -> p label (read ()))

let () =
  (* The whole run's bound, armed before any device work: the control's searches and reads too. *)
  if String.equal backend_name "metal" then arm_watchdog ~seconds:360;
  (* --- The decline count's reading, over the census the 2026-11-18 window's search reported (the
     baseline's refusal beside the candidate's decline) and over the baseline's refusal alone. *)
  let decline key = { Autotune.key; count = 1; sample_details = [] } in
  let wait_bound_key = SO.Backend_rejected_key ("metal", SO.wait_bound_stage, SO.Expected) in
  let baseline = decline (SO.Not_dispatched_key "baseline") in
  p "the candidate's decline is read past the baseline's refusal before dispatch"
    (one_dispatched_decline [ baseline; decline wait_bound_key ] wait_bound_key);
  p "the baseline's refusal alone is no candidate decline"
    (not (one_dispatched_decline [ baseline ] wait_bound_key));
  (* --- The control: the same reduction, finite, searched to completion. *)
  let control =
    lazy
      (let comp, output = reduction ~label:"control" ~outer:2 in
       let r = search ~run_routine:true ~comp ~output in
       describe r;
       r)
  in
  Verdict.in_leg ~leg "the finite search timed a candidate through a completed bounded wait"
    (fun label -> p_exists label (Lazy.force control).run_waits ~f:(fun w -> w.completed));
  (* Within a relative 10^-4 either way, far closer than the neighbouring answers (each row's sum
     once, or 16 times) that a product space iterated otherwise would give. *)
  Verdict.in_leg ~leg "the finite search's routine computed four times every row's sum"
    (fun label ->
      let got = Result.ok (Lazy.force control).result |> Option.value ~default:[||] in
      p_close label got
        (Array.init m ~f:(fun o -> 4. *. row_sum o))
        ~tol:(fun want -> 1e-4 *. Float.abs want));
  if control_only then (
    claim "the never-ending reduction compiles (not run)" (fun () ->
        let comp, _ = reduction ~label:"compiled" ~outer:outer_hang in
        let ctx, _routine = Context.compile (Context.auto ()) comp Idx.Empty in
        ignore (ctx : Context.t);
        true);
    claim "the GPU recovery count reads, as the legs below are chosen on" (fun () ->
        let count = recovery_count () in
        Stdio.eprintf "GPU recoveries: %d (not part of the golden)\n%!" count;
        count >= 0))
  else
    let o () = Lazy.force hang in
    let wait_bound_key = SO.Backend_rejected_key ("metal", SO.wait_bound_stage, SO.Expected) in
    let hang_key = SO.Backend_rejected_key ("metal", MF.gpu_hang_stage, SO.Expected) in
    let expired () = List.filter (o ()).run.run_waits ~f:(fun w -> not w.completed) in
    claim "the search over the never-ending reduction raised" (fun () ->
        Result.is_error (o ()).run.result);
    claim "the search ended Search_died at a candidate's Sync" (fun () ->
        died_at_a_candidate_sync (o ()).run);
    claim ~leg:not_killed_leg "the one decline is the candidate's wait_bound, counted once"
      (fun () -> the_one_decline (o ()).run wait_bound_key);
    claim ~leg:not_killed_leg "exactly one bounded wait expired, the search's last" (fun () ->
        match (expired (), List.last (o ()).run.run_waits) with
        | [ w ], Some last -> phys_equal w last
        | _ -> false);
    claim ~leg:not_killed_leg "the expired wait was bounded at the 60 s floor" (fun () ->
        match expired () with [ w ] -> Float.(w.bound = Autotune.wait_bound_floor_s) | _ -> false);
    claim ~leg:not_killed_leg "the expired wait lasted its bound, and under 30 s more" (fun () ->
        match expired () with
        | [ w ] -> Float.(w.took >= w.bound && w.took < w.bound +. 30.)
        | _ -> false);
    claim ~leg:not_killed_leg
      "the search raised the decline's rendering, naming the host-side bound" (fun () ->
        match (o ()).run.result with
        | Error (Invalid_argument m) -> String.is_substring m ~substring:"host-side bound"
        | Ok _ | Error _ -> false);
    claim ~leg:killed_leg "the one decline is the candidate's gpu_hang, counted once" (fun () ->
        the_one_decline (o ()).run hang_key)
