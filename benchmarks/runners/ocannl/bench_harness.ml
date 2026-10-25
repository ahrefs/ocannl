(* Shared scaffolding for the OCANNL benchmark runners: fixture metadata access, weight injection
   into block-created params by debug-name tokens, the reduced-precision legs and training-step
   shapes, the measurement protocol (parity losses, warmup, per-step-synced percentiles, queued
   mean), and JSON emission. See benchmarks/README.md for the protocol. *)

open Base
open Ocannl
module St = Safetensors
module Tn = Ir.Tnode
module Asgns = Ir.Assignments

let get_meta st k = List.Assoc.find_exn (St.metadata st) ~equal:String.equal k
let meta_int st k = Int.of_string (get_meta st k)

let meta_default st k ~default =
  match List.Assoc.find (St.metadata st) ~equal:String.equal k with Some v -> v | None -> default

(** Whether the fixture describes a training workload ([mode: train], the generator's default) as
    opposed to a forward-only one ([mode: infer]). Every runner dispatches its step shape on this,
    the same way the Python runners do. *)
let is_training st = String.equal (meta_default st "mode" ~default:"train") "train"

(* Keys added after fixtures existed (e.g. stride1/stride2) default rather than fail, so
   pre-existing fixtures keep working. *)
let meta_int_default st k ~default =
  match List.Assoc.find (St.metadata st) ~equal:String.equal k with
  | Some v -> Int.of_string v
  | None -> default

let env_flag name = match Stdlib.Sys.getenv_opt name with Some "1" -> true | _ -> false

(** {1 Reduced-precision legs and training-step shapes (gh-ocannl-492 tasks 4 and 5)}

    These live here rather than in a runner because a flag implemented in {e one} runner is
    indistinguishable, from the report, from a cell nobody ran: [BENCH_STATIC_SCALE] and
    [BENCH_GATE_INTERVAL] existed in [bench_mlp] alone, which silently made the gate-cost experiment
    unavailable for every other workload — including [gpt2_mini], the matmul-dominated one where
    reduced precision matters most (gh-ocannl-551). A runner now opts in by calling {!precision_leg}
    with whether its fixture is a training one, and every leg is either available or refused with a
    message naming why. *)

type precision_leg = {
  label : string;
      (** The report's [precision] field: [f32 | bf16 | f16 | f16-static | f16-gatedN]. *)
  base : string;  (** [f32 | bf16 | f16] — the storage precision, without the gate-leg suffix. *)
  prec : Ir.Ops.prec option;  (** [None] at f32. *)
  static_scale : bool;  (** f16 with a fixed scale: no gate, no host read. *)
  gate_interval : int option;  (** f16 with the fused on-device gate, sampled every N steps. *)
  init_scale : float;
      (** The f16 loss scale to start from — the fixture's [loss_scale] metadata (torch's 65536 when
          absent), overridable with [BENCH_LOSS_SCALE]. It is a workload property: a scale that
          overflows on the first step costs the dynamic legs a few backoff steps (which the parity
          window then sees) and makes the static leg — which never adjusts — diverge outright. *)
}

(** Parses [BENCH_PRECISION] / [BENCH_STATIC_SCALE] / [BENCH_GATE_INTERVAL]. [runner] prefixes error
    messages; [training] is {!is_training} of the fixture — the gate legs measure the cost of the
    loss-scaling gate, which only a training step has, so on a forward-only fixture they are refused
    rather than silently ignored. *)
let precision_leg ~runner ~training ?st () =
  let base, prec =
    match Stdlib.Sys.getenv_opt "BENCH_PRECISION" with
    | None | Some "" | Some "0" | Some "f32" -> ("f32", None)
    | Some "bf16" -> ("bf16", Some Ir.Ops.bfloat16)
    | Some "f16" -> ("f16", Some Ir.Ops.half)
    | Some other -> failwith (runner ^ ": unknown BENCH_PRECISION: " ^ other)
  in
  let static_scale = env_flag "BENCH_STATIC_SCALE" in
  let gate_interval =
    match Stdlib.Sys.getenv_opt "BENCH_GATE_INTERVAL" with
    | None | Some "" | Some "0" -> None
    | Some n -> Some (Int.of_string n)
  in
  Option.iter gate_interval ~f:(fun n ->
      if n <= 0 then failwith (runner ^ ": BENCH_GATE_INTERVAL must be a positive integer"));
  let gate_leg = static_scale || Option.is_some gate_interval in
  if gate_leg && not (String.equal base "f16") then
    failwith (runner ^ ": BENCH_STATIC_SCALE / BENCH_GATE_INTERVAL require BENCH_PRECISION=f16");
  if static_scale && Option.is_some gate_interval then
    failwith (runner ^ ": BENCH_STATIC_SCALE and BENCH_GATE_INTERVAL are mutually exclusive");
  if gate_leg && not training then
    failwith
      (runner
     ^ ": BENCH_STATIC_SCALE / BENCH_GATE_INTERVAL measure the loss-scaling gate, which only a \
        training step has; this fixture is forward-only (metadata mode=infer)");
  let label =
    if static_scale then base ^ "-static"
    else match gate_interval with Some n -> Printf.sprintf "%s-gated%d" base n | None -> base
  in
  let init_scale =
    match Stdlib.Sys.getenv_opt "BENCH_LOSS_SCALE" with
    | Some s when not (String.is_empty s) -> Float.of_string s
    | _ -> (
        match st with
        | Some st -> Float.of_string (meta_default st "loss_scale" ~default:"65536.")
        | None -> 65536.)
  in
  if Float.(init_scale <= 0.) then failwith (runner ^ ": loss scale must be positive");
  { label; base; prec; static_scale; gate_interval; init_scale }

(** The report's label of a storage precision: [f32], [bf16], [f16], [f64], else
    {!Ir.Ops.prec_string}. *)
let precision_label prec =
  match prec with
  | Ir.Ops.Single_prec _ -> "f32"
  | Ir.Ops.Bfloat16_prec _ -> "bf16"
  | Ir.Ops.Half_prec _ -> "f16"
  | Ir.Ops.Double_prec _ -> "f64"
  | _ -> Ir.Ops.prec_string prec

(** The storage precision [t]'s value node was compiled at, as {!precision_label} spells it — the
    result line's [storage_precision] (gh-ocannl-1262). A runner hands {!measure_and_emit} its model
    body's output (the logits): every precision leg's storage policy re-precisions that subtree and
    pins only the loss head, so a [BENCH_PRECISION] the runner never applied reads back as the
    precision the cell did run at, where echoing the leg it parsed would agree with the request
    whatever ran. Read once the graph is compiled, when the precision is settled. *)
let storage_precision (t : Tensor.t) = precision_label (Lazy.force t.Tensor.value.Tn.storage_prec)

(** {1 The dominant-kernel instrument's deadline (gh-ocannl-1260)} *)

exception Instrument_over_budget

(** Whether {!with_deadline} can stop work in progress on this platform: POSIX's interval timer
    delivers [SIGALRM], whose handler raises at the next point the runtime polls -- in a sleep, a
    [waitpid] on a compiler, an OCaml loop; not inside a long foreign call, which it interrupts only
    once that call returns. Windows has no interval timer, and there the instrument stops only
    between kernels. *)
let instrument_preemptible = not Stdlib.Sys.win32

(** [with_deadline ~seconds f] runs [f], raising {!Instrument_over_budget} out of it once [seconds]
    have passed (where {!instrument_preemptible}). The raise is one-shot and is disarmed the moment
    [f] returns, so it cannot land after [f] did.

    The process may already hold an interval timer: a driver's cap set as
    [perl -e 'alarm N; exec ...'] survives the [exec], and its [SIGALRM] at the default disposition
    is what ends a cell that outlives the cap. So that timer is read first and given back
    afterwards, with its own disposition and the time it has left: the deadline here is the earlier
    of the two, and a cap that came due meanwhile is re-armed to fire at once, so the cell still
    ends at its cap rather than running on past it with no alarm left. *)
let with_deadline ~seconds f =
  if not instrument_preemptible then f ()
  else
    let set value interval =
      Unix.setitimer Unix.ITIMER_REAL { Unix.it_interval = interval; it_value = value }
    in
    let started = Unix.gettimeofday () in
    let outer = Unix.getitimer Unix.ITIMER_REAL in
    let outer_left = outer.Unix.it_value in
    let deadline = if Float.(outer_left > 0.) then Float.min seconds outer_left else seconds in
    let armed = ref true in
    let previous =
      Stdlib.Sys.signal Stdlib.Sys.sigalrm
        (Stdlib.Sys.Signal_handle
           (fun _ ->
             if !armed then (
               armed := false;
               raise Instrument_over_budget)))
    in
    let outcome =
      try
        ignore (set (Float.max deadline 0.001) 0. : Unix.interval_timer_status);
        let v = f () in
        armed := false;
        Ok v
      with e ->
        armed := false;
        Error e
    in
    ignore (set 0. 0. : Unix.interval_timer_status);
    Stdlib.Sys.set_signal Stdlib.Sys.sigalrm previous;
    (if Float.(outer_left > 0.) then
       let left = outer_left -. (Unix.gettimeofday () -. started) in
       ignore (set (Float.max left 0.001) outer.Unix.it_interval : Unix.interval_timer_status));
    match outcome with Ok v -> v | Error e -> raise e

(** The step shapes of the training legs, before compilation. *)
type train_parts =
  | Plain_step of Asgns.comp  (** One fused routine: f32, bf16, and the f16 static-scale leg. *)
  | Host_gated of Mixed_prec.Loss_scaler.t * Tensor.t * Asgns.comp * Asgns.comp
      (** Gradient routine, host-read checksum gate, optimizer routine (the f16 default). *)
  | Device_gated of Mixed_prec.Loss_scaler.t * Tensor.t * Asgns.comp * int
      (** One routine with the gate on device; the host samples the window checksum every N. *)

(** The step shape of [leg]. f16 without a gate-leg flag is the dynamic host-read gate (its per-step
    device sync is part of what the leg measures); [no_sgd] builds the gradient update alone
    (f32/bf16 only — a debugging shape, not a comparable cell). *)
let train_step_parts ?(setup_for_parallel = false) ?(no_sgd = false) ~leg ~learning_rate loss =
  match leg with
  | { base = "f16"; static_scale = false; gate_interval = Some interval; _ } ->
      let scaler = Mixed_prec.Loss_scaler.create ~init_scale:leg.init_scale () in
      let wflag, comp =
        Mixed_prec.gated_scaled_update ~setup_for_parallel scaler ~learning_rate loss
      in
      Device_gated (scaler, wflag, comp, interval)
  | { base = "f16"; static_scale = false; _ } ->
      let scaler = Mixed_prec.Loss_scaler.create ~init_scale:leg.init_scale () in
      let checksum, grad_comp = Mixed_prec.scaled_grad_update ~setup_for_parallel scaler loss in
      let sgd_comp = Mixed_prec.scaled_sgd_update scaler ~learning_rate loss in
      Host_gated (scaler, checksum, grad_comp, sgd_comp)
  | { base = "f16"; static_scale = true; _ } ->
      (* Scaled backprop and unscaled optimizer as ONE routine, no checksum, no gate, no host read —
         the scale scalars are set once and never adjusted. *)
      let scaler = Mixed_prec.Loss_scaler.create ~init_scale:leg.init_scale () in
      Plain_step
        (Asgns.sequence
           [
             Train.grad_update ~setup_for_parallel ~loss_scale:scaler.Mixed_prec.Loss_scaler.scale
               loss;
             Train.sgd_update ~learning_rate ~grad_unscale:scaler.Mixed_prec.Loss_scaler.unscale
               loss;
           ])
  | _ ->
      let update = Train.grad_update ~setup_for_parallel loss in
      Plain_step
        (if no_sgd then update else Asgns.sequence [ update; Train.sgd_update ~learning_rate loss ])

(** The compiled counterpart of {!train_parts}. A forward-only runner reuses [Plain] for its forward
    routine, so one step driver ({!run_train_step}) serves both modes. *)
type train_routines =
  | Plain of Context.routine
  | Host_gate of Mixed_prec.Loss_scaler.t * Tensor.t * Context.routine * Context.routine
  | Device_gate of Mixed_prec.Loss_scaler.t * Tensor.t * Context.routine * int

type step_shape = [ `Train of train_parts | `Forward of Asgns.comp ]

let compile_default ctx comp bindings = Autotune.untuned_compile ctx comp bindings

(** Compiles a step shape. [tuned] is the runner's autotuning compile (it needs the loss tensor, so
    the runner supplies it). Each leg tunes the routine that carries the work: the
    dynamic-loss-scaling legs keep their step SHAPE — the gate is what they measure — so only the
    gradient/fused routine is tuned and the tiny optimizer routine is compiled plainly, from the
    tuned context so the lineage's compile order is unchanged. The untuned arm is uniform by
    contrast: every routine of the step goes through the model gate when it is enabled. *)
let compile_train_step ~tune ~tuned ctx bindings parts =
  (* gh-ocannl-491: the model-picked untuned default (config [model_default_schedule=true]) — run
     the same benchmark with the gate off vs on for the before/after comparison. Every leg's
     work-carrying routine takes this path when it is not tuned, so the gated (f16) legs compare a
     real model pick against the plain default rather than two identical executions. *)
  let untuned ctx comp = compile_default ctx comp bindings in
  match parts with
  | Plain_step comp ->
      let ctx, routine = if tune then tuned ctx comp else untuned ctx comp in
      (ctx, Plain routine)
  | Host_gated (scaler, checksum, grad_comp, sgd_comp) ->
      let ctx, grad_routine = if tune then tuned ctx grad_comp else untuned ctx grad_comp in
      (* The optimizer routine is not TUNED — timing it is not what this leg measures, and its plain
         compile from the tuned context keeps the lineage's compile order unchanged — but it does
         take the untuned arm's model gate: the leg's reported step time is both routines, so a gate
         that skipped this one would understate its own treatment (and the config reference defines
         the gate over a runner's untuned arm, not over a chosen routine of it). *)
      let ctx, sgd_routine =
        if tune then Context.compile ctx sgd_comp bindings else untuned ctx sgd_comp
      in
      (ctx, Host_gate (scaler, checksum, grad_routine, sgd_routine))
  | Device_gated (scaler, wflag, comp, interval) ->
      let ctx, routine = if tune then tuned ctx comp else untuned ctx comp in
      (ctx, Device_gate (scaler, wflag, routine, interval))

(** Shared benchmark/diagnostic dispatch, including the model-default gate in both modes. *)
let compile_step ~tune ~tuned ctx bindings : step_shape -> Context.t * train_routines = function
  | `Train parts -> compile_train_step ~tune ~tuned ctx bindings parts
  | `Forward comp ->
      let ctx, routine = if tune then tuned ctx comp else compile_default ctx comp bindings in
      (ctx, Plain routine)

(** Every compiled routine, including the conditional optimizer before any step has run. The
    diagnostic inventories compiled work; {!step_routines} filters work observed in a benchmark. *)
let compiled_step_routines = function
  | Plain routine | Device_gate (_, _, routine, _) -> [ routine ]
  | Host_gate (_, _, grad_routine, sgd_routine) -> [ grad_routine; sgd_routine ]

let train_step_bindings = function
  | Plain routine -> routine.Context.bindings
  | Host_gate (_, _, grad_routine, _) -> grad_routine.Context.bindings
  | Device_gate (_, _, routine, _) -> routine.Context.bindings

(** How many times a host-gated step has launched its optimizer routine in this process — the gate
    skips it on a non-finite gradient checksum. {!step_routines} reads it against the count at the
    start of the timed window, so a kernel no measured step launched is not offered as the step's
    dominant one. *)
let host_gated_optimizer_runs = ref 0

(** [!host_gated_optimizer_runs] when the timed window opened; set by {!measure_and_emit}. *)
let optimizer_runs_at_window = ref 0

(** Runs one step of [routines] — a training step, or a forward pass when the runner compiled its
    forward code as [Plain]. The scaled legs thread the context (the scaler overwrites the scale
    tensors), hence the reference. [step] is 0-based. *)
let run_train_step routines ctx_ref ~step =
  match routines with
  | Plain routine -> Train.run !ctx_ref routine
  | Host_gate (scaler, checksum, grad_routine, sgd_routine) ->
      (* The step's launch parameters are assigned on [train_step_bindings] -- the gradient
         routine's -- and the optimizer routine, compiled with the same bindings, takes the same
         values: linking left its own unassigned (gh-ocannl-940). *)
      List.iter sgd_routine.Context.bindings ~f:(fun (sym, cell) ->
          cell := !(Ir.Indexing.find_exn grad_routine.Context.bindings sym));
      let ctx, ran = Mixed_prec.scaled_step ~scaler ~grad_routine ~sgd_routine ~checksum !ctx_ref in
      if ran then Int.incr host_gated_optimizer_runs;
      ctx_ref := ctx
  | Device_gate (scaler, wflag, routine, interval) ->
      let ctx, _window_finite =
        Mixed_prec.gated_step ~scaler ~routine ~window_checksum:wflag ~check_interval:interval ~step
          !ctx_ref
      in
      ctx_ref := ctx

(* The floating-point licences a cell compiled under (gh-ocannl-998): each key's resolved value and
   where it came from, so a report can tell an exact cell from one that ran licensed algebra. *)
let fp_licence_record () =
  List.map
    Ir.Low_level.Fp_licence.(fields (of_config ()))
    ~f:(fun (key, value) ->
      let source =
        match Utils.profile_key_source key with None -> Utils.From_default | Some (_, s) -> s
      in
      (key, value, Utils.config_source_label source))

let percentile sorted p =
  let n = Array.length sorted in
  let idx = Float.to_int (Float.round_nearest (p /. 100. *. Float.of_int (n - 1))) in
  sorted.(idx)

(** {1 Where a tuning session's wall goes (gh-ocannl-834)}

    [BENCH_TIMING_TRACE=1] splits a searching process's wall between candidate timing and everything
    else, without changing what is measured: it only observes {!Autotune}'s seams. Each
    [Autotune.time_routine] call reports one record as it returns
    ({!Autotune.install_timing_call_observer}, gh-ocannl-1263), printed as its calibration probes,
    one call line and one [decision] line -- everything about a call comes from that call's own
    record, so no line depends on the order in which separate seams fired. The call line carries the
    batch depth the call settled on, the launches it dispatched (the warmup, the calibration's and
    [depth] per timed batch the loop itself dispatched, a discarded contention window's included),
    and the wall of its warmup plus calibration (from the pre-dispatch validation to the depth
    decision) and of its timed loop. A call that settles at depth 1 resumes the calibration's
    synchronized singles as its window (gh-ocannl-1074): its line says how many of its batches were
    [reused] -- already dispatched, and counted, as calibration -- and its timed wall is only the
    top-up the caller's [repeats] floor asked for beyond them. Every line also carries the running
    totals, so the last line of a run killed at a cell cap is still a lower bound on the whole — an
    [at_exit] summary would never print there. A call made without failure tagging — the untuned
    control [autotune_log=true] times after a search — is marked [untagged] and counted apart.

    Each candidate the tuner attempts prints an [attempt] line with the elapsed wall, which is how
    far a killed search got; an attempt is a compile unless [autotune_bound_pruning] prunes it
    first. Each search's [baseline] is an attempt too: it fires the attempt seam inside its
    compile's transform, which on a GPU backend refuses it there, before codegen, since it would
    bind no hardware dimension (gh-ocannl-532, gh-ocannl-1340), and [autotune_progress]'s [attempts]
    does not count it, so this count runs one per search ahead of that log's. Every compile a search
    makes -- the baseline's, each candidate's, a replay's, the winner's -- prints a [compile] line
    as it returns ({!Autotune.install_compile_observer}, gh-ocannl-1314): its stage, label and
    verdict, its wall and its phases (lowering, transform, codegen, the vendor compiler, link) and
    how many of its kernels the vendor compiler was given, which is fewer than it rendered where the
    backend reused compiled kernels. A tagged call that raises after its pre-dispatch validation (a
    classified launch or synchronization failure the search survives) returns no record: the trace
    counts it as raised -- validations started less tagged records returned -- and a summary with
    any says INCOMPLETE, since that call's cost is in no total.

    The [decision] line carries the policy's own metadata (gh-ocannl-1199): the search phase, the
    exit that chose the depth, the fitted fixed and marginal costs and the fit's verdict, what the
    post-calibration bounds changed, the probe budget used and the admission verdict, so diagnosing
    a refusal needs no parser that reimplements the calibration; [results.jsonl] carries the same
    settle and admission per arm as counts ({!Autotune.report}'s [timing_decisions]). What is not
    counted: the cc backend's in-kernel fork/joins per launch are a property of each candidate's
    rendering, so a launch count bounds them only together with the candidate's parallel-region
    count. *)

(** {1 A search cut short (gh-ocannl-1388)}

    [BENCH_TUNE_CUT_AFTER=N] keeps the readings of each search's first [N] admitted timing windows
    and refuses every later one, through {!Autotune.on_candidate_measured}: a refused reading counts
    as a contended window, so the search goes on compiling and attempting its candidates but crowns
    among the first [N] timed. That is the search a budget or a contended box cuts short, and with
    [N = 1] it shows the floor a search's first candidate sets: on CUDA and HIP the seeded untuned
    plan, which the tuner seeds first. A search starts at its [baseline] attempt, which resets the
    count. The cut is visible in the result line: each [tune] arm's [timings_contended] counts every
    window after the first [N]. *)
let install_search_cut () =
  match Stdlib.Sys.getenv_opt "BENCH_TUNE_CUT_AFTER" with
  | None -> ()
  | Some raw ->
      let n =
        match Int.of_string_opt (String.strip raw) with
        | Some n when n >= 1 -> n
        | _ -> failwith ("BENCH_TUNE_CUT_AFTER must be a positive integer; found: " ^ raw)
      in
      Stdio.eprintf "bench: every search crowns among its first %d admitted timing(s)\n%!" n;
      let admitted = ref 0 in
      let prev_attempt = Autotune.Seam.get Autotune.on_candidate_attempt in
      ( Autotune.Seam.set Autotune.on_candidate_attempt @@ fun label ->
        if String.equal label "baseline" then admitted := 0;
        prev_attempt label );
      let prev_measured = Autotune.Seam.get Autotune.on_candidate_measured in
      Autotune.Seam.set Autotune.on_candidate_measured @@ fun ~label ~digest ms ->
      let ms = prev_measured ~label ~digest ms in
      Int.incr admitted;
      if !admitted <= n then ms else Float.nan

let timing_trace_on = ref false

(** Marks the end of the measured compile (the search, when tuning) on a traced run, so a cell cut
    off after it -- in the fixture reinjection or the timed steps -- is not read as a search that
    was still going. *)
let trace_search_done ~compile_s =
  if !timing_trace_on then
    Stdlib.Printf.eprintf "timing-trace: search done: compile_s %.3f\n%!" compile_s

(** The [decision] line of a traced timing call (gh-ocannl-1199): the decision behind the call line
    printed just before it, from the policy's own metadata ({!Autotune.timing_call}'s [decision]) --
    the search phase, what chose the depth, the fit it rested on, what bounded it, the probe budget
    used and the admission verdict. Pure, so a test pins it against the decision taken. *)
let decision_line ~call (d : Autotune.batch_decision) =
  let ms = Printf.sprintf "%.6f" in
  let opt o ~f = Option.value_map o ~default:"-" ~f in
  let fit =
    opt d.fit ~f:(fun (f : Autotune.affine_fit) ->
        let verdict =
          match f.verdict with
          | Fit_unordered -> "unordered"
          | Fit_unreadable -> "unreadable"
          | Fit_nonpositive_marginal -> "nonpositive_marginal"
          | Fit_negative_fixed -> "negative_fixed"
          | Fit_fixed_dominated -> "fixed_dominated"
          | Fit_base_crossing -> "base_crossing"
          | Fit_boundary_floor -> "boundary_floor"
          | Fit_shallower_crossing -> "shallower_crossing"
          | Fit_deeper_crossing -> "deeper_crossing"
        in
        Printf.sprintf "%s (depths %d/%d, fixed %s ms, marginal %s ms)" verdict f.base_depth
          f.probe_depth (ms f.fixed_ms) (ms f.marginal_ms))
  in
  Printf.sprintf
    "timing-trace: decision %d: phase %s, %s, depth %d by %s (cap %d, target %.1f ms, estimated \
     wall %s ms), fit %s, %d cap-directed probes, bounded from %s, fallback from %s, rescue %s, \
     probes %d of %d (%.3f of %.0f ms%s), %s%s%s"
    call (opt d.phase ~f:Fn.id) (Autotune.timing_string d.timing) d.depth
    (Autotune.batch_settle_name d.settle)
    d.cap d.target_ms (opt d.estimated_wall_ms ~f:ms) fit d.cap_directed_probes
    (opt d.bounded_from ~f:Int.to_string)
    (opt d.fallback_from ~f:Int.to_string)
    (opt d.rescue ~f:(fun (depth, min_ms) -> Printf.sprintf "depth %d min %s ms" depth (ms min_ms)))
    d.probes Autotune.queue_calibration_max_probes d.probe_wall_ms
    Autotune.queue_calibration_wall_ms
    (if d.budget_spent then ", spent" else "")
    (Autotune.batch_admission_name d.admission)
    (if d.retried then " after a retry" else "")
    (* gh-ocannl-1558: the co-tenant verdict a CPU backend's window was attributed by; absent under
       the dispersion-only policy, so a GPU line reads as before. *)
    (match d.host_load with
    | None -> ""
    | Some (Quiet cores) -> Printf.sprintf ", host load quiet (%.2f foreign cores)" cores
    | Some (Cotenant cores) -> Printf.sprintf ", host load co-tenant (%.2f foreign cores)" cores
    | Some Unavailable -> ", host load unavailable")

let install_timing_trace () =
  if env_flag "BENCH_TIMING_TRACE" then begin
    timing_trace_on := true;
    (* Monotonic, like every timing path in this harness: a search runs for tens of minutes, and a
       wall-clock step inside it would skew every interval the trace reports. *)
    let now () = Mtime.Span.to_float_ns (Mtime_clock.elapsed ()) /. 1e9 in
    let t0 = now () in
    let calls = ref 0 and attempts = ref 0 and launches = ref 0 and untagged = ref 0 in
    (* Calls whose window resumed the calibration's singles, and the batches those windows
       reused. *)
    let reusing_calls = ref 0 and reused_batches = ref 0 in
    (* Pre-dispatch validations started, and the tagged calls that returned a record. A tagged call
       that raised after its validation returns none, so the difference counts exactly those calls,
       whose warmup, calibration and any timed batches are in no total -- by count, so no order in
       which the seams fire can move it. Both counts are this domain's -- the records reach only
       this domain's observer, and the preflight seam is per domain too ({!Autotune.Seam}) -- so a
       search in another domain moves neither. *)
    let preflights = ref 0 and tagged = ref 0 in
    let raised () = !preflights - !tagged in
    let calib_s = ref 0. and timed_s = ref 0. in
    let depths = Hashtbl.create (module Int) in
    let pr fmt = Stdlib.Printf.kfprintf Stdlib.flush Stdlib.stderr fmt in
    let prev_attempt = Autotune.Seam.get Autotune.on_candidate_attempt in
    ( Autotune.Seam.set Autotune.on_candidate_attempt @@ fun label ->
      Int.incr attempts;
      pr "timing-trace: attempt %d at %.1fs: %s\n" !attempts (now () -. t0) label;
      prev_attempt label );
    let compiles = ref 0 and compile_s = ref 0. and device_s = ref 0. in
    let kernels = ref 0 and kernels_compiled = ref 0 in
    Autotune.install_compile_observer (fun (r : Autotune.compile_record) ->
        let c = r.compile_cost in
        Int.incr compiles;
        compile_s := !compile_s +. r.compile_wall_s;
        device_s := !device_s +. Option.value c.device_s ~default:0.;
        kernels := !kernels + c.kernels;
        kernels_compiled := !kernels_compiled + c.kernels_compiled;
        pr
          "timing-trace: compile %d at %.1fs: %s %S %s, %.3f s (lower %.3f, transform %.3f, \
           codegen %.3f, device %s, link %.3f), %d of %d kernels compiled | totals: %d compiles, \
           %.2f s, device %.2f s, %d of %d kernels compiled\n"
          !compiles
          (now () -. t0)
          r.compile_stage r.compile_attempt r.compile_verdict r.compile_wall_s c.lower_s
          c.transform_s (Ir.Compile_cost.codegen_s c)
          (Option.value_map c.device_s ~default:"none" ~f:(Printf.sprintf "%.3f"))
          c.link_s c.kernels_compiled c.kernels !compiles !compile_s !device_s !kernels_compiled
          !kernels);
    let prev_preflight = Autotune.Seam.get Autotune.on_candidate_preflight in
    ( Autotune.Seam.set Autotune.on_candidate_preflight @@ fun name ->
      Int.incr preflights;
      prev_preflight name );
    Autotune.install_timing_call_observer (fun (c : Autotune.timing_call) ->
        List.iter c.calibration_probes ~f:(fun (probe : Autotune.calibration_probe) ->
            let role =
              match probe.role with
              | Provisional_probe -> "provisional"
              | Validation_probe -> "validation"
              | Confirmation_probe -> "confirmation"
              | Stall_retry_probe -> "stall_retry"
              | Crossing_probe -> "crossing"
              | Rescue_probe -> "rescue"
            in
            pr
              "timing-trace: calibration probe: role %s, depth %d, %d batches, min %.6f ms, wall \
               %.6f ms\n"
              role probe.depth probe.runs probe.min_ms probe.wall_ms);
        let depth = c.decision.depth in
        let dispatched (w : Autotune.timed_window) = depth * (w.samples - w.reused) in
        (* The warmup, the calibration, and every batch the timed loop dispatched itself -- a
           discarded contention window's included. *)
        let n =
          1 + c.calibration_launches + dispatched c.window
          + Option.value_map c.discarded_window ~default:0 ~f:dispatched
        in
        Int.incr calls;
        if c.tagged then Int.incr tagged else Int.incr untagged;
        if c.window.reused > 0 then Int.incr reusing_calls;
        reused_batches := !reused_batches + c.window.reused;
        launches := !launches + n;
        calib_s := !calib_s +. (c.setup_ms /. 1e3);
        timed_s := !timed_s +. (c.timed_ms /. 1e3);
        Hashtbl.update depths depth ~f:(fun k -> 1 + Option.value k ~default:0);
        pr
          "timing-trace: call %d at %.1fs: depth %d, %d batches (%d reused)%s, %d launches, calib \
           %.1f ms%s, timed %.1f ms (median batch %.3f ms, spread %.2fx) | totals: %d calls, %d \
           launches, calib %.2f s (%d calls untagged), timed %.2f s, %d calls raised, %d calls \
           reused %d batches\n"
          !calls
          (now () -. t0)
          depth c.window.samples c.window.reused
          (Option.value_map c.discarded_window ~default:"" ~f:(fun w ->
               Printf.sprintf " after a discarded window of %d" w.samples))
          n c.setup_ms
          (if c.tagged then "" else " (untagged call)")
          c.timed_ms c.window.median_wall_ms c.window.spread !calls !launches !calib_s !untagged
          !timed_s (raised ()) !reusing_calls !reused_batches;
        pr "%s\n" (decision_line ~call:!calls c.decision));
    Stdlib.at_exit (fun () ->
        let hist =
          Hashtbl.to_alist depths
          |> List.sort ~compare:(fun (a, _) (b, _) -> Int.compare a b)
          |> List.map ~f:(fun (d, k) -> Printf.sprintf "%dx%d" k d)
          |> String.concat ~sep:" "
        in
        pr
          "timing-trace: summary: %.1fs wall, %d candidate attempts, %d timing calls, %d launches, \
           calib %.2f s (%d calls untagged), timed %.2f s; depth histogram (calls x depth): %s; %d \
           calls reused %d calibration batches as their window%s\n"
          (now () -. t0)
          !attempts !calls !launches !calib_s !untagged !timed_s hist !reusing_calls !reused_batches
          (if raised () = 0 then ""
           else
             Printf.sprintf
               "; INCOMPLETE: %d tagged timing calls raised after validation, and their cost is in \
                none of these totals"
               (raised ())))
  end

(** {1 Placement A/B arms in the emitted result (gh-ocannl-546)}

    A per-arm search outcome that never reaches the result line is invisible in every end-to-end
    number the sweep reports: a tensorized candidate can win its arm and then be discarded whole
    when the other arm ships, and the only trace is an [OCANNL_AUTOTUNE_LOG] stderr stream that a
    successful cell throws away. So each arm's crowned candidate — its label, whether it tensorizes,
    and how the best {e timed} tensorized candidate compares — is collected here and emitted with
    the measurement, where `results.jsonl` keeps it.

    {!Train.tune_placements} calls [report] once per arm, arm A (default placements) first. Arms are
    named in arrival order, so one collector describes one placement A/B — every step shape in
    {!compile_train_step} tunes exactly one routine.

    Which arm {e shipped} is recorded from {!Train.tune_placements}' own [on_ship] callback
    (gh-ocannl-638) rather than re-derived from the arms' [best_ms]. The derivation was only ever
    valid while nothing could override the timing comparison — config [tune_ship_arm] now can, and
    under it the derived answer would name the arm the search preferred while the result line's
    losses came from the other one, which is the single fact a forced-arm measurement is run to
    establish. It also never described a flip-refined result (["flip"]), which is not an arm at all.
    The derivation is kept as the fallback for a caller that reports arms without wiring [on_ship].
*)

type tune_arms = {
  mutable arm_reports : Autotune.report list; (* reverse order *)
  mutable shipped : string option;
  mutable placement_replay : bool option;
  mutable shipped_digest : string option;
      (** The rest of [on_ship]'s record (gh-ocannl-1438): whether the placement was replayed from
          the placement-decision store, and the shipped search's [source_digest]. *)
  mutable ship_substituted : bool option;
  mutable substituted_from : string option;
      (** Whether no measured schedule shipped ([Train.shipped]'s [ship_substituted],
          gh-ocannl-1559), and the [best_label] of the crown the substitute replaced -- [None] when
          nothing was substituted, or on the last resort, which replaced no search's crown. *)
  mutable searches : int;
  mutable replays : int;
  mutable no_searches : int;
  mutable shipped_mma : Ir.C_syntax.mma_summary option;
      (** The census of the routine(s) this cell actually times (gh-ocannl-626), recorded by
          {!collect_shipped}. Kept apart from the arms' because the crowned candidate of an arm is
          not always the shipped artifact — see {!collect_shipped}. *)
}

let tune_arms () =
  {
    arm_reports = [];
    shipped = None;
    placement_replay = None;
    shipped_digest = None;
    ship_substituted = None;
    substituted_from = None;
    searches = 0;
    replays = 0;
    no_searches = 0;
    shipped_mma = None;
  }

(** Counts one reported search by its provenance (gh-ocannl-644). Kept separate from {!collect_arm}
    because a search this process ran is a search whether or not it was one of the placement arms:
    {!Train.tune_placements}' flip refinements report through its [flip_report] and must {e not}
    enter [arm_reports] (their arrival order would misname the arms in {!tune_json}), yet a flip
    search loads this process with accumulated modules and buffers exactly like an arm search does —
    which is the whole reason the two-pass protocol exists. A caller that runs any search whose
    outcome it does not otherwise collect should still route it here.

    A report is counted by matching its [Autotune.outcome] (gh-ocannl-677), which is what makes the
    third bucket visible: under [autotune_search=false] — the reproducible profile — and on a
    pre-search failure, a call neither searches nor replays, having shipped the untuned default.
    Such an arm must not be counted as a search: it would fail the sweep's provenance gate on both
    passes of a tuned cell, and there is nothing wrong with either. Nor as a replay: it carries no
    tuned artifact to credit the row with. Counting it explicitly is what lets the sweep state the
    third case instead of recovering it from two counters that are both zero. *)
let collect_search t (r : Autotune.report) =
  match r.Autotune.outcome with
  | Autotune.Searched | Autotune.Search_died _ | Autotune.Abandoned _ ->
      t.searches <- t.searches + 1
  | Autotune.Cache_replay | Autotune.Abandonment_replay _ -> t.replays <- t.replays + 1
  | Autotune.Search_disabled | Autotune.Pre_search_failure _ | Autotune.Lineage_refused _ ->
      t.no_searches <- t.no_searches + 1

let collect_arm t (r : Autotune.report) =
  collect_search t r;
  t.arm_reports <- r :: t.arm_reports

let collect_ship t (s : Train.shipped) =
  t.shipped <- Some s.Train.ship_arm;
  t.placement_replay <- Some s.Train.ship_replayed;
  t.shipped_digest <-
    Option.bind s.Train.ship_report ~f:(fun r ->
        Option.some_if (not (String.is_empty r.Autotune.source_digest)) r.Autotune.source_digest);
  t.ship_substituted <- Some s.Train.ship_substituted;
  t.substituted_from <-
    (if s.Train.ship_substituted then
       Option.bind s.Train.ship_report ~f:(fun r ->
           Option.some_if (not (String.is_empty r.Autotune.best_label)) r.Autotune.best_label)
     else None)

(** Census of all compiled step routines, including a host-gated optimizer before execution. Unlike
    the dominant-kernel filter, this describes compiled work, not which gates passed. *)
let step_census routines =
  Ir.C_syntax.merge_mma_summaries
    (List.map (compiled_step_routines routines) ~f:(fun r -> r.Context.mma))

let mma_wire (m : Ir.C_syntax.mma_summary) =
  ( Ir.C_syntax.tensorization_name m.Ir.C_syntax.tensorization,
    m.Ir.C_syntax.statements,
    m.Ir.C_syntax.scalar_fallbacks )

(** Records what the artifact this cell TIMES actually emitted (gh-ocannl-626). Every runner calls
    it with the routines it goes on to step, right after compiling them.

    This is deliberately not derived from the arm reports, and overrides them downstream, because a
    crowned arm candidate is not always the shipped artifact. Two ways they come apart, both live: a
    gh-555 flip refinement that beats the A/B winner ships under [on_ship "flip"] and is not an arm
    at all (flip reports stay out of [arm_reports] — their arrival order would misname the arms), so
    an arm lookup finds nothing and the cell would report no census; and on the [timing_ctx] path
    {!Autotune.tune} recompiles the winner in the production context and falls back to the untuned
    default when that replay is rejected or lands unparallelized, so the arm describes a schedule
    that was discarded. In both cases the arm's label can claim tensorized over a routine that
    emitted no mma, which is the exact failure this whole field exists to prevent — so the fact is
    taken from the compiled routine, which cannot be wrong about itself. *)
let collect_shipped t routines = t.shipped_mma <- Some (step_census routines)

(** Whether this process ran a schedule search rather than replaying cached winners throughout — the
    [searched] field of the result line. See {!measure_and_emit}. *)
let searched t = t.searches > 0

(** A report's settle-by-admission counts (gh-ocannl-1263) as {!Bench_json.tune_arm} takes them, in
    Autotune's own spellings. *)
let timing_decisions_wire decisions =
  List.map decisions ~f:(fun (settle, admission, calls) ->
      (Autotune.batch_settle_name settle, Autotune.batch_admission_name admission, calls))

(** The [tune] JSON object, or [None] when no arm reported (an untuned cell). Times are
    milliseconds, and a time that was never measured is [null], not [inf]: [best_ms] is [infinity]
    when an arm timed nothing at all (every candidate failed and the GPU baseline was not
    dispatched) and [mma_best_ms] when it timed no tensorized candidate. Those are exactly the runs
    whose evidence this object exists to preserve, so they must not be the runs whose result line
    fails to parse.

    Every time here is a reading of the tuner's configured {!Autotune.timing_mode}, and nothing in
    the line records which one (gh-ocannl-755). Under the default [queued] they are per-launch
    steady-state times and comparable with the step timings beside them; under
    [autotune_timing= isolated] each carries one host submit/sync round trip, which on a sub-100-us
    kernel is tens of percent to 2x, varying per candidate. So a [best_ms] is only comparable across
    result lines taken under the same setting, and never against [step_ms] or [queued_step_ms]
    unless it was queued.

    An arm that terminated on a failure carries [terminal_failure] and is {e never} the shipped one,
    whatever its pre-failure [best_ms] says (gh-ocannl-550): the search raised, so no routine was
    compiled from it — [Train.tune_placements] ranks it at [infinity] and this attribution follows
    the same rule rather than re-deriving a winner from times alone.

    Each arm also records its outcome [state] — the {!Autotune.outcome_name} of what it did about
    searching — and the object totals the three provenance buckets over every search this process
    reported ([searches] / [replays] / [no_searches], flip refinements included). That is the
    per-arm detail behind the result line's [searched] field (gh-ocannl-644): a cell can be mixed —
    one arm cached, the other searched because its half of the A/B was never cached — and only the
    per-arm breakdown says which. The legacy [searched] and [cache_hit] booleans stay in the wire
    format for artifacts and readers that predate gh-ocannl-677; [state] is the one that names the
    outcome, [no_searches] the one that spares a reader deriving the third case from two zeros. *)
let tune_json t =
  match List.rev t.arm_reports with
  | [] -> None
  | reports ->
      (* Arms are named by arrival order on a cold run, which reports both. A placement-store replay
         (gh-ocannl-786) reports exactly one search -- the shipped placement's -- so a lone report
         is named by what [on_ship] said shipped rather than as an arm A it may not be. *)
      let named =
        match (reports, t.shipped) with
        | [ r ], Some what -> [ (what, r) ]
        | _ -> List.mapi reports ~f:(fun i r -> (Printf.sprintf "%c" (Char.of_int_exn (65 + i)), r))
      in
      let shipped =
        match t.shipped with
        | Some what -> what
        | None ->
            List.fold named ~init:None ~f:(fun acc (name, (r : Autotune.report)) ->
                if Option.is_some (Autotune.terminal_failure r) then acc
                else
                  match acc with
                  | Some (_, best) when Float.( <= ) best r.best_ms -> acc
                  | _ -> Some (name, r.best_ms))
            |> Option.value_map ~default:"?" ~f:fst
      in
      let arm (name, (r : Autotune.report)) =
        let searched, cache_hit =
          match r.Autotune.outcome with
          | Autotune.Searched | Autotune.Search_died _ | Autotune.Abandoned _ -> (true, false)
          | Autotune.Cache_replay | Autotune.Abandonment_replay _ -> (false, true)
          | Autotune.Search_disabled | Autotune.Pre_search_failure _ | Autotune.Lineage_refused _ ->
              (false, false)
        in
        Bench_json.tune_arm ~name
          ~state:(Autotune.outcome_name r.Autotune.outcome)
          ~searched ~cache_hit
          ~timing:(Autotune.timing_string r.Autotune.timing)
          ~rounds_run:r.Autotune.rounds_run ~beam_width:r.Autotune.beam_width
          ~budget_cut:r.Autotune.budget_cut ~budget_s:r.Autotune.budget_s
          ~call_budget_s:r.Autotune.call_budget_s ~call_share_s:r.Autotune.call_share_s
          ~timings_contended:r.Autotune.timings_contended
          ~timings_unbatched:r.Autotune.timings_unbatched
          ~timings_unreadable:r.Autotune.timings_unreadable
          ~timing_decisions:(timing_decisions_wire r.Autotune.timing_decisions)
          ~best_ms:r.Autotune.best_ms ~best_label:r.Autotune.best_label
          ~tensorized:r.Autotune.best_tensorized
          ~tensorization:
            (Option.map r.Autotune.best_tensorization ~f:Ir.C_syntax.tensorization_name)
          ~mma_statements:r.Autotune.best_mma_statements
          ~mma_scalar_fallbacks:r.Autotune.best_mma_scalar_fallbacks
          ~mma_seeded:r.Autotune.mma_candidates ~mma_timed:r.Autotune.mma_timed
          ~mma_best_ms:r.Autotune.mma_best_ms
          ~terminal_failure:
            (Option.map (Autotune.terminal_failure r) ~f:(fun tf -> tf.Autotune.detail))
      in
      Some
        (Bench_json.tune_object ~shipped ?placement_replay:t.placement_replay
           ?shipped_digest:t.shipped_digest ?ship_substituted:t.ship_substituted
           ?substituted_from:t.substituted_from ~searches:t.searches ~replays:t.replays
           ~no_searches:t.no_searches
           ~shipped_mma:(Option.map t.shipped_mma ~f:mma_wire)
           ~arms:(List.map named ~f:arm) ())

let floats_of_gen g =
  let n = Array.fold (Bigarray.Genarray.dims g) ~init:1 ~f:( * ) in
  let a1 = Bigarray.reshape_1 g n in
  Array.init n ~f:(Bigarray.Array1.get a1)

(* Debug-name token matching: a param matches a fixture key when every required token appears among
   the underscore-separated tokens of its debug name. *)
let tokens_of dn = String.split dn ~on:'_' |> List.filter ~f:(Fn.non String.is_empty)

let matches ~required dn =
  let toks = tokens_of dn in
  List.for_all required ~f:(fun t -> List.mem toks t ~equal:String.equal)

(** [inject ctx st loss mapping] overwrites each param of [loss] with the fixture tensor whose
    required tokens all appear in the param's debug name. [mapping]: (fixture_key, required tokens).
    Every param must match exactly one mapping entry (and sizes must agree). Params matching no
    entry are left at their initialization (pass them deliberately!). *)
let inject ctx st loss mapping =
  Set.fold loss.Tensor.params ~init:ctx ~f:(fun ctx p ->
      let tn = p.Tensor.value in
      let dn = Tn.debug_name tn in
      match List.filter mapping ~f:(fun (_, required) -> matches ~required dn) with
      | [] -> failwith ("bench: no fixture entry matches param " ^ dn)
      | [ (key, _) ] ->
          let values = floats_of_gen (St.to_float32 st key) in
          let n = Tn.num_elems tn in
          if n <> Array.length values then
            failwith
              (Printf.sprintf "bench: %s has %d elems but fixture %s has %d" dn n key
                 (Array.length values));
          Context.set_values ctx tn values
      | ms ->
          failwith
            ("bench: param " ^ dn ^ " matches multiple fixture entries: "
            ^ String.concat ~sep:", " (List.map ms ~f:fst)))

(** The scratch lineage a tuned cell's search compiles and times its candidates on: a fresh root
    context, so the search's training-step executions never touch the cell's own weights, holding
    the FIXTURE's weights — the ones the timed steps train — rather than the init draw
    [Train.init_params] leaves there (gh-ocannl-1557). A crown is a time on the timing context's
    values. On cifar_resnet/cc the default-placements arm inlines the ReLU and max-pool backward
    selects into its consumer nests, where they compile to data-dependent branches, and its shipped
    routine ran 23% slower on the fixture weights than on the init draw it had been crowned on; the
    materialize-all arm, which evaluates each select once per element, moved by 3%. The search's own
    training steps still drift the scratch weights (one fixed batch, thousands of steps), so on such
    code a crown stays a few percent optimistic. [BENCH_TUNE_SCRATCH_WEIGHTS=init] keeps the init
    draw, the pre-gh-1557 search, so the A/B stays reproducible from one build
    ([benchmarks/gh1557_cells.sh]). *)
let tuning_scratch st loss mapping bindings =
  let scratch = Train.init_params (Context.auto ()) bindings loss in
  match Stdlib.Sys.getenv_opt "BENCH_TUNE_SCRATCH_WEIGHTS" with
  | None | Some ("" | "fixture") -> inject scratch st loss mapping
  | Some "init" -> scratch
  | Some other -> failwith ("BENCH_TUNE_SCRATCH_WEIGHTS must be fixture or init; found: " ^ other)

let dump_params loss =
  Set.iter loss.Tensor.params ~f:(fun p ->
      let tn = p.Tensor.value in
      Stdio.printf "param %s dims [%s]\n" (Tn.debug_name tn)
        (String.concat ~sep:";"
           (Array.to_list (Array.map (Lazy.force tn.Tn.dims) ~f:Int.to_string))))

(** Captures [comp]'s optimized lowering, the input every diagnostic below works from. Supplying a
    [?lowered_transform] bypasses the default annotator, so the routine this links is the
    unscheduled serial form — for a large graph that is the whole working set in one work-item's
    stack frame, and on HIP the gh-ocannl-533 scratch validator declines it (gpt2_mini's forward
    asks for 163,856 B). The lowering is captured inside the transform, i.e. before codegen and
    link, so a typed rejection costs nothing here: the routine is discarded either way. An untyped
    failure still propagates. *)
let capture_lowering ctx comp bindings =
  let stash = ref None in
  let outcome =
    Context.compile_outcome
      ~lowered_transform:(fun opt ->
        stash := Some opt;
        [ opt ])
      ~provenance:Ir.Schedule_outcome.User_schedule ctx comp bindings
  in
  (match (outcome, !stash) with
  | Ok _, Some _ -> ()
  | Ok _, None -> failwith "capture_lowering: the backend did not invoke lowered_transform"
  (* Failed before the transform ran: no lowering to keep, so nothing to continue with. *)
  | Error failure, None -> Ir.Schedule_outcome.raise_failure failure
  | Error (Ir.Schedule_outcome.Fatal _ as failure), Some _ ->
      Ir.Schedule_outcome.raise_failure failure
  | Error (Ir.Schedule_outcome.Classified classified), Some _ ->
      Stdio.printf "note: the unscheduled whole-routine form does not link here (%s)\n%!"
        (Ir.Schedule_outcome.detail_of_cause classified.Ir.Schedule_outcome.cause));
  Option.value_exn ~here:[%here] !stash

(** Diagnostic: prints the default fission-pipeline segment census for the captured lowered routine
    — per segment its kind, launch geometry and schedule size, and per top-level nest the loop
    extents (with axis-type letters) and written tensor nodes ([!] materialized, [~] routine-local).
    Used by the [bench_*_diag] runners; not part of the benchmark protocol. *)
let print_census ?promote_locals ~backend ~limits ~static_indices opt =
  let module LL = Ir.Low_level in
  let module Sched = Ir.Schedule in
  let stmt_detail plc stmt =
    let loops = ref [] and writes = ref [] and zeros = ref [] in
    let rec code (llc : LL.t) =
      match llc with
      | LL.Noop | LL.Comment _ | LL.Declare_local _ | LL.Staged_compilation _ | LL.Workgroup_barrier
      | LL.Tile_mma _ ->
          ()
      | LL.Seq (a, b) ->
          code a;
          code b
      | LL.For_loop { from_; to_; body; axis; _ } ->
          loops :=
            ( to_ - from_ + 1,
              match axis with
              | LL.Serial -> "s"
              | LL.Grid -> "G"
              | LL.Workgroup -> "W"
              | LL.Workgroup_reduce -> "R"
              | LL.Vectorized -> "v"
              | LL.Unrolled -> "u" )
            :: !loops;
          code body
      | LL.Scan_loop { from_; to_; body; _ } ->
          loops := (to_ - from_ + 1, "scan") :: !loops;
          code body
      | LL.Zero_out tn -> zeros := tn :: !zeros
      | LL.Set { tn; _ } -> writes := tn :: !writes
      | LL.Set_dynamic { tn; _ } -> writes := tn :: !writes
      | LL.Set_from_vec { tn; _ } -> writes := tn :: !writes
      | LL.Set_local _ -> ()
      | LL.If { body; _ } -> code body
    in
    code stmt;
    let tn_s tn =
      Printf.sprintf "%s%s(%d)" (Tn.debug_name tn)
        (if Tn.Placements.is_materialized_peek plc tn then "!" else "~")
        (Tn.num_elems tn)
    in
    let loops_s =
      String.concat ~sep:"," (List.rev_map !loops ~f:(fun (n, k) -> Printf.sprintf "%d%s" n k))
    in
    let ws = List.dedup_and_sort ~compare:Tn.compare (!writes @ !zeros) in
    if List.is_empty ws && String.is_empty loops_s then None
    else
      Some (Printf.sprintf "loops[%s] w:%s" loops_s (String.concat ~sep:" " (List.map ws ~f:tn_s)))
  in
  let gpu = Sched.backend_is_gpu backend in
  let promote_locals = Option.value promote_locals ~default:gpu in
  let preset o = if gpu then Sched.default_gpu ~limits o else Sched.default_cpu o in
  let zero_sched tns = if gpu then Sched.zero_expansion ~limits tns else [] in
  (* The default pipeline's own merge rule (gh-ocannl-1126), or this reconstructs another one. *)
  let keep_mapping = Sched.fission_keep_mapping ~is_gpu:gpu ~limits in
  let segs =
    Sched.fission_scheduled ~promote_locals ?keep_mapping ~preset ~zero_sched ~static_indices opt
  in
  Stdio.printf "default pipeline: %d segments\n" (List.length segs);
  List.iteri segs ~f:(fun i (kind, pre, sched, post) ->
      let dims = LL.launch_dims post.LL.llc in
      let np = Array.fold dims.grid ~init:1 ~f:( * ) * Array.fold dims.block ~init:1 ~f:( * ) in
      let stmts = List.length (LL.flat_lines [ post.LL.llc ]) in
      let kind_s = match kind with `Normal -> "N" | `Zeros -> "Z" | `Solo -> "S" in
      Stdio.printf "  seg%-3d %s threads=%-8d grid=[%d;%d;%d] block=[%d;%d;%d] ops=%d stmts=%d\n" i
        kind_s np dims.grid.(0) dims.grid.(1) dims.grid.(2) dims.block.(0) dims.block.(1)
        dims.block.(2) (List.length sched) stmts;
      let plc = pre.LL.optimize_ctx.LL.placements in
      List.iter (LL.flat_lines [ pre.LL.llc ]) ~f:(fun stmt ->
          match stmt_detail plc stmt with Some s -> Stdio.printf "        %s\n" s | None -> ()));
  Stdio.Out_channel.flush Stdio.stdout

(** Diagnostic companion to the [BENCH_SR_SITES] site listing (gh-ocannl-484 task 3): {e why} the
    accumulations that are absent from it were rejected. [Autotune.split_reduce_sites] returns only
    the reduction loops that clear its extent floor {e and} probe [Op_legal], and the two are
    indistinguishable from the listing alone — which left "find out why the conv-gradient
    accumulations are rejected" as the open question of the CUDA leg. This prints every low-output
    write with its enclosing loop nest and, per enclosing serial loop, the
    {!Ir.Schedule.op_legality} verdict of splitting that loop, so a missing site names the
    recognizer rule that rejected it. Used by the [bench_*_diag] runners; not part of the benchmark
    protocol. *)
let print_split_reduce_verdicts opt =
  let module LL = Ir.Low_level in
  let module Sched = Ir.Schedule in
  let module Idx = Ir.Indexing in
  (* The same output-parallelism bound the detector uses, so this probe covers exactly the writes it
     considers — a write above the bound is out of the family's scope by design, not by
     rejection. *)
  let out_max = 4096 in
  Stdio.printf "split-reduce probe (writes with <= %d cells):\n" out_max;
  let rec walk enclosing (llc : LL.t) =
    match llc with
    | LL.Seq (a, b) ->
        walk enclosing a;
        walk enclosing b
    | LL.If { body; _ } -> walk enclosing body
    | LL.For_loop { index; from_; to_; body; axis; _ } ->
        walk (enclosing @ [ (index, to_ - from_ + 1, axis) ]) body
    | (LL.Set { tn; _ } | LL.Set_dynamic { tn; _ }) as stmt ->
        let cells = try Tn.num_elems tn with _ -> 0 in
        if cells >= 1 && cells <= out_max then (
          Stdio.printf "  w:%s(%d) loops[%s]\n" (Tn.debug_name tn) cells
            (String.concat ~sep:","
               (List.map enclosing ~f:(fun (s, n, ty) ->
                    Printf.sprintf "%s=%d%s" (Idx.symbol_ident s) n
                      (match ty with LL.Serial -> "s" | _ -> "p"))));
          List.iter enclosing ~f:(fun (s, n, ty) ->
              if LL.equal_axis_type ty LL.Serial then
                let verdict =
                  match Sched.split_reduce ~axis:s ~target:tn ~num_blocks:2 with
                  | op, _, _, _ -> (
                      (* gh-ocannl-537: distinguish the rejection an interchange removes (seeding
                         hoists these and re-probes) from the ones that end the site. *)
                      let hoist () =
                        match Sched.split_reduce_hoist opt op with
                        | [] -> ""
                        | syms ->
                            " [hoistable: "
                            ^ String.concat ~sep:"," (List.map syms ~f:Idx.symbol_ident)
                            ^ "]"
                      in
                      match Sched.op_legality opt op with
                      | Sched.Op_legal -> "LEGAL"
                      | Sched.Op_illegal m -> "illegal: " ^ m ^ hoist ()
                      | Sched.Op_unknown m -> "unknown: " ^ m ^ hoist ())
                  | exception Invalid_argument m -> "raised: " ^ m
                in
                Stdio.printf "      axis %s extent %d -> %s\n" (Idx.symbol_ident s) n
                  (String.substr_replace_all verdict ~pattern:"\n" ~with_:" "));
          (* gh-ocannl-1414: the detector's fallback merges the two outermost reduction loops (the
             loops the cell does not mention); the coalescing's own verdict on them, as lowered. *)
          let idcs =
            match stmt with LL.Set { idcs; _ } | LL.Set_dynamic { idcs; _ } -> idcs | _ -> [||]
          in
          match
            List.filter enclosing ~f:(fun (s, _, _) ->
                not (Array.exists idcs ~f:(Idx.axis_index_mentions_symbol s)))
          with
          | (o, p, _) :: (i, q, _) :: _ ->
              let verdict =
                match Sched.op_legality opt (fst (Sched.coalesce ~outer:o ~inner:i)) with
                | Sched.Op_legal -> "LEGAL"
                | Sched.Op_illegal m -> "illegal: " ^ m
                | Sched.Op_unknown m -> "unknown: " ^ m
              in
              Stdio.printf "      coalesce %s*%s extent %d -> %s\n" (Idx.symbol_ident o)
                (Idx.symbol_ident i) (p * q)
                (String.substr_replace_all verdict ~pattern:"\n" ~with_:" ")
          | _ -> ())
    | _ -> ()
  in
  walk [] opt.LL.llc;
  Stdio.Out_channel.flush Stdio.stdout

(** The nodes a kernel writes, deduplicated — the label per-segment tables name a kernel by. *)
let writes_of (llc : Ir.Low_level.t) =
  let module LL = Ir.Low_level in
  let writes = ref [] in
  let rec code (l : LL.t) =
    match l with
    | LL.Noop | LL.Comment _ | LL.Declare_local _ | LL.Staged_compilation _ | LL.Workgroup_barrier
    | LL.Set_local _ ->
        ()
    (* A tile multiply-accumulate writes its accumulator block: without this a tensorized kernel
       whose only store is the [Tile_mma] -- a q/k/v projection's, say -- is listed as writing
       nothing, and the per-kernel table cannot say which kernel it is (gh-ocannl-728). *)
    | LL.Tile_mma { d = tn, _; _ } -> writes := tn :: !writes
    | LL.Seq (a, b) ->
        code a;
        code b
    | LL.For_loop { body; _ } | LL.If { body; _ } | LL.Scan_loop { body; _ } -> code body
    | LL.Zero_out tn | LL.Set { tn; _ } | LL.Set_dynamic { tn; _ } | LL.Set_from_vec { tn; _ } ->
        writes := tn :: !writes
  in
  code llc;
  List.dedup_and_sort ~compare:Tn.compare !writes

(** One kernel timed on its own: [compile ()] compiles it as a routine of its own, which is then run
    once and timed min-of-[repeats] with a device sync per run — so the number is one kernel's wall
    time including its launch — in the context that compile returned, and that context is released
    afterwards (it is a leaf: nothing is compiled from it). [bind] sets the routine's static indices
    before it runs.

    A segment compiled hermetically is not the segment as the full routine runs it: alone it keeps
    the whole per-thread working set that the full pipeline's promotions relieve, and on HIP the
    gh-ocannl-533 scratch validator declines it (gpt2_mini's cross-entropy head asks for 163,856 B
    per work-item). That is a limitation of the instrument, not of the workload, so a classified
    refusal comes back as [Error] with its detail and the caller carries on with the other segments.
    An untyped failure still propagates: [User_schedule] provenance keeps the instrument honest
    about compiler bugs. *)
let time_hermetic ~repeats ~bind compile =
  match compile () with
  | Error (Ir.Schedule_outcome.Fatal _ as failure) -> Ir.Schedule_outcome.raise_failure failure
  | Error (Ir.Schedule_outcome.Classified classified) ->
      Error (Ir.Schedule_outcome.detail_of_cause classified.Ir.Schedule_outcome.cause)
  | Ok (ctx, routine) ->
      let elapsed_ms c0 = Mtime.Span.to_float_ns (Mtime_clock.count c0) /. 1e6 in
      Exn.protect
        ~finally:(fun () -> Context.release ctx)
        ~f:(fun () ->
          bind routine;
          Train.run ctx routine;
          Context.sync ctx;
          let best = ref Float.infinity in
          for _ = 1 to repeats do
            let c0 = Mtime_clock.counter () in
            Train.run ctx routine;
            Context.sync ctx;
            best := Float.min !best (elapsed_ms c0)
          done;
          Ok (!best, routine))

(** Diagnostic: per-segment (approximately per-layer) wall times of the default fission pipeline,
    each by {!time_hermetic}. Run the full step once before calling so segment inputs are populated;
    timing mutates segment outputs (and re-accumulates accumulators), so restore any state that
    matters afterwards. [bind] binds the routine's static indices (e.g. the batch index). Used by
    the [bench_*_diag] runners; the benchmark protocol's counterpart, over the kernels a cell
    actually shipped, is {!dominant_kernel}. *)
let time_segments ?promote_locals ?(repeats = 20) ~backend ~limits ~static_indices ~ctx ~comp
    ~bindings ~bind opt =
  let module LL = Ir.Low_level in
  let module Sched = Ir.Schedule in
  let gpu = Sched.backend_is_gpu backend in
  let promote_locals = Option.value promote_locals ~default:gpu in
  let preset o = if gpu then Sched.default_gpu ~limits o else Sched.default_cpu o in
  let zero_sched tns = if gpu then Sched.zero_expansion ~limits tns else [] in
  (* The default pipeline's own merge rule (gh-ocannl-1126), or this reconstructs another one. *)
  let keep_mapping = Sched.fission_keep_mapping ~is_gpu:gpu ~limits in
  let segs =
    Sched.fission_scheduled ~promote_locals ?keep_mapping ~preset ~zero_sched ~static_indices opt
  in
  Stdio.printf "segment times (min of %d runs, ms):\n" repeats;
  let total = ref 0. in
  let declined = ref 0 in
  (* Per-segment tensorization, straight off each compiled routine (gh-ocannl-626): this table is
     where a per-kernel number is read, so it is where "this kernel emitted tensor cores" has to be
     legible. Without it a segment that declined its [Tile_mma] is indistinguishable here from one
     that ran on tensor cores, and the per-kernel attribution is what a measurement campaign
     quotes. *)
  let censuses = ref [] in
  List.iteri segs ~f:(fun i (kind, pre, _sched, post) ->
      let kind_s = match kind with `Normal -> "N" | `Zeros -> "Z" | `Solo -> "S" in
      let ws = String.concat ~sep:" " (List.map (writes_of pre.LL.llc) ~f:Tn.debug_name) in
      (* Substituted for the compile's fresh lowering of [comp] through the [lowered_transform]
         seam, autotune-style: the analysis layer still reads that lowering. *)
      let compile () =
        Context.compile_outcome
          ~lowered_transform:(fun _ -> [ post ])
          ~provenance:Ir.Schedule_outcome.User_schedule ctx comp bindings
      in
      match time_hermetic ~repeats ~bind compile with
      | Error detail ->
          Int.incr declined;
          Stdio.printf "  seg%-3d %s DECLINED (%s)  w:%s\n" i kind_s detail ws
      | Ok (best, routine) ->
          total := !total +. best;
          censuses := routine.Context.mma :: !censuses;
          (* The volatility census beside the tensorization one (gh-ocannl-782/820): on Metal the
             compiler-bug workaround uses volatile device reads inside a serial accumulation. A
             segment timing that looks unexpectedly slow should therefore be read together with how
             many accumulation sites carry those reads. *)
          Stdio.printf "  seg%-3d %s %8.4f ms  mma:%s  vol:%s  w:%s\n" i kind_s best
            (Ir.C_syntax.mma_summary_string routine.Context.mma)
            (Ir.C_syntax.volatility_summary_string routine.Context.volatility)
            ws);
  Stdio.printf "  total (sum of per-segment minima): %.4f ms%s\n" !total
    (if !declined = 0 then ""
     else Printf.sprintf " (INCOMPLETE: %d of %d segments declined)" !declined (List.length segs));
  let all = Ir.C_syntax.merge_mma_summaries !censuses in
  if all.Ir.C_syntax.scalar_fallbacks > 0 then
    Stdio.printf
      "  WARNING: %d of %d Tile_mma statements across these segments rendered the lane-0 scalar \
       fallback — those segment times are NOT tensorized timings \
       (--ocannl_schedule_log_declines=true names the rule)\n"
      all.Ir.C_syntax.scalar_fallbacks all.Ir.C_syntax.statements;
  Stdio.Out_channel.flush Stdio.stdout

(** {1 The dominant kernel's %-of-peak (gh-ocannl-1006)}

    The report's per-cell attainment column. Every kernel the step SHIPPED ([Context.routine]'s
    [segments] — for a tuned cell the searched winner's, which re-lowering the computation would not
    reproduce) is timed on its own by {!time_hermetic}; the slowest is the dominant kernel, and
    {!Bench_json.dominant_kernel_object} scores its [Ir.Cost_model.analyze] counts against the
    envelope ceiling matched to it. Dominance is by measured time, not by the model's roofline
    bound: the column exists partly to check the model, and choosing by its prediction would be
    circular.

    What the instrument costs and why it is still on for every cell: one compile per kernel and 21
    runs of each, after the timed steps (so none of it is in the step times or the memory column),
    which on a many-kernel workload is a visible share of a cell's wall time.
    [BENCH_DOMINANT_KERNEL=0] turns it off, and the result line then carries [null]. *)

let dominant_kernel_enabled () =
  match Stdlib.Sys.getenv_opt "BENCH_DOMINANT_KERNEL" with Some "0" -> false | _ -> true

(** The dominant-kernel instrument's wall-clock budget, in seconds (gh-ocannl-1260):
    [BENCH_DOMINANT_KERNEL_BUDGET_S], a positive number, default
    {!default_dominant_kernel_budget_s}.

    The instrument runs after the whole measurement has been paid for, and on a many-kernel step (a
    C compile per kernel on [cc], a seq-1024 training step anywhere) it can outlast the workload --
    long enough to eat the cell's cap, which the driver then reports as a timed-out cell with no
    result line at all. Past its budget it stops and reports ["no-kernel"], instrument over budget,
    and the cell keeps its result. The default is generous next to a typical instrument (seconds on
    a GPU, a minute or two for a cc kernel table) and well inside the sweeps' 1800 s cell caps; a
    driver with a tighter cap sets it lower. A malformed value fails the cell before it measures
    anything ({!measure_and_emit} reads it up front), never after. *)
let default_dominant_kernel_budget_s = 600.

let dominant_kernel_budget_s () =
  match Stdlib.Sys.getenv_opt "BENCH_DOMINANT_KERNEL_BUDGET_S" with
  | None | Some "" -> default_dominant_kernel_budget_s
  | Some s -> (
      match Float.of_string_opt s with
      | Some b when Float.(is_finite b && b > 0.) -> b
      | _ ->
          failwith ("BENCH_DOMINANT_KERNEL_BUDGET_S must be a positive number of seconds, got " ^ s)
      )

(** [BENCH_KERNEL_TABLE=1] (gh-ocannl-1002): {!dominant_kernel} also prints every kernel it timed on
    stderr -- one line per shipped segment with its min-of-N time, launch geometry, tensorization
    census and written nodes. *)
let kernel_table_enabled () = env_flag "BENCH_KERNEL_TABLE"

(** A shipped segment, copied before it is compiled again: the codegen settles placements in the
    record it is handed, and the routine's own IR is not this instrument's to mutate. *)
let scratch_segment (seg : Ir.Low_level.optimized) =
  let module LL = Ir.Low_level in
  {
    seg with
    LL.traced_store = Hashtbl.copy seg.LL.traced_store;
    LL.optimize_ctx = LL.copy_optimize_ctx seg.LL.optimize_ctx;
  }

(** The segment order of the compiled step, shared by its census and timing instruments. *)
let shipped_kernels routines =
  List.concat_map routines ~f:(fun (routine : Context.routine) ->
      List.map routine.Context.segments ~f:(fun seg -> (routine, seg)))

(** Compile the shipped IR itself: re-lowering from the step's context loses routine-local
    intermediates, and re-segmenting before it need not reproduce the pipeline that shipped. *)
let compile_shipped_kernel ~ctx ~bindings ~i (shipped : Context.routine) seg =
  let bind (r : Context.routine) =
    List.iter r.Context.bindings ~f:(fun (sym, cell) ->
        Option.iter
          (List.Assoc.find shipped.Context.bindings sym ~equal:Ir.Indexing.equal_static_symbol)
          ~f:(fun v -> cell := !v))
  in
  let compile () =
    Context.compile_outcome
      ~name:(Printf.sprintf "%s__kernel%d" shipped.Context.name i)
      ~prelowered:(scratch_segment seg)
      ~lowered_transform:(fun o -> [ o ])
      ~provenance:Ir.Schedule_outcome.User_schedule ctx Ir.Assignments.empty_comp bindings
  in
  (bind, compile)

let segment_geometry seg =
  let d = Ir.Low_level.launch_dims seg.Ir.Low_level.llc in
  let dims a = String.concat_array ~sep:";" (Array.map a ~f:Int.to_string) in
  Printf.sprintf "grid=[%s] block=[%s]" (dims d.grid) (dims d.block)

(** A segment's cost-model counts ([s], {!Ir.Cost_model.analyze} of its code) with the byte counts
    restricted to GLOBAL memory (gh-ocannl-1444, gh-ocannl-1520): [per_node], [read_bytes] and
    [write_bytes] keep only the nodes {!Ir.Cost_model.global_in} places in global memory --
    workgroup-shared tiles and thread-private scratch stay on chip and are left out, so a staged
    kernel is not charged its staging twice against a DRAM ceiling. The op count and the opacity
    flag are the whole kernel's, as analyzed. The one accounting behind both the kernel table's
    [io:] field ({!segment_io}) and the dominant kernel's [bytes] ({!dominant_kernel_record}). *)
let global_counts seg (s : Ir.Cost_model.summary) =
  let module CM = Ir.Cost_model in
  let per_node = List.filter s.CM.per_node ~f:(fun (tn, _) -> CM.global_in seg tn) in
  let sum f = List.sum (module Int) per_node ~f:(fun (_, fp) -> f fp) in
  {
    s with
    CM.per_node;
    read_bytes = sum (fun fp -> fp.CM.fp_read_bytes);
    write_bytes = sum (fun fp -> fp.CM.fp_write_bytes);
  }

(** A segment's global-memory traffic, {!global_counts}: [io:<read>/<written>] bytes, the distinct
    cells each global node's accesses touch times its element size. A trailing [~] marks a count
    that is an upper bound (an approximate access image among the global nodes), a trailing [?] code
    the model cannot see (an [opaque] kernel, whose counts may under-count). *)
let segment_io seg =
  let module CM = Ir.Cost_model in
  let g = global_counts seg (CM.analyze seg.Ir.Low_level.llc) in
  Printf.sprintf "io:%d/%d%s%s" g.CM.read_bytes g.CM.write_bytes
    (if CM.footprint_approximate g then "~" else "")
    (if g.CM.opaque then "?" else "")

(** Diagnostic census of the segments actually compiled, including gradient zeroing, backward and
    optimizer kernels for a training step. IDs match {!time_shipped_segments}. *)
let print_shipped_census ?(out = Stdio.stdout) routines =
  let kernels = shipped_kernels routines in
  Stdio.Out_channel.fprintf out "shipped pipeline: %d segments\n" (List.length kernels);
  List.iteri kernels ~f:(fun i (_, seg) ->
      Stdio.Out_channel.fprintf out "  seg%-3d %s stmts=%d w:%s\n" i (segment_geometry seg)
        (List.length (Ir.Low_level.flat_lines [ seg.Ir.Low_level.llc ]))
        (String.concat ~sep:" " (List.map (writes_of seg.Ir.Low_level.llc) ~f:Tn.debug_name)));
  Stdio.Out_channel.flush out

(** One timing loop and row format for both the standalone diagnostic and the benchmark's
    dominant-kernel column. Results retain refused rows in census order; fatal failures propagate
    (the benchmark caller catches them to preserve its already-paid-for result line).

    [stop], asked before each kernel, ends the loop there: the rows returned are then the kernels
    timed before it said so, fewer than the census. [time_kernel ~i timing] runs kernel [i]'s
    compile-and-time [timing] (default: just runs it) -- the seam a test stands a slow kernel in. *)
let time_shipped_kernels ?out ?(repeats = 20) ?(stop = fun () -> false)
    ?(time_kernel = fun ~i:_ timing -> timing ()) ~ctx ~bindings routines =
  let kernels = shipped_kernels routines in
  let n = List.length kernels in
  let rec attempted i = function
    | [] -> []
    | _ when stop () -> []
    | k :: rest ->
        let row = time_one i k in
        row :: attempted (i + 1) rest
  and time_one i (shipped, seg) =
    let bind, compile = compile_shipped_kernel ~ctx ~bindings ~i shipped seg in
    let result = time_kernel ~i (fun () -> time_hermetic ~repeats ~bind compile) in
    let row_out =
      match (out, result) with
      | Some out, _ -> Some out
      | None, Error _ -> Some Stdio.stderr
      | None, Ok _ -> None
    in
    Option.iter row_out ~f:(fun out ->
        let ws =
          String.concat ~sep:" " (List.map (writes_of seg.Ir.Low_level.llc) ~f:Tn.debug_name)
        in
        (match result with
        | Error detail ->
            Stdio.Out_channel.fprintf out "bench: kernel %d/%d %s DECLINED (%s) w: %s\n" i n
              (segment_geometry seg) detail ws
        | Ok (ms, routine) ->
            Stdio.Out_channel.fprintf out "bench: kernel %d/%d %.4f ms %s %s mma:%s vol:%s w: %s\n"
              i n ms (segment_geometry seg) (segment_io seg)
              (Ir.C_syntax.mma_summary_string routine.Context.mma)
              (Ir.C_syntax.volatility_summary_string routine.Context.volatility)
              ws);
        Stdio.Out_channel.flush out);
    (i, seg, result)
  in
  attempted 0 kernels

(** Full-step controls precede this: isolated backward/SGD mutates gradients/weights. Minima include
    a launch/sync floor per kernel and their sum is not step latency. *)
let time_shipped_segments ?(out = Stdio.stdout) ?(repeats = 20) ~ctx ~bindings routines =
  Stdio.Out_channel.fprintf out "segment times (min of %d runs, ms; isolated launch + sync):\n"
    repeats;
  let outcomes =
    time_shipped_kernels ~out ~repeats ~ctx ~bindings routines
    |> List.map ~f:(fun (_, _, result) -> Result.map result ~f:fst)
  in
  let timed = List.filter_map outcomes ~f:Result.ok in
  Stdio.Out_channel.fprintf out
    "  total (sum of per-segment minima, not step latency): %.4f ms (%d/%d timed)\n"
    (List.sum (module Float) timed ~f:Fn.id)
    (List.length timed) (List.length outcomes);
  Stdio.Out_channel.flush out;
  outcomes

(** Whether every node the kernel touches is 16-bit float storage computed at 16 bits — the
    [f16-native] ceiling's case (gh-ocannl-575), which only a target with native 16-bit arithmetic
    has: the same [Numerics.cpu_compute_prec] resolution the emitter uses. *)
let narrow_native ~(limits : Ir.Backend_intf.hardware_limits) (s : Ir.Cost_model.summary) =
  let is_half = function Ir.Ops.Half_prec _ -> true | _ -> false in
  limits.Ir.Backend_intf.native_fp16_arithmetic
  && (not (List.is_empty s.Ir.Cost_model.per_node))
  && List.for_all s.Ir.Cost_model.per_node ~f:(fun (tn, _) ->
      is_half (Lazy.force tn.Tn.storage_prec))
  && is_half (Ir.Numerics.cpu_compute_prec ~native_fp16_arithmetic:true Ir.Ops.half)

(** How many written nodes a kernel's label names before eliding the rest. *)
let label_writes = 3

(** The dominant kernel's {!Bench_json.kernel} record for shipped segment [seg], with what was
    measured of it, and its whole-kernel {!Ir.Cost_model.analyze} summary (the ceiling's
    [narrow_native] reads every node it touches, staging included). The op counts are the whole
    kernel's; [bytes] and [bytes_exact] are global memory's only ({!global_counts}, gh-ocannl-1520),
    the traffic the bandwidth ceiling it is scored against moves -- the same count as the kernel
    table's [io:] field. *)
let dominant_kernel_record ~segment ~segments ~declined ~seg_ms ~segments_ms ~tensorization seg =
  let module CM = Ir.Cost_model in
  let s = CM.analyze seg.Ir.Low_level.llc in
  let g = global_counts seg s in
  let writes =
    let names = List.map (writes_of seg.Ir.Low_level.llc) ~f:Tn.debug_name in
    let shown = List.take names label_writes in
    String.concat ~sep:" " shown
    ^
    if List.length names > label_writes then Printf.sprintf " +%d" (List.length names - label_writes)
    else ""
  in
  ( {
      Bench_json.segment;
      segments;
      declined;
      writes;
      seg_ms;
      segments_ms;
      tensorization;
      flops = s.CM.flops;
      bytes = CM.total_bytes g;
      flops_exact = not s.CM.flops_approx;
      bytes_exact = not (CM.footprint_approximate g);
      opaque = s.CM.opaque;
    },
    s )

(** Times every kernel of [routines] — the compiled routines of one step — and returns the
    {!Bench_json.dominant_kernel_object} of the slowest. [ctx] is the context the step last ran in
    and [bindings] the ones the step was compiled with; call after the timed steps, since timing a
    kernel alone mutates its outputs. Never raises: the instrument runs after the whole measurement
    has been paid for, and a cell must not lose its result line to a diagnostic column, so any
    failure is reported on stderr and as a ["no-kernel"] verdict naming it.

    Nor may it eat the cell: past [budget_s] seconds (default {!dominant_kernel_budget_s}) it stops
    and reports ["no-kernel"], instrument over budget (gh-ocannl-1260). It is stopped mid-kernel
    where {!instrument_preemptible} ({!with_deadline}), and in any case starts no kernel once the
    budget has passed. A kernel that started inside the budget and is not interrupted (Windows, or a
    long foreign call) may finish past it; when that was the last one, every kernel was timed and
    the column is the full answer, so it is reported -- the budget bounds the instrument's cost, and
    discarding a completed measurement would recover none of it. [time_kernel] is
    {!time_shipped_kernels}'. *)
let dominant_kernel ?(repeats = 20) ?budget_s ?(time_kernel = fun ~i:_ timing -> timing ()) ~ctx
    ~bindings routines =
  let budget_s = Option.value_or_thunk budget_s ~default:dominant_kernel_budget_s in
  let clock = Mtime_clock.counter () in
  let elapsed_s () = Mtime.Span.to_float_ns (Mtime_clock.count clock) /. 1e9 in
  let over_budget () = Float.(elapsed_s () > budget_s) in
  let started = ref 0 in
  let time_kernel ~i timing =
    started := i + 1;
    time_kernel ~i timing
  in
  let limits = Context.hardware_limits ctx in
  let backend = Context.backend_name ctx in
  let gpu = Ir.Schedule.backend_is_gpu backend in
  let flops_leg, bandwidth_leg = Autotune.envelope_legs ~limits in
  let leg_source key = function
    | v, `Backend -> (v, Printf.sprintf "%s class constant" backend)
    | v, `Config -> (v, Printf.sprintf "%s config" key)
  in
  let peak_flops = Option.map flops_leg ~f:(leg_source "model_peak_flops") in
  let peak_memory_bandwidth =
    Option.map bandwidth_leg ~f:(leg_source "model_peak_memory_bandwidth")
  in
  let kernels = shipped_kernels routines in
  let n = List.length kernels in
  let no_kernel note =
    Stdio.eprintf "bench: dominant kernel: %s\n%!" note;
    Bench_json.dominant_kernel_object ~note ~ceiling:(Error note) None
  in
  let over_budget_note () =
    Printf.sprintf
      "instrument over budget: stopped after %.1f s with %d of %d kernels started, past its %g s \
       budget (BENCH_DOMINANT_KERNEL_BUDGET_S)"
      (elapsed_s ()) !started n budget_s
  in
  match
    with_deadline ~seconds:budget_s (fun () ->
        time_shipped_kernels
          ?out:(if kernel_table_enabled () then Some Stdio.stderr else None)
          ~repeats ~stop:over_budget ~time_kernel ~ctx ~bindings routines)
  with
  | exception Instrument_over_budget -> no_kernel (over_budget_note ())
  | exception exn -> no_kernel ("instrument failed: " ^ Exn.to_string exn)
  | rows when List.length rows < n -> no_kernel (over_budget_note ())
  | rows -> (
      match
        List.filter_map rows ~f:(fun (i, seg, result) ->
            Result.ok result
            |> Option.map ~f:(fun (ms, routine) -> (i, seg, ms, routine.Context.mma)))
      with
      | [] when n = 0 -> no_kernel "the step shipped no kernel segments"
      | [] -> no_kernel (Printf.sprintf "all %d kernels declined to compile on their own" n)
      | timed ->
          let segments_ms = List.sum (module Float) timed ~f:(fun (_, _, ms, _) -> ms) in
          let i, seg, seg_ms, mma =
            List.max_elt timed ~compare:(fun (_, _, a, _) (_, _, b, _) -> Float.compare a b)
            |> Option.value_exn ~here:[%here]
          in
          let tensorization = Ir.C_syntax.tensorization_name mma.Ir.C_syntax.tensorization in
          let kernel, s =
            dominant_kernel_record ~segment:i ~segments:n
              ~declined:(n - List.length timed)
              ~seg_ms ~segments_ms ~tensorization seg
          in
          let ceiling =
            Bench_json.choose_ceiling ~gpu ~tensorization ~narrow_native:(narrow_native ~limits s)
              ~peak_flops ~peak_memory_bandwidth
          in
          Bench_json.dominant_kernel_object ~ceiling (Some kernel))

(** The compiled routines of a step shape, for {!dominant_kernel}. *)
let step_routines = function
  | Plain routine | Device_gate (_, _, routine, _) -> [ routine ]
  | Host_gate (_, _, grad_routine, sgd_routine) ->
      (* The host gate skips the optimizer on a non-finite checksum: if no timed step launched it,
         its kernels are not part of the measured step and cannot be its dominant kernel. *)
      if !host_gated_optimizer_runs > !optimizer_runs_at_window then [ grad_routine; sgd_routine ]
      else [ grad_routine ]

(** {1 The measurement protocol's parameters, apart from the fixture (gh-ocannl-702)}

    [benchmarks/fixtures/] holds only [DIGESTS.txt] in a fresh checkout: the [.safetensors] files
    are generated by [gen_fixtures.py], which needs a provisioned Python ML environment. So for as
    long as {!measure_and_emit} read its step counts off a [Safetensors.t] directly, nothing in a
    checkout could run it at all — the seam every benchmark number in every report passes through
    was guarded by the type checker agreeing that eleven labelled arguments have the right types,
    and by unit tests of [Bench_json.result_line] fed fabricated values. Naming the four values the
    protocol actually reads is what lets {!run_self_test} drive the whole of it on a model
    fabricated in memory, with no file on disk and no Python. *)

(** What an OCANNL cell's result line calls its memory counter (gh-ocannl-1006): the short tag the
    report prints on the row, and the long description its legend expands that tag into.

    Named on the wire rather than left to the [framework] column because the counters the report
    puts side by side are not one quantity: this one and [torch.cuda.max_memory_allocated] are both
    an allocator's high-water mark in requested bytes, while a current gauge sampled at step
    boundaries is a lower bound on the same window. What it covers is {!Ir.Alloc_census}'s coverage
    -- the pools recorded at the shared allocator seam -- which excludes a device's reserved
    merge-buffer slab, the loaded code modules and the host-side arrays. The tag says [all backends]
    because that coverage is backend-independent: a [cc] pool is counted exactly as a CUDA one. *)
let peak_memory_counter = "ocannl-seam"

let peak_memory_source = "OCANNL allocator seam high-water (requested bytes, all backends)"

type protocol = {
  workload : string;  (** The result line's [workload] field: the fixture's [name] metadata. *)
  parity_steps : int;  (** Steps whose losses are reported one by one, as the parity checksum. *)
  warmup_steps : int;  (** Untimed steps between the parity window and the timed one. *)
  timed_steps : int;  (** Steps timed twice over: per-step synced, then queued. *)
  fixture : (string * int) option;
      (** The fixture's path and size in bytes, for the checkpoints' identity (gh-ocannl-1209);
          [None] for a model fabricated in memory. *)
}

(** [fixture] is the path [st] was read from. *)
let protocol_of_st ~fixture st =
  {
    workload = get_meta st "name";
    parity_steps = meta_int st "parity_steps";
    warmup_steps = meta_int st "warmup_steps";
    timed_steps = meta_int st "timed_steps";
    fixture = Some (fixture, (Unix.stat fixture).Unix.st_size);
  }

(** Runs the measurement protocol and emits the JSON result line, which it also returns. [run_step]
    advances the batch binding and enqueues one step; [read_loss] returns the current loss value
    (awaits the device); [sync] awaits all queued work. [out] is where the line is written —
    [stdout], where [orchestrate.py] reads a cell's result, except for {!run_self_test}, which
    redirects it so that a golden carries no timings.

    Keep the protocol here rather than in a caller (gh-ocannl-702): {!run_self_test} is what stands
    behind this function in a fresh checkout, and it stands behind exactly what this function does.

    [storage_of] is the model body's output, whose compiled storage precision the line reports as
    [storage_precision] beside the requested [precision] ({!storage_precision}).

    [dominant_kernel] measures the report's %-of-peak column (see {!dominant_kernel}); it is called
    once, after the timed steps and the memory reading, unless [BENCH_DOMINANT_KERNEL=0]. Without it
    the line carries [null] there.

    Until the result line, the protocol writes {!Bench_json.checkpoint_line}s to [checkpoint_out]
    (default [stderr]), flushed, behind {!Bench_json.checkpoint_prefix} (gh-ocannl-1209): one after
    every parity step, and one after the timed steps, before the dominant-kernel instrument. A
    driver that kills the cell during a later stage — the instrument included — still finds the
    completed losses in the cell's log. A checkpoint carries no timing: only the result line is a
    benchmark result, and a cell that never emitted one is a failed cell whatever it checkpointed.

    Every number in the line goes through {!Bench_json}, so a non-finite one is [null] rather than
    OCaml's [nan] / [inf]: a training run that diverges is exactly the run whose loss trajectory the
    report needs, and a line that does not parse is a cell [orchestrate.py] drops as a broken runner
    after the whole measurement has been paid for (gh-ocannl-676).

    The line's [searched] field states whether {e this process} ran a schedule search
    (gh-ocannl-644). A tuned cell is measured by a two-pass protocol — pass 1 searches and populates
    [autotune_cache/], a fresh pass 2 replays the cached winner and provides the step times, because
    a searching process is measurably slower per launch (accumulated modules and buffers; measured
    at +10.3% on small CUDA kernels behind a 16 s search, and ~0 behind a cheap one or where a step
    is milliseconds rather than microseconds -- gh-ocannl-675). Both passes emit the same
    [framework]/[backend]/[variant]/[precision], so without this field a report can quote pass-1
    timings as protocol-compliant ones indefinitely, and nothing in the artifact contradicts it —
    which is what [report-gh612-hip.md] did for fifteen revisions. [searched] is [false] for an
    untuned cell too: it says no search ran in this process, which for a cell that tunes nothing is
    both true and the condition the protocol wants — as it is for a tuned cell under
    [autotune_search=false], which ships the untuned default having neither searched nor replayed
    (the [tune] object's [no_searches] is what tells that apart from a replay — a count of arms
    whose {!Autotune.outcome} was one of the two states that search nothing, rather than an
    inference from two counters that are both zero). *)
let measure_and_emit ~routines ~protocol ~backend ~variant ?(precision = "f32") ?storage_of
    ~compile_s ?tokens_per_step ?tune ?(out = Stdio.stdout) ?(checkpoint_out = Stdio.stderr)
    ?dominant_kernel ~run_step ~read_loss ~sync () =
  let { workload; parity_steps; warmup_steps; timed_steps; fixture } = protocol in
  Stdio.eprintf "bench: compiled in %.1fs, starting %d parity steps\n%!" compile_s parity_steps;
  let dominant_kernel = if dominant_kernel_enabled () then dominant_kernel else None in
  (* Read now, so a malformed budget fails the cell before it measures anything. *)
  if Option.is_some dominant_kernel then ignore (dominant_kernel_budget_s () : float);
  let profile = Option.map Utils.active_profile ~f:(fun (_, name, _) -> name) in
  (* Resolved once, so every checkpoint and the result line carry the same record of the regime. *)
  let regime_knobs =
    List.map (Utils.profile_payload_sources "approximate") ~f:(fun (key, resolution) ->
        ( key,
          Option.map resolution ~f:(fun (value, source) ->
              (value, Utils.config_source_label source)) ))
  in
  (* The same record for every key the process inherited from its environment (gh-ocannl-1337): the
     observation the driver's [ambient_ocannl_shadowed] stamp is checked against. *)
  let ambient_resolved =
    List.map (Utils.inherited_env_key_sources ()) ~f:(fun (var, key, resolution) ->
        ( var,
          key,
          Option.map resolution ~f:(fun (value, source) ->
              (value, Utils.config_source_label source)) ))
  in
  let checkpoint ~completed_steps at losses =
    Stdio.Out_channel.output_string checkpoint_out
      (Bench_json.checkpoint_prefix
      ^ Bench_json.checkpoint_line ~backend ~variant ~precision ~profile ~regime_knobs
          ~ambient_resolved ~workload ~fixture ~executable:Stdlib.Sys.executable_name ~parity_steps
          ~dominant_kernel:(Option.is_some dominant_kernel) ~completed_steps ~at ~losses
      ^ "\n");
    (* Flushed at once: a kill does not unwind, so a checkpoint still in this process's buffer is
       one the driver never sees. *)
    Stdio.Out_channel.flush checkpoint_out
  in
  (* Monotonic high-resolution clock (not [Unix.gettimeofday]): on Windows the latter ticks at ~1
     ms, which floors sub-millisecond step times to 0. *)
  let elapsed_ms c0 = Mtime.Span.to_float_ns (Mtime_clock.count c0) /. 1e6 in
  let losses = Array.create ~len:parity_steps Float.nan in
  for i = 0 to parity_steps - 1 do
    let c0 = Mtime_clock.counter () in
    run_step ();
    let l = read_loss () in
    losses.(i) <- l;
    Stdio.eprintf "bench: parity step %d loss %.6g (%.2fs)\n%!" i l (elapsed_ms c0 /. 1000.);
    checkpoint ~completed_steps:(i + 1)
      (Bench_json.In_parity (i + 1))
      (Array.sub losses ~pos:0 ~len:(i + 1))
  done;
  for _ = 1 to warmup_steps do
    run_step ()
  done;
  sync ();
  (* The memory column's bracket (gh-ocannl-1006). Here, not at process exit: a tuned cell's search
     allocates a candidate buffer per arm, so a counter read at the end reports the SEARCH's high
     water rather than the workload's. Rebasing after the warmup makes the reading the workload's
     steady-state footprint -- everything still held at this point, plus anything the timed steps go
     on to allocate -- which is the quantity a footprint-scoped materialization trades time for. *)
  Ir.Alloc_census.reset_peak ();
  optimizer_runs_at_window := !host_gated_optimizer_runs;
  let synced =
    Array.init timed_steps ~f:(fun _ ->
        let c0 = Mtime_clock.counter () in
        run_step ();
        sync ();
        elapsed_ms c0)
  in
  let c0 = Mtime_clock.counter () in
  for _ = 1 to timed_steps do
    run_step ()
  done;
  sync ();
  let queued_ms = elapsed_ms c0 /. Float.of_int timed_steps in
  let peak_memory =
    Some
      ( (Ir.Alloc_census.snapshot ()).Ir.Alloc_census.peak_pool_bytes,
        peak_memory_counter,
        peak_memory_source )
  in
  (* The %-of-peak column's instrument (gh-ocannl-1006): after the timed steps AND after the memory
     reading, because it compiles one routine per kernel and runs each on its own -- which is
     neither the workload's step time nor its footprint, and mutates the kernels' outputs. It can
     also cost more than the workload, so the completed losses are checkpointed first. *)
  checkpoint
    ~completed_steps:(parity_steps + warmup_steps + (2 * timed_steps))
    Bench_json.Before_diagnostics losses;
  let dominant_kernel = Option.map dominant_kernel ~f:(fun measure -> measure ()) in
  Array.sort synced ~compare:Float.compare;
  let line =
    Bench_json.result_line ~backend ~variant ~precision
      ?storage_precision:(Option.map storage_of ~f:storage_precision)
      ~profile ~regime_knobs ~ambient_resolved ~fp_licences:(fp_licence_record ()) ~workload
      ~compile_s
      ~searched:(Option.value_map tune ~default:false ~f:searched)
      ~shipped_mma:(mma_wire (step_census routines))
      ?tokens_per_step ?tune:(Option.bind tune ~f:tune_json) ~p10:(percentile synced 10.)
      ~p50:(percentile synced 50.) ~p90:(percentile synced 90.) ~queued_ms ~timed_steps ~peak_memory
      ?dominant_kernel ~losses ()
  in
  Stdio.Out_channel.output_string out (line ^ "\n");
  Stdio.Out_channel.flush out;
  line

(** {1 Fixture-free self-test of the measurement path (gh-ocannl-702)}

    [benchmarks/fixtures/] is empty in a fresh checkout and the runners are dispatched through
    [benchmarks/.venv], so without a provisioned Python ML environment no benchmark cell can be run
    at all — including the OCANNL ones, which need nothing from torch but the bytes. That left
    {!measure_and_emit} — the emitter every OCANNL benchmark cell's result flows through — with no
    executable guard anywhere: a break in it would first show up as a wrong number on a GPU box,
    days later.

    {!run_self_test} closes that with a run that needs no file on disk. It is deliberately {e not} a
    comparable measurement: its model is fabricated in this process, so its bytes are not the
    byte-identical fixture the cross-framework parity gate is built on, and its workload name says
    so. It sidesteps the fixture contract rather than weakening it. *)

(** The self-test's protocol. [selftest-tiny] is not a benchmark cell and its numbers compare to
    nothing — see above. The step counts keep the whole run to seconds on any backend while leaving
    [timed_steps] large enough that {!percentile} lands the three reported percentiles on three
    different samples: a swap of [~p10] and [~p90] then emits a line whose percentiles are out of
    order, and that argument mapping into [Bench_json.result_line] is the one link between the
    protocol and the wire format that no unit test of either half can reach. *)
let self_test_protocol =
  {
    workload = "selftest-tiny";
    parity_steps = 2;
    warmup_steps = 1;
    timed_steps = 5;
    fixture = None;
  }

(** The f32 leg, built rather than parsed. {!precision_leg} reads [BENCH_PRECISION] and friends from
    the environment, which is right for a runner and wrong here: the self-test's emitted record is
    diffed against a golden, so an ambient [BENCH_PRECISION] must not be able to change what it
    reports. *)
let self_test_leg =
  {
    label = "f32";
    base = "f32";
    prec = None;
    static_scale = false;
    gate_interval = None;
    init_scale = 65536.;
  }

(** Trains a tiny MLP over data fabricated in memory, through the same step machinery every runner
    uses ({!train_step_parts}, {!compile_train_step}, {!run_train_step}), drives the full
    measurement protocol, and returns the emitted result line. [out] is where the line is emitted
    (default [stdout], as for a real cell).

    Not a benchmark, and not comparable to one: see {!self_test_protocol}. The backend is chosen the
    usual OCANNL way, so the same call smoke-tests the measurement path on whatever backend the
    caller is configured for.

    [checkpoint_out] is {!measure_and_emit}'s. [late_probe], when given, runs in place of the
    dominant-kernel instrument — the protocol's last stage — and its result stands in for that
    column: it is how a test stands a stalled or killed diagnostic where the real one runs.
    [dominant_kernel_budget_s] and [time_kernel] are {!dominant_kernel}'s [budget_s] and
    [time_kernel], for the real instrument; [before_instrument] runs right before it, after every
    measured stage -- where a test arms what the instrument must meet on entry (a timer), so that
    how long compiling and the timed steps took on the box cannot decide the outcome. *)
let run_self_test ?(out = Stdio.stdout) ?checkpoint_out ?(leg = self_test_leg) ?late_probe
    ?dominant_kernel_budget_s ?time_kernel ?before_instrument ?inspect_compiled ?inspect_step () =
  let module TDSL = Operation.DSL_modules.TDSL in
  let module IDX = Train.IDX in
  let n_samples = 8 and n_features = 4 and n_hidden = 5 and n_classes = 3 in
  (* Deterministic and RNG-free: what the self-test asserts on is the SHAPE of the emitted record,
     and a model that varied per run would turn a divergence to nan into a flake rather than a
     failure. *)
  let wave k = Float.of_int ((k * 37 % 19) - 9) /. 10. in
  let nd debug dims f = Ir.Ndarray.init_array ~debug Ir.Ops.single ~dims ~padding:None ~f in
  let x_nd =
    nd "selftest_x" [| n_samples; n_features |] (fun i -> wave ((i.(0) * n_features) + i.(1)))
  in
  let y_nd =
    nd "selftest_y" [| n_samples; n_classes |] (fun i ->
        if i.(1) = i.(0) % n_classes then 1. else 0.)
  in
  let w_nd name ~dout ~din = nd name [| dout; din |] (fun i -> wave (3 + (i.(0) * din) + i.(1))) in
  let b_nd name ~dout = nd name [| dout |] (fun _ -> 0.) in
  let xs = TDSL.rebatch ~l:"xs" x_nd () in
  let ys = TDSL.rebatch ~l:"ys" y_nd () in
  let w1 =
    TDSL.wrap_param ~l:"w1" ~i:[ n_features ] ~o:[ n_hidden ]
      (w_nd "w1" ~dout:n_hidden ~din:n_features)
      ()
  in
  let b1 = TDSL.wrap_param ~l:"b1" ~o:[ n_hidden ] (b_nd "b1" ~dout:n_hidden) () in
  let w2 =
    TDSL.wrap_param ~l:"w2" ~i:[ n_hidden ] ~o:[ n_classes ]
      (w_nd "w2" ~dout:n_classes ~din:n_hidden)
      ()
  in
  let b2 = TDSL.wrap_param ~l:"b2" ~o:[ n_classes ] (b_nd "b2" ~dout:n_classes) () in
  let logits =
    let open TDSL.O in
    b2 + (w2 * relu (b1 + (w1 * xs)))
  in
  let%op loss =
    Nn_blocks.cross_entropy_loss ~spec:"...|v" ~normalize_by:!..n_samples () ~logits ~targets:ys
  in
  let learning_rate = TDSL.O.( !. ) 0.01 in
  let parts = train_step_parts ~leg ~learning_rate loss in
  let ctx = Context.auto () in
  let backend = Context.backend_name ctx in
  let bindings = IDX.empty in
  let ctx = Train.init_params ctx bindings loss in
  let t0 = Unix.gettimeofday () in
  let ctx, routines =
    compile_train_step ~tune:false
      ~tuned:(fun _ _ -> failwith "bench self-test: the self-test does not autotune")
      ctx bindings parts
  in
  let compile_s = Unix.gettimeofday () -. t0 in
  Option.iter inspect_compiled ~f:(fun inspect -> inspect loss routines);
  let ctx_ref = ref ctx in
  let step_count = ref 0 in
  let run_step () =
    run_train_step routines ctx_ref ~step:!step_count;
    Int.incr step_count
  in
  let open Operation.At in
  (* No [~tune]: an untuned cell, so the line's [searched] is false and it carries no [tune] object.
     What the self-test guards is the protocol and the emitter, not the search. *)
  let line =
    measure_and_emit ~routines ~protocol:self_test_protocol ~backend ~variant:"self-test" ~compile_s
      ~storage_of:logits ~out ?checkpoint_out
      ~dominant_kernel:
        (Option.value late_probe ~default:(fun () ->
             Option.iter before_instrument ~f:(fun f -> f ());
             dominant_kernel ?budget_s:dominant_kernel_budget_s ?time_kernel ~ctx:!ctx_ref ~bindings
               (step_routines routines)))
      ~run_step
      ~read_loss:(fun () -> (!ctx_ref, loss).@[0])
      ~sync:(fun () -> Context.sync !ctx_ref)
      ()
  in
  Option.iter inspect_step ~f:(fun inspect ->
      inspect !ctx_ref bindings loss (step_routines routines));
  line

(** Diagnostic (gh-ocannl-1413): the product-site census of a lowered routine under the four GPU
    segmentations (the default's and the search's, each plain and [arity_cuts]) — per [`Normal]
    segment, its product accumulations (a serial nest assigning a cell from its own read and exactly
    two other reads), which of them is the segment's matmul site as the seeders see it
    ({!Autotune.detect_matmul_canonical}, marked [*]), which other one is a site in a segment of its
    own ([+]), and how many seeds the search's lists hold for the segment. A site is not a seed: a
    companion can refute every seed of its segment, so read the seed counts (and, for the untuned
    plan, [bench_gpt_diag]'s seeded-default line) before concluding a site is tiled. Prints the
    segments with several products or none that is a site. Not part of the benchmark protocol. *)
let print_site_census ~limits ~static_indices opt =
  let module LL = Ir.Low_level in
  let module Sched = Ir.Schedule in
  let scratch (o : LL.optimized) =
    {
      o with
      LL.traced_store = Hashtbl.copy o.LL.traced_store;
      LL.optimize_ctx = LL.copy_optimize_ctx o.LL.optimize_ctx;
    }
  in
  let rec nest (llc : LL.t) depth =
    match llc with
    | LL.For_loop { body; _ } -> (
        match
          List.filter (LL.flat_lines [ body ]) ~f:(function
            | LL.Noop | LL.Comment _ -> false
            | _ -> true)
        with
        | [ single ] -> nest single (depth + 1)
        | _ -> None)
    | LL.If { body; _ } -> nest body depth
    | LL.Set { tn; idcs; llsc; _ } -> (
        let rec gets (sc : LL.scalar_t) =
          let arg (s, _) = gets s in
          match sc with
          | LL.Get (t, i) -> [ (t, i) ]
          | LL.Ternop (_, a, b, c) -> arg a @ arg b @ arg c
          | LL.Binop (_, a, b) -> arg a @ arg b
          | LL.Unop (_, a) -> arg a
          | _ -> []
        in
        let gets = gets llsc in
        let self (t, i) = phys_equal t tn && Array.equal Ir.Indexing.equal_axis_index i idcs in
        match List.partition_tf gets ~f:self with _ :: _, [ _; _ ] -> Some (tn, depth) | _ -> None)
    | _ -> None
  in
  let products (pre : LL.optimized) =
    List.filter_map (LL.flat_lines [ pre.LL.llc ]) ~f:(fun stmt ->
        Option.map (nest stmt 0) ~f:(fun (tn, depth) -> (stmt, tn, depth)))
  in
  let one ~label ~arity_cuts ~fold_zeros ~min_parallel =
    let preset seg = Sched.default_gpu ?min_parallel ~limits seg in
    let zero_sched tns = Sched.zero_expansion ~limits tns in
    match
      Sched.fission_segmented ~promote_locals:true ~arity_cuts ~fold_zeros
        ?keep_mapping:(Sched.fission_keep_mapping ~is_gpu:true ~limits)
        ~preset ~zero_sched ~static_indices (scratch opt)
    with
    | exception Ir.Schedule_outcome.Cause_at (_, c) ->
        Stdio.printf "site census [%s]: fission refused (%s)\n" label
          (Ir.Schedule_outcome.detail_of_cause c)
    | plan, tuples ->
        let at = ref 0 in
        let total = ref 0 and sites = ref 0 and alone = ref 0 and multi = ref 0 in
        let seedable = ref 0 in
        Stdio.printf "site census [%s]: %d segments\n" label (List.length plan);
        List.iteri (List.zip_exn plan tuples) ~f:(fun i ((kind, n), (_, pre, _, _)) ->
            let start = !at in
            at := !at + n;
            match Ir.Schedule.segment_class kind with
            | `Zeros | `Solo -> ()
            | `Normal -> (
                match products pre with
                | [] -> ()
                | ps ->
                    let site =
                      Option.map (Autotune.detect_matmul_canonical pre) ~f:(fun (s, _, _) ->
                          s.Autotune.m_d)
                    in
                    total := !total + List.length ps;
                    if List.length ps > 1 then Int.incr multi;
                    let items =
                      List.map ps ~f:(fun (stmt, tn, depth) ->
                          let is_site = Option.exists site ~f:(phys_equal tn) in
                          if is_site then Int.incr sites;
                          let alone_site =
                            (not is_site)
                            && Option.is_some
                                 (Autotune.detect_matmul_canonical
                                    { (scratch pre) with LL.llc = stmt })
                          in
                          if alone_site then Int.incr alone;
                          Printf.sprintf "%s%s(%d,d%d)" (Tn.debug_name tn)
                            (if is_site then "*" else if alone_site then "+" else "")
                            (Tn.num_elems tn) depth)
                    in
                    (* The search's seed list for the segment, as its per-segment seeding sees
                       it. *)
                    let seeds =
                      List.length
                        (Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits pre
                        @ Autotune.coalesced_seed_params ~is_gpu:true ~is_cpu:false ~limits pre)
                    in
                    if seeds > 0 then Int.incr seedable;
                    if List.length ps > 1 || Option.is_none site then
                      Stdio.printf "  seg%d u%d n%d seeds=%d: %s\n" i start n seeds
                        (String.concat ~sep:" " items)));
        Stdio.printf
          "site census [%s]: %d product accumulations, %d their segment's site, %d more sites \
           alone, %d multi-product segments, %d product segments the search seeds\n\
           %!"
          label !total !sites !alone !multi !seedable
  in
  one ~label:"default" ~arity_cuts:false ~fold_zeros:false ~min_parallel:None;
  one ~label:"default arity_cuts" ~arity_cuts:true ~fold_zeros:false ~min_parallel:None;
  one ~label:"search" ~arity_cuts:false ~fold_zeros:true ~min_parallel:(Some 1);
  one ~label:"search arity_cuts" ~arity_cuts:true ~fold_zeros:true ~min_parallel:(Some 1)
