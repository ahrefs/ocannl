(* Standalone discriminator for gh-ocannl-828: do two long Metal command buffers become much more
   expensive merely because both are submitted before the host waits, or only when OCANNL's
   SharedEvent ordering shape is present?

   This links the Metal bindings directly and contains no OCANNL lowering, scheduling, context or
   stream code. It measures three shapes over the same pipeline and two independent output buffers:

   - [sync-between]: commit one kernel and await it before committing the next.

   - [raw-queued]: commit both kernels, then await both (no intervening host synchronization).

   - [event-chain]: Metal_backend's command-buffer sequence: kernel, signal, wait+kernel, signal,
   followed by one host wait on the final SharedEvent value. The wait is encoded BEFORE the second
   kernel's compute pass, as [Metal_backend.link_proc] encodes it ([encode_wait_for_enqueued] runs
   before the encoder is created), so the second kernel cannot start until the signal fires.

   - [wait-after-kernel]: the same buffers with the wait encoded AFTER the second kernel's compute
   pass — [encodeWaitForEvent] takes effect at the point in the buffer where it is encoded, so a
   wait appended to an already-encoded buffer orders nothing before it. This is the shape the
   event-chain arm had until gh-ocannl-909, and it is why staging#641 measured the two kernels
   overlapping "through the backend's sequence": the probe was not reproducing the backend. Kept as
   the artifact's record; the executed pin of the backend's own ordering is
   [test/operations/back_to_back_runs.ml].

   The kernel's loop bound is runtime data and every iteration performs a volatile device read, so
   the compiler cannot fold the long loop away. The kernel length is fixed one of two ways.

   [target_ms] (positional, default 1000) is a duration target for one kernel. Calibration measures
   a short kernel, scales its iteration count to the target, then feedback-corrects: it re-times the
   scaled kernel and rescales by the miss until the measurement lands within [--tolerance] of the
   target (default 2%) or [--max-corrections] rounds run out. One-shot scaling alone undershoots,
   because the fixed launch overhead inflates the short seed kernel's per-iteration cost.

   [--iterations=N] is an exact loop count, no calibration. The report always prints the iteration
   count the arms ran with, so a threshold found under a duration target is rerun exactly by pinning
   that count.

   The report prints the requested duration and the achieved single-kernel duration side by side.
   Run only on an otherwise idle Metal Mac, recording [uptime] beside the output:

   dune exec benchmarks/runners/ocannl/metal_queue_probe.exe -- 2000

   dune exec benchmarks/runners/ocannl/metal_queue_probe.exe -- --iterations=734003200

   [--deadline-seconds=N] bounds the whole run for unattended use; see [arm_deadline]. *)

module Me = Metal

let source =
  {|
#include <metal_stdlib>
using namespace metal;

kernel void spin(
    device const volatile float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& iterations [[buffer(2)]]) {
  float x = out[0];
  for (uint i = 0; i < iterations; ++i) {
    x = fma(x, 1.00000011920928955078125f, in[i & 1023] * 0.00000011920928955078125f);
  }
  out[0] = x;
}
|}

let ropts =
  Me.ResourceOptions.(
    storage_mode_shared + cpu_cache_mode_write_combined + hazard_tracking_mode_untracked)

let now () = Unix.gettimeofday ()
let elapsed_ms start = (now () -. start) *. 1000.

(* [?wait]: a SharedEvent wait encoded before the compute pass, where it orders the kernel. *)
let command_buffer ?wait ~queue ~pso ~input ~output ~iterations_buf () =
  let cb = Me.CommandBuffer.on_queue queue in
  Option.iter
    (fun (event, value) ->
      Me.CommandBuffer.encode_wait_for_event cb (Me.SharedEvent.super event) value)
    wait;
  let enc = Me.ComputeCommandEncoder.on_buffer cb in
  Me.ComputeCommandEncoder.set_compute_pipeline_state enc pso;
  Me.ComputeCommandEncoder.set_buffer enc ~index:0 input;
  Me.ComputeCommandEncoder.set_buffer enc ~index:1 output;
  Me.ComputeCommandEncoder.set_buffer enc ~index:2 iterations_buf;
  Me.ComputeCommandEncoder.dispatch_threadgroups enc
    ~threadgroups_per_grid:{ width = 1; height = 1; depth = 1 }
    ~threads_per_threadgroup:{ width = 1; height = 1; depth = 1 };
  Me.ComputeCommandEncoder.end_encoding enc;
  cb

let commit_signal queue event value =
  let cb = Me.CommandBuffer.on_queue queue in
  Me.CommandBuffer.encode_signal_event cb (Me.SharedEvent.super event) value;
  Me.CommandBuffer.commit cb

let check_completed label cb =
  match Me.CommandBuffer.get_status cb with
  | Completed -> ()
  | Error ->
      Printf.eprintf "%s command buffer failed: %s\n" label
        (Option.value (Me.CommandBuffer.get_error cb) ~default:"unknown Metal error");
      exit 1
  | status ->
      Printf.eprintf "%s command buffer ended in status %s\n" label
        (Sexplib0.Sexp.to_string_hum (Me.CommandBuffer.Status.sexp_of_t status));
      exit 1

let usage () =
  Printf.eprintf
    "usage: metal_queue_probe [target_ms] [--iterations=N] [--tolerance=FRACTION] \
     [--max-corrections=N] [--deadline-seconds=N]\n";
  exit 2

(* [--deadline-seconds=N] bounds the whole run, for the unattended smoke use: every wait here is
   unbounded on purpose -- [wait_until_completed] has no timeout and the SharedEvent wait is given
   [ULLong.max_int] -- so a binding or driver regression that WEDGES a submission does not fail, it
   hangs, and an unattended run then holds its CI job until the job's own timeout rather than
   reporting anything.

   Deliberately no OCaml signal handler: SIGALRM's default action kills the process from the kernel,
   which is what reaches a thread parked inside a blocking Metal call. An OCaml handler runs only at
   a safepoint, so the one case worth bounding is exactly the case it would not interrupt. The
   process therefore dies by signal, and its runner reports the action as failed. *)
let arm_deadline seconds = ignore (Unix.alarm seconds : int)
let max_iterations = 1 lsl 30

type request = Target_ms of float | Exact_iterations of int

let () =
  let request = ref None and tolerance = ref 0.02 and max_corrections = ref 6 in
  let deadline_seconds = ref None in
  let set_request value =
    if Option.is_some !request then (
      Printf.eprintf "metal_queue_probe: give either target_ms or --iterations, once\n";
      usage ());
    request := Some value
  in
  let positive_float ~name value =
    match Float.of_string_opt value with
    | Some value when Float.is_finite value && value > 0. -> value
    | _ ->
        Printf.eprintf "metal_queue_probe: %s must be a positive finite number\n" name;
        usage ()
  in
  let positive_int ~name ~max value =
    match int_of_string_opt value with
    | Some value when value > 0 && value <= max -> value
    | _ ->
        Printf.eprintf "metal_queue_probe: %s must be an integer in 1..%d\n" name max;
        usage ()
  in
  List.iter
    (fun arg ->
      let flag name = String.starts_with ~prefix:(name ^ "=") arg in
      let value name =
        String.sub arg (String.length name + 1) (String.length arg - String.length name - 1)
      in
      if flag "--iterations" then
        set_request
          (Exact_iterations
             (positive_int ~name:"--iterations" ~max:max_iterations (value "--iterations")))
      else if flag "--tolerance" then
        tolerance := positive_float ~name:"--tolerance" (value "--tolerance")
      else if flag "--max-corrections" then
        max_corrections :=
          positive_int ~name:"--max-corrections" ~max:100 (value "--max-corrections")
      else if flag "--deadline-seconds" then
        deadline_seconds :=
          Some (positive_int ~name:"--deadline-seconds" ~max:86400 (value "--deadline-seconds"))
      else if String.starts_with ~prefix:"-" arg then usage ()
      else set_request (Target_ms (positive_float ~name:"target_ms" arg)))
    (List.tl (Array.to_list Sys.argv));
  let request = Option.value !request ~default:(Target_ms 1000.) in
  let tolerance = !tolerance and max_corrections = !max_corrections in
  Option.iter arm_deadline !deadline_seconds;
  let device = Me.Device.create_system_default () in
  let queue = Me.CommandQueue.on_device device in
  let options = Me.CompileOptions.init () in
  Me.CompileOptions.set_language_version options Me.CompileOptions.LanguageVersion.version_3_1;
  let library = Me.Library.on_device device ~source options in
  let fn = Me.Library.new_function_with_name library "spin" in
  let pso, _ = Me.ComputePipelineState.on_device_with_function device fn in
  let input = Me.Buffer.on_device device ~length:(1024 * 4) ropts in
  let output_a = Me.Buffer.on_device device ~length:4 ropts in
  let output_b = Me.Buffer.on_device device ~length:4 ropts in
  let iterations_buf = Me.Buffer.on_device device ~length:4 ropts in
  let open Ctypes in
  let input_f = coerce (ptr void) (ptr float) (Me.Buffer.contents input) in
  let output_a_f = coerce (ptr void) (ptr float) (Me.Buffer.contents output_a) in
  let output_b_f = coerce (ptr void) (ptr float) (Me.Buffer.contents output_b) in
  let iterations_u32 = coerce (ptr void) (ptr uint32_t) (Me.Buffer.contents iterations_buf) in
  for i = 0 to 1023 do
    input_f +@ i <-@ Float.of_int ((i mod 29) + 1) /. 29.
  done;
  let set_iterations n =
    iterations_u32 <-@ Unsigned.UInt32.of_int n;
    output_a_f <-@ 0.25;
    output_b_f <-@ 0.5
  in
  let run_one output =
    let cb = command_buffer ~queue ~pso ~input ~output ~iterations_buf () in
    let start = now () in
    Me.CommandBuffer.commit cb;
    Me.CommandBuffer.wait_until_completed cb;
    check_completed "calibration" cb;
    (elapsed_ms start, cb)
  in
  (* Warm compilation/dispatch before asking the calibration clock a question. *)
  set_iterations 1;
  ignore (run_one output_a);
  let timed_single iterations =
    set_iterations iterations;
    let ms, _ = run_one output_a in
    if (not (Float.is_finite ms)) || ms <= 0. then (
      Printf.eprintf "metal_queue_probe: calibration clock returned %.9g ms for %d iterations\n" ms
        iterations;
      exit 1);
    ms
  in
  let rescale iterations ~measured_ms ~target_ms =
    let scaled = Float.ceil (Float.of_int iterations *. target_ms /. measured_ms) |> Int.of_float in
    Int.max 1 (Int.min max_iterations scaled)
  in
  (* [iterations] is what the arms run with; [achieved_ms] is a single kernel measured at exactly
     that count, so the report's achieved figure is never an extrapolation. *)
  let iterations, achieved_ms, corrections =
    match request with
    | Exact_iterations iterations -> (iterations, timed_single iterations, 0)
    | Target_ms target_ms ->
        let seed_iterations = 1 lsl 18 in
        let seed_ms = timed_single seed_iterations in
        (* Feedback loop: each round re-times the current count and rescales by the miss. A fixed
           launch overhead makes the seed's ms/iteration too high, so the first scaled count
           undershoots; rounds after the first converge as the kernel dominates the launch. *)
        let rec correct iterations round =
          let measured_ms = timed_single iterations in
          let off = Float.abs (measured_ms -. target_ms) /. target_ms in
          if
            off <= tolerance || round >= max_corrections
            || (iterations = max_iterations && measured_ms < target_ms)
            || (iterations = 1 && measured_ms > target_ms)
          then (iterations, measured_ms, round)
          else correct (rescale iterations ~measured_ms ~target_ms) (round + 1)
        in
        correct (rescale seed_iterations ~measured_ms:seed_ms ~target_ms) 0
  in
  Printf.printf "device: %s\n" (Me.Device.get_attributes device).name;
  (match request with
  | Target_ms target_ms ->
      Printf.printf "requested: %.3f ms single kernel (tolerance %.1f%%, correction rounds: %d)\n"
        target_ms (tolerance *. 100.) corrections;
      Printf.printf "achieved:  %.3f ms single kernel (%+.2f%% vs requested); iterations: %d\n"
        achieved_ms
        ((achieved_ms -. target_ms) /. target_ms *. 100.)
        iterations;
      if Float.abs (achieved_ms -. target_ms) /. target_ms > tolerance then
        Printf.printf
          "achieved duration is outside tolerance: rerun with --iterations=%d to pin this count, \
           or raise --max-corrections\n"
          iterations
  | Exact_iterations _ ->
      Printf.printf "requested: exact iterations, no duration target; iterations: %d\n" iterations;
      Printf.printf "achieved:  %.3f ms single kernel\n" achieved_ms);

  let sync_between () =
    set_iterations iterations;
    let start = now () in
    let a = command_buffer ~queue ~pso ~input ~output:output_a ~iterations_buf () in
    Me.CommandBuffer.commit a;
    Me.CommandBuffer.wait_until_completed a;
    check_completed "sync-between first" a;
    let b = command_buffer ~queue ~pso ~input ~output:output_b ~iterations_buf () in
    Me.CommandBuffer.commit b;
    Me.CommandBuffer.wait_until_completed b;
    check_completed "sync-between second" b;
    elapsed_ms start
  in
  let raw_queued () =
    set_iterations iterations;
    let a = command_buffer ~queue ~pso ~input ~output:output_a ~iterations_buf () in
    let b = command_buffer ~queue ~pso ~input ~output:output_b ~iterations_buf () in
    let start = now () in
    Me.CommandBuffer.commit a;
    Me.CommandBuffer.commit b;
    (* Both waits are after both submissions; the second is normally already complete. *)
    Me.CommandBuffer.wait_until_completed a;
    Me.CommandBuffer.wait_until_completed b;
    check_completed "raw-queued first" a;
    check_completed "raw-queued second" b;
    elapsed_ms start
  in
  (* [~wait_before]: the wait encoded before the second kernel (the backend's shape) or after it
     (the artifact). Both arms run the same buffers otherwise. *)
  let event_chain_arm ~wait_before label () =
    set_iterations iterations;
    let event = Me.SharedEvent.on_device device in
    let one = Unsigned.ULLong.one in
    let two = Unsigned.ULLong.of_int 2 in
    let a = command_buffer ~queue ~pso ~input ~output:output_a ~iterations_buf () in
    let b =
      if wait_before then
        command_buffer ~wait:(event, one) ~queue ~pso ~input ~output:output_b ~iterations_buf ()
      else begin
        let b = command_buffer ~queue ~pso ~input ~output:output_b ~iterations_buf () in
        Me.CommandBuffer.encode_wait_for_event b (Me.SharedEvent.super event) one;
        b
      end
    in
    let start = now () in
    Me.CommandBuffer.commit a;
    commit_signal queue event one;
    Me.CommandBuffer.commit b;
    commit_signal queue event two;
    let completed =
      Me.SharedEvent.wait_until_signaled_value event ~value:two ~timeout_ms:Unsigned.ULLong.max_int
    in
    if not completed then (
      Printf.eprintf "%s final SharedEvent wait timed out\n" label;
      exit 1);
    Me.CommandBuffer.wait_until_completed a;
    Me.CommandBuffer.wait_until_completed b;
    check_completed (label ^ " first") a;
    check_completed (label ^ " second") b;
    elapsed_ms start
  in
  let event_chain = event_chain_arm ~wait_before:true "event-chain" in
  let wait_after_kernel = event_chain_arm ~wait_before:false "wait-after-kernel" in
  (* Warm every submission shape once, then rotate their measurement order across the passes. *)
  ignore (sync_between ());
  ignore (raw_queued ());
  ignore (event_chain ());
  ignore (wait_after_kernel ());
  let arms =
    [
      ("sync-between", sync_between);
      ("raw-queued", raw_queued);
      ("event-chain", event_chain);
      ("wait-after-kernel", wait_after_kernel);
    ]
  in
  let rotate n values =
    let rec split i left = function
      | rest when i = 0 -> (List.rev left, rest)
      | [] -> (List.rev left, [])
      | value :: rest -> split (i - 1) (value :: left) rest
    in
    let left, right = split n [] values in
    right @ left
  in
  let samples = Hashtbl.create 4 in
  List.iter (fun (label, _) -> Hashtbl.add samples label []) arms;
  for pass = 0 to 3 do
    List.iter
      (fun (label, run) ->
        let ms = run () in
        Hashtbl.replace samples label (ms :: Hashtbl.find samples label))
      (rotate pass arms)
  done;
  let median values =
    let values = Array.of_list values in
    Array.sort Float.compare values;
    values.(Array.length values / 2)
  in
  let baseline = median (Hashtbl.find samples "sync-between") in
  Printf.printf "%-18s %12s %10s\n" "submission" "two-kernel ms" "vs synced";
  List.iter
    (fun (label, _) ->
      let ms = median (Hashtbl.find samples label) in
      Printf.printf "%-18s %12.3f %9.3fx\n" label ms (ms /. baseline))
    arms;
  Printf.printf "outputs: %.9g %.9g\n" !@output_a_f !@output_b_f
