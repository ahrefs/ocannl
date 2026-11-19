(* gh-ocannl-2082, gh-ocannl-2139: what macOS does to a Metal command buffer that does not end.

   Raw Metal, no OCANNL: one cell per process, chosen by the first argument, so that a cell's GPU
   state cannot leak into the next one's except through the OS. Every line goes to stdout, prefixed
   with the cell and the seconds since its commit. [run_cells.sh] beside this file builds it and
   runs cells in order; no alias runs it, since a stall cell needs a Mac with the GPU to itself. The
   GPU recovery count comes from [Gpu_recovery.recovery_count], the one reader the real-stall slow
   tests share: the sum over every accelerator, printed [recoveryCount=unknown (...)] when it cannot
   be read, never as 0. The cells:

   [stall THREADS MAX_S] commits a kernel of THREADS threads (one threadgroup of one thread is what
   OCANNL dispatches for an all-serial kernel), each looping 10^15 times, then a second buffer that
   only signals an event, as OCANNL's [sync] waits on. Polls both buffers' statuses and the event
   every 250 ms, and the recovery count every 2 s, until the kernel's buffer reads a final status or
   MAX_S passes. At 5 s it commits a one-thread kernel on a second queue and reports when that one
   completes: whether the GPU still serves other work under the stall. Exits without waiting for
   anything.

   [spin ITERS MAX_S] runs a bounded kernel of one thread, ITERS dependent multiply-adds, waited for
   up to MAX_S (default 600), and reports its GPU time and status: whether a long legitimate kernel
   is killed.

   [plain] runs one trivial kernel, waited for with a 60 s bound. Run after a stall cell, it says
   whether the GPU came back once the stalling process exited.

   [pulse SECS] commits a trivial kernel every half second for SECS seconds and reports its latency.
   Run beside a stall cell in another process, it says whether the stall starves other processes'
   work, and whether that contention is what makes the OS kill it. *)

module Me = Metal

let source =
  {|
#include <metal_stdlib>
using namespace metal;

kernel void stall(device float *x [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  for (int i = 0; i < 100000; i++)
    for (int j = 0; j < 100000; j++)
      for (int k = 0; k < 100000; k++)
        x[tid] = x[tid] + 1.0f;
}

kernel void spin(device float *x [[buffer(0)]], constant uint &n [[buffer(1)]],
                 uint tid [[thread_position_in_grid]]) {
  float acc = x[tid];
  for (uint i = 0; i < n; i++) acc = fma(acc, 0.99999994f, 1.0f);
  x[tid] = acc;
}

kernel void plain(device float *x [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  x[tid] = x[tid] + 1.0f;
}
|}

let device = Me.Device.create_system_default ()
let library = Me.Library.on_device device ~source (Me.CompileOptions.init ())

(* One pipeline per kernel, made on first use: a pipeline made per launch would put its creation
   inside every latency [pulse] reads. *)
let psos = Hashtbl.create 4

let pso name =
  match Hashtbl.find_opt psos name with
  | Some p -> p
  | None ->
      let p =
        fst
          (Me.ComputePipelineState.on_device_with_function device
             (Me.Library.new_function_with_name library name))
      in
      Hashtbl.add psos name p;
      p

let buffer bytes =
  Me.Buffer.on_device device ~length:bytes
    (Me.ResourceOptions.make ~storage_mode:Me.ResourceOptions.storage_mode_shared ())

let cell = if Array.length Sys.argv > 1 then Sys.argv.(1) else "plain"
let t0 = ref (Unix.gettimeofday ())

let say fmt =
  Printf.ksprintf
    (fun s -> Printf.printf "cell=%s t=%.2f %s\n%!" cell (Unix.gettimeofday () -. !t0) s)
    fmt

let recovery_reading () =
  match Gpu_recovery.recovery_count () with n -> Ok n | exception Failure msg -> Error msg

let render = function
  | Ok n -> Printf.sprintf "recoveryCount=%d" n
  | Error msg -> Printf.sprintf "recoveryCount=unknown (%s)" msg

let recovery () = render (recovery_reading ())

(* The count after a stall, given up to 2 s to rise past [before], as the slow tests give it: the
   IORegistry can count a kill after Metal has already reported the buffer's error. *)
let settled_recovery before =
  let deadline = Unix.gettimeofday () +. 2. in
  let rec go () =
    match (before, recovery_reading ()) with
    | Ok b, Ok n when n <= b && Unix.gettimeofday () < deadline ->
        Unix.sleepf 0.1;
        go ()
    | _, r -> r
  in
  render (go ())

let status cb = Me.CommandBuffer.(Status.sexp_of_t (get_status cb)) |> Sexplib0.Sexp.to_string
let final cb = Me.CommandBuffer.(match get_status cb with Completed | Error -> true | _ -> false)
let error cb = Option.value (Me.CommandBuffer.get_error cb) ~default:"-"

let launch ?(bytes = []) queue ~name ~threads x =
  let cb = Me.CommandBuffer.on_queue queue in
  let enc = Me.ComputeCommandEncoder.on_buffer cb in
  Me.ComputeCommandEncoder.set_compute_pipeline_state enc (pso name);
  Me.ComputeCommandEncoder.set_buffer enc ~index:0 x;
  List.iteri
    (fun i v ->
      let p = Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int v) |> to_voidp) in
      Me.ComputeCommandEncoder.set_bytes enc ~bytes:p ~length:4 ~index:(i + 1))
    bytes;
  if threads < 1 || (threads > 256 && threads mod 256 <> 0) then
    invalid_arg
      (Printf.sprintf "%d threads: give 1 to 256, or a multiple of 256, so the grid is exact"
         threads);
  let per = min threads 256 in
  Me.ComputeCommandEncoder.dispatch_threadgroups enc
    ~threadgroups_per_grid:{ width = (threads + per - 1) / per; height = 1; depth = 1 }
    ~threads_per_threadgroup:{ width = per; height = 1; depth = 1 };
  Me.ComputeCommandEncoder.end_encoding enc;
  Me.CommandBuffer.commit cb;
  cb

let gpu_time cb = Me.CommandBuffer.(get_gpu_end_time cb -. get_gpu_start_time cb)

let wait_bounded ~max_s cb =
  let deadline = Unix.gettimeofday () +. max_s in
  while (not (final cb)) && Unix.gettimeofday () < deadline do
    Unix.sleepf 0.01
  done

let stall threads max_s =
  let queue = Me.CommandQueue.on_device device and side = Me.CommandQueue.on_device device in
  let x = buffer (4 * threads) and y = buffer 32 in
  let event = Me.SharedEvent.on_device device in
  let before = recovery_reading () in
  say "recovery_before %s" (render before);
  t0 := Unix.gettimeofday ();
  let k = launch queue ~name:"stall" ~threads x in
  let s = Me.CommandBuffer.on_queue queue in
  Me.CommandBuffer.encode_signal_event s (Me.SharedEvent.super event) (Unsigned.ULLong.of_int 1);
  Me.CommandBuffer.commit s;
  let side_cb = ref None and side_done = ref false in
  (* The side kernel's state as a status, not a flag: these lines are readings, not verdicts. *)
  let side_state () = match !side_cb with None -> "not-committed" | Some cb -> status cb in
  let last = ref "" and last_rc = ref "" and beat = ref 0. in
  let stop = ref false in
  while not !stop do
    let now = Unix.gettimeofday () -. !t0 in
    let rc =
      if now -. !beat >= 2. then (
        beat := now;
        recovery ())
      else !last_rc
    in
    let line =
      Printf.sprintf "kernel=%s signal=%s event=%s" (status k) (status s)
        (Unsigned.ULLong.to_string (Me.SharedEvent.get_signaled_value event))
    in
    if line <> !last || rc <> !last_rc then say "%s %s" line rc;
    last := line;
    last_rc := rc;
    if !side_cb = None && now >= 5. then (
      side_cb := Some (launch side ~name:"plain" ~threads:1 y);
      say "side kernel committed on a second queue");
    (match !side_cb with
    | Some cb when (not !side_done) && final cb ->
        side_done := true;
        say "side kernel %s gpu=%.4fs error=%s" (status cb) (gpu_time cb) (error cb)
    | _ -> ());
    if final k then (
      say "KERNEL FINAL %s error=%s gpu=%.2fs" (status k) (error k) (gpu_time k);
      wait_bounded ~max_s:2. s;
      say "signal buffer %s error=%s event=%s" (status s) (error s)
        (Unsigned.ULLong.to_string (Me.SharedEvent.get_signaled_value event));
      stop := true)
    else if now >= max_s then (
      say "NOT FINAL after %.0fs kernel=%s side=%s" max_s (status k) (side_state ());
      stop := true)
    else Unix.sleepf 0.25
  done;
  say "recovery_after %s" (settled_recovery before);
  say "RESULT stall threads=%d kernel=%s side=%s" threads (status k) (side_state ())

let spin iters max_s =
  let queue = Me.CommandQueue.on_device device in
  let x = buffer 32 in
  say "recovery_before %s" (recovery ());
  t0 := Unix.gettimeofday ();
  let k = launch queue ~bytes:[ iters ] ~name:"spin" ~threads:1 x in
  wait_bounded ~max_s k;
  if not (final k) then say "NOT FINAL after %.0fs kernel=%s" max_s (status k);
  say "RESULT spin iters=%d %s gpu=%.3fs error=%s %s" iters (status k) (gpu_time k) (error k)
    (recovery ())

let plain () =
  let queue = Me.CommandQueue.on_device device in
  let x = buffer 32 in
  say "recovery_before %s" (recovery ());
  t0 := Unix.gettimeofday ();
  let k = launch queue ~name:"plain" ~threads:1 x in
  wait_bounded ~max_s:60. k;
  say "RESULT plain %s gpu=%.4fs error=%s %s" (status k) (gpu_time k) (error k) (recovery ())

(* Commits a trivial kernel every half second for [secs] seconds, each waited for with a bound of
   the remaining time, and reports its latency: whether another process's stall starves this one. *)
let pulse secs =
  let queue = Me.CommandQueue.on_device device in
  let x = buffer 32 in
  ignore (pso "plain");
  t0 := Unix.gettimeofday ();
  let stop = !t0 +. secs in
  let pending = ref 0 in
  while Unix.gettimeofday () < stop do
    let start = Unix.gettimeofday () in
    let k = launch queue ~name:"plain" ~threads:1 x in
    wait_bounded ~max_s:(stop -. start) k;
    if final k then
      say "pulse %s latency=%.3fs error=%s" (status k) (Unix.gettimeofday () -. start) (error k)
    else (
      incr pending;
      say "pulse NOT FINAL after %.1fs (%s)" (Unix.gettimeofday () -. start) (status k));
    Unix.sleepf 0.5
  done;
  say "RESULT pulse unfinished=%d %s" !pending (recovery ())

(* Each cell runs under [Gpu_recovery.arm_watchdog], a bound from outside the process 60 s past the
   cell's own: a native call stuck on the GPU defers any bound the cell could keep itself. *)
let () =
  let arg i d = if Array.length Sys.argv > i then Sys.argv.(i) else d in
  let bounded seconds run =
    Gpu_recovery.arm_watchdog ~seconds:(int_of_float (Float.ceil seconds) + 60);
    run ()
  in
  match cell with
  | "stall" ->
      let max_s = float_of_string (arg 3 "60") in
      bounded max_s (fun () -> stall (int_of_string (arg 2 "1")) max_s)
  | "spin" ->
      let max_s = float_of_string (arg 3 "600") in
      bounded max_s (fun () -> spin (int_of_string (arg 2 "1000000")) max_s)
  | "plain" -> bounded 60. plain
  | "pulse" ->
      let secs = float_of_string (arg 2 "30") in
      bounded secs (fun () -> pulse secs)
  | other -> failwith ("unknown cell " ^ other)
