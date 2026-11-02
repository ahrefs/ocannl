(* gh-ocannl-1773: a CUDA launch refused device memory for its local-memory reservation.

   The driver backs a kernel's local memory (its [__local_depot]: [Local] arrays, spills) for every
   thread the device can keep resident, whatever the launch geometry, and keeps that reservation for
   the context's life. So a one-thread kernel with a 48 KiB frame -- the unscheduled
   [schedule_strided_1x1] resnet-block reference -- reserves 46 x 1536 x 48 KiB, 3.3 GiB, on
   rog-nv's RTX 5070 Ti, and under a concurrent GPU batch [cu_launch_kernel] answered
   [CUDA_ERROR_OUT_OF_MEMORY]. The backend now backs off and retries such a launch, unless the
   reservation exceeds the device's total memory. Both outcomes are forced here through the real
   driver, with no injection.

   The default run is the hopeless case: a frame whose reservation exceeds the device. The launch is
   refused without a retry, contained by [Schedule_outcome.protect] as the [No_device_writes]
   decline a search continues past, and the context still runs a following kernel correctly. A
   refused reservation holds nothing, so this run is safe beside other GPU batches.

   [--contended] is the retried case, and it needs the device to itself, so no alias runs it: a
   child process (this executable, re-run with [--hold]) takes nearly all the free memory with a
   reservation of its own -- the flake's mechanism -- for two seconds, which would refuse every
   neighbour's allocation meanwhile. A launch whose reservation fits the device but not what the
   child left free is retried, and runs once the child is gone, computing the right values. Run it
   as [OCANNL_BACKEND=cuda dune build test/operations/cuda_launch_reservation.exe], then
   [./cuda_launch_reservation.exe --contended] in [_build/default/test/operations], under an
   exclusive hold of the box; a false claim exits 1.

   The executed kernels discriminate: the frame is filled forward with [1 + x] and read back in
   REVERSE ([out[y] = n - y]), which also keeps the device compiler from forwarding each store to
   its load and deleting the array. The retry and refusal counts are read from the backend's global
   debug info; the sizes and timings vary with the device and its load, so they go to stderr. *)

open Base
open Ll_test
open Verdict.Claims
module SO = Ir.Schedule_outcome

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_cuda = String.equal backend_name "cuda"

(* Near the 512 KiB per-thread local-memory limit: reserved for every resident thread, more than any
   current device holds (rog-nv: 46 x 1536 threads, 32 GiB against 12 GiB). *)
let hopeless_floats = 120_000

(* A one-thread kernel whose frame is a [Local] array of [n] floats, filled forward and drained in
   reverse into the materialized [out]. *)
let frame_kernel ~first_id ~name n =
  let mk = node_factory ~first_id ~dims:[| n |] () in
  let out = mk "clr_out" in
  materialize out;
  let frame = mk "clr_frame" in
  Tn.update_memory_mode frame Tn.Local (Site "99:test-setup");
  let x = sym () and y = sym () in
  let llc =
    seq
      (loop_n x n (set frame [| iter x |] (tick x)))
      (loop_n y n (set out [| iter y |] (get frame [| aff [ (-1, y) ] (n - 1) |])))
  in
  (optimize ~materialized:[ out ] ~name llc, out)

let expected n = Array.init n ~f:(fun y -> Float.of_int (n - y))

let run_frame ?ctx ~first_id ~name n =
  let o, out = frame_kernel ~first_id ~name n in
  match execute ?ctx ~name o ~seed:[ (out, blank n) ] ~read:[ out ] with
  | [ got ] -> got
  | _ -> assert false

(* The child: take [floats]-per-thread frame's reservation, say so, hold it for [seconds], exit. *)
let () =
  match Sys.get_argv () with
  | [| _; "--hold"; floats; seconds |] ->
      ignore (run_frame ~first_id:1773900 ~name:"clr_hold" (Int.of_string floats) : float array);
      Stdio.printf "held\n%!";
      Unix.sleepf (Float.of_string seconds);
      Stdlib.exit 0
  | _ -> ()

let counter ctx field =
  match Context.global_debug_info ctx with
  | Sexp.List (_ :: fields) ->
      List.find_map fields ~f:(function
        | Sexp.List [ Sexp.Atom f; Sexp.Atom n ] when String.equal f field -> Int.of_string_opt n
        | _ -> None)
      |> Option.value ~default:0
  | _ -> 0

let () =
  let ctx = Context.auto () in
  Stdio.eprintf "backend: %s (not part of the golden)\n%!" (Context.backend_name ctx);
  let cuda = Verdict.leg ~when_:on_cuda ~on:backend_name () in
  let resident =
    lazy
      (Option.value ~default:0 (Context.hardware_limits ctx).Ir.Backend_intf.max_resident_threads)
  in
  let free_and_total () = Option.value ~default:(0, 0) (Context.get_free_and_total_memory ctx) in

  let contended_mode = match Sys.get_argv () with [| _; "--contended" |] -> true | _ -> false in
  (* --- 1. Hopeless: refused at once, contained, and the context runs on. --- *)
  if not contended_mode then (
    let hopeless =
      Verdict.sub_leg cuda ~on:backend_name
        ~when_:
          (lazy
            (let _, total = free_and_total () in
             let r = hopeless_floats * 4 * Lazy.force resident in
             Stdio.eprintf
               "hopeless: reservation %d B against %d B total (not part of the golden)\n%!" r total;
             r > total))
    in
    let outcome =
      lazy
        (let retries = counter ctx "launch_memory_retries"
         and refusals = counter ctx "launch_memory_refusals" in
         let o, out = frame_kernel ~first_id:1773000 ~name:"clr_hopeless" hopeless_floats in
         let linked = link ~name:"clr_hopeless" o in
         let r =
           SO.protect ~classify_backend:(Context.failure_classifier ctx)
             ~provenance:SO.User_schedule ~phase:SO.Launch (fun () ->
               Context.get_values (run_linked linked ~seed:[ (out, blank hopeless_floats) ]) out)
         in
         ( r,
           counter ctx "launch_memory_retries" - retries,
           counter ctx "launch_memory_refusals" - refusals ))
    in
    in_leg ~leg:hopeless
      "hopeless: the launch is declined as the driver's out-of-memory refusal, with no device \
       writes" (fun label ->
        p label
          (match Lazy.force outcome with
          | ( Error
                (SO.Classified
                   {
                     phase = SO.Launch;
                     cause = SO.Backend_rejected { stage = "CUDA_ERROR_OUT_OF_MEMORY"; _ };
                     execution_effect = SO.No_device_writes;
                     _;
                   }),
              _,
              _ ) ->
              true
          | _ -> false));
    in_leg ~leg:hopeless "hopeless: refused once, without a retry" (fun label ->
        let _, retries, refusals = Lazy.force outcome in
        p label (retries = 0 && refusals = 1));
    in_leg ~leg:hopeless "hopeless: the context then runs a small-frame kernel correctly"
      (fun label ->
        let n = 256 in
        p label
          (Array.equal Float.equal
             (run_frame ~ctx ~first_id:1773100 ~name:"clr_after" n)
             (expected n))));

  (* --- 2. Contended: retried while a neighbour holds the memory, run once it is gone. --- *)
  if contended_mode then (
    let free0, _ = if on_cuda then free_and_total () else (0, 0) in
    (* A reservation of a third of what is free, at most 2 GiB; the child leaves a quarter of it. *)
    let floats =
      if not on_cuda then 0
      else min hopeless_floats (min (free0 / 3) (1 lsl 31) / 4 / max 1 (Lazy.force resident))
    in
    let reservation = floats * 4 * if on_cuda then Lazy.force resident else 0 in
    let child_floats =
      if not on_cuda then 0
      else min hopeless_floats ((free0 - (reservation / 4)) / 4 / max 1 (Lazy.force resident))
    in
    let child =
      lazy
        (let read_end, write_end = Unix.pipe ~cloexec:true () in
         let pid =
           Unix.create_process Stdlib.Sys.executable_name
             [| Stdlib.Sys.executable_name; "--hold"; Int.to_string child_floats; "2.0" |]
             Unix.stdin write_end Unix.stderr
         in
         Unix.close write_end;
         let line = Stdio.In_channel.input_line (Unix.in_channel_of_descr read_end) in
         let free1, _ = free_and_total () in
         Stdio.eprintf
           "contended: %d floats per thread (%d B), the child holds %d floats per thread, %d B \
            free before it, %d B with it (not part of the golden)\n\
            %!"
           floats reservation child_floats free0 free1;
         (pid, Option.equal String.equal line (Some "held"), free1))
    in
    let contended =
      Verdict.sub_leg cuda ~on:backend_name
        ~when_:
          (lazy
            (floats >= 1024
            &&
            let _, held, free1 = Lazy.force child in
            held && free1 < reservation))
    in
    let contended_run =
      lazy
        (let retries = counter ctx "launch_memory_retries"
         and refusals = counter ctx "launch_memory_refusals" in
         let start = Unix.gettimeofday () in
         let got = run_frame ~ctx ~first_id:1773200 ~name:"clr_contended" floats in
         Stdio.eprintf "contended: ran after %.1fs (not part of the golden)\n%!"
           (Unix.gettimeofday () -. start);
         ( got,
           counter ctx "launch_memory_retries" - retries,
           counter ctx "launch_memory_refusals" - refusals ))
    in
    in_leg ~leg:contended "contended: the launch is retried, and not given up on" (fun label ->
        let _, retries, refusals = Lazy.force contended_run in
        p label (retries >= 1 && refusals = 0));
    in_leg ~leg:contended "contended: the retried kernel computes the right values" (fun label ->
        let got, _, _ = Lazy.force contended_run in
        p label (Array.equal Float.equal got (expected floats)));
    if Lazy.is_val child then
      let pid, _, _ = Lazy.force child in
      ignore (Unix.waitpid [] pid : int * Unix.process_status))
