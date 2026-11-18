(* gh-ocannl-1773, gh-ocannl-1872: a launch's local-memory reservation -- refused under contention,
   and held for the process's life.

   The driver backs a kernel's local memory (its [__local_depot]: [Local] arrays, spills) for every
   thread the device can keep resident, whatever the launch geometry. So a one-thread kernel with a
   48 KiB frame -- the unscheduled [schedule_strided_1x1] resnet-block reference, s11d_ref --
   reserves 46 x 1536 x 48 KiB, 3.3 GiB, on rog-nv's RTX 5070 Ti. Under a concurrent GPU batch
   [cu_launch_kernel] answered [CUDA_ERROR_OUT_OF_MEMORY], and the CUDA backend now backs off and
   retries such a launch, plain or as a graph, unless the reservation exceeds the device's total
   memory (gh-ocannl-1773). And the driver used to KEEP a grown reservation until the process
   exited, so one such launch starved every neighbour for the rest of the run: after a launch whose
   reservation passes a sixteenth of device memory, the backend now sets the context's stack-size
   limit back, which releases it (gh-ocannl-1872). The HIP backend backs off the same way, with no
   reset: HIP budgets scratch per work-item and declines an over-budget frame at link. Everything is
   forced through the real driver, with no injection.

   The default run, on CUDA. Hopeless: a frame whose reservation exceeds the device is refused
   without a retry, contained by [Schedule_outcome.protect] as the [No_device_writes] decline a
   search continues past, and the context still runs a following kernel correctly; a refused
   reservation holds nothing. Bound: an s11d_ref-sized frame runs, and the device's free memory
   after it is back within half the reservation of where it was before (on the base it stayed a
   whole reservation lower), while a small frame resets nothing, so ordinary launches stay
   asynchronous. These depot witnesses launch two blocks, so their frame remains thread-private. The
   serial witness uses three simultaneously live 16 KiB arrays and checks that its launch avoids the
   old reservation, computes the right values on reuse, and works in a cached graph. Graph: the
   two-block depot frame as one segment of a two-segment routine, which [sequence_segments] captures
   as a CUDA graph, whose launch releases the reservation too. A hopeless frame in a graph is
   refused at the graph's instantiation, where its kernels first reserve (measured on rog-nv), at
   once, and then as the plain launch the routine falls back to. The default run, on HIP: the
   scratch frame runs, and the back-off's counters are reported and untouched; the free memory
   around it goes to stderr, for the record.

   [--contended] is the retried case, and it needs the device to itself, so no alias runs it: a
   child process (this executable, re-run with [--hold]) takes nearly all the free memory in device
   buffers for two seconds, which would refuse every neighbour's allocation meanwhile. A launch
   whose reservation fits the device but not what the child left free is retried, and runs once the
   child is gone, computing the right values, on CUDA: a plain launch, a graph's first run (its
   instantiation), and a cached graph's replay, which regrows the released reservation -- a refused
   graph launch drops its exec and runs the segments plainly, never relaunching it (a relaunch
   faulted on rog-nv). On HIP the same holds are not refused at all (ROCr backs scratch on demand),
   which the run pins instead: HIP's retry path is not reached. Run it as [OCANNL_BACKEND=cuda dune
   build test/operations/cuda_launch_reservation.exe] (or [hip]), then
   [./cuda_launch_reservation.exe --contended] in [_build/default/test/operations], under an
   exclusive hold of the box; a false claim exits 1.

   [--pool-backing] is the CUDA graph memory probe. Under an exclusive hold, run it from the same
   build directory. Every sampled graph has completed. The probe checks that changing the binding
   fills the cache and then evicts a live exec, and that routine release removes the remaining
   execs. Its free-memory readings assume an idle device. It never requests a driver trim.

   The executed kernels discriminate: the frame is filled forward with [1 + x] and read back in
   REVERSE ([out[y] = n - y]), which also keeps the device compiler from forwarding each store to
   its load and deleting the array; the graph's second segment adds one. The counters are read from
   the backend's global debug info; the sizes, free memory and timings vary with the device and its
   load, so they go to stderr. *)

open Base
open Ll_test
open Verdict.Claims
module SO = Ir.Schedule_outcome
module FI = Ir.Resource_fault_injection

exception Scratch_fault

module Generated = Test_utils.Generated

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let on_cuda = String.equal backend_name "cuda"
let on_hip = String.equal backend_name "hip"

(* Near the 512 KiB per-thread local-memory limit: reserved for every resident thread, more than any
   current device holds (rog-nv: 46 x 1536 threads, 32 GiB against 12 GiB). *)
let hopeless_floats = 120_000

(* s11d_ref's frame: three 16 KiB [Local] nodes, 48 KiB for its one thread. *)
let s11d_floats = 12_288

(* A one-thread kernel whose frame is a [Local] array of [n] floats, filled forward and drained in
   reverse into the materialized [out]. With [~graph], a second statement adds one to [out] into a
   materialized [next], and the routine is cut into two segments there, so a GPU backend captures it
   as a graph: [n - y + 1] then. Returns the program and the node to read. *)
let frame_program ~first_id ~name ?(graph = false) ?(multi = false) n =
  let outputs = if multi then 2 * n else n in
  let mk_output = node_factory ~first_id ~dims:[| outputs |] () in
  let out = mk_output "clr_out" in
  materialize out;
  let frame = node_factory ~first_id:(first_id + 2) ~dims:[| n |] () "clr_frame" in
  Tn.update_memory_mode frame Tn.Local (Site "99:test-setup");
  let x = sym () and y = sym () and block = sym () in
  let offset = if multi then aff [ (n, block); (1, y) ] 0 else iter y in
  let fill_drain =
    seq
      (loop_n x n (set frame [| iter x |] (tick x)))
      (loop_n y n (set out [| offset |] (get frame [| aff [ (-1, y) ] (n - 1) |])))
  in
  let fill_drain = if multi then loop_n ~axis:Grid block 2 fill_drain else fill_drain in
  if not graph then (optimize ~materialized:[ out ] ~name fill_drain, out)
  else
    let next = mk_output "clr_next" in
    materialize next;
    let z = sym () in
    let llc =
      seq fill_drain (loop_n z outputs (set next [| iter z |] (add (c 1.) (get out [| iter z |]))))
    in
    (optimize ~materialized:[ out; next ] ~name llc, next)

(* s11d_ref has three simultaneously live arrays. Different values keep overlapping slices visible,
   and [c_frame] reads its declaration's zero on each invocation. *)
let serial_frame_program ?index ~first_id ~name ~graph n =
  let mk = node_factory ~first_id ~dims:[| n |] () in
  let a = mk "serial_a" and b = mk "serial_b" and c_frame = mk "serial_c" in
  List.iter [ a; b; c_frame ] ~f:(fun tn ->
      Tn.update_memory_mode tn Tn.Local (Site "99:test-setup"));
  let out = mk "serial_out" in
  materialize out;
  let x = sym () and y = sym () in
  let value = tick x and forward = [| iter x |] and reverse = [| aff [ (-1, y) ] (n - 1) |] in
  let fill =
    loop_n x n
      (seq (set a forward value)
         (seq
            (set b forward (mul (c 2.) value))
            (set c_frame forward (add (get c_frame forward) (mul (c 4.) value)))))
  in
  let drain =
    loop_n y n
      (set out [| iter y |] (add (get a reverse) (add (get b reverse) (get c_frame reverse))))
  in
  let frame = seq (zero c_frame) (seq fill drain) in
  if not graph then (optimize ~materialized:[ out ] ~name frame, out)
  else
    let next = mk "serial_next" in
    materialize next;
    let z = sym () in
    let offset =
      match index with None -> c 1. | Some (s : Idx.static_symbol) -> tick s.static_symbol
    in
    let consume = loop_n z n (set next [| iter z |] (add (get out [| iter z |]) offset)) in
    ( optimize ~materialized:[ out; next ] ~static_indices:(Option.to_list index) ~name
        (seq frame consume),
      next )

(* The graph routine's two segments: the frame's fill and drain, then the consumer. A recorded
   segmentation is applied as given, never coalesced, so the cut is the test's. *)
let two_segments ~static_indices (o : Ir.Low_level.optimized) =
  let frame_statements = List.length (Ir.Low_level.flat_lines [ o.llc ]) - 1 in
  let _, tuples =
    Ir.Schedule.fission_segmented
      ~replay:([ (`Solo, frame_statements); (`Solo, 1) ], fun _ _ -> None)
      ~preset:(fun _ -> [])
      ~zero_sched:(fun _ -> [])
      ~static_indices o
  in
  List.map tuples ~f:(fun (_, _, _, post) -> post)

let link_program ?ctx ?(bindings = Idx.Empty) ~name ~graph (o, out) =
  let linked =
    if not graph then link ?ctx ~bindings ~name o
    else
      let ctx = match ctx with Some ctx -> ctx | None -> Lazy.force base_ctx in
      let static_indices = Idx.bound_symbols bindings in
      Context.compile ~name ~prelowered:o ~lowered_transform:(two_segments ~static_indices) ctx
        Ir.Assignments.empty_comp bindings
  in
  (linked, out)

let expected ?(graph = false) ?(multi = false) n =
  Array.init
    (if multi then 2 * n else n)
    ~f:(fun y -> Float.of_int (n - (y % n) + if graph then 1 else 0))

let run_frame ?ctx ?(graph = false) ?(multi = false) ~first_id ~name n =
  let (ctx, routine), out =
    link_program ?ctx ~name ~graph (frame_program ~first_id ~name ~graph ~multi n)
  in
  Context.get_values (run_linked (ctx, routine) ~seed:[]) out

(* The child: take [bytes] of device memory in buffers (1 GiB nodes; linking allocates them), say
   so, hold it for [seconds], exit. *)
let () =
  match Sys.get_argv () with
  | [| _; "--hold"; bytes; seconds |] ->
      let chunk = 1 lsl 28 in
      let floats = Int.of_string bytes / 4 in
      let dims =
        List.init ((floats + chunk - 1) / chunk) ~f:(fun i -> min chunk (floats - (i * chunk)))
      in
      let nodes =
        List.filter_mapi dims ~f:(fun i d ->
            if d <= 0 then None
            else
              let tn = (node_factory ~first_id:(1773900 + (10 * i)) ~dims:[| d |] ()) "clr_hold" in
              materialize tn;
              Some tn)
      in
      (* One loop per node, each minting its own symbol: a routine binds a symbol once. *)
      let llc =
        List.map nodes ~f:(fun tn ->
            let x = sym () in
            loop_n x 1 (set tn [| iter x |] (c 1.)))
        |> List.reduce_exn ~f:seq
      in
      let _held = link ~name:"clr_hold" (optimize ~materialized:nodes ~name:"clr_hold" llc) in
      Stdio.printf "held\n%!";
      Unix.sleepf (Float.of_string seconds);
      Stdlib.exit 0
  | _ -> ()

let counter = debug_counter
let count ctx field = Option.value ~default:0 (counter ctx field)
let serial_cells = s11d_floats / 3

let serial_expected ?(binding = 0) ~graph n =
  Array.init n ~f:(fun y -> Float.of_int ((7 * (n - y)) + if graph then binding + 1 else 0))

let stream_pool_reserved ctx = counter ctx "stream_pool_reserved_bytes"
let graph_pool_reserved ctx = counter ctx "graph_pool_reserved_bytes"

let graph_pool_probe ctx =
  if not on_cuda then failwith "--pool-backing requires CUDA";
  let name = "clr_graph_pool_probe" in
  let max_bindings = 1024 in
  let index, bindings = Idx.get_static_symbol ~static_range:max_bindings Idx.Empty in
  let (compiled, routine), out =
    link_program ~ctx ~bindings ~name ~graph:true
      (serial_frame_program ~index ~first_id:1774500 ~name ~graph:true serial_cells)
  in
  let baseline = count ctx "live_graph_execs" in
  let sample phase =
    let free, _ = Option.value_exn (Context.get_free_and_total_memory ctx) in
    Stdio.eprintf
      "%s: free=%d B, stream reserved=%d B, graph reserved=%d B, live execs=%d (not part of the \
       golden)\n\
       %!"
      phase free
      (Option.value_exn (stream_pool_reserved ctx))
      (Option.value_exn (graph_pool_reserved ctx))
      (count ctx "live_graph_execs")
  in
  let launch i current =
    Idx.find_exn routine.Context.bindings index := i;
    let captures = ref 0 in
    let callback = function FI.Cuda_scratch_allocated -> Int.incr captures | _ -> () in
    let ran = FI.with_callback callback ~f:(fun () -> Context.run current routine) in
    Context.sync ran;
    if !captures <> 1 then failwith "expected a new scratch graph for this binding";
    ran
  in
  sample "after link";
  let first = launch 0 compiled in
  if count ctx "live_graph_execs" <> baseline + 1 then failwith "first graph was not cached";
  sample "first graph completed";
  let rec evict i current cached =
    if i >= max_bindings then failwith "graph cache did not evict";
    let ran = launch i current in
    let now = count ctx "live_graph_execs" in
    if now = cached then (i, ran)
    else if now = cached + 1 then evict (i + 1) ran now
    else failwith "unexpected live graph count"
  in
  let i, ran = evict 1 first (baseline + 1) in
  if
    not
      (Array.equal Float.equal (Context.get_values ran out)
         (serial_expected ~binding:i ~graph:true serial_cells))
  then failwith "eviction launch produced the wrong output";
  sample "replacement graph completed after eviction";
  Context.release ran;
  if count ctx "live_graph_execs" <> baseline then failwith "release left cached graph execs";
  sample "after routine release"

(* What [f] changed the named counters by. *)
let deltas ctx fields f =
  let before = List.map fields ~f:(count ctx) in
  let r = f () in
  (r, List.map2_exn fields before ~f:(fun field b -> count ctx field - b))

let () =
  let ctx = Context.auto () in
  if on_cuda then (
    Utils.settings.output_debug_files_in_build_directory <- true;
    Generated.init ~backend_name);
  Stdio.eprintf "backend: %s (not part of the golden)\n%!" (Context.backend_name ctx);
  (match Sys.get_argv () with
  | [| _; "--pool-backing" |] ->
      graph_pool_probe ctx;
      Stdlib.exit 0
  | _ -> ());
  let cuda = Verdict.leg ~when_:on_cuda ~on:backend_name () in
  let hip = Verdict.leg ~when_:on_hip ~on:backend_name () in
  let gpu = Verdict.leg ~when_:(on_cuda || on_hip) ~on:backend_name () in
  (* Graph claims need [sequence_segments] to capture: it skips capture under
     [gpu_graph_capture=false] and while routines log, as [graph_exec_release] gates too. *)
  let capturing =
    lazy
      (Utils.get_global_flag ~default:true ~arg_name:"gpu_graph_capture"
      && not (Utils.debug_log_from_routines ()))
  in
  let resident =
    lazy
      (Option.value ~default:0 (Context.hardware_limits ctx).Ir.Backend_intf.max_resident_threads)
  in
  let free_and_total () = Option.value ~default:(0, 0) (Context.get_free_and_total_memory ctx) in
  let contended_mode = match Sys.get_argv () with [| _; "--contended" |] -> true | _ -> false in
  let backoff = [ "launch_memory_retries"; "launch_memory_refusals" ] in

  (* --- 1. Hopeless: refused at once, contained, and the context runs on. --- *)
  let declined_oom = function
    | Error
        (SO.Classified
           {
             phase = SO.Launch;
             cause = SO.Backend_rejected { stage = "CUDA_ERROR_OUT_OF_MEMORY"; _ };
             execution_effect = SO.No_device_writes;
             _;
           }) ->
        true
    | _ -> false
  in
  let hopeless_gate =
    lazy
      (let _, total = free_and_total () in
       let r = hopeless_floats * 4 * Lazy.force resident in
       Stdio.eprintf "hopeless: reservation %d B against %d B total (not part of the golden)\n%!" r
         total;
       r > total)
  in
  let hopeless_run ~graph ~first_id ~name =
    lazy
      (deltas ctx backoff (fun () ->
           let linked, out =
             link_program ~name ~graph
               (frame_program ~first_id ~name ~graph ~multi:true hopeless_floats)
           in
           let r =
             SO.protect ~classify_backend:(Context.failure_classifier ctx)
               ~provenance:SO.User_schedule ~phase:SO.Launch (fun () ->
                 Context.get_values (run_linked linked ~seed:[]) out)
           in
           (match r with
           | Error (SO.Classified { cause = SO.Backend_rejected { detail; _ }; _ }) ->
               Stdio.eprintf "%s: declined, %s (not part of the golden)\n%!" name detail
           | Error _ -> Stdio.eprintf "%s: failed otherwise (not part of the golden)\n%!" name
           | Ok _ -> ());
           r))
  in
  if not contended_mode then (
    let hopeless = Verdict.sub_leg cuda ~on:backend_name ~when_:hopeless_gate in
    let outcome = hopeless_run ~graph:false ~first_id:1773000 ~name:"clr_hopeless" in
    in_leg ~leg:hopeless
      "hopeless: the launch is declined as the driver's out-of-memory refusal, with no device \
       writes" (fun label -> p label (declined_oom (fst (Lazy.force outcome))));
    in_leg ~leg:hopeless "hopeless: refused once, without a retry" (fun label ->
        p label (List.equal Int.equal (snd (Lazy.force outcome)) [ 0; 1 ]));
    in_leg ~leg:hopeless "hopeless: the context then runs a small-frame kernel correctly"
      (fun label ->
        let n = 256 in
        p label
          (Array.equal Float.equal
             (run_frame ~ctx ~first_id:1773100 ~name:"clr_after" n)
             (expected n)));

    (* --- 2. Bound: an s11d_ref-sized frame's reservation is released after its launch. --- *)
    let reservation = lazy (s11d_floats * 4 * Lazy.force resident) in
    (* The reset's threshold, [total memory / share], with the share the backend reports (its
       [local_memory_reset_share]), so no claim here restates the gate. *)
    let threshold =
      lazy
        (let _, total = free_and_total () in
         Option.map (counter ctx "local_memory_reset_share") ~f:(fun share -> total / share))
    in
    let fits =
      lazy
        (let free, total = free_and_total () in
         let r = Lazy.force reservation in
         Stdio.eprintf
           "bound: an s11d_ref-sized frame reserves %d B; %d B free of %d B, the reset threshold \
            %s B (not part of the golden)\n\
            %!"
           r free total
           (Option.value_map (Lazy.force threshold) ~default:"-" ~f:Int.to_string);
         match Lazy.force threshold with
         | Some threshold -> r > threshold && free > r + (r / 2)
         | None -> false)
    in
    let bound = Verdict.sub_leg cuda ~on:backend_name ~when_:fits in
    let measured ~graph ~first_id ~name =
      lazy
        (let free0, _ = free_and_total () in
         let got, d =
           deltas ctx [ "local_memory_resets" ] (fun () ->
               run_frame ~ctx ~graph ~multi:true ~first_id ~name s11d_floats)
         in
         let free1, _ = free_and_total () in
         Stdio.eprintf
           "%s: free %d B before the launch, %d B after it, a change of %d B against the %d B \
            reservation (not part of the golden)\n\
            %!"
           name free0 free1 (free1 - free0) (Lazy.force reservation);
         (got, free0, free1, d))
    in
    (* Two-sided: the free memory after the launch neither stays a reservation lower (the base kept
       it) nor rises past what the launch could have taken. Off by more, [within] names both values
       and the bound on stderr. *)
    let returned label (_, free0, free1, _) =
      Verdict.within label
        [| Float.of_int free1 |]
        [| Float.of_int free0 |]
        ~tol:(fun _ -> Float.of_int (Lazy.force reservation / 2))
    in
    let plain = measured ~graph:false ~first_id:1773300 ~name:"clr_s11d" in
    in_leg ~leg:bound
      "bound: a two-block kernel with a 48 KiB frame per thread computes the right values"
      (fun label ->
        let got, _, _, _ = Lazy.force plain in
        p label (Array.equal Float.equal got (expected ~multi:true s11d_floats)));
    in_leg ~leg:bound
      "bound: the device's free memory after its launch is within half its reservation of before it"
      (fun label -> p label (returned label (Lazy.force plain)));
    in_leg ~leg:bound "bound: its launch released the reservation once, and a small frame's none"
      (fun label ->
        let _, _, _, d = Lazy.force plain in
        let _, small =
          deltas ctx [ "local_memory_resets" ] (fun () ->
              run_frame ~ctx ~first_id:1773400 ~name:"clr_small" 256)
        in
        p label (List.equal Int.equal d [ 1 ] && List.equal Int.equal small [ 0 ]));
    (* The gate's boundary, from both sides: frames sized a quarter above and a quarter below the
       threshold, so the share cannot drift by more than that with every claim staying green. *)
    in_leg ~leg:bound
      "bound: a frame reserving a quarter more than the reset threshold resets, one a quarter less \
       does not" (fun label ->
        let threshold = Option.value_exn (Lazy.force threshold) in
        let floats_for bytes = bytes / 4 / Lazy.force resident in
        let resets_of ~first_id ~name n =
          snd
            (deltas ctx [ "local_memory_resets" ] (fun () ->
                 Array.equal Float.equal
                   (run_frame ~ctx ~multi:true ~first_id ~name n)
                   (expected ~multi:true n)))
        in
        let above = floats_for (threshold + (threshold / 4))
        and below = floats_for (threshold - (threshold / 4)) in
        Stdio.eprintf
          "bound: the threshold is %d B: %d floats per thread above it, %d below it (not part of \
           the golden)\n\
           %!"
          threshold above below;
        p label
          (List.equal Int.equal (resets_of ~first_id:1773420 ~name:"clr_above" above) [ 1 ]
          && List.equal Int.equal (resets_of ~first_id:1773440 ~name:"clr_below" below) [ 0 ]));

    (* --- 3. Graph: the same, launched as a captured graph. --- *)
    let graph_bound = Verdict.sub_leg bound ~when_:capturing ~on:backend_name in
    let graph_hopeless = Verdict.sub_leg hopeless ~when_:capturing ~on:backend_name in
    let graph_run =
      lazy
        (let r, d =
           deltas ctx [ "live_graph_execs" ] (fun () ->
               Lazy.force (measured ~graph:true ~first_id:1773500 ~name:"clr_s11d_graph"))
         in
         (r, d))
    in
    in_leg ~leg:graph_bound "graph: the two-segment routine runs as a captured graph" (fun label ->
        p label (List.equal Int.equal (snd (Lazy.force graph_run)) [ 1 ]));
    in_leg ~leg:graph_bound "graph: it computes the right values" (fun label ->
        let (got, _, _, _), _ = Lazy.force graph_run in
        p label (Array.equal Float.equal got (expected ~graph:true ~multi:true s11d_floats)));
    in_leg ~leg:graph_bound
      "graph: the free memory after the graph launch is within half the reservation of before it, \
       released once" (fun label ->
        let ((_, _, _, d) as r), _ = Lazy.force graph_run in
        p label (returned label r && List.equal Int.equal d [ 1 ]));
    let hopeless_graph = hopeless_run ~graph:true ~first_id:1773600 ~name:"clr_hopeless_graph" in
    in_leg ~leg:graph_hopeless
      "graph: a hopeless frame in a graph is refused at its instantiation, then as its plain \
       launch, and declined as the driver's out-of-memory refusal, without a retry" (fun label ->
        (* Two refusals, no retry: the instantiation's back-off gives up at once and the routine
           runs its segments plainly, where the kernel launch's gives up too. *)
        let r, d = Lazy.force hopeless_graph in
        p label (declined_oom r && List.equal Int.equal d [ 0; 2 ]));

    let serial_scratch =
      let supported =
        lazy
          (Option.value_map (counter ctx "serial_device_scratch_budget") ~default:true
             ~f:(fun bytes -> bytes > 0))
      in
      Verdict.sub_leg cuda ~when_:supported ~on:backend_name
    in
    let serial_graph = Verdict.sub_leg serial_scratch ~when_:capturing ~on:backend_name in
    in_leg ~leg:serial_scratch "serial scratch: CUDA reports stream backing until synchronization"
      (fun label ->
        let name = "clr_stream_backing" in
        let (compiled, routine), _ =
          link_program ~ctx ~name ~graph:false
            (serial_frame_program ~first_id:1774520 ~name ~graph:false serial_cells)
        in
        let before = stream_pool_reserved ctx in
        let allocated = ref None in
        let callback = function
          | FI.Cuda_scratch_allocated -> allocated := stream_pool_reserved ctx
          | _ -> ()
        in
        let ran = FI.with_callback callback ~f:(fun () -> Context.run compiled routine) in
        Context.sync ran;
        let synced = stream_pool_reserved ctx in
        Context.release ran;
        let show = Option.value_map ~default:"unavailable" ~f:Int.to_string in
        Stdio.eprintf
          "stream backing in bytes before/allocated/synchronized = %s, %s, %s (not part of the \
           golden)\n\
           %!"
          (show before) (show !allocated) (show synced);
        p label
          (match (before, !allocated, synced) with
          | Some before, Some allocated, Some synced ->
              allocated - before >= s11d_floats * 4
              && Verdict.within label
                   [| Float.of_int synced |]
                   [| Float.of_int before |]
                   ~tol:(fun _ -> 0.)
          | _ -> false));
    in_leg ~leg:serial_graph "serial scratch: CUDA reports graph backing across routine release"
      (fun label ->
        let name = "clr_graph_backing" in
        let (compiled, routine), _ =
          link_program ~ctx ~name ~graph:true
            (serial_frame_program ~first_id:1774510 ~name ~graph:true serial_cells)
        in
        let before = graph_pool_reserved ctx in
        let stream_before = stream_pool_reserved ctx in
        let ran = Context.run compiled routine in
        Context.sync ran;
        let cached = graph_pool_reserved ctx in
        Context.release ran;
        let released = graph_pool_reserved ctx in
        let stream_after = stream_pool_reserved ctx in
        let readings =
          List.map
            [ before; cached; released; stream_before; stream_after ]
            ~f:(Option.value_map ~default:"unavailable" ~f:Int.to_string)
        in
        Stdio.eprintf
          "pool backing in bytes: graph before/cached/released, stream before/after = %s (not part \
           of the golden)\n\
           %!"
          (String.concat ~sep:", " readings);
        p label
          (match (before, cached, released, stream_before, stream_after) with
          | Some before, Some cached, Some released, Some stream_before, Some stream_after ->
              cached - before >= s11d_floats * 4
              && Verdict.within label
                   [| Float.of_int released; Float.of_int stream_after |]
                   [| Float.of_int cached; Float.of_int stream_before |]
                   ~tol:(fun _ -> 0.)
          | _ -> false));
    let serial_measurement ~graph ~first_id ~name =
      lazy
        (let n = serial_cells in
         let linked, out =
           link_program ~ctx ~name ~graph (serial_frame_program ~first_id ~name ~graph n)
         in
         let free0, _ = free_and_total () in
         let fields = [ "local_memory_resets"; "live_graph_execs" ] in
         let run linked =
           deltas ctx fields (fun () ->
               let ran = run_linked linked ~seed:[] in
               (ran, Context.get_values ran out))
         in
         let (ran, got), first = run linked in
         let free1, _ = free_and_total () in
         (* Replay must write the result again using the cached graph's original output buffer. *)
         let ran = Context.set_values ran out (Array.create ~len:n 0.) in
         let (_, replayed), second = run (ran, snd linked) in
         let free2, _ = free_and_total () in
         let want = serial_expected ~graph n in
         Stdio.eprintf
           "%s: %d B frame, %d B base depot estimate. Free before %d B, after %d B, after replay \
            %d B (not part of the golden)\n\
            %!"
           name (s11d_floats * 4) (Lazy.force reservation) free0 free1 free2;
         ( Array.equal Float.equal got want && Array.equal Float.equal replayed want,
           [| Float.of_int free1; Float.of_int free2 |],
           Float.of_int free0,
           first,
           second ))
    in
    let serial_name = "clr_serial_frame" in
    let plain_serial = serial_measurement ~graph:false ~first_id:1774000 ~name:serial_name in
    let graph_serial = serial_measurement ~graph:true ~first_id:1774100 ~name:"clr_serial_graph" in
    in_leg ~leg:serial_scratch
      "serial scratch: three private arrays compute the right values on both runs" (fun label ->
        let values, _, _, _, _ = Lazy.force plain_serial in
        p label values);
    in_leg ~leg:serial_scratch "serial scratch: the emitted one-thread frame uses the device buffer"
      (fun label ->
        ignore (Lazy.force plain_serial);
        p label
          (String.is_substring (Generated.read serial_name) ~substring:"ocannl_serial_scratch"
          && not
               (String.is_substring
                  (Generated.read ~ext:".ptx" serial_name)
                  ~substring:"__local_depot")));
    let scratch_headroom label (_, available, before, first, second) =
      (* The before/after oracle reads device-wide free memory, so it admits the same neighbour
         noise as the depot test. The emitted PTX and reset count pin placement independently. *)
      let tol _ = Float.of_int (Lazy.force reservation / 2) in
      let free = Verdict.within label available [| before; before |] ~tol in
      p label (free && Int.equal (List.hd_exn first) 0 && Int.equal (List.hd_exn second) 0)
    in
    in_leg ~leg:serial_scratch
      "serial scratch: free memory stays clear of a depot-sized reservation without a reset"
      (fun label -> scratch_headroom label (Lazy.force plain_serial));
    in_leg ~leg:serial_graph
      "serial scratch: graph capture and cached replay compute the right values" (fun label ->
        let values, _, _, first, second = Lazy.force graph_serial in
        p label
          (values && List.equal Int.equal first [ 0; 1 ] && List.equal Int.equal second [ 0; 0 ]));
    in_leg ~leg:serial_graph
      "serial scratch: graph capture and replay stay clear of a depot-sized reservation"
      (fun label -> scratch_headroom label (Lazy.force graph_serial));
    let stack_budget =
      lazy
        (Option.value ~default:(s11d_floats / 3 * 4) (counter ctx "serial_device_scratch_budget"))
    in
    in_leg ~leg:serial_scratch "serial scratch: the cutoff still permits a one-thread local depot"
      (fun label ->
        let budget = Lazy.force stack_budget in
        let n = budget / 4 in
        let name = "clr_stack_boundary" in
        let values = run_frame ~ctx ~first_id:1774200 ~name n in
        p label
          (Array.equal Float.equal values (expected n)
          && String.is_substring (Generated.read ~ext:".ptx" name) ~substring:"__local_depot"
          && not (String.is_substring (Generated.read name) ~substring:"ocannl_serial_scratch")));

    in_leg ~leg:serial_scratch
      "serial scratch: three smaller arrays just over the combined cutoff use the buffer and \
       compute correctly" (fun label ->
        (* Together three float arrays of this length exceed the budget by at most one alignment per
           padded slice and one extra element per array. Each alone fits in the depot. *)
        let n = (Lazy.force stack_budget / 12) + 1 in
        let name = "clr_scratch_boundary" in
        let linked, out =
          link_program ~ctx ~name ~graph:false
            (serial_frame_program ~first_id:1774500 ~name ~graph:false n)
        in
        let ran = run_linked linked ~seed:[] in
        p label
          (Array.equal Float.equal (Context.get_values ran out) (serial_expected ~graph:false n)
          && String.is_substring (Generated.read name) ~substring:"ocannl_serial_scratch"
          && not (String.is_substring (Generated.read ~ext:".ptx" name) ~substring:"__local_depot")
          ));

    (* Each scratch lifetime is counted around the real driver calls. A graph failure must also
       leave the stream ready to capture again, with no executable published by the failed run. *)
    List.iter
      [ (false, "plain", serial_scratch, 1774300); (true, "capture", serial_graph, 1774400) ]
      ~f:(fun (graph, kind, leg, first_id) ->
        List.iteri
          [ (FI.Cuda_scratch_allocated, "after allocation"); (FI.Cuda_scratch_freed, "at cleanup") ]
          ~f:(fun i (fault, phase) ->
            in_legf ~leg "serial scratch: %s failure %s frees exactly once and a retry runs" kind
              phase (fun label ->
                let name = Printf.sprintf "clr_fault_%s_%d" kind i in
                let linked, out =
                  link_program ~ctx ~name ~graph
                    (serial_frame_program ~first_id:(first_id + (10 * i)) ~name ~graph serial_cells)
                in
                let events = ref [] in
                let callback = function
                  | (FI.Cuda_scratch_allocated | FI.Cuda_scratch_freed) as point ->
                      events := point :: !events;
                      if FI.equal_point point fault then raise Scratch_fault
                  | _ -> ()
                in
                let failed, cached =
                  deltas ctx [ "live_graph_execs" ] (fun () ->
                      match FI.with_callback callback ~f:(fun () -> run_linked linked ~seed:[]) with
                      | exception Scratch_fault -> true
                      | _ -> false)
                in
                let value, rebuilt =
                  deltas ctx [ "live_graph_execs" ] (fun () ->
                      let ran = run_linked linked ~seed:[] in
                      Context.get_values ran out)
                in
                p label
                  (failed
                  && List.equal FI.equal_point (List.rev !events)
                       [ FI.Cuda_scratch_allocated; FI.Cuda_scratch_freed ]
                  && List.equal Int.equal cached [ 0 ]
                  && List.equal Int.equal rebuilt [ (if graph then 1 else 0) ]
                  && Array.equal Float.equal value (serial_expected ~graph serial_cells)))));

    (* --- 4. HIP: the scratch frame runs; the back-off is in place and untouched. --- *)
    let hip_floats =
      lazy
        (match (Context.hardware_limits ctx).Ir.Backend_intf.max_thread_private_bytes with
        | Some limit -> min s11d_floats ((limit - 2048) / 4)
        | None -> s11d_floats)
    in
    let hip_run =
      lazy
        (let n = Lazy.force hip_floats in
         let free0, _ = free_and_total () in
         let got, d =
           deltas ctx backoff (fun () -> run_frame ~ctx ~first_id:1773700 ~name:"clr_hip" n)
         in
         let free1, _ = free_and_total () in
         Stdio.eprintf
           "hip: a %d B scratch frame for one work-item, %d resident work-items; free %d B before \
            the launch, %d B after it (not part of the golden)\n\
            %!"
           (n * 4) (Lazy.force resident) free0 free1;
         (n, got, d))
    in
    in_leg ~leg:hip "hip: a scratch frame for one work-item computes the right values" (fun label ->
        let n, got, _ = Lazy.force hip_run in
        p label (Array.equal Float.equal got (expected n)));
    in_leg ~leg:hip "hip: the backend reports the launch back-off's counters, and it was not needed"
      (fun label ->
        let _, _, d = Lazy.force hip_run in
        p label
          (Option.is_some (counter ctx "launch_memory_retries")
          && Option.is_some (counter ctx "launch_memory_refusals")
          && List.equal Int.equal d [ 0; 0 ])));

  (* --- 5. Contended: retried while a neighbour holds the memory, run once it is gone. --- *)
  if contended_mode then
    let free0, _ = if on_cuda || on_hip then free_and_total () else (0, 0) in
    (* A reservation of a third of what is free, at most 2 GiB (on HIP, within the scratch budget);
       the child leaves a quarter of it. *)
    let per_item_cap =
      match (Context.hardware_limits ctx).Ir.Backend_intf.max_thread_private_bytes with
      | Some limit -> (limit - 2048) / 4
      | None -> hopeless_floats
    in
    let floats =
      if not (on_cuda || on_hip) then 0
      else
        min per_item_cap
          (min hopeless_floats (min (free0 / 3) (1 lsl 31) / 4 / max 1 (Lazy.force resident)))
    in
    let reservation = floats * 4 * if on_cuda || on_hip then Lazy.force resident else 0 in
    let hold ~what ~before =
      lazy
        (before ();
         let free_now, _ = free_and_total () in
         let bytes = free_now - (reservation / 4) in
         let read_end, write_end = Unix.pipe ~cloexec:true () in
         let pid =
           Unix.create_process Stdlib.Sys.executable_name
             [| Stdlib.Sys.executable_name; "--hold"; Int.to_string bytes; "2.0" |]
             Unix.stdin write_end Unix.stderr
         in
         Unix.close write_end;
         let line = Stdio.In_channel.input_line (Unix.in_channel_of_descr read_end) in
         let free1, _ = free_and_total () in
         Stdio.eprintf
           "contended %s: %d floats per thread (%d B), the child holds %d B, %d B free before it, \
            %d B with it (not part of the golden)\n\
            %!"
           what floats reservation bytes free_now free1;
         (pid, Option.equal String.equal line (Some "held"), free1))
    in
    (* Three submissions a neighbour's hold can refuse: a plain launch; a graph's first run, whose
       instantiation reserves its kernels' local memory; and a cached graph's replay, run once
       beforehand so that its reservation was taken and released again (CUDA only: HIP does not
       release it, so its replay would need nothing). *)
    let cases =
      [
        ("plain", `Plain, gpu, 1773200);
        ( "graph, first capture",
          `First_capture,
          Verdict.sub_leg gpu ~when_:capturing ~on:backend_name,
          1773800 );
        ( "graph, cached replay",
          `Replay,
          Verdict.sub_leg cuda ~when_:capturing ~on:backend_name,
          1773850 );
      ]
    in
    List.iter cases ~f:(fun (what, case, outer, first_id) ->
        let graph = match case with `Plain -> false | `First_capture | `Replay -> true in
        let name = Printf.sprintf "clr_contended_%d" first_id in
        let prepared =
          lazy
            (let (ctx, routine), out =
               link_program ~ctx ~name ~graph
                 (frame_program ~first_id ~name ~graph ~multi:on_cuda floats)
             in
             let ctx =
               match case with
               | `Replay ->
                   let ctx = run_linked (ctx, routine) ~seed:[] in
                   ignore (Context.get_values ctx out : float array);
                   ctx
               | `Plain | `First_capture -> ctx
             in
             ((ctx, routine), out))
        in
        let child = hold ~what ~before:(fun () -> ignore (Lazy.force prepared)) in
        (* The gate reads only the device (enough free memory for a frame worth contending); the
           child's hold is a claim of its own, so a child that never held fails the run loudly
           rather than skipping the claims it was there to set up. *)
        let leg = Verdict.sub_leg outer ~on:backend_name ~when_:(lazy (floats >= 1024)) in
        in_legf ~leg "contended, %s: the child holds the memory, leaving less than the reservation"
          what (fun label ->
            let _, held, free1 = Lazy.force child in
            p label (held && free1 < reservation));
        let run =
          lazy
            (let start = Unix.gettimeofday () in
             let got, d =
               deltas ctx
                 (backoff @ [ "graph_launch_memory_drops"; "live_graph_execs" ])
                 (fun () ->
                   let linked, out = Lazy.force prepared in
                   Context.get_values (run_linked linked ~seed:[]) out)
             in
             Stdio.eprintf
               "contended %s: ran after %.1fs, live_graph_execs %s (not part of the golden)\n%!"
               what
               (Unix.gettimeofday () -. start)
               (Option.value_map (counter ctx "live_graph_execs") ~default:"-" ~f:Int.to_string);
             let pid, _, _ = Lazy.force child in
             ignore (Unix.waitpid [] pid : int * Unix.process_status);
             (got, d))
        in
        in_legf
          ~leg:(Verdict.sub_leg leg ~when_:(lazy on_cuda) ~on:backend_name)
          "contended, %s: the submission is retried, and not given up on" what
          (fun label ->
            match snd (Lazy.force run) with
            | [ retries; refusals; _; _ ] -> p label (retries >= 1 && refusals = 0)
            | _ -> p label false);
        (* Measured on tuf-amd-linux (discrete memory, ROCm): with the child leaving 380 MB of 8.4
           GB free, a frame whose full-occupancy scratch would be 2 GiB launched at once. ROCr backs
           scratch for the waves actually launched, on demand, so a one-work-item frame is never
           refused, and HIP's retry path is not reached here. Pinned, so that a ROCr that starts
           refusing turns this red and the retry claim above gets a HIP leg. *)
        in_legf
          ~leg:(Verdict.sub_leg leg ~when_:(lazy on_hip) ~on:backend_name)
          "contended, %s: on HIP, the launch under the hold is not refused, so not retried" what
          (fun label ->
            match snd (Lazy.force run) with
            | [ retries; refusals; _; _ ] -> p label (retries = 0 && refusals = 0)
            | _ -> p label false);
        in_legf ~leg "contended, %s: the retried kernels compute the right values" what
          (fun label ->
            p label
              (Array.equal Float.equal
                 (fst (Lazy.force run))
                 (expected ~graph ~multi:on_cuda floats)));
        (* The cached replay's graph launch is the one refused: its exec must be dropped, not
           relaunched -- a relaunch after the refusal ran without its reservation and faulted on
           rog-nv -- and the run left to the plain launches, which recapture nothing. *)
        (match case with
        | `Replay ->
            in_legf ~leg
              "contended, %s: the refused graph launch dropped its exec, leaving none cached" what
              (fun label ->
                match snd (Lazy.force run) with
                | [ _; _; drops; live ] -> p label (drops = 1 && live = -1)
                | _ -> p label false)
        | `Plain | `First_capture -> ());
        if Lazy.is_val child && not (Lazy.is_val run) then
          let pid, _, _ = Lazy.force child in
          ignore (Unix.waitpid [] pid : int * Unix.process_status))
