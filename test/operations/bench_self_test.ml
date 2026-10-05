(* The executable smoke test of the OCANNL benchmark measurement path (gh-ocannl-702).

   benchmarks/fixtures/ holds only DIGESTS.txt in a fresh checkout — the .safetensors files come
   from gen_fixtures.py, which imports numpy and safetensors.numpy, and the runners are dispatched
   through benchmarks/.venv — so without a provisioned Python ML environment no benchmark cell can
   be run at all, including the OCANNL ones, which need nothing from torch but the bytes. That left
   Bench_harness.measure_and_emit, the emitter every OCANNL benchmark cell's result flows through,
   with nothing executable standing behind it: the argument mapping into Bench_json.result_line
   (which percentile reaches p10) was held up by the type checker agreeing that eleven labelled
   arguments have the right types, and a break in it would first surface as a wrong number in a
   report from a GPU box, days later.

   So this runs one cell. Not a comparable one: Bench_harness.run_self_test fabricates its model in
   memory, deliberately NOT the byte-identical fixture the cross-framework parity gate is built on,
   and the emitted record says so in its workload and variant fields. The measurement path it drives
   is the real one, end to end — compile, parity window, warmup, per-step-synced percentiles, queued
   mean, and the emit.

   The claims below are about the SHAPE of the emitted record, which is backend-uniform; the record
   itself, timings and all, goes to stderr, so the golden stays portable. The percentile ordering is
   the claim that pins the argument mapping: the harness sorts before it reads percentiles, so p10 >
   p50 or p50 > p90 in the emitted line can only come from the three arguments being crossed. *)

open Base
module H = Bench_harness
module U = Yojson.Safe.Util
open Ocannl.Operation.DSL_modules

let field j k = try Some (U.member k j) with _ -> None

let number j k =
  match field j k with
  | Some (`Float f) -> Some f
  | Some (`Int i) -> Some (Float.of_int i)
  | _ -> None

let string_field j k = match field j k with Some (`String s) -> Some s | _ -> None

let is_str j k expected =
  Option.value_map (string_field j k) ~default:false ~f:(String.equal expected)

(* gh-ocannl-1209: the checkpoints a cell leaves before its later stages, read back the way a driver
   reads a cell's log -- every line behind the prefix, in order. *)
let checkpoints_in text =
  String.split_lines text
  |> List.filter_map ~f:(fun line ->
      Option.bind (String.chop_prefix line ~prefix:Bench_json.checkpoint_prefix) ~f:(fun obj ->
          try Some (Yojson.Safe.from_string obj) with _ -> None))

let capture_file suffix = Stdlib.Filename.temp_file "bench_self_test" suffix

let read_and_remove path =
  let text = Stdio.In_channel.read_all path in
  (try Unix.unlink path with Unix.Unix_error _ -> ());
  text

(* The control the checkpoints exist for: a cell killed in its LAST stage, after all its losses were
   observed, the way a driver's cap killed the TUF s1024 cell inside the dominant-kernel instrument.
   The child runs the real protocol with the instrument replaced by a probe that kills its own
   process with SIGKILL -- what the cap delivers: nothing unwinds, no [at_exit] flushes a buffer. An
   argv marker rather than an environment variable, so nothing ambient can put a run into this mode
   (as in atomic_file_race). The kill is deterministic: no deadline races the compile. *)
let killed_in_diagnostics_arg = "--killed-in-diagnostics"
let late_probe_marker = "bench_self_test: late probe reached; killing this process"

let () =
  if Array.exists Stdlib.Sys.argv ~f:(String.equal killed_in_diagnostics_arg) then (
    ignore
      (H.run_self_test
         ~late_probe:(fun () ->
           Stdio.eprintf "%s\n%!" late_probe_marker;
           Unix.kill (Unix.getpid ()) Stdlib.Sys.sigkill;
           (* Unreachable unless the kill was refused; the parent reads the exit status. *)
           Stdlib.exit 3)
         ()
        : string);
    (* Reaching here means the late probe returned: a result line went to stdout, which the parent
       counts against the control. *)
    Stdlib.exit 0)

let () =
  let exe = Stdlib.Sys.executable_name in
  let out_path = capture_file ".out" and err_path = capture_file ".err" in
  let open_capture path = Unix.openfile path [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
  let out = open_capture out_path and err = open_capture err_path in
  let pid = Unix.create_process exe [| exe; killed_in_diagnostics_arg |] Unix.stdin out err in
  let _, status = Unix.waitpid [] pid in
  Unix.close out;
  Unix.close err;
  let stdout_text = read_and_remove out_path and stderr_text = read_and_remove err_path in
  Stdio.eprintf "bench_self_test: the killed child's stderr follows\n%s\n%!" stderr_text;
  let protocol = H.self_test_protocol in
  Verdict.p "the child was killed in its late probe rather than exiting"
    ((match status with Unix.WEXITED 0 | Unix.WSTOPPED _ -> false | _ -> true)
    && String.is_substring stderr_text ~substring:late_probe_marker);
  (* Over the child's combined output, as [orchestrate.py] reads a cell's: its stdout is expected to
     be empty, so a claim over that alone would rest on an empty population. *)
  let output_lines = String.split_lines stdout_text @ String.split_lines stderr_text in
  Verdict.p_empty "no line of the killed child's output is a result line" ~over:output_lines
    (List.filter output_lines ~f:(String.is_prefix ~prefix:"{"));
  let kept = checkpoints_in stderr_text in
  Verdict.p "the killed child checkpointed every parity step, then the timed stages"
    (List.length kept = protocol.H.parity_steps + 1);
  match List.last kept with
  | None -> Verdict.fail "the killed child left no checkpoint"
  | Some last ->
      Verdict.p "its last checkpoint says every measured stage completed and the probe was running"
        (Option.equal Yojson.Safe.equal (field last "stages")
           (Some
              (`Assoc
                 [
                   ("parity", `String "complete");
                   ("warmup", `String "complete");
                   ("timing", `String "complete");
                   ("dominant_kernel", `String "running");
                   ("result", `String "pending");
                 ])));
      Verdict.p "and it is marked unaccepted"
        (match field last "accepted" with Some (`Bool false) -> true | _ -> false);
      let losses = match field last "losses" with Some (`List l) -> l | _ -> [] in
      Verdict.p "the kill kept one loss per parity step"
        (List.length losses = protocol.H.parity_steps);
      Verdict.p_all "every loss the kill kept is finite" losses ~f:(function
        | `Float _ | `Int _ -> true
        | _ -> false)

let () =
  (* Emitted to stderr rather than stdout: the line carries wall-clock digits, and the golden is
     diffed. It is echoed rather than dropped so a failing run is diagnosable from the log. *)
  let expected_mma = ref None in
  let inspect_step ctx bindings loss routines =
    let kernels = H.shipped_kernels routines in
    let writes = List.concat_map kernels ~f:(fun (_, seg) -> H.writes_of seg.Ir.Low_level.llc) in
    let writes_node tn = List.mem writes tn ~equal:Ir.Tnode.equal in
    Verdict.p "training segment census includes the forward loss"
      (writes_node loss.Ocannl.Tensor.value);
    let params = Ocannl.Train.trainable_params loss |> Set.to_list in
    Verdict.p_all "training segment census includes every parameter gradient and optimizer write"
      params ~f:(fun p ->
        writes_node p.Ocannl.Tensor.value
        && Option.value_map p.Ocannl.Tensor.diff ~default:false ~f:(fun d ->
            writes_node d.Ocannl.Tensor.grad));
    H.print_shipped_census ~out:Stdio.stderr routines;
    let outcomes = H.time_shipped_segments ~out:Stdio.stderr ~repeats:1 ~ctx ~bindings routines in
    Verdict.p_all "every tiny training segment executes with a positive standalone time" outcomes
      ~f:(fun result -> match result with Ok ms -> Float.(ms > 0.) | Error _ -> false)
  in
  let checkpoint_path = capture_file ".checkpoints" in
  let line =
    Stdio.Out_channel.with_file checkpoint_path ~f:(fun checkpoint_out ->
        H.run_self_test ~out:Stdio.stderr ~checkpoint_out ~inspect_step
          ~inspect_compiled:(fun _ routines -> expected_mma := Some (H.step_census routines))
          ())
  in
  let kept = checkpoints_in (read_and_remove checkpoint_path) in
  (* Before any host-gated step runs, its conditional SGD is nevertheless compiled work. *)
  ignore
    (H.run_self_test ~out:Stdio.stderr
       ~leg:{ H.self_test_leg with base = "f16" }
       ~inspect_compiled:(fun loss routines ->
         let shipped = H.compiled_step_routines routines in
         let writes =
           List.concat_map (H.shipped_kernels shipped) ~f:(fun (_, seg) ->
               H.writes_of seg.Ir.Low_level.llc)
         in
         Verdict.p "a host-gated census before execution includes its separately compiled optimizer"
           (match routines with
           | H.Host_gate (_, _, grad, sgd) ->
               (not (String.equal grad.Context.name sgd.Context.name))
               && List.exists shipped ~f:(fun r -> String.equal r.Context.name sgd.Context.name)
           | _ -> false);
         let m = H.step_census routines in
         Verdict.p "MMA census includes both compiled host-gated routines before execution"
           (List.length shipped = 2
           && m.statements = List.sum (module Int) shipped ~f:(fun r -> r.Context.mma.statements)
           && m.scalar_fallbacks
              = List.sum (module Int) shipped ~f:(fun r -> r.Context.mma.scalar_fallbacks));
         Verdict.p_all "that pre-execution census includes every optimizer parameter write"
           (Ocannl.Train.trainable_params loss |> Set.to_list)
           ~f:(fun p -> List.mem writes p.Ocannl.Tensor.value ~equal:Ir.Tnode.equal))
       ()
      : string);
  let protocol = H.self_test_protocol in
  let parsed =
    match try Some (Yojson.Safe.from_string line) with _ -> None with
    | Some (`Assoc _ as j) -> Some j
    | Some _ | None -> None
  in
  (* Claimed before the match, so the claim is decided by the parse rather than by which branch we
     are standing in: inside the successful branch it could only ever have been [true]. *)
  Verdict.p "the emitted result line parses as one JSON object" (Option.is_some parsed);
  match parsed with
  | Some j ->
      Verdict.p "untuned result census matches every compiled step routine"
        (Option.value_map !expected_mma ~default:false ~f:(fun m ->
             Yojson.Safe.equal (U.member "shipped_mma" j)
               (Yojson.Safe.from_string (Bench_json.mma_object (Some (H.mma_wire m))))));
      Verdict.p "framework is ocannl" (is_str j "framework" "ocannl");
      Verdict.p "backend names the backend the cell ran on"
        (Option.value_map (string_field j "backend") ~default:false ~f:(Fn.non String.is_empty));
      Verdict.p "workload names the self-test model, not a benchmark cell"
        (is_str j "workload" protocol.H.workload);
      Verdict.p "variant names the self-test" (is_str j "variant" "self-test");
      Verdict.p "precision is f32" (is_str j "precision" "f32");
      let algebra, source =
        Utils.get_global_arg_with_source ~default:"all" ~arg_name:"simplify_fp_algebra"
      in
      let recorded_algebra = Option.value (field j "simplify_fp_algebra") ~default:`Null in
      Verdict.p "float algebra metadata records the resolved selector and its source"
        (is_str recorded_algebra "value" algebra
        && is_str recorded_algebra "source" (Utils.config_source_label source));
      Verdict.p "compile_s is a non-negative number"
        (Option.value_map (number j "compile_s") ~default:false ~f:(fun s -> Float.(s >= 0.)));
      Verdict.p "searched is false in an untuned cell"
        (match field j "searched" with Some (`Bool b) -> not b | _ -> false);
      let step_ms = Option.value (field j "step_ms") ~default:`Null in
      let p10 = number step_ms "p10"
      and p50 = number step_ms "p50"
      and p90 = number step_ms "p90" in
      Verdict.p "step_ms carries all three percentiles"
        (Option.is_some p10 && Option.is_some p50 && Option.is_some p90);
      let percentiles = List.filter_map [ p10; p50; p90 ] ~f:Fn.id in
      Verdict.p "every reported percentile is a positive time"
        (List.length percentiles = 3 && List.for_all percentiles ~f:(fun t -> Float.(t > 0.)));
      Verdict.p "the percentiles are emitted in order p10 <= p50 <= p90"
        (match (p10, p50, p90) with
        | Some a, Some b, Some c -> Float.(a <= b) && Float.(b <= c)
        | _ -> false);
      Verdict.p "queued_step_ms is a positive time"
        (Option.value_map (number j "queued_step_ms") ~default:false ~f:(fun t -> Float.(t > 0.)));
      (* gh-ocannl-1006: the memory column's bracket, which lives in [measure_and_emit] beside the
         timing loops and is reachable from nowhere else. A bracket placed after the read, or a
         backend the shared allocator seam does not reach, emits a zero here -- which the report
         would print as a workload with no footprint rather than as a cell that measured none. The
         bound is the model's own bytes: the self-test trains a real MLP, so its parameters,
         activations and gradients are on the device whatever the backend. *)
      Verdict.p "peak_memory_bytes is a positive byte count on every backend"
        (match field j "peak_memory_bytes" with Some (`Int b) -> b > 0 | _ -> false);
      (* Both spellings: the short tag the report prints ON the row, so a row states its own
         counter, and the long description its legend expands that tag into (review round 1). *)
      Verdict.p_all "and it names the counter it was read from, in both spellings"
        [ "peak_memory_counter"; "peak_memory_source" ] ~f:(fun k ->
          Option.value_map (string_field j k) ~default:false ~f:(Fn.non String.is_empty));
      (* gh-ocannl-1006: the %-of-peak column's instrument, which lives in [measure_and_emit] after
         the timed steps and compiles every kernel the step SHIPPED ([Context.routine.segments]) on
         its own. What is backend-uniform is that it ran and found a kernel: a routine whose
         segments the census failed to record reaches here as [no-kernel], which is the break this
         guards. Whether the number prints depends on the backend's envelope (the C backends carry
         none), so the verdict itself goes to stderr, and the claim is only that the verdict and the
         number agree. *)
      let dk = Option.value (field j "dominant_kernel") ~default:`Null in
      Stdio.eprintf "bench_self_test: dominant kernel %s\n%!" (Yojson.Safe.to_string dk);
      Verdict.p "dominant_kernel names a kernel timed on its own, with a positive time"
        ((match field dk "segment" with Some (`Int i) -> i >= 0 | _ -> false)
        && Option.value_map (number dk "seg_ms") ~default:false ~f:(fun t -> Float.(t > 0.)));
      Verdict.p "its verdict is one the report renders, and a number is printed exactly when exact"
        (match (string_field dk "verdict", field dk "pct_of_peak") with
        | Some "exact", Some (`Float _ | `Int _) -> true
        | Some ("approximate" | "opaque" | "no-ceiling"), Some `Null -> true
        | _ -> false);
      Verdict.p "timed_steps is the count the protocol asked for"
        (match field j "timed_steps" with
        | Some (`Int n) -> n = protocol.H.timed_steps
        | _ -> false);
      let losses = match field j "losses" with Some (`List l) -> l | _ -> [] in
      Verdict.p "losses carries one parity checksum per parity step"
        (List.length losses = protocol.H.parity_steps);
      Verdict.p_all "every parity checksum is a finite number" losses ~f:(function
        | `Float _ | `Int _ -> true
        | _ -> false);
      (* gh-ocannl-1209: what a kill would have kept is what the result line reports. *)
      Verdict.p "the last checkpoint before the instrument holds the result line's own losses"
        (Option.value_map (List.last kept) ~default:false ~f:(fun last ->
             Option.equal Yojson.Safe.equal (field last "losses") (field j "losses")))
  | None ->
      (* The claim above has already failed the run; naming the line is what makes it
         diagnosable. *)
      Verdict.fail ("the emitted result line is not a JSON object: " ^ line)
