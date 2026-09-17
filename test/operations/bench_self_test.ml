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

let field j k = try Some (U.member k j) with _ -> None

let number j k =
  match field j k with
  | Some (`Float f) -> Some f
  | Some (`Int i) -> Some (Float.of_int i)
  | _ -> None

let string_field j k = match field j k with Some (`String s) -> Some s | _ -> None

let is_str j k expected =
  Option.value_map (string_field j k) ~default:false ~f:(String.equal expected)

let () =
  (* Emitted to stderr rather than stdout: the line carries wall-clock digits, and the golden is
     diffed. It is echoed rather than dropped so a failing run is diagnosable from the log. *)
  let line = H.run_self_test ~out:Stdio.stderr () in
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
      Verdict.p "framework is ocannl" (is_str j "framework" "ocannl");
      Verdict.p "backend names the backend the cell ran on"
        (Option.value_map (string_field j "backend") ~default:false ~f:(Fn.non String.is_empty));
      Verdict.p "workload names the self-test model, not a benchmark cell"
        (is_str j "workload" protocol.H.workload);
      Verdict.p "variant names the self-test" (is_str j "variant" "self-test");
      Verdict.p "precision is f32" (is_str j "precision" "f32");
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
      Verdict.p "and it names the counter it was read from"
        (Option.value_map
           (string_field j "peak_memory_source")
           ~default:false ~f:(Fn.non String.is_empty));
      Verdict.p "timed_steps is the count the protocol asked for"
        (match field j "timed_steps" with
        | Some (`Int n) -> n = protocol.H.timed_steps
        | _ -> false);
      let losses = match field j "losses" with Some (`List l) -> l | _ -> [] in
      Verdict.p "losses carries one parity checksum per parity step"
        (List.length losses = protocol.H.parity_steps);
      Verdict.p_all "every parity checksum is a finite number" losses ~f:(function
        | `Float _ | `Int _ -> true
        | _ -> false)
  | None ->
      (* The claim above has already failed the run; naming the line is what makes it
         diagnosable. *)
      Verdict.fail ("the emitted result line is not a JSON object: " ^ line)
