(* The benchmark runners' JSON result line, on fabricated values (gh-ocannl-676).

   [orchestrate.py] reads a cell's measurement by taking the last '{'-prefixed line of its output
   and [json.loads]ing it; a line that does not parse is reported as `!!! <label> failed` and the
   cell is dropped, after its whole measurement has been paid for. The line is one ~300-character
   format with two optional pre-formatted fragments and a nested object, and until this test nothing
   anywhere parsed it.

   The values here are the ones a happy-path run never produces and a report needs most: a diverged
   loss trajectory (OCaml's [%g] spells a non-finite float [nan] / [inf] / [-inf], none of which is
   JSON), a time that was never measured ([infinity]), and a backend diagnostic carrying quotes and
   control characters. The parse oracle is Yojson, which rejects exactly those OCaml spellings — the
   negative control at the end shows it does — while accepting [null]. *)

open Base
open Verdict.Claims

let parses s = match Yojson.Safe.from_string s with _ -> true | exception _ -> false
let member k j = Yojson.Safe.Util.member k j

let () =
  Stdio.printf "=== scalars ===\n";
  List.iter
    [ ("nan", Float.nan); ("inf", Float.infinity); ("-inf", Float.neg_infinity) ]
    ~f:(fun (name, v) ->
      p (Printf.sprintf "num %s is null" name) (String.equal (Bench_json.num v) "null");
      p (Printf.sprintf "fixed %s is null" name) (String.equal (Bench_json.fixed v) "null"));
  p "num of a finite value is the number"
    (String.equal (Bench_json.num 1.25) "1.25" && String.equal (Bench_json.fixed 1.25) "1.250");
  Stdio.printf "nums of a diverged trajectory: [%s]\n"
    (Bench_json.nums ~prec:9 [| 1.5; Float.nan; Float.infinity; Float.neg_infinity |])

(* A tuned cell whose arm A timed nothing at all and terminated on a failure whose message carries
   the characters that would invalidate the record: a quote, a backslash, a NUL and an ESC.

   Arms A, B and C are the three provenance buckets the [tune] object totals (gh-ocannl-677): a
   search that died mid-way, a replayed cache entry, and an arm that neither searched nor replayed
   because the search was off. That last one is why [no_searches] exists — before it, a reader had
   to infer the case from [searches] and [replays] both being zero, which is exactly the derivation
   the outcome type replaced.

   Arm D ran one beam round and the others none, so [rounds_run] beside [beam_width] reads as the
   search's own record of how far it went, apart from the configuration (gh-ocannl-1137).

   Arms B, D and E carry the three [tensorization] labels (gh-ocannl-626), and A and C carry the
   [null] that says no census was consulted. B is the case the field exists for: [tensorized: true]
   — the crowned schedule carries a [Tensorize] — with every one of its [Tile_mma] statements
   rendered as the lane-0 scalar fallback, so its 0.75 ms is a scalar timing under a tensorized
   label. *)
let tune =
  (* [shipped_mma] is the shipped ARTIFACT's census, and it disagrees with arm B's on purpose: this
     cell shipped a flip refinement or a replay fallback, so the arm describes a schedule that was
     discarded and only this field describes what ran (gh-ocannl-626). *)
  Bench_json.tune_object ~shipped:"B" ~searches:3 ~replays:1 ~no_searches:1
    ~shipped_mma:(Some ("tensorized", 4, 0))
    ~arms:
      [
        Bench_json.tune_arm ~name:"A" ~state:"search-died" ~searched:true
          ~cache_hit:false
            (* An arm that timed nothing still names an objective: [tune] resolves it before it can
               construct any report, so every arm on the line carries one. *)
          ~timing:"queued" ~rounds_run:0 ~beam_width:4 ~timings_contended:0 ~timings_unbatched:0
          ~best_ms:Float.infinity ~best_label:"tile 32x32" ~tensorized:false ~tensorization:None
          ~mma_statements:0 ~mma_scalar_fallbacks:0 ~mma_seeded:4 ~mma_timed:0
          ~mma_best_ms:Float.infinity
          ~terminal_failure:
            (Some
               (Printf.sprintf "compile failed: \"kernel\" \\ path%c%c ESC" (Char.of_int_exn 0)
                  (Char.of_int_exn 27)));
        Bench_json.tune_arm ~name:"B" ~state:"cache-replay" ~searched:false ~cache_hit:true
          ~timing:"queued" ~rounds_run:0 ~beam_width:4 ~timings_contended:0 ~timings_unbatched:0
          ~best_ms:0.75 ~best_label:"grid 128" ~tensorized:true
          ~tensorization:(Some "scalar-fallback") ~mma_statements:2 ~mma_scalar_fallbacks:2
          ~mma_seeded:6 ~mma_timed:3 ~mma_best_ms:0.8 ~terminal_failure:None;
        (* Neither searched nor replayed: every counter zero, no winner to name. *)
        Bench_json.tune_arm ~name:"C" ~state:"search-disabled" ~searched:false ~cache_hit:false
          ~timing:"queued" ~rounds_run:0 ~beam_width:2 ~timings_contended:0 ~timings_unbatched:0
          ~best_ms:Float.infinity ~best_label:"" ~tensorized:false ~tensorization:None
          ~mma_statements:0 ~mma_scalar_fallbacks:0 ~mma_seeded:0 ~mma_timed:0
          ~mma_best_ms:Float.infinity ~terminal_failure:None;
        (* An honestly tensorized winner, and an ordinary one that never asked. *)
        Bench_json.tune_arm ~name:"D" ~state:"searched" ~searched:true ~cache_hit:false
          ~timing:"queued" ~rounds_run:1 ~beam_width:4 ~timings_contended:2 ~timings_unbatched:1
          ~best_ms:0.5 ~best_label:"mma-gpu 16x16x16" ~tensorized:true
          ~tensorization:(Some "tensorized") ~mma_statements:4 ~mma_scalar_fallbacks:0 ~mma_seeded:6
          ~mma_timed:5 ~mma_best_ms:0.5 ~terminal_failure:None;
        Bench_json.tune_arm ~name:"E" ~state:"searched" ~searched:true
          ~cache_hit:false
            (* The other objective, so the golden shows both spellings on one line. *)
          ~timing:"isolated" ~rounds_run:0 ~beam_width:2 ~timings_contended:0 ~timings_unbatched:0
          ~best_ms:1.25 ~best_label:"grid 64" ~tensorized:false
          ~tensorization:(Some "not-requested") ~mma_statements:0 ~mma_scalar_fallbacks:0
          ~mma_seeded:0 ~mma_timed:0 ~mma_best_ms:Float.infinity ~terminal_failure:None;
      ]

(* gh-ocannl-1006: the dominant kernel's %-of-peak, one fabricated kernel per verdict. The constants
   are round so the arithmetic can be checked by eye: 1e9 bytes at 1e12 B/s is 1 ms, 1e9 ops at 5e12
   FLOP/s is 0.2 ms, so the memory leg binds and a 4 ms kernel attains 25%. *)
let class_legs = (Some (5e12, "fab class constant"), Some (1e12, "fab class constant"))

let ceiling ?(gpu = true) ?(tensorization = "not-requested") ?(narrow_native = false)
    ?(legs = class_legs) () =
  Bench_json.choose_ceiling ~gpu ~tensorization ~narrow_native ~peak_flops:(fst legs)
    ~peak_memory_bandwidth:(snd legs)

let kernel ?(flops = 1_000_000_000) ?(bytes = 1_000_000_000) ?(flops_exact = true)
    ?(bytes_exact = true) ?(opaque = false) ?(tensorization = "not-requested") ?(seg_ms = 4.) () =
  {
    Bench_json.segment = 3;
    segments = 12;
    declined = 0;
    (* A label with the characters that would invalidate the record, as a node name can carry. *)
    writes = "w1.grad \"b1\"";
    seg_ms;
    segments_ms = 20.;
    tensorization;
    flops;
    bytes;
    flops_exact;
    bytes_exact;
    opaque;
  }

let dominant_kernels =
  [
    ("exact", Bench_json.dominant_kernel_object ~ceiling:(ceiling ()) (Some (kernel ())));
    ( "exact f16-native",
      (* 4e10 ops at twice 5e12 is 4 ms against 1 ms of bytes: compute binds, 8 ms is 50%. *)
      Bench_json.dominant_kernel_object
        ~ceiling:(ceiling ~gpu:false ~narrow_native:true ())
        (Some (kernel ~flops:40_000_000_000 ~seg_ms:8. ())) );
    ( "approximate",
      (* The memory leg binds, and its byte count is only an upper bound. *)
      Bench_json.dominant_kernel_object ~ceiling:(ceiling ()) (Some (kernel ~bytes_exact:false ()))
    );
    ( "exact over an inexact leg",
      (* 4e10 ops at 5e12 is 8 ms against at most 1 ms of bytes: the exact leg binds, so the
         upper-bound byte count cannot matter and 16 ms is exactly 50%. *)
      Bench_json.dominant_kernel_object ~ceiling:(ceiling ())
        (Some (kernel ~flops:40_000_000_000 ~bytes_exact:false ~seg_ms:16. ())) );
    ( "opaque",
      Bench_json.dominant_kernel_object ~ceiling:(ceiling ()) (Some (kernel ~opaque:true ())) );
    ( "no-ceiling (no constants)",
      Bench_json.dominant_kernel_object
        ~ceiling:(ceiling ~gpu:false ~legs:(None, Some (1e12, "fab config")) ())
        (Some (kernel ())) );
    ( "no-ceiling (gpu tensor cores)",
      Bench_json.dominant_kernel_object
        ~ceiling:(ceiling ~tensorization:"tensorized" ())
        (Some (kernel ~tensorization:"tensorized" ())) );
    ( "no-kernel",
      Bench_json.dominant_kernel_object ~note:"all 12 kernels declined to compile on their own"
        ~ceiling:(Error "no kernel") None );
  ]

let ordinary =
  Bench_json.result_line ~shipped_mma:("scalar-fallback", 3, 3) ~backend:"cc" ~variant:"default"
    ~precision:"f32" ~profile:None
    ~regime_knobs:[ ("tf32_matmuls", None); ("cc_backend_fast_math", None) ]
    ~workload:"mlp3" ~compile_s:2.5 ~searched:false ~p10:0.5 ~p50:0.75 ~p90:1.25 ~queued_ms:0.625
    ~timed_steps:20
      (* A fabricated counter name, deliberately not the harness's own spelling: what this test pins
         is that the wire format carries the pair, not what any one runner calls its counter. *)
    ~peak_memory:(Some (2097152, "fab-hw", "fabricated \"high-water\" counter (requested bytes)"))
    ~dominant_kernel:(List.Assoc.find_exn dominant_kernels "exact" ~equal:String.equal)
    ~losses:[| 2.5; 1.75; 1.25 |] ()

(* Everything a diverged, half-measured, tuned cell reports at once. *)
let diverged =
  Bench_json.result_line ~backend:"metal" ~variant:"tuned" ~precision:"f16"
    ~profile:(Some "approximate")
    ~regime_knobs:
      [
        ("tf32_matmuls", Some ("true", "profile 'approximate' via the commandline"));
        (* An override beside the profile, and a quote to escape. *)
        ("tune_inline_flips", Some ("5", "environment \"OCANNL_TUNE_INLINE_FLIPS\""));
      ]
    ~workload:"gpt2_mini" ~compile_s:Float.nan ~searched:true ~tokens_per_step:4096 ~tune
    ~p10:Float.infinity ~p50:Float.nan ~p90:Float.neg_infinity ~queued_ms:Float.nan
    ~timed_steps:0
      (* The cell that measured no footprint at all: the column has to say so as a dash, and a
         missing counter must never reach the report as a zero-byte workload. *)
    ~peak_memory:None
    ~losses:[| 1.5; Float.nan; Float.infinity; Float.neg_infinity |]
    ()

let () =
  (* Not '{'-prefixed: [orchestrate.py] takes a cell's result from the last line that is, and
     [benchmarks/test_orchestrate.py] reads these by their prefix to render every verdict. *)
  Stdio.printf "=== dominant kernel objects ===\n";
  List.iter dominant_kernels ~f:(fun (name, o) -> Stdio.printf "dominant_kernel %s: %s\n" name o);
  Stdio.printf "\n=== ordinary cell ===\n%s\n" ordinary;
  Stdio.printf "\n=== diverged cell ===\n%s\n" diverged;
  Stdio.printf "\n=== verdicts ===\n";
  List.iter
    [ ("ordinary", ordinary); ("diverged", diverged) ]
    ~f:(fun (name, line) ->
      p (Printf.sprintf "%s line parses as JSON" name) (parses line);
      p
        (Printf.sprintf "%s line is one line" name)
        (not (String.exists line ~f:(fun c -> Char.equal c '\n' || Char.equal c '\r')));
      p
        (Printf.sprintf "%s line has no byte below U+0020" name)
        (not (String.exists line ~f:(fun c -> Char.to_int c < 0x20))))

let () =
  let j = Yojson.Safe.from_string diverged in
  let losses = member "losses" j in
  p "a diverged loss trajectory keeps its finite steps and nulls the rest"
    (Yojson.Safe.equal losses (`List [ `Float 1.5; `Null; `Null; `Null ]));
  p_all "an unmeasured time is null, not a number"
    (List.map [ "p10"; "p50"; "p90" ] ~f:(fun p -> member p (member "step_ms" j))
    @ [ member "queued_step_ms" j; member "compile_s" j ])
    ~f:(fun time -> Yojson.Safe.equal time `Null);
  let arm_a = List.hd_exn (Yojson.Safe.Util.to_list (member "arms" (member "tune" j))) in
  p "an arm that timed nothing reports null times"
    (Yojson.Safe.equal (member "best_ms" arm_a) `Null
    && Yojson.Safe.equal (member "mma_best_ms" arm_a) `Null);
  p "a diagnostic survives as a scrubbed string"
    (match member "terminal_failure" arm_a with
    | `String s -> String.is_prefix s ~prefix:"compile failed: 'kernel' / path"
    | _ -> false);
  (* gh-ocannl-626: the wire format has to distinguish "asked and got scalar code" from "asked and
     got tensor cores" from "never asked" from "no census to consult", or a reader cannot tell a
     tensorized timing from a scalar one. *)
  let arms = Yojson.Safe.Util.to_list (member "arms" (member "tune" j)) in
  let arm name =
    List.find_exn arms ~f:(fun a -> Yojson.Safe.equal (member "arm" a) (`String name))
  in
  (* Every millisecond on an arm's line was measured under an objective, and since [tune] resolves
     it before it constructs any report, there is no arm that can omit it -- not even one that timed
     nothing (arm A). A reader comparing a [best_ms] with another artifact's needs this field to be
     there, since the two objectives differ by up to 2x and not by a constant (gh-ocannl-755). *)
  p_all "every arm names the objective its times were measured under" arms ~f:(fun a ->
      match member "timing" a with
      | `String spelling -> List.mem [ "queued"; "isolated" ] spelling ~equal:String.equal
      | _ -> false);
  p "a contention-affected arm preserves its refusal count"
    (Yojson.Safe.equal (member "timings_contended" (arm "D")) (`Int 2));
  (* gh-ocannl-1098: of those, the ones refused because calibration measured no batch within the
     target, which a queue threshold repeats on every rerun and a stall does not. *)
  p "an arm's no-batch refusals reach the wire apart from its contention count"
    (Yojson.Safe.equal (member "timings_unbatched" (arm "D")) (`Int 1));
  p_all "an arm with no crowned candidate reports a null tensorization, not a label" [ "A"; "C" ]
    ~f:(fun n -> Yojson.Safe.equal (member "tensorization" (arm n)) `Null);
  p_all "the three tensorization labels reach the wire"
    [ ("B", "scalar-fallback"); ("D", "tensorized"); ("E", "not-requested") ]
    ~f:(fun (n, label) -> Yojson.Safe.equal (member "tensorization" (arm n)) (`String label));
  p "a tensorized label over a scalar-fallback emission is visible as the pair"
    (Yojson.Safe.equal (member "tensorized" (arm "B")) (`Bool true)
    && Yojson.Safe.equal (member "tensorization" (arm "B")) (`String "scalar-fallback")
    && Yojson.Safe.equal (member "mma_statements" (arm "B")) (`Int 2)
    && Yojson.Safe.equal (member "mma_scalar_fallbacks" (arm "B")) (`Int 2));
  let untuned = Yojson.Safe.from_string ordinary in
  let census = member "shipped_mma" untuned in
  p "an untuned result carries the compiled MMA census without a tune object"
    (Yojson.Safe.equal (member "tune" untuned) `Null
    && Yojson.Safe.equal (member "tensorization" census) (`String "scalar-fallback")
    && Yojson.Safe.equal (member "statements" census) (`Int 3)
    && Yojson.Safe.equal (member "scalar_fallbacks" census) (`Int 3));
  (* The shipped artifact's own census, which the arms cannot always speak for. *)
  let shipped_mma = member "shipped_mma" (member "tune" j) in
  p "the shipped artifact's census is carried apart from the arms'"
    (Yojson.Safe.equal (member "tensorization" shipped_mma) (`String "tensorized")
    && Yojson.Safe.equal (member "statements" shipped_mma) (`Int 4)
    && Yojson.Safe.equal (member "scalar_fallbacks" shipped_mma) (`Int 0)
    (* And it is free to disagree with the arm named as shipped: that is the case it exists for. *)
    && Yojson.Safe.equal (member "tensorization" (arm "B")) (`String "scalar-fallback"));
  (* gh-ocannl-1006: a cell with no counter reports [null] for both halves of the memory column --
     the bytes AND the counter's name -- so the report can print a dash. A zero here would read as a
     workload with no footprint, which is the one wrong answer this pair exists to exclude. *)
  p "a cell that measured no footprint reports null bytes and null counter, not zero"
    (Yojson.Safe.equal (member "peak_memory_bytes" j) `Null
    && Yojson.Safe.equal (member "peak_memory_counter" j) `Null
    && Yojson.Safe.equal (member "peak_memory_source" j) `Null);
  p "a measured footprint carries its byte count and names the counter it came from"
    (let o = Yojson.Safe.from_string ordinary in
     Yojson.Safe.equal (member "peak_memory_bytes" o) (`Int 2097152)
     (* Both spellings: the short tag the report prints ON the row, so a row states its own counter,
        and the long one its legend expands that tag into (review round 1). *)
     && Yojson.Safe.equal (member "peak_memory_counter" o) (`String "fab-hw")
     &&
     match member "peak_memory_source" o with
     | `String s -> String.equal s "fabricated 'high-water' counter (requested bytes)"
     | _ -> false);
  p "a tune object that recorded no shipped census says null, not a label"
    (Yojson.Safe.equal
       (member "shipped_mma"
          (Yojson.Safe.from_string
             (Bench_json.tune_object ~shipped:"A" ~searches:1 ~replays:0 ~no_searches:0
                ~shipped_mma:None ~arms:[])))
       `Null)

let () =
  let dk name =
    Yojson.Safe.from_string (List.Assoc.find_exn dominant_kernels name ~equal:String.equal)
  in
  let verdict name = member "verdict" (dk name) in
  p_all "every dominant-kernel object parses as JSON" dominant_kernels ~f:(fun (_, o) -> parses o);
  p "an exact memory-bound kernel attains its roofline over its time: 1 ms of bytes in 4 ms is 25%"
    (Yojson.Safe.equal (verdict "exact") (`String "exact")
    && Yojson.Safe.equal (member "pct_of_peak" (dk "exact")) (`Int 25)
    && Yojson.Safe.equal (member "bound" (dk "exact")) (`String "memory")
    && Yojson.Safe.equal (member "ceiling" (dk "exact")) (`String "f32"));
  p "the f16-native ceiling doubles the flops constant, and compute binds at 50%"
    (let o = dk "exact f16-native" in
     Yojson.Safe.equal (member "ceiling" o) (`String "f16-native")
     && Yojson.Safe.equal (member "ceiling_flops" o) (`Float 1e13)
     && Yojson.Safe.equal (member "bound" o) (`String "compute")
     && Yojson.Safe.equal (member "pct_of_peak" o) (`Int 50));
  p_all "an inexact or opaque count prints no number, but keeps its counts on the line"
    [ ("approximate", "approximate"); ("opaque", "opaque") ]
    ~f:(fun (name, v) ->
      Yojson.Safe.equal (verdict name) (`String v)
      && Yojson.Safe.equal (member "pct_of_peak" (dk name)) `Null
      && Yojson.Safe.equal (member "bytes" (dk name)) (`Int 1_000_000_000));
  p "an inexact count on the leg that does not bind leaves the attainment exact"
    (let o = dk "exact over an inexact leg" in
     Yojson.Safe.equal (member "verdict" o) (`String "exact")
     && Yojson.Safe.equal (member "bound" o) (`String "compute")
     && Yojson.Safe.equal (member "pct_of_peak" o) (`Int 50)
     && Yojson.Safe.equal (member "bytes_exact" o) (`Bool false));
  p "one missing envelope leg is no ceiling, naming the missing leg"
    (Yojson.Safe.equal (verdict "no-ceiling (no constants)") (`String "no-ceiling")
    && Yojson.Safe.equal (member "pct_of_peak" (dk "no-ceiling (no constants)")) `Null
    &&
    match member "note" (dk "no-ceiling (no constants)") with
    | `String n -> String.is_substring n ~substring:"no peak_flops"
    | _ -> false);
  p "a GPU tensor-core kernel is not scored against the scalar f32 constant"
    (Yojson.Safe.equal (verdict "no-ceiling (gpu tensor cores)") (`String "no-ceiling")
    && Yojson.Safe.equal (member "ceiling" (dk "no-ceiling (gpu tensor cores)")) `Null);
  p "a C backend's register tile keeps the scalar ceiling, since it runs on the SIMD units"
    (match ceiling ~gpu:false ~tensorization:"tensorized" () with
    | Ok c -> String.equal c.Bench_json.tag "f32"
    | Error _ -> false);
  p "a kernel that was never timed carries no counts and says why"
    (Yojson.Safe.equal (verdict "no-kernel") (`String "no-kernel")
    && Yojson.Safe.equal (member "seg_ms" (dk "no-kernel")) `Null
    && Yojson.Safe.equal
         (member "note" (dk "no-kernel"))
         (`String "all 12 kernels declined to compile on their own"));
  p "a cell that did not run the instrument says null on the result line"
    (Yojson.Safe.equal (member "dominant_kernel" (Yojson.Safe.from_string diverged)) `Null);
  p "a cell that ran it carries the object"
    (Yojson.Safe.equal
       (member "verdict" (member "dominant_kernel" (Yojson.Safe.from_string ordinary)))
       (`String "exact"))

(* gh-ocannl-1209: the checkpoints a cell writes before its later stages, which is what a cell
   killed in one of those stages leaves behind. Printed with their prefix, as a runner writes them:
   [benchmarks/test_orchestrate.py] feeds these very lines to the driver's salvage path, so the
   prefix and the fields it reads are pinned from both sides. One is taken mid-parity on a diverged
   trajectory, the other after the timed steps, as the dominant-kernel instrument starts. *)
let checkpoints =
  [
    ( "mid-parity",
      Bench_json.checkpoint_line ~backend:"hip" ~variant:"default" ~precision:"f16"
        ~workload:"gpt2_mini_train_s1024"
        ~fixture:(Some ("fixtures/gpt2_mini_train_s1024.safetensors", 14756136))
        ~executable:"bench_gpt.exe" ~parity_steps:6 ~dominant_kernel:true ~completed_steps:2
        ~at:(Bench_json.In_parity 2) ~losses:[| 10.375; Float.nan |] );
    ( "before diagnostics",
      Bench_json.checkpoint_line ~backend:"hip" ~variant:"default" ~precision:"f16"
        ~workload:"gpt2_mini_train_s1024"
        ~fixture:(Some ("fixtures/gpt2_mini_train_s1024.safetensors", 14756136))
        ~executable:"bench_gpt.exe" ~parity_steps:6 ~dominant_kernel:true ~completed_steps:46
        ~at:Bench_json.Before_diagnostics
        ~losses:[| 10.375; 10.25; 10.125; 10.0; 9.875; 9.75 |] );
    ( "in memory, instrument off",
      Bench_json.checkpoint_line ~backend:"cc" ~variant:"self-test" ~precision:"f32"
        ~workload:"selftest-tiny" ~fixture:None ~executable:"bench_self_test.exe" ~parity_steps:2
        ~dominant_kernel:false ~completed_steps:13 ~at:Bench_json.Before_diagnostics
        ~losses:[| 1.25; 1.125 |] );
  ]

let () =
  Stdio.printf "\n=== checkpoint lines ===\n";
  List.iter checkpoints ~f:(fun (_, line) ->
      Stdio.printf "%s%s\n" Bench_json.checkpoint_prefix line);
  let parsed = List.map checkpoints ~f:(fun (name, line) -> (name, Yojson.Safe.from_string line)) in
  let stage name j = member name (member "stages" j) in
  p "the checkpoint prefix cannot be mistaken for a result line"
    (not (String.is_prefix Bench_json.checkpoint_prefix ~prefix:"{"));
  p_all "every checkpoint names itself a checkpoint, unaccepted, with its result still pending"
    parsed ~f:(fun (_, j) ->
      Yojson.Safe.equal (member "record" j) (`String "checkpoint")
      && Yojson.Safe.equal (member "accepted" j) (`Bool false)
      && Yojson.Safe.equal (stage "result" j) (`String "pending"));
  p_none "no checkpoint carries a timing" parsed ~f:(fun (_, j) ->
      List.exists [ "step_ms"; "queued_step_ms"; "compile_s"; "dominant_kernel" ] ~f:(fun k ->
          not (Yojson.Safe.equal (member k j) `Null)));
  let mid = List.Assoc.find_exn parsed ~equal:String.equal "mid-parity" in
  p "a mid-parity checkpoint keeps its completed losses, a diverged one as null"
    (Yojson.Safe.equal (member "losses" mid) (`List [ `Float 10.375; `Null ])
    && Yojson.Safe.equal (stage "parity" mid) (`String "running")
    && Yojson.Safe.equal (stage "timing" mid) (`String "pending")
    && Yojson.Safe.equal (stage "dominant_kernel" mid) (`String "pending"));
  let late = List.Assoc.find_exn parsed ~equal:String.equal "before diagnostics" in
  p "the checkpoint before the instrument says every measured stage is complete and it is running"
    (Yojson.Safe.equal (member "stages" late)
       (`Assoc
          [
            ("parity", `String "complete");
            ("warmup", `String "complete");
            ("timing", `String "complete");
            ("dominant_kernel", `String "running");
            ("result", `String "pending");
          ]));
  let off = List.Assoc.find_exn parsed ~equal:String.equal "in memory, instrument off" in
  p "an instrument switched off is skipped, and an in-memory model names no fixture"
    (Yojson.Safe.equal (stage "dominant_kernel" off) (`String "skipped")
    && Yojson.Safe.equal (member "fixture" off) `Null)

(* The negative control: without the mapping the line carries OCaml's own spellings, and this oracle
   rejects each of them — which is what makes the verdicts above evidence rather than ceremony. (A
   JSON parser that admits `NaN` as an extension still rejects `nan`.) *)
let () =
  p_all "the pre-fix spellings do not parse" [ "nan"; "inf"; "-inf" ] ~f:(fun spelling ->
      not (parses (Printf.sprintf {|{"losses":[%s]}|} spelling)));
  p_all "OCaml's own float conversion spells them the way this oracle rejects"
    [ Float.nan; Float.infinity; Float.neg_infinity ] ~f:(fun v ->
      not (parses (Printf.sprintf {|{"losses":[%.9g]}|} v)))
