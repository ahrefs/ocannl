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
          ~timing:"queued" ~timings_contended:0 ~best_ms:Float.infinity ~best_label:"tile 32x32"
          ~tensorized:false ~tensorization:None ~mma_statements:0 ~mma_scalar_fallbacks:0
          ~mma_seeded:4 ~mma_timed:0 ~mma_best_ms:Float.infinity
          ~terminal_failure:
            (Some
               (Printf.sprintf "compile failed: \"kernel\" \\ path%c%c ESC" (Char.of_int_exn 0)
                  (Char.of_int_exn 27)));
        Bench_json.tune_arm ~name:"B" ~state:"cache-replay" ~searched:false ~cache_hit:true
          ~timing:"queued" ~timings_contended:0 ~best_ms:0.75 ~best_label:"grid 128"
          ~tensorized:true ~tensorization:(Some "scalar-fallback") ~mma_statements:2
          ~mma_scalar_fallbacks:2 ~mma_seeded:6 ~mma_timed:3 ~mma_best_ms:0.8 ~terminal_failure:None;
        (* Neither searched nor replayed: every counter zero, no winner to name. *)
        Bench_json.tune_arm ~name:"C" ~state:"search-disabled" ~searched:false ~cache_hit:false
          ~timing:"queued" ~timings_contended:0 ~best_ms:Float.infinity ~best_label:""
          ~tensorized:false ~tensorization:None ~mma_statements:0 ~mma_scalar_fallbacks:0
          ~mma_seeded:0 ~mma_timed:0 ~mma_best_ms:Float.infinity ~terminal_failure:None;
        (* An honestly tensorized winner, and an ordinary one that never asked. *)
        Bench_json.tune_arm ~name:"D" ~state:"searched" ~searched:true ~cache_hit:false
          ~timing:"queued" ~timings_contended:2 ~best_ms:0.5 ~best_label:"mma-gpu 16x16x16"
          ~tensorized:true ~tensorization:(Some "tensorized") ~mma_statements:4
          ~mma_scalar_fallbacks:0 ~mma_seeded:6 ~mma_timed:5 ~mma_best_ms:0.5 ~terminal_failure:None;
        Bench_json.tune_arm ~name:"E" ~state:"searched" ~searched:true
          ~cache_hit:false
            (* The other objective, so the golden shows both spellings on one line. *)
          ~timing:"isolated" ~timings_contended:0 ~best_ms:1.25 ~best_label:"grid 64"
          ~tensorized:false ~tensorization:(Some "not-requested") ~mma_statements:0
          ~mma_scalar_fallbacks:0 ~mma_seeded:0 ~mma_timed:0 ~mma_best_ms:Float.infinity
          ~terminal_failure:None;
      ]

let ordinary =
  Bench_json.result_line ~backend:"cc" ~variant:"default" ~precision:"f32" ~profile:None
    ~regime_knobs:[ ("tf32_matmuls", None); ("cc_backend_fast_math", None) ]
    ~workload:"mlp3" ~compile_s:2.5 ~searched:false ~p10:0.5 ~p50:0.75 ~p90:1.25 ~queued_ms:0.625
    ~timed_steps:20 ~losses:[| 2.5; 1.75; 1.25 |] ()

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
    ~p10:Float.infinity ~p50:Float.nan ~p90:Float.neg_infinity ~queued_ms:Float.nan ~timed_steps:0
    ~losses:[| 1.5; Float.nan; Float.infinity; Float.neg_infinity |]
    ()

let () =
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
  p "an unmeasured time is null, not a number"
    (List.for_all [ "p10"; "p50"; "p90" ] ~f:(fun p ->
         Yojson.Safe.equal (member p (member "step_ms" j)) `Null)
    && Yojson.Safe.equal (member "queued_step_ms" j) `Null
    && Yojson.Safe.equal (member "compile_s" j) `Null);
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
  p "an arm with no crowned candidate reports a null tensorization, not a label"
    (List.for_all [ "A"; "C" ] ~f:(fun n ->
         Yojson.Safe.equal (member "tensorization" (arm n)) `Null));
  p "the three tensorization labels reach the wire"
    (List.for_all
       [ ("B", "scalar-fallback"); ("D", "tensorized"); ("E", "not-requested") ]
       ~f:(fun (n, label) -> Yojson.Safe.equal (member "tensorization" (arm n)) (`String label)));
  p "a tensorized label over a scalar-fallback emission is visible as the pair"
    (Yojson.Safe.equal (member "tensorized" (arm "B")) (`Bool true)
    && Yojson.Safe.equal (member "tensorization" (arm "B")) (`String "scalar-fallback")
    && Yojson.Safe.equal (member "mma_statements" (arm "B")) (`Int 2)
    && Yojson.Safe.equal (member "mma_scalar_fallbacks" (arm "B")) (`Int 2));
  (* The shipped artifact's own census, which the arms cannot always speak for. *)
  let shipped_mma = member "shipped_mma" (member "tune" j) in
  p "the shipped artifact's census is carried apart from the arms'"
    (Yojson.Safe.equal (member "tensorization" shipped_mma) (`String "tensorized")
    && Yojson.Safe.equal (member "statements" shipped_mma) (`Int 4)
    && Yojson.Safe.equal (member "scalar_fallbacks" shipped_mma) (`Int 0)
    (* And it is free to disagree with the arm named as shipped: that is the case it exists for. *)
    && Yojson.Safe.equal (member "tensorization" (arm "B")) (`String "scalar-fallback"));
  p "a tune object that recorded no shipped census says null, not a label"
    (Yojson.Safe.equal
       (member "shipped_mma"
          (Yojson.Safe.from_string
             (Bench_json.tune_object ~shipped:"A" ~searches:1 ~replays:0 ~no_searches:0
                ~shipped_mma:None ~arms:[])))
       `Null)

(* The negative control: without the mapping the line carries OCaml's own spellings, and this oracle
   rejects each of them — which is what makes the verdicts above evidence rather than ceremony. (A
   JSON parser that admits `NaN` as an extension still rejects `nan`.) *)
let () =
  p "the pre-fix spellings do not parse"
    (List.for_all [ "nan"; "inf"; "-inf" ] ~f:(fun spelling ->
         not (parses (Printf.sprintf {|{"losses":[%s]}|} spelling))));
  p "OCaml's own float conversion spells them the way this oracle rejects"
    (List.for_all [ Float.nan; Float.infinity; Float.neg_infinity ] ~f:(fun v ->
         not (parses (Printf.sprintf {|{"losses":[%.9g]}|} v))))
