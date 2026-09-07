(* JSON scalars and the result line of the OCANNL benchmark runners (gh-ocannl-676).

   Separate from [Bench_harness] — which is a module of the runner executables and drags in the
   whole library — so that a test can feed the line fabricated values: a diverged loss vector, a
   time that was never measured, a backend diagnostic full of control characters. The line is one
   ~300-character format with optional pre-formatted fragments and a nested object, and
   [orchestrate.py] reads a cell's result by [json.loads]ing it; before this module nothing anywhere
   parsed it, and a cell whose line does not parse is reported as a broken runner after the whole
   measurement has been paid for. *)

open Base

(** A JSON number, or [null] when the value is not finite.

    Every non-finite float in this file becomes [null] rather than OCaml's [nan] / [inf] / [-inf],
    none of which JSON has: a diverged training run is exactly the run whose evidence the result
    line exists to carry, so it must not be the run whose result line fails to parse. The consumer
    side of the same rule is [orchestrate.py]'s DIVERGED verdict — [null] in a loss vector means the
    cell ran and diverged, which is a parity failure naming its cause, not a missing cell. *)
let num ?(prec = 6) v = if Float.is_finite v then Printf.sprintf "%.*g" prec v else "null"

(** As {!num}, with a fixed number of decimals ([%f] rather than [%g]). *)
let fixed ?(prec = 3) v = if Float.is_finite v then Printf.sprintf "%.*f" prec v else "null"

(** A JSON array of numbers, each by {!num}. *)
let nums ?prec arr = String.concat ~sep:"," (Array.to_list (Array.map arr ~f:(num ?prec)))

(** Quote-and-control-character scrubbing rather than escaping: these strings are diagnostics
    (schedule labels, an exception's message) and the result line has to stay one parseable JSON
    line. JSON forbids every unescaped byte below U+0020, not just the whitespace ones, so the test
    is the code point — a NUL or an ESC from a backend diagnostic would otherwise invalidate the
    record and cost the whole measurement. *)
let string s =
  String.map s ~f:(function
    | '"' -> '\''
    | '\\' -> '/'
    | c when Char.to_int c < 0x20 || Char.to_int c = 0x7f -> ' '
    | c -> c)

(** One arm of the [tune] object: the crowned candidate of one placement arm, its search provenance,
    and how its best timed tensorized candidate compared (gh-ocannl-546). [best_ms] and
    [mma_best_ms] are [infinity] when the arm timed nothing at all, which {!num} renders [null].

    [timing] is the {!Autotune.timing_mode} every millisecond on this line was measured under
    (gh-ocannl-755) — ["queued"] or ["isolated"], always one of the two: every report carries a
    resolved objective. It is here because [best_ms], [baseline_ms] and [mma_best_ms] mean different
    quantities under the two, differing by tens of percent to 2x and not by a constant, so an
    artifact that omitted it could not be compared with another after the process exited. Taken from
    the arm's own report rather than read from configuration at emit time, which a caller's explicit
    [?timing] need not agree with.

    [state] names what the arm did about searching — the {!Autotune.outcome_name} of its outcome
    (gh-ocannl-677), one of ["searched"], ["search-died"], ["cache-replay"], ["search-disabled"],
    ["pre-search-failure"]. [searched] and [cache_hit] are that same fact projected onto the two
    booleans the wire format carried before, kept for readers that predate the field; they are NOT
    complements, and deriving the state from them is the mistake the outcome type exists to stop.

    [timings_contended] is the number of timing windows refused because host contention dominated
    their samples (gh-ocannl-855). A nonzero count means the finite winner, if any, came from an
    incomplete candidate set and was deliberately not written to the schedule cache.

    [tensorized] and [tensorization] are the two halves of the honesty of a tensorized timing
    (gh-ocannl-626). [tensorized] says the crowned SCHEDULE carries a [Tensorize]; [tensorization]
    says what the EMISSION did, as the {!Ir.C_syntax.tensorization_name} of the compiled routine's
    census — ["tensorized"], ["scalar-fallback"] (every emitted [Tile_mma] declined to the lane-0
    scalar path) or ["not-requested"] (codegen emitted no [Tile_mma] at all) — and [null] when there
    was no crowned candidate to consult, so an arm that consulted no census cannot read as
    tensorized. [mma_statements] is the denominator [mma_scalar_fallbacks] is a count out of. An arm
    with [tensorized: true] and a [tensorization] other than ["tensorized"] measured scalar code
    under a tensorized label; [orchestrate.py] marks that cell rather than letting the number stand.
*)
let tune_arm ~name ~state ~searched ~cache_hit ~timing ~timings_contended ~best_ms ~best_label
    ~tensorized ~tensorization ~mma_statements ~mma_scalar_fallbacks ~mma_seeded ~mma_timed
    ~mma_best_ms ~terminal_failure =
  Printf.sprintf
    {|{"arm":"%s","state":"%s","searched":%b,"cache_hit":%b,"timing":"%s","timings_contended":%d,"best_ms":%s,"best_label":"%s","tensorized":%b,"tensorization":%s,"mma_statements":%d,"mma_scalar_fallbacks":%d,"mma_seeded":%d,"mma_timed":%d,"mma_best_ms":%s,"terminal_failure":%s}|}
    (string name) (string state) searched cache_hit (string timing) timings_contended (num best_ms)
    (string best_label) tensorized
    (Option.value_map tensorization ~default:"null" ~f:(fun t -> Printf.sprintf {|"%s"|} (string t)))
    mma_statements mma_scalar_fallbacks mma_seeded mma_timed (num mma_best_ms)
    (Option.value_map terminal_failure ~default:"null" ~f:(fun detail ->
         Printf.sprintf {|"%s"|} (string detail)))

(** The [tune] object of the result line, over arms already rendered by {!tune_arm}.

    Three provenance totals, not two (gh-ocannl-677): [no_searches] counts the arms that neither
    searched nor replayed — [autotune_search=false] and every pre-search failure — so
    [orchestrate.py] reads that case instead of inferring it from [searches] and [replays] both
    being zero.

    [shipped_mma] is the census of the routine this cell's step times actually ran, as
    [{"tensorization": …, "statements": N, "scalar_fallbacks": N}] (gh-ocannl-626). It is a separate
    field from the arms', and authoritative over them, because a crowned ARM CANDIDATE is not always
    the shipped ARTIFACT: a gh-555 flip refinement that beats the A/B winner ships under
    [shipped: "flip"] and is deliberately not an arm at all, and on the [timing_ctx] path
    {!Autotune.tune} recompiles the winner in the production context and falls back to the untuned
    default when that replay is rejected or lands unparallelized. In both cases the arm describes a
    schedule that was discarded. [null] when the harness reported arms without recording it — which
    reads as UNKNOWN downstream, never as a tensorized cell. *)
let mma_object = function
  | None -> "null"
  | Some (tensorization, statements, scalar_fallbacks) ->
      Printf.sprintf {|{"tensorization":"%s","statements":%d,"scalar_fallbacks":%d}|}
        (string tensorization) statements scalar_fallbacks

let tune_object ~shipped ~searches ~replays ~no_searches ~shipped_mma ~arms =
  Printf.sprintf
    {|{"shipped":"%s","searches":%d,"replays":%d,"no_searches":%d,"shipped_mma":%s,"arms":[%s]}|}
    (string shipped) searches replays no_searches (mma_object shipped_mma)
    (String.concat ~sep:"," arms)

(** The [regime_knobs] object: where each key of the approximate profile's payload resolved from in
    this process, keyed by the setting's name -- [{"source":"default"}] for a key nothing set,
    [{"value":…,"source":…}] otherwise, the source being {!Utils.config_source_label}'s spelling.
    The orchestrator's regime gate reads it (gh-ocannl-719): an exact cell owns only defaults, an
    approximate cell only the approximate profile. *)
let regime_knobs_object knobs =
  "{"
  ^ String.concat ~sep:","
      (List.map knobs ~f:(fun (key, resolution) ->
           match resolution with
           | None -> Printf.sprintf {|"%s":{"source":"default"}|} (string key)
           | Some (value, source) ->
               Printf.sprintf {|"%s":{"value":"%s","source":"%s"}|} (string key) (string value)
                 (string source)))
  ^ "}"

(** The result line [orchestrate.py] parses, as a string without its trailing newline.

    [tune] is the already-built [tune] object (see [Bench_harness.tune_json]) or [None] for an
    untuned cell; [tokens_per_step] is present only for workloads that have one. The percentiles and
    [queued_ms] are milliseconds. [profile] is the name of the configuration profile this process
    resolved ([Utils.active_profile]), or [None] for no profile: the orchestrator dispatches a
    cell's regime as [--ocannl_profile=...] and checks the row against what the runner reports, so a
    regime a row claims is the one the process actually ran under (gh-ocannl-719); [regime_knobs]
    (see {!regime_knobs_object}) is the same fact per setting, which is what catches an ambient
    numerics flag that no profile name shows. *)
let result_line ~backend ~variant ~precision ~profile ~regime_knobs ~workload ~compile_s ~searched
    ?tokens_per_step ?tune ~p10 ~p50 ~p90 ~queued_ms ~timed_steps ~losses () =
  let tokens_field =
    match tokens_per_step with Some t -> Printf.sprintf {|"tokens_per_step":%d,|} t | None -> ""
  in
  let tune_field = match tune with Some j -> Printf.sprintf {|"tune":%s,|} j | None -> "" in
  let profile_field =
    match profile with Some p -> Printf.sprintf {|"%s"|} (string p) | None -> "null"
  in
  Printf.sprintf
    {|{"framework":"ocannl","backend":"%s","variant":"%s","precision":"%s","profile":%s,"regime_knobs":%s,"workload":"%s","compile_s":%s,"searched":%b,%s%s"step_ms":{"p10":%s,"p50":%s,"p90":%s},"queued_step_ms":%s,"timed_steps":%d,"losses":[%s]}|}
    (string backend) (string variant) (string precision) profile_field
    (regime_knobs_object regime_knobs)
    (string workload) (fixed compile_s) searched tokens_field tune_field (num p10) (num p50)
    (num p90) (num queued_ms) timed_steps (nums ~prec:9 losses)
