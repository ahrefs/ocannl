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

    [rounds_run] and [beam_width] are the search's own record of how far it went (gh-ocannl-1137):
    the result line's [regime_knobs] carries what the configuration ASKED for, and a runner passing
    [~rounds:0] — every OCANNL benchmark runner does — runs no beam round whatever [autotune_rounds]
    says. A replayed or disabled arm ran none either way.

    [timings_contended] is the number of timing windows refused because host contention dominated
    their samples (gh-ocannl-855). A nonzero count means the finite winner, if any, came from an
    incomplete candidate set and was deliberately not written to the schedule cache.
    [timings_unbatched] is the part of that count refused for a different reason (gh-ocannl-1098):
    queued calibration measured no batch within its target, even at its rescue probe. That is what
    was measured, not a diagnosis: a queue threshold and host load stalling every batched probe read
    the same, and either leaves the measurement set incomplete. A count that persists across idle
    reruns is the threshold's signature.

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
let tune_arm ~name ~state ~searched ~cache_hit ~timing ~rounds_run ~beam_width ~timings_contended
    ~timings_unbatched ~best_ms ~best_label ~tensorized ~tensorization ~mma_statements
    ~mma_scalar_fallbacks ~mma_seeded ~mma_timed ~mma_best_ms ~terminal_failure =
  Printf.sprintf
    {|{"arm":"%s","state":"%s","searched":%b,"cache_hit":%b,"timing":"%s","rounds_run":%d,"beam_width":%d,"timings_contended":%d,"timings_unbatched":%d,"best_ms":%s,"best_label":"%s","tensorized":%b,"tensorization":%s,"mma_statements":%d,"mma_scalar_fallbacks":%d,"mma_seeded":%d,"mma_timed":%d,"mma_best_ms":%s,"terminal_failure":%s}|}
    (string name) (string state) searched cache_hit (string timing) rounds_run beam_width
    timings_contended timings_unbatched (num best_ms) (string best_label) tensorized
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

(** {1 The dominant kernel's %-of-peak (gh-ocannl-1006, the report's second column)}

    One kernel per cell: the one that takes longest when every kernel the step SHIPPED is timed on
    its own ([Bench_harness.dominant_kernel] — min-of-N, a device sync per run). Chosen by measured
    time rather than by the model's own roofline bound, because checking the model is the point:
    ranking by its prediction would pick the kernel the model already agrees with. Its attainment is
    the roofline lower bound over the measured time —
    [max (flops / peak_flops, bytes / peak_memory_bandwidth)] over [seg_ms] — so a memory-bound
    kernel is scored against bandwidth and a compute-bound one against arithmetic, and [bound] names
    which leg decided.

    Everything that makes such a number mislead is a rule here, so that the rule is written once and
    pinned by [test/operations/bench_result_line] on fabricated values. *)

type ceiling = {
  tag : string;
      (** The short name a report row carries: ["f32"] for the backend's single-precision scalar
          constant, ["f16-native"] for twice it on a target whose 16-bit arithmetic is native
          (gh-ocannl-575). *)
  ceiling_flops : float;  (** FLOP/s, FMA counted as two, as [Cost_model]'s op counts are. *)
  ceiling_bandwidth : float;  (** bytes/s. *)
  source : string;  (** Whose constants: the backend's class constants, or a config override. *)
}
(** The ceiling a kernel is scored against. *)

(** The ceiling matched to the kernel, or why there is none.

    - Both legs or nothing. A roofline with one leg missing is still a lower bound on the time, so
      the attainment it gives is a lower bound too — a compute-bound kernel read against bandwidth
      alone would score near zero, which reads as "far from peak" rather than "unscored". The C
      backends carry no class constant at all ([model_peak_flops] / [model_peak_memory_bandwidth]
      supply one per machine).
    - A tensorized kernel on a GPU backend has no ceiling. [peak_flops] is a scalar single-precision
      constant; tensor cores are a separate unit several times faster, so scoring against it reads
      above 100% on exactly the rows this column exists for, and no class constant for the mma unit
      exists to use instead. On the C backends the [Tile_mma] register tile runs on the SIMD units
      [peak_flops] describes, so it keeps the scalar ceiling.
    - A kernel whose arithmetic is all 16-bit on a target where that is native gets twice the scalar
      ceiling ([native_fp16_arithmetic], gh-ocannl-575). *)
let choose_ceiling ~gpu ~tensorization ~narrow_native ~peak_flops ~peak_memory_bandwidth =
  match (peak_flops, peak_memory_bandwidth) with
  | None, _ | _, None ->
      let missing =
        String.concat ~sep:" and "
          (List.filter_map
             [ ("peak_flops", peak_flops); ("peak_memory_bandwidth", peak_memory_bandwidth) ]
             ~f:(fun (name, leg) -> if Option.is_none leg then Some name else None))
      in
      Error
        (Printf.sprintf
           "no %s for this backend (the C backends carry no class constant; set model_peak_flops \
            and model_peak_memory_bandwidth)"
           missing)
  | Some _, Some _ when gpu && String.equal tensorization "tensorized" ->
      Error "tensor-core kernel: no class constant for the mma unit, and peak_flops is scalar f32"
  | Some (flops, flops_source), Some (bandwidth, bandwidth_source) ->
      let source =
        if String.equal flops_source bandwidth_source then flops_source
        else Printf.sprintf "flops: %s; bandwidth: %s" flops_source bandwidth_source
      in
      if narrow_native then
        Ok
          { tag = "f16-native"; ceiling_flops = 2. *. flops; ceiling_bandwidth = bandwidth; source }
      else Ok { tag = "f32"; ceiling_flops = flops; ceiling_bandwidth = bandwidth; source }

type kernel = {
  segment : int;  (** Its position among the step's kernels, in launch order. *)
  segments : int;  (** How many kernels the step shipped. *)
  declined : int;
      (** Kernels that could not be timed on their own (a hermetic compile the backend refused): the
          dominant one is dominant among the others only. *)
  writes : string;  (** The nodes it writes, the label a reader finds it by. *)
  seg_ms : float;  (** Its min-of-N time on its own, launch and sync included. *)
  segments_ms : float;  (** The same, summed over the kernels that were timed. *)
  tensorization : string;  (** [Ir.C_syntax.tensorization_name] of its own compile. *)
  flops : int;
  bytes : int;
  flops_exact : bool;
  bytes_exact : bool;
  opaque : bool;
}
(** The dominant kernel, as measured and as counted by [Ir.Cost_model.analyze]. *)

(** The [dominant_kernel] object. [verdict] says whether [pct_of_peak] is a number and, when not,
    why — in the order the report needs to say it:

    - ["no-kernel"]: nothing was timed ([kernel] is [None]; [note] says why).
    - ["opaque"]: the kernel has code the cost model cannot see, so its counts may under-count.
    - ["no-ceiling"]: nothing to score the counts against ({!choose_ceiling}'s reason).
    - ["approximate"]: the roofline leg that binds has an upper-bound count rather than an exact one
      ([Cost_model]'s [flops_approx] / [footprint_approximate]) — the same per-leg exactness rule
      the calibration fit follows, so this column and the fit agree on what counts as evidence. An
      inexact count on the leg that does NOT bind is harmless: it can only shrink, so it cannot
      overtake the exact leg, and the attainment stays exact.
    - ["exact"]: [pct_of_peak] and [bound] are set.

    The counts ride on the line whatever the verdict, so a reader can redo the arithmetic under
    another ceiling. *)
let dominant_kernel_object ?note ~ceiling (kernel : kernel option) =
  let str s = Printf.sprintf {|"%s"|} (string s) in
  let opt f = Option.value_map ~default:"null" ~f in
  let verdict, pct, bound, note =
    match kernel with
    | None -> ("no-kernel", None, None, Option.value note ~default:"no kernel was timed")
    | Some k when k.opaque -> ("opaque", None, None, "the cost model cannot see all of its code")
    | Some k -> (
        match ceiling with
        | Error reason -> ("no-ceiling", None, None, reason)
        | Ok c ->
            let compute_s = Float.of_int k.flops /. c.ceiling_flops
            and memory_s = Float.of_int k.bytes /. c.ceiling_bandwidth in
            let compute_binds = Float.(compute_s >= memory_s) in
            (* Per leg, as the calibration fit reads exactness: an inexact count is an upper bound,
               so its leg's time can only shrink. When the EXACT leg binds against that bound it
               binds against the truth too, and the attainment is exact; when the inexact leg binds,
               the number is only an upper bound on the attainment. *)
            if (compute_binds && k.flops_exact) || ((not compute_binds) && k.bytes_exact) then
              ( "exact",
                Some (100. *. Float.max compute_s memory_s /. (k.seg_ms /. 1000.)),
                Some (if compute_binds then "compute" else "memory"),
                "" )
            else
              ( "approximate",
                None,
                None,
                if compute_binds then "the binding op count is an upper bound"
                else "the binding byte count is an upper bound" ))
  in
  let ceiling_fields =
    match ceiling with
    | Ok c ->
        Printf.sprintf
          {|"ceiling":%s,"ceiling_flops":%s,"ceiling_bandwidth":%s,"ceiling_source":%s|} (str c.tag)
          (num c.ceiling_flops) (num c.ceiling_bandwidth) (str c.source)
    | Error _ ->
        {|"ceiling":null,"ceiling_flops":null,"ceiling_bandwidth":null,"ceiling_source":null|}
  in
  let kernel_fields =
    match kernel with
    | None ->
        {|"segment":null,"segments":null,"declined":null,"writes":null,"seg_ms":null,"segments_ms":null,"tensorization":null,"flops":null,"bytes":null,"flops_exact":null,"bytes_exact":null,"opaque":null|}
    | Some k ->
        Printf.sprintf
          {|"segment":%d,"segments":%d,"declined":%d,"writes":%s,"seg_ms":%s,"segments_ms":%s,"tensorization":%s,"flops":%d,"bytes":%d,"flops_exact":%b,"bytes_exact":%b,"opaque":%b|}
          k.segment k.segments k.declined (str k.writes) (num k.seg_ms) (num k.segments_ms)
          (str k.tensorization) k.flops k.bytes k.flops_exact k.bytes_exact k.opaque
  in
  Printf.sprintf {|{%s,"verdict":"%s",%s,"bound":%s,"pct_of_peak":%s,"note":%s}|} kernel_fields
    verdict ceiling_fields (opt str bound)
    (opt (num ~prec:4) pct)
    (if String.is_empty note then "null" else str note)

(** The result line [orchestrate.py] parses, as a string without its trailing newline.

    [shipped_mma] inventories every compiled step routine, including conditional host-gated SGD,
    independently of tuning and without measuring kernels. [None] reaches the wire as [null].

    [tune] is the already-built [tune] object (see [Bench_harness.tune_json]) or [None] for an
    untuned cell; [tokens_per_step] is present only for workloads that have one. The percentiles and
    [queued_ms] are milliseconds. [profile] is the name of the configuration profile this process
    resolved ([Utils.active_profile]), or [None] for no profile: the orchestrator dispatches a
    cell's regime as [--ocannl_profile=...] and checks the row against what the runner reports, so a
    regime a row claims is the one the process actually ran under (gh-ocannl-719); [regime_knobs]
    (see {!regime_knobs_object}) is the same fact per setting, which is what catches an ambient
    numerics flag that no profile name shows.

    [peak_memory] is the cell's peak device footprint over the timed steps, as
    [(bytes, counter, source)] -- or [None] for a cell that measured none, which reaches the report
    as a dash rather than a zero (gh-ocannl-1006). The counter is NAMED on the wire rather than left
    to be inferred from the framework and backend columns, because the available counters are not
    all the same quantity: an OCANNL row and a [torch.cuda.max_memory_allocated] row are both
    requested bytes off an allocator's high-water mark and compare honestly, whereas a current gauge
    sampled at step boundaries ([torch.mps], tinygrad) is a lower bound on the same window. A reader
    who cannot see which one a row carries would compare them as though they were one column.

    Two spellings of it, because a report needs the name in two places at two lengths (review round
    1): [counter] is the short tag that goes ON each table row, so a row states its own counter
    rather than leaving the reader to a section-wide list that says only which counters occur
    somewhere; [source] is the long description the legend expands that tag into.

    [dominant_kernel] is the already-built {!dominant_kernel_object}, or [None] — [null] on the wire
    — for a cell that did not run the instrument, which the report prints as a dash. *)
let result_line ~backend ~variant ~precision ~profile ~regime_knobs ~workload ~compile_s ~searched
    ?tokens_per_step ?tune ?shipped_mma ?simplify_fp_algebra ~p10 ~p50 ~p90 ~queued_ms ~timed_steps
    ~peak_memory ?dominant_kernel ~losses () =
  let tokens_field =
    match tokens_per_step with Some t -> Printf.sprintf {|"tokens_per_step":%d,|} t | None -> ""
  in
  let tune_field = match tune with Some j -> Printf.sprintf {|"tune":%s,|} j | None -> "" in
  let algebra_field =
    match simplify_fp_algebra with
    | None -> ""
    | Some (value, source) ->
        Printf.sprintf {|"simplify_fp_algebra":{"value":"%s","source":"%s"},|} (string value)
          (string source)
  in
  let profile_field =
    match profile with Some p -> Printf.sprintf {|"%s"|} (string p) | None -> "null"
  in
  let peak_bytes_field, peak_counter_field, peak_source_field =
    match peak_memory with
    | None -> ("null", "null", "null")
    | Some (bytes, counter, source) ->
        ( Int.to_string bytes,
          Printf.sprintf {|"%s"|} (string counter),
          Printf.sprintf {|"%s"|} (string source) )
  in
  Printf.sprintf
    {|{"framework":"ocannl","backend":"%s","variant":"%s","precision":"%s","profile":%s,"regime_knobs":%s,"workload":"%s","compile_s":%s,"searched":%b,%s%s%s"step_ms":{"p10":%s,"p50":%s,"p90":%s},"queued_step_ms":%s,"timed_steps":%d,"peak_memory_bytes":%s,"peak_memory_counter":%s,"peak_memory_source":%s,"dominant_kernel":%s,"shipped_mma":%s,"losses":[%s]}|}
    (string backend) (string variant) (string precision) profile_field
    (regime_knobs_object regime_knobs)
    (string workload) (fixed compile_s) searched tokens_field tune_field algebra_field (num p10)
    (num p50) (num p90) (num queued_ms) timed_steps peak_bytes_field peak_counter_field
    peak_source_field
    (Option.value dominant_kernel ~default:"null")
    (mma_object shipped_mma) (nums ~prec:9 losses)

(** {1 Checkpoints of a measurement still in progress (gh-ocannl-1209)}

    The result line is emitted once, at the very end, after every stage of the protocol — including
    the dominant-kernel instrument, which compiles and times each shipped kernel on its own and can
    cost more than the workload. A cell killed by its driver's cap during that instrument used to
    leave nothing structured behind: a TUF [gpt2_mini_train_s1024] cell produced six finite parity
    losses and then lost all of them to a 90 s cap expiring inside the instrument.

    So the protocol checkpoints what it has completed, as it completes it: one line per completed
    parity step, and one after the timed steps, before any diagnostic runs. A checkpoint carries the
    losses observed so far, the stages' statuses, the step count and the cell's identity — and NO
    timing. It is evidence that loss/parity work was done, never an accepted benchmark: its [record]
    is ["checkpoint"], its [accepted] is [false], its [result] stage is always ["pending"], and it
    is written behind {!checkpoint_prefix} so that no reader looking for a result line (a line
    starting with [{]) can pick it up. *)

(** The prefix every checkpoint line carries. [orchestrate.py] matches the same text
    ([CHECKPOINT_PREFIX]); the golden of [test/operations/bench_result_line] holds lines built here,
    which [test_orchestrate.py] reads back, so the two cannot drift apart unnoticed. *)
let checkpoint_prefix = "bench: checkpoint "

(** How far the protocol had got when the checkpoint was written. *)
type checkpoint_at =
  | In_parity of int  (** This many parity steps are complete. *)
  | Before_diagnostics  (** Every timed step is complete; the diagnostics come next. *)

(** One checkpoint, as the JSON object without its prefix. [dominant_kernel] says whether the cell
    runs the dominant-kernel instrument at all: its stage is ["skipped"] when not, and ["running"]
    from {!Before_diagnostics} when it does. [fixture] is the fixture's path and size in bytes, or
    [None] for a model fabricated in memory; the drivers stamp content digests and revisions, as
    they do on result lines. [losses] are the parity losses completed so far, in the result line's
    own spelling ([nums ~prec:9]), so a checkpoint and the result line of the same run carry the
    same bytes for the same step. *)
let checkpoint_line ~backend ~variant ~precision ~workload ~fixture ~executable ~parity_steps
    ~dominant_kernel ~completed_steps ~at ~losses =
  let parity, warmup, timing, instrument =
    match at with
    | In_parity k ->
        ((if k >= parity_steps then "complete" else "running"), "pending", "pending", "pending")
    | Before_diagnostics -> ("complete", "complete", "complete", "running")
  in
  let instrument = if dominant_kernel then instrument else "skipped" in
  let fixture_field =
    match fixture with
    | None -> "null"
    | Some (path, bytes) -> Printf.sprintf {|{"path":"%s","bytes":%d}|} (string path) bytes
  in
  Printf.sprintf
    {|{"record":"checkpoint","accepted":false,"framework":"ocannl","backend":"%s","variant":"%s","precision":"%s","workload":"%s","fixture":%s,"executable":"%s","stages":{"parity":"%s","warmup":"%s","timing":"%s","dominant_kernel":"%s","result":"pending"},"parity_steps":%d,"completed_steps":%d,"losses":[%s]}|}
    (string backend) (string variant) (string precision) (string workload) fixture_field
    (string executable) parity warmup timing instrument parity_steps completed_steps
    (nums ~prec:9 losses)
