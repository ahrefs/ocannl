(* gh-ocannl-668: no test source prints a self-decided claim outside `Verdict`.

   gh-ocannl-601 settled the rule. A `(test)` stanza gates a run on two things — the exit status,
   and the diff against the golden — and a test that prints `<claim>: false` and exits 0 has only
   the second. That gate is promotable: the diff fails, the natural next move is `dune promote`, and
   the failure becomes the expected output. In a golden made of verdict lines a blessed regression
   and a deliberately recorded negative fact are the same text, so nothing fails again until someone
   reads the file. Routing the claim through `Verdict` adds the first gate by construction.

   The sweep that converted the 125 literal-label sites then in tree was one-time, and it left no
   mechanical trace. `test/operations/bandwidth_calibration.ml`, written afterwards for
   gh-ocannl-578, arrived with four fresh `Stdio.printf "…: %b\n"` claims and nothing failed,
   warned, or so much as remarked on it -- they were converted much later, in passing, by work whose
   subject was something else. The convention lived in prose, and a new test is written by matching
   a neighbour: in `test/operations` the neighbours are a mixture, legitimate descriptive `%b`
   prints sitting next to converted assertions, so the local example does not teach the rule. This
   is the mechanical trace.

   What it flags is a format whose LAST argument-consuming conversion is a bare `%b` at the end,
   behind a label ending in a colon, an equals sign or an arrow -- in either of two kinds. A LITERAL
   label is written out (`"k-blocks fused: %b\n"`); a COMPUTED one is built from arguments (`"%s
   aligned: %b\n"`, `"Epoch %d, loss below threshold=%b\n"`).

   Both were once out of reach, for different reasons, and gh-ocannl-624 closed both. The computed
   form had nowhere to go: `Verdict.p` takes a label, not a format, so converting a computed claim
   meant splitting the line by hand and every site was a judgement call. `Verdict.pf` and
   `Verdict.claimf` are that missing entry point, and with the sweep done the shape can be held. The
   separator was the quieter half: reading only a colon, this check could not see `"round-trip
   identity = %b\n"` -- the spelling most of the sites it was written for actually used, in
   `data_parallel`, `shard_transfer`, `test_buffer_loc` and a dozen more. Neither hole showed up as
   a failure. Both showed up as a clean report.

   A descriptive `%b` print therefore has one escape hatch left, not two. Carrying a second
   conversion no longer works, because a computed label carries one by construction; what remains is
   a named exemption below with the reason the line is not an assertion. The list is short, and the
   reason it stays short is structural: a print whose boolean is not a verdict is a census row or a
   table, and those rarely END on the boolean.

   gh-ocannl-801 closes the same escape hatch one level deeper. A parity or collection quantifier
   can sit in a file-local helper, with only [p "claim" (close got want)] left at the claim site.
   The gh-ocannl-729/746 sweeps could not find that shape by looking for the quantifier beside [p],
   and ten helpers carried the empty-population hole until a manual read found them. The second
   reader follows a Verdict boolean back to [for_all], [for_all2_exn], [is_empty], or a negated
   [exists] -- written directly in the claim's argument (gh-ocannl-908), or reached through any
   binding, helper, wrapper, match, open or module between the two (gh-ocannl-887) -- and requires a
   witness that the population is there. That reader is [Test_utils.Verdict_provenance]
   (gh-ocannl-931): one model of scope and polarity for every syntax form, which this file only
   consumes, through [quantified_claims]. What stays here is the policy -- the exemption lists,
   their one-key/one-definition contract, the diagnostics, the controls and the syntax coverage
   matrix that pins the model.

   gh-ocannl-997 adds a third reader, about DIALECT rather than gating: a claim label reported
   through both [pass_fail] and [skipped] breaks its golden on whichever host skips it, since a skip
   prints [p]'s [<claim>: true]. [Verdict.gated] picks the dialect itself; the pairing is refused
   here so that the mixed spelling does not regrow ([Verdict_scan.dialect_census]).

   gh-ocannl-1289 adds a fourth, about what a GATE may read. [Verdict.gated_all] and its siblings
   skip where [~when_] is false, so a gate computed from the population under test turns the
   regression that empties or changes it into a skip -- the gh-ocannl-1115 trap, shipped twice in
   one wave through these very forms. [Verdict_scan.gate_census] compares what the gate reads with
   what the population is built from, by lexical identity, and refuses a shared binding; a gate may
   still read what the population is COMPUTED from, which is how a capability judgment reads.

   gh-ocannl-1361 gives the readers one answer to whose a claim is. [Test_utils.Verdict_callee]
   resolves every claim name once, and the provenance layer, the line-break and dialect readers and
   the gate reader look its answer up rather than each deciding for themselves; a name that is
   [Verdict]'s only past another module's [open] -- qualified or not -- is ambiguous, still read,
   and its refusal names the remedy of qualifying the call. The agreement controls hold the readers
   to one answer per spelling. The same issue has the gate reader read [Verdict.gated]'s boolean:
   its population is what the boolean's quantifiers range over.

   gh-ocannl-1310 adds a fifth, about a gate FOLDED INTO THE VALUE: [p "…" ((not on_cpu) || ok)]
   prints [true] on a GPU -- a pass where the honest report is a skip, and the run's transcript
   cannot tell it from a CPU run that checked [ok]. [Verdict_scan.folded_gate_census] splits each
   claim's value into the disjuncts that alone pass it and refuses a run-constant one -- a value the
   configuration fixes -- beside one the test computed; [Verdict.gated] is the remedy.

   Every synthetic control below earned its place by a mutation run -- the scanner mechanism it pins
   disabled, this alias re-run, exactly that control failing. The manifest of those runs, one row
   per mechanism with the control labels and the retained `tools/test-run.sh` run ids, is
   `test/operations/verdict_ratchet_controls.md`; renaming or adding a control updates both. *)

open Base
open Stdio
module Scan = Test_utils.Verdict_scan
module Dune = Test_utils.Dune_stanza_scan
module Sources = Test_utils.Config_key_scan
module Provenance = Test_utils.Verdict_provenance

type quantifier_kind = Provenance.quantifier_kind = For_all | For_all2 | Is_empty | Not_exists

let quantifier_name = Provenance.quantifier_name

type definition_site = Provenance.site = { line : int; column : int; position : int }

(* A quantified binding reaching a claim without a witness: [helper] is the binding that wrote the
   quantifier, or -- for a quantifier written directly in the claim's argument -- the wrapper the
   claim went through, or the native claim's label; [helper_site] is that binding's definition, or
   the argument. [direct] tells the two apart for the diagnostic. *)
type quantified_claim = {
  helper : string;
  helper_site : definition_site;
  claim_line : int;
  quantifiers : quantifier_kind list;
  direct : bool;
  ambiguous : bool;
      (** The claim reached [Verdict] only past an [open] of a module [Verdict_callee] cannot read
          (gh-ocannl-1361): its refusal adds the remedy of qualifying the call. *)
}

let describe_site = Provenance.describe_site

(* Every claim the provenance layer finds, reduced to what the ratchet reports: one finding per
   binding whose quantifier the claim can rest on, and one per argument site for quantifiers written
   directly into the claim. The layer has already removed every source a witness covers. *)
let quantified_claims structure =
  Provenance.claims structure
  |> List.concat_map ~f:(fun (claim : Provenance.claim) ->
      let claim_line = claim.fired.line in
      let quantified =
        List.filter_map claim.value.sources ~f:(fun source ->
            match source.origin with
            | Provenance.Quantifier { kind; _ } -> Some (source, kind)
            | Provenance.Parameter _ | Provenance.Steering _ -> None)
      in
      let keyed =
        List.map quantified ~f:(fun (source, kind) ->
            match source.owner with
            | Some owner -> ((owner.name, owner.site, false), kind)
            | None ->
                let helper =
                  Option.value claim.helper ~default:(Provenance.label_text claim.label)
                in
                ((helper, source.written, true), kind))
      in
      List.map keyed ~f:fst
      |> List.dedup_and_sort ~compare:(fun (a, a_site, _) (b, b_site, _) ->
          match Int.compare a_site.position b_site.position with
          | 0 -> String.compare a b
          | order -> order)
      |> List.map ~f:(fun (helper, helper_site, direct) ->
          {
            helper;
            helper_site;
            claim_line;
            direct;
            ambiguous = claim.ambiguous;
            quantifiers =
              List.filter_map keyed ~f:(fun ((h, s, _), kind) ->
                  if String.equal h helper && s.position = helper_site.position then Some kind
                  else None)
              |> List.dedup_and_sort ~compare:Poly.compare;
          }))
  |> List.dedup_and_sort ~compare:(fun a b ->
      match Int.compare a.claim_line b.claim_line with
      | 0 -> (
          match Int.compare a.helper_site.position b.helper_site.position with
          | 0 -> String.compare a.helper b.helper
          | order -> order)
      | order -> order)

(* Sources whose claim-shaped literals are this check's own input rather than anything printed: the
   table that pins the shape reader on hostile formats, which has to spell the shapes out to pin
   them.

   The file is the honest unit here, not its labels one at a time. That table grows a case whenever
   the reader learns a distinction, so a list of its labels would be a second copy of it, maintained
   by whoever adds the case and read by nobody -- and the labels are fixture words like "fused",
   which say nothing about whether a literal is a print. What keeps the exemption from being a hole
   is the canary list below: two of those literals are named there, and this check fails if its scan
   of this file stops finding them. *)
let data_sources =
  [
    ( "test/operations/verdict_scan_cases.ml",
      "the fixture table for the shape reader: its claim-shaped formats are inputs, compared \
       against the labels they should yield, and never printed" );
  ]

(* Individual claim-shaped literals that are not assertions. Each has to earn its place on every run
   (see the staleness check at the end): an exemption is a claim about a line of code, and a claim
   that stops being true is not a free pass.

   LITERAL-label sites, keyed by "<repository-relative path>:<label>". Empty, and that is the state
   of the tree rather than an oversight: a bare `"<label>: %b"` line with nothing else on it is an
   assertion in every case the sweeps found. *)
let exempt_sites : (string * string) list = []

(* COMPUTED-label sites, keyed by "<repository-relative path>:<the format up to the boolean>". The
   head rather than the label, because a computed label is only what survives rendering a head whose
   arguments this reader cannot fill in -- a hint for a report, not an identity. And the head rather
   than the whole format, because the whole format IS the claim shape: a list of them written out
   here would be a list of claims in a test source, and this check would have to exempt its own file
   to hold everyone else to the rule. A head stops before the boolean, so it is not one.

   Every entry here but the first is a row of a table or a census, where the boolean records what
   happened rather than deciding whether it was right. Each also carries its assertion separately,
   through `Verdict.claim`/`claimf` on the same bound boolean, which is the pattern that lets a row
   keep its shape without losing its gate -- so what is exempted is the PRINT, never the check. An
   entry whose test has no such claim beside it is an exemption that should not have been granted,
   and that is not a hypothetical: `affine_extraction`'s parallelizability table was exempted here
   while nothing claimed it, so a conflict analysis that stopped seeing the reduction's cross-thread
   dependence would have flipped a row to `true`, exited zero, and been promotable (Codex P2, round
   2). All seven entries were audited against the invariant when that one was found.

   The first entry is the structural exception, and it is the only kind there can be: the body of
   the claim printer itself, which is not a row and has no claim beside it because it IS the claim.
   A second entry of that kind would mean a second gate. *)
let exempt_computed_sites =
  [
    ( "test/support/verdict.ml:%s: ",
      "the body of `Verdict.p` itself -- the claim printer every converted site routes THROUGH, \
       which is the one place in the tree where printing `<label>: <bool>` is the gate rather than \
       a way around it" );
    ( "test/operations/flip_bound_pruning.ml:bound pruning (not part of the golden): incumbent=%g \
       ms, timed=%d refused=%d, searches=%d, inline=%d, decisive=",
      "the evidence census on stderr: false means undecided, reported through Verdict.skipped; a \
       decisive observation is checked by the adjacent pruning claim" );
    ( "test/operations/affine_extraction.ml:%s %s parallelizable: ",
      "the per-symbol parallelizability table: a reduced axis is legitimately not parallelizable, \
       so `false` is a fact the golden pins rather than a defect" );
    ( "test/operations/bench_args_parsing.ml:%-22s option: ",
      "the argument-classification census, whose whole point is that some strings are options and \
       others are not; the assertion sits beside it as `Verdict.claim (s ^ \" classified as \
       documented\")`" );
    ( "test/operations/reduction_inline_guard.ml:small reduction (K=4): virtual=%b non-virtual=",
      "a tri-state placement row: the pair of booleans is the reading, and each is claimed \
       separately beside it" );
    ( "test/operations/reduction_inline_guard.ml:large reduction (K=64): virtual=%b non-virtual=",
      "the same row for the large reduction" );
    ( "test/operations/reduction_inline_guard.ml:dead large reduction (K=64): virtual=%b \
       non-virtual=",
      "the same row for the dead large reduction" );
    ( "test/operations/test_execution_deps.ml:%s refused, names the routine: %b, names the cause: ",
      "a two-property row about one refusal; both properties are claimed beside it" );
    ( "test/operations/observable_grads.ml:%s placement: %s; in context: %b; observable intent: ",
      "`in context` is legitimately false for a virtualized leg, so the row describes; the \
       assertion is `observable intent`, claimed beside it" );
  ]

(* Literals planted in the fixture file so that this scan has something it MUST find. They are its
   inputs there -- the two spellings whose decoded value is the claim shape -- and they are what
   says the corpus walk is still walking: an empty offender list means "no test prints a bare claim"
   only if the reader that produced it can still see one. A walk that went blind reports the same
   empty list as a clean tree, and these are the difference.

   The second is deliberately spelled over a line continuation. A reader matching text would find
   the first and miss it, which is the failure mode that argues for parsing rather than the one that
   argues for a canary -- both are worth pinning. *)
let canary_sites =
  [
    ( "test/operations/verdict_scan_cases.ml:planted canary",
      "the plain spelling, a fixture input for the shape reader" );
    ( "test/operations/verdict_scan_cases.ml:planted canary over a continuation",
      "the same literal written over a line continuation, which only a reader of decoded values \
       finds" );
  ]

(* Quantified bindings whose passing meaning genuinely ALLOWS an empty population. Keyed by
   [<repository-relative path>:<binding name>], and stale-checked below. The exemption is on the
   binding rather than every claim that uses it: the binding is the unit whose boolean semantics
   decide what empty means, and every use reaches the same decision.

   That last sentence is a precondition, not a fact about names, so it is checked below rather than
   assumed. A name is the unit only while it denotes ONE definition: a file that shadows `refused`
   with a second `refused`, or defines one per local scope, hands both to the same key, and the
   exemption -- granted after reading one body -- would license the other silently, which is exactly
   the shape the reader was widened to catch. Two definitions under one exempted key therefore
   REFUSE the run; the fix is to give them separate names, or to hoist them into one. *)
let exempt_quantified_helpers =
  [
    ( "test/operations/refused_action_scan_cases.ml:clean",
      "empty findings are the passing evidence for concrete accepted-action fixtures; adjacent \
       violating fixtures and captured shipping controls prove the scanner sees the obligation" );
    ( "test/operations/shell_scripts_parse.ml:compliant",
      "absence of private helper definitions is the passing evidence; missing lifecycle text is \
       rejected separately, and negative controls deliberately accept malformed or empty text as a \
       refusal" );
    ( "test/operations/shell_scripts_parse.ml:Harness_contract.compliant",
      "the qualified use of the same text classifier permits no private definitions; lifecycle \
       presence and the nonempty source guard separately prevent an empty script from passing" );
    ( "test/operations/backend_golden_family_scan.ml:complete",
      "empty incomplete/error lists are the passing evidence; the non-empty synthetic-control \
       population is guarded in the same binding" );
    ( "test/operations/autotune_routine_name.ml:contributed",
      "a contended search may legitimately contribute no rows; its report counters separately \
       prove whether that absence came from refused timings rather than a lost result" );
    ( "test/operations/fission_schedule.ml:annotated",
      "the merge-back case deliberately requires the consumer segment to have no hardware axes; \
       the same claim also requires the producer segment to be annotated" );
    ( "test/operations/reduction_forms.ml:changed",
      "an empty schedule deliberately needs no IR change, and an empty per-op result means there \
       were no per-op transformations to validate" );
    ( "test/operations/reduction_forms.ml:extra_ok",
      "an empty extra-fragment list deliberately means the member requires no additional emitted \
       assignment fragments" );
    (* Quantifiers written directly into a native claim are keyed by the claim's label
       (gh-ocannl-908): the sites the audited migration left on the unguarded spelling, each because
       emptiness is what the claim means and nothing in scope must be non-empty. *)
    ( "test/operations/autotune_batched_companion.ml:bc: reduction-over-j companion refutes the \
       GPU family pre-proposal",
      "the empty seed list is the consequence the claim conjoins with the family's \
       coverage-refutation witness on the sketch tree, which is the non-vacuous half and is itself \
       guarded on a non-empty refutation list" );
    ( "test/operations/autotune_batched_companion.ml:lm: coarse fission keeps the row-max \
       companion in the GEMM's segment",
      "the single coarse segment's empty seed list is conjoined with the tree's \
       coverage-refutation witness, which carries the claim" );
    ( "test/operations/autotune_smoke.ml:a replayed second report has no declines",
      "a cache replay compiles no candidate, so its census has no rows to be derived from; the \
       claim conjoins it with a zero failure count and is conditioned on the replay having \
       happened" );
    ( "test/operations/autotune_smoke.ml:search off without a cache times nothing",
      "with the search off no candidate exists to populate a schedule; the empty best schedule is \
       one of four zero counters the claim conjoins" );
    ( "test/operations/hip_scratch_tune_survives.ml:scratch/tune: the declined baseline is not \
       also counted as a gh-532 refusal",
      "conditioned on the baseline having been declined, and on a device that backs the frame the \
       decline census is legitimately empty (announced vacuous on stderr); where the baseline is \
       declined the claim just above witnesses the census" );
    ( "test/operations/seeded_default.ml:off CUDA/HIP the seeded default seeds nothing",
      "the seeded default is CUDA/HIP-only by design, so off those backends an empty seed list is \
       the claim; it is gated to them, and on CUDA/HIP the claim beside it requires a non-empty \
       list from the same fixture" );
    ( "test/operations/seeded_refine_blocks.ml:off CUDA/HIP the seeded default seeds nothing",
      "the seeded default is CUDA/HIP-only by design, so off those backends an empty seed list is \
       the claim; it is gated to them, and on CUDA/HIP the claim beside it requires exactly one \
       seed from the same fixture" );
    ( "test/operations/seeded_default.ml:a threshold above every reduction extent seeds nothing",
      "the threshold exceeds every contraction of the fixture, so no segment qualifies and an \
       empty seed list is the claim; the bitwise-gradient claim beside it witnesses the routine's \
       non-empty output" );
    ( "test/operations/reduction_sink.ml:off CUDA/HIP nothing is sunk",
      "the reduction sink is CUDA/HIP-only by design, so off those backends an empty sunk list is \
       the claim; it is gated to them, and on CUDA/HIP the claim beside it requires a non-empty \
       list from the same fixture" );
    ( "test/operations/reduction_sink.ml:with the seeding off nothing is seeded",
      "the fixture compiles at threshold 0, the documented off switch of the seeding, so that \
       every change is the sink's; an empty seed list is the claim, and seeded_default pins a \
       non-empty one at a positive threshold" );
    ( "test/operations/reduction_sink.ml:reduction_sink:false sinks nothing",
      "~reduction_sink:false is the off switch, so an empty sunk list is the claim; the claims \
       above pin a non-empty one with the sink on, and the bitwise claim beside it witnesses the \
       routine's non-empty output" );
    ( "test/operations/seeded_default.ml:a threshold of 0 disables the seeding",
      "a threshold of 0 is the documented off switch, so an empty seed list is the claim; the \
       claims above pin a non-empty one at a positive threshold on the backends that seed" );
    ( "test/operations/seeded_split_reduce.ml:off CUDA and HIP the seeded default splits nothing",
      "the untuned split runs only where the seeded default does (Autotune.seeded_split_backend), \
       so elsewhere an empty split list is the claim; it is gated there, and on CUDA and HIP the \
       claim beside it requires the bias gradient in the list from the same fixture" );
    ( "test/operations/seeded_split_reduce.ml:a threshold above every serial chain splits nothing",
      "the threshold exceeds every reduction's chain in the fixture, the loss sum's included, so \
       nothing qualifies and an empty split list is the claim; the bitwise-gradient claim beside \
       it witnesses the routine's non-empty output" );
    ( "test/operations/seeded_split_reduce.ml:a threshold of 0 disables the split",
      "a threshold of 0 is the documented off switch, so an empty split list is the claim; the \
       claims above pin a non-empty one at a positive threshold on the backends that split" );
    ( "test/operations/split_reduce_pair.ml:off CUDA and HIP the untuned plan splits nothing",
      "the untuned split runs only where the seeded default does (Autotune.seeded_split_backend), \
       so elsewhere an empty split list is the claim; it is gated there, and on CUDA and HIP the \
       claim beside it requires the bias gradient in the list from the same fixture" );
    ( "test/operations/schedule_batched_mma.ml:variance-like site: no cpu mma seeds",
      "the variance-style self-product is deliberately not recognized as a matmul or conv site, so \
       the sketch seeder yields no seeds of any kind and there is no seed population to quantify \
       the mma subset over" );
    ( "test/operations/schedule_batched_mma.ml:variance-like site: no gpu mma seeds",
      "the same site, on the GPU seeder" );
    ( "test/operations/splice_semantics.ml:cap-selected accumulator: offers no Inline flip",
      "a read-before-write node carries its entry value, so no flip candidate for it is the \
       designed reading, and a routine with no flip candidates at all offers it none; the claim \
       beside it pins the node read-before-write and an input" );
    ( "test/operations/conv_detection_boundary.ml:refused",
      "a refused conv site is deliberately recognized by no family, so both seeders yield no seeds \
       of any kind and there is no seed population to quantify over; the claim beside it pins the \
       refusal's reason on the lowered maps, which are non-empty" );
    (* Equivalences `Bool.equal <quantifier> <backend fact>` (staging#681 round 11): the reader sees
       that the quantifier's false polarity -- the empty population -- is what the claim accepts on
       one side of the fact, and on that side emptiness is the designed reading. *)
    ( "test/operations/autotune_candidate_release.ml:hoisted_attempted",
      "the equivalence `constant class grows iff hoisted candidates were attempted` is asserted \
       precisely so that a GPU backend, whose seeder proposes no hoisted candidate, reads `no \
       hoist labels, no constant growth` rather than a vacuous growth claim; the label population \
       is non-empty on cc, where the split is pinned" );
    ( "test/operations/autotune_serial_baseline.ml:the refusal is recorded in the decline census, \
       on GPU backends only",
      "on a CPU backend the baseline is dispatched and no refusal is recorded, so an empty census \
       is the passing reading there; on GPU the equivalence requires the entry" );
    ( "test/operations/autotune_smoke.ml:tensorized_schedule",
      "the flag and the schedule are read off the same winner, so a schedule with no Tensorize op \
       must agree with a false flag: the empty case is one side of the equivalence, and the \
       tensorized side is exercised by the searched report" );
    ( "test/operations/shell_scripts_parse.ml:Shebang.mentions_a_shell",
      "the scope table pairs each line with the reading the predicate must give it, and a line \
       mentioning no shell is `outside this check's scope` by design; the positive rows exercise \
       the non-empty word list" );
    ( "test/operations/agent_notes_structure.ml:every exemption still names a bullet that needs one",
      "over the file's own exemption list, which is empty today: a stale entry is reported the \
       moment one is added, and there is no population beneath an empty list to witness" );
    ( "test/operations/atomic_file_race.ml:the rerun left the scratch directory as it found it",
      "the directory must end empty, and the tree the rerun published and removed is not \
       enumerated, so nothing in scope witnesses the final listing; the fixtures an interrupted \
       run leaves behind are witnessed by the clearing claims just above" );
    ( "test/operations/simd_lane_choice.ml:a backend that renders no vectors offers no rungs",
      "a zero vector width yields no ladder by construction; the rungs the widths that do render \
       offer are pinned by the claims beside it" );
    ( "test/operations/test_cpu_topology.ml:classes well-formed",
      "a live probe of the host's core classes: a host the probe cannot classify legitimately \
       reports none, and the invariant is over whatever it found" );
    ( "test/operations/tile_mma_geometry.ml:a column extent below one vector has no default and no \
       alternatives",
      "the absence of alternatives is the geometry rule the claim states, for an extent that \
       admits no tile; the shapes that do admit one are quantified by the claims above it" );
  ]

(* Synthetic inputs state the helper rule independently of whatever helpers happen to be in the
   repository today. The first four are negative controls: the rule must return an offender for
   each, which is the same list the corpus loop below turns into a [Verdict.fail]. The rest are the
   nearest accepted forms, so widening the ratchet until ordinary boolean helpers need exemptions
   also fails here rather than growing a noisy central list. Which scanner mechanism each label
   guards, and the mutation run that proved it, is the table in verdict_ratchet_controls.md. *)
let quantified_helper_controls =
  [
    ( "refuses an unguarded for_all2_exn helper behind a local Verdict alias",
      {ocaml|let p = Verdict.p
let close got want = Array.for_all2_exn got want ~f:Float.equal
let () = p "the values agree" (close got want)|ocaml},
      [ "close" ] );
    ( "refuses an unguarded helper behind an open of Verdict.Claims",
      {ocaml|open Verdict.Claims
let close got want = Array.for_all2_exn got want ~f:Float.equal
let () = p "the values agree" (close got want)|ocaml},
      [ "close" ] );
    (* gh-ocannl-997: [Verdict.gated] is [p] with a gate, so a quantifier handed to it is read
       exactly as one handed to [p] -- qualified, and through the open. *)
    ( "refuses an unguarded quantifier handed to Verdict.gated",
      {ocaml|let close got want = Array.for_all2_exn got want ~f:Float.equal
let () = Verdict.gated ~when_:on_gpu ~on:"cc" "the values agree" (close got want)|ocaml},
      [ "close" ] );
    ( "refuses an unguarded quantifier handed to gated through an open of Verdict.Claims",
      {ocaml|open Verdict.Claims
let close got want = Array.for_all2_exn got want ~f:Float.equal
let () = gated ~when_:on_gpu ~on:"cc" "the values agree" (close got want)|ocaml},
      [ "close" ] );
    ( "keeps an open of Verdict.Claims inside its local scope",
      {ocaml|let close got want = Array.for_all2_exn got want ~f:Float.equal
let guarded () =
  let open Verdict.Claims in
  p "the values agree" (close got want)
let () = p "unrelated local function" true|ocaml},
      [ "close" ] );
    ( "refuses a sibling for_all helper through an intermediate result binding",
      {ocaml|let agrees xs = List.for_all xs ~f:Fn.id
let ok = agrees samples
let () = Verdict.claim "every sample agrees" ok|ocaml},
      [ "agrees" ] );
    ( "refuses a fully applied quantifier bound before the claim",
      {ocaml|let close = Array.for_all2_exn got want ~f:Float.equal
let () = Verdict.p "the values agree" close|ocaml},
      [ "close" ] );
    ( "refuses a quantified binding passed through a Verdict wrapper",
      {ocaml|let print_check name passed = Verdict.pass_fail ("  " ^ name) passed
let all_pass = List.for_all rows ~f:Fn.id
let () = print_check "all rows pass" all_pass|ocaml},
      [ "all_pass" ] );
    ( "refuses a direct quantifier passed through a Verdict wrapper",
      {ocaml|let print_check name passed = Verdict.pass_fail ("  " ^ name) passed
let () = print_check "all rows pass" (List.for_all rows ~f:Fn.id)|ocaml},
      [ "print_check" ] );
    ( "refuses a direct quantifier returned by an immediately invoked function",
      {ocaml|let check ok = Verdict.p "all rows pass" ok
let () = check ((fun () -> List.for_all rows ~f:Fn.id) ())|ocaml},
      [ "check" ] );
    ( "accepts a negated quantifier returned by an immediately invoked function",
      {ocaml|let check ok = Verdict.p "some row fails" ok
let () = check ((fun () -> not (List.for_all rows ~f:Fn.id)) ())|ocaml},
      [] );
    ( "refuses a quantified binding returned by an immediately invoked function",
      {ocaml|let all = List.for_all rows ~f:Fn.id
let check ok = Verdict.p "all rows pass" ok
let () = check ((fun () -> all) ())|ocaml},
      [ "all" ] );
    ( "refuses a direct quantifier called through a function alias",
      {ocaml|let every = List.for_all
let check ok = Verdict.p "all rows pass" ok
let () = check (every rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "accepts a negated quantifier called through a function alias",
      {ocaml|let every = List.for_all
let check ok = Verdict.p "some row fails" ok
let () = check (not (every rows ~f:Fn.id))|ocaml},
      [] );
    ( "accepts a guarded direct quantifier passed through a Verdict wrapper",
      {ocaml|let print_check name passed = Verdict.pass_fail ("  " ^ name) passed
let () =
  print_check "all rows pass"
    ((not (List.is_empty rows)) && List.for_all rows ~f:Fn.id)|ocaml},
      [] );
    ( "accepts a negated direct quantifier passed through a Verdict wrapper",
      {ocaml|let print_check name passed = Verdict.pass_fail ("  " ^ name) passed
let () = print_check "some row fails" (not (List.for_all rows ~f:Fn.id))|ocaml},
      [] );
    ( "refuses a direct exists negated by a labeled Verdict wrapper parameter",
      {ocaml|let check ~ok = Verdict.p "no rows match" (not ok)
let () = check ~ok:(List.exists rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "refuses a bound exists negated by a labeled Verdict wrapper parameter",
      {ocaml|let check ~ok = Verdict.p "no rows match" (not ok)
let some_match = List.exists rows ~f:Fn.id
let () = check ~ok:some_match|ocaml},
      [ "some_match" ] );
    ( "accepts a positive exists passed through a labeled Verdict wrapper parameter",
      {ocaml|let check ~ok = Verdict.p "some row matches" ok
let () = check ~ok:(List.exists rows ~f:Fn.id)|ocaml},
      [] );
    ( "uses an omitted optional default that feeds a Verdict wrapper claim",
      {ocaml|let all = List.for_all rows ~f:Fn.id
let check ?(ok = all) () = Verdict.p "all rows pass" ok
let () = check ()|ocaml},
      [ "all" ] );
    ( "does not use a Verdict wrapper default when its argument is supplied",
      {ocaml|let all = List.for_all rows ~f:Fn.id
let check ?(ok = all) () = Verdict.p "the supplied constant passes" ok
let () = check ~ok:true ()|ocaml},
      [] );
    ( "uses a Verdict wrapper default preserved through partial optional None",
      {ocaml|let check ?(ok = List.for_all rows ~f:Fn.id) () = Verdict.p "all rows pass" ok
let use = check ?ok:None
let () = use ()|ocaml},
      [ "ok" ] );
    ( "inspects the possible payload of an unknown forwarded wrapper option",
      {ocaml|let forwarded = Some (List.for_all rows ~f:Fn.id)
let check ?(ok = false) () = Verdict.p "all rows pass" ok
let () = check ?ok:forwarded ()|ocaml},
      [ "forwarded" ] );
    ( "refuses a direct quantifier passed to a partially applied Verdict claim",
      {ocaml|let check = Verdict.p "all rows pass"
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "refuses a direct quantifier passed to a curried partial Verdict wrapper",
      {ocaml|let check label = Verdict.p label
let () = check "all rows pass" (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "refuses a direct quantifier passed to a formatted partial Verdict wrapper",
      {ocaml|let check label = Verdict.pf "%s rows pass" label
let () = check "all" (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "refuses a direct quantifier passed to a partially applied local wrapper",
      {ocaml|let check label ok = Verdict.p label ok
let use = check "all rows pass"
let () = use (List.for_all rows ~f:Fn.id)|ocaml},
      [ "use" ] );
    ( "refuses a direct quantifier passed through a wrapper with tail setup",
      {ocaml|let check ok =
  let label = "all rows pass" in
  Verdict.p label ok
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "refuses a direct quantifier passed through a wrapper setup alias",
      {ocaml|let check ok =
  let result = ok in
  Verdict.p "all rows pass" result
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "does not connect a wrapper parameter hidden by a setup constant",
      {ocaml|let check ok =
  let result = true in
  Verdict.p "the constant passes" result
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [] );
    ( "refuses every quantified argument passed through a sequential wrapper",
      {ocaml|let check first second =
  Verdict.p "first rows pass" first;
  Verdict.p "second rows pass" second
let () = check (List.for_all first_rows ~f:Fn.id) true|ocaml},
      [ "check" ] );
    ( "refuses a quantified argument claimed inside wrapper control flow",
      {ocaml|let enabled = true
let check ok = if enabled then Verdict.p "all rows pass" ok else ()
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "refuses a quantified condition used as a wrapper claim value",
      {ocaml|let check ok = Verdict.p "all rows pass" (if ok then true else false)
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "accepts an inverted quantified condition used as a wrapper claim value",
      {ocaml|let check ok = Verdict.p "some row fails" (if ok then false else true)
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [] );
    ( "refuses a quantified argument claimed inside an eager wrapper call",
      {ocaml|let check ok = ignore (Verdict.p "all rows pass" ok)
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "refuses a quantified argument claimed under a local Verdict open",
      {ocaml|let check ok =
  let open Verdict.Claims in
  p "all rows pass" ok
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "refuses a quantified argument claimed by a function-case wrapper",
      {ocaml|let check = function ok -> Verdict.p "all rows pass" ok
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "refuses a quantified argument forwarded through a match wrapper",
      {ocaml|let check ok = Verdict.p "all rows pass" (match ok with value -> value)
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "refuses a quantified argument forwarded by a Boolean match wrapper",
      {ocaml|let check ok =
  Verdict.p "all rows pass" (match ok with true -> true | false -> false)
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "accepts a quantified argument inverted by a Boolean match wrapper",
      {ocaml|let check ok =
  Verdict.p "some row fails" (match ok with true -> false | false -> true)
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [] );
    ( "refuses a quantified argument claimed inside a callback",
      {ocaml|let check ok =
  List.iter [ () ] ~f:(fun () -> Verdict.p "all rows pass" ok)
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "does not connect a callback-shadowed parameter to its wrapper",
      {ocaml|let check ok =
  List.iter [ true ] ~f:(fun ok -> Verdict.p "the constant passes" ok)
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [] );
    ( "refuses a quantified argument passed to a qualified local-module wrapper",
      {ocaml|module Checks = struct
  let check ok = Verdict.p "all rows pass" ok
end
let () = Checks.check (List.for_all rows ~f:Fn.id)|ocaml},
      [ "Checks.check" ] );
    ( "refuses a quantified argument passed through an opened local module",
      {ocaml|module Checks = struct
  let check ok = Verdict.p "all rows pass" ok
end
open Checks
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "refuses a quantified helper called through a local module",
      {ocaml|module Checks = struct
  let all rows = List.for_all rows ~f:Fn.id
end
let () = Verdict.p "all rows pass" (Checks.all rows)|ocaml},
      [ "Checks.all" ] );
    ( "refuses a quantified helper called through an opened local module",
      {ocaml|module Checks = struct
  let all rows = List.for_all rows ~f:Fn.id
end
open Checks
let () = Verdict.p "all rows pass" (all rows)|ocaml},
      [ "all" ] );
    ( "accepts a fully applied quantified binding with a non-empty witness",
      {ocaml|let close =
  (not (Array.is_empty got)) && Array.for_all2_exn got want ~f:Float.equal
let () = Verdict.p "the values agree" close|ocaml},
      [] );
    ( "accepts a negated fully applied quantified binding",
      {ocaml|let differs = not (Array.for_all2_exn got want ~f:Float.equal)
let () = Verdict.p "some value differs" differs|ocaml},
      [] );
    ( "refuses a fully applied quantifier compared with true",
      {ocaml|let close = Array.for_all2_exn got want ~f:Float.equal |> Bool.equal true
let () = Verdict.p "the values agree" close|ocaml},
      [ "close" ] );
    ( "accepts a fully applied quantifier compared with false",
      {ocaml|let differs = Array.for_all2_exn got want ~f:Float.equal |> Bool.equal false
let () = Verdict.p "some value differs" differs|ocaml},
      [] );
    ( "refuses a direct Bool.equal true around a fully applied quantifier",
      {ocaml|let close = Bool.equal (List.for_all rows ~f:Fn.id) true
let () = Verdict.p "all rows pass" close|ocaml},
      [ "close" ] );
    ( "accepts a direct Bool.equal false around a fully applied quantifier",
      {ocaml|let differs = Bool.equal (List.for_all rows ~f:Fn.id) false
let () = Verdict.p "some row fails" differs|ocaml},
      [] );
    ( "refuses a bound exists compared with a false Boolean alias",
      {ocaml|let some = List.exists rows ~f:Fn.id
let no = false
let result = Bool.equal some no
let () = Verdict.p "no rows match" result|ocaml},
      [ "some" ] );
    ( "refuses a direct exists compared with a false Boolean alias",
      {ocaml|let no = false
let result = Bool.equal (List.exists rows ~f:Fn.id) no
let () = Verdict.p "no rows match" result|ocaml},
      [ "result" ] );
    ( "refuses a fully applied quantifier through a transparent Boolean wrapper",
      {ocaml|let ok = Fn.id (List.for_all rows ~f:Fn.id)
let () = Verdict.p "all rows pass" ok|ocaml},
      [ "ok" ] );
    ( "refuses a returned quantifier behind a local open",
      {ocaml|let result = let open Base in List.for_all rows ~f:Fn.id
let () = Verdict.p "all rows pass" result|ocaml},
      [ "result" ] );
    ( "refuses a returned quantifier behind local module setup",
      {ocaml|let result = let module M = struct end in List.for_all rows ~f:Fn.id
let () = Verdict.p "all rows pass" result|ocaml},
      [ "result" ] );
    ( "refuses a positive intermediate binding",
      {ocaml|let close = List.for_all rows ~f:Fn.id
let still_close = close
let () = Verdict.p "all rows pass" still_close|ocaml},
      [ "close" ] );
    ( "accepts a negated intermediate binding",
      {ocaml|let close = List.for_all rows ~f:Fn.id
let differs = not close
let () = Verdict.p "some row fails" differs|ocaml},
      [] );
    ( "accepts a piped negated intermediate binding",
      {ocaml|let close = List.for_all rows ~f:Fn.id
let differs = close |> not
let () = Verdict.p "some row fails" differs|ocaml},
      [] );
    ( "accepts a directly quantified value piped through not",
      {ocaml|let differs = List.for_all rows ~f:Fn.id |> not
let () = Verdict.p "some row fails" differs|ocaml},
      [] );
    ( "refuses a negated exists written with the application operator",
      {ocaml|let none = not @@ List.exists rows ~f:bad
let () = Verdict.p "no row is bad" none|ocaml},
      [ "none" ] );
    ( "accepts a positive bound exists",
      {ocaml|let some_bad = List.exists rows ~f:bad
let () = Verdict.p "some row is bad" some_bad|ocaml},
      [] );
    ( "refuses a negated bound exists",
      {ocaml|let some_bad = List.exists rows ~f:bad
let () = Verdict.p "no row is bad" (not some_bad)|ocaml},
      [ "some_bad" ] );
    ( "accepts a guarded intermediate binding",
      {ocaml|let close = List.for_all rows ~f:Fn.id
let guarded = (not (List.is_empty rows)) && close
let () = Verdict.p "all rows pass" guarded|ocaml},
      [] );
    ( "accepts a quantifier guarded by the same filtered population",
      {ocaml|let selected = List.filter rows ~f:eligible
let close = (not (List.is_empty selected)) && List.for_all selected ~f:Fn.id
let () = Verdict.p "all selected rows pass" close|ocaml},
      [] );
    ( "refuses a guard on a differently filtered population",
      {ocaml|let close =
  (not (List.is_empty (List.filter rows ~f:p1)))
  && List.for_all (List.filter rows ~f:p2) ~f:q
let () = Verdict.p "all selected rows pass" close|ocaml},
      [ "close" ] );
    ( "accepts an outer guard forwarded to a helper call over the same actual",
      {ocaml|let all xs = List.for_all xs ~f:Fn.id
let checked xs = (not (List.is_empty xs)) && all xs
let () = Verdict.p "all rows pass" (checked rows)|ocaml},
      [] );
    ( "refuses a helper applied to an expression that is not a population",
      {ocaml|let all xs = List.for_all xs ~f:Fn.id
let checked xs = (not (List.is_empty xs)) && all (List.rev xs)
let () = Verdict.p "all rows pass" (checked rows)|ocaml},
      [ "all" ] );
    ( "accepts a witness a helper establishes over the same actual",
      {ocaml|let present xs = not (List.is_empty xs)
let checked = present rows && List.for_all rows ~f:Fn.id
let () = Verdict.p "all rows pass" checked|ocaml},
      [] );
    ( "refuses a witness a helper establishes over a different actual",
      {ocaml|let present xs = not (List.is_empty xs)
let checked = present other_rows && List.for_all rows ~f:Fn.id
let () = Verdict.p "all rows pass" checked|ocaml},
      [ "checked" ] );
    ( "refuses a mismatched actual hidden by equal formal names",
      {ocaml|let all xs = List.for_all xs ~f:Fn.id
let checked xs other = (not (List.is_empty xs)) && all other
let () = Verdict.p "all rows pass" (checked rows other_rows)|ocaml},
      [ "all" ] );
    ( "refuses a shadowed guard identity across a nested alias",
      {ocaml|let close = List.for_all rows ~f:Fn.id
let guarded =
  let rows = [ true ] in
  (not (List.is_empty rows)) && close
let () = Verdict.p "all outer rows pass" guarded|ocaml},
      [ "close" ] );
    ( "refuses a binding nested directly inside a claim argument",
      {ocaml|let () =
  Verdict.p "all rows pass" (let ok = List.for_all rows ~f:Fn.id in ok)|ocaml},
      [ "ok" ] );
    ( "accepts a guarded binding nested directly inside a claim argument",
      {ocaml|let () =
  Verdict.p "all rows pass"
    (let ok = (not (List.is_empty rows)) && List.for_all rows ~f:Fn.id in ok)|ocaml},
      [] );
    ( "keeps an outer witness from guarding a shadowed nested population",
      {ocaml|let checked =
  (not (List.is_empty rows))
  && (let rows = [] in List.for_all rows ~f:Fn.id)
let () = Verdict.p "all rows pass" checked|ocaml},
      [ "checked" ] );
    ( "accepts a nested quantified population with its own witness",
      {ocaml|let checked =
  let rows = [ true ] in
  (not (List.is_empty rows)) && List.for_all rows ~f:Fn.id
let () = Verdict.p "all rows pass" checked|ocaml},
      [] );
    ( "accepts a negated binding nested directly inside a claim argument",
      {ocaml|let () =
  Verdict.p "some row fails" (let ok = List.for_all rows ~f:Fn.id in not ok)|ocaml},
      [] );
    ( "does not return a quantified binding shadowed by a later local",
      {ocaml|let result =
  let ok = List.for_all rows ~f:Fn.id in
  let ok = true in
  ok
let () = Verdict.p "the constant passes" result|ocaml},
      [] );
    ( "does not let an outer guard witness a match-bound population",
      {ocaml|let result =
  (not (List.is_empty rows))
  && match [] with rows -> List.for_all rows ~f:Fn.id
let () = Verdict.p "all inner rows pass" result|ocaml},
      [ "result" ] );
    ( "refuses a quantified component of a destructured tuple binding",
      {ocaml|let () =
  Verdict.p "all rows pass"
    (let ok, detail = List.for_all rows ~f:Fn.id, info in
     ok)|ocaml},
      [ "ok" ] );
    ( "conservatively refuses a quantified component of a record binding",
      {ocaml|let () =
  Verdict.p "all rows pass"
    (let { ok; detail } = { ok = List.for_all rows ~f:Fn.id; detail = info } in
     ok)|ocaml},
      [ "ok" ] );
    ( "refuses a quantified component destructured from an intermediate aggregate",
      {ocaml|let packed = List.for_all rows ~f:Fn.id, true
let ok, _ = packed
let () = Verdict.p "all rows pass" ok|ocaml},
      [ "packed" ] );
    ( "refuses a helper that returns a fully applied quantified local binding",
      {ocaml|let close xs =
  let ok = List.for_all xs ~f:Fn.id in
  ok
let () = Verdict.p "every sample agrees" (close samples)|ocaml},
      [ "close" ] );
    ( "refuses a helper that returns a match-bound quantified value",
      {ocaml|let close xs =
  match List.for_all xs ~f:Fn.id with ok -> ok
let () = Verdict.p "every sample agrees" (close samples)|ocaml},
      [ "close" ] );
    ( "refuses a direct quantifier forwarded by a Boolean constructor match",
      {ocaml|let close =
  match List.for_all rows ~f:Fn.id with true -> true | false -> false
let () = Verdict.p "all rows pass" close|ocaml},
      [ "close" ] );
    ( "refuses a direct quantifier forwarded by a wildcard Boolean match",
      {ocaml|let close =
  match List.for_all rows ~f:Fn.id with false -> false | _ -> true
let () = Verdict.p "all rows pass" close|ocaml},
      [ "close" ] );
    ( "accepts a direct quantifier inverted by a wildcard Boolean match",
      {ocaml|let differs =
  match List.for_all rows ~f:Fn.id with false -> true | _ -> false
let () = Verdict.p "some row fails" differs|ocaml},
      [] );
    ( "accepts a direct quantifier inverted by a Boolean constructor match",
      {ocaml|let differs =
  match List.for_all rows ~f:Fn.id with true -> false | false -> true
let () = Verdict.p "some row fails" differs|ocaml},
      [] );
    ( "refuses a bound exists inverted by a Boolean constructor match",
      {ocaml|let some = List.exists rows ~f:Fn.id
let none = match some with true -> false | false -> true
let () = Verdict.p "no rows match" none|ocaml},
      [ "some" ] );
    ( "refuses a bound exists inverted by aliased Boolean match outcomes",
      {ocaml|let some = List.exists rows ~f:Fn.id
let yes = true
let no = false
let result = match some with true -> no | false -> yes
let () = Verdict.p "no rows match" result|ocaml},
      [ "some" ] );
    ( "refuses a direct exists inverted by aliased Boolean match outcomes",
      {ocaml|let yes = true
let no = false
let result = match List.exists rows ~f:Fn.id with true -> no | false -> yes
let () = Verdict.p "no rows match" result|ocaml},
      [ "result" ] );
    ( "refuses a bound quantifier returned through an if condition",
      {ocaml|let all = List.for_all rows ~f:Fn.id
let close = if all then true else false
let () = Verdict.p "all rows pass" close|ocaml},
      [ "all" ] );
    ( "accepts an inverted bound quantifier returned through an if condition",
      {ocaml|let all = List.for_all rows ~f:Fn.id
let differs = if all then false else true
let () = Verdict.p "some row fails" differs|ocaml},
      [] );
    ( "refuses a direct quantifier returned through an if condition",
      {ocaml|let close = if List.for_all rows ~f:Fn.id then true else false
let () = Verdict.p "all rows pass" close|ocaml},
      [ "close" ] );
    ( "accepts an inverted direct quantifier returned through an if condition",
      {ocaml|let differs = if List.for_all rows ~f:Fn.id then false else true
let () = Verdict.p "some row fails" differs|ocaml},
      [] );
    ( "refuses a direct if condition whose false outcome is a Boolean alias",
      {ocaml|let no = false
let result = if List.exists rows ~f:Fn.id then no else true
let () = Verdict.p "no rows match" result|ocaml},
      [ "result" ] );
    ( "refuses a bound if condition whose false outcome is a Boolean alias",
      {ocaml|let some = List.exists rows ~f:Fn.id
let no = false
let result = if some then no else true
let () = Verdict.p "no rows match" result|ocaml},
      [ "some" ] );
    ( "refuses a direct if condition whose local false outcome is a Boolean alias",
      {ocaml|let result =
  let no = false in
  if List.exists rows ~f:Fn.id then no else true
let () = Verdict.p "no rows match" result|ocaml},
      [ "result" ] );
    ( "refuses an aliased quantified condition in a protected try body",
      {ocaml|let no = false
let result = try if List.exists rows ~f:Fn.id then no else true with _ -> false
let () = Verdict.p "no rows match" result|ocaml},
      [ "result" ] );
    ( "does not attribute a condition whose branches return the same literal",
      {ocaml|let all = List.for_all rows ~f:Fn.id
let result = if all then true else true
let () = Verdict.p "the constant passes" result|ocaml},
      [] );
    ( "refuses a bound quantifier returned through a match guard",
      {ocaml|let all = List.for_all rows ~f:Fn.id
let result = match () with () when all -> true | () -> false
let () = Verdict.p "all rows pass" result|ocaml},
      [ "all" ] );
    ( "accepts an inverted bound quantifier returned through a match guard",
      {ocaml|let all = List.for_all rows ~f:Fn.id
let result = match () with () when all -> false | () -> true
let () = Verdict.p "some row fails" result|ocaml},
      [] );
    ( "refuses a direct quantifier returned through a match guard",
      {ocaml|let result =
  match () with () when List.for_all rows ~f:Fn.id -> true | () -> false
let () = Verdict.p "all rows pass" result|ocaml},
      [ "result" ] );
    ( "accepts an inverted direct quantifier returned through a match guard",
      {ocaml|let result =
  match () with () when List.for_all rows ~f:Fn.id -> false | () -> true
let () = Verdict.p "some row fails" result|ocaml},
      [] );
    ( "refuses a direct match guard whose false result is a Boolean alias",
      {ocaml|let no = false
let result = match () with () when List.exists rows ~f:Fn.id -> no | () -> true
let () = Verdict.p "no rows match" result|ocaml},
      [ "result" ] );
    ( "refuses a bound match guard whose false result is a Boolean alias",
      {ocaml|let some = List.exists rows ~f:Fn.id
let no = false
let result = match () with () when some -> no | () -> true
let () = Verdict.p "no rows match" result|ocaml},
      [ "some" ] );
    ( "refuses a bound quantifier returned from a protected try body",
      {ocaml|let all = List.for_all rows ~f:Fn.id
let close = try all with _ -> false
let () = Verdict.p "all rows pass" close|ocaml},
      [ "all" ] );
    ( "refuses a direct quantifier returned through a try-case guard",
      {ocaml|let close =
  try raise Exit with
  | Exit when List.for_all rows ~f:Fn.id -> true
  | Exit -> false
let () = Verdict.p "all rows pass" close|ocaml},
      [ "close" ] );
    ( "accepts an inverted direct quantifier returned through a try-case guard",
      {ocaml|let differs =
  try raise Exit with
  | Exit when List.for_all rows ~f:Fn.id -> false
  | Exit -> true
let () = Verdict.p "some row fails" differs|ocaml},
      [] );
    ( "refuses a bound exists inverted through a try-case guard",
      {ocaml|let some = List.exists rows ~f:Fn.id
let none =
  try raise Exit with
  | Exit when some -> false
  | Exit -> true
let () = Verdict.p "no rows match" none|ocaml},
      [ "some" ] );
    ( "refuses a quantified helper reached through a mutually recursive sibling",
      {ocaml|let rec close xs = all xs
and all xs = List.for_all xs ~f:Fn.id
let () = Verdict.p "every sample agrees" (close samples)|ocaml},
      [ "all" ] );
    ( "does not resolve an outer quantified binding shadowed by a function parameter",
      {ocaml|let ok = List.for_all rows ~f:Fn.id
let identity ok = ok
let () = Verdict.p "constant identity passes" (identity true)|ocaml},
      [] );
    ( "does not resolve an outer binding shadowed by a match pattern",
      {ocaml|let ok = List.for_all rows ~f:Fn.id
let identity x = match x with ok -> ok
let () = Verdict.p "constant identity passes" (identity true)|ocaml},
      [] );
    ( "does not return an outer quantified local shadowed by a match pattern",
      {ocaml|let result =
  let ok = List.for_all rows ~f:Fn.id in
  ignore ok;
  match true with ok -> ok
let () = Verdict.p "the matched constant passes" result|ocaml},
      [] );
    ( "still resolves a non-shadowed quantified binding returned by a function",
      {ocaml|let ok = List.for_all rows ~f:Fn.id
let return_ok value = ok
let () = Verdict.p "all rows pass" (return_ok true)|ocaml},
      [ "ok" ] );
    ( "resolves an outer quantified binding used by an optional default",
      {ocaml|let ok = List.for_all rows ~f:Fn.id
let use ?(ok = ok) () = ok
let () = Verdict.p "all rows pass" (use ())|ocaml},
      [ "ok" ] );
    ( "does not use an optional default dependency when the caller supplies the argument",
      {ocaml|let outer_ok = List.for_all rows ~f:Fn.id
let use ?(ok = outer_ok) () = ok
let () = Verdict.p "the supplied constant passes" (use ~ok:true ())|ocaml},
      [] );
    ( "uses an optional default when a forwarded argument is None",
      {ocaml|let all = List.for_all rows ~f:Fn.id
let use ?(ok = all) () = ok
let () = Verdict.p "all rows pass" (use ?ok:None ())|ocaml},
      [ "all" ] );
    ( "does not use an optional default when a forwarded argument is definitely Some",
      {ocaml|let all = List.for_all rows ~f:Fn.id
let use ?(ok = all) () = ok
let () = Verdict.p "the forwarded constant passes" (use ?ok:(Some true) ())|ocaml},
      [] );
    ( "resolves a quantified binding through chained optional defaults",
      {ocaml|let all = List.for_all rows ~f:Fn.id
let use ?(x = all) ?(result = x) () = result
let () = Verdict.p "all rows pass" (use ())|ocaml},
      [ "all" ] );
    ( "suppresses an earlier default inside a later default when the caller supplies it",
      {ocaml|let all = List.for_all rows ~f:Fn.id
let use ?(x = all) ?(result = x) () = result
let () = Verdict.p "the supplied constant passes" (use ~x:true ())|ocaml},
      [] );
    ( "preserves polarity through an optional default",
      {ocaml|let ok = List.for_all rows ~f:Fn.id
let use ?(ok = not ok) () = ok
let () = Verdict.p "some row fails" (use ())|ocaml},
      [] );
    ( "refuses a quantified helper written with function-case syntax",
      {ocaml|let close = function xs -> List.for_all xs ~f:Fn.id
let () = Verdict.p "all rows pass" (close rows)|ocaml},
      [ "close" ] );
    ( "accepts a guarded quantified helper written with function-case syntax",
      {ocaml|let close = function
  | xs -> (not (List.is_empty xs)) && List.for_all xs ~f:Fn.id
let () = Verdict.p "all rows pass" (close rows)|ocaml},
      [] );
    ( "does not share a function-case guard with another case",
      {ocaml|let close = function
  | true, xs -> (not (List.is_empty xs)) && List.for_all xs ~f:Fn.id
  | false, xs -> List.for_all xs ~f:Fn.id
let () = Verdict.p "all rows pass" (close (false, rows))|ocaml},
      [ "close" ] );
    ( "accepts a helper that negates a quantified local binding",
      {ocaml|let differs xs =
  let close = List.for_all xs ~f:Fn.id in
  not close
let () = Verdict.p "some sample differs" (differs samples)|ocaml},
      [] );
    ( "refuses a double negation around a quantified local binding",
      {ocaml|let differs xs =
  let close = List.for_all xs ~f:Fn.id in
  not close
let () = Verdict.p "every sample agrees" (not (differs samples))|ocaml},
      [ "differs" ] );
    ( "refuses a quantifier written in pipeline style",
      {ocaml|let close xs = xs |> List.for_all ~f:Fn.id
let () = Verdict.p "every sample agrees" (close samples)|ocaml},
      [ "close" ] );
    ( "keeps helper resolution inside its lexical scope",
      {ocaml|let close xs = List.for_all xs ~f:Fn.id
let unrelated () =
  let close xs = (not (List.is_empty xs)) && List.for_all xs ~f:Fn.id in
  close samples
let () = Verdict.p "every sample agrees" (close samples)|ocaml},
      [ "close" ] );
    ( "refuses a reversed length upper bound masquerading as a witness",
      {ocaml|let close xs = 4 > List.length xs && List.for_all xs ~f:Fn.id
let () = Verdict.p "every sample agrees" (close samples)|ocaml},
      [ "close" ] );
    ( "preserves positive polarity through comparison with false",
      {ocaml|let close xs = List.for_all xs ~f:Fn.id
let () = Verdict.p "every sample agrees" (not (Bool.equal (close samples) false))|ocaml},
      [ "close" ] );
    ( "refuses an is_empty helper whose claim can pass on an empty source",
      {ocaml|let no_bad xs = List.is_empty (List.filter xs ~f:bad)
let () = Verdict.p "no sample is bad" (no_bad samples)|ocaml},
      [ "no_bad" ] );
    ( "refuses a negated exists helper with the same empty-population hole",
      {ocaml|let none_bad xs = not (Array.exists xs ~f:bad)
let () = Verdict.pass_fail "no sample is bad" (none_bad samples)|ocaml},
      [ "none_bad" ] );
    ( "accepts the explicit non-empty guard installed by the parity sweep",
      {ocaml|let close got want =
  (not (Array.is_empty got)) && Array.for_all2_exn got want ~f:Float.equal
let () = Verdict.p "the values agree" (close got want)|ocaml},
      [] );
    ( "does not let a guard on somebody else's population answer for the helper",
      {ocaml|let close got want other =
  (not (Array.is_empty other)) && Array.for_all2_exn got want ~f:Float.equal
let () = Verdict.p "the values agree" (close got want other)|ocaml},
      [ "close" ] );
    ( "accepts a positive literal length as the non-empty witness",
      {ocaml|let close got want =
  Array.length got = 4 && Array.for_all2_exn got want ~f:Float.equal
let () = Verdict.p "the values agree" (close got want)|ocaml},
      [] );
    ( "accepts a negated for_all2_exn discrimination helper",
      {ocaml|let differs got want = not (Array.for_all2_exn got want ~f:Float.equal)
let () = Verdict.p "some value differs" (differs got want)|ocaml},
      [] );
    ( "accepts a positive exists helper, which is false on an empty population",
      {ocaml|let some_bad xs = List.exists xs ~f:bad
let () = Verdict.p "some sample is bad" (some_bad samples)|ocaml},
      [] );
    ( "ignores a quantified helper that reaches no Verdict claim",
      {ocaml|let close got want = Array.for_all2_exn got want ~f:Float.equal
let () = if close got want then Stdio.printf "same\n"|ocaml},
      [] );
    (* gh-ocannl-908 (1): a guarded population survives a double negation. *)
    ( "accepts a guarded quantifier assigned through a double negation",
      {ocaml|let inverted = not ((not (List.is_empty rows)) && List.for_all rows ~f:Fn.id)
let () = Verdict.p "all rows pass" (not inverted)|ocaml},
      [] );
    ( "refuses a double negation without the non-empty witness",
      {ocaml|let inverted = not (List.for_all rows ~f:Fn.id)
let () = Verdict.p "all rows pass" (not inverted)|ocaml},
      [ "inverted" ] );
    ( "refuses a double negation guarded over a different population",
      {ocaml|let inverted = not ((not (List.is_empty other_rows)) && List.for_all rows ~f:Fn.id)
let () = Verdict.p "all rows pass" (not inverted)|ocaml},
      [ "inverted" ] );
    (* gh-ocannl-908 (2): a condition whose every branch returns the same Boolean is that Boolean,
       however the branches spell it; one the reader cannot prove equal still steers. *)
    ( "does not attribute a condition whose branches return the same Boolean alias",
      {ocaml|let all = List.for_all rows ~f:Fn.id
let yes = true
let result = if all then yes else true
let () = Verdict.p "the constant passes" result|ocaml},
      [] );
    ( "does not attribute a condition whose branches agree through a nested condition",
      {ocaml|let all = List.for_all rows ~f:Fn.id
let some = List.exists rows ~f:Fn.id
let result = if all then true else if some then true else true
let () = Verdict.p "the constant passes" result|ocaml},
      [] );
    ( "refuses a condition with one constant branch and one the reader cannot prove",
      {ocaml|let all = List.for_all rows ~f:Fn.id
let result = if all then true else other_flag
let () = Verdict.p "all rows pass" result|ocaml},
      [ "all" ] );
    ( "does not attribute a condition steering between two unproven branches",
      {ocaml|let all = List.for_all rows ~f:Fn.id
let result = if all then compute () else other ()
let () = Verdict.p "the computed value holds" result|ocaml},
      [] );
    ( "accepts a witness from a condition selecting the claimed branch",
      {ocaml|let result = if not (List.is_empty rows) then List.for_all rows ~f:Fn.id else false
let () = Verdict.p "all rows pass" result|ocaml},
      [] );
    (* gh-ocannl-908 (3): a wrapper's parameter shadows an outer quantified binding while its body
       is scanned; a wrapper that closes over the binding instead is still refused. *)
    ( "accepts a constant argument through a parameter shadowing an outer quantified binding",
      {ocaml|let ok = List.for_all rows ~f:Fn.id
let check ok = Verdict.p "the constant passes" ok
let () = check true|ocaml},
      [] );
    ( "still refuses the outer quantified binding a wrapper closes over",
      {ocaml|let ok = List.for_all rows ~f:Fn.id
let check () = Verdict.p "all rows pass" ok
let () = check ()|ocaml},
      [ "ok" ] );
    ( "evaluates a wrapper's optional default before its parameter shadows the name",
      {ocaml|let ok = List.for_all rows ~f:Fn.id
let check ?(ok = ok) () = Verdict.p "all rows pass" ok
let () = check ()|ocaml},
      [ "ok" ] );
    (* gh-ocannl-908 (4): a destructured wrapper parameter claims its own component only. *)
    ( "accepts a quantified sibling ignored by a destructured wrapper parameter",
      {ocaml|let check (_, ok) = Verdict.p "the constant passes" ok
let () = check (List.for_all rows ~f:Fn.id, true)|ocaml},
      [] );
    ( "refuses the claimed component of a destructured wrapper parameter",
      {ocaml|let check (_, ok) = Verdict.p "all rows pass" ok
let () = check (true, List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "accepts a quantified field ignored by a record wrapper parameter",
      {ocaml|let check { ok; _ } = Verdict.p "the constant passes" ok
let () = check { ok = true; detail = List.for_all rows ~f:Fn.id }|ocaml},
      [] );
    ( "refuses the claimed field of a record wrapper parameter",
      {ocaml|let check { ok; _ } = Verdict.p "all rows pass" ok
let () = check { ok = List.for_all rows ~f:Fn.id; detail = info }|ocaml},
      [ "check" ] );
    ( "conservatively inspects a wrapper argument its pattern cannot align",
      {ocaml|let packed = (List.for_all rows ~f:Fn.id, true)
let check (_, ok) = Verdict.p "the constant passes" ok
let () = check packed|ocaml},
      [ "packed" ] );
    (* gh-ocannl-908 (5): a quantifier written directly into a native claim. *)
    ( "refuses a quantifier written directly in a native claim",
      {ocaml|let () = Verdict.p "all rows pass" (List.for_all rows ~f:Fn.id)|ocaml},
      [ "all rows pass" ] );
    ( "accepts a guarded quantifier written directly in a native claim",
      {ocaml|let () =
  Verdict.p "all rows pass" ((not (List.is_empty rows)) && List.for_all rows ~f:Fn.id)|ocaml},
      [] );
    ( "accepts a negated quantifier written directly in a native claim",
      {ocaml|let () = Verdict.p "some row fails" (not (List.for_all rows ~f:Fn.id))|ocaml},
      [] );
    ( "refuses a quantifier written directly in a computed-label native claim",
      {ocaml|let () = Verdict.pf "%s rows pass" tag (List.for_all rows ~f:Fn.id)|ocaml},
      [ "%s rows pass" ] );
    ( "refuses a quantifier piped into a native claim",
      {ocaml|let () = List.for_all rows ~f:Fn.id |> Verdict.p "all rows pass"|ocaml},
      [ "all rows pass" ] );
    ( "refuses a quantifier written directly in an opened native claim",
      {ocaml|open Verdict.Claims
let () = p "all rows pass" (List.for_all rows ~f:Fn.id)|ocaml},
      [ "all rows pass" ] );
    ( "names a native claim's non-literal label by its expression",
      {ocaml|let () = Verdict.p (tag ^ " rows pass") (List.for_all rows ~f:Fn.id)|ocaml},
      [ "tag ^ \" rows pass\"" ] );
    (* gh-ocannl-908 (6): quantifiers reached through an open of List or Array, a local open, or a
       module alias -- and an unqualified name with none of those in scope is not one. *)
    ( "refuses a quantifier reached through an open of List",
      {ocaml|open List
let close = for_all rows ~f:Fn.id
let () = Verdict.p "all rows pass" close|ocaml},
      [ "close" ] );
    ( "accepts an inverted quantifier reached through an open of List",
      {ocaml|open List
let differs = not (for_all rows ~f:Fn.id)
let () = Verdict.p "some row fails" differs|ocaml},
      [] );
    ( "refuses a negated exists reached through an open of Array",
      {ocaml|open Array
let none = not (exists rows ~f:bad)
let () = Verdict.p "no row is bad" none|ocaml},
      [ "none" ] );
    ( "refuses a quantifier reached through a module alias",
      {ocaml|module L = List
let close = L.for_all rows ~f:Fn.id
let () = Verdict.p "all rows pass" close|ocaml},
      [ "close" ] );
    ( "refuses a quantifier reached through a local open",
      {ocaml|let close = List.(for_all rows ~f:Fn.id)
let () = Verdict.p "all rows pass" close|ocaml},
      [ "close" ] );
    ( "refuses a quantifier reached through a let-open of Array",
      {ocaml|let close =
  let open Array in
  for_all2_exn got want ~f:Float.equal
let () = Verdict.p "the values agree" close|ocaml},
      [ "close" ] );
    ( "does not resolve an unqualified for_all with no open in scope",
      {ocaml|let close = for_all rows ~f:Fn.id
let () = Verdict.p "all rows pass" close|ocaml},
      [] );
    ( "keeps a local open of List inside its scope",
      {ocaml|let guarded () =
  let open List in
  for_all rows ~f:Fn.id
let close = for_all rows ~f:Fn.id
let () = Verdict.p "all rows pass" close|ocaml},
      [] );
    (* staging#681 round 1: an unknown forwarded option keeps the default alive; a quantifier is a
       value only once its populations and predicate have all arrived; a projection out of an
       aggregate conservatively carries the aggregate. *)
    ( "keeps a wrapper's default alive through an unknown forwarded option",
      {ocaml|let check ?(ok = List.for_all rows ~f:Fn.id) () = Verdict.p "all rows pass" ok
let use opt = check ?ok:opt ()
let () = use None|ocaml},
      [ "ok" ] );
    ( "refuses a quantifier completed by a later predicate argument",
      {ocaml|let all = List.for_all rows
let () = Verdict.p "all rows pass" (all ~f:Fn.id)|ocaml},
      [ "all rows pass" ] );
    ( "refuses a for_all2_exn completed one population at a time",
      {ocaml|let agree = Array.for_all2_exn got
let () = Verdict.p "the values agree" (agree want ~f:Float.equal)|ocaml},
      [ "the values agree" ] );
    ( "does not treat a partially applied quantifier as a Boolean",
      {ocaml|let all = List.for_all rows
let () = Verdict.p "the constant passes" (let _pending = all in true)|ocaml},
      [] );
    ( "refuses a quantified field read from a record binding",
      {ocaml|let result = { ok = List.for_all rows ~f:Fn.id; detail = info }
let () = Verdict.p "all rows pass" result.ok|ocaml},
      [ "result" ] );
    ( "conservatively refuses a sibling field read from a quantified record binding",
      {ocaml|let result = { ok = List.for_all rows ~f:Fn.id; detail = true }
let () = Verdict.p "the detail holds" result.detail|ocaml},
      [ "result" ] );
    ( "refuses a quantified component read through fst",
      {ocaml|let packed = (List.for_all rows ~f:Fn.id, info)
let () = Verdict.p "all rows pass" (fst packed)|ocaml},
      [ "packed" ] );
    (* staging#681 round 2: the native claims reached through Verdict itself; a length inequality
       with zero as a witness; ordered avoidance in a match; a dynamically formatted claim partially
       applied; a condition selecting wrapper parameters that later become constants. *)
    ( "refuses a quantifier written directly in a claim opened from Verdict",
      {ocaml|open Verdict
let () = p "all rows pass" (List.for_all rows ~f:Fn.id)|ocaml},
      [ "all rows pass" ] );
    ( "refuses a quantifier written directly in a claim through a module alias of Verdict",
      {ocaml|module V = Verdict
let () = V.p "all rows pass" (List.for_all rows ~f:Fn.id)|ocaml},
      [ "all rows pass" ] );
    ( "accepts a length inequality with zero as the non-empty witness",
      {ocaml|let close xs = List.length xs <> 0 && List.for_all xs ~f:Fn.id
let () = Verdict.p "every sample agrees" (close samples)|ocaml},
      [] );
    ( "refuses a length inequality with a positive literal as a witness",
      {ocaml|let close xs = List.length xs <> 1 && List.for_all xs ~f:Fn.id
let () = Verdict.p "every sample agrees" (close samples)|ocaml},
      [ "close" ] );
    ( "does not let a later case's guard witness an earlier case's quantifier",
      {ocaml|let result =
  match () with
  | () when List.for_all rows ~f:Fn.id -> true
  | () when List.is_empty rows -> false
  | () -> false
let () = Verdict.p "all rows pass" result|ocaml},
      [ "result" ] );
    ( "accepts a witness from an earlier avoided case",
      {ocaml|let result =
  match () with
  | () when List.is_empty rows -> false
  | () when List.for_all rows ~f:Fn.id -> true
  | () -> false
let () = Verdict.p "all rows pass" result|ocaml},
      [] );
    ( "refuses a quantifier reaching a dynamically formatted claim through a partial application",
      {ocaml|let check = Verdict.pf fmt label
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [ "fmt" ] );
    ( "refuses a quantified condition selecting constant wrapper arguments",
      {ocaml|let check yes no = Verdict.p "all rows pass" (if List.for_all rows ~f:Fn.id then yes else no)
let () = check true false|ocaml},
      [ "check" ] );
    ( "accepts a quantified condition selecting inverted constant wrapper arguments",
      {ocaml|let check yes no = Verdict.p "some row fails" (if List.for_all rows ~f:Fn.id then yes else no)
let () = check false true|ocaml},
      [] );
    ( "defers a quantified condition through a wrapper that forwards its arguments",
      {ocaml|let check yes no = Verdict.p "all rows pass" (if List.for_all rows ~f:Fn.id then yes else no)
let forward a b = check a b
let () = forward true false|ocaml},
      [ "check" ] );
    (* staging#681 round 3: constructor payloads; locally rebound builtins; bound aggregates handed
       to a destructured parameter; a claim function selected by control flow. *)
    ( "refuses a quantifier carried in a constructor payload and matched out",
      {ocaml|let result = Ok (List.for_all rows ~f:Fn.id)
let () = match result with Ok ok -> Verdict.p "all rows pass" ok | Error _ -> ()|ocaml},
      [ "result" ] );
    ( "accepts a negated quantifier carried in a constructor payload",
      {ocaml|let result = Ok (not (List.for_all rows ~f:Fn.id))
let () = match result with Ok differs -> Verdict.p "some row fails" differs | Error _ -> ()|ocaml},
      [] );
    ( "reads a constructor pattern's payload exactly against a matching constructor",
      {ocaml|let () =
  Verdict.p "the constant passes"
    (match Ok (true, List.for_all rows ~f:Fn.id) with Ok (ok, _) -> ok | Error _ -> false)|ocaml},
      [] );
    ( "does not read a locally bound not as the Boolean primitive",
      {ocaml|let not _ = List.for_all rows ~f:Fn.id
let () = Verdict.p "all rows pass" (not ())|ocaml},
      [ "not" ] );
    ( "does not read a locally bound fst as a projection",
      {ocaml|let fst _ = List.for_all rows ~f:Fn.id
let () = Verdict.p "all rows pass" (fst (true, true))|ocaml},
      [ "fst" ] );
    ( "does not let a bound aggregate's witness cover a sibling formal",
      {ocaml|let check (guarded, tested) =
  Verdict.p "all rows pass" ((not (List.is_empty guarded)) && List.for_all tested ~f:Fn.id)
let pair = (nonempty, empty)
let () = check pair|ocaml},
      [ "check" ] );
    ( "accepts a literal tuple argument whose witness and quantifier share the actual",
      {ocaml|let check (guarded, tested) =
  Verdict.p "all rows pass" ((not (List.is_empty guarded)) && List.for_all tested ~f:Fn.id)
let () = check (rows, rows)|ocaml},
      [] );
    ( "refuses a quantifier passed to a native claim selected by control flow",
      {ocaml|let check = if verbose then Verdict.p else Verdict.claim
let () = check "all rows pass" (List.for_all rows ~f:Fn.id)|ocaml},
      [ "all rows pass" ] );
    ( "refuses a quantifier passed to a wrapper selected by a match",
      {ocaml|let loud ok = Verdict.p "all rows pass" ok
let quiet ok = Verdict.claim "all rows pass" ok
let check = match mode with `Loud -> loud | `Quiet -> quiet
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    (* staging#681 round 4: polymorphic variant payloads; a rebound pipeline operator; the recursive
       group's two rounds against a three-sibling chain and a cycle. *)
    ( "refuses a quantifier carried in a polymorphic variant payload and matched out",
      {ocaml|let result = `Ok (List.for_all rows ~f:Fn.id)
let () = match result with `Ok ok -> Verdict.p "all rows pass" ok | `Error -> ()|ocaml},
      [ "result" ] );
    ( "reads a variant pattern's payload exactly against a matching tag",
      {ocaml|let () =
  Verdict.p "the constant passes"
    (match `Ok (true, List.for_all rows ~f:Fn.id) with `Ok (ok, _) -> ok | `Error -> false)|ocaml},
      [] );
    ( "does not read a locally bound pipeline operator as the builtin",
      {ocaml|let ( |> ) _ _ = List.for_all rows ~f:Fn.id
let () = Verdict.p "all rows pass" (() |> ())|ocaml},
      [ "|>" ] );
    ( "refuses a quantifier reached through a three-sibling recursive chain",
      {ocaml|let rec first xs = middle xs
and middle xs = last xs
and last xs = List.for_all xs ~f:Fn.id
let () = Verdict.p "all rows pass" (first rows)|ocaml},
      [ "last" ] );
    ( "refuses a quantifier reached around a recursive cycle",
      {ocaml|let rec first xs = if stop then middle xs else last xs
and middle xs = first xs
and last xs = List.for_all xs ~f:Fn.id
let () = Verdict.p "all rows pass" (first rows)|ocaml},
      [ "last" ] );
    (* staging#681 round 5: guarded Boolean cases; a filtered view's predicate identity; the
       parameter mark inside printed text; a quantifier partially applied through a helper. *)
    ( "refuses a direct quantifier selected by a guarded Boolean case",
      {ocaml|let close = match List.for_all rows ~f:Fn.id with true when enabled -> true | _ -> false
let () = Verdict.p "all rows pass" close|ocaml},
      [ "close" ] );
    ( "accepts a direct quantifier inverted by a guarded Boolean case",
      {ocaml|let differs = match List.for_all rows ~f:Fn.id with true when enabled -> false | _ -> true
let () = Verdict.p "some row fails" differs|ocaml},
      [] );
    ( "refuses a filtered population whose predicate was rebound after the witness",
      {ocaml|let p = keep
let present = not (List.is_empty (List.filter rows ~f:p))
let p = drop
let () = Verdict.p "all kept rows pass" (present && List.for_all (List.filter rows ~f:p) ~f:Fn.id)|ocaml},
      [ "all kept rows pass" ] );
    ( "accepts a filtered population witnessed under the same predicate binding",
      {ocaml|let p = keep
let present = not (List.is_empty (List.filter rows ~f:p))
let () = Verdict.p "all kept rows pass" (present && List.for_all (List.filter rows ~f:p) ~f:Fn.id)|ocaml},
      [] );
    ( "does not mistake a parameter mark spelled inside a filter predicate",
      {ocaml|let () =
  Verdict.p "all rows pass"
    (List.for_all (List.filter rows ~f:(fun s -> String.equal s "@P")) ~f:Fn.id)|ocaml},
      [ "all rows pass" ] );
    ( "refuses a quantifier completed after its population passed through a helper",
      {ocaml|let every xs = List.for_all xs
let all_rows = every rows
let () = Verdict.p "all rows pass" (all_rows ~f:Fn.id)|ocaml},
      [ "all rows pass" ] );
    (* staging#681 round 6: many alternatives; a rebound List.length; a conjunct parameter's
       witnesses; a claim function passed as an argument; Stdlib's argument order; a recursive
       module's exports. *)
    ( "refuses a quantifier passed to the fifth of five alternative claim functions",
      {ocaml|let quiet_one _ = ()
let quiet_two _ = ()
let quiet_three _ = ()
let quiet_four _ = ()
let check =
  match mode with
  | 1 -> quiet_one
  | 2 -> quiet_two
  | 3 -> quiet_three
  | 4 -> quiet_four
  | _ -> Verdict.p "all rows pass"
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "does not read a locally bound List.length as a witness",
      {ocaml|module List = struct
  let length _ = 1
  let for_all = List.for_all
end
let close = List.length rows > 0 && List.for_all rows ~f:Fn.id
let () = Verdict.p "all rows pass" close|ocaml},
      [ "close" ] );
    ( "accepts a witness passed to a wrapper as a conjunct of its claim",
      {ocaml|let check nonempty = Verdict.p "all rows pass" (nonempty && List.for_all rows ~f:Fn.id)
let () = check (not (List.is_empty rows))|ocaml},
      [] );
    ( "refuses a witness passed to a wrapper as an alternative of its claim",
      {ocaml|let check nonempty = Verdict.p "all rows pass" (nonempty || List.for_all rows ~f:Fn.id)
let () = check (not (List.is_empty rows))|ocaml},
      [ "check" ] );
    ( "refuses a quantifier claimed through a claim function passed as an argument",
      {ocaml|let apply claim value = claim "all rows pass" value
let () = apply Verdict.p (List.for_all rows ~f:Fn.id)|ocaml},
      [ "all rows pass" ] );
    ( "refuses a quantifier claimed through a wrapper passed as an argument",
      {ocaml|let check ok = Verdict.p "all rows pass" ok
let apply claim value = claim value
let () = apply check (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "refuses a quantifier spelled in Stdlib's argument order",
      {ocaml|let () = Verdict.p "all rows pass" (Stdlib.List.for_all Fn.id rows)|ocaml},
      [ "all rows pass" ] );
    ( "accepts a negated quantifier spelled in Stdlib's argument order",
      {ocaml|let () = Verdict.p "some row fails" (not (Stdlib.List.for_all Fn.id rows))|ocaml},
      [] );
    ( "refuses a quantifier passed to a recursive module's claim wrapper",
      {ocaml|module rec Checks : sig
  val check : string -> bool -> unit
end = struct
  let check = Verdict.p
end
let () = Checks.check "all rows pass" (List.for_all rows ~f:Fn.id)|ocaml},
      [ "all rows pass" ] );
    (* staging#681 round 7: a deferred call's result; a predicate deciding its quantifier; let
       operators. *)
    ( "refuses a quantifier returned through a function parameter applied for its result",
      {ocaml|let apply all xs = all xs ~f:Fn.id
let () = Verdict.p "all rows pass" (apply List.for_all rows)|ocaml},
      [ "all rows pass" ] );
    ( "accepts a negated quantifier returned through a function parameter applied for its result",
      {ocaml|let apply all xs = not (all xs ~f:Fn.id)
let () = Verdict.p "some row fails" (apply List.for_all rows)|ocaml},
      [] );
    ( "refuses a vacuous quantifier inside the predicate deciding its quantifier",
      {ocaml|let () =
  Verdict.p "some group passes" (List.exists groups ~f:(fun rows -> List.for_all rows ~f:Fn.id))|ocaml},
      [ "some group passes" ] );
    ( "accepts a guarded quantifier inside the predicate deciding its quantifier",
      {ocaml|let () =
  Verdict.p "some group passes"
    (List.exists groups ~f:(fun rows -> (not (List.is_empty rows)) && List.for_all rows ~f:Fn.id))|ocaml},
      [] );
    ( "refuses a quantifier bound by a let operator",
      {ocaml|let ( let* ) x f = f x
let () =
  let* ok = List.for_all rows ~f:Fn.id in
  Verdict.p "all rows pass" ok|ocaml},
      [ "let*" ] );
    ( "accepts a negated quantifier bound by a let operator",
      {ocaml|let ( let* ) x f = f x
let () =
  let* differs = not (List.for_all rows ~f:Fn.id) in
  Verdict.p "some row fails" differs|ocaml},
      [] );
    (* staging#681 round 8: Stdlib through an alias; recursive modules calling later siblings; field
       and qualified populations; callbacks handed to unmodelled functions; refutable cases' guards;
       a predicate selected by control flow. *)
    ( "refuses a quantifier spelled in Stdlib's argument order through a module alias",
      {ocaml|module L = Stdlib.List
let () = Verdict.p "all rows pass" (L.for_all Fn.id rows)|ocaml},
      [ "all rows pass" ] );
    ( "refuses a quantifier passed through an earlier recursive module calling a later one",
      {ocaml|module rec A : sig
  val check : string -> bool -> unit
end = struct
  let check label value = B.check label value
end
and B : sig
  val check : string -> bool -> unit
end = struct
  let check = Verdict.p
end
let () = A.check "all rows pass" (List.for_all rows ~f:Fn.id)|ocaml},
      [ "A.check" ] );
    ( "accepts a witness over a record field guarding the same field's quantifier",
      {ocaml|let () =
  Verdict.p "all rows pass"
    ((not (List.is_empty state.rows)) && List.for_all state.rows ~f:Fn.id)|ocaml},
      [] );
    ( "refuses a witness over one record field against another's quantifier",
      {ocaml|let () =
  Verdict.p "all rows pass"
    ((not (List.is_empty state.other)) && List.for_all state.rows ~f:Fn.id)|ocaml},
      [ "all rows pass" ] );
    ( "accepts a witness over a qualified population guarding its quantifier",
      {ocaml|let () =
  Verdict.p "all rows pass" ((not (List.is_empty Fixture.rows)) && List.for_all Fixture.rows ~f:Fn.id)|ocaml},
      [] );
    ( "refuses a quantifier claimed inside a callback over the callback's own population",
      {ocaml|let () =
  List.iter groups ~f:(fun rows -> Verdict.p "all rows pass" (List.for_all rows ~f:Fn.id))|ocaml},
      [ "all rows pass" ] );
    ( "accepts a guarded quantifier claimed inside a callback over its own population",
      {ocaml|let () =
  List.iter groups ~f:(fun rows ->
      Verdict.p "all rows pass" ((not (List.is_empty rows)) && List.for_all rows ~f:Fn.id))|ocaml},
      [] );
    ( "does not let a refutable case's guard witness a later case",
      {ocaml|let result =
  match option with
  | None when List.is_empty rows -> false
  | _ -> List.for_all rows ~f:Fn.id
let () = Verdict.p "all rows pass" result|ocaml},
      [ "result" ] );
    ( "accepts an irrefutable case's guard as a witness for a later case",
      {ocaml|let result =
  match option with
  | _ when List.is_empty rows -> false
  | _ -> List.for_all rows ~f:Fn.id
let () = Verdict.p "all rows pass" result|ocaml},
      [] );
    ( "refuses a vacuous quantifier inside a predicate selected by control flow",
      {ocaml|let pred = if enabled then fun rows -> List.for_all rows ~f:Fn.id else fun _ -> false
let () = Verdict.p "some group passes" (List.exists groups ~f:pred)|ocaml},
      [ "pred" ] );
    (* staging#681 round 9: a predicate's own claims; a rebound module behind a qualified
       population; applied functors; a local module shadowing List; phys_equal; not as a value. *)
    ( "refuses a claim fired inside a predicate over its own population",
      {ocaml|let () =
  ignore
    (List.for_all groups ~f:(fun rows ->
         Verdict.p "all rows pass" (List.for_all rows ~f:Fn.id);
         true))|ocaml},
      [ "all rows pass" ] );
    ( "accepts a guarded claim fired inside a predicate over its own population",
      {ocaml|let () =
  ignore
    (List.for_all groups ~f:(fun rows ->
         Verdict.p "all rows pass" ((not (List.is_empty rows)) && List.for_all rows ~f:Fn.id);
         true))|ocaml},
      [] );
    ( "refuses a qualified population whose module was rebound after the witness",
      {ocaml|module Fixture = struct let rows = full end
let present = not (List.is_empty Fixture.rows)
module Fixture = struct let rows = [] end
let () = Verdict.p "all rows pass" (present && List.for_all Fixture.rows ~f:Fn.id)|ocaml},
      [ "all rows pass" ] );
    ( "refuses a quantifier passed to a wrapper exported by an applied functor",
      {ocaml|module Make () = struct let check = Verdict.p end
module Checks = Make ()
let () = Checks.check "all rows pass" (List.for_all rows ~f:Fn.id)|ocaml},
      [ "all rows pass" ] );
    ( "resolves an opened local module that shadows List to its own members",
      {ocaml|module List = struct let all xs = Base.List.for_all xs ~f:Fn.id end
open List
let () = Verdict.p "all rows pass" (all rows)|ocaml},
      [ "all" ] );
    ( "refuses a quantifier compared with true through phys_equal",
      {ocaml|let () = Verdict.p "all rows pass" (phys_equal (List.for_all rows ~f:Fn.id) true)|ocaml},
      [ "all rows pass" ] );
    ( "accepts a quantifier compared with false through phys_equal",
      {ocaml|let () = Verdict.p "some row fails" (phys_equal (List.for_all rows ~f:Fn.id) false)|ocaml},
      [] );
    ( "refuses a negated exists through an alias of not",
      {ocaml|let none = not
let () = Verdict.p "no row matches" (none (List.exists rows ~f:Fn.id))|ocaml},
      [ "no row matches" ] );
    ( "accepts a negated for_all through an alias of not",
      {ocaml|let none = not
let () = Verdict.p "some row fails" (none (List.for_all rows ~f:Fn.id))|ocaml},
      [] );
    (* staging#681 round 10: functor arguments; callable optional parameters; callables in
       aggregates; indexed quantifiers; longer recursive module chains; let operators applied. *)
    ( "refuses a quantifier passed through a functor parameter's claim",
      {ocaml|module Make (C : S) = struct let check = C.p end
module Checks = Make (Verdict)
let () = Checks.check "all rows pass" (List.for_all rows ~f:Fn.id)|ocaml},
      [ "all rows pass" ] );
    ( "refuses a quantifier passed to a supplied callable optional parameter",
      {ocaml|let check ?(claim = fun _ _ -> ()) value = claim "all rows pass" value
let () = check ~claim:Verdict.p (List.for_all rows ~f:Fn.id)|ocaml},
      [ "all rows pass" ] );
    ( "does not fire a callable optional parameter's inert default",
      {ocaml|let check ?(claim = fun _ _ -> ()) value = claim "all rows pass" value
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [] );
    ( "refuses a quantifier passed to a claim function stored in a record field",
      {ocaml|let callbacks = { check = Verdict.p }
let () = callbacks.check "all rows pass" (List.for_all rows ~f:Fn.id)|ocaml},
      [ "all rows pass" ] );
    ( "refuses a vacuous indexed quantifier",
      {ocaml|let () = Verdict.p "all rows pass" (List.for_alli rows ~f:(fun _ row -> row))|ocaml},
      [ "all rows pass" ] );
    ( "accepts a positive indexed exists",
      {ocaml|let () = Verdict.p "some row matches" (Array.existsi rows ~f:(fun _ row -> row))|ocaml},
      [] );
    ( "refuses a quantifier passed through a three-module recursive chain",
      {ocaml|module rec A : sig
  val check : string -> bool -> unit
end = struct
  let check label value = B.check label value
end
and B : sig
  val check : string -> bool -> unit
end = struct
  let check label value = C.check label value
end
and C : sig
  val check : string -> bool -> unit
end = struct
  let check = Verdict.p
end
let () = A.check "all rows pass" (List.for_all rows ~f:Fn.id)|ocaml},
      [ "A.check" ] );
    ( "refuses a quantifier claimed by a let operator's own definition",
      {ocaml|let ( let* ) x f =
  Verdict.p "all rows pass" x;
  f x
let () =
  let* _ = List.for_all rows ~f:Fn.id in
  ()|ocaml},
      [ "let*" ] );
    ( "conservatively binds a let operator defined elsewhere as an identity",
      {ocaml|open Let_syntax
let () =
  let* ok = List.for_all rows ~f:Fn.id in
  Verdict.p "all rows pass" ok|ocaml},
      [ "ok" ] );
    (* staging#681 round 11: a repeated pattern's guard; a functor bound by let module; two Booleans
       compared. *)
    ( "does not let a repeated pattern's guard witness a differently matched case",
      {ocaml|let result =
  match option with
  | None when List.is_empty rows -> false
  | None -> false
  | Some _ -> List.for_all rows ~f:Fn.id
let () = Verdict.p "all rows pass" result|ocaml},
      [ "result" ] );
    ( "accepts a repeated pattern's guard as a witness for the case that repeats it",
      {ocaml|let result =
  match option with
  | None when List.is_empty rows -> false
  | None -> List.for_all rows ~f:Fn.id
  | Some _ -> true
let () = Verdict.p "all rows pass" result|ocaml},
      [] );
    ( "refuses a quantifier passed through a functor parameter's claim inside an expression",
      {ocaml|let () =
  let module Make = functor (C : S) -> struct let check = C.p end in
  let module Checks = Make (Verdict) in
  Checks.check "all rows pass" (List.for_all rows ~f:Fn.id)|ocaml},
      [ "all rows pass" ] );
    ( "refuses two quantifiers compared for equality",
      {ocaml|let () = Verdict.p "the checks agree" (List.for_all rows ~f:p = List.for_all rows ~f:q)|ocaml},
      [ "the checks agree" ] );
    ( "accepts two quantifiers compared for equality under a witness",
      {ocaml|let () =
  Verdict.p "the checks agree"
    ((not (List.is_empty rows)) && List.for_all rows ~f:p = List.for_all rows ~f:q)|ocaml},
      [] );
    (* staging#681 round 12: a functor applied by its qualified path; a for loop's variable; a
       witness on a field the file assigns. *)
    ( "refuses a quantifier through a functor applied by its qualified path",
      {ocaml|module Outer = struct
  module Make (C : S) = struct
    let check = C.p
  end
end
module Checks = Outer.Make (Verdict)
let () = Checks.check "all rows pass" (List.for_all rows ~f:Fn.id)|ocaml},
      [ "all rows pass" ] );
    ( "accepts a guarded quantifier through a functor applied by its qualified path",
      {ocaml|module Outer = struct
  module Make (C : S) = struct
    let check = C.p
  end
end
module Checks = Outer.Make (Verdict)
let () =
  Checks.check "all rows pass" ((not (List.is_empty rows)) && List.for_all rows ~f:Fn.id)|ocaml},
      [] );
    ( "binds a for loop's variable apart from an outer namesake",
      {ocaml|let i = 0
let present = not (List.is_empty (List.filter rows ~f:(fun r -> r.group = i)))
let () =
  for i = 1 to 2 do
    Verdict.p "all rows pass"
      (present && List.for_all (List.filter rows ~f:(fun r -> r.group = i)) ~f:Fn.id)
  done|ocaml},
      [ "all rows pass" ] );
    ( "accepts a for loop body's quantifier witnessed on the loop's own variable",
      {ocaml|let () =
  for i = 1 to 2 do
    Verdict.p "all rows pass"
      ((not (List.is_empty (List.filter rows ~f:(fun r -> r.group = i))))
      && List.for_all (List.filter rows ~f:(fun r -> r.group = i)) ~f:Fn.id)
  done|ocaml},
      [] );
    ( "refuses a witness on a field the file assigns",
      {ocaml|let () =
  let present = not (List.is_empty state.rows) in
  state.rows <- [];
  Verdict.p "all rows pass" (present && List.for_all state.rows ~f:Fn.id)|ocaml},
      [ "all rows pass" ] );
    ( "accepts a witness on a field nothing in the file assigns",
      {ocaml|let () =
  let present = not (List.is_empty state.rows) in
  Verdict.p "all rows pass" (present && List.for_all state.rows ~f:Fn.id)|ocaml},
      [] );
    (* gh-ocannl-968: first-class Boolean operators, residual functors, deferred populations, and
       Boolean ordering retain the same provenance as their direct forms. *)
    ( "refuses a quantifier through an aliased Boolean conjunction",
      {ocaml|let both = ( && )
let () = Verdict.p "all rows pass" (both true (List.for_all rows ~f:Fn.id))|ocaml},
      [ "all rows pass" ] );
    ( "accepts a witnessed quantifier through an aliased Boolean conjunction",
      {ocaml|let both = ( && )
let () = Verdict.p "all rows pass" (not (List.is_empty rows) && both true (List.for_all rows ~f:Fn.id))|ocaml},
      [] );
    ( "accepts a shadowed Boolean conjunction alias",
      {ocaml|let ( && ) _ _ = true
let both = ( && )
let () = Verdict.p "all rows pass" (both true (List.for_all rows ~f:Fn.id))|ocaml},
      [] );
    ( "refuses a quantifier through an aliased Boolean disjunction",
      {ocaml|let both = ( || )
let () = Verdict.p "all rows pass" (both false (List.for_all rows ~f:Fn.id))|ocaml},
      [ "all rows pass" ] );
    ( "accepts a witnessed quantifier through an aliased Boolean disjunction",
      {ocaml|let both = ( || )
let () = Verdict.p "all rows pass" (not (List.is_empty rows) && both false (List.for_all rows ~f:Fn.id))|ocaml},
      [] );
    ( "accepts a shadowed Boolean disjunction alias",
      {ocaml|let ( || ) _ _ = true
let both = ( || )
let () = Verdict.p "all rows pass" (both false (List.for_all rows ~f:Fn.id))|ocaml},
      [] );
    ( "refuses full first-class ordering >",
      {ocaml|let compare = (>)
let () = Verdict.p "ok" (compare (List.for_all rows ~f:Fn.id) false)|ocaml},
      [ "ok" ] );
    ( "accepts witnessed full first-class ordering >",
      {ocaml|let compare = (>)
let () = Verdict.p "ok" (not (List.is_empty rows) && compare (List.for_all rows ~f:Fn.id) false)|ocaml},
      [] );
    ( "refuses partial first-class ordering >",
      {ocaml|let compare = (>)
let above = compare (List.for_all rows ~f:Fn.id)
let () = Verdict.p "ok" (above false)|ocaml},
      [ "above" ] );
    ( "accepts witnessed partial first-class ordering >",
      {ocaml|let compare = (>)
let above = compare (List.for_all rows ~f:Fn.id)
let () = Verdict.p "ok" (not (List.is_empty rows) && above false)|ocaml},
      [] );
    ( "refuses full first-class ordering >=",
      {ocaml|let compare = (>=)
let () = Verdict.p "ok" (compare (List.for_all rows ~f:Fn.id) true)|ocaml},
      [ "ok" ] );
    ( "accepts witnessed full first-class ordering >=",
      {ocaml|let compare = (>=)
let () = Verdict.p "ok" (not (List.is_empty rows) && compare (List.for_all rows ~f:Fn.id) true)|ocaml},
      [] );
    ( "refuses partial first-class ordering >=",
      {ocaml|let compare = (>=)
let above = compare (List.for_all rows ~f:Fn.id)
let () = Verdict.p "ok" (above true)|ocaml},
      [ "above" ] );
    ( "accepts witnessed partial first-class ordering >=",
      {ocaml|let compare = (>=)
let above = compare (List.for_all rows ~f:Fn.id)
let () = Verdict.p "ok" (not (List.is_empty rows) && above true)|ocaml},
      [] );
    ( "refuses full first-class ordering <",
      {ocaml|let compare = (<)
let () = Verdict.p "ok" (compare (not (List.for_all rows ~f:Fn.id)) true)|ocaml},
      [ "ok" ] );
    ( "accepts witnessed full first-class ordering <",
      {ocaml|let compare = (<)
let () = Verdict.p "ok" (not (List.is_empty rows) && compare (not (List.for_all rows ~f:Fn.id)) true)|ocaml},
      [] );
    ( "refuses partial first-class ordering <",
      {ocaml|let compare = (<)
let above = compare (not (List.for_all rows ~f:Fn.id))
let () = Verdict.p "ok" (above true)|ocaml},
      [ "above" ] );
    ( "accepts witnessed partial first-class ordering <",
      {ocaml|let compare = (<)
let above = compare (not (List.for_all rows ~f:Fn.id))
let () = Verdict.p "ok" (not (List.is_empty rows) && above true)|ocaml},
      [] );
    ( "refuses full first-class ordering <=",
      {ocaml|let compare = (<=)
let () = Verdict.p "ok" (compare (not (List.for_all rows ~f:Fn.id)) false)|ocaml},
      [ "ok" ] );
    ( "accepts witnessed full first-class ordering <=",
      {ocaml|let compare = (<=)
let () = Verdict.p "ok" (not (List.is_empty rows) && compare (not (List.for_all rows ~f:Fn.id)) false)|ocaml},
      [] );
    ( "refuses partial first-class ordering <=",
      {ocaml|let compare = (<=)
let above = compare (not (List.for_all rows ~f:Fn.id))
let () = Verdict.p "ok" (above false)|ocaml},
      [ "above" ] );
    ( "accepts witnessed partial first-class ordering <=",
      {ocaml|let compare = (<=)
let above = compare (not (List.for_all rows ~f:Fn.id))
let () = Verdict.p "ok" (not (List.is_empty rows) && above false)|ocaml},
      [] );
    ( "accepts opaque shadowed first-class ordering",
      {ocaml|let (>) _ _ = true
let compare = (>)
let () = Verdict.p "ok" (compare (List.for_all rows ~f:Fn.id) false)|ocaml},
      [] );
    ( "accepts opaque external first-class ordering",
      {ocaml|let compare = External.(>)
let () = Verdict.p "ok" (compare (List.for_all rows ~f:Fn.id) false)|ocaml},
      [] );
    ( "accepts opaque unknown-open first-class ordering",
      {ocaml|open External
let compare = (>)
let () = Verdict.p "ok" (compare (List.for_all rows ~f:Fn.id) false)|ocaml},
      [] );
    ( "accepts opaque aggregate first-class ordering",
      {ocaml|let compare = (>)
let () = Verdict.p "ok" (compare (true, List.for_all rows ~f:Fn.id) (false, true))|ocaml},
      [] );
    ( "refuses qualified Stdlib first-class ordering",
      {ocaml|open External
let compare = Stdlib.(>)
let () = Verdict.p "ok" (compare (List.for_all rows ~f:Fn.id) false)|ocaml},
      [ "ok" ] );
    ( "refuses qualified Base first-class ordering",
      {ocaml|open External
let compare = Base.(>)
let () = Verdict.p "ok" (compare (List.for_all rows ~f:Fn.id) false)|ocaml},
      [ "ok" ] );
    ( "refuses first-class ordering forwarded through a helper",
      {ocaml|let apply f x y = f x y
let () = Verdict.p "ok" (apply (>) (List.for_all rows ~f:Fn.id) false)|ocaml},
      [ "ok" ] );
    ( "refuses named functor-valued module argument",
      {ocaml|module Make (V : S) = struct let check = V.p end
module Use (F : functor (V : S) -> S) = struct module C = F (Verdict) let check = C.check end
module C = Use (Make)
let () = C.check "ok" (List.for_all rows ~f:Fn.id)|ocaml},
      [ "ok" ] );
    ( "accepts witnessed named functor-valued module argument",
      {ocaml|module Make (V : S) = struct let check = V.p end
module Use (F : functor (V : S) -> S) = struct module C = F (Verdict) let check = C.check end
module C = Use (Make)
let () = C.check "ok" (not (List.is_empty rows) && List.for_all rows ~f:Fn.id)|ocaml},
      [] );
    ( "refuses anonymous functor-valued module argument",
      {ocaml|module Make (V : S) = struct let check = V.p end
module Use (F : functor (V : S) -> S) = struct module C = F (Verdict) let check = C.check end
module C = Use ((functor (V : S) -> struct let check = V.p end))
let () = C.check "ok" (List.for_all rows ~f:Fn.id)|ocaml},
      [ "ok" ] );
    ( "accepts witnessed anonymous functor-valued module argument",
      {ocaml|module Make (V : S) = struct let check = V.p end
module Use (F : functor (V : S) -> S) = struct module C = F (Verdict) let check = C.check end
module C = Use ((functor (V : S) -> struct let check = V.p end))
let () = C.check "ok" (not (List.is_empty rows) && List.for_all rows ~f:Fn.id)|ocaml},
      [] );
    ( "refuses partial functor-valued module argument",
      {ocaml|module Make (Unused : S) (V : S) = struct let check = V.p end
module Use (F : functor (V : S) -> S) = struct module C = F (Verdict) let check = C.check end
module C = Use (Make (Verdict))
let () = C.check "ok" (List.for_all rows ~f:Fn.id)|ocaml},
      [ "ok" ] );
    ( "accepts witnessed partial functor-valued module argument",
      {ocaml|module Make (Unused : S) (V : S) = struct let check = V.p end
module Use (F : functor (V : S) -> S) = struct module C = F (Verdict) let check = C.check end
module C = Use (Make (Verdict))
let () = C.check "ok" (not (List.is_empty rows) && List.for_all rows ~f:Fn.id)|ocaml},
      [] );
    ( "refuses qualified functor-valued module argument",
      {ocaml|module Nested = struct module Make (V : S) = struct let check = V.p end
end
module Use (F : functor (V : S) -> S) = struct module C = F (Verdict) let check = C.check end
module C = Use (Nested.Make)
let () = C.check "ok" (List.for_all rows ~f:Fn.id)|ocaml},
      [ "ok" ] );
    ( "accepts witnessed qualified functor-valued module argument",
      {ocaml|module Nested = struct module Make (V : S) = struct let check = V.p end
end
module Use (F : functor (V : S) -> S) = struct module C = F (Verdict) let check = C.check end
module C = Use (Nested.Make)
let () = C.check "ok" (not (List.is_empty rows) && List.for_all rows ~f:Fn.id)|ocaml},
      [] );
    ( "accepts shadowed functor-valued module argument",
      {ocaml|module Make (V : S) = struct let check = V.p end
module Make (V : S) = struct let check _ _ = () end
module Use (F : functor (V : S) -> S) = struct module C = F (Verdict) let check = C.check end
module C = Use (Make)
let () = C.check "ok" (List.for_all rows ~f:Fn.id)|ocaml},
      [] );
    ( "retains named refusal recursive-empty-standard-namespace",
      {ocaml|open External
module rec Base : sig end = struct end
open Base
let both = (&&)
let () = Verdict.p "ok" (both true (List.for_all rows ~f:Fn.id))|ocaml},
      [ "ok" ] );
    ( "retains named refusal partial-ordering-optional-default",
      {ocaml|let check ?(x=true) ?(y=false) () = Verdict.p "ok" (x >= y)
let partial = check ~x:(List.for_all rows ~f:Fn.id)
let () = partial ()|ocaml},
      [ "partial" ] );
    ( "accepts full ordering with an omitted annihilating default",
      {ocaml|let check ?(x=true) ?(y=false) () = Verdict.p "ok" (x >= y)
let () = check ~x:(List.for_all rows ~f:Fn.id) ()|ocaml},
      [] );
    ( "retains named refusal supplied-unapplied-callable-default",
      {ocaml|let check ?(op=(&&)) value = Verdict.p "ok" (op true value)
let () = check ~op:(fun _ _ -> true) (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "refuses omitted unapplied callable default",
      {ocaml|let check ?(op=(&&)) value = Verdict.p "ok" (op true value)
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "refuses a quantifier through a curried functor",
      {ocaml|module Make (A : S) (B : S) = struct let check = A.p let other = B.p end
module Checks = Make (Verdict) (Verdict)
let () = Checks.check "first" (List.for_all rows ~f:Fn.id); Checks.other "second" (List.for_all rows ~f:Fn.id)|ocaml},
      [ "first"; "second" ] );
    ( "accepts a witnessed quantifier through a curried functor",
      {ocaml|module Make (A : S) (B : S) = struct let check = A.p let other = B.p end
module Checks = Make (Verdict) (Verdict)
let () = Checks.check "first" (not (List.is_empty rows) && List.for_all rows ~f:Fn.id); Checks.other "second" (not (List.is_empty rows) && List.for_all rows ~f:Fn.id)|ocaml},
      [] );
    ( "refuses a quantifier through a partially applied functor",
      {ocaml|module Make (A : S) (B : S) = struct let check = A.p let other = B.p end
module Partial = Make (Verdict)
module Checks = Partial (Verdict)
let () = Checks.check "first" (List.for_all rows ~f:Fn.id); Checks.other "second" (List.for_all rows ~f:Fn.id)|ocaml},
      [ "first"; "second" ] );
    ( "accepts a witnessed quantifier through a partially applied functor",
      {ocaml|module Make (A : S) (B : S) = struct let check = A.p let other = B.p end
module Partial = Make (Verdict)
module Checks = Partial (Verdict)
let () = Checks.check "first" (not (List.is_empty rows) && List.for_all rows ~f:Fn.id); Checks.other "second" (not (List.is_empty rows) && List.for_all rows ~f:Fn.id)|ocaml},
      [] );
    ( "accepts the same population witness through a deferred helper",
      {ocaml|let apply f xs = f xs
let all xs = List.for_all xs ~f:Fn.id
let () = Verdict.p "all rows pass" (not (List.is_empty rows) && apply all rows)|ocaml},
      [] );
    ( "refuses a different population witness through a deferred helper",
      {ocaml|let apply f xs = f xs
let all xs = List.for_all xs ~f:Fn.id
let () = Verdict.p "all rows pass" (not (List.is_empty other) && apply all rows)|ocaml},
      [ "all" ] );
    ( "accepts the same population witness through a deferred partial helper",
      {ocaml|let apply f xs = f xs
let all xs = List.for_all xs ~f:Fn.id
let forward = apply all
let () = Verdict.p "all rows pass" (not (List.is_empty rows) && forward rows)|ocaml},
      [] );
    ( "refuses a different population witness through a deferred partial helper",
      {ocaml|let apply f xs = f xs
let all xs = List.for_all xs ~f:Fn.id
let forward = apply all
let () = Verdict.p "all rows pass" (not (List.is_empty other) && forward rows)|ocaml},
      [ "all" ] );
    ( "refuses a quantifier in Boolean greater or equal ordering",
      {ocaml|let () = Verdict.p "all rows pass" ((List.for_all rows ~f:Fn.id) >= true)|ocaml},
      [ "all rows pass" ] );
    ( "accepts a witnessed quantifier in Boolean greater or equal ordering",
      {ocaml|let () = Verdict.p "all rows pass" (not (List.is_empty rows) && ((List.for_all rows ~f:Fn.id) >= true))|ocaml},
      [] );
    ( "refuses a quantifier in Boolean greater ordering",
      {ocaml|let () = Verdict.p "all rows pass" ((List.for_all rows ~f:Fn.id) > false)|ocaml},
      [ "all rows pass" ] );
    ( "accepts a witnessed quantifier in Boolean greater ordering",
      {ocaml|let () = Verdict.p "all rows pass" (not (List.is_empty rows) && ((List.for_all rows ~f:Fn.id) > false))|ocaml},
      [] );
    ( "refuses a quantifier in Boolean less or equal ordering",
      {ocaml|let () = Verdict.p "all rows pass" (true <= (List.for_all rows ~f:Fn.id))|ocaml},
      [ "all rows pass" ] );
    ( "accepts a witnessed quantifier in Boolean less or equal ordering",
      {ocaml|let () = Verdict.p "all rows pass" (not (List.is_empty rows) && (true <= (List.for_all rows ~f:Fn.id)))|ocaml},
      [] );
    ( "refuses a quantifier in Boolean less ordering",
      {ocaml|let () = Verdict.p "all rows pass" (false < (List.for_all rows ~f:Fn.id))|ocaml},
      [ "all rows pass" ] );
    ( "accepts a witnessed quantifier in Boolean less ordering",
      {ocaml|let () = Verdict.p "all rows pass" (not (List.is_empty rows) && (false < (List.for_all rows ~f:Fn.id)))|ocaml},
      [] );
    ( "accepts an annihilated aliased Boolean conjunction",
      {ocaml|let both = ( && )
let () = Verdict.p "all rows pass" (both false (List.for_all rows ~f:Fn.id))|ocaml},
      [] );
    ( "refuses a quantifier through a partially applied Boolean conjunction",
      {ocaml|let both = ( && ) true
let () = Verdict.p "all rows pass" (both (List.for_all rows ~f:Fn.id))|ocaml},
      [ "all rows pass" ] );
    ( "accepts an annihilated aliased Boolean disjunction",
      {ocaml|let both = ( || )
let () = Verdict.p "all rows pass" (both true (List.for_all rows ~f:Fn.id))|ocaml},
      [] );
    ( "refuses a quantifier through a partially applied Boolean disjunction",
      {ocaml|let both = ( || ) false
let () = Verdict.p "all rows pass" (both (List.for_all rows ~f:Fn.id))|ocaml},
      [ "all rows pass" ] );
    (* staging#697 round 1: provenance-carrying values are not necessarily Booleans, and captured
       operands obey the same scope and default ownership as direct expressions. *)
    ( "does not assume a qualified external operator is Boolean",
      {ocaml|let both = External.( && )
let () = Verdict.p "all rows pass" (both true (List.for_all rows ~f:Fn.id))|ocaml},
      [] );
    ( "does not assume an opened external operator is Boolean",
      {ocaml|open Base
open External
let both = ( && )
let () = Verdict.p "all rows pass" (both true (List.for_all rows ~f:Fn.id))|ocaml},
      [] );
    ( "refuses a quantifier through an explicitly qualified standard Boolean alias",
      {ocaml|open External
let both = Stdlib.( && )
let () = Verdict.p "all rows pass" (both true (List.for_all rows ~f:Fn.id))|ocaml},
      [ "all rows pass" ] );
    ( "refuses a quantifier after reopening standard Boolean operators",
      {ocaml|open External
open Base
let both = ( && )
let () = Verdict.p "all rows pass" (both true (List.for_all rows ~f:Fn.id))|ocaml},
      [ "all rows pass" ] );
    ( "does not use Boolean ordering for tuples carrying a quantifier",
      {ocaml|let () = Verdict.p "tuple ordering" ((true, List.for_all rows ~f:Fn.id) > (false, true))|ocaml},
      [] );
    ( "does not use Boolean ordering for bound tuples carrying a quantifier",
      {ocaml|let pair = (true, List.for_all rows ~f:Fn.id)
let () = Verdict.p "tuple ordering" (pair > (false, true))|ocaml},
      [] );
    ( "does not use Boolean ordering for constructor payloads",
      {ocaml|let () = Verdict.p "constructor ordering" (Some (List.for_all rows ~f:Fn.id) > None)|ocaml},
      [] );
    ( "does not use Boolean ordering for record payloads",
      {ocaml|let () = Verdict.p "record ordering" ({ first = true; second = List.for_all rows ~f:Fn.id } > { first = false; second = true })|ocaml},
      [] );
    ( "refuses ordering of a bound Boolean quantifier",
      {ocaml|let all = List.for_all rows ~f:Fn.id
let () = Verdict.p "all rows pass" (all >= true)|ocaml},
      [ "all" ] );
    ( "refuses an omitted partial Boolean default",
      {ocaml|let check ?(both = (&&) (List.for_all rows ~f:Fn.id)) value = Verdict.p "ok" (both value)
let () = check true|ocaml},
      [ "both" ] );
    ( "accepts a supplied replacement for a partial Boolean default",
      {ocaml|let check ?(both = (&&) (List.for_all rows ~f:Fn.id)) value = Verdict.p "ok" (both value)
let () = check ~both:((||) true) true|ocaml},
      [] );
    ( "accepts a supplied replacement for a selected partial Boolean default",
      {ocaml|let check ?(both = if flag then (&&) (List.for_all rows ~f:Fn.id) else (&&) true) value = Verdict.p "ok" (both value)
let () = check ~both:((||) true) true|ocaml},
      [] );
    ( "does not assume operators from an aliased external module are Boolean",
      {ocaml|module E = External
let both = E.( && )
let () = Verdict.p "all rows pass" (both true (List.for_all rows ~f:Fn.id))|ocaml},
      [] );
    ( "keeps unknown local opens out of the outer Boolean alias scope",
      {ocaml|let ignored = External.(let both = ( && ) in both)
let both = ( && )
let () = Verdict.p "all rows pass" (both true (List.for_all rows ~f:Fn.id))|ocaml},
      [ "all rows pass" ] );
    ( "keeps unknown nested module opens out of the outer Boolean alias scope",
      {ocaml|module Hidden = struct open External let both = ( && ) end
let both = ( && )
let () = Verdict.p "all rows pass" (both true (List.for_all rows ~f:Fn.id))|ocaml},
      [ "all rows pass" ] );
    ( "does not retain an earlier Boolean operator binding across an unknown open",
      {ocaml|let (&&) = Stdlib.( && )
open External
let both = ( && )
let () = Verdict.p "all rows pass" (both true (List.for_all rows ~f:Fn.id))|ocaml},
      [] );
    ( "resolves a local Boolean operator defined after an unknown open",
      {ocaml|open External
let (&&) _ value = value
let both = ( && )
let () = Verdict.p "all rows pass" (both false (List.for_all rows ~f:Fn.id))|ocaml},
      [ "all rows pass" ] );
    ( "reopening standard operators shadows an earlier local Boolean operator",
      {ocaml|let (&&) _ _ = true
open Base
let both = ( && )
let () = Verdict.p "all rows pass" (both true (List.for_all rows ~f:Fn.id))|ocaml},
      [ "all rows pass" ] );
    ( "refuses a Boolean quantifier through a known standard module alias",
      {ocaml|module S = Stdlib
let both = S.( && )
let () = Verdict.p "all rows pass" (both true (List.for_all rows ~f:Fn.id))|ocaml},
      [ "all rows pass" ] );
    (* staging#697 round 2: direct parameter results inherit the actual Boolean identity, while
       constructors remain distinct; a Boolean literal establishes its comparison type. *)
    ( "refuses Boolean ordering after an identity helper forwards a quantifier",
      {ocaml|let id x = x
let () = Verdict.p "ok" (id (List.for_all rows ~f:Fn.id) >= (n = 0))|ocaml},
      [ "ok" ] );
    ( "accepts witnessed Boolean ordering after an identity helper",
      {ocaml|let id x = x
let () = Verdict.p "ok" (not (List.is_empty rows) && id (List.for_all rows ~f:Fn.id) >= (n = 0))|ocaml},
      [] );
    ( "refuses Boolean ordering after a deferred call returns a quantifier",
      {ocaml|let apply f xs = f xs
let all xs = List.for_all xs ~f:Fn.id
let () = Verdict.p "ok" (apply all rows >= (n = 0))|ocaml},
      [ "all" ] );
    ( "accepts witnessed Boolean ordering after a deferred call",
      {ocaml|let apply f xs = f xs
let all xs = List.for_all xs ~f:Fn.id
let () = Verdict.p "ok" (not (List.is_empty rows) && apply all rows >= (n = 0))|ocaml},
      [] );
    ( "keeps a helper constructor result separate from its Boolean argument",
      {ocaml|let wrap x = Some x
let () = Verdict.p "constructor ordering" (wrap (List.for_all rows ~f:Fn.id) > None)|ocaml},
      [] );
    ( "refuses a Boolean literal comparison inside a helper",
      {ocaml|let gt x = x > false
let () = Verdict.p "ok" (gt (List.for_all rows ~f:Fn.id))|ocaml},
      [ "ok" ] );
    ( "accepts a witnessed Boolean literal comparison inside a helper",
      {ocaml|let gt x = x > false
let () = Verdict.p "ok" (not (List.is_empty rows) && gt (List.for_all rows ~f:Fn.id))|ocaml},
      [] );
    (* staging#697 round 3: deferred ordering, namespace protection, and captured constants. *)
    ( "refuses ordering after both helper operands are substituted",
      {ocaml|let gt x y = x > y
let () = Verdict.p "ok" (gt (List.for_all rows ~f:Fn.id) false)|ocaml},
      [ "ok" ] );
    ( "accepts witnessed ordering after both helper operands are substituted",
      {ocaml|let gt x y = x > y
let () = Verdict.p "ok" (not (List.is_empty rows) && gt (List.for_all rows ~f:Fn.id) false)|ocaml},
      [] );
    ( "refuses ordering after a partially applied comparison helper",
      {ocaml|let gt x y = x > y
let above = gt (List.for_all rows ~f:Fn.id)
let () = Verdict.p "ok" (above false)|ocaml},
      [ "above" ] );
    ( "accepts witnessed ordering after a partially applied comparison helper",
      {ocaml|let gt x y = x > y
let above = gt (List.for_all rows ~f:Fn.id)
let () = Verdict.p "ok" (not (List.is_empty rows) && above false)|ocaml},
      [] );
    ( "keeps substituted aggregate ordering outside Boolean algebra",
      {ocaml|let gt x y = x > y
let () = Verdict.p "ok" (gt (true, List.for_all rows ~f:Fn.id) (false, true))|ocaml},
      [] );
    ( "refuses ordering in a nested helper capturing its outer operand",
      {ocaml|let make x = let gt y = x > y in gt
let () = Verdict.p "ok" (make (List.for_all rows ~f:Fn.id) false)|ocaml},
      [ "ok" ] );
    ( "accepts witnessed ordering in a nested helper capturing its outer operand",
      {ocaml|let make x = let gt y = x > y in gt
let () = Verdict.p "ok" (not (List.is_empty rows) && make (List.for_all rows ~f:Fn.id) false)|ocaml},
      [] );
    ( "refuses a claim inside a helper comparing two parameters",
      {ocaml|let check x y = Verdict.p "ok" (x > y)
let () = check (List.for_all rows ~f:Fn.id) false|ocaml},
      [ "check" ] );
    ( "does not apply Boolean ordering to a qualified external operator",
      {ocaml|let () = Verdict.p "ok" (External.( > ) (List.for_all rows ~f:Fn.id) false)|ocaml},
      [] );
    ( "does not apply Boolean ordering after an unknown open",
      {ocaml|open External
let () = Verdict.p "ok" (List.for_all rows ~f:Fn.id > false)|ocaml},
      [] );
    ( "does not apply Boolean ordering through an unknown module alias",
      {ocaml|module E = External
let () = Verdict.p "ok" (E.( > ) (List.for_all rows ~f:Fn.id) false)|ocaml},
      [] );
    ( "restores Boolean ordering after reopening standard operators",
      {ocaml|open External
open Base
let () = Verdict.p "ok" (List.for_all rows ~f:Fn.id > false)|ocaml},
      [ "ok" ] );
    ( "resolves Boolean ordering through a known standard module alias",
      {ocaml|module S = Stdlib
let () = Verdict.p "ok" (S.( > ) (List.for_all rows ~f:Fn.id) false)|ocaml},
      [ "ok" ] );
    ( "keeps unknown local opens out of the outer ordering scope",
      {ocaml|let _ = let open External in 0
let () = Verdict.p "ok" (List.for_all rows ~f:Fn.id > false)|ocaml},
      [ "ok" ] );
    ( "accepts a true annihilator captured through a disjunction helper",
      {ocaml|let with_left x = (||) x
let either = with_left true
let () = Verdict.p "ok" (either (List.for_all rows ~f:Fn.id))|ocaml},
      [] );
    ( "refuses a quantifier behind a false operand captured through a disjunction helper",
      {ocaml|let with_left x = (||) x
let either = with_left false
let () = Verdict.p "ok" (either (List.for_all rows ~f:Fn.id))|ocaml},
      [ "ok" ] );
    ( "accepts a false annihilator captured through a conjunction helper",
      {ocaml|let with_left x = (&&) x
let both = with_left false
let () = Verdict.p "ok" (both (List.for_all rows ~f:Fn.id))|ocaml},
      [] );
    ( "refuses a quantifier behind a true operand captured through a conjunction helper",
      {ocaml|let with_left x = (&&) x
let both = with_left true
let () = Verdict.p "ok" (both (List.for_all rows ~f:Fn.id))|ocaml},
      [ "ok" ] );
    ( "does not retain a local ordering operator across an unknown open",
      {ocaml|let ( > ) x _ = x
open External
let () = Verdict.p "ok" (List.for_all rows ~f:Fn.id > false)|ocaml},
      [] );
    ( "reopening standard ordering shadows an earlier local operator",
      {ocaml|let ( > ) _ _ = true
open Base
let () = Verdict.p "ok" (List.for_all rows ~f:Fn.id > false)|ocaml},
      [ "ok" ] );
    (* staging#697 round 4: optional defaults and direct callable parameter results. *)
    ( "refuses ordering through supplied optional Boolean parameters",
      {ocaml|let check ?(x=true) ?(y=false) () = Verdict.p "ok" (x > y)
let () = check ~x:(List.for_all rows ~f:Fn.id) ()|ocaml},
      [ "check" ] );
    ( "accepts witnessed ordering through optional Boolean parameters",
      {ocaml|let check ?(x=true) ?(y=false) () = Verdict.p "ok" (not (List.is_empty rows) && x > y)
let () = check ~x:(List.for_all rows ~f:Fn.id) ()|ocaml},
      [] );
    ( "accepts ordering through omitted constant Boolean defaults",
      {ocaml|let check ?(x=true) ?(y=false) () = Verdict.p "ok" (x > y)
let () = check ()|ocaml},
      [] );
    ( "accepts an omitted Boolean default annihilating ordering",
      {ocaml|let check ?(x=false) ?(y=false) () = Verdict.p "ok" (x <= y)
let () = check ~y:(List.for_all rows ~f:Fn.id) ()|ocaml},
      [] );
    ( "refuses ordering through an omitted quantified Boolean default",
      {ocaml|let check ?(x=List.for_all rows ~f:Fn.id) ?(y=false) () = Verdict.p "ok" (x > y)
let () = check ()|ocaml},
      [ "x" ] );
    ( "accepts an explicit replacement of a quantified ordering default",
      {ocaml|let check ?(x=List.for_all rows ~f:Fn.id) ?(y=false) () = Verdict.p "ok" (x > y)
let () = check ~x:true ()|ocaml},
      [] );
    ( "refuses ordering through a partially supplied optional helper",
      {ocaml|let check ?(x=true) ?(y=false) () = Verdict.p "ok" (x > y)
let partial = check ~x:(List.for_all rows ~f:Fn.id)
let () = partial ()|ocaml},
      [ "partial" ] );
    ( "accepts a captured outer default annihilating ordering",
      {ocaml|let make a = let check ?(x=a) ?(y=false) () = Verdict.p "ok" (x <= y) in check
let check = make false
let () = check ~y:(List.for_all rows ~f:Fn.id) ()|ocaml},
      [] );
    ( "refuses a Boolean alias returned through an identity helper",
      {ocaml|let id x = x
let both = id (&&)
let () = Verdict.p "ok" (both true (List.for_all rows ~f:Fn.id))|ocaml},
      [ "ok" ] );
    ( "accepts a witnessed Boolean alias returned through an identity helper",
      {ocaml|let id x = x
let both = id (&&)
let () = Verdict.p "ok" (not (List.is_empty rows) && both true (List.for_all rows ~f:Fn.id))|ocaml},
      [] );
    ( "accepts an annihilating Boolean alias returned through an identity helper",
      {ocaml|let id x = x
let either = id (||)
let () = Verdict.p "ok" (either true (List.for_all rows ~f:Fn.id))|ocaml},
      [] );
    ( "accepts a local operator returned through an identity helper",
      {ocaml|let id x = x
let (&&) _ _ = true
let both = id (&&)
let () = Verdict.p "ok" (both true (List.for_all rows ~f:Fn.id))|ocaml},
      [] );
    (* staging#697 round 5: replay ownership, operand type evidence, empty modules, predicates. *)
    ( "refuses a Boolean combinator forwarded through two helpers",
      {ocaml|let apply op x y = op x y
let forward op x y = apply op x y
let () = Verdict.p "ok" (forward (&&) true (List.for_all rows ~f:Fn.id))|ocaml},
      [ "ok" ] );
    ( "accepts a witnessed Boolean combinator forwarded through two helpers",
      {ocaml|let apply op x y = op x y
let forward op x y = apply op x y
let () = Verdict.p "ok" (not (List.is_empty rows) && forward (&&) true (List.for_all rows ~f:Fn.id))|ocaml},
      [] );
    ( "refuses an ordering call forwarded with two unresolved operands",
      {ocaml|let gt x y = x > y
let forward a b = gt a b
let () = Verdict.p "ok" (forward (List.for_all rows ~f:Fn.id) false)|ocaml},
      [ "ok" ] );
    ( "accepts witnessed ordering forwarded with two unresolved operands",
      {ocaml|let gt x y = x > y
let forward a b = gt a b
let () = Verdict.p "ok" (not (List.is_empty rows) && forward (List.for_all rows ~f:Fn.id) false)|ocaml},
      [] );
    ( "refuses a different population witness around forwarded ordering",
      {ocaml|let gt x y = x > y
let forward a b = gt a b
let () = Verdict.p "ok" (not (List.is_empty other) && forward (List.for_all rows ~f:Fn.id) false)|ocaml},
      [ "ok" ] );
    ( "accepts an annihilating Boolean combinator forwarded through two helpers",
      {ocaml|let apply op x y = op x y
let forward op x y = apply op x y
let () = Verdict.p "ok" (forward (||) true (List.for_all rows ~f:Fn.id))|ocaml},
      [] );
    ( "refuses a partially applied ordering with a branch-selected first operand",
      {ocaml|let choose c x y = if c then x else y
let gt x y = x > y
let above = gt (choose c (List.for_all rows ~f:Fn.id) false)
let () = Verdict.p "ok" (above false)|ocaml},
      [ "ok" ] );
    ( "accepts witnessed partial ordering with a branch-selected first operand",
      {ocaml|let choose c x y = if c then x else y
let gt x y = x > y
let above = gt (choose c (List.for_all rows ~f:Fn.id) false)
let () = Verdict.p "ok" (not (List.is_empty rows) && above false)|ocaml},
      [] );
    ( "refuses ordering of a branch-selected Boolean helper result",
      {ocaml|let choose c x y = if c then x else y
let () = Verdict.p "ok" (choose c (List.for_all rows ~f:Fn.id) false > (n = 0))|ocaml},
      [ "ok" ] );
    ( "accepts witnessed ordering of a branch-selected Boolean result",
      {ocaml|let choose c x y = if c then x else y
let () = Verdict.p "ok" (not (List.is_empty rows) && choose c (List.for_all rows ~f:Fn.id) false > (n = 0))|ocaml},
      [] );
    ( "refuses ordering when only the left operand establishes Boolean type",
      {ocaml|let choose c x y = if c then x else y
let () = Verdict.p "ok" ((n = 0) < choose c (List.for_all rows ~f:Fn.id) false)|ocaml},
      [ "ok" ] );
    ( "keeps branch-selected aggregate ordering outside Boolean algebra",
      {ocaml|let choose c x y = if c then x else y
let () = Verdict.p "ok" (choose c (true, List.for_all rows ~f:Fn.id) (true, true) > (false, true))|ocaml},
      [] );
    ( "does not reopen standard ordering through an empty local Base",
      {ocaml|open External
module Base = struct end
open Base
let () = Verdict.p "ok" (List.for_all rows ~f:Fn.id > false)|ocaml},
      [] );
    ( "does not reopen standard Boolean aliases through an empty local Stdlib",
      {ocaml|open External
module Stdlib = struct end
open Stdlib
let both = (&&)
let () = Verdict.p "ok" (both true (List.for_all rows ~f:Fn.id))|ocaml},
      [] );
    ( "retains implicit standard ordering after opening an empty local module",
      {ocaml|module Base = struct end
open Base
let () = Verdict.p "ok" (List.for_all rows ~f:Fn.id > false)|ocaml},
      [ "ok" ] );
    ( "restores a captured standard namespace after an empty local Base",
      {ocaml|module Real = Stdlib
open External
module Base = struct end
open Base
open Real
let () = Verdict.p "ok" (List.for_all rows ~f:Fn.id > false)|ocaml},
      [ "ok" ] );
    ( "refuses a vacuous operand captured by a Boolean conjunction predicate",
      {ocaml|let () = Verdict.p "ok" (List.exists groups ~f:((&&) (List.for_all rows ~f:Fn.id)))|ocaml},
      [ "ok" ] );
    ( "accepts a witnessed operand captured by a Boolean conjunction predicate",
      {ocaml|let () = Verdict.p "ok" (not (List.is_empty rows) && List.exists groups ~f:((&&) (List.for_all rows ~f:Fn.id)))|ocaml},
      [] );
    ( "refuses a vacuous operand captured by a Boolean disjunction predicate",
      {ocaml|let () = Verdict.p "ok" (List.exists groups ~f:((||) (List.for_all rows ~f:Fn.id)))|ocaml},
      [ "ok" ] );
    ( "accepts an annihilated operand in a Boolean predicate",
      {ocaml|let () = Verdict.p "ok" (List.exists groups ~f:((&&) (false && List.for_all rows ~f:Fn.id)))|ocaml},
      [] );
    ( "refuses a vacuous operand in a selected Boolean predicate",
      {ocaml|let pred = if flag then (&&) (List.for_all rows ~f:Fn.id) else (&&) true
let () = Verdict.p "ok" (List.exists groups ~f:pred)|ocaml},
      [ "pred" ] );
    ( "accepts a witnessed operand in a selected Boolean predicate",
      {ocaml|let pred = if flag then (&&) (List.for_all rows ~f:Fn.id) else (&&) true
let () = Verdict.p "ok" (not (List.is_empty rows) && List.exists groups ~f:pred)|ocaml},
      [] );
  ]

(* The syntax coverage matrix (gh-ocannl-931). The controls above each pin one shape a review round
   found; what they cannot show is which CROSS-PRODUCTS nobody wrote. This generates them: every
   value form the provenance layer models (a binding, a helper, a wrapper parameter, a match, a
   module, an open, a callback, an application), under every quantifier kind, under every way of
   spelling the quantifier's function, in the four cases each pairing wants -- the refusal (the
   quantifier reaching the claim in its vacuous polarity, which must be refused), the inverted
   spelling (accepted: the polarity is the wrong one for vacuity), the guarded spelling (accepted:
   the population is witnessed), and the shadowed spelling (accepted: a constant intercepts the
   value where the form binds a name, or an ignored sibling receives it). A missing combination is a
   cell nobody added, visible in the grid the golden prints, rather than a shape found by review.

   Each family is a template placing the quantified value [v] -- and [helper], the name the refusal
   must report. The four cases differ only in [v] and, for the shadowed case, the template's
   [shadow]; the expected verdict follows from the case, never from the family, which is what keeps
   the matrix data rather than a second list of hand-decided controls. *)
type matrix_family = {
  family : string;
  reports : string;  (** The name the refusal case must report. *)
  place : string -> string;
  shadow : string -> string;
}

let matrix_families =
  [
    {
      family = "a native claim's argument";
      reports = "the claim";
      place = (fun v -> Printf.sprintf "let () = Verdict.p \"the claim\" (%s)" v);
      shadow =
        (fun v ->
          Printf.sprintf "let () = Verdict.p \"the claim\" (let v = %s in let v = true in v)" v);
    };
    {
      family = "a structure-level binding";
      reports = "v";
      place = (fun v -> Printf.sprintf "let v = %s\nlet () = Verdict.p \"the claim\" v" v);
      shadow =
        (fun v -> Printf.sprintf "let v = %s\nlet v = true\nlet () = Verdict.p \"the claim\" v" v);
    };
    {
      family = "a binding local to the argument";
      reports = "v";
      place = (fun v -> Printf.sprintf "let () = Verdict.p \"the claim\" (let v = %s in v)" v);
      shadow =
        (fun v ->
          Printf.sprintf "let () = Verdict.p \"the claim\" (let v = %s in let v = true in v)" v);
    };
    {
      family = "a helper applied to the population";
      reports = "h";
      place =
        (fun v -> Printf.sprintf "let h rows = %s\nlet () = Verdict.p \"the claim\" (h rows)" v);
      shadow =
        (fun v ->
          Printf.sprintf
            "let h rows = %s\nlet h rows = true\nlet () = Verdict.p \"the claim\" (h rows)" v);
    };
    {
      family = "a function-case helper";
      reports = "h";
      place =
        (fun v ->
          Printf.sprintf "let h = function rows -> %s\nlet () = Verdict.p \"the claim\" (h rows)" v);
      shadow =
        (fun v ->
          Printf.sprintf
            "let h = function rows -> %s\n\
             let h = function rows -> true\n\
             let () = Verdict.p \"the claim\" (h rows)"
            v);
    };
    {
      family = "a wrapper's positional parameter";
      reports = "check";
      place =
        (fun v -> Printf.sprintf "let check ok = Verdict.p \"the claim\" ok\nlet () = check (%s)" v);
      shadow =
        (fun v ->
          Printf.sprintf
            "let check ok = let ok = true in Verdict.p \"the claim\" ok\nlet () = check (%s)" v);
    };
    {
      family = "a wrapper's labelled parameter";
      reports = "check";
      place =
        (fun v ->
          Printf.sprintf "let check ~ok = Verdict.p \"the claim\" ok\nlet () = check ~ok:(%s)" v);
      shadow =
        (fun v ->
          Printf.sprintf
            "let check ~ok = let ok = true in Verdict.p \"the claim\" ok\nlet () = check ~ok:(%s)" v);
    };
    {
      family = "a wrapper's optional default";
      reports = "ok";
      place =
        (fun v ->
          Printf.sprintf "let check ?(ok = %s) () = Verdict.p \"the claim\" ok\nlet () = check ()" v);
      shadow =
        (fun v ->
          Printf.sprintf
            "let check ?(ok = %s) () = Verdict.p \"the claim\" ok\nlet () = check ~ok:true ()" v);
    };
    {
      family = "a destructured wrapper parameter";
      reports = "check";
      place =
        (fun v ->
          Printf.sprintf "let check (_, ok) = Verdict.p \"the claim\" ok\nlet () = check (true, %s)"
            v);
      shadow =
        (fun v ->
          Printf.sprintf "let check (_, ok) = Verdict.p \"the claim\" ok\nlet () = check (%s, true)"
            v);
    };
    {
      family = "a match forwarding its scrutinee";
      reports = "v";
      place =
        (fun v ->
          Printf.sprintf "let v = match %s with ok -> ok\nlet () = Verdict.p \"the claim\" v" v);
      shadow =
        (fun v ->
          Printf.sprintf "let v = match %s with _ -> true\nlet () = Verdict.p \"the claim\" v" v);
    };
    {
      family = "a Boolean constructor match";
      reports = "v";
      place =
        (fun v ->
          Printf.sprintf
            "let v = match %s with true -> true | false -> false\n\
             let () = Verdict.p \"the claim\" v"
            v);
      shadow =
        (fun v ->
          Printf.sprintf
            "let v = match %s with true -> true | false -> true\nlet () = Verdict.p \"the claim\" v"
            v);
    };
    {
      family = "an if condition";
      reports = "v";
      place =
        (fun v ->
          Printf.sprintf "let v = if %s then true else false\nlet () = Verdict.p \"the claim\" v" v);
      shadow =
        (fun v ->
          Printf.sprintf "let v = if %s then true else true\nlet () = Verdict.p \"the claim\" v" v);
    };
    {
      family = "a match guard";
      reports = "v";
      place =
        (fun v ->
          Printf.sprintf
            "let v = match () with () when %s -> true | () -> false\n\
             let () = Verdict.p \"the claim\" v"
            v);
      shadow =
        (fun v ->
          Printf.sprintf
            "let v = match () with () when %s -> true | () -> true\n\
             let () = Verdict.p \"the claim\" v"
            v);
    };
    {
      family = "a protected try body";
      reports = "v";
      place =
        (fun v ->
          Printf.sprintf "let v = try %s with _ -> false\nlet () = Verdict.p \"the claim\" v" v);
      shadow =
        (fun v ->
          Printf.sprintf
            "let v = try %s with _ -> false\nlet v = true\nlet () = Verdict.p \"the claim\" v" v);
    };
    {
      family = "a member of a local module";
      reports = "M.v";
      place =
        (fun v ->
          Printf.sprintf "module M = struct let v = %s end\nlet () = Verdict.p \"the claim\" M.v" v);
      shadow =
        (fun v ->
          Printf.sprintf
            "module M = struct let v = %s let v = true end\nlet () = Verdict.p \"the claim\" M.v" v);
    };
    {
      family = "a member reached through open";
      reports = "v";
      place =
        (fun v ->
          Printf.sprintf
            "module M = struct let v = %s end\nopen M\nlet () = Verdict.p \"the claim\" v" v);
      shadow =
        (fun v ->
          Printf.sprintf
            "module M = struct let v = %s end\n\
             open M\n\
             let v = true\n\
             let () = Verdict.p \"the claim\" v"
            v);
    };
    {
      family = "a member reached through a local open";
      reports = "v";
      place =
        (fun v ->
          Printf.sprintf
            "module M = struct let v = %s end\nlet () = let open M in Verdict.p \"the claim\" v" v);
      shadow =
        (fun v ->
          Printf.sprintf
            "module M = struct let v = %s end\n\
             let () = let open M in let v = true in Verdict.p \"the claim\" v"
            v);
    };
    {
      family = "a claim inside a callback";
      reports = "check";
      place =
        (fun v ->
          Printf.sprintf
            "let check ok = List.iter [ () ] ~f:(fun () -> Verdict.p \"the claim\" ok)\n\
             let () = check (%s)"
            v);
      shadow =
        (fun v ->
          Printf.sprintf
            "let check ok = List.iter [ true ] ~f:(fun ok -> Verdict.p \"the claim\" ok)\n\
             let () = check (%s)"
            v);
    };
    {
      family = "an immediately invoked function";
      reports = "the claim";
      place = (fun v -> Printf.sprintf "let () = Verdict.p \"the claim\" ((fun () -> %s) ())" v);
      shadow =
        (fun v ->
          Printf.sprintf
            "let () = Verdict.p \"the claim\" ((fun () -> let v = %s in let v = true in v) ())" v);
    };
    {
      family = "a partially applied native claim";
      reports = "check";
      place = (fun v -> Printf.sprintf "let check = Verdict.p \"the claim\"\nlet () = check (%s)" v);
      shadow =
        (fun v ->
          Printf.sprintf
            "let check = Verdict.p \"the claim\"\nlet check b = ignore b\nlet () = check (%s)" v);
    };
    {
      family = "a pipeline into the claim";
      reports = "the claim";
      place = (fun v -> Printf.sprintf "let () = (%s) |> Verdict.p \"the claim\"" v);
      shadow =
        (fun v ->
          Printf.sprintf "let () = (let v = %s in let v = true in v) |> Verdict.p \"the claim\"" v);
    };
    {
      family = "a sequence's tail";
      reports = "v";
      place =
        (fun v -> Printf.sprintf "let v = (ignore rows; %s)\nlet () = Verdict.p \"the claim\" v" v);
      shadow =
        (fun v ->
          Printf.sprintf "let v = (ignore (%s); true)\nlet () = Verdict.p \"the claim\" v" v);
    };
    {
      family = "a tuple component destructured at binding";
      reports = "v";
      place =
        (fun v -> Printf.sprintf "let v, _ = (%s, true)\nlet () = Verdict.p \"the claim\" v" v);
      shadow =
        (fun v -> Printf.sprintf "let _, v = (%s, true)\nlet () = Verdict.p \"the claim\" v" v);
    };
    {
      family = "a field of a record binding";
      reports = "r";
      place =
        (fun v ->
          Printf.sprintf "let r = { ok = %s; detail = info }\nlet () = Verdict.p \"the claim\" r.ok"
            v);
      shadow =
        (fun v ->
          Printf.sprintf
            "let r = { ok = %s; detail = info }\n\
             let r = { ok = true; detail = info }\n\
             let () = Verdict.p \"the claim\" r.ok"
            v);
    };
    {
      family = "a comparison with true";
      reports = "v";
      place = (fun v -> Printf.sprintf "let v = (%s) = true\nlet () = Verdict.p \"the claim\" v" v);
      shadow =
        (fun v ->
          Printf.sprintf "let v = (%s) = true\nlet v = true\nlet () = Verdict.p \"the claim\" v" v);
    };
  ]

(* A quantifier kind, as a value over the free population [rows] (and [want]), given the text that
   names its function. *)
type matrix_quantifier = {
  quantifier : string;
  container : string;
  member : string;
  refusal : string -> string;
  inverted : string -> string;
  guarded : string -> string;
}

let matrix_quantifiers =
  [
    {
      quantifier = "for_all";
      container = "List";
      member = "for_all";
      refusal = (fun q -> q ^ " rows ~f:Fn.id");
      inverted = (fun q -> "not (" ^ q ^ " rows ~f:Fn.id)");
      guarded = (fun q -> "(not (List.is_empty rows)) && " ^ q ^ " rows ~f:Fn.id");
    };
    {
      quantifier = "for_all2_exn";
      container = "Array";
      member = "for_all2_exn";
      refusal = (fun q -> q ^ " rows want ~f:Float.equal");
      inverted = (fun q -> "not (" ^ q ^ " rows want ~f:Float.equal)");
      guarded = (fun q -> "(not (Array.is_empty rows)) && " ^ q ^ " rows want ~f:Float.equal");
    };
    {
      quantifier = "is_empty";
      container = "List";
      member = "is_empty";
      refusal = (fun q -> q ^ " rows");
      inverted = (fun q -> "not (" ^ q ^ " rows)");
      guarded = (fun q -> "List.length rows > 0 && " ^ q ^ " rows");
    };
    {
      quantifier = "exists";
      container = "List";
      member = "exists";
      refusal = (fun q -> "not (" ^ q ^ " rows ~f:Fn.id)");
      inverted = (fun q -> q ^ " rows ~f:Fn.id");
      guarded = (fun q -> "(not (List.is_empty rows)) && not (" ^ q ^ " rows ~f:Fn.id)");
    };
  ]

(* How the quantifier's function is spelled: qualified, through a structure-level open, through a
   module alias, or through a local open (gh-ocannl-908 item 6). *)
type matrix_spelling = {
  spelling : string;
  prelude : string -> string;
  call : string -> string -> string;
}

let matrix_spellings =
  [
    {
      spelling = "qualified";
      prelude = (fun _ -> "");
      call = (fun container member -> container ^ "." ^ member);
    };
    {
      spelling = "opened";
      prelude = (fun container -> "open " ^ container ^ "\n");
      call = (fun _ member -> member);
    };
    {
      spelling = "aliased";
      prelude = (fun container -> "module Q = " ^ container ^ "\n");
      call = (fun _ member -> "Q." ^ member);
    };
    {
      spelling = "local open";
      prelude = (fun _ -> "");
      call = (fun container member -> container ^ ".(" ^ member ^ ")");
    };
  ]

type matrix_case = Refusal | Inverted | Guarded | Shadowed

let matrix_cases = [ Refusal; Inverted; Guarded; Shadowed ]

let matrix_case_letter = function
  | Refusal -> 'R'
  | Inverted -> 'I'
  | Guarded -> 'G'
  | Shadowed -> 'S'

let matrix_source family quantifier spelling case =
  let q = spelling.call quantifier.container quantifier.member in
  let value, place =
    match case with
    | Refusal -> (quantifier.refusal q, family.place)
    | Inverted -> (quantifier.inverted q, family.place)
    | Guarded -> (quantifier.guarded q, family.place)
    | Shadowed -> (quantifier.refusal q, family.shadow)
  in
  spelling.prelude quantifier.container ^ place value

let matrix_expected family = function
  | Refusal -> [ family.reports ]
  | Inverted | Guarded | Shadowed -> []

(* One matrix cell's helpers, or the placeholder its failure reads as with the diagnostic naming it:
   a cell that does not parse is a fixture defect, and an exception from one that parses is the
   claim reader's (gh-ocannl-1292), told apart rather than both called a parse failure. [claims] is
   the reader, a parameter so the control below can plant the defect. *)
let matrix_cell_helpers ?(claims = quantified_claims) source =
  match
    Test_utils.Scan_attempt.run (fun () ->
        claims (Sources.structure_of source)
        |> List.map ~f:(fun claim -> claim.helper)
        |> List.dedup_and_sort ~compare:String.compare)
  with
  | Scanned helpers -> (helpers, None)
  | Unparsed where -> ([ "<does not parse>" ], Some ("does not parse: " ^ where))
  | Raised exn ->
      ( [ "<reader raised>" ],
        Some ("the claim reader raised " ^ exn ^ " on it, which parses -- a reader defect") )

let run_matrix_reader_control () =
  let planted, planted_why =
    matrix_cell_helpers ~claims:(fun _ -> invalid_arg "planted") "let x = 1"
  in
  let unparsed, unparsed_why = matrix_cell_helpers "let =" in
  let says why fragment =
    Option.value_map why ~default:false ~f:(String.is_substring ~substring:fragment)
  in
  ( "a syntax matrix cell whose reader raises reads as a reader defect, not a parse failure",
    List.equal String.equal planted [ "<reader raised>" ]
    && says planted_why "raised Invalid_argument(\"planted\")"
    && List.equal String.equal unparsed [ "<does not parse>" ]
    && says unparsed_why "does not parse: line 1" )

(* Runs the matrix: per family, the grid row and whether every cell read as expected. A cell that
   does not parse is a fixture defect and reads as unexpected too, named on stderr. *)
let run_syntax_matrix () =
  List.map matrix_families ~f:(fun family ->
      let cells =
        List.concat_map matrix_quantifiers ~f:(fun quantifier ->
            List.map matrix_spellings ~f:(fun spelling ->
                String.of_char_list
                  (List.map matrix_cases ~f:(fun case ->
                       let source = matrix_source family quantifier spelling case in
                       let found, failure = matrix_cell_helpers source in
                       Option.iter failure ~f:(fun why ->
                           eprintf "syntax matrix: %s / %s / %s %s\n%s\n" family.family
                             quantifier.quantifier spelling.spelling why source);
                       let expected = matrix_expected family case in
                       if List.equal String.equal found expected then matrix_case_letter case
                       else (
                         eprintf "syntax matrix: %s / %s / %s / %c expected [%s], found [%s]:\n%s\n"
                           family.family quantifier.quantifier spelling.spelling
                           (matrix_case_letter case)
                           (String.concat ~sep:", " expected)
                           (String.concat ~sep:", " found) source;
                         '!')))))
      in
      let ok = List.for_all cells ~f:(fun cell -> not (String.contains cell '!')) in
      (family.family, String.concat ~sep:" " cells, ok))

let matrix_claim_label family = family ^ ": every syntax matrix cell reads as expected"

let print_syntax_matrix rows =
  printf
    "\n\
     Syntax coverage matrix (gh-ocannl-931): each value form, under each quantifier and each\n\
     spelling of its function (%s), in the four cases\n\
     refusal / inverted / guarded / shadowed. A cell reads its case letter (R I G S) where the\n\
     verdict is as the case requires -- refused for R, accepted for the rest -- and `!` otherwise.\n\n"
    (String.concat ~sep:", " (List.map matrix_spellings ~f:(fun s -> s.spelling)));
  let width =
    List.fold matrix_families ~init:0 ~f:(fun acc f -> Int.max acc (String.length f.family))
  in
  let group = (4 * List.length matrix_spellings) + List.length matrix_spellings - 1 in
  printf "  %-*s  %s\n" width ""
    (String.concat ~sep:"  "
       (List.map matrix_quantifiers ~f:(fun q -> Printf.sprintf "%-*s" group q.quantifier))
    |> String.rstrip);
  List.iter rows ~f:(fun (family, cells, _) ->
      let groups =
        List.chunks_of (String.split cells ~on:' ') ~length:(List.length matrix_spellings)
        |> List.map ~f:(String.concat ~sep:" ")
      in
      printf "  %-*s  %s\n" width family (String.concat ~sep:"  " groups))

(* The shadowing fixtures, which the control list above cannot state: those cases compare the helper
   NAMES a source yields, and a name shadowed by a second definition of itself appears once in that
   comparison however many bodies carry it. What has to be pinned here is the opposite -- that one
   name comes back as two definitions -- so these are read for their definition SITES, and each
   collision is then handed to the refusal it must produce. Both claims of the same name are
   unguarded in each, so the reader would find every body on its own; the point is that a single
   exemption key would cover them all.

   The second fixture is the one a line number cannot serve. Its two `close` bindings are local to
   separate expressions written on ONE line -- the shape of a scanner test that spells a small
   helper inline in each of its cases -- so a definition identified by its line is one definition,
   and the exemption covers a body nobody read while this check reports green. *)
let shadowed_helper_fixture =
  {ocaml|let close got want = Array.for_all2_exn got want ~f:Float.equal
let () = Verdict.p "the first pair agrees" (close got want)
let close got want = Array.for_all2_exn got want ~f:Float.equal
let () = Verdict.p "the second pair agrees" (close got want)|ocaml}

let same_line_shadowed_helper_fixture =
  {ocaml|let () = (let close got want = Array.for_all2_exn got want ~f:Float.equal in Verdict.p "the first pair agrees" (close got want)); (let close got want = Array.for_all2_exn got want ~f:Float.equal in Verdict.p "the second pair agrees" (close got want))|ocaml}

let repeated_wrapper_call_fixture =
  {ocaml|let check value = Verdict.p "the collection property holds" value
let () = check (List.is_empty optional_rows)
let () = check (List.for_all rows ~f:Fn.id)|ocaml}

let multi_slot_wrapper_call_fixture =
  {ocaml|let check first second =
  Verdict.p "the optional rows are absent" first;
  Verdict.p "all rows pass" second
let () = check (List.is_empty optional_rows) (List.for_all rows ~f:Fn.id)|ocaml}

(* The definitions each exemption key resolves to, so that "one key, one helper" is something this
   check reads off the corpus rather than a property of names it hopes holds. Keyed by offset and
   carrying the printable site, so the report says where each body is and the identity does not
   depend on the report's precision.

   [record_definition] is one function rather than two spellings of an update because the corpus and
   the control below must agree on both halves of it -- the key, and what counts as a definition. A
   control that reproduced the aggregation instead of calling it would pass while the corpus stopped
   recording, and with one exempted helper in the tree nothing else would notice. *)
let quantified_exemption_key ~source claim = source ^ ":" ^ claim.helper

let scan_exemption_key ~source site =
  let identity =
    Scan.(match site.kind with Literal_label -> site.label | Computed_label -> site.head)
  in
  source ^ ":" ^ identity

let record_definition definitions ~key ~position ~description =
  Map.update definitions key ~f:(fun previous ->
      Map.set
        (Option.value previous ~default:(Map.empty (module Int)))
        ~key:position ~data:description)

let record_quantified_definition ~source definitions claim =
  record_definition definitions
    ~key:(quantified_exemption_key ~source claim)
    ~position:claim.helper_site.position ~description:(describe_site claim.helper_site)

let record_scan_definition ~source definitions site =
  record_definition definitions ~key:(scan_exemption_key ~source site) ~position:site.Scan.position
    ~description:(Printf.sprintf "%d:%d" site.Scan.line site.Scan.column)

let definition_sites ~source claims =
  List.fold claims ~init:(Map.empty (module String)) ~f:(record_quantified_definition ~source)

let colliding_exemptions definitions =
  Map.to_alist definitions
  |> List.filter_map ~f:(fun (key, sites) ->
      if Map.length sites > 1 then Some (key, Map.data sites) else None)

let run_quantified_helper_controls () =
  List.map quantified_helper_controls ~f:(fun (label, source, expected) ->
      let found =
        quantified_claims (Sources.structure_of source)
        |> List.map ~f:(fun claim -> claim.helper)
        |> List.dedup_and_sort ~compare:String.compare
      in
      let ok = List.equal String.equal found expected in
      if not ok then
        eprintf "quantified-helper control %S expected [%s], found [%s]\n" label
          (String.concat ~sep:", " expected)
          (String.concat ~sep:", " found);
      (label, ok))

(* The manifest's pin. [verdict_ratchet_controls.md] is prose, and prose drifts: a control renamed
   here and not there leaves a row nobody can find, and a row whose phrase names nothing leaves an
   inventory that reads complete. So the two are held equal from where the labels already are, in
   both directions -- every control label printed under "Synthetic helper-rule controls:", the
   run_*_control families included, appears in the manifest, and every phrase the manifest sets in
   backticks with a space in it (its convention for naming a control; a phrase starting with [dune]
   is a command) is such a label. The manifest is handed over by the rule's [(deps ...)], which is
   what makes a change to it re-run this. *)
let manifest_file = "verdict_ratchet_controls.md"

let manifest_control_phrases text =
  let rec collect acc from =
    match String.index_from text from '`' with
    | None -> acc
    | Some start -> (
        match String.index_from text (start + 1) '`' with
        | None -> acc
        | Some stop ->
            let span = String.sub text ~pos:(start + 1) ~len:(stop - start - 1) in
            let names_a_control =
              String.contains span ' '
              && (not (String.contains span '\n'))
              && not (String.is_prefix span ~prefix:"dune")
            in
            collect (if names_a_control then span :: acc else acc) (stop + 1))
  in
  List.rev (collect [] 0)

let manifest_row_label = "every synthetic control has a row in the mutation-run manifest"
let manifest_phrase_label = "every control phrase in the mutation-run manifest names a live control"
let manifest_distinct_label = "synthetic control labels are pairwise distinct"
let manifest_once_label = "every control phrase appears once in the mutation-run manifest"

(* [controls] is every control result printed under "Synthetic helper-rule controls:" before these
   two -- the quantified list AND the run_*_control families, since the manifest promises them all
   -- so a case added to any family without a row fails here, not only one added to the list. *)
let run_manifest_controls ~manifest ~controls =
  let labels =
    List.map controls ~f:fst
    @ [ manifest_row_label; manifest_phrase_label; manifest_distinct_label; manifest_once_label ]
  in
  (* Two controls under one label are one row here and one line in the golden: the second identity
     is gone before either inventory check runs, so the label set is held duplicate-free first. *)
  let duplicate = List.find_a_dup labels ~compare:String.compare in
  Option.iter duplicate ~f:(fun label ->
      eprintf "two synthetic controls share the label %S -- give each its own\n" label);
  let phrases =
    match manifest with
    | Some text -> manifest_control_phrases text
    | None ->
        eprintf "%s is not among the arguments -- the rule's deps no longer hand it over\n"
          manifest_file;
        []
  in
  (* And the manifest names each control once: a label repeated in a second row is a mapping that
     has become ambiguous, which the two set-based checks below would read as covered. *)
  let repeated = List.find_a_dup phrases ~compare:String.compare in
  Option.iter repeated ~f:(fun phrase ->
      eprintf "%s names the control %S in two places -- keep one row per control\n" manifest_file
        phrase);
  let label_set = Set.of_list (module String) labels in
  let phrase_set = Set.of_list (module String) phrases in
  List.iter labels ~f:(fun label ->
      if not (Set.mem phrase_set label) then
        eprintf "synthetic control without a row in %s: %S\n" manifest_file label);
  List.iter phrases ~f:(fun phrase ->
      if not (Set.mem label_set phrase) then
        eprintf "phrase in %s names no synthetic control: %S\n" manifest_file phrase);
  [
    (manifest_row_label, (not (List.is_empty labels)) && List.for_all labels ~f:(Set.mem phrase_set));
    ( manifest_phrase_label,
      (not (List.is_empty phrases)) && List.for_all phrases ~f:(Set.mem label_set) );
    (manifest_distinct_label, Option.is_none duplicate);
    (manifest_once_label, Option.is_none repeated);
  ]

(* The remedy a claim that is [Verdict]'s only past an [open] of another module adds
   (gh-ocannl-1361): that module may export the claim name, or a [Verdict] module of its own, so the
   call may be its. *)
let ambiguity_remedy ~ambiguous ~callee =
  if ambiguous then
    ". "
    ^ Option.value_map callee ~default:"Its claim" ~f:(fun callee -> "`" ^ callee ^ "`")
    ^ " reaches Verdict's only past an open of another module, which may export its own: if the \
       call is that module's, qualify it"
  else ""

let quantified_failure source claim =
  let key = quantified_exemption_key ~source claim in
  let quantifiers = List.map claim.quantifiers ~f:quantifier_name |> String.concat ~sep:", " in
  let qualify = ambiguity_remedy ~ambiguous:claim.ambiguous ~callee:None in
  if claim.direct then
    Printf.sprintf
      "%s:%d claims `%s` over a `%s` written directly in the argument at line %d, which can pass \
       on an empty population -- use the matching `Verdict.p_*` combinator (`p_all`, `p_none`, \
       `p_empty ~over`, `p_all2`), or make non-emptiness part of the claimed value. If emptiness \
       is the intended passing case, exempt `%s` by name in verdict_ratchet.ml and say why%s"
      source claim.claim_line claim.helper quantifiers claim.helper_site.line key qualify
  else
    Printf.sprintf
      "%s:%d sends `%s` from line %d into a Verdict claim, but that binding's `%s` can pass on an \
       empty population -- use the matching `Verdict.p_*` combinator, or make non-emptiness part \
       of the binding's passing result. If emptiness is the intended passing case, exempt `%s` by \
       name in verdict_ratchet.ml and say why%s"
      source claim.claim_line claim.helper claim.helper_site.line quantifiers key qualify

(* The two refusal diagnostics, each exercised by re-running this executable on a planted fixture in
   a child process: the fixture is the first control (a bound helper) and the native-claim control
   (a quantifier written directly in the argument), and the child must exit 1 printing the
   diagnostic that fixture earns. *)
let refusal_mode = "--quantified-helper-refusal-control"
let direct_refusal_mode = "--direct-quantifier-refusal-control"

let refusal_children =
  [
    ( refusal_mode,
      "the shipping ratchet process refuses the planted helper fixture",
      "refuses an unguarded for_all2_exn helper behind a local Verdict alias",
      [ "control_fixture.ml:3 sends `close`"; "can pass on an empty population" ] );
    ( direct_refusal_mode,
      "the shipping ratchet process refuses a quantifier written directly in a native claim",
      "refuses a quantifier written directly in a native claim",
      [ "control_fixture.ml:1 claims `all rows pass`"; "written directly in the argument" ] );
  ]

let refusal_fixture mode =
  List.find_map refusal_children ~f:(fun (found_mode, _, control, _) ->
      if String.equal found_mode mode then
        List.find_map quantified_helper_controls ~f:(fun (label, source, _) ->
            if String.equal label control then Some source else None)
      else None)

let run_refusal_control (mode, label, _, expected) =
  let exe = Stdlib.Sys.executable_name in
  let capture suffix = Stdlib.Filename.temp_file "verdict_ratchet_control" suffix in
  let out_path = capture ".out" and err_path = capture ".err" in
  let open_capture path = Unix.openfile path [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
  let out = open_capture out_path and err = open_capture err_path in
  let pid = Unix.create_process exe [| exe; mode |] Unix.stdin out err in
  let _, status = Unix.waitpid [] pid in
  Unix.close out;
  Unix.close err;
  let output = In_channel.read_all out_path ^ In_channel.read_all err_path in
  let unlink path = try Unix.unlink path with Unix.Unix_error _ -> () in
  unlink out_path;
  unlink err_path;
  let ok =
    (match status with Unix.WEXITED 1 -> true | _ -> false)
    && List.for_all expected ~f:(fun substring -> String.is_substring output ~substring)
  in
  if not ok then
    eprintf "the %s child did not reject its planted fixture as designed:\n%s\n" mode output;
  (label, ok)

let run_refusal_controls () = List.map refusal_children ~f:run_refusal_control

let refuse_stale_quantified ~fail stale_quantified =
  if not (Set.is_empty stale_quantified) then
    fail
      (Printf.sprintf
         "exempted quantified bindings that no Verdict claim reaches any more -- drop them from \
          the exemption list: %s"
         (String.concat ~sep:", " (Set.to_list stale_quantified)))

let refuse_colliding_quantified ~fail colliding =
  if not (List.is_empty colliding) then
    fail
      (Printf.sprintf
         "exempted quantified bindings whose key names more than one definition, so one granted \
          exemption is silently covering helpers nobody read -- give the shadowing definitions \
          separate names, or hoist them into one: %s"
         (String.concat ~sep:", "
            (List.map colliding ~f:(fun (key, sites) ->
                 Printf.sprintf "%s (definitions at %s)" key (String.concat ~sep:", " sites)))))

let run_shadowed_quantified_control ?(helper = "close") ?(site_kind = "definitions") label fixture =
  let colliding =
    quantified_claims (Sources.structure_of fixture)
    |> definition_sites ~source:"fixture"
    |> colliding_exemptions
  in
  let two_definitions =
    match colliding with
    | [ (key, [ first; second ]) ] ->
        String.equal key ("fixture:" ^ helper) && not (String.equal first second)
    | _ -> false
  in
  if not two_definitions then
    eprintf "the %s fixture resolved to %s, not to one name with two %s\n" label
      (String.concat ~sep:", "
         (List.map colliding ~f:(fun (key, sites) ->
              Printf.sprintf "%s (%s)" key (String.concat ~sep:", " sites))))
      site_kind;
  let source = "test/operations/verdict_ratchet.ml" in
  let format =
    "exempted quantified bindings whose key names more than one definition, so one granted \
     exemption is silently covering helpers nobody read -- give the shadowing definitions separate \
     names, or hoist them into one: %s"
  in
  let refused = ref false in
  let fail _message =
    refused := true;
    Test_utils.Refusal_control_manifest.observe_failure ~source ~format
  in
  refuse_colliding_quantified ~fail colliding;
  [
    (Printf.sprintf "a %s helper name resolves to two %s, not one" label site_kind, two_definitions);
    (Printf.sprintf "refuses an exemption key that names both %s %s" label site_kind, !refused);
  ]

let run_shadowed_quantified_controls () =
  run_shadowed_quantified_control "shadowed" shadowed_helper_fixture
  @ run_shadowed_quantified_control "same-line shadowed" same_line_shadowed_helper_fixture
  @ run_shadowed_quantified_control ~helper:"check" ~site_kind:"call sites" "reused wrapper call"
      repeated_wrapper_call_fixture
  @ run_shadowed_quantified_control ~helper:"check" ~site_kind:"call slots"
      "multi-slot wrapper call" multi_slot_wrapper_call_fixture

let run_stale_quantified_control () =
  let source = "test/operations/verdict_ratchet.ml" in
  let format =
    "exempted quantified bindings that no Verdict claim reaches any more -- drop them from the \
     exemption list: %s"
  in
  let refused = ref false in
  let fail _message =
    refused := true;
    Test_utils.Refusal_control_manifest.observe_failure ~source ~format
  in
  refuse_stale_quantified ~fail (Set.singleton (module String) "fixture:stale");
  ("refuses a stale quantified-helper exemption", !refused)

let repeated_literal_site_fixture =
  {ocaml|let () = Stdio.printf "repeated label: %b\n" first
let () = Stdio.printf "repeated label: %b\n" second|ocaml}

let same_line_repeated_literal_site_fixture =
  {ocaml|let () = Stdio.printf "repeated label: %b\n" first; Stdio.printf "repeated label: %b\n" second|ocaml}

let repeated_computed_site_fixture =
  {ocaml|let () = Stdio.printf "%s repeated row: %b\n" first_name first
let () = Stdio.printf "%s repeated row: %b\n" second_name second|ocaml}

let same_line_repeated_computed_site_fixture =
  {ocaml|let () = Stdio.printf "%s repeated row: %b\n" first_name first; Stdio.printf "%s repeated row: %b\n" second_name second|ocaml}

let refuse_colliding_sites ~fail colliding =
  if not (List.is_empty colliding) then
    fail
      (Printf.sprintf
         "exempted claim-shaped literal keys that name more than one source site, so one granted \
          exemption is silently covering prints nobody read -- give the sites distinct labels or \
          formats, or route them through one shared printer: %s"
         (String.concat ~sep:", "
            (List.map colliding ~f:(fun (key, sites) ->
                 Printf.sprintf "%s (sites at %s)" key (String.concat ~sep:", " sites)))))

let site_definitions ~source ~kind fixture =
  (Scan.scan fixture).Scan.sites
  |> List.filter ~f:(fun site -> Poly.equal site.Scan.kind kind)
  |> List.fold ~init:(Map.empty (module String)) ~f:(record_scan_definition ~source)

let run_colliding_site_control label kind expected_key fixture =
  let colliding = site_definitions ~source:"fixture" ~kind fixture |> colliding_exemptions in
  let two_sites =
    match colliding with
    | [ (key, [ first; second ]) ] ->
        String.equal key expected_key && not (String.equal first second)
    | _ -> false
  in
  if not two_sites then
    eprintf "the %s fixture resolved to %s, not to one key with two source sites\n" label
      (String.concat ~sep:", "
         (List.map colliding ~f:(fun (key, sites) ->
              Printf.sprintf "%s (%s)" key (String.concat ~sep:", " sites))));
  let source = "test/operations/verdict_ratchet.ml" in
  let format =
    "exempted claim-shaped literal keys that name more than one source site, so one granted \
     exemption is silently covering prints nobody read -- give the sites distinct labels or \
     formats, or route them through one shared printer: %s"
  in
  let refused = ref false in
  let fail _message =
    refused := true;
    Test_utils.Refusal_control_manifest.observe_failure ~source ~format
  in
  refuse_colliding_sites ~fail colliding;
  [
    (Printf.sprintf "a %s exemption key resolves to two source sites, not one" label, two_sites);
    (Printf.sprintf "refuses an exemption key that names both %s source sites" label, !refused);
  ]

let run_colliding_site_controls () =
  run_colliding_site_control "repeated literal-label" Scan.Literal_label "fixture:repeated label"
    repeated_literal_site_fixture
  @ run_colliding_site_control "same-line repeated literal-label" Scan.Literal_label
      "fixture:repeated label" same_line_repeated_literal_site_fixture
  @ run_colliding_site_control "repeated computed-label" Scan.Computed_label
      "fixture:%s repeated row: " repeated_computed_site_fixture
  @ run_colliding_site_control "same-line repeated computed-label" Scan.Computed_label
      "fixture:%s repeated row: " same_line_repeated_computed_site_fixture

(* {1 One dialect per gated claim (gh-ocannl-997)}

   [Verdict.skipped] prints [Verdict.p]'s line, [<claim>: true], and never [pass_fail]'s [<claim>:
   PASS]. A label reported through both breaks its golden on exactly the host that skips it,
   silently until the gate fires there. [Verdict.gated] is the remedy -- one call that picks the
   dialect itself -- and this is the ratchet that keeps the mixed spelling from regrowing. The
   reader is [Verdict_scan.dialect_census]; its header lists what it can and cannot see. The
   controls are the pairings it must find (their labels, in order) and the near misses it must not;
   the last pins the documented blind spot, so a reader that started pairing every computed label
   with every other would fail rather than grow noise. *)
let dialect_pairing_controls =
  [
    ( "refuses a pass_fail claim whose label binding a skip also reports",
      {ocaml|let claim = "the queued reading is per launch"
let () =
  if deep then Verdict.pass_fail claim ok
  else Verdict.skipped ~aggregation:`Environment ~backend:"cc" claim|ocaml},
      [ "the queued reading is per launch" ] );
    ( "refuses an opened pass_fail_all2 paired with a bound skip wrapper",
      {ocaml|open Verdict.Claims
let skipped = Verdict.skipped ~backend:backend_name
let () =
  if on_gpu then pass_fail_all2 "the readback matches" got want ~f:Float.equal
  else skipped "the readback matches"|ocaml},
      [ "the readback matches" ] );
    ( "resolves a label binding to the literal the other side spells out",
      {ocaml|let claim = "the leg holds"
let () =
  if gate then Verdict.pass_fail "the leg holds" ok else Verdict.skipped ~backend:"cc" claim|ocaml},
      [ "the leg holds" ] );
    ( "refuses a label computed by the same expression on both sides",
      {ocaml|let () =
  if gate then Verdict.pass_fail (leg ^ " holds") ok
  else Verdict.skipped ~backend:"cc" (leg ^ " holds")|ocaml},
      [ "leg ^ \" holds\"" ] );
    ( "accepts a gated claim reported through p on its evaluated side",
      {ocaml|let claim = "the leg holds"
let () = if gate then Verdict.p claim ok else Verdict.skipped ~backend:"cc" claim|ocaml},
      [] );
    ( "accepts pass_fail and skipped on different labels",
      {ocaml|let () = Verdict.pass_fail "the direction holds" ok
let () = Verdict.skipped ~backend:"cc" "the depth-gated leg holds"|ocaml},
      [] );
    ( "does not pair labels computed by different expressions, the blind spot gated closes",
      {ocaml|let () =
  if gate then Verdict.pass_fail (Printf.sprintf "%s holds" leg) ok
  else Verdict.skipped ~backend:"cc" (leg ^ " holds")|ocaml},
      [] );
  ]

let run_dialect_pairing_controls () =
  List.map dialect_pairing_controls ~f:(fun (label, source, expected) ->
      let found =
        (Scan.dialect_census source).Scan.pairings
        |> List.map ~f:(fun (pairing : Scan.dialect_pairing) ->
            Scan.label_key_text pairing.pass_fail.key)
      in
      let ok = List.equal String.equal found expected in
      if not ok then
        eprintf "dialect-pairing control %S expected [%s], found [%s]\n" label
          (String.concat ~sep:"; " expected)
          (String.concat ~sep:"; " found);
      (label, ok))

let refuse_dialect_pairing ~fail ~source (pairing : Scan.dialect_pairing) =
  fail
    (Printf.sprintf
       "%s:%d reports the claim `%s` through `%s`, and line %d skips the same claim through `%s` \
        -- a skip prints `<claim>: true`, which is Verdict.p's line and never PASS, so the golden \
        promoted where the claim is evaluated breaks on exactly the host that skips it. Report the \
        claim through `Verdict.gated ~when_ ~on`, which picks the dialect itself"
       source pairing.pass_fail.line
       (Scan.label_key_text pairing.pass_fail.key)
       pairing.pass_fail.callee pairing.skipped.line pairing.skipped.callee)

(* The wiring from a pairing to the failure, exercised on the first control's fixture: the reader's
   controls above say what it finds, this says that a finding refuses the run. *)
let run_dialect_refusal_control () =
  let source = "test/operations/verdict_ratchet.ml" in
  let format =
    "%s:%d reports the claim `%s` through `%s`, and line %d skips the same claim through `%s` -- a \
     skip prints `<claim>: true`, which is Verdict.p's line and never PASS, so the golden promoted \
     where the claim is evaluated breaks on exactly the host that skips it. Report the claim \
     through `Verdict.gated ~when_ ~on`, which picks the dialect itself"
  in
  let refused = ref false in
  let fail _message =
    refused := true;
    Test_utils.Refusal_control_manifest.observe_failure ~source ~format
  in
  let _, fixture, _ = List.hd_exn dialect_pairing_controls in
  List.iter (Scan.dialect_census fixture).Scan.pairings
    ~f:(refuse_dialect_pairing ~fail ~source:"fixture.ml");
  ("refuses a planted pass_fail and skipped pairing, naming the gated remedy", !refused)

(* {1 One row per claim: no line break in a literal label (post-wave task of staging#1134)}

   A label with a newline in it split its claim over two lines on each stream, and
   [tools/mutation-run.sh] -- which lists a run's [FAIL: …: false] lines and counts its stdout rows
   against the golden's -- read the halves as an extra row and an unlisted failure. [Verdict] now
   prints every claim line through [Verdict.one_line], which keeps a computed label on one row; a
   literal label written with a line break is refused here, where the newline can only be a mistake.
   The reader is [Verdict_scan.line_break_census]; its header lists what it cannot see. The controls
   are the labels it must find, in order, and the near misses it must not. *)
let line_break_controls =
  [
    ( "refuses a claim label literal that starts with a line break",
      {ocaml|let () = Verdict.pass_fail "\nthe streams bifurcate" ok|ocaml},
      [ "\nthe streams bifurcate" ] );
    ( "refuses a line break in the format an opened pf renders its label from",
      {ocaml|open Verdict.Claims
let () = pf "%s holds\n" leg ok|ocaml},
      [ "%s holds\n" ] );
    ( "resolves a label binding to the literal whose line break it carries",
      {ocaml|let label = "first\r\nsecond"
let () = Verdict.Claims.p label ok|ocaml},
      [ "first\r\nsecond" ] );
    ( "accepts a multi-line failure message, which fail prefixes line by line",
      {ocaml|let () = Verdict.fail "the first line\n  expected a\n  got b"|ocaml},
      [] );
    ( "does not read a claim name the source binds itself as Verdict's",
      {ocaml|let p = Stdio.printf
let () = p "rows: %d\n" n|ocaml},
      [] );
    ( "accepts a computed label, which Verdict prints on one line",
      {ocaml|let () = Verdict.p (leg ^ "\nholds") ok|ocaml},
      [] );
    ( "resolves a claim qualified through a module alias of Verdict",
      {ocaml|module V = Verdict
let () = V.p "first\nsecond" ok|ocaml},
      [ "first\nsecond" ] );
    ( "resolves a claim name a local open of Verdict.Claims brings in",
      {ocaml|let () = Verdict.Claims.(pass_fail "first\nsecond" ok)|ocaml},
      [ "first\nsecond" ] );
    ( "does not read a module that took over the name Verdict as Verdict",
      {ocaml|module Verdict = Local_printer
let () = Verdict.p "rows: %d\n" n|ocaml},
      [] );
    ( "does not read a path that only passes through a module named Verdict",
      {ocaml|let () = Other.Verdict.p "rows: %d\n" n|ocaml},
      [] );
    ( "does not read an unqualified claim name no open of Verdict brings in",
      {ocaml|open Printer
let () = p "rows: %d\n" n|ocaml},
      [] );
    ( "refuses a claim name past a later open as ambiguous, naming the qualified remedy",
      {ocaml|open Verdict.Claims
open Local_printer
let () = p "rows: %d\n" n|ocaml},
      [ "rows: %d\n [ambiguous]" ] );
    ( "reads a claim name a later let binds as the source's own, past an open of Verdict",
      {ocaml|open Verdict.Claims
let p = Stdio.printf
let () = p "rows: %d\n" n|ocaml},
      [] );
    ( "refuses a literal label under a type constraint",
      {ocaml|let () = Verdict.p ("first\nsecond" : string) ok|ocaml},
      [ "first\nsecond" ] );
    ( "reads a claim whose callee is under a type constraint",
      {ocaml|let () = (Verdict.p : string -> bool -> unit) "first\nsecond" ok|ocaml},
      [ "first\nsecond" ] );
    ( "resolves a label through a chain of let renamings",
      {ocaml|let raw = "first\nsecond"
let () =
  let label = raw in
  Verdict.p label ok|ocaml},
      [ "first\nsecond" ] );
    ( "resolves a label binding written with a type annotation",
      {ocaml|let label : string = ("first\nsecond" :> string)
let () = Verdict.p label ok|ocaml},
      [ "first\nsecond" ] );
  ]

let run_line_break_controls () =
  List.map line_break_controls ~f:(fun (label, source, expected) ->
      let found =
        (Scan.line_break_census source).Scan.broken
        |> List.map ~f:(fun ({ site; ambiguous } : Scan.line_break_site) ->
            Scan.label_key_text site.key ^ if ambiguous then " [ambiguous]" else "")
      in
      let ok = List.equal String.equal found expected in
      if not ok then
        eprintf "line-break control %S expected [%s], found [%s]\n" label
          (String.concat ~sep:"; " (List.map expected ~f:String.escaped))
          (String.concat ~sep:"; " (List.map found ~f:String.escaped));
      (label, ok))

let refuse_line_break ~fail ~source ({ site; ambiguous } : Scan.line_break_site) =
  (* The remedy an ambiguous site adds: the call may be another module's. *)
  let qualify = ambiguity_remedy ~ambiguous ~callee:(Some site.callee) in
  fail
    (Printf.sprintf
       "%s:%d reports the claim \"%s\" through `%s`, with a line break in its label -- a claim \
        prints as one row on each stream, and Verdict renders the break as a space. Put a blank \
        line before the claim in a printf of its own, and write the label on one line%s"
       source site.line
       (String.escaped (Scan.label_key_text site.key))
       site.callee qualify)

(* The wiring from a finding to the failure, exercised on the first control's fixture. *)
let run_line_break_refusal_control () =
  let source = "test/operations/verdict_ratchet.ml" in
  let format =
    "%s:%d reports the claim \"%s\" through `%s`, with a line break in its label -- a claim prints \
     as one row on each stream, and Verdict renders the break as a space. Put a blank line before \
     the claim in a printf of its own, and write the label on one line%s"
  in
  let refused = ref false in
  let fail _message =
    refused := true;
    Test_utils.Refusal_control_manifest.observe_failure ~source ~format
  in
  let _, fixture, _ = List.hd_exn line_break_controls in
  List.iter (Scan.line_break_census fixture).Scan.broken
    ~f:(refuse_line_break ~fail ~source:"fixture.ml");
  ("refuses a planted label literal carrying a line break, naming the one-line remedy", !refused)

(* {1 A gate never reads the population it quantifies (gh-ocannl-1289)}

   [Verdict.gated_all ~when_ ~on label xs ~f] skips where [when_] is false. A gate computed from
   [xs], or from the collection [xs] is mapped from, turns the regression the claim exists for into
   a skip -- the gh-ocannl-1115 trap, which shipped twice in one wave through these forms. The
   reader is [Verdict_scan.gate_census]; its header states what it follows and what it does not. The
   controls are the readings it must report (the population bindings each gate reads, or [unread
   <form>] for a claim it cannot complete) and the near misses it must accept: above all a gate that
   reads what the population is COMPUTED from -- the program, the limits, the backend that steers
   the population's [if] -- which is how a capability judgment reads. *)
let gate_population_controls =
  [
    ( "refuses a gate that tests its own population for emptiness",
      {ocaml|open Verdict.Claims
let () =
  let results = List.map seeds ~f:run in
  gated_all ~when_:(not (List.is_empty results)) ~on:backend_name
    "every GPU sketch matches the materialized run" results ~f:snd|ocaml},
      [ "results, run, seeds" ] );
    ( "refuses a gate read off the collection the population is mapped from",
      {ocaml|open Verdict.Claims
let () =
  let real = if on_gpu then gpu_seeds ~limits pre else [] in
  let results = List.mapi real ~f:(fun i q -> (q, run i q)) in
  gated_exists
    ~when_:(List.exists real ~f:(fun q -> q.Autotune.sk_mma))
    ~on:backend_name "a tensorized sketch matches the materialized run" results
    ~f:(fun (q, ok) -> q.Autotune.sk_mma && ok)|ocaml},
      [ "real" ] );
    ( "refuses a gate bound to a name that reads the population's source",
      {ocaml|let () =
  let real = if on_gpu then gpu_seeds ~limits pre else [] in
  let has_mma = List.exists real ~f:(fun q -> q.Autotune.sk_mma) in
  let results = List.filter_map real ~f:run in
  Verdict.gated_exists ~when_:has_mma ~on:backend_name "a tensorized sketch matches" results
    ~f:snd|ocaml},
      [ "real" ] );
    ( "refuses a gate read off a collection a match branch builds the population from",
      {ocaml|let () =
  let raw = probe () in
  let rows = match mode with `Fast -> List.filter raw ~f:Fn.id | `Slow -> [] in
  Verdict.gated_alli ~when_:(List.length raw > 2) ~on:"cc" "rows ascend" rows ~f:(fun i r -> r > i)|ocaml},
      [ "raw" ] );
    ( "refuses a gate read off a collection a protected try body builds the population from",
      {ocaml|let () =
  let raw = probe () in
  let rows = try List.map raw ~f:run with _ -> [] in
  Verdict.gated_all ~when_:(not (List.is_empty raw)) ~on:"cc" "all rows pass" rows ~f:Fn.id|ocaml},
      [ "raw" ] );
    ( "refuses a gate that reads a dereferenced population appended with another",
      {ocaml|let () =
  let staged = ref [] and other = ref [] in
  publish staged other;
  Verdict.gated_all ~when_:(not (List.is_empty !staged)) ~on:"cc" "every staged name fits"
    (!staged @ !other) ~f:fits|ocaml},
      [ "staged" ] );
    ( "refuses a population gate behind a partially applied gated wrapper",
      {ocaml|let () =
  let rows = probe () in
  let check = Verdict.gated_all ~when_:(not (List.is_empty rows)) ~on:"cc" in
  check "all rows pass" rows ~f:Fn.id|ocaml},
      [ "rows" ] );
    ( "refuses a population gate applied through a pipeline",
      {ocaml|open Verdict.Claims
let () =
  let rows = probe () in
  rows |> gated_all ~when_:(List.length rows > 0) ~on:"cc" "all rows pass" ~f:Fn.id|ocaml},
      [ "rows" ] );
    ( "refuses a population gate on a wrapper's own parameter",
      {ocaml|open Verdict.Claims
let check label xs = gated_all ~when_:(not (List.is_empty xs)) ~on:"cc" label xs ~f:Fn.id
let () = check "all rows pass" (probe ())|ocaml},
      [ "xs" ] );
    ( "accepts a capability-derived gate over a population built from the same program",
      {ocaml|open Verdict.Claims
let folded_seed_claims ~pre ~on_gpu ~run =
  let limits = Context.hardware_limits (Context.auto ()) in
  let site = Option.value_exn (Autotune.detect_matmul pre.LL.llc) in
  let capability =
    if on_gpu then Ll_test.tensorized_matmul_capability ~limits ~a:site.Autotune.m_a
    else `Withheld `Backend
  in
  let real = if on_gpu then gpu_seeds limits pre else [] in
  let results = List.mapi real ~f:(fun i q -> (q, run i q)) in
  gated_exists
    ~when_:(Poly.equal capability `Advertised)
    ~on:backend_name "a tensorized sketch matches the materialized run" results
    ~f:(fun (q, ok) -> q.Autotune.sk_mma && ok)|ocaml},
      [] );
    ( "accepts a backend gate that also steers the population's construction",
      {ocaml|open Verdict.Claims
let on_gpu = Sched.backend_is_gpu backend_name
let () =
  let real = if on_gpu then gpu_seeds pre else [] in
  let results = List.map real ~f:run in
  gated_all ~when_:on_gpu ~on:backend_name "every GPU sketch matches" results ~f:snd|ocaml},
      [] );
    ( "accepts a host gate behind a partially applied gated wrapper",
      {ocaml|let () =
  let fits = force long_names_fit_here in
  let long_claim_all = Verdict.gated_all ~aggregation:`Environment ~when_:fits ~on:"path budget" in
  let staged = ref [] in
  long_claim_all "every staging name fits one component" !staged ~f:short|ocaml},
      [] );
    ( "accepts a host gate behind a renamed partially applied wrapper",
      {ocaml|let () =
  let check = Verdict.gated_all ~when_:on_gpu ~on:"gpu" in
  let alias = check in
  alias "rows pass" rows ~f:Fn.id|ocaml},
      [] );
    ( "does not take a shadowed binding of the same name for the population",
      {ocaml|let () =
  let rows = probe () in
  let gate = not (List.is_empty rows) in
  let rows = List.filter all_rows ~f:Fn.id in
  Verdict.gated_all ~when_:gate ~on:"cc" "all rows pass" rows ~f:Fn.id|ocaml},
      [] );
    ( "names a gated claim it cannot complete instead of counting it clean",
      {ocaml|let () =
  List.iter2_exn labels populations ~f:(Verdict.gated_all ~when_:on_gpu ~on:"cc" ~f:Fn.id)|ocaml},
      [ "unread Verdict.gated_all" ] );
    ( "names a gated wrapper handed to another function beside a use it reads",
      {ocaml|let () =
  let raw = probe () in
  let check = Verdict.gated_all ~when_:(not (List.is_empty raw)) ~on:"cc" ~f:Fn.id in
  check "other rows pass" [ true ];
  List.iter2_exn [ "rows pass" ] [ raw ] ~f:check|ocaml},
      [ "unread check" ] );
    ( "names a bound partial claim whose only use goes through a module path",
      {ocaml|module Checks = struct
  let check = Verdict.gated_all ~when_:(not (List.is_empty rows)) ~on:"cc"
end
let () = Checks.check "all rows pass" rows ~f:Fn.id|ocaml},
      [ "unread Verdict.gated_all" ] );
    ( "refuses a gate on a population name an unmodelled open brings in",
      {ocaml|open Fixture
let () = Verdict.gated_all ~when_:(not (List.is_empty rows)) ~on:"cc" "all rows pass" rows ~f:Fn.id|ocaml},
      [ "rows [ambiguous]" ] );
    ( "accepts a backend gate on the key that selects the population",
      {ocaml|open Base
let () =
  let rows = Map.find_exn results backend_name in
  Verdict.gated_all ~when_:(String.equal backend_name "cuda") ~on:backend_name "all rows pass" rows
    ~f:Fn.id|ocaml},
      [] );
    ( "does not take a file's own gated_all for Verdict's",
      {ocaml|let gated_all ~when_ ~on label xs ~f = if when_ then List.iter xs ~f:(fun x -> f x label on)
let () = gated_all ~when_:(not (List.is_empty rows)) ~on:"cc" "all rows pass" rows ~f:check|ocaml},
      [] );
    ( "does not take another module's gated_all for Verdict's",
      {ocaml|module Verdict = Local_verdict
let () =
  Verdict.gated_all ~when_:(not (List.is_empty rows)) ~on:"cc" "all rows pass" rows ~f:Fn.id|ocaml},
      [] );
    ( "refuses a population gate through a module alias of Verdict",
      {ocaml|module V = Verdict
let () = V.gated_all ~when_:(not (List.is_empty rows)) ~on:"cc" "all rows pass" rows ~f:Fn.id|ocaml},
      [ "rows" ] );
    ( "accepts a backend gate on the key a Map.remove drops from the population",
      {ocaml|open Base
let () =
  let rows = Map.data (Map.remove results backend_name) in
  Verdict.gated_all ~when_:(String.equal backend_name "cuda") ~on:backend_name "all rows pass" rows
    ~f:Fn.id|ocaml},
      [] );
    ( "refuses a gate read off the default the population falls back to",
      {ocaml|open Base
let () =
  let rows = Option.value maybe_rows ~default:fallback in
  Verdict.gated_all ~when_:(not (List.is_empty fallback)) ~on:"cc" "all rows pass" rows ~f:Fn.id|ocaml},
      [ "fallback" ] );
    ( "refuses a gate read off a collection a local module expression builds the population from",
      {ocaml|open Base
let () =
  let rows =
    let module M = Impl in
    List.map raw ~f:M.run
  in
  Verdict.gated_all ~when_:(not (List.is_empty raw)) ~on:"cc" "all rows pass" rows ~f:Fn.id|ocaml},
      [ "raw" ] );
    ( "follows a collection module through its alias",
      {ocaml|open Base
module Rows = List
let () =
  let rows = Rows.map raw ~f:run in
  Verdict.gated_all ~when_:(not (Rows.is_empty raw)) ~on:"cc" "all rows pass" rows ~f:Fn.id|ocaml},
      [ "raw" ] );
    ( "does not take a shadowed collection module for the standard one",
      {ocaml|open Base
module Map = Custom_selectors
let () =
  let rows = Map.merge all_rows backend_name in
  Verdict.gated_all ~when_:(String.equal backend_name "cuda") ~on:backend_name "all rows pass" rows
    ~f:Fn.id|ocaml},
      [] );
    ( "follows the collection a Base function takes after a first-class module",
      {ocaml|open Base
let () =
  let rows = Map.data (Map.of_alist_exn (module String) raw) in
  Verdict.gated_all ~when_:(not (List.is_empty raw)) ~on:"cc" "all rows pass" rows ~f:snd|ocaml},
      [ "raw" ] );
    ( "refuses a gate read off the second list a population is appended from",
      {ocaml|open Base
let () =
  let rows = List.append first raw in
  Verdict.gated_all ~when_:(not (List.is_empty raw)) ~on:"cc" "all rows pass" rows ~f:Fn.id|ocaml},
      [ "raw" ] );
    ( "follows a Stdlib-ordered collection call through its last argument",
      {ocaml|let () =
  let rows = Stdlib.List.find (fun rows -> rows <> []) groups in
  Verdict.gated_all ~when_:(groups <> []) ~on:"cc" "all rows pass" rows ~f:Fun.id|ocaml},
      [ "groups" ] );
    ( "reads an unqualified collection call before a later open Base in Stdlib's order",
      {ocaml|let rows = List.find (fun rows -> rows <> []) groups
open Base
let () =
  Verdict.gated_all ~when_:(not (List.is_empty groups)) ~on:"cc" "all rows pass" rows ~f:Fn.id|ocaml},
      [ "groups" ] );
    ( "refuses a gate read off an element inserted into the population",
      {ocaml|open Base
let () =
  let rows = Set.to_list (Set.add existing sample) in
  Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" rows ~f:positive|ocaml},
      [ "sample" ] );
    ( "refuses a gate read off a value assigned into a dereferenced population",
      {ocaml|open Base
let () =
  let rows = ref [] in
  rows := List.map raw ~f:run;
  Verdict.gated_all ~when_:(not (List.is_empty raw)) ~on:"cc" "all rows pass" !rows ~f:Fn.id|ocaml},
      [ "raw" ] );
    ( "refuses a gate read off a value pushed into a mutable population",
      {ocaml|open Base
let () =
  let queue = Queue.create () in
  Queue.enqueue queue sample;
  Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" (Queue.to_list queue)
    ~f:positive|ocaml},
      [ "sample" ] );
    ( "refuses a gate read off a ref's initializer",
      {ocaml|open Base
let () =
  let rows = ref (List.map raw ~f:run) in
  Verdict.gated_all ~when_:(not (List.is_empty raw)) ~on:"cc" "all rows pass" !rows ~f:Fn.id|ocaml},
      [ "raw" ] );
    ( "refuses a gate read off a value a Stdlib writer adds before its container",
      {ocaml|let () =
  let queue = Queue.create () in
  Queue.add sample queue;
  Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" (List.of_seq (Queue.to_seq queue))
    ~f:positive|ocaml},
      [ "sample" ] );
    ( "follows List.Assoc under an open Base",
      {ocaml|open Base
let () =
  let rows = List.Assoc.find_exn entries ~equal:String.equal backend_name in
  Verdict.gated_all ~when_:(not (List.is_empty entries)) ~on:"cc" "all rows pass" rows ~f:Fn.id|ocaml},
      [ "entries" ] );
    ( "accepts a gate on an element a persistent Set.add leaves out of the original",
      {ocaml|open Base
let () =
  let existing = Set.of_list (module Int) raw in
  let bigger = Set.add existing sample in
  report bigger;
  Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" (Set.to_list existing)
    ~f:positive|ocaml},
      [] );
    ( "accepts a gate and a population destructured from one pattern",
      {ocaml|let check (enabled, rows) = Verdict.gated_all ~when_:enabled ~on:"cc" "rows pass" rows ~f:Fn.id
let () = check (on_gpu, probe ())|ocaml},
      [] );
    ( "refuses a gate read off the hash set a population is listed from",
      {ocaml|open Base
let () =
  let rows = Hash_set.to_list seen in
  Verdict.gated_all ~when_:(not (Hash_set.is_empty seen)) ~on:"cc" "all rows pass" rows ~f:Fn.id|ocaml},
      [ "seen" ] );
    ( "refuses a gate read off a value pushed through an alias of a mutable module",
      {ocaml|open Base
module Q = Queue
let () =
  let queue = Q.create () in
  Q.enqueue queue sample;
  Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" (Q.to_list queue) ~f:positive|ocaml},
      [ "sample" ] );
    ( "refuses a gate read off the value a constructor fills the population with",
      {ocaml|open Base
let () =
  let rows = Array.to_list (Array.create ~len:3 sample) in
  Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" rows ~f:positive|ocaml},
      [ "sample" ] );
    ( "refuses a gate read off the scrutinee a match case returns a part of",
      {ocaml|open Base
let () =
  let rows = match raw with [] -> [] | _ :: tail -> tail in
  Verdict.gated_all ~when_:(not (List.is_empty raw)) ~on:"cc" "all rows pass" rows ~f:Fn.id|ocaml},
      [ "raw" ] );
    ( "reads a collection call after open Stdlib in Stdlib's order",
      {ocaml|open Base
open Stdlib
let () =
  let rows = List.find (fun rows -> rows <> []) groups in
  Verdict.gated_all ~when_:(groups <> []) ~on:"cc" "all rows pass" rows ~f:Fun.id|ocaml},
      [ "groups" ] );
    ( "refuses a gate read off a thunk the population defaults to",
      {ocaml|open Base
let () =
  let rows = Hashtbl.find_or_add table key ~default:(fun () -> List.map raw ~f:run) in
  Verdict.gated_all ~when_:(not (List.is_empty raw)) ~on:"cc" "all rows pass" rows ~f:Fn.id|ocaml},
      [ "raw" ] );
    ( "refuses a gate a later write computes from the population",
      {ocaml|open Base
let () =
  let rows = probe () in
  let enabled = ref false in
  enabled := not (List.is_empty rows);
  Verdict.gated_all ~when_:!enabled ~on:"cc" "all rows pass" rows ~f:Fn.id|ocaml},
      [ "rows" ] );
    ( "refuses a gate read off what an initializer's callback fills the population with",
      {ocaml|open Base
let () =
  let rows = Array.to_list (Array.init 3 ~f:(fun _ -> sample)) in
  Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" rows ~f:positive|ocaml},
      [ "sample" ] );
    ( "refuses a gate read off a value written through an alias of the population",
      {ocaml|open Base
let () =
  let rows = ref [] in
  let alias = rows in
  alias := List.map raw ~f:run;
  Verdict.gated_all ~when_:(not (List.is_empty raw)) ~on:"cc" "all rows pass" !rows ~f:Fn.id|ocaml},
      [ "raw" ] );
    ( "refuses a gate read off what a binding operator builds the population from",
      {ocaml|open Base
let () =
  let maybe =
    let open Option.Let_syntax in
    let+ found = maybe_rows in
    found
  in
  Verdict.gated_all ~when_:(Option.is_some maybe_rows) ~on:"cc" "all rows pass"
    (Option.value maybe ~default:[]) ~f:Fn.id|ocaml},
      [ "maybe_rows" ] );
    ( "refuses a gate read off a positional Stdlib initializer callback",
      {ocaml|let () =
  let rows = Stdlib.List.init 3 (fun _ -> sample) in
  Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" rows ~f:positive|ocaml},
      [ "sample" ] );
    ( "refuses a gate read off a function-case callback",
      {ocaml|open Base
let () =
  let rows = Array.to_list (Array.init 3 ~f:(function _ -> sample)) in
  Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" rows ~f:positive|ocaml},
      [ "sample" ] );
    ( "refuses a gate read off the source a labeled blit copies into the population",
      {ocaml|open Base
let () =
  let dst = Array.create ~len:3 0 in
  Array.blit ~src ~src_pos:0 ~dst ~dst_pos:0 ~len:3;
  Verdict.gated_all ~when_:(Array.length src > 0) ~on:"cc" "all rows pass" (Array.to_list dst)
    ~f:positive|ocaml},
      [ "src" ] );
    ( "accepts a gate on the bounds of a range population",
      {ocaml|open Base
let () =
  let rows = List.range start stop in
  Verdict.gated_all ~when_:(stop > start) ~on:"cc" "all rows pass" rows ~f:positive|ocaml},
      [] );
    ( "refuses a gate read off a key inserted into the population",
      {ocaml|open Base
let () =
  let rows = Map.to_alist (Map.set table ~key:sample ~data:true) in
  Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" rows ~f:snd|ocaml},
      [ "sample" ] );
    ( "refuses a gate read off the source a transfer drains into the population",
      {ocaml|open Base
let () =
  let dst = Queue.create () in
  Queue.transfer ~src ~dst;
  Verdict.gated_all ~when_:(not (Queue.is_empty src)) ~on:"cc" "all rows pass" (Queue.to_list dst)
    ~f:positive|ocaml},
      [ "src" ] );
    ( "does not take a gated_all a local module brings in by open for Verdict's",
      {ocaml|module Checks = struct
  let gated_all ~when_ ~on label xs ~f = if when_ then List.iter xs ~f:(fun x -> f x label on)
end
open Checks
let () = gated_all ~when_:(not (List.is_empty rows)) ~on:"cc" "all rows pass" rows ~f:check|ocaml},
      [] );
    ( "follows a collection module beneath an alias of Base",
      {ocaml|module B = Base
let () =
  let rows = B.List.map raw ~f:run in
  Verdict.gated_all ~when_:(not (B.List.is_empty raw)) ~on:"cc" "all rows pass" rows ~f:Fn.id|ocaml},
      [ "raw" ] );
    ( "refuses a gate read off a default find_or_add inserts into the population",
      {ocaml|open Base
let () =
  let table = Hashtbl.create (module String) in
  ignore (Hashtbl.find_or_add table key ~default:(fun () -> sample) : int);
  Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" (Hashtbl.data table) ~f:positive|ocaml},
      [ "sample" ] );
    ( "refuses a gate read off a value written through a constrained alias",
      {ocaml|open Base
let () =
  let rows = ref [] in
  let alias = (rows : int list ref) in
  alias := List.map raw ~f:run;
  Verdict.gated_all ~when_:(not (List.is_empty raw)) ~on:"cc" "all rows pass" !rows ~f:positive|ocaml},
      [ "raw" ] );
    ( "refuses a gate read off what a map's callback returns",
      {ocaml|open Base
let () =
  let rows = List.map [ () ] ~f:(fun _ -> sample) in
  Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" rows ~f:positive|ocaml},
      [ "sample" ] );
    ( "refuses a gate read off what a Map.update callback puts in the population",
      {ocaml|open Base
let () =
  let rows = Map.data (Map.update table key ~f:(fun _ -> sample)) in
  Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" rows ~f:positive|ocaml},
      [ "sample" ] );
    ( "refuses a gate read off what a Hashtbl.update callback writes into the population",
      {ocaml|open Base
let () =
  let table = Hashtbl.create (module String) in
  Hashtbl.update table key ~f:(fun _ -> sample);
  Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" (Hashtbl.data table) ~f:positive|ocaml},
      [ "sample" ] );
    ( "refuses a gate a recursive peer reads the population through",
      {ocaml|open Base
let check rows =
  let rec enabled () = inspect ()
  and inspect () = not (List.is_empty rows) in
  Verdict.gated_all ~when_:(enabled ()) ~on:"cc" "all rows pass" rows ~f:Fn.id|ocaml},
      [ "rows" ] );
    ( "refuses a gate read off a value piped into a writer",
      {ocaml|open Base
let () =
  let queue = Queue.create () in
  sample |> Queue.enqueue queue;
  Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" (Queue.to_list queue) ~f:positive|ocaml},
      [ "sample" ] );
    ( "refuses a gate read off a value an unsafe setter writes",
      {ocaml|open Base
let () =
  let cells = Array.create ~len:3 0 in
  Array.unsafe_set cells 0 sample;
  Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" (Array.to_list cells) ~f:positive|ocaml},
      [ "sample" ] );
    ( "follows a population through Result",
      {ocaml|let () =
  let rows = Stdlib.Result.get_ok (Ok raw) in
  Verdict.gated_all ~when_:(raw <> []) ~on:"cc" "all rows pass" rows ~f:positive|ocaml},
      [ "raw" ] );
    ( "follows a collection call under a type annotation",
      {ocaml|let () =
  let rows = (Stdlib.List.map : (int -> int) -> int list -> int list) run raw in
  Verdict.gated_all ~when_:(raw <> []) ~on:"cc" "all rows pass" rows ~f:positive|ocaml},
      [ "raw" ] );
    (* gh-ocannl-1361: [Verdict.gated]'s boolean, whose population is what its quantifiers range
       over. *)
    ( "refuses a gated boolean whose gate tests the population its quantifier ranges over",
      {ocaml|open Base
let () =
  Verdict.gated ~when_:(not (List.is_empty xs)) ~on:"cc" "every x holds"
    ((not (List.is_empty xs)) && List.for_all xs ~f:Fn.id)|ocaml},
      [ "xs" ] );
    ( "refuses a gated boolean bound to a name before the claim",
      {ocaml|open Base
let () =
  let ok = List.for_all rows ~f:Fn.id in
  Verdict.gated ~when_:(List.length rows > 0) ~on:"cc" "all rows pass" ok|ocaml},
      [ "rows" ] );
    ( "refuses a gated boolean whose quantified population is mapped from what the gate reads",
      {ocaml|open Verdict.Claims
let () =
  let rows = List.map raw ~f:run in
  gated ~when_:(not (List.is_empty raw)) ~on:"cc" "no row fails" (not (List.exists rows ~f:failed))|ocaml},
      [ "raw" ] );
    ( "refuses a gated boolean whose Stdlib quantifier ranges over the gated population",
      {ocaml|let () =
  Verdict.gated ~when_:(rows <> []) ~on:"cc" "all rows pass" (Stdlib.List.for_all Fun.id rows)|ocaml},
      [ "rows" ] );
    ( "refuses a gated boolean written into a ref the claim dereferences",
      {ocaml|open Base
let () =
  let ok = ref false in
  ok := List.for_all rows ~f:Fn.id;
  Verdict.gated ~when_:(not (List.is_empty rows)) ~on:"cc" "all rows pass" !ok|ocaml},
      [ "rows" ] );
    ( "refuses a population gate behind a partially applied gated boolean wrapper",
      {ocaml|open Base
let () =
  let check = Verdict.gated ~when_:(not (List.is_empty rows)) ~on:"cc" in
  check "all rows pass" (List.for_all rows ~f:Fn.id)|ocaml},
      [ "rows" ] );
    ( "refuses a gated boolean whose pairwise existential is Base's exists2_exn",
      {ocaml|open Base
let () =
  Verdict.gated ~when_:(not (List.is_empty xs)) ~on:"cc" "some pair differs"
    (List.exists2_exn xs ys ~f:( <> ))|ocaml},
      [ "xs" ] );
    ( "refuses a gated boolean whose quantifier's predicate quantifies the gated population",
      {ocaml|open Base
let () =
  Verdict.gated ~when_:(not (List.is_empty ys)) ~on:"cc" "every x meets every y"
    (List.for_all xs ~f:(fun _ -> (not (List.is_empty ys)) && List.for_all ys ~f:pred))|ocaml},
      [ "ys" ] );
    ( "does not read a connective the source rebinds as the standard one",
      {ocaml|open Base
let not _ = true
let () =
  Verdict.gated ~when_:(List.length xs > 0) ~on:"cc" "the probe holds" (not (List.for_all xs ~f:Fn.id))|ocaml},
      [] );
    ( "accepts a gated boolean over a population a backend gate steers",
      {ocaml|open Base
let () =
  let rows = if on_gpu then run_gpu () else [] in
  Verdict.gated ~when_:on_gpu ~on:backend_name "every GPU row passes"
    ((not (List.is_empty rows)) && List.for_all rows ~f:Fn.id)|ocaml},
      [] );
    ( "accepts a gated boolean whose condition, not its value, quantifies what the gate reads",
      {ocaml|open Base
let () =
  Verdict.gated ~when_:(not (List.is_empty modes)) ~on:"cc" "the selected mode passes"
    (if List.exists modes ~f:is_fast then fast_ok else slow_ok)|ocaml},
      [] );
    ( "accepts a gated boolean comparing against the invariant its gate reads",
      {ocaml|open Base
let () =
  Verdict.gated ~when_:(String.equal backend_name "cuda") ~on:backend_name
    "the kernel ran on the selected backend" (String.equal ran_on backend_name)|ocaml},
      [] );
    ( "names a gated boolean claim it cannot complete instead of counting it clean",
      {ocaml|let () = List.iter2_exn labels oks ~f:(Verdict.gated ~when_:on_gpu ~on:"cc")|ocaml},
      [ "unread Verdict.gated" ] );
    ( "reads a gated claim nothing binds as Verdict's, its name being Verdict's alone",
      {ocaml|let () = gated_all ~when_:(not (List.is_empty rows)) ~on:"cc" "all rows pass" rows ~f:Fn.id|ocaml},
      [ "rows" ] );
    (* gh-ocannl-1303: the collection-module functions are a closed table; one it does not list is
       reported rather than read by a guess. *)
    ( "names a collection-module function the table does not list instead of guessing",
      {ocaml|open Base
let () =
  let rows = List.permute raw in
  Verdict.gated_all ~when_:on_gpu ~on:"cc" "all rows pass" rows ~f:Fn.id|ocaml},
      [ "unlisted List.permute" ] );
    ( "names an unlisted quantifier-position function of a gated boolean",
      {ocaml|open Base
let () =
  Verdict.gated ~when_:on_gpu ~on:"cc" "a sampled row passes" (List.random_element_exn rows)|ocaml},
      [ "unlisted List.random_element_exn" ] );
    ( "names an unlisted function handed a mutable container the population reads",
      {ocaml|open Base
let () =
  let cells = Array.of_list raw in
  Array.permute cells;
  Verdict.gated_all ~when_:on_gpu ~on:"cc" "all cells pass" (Array.to_list cells) ~f:Fn.id|ocaml},
      [ "unlisted Array.permute" ] );
    ( "names an unlisted function the gate reaches",
      {ocaml|open Base
let () =
  Verdict.gated_all ~when_:(not (List.is_empty (List.permute flags))) ~on:"cc" "all rows pass" rows
    ~f:Fn.id|ocaml},
      [ "unlisted List.permute" ] );
    ( "refuses a gate read off the separator interspersed into the population",
      {ocaml|open Base
let () =
  let rows = List.intersperse raw ~sep:sample in
  Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" rows ~f:positive|ocaml},
      [ "sample" ] );
    ( "refuses a gate that closes on the collection a gated membership claim ranges over",
      {ocaml|open Base
let () =
  Verdict.gated ~when_:(not (List.is_empty rows)) ~on:"cc" "the sample is present"
    (List.mem rows sample ~equal:Int.equal)|ocaml},
      [ "rows" ] );
    ( "refuses a gate that closes on the string a gated containment claim reads",
      {ocaml|open Base
let () =
  Verdict.gated ~when_:(not (String.is_empty rows)) ~on:"cc" "the sample is present"
    (String.contains rows sample)|ocaml},
      [ "rows" ] );
    ( "follows a collection module a local module exports through an open of it",
      {ocaml|open Base
module Collections = struct
  module Rows = List
end
open Collections
let () =
  let rows = Rows.map raw ~f:Fn.id in
  Verdict.gated_all ~when_:(not (List.is_empty raw)) ~on:"cc" "all rows pass" rows ~f:Fn.id|ocaml},
      [ "raw" ] );
    ( "follows a collection module a local module exports through a path into it",
      {ocaml|open Base
module Collections = struct
  module Rows = List
end
let () =
  let rows = Collections.Rows.map raw ~f:Fn.id in
  Verdict.gated_all ~when_:(not (List.is_empty raw)) ~on:"cc" "all rows pass" rows ~f:Fn.id|ocaml},
      [ "raw" ] );
    ( "refuses a gate that closes on the labelled collection a gated subset claim reads",
      {ocaml|open Base
let () =
  Verdict.gated ~when_:(not (Set.is_empty allowed)) ~on:"cc" "every requirement is allowed"
    (Set.is_subset required ~of_:allowed)|ocaml},
      [ "allowed" ] );
    ( "refuses a gate read off the key a map update inserts into the population",
      {ocaml|open Base
let () =
  let table = Map.update empty sample ~f:(fun _ -> 1) in
  Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all entries pass" (Map.to_alist table) ~f:positive|ocaml},
      [ "sample" ] );
    ( "refuses a gate read off a field a matched record update gives",
      {ocaml|open Base
let () =
  let probe = { defaults with found = raw } in
  let rows = match probe with { found; _ } -> found in
  Verdict.gated_all ~when_:(not (List.is_empty raw)) ~on:"cc" "all rows pass" rows ~f:positive|ocaml},
      [ "raw" ] );
    ( "refuses a gate read off the base of a matched record update for a field it leaves",
      {ocaml|open Base
let () =
  let probe = { defaults with found = raw } in
  let rows = match probe with { kept; _ } -> kept in
  Verdict.gated_all ~when_:(not (is_default defaults)) ~on:"cc" "all rows pass" rows ~f:positive|ocaml},
      [ "defaults" ] );
    ( "refuses a gate read off the value a merge's combining callback returns",
      {ocaml|open Base
let () =
  let merged = Map.merge_skewed left right ~combine:(fun ~key:_ _ _ -> sample) in
  Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all entries pass" (Map.to_alist merged) ~f:positive|ocaml},
      [ "sample" ] );
    ( "refuses a gate that closes on the string a gated validity claim reads",
      {ocaml|open Base
let () =
  Verdict.gated ~when_:(not (String.contains rows bad_byte)) ~on:"cc" "the text is UTF-8"
    (String.is_valid_utf_8 rows)|ocaml},
      [ "rows" ] );
    ( "refuses a gate read off the default of a callback's optional parameter",
      {ocaml|open Base
let () =
  let rows = Array.init 1 ~f:(fun ?(value = sample) _ -> value) in
  Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" (Array.to_list rows) ~f:positive|ocaml},
      [ "sample" ] );
    ( "refuses a gate read off the initializer of a qualified Stdlib reference",
      {ocaml|let () =
  let rows = Stdlib.ref [ sample ] in
  Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" (Stdlib.( ! ) rows) ~f:positive|ocaml},
      [ "sample" ] );
    ( "follows a collection function through a value alias",
      {ocaml|open Base
let () =
  let map = List.map in
  let rows = map raw ~f:run in
  Verdict.gated_all ~when_:(not (List.is_empty raw)) ~on:"cc" "all rows pass" rows ~f:Fn.id|ocaml},
      [ "raw" ] );
    ( "follows a collection function a local open of List brings in",
      {ocaml|open Base
let () =
  let rows = List.(map raw ~f:Fn.id) in
  Verdict.gated_all ~when_:(not (List.is_empty raw)) ~on:"cc" "all rows pass" rows ~f:Fn.id|ocaml},
      [ "raw" ] );
    ( "accepts a gate beside a helper bound inside an open of List under a listed name",
      {ocaml|open Base
let () =
  let rows =
    let open List in
    let map _ ~f:_ = fresh () in
    map raw ~f:Fn.id
  in
  Verdict.gated_all ~when_:(not (List.is_empty raw)) ~on:"cc" "all rows pass" rows ~f:Fn.id|ocaml},
      [] );
    ( "accepts a gate beside a helper an unknown open brings in after an open of List",
      {ocaml|open Base
let () =
  let rows =
    let open List in
    let open Custom in
    map raw ~f:Fn.id
  in
  Verdict.gated_all ~when_:(not (List.is_empty raw)) ~on:"cc" "all rows pass" rows ~f:Fn.id|ocaml},
      [] );
    ( "refuses a gate read off the increment a hash table counter is bumped by",
      {ocaml|open Base
let () =
  let table = Hashtbl.create (module String) in
  Hashtbl.incr table "rows" ~by:sample;
  Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all counts pass" (Hashtbl.data table) ~f:positive|ocaml},
      [ "sample" ] );
    ( "refuses a gate read off the key removed from the hash table the population reads",
      {ocaml|open Base
let () =
  let table = Hashtbl.create (module String) in
  Hashtbl.remove table sample;
  Verdict.gated_all ~when_:(String.is_empty sample) ~on:"cc" "all counts pass" (Hashtbl.data table)
    ~f:positive|ocaml},
      [ "sample" ] );
    ( "refuses a gate read off the value a mutable field of a matched record is set to",
      {ocaml|open Base
type probe = { mutable found : int list }
let () =
  let probe = { found = raw } in
  probe.found <- filtered;
  let rows = match probe with { found } -> found in
  Verdict.gated_all ~when_:(not (List.is_empty filtered)) ~on:"cc" "all rows pass" rows ~f:positive|ocaml},
      [ "filtered" ] );
    ( "accepts a gate read off a value written into another field of the matched record",
      {ocaml|open Base
type probe = { mutable found : int list; mutable metadata : int }
let () =
  let probe = { found = raw; metadata = 0 } in
  probe.metadata <- sample;
  let rows = match probe with { found; _ } -> found in
  Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" rows ~f:positive|ocaml},
      [] );
    ( "names a writer partially applied to the container the population reads",
      {ocaml|open Base
let () =
  let queue = Queue.create () in
  let push = Queue.enqueue queue in
  push sample;
  Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" (Queue.to_list queue) ~f:positive|ocaml},
      [ "unlisted Queue.enqueue (partially applied)" ] );
    ( "names a writer bound to a name after some of its arguments",
      {ocaml|open Base
let () =
  let rows = Array.create ~len:3 0 in
  let put = Array.set rows 0 in
  put sample;
  Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" (Array.to_list rows) ~f:positive|ocaml},
      [ "unlisted Array.set (partially applied)" ] );
    ( "follows a quantifier of a gated boolean through a value alias",
      {ocaml|open Base
let () =
  let all = List.for_all in
  Verdict.gated ~when_:(not (List.is_empty rows)) ~on:"cc" "all rows pass" (all rows ~f:Fn.id)|ocaml},
      [ "rows" ] );
    ( "refuses a gate read off a value pushed through a writer's value alias",
      {ocaml|open Base
let () =
  let queue = Queue.create () in
  let push = Queue.enqueue in
  push queue sample;
  Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" (Queue.to_list queue) ~f:positive|ocaml},
      [ "sample" ] );
    ( "follows a Verdict alias a local module exports through an open of it",
      {ocaml|module Checks = struct
  let gated_all = Verdict.gated_all
end
open Checks
let () = gated_all ~when_:(not (List.is_empty rows)) ~on:"cc" "all rows pass" rows ~f:Fn.id|ocaml},
      [ "rows" ] );
    ( "follows a Verdict alias a local module exports past a later open of a same-named value",
      {ocaml|module Local = struct
  let gated_all _ = ()
end
module Checks = struct
  let gated_all = Verdict.gated_all
  open Local
end
open Checks
let () = gated_all ~when_:(not (List.is_empty rows)) ~on:"cc" "all rows pass" rows ~f:Fn.id|ocaml},
      [ "rows" ] );
    ( "accepts a gate beside an alias a local module exports past its own recursive List",
      {ocaml|open Base
module M = struct
  module rec List : sig
    val map : 'a -> f:'b -> 'a
  end = struct
    let map x ~f:_ = x
  end
  let map = List.map
end
open M
let () =
  let rows = map raw ~f:run in
  Verdict.gated_all ~when_:(not (List.is_empty raw)) ~on:"cc" "all rows pass" rows ~f:Fn.id|ocaml},
      [] );
    ( "accepts a backend gate beside a population a match takes apart from a literal pair",
      {ocaml|open Base
let () =
  let pair = (on_gpu, raw) in
  let rows = match pair with _, xs -> xs in
  Verdict.gated_all ~when_:on_gpu ~on:"cc" "all rows pass" rows ~f:Fn.id|ocaml},
      [] );
    ( "refuses a gate read off the component a match takes the population from",
      {ocaml|open Base
let () =
  let rows = match (on_gpu, raw) with _, xs -> xs in
  Verdict.gated_all ~when_:(not (List.is_empty raw)) ~on:"cc" "all rows pass" rows ~f:Fn.id|ocaml},
      [ "raw" ] );
    ( "refuses a gate read off the literal record field a match takes the population from",
      {ocaml|open Base
type probe = { enabled : bool; found : bool list }
let () =
  let probe = { enabled = on_gpu; found = raw } in
  let rows = match probe with { found; _ } -> found in
  Verdict.gated_all ~when_:(not (List.is_empty raw)) ~on:"cc" "all rows pass" rows ~f:Fn.id|ocaml},
      [ "raw" ] );
    ( "accepts a backend gate beside a population a match takes from a constructor payload",
      {ocaml|open Base
let () =
  let rows = match Some (on_gpu, raw) with Some (_, xs) -> xs | None -> [] in
  Verdict.gated_all ~when_:on_gpu ~on:"cc" "all rows pass" rows ~f:Fn.id|ocaml},
      [] );
    ( "accepts a backend gate beside a gated boolean a match takes apart from a literal pair",
      {ocaml|open Base
let () =
  let ok = match (on_gpu, rows) with _, xs -> List.for_all xs ~f:Fn.id in
  Verdict.gated ~when_:on_gpu ~on:"cc" "all rows pass" ok|ocaml},
      [] );
    (* Only a write that may run before the claim shapes its population. *)
    ( "accepts a gate on a value written into the population only after the claim",
      {ocaml|open Base
let () =
  let queue = Queue.create () in
  Queue.enqueue queue first;
  Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" (Queue.to_list queue) ~f:positive;
  Queue.enqueue queue sample|ocaml},
      [] );
    ( "refuses a gate on a value written after the claim in the loop that repeats it",
      {ocaml|open Base
let () =
  let queue = Queue.create () in
  for _ = 1 to 3 do
    Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" (Queue.to_list queue)
      ~f:positive;
    Queue.enqueue queue sample
  done|ocaml},
      [ "sample" ] );
    ( "refuses a gate on a value written between two calls of the function making the claim",
      {ocaml|open Base
let () =
  let queue = Queue.create () in
  let check () =
    Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" (Queue.to_list queue) ~f:positive
  in
  check ();
  Queue.enqueue queue sample;
  check ()|ocaml},
      [ "sample" ] );
    ( "accepts a gate on a value written after the one call of the function making the claim",
      {ocaml|open Base
let () =
  let queue = Queue.create () in
  let check () =
    Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" (Queue.to_list queue) ~f:positive
  in
  check ();
  Queue.enqueue queue sample|ocaml},
      [] );
    ( "refuses a gate on a value written after a once-called claim inside the loop calling it",
      {ocaml|open Base
let () =
  let queue = Queue.create () in
  let check () =
    Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" (Queue.to_list queue) ~f:positive
  in
  while more () do
    check ();
    Queue.enqueue queue sample
  done|ocaml},
      [ "sample" ] );
    ( "accepts a gate on a value written after the iteration whose callback makes the claim",
      {ocaml|open Base
let () =
  let queue = Queue.create () in
  List.iter rows ~f:(fun row ->
      Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "the row passes" (row :: Queue.to_list queue)
        ~f:positive);
  Queue.enqueue queue sample|ocaml},
      [] );
    ( "accepts a gate on a value written after an aliased iteration whose module is later shadowed",
      {ocaml|open Base
let () =
  let queue = Queue.create () in
  let iter = List.iter in
  let module List = Custom in
  iter rows ~f:(fun row ->
      Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "the row passes" (row :: Queue.to_list queue)
        ~f:positive);
  Queue.enqueue queue sample|ocaml},
      [] );
    ( "accepts a gate on a value written after a claim in a for loop's bound",
      {ocaml|open Base
let () =
  let queue = Queue.create () in
  for _ = 1 to (Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" (Queue.to_list queue) ~f:positive; 3) do
    Queue.enqueue queue sample
  done|ocaml},
      [] );
    ( "accepts a gate on a value written after a claim in a binding operator's operand",
      {ocaml|open Base
let () =
  let queue = Queue.create () in
  let _ =
    let+ x = (Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" (Queue.to_list queue) ~f:positive; source) in
    x
  in
  Queue.enqueue queue sample|ocaml},
      [] );
    ( "refuses a gate on a value written before a partially applied iteration's callback runs",
      {ocaml|open Base
let () =
  let queue = Queue.create () in
  let check =
    List.iter ~f:(fun row ->
        Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "the row passes" (row :: Queue.to_list queue)
          ~f:positive)
  in
  Queue.enqueue queue sample;
  check rows|ocaml},
      [ "sample" ] );
    ( "accepts a gate on a value written after a piped iteration whose callback makes the claim",
      {ocaml|open Base
let () =
  let queue = Queue.create () in
  rows
  |> List.iter ~f:(fun row ->
      Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "the row passes" (row :: Queue.to_list queue)
        ~f:positive);
  Queue.enqueue queue sample|ocaml},
      [] );
    ( "refuses a gate on a value written in an argument evaluated in unspecified order beside the \
       claim",
      {ocaml|open Base
let () =
  let queue = Queue.create () in
  consume
    (Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" (Queue.to_list queue) ~f:positive;
     ())
    (Queue.enqueue queue sample)|ocaml},
      [ "sample" ] );
    ( "refuses a gate on a value written in a top-level binding beside the claim's",
      {ocaml|open Base
let queue = Queue.create ()
let _ = Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" (Queue.to_list queue) ~f:positive
and _ = Queue.enqueue queue sample|ocaml},
      [ "sample" ] );
    ( "accepts a gate on a value written in the right operand of a conjunction after the claim",
      {ocaml|open Base
let () =
  let queue = Queue.create () in
  if
    (Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" (Queue.to_list queue) ~f:positive;
     ready ())
    && (Queue.enqueue queue sample; true)
  then ()|ocaml},
      [] );
    ( "refuses a gate on a value written in an operand of a rebound conjunction",
      {ocaml|open Base
let () =
  let ( && ) a b = a || b in
  let queue = Queue.create () in
  if
    (Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" (Queue.to_list queue) ~f:positive;
     ready ())
    && (Queue.enqueue queue sample; true)
  then ()|ocaml},
      [ "sample" ] );
    ( "accepts a gate on a value a writer stores once its argument making the claim is evaluated",
      {ocaml|open Base
let () =
  let queue = Queue.create () in
  Queue.enqueue queue
    (Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" (Queue.to_list queue) ~f:positive;
     sample)|ocaml},
      [] );
    ( "refuses a gate on a value written by a writer whose callback makes the claim",
      {ocaml|open Base
let () =
  let table = Hashtbl.create (module String) in
  Hashtbl.update table "rows" ~f:(fun _ ->
      Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" (Hashtbl.data table)
        ~f:positive;
      sample)|ocaml},
      [ "sample" ] );
    ( "refuses a gate on a value written after a binding operator whose body makes the claim",
      {ocaml|open Base
let () =
  let queue = Queue.create () in
  let _ =
    let+ x = source in
    Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" (x :: Queue.to_list queue) ~f:positive
  in
  Queue.enqueue queue sample|ocaml},
      [ "sample" ] );
    ( "refuses a gate on a value written before a claim a collection stores is called",
      {ocaml|open Base
let () =
  let queue = Queue.create () in
  let checks =
    List.singleton (fun () ->
        Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" (Queue.to_list queue)
          ~f:positive)
  in
  Queue.enqueue queue sample;
  List.iter checks ~f:(fun check -> check ())|ocaml},
      [ "sample" ] );
    ( "refuses a gate on a value written before a claim a Stdlib collection stores is called",
      {ocaml|let () =
  let queue = Queue.create () in
  let checks =
    List.cons
      (fun () ->
        Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" (List.of_seq (Queue.to_seq queue))
          ~f:positive)
      []
  in
  Queue.add sample queue;
  List.iter (fun check -> check ()) checks|ocaml},
      [ "sample" ] );
    ( "refuses a gate on a value written before a lazy claim is forced",
      {ocaml|open Base
let () =
  let queue = Queue.create () in
  let verdict =
    lazy
      (Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" (Queue.to_list queue)
         ~f:positive)
  in
  Queue.enqueue queue sample;
  Lazy.force verdict|ocaml},
      [ "sample" ] );
    ( "refuses a gate on a value written after an expect test making the claim",
      {ocaml|open Base
let queue = Queue.create ()
let%expect_test "rows" =
  Verdict.gated_all ~when_:(sample > 0) ~on:"cc" "all rows pass" (Queue.to_list queue) ~f:positive
let () = Queue.enqueue queue sample|ocaml},
      [ "sample" ] );
    ( "refuses a gate read off the set an in-place filter keeps the population's keys by",
      {ocaml|open Base
let () =
  let table = Hashtbl.of_alist_exn (module String) raw in
  Hashtbl.filter_keys_inplace table ~f:(Set.mem keep);
  Verdict.gated_all ~when_:(not (Set.is_empty keep)) ~on:"cc" "all kept rows pass"
    (Hashtbl.data table) ~f:Fn.id|ocaml},
      [ "keep" ] );
  ]

let gate_control_findings source =
  let census = Scan.gate_census source in
  List.map census.Scan.readings ~f:(fun (reading : Scan.gate_reading) ->
      String.concat ~sep:", " reading.reads ^ if reading.ambiguous then " [ambiguous]" else "")
  @ List.map census.Scan.unread ~f:(fun (form, _) -> "unread " ^ form)
  @ List.map census.Scan.unlisted ~f:(fun (call : Scan.unlisted_call) -> "unlisted " ^ call.callee)

let run_gate_population_controls () =
  List.map gate_population_controls ~f:(fun (label, source, expected) ->
      let found = gate_control_findings source in
      let ok = List.equal String.equal found expected in
      if not ok then
        eprintf "gate-population control %S expected [%s], found [%s]\n" label
          (String.concat ~sep:"; " expected)
          (String.concat ~sep:"; " found);
      (label, ok))

let refuse_gate_reading ~fail ~source (reading : Scan.gate_reading) =
  fail
    (Printf.sprintf
       "%s:%d gates `%s` on `%s`, which reads `%s` -- the population the claim quantifies, or a \
        collection it is built from -- so a regression that empties or changes that population \
        closes the gate and turns the failure into a skip (gh-ocannl-1115). Gate on an invariant \
        the population does not decide: the backend, an advertised capability, the configuration%s"
       source reading.line reading.form reading.gate
       (String.concat ~sep:"`, `" reading.reads)
       (ambiguity_remedy ~ambiguous:reading.ambiguous ~callee:(Some reading.form)))

(* The gate reader over one source, its failures told apart (gh-ocannl-1289 review): a source that
   does not parse is reported by the catch-all the other readers share, and an exception from a
   source that parses is the gate reader's own defect, named as such rather than as a syntax
   error. *)
let read_gates ~fail ~source ~census content =
  match Test_utils.Scan_attempt.run (fun () -> census content) with
  | Test_utils.Scan_attempt.Scanned gates -> gates
  | Unparsed _ -> { Scan.readings = []; gated_sites = 0; unread = []; unlisted = [] }
  | Raised exn ->
      fail
        (Printf.sprintf
           "the gate reader raised %s on %s, which parses -- a defect of Verdict_scan.gate_census, \
            not of the source, and the source's gated claims went uncompared"
           exn source);
      { Scan.readings = []; gated_sites = 0; unread = []; unlisted = [] }

(* The four claim readers over one source, their failures told apart (gh-ocannl-1292): a source that
   does not parse is reported by name, and an exception from one that parses is a reader defect,
   named as such rather than as a syntax error. [readers] is a parameter so the control below can
   plant the defect. *)
let all_readers content =
  ( Scan.scan content,
    quantified_claims (Sources.structure_of content),
    Scan.dialect_census content,
    Scan.line_break_census content )

let read_claims ?(readers = all_readers) ~fail ~source content =
  let nothing () =
    ( { Scan.sites = []; literals = 0; applied_literals = 0 },
      [],
      { Scan.pairings = []; skipped_sites = 0; pass_fail_sites = 0 },
      { Scan.broken = []; label_sites = 0 } )
  in
  match Test_utils.Scan_attempt.run (fun () -> readers content) with
  | Scanned readings -> readings
  | Unparsed where ->
      fail
        (Printf.sprintf "%s does not parse as OCaml, so this check cannot vouch for it: %s" source
           where);
      nothing ()
  | Raised exn ->
      fail
        (Printf.sprintf
           "the claim readers raised %s on %s, which parses -- a defect in the scanner rather than \
            in the source, so the source's claims went unchecked"
           exn source);
      nothing ()

let run_claim_reader_refusal_control () =
  let source = "test/operations/verdict_ratchet.ml" in
  let format =
    "the claim readers raised %s on %s, which parses -- a defect in the scanner rather than in the \
     source, so the source's claims went unchecked"
  in
  let messages = ref [] in
  ignore
    (read_claims ~source:"fixture.ml"
       ~fail:(fun message -> messages := message :: !messages)
       ~readers:(fun _ -> invalid_arg "planted")
       "let () = ()"
      : Scan.scan * _ * Scan.dialect_census * Scan.line_break_census);
  let routed =
    match !messages with
    | [ m ] -> String.is_substring m ~substring:"raised Invalid_argument(\"planted\") on fixture.ml"
    | _ -> false
  in
  if routed then Test_utils.Refusal_control_manifest.observe_failure ~source ~format;
  ( "a claim reader's exception on a source that parses reaches the scanner-defect refusal, not \
     the parse one",
    routed )

let refuse_unread_gate ~fail ~source (form, line) =
  fail
    (Printf.sprintf
       "%s:%d applies `%s` in a form the gate reader cannot complete, so it cannot say whether the \
        gate reads the population. Apply the claim, or a `let`-bound partial application of it, to \
        its label and population directly"
       source line form)

let refuse_unlisted_gate ~fail ~source (call : Scan.unlisted_call) =
  fail
    (Printf.sprintf
       "%s:%d applies `%s` to a population or gate that reaches `%s` (line %d), a \
        collection-module function Verdict_scan.collection_functions does not list, so the gate \
        reader cannot say which of its arguments carry the population or whether it writes into \
        what it is handed. Add the function to that table with the role it plays"
       source call.claim_line call.claim call.callee call.callee_line)

(* The wiring from a finding to the failure, exercised on a refusing control of each kind: the
   controls above say what the reader finds, this says that a finding refuses the run. *)
let run_gate_refusal_control () =
  let source = "test/operations/verdict_ratchet.ml" in
  let reading_format =
    "%s:%d gates `%s` on `%s`, which reads `%s` -- the population the claim quantifies, or a \
     collection it is built from -- so a regression that empties or changes that population closes \
     the gate and turns the failure into a skip (gh-ocannl-1115). Gate on an invariant the \
     population does not decide: the backend, an advertised capability, the configuration%s"
  and unread_format =
    "%s:%d applies `%s` in a form the gate reader cannot complete, so it cannot say whether the \
     gate reads the population. Apply the claim, or a `let`-bound partial application of it, to \
     its label and population directly"
  and unlisted_format =
    "%s:%d applies `%s` to a population or gate that reaches `%s` (line %d), a collection-module \
     function Verdict_scan.collection_functions does not list, so the gate reader cannot say which \
     of its arguments carry the population or whether it writes into what it is handed. Add the \
     function to that table with the role it plays"
  in
  let fixture label =
    List.find_map_exn gate_population_controls ~f:(fun (found, source, _) ->
        if String.equal found label then Some source else None)
  in
  let refused_reading = ref false and refused_unread = ref false and refused_unlisted = ref false in
  let census =
    Scan.gate_census (fixture "refuses a gate that tests its own population for emptiness")
  in
  List.iter census.Scan.readings
    ~f:
      (refuse_gate_reading ~source:"fixture.ml" ~fail:(fun _ ->
           refused_reading := true;
           Test_utils.Refusal_control_manifest.observe_failure ~source ~format:reading_format));
  let census =
    Scan.gate_census (fixture "names a gated claim it cannot complete instead of counting it clean")
  in
  List.iter census.Scan.unread
    ~f:
      (refuse_unread_gate ~source:"fixture.ml" ~fail:(fun _ ->
           refused_unread := true;
           Test_utils.Refusal_control_manifest.observe_failure ~source ~format:unread_format));
  let census =
    Scan.gate_census
      (fixture "names a collection-module function the table does not list instead of guessing")
  in
  List.iter census.Scan.unlisted
    ~f:
      (refuse_unlisted_gate ~source:"fixture.ml" ~fail:(fun _ ->
           refused_unlisted := true;
           Test_utils.Refusal_control_manifest.observe_failure ~source ~format:unlisted_format));
  let raised_format =
    "the gate reader raised %s on %s, which parses -- a defect of Verdict_scan.gate_census, not of \
     the source, and the source's gated claims went uncompared"
  in
  let refused_raise = ref false in
  ignore
    (read_gates ~source:"fixture.ml"
       ~fail:(fun _ ->
         refused_raise := true;
         Test_utils.Refusal_control_manifest.observe_failure ~source ~format:raised_format)
       ~census:(fun _ -> raise Stdlib.Exit)
       "let () = ()"
      : Scan.gate_census);
  ( "refuses a planted gate that reads its population, and a gated claim it cannot read",
    !refused_reading && !refused_unread && !refused_unlisted && !refused_raise )

(* {1 A gate is a skip, not a disjunct of the claim (gh-ocannl-1310)}

   [p "…" ((not on_cpu) || ok)] prints [true] on a GPU: a pass, byte-identical to a CPU run that
   checked [ok], where the honest report is a skip. The reader is [Verdict_scan.folded_gate_census];
   its header states what a run-constant gate is and what it does not read. The controls are the
   gates it must report, each by the [~when_] the remedy would carry ([ [ambiguous] ] marking a
   claim past an open of another module), and the near misses it must accept: above all an
   implication whose antecedent the test computed, and a value computed from the program that read
   the backend on its way, which is how a census reads. *)
let folded_gate_prelude =
  {ocaml|open Verdict.Claims
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_cpu = Sched.backend_is_cpu backend_name
let on_gpu = Sched.backend_is_gpu backend_name
let is_metal = String.equal backend_name "metal"
|ocaml}

let folded_gate_controls =
  List.map
    ~f:(fun (label, body, expected) -> (label, folded_gate_prelude ^ body, expected))
    [
      ( "refuses a negated backend gate folded into a claim's value",
        {ocaml|let () = p "the census renders tiled" ((not on_cpu) || census_ok (Context.auto ()))|ocaml},
        [ "on_cpu" ] );
      ( "refuses a backend gate that is itself a disjunct of the claim",
        {ocaml|let () = p "the values agree" (on_gpu || Float.equal got want)|ocaml},
        [ "not (on_gpu)" ] );
      ( "refuses a gate written after the claim it folds",
        {ocaml|let () = p "the census renders tiled" (census_ok (Context.auto ()) || not on_cpu)|ocaml},
        [ "on_cpu" ] );
      ( "refuses a fold bound to a name before the claim",
        {ocaml|let () =
  let ok = (not on_cpu) || census_ok (Context.auto ()) in
  p "the census renders tiled" ok|ocaml},
        [ "on_cpu" ] );
      ( "refuses a fold spelled as an if whose else branch passes",
        {ocaml|let () = p "the census renders tiled" (if on_cpu then census_ok (Context.auto ()) else true)|ocaml},
        [ "on_cpu" ] );
      ( "refuses a fold spelled as an if whose then branch passes",
        {ocaml|let () = p "the values agree" (if on_gpu then true else census_ok (Context.auto ()))|ocaml},
        [ "not (on_gpu)" ] );
      ( "refuses a fold behind a negated conjunction",
        {ocaml|let () = p "the census renders tiled" (not (on_cpu && not (census_ok (Context.auto ()))))|ocaml},
        [ "on_cpu" ] );
      ( "refuses a fold in a quantified claim's predicate",
        {ocaml|let () = p_all "every row renders tiled" rows ~f:(fun row -> (not on_cpu) || row.tiled)|ocaml},
        [ "on_cpu" ] );
      ( "refuses a fold in a predicate a name is bound to",
        {ocaml|let () =
  let tiled row = (not on_cpu) || row.tiled in
  p_all "every row renders tiled" rows ~f:tiled|ocaml},
        [ "on_cpu" ] );
      ( "refuses a fold in p_none's predicate, whose element passes where it is false",
        {ocaml|let () = p_none "no row falls back" rows ~f:(fun row -> on_cpu && row.fell_back)|ocaml},
        [ "on_cpu" ] );
      ( "refuses a fold in a computed-label claim",
        {ocaml|let () = pf "%s renders tiled" tag ((not on_cpu) || census_ok (Context.auto ()))|ocaml},
        [ "on_cpu" ] );
      ( "refuses a fold in a gated boolean, beside its own gate",
        {ocaml|let () =
  gated ~when_:on_gpu ~on:backend_name "the dialect text is Metal's"
    ((not is_metal) || dialect_ok (Context.auto ()))|ocaml},
        [ "is_metal" ] );
      ( "refuses a gate read off a lazy hardware probe",
        {ocaml|let tile = lazy (Option.is_some (Context.hardware_limits (Context.auto ())).mma)
let () = p "the intrinsic renders" ((not (Lazy.force tile)) || census_ok (Context.auto ()))|ocaml},
        [ "Lazy.force tile" ] );
      ( "refuses a gate a match on the backend computes",
        {ocaml|let () =
  let gpu = match backend_name with "metal" | "cuda" | "hip" -> true | _ -> false in
  p "the census renders the intrinsic" ((not gpu) || census_ok (Context.auto ()))|ocaml},
        [ "gpu" ] );
      ( "refuses a host gate folded into a claim's value",
        {ocaml|let () = p "the path is portable" (Sys.win32 || portable (Context.auto ()))|ocaml},
        [ "not (Sys.win32)" ] );
      ( "refuses a gate read off the settings record",
        {ocaml|let () =
  p "the debug source is kept"
    (Utils.settings.output_debug_files_in_build_directory || kept (Context.auto ()))|ocaml},
        [ "not (Utils.settings.output_debug_files_in_build_directory)" ] );
      ( "accepts a disjunct that is a library constant, which the test did not compute",
        {ocaml|let () = p "a library is kept where requested" (on_cpu || Kept.always)|ocaml},
        [] );
      ( "refuses a fold read off any value of Sys",
        {ocaml|let () = p "the layout is native" (Sys.big_endian || native_layout (Context.auto ()))|ocaml},
        [ "not (Sys.big_endian)" ] );
      ( "refuses a gate a later assignment writes into a ref",
        {ocaml|let gate = ref false
let () = gate := Sys.win32
let () = p "the path is portable" (!gate || portable (Context.auto ()))|ocaml},
        [ "not (!gate)" ] );
      ( "refuses a ref that ever holds a run-constant, whatever overwrites it",
        {ocaml|let () =
  let ready = ref on_cpu in
  ready := probe (Context.auto ());
  p "the probe passes or the census renders" (!ready || census_ok (Context.auto ()))|ocaml},
        [ "not (!ready)" ] );
      ( "refuses a gate assigned over a ref the test initialized",
        {ocaml|let () =
  let gate = ref (probe (Context.auto ())) in
  gate := Sys.win32;
  p "the path is portable" (!gate || portable (Context.auto ()))|ocaml},
        [ "not (!gate)" ] );
      ( "refuses a gate written through an alias of the ref",
        {ocaml|let () =
  let gate = ref false in
  let alias = gate in
  alias := Sys.win32;
  p "the path is portable" (!gate || portable (Context.auto ()))|ocaml},
        [ "not (!gate)" ] );
      ( "refuses a fold in a function-case predicate's guard",
        {ocaml|let () = p_all "every row renders tiled" rows ~f:(function row when on_cpu -> row.tiled | _ -> true)|ocaml},
        [ "not (match element with | row when on_cpu -> false | _ -> true)" ] );
      ( "refuses a fold in a match guard whose patterns match anything",
        {ocaml|let () =
  let census = compile_census (Context.auto ()) in
  p "the census renders tiled" (match census with c when on_cpu -> renders c | _ -> true)|ocaml},
        [ "not (match census with | c when on_cpu -> false | _ -> true)" ] );
      ( "accepts a function-case predicate that selects on the element",
        {ocaml|let () = p_all "every present row matches" rows ~f:(function Some row -> row.ok | None -> true)|ocaml},
        [] );
      ( "refuses a gate a module of the source binds",
        {ocaml|module Gates = struct
  let cpu = Sched.backend_is_cpu backend_name
end

let () = p "the census renders tiled" ((not Gates.cpu) || census_ok (Context.auto ()))|ocaml},
        [ "Gates.cpu" ] );
      ( "accepts a value of a module the source names Sys",
        {ocaml|module Sys = struct
  let win32 = false
end

let () = p "the path is portable" (Sys.win32 || portable (Context.auto ()))|ocaml},
        [] );
      ( "refuses a gate a try computes from the configuration",
        {ocaml|let cc_requested = try String.equal (Sys.getenv "BACKEND") "cc" with Not_found -> false
let () = p "the census renders tiled" ((not cc_requested) || census_ok (Context.auto ()))|ocaml},
        [ "cc_requested" ] );
      ( "refuses a gate read off a field of a record that also holds the test's data",
        {ocaml|let () =
  let state = { gate = on_cpu; observation = probe (Context.auto ()) } in
  p "the census renders tiled" ((not state.gate) || census_ok (Context.auto ()))|ocaml},
        [ "state.gate" ] );
      ( "accepts a field of a record that holds the test's data",
        {ocaml|let () =
  let state = { gate = on_cpu; observation = probe (Context.auto ()) } in
  p "the probe passes or the census renders" (state.observation || census_ok (Context.auto ()))|ocaml},
        [] );
      ( "refuses a fold inside a protected expression",
        {ocaml|let () =
  p "the census renders tiled"
    (try (not on_cpu) || census_ok (Context.auto ()) with _ -> fallback (Context.auto ()))|ocaml},
        [ "on_cpu" ] );
      ( "refuses a gate a module of the source writes into its cell",
        {ocaml|module Gates = struct
  let gate = ref false
  let () = gate := Sys.win32
end

let () = p "the path is portable" (!Gates.gate || portable (Context.auto ()))|ocaml},
        [ "not (!Gates.gate)" ] );
      ( "refuses a gate read off the backend's codegen capabilities",
        {ocaml|let supports_f64 = (Context.codegen_capabilities (Context.auto ())).supports_f64
let () = p "doubles round-trip" ((not supports_f64) || round_trips (Context.auto ()))|ocaml},
        [ "supports_f64" ] );
      ( "accepts a ref the test alone writes",
        {ocaml|let () =
  let ready = ref false in
  ready := probe (Context.auto ());
  p "the probe passes or the census renders" (!ready || census_ok (Context.auto ()))|ocaml},
        [] );
      ( "refuses a fold spelled as a match on the backend with a passing case",
        {ocaml|let () =
  p "the census renders tiled"
    (match backend_name with "cc" -> census_ok (Context.auto ()) | _ -> true)|ocaml},
        [ "not (match backend_name with | \"cc\" -> false | _ -> true)" ] );
      ( "accepts a match on the backend every case of which the test computes",
        {ocaml|let () =
  p "the census renders"
    (match backend_name with "cc" -> tiled (Context.auto ()) | _ -> intrinsic (Context.auto ()))|ocaml},
        [] );
      ( "accepts a gated claim whose value repeats its own gate",
        {ocaml|let () =
  gated ~when_:on_cpu ~on:backend_name "the census renders tiled"
    ((not on_cpu) || census_ok (Context.auto ()))|ocaml},
        [] );
      ( "accepts a gated_all whose predicate repeats a conjunct of its gate",
        {ocaml|let () =
  gated_all ~when_:(on_gpu && is_metal) ~on:backend_name "every row renders the intrinsic" rows
    ~f:(fun row -> (not is_metal) || row.intrinsic)|ocaml},
        [] );
      ( "marks a fold past an open of another module ambiguous",
        {ocaml|open Other
let () = p "the census renders tiled" ((not on_cpu) || census_ok (Context.auto ()))|ocaml},
        [ "on_cpu [ambiguous]" ] );
      ( "accepts the gated spelling of the same claim",
        {ocaml|let () = gated ~when_:on_cpu ~on:backend_name "the census renders tiled" (census_ok (Context.auto ()))|ocaml},
        [] );
      ( "accepts a skip spelled as a branch",
        {ocaml|let () =
  if on_cpu then p "the census renders tiled" (census_ok (Context.auto ()))
  else skipped "the census renders tiled"|ocaml},
        [] );
      ( "accepts an implication whose antecedent the test computed",
        {ocaml|let () =
  p_alli "losses on good steps are finite" stepped ~f:(fun i ran ->
      (not ran) || Float.is_finite losses.(i))|ocaml},
        [] );
      ( "accepts a disjunct computed from the program though it read the backend on its way",
        {ocaml|let () =
  let census = compile_census (Context.auto ()) ~tensorize:on_cpu in
  p "the census renders tiled or declines" (renders census || declines (Context.auto ()))|ocaml},
        [] );
      ( "refuses a backend gate conjoined into an implication's antecedent",
        {ocaml|let () =
  p_all "every staged GPU row matches" rows ~f:(fun row -> (not (on_gpu && row.staged)) || row.ok)|ocaml},
        [ "on_gpu" ] );
      ( "accepts an antecedent that compares the element under test with the backend",
        {ocaml|let () =
  p_all "every row this backend recorded matches" rows ~f:(fun row ->
      (not (String.equal row.backend backend_name)) || row.ok)|ocaml},
        [] );
      ( "does not read a claim name the source binds itself",
        {ocaml|let () =
  let p label ok = Stdio.printf "%s\n" (if ok then label else "") in
  p "the census renders tiled" ((not on_cpu) || census_ok (Context.auto ()))|ocaml},
        [] );
      ( "accepts a gate folded below a conjunction, which the claim checks in part everywhere",
        {ocaml|let () =
  p "the census agrees and the dialect text is Metal's"
    (census_ok (Context.auto ()) && ((not is_metal) || dialect_ok (Context.auto ())))|ocaml},
        [] );
      ( "accepts a claim about the configuration alone",
        {ocaml|let () = p "the backend is a CPU or a GPU" (on_cpu || on_gpu)|ocaml},
        [] );
      ( "accepts an if both of whose branches the test computes",
        {ocaml|let () = p "the census renders" (if on_cpu then tiled (Context.auto ()) else intrinsic (Context.auto ()))|ocaml},
        [] );
      ( "does not split a disjunction whose connective the source rebinds",
        {ocaml|let ( || ) a b = a && b
let () = p "the census renders tiled" ((not on_cpu) || census_ok (Context.auto ()))|ocaml},
        [] );
    ]

let folded_gate_findings source =
  let census = Scan.folded_gate_census source in
  List.map census.Scan.folded ~f:(fun (fold : Scan.folded_gate) ->
      fold.when_ ^ if fold.folded_ambiguous then " [ambiguous]" else "")

let run_folded_gate_controls () =
  List.map folded_gate_controls ~f:(fun (label, source, expected) ->
      let found = folded_gate_findings source in
      let ok = List.equal String.equal found expected in
      if not ok then
        eprintf "folded-gate control %S expected [%s], found [%s]\n" label
          (String.concat ~sep:"; " expected)
          (String.concat ~sep:"; " found);
      (label, ok))

let refuse_folded_gate ~fail ~source (fold : Scan.folded_gate) =
  fail
    (Printf.sprintf
       "%s:%d folds a run-constant gate into the value of `%s`: wherever `%s` is false the claim \
        prints `true`, a pass byte-identical to a run that checked it, where the honest report is \
        a skip (gh-ocannl-1310). Report it through `Verdict.gated ~when_:(%s) ~on:<what the run \
        was on>` -- `gated_all`, `gated_alli` or `gated_exists` for a quantified claim -- or a \
        branch that calls `skipped`%s"
       source fold.folded_line fold.folded_form fold.when_ fold.when_
       (ambiguity_remedy ~ambiguous:fold.folded_ambiguous ~callee:(Some fold.folded_form)))

(* The folded-gate reader over one source, its failures told apart as {!read_gates} tells the gate
   reader's apart. *)
let read_folded_gates ~fail ~source ~census content =
  match Test_utils.Scan_attempt.run (fun () -> census content) with
  | Test_utils.Scan_attempt.Scanned folds -> folds
  | Unparsed _ -> { Scan.folded = []; folded_claim_sites = 0 }
  | Raised exn ->
      fail
        (Printf.sprintf
           "the folded-gate reader raised %s on %s, which parses -- a defect of \
            Verdict_scan.folded_gate_census, not of the source, and the source's claim values went \
            unread"
           exn source);
      { Scan.folded = []; folded_claim_sites = 0 }

(* The wiring from a finding to the failure, exercised on a refusing control. *)
let run_folded_gate_refusal_control () =
  let source = "test/operations/verdict_ratchet.ml" in
  let fold_format =
    "%s:%d folds a run-constant gate into the value of `%s`: wherever `%s` is false the claim \
     prints `true`, a pass byte-identical to a run that checked it, where the honest report is a \
     skip (gh-ocannl-1310). Report it through `Verdict.gated ~when_:(%s) ~on:<what the run was \
     on>` -- `gated_all`, `gated_alli` or `gated_exists` for a quantified claim -- or a branch \
     that calls `skipped`%s"
  and raised_format =
    "the folded-gate reader raised %s on %s, which parses -- a defect of \
     Verdict_scan.folded_gate_census, not of the source, and the source's claim values went unread"
  in
  let fixture =
    List.find_map_exn folded_gate_controls ~f:(fun (label, source, _) ->
        if String.equal label "refuses a negated backend gate folded into a claim's value" then
          Some source
        else None)
  in
  let refused_fold = ref false and refused_raise = ref false in
  List.iter (Scan.folded_gate_census fixture).Scan.folded
    ~f:
      (refuse_folded_gate ~source:"fixture.ml" ~fail:(fun _ ->
           refused_fold := true;
           Test_utils.Refusal_control_manifest.observe_failure ~source ~format:fold_format));
  ignore
    (read_folded_gates ~source:"fixture.ml"
       ~fail:(fun _ ->
         refused_raise := true;
         Test_utils.Refusal_control_manifest.observe_failure ~source ~format:raised_format)
       ~census:(fun _ -> raise Stdlib.Exit)
       "let () = ()"
      : Scan.folded_census);
  ( "refuses a planted folded gate, and names a folded-gate reader that raised",
    !refused_fold && !refused_raise )

(* {1 A float tolerance belongs to p_close, not to a parity predicate (gh-ocannl-1224)}

   [Verdict.p_close] names a failure's differing cells, its worst cell with both values and the
   reference's magnitude on stderr; a [p_all2] whose predicate hand-rolls the same tolerance prints
   a bare [false], which is how gh-ocannl-1217's sweep came to need a reproduction round on a remote
   box. The reader is [Verdict_scan.tolerance_census]; its header lists what it cannot see. The
   controls are the predicates it must find, as [<line>:<predicate>], and the near misses it must
   not. *)
let tolerance_controls =
  [
    ( "refuses an inline abs-of-difference predicate in an opened p_all2",
      {ocaml|open Verdict.Claims
let () = p_all2 "matches" got want ~f:(fun a b -> Float.(abs (a - b) < 1e-3))|ocaml},
      [ "2:an inline function" ] );
    ( "refuses a predicate bound to a tolerance function, in each parity form",
      {ocaml|let approx a b = Float.abs (a -. b) < 1e-4
let () =
  Verdict.pf_all2 "%s matches" leg got want ~f:approx;
  Verdict.pass_fail_all2 "matches" got want ~f:approx|ocaml},
      [ "3:approx"; "4:approx" ] );
    ( "refuses a partially applied tolerance function and a renaming of one",
      {ocaml|let close ~tol a b = Float.(abs (a - b) <= tol *. max 1. (abs b))
let near = close ~tol:1e-5
let () =
  Verdict.p_all2 "first" got want ~f:(close ~tol:1e-4);
  Verdict.p_all2 "second" got want ~f:near|ocaml},
      [ "4:close"; "5:near" ] );
    ( "accepts an exact or bitwise parity predicate, which has no tolerance to name",
      {ocaml|let same_bits a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)
let () =
  Verdict.p_all2 "equal" got want ~f:Float.equal;
  Verdict.p_all2 "bitwise" got want ~f:same_bits|ocaml},
      [] );
    ( "does not read a predicate name a later binding takes over",
      {ocaml|let approx a b = Float.(abs (a - b) < 1e-4)
let approx = Float.equal
let () = Verdict.p_all2 "matches" got want ~f:approx|ocaml},
      [] );
    ( "does not read a p_all2 the source binds itself",
      {ocaml|let p_all2 _ _ _ ~f:_ = ()
let () = p_all2 "matches" got want ~f:(fun a b -> Float.(abs (a - b) < 1e-3))|ocaml},
      [] );
  ]

let tolerance_findings source =
  List.map (Scan.tolerance_census source).Scan.hand_rolled ~f:(fun (site : Scan.tolerance_site) ->
      Printf.sprintf "%d:%s" site.tolerance_line site.predicate)

let run_tolerance_controls () =
  List.map tolerance_controls ~f:(fun (label, source, expected) ->
      let found = tolerance_findings source in
      let ok = List.equal String.equal found expected in
      if not ok then
        eprintf "tolerance control %S expected [%s], found [%s]\n" label
          (String.concat ~sep:"; " expected)
          (String.concat ~sep:"; " found);
      (label, ok))

let refuse_hand_rolled_tolerance ~fail ~source (site : Scan.tolerance_site) =
  fail
    (Printf.sprintf
       "%s:%d compares float readbacks through `%s` with %s, which hand-rolls a tolerance (an \
        `abs` of a difference): a failure prints a bare `false`, naming no cell, no values and no \
        magnitude. Write it as `Verdict.p_close <label> got want ~tol:(fun want -> <bound>)` (or \
        `pf_close`, `pass_fail_close`), whose failure names all three on stderr (gh-ocannl-1224)%s"
       source site.tolerance_line site.tolerance_callee
       (if String.equal site.predicate "an inline function" then site.predicate
        else "`" ^ site.predicate ^ "`")
       (ambiguity_remedy ~ambiguous:site.tolerance_ambiguous ~callee:(Some site.tolerance_callee)))

(* The tolerance reader over one source, its failures told apart as {!read_gates} tells the gate
   reader's apart. *)
let read_tolerances ~fail ~source ~census content =
  match Test_utils.Scan_attempt.run (fun () -> census content) with
  | Test_utils.Scan_attempt.Scanned census -> census
  | Unparsed _ -> { Scan.hand_rolled = []; parity_sites = 0 }
  | Raised exn ->
      fail
        (Printf.sprintf
           "the tolerance reader raised %s on %s, which parses -- a defect of \
            Verdict_scan.tolerance_census, not of the source, and the source's parity predicates \
            went unread"
           exn source);
      { Scan.hand_rolled = []; parity_sites = 0 }

(* The wiring from a finding to the failure, exercised on a refusing control. *)
let run_tolerance_refusal_control () =
  let source = "test/operations/verdict_ratchet.ml" in
  let site_format =
    "%s:%d compares float readbacks through `%s` with %s, which hand-rolls a tolerance (an `abs` \
     of a difference): a failure prints a bare `false`, naming no cell, no values and no \
     magnitude. Write it as `Verdict.p_close <label> got want ~tol:(fun want -> <bound>)` (or \
     `pf_close`, `pass_fail_close`), whose failure names all three on stderr (gh-ocannl-1224)%s"
  and raised_format =
    "the tolerance reader raised %s on %s, which parses -- a defect of \
     Verdict_scan.tolerance_census, not of the source, and the source's parity predicates went \
     unread"
  in
  let _, fixture, _ = List.hd_exn tolerance_controls in
  let refused_site = ref false and refused_raise = ref false in
  List.iter (Scan.tolerance_census fixture).Scan.hand_rolled
    ~f:
      (refuse_hand_rolled_tolerance ~source:"fixture.ml" ~fail:(fun _ ->
           refused_site := true;
           Test_utils.Refusal_control_manifest.observe_failure ~source ~format:site_format));
  ignore
    (read_tolerances ~source:"fixture.ml"
       ~fail:(fun _ ->
         refused_raise := true;
         Test_utils.Refusal_control_manifest.observe_failure ~source ~format:raised_format)
       ~census:(fun _ -> raise Stdlib.Exit)
       "let () = ()"
      : Scan.tolerance_census);
  ( "refuses a planted hand-rolled tolerance, and names a tolerance reader that raised",
    !refused_site && !refused_raise )

(* {1 One resolver: every reader agrees whose a claim is (gh-ocannl-1361)}

   Whether a callee is [Verdict]'s is decided once, by [Test_utils.Verdict_callee], and every reader
   looks the answer up. These controls hold the readers to that: one claim that each of the three
   would refuse -- a [gated] claim with a line break in its label, a population gate, and an
   unguarded quantifier -- spelled through each form the resolver distinguishes, and every reader
   must read it, with the same ambiguity, or none may. Before the resolver, the provenance layer
   read a taken-over [Verdict] as [Verdict] and nothing marked a qualified call past another
   module's [open] as ambiguous. *)
type reading = Read | Read_ambiguous | Unread

let agreement_controls =
  [
    ("the readers agree on a claim qualified by Verdict", "", "Verdict.gated", Read);
    ( "the readers agree on a claim an open of Verdict.Claims brings in",
      "open Verdict.Claims\n",
      "gated",
      Read );
    ( "the readers agree on a claim through a module alias of Verdict",
      "module V = Verdict\n",
      "V.gated",
      Read );
    ( "the readers agree on a claim a local structure includes from Verdict.Claims",
      "module C = struct include Verdict.Claims end\n",
      "C.gated",
      Read );
    ( "the readers agree on a claim qualified by Verdict past an open of a standard library",
      "open Base\n",
      "Verdict.gated",
      Read );
    ( "the readers agree that a Verdict past an open of another module is ambiguous",
      "open Other\n",
      "Verdict.gated",
      Read_ambiguous );
    ( "the readers agree that a claim name past an open of another module is ambiguous",
      "open Verdict.Claims\nopen Other\n",
      "gated",
      Read_ambiguous );
    ( "the readers agree that an alias of Verdict past an open of another module is ambiguous",
      "module V = Verdict\nopen Other\n",
      "V.gated",
      Read_ambiguous );
    ( "the readers agree that a module taking over the name Verdict is not Verdict",
      "module Verdict = Local_printer\n",
      "Verdict.gated",
      Unread );
    ( "the readers agree that a path through another module's Verdict is not Verdict",
      "",
      "Other.Verdict.gated",
      Unread );
    ( "the readers agree that a structure including its own Verdict module does not include Verdict",
      "module C = struct module Verdict = Local_printer include Verdict end\n",
      "C.gated",
      Unread );
    ( "the readers agree that a Verdict module an opened structure binds is not Verdict",
      "module M = struct module Verdict = Local_printer end\nopen M\n",
      "Verdict.gated",
      Unread );
    ( "the readers agree that a structure's later unreadable include leaves its Verdict names \
       ambiguous",
      "module C = struct include Verdict.Claims include Other end\n",
      "C.gated",
      Read_ambiguous );
    ( "the readers agree that a structure's unpacked Verdict module is not Verdict",
      "module C = struct let (module Verdict : S) = packed include Verdict end\n",
      "C.gated",
      Unread );
    ( "the readers agree that an opened structure including an unreadable module leaves Verdict \
       ambiguous",
      "module C = struct include Other end\nopen C\n",
      "Verdict.gated",
      Read_ambiguous );
    ( "the readers agree that a claim name a local structure binds is its own",
      "open Verdict.Claims\n\
       module Checks = struct let gated ~when_:_ ~on:_ _ _ = () end\n\
       open Checks\n",
      "gated",
      Unread );
    ( "the readers agree that a claim name the source binds is its own",
      "open Verdict.Claims\nlet gated ~when_:_ ~on:_ _ _ = ()\n",
      "gated",
      Unread );
  ]

let agreement_fixture ~prelude ~callee =
  prelude ^ "let () =\n  " ^ callee
  ^ " ~when_:(not (List.is_empty rows)) ~on:\"cc\" \"first\\nsecond\" (List.for_all rows ~f:Fn.id)"

let run_agreement_controls () =
  List.map agreement_controls ~f:(fun (label, prelude, callee, expected) ->
      let source = agreement_fixture ~prelude ~callee in
      let reading = function
        | [] -> Unread
        | [ false ] -> Read
        | [ true ] -> Read_ambiguous
        | _ -> Unread (* More than one finding is no reading this fixture can earn. *)
      in
      let views =
        [
          ( "quantified",
            reading
              (List.map
                 (quantified_claims (Sources.structure_of source))
                 ~f:(fun claim -> claim.ambiguous)) );
          ( "line-break",
            reading
              (List.map (Scan.line_break_census source).Scan.broken
                 ~f:(fun (site : Scan.line_break_site) -> site.ambiguous)) );
          ( "gate",
            reading
              (List.map (Scan.gate_census source).Scan.readings
                 ~f:(fun (reading : Scan.gate_reading) -> reading.ambiguous)) );
        ]
      in
      let describe = function
        | Read -> "read"
        | Read_ambiguous -> "read as ambiguous"
        | Unread -> "not read"
      in
      let ok = List.for_all views ~f:(fun (_, found) -> Poly.equal found expected) in
      if not ok then
        eprintf "agreement control %S expected every reader to have %s it; %s\n" label
          (describe expected)
          (String.concat ~sep:", "
             (List.map views ~f:(fun (reader, found) -> reader ^ ": " ^ describe found)));
      (label, ok))

(* The remedy an ambiguous claim's refusal adds, through the quantified reader's own diagnostic:
   where the claim is [Verdict]'s only past another module's [open] the refusal names the qualified
   remedy, and past a standard library's it does not. *)
let run_ambiguous_remedy_controls () =
  let remedy prelude =
    quantified_claims
      (Sources.structure_of
         (prelude ^ "let () = Verdict.p \"all rows pass\" (List.for_all rows ~f:Fn.id)"))
    |> List.map ~f:(quantified_failure "fixture.ml")
    |> List.map ~f:(fun message -> String.is_substring message ~substring:"qualify it")
  in
  [
    ( "names the qualified remedy for a quantified claim past an open of another module",
      List.equal Bool.equal (remedy "open Other\n") [ true ] );
    ( "omits the qualified remedy for a quantified claim past an open of a standard library",
      List.equal Bool.equal (remedy "open Base\n") [ false ] );
    (* Control flow choosing between a definite claim function and an ambiguous one fires one claim
       per alternative; collapsed into one, it stays ambiguous. *)
    ( "names the qualified remedy when either alternative claim function is ambiguous",
      List.for_all [ "definite else Verdict.p"; "Verdict.p else definite" ] ~f:(fun branches ->
          List.equal Bool.equal
            (quantified_claims
               (Sources.structure_of
                  ("let definite = Verdict.p\nopen Other\nlet () = (if c then " ^ branches
                 ^ ") \"all rows pass\" (List.for_all rows ~f:Fn.id)"))
            |> List.map ~f:(fun claim -> claim.ambiguous))
            [ true ]) );
  ]

let base_dir = Dune.base_dir
let repo_relative = Dune.repo_relative

(* The sources arrive as a `@<path>` response file: the list outgrew a Windows command line
   (gh-ocannl-1720); see [Test_utils.Scan_argv]. *)
let argv = Test_utils.Scan_argv.expand Stdlib.Sys.argv

let () =
  (match Option.bind (List.nth (Array.to_list argv) 1) ~f:refusal_fixture with
  | None -> ()
  | Some source ->
      let claims = quantified_claims (Sources.structure_of source) in
      if List.is_empty claims then (
        eprintf "the planted fixture produced no finding\n";
        Stdlib.exit 2);
      List.iter claims ~f:(fun claim ->
          Verdict.fail (quantified_failure "control_fixture.ml" claim));
      Stdlib.exit 1);
  if Array.length argv < 2 then (
    eprintf "Usage: %s <workspace_root> <source...>\n" argv.(0);
    Stdlib.exit 1);
  let base = base_dir argv.(1) in
  (* Reported repository-relative, opened as dune handed them over: the working directory is the
     rule's own, deep in the build tree. *)
  let arguments =
    Array.to_list (Array.subo argv ~pos:2)
    |> List.map ~f:(fun path -> (repo_relative base path, path))
  in
  let on_disk = Map.of_alist_reduce (module String) arguments ~f:(fun first _ -> first) in
  (* `.ml` files, minus dune's preprocessed twin of one already in the list: the twin is the ppx
     expansion of a file scanned anyway, and it exists only where the library that owns it is built.
     Shared with the configuration scans, which need the same thing of the same `%{deps}`. *)
  let sources = Sources.sources_among (List.map arguments ~f:fst) in
  if List.is_empty sources then (
    Verdict.fail "no OCaml sources among the arguments -- the rule's globs match nothing";
    Stdlib.exit 1);
  (* Failures go through [Verdict]: the module whose absence at these sites is the whole subject.
     Reported on both channels, and the run exits nonzero from its teardown, so the exit status
     rather than a promotable golden diff carries the verdict (gh-ocannl-601). *)
  let fail message = Verdict.fail message in
  let exemptions = Map.of_alist_exn (module String) exempt_sites in
  let computed_exemptions = Map.of_alist_exn (module String) exempt_computed_sites in
  let quantified_exemptions = Map.of_alist_exn (module String) exempt_quantified_helpers in
  let computed_used = ref (Set.empty (module String)) in
  let quantified_used = ref (Set.empty (module String)) in
  let literal_definitions = ref (Map.empty (module String)) in
  let computed_definitions = ref (Map.empty (module String)) in
  let quantified_definitions = ref (Map.empty (module String)) in
  let canaries = Map.of_alist_exn (module String) canary_sites in
  let data = Map.of_alist_exn (module String) data_sources in
  let exemptions_used = ref (Set.empty (module String)) in
  let canaries_found = ref (Set.empty (module String)) in
  let data_used = ref (Set.empty (module String)) in
  let literals = ref 0 and applied = ref 0 and offenders = ref 0 in
  let quantified_offenders = ref 0 in
  let dialect_offenders = ref 0 and skipped_sites = ref 0 and pass_fail_sites = ref 0 in
  let line_break_offenders = ref 0 and label_sites = ref 0 in
  let gate_offenders = ref 0 and gated_sites = ref 0 in
  let folded_offenders = ref 0 and folded_claim_sites = ref 0 in
  let tolerance_offenders = ref 0 and parity_sites = ref 0 in
  let manifest =
    List.find_map arguments ~f:(fun (relative, path) ->
        if String.is_suffix relative ~suffix:("/" ^ manifest_file) then
          Some (Stdio.In_channel.read_all path)
        else None)
  in
  let control_results =
    run_quantified_helper_controls () @ run_refusal_controls ()
    @ [ run_stale_quantified_control () ]
    @ run_shadowed_quantified_controls ()
    @ run_colliding_site_controls () @ run_dialect_pairing_controls ()
    @ [ run_dialect_refusal_control () ]
    @ run_line_break_controls ()
    @ [ run_line_break_refusal_control () ]
    @ run_gate_population_controls ()
    @ [ run_gate_refusal_control () ]
    @ [ run_claim_reader_refusal_control (); run_matrix_reader_control () ]
    @ run_folded_gate_controls ()
    @ [ run_folded_gate_refusal_control () ]
    @ run_tolerance_controls ()
    @ [ run_tolerance_refusal_control () ]
    @ run_agreement_controls () @ run_ambiguous_remedy_controls ()
  in
  let matrix_rows = run_syntax_matrix () in
  let matrix_results =
    List.map matrix_rows ~f:(fun (family, _, ok) -> (matrix_claim_label family, ok))
  in
  let control_results =
    control_results @ run_manifest_controls ~manifest ~controls:(control_results @ matrix_results)
  in
  let per_directory = Hashtbl.create (module String) in
  printf
    "Test sources that print a claim they decided themselves, outside `Verdict`: a format whose\n\
     last argument-consuming conversion is a bare `%%b` at the end, behind a label ending in `:`,\n\
     `=` or `->` -- written out (gh-ocannl-668) or computed from arguments (gh-ocannl-624). Such\n\
     a line is gated only by the golden diff, and a golden diff is `dune promote`-able -- which\n\
     is how a failure gets recorded as the expected output.\n\n";
  List.iter sources ~f:(fun source ->
      let path = Map.find_exn on_disk source in
      let content = In_channel.read_all path in
      (* A source this reader cannot read is reported by NAME and the scan carries on, rather than
         taking the run down with a syntax error naming no file: the corpus is globbed, so what
         arrives is whatever the test directories hold -- including whatever a `(select …)` or a ppx
         put there -- and the one thing worse than a parse failure here is one that leaves nobody
         knowing which of three hundred files it was about. *)
      let scanned, helper_claims, dialects, line_breaks = read_claims ~fail ~source content in
      label_sites := !label_sites + line_breaks.Scan.label_sites;
      List.iter line_breaks.Scan.broken ~f:(fun site ->
          Int.incr line_break_offenders;
          refuse_line_break ~fail ~source site);
      let gates = read_gates ~fail ~source ~census:Scan.gate_census content in
      gated_sites := !gated_sites + gates.Scan.gated_sites;
      List.iter gates.Scan.readings ~f:(fun reading ->
          Int.incr gate_offenders;
          refuse_gate_reading ~fail ~source reading);
      List.iter gates.Scan.unread ~f:(fun unread ->
          Int.incr gate_offenders;
          refuse_unread_gate ~fail ~source unread);
      List.iter gates.Scan.unlisted ~f:(fun call ->
          Int.incr gate_offenders;
          refuse_unlisted_gate ~fail ~source call);
      let folds = read_folded_gates ~fail ~source ~census:Scan.folded_gate_census content in
      folded_claim_sites := !folded_claim_sites + folds.Scan.folded_claim_sites;
      List.iter folds.Scan.folded ~f:(fun fold ->
          Int.incr folded_offenders;
          refuse_folded_gate ~fail ~source fold);
      let tolerances = read_tolerances ~fail ~source ~census:Scan.tolerance_census content in
      parity_sites := !parity_sites + tolerances.Scan.parity_sites;
      List.iter tolerances.Scan.hand_rolled ~f:(fun site ->
          Int.incr tolerance_offenders;
          refuse_hand_rolled_tolerance ~fail ~source site);
      skipped_sites := !skipped_sites + dialects.Scan.skipped_sites;
      pass_fail_sites := !pass_fail_sites + dialects.Scan.pass_fail_sites;
      List.iter dialects.Scan.pairings ~f:(fun pairing ->
          Int.incr dialect_offenders;
          refuse_dialect_pairing ~fail ~source pairing);
      literals := !literals + scanned.Scan.literals;
      applied := !applied + scanned.Scan.applied_literals;
      Hashtbl.update per_directory (Stdlib.Filename.dirname source) ~f:(fun previous ->
          let files, found = Option.value previous ~default:(0, 0) in
          (files + 1, found + List.length scanned.Scan.sites));
      List.iter scanned.Scan.sites ~f:(fun site ->
          (* A literal-label site is named by its label, which IS what the format says; a computed
             one by the whole format, because its label is only what survived rendering a head this
             reader cannot fill in. *)
          let computed =
            Scan.(match site.kind with Computed_label -> true | Literal_label -> false)
          in
          let key = scan_exemption_key ~source site in
          let where = Printf.sprintf "%s:%d:%d" source site.Scan.line site.Scan.column in
          let how =
            match site.Scan.printer with
            | Some printer -> Printf.sprintf " through `%s`" printer
            | None -> ""
          in
          let canary_key = source ^ ":" ^ site.Scan.label in
          if Map.mem canaries canary_key then (
            canaries_found := Set.add !canaries_found canary_key;
            data_used := Set.add !data_used source)
          else if Map.mem data source then data_used := Set.add !data_used source
          else if (not computed) && Map.mem exemptions key then (
            exemptions_used := Set.add !exemptions_used key;
            literal_definitions := record_scan_definition ~source !literal_definitions site)
          else if computed && Map.mem computed_exemptions key then (
            computed_used := Set.add !computed_used key;
            computed_definitions := record_scan_definition ~source !computed_definitions site)
          else (
            Int.incr offenders;
            let remedy =
              if computed then
                Printf.sprintf
                  "write it as `Verdict.pf \"%s\" <args> <bool>` (or `Verdict.claimf`, if the \
                   surrounding row must keep its shape)"
                  (String.substr_replace_all
                     (String.chop_suffix_if_exists site.Scan.format ~suffix:"\n")
                     ~pattern:": %b" ~with_:"")
              else Printf.sprintf "write it as `Verdict.p \"%s\" <bool>`" site.Scan.label
            in
            fail
              (Printf.sprintf
                 "%s prints the claim `%s`%s, deciding its own verdict outside `Verdict` -- %s, so \
                  that a false exits the run instead of being `dune promote`d into %s. If the line \
                  describes rather than asserts, exempt it by name in verdict_ratchet.ml with the \
                  reason it is not an assertion"
                 where site.Scan.label how remedy
                 (Stdlib.Filename.remove_extension (Stdlib.Filename.basename source) ^ ".expected"))));
      List.iter helper_claims ~f:(fun claim ->
          let key = quantified_exemption_key ~source claim in
          if Map.mem quantified_exemptions key then (
            quantified_used := Set.add !quantified_used key;
            quantified_definitions :=
              record_quantified_definition ~source !quantified_definitions claim)
          else (
            Int.incr quantified_offenders;
            fail (quantified_failure source claim))));
  (* Which directories the corpus came from, by name and not by count: a file added anywhere under
     `test/` moved a tally here, so every contributor would promote this file over a change that
     never touched it -- a promote indistinguishable from blessing a real regression (the lesson of
     gh-ocannl-665). The counts go to stderr, which a `(test)` stanza does not diff. A directory
     that stops being scanned still shows up, by leaving this line. *)
  let directories = Hashtbl.keys per_directory |> List.sort ~compare:String.compare in
  printf "Directories scanned: %s\n\n" (String.concat ~sep:", " directories);
  printf "Sources whose claim-shaped literals are this check's own input, not prints:\n";
  List.iter data_sources ~f:(fun (path, why) -> printf "  %s -- %s\n" path why);
  printf "\nPlanted in that fixture so that a scan which went blind cannot report a clean tree:\n";
  List.iter canary_sites ~f:(fun (key, why) -> printf "  %s -- %s\n" key why);
  printf "\nLiteral-label claims exempted, with the reason each is not an assertion:\n";
  if List.is_empty exempt_sites then
    printf
      "  (none: a bare `<label>: %%b` line with nothing else on it has always been a verdict)\n"
  else List.iter exempt_sites ~f:(fun (key, why) -> printf "  %s -- %s\n" key why);
  printf
    "\n\
     Computed-label claims exempted -- rows and tables that describe rather than decide,\n\
     each carrying its assertion separately through `Verdict.claim`/`claimf`:\n";
  List.iter exempt_computed_sites ~f:(fun (key, why) -> printf "  %s -- %s\n" key why);
  printf "\nQuantified bindings exempted because emptiness is their passing meaning:\n";
  if List.is_empty exempt_quantified_helpers then
    printf "  (none: every quantified binding in a claim must witness a population)\n"
  else List.iter exempt_quantified_helpers ~f:(fun (key, why) -> printf "  %s -- %s\n" key why);
  printf "\nSynthetic helper-rule controls:\n";
  List.iter control_results ~f:(fun (label, ok) -> Verdict.pf "%s" label ok);
  print_syntax_matrix matrix_rows;
  List.iter matrix_results ~f:(fun (label, ok) -> Verdict.pf "%s" label ok);
  let stale =
    Set.union
      (Set.diff (Set.of_list (module String) (List.map exempt_sites ~f:fst)) !exemptions_used)
      (Set.diff
         (Set.of_list (module String) (List.map exempt_computed_sites ~f:fst))
         !computed_used)
  in
  let stale_quantified =
    Set.diff
      (Set.of_list (module String) (List.map exempt_quantified_helpers ~f:fst))
      !quantified_used
  in
  let colliding_quantified = colliding_exemptions !quantified_definitions in
  let colliding_sites =
    colliding_exemptions !literal_definitions @ colliding_exemptions !computed_definitions
  in
  if not (Set.is_empty stale) then
    fail
      (Printf.sprintf
         "exempted literals that no source carries any more -- drop them from the exemption list: \
          %s"
         (String.concat ~sep:", " (Set.to_list stale)));
  refuse_stale_quantified ~fail stale_quantified;
  refuse_colliding_sites ~fail colliding_sites;
  refuse_colliding_quantified ~fail colliding_quantified;
  (* An exempted source that carries no claim-shaped literal is either a file that stopped being a
     fixture, or one this scan stopped reading -- and the second is what a blanket exemption is
     capable of hiding, so it is checked rather than trusted. *)
  let unread = Set.diff (Set.of_list (module String) (List.map data_sources ~f:fst)) !data_used in
  if not (Set.is_empty unread) then
    fail
      (Printf.sprintf
         "sources exempted as this check's own input that carry no claim-shaped literal any more \
          -- either they are no longer fixtures, or the scan is no longer reading them: %s"
         (String.concat ~sep:", " (Set.to_list unread)));
  let missing =
    Set.diff (Set.of_list (module String) (List.map canary_sites ~f:fst)) !canaries_found
  in
  if not (Set.is_empty missing) then
    fail
      (Printf.sprintf
         "planted canaries the scan did not find: %s -- either the fixture no longer carries them, \
          or this scan has stopped reading the corpus and its empty offender list means nothing"
         (String.concat ~sep:", " (Set.to_list missing)));
  eprintf "Sources scanned per directory (not diffed -- see gh-ocannl-665):\n";
  List.iter directories ~f:(fun directory ->
      let files, found = Hashtbl.find_exn per_directory directory in
      eprintf "  %s: %d source%s, %d claim-shaped literal%s\n" directory files
        (if files = 1 then "" else "s")
        found
        (if found = 1 then "" else "s"));
  eprintf "Totals: %d sources, %d string literals (%d of them an argument of a named function).\n"
    (List.length sources) !literals !applied;
  eprintf "Gated quantified claims whose gate and population were compared: %d.\n" !gated_sites;
  eprintf "Verdict claim labels read for line breaks: %d.\n" !label_sites;
  eprintf "Verdict claim values read for a folded gate: %d.\n" !folded_claim_sites;
  eprintf "Verdict parity claims whose predicate was read for a hand-rolled tolerance: %d.\n"
    !parity_sites;
  printf "\n";
  (* Stated so that `true` is the passing reading, as every line of a golden should be. *)
  Verdict.p "every test source decides its claims through Verdict" (!offenders = 0);
  Verdict.p "every quantified binding used by a claim witnesses a non-empty population"
    (!quantified_offenders = 0);
  Verdict.p "no claim label is reported through both pass_fail and skipped" (!dialect_offenders = 0);
  Verdict.p "and the dialect reader found both skipped and pass_fail claims to pair"
    (!skipped_sites > 0 && !pass_fail_sites > 0);
  Verdict.p "no claim label literal carries a line break" (!line_break_offenders = 0);
  Verdict.p "and the line-break reader read the labels of Verdict claims" (!label_sites > 0);
  Verdict.p "no parity claim hand-rolls a float tolerance into its predicate"
    (!tolerance_offenders = 0);
  Verdict.p "and the tolerance reader read the predicates of Verdict parity claims"
    (!parity_sites > 0);
  (* The claim forms whose labels it reads, held equal to every [Verdict.Claims] member but [fail],
     whose argument is a multi-line message: a new claim form it did not read would be a label it
     never checks. *)
  let claims_members =
    Option.bind (Map.find on_disk "test/support/verdict.ml") ~f:(fun path ->
        Test_utils.Refusal_control_scan.claims_members (In_channel.read_all path))
    |> Option.value ~default:[ "<no Claims module in test/support/verdict.ml>" ]
  in
  (* The names whose ownership the one resolver decides, held equal to [Verdict.Claims] itself: a
     member it did not know would be a claim every reader passes over (gh-ocannl-1361). *)
  let resolved_members = List.sort Test_utils.Verdict_callee.members ~compare:String.compare in
  let resolves_every_member =
    List.equal String.equal (List.sort claims_members ~compare:String.compare) resolved_members
  in
  if not resolves_every_member then
    eprintf "Verdict.Claims exports [%s]; Verdict_callee resolves [%s]\n"
      (String.concat ~sep:"; " claims_members)
      (String.concat ~sep:"; " resolved_members);
  Verdict.p "and the claim-name resolver decides every member Verdict.Claims exports"
    resolves_every_member;
  let unlabelled = [ "fail" ] in
  let exported_label_forms =
    List.filter claims_members ~f:(fun member ->
        not (List.mem unlabelled member ~equal:String.equal))
    |> List.sort ~compare:String.compare
  in
  let read_label_forms = List.sort Scan.labelled_claims ~compare:String.compare in
  let reads_every_label_form = List.equal String.equal exported_label_forms read_label_forms in
  if not reads_every_label_form then
    eprintf "Verdict.Claims exports [%s]; the line-break reader reads the labels of [%s]\n"
      (String.concat ~sep:"; " claims_members)
      (String.concat ~sep:"; " Scan.labelled_claims);
  Verdict.p "and it reads the label of every claim form Verdict.Claims exports"
    reads_every_label_form;
  Verdict.p "no gated claim's gate reads the population it quantifies" (!gate_offenders = 0);
  (* The forms the gate reader recognizes, held equal to the gated forms [Verdict.Claims] exports --
     [gated] and the quantified [gated_*]: a new one the reader did not know would be a claim it
     never compares. *)
  let recognized_gated_forms =
    List.sort (Scan.gated_boolean :: Scan.gated_forms) ~compare:String.compare
  in
  let exported_gated_forms =
    Option.bind (Map.find on_disk "test/support/verdict.ml") ~f:(fun path ->
        Test_utils.Refusal_control_scan.claims_members (In_channel.read_all path))
    |> Option.value ~default:[ "<no Claims module in test/support/verdict.ml>" ]
    |> List.filter ~f:(fun member ->
        String.equal member "gated"
        || String.is_prefix member ~prefix:"gated_"
        || String.is_prefix member ~prefix:"<")
    |> List.sort ~compare:String.compare
  in
  let recognizes_every_gated_form =
    List.equal String.equal exported_gated_forms recognized_gated_forms
  in
  if not recognizes_every_gated_form then
    eprintf "Verdict.Claims exports the gated forms [%s]; the gate reader recognizes [%s]\n"
      (String.concat ~sep:"; " exported_gated_forms)
      (String.concat ~sep:"; " recognized_gated_forms);
  Verdict.p "and it recognizes every gated form Verdict.Claims exports" recognizes_every_gated_form;
  Verdict.p "and the gate reader completed gated claims to compare" (!gated_sites > 0);
  Verdict.p "no claim folds a run-constant gate into its value" (!folded_offenders = 0);
  (* The members whose values the folded-gate reader reads, and those it judges to have none, held
     equal together to [Verdict.Claims]: a new member in neither would be a claim whose value it
     never reads. *)
  let folded_judged =
    List.map Scan.folded_slots ~f:(fun (member, _, _) -> member) @ Scan.folded_unread_members
    |> List.sort ~compare:String.compare
  in
  let judges_every_member =
    List.equal String.equal (List.sort claims_members ~compare:String.compare) folded_judged
  in
  if not judges_every_member then
    eprintf "Verdict.Claims exports [%s]; the folded-gate reader judges [%s]\n"
      (String.concat ~sep:"; " claims_members)
      (String.concat ~sep:"; " folded_judged);
  Verdict.p "and the folded-gate reader judges every member Verdict.Claims exports"
    judges_every_member;
  Verdict.p "and it read the values of Verdict claims" (!folded_claim_sites > 0);
  Verdict.p "the scan found every literal planted for it" (Set.is_empty missing);
  Verdict.p "every exemption on this check's lists is still earned"
    (Set.is_empty unread && Set.is_empty stale && Set.is_empty stale_quantified);
  (* Over the exemption lists, which is where a shared key would have to be written. *)
  Verdict.p_empty "every exempted claim-shaped literal is one source site, not a shared key"
    ~over:(exempt_sites @ exempt_computed_sites)
    colliding_sites;
  Verdict.p_empty "every exempted quantified binding is one definition, not a shared name"
    ~over:exempt_quantified_helpers colliding_quantified;
  (* What a blind walk cannot produce. Without these, "no offenders" and "read nothing" are the same
     result -- and the second is the one that arrives silently. *)
  Verdict.p "the walk read string literals out of these sources" (!literals > 0);
  Verdict.p "and placed some of them as arguments of a named function" (!applied > 0);
  Verdict.p "over more than one test directory" (List.length directories > 1);
  Verdict.success_trailer ~blank_line:true
    [ "test claims route through `Verdict`, and quantified bindings cannot pass on nothing." ];
  Test_utils.Refusal_control_manifest.print "verdict_ratchet.ml"
