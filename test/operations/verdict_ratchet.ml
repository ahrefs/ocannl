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

   Every synthetic control below earned its place by a mutation run -- the scanner mechanism it pins
   disabled, this alias re-run, exactly that control failing. The manifest of those runs, one row
   per mechanism with the control labels and the retained `tools/test-run.sh` run ids, is
   `test/operations/verdict_ratchet_controls.md`; renaming or adding a control updates both. *)

open Base
open Stdio

let printf = Test_utils.Refusal_control_manifest.printf

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
    ( "test/operations/autotune_fission_sketch.ml:multi-site composite (not part of the golden): \
       eligible=%b timed=",
      "a measurement census on stderr: an ineligible composite correctly has no window; the \
       adjacent Verdict claim checks eligibility against arrival and the exact CPU count" );
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
    ( "test/operations/backend_golden_family_scan.ml:complete",
      "empty incomplete/error lists are the passing evidence; the non-empty synthetic-control \
       population is guarded in the same binding" );
    ( "test/operations/autotune_routine_name.ml:contributed",
      "a contended search may legitimately contribute no rows; its report counters separately \
       prove whether that absence came from refused timings rather than a lost result" );
    ( "test/operations/epilogue_fusion_mma_seeds.ml:vacuous",
      "an empty GPU mma family deliberately selects the environment-gated vacuity path; the \
       non-vacuous path separately requires and executes the epilogue twins" );
    ( "test/operations/fission_schedule.ml:annotated",
      "the merge-back case deliberately requires the consumer segment to have no hardware axes; \
       the same claim also requires the producer segment to be annotated" );
    ( "test/operations/ocamlformat_ignore_scan.ml:refused",
      "the message list is an optional strengthening of the child-exit refusal: an empty list \
       deliberately means that the nonzero status alone is the passing evidence" );
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
    ( "test/operations/schedule_batched_mma.ml:variance-like site: no cpu mma seeds",
      "the variance-style self-product is deliberately not recognized as a matmul or conv site, so \
       the sketch seeder yields no seeds of any kind and there is no seed population to quantify \
       the mma subset over" );
    ( "test/operations/schedule_batched_mma.ml:variance-like site: no gpu mma seeds",
      "the same site, on the GPU seeder" );
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
                       let found =
                         match Sources.structure_of source with
                         | structure ->
                             quantified_claims structure
                             |> List.map ~f:(fun claim -> claim.helper)
                             |> List.dedup_and_sort ~compare:String.compare
                         | exception exception_ ->
                             eprintf "syntax matrix: %s / %s / %s does not parse: %s\n%s\n"
                               family.family quantifier.quantifier spelling.spelling
                               (Exn.to_string exception_) source;
                             [ "<does not parse>" ]
                       in
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

let quantified_failure source claim =
  let key = quantified_exemption_key ~source claim in
  let quantifiers = List.map claim.quantifiers ~f:quantifier_name |> String.concat ~sep:", " in
  if claim.direct then
    Printf.sprintf
      "%s:%d claims `%s` over a `%s` written directly in the argument at line %d, which can pass \
       on an empty population -- use the matching `Verdict.p_*` combinator (`p_all`, `p_none`, \
       `p_empty ~over`, `p_all2`), or make non-emptiness part of the claimed value. If emptiness \
       is the intended passing case, exempt `%s` by name in verdict_ratchet.ml and say why"
      source claim.claim_line claim.helper quantifiers claim.helper_site.line key
  else
    Printf.sprintf
      "%s:%d sends `%s` from line %d into a Verdict claim, but that binding's `%s` can pass on an \
       empty population -- use the matching `Verdict.p_*` combinator, or make non-emptiness part \
       of the binding's passing result. If emptiness is the intended passing case, exempt `%s` by \
       name in verdict_ratchet.ml and say why"
      source claim.claim_line claim.helper claim.helper_site.line quantifiers key

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

let base_dir = Dune.base_dir
let repo_relative = Dune.repo_relative

let () =
  (match Option.bind (List.nth (Array.to_list Stdlib.Sys.argv) 1) ~f:refusal_fixture with
  | None -> ()
  | Some source ->
      let claims = quantified_claims (Sources.structure_of source) in
      if List.is_empty claims then (
        eprintf "the planted fixture produced no finding\n";
        Stdlib.exit 2);
      List.iter claims ~f:(fun claim ->
          Verdict.fail (quantified_failure "control_fixture.ml" claim));
      Stdlib.exit 1);
  if Array.length Stdlib.Sys.argv < 2 then (
    eprintf "Usage: %s <workspace_root> <source...>\n" Stdlib.Sys.argv.(0);
    Stdlib.exit 1);
  let base = base_dir Stdlib.Sys.argv.(1) in
  (* Reported repository-relative, opened as dune handed them over: the working directory is the
     rule's own, deep in the build tree. *)
  let arguments =
    Array.to_list (Array.subo Stdlib.Sys.argv ~pos:2)
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
    @ run_colliding_site_controls ()
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
      let scanned, helper_claims =
        try (Scan.scan content, quantified_claims (Sources.structure_of content))
        with exception_ ->
          fail
            (Printf.sprintf "%s does not parse as OCaml, so this check cannot vouch for it: %s"
               source (Exn.to_string exception_));
          ({ Scan.sites = []; literals = 0; applied_literals = 0 }, [])
      in
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
  printf "\n";
  (* Stated so that `true` is the passing reading, as every line of a golden should be. *)
  Verdict.p "every test source decides its claims through Verdict" (!offenders = 0);
  Verdict.p "every quantified binding used by a claim witnesses a non-empty population"
    (!quantified_offenders = 0);
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
  if not (Verdict.any_failed ()) then
    printf
      "\nOK: test claims route through `Verdict`, and quantified bindings cannot pass on nothing.\n";
  Test_utils.Refusal_control_manifest.print "verdict_ratchet.ml"
