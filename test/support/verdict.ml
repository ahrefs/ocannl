(** Test verdicts that cannot be [dune promote]d into passing (gh-ocannl-601).

    A [(test)] stanza gates a run on two things: the executable's exit status, and the diff between
    its stdout and the [.expected] golden. A test that decides its own verdict — printing ["PASS"],
    ["FAIL: …"] or ["<claim>: false"] — and exits 0 regardless has only the second gate, and the
    second gate is promotable. A genuine failure changes stdout, the diff fails, and the natural
    next move — [dune promote] — records the failure text as the expected output. Nothing fails
    again until someone reads the golden, and in a golden that is nothing but verdict lines a
    blessed failure is indistinguishable from a designed one: the boolean form is the worst of it,
    since a promoted [false] reads exactly like an intentionally recorded negative fact.

    So verdicts go through here. Each check records its outcome; a run with any failure exits
    nonzero from a teardown registered once, whatever the checks are and however many of them there
    are — so a test built on this module gets the exit-status gate by construction, without an
    end-of-file call it could forget. The failures are also echoed to stderr, because dune neither
    displays nor diffs the redirected stdout of a process that exits nonzero: it does write it, to
    [<name>.exe.output] (or the rule's [.actual]) under [_build/default/<dir>/], where
    [tools/mutation-run.sh] reads it, but stderr is the channel on which the message reaches whoever
    ran the build.

    Assertions belong here; descriptive output does not. Printing a fact the golden pins ("losses:
    [1.0; 0.5]", "producer inlined: true" where the point is to record what happened) stays a plain
    [printf]. What comes here is the line whose content IS the pass/fail decision. *)

open Base

(* The names of the checks that failed, most recent first: the label of a claim, or the first line
   of a {!fail} message. Their number is the run's failure count, so the teardown line can name them
   without a second counter to drift from it. *)
let failed_names = ref []
let record_failure name = failed_names := name :: !failed_names

(* What the teardown at the end of this file raises out of [at_exit] once a check has failed; see
   there. Declared up here because {!case} must let it through. *)
exception Checks_failed

(* The case an explicit [exit] ended the run in, once a check had failed; see {!case}. *)
let exited_in_case = ref None
let passed_label_history = ref []

(** The labels of claims this process actually evaluated successfully, in evaluation order.
    Scanner-control goldens use this as causal evidence: a diagnostic marker is emitted only after
    the claim that would emit that diagnostic has run and passed. *)
let passed_labels () = List.rev !passed_label_history

let record_pass label = passed_label_history := label :: !passed_label_history

(** The names of the checks that failed so far, in order: a claim's label, or the first line of a
    {!fail} message. [Refusal_control_manifest.print_section] reads them to say why a marker did not
    print -- its claim among them, or nothing behind it run (gh-ocannl-1499). *)
let failed_checks () = List.rev !failed_names

(* [text] on one physical line: every control character a space. A claim is one row on each channel
   it prints to: [tools/mutation-run.sh] reads a run's [FAIL: …: false] lines off its log and counts
   its stdout rows against the golden's, and a label carrying a line break split its claim into a
   [FAIL: ] head that is not a false claim and an unprefixed tail that is a row the golden lacks. So
   every claim line renders its label (and a failure's detail) through this, and a computed label
   with a newline in it still reports as one row; [verdict_ratchet] refuses a LITERAL label with a
   line break, where the newline can only be a mistake. What is recorded -- {!passed_labels}, the
   names the teardown line quotes -- is the label as given. *)
let one_line text =
  let is_control c = Char.to_int c < 0x20 || Char.equal c '\x7f' in
  if String.exists text ~f:is_control then
    String.map text ~f:(fun c -> if is_control c then ' ' else c)
  else text

(* [msg] with every line of it prefixed by ["FAIL: "]. A [FAIL: ] row is a run's own (a golden has
   one only where it captures a refusing child), and [tools/mutation-run.sh] counts none toward the
   golden's rows and matches a mutated run's other rows against the golden row by row: an unprefixed
   continuation line -- an "expected/got" pair, a child's captured output -- would be a row the
   golden lacks, and turn a caught mutant into one that "stopped early". A trailing newline adds no
   empty row. *)
let fail_lines msg =
  let lines =
    match String.split msg ~on:'\n' with
    | lines when String.is_suffix msg ~suffix:"\n" -> List.drop_last_exn lines
    | lines -> lines
  in
  String.concat (List.map lines ~f:(fun line -> "FAIL: " ^ line ^ "\n"))

(** Records a failure, on stdout (where the golden sees it) and stderr (where it survives the
    nonzero exit). [msg] should read as the thing that is wrong, without a "FAIL" prefix of its own;
    it may span lines, each of which is printed with the prefix. *)
let fail msg =
  record_failure (match String.lsplit2 msg ~on:'\n' with Some (first, _) -> first | None -> msg);
  let text = fail_lines msg in
  Stdio.print_string text;
  Stdio.eprintf "%s" text

(** [claim name b] fails the run when [b] is false, printing nothing on stdout: for a test that
    renders the boolean itself, in a column layout of its own. Prefer {!p}, which does both. *)
let claim name b =
  if b then record_pass name
  else (
    record_failure name;
    Stdio.eprintf "FAIL: %s: false\n" (one_line name))

(** [p name b] prints the named boolean fact [name: b] and fails the run when [b] is false. Booleans
    keep [.expected] files backend-stable, which is why so many tests report this way; a fact whose
    expected value is [false] should be renamed to the claim that holds, so that every line of a
    golden reads as an assertion that passed. *)
let p name b =
  (* Flushed, so that a run cut short by a crash still shows every check that had passed -- the
     tests that spelled this helper themselves were split on whether to add [%!], and the flush is
     free at test scale. It cannot reorder anything: flushing empties a buffer, it does not move
     writes past one another. *)
  Stdio.printf "%s: %b\n%!" (one_line name) b;
  claim name b

(** [pf fmt … b] is {!p} with a COMPUTED label: the format and its arguments render the label, and
    the boolean follows, so the call reads in the same order as [p name b] —
    [Verdict.pf "%s gradients match the oracle" label ok] prints
    ["… gradients match the oracle: true"] and fails the run on [false].

    This is the entry point for the claims gh-ocannl-601 could not convert (gh-ocannl-624): a claim
    whose label names which leg, which epoch or which measured quantity it is about has to build
    that label from an argument, and before this it had no choice but a bare [printf] — which exits
    0 on [false] and lets the failure be [dune promote]d into the golden, the exact hazard the
    literal-label sites were swept for. A computed label is not a weaker claim than a literal one,
    so it must not have a weaker gate.

    The rendered label is what {!p} prints and what a failure names on stderr, so it should read as
    the fact that holds — the same rule as {!p}: phrase it so [true] is the passing reading. *)
let pf fmt = Printf.ksprintf (fun label b -> p label b) fmt

(** [claimf fmt … b] is {!claim} with a computed label: it fails the run without printing anything
    on stdout, for a test that renders the boolean itself in a column layout of its own. Prefer
    {!pf}, which does both; reach for this only where the surrounding line is a table whose shape is
    the point, and bind the boolean to a name used by both the print and the claim, so the two
    cannot drift apart. *)
let claimf fmt = Printf.ksprintf (fun label b -> claim label b) fmt

(** {1 Quantified claims}

    ["every X …"] is TRUE of an empty X, and in a golden that line is byte-identical to one a real
    population passed (gh-ocannl-729). The hole is invisible by construction, and it opens exactly
    where the claim is most worth making: quantified over a DERIVED collection — the seeds a family
    tree yields, the refutations a gate raises, the statements a kernel emits — where the collection
    being empty is a plausible regression rather than an impossible one. A seeding change that
    empties a family silently converts several claims from checks into decoration.

    So a quantified claim goes through one of the combinators below rather than through {!p} applied
    to a [List.for_all]: they carry the non-emptiness guard, which makes the guarded form the
    shortest one to write. A non-empty collection prints exactly what {!p} prints, so goldens keep
    their shape; an empty one prints a DISTINCT [<claim> (empty): false], so a reader sees why the
    line failed without opening the source.

    The single-collection combinators take lists; an array reaches them through [Array.to_list],
    which is free at test scale and keeps this library's surface small. The PAIRWISE one, {!p_all2},
    is the exception and takes arrays — the executed-parity genre it serves reads both sides out of
    [Context.get_values], and it has to compare their LENGTHS, which is the one question a list
    cannot answer without walking it.

    [?min] raises the floor for a site that knows one — ["every one of the four curated tiles …"] is
    a different claim from ["every tile …"], and a menu that silently shrank to one member should
    fail it. In {!p_all}, {!p_none} and {!p_empty} it bounds the COLLECTION, which is where those
    claims can go vacuous; in {!p_exists} it counts WITNESSES, for the reason given there. Below the
    floor the line names the shortfall: [<claim> (only 1 of 4): false]. *)

(* A failure the claim itself could not express: the collection was too small for the quantifier to
   mean anything, so the line says so where a reader will find it rather than printing the bare
   [false] a satisfied-looking claim shares. *)
let short_fail name detail =
  record_failure name;
  let line = Printf.sprintf "%s (%s): false" (one_line name) (one_line detail) in
  Stdio.printf "%s\n%!" line;
  Stdio.eprintf "FAIL: %s\n" line

(* The one place the POPULATION floor is checked, so that every combinator resting on it reports a
   short collection the same way. [holds] is not evaluated when the floor fails: on an empty
   collection a quantifier answers without looking, and answering is what we are refusing to accept.
   The default floor asks [is_empty] rather than [length], so the common path stays O(1) over a
   collection that can be a whole tensor readback; the length is computed only to report a
   failure. *)
let with_population ?(min = 1) name ~is_empty ~length holds =
  if if min <= 1 then not (is_empty ()) else length () >= min then p name (holds ())
  else
    let length = length () in
    short_fail name (if length = 0 then "empty" else Printf.sprintf "only %d of %d" length min)

let quantified ?min name xs holds =
  with_population ?min name
    ~is_empty:(fun () -> List.is_empty xs)
    ~length:(fun () -> List.length xs)
    holds

(** [p_all name xs ~f] claims that every element of [xs] satisfies [f], and that there is an element
    — the guarded form of [p name (List.for_all xs ~f)]. Reach for it wherever the claim reads
    ["every …"]. Once the population floor is met, [f] runs on every element in list order, even
    after a refutation, so diagnostic side effects are not lost. *)
let p_all ?min name xs ~f =
  quantified ?min name xs (fun () ->
      List.fold xs ~init:true ~f:(fun holds x ->
          let element_holds = f x in
          holds && element_holds))

(** [p_all2 name got want ~f] claims that [got] and [want] agree cell for cell under [f], that they
    have the same length, and that there is a cell — the guarded form of
    [p name (Array.for_all2_exn got want ~f)], and the executed-parity genre's entry point
    (gh-ocannl-746). Reach for it wherever a claim compares a readback against a reference.

    Emptiness is not far-fetched here, which is why the guard matters more than it looks. Both sides
    usually come from [Context.get_values], and a node that stopped being materialized, a reference
    run whose own setup collapsed, or a readback of a virtualized node answers with NOTHING ON BOTH
    SIDES AT ONCE — the reference does not discriminate, because it went through the same path.
    [Array.for_all2_exn [||] [||]] is [true], so the line that reports the strongest kind of check
    AGENTS.md asks for is byte-identical to one a real readback passed.

    A LENGTH MISMATCH is reported as a failed claim, [<claim> (length 12 vs 16): false], rather than
    as the [Invalid_argument] [Array.for_all2_exn] raises: an exception fails the run too, but it
    fails it without a claim line and without naming which claim, and the mismatch is a finding
    about the run (one side was reshaped, or one readback was truncated) that deserves to be read
    off the transcript. Reported before emptiness, so a one-sided empty says which side it was.

    [?min] bounds the COMMON length, as in {!p_all}.

    A VALUE failure prints the bare [<claim>: false] {!p} prints, and adds one line on stderr, which
    no golden diffs: how many cells failed [f] and the first of them,
    [<claim>: 2 of 64 cells fail the predicate; first at 17 (not part of the golden)]
    (gh-ocannl-1224). [f] therefore runs on every cell once the population floor is met, as in
    {!p_all}. It cannot print the values, since it does not know their type: a claim over float
    readbacks under a tolerance goes through {!p_close}, which does. *)
type all2_outcome = All2_short of string | All2_holds of { failed : int; first : int }

(* Shared by the two output dialects below: [p_all2] renders booleans and [pass_fail_all2] renders
   PASS/FAIL, but a missing population and a length mismatch are the same refusal in either dialect.
   Keeping the classification here also keeps [f] unevaluated when the pair is structurally too
   short to support its claim. Past the floor [f] visits every cell, so a failure can be counted;
   [first] is meaningful only where [failed > 0]. *)
let all2_outcome ?(min = 1) got want ~f =
  let n_got = Array.length got and n_want = Array.length want in
  if n_got <> n_want then All2_short (Printf.sprintf "length %d vs %d" n_got n_want)
  else if if min <= 1 then n_got > 0 else n_got >= min then (
    let failed = ref 0 and first = ref (-1) in
    Array.iteri got ~f:(fun i a ->
        if not (f a want.(i)) then (
          if !failed = 0 then first := i;
          Int.incr failed));
    All2_holds { failed = !failed; first = !first })
  else All2_short (if n_got = 0 then "empty" else Printf.sprintf "only %d of %d" n_got min)

(* The stderr line a value failure of [p_all2] or [pass_fail_all2] adds: a count and a position,
   which is all a predicate over an unknown type can name. Not prefixed [FAIL: ]: the claim's own
   line is the one [tools/mutation-run.sh] reads, and this is diagnosis beside it. *)
let report_all2_failure name ~cells ~failed ~first =
  Stdio.eprintf "%s: %d of %d cell%s fail the predicate; first at %d (not part of the golden)\n%!"
    (one_line name) failed cells
    (if cells = 1 then "" else "s")
    first

let p_all2 ?min name got want ~f =
  match all2_outcome ?min got want ~f with
  | All2_short detail -> short_fail name detail
  | All2_holds { failed = 0; _ } -> p name true
  | All2_holds { failed; first } ->
      p name false;
      report_all2_failure name ~cells:(Array.length got) ~failed ~first

(** [pf_all2 fmt … got want ~f] is {!p_all2} with a COMPUTED label, as {!pf} is to {!p}. Format
    arguments precede the two arrays and their predicate, so the call keeps the claim-shaped order:
    [pf_all2 "%s values match" leg got want ~f]. Empty or mismatched readbacks retain [p_all2]'s
    distinct refusal wording; [?min] has the same common-length meaning. *)
let pf_all2 ?min fmt = Printf.ksprintf (fun label got want ~f -> p_all2 ?min label got want ~f) fmt

(** {2 Float parity under a tolerance (gh-ocannl-1224)}

    [p_close name got want ~tol] claims that the float readback [got] matches the reference [want]
    cell for cell: each [got.(i)] equals [want.(i)], or lies within [tol want.(i)] of it
    ([Float.(abs (got - want) <= tol want)]). [tol] answers the admitted difference for a reference
    value, so one argument spells every tolerance the parity tests use: absolute ([fun _ -> 1e-3]),
    relative with an absolute floor ([fun w -> 1e-3 +. (1e-5 *. Float.abs w)]), relative above
    magnitude one ([fun w -> 1e-4 *. Float.max 1. (Float.abs w)]), or exact ([fun _ -> 0.]). Its
    structural refusals are {!p_all2}'s -- an empty pair, a length mismatch, a pair below [?min] --
    and so is its passing line.

    What it adds is the failure: the bare [<claim>: false] on stdout, as {!p} prints it, and on
    stderr one line that answers the triage questions a bare [false] cannot, outside every golden:
    [<claim> (<what_ran>): 3 of 4096 cells differ beyond tolerance; first at 17, worst at 902: got
     196608.031 vs want 196608 (|diff| 0.0312, tolerance 0.001 there); max |want| 2.1e+05 (not part
     of the golden)]. The worst cell is the failing one furthest from its reference (a NaN on either
    side ranks first), and max |want| is the magnitude that tells a few ulps of reordered rounding
    from a misread operand at a glance. gh-ocannl-1217 needed a reproduction round on a remote box
    to learn exactly that; with this line, the sweep log alone carries it. [?what_ran] names the
    candidate that produced [got] (a seed, a schedule, a preset) when the label is shared across a
    loop of them.

    A cell where the two values are EQUAL holds whatever [tol] says, so matching infinities match,
    and a NaN on either side never does. The bound is inclusive, so [fun _ -> 0.] is exact equality.
*)
type close_outcome =
  | Close_short of string
  | Close_holds of {
      differing : int;
      first : int;
      worst : int;
      magnitude : float;
      worst_tol : float;
    }

let close_outcome ?(min = 1) ~tol got want =
  let n_got = Array.length got and n_want = Array.length want in
  if n_got <> n_want then Close_short (Printf.sprintf "length %d vs %d" n_got n_want)
  else if if min <= 1 then n_got > 0 else n_got >= min then (
    let differing = ref 0 and first = ref (-1) and worst = ref (-1) and magnitude = ref 0. in
    (* How far a failing cell is from its reference, for ranking: a NaN is a tier of its own above
       every distance, the infinite ones included, so a NaN cell is the worst whatever precedes
       it. *)
    let rank i =
      let d = Float.abs (got.(i) -. want.(i)) in
      if Float.is_nan d then (1, 0.) else (0, d)
    in
    let further a b =
      let (tier_a, d_a), (tier_b, d_b) = (rank a, rank b) in
      tier_a > tier_b || (tier_a = tier_b && Float.(d_a > d_b))
    in
    Array.iteri got ~f:(fun i g ->
        let w = want.(i) in
        if not (Float.is_nan w) then magnitude := Float.max !magnitude (Float.abs w);
        (* Only equal values pass without a finite difference: a finite value against an infinite
           reference, or opposite infinities, differ by an infinite amount, which no tolerance -- an
           infinite relative one included -- admits. *)
        let d = Float.abs (g -. w) in
        if not (Float.equal g w || (Float.is_finite d && Float.(d <= tol w))) then (
          if !differing = 0 then first := i;
          Int.incr differing;
          (* Strictly further: among equals the earliest stands, and an earlier NaN is not displaced
             by a later one. *)
          if !worst < 0 || further i !worst then worst := i));
    Close_holds
      {
        differing = !differing;
        first = !first;
        worst = !worst;
        magnitude = !magnitude;
        worst_tol = (if !worst < 0 then 0. else tol want.(!worst));
      })
  else Close_short (if n_got = 0 then "empty" else Printf.sprintf "only %d of %d" n_got min)

let report_close_failure ?what_ran name got want ~differing ~first ~worst ~magnitude ~worst_tol =
  Stdio.eprintf
    "%s%s: %d of %d cell%s differ%s beyond tolerance; first at %d, worst at %d: got %.9g vs want \
     %.9g (|diff| %.3g, tolerance %.3g there); max |want| %.6g (not part of the golden)\n\
     %!"
    (one_line name)
    (match what_ran with None -> "" | Some what -> " (" ^ one_line what ^ ")")
    differing (Array.length got)
    (if Array.length got = 1 then "" else "s")
    (if differing = 1 then "s" else "")
    first worst got.(worst) want.(worst)
    (Float.abs (got.(worst) -. want.(worst)))
    worst_tol magnitude

let p_close ?min ?what_ran name got want ~tol =
  match close_outcome ?min ~tol got want with
  | Close_short detail -> short_fail name detail
  | Close_holds { differing = 0; _ } -> p name true
  | Close_holds { differing; first; worst; magnitude; worst_tol } ->
      p name false;
      report_close_failure ?what_ran name got want ~differing ~first ~worst ~magnitude ~worst_tol

(** [pf_close fmt … got want ~tol] is {!p_close} with a COMPUTED label, as {!pf_all2} is to
    {!p_all2}: [pf_close "%s: the tuned routine matches the plain compile" leg got want ~tol]. *)
let pf_close ?min ?what_ran fmt =
  Printf.ksprintf (fun label got want ~tol -> p_close ?min ?what_ran label got want ~tol) fmt

(** [within name got want ~tol] answers whether {!p_close} would pass, WITHOUT making a claim: on a
    [false] it prints {!p_close}'s stderr line (or the structural reason,
    [<name>: empty (not part of the golden)]) and records nothing. It is the predicate of an
    aggregate claim over candidates whose number is not the golden's to pin --
    [p_all "every seed matches" runs ~f:(fun (seed, got) -> Verdict.within ~what_ran:seed "the
     block" got want ~tol)] -- so that the claim stays one row while the candidate that failed it
    still names its worst cell. A single readback goes through {!p_close}. Not a [Claims] member,
    being no claim. *)
let within ?min ?what_ran name got want ~tol =
  match close_outcome ?min ~tol got want with
  | Close_holds { differing = 0; _ } -> true
  | Close_holds { differing; first; worst; magnitude; worst_tol } ->
      report_close_failure ?what_ran name got want ~differing ~first ~worst ~magnitude ~worst_tol;
      false
  | Close_short reason ->
      Stdio.eprintf "%s%s: %s (not part of the golden)\n%!" (one_line name)
        (match what_ran with None -> "" | Some what -> " (" ^ one_line what ^ ")")
        (one_line reason);
      false

(** [p_none name xs ~f] claims that no element of [xs] satisfies [f], and that there is an element —
    the guarded form of [p name (not (List.exists xs ~f))] and of
    [p name (List.is_empty (List.filter xs ~f))]. The mirror of {!p_all}, and the one the
    ["no X is …"] claims want: filtering an empty collection also yields nothing, so the unguarded
    spelling passes on an empty input just as [List.for_all] does. *)
let p_none ?min name xs ~f = p_all ?min name xs ~f:(fun x -> not (f x))

(** [p_alli name xs ~f] is {!p_all} with each element's index beside it -- the guarded form of
    [p name (List.for_alli xs ~f)], for the claims that compare an element with its neighbour or
    with a reference at the same position. Arrays go through [Array.to_list], as for {!p_all}. Like
    {!p_all}, it evaluates every element once the population floor is met. *)
let p_alli ?min name xs ~f =
  quantified ?min name xs (fun () ->
      List.foldi xs ~init:true ~f:(fun i holds x ->
          let element_holds = f i x in
          holds && element_holds))

(** [p_empty name ~over xs] claims that the derived collection [xs] is empty, and that the
    collection it was derived from, [over], is not — the guarded form of [p name (List.is_empty xs)]
    where [xs] is a precomputed subset (the invalid seeds, the declined candidates, the offending
    rows) of a population that must itself exist. Prefer {!p_none} where the predicate can simply be
    passed; this is for the sites that keep the derived list around to report it. *)
let p_empty ?min name ~over xs = quantified ?min name over (fun () -> List.is_empty xs)

(** [p_exists name xs ~f] claims that some element of [xs] satisfies [f], and [p_exists ~min:n] that
    at least [n] do. Its [?min] counts WITNESSES where its siblings' counts the population, and the
    asymmetry is the point rather than an inconsistency: for an existential a population floor buys
    nothing, since [List.exists] already answers [false] on an empty collection, so the only reading
    of [~min:2] that adds a constraint is "two of them". Reading it as a population floor is what
    would produce the false green this module exists to prevent — a two-element list with one
    witness passing ["at least two of the seeds pipeline"] (Codex P2, round 1).

    An empty collection still says so: [<claim> (empty): false], because "nothing satisfied [f]" and
    "there was nothing to satisfy it" are different findings. Too few witnesses under an explicit
    floor names the shortfall as [<claim> (only 1 of 2 match): false] — "match", so that the line
    cannot be read as the population shortfall its siblings print. At the default floor of one the
    line is the bare [<claim>: false] {!p} would print: there is no shortfall to name. *)
let p_exists ?(min = 1) name xs ~f =
  if List.is_empty xs then short_fail name "empty"
  else
    let witnesses = List.count xs ~f in
    if witnesses >= min then p name true
    else if min <= 1 then p name false
    else short_fail name (Printf.sprintf "only %d of %d match" witnesses min)

(** [p_pairwise_distinct name xs ~equal ~to_string] claims that [xs] contains at least two elements
    and that no two of them are equal. It reports the first colliding pair in source order directly:
    [<claim> (collision: x = y): false]. [to_string] is evaluated only on a collision, so a passing
    claim pays only for the comparisons and keeps the same stdout line as {!p}.

    This is the guarded form of building every unordered pair and passing it to {!p_none}. The
    source-population floor belongs here rather than on the derived pairs: zero source elements
    produce zero pairs, but one source element does too, and neither is evidence of distinctness
    (gh-ocannl-815). *)
let p_pairwise_distinct name xs ~equal ~to_string =
  let rec first_collision = function
    | [] -> None
    | x :: rest -> (
        match List.find rest ~f:(equal x) with
        | Some y -> Some (x, y)
        | None -> first_collision rest)
  in
  match first_collision xs with
  | Some (x, y) -> short_fail name (Printf.sprintf "collision: %s = %s" (to_string x) (to_string y))
  | None -> quantified ~min:2 name xs (fun () -> true)

(* Per-action machine records (gh-ocannl-1114). Every machine-readable record this module emits --
   today the skip record of {!skipped} -- is a line [OCANNL_TOOL_VERDICT_<KIND><TAB><fields>],
   echoed on stderr where a human reading the log sees it, and, when [OCANNL_TOOL_VERDICT_RECORDS]
   names a directory, also appended to ONE FILE PER ACTION in it. The file is keyed by what
   identifies the action that ran this process -- its working directory, executable and arguments,
   which is how `test_cse.exe` under `runtest-test_cse` and under `runtest-test_cse-cse_d_fwd` stay
   two actions -- and is TRUNCATED when the process starts, before any record. So the file always
   holds its action's latest run: a retry of the action rewrites it, one that now records nothing
   empties it, and an action never re-run keeps its own. `tools/sweep.sh` names the directory for a
   unit's first attempt, serial rerun and completion pass alike, and reads it back afterwards: each
   action's final attempt by construction, where the merged stderr of the three phases could only be
   attributed by inference, and where dune truncating a noisy action's output could drop records
   outright.

   The file opens with an [OCANNL_TOOL_VERDICT_ACTION] header naming the action, which is also what
   says the action ran at all; a record kind the sweep's aggregator does not judge -- the bypass
   record gh-ocannl-996 proposes, say -- is one more [<KIND>] in the same stream, not another file.
   Two actions identical in directory, executable and arguments but run under different environments
   would share a file, the later one's records standing -- which matters only if the two skip
   different claims, so arguments, not a `setenv` alone, are what should tell such twins apart.

   Only the action's own process writes. Having read the variable, Verdict EMPTIES it in its own
   environment, so no descendant -- however it is spawned, whatever helper builds its environment
   from this one -- sees a directory: a child is the parent's fixture, its records belong to the
   parent's judgment, and a child relaunching the same executable with the same arguments in the
   same directory (`cc_march_census` does) would otherwise truncate the parent's own file and erase
   its skips. A process that is not Verdict-linked, a shell harness running fixtures, must unset it
   itself (`sweep_harness.sh` does). Opening or writing the file failing is a failed check, never a
   silently missing record, since a missing record reads as an executed claim. *)

let record_prefix = "OCANNL_TOOL_VERDICT_"
let record_line kind fields = String.concat ~sep:"\t" ((record_prefix ^ kind) :: fields)

let action_records =
  match Stdlib.Sys.getenv_opt "OCANNL_TOOL_VERDICT_RECORDS" with
  | None | Some "" -> None
  | Some dir -> (
      (* Before anything else can spawn: every descendant inherits the emptied value, which reads as
         unset above. *)
      Unix.putenv "OCANNL_TOOL_VERDICT_RECORDS" "";
      let exe = Stdlib.Sys.executable_name and cwd = Stdlib.Sys.getcwd () in
      let args = List.tl (Array.to_list Stdlib.Sys.argv) |> Option.value ~default:[] in
      let key =
        Stdlib.Digest.to_hex (Stdlib.Digest.string (String.concat ~sep:"\000" (cwd :: exe :: args)))
      in
      let name = Stdlib.Filename.basename exe in
      let path = Stdlib.Filename.concat dir (Printf.sprintf "%s-%s.tsv" name key) in
      (* [Open_append] beside [Open_trunc]: the open empties the file, and every later write lands
         at its end, so two same-keyed processes overlapping can duplicate a line but never
         overwrite one or leave a hole. *)
      match
        Stdlib.open_out_gen
          [ Open_wronly; Open_creat; Open_trunc; Open_append; Open_binary ]
          0o644 path
      with
      | channel -> (
          let escaped = Stdlib.String.escaped in
          let header =
            record_line "ACTION"
              [ escaped name; escaped cwd; String.concat ~sep:" " (List.map args ~f:escaped) ]
          in
          match
            Stdlib.output_string channel (header ^ "\n");
            Stdlib.flush channel
          with
          | () -> Some (path, channel)
          | exception Sys_error e ->
              fail (Printf.sprintf "cannot write the per-action verdict records %s: %s" path e);
              None)
      | exception Sys_error e ->
          fail ("cannot open the per-action verdict records: " ^ e);
          None)

(* One machine record: on stderr, and in this action's record file when there is one. *)
let emit_record kind fields =
  let line = record_line kind fields in
  Stdio.eprintf "%s\n%!" line;
  Option.iter action_records ~f:(fun (path, channel) ->
      match
        Stdlib.output_string channel (line ^ "\n");
        Stdlib.flush channel
      with
      | () -> ()
      | exception Sys_error e ->
          fail (Printf.sprintf "cannot write the per-action verdict records %s: %s" path e))

type skip_aggregation = [ `Backend | `Environment | `Outside_sweep ]
(** [skipped ~backend name] reports a leg the run's backend cannot evaluate: a GPU intrinsic on a
    CPU backend, a tf32 policy outside CUDA. It prints the same stdout line {!p} would — the
    [.expected] goldens are backend-uniform, and a [(test)] stanza diffs stdout ONLY, so stderr is
    free — and announces the skip on stderr, naming the claim. [grep SKIPPED] over a run then
    enumerates exactly what that hardware did not verify. A second
    [OCANNL_TOOL_VERDICT_SKIP<TAB>scope<TAB>executable<TAB>claim] record on stderr is the
    machine-readable form, and the one the action's record file receives (the per-action records
    above), which is what the forced sweep intersects across backends; the executable identity keeps
    equal labels in different test legs distinct there. [scope] is [backend] by default. Pass
    [~aggregation:`Environment] when the leg is gated by a host or configuration capability rather
    than the selected backend, such as a compiler target, preprocessing flag or filesystem feature;
    its human announcement remains visible, and the forced fleet sweep aggregates it across the
    measurement boxes declared by [benchmarks/fixtures/DIGESTS.txt] instead of mistaking it for
    unsupported backend coverage. Scope describes this observation, not the claim identity: if any
    declared-box log marks a claim [Environment], a [Backend] skip carrying the same executable and
    claim key on another box is also evidence that box did not execute it. Pass
    [~aggregation:`Outside_sweep] only when a separate checked matrix owns execution of the claim
    (for example the compiler-trace CI leg); the record remains validated and human-visible but
    neither sweep dimension treats its absence as evidence.

    Use it in place of a bare [p name true]: that line is byte-identical to a verified run's, so
    neither the transcript nor a reviewer can tell the claim was never evaluated — which is how a
    [Tensorize] leg came to "cover" the gh-ocannl-528 interior-batch bug without ever checking it. A
    gate folded into the claim's value, [p name ((not gate) || b)], prints that same line where the
    gate is closed, and [verdict_ratchet] refuses it (gh-ocannl-1310). The other honest form is
    putting the condition into the label itself ("… (skipped: non-C backend)"), which distinguishes
    the golden line; only the indistinguishable bare [true] is the one to reject.

    [~backend] is the run's backend name, as each test already derives it for its own gating
    ([String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")]). It is passed in
    rather than read here so that this library keeps depending on no OCANNL library, only [base],
    [stdio] and [unix]: reporting a verdict is not a reason to link OCANNL's configuration
    machinery. *)

let skipped ?(aggregation = (`Backend : skip_aggregation)) ~backend name =
  Stdio.eprintf "SKIPPED on %s (vacuous): %s\n%!" (one_line backend) (one_line name);
  (* A second, machine-oriented record gives cross-run consumers a stable test-leg identity without
     changing the human line above (the documented [grep SKIPPED] convention). The executable
     basename is stable across worktrees and machines; pairing it with the claim prevents equal
     labels in different tests from being conflated. [String.escaped] keeps each field on one TSV
     line even if a computed label contains a control character. *)
  emit_record "SKIP"
    [
      (match aggregation with
      | `Backend -> "backend"
      | `Environment -> "environment"
      | `Outside_sweep -> "outside-sweep");
      Stdlib.String.escaped (Stdlib.Filename.basename Stdlib.Sys.executable_name);
      Stdlib.String.escaped name;
    ];
  p name true

(* The one place a gate chooses between evaluating a claim and announcing its skip, so every gated
   form below skips the same way and nothing is evaluated on the closed side. *)
let gate ?aggregation ~when_ ~on label evaluate =
  if when_ then evaluate () else skipped ?aggregation ~backend:on label

(** [gated ~when_ ~on label b] is a claim that is evaluated only where the gate [when_] is open:
    there it is {!p} — [label: true], or a failure — and where the gate is closed it is {!skipped}
    [~backend:on], [b] unread. Both outcomes print [label: true] on a passing run, so a host that
    legitimately skips cannot break the [.expected] golden (gh-ocannl-997).

    It exists because that byte-identity was a convention the call site had to know. A gated claim
    written as [if gate then pass_fail label b else skipped ~backend label] records [label: PASS] in
    the golden on the machine that promotes it, and fails the diff with [label: true] on the one
    machine that reaches the skipping regime — which is the machine least likely to be promoting.
    Twice this shipped as far as review (staging#735's depth gate, and a host gate in
    gh-ocannl-1031's first draft). This entry point picks the dialect itself, and [verdict_ratchet]
    refuses a label that reaches both {!pass_fail} and {!skipped}.

    [~on] names what the run was on when it skipped — a backend for the default [`Backend]
    aggregation, a host or configuration capability for [`Environment] — and is printed as
    [SKIPPED on <on> (vacuous): <label>]. It is [~on] rather than {!skipped}'s [~backend] because a
    host gate is not a backend. [?aggregation] is {!skipped}'s.

    [?detail] is what {!pass_fail} offered and a gated claim had to give up: evaluated only on an
    evaluated failure, and rendered the way this dialect renders what a bare [false] cannot say,
    [<label> (<detail>): false] — the place for a measured number that must stay out of a passing
    golden.

    [b] is an ordinary [bool], evaluated by the caller whatever the gate says, so it must be
    computable off-gate — which is also what lets [verdict_ratchet]'s quantifier reader treat this
    exactly as {!p}. A leg that cannot even compute its boolean without the gate keeps its own
    branch and reports through {!p} on one side and {!skipped} on the other: the same dialect. A
    quantified claim goes through {!gated_all} and its siblings. *)
let gated ?aggregation ?detail ~when_ ~on label b =
  gate ?aggregation ~when_ ~on label (fun () ->
      match detail with
      | Some detail when not b -> short_fail label (detail ())
      | Some _ | None -> p label b)

(** [gated_all ~when_ ~on label xs ~f] is {!gated} for a quantified claim: {!p_all} where the gate
    is open, {!skipped} [~on] where it is closed. [gated_alli] and [gated_exists] do the same for
    {!p_alli} and {!p_exists}, and [?min] and [?aggregation] mean what they mean there.

    {!gated} cannot carry these: [gated label (List.for_all xs ~f)] is the unguarded quantifier
    [verdict_ratchet] refuses in {!p} too, since a gate does not fill the population. Without this
    family the guarded spelling was a hand-written branch,
    [if gate then p_alli label xs ~f else skipped ~backend label], repeating the label and the
    skip's arguments at every site.

    Where the gate is closed, neither [xs] nor [f] is read. A leg whose population exists only where
    it ran (the seeds a GPU proposes, the files a host could create) can therefore pass the empty
    list it is left with off-gate, computed unconditionally; the non-emptiness floor applies exactly
    where the claim is evaluated. A leg whose population cannot be computed off-gate at all (a
    device readback) keeps its own branch, as {!gated} says of its boolean. *)
let gated_all ?aggregation ?min ~when_ ~on label xs ~f =
  gate ?aggregation ~when_ ~on label (fun () -> p_all ?min label xs ~f)

let gated_alli ?aggregation ?min ~when_ ~on label xs ~f =
  gate ?aggregation ~when_ ~on label (fun () -> p_alli ?min label xs ~f)

let gated_exists ?aggregation ?min ~when_ ~on label xs ~f =
  gate ?aggregation ~when_ ~on label (fun () -> p_exists ?min label xs ~f)

(** [pass_fail label b] prints [label: PASS] or [label: FAIL], and fails the run in the latter case,
    echoing that row behind [FAIL: ] on stderr. [?detail] is evaluated only on failure and appended
    in parentheses — the place for a machine- specific number (a measured value, a difference) that
    must stay out of a passing golden.

    Not for a claim some host skips: {!skipped} prints [label: true], never [label: PASS], so that
    pairing breaks the golden wherever the skip fires. A gated claim goes through {!gated}. *)
let pass_fail ?detail label b =
  if b then (
    record_pass label;
    Stdio.printf "%s: PASS\n" (one_line label))
  else
    let detail = match detail with None -> "" | Some f -> " (" ^ one_line (f ()) ^ ")" in
    let row = Printf.sprintf "%s: FAIL%s" (one_line label) detail in
    record_failure label;
    (* Its stderr line is the row behind [FAIL: ], as every claim's is ([FAIL: <label>: false]):
       [tools/mutation-run.sh] lists a run's failed claims by that line's ending alone. *)
    Stdio.printf "%s\n" row;
    Stdio.eprintf "FAIL: %s\n" row

(** [pass_fail_all2 label got want ~f] is {!p_all2} in {!pass_fail}'s output dialect. It refuses
    empty, short, and mismatched pairs before evaluating [f]. On those structural refusals the
    automatic reason is kept even when [?detail] is supplied; the caller's lazy detail follows it,
    separated by a semicolon, so an empty readback can report both [(empty)] and the useful
    [(got [])] diagnostic. On an elementwise mismatch [?detail] behaves exactly as in {!pass_fail}.
*)
let pass_fail_all2 ?min ?detail label got want ~f =
  match all2_outcome ?min got want ~f with
  | All2_holds { failed = 0; _ } -> pass_fail ?detail label true
  | All2_holds { failed; first } ->
      pass_fail ?detail label false;
      report_all2_failure label ~cells:(Array.length got) ~failed ~first
  | All2_short reason ->
      let detail =
        match detail with
        | None -> fun () -> reason
        | Some caller_detail -> fun () -> reason ^ "; " ^ caller_detail ()
      in
      pass_fail ~detail label false

(** [pass_fail_close label got want ~tol] is {!p_close} in {!pass_fail}'s output dialect: a
    structural refusal is the [<label>: FAIL (<reason>)] row {!pass_fail_all2} prints, and a value
    failure the bare [<label>: FAIL] row, with {!p_close}'s worst-cell line beside it on stderr. *)
let pass_fail_close ?min ?what_ran label got want ~tol =
  match close_outcome ?min ~tol got want with
  | Close_short reason -> pass_fail ~detail:(fun () -> reason) label false
  | Close_holds { differing = 0; _ } -> pass_fail label true
  | Close_holds { differing; first; worst; magnitude; worst_tol } ->
      pass_fail label false;
      report_close_failure ?what_ran label got want ~differing ~first ~worst ~magnitude ~worst_tol

(** [case label f] runs one case of a test -- a group of claims that share a setup -- and turns an
    exception escaping it into a failed claim,
    [<label>: the case ran to completion (raised <exception>): false], so that the cases after it
    still run (gh-ocannl-1067). It prints nothing of its own when [f] returns, so wrapping a case is
    golden-neutral.

    Without it an exception ends the run at the case that raised. The teardown then says so
    ([STOPPED EARLY] on stderr, and the exception itself), but the rows after that case are never
    evaluated: a negative control that made one case raise learns nothing about the others. With it,
    each case is its own verdict, which is what a test of many independent rows wants.

    The exception's text goes on the claim line (echoed to stderr like every failure), and the
    backtrace, when one is recorded, to stderr. What an explicit [exit] raises is let through, not
    recorded: that is the process ending, not a case failing. Once a check has failed, that ending
    is reported as [STOPPED EARLY: an exit inside case "<label>" …] on stderr (gh-ocannl-1084): a
    run that ended inside a case never ran the cases after it, and a reader counting rows
    ([tools/mutation-run.sh]) cannot otherwise tell it from a run whose earlier case raised and
    whose later ones completed. *)
let case label f =
  match f () with
  | () -> ()
  | exception (Checks_failed as e) ->
      (* The teardown raises this out of an [exit] only, so the case it passes through is the one
         that called [exit]; the innermost case records it first. *)
      if Option.is_none !exited_in_case then exited_in_case := Some label;
      raise e
  | exception e ->
      let backtrace = Stdlib.Printexc.get_raw_backtrace () in
      (* [short_fail] puts the exception's text on the claim's one line, the line
         tools/mutation-run.sh reads; [verdict_teardown]'s golden pins it, and
         tools/test-mutation-run.sh replays it through the runner. *)
      let text = Stdlib.Printexc.to_string e in
      short_fail (label ^ ": the case ran to completion") ("raised " ^ text);
      if Stdlib.Printexc.raw_backtrace_length backtrace > 0 then (
        Stdlib.Printexc.print_raw_backtrace Stdlib.stderr backtrace;
        Stdlib.flush Stdlib.stderr)

(** Whether any check has failed so far. For a test that wants to say something extra about a bad
    run; the exit status is taken care of without it. *)
let any_failed () = not (List.is_empty !failed_names)

(** [success_trailer ?blank_line rows] prints each of [rows] as an [OK: <row>] line, behind an empty
    line when [blank_line], provided no check has failed so far -- and nothing otherwise: a scan's
    closing statement of what its claims established, which a failing run must not print. It is the
    one shape a success trailer takes. [tools/mutation-run.sh] matches a failing run of a scan
    against its golden without the trailer's [OK: ] rows and the empty line before them, and
    [tools/test-mutation-run.sh] replays [verdict_teardown]'s transcripts of runs that print through
    here, so a trailer reworded here turns that harness red instead of the runner reading every
    failing run of every scan as stopped early (gh-ocannl-1364). Each row prints on one line
    ({!one_line}). *)
let success_trailer ?(blank_line = false) rows =
  if not (any_failed ()) then (
    if blank_line then Stdio.print_string "\n";
    List.iter rows ~f:(fun row -> Stdio.printf "OK: %s\n" (one_line row)))

(** Ends a NEGATIVE-CONTROL run, one whose failures are the output its golden pins: exits 1 when
    something failed and 0 when nothing did, without the teardown's ["FAILED: n checks …"] line.
    That line totals the failure lines above it, and a total in a golden merges wrong -- two
    branches each adding a control merge the lines right and the count wrong (gh-ocannl-1056). The
    rule running such a control accepts exit 1 alone, so a control that stopped refusing still
    fails. *)
let exit_negative_control () =
  let failed = any_failed () in
  failed_names := [];
  Stdio.Out_channel.flush Stdio.stdout;
  Stdio.Out_channel.flush Stdio.stderr;
  Stdlib.exit (if failed then 1 else 0)

(** The claim surface intended for [open Verdict.Claims]. Keeping the open-oriented names in one
    module means a new combinator is exposed to every migrated test by changing this file alone; the
    top-level bindings above remain for compatibility with qualified call sites. State and
    scanner-control helpers deliberately stay outside this module. *)
module Claims = struct
  let fail = fail
  let claim = claim
  let p = p
  let pf = pf
  let claimf = claimf
  let p_all = p_all
  let p_all2 = p_all2
  let pf_all2 = pf_all2
  let p_close = p_close
  let pf_close = pf_close
  let p_none = p_none
  let p_alli = p_alli
  let p_empty = p_empty
  let p_exists = p_exists
  let p_pairwise_distinct = p_pairwise_distinct
  let skipped = skipped
  let gated = gated
  let gated_all = gated_all
  let gated_alli = gated_alli
  let gated_exists = gated_exists
  let pass_fail = pass_fail
  let pass_fail_all2 = pass_fail_all2
  let pass_fail_close = pass_fail_close
  let case = case
end

(* The teardown (gh-ocannl-1067). Every path out of the process runs [at_exit], so this is where a
   failed check turns into a nonzero status whatever the test's last line was. It must not do that
   by calling [exit] itself, which is what it used to do: the runtime runs [at_exit] BEFORE it
   prints an uncaught exception, so an [exit 1] from here ended the process before the exception was
   ever reported. A test whose later case raised showed its earlier [FAIL:] lines and nothing else,
   and every row after the raise silently never ran -- gh-ocannl-1016's negative control read such a
   run as "six rows failed, the rest passed". Nothing at this point can tell a normal end from an
   escaping exception, so the handler does not decide: it RAISES [Checks_failed], and the
   uncaught-exception handler below, which does see the exception, says what happened.

   At a normal end, or an explicit [exit], [Checks_failed] is the one uncaught exception, and the
   handler reports the failures and exits 1, as before. When an exception escapes the test, the
   runtime swallows the one raised from [at_exit] and hands the handler the ORIGINAL exception,
   which it reports as a run that stopped early and prints as the runtime would have, exiting with
   the runtime's status, 2.

   Raising out of [at_exit] skips the handlers registered before this module's, so the handler ends
   through [Stdlib.exit], which runs them, and [settling] keeps this teardown from raising again on
   that pass. Until then the teardown re-registers itself before it raises: an explicit [exit] now
   raises out of the call, and a test whose catch-all swallowed that would otherwise end on a
   consumed teardown and exit 0. *)
let settling = ref false

let rec teardown () =
  if any_failed () && not !settling then (
    Stdio.Out_channel.flush Stdio.stdout;
    Stdlib.at_exit teardown;
    raise Checks_failed)

(* How many failed checks the teardown line names, and the most of each name it keeps, in bytes. *)
let teardown_names_shown = 3
let teardown_name_bytes = 60

(* [name] on one physical line ({!one_line}: a computed label can carry a newline, and the teardown
   must stay the LAST line, the one [tools/mutation-run.sh] and a truncated log read), cut to
   {!teardown_name_bytes}, at a character boundary of its UTF-8 so a cut never splits one, and
   marked as cut. *)
let shorten name =
  let name = one_line name in
  if String.length name <= teardown_name_bytes then name
  else
    let rec boundary i =
      if i > 0 && Char.to_int name.[i] land 0xC0 = 0x80 then boundary (i - 1) else i
    in
    String.prefix name (boundary teardown_name_bytes) ^ "..."

(** The teardown line for the failed checks [names], in the order they failed, without its newline:
    [FAILED: <n> check(s) did not hold: "<first>", "<second>", "<third>" and <k> more.] It names the
    first {!teardown_names_shown} of them, each cut to {!teardown_name_bytes} bytes, so the line
    stays short however many failed and however long their labels are. The names are on this line,
    not only on the [FAIL:] lines above it, because dune cuts a long stderr down to its head and
    tail ([...TRUNCATED BY DUNE...]), and then this line is the one that survives.

    Exposed so that [census_total_ratchet], which refuses this line in scan goldens, and
    [verdict_teardown] build their controls from the text the teardown really writes. *)
let teardown_line names =
  let n = List.length names in
  let shown, rest = List.split_n names teardown_names_shown in
  let quoted =
    String.concat ~sep:", " (List.map shown ~f:(fun name -> "\"" ^ shorten name ^ "\""))
  in
  let more = match List.length rest with 0 -> "" | k -> Printf.sprintf " and %d more" k in
  Printf.sprintf "FAILED: %d check%s did not hold: %s%s." n (if n = 1 then "" else "s") quoted more

let report_failures () =
  Stdio.Out_channel.flush Stdio.stdout;
  Stdio.eprintf "%s\n" (teardown_line (List.rev !failed_names));
  Stdio.Out_channel.flush Stdio.stderr

let () =
  Stdlib.Printexc.set_uncaught_exception_handler (fun exn backtrace ->
      match exn with
      | Checks_failed ->
          settling := true;
          report_failures ();
          Option.iter !exited_in_case ~f:(fun label ->
              Stdio.eprintf
                "STOPPED EARLY: an exit inside case %S ended the run, so no case after it ran\n%!"
                label);
          Stdlib.exit 1
      | _ when any_failed () ->
          settling := true;
          report_failures ();
          Stdio.eprintf
            "STOPPED EARLY: an uncaught exception ended the run, so no check after it ran:\n%!";
          Stdlib.Printexc.default_uncaught_exception_handler exn backtrace;
          Stdlib.exit 2
      | _ -> Stdlib.Printexc.default_uncaught_exception_handler exn backtrace);
  Stdlib.at_exit teardown
