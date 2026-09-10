(* The non-emptiness guarantee of Verdict's quantified claims (gh-ocannl-729, gh-ocannl-746).

   "every X holds" is TRUE of an empty X. Written as `p "every seed spreads j" (List.for_all seeds
   ~f:...)` it prints the same passing line whether the property was checked on a hundred seeds or
   on none, and the golden records it as verified either way -- the gh-ocannl-601 hazard one level
   up, arriving through `Verdict.p` itself. `p_all`, `p_none`, `p_empty` and `p_exists` carry the
   guard, so the claim is the shortest thing to write AND cannot pass on nothing.

   `p_all2` (gh-ocannl-746) is the same guarantee one type over, on the executed-parity genre: `p L
   (Array.for_all2_exn got want ~f)`, "every cell matches the reference". `Array.for_all2_exn [||]
   [||]` is `true` too, and the emptiness is REACHABLE -- both sides usually come out of
   `Context.get_values`, and a node that stopped being materialized empties them at once, so the
   reference does not discriminate, because it went through the same path. It carries one refusal
   its siblings have no analogue of: a length mismatch is a failed CLAIM naming both lengths, not
   the `Invalid_argument` `Array.for_all2_exn` raises without saying which claim raised it.

   Every guarantee here fires only when a collection is empty, which in a green suite is never --
   the same shape as `generated_provenance`, and the same construction: the passing forms run
   directly, so a module that refused everything could not pass them, and the refusals run as CHILD
   processes whose streams this one captures. Capturing matters for the same reason it does there: a
   refusal prints this repository's failure marker (`FAIL:`, `FAILED:`), and a green run's log must
   not carry those words.

   Two properties, not one. The claim must FAIL on an empty collection -- exit status and a line
   naming emptiness -- and it must print BYTE-IDENTICALLY to `Verdict.p` when the collection is not
   empty, which is what lets ~90 test files convert without their goldens moving. The second is
   checked by running `p` beside each combinator in two children and comparing what each wrote. *)

open Base
open Verdict.Claims

let describe_status = function
  | Unix.WEXITED n -> Printf.sprintf "exited %d" n
  | Unix.WSIGNALED n -> Printf.sprintf "was killed by signal %d" n
  | Unix.WSTOPPED n -> Printf.sprintf "was stopped by signal %d" n

let ignore_unix f x = try f x with Unix.Unix_error _ -> ()

(* Runs one mode in a child and answers its status with its two streams kept APART: the shape check
   compares what a child wrote on stdout, which is the stream a `(test)` stanza diffs, and a
   refusal's stderr echo would drown that comparison. Through temporary files rather than pipes, for
   the reason `generated_provenance.run_child` gives: reading two pipes in sequence deadlocks once
   the unread one fills. *)
let run_child mode =
  let exe = Stdlib.Sys.executable_name in
  let capture suffix = Stdlib.Filename.temp_file "vq_child" suffix in
  let out_path = capture ".out" and err_path = capture ".err" in
  let open_capture p = Unix.openfile p [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
  let out = open_capture out_path and err = open_capture err_path in
  let pid = Unix.create_process exe [| exe; mode |] Unix.stdin out err in
  let _, status = Unix.waitpid [] pid in
  Unix.close out;
  Unix.close err;
  let stdout_text = Stdio.In_channel.read_all out_path in
  let stderr_text = Stdio.In_channel.read_all err_path in
  ignore_unix Unix.unlink out_path;
  ignore_unix Unix.unlink err_path;
  (status, stdout_text, stderr_text)

(* Whether a child refused the way the mode under test is about: exit 1, and [line] among what it
   printed on stdout -- the stream the golden of a converted test is made of, so it is where the
   distinct wording has to appear. A failing check prints the whole capture to stderr, because the
   child's own account is what a reader needs and it is only withheld from PASSING runs. *)
let refused claim ~line (status, stdout_text, stderr_text) =
  let ok =
    (match status with Unix.WEXITED 1 -> true | _ -> false)
    && String.is_substring stdout_text ~substring:line
  in
  if not ok then
    Stdio.eprintf "the child %s without printing %S on stdout. Its capture:\n%s%s\n"
      (describe_status status) line stdout_text stderr_text;
  Verdict.p claim ok

let seeds = [ 2; 4; 6 ]
let even n = n % 2 = 0
let odd n = n % 2 = 1

(* The executed-parity shape `p_all2` serves: a readback and the reference it is checked against. *)
let got = [| 1.0; 2.0; 3.0 |]
let want = [| 1.0; 2.0; 3.0 |]
let distinct = [ 2; 4; 6 ]

(* Every claim below is checked BYTE for byte, including its newline, so stdout has to be the bytes
   `Verdict` wrote rather than the bytes a platform decided they should become. OCaml's stdout is in
   TEXT mode on Windows, which rewrites each "\n" to "\r\n" on the way out; the three shape checks
   that compare a capture against a literal ("the claim: true\n") then failed there and only there,
   while the four that compare two captures against each other passed -- both children being
   rewritten identically. Normalising the comparison instead would have made those three checks pass
   without checking what the module actually prints, which is the whole question. Dune's `(diff)` is
   a TEXT diff, so putting this process's own golden output in binary changes nothing for the
   `.expected` on any platform. *)
let () = Stdlib.set_binary_mode_out Stdlib.stdout true

let () =
  let mode =
    match Array.to_list Stdlib.Sys.argv with
    | _ :: m :: _ when not (String.is_prefix m ~prefix:"--") -> m
    | _ -> "holds"
  in
  match mode with
  (* === Must succeed: run directly by the dune rule. === *)
  | "holds" ->
      Verdict.p_all "every seed is even" seeds ~f:even;
      Verdict.p_alli "every seed exceeds its index" seeds ~f:(fun i n -> n > i);
      Verdict.p_all ~min:3 "every one of the three seeds is even" seeds ~f:even;
      Verdict.p_none "no seed is odd" seeds ~f:odd;
      Verdict.p_exists "some seed exceeds four" seeds ~f:(fun n -> n > 4);
      Verdict.p_exists ~min:2 "at least two seeds exceed two" seeds ~f:(fun n -> n > 2);
      Verdict.p_empty "every seed validates" ~over:seeds (List.filter seeds ~f:odd);
      (* An array reaches the SINGLE-collection combinators through [Array.to_list]; the eight array
         sites in the gh-ocannl-729 sweep spell it that way rather than growing a second family of
         entry points. The pairwise one takes arrays, because it compares their lengths. *)
      Verdict.p_all "every sampled value is finite"
        (Array.to_list [| 1.0; 2.0 |])
        ~f:Float.is_finite;
      Verdict.p_all2 "the readback matches the reference cell for cell" got want ~f:Float.equal;
      Verdict.p_all2 ~min:3 "all three cells match the reference" got want ~f:Float.equal;
      pf_all2 "computed %s matches the reference cell for cell" "readback" got want ~f:Float.equal;
      pass_fail_all2 "the PASS/FAIL readback matches the reference" got want ~f:Float.equal;
      p_pairwise_distinct "the three source values are pairwise distinct" distinct ~equal:Int.equal
        ~to_string:Int.to_string
  (* === Shape: what a non-empty collection prints, compared against [p]'s own line. === *)
  | "shape_p" -> Verdict.p "the claim" true
  | "shape_p_all" -> Verdict.p_all "the claim" seeds ~f:even
  | "shape_p_all2" -> Verdict.p_all2 "the claim" got want ~f:Float.equal
  | "shape_pf_all2" -> Verdict.pf_all2 "%s" "the claim" got want ~f:Float.equal
  | "shape_pass_fail" -> Verdict.pass_fail "the claim" true
  | "shape_pass_fail_all2" -> Verdict.pass_fail_all2 "the claim" got want ~f:Float.equal
  | "shape_p_pairwise_distinct" ->
      p_pairwise_distinct "the claim" distinct ~equal:Int.equal ~to_string:Int.to_string
  | "shape_p_false" -> Verdict.p "the claim" false
  | "shape_p_all_false" -> Verdict.p_all "the claim" seeds ~f:odd
  | "shape_p_all2_false" -> Verdict.p_all2 "the claim" got [| 1.0; 9.0; 3.0 |] ~f:Float.equal
  (* === Must be refused: run as children by [refusals]. === *)
  | "all_empty" -> Verdict.p_all "every seed is even" [] ~f:even
  | "all_short" -> Verdict.p_all ~min:3 "every one of the three seeds is even" [ 2 ] ~f:even
  | "none_empty" -> Verdict.p_none "no seed is odd" [] ~f:odd
  | "alli_empty" -> Verdict.p_alli "every seed exceeds its index" [] ~f:(fun i n -> n > i)
  | "exists_empty" -> Verdict.p_exists "some seed exceeds four" [] ~f:(fun n -> n > 4)
  | "exists_short" ->
      Verdict.p_exists ~min:2 "at least two seeds exceed four" seeds ~f:(fun n -> n > 4)
  | "exists_none" -> Verdict.p "the claim" (List.exists seeds ~f:odd)
  | "exists_none_combinator" -> Verdict.p_exists "the claim" seeds ~f:odd
  | "empty_over_empty" -> Verdict.p_empty "every seed validates" ~over:[] []
  (* gh-ocannl-746: the pairwise refusals. A de-materialized node empties BOTH sides at once, so
     "both empty" is the shape the executed-parity claims actually go vacuous in. *)
  | "all2_empty" ->
      Verdict.p_all2 "the readback matches the reference cell for cell" [||] [||] ~f:Float.equal
  | "all2_length" ->
      Verdict.p_all2 "the readback matches the reference cell for cell" got [| 1.0; 2.0 |]
        ~f:Float.equal
  | "all2_length_left_empty" ->
      Verdict.p_all2 "the readback matches the reference cell for cell" [||] want ~f:Float.equal
  | "all2_short" ->
      Verdict.p_all2 ~min:3 "all three cells match the reference" [| 1.0 |] [| 1.0 |] ~f:Float.equal
  | "pf_all2_empty" ->
      Verdict.pf_all2 "computed %s matches the reference" "readback" [||] [||] ~f:Float.equal
  | "pass_fail_all2_empty_detail" ->
      Verdict.pass_fail_all2 "the readback matches the reference" [||] [||] ~f:Float.equal
        ~detail:(fun () -> "got []")
  | "pass_fail_all2_length" ->
      Verdict.pass_fail_all2 "the readback matches the reference" got [| 1.0; 2.0 |] ~f:Float.equal
  | "pairwise_empty" ->
      p_pairwise_distinct "the source values are pairwise distinct" [] ~equal:Int.equal
        ~to_string:Int.to_string
  | "pairwise_singleton" ->
      p_pairwise_distinct "the source values are pairwise distinct" [ 7 ] ~equal:Int.equal
        ~to_string:Int.to_string
  | "pairwise_collision" ->
      p_pairwise_distinct "the source values are pairwise distinct" [ 7; 11; 7 ] ~equal:Int.equal
        ~to_string:Int.to_string
  | "refusals" ->
      refused "an `every` claim over an empty collection fails rather than passing vacuously"
        ~line:"every seed is even (empty): false" (run_child "all_empty");
      refused "a collection below its stated floor fails, naming the shortfall"
        ~line:"every one of the three seeds is even (only 1 of 3): false" (run_child "all_short");
      refused "a `no X is` claim over an empty collection fails rather than passing vacuously"
        ~line:"no seed is odd (empty): false" (run_child "none_empty");
      refused
        "an indexed `every` claim over an empty collection fails rather than passing vacuously"
        ~line:"every seed exceeds its index (empty): false" (run_child "alli_empty");
      refused "a `some X` claim over an empty collection names emptiness rather than the property"
        ~line:"some seed exceeds four (empty): false" (run_child "exists_empty");
      (* Codex P2, round 1: `~min` on an existential counts WITNESSES. Read as a population floor it
         would pass "at least two seeds exceed four" on a three-seed list with one such seed, which
         is the false green this module exists to prevent. *)
      refused "a `~min:n` existential counts witnesses, not the population it searched"
        ~line:"at least two seeds exceed four (only 1 of 2 match): false" (run_child "exists_short");
      refused "an emptiness claim about a derived subset fails when the population is empty too"
        ~line:"every seed validates (empty): false" (run_child "empty_over_empty");
      refused
        "an executed-parity claim over two empty readbacks fails rather than passing vacuously"
        ~line:"the readback matches the reference cell for cell (empty): false"
        (run_child "all2_empty");
      (* A mismatch is a CLAIM line, not the `Invalid_argument` `Array.for_all2_exn` raises: an
         exception fails the run without naming which claim, and the two lengths are the finding. *)
      refused "a length mismatch is reported as a failed claim naming both lengths"
        ~line:"the readback matches the reference cell for cell (length 3 vs 2): false"
        (run_child "all2_length");
      refused "a one-sided empty readback reports the mismatch rather than plain emptiness"
        ~line:"the readback matches the reference cell for cell (length 0 vs 3): false"
        (run_child "all2_length_left_empty");
      refused "a pair below its stated floor fails, naming the shortfall"
        ~line:"all three cells match the reference (only 1 of 3): false" (run_child "all2_short");
      refused "a computed-label executed-parity claim refuses two empty readbacks"
        ~line:"computed readback matches the reference (empty): false" (run_child "pf_all2_empty");
      refused "a PASS/FAIL parity claim keeps its empty reason and caller detail"
        ~line:"the readback matches the reference: FAIL (empty; got [])"
        (run_child "pass_fail_all2_empty_detail");
      refused "a PASS/FAIL parity claim reports a length mismatch without raising"
        ~line:"the readback matches the reference: FAIL (length 3 vs 2)"
        (run_child "pass_fail_all2_length");
      refused "pairwise distinctness refuses an empty source population"
        ~line:"the source values are pairwise distinct (empty): false" (run_child "pairwise_empty");
      refused "pairwise distinctness requires at least two source values"
        ~line:"the source values are pairwise distinct (only 1 of 2): false"
        (run_child "pairwise_singleton");
      refused "pairwise distinctness reports the first colliding pair directly"
        ~line:"the source values are pairwise distinct (collision: 7 = 7): false"
        (run_child "pairwise_collision");
      (* The conversion is golden-neutral exactly to the extent that this holds. *)
      let _, plain_true, _ = run_child "shape_p" in
      let _, all_true, _ = run_child "shape_p_all" in
      Verdict.p "a satisfied quantified claim prints what `Verdict.p` prints"
        (String.equal plain_true all_true && String.equal plain_true "the claim: true\n");
      let _, all2_true, _ = run_child "shape_p_all2" in
      Verdict.p "a satisfied executed-parity claim prints what `Verdict.p` prints"
        (String.equal plain_true all2_true);
      let _, pf_all2_true, _ = run_child "shape_pf_all2" in
      Verdict.p "a satisfied computed-label parity claim prints what `Verdict.p` prints"
        (String.equal plain_true pf_all2_true);
      let _, pass_fail_true, _ = run_child "shape_pass_fail" in
      let _, pass_fail_all2_true, _ = run_child "shape_pass_fail_all2" in
      Verdict.p "a satisfied PASS/FAIL parity claim prints what `Verdict.pass_fail` prints"
        (String.equal pass_fail_true pass_fail_all2_true);
      let _, pairwise_true, _ = run_child "shape_p_pairwise_distinct" in
      Verdict.p "a satisfied pairwise-distinct claim prints what `Verdict.p` prints"
        (String.equal plain_true pairwise_true);
      let _, plain_false, _ = run_child "shape_p_false" in
      let _, all_false, _ = run_child "shape_p_all_false" in
      Verdict.p "a non-empty collection that refutes the claim prints what `Verdict.p` prints"
        (String.equal plain_false all_false && String.equal plain_false "the claim: false\n");
      (* The elementwise refutation — the failure the sweep must keep catching — still prints the
         bare line: the `(…)` detail is reserved for what the claim could not express. *)
      let _, all2_false, _ = run_child "shape_p_all2_false" in
      Verdict.p "a genuinely differing cell refutes the claim and prints what `Verdict.p` prints"
        (String.equal plain_false all2_false);
      (* At the default floor there is no shortfall to name, so an unwitnessed existential prints
         the bare line too -- the `(…)` detail is reserved for what the claim could not express. *)
      let _, plain_none, _ = run_child "exists_none" in
      let _, exists_none, _ = run_child "exists_none_combinator" in
      Verdict.p "an unwitnessed existential at the default floor prints what `Verdict.p` prints"
        (String.equal plain_none exists_none && String.equal plain_none "the claim: false\n")
  | other -> failwith ("verdict_quantified: unknown mode " ^ other)
