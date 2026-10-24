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
   checked by running `p` beside each combinator in two children and comparing what each wrote.

   `Verdict.gated` (gh-ocannl-997) rides the same harness for the same second property, one gate
   over: skipped or evaluated, a passing gated claim prints `p`'s line, which is the whole reason it
   exists.

   `p_close` (gh-ocannl-1224) is `p_all2` specialized to float readbacks under a tolerance, and
   rides it for all three properties, plus a fourth: what a VALUE failure adds on stderr -- the
   differing-cell count, the first and the worst cell with both values, and the reference's
   magnitude -- while its stdout stays `p`'s bare `false` line. `p_all2`'s own value failure adds a
   count and a first index there, which is all a predicate over an unknown type can name. *)

open Base
open Verdict.Claims

let run_child mode = Fresh_process.run [ mode ]

(* Refusal needs both the status and the causal stdout line. *)
let refused claim ~line child =
  let ok = Fresh_process.matches ~stream:`Stdout ~exit:1 ~contains:[ line ] child in
  if not ok then Fresh_process.report ~label:claim child;
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
   `.expected` on any platform. The same holds for stderr: the value-failure details
   (gh-ocannl-1224) are checked there byte for byte too, so a child's "\n" must reach the parent as
   "\n" on that stream as well. *)
let () =
  Stdlib.set_binary_mode_out Stdlib.stdout true;
  Stdlib.set_binary_mode_out Stdlib.stderr true

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
      p_close "the readback is within an absolute tolerance of the reference" got
        [| 1.25; 2.0; 2.75 |] ~tol:(fun _ -> 0.25);
      p_close "the readback is within a relative tolerance of the reference" [| 100.; 2.0 |]
        [| 101.; 2.0 |] ~tol:(fun w -> 0.01 *. Float.abs w);
      p_close "a zero tolerance admits exactly the equal cells, infinities included"
        [| 1.0; Float.infinity; Float.neg_infinity |] [| 1.0; Float.infinity; Float.neg_infinity |]
        ~tol:(fun _ -> 0.);
      pf_close "computed %s is within tolerance of the reference" "readback" got want ~tol:(fun _ ->
          0.);
      pass_fail_close "the PASS/FAIL readback is within tolerance of the reference" got want
        ~tol:(fun _ -> 0.);
      p_pairwise_distinct "the three source values are pairwise distinct" distinct ~equal:Int.equal
        ~to_string:Int.to_string
  (* === Shape: what a non-empty collection prints, compared against [p]'s own line. === *)
  | "shape_p" -> Verdict.p "the claim" true
  | "shape_p_all" -> Verdict.p_all "the claim" seeds ~f:even
  | "shape_p_all2" -> Verdict.p_all2 "the claim" got want ~f:Float.equal
  | "shape_pf_all2" -> Verdict.pf_all2 "%s" "the claim" got want ~f:Float.equal
  | "shape_pass_fail" -> Verdict.pass_fail "the claim" true
  | "shape_pass_fail_all2" -> Verdict.pass_fail_all2 "the claim" got want ~f:Float.equal
  | "shape_p_close" -> p_close "the claim" got want ~tol:(fun _ -> 0.)
  | "shape_pf_close" -> pf_close "%s" "the claim" got want ~tol:(fun _ -> 0.)
  | "shape_pass_fail_close" -> pass_fail_close "the claim" got want ~tol:(fun _ -> 0.)
  (* gh-ocannl-1224: two of four cells differ beyond 0.25; the second-differing one, at index 3, is
     the furthest from its reference. *)
  | "shape_p_close_false" ->
      p_close ~what_ran:"seed 3" "the claim" [| 1.0; 2.5; 3.0; 4.0 |] [| 1.0; 2.0; 3.0; 8.0 |]
        ~tol:(fun _ -> 0.25)
  | "shape_pass_fail_close_false" ->
      pass_fail_close "the claim" [| 1.0; 2.5; 3.0; 4.0 |] [| 1.0; 2.0; 3.0; 8.0 |] ~tol:(fun _ ->
          0.25)
  (* A NaN outranks a finite distance however large, and the relative tolerance is evaluated at the
     worst cell's reference. *)
  | "close_nan_worst" ->
      p_close "the claim" [| 1.0; Float.nan; 10.0 |] [| 1.0; 2.0; 3.0 |] ~tol:(fun w -> 0.5 *. w)
  (* An infinite distance before a NaN: the NaN is still the worst. *)
  | "close_nan_after_infinity" ->
      p_close "the claim" [| Float.infinity; Float.nan |] [| 1.0; 2.0 |] ~tol:(fun _ -> 0.5)
  (* An infinite reference under a relative tolerance: the tolerance is infinite there, and still
     admits neither a finite value nor the opposite infinity. *)
  | "close_infinite_reference" ->
      p_close "the claim" [| 1.0; Float.neg_infinity |] [| Float.infinity; Float.infinity |]
        ~tol:(fun w -> 1e-5 *. Float.abs w)
  | "shape_p_pairwise_distinct" ->
      p_pairwise_distinct "the claim" distinct ~equal:Int.equal ~to_string:Int.to_string
  | "shape_p_false" -> Verdict.p "the claim" false
  (* gh-ocannl-1160: opposing predicates have the same visits, including after the first refutation.
     The indexed leg records both the supplied index and element. Keep diagnostics on stderr so each
     child's stdout can still be compared byte-for-byte with [p]. *)
  | ( "visits_all_true" | "visits_all_false" | "visits_none_true" | "visits_none_false"
    | "visits_alli_true" | "visits_alli_false" ) as mode ->
      let holds = String.is_suffix mode ~suffix:"_true" in
      let visit i n =
        Stdio.eprintf "visit %d=%d\n%!" i n;
        holds || n <> 2
      in
      if String.is_prefix mode ~prefix:"visits_alli" then Verdict.p_alli "the claim" seeds ~f:visit
      else if String.is_prefix mode ~prefix:"visits_none" then
        Verdict.p_none "the claim" seeds ~f:(fun n -> not (visit 0 n))
      else Verdict.p_all "the claim" seeds ~f:(visit 0)
  (* gh-ocannl-997: a gated claim is [p] where its gate is open and a skip where it is closed, and
     both must print [p]'s passing line. The closed gate is handed a FALSE value and a detail that
     raises, so a skip that evaluated either would show. Children only: a skip announces itself on
     stderr with a sweep record, which this suite's own log must not carry. *)
  | "gated_open" -> gated ~when_:true ~on:"a fixture backend" "the gated claim" true
  | "gated_closed" ->
      gated ~when_:false ~on:"a fixture backend"
        ~detail:(fun () -> failwith "detail evaluated")
        "the gated claim" false
  | "gated_false" ->
      gated ~when_:true ~on:"a fixture backend"
        ~detail:(fun () -> "measured 3 vs bound 2")
        "the gated claim" false
  (* The quantified gated forms: open, each prints its combinator's line; closed, each is handed an
     EMPTY population and a predicate that raises, so a skip that read either would show. *)
  | "gated_all_open" ->
      gated_all ~when_:true ~on:"a fixture backend" "the gated claim" seeds ~f:even
  | "gated_alli_open" ->
      gated_alli ~when_:true ~on:"a fixture backend" "the gated claim" seeds ~f:(fun i n -> n > i)
  | "gated_exists_open" ->
      gated_exists ~when_:true ~on:"a fixture backend" "the gated claim" seeds ~f:(fun n -> n > 4)
  | "gated_all_closed" ->
      gated_all ~when_:false ~on:"a fixture backend" "the gated claim" [] ~f:(fun _ ->
          failwith "predicate evaluated")
  | "gated_alli_closed" ->
      gated_alli ~when_:false ~on:"a fixture backend" "the gated claim" [] ~f:(fun _ _ ->
          failwith "predicate evaluated")
  | "gated_exists_closed" ->
      gated_exists ~when_:false ~on:"a fixture backend" "the gated claim" [] ~f:(fun _ ->
          failwith "predicate evaluated")
  | "gated_all_empty" -> gated_all ~when_:true ~on:"a fixture backend" "the gated claim" [] ~f:even
  | "gated_alli_empty" ->
      gated_alli ~when_:true ~on:"a fixture backend" "the gated claim" [] ~f:(fun i n -> n > i)
  | "gated_exists_empty" ->
      gated_exists ~when_:true ~on:"a fixture backend" "the gated claim" [] ~f:(fun n -> n > 4)
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
  | "close_empty" ->
      p_close "the readback is within tolerance of the reference" [||] [||] ~tol:(fun _ -> 1.)
  | "close_length" ->
      p_close "the readback is within tolerance of the reference" got [| 1.0; 2.0 |] ~tol:(fun _ ->
          1.)
  | "close_short" ->
      p_close ~min:3 "all three cells are within tolerance" [| 1.0 |] [| 1.0 |] ~tol:(fun _ -> 1.)
  | "pf_close_empty" ->
      pf_close "computed %s is within tolerance" "readback" [||] [||] ~tol:(fun _ -> 1.)
  | "pass_fail_close_length" ->
      pass_fail_close "the readback is within tolerance" got [| 1.0; 2.0 |] ~tol:(fun _ -> 1.)
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
      List.iter [ "all"; "none"; "alli" ] ~f:(fun quantifier ->
          List.iter [ true; false ] ~f:(fun holds ->
              let mode = Printf.sprintf "visits_%s_%b" quantifier holds in
              let status, stdout, stderr = run_child mode in
              let visits =
                String.split_lines stderr
                |> List.filter ~f:(fun line -> String.is_prefix line ~prefix:"visit ")
              in
              let expected_visits =
                if String.equal quantifier "alli" then [ "visit 0=2"; "visit 1=4"; "visit 2=6" ]
                else [ "visit 0=2"; "visit 0=4"; "visit 0=6" ]
              in
              Verdict.claimf "%s evaluates every element in order and preserves its verdict" mode
                (Poly.equal status (Unix.WEXITED (if holds then 0 else 1))
                && String.equal stdout (if holds then "the claim: true\n" else "the claim: false\n")
                && List.equal String.equal visits expected_visits)));
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
      refused "a float-parity claim over two empty readbacks fails rather than passing vacuously"
        ~line:"the readback is within tolerance of the reference (empty): false"
        (run_child "close_empty");
      refused "a float-parity claim reports a length mismatch as a failed claim naming both lengths"
        ~line:"the readback is within tolerance of the reference (length 3 vs 2): false"
        (run_child "close_length");
      refused "a float-parity pair below its stated floor fails, naming the shortfall"
        ~line:"all three cells are within tolerance (only 1 of 3): false" (run_child "close_short");
      refused "a computed-label float-parity claim refuses two empty readbacks"
        ~line:"computed readback is within tolerance (empty): false" (run_child "pf_close_empty");
      refused "a PASS/FAIL float-parity claim reports a length mismatch without raising"
        ~line:"the readback is within tolerance: FAIL (length 3 vs 2)"
        (run_child "pass_fail_close_length");
      refused "pairwise distinctness refuses an empty source population"
        ~line:"the source values are pairwise distinct (empty): false" (run_child "pairwise_empty");
      refused "pairwise distinctness requires at least two source values"
        ~line:"the source values are pairwise distinct (only 1 of 2): false"
        (run_child "pairwise_singleton");
      refused "pairwise distinctness reports the first colliding pair directly"
        ~line:"the source values are pairwise distinct (collision: 7 = 7): false"
        (run_child "pairwise_collision");
      refused "an evaluated gated claim that fails names its detail in p's dialect"
        ~line:"the gated claim (measured 3 vs bound 2): false" (run_child "gated_false");
      (* The trap gh-ocannl-997 removes: a skip must print what an evaluated pass prints, so the
         golden is one file whichever side of the gate a host lands on. *)
      let open_status, gated_open, _ = run_child "gated_open" in
      let closed_status, gated_closed, closed_stderr = run_child "gated_closed" in
      Verdict.p "a gated claim prints the same passing line whether evaluated or skipped"
        (String.equal gated_open "the gated claim: true\n" && String.equal gated_open gated_closed);
      Verdict.p "a closed gate exits 0 without evaluating the value or the detail"
        (Poly.equal open_status (Unix.WEXITED 0)
        && Poly.equal closed_status (Unix.WEXITED 0)
        && not (String.is_substring closed_stderr ~substring:"detail evaluated"));
      Verdict.p "a closed gate announces the skip on stderr, naming what it ran on"
        (String.is_substring closed_stderr
           ~substring:"SKIPPED on a fixture backend (vacuous): the gated claim");
      (* The quantified gated forms keep both properties, and an open gate keeps the floor. *)
      List.iter [ "gated_all"; "gated_alli"; "gated_exists" ] ~f:(fun form ->
          let open_status, opened, _ = run_child (form ^ "_open") in
          let closed_status, closed, closed_stderr = run_child (form ^ "_closed") in
          Verdict.claimf "%s prints gated's passing line whether evaluated or skipped" form
            (String.equal opened gated_open && String.equal closed gated_open);
          Verdict.claimf
            "%s's closed gate exits 0, announcing the skip, without reading its population" form
            (Poly.equal open_status (Unix.WEXITED 0)
            && Poly.equal closed_status (Unix.WEXITED 0)
            && String.is_substring closed_stderr
                 ~substring:"SKIPPED on a fixture backend (vacuous): the gated claim"
            && not (String.is_substring closed_stderr ~substring:"predicate evaluated"));
          refused
            (Printf.sprintf "%s refuses an empty population where its gate is open" form)
            ~line:"the gated claim (empty): false"
            (run_child (form ^ "_empty")));
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
      let _, close_true, _ = run_child "shape_p_close" in
      let _, pf_close_true, _ = run_child "shape_pf_close" in
      let _, pass_fail_close_true, _ = run_child "shape_pass_fail_close" in
      Verdict.p
        "a satisfied float-parity claim prints what `Verdict.p` prints, in either label form"
        (String.equal plain_true close_true && String.equal plain_true pf_close_true);
      Verdict.p "a satisfied PASS/FAIL float-parity claim prints what `Verdict.pass_fail` prints"
        (String.equal pass_fail_true pass_fail_close_true);
      let _, pairwise_true, _ = run_child "shape_p_pairwise_distinct" in
      Verdict.p "a satisfied pairwise-distinct claim prints what `Verdict.p` prints"
        (String.equal plain_true pairwise_true);
      let _, plain_false, _ = run_child "shape_p_false" in
      let _, all_false, _ = run_child "shape_p_all_false" in
      Verdict.p "a non-empty collection that refutes the claim prints what `Verdict.p` prints"
        (String.equal plain_false all_false && String.equal plain_false "the claim: false\n");
      (* The elementwise refutation — the failure the sweep must keep catching — still prints the
         bare line: the `(…)` detail is reserved for what the claim could not express. *)
      let all2_status, all2_false, all2_stderr = run_child "shape_p_all2_false" in
      Verdict.p "a genuinely differing cell refutes the claim and prints what `Verdict.p` prints"
        (String.equal plain_false all2_false);
      (* gh-ocannl-1224: the count and position go to stderr, outside the golden, and the run still
         exits 1 on the claim. *)
      Verdict.p
        "a parity claim's value failure names its failing-cell count and first cell on stderr"
        (Poly.equal all2_status (Unix.WEXITED 1)
        && String.is_substring all2_stderr
             ~substring:
               "the claim: 1 of 3 cells fail the predicate; first at 1 (not part of the golden)\n");
      let close_status, close_false, close_stderr = run_child "shape_p_close_false" in
      Verdict.p "a float-parity value failure prints what `Verdict.p` prints on stdout"
        (Poly.equal close_status (Unix.WEXITED 1) && String.equal plain_false close_false);
      Verdict.p
        "a float-parity value failure names on stderr the differing count, the first and the worst \
         cell with both values and the tolerance there, the reference's magnitude and what ran"
        (String.is_substring close_stderr
           ~substring:
             "the claim (seed 3): 2 of 4 cells differ beyond tolerance; first at 1, worst at 3: \
              got 4 vs want 8 (|diff| 4, tolerance 0.25 there); max |want| 8 (not part of the \
              golden)\n");
      Verdict.p_none
        "the detail line is not a `FAIL: ` row, so a mutation run counts only the claim"
        (String.split_lines close_stderr) ~f:(fun line ->
          String.is_prefix line ~prefix:"FAIL: "
          && String.is_substring line ~substring:"differ beyond tolerance");
      let _, pass_fail_close_false, pass_fail_close_stderr =
        run_child "shape_pass_fail_close_false"
      in
      Verdict.p
        "a PASS/FAIL float-parity value failure keeps the bare FAIL row and the stderr detail"
        (String.equal pass_fail_close_false "the claim: FAIL\n"
        && String.is_substring pass_fail_close_stderr
             ~substring:"the claim: 2 of 4 cells differ beyond tolerance; first at 1, worst at 3:");
      let _, _, nan_stderr = run_child "close_nan_worst" in
      Verdict.p "a NaN cell ranks as the worst, ahead of any finite distance"
        (String.is_substring nan_stderr
           ~substring:"the claim: 2 of 3 cells differ beyond tolerance; first at 1, worst at 1:");
      let _, _, nan_after_inf_stderr = run_child "close_nan_after_infinity" in
      Verdict.p "a NaN cell ranks as the worst, ahead of an infinite distance before it"
        (String.is_substring nan_after_inf_stderr
           ~substring:"the claim: 2 of 2 cells differ beyond tolerance; first at 0, worst at 1:");
      let infinite_status, _, infinite_stderr = run_child "close_infinite_reference" in
      Verdict.p
        "an infinite reference admits neither a finite value nor the opposite infinity under an \
         infinite relative tolerance"
        (Poly.equal infinite_status (Unix.WEXITED 1)
        && String.is_substring infinite_stderr ~substring:"the claim: 2 of 2 cells differ");
      (* At the default floor there is no shortfall to name, so an unwitnessed existential prints
         the bare line too -- the `(…)` detail is reserved for what the claim could not express. *)
      let _, plain_none, _ = run_child "exists_none" in
      let _, exists_none, _ = run_child "exists_none_combinator" in
      Verdict.p "an unwitnessed existential at the default floor prints what `Verdict.p` prints"
        (String.equal plain_none exists_none && String.equal plain_none "the claim: false\n")
  | other -> failwith ("verdict_quantified: unknown mode " ^ other)
