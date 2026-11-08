(* gh-ocannl-1729: the exact-partials reader on synthetic sources -- each rule on the violation it
   must flag AND on the nearest text it must not, since the tree it polices is mostly clean once its
   debt is in rows, and a blind reader would look the same as an intact one. The reader's boundary
   is in test/support/exact_partials_scan.ml's header; these cases pin it, gaps included. *)
open Base
open Verdict.Claims
module Scan = Test_scan.Exact_partials_scan

let read source = Scan.read ~prose:(Prose_lexemes.of_source source) source

let sentences source =
  List.map (Scan.claims (Prose_lexemes.of_source source)) ~f:(fun c -> c.sentence)

(* How many claims [source]'s prose states: a flagged snippet states exactly one, so a reader that
   split one sentence in two, or read a comment twice, fails as surely as a blind one. *)
let claim_count source = List.length (sentences source)
let covered source = List.length (read source).covered
let uncovered source = List.length (read source).uncovered

let () =
  (* --- The phrase. --- *)
  p_all "a sentence holding [partial sum(s)] and [exact] or [exactly] is a claim"
    [
      "(* every partial sum is exact in f32 *)";
      "(* Every product and partial sum is exactly representable. *)";
      "(* the PARTIAL SUMS are f32-exact *)";
      "(* exact partial sums, so any order agrees *)";
      "(* every partial-sum is exact *)";
      "let label = \"every partial sum is exact in bf16\"";
      "(** every partial sum is exact *)\nlet x = 1";
      "let s = {|every partial sum is exact|}";
      "let s = {%ext|every partial sum is exact|}";
    ] ~f:(fun source -> claim_count source = 1);
  p_all "its nearest neighbours are not: a negation, another word, a sentence apart"
    [
      "(* every partial sum rounds *)";
      "(* the cells are exact while their partial sums are not *)";
      "(* inexact partial sums, so the claim is approximate *)";
      "(* Ll_test.exact_partials accepts every partial sum here *)";
      "(* the exactness of every partial sum *)";
      "(* partial sums are computed here. They are exact. *)";
      "(* partial sums are computed here\n\n   and exact *)";
      "(* the partial products are exact *)";
    ] ~f:(fun source -> claim_count source = 0);
  p "a known gap, pinned: bare [partials] is not the phrase"
    (claim_count "(* exact partials, so any association agrees *)" = 0);
  p "a decimal point ends no sentence: the phrase and [exact] around 0.5 are one claim"
    (claim_count "(* the partial sums of 0.5 and 0.25 are exact *)" = 1);
  p "the claim is reported whitespace-collapsed, one per sentence"
    (List.equal String.equal
       (sentences
          "(* First, every partial sum\n   is exact. Then every partial sum is exactly 2. *)")
       [ "First, every partial sum is exact."; "Then every partial sum is exactly 2." ]);
  (* --- The lexer. --- *)
  p "a comment holding a string with a comment's closing delimiter is one comment"
    (claim_count "(* \"*)\" every partial sum is exact *)\nlet x = 1" = 1);
  p "a character literal of a double quote opens no string"
    (claim_count "let c = '\"' (* every partial sum is exact *)" = 1);
  p "a nested comment is read with its parent"
    (claim_count "(* outer (* inner *) every partial sum is exact *)" = 1);
  p "a string literal is read decoded: an escape spelling [exact] is the word"
    (claim_count "let s = \"every partial sum is ex\\x61ct\"" = 1);
  p "code is not prose: an identifier spelling the words is not a claim"
    (claim_count "let partial_sum_is_exact = true" = 0);
  p "a source that does not lex is refused as the input's fault, with its line"
    (match Scan.attempt ~source:"t.ml" "let s = \"unterminated" with
    | Error refusal ->
        String.is_prefix refusal ~prefix:"t.ml does not lex or parse: "
        && String.is_substring refusal ~substring:"line 1"
    | Ok _ -> false);
  p "and so is one that lexes but does not parse"
    (match Scan.attempt ~source:"t.ml" "(* every partial sum is exact *) let let" with
    | Error refusal -> String.is_prefix refusal ~prefix:"t.ml does not lex or parse: "
    | Ok _ -> false);
  (* --- Beside. --- *)
  let derive = "let () = Verdict.p \"d\" (Ll_test.exact_partials ~sig_bits:24 [ 1. ])" in
  p "a comment inside the item that derives the premise is covered"
    (covered
       "let () =\n\
       \  (* every partial sum is exact in f32 *)\n\
       \  Verdict.p \"d\" (Ll_test.exact_partials ~sig_bits:24 [ 1. ])"
    = 1);
  p "a comment leading the item that derives it is covered"
    (covered ("(* every partial sum is exact *)\n" ^ derive) = 1);
  p "a claim label in the deriving item is covered"
    (covered
       "let () =\n\
       \  Verdict.Claims.p_all \"every partial sum is exact\" [ [ 1. ] ]\n\
       \    ~f:(Ll_test.exact_partials ~sig_bits:24)"
    = 1);
  p "a file header ahead of an unrelated item is uncovered, though a later item derives"
    (uncovered ("(* every partial sum is exact *)\nopen Base\n" ^ derive) = 1);
  p "a derivation named only in prose covers nothing"
    (uncovered "(* every partial sum is exact, see [Ll_test.exact_partials] *)\nlet x = 1" = 1);
  p "a bare exact_partials after an open is not the path, so it covers nothing"
    (uncovered
       "open Ll_test\n\
        (* every partial sum is exact *)\n\
        let () = Verdict.p \"d\" (exact_partials ~sig_bits:24 [ 1. ])"
    = 1);
  p "nor does a module alias's"
    (uncovered
       "module L = Ll_test\n\
        (* every partial sum is exact *)\n\
        let () = Verdict.p \"d\" (L.exact_partials ~sig_bits:24 [ 1. ])"
    = 1);
  p "a trailing comment after the last item has no item to derive it"
    (uncovered (derive ^ "\n(* every partial sum is exact *)\n") = 1);
  p "the reader counts the items that derive" ((read (derive ^ "\n" ^ derive)).deriving_items = 2);
  (* --- The ratchet. --- *)
  let reading source = read source in
  let one = reading "(* every partial sum is exact *)\nlet x = 1" in
  let two =
    reading
      "(* every partial sum is exact *)\nlet x = 1\n(* every partial sum is exact *)\nlet y = 2"
  in
  let clean = reading "let x = 1" in
  let row path allowance = { Scan.path; allowance; reason = "control" } in
  let findings ~exemptions rows = fst (Scan.violations ~exemptions rows) in
  let stale ~exemptions rows = snd (Scan.violations ~exemptions rows) in
  p "an uncovered sentence in a file without a row is a finding, one per sentence"
    (List.length (findings ~exemptions:[] [ ("t.ml", two) ]) = 2);
  p_empty "a row holding the file's count exactly absorbs it" ~over:two.uncovered
    (findings ~exemptions:[ row "t.ml" (Uncovered 2) ] [ ("t.ml", two) ]);
  p "a row holding another count is a finding: a new sentence, or a conversion the row missed"
    (List.length (findings ~exemptions:[ row "t.ml" (Uncovered 1) ] [ ("t.ml", two) ]) = 1
    && List.length (findings ~exemptions:[ row "t.ml" (Uncovered 2) ] [ ("t.ml", one) ]) = 1);
  p_empty "an Every row absorbs any count" ~over:two.uncovered
    (findings ~exemptions:[ row "t.ml" Every ] [ ("t.ml", two) ]);
  p "a row whose file holds no uncovered sentence is stale, and so is a row naming no source"
    (List.length (stale ~exemptions:[ row "t.ml" (Uncovered 1) ] [ ("t.ml", clean) ]) = 1
    && List.length (stale ~exemptions:[ row "t.ml" Every ] [ ("t.ml", clean) ]) = 1
    && List.length (stale ~exemptions:[ row "gone.ml" (Uncovered 1) ] [ ("t.ml", clean) ]) = 1);
  p_empty "an earned row is not stale" ~over:one.uncovered
    (stale ~exemptions:[ row "t.ml" (Uncovered 1) ] [ ("t.ml", one) ]);
  p "two rows naming one file are refused"
    (List.length
       (findings ~exemptions:[ row "t.ml" (Uncovered 1); row "t.ml" Every ] [ ("t.ml", one) ])
    = 1)
