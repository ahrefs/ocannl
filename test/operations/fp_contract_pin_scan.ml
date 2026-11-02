(* gh-ocannl-1696: every test source that claims bitwise equality of executed values either pins
   [cc_backend_fp_contract=off] or says, in place, why the C compiler's contraction cannot move a
   bit.

   Under the default [cc_backend_fp_contract=auto], whether gcc fuses a product into the add after
   it depends on the kernel's SHAPE -- a vectorized body against its scalar remainder, a packed tile
   against the serial loop, unrolled against looped -- and on the compiler and the host: gcc before
   14 treats [-ffp-contract=on] as [off], and only the AVX512-FP16 CI runners compute f16 natively
   (gh-ocannl-1679, gh-ocannl-1682). So a bitwise claim between two kernel shapes can pass on the
   fleet's gcc 15 and fail on CI's gcc 13, or the other way round. [off] is the one mode every
   compiler reads the same way.

   {1 What it reads -- a LINT, and its boundary}

   - THE SUBJECT: a test source (test/ and arrayjit/test/, as [Ll_test_scan.test_source] has it) one
   of whose string literals -- a Verdict claim's name, in practice -- holds [bitwise], [bit-for-bit]
   or [bit-identical], in any case. Read with the compiler's lexer ({!Prose_lexemes}), so a comment
   saying "bitwise" is not a claim. A bitwise comparison whose claim text says neither (a bare
   [Float.equal] under "matches") is this lint's gap, not a pass. - A PIN: the string literal
   [OCANNL_CC_BACKEND_FP_CONTRACT] whose next string literal is [off] (the [Unix.putenv] the pinned
   tests do at module initialization). A pin made elsewhere -- a dune stanza's flag or profile, a
   directory's ocannl_config -- is not read: such a source carries a marker instead, or pins
   in-process too. - A MARKER: a comment whose whole text is [cc-fp-contract], a colon, a kind, [--]
   and a reason of at least two words. Every comment holding the sentinel followed by a colon must
   be one, so a misspelt kind or a missing reason fails rather than reads as absent. The kinds, a
   closed vocabulary ([kinds] below), each name why contraction cannot reach the claim.

   Every subject is listed in the golden with its disposition, so the list is the review surface: a
   new bitwise claim lands with its pin or its marker in the same diff. A source with a pin AND a
   marker, two markers, or a marker but no claim fails too, so a disposition never outlives the
   claim it was written for. *)
open Base
open Stdio
open Verdict.Claims
module Dune = Test_utils.Dune_stanza_scan
module Sources = Test_utils.Config_key_scan

(* cc-fp-contract: not-executed -- its bitwise text is fixture sources and claim names, no
   kernels *)

let kinds =
  [
    ( "exact-operands",
      "every product and partial sum is exact at the compared precision, so a fused multiply-add \
       rounds where the separate ones do: nowhere" );
    ( "no-contraction",
      "the compared values involve no floating product feeding an add: nothing for the compiler to \
       fuse" );
    ("same-shape", "on cc, both sides run the same kernel code, so they contract alike");
    ( "gated",
      "the claim reads the kernel's flags and target at run time and skips where they differ" );
    ("not-cc", "only a GPU backend evaluates the claim; cc skips it");
    ("not-executed", "no claim is about executed values: fixtures, codegen text, host arithmetic");
  ]

let sentinel = "cc-fp-contract:"

(* The pin's two literals, the value spelled first: the other order would make this file's own
   source read as a pin. *)
let pin_value = "off"
let pin_key = "OCANNL_CC_BACKEND_FP_CONTRACT"
let subject_phrases = [ "bitwise"; "bit-for-bit"; "bit-identical" ]

type disposition = Pinned | Marked of string * string

type reading = {
  claim_lines : int list;
  pinned : bool;
  markers : (int * (string * string, string) Result.t) list;
}

(* [text] is a marker's whole comment text; [Error] says why it is not a well-formed one. *)
let parse_marker text =
  let text = String.strip text in
  match String.chop_prefix text ~prefix:sentinel with
  | None -> Error "the sentinel must open the comment"
  | Some rest -> (
      match String.substr_index rest ~pattern:" -- " with
      | None -> Error "expected `<kind> -- <reason>` after the sentinel"
      | Some i ->
          let kind = String.strip (String.prefix rest i) in
          (* A reason may wrap across lines; it is read, and listed, as its words. *)
          let reason =
            String.drop_prefix rest (i + 4)
            |> String.split_on_chars ~on:[ ' '; '\n'; '\t'; '\r' ]
            |> List.filter ~f:(Fn.non String.is_empty)
            |> String.concat ~sep:" "
          in
          if not (List.Assoc.mem kinds ~equal:String.equal kind) then
            Error ("unknown kind `" ^ kind ^ "`")
          else if List.length (String.split reason ~on:' ') < 2 then
            Error "the reason needs at least two words"
          else Ok (kind, reason))

let read content =
  let lexemes = Prose_lexemes.of_source content in
  let literals =
    List.filter lexemes ~f:(fun (l : Prose_lexemes.t) ->
        match l.kind with String_literal -> true | Comment -> false)
  in
  let claim_lines =
    List.filter_map literals ~f:(fun l ->
        let lower = String.lowercase l.text in
        if
          List.exists subject_phrases ~f:(fun pattern ->
              String.is_substring lower ~substring:pattern)
        then Some l.line
        else None)
  in
  let rec pinned = function
    | (a : Prose_lexemes.t) :: (b :: _ as rest) ->
        (String.equal a.text pin_key && String.equal b.text pin_value) || pinned rest
    | _ -> false
  in
  let markers =
    List.filter_map lexemes ~f:(fun (l : Prose_lexemes.t) ->
        match l.kind with
        | Comment when String.is_substring l.text ~substring:sentinel ->
            Some (l.line, parse_marker l.text)
        | Comment | String_literal -> None)
  in
  { claim_lines; pinned = pinned literals; markers }

(* The source's disposition, or why it has none. [None] for a source outside the subject. *)
let judge reading =
  let malformed =
    List.filter_map reading.markers ~f:(fun (line, m) ->
        match m with
        | Error why -> Some (Printf.sprintf "line %d: malformed marker: %s" line why)
        | Ok _ -> None)
  in
  let well_formed = List.filter_map reading.markers ~f:(fun (_, m) -> Result.ok m) in
  match (malformed, reading.claim_lines, reading.pinned, well_formed) with
  | _ :: _, _, _, _ -> Some (Error (String.concat ~sep:"; " malformed))
  | [], [], _, [] -> None
  | [], [], _, _ :: _ -> Some (Error "a marker, but no bitwise claim for it to dispose of")
  | [], _ :: _, true, [] -> Some (Ok Pinned)
  | [], _ :: _, true, _ :: _ -> Some (Error "both a pin and a marker: drop the marker")
  | [], _ :: _, false, [ (kind, reason) ] -> Some (Ok (Marked (kind, reason)))
  | [], _ :: _, false, _ :: _ :: _ -> Some (Error "more than one marker")
  | [], line :: _, false, [] ->
      Some
        (Error
           (Printf.sprintf
              "line %d claims bitwise equality, but the source neither pins \
               OCANNL_CC_BACKEND_FP_CONTRACT=off nor carries a marker"
              line))

let judge_text content = judge (read content)

(* The reader against synthetic sources: each direction, and the nearest text it must not take. *)
let cases () =
  let claim = "let () = p \"tiled matches serial bitwise\" true\n" in
  let pin = "let () = Unix.putenv \"OCANNL_CC_BACKEND_FP_CONTRACT\" \"off\"\n" in
  let marker kind reason = Printf.sprintf "(* %s %s -- %s *)\n" sentinel kind reason in
  let refused src = match judge_text src with Some (Error _) -> true | _ -> false in
  p "case: a pinned claim is listed as pinned"
    (match judge_text (pin ^ claim) with Some (Ok Pinned) -> true | _ -> false);
  p "case: a marked claim is listed with its kind"
    (match judge_text (marker "same-shape" "one kernel, twice" ^ claim) with
    | Some (Ok (Marked ("same-shape", "one kernel, twice"))) -> true
    | _ -> false);
  p "case: a claim with neither pin nor marker is refused" (refused claim);
  p "case: a claim in capitals is a claim too"
    (refused "let () = p \"tiled and serial agree BIT-FOR-BIT\" true\n");
  p "case: a pin to another mode is no pin"
    (refused ("let () = Unix.putenv \"OCANNL_CC_BACKEND_FP_CONTRACT\" \"fast\"\n" ^ claim));
  p "case: a marker of an unknown kind is refused"
    (refused (marker "trust-me" "it is fine" ^ claim));
  p "case: a one-word reason is refused" (refused (marker "gated" "flags" ^ claim));
  p "case: a sentinel that does not open its comment is refused"
    (refused ("(* see " ^ sentinel ^ " gated -- reads the flags *)\n" ^ claim));
  p "case: two markers are refused"
    (refused (marker "gated" "reads the flags" ^ marker "not-cc" "GPU legs only" ^ claim));
  p "case: a pin and a marker together are refused"
    (refused (pin ^ marker "gated" "reads the flags" ^ claim));
  p "case: a marker with no claim is refused" (refused (marker "gated" "reads the flags"));
  p "case: the word in a comment is no claim"
    (Option.is_none (judge_text "(* the two agree bitwise *)\nlet () = p \"tiled matches\" true\n"))

(* The sources arrive as a `@<path>` response file, as verdict_ratchet's do; see
   [Test_utils.Scan_argv]. *)
let argv = Test_utils.Scan_argv.expand Stdlib.Sys.argv

let () =
  if Array.length argv < 2 then (
    eprintf "Usage: %s <workspace_root> <source...>\n" argv.(0);
    Stdlib.exit 2);
  let base = Dune.base_dir argv.(1) in
  let arguments =
    Array.to_list (Array.subo argv ~pos:2)
    |> List.map ~f:(fun path -> (Dune.repo_relative base path, path))
  in
  let on_disk = Map.of_alist_reduce (module String) arguments ~f:(fun first _ -> first) in
  let sources =
    Sources.sources_among (List.map arguments ~f:fst)
    |> List.filter ~f:Test_utils.Ll_test_scan.test_source
  in
  let read_sources, unlexed =
    List.partition_map sources ~f:(fun source ->
        match judge_text (In_channel.read_all (Map.find_exn on_disk source)) with
        | verdict -> First (source, verdict)
        | exception exn -> Second (source ^ ": " ^ Stdlib.Printexc.to_string exn))
  in
  let listed =
    List.filter_map read_sources ~f:(fun (source, v) -> Option.map v ~f:(fun v -> (source, v)))
  in
  let entries =
    List.filter_map listed ~f:(fun (s, v) -> Result.ok v |> Option.map ~f:(fun d -> (s, d)))
  in
  let refusals =
    List.filter_map listed ~f:(fun (s, v) ->
        match v with Error why -> Some (s ^ ": " ^ why) | Ok _ -> None)
  in
  eprintf "fp-contract pin scan: %d sources read, %d listed (not part of the golden)\n"
    (List.length read_sources) (List.length listed);
  List.iter (unlexed @ refusals) ~f:(eprintf "  refused %s\n");
  cases ();
  p_empty "every test source lexes" ~over:sources unlexed;
  p_exists "the walk read the arrayjit package's test sources too" read_sources ~f:(fun (path, _) ->
      String.is_prefix path ~prefix:"arrayjit/test/");
  p_exists "some listed source pins the mode in-process" entries ~f:(fun (_, d) ->
      match d with Pinned -> true | Marked _ -> false);
  p_exists "some listed source carries a marker" entries ~f:(fun (_, d) ->
      match d with Marked _ -> true | Pinned -> false);
  p_empty
    "every test source claiming bitwise equality pins cc_backend_fp_contract=off or carries one \
     well-formed marker, and every marker disposes of a claim"
    ~over:listed refusals;
  printf "\nKinds:\n";
  List.iter kinds ~f:(fun (kind, meaning) -> printf "%s -- %s\n" kind meaning);
  printf "\nTest sources whose string literals claim bitwise equality:\n";
  List.iter entries ~f:(fun (source, d) ->
      match d with
      | Pinned -> printf "%s -- pinned off\n" source
      | Marked (kind, reason) -> printf "%s -- %s: %s\n" source kind reason);
  Test_utils.Refusal_control_manifest.print "fp_contract_pin_scan.ml"
