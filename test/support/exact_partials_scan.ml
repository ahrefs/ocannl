(** Prose claiming that partial sums are exact, and whether a derivation stands beside it
    (gh-ocannl-1729).

    A bitwise claim over a reduction rests on a premise: the summands are such that every partial
    sum, in whatever association the backend chooses, is exactly representable at the accumulator,
    so every correct schedule returns the same bits. Written in prose, that premise was wrong twice
    (gh-ocannl-1658: the magnitudes did not fit the format the comment named), and nothing could
    notice, because a comment is not evaluated. [Ll_test.exact_partials] derives it from the
    summands themselves (gh-ocannl-1693), so the rule is: prose that states the premise sits beside
    the claim that derives it.

    {1 What it reads — a LINT, and its boundary}

    This is a best-effort prose reader, fail-closed over a short phrase list, not a model of what a
    sentence means. Precisely:

    - THE TEXT: every comment (documentation comments included) and every string literal of the
      source, as {!Prose_lexemes} reads them with the compiler's lexer.
    - A CLAIM: one sentence of that text — split at [.], [;], [!] or [?] followed by whitespace or
      the end, and at a blank line — holding one of these closed forms (any case, hyphenated or
      not): [partial sum] or [partial sums] with [exact] or [exactly]; [exact partials]; or a sum
      word ([sum], [sums], [partials], [addition], [accumulation], [association]) with [exact] and
      an order marker ([any order], [every order], [any association], [every association]). The
      words and marker may occur in either order anywhere in the same sentence; the words within
      each bracketed phrase are adjacent. Words are maximal runs of letters, digits, [_] and [']:
      [exact_partials] and [inexact] are words of their own, neither of them [exact], and
      [f32-exact] holds the word [exact]. The lists are {!sum_words} and {!order_markers}. An
      association marker itself supplies the sum word: [exact in any association] and
      [exact in every association] count, even in "the inputs are exact in any association". This
      lexical reader does not resolve the sentence's subject.
    - WHAT IT DOES NOT READ: any other wording of the premise ("the inputs are exact", "the
      accumulation is exact", "dyadic inputs: any schedule computes the same bits"). A missed
      phrasing is a gap of this lint, not a pass the rule grants; widening the list is one follow-up
      issue, not a review round.
    - NEGATION, read crudely: a sentence holding any of the words {!negations} ([not], [never],
      [inexact], [round], [rounds], [isn't], [aren't]) is not a claim, whatever it negates — "the
      inputs are exact while their partial sums are not" states the opposite premise, the one a
      discrimination claim exhibits rather than derives. So "every partial sum is exact, not
      approximate" is read as no claim: a missed phrasing, on the side the header already gives up.

    {1 What "beside" means}

    A claim is COVERED when the top-level structure item it sits in — or, for prose between two
    items, the item that follows it, which is the item a leading comment describes — contains a
    reference to [Ll_test.exact_partials], spelled with that path. Read on ppxlib's parse tree, so a
    reference is code, never the identifier quoted in prose. So a file header stating the premise
    for legs further down is uncovered: it points at the leg whose claim derives it instead.

    Not resolved: what the path names (a local [module Ll_test] shadowing the library's would
    count), whether the reference's result reaches a [Verdict] claim, and whether its [~sig_bits]
    matches the format the prose names. A bare [exact_partials] after an [open Ll_test] does not
    count, which errs toward refusal. *)

open Base
open Ppxlib.Parsetree

let is_word_char c = Char.is_alphanum c || Char.equal c '_' || Char.equal c '\''

(* The words of [text], lowercased. *)
let words text =
  let out = ref [] in
  let n = String.length text in
  let i = ref 0 in
  while !i < n do
    if is_word_char text.[!i] then (
      let j = ref !i in
      while !j < n && is_word_char text.[!j] do
        Int.incr j
      done;
      out := String.lowercase (String.sub text ~pos:!i ~len:(!j - !i)) :: !out;
      i := !j)
    else Int.incr i
  done;
  List.rev !out

(** [sentences text] is [text] split into sentences: a sentence ends at [.], [;], [!] or [?]
    followed by whitespace or the end of [text], and at a blank line. *)
let sentences text =
  let n = String.length text in
  let out = ref [] in
  let start = ref 0 in
  let cut stop next =
    if stop > !start then out := String.sub text ~pos:!start ~len:(stop - !start) :: !out;
    start := next
  in
  let i = ref 0 in
  while !i < n do
    let c = text.[!i] in
    let next_is_space = !i + 1 >= n || Char.is_whitespace text.[!i + 1] in
    if
      (Char.equal c '.' || Char.equal c ';' || Char.equal c '!' || Char.equal c '?')
      && next_is_space
    then cut (!i + 1) (!i + 1)
    else if Char.equal c '\n' then (
      (* A blank line: a newline, then only spaces or tabs, then another newline. *)
      let j = ref (!i + 1) in
      while !j < n && (Char.equal text.[!j] ' ' || Char.equal text.[!j] '\t') do
        Int.incr j
      done;
      if !j < n && Char.equal text.[!j] '\n' then cut !i !j);
    Int.incr i
  done;
  cut n n;
  List.rev !out

(** The words that make a sentence no claim (the header's NEGATION bullet). *)
let negations = [ "not"; "never"; "inexact"; "round"; "rounds"; "isn't"; "aren't" ]

(** Sum words of the closed order-based form (including a marker's own [association]). *)
let sum_words = [ "sum"; "sums"; "partials"; "addition"; "accumulation"; "association" ]

(** Adjacent word pairs that mark arbitrary accumulation order. *)
let order_markers =
  [ ("any", "order"); ("every", "order"); ("any", "association"); ("every", "association") ]

(** Whether one sentence's words state the premise, as the header defines it. *)
let states_exact_partials sentence_words =
  let has word = List.mem sentence_words word ~equal:String.equal in
  let has_phrase (first, second) =
    let rec walk = function
      | a :: b :: _ when String.equal a first && String.equal b second -> true
      | _ :: rest -> walk rest
      | [] -> false
    in
    walk sentence_words
  in
  (has_phrase ("exact", "partials")
  || (has "exact" || has "exactly")
     && (has_phrase ("partial", "sum") || has_phrase ("partial", "sums"))
  || (has "exact" && List.exists sum_words ~f:has && List.exists order_markers ~f:has_phrase))
  && not (List.exists negations ~f:has)

type claim = { line : int; offset : int; sentence : string }
(** One claim: the line its lexeme starts on, and the sentence with its whitespace collapsed. *)

let collapse text =
  String.split_on_chars text ~on:[ ' '; '\n'; '\t'; '\r' ]
  |> List.filter ~f:(Fn.non String.is_empty)
  |> String.concat ~sep:" "

(** [claims prose] is every sentence of [prose] that states the premise. *)
let claims (prose : Prose_lexemes.t list) =
  List.concat_map prose ~f:(fun (lexeme : Prose_lexemes.t) ->
      List.filter_map (sentences lexeme.text) ~f:(fun sentence ->
          if states_exact_partials (words sentence) then
            Some { line = lexeme.line; offset = lexeme.start; sentence = collapse sentence }
          else None))

let derivation_path = [ "Ll_test"; "exact_partials" ]

(** Whether [item] holds a reference to [Ll_test.exact_partials] as code. *)
let derives item =
  let found = ref false in
  let visitor =
    object
      inherit Ppxlib.Ast_traverse.iter as super

      method! expression expr =
        (match expr.pexp_desc with
        | Pexp_ident { txt; _ } -> (
            match Ppxlib.Longident.flatten_exn txt with
            | path when List.equal String.equal path derivation_path -> found := true
            | _ -> ()
            | exception _ -> ())
        | _ -> ());
        super#expression expr
    end
  in
  visitor#structure_item item;
  !found

type reading = { covered : claim list; uncovered : claim list; deriving_items : int }
(** A source's reading: its claims, split by whether a derivation covers them, and how many of its
    structure items hold a derivation. *)

(** [read ~prose content] places each claim of [prose] in the structure item of [content] that holds
    it or, between items, the next one, and asks that item for a derivation. Raises whatever
    ppxlib's parser raises on a source that does not parse. *)
let read ~prose content =
  let items = Config_key_scan.structure_of content in
  let derived = List.map items ~f:(fun item -> (item.pstr_loc.loc_end.pos_cnum, derives item)) in
  let covered, uncovered =
    List.partition_tf (claims prose) ~f:(fun claim ->
        match List.find derived ~f:(fun (stop, _) -> claim.offset < stop) with
        | Some (_, derives) -> derives
        | None -> false)
  in
  { covered; uncovered; deriving_items = List.count derived ~f:snd }

(** [attempt ~source content] reads [content] (the file at [source]) or says why it could not: a
    source that does not lex or parse is the input's fault, any other exception the reader's
    ({!Scan_attempt}). *)
let attempt ~source content =
  match Scan_attempt.run (fun () -> read ~prose:(Prose_lexemes.of_source content) content) with
  | Scanned reading -> Ok reading
  | Unparsed report -> Error (Printf.sprintf "%s does not lex or parse: %s" source report)
  | Raised text -> Error (Printf.sprintf "%s: the exact-partials reader raised %s" source text)

(** {1 The ratchet} *)

(** What one file's exemption row allows: exactly [n] uncovered sentences, written before the rule
    and converted when the file is next touched; or [Every] sentence, for a file whose prose quotes
    the phrases as its subject (this reader's own documentation, its cases). *)
type allowance = Uncovered of int | Every

type exemption = { path : string; allowance : allowance; reason : string }

let finding path (claim : claim) =
  Printf.sprintf
    "%s:%d: prose states exact partial sums with no Ll_test.exact_partials reference in its \
     structure item: \"%s\" -- derive the premise there, or reword a sentence whose claim is not \
     about exactness (test/support/exact_partials_scan.ml)"
    path claim.line claim.sentence

(** [violations ~exemptions rows] is the ratchet's findings over [rows] (each source's path and
    reading), then the exemption rows nothing answers to. A file without a row may hold no uncovered
    sentence; a [Uncovered n] row's file holds exactly [n]; a row whose file holds none, or names no
    source read, is stale; a path with two rows is refused. *)
let violations ~exemptions rows =
  let rows = Map.of_alist_reduce (module String) rows ~f:(fun first _ -> first) in
  let duplicated =
    List.filter_map
      (List.find_all_dups exemptions ~compare:(fun a b -> String.compare a.path b.path))
      ~f:(fun e -> Some (Printf.sprintf "%s: two exemption rows name this file" e.path))
  in
  let exempt =
    Map.of_alist_reduce
      (module String)
      (List.map exemptions ~f:(fun e -> (e.path, e)))
      ~f:(fun first _ -> first)
  in
  let findings =
    Map.fold rows ~init:[] ~f:(fun ~key:path ~data:reading acc ->
        let uncovered = reading.uncovered in
        match Map.find exempt path with
        | None -> acc @ List.map uncovered ~f:(finding path)
        | Some { allowance = Every; _ } -> acc
        | Some { allowance = Uncovered n; _ } ->
            let actual = List.length uncovered in
            if actual = n || actual = 0 then acc
            else
              acc
              @ [
                  Printf.sprintf
                    "%s: %d uncovered sentences where its exemption row holds %d -- converting one \
                     lowers the row in the same commit, and a new one is a finding:\n\
                     %s"
                    path actual n
                    (String.concat ~sep:"\n"
                       (List.map uncovered ~f:(fun c ->
                            Printf.sprintf "  line %d: \"%s\"" c.line c.sentence)));
                ])
  in
  let stale =
    List.filter_map exemptions ~f:(fun e ->
        match (Map.find rows e.path, e.allowance) with
        | None, _ ->
            Some (Printf.sprintf "%s: an exemption row names no source the scan read" e.path)
        | Some reading, Uncovered _ when List.is_empty reading.uncovered ->
            Some
              (Printf.sprintf "%s: stale exemption row -- the file holds no uncovered sentence"
                 e.path)
        | Some reading, Every when List.is_empty (reading.covered @ reading.uncovered) ->
            Some (Printf.sprintf "%s: stale exemption row -- the file states no claim" e.path)
        | Some _, _ -> None)
  in
  (duplicated @ findings, stale)
