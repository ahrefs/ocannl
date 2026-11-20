(** Sorting a forced-decline sweep's failures (gh-ocannl-1308's leg; the post-wave sorter of
    lukstafi/ocannl-staging#2233).

    Under [OCANNL_TOOL_TEST_FORCE_TILE_MMA_DECLINE=1] every [Tile_mma] declines to the lane-0 scalar
    fallback, so a sweep over the census-judging stanzas is red by design: its census claims are
    expected to fail and its parity claims are expected to hold. This reader takes such a sweep's
    log -- a [tools/test-run.sh] run record's [log], or any dune output of the same runs -- and
    sorts each failure into what the leg predicts and what it does not:

    - a {b census failure} (expected): a failed claim, in a stanza running a census-judging test,
      whose label asserts no parity and speaks the census's vocabulary ({!names_census});
    - an {b expected value failure}: a failed claim asserting parity that {!decline_sensitive}
      lists, each with the reason its reference is one rendering's numerics or its label conjoins a
      census judgment (build-and-test.md#verdict-claims names them);
    - an {b anomaly}, with its stanza: a failed parity claim nothing lists, a crash, a stanza whose
      failure count the reader cannot account for, a failure in a stanza that runs no census-judging
      test, and any failure dune reports on a file other than a dune file (a golden diff, a build
      error).

    {1 What it reads}

    - The census-judging tests are [tensorized_census_ratchet]'s own reading
      ({!Tensorized_census_scan.judged_paths}): the caller hands them in, never a list of its own.
    - "Asserts parity" is {!asserts_parity}: the census scan's own vocabulary
      ({!Tensorized_census_scan.asserts_parity}) widened by the agreement words a tolerance claim
      uses ([agree], [equal], [identical], [within], [same as]), applied to the label as the [FAIL:]
      line renders it. Wider than the scan, which decides what needs a census, because here a word
      missed is a value regression read as an expected census failure.
    - Lines are read with their ANSI SGR escapes removed, so a colorized dune log
      ([CLICOLOR_FORCE=1]) sorts as a plain one.
    - A failing action is a dune header, [File "<dir>/dune", line(s) ...], with the excerpt dune
      prints under it. The stanza is the [(name X)] it highlights, or the [(target X.actual)] of a
      rule; the test is the executable the excerpt runs ([%{dep:Y.exe}]), else [X] itself, so a
      variant rule sorts against its program's source [<dir>/Y.ml].
    - Inside a block: [FAIL: ] lines (claims; the trailing [: false] dropped), the teardown line
      [FAILED: <n> check(s) did not hold ...], and the crash markers [STOPPED EARLY], [Fatal error:]
      and [Uncaught exception]. A block is sorted only when it has a teardown line whose count
      equals its [FAIL:] lines and dune truncated nothing ([...TRUNCATED BY DUNE...]); anything else
      is an anomaly, so a failure the reader cannot see is never counted as a census one.
    - The aliases the run asked for, read off the line naming the most [@<dir>/runtest-<name>]
      targets, so the report can list the stanzas that held; and a run record's closing [exit: <n>]
      line. A run that ended other than 0 or 1 (a cap, a cancellation, a refused slot) did not
      finish its stanzas: that is an anomaly, and no stanza is reported held.

    {1 What it deliberately does not read}

    - A claim is read by its rendered label alone, as the census scan reads claims. So a failed
      claim whose label asserts no parity yet names the census vocabulary sorts as a census failure
      even if it was in fact a value claim; one naming neither is an anomaly (unread), so a new
      census wording is a loud red until {!names_census} learns it, never a silent pass.
    - It does not decide which stanzas SHOULD have run: a judged test missing from the sweep is the
      leg's own boundary ([tensorized_census_ratchet] holds every running stanza to declaring the
      variable), not something a log shows. *)

(* cc-fp-contract: not-executed -- a log reader matching claim labels as text *)

open Base

type sensitive = { source : string; label : string; reason : string }
(** A failed claim whose label asserts parity and which the decline is meant to fail: [source] is
    the test file, [label] the claim's rendered label with [*] standing for any text, and [reason]
    why a declined [Tile_mma] fails it without any parity regression. Each entry must name a claim
    its source still writes ({!stale}). *)

let decline_sensitive =
  [
    {
      source = "test/operations/schedule_mma_matmul.ml";
      label = "staged+tensorized * matmul equals the once-narrowed wide reference bitwise";
      reason =
        "its reference is the persistent fragment's once-per-tile narrowing, which the declined \
         scope's per-block narrowing misses by construction";
    };
    {
      source = "test/operations/qkv_zero_init.ml";
      label = "every tensorized * sketch renders tensorized and matches the materialized run";
      reason =
        "the label conjoins a census judgment (renders tensorized) with parity; the separate \
         parity claim over the same sketches is the one that must hold";
    };
  ]

(* [pattern] with [*] standing for any text, matched against the whole of [text]. *)
let glob_matches ~pattern text =
  match String.split pattern ~on:'*' with
  | [] -> String.is_empty text
  | [ whole ] -> String.equal whole text
  | first :: rest ->
      let last = List.last_exn rest and middle = List.drop_last_exn rest in
      let limit = String.length text - String.length last in
      let rec go pos = function
        | [] -> pos <= limit
        | piece :: more -> (
            match String.substr_index text ~pattern:piece ~pos with
            | Some i when i + String.length piece <= limit -> go (i + String.length piece) more
            | _ -> false)
      in
      String.is_prefix text ~prefix:first
      && String.is_suffix text ~suffix:last
      && go (String.length first) middle

(** The entries of {!decline_sensitive} whose label no longer appears in their source: some literal
    piece of the pattern is missing from the file (or the file is). [sources] is [(path, content)].
*)
let stale sources =
  List.filter decline_sensitive ~f:(fun e ->
      match List.Assoc.find sources e.source ~equal:String.equal with
      | None -> true
      | Some content ->
          List.exists (String.split e.label ~on:'*') ~f:(fun piece ->
              (not (String.is_empty (String.strip piece)))
              && not (String.is_substring content ~substring:(String.strip piece))))

(** Whether a rendered claim label asserts agreement with a reference: see the header. *)
let asserts_parity label =
  Tensorized_census_scan.asserts_parity label
  ||
  let text = String.lowercase label in
  List.exists [ "agree"; "equal"; "identical" ] ~f:(Tensorized_census_scan.has_stem text)
  || Tensorized_census_scan.has_word text "within"
  || String.is_substring text ~substring:"same as"

(** Whether a rendered label speaks the census's vocabulary: tensorization by the census scan's own
    words ({!Tensorized_census_scan.names_tensorization}), or a word of the rendering a census claim
    describes -- a stem of [render], [register-til], [register til], [label], [structure], [census],
    [declin], [fallback], [fragment], [intrinsic], [simdgroup], [micro-kernel], [emit], [tile],
    [tiling], [pack], [panel], [geometry], [bridge], [splat], [lane]. A closed list read off every
    census claim the saved sweeps fail; a census claim worded outside it reads as an anomaly (see
    the header). *)
let names_census label =
  Tensorized_census_scan.names_tensorization label
  ||
  let text = String.lowercase label in
  List.exists
    [
      "render";
      "register-til";
      "register til";
      "label";
      "structure";
      "census";
      "declin";
      "fallback";
      "fragment";
      "intrinsic";
      "simdgroup";
      "micro-kernel";
      "emit";
      "tile";
      "tiling";
      "pack";
      "panel";
      "geometry";
      "bridge";
      "splat";
      "lane";
    ]
    ~f:(Tensorized_census_scan.has_stem text)

(** {1 Reading a log} *)

(* [line] without its ANSI SGR escapes ([ESC [ ... m]), which a colorized dune writes around a
   header's parts. *)
let strip_sgr line =
  if not (String.mem line '\027') then line
  else
    let buf = Buffer.create (String.length line) in
    let n = String.length line in
    let rec go i =
      if i >= n then ()
      else if Char.equal line.[i] '\027' && i + 1 < n && Char.equal line.[i + 1] '[' then
        match String.index_from line (i + 2) 'm' with Some j -> go (j + 1) | None -> ()
      else (
        Buffer.add_char buf line.[i];
        go (i + 1))
    in
    go 0;
    Buffer.contents buf

type block = {
  dir : string;  (** The directory of the dune file the header names. *)
  stanza : string;  (** The test's name or the rule's target stem. *)
  program : string option;  (** The executable the excerpt runs, without [.exe]. *)
  fails : string list;  (** Claim labels, in order. *)
  teardown : int option;  (** The teardown line's count. *)
  crash : string option;  (** The first crash marker. *)
  truncated : bool;
}

type log = {
  blocks : block list;
  others : string list;
      (** Failures outside any stanza block, described: a header on a file other than a dune file (a
          golden diff), or a dune error with no location. *)
  requested : string list;  (** Stanza names of the [runtest-<name>] aliases the run built. *)
  run_exit : int option;  (** A run record's closing [exit: <n>], when the log has one. *)
}

(* [File "<path>", line N, characters A-B:] or [File "<path>", lines N-M, characters A-B:]. *)
let parse_header line =
  match String.chop_prefix line ~prefix:"File \"" with
  | None -> None
  | Some rest -> (
      match String.lsplit2 rest ~on:'"' with
      | Some (path, tail) when String.is_prefix tail ~prefix:", line" ->
          let chars =
            match String.substr_index tail ~pattern:"characters " with
            | None -> None
            | Some i -> (
                let spec =
                  String.drop_prefix tail (i + String.length "characters ")
                  |> String.rstrip ~drop:(fun c -> Char.equal c ':')
                in
                match String.lsplit2 spec ~on:'-' with
                | Some (a, b) -> (
                    match (Int.of_string_opt a, Int.of_string_opt b) with
                    | Some a, Some b -> Some (a, b)
                    | _ -> None)
                | None -> None)
          in
          Some (path, String.is_prefix tail ~prefix:", lines ", chars)
      | _ -> None)

(* An excerpt line dune prints under a header: [ N | text], an elision [.......], or the caret line.
   Returns the source text of a numbered line. *)
let excerpt_line line =
  let trimmed = String.lstrip line in
  if (not (String.is_empty trimmed)) && String.for_all trimmed ~f:(Char.equal '.') then Some None
  else if (not (String.is_empty trimmed)) && String.for_all trimmed ~f:(Char.equal '^') then
    Some None
  else
    match String.lsplit2 trimmed ~on:'|' with
    | Some (n, text)
      when (not (String.is_empty (String.strip n)))
           && String.for_all (String.strip n) ~f:Char.is_digit ->
        (* dune writes [N | text]: one space after the bar. *)
        Some (Some (String.chop_prefix text ~prefix:" " |> Option.value ~default:text))
    | _ -> None

(* The longest prefix of [s] whose characters satisfy [f]. *)
let take_while s ~f =
  match String.lfindi s ~f:(fun _ c -> not (f c)) with Some i -> String.prefix s i | None -> s

(* The word after [opening] in [text], up to a space or closing parenthesis. *)
let word_after text ~opening =
  Option.map (String.substr_index text ~pattern:opening) ~f:(fun i ->
      String.drop_prefix text (i + String.length opening)
      |> take_while ~f:(fun c -> not (Char.is_whitespace c || Char.equal c ')')))

let program_of excerpt =
  List.find_map excerpt ~f:(fun text ->
      List.find_map [ "%{dep:"; "%{exe:" ] ~f:(fun opening ->
          match word_after text ~opening with
          | Some w -> (
              let w = String.rstrip w ~drop:(Char.equal '}') in
              match String.chop_suffix (Stdlib.Filename.basename w) ~suffix:".exe" with
              | Some p -> Some p
              | None -> None)
          | None -> None))

let label_of_fail rest =
  match String.chop_suffix rest ~suffix:": false" with
  | Some label -> label
  | None -> (
      (* [pass_fail]'s row, [<label>: FAIL<detail>]. *)
      match String.substr_index_all rest ~may_overlap:false ~pattern:": FAIL" |> List.last with
      | Some i -> String.prefix rest i
      | None -> rest)

let teardown_count line =
  Option.bind (String.chop_prefix line ~prefix:"FAILED: ") ~f:(fun rest ->
      Int.of_string_opt (take_while rest ~f:Char.is_digit))

let crash_markers = [ "STOPPED EARLY"; "Fatal error:"; "Uncaught exception" ]

(* The [runtest-<name>] aliases of the dune command: the line naming the most of them, since a
   runner's own chatter can name one in passing. *)
let requested_of lines =
  let names line =
    String.split line ~on:' '
    |> List.filter_map ~f:(fun word ->
        if String.is_prefix word ~prefix:"@" then
          Option.map (String.substr_index word ~pattern:"/runtest-") ~f:(fun i ->
              String.drop_prefix word (i + String.length "/runtest-")
              |> String.rstrip ~drop:(fun c -> Char.equal c ',' || Char.equal c '.'))
        else None)
    |> List.dedup_and_sort ~compare:String.compare
  in
  List.fold lines ~init:[] ~f:(fun best line ->
      let here = names line in
      if List.length here > List.length best then here else best)

let read text =
  let lines = String.split_lines text |> List.map ~f:strip_sgr in
  let blocks = ref [] and others = ref [] in
  (* The open block: its fields so far, and whether its teardown line has been read (after which
     only crash markers are still attributed to it). *)
  let current = ref None in
  let close () =
    Option.iter !current ~f:(fun (b, _) -> blocks := { b with fails = List.rev b.fails } :: !blocks);
    current := None
  in
  let rec go = function
    | [] -> close ()
    | line :: rest -> (
        match parse_header line with
        | Some (path, multi, chars) ->
            close ();
            let excerpt, rest =
              let rec take acc = function
                | l :: more -> (
                    match excerpt_line l with
                    | Some (Some text) -> take (text :: acc) more
                    | Some None -> take acc more
                    | None -> (List.rev acc, l :: more))
                | [] -> (List.rev acc, [])
              in
              take [] rest
            in
            if String.equal (Stdlib.Filename.basename path) "dune" then
              let highlighted =
                match (multi, chars, excerpt) with
                | false, Some (a, b), [ text ] when b <= String.length text && a < b ->
                    Some (String.sub text ~pos:a ~len:(b - a))
                | _ -> None
              in
              let target =
                List.find_map excerpt ~f:(fun text -> word_after text ~opening:"(target ")
                |> Option.map ~f:(fun t ->
                    String.chop_suffix t ~suffix:".actual" |> Option.value ~default:t)
              in
              let stanza =
                match (target, highlighted) with
                | Some t, _ -> t
                | None, Some h -> h
                | None, None -> String.concat ~sep:" " excerpt
              in
              current :=
                Some
                  ( {
                      dir = Stdlib.Filename.dirname path;
                      stanza;
                      program =
                        (match program_of excerpt with Some p -> Some p | None -> highlighted);
                      fails = [];
                      teardown = None;
                      crash = None;
                      truncated = false;
                    },
                    false )
            else others := ("a failure on a file other than a dune file: " ^ line) :: !others;
            go rest
        | None ->
            (match !current with
            | (None | Some (_, true)) when String.is_prefix line ~prefix:"Error: " ->
                (* A dune error with no location (an empty alias, a refused invocation): no test
                   block holds it, and nothing it names ran. *)
                others := ("a dune error with no location: " ^ line) :: !others
            | None -> ()
            | Some (b, torn) ->
                let b =
                  if String.is_substring line ~substring:"...TRUNCATED BY DUNE..." then
                    { b with truncated = true }
                  else b
                in
                let b =
                  match
                    ( b.crash,
                      List.find crash_markers ~f:(fun m -> String.is_substring line ~substring:m) )
                  with
                  | None, Some _ -> { b with crash = Some (String.strip line) }
                  | _ -> b
                in
                let b, torn =
                  if torn then (b, torn)
                  else
                    match String.chop_prefix line ~prefix:"FAIL: " with
                    | Some r -> ({ b with fails = label_of_fail r :: b.fails }, false)
                    | None -> (
                        match teardown_count line with
                        | Some n -> ({ b with teardown = Some n }, true)
                        | None -> (b, false))
                in
                current := Some (b, torn));
            go rest)
  in
  go lines;
  let run_exit =
    List.rev lines
    |> List.find_map ~f:(fun line ->
        Option.bind (String.chop_prefix line ~prefix:"exit: ") ~f:Int.of_string_opt)
  in
  { blocks = List.rev !blocks; others = List.rev !others; requested = requested_of lines; run_exit }

(* Whether the run finished its stanzas: dune's own 0 or 1, or a log with no run-record footer. *)
let finished log = match log.run_exit with None | Some (0 | 1) -> true | Some _ -> false

(** {1 Sorting} *)

type verdict =
  | Census  (** Expected under the leg. *)
  | Unread  (** Anomaly: neither parity nor the census vocabulary. *)
  | Expected_value of string  (** A listed decline-sensitive value claim, with its reason. *)
  | Parity  (** Anomaly: a parity claim failed. *)

type sorted_block = {
  block : block;
  source : string option;  (** The census-judging test it runs, if it runs one. *)
  claims : (string * verdict) list;
  anomaly : string option;  (** Why the block as a whole is not sortable. *)
}

let sort ~judged log =
  let judged_source b =
    let candidates =
      match b.program with
      | Some p -> [ p ]
      | None ->
          (* No program in the excerpt: the stanza name, then each prefix before a [-variant]. *)
          let parts = String.split b.stanza ~on:'-' in
          List.init (List.length parts) ~f:(fun k ->
              String.concat ~sep:"-" (List.take parts (List.length parts - k)))
    in
    List.find_map candidates ~f:(fun p ->
        let path = b.dir ^ "/" ^ p ^ ".ml" in
        Option.some_if (List.mem judged path ~equal:String.equal) path)
  in
  List.map log.blocks ~f:(fun block ->
      let source = judged_source block in
      let n = List.length block.fails in
      let anomaly =
        match (source, block.crash, block.teardown) with
        | None, _, _ ->
            Some
              (Printf.sprintf "fails, and runs no census-judging test (%s)"
                 (match block.program with
                 | Some p -> block.dir ^ "/" ^ p ^ ".ml"
                 | None -> "its program is not in dune's excerpt"))
        | _, Some marker, _ -> Some ("crashed: " ^ marker)
        | _, None, None -> Some (Printf.sprintf "no teardown line under %d FAIL line(s)" n)
        | _, None, Some _ when block.truncated ->
            Some "dune truncated its output, so some failed claims are not in the log"
        | _, None, Some m when m <> n ->
            Some (Printf.sprintf "%d FAIL line(s) where its teardown counts %d" n m)
        | _ -> None
      in
      let claims =
        List.map block.fails ~f:(fun label ->
            let verdict =
              if not (asserts_parity label) then if names_census label then Census else Unread
              else
                match
                  List.find decline_sensitive ~f:(fun e ->
                      Option.equal String.equal (Some e.source) source
                      && glob_matches ~pattern:e.label label)
                with
                | Some e -> Expected_value e.reason
                | None -> Parity
            in
            (label, verdict))
      in
      { block; source; claims; anomaly })

let unfinished log =
  Option.map log.run_exit ~f:(fun n ->
      Printf.sprintf
        "the run ended with exit %d, not dune's 0 or 1: its stanzas did not all finish, so none is \
         known to have held"
        n)
  |> Option.filter ~f:(fun _ -> not (finished log))

let anomalies log sorted =
  Option.to_list (unfinished log)
  @ log.others
  @ List.concat_map sorted ~f:(fun s ->
      Option.to_list (Option.map s.anomaly ~f:(fun a -> s.block.stanza ^ ": " ^ a))
      @ List.filter_map s.claims ~f:(fun (label, v) ->
          match v with
          | Parity -> Some (Printf.sprintf "%s: a parity claim failed: %S" s.block.stanza label)
          | Unread ->
              Some
                (Printf.sprintf "%s: a failed claim reads as neither parity nor census: %S"
                   s.block.stanza label)
          | Census | Expected_value _ -> None))

(** The report, one line a stanza, then its expected value failures and every anomaly, then the
    requested stanzas that held. *)
let render log sorted =
  let buf = Buffer.create 1024 in
  let line fmt = Printf.ksprintf (fun s -> Buffer.add_string buf (s ^ "\n")) fmt in
  List.iter sorted ~f:(fun s ->
      let census =
        List.count s.claims ~f:(fun (_, v) -> match v with Census -> true | _ -> false)
      in
      line "  %s: %d census claim(s) failed%s" s.block.stanza census
        (match s.source with Some src -> " (" ^ src ^ ")" | None -> "");
      (* Each reason once, with the labels it covers. *)
      let reasons =
        List.filter_map s.claims ~f:(fun (_, v) ->
            match v with Expected_value r -> Some r | _ -> None)
        |> List.stable_dedup ~compare:String.compare
      in
      List.iter reasons ~f:(fun reason ->
          line "    expected value failure(s), since %s:" reason;
          List.iter s.claims ~f:(fun (label, v) ->
              match v with
              | Expected_value r when String.equal r reason -> line "      %S" label
              | _ -> ()));
      List.iter s.claims ~f:(fun (label, v) ->
          match v with
          | Parity -> line "    ANOMALY: a parity claim failed: %S" label
          | Unread ->
              line "    ANOMALY: a failed claim reads as neither parity nor census: %S" label
          | _ -> ());
      Option.iter s.anomaly ~f:(fun a -> line "    ANOMALY: %s" a));
  List.iter log.others ~f:(fun o -> line "  ANOMALY: %s" o);
  Option.iter (unfinished log) ~f:(fun a -> line "  ANOMALY: %s" a);
  let red = List.map sorted ~f:(fun s -> s.block.stanza) in
  let held =
    if finished log then
      List.filter log.requested ~f:(fun r -> not (List.mem red r ~equal:String.equal))
    else []
  in
  if not (List.is_empty held) then line "  held: %s" (String.concat ~sep:" " held);
  let a = List.length (anomalies log sorted) in
  line "  %d stanza(s) red, %d held, %d anomaly(ies)%s" (List.length sorted) (List.length held) a
    (if a = 0 then ": every failure is one the forced decline predicts" else "");
  Buffer.contents buf
