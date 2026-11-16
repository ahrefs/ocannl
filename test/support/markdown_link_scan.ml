(** Closed dialect: single-line inline links and images with unnested labels and literal relative
    file/directory destinations, angle destinations and optional titles, through
    [Agent_notes_scan.links_of]. Reference links, HTML links, code spans, fences and comments are
    outside the destination reader. Schemed and protocol-relative URLs are ignored; root-relative
    paths, escapes, percent encoding, queries, entities, colons and parentheses are refused.

    Nonempty fragments on Markdown targets in [in_corpus], including same-file fragments, are
    checked. Non-Markdown and out-of-corpus fragments are ignored (including source #L line links),
    but their files must exist. Empty fragments mean the top of the file.

    Heading dialect: ATX (one to six hashes, up to three leading spaces), outside fences, comments
    and code spans; top-level indented code is ignored. Container heading candidates
    (quote/list-prefixed ATX or Setext and indented continuations, including code-like candidates)
    refuse the target because container block contexts are not interpreted. ASCII text plus U+2014
    em dash (removed), unpadded inline code and asterisk emphasis are supported: lowercase, spaces
    become hyphens, punctuation is removed except underscores and hyphens. Duplicate IDs use
    GitHub's first unused -1, -2, ... suffix, including collisions with literal suffixed titles.
    Setext headings, links/images, HTML, entities, escapes, other Unicode, controls, padded code
    spans and underscore emphasis in headings refuse the linked file rather than guessing its IDs.
*)

open Base
module Notes = Agent_notes_scan

let in_corpus path =
  String.is_suffix path ~suffix:".md"
  && (String.equal path "README.md" || String.equal path "CHANGES.md"
     || String.is_prefix path ~prefix:"docs/"
     || String.is_prefix path ~prefix:"benchmarks/")

(** The root directory has the empty normalized path; other directories exist when a source descends
    from them. The inventory itself contains files, not directory entries. *)
let path_exists ~files path =
  String.is_empty path
  || List.exists files ~f:(fun file ->
      String.equal file path || String.is_prefix file ~prefix:(path ^ "/"))

let relative path =
  let scheme =
    match String.lsplit2 path ~on:':' with
    | Some (prefix, _) ->
        (not (String.is_empty prefix))
        && Char.is_alpha prefix.[0]
        && String.for_all prefix ~f:(fun c ->
            Char.is_alphanum c || List.mem [ '+'; '-'; '.' ] c ~equal:Char.equal)
    | None -> false
  in
  not (String.is_empty path || String.is_prefix path ~prefix:"//" || scheme)

(* Published odoc/site routes, not source files. Exact pairs keep the exception narrow. *)
let published =
  [
    ("docs/migration_guide.md", "../dev/neural_nets_lib/Ocannl/Shape/index.html");
    ("docs/migration_guide.md", "syntax_extensions.html");
    ("docs/slides-shapes_and_einsum.md", "syntax_extensions.html");
    ("docs/slides-shapes_and_einsum.md", "../dev/neural_nets_lib/Ocannl/Shape/index.html");
  ]

(* The shared notes parser has a deliberately narrower heading contract. Keep its ID allocator, but
   parse closing hashes here: only a space-delimited trailing run is decoration. *)
let atx_heading line =
  if Notes.indent_of line > 3 then None
  else
    let text = String.strip line in
    let hashes = Notes.run_length text 0 '#' in
    if hashes = 0 || hashes > 6 then None
    else
      let rest = String.drop_prefix text hashes in
      if not (String.is_empty rest || Char.is_whitespace rest.[0]) then None
      else
        let rest = String.strip rest in
        let bare = String.rstrip rest ~drop:(Char.equal '#') in
        let rest =
          if
            String.is_empty bare || String.is_suffix bare ~suffix:" "
            || String.is_suffix bare ~suffix:"\t"
          then String.rstrip bare
          else rest
        in
        Some rest

let heading_text line = String.substr_replace_all line ~pattern:"—" ~with_:""

(* A conservative refusal, not a container parser: once a quote/list prefix is present,
   heading-shaped content cannot participate in ID allocation with an unknown block context. *)
let container_content line =
  let rec strip consumed text =
    let text = String.strip text in
    let n = String.length text in
    let next =
      if String.is_prefix text ~prefix:">" then Some (String.drop_prefix text 1)
      else if
        n >= 2
        && List.mem [ '-'; '+'; '*' ] text.[0] ~equal:Char.equal
        && Char.is_whitespace text.[1]
      then Some (String.drop_prefix text 2)
      else
        let digits = ref 0 in
        while !digits < n && Char.is_digit text.[!digits] do
          Int.incr digits
        done;
        if
          !digits > 0 && !digits <= 9
          && !digits + 1 < n
          && List.mem [ '.'; ')' ] text.[!digits] ~equal:Char.equal
          && Char.is_whitespace text.[!digits + 1]
        then Some (String.drop_prefix text (!digits + 2))
        else None
    in
    match next with Some rest -> strip true rest | None -> if consumed then Some text else None
  in
  strip false line

let heading_errors contents =
  let scan = Notes.inert_by_line contents in
  let lines = Notes.lines contents in
  let unindented =
    List.filter_map lines ~f:(fun (lineno, line) ->
        if Notes.indent_of line <= 3 then Some lineno else None)
    |> Set.of_list (module Int)
  in
  let paragraphs =
    Set.inter unindented
      (Notes.paragraph_lines ~inert:scan.ranges ~fences_at:scan.fence_ranges
         ~comments_at:scan.comment_ranges contents)
  in
  let previous_container_prose = ref false in
  let in_container = ref false in
  let after_blank = ref true in
  List.filter_map lines ~f:(fun (lineno, line) ->
      let visible =
        Set.mem unindented lineno
        && Notes.marker_is_text ~spans:(Notes.spans_at scan.ranges lineno) line
      in
      let prefix = if visible then container_content line else None in
      let text = String.strip line in
      let container =
        match prefix with
        | Some _ -> prefix
        | None ->
            if
              !in_container
              && Notes.indent_of line > 0
              && Notes.marker_is_text ~spans:(Notes.spans_at scan.ranges lineno) line
            then Some text
            else None
      in
      (* Lists can continue after a blank, and prose can continue lazily without one. An unindented
         new paragraph or interrupting block ends this conservative candidate region. *)
      if Option.is_some prefix then in_container := true
      else if
        (not (String.is_empty text))
        && Notes.indent_of line = 0
        && (!after_blank
           || Option.is_some (atx_heading line)
           || Option.is_some (Notes.thematic_break text)
           || Option.is_some (Notes.fence_at line))
      then in_container := false;
      after_blank := String.is_empty text;
      let container_error =
        Option.bind container ~f:(fun text ->
            if
              Option.is_some (atx_heading text)
              || Option.is_some
                   (Notes.setext_underline ~under_paragraph:!previous_container_prose text)
            then Some "a heading candidate in an unsupported quote/list container"
            else None)
      in
      previous_container_prose :=
        Option.value_map container ~default:false ~f:(fun text ->
            (not (String.is_empty text))
            && Option.is_none (atx_heading text)
            && Option.is_none (Notes.thematic_break text));
      let error =
        match container_error with
        | Some _ -> container_error
        | None -> (
            if not visible then None
            else
              match atx_heading line with
              | Some _ ->
                  if String.mem line '\\' then Some "escapes in heading"
                  else Notes.heading_markup (heading_text line)
              | None ->
                  Notes.setext_underline
                    ~under_paragraph:(Set.mem paragraphs (lineno - 1))
                    (String.strip line))
      in
      Option.map error ~f:(fun message -> Printf.sprintf "line %d: %s" lineno message))

let heading_ids contents =
  let scan = Notes.inert_by_line contents in
  Notes.lines contents
  |> List.filter_map ~f:(fun (lineno, line) ->
      if Notes.marker_is_text ~spans:(Notes.spans_at scan.ranges lineno) line then
        Option.map (atx_heading line) ~f:heading_text
      else None)
  |> Notes.ids_of_headings

let check ~exists ~read ~from_file contents =
  Notes.links_of ~include_images:true contents
  |> List.filter_map ~f:(fun (_, target) ->
      let path, fragment =
        match String.lsplit2 target ~on:'#' with
        | None -> (target, None)
        | Some (path, fragment) -> (path, Some fragment)
      in
      if
        ((not (String.is_empty path)) && not (relative path))
        || String.is_empty target
        || List.mem published (from_file, target) ~equal:(fun (a, b) (c, d) ->
            String.equal a c && String.equal b d)
      then None
      else
        let unsupported =
          String.is_prefix path ~prefix:"/"
          || String.exists path ~f:(fun c ->
              List.mem [ '('; ')'; '\\'; '%'; '?'; '&'; ':' ] c ~equal:Char.equal)
        in
        if unsupported then Some (target, "unsupported relative destination syntax")
        else
          let resolved =
            if String.is_empty path then Some from_file else Notes.resolve_link ~from_file path
          in
          match resolved with
          | Some path when exists path -> (
              match fragment with
              | Some fragment when in_corpus path && not (String.is_empty fragment) -> (
                  match read path with
                  | None -> Some (target, "Markdown target cannot be read")
                  | Some contents -> (
                      match heading_errors contents with
                      | error :: _ -> Some (target, "unsupported heading syntax: " ^ error)
                      | [] ->
                          if List.mem (heading_ids contents) fragment ~equal:String.equal then None
                          else Some (target, "Markdown heading anchor does not exist")))
              | _ -> None)
          | _ -> Some (target, "relative target does not exist"))
