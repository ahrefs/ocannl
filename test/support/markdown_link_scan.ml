(** Closed dialect: single-line inline links and images with unnested labels and literal relative
    file/directory destinations, angle destinations and optional titles, through
    [Agent_notes_scan.links_of]. Reference links, HTML links, code spans, fences and comments are
    outside the destination reader. Schemed and protocol-relative URLs are ignored; root-relative
    paths, escapes, percent encoding, queries, entities, colons and parentheses are refused.

    Nonempty fragments on Markdown targets in [in_corpus], including same-file fragments, are
    checked. Non-Markdown and out-of-corpus fragments are ignored (including source #L line links),
    but their files must exist. Empty fragments mean the top of the file.

    Heading dialect: ATX (one to six hashes, up to three leading spaces), outside fences, comments
    and code spans. ASCII text plus U+2014 em dash (removed), unpadded inline code and asterisk
    emphasis are supported: lowercase, spaces become hyphens, punctuation is removed except
    underscores and hyphens. Duplicate IDs use GitHub's first unused -1, -2, ... suffix, including
    collisions with literal suffixed titles. Setext headings, links/images, HTML, entities, escapes,
    other Unicode, controls, padded code spans and underscore emphasis in headings refuse the linked
    file rather than guessing its IDs. *)

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

let heading_errors contents =
  let scan = Notes.inert_by_line contents in
  let paragraphs =
    Notes.paragraph_lines ~inert:scan.ranges ~fences_at:scan.fence_ranges
      ~comments_at:scan.comment_ranges contents
  in
  List.filter_map (Notes.lines contents) ~f:(fun (lineno, line) ->
      if not (Notes.marker_is_text ~spans:(Notes.spans_at scan.ranges lineno) line) then None
      else
        let error =
          match atx_heading line with
          | Some _ ->
              if String.mem line '\\' then Some "escapes in heading"
              else Notes.heading_markup (heading_text line)
          | None ->
              Notes.setext_underline
                ~under_paragraph:(Set.mem paragraphs (lineno - 1))
                (String.strip line)
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
