open Base
(** Bounded executable Markdown: only explicitly selected OCaml is compiled. The reader accepts
    top-level backtick/tilde fences (up to three leading spaces), the same fences inside one level
    of block quote ([> ```ocaml], as slide containers write them), and one-line [Doc-check] code
    spans in agent notes. Unsupported annotated syntax fails loudly. Unselected OCaml fences are
    inventoried, never guessed into self-contained programs.

    A fence of any language may also declare where its code comes from (gh-ocannl-1265), right after
    the language word: [doc-excerpt=<path>] says it is a verbatim excerpt of that repository file,
    which {!check_provenance} holds it to (up to whitespace); [doc-adapted=<path>] says it is
    adapted from that file and is held only to the file existing. A fence whose caption -- the line
    directly above its opening, with no blank line between -- says [from <file>.ml] must declare one
    or the other, so library code presented as an excerpt cannot drift unchecked. *)

type status = Check of string | Skip of string | Unchecked
type provenance = Excerpt of string | Adapted of string

type block = {
  path : string;
  line : int;
  code : string;
  status : status;
  provenance : provenance option;
}

let valid_id id =
  (not (String.is_empty id))
  && Char.is_lowercase id.[0]
  && String.for_all id ~f:(fun c -> Char.is_lowercase c || Char.is_digit c || Char.equal c '_')

let fail path line message = failwith (Printf.sprintf "%s:%d: %s" path line message)

let words info =
  String.split_on_chars info ~on:[ ' '; '\t' ] |> List.filter ~f:(Fn.non String.is_empty)

let annotated info =
  List.exists [ "doc-check"; "doc-skip"; "doc-excerpt"; "doc-adapted" ] ~f:(fun substring ->
      String.is_substring info ~substring)

let ocaml_info info = match words info with "ocaml" :: _ -> true | _ -> false

(** A repository-relative file path: no absolute path, no [..], no empty component. *)
let valid_source source =
  (not (String.is_empty source))
  && (not (String.is_prefix source ~prefix:"/"))
  && List.for_all (String.split source ~on:'/') ~f:(fun component ->
      not (String.is_empty component || String.equal component ".."))

(** The provenance declaration right after the language word, and the info words without it. *)
let provenance ~path ~line info =
  let source annotation prefix =
    let source = String.drop_prefix annotation (String.length prefix) in
    if not (valid_source source) then
      fail path line (Printf.sprintf "%s needs a repository-relative file path" prefix);
    source
  in
  match words info with
  | language :: annotation :: rest when String.is_prefix annotation ~prefix:"doc-excerpt=" ->
      (Some (Excerpt (source annotation "doc-excerpt=")), language :: rest)
  | language :: annotation :: rest when String.is_prefix annotation ~prefix:"doc-adapted=" ->
      (Some (Adapted (source annotation "doc-adapted=")), language :: rest)
  | words ->
      if
        List.exists words ~f:(fun word ->
            String.is_prefix word ~prefix:"doc-excerpt"
            || String.is_prefix word ~prefix:"doc-adapted")
      then fail path line "doc-excerpt=/doc-adapted= goes right after the fence's language word";
      (None, words)

let status ~path ~line ~declared = function
  | [ "ocaml" ] -> Unchecked
  | [ "ocaml"; annotation ] when String.is_prefix annotation ~prefix:"doc-check=" ->
      let id = String.drop_prefix annotation 10 in
      if not (valid_id id) then fail path line "invalid doc-check group";
      Check id
  | "ocaml" :: "doc-skip" :: reason when not (List.is_empty reason) ->
      Skip (String.concat ~sep:" " reason)
  | [ language ] when declared && not (String.is_prefix language ~prefix:"doc-") -> Unchecked
  | _ -> fail path line "expected ocaml doc-check=<group> or ocaml doc-skip <reason>"

let fence line =
  let trimmed = String.lstrip line in
  let indent = String.length line - String.length trimmed in
  if
    indent > 3
    || String.length trimmed < 3
    || not (String.for_all (String.prefix line indent) ~f:(Char.equal ' '))
  then None
  else
    let c = trimmed.[0] in
    if not (Char.equal c '`' || Char.equal c '~') then None
    else
      let rec count n =
        if n < String.length trimmed && Char.equal trimmed.[n] c then count (n + 1) else n
      in
      let n = count 0 in
      if n < 3 then None else Some (c, n, String.strip (String.drop_prefix trimmed n))

(** [line] inside one level of block quote: its [>] marker (after up to three spaces) and one
    following space removed. *)
let unquote line =
  let trimmed = String.lstrip ~drop:(Char.equal ' ') line in
  if String.length line - String.length trimmed > 3 then None
  else
    Option.map (String.chop_prefix trimmed ~prefix:">") ~f:(fun rest ->
        Option.value (String.chop_prefix rest ~prefix:" ") ~default:rest)

(** The OCaml source files a caption line presents the fence below it as code from, lowercased: each
    [from] followed by a file name ending in [.ml] or [.mli]. The name is the first run of path
    characters after the blanks, with the Markdown that may wrap it -- any punctuation outside
    letters, digits, [/], [.], [-] and [_], and emphasis underscores at either end -- read as a
    delimiter, so a link, a code span, a parenthesis, emphasis or a quote around it does not hide
    it. *)
let cited_sources caption =
  let caption = String.lowercase caption in
  let length = String.length caption in
  let is_name c = Char.is_alphanum c || List.mem [ '_'; '.'; '/'; '-' ] c ~equal:Char.equal in
  let rec from_at start =
    match String.substr_index caption ~pattern:"from " ~pos:start with
    | None -> []
    | Some at ->
        let word_start = at = 0 || not (Char.is_alphanum caption.[at - 1]) in
        let rec skip index =
          if index < length && not (is_name caption.[index] && not (Char.equal caption.[index] '_'))
          then skip (index + 1)
          else index
        in
        let name_start = skip (at + 5) in
        let rec name_end index =
          if index < length && is_name caption.[index] then name_end (index + 1) else index
        in
        let name =
          String.strip
            ~drop:(fun c -> Char.equal c '.' || Char.equal c '_')
            (String.sub caption ~pos:name_start ~len:(name_end name_start - name_start))
        in
        let cited =
          word_start && (String.is_suffix name ~suffix:".ml" || String.is_suffix name ~suffix:".mli")
        in
        (if cited then [ name ] else []) @ from_at (at + 5)
  in
  from_at 0

let parse ~path text =
  let lines = String.split_lines text in
  (* [caption]: the line directly above, with its quote marker removed when it is quoted. *)
  let rec loop number active caption acc = function
    | [] -> (
        match active with
        | None -> List.rev acc
        | Some (_, _, _, opening, _, _, _) -> fail path opening "unclosed fence")
    | source :: rest -> (
        match active with
        | Some (c, n, info, opening, code, quoted, opened_caption) -> (
            let finish acc =
              let declared, info_words = provenance ~path ~line:opening info in
              let code = String.concat ~sep:"\n" (List.rev code) in
              let cited = Option.value_map opened_caption ~default:[] ~f:cited_sources in
              (match declared with
              | _ when List.is_empty cited -> ()
              | None ->
                  fail path opening
                    "a fence captioned as code `from <file>.ml` declares doc-excerpt=<path> or \
                     doc-adapted=<path>"
              | Some (Excerpt source | Adapted source) ->
                  (* The declaration names the file the caption cites, by its repository path. *)
                  let source = String.lowercase source in
                  if
                    not
                      (List.exists cited ~f:(fun name ->
                           String.equal source name || String.is_suffix source ~suffix:("/" ^ name)))
                  then
                    fail path opening
                      (Printf.sprintf "the caption cites %s, but the fence declares %s"
                         (String.concat ~sep:", " cited) source));
              if ocaml_info info || annotated info then
                {
                  path;
                  line = opening + 1;
                  code;
                  status = status ~path ~line:opening ~declared:(Option.is_some declared) info_words;
                  provenance = declared;
                }
                :: acc
              else acc
            in
            let content = if quoted then unquote source else Some source in
            match content with
            | None ->
                (* The block quote ended before the fence closed. *)
                if annotated info then
                  fail path opening "a quoted fence ends before its closing fence";
                loop (number + 1) None (Some source) (finish acc) rest
            | Some content -> (
                match fence content with
                | Some (c', n', "") when Char.equal c c' && n' >= n ->
                    loop (number + 1) None (Some content) (finish acc) rest
                | _ ->
                    loop (number + 1)
                      (Some (c, n, info, opening, content :: code, quoted, opened_caption))
                      (Some content) acc rest))
        | None -> (
            let opened =
              match fence source with
              | Some (c, n, info) -> Some (c, n, info, false)
              | None ->
                  Option.bind (unquote source) ~f:(fun inner ->
                      Option.map (fence inner) ~f:(fun (c, n, info) -> (c, n, info, true)))
            in
            match opened with
            | Some (c, n, info, quoted) ->
                loop (number + 1) (Some (c, n, info, number, [], quoted, caption)) None acc rest
            | None ->
                let acc =
                  match String.substr_index source ~pattern:"Doc-check `" with
                  | None ->
                      if
                        annotated source
                        && (String.is_substring source ~substring:"```"
                           || String.is_substring source ~substring:"~~~")
                      then fail path number "doc-check annotation outside a supported fence";
                      acc
                  | Some start ->
                      let tail = String.drop_prefix source (start + 11) in
                      let id, tail =
                        match String.lsplit2 tail ~on:'`' with
                        | Some pair -> pair
                        | None -> fail path number "unclosed Doc-check group"
                      in
                      if not (valid_id id) then fail path number "invalid Doc-check group";
                      let code =
                        match String.chop_prefix tail ~prefix:": `" with
                        | Some code -> (
                            match String.chop_suffix code ~suffix:"`." with
                            | Some code when not (String.contains code '`') -> code
                            | _ ->
                                fail path number
                                  "Doc-check must end with one code span and a period")
                        | None -> fail path number "Doc-check requires a single code span"
                      in
                      { path; line = number; code; status = Check id; provenance = None } :: acc
                in
                let caption =
                  if String.is_empty (String.strip source) then None
                  else Some (Option.value (unquote source) ~default:source)
                in
                loop (number + 1) None caption acc rest))
  in
  loop 1 None None [] lines

let selected blocks =
  List.filter_map blocks ~f:(fun block ->
      match block.status with Check id -> Some (id, block) | Skip _ | Unchecked -> None)

let require_selected blocks =
  if List.is_empty (selected blocks) then failwith "no selected documentation examples"

let render blocks =
  let groups =
    selected blocks
    |> List.map ~f:(fun (id, block) -> (block.path, id))
    |> List.dedup_and_sort ~compare:(fun (a, b) (c, d) ->
        let n = String.compare a c in
        if n = 0 then String.compare b d else n)
  in
  String.concat ~sep:"\n"
    (List.mapi groups ~f:(fun index (path, id) ->
         let body =
           selected blocks
           |> List.filter_map ~f:(fun (group, block) ->
               if String.equal id group && String.equal path block.path then
                 Some (Printf.sprintf "# %d %S\n%s\n" block.line block.path block.code)
               else None)
           |> String.concat ~sep:"\n"
         in
         Printf.sprintf "module Example_%d_%s = struct\n%s\nend\n" index id body))

let coverage blocks =
  List.concat_map blocks ~f:(fun block ->
      let status =
        match block.status with
        | Check id -> [ Printf.sprintf "%s: checked %s" block.path id ]
        | Skip reason -> [ Printf.sprintf "%s: excluded %s" block.path reason ]
        | Unchecked -> []
      in
      let provenance =
        match block.provenance with
        | Some (Excerpt source) -> [ Printf.sprintf "%s: excerpt of %s" block.path source ]
        | Some (Adapted source) -> [ Printf.sprintf "%s: adapted from %s" block.path source ]
        | None -> []
      in
      status @ provenance)
  |> List.sort ~compare:String.compare |> String.concat ~sep:"\n"

(** [text] with every whitespace run, line breaks included, read as one space. *)
let normalize text =
  String.split_on_chars text ~on:[ ' '; '\t'; '\r'; '\n' ]
  |> List.filter ~f:(Fn.non String.is_empty)
  |> String.concat ~sep:" "

(** The provenance failures of [blocks], [read] giving a repository file's contents: an excerpt must
    be a nonempty whitespace-normalized substring of its source, and both declarations must name a
    file that exists. A failure names the excerpt's first line its source lacks. *)
let check_provenance ~read blocks =
  List.filter_map blocks ~f:(fun block ->
      let at = Printf.sprintf "%s:%d" block.path block.line in
      match block.provenance with
      | None -> None
      | Some (Adapted source) ->
          Option.some_if
            (Option.is_none (read source))
            (Printf.sprintf
               "%s: doc-adapted=%s names no file in the repository (or one the doc_examples rule \
                in test/operations/dune does not declare)"
               at source)
      | Some (Excerpt source) -> (
          match read source with
          | None ->
              Some
                (Printf.sprintf
                   "%s: doc-excerpt=%s names no file in the repository (or one the doc_examples \
                    rule in test/operations/dune does not declare)"
                   at source)
          | Some text ->
              let excerpt = normalize block.code in
              let original = normalize text in
              if String.is_empty excerpt then
                Some (Printf.sprintf "%s: doc-excerpt=%s marks an empty fence" at source)
              else if String.is_substring original ~substring:excerpt then None
              else
                let first_missing =
                  List.find (String.split_lines block.code) ~f:(fun line ->
                      let line = normalize line in
                      (not (String.is_empty line))
                      && not (String.is_substring original ~substring:line))
                in
                Some
                  (Printf.sprintf
                     "%s: doc-excerpt=%s is not a verbatim excerpt of its source (up to \
                      whitespace)%s"
                     at source
                     (match first_missing with
                     | Some line -> Printf.sprintf "; %s has no line %S" source (String.strip line)
                     | None -> "; every line occurs there, but not in this order"))))
