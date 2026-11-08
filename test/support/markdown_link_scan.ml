open Base
(** Closed dialect: the agent-notes inline link reader, with literal relative paths. Fragments are
    stripped by the shared resolver; general-document anchors are not checked. Unsupported relative
    destination syntax is refused rather than partially resolved. *)

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

let relative target =
  let path = List.hd_exn (String.split target ~on:'#') in
  let scheme =
    match String.lsplit2 path ~on:':' with
    | Some (prefix, _) ->
        (not (String.is_empty prefix))
        && Char.is_alpha prefix.[0]
        && String.for_all prefix ~f:(fun c ->
            Char.is_alphanum c || List.mem [ '+'; '-'; '.' ] c ~equal:Char.equal)
    | None -> false
  in
  not (String.is_empty path || String.is_prefix path ~prefix:"/" || scheme)

(* Published odoc/site routes, not source files. Exact pairs keep the exception narrow. *)
let published =
  [
    ("docs/migration_guide.md", "../dev/neural_nets_lib/Ocannl/Shape/index.html");
    ("docs/migration_guide.md", "syntax_extensions.html");
    ("docs/slides-shapes_and_einsum.md", "syntax_extensions.html");
    ("docs/slides-shapes_and_einsum.md", "../dev/neural_nets_lib/Ocannl/Shape/index.html");
  ]

let check ~exists ~from_file contents =
  Notes.links_of contents
  |> List.filter_map ~f:(fun (_, target) ->
      if
        (not (relative target))
        || List.mem published (from_file, target) ~equal:(fun (a, b) (c, d) ->
            String.equal a c && String.equal b d)
      then None
      else
        let unsupported =
          String.exists
            (List.hd_exn (String.split target ~on:'#'))
            ~f:(fun c -> List.mem [ '('; ')'; '\\'; '%'; '?'; '&'; ':' ] c ~equal:Char.equal)
        in
        if unsupported then Some (target, "unsupported relative destination syntax")
        else
          match Notes.resolve_link ~from_file target with
          | Some path when exists path -> None
          | _ -> Some (target, "relative target does not exist"))
