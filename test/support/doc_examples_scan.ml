open Base
(** Bounded executable Markdown: only explicitly selected OCaml is compiled. The reader accepts
    top-level backtick/tilde fences (up to three leading spaces), and one-line [Doc-check] code
    spans in agent notes. Unsupported annotated syntax fails loudly. Unselected OCaml fences are
    inventoried, never guessed into self-contained programs. *)

type status = Check of string | Skip of string | Unchecked
type block = { path : string; line : int; code : string; status : status }

let valid_id id =
  (not (String.is_empty id))
  && Char.is_lowercase id.[0]
  && String.for_all id ~f:(fun c -> Char.is_lowercase c || Char.is_digit c || Char.equal c '_')

let fail path line message = failwith (Printf.sprintf "%s:%d: %s" path line message)

let status ~path ~line info =
  match String.split info ~on:' ' |> List.filter ~f:(Fn.non String.is_empty) with
  | [ "ocaml" ] -> Unchecked
  | [ "ocaml"; annotation ] when String.is_prefix annotation ~prefix:"doc-check=" ->
      let id = String.drop_prefix annotation 10 in
      if not (valid_id id) then fail path line "invalid doc-check group";
      Check id
  | "ocaml" :: "doc-skip" :: reason when not (List.is_empty reason) ->
      Skip (String.concat ~sep:" " reason)
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

let parse ~path text =
  let lines = String.split_lines text in
  let rec loop number active acc = function
    | [] -> (
        match active with
        | None -> List.rev acc
        | Some (_, _, _, opening, _) -> fail path opening "unclosed fence")
    | source :: rest -> (
        match active with
        | Some (c, n, info, opening, code) -> (
            match fence source with
            | Some (c', n', "") when Char.equal c c' && n' >= n ->
                let acc =
                  if String.equal info "ocaml" || String.is_prefix info ~prefix:"ocaml " then
                    {
                      path;
                      line = opening + 1;
                      code = String.concat ~sep:"\n" (List.rev code);
                      status = status ~path ~line:opening info;
                    }
                    :: acc
                  else acc
                in
                loop (number + 1) None acc rest
            | _ -> loop (number + 1) (Some (c, n, info, opening, source :: code)) acc rest)
        | None -> (
            match fence source with
            | Some (c, n, info) -> loop (number + 1) (Some (c, n, info, number, [])) acc rest
            | None ->
                let acc =
                  match String.substr_index source ~pattern:"Doc-check `" with
                  | None ->
                      if
                        String.is_substring source ~substring:"```ocaml doc-check="
                        || String.is_substring source ~substring:"~~~ocaml doc-check="
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
                      { path; line = number; code; status = Check id } :: acc
                in
                loop (number + 1) None acc rest))
  in
  loop 1 None [] lines

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
  List.filter_map blocks ~f:(fun block ->
      match block.status with
      | Check id -> Some (Printf.sprintf "%s: checked %s" block.path id)
      | Skip reason -> Some (Printf.sprintf "%s: excluded %s" block.path reason)
      | Unchecked -> None)
  |> List.sort ~compare:String.compare |> String.concat ~sep:"\n"
