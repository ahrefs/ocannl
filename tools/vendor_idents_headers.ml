(* Literal include traversal for the vendor-name harvest. The probe's header in vendor_idents.ml
   defines which source forms and SDK paths are in scope. *)

open Base

let is_ident_start c = Char.is_alpha c || Char.equal c '_'
let is_ident_char c = Char.is_alphanum c || Char.equal c '_'

(* [text] with comments, string and character literals blanked, newlines kept. *)
let blank_comments_and_literals text =
  let n = String.length text in
  let out = Bytes.of_string text in
  let blank i = if not (Char.equal (Bytes.get out i) '\n') then Bytes.set out i ' ' in
  let rec code i =
    if i >= n then ()
    else
      match text.[i] with
      | '/' when i + 1 < n && Char.equal text.[i + 1] '/' -> line_comment i
      | '/' when i + 1 < n && Char.equal text.[i + 1] '*' ->
          blank i;
          blank (i + 1);
          block_comment (i + 2)
      | ('"' | '\'') as q -> literal q (i + 1)
      | _ -> code (i + 1)
  and line_comment i =
    if i >= n || Char.equal text.[i] '\n' then code i
    else (
      blank i;
      line_comment (i + 1))
  and block_comment i =
    if i >= n then ()
    else if Char.equal text.[i] '*' && i + 1 < n && Char.equal text.[i + 1] '/' then (
      blank i;
      blank (i + 1);
      code (i + 2))
    else (
      blank i;
      block_comment (i + 1))
  and literal q i =
    if i >= n then ()
    else if Char.equal text.[i] '\\' && i + 1 < n then (
      blank i;
      blank (i + 1);
      literal q (i + 2))
    else if Char.equal text.[i] q then code (i + 1)
    else if Char.equal text.[i] '\n' then code i
    else (
      blank i;
      literal q (i + 1))
  in
  code 0;
  Bytes.to_string out

(* The identifier tokens of [text] (already blanked): a number literal is consumed whole, so its
   exponent or suffix letters are not read as a name. *)
let identifiers text =
  let n = String.length text in
  let acc = ref [] in
  let rec scan i =
    if i >= n then ()
    else if is_ident_start text.[i] then (
      let j = ref i in
      while !j < n && is_ident_char text.[!j] do
        Int.incr j
      done;
      acc := String.sub text ~pos:i ~len:(!j - i) :: !acc;
      scan !j)
    else if Char.is_digit text.[i] then (
      let j = ref i in
      while !j < n && (is_ident_char text.[!j] || Char.equal text.[!j] '.') do
        Int.incr j
      done;
      scan !j)
    else scan (i + 1)
  in
  scan 0;
  !acc

(* The [#include] / [#include_next] targets of [text] (raw, so the quotes survive): [`Quoted] for
   ["x"], [`Angled] for [<x>]. *)
let includes text =
  List.filter_map (String.split_lines text) ~f:(fun line ->
      let line = String.strip line in
      match String.chop_prefix line ~prefix:"#" with
      | None -> None
      | Some rest -> (
          let rest = String.lstrip rest in
          let rest =
            match String.chop_prefix rest ~prefix:"include_next" with
            | Some r -> Some r
            | None -> String.chop_prefix rest ~prefix:"include"
          in
          match Option.map rest ~f:String.strip with
          | Some r when String.length r >= 2 -> (
              let close = function '"' -> Some '"' | '<' -> Some '>' | _ -> None in
              match close r.[0] with
              | None -> None
              | Some c -> (
                  match String.index_from r 1 c with
                  | Some j ->
                      let target = String.sub r ~pos:1 ~len:(j - 1) in
                      Some ((if Char.equal c '"' then `Quoted else `Angled), target)
                  | None -> None))
          | _ -> None))

let is_file p = Stdlib.Sys.file_exists p && not (Stdlib.Sys.is_directory p)

(* Count each file identity once, and traverse its includes from each opening directory. *)
let header_closure ~dirs ~entries ~header_identity =
  let seen = Hash_set.Poly.create () in
  let parsed = Hashtbl.create (module String) in
  let unresolved = Hash_set.create (module String) in
  let tokens = Hash_set.create (module String) in
  let resolve ~from (kind, target) =
    let local =
      match (kind, from) with `Quoted, Some dir -> [ Stdlib.Filename.concat dir target ] | _ -> []
    in
    List.find (local @ List.map dirs ~f:(fun d -> Stdlib.Filename.concat d target)) ~f:is_file
    |> Option.bind ~f:(fun path -> Option.map (header_identity path) ~f:(fun key -> (path, key)))
  in
  let rec visit ~from inc =
    match resolve ~from inc with
    | None -> Hash_set.add unresolved (snd inc)
    | Some (_, key) when Hash_set.mem seen key -> ()
    | Some (path, ((file, _) as key)) ->
        Hash_set.add seen key;
        let targets =
          Hashtbl.find_or_add parsed file ~default:(fun () ->
              let raw = Stdio.In_channel.read_all path in
              List.iter (identifiers (blank_comments_and_literals raw)) ~f:(Hash_set.add tokens);
              includes raw)
        in
        List.iter targets ~f:(visit ~from:(Some (Stdlib.Filename.dirname path)))
  in
  List.iter entries ~f:(visit ~from:None);
  (Hashtbl.length parsed, Hash_set.to_list unresolved, tokens)

let hip_sdk_header_identity dirs =
  let canonical path = try Some (Unix.realpath path) with Unix.Unix_error _ -> None in
  let subtrees =
    List.concat_map dirs ~f:(fun dir ->
        List.filter_map [ "hip"; "hsa"; "rocwmma" ] ~f:(fun subtree ->
            let path = Stdlib.Filename.concat dir subtree in
            if Stdlib.Sys.file_exists path && Stdlib.Sys.is_directory path then canonical path
            else None))
  in
  fun path ->
    Option.bind (canonical path) ~f:(fun file ->
        if
          List.exists subtrees ~f:(fun root ->
              String.is_prefix file ~prefix:(Stdlib.Filename.concat root ""))
        then
          (* The same symlink target can get different quoted includes from different
             directories. *)
          Option.map (canonical (Stdlib.Filename.dirname path)) ~f:(fun dir -> (file, dir))
        else None)
