(** The reader behind [test/operations/nonvirtual_code_inventory] (gh-ocannl-1015): where the
    virtualizer's rejection codes are minted, and which files name them.

    A code carries two facts a reader needs -- which pass raises it, and whether it can fire at all
    -- and the second is a function of PIPELINE ORDER, not of the raise site. gh-ocannl-483 put a
    rewrite tier ahead of [Low_level.optimize] and made a code everyone had described as defensive
    reachable; the claim that it could not fire had six live copies, and nothing listed them. So the
    set of codes is derived from the library sources, every file naming one is enumerated, and the
    enumeration is the checklist a pipeline-order change reads.

    {1 What a code is}

    A string literal shaped like a placement tag -- decimal digits, a colon, a kebab-case reason --
    lexically inside the scope of a local exception named [Non_virtual]
    ([let exception Non_virtual of string in ...]). That covers the literal at a [raise], the one
    handed to a helper that raises it ([~code:"..."]), and the one a handler records directly
    without raising, all of which are the same tag. The function a code is minted in is the
    innermost value binding enclosing the exception's declaration -- a [Non_virtual] declared anew
    inside that scope opens its own -- which is what decides its PHASE: the store-time check, or the
    consumption-time inliner. So a tag is minted in ONE function, or the provenance a node records
    cannot say which phase refused it, and the inventory refuses it. A tag computed at run time is
    not read.

    {1 What naming a code is}

    Two spellings, both mechanical:
    - the tag itself, exactly as minted, not continued by another tag character on its right nor
      preceded by an identifier character on its left;
    - the constructor's name, whitespace -- a line break included, since prose and comments wrap
      there -- and the decimal code, which is how a comment or a Markdown page cites one. Such a
      spelling whose number no raise site mints is a STALE reference and is refused: a retired code,
      or a tag of another family (a heuristic cap is not one of these codes).

    A tag spelled out is recognized only while some raise site still mints it: nothing tells a
    retired [N:reason] from a tag of another family, of which the library and its tests mint many.
    So a retired code's tag citations are not refused; they drop out of the golden, whose diff on
    the change that retires the code names every file that cited it -- the checklist, read at the
    moment it is needed.

    Bare numerals are not read: a paragraph that says "refused as 148 first" is found only through
    another spelling in the same file. The checklist is therefore a list of FILES; the codes beside
    each are the recognizable ones, a guide rather than a bound. *)

open Base
open Ppxlib

type code = { number : int; tag : string; source : string; minter : string }

let constructor = "Non_virtual"

(* [digits ":" lowercase-alnum ("-" lowercase-alnum)*], and the number it starts with. *)
let tag_number s =
  match String.lsplit2 s ~on:':' with
  | Some (digits, reason)
    when (not (String.is_empty digits))
         && String.for_all digits ~f:Char.is_digit
         && (not (String.is_empty reason))
         && Char.is_lowercase reason.[0]
         && (not (String.is_suffix reason ~suffix:"-"))
         && (not (String.is_substring reason ~substring:"--"))
         && String.for_all reason ~f:(fun c ->
             Char.is_lowercase c || Char.is_digit c || Char.equal c '-') ->
      Int.of_string_opt digits
  | _ -> None

let binding_name (vb : value_binding) =
  match vb.pvb_pat.ppat_desc with
  | Ppat_var { txt; _ } -> Some txt
  | Ppat_constraint ({ ppat_desc = Ppat_var { txt; _ }; _ }, _) -> Some txt
  | _ -> None

(** The codes one OCaml source mints, in source order, duplicates kept (the inventory merges them).
*)
let minted ~source content =
  let found = ref [] in
  let walker =
    object (self)
      inherit Ast_traverse.iter as super
      val mutable enclosing = "(top level)"

      (* The minter of the innermost [Non_virtual] scope the walk is in. A helper bound inside that
         scope raises the enclosing exception, so it does not move this; a [Non_virtual] declared
         anew inside the scope shadows it, and does. *)
      val mutable scope = None

      method! value_binding vb =
        let saved = enclosing in
        Option.iter (binding_name vb) ~f:(fun name -> enclosing <- name);
        super#value_binding vb;
        enclosing <- saved

      method! expression e =
        match e.pexp_desc with
        | Pexp_letexception (({ pext_name = { txt; _ }; _ } as ext), body)
          when String.equal txt constructor ->
            let saved = scope in
            self#extension_constructor ext;
            scope <- Some enclosing;
            self#expression body;
            scope <- saved
        | Pexp_constant (Pconst_string (s, _, _)) -> (
            match (scope, tag_number s) with
            | Some minter, Some number -> found := { number; tag = s; source; minter } :: !found
            | _ -> ())
        | _ -> super#expression e
    end
  in
  walker#structure (Parse.implementation (Lexing.from_string content));
  List.rev !found

(** Merged across sources: one entry per tag, ordered by number. *)
let merge codes =
  List.dedup_and_sort codes ~compare:(fun a b ->
      match Int.compare a.number b.number with
      | 0 -> (
          match String.compare a.tag b.tag with
          | 0 -> (
              match String.compare a.source b.source with
              | 0 -> String.compare a.minter b.minter
              | c -> c)
          | c -> c)
      | c -> c)

let is_ident_char c = Char.is_alphanum c || Char.equal c '_' || Char.equal c '\''
let is_tag_char c = Char.is_lowercase c || Char.is_digit c || Char.equal c '-'

(* The numbers of every [Non_virtual <whitespace> <digits>] in [text]. *)
let constructor_numbers text =
  let len = String.length text in
  let rec skip i ~f = if i < len && f text.[i] then skip (i + 1) ~f else i in
  String.substr_index_all text ~may_overlap:false ~pattern:constructor
  |> List.filter_map ~f:(fun i ->
      let after = i + String.length constructor in
      let digits = skip after ~f:Char.is_whitespace in
      let stop = skip digits ~f:Char.is_digit in
      if
        (i = 0 || not (is_ident_char text.[i - 1]))
        && digits > after && stop > digits
        && (stop = len || not (is_ident_char text.[stop]))
      then Int.of_string_opt (String.sub text ~pos:digits ~len:(stop - digits))
      else None)

let names_tag text tag =
  let len = String.length text and tlen = String.length tag in
  List.exists (String.substr_index_all text ~may_overlap:false ~pattern:tag) ~f:(fun i ->
      (i = 0 || not (is_ident_char text.[i - 1]))
      && (i + tlen = len || not (is_tag_char text.[i + tlen])))

(** The codes [text] names, as sorted distinct numbers, and the [Non_virtual N] spellings whose
    number no code has. *)
let mentions ~codes text =
  let known = Set.of_list (module Int) (List.map codes ~f:(fun c -> c.number)) in
  let spelled = constructor_numbers text in
  let by_tag = List.filter_map codes ~f:(fun c -> Option.some_if (names_tag text c.tag) c.number) in
  let named =
    List.filter spelled ~f:(Set.mem known) @ by_tag |> List.dedup_and_sort ~compare:Int.compare
  in
  let unknown =
    List.filter spelled ~f:(Fn.non (Set.mem known)) |> List.dedup_and_sort ~compare:Int.compare
  in
  (named, unknown)

(** [(tag, phase)] pairs of the top-level [phase_table] binding: a list literal of
    [("<tag>", <Constructor>)] tuples. [None] when there is no such binding or an element has
    another shape -- the table is read, never guessed at. *)
let phase_table content =
  let entry (e : expression) =
    match e.pexp_desc with
    | Pexp_tuple
        [
          { pexp_desc = Pexp_constant (Pconst_string (tag, _, _)); _ };
          { pexp_desc = Pexp_construct ({ txt = Lident phase; _ }, None); _ };
        ] ->
        Some (tag, phase)
    | _ -> None
  in
  let rec elements (e : expression) =
    match e.pexp_desc with
    | Pexp_construct ({ txt = Lident "[]"; _ }, None) -> Some []
    | Pexp_construct ({ txt = Lident "::"; _ }, Some { pexp_desc = Pexp_tuple [ hd; tl ]; _ }) -> (
        match (entry hd, elements tl) with Some x, Some xs -> Some (x :: xs) | _ -> None)
    | _ -> None
  in
  List.find_map
    (Parse.implementation (Lexing.from_string content))
    ~f:(fun item ->
      match item.pstr_desc with
      | Pstr_value (_, vbs) ->
          List.find_map vbs ~f:(fun vb ->
              match binding_name vb with
              | Some "phase_table" -> Some (elements vb.pvb_expr)
              | _ -> None)
      | _ -> None)
  |> Option.join

(** Everything the inventory refuses, given the merged [codes], the phase table read from
    [table_source] with the function each phase names, and [(path, named, unknown)] for each file
    read. *)
let violations ~codes ~table_source ~table ~phases ~files =
  let no_codes =
    if List.is_empty codes then
      [
        "no `Non_virtual` exception scope mints a code in the library sources: the reader is blind";
      ]
    else []
  in
  let ambiguous =
    List.group codes ~break:(fun a b -> a.number <> b.number)
    |> List.filter_map ~f:(fun group ->
        match List.dedup_and_sort (List.map group ~f:(fun c -> c.tag)) ~compare:String.compare with
        | _ :: _ :: _ as tags ->
            Some
              (Printf.sprintf "code %d is minted as %s: a numeric reference to it is ambiguous"
                 (List.hd_exn group).number (String.concat ~sep:" and " tags))
        | _ -> None)
  in
  let two_phases =
    List.sort codes ~compare:(fun a b -> String.compare a.tag b.tag)
    |> List.group ~break:(fun a b -> not (String.equal a.tag b.tag))
    |> List.filter_map ~f:(fun group ->
        match
          List.dedup_and_sort ~compare:String.compare
            (List.map group ~f:(fun c -> c.source ^ " " ^ c.minter))
        with
        | _ :: _ :: _ as minters ->
            Some
              (Printf.sprintf
                 "%s is minted in %s: the recorded provenance can no longer say which phase \
                  refused a node -- give each its own tag"
                 (List.hd_exn group).tag
                 (String.concat ~sep:" and " minters))
        | _ -> None)
  in
  let stale =
    List.concat_map files ~f:(fun (path, _, unknown) ->
        List.map unknown ~f:(fun n ->
            Printf.sprintf "%s: `%s %d` names no code a raise site mints" path constructor n))
  in
  (* The mention reader's own floor: a minting source names each of its codes by construction, so
     one it does not see there is the reader gone blind, not the source. *)
  let blind =
    List.filter_map codes ~f:(fun c ->
        match List.find files ~f:(fun (path, _, _) -> String.equal path c.source) with
        | Some (_, named, _) when List.mem named c.number ~equal:Int.equal -> None
        | _ ->
            Some
              (Printf.sprintf "%s mints %s, but the mention reader does not see it there" c.source
                 c.tag))
  in
  let table_problems =
    match table with
    | None ->
        [
          Printf.sprintf
            "%s: no `phase_table` binding of (\"<tag>\", <Phase>) literals to check against the \
             raise sites"
            table_source;
        ]
    | Some entries ->
        let per_entry =
          List.filter_map entries ~f:(fun (tag, phase) ->
              match
                ( List.filter codes ~f:(fun c -> String.equal c.tag tag),
                  List.Assoc.find phases ~equal:String.equal phase )
              with
              | [], _ ->
                  Some
                    (Printf.sprintf "%s: phase table entry %s is minted nowhere" table_source tag)
              | _, None ->
                  Some
                    (Printf.sprintf "%s: phase table entry %s names phase %s, which has no minter"
                       table_source tag phase)
              | minted, Some minter ->
                  if List.exists minted ~f:(fun c -> String.equal c.minter minter) then None
                  else
                    Some
                      (Printf.sprintf "%s: phase table puts %s at %s, but it is minted in %s"
                         table_source tag phase
                         (String.concat ~sep:" and "
                            (List.dedup_and_sort ~compare:String.compare
                               (List.map minted ~f:(fun c -> c.minter))))))
        in
        let stale_phases =
          List.filter_map phases ~f:(fun (phase, minter) ->
              if List.exists codes ~f:(fun c -> String.equal c.minter minter) then None
              else Some (Printf.sprintf "phase %s names %s, which mints no code" phase minter))
        in
        per_entry @ stale_phases
  in
  no_codes @ ambiguous @ two_phases @ stale @ blind @ table_problems

(** The [(prefix, reason)] exclusions no path in [paths] lives under. *)
let stale_records ~records paths =
  List.filter_map records ~f:(fun (prefix, _) ->
      if List.exists paths ~f:(String.is_prefix ~prefix) then None
      else Some (Printf.sprintf "%s: no file lives under this excluded prefix" prefix))
