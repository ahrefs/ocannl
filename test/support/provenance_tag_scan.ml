(** The reader behind [test/operations/provenance_tag_inventory]: where every placement-provenance
    tag is minted, and which files cite one (gh-ocannl-1015 for the virtualizer's rejection codes,
    gh-ocannl-1081 for every other family).

    A tag carries facts a reader needs -- which pass records it, and for a rejection code whether it
    can fire at all, the second being a function of PIPELINE ORDER rather than of the raise site
    (gh-ocannl-483 made a code everyone had described as defensive reachable, and the claim that it
    could not fire had six live copies). Tags are numbered by folklore across nine modules, so a
    number could mean two things with nobody noticing, and a retired [N:reason] left in prose read
    exactly like a live one. So the set of tags is derived from the sources, every file citing one
    is enumerated, and a citation no source mints is refused.

    {1 The families}

    The families are read off [Tnode.provenance] itself, never listed: its declaration and its
    renderer are the owner of what a tag can be.
    - A constructor with no argument ([Visit_cap], ...) mints ONE tag: the string literal its case
      of [provenance_to_string] returns, and no other constructor or family may mint that text, or a
      printed provenance could not say which decision it records.
    - A constructor carrying a [string] ([Site]) is open: it mints every string literal it is
      applied to ([Tn.Site], [Ir.Tnode.Site], or unqualified), and a literal that is not a tag is
      refused rather than skipped. A constructor of the same name that another module declares with
      a [string] argument ([Operand_key_scan]'s [Site of string]) is not a provenance: applied
      qualified by that module, directly or through a module binding of the source, or unqualified
      inside it, it mints nothing. Those modules are derived from their declarations; the owner's
      constructor is assumed everywhere else, since opens and aliases reach it in ways a reader of
      one file cannot follow.
    - A constructor carrying provenances only ([Refined]) composes and mints nothing. Any other
      shape is refused as unread.
    - A string constructor applied to a VARIABLE opens a relayed family when that variable is the
      payload a handler caught: [with Non_virtual i -> ... (Site i)] makes every tag literal in the
      scope of the local [let exception Non_virtual of string] a tag of the family "Site via
      Non_virtual". That covers the literal at a [raise], the one handed to a helper that raises it,
      and the one a handler records directly. The function such a tag is minted in is the innermost
      value binding enclosing the exception's declaration -- a declaration anew inside the scope
      opens its own -- which is what decides its PHASE, so a relayed tag minted in two functions is
      refused. A tag computed at run time is not read.

    The minter of a tag of the other families is the innermost value binding around the literal.

    {1 Library tags and test tags}

    A source under a [test] directory mints no library tag. A tag it applies a string constructor to
    that some library source mints is a citation; any other is the test's own, and must be spelled
    [N:test-<reason>] -- else a test constructing a RETIRED library tag would silently adopt it. The
    [test-] prefix is also what keeps a test tag out of the library's number space: its reason
    already says whose it is, so it may reuse a library number. A library tag may not use the
    prefix.

    {1 What citing a tag is}

    A tag's reason is kebab-case with at least two words; the reader refuses a minted tag with a
    one-word reason, because that is what lets a citation be told from other colon-joined text (a
    PCI address, a named dimension, a shell [case] label). Three spellings cite one, all mechanical:
    - the tag itself: decimal digits not continuing a word, a path, a version or another number (the
      character before them is none of [A-Za-z0-9_'.:/-]), a colon, a kebab-case reason of two or
      more words, and no identifier character or hyphen after it. EVERY such token is read, which is
      what detects a retired tag: one no source mints is a STALE citation and is refused;
    - a relaying exception's name, whitespace -- a line break included, since prose wraps there --
      and a number, as in [Non_virtual 13]: stale unless that family mints the number;
    - the word [provenance], whitespace, optional Markdown emphasis or backquotes, and a number, as
      in "provenance **41**": stale unless some library tag has the number.

    Bare numerals are not read: a paragraph that says "refused as 148 first" is found only through
    another spelling in the same file. The checklist is therefore a list of FILES; the numbers
    beside each are the recognizable ones, a guide rather than a bound.

    {1 Numbers}

    A number minted under two library tags makes every numeric citation of it ambiguous. The
    collisions that predate this reader are PINNED by the inventory, tag set and all; a new one, a
    third tag on a pinned number, or a pin that no longer collides is refused. *)

open Base
open Ppxlib

type family =
  | Rendered of string  (** A nullary constructor, rendered by the renderer's case for it. *)
  | Applied of string  (** A string-carrying constructor applied to a literal. *)
  | Relayed of { exn : string; via : string }
      (** Literals in the scope of a local exception whose payload a handler hands to [via]. *)

let family_label = function
  | Rendered c | Applied c -> c
  | Relayed { exn; via } -> via ^ " via " ^ exn

type mint = { number : int; tag : string; family : family; source : string; minter : string }

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

let reason tag = Option.value_map (String.lsplit2 tag ~on:':') ~default:"" ~f:snd

(** A reason of two or more words: the shape the citation reader recognizes. *)
let citable tag = String.mem (reason tag) '-'

let test_prefix = "test-"
let is_test_tag tag = String.is_prefix (reason tag) ~prefix:test_prefix

let is_test_source path =
  String.is_prefix path ~prefix:"test/" || String.is_substring path ~substring:"/test/"

let binding_name (vb : value_binding) =
  match vb.pvb_pat.ppat_desc with
  | Ppat_var { txt; _ } -> Some txt
  | Ppat_constraint ({ ppat_desc = Ppat_var { txt; _ }; _ }, _) -> Some txt
  | _ -> None

let last_name (lid : longident) =
  match lid with Lident s | Ldot (_, s) -> Some s | Lapply _ -> None

let parse content = Parse.implementation (Lexing.from_string content)

(** {1 The type} *)

type shape = {
  rendered : string list;  (** Nullary constructors. *)
  carriers : string list;  (** Constructors carrying one [string]. *)
  composite : string list;  (** Constructors carrying provenances only. *)
  unread : string list;  (** Any other shape: refused. *)
}

(** The shape of the [type_name] declaration in [content], if there is one. *)
let type_shape ~type_name content =
  List.find_map (parse content) ~f:(fun item ->
      match item.pstr_desc with
      | Pstr_type (_, decls) ->
          List.find_map decls ~f:(fun decl ->
              match decl.ptype_kind with
              | Ptype_variant cds when String.equal decl.ptype_name.txt type_name ->
                  let classify (cd : constructor_declaration) =
                    let is_constr name (t : core_type) =
                      match t.ptyp_desc with
                      | Ptyp_constr ({ txt = Lident n; _ }, []) -> String.equal n name
                      | _ -> false
                    in
                    match cd.pcd_args with
                    | Pcstr_tuple [] -> `Rendered
                    | Pcstr_tuple [ t ] when is_constr "string" t -> `Carrier
                    | Pcstr_tuple ts when List.for_all ts ~f:(is_constr type_name) -> `Composite
                    | _ -> `Unread
                  in
                  let pick kind =
                    List.filter_map cds ~f:(fun cd ->
                        Option.some_if (Poly.equal (classify cd) kind) cd.pcd_name.txt)
                  in
                  Some
                    {
                      rendered = pick `Rendered;
                      carriers = pick `Carrier;
                      composite = pick `Composite;
                      unread = pick `Unread;
                    }
              | _ -> None)
      | _ -> None)

(** [(constructor, tag)] for each case [C -> "tag"] of the top-level [renderer] binding. *)
let renderings ~renderer ~source content =
  let found = ref [] in
  let walker =
    object
      inherit Ast_traverse.iter as super

      method! case c =
        (match (c.pc_lhs.ppat_desc, c.pc_rhs.pexp_desc) with
        | Ppat_construct ({ txt; _ }, None), Pexp_constant (Pconst_string (s, _, _)) -> (
            match (last_name txt, tag_number s) with
            | Some constructor, Some number ->
                found :=
                  { number; tag = s; family = Rendered constructor; source; minter = renderer }
                  :: !found
            | _ -> ())
        | _ -> ());
        super#case c
    end
  in
  List.iter (parse content) ~f:(fun item ->
      match item.pstr_desc with
      | Pstr_value (_, vbs) ->
          List.iter vbs ~f:(fun vb ->
              match binding_name vb with
              | Some name when String.equal name renderer -> walker#expression vb.pvb_expr
              | _ -> ())
      | _ -> ());
  List.rev !found

(** {1 The sources} *)

type read = {
  path : string;  (** The source read. *)
  mints : mint list;
      (** [Applied] literals, and the tag literals of every local-exception scope as [Relayed] with
          [via] empty: which exceptions relay is decided across sources, by {!relays}. *)
  relays : (string * string) list;  (** [(exception, carrier)] handlers in this source. *)
  malformed : string list;  (** String literals a carrier is applied to that are no tag. *)
}

(** The module a source file defines: [arrayjit/lib/tnode.ml] is [Tnode]. *)
let module_of_path path =
  String.capitalize (Stdlib.Filename.remove_extension (Stdlib.Filename.basename path))

(* The module each module name [structure] binds resolves to, at any depth: [module Scan =
   Test_utils.Operand_key_scan] maps [Scan] to [Operand_key_scan]. *)
let module_bindings structure =
  let found = ref [] in
  let note name (me : module_expr) =
    match (name, me.pmod_desc) with
    | Some name, Pmod_ident { txt; _ } ->
        Option.iter (last_name txt) ~f:(fun target -> found := (name, target) :: !found)
    | _ -> ()
  in
  let walker =
    object
      inherit Ast_traverse.iter as super

      method! module_binding mb =
        note mb.pmb_name.txt mb.pmb_expr;
        super#module_binding mb

      method! expression e =
        (match e.pexp_desc with Pexp_letmodule ({ txt; _ }, me, _) -> note txt me | _ -> ());
        super#expression e
    end
  in
  walker#structure structure;
  !found

(* The constructor names [structure] declares with a single [string] argument. *)
let own_string_constructors structure =
  List.concat_map structure ~f:(fun item ->
      match item.pstr_desc with
      | Pstr_type (_, decls) ->
          List.concat_map decls ~f:(fun decl ->
              match decl.ptype_kind with
              | Ptype_variant cds ->
                  List.filter_map cds ~f:(fun cd ->
                      match cd.pcd_args with
                      | Pcstr_tuple
                          [ { ptyp_desc = Ptyp_constr ({ txt = Lident "string"; _ }, []); _ } ] ->
                          Some cd.pcd_name.txt
                      | _ -> None)
              | _ -> [])
      | _ -> [])

(** Whether [content], a source other than the owner's, declares a [string]-carrying constructor
    under a carrier's name -- one that is NOT a provenance, however it is spelled. *)
let declares_own_carrier ~carriers content =
  List.exists (own_string_constructors (parse content)) ~f:(List.mem carriers ~equal:String.equal)

(** One OCaml source's mints and relays, in source order, duplicates kept.

    A constructor with a carrier's name is taken for the owner's unless it names another: a module
    in [foreign] declares its own [string]-carrying constructor of that name (as [Operand_key_scan]
    declares [Site of string]), so an application qualified by it -- directly or through a module
    the source binds to it ([module Scan = Test_utils.Operand_key_scan]) -- is not a provenance, nor
    is an unqualified one in [foreign]'s own source. The owner's constructors are reached through
    aliases and opens this reader cannot follow, which is why the rule names what is excluded rather
    than what is included. *)
let read_source ~carriers ?(foreign = []) ~source content =
  let mints = ref [] and relays = ref [] and malformed = ref [] in
  let structure = parse content in
  let bindings = module_bindings structure in
  let resolve q = Option.value (List.Assoc.find bindings q ~equal:String.equal) ~default:q in
  let is_foreign m = List.mem foreign m ~equal:String.equal in
  let is_carrier (lid : longident) =
    match lid with
    | Lident c ->
        List.mem carriers c ~equal:String.equal && not (is_foreign (module_of_path source))
    | Ldot (q, c) ->
        List.mem carriers c ~equal:String.equal
        && not (Option.value_map (last_name q) ~default:false ~f:(fun q -> is_foreign (resolve q)))
    | Lapply _ -> false
  in
  (* The carriers applied to the variable [v] anywhere in [e]. *)
  let carriers_of_var v e =
    let hits = ref [] in
    let finder =
      object
        inherit Ast_traverse.iter as super

        method! expression e =
          (match e.pexp_desc with
          | Pexp_construct ({ txt; _ }, Some { pexp_desc = Pexp_ident { txt = Lident x; _ }; _ })
            when String.equal x v && is_carrier txt ->
              Option.iter (last_name txt) ~f:(fun c -> hits := c :: !hits)
          | _ -> ());
          super#expression e
      end
    in
    finder#expression e;
    !hits
  in
  let walker =
    object (self)
      inherit Ast_traverse.iter as super
      val mutable enclosing = "(top level)"

      (* The innermost local string exceptions the walk is in, each with its minter. A helper bound
         inside a scope raises the enclosing exception, so it does not move this; an exception
         declared anew inside the scope shadows it, and does. *)
      val mutable scopes : (string * string) list = []

      method! value_binding vb =
        let saved = enclosing in
        Option.iter (binding_name vb) ~f:(fun name -> enclosing <- name);
        super#value_binding vb;
        enclosing <- saved

      method! case c =
        (* A handler [E v -> ... (Carrier v)], directly or under [exception]. *)
        let rec caught (p : pattern) =
          match p.ppat_desc with
          | Ppat_construct ({ txt; _ }, Some (_, { ppat_desc = Ppat_var { txt = v; _ }; _ })) ->
              Option.value_map (last_name txt) ~default:[] ~f:(fun e -> [ (e, v) ])
          | Ppat_exception p | Ppat_alias (p, _) | Ppat_constraint (p, _) -> caught p
          | Ppat_or (a, b) -> caught a @ caught b
          | _ -> []
        in
        List.iter (caught c.pc_lhs) ~f:(fun (e, v) ->
            List.iter (carriers_of_var v c.pc_rhs) ~f:(fun carrier ->
                relays := (e, carrier) :: !relays));
        super#case c

      method! expression e =
        match e.pexp_desc with
        | Pexp_letexception
            ( ({ pext_name = { txt; _ }; pext_kind = Pext_decl (_, Pcstr_tuple [ _ ], None); _ } as
               ext),
              body ) ->
            let saved = scopes in
            self#extension_constructor ext;
            scopes <-
              (txt, enclosing) :: List.filter scopes ~f:(fun (n, _) -> not (String.equal n txt));
            self#expression body;
            scopes <- saved
        | Pexp_construct
            ({ txt; _ }, Some { pexp_desc = Pexp_constant (Pconst_string (s, _, _)); _ })
          when is_carrier txt -> (
            (match (last_name txt, tag_number s) with
            | Some carrier, Some number ->
                mints :=
                  { number; tag = s; family = Applied carrier; source; minter = enclosing }
                  :: !mints
            | _ -> malformed := s :: !malformed);
            (* The same literal is also in any exception scope around it. *)
            match tag_number s with
            | Some number -> self#scoped number s
            | None -> ())
        | Pexp_constant (Pconst_string (s, _, _)) -> (
            match tag_number s with Some number -> self#scoped number s | None -> ())
        | _ -> super#expression e

      method scoped number tag =
        List.iter scopes ~f:(fun (exn, minter) ->
            mints := { number; tag; family = Relayed { exn; via = "" }; source; minter } :: !mints)
    end
  in
  walker#structure structure;
  {
    path = source;
    mints = List.rev !mints;
    relays = List.rev !relays;
    malformed = List.rev !malformed;
  }

(** The relaying exceptions, [(exception, carrier)], deduplicated. *)
let relays reads =
  List.concat_map reads ~f:(fun r -> r.relays) |> List.dedup_and_sort ~compare:Poly.compare

(** The mints of [reads], each scope literal kept only under the exceptions that relay, and bound to
    the carrier relaying it. *)
let resolve ~relays reads =
  List.concat_map reads ~f:(fun r ->
      List.concat_map r.mints ~f:(fun m ->
          match m.family with
          | Relayed { exn; _ } ->
              List.filter_map relays ~f:(fun (e, via) ->
                  Option.some_if (String.equal e exn) { m with family = Relayed { exn; via } })
          | _ -> [ m ]))

let compare_mint a b =
  Poly.compare
    (a.number, a.tag, a.family, a.source, a.minter)
    (b.number, b.tag, b.family, b.source, b.minter)

(** Merged across sources: one entry per tag, family and site, ordered by number. *)
let merge mints = List.dedup_and_sort mints ~compare:compare_mint

(** {1 Citations} *)

let is_ident_char c = Char.is_alphanum c || Char.equal c '_' || Char.equal c '\''
let is_tag_char c = Char.is_lowercase c || Char.is_digit c || Char.equal c '-'

let tag_tokens text =
  let len = String.length text in
  let rec skip i ~f = if i < len && f text.[i] then skip (i + 1) ~f else i in
  let rec go i acc =
    if i >= len then List.rev acc
    else if
      Char.is_digit text.[i]
      && (i = 0
         ||
         let c = text.[i - 1] in
         not (is_ident_char c || List.mem [ '.'; ':'; '/'; '-' ] c ~equal:Char.equal))
    then
      let colon = skip i ~f:Char.is_digit in
      let stop =
        if colon < len && Char.equal text.[colon] ':' then skip (colon + 1) ~f:is_tag_char
        else colon
      in
      let token = String.sub text ~pos:i ~len:(stop - i) in
      let bounded = stop = len || not (is_ident_char text.[stop]) in
      let acc =
        if bounded && stop > colon + 1 && Option.is_some (tag_number token) && citable token then
          token :: acc
        else acc
      in
      go (Int.max stop (i + 1)) acc
    else go (i + 1) acc
  in
  go 0 []

(* The numbers of every [<word> <whitespace> [emphasis] <digits>] in [text]. *)
let spelled_numbers ~word text =
  let len = String.length text in
  let rec skip i ~f = if i < len && f text.[i] then skip (i + 1) ~f else i in
  let emphasis c = Char.equal c '*' || Char.equal c '`' in
  String.substr_index_all text ~may_overlap:false ~pattern:word
  |> List.filter_map ~f:(fun i ->
      let after = i + String.length word in
      let spaced = skip after ~f:Char.is_whitespace in
      let digits = if String.equal word "provenance" then skip spaced ~f:emphasis else spaced in
      let stop = skip digits ~f:Char.is_digit in
      if
        (i = 0 || not (is_ident_char text.[i - 1]))
        && spaced > after && stop > digits
        && (stop = len || not (is_ident_char text.[stop]))
      then Int.of_string_opt (String.sub text ~pos:digits ~len:(stop - digits))
      else None)

let provenance_word = "provenance"

type mention = {
  named : int list;  (** The library tag numbers [text] cites, sorted and distinct. *)
  stale : string list;  (** Each citation no source mints, as spelled. *)
}

(** What [text] cites, given every library mint and the test-owned tags. *)
let mentions ~(mints : mint list) ~test_tags text =
  let library = List.filter mints ~f:(fun m -> not (is_test_source m.source)) in
  let library_tags = Set.of_list (module String) (List.map library ~f:(fun m -> m.tag)) in
  let known_tags = Set.union library_tags (Set.of_list (module String) test_tags) in
  let numbers ms = Set.of_list (module Int) (List.map ms ~f:(fun m -> m.number)) in
  let all_numbers = numbers library in
  let exceptions =
    List.filter_map library ~f:(fun m ->
        match m.family with Relayed { exn; _ } -> Some exn | _ -> None)
    |> List.dedup_and_sort ~compare:String.compare
  in
  let tokens = tag_tokens text in
  let by_tag =
    List.filter_map tokens ~f:(fun t -> if Set.mem library_tags t then tag_number t else None)
  in
  let stale_tags = List.filter tokens ~f:(Fn.non (Set.mem known_tags)) in
  let spelled =
    (provenance_word, all_numbers)
    :: List.map exceptions ~f:(fun exn ->
        ( exn,
          numbers
            (List.filter library ~f:(fun m ->
                 match m.family with Relayed r -> String.equal r.exn exn | _ -> false)) ))
  in
  let named_spelled, stale_spelled =
    List.fold spelled ~init:([], []) ~f:(fun (named, stale) (word, known) ->
        let found = spelled_numbers ~word text in
        ( List.filter found ~f:(Set.mem known) @ named,
          List.filter_map found ~f:(fun n ->
              Option.some_if (not (Set.mem known n)) (Printf.sprintf "`%s %d`" word n))
          @ stale ))
  in
  {
    named = List.dedup_and_sort (by_tag @ named_spelled) ~compare:Int.compare;
    stale =
      List.dedup_and_sort ~compare:String.compare
        (List.map stale_tags ~f:(fun t -> "`" ^ t ^ "`") @ stale_spelled);
  }

(** {1 The phase table} *)

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
  List.find_map (parse content) ~f:(fun item ->
      match item.pstr_desc with
      | Pstr_value (_, vbs) ->
          List.find_map vbs ~f:(fun vb ->
              match binding_name vb with
              | Some "phase_table" -> Some (elements vb.pvb_expr)
              | _ -> None)
      | _ -> None)
  |> Option.join

(** {1 Refusals} *)

(** The library numbers minted under more than one tag, each with its sorted tags. *)
let collisions (mints : mint list) =
  List.filter mints ~f:(fun m -> not (is_test_source m.source))
  |> List.map ~f:(fun m -> (m.number, m.tag))
  |> List.dedup_and_sort ~compare:Poly.compare
  |> List.group ~break:(fun (a, _) (b, _) -> a <> b)
  |> List.filter_map ~f:(function
    | (n, _) :: _ :: _ as group -> Some (n, List.map group ~f:snd)
    | _ -> None)

(** What the families themselves refuse: an unread constructor, a rendering missing or doubled, an
    open family minting nothing. *)
let family_violations ~type_source ~shape ~(mints : mint list) =
  match shape with
  | None ->
      [
        Printf.sprintf "%s: no variant `provenance` declaration: the families cannot be derived"
          type_source;
      ]
  | Some shape ->
      List.map shape.unread ~f:(fun c ->
          Printf.sprintf
            "%s: constructor %s carries neither nothing, one string, nor provenances only: the \
             reader cannot say what it mints"
            type_source c)
      @ List.filter_map shape.rendered ~f:(fun c ->
          match
            List.filter mints ~f:(fun m -> Poly.equal m.family (Rendered c))
            |> List.map ~f:(fun m -> m.tag)
            |> List.dedup_and_sort ~compare:String.compare
          with
          | [ _ ] -> None
          | [] -> Some (Printf.sprintf "%s: constructor %s has no tag rendering" type_source c)
          | tags ->
              Some
                (Printf.sprintf "%s: constructor %s renders as %s" type_source c
                   (String.concat ~sep:" and " tags)))
      @ List.filter_map shape.carriers ~f:(fun c ->
          if List.exists mints ~f:(fun m -> Poly.equal m.family (Applied c)) then None
          else Some (Printf.sprintf "no source applies %s to a tag literal: the reader is blind" c))

(** Everything else the inventory refuses. [mints] are resolved and merged, from library and test
    sources alike; [malformed] is [(source, literal)] per carrier applied to a string that is no
    tag; [pinned] is [(number, tags)] per tolerated collision; [files] is [(path, mention)] per file
    read. *)
let violations ?(malformed = []) ~(mints : mint list) ~pinned ~files () =
  let library, tests = List.partition_tf mints ~f:(fun m -> not (is_test_source m.source)) in
  let library_tags = Set.of_list (module String) (List.map library ~f:(fun m -> m.tag)) in
  let one_word =
    List.filter_map mints ~f:(fun m ->
        Option.some_if
          (not (citable m.tag))
          (Printf.sprintf
             "%s mints %s: a one-word reason cannot be told from other colon-joined text in prose \
              -- give it at least two kebab-case words"
             m.source m.tag))
  in
  let not_tags =
    List.map malformed ~f:(fun (source, literal) ->
        Printf.sprintf
          "%s applies a provenance carrier to %S, which is not a tag (<digits>:<kebab-reason>): \
           the inventory can neither list it nor check its number"
          source literal)
  in
  (* A constructor's rendering is what a printed provenance says; a second minter of the same text
     makes the two decisions indistinguishable wherever it is printed. *)
  let shared_renderings =
    List.filter_map library ~f:(fun m ->
        match m.family with
        | Rendered c ->
            let others =
              List.filter library ~f:(fun o ->
                  String.equal o.tag m.tag && not (Poly.equal o.family m.family))
              |> List.map ~f:(fun o -> family_label o.family ^ " in " ^ o.source)
              |> List.dedup_and_sort ~compare:String.compare
            in
            if List.is_empty others then None
            else
              Some
                (Printf.sprintf
                   "%s is %s's rendering, and is also minted by %s: a printed provenance can no \
                    longer say which decision it records"
                   m.tag c
                   (String.concat ~sep:" and " others))
        | _ -> None)
  in
  let test_prefixed =
    List.filter_map library ~f:(fun m ->
        Option.some_if (is_test_tag m.tag)
          (Printf.sprintf "%s mints %s: the `%s` prefix is reserved for a test's own tags" m.source
             m.tag test_prefix))
  in
  let unowned =
    List.filter_map tests ~f:(fun m ->
        Option.some_if
          ((not (Set.mem library_tags m.tag)) && not (is_test_tag m.tag))
          (Printf.sprintf
             "%s constructs %s, which no library source mints: a retired tag, or a test's own tag \
              not spelled `N:%s<reason>`"
             m.source m.tag test_prefix))
  in
  let found = collisions mints in
  let equal_collision = Poly.equal in
  let collided =
    List.filter_map found ~f:(fun ((n, tags) as c) ->
        if List.mem pinned c ~equal:equal_collision then None
        else
          Some
            (Printf.sprintf
               "number %d is minted as %s: a numeric citation of it is ambiguous -- give the new \
                tag a free number"
               n (String.concat ~sep:" and " tags)))
  in
  let stale_pins =
    List.filter_map pinned ~f:(fun ((n, tags) as c) ->
        if List.mem found c ~equal:equal_collision then None
        else
          Some
            (Printf.sprintf
               "the pinned collision of %d on %s no longer holds as pinned: update the pin" n
               (String.concat ~sep:" and " tags)))
  in
  let two_phases =
    List.filter_map library ~f:(fun m ->
        match m.family with Relayed r -> Some (r.exn, m) | _ -> None)
    |> List.sort ~compare:(fun (e1, a) (e2, b) -> Poly.compare (e1, a.tag) (e2, b.tag))
    |> List.group ~break:(fun (e1, a) (e2, b) ->
        not (String.equal e1 e2 && String.equal a.tag b.tag))
    |> List.filter_map ~f:(fun group ->
        match
          List.dedup_and_sort ~compare:String.compare
            (List.map group ~f:(fun (_, m) -> m.source ^ " " ^ m.minter))
        with
        | _ :: _ :: _ as minters ->
            Some
              (Printf.sprintf
                 "%s is minted in %s: the recorded provenance can no longer say which phase \
                  refused a node -- give each its own tag"
                 (snd (List.hd_exn group)).tag
                 (String.concat ~sep:" and " minters))
        | _ -> None)
  in
  let stale =
    List.concat_map files ~f:(fun (path, mention) ->
        List.map mention.stale ~f:(fun s ->
            Printf.sprintf "%s: %s cites a tag no source mints" path s))
  in
  (* The citation reader's own floor: a minting source cites each of its tags by construction, so
     one it does not see there is the reader gone blind, not the source. *)
  let blind =
    List.filter_map library ~f:(fun m ->
        match List.find files ~f:(fun (path, _) -> String.equal path m.source) with
        | Some (_, mention) when List.mem mention.named m.number ~equal:Int.equal -> None
        | _ ->
            Some
              (Printf.sprintf "%s mints %s, but the citation reader does not see it there" m.source
                 m.tag))
    |> List.dedup_and_sort ~compare:String.compare
  in
  not_tags @ shared_renderings @ one_word @ test_prefixed @ unowned @ collided @ stale_pins
  @ two_phases @ stale @ blind

(** What the phase table refuses, given the relayed family [exn] whose phases [phases] names. *)
let table_violations ~(mints : mint list) ~exn ~table_source ~table ~phases =
  let codes =
    List.filter mints ~f:(fun m ->
        (not (is_test_source m.source))
        && match m.family with Relayed r -> String.equal r.exn exn | _ -> false)
  in
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
                Some (Printf.sprintf "%s: phase table entry %s is minted nowhere" table_source tag)
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
            else Some (Printf.sprintf "phase %s names %s, which mints no %s code" phase minter exn))
      in
      per_entry @ stale_phases

(** The [(prefix, reason)] exclusions no path in [paths] lives under. *)
let stale_records ~records paths =
  List.filter_map records ~f:(fun (prefix, _) ->
      if List.exists paths ~f:(String.is_prefix ~prefix) then None
      else Some (Printf.sprintf "%s: no file lives under this excluded prefix" prefix))
