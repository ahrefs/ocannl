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

    This is a SYNTACTIC reader of OCaml, and it holds one rule throughout: it accepts the canonical
    shapes and reads anything else as "does not hold", on the side where the answer is loud. A
    renderer case it cannot read leaves its constructor unrendered, which the inventory refuses; a
    relay it cannot follow mints nothing, so the scope's tags vanish and every citation of them and
    the phase table refuse. What it never does is guess a relay or a rendering into existence.

    - A constructor with no argument ([Visit_cap], ...) mints ONE tag: the string literal its case
      of [provenance_to_string] returns, and no other constructor or family may mint that text, or a
      printed provenance could not say which decision it records.
    - A constructor carrying a [string] ([Site]) is open: it mints every string literal it is
      applied to ([Tn.Site], [Ir.Tnode.Site], or unqualified, a type constraint on the argument
      peeled), and a literal that is not a tag is refused rather than skipped. The renderer must
      return its string unchanged. A constructor of the same name that another module declares with
      a [string] argument ([Operand_key_scan]'s [Site of string]) is not a provenance: applied
      qualified by that module -- directly or through module bindings, resolved where they are in
      scope -- unqualified inside it, or unqualified where an [open] or [include] of it is in scope,
      it mints nothing; the same goes for a module the source itself declares with such a
      constructor, or includes a module that does. Those modules are derived from their
      declarations; the owner's constructor is assumed everywhere else, since opens and aliases
      reach it in ways a reader of one file cannot follow.
    - A constructor carrying provenances only ([Refined]) composes and mints nothing; its case of
      the renderer must return a concatenation of string literals and recursive calls rendering each
      of its arguments exactly once, in order, or a printed provenance could drop or reorder a
      recorded tag. Any other shape is refused as unread.
    - Each of those renderer checks reads a constructor's ONE unguarded case of the renderer's
      defining match -- [function | ...], or a [match] on its one parameter, of its one top-level
      binding (a renderer bound twice is refused); several cases, a guard or an or-pattern leave the
      constructor unread.
    - A string constructor applied to a VARIABLE opens a relayed family when that variable is the
      payload a handler caught: [with Non_virtual i -> ... (Site i)] makes every tag literal in the
      scope of that local [let exception Non_virtual of string] a tag of the family "Site via
      Non_virtual". The relay belongs to the SCOPE whose handler it is -- local exceptions are
      generative, so a same-named exception elsewhere proves nothing, and a handler names the local
      exception unqualified. It may take one hop through a result: a handler returning the payload
      wrapped in a constructor ([Non_virtual i -> Error i]) relays when a caller of the declaring
      function -- qualified by its module in one component ([Low_level.f], or an alias of it), or
      unqualified in its own source where nothing else binds the name -- matches that constructor
      into a carrier ([match instantiate_computations ... with Error i -> ... (Site i)]). A handler
      or caller body that rebinds the payload's name anywhere, or changes module scope ([let open],
      [let module]), is not read, nor is a guarded case. A scope reaching no carrier mints nothing.
      That covers the literal at a [raise], the one handed to a helper that raises it, and the one a
      handler records directly; a string the exception itself is applied to that is no tag is
      refused. The function such a tag is minted in is the innermost value binding enclosing the
      exception's declaration -- a declaration anew inside the scope opens its own -- which is what
      decides its PHASE, so a relayed tag minted in two functions is refused. A tag computed at run
      time is not read.

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

(* The cases of the top-level [renderer] binding's defining match -- [function | ...], or a function
   whose body is a [match] -- and none nested beneath them: a case inside a case's right-hand side
   renders something else. *)
let renderer_bindings ~renderer content =
  List.concat_map (parse content) ~f:(fun item ->
      match item.pstr_desc with
      | Pstr_value (_, vbs) ->
          List.filter vbs ~f:(fun vb -> Option.equal String.equal (binding_name vb) (Some renderer))
      | _ -> [])

let renderer_cases ~renderer content =
  (* [function | ...], or [fun p -> match p with ...] scrutinizing the parameter itself. *)
  let rec defining (e : expression) =
    match e.pexp_desc with
    | Pexp_function ([], _, Pfunction_cases (cases, _, _)) -> cases
    | Pexp_function
        ( [
            {
              pparam_desc = Pparam_val (Nolabel, None, { ppat_desc = Ppat_var { txt = p; _ }; _ });
              _;
            };
          ],
          _,
          Pfunction_body body ) ->
        scrutinizing p body
    | Pexp_constraint (e, _) | Pexp_newtype (_, e) -> defining e
    | _ -> []
  and scrutinizing p (e : expression) =
    match e.pexp_desc with
    | Pexp_match ({ pexp_desc = Pexp_ident { txt = Lident x; _ }; _ }, cases) when String.equal x p
      ->
        cases
    | Pexp_constraint (e, _) -> scrutinizing p e
    | _ -> []
  in
  (* A renderer bound more than once is read from none of its bindings: which one callers reach is a
     question of order and scope this reader does not answer. *)
  match renderer_bindings ~renderer content with
  | [ vb ] -> defining vb.pvb_expr
  | _ -> []

(* Each constructor's ONE case of the renderer's defining match, as [(constructor, case)]: a
   constructor matched by more than one case, by a guarded case, or inside an or-pattern has no
   entry. What such a constructor prints cannot be read off one case, so every check below finds no
   rendering for it and refuses -- the reader accepts the canonical shapes and nothing it would have
   to guess at. *)
let canonical_cases ~renderer content =
  let cases = renderer_cases ~renderer content in
  let rec heads (p : pattern) =
    match p.ppat_desc with
    | Ppat_construct ({ txt; _ }, _) -> Option.to_list (last_name txt)
    | Ppat_or (a, b) -> heads a @ heads b
    | Ppat_alias (p, _) | Ppat_constraint (p, _) -> heads p
    | _ -> []
  in
  let count c =
    List.count cases ~f:(fun case -> List.mem (heads case.pc_lhs) c ~equal:String.equal)
  in
  List.filter_map cases ~f:(fun case ->
      match (case.pc_lhs.ppat_desc, case.pc_guard) with
      | Ppat_construct ({ txt; _ }, _), None -> (
          match last_name txt with Some c when count c = 1 -> Some (c, case) | _ -> None)
      | _ -> None)

(** [(constructor, tag)] for each case [C -> "tag"] of the renderer's defining match. *)
let renderings ~renderer ~source content =
  List.filter_map (canonical_cases ~renderer content) ~f:(fun (constructor, c) ->
      match (c.pc_lhs.ppat_desc, c.pc_rhs.pexp_desc) with
      | Ppat_construct (_, None), Pexp_constant (Pconst_string (s, _, _)) ->
          Option.map (tag_number s) ~f:(fun number ->
              { number; tag = s; family = Rendered constructor; source; minter = renderer })
      | _ -> None)

(** The constructors whose case of the renderer's defining match returns their argument unchanged
    ([Site s -> s]): only for those is the literal a carrier is applied to the text a printed
    provenance shows. *)
let identity_renderings ~renderer content =
  List.filter_map (canonical_cases ~renderer content) ~f:(fun (constructor, c) ->
      match (c.pc_lhs.ppat_desc, c.pc_rhs.pexp_desc) with
      | ( Ppat_construct (_, Some (_, { ppat_desc = Ppat_var { txt = v; _ }; _ })),
          Pexp_ident { txt = Lident v'; _ } )
        when String.equal v v' ->
          Some constructor
      | _ -> None)

(** The constructors whose case of the renderer's defining match binds each argument to a variable
    and returns a concatenation ([^]) of string literals and recursive calls on those variables,
    each variable rendered exactly once and in order
    ([Refined (a, b) -> renderer a ^ " -> " ^ renderer b]). *)
let composite_renderings ~renderer content =
  (* The variables a concatenation renders, in order; [None] for any other expression. *)
  let rec rendered (e : expression) =
    match e.pexp_desc with
    | Pexp_apply
        ({ pexp_desc = Pexp_ident { txt = Lident "^"; _ }; _ }, [ (Nolabel, l); (Nolabel, r) ]) ->
        Option.map2 (rendered l) (rendered r) ~f:( @ )
    | Pexp_constant (Pconst_string _) -> Some []
    | Pexp_apply
        ( { pexp_desc = Pexp_ident { txt = Lident f; _ }; _ },
          [ (Nolabel, { pexp_desc = Pexp_ident { txt = Lident x; _ }; _ }) ] )
      when String.equal f renderer ->
        Some [ x ]
    | _ -> None
  in
  let var (p : pattern) = match p.ppat_desc with Ppat_var { txt; _ } -> Some txt | _ -> None in
  List.filter_map (canonical_cases ~renderer content) ~f:(fun (constructor, c) ->
      match c.pc_lhs.ppat_desc with
      | Ppat_construct (_, Some (_, arg)) -> (
          let vars =
            match arg.ppat_desc with
            | Ppat_tuple ps -> Option.all (List.map ps ~f:var)
            | _ -> Option.map (var arg) ~f:List.return
          in
          match (vars, rendered c.pc_rhs) with
          | Some (_ :: _ as vars), Some calls when List.equal String.equal calls vars ->
              Some constructor
          | _ -> None)
      | _ -> None)

(** {1 The sources} *)

(** The text every source declaring a local exception contains, so a relaying source is read even
    when it does not spell a carrier. *)
let exception_keyword = "let exception"

type scope = {
  exn : string;
  declared_in : string;
  direct : string list;  (** Carriers a handler of THIS scope applies the payload to. *)
  results : string list;
      (** Constructors a handler of this scope wraps the payload in ([Non_virtual i -> Error i]),
          for a caller of [declared_in] to hand on. *)
  literals : mint list;  (** Its tag literals, as [Relayed] with [via] still empty. *)
  not_tags : string list;  (** Strings the exception itself is applied to that are no tag. *)
}
(** A local-exception scope of one source: its exception, the function declaring it, and what its
    handlers do with the caught payload. *)

type read = {
  path : string;  (** The source read. *)
  mints : mint list;  (** [Applied] literals. *)
  scopes : scope list;
  consumers : (string * string * string * string) list;
      (** [(m, f, k, carrier)]: a [match f ... with k v -> ... (carrier v)] in this source, [f]
          being module [m]'s function -- the module qualifying the call as resolved there, or the
          source's own for an unqualified one. *)
  malformed : string list;  (** String literals a carrier is applied to that are no tag. *)
}

(** The module a source file defines: [arrayjit/lib/tnode.ml] is [Tnode]. *)
let module_of_path path =
  String.capitalize (Stdlib.Filename.remove_extension (Stdlib.Filename.basename path))

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

(* Whether [p] binds the variable [v]. *)
let pattern_binds v (p : pattern) =
  let found = ref false in
  let finder =
    object
      inherit Ast_traverse.iter as super

      method! pattern p =
        (match p.ppat_desc with
        | (Ppat_var { txt; _ } | Ppat_alias (_, { txt; _ })) when String.equal txt v ->
            found := true
        | _ -> ());
        super#pattern p
    end
  in
  finder#pattern p;
  !found

(** One OCaml source's mints, scopes and result consumers, in source order, duplicates kept.

    A constructor with a carrier's name is taken for the owner's unless it names another: a module
    in [foreign] declares its own [string]-carrying constructor of that name (as [Operand_key_scan]
    declares [Site of string]), so an application qualified by it -- directly or through a chain of
    module bindings in the source ([module Scan = Test_utils.Operand_key_scan]) -- is not a
    provenance, nor is an unqualified one in [foreign]'s own source. A module name is resolved
    against the binding in scope where it is used -- structure items bind for the items after them,
    [let module] for its body, a nested structure for itself -- and a binding resolves its own
    target when it is made, so a chain needs no second lookup and cannot cycle. The owner's
    constructors are reached through aliases and opens this reader cannot follow, which is why the
    rule names what is excluded rather than what is included. *)
let read_source ~carriers ?(foreign = []) ~source content =
  let mints = ref [] and malformed = ref [] and consumers = ref [] and closed = ref [] in
  let structure = parse content in
  let own_module = module_of_path source in
  (* The module bindings in scope, innermost first, each to the module it resolved to. *)
  let env = ref [] in
  let resolve q = Option.value (List.Assoc.find !env q ~equal:String.equal) ~default:q in
  (* A module this source declares with its own carrier-named [string] constructor, under the name
     it resolves to; a non-alias module expression otherwise resolves to its own name. *)
  let local_foreign = ref [] in
  let declares_carrier items =
    List.exists (own_string_constructors items) ~f:(List.mem carriers ~equal:String.equal)
  in
  let is_foreign m =
    List.mem foreign m ~equal:String.equal || List.mem !local_foreign m ~equal:String.equal
  in
  let rec peel_module (me : module_expr) =
    match me.pmod_desc with Pmod_constraint (me, _) -> peel_module me | _ -> me
  in
  (* Whether the module [me] exports a foreign carrier constructor: it is a foreign module (or an
     alias of one in scope), or a structure declaring one or including a module that does. *)
  let rec foreign_module (me : module_expr) =
    match (peel_module me).pmod_desc with
    | Pmod_ident { txt; _ } ->
        Option.value_map (last_name txt) ~default:false ~f:(fun m -> is_foreign (resolve m))
    | Pmod_structure items ->
        declares_carrier items
        || List.exists items ~f:(fun item ->
            match item.pstr_desc with
            | Pstr_include { pincl_mod; _ } -> foreign_module pincl_mod
            | _ -> false)
    | _ -> false
  in
  let bind name (me : module_expr) =
    Option.iter name ~f:(fun name ->
        let target =
          match (peel_module me).pmod_desc with
          | Pmod_ident { txt; _ } -> Option.value_map (last_name txt) ~default:name ~f:resolve
          | Pmod_structure _ when foreign_module me ->
              let target = source ^ ":" ^ name in
              local_foreign := target :: !local_foreign;
              target
          | _ -> name
        in
        env := (name, target) :: !env)
  in
  (* Whether an unqualified carrier name here is some other constructor: in [foreign]'s own source,
     or inside a nested structure declaring one of its own. *)
  let unqualified_foreign = ref (is_foreign own_module) in
  let is_carrier (lid : longident) =
    match lid with
    | Lident c -> List.mem carriers c ~equal:String.equal && not !unqualified_foreign
    | Ldot (q, c) ->
        List.mem carriers c ~equal:String.equal
        && not (Option.value_map (last_name q) ~default:false ~f:(fun q -> is_foreign (resolve q)))
    | Lapply _ -> false
  in
  (* What a handler or consumer body [e] does with the variable [v]: the carriers applied to it
     anywhere -- applying one is what records a provenance -- and the constructors wrapping it in a
     position [e] returns, the only place a wrapped payload reaches the caller. A body that rebinds
     [v] ANYWHERE -- [let], [let*], a parameter, a case, a loop index -- is not read at all: which
     occurrence is the payload is a question of scope this reader does not answer, and relaying
     nothing is the loud answer (the scope's tags vanish, and every citation of them fails). *)
  let wrappers_of_var v e =
    let rebinds = ref false in
    let binders =
      object
        inherit Ast_traverse.iter as super

        method! pattern p =
          if pattern_binds v p then rebinds := true;
          super#pattern p

        (* A body that changes module scope could change what a carrier name refers to. *)
        method! expression e =
          (match e.pexp_desc with Pexp_open _ | Pexp_letmodule _ -> rebinds := true | _ -> ());
          super#expression e
      end
    in
    binders#expression e;
    if !rebinds then ([], [])
    else
      let applied_to_v (e : expression) =
        match e.pexp_desc with
        | Pexp_construct ({ txt; _ }, Some { pexp_desc = Pexp_ident { txt = Lident x; _ }; _ })
          when String.equal x v ->
            Some txt
        | _ -> None
      in
      let carriers_hit = ref [] in
      let finder =
        object
          inherit Ast_traverse.iter as super

          method! expression e =
            (match applied_to_v e with
            | Some txt when is_carrier txt ->
                Option.iter (last_name txt) ~f:(fun c -> carriers_hit := c :: !carriers_hit)
            | _ -> ());
            super#expression e
        end
      in
      finder#expression e;
      let rec returned (e : expression) =
        match e.pexp_desc with
        | Pexp_sequence (_, e)
        | Pexp_let (_, _, e)
        | Pexp_letmodule (_, _, e)
        | Pexp_letexception (_, e)
        | Pexp_open (_, e)
        | Pexp_constraint (e, _) ->
            returned e
        | Pexp_ifthenelse (_, a, b) -> returned a @ Option.value_map b ~default:[] ~f:returned
        | Pexp_match (_, cases) -> List.concat_map cases ~f:(fun c -> returned c.pc_rhs)
        | Pexp_try (body, cases) ->
            returned body @ List.concat_map cases ~f:(fun c -> returned c.pc_rhs)
        | _ -> [ e ]
      in
      let wrapped =
        List.filter_map (returned e) ~f:(fun e ->
            match applied_to_v e with
            | Some txt when not (is_carrier txt) -> last_name txt
            | _ -> None)
      in
      (!carriers_hit, wrapped)
  in
  (* [(constructor, v)] for each [C v] a pattern catches, the constructor as written. *)
  let rec caught (p : pattern) =
    match p.ppat_desc with
    | Ppat_construct ({ txt; _ }, Some (_, { ppat_desc = Ppat_var { txt = v; _ }; _ })) ->
        [ (txt, v) ]
    | Ppat_exception p | Ppat_alias (p, _) | Ppat_constraint (p, _) -> caught p
    | Ppat_or (a, b) -> caught a @ caught b
    | _ -> []
  in
  (* An unqualified call names the source's own top-level function only when nothing else in the
     source binds that name: a local or nested-module definition of it could be what is called. *)
  let own_function =
    let bound = Hashtbl.create (module String) in
    let counter =
      object
        inherit Ast_traverse.iter as super

        method! pattern p =
          (match p.ppat_desc with
          | Ppat_var { txt; _ } | Ppat_alias (_, { txt; _ }) -> Hashtbl.incr bound txt
          | _ -> ());
          super#pattern p
      end
    in
    counter#structure structure;
    let top_level =
      List.concat_map structure ~f:(fun item ->
          match item.pstr_desc with
          | Pstr_value (_, vbs) -> List.filter_map vbs ~f:binding_name
          | _ -> [])
    in
    fun f ->
      List.mem top_level f ~equal:String.equal
      && Option.equal Int.equal (Hashtbl.find bound f) (Some 1)
  in
  (* Whether an [open] or [include] of [me] brings a foreign carrier constructor into scope. *)
  let opens_foreign = foreign_module in
  let walker =
    object (self)
      inherit Ast_traverse.iter as super
      val mutable enclosing = "(top level)"

      (* The local string exceptions the walk is in, innermost first. A helper bound inside a scope
         raises the enclosing exception, so it does not move this; an exception declared anew inside
         the scope shadows it, and does. *)
      val mutable scopes : scope ref list = []

      method! value_binding vb =
        let saved = enclosing in
        Option.iter (binding_name vb) ~f:(fun name -> enclosing <- name);
        super#value_binding vb;
        enclosing <- saved

      val mutable depth = 0

      method! structure items =
        let saved = !env and saved_foreign = !unqualified_foreign in
        if depth > 0 && declares_carrier items then unqualified_foreign := true;
        depth <- depth + 1;
        List.iter items ~f:self#structure_item;
        depth <- depth - 1;
        env := saved;
        unqualified_foreign := saved_foreign

      method! structure_item item =
        match item.pstr_desc with
        | Pstr_module mb ->
            self#module_binding mb;
            bind mb.pmb_name.txt mb.pmb_expr
        | Pstr_recmodule mbs ->
            List.iter mbs ~f:(fun mb -> bind mb.pmb_name.txt mb.pmb_expr);
            List.iter mbs ~f:self#module_binding
        | Pstr_include { pincl_mod = me; _ } | Pstr_open { popen_expr = me; _ } ->
            super#structure_item item;
            (* For the rest of the enclosing structure, which restores the flag on exit. *)
            if opens_foreign me then unqualified_foreign := true
        | _ -> super#structure_item item

      method! case c =
        (* A handler of an open scope's exception -- spelled unqualified, as a local exception is;
           [M.Non_virtual] is another constructor -- and what it does with the payload. *)
        (* A guarded case is not read: whether its body runs is not a syntactic fact. *)
        List.iter
          (if Option.is_some c.pc_guard then [] else caught c.pc_lhs)
          ~f:(fun (e, v) ->
            match e with
            | Lident e -> (
                match List.find scopes ~f:(fun sc -> String.equal !sc.exn e) with
                | Some sc ->
                    let direct, results = wrappers_of_var v c.pc_rhs in
                    sc := { !sc with direct = direct @ !sc.direct; results = results @ !sc.results }
                | None -> ())
            | _ -> ());
        super#case c

      method! expression e =
        match e.pexp_desc with
        | Pexp_letexception
            ( ({ pext_name = { txt; _ }; pext_kind = Pext_decl (_, Pcstr_tuple [ _ ], None); _ } as
               ext),
              body ) ->
            let saved = scopes in
            self#extension_constructor ext;
            let sc =
              ref
                {
                  exn = txt;
                  declared_in = enclosing;
                  direct = [];
                  results = [];
                  literals = [];
                  not_tags = [];
                }
            in
            scopes <- sc :: List.filter scopes ~f:(fun o -> not (String.equal !o.exn txt));
            self#expression body;
            closed := !sc :: !closed;
            scopes <- saved
        | Pexp_open (({ popen_expr = me; _ } as od), body) when opens_foreign me ->
            self#open_declaration od;
            let saved = !unqualified_foreign in
            unqualified_foreign := true;
            self#expression body;
            unqualified_foreign := saved
        | Pexp_construct (lid, Some { pexp_desc = Pexp_constraint (arg, _); _ }) ->
            self#expression { e with pexp_desc = Pexp_construct (lid, Some arg) }
        | Pexp_letmodule ({ txt; _ }, me, body) ->
            self#module_expr me;
            let saved = !env in
            bind txt me;
            self#expression body;
            env := saved
        | Pexp_match
            ({ pexp_desc = Pexp_apply ({ pexp_desc = Pexp_ident { txt = f; _ }; _ }, _); _ }, cases)
          ->
            let callee =
              match f with
              | Ldot (Lident q, name) -> Some (resolve q, name)
              | Lident name when own_function name -> Some (own_module, name)
              | Ldot _ | Lident _ | Lapply _ -> None
            in
            Option.iter callee ~f:(fun (m, f) ->
                List.iter cases ~f:(fun (c : case) ->
                    List.iter
                      (if Option.is_some c.pc_guard then [] else caught c.pc_lhs)
                      ~f:(fun (k, v) ->
                        Option.iter (last_name k) ~f:(fun k ->
                            List.iter
                              (fst (wrappers_of_var v c.pc_rhs))
                              ~f:(fun carrier -> consumers := (m, f, k, carrier) :: !consumers)))));
            super#expression e
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
        | Pexp_construct
            ({ txt = Lident e; _ }, Some { pexp_desc = Pexp_constant (Pconst_string (s, _, _)); _ })
          when List.exists scopes ~f:(fun sc -> String.equal !sc.exn e) -> (
            match tag_number s with
            | Some number -> self#scoped number s
            | None ->
                let sc = List.find_exn scopes ~f:(fun sc -> String.equal !sc.exn e) in
                sc := { !sc with not_tags = s :: !sc.not_tags })
        | Pexp_constant (Pconst_string (s, _, _)) -> (
            match tag_number s with Some number -> self#scoped number s | None -> ())
        | _ -> super#expression e

      method scoped number tag =
        List.iter scopes ~f:(fun sc ->
            let m =
              {
                number;
                tag;
                family = Relayed { exn = !sc.exn; via = "" };
                source;
                minter = !sc.declared_in;
              }
            in
            sc := { !sc with literals = m :: !sc.literals })
    end
  in
  walker#structure structure;
  {
    path = source;
    mints = List.rev !mints;
    scopes = List.rev_map !closed ~f:(fun sc -> { sc with literals = List.rev sc.literals });
    consumers = List.rev !consumers;
    malformed = List.rev !malformed;
  }

(** The carriers a scope of the source [path] reaches: those its own handlers apply to its payload,
    and those a caller of its declaring function -- that function of [path]'s module, not merely one
    of its name -- applies to a result constructor its handlers wrap the payload in. *)
let scope_carriers ~consumers ~path (sc : scope) =
  sc.direct
  @ List.concat_map sc.results ~f:(fun k ->
      List.filter_map consumers ~f:(fun (m, f, k', carrier) ->
          Option.some_if
            (String.equal m (module_of_path path)
            && String.equal f sc.declared_in && String.equal k k')
            carrier))
  |> List.dedup_and_sort ~compare:String.compare

(** Every mint of [reads]: the [Applied] literals, and each scope's literals under every carrier its
    payload reaches -- a scope reaching none mints nothing, its literals being no provenance. Also
    [(source, literal)] for every string a carrier, or a relaying scope's exception, is applied to
    that is no tag. *)
let resolve reads =
  let consumers = List.concat_map reads ~f:(fun r -> r.consumers) in
  let relayed =
    List.concat_map reads ~f:(fun r ->
        List.concat_map r.scopes ~f:(fun sc ->
            List.map (scope_carriers ~consumers ~path:r.path sc) ~f:(fun via -> (r.path, sc, via))))
  in
  ( List.concat_map reads ~f:(fun r -> r.mints)
    @ List.concat_map relayed ~f:(fun (_, sc, via) ->
        List.map sc.literals ~f:(fun m -> { m with family = Relayed { exn = sc.exn; via } })),
    List.concat_map reads ~f:(fun r -> List.map r.malformed ~f:(fun l -> (r.path, l)))
    @ (List.concat_map relayed ~f:(fun (path, sc, _) ->
           List.map sc.not_tags ~f:(fun l -> (path, l)))
      |> List.dedup_and_sort ~compare:Poly.compare) )

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

(** What [text] cites, given every library mint and the test-owned tags. [spellings] are
    relayed-family names read whether or not the family mints anything now: the one a retired
    family's [Non_virtual N] citations need, to be refused rather than unread. *)
let mentions ?(spellings = []) ~(mints : mint list) ~test_tags text =
  let library = List.filter mints ~f:(fun m -> not (is_test_source m.source)) in
  let library_tags = Set.of_list (module String) (List.map library ~f:(fun m -> m.tag)) in
  let known_tags = Set.union library_tags (Set.of_list (module String) test_tags) in
  let numbers ms = Set.of_list (module Int) (List.map ms ~f:(fun m -> m.number)) in
  let all_numbers = numbers library in
  let exceptions =
    spellings
    @ List.filter_map library ~f:(fun m ->
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
let family_violations ?(identities = []) ?(composed = []) ?(renderer_bindings = 1) ~type_source
    ~shape ~(mints : mint list) () =
  match shape with
  | None ->
      [
        Printf.sprintf "%s: no variant `provenance` declaration: the families cannot be derived"
          type_source;
      ]
  | Some shape ->
      (if renderer_bindings = 1 then []
       else
         [
           Printf.sprintf
             "%s: the renderer is bound %d times, and is read only from its one binding: no \
              rendering can be derived"
             type_source renderer_bindings;
         ])
      @ List.map shape.unread ~f:(fun c ->
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
      @ List.filter_map shape.composite ~f:(fun c ->
          Option.some_if
            (not (List.mem composed c ~equal:String.equal))
            (Printf.sprintf
               "%s: the renderer's case for %s does not render each of its provenances, in order, \
                by a recursive call: a printed provenance could drop or reorder a recorded tag"
               type_source c))
      @ List.filter_map shape.carriers ~f:(fun c ->
          Option.some_if
            (not (List.mem identities c ~equal:String.equal))
            (Printf.sprintf
               "%s: the renderer does not return %s's string unchanged, so the literals it is \
                applied to are not what a printed provenance shows"
               type_source c))
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

(** What the phase table refuses, given the relayed family [exn] whose phases [phases] names, each
    phase as [(phase, (source, function))]: the function is bound to its source, so a same-named
    function elsewhere cannot stand in for the one that moved. *)
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
            | minted, Some (source, minter) ->
                if
                  List.exists minted ~f:(fun c ->
                      String.equal c.source source && String.equal c.minter minter)
                then None
                else
                  Some
                    (Printf.sprintf "%s: phase table puts %s at %s, but it is minted in %s"
                       table_source tag phase
                       (String.concat ~sep:" and "
                          (List.dedup_and_sort ~compare:String.compare
                             (List.map minted ~f:(fun c -> c.source ^ " " ^ c.minter))))))
      in
      let stale_phases =
        List.filter_map phases ~f:(fun (phase, (source, minter)) ->
            if
              List.exists codes ~f:(fun c ->
                  String.equal c.source source && String.equal c.minter minter)
            then None
            else
              Some
                (Printf.sprintf "phase %s names %s in %s, which mints no %s code" phase minter
                   source exn))
      in
      per_entry @ stale_phases

(** The [(prefix, reason)] exclusions no path in [paths] lives under. *)
let stale_records ~records paths =
  List.filter_map records ~f:(fun (prefix, _) ->
      if List.exists paths ~f:(String.is_prefix ~prefix) then None
      else Some (Printf.sprintf "%s: no file lives under this excluded prefix" prefix))
