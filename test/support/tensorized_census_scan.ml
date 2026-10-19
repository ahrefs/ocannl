(** The reader behind [test/operations/tensorized_census_ratchet] (gh-ocannl-1309): an executed
    claim about a TENSORIZED routine needs a census judgment beside it.

    A schedule that asks for tensorization and declines at codegen renders the lane-0 scalar
    fallback, which computes the same values. So a parity claim over an executed tensorized
    candidate -- "the packed+tensorized matmul matches the serial twin bitwise" -- passes whether
    the tensor-core path ran or not; only the compiled routine's mma census ([Context.routine.mma],
    gh-ocannl-626) tells the two apart. gh-ocannl-1283 found eleven tests whose tensorized claims
    rested on parity alone and gave each a census claim; this reader stops the twelfth.

    {1 The rule}

    A [Verdict] claim whose label names tensorization AND asserts parity, in a top-level structure
    item that executes a routine, needs a census JUDGMENT in that item. "In that item" includes the
    file's own top-level bindings the item references, transitively, so a helper that executes (or
    judges) on the item's behalf counts for it. A tensorized claim that asserts no parity -- "the
    tensorized seeds are proposed", "the twin constructs and validates" -- is about structure, which
    the census does not decide, and is not read.

    {1 The closed lists of recognised forms}

    - {b A claim} is an application, by its path, of a [Verdict.Claims] member other than [fail]
      (whose argument is a message) and [skipped] (which decides nothing), that {!Verdict_callee}
      resolves as [Verdict]'s or as ambiguous. Its label is its first positional argument -- the
      format, for [pf]/[claimf]/[pf_all2].
    - {b A label's fragments} are the string literals written in the label expression, and those of
      every name it references, followed through [let] bindings in scope (a function parameter and a
      function-valued binding contribute none). Both tests below are case-insensitive and pass when
      SOME fragment passes, not necessarily the same one.
    - {b A label names tensorization} by the substrings [tensoriz] (tensorized, tensorizes,
      tensorization), [tensor core] or [tensor-core], or the words [mma] or [wmma], where a word is
      bounded by a non-letter or the string's end (a plural [s] allowed), so [Tile_mma], [sk_mma]
      and [qkv_mma:] are words and [comma] or [gemma] are not.
    - {b A label asserts parity} by a word starting [match], [bitwise], [parity] or [reference], or
      the phrases [within tolerance] or [same values].
    - {b An item executes} when it references one of: [Ll_test.execute], [execute_with_census],
      [run], [run_linked], [optimize_and_execute]; [Context.run]; [Train.run], [run_once],
      [forward_once], [update_once].
    - {b An item judges the census} when it references one of the census PREDICATES --
      [Ll_test.rendered_tensorized], [all_tensorized], [all_declined], [all_rendered],
      [none_emitted], [census_all], [census_agrees_with_capability]; [C_syntax.equal_tensorization],
      [compare_tensorization], [is_tensorized_rendering], [equal_mma_rendering],
      [compare_mma_rendering] -- writes one of the census's constructors, in an expression or a
      pattern ([Tensorized], [Scalar_fallback], [Not_requested], [Mma_intrinsics],
      [Mma_intrinsics_ldmatrix], [Mma_register_tiled], [Mma_scalar_fallback]), unqualified or
      qualified by [C_syntax], or reads one of the census's fields ([Ir.C_syntax.mma_summary]'s
      [tensorization], [scalar_fallbacks], [renderings]; [Autotune.report]'s [best_tensorization],
      [best_mma_scalar_fallbacks]), unqualified or qualified by [C_syntax] or [Autotune]. Obtaining
      the census is not judging it: [execute_with_census], [Ll_test.renderings] and a read of
      [routine.Context.mma] count only when something above decides on the value, since a census
      bound and discarded ([Some (got, _)]) is exactly the parity-only shape.

    A form is matched by its module and name. The module is the qualifier's last component
    ([Ir.C_syntax], [Test_utils.Ll_test]), or what a local [module L = Ll_test] alias denotes; an
    [open] (structure-level or local) of one of the modules above brings that module's forms into
    unqualified scope. Names are resolved lexically ({!Lexical_scope}): a parameter or local binding
    shadowing a form, or a top-level name, is that binding instead.

    {1 What it deliberately does not read}

    Each is a boundary, stated so a miss past it is a known deferral:

    - Granularity is the top-level item. A census judgment anywhere in the item (or a helper it
      references) covers every tensorized claim in it, even one about a different execution.
    - A claim reached through a wrapper the source defines
      ([let mma_gated label b = gated … label b]) is read at the wrapper's own claim, whose label is
      a parameter with no fragments -- so the wrapper's call sites are not read.
    - A label whose tensorization word arrives through a function parameter
      ([Printf.sprintf "%s: every %s candidate matches" tag what] with [what] a parameter) has no
      fragment that says so, and the claim is not a tensorized claim to this reader. The same holds
      for a claim whose label never says "tensorized" although its routine was a tensorized seed:
      the label is the closed signal this reader has.
    - An execution through anything other than the forms above -- [Autotune]'s validation of a
      candidate, a backend's own [run] -- does not make an item executing.
    - A judgment the item receives as a function parameter ([leg ~check] where the caller passes a
      census check) is outside the item, so such an item's tensorized parity claim is refused -- the
      loud direction; no item in the tree has the shape today.
    - Reading is not deciding: a census field read only to print it, a source's own variant with a
      [Tensorized] constructor, or a census predicate whose result is dropped counts as a judgment.
      Each needs code written against the rule's intent. *)

open Base
open Ppxlib
module Read = Config_key_scan

(** {1 Forms} *)

type form = Execution | Judgment

let forms =
  [
    ( "Ll_test",
      [
        ("execute", Execution);
        ("execute_with_census", Execution);
        ("run", Execution);
        ("run_linked", Execution);
        ("optimize_and_execute", Execution);
        ("rendered_tensorized", Judgment);
        ("all_tensorized", Judgment);
        ("all_declined", Judgment);
        ("all_rendered", Judgment);
        ("none_emitted", Judgment);
        ("census_all", Judgment);
        ("census_agrees_with_capability", Judgment);
      ] );
    ("Context", [ ("run", Execution) ]);
    ( "Train",
      [
        ("run", Execution);
        ("run_once", Execution);
        ("forward_once", Execution);
        ("update_once", Execution);
      ] );
    ( "C_syntax",
      [
        ("equal_tensorization", Judgment);
        ("compare_tensorization", Judgment);
        ("is_tensorized_rendering", Judgment);
        ("equal_mma_rendering", Judgment);
        ("compare_mma_rendering", Judgment);
      ] );
  ]

let form_of ~module_ name =
  Option.bind (List.Assoc.find forms module_ ~equal:String.equal) ~f:(fun names ->
      List.Assoc.find names name ~equal:String.equal)

(** The census's constructors: [Ir.C_syntax.tensorization] and [Ir.C_syntax.mma_rendering]. *)
let census_constructors =
  [
    "Tensorized";
    "Scalar_fallback";
    "Not_requested";
    "Mma_intrinsics";
    "Mma_intrinsics_ldmatrix";
    "Mma_register_tiled";
    "Mma_scalar_fallback";
  ]

(** The census's fields: [Ir.C_syntax.mma_summary]'s, and [Autotune.report]'s census of its winner.
*)
let census_fields =
  [
    "tensorization";
    "scalar_fallbacks";
    "renderings";
    "best_tensorization";
    "best_mma_scalar_fallbacks";
  ]

(** The claim members read: every [Verdict.Claims] member but [fail] and [skipped]. *)
let claim_members =
  List.filter Verdict_callee.members ~f:(fun member ->
      not (List.mem [ "fail"; "skipped" ] member ~equal:String.equal))

(** {1 Vocabulary} *)

let is_letter c = Char.is_alpha c

(* [word] occurs in [text] bounded by non-letters, allowing one plural [s]. *)
let has_word text word =
  let n = String.length text and w = String.length word in
  let rec from pos =
    match String.substr_index text ~pattern:word ~pos with
    | None -> false
    | Some i ->
        let before_ok = i = 0 || not (is_letter text.[i - 1]) in
        let after = i + w in
        let after_ok =
          after >= n
          || (not (is_letter text.[after]))
          || (Char.equal text.[after] 's' && (after + 1 >= n || not (is_letter text.[after + 1])))
        in
        if before_ok && after_ok then true else from (i + 1)
  in
  w > 0 && from 0

let names_tensorization fragment =
  let text = String.lowercase fragment in
  List.exists [ "tensoriz"; "tensor core"; "tensor-core" ] ~f:(fun substring ->
      String.is_substring text ~substring)
  || has_word text "mma" || has_word text "wmma"

(* [stem] occurs in [text] starting a word: preceded by a non-letter or the string's start. *)
let has_stem text stem =
  let rec from pos =
    match String.substr_index text ~pattern:stem ~pos with
    | None -> false
    | Some i -> if i = 0 || not (is_letter text.[i - 1]) then true else from (i + 1)
  in
  (not (String.is_empty stem)) && from 0

let asserts_parity fragment =
  let text = String.lowercase fragment in
  List.exists [ "match"; "bitwise"; "parity"; "reference" ] ~f:(has_stem text)
  || List.exists [ "within tolerance"; "same values" ] ~f:(fun substring ->
      String.is_substring text ~substring)

(** {1 The walk} *)

(* What a value name denotes: one of the forms, brought in by an [open]; or a binding the source
   makes, with the string fragments its value carries and, for a top-level binding, the item that
   defines it. *)
type value = Form of form | Binding of { fragments : string list; top : int option }

type claim = {
  item : int;
  line : int;
  member : string;
  label : string;
  tensorized : bool;  (** The label names tensorization and asserts parity. *)
}

type item = {
  id : int;
  item_line : int;
  mutable executes : bool;
  mutable judges : bool;
  mutable refs : int list;
}

type census = { claims : claim list; items : item list }

let one_line text =
  String.split_on_chars text ~on:[ ' '; '\n'; '\t'; '\r' ]
  |> List.filter ~f:(Fn.non String.is_empty)
  |> String.concat ~sep:" "

let rec peel expr =
  match expr.pexp_desc with
  | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) -> peel inner
  | _ -> expr

(* The canonical module a path names: what an alias in scope denotes, else its last component. *)
let canonical modules = function
  | Longident.Lident m -> Some (Option.value (Map.find modules m) ~default:m)
  | Ldot (_, last) -> Some last
  | Lapply _ -> None

let read content =
  let ast = Read.structure_of content in
  let owners = Verdict_callee.of_structure ast in
  let items = Hashtbl.create (module Int) in
  let current = ref None in
  let claims = ref [] in
  let enter id line =
    let item =
      Hashtbl.find_or_add items id ~default:(fun () ->
          { id; item_line = line; executes = false; judges = false; refs = [] })
    in
    current := Some item
  in
  let note f = Option.iter !current ~f in
  let record_form = function
    | Execution -> note (fun item -> item.executes <- true)
    | Judgment -> note (fun item -> item.judges <- true)
  in
  let lookup env name = Lexical_scope.lookup env name in
  (* The string fragments an expression carries: its literals, and those of the names it references
     (attributes, function bodies and the bindings a name reaches are not walked again). *)
  let fragments_of env expr =
    let found = ref [] in
    let collector =
      object
        inherit Ast_traverse.iter as super
        method! attribute _ = ()

        method! expression e =
          match e.pexp_desc with
          | Pexp_constant (Pconst_string (s, _, _)) -> found := s :: !found
          | Pexp_ident { txt = Lident name; _ } -> (
              match lookup env name with
              | Some (Binding { fragments; _ }) -> found := fragments @ !found
              | _ -> ())
          | Pexp_function _ -> ()
          | _ -> super#expression e
      end
    in
    collector#expression expr;
    !found
  in
  let top_group = ref None in
  let walker =
    object (self)
      inherit [value, string] Lexical_scope.scoped as super
      method local = Binding { fragments = []; top = None }
      method module_path env path = canonical env.Lexical_scope.modules path
      method! attribute _ attr = attr

      method! binding_denotes env binding =
        let top =
          match !top_group with
          | Some (id, group) when List.exists group ~f:(fun b -> phys_equal b binding) -> Some id
          | _ -> None
        in
        let rhs = peel binding.pvb_expr in
        let fragments =
          match rhs.pexp_desc with Pexp_function _ -> [] | _ -> fragments_of env rhs
        in
        Binding { fragments; top }

      method! recursive_denotes ~top env bindings =
        let saved = !top_group in
        if top then
          top_group :=
            Option.map (List.hd bindings) ~f:(fun b -> (b.pvb_loc.loc_start.pos_cnum, bindings));
        let result = super#recursive_denotes ~top env bindings in
        top_group := saved;
        result

      method! define ~top env rec_flag bindings ~walk =
        if top then (
          let first = List.hd_exn bindings in
          enter first.pvb_loc.loc_start.pos_cnum first.pvb_loc.loc_start.pos_lnum;
          let saved = !top_group in
          top_group := Some (first.pvb_loc.loc_start.pos_cnum, bindings);
          let result = super#define ~top env rec_flag bindings ~walk in
          top_group := saved;
          result)
        else super#define ~top env rec_flag bindings ~walk

      method! opened ~top:_ ~include_:_ env module_ =
        match Option.bind module_ ~f:(fun m -> List.Assoc.find forms m ~equal:String.equal) with
        | Some names ->
            Lexical_scope.push_frame env
              (Map.of_alist_reduce
                 (module String)
                 (List.map names ~f:(fun (name, form) -> (name, Form form)))
                 ~f:(fun _ later -> later))
        | None -> env

      method! structure_item env si =
        if List.exists ast ~f:(fun original -> phys_equal original si) then
          enter si.pstr_loc.loc_start.pos_cnum si.pstr_loc.loc_start.pos_lnum;
        super#structure_item env si

      method! ident env ident =
        match ident.txt with
        | Lident name -> (
            match lookup env name with
            | Some (Form form) -> record_form form
            | Some (Binding { top = Some id; _ }) ->
                note (fun item -> if id <> item.id then item.refs <- id :: item.refs)
            | Some (Binding { top = None; _ }) | None -> ())
        | Ldot (path, name) ->
            Option.iter
              (Option.bind (canonical env.modules path) ~f:(fun module_ -> form_of ~module_ name))
              ~f:record_form
        | Lapply _ -> ()

      method! pattern env p =
        (match p.ppat_desc with Ppat_construct ({ txt; _ }, _) -> self#constructor txt | _ -> ());
        super#pattern env p

      method constructor txt =
        let qualifier_ok =
          match txt with
          | Longident.Lident _ -> true
          | Ldot (Ldot (_, m), _) | Ldot (Lident m, _) -> String.equal m "C_syntax"
          | _ -> false
        in
        if qualifier_ok && List.mem census_constructors (Longident.last_exn txt) ~equal:String.equal
        then record_form Judgment

      method! expression env expr =
        (match expr.pexp_desc with
        | Pexp_construct ({ txt; _ }, _) -> self#constructor txt
        | Pexp_field (_, { txt; _ }) ->
            let qualifier_ok =
              match txt with
              | Longident.Lident _ -> true
              | Ldot (Ldot (_, m), _) | Ldot (Lident m, _) ->
                  List.mem [ "C_syntax"; "Autotune" ] m ~equal:String.equal
              | _ -> false
            in
            if qualifier_ok && List.mem census_fields (Longident.last_exn txt) ~equal:String.equal
            then record_form Judgment
        | Pexp_apply (callee, arguments) -> (
            let callee = peel callee in
            match (callee.pexp_desc, Read.longident_of callee) with
            | Pexp_ident _, Some path
              when List.mem claim_members (List.last_exn path) ~equal:String.equal -> (
                let ours =
                  match Verdict_callee.resolve owners callee with
                  | Some { ownership = Verdicts | Ambiguous; _ } -> true
                  | _ -> false
                in
                let label =
                  List.find_map arguments ~f:(function
                    | Nolabel, argument -> Some (peel argument)
                    | _ -> None)
                in
                match (ours, label, !current) with
                | true, Some label, Some item ->
                    let fragments = fragments_of env label in
                    claims :=
                      {
                        item = item.id;
                        line = expr.pexp_loc.loc_start.pos_lnum;
                        member = List.last_exn path;
                        label = one_line (Pprintast.string_of_expression label);
                        tensorized =
                          List.exists fragments ~f:names_tensorization
                          && List.exists fragments ~f:asserts_parity;
                      }
                      :: !claims
                | _ -> ())
            | _ -> ())
        | _ -> ());
        super#expression env expr
    end
  in
  ignore
    (walker#structure { Lexical_scope.frames = []; modules = Map.empty (module String) } ast
      : structure);
  { claims = List.rev !claims; items = Hashtbl.data items }

(** {1 The verdict} *)

(* Whether an item, or a top-level binding it references (transitively), satisfies [holds]. *)
let reaches (census : census) ~holds =
  let by_id = Hashtbl.of_alist_exn (module Int) (List.map census.items ~f:(fun i -> (i.id, i))) in
  let memo = Hashtbl.create (module Int) in
  let rec go visiting id =
    match Hashtbl.find memo id with
    | Some answer -> answer
    | None ->
        if Set.mem visiting id then false
        else
          let answer =
            match Hashtbl.find by_id id with
            | None -> false
            | Some item -> holds item || List.exists item.refs ~f:(go (Set.add visiting id))
          in
          Hashtbl.set memo ~key:id ~data:answer;
          answer
  in
  go (Set.empty (module Int))

(** The tensorized claims of a source whose item executes a routine, each with whether that item
    judges the census. *)
let tensorized_executed content =
  let census = read content in
  let executes = reaches census ~holds:(fun i -> i.executes) in
  let judges = reaches census ~holds:(fun i -> i.judges) in
  List.filter_map census.claims ~f:(fun claim ->
      if claim.tensorized && executes claim.item then Some (claim, judges claim.item) else None)

(** The claims a source must not ship: tensorized, executed, and with no census judgment in reach.
*)
let unjudged content =
  List.filter_map (tensorized_executed content) ~f:(fun (claim, judged) ->
      Option.some_if (not judged) claim)

type exemption = { path : string; label : string; reason : string }

(** Refusals for [(path, content)] sources. An exemption row absorbs every unjudged claim of its
    file whose label (the label expression, whitespace collapsed) it names; a row that absorbs
    nothing is stale. *)
let violations ~exemptions sources =
  let used = Hash_set.create (module Int) in
  let refused =
    List.concat_map sources ~f:(fun (path, content) ->
        List.filter_map (unjudged content) ~f:(fun claim ->
            match
              List.findi exemptions ~f:(fun _ e ->
                  String.equal e.path path && String.equal e.label claim.label)
            with
            | Some (index, _) ->
                Hash_set.add used index;
                None
            | None ->
                Some
                  (Printf.sprintf
                     "%s:%d: the tensorized claim %s (%s) sits in an item that executes a routine \
                      but judges no mma census: parity alone also passes for a Tile_mma that \
                      declined to the scalar fallback. Read the routine's census \
                      (Ll_test.execute_with_census, routine.Context.mma) and claim on it \
                      (Ll_test.rendered_tensorized, all_tensorized, census_agrees_with_capability) \
                      in the same item (gh-ocannl-1309)."
                     path claim.line claim.label claim.member)))
  in
  let stale =
    List.filter_mapi exemptions ~f:(fun index e ->
        Option.some_if
          (not (Hash_set.mem used index))
          (Printf.sprintf "%s: the exemption for the claim %s absorbs no unjudged claim (stale)"
             e.path e.label))
  in
  refused @ stale
