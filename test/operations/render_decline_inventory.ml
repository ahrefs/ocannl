(* gh-ocannl-1351: every render-time typed decline is classified against the op-legality oracle.

   A render-time typed decline is a [Schedule_outcome.Cause_at] constructed at phase
   [Backend_codegen]: one candidate's decline, raised while its code is generated. One the oracle
   ([Schedule.op_legality]) does not also state is met only once a candidate carrying the move is
   applied and rendering starts, where the beam menu could have pruned the move. gh-ocannl-1231 gave
   the two pipelined-tile declines an oracle counterpart and built the renderer's decline FROM the
   oracle's reason; the others were never audited. The audit lives in the agent note's section
   [section_heading] (docs/agent-notes/scheduling-and-autotune.md), one bullet per decline, and this
   scan relates it to the code in both directions.

   Every [Cause_at (Backend_codegen, cause)] constructed anywhere in [arrayjit/lib] needs an entry,
   keyed by what the cause names. A "literal" site names it in the cause's own record (the string of
   an [Illegal_schedule]'s [check], an [Unsupported]'s [feature], a [Backend_rejected]'s [stage], or
   a [Resource_exceeded]'s [resource] constructor); an "oracle" site builds the cause from the
   oracle's own reason type, the constructor handed to a [Schedule.*_cause] function. An entry says
   "paired (`Schedule.X`)" exactly when its site is an oracle site built from [X], and "render-only,
   because ..." exactly when it is a literal site: a pairing is a property of how the decline is
   built, not a claim the note can make on its own. Every entry names a decline some site raises, so
   a removed decline cannot leave its entry behind.

   And an oracle site's constructor must be one the oracle can still produce: [X] is constructed (as
   an expression, not matched) in the body of [op_legality] or of a top-level binding of
   [oracle_file] that body reaches by name, transitively. Reaching is by unqualified reference, an
   over-approximation of calling; a constructor built only in a nested module, or only outside
   [oracle_file], is not reached and the site fails as unconsulted -- loudly, never passed.

   A [Backend_codegen] cause the scan cannot key (a variable, a computed check name) fails: a new
   decline has to be written in one of the readable forms, which is what makes it countable. So does
   a [Cause_at] anywhere in the corpus whose phase is not spelled out as a constant constructor (a
   variable, a constraint): such a site may be a render-time decline the scan cannot see. And in
   [renderer_file] every [Cause_at] is constructed at [Backend_codegen], so a render-time decline
   cannot leave the inventory by naming another phase.

   An oracle site is read by the NAME [Schedule], which denotes [arrayjit/lib/schedule.ml] only
   while nothing in the corpus rebinds it; so a module of that name bound anywhere in the corpus (a
   module binding or declaration, a [let module], a functor parameter, a first-class module unpacked
   into it) fails, and an alias under another name leaves the call unreadable. The libraries outside
   the corpus that it opens (Base, Stdio, ...) have no [Schedule] module.

   The recognised forms, closed. A site is an EXPRESSION [Pexp_construct] whose constructor's last
   component is [Cause_at], applied to a two-element tuple whose first element is the constant
   constructor [Backend_codegen] (any qualification). Its cause is keyed only as: (a) a constructor
   [Illegal_schedule] / [Unsupported] / [Backend_rejected] / [Resource_exceeded] applied to a record
   literal without [with], whose field [check] / [feature] / [stage] / [resource] respectively is a
   string literal or a constant constructor -- a literal site; or (b) an application of a function
   whose path ends [Schedule.<name>_cause] with exactly one argument that is a constructor
   application, whose last component is the key -- an oracle site. Anything else at
   [Backend_codegen] is unread and refused, never passed. A pattern ([| Cause_at (Backend_codegen,
   _) ->]) is not a construction, and a comment or string naming one is not a site. An exception
   rebinding [Cause_at] under another name ([exception D = Cause_at], at top level or as a [let
   exception]) would raise sites under a name this list does not read, so it is refused.

   The corpus is every checked-in [arrayjit/lib/**.ml], selected from [Test_utils.Source_inventory]
   over the rule's clean sandbox (so no build output joins it), each parse classified through
   [Test_utils.Scan_attempt]; the note is read from the same inventory. *)

open Base
open Stdio
open Verdict.Claims
module Read = Test_utils.Config_key_scan
module Ast_traverse = Ppxlib.Ast_traverse

let render_phase = "Backend_codegen"
let renderer_file = "c_syntax.ml"
let section_title = "Render-time typed declines"
let section_heading = "## " ^ section_title
let oracle_file = "schedule.ml"
let oracle_entry = "op_legality"

(* The field that names a decline, per cause constructor: a string literal, or a constant
   constructor such as [Resource_exceeded]'s [resource]. *)
let literal_fields =
  [
    ("Illegal_schedule", "check");
    ("Unsupported", "feature");
    ("Resource_exceeded", "resource");
    ("Backend_rejected", "stage");
  ]

let oracle_module = "Schedule"
let cause_suffix = "_cause"
let paired_marker = "paired (`" ^ oracle_module ^ "."
let render_only_marker = "render-only, because "

type origin = Literal | Oracle of string
type site = { file : string; line : int; key : string; origin : origin }
type classification = Paired of string | Render_only | Unlabelled
type entry = { entry_key : string; classification : classification }

type problem =
  | No_section
  | Unreadable of { file : string; line : int }
  | Off_phase of { file : string; line : int }
  | Opaque_phase of { file : string; line : int }
  | Shadows_oracle of { file : string; line : int }
  | Rebinds_cause_at of { file : string; line : int }
  | Unconsulted of site
  | Unclassified of site
  | Disagrees of site * entry
  | Stale of entry
  | Unlabelled_entry of entry
  | Duplicate_entry of string
  | Unparsed of string
  | Scanner_raised of string * string
  | Note_missing of string

let problem_kind = function
  | No_section -> "no-section"
  | Unreadable _ -> "unreadable"
  | Off_phase _ -> "off-phase"
  | Opaque_phase _ -> "opaque-phase"
  | Shadows_oracle _ -> "shadows-oracle"
  | Rebinds_cause_at _ -> "rebinds-cause-at"
  | Unconsulted _ -> "unconsulted"
  | Unclassified _ -> "unclassified"
  | Disagrees _ -> "disagrees"
  | Stale _ -> "stale"
  | Unlabelled_entry _ -> "unlabelled"
  | Duplicate_entry _ -> "duplicate"
  | Unparsed _ -> "unparsed"
  | Scanner_raised _ -> "raised"
  | Note_missing _ -> "note-missing"

let origin_to_string = function Literal -> "a literal" | Oracle via -> "built through " ^ via

let describe = function
  | No_section -> "the agent note has no section headed " ^ section_heading
  | Unreadable { file; line } ->
      Printf.sprintf
        "%s:%d: a Cause_at (%s, _) whose decline the scan cannot key: name it in the cause's \
         record with a literal, or build it from an oracle reason through a %s.*%s function"
        file line render_phase oracle_module cause_suffix
  | Off_phase { file; line } ->
      Printf.sprintf
        "%s:%d: a Cause_at in the renderer constructed at a phase other than %s: a render-time \
         decline is a %s decline"
        file line render_phase render_phase
  | Opaque_phase { file; line } ->
      Printf.sprintf
        "%s:%d: a Cause_at whose phase is not a constant constructor, so the scan cannot tell \
         whether it is a render-time decline: spell the phase out"
        file line
  | Shadows_oracle { file; line } ->
      Printf.sprintf
        "%s:%d: binds a module named %s, so an oracle site's %s.*%s call would no longer name the \
         op-legality oracle"
        file line oracle_module oracle_module cause_suffix
  | Rebinds_cause_at { file; line } ->
      Printf.sprintf
        "%s:%d: rebinds the Cause_at exception under another name, so declines raised through it \
         would escape the inventory: raise Cause_at itself"
        file line
  | Unconsulted s ->
      Printf.sprintf
        "%s:%d: the decline `%s` is %s, but no code %s.%s reaches constructs %s: the oracle no \
         longer produces it, so the pairing is gone"
        s.file s.line s.key (origin_to_string s.origin) oracle_module oracle_entry s.key
  | Unclassified s ->
      Printf.sprintf
        "%s:%d: the render-time decline `%s` (%s) has no entry in the note's %S section: add one, \
         paired or render-only"
        s.file s.line s.key (origin_to_string s.origin) section_heading
  | Disagrees (s, _) ->
      Printf.sprintf
        "%s:%d: the note classifies `%s` against how its site builds it (%s): an oracle-built \
         decline is \"paired (`%s.%s`)\", a literal one \"render-only, because ...\""
        s.file s.line s.key (origin_to_string s.origin) oracle_module s.key
  | Stale e -> Printf.sprintf "the note's entry `%s` names no decline any site raises" e.entry_key
  | Unlabelled_entry e ->
      Printf.sprintf "the note's entry `%s` says neither %S nor %S" e.entry_key
        (paired_marker ^ "X`)") render_only_marker
  | Duplicate_entry key -> Printf.sprintf "the note has more than one entry `%s`" key
  | Unparsed file -> Printf.sprintf "%s: does not parse, so its declines cannot be read" file
  | Scanner_raised (file, exn) ->
      Printf.sprintf "%s: parses, but the scan raised %s on it (a scanner defect)" file exn
  | Note_missing path -> Printf.sprintf "the source tree holds no %s" path

let last_component txt = List.last_exn (Read.flatten_longident txt)

let key_literal (value : Ppxlib.expression) =
  match (Read.string_literal value, value.pexp_desc) with
  | Some key, _ -> Some key
  | None, Pexp_construct ({ txt; _ }, None) -> Some (last_component txt)
  | None, _ -> None

(* The key and origin of a [Backend_codegen] cause, or [None] where it is in neither readable
   form. *)
let identity (cause : Ppxlib.expression) : (string * origin) option =
  match cause.pexp_desc with
  | Pexp_construct ({ txt; _ }, Some { pexp_desc = Pexp_record (fields, None); _ }) ->
      Option.bind
        (List.Assoc.find literal_fields (last_component txt) ~equal:String.equal)
        ~f:(fun field ->
          List.find_map fields ~f:(fun ({ txt = label; _ }, value) ->
              if String.equal (last_component label) field then
                Option.map (key_literal value) ~f:(fun key -> (key, Literal))
              else None))
  | Pexp_apply (fn, args) -> (
      match Option.map (Read.longident_of fn) ~f:List.rev with
      | Some (name :: m :: _)
        when String.equal m oracle_module && String.is_suffix name ~suffix:cause_suffix -> (
          let via = oracle_module ^ "." ^ name in
          match
            List.filter_map args ~f:(fun (_, (arg : Ppxlib.expression)) ->
                match arg.pexp_desc with
                | Pexp_construct ({ txt; _ }, _) -> Some (last_component txt)
                | _ -> None)
          with
          | [ constructor ] -> Some (constructor, Oracle via)
          | _ -> None)
      | _ -> None)
  | _ -> None

let scan_source ~file content : site list * problem list =
  let sites = ref [] and problems = ref [] in
  let in_renderer = String.equal (Stdlib.Filename.basename file) renderer_file in
  let binds ({ txt; loc } : string option Ppxlib.loc) =
    if Option.equal String.equal txt (Some oracle_module) then
      problems := Shadows_oracle { file; line = loc.loc_start.pos_lnum } :: !problems
  in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! expression expression =
        let line = expression.pexp_loc.loc_start.pos_lnum in
        (match expression.pexp_desc with
        | Pexp_construct ({ txt; _ }, arg) when String.equal (last_component txt) "Cause_at" -> (
            let phase_and_cause =
              match arg with
              | Some
                  {
                    pexp_desc =
                      Pexp_tuple [ { pexp_desc = Pexp_construct ({ txt; _ }, None); _ }; cause ];
                    _;
                  } ->
                  Some (last_component txt, cause)
              | _ -> None
            in
            match phase_and_cause with
            | None -> problems := Opaque_phase { file; line } :: !problems
            | Some (p, cause) when String.equal p render_phase -> (
                match identity cause with
                | Some (key, origin) -> sites := { file; line; key; origin } :: !sites
                | None -> problems := Unreadable { file; line } :: !problems)
            | Some _ -> if in_renderer then problems := Off_phase { file; line } :: !problems)
        | Pexp_letmodule (name, _, _) -> binds name
        | _ -> ());
        super#expression expression

      method! extension_constructor ext =
        (match ext.pext_kind with
        | Pext_rebind { txt; _ } when String.equal (last_component txt) "Cause_at" ->
            problems :=
              Rebinds_cause_at { file; line = ext.pext_loc.loc_start.pos_lnum } :: !problems
        | _ -> ());
        super#extension_constructor ext

      method! module_binding mb =
        binds mb.pmb_name;
        super#module_binding mb

      method! module_declaration md =
        binds md.pmd_name;
        super#module_declaration md

      method! functor_parameter param =
        (match param with Named (name, _) -> binds name | Unit -> ());
        super#functor_parameter param

      method! pattern pattern =
        (match pattern.ppat_desc with Ppat_unpack name -> binds name | _ -> ());
        super#pattern pattern
    end
  in
  iterator#structure (Read.structure_of content);
  (List.rev !sites, List.rev !problems)

(* The section's top-level bullets, each its start line and continuation lines joined by single
   spaces; [None] when the note has no such section. The dialect is the notes' own
   ([Agent_notes_scan]): a bullet opens with ["- "] at column 0, and an INDENTED line continues it,
   after a blank line too -- that is a second paragraph of the same bullet. An unindented line that
   is not a marker ends it (prose after the list), and so does a heading, which ends the section: a
   heading as the notes' own parser reads one ([Agent_notes_scan.atx_heading]), indented up to three
   spaces too. *)
let section_bullets note =
  let lines = String.split_lines note in
  match
    List.drop_while lines ~f:(fun l ->
        not
          (String.is_prefix (String.lstrip l) ~prefix:"## "
          && Option.equal String.equal
               (Test_utils.Agent_notes_scan.atx_heading l)
               (Some section_title)))
  with
  | [] -> None
  | _ :: rest ->
      let body =
        List.take_while rest ~f:(fun l ->
            Option.is_none (Test_utils.Agent_notes_scan.atx_heading l))
      in
      let bullets, current =
        List.fold body ~init:([], None) ~f:(fun (done_, current) line ->
            let text = String.strip line in
            match String.chop_prefix line ~prefix:"- " with
            | Some start -> (Option.to_list current @ done_, Some [ start ])
            | None when String.is_empty text -> (done_, current)
            | None when Char.is_whitespace line.[0] -> (
                match current with
                | Some parts -> (done_, Some (text :: parts))
                | None -> (done_, None))
            | None -> (Option.to_list current @ done_, None))
      in
      Some
        (List.rev_map
           (Option.to_list current @ bullets)
           ~f:(fun parts -> String.concat ~sep:" " (List.rev parts)))

(* An entry is a bullet that opens with its key in backticks; the first of the two markers after the
   key is its classification. *)
let entry_of_bullet bullet =
  match String.chop_prefix bullet ~prefix:"`" with
  | None -> None
  | Some rest -> (
      match String.lsplit2 rest ~on:'`' with
      | None -> None
      | Some (key, after) ->
          let at marker = String.substr_index after ~pattern:marker in
          let paired i =
            let named = String.drop_prefix after (i + String.length paired_marker) in
            match String.substr_index named ~pattern:"`)" with
            | Some j -> Paired (String.prefix named j)
            | None -> Unlabelled
          in
          let classification =
            match (at paired_marker, at render_only_marker) with
            | Some i, None -> paired i
            | Some i, Some j when i < j -> paired i
            | _, Some j ->
                let reason =
                  String.strip (String.drop_prefix after (j + String.length render_only_marker))
                in
                if String.is_empty reason then Unlabelled else Render_only
            | None, None -> Unlabelled
          in
          Some { entry_key = key; classification })

(* The constructors [oracle_entry] can produce: those constructed in its body or in the body of a
   top-level binding of [oracle_file] it reaches by unqualified name, transitively. *)
let oracle_reasons (structure : Ppxlib.structure) =
  let rec bound_name (p : Ppxlib.pattern) =
    match p.ppat_desc with
    | Ppat_var { txt; _ } -> Some txt
    | Ppat_constraint (p, _) -> bound_name p
    | _ -> None
  in
  let bodies = Hashtbl.create (module String) in
  List.iter structure ~f:(fun (item : Ppxlib.structure_item) ->
      match item.pstr_desc with
      | Pstr_value (_, bindings) ->
          List.iter bindings ~f:(fun (vb : Ppxlib.value_binding) ->
              Option.iter (bound_name vb.pvb_pat) ~f:(fun name ->
                  Hashtbl.add_multi bodies ~key:name ~data:vb.pvb_expr))
      | _ -> ());
  let refs_and_constructors body =
    let refs = ref [] and constructors = ref [] in
    let iterator =
      object
        inherit Ast_traverse.iter as super

        method! expression e =
          (match e.pexp_desc with
          | Pexp_ident { txt = Lident name; _ } -> refs := name :: !refs
          | Pexp_construct ({ txt; _ }, _) -> constructors := last_component txt :: !constructors
          | _ -> ());
          super#expression e
      end
    in
    iterator#expression body;
    (!refs, !constructors)
  in
  let reached = Hash_set.create (module String) in
  let reasons = Hash_set.create (module String) in
  let rec visit name =
    if not (Hash_set.mem reached name) then (
      Hash_set.add reached name;
      List.iter (Hashtbl.find_multi bodies name) ~f:(fun body ->
          let refs, constructors = refs_and_constructors body in
          List.iter constructors ~f:(Hash_set.add reasons);
          List.iter refs ~f:(fun r -> if Hashtbl.mem bodies r then visit r)))
  in
  visit oracle_entry;
  reasons

let agrees site entry =
  match (site.origin, entry.classification) with
  | Literal, Render_only -> true
  | Oracle _, Paired constructor -> String.equal constructor site.key
  | _ -> false

let audit ~sources ~note : site list * entry list * problem list =
  let scanned =
    List.map sources ~f:(fun (file, content) ->
        match Test_utils.Scan_attempt.run (fun () -> scan_source ~file content) with
        | Scanned found -> found
        | Unparsed _ -> ([], [ Unparsed file ])
        | Raised exn -> ([], [ Scanner_raised (file, exn) ]))
  in
  let sites = List.concat_map scanned ~f:fst in
  let reasons = Hash_set.create (module String) and oracle_raised = ref [] in
  List.iter sources ~f:(fun (file, content) ->
      if String.equal (Stdlib.Filename.basename file) oracle_file then
        match
          Test_utils.Scan_attempt.run (fun () -> oracle_reasons (Read.structure_of content))
        with
        | Scanned found -> Hash_set.iter found ~f:(Hash_set.add reasons)
        | Unparsed _ -> (* reported by the site scan above *) ()
        | Raised exn -> oracle_raised := Scanner_raised (file, exn) :: !oracle_raised);
  let unconsulted =
    List.filter_map sites ~f:(fun s ->
        match s.origin with
        | Oracle _ when not (Hash_set.mem reasons s.key) -> Some (Unconsulted s)
        | _ -> None)
  in
  let scan_problems = List.concat_map scanned ~f:snd @ List.rev !oracle_raised @ unconsulted in
  match section_bullets note with
  | None -> (sites, [], scan_problems @ [ No_section ])
  | Some bullets ->
      let entries = List.filter_map bullets ~f:entry_of_bullet in
      let find key = List.find entries ~f:(fun e -> String.equal e.entry_key key) in
      let site_problems =
        List.filter_map sites ~f:(fun s ->
            match find s.key with
            | None -> Some (Unclassified s)
            | Some { classification = Unlabelled; _ } -> None (* reported as [Unlabelled_entry] *)
            | Some e -> if agrees s e then None else Some (Disagrees (s, e)))
      in
      let entry_problems =
        List.concat_map entries ~f:(fun e ->
            (if List.exists sites ~f:(fun s -> String.equal s.key e.entry_key) then []
             else [ Stale e ])
            @ match e.classification with Unlabelled -> [ Unlabelled_entry e ] | _ -> [])
      in
      let duplicates =
        List.map entries ~f:(fun e -> e.entry_key)
        |> List.sort ~compare:String.compare
        |> List.group ~break:(fun a b -> not (String.equal a b))
        |> List.filter_map ~f:(function k :: _ :: _ -> Some (Duplicate_entry k) | _ -> None)
      in
      (sites, entries, scan_problems @ site_problems @ entry_problems @ duplicates)

(* Synthetic controls, each a corpus whose findings are known: both readable forms, both
   classifications, each way the two sides can disagree, and what is not a site. Every snippet but
   the parse-failure control type-checks (build-and-test.md, Repository scans) as a module of its
   own opening this prelude, which stands in for [Schedule_outcome]; the [oracle_file] fixture is
   the module [Schedule], and a corpus is checked in its listed order, the oracle fixture first:

   {v type phase = Transform | Backend_codegen | Backend_compile type resource = Workgroup_memory
   type cause = | Illegal_schedule of { check : string; detail : string } | Unsupported of { feature
   : string; detail : string } | Resource_exceeded of { resource : resource; requested : int; limit
   : int option; detail : string } exception Cause_at of phase * cause module Schedule_outcome =
   struct type nonrec phase = phase = Transform | Backend_codegen | Backend_compile exception
   Cause_at = Cause_at end module type S = sig end module M = struct end v} *)
let control_cases =
  let literal =
    "let f () = raise (Cause_at (Backend_codegen, Illegal_schedule { check = \"k\"; detail = \"d\" \
     }))"
  in
  let oracle =
    "let g () = raise (Cause_at (Backend_codegen, Schedule.v_cause ~site:\"s\" (Schedule.Ctor { t \
     = 1 })))"
  in
  let reason_and_cause =
    "type reason = Ctor of { t : int }\n\
     let v_cause ~site:_ (Ctor _) = Unsupported { feature = \"f\"; detail = \"\" }\n"
  in
  let oracle_ml =
    ( oracle_file,
      reason_and_cause
      ^ "let violation x = if x then Some (Ctor { t = 1 }) else None\n\
         let op_legality x = violation x" )
  in
  let note entries =
    "# N\n## Other\n- `k`: render-only, because elsewhere.\n" ^ section_heading
    ^ "\n- Intro `k` prose.\n" ^ entries ^ "## Next\n- `stray`: paired (`Schedule.stray`).\n"
  in
  let k_render = "- `k` (`f`): render-only, because the backend\n  decides it.\n" in
  let ctor_paired = "- `Ctor` (`g`): paired\n  (`Schedule.Ctor`). Prose.\n" in
  [
    ("a literal site with its render-only entry", [ ("a.ml", literal) ], note k_render, []);
    ("an oracle site with its paired entry", [ ("a.ml", oracle); oracle_ml ], note ctor_paired, []);
    ("a literal site with no entry", [ ("a.ml", literal) ], note "", [ "unclassified" ]);
    ( "an oracle site marked render-only",
      [ ("a.ml", oracle); oracle_ml ],
      note "- `Ctor`: render-only, because x.\n",
      [ "disagrees" ] );
    ( "a literal site marked paired",
      [ ("a.ml", literal) ],
      note "- `k`: paired (`Schedule.k`).\n",
      [ "disagrees" ] );
    ( "an oracle site paired with another constructor",
      [ ("a.ml", oracle); oracle_ml ],
      note "- `Ctor`: paired (`Schedule.Other`).\n",
      [ "disagrees" ] );
    ( "an oracle site whose constructor op_legality only matches, built where it does not reach",
      [
        ("a.ml", oracle);
        ( oracle_file,
          reason_and_cause
          ^ "let unreached () = Ctor { t = 1 }\n\
             let feature (Ctor _) = \"f\"\n\
             let op_legality r = feature r" );
      ],
      note ctor_paired,
      [ "unconsulted" ] );
    ("an entry no site raises", [ ("a.ml", literal) ], note (k_render ^ ctor_paired), [ "stale" ]);
    ( "an entry with neither label",
      [ ("a.ml", literal) ],
      note "- `k`: the backend decides it.\n",
      [ "unlabelled" ] );
    ( "an empty render-only reason",
      [ ("a.ml", literal) ],
      note "- `k`: render-only, because \n",
      [ "unlabelled" ] );
    ( "two entries for one decline",
      [ ("a.ml", literal) ],
      note (k_render ^ k_render),
      [ "duplicate" ] );
    ( "a cause with a computed check",
      [
        ( "a.ml",
          "let f c = raise (Cause_at (Backend_codegen, Illegal_schedule { check = c; detail = \"\" \
           }))" );
      ],
      note "",
      [ "unreadable" ] );
    ( "a resource named by its constructor",
      [
        ( "a.ml",
          "let f () = raise (Cause_at (Backend_codegen, Resource_exceeded { resource = \
           Workgroup_memory; requested = 1; limit = None; detail = \"\" }))" );
      ],
      note "- `Workgroup_memory`: render-only, because x.\n",
      [] );
    ( "a cause held in a variable",
      [
        ( "a.ml",
          "let f cause = raise (Schedule_outcome.Cause_at (Schedule_outcome.Backend_codegen, \
           cause))" );
      ],
      note "",
      [ "unreadable" ] );
    ( "an unqualified cause function",
      [
        ( "a.ml",
          "let v_cause = Schedule.v_cause\n\
           let f () = raise (Cause_at (Backend_codegen, v_cause ~site:\"s\" (Schedule.Ctor { t = 1 \
           })))" );
        oracle_ml;
      ],
      note "",
      [ "unreadable" ] );
    ( "another phase outside the renderer",
      [
        ( "a.ml",
          "let f () = raise (Cause_at (Transform, Illegal_schedule { check = \"t\"; detail = \"\" \
           }))" );
      ],
      note "",
      [] );
    ( "another phase in the renderer",
      [
        ( "c_syntax.ml",
          "let f () = raise (Cause_at (Backend_compile, Illegal_schedule { check = \"t\"; detail = \
           \"\" }))" );
      ],
      note "",
      [ "off-phase" ] );
    ( "a phase held in a variable in the renderer",
      [
        ( "c_syntax.ml",
          "let f p = raise (Cause_at (p, Illegal_schedule { check = \"t\"; detail = \"\" }))" );
      ],
      note "",
      [ "opaque-phase" ] );
    ( "a phase held in a variable outside the renderer",
      [
        ("a.ml", "let f p = raise (Cause_at (p, Illegal_schedule { check = \"t\"; detail = \"\" }))");
      ],
      note "",
      [ "opaque-phase" ] );
    ( "a constrained phase outside the renderer",
      [
        ( "a.ml",
          "let f () = raise (Cause_at ((Backend_codegen : phase), Illegal_schedule { check = \
           \"t\"; detail = \"\" }))" );
      ],
      note "",
      [ "opaque-phase" ] );
    ( "Cause_at rebound under another name, at top level",
      [
        ( "a.ml",
          "exception Decline = Schedule_outcome.Cause_at\n\
           let f () = raise (Decline (Backend_codegen, Illegal_schedule { check = \"t\"; detail = \
           \"\" }))" );
      ],
      note "",
      [ "rebinds-cause-at" ] );
    (* A local rebinding, in the form OCaml 5.3 parses: [let exception D = Cause_at in] is 5.5
       syntax, which no source can use while 5.3 is the floor (gh-ocannl-1588). *)
    ( "Cause_at rebound under another name, in a local module",
      [
        ( "a.ml",
          "let f () =\n\
          \  let module L = struct exception D = Cause_at end in\n\
          \  raise (L.D (Backend_codegen, Illegal_schedule { check = \"t\"; detail = \"\" }))" );
      ],
      note "",
      [ "rebinds-cause-at" ] );
    ( "a local module shadowing the oracle",
      [ ("a.ml", "module Schedule = struct\n" ^ reason_and_cause ^ "end\n" ^ oracle); oracle_ml ],
      note ctor_paired,
      [ "shadows-oracle" ] );
    ( "the oracle's name bound by let module, a functor, a declaration and an unpacked module",
      [
        ( "a.ml",
          "let f () = let module Schedule = M in ()\n\
           module F (Schedule : S) = struct end\n\
           module type T = sig module Schedule : S end\n\
           let g (module Schedule : S) = ()" );
      ],
      note "",
      [ "shadows-oracle"; "shadows-oracle"; "shadows-oracle"; "shadows-oracle" ] );
    ( "an aliased oracle module",
      [
        ( "a.ml",
          "module S = Schedule\n\
           let g () = raise (Cause_at (Backend_codegen, S.v_cause ~site:\"s\" (S.Ctor { t = 1 })))"
        );
        oracle_ml;
      ],
      note "",
      [ "unreadable" ] );
    ( "a pattern naming the phase",
      [ ("c_syntax.ml", "let f = function Cause_at (Backend_codegen, _) -> () | _ -> ()") ],
      note "",
      [] );
    ( "a comment and a string naming a site",
      [
        ( "c_syntax.ml",
          "(* Cause_at (Backend_codegen, x) *)\nlet s = \"Cause_at (Backend_codegen, x)\"" );
      ],
      note "",
      [] );
    ( "a classification in the bullet's second paragraph",
      [ ("a.ml", literal) ],
      note "- `k` (`f`): the backend decides it.\n\n  So: render-only, because x.\n",
      [] );
    ( "the section under an indented heading with a closing run of hashes",
      [ ("a.ml", literal) ],
      "# N\n  ## " ^ section_title ^ " ##\n" ^ k_render,
      [] );
    ( "a bullet under an indented next heading, which is outside the section",
      [ ("a.ml", literal) ],
      "# N\n" ^ section_heading ^ "\n" ^ k_render
      ^ "   ## Next\n- `stray`: paired (`Schedule.stray`).\n",
      [] );
    ( "unindented prose after the bullet, which is not part of it",
      [ ("a.ml", literal) ],
      note "- `k` (`f`): the backend decides it.\n\nrender-only, because x.\n",
      [ "unlabelled" ] );
    ("a source that does not parse", [ ("a.ml", "let f = (") ], note "", [ "unparsed" ]);
    ( "a note without the section",
      [ ("a.ml", literal) ],
      "# N\n## Other\n" ^ k_render,
      [ "no-section" ] );
  ]

let source_dir = "arrayjit/lib/"
let note_path = "docs/agent-notes/scheduling-and-autotune.md"

(* Binary streams, as the sibling scans keep them: Windows' text mode would end each line "\r\n". *)
let () =
  Stdlib.set_binary_mode_out Stdlib.stdout true;
  Stdlib.set_binary_mode_out Stdlib.stderr true;
  let inventory =
    match Array.to_list Stdlib.Sys.argv with
    | _ :: workspace_root :: generated ->
        Test_utils.Source_inventory.of_dune_sandbox ~workspace_root ~generated
    | _ ->
        eprintf "usage: render_decline_inventory <workspace root> <generated files...>\n";
        Stdlib.exit 2
  in
  List.iter control_cases ~f:(fun (label, sources, note, expected) ->
      let _, _, problems = audit ~sources ~note in
      let found = List.map problems ~f:problem_kind in
      pf "control: %s finds [%s]" label
        (String.concat ~sep:"; " expected)
        (List.equal String.equal found expected));
  let read (file : Test_utils.Source_inventory.file) = In_channel.read_all file.on_disk in
  let sources =
    Test_utils.Source_inventory.select inventory ~f:(fun path ->
        String.is_prefix path ~prefix:source_dir && String.is_suffix path ~suffix:".ml")
    |> List.map ~f:(fun (file : Test_utils.Source_inventory.file) -> (file.path, read file))
  in
  let note_file =
    List.find (Test_utils.Source_inventory.files inventory) ~f:(fun file ->
        String.equal file.path note_path)
  in
  let sites, entries, problems =
    audit ~sources ~note:(Option.value_map note_file ~default:"" ~f:read)
  in
  let problems =
    if Option.is_some note_file then problems else Note_missing note_path :: problems
  in
  eprintf "render_decline_inventory: %d sources, %d sites, %d entries (not part of the golden)\n"
    (List.length sources) (List.length sites) (List.length entries);
  List.iter problems ~f:(fun problem -> eprintf "%s\n" (describe problem));
  let of_kinds kinds =
    List.filter problems ~f:(fun pr -> List.mem kinds (problem_kind pr) ~equal:String.equal)
  in
  p_empty "the agent note has the render-time typed declines section" ~over:sources
    (of_kinds [ "no-section"; "note-missing" ]);
  p_empty "every arrayjit/lib source parses, and the scan reads it without raising" ~over:sources
    (of_kinds [ "unparsed"; "raised" ]);
  p_all "the scan finds render-time typed declines, and every one has an entry classifying it" sites
    ~f:(fun s ->
      not (List.exists problems ~f:(function Unclassified u -> phys_equal u s | _ -> false)));
  p_all "every entry's classification is how its site builds the decline: paired iff oracle-built"
    sites ~f:(fun s ->
      not (List.exists problems ~f:(function Disagrees (u, _) -> phys_equal u s | _ -> false)));
  p_all "every entry names a decline some site raises, once, as paired or render-only" entries
    ~f:(fun e ->
      not
        (List.exists problems ~f:(function
          | Stale u | Unlabelled_entry u -> phys_equal u e
          | Duplicate_entry k -> String.equal k e.entry_key
          | _ -> false)));
  p_empty
    "every Backend_codegen cause is readable, and the renderer raises typed declines only there"
    ~over:sources
    (of_kinds [ "unreadable"; "off-phase" ]);
  p_empty
    "every Cause_at spells its phase out under its own name, and no source rebinds the oracle's \
     module name"
    ~over:sources
    (of_kinds [ "opaque-phase"; "shadows-oracle"; "rebinds-cause-at" ]);
  p_all "every paired decline's constructor is one op_legality still produces" sites ~f:(fun s ->
      not (List.exists problems ~f:(function Unconsulted u -> phys_equal u s | _ -> false)));
  p_exists "some decline is paired with an oracle constructor" sites ~f:(fun s ->
      match s.origin with Oracle _ -> true | Literal -> false);
  Test_utils.Refusal_control_manifest.print "render_decline_inventory.ml"
