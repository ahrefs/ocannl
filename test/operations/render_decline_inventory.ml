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

   A [Backend_codegen] cause the scan cannot key (a variable, a computed check name) fails: a new
   decline has to be written in one of the readable forms, which is what makes it countable. And in
   [renderer_file] every [Cause_at] is constructed at [Backend_codegen] with the phase spelled out,
   so a render-time decline cannot leave the inventory by naming another phase.

   Sites are read on the parse tree: a pattern ([| Cause_at (Backend_codegen, _) ->]) is not a
   construction, and a comment or string naming one is not a site. *)

open Base
open Stdio
open Verdict.Claims
module Read = Test_utils.Config_key_scan
module Ast_traverse = Ppxlib.Ast_traverse

let render_phase = "Backend_codegen"
let renderer_file = "c_syntax.ml"
let section_heading = "## Render-time typed declines"

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
  | Unclassified of site
  | Disagrees of site * entry
  | Stale of entry
  | Unlabelled_entry of entry
  | Duplicate_entry of string

let problem_kind = function
  | No_section -> "no-section"
  | Unreadable _ -> "unreadable"
  | Off_phase _ -> "off-phase"
  | Unclassified _ -> "unclassified"
  | Disagrees _ -> "disagrees"
  | Stale _ -> "stale"
  | Unlabelled_entry _ -> "unlabelled"
  | Duplicate_entry _ -> "duplicate"

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
        "%s:%d: a Cause_at in the renderer not constructed at a spelled-out %s phase: a \
         render-time decline is a %s decline"
        file line render_phase render_phase
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
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! expression expression =
        let line = expression.pexp_loc.loc_start.pos_lnum in
        (match expression.pexp_desc with
        | Pexp_construct ({ txt; _ }, arg) when String.equal (last_component txt) "Cause_at" -> (
            let phase, cause =
              match arg with
              | Some { pexp_desc = Pexp_tuple [ phase; cause ]; _ } -> (
                  match phase.pexp_desc with
                  | Pexp_construct ({ txt; _ }, None) -> (Some (last_component txt), Some cause)
                  | _ -> (None, Some cause))
              | _ -> (None, None)
            in
            match (phase, cause) with
            | Some p, Some cause when String.equal p render_phase -> (
                match identity cause with
                | Some (key, origin) -> sites := { file; line; key; origin } :: !sites
                | None -> problems := Unreadable { file; line } :: !problems)
            | _ -> if in_renderer then problems := Off_phase { file; line } :: !problems)
        | _ -> ());
        super#expression expression
    end
  in
  iterator#structure (Read.structure_of content);
  (List.rev !sites, List.rev !problems)

(* The section's top-level bullets, each its start line and continuation lines joined by single
   spaces; [None] when the note has no such section. *)
let section_bullets note =
  let lines = String.split_lines note in
  match
    List.drop_while lines ~f:(fun l -> not (String.equal (String.rstrip l) section_heading))
  with
  | [] -> None
  | _ :: rest ->
      let body = List.take_while rest ~f:(fun l -> not (String.is_prefix l ~prefix:"#")) in
      let bullets, current =
        List.fold body ~init:([], None) ~f:(fun (done_, current) line ->
            match String.chop_prefix line ~prefix:"- " with
            | Some start -> (Option.to_list current @ done_, Some [ start ])
            | None -> (
                let text = String.strip line in
                match current with
                | Some parts when not (String.is_empty text) -> (done_, Some (text :: parts))
                | _ -> (Option.to_list current @ done_, None)))
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

let agrees site entry =
  match (site.origin, entry.classification) with
  | Literal, Render_only -> true
  | Oracle _, Paired constructor -> String.equal constructor site.key
  | _ -> false

let audit ~sources ~note : site list * entry list * problem list =
  let scanned = List.map sources ~f:(fun (file, content) -> scan_source ~file content) in
  let sites = List.concat_map scanned ~f:fst in
  let scan_problems = List.concat_map scanned ~f:snd in
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
   classifications, each way the two sides can disagree, and what is not a site. *)
let control_cases =
  let literal =
    "let f () = raise (Cause_at (Backend_codegen, Illegal_schedule { check = \"k\"; detail = \"d\" \
     }))"
  in
  let oracle =
    "let g () = raise (Cause_at (Backend_codegen, Schedule.v_cause ~site:\"s\" (Schedule.Ctor { t \
     = 1 })))"
  in
  let note entries =
    "# N\n## Other\n- `k`: render-only, because elsewhere.\n" ^ section_heading
    ^ "\n- Intro `k` prose.\n" ^ entries ^ "## Next\n- `stray`: paired (`Schedule.stray`).\n"
  in
  let k_render = "- `k` (`f`): render-only, because the backend\n  decides it.\n" in
  let ctor_paired = "- `Ctor` (`g`): paired\n  (`Schedule.Ctor`). Prose.\n" in
  [
    ("a literal site with its render-only entry", [ ("a.ml", literal) ], note k_render, []);
    ("an oracle site with its paired entry", [ ("a.ml", oracle) ], note ctor_paired, []);
    ("a literal site with no entry", [ ("a.ml", literal) ], note "", [ "unclassified" ]);
    ( "an oracle site marked render-only",
      [ ("a.ml", oracle) ],
      note "- `Ctor`: render-only, because x.\n",
      [ "disagrees" ] );
    ( "a literal site marked paired",
      [ ("a.ml", literal) ],
      note "- `k`: paired (`Schedule.k`).\n",
      [ "disagrees" ] );
    ( "an oracle site paired with another constructor",
      [ ("a.ml", oracle) ],
      note "- `Ctor`: paired (`Schedule.Other`).\n",
      [ "disagrees" ] );
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
      [ ("a.ml", "let f () = raise (Cause_at (Backend_codegen, v_cause (Ctor 1)))") ],
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
      [ "off-phase" ] );
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
    ( "a note without the section",
      [ ("a.ml", literal) ],
      "# N\n## Other\n" ^ k_render,
      [ "no-section" ] );
  ]

let () =
  let note_path, sources =
    match Array.to_list Stdlib.Sys.argv with
    | _ :: note :: deps -> (note, Read.sources_among deps)
    | _ ->
        eprintf "usage: render_decline_inventory <agent note> <arrayjit/lib sources...>\n";
        Stdlib.exit 2
  in
  List.iter control_cases ~f:(fun (label, sources, note, expected) ->
      let _, _, problems = audit ~sources ~note in
      let found = List.map problems ~f:problem_kind in
      pf "control: %s finds [%s]" label
        (String.concat ~sep:"; " expected)
        (List.equal String.equal found expected));
  let sources = List.map sources ~f:(fun file -> (file, In_channel.read_all file)) in
  let sites, entries, problems = audit ~sources ~note:(In_channel.read_all note_path) in
  eprintf "render_decline_inventory: %d sources, %d sites, %d entries (not part of the golden)\n"
    (List.length sources) (List.length sites) (List.length entries);
  List.iter problems ~f:(fun problem -> eprintf "%s\n" (describe problem));
  let of_kinds kinds =
    List.filter problems ~f:(fun pr -> List.mem kinds (problem_kind pr) ~equal:String.equal)
  in
  p_empty "the agent note has the render-time typed declines section" ~over:sources
    (of_kinds [ "no-section" ]);
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
  p_exists "some decline is paired with an oracle constructor" sites ~f:(fun s ->
      match s.origin with Oracle _ -> true | Literal -> false);
  Test_utils.Refusal_control_manifest.print "render_decline_inventory.ml"
