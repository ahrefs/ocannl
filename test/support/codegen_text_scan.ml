(** What in this repository pins the TEXT of generated code (gh-ocannl-712).

    A change to code generation -- how a float constant is spelled, how a loop is opened, which
    intrinsic a reduction renders as -- has a blast radius made of two populations that no single
    search finds:

    - {b goldens}, [.expected] files holding emitted kernel or IR source, in [test/] and in
      [arrayjit/test/] both. Scanning one tree and concluding is how gh-ocannl-623's first CI run
      went red: three [arrayjit/test] goldens quote emitted constants and no [test/*] glob sees
      them.
    - {b test sources}, which assert on emitted text from a string literal in the [.ml] rather than
      from a golden -- [Generated.assert_emits ~contains:"..."], or [Generated.read] followed by a
      substring test. No [.expected] scan of any thoroughness finds these, and because they are
      {!Verdict} claims they exit nonzero, so they fail a plain [dune build] rather than only
      [dune runtest]. That is what CI actually tripped on.

    This module decides both, as pure functions over a path and its contents, so that
    [codegen_text_inventory] can run them over the live tree and [codegen_text_scan_cases] can run
    the same code over input built to break it.

    {1 How a golden is recognised}

    Two independent routes, either sufficient.

    By {b extension}: [.c.expected], [.cu.expected], [.hip.expected], [.metal.expected],
    [.ll.expected], [.cd.expected]. These name the artifact they snapshot, so they are members
    whatever they contain -- on a machine without the toolchain such a file can hold a skip notice
    and it is still that backend's snapshot, to be re-recorded when the hardware next runs.

    By {b content markers}: a file whose text carries the syntax of an emitted kernel or of the
    low-level IR dump. Markers are grouped into families (c, cuda, hip, metal, ll, routine-log) so
    that the inventory says which substrate a member needs re-running on, and each is a string that
    only emitted text spells -- [for (int32_t ], [*restrict ], [(float)(], [__global__],
    [threadgroup float], [[] := ], [/* end */].

    A marker only counts on a line that is not a {!Verdict} claim. Claim labels are prose ABOUT a
    kernel and routinely quote its vocabulary --
    ["padded GPU intrinsics fire against the threadgroup fragment: true"] is a verdict, not Metal
    source -- and a golden made of such lines moves when the claim is reworded, never when codegen
    changes.

    {1 How a source site is recognised}

    A file is a member when it reaches generated text at all, by any of three routes -- and it is
    three because a rule naming two of them missed a whole population once already (Codex P2, round
    2):

    - through {!Test_utils.Generated}, the freshness-checked artifact reader;
    - by opening [build_files/] itself, which two tests predating that module still do, and which
      the inventory tags, since such a read is unchecked for freshness;
    - {b in memory}, by calling an emitter and rendering the document it returns, or by handing one
      the buffer to write into -- [C_syntax]'s [compile_proc] / [compile_main], [Low_level]'s
      [to_doc] / [to_doc_cstyle], [Canonical_render]'s [emit]. Such a test never touches
      [build_files/] at all, and every one of them was invisible: the three [arrayjit/test] codegen
      tests whose GOLDENS the inventory already listed, plus [ll_printer_constants], which pins the
      spelling of every dumped float constant. WHICH values those are is not decided here: the
      caller passes the set, and [codegen_text_inventory] derives it from the compiler libraries'
      interfaces through {!Emitter_frontier} (gh-ocannl-748), so a renderer added to a library is on
      the frontier the day it is exported.

    Under each member the inventory itemises the text it pins, found by the same literal-spelling
    discipline the configuration scan uses: the argument of a substring test, {e at the call site}.
    A [Printf.sprintf] format is itemised as well, because a pinned fragment with a hole in it --
    ["< (int)(%d.0))) {"] -- is exactly the context gh-ocannl-623 was found in only by reading a
    failing kernel.

    Anything else the scan reports rather than assumes: a site whose text is computed, or reached
    through a helper that takes it as a parameter, marks the file's itemisation partial. That is the
    documented limit of the discipline, and the reason it is a limit rather than a hole: the FILE is
    still listed, so a codegen change is still told to re-run it -- only the fragment cannot be
    named.

    A compiler-plan classifier in a test that also reads generated source is the one deliberately
    annotated exception, through the [ocannl.codegen_text.compiler_plan] attribute. Its text tests
    describe compiler flags, not emitted code, and are excluded only within that binding. A
    generated-text assertion elsewhere in the same file remains a pin.

    {1 Reading these sources as OCaml}

    The source half parses, for the reasons {!Config_key_scan}'s header sets out at length: an
    approximation of OCaml has no natural stopping point, the grammar does. The golden half matches
    text, because a golden is not a language -- it is whatever the backend printed. *)

open Base
open Ppxlib.Parsetree
module Ast_traverse = Ppxlib.Ast_traverse
module Asttypes = Ppxlib.Asttypes
module Longident = Ppxlib.Longident
module Parse = Ppxlib.Parse

(* ------------------------------------------------------------------ goldens *)

(** The families a member belongs to. The name is what the inventory prints, and what a reader greps
    for when asking "which backends must re-record". *)
type family = C | Cuda | Hip | Metal | Ll | Routine_log

let family_name = function
  | C -> "c"
  | Cuda -> "cuda"
  | Hip -> "hip"
  | Metal -> "metal"
  | Ll -> "ll"
  | Routine_log -> "routine-log"

let family_rank = function C -> 0 | Cuda -> 1 | Hip -> 2 | Metal -> 3 | Ll -> 4 | Routine_log -> 5

(** Extensions that DECLARE the artifact snapshotted, ordered longest-first so that [.cu.expected]
    is tried before a hypothetical [.expected]. A file matching one is a member whatever it
    contains. *)
let declared_extensions =
  [
    (".c.expected", C);
    (".cu.expected", Cuda);
    (".hip.expected", Hip);
    (".metal.expected", Metal);
    (".msl.expected", Metal);
    (".ll.expected", Ll);
    (".cd.expected", Ll);
  ]

(** A line the scan must not read as emitted text: the output of {!Verdict}, whose labels are prose
    about a kernel and freely quote its vocabulary. *)
let is_claim_line line =
  let line = String.rstrip line in
  String.is_prefix line ~prefix:"FAIL: "
  || List.exists [ ": true"; ": false"; ": PASS"; ": FAIL" ] ~f:(fun suffix ->
      String.is_suffix line ~suffix)

(** A low-level-IR loop header, [for i12 = 0 to 4 {]. Recognised by shape rather than by a needle
    because the induction variable carries a number: ["for i"], digits, [" = "]. *)
let is_ll_loop_line line =
  let n = String.length line in
  let rec from i =
    match String.substr_index ~pos:i line ~pattern:"for i" with
    | None -> false
    | Some start ->
        let j = ref (start + String.length "for i") in
        while !j < n && Char.is_digit line.[!j] do
          Int.incr j
        done;
        !j > start + String.length "for i"
        && String.is_prefix (String.drop_prefix line !j) ~prefix:" = "
        || from (start + 1)
  in
  from 0

type test = Needles of string list | Shape of (string -> bool)
type marker = { tag : string; family : family; test : test }

(** The content markers, by family.

    Each needle is a string only emitted text spells. Two temptations were measured and rejected: a
    bare ["threadgroup "] (matches the claim label "against the threadgroup fragment"), and
    ["device "] (matches every [On_device] table and every "device footprint" verdict in the tree).
    The survivors produce no false member over the repository as it stands, and the cases test pins
    the near misses. *)
let markers =
  [
    (* Both index widths: [large_models] makes the loop index [int64_t], and a golden taken under it
       would otherwise show no [c-for] at all. *)
    { tag = "c-for"; family = C; test = Needles [ "for (int32_t "; "for (int64_t " ] };
    { tag = "c-restrict"; family = C; test = Needles [ "*restrict " ] };
    { tag = "c-prec-cast"; family = C; test = Needles [ "(float)("; "(double)("; "(int)(" ] };
    { tag = "c-decl-banner"; family = C; test = Needles [ "/* Local declarations" ] };
    { tag = "c-logic-banner"; family = C; test = Needles [ "/* Main logic. */" ] };
    { tag = "c-align-attr"; family = C; test = Needles [ "__attribute__((aligned" ] };
    {
      tag = "cuda-kernel";
      family = Cuda;
      test = Needles [ "__global__"; "threadIdx."; "blockIdx."; "__shared__"; "__syncthreads(" ];
    };
    {
      tag = "metal-kernel";
      family = Metal;
      test =
        Needles
          [
            "kernel void ";
            "[[kernel]]";
            "threadgroup float";
            "threadgroup half";
            "threadgroup_barrier";
            "simdgroup_";
            "[[thread_position_in_";
          ];
    };
    { tag = "ll-loop"; family = Ll; test = Shape is_ll_loop_line };
    (* Both spellings the IR has: the pretty dump separates the arrow with spaces, and
       [Canonical_render]'s compact serialization does not. A marker keyed on one of them is keyed
       on whitespace, which is not what makes a line an assignment (Codex P2, round 4). *)
    { tag = "ll-assign"; family = Ll; test = Needles [ "] := "; "]:=" ] };
    { tag = "ll-end"; family = Ll; test = Needles [ "/* end */" ] };
    {
      tag = "routine-log";
      family = Routine_log;
      test = Needles [ "{=MAYBE UNINITIALIZED}"; "COMMENT: " ];
    };
  ]

let marker_matches marker line =
  match marker.test with
  | Needles needles -> List.exists needles ~f:(fun n -> String.is_substring line ~substring:n)
  | Shape f -> f line

type golden = {
  path : string;
  by_extension : string option;
      (** The declaring extension, when the file has one: a member whatever it contains. *)
  families : string list;  (** Sorted family names, for the inventory line. *)
  tags : string list;  (** Sorted content-marker tags; empty for an extension-only member. *)
  beside : string option;
      (** The source member this golden belongs to, when nothing about the file itself made it a
          member. See {!classify_associated}. *)
}

(** The family of a golden that holds text DERIVED from generated code in a shape no marker names: a
    table of dumped constants, a census of the schedule decisions a kernel was built from. *)
let derived_family = "derived"

(** [classify_golden ~path ~contents] is [Some] when the file pins emitted text. [path] is used for
    its extension only, so it may carry any prefix dune's globs put on it. *)
let classify_golden ~path ~contents =
  let basename = Stdlib.Filename.basename path in
  let by_extension =
    List.find declared_extensions ~f:(fun (ext, _) -> String.is_suffix basename ~suffix:ext)
  in
  let lines = String.split_lines contents |> List.filter ~f:(fun l -> not (is_claim_line l)) in
  let hit marker = List.exists lines ~f:(marker_matches marker) in
  let matched = List.filter markers ~f:hit in
  match (by_extension, matched) with
  | None, [] -> None
  | _ ->
      (* The DECLARING extension is authoritative about which substrate produced the file, and the
         markers are evidence only where there is none. CUDA and HIP spell the same launch
         vocabulary, so a [.hip.expected] matching [cuda-kernel] is HIP text, not CUDA text -- and a
         snapshot named for its backend needs re-recording on that backend whatever else its markers
         say. Where nothing declares, the markers are all there is, and a file carrying several
         dialects (a routine log holds both the IR line and the C rendering) names them all. *)
      let families =
        (match by_extension with
          | Some (_, family) -> [ family ]
          | None -> List.map matched ~f:(fun m -> m.family))
        |> List.dedup_and_sort ~compare:(fun a b -> Int.compare (family_rank a) (family_rank b))
        |> List.map ~f:family_name
      in
      Some
        {
          path;
          by_extension = Option.map by_extension ~f:fst;
          families;
          tags = List.map matched ~f:(fun m -> m.tag) |> List.dedup_and_sort ~compare:String.compare;
          beside = None;
        }

(** Whether every line of [contents] is {!Verdict} output: a golden made of claims and nothing else.

    This is what tells a test's OWN golden apart from a golden that holds what the test rendered. A
    boolean column does not move when codegen does -- the claim goes on reading [true] -- so pulling
    such a file into the inventory would add a line per schedule test and train the reader to skim.
    A line that is not a claim is the test printing something, and where the test renders generated
    text that something is derived from it. *)
let holds_only_claims contents =
  String.split_lines contents
  |> List.for_all ~f:(fun line -> String.is_empty (String.strip line) || is_claim_line line)

(** The stem a test source is known by: its path without [.ml], and without the [.real] / [.missing]
    infix a [(select)] pair carries. *)
let source_stem path =
  let stem = Option.value (String.chop_suffix path ~suffix:".ml") ~default:path in
  match String.chop_suffix stem ~suffix:".real" with
  | Some stem -> stem
  | None -> Option.value (String.chop_suffix stem ~suffix:".missing") ~default:stem

(** [classify_associated ~path ~contents ~source] is [Some] when a golden that no extension declares
    and no marker recognises is still a member, because it is the golden of a test that renders
    generated text and it holds more than that test's verdicts.

    The route exists because the markers describe whole dumps -- a loop nest, a launch signature, an
    assignment -- and a golden can pin emitted text in fragments instead: a table whose columns are
    the [%cd] and C-style spellings of one constant, a census of the schedule decisions a kernel was
    built from. Those move when codegen moves, and no marker written for kernel syntax will ever see
    them (Codex P2, round 2). What makes the association sound rather than a guess is the pairing:
    the test beside it demonstrably reaches generated text, so what it prints comes from there.

    Only the exact stem pairs. A test writing a differently-named golden
    ([micrograd_demo_logging-cc-0-0.log.expected]) is found by content, which is the primary route
    and stays so. *)
let classify_associated ~path ~contents ~source =
  if holds_only_claims contents then None
  else
    Some
      { path; by_extension = None; families = [ derived_family ]; tags = []; beside = Some source }

(* ------------------------------------------------------------- source sites *)

let string_literal expr =
  match expr.pexp_desc with Pexp_constant (Pconst_string (value, _, _)) -> Some value | _ -> None

(* [Longident.flatten_exn] fatal-errors on a functor application, which cannot arise where it is
   used here: in EXPRESSION position OCaml gives no [Pexp_ident] an applied path, and a MODULE
   expression reaches it only through [Pmod_ident], which is a plain path by construction. The same
   reasoning [Config_key_scan]'s header sets out at length. *)
let flatten_longident = Longident.flatten_exn

let longident_of expr =
  match expr.pexp_desc with Pexp_ident { txt; _ } -> Some (flatten_longident txt) | _ -> None

(** Raises if [content] does not parse: a scan that cannot read its input must say so rather than
    report an empty census. *)
let structure_of content = Parse.implementation (Lexing.from_string content)

let path_ends path ~name = match List.last path with Some n -> String.equal n name | None -> false

(** Every unqualified identifier occurring in [expr]: the names taint can travel along. *)
let idents_in expr =
  let found = ref [] in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! expression e =
        (match e.pexp_desc with
        | Pexp_ident { txt = Longident.Lident name; _ } -> found := name :: !found
        | _ -> ());
        super#expression e
    end
  in
  iterator#expression expr;
  !found

(** The module a qualified path calls into: the component before the last. [Generated.read] and
    [Test_utils.Generated.read] both answer ["Generated"], [G.read] answers ["G"], and an
    unqualified [read] answers nothing -- which is what keeps a test's own local [read] from being
    taken for the artifact reader. *)
let qualifier_of path =
  match List.rev path with _last :: qualifier :: _ -> Some qualifier | _ -> None

(** Reads of [build_files/] that do not go through {!Test_utils.Generated}, and so are unchecked for
    freshness: [Utils.build_files_dir], [Utils.build_file].

    The module qualifier is required, and that is the whole point: [build_file] is an ordinary name
    a test may bind for itself ([test_safetensors] writes its safetensors fixtures through one),
    while [Utils.build_file] is a read of the artifact directory. Keying on the last component alone
    read the first as the second. Aliases of [Utils] resolve like aliases of [Generated]. *)
let direct_artifact_names = [ "build_files_dir"; "build_file" ]

let direct_artifact_module = "Utils"

type destination =
  | At_label of string
  | At_position of int
      (** Where a call to a buffer-writing emitter leaves its text: an argument named by its label,
          or the n-th of the arguments that carry none. Positions count only the unlabelled
          arguments, so an optional argument the call site omits does not shift them. *)

type emitter = {
  emitter_name : string;  (** The value's name, which is what a call site spells. *)
  origins : string list;
      (** The qualified paths defining it, as the interfaces spell them
          ([Ir.Low_level.Canonical_render.emit]). Used to reject an [open] that would hide a call
          from this scan. Empty for a local name this file bound to an emitter. *)
  destinations : destination list;
      (** The arguments generated text lands in, for an emitter that writes into a buffer rather
          than returning a document. *)
}
(** An emitter: a library value that hands a test generated text without an artifact in between --
    [C_syntax.compile_proc] / [compile_main], [Low_level.to_doc] / [to_doc_cstyle],
    [Canonical_render.emit].

    This set is DERIVED from the compiler libraries' interfaces rather than listed here
    (gh-ocannl-748; see {!Emitter_frontier}, and [codegen_text_inventory] for the hand-over). It was
    a written list until then, and it was the one hand-maintained frontier left in this scan: three
    of the four review rounds on gh-ocannl-712 found a member of exactly that shape, because a route
    the list does not name does not shrink the inventory visibly -- it just leaves files off it.

    Matched by NAME behind any qualifier, rather than against a resolved module. That is deliberate,
    and it is the one place this scan errs toward including: the qualifier here is routinely a local
    module bound by a FUNCTOR APPLICATION -- [let module Syntax = Ir.C_syntax.C_syntax (...) in] --
    which no alias table can resolve to a target, so demanding one would reinstate exactly the blind
    spot this family exists to close. A qualifier is still required, so a test's own [to_doc] is not
    swept in; and if some unrelated [X.to_doc] appears one day it costs an inventory line, whereas a
    miss here costs a silent omission. What a qualifier cannot survive is an [open], which is why
    {!rejections} refuses that spelling outright instead of guessing. *)

(* -------------------------------------------------------------- lexical scope *)

(** What a module name denotes where it is spelled: the module it NAMES, by the last component of
    the path an alias chain reaches -- [Generated] for [Test_utils.Generated] and for
    [module G = Test_utils.Generated] alike -- or, when a binding this scan cannot see into took the
    name over (a [struct], a functor parameter, an unpack), nothing it attributes.

    A name no binding in the file reaches is the library module of that name: [Generated], [Utils],
    [Buffer]. A functor APPLICATION names its functor -- [Ir.C_syntax.C_syntax (Cfg)] is the module
    whose values the interfaces record under [C_syntax], and is how every backend and codegen test
    reaches [compile_proc] -- and a signature constraint is the module it constrains
    ([module B = (Buffer : module type of Buffer)]), since otherwise the backstop reading
    [B.contents] does not fire, and a backstop that fails silently is worse than none (Codex rounds
    4 and 5 on lukstafi/ocannl-staging#487). *)
type module_denotes = Named of string | Opaque

(** What an unqualified value name denotes where it is spelled: the string a [let] binds it to
    directly, a name an [open] of an attributed module made unqualified ({!rejections}), or any
    other binding the file makes. *)
type value_denotes =
  | Literal_binding of string
  | Hidden of { opened : string; name : string }
  | Function_binding of (int * int)
  | Bound

type scope = {
  qualifiers : (int * int, module_denotes) Hashtbl.t;
      (** For a qualified identifier: what its qualifier denotes. *)
  values : (int * int, value_denotes) Hashtbl.t;
      (** For an unqualified one: the binding it reaches, absent for a name the file does not bind.
      *)
}
(** Each identifier of a file resolved where it is spelled, keyed by its source span. *)

let span (loc : Ppxlib.Location.t) = (loc.loc_start.pos_cnum, loc.loc_end.pos_cnum)

(** The names an [open] or [include] of the module named [target] makes unqualified, among those
    this scan attributes by their qualifier: the artifact readers by their module, the emitters by
    the module whose interface defines them. *)
let hidden_names ~emitters target =
  (if String.equal target "Generated" then [ "read"; "assert_emits"; "assert_omits" ] else [])
  @ (if String.equal target direct_artifact_module then direct_artifact_names else [])
  @ List.filter_map emitters ~f:(fun emitter ->
      let defined_in origin =
        match List.rev (String.split origin ~on:'.') with
        | _value :: enclosing :: _ -> String.equal enclosing target
        | _ -> false
      in
      Option.some_if (List.exists emitter.origins ~f:defined_in) emitter.emitter_name)

(** Resolves every identifier of [structure] over {!Lexical_scope}'s model (gh-ocannl-1079).

    Both the module aliases and the literal [let]s used to be looked up file-wide, and each way was
    silently wrong. An alias set collected over the whole file credits [G.read] in a scope where [G]
    is some other module, and a later [module CR = ...] rebinds an earlier [open CR]. A literal
    table keyed by name gave a pin site whichever binding of the name was unique in the file -- a
    labelled PARAMETER read as the Metal leg's [let body_begin] several hundred lines below -- and
    dropped every binding of a name bound twice, equal values included (gh-ocannl-1079, from
    staging#855): the inventory recorded the wrong text for one leg and none for another, and
    nothing failed. Here each identifier reaches the binding OCaml gives it.

    What this does NOT reach is an alias in another FILE: this scan decides one source at a time, so
    a wrapper module that some other file defines around the reader would leave its callers
    unrecognised. Nothing in the tree does that today -- the one wrapper, [Test_utils.Generated], IS
    the target -- and a shared helper of that shape would have to be added to the seeds here. *)
let scope_of ~emitters structure =
  let qualifiers = Hashtbl.Poly.create () and values = Hashtbl.Poly.create () in
  let resolver =
    object (self)
      inherit [value_denotes, module_denotes] Lexical_scope.scoped as super
      method local = Bound
      method! shadowed = Some Opaque

      method module_path env path =
        match (path, Map.find env.modules (Lexical_scope.head path)) with
        | _, Some Opaque -> Some Opaque
        | Longident.Lident _, Some (Named target) -> Some (Named target)
        | (Lident last | Ldot (_, last)), _ -> Some (Named last)
        | Lapply _, _ -> Some Opaque

      method! module_of env module_expr =
        match module_expr.pmod_desc with
        | Pmod_apply (functor_, _) | Pmod_apply_unit functor_ -> self#module_of env functor_
        | _ -> super#module_of env module_expr

      method! let_denotes vb =
        match (vb.pvb_pat.ppat_desc, string_literal vb.pvb_expr) with
        | Ppat_var _, Some text -> Literal_binding text
        | _ -> (
            match vb.pvb_expr.pexp_desc with
            | Pexp_function _ -> Function_binding (span vb.pvb_expr.pexp_loc)
            | _ -> Bound)

      method! opened ~top:_ ~include_:_ env denoted =
        match denoted with
        | Some (Named target) -> (
            match hidden_names ~emitters target with
            | [] -> env
            | names ->
                Lexical_scope.push_frame env
                  (Map.of_alist_reduce
                     (module String)
                     (List.map names ~f:(fun name -> (name, Hidden { opened = target; name })))
                     ~f:(fun _ later -> later)))
        | Some Opaque | None -> env

      method! expression env e =
        (match e.pexp_desc with
        | Pexp_ident { txt = Lident name; _ } ->
            Option.iter (Lexical_scope.lookup env name) ~f:(fun denotes ->
                Hashtbl.set values ~key:(span e.pexp_loc) ~data:denotes)
        | Pexp_ident { txt = Ldot (qualifier, _); _ } ->
            Option.iter (self#module_path env qualifier) ~f:(fun denotes ->
                Hashtbl.set qualifiers ~key:(span e.pexp_loc) ~data:denotes)
        | _ -> ());
        super#expression env e
    end
  in
  ignore
    (resolver#structure { frames = []; modules = Map.empty (module String) } structure : structure);
  { qualifiers; values }

(** Whether [e] calls [name] on the module named [target], through whatever alias is in scope where
    it is spelled. An unqualified [read] answers false whatever it reaches -- which is what keeps a
    test's own local [read] from being taken for the artifact reader. *)
let calls scope e ~target ~name =
  match longident_of e with
  | Some path when path_ends path ~name && Option.is_some (qualifier_of path) -> (
      match Hashtbl.find scope.qualifiers (span e.pexp_loc) with
      | Some (Named module_name) -> String.equal module_name target
      | Some Opaque | None -> false)
  | _ -> false

(** The reader that hands a test its generated source: [read] on {!Test_utils.Generated} under any
    prefix or alias. *)
let mentions_generated_read scope expr =
  let found = ref false in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! expression e =
        if calls scope e ~target:"Generated" ~name:"read" then found := true;
        super#expression e
    end
  in
  iterator#expression expr;
  !found

(** The emitter a path names, if any: an emitter's name behind a qualifier, or -- for [aliases] -- a
    local name this file bound to one. *)
let emitter_of_path ~emitters ~aliases path =
  match path with
  | [ name ] -> List.Assoc.find aliases name ~equal:String.equal
  | _ ->
      if Option.is_some (qualifier_of path) then
        List.find emitters ~f:(fun e -> path_ends path ~name:e.emitter_name)
      else None

let renders_generated_text ~emitters ~aliases expr =
  let found = ref false in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! expression e =
        (match longident_of e with
        | Some path when Option.is_some (emitter_of_path ~emitters ~aliases path) -> found := true
        | _ -> ());
        super#expression e
    end
  in
  iterator#expression expr;
  !found

type parameter_destination = {
  destination : destination;
  default : expression option;
  position : int;
  optional : bool;
}
(** How each named parameter is addressed at a call site, with its optional default and the number
    of preceding positional parameters. An omitted default is selected only once a later positional
    argument is supplied. Labels consume no positional index. *)

let param_destinations params =
  List.folding_map params ~init:0 ~f:(fun position (label, name, default) ->
      match label with
      | Asttypes.Nolabel ->
          ( position + 1,
            (name, { destination = At_position position; default; position; optional = false }) )
      | Asttypes.Labelled label ->
          (position, (name, { destination = At_label label; default; position; optional = false }))
      | Asttypes.Optional label ->
          (position, (name, { destination = At_label label; default; position; optional = true })))

(** The unlabelled arguments of an application, in order. *)
let positional args =
  List.filter_map args ~f:(fun (label, arg) ->
      match label with Asttypes.Nolabel -> Some arg | _ -> None)

(** The argument a destination names at one call site. *)
let argument_at ~destination args =
  match destination with
  | At_label label ->
      List.find_map args ~f:(fun (argument_label, argument) ->
          match argument_label with
          | (Asttypes.Labelled l | Asttypes.Optional l) when String.equal l label -> Some argument
          | _ -> None)
  | At_position position -> List.nth (positional args) position

(** Resolve a predicate argument, including static optional forwarding and default erasure. *)
let predicate_argument_at { destination; default; position; optional } args =
  let selected_default () =
    if optional && List.length (positional args) > position then default else None
  in
  let forwarded =
    match destination with
    | At_position _ -> None
    | At_label label ->
        List.find_map args ~f:(function
          | Asttypes.Optional name, expr when String.equal name label -> Some expr
          | _ -> None)
  in
  match forwarded with
  | Some { pexp_desc = Pexp_construct ({ txt = Lident "None"; _ }, None); _ } -> selected_default ()
  | Some { pexp_desc = Pexp_construct ({ txt = Lident "Some"; _ }, Some e); _ } -> Some e
  | Some forwarded -> Some forwarded
  | None -> (
      match argument_at ~destination args with
      | Some argument -> Some argument
      | None -> selected_default ())

let parameter_supplied parameter args =
  Option.is_some (argument_at ~destination:parameter.destination args)
  || (parameter.optional && List.length (positional args) > parameter.position)

let dynamic_optional parameter args =
  match parameter.destination with
  | At_position _ -> None
  | At_label label ->
      List.find_map args ~f:(function
        | Asttypes.Optional name, e when String.equal name label -> (
            match e.pexp_desc with
            | Pexp_construct ({ txt = Lident "None"; _ }, None)
            | Pexp_construct ({ txt = Lident "Some"; _ }, Some _) ->
                None
            | _ -> Some e)
        | _ -> None)

(** The names an emitter call deposits generated text INTO: the arguments at an emitter's buffer
    labels.

    [Canonical_render.emit] writes into its [~buf] rather than returning a document, and a caller
    can split the write from the read across bindings -- [let () = CR.emit ~buf policy llc] and
    [let source = Buffer.contents buf] later. Neither binding carries taint on its own: the first
    binds no name, the second calls no emitter. So the DESTINATION is seeded directly, and [buf] is
    generated source for the rest of the file, exactly as a returned document would be
    (gh-ocannl-748, from Codex round 5 on gh-ocannl-712).

    An emitter whose buffer argument is unlabelled takes every positional argument of the call as a
    destination. Nothing in the tree has that shape; over-taint costs an inventory line, and the
    alternative -- matching by argument position -- would need the position to survive optional
    arguments the call site omits. *)
let buffer_destinations ~emitters ~aliases structure =
  let names = ref [] in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! expression e =
        (match e.pexp_desc with
        | Pexp_apply (callee, args) -> (
            match Option.bind (longident_of callee) ~f:(emitter_of_path ~emitters ~aliases) with
            | Some emitter ->
                List.iter emitter.destinations ~f:(fun destination ->
                    Option.iter (argument_at ~destination args) ~f:(fun argument ->
                        names := idents_in argument @ !names))
            | None -> ())
        | _ -> ());
        super#expression e
    end
  in
  iterator#structure structure;
  !names

let reads_artifact scope e =
  List.exists direct_artifact_names ~f:(fun name ->
      calls scope e ~target:direct_artifact_module ~name)

let reads_artifacts_directly scope expr =
  let found = ref false in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! expression e =
        if reads_artifact scope e then found := true;
        super#expression e
    end
  in
  iterator#expression expr;
  !found

(** The labelled arguments that name a fragment of text to look for. [~substring] and [~contains]
    are the assertion spellings; [~pattern] is [String.substr_index]/[substr_index_all], which the
    tests that COUNT occurrences of a fragment use, and which pins text exactly as an assertion
    does. *)
let text_test_labels = [ "substring"; "contains"; "pattern" ]

type text_test = {
  text : expression;  (** The fragment argument. *)
  tested : expression option;  (** The haystack, when the call takes one positionally. *)
  inherent : bool;
      (** Whether the call is generated-source-testing by construction: {!Generated.assert_emits}
          and {!Generated.assert_omits} name the routine rather than passing the source, and their
          remaining positional argument is the claim, not the haystack. *)
}

let text_test scope expr =
  match expr.pexp_desc with
  | Pexp_apply (callee, args) -> (
      match
        List.find_map args ~f:(fun (label, arg) ->
            match label with
            | Asttypes.Labelled l when List.exists text_test_labels ~f:(String.equal l) -> Some arg
            | _ -> None)
      with
      | None -> None
      | Some text ->
          let inherent =
            List.exists [ "assert_emits"; "assert_omits" ] ~f:(fun name ->
                calls scope callee ~target:"Generated" ~name)
          in
          Some { text; tested = (if inherent then None else List.hd (positional args)); inherent })
  | _ -> None

(** The name a simple [let] binds, or [None] for a pattern this scan does not follow. Used for the
    parameter peel, where a position has to be exact. *)
let rec bound_name pattern =
  match pattern.ppat_desc with
  | Ppat_var { txt; _ } -> Some txt
  | Ppat_constraint (inner, _) -> bound_name inner
  | _ -> None

(** Every variable a pattern binds, tuple and record patterns included. Used for taint, where
    [let values, src = run () in] must taint [src] -- a source reached through a tuple is a source.
*)
let pattern_names pattern =
  let found = ref [] in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! pattern p =
        (match p.ppat_desc with Ppat_var { txt; _ } -> found := txt :: !found | _ -> ());
        super#pattern p
    end
  in
  iterator#pattern pattern;
  !found

(** A deliberately narrow escape hatch for text classifiers over the compiler invocation rather than
    over generated code. The attribute belongs to one value binding: suppressing a whole file would
    let an unrelated real pin disappear with it (gh-ocannl-865). *)
let compiler_plan_attribute = "ocannl.codegen_text.compiler_plan"

let classifies_compiler_plan (vb : value_binding) =
  List.exists vb.pvb_attributes ~f:(fun attribute ->
      String.equal attribute.attr_name.txt compiler_plan_attribute)

let case_parameter = "\000case"

(** Parameters with their labels, stopping at a pattern this scan cannot name. Labels do not consume
    a positional argument; optional defaults do not change how a caller addresses one. *)
let peel_params expr =
  let rec go acc expr =
    match expr.pexp_desc with
    | Pexp_function (params, _, body) -> (
        let rec take = function
          | [] -> []
          | param :: rest -> (
              match param.pparam_desc with
              | Pparam_val (label, default, pat) -> (
                  match bound_name pat with
                  | Some p -> (label, p, default) :: take rest
                  | None -> [])
              | _ -> [])
        in
        let taken = take params in
        let acc = acc @ taken in
        let complete = List.length taken = List.length params in
        match body with
        | Pfunction_body inner when complete -> go acc inner
        | Pfunction_body inner -> (acc, inner)
        | Pfunction_cases (cases, _, _) ->
            let loc = expr.pexp_loc in
            let scrutinee =
              Ppxlib.Ast_builder.Default.pexp_ident ~loc { txt = Lident case_parameter; loc }
            in
            ( acc @ [ (Asttypes.Nolabel, case_parameter, None) ],
              Ppxlib.Ast_builder.Default.pexp_match ~loc scrutinee cases ))
    | _ -> (acc, expr)
  in
  go [] expr

type binding = {
  binding_id : int * int;
  names : string list;
  params : (Asttypes.arg_label * string * expression option) list;
  body : expression;
}

(** Every [let]-bound value in [expr] or [structure], nested ones included. *)
let bindings_of ?(include_compiler_plan = false) collect =
  let found = ref [] in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! value_binding vb =
        if include_compiler_plan || not (classifies_compiler_plan vb) then (
          let params, body = peel_params vb.pvb_expr in
          found :=
            {
              binding_id = span vb.pvb_expr.pexp_loc;
              names = pattern_names vb.pvb_pat;
              params;
              body;
            }
            :: !found;
          super#value_binding vb)
    end
  in
  collect iterator;
  List.rev !found

let bindings_in_structure structure = bindings_of (fun it -> it#structure structure)
let bindings_in_expression expr = bindings_of (fun it -> it#expression expr)

(** Single-name bindings for emitter aliases, including compiler-plan classifiers: the annotation
    suppresses their pins, but an emitter alias they export can still be used outside the
    classifier. *)
let emitter_bindings structure =
  bindings_of ~include_compiler_plan:true (fun it -> it#structure structure)
  |> List.filter_map ~f:(function
    | { names = [ name ]; params; body; _ } -> Some (name, params, body)
    | _ -> None)

(** Local names that reach an emitter, to a fixed point, with where a call to each of them leaves
    its generated text.

    A qualifier is what attributes a call, and putting the emitter behind a local name takes the
    qualifier away at every later call site. Two shapes, both of which left a test's fragment out of
    the inventory while the FILE stayed listed -- the invisible-omission shape, since a member that
    itemises nothing looks exactly like a member with nothing to itemise (Codex rounds 1 and 2 on
    lukstafi/ocannl-staging#487):

    - an {b alias}, [let write = CR.emit] and [let w = write] after it, which is the emitter under
      another name and carries its destinations unchanged;
    - a {b wrapper}, [let write ~buf p llc = CR.emit ~buf p llc], whose own parameter is what the
      caller's buffer arrives through -- so the wrapper's destinations are the positions and labels
      of ITS parameters that reach a destination of the emitter it calls.

    A wrapper's parameter counts when the destination argument mentions it directly. Reaching one
    through a local binding inside the wrapper is not followed, and does not pass silently: the
    caller's buffer is then read untainted, which {!classify_source} reports as an itemisation this
    scan cannot complete. *)
let emitter_aliases ~emitters structure =
  let candidates = emitter_bindings structure in
  let aliases = ref [] in
  let known path = emitter_of_path ~emitters ~aliases:!aliases path in
  let wrapper_destinations ~params body =
    let found = ref [] in
    let iterator =
      object
        inherit Ast_traverse.iter as super

        method! expression e =
          (match e.pexp_desc with
          | Pexp_apply (callee, args) -> (
              match Option.bind (longident_of callee) ~f:known with
              | Some emitter ->
                  List.iter emitter.destinations ~f:(fun destination ->
                      Option.iter (argument_at ~destination args) ~f:(fun argument ->
                          let carried = idents_in argument in
                          List.iter (param_destinations params)
                            ~f:(fun (name, { destination; _ }) ->
                              if List.mem carried name ~equal:String.equal then
                                found := destination :: !found)))
              | None -> ())
          | _ -> ());
          super#expression e
      end
    in
    iterator#expression body;
    List.dedup_and_sort !found ~compare:Poly.compare
  in
  let changed = ref true in
  while !changed do
    changed := false;
    List.iter candidates ~f:(fun (name, params, body) ->
        if not (List.Assoc.mem !aliases name ~equal:String.equal) then
          let found =
            match params with
            | [] ->
                Option.bind (longident_of body) ~f:(fun path ->
                    Option.map (known path) ~f:(fun emitter -> { emitter with emitter_name = name }))
            | _ -> (
                match wrapper_destinations ~params body with
                | [] -> None
                | destinations -> Some { emitter_name = name; origins = []; destinations })
          in
          Option.iter found ~f:(fun emitter ->
              aliases := (name, emitter) :: !aliases;
              changed := true))
  done;
  !aliases

type uncertainty = Known | Unresolved | Replaced

let join_uncertainty a b =
  match (a, b) with
  | Unresolved, _ | _, Unresolved -> Unresolved
  | Replaced, _ | _, Replaced -> Replaced
  | Known, Known -> Known

let parameter_symbol (first, last) name = Printf.sprintf "\000%d:%d:%s" first last name

type provenance = {
  generated : bool;
  parameter_dependent : bool;
      (** The returned data depends on an argument, rather than an internal generated-text read. *)
  parameters : Set.M(String).t;
  buffer : bool;
  uncertainty : uncertainty;
  functions : (int * int) list;
}
(** One provenance rule for parameters, bindings and expression uses. A parameter is symbolic until
    a validated call seeds it; a let inherits its right-hand side's provenance in the lexical scope
    where that side is evaluated. Thus [let src = strip src] preserves the original parameter, while
    [let src = backend_name] and lambda patterns shadow it. Readers, emitter results and emitter
    buffer destinations seed generated text. Unknown buffer reads retain their own uncertainty. The
    fixed point only resolves forward dependencies; it never unions bindings that happen to share a
    spelling. *)

let no_provenance =
  {
    generated = false;
    parameter_dependent = false;
    parameters = Set.empty (module String);
    buffer = false;
    uncertainty = Known;
    functions = [];
  }

let join_provenance a b =
  {
    generated = a.generated || b.generated;
    parameter_dependent =
      (a.parameter_dependent || b.parameter_dependent)
      && not
           ((a.generated && not a.parameter_dependent) || (b.generated && not b.parameter_dependent));
    parameters = Set.union a.parameters b.parameters;
    buffer = a.buffer || b.buffer;
    uncertainty = join_uncertainty a.uncertainty b.uncertainty;
    functions = List.dedup_and_sort (a.functions @ b.functions) ~compare:Poly.compare;
  }

let equal_provenance a b =
  Bool.equal a.generated b.generated
  && Bool.equal a.parameter_dependent b.parameter_dependent
  && Set.equal a.parameters b.parameters
  && Bool.equal a.buffer b.buffer
  && Poly.equal a.uncertainty b.uncertainty
  && Poly.equal a.functions b.functions

let argument_provenance of_expr parameter args =
  let result =
    Option.value_map (predicate_argument_at parameter args) ~default:no_provenance ~f:of_expr
  in
  match dynamic_optional parameter args with
  | None -> result
  | Some _ ->
      let fallback =
        if parameter.optional && List.length (positional args) > parameter.position then
          Option.value_map parameter.default ~default:no_provenance ~f:of_expr
        else no_provenance
      in
      let result = join_provenance result fallback in
      if
        result.generated
        || (not (Set.is_empty result.parameters))
        || Poly.equal result.uncertainty Unresolved
      then { result with uncertainty = Unresolved }
      else result

let scoped_provenance scope ~emitters ~aliases ~seeds ?(parameters = [])
    ?(source_parameters = Hashtbl.Poly.create ()) ?(function_parameters = Hashtbl.Poly.create ())
    ?(outer = fun _ -> no_provenance) collect =
  let uses = Hashtbl.Poly.create () in
  let changed = ref false in
  let produces_position_or_bool e =
    match e.pexp_desc with
    | Pexp_apply (callee, _) ->
        List.exists [ "is_substring"; "substr_index"; "substr_index_all" ] ~f:(fun name ->
            calls scope callee ~target:"String" ~name)
    | _ -> false
  in
  let rec of_expr expr =
    let found = ref no_provenance in
    let iterator =
      object
        inherit Ast_traverse.iter as super
        method! attribute _ = ()

        method! expression e =
          if not (produces_position_or_bool e) then (
            (match e.pexp_desc with
            | Pexp_ident { txt = Lident _; _ } ->
                found :=
                  join_provenance !found
                    (Option.value (Hashtbl.find uses (span e.pexp_loc)) ~default:no_provenance)
            | _ -> ());
            if
              calls scope e ~target:"Generated" ~name:"read"
              || reads_artifact scope e
              || Option.is_some
                   (Option.bind (longident_of e) ~f:(emitter_of_path ~emitters ~aliases))
            then found := join_provenance !found { no_provenance with generated = true };
            if calls scope e ~target:"Buffer" ~name:"contents" then
              found := { !found with buffer = true };
            super#expression e)
      end
    in
    iterator#expression expr;
    match expr.pexp_desc with
    | Pexp_function (_, _, body) ->
        let result =
          match body with
          | Pfunction_body body -> of_expr body
          | Pfunction_cases (cases, _, _) ->
              List.fold cases ~init:no_provenance ~f:(fun acc c ->
                  join_provenance acc (of_expr c.pc_rhs))
        in
        (* Buffer contents can depend on preceding writes. Keep used body inputs at this unsupported
           effect boundary instead of pretending the return expression is pure. *)
        let result =
          if result.buffer && not result.generated then
            join_provenance result { !found with uncertainty = Unresolved; functions = [] }
          else result
        in
        { result with functions = [ span expr.pexp_loc ] }
    | Pexp_apply (callee, args) ->
        let callee_value = of_expr callee in
        let direct_source =
          calls scope callee ~target:"Generated" ~name:"read"
          || reads_artifact scope callee
          || Option.is_some
               (Option.bind (longident_of callee) ~f:(emitter_of_path ~emitters ~aliases))
        in
        let found =
          if direct_source then
            {
              !found with
              generated = true;
              parameter_dependent = false;
              parameters = Set.empty (module String);
              uncertainty = Known;
            }
          else if not (List.is_empty callee_value.functions) then
            let formal_parameters =
              List.concat_map callee_value.functions ~f:(fun id ->
                  Option.value_map (Hashtbl.find function_parameters id) ~default:[]
                    ~f:(fun parameters ->
                      List.map parameters ~f:(fun (name, destination) ->
                          (parameter_symbol id name, destination))))
            in
            let depends_on_argument =
              List.exists formal_parameters ~f:(fun (symbol, _) ->
                  Set.mem callee_value.parameters symbol)
            in
            let unsupplied_dependency =
              List.exists formal_parameters ~f:(fun (symbol, destination) ->
                  Set.mem callee_value.parameters symbol
                  && not (parameter_supplied destination args))
            in
            let result =
              {
                callee_value with
                generated =
                  callee_value.generated
                  && ((not callee_value.parameter_dependent) || not depends_on_argument);
                parameter_dependent =
                  callee_value.parameter_dependent
                  && ((not depends_on_argument) || unsupplied_dependency);
                parameters =
                  List.fold formal_parameters ~init:callee_value.parameters
                    ~f:(fun acc (symbol, destination) ->
                      if parameter_supplied destination args then Set.remove acc symbol else acc);
                uncertainty =
                  (if unsupplied_dependency then Unresolved
                   else if depends_on_argument && not callee_value.buffer then Known
                   else callee_value.uncertainty);
                functions = [];
              }
            in
            if callee_value.generated && not callee_value.parameter_dependent then result
            else
              List.fold formal_parameters ~init:result ~f:(fun acc (symbol, destination) ->
                  if Set.mem callee_value.parameters symbol then
                    join_provenance acc (argument_provenance of_expr destination args)
                  else acc)
          else !found
        in
        let functions =
          List.filter callee_value.functions ~f:(fun id ->
              Option.value_map (Hashtbl.find function_parameters id) ~default:false
                ~f:(fun parameters ->
                  List.exists parameters ~f:(fun (_name, parameter) ->
                      not (parameter_supplied parameter args))))
        in
        { found with functions }
    | Pexp_ident _ -> !found
    | Pexp_let (_, _, body)
    | Pexp_sequence (_, body)
    | Pexp_constraint (body, _)
    | Pexp_open (_, body) ->
        of_expr body
    | Pexp_ifthenelse (_, yes, no) ->
        join_provenance (of_expr yes) (Option.value_map no ~default:no_provenance ~f:of_expr)
    | Pexp_match (_, cases) ->
        List.fold cases ~init:no_provenance ~f:(fun acc c -> join_provenance acc (of_expr c.pc_rhs))
    | Pexp_try (body, cases) ->
        List.fold cases ~init:(of_expr body) ~f:(fun acc c ->
            join_provenance acc (of_expr c.pc_rhs))
    | _ -> { !found with functions = [] }
  in
  (* Split only components supplied explicitly by syntax. Opaque aggregates retain possible source
     evidence with uncertainty rather than assigning every component a definite source. *)
  let rec bind_pattern env pattern expression payload =
    let bind_all payload = Lexical_scope.bind_values env (pattern_names pattern) payload in
    match (pattern.ppat_desc, Option.map expression ~f:(fun e -> e.pexp_desc)) with
    | Ppat_constraint (inner, _), _ -> bind_pattern env inner expression payload
    | Ppat_alias (inner, { txt = name; _ }), _ ->
        bind_pattern (Lexical_scope.bind_values env [ name ] payload) inner expression payload
    | Ppat_tuple patterns, Some (Pexp_tuple expressions)
      when List.length patterns = List.length expressions ->
        List.fold2_exn patterns expressions ~init:env ~f:(fun env pattern expression ->
            bind_pattern env pattern (Some expression) (of_expr expression))
    | Ppat_record (patterns, _), Some (Pexp_record (expressions, None)) ->
        List.fold patterns ~init:env ~f:(fun env (label, pattern) ->
            let expression =
              List.find_map expressions ~f:(fun (candidate, expression) ->
                  if Poly.equal label.txt candidate.txt then Some expression else None)
            in
            bind_pattern env pattern expression
              (Option.value_map expression
                 ~default:{ payload with uncertainty = Unresolved }
                 ~f:of_expr))
    | ( Ppat_construct ({ txt = constructor; _ }, Some (_, inner)),
        Some (Pexp_construct ({ txt = actual; _ }, Some expression)) )
      when Poly.equal constructor actual ->
        bind_pattern env inner (Some expression) (of_expr expression)
    | (Ppat_tuple _ | Ppat_record _), _ ->
        bind_all
          (if payload.generated || (not (Set.is_empty payload.parameters)) || payload.buffer then
             { payload with uncertainty = Unresolved }
           else payload)
    | _ -> bind_all payload
  in
  let resolver =
    object (self)
      inherit [provenance, unit] Lexical_scope.scoped as super
      method local = no_provenance
      method module_path _ _ = None
      method! attribute _ attribute = attribute
      method! let_denotes vb = of_expr vb.pvb_expr

      method! bind_group env bindings denotes =
        List.fold2_exn bindings denotes ~init:(super#bind_group env bindings denotes)
          ~f:(fun env vb payload -> bind_pattern env vb.pvb_pat (Some vb.pvb_expr) payload)

      method! bindings env rec_flag bindings =
        let inner = super#bindings env rec_flag bindings in
        List.fold bindings ~init:inner ~f:(fun inner vb ->
            let denotes = of_expr vb.pvb_expr in
            let replaced =
              List.exists (pattern_names vb.pvb_pat) ~f:(fun name ->
                  Option.exists (Lexical_scope.lookup env name) ~f:(fun previous ->
                      (not (Set.is_empty previous.parameters))
                      && Set.is_empty (Set.inter previous.parameters denotes.parameters)))
            in
            bind_pattern inner vb.pvb_pat (Some vb.pvb_expr)
              {
                denotes with
                uncertainty =
                  (if replaced then join_uncertainty denotes.uncertainty Replaced
                   else denotes.uncertainty);
              })

      method! expression env e =
        match e.pexp_desc with
        | Pexp_function (params, _, body) ->
            let env =
              List.fold params ~init:env ~f:(fun env param ->
                  match param.pparam_desc with
                  | Pparam_val (_, default, pat) ->
                      Option.iter default ~f:(fun d -> ignore (self#expression env d : expression));
                      let names = pattern_names pat in
                      let denotes =
                        List.fold names ~init:no_provenance ~f:(fun acc name ->
                            join_provenance acc
                              (Option.value
                                 (Hashtbl.find source_parameters (span e.pexp_loc, name))
                                 ~default:{ no_provenance with uncertainty = Unresolved }))
                      in
                      Lexical_scope.bind_values env names
                        {
                          denotes with
                          parameter_dependent = true;
                          parameters =
                            Set.of_list
                              (module String)
                              (List.map names ~f:(parameter_symbol (span e.pexp_loc)));
                        }
                  | _ -> env)
            in
            (match body with
            | Pfunction_body body -> ignore (self#expression env body : expression)
            | Pfunction_cases (cases, _, _) ->
                let payload =
                  Option.value
                    (Hashtbl.find source_parameters (span e.pexp_loc, case_parameter))
                    ~default:{ no_provenance with uncertainty = Unresolved }
                in
                let payload =
                  {
                    payload with
                    parameter_dependent = true;
                    parameters =
                      Set.singleton
                        (module String)
                        (parameter_symbol (span e.pexp_loc) case_parameter);
                  }
                in
                List.iter cases ~f:(fun c ->
                    let inner = bind_pattern env c.pc_lhs None payload in
                    Option.iter c.pc_guard ~f:(fun guard ->
                        ignore (self#expression inner guard : expression));
                    ignore (self#expression inner c.pc_rhs : expression)));
            e
        | Pexp_match (scrutinee, cases) | Pexp_try (scrutinee, cases) ->
            ignore (self#expression env scrutinee : expression);
            let payload =
              match e.pexp_desc with Pexp_try _ -> no_provenance | _ -> of_expr scrutinee
            in
            List.iter cases ~f:(fun case ->
                let payload =
                  match case.pc_lhs.ppat_desc with
                  | Ppat_exception _ -> no_provenance
                  | _ -> payload
                in
                let expression =
                  match (e.pexp_desc, case.pc_lhs.ppat_desc) with
                  | Pexp_try _, _ | _, Ppat_exception _ -> None
                  | _ -> Some scrutinee
                in
                let inner = bind_pattern env case.pc_lhs expression payload in
                Option.iter case.pc_guard ~f:(fun guard ->
                    ignore (self#expression inner guard : expression));
                ignore (self#expression inner case.pc_rhs : expression));
            e
        | Pexp_ident { txt = Lident name; _ } ->
            let denotes = Option.value (Lexical_scope.lookup env name) ~default:(outer e) in
            let denotes =
              {
                denotes with
                generated = denotes.generated || List.mem seeds name ~equal:String.equal;
              }
            in
            let previous =
              Option.value (Hashtbl.find uses (span e.pexp_loc)) ~default:no_provenance
            in
            if not (equal_provenance previous denotes) then changed := true;
            Hashtbl.set uses ~key:(span e.pexp_loc) ~data:denotes;
            e
        | _ -> super#expression env e
    end
  in
  let env =
    List.fold parameters
      ~init:{ Lexical_scope.frames = []; modules = Map.empty (module String) }
      ~f:(fun env name ->
        Lexical_scope.bind_values env [ name ]
          {
            no_provenance with
            parameter_dependent = true;
            parameters = Set.singleton (module String) name;
          })
  in
  changed := true;
  while !changed do
    changed := false;
    collect resolver env
  done;
  of_expr

type predicate = {
  pred_binding : int * int;
  text_at : parameter_destination option;
      (** The caller-supplied fragment, with its destination and optional default. *)
  body_text : expression option;
      (** A hard-coded or composite fragment in the predicate body, recorded only after the call's
          own source is validated. Composite context is retained alongside caller components. *)
  source_at : parameter_destination option;
      (** Label or positional index of the parameter that carries the generated source, when the
          predicate takes it rather than closing over it. Checked at each call site: a predicate is
          only pinning where the haystack it is handed really is generated source. *)
}

(** Predicates whose literal argument IS a pinned fragment: the
    [let has s = String.is_substring src ~substring:s] idiom and its variants -- one that takes the
    source as a parameter ([let src_has src s = ...]), one that reaches it through a local binding,
    one that counts occurrences with [~pattern].

    Also returns the ranges handled by predicate calls. Caller markers, hard-coded body fragments
    and composite context all pass the same source check at those calls; recording them directly in
    the body would attribute literals even when the caller supplies ordinary text. *)
let predicates scope ~emitters ~aliases ~seeds ~outer ~function_parameters bindings =
  let consumed = ref [] in
  let uncertain_source = ref false in
  let predicates =
    List.concat_map bindings ~f:(fun { binding_id; names; params; body } ->
        match (names, params) with
        | [ _name ], _ :: _ ->
            let destinations = param_destinations params in
            let destination_of p = List.Assoc.find destinations p ~equal:String.equal in
            let params = List.map params ~f:(fun (_label, name, _default) -> name) in
            let captured_params =
              List.concat_map bindings ~f:(fun enclosing ->
                  let first, last = span body.pexp_loc in
                  let outer_first, outer_last = span enclosing.body.pexp_loc in
                  if
                    outer_first <= first && last <= outer_last
                    && not (outer_first = first && outer_last = last)
                  then List.map enclosing.params ~f:(fun (_label, name, _default) -> name)
                  else [])
            in
            let provenance =
              scoped_provenance scope ~emitters ~aliases ~seeds ~outer ~function_parameters
                ~parameters:(params @ captured_params) (fun resolver env ->
                  ignore (resolver#expression env body : expression))
            in
            let derived_params e =
              Set.inter (provenance e).parameters (Set.of_list (module String) params)
            in
            let generated_locally e =
              (provenance e).generated || (Set.is_empty (derived_params e) && (outer e).generated)
            in
            let derives_from_capture e =
              Set.inter (provenance e).parameters (Set.of_list (module String) captured_params)
            in
            let result = ref [] in
            let consider { text; tested; inherent } =
              let text_params = List.filter params ~f:(fun p -> Set.mem (derived_params text) p) in
              let consider_param text_param =
                (* Keep each text parameter: one predicate can pin several caller-supplied
                   markers. *)
                let source_param =
                  Option.bind tested ~f:(fun tested ->
                      (* A generated read inside the helper is already a source. Its dependence on a
                         parameter used to compile the routine does not make that parameter
                         source. *)
                      if generated_locally tested then None
                      else
                        match Set.to_list (derived_params tested) with
                        | [ p ] when not (List.mem text_params p ~equal:String.equal) -> Some p
                        | _ -> None)
                in
                let record source =
                  consumed :=
                    (text.pexp_loc.loc_start.pos_cnum, text.pexp_loc.loc_end.pos_cnum) :: !consumed;
                  result :=
                    {
                      pred_binding = binding_id;
                      text_at = Option.bind text_param ~f:destination_of;
                      body_text =
                        (match (text_param, text.pexp_desc) with
                        | Some _, Pexp_ident _ -> None
                        | _ -> Some text);
                      source_at = Option.bind source ~f:destination_of;
                    }
                    :: !result
                in
                (* A parameter that IS the haystack makes this a predicate whether or not the caller
                   supplies the fragment. Requiring both left [let has_barrier src = ... ~substring:
                   "__syncthreads()"] unrecognised, so neither the literal nor a partial mark
                   reached the inventory (Codex P2, round 3). *)
                if Option.is_some source_param then record source_param
                else if
                  inherent
                  (* A partially applied substring predicate has no explicit haystack; keep the
                     enclosing source evidence for that existing higher-order idiom. *)
                  || Option.value_map tested ~default:(generated_locally body) ~f:generated_locally
                then record None
                else if
                  Option.value_map tested ~default:false ~f:(fun tested ->
                      not (Set.is_empty (derives_from_capture tested)))
                then uncertain_source := true
                else if
                  Option.value_map tested ~default:false ~f:(fun e ->
                      not (Poly.equal (provenance e).uncertainty Known))
                then uncertain_source := true
              in
              List.iter
                (match text_params with [] -> [ None ] | ps -> List.map ps ~f:Option.some)
                ~f:consider_param
            in
            let iterator =
              object
                inherit Ast_traverse.iter as super

                (* Nested bindings are classified separately by bindings_of. Their parameter tests
                   must not become computed body fragments of the enclosing predicate. *)
                method! value_binding vb =
                  match vb.pvb_expr.pexp_desc with
                  | Pexp_function _ -> ()
                  | _ -> super#value_binding vb

                method! expression e =
                  (match text_test scope e with Some t -> consider t | None -> ());
                  super#expression e
              end
            in
            iterator#expression body;
            List.rev !result
        | _ -> [])
  in
  (predicates, !consumed, !uncertain_source)

type pin = Literal of string | Format of string | Interpolated of string | Computed

(** How a pin argument reads.

    A literal is itself. A [Printf.sprintf] format, and a concatenation with a literal part, are
    fragments with a HOLE in them -- ["(float)(" ^ spelling ^ ")"], ["< (int)(%d.0))) {"] -- and
    both are itemised with the hole shown, because that is the context gh-ocannl-623 was found in
    only by reading a failing kernel. Anything with no literal part at all is [Computed], which
    marks the file's itemisation partial rather than being dropped silently. *)
let rec pin_of_expr scope expr =
  match string_literal expr with
  | Some text -> Literal text
  | None -> (
      (* A fragment named through a binding is still that fragment: [let arrow = " := " in
         String.substr_index statement ~pattern:arrow] pins the IR dump's assignment arrow as surely
         as spelling it at the call site would; without it the site reported only that it pins
         something the scan cannot name. The binding is the one the name reaches where it is spelled
         ({!scope_of}), so a parameter or a shadowing [let] of the name is not some literal
         elsewhere in the file. *)
      match (expr.pexp_desc, Hashtbl.find scope.values (span expr.pexp_loc)) with
      | Pexp_ident _, Some (Literal_binding text) -> Literal text
      | _ -> (
          match expr.pexp_desc with
          | Pexp_apply (callee, args) -> (
              match longident_of callee with
              | Some path when path_ends path ~name:"sprintf" || path_ends path ~name:"ksprintf"
                -> (
                  match List.filter_map (positional args) ~f:string_literal with
                  | fmt :: _ -> Format fmt
                  | [] -> Computed)
              | Some path when path_ends path ~name:"^" ->
                  let parts =
                    List.map (positional args) ~f:(fun a ->
                        match pin_of_expr scope a with
                        | Literal text -> Printf.sprintf "%S" text
                        | Interpolated text -> text
                        | Format _ | Computed -> "...")
                  in
                  let rendered = String.concat ~sep:" ^ " parts in
                  if String.is_substring rendered ~substring:"\"" then Interpolated rendered
                  else Computed
              | _ -> Computed)
          | _ -> Computed))

type site = {
  site_path : string;
  pins : string list;  (** Sorted, deduplicated, each rendered ready for the inventory. *)
  partial : bool;  (** Some pinned fragment could not be named at its call site. *)
  direct : bool;  (** Reads [build_files/] without going through {!Test_utils.Generated}. *)
  rendered : bool;
      (** Renders generated text in memory, through an emitter or a dump printer, rather than
          reading an artifact. Such a test has no [build_files/] output to inspect. *)
}

let render_pin = function
  | Literal text -> Some (Printf.sprintf "%S" text)
  | Format fmt -> Some (Printf.sprintf "sprintf %S" fmt)
  | Interpolated rendered -> Some rendered
  | Computed -> None

(** Spellings this scan refuses rather than approximates, each reported as a failure by
    [codegen_text_inventory].

    Every route to generated text is attributed by the QUALIFIER at the call site --
    [Generated.read], [Utils.build_file], [Syntax.compile_proc]. An [open] removes the qualifier,
    and then the call is indistinguishable from a local function of the same name, so the file drops
    out of the census entirely: the silent direction, and the one every miss on gh-ocannl-712 took.

    Refusing is a checkable fact, and it costs nothing: the qualified spelling is what every test in
    the tree already uses (gh-ocannl-748, from Codex round 5). What is refused is decided over the
    lexical scope ({!scope_of}): an unqualified name refuses where the innermost binding it reaches
    is one an [open] or [include] of an attributed module brought in -- the readers' module, an
    emitter's own, through whatever alias or functor application names it. So an open governs its
    own scope (a structure-level one the items after it, [let open M in] its body, a nested
    structure's dies with it), and a name the file binds is the file's own wherever that binding is
    in scope: [open Ir.Low_level] followed by [let to_doc x = local_render x] and then
    [to_doc value] is valid code calling the local function, where a [to_doc] bound only inside some
    other function leaves the opened one in reach (Codex rounds 3 and 6 on
    lukstafi/ocannl-staging#487, then gh-ocannl-1079).

    Raises if [contents] does not parse. *)
let rejections ~emitters ~path ~contents =
  let scope = scope_of ~emitters (structure_of contents) in
  Hashtbl.data scope.values
  |> List.filter_map ~f:(function
    | Hidden { opened; name } -> Some (opened, name)
    | Literal_binding _ | Function_binding _ | Bound -> None)
  |> List.dedup_and_sort ~compare:Poly.compare
  |> List.map ~f:(fun (opened, name) ->
      Printf.sprintf
        "%s opens %s and then uses %s unqualified, which this scan attributes by its qualifier -- \
         so the call is invisible to it and the file can drop out of the inventory. Write %s.%s \
         (or an alias of it) instead."
        path opened name opened name)

(** [classify_source ~emitters ~path ~contents] is [Some] when the file reads generated source at
    all.

    Raises if [contents] does not parse. *)
let classify_source ~emitters ~path ~contents =
  let structure = structure_of contents in
  (* Every qualifier and every literal-bound name resolved where it is spelled: [Generated] and
     [Utils] for the artifact readers, and [Buffer] for the backstop below -- [module B = Buffer]
     then [B.contents buf] reads a buffer as surely as the bare spelling does. *)
  let scope = scope_of ~emitters structure in
  (* The bindings come first because the emitter aliases do: an emitter bound to a local name is
     called without a qualifier afterwards, and every rule below -- membership, taint, the buffer
     destinations, the pin walk -- has to recognise the same set of calls. Rules that know different
     routes are how a file stayed listed while the fragment it pins went missing. *)
  let bindings = bindings_in_structure structure in
  let aliases = emitter_aliases ~emitters structure in
  let reads_generated = ref false in
  let reads_direct = ref false in
  let renders = ref false in
  let scan_reads =
    object
      inherit Ast_traverse.iter as super

      method! expression e =
        (if
           List.exists [ "read"; "assert_emits"; "assert_omits" ] ~f:(fun name ->
               calls scope e ~target:"Generated" ~name)
         then reads_generated := true
         else if reads_artifact scope e then reads_direct := true
         else
           match longident_of e with
           | Some p when Option.is_some (emitter_of_path ~emitters ~aliases p) -> renders := true
           | _ -> ());
        super#expression e
    end
  in
  scan_reads#structure structure;
  if not (!reads_generated || !reads_direct || !renders) then None
  else
    let seeds = buffer_destinations ~emitters ~aliases structure in
    let functions = Hashtbl.Poly.create () in
    let collect_functions =
      object
        inherit Ast_traverse.iter as super
        method! value_binding vb = if not (classifies_compiler_plan vb) then super#value_binding vb

        method! expression e =
          (match e.pexp_desc with
          | Pexp_function _ ->
              let params, _ = peel_params e in
              Hashtbl.set functions ~key:(span e.pexp_loc) ~data:(param_destinations params)
          | _ -> ());
          super#expression e
      end
    in
    collect_functions#structure structure;
    let source_parameters = Hashtbl.Poly.create () in
    let source_provenance () =
      scoped_provenance scope ~emitters ~aliases ~seeds ~source_parameters
        ~function_parameters:functions (fun resolver env ->
          ignore (resolver#structure env structure : structure))
    in
    let provenance = ref (source_provenance ()) in
    let changed = ref true in
    let next_parameters = Hashtbl.Poly.create () in
    let propagate =
      object
        inherit Ast_traverse.iter as super
        method! value_binding vb = if not (classifies_compiler_plan vb) then super#value_binding vb

        method! expression e =
          (match e.pexp_desc with
          | Pexp_apply (callee, args) ->
              List.iter (!provenance callee).functions ~f:(fun id ->
                  Option.iter (Hashtbl.find functions id) ~f:(fun parameters ->
                      List.iter parameters ~f:(fun (name, destination) ->
                          if parameter_supplied destination args then
                            let previous =
                              Option.value
                                (Hashtbl.find next_parameters (id, name))
                                ~default:no_provenance
                            in
                            Hashtbl.set next_parameters ~key:(id, name)
                              ~data:
                                (join_provenance previous
                                   (argument_provenance !provenance destination args)))))
          | _ -> ());
          super#expression e
      end
    in
    while !changed do
      Hashtbl.clear next_parameters;
      propagate#structure structure;
      changed :=
        Hashtbl.length source_parameters <> Hashtbl.length next_parameters
        || not
             (Hashtbl.for_alli next_parameters ~f:(fun ~key ~data ->
                  Option.value_map
                    (Hashtbl.find source_parameters key)
                    ~default:false ~f:(equal_provenance data)));
      if !changed then (
        Hashtbl.clear source_parameters;
        Hashtbl.iteri next_parameters ~f:(fun ~key ~data ->
            Hashtbl.set source_parameters ~key ~data);
        provenance := source_provenance ())
    done;
    let outer = !provenance in
    let predicates, consumed, uncertain_source =
      predicates scope ~emitters ~aliases ~seeds ~outer ~function_parameters:functions bindings
    in
    let predicates_at callee =
      List.filter predicates ~f:(fun p ->
          List.mem (!provenance callee).functions p.pred_binding ~equal:Poly.equal)
    in
    (* Ordinary forwarding wrappers do not expose their caller's marker as a predicate parameter.
       Keep unresolved calls visibly partial, including chains of wrappers and generated
       defaults. *)
    let forwarding = Hashtbl.Poly.create () in
    let function_at callee =
      match Hashtbl.find scope.values (span callee.pexp_loc) with
      | Some (Function_binding binding_id) -> Some binding_id
      | _ -> None
    in
    let changed = ref true in
    while !changed do
      changed := false;
      List.iter bindings ~f:(fun binding ->
          if not (Hashtbl.mem forwarding binding.binding_id) then (
            let forwards = ref false in
            let provenance =
              scoped_provenance scope ~emitters ~aliases ~seeds ~outer
                ~function_parameters:functions (fun resolver env ->
                  ignore (resolver#expression env binding.body : expression))
            in
            let iterator =
              object
                inherit Ast_traverse.iter as super

                method! expression e =
                  (match e.pexp_desc with
                  | Pexp_apply (callee, args) ->
                      if
                        List.exists (predicates_at callee) ~f:(fun p ->
                            Option.exists p.source_at ~f:(fun source ->
                                not
                                  (Option.value_map (predicate_argument_at source args)
                                     ~default:false ~f:(fun e -> (provenance e).generated))))
                        || Option.exists (function_at callee) ~f:(Hashtbl.mem forwarding)
                      then forwards := true
                  | _ -> ());
                  super#expression e
              end
            in
            iterator#expression binding.body;
            if !forwards then (
              Hashtbl.set forwarding ~key:binding.binding_id ~data:();
              changed := true)))
    done;
    let pins = ref [] in
    let is_consumed (e : expression) =
      List.mem consumed (e.pexp_loc.loc_start.pos_cnum, e.pexp_loc.loc_end.pos_cnum)
        ~equal:(fun (a, b) (c, d) -> a = c && b = d)
    in
    let record text = if not (is_consumed text) then pins := pin_of_expr scope text :: !pins in
    (* Generated source in the haystack, by any of the three routes -- a tainted name, an inline
       [Generated.read], an inline emitter render, an inline [build_files/] read. Naming only the
       first two here left an assertion that renders inline ([String.is_substring (render (LL.to_doc
       () llc)) ~substring:"-0.0"]) with its fragment silently dropped: the FILE stayed in the
       census through the membership branches, so nothing looked wrong, while grepping the inventory
       for the moved spelling missed the assertion (Codex P2, round 3). The membership rules and the
       pin rules have to know the same routes. *)
    let mentions_tainted e = (!provenance e).generated in
    let called = Hashtbl.Poly.create () in
    let applied = Hashtbl.Poly.create () in
    let calls =
      object
        inherit Ast_traverse.iter as super
        method! value_binding vb = if not (classifies_compiler_plan vb) then super#value_binding vb

        method! expression e =
          (match e.pexp_desc with
          | Pexp_apply (callee, _) ->
              Hashtbl.set applied ~key:(span callee.pexp_loc) ~data:();
              List.iter (predicates_at callee) ~f:(fun p ->
                  Hashtbl.set called ~key:p.pred_binding ~data:())
          | _ -> ());
          super#expression e
      end
    in
    calls#structure structure;
    (* A helper reached without an explicit call, for example as a callback, has no validated caller
       source. Keep that uncertainty visible rather than silently dropping its body text. *)
    List.iter predicates ~f:(fun predicate ->
        if not (Hashtbl.mem called predicate.pred_binding) then
          if Option.is_some predicate.source_at then pins := Computed :: !pins
          else
            Option.iter predicate.body_text ~f:(fun text -> pins := pin_of_expr scope text :: !pins));
    (* Unresolved buffer flows use the same scoped provenance, including local aliases. *)
    let reads_a_buffer e = (!provenance e).buffer in
    let unattributed = ref false in
    let iterator =
      object
        inherit Ast_traverse.iter as super
        method! value_binding vb = if not (classifies_compiler_plan vb) then super#value_binding vb

        method! expression e =
          if
            Option.is_some (longident_of e)
            && (not (Hashtbl.mem applied (span e.pexp_loc)))
            && ((not (List.is_empty (predicates_at e)))
               || Option.exists (function_at e) ~f:(Hashtbl.mem forwarding))
          then unattributed := true;
          (match text_test scope e with
          | Some { text; tested; inherent } ->
              if inherent || Option.value_map tested ~default:false ~f:mentions_tainted then (
                record text;
                if
                  Option.value_map tested ~default:false ~f:(fun e ->
                      let source = !provenance e in
                      Poly.equal source.uncertainty Unresolved)
                then unattributed := true)
              else if
                Option.value_map tested ~default:false ~f:(fun e ->
                    not (Poly.equal (!provenance e).uncertainty Known))
              then (
                if
                  Option.value_map tested ~default:false ~f:(fun e ->
                      let source = !provenance e in
                      (not source.buffer) && Poly.equal source.uncertainty Unresolved)
                then record text;
                unattributed := true)
              else if Option.value_map tested ~default:false ~f:reads_a_buffer then
                unattributed := true
          | None -> (
              match e.pexp_desc with
              | Pexp_apply (callee, args) ->
                  if
                    List.is_empty (predicates_at callee)
                    && Option.exists (function_at callee) ~f:(Hashtbl.mem forwarding)
                  then unattributed := true;
                  List.iter (predicates_at callee) ~f:(fun predicate ->
                      let at parameter = predicate_argument_at parameter args in
                      let source_value =
                        Option.map predicate.source_at ~f:(fun parameter ->
                            argument_provenance !provenance parameter args)
                      in
                      let source_ok =
                        match predicate.source_at with
                        | None -> true
                        | Some _ ->
                            Option.value_map source_value ~default:false ~f:(fun p -> p.generated)
                      in
                      (* The backstop belongs on this path as much as on a direct test: a helper is
                         how a test reads a buffer one indirection further out, and a guard that
                         fires only for the spelling written first is not one (Codex round 5). *)
                      if
                        (not source_ok)
                        && Option.value_map source_value ~default:false ~f:(fun p -> p.buffer)
                      then unattributed := true;
                      if source_ok then (
                        if
                          Option.value_map source_value ~default:false ~f:(fun p ->
                              Poly.equal p.uncertainty Unresolved)
                        then unattributed := true;
                        Option.iter predicate.body_text ~f:(fun text ->
                            pins := pin_of_expr scope text :: !pins);
                        match Option.bind predicate.text_at ~f:at with
                        | Some text -> record text
                        | None when Option.is_some predicate.text_at -> pins := Computed :: !pins
                        | None -> ())
                      else (
                        (* An unresolved parameter/callback can still name a real fragment. Keep its
                           known text alongside the partial marker, while a known ordinary haystack
                           contributes no fragment. *)
                        if
                          Option.value_map source_value ~default:false ~f:(fun source ->
                              (not source.buffer) && Poly.equal source.uncertainty Unresolved)
                        then (
                          Option.iter predicate.body_text ~f:(fun text ->
                              pins := pin_of_expr scope text :: !pins);
                          Option.iter (Option.bind predicate.text_at ~f:at) ~f:record);
                        pins := Computed :: !pins))
              | _ -> ()));
          super#expression e
      end
    in
    iterator#structure structure;
    let all = !pins in
    Some
      {
        site_path = path;
        pins = List.filter_map all ~f:render_pin |> List.dedup_and_sort ~compare:String.compare;
        partial =
          uncertain_source || !unattributed
          || List.exists all ~f:(function Computed -> true | _ -> false);
        direct = !reads_direct;
        rendered = !renders;
      }
