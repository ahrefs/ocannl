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
        | _ -> Bound

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

(** How each named parameter is addressed at a call site, with its optional default and the number
    of preceding positional parameters. An omitted default is selected only once a later positional
    argument is supplied. Labels consume no positional index. *)
let param_destinations params =
  List.folding_map params ~init:0 ~f:(fun position (label, name, default) ->
      match label with
      | Asttypes.Nolabel -> (position + 1, (name, (At_position position, default, position)))
      | Asttypes.Labelled label | Asttypes.Optional label ->
          (position, (name, (At_label label, default, position))))

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
let bound_name pattern = match pattern.ppat_desc with Ppat_var { txt; _ } -> Some txt | _ -> None

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
        | Pfunction_cases _ -> (acc, expr))
    | _ -> (acc, expr)
  in
  go [] expr

type binding = {
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
          found := { names = pattern_names vb.pvb_pat; params; body } :: !found;
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
    | { names = [ name ]; params; body } -> Some (name, params, body)
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
                            ~f:(fun (name, (destination, _default, _position)) ->
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

(** Names carrying generated source, to a fixed point: seeded by [Generated.read], by a direct
    [build_files/] read and by the destinations of a buffer-writing emitter ([seeds]), and spreading
    along [let] bindings whose right-hand side mentions one.

    Scope is deliberately ignored -- a name is tainted for the whole file. Taint only decides
    WHETHER a test's haystack is generated source, so over-reach costs an extra inventory line and
    under-reach a missed pin. The fragment a pin names is another matter, and is resolved in scope
    ({!scope_of}): a literal read from the wrong binding puts the wrong text in the inventory. *)
let tainted_names scope ~emitters ~aliases ~seeds bindings =
  let tainted = ref (Set.of_list (module String) seeds) in
  let seeded body =
    mentions_generated_read scope body
    || reads_artifacts_directly scope body
    || renders_generated_text ~emitters ~aliases body
  in
  let changed = ref true in
  while !changed do
    changed := false;
    List.iter bindings ~f:(fun { names; params = _; body } ->
        if not (List.for_all names ~f:(Set.mem !tainted)) then
          if seeded body || List.exists (idents_in body) ~f:(fun i -> Set.mem !tainted i) then (
            tainted := List.fold names ~init:!tainted ~f:Set.add;
            changed := true))
  done;
  !tainted

(** Generated-source identifiers resolved inside one helper, at each use rather than by name.
    Parameters start untainted; a later or nested read cannot change an earlier binding's meaning.
*)
let local_generated_source scope ~emitters ~aliases body =
  let uses = Hashtbl.Poly.create () in
  let mentions_source expr =
    let found = ref false in
    let iterator =
      object
        inherit Ast_traverse.iter as super

        method! expression e =
          (match e.pexp_desc with
          | Pexp_ident _ when Hashtbl.mem uses (span e.pexp_loc) -> found := true
          | _ -> ());
          super#expression e
      end
    in
    iterator#expression expr;
    !found
  in
  let generated expr =
    mentions_generated_read scope expr
    || reads_artifacts_directly scope expr
    || renders_generated_text ~emitters ~aliases expr
    || mentions_source expr
  in
  let resolver =
    object
      inherit [bool, unit] Lexical_scope.scoped as super
      method local = false
      method module_path _ _ = None
      method! let_denotes vb = generated vb.pvb_expr

      method! expression env e =
        (match e.pexp_desc with
        | Pexp_ident { txt = Lident name; _ }
          when Option.value (Lexical_scope.lookup env name) ~default:false ->
            Hashtbl.set uses ~key:(span e.pexp_loc) ~data:()
        | _ -> ());
        super#expression env e
    end
  in
  ignore
    (resolver#expression { frames = []; modules = Map.empty (module String) } body : expression);
  generated

type predicate = {
  pred_name : string;
  text_at : (destination * expression option * int) option;
      (** Label or positional index of the pinned-text parameter, when the fragment is one the
          CALLER supplies. [None] where the helper hard-codes the fragment itself
          ([let has_barrier src = String.is_substring src ~substring:"__syncthreads()"]) -- the
          helper is still a predicate, because its parameter is still generated source, and the
          literal in its body is picked up by the pin walk once that parameter joins the tainted
          set. *)
  source_param : string option;
      (** The parameter that carries the generated source, by name, when the predicate takes it
          rather than closing over it. Such a parameter IS generated source inside the predicate's
          body, so a literal the body tests against it -- the ["Main logic"] banner a helper slices
          on -- is a pin like any other, and the name joins the tainted set for the pin walk. *)
  source_at : (destination * expression option * int) option;
      (** Label or positional index of the parameter that carries the generated source, when the
          predicate takes it rather than closing over it. Checked at each call site: a predicate is
          only pinning where the haystack it is handed really is generated source. *)
}

(** Which of [params] a name inside [body] derives from, to a fixed point.

    The haystack a predicate tests is not always a parameter spelled at the test:
    [let has sub s = let body = strip s in String.is_substring body ~substring:sub] reaches it
    through a local binding. Following that is what tells this predicate -- which pins -- apart from
    [let has s = String.is_substring backend_name ~substring:s], which tests the backend's NAME and
    pins nothing. A name deriving from exactly one parameter identifies it; from several, the
    predicate falls back to the closing-over-a-tainted-name route. *)
let params_derived_in ~params body =
  let table = Hashtbl.create (module String) in
  let of_expr e =
    List.fold (idents_in e)
      ~init:(Set.empty (module String))
      ~f:(fun acc name ->
        let acc = if List.exists params ~f:(String.equal name) then Set.add acc name else acc in
        match Hashtbl.find table name with Some s -> Set.union acc s | None -> acc)
  in
  let inner = bindings_in_expression body in
  let changed = ref true in
  while !changed do
    changed := false;
    List.iter inner ~f:(fun { names; params = _; body } ->
        let from = of_expr body in
        if not (Set.is_empty from) then
          List.iter names ~f:(fun name ->
              let previous =
                Option.value (Hashtbl.find table name) ~default:(Set.empty (module String))
              in
              if not (Set.equal previous (Set.union previous from)) then (
                Hashtbl.set table ~key:name ~data:(Set.union previous from);
                changed := true)))
  done;
  of_expr

(** Predicates whose literal argument IS a pinned fragment: the
    [let has s = String.is_substring src ~substring:s] idiom and its variants -- one that takes the
    source as a parameter ([let src_has src s = ...]), one that reaches it through a local binding,
    one that counts occurrences with [~pattern].

    Also returns the source ranges of the text arguments inside those definitions. Those sites test
    a PARAMETER, not a fragment, and reading them as pins would mark every file using the idiom as
    pinning text the scan cannot name. Skipped by range at the pin walk rather than by skipping the
    whole binding, so a literal a predicate's body pins alongside its parameter still counts. *)
let predicates scope ~emitters ~aliases ~tainted bindings =
  let consumed = ref [] in
  let predicates =
    List.concat_map bindings ~f:(fun { names; params; body } ->
        match (names, params) with
        | [ name ], _ :: _ ->
            let destinations = param_destinations params in
            let destination_of p = List.Assoc.find destinations p ~equal:String.equal in
            let rebound = ref (Set.empty (module String)) in
            let patterns =
              object
                inherit Ast_traverse.iter as super

                method! pattern p =
                  rebound := List.fold (pattern_names p) ~init:!rebound ~f:Set.add;
                  super#pattern p
              end
            in
            patterns#expression body;
            let generated_locally = local_generated_source scope ~emitters ~aliases body in
            let params = List.map params ~f:(fun (_label, name, _default) -> name) in
            let derived_params = params_derived_in ~params body in
            let result = ref [] in
            let consider { text; tested; inherent } =
              let text_params =
                List.filter params ~f:(fun p ->
                    (not (Set.mem !rebound p)) && List.exists (idents_in text) ~f:(String.equal p))
              in
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
                  (* The fragment argument inside a predicate's own definition tests a PARAMETER,
                     not a fragment; reading it as a pin would mark every file using the idiom
                     partial. Consume only the identifier itself: a composite expression keeps its
                     literal context in the pin walk alongside the caller-supplied fragment. *)
                  (match (text_param, text.pexp_desc) with
                  | Some _, Pexp_ident _ ->
                      consumed :=
                        (text.pexp_loc.loc_start.pos_cnum, text.pexp_loc.loc_end.pos_cnum)
                        :: !consumed
                  | _ -> ());
                  result :=
                    {
                      pred_name = name;
                      text_at = Option.bind text_param ~f:destination_of;
                      source_param = source;
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
                  Option.is_some text_param
                  && (inherent
                     (* A partially applied substring predicate has no explicit haystack; keep the
                        enclosing source evidence for that existing higher-order idiom. *)
                     || Option.value_map tested
                          ~default:
                            (List.exists (idents_in body) ~f:(fun name -> Set.mem tainted name))
                          ~f:(fun tested ->
                            generated_locally tested
                            || List.exists (idents_in tested) ~f:(fun name -> Set.mem tainted name))
                     )
                then record None
              in
              List.iter
                (match text_params with [] -> [ None ] | ps -> List.map ps ~f:Option.some)
                ~f:consider_param
            in
            let iterator =
              object
                inherit Ast_traverse.iter as super

                method! expression e =
                  (match text_test scope e with Some t -> consider t | None -> ());
                  super#expression e
              end
            in
            iterator#expression body;
            List.rev !result
        | _ -> [])
  in
  (predicates, !consumed)

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
    | Literal_binding _ | Bound -> None)
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
    let tainted = tainted_names scope ~emitters ~aliases ~seeds bindings in
    let predicates, consumed = predicates scope ~emitters ~aliases ~tainted bindings in
    (* A predicate's source parameter IS generated source, inside that predicate's body. Adding the
       name to the tainted set is how the literals a helper tests against it -- the banner it slices
       on, a second fragment it checks alongside its own argument -- become pins rather than being
       lost with the helper. Names are file-global here, as they are for taint. *)
    let tainted =
      List.fold predicates ~init:tainted ~f:(fun acc p ->
          match p.source_param with Some name -> Set.add acc name | None -> acc)
    in
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
    let mentions_tainted e =
      List.exists (idents_in e) ~f:(fun i -> Set.mem tainted i)
      || mentions_generated_read scope e
      || renders_generated_text ~emitters ~aliases e
      || reads_artifacts_directly scope e
    in
    (* The backstop for every indirection this scan cannot follow. A buffer is where generated text
       lands without a name to carry it, and the ways it can be filled do not end: an emitter behind
       a wrapper whose parameter reaches it through a local binding, a document handed to PPrint's
       own [ToBuffer] renderers, a buffer stored in a record. Each of those leaves a test asserting
       on [Buffer.contents buf] with [buf] untainted -- and, before this, leaves the FILE listed
       with the fragment silently missing, which is the shape every miss on gh-ocannl-712 and
       gh-ocannl-748 took. So a text test whose haystack reads a buffer this scan did not see filled
       marks the itemisation partial: the file is still listed, the fragment is still unnamed, and
       the inventory SAYS so. *)
    let reads_a_buffer_inline e =
      let found = ref false in
      let iterator =
        object
          inherit Ast_traverse.iter as super

          method! expression inner =
            if calls scope inner ~target:"Buffer" ~name:"contents" then found := true;
            super#expression inner
        end
      in
      iterator#expression e;
      !found
    in
    (* And through the bindings the read travels along, to a fixed point, exactly as taint does: a
       test that writes [let source = Buffer.contents buf] and asserts on [source] later is the same
       situation one binding removed, and the backstop has to see it or it is a backstop only for
       the shape someone happened to write first. Nothing here is subtracted from taint -- a name
       the taint walk reached is recorded as a pin before this is consulted. *)
    let buffer_derived =
      let derived = ref (Set.empty (module String)) in
      let changed = ref true in
      while !changed do
        changed := false;
        List.iter bindings ~f:(fun { names; params = _; body } ->
            if not (List.for_all names ~f:(Set.mem !derived)) then
              if
                reads_a_buffer_inline body
                || List.exists (idents_in body) ~f:(fun i -> Set.mem !derived i)
              then (
                derived := List.fold names ~init:!derived ~f:Set.add;
                changed := true))
      done;
      !derived
    in
    let reads_a_buffer e =
      reads_a_buffer_inline e || List.exists (idents_in e) ~f:(fun i -> Set.mem buffer_derived i)
    in
    let unattributed = ref false in
    let iterator =
      object
        inherit Ast_traverse.iter as super
        method! value_binding vb = if not (classifies_compiler_plan vb) then super#value_binding vb

        method! expression e =
          (match text_test scope e with
          | Some { text; tested; inherent } ->
              if inherent || Option.value_map tested ~default:false ~f:mentions_tainted then
                record text
              else if Option.value_map tested ~default:false ~f:reads_a_buffer then
                unattributed := true
          | None -> (
              match e.pexp_desc with
              | Pexp_apply (callee, args) -> (
                  match longident_of callee with
                  | Some [ name ] ->
                      List.iter
                        (List.filter predicates ~f:(fun p -> String.equal p.pred_name name))
                        ~f:(fun predicate ->
                          let at (destination, default, position) =
                            match argument_at ~destination args with
                            | Some argument -> Some argument
                            | None when List.length (positional args) > position -> default
                            | None -> None
                          in
                          let source = Option.bind predicate.source_at ~f:at in
                          let source_ok =
                            match predicate.source_at with
                            | None -> true
                            | Some _ -> Option.value_map source ~default:false ~f:mentions_tainted
                          in
                          (* The backstop belongs on this path as much as on a direct test: a helper
                             is how a test reads a buffer one indirection further out, and a guard
                             that fires only for the spelling written first is not one (Codex round
                             5). *)
                          if
                            (not source_ok)
                            && Option.value_map source ~default:false ~f:reads_a_buffer
                          then unattributed := true;
                          match (source_ok, Option.bind predicate.text_at ~f:at) with
                          | true, Some text -> record text
                          | true, None when Option.is_some predicate.text_at ->
                              pins := Computed :: !pins
                          | _ -> ())
                  | _ -> ())
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
        partial = !unattributed || List.exists all ~f:(function Computed -> true | _ -> false);
        direct = !reads_direct;
        rendered = !renders;
      }
