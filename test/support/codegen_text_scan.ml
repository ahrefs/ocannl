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
    [threadgroup float], [] := ], [/* end */].

    A marker only counts on a line that is not a {!Verdict} claim. Claim labels are prose ABOUT a
    kernel and routinely quote its vocabulary -- ["padded GPU intrinsics fire against the threadgroup
    fragment: true"] is a verdict, not Metal source -- and a golden made of such lines moves when
    the claim is reworded, never when codegen changes.

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

(** The name a module expression ultimately reaches, by its last component.

    Four spellings, all of them the same module as far as any call site is concerned: the path
    itself, a path under a signature constraint ([(Buffer : module type of Buffer)]), and a functor
    APPLICATION, which names its functor -- [Ir.C_syntax.C_syntax (Cfg)] is the module whose values
    the interfaces record under [C_syntax], and is how every backend and codegen test reaches
    [compile_proc]. Reading only the bare path left each of the others attributing nothing, which
    for an [open] means the call under it is neither refused nor recognised (Codex rounds 4 and 5 on
    lukstafi/ocannl-staging#487). A [struct ... end] reaches no name and answers [None]. *)
let rec module_source_name = function
  | { pmod_desc = Pmod_ident { txt; _ }; _ } -> List.last (flatten_longident txt)
  | { pmod_desc = Pmod_constraint (inner, _); _ } -> module_source_name inner
  | { pmod_desc = Pmod_apply (functor_, _); _ } -> module_source_name functor_
  | { pmod_desc = Pmod_apply_unit functor_; _ } -> module_source_name functor_
  | _ -> None

(** The names [target] goes by in this file: itself, plus every module alias that resolves to it.

    Resolved rather than assumed, because [module G = Test_utils.Generated] followed by [G.read "r"]
    is ordinary OCaml, and a scan matching the literal component [Generated] would not merely
    mis-attribute such a file -- it would drop it from the inventory entirely, which is the silent
    direction (Codex P2, round 1). An alias of an alias is an alias, and structure items are visited
    in order, so a name already recognised is available to the binding that borrows it. The
    expression-position spelling ([let module G = ... in]) is matched on ppxlib's tree, where 5.4's
    [Pexp_letmodule] and 5.5's structure-item-inside-an-expression have the one spelling.

    What this does NOT reach is an alias in another FILE: this scan decides one source at a time, so
    a wrapper module that some other file defines around the reader would leave its callers
    unrecognised. Nothing in the tree does that today -- the one wrapper, [Test_utils.Generated], IS
    the target -- and a scan of one file cannot see it; a shared helper of that shape would have to
    be added to the seeds here. *)
let module_aliases ~target structure =
  let aliases = Hash_set.of_list (module String) [ target ] in
  (* Through the same spellings {!module_source_name} reads, so a constrained alias ([module B =
     (Buffer : module type of Buffer)]) is the module it constrains -- otherwise the backstop that
     reads [B.contents] does not fire, and a backstop that fails silently is worse than none (Codex
     round 5). *)
  let resolves module_expr =
    match module_source_name module_expr with
    | Some last -> Hash_set.mem aliases last
    | None -> false
  in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! structure_item item =
        (match item.pstr_desc with
        | Pstr_module { pmb_name = { txt = Some alias; _ }; pmb_expr; _ } when resolves pmb_expr ->
            Hash_set.add aliases alias
        | _ -> ());
        super#structure_item item

      method! expression expr =
        (match expr.pexp_desc with
        | Pexp_letmodule ({ txt = Some alias; _ }, module_expr, _) when resolves module_expr ->
            Hash_set.add aliases alias
        | _ -> ());
        super#expression expr
    end
  in
  iterator#structure structure;
  aliases

(** Whether the path calls [name] on a module [aliases] recognises. *)
let calls ~aliases path ~name =
  path_ends path ~name
  && Option.value_map (qualifier_of path) ~default:false ~f:(Hash_set.mem aliases)

(** The reader that hands a test its generated source: [read] on {!Test_utils.Generated} under any
    prefix or alias. *)
let mentions_generated_read ~generated expr =
  let found = ref false in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! expression e =
        (match longident_of e with
        | Some path when calls ~aliases:generated path ~name:"read" -> found := true
        | _ -> ());
        super#expression e
    end
  in
  iterator#expression expr;
  !found

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

(** Each parameter with its position among the parameters that carry no label, which is how a call
    site addresses it. A labelled parameter keeps its label and takes no position. *)
let positional_params params =
  List.folding_map params ~init:0 ~f:(fun position (label, name) ->
      match label with
      | Asttypes.Nolabel -> (position + 1, (position, (label, name)))
      | _ -> (position, (position, (label, name))))

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

let reads_artifacts_directly ~utils expr =
  let found = ref false in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! expression e =
        (match longident_of e with
        | Some path
          when List.exists direct_artifact_names ~f:(fun n -> calls ~aliases:utils path ~name:n) ->
            found := true
        | _ -> ());
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

let text_test ~generated expr =
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
            match longident_of callee with
            | Some path ->
                List.exists [ "assert_emits"; "assert_omits" ] ~f:(fun name ->
                    calls ~aliases:generated path ~name)
            | None -> false
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

(* Positional parameters, in order, stopping at the first one this scan does not follow: a labelled
   or optional parameter, or a pattern that is not a plain name. Stopping keeps the positions honest
   -- a call site is matched by argument POSITION, and a parameter skipped rather than stopped at
   would shift every position after it. *)
let peel_params expr =
  let rec go acc expr =
    match expr.pexp_desc with
    | Pexp_function (params, _, body) -> (
        let rec take = function
          | [] -> []
          | param :: rest -> (
              match param.pparam_desc with
              | Pparam_val (Asttypes.Nolabel, None, pat) -> (
                  match bound_name pat with Some p -> p :: take rest | None -> [])
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

type binding = { names : string list; params : string list; body : expression }

(** Every [let]-bound value in [expr] or [structure], nested ones included. *)
let bindings_of collect =
  let found = ref [] in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! value_binding vb =
        if not (classifies_compiler_plan vb) then (
          let params, body = peel_params vb.pvb_expr in
          found := { names = pattern_names vb.pvb_pat; params; body } :: !found;
          super#value_binding vb)
    end
  in
  collect iterator;
  List.rev !found

let bindings_in_structure structure = bindings_of (fun it -> it#structure structure)
let bindings_in_expression expr = bindings_of (fun it -> it#expression expr)

(** Every [let]-bound value in the file with its parameters AND their labels, in order, stopping at
    the first parameter this scan does not follow.

    {!bindings_of} drops the labels, because the pin walk matches a predicate's arguments by
    position. An emitter's destination is matched by label as readily as by position, so the wrapper
    analysis below needs them kept. *)
let labelled_bindings structure =
  let found = ref [] in
  let peel expr =
    let rec go acc expr =
      match expr.pexp_desc with
      | Pexp_function (params, _, body) -> (
          let rec take = function
            | [] -> []
            | param :: rest -> (
                match param.pparam_desc with
                | Pparam_val (label, None, pat) -> (
                    match bound_name pat with Some p -> (label, p) :: take rest | None -> [])
                | _ -> [])
          in
          let taken = take params in
          let acc = acc @ taken in
          match body with
          | Pfunction_body inner when List.length taken = List.length params -> go acc inner
          | Pfunction_body inner -> (acc, inner)
          | Pfunction_cases _ -> (acc, expr))
      | _ -> (acc, expr)
    in
    go [] expr
  in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! value_binding vb =
        (match bound_name vb.pvb_pat with
        | Some name ->
            let params, body = peel vb.pvb_expr in
            found := (name, params, body) :: !found
        | None -> ());
        super#value_binding vb
    end
  in
  iterator#structure structure;
  List.rev !found

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
  let candidates = labelled_bindings structure in
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
                          List.iter (positional_params params) ~f:(fun (position, (label, name)) ->
                              if List.mem carried name ~equal:String.equal then
                                found :=
                                  (match label with
                                  | Asttypes.Nolabel -> At_position position
                                  | Asttypes.Labelled l | Asttypes.Optional l -> At_label l)
                                  :: !found)))
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

    Scope is deliberately ignored -- a name is tainted for the whole file. A scan that itemises what
    a file pins does not need to know which [let] shadowed which; over-reach costs an extra
    inventory line, under-reach costs a missed pin. *)
let tainted_names ~generated ~utils ~emitters ~aliases ~seeds bindings =
  let tainted = ref (Set.of_list (module String) seeds) in
  let seeded body =
    mentions_generated_read ~generated body
    || reads_artifacts_directly ~utils body
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

type predicate = {
  pred_name : string;
  text_at : int option;
      (** Position of the pinned-text parameter among the positional arguments, when the fragment is
          one the CALLER supplies. [None] where the helper hard-codes the fragment itself
          ([let has_barrier src = String.is_substring src ~substring:"__syncthreads()"]) -- the
          helper is still a predicate, because its parameter is still generated source, and the
          literal in its body is picked up by the pin walk once that parameter joins the tainted
          set. *)
  source_param : string option;
      (** The parameter that carries the generated source, by name, when the predicate takes it
          rather than closing over it. Such a parameter IS generated source inside the predicate's
          body, so a literal the body tests against it -- the ["Main logic"] banner a helper slices
          on -- is a pin like any other, and the name joins the tainted set for the pin walk. *)
  source_at : int option;
      (** Position of the parameter that carries the generated source, when the predicate takes it
          rather than closing over it. Checked at each call site: a predicate is only pinning where
          the haystack it is handed really is generated source. *)
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
let predicates ~generated ~tainted bindings =
  let consumed = ref [] in
  let predicates =
    List.filter_map bindings ~f:(fun { names; params; body } ->
        match (names, params) with
        | [ name ], _ :: _ ->
            let index_of p =
              List.findi params ~f:(fun _ q -> String.equal p q) |> Option.map ~f:fst
            in
            let closes_over_source = List.exists (idents_in body) ~f:(fun i -> Set.mem tainted i) in
            let derived_params = params_derived_in ~params body in
            let result = ref None in
            let consider { text; tested; inherent } =
              let text_param =
                List.find params ~f:(fun p -> List.exists (idents_in text) ~f:(String.equal p))
              in
              (* The fragment argument inside a predicate's own definition tests a PARAMETER, not a
                 fragment; reading it as a pin would mark every file using the idiom partial. Only
                 that shape is consumed -- a literal the body hard-codes is left for the pin
                 walk. *)
              Option.iter text_param ~f:(fun _ ->
                  consumed :=
                    (text.pexp_loc.loc_start.pos_cnum, text.pexp_loc.loc_end.pos_cnum) :: !consumed);
              (* A body can hold several text tests -- a helper that slices on a banner and THEN
                 tests its own parameter. The one that takes the fragment from the caller is the
                 more informative reading, so it wins over one already recorded without a fragment
                 parameter, whichever the traversal reached first. *)
              let better candidate =
                match !result with
                | None -> true
                | Some existing -> Option.is_none existing.text_at && Option.is_some candidate
              in
              if better (Option.bind text_param ~f:index_of) then
                let source_param =
                  Option.bind tested ~f:(fun tested ->
                      match Set.to_list (derived_params tested) with
                      | [ p ] when not (Option.exists text_param ~f:(String.equal p)) -> Some p
                      | _ -> None)
                in
                let record source =
                  result :=
                    Some
                      {
                        pred_name = name;
                        text_at = Option.bind text_param ~f:index_of;
                        source_param = source;
                        source_at = Option.bind source ~f:index_of;
                      }
                in
                (* A parameter that IS the haystack makes this a predicate whether or not the caller
                   supplies the fragment. Requiring both left [let has_barrier src = ... ~substring:
                   "__syncthreads()"] unrecognised, so neither the literal nor a partial mark
                   reached the inventory (Codex P2, round 3). *)
                if Option.is_some source_param then record source_param
                else if Option.is_some text_param && (closes_over_source || inherent) then
                  record None
            in
            let iterator =
              object
                inherit Ast_traverse.iter as super

                method! expression e =
                  (match text_test ~generated e with Some t -> consider t | None -> ());
                  super#expression e
              end
            in
            iterator#expression body;
            !result
        | _ -> None)
  in
  (predicates, !consumed)

type pin = Literal of string | Format of string | Interpolated of string | Computed

(** How a pin argument reads.

    A literal is itself. A [Printf.sprintf] format, and a concatenation with a literal part, are
    fragments with a HOLE in them -- ["(float)(" ^ spelling ^ ")"], ["< (int)(%d.0))) {"] -- and
    both are itemised with the hole shown, because that is the context gh-ocannl-623 was found in
    only by reading a failing kernel. Anything with no literal part at all is [Computed], which
    marks the file's itemisation partial rather than being dropped silently. *)
let rec pin_of_expr ~literals expr =
  match string_literal expr with
  | Some text -> Literal text
  | None -> (
      (* A fragment named through a binding is still that fragment: [let arrow = " := " in
         String.substr_index statement ~pattern:arrow] pins the IR dump's assignment arrow as surely
         as spelling it at the call site would. Same resolution [Cache_dir_scan] makes for a
         directory name reached through a binding; without it the site reported only that it pins
         something the scan cannot name. *)
      match longident_of expr with
      | Some [ name ] when Map.mem literals name -> Literal (Map.find_exn literals name)
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
                        match pin_of_expr ~literals a with
                        | Literal text -> Printf.sprintf "%S" text
                        | Interpolated text -> text
                        | Format _ | Computed -> "...")
                  in
                  let rendered = String.concat ~sep:" ^ " parts in
                  if String.is_substring rendered ~substring:"\"" then Interpolated rendered
                  else Computed
              | _ -> Computed)
          | _ -> Computed))

(** Names bound directly to a string literal, for {!pin_of_expr} to resolve a fragment through.
    Simple bindings only: a name bound twice is dropped rather than guessed at. *)
let literal_bindings bindings =
  List.filter_map bindings ~f:(fun { names; params; body } ->
      match (names, params, string_literal body) with
      | [ name ], [], Some text -> Some (name, text)
      | _ -> None)
  |> List.sort_and_group ~compare:(fun (a, _) (b, _) -> String.compare a b)
  |> List.filter_map ~f:(function [ one ] -> Some one | _ -> None)
  |> Map.of_alist_exn (module String)

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

(** What each module alias in the file ultimately names, by last component:
    [module CR = Ir.Low_level.Canonical_render] answers [CR -> "Canonical_render"], and
    [module C = CR] answers the same for [C].

    {!module_aliases} answers the same question for ONE known target, which is what the artifact
    readers need. The emitters need it the other way round -- there are dozens of origin modules and
    one opened name to place -- and an [open] of an aliased emitter module is otherwise invisible
    twice over: not rejected, and not recognised at the call site either (Codex round 2 on
    lukstafi/ocannl-staging#487). *)
let module_alias_targets structure =
  let targets = Hashtbl.create (module String) in
  let resolve name = Option.value (Hashtbl.find targets name) ~default:name in
  let record alias expr =
    match module_source_name expr with
    | Some target -> Hashtbl.set targets ~key:alias ~data:(resolve target)
    | None -> ()
  in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! structure_item item =
        (match item.pstr_desc with
        | Pstr_module { pmb_name = { txt = Some alias; _ }; pmb_expr; _ } -> record alias pmb_expr
        | _ -> ());
        super#structure_item item

      method! expression expr =
        (match expr.pexp_desc with
        | Pexp_letmodule ({ txt = Some alias; _ }, module_expr, _) -> record alias module_expr
        | _ -> ());
        super#expression expr
    end
  in
  iterator#structure structure;
  resolve

(** Every name the file binds anywhere: a [let], a parameter, a pattern in a match.

    What this is for is the refusal below, and the direction matters. A name the file binds is a
    name an unqualified call MIGHT resolve to instead of to the opened module's, and telling which
    takes the scoping this scan deliberately does not carry -- [open Ir.Low_level] followed by
    [let to_doc x = local_render x] and then [to_doc value] is valid code calling the local
    function. Refusing it would fail the repository-wide inventory on a file that hides nothing, and
    a false refusal is a red build for everyone, where a refusal not made is one more member of the
    residue the partial marker already covers (Codex round 6 on lukstafi/ocannl-staging#487). *)
let names_bound_anywhere structure =
  let found = ref [] in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! pattern p =
        (match p.ppat_desc with Ppat_var { txt; _ } -> found := txt :: !found | _ -> ());
        super#pattern p
    end
  in
  iterator#structure structure;
  Set.of_list (module String) !found

(** Spellings this scan refuses rather than approximates, each reported as a failure by
    [codegen_text_inventory].

    Every route to generated text is attributed by the QUALIFIER at the call site --
    [Generated.read], [Utils.build_file], [Syntax.compile_proc]. An [open] removes the qualifier,
    and then the call is indistinguishable from a local function of the same name, so the file drops
    out of the census entirely: the silent direction, and the one every miss on gh-ocannl-712 took.

    The alternative to refusing is to track opened modules with their scope, which is one more
    approximation with one more edge. Refusing is a checkable fact, and it costs nothing: the
    qualified spelling is what every test in the tree already uses (gh-ocannl-748, from Codex round
    5).

    Raises if [contents] does not parse. *)
let rejections ~emitters ~path ~contents =
  let structure = structure_of contents in
  let generated = module_aliases ~target:"Generated" structure in
  let utils = module_aliases ~target:direct_artifact_module structure in
  let resolve = module_alias_targets structure in
  (* Which names an open of [opened_name] would make unqualified: the artifact readers by their
     module, the emitters by the module whose interface defines them -- through the file's own
     module aliases, since [open CR] after [module CR = Ir.Low_level.Canonical_render] opens the
     emitter's module under a name no origin spells. *)
  let hidden_by opened_name =
    let of_module names = List.map names ~f:(fun name -> (opened_name, name)) in
    let opened_target = resolve opened_name in
    (if Hash_set.mem generated opened_name then of_module [ "read"; "assert_emits"; "assert_omits" ]
     else [])
    @ (if Hash_set.mem utils opened_name then of_module direct_artifact_names else [])
    @ List.filter_map emitters ~f:(fun emitter ->
        let defined_in origin =
          match List.rev (String.split origin ~on:'.') with
          | _value :: enclosing :: _ -> String.equal enclosing opened_target
          | _ -> false
        in
        if List.exists emitter.origins ~f:defined_in then Some (opened_name, emitter.emitter_name)
        else None)
  in
  (* Through {!module_source_name}, so that opening a functor application directly -- [let open
     Ir.C_syntax.C_syntax (Cfg) in compile_proc ...], with no intermediate module to alias -- is the
     same open as one through a name. *)
  let opened_name declaration = module_source_name declaration.popen_expr in
  let extend hidden declaration =
    match opened_name declaration with Some name -> hidden_by name @ hidden | None -> hidden
  in
  (* Names the file binds for itself are struck from every refusal: see {!names_bound_anywhere}. *)
  let bound = names_bound_anywhere structure in
  let found = ref [] in
  (* Each open is judged over ITS OWN scope, which is what tells a file that hides a route from one
     that merely opens a module somewhere. Comparing the file's opens against the file's unqualified
     uses cross-products the two: a scoped [let open Ir.Low_level in ...] that calls nothing would
     then refuse an unrelated local [to_doc] elsewhere in the file, and this check fails the
     repository-wide inventory -- a false refusal is a red build on valid code (Codex round 3 on
     lukstafi/ocannl-staging#487). A structure-level open governs the items after it, an
     expression-level one its body, and a nested structure's opens die with it, which is the
     language's own rule. *)
  let walker =
    object (self)
      inherit [(string * string) list] Ast_traverse.map_with_context as super

      method! structure hidden items =
        ignore
          (List.fold items ~init:hidden ~f:(fun hidden item ->
               ignore (self#structure_item hidden item : structure_item);
               match item.pstr_desc with
               | Pstr_open declaration -> extend hidden declaration
               (* An [include] removes the qualifier exactly as an [open] does, and for the rest of
                  the structure just the same -- it also re-exports, which is beside the point here:
                  what matters is that [emit] afterwards is the emitter's (Codex round 6). *)
               | Pstr_include declaration -> (
                   match module_source_name declaration.pincl_mod with
                   | Some name -> hidden_by name @ hidden
                   | None -> hidden)
               | _ -> hidden));
        items

      method! expression hidden e =
        (match e.pexp_desc with
        | Pexp_open (declaration, body) ->
            ignore (self#expression (extend hidden declaration) body : expression)
        | Pexp_ident { txt = Longident.Lident name; _ } ->
            if not (Set.mem bound name) then
              List.iter hidden ~f:(fun (opened, hidden_name) ->
                  if String.equal name hidden_name then found := (opened, name) :: !found);
            ignore (super#expression hidden e : expression)
        | _ -> ignore (super#expression hidden e : expression));
        e
    end
  in
  ignore (walker#structure [] structure : structure);
  List.dedup_and_sort !found ~compare:Poly.compare
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
  let generated = module_aliases ~target:"Generated" structure in
  let utils = module_aliases ~target:direct_artifact_module structure in
  (* The bindings come first because the emitter aliases do: an emitter bound to a local name is
     called without a qualifier afterwards, and every rule below -- membership, taint, the buffer
     destinations, the pin walk -- has to recognise the same set of calls. Rules that know different
     routes are how a file stayed listed while the fragment it pins went missing. *)
  let bindings = bindings_in_structure structure in
  let aliases = emitter_aliases ~emitters structure in
  (* [Buffer] under whatever name this file gave it, for the backstop below: [module B = Buffer]
     then [B.contents buf] reads a buffer as surely as the bare spelling does. Resolved the way the
     artifact readers are. *)
  let buffers = module_aliases ~target:"Buffer" structure in
  let reads_generated = ref false in
  let reads_direct = ref false in
  let renders = ref false in
  let scan_reads =
    object
      inherit Ast_traverse.iter as super

      method! expression e =
        (match longident_of e with
        | Some p
          when List.exists [ "read"; "assert_emits"; "assert_omits" ] ~f:(fun name ->
                   calls ~aliases:generated p ~name) ->
            reads_generated := true
        | Some p
          when List.exists direct_artifact_names ~f:(fun name -> calls ~aliases:utils p ~name) ->
            reads_direct := true
        | Some p when Option.is_some (emitter_of_path ~emitters ~aliases p) -> renders := true
        | _ -> ());
        super#expression e
    end
  in
  scan_reads#structure structure;
  if not (!reads_generated || !reads_direct || !renders) then None
  else
    let seeds = buffer_destinations ~emitters ~aliases structure in
    let tainted = tainted_names ~generated ~utils ~emitters ~aliases ~seeds bindings in
    let predicates, consumed = predicates ~generated ~tainted bindings in
    (* A predicate's source parameter IS generated source, inside that predicate's body. Adding the
       name to the tainted set is how the literals a helper tests against it -- the banner it slices
       on, a second fragment it checks alongside its own argument -- become pins rather than being
       lost with the helper. Names are file-global here, as everywhere in this scan. *)
    let tainted =
      List.fold predicates ~init:tainted ~f:(fun acc p ->
          match p.source_param with Some name -> Set.add acc name | None -> acc)
    in
    let pins = ref [] in
    let is_consumed (e : expression) =
      List.mem consumed (e.pexp_loc.loc_start.pos_cnum, e.pexp_loc.loc_end.pos_cnum)
        ~equal:(fun (a, b) (c, d) -> a = c && b = d)
    in
    let literals = literal_bindings bindings in
    let record text = if not (is_consumed text) then pins := pin_of_expr ~literals text :: !pins in
    (* Generated source in the haystack, by any of the three routes -- a tainted name, an inline
       [Generated.read], an inline emitter render, an inline [build_files/] read. Naming only the
       first two here left an assertion that renders inline ([String.is_substring (render (LL.to_doc
       () llc)) ~substring:"-0.0"]) with its fragment silently dropped: the FILE stayed in the
       census through the membership branches, so nothing looked wrong, while grepping the inventory
       for the moved spelling missed the assertion (Codex P2, round 3). The membership rules and the
       pin rules have to know the same routes. *)
    let mentions_tainted e =
      List.exists (idents_in e) ~f:(fun i -> Set.mem tainted i)
      || mentions_generated_read ~generated e
      || renders_generated_text ~emitters ~aliases e
      || reads_artifacts_directly ~utils e
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
            (match longident_of inner with
            | Some path when calls ~aliases:buffers path ~name:"contents" -> found := true
            | _ -> ());
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
          (match text_test ~generated e with
          | Some { text; tested; inherent } ->
              if inherent || Option.value_map tested ~default:false ~f:mentions_tainted then
                record text
              else if Option.value_map tested ~default:false ~f:reads_a_buffer then
                unattributed := true
          | None -> (
              match e.pexp_desc with
              | Pexp_apply (callee, args) -> (
                  match longident_of callee with
                  | Some [ name ] -> (
                      match List.find predicates ~f:(fun p -> String.equal p.pred_name name) with
                      | None -> ()
                      | Some predicate -> (
                          let args = positional args in
                          let source = Option.bind predicate.source_at ~f:(List.nth args) in
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
                          match (source_ok, Option.bind predicate.text_at ~f:(List.nth args)) with
                          | true, Some text -> record text
                          | _ -> ()))
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
