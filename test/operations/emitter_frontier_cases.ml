(** What {!Emitter_frontier} derives, on interfaces built to break it rather than on whatever the
    compiler libraries export today (gh-ocannl-748).

    [codegen_text_inventory] derives the emitter frontier from the libraries' compiled interfaces,
    which removed the hand-maintained list three of the four review rounds on gh-ocannl-712 found a
    member missing from. A derivation has a different failure mode than a list, and a quieter one: a
    rule that stops recognising a shape, or a hand-over that loses a module, leaves the census
    SMALLER and no less confident. Neither shows up in the live inventory as anything but an
    absence.

    So both halves are controlled here on the fixture library beside this file. Its [.mli] declares
    every shape the rule has to answer for -- a renderer, a renderer whose document comes back in a
    tuple, one that writes into a buffer -- next to the near misses that decide the rule: a document
    combinator, a printer that accepts documents, a function given the code that hands back a
    string. And [emitter_fixture_b] has no [.mli] and no annotation at all, which is
    [C_syntax.compile_proc]'s situation and the reason the derivation reads types rather than
    sources.

    The last claim is the tripwire's own control: handed a wrapper interface with none of its
    members beside it, the derivation must SAY the modules are missing rather than report a frontier
    without them. *)

open Base
open Stdio

(** Where dune puts the fixture library's compiled interfaces, relative to this test's working
    directory. If dune ever puts them elsewhere the claims below fail rather than passing over an
    empty census -- which is the failure this whole test is about. *)
let fixture_objs = ".emitter_fixture.objs/byte"

let interfaces_in directory =
  match Stdlib.Sys.readdir directory with
  | exception _ -> []
  | entries ->
      Array.to_list entries
      |> List.filter ~f:(fun entry -> String.is_suffix entry ~suffix:".cmi")
      |> List.map ~f:(fun entry -> Stdlib.Filename.concat directory entry)
      |> List.sort ~compare:String.compare

let names emitters = List.map emitters ~f:(fun e -> e.Emitter_frontier.name)
let show list = String.concat ~sep:" " list

(** A claim about two sets of identities: the lists go to stderr whichever way it comes out, so a
    failure names what moved instead of only that something did. *)
let same claim ~derived ~declared =
  eprintf "%s\n  derived:  %s\n  declared: %s\n" claim (show derived) (show declared);
  Verdict.p claim (List.equal String.equal derived declared)

let () =
  let derived = Emitter_frontier.derive (interfaces_in fixture_objs) in
  let module F = Emitter_frontier in
  (* [renders_under_a_module_type] is declared in a named signature and exported as [Named], so it
     has to arrive with an origin naming the MODULE -- the name a call site spells -- and not only
     the signature it was declared in. *)
  same "the fixture's renderers are the frontier derived from it"
    ~derived:(names derived.F.emitters)
    ~declared:
      [
        "renders_a_document";
        "renders_a_triple";
        "renders_from_a_nested_module_type";
        "renders_through_a_chain";
        "renders_through_an_alias";
        "renders_through_an_option";
        "renders_under_a_module_type";
        "renders_without_an_annotation";
        "writes_into_a_buffer";
        "writes_into_an_aliased_buffer";
        "writes_into_an_unlabelled_buffer";
      ];
  (* The near misses, and why each is one: [combines_documents] and [joins_documents] produce a
     document out of numbers and other documents, so they render no program -- and since the scan
     matches an emitter by name behind any qualifier, admitting them would make a member of every
     test calling something so named. They are reported rather than dropped, so a renderer that
     lands in this bucket is a line in a diff. [consumes_documents] and [describes_the_code] are in
     neither list: one accepts documents, the other produces no text this scan can recognise. *)
  same "the fixture's document combinators are told apart from its renderers"
    ~derived:(names derived.F.combinators)
    ~declared:[ "combines_documents"; "joins_documents" ];
  let destinations =
    List.concat_map derived.F.emitters ~f:(fun e ->
        List.map e.F.destinations ~f:(fun destination ->
            e.F.name ^ ": " ^ F.render_destination destination))
  in
  same "a buffer-writing emitter comes back with where its text lands" ~derived:destinations
    ~declared:
      [
        "writes_into_a_buffer: ~buf";
        "writes_into_an_aliased_buffer: ~buf";
        "writes_into_an_unlabelled_buffer: argument 1";
      ];
  let origins_of name =
    List.concat_map derived.F.emitters ~f:(fun e ->
        if String.equal e.F.name name then e.F.origins else [])
  in
  same "a module type reachable only through another one is resolved too"
    ~derived:(origins_of "renders_from_a_nested_module_type")
    ~declared:
      [
        "Emitter_fixture.Emitter_fixture_a.OUTER.NESTED.renders_from_a_nested_module_type";
        "Emitter_fixture.Emitter_fixture_a.Outer.NESTED.renders_from_a_nested_module_type";
        "Emitter_fixture.Emitter_fixture_a.Uses_nested.renders_from_a_nested_module_type";
      ];
  same "a value exported under a module type is attributed to the module exporting it"
    ~derived:(origins_of "renders_under_a_module_type")
    ~declared:
      [
        "Emitter_fixture.Emitter_fixture_a.Named.renders_under_a_module_type";
        "Emitter_fixture.Emitter_fixture_a.RENDERER.renders_under_a_module_type";
      ];
  let declared = List.concat_map derived.F.interfaces ~f:(fun i -> i.F.declared) in
  let read = List.concat_map derived.F.interfaces ~f:(fun i -> i.F.read) in
  same "every module the fixture's wrapper interface declares was read" ~derived:read ~declared;
  (* The tripwire's control. A wrapper alone in a directory of its own is the shape a broken
     hand-over takes -- a glob that stopped matching, a library resolved from somewhere its members
     are not. The derivation must report the modules it could not read; reporting a frontier without
     them is how a census silently shrinks. *)
  let elsewhere = Stdlib.Filename.temp_file "emitter_frontier" ".dir" in
  Stdlib.Sys.remove elsewhere;
  Stdlib.Sys.mkdir elsewhere 0o700;
  let wrapper = Stdlib.Filename.concat fixture_objs "emitter_fixture.cmi" in
  let copy = Stdlib.Filename.concat elsewhere "emitter_fixture.cmi" in
  Stdlib.Out_channel.with_open_bin copy (fun out ->
      Stdlib.Out_channel.output_string out
        (Stdlib.In_channel.with_open_bin wrapper Stdlib.In_channel.input_all));
  let alone = Emitter_frontier.derive [ copy ] in
  let missing = List.concat_map alone.F.interfaces ~f:(fun i -> i.F.missing) in
  same "a hand-over that lost the member interfaces reports them as missing" ~derived:missing
    ~declared;
  Verdict.p_empty "a hand-over that lost the member interfaces derives no frontier from them"
    ~over:missing
    (alone.F.emitters @ alone.F.combinators);
  Stdlib.Sys.remove copy;
  Stdlib.Sys.rmdir elsewhere
