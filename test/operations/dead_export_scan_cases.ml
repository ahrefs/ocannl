open Base
open Ppxlib
module Scan = Test_utils.Dead_export_scan

let export_keys exports = List.map exports ~f:Scan.export_key |> List.sort ~compare:String.compare

let fixture_exports =
  Scan.exports_of_source ~source:"arrayjit/lib/sample.ml"
    {|
let plain = 1
let _x = 1
let x = 1
let _pair, public_pair = (2, 3)
let%trace _extended = 4
let pair, alias = (2, 3)
let (!@) x = x
let%trace extended = 4
include struct
  let included = 7
end
include (struct let constrained = 8 end : sig val constrained : int end)
external primitive : int -> int = "fixture_primitive"
let sexp_of_handwritten () = Sexplib0.Sexp.List []
type t = Root [@@deriving sexp_of, compare, equal]
type named = Named [@@deriving sexp, compare, equal]
type poly = [ `One | `Two ] [@@deriving sexp]
type group_a = Group_a and group_b = Group_b [@@deriving equal]
let outer =
  let nested = 5 in
  nested
module Nested = struct
  let hidden = 6
  type nested = Nested [@@deriving equal]
end
|}

let refs sources = Scan.references ~exports:fixture_exports ~sources

let referenced refs value =
  List.exists refs ~f:(fun (reference : Scan.reference) -> String.equal reference.value value)

let single_type_declaration source =
  match Test_utils.Config_key_scan.structure_of source with
  | [ { pstr_desc = Pstr_type (rec_flag, [ declaration ]); _ } ] -> (rec_flag, declaration)
  | _ -> failwith "expected exactly one type declaration"

let generated_value_names structure =
  List.concat_map structure ~f:(fun item ->
      match item.pstr_desc with
      | Pstr_value (_, bindings) ->
          List.concat_map bindings ~f:(fun binding -> Scan.pattern_names binding.pvb_pat)
      | _ -> [])
  |> List.dedup_and_sort ~compare:String.compare

let deriver_context loc =
  let base =
    Expansion_context.Base.top_level ~tool_name:"ocamlc" ~file_path:"fixture.ml"
      ~input_name:"fixture.ml"
  in
  Expansion_context.Deriver.make ~derived_item_loc:loc ~inline:false ~base ()

let expanded_names deriver source =
  let rec_flag, declaration = single_type_declaration source in
  let loc = declaration.ptype_loc in
  let declarations = (rec_flag, [ declaration ]) in
  let expansion =
    match deriver with
    (* The [sexp_of] half of [sexp] is excluded from the census by policy, so only the parser half
       of the expansion is expected. *)
    | "of_sexp" | "sexp" ->
        Ppx_sexp_conv_expander.Of_sexp.str_type_decl ~loc ~poly:false ~path:"Fixture" declarations
    | "compare" ->
        Ppx_compare_expander.Compare.str_type_decl ~ctxt:(deriver_context loc) declarations false
    | "equal" ->
        Ppx_compare_expander.Equal.str_type_decl ~ctxt:(deriver_context loc) declarations false
    | _ -> failwith "unsupported fixture deriver"
  in
  (declaration, generated_value_names expansion)

let () =
  Verdict.p "top-level lets, patterns, extensions, include structs, and externals are exports"
    (List.equal String.equal (export_keys fixture_exports)
       [
         "Sample.!@";
         "Sample.__poly_of_sexp__";
         "Sample.alias";
         "Sample.compare";
         "Sample.compare_named";
         "Sample.equal";
         "Sample.equal_group_a";
         "Sample.equal_group_b";
         "Sample.equal_named";
         "Sample.extended";
         "Sample.included";
         "Sample.named_of_sexp";
         "Sample.outer";
         "Sample.pair";
         "Sample.plain";
         "Sample.poly_of_sexp";
         "Sample.primitive";
         "Sample.public_pair";
         "Sample.x";
       ]);
  Verdict.p_none "a constrained include publishes through its own signature and is not censused"
    [ "Sample.constrained" ] ~f:(fun key ->
      List.mem (export_keys fixture_exports) key ~equal:String.equal);
  Verdict.p_none "underscore-prefixed lets, patterns, and extensions are private by policy"
    [ "Sample._x"; "Sample._pair"; "Sample._extended" ] ~f:(fun key ->
      List.mem (export_keys fixture_exports) key ~equal:String.equal);
  let private_counts = Scan.counts ~exports:fixture_exports [] in
  Verdict.p "the opposing unprefixed value remains a detected zero-reference export"
    (Hashtbl.find_exn private_counts "Sample.x" = 0);
  Verdict.p_none
    "derived and hand-written sexp_of converters are excluded from the census by policy"
    [
      "Sample.sexp_of_t";
      "Sample.sexp_of_named";
      "Sample.sexp_of_poly";
      "Sample.sexp_of_handwritten";
    ] ~f:(fun key -> List.mem (export_keys fixture_exports) key ~equal:String.equal);
  Verdict.p_none "nested lets and nested-module derived values are not exports" fixture_exports
    ~f:(fun (export : Scan.export) ->
      String.equal export.value "nested"
      || String.equal export.value "hidden"
      || String.equal export.value "equal_nested");
  Verdict.p "a deriving on the last declaration covers its whole recursive type group"
    (List.mem (export_keys fixture_exports) "Sample.equal_group_a" ~equal:String.equal
    && List.mem (export_keys fixture_exports) "Sample.equal_group_b" ~equal:String.equal);
  Verdict.p "sexp derives the polymorphic-variant parser helper"
    (List.mem (export_keys fixture_exports) "Sample.__poly_of_sexp__" ~equal:String.equal);
  let expansion_cases =
    [
      ("of_sexp", "type t = Root\n");
      ("of_sexp", "type named = Named\n");
      ("of_sexp", "type poly = [ `One | `Two ]\n");
      ("sexp", "type named = Named\n");
      ("sexp", "type poly = [ `One | `Two ]\n");
      ("compare", "type t = Root\n");
      ("compare", "type named = Named\n");
      ("equal", "type t = Root\n");
      ("equal", "type named = Named\n");
    ]
  in
  Verdict.p_all "every derived export-name set matches its complete PPX expansion" expansion_cases
    ~f:(fun (deriver, source) ->
      let declaration, actual = expanded_names deriver source in
      List.equal String.equal (Scan.derived_names ~derivers:[ deriver ] declaration) actual);
  Verdict.p "dune select alternatives are not mistaken for module sources"
    (Option.is_none (Scan.module_name_of_source "arrayjit/lib/sample.missing.ml"));
  let direct =
    refs
      [
        ("consumer.ml", "let a = Sample.plain\nlet b = Wrapper.Sample.pair\nlet c = Sample.(!@) 1\n");
        ("arrayjit/lib/sample.ml", "let self = Sample.outer\n");
      ]
  in
  Verdict.p "direct and nested qualified paths count outside the defining source"
    (referenced direct "plain" && referenced direct "pair" && referenced direct "!@");
  Verdict.p_none "references in the defining source do not count" direct
    ~f:(fun (reference : Scan.reference) -> String.equal reference.value "outer");
  let aliased =
    refs
      [
        ( "aliases.ml",
          "module S = Sample\n\
           module T = S\n\
           let a = T.alias\n\
           let b = let module U = T in U.extended\n" );
      ]
  in
  Verdict.p "structure, chained, and local module aliases count"
    (referenced aliased "alias" && referenced aliased "extended");
  let opened =
    refs
      [
        ( "opened.ml",
          "open Sample\n\
           let a = plain\n\
           let b = let open Sample in pair\n\
           let c = let primitive = 0 in primitive\n" );
      ]
  in
  Verdict.p "unqualified identifiers in an open scope count conservatively"
    (referenced opened "plain" && referenced opened "pair" && referenced opened "primitive");
  let inert_sources = [ ("inert.ml", "let s = \"Sample.outer\" (* Sample.alias *)\n") ] in
  let inert = refs inert_sources in
  Verdict.p_empty "comments and strings do not count as references" ~over:inert_sources inert;
  let extensions =
    refs
      [
        ( "extensions.ml",
          "module S = Sample\n\
           let a = [%equal: S.named] Named Named\n\
           let b = [%compare: Wrapper.Sample.t] Root Root\n\
           let c = [%sexp_of: Sample.named list]\n\
           let d = [%of_sexp: Sample.named]\n" );
      ]
  in
  Verdict.p "equal, compare, and of_sexp extensions reference derived values"
    (referenced extensions "equal_named"
    && referenced extensions "compare"
    && referenced extensions "named_of_sexp");
  Verdict.p_none "a sexp_of extension references nothing" extensions
    ~f:(fun (reference : Scan.reference) -> String.is_prefix reference.value ~prefix:"sexp_of_");
  Verdict.p_none "an extension does not credit a different derivation from the same type"
    [ "compare_named" ] ~f:(referenced extensions);
  let derived =
    refs [ ("derived.ml", "type t = Sample.named [@@deriving sexp, compare, equal]\n") ]
  in
  Verdict.p "a deriving references converters of its component types"
    (referenced derived "named_of_sexp"
    && referenced derived "compare_named"
    && referenced derived "equal_named");
  let inherited =
    refs [ ("inherited.ml", "type t = [ Sample.poly | `Three ] [@@deriving of_sexp]\n") ]
  in
  Verdict.p_all "an inherited row references its internal parser helper"
    [ ("__poly_of_sexp__", true); ("poly_of_sexp", false) ]
    ~f:(fun (value, expected) -> Bool.equal (referenced inherited value) expected);
  let poly_rec_flag, poly_declaration = single_type_declaration "type poly = [ `One | `Two ]\n" in
  let poly_expansion =
    Ppx_sexp_conv_expander.Of_sexp.str_type_decl ~loc:poly_declaration.ptype_loc ~poly:false
      ~path:"Sample"
      (poly_rec_flag, [ poly_declaration ])
  in
  Verdict.p "the internal parser helper name matches the ppx_sexp_conv expansion"
    (List.mem (generated_value_names poly_expansion) "__poly_of_sexp__" ~equal:String.equal);
  let inherited_extension =
    refs [ ("extension.ml", "let f = [%of_sexp: [ Sample.poly | `Three ]]\n") ]
  in
  Verdict.p_all "an inherited row in an extension also references its internal parser helper"
    [ ("__poly_of_sexp__", true); ("poly_of_sexp", false) ]
    ~f:(fun (value, expected) -> Bool.equal (referenced inherited_extension value) expected);
  let ignored =
    refs
      [
        ( "ignored.ml",
          "type a = (Sample.named[@sexp.opaque]) [@@deriving sexp]\n\
           type b = (Sample.named[@compare.ignore]) [@@deriving compare]\n\
           type c = (Sample.named[@equal.ignore]) [@@deriving equal]\n\
           type d = (Sample.named[@sexp.ignore]) [@@deriving sexp]\n\
           let a = [%of_sexp: (Sample.named[@sexp.opaque]) list]\n\
           let b = [%of_sexp: Sample.named sexp_opaque]\n" );
      ]
  in
  Verdict.p_none "opaque and ignored types do not credit unused derived values"
    [ "named_of_sexp"; "compare_named"; "equal_named" ]
    ~f:(referenced ignored);
  let deriving_attribute =
    refs [ ("attribute.ml", "open Sample\ntype t = T [@@deriving equal]\n") ]
  in
  Verdict.p_none "a deriving attribute is not an ordinary opened value reference" [ "equal" ]
    ~f:(referenced deriving_attribute);
  let functor_sources = [ ("functor.ml", "let f = [%of_sexp: F(X).t]\n") ] in
  let functor_application = refs functor_sources in
  Verdict.p_empty "a functor-application type path is accepted without guessed credit"
    ~over:functor_sources functor_application;
  (* gh-ocannl-1085: an include re-exports, it does not use. [Datatypes.mutable_list] hid for years
     behind [utils.ml]'s [include Datatypes] when the include itself credited every value. *)
  let include_only_sources = [ ("included.ml", "include Sample\n") ] in
  Verdict.p_empty "an include alone references nothing it re-exports" ~over:include_only_sources
    (refs include_only_sources);
  let included_counts =
    Scan.counts ~exports:fixture_exports
      (refs [ ("included.ml", "include Sample\n"); ("user.ml", "let a = Included.plain\n") ])
  in
  Verdict.p "a value used only through the includer is live"
    (Hashtbl.find_exn included_counts "Sample.plain" > 0);
  Verdict.p "a value the includer re-exports but nobody uses stays dead"
    (Hashtbl.find_exn included_counts "Sample.x" = 0);
  let through_includers =
    refs
      [
        ("included.ml", "include Sample\n");
        ("twice.ml", "include Included\n");
        ( "wrapper.ml",
          "module S = Sample\n\
           module Inner = struct include S end\n\
           module Rec = struct include struct include Sample end end\n" );
        ("selected.cudajit.ml", "include Sample\n");
        ("inside.ml", "include Sample\nlet own = alias\n");
        ("local.ml", "let a = let module L = struct include Sample end in L.included\n");
        ("exported.ml", "module Alias = Included\nmodule Direct = Sample\n");
        ( "user.ml",
          "let a = Twice.x\n\
           let g = Exported.Alias.outer\n\
           let h = Exported.Direct.poly_of_sexp\n\
           let b = Wrapper.Inner.pair\n\
           let c = Selected.extended\n\
           let d = Rec.public_pair\n\
           open Included\n\
           let e = primitive\n\
           let f = [%equal: Twice.named] Named Named\n" );
      ]
  in
  Verdict.p_all
    "includers are receivers transitively, nested, through aliases and select alternatives, \
     opened, below the include itself, local to an expression, and aliased in another source"
    [
      "x";
      "pair";
      "extended";
      "public_pair";
      "primitive";
      "equal_named";
      "alias";
      "included";
      "outer";
      "poly_of_sexp";
    ]
    ~f:(referenced through_includers);
  let unrelated =
    refs
      [
        ("included.ml", "include Sample\n");
        ("other.ml", "include Unrelated\n");
        ( "user.ml",
          "let a = Other.plain\nmodule Functorized = F (Included)\nlet b = Functorized.x\n" );
      ]
  in
  Verdict.p_none "an include of another module, or a functor application, makes no receiver"
    [ "plain"; "x" ] ~f:(referenced unrelated);
  let counts = Scan.counts ~exports:fixture_exports direct in
  Verdict.p "a synthetic zero-reference export is detected"
    (Hashtbl.find_exn counts "Sample.primitive" = 0);
  Verdict.p "a synthetic zero-reference derived export is detected"
    (Hashtbl.find_exn counts "Sample.equal_named" = 0);
  let extension_counts = Scan.counts ~exports:fixture_exports extensions in
  Verdict.p "an export referenced only through an extension point is not reported dead"
    (Hashtbl.find_exn extension_counts "Sample.equal_named" > 0)

(* gh-ocannl-1009: the type census. The fixture's [example_train_result] is the record that
   motivated it -- live-looking labels, no mention of its name anywhere. *)
let type_fixture =
  {|
type example_train_result = { inputs : int; outputs : int }
type tree = Leaf | Node of tree list
type annotated = A
type interfaced = I
type prose = P
type by_label = { only_label : int }
type by_constructor = Only_constructor
type derived = D [@@deriving equal]
type derived_sexp = S [@@deriving sexp_of]
type in_extension = E
type in_with = W
type in_package = Pk
type in_pattern = [ `Ip ]
type in_functor_path = Fp
type by_fields = { fields_label : int } [@@deriving fields]
type fields_module = { module_label : int } [@@deriving fields]
type fields_alias = { alias_label : int } [@@deriving fields]
type variants_open = Vo [@@deriving variants]
include struct
  type in_include = Ii
end
type extensible = ..
type _private = X
let own_use (x : annotated) = x
module Nested = struct
  type nested_only = N
end
|}

let type_exports = Scan.type_exports_of_source ~source:"arrayjit/lib/sample.ml" type_fixture

let type_counts ?(interfaces = []) sources =
  Scan.type_mention_counts ~type_exports
    ~implementations:(("arrayjit/lib/sample.ml", type_fixture) :: sources)
    ~interfaces

let mentions counts name = Hashtbl.find_exn counts ("Sample." ^ name)

let () =
  Verdict.p "top-level types are censused; underscore-prefixed and nested ones are not"
    (List.equal String.equal
       (List.map type_exports ~f:Scan.type_export_key)
       [
         "Sample.annotated";
         "Sample.by_constructor";
         "Sample.by_fields";
         "Sample.by_label";
         "Sample.derived";
         "Sample.derived_sexp";
         "Sample.example_train_result";
         "Sample.extensible";
         "Sample.fields_alias";
         "Sample.fields_module";
         "Sample.in_extension";
         "Sample.in_functor_path";
         "Sample.in_include";
         "Sample.in_package";
         "Sample.in_pattern";
         "Sample.in_with";
         "Sample.interfaced";
         "Sample.prose";
         "Sample.tree";
         "Sample.variants_open";
       ]);
  let consumers =
    [
      ( "consumer.ml",
        "(* A comment naming Sample.prose and example_train_result. *)\n\
         (** A docstring naming [prose]. *)\n\
         let s = \"prose example_train_result\"\n\
         let r = { Other.inputs = 1; outputs = 2 }\n\
         let l = { Sample.only_label = 1 }\n\
         let c = Sample.Only_constructor\n\
         let e = Sample.equal_derived\n\
         let x = Sample.sexp_of_derived_sexp\n\
         let f = [%compare: Sample.in_extension]\n\
         module type S = sig type t end with type t = Sample.in_with\n\
         type Sample.extensible += More\n\
         let p (module M : S with type in_package = int) = ()\n\
         let q = function #Sample.in_pattern -> true | _ -> false\n\
         let r (x : F(X).in_functor_path) = x\n\
         let g x = (x.Other.fields_label, fields_label x, { Other.fields_label = 1 })\n\
         let n = Sample.Fields_of_fields_module.names\n" );
    ]
  in
  let counts =
    type_counts
      ~interfaces:
        [
          ( "other.mli",
            "val f : Sample.interfaced -> unit\n\
             module F = Sample.Fields_of_fields_alias\n\
             open Sample.Variants_of_variants_open\n" );
        ]
      consumers
  in
  Verdict.p "the dead record is detected though another record shares its label names"
    (mentions counts "example_train_result" = 0);
  Verdict.p "a recursive type does not credit itself" (mentions counts "tree" = 0);
  Verdict.p "comments, docstrings, and strings do not mention a type" (mentions counts "prose" = 0);
  Verdict.p "labels and constructors are not resolved to their type, by stated boundary"
    (mentions counts "by_label" = 0 && mentions counts "by_constructor" = 0);
  Verdict.p "an annotation in the defining source mentions the type"
    (mentions counts "annotated" > 0);
  Verdict.p "an interface signature mentions the type" (mentions counts "interfaced" > 0);
  Verdict.p_all "a spelled derived value mentions its type" [ "derived"; "derived_sexp" ]
    ~f:(fun name -> mentions counts name > 0);
  Verdict.p_all "extension payloads, with-constraints, and type extensions mention the type"
    [ "in_extension"; "in_with"; "extensible" ] ~f:(fun name -> mentions counts name > 0);
  Verdict.p "a type declared inside a top-level include struct is censused and can be unmentioned"
    (mentions counts "in_include" = 0);
  Verdict.p_all "a derived module mentions its type from an interface's alias or open"
    [ "fields_alias"; "variants_open" ] ~f:(fun name -> mentions counts name > 0);
  Verdict.p "a first-class module's package constraint mentions the type"
    (mentions counts "in_package" > 0);
  Verdict.p "a #type pattern mentions the polymorphic variant it names"
    (mentions counts "in_pattern" > 0);
  Verdict.p "a type path through a functor application mentions its last component"
    (mentions counts "in_functor_path" > 0);
  Verdict.p "a fields deriving is credited by its module, never by a same-named label or value"
    (mentions counts "fields_module" > 0 && mentions counts "by_fields" = 0);
  let deriving_payload =
    Scan.type_exports_of_source ~source:"arrayjit/lib/payload.ml" "type t = T [@@deriving equal]"
  in
  let payload_counts =
    Scan.type_mention_counts ~type_exports:deriving_payload
      ~implementations:
        [
          ("arrayjit/lib/payload.ml", "type t = T [@@deriving equal]");
          ("elsewhere.ml", "type u = U [@@deriving equal, compare]\n");
        ]
      ~interfaces:[]
  in
  Verdict.p "a deriving payload naming a deriver is not a spelling of the derived value"
    (Hashtbl.find_exn payload_counts "Payload.t" = 0);
  let unqualified =
    type_counts [ ("opened.ml", "open Sample\nlet f (x : example_train_result) = x\n") ]
  in
  Verdict.p "an unqualified mention under an open counts"
    (mentions unqualified "example_train_result" > 0);
  let same_name =
    type_counts
      [
        ("elsewhere.ml", "type example_train_result = int\nlet f (x : example_train_result) = x\n");
      ]
  in
  Verdict.p "a same-named type elsewhere credits by name, conservatively"
    (mentions same_name "example_train_result" > 0);
  Verdict.p "dune's preprocessed interface beside its source is not an interface source"
    (List.equal String.equal
       (Scan.interfaces_among [ "a.mli"; "a.pp.mli"; "b.pp.mli"; "c.ml" ])
       [ "a.mli"; "b.pp.mli" ])

(* The derived half of a type's mentions, held equal to the derivers' own expansion rather than to a
   restatement of their naming rules. Referencing each registration links it, so [Driver] expands
   the fixture with the real derivers; [ppx_hash] exports nothing a program can reference, but its
   archive is force-linked, so listing the library registers it. *)
let _registered =
  ( Ppx_sexp_conv.sexp,
    Ppx_compare.compare,
    Ppx_enumerate.enumerate,
    Ppx_variants_conv.variants,
    Ppx_fields_conv.fields )

let generated_names structure =
  let rec items structure = List.concat_map structure ~f:item
  and item structure_item =
    match structure_item.pstr_desc with
    | Pstr_value (_, bindings) ->
        List.concat_map bindings ~f:(fun binding -> Scan.pattern_names binding.pvb_pat)
    | Pstr_module { pmb_name = { txt = Some name; _ }; _ } -> [ name ]
    | Pstr_include { pincl_mod = { pmod_desc = Pmod_structure nested; _ }; _ } -> items nested
    | _ -> []
  in
  items structure |> List.dedup_and_sort ~compare:String.compare

let () =
  let fixtures =
    [
      "type t = A | B of int [@@deriving sexp, compare, equal, hash, enumerate, variants]";
      "type named = A | B of int [@@deriving sexp, compare, equal, hash, enumerate, variants]";
      "type t = { label_a : int; label_b : string } [@@deriving sexp_of, of_sexp, fields]";
      "type record = { label_a : int; mutable label_b : string } [@@deriving fields, hash]";
      "type poly = [ `One | `Two ] [@@deriving sexp, compare]";
      "type ('a, 'b) param = 'a * 'b [@@deriving sexp_of, equal, hash]";
    ]
  in
  Verdict.p_all "a type's derived mentions are exactly what its derivers generate, members aside"
    fixtures ~f:(fun source ->
      let _, declaration = single_type_declaration source in
      let members =
        match declaration.ptype_kind with
        | Ptype_record labels -> List.map labels ~f:(fun label -> label.pld_name.txt)
        | Ptype_variant constructors ->
            List.map constructors ~f:(fun constructor ->
                String.uncapitalize constructor.pcd_name.txt)
        | Ptype_abstract | Ptype_open -> []
      in
      let structure = Test_utils.Config_key_scan.structure_of source in
      let expanded =
        generated_names (Ppxlib.Driver.map_structure structure)
        |> List.filter ~f:(fun name ->
            (* What [fields] names after a label and [variants] after a constructor falls under the
               member boundary: crediting it would credit the member's name. *)
            not
              (List.exists members ~f:(fun member ->
                   List.mem
                     [ member; "set_" ^ member; "is_" ^ member; member ^ "_val" ]
                     name ~equal:String.equal)))
      in
      let mentions =
        match Scan.type_exports_of_source ~source:"arrayjit/lib/fixture.ml" source with
        | [ export ] -> export.mentioned_by
        | _ -> []
      in
      let equal = List.equal String.equal expanded mentions in
      if not equal then
        Stdio.eprintf "%s\n  expansion: %s\n  mentions:  %s\n" source
          (String.concat ~sep:" " expanded) (String.concat ~sep:" " mentions);
      equal)

(* gh-ocannl-913: the value census over [hash], [enumerate], [fields] and [variants]. Each export
   set is held equal to what the installed derivers generate -- values and modules alike,
   member-named ones included -- rather than to a restatement of their naming rules. *)
let () =
  let fixtures =
    [
      "type t = A | B of int [@@deriving hash]";
      "type named = A | B of int [@@deriving hash]";
      "type ('a, 'b) param = 'a * 'b [@@deriving hash]";
      "type alias = int [@@deriving hash]";
      "type t = A | B of bool [@@deriving enumerate]";
      "type poly = [ `One | `Two of bool ] [@@deriving enumerate]";
      "type t = { label_a : int; mutable label_b : string } [@@deriving fields]";
      "type record = { label_a : int; mutable label_b : string } [@@deriving fields]";
      "type t = Open | RHS1 | Undet of int [@@deriving variants]";
      "type poly = [ `Method | `B of int ] [@@deriving variants]";
      "type named = A [@@deriving sexp, compare, equal, hash, enumerate, variants]";
    ]
  in
  Verdict.p_all "every derived export set is exactly what the installed derivers generate" fixtures
    ~f:(fun source ->
      let _, declaration = single_type_declaration source in
      let expanded =
        generated_names
          (Ppxlib.Driver.map_structure (Test_utils.Config_key_scan.structure_of source))
        |> List.filter ~f:(fun name -> not (Scan.is_sexp_of_converter name))
      in
      let derivers = Scan.derivers_of_type_declaration declaration in
      let model = Scan.derived_names ~derivers declaration in
      let equal = List.equal String.equal expanded model in
      if not equal then
        Stdio.eprintf "%s\n  expansion: %s\n  model:     %s\n" source
          (String.concat ~sep:" " expanded) (String.concat ~sep:" " model);
      equal);
  let gen =
    {|
type h = H [@@deriving hash]
type e = E [@@deriving enumerate]
type r = { field_a : int } [@@deriving fields]
type v = Va [@@deriving variants]
type unused = U [@@deriving hash, enumerate]
type r2 = { other_field : int } [@@deriving fields]
|}
  in
  let exports = Scan.exports_of_source ~source:"arrayjit/lib/gen.ml" gen in
  let count sources key =
    let sources = ("arrayjit/lib/gen.ml", gen) :: sources in
    Hashtbl.find_exn (Scan.counts ~exports (Scan.references ~exports ~sources)) ("Gen." ^ key)
  in
  let through_extensions =
    [
      ( "user.ml",
        "let a = [%hash: Gen.h]\n\
         let b = [%hash_fold: Gen.h list]\n\
         let c = [%all: Gen.e option]\n\
         let d = Gen.Fields_of_r.names\n\
         open Gen\n\
         let e = Variants_of_v.to_rank\n" );
    ]
  in
  Verdict.p_all
    "a generated helper referenced only through an extension point, or held by a used generated \
     module, is live"
    [ "hash_h"; "hash_fold_h"; "all_of_e"; "Fields_of_r"; "field_a"; "Variants_of_v"; "va" ]
    ~f:(fun key -> count through_extensions key > 0);
  Verdict.p_all "a genuinely unused generated helper is reported"
    [
      "hash_unused";
      "hash_fold_unused";
      "all_of_unused";
      "is_va";
      "va_val";
      "other_field";
      "Fields_of_r2";
    ] ~f:(fun key -> count through_extensions key = 0);
  let bare_hash =
    Scan.references ~exports ~sources:[ ("user.ml", "let a = [%hash: Gen.unused]\n") ]
    |> List.map ~f:(fun (reference : Scan.reference) -> reference.value)
  in
  Verdict.p "a [%hash] of a bare constructor calls hash_<type>, not its fold"
    (List.equal String.equal bare_hash [ "hash_unused" ]);
  Verdict.p "a deriving with arguments the census does not model is refused"
    (match
       Scan.exports_of_source ~source:"arrayjit/lib/args.ml"
         "type r = { a : int } [@@deriving sexp_of, fields ~getters]"
     with
    | _ -> false
    | exception Scan.Unmodelled_deriving_arguments { deriver = "fields"; line = 1 } -> true)

(* gh-ocannl-914: the differential corpus. For each fixture, what the scanner infers from the
   unexpanded source is held equal to what the installed PPX expansion actually calls: the
   [Sample.x] paths the expanded consumer spells, and, inside the expanded [Sample] itself, which
   generated value calls which (the support references). A PPX that changes a name or a call fails
   here as contract drift, before the census can miscount. *)
let flatten longident = try Some (Longident.flatten_exn longident) with _ -> None

(* The values and modules of [Sample] an expanded consumer spells, qualified. *)
let sample_spellings structure =
  let found = ref [] in
  let spell txt =
    match flatten txt with Some ("Sample" :: name :: _) -> found := name :: !found | _ -> ()
  in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! expression expression =
        (match expression.pexp_desc with Pexp_ident { txt; _ } -> spell txt | _ -> ());
        super#expression expression

      method! module_expr module_expr =
        (match module_expr.pmod_desc with Pmod_ident { txt; _ } -> spell txt | _ -> ());
        super#module_expr module_expr
    end
  in
  iterator#structure structure;
  !found

(* [(caller, callee)] for every generated value or module of an expansion (what sits in the [include
   struct ... end] a deriving emits) whose body names [callee] unqualified. *)
let generated_calls structure =
  let callees body_iter =
    let found = ref [] in
    let iterator =
      object
        inherit Ast_traverse.iter as super

        method! expression expression =
          (match expression.pexp_desc with
          | Pexp_ident { txt = Lident name; _ } -> found := name :: !found
          | _ -> ());
          super#expression expression
      end
    in
    body_iter iterator;
    !found
  in
  List.concat_map structure ~f:(fun item ->
      match item.pstr_desc with
      | Pstr_include { pincl_mod = { pmod_desc = Pmod_structure generated; _ }; _ } ->
          List.concat_map generated ~f:(fun item ->
              match item.pstr_desc with
              | Pstr_value (_, bindings) ->
                  List.concat_map bindings ~f:(fun binding ->
                      match Scan.pattern_names binding.pvb_pat with
                      | [ caller ] ->
                          List.map
                            (callees (fun it -> it#expression binding.pvb_expr))
                            ~f:(fun callee -> (caller, callee))
                      | _ -> [])
              | Pstr_module { pmb_name = { txt = Some caller; _ }; pmb_expr; _ } ->
                  List.map
                    (callees (fun it -> it#module_expr pmb_expr))
                    ~f:(fun callee -> (caller, callee))
              | _ -> [])
      | _ -> [])

let pairs_equal = List.equal (fun (a, b) (c, d) -> String.equal a c && String.equal b d)

let sorted_pairs pairs =
  List.dedup_and_sort pairs ~compare:(fun (a, b) (c, d) ->
      match String.compare a c with 0 -> String.compare b d | ordering -> ordering)

let () =
  let corpus =
    [
      ( "every family over a plain and a [t] type, through derivings and extensions",
        "type named = Named [@@deriving sexp, compare, equal, hash, enumerate]\n\
         type t = Root [@@deriving sexp, compare, equal, hash, enumerate]\n",
        "type c = { n : Sample.named; r : Sample.t } [@@deriving sexp, compare, equal, hash, \
         enumerate]\n\
         let a = [%compare: Sample.named]\n\
         let b = [%equal: Sample.t]\n\
         let c = [%of_sexp: Sample.named list]\n\
         let d = [%hash: Sample.named]\n\
         let e = [%hash: Sample.t]\n\
         let f = [%hash: Sample.named list]\n\
         let g = [%hash_fold: Sample.t]\n\
         let h = [%all: Sample.named option]\n\
         let i = [%compare.equal: Sample.t]\n" );
      ( "an alias of a primitive reached through a tuple in an option in a record field \
         (staging#740)",
        "type provenance = string [@@deriving sexp, compare, equal, hash]\n\
         type record = { intent : (int * provenance) option } [@@deriving sexp, compare, equal, \
         hash]\n",
        "type a = Sample.provenance [@@deriving hash, compare]\n\
         type b = (int * Sample.provenance) option [@@deriving sexp, equal]\n" );
      ( "inherited polymorphic rows",
        "type poly = [ `One | `Two ] [@@deriving sexp, compare, equal, hash, enumerate]\n\
         type wider = [ poly | `Three ] [@@deriving sexp, compare, hash, enumerate]\n",
        "type c = [ Sample.poly | `Four ] [@@deriving sexp, equal, hash, enumerate]\n\
         let f = [%of_sexp: [ Sample.poly | `Five ]]\n" );
      (* Each ignored component is the only carrier of its type, so a misread attribute shows. *)
      ( "ignored and opaque components",
        "type u1 = U1 [@@deriving sexp, compare, equal, hash]\n\
         type u2 = U2 [@@deriving sexp, compare, equal, hash]\n\
         type u3 = U3 [@@deriving sexp, compare, equal, hash]\n\
         type u4 = U4 [@@deriving sexp, compare, equal, hash]\n\
         type r = { a : u1 [@compare.ignore]; b : (u2[@sexp.opaque]); c : u3 [@hash.ignore];\n\
        \  d : (u4[@compare.ignore]) } [@@deriving sexp, compare, hash]\n\
         type q = { e : u1 [@equal.ignore]; f : u2 } [@@deriving equal]\n",
        "type c = { a : Sample.u1 [@compare.ignore]; b : (Sample.u2[@sexp.opaque]);\n\
        \  c : Sample.u3 [@hash.ignore]; d : Sample.u4 } [@@deriving sexp, compare, hash]\n\
         type e = { e : Sample.u1 [@equal.ignore]; f : Sample.u4 } [@@deriving equal]\n" );
      ( "a GADT's result index, which no deriving calls",
        "type u = U [@@deriving hash]\n\
         type v = V [@@deriving hash]\n\
         type _ g = G : u -> v g [@@deriving hash]\n",
        "type _ c = C : Sample.u -> Sample.v c [@@deriving hash]\n" );
      ( "a mutually recursive group",
        "type a = A of b option | A0 and b = B of a list [@@deriving sexp, compare, hash]\n",
        "type c = Sample.a * Sample.b [@@deriving sexp, compare, hash]\n" );
      ( "hash aliases, inside the module and out",
        "type base = B [@@deriving hash]\n\
         type al = base [@@deriving hash]\n\
         type pal = base list [@@deriving hash]\n",
        "type t = Sample.al [@@deriving hash]\ntype u = Sample.pal [@@deriving hash]\n" );
      ( "the modules fields and variants generate",
        "type r = { x : int; mutable y : int } [@@deriving fields]\n\
         type v = Va | Vb of int [@@deriving variants]\n",
        "let a = Sample.Fields_of_r.names\nmodule V = Sample.Variants_of_v\n" );
    ]
  in
  let exercised = ref [] in
  Verdict.p_all "the scanner's derived references are exactly the expansion's calls" corpus
    ~f:(fun (label, sample, consumer) ->
      let sample_path = "arrayjit/lib/sample.ml" in
      let exports = Scan.exports_of_source ~source:sample_path sample in
      let names = Set.of_list (module String) (List.map exports ~f:(fun e -> e.Scan.value)) in
      let references =
        Scan.references ~exports ~sources:[ (sample_path, sample); ("consumer.ml", consumer) ]
      in
      let external_model =
        List.filter_map references ~f:(fun (r : Scan.reference) ->
            Option.some_if (Option.is_none r.via) r.value)
        |> List.dedup_and_sort ~compare:String.compare
      in
      let external_truth =
        sample_spellings
          (Ppxlib.Driver.map_structure (Test_utils.Config_key_scan.structure_of consumer))
        |> List.filter ~f:(Set.mem names)
        |> List.dedup_and_sort ~compare:String.compare
      in
      let support_model =
        List.filter_map references ~f:(fun (r : Scan.reference) ->
            Option.map r.via ~f:(fun caller -> (caller, r.value)))
        |> sorted_pairs
      in
      let support_truth =
        generated_calls
          (Ppxlib.Driver.map_structure (Test_utils.Config_key_scan.structure_of sample))
        |> List.filter ~f:(fun (caller, callee) ->
            Set.mem names caller && Set.mem names callee && not (String.equal caller callee))
        |> sorted_pairs
      in
      let show pairs = String.concat ~sep:" " (List.map pairs ~f:(fun (a, b) -> a ^ "->" ^ b)) in
      if (not (List.is_empty external_truth)) && not (List.is_empty support_truth) then
        exercised := label :: !exercised;
      let external_ok = List.equal String.equal external_model external_truth in
      let support_ok = pairs_equal support_model support_truth in
      if not external_ok then
        Stdio.eprintf "%s: consumer\n  scanner:   %s\n  expansion: %s\n" label
          (String.concat ~sep:" " external_model)
          (String.concat ~sep:" " external_truth);
      if not support_ok then
        Stdio.eprintf "%s: support\n  scanner:   %s\n  expansion: %s\n" label (show support_model)
          (show support_truth);
      external_ok && support_ok);
  Verdict.p_all "every corpus fixture exercises both consumer calls and support calls" corpus
    ~f:(fun (label, _, _) -> List.mem !exercised label ~equal:String.equal);
  (* The synthetic controls the corpus exists for: a helper only a sibling deriving needs is kept
     alive by that sibling's liveness, and by nothing else. *)
  let sample =
    "type scope = Local | Global [@@deriving sexp, compare, equal]\n\
     type entry = { scope : scope } [@@deriving sexp, compare, equal]\n"
  in
  let exports = Scan.exports_of_source ~source:"arrayjit/lib/sample.ml" sample in
  let count consumer key =
    Hashtbl.find_exn
      (Scan.counts ~exports
         (Scan.references ~exports
            ~sources:[ ("arrayjit/lib/sample.ml", sample); ("user.ml", consumer) ]))
      ("Sample." ^ key)
  in
  let user = "let a = Sample.compare_entry\nlet b = [%of_sexp: Sample.entry]\n" in
  Verdict.p_all "a helper a live sibling deriving calls is live (gh-ocannl-836)"
    [ "compare_scope"; "scope_of_sexp" ] ~f:(fun key -> count user key > 0);
  Verdict.p_all "a helper whose calling sibling is dead stays dead" [ "equal_scope"; "equal_entry" ]
    ~f:(fun key -> count user key = 0);
  Verdict.p "with no outside reference, support alone keeps nothing alive"
    (count "let unrelated = ()\n" "compare_scope" = 0)
