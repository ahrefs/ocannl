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
  Verdict.p "top-level lets, patterns, extensions, and externals are exports"
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
         "Sample.named_of_sexp";
         "Sample.outer";
         "Sample.pair";
         "Sample.plain";
         "Sample.poly_of_sexp";
         "Sample.primitive";
         "Sample.public_pair";
         "Sample.x";
       ]);
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
  let included = refs [ ("included.ml", "include Sample\n") ] in
  Verdict.p "include counts as a reference to every re-exported value"
    (List.length included = List.length fixture_exports);
  let counts = Scan.counts ~exports:fixture_exports direct in
  Verdict.p "a synthetic zero-reference export is detected"
    (Hashtbl.find_exn counts "Sample.primitive" = 0);
  Verdict.p "a synthetic zero-reference derived export is detected"
    (Hashtbl.find_exn counts "Sample.equal_named" = 0);
  let extension_counts = Scan.counts ~exports:fixture_exports extensions in
  Verdict.p "an export referenced only through an extension point is not reported dead"
    (Hashtbl.find_exn extension_counts "Sample.equal_named" > 0)
