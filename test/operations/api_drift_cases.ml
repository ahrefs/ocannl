open Base
module Surface = Test_utils.Api_drift

let declarations ?(paths = []) source text = Surface.declarations ~paths ~source text

let changed ?(paths = []) source before after =
  Surface.changes (declarations ~paths source before) (declarations ~paths source after)

let () =
  let inventory =
    [
      "lib/a.ml";
      "lib/a.mli";
      "lib/b.ml";
      "tensor/c.mli";
      "arrayjit/lib/d.ml";
      "bin/e.ml";
      "lib/sub/f.mli";
    ]
  in
  let implicit = Test_utils.Dead_export_scan.implicit_implementations inventory in
  Verdict.p "editorial implicit modules are selected by the dead-export census"
    (List.equal String.equal implicit [ "lib/b.ml"; "arrayjit/lib/d.ml" ]
    && List.equal String.equal (Surface.sources inventory)
         [ "arrayjit/lib/d.ml"; "lib/a.mli"; "lib/b.ml"; "tensor/c.mli" ]);
  let selected_paths =
    [ "arrayjit/lib/impl.cudajit.ml"; "arrayjit/lib/impl.missing.ml"; "arrayjit/lib/impl.mli" ]
  in
  let selected_dune =
    "(library (name backend) (public_name pkg.backend) (modules impl) (libraries (select impl.ml \
     from (cuda -> impl.cudajit.ml) (-> impl.missing.ml))))"
  in
  Verdict.p "select arms use the compiled target interface relationship"
    (List.equal String.equal
       (Surface.sources ~dunes:[ ("arrayjit/lib/dune", selected_dune) ] selected_paths)
       [ "arrayjit/lib/dune"; "arrayjit/lib/impl.mli" ]
    && List.equal String.equal
         (Surface.sources
            ~dunes:[ ("arrayjit/lib/dune", selected_dune) ]
            (List.take selected_paths 2))
         [ "arrayjit/lib/dune"; "arrayjit/lib/impl.cudajit.ml"; "arrayjit/lib/impl.missing.ml" ]);
  let generator_dune =
    "(ocamllex lexer private_lexer) (menhir (modules parser)) (library (name parserlib) \
     (public_name pkg.parserlib) (modules lexer parser)) (executable (name private) (modules \
     private_lexer))"
  in
  let generator_paths = [ "tensor/lexer.mll"; "tensor/parser.mly"; "tensor/private_lexer.mll" ] in
  Verdict.p "public generator inputs follow Dune module ownership and explicit interfaces"
    (List.equal String.equal
       (Surface.sources ~dunes:[ ("tensor/dune", generator_dune) ] generator_paths)
       [ "tensor/dune"; "tensor/lexer.mll"; "tensor/parser.mly" ]
    && List.equal String.equal
         (Surface.sources
            ~dunes:[ ("tensor/dune", generator_dune) ]
            ("tensor/lexer.mli" :: generator_paths))
         [ "tensor/dune"; "tensor/lexer.mli"; "tensor/parser.mly" ]);
  Verdict.p "parser tokens and lexer body changes produce generated-interface review entries"
    (List.length (changed "tensor/parser.mly" "%token OLD\n%%" "%token NEW\n%%") = 1
    && List.length
         (changed "tensor/lexer.mll" "{let v = 1}\nrule token = parse | eof { () }"
            "{let v = true}\nrule token = parse | eof { () }")
       = 1);
  let private_generator_dune =
    "(ocamllex lexer) (menhir (modules parser)) (library (name parserlib) (public_name \
     pkg.parserlib) (modules lexer parser) (private_modules Lexer))"
  in
  (* A public module can [include] a private one, so privacy does not remove the evidence. *)
  Verdict.p "private generated modules of a public library remain conservative entries"
    (List.equal String.equal
       (Surface.sources ~dunes:[ ("tensor/dune", private_generator_dune) ] generator_paths)
       [ "tensor/dune"; "tensor/lexer.mll"; "tensor/parser.mly" ]
    && List.equal String.equal
         (Surface.sources
            ~dunes:
              [
                ( "tensor/dune",
                  "(ocamllex lexer) (menhir (modules parser)) (library (name parserlib) \
                   (public_name pkg.parserlib) (modules lexer parser) (private_modules (:standard \
                   \\ parser)))" );
              ]
            generator_paths)
         [ "tensor/dune"; "tensor/lexer.mll"; "tensor/parser.mly" ]);
  Verdict.p "Dune empty-interface policy changes produce publication-input review entries"
    (List.length
       (changed "lib/dune" "(library (name lib) (public_name pkg.lib) (modules a))"
          "(library (name lib) (public_name pkg.lib) (modules a) \
           (empty_module_interface_if_absent))")
    = 1);
  let public_configuration field =
    "(library (name backend) (public_name pkg.backend) (modules a) " ^ field ^ ")"
  in
  Verdict.p_all "availability preprocessing and driver inputs remain literal review evidence"
    [
      ("", "(optional)");
      ("(enabled_if true)", "(enabled_if false)");
      ("(preprocess (pps ppx_sexp_conv))", "(preprocess (pps ppx_compare))");
      ("(preprocessor_deps earlier)", "(preprocessor_deps later)");
      ("(kind normal)", "(kind ppx_rewriter)");
      ("(flags :standard)", "(flags :standard -opaque)");
      ("(modes byte)", "(modes best)");
      ("(foreign_stubs (language c) (names earlier))", "(foreign_stubs (language c) (names later))");
      ("(c_library_flags -lpthread)", "(c_library_flags -lother)");
    ]
    ~f:(fun (before, after) ->
      match changed "lib/dune" (public_configuration before) (public_configuration after) with
      | [ (Some previous, Some entry) ] -> not (String.equal previous.text entry.text)
      | _ -> false);
  Verdict.p_empty "independent public configuration field reordering stays quiet"
    ~over:
      (declarations "lib/dune" (public_configuration "(optional) (preprocess (pps ppx_compare))"))
    (changed "lib/dune"
       (public_configuration "(optional) (preprocess (pps ppx_compare))")
       "(library (preprocess (pps ppx_compare)) (public_name pkg.backend) (optional) (modules a) \
        (name backend))");
  let owner_before =
    "(library (name first) (public_name pkg.first) (modules a)) (library (name second) \
     (public_name pkg.second) (modules b))"
  in
  let owner_after =
    "(library (name first) (public_name pkg.first) (modules b)) (library (name second) \
     (public_name pkg.second) (modules a))"
  in
  let owner_reordered =
    "(library (name second) (public_name pkg.second) (modules b)) (library (name first) \
     (public_name pkg.first) (modules a))"
  in
  Verdict.p_empty "independent Dune stanza reordering does not create API drift"
    ~over:(declarations "lib/dune" owner_before)
    (changed "lib/dune" owner_before owner_reordered);
  Verdict.p "Dune module ownership moves produce conservative publication-input entries"
    (List.length (changed "lib/dune" owner_before owner_after) = 2
    && List.length (changed "lib/dune" owner_before "(library (name first) (modules a))") = 2);
  let parser_config flags =
    "(menhir (modules parser) (flags " ^ flags
    ^ ")) (library (name parserlib) (public_name pkg.parserlib) (modules parser))"
  in
  Verdict.p "owning generator configuration changes produce manual-review entries"
    (List.length
       (changed ~paths:[ "tensor/parser.mly" ] "tensor/dune" (parser_config "--table")
          (parser_config "--code"))
     = 1
    && List.length
         (changed ~paths:[ "tensor/parser.mly" ] "tensor/dune"
            "(menhir (modules parser) (merge_into earlier)) (library (name lib) (public_name \
             pkg.lib))"
            "(menhir (modules parser) (merge_into later)) (library (name lib) (public_name \
             pkg.lib))")
       = 2);
  Verdict.p_empty "independent generator configuration field reordering stays quiet"
    ~over:(declarations ~paths:[ "tensor/parser.mly" ] "tensor/dune" (parser_config "--table"))
    (changed ~paths:[ "tensor/parser.mly" ] "tensor/dune" (parser_config "--table")
       "(menhir (flags --table) (modules parser)) (library (name parserlib) (public_name \
        pkg.parserlib) (modules parser))");
  let select_config condition =
    "(library (name backend) (public_name pkg.backend) (modules impl) (libraries (select impl.ml \
     from (" ^ condition ^ " -> impl.cudajit.ml) (-> impl.missing.ml))))"
  in
  Verdict.p "select configuration is visible when its target has an implicit interface"
    (List.length
       (changed ~paths:(List.take selected_paths 2) "arrayjit/lib/dune" (select_config "cuda")
          (select_config "hip"))
    = 1);
  Verdict.p_empty "select configuration with an explicit target interface stays quiet"
    ~over:(declarations ~paths:selected_paths "arrayjit/lib/dune" (select_config "cuda"))
    (changed ~paths:selected_paths "arrayjit/lib/dune" (select_config "cuda") (select_config "hip"));
  let private_config flag =
    "(menhir (modules parser) (flags " ^ flag
    ^ ")) (library (name lib) (public_name pkg.lib) (modules parser) (private_modules parser))"
  in
  Verdict.p "private generator configuration edits remain manual-review entries"
    (List.length
       (changed ~paths:[ "tensor/parser.mly" ] "tensor/dune" (private_config "--table")
          (private_config "--code"))
    = 1);
  let publication_before =
    declarations "lib/dune"
      "(library (name public) (public_name pkg.public) (modules a) (libraries earlier) (synopsis \
       earlier)) (library (name private) (modules x) (preprocess (pps earlier)))"
  in
  let publication_after =
    declarations "lib/dune"
      "; changed prose\n\
       (library (name public) (public_name pkg.public) (modules a) (libraries later) (synopsis \
       later)) (library (name private) (modules y) (preprocess (pps later)) (optional))"
  in
  Verdict.p_empty "private ownership dependency and prose edits do not change publication entries"
    ~over:publication_before
    (Surface.changes publication_before publication_after);
  let dune_refusal dune =
    match Surface.sources ~dunes:[ ("lib/dune", dune) ] [ "lib/a.ml" ] with
    | _ -> None
    | exception Failure message -> Some message
  in
  Verdict.p_all "module lists and configuration read from other files refuse"
    [
      "(library (name lib) (public_name pkg.lib) (modules (:include modules.sexp)))";
      "(library (name lib) (public_name pkg.lib) (flags (:include flags.sexp)))";
      "(executable (name main) (modules %{read-lines:modules.txt}))";
      "(library (name lib) (public_name pkg.lib) (flags %{read:flags.txt}))";
      "(include dune.inc)";
      "(dynamic_include generated.inc)";
    ] ~f:(fun dune ->
      Option.value_map (dune_refusal dune) ~default:false ~f:(fun message ->
          String.is_substring message ~substring:"lib/dune: "
          && String.is_substring message ~substring:"gh-ocannl-1201"));
  Verdict.p
    "a bare include atom, a variable named like a read pform and an include outside module owners \
     and generators pass"
    (Option.is_none
       (dune_refusal
          "(library (name lib) (public_name pkg.lib) (preprocess (action (run ./pp.exe :include \
           %{reader}))) (preprocessor_deps (:reader config))) (rule (deps (:include deps.sexp)) \
           (action (progn)))"));
  Verdict.p "a selected interface target refuses by name"
    (Option.value_map
       (dune_refusal
          "(library (name lib) (public_name pkg.lib) (libraries (select impl.mli from (cuda -> \
           impl.cudajit.mli) (-> impl.missing.mli))))")
       ~default:false
       ~f:(String.is_substring ~substring:"unsupported select target impl.mli in lib/dune"));
  let re_export libraries =
    "(library (name lib) (public_name pkg.lib) (modules a) (libraries " ^ libraries ^ "))"
  in
  Verdict.p_all "re-exported dependencies remain publication-input evidence at any depth"
    [ ("base", "base (re_export stdio)"); ("(re_export stdio) base", "(re_export ppxlib) base") ]
    ~f:(fun (before, after) ->
      match changed "lib/dune" (re_export before) (re_export after) with
      | [ (Some _, Some entry) ] -> String.is_substring entry.text ~substring:"(re_export"
      | _ -> false);
  Verdict.p_empty "ordinary dependency edits beside a re-export stay quiet"
    ~over:(declarations "lib/dune" (re_export "(re_export stdio) base"))
    (changed "lib/dune" (re_export "(re_export stdio) base") (re_export "(re_export stdio) unix"));
  let private_paths =
    [ "lib/a.ml"; "lib/b.ml"; "lib/b.mli"; "lib/c.ml"; "lib/c.mli"; "tensor/d.ml" ]
  in
  Verdict.p "ordinary private modules remain in the census inventory"
    (List.equal String.equal
       (Surface.sources
          ~dunes:
            [
              ( "lib/dune",
                "(library (name lib) (public_name pkg.lib) (modules a b c) (private_modules a b))"
              );
            ]
          private_paths)
       [ "lib/a.ml"; "lib/b.mli"; "lib/c.mli"; "lib/dune"; "tensor/d.ml" ]
    && List.equal String.equal
         (Surface.sources
            ~dunes:
              [
                ( "lib/dune",
                  "(library (name lib) (public_name pkg.lib) (private_modules (:standard \\ c)))" );
              ]
            private_paths)
         [ "lib/a.ml"; "lib/b.mli"; "lib/c.mli"; "lib/dune"; "tensor/d.ml" ]
    && List.equal String.equal (Surface.sources private_paths)
         [ "lib/a.ml"; "lib/b.mli"; "lib/c.mli"; "tensor/d.ml" ]);
  Verdict.p "a multiline value signature change is visible"
    (List.length (changed "lib/a.mli" "val run :\n int ->\n int" "val run :\n int ->\n string") = 1);
  Verdict.p "record fields and constructors retain their symbol spellings"
    (match
       changed "lib/a.mli" "type t = { old_scope : bool }\ntype kind = Old"
         "type t = { scopes : int list }\ntype kind = New of int"
     with
    | [ (_, Some kind); (_, Some record) ] ->
        String.is_substring kind.text ~substring:"New of int"
        && String.is_substring record.text ~substring:"scopes: int list"
    | _ -> false);
  let before = declarations "lib/a.mli" "(** Old prose *)\nval run : int -> int" in
  let after = declarations "lib/a.mli" "(** New prose *)\nval run:\nint -> int" in
  Verdict.p_empty "documentation and whitespace changes do not count as declarations" ~over:before
    (Surface.changes before after);
  let floating =
    declarations "lib/a.mli"
      "[@@@ocaml.text \"old floating docs\"]\n\
       module M : sig [@@@ocaml.doc \"old nested docs\"] val x : int end"
  in
  let floating_after =
    declarations "lib/a.mli"
      "[@@@ocaml.text \"new floating docs\"]\n\
       module M : sig [@@@ocaml.doc \"new nested docs\"] val x : int end"
  in
  Verdict.p_empty "floating and nested documentation attributes do not count as declarations"
    ~over:floating
    (Surface.changes floating floating_after);
  let floating_ml =
    declarations "lib/a.ml"
      "[@@@ocaml.text \"old\"]\nmodule M = struct [@@@ocaml.doc \"old\"] let x = 1 end"
  in
  let floating_ml_after =
    declarations "lib/a.ml"
      "[@@@ocaml.text \"new\"]\nmodule M = struct [@@@ocaml.doc \"new\"] let x = 1 end"
  in
  Verdict.p_empty "floating implementation documentation is discarded recursively" ~over:floating_ml
    (Surface.changes floating_ml floating_ml_after);
  Verdict.p "nested signatures and includes stay in the checklist"
    (List.length
       (changed "lib/a.mli" "module M : sig val x : int end\ninclude S"
          "module M : sig val x : string end\ninclude T")
    = 2);
  Verdict.p "implicit lets externals and types including deriving changes are visible"
    (List.length
       (changed "lib/a.ml"
          "let value = 1\nexternal call : int -> int = \"call\"\ntype t = A [@@deriving sexp]"
          "let value = \"s\"\n\
           external call : string -> int = \"call\"\n\
           type t = B [@@deriving compare]")
    = 3);
  Verdict.p "non-documentation attribute payload changes stay visible"
    (List.length
       (changed "lib/a.ml" "type t = A [@@deriving sexp]" "type t = A [@@deriving compare]")
     = 1
    && List.length
         (changed "lib/a.mli" "type t = A [@@deriving sexp]" "type t = A [@@deriving compare]")
       = 1
    && List.length
         (changed "lib/a.ml" "[@@@publish earlier] let x = 1" "[@@@publish later] let x = 1")
       = 1);
  let named_before =
    declarations "lib/a.ml"
      "let exported = 1\n\
       let () = print_endline \"old\"\n\
       let _ = 2;;\n\
       print_endline \"old bare eval\";;\n\
       module M = struct let visible = 1 let () = print_endline \"old nested\" end\n\
       module _ = struct let x = earlier let () = earlier end\n\
       (** documented *)\n\
       let () = earlier"
  in
  let named_after =
    declarations "lib/a.ml"
      "let exported = 1\n\
       let () = print_endline \"new\"\n\
       let _ = 3;;\n\
       print_endline \"new bare eval\";;\n\
       module M = struct let visible = 1 let () = print_endline \"new nested\" end\n\
       module _ = Make (struct let x = later let () = later end)\n\
       (** documented *)\n\
       let () = later"
  in
  Verdict.p_empty
    "unnamed initializers anonymous modules and bare evaluations do not count as exported \
     declarations"
    ~over:named_before
    (Surface.changes named_before named_after);
  Verdict.p "named pattern aliases remain exported declarations"
    (List.length (changed "lib/a.ml" "let ([] as exported) = []" "let ([] as exported) = [1]") = 1
    && List.equal String.equal
         (Test_utils.Dead_export_scan.exports_of_source ~source:"lib/a.ml"
            "let ([] as exported) = []"
         |> List.map ~f:Test_utils.Dead_export_scan.export_key)
         [ "A.exported" ]);
  Verdict.p "named module unpack bindings remain exported declarations"
    (match
       changed "lib/a.ml" "let (module M : S) = package" "let (module M : S) = other_package"
     with
    | [ (Some before, Some after) ] ->
        String.equal before.name "let M[0]" && String.equal after.name "let M[0]"
    | _ -> false);
  Verdict.p "extension inputs stay visible even when their payload binds no source name"
    (List.length (changed "lib/a.ml" "[%%publish earlier]" "[%%publish later]") = 1
    && List.length (changed "lib/a.ml" "[%%publish let () = earlier]" "[%%publish let () = later]")
       = 1);
  let refusal source text =
    match declarations source text with _ -> None | exception Failure message -> Some message
  in
  Verdict.p_all "non-documentation attributes on anonymous items refuse with their location"
    [
      ("let () = setup () [@@publish earlier]", "lib/a.ml:1:", "value binding");
      ("let x = 1\nlet[@publish] _ = setup ()", "lib/a.ml:2:", "value binding");
      ("let exported = 1 and () = setup () [@@publish]", "lib/a.ml:1:", "value binding");
      ("let x = 1;;\nsetup () [@@publish earlier]", "lib/a.ml:2:", "evaluation");
      ("module _ = struct end [@@publish]", "lib/a.ml:1:", "module binding");
      ( "module M = struct\n let () = setup () [@@warning \"-8\"] end",
        "lib/a.ml:2:",
        "value binding" );
      ("module _ = struct\n let () = () [@@warning \"-8\"] end", "lib/a.ml:2:", "value binding");
      ("let x = 1\nlet (_ [@publish]) = setup ()", "lib/a.ml:2:", "value binding");
      ("let x = 1\nlet ((() [@publish]), _) = setup ()", "lib/a.ml:2:", "value binding");
      ( "let () =\n let module M = struct let () = () [@@warning \"-8\"] end in ()",
        "lib/a.ml:2:",
        "value binding" );
      ("module _ = struct\n module _ = struct end [@@publish] end", "lib/a.ml:2:", "module binding");
      ( "let exported = 1 and () =\n let module M = struct let _ = () [@@x] end in ()",
        "lib/a.ml:2:",
        "value binding" );
    ]
    ~f:(fun (text, location, kind) ->
      match refusal "lib/a.ml" text with
      | Some message ->
          String.is_prefix message ~prefix:location
          && String.is_substring message ~substring:("anonymous " ^ kind)
          && String.is_substring message ~substring:"gh-ocannl-1201"
      | None -> false);
  Verdict.p_none "named items and extension payloads keep their attributes without refusal"
    [
      "let exported = setup () [@@publish]";
      "module M = struct end [@@publish]";
      "[%%publish let () = setup () [@@publish]]";
      "module _ = struct [%%publish let () = setup () [@@publish]] end";
      "let () = (setup () [@inline])";
      "type t = A [@@deriving sexp]";
    ] ~f:(fun text -> Option.is_some (refusal "lib/a.ml" text));
  let mixed_before = declarations "lib/a.ml" "let exported = 1 and () = earlier" in
  let mixed_after = declarations "lib/a.ml" "let exported = 1 and () = later" in
  Verdict.p_empty "anonymous bindings inside mixed let groups do not create drift"
    ~over:mixed_before
    (Surface.changes mixed_before mixed_after);
  Verdict.p "pattern extension inputs remain visible without an ordinary binder"
    (List.length
       (changed "lib/a.ml" "let [%publish earlier] = package" "let [%publish later] = package")
     = 1
    && List.length
         (changed "lib/a.ml" "let exported = 1 and [%publish earlier] = package"
            "let exported = 1 and [%publish later] = package")
       = 1);
  Verdict.p "declaration movement across opens is visible in signatures and implementations"
    (List.length (changed "lib/a.mli" "open B open A val x : t" "open B val x : t open A") = 2
    && List.length
         (changed "lib/a.ml" "open B open A let x = make ()" "open B let x = make () open A")
       = 2);
  Verdict.p "adding a declaration does not report every following unchanged declaration"
    (List.length
       (changed "lib/a.mli" "val x : int val y : int" "val fresh : bool val x : int val y : int")
    = 1);
  Verdict.p "declaration removals and additions are distinct"
    (match changed "lib/a.mli" "val old : int" "val fresh : int" with
    | [ (None, Some _); (Some _, None) ] -> true
    | _ -> false);
  let refused =
    try
      ignore (declarations "lib/bad.mli" "val" : Surface.declaration list);
      false
    with _ -> true
  in
  Verdict.p "invalid OCaml refuses instead of reporting an empty inventory" refused;
  let text lines = String.concat ~sep:"\n" lines in
  let decl line lines = { Surface.name = "let f[0]"; line; text = text lines } in
  let body = List.init 20 ~f:(fun i -> Printf.sprintf "l%d" (i + 1)) in
  let edit i replacement = List.mapi body ~f:(fun j l -> if j = i then replacement else l) in
  Verdict.p "compact rendering keeps both headers, the changed lines, context and omitted counts"
    (List.equal String.equal
       (Surface.render ~context:1 (Some (decl 3 body), Some (decl 4 (edit 9 "L10"))))
       [
         "- let f[0] (line 3)";
         "+ let f[0] (line 4)";
         "~ 8 unchanged lines";
         "  l9";
         "- l10";
         "+ L10";
         "  l11";
         "~ 9 unchanged lines";
       ]);
  Verdict.p "distant edits are separate hunks and near ones share context"
    (List.equal String.equal
       (Surface.render ~context:1
          ( Some (decl 1 body),
            Some (decl 1 (edit 1 "L2" |> List.mapi ~f:(fun j l -> if j = 17 then "L18" else l))) ))
       [
         "- let f[0] (line 1)";
         "+ let f[0] (line 1)";
         "  l1";
         "- l2";
         "+ L2";
         "  l3";
         "~ 13 unchanged lines";
         "  l17";
         "- l18";
         "+ L18";
         "  l19";
         "~ 1 unchanged line";
       ]
    && List.equal String.equal
         (Surface.render ~context:2
            ( Some (decl 1 body),
              Some (decl 1 (edit 5 "L6" |> List.mapi ~f:(fun j l -> if j = 9 then "L10" else l))) ))
         [
           "- let f[0] (line 1)";
           "+ let f[0] (line 1)";
           "~ 3 unchanged lines";
           "  l4";
           "  l5";
           "- l6";
           "+ L6";
           "  l7";
           "  l8";
           "  l9";
           "- l10";
           "+ L10";
           "  l11";
           "  l12";
           "~ 8 unchanged lines";
         ]);
  Verdict.p "full rendering prints both sides; one-sided entries print in full when compact"
    (List.length (Surface.render (Some (decl 3 body), Some (decl 4 (edit 9 "L10")))) = 42
    && List.length (Surface.render ~context:0 (None, Some (decl 4 body))) = 21
    && List.length (Surface.render ~context:0 (Some (decl 4 body), None)) = 21);
  Verdict.p "a moved but unchanged entry says only its position changed"
    (List.equal String.equal
       (Surface.render ~context:3 (Some (decl 1 body), Some (decl 9 body)))
       [
         "- let f[0] (line 1)";
         "+ let f[0] (line 9)";
         "~ 20 unchanged lines; only the position among surviving entries changed";
       ]);
  let reconstructs before after =
    let edits = Surface.line_edits before after in
    List.equal String.equal before
      (List.filter_map edits ~f:(function Surface.Same l | Removed l -> Some l | Added _ -> None))
    && List.equal String.equal after
         (List.filter_map edits ~f:(function
           | Surface.Same l | Added l -> Some l
           | Removed _ -> None))
  in
  let wide n tag = List.init n ~f:(fun i -> Printf.sprintf "%s%d" tag i) in
  Verdict.p_all "line edit scripts reconstruct both sides, beyond the comparison bound too"
    [
      (body, edit 9 "L10");
      ([], body);
      (body, []);
      (body, List.rev body);
      (("x" :: wide 1500 "a") @ [ "y" ], ("x" :: wide 1500 "b") @ [ "y" ]);
    ]
    ~f:(fun (before, after) -> reconstructs before after)
