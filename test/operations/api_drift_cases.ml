open Base
module Surface = Test_utils.Api_drift

let declarations source text = Surface.declarations ~source text

let changed source before after =
  Surface.changes (declarations source before) (declarations source after)

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
       [ "arrayjit/lib/impl.mli" ]
    && List.equal String.equal
         (Surface.sources
            ~dunes:[ ("arrayjit/lib/dune", selected_dune) ]
            (List.take selected_paths 2))
         [ "arrayjit/lib/impl.cudajit.ml"; "arrayjit/lib/impl.missing.ml" ]);
  let generator_dune =
    "(ocamllex lexer private_lexer) (menhir (modules parser)) (library (name parserlib) \
     (public_name pkg.parserlib) (modules lexer parser)) (executable (name private) (modules \
     private_lexer))"
  in
  let generator_paths = [ "tensor/lexer.mll"; "tensor/parser.mly"; "tensor/private_lexer.mll" ] in
  Verdict.p "public generator inputs follow Dune module ownership and explicit interfaces"
    (List.equal String.equal
       (Surface.sources ~dunes:[ ("tensor/dune", generator_dune) ] generator_paths)
       [ "tensor/lexer.mll"; "tensor/parser.mly" ]
    && List.equal String.equal
         (Surface.sources
            ~dunes:[ ("tensor/dune", generator_dune) ]
            ("tensor/lexer.mli" :: generator_paths))
         [ "tensor/lexer.mli"; "tensor/parser.mly" ]);
  Verdict.p "parser tokens and lexer body changes produce generated-interface review entries"
    (List.length (changed "tensor/parser.mly" "%token OLD\n%%" "%token NEW\n%%") = 1
    && List.length
         (changed "tensor/lexer.mll" "{let v = 1}\nrule token = parse | eof { () }"
            "{let v = true}\nrule token = parse | eof { () }")
       = 1);
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
  let named_before =
    declarations "lib/a.ml"
      "let exported = 1\n\
       let () = print_endline \"old\"\n\
       let _ = 2;;\n\
       print_endline \"old bare eval\";;\n\
       module M = struct let visible = 1 let () = print_endline \"old nested\" end"
  in
  let named_after =
    declarations "lib/a.ml"
      "let exported = 1\n\
       let () = print_endline \"new\"\n\
       let _ = 3;;\n\
       print_endline \"new bare eval\";;\n\
       module M = struct let visible = 1 let () = print_endline \"new nested\" end"
  in
  Verdict.p_empty "unnamed initializers and bare evaluations do not count as exported declarations"
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
  Verdict.p "invalid OCaml refuses instead of reporting an empty inventory" refused
