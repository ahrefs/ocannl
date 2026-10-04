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
