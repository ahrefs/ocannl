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
