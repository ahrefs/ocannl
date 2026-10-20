open Base
open Verdict.Claims
module E = Test_utils.Doc_examples_scan

let parse = E.parse ~path:"fixture.md"

let refuses source =
  try
    ignore (parse source : E.block list);
    false
  with Failure _ -> true

let () =
  let blocks =
    parse "```ocaml doc-check=api\nlet x = 1\n```\n~~~ocaml doc-check=api\nlet y = x\n~~~\n"
  in
  p "selected fences preserve order, scope and source locations"
    (String.is_substring (E.render blocks) ~substring:"# 2 \"fixture.md\"\nlet x = 1"
    && String.is_substring (E.render blocks) ~substring:"let y = x");
  let rendered = E.render blocks in
  p "shared groups preserve source order"
    (Option.value_exn (String.substr_index rendered ~pattern:"let x = 1")
    < Option.value_exn (String.substr_index rendered ~pattern:"let y = x"));
  let other = E.parse ~path:"other.md" "```ocaml doc-check=api\nlet x = 2\n```" in
  p "different documents have independent scopes even for the same group name"
    (List.count
       (String.split_lines (E.render (blocks @ other)))
       ~f:(fun line -> String.is_prefix line ~prefix:"module Example_")
    = 2);
  p "coverage counts every selected block rather than deduplicating groups"
    (List.length (String.split_lines (E.coverage blocks)) = List.length blocks);
  p "horizontal whitespace in fence metadata keeps the annotation active"
    (List.length (E.selected (parse "```ocaml\tdoc-check=api\nlet x = 1\n```")) = 1);
  p "an annotation on an unsupported language fails loudly"
    (refuses "```ocmal doc-check=api\nlet x = 1\n```"
    && refuses "> ```ocaml\tdoc-check=api\nlet x = 1\n> ```");
  p "agent-note code spans are extracted"
    (List.length (E.selected (parse "- Doc-check `note`: `let x = 1`.")) = 1);
  let unselected =
    parse "```ocaml doc-skip historical API\nremoved ()\n```\n```ocaml\nalso_removed ()\n```"
  in
  p_empty "historical and unselected blocks stay outside compilation" ~over:unselected
    (E.selected unselected);
  let non_ocaml = "````text\n```ocaml doc-check=fake\nremoved ()\n```\n````" in
  p_empty "non-OCaml fences do not expose nested fake annotations" ~over:[ non_ocaml ]
    (E.selected (parse non_ocaml));
  let empty_refused =
    try
      E.require_selected unselected;
      false
    with Failure message -> String.equal message "no selected documentation examples"
  in
  p "an empty compilation selection is refused" empty_refused;
  p "malformed annotations fail loudly"
    (refuses "```ocaml doc-chek=typo\nlet x = 1\n```"
    && refuses "```ocaml doc-skip\nlet x = 1\n```"
    && refuses "```ocaml doc-check=Bad\nlet x = 1\n```");
  p "unsupported annotated indentation and unclosed fences fail loudly"
    (refuses "    ```ocaml doc-check=hidden\nlet x = 1\n    ```"
    && refuses "\t```ocaml doc-check=hidden\nlet x = 1\n```"
    && refuses "```ocaml doc-check=api\nlet x = 1");
  p "malformed inline selection cannot silently disappear"
    (refuses "- Doc-check `api`: `let x = 1`" && refuses "- Doc-check `api`: missing code.");
  (* Provenance (gh-ocannl-1265): excerpts held to the repository file they name, here the real
     [lib/nn_blocks.ml] the shapes slides quote. *)
  let library = Stdio.In_channel.read_all "../../lib/nn_blocks.ml" in
  let read = function "lib/nn_blocks.ml" -> Some library | _ -> None in
  let failures source = E.check_provenance ~read (parse source) in
  let layer_norm_head =
    List.find_exn (String.split_lines library) ~f:(String.is_prefix ~prefix:"let%op layer_norm")
  in
  (* The same words, one per quoted line and indented: different whitespace only. *)
  let respaced =
    String.concat
      (List.map (String.split ~on:' ' layer_norm_head) ~f:(fun word -> ">   " ^ word ^ "\n"))
  in
  p_empty "an excerpt matching its source up to whitespace is accepted" ~over:[ layer_norm_head ]
    (failures ("> ```ocaml doc-excerpt=lib/nn_blocks.ml\n" ^ respaced ^ "> ```\n"));
  let drifted_capture =
    "> ```ocaml doc-excerpt=lib/nn_blocks.ml\n\
     > let%op layer_norm ~label ?(epsilon = 1e-5) () x =\n\
     >   let mean = x ++ \" ... | ..d.. => ... | 0 \" [ \"d\" ] in\n\
     >   let centered = (x - mean) /. dim d in\n\
     > ```\n"
  in
  p
    "the drifted capture slide, reintroduced under its annotation, is refused at its first drifted \
     line"
    (match failures drifted_capture with
    | [ failure ] -> String.is_substring failure ~substring:"has no line \"let mean = x ++"
    | _ -> false);
  let drifted_examples =
    "{pause down #pattern-examples .example title=\"Real Examples from nn_blocks.ml\"}\n\
     ```ocaml\n\
     let%op layer_norm () x =\n\
    \  let mean = x ++ \"... | ..d.. => ... | 0\" [\"d\"] in\n\
    \  let normalized = (x - mean) /. dim d in\n\
    \  ({ gamma = 1. } *. normalized) + { beta = 0. }\n\
     ```\n"
  in
  p "the drifted examples slide is refused without a provenance declaration"
    (refuses drifted_examples);
  p "the drifted examples slide is refused as an excerpt"
    (List.length
       (failures
          (String.substr_replace_first drifted_examples ~pattern:"```ocaml\n"
             ~with_:"```ocaml doc-excerpt=lib/nn_blocks.ml\n"))
    = 1);
  p "an adaptation naming no file is refused"
    (List.length (failures "```ocaml doc-adapted=lib/no_such_file.ml\nlet x = 1\n```\n") = 1);
  p_empty "an adaptation naming an existing file is accepted" ~over:[ "lib/nn_blocks.ml" ]
    (failures "```ocaml doc-adapted=lib/nn_blocks.ml\nlet x = 1\n```\n");
  p "an empty excerpt is refused"
    (List.length (failures "```ocaml doc-excerpt=lib/nn_blocks.ml\n```\n") = 1);
  p "a caption citing an OCaml source directly above a fence requires a declaration"
    (refuses "**OCANNL** (from train.ml):\n```ocaml\nlet x = 1\n```\n"
    && refuses "> Code from [`nn_blocks.ml`](../lib/nn_blocks.ml):\n> ```text\nx\n> ```\n"
    && refuses "Code from **nn_blocks.ml**:\n```ocaml\nlet x = 1\n```\n");
  p "a declaration must name the file its caption cites"
    (refuses "**OCANNL** (from train.ml):\n```ocaml doc-adapted=lib/nn_blocks.ml\nlet x = 1\n```\n"
    && not
         (refuses "**OCANNL** (from train.ml):\n```ocaml doc-adapted=lib/train.ml\nlet x = 1\n```\n")
    );
  p "a caption apart from the fence, or citing another language, requires none"
    ((not (refuses "the pattern from `nn_blocks.ml`:\n\n```ocaml\nlet x = 1\n```\n"))
    && not (refuses "**PyTorch** (from optim/sgd.py):\n```python\nx = 1\n```\n"));
  p "a provenance declaration is read on any fence language, next to the language word only"
    (Poly.equal
       (List.map (parse "```bash doc-adapted=lib/nn_blocks.ml\necho\n```\n") ~f:(fun b ->
            b.E.provenance))
       [ Some (E.Adapted "lib/nn_blocks.ml") ]
    && refuses "```ocaml doc-check=api doc-excerpt=lib/nn_blocks.ml\nlet x = 1\n```\n"
    && refuses "```ocaml doc-excerpt=../lib/nn_blocks.ml\nlet x = 1\n```\n");
  p "a quoted fence is read through its quote markers, and checked when annotated"
    (Poly.equal
       (List.map (parse "> ```ocaml doc-check=api\n> let x = 1\n>\n> let y = x\n> ```\n")
          ~f:(fun b -> b.E.code))
       [ "let x = 1\n\nlet y = x" ]);
  p "a quoted annotated fence the quote abandons, or a nested quote, fails loudly"
    (refuses "> ```ocaml doc-excerpt=lib/nn_blocks.ml\n> let x = 1\nlet y = 2\n"
    && refuses "> > ```ocaml doc-check=api\n> > let x = 1\n> > ```\n")
