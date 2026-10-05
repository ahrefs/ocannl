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
    (refuses "- Doc-check `api`: `let x = 1`" && refuses "- Doc-check `api`: missing code.")
