open Base
open Verdict.Claims
module Scan = Test_scan.Markdown_link_scan

let () =
  let read _ = Some "# Heading\n" in
  let check =
    Scan.check ~read
      ~exists:(fun path -> List.mem [ "docs/target.md"; "docs/image.png" ] path ~equal:String.equal)
      ~from_file:"docs/from.md"
  in
  p "a deliberately broken relative link is rejected"
    (not (List.is_empty (check "[broken](missing.md)")));
  p_empty "relative paths, angle destinations, titles and fragments share resolution"
    ~over:[ "valid links" ]
    (check "[a](./target.md#heading) [b](<target.md> \"title\")");
  p_empty "URLs, reference links, code spans, comments and fences are outside the corpus"
    ~over:[ "excluded forms" ]
    (check
       "[a](https://example.org) [network](//example.org/x) [b][ref] ` [c](missing.md) `\n\
        <!-- [d](missing.md) -->\n\
        ```md\n\
        [e](missing.md)\n\
        ```");
  p "a missing Markdown anchor is rejected" (not (List.is_empty (check "[a](./target.md#missing)")));
  p "a missing non-Markdown target with a line fragment is rejected"
    (not (List.is_empty (check "[source](missing.ml#L1)")));
  p "a missing image is rejected" (not (List.is_empty (check "![alt](missing.png)")));
  p_empty "an existing image destination passes" ~over:[ "image" ] (check "![alt](image.png)");
  p "root-relative destinations fail closed" (not (List.is_empty (check "[root](/docs/target.md)")));
  p "a colon in a fragment cannot hide a broken relative target"
    (not (List.is_empty (check "[broken](./missing.md#topic:detail)")));
  let root_links = [ ("README.md", "[repository](.)"); ("docs/from.md", "[repository](../)") ] in
  p_empty "relative links to the repository root resolve as directories" ~over:root_links
    (List.concat_map root_links ~f:(fun (from_file, text) ->
         Scan.check ~read ~exists:(Scan.path_exists ~files:[ "docs/target.md" ]) ~from_file text));
  p "a path escaping the repository is rejected"
    (not (List.is_empty (check "[a](../../missing.md)")));
  p "unsupported relative destination syntax fails closed"
    (not (List.is_empty (check "[a](target%20name.md)")));
  p_empty "only the exact published-site route is exempt" ~over:[ "published route" ]
    (Scan.check ~read
       ~exists:(fun _ -> false)
       ~from_file:"docs/migration_guide.md" "[site](syntax_extensions.html)");
  p "another HTML destination is checked rather than exempted"
    (not (List.is_empty (check "[site](missing.html)")));
  p_all "every declared document scope is selected"
    [ "README.md"; "CHANGES.md"; "docs/example.md"; "benchmarks/example.md" ]
    ~f:Scan.in_corpus;
  p "archive is outside the document corpus" (not (Scan.in_corpus "archive/old.md"))

let () =
  let check contents link =
    Scan.check ~exists:(fun _ -> true) ~read:(fun _ -> Some contents) ~from_file:"docs/from.md" link
  in
  let valid =
    [
      ("# Same\n# Same\n# Same-1\n# Same", "same-2");
      ("# Same\n# Same-1\n# Same", "same-2");
      ("# With `inline_code` and *emphasis*", "with-inline_code-and-emphasis");
      ("# A — B", "a--b");
      ("# Valid\n```html\n<pre>\n# Fake\n</pre>\n```", "valid");
      ("# Valid\n\n    value\n    ---", "valid");
      ("   ### Heading ###", "heading");
      ("# Heading#", "heading");
      ("# Heading\n```md\n# Heading\n```\n# Heading", "heading-1");
      ("# Heading\n<!--\n# Heading\n-->\n# Heading", "heading-1");
    ]
  in
  p_empty "supported heading forms use GitHub IDs and collision suffixes" ~over:valid
    (List.concat_map valid ~f:(fun (contents, anchor) -> check contents ("[a](#" ^ anchor ^ ")")));
  let unsupported =
    [
      "> # Same";
      "- # Same";
      "1. # Same";
      "1. Item\n\n    # Same";
      "1. Item\n\n    Same\n    ---";
      "> Same\n> ---";
      "Title\n======";
      "Title\n--";
      "# [linked](target.md)";
      "# <em>HTML</em>";
      "<pre>\n# Fake\n</pre>";
      "<div>\n# Fake\n</div>";
      "<script>\n# Fake\n</script>";
      "# A &amp; B";
      "# _emphasis_";
      "# Unicode é";
      "# ` padded `";
      "# Escaped \\*";
    ]
  in
  p_all "unsupported headings refuse even a different valid anchor in the same file" unsupported
    ~f:(fun heading ->
      List.exists
        (check ("# Valid\n\n" ^ heading) "[a](#valid)")
        ~f:(fun (_, message) -> String.is_prefix message ~prefix:"unsupported heading syntax:"));
  p_empty "non-Markdown fragments and empty fragments only require an existing target"
    ~over:[ "source line"; "top" ]
    (check "" "[source](source.ml#L999999) [top](#)");
  p "an unreadable Markdown target refuses fragment validation"
    (not
       (List.is_empty
          (Scan.check
             ~exists:(fun _ -> true)
             ~read:(fun _ -> None)
             ~from_file:"docs/from.md" "[a](./target.md#heading)")))
