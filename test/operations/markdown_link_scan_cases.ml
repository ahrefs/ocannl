open Base
open Verdict.Claims
module Scan = Test_scan.Markdown_link_scan

let () =
  let check = Scan.check ~exists:(String.equal "docs/target.md") ~from_file:"docs/from.md" in
  p "a deliberately broken relative link is rejected"
    (not (List.is_empty (check "[broken](missing.md)")));
  p_empty "relative paths, angle destinations, titles and fragments share resolution"
    ~over:[ "valid links" ]
    (check "[a](./target.md#heading) [b](<target.md> \"title\")");
  p_empty "URLs, reference links, images, code spans, comments and fences are outside the corpus"
    ~over:[ "excluded forms" ]
    (check
       "[a](https://example.org) [network](//example.org/x) [b][ref] ![image](missing.png) ` \
        [c](missing.md) `\n\
        <!-- [d](missing.md) -->\n\
        ```md\n\
        [e](missing.md)\n\
        ```");
  p "root-relative destinations fail closed" (not (List.is_empty (check "[root](/docs/target.md)")));
  p "a colon in a fragment cannot hide a broken relative target"
    (not (List.is_empty (check "[broken](./missing.md#topic:detail)")));
  let root_links = [ ("README.md", "[repository](.)"); ("docs/from.md", "[repository](../)") ] in
  p_empty "relative links to the repository root resolve as directories" ~over:root_links
    (List.concat_map root_links ~f:(fun (from_file, text) ->
         Scan.check ~exists:(Scan.path_exists ~files:[ "docs/target.md" ]) ~from_file text));
  p "a path escaping the repository is rejected"
    (not (List.is_empty (check "[a](../../missing.md)")));
  p "unsupported relative destination syntax fails closed"
    (not (List.is_empty (check "[a](target%20name.md)")));
  p_empty "only the exact published-site route is exempt" ~over:[ "published route" ]
    (Scan.check
       ~exists:(fun _ -> false)
       ~from_file:"docs/migration_guide.md" "[site](syntax_extensions.html)");
  p "another HTML destination is checked rather than exempted"
    (not (List.is_empty (check "[site](missing.html)")));
  p_all "every declared document scope is selected"
    [ "README.md"; "CHANGES.md"; "docs/example.md"; "benchmarks/example.md" ]
    ~f:Scan.in_corpus;
  p "archive is outside the document corpus" (not (Scan.in_corpus "archive/old.md"))
