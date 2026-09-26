(* gh-ocannl-964: parser, membership and shipping-executable negative controls. *)
open Base
open Stdio
module Scan = Test_utils.Ll_test_scan
open Verdict.Claims

let declaration =
  "type t = For_loop of { body : t } | Seq of t * t | Noop and scalar_t = Constant of float"

let constructors = Scan.constructors declaration
let record = "let x = Alias.For_loop { body = Alias.Noop }\n"

(* What a test that calls the harness's IR surface adds to its hand-built record: under
   gh-ocannl-1052 linking the harness is not adoption, calling into it is. *)
let use = "let _ = Ll_test.seq\n"
let adopted = record ^ use

(* A synthetic harness in the shipping layout: a builder tier with an [Ir] alias, and a harness that
   includes it and adds one builder caller and one operand helper that touches no IR. *)
let builders_source = "module LL = Ir.Low_level\nlet seq a b : LL.t = LL.Seq (a, b)\n"

let harness_source =
  "include Ll_builders\nlet twice x = seq x x\nlet cycle ~modulus i = i mod modulus\n"

let walker =
  "let rec walk = function Alias.Seq (a,b) -> walk a + walk b | Alias.For_loop {body} -> walk body \
   | _ -> 0\n"

let () =
  let counts source = Scan.census ~constructors source in
  p "zero records stay below the adoption floor" (not (Scan.needs_harness (counts "let x = 0")));
  p "the first record requires harness adoption" (Scan.needs_harness (counts record));
  p "two records also require harness adoption" (Scan.needs_harness (counts (record ^ record)));
  p "a private recursive walker reaches the floor without any construction"
    (let c = counts walker in
     c.records = 0 && c.traversals = 1 && Scan.needs_harness c);
  p "comment and quoted fixture records are not constructions"
    ((counts ("(* " ^ record ^ " *)\nlet fixture = {|" ^ record ^ "|}")).records = 0);
  p "record match arms are not constructions"
    ((counts "let inspect = function Alias.For_loop {body} -> body | x -> x").records = 0);
  p "tuple constructors with foreign record payloads do not count"
    ((counts "let foreign = Other.Seq { arbitrary = 1 }").records = 0);
  p "derived constructor membership includes a new inline-record constructor"
    ((Scan.census
        ~constructors:
          (Scan.constructors
             (String.substr_replace_all declaration ~pattern:"| Noop"
                ~with_:"| Noop | Future_node of { body : t }"))
        "let x = Vendor.Future_node { body = Vendor.Noop }")
       .records = 1);
  p "constructors in unrelated nested type declarations do not widen the census"
    ((Scan.census
        ~constructors:
          (Scan.constructors
             (declaration ^ "\nmodule Foreign = struct type t = Alien of {body : int} end"))
        "let x = Foreign.Alien {body = 1}")
       .records = 0);
  p "nested private walkers are counted once, not again through their parent"
    ((counts ("let rec outer x = " ^ walker ^ " in walk x")).traversals = 1);
  p "a walker inside a local module is not counted through its recursive parent"
    ((counts ("let rec outer x = let module M = struct " ^ walker ^ " end in M.walk x")).traversals
   = 1);
  let surface = Scan.surface [ builders_source; harness_source ] in
  p "the IR surface is derived: builders and their callers are in it, an operand helper is not"
    (Set.equal surface.ir (Set.of_list (module String) [ "seq"; "twice" ])
    && Set.equal surface.other (Set.of_list (module String) [ "cycle" ]));
  p "a local binding that shadows a builder is not a call into the IR surface"
    (Set.mem (Scan.surface [ builders_source; "let bump seq = seq + 1\n" ]).other "bump");
  p "a type annotation through an Ir alias puts a value in the IR surface"
    (Set.mem (Scan.surface [ builders_source; "let id (x : LL.t) = x\n" ]).ir "id");
  let uses source = Scan.uses_surface ~surface source in
  p "a qualified builder call uses the IR surface" (uses "let _ = Ll_test.seq");
  p "linking for an operand helper alone uses nothing of the IR surface"
    (not (uses "let _ = Ll_test.cycle ~modulus:3 1"));
  p "an alias of the harness reaches the IR surface"
    (uses "module L = Ll_test\nlet _ = L.twice" && not (uses "module L = Ll_test\nlet _ = L.cycle"));
  p "an alias of the public builder tier reaches the IR surface"
    (uses "module B = Ll_builders\nlet _ = B.seq");
  p "an unqualified builder counts only under an open of the harness"
    (uses "open Ll_test\nlet _ = seq"
    && (not (uses "let _ = seq"))
    && not (uses "let _ = seq\nopen Ll_test"));
  p "a local open covers its body alone"
    (uses "let _ = Ll_test.(seq)" && not (uses "let _ = Ll_test.(cycle)\nlet _ = seq"));
  p "a name the file binds for itself is not taken for the builder under an open"
    (not (uses "open Ll_test\nlet f seq = seq"));
  let linked content =
    Scan.linked ~directory_modules:[ "new"; "other" ] ~module_name:"new"
      (Test_utils.Dune_stanza_scan.stanzas content)
  in
  p "a sibling stanza's harness does not cover this module"
    (not
       (linked
          "(test (name new) (modules new)) (test (name other) (modules other) (libraries ll_test))"));
  p "default module membership links the owning harness"
    (linked "(test (name new) (libraries ll_test)) (test (name other) (modules other))");
  p "ordered-set exclusion prevents harness membership"
    (not (linked "(library (name tests) (modules (:standard \\ new)) (libraries ll_test))"));
  p "a new subdirectory in either package is in scope"
    (Scan.test_source "test/future/new.ml" && Scan.test_source "arrayjit/test/future/new.ml");
  let exe = Stdlib.Sys.argv.(1) in
  let exe =
    if Stdlib.Filename.is_relative exe then Stdlib.Filename.concat (Stdlib.Sys.getcwd ()) exe
    else exe
  in
  let root = Stdlib.Filename.temp_dir "ll ratchet control " "" in
  let write path data = Out_channel.write_all (Stdlib.Filename.concat root path) ~data in
  List.iter [ "test"; "arrayjit"; "arrayjit/test"; "arrayjit/lib" ] ~f:(fun dir ->
      Unix.mkdir (Stdlib.Filename.concat root dir) 0o700);
  write "arrayjit/lib/low_level.ml" declaration;
  Unix.mkdir (Stdlib.Filename.concat root "test/support") 0o700;
  write "test/support/ll_builders.ml" builders_source;
  write "test/support/ll_test.ml" harness_source;
  List.iter
    [ ("test", 200); ("arrayjit/test", 20) ]
    ~f:(fun (dir, count) ->
      for i = 1 to count do
        write (Printf.sprintf "%s/empty%d.ml" dir i) ""
      done);
  write "test/dune" "(test (name new) (modules new))";
  let run ?(exempt = false) ?(permanent = false) () =
    let out = Stdlib.Filename.temp_file "ll-ratchet" ".out" in
    let fd = Unix.openfile out [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
    let pid =
      Unix.create_process exe
        [|
          exe;
          (if permanent then "--fixture-permanent"
           else if exempt then "--fixture-exempt"
           else "--fixture");
          root;
        |]
        Unix.stdin fd fd
    in
    let _, status = Unix.waitpid [] pid in
    Unix.close fd;
    let text = In_channel.read_all out in
    Unix.unlink out;
    (status, text)
  in
  let check label ~exit ~message (status, text) =
    let ok =
      (match status with Unix.WEXITED n -> n = exit | _ -> false)
      && String.is_substring text ~substring:message
    in
    if not ok then eprintf "%s captured output:\n%s\n" label text;
    p label ok
  in
  write "test/new.ml" adopted;
  check "shipping scanner refuses the first unlinked record builder" ~exit:1
    ~message:"test/new.ml: requires ll_test" (run ());
  write "test/dune" "(test (name new) (modules new) (libraries ll_test))";
  check "shipping scanner accepts adoption without golden churn" ~exit:0
    ~message:"Adoption threshold:" (run ());
  check "shipping scanner prints the derived operand helpers that adopt nothing" ~exit:0
    ~message:"Harness values outside the IR surface (adopt nothing): cycle\n" (run ());
  (* The gh-ocannl-1052 negative control: the stanza links ll_test, and the source calls only an
     operand helper, so its hand-built record is still debt. *)
  write "test/new.ml" (record ^ "let _ = Ll_test.cycle ~modulus:3 1\n");
  check "linking ll_test for an operand helper alone retires nothing" ~exit:1
    ~message:"test/new.ml: links ll_test but calls none of its IR surface" (run ());
  check "a linked-but-unused file is held to its migration row" ~exit:0 ~message:"control exemption"
    (run ~exempt:true ());
  write "test/new.ml" adopted;
  let base_stanza = "(test (name new) (modules new) (libraries ll_test))" in
  let selection ?(modules = "(modules choice)") ?(harness = "ll_test") () =
    "(test (name choice) " ^ modules ^ " (libraries " ^ harness
    ^ " (select choice.ml from (backend -> choice.real.ml) (-> choice.missing.ml))))"
  in
  write "test/choice.real.ml" adopted;
  write "test/choice.missing.ml" adopted;
  write "test/dune" (base_stanza ^ selection ());
  check "both select arms inherit the generated target module's harness" ~exit:0
    ~message:"Adoption threshold:" (run ());
  write "test/dune" (base_stanza ^ selection ~modules:"" ());
  check "default module ownership includes select targets, not arm basenames" ~exit:0
    ~message:"Adoption threshold:" (run ());
  write "test/dune" (base_stanza ^ selection ~harness:"" ());
  let missing_select_harness = run () in
  check "a sibling harness cannot cover the selected real source" ~exit:1
    ~message:"test/choice.real.ml: requires ll_test" missing_select_harness;
  check "the off-platform select arm is checked independently too" ~exit:1
    ~message:"test/choice.missing.ml: requires ll_test" missing_select_harness;
  write "test/dune" (base_stanza ^ selection ~modules:"(modules (:standard \\ choice))" ());
  check "select target exclusion prevents harness ownership" ~exit:1
    ~message:"test/choice.real.ml: requires ll_test" (run ());
  write "test/dune"
    (base_stanza ^ selection ()
   ^ "(test (name other) (modules other) (libraries (select other.ml from (-> choice.real.ml))))");
  check "every owning select stanza must link the harness for a shared arm" ~exit:1
    ~message:"test/choice.real.ml: requires ll_test" (run ());
  Unix.unlink (Stdlib.Filename.concat root "test/choice.real.ml");
  Unix.unlink (Stdlib.Filename.concat root "test/choice.missing.ml");
  write "test/dune" base_stanza;
  Unix.mkdir (Stdlib.Filename.concat root "test/shared") 0o700;
  write "test/shared/copied.ml" adopted;
  let copied_stanza ?(harness = "ll_test") ?(modules = "(modules copied)") () =
    "(test (name copied) " ^ modules ^ " (libraries " ^ harness ^ "))"
  in
  let copy = "(copy_files (files shared/copied.ml))" in
  write "test/dune" (base_stanza ^ copy ^ copied_stanza ());
  check "an unowned copy source inherits its destination harness" ~exit:0
    ~message:"Adoption threshold:" (run ());
  write "test/dune" (base_stanza ^ copy ^ copied_stanza ~modules:"" ());
  check "copied targets enter default module ownership" ~exit:0 ~message:"Adoption threshold:"
    (run ());
  write "test/shared/dune" "(library (name original) (modules copied) (libraries ll_test))";
  write "test/dune" (base_stanza ^ copy ^ copied_stanza ~harness:"" ());
  check "a linked original cannot hide an unlinked copied consumer" ~exit:1
    ~message:"test/shared/copied.ml: requires ll_test" (run ());
  write "test/dune" (base_stanza ^ "(copy_files# shared/copied.ml)" ^ copied_stanza ());
  check "short copy_files# retains source ownership" ~exit:0 ~message:"Adoption threshold:" (run ());
  write "arrayjit/test/dune"
    "(copy_files ../../test/copied.ml) (test (name copied) (modules copied) (libraries ll_test))";
  check "literal copy chains retain every consumer" ~exit:0 ~message:"Adoption threshold:" (run ());
  write "arrayjit/test/dune"
    "(copy_files ../../test/copied.ml) (test (name copied) (modules copied))";
  check "an unlinked consumer at the end of a copy chain is refused" ~exit:1
    ~message:"test/shared/copied.ml: requires ll_test" (run ());
  write "arrayjit/test/dune" "";
  write "test/dune" (base_stanza ^ "(copy_files shared/*.ml)" ^ copied_stanza ());
  check "unsupported copy globs are refused explicitly" ~exit:1
    ~message:"unsupported copy_files glob or dynamic source" (run ());
  write "test/dune" (base_stanza ^ "(copy_files ../outside.ml)" ^ copied_stanza ());
  check "copy inputs outside the declared corpus are refused explicitly" ~exit:1
    ~message:"source input outside declared test corpus" (run ());
  write "dune" "(subdir staging (copy_files ../outside.ml))";
  write "test/dune" (base_stanza ^ "(copy_files ../staging/outside.ml)" ^ copied_stanza ());
  check "copy chains cannot hide an origin outside the declared corpus" ~exit:1
    ~message:"source input outside declared test corpus" (run ());
  write "dune" "";
  Unix.unlink (Stdlib.Filename.concat root "test/shared/copied.ml");
  write "test/shared/dune" "";
  write "test/dune" base_stanza;
  Unix.mkdir (Stdlib.Filename.concat root "test/ppx") 0o700;
  write "test/ppx/fixture_expected.ml" record;
  check "PPX output goldens do not require a library-owning stanza" ~exit:0
    ~message:"Adoption threshold:" (run ());
  write "test/fixture_expected.ml" record;
  check "ordinary unowned sources cannot borrow the PPX golden exclusion" ~exit:1
    ~message:"test/fixture_expected.ml: requires ll_test" (run ());
  Unix.unlink (Stdlib.Filename.concat root "test/fixture_expected.ml");
  Unix.unlink (Stdlib.Filename.concat root "test/dune");
  write "dune" "(subdir test (test (name new) (modules new) (libraries ll_test)))";
  check "shipping scanner resolves a parent subdir owning the adopted test" ~exit:0
    ~message:"Adoption threshold:" (run ());
  write "test/dune" "(test (name other) (modules other))";
  write "dune" "(subdir test (test (name new) (libraries ll_test)))";
  check "parent subdir defaults share ownership with physical child stanzas" ~exit:0
    ~message:"Adoption threshold:" (run ());
  Unix.unlink (Stdlib.Filename.concat root "test/dune");
  write "dune" "(include_subdirs unqualified) (library (name parent) (libraries ll_test))";
  check "shipping scanner explicitly refuses include_subdirs ownership" ~exit:1
    ~message:"unsupported module ownership directive include_subdirs" (run ());
  write "dune" "(subdir test (include_subdirs qualified))";
  check "nested include_subdirs cannot escape the ownership refusal" ~exit:1
    ~message:"test: unsupported module ownership directive include_subdirs" (run ());
  write "dune" "(subdir test (include test_stanzas))";
  check "unresolved Dune includes are explicit ownership refusals" ~exit:1
    ~message:"test: unsupported module ownership directive include" (run ());
  write "dune" "(include_subdirs no)";
  write "test/dune" "(test (name new) (modules new) (libraries ll_test))";
  check "include_subdirs no keeps direct ownership valid" ~exit:0 ~message:"Adoption threshold:"
    (run ());
  write "dune" "(subdir unrelated (include_subdirs unqualified))";
  check "unrelated ownership directives do not widen scanner scope" ~exit:0
    ~message:"Adoption threshold:" (run ());
  check "shipping scanner refuses stale exemptions after adoption" ~exit:1
    ~message:"test/new.ml: stale ll_test exemption" (run ~exempt:true ());
  write "test/dune" "(test (name new) (modules new))";
  write "test/new.ml" walker;
  check "shipping scanner refuses a private traversal alone" ~exit:1
    ~message:"test/new.ml: requires ll_test" (run ());
  check "shipping scanner accepts an explicitly exempt migration" ~exit:0
    ~message:"control exemption" (run ~exempt:true ());
  write "test/new.ml" (record ^ record);
  check "migration debt at its recorded builder cap remains valid" ~exit:0
    ~message:"control exemption" (run ~exempt:true ());
  write "test/new.ml" record;
  check "decreased migration debt remains valid" ~exit:0 ~message:"control exemption"
    (run ~exempt:true ());
  write "test/new.ml" (record ^ record ^ record);
  check "new builders in an exempt file exceed its migration cap" ~exit:1
    ~message:"test/new.ml: migration debt grew beyond ll_test baseline (records 3/2"
    (run ~exempt:true ());
  check "an intentional permanent exception is not a migration quota" ~exit:0
    ~message:"control permanent exemption" (run ~permanent:true ());
  write "test/new.ml" (walker ^ walker);
  check "new private traversals in an exempt file exceed their independent cap" ~exit:1
    ~message:"traversals 2/1" (run ~exempt:true ());
  write "test/new.ml" (walker ^ walker ^ use);
  write "test/dune" "(test (name new) (modules new) (libraries ll_test))";
  check "permanent exemptions also become stale after harness adoption" ~exit:1
    ~message:"test/new.ml: stale ll_test exemption" (run ~permanent:true ());
  write "test/dune" "(test (name new) (modules new))";
  write "test/new.ml" "";
  write "arrayjit/test/new.ml" (record ^ record ^ record ^ "module B = Ll_builders\nlet _ = B.seq\n");
  check "new arrayjit debt requires explicit adoption" ~exit:1
    ~message:"arrayjit/test/new.ml: requires ll_test" (run ());
  write "arrayjit/test/dune" "(test (name new) (modules new) (libraries arrayjit.ll_builders))";
  check "public arrayjit builders satisfy package adoption" ~exit:0 ~message:"Adoption threshold:"
    (run ());
  write "arrayjit/test/dune" "(test (name new) (modules new) (libraries ll_builders))";
  check "private builder spelling does not satisfy package adoption" ~exit:1
    ~message:"arrayjit/test/new.ml: requires ll_test" (run ());
  write "arrayjit/test/dune"
    "(test (name new) (modules new)) (library (name other) (modules other) (libraries \
     arrayjit.ll_builders))";
  check "public builders in another stanza do not cover this consumer" ~exit:1
    ~message:"arrayjit/test/new.ml: requires ll_test" (run ());
  write "arrayjit/test/dune" "";
  write "arrayjit/test/new.ml" "";
  for i = 1 to 20 do
    Unix.unlink (Stdlib.Filename.concat root (Printf.sprintf "arrayjit/test/empty%d.ml" i))
  done;
  check "shipping scanner refuses an emptied arrayjit source root" ~exit:1
    ~message:"arrayjit/test/: source inventory below floor" (run ());
  let rec remove path =
    if Stdlib.Sys.is_directory path then (
      Array.iter (Stdlib.Sys.readdir path) ~f:(fun name ->
          remove (Stdlib.Filename.concat path name));
      Unix.rmdir path)
    else Unix.unlink path
  in
  remove root
