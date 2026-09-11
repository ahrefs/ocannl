(* gh-ocannl-964: parser, membership and shipping-executable negative controls. *)
open Base
open Stdio
module Scan = Test_utils.Ll_test_scan
open Verdict.Claims

let declaration =
  "type t = For_loop of { body : t } | Seq of t * t | Noop and scalar_t = Constant of float"

let constructors = Scan.constructors declaration
let record = "let x = Alias.For_loop { body = Alias.Noop }\n"

let walker =
  "let rec walk = function Alias.Seq (a,b) -> walk a + walk b | Alias.For_loop {body} -> walk body \
   | _ -> 0\n"

let () =
  let counts source = Scan.census ~constructors source in
  p "two records stay below the adoption floor"
    (not (Scan.needs_harness (counts (record ^ record))));
  p "three records reach the adoption floor"
    (Scan.needs_harness (counts (record ^ record ^ record)));
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
  let linked content =
    Scan.linked ~directory_modules:[ "new"; "other" ] ~module_name:"new" content
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
  List.iter
    [ ("test", 200); ("arrayjit/test", 20) ]
    ~f:(fun (dir, count) ->
      for i = 1 to count do
        write (Printf.sprintf "%s/empty%d.ml" dir i) ""
      done);
  write "test/dune" "(test (name new) (modules new))";
  let run ?(exempt = false) () =
    let out = Stdlib.Filename.temp_file "ll-ratchet" ".out" in
    let fd = Unix.openfile out [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
    let pid =
      Unix.create_process exe
        [| exe; (if exempt then "--fixture-exempt" else "--fixture"); root |]
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
  write "test/new.ml" (record ^ record ^ record);
  check "shipping scanner refuses unlinked record builders" ~exit:1
    ~message:"test/new.ml: requires ll_test" (run ());
  write "test/dune" "(test (name new) (modules new) (libraries ll_test))";
  check "shipping scanner accepts adoption without golden churn" ~exit:0
    ~message:"Adoption threshold:" (run ());
  check "shipping scanner refuses stale exemptions after adoption" ~exit:1
    ~message:"test/new.ml: stale ll_test exemption" (run ~exempt:true ());
  write "test/dune" "(test (name new) (modules new))";
  write "test/new.ml" walker;
  check "shipping scanner refuses a private traversal alone" ~exit:1
    ~message:"test/new.ml: requires ll_test" (run ());
  check "shipping scanner accepts an explicitly exempt migration" ~exit:0
    ~message:"control exemption" (run ~exempt:true ());
  write "test/new.ml" "";
  write "arrayjit/test/new.ml" (record ^ record ^ record);
  check "new arrayjit debt is not implicitly exempted by the package blocker" ~exit:1
    ~message:"arrayjit/test/new.ml: requires ll_test" (run ());
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
