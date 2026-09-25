(* gh-ocannl-1018: the operand-key reader's rule, put to synthesized sources -- each spelling it
   must flag beside the nearest legitimate text it must not -- and the shipping scanner run on a
   synthetic tree holding a blind fixture it must refuse. *)
open Base
open Stdio
module Scan = Test_utils.Operand_key_scan
open Verdict.Claims

let count source = List.length (Scan.sites source)

let spellings source =
  List.map (Scan.sites source) ~f:(fun (site : Scan.site) -> Scan.spelling_name site.spelling)

let () =
  (* The spellings the reader exists for. *)
  p "the blind 20x20 fixture that started the issue is a multi-index site"
    (List.equal String.equal
       (spellings
          "let wb = NTDSL.init ~o:[ 20; 20 ]\n\
          \  ~f:(fun idcs -> Float.of_int (((idcs.(0) * 20) + idcs.(1)) % 5) -. 2.) ()")
       [ "multi-index" ]);
  p "a weighted key over several axes is a site"
    (count "let f = fun idcs -> Float.of_int ((idcs.(0) + (2 * idcs.(1))) % 5)" = 1);
  p "a key let-bound inside the body is followed to its remainder"
    (count
       "let f = fun idcs ->\n  let flat = (idcs.(0) * n) + idcs.(1) in\n  Float.of_int (flat % 7)"
    = 1);
  p "an axis aliased one name at a time is followed too"
    (count "let f idcs = let h = idcs.(0) and w = idcs.(1) in Float.of_int (((h * 7) + w) % 4)" = 1);
  p "a named function is read, not only an ~f argument"
    (count "let value idx = Float.of_int (((idx.(0) * 32) + idx.(1)) % 13)" = 1);
  p "mod, Int.rem and a qualified ( % ) are remainders too"
    (count
       "let a v = (v.(0) + v.(1)) mod 3\n\
        let b v = Int.rem (v.(0) + v.(1)) 3\n\
        let c v = Int.( % ) (v.(0) + v.(1)) 3"
    = 3);
  p "the Array.get spelling of an axis read is the same read"
    (count "let f v = (Array.get v 0 + Array.get v 1) % 5" = 1);
  p "an axis read through a computed index counts as several axes"
    (count "let f dims idcs = (idcs.(Array.length dims - 1) * 3) % 5" = 1);
  p "the flat spelling over a product length is a flat site"
    (List.equal String.equal
       (spellings "let av = Array.init (m * k) ~f:(fun i -> Float.of_int (i % 13) *. 0.25)")
       [ "flat" ]);
  p "a flat closure spread over several lines is read whole: both remainders are sites"
    (count
       "let av =\n\
       \  Array.init (m * k) ~f:(fun x ->\n\
       \      ((Float.of_int (x % 23) -. 11.) *. 0.03125)\n\
       \      +. (Float.of_int (x % 3) *. 0.0007))"
    = 2);
  p "the positional Stdlib form and List.init are read too"
    (count
       "let a = Stdlib.Array.init (m * k) (fun i -> float_of_int (i mod 13))\n\
        let b = List.init (m * k) ~f:(fun i -> i % 7)"
    = 2);
  p "a divisor that is a LEADING factor is not unflattening"
    (count "let a = Array.init (rows * cols) ~f:(fun i -> Float.of_int (i % rows))" = 1);
  (* The nearest legitimate text, which a rule that fires on it would get switched off over. *)
  p "a site converted onto the guard has no remainder left"
    (count
       "let a = NTDSL.init ~o:[ 20; 20 ] ~f:(Ll_test.cycle ~dims:[| 20; 20 |] ~modulus:9 \
        ~offset:(-4.) ~stride:0.5) ()\n\
        let b = Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:7 ~offset:0. \
        ~stride:0.5)"
    = 0);
  p "one literal axis under a remainder is out of scope"
    (count
       "let a = fun idcs -> Float.of_int (idcs.(0) % 3) -. 1.\n\
        let b = fun idcs -> if idcs.(1) % bm = 0 then 1. else 0."
    = 0);
  p "unflattening by the trailing factor, or a trailing product, is the correct idiom"
    (count
       "let x = Array.init (rows * cols) ~f:(fun n -> cell (n / cols) (n % cols))\n\
        let y = Array.init (b * n * m) ~f:(fun idx -> (idx / (n * m)) + (idx % (n * m)))"
    = 0);
  p "a flat length that is not a product at the call is out of scope"
    (count "let a = Array.init n ~f:(fun i -> Float.of_int (i % 13))" = 0);
  p "comments and string literals are not code"
    (count
       "(* ~f:(fun idcs -> Float.of_int (((idcs.(0) * 20) + idcs.(1)) % 5)) *)\n\
        let s = {|fun idcs -> ((idcs.(0) * 20) + idcs.(1)) % 5|}"
    = 0);
  p "a remainder that reads the parameter only on its right is not a key"
    (count "let f idcs = 7 % (idcs.(0) + idcs.(1) + 1)" = 0);
  p "a remainder of values unrelated to the parameter is not a key"
    (count "let f idcs = Float.of_int ((a + b) % 5) +. Float.of_int idcs.(0)" = 0);
  (* The site key is the source text, whitespace collapsed, so a reformat cannot move it. *)
  p "the site key is the remainder's source text with whitespace collapsed"
    (match
       Scan.sites "let f idcs =\n  Float.of_int\n    (((idcs.(0) * 20)\n     + idcs.(1)) % 5)"
     with
    | [ site ] -> String.equal site.key "(((idcs.(0) * 20) + idcs.(1)) % 5)"
    | _ -> false);
  (* The exemption multiset. *)
  let two = "let f idcs = (idcs.(0) + idcs.(1)) % 3\nlet g idcs = (idcs.(0) + idcs.(1)) % 3" in
  let rows source = [ ("test/x.ml", Scan.sites source) ] in
  let key = "(idcs.(0) + idcs.(1)) % 3" in
  p "one Site row absorbs one of two identical sites, and the other is refused"
    (List.length (Scan.violations ~exemptions:[ ("test/x.ml", Scan.Site key, "r") ] (rows two)) = 1);
  let sites_of_two = Scan.sites two in
  p_empty "two Site rows absorb two identical sites" ~over:sites_of_two
    (Scan.violations
       ~exemptions:[ ("test/x.ml", Scan.Site key, "r"); ("test/x.ml", Scan.Site key, "r") ]
       (rows two));
  p_empty "a File row absorbs every site of its file" ~over:sites_of_two
    (Scan.violations ~exemptions:[ ("test/x.ml", Scan.File, "r") ] (rows two));
  p "a Site row for another file absorbs nothing, and is stale"
    (List.length (Scan.violations ~exemptions:[ ("test/y.ml", Scan.Site key, "r") ] (rows two)) = 3);
  p "a File row over a file with no sites is stale"
    (match Scan.violations ~exemptions:[ ("test/x.ml", Scan.File, "r") ] (rows "let x = 0") with
    | [ msg ] -> String.is_substring msg ~substring:"stale operand-key file exemption"
    | _ -> false);
  (* The shipping scanner on a synthetic tree: the negative control it must refuse. *)
  let exe = Stdlib.Sys.argv.(1) in
  let exe =
    if Stdlib.Filename.is_relative exe then Stdlib.Filename.concat (Stdlib.Sys.getcwd ()) exe
    else exe
  in
  let root = Stdlib.Filename.temp_dir "operand key control " "" in
  let write path data = Out_channel.write_all (Stdlib.Filename.concat root path) ~data in
  List.iter [ "test"; "arrayjit"; "arrayjit/test" ] ~f:(fun dir ->
      Unix.mkdir (Stdlib.Filename.concat root dir) 0o700);
  List.iter
    [ ("test", 200); ("arrayjit/test", 20) ]
    ~f:(fun (dir, n) ->
      for i = 1 to n do
        write (Printf.sprintf "%s/empty%d.ml" dir i) ""
      done);
  let run ?(exempt = false) () =
    let out = Stdlib.Filename.temp_file "operand-key" ".out" in
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
  let blind =
    "let wb =\n\
    \  NTDSL.init ~o:[ 20; 20 ]\n\
    \    ~f:(fun idcs -> Float.of_int (((idcs.(0) * 20) + idcs.(1)) % 5) -. 2.)\n\
    \    ()\n"
  in
  write "test/new.ml" blind;
  check "shipping scanner refuses the blind hand-rolled fixture" ~exit:1
    ~message:"test/new.ml:3: multi-index operand key `(((idcs.(0) * 20) + idcs.(1)) % 5)`"
    (run ());
  check "shipping scanner accepts the fixture under a named exemption" ~exit:0
    ~message:"control exemption" (run ~exempt:true ());
  write "test/new.ml"
    "let wb = NTDSL.init ~o:[ 20; 20 ] ~f:(Ll_test.cycle ~dims:[| 20; 20 |] ~modulus:9 \
     ~offset:(-4.) ~stride:0.5) ()\n";
  check "shipping scanner accepts the fixture converted onto the guard" ~exit:0
    ~message:"Source floor:" (run ());
  check "shipping scanner refuses the exemption the conversion left stale" ~exit:1
    ~message:"test/new.ml: stale operand-key exemption" (run ~exempt:true ());
  write "arrayjit/test/new.ml" "let av = Array.init (m * k) ~f:(fun i -> Float.of_int (i % 13))\n";
  check "shipping scanner reads the arrayjit test root too" ~exit:1
    ~message:"arrayjit/test/new.ml:1: flat operand key" (run ());
  Unix.unlink (Stdlib.Filename.concat root "arrayjit/test/new.ml");
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
