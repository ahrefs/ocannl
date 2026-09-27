(* gh-ocannl-1015: the rejection-code reader's rules, put to synthesized sources -- each spelling it
   must read beside the nearest text it must not -- and the shipping inventory run on a synthetic
   tree it must accept, then refuse once a citation goes stale or the phase table lies.

   The constructor's name is spelled in pieces ([nv]) wherever it would be followed by a number: the
   inventory reads this file too, and a fixture citing a code the library does not mint is exactly
   what it refuses. The fixture tags are invented for the same reason. *)
open Base
open Stdio
module Scan = Test_utils.Nonvirtual_code_scan
open Verdict.Claims

let nv = "Non" ^ "_virtual"

let library_with ~store ~consume =
  String.concat ~sep:"\n"
    [
      "let%diagn2_sexp " ^ store ^ " x =";
      "  let exception " ^ nv ^ " of string in";
      "  let helper ~code = if x then raise @@ " ^ nv ^ " code in";
      "  (* \"5:fixture-in-a-comment\" is not a literal *)";
      "  try";
      "    helper ~code:\"9:fixture-helper\";";
      "    if x then raise @@ " ^ nv ^ " \"7:fixture-store\";";
      "    ignore [ \"INFO: 3 things\"; \"7:\"; \":reason\"; \"7:Upper\"; \"7:trailing-\" ]";
      "  with " ^ nv ^ " i -> record (Site i)";
      "";
      "let " ^ consume ^ " y =";
      "  let exception " ^ nv ^ " of string in";
      "  let rec go = function [] -> () | _ :: tl -> go tl in";
      "  go y;";
      "  record (Site \"8:fixture-recorded\");";
      "  raise (" ^ nv ^ " \"4:fixture-consume\")";
      "";
      "let elsewhere () = record (Site \"6:fixture-elsewhere\")";
    ]

let library = library_with ~store:"store_check" ~consume:"consume"
let tags codes = List.map codes ~f:(fun (c : Scan.code) -> c.tag)
let minted = Scan.merge (Scan.minted ~source:"lib/fixture.ml" library)

let minter_of tag =
  List.find_map minted ~f:(fun (c : Scan.code) -> Option.some_if (String.equal c.tag tag) c.minter)

let named text = fst (Scan.mentions ~codes:minted text)
let unknown text = snd (Scan.mentions ~codes:minted text)
let ints = List.equal Int.equal

let table =
  "let phase_table =\n\
  \  [ (\"7:fixture-store\", Store); (\"9:fixture-helper\", Store); (\"4:fixture-consume\", \
   Consumption) ]\n"

let phases = [ ("Store", "store_check"); ("Consumption", "consume") ]

let violations ?(codes = minted) ?(table = Scan.phase_table table) ?(files = []) () =
  let files =
    ("lib/fixture.ml", List.map codes ~f:(fun (c : Scan.code) -> c.number), []) :: files
  in
  Scan.violations ~codes ~table_source:"t.ml" ~table ~phases ~files

let has ~substring = List.exists ~f:(String.is_substring ~substring)

let () =
  (* What a code is. *)
  p "every tag literal in an exception's scope is a code, a helper's argument and a record included"
    (List.equal String.equal (tags minted)
       [ "4:fixture-consume"; "7:fixture-store"; "8:fixture-recorded"; "9:fixture-helper" ]);
  p "a code belongs to the function declaring its exception"
    (List.equal (Option.equal String.equal)
       (List.map
          [ "7:fixture-store"; "9:fixture-helper"; "4:fixture-consume"; "8:fixture-recorded" ]
          ~f:minter_of)
       [ Some "store_check"; Some "store_check"; Some "consume"; Some "consume" ]);
  p_none "a tag outside every exception scope, or inside a comment, is not a code" minted
    ~f:(fun (c : Scan.code) ->
      List.mem [ "6:fixture-elsewhere"; "5:fixture-in-a-comment" ] c.tag ~equal:String.equal);
  (let nested =
     String.concat ~sep:"\n"
       [
         "let outer x =";
         "  let exception " ^ nv ^ " of string in";
         "  let check () = if x then raise @@ " ^ nv ^ " \"2:fixture-in-helper\" in";
         "  let inner () =";
         "    let exception " ^ nv ^ " of string in";
         "    raise (" ^ nv ^ " \"3:fixture-inner\")";
         "  in";
         "  check (); inner (); raise (" ^ nv ^ " \"7:fixture-store\")";
       ]
   in
   p "a helper inside a scope mints for the scope, and a scope declared anew inside it is its own"
     (List.equal String.equal
        (List.map
           (Scan.merge (Scan.minted ~source:"n.ml" nested))
           ~f:(fun (c : Scan.code) -> c.tag ^ " " ^ c.minter))
        [ "2:fixture-in-helper outer"; "3:fixture-inner inner"; "7:fixture-store outer" ]));
  p "a function declaring no such exception adds no code to a source that does"
    (List.equal String.equal
       (tags
          (Scan.merge
             (Scan.minted ~source:"lib/fixture.ml"
                ("let before () = record (Site \"3:fixture-free\")\n" ^ library))))
       (tags minted));
  (* What naming one is. *)
  p "the constructor with the number names the code" (ints (named (nv ^ " 7 fires")) [ 7 ]);
  p "a citation wrapped at the line break still names it"
    (ints (named ("see [" ^ nv ^ "\n     9] here")) [ 9 ]);
  p "the tag itself names the code, in prose or in a string"
    (ints (named "rejected as 4:fixture-consume, or \"8:fixture-recorded\".") [ 4; 8 ]);
  p "a tag continued by another tag character, or by a longer number, is a different tag"
    (ints (named "7:fixture-store-twice 17:fixture-store 9:fixture-helper") [ 9 ]);
  p "an identifier ending in the name is not the constructor"
    (ints (named ("known_" ^ nv ^ " 7 and X" ^ nv ^ " 7, but " ^ nv ^ " 9")) [ 9 ]);
  p "the constructor with no number, or with a variable, names nothing"
    (ints (named (nv ^ " i, " ^ nv ^ " of string, " ^ nv ^ ". Then " ^ nv ^ " 4")) [ 4 ]);
  p "a longer number is not a prefix match on a shorter code"
    (let text = nv ^ " 77 and " ^ nv ^ " 7" in
     ints (named text) [ 7 ] && ints (unknown text) [ 77 ]);
  p "a citation of a number no raise site mints is stale" (ints (unknown (nv ^ " 3")) [ 3 ]);
  p "a bare numeral is not read" (ints (named "refused as 7 first, then as 9:fixture-helper") [ 9 ]);
  (* The phase table. *)
  p "the table reads as its literal pairs"
    (Option.equal
       (List.equal (fun (a, b) (c, d) -> String.equal a c && String.equal b d))
       (Scan.phase_table table)
       (Some
          [
            ("7:fixture-store", "Store");
            ("9:fixture-helper", "Store");
            ("4:fixture-consume", "Consumption");
          ]));
  p "a table with an element of another shape is unread, not guessed at"
    (Option.is_none (Scan.phase_table "let phase_table = [ (\"7:fixture-store\", phase) ]"));
  p "a source with no table binding has no table"
    (Option.is_none (Scan.phase_table "let other_table = []"));
  (* The refusals, and the clean case they must leave alone. *)
  p_empty "a table agreeing with the minting functions is accepted" ~over:minted (violations ());
  p "a table entry placed in the other function's phase is refused"
    (has ~substring:"but it is minted in consume"
       (violations
          ~table:(Scan.phase_table "let phase_table = [ (\"4:fixture-consume\", Store) ]")
          ()));
  p "a table entry no raise site mints is refused"
    (has ~substring:"is minted nowhere"
       (violations
          ~table:(Scan.phase_table "let phase_table = [ (\"6:fixture-elsewhere\", Store) ]")
          ()));
  p "a table entry naming a phase with no minter is refused"
    (has ~substring:"which has no minter"
       (violations ~table:(Scan.phase_table "let phase_table = [ (\"7:fixture-store\", Cap) ]") ()));
  p "an unreadable table is refused rather than skipped"
    (has ~substring:"no `phase_table` binding" (violations ~table:None ()));
  p "a phase whose function mints nothing is stale"
    (has ~substring:"names consume, which mints no code"
       (violations
          ~codes:
            (List.filter minted ~f:(fun (c : Scan.code) -> String.equal c.minter "store_check"))
          ~table:(Scan.phase_table "let phase_table = []")
          ()));
  p "a number minted under two tags is ambiguous"
    (has ~substring:"a numeric reference to it is ambiguous"
       (violations
          ~codes:
            (Scan.merge
               (minted
               @ [
                   {
                     Scan.number = 7;
                     tag = "7:fixture-other";
                     source = "lib/fixture.ml";
                     minter = "store_check";
                   };
                 ]))
          ~table:(Scan.phase_table "let phase_table = []")
          ()));
  p "a tag minted in two functions is refused: its provenance cannot name the phase"
    (has
       ~substring:
         "7:fixture-store is minted in lib/fixture.ml consume and lib/fixture.ml store_check"
       (violations
          ~codes:
            (Scan.merge
               (minted
               @ [
                   {
                     Scan.number = 7;
                     tag = "7:fixture-store";
                     source = "lib/fixture.ml";
                     minter = "consume";
                   };
                 ]))
          ()));
  p "a stale citation is refused with its file"
    (has ~substring:"doc.md: `Non"
       (violations
          ~files:[ ("doc.md", [], [ 3 ]) ]
          ~table:(Scan.phase_table "let phase_table = []")
          ()));
  p "no code at all is the reader gone blind"
    (has ~substring:"the reader is blind"
       (violations ~codes:[] ~table:(Scan.phase_table "let phase_table = []") ()));
  p "a minting source whose codes the mention reader misses is the reader gone blind"
    (has ~substring:"but the mention reader does not see it there"
       (Scan.violations ~codes:minted ~table_source:"t.ml"
          ~table:(Scan.phase_table "let phase_table = []")
          ~phases
          ~files:[ ("lib/fixture.ml", [ 7 ], []) ]));
  p "an excluded prefix no file lives under is stale"
    (List.length
       (Scan.stale_records
          ~records:[ ("docs/proposals/", "r"); ("docs/gone/", "r") ]
          [ "docs/proposals/a.md"; "docs/x.md" ])
    = 1);
  (* The shipping inventory on a synthetic tree. *)
  let exe = Stdlib.Sys.argv.(1) in
  let exe =
    if Stdlib.Filename.is_relative exe then Stdlib.Filename.concat (Stdlib.Sys.getcwd ()) exe
    else exe
  in
  let root = Stdlib.Filename.temp_dir "nonvirtual code control " "" in
  let write path data = Out_channel.write_all (Stdlib.Filename.concat root path) ~data in
  List.iter [ "arrayjit"; "arrayjit/lib"; "test"; "test/operations"; "docs" ] ~f:(fun dir ->
      Unix.mkdir (Stdlib.Filename.concat root dir) 0o700);
  write "arrayjit/lib/low_level.ml"
    (library_with ~store:"check_and_store_virtual" ~consume:"instantiate_computations");
  let boundary = "test/operations/virtual_rejection_boundary.ml" in
  write boundary table;
  write "docs/page.md"
    ("A candidate is refused as `" ^ nv ^ " 9`, and the store as 7:fixture-store.\n");
  let run () =
    let out = Stdlib.Filename.temp_file "nonvirtual-code" ".out" in
    let fd = Unix.openfile out [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
    let pid = Unix.create_process exe [| exe; "--fixture"; root |] Unix.stdin fd fd in
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
  check "shipping inventory lists a page citing a code both ways" ~exit:0
    ~message:"docs/page.md -- 7 9" (run ());
  check "shipping inventory groups the codes under the function minting them" ~exit:0
    ~message:"arrayjit/lib/low_level.ml, instantiate_computations:\n  4:fixture-consume\n" (run ());
  write "docs/page.md" ("A candidate is refused as `" ^ nv ^ " 3`.\n");
  check "shipping inventory refuses a citation of a code no raise site mints" ~exit:1
    ~message:"docs/page.md: `Non" (run ());
  write "docs/page.md" "";
  write boundary "let phase_table = [ (\"4:fixture-consume\", Store) ]\n";
  check "shipping inventory refuses a phase table placing a code in the wrong phase" ~exit:1
    ~message:"puts 4:fixture-consume at Store, but it is minted in instantiate_computations"
    (run ());
  let rec remove path =
    if Stdlib.Sys.is_directory path then (
      Array.iter (Stdlib.Sys.readdir path) ~f:(fun name ->
          remove (Stdlib.Filename.concat path name));
      Unix.rmdir path)
    else Unix.unlink path
  in
  remove root
