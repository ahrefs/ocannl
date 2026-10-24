(* gh-ocannl-1579: the open-node reader's rules, put to synthesized sources -- each form it must
   refuse beside the nearest legitimate code it must accept. The shipping scanner's own negative
   control is open_node_scan_control, over a committed fixture tree. *)
open Base
open Verdict.Claims
module Scan = Test_utils.Open_node_scan

(* A stand-in tnode.ml: [create] mints a uid, [create_pair] returns a node with its host buffer
   through [create], [find] mints none. *)
let tnode =
  "let fresh_uid =\n\
  \  let next = ref 0 in\n\
  \  fun () ->\n\
  \    Int.incr next;\n\
  \    !next\n\
   let create () = { uid = fresh_uid () }\n\
   let create_pair () = (create (), lazy ())\n\
   let find () = { uid = -1 }\n"

let derive source =
  Test_utils.Scan_attempt.attempted ~what:"open-node cases" ~name:(String.prefix source 60)
    ~default:{ Scan.names = []; tops = []; derivation_findings = [ "<the scan failed>" ] }
    (fun () -> Scan.derive_creators ~file:"arrayjit/lib/tnode.ml" source)

let creators = derive tnode

(* Most cases stand in for tensor.ml, whose top-level note_open_node is the real one. *)
let findings ?(file = Scan.owner_file) source =
  Test_utils.Scan_attempt.attempted ~what:"open-node cases" ~name:(String.prefix source 60)
    ~default:[ "<the scan failed>" ] (fun () ->
      (Scan.scan ~file ~creators ("module Tn = Ir.Tnode\n" ^ source)).findings)

let count ?file source = List.length (findings ?file source)

let says ?file source fragment =
  List.exists (findings ?file source) ~f:(fun f -> String.is_substring f ~substring:fragment)

let note = "let note_open_node _ = ()\n"

let () =
  (* The derivation. *)
  p "the creators are the values reaching fresh_uid, transitively"
    (List.equal String.equal creators.names [ "create"; "create_pair" ]);
  p_empty "the derivation of the stand-in refuses nothing" ~over:creators.names
    creators.derivation_findings;
  p "a tnode.ml with no top-level fresh_uid is refused"
    (List.exists (derive "let create () = { uid = 0 }").derivation_findings ~f:(fun f ->
         String.is_substring f ~substring:"no top-level fresh_uid"));
  p "a local fresh_uid shadowing the top-level one mints nothing"
    (List.equal String.equal
       (derive (tnode ^ "let fake () = let fresh_uid () = 0 in fresh_uid ()\n")).names
       [ "create"; "create_pair" ]);
  p "a uid minted inside a nested module is refused, not ignored"
    (List.exists
       (derive (tnode ^ "module M = struct let mk () = { uid = fresh_uid () } end\n"))
         .derivation_findings ~f:(fun f -> String.is_substring f ~substring:"cannot name"));
  (* Pairing. *)
  p "a creation bound and noted is accepted"
    (count (note ^ "let f () =\n  let tn = Tn.create () in\n  note_open_node tn;\n  tn") = 0);
  p "a creation bound and never noted is refused"
    (says (note ^ "let f () =\n  let tn = Tn.create () in\n  tn") "never reaches note_open_node");
  p "a pair-returning creator bound by a tuple pattern and noted is accepted"
    (count
       (note
      ^ "let f () =\n  let tn, init = Tn.create_pair () in\n  note_open_node tn;\n  (tn, init)")
    = 0);
  p "a creation flowing through match arms into a noted variable is accepted, every arm"
    (count
       (note
      ^ "let f b =\n\
        \  let v =\n\
        \    match b with\n\
        \    | true ->\n\
        \        let tn = Tn.create () in\n\
        \        ignore tn;\n\
        \        tn\n\
        \    | false -> Tn.create ()\n\
        \  in\n\
        \  note_open_node v;\n\
        \  v")
    = 0);
  p "the same flow with no note refuses both creations"
    (count
       (note
      ^ "let f b =\n\
        \  let v =\n\
        \    match b with\n\
        \    | true ->\n\
        \        let tn = Tn.create () in\n\
        \        tn\n\
        \    | false -> Tn.create ()\n\
        \  in\n\
        \  v")
    = 2);
  p "noting a different variable of the same name does not count"
    (count
       (note
      ^ "let f other =\n\
        \  let tn = Tn.create () in\n\
        \  let tn = (other : int) in\n\
        \  note_open_node tn;\n\
        \  tn")
    = 1);
  p "a creation bound to no variable is refused"
    (says (note ^ "let f () = let _ = Tn.create () in ()") "bound to no variable");
  p "a creation in argument position is refused as unfollowed"
    (says (note ^ "let f () = note_open_node (Tn.create ())") "not bound by a let");
  p "a creation returned from a function body is refused as unfollowed"
    (says (note ^ "let f () = Tn.create ()") "not bound by a let");
  p "a creator referenced without being applied is refused"
    (says (note ^ "let mk = Tn.create") "referenced without being applied");
  p "a call of a value that mints no uid is not a creation"
    (count (note ^ "let f () = Tn.find ()") = 0);
  (* Which note_open_node is the real one. *)
  p "a shadowing local note_open_node is not the real one"
    (count
       (note
      ^ "let f () =\n\
        \  let note_open_node _ = () in\n\
        \  let tn = Tn.create () in\n\
        \  note_open_node tn;\n\
        \  tn")
    = 1);
  p "outside tensor.ml, a top-level note_open_node is not the real one"
    (says ~file:"tensor/operation.ml"
       (note ^ "let f () =\n  let tn = Tn.create () in\n  note_open_node tn;\n  tn")
       "only tensor/tensor.ml's own top-level one");
  p "a note_open_node defined in a nested module is not the real one"
    (count
       "module M = struct let note_open_node _ = () end\n\
        open M\n\
        let f () =\n\
       \  let tn = Tn.create () in\n\
       \  note_open_node tn;\n\
       \  tn"
    = 1);
  p "a second top-level note_open_node is refused"
    (says (note ^ note ^ "let f () = ()") "a second top-level note_open_node");
  (* Which paths reach Ir.Tnode. *)
  p "Ir.Tnode.create, spelled out, is a creation"
    (count "let f () = let tn = Ir.Tnode.create () in tn" = 1);
  p "Tnode.create after open Ir is a creation"
    (count "open Ir\nlet f () = let tn = Tnode.create () in tn" = 1);
  p "a bare create under a local open of Tn is a creation"
    (count "let f () = Tn.(let tn = create () in tn)" = 1);
  p "the same under the open, noted, is accepted"
    (count (note ^ "let f () = Tn.(let tn = create () in note_open_node tn; tn)") = 0);
  p "a name the open of Tn brings in shadows a local binding of it"
    (count
       (note ^ "let f () =\n  let find = Tn.create () in\n  let open Tn in\n  note_open_node find")
    = 1);
  p "a locally bound module Tn is not Ir.Tnode"
    (count
       "let f () =\n\
       \  let module Tn = struct let create () = 0 end in\n\
       \  let tn = Tn.create () in\n\
       \  tn"
    = 0);
  p "a locally bound module Ir is not the library"
    (count
       "module Ir = struct module Tnode = struct let create () = 0 end end\n\
        module T = Ir.Tnode\n\
        let f () = let tn = T.create () in tn"
    = 0);
  (* Failure routing. *)
  p "a source that does not parse is the input's fault, not the scanner's"
    (match Test_utils.Scan_attempt.run (fun () -> Scan.scan ~file:"x.ml" ~creators "let = (") with
    | Unparsed _ -> true
    | Scanned _ | Raised _ -> false)
