(* gh-ocannl-1506: the mutation-serial reader's rules, put to synthesized sources -- each form it
   must refuse beside the nearest legitimate code it must accept. The shipping scanner's own
   negative control is mutation_serial_scan_control, over a committed fixture tree. *)
open Base
open Verdict.Claims
module Scan = Test_utils.Mutation_serial_scan

let findings ?interface source =
  Test_utils.Scan_attempt.attempted ~what:"mutation-serial cases" ~name:(String.prefix source 60)
    ~default:[ "<the scan failed>" ] (fun () ->
      (Scan.scan ~file:"x.ml" ?interface ("open Base\n" ^ source)).findings)

let count ?interface source = List.length (findings ?interface source)

(* The finding's message, without its [file:line: ] prefix. *)
let says ?interface source fragment =
  List.exists (findings ?interface source) ~f:(fun f -> String.is_substring f ~substring:fragment)

let table = "let note_mutation () = ()\nlet seen = Hash_set.create (module Int)\n"

let () =
  (* Accounting. *)
  p "an unnoted write to a module-level table is refused"
    (says (table ^ "let add v = Hash_set.add seen v") "add writes seen without note_mutation");
  p "the same write after note_mutation () is accounted for"
    (count (table ^ "let add v =\n  note_mutation ();\n  Hash_set.add seen v") = 0);
  p "a second, unnoted writer beside a noted one is refused"
    (count
       (table
      ^ "let add v =\n  note_mutation ();\n  Hash_set.add seen v\nlet sneak v = Hash_set.add seen v"
       )
    = 1);
  p "a note in an unrelated function does not account for the write"
    (count (table ^ "let touch () = note_mutation ()\nlet add v = Hash_set.add seen v") = 1);
  p "a shadowing local note_mutation is not the real one"
    (count
       (table
      ^ "let add v =\n  let note_mutation () = () in\n  note_mutation ();\n  Hash_set.add seen v")
    = 1);
  p "Row.note_mutation () is the real one from another module"
    (count
       "let seen = Hash_set.create (module Int)\n\
        let add v =\n\
       \  Row.note_mutation ();\n\
       \  Hash_set.add seen v"
    = 0);
  p "Row.note_mutation () through a locally bound Row is not the real one"
    (count
       "let seen = Hash_set.create (module Int)\n\
        let add v =\n\
       \  let module Row = struct let note_mutation () = () end in\n\
       \  Row.note_mutation ();\n\
       \  Hash_set.add seen v"
    = 1);
  p "Row.note_mutation () under a module Row the file binds is not the real one"
    (count
       "module Row = struct let note_mutation () = () end\n\
        let seen = Hash_set.create (module Int)\n\
        let add v =\n\
       \  Row.note_mutation ();\n\
       \  Hash_set.add seen v"
    = 1);
  p "a bare note_mutation () that resolves to nothing in this file is not the real one"
    (count
       "let seen = Hash_set.create (module Int)\n\
        let add v =\n\
       \  note_mutation ();\n\
       \  Hash_set.add seen v"
    = 1);
  p "find_or_add with the note in its ~default closure is accounted for"
    (count
       "let note_mutation () = ()\n\
        let cache = Hashtbl.Poly.create ()\n\
        let get k = Hashtbl.find_or_add cache k ~default:(fun () -> note_mutation (); k)"
    = 0);
  p "find_or_add with no note is refused"
    (count
       "let note_mutation () = ()\n\
        let cache = Hashtbl.Poly.create ()\n\
        let get k = Hashtbl.find_or_add cache k ~default:(fun () -> k)"
    = 1);
  p "a write inside an anonymous closure of a noting function is accounted for"
    (count
       (table
      ^ "let add_all vs =\n  note_mutation ();\n  List.iter vs ~f:(fun v -> Hash_set.add seen v)")
    = 0);
  p "a write inside a nested let-bound function is judged by that function"
    (count
       (table
      ^ "let add_all vs =\n\
        \  note_mutation ();\n\
        \  let add v = Hash_set.add seen v in\n\
        \  List.iter vs ~f:add")
    = 1);
  p "a write under let%ext is judged like any other"
    (count (table ^ "let%debug_sexp add (v : int) : unit = Hash_set.add seen v") = 1);
  (* Refs. *)
  let counter = "let note_mutation () = ()\nlet hits = ref 0\n" in
  p ":= on an unnoted path is refused" (count (counter ^ "let reset () = hits := 0") = 1);
  p "incr, Int.incr and Stdlib.incr are writes"
    (count (counter ^ "let a () = incr hits\nlet b () = Int.incr hits\nlet c () = Stdlib.incr hits")
    = 3);
  p "reading a ref is not a write" (count (counter ^ "let get () = !hits") = 0);
  p "a ref given to a helper is a use the scan does not classify"
    (says (counter ^ "let bump () = Ref.replace hits (fun x -> x + 1)") "does not classify");
  p "an array element assignment is Array.set, a write"
    (count
       "let note_mutation () = ()\nlet slots = Array.create ~len:4 0\nlet put i = slots.(i) <- 1"
    = 1);
  (* What is not the table. *)
  p "a function-local table is not module-level state"
    (count "let f () =\n  let t = Hash_set.create (module Int) in\n  Hash_set.add t 1" = 0);
  p "a parameter spelled like the table is not the table"
    (count (table ^ "let add seen v = Hash_set.add seen v") = 0);
  p "comments and string literals are not code"
    (count (table ^ "(* Hash_set.add seen 1 *)\nlet s = \"Hash_set.add seen 1\"") = 0);
  (* Unclassified uses and unread state. *)
  p "a container function outside the lists is refused"
    (says (table ^ "let f () = Hash_set.copy seen") "does not classify");
  p "the table escaping to a caller is refused"
    (says (table ^ "let get () = seen") "does not classify");
  p "a write at module initialization is refused"
    (says (table ^ "let () = Hash_set.add seen 1") "at module initialization");
  p "state hidden under a local let of a module-level value is refused"
    (says "let next =\n  let c = ref 0 in\n  fun () ->\n    incr c;\n    !c"
       "hidden under a local let");
  p "hidden state on an exempt binding is admitted"
    (count
       "let next =\n\
       \  let c = ref 0 in\n\
       \  fun () ->\n\
       \    incr c;\n\
       \    !c\n\
        [@@mutation_serial.exempt \"a fresh-id counter never restored\"]"
    = 0);
  p "state inside a nested module is refused"
    (says "module M = struct\n  let t = ref 0\nend" "inside a nested module");
  p "an allocation by a constructor outside the closed list is refused"
    (says "let t = Weak.create 4" "not a constructor the scan reads");
  p "a mutable allocation bound by a pattern is refused"
    (says "let a, b = (ref 0, ref 1)" "bound by a pattern");
  p "a table the interface exports is refused"
    (says ~interface:"val seen : int Base.Hash_set.t" table "exported by the interface");
  p "an interface exporting only functions is fine"
    (count ~interface:"val add : int -> unit"
       (table ^ "let add v =\n  note_mutation ();\n  Hash_set.add seen v")
    = 0);
  (* Classifications. *)
  p "an exempt binding's unnoted writes are admitted"
    (count
       "let uid = ref 0 [@@mutation_serial.exempt \"a fresh-id counter never restored\"]\n\
        let fresh () = Int.incr uid"
    = 0);
  p "an exemption no write needs is stale"
    (says
       "let note_mutation () = ()\n\
        let uid = ref 0 [@@mutation_serial.exempt \"a fresh-id counter never restored\"]\n\
        let fresh () =\n\
       \  note_mutation ();\n\
       \  Int.incr uid"
       "stale exemption on uid");
  let fingerprinted read =
    "let state = ref 0 [@@mutation_serial.fingerprinted]\n\
     let set v = state := v\n\
     let mutated_since fp = " ^ read
  in
  p "a fingerprinted ref that mutated_since reads needs no note"
    (count (fingerprinted "fp <> !state") = 0);
  p "a fingerprinted ref that mutated_since does not read is refused"
    (says (fingerprinted "fp <> 0") "mutated_since does not read !state");
  p "only a ref can be fingerprinted"
    (says
       "let t = Hash_set.create (module Int) [@@mutation_serial.fingerprinted]\n\
        let mutated_since () = Hash_set.length t"
       "only a ref can be fingerprinted");
  p "a boundary function's writes are admitted"
    (count
       (table
      ^ "let reset () = Hash_set.clear seen\n[@@mutation_serial.boundary \"the session reset\"]")
    = 0);
  p "a boundary whose function needs none is stale"
    (says
       (table
      ^ "let reset () =\n\
        \  note_mutation ();\n\
        \  Hash_set.clear seen\n\
         [@@mutation_serial.boundary \"the session reset\"]")
       "stale boundary on reset");
  p "a boundary only excuses its own function"
    (count
       (table
      ^ "let reset () = ()\n\
         [@@mutation_serial.boundary \"the session reset\"]\n\
         let add v = Hash_set.add seen v")
    = 2);
  (* The attribute grammar. *)
  p "a one-word reason is malformed"
    (says "let uid = ref 0 [@@mutation_serial.exempt \"counter\"]\nlet f () = incr uid"
       "malformed attribute");
  p "an unknown attribute name is malformed"
    (says "let uid = ref 0 [@@mutation_serial.exmpt \"a fresh counter\"]\nlet f () = incr uid"
       "is not one of");
  p "fingerprinted takes no payload"
    (says
       "let s = ref 0 [@@mutation_serial.fingerprinted \"physically compared\"]\n\
        let mutated_since () = !s"
       "takes no payload");
  p "an attribute on something it does not classify is refused"
    (says "let f x = x [@@mutation_serial.exempt \"a fresh counter\"]" "attached to nothing");
  p "two classifications on one binding are refused"
    (says
       "let uid = ref 0 [@@mutation_serial.exempt \"a fresh counter\"] \
        [@@mutation_serial.fingerprinted]\n\
        let f () = incr uid"
       "mutation_serial attributes on one binding");
  (* Failure routing. *)
  p "a source that does not parse is the input's fault, not the scanner's"
    (match Test_utils.Scan_attempt.run (fun () -> Scan.scan ~file:"x.ml" "let = (") with
    | Unparsed _ -> true
    | Scanned _ | Raised _ -> false)
