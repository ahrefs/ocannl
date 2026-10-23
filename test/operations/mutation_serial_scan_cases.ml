(* gh-ocannl-1506: the mutation-serial reader's rules, put to synthesized sources -- each form it
   must refuse beside the nearest legitimate code it must accept. The shipping scanner's own
   negative control is mutation_serial_scan_control, over a committed fixture tree. *)
open Base
open Verdict.Claims
module Scan = Test_utils.Mutation_serial_scan

(* Most cases stand in for row.ml, the serial's own module, whose bare [note_mutation] is the real
   one; [~serial_owner:false] stands in for any other. *)
let findings ?(serial_owner = true) ?interface source =
  Test_utils.Scan_attempt.attempted ~what:"mutation-serial cases" ~name:(String.prefix source 60)
    ~default:[ "<the scan failed>" ] (fun () ->
      (Scan.scan ~file:"x.ml" ~serial_owner ?interface ("open Base\n" ^ source)).findings)

let count ?serial_owner ?interface source = List.length (findings ?serial_owner ?interface source)

(* The finding's message, without its [file:line: ] prefix. *)
let says ?serial_owner ?interface source fragment =
  List.exists (findings ?serial_owner ?interface source) ~f:(fun f ->
      String.is_substring f ~substring:fragment)

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
  p "outside the serial's own module, a top-level note_mutation is not the real one"
    (count ~serial_owner:false (table ^ "let add v =\n  note_mutation ();\n  Hash_set.add seen v")
    = 1);
  p "a container function through a locally bound Hash_set is not classified"
    (says
       (table
      ^ "let probe v =\n\
        \  let module Hash_set = struct let mem t v = Base.Hash_set.add t v; true end in\n\
        \  Hash_set.mem seen v")
       "does not classify");
  p "an assignment through a shadowing ( := ) is not classified"
    (says "let hits = ref 0\nlet ( := ) r v = ignore (r, v)\nlet reset () = hits := 0"
       "does not classify");
  p "state in an included structure is refused"
    (says "include struct\n  let cache = ref 0\nend\nlet bump () = incr cache"
       "an included structure");
  p "an unlisted constructor hidden under a local let is refused"
    (says "let next =\n  let w = Weak.create 4 in\n  fun () -> Weak.length w"
       "hidden under a local let");
  p "two same-named fingerprinted refs are judged apart"
    (says
       "let state = ref 0 [@@mutation_serial.fingerprinted]\n\
        let set_early v = state := v\n\
        let state = ref 1 [@@mutation_serial.fingerprinted]\n\
        let fingerprint () = { fp_state = !state }\n\
        let mutated_since fp = not (phys_equal fp.fp_state !state)"
       "mutated_since does not compare !state");
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
  p "a note inside an uncalled nested helper does not account for the enclosing write"
    (count
       (table
      ^ "let outer () =\n\
        \  let note_only () = note_mutation () in\n\
        \  ignore note_only;\n\
        \  Hash_set.add seen 1")
    = 1);
  p "a partial application of a mutator is refused"
    (says
       (table ^ "let make_adder () =\n  note_mutation ();\n  Hash_set.add seen")
       "partial application of a mutator");
  p "an annotated binding head is tracked, not refused as hidden state"
    (count "let hits : int ref = ref 0\nlet get () = !hits" = 0);
  p "a type-constrained function invoked on the spot is entered too"
    (says "let next = ((fun () -> let c = ref 0 in fun () -> incr c) : unit -> unit -> unit) ()"
       "hidden under a local let");
  p "a closure that escapes a noting function is judged on its own"
    (says
       (table ^ "let make_writer () =\n  note_mutation ();\n  fun v -> Hash_set.add seen v")
       "a closure escaping make_writer writes seen");
  p "state a module-level initializer creates through a local helper is refused"
    (says
       "let next =\n\
       \  let make () =\n\
       \    let c = ref 0 in\n\
       \    fun () -> Int.incr c\n\
       \  in\n\
       \  make ()"
       "hidden under a local let");
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
  p "a table exported through an included signature is refused"
    (says ~interface:"include sig\n  val seen : int Base.Hash_set.t\nend" table
       "exported by the interface");
  p "an interface include of a module type the scan cannot read is refused"
    (says ~interface:"include Some_sig" table "does not read");
  p "state an initializer creates in a function it invokes on the spot is refused"
    (says "let next = (fun () -> let c = ref 0 in fun () -> incr c; !c) ()"
       "hidden under a local let");
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
     let fingerprint () = { fp_state = !state }\n\
     let mutated_since fp = " ^ read
  in
  p "a fingerprinted ref mutated_since compares physically needs no note"
    (count (fingerprinted "not (phys_equal fp.fp_state !state)") = 0);
  p "a fingerprinted ref that mutated_since does not read is refused"
    (says (fingerprinted "fp <> 0") "mutated_since does not compare !state");
  p "a fingerprinted ref that mutated_since reads but does not compare is refused"
    (says (fingerprinted "ignore !state; fp <> 0") "mutated_since does not compare !state");
  p "a fingerprinted ref compared with itself is refused"
    (says (fingerprinted "not (phys_equal !state !state)") "mutated_since does not compare !state");
  p "a fingerprinted ref compared with a field fingerprint does not capture is refused"
    (says
       (fingerprinted "not (phys_equal fp.fp_other !state)")
       "mutated_since does not compare !state");
  p "a capture that is not the record fingerprint returns does not count"
    (says
       "let state = ref 0 [@@mutation_serial.fingerprinted]\n\
        let set v = state := v\n\
        let fingerprint () =\n\
       \  ignore { fp_state = !state };\n\
       \  { fp_state = 0 }\n\
        let mutated_since fp = not (phys_equal fp.fp_state !state)"
       "mutated_since does not compare !state");
  p "a comparison that does not reach mutated_since's result does not count"
    (says
       (fingerprinted "ignore (phys_equal fp.fp_state !state);\n  false")
       "mutated_since does not compare !state");
  p "a comparison inside a disjunction of the result counts"
    (count (fingerprinted "fp.serial <> 0 || not (phys_equal fp.fp_state !state)") = 0);
  p "a fingerprinted ref compared through a shadowing phys_equal is refused"
    (says
       ("let phys_equal _ _ = true\n" ^ fingerprinted "not (phys_equal fp.fp_state !state)")
       "mutated_since does not compare !state");
  p "only a ref can be fingerprinted"
    (says
       "let t = Hash_set.create (module Int) [@@mutation_serial.fingerprinted]\n\
        let mutated_since () = phys_equal t t"
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
        let mutated_since () = phys_equal 0 !s"
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
