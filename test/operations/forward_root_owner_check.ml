(* gh-ocannl-1899: [debug_check_forward_root_owners], the live-root scan held against the owner
   cells at every [consume_forward_code] that decides a handout. A missing owner edge is otherwise
   silent: the cells report no conflict and the reader of an unconsumed root's node is handed out.
   The negative control drops an edge from a held graph, where the scan still sees the root. The
   dune stanza runs this twice, as the configuration says (the check on, in test/config, which is
   what these claims hold it to) and with the check turned off on the command line. Graphs only:
   nothing is lowered or run. *)
open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims
module Asgns = Ir.Assignments

let check_on = Utils.get_global_flag ~default:false ~arg_name:"debug_check_forward_root_owners"

let leaf name =
  Tensor.term_init [| 1.; 2.; 3. |] ~label:[ name ] ~output_dims:[ 3 ] ~grad_spec:Prohibit_grad ()

let name_of (t : Tensor.t) = Ir.Tnode.label_name t.value
let has msg sub = String.is_substring msg ~substring:sub

(* What one [consume_forward_code] came to. *)
type outcome = Handed_out | Refused of string | Check_failed of string

let consume t =
  match Tensor.consume_forward_code t with
  | (_ : Asgns.comp) -> Handed_out
  | exception Tensor.Session_error (msg, _) -> Refused msg
  | exception Failure msg -> Check_failed msg

let is_handed_out = function Handed_out -> true | Refused _ | Check_failed _ -> false

let refused_over outcome ~root =
  match outcome with
  | Refused msg -> has msg "found conflicting roots" && has msg root
  | Handed_out | Check_failed _ -> false

let check_named outcome ~root =
  match outcome with
  | Check_failed msg -> has msg "debug_check_forward_root_owners" && has msg root
  | Handed_out | Refused _ -> false

(* [owner] embeds [shared]'s code, [sibling] reads [shared] computed. *)
let trio prefix =
  let shared = NTDSL.O.relu ~label:[ prefix ^ "_shared" ] (leaf (prefix ^ "_x")) in
  let owner = NTDSL.O.relu ~label:[ prefix ^ "_owner" ] shared in
  let sibling = NTDSL.O.relu ~label:[ prefix ^ "_sibling" ] shared in
  (shared, owner, sibling)

let names = List.equal String.equal
let by_cells = Tensor.For_testing.forward_conflicts_by_cells
let by_scan = Tensor.For_testing.forward_conflicts_by_live_roots

let () =
  Stdio.printf "the check is %s\n" (if check_on then "on" else "off");
  case "intact owner edges pass, and the refusal stands" (fun () ->
      let _shared, owner, sibling = trio "intact" in
      p "intact: the cells and the scan name the owner"
        (names (by_cells sibling) [ name_of owner ] && names (by_scan sibling) [ name_of owner ]);
      p "intact: the sibling is refused over the owner"
        (refused_over (consume sibling) ~root:(name_of owner));
      p "intact: the owner is handed out" (is_handed_out (consume owner));
      p "intact: then the sibling is" (is_handed_out (consume sibling)));
  case "a dropped owner edge leaves the cells blind to a root the scan finds" (fun () ->
      let shared, owner, sibling = trio "dropped" in
      Tensor.For_testing.drop_owner_edges shared;
      p "dropped: the cells name no root, the scan names the owner"
        (names (by_cells sibling) [] && names (by_scan sibling) [ name_of owner ]);
      let outcome = consume sibling in
      if check_on then
        p "dropped: the check fails the consume, naming the owner"
          (check_named outcome ~root:(name_of owner))
      else p "dropped: unchecked, the reader is handed out" (is_handed_out outcome));
  case "the check runs ahead of a refusal the cells do decide" (fun () ->
      let shared1, owner1, _ = trio "seen" and shared2, owner2, _ = trio "missed" in
      let reader = NTDSL.O.(shared1 + shared2) in
      Tensor.For_testing.drop_owner_edges shared2;
      p "partial: the cells name one owner, the scan both"
        (names (by_cells reader) [ name_of owner1 ]
        && names
             (List.sort (by_scan reader) ~compare:String.compare)
             (List.sort [ name_of owner1; name_of owner2 ] ~compare:String.compare));
      let outcome = consume reader in
      if check_on then
        p "partial: the check fails the consume, naming the missed owner"
          (check_named outcome ~root:(name_of owner2)
          && not (check_named outcome ~root:(name_of owner1)))
      else
        p "partial: unchecked, the reader is refused over the other owner alone"
          (refused_over outcome ~root:(name_of owner1)
          && not (refused_over outcome ~root:(name_of owner2))))
