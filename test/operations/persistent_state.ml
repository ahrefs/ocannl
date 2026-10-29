(* gh-ocannl-793: an assignment that reads its own lhs without the initializing [=:] prefix -- here
   [=:] whose right-hand side names the lhs -- declares state that persists from one run of the
   routine to the next. The node needs no materialization call: the placement decider keeps it on
   the device as read-before-write. Forcing it virtual contradicts that, and the refusal names the
   node, the placement and the remedy. *)

open Base
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module Asgns = Ir.Assignments
module Tn = Ir.Tnode

let dims = 4

let setup () =
  Tensor.unsafe_reinitialize ();
  let x =
    NTDSL.init ~l:"x" ~prec:Ir.Ops.single ~o:[ dims ]
      ~f:(function [| i |] -> Float.of_int (i + 1) | _ -> assert false)
      ()
  in
  Ocannl.Train.set_materialized x.Tensor.value;
  let comp = [%cd { acc } =: (!.0.5 *. acc) + x] in
  let acc =
    Set.find_exn comp.Asgns.embedded_nodes ~f:(fun tn ->
        match tn.Tn.label with "acc" :: _ -> true | _ -> false)
  in
  (comp, acc)

let () =
  let comp, acc = setup () in
  let ctx, routine = Context.compile ~name:"pst_carry" (Context.cpu ()) comp Ir.Indexing.Empty in
  let seed = Array.init dims ~f:(fun i -> 10. *. Float.of_int (i + 1)) in
  let ctx = Context.set_values ctx acc seed in
  let ctx = Context.run ctx routine in
  let ctx = Context.run ctx routine in
  (* Two runs of acc <- acc / 2 + x from the seed: seed / 4 + 3x / 2 -- the second run reads what
     the first left, and both read past the seed only if the buffer persists. *)
  p_all2 "the accumulator carries its value across runs" (Context.get_values ctx acc)
    (Array.init dims ~f:(fun i -> (seed.(i) /. 4.) +. (1.5 *. Float.of_int (i + 1))))
    ~f:Float.equal;
  let comp, acc = setup () in
  Ocannl.Train.set_virtual acc;
  let refusal =
    match Context.compile ~name:"pst_virtual" (Context.cpu ()) comp Ir.Indexing.Empty with
    | _ -> None
    | exception Utils.User_error msg -> Some msg
  in
  let says sub = Option.exists refusal ~f:(fun msg -> String.is_substring msg ~substring:sub) in
  p "forced virtual: compilation is refused, naming the node and its placement"
    (says (Tn.label_name acc) && says "29:set-virtual");
  p "forced virtual: the refusal names the cause and the remedy"
    (says "carries state from one run of the routine to the next" && says "Train.set_materialized");
  Stdio.printf "refusal: %s\n" (Option.value refusal ~default:"<none>")
