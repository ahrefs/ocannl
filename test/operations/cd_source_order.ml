(* gh-ocannl-1770: [%cd] builds a statement sequence in source order. A statement naming a
   forward-root tensor embeds that tensor's forward code ([Tensor.take_forward_code]), and the
   statements run in list order, so the FIRST statement naming the root has to be the one that
   embeds it. The sequence used to be a list literal, which OCaml evaluates right to left: the LAST
   statement embedded the root, and every earlier statement read it before it was computed -- a
   silent wrong value.

   The probe is [x = 1;2;3], [a = x *. x], then statements [pp =: a + 1; qq =: a + 2; rr =: a + 3].
   Every value is a small integer, exact by construction, compared bitwise. Built right to left,
   [pp] and [qq] read the uninitialized [a] (zeros): [pp = 1;1;1], [qq = 2;2;2]. *)
open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims
module IDX = Train.IDX
module Asgns = Ir.Assignments

let exact name got want = p_close name got want ~tol:(fun _ -> 0.)
let has msg substring = String.is_substring msg ~substring

let squares prefix =
  let x = TDSL.ndarray [| 1.; 2.; 3. |] ~label:[ prefix ^ "_x" ] ~output_dims:[ 3 ] () in
  let%op a = x *. x in
  a

(* The positions, among the leaves of [comp]'s assignments, of the leaves writing [tn]. *)
let writes_of comp tn =
  let _, positions =
    Asgns.fold_leaves comp.Asgns.asgns ~init:(0, []) ~f:(fun (i, acc) leaf ->
        match leaf with
        | Asgns.Accum_op { lhs; _ } when Ir.Tnode.equal lhs tn -> (i + 1, i :: acc)
        | _ -> (i + 1, acc))
  in
  List.rev positions

let run_values comp tns =
  List.iter tns ~f:Train.set_materialized;
  let ctx, routine = Train.to_routine (Context.auto ()) IDX.empty comp in
  let ctx = Context.run ctx routine in
  List.map tns ~f:(Context.get_values ctx)

let () =
  case "S: three statements reading one root" (fun () ->
      let a = squares "cso_s" in
      let%cd comp =
        { pp } =: a + 1;
        { qq } =: a + 2;
        { rr } =: a + 3
      in
      (* Structure: [a] is computed once, by the code the first statement embedded, before [pp]. *)
      let a_writes = writes_of comp a.value and pp_writes = writes_of comp pp.value in
      p "S: a is written by exactly one leaf" (List.length a_writes = 1);
      p "S: a is written before pp"
        (match (a_writes, pp_writes) with [ wa ], [ wp ] -> wa < wp | _ -> false);
      match run_values comp [ pp.value; qq.value; rr.value ] with
      | [ pp_v; qq_v; rr_v ] ->
          exact "S: pp = a + 1" pp_v [| 2.; 5.; 10. |];
          exact "S: qq = a + 2" qq_v [| 3.; 6.; 11. |];
          exact "S: rr = a + 3" rr_v [| 4.; 7.; 12. |]
      | _ -> assert false);
  (* An explicit [a.forward] first carries the root's code; the statement after it reads [a] as
     computed and embeds nothing. Built right to left, the statement embedded [a] first and the
     explicit [a.forward] was refused as a second copy. *)
  case "F: an explicit forward first" (fun () ->
      let a = squares "cso_f" in
      let%cd comp =
        a.forward;
        { pf } =: a + 1
      in
      p "F: a is written by exactly one leaf" (List.length (writes_of comp a.value) = 1);
      match run_values comp [ pf.value ] with
      | [ pf_v ] -> exact "F: pf = a + 1" pf_v [| 2.; 5.; 10. |]
      | _ -> assert false);
  (* The reverse order is the second copy: the statement embeds [a], then [a.forward] hands the same
     fragment out again. The refusal's hint names this order. *)
  case "R: an explicit forward after a statement naming the root" (fun () ->
      let a = squares "cso_r" in
      match
        [%cd
          { pr } =: a + 1;
          a.forward]
      with
      | _ -> fail "R: was accepted"
      | exception Utils.User_error msg ->
          Stdio.eprintf "R: %s (not part of the golden)\n%!" msg;
          p "R: refused as a second copy, the hint naming the order"
            (has msg "occurs in more than one member of this sequence"
            && has msg "an explicit [t.forward] after a statement naming [t]"))
