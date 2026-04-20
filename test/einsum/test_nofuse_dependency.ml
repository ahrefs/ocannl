open Ocannl
open Nn_blocks.DSL_modules
open Stdio

(* Regression test: sibling For_loops with a data dependency must NOT be fused.

   We create two independent tensor computations combined in one routine:
   - e = a + b  (writes to e, reads a and b)
   - g = a + e  (writes to g, reads a and e)

   The first loop writes e[i], the second loop reads e[i].
   Since g depends on e, forward(g) includes e's computation -- so they'll
   already be in the same routine but with e computed first.

   The key property: the two output loops must remain SEPARATE because g reads
   from e which e's init loop writes. The .ll must show two separate for loops
   for the outputs, not a fused one. *)

let () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let a = TDSL.range_of_shape ~label:[ "a" ] ~output_dims:[ 4 ] () in
  let b = TDSL.range_of_shape ~label:[ "b" ] ~output_dims:[ 4 ] () in
  let%op e = a + b in
  (* g reads e, so g's forward includes e's forward.
     This tests that the loops writing e and reading e for g are not fused. *)
  let%op g = a + e in
  Train.set_hosted a.value;
  Train.set_hosted b.value;
  (* Force e to be materialized so it gets its own output loop *)
  Train.set_hosted e.value;
  Train.set_hosted g.value;
  ignore (Train.forward_once ctx g);
  (* a = [0,1,2,3], b = [0,1,2,3], e = [0,2,4,6], g = [0,3,6,9] *)
  printf "g = a + e where e = a + b:\n%!";
  Tensor.print ~here:[%here] ~force:true ~with_code:false ~with_grad:false `Inline g;
  printf "\n%!"
