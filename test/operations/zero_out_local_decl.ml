(* Regression test for gh-ocannl-420: a legitimately-needed [Zero_out] on a *local* (non-virtual,
   non-materialized) tensor node must not be double-zeroed. The node's C declaration already gets [=
   {0}], so the explicit zeroing loop is redundant and elided.

   [sparse = input ++ "i=>i0"] is a non-surjective einsum (only column 0 is written), so it
   genuinely needs zero-initialization for the other columns. Consuming [sparse] twice keeps it from
   being virtualized, leaving it as a [Local] node -- exactly the case Step 2 of the fix targets.
   The generated code should show [sparse[...] = {0}] in the declaration but no [sparse[...] = 0;]
   zeroing loop in the main logic. *)

module Train = Ocannl.Train
open Ocannl.Nn_blocks.DSL_modules

let () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let input = TDSL.range 4 in
  let%op sparse = input ++ "i=>i0" in
  let%op out = (sparse ++ "i2=>i") *. (sparse ++ "i2=>i") in
  Ocannl.Train.set_materialized out.value;
  let ctx = Ocannl.Train.forward_once ctx out in
  Train.printf_tree ctx out
