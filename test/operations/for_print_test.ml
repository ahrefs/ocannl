(* Regression test for gh-ocannl-333 AC 5: the [%cd "for_print" =: t] trick.

   [t] here is a [range_of_shape] tensor — its value is produced by an on-device fetch (NOT an
   ndarray-backed literal, so it is not in the Host_inits table), and we deliberately never run it,
   so its node is absent from the context. Printing / reading it through [Train.printf] must still
   yield real values: [printf] recompiles a copy of [t] into the context (the for-print route) and
   registers it as a proxy that [Context.to_host] reads through.

   The printed values [0 1 2 3] below come from the for-print recompile. If the for-print
   compilation/run is removed, the printer can only fall back to a placeholder, so the rendered row
   would read "<not-in-context>" instead of the numbers — the .expected diff then fails. This is
   thus non-vacuous: it requires the recompute, not merely a traversal of the to_host path. *)

open Ocannl
open Ocannl.Operation.DSL_modules

let () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let t = TDSL.range_of_shape ~output_dims:[ 4 ] () in
  (* Harness condition: t has never been run in ctx, so its node is not materialized there. *)
  Stdio.printf "in context before print: %b\n" (Context.mem ctx t.Tensor.value);
  (* The for-print route recompiles t into ctx so its real values are shown. *)
  Train.printf ~with_grad:false ~with_code:false ctx t
