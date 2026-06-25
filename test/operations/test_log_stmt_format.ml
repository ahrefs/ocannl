open! Base
open Ocannl
open Nn_blocks.DSL_modules

(* Regression test for gh-ocannl-179: fprintf/printf log statements in generated C/CUDA/Metal source
   should wrap cleanly (arguments indented consistently) rather than breaking at arbitrary mid-call
   positions. Run with --ocannl_debug_log_from_routines=true to produce a source file that contains
   fprintf calls; the expected file captures the formatted output after the fix. *)
let () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let%op c = { a = [ 1.0 ] } + { b = [ 1.0 ] } in
  let _ctx = Train.forward_once ctx c in
  ()
