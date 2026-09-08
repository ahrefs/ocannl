open! Base
open Ocannl
open Nn_blocks.DSL_modules

(* The per-iteration trace line a loop emits under [debug_log_from_routines] names the loop index,
   so its printf conversion has to match the width of the backend's loop index type: 32-bit
   normally, 64-bit under [large_models]. Passing a 64-bit index to [%d] is a variadic type
   mismatch, so the goldens pin the emitted line at both widths -- the narrow one to keep the
   ordinary spelling honest, the wide one because that is the spelling that was wrong.

   The wide golden also shows the same mismatch still standing one line below, on the array-offset
   conversions of the value-log statement ([d[%u]] against an [int64_t] argument). That is
   gh-ocannl-953, deliberately out of scope here; the golden records it, it does not bless it. *)
let () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let%op d = { p = [ 1.0; 2.0; 3.0; 4.0 ] } + { q = [ 5.0; 6.0; 7.0; 8.0 ] } in
  let _ctx = Train.forward_once ctx d in
  ()
