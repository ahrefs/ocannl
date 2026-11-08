(* The stress gate itself must reclaim a dropped forward-only consumer: there is no explicit
   collection or reclamation witness in this fixture. Run only with debug_gc_backprop_roots=true. *)
open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims
module LC = Lineage_claims

let[@inline never] forward_only ctx w =
  let%op c = w *. 2. in
  Train.forward_once ctx c

let () =
  p "the GC stress mode is enabled"
    (Utils.get_global_flag ~default:false ~arg_name:"debug_gc_backprop_roots");
  let w = TDSL.param ~values:[| 1.; 2.; 3. |] "gc_gate_w" ~output_dims:[ 3 ] () in
  let ctx = forward_only (Context.cpu ()) w in
  let%op l = w *. w in
  let update = Train.grad_update l in
  p "the next consumer owns the reclaimed leaf's gradient reset"
    (Set.mem update.zeroes (LC.grad_of w));
  let ctx, step = Train.to_routine ctx Train.IDX.empty update in
  let ctx = Context.run ctx step in
  LC.exact "the gradient is 2 w" (Context.get_values ctx (LC.grad_of w)) [| 2.; 4.; 6. |]
