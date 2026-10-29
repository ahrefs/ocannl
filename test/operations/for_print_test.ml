(* Printing a node that has no buffer in the printing context (gh-ocannl-333 AC 5, reworked by
   gh-ocannl-777).

   [Train.printf] used to compile a [for_print = t + 0] copy through [forward_once] and register it
   as a proxy; any failure fell back to a placeholder. Printing now reads through [Context.to_host],
   which recomputes a virtual node from the computation its lineage stored, and the printer asks
   [Context.observability] first, so an unreadable node is shown with the reason.

   Four legs: - [t] ([range_of_shape], produced by an on-device fetch, not host-initialized) is
   printed before any routine computes it: no routine of the context's lineage stored its
   computation, so it prints [<not-computed>]. (The retired copy trick ran [t]'s forward code -- and
   parameter initialization -- as a side effect of printing.) - Once a routine consuming [t] ran,
   [t] is virtual in that lineage and printing recomputes its values [0 1 2 3], which the .expected
   file pins: a placeholder there would fail the diff. - A node depending on [Local] scratch prints
   the reason, not a generic "not in context". - A failure of a read the query approved propagates
   out of the printer, which used to swallow every exception into a placeholder. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let t = TDSL.range_of_shape ~output_dims:[ 4 ] () in
  p "not in context before print" (not (Context.mem ctx t.Tensor.value));
  Train.printf ~with_grad:false ~with_code:false ctx t;
  let%op y = t *. 2 in
  let ctx = Train.forward_once ctx y in
  p "t is virtual in the lineage that computed y"
    (match Ir.Tnode.Placements.get (Context.placements ctx) t.value with
    | Some (Ir.Tnode.Virtual, _) -> true
    | _ -> false);
  Train.printf ~with_grad:false ~with_code:false ctx t;
  Train.printf_tree ~with_grad:false ctx y;
  Stdio.print_endline "";
  (* A Local dependency: [h]'s reduction extent trips the inline-reduction cap. *)
  let inner = 32 in
  let x =
    TDSL.ndarray (Array.init inner ~f:Float.of_int) ~label:[ "fp_x" ] ~output_dims:[ inner ] ()
  in
  let a =
    TDSL.ndarray
      (Array.init (2 * inner) ~f:(fun n -> Float.of_int ((n / inner) + 1)))
      ~label:[ "fp_a" ] ~input_dims:[ inner ] ~output_dims:[ 2 ] ()
  in
  let%op h = a * x in
  let%op u = h + 3 in
  let%op out = u *. h in
  let lctx = Train.forward_once ctx out in
  Train.printf ~with_grad:false ~with_code:false ~style:`Inline lctx u;
  (* A read the query approves, failing in the transfer: the printer must not swallow it. *)
  let propagated =
    Ir.Resource_fault_injection.with_callback
      (function Ir.Resource_fault_injection.To_host_before_copy -> failwith "injected" | _ -> ())
      ~f:(fun () ->
        match Train.printf ~with_grad:false ~with_code:false lctx out with
        | () -> false
        | exception Failure msg -> String.equal msg "injected")
  in
  p "a failing read of a stored node propagates out of the printer" propagated
