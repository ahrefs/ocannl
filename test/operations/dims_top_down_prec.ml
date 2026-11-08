(* gh-ocannl-2026: isolate layout readiness from shape closure. Ordinary dataless frontend leaves
   close inference when their dims are forced; data-backed leaves already have Specified precision.
   These record copies deliberately supply known layout dimensions without closing the shape, to
   exercise Tensor.op's guard. No kernels run: the oracle executes precision inference for values
   and gradients. *)
open Base
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module Tn = Ir.Tnode

let known_dims tn = { tn with Tn.dims = Lazy.from_val [| 2 |] }
let is_double tn = Ir.Ops.equal_prec (Lazy.force tn.Tn.storage_prec) Ir.Ops.double
let grad t = (Option.value_exn t.Tensor.diff).grad

let () =
  List.iter [ false; true ] ~f:(fun ready ->
      Verdict.case
        (if ready then "known layout" else "deferred layout")
        (fun () ->
          Tensor.unsafe_reinitialize ();
          let a =
            Ocannl.Operation.range_of_shape ~grad_spec:Tensor.Require_grad ~output_dims:[ 2 ] ()
          in
          let a =
            if ready then
              {
                a with
                value = known_dims a.value;
                diff = Option.map a.diff ~f:(fun d -> { d with grad = known_dims d.grad });
              }
            else a
          in
          p "the operand shape is open" (not a.shape.Shape.closed);
          p "layout readiness matches the case" (Bool.equal (Lazy.is_val a.value.dims) ready);
          p "value and gradient precision are unsettled"
            (not (Lazy.is_val a.value.storage_prec || Lazy.is_val (grad a).storage_prec));
          let b =
            Ocannl.Operation.range_of_shape ~grad_spec:Tensor.Require_grad ~output_dims:[ 2 ] ()
          in
          let%op t = a + b in
          Tn.update_prec t.value Ir.Ops.double;
          Tn.update_prec (grad t) Ir.Ops.double;
          p "the value takes the consumer precision" (is_double a.value);
          p "the gradient takes the consumer precision" (is_double (grad a))));
  Verdict.case "a closed operand" (fun () ->
      Tensor.unsafe_reinitialize ();
      let a =
        Ocannl.Operation.range_of_shape ~grad_spec:Tensor.Require_grad ~output_dims:[ 2 ] ()
      in
      let (_ : int array) = Lazy.force a.value.dims in
      p "forcing frontend dimensions closes the shape" a.shape.Shape.closed;
      let before = Lazy.force a.value.storage_prec in
      let%op t = a + a in
      Tn.update_prec t.value Ir.Ops.double;
      p "a closed operand keeps its committed precision"
        (Ir.Ops.equal_prec before (Lazy.force a.value.storage_prec)));
  Verdict.case "an open operand with read precision and known layout" (fun () ->
      Tensor.unsafe_reinitialize ();
      let a = TDSL.range_of_shape ~output_dims:[ 2 ] () in
      let a = { a with value = known_dims a.value } in
      let (_ : Ir.Ops.prec) = Lazy.force a.value.storage_prec in
      match
        let%op t = a + a in
        t
      with
      | exception Utils.User_error msg ->
          p "known layout does not hide a forbidden precision update"
            (String.is_substring msg ~substring:"already been forced")
      | _ -> p "known layout does not hide a forbidden precision update" false)
