(* Constructor precision is frontend policy. Use cc explicitly to execute double values whose low
   bits distinguish ingestion at double from an accidental single-precision host buffer. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let constructors =
  [
    ("Tensor", fun ~prec values ~output_dims () -> Tensor.ndarray ~prec values ~output_dims ());
    ("TDSL", fun ~prec values ~output_dims () -> TDSL.ndarray ~prec values ~output_dims ());
    ("NTDSL", fun ~prec values ~output_dims () -> NTDSL.ndarray ~prec values ~output_dims ());
  ]

let () =
  List.iter constructors ~f:(fun (name, ndarray) ->
      List.iter [ 8; 64 ] ~f:(fun n ->
          Verdict.case (Printf.sprintf "%s %d cells" name n) (fun () ->
              Tensor.unsafe_reinitialize ();
              let values = Array.init n ~f:(fun i -> 1. +. (Float.of_int (i + 1) *. 0x1p-30)) in
              let x = ndarray ~prec:Ir.Ops.double values ~output_dims:[ n ] () in
              let%op y = x *. 2 in
              Train.set_materialized y.Tensor.value;
              let ctx = Train.forward_once (Context.cpu ()) y in
              p "the source stores double"
                (Ir.Ops.equal_prec (Lazy.force x.Tensor.value.Ir.Tnode.storage_prec) Ir.Ops.double);
              pf_all2 "execution preserves the double data"
                (Context.get_values ctx y.Tensor.value)
                (Array.map values ~f:(fun v -> 2. *. v))
                ~f:Float.equal;
              Context.release ctx)));
  List.iter [ false; true ] ~f:(fun read_first ->
      Verdict.case (Printf.sprintf "data precision conflict, read first %b" read_first) (fun () ->
          let x = TDSL.ndarray (Array.create ~len:64 1.) ~output_dims:[ 64 ] () in
          if read_first then ignore (Lazy.force x.Tensor.value.Ir.Tnode.storage_prec : Ir.Ops.prec);
          let message =
            match Ir.Tnode.update_prec x.Tensor.value Ir.Ops.double with
            | () -> ""
            | exception Utils.User_error message -> message
          in
          p "the refusal explains data provenance and construction-time precision"
            (String.is_substring message ~substring:"precision fixed by data at creation"
            && String.is_substring message ~substring:"~prec")));
  Verdict.case "explicit precision without data" (fun () ->
      let x = TDSL.ndarray ~prec:Ir.Ops.single [| 1. |] ~output_dims:[ 1 ] () in
      let message =
        match Ir.Tnode.update_prec x.Tensor.value Ir.Ops.double with
        | () -> ""
        | exception Utils.User_error message -> message
      in
      p "the refusal does not attribute an explicit precision to data"
        (String.is_substring message ~substring:"already set"
        && not (String.is_substring message ~substring:"fixed by data")))
