open Base
module Train = Ocannl.Train
open Ocannl.Operation.DSL_modules

let () =
  Tensor.unsafe_reinitialize ();
  let constant =
    NTDSL.init ~l:"embedded_constant" ~prec:Ir.Ops.single ~b:[] ~i:[] ~o:[ 4 ]
      ~f:(fun idx -> Float.of_int (idx.(0) + 1))
      ()
  in
  let%op result = constant + 1.0 in
  let routine = Train.to_routine (Context.auto ()) Train.IDX.empty (Train.forward result) in
  let ctx = Context.run (Context.context routine) routine in
  let actual = Context.get_values ctx result.Tensor.value in
  let expected = [| 2.; 3.; 4.; 5. |] in
  if not (Array.equal Float.equal actual expected) then
    failwith
      (Printf.sprintf "expected [%s], got [%s]"
         (String.concat ~sep:"; " (Array.to_list @@ Array.map expected ~f:Float.to_string))
         (String.concat ~sep:"; " (Array.to_list @@ Array.map actual ~f:Float.to_string)))
