open Base
module Train = Ocannl.Train
open Ocannl.Operation.DSL_modules

let () =
  Tensor.unsafe_reinitialize ();
  let expected_constant = Array.init 20 ~f:(fun i -> Float.of_int (i + 1)) in
  let constant =
    NTDSL.init ~l:"embedded_constant" ~prec:Ir.Ops.single ~b:[] ~i:[] ~o:[ 20 ]
      ~f:(fun idx -> Float.of_int (idx.(0) + 1))
      ()
  in
  let ctx = Context.auto () in
  if not (Ir.Host_inits.mem constant.Tensor.value) then
    failwith "expected NTDSL.init tensor to be registered in Host_inits";
  if Context.mem ctx constant.Tensor.value then
    failwith "embedded constant unexpectedly had a device buffer before compilation";
  let%op result = constant + 1.0 in
  let routine = Train.to_routine ctx Train.IDX.empty (Train.forward result) in
  let ctx = Context.context routine in
  if not (Context.mem ctx constant.Tensor.value) then
    failwith "embedded constant was not allocated in the linked routine context";
  let copied = Context.get_values ctx constant.Tensor.value in
  if not (Array.equal Float.equal copied expected_constant) then
    failwith "embedded constant data was not copied into the linked routine context";
  let ctx = Context.run ctx routine in
  let actual = Context.get_values ctx result.Tensor.value in
  let expected = Array.map expected_constant ~f:(( +. ) 1.0) in
  if not (Array.equal Float.equal actual expected) then
    failwith
      (Printf.sprintf "expected [%s], got [%s]"
         (String.concat ~sep:"; " (Array.to_list @@ Array.map expected ~f:Float.to_string))
         (String.concat ~sep:"; " (Array.to_list @@ Array.map actual ~f:Float.to_string)))
