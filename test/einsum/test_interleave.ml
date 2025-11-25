open Base
open Ocannl
open Nn_blocks.DSL_modules

let () =
  Tensor.unsafe_reinitialize ();
  let%op t1 = [ 1.0; 2.0; 3.0 ] in
  let%op t2 = [ 4.0; 5.0; 6.0 ] in
  let t3 = Operation.interleave t1 t2 () in

  (* t3 should be [1.0; 4.0; 2.0; 5.0; 3.0; 6.0] *)
  let ctx = Context.auto () in

  try
    let _ctx = Train.forward_once ctx t3 in
    Stdio.printf "Test failed! Expected error was not raised.\n";
    Stdlib.exit 1
  with Utils.User_error msg ->
    if
      String.equal msg
        "Defined_by_cd_logic: use explicit ~logic annotations when defining this operation"
    then Stdio.printf "Test passed! Caught expected error: %s\n" msg
    else (
      Stdio.printf "Test failed! Caught unexpected error: %s\n" msg;
      Stdlib.exit 1)
