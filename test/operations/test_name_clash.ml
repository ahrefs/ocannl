open Ocannl.Operation.DSL_modules

(* These tests verify that the code compiles without errors when there are no name clashes *)

let%expect_test "test_op_single_definition" =
  let open TDSL in
  (* Normal case - should work fine *)
  let x = [%op { x = uniform () }] in
  let _ = Tensor.shape x in
  Stdio.printf "Test passed - single definition works\n";
  [%expect {| Test passed - single definition works |}]

let%expect_test "test_cd_single_definition" =
  let open NTDSL in
  (* Normal case - should work fine *)
  let comp = [%cd { x }] in
  let _ = Ir.Assignments.noop comp in
  Stdio.printf "Test passed - single inline definition works\n";
  [%expect {| Test passed - single inline definition works |}]

let%expect_test "test_einsum_single_capture" =
  let open TDSL in
  let a = param "a" ~input_dims:[ 3; 2 ] () in
  let b = param "b" ~input_dims:[ 2; 4 ] () in
  (* Normal case - should work fine *)
  let c = [%op a +* "ab; bc => ac" ["d"] b] in
  let _ = Tensor.shape c in
  Stdio.printf "Test passed - einsum with single capture works\n";
  [%expect {| Test passed - einsum with single capture works |}]