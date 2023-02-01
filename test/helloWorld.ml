open Base
open Ocannl

let%expect_test "Hello World" =
  Stdio.printf "Hello World!\n";
  [%expect {| Hello World! |}]

let%expect_test "Print scalar variable term" =
  let open Operation.CLI in
  let t = FO.(!. 7.0 *. !~ "hi") in
  refresh_session ();
  print_formula ~with_code:false ~with_grad:false `Default t;
  [%expect {| |}]
