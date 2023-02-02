open Base
open Ocannl

let%expect_test "Hello World" =
  Stdio.printf "Hello World!\n";
  [%expect {| Hello World! |}]

let%expect_test "Pointwise multiplication" =
  Random.init 0;
  let open Operation.CLI in
  (* Hey is inferred to be a scalar. *)
  let t = FO.(!. 2.0  *. !~ "hey") in
  refresh_session ();
  print_formula ~with_code:false ~with_grad:false `Default t;
  [%expect {|
    [4] heyv2p: 1
    dims: (1)
    0=0   -1.334558 |}]

let%expect_test "Matrix multiplication" =
  Operation.drop_session();
  Random.init 0;
  let open Operation.CLI in
  (* Hey is inferred to be a matrix. *)
  let hey = FO.(!~ "hey") in
  let t = FO.(number ~axis_label:"q" 2.0 * hey + number ~axis_label:"p" 1.0) in
  refresh_session ();
  print_formula ~with_code:false ~with_grad:false `Default hey;
  [%expect {| |}];
  print_formula ~with_code:false ~with_grad:false `Default t;
  [%expect {| |}]
