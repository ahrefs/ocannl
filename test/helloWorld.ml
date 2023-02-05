open Base
open Ocannl

let%expect_test "Hello World" =
  Stdio.printf "Hello World!\n";
  [%expect {| Hello World! |}]

let%expect_test "Pointwise multiplication dims 1" =
  Random.init 0;
  (* Hey is inferred to be a scalar. *)
  let%ocannl y = 2 * "hey" in
  let open Operation.CLI in
  refresh_session ();
  print_formula ~with_code:false ~with_grad:false `Default @@ Network.unpack y;
  [%expect {|
    [4] heyv2p: shape 1 layout: 0:1
    │_=0
    ┼──────────────────────────────
    │-1.335 |}]

let%expect_test "Matrix multiplication dims 1x1" =
  Operation.drop_session();
  Random.init 0;
  (* Hey is inferred to be a matrix. *)
  let%ocannl hey = "hey" in
  let%ocannl y = "q" 2.0 * hey + "p" 1.0 in
  let open Operation.CLI in
  refresh_session ();
  print_formula ~with_code:false ~with_grad:false `Default @@ Network.unpack hey;
  [%expect {|
    [1] hey: shape q:1->p:1 layout: 0:1 x 1:1
    │0@p=0
    │q=1
    ┼────────────────────────────────────────
    │-0.667 |}];
  print_formula ~with_code:false ~with_grad:false `Default @@ Network.unpack y;
  [%expect {|
    [5] v1ptheyv2p: shape p:1 layout: 0:1
    │p=0
    ┼────────────────────────────────────
    │-0.335 |}]
