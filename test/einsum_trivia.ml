open Base
open Ocannl

module FDSL = Operation.FDSL

let () = Session.SDSL.set_executor OCaml

let%expect_test "einsum1 permute axes" =
  let open Session.SDSL in
  drop_all_sessions();
  Random.init 0;
  let hey =
    FDSL.range_of_shape ~batch_dims:[2] ~input_dims:[3] ~output_dims:[4] () in
  let%nn_op ho = hey++"b|i->o => o|b->i" in
  refresh_session ();
  print_formula ~with_code:false ~with_grad:false `Default @@ hey;
  [%expect {| |} ];
  print_formula ~with_code:false ~with_grad:false `Default @@ ho;
  [%expect {| |} ]
