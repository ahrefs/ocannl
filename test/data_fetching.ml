open Base
open Ocannl
module FDSL = Operation.FDSL
module NFDSL = Operation.NFDSL
module CDSL = Code.CDSL

let () = Session.SDSL.set_executor Gccjit

let%expect_test "Synthetic data" =
  (* let open Operation.FDSL in *)
  let open Session.SDSL in
  drop_all_sessions ();
  Random.init 0;
  let%nn_dt session_step ~o:1 = n =+ 1 in
  let c_data =
    FDSL.term ~label:"fetch_callback" ~needs_gradient:false
      ~batch_dims:[ CDSL.dim 1 ]
      ~input_dims:[]
      ~output_dims:[ CDSL.dim 2; CDSL.dim 3 ]
      ~init_op:Code.Range_over_offsets
      ~fetch_op:(fun ~n -> Synthetic [%nn_cd n =+ session_step *. 100])
      ()
  in
  refresh_session ();
  print_tensor ~with_code:false ~with_grad:false `Default @@ c_data;
  [%expect
    {|
    ┌────────────────────────────────────────┐
    │[3]: <fetch_callback> shape 0:1|1:2,2:3 │
    │┌──────┬───────────────────────────┐    │
    ││      │0 @ 0                      │    │
    ││      │axis 2                     │    │
    │├──────┼───────────────────────────┼─── │
    ││axis 1│ 1.00e+2  1.01e+2  1.02e+2 │    │
    ││      │ 1.03e+2  1.04e+2  1.05e+2 │    │
    │└──────┴───────────────────────────┘    │
    └────────────────────────────────────────┘ |}];
  refresh_session ();
  print_tensor ~with_code:false ~with_grad:false `Default @@ c_data;
  [%expect
    {|
    ┌────────────────────────────────────────┐
    │[3]: <fetch_callback> shape 0:1|1:2,2:3 │
    │┌──────┬───────────────────────────┐    │
    ││      │0 @ 0                      │    │
    ││      │axis 2                     │    │
    │├──────┼───────────────────────────┼─── │
    ││axis 1│ 3.00e+2  3.01e+2  3.02e+2 │    │
    ││      │ 3.03e+2  3.04e+2  3.05e+2 │    │
    │└──────┴───────────────────────────┘    │
    └────────────────────────────────────────┘ |}];
  refresh_session ();
  print_tensor ~with_code:false ~with_grad:false `Default @@ c_data;
  [%expect
    {|
    ┌────────────────────────────────────────┐
    │[3]: <fetch_callback> shape 0:1|1:2,2:3 │
    │┌──────┬───────────────────────────┐    │
    ││      │0 @ 0                      │    │
    ││      │axis 2                     │    │
    │├──────┼───────────────────────────┼─── │
    ││axis 1│ 6.00e+2  6.01e+2  6.02e+2 │    │
    ││      │ 6.03e+2  6.04e+2  6.05e+2 │    │
    │└──────┴───────────────────────────┘    │
    └────────────────────────────────────────┘ |}]
