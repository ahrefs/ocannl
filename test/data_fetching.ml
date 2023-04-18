open Base
open Ocannl
module FDSL = Operation.FDSL
module NFDSL = Operation.NFDSL
module CDSL = Code.CDSL


let () = Session.SDSL.set_executor OCaml

let%expect_test "Synthetic data" =
  (* let open Operation.FDSL in *)
  let open Session.SDSL in
  drop_all_sessions();
  Random.init 0;
  let session_step = FDSL.data ~label:"session_step" ~batch_dims:[] ~output_dims:[1]
      (fun ~n -> Synthetic [%nn_cd n =+ 1]) in
  let c_data = FDSL.data ~label:"fetch_callback" ~batch_dims:[1] ~output_dims:[2;3]
    (fun ~n -> Synthetic [%nn_cd n =+ (session_step *. 100)]) in
  refresh_session ();
  print_formula ~with_code:false ~with_grad:false `Default @@ c_data;
  [%expect {|
    ┌────────────────────────────────────────┐
    │[2]: <fetch_callback> shape 0:1|1:2,2:3 │
    │┌──────┬───────────────────────────┐    │
    ││      │0 @ 0                      │    │
    ││      │axis 2                     │    │
    │├──────┼───────────────────────────┼─── │
    ││axis 1│ 3.00e+2  3.01e+2  3.02e+2 │    │
    ││      │ 3.10e+2  3.11e+2  3.12e+2 │    │
    │└──────┴───────────────────────────┘    │
    └────────────────────────────────────────┘ |}];
  refresh_session ();
  print_formula ~with_code:false ~with_grad:false `Default @@ c_data;
  [%expect {|
    ┌────────────────────────────────────────┐
    │[2]: <fetch_callback> shape 0:1|1:2,2:3 │
    │┌──────┬───────────────────────────┐    │
    ││      │0 @ 0                      │    │
    ││      │axis 2                     │    │
    │├──────┼───────────────────────────┼─── │
    ││axis 1│ 4.00e+2  4.01e+2  4.02e+2 │    │
    ││      │ 4.10e+2  4.11e+2  4.12e+2 │    │
    │└──────┴───────────────────────────┘    │
    └────────────────────────────────────────┘ |}];
  refresh_session ();
  print_formula ~with_code:false ~with_grad:false `Default @@ c_data;
  [%expect {|
    ┌────────────────────────────────────────┐
    │[2]: <fetch_callback> shape 0:1|1:2,2:3 │
    │┌──────┬───────────────────────────┐    │
    ││      │0 @ 0                      │    │
    ││      │axis 2                     │    │
    │├──────┼───────────────────────────┼─── │
    ││axis 1│ 5.00e+2  5.01e+2  5.02e+2 │    │
    ││      │ 5.10e+2  5.11e+2  5.12e+2 │    │
    │└──────┴───────────────────────────┘    │
    └────────────────────────────────────────┘ |}]
