open Base
open Ocannl
module FDSL = Operation.FDSL
module NFDSL = Operation.NFDSL
module CDSL = Code.CDSL

let () = Session.SDSL.set_executor OCaml

let () =
  (* let open Operation.FDSL in *)
  let open Session.SDSL in
  drop_all_sessions();
  Random.init 0;
  let session_step = FDSL.data ~label:"session_step" ~batch_dims:[] ~output_dims:[1]
      (fun ~n -> Synthetic
          [%nn_cd n =+ ~= 1 ~logic:"."]) in
  let c_data = FDSL.data ~label:"fetch_callback" ~batch_dims:[1] ~output_dims:[2;3]
    (fun ~n -> Synthetic
          [%nn_cd n =+ ~= (session_step *. 100) ~logic:"."]) in
  refresh_session ();
  print_formula ~with_code:false ~with_grad:false `Default @@ c_data;
  refresh_session ();
  print_formula ~with_code:false ~with_grad:false `Default @@ c_data;
  refresh_session ();
  print_formula ~with_code:false ~with_grad:false `Default @@ c_data
  