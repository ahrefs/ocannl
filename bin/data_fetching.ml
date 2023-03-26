open Base
open Ocannl
module FDSL = Operation.FDSL

let () = Session.SDSL.set_executor OCaml

let () =
  (* let open Operation.FDSL in *)
  let open Session.SDSL in
  drop_session();
  Random.init 0;
  let c_data = FDSL.data ~label:"fetch_callback" ~batch_dims:[1] ~output_dims:[2;3]
    (Compute_point (fun ~session_step ~dims:_ ~idcs ->
          Int.to_float @@ session_step*100 + idcs.(1)*10 + idcs.(2))) in
  refresh_session ();
  print_formula ~with_code:false ~with_grad:false `Default @@ c_data;
  refresh_session ();
  print_formula ~with_code:false ~with_grad:false `Default @@ c_data;
  refresh_session ();
  print_formula ~with_code:false ~with_grad:false `Default @@ c_data
  