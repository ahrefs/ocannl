open Base
open Ocannl
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module CDSL = Low_level.CDSL

let () = Session.SDSL.set_executor Gccjit

let () =
  (* let open Operation.TDSL in *)
  let open Session.SDSL in
  drop_all_sessions ();
  Random.init 0;
  let%nn_dt session_step ~o:1 = v =+ 1 in
  let%nn_dt fetch_callback ~b:1 ~o:(2, 3) = v =+ session_step *. 100 in
  refresh_session ();
  print_tensor ~with_code:false ~with_grad:false `Default @@ fetch_callback;
  refresh_session ();
  print_tensor ~with_code:false ~with_grad:false `Default @@ fetch_callback;
  refresh_session ();
  print_tensor ~with_code:false ~with_grad:false `Default @@ fetch_callback
