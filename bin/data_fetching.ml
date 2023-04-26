open Base
open Ocannl
module FDSL = Operation.FDSL
module NFDSL = Operation.NFDSL
module CDSL = Code.CDSL

let () = Session.SDSL.set_executor Gccjit

let () =
  (* let open Operation.FDSL in *)
  let open Session.SDSL in
  drop_all_sessions ();
  Random.init 0;
  let%nn_dt session_step ~o:[ 1 ] = n =+ 1 in
  let%nn_dt fetch_callback ~b:[ 1 ] ~o:[ 2; 3 ] = n =+ session_step *. 100 in
  refresh_session ();
  print_formula ~with_code:false ~with_grad:false `Default @@ fetch_callback;
  refresh_session ();
  print_formula ~with_code:false ~with_grad:false `Default @@ fetch_callback;
  refresh_session ();
  print_formula ~with_code:false ~with_grad:false `Default @@ fetch_callback
