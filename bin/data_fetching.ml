open Base
open Ocannl
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module CDSL = Arrayjit.Low_level.CDSL




let () =
  (* let open Operation.TDSL in *)

  (* drop_all_sessions (); *)
  Random.init 0;
  let open NTDSL in
  let session_step = O.(counter !..1) in
  Tensor.print ~with_code:false ~with_grad:false `Default session_step;
  let biggo = O.(counter (session_step *. !..100)) in
  Tensor.print ~with_code:false ~with_grad:false `Default biggo
(* (* refresh_session (); *)
  Tensor.print ~with_code:false ~with_grad:false `Default @@ fetch_callback;
  (* refresh_session (); *)
  Tensor.print ~with_code:false ~with_grad:false `Default @@ fetch_callback;
  (* refresh_session (); *)
  Tensor.print ~with_code:false ~with_grad:false `Default @@ fetch_callback*)
