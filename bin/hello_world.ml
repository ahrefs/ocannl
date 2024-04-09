open Base
open Ocannl
module Tn = Arrayjit.Tnode
module IDX = Arrayjit.Indexing.IDX
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module CDSL = Arrayjit.Low_level.CDSL
module Utils = Arrayjit.Utils

let hello1 () =
  Random.init 0;
  let module Backend = (val Train.fresh_backend ()) in
  Utils.settings.with_debug <- true;
  (* Utils.settings.output_debug_files_in_run_directory <- true; *)
  let device = Backend.get_device ~ordinal:0 in
  let ctx = Backend.init device in
  let open Operation.TDSL in
  (* Hey is inferred to be a matrix. *)
  let hey = range_of_shape ~batch_dims:[ 7 ] ~input_dims:[ 9; 10; 11 ] ~output_dims:[ 13; 14 ] () in
  let%op hoo = ((1 + 1) * hey) - 10 in
  (* For convenience, Train.forward will set hoo.value as fully on host.  *)
  Train.forward_and_forget (module Backend) ctx hoo;
  (* Disable line wrapping for viewing the output. In VSCode: `View: Toggle Word Wrap`. *)
  Tensor.print_tree ~with_grad:false ~depth:99 hoo;
  Tensor.print ~with_code:false ~with_grad:false `Default hoo

let hello2 () =
  Random.init 0;
  let module Backend = (val Train.fresh_backend ()) in
  Utils.settings.with_debug <- true;
  (* Utils.settings.output_debug_files_in_run_directory <- true; *)
  (* Utils.settings.debug_log_from_routines <- true; *)
  let device = Backend.get_device ~ordinal:0 in
  let ctx = Backend.init device in
  (* Hey is inferred to be a matrix. *)
  let%op y = ("hey" * 'q' 2.0) + 'p' 1.0 in
  (* Punning for ["hey"] above introduced the [hey] identifier. *)
  Train.every_non_literal_on_host y;
  Train.forward_and_forget (module Backend) ctx y;
  Tensor.print ~with_code:false ~with_grad:false `Default @@ hey;
  Tensor.print ~with_code:false ~with_grad:false `Default @@ y

let hello3 () =
  Random.init 0;
  let module Backend = (val Train.fresh_backend ()) in
  Utils.settings.output_debug_files_in_run_directory <- true;
  (* Utils.settings.debug_log_from_routines <- true; *)
  let device = Backend.get_device ~ordinal:0 in
  let ctx = Backend.init device in
  (* Hey is inferred to be a matrix. *)
  let hey = TDSL.O.(!~"hey") in
  let zero_to_twenty = TDSL.range 20 in
  let y = TDSL.O.(( + ) ~label:[ "y" ] (hey * zero_to_twenty) zero_to_twenty) in
  Train.set_hosted hey.value;
  let routine = Backend.jit ctx IDX.empty @@ Train.forward y in
  if Backend.from_host routine.context hey.value then Stdio.printf "Transferred <hey> to device.\n%!";
  if Backend.from_host routine.context zero_to_twenty.value then
    Stdio.printf "Transferred <zero_to_twenty> to device.\n%!";
  Tensor.print ~with_code:true ~with_grad:false `Inline zero_to_twenty;
  Tensor.print ~with_code:true ~with_grad:false `Default zero_to_twenty;
  Tensor.print_tree ~with_grad:false ~depth:9 zero_to_twenty;
  Stdlib.Format.print_newline ();
  Train.run routine;
  Backend.await device;
  if Backend.to_host routine.context y.value then Stdio.printf "Transferred <hey> to to host.\n%!";
  Tensor.print ~with_code:true ~with_grad:false `Default y;
  Stdlib.Format.force_newline ();
  Tensor.print_tree ~with_grad:false ~depth:9 y;
  Stdlib.Format.force_newline ()

let hello4 () =
  let module Backend = (val Train.fresh_backend ()) in
  let backend = (module Backend : Train.Backend_type with type context = Backend.context) in
  let device = Backend.get_device ~ordinal:0 in
  let ctx = Backend.init device in
  Random.init 0;
  let ri = TDSL.range 3 in
  (* Note: using ti = ri ++ "i=>i0" would make ti have dim-1 in the last axis and then broadcast. *)
  (* TODO: allow specifying dimensions explicitely? *)
  let%op ti = ri ++ "i=>i1" in
  let rj = TDSL.range 4 in
  let%op tj = rj ++ "j=>j2" in
  let rk = TDSL.range 5 in
  let%op tk = rk ++ "k=>k3" in
  let positions = TDSL.outer_sum "ijl;kl=>ijkl" (TDSL.outer_sum "il;jl=>ijl" ti tj) tk in
  Train.set_hosted tk.value;
  Train.forward_and_forget backend ctx positions;
  Stdio.print_endline "positions:";
  Tensor.print ~force:true ~with_code:false ~with_grad:false `Default @@ positions;
  Stdio.print_endline "tk:";
  Tensor.print ~force:true ~with_code:false ~with_grad:false `Default @@ tk;
  Stdio.printf "\n%!"
(* FIXME: this should be 6x3. *)

let () =
  ignore (hello1, hello2, hello3);
  hello4 ()
