open Base
open Ocannl
module LA = Arrayjit.Lazy_array
module IDX = Arrayjit.Indexing.IDX
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module CDSL = Arrayjit.Low_level.CDSL
module Utils = Arrayjit.Utils

let hello1 () =
  Random.init 0;
  let module Backend = (val Train.fresh_backend ()) in
  Utils.settings.with_debug <- true;
  (* Utils.settings.keep_files_in_run_directory <- true; *)
  let device = Backend.get_device ~ordinal:0 in
  let ctx = Backend.init device in
  let open Operation.TDSL in
  (* Hey is inferred to be a matrix. *)
  let hey = range_of_shape ~batch_dims:[ 7 ] ~input_dims:[ 9; 10; 11 ] ~output_dims:[ 13; 14 ] () in
  let%op hoo = ((1 + 1) * hey) - 10 in
  Train.set_fully_on_host hoo.value;
  let jitted = Backend.jit ctx ~verbose:true IDX.empty @@ Train.forward hoo in
  Train.sync_run ~verbose:true (module Backend) jitted hoo;
  (* Disable line wrapping for viewing the output. In VSCode: `View: Toggle Word Wrap`. *)
  Tensor.print_tree ~with_grad:false ~depth:99 hoo;
  Tensor.print ~with_code:false ~with_grad:false `Default hoo

let hello2 () =
  Random.init 0;
  let module Backend = (val Train.fresh_backend ()) in
  Utils.settings.with_debug <- true;
  (* Utils.settings.keep_files_in_run_directory <- true; *)
  (* Utils.settings.debug_log_jitted <- true; *)
  let device = Backend.get_device ~ordinal:0 in
  let ctx = Backend.init device in
  (* Hey is inferred to be a matrix. *)
  let%op y = ("hey" * 'q' 2.0) + 'p' 1.0 in
  (* Punning for ["hey"] above introduced the [hey] identifier. *)
  Train.every_non_literal_fully_on_host y;
  let jitted = Backend.jit ctx ~verbose:true IDX.empty @@ Train.forward y in
  Train.sync_run ~verbose:true (module Backend) jitted y;
  Tensor.print ~with_code:false ~with_grad:false `Default @@ hey;
  Tensor.print ~with_code:false ~with_grad:false `Default @@ y

let hello3 () =
  Random.init 0;
  let module Backend = (val Train.fresh_backend ()) in
  Utils.settings.with_debug <- true;
  Utils.settings.keep_files_in_run_directory <- true;
  (* Utils.settings.debug_log_jitted <- true; *)
  let device = Backend.get_device ~ordinal:0 in
  let ctx = Backend.init device in
  (* Hey is inferred to be a matrix. *)
  let hey = TDSL.O.(!~"hey") in
  let zero_to_twenty = TDSL.range 20 in
  let y = TDSL.O.(( + ) ~label:[ "y" ] (hey * zero_to_twenty) zero_to_twenty) in
  Train.set_fully_on_host hey.value;
  Train.set_fully_on_host y.value;
  let jitted = Backend.jit ctx ~verbose:true IDX.empty @@ Train.forward y in
  if Backend.from_host jitted.context hey.value then Stdio.printf "Transferred <hey> to device.\n%!";
  if Backend.from_host jitted.context zero_to_twenty.value then
    Stdio.printf "Transferred <zero_to_twenty> to device.\n%!";
  Tensor.print ~with_code:true ~with_grad:false `Inline zero_to_twenty;
  Tensor.print ~with_code:true ~with_grad:false `Default zero_to_twenty;
  Tensor.print_tree ~with_grad:false ~depth:9 zero_to_twenty;
  Stdlib.Format.print_newline ();
  jitted.run ();
  Backend.await device;
  if Backend.to_host jitted.context y.value then Stdio.printf "Transferred <hey> to to host.\n%!";
  Tensor.print ~with_code:true ~with_grad:false `Default y;
  Stdlib.Format.force_newline();
  Tensor.print_tree ~with_grad:false ~depth:9 y;
  Stdlib.Format.force_newline()

let () =
  ignore (hello1, hello2, hello3);
  hello3 ()
