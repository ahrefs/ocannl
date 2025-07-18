open Base
open Ocannl
module Tn = Ir.Tnode
module IDX = Train.IDX
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module CDSL = Train.CDSL
module Rand = Ir.Rand.Lib

module type Backend = Ir.Backend_intf.Backend

let hello1 () =
  Rand.init 0;
  let module Backend = (val Backends.fresh_backend ()) in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  let open Operation.TDSL in
  (* Hey is inferred to be a matrix. *)
  let hey = range_of_shape ~batch_dims:[ 7 ] ~input_dims:[ 9; 10; 11 ] ~output_dims:[ 13; 14 ] () in
  let%op hoo = ((1 + 1) * hey) - 10 in
  (* For convenience, Train.forward will set hoo.value as fully on host. *)
  Train.forward_and_forget (module Backend) ctx hoo;
  (* Disable line wrapping for viewing the output. In VSCode: `View: Toggle Word Wrap`. *)
  Tensor.print_tree ~with_grad:false ~depth:99 hoo;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default hoo

let hello2 () =
  Rand.init 0;
  let module Backend = (val Backends.fresh_backend ()) in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  (* Hey is inferred to be a matrix. *)
  let%op y = ("hey" * 'q' 2.0) + 'p' 1.0 in
  (* Punning for ["hey"] above introduced the [hey] identifier. *)
  Train.every_non_literal_on_host y;
  Train.forward_and_forget (module Backend) ctx y;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ hey;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ y

let hello3 () =
  Rand.init 0;
  let module Backend = (val Backends.fresh_backend ()) in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  (* Hey is inferred to be a matrix. *)
  let hey = TDSL.param "hey" in
  let zero_to_twenty = TDSL.range 20 in
  let y = TDSL.O.(( + ) ~label:[ "y" ] (hey * zero_to_twenty) zero_to_twenty) in
  Train.set_hosted hey.value;
  let routine = Train.to_routine (module Backend) ctx IDX.empty @@ Train.forward y in
  Stdio.printf "\n%!";
  Train.run routine;
  Tensor.print ~here:[%here] ~with_code:true ~with_grad:false `Default y;
  Stdio.printf "\n%!";
  Tensor.print_tree ~with_grad:false ~depth:9 y;
  Stdio.printf "\n%!"

let hello4 () =
  let module Backend = (val Backends.fresh_backend ()) in
  let backend =
    (module Backend : Backend
      with type buffer_ptr = Backend.buffer_ptr
       and type dev = Backend.dev
       and type runner = Backend.runner
       and type event = Backend.event
       and type optimize_ctx = Backend.optimize_ctx)
  in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  Rand.init 0;
  let ri = TDSL.range 3 in
  let%op ti = ri ++ "i=>i0" in
  (* Write position 2 of ti, otherwise shape inference concludes it's dim-1 and broadcasted. *)
  let%cd _ = ti =: 0 ++ "i=>i2" in
  let rj = TDSL.range 4 in
  let%op tj = rj ++ "j=>j1" in
  let rk = TDSL.range 5 in
  let%op tk = rk ++ "k=>k2" in
  let positions = TDSL.outer_sum "ijl;kl=>ijkl" (TDSL.outer_sum "il;jl=>ijl" ti tj) tk in
  Train.set_hosted ti.value;
  Train.set_hosted tk.value;
  Train.forward_and_forget backend ctx positions;
  Stdio.print_endline "positions:";
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ positions;
  Stdio.print_endline "tk:";
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ tk;
  Stdio.print_endline "ti:";
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ ti;
  Stdio.printf "\n%!"

let hello5 () =
  let module Backend = (val Backends.fresh_backend ()) in
  let backend =
    (module Backend : Backend
      with type buffer_ptr = Backend.buffer_ptr
       and type dev = Backend.dev
       and type runner = Backend.runner
       and type event = Backend.event
       and type optimize_ctx = Backend.optimize_ctx)
  in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  Rand.init 0;
  let hey = TDSL.range_of_shape ~batch_dims:[ 2 ] ~input_dims:[ 3 ] ~output_dims:[ 4 ] () in
  let%op ho = hey ++ "...|1->... => ...|..." in
  Train.forward_and_forget backend ctx ho;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ ho

let hello6 () =
  let module Backend = (val Backends.fresh_backend ()) in
  let backend =
    (module Backend : Backend
      with type buffer_ptr = Backend.buffer_ptr
       and type dev = Backend.dev
       and type runner = Backend.runner
       and type event = Backend.event
       and type optimize_ctx = Backend.optimize_ctx)
  in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  Rand.init 0;
  (* "Hey" is inferred to be a scalar. *)
  let%op y = 2 *. "hey" in
  Train.forward_and_forget backend ctx y;
  (* Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ hey; *)
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ y

let () =
  ignore (hello1, hello2, hello3, hello4, hello5, hello6);
  hello1 ();
  hello2 ()
