open Base
open Ocannl
module Tn = Ir.Tnode
module IDX = Train.IDX
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module CDSL = Train.CDSL

module type Backend = Ir.Backend_intf.Backend

let hello1 () =
  let module Backend = (val Backends.fresh_backend ()) in
  let open Operation.TDSL in
  (* Hey is inferred to be a matrix. *)
  let hey = range_of_shape ~batch_dims:[ 7 ] ~input_dims:[ 9; 10; 11 ] ~output_dims:[ 13; 14 ] () in
  let%op hoo = ((1 + 1) * hey) - 10 in
  (* For convenience, Train.forward will set hoo.value as fully on host. *)
  ignore (Train.forward_once (module Backend) hoo);
  (* Disable line wrapping for viewing the output. In VSCode: `View: Toggle Word Wrap`. *)
  Train.printf_tree ~with_grad:false ~depth:99 hoo;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false hoo

let hello2 () =
  let module Backend = (val Backends.fresh_backend ()) in
  (* Hey is inferred to be a matrix. *)
  let%op y = ("hey" * 'q' 2.0) + 'p' 1.0 in
  (* Punning for ["hey"] above introduced the [hey] identifier. *)
  Train.every_non_literal_on_host y;
  ignore (Train.forward_once (module Backend) y);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false hey;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false y

let hello3 () =
  let module Backend = (val Backends.fresh_backend ()) in
  (* Hey is inferred to be a matrix. *)
  let hey = TDSL.param "hey" () in
  let zero_to_twenty = TDSL.range 20 in
  let y = TDSL.O.(( + ) ~label:[ "y" ] (hey * zero_to_twenty) zero_to_twenty) in
  Train.set_hosted hey.value;
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  let routine = Train.to_routine (module Backend) ctx IDX.empty @@ Train.forward y in
  Stdio.printf "\n%!";
  Train.run routine;
  Train.printf ~here:[%here] ~with_code:true ~with_grad:false y;
  Stdio.printf "\n%!";
  Train.printf_tree ~with_grad:false ~depth:9 y;
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

  let ri = TDSL.range 3 in
  let%op ti = ri ++ "i=>i0" in
  (* Write position 2 of ti, otherwise shape inference concludes it's dim-1 and broadcasted. *)
  let%cd _ = ti =: 0 ++ "i=>i2" in
  let rj = TDSL.range 4 in
  let%op tj = rj ++ "j=>j1" in
  let rk = TDSL.range 5 in
  let%op tk = rk ++ "k=>k2" in
  let positions = TDSL.outer_sum "ijl;kl=>ijkl" (TDSL.outer_sum "il;jl=>ijl" ti tj ()) tk () in
  Train.set_hosted ti.value;
  Train.set_hosted tk.value;
  ignore (Train.forward_once backend positions);
  Stdio.print_endline "positions:";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false positions;
  Stdio.print_endline "tk:";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false tk;
  Stdio.print_endline "ti:";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ti;
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

  let hey = TDSL.range_of_shape ~batch_dims:[ 2 ] ~input_dims:[ 3 ] ~output_dims:[ 4 ] () in
  let%op ho = hey ++ "...|1->... => ...|..." in
  ignore (Train.forward_once backend ho);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ho

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

  (* "Hey" is inferred to be a scalar. *)
  let%op y = 2 *. "hey" in
  ignore (Train.forward_once backend y);
  (* Train.printf ~here:[%here] ~with_code:false ~with_grad:false hey; *)
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false y

let () =
  ignore (hello1, hello2, hello3, hello4, hello5, hello6);
  hello1 ();
  hello2 ()
