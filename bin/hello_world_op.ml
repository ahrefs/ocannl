open Base
open Ocannl
module Tn = Ir.Tnode
module IDX = Train.IDX
module CDSL = Train.CDSL
module TDSL = Operation.TDSL

module type Backend = Ir.Backend_intf.Backend

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_HELLO_WORLD_OP=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_HELLO_WORLD_OP"]

let%track2_sexp _Pointwise_multiplication_dims_1 (() : unit) : unit =
  Tensor.unsafe_reinitialize ();
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
  let%op ya = 2 *. { hey = 7.0 } in
  ignore (Train.forward_once backend ya);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ya

let%track2_sexp _Matrix_multiplication_dims_1x1 (() : unit) : unit =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
  let backend =
    (module Backend : Backend
      with type buffer_ptr = Backend.buffer_ptr
       and type dev = Backend.dev
       and type runner = Backend.runner
       and type event = Backend.event
       and type optimize_ctx = Backend.optimize_ctx)
  in

  (* Hey is inferred to be a matrix because of matrix multiplication [*]. *)
  let%op yb = ({ hey = 7.0 } * 'q' 2.0) + 'p' 1.0 in
  ignore (Train.forward_once backend yb);
  (* Punning for ["hey"] above introduced the [hey] identifier. *)
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false hey;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false yb

let%track2_sexp _Print_constant_tensor_too_early (() : unit) : unit =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
  let _print_tensor = Train.printf ~with_code:false ~with_grad:false in

  let%op a = [| 1.; 2.; 3.; 4. |] in
  let%op b = [| 2.; 3.; 4.; 5. |] in
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Inline a;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Inline b;
  let%op c = a *. b in

  ignore (Train.forward_once (module Backend) c);
  Train.printf ~here:[%here] c

let%track2_sexp _Print_constant_tensor (() : unit) : unit =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
  let backend =
    (module Backend : Backend
      with type buffer_ptr = Backend.buffer_ptr
       and type dev = Backend.dev
       and type runner = Backend.runner
       and type event = Backend.event
       and type optimize_ctx = Backend.optimize_ctx)
  in

  let%op hey = [ (1, 2, 3); (4, 5, 6) ] in
  ignore (Train.forward_once backend hey);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false @@ hey;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false hey;
  let%op hoo = [| [ 1; 2; 3 ]; [ 4; 5; 6 ] |] in
  ignore (Train.forward_once backend hoo);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false @@ hoo;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false hoo;
  let%op hey2 =
    [
      ((1, 2, 3), (4, 5, 6));
      ((7, 8, 9), (10, 11, 12));
      ((13, 14, 15), (16, 17, 18));
      ((19, 20, 21), (22, 23, 24));
    ]
  in
  ignore (Train.forward_once backend hey2);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false @@ hey2;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false hey2;
  let%op hoo2 =
    [|
      [ [ 1; 2; 3 ]; [ 4; 5; 6 ] ];
      [ [ 7; 8; 9 ]; [ 10; 11; 12 ] ];
      [ [ 13; 14; 15 ]; [ 16; 17; 18 ] ];
      [ [ 19; 20; 21 ]; [ 22; 23; 24 ] ];
    |]
  in
  ignore (Train.forward_once backend hoo2);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false @@ hoo2;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false hoo2;
  let%op heyhoo =
    [|
      [| [ 1; 2; 3 ]; [ 4; 5; 6 ] |];
      [| [ 7; 8; 9 ]; [ 10; 11; 12 ] |];
      [| [ 13; 14; 15 ]; [ 16; 17; 18 ] |];
      [| [ 19; 20; 21 ]; [ 22; 23; 24 ] |];
    |]
  in
  ignore (Train.forward_once backend heyhoo);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false @@ heyhoo;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false heyhoo;
  let%op heyhoo2 =
    [|
      [| [ [ 1; 31 ]; [ 2; 32 ]; [ 3; 33 ] ]; [ [ 4; 34 ]; [ 5; 35 ]; [ 6; 36 ] ] |];
      [| [ [ 7; 37 ]; [ 8; 38 ]; [ 9; 39 ] ]; [ [ 10; 40 ]; [ 11; 41 ]; [ 12; 42 ] ] |];
      [| [ [ 13; 43 ]; [ 14; 44 ]; [ 15; 45 ] ]; [ [ 16; 46 ]; [ 17; 47 ]; [ 18; 48 ] ] |];
      [| [ [ 19; 49 ]; [ 20; 50 ]; [ 21; 51 ] ]; [ [ 22; 52 ]; [ 23; 53 ]; [ 24; 54 ] ] |];
    |]
  in
  ignore (Train.forward_once backend heyhoo2);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false @@ heyhoo2;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false heyhoo2;
  let%op heyhoo3 =
    [|
      [|
        [ [ [ 1; 31 ]; [ 2; 32 ]; [ 3; 33 ] ]; [ [ 4; 34 ]; [ 5; 35 ]; [ 6; 36 ] ] ];
        [ [ [ 7; 37 ]; [ 8; 38 ]; [ 9; 39 ] ]; [ [ 10; 40 ]; [ 11; 41 ]; [ 12; 42 ] ] ];
      |];
      [|
        [ [ [ 13; 43 ]; [ 14; 44 ]; [ 15; 45 ] ]; [ [ 16; 46 ]; [ 17; 47 ]; [ 18; 48 ] ] ];
        [ [ [ 19; 49 ]; [ 20; 50 ]; [ 21; 51 ] ]; [ [ 22; 52 ]; [ 23; 53 ]; [ 24; 54 ] ] ];
      |];
    |]
  in
  ignore (Train.forward_once backend heyhoo3);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false @@ heyhoo3;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false heyhoo3;
  let%op heyhoo4 =
    [|
      [
        [ [ (1, 31); (2, 32); (3, 33) ]; [ (4, 34); (5, 35); (6, 36) ] ];
        [ [ (7, 37); (8, 38); (9, 39) ]; [ (10, 40); (11, 41); (12, 42) ] ];
      ];
      [
        [ [ (13, 43); (14, 44); (15, 45) ]; [ (16, 46); (17, 47); (18, 48) ] ];
        [ [ (19, 49); (20, 50); (21, 51) ]; [ (22, 52); (23, 53); (24, 54) ] ];
      ];
    |]
  in
  ignore (Train.forward_once backend heyhoo4);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false @@ heyhoo4;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false heyhoo4

let%track2_sexp _Matrix_multiplication_dims_2x3 (() : unit) : unit =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
  let backend =
    (module Backend : Backend
      with type buffer_ptr = Backend.buffer_ptr
       and type dev = Backend.dev
       and type runner = Backend.runner
       and type event = Backend.event
       and type optimize_ctx = Backend.optimize_ctx)
  in

  (* Hey is inferred to be a matrix. *)
  let%op yc = ({ hey = 7.0 } * [ 2; 3 ]) + [ 4; 5; 6 ] in
  ignore (Train.forward_once backend yc);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false hey;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false yc

let%track2_sexp _Big_matrix (() : unit) : unit =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
  let backend =
    (module Backend : Backend
      with type buffer_ptr = Backend.buffer_ptr
       and type dev = Backend.dev
       and type runner = Backend.runner
       and type event = Backend.event
       and type optimize_ctx = Backend.optimize_ctx)
  in

  (* Hey is inferred to be a matrix. *)
  let hey = TDSL.param ~value:0.5 "hey" () in
  let zero_to_twenty = TDSL.range 20 in
  let%op yd = (hey * zero_to_twenty) + zero_to_twenty in
  ignore (Train.forward_once backend yd);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false hey;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false yd

let%track2_sexp _Very_big_tensor (() : unit) : unit =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
  let backend =
    (module Backend : Backend
      with type buffer_ptr = Backend.buffer_ptr
       and type dev = Backend.dev
       and type runner = Backend.runner
       and type event = Backend.event
       and type optimize_ctx = Backend.optimize_ctx)
  in

  let hey = TDSL.range_of_shape ~batch_dims:[ 6 ] ~input_dims:[ 7; 8 ] ~output_dims:[ 9 ] () in
  let%op ye = (hey * (1 + 1)) - 10 in
  ignore (Train.forward_once backend ye);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ye

let _suspended (() : unit) : unit =
  _Matrix_multiplication_dims_2x3 ();
  _Big_matrix ()

let _suspended (() : unit) : unit =
  _Pointwise_multiplication_dims_1 ();
  _Matrix_multiplication_dims_1x1 ();
  _Print_constant_tensor ();
  _Matrix_multiplication_dims_2x3 ();
  _Big_matrix ();
  _Very_big_tensor ()

let (() : unit) : unit = _Print_constant_tensor_too_early ()
