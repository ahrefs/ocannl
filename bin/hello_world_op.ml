open Base
open Ocannl
module Tn = Arrayjit.Tnode
module IDX = Train.IDX
module CDSL = Train.CDSL
module TDSL = Operation.TDSL
module Rand = Arrayjit.Rand.Lib

module type Backend = Arrayjit.Backend_intf.Backend

module Debug_runtime = Arrayjit.Utils.Debug_runtime

let _get_local_debug_runtime = Arrayjit.Utils._get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

let setup () =
  Arrayjit.Utils.settings.output_debug_files_in_build_directory <- true;
  Arrayjit.Utils.settings.debug_log_from_routines <- true

let%track2_sexp _Pointwise_multiplication_dims_1 () =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Arrayjit.Backends.fresh_backend ()) in
  let backend =
    (module Backend : Backend
      with type buffer_ptr = Backend.buffer_ptr
       and type dev = Backend.dev
       and type runner = Backend.runner
       and type event = Backend.event)
  in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  Rand.init 0;
  (* "Hey" is inferred to be a scalar. *)
  let%op y = 2 *. "hey" 7.0 in
  Train.forward_and_forget backend ctx y;
  Tensor.print ~with_code:false ~with_grad:false `Default @@ y

let%track2_sexp _Matrix_multiplication_dims_1x1 () =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Arrayjit.Backends.fresh_backend ()) in
  let backend =
    (module Backend : Backend
      with type buffer_ptr = Backend.buffer_ptr
       and type dev = Backend.dev
       and type runner = Backend.runner
       and type event = Backend.event)
  in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  Rand.init 0;
  (* Hey is inferred to be a matrix because of matrix multiplication [*]. *)
  let%op y = ("hey" 7.0 * 'q' 2.0) + 'p' 1.0 in
  Train.forward_and_forget backend ctx y;
  (* Punning for ["hey"] above introduced the [hey] identifier. *)
  Tensor.print ~with_code:false ~with_grad:false `Default @@ hey;
  Tensor.print ~with_code:false ~with_grad:false `Default @@ y

let%track2_sexp _Print_constant_tensor () =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Arrayjit.Backends.fresh_backend ()) in
  let backend =
    (module Backend : Backend
      with type buffer_ptr = Backend.buffer_ptr
       and type dev = Backend.dev
       and type runner = Backend.runner
       and type event = Backend.event)
  in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  Rand.init 0;
  let%op hey = [ (1, 2, 3); (4, 5, 6) ] in
  Train.forward_and_forget backend ctx hey;
  Tensor.print ~force:true ~with_code:false ~with_grad:false `Inline @@ hey;
  Tensor.print ~with_code:false ~with_grad:false `Default @@ hey;
  let%op hoo = [| [ 1; 2; 3 ]; [ 4; 5; 6 ] |] in
  Train.forward_and_forget backend ctx hoo;
  Tensor.print ~force:true ~with_code:false ~with_grad:false `Inline @@ hoo;
  Tensor.print ~with_code:false ~with_grad:false `Default @@ hoo;
  let%op hey2 =
    [
      ((1, 2, 3), (4, 5, 6));
      ((7, 8, 9), (10, 11, 12));
      ((13, 14, 15), (16, 17, 18));
      ((19, 20, 21), (22, 23, 24));
    ]
  in
  Train.forward_and_forget backend ctx hey2;
  Tensor.print ~force:true ~with_code:false ~with_grad:false `Inline @@ hey2;
  Tensor.print ~with_code:false ~with_grad:false `Default @@ hey2;
  let%op hoo2 =
    [|
      [ [ 1; 2; 3 ]; [ 4; 5; 6 ] ];
      [ [ 7; 8; 9 ]; [ 10; 11; 12 ] ];
      [ [ 13; 14; 15 ]; [ 16; 17; 18 ] ];
      [ [ 19; 20; 21 ]; [ 22; 23; 24 ] ];
    |]
  in
  Train.forward_and_forget backend ctx hoo2;
  Tensor.print ~force:true ~with_code:false ~with_grad:false `Inline @@ hoo2;
  Tensor.print ~force:true ~with_code:false ~with_grad:false `Default @@ hoo2;
  let%op heyhoo =
    [|
      [| [ 1; 2; 3 ]; [ 4; 5; 6 ] |];
      [| [ 7; 8; 9 ]; [ 10; 11; 12 ] |];
      [| [ 13; 14; 15 ]; [ 16; 17; 18 ] |];
      [| [ 19; 20; 21 ]; [ 22; 23; 24 ] |];
    |]
  in
  Train.forward_and_forget backend ctx heyhoo;
  Tensor.print ~force:true ~with_code:false ~with_grad:false `Inline @@ heyhoo;
  Tensor.print ~force:true ~with_code:false ~with_grad:false `Default @@ heyhoo;
  let%op heyhoo2 =
    [|
      [| [ [ 1; 31 ]; [ 2; 32 ]; [ 3; 33 ] ]; [ [ 4; 34 ]; [ 5; 35 ]; [ 6; 36 ] ] |];
      [| [ [ 7; 37 ]; [ 8; 38 ]; [ 9; 39 ] ]; [ [ 10; 40 ]; [ 11; 41 ]; [ 12; 42 ] ] |];
      [| [ [ 13; 43 ]; [ 14; 44 ]; [ 15; 45 ] ]; [ [ 16; 46 ]; [ 17; 47 ]; [ 18; 48 ] ] |];
      [| [ [ 19; 49 ]; [ 20; 50 ]; [ 21; 51 ] ]; [ [ 22; 52 ]; [ 23; 53 ]; [ 24; 54 ] ] |];
    |]
  in
  Train.forward_and_forget backend ctx heyhoo2;
  Tensor.print ~force:true ~with_code:false ~with_grad:false `Inline @@ heyhoo2;
  Tensor.print ~force:true ~with_code:false ~with_grad:false `Default @@ heyhoo2;
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
  Train.forward_and_forget backend ctx heyhoo3;
  Tensor.print ~force:true ~with_code:false ~with_grad:false `Inline @@ heyhoo3;
  Tensor.print ~force:true ~with_code:false ~with_grad:false `Default @@ heyhoo3;
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
  Train.forward_and_forget backend ctx heyhoo4;
  Tensor.print ~force:true ~with_code:false ~with_grad:false `Inline @@ heyhoo4;
  Tensor.print ~force:true ~with_code:false ~with_grad:false `Default @@ heyhoo4

let%track2_sexp _Matrix_multiplication_dims_2x3 () =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Arrayjit.Backends.fresh_backend ()) in
  let backend =
    (module Backend : Backend
      with type buffer_ptr = Backend.buffer_ptr
       and type dev = Backend.dev
       and type runner = Backend.runner
       and type event = Backend.event)
  in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  Rand.init 0;
  (* Hey is inferred to be a matrix. *)
  let%op y = ("hey" 7.0 * [ 2; 3 ]) + [ 4; 5; 6 ] in
  Train.forward_and_forget backend ctx y;
  Tensor.print ~with_code:false ~with_grad:false `Default @@ hey;
  Tensor.print ~with_code:false ~with_grad:false `Default @@ y

let%track2_sexp _Big_matrix () =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Arrayjit.Backends.fresh_backend ()) in
  let backend =
    (module Backend : Backend
      with type buffer_ptr = Backend.buffer_ptr
       and type dev = Backend.dev
       and type runner = Backend.runner
       and type event = Backend.event)
  in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  Rand.init 0;
  (* Hey is inferred to be a matrix. *)
  let hey = Tensor.param ~values:[| 0.5 |] "hey" in
  let zero_to_twenty = TDSL.range 20 in
  let y = TDSL.O.((hey * zero_to_twenty) + zero_to_twenty) in
  Train.forward_and_forget backend ctx y;
  Tensor.print ~with_code:false ~with_grad:false `Inline zero_to_twenty;
  Tensor.print ~with_code:false ~with_grad:false `Default zero_to_twenty;
  Tensor.print ~with_code:false ~with_grad:false `Default hey;
  Tensor.print ~with_code:false ~with_grad:false `Default y

let%track2_sexp _Very_big_tensor () =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Arrayjit.Backends.fresh_backend ()) in
  let backend =
    (module Backend : Backend
      with type buffer_ptr = Backend.buffer_ptr
       and type dev = Backend.dev
       and type runner = Backend.runner
       and type event = Backend.event)
  in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  Rand.init 0;
  let hey =
    TDSL.range_of_shape ~batch_dims:[ 6 ] ~input_dims:[ 7; 8; 9 ] ~output_dims:[ 10; 11 ] ()
  in
  let%op hoo = (hey * (1 + 1)) - 10 in
  Train.forward_and_forget backend ctx hoo;
  Tensor.print ~with_code:false ~with_grad:false `Default hey;
  Tensor.print ~with_code:false ~with_grad:false `Default hoo

let () =
  setup ();
  _Pointwise_multiplication_dims_1 ();
  _Matrix_multiplication_dims_1x1 ();
  _Print_constant_tensor ();
  _Matrix_multiplication_dims_2x3 ();
  _Big_matrix ();
  _Very_big_tensor ()