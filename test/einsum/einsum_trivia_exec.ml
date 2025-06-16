open Base
open Ocannl
module Tn = Ir.Tnode
module IDX = Train.IDX
module CDSL = Train.CDSL
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module Rand = Ir.Rand.Lib

module type Backend = Ir.Backend_intf.Backend

(* A standalone-executable version of einsum_trivia.ml. *)

let () =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
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
  let hey = TDSL.range_of_shape ~batch_dims:[ 2 ] ~input_dims:[ 3 ] ~output_dims:[ 4 ] () in
  let%op ho = hey ++ "b|i->o => o|b->i" in
  Train.forward_and_forget backend ctx ho;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ hey;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ ho;
  let hey2 =
    TDSL.range_of_shape ~batch_dims:[ 2; 3 ] ~input_dims:[ 4; 5 ] ~output_dims:[ 6; 7 ] ()
  in
  let%op ho2 = hey2 ++ "ab|cd->ef => cf|ae->db" in
  Train.forward_and_forget backend ctx ho2;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ hey2;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ ho2;
  let a = TDSL.range_of_shape ~batch_dims:[ 3 ] ~input_dims:[ 4 ] ~output_dims:[ 2 ] () in
  let b = TDSL.range_of_shape ~batch_dims:[ 3 ] ~input_dims:[ 1 ] ~output_dims:[ 4 ] () in
  let%op c = a *+ "...|i->1; ...|...->i => ...|i" b in
  Train.forward_and_forget backend ctx c;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ a;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ b;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ b;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ c

let () =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
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
  let hey = TDSL.range_of_shape ~batch_dims:[ 2 ] ~input_dims:[ 3 ] ~output_dims:[ 4 ] () in
  let%op ho = hey ++ "b|i->o => b|i" in
  Train.forward_and_forget backend ctx ho;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ hey;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ ho;
  let hey2 =
    TDSL.range_of_shape ~batch_dims:[ 2; 3 ] ~input_dims:[ 4; 5 ] ~output_dims:[ 6; 7 ] ()
  in
  let%op ho2 = hey2 ++ "ab|cd->ef => c|a->d" in
  Train.forward_and_forget backend ctx ho2;
  (* Axis 5 of hey2, i.e. d in the einsum spec, has the lowest variation (progresses by 1), that's
     why axis 1 of ho2 appears nearly constant. *)
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ ho2

let () =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
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
  let a = TDSL.range_of_shape ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] () in
  let b = TDSL.range_of_shape ~batch_dims:[] ~input_dims:[] ~output_dims:[ 3 ] () in
  let%op c = (a + 1) *+ "i; j => i->j" b in
  Train.forward_and_forget backend ctx c;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ a;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ b;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ c;
  let a = TDSL.range_of_shape ~batch_dims:[ 2 ] ~input_dims:[ 3 ] ~output_dims:[ 4 ] () in
  let b = TDSL.range_of_shape ~batch_dims:[ 5 ] ~input_dims:[ 6 ] ~output_dims:[ 7 ] () in
  let%op c = a *+ "i|j->k; l|m->n => il|jm->kn" b in
  Train.forward_and_forget backend ctx c;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ a;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ b;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ c

let () =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
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
  let a = TDSL.range_of_shape ~batch_dims:[ 2 ] ~input_dims:[ 3 ] ~output_dims:[ 4 ] () in
  let b = TDSL.range_of_shape ~batch_dims:[ 2 ] ~input_dims:[ 4 ] ~output_dims:[ 5 ] () in
  let%op a2 = a *+ "b|i->o; b|i->o => b|i->o" a in
  let ctx = Train.forward_and_ctx backend ctx a2 in
  let%op c = b *+ "b|h->o; b|i->h => b|i->o" a in
  let ctx = Train.forward_and_ctx backend ctx c in
  let%op d = a *+ "a|i->h; b|h->o => ab|i->o" b in
  Train.forward_and_forget backend ctx d;
  let%op e = a *+ "b|i->h; b|h->o => i->o" b in
  Train.forward_and_forget backend ctx e;
  let%op f = a *+ "a|i->h; b|h->o => i->o" b in
  Train.forward_and_forget backend ctx f;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ a2;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ c;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ d;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ e;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ f

let () =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
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
  let hey = TDSL.range_of_shape ~batch_dims:[ 2 ] ~input_dims:[ 3 ] ~output_dims:[ 4 ] () in
  let%op ho = hey ++ "...|i->o => ...|o->i" in
  let ctx = Train.forward_and_ctx backend ctx ho in
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ hey;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ ho;
  let%op ho2 = hey ++ "b|...->o => o|...->b" in
  Train.forward_and_forget backend ctx ho2;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ ho2;

  let hey2 =
    TDSL.range_of_shape ~batch_dims:[ 2; 3 ] ~input_dims:[ 4; 5 ] ~output_dims:[ 6; 7 ] ()
  in
  let%op ho3 = hey2 ++ "...b|...i->...o => ...i|...o->...b" in
  let ctx = Train.forward_and_ctx backend ctx ho3 in
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ hey2;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ ho3;

  let%op ho4 = hey2 ++ "...b|...i->...o => i|o->b" in
  Train.forward_and_forget backend ctx ho4;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ ho4;
  let%op ho5 = hey ++ "...|...->...o => o" in
  Train.forward_and_forget backend ctx ho5;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ hey;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ ho5;
  let hey3 = TDSL.range_of_shape ~output_dims:[ 3; 4 ] () in
  let%op ho6 = hey3 ++ "...|...->...o => o" in
  Train.forward_and_forget backend ctx ho6;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ hey3;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ ho6;
  (* Broadcast with a shift. *)
  let hey4 = TDSL.range_of_shape ~input_dims:[ 2 ] ~output_dims:[ 3; 4 ] () in
  let%op ho7 = hey4 ++ "i->...o => ...io" in
  Train.forward_and_forget backend ctx ho7;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ hey4;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ ho7

let () =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
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
  let a = TDSL.range_of_shape ~batch_dims:[ 3 ] ~input_dims:[ 4 ] ~output_dims:[ 2 ] () in
  let b = TDSL.range_of_shape ~batch_dims:[ 3 ] ~input_dims:[ 1 ] ~output_dims:[ 4 ] () in
  let%op c = a *+ "...|i->...; ...|...->i => ...|i" b in
  Train.forward_and_forget backend ctx c;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ a;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ b;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ c;
  (* Broadcast with a shift. *)
  let d = TDSL.range_of_shape ~input_dims:[ 2 ] ~output_dims:[ 3 ] () in
  let e = TDSL.range_of_shape ~input_dims:[ 4 ] ~output_dims:[ 3 ] () in
  let%op f = d *+ "i->...;j->... => ...ij" e in
  Train.forward_and_forget backend ctx f;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ d;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ e;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ f

let () =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
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
  let hey = TDSL.range_of_shape ~batch_dims:[ 2 ] ~input_dims:[ 3 ] ~output_dims:[ 4 ] () in
  let%op ho = hey ++ "...|1->... => ...|..." in
  let ctx = Train.forward_and_ctx backend ctx ho in
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ hey;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ ho;
  let%op ho2 = hey ++ "...|...->... => ...|...->0" in
  let ctx = Train.forward_and_ctx backend ctx ho2 in
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ hey;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ ho2;

  let hey2 = TDSL.range_of_shape ~input_dims:[ 2 ] ~output_dims:[ 3 ] () in
  let%op ho3 = hey2 ++ "...|...->... => 0" in
  let ctx = Train.forward_and_ctx backend ctx ho3 in
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ hey2;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ ho3;
  let%op ho4 = hey2 ++ "i->j => i0j" in
  Train.forward_and_forget backend ctx ho4;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ ho4

let () =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
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
  let a = TDSL.range_of_shape ~batch_dims:[ 3 ] ~input_dims:[ 4 ] ~output_dims:[ 2 ] () in
  let b = TDSL.range_of_shape ~batch_dims:[ 3 ] ~input_dims:[ 1 ] ~output_dims:[ 4 ] () in
  let%op c = a *+ "...|i->1; ...|...->i => ...|i" b in
  Train.forward_and_forget backend ctx c;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ a;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ b;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ c

let () =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
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
  let ri = TDSL.range 3 in
  let%op ti = ri ++ "i=>i0" in
  (* Write position 2 of ti, otherwise shape inference concludes it's dim-1 and broadcasted. *)
  let%cd _ = ti =: 0 ++ "i=>i2" in
  let rj = TDSL.range 4 in
  let%op tj = rj ++ "j=>j1" in
  let rk = TDSL.range 5 in
  let%op tk = rk ++ "k=>k2" in
  let positions = TDSL.outer_sum "ijl;kl=>ijkl" (TDSL.outer_sum "il;jl=>ijl" ti tj) tk in
  Train.set_hosted tk.value;
  Train.forward_and_forget backend ctx positions;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ positions;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ ti;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ tk

let () =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
  let backend =
    (module Backend : Backend
      with type buffer_ptr = Backend.buffer_ptr
       and type dev = Backend.dev
       and type runner = Backend.runner
       and type event = Backend.event)
  in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  let a =
    TDSL.range_of_shape ~label:[ "a" ] ~batch_dims:[ 3 ] ~input_dims:[ 4 ] ~output_dims:[ 2 ] ()
  in
  let b =
    TDSL.range_of_shape ~label:[ "b" ] ~batch_dims:[ 3 ] ~input_dims:[ 2; 3 ] ~output_dims:[ 4 ] ()
  in
  let%op c = a *+ "...|i->1; ...|j...->i => ...|ij" b in
  Train.forward_and_forget backend ctx c;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ a;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ b;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ c

let () =
  Tensor.unsafe_reinitialize ();
  let module Backend = (val Backends.fresh_backend ()) in
  let backend =
    (module Backend : Backend
      with type buffer_ptr = Backend.buffer_ptr
       and type dev = Backend.dev
       and type runner = Backend.runner
       and type event = Backend.event)
  in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  let a = TDSL.range_of_shape ~label:[ "a" ] ~input_dims:[ 2 ] ~output_dims:[ 2 ] () in
  let b = TDSL.range_of_shape ~label:[ "b" ] ~input_dims:[ 2; 3; 4 ] ~output_dims:[ 2 ] () in
  let%op c = a *+ "i->1; ij...->0 => ...->ji" b in
  Train.forward_and_forget backend ctx c;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ a;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ b;
  Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default @@ c
