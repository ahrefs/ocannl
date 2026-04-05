open! Base
open Ocannl.Nn_blocks.DSL_modules
module At = Ocannl_tensor.Operation.At

(* === Test 1: Deinterleave roundtrip === *)
let () =
  Stdio.printf "=== Test 1: Deinterleave roundtrip ===\n";
  let ctx = Context.auto () in
  let x_deint =
    NTDSL.init ~l:"x_deint" ~prec:Ir.Ops.single ~b:[] ~i:[] ~o:[ 6 ]
      ~f:(function [| i |] -> Float.of_int (i + 1) | _ -> assert false)
      ()
  in
  let%op roundtrip = interleave (deinterleave_even x_deint) (deinterleave_odd x_deint) in
  let _ctx = Ocannl.Train.forward_once ctx roundtrip in
  Stdio.printf "Original: ";
  for i = 0 to 5 do Stdio.printf "%.0f " At.(x_deint.@{[| i |]}) done;
  Stdio.printf "\nRoundtrip: ";
  for i = 0 to 5 do Stdio.printf "%.0f " At.(roundtrip.@{[| i |]}) done;
  let ok = ref true in
  for i = 0 to 5 do
    if Float.(abs (At.(x_deint.@{[| i |]}) - At.(roundtrip.@{[| i |]})) > 1e-5) then ok := false
  done;
  Stdio.printf "\nMatch: %b\n\n" !ok

(* === Test 2: RoPE basic shape and numeric === *)
let () =
  Tensor.unsafe_reinitialize ();
  Stdio.printf "=== Test 2: RoPE basic shape and numeric ===\n";
  let ctx = Context.auto () in
  let d_k = 4 in
  let seq_len = 3 in
  let freqs = Ocannl.Nn_blocks.rope_frequencies ~half_d:(d_k / 2) () in
  let positions = Ocannl.Nn_blocks.position_indices ~seq_len () in
  let x =
    NTDSL.init ~l:"x" ~prec:Ir.Ops.single ~b:[ seq_len ] ~i:[] ~o:[ d_k ]
      ~f:(function
        | [| pos; i |] -> Float.of_int ((pos * d_k) + i + 1)
        | _ -> assert false)
      ()
  in
  let rotated = Ocannl.Nn_blocks.rope ~freqs ~positions x in
  let _ctx = Ocannl.Train.forward_once ctx rotated in
  (* Position 0: identity (all angles = 0) *)
  Stdio.printf "Position 0 (should match [1,2,3,4]):\n";
  for i = 0 to d_k - 1 do
    Stdio.printf "  rotated[0,%d]=%.2f\n" i At.(rotated.@{[| 0; i |]})
  done;
  Stdio.printf "Position 1 (rotated [5,6,7,8]):\n";
  for i = 0 to d_k - 1 do
    Stdio.printf "  rotated[1,%d]=%.4f\n" i At.(rotated.@{[| 1; i |]})
  done;
  Stdio.printf "\n"

(* === Test 3: Sinusoidal encoding === *)
let () =
  Tensor.unsafe_reinitialize ();
  Stdio.printf "=== Test 3: Sinusoidal encoding ===\n";
  let ctx = Context.auto () in
  let enc = Ocannl.Nn_blocks.sinusoidal_position_encoding ~d_model:8 ~max_len:4 () in
  let _ctx = Ocannl.Train.forward_once ctx enc in
  Stdio.printf "PE(0,0)=%.4f PE(0,1)=%.4f PE(1,0)=%.4f PE(1,1)=%.4f\n\n"
    At.(enc.@{[| 0; 0 |]}) At.(enc.@{[| 0; 1 |]})
    At.(enc.@{[| 1; 0 |]}) At.(enc.@{[| 1; 1 |]})

(* === Test 4: RoPE with multi_head_attention === *)
let () =
  Tensor.unsafe_reinitialize ();
  Stdio.printf "=== Test 4: RoPE attention ===\n";
  let ctx = Context.auto () in
  (* Follow existing attention_test.ml pattern: d_k = d_model *)
  let d_model = 16 in
  let num_heads = 2 in
  let d_k = d_model in
  let seq_len = 4 in
  let freqs = Ocannl.Nn_blocks.rope_frequencies ~half_d:(d_k / 2) () in
  let positions = Ocannl.Nn_blocks.position_indices ~seq_len () in
  let attn =
    Ocannl.Nn_blocks.multi_head_attention ~label:[ "rope_attn" ] ~num_heads ~d_k ~d_v:d_k
      ~pos_embed:(RoPE { freqs; positions }) ()
  in
  let x =
    TDSL.range_of_shape ~label:[ "x" ] ~batch_dims:[ 1; seq_len ] ~input_dims:[]
      ~output_dims:[ d_model ] ()
  in
  let mask =
    NTDSL.init ~l:"mask" ~prec:Ir.Ops.single ~b:[ seq_len ] ~i:[ seq_len ] ~o:[]
      ~f:(function [| s; t |] -> if s >= t then 1. else 0. | _ -> assert false)
      ()
  in
  (* Add x to output to constrain output shape via shape inference, like attention_test.ml *)
  let%op out = x + attn ~train_step:None ~mask x in
  let _ctx = Ocannl.Train.forward_once ctx out in
  Stdio.printf "RoPE attention output shape: %s\n\n"
    (Sexp.to_string_hum ([%sexp_of: Shape.t] out.Tensor.shape))

(* === Test 5: PoPE with multi_head_attention === *)
let () =
  Tensor.unsafe_reinitialize ();
  Stdio.printf "=== Test 5: PoPE attention ===\n";
  let ctx = Context.auto () in
  let d_model = 16 in
  let num_heads = 2 in
  let d_k = d_model in
  let seq_len = 4 in
  let freqs = Ocannl.Nn_blocks.rope_frequencies ~half_d:(d_k / 2) () in
  let positions = Ocannl.Nn_blocks.position_indices ~seq_len () in
  let attn =
    Ocannl.Nn_blocks.multi_head_attention ~label:[ "pope_attn" ] ~num_heads ~d_k ~d_v:d_k
      ~pos_embed:(PoPE { freqs; positions }) ()
  in
  let x =
    TDSL.range_of_shape ~label:[ "x" ] ~batch_dims:[ 1; seq_len ] ~input_dims:[]
      ~output_dims:[ d_model ] ()
  in
  let%op out = x + attn ~train_step:None x in
  let _ctx = Ocannl.Train.forward_once ctx out in
  Stdio.printf "PoPE attention output shape: %s\n\n"
    (Sexp.to_string_hum ([%sexp_of: Shape.t] out.Tensor.shape))

(* === Test 6: Gradient flow through RoPE === *)
let () =
  Tensor.unsafe_reinitialize ();
  Stdio.printf "=== Test 6: Gradient flow ===\n";
  let ctx = Context.auto () in
  let d_model = 16 in
  let num_heads = 2 in
  let d_k = d_model in
  let seq_len = 4 in
  let freqs = Ocannl.Nn_blocks.rope_frequencies ~half_d:(d_k / 2) () in
  let positions = Ocannl.Nn_blocks.position_indices ~seq_len () in
  let attn =
    Ocannl.Nn_blocks.multi_head_attention ~label:[ "grad_attn" ] ~num_heads ~d_k ~d_v:d_k
      ~pos_embed:(RoPE { freqs; positions }) ()
  in
  let x =
    TDSL.range_of_shape ~label:[ "x" ] ~batch_dims:[ 1; seq_len ] ~input_dims:[]
      ~output_dims:[ d_model ] ()
  in
  let%op out = x + attn ~train_step:None x in
  let%op loss = out ++ "...|... => 0" in
  let _ctx = Ocannl.Train.update_once ctx loss in
  Stdio.printf "Loss: %.4f\n" At.(loss.@{[| 0 |]});
  Stdio.printf "freqs has grad: %b (expect false)\n" (Option.is_some freqs.Tensor.diff);
  Stdio.printf "positions has grad: %b (expect false)\n\n" (Option.is_some positions.Tensor.diff)

(* Transformer default regression is covered by transformer_test.ml *)
