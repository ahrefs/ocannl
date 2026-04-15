open! Base
open Ocannl.Nn_blocks.DSL_modules

let () =
  let ctx = Context.auto () in
  let batch_size = 2 in
  let seq_len = 4 in
  let d_model = 16 in
  let num_heads = 2 in
  let d_ff = 32 in

  Stdio.printf "Testing decoder_only (2-layer stack)\n";

  (* decoder_only internally creates decoder_only_block instances,
     so this exercises both functions. *)
  let stack =
    Ocannl.Nn_blocks.decoder_only ~label:[ "test_stack" ] ~num_layers:2 ~num_heads ~d_k:d_model
      ~d_v:d_model ~d_ff ()
  in

  let input =
    TDSL.range_of_shape ~label:[ "input" ] ~batch_dims:[ batch_size; seq_len ] ~input_dims:[]
      ~output_dims:[ d_model ] ()
  in

  let mask =
    NTDSL.init ~l:"mask" ~prec:Ir.Ops.single ~b:[ seq_len ] ~i:[ seq_len ] ~o:[]
      ~f:(function
        | [| s; t |] -> if s >= t then 1. else 0.
        | _ -> failwith "unexpected mask indices")
      ()
  in

  let output = stack ~train_step:None input ~mask in
  let _ctx = Ocannl.Train.forward_once ctx output in

  Stdio.printf "Output shape:\n%s\n%!"
    (Sexp.to_string_hum ([%sexp_of: Shape.t] output.Tensor.shape))
