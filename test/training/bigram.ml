open Base
open Ocannl
open Stdio
open Bigarray
module Tn = Ir.Tnode
module IDX = Train.IDX
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module CDSL = Train.CDSL
module Asgns = Ir.Assignments

module type Backend = Ir.Backend_intf.Backend

let tensor_of_int_list lst =
  let len = List.length lst in
  let arr = lst |> List.map ~f:Float.of_int |> Array.of_list in
  (* Metal backend doesn't support double precision. *)
  let genarray =
    Genarray.create Bigarray.Float32 Bigarray.c_layout [| len; Datasets.Names.dict_size |]
  in
  (* convert to one-hot vectors *)
  for i = 0 to len - 1 do
    Genarray.set genarray [| i; Int.of_float arr.(i) |] 1.
  done;
  TDSL.rebatch ~l:"tensor" (Ir.Ndarray.as_array Ir.Ops.Single genarray) ()

let () =
  let seed = 13 in
  Utils.settings.fixed_state_for_init <- Some seed;

  let bigrams = Datasets.Names.get_all_bigrams () |> Datasets.Names.bigrams_to_indices in
  Stdio.printf "bigrams: %d\n%!" (List.length bigrams);
  let batch_size = 1000 in
  let round_up_by = batch_size - (List.length bigrams % batch_size) in
  let bigrams = List.take bigrams round_up_by @ bigrams in

  let int_input, int_output = List.unzip bigrams in
  let input_size = List.length int_input in
  Stdio.printf "input_size: %d\n%!" input_size;

  let inputs = tensor_of_int_list int_input in
  let outputs = tensor_of_int_list int_output in

  let n_batches = input_size / batch_size in
  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in

  let%op input = inputs @| batch_n in
  let%op output = outputs @| batch_n in

  let%op mlp input =
    let counts = exp (("w" + 1) * input) in
    counts /. (counts ++ "...|... => ...|0")
  in

  let%op output_probs = (mlp input *. output) ++ "...|... => ...|0" in
  let%op loss = neg (log output_probs) in
  let%op batch_loss = (loss ++ "...|... => 0") /. !..batch_size in

  Train.every_non_literal_on_host batch_loss;

  let update = Train.grad_update batch_loss in
  let%op learning_rate = 1 in
  let sgd = Train.sgd_update ~learning_rate batch_loss in

  let module Backend = (val Backends.fresh_backend ()) in
  let ctx = Train.init_params (module Backend) bindings batch_loss in
  let sgd_step = Train.to_routine (module Backend) ctx bindings (Asgns.sequence [ update; sgd ]) in
  Train.printf w ~with_grad:false;

  let open Operation.At in
  let batch_ref = IDX.find_exn sgd_step.bindings batch_n in
  for epoch = 0 to 100 do
    for batch = 0 to n_batches - 1 do
      batch_ref := batch;
      Train.run sgd_step
    done;
    Stdio.printf "Epoch %d, loss=%f\n%!" epoch batch_loss.@[0]
  done;
  Train.printf_tree batch_loss;

  let%cd infer_probs = mlp "cha" in
  Train.set_on_host infer_probs.value;
  let infer_probs_routine =
    Train.to_routine (module Backend) sgd_step.context IDX.empty infer_probs.forward
  in
  let infer c =
    let c_one_hot = Datasets.Names.char_to_one_hot c in
    Tn.set_values cha.value c_one_hot;
    Train.run infer_probs_routine;

    let dice = Random.float 1. in

    let rec aux i sum =
      let prob = infer_probs.@{[| i |]} in
      let new_sum = sum +. prob in
      if Float.compare new_sum dice > 0 then List.nth_exn Datasets.Names.letters_with_dot i
      else aux (i + 1) new_sum
    in

    aux 0 0.
  in

  let gen_name () =
    let rec aux c name =
      if (Char.equal c '.' || Char.equal c ' ') && not (String.equal name "") then name
      else
        let next_char = infer c in
        aux next_char (name ^ String.make 1 c)
    in
    let name_with_dot = aux '.' "" in
    String.drop_prefix name_with_dot 1
  in

  let names = Array.init 20 ~f:(fun _ -> gen_name ()) in
  Array.iter names ~f:print_endline
