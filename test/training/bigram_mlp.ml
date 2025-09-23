open Base
open Ocannl
open Stdio
open Bigarray
module Tn = Ir.Tnode
module IDX = Train.IDX
open Nn_blocks.DSL_modules
module CDSL = Train.CDSL
module Asgns = Ir.Assignments

module type Backend = Ir.Backend_intf.Backend

let tensor_of_int_list lst =
  let len = List.length lst in
  let arr = lst |> Array.of_list in
  (* Metal backend doesn't support double precision. *)
  let genarray =
    Genarray.create Bigarray.Float32 Bigarray.c_layout [| len; Datasets.Names.dict_size |]
  in
  (* convert to one-hot vectors *)
  for i = 0 to len - 1 do
    Genarray.set genarray [| i; arr.(i) |] 1.
  done;
  TDSL.rebatch ~l:"tensor" (Ir.Ndarray.as_array Ir.Ops.Single genarray) ()

let () =
  Utils.settings.fixed_state_for_init <- Some 13;
  Tensor.unsafe_reinitialize ();

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
  let step_n, bindings = IDX.get_static_symbol bindings in
  let epochs = 5 in
  let steps = epochs * n_batches in

  let%op input_gram = inputs @| batch_n in
  let%op output_gram = outputs @| batch_n in

  let%op mlp input =
    let counts = exp (({ w2 } + 1) * relu ({ b1; o = [ 24 ] } + ({ w1 } * input))) in
    counts /. (counts ++ "...|... => ...|0")
  in

  let%op output_probs = (mlp input_gram *. output_gram) ++ "...|... => ...|0" in
  let%op loss = neg (log output_probs) in
  let%op batch_loss = (loss ++ "...|... => 0") /. !..batch_size in

  (* When using as a tutorial, try both with the following source line included and commented out.
     Run with the option --ocannl_output_debug_files_in_build_directory=true and check the
     build_files/ directory for the generated code. *)
  (* FIXME(#344): When uncommented, this exceeds the number of buffer arguments supported by the Metal backend. *)
  (* Train.every_non_literal_on_host batch_loss; *)
  let update = Train.grad_update batch_loss in
  let%op learning_rate = 1.5 *. ((1.5 *. !..steps) - !@step_n) /. !..steps in
  let sgd = Train.sgd_update ~learning_rate (* ~weight_decay:0.0001 *) batch_loss in

  let ctx = Context.auto () in
  let ctx = Train.init_params ctx bindings batch_loss in
  let sgd_step = Train.to_routine ctx bindings (Asgns.sequence [ update; sgd ]) in

  let open Operation.At in
  let batch_ref = IDX.find_exn (Context.bindings sgd_step) batch_n in
  let step_ref = IDX.find_exn (Context.bindings sgd_step) step_n in
  let epoch_loss_target_limits = [| 595.; 567.; 565.; 563.; 562.; 561.; 560.; 559.; 558.; 557. |] in
  for epoch = 0 to epochs - 1 do
    let epoch_loss = ref 0. in
    for batch = 0 to n_batches - 1 do
      batch_ref := batch;
      Train.run ctx sgd_step;
      let loss = batch_loss.@[0] in
      epoch_loss := !epoch_loss +. loss;
      Int.incr step_ref
    done;
    Stdio.printf "Epoch %d, epoch loss below %g=%b%s\n%!" epoch epoch_loss_target_limits.(epoch)
      Float.(epoch_loss_target_limits.(epoch) > !epoch_loss)
      (if Float.(epoch_loss_target_limits.(epoch) > !epoch_loss) then ""
       else ", actual loss: " ^ Float.to_string !epoch_loss)
  done;

  (* Train.printf_tree batch_loss; *)
  let counter_n, bindings = IDX.get_static_symbol IDX.empty in
  let%cd infer_probs = mlp { cha } in
  let%cd infer_step =
    infer_probs.forward;
    { dice } =: uniform_at !@counter_n
  in
  Train.set_on_host infer_probs.value;
  let infer_step = Train.to_routine (Context.context sgd_step) bindings infer_step in
  let counter_ref = IDX.find_exn (Context.bindings infer_step) counter_n in
  counter_ref := 0;

  let infer c =
    let c_one_hot = Datasets.Names.char_to_one_hot c in
    Tn.set_values cha.value c_one_hot;
    Int.incr counter_ref;
    Train.run ctx infer_step;
    let dice_value = dice.@[0] in

    let rec aux i sum =
      let prob = infer_probs.@{[| i |]} in
      let new_sum = sum +. prob in
      if Float.compare new_sum dice_value > 0 then List.nth_exn Datasets.Names.letters_with_dot i
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

  (* Generate very few names because different hardware backends diverge quickly. *)
  let names = Array.init 3 ~f:(fun _ -> gen_name ()) in
  Array.iter names ~f:print_endline
