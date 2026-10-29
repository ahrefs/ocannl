open Base
open Ocannl
open Stdio
module Tn = Ir.Tnode
module IDX = Train.IDX
open Nn_blocks.DSL_modules
module CDSL = Train.CDSL
module Asgns = Ir.Assignments

module type Backend = Ir.Backend_intf.Backend

(* Makemore progression — Part 1 (Bigram).

   Corresponds to Karpathy's "Building makemore Part 1: bigrams". This file implements the
   neural-network formulation directly (a single weight matrix [w] producing logits via [exp (w *
   input + 1)], renormalized per-row to probabilities). The earlier counts-only notebook step from
   the lecture is intentionally omitted — the neural formulation already exercises the end-to- end
   training + generation loop. Karpathy additionally adds a small [0.01 * (W ** 2).mean ()]
   regularizer; we leave it out for clarity and the loss thresholds below already accommodate the
   absence. See [docs/makemore_tutorial.md] for the full progression. *)

let () =
  Utils.settings.fixed_state_for_init <- Some 13;
  Tensor.unsafe_reinitialize ();

  let bigrams = Dataprep.Names.get_all_bigrams () |> Dataprep.Names.bigrams_to_indices in
  Stdio.printf "bigrams: %d\n%!" (List.length bigrams);
  let batch_size = 1000 in
  let round_up_by = batch_size - (List.length bigrams % batch_size) in
  let bigrams = List.take bigrams round_up_by @ bigrams in

  let int_input, int_output = List.unzip bigrams in
  let input_size = List.length int_input in
  Stdio.printf "input_size: %d\n%!" input_size;

  let inputs =
    Nn_blocks.dense_one_hot_of_int_list ~num_classes:Dataprep.Names.dict_size int_input
  in
  let outputs =
    Nn_blocks.dense_one_hot_of_int_list ~num_classes:Dataprep.Names.dict_size int_output
  in

  let n_batches = input_size / batch_size in
  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in

  let%op input_gram = inputs @| batch_n in
  let%op output_gram = outputs @| batch_n in

  let%op mlp input =
    let counts = exp (({ w } + 1) * input) in
    counts /. (counts ++ "... => 0")
  in

  let%op output_probs = (mlp input_gram *. output_gram) ++ "... => 0" in
  let%op loss = neg (log output_probs) in
  let%op batch_loss = (loss ++ "...|... => |->0") /. !..batch_size in

  (* When using as a tutorial, try both with the following source line included and commented out.
     Run with the option --ocannl_output_debug_files_in_build_directory=true and check the
     build_files/ directory for the generated code. Materializing the intermediates lets kernel
     fission cut the step at every dependency edge, so each nest gets its own launch geometry — the
     autotuner then makes this ~9x faster than the whole-routine default on Metal. (The historical
     FIXME(#344) Metal buffer-argument limit no longer applies: buffers bind through the pooled slot
     table.) *)
  Train.every_non_literal_materialized batch_loss;
  let update = Train.grad_update batch_loss in
  let%op learning_rate = 1 in
  let sgd = Train.sgd_update ~learning_rate batch_loss in

  let ctx = Context.auto () in
  let ctx = Train.init_params ctx bindings batch_loss in
  (* Tune the step schedule empirically. [rounds:0] keeps only the preset seed candidates, which all
     preserve reduction order — the trained values are schedule-invariant, so this file's expected
     output stays deterministic no matter which seed wins. [timing_ctx] gives the tuner a scratch
     lineage (with its own freshly initialized parameter buffers) to time candidates against, so the
     timing runs cannot perturb the real training state. *)
  (* The crown and the finals outcome go to stderr ([Tune_crown]): a golden diff on another box is
     traced to the seed that won without rerunning under [autotune_log]. *)
  let scratch = Train.init_params (Context.auto ()) bindings batch_loss in
  let ctx, sgd_step =
    Autotune.tune ~rounds:0 ~timing_ctx:scratch
      ~report:(Tune_crown.eprint ~site:"bigram")
      ctx
      (Asgns.sequence [ update; sgd ])
      bindings
  in
  (* Train.printf w ~with_grad:false; *)

  let open Operation.At in
  let batch_ref = IDX.find_exn sgd_step.Context.bindings batch_n in
  let epochs = 10 in
  (* Exact trajectory digits to stderr, claims on stdout (gh-ocannl-725): every one of these numbers
     is a trained loss reached through a long floating-point reduction, so its low digits depend on
     reduction order -- backend, SIMD width, worker count -- and NO fixed print precision is
     portable; four significant digits of an epoch sum near 570 is a tie waiting for the next
     backend. What the trajectory was there to show is on stdout below: the losses stay finite, the
     epoch sum falls across training, and it lands under a threshold well inside the trained regime
     (a bigram model that learned nothing sits at 229000/1000 * ln 27 ~ 754). *)
  let epoch_losses = ref [] in
  (* Two-sided, because an upper bound alone is one-sided: a dropped negation or a backend sign
     error yields a FINITE NEGATIVE cross-entropy that falls, stays under any threshold, and would
     have been caught only by the digits this commit removed (Codex round 1, P2). Cross-entropy is
     -log p >= 0 by construction, so a materially negative value is invalid whatever its magnitude;
     the tolerance is for accumulated rounding at exactly zero, nothing more. *)
  let valid_loss x = Float.(is_finite x && x >= -1e-3) in
  let all_valid = ref true in
  for epoch = 0 to epochs do
    let epoch_loss = ref 0. in
    for batch = 0 to n_batches - 1 do
      batch_ref := batch;
      Train.run ctx sgd_step;
      let loss = (ctx, batch_loss).@[0] in
      epoch_loss := !epoch_loss +. loss;
      if not (valid_loss loss) then all_valid := false;
      if batch % 100 = 0 then
        Stdio.eprintf "Epoch %d, batch %d, loss=%.4g (not part of the golden)\n%!" epoch batch loss
    done;
    if not (valid_loss !epoch_loss) then all_valid := false;
    epoch_losses := !epoch_loss :: !epoch_losses;
    Stdio.eprintf "Epoch %d, epoch loss=%.4g (not part of the golden)\n%!" epoch !epoch_loss
  done;
  let epoch_losses = Array.of_list_rev !epoch_losses in
  Verdict.p "every batch and epoch loss is a finite, nonnegative cross-entropy" !all_valid;
  let first_loss = epoch_losses.(0) and last_loss = epoch_losses.(epochs) in
  Verdict.pf "epoch loss fell from epoch 0 to epoch %d" epochs Float.(last_loss < first_loss);
  let epoch_below = 600. in
  Verdict.pf "final epoch loss below %g" epoch_below Float.(last_loss < epoch_below);

  (* Train.printf_tree batch_loss; *)
  let counter_n, bindings = IDX.get_static_symbol IDX.empty in
  let%cd infer_probs = mlp { cha } in
  let%cd infer_step =
    infer_probs.forward;
    { dice } =: uniform_at !@counter_n
  in
  Train.set_materialized infer_probs.value;
  let ctx, infer_step = Train.to_routine sgd_step.Context.context bindings infer_step in
  let counter_ref = IDX.find_exn infer_step.Context.bindings counter_n in
  counter_ref := 0;

  let infer c =
    let c_one_hot = Dataprep.Names.char_to_one_hot c in
    ignore (Context.set_values ctx cha.value c_one_hot : Context.t);
    Int.incr counter_ref;
    Train.run ctx infer_step;
    let dice_value = (ctx, dice).@[0] in

    let max_i = List.length Dataprep.Names.letters_with_dot - 1 in
    let rec aux i sum =
      if i >= max_i then '.'
      else
        let prob = (ctx, infer_probs).@{[| i |]} in
        let new_sum = sum +. prob in
        if Float.compare new_sum dice_value > 0 then List.nth_exn Dataprep.Names.letters_with_dot i
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
