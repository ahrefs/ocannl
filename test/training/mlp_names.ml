open Base
open Ocannl
open Stdio
module Tn = Ir.Tnode
module IDX = Train.IDX
open Nn_blocks.DSL_modules
module CDSL = Train.CDSL
module Asgns = Ir.Assignments

module type Backend = Ir.Backend_intf.Backend

(* Makemore progression — Part 2 (Bengio MLP on Names).

   Corresponds to Karpathy's "Building makemore Part 2: MLP". Each prediction is conditioned on a
   context window of [block_size] preceding characters, each of which is mapped through a learned
   embedding table [c] of width [embed_dim]. The concatenated context embeddings are fed through a
   [tanh]-activated hidden layer and then a linear output. Shape inference handles the "flattening"
   of the [block_size, embed_dim] input axes into the hidden linear weight — no manual reshape
   needed.

   The dataset is split deterministically 80/10/10 over [Dataprep.Names]; we report final train /
   dev / test NLL numerically plus a coarse below-threshold boolean guard for `.expected`. See
   [docs/makemore_tutorial.md]. *)

let block_size = 3
let embed_dim = 10
let hid_dim = 200
let vocab_size = Dataprep.Names.dict_size
let batch_size = 1000
let epochs = 15
let split_seed = 42

(* === Data preparation === *)

(** Slide a [block_size + 1] window over [pad * block_size @ name @ ['.']] and emit
    [(context_indices, target_index)] pairs. *)
let name_to_contexts name =
  let padded = List.init block_size ~f:(fun _ -> '.') @ String.to_list name @ [ '.' ] in
  let n = List.length padded in
  let indices = Array.of_list (List.map padded ~f:Dataprep.Names.char_index) in
  let pairs = ref [] in
  for i = 0 to n - block_size - 1 do
    let ctx = Array.sub indices ~pos:i ~len:block_size in
    let tgt = indices.(i + block_size) in
    pairs := (ctx, tgt) :: !pairs
  done;
  List.rev !pairs

(** Deterministic Fisher–Yates shuffle using the given seed. *)
let shuffle_names names ~seed =
  let rng = Random.State.make [| seed |] in
  let a = Array.of_list names in
  let n = Array.length a in
  for i = n - 1 downto 1 do
    let j = Random.State.int rng (i + 1) in
    let tmp = a.(i) in
    a.(i) <- a.(j);
    a.(j) <- tmp
  done;
  Array.to_list a

(** 80/10/10 split over a deterministically-shuffled name list. *)
let split_names names =
  let shuffled = shuffle_names names ~seed:split_seed in
  let n = List.length shuffled in
  let n_train = n * 8 / 10 in
  let n_dev = n / 10 in
  let train = List.take shuffled n_train in
  let rest = List.drop shuffled n_train in
  let dev = List.take rest n_dev in
  let test = List.drop rest n_dev in
  (train, dev, test)

(** Flatten a list of names into aligned [(contexts, targets)] int arrays, then truncate to a
    multiple of [batch_size]. *)
let names_to_examples names =
  let pairs = List.concat_map names ~f:name_to_contexts in
  let n = List.length pairs in
  let n = n - (n % batch_size) in
  let pairs = List.take pairs n in
  let contexts = Array.create ~len:(n * block_size) 0 in
  let targets = Array.create ~len:n 0 in
  List.iteri pairs ~f:(fun i (ctx, tgt) ->
      for j = 0 to block_size - 1 do
        contexts.((i * block_size) + j) <- ctx.(j)
      done;
      targets.(i) <- tgt);
  (contexts, targets, n)

(** Fill a per-batch flat buffer of context token IDs (shape [batch_size * block_size]). gh-343: the
    embedding is now a logical one-hot of these IDs, so the input is compact token IDs rather than a
    dense [batch_size * block_size * vocab_size] one-hot. *)
let fill_ctx_ids buf contexts ~offset =
  for i = 0 to batch_size - 1 do
    for t = 0 to block_size - 1 do
      buf.((i * block_size) + t) <- Float.of_int contexts.(((offset + i) * block_size) + t)
    done
  done

(** Fill a per-batch flat one-hot buffer for targets (shape [batch_size * vocab_size]). *)
let fill_tgt_one_hot buf targets ~offset =
  Array.fill buf ~pos:0 ~len:(Array.length buf) 0.;
  for i = 0 to batch_size - 1 do
    buf.((i * vocab_size) + targets.(offset + i)) <- 1.
  done

(* === Main === *)

let () =
  Utils.settings.fixed_state_for_init <- Some 3;
  Tensor.unsafe_reinitialize ();

  let names = Dataprep.Names.read_names () in
  printf "Names loaded: %d\n%!" (List.length names);
  let train_names, dev_names, test_names = split_names names in
  let train_ctx, train_tgt, n_train = names_to_examples train_names in
  let dev_ctx, dev_tgt, n_dev = names_to_examples dev_names in
  let test_ctx, test_tgt, n_test = names_to_examples test_names in
  printf "train/dev/test examples (after batch truncation): %d/%d/%d\n%!" n_train n_dev n_test;

  let n_batches = n_train / batch_size in
  let step_n, bindings = IDX.get_static_symbol IDX.empty in

  (* === Data tensors === *)
  let make_ctx_tensor label =
    let open Bigarray in
    (* gh-343: token IDs, one per context position (no dense vocab_size axis). *)
    let ga = Genarray.create Float32 c_layout [| batch_size; block_size |] in
    Bigarray.Genarray.fill ga 0.;
    let nd = Ir.Ndarray.as_array Ir.Ops.Single ga in
    Tensor.term ~init_data:(Reshape nd) ~grad_spec:If_needed ~label:[ label ]
      ~batch_dims:[ batch_size; block_size ] ~input_dims:[] ~output_dims:[] ()
  in
  let make_tgt_tensor label =
    let open Bigarray in
    let ga = Genarray.create Float32 c_layout [| batch_size; vocab_size |] in
    Bigarray.Genarray.fill ga 0.;
    let nd = Ir.Ndarray.as_array Ir.Ops.Single ga in
    Tensor.term ~init_data:(Reshape nd) ~grad_spec:If_needed ~label:[ label ]
      ~batch_dims:[ batch_size ] ~input_dims:[] ~output_dims:[ vocab_size ] ()
  in
  let input_batch = make_ctx_tensor "input_batch" in
  let target_batch = make_tgt_tensor "target_batch" in

  (* === Model === embed: per-position character embedding. [input] has [block_size] as a batch
     axis; ordinary matmul with [c : vocab_size -> embed_dim] gives embeddings per-position. hidden:
     contracts [block_size] (batch axis of embed) and [embed_dim] (output axis of embed) into
     [hid_dim] via a block-weighted linear layer. This mirrors Bengio's "concatenate context
     embeddings and linear" — here expressed as an explicit einsum contraction with [w1] shaped
     [block_size, embed_dim -> hid_dim]. logits: final linear projection to [vocab_size]. *)
  (* gh-343: [input] carries token IDs (one per context position); the logical one-hot
     [range vocab_size == input] feeds the embedding-table contraction, which the low-level optimizer
     collapses into a direct gather instead of materializing a dense [vocab_size] one-hot. *)
  let%op embed input =
    { c; o = [ embed_dim ] } * Nn_blocks.one_hot_of_ids ~num_classes:vocab_size input
  in
  let%op hidden x =
    tanh (embed x +* { w1 } "bs|->e; |se->h => b|->h" [ "s"; "e" ] + { b1; o = [ hid_dim ] })
  in
  let%op logits x = ({ w2 } * hidden x) + { b2 } in

  let train_logits = logits input_batch in
  (* Numerically stable log-softmax cross-entropy, matching transformer_names.ml. *)
  let%op max_l = train_logits @^^ "... | ... => ... | 0" in
  let%op shifted = train_logits - max_l in
  let%op lse = log (exp shifted ++ "... | ... => ... | 0") in
  let%op log_probs = shifted - lse in
  let%op nll = neg ((target_batch *. log_probs) ++ "... | ... => 0") in
  let%op batch_loss = (nll ++ "... => 0") /. !..batch_size in

  (* FIXME(#344): When uncommented, this exceeds the number of buffer arguments
     supported by the Metal backend. Carried forward from bigram_mlp.ml. *)
  (* Train.every_non_literal_materialized batch_loss; *)
  let update = Train.grad_update batch_loss in
  let steps = epochs * n_batches in
  let%op learning_rate = 0.1 *. ((1.5 *. !..steps) - !@step_n) /. !..steps in
  let sgd = Train.sgd_update ~learning_rate batch_loss in

  let ctx = Context.auto () in
  let ctx = Train.init_params ctx bindings batch_loss in
  (* Recenter all-positive uniform1 inits to [-0.25, 0.25). Same mitigation as transformer_names.ml
     / fsm_transformer.ml — OCANNL's default init produces non-negative weights, which makes the
     hidden preactivation saturate and traps SGD at a high-loss plateau. *)
  Set.iter batch_loss.Tensor.params ~f:(fun p ->
      let tn = p.Tensor.value in
      Train.set_materialized tn;
      let vals = Context.get_values ctx tn in
      Array.iteri vals ~f:(fun i v -> vals.(i) <- 0.5 *. (v -. 0.5));
      ignore (Context.set_values ctx tn vals : Context.t));

  let sgd_step = Train.to_routine ctx bindings (Asgns.sequence [ update; sgd ]) in
  let ctx = Context.context sgd_step in
  let open Operation.At in
  let step_ref = IDX.find_exn (Context.bindings sgd_step) step_n in
  Train.set_materialized batch_loss.value;

  let ctx_buf = Array.create ~len:(batch_size * block_size) 0. in
  let tgt_buf = Array.create ~len:(batch_size * vocab_size) 0. in

  (* === Training === *)
  (* Coarse threshold guard: monotonically decreasing upper bound. *)
  let epoch_loss_limit epoch =
    if epoch = 0 then 4.0 else if epoch < 5 then 3.0 else if epoch < 10 then 2.6 else 2.5
  in
  for epoch = 0 to epochs - 1 do
    let epoch_loss = ref 0. in
    for batch = 0 to n_batches - 1 do
      let offset = batch * batch_size in
      fill_ctx_ids ctx_buf train_ctx ~offset;
      fill_tgt_one_hot tgt_buf train_tgt ~offset;
      ignore (Context.set_values ctx input_batch.value ctx_buf : Context.t);
      ignore (Context.set_values ctx target_batch.value tgt_buf : Context.t);
      Train.run ctx sgd_step;
      epoch_loss := !epoch_loss +. (ctx, batch_loss).@[0];
      Int.incr step_ref
    done;
    let mean_loss = !epoch_loss /. Float.of_int n_batches in
    let limit = epoch_loss_limit epoch in
    printf "Epoch %d, mean train loss=%.4f below %g=%b\n%!" epoch mean_loss limit
      Float.(mean_loss < limit)
  done;

  (* === Evaluation on train/dev/test === Build a separate forward-only subgraph with fresh
     input/target tensors so the trained batch_loss's forward code (already consumed by grad_update)
     isn't re-used. Weights/embeddings are shared because the model is invoked a second time under
     %cd. *)
  let eval_input =
    let open Bigarray in
    (* gh-343: context token IDs (see [make_ctx_tensor]). *)
    let ga = Genarray.create Float32 c_layout [| batch_size; block_size |] in
    Bigarray.Genarray.fill ga 0.;
    let nd = Ir.Ndarray.as_array Ir.Ops.Single ga in
    Tensor.term ~init_data:(Reshape nd) ~grad_spec:Prohibit_grad ~label:[ "eval_input" ]
      ~batch_dims:[ batch_size; block_size ] ~input_dims:[] ~output_dims:[] ()
  in
  let eval_target =
    let open Bigarray in
    let ga = Genarray.create Float32 c_layout [| batch_size; vocab_size |] in
    Bigarray.Genarray.fill ga 0.;
    let nd = Ir.Ndarray.as_array Ir.Ops.Single ga in
    Tensor.term ~init_data:(Reshape nd) ~grad_spec:Prohibit_grad ~label:[ "eval_target" ]
      ~batch_dims:[ batch_size ] ~input_dims:[] ~output_dims:[ vocab_size ] ()
  in
  let%cd eval_logits = logits eval_input in
  let%cd eval_max_l = eval_logits @^^ "... | ... => ... | 0" in
  let%cd eval_shifted = eval_logits - eval_max_l in
  let%cd eval_lse = log (exp eval_shifted ++ "... | ... => ... | 0") in
  let%cd eval_log_probs = eval_shifted - eval_lse in
  let%cd eval_nll = neg ((eval_target *. eval_log_probs) ++ "... | ... => 0") in
  let%cd eval_loss = (eval_nll ++ "... => 0") /. !..batch_size in
  Train.set_materialized eval_loss.value;
  Train.set_materialized eval_input.value;
  Train.set_materialized eval_target.value;
  let%cd eval_comp =
    ~~("mlp_names eval";
       eval_loss.forward)
  in
  let eval_step = Train.to_routine (Context.context sgd_step) IDX.empty eval_comp in
  let ctx = Context.context eval_step in
  let mean_loss_over (ctx_arr, tgt_arr, n) =
    let nb = n / batch_size in
    if nb = 0 then 0.0
    else begin
      let acc = ref 0. in
      for batch = 0 to nb - 1 do
        let offset = batch * batch_size in
        fill_ctx_ids ctx_buf ctx_arr ~offset;
        fill_tgt_one_hot tgt_buf tgt_arr ~offset;
        ignore (Context.set_values ctx eval_input.value ctx_buf : Context.t);
        ignore (Context.set_values ctx eval_target.value tgt_buf : Context.t);
        Train.run ctx eval_step;
        acc := !acc +. (ctx, eval_loss).@[0]
      done;
      !acc /. Float.of_int nb
    end
  in

  let final_train = mean_loss_over (train_ctx, train_tgt, n_train) in
  let final_dev = mean_loss_over (dev_ctx, dev_tgt, n_dev) in
  let final_test = mean_loss_over (test_ctx, test_tgt, n_test) in
  (* Thresholds ~3% above observed sync_cc values under the fixed seed. *)
  let train_below = 2.5 in
  let dev_below = 2.55 in
  let test_below = 2.55 in
  printf "Final train loss=%.4f train_below=%b\n%!" final_train Float.(final_train < train_below);
  printf "Final dev   loss=%.4f dev_below=%b\n%!" final_dev Float.(final_dev < dev_below);
  printf "Final test  loss=%.4f test_below=%b\n%!" final_test Float.(final_test < test_below);

  (* === Generation === Autoregressive sampling from a rolling [block_size] context. *)
  let infer_input =
    let open Bigarray in
    (* gh-343: context token IDs (see [make_ctx_tensor]). *)
    let ga = Genarray.create Float32 c_layout [| 1; block_size |] in
    Bigarray.Genarray.fill ga 0.;
    let nd = Ir.Ndarray.as_array Ir.Ops.Single ga in
    Tensor.term ~init_data:(Reshape nd) ~grad_spec:Prohibit_grad ~label:[ "infer_input" ]
      ~batch_dims:[ 1; block_size ] ~input_dims:[] ~output_dims:[] ()
  in
  let counter_n, infer_bindings = IDX.get_static_symbol IDX.empty in
  let%cd infer_logits = logits infer_input in
  let%cd infer_comp =
    ~~("names infer";
       infer_logits.forward;
       { dice } =: uniform_at !@counter_n)
  in
  Train.set_materialized infer_logits.value;
  Train.set_materialized infer_input.value;
  let infer_step = Train.to_routine (Context.context eval_step) infer_bindings infer_comp in
  let ctx = Context.context infer_step in
  let counter_ref = IDX.find_exn (Context.bindings infer_step) counter_n in
  counter_ref := 0;

  let dot_idx = Dataprep.Names.char_index '.' in
  let set_ctx_ids context =
    (* gh-343: feed context token IDs, one per position. *)
    let buf = Array.create ~len:block_size 0. in
    for t = 0 to block_size - 1 do
      buf.(t) <- Float.of_int context.(t)
    done;
    ignore (Context.set_values ctx infer_input.value buf : Context.t)
  in

  let max_len = 20 in
  let gen_name () =
    let context = Array.create ~len:block_size dot_idx in
    let buf = Buffer.create 16 in
    let rec aux steps =
      if steps >= max_len then Buffer.contents buf
      else begin
        set_ctx_ids context;
        Int.incr counter_ref;
        Train.run ctx infer_step;
        let dice_value = (ctx, dice).@[0] in
        let logits_arr = Array.init vocab_size ~f:(fun v -> (ctx, infer_logits).@{[| 0; v |]}) in
        let max_logit = Array.fold logits_arr ~init:Float.neg_infinity ~f:Float.max in
        let exp_logits = Array.map logits_arr ~f:(fun l -> Float.exp (l -. max_logit)) in
        let sum_exp = Array.fold exp_logits ~init:0. ~f:( +. ) in
        let probs = Array.map exp_logits ~f:(fun e -> e /. sum_exp) in
        let max_i = vocab_size - 1 in
        let rec sample i acc =
          if i >= max_i then i
          else
            let new_acc = acc +. probs.(i) in
            if Float.(new_acc > dice_value) then i else sample (i + 1) new_acc
        in
        let sampled = sample 0 0. in
        let sampled_char = List.nth_exn Dataprep.Names.letters_with_dot sampled in
        if Char.equal sampled_char '.' || Char.equal sampled_char ' ' then Buffer.contents buf
        else begin
          Buffer.add_char buf sampled_char;
          for t = 0 to block_size - 2 do
            context.(t) <- context.(t + 1)
          done;
          context.(block_size - 1) <- sampled;
          aux (steps + 1)
        end
      end
    in
    aux 0
  in

  (* Generate very few names because different hardware backends diverge quickly. *)
  let names = Array.init 3 ~f:(fun _ -> gen_name ()) in
  Array.iter names ~f:print_endline
