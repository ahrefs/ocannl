open Base
open Ocannl
open Stdio
module IDX = Train.IDX
open Nn_blocks.DSL_modules
module Asgns = Ir.Assignments

(* === Configuration === *)

let ctx_len = 16
let eff_seq_len = ctx_len - 1
let d_model = 16
let num_heads = 2
let d_k = 8
let d_v = 8
let d_ff = 32
let vocab_size = Dataprep.Names.dict_size

(* Batch 128 keeps GPU wall time bounded: the fissioned sgd step has a near-constant per-step
   dispatch cost, so fewer, larger steps at the same samples-per-epoch are strictly Metal-friendlier
   (10k steps at batch 32 -> 2.5k steps at batch 128). *)
let batch_size = 128
let epochs = 10
let pad_char = ' '
let pad_idx = Dataprep.Names.char_index pad_char
let bos_idx = Dataprep.Names.char_index '.'

(* === Data preparation === *)

(** Convert a name to a fixed-length integer sequence. "emma" -> [0; 5; 13; 13; 1; 0; 1; 1; ...]
    where 0='.' and 1=' ' (padding). Returns (input_indices, target_indices) for teacher forcing. *)
let name_to_sequences name =
  let chars = '.' :: (String.to_list name @ [ '.' ]) in
  let len = List.length chars in
  let padded =
    if len >= ctx_len then List.take chars ctx_len
    else chars @ List.init (ctx_len - len) ~f:(fun _ -> pad_char)
  in
  let indices = List.map padded ~f:Dataprep.Names.char_index in
  let input_indices = List.take indices eff_seq_len in
  let target_indices = List.tl_exn (List.take indices ctx_len) in
  (Array.of_list input_indices, Array.of_list target_indices)

let prepare_dataset () =
  let names = Dataprep.Names.read_names () in
  let num_names = List.length names in
  printf "Names loaded: %d\n%!" num_names;
  let pairs = List.map names ~f:name_to_sequences in
  let inputs = Array.of_list (List.map pairs ~f:fst) in
  let targets = Array.of_list (List.map pairs ~f:snd) in
  let num_examples = Array.length inputs in
  (* Round down to multiple of batch_size *)
  let num_examples = num_examples - (num_examples % batch_size) in
  printf "Training examples: %d\n%!" num_examples;
  (inputs, targets, num_examples)

(* The whole dataset as a one-hot device tensor (num_examples x eff_seq_len x vocab, ~54 MB in
   single precision): batches are selected on device via [@| batch_n], so the training loop does no
   per-step host transfers — a [Context.set_values] awaits the whole device, serializing the stream
   (see [Train.grad_update]'s [?accum_loss] doc). *)
let full_one_hot ~label (seqs : int array array) ~num_examples ~n_batches =
  let open Bigarray in
  let ga = Genarray.create Float32 c_layout [| num_examples; eff_seq_len; vocab_size |] in
  Genarray.fill ga 0.;
  for i = 0 to num_examples - 1 do
    for t = 0 to eff_seq_len - 1 do
      Genarray.set ga [| i; t; seqs.(i).(t) |] 1.
    done
  done;
  let nd = Ir.Ndarray.as_array Ir.Ops.Single ga in
  (* The leading batch axis is declared pre-split into [n_batches; batch_size] (same row-major
     layout), so [@| batch_n] can slice off the batch-count axis. *)
  Tensor.term ~init_data:(Reshape nd) ~grad_spec:If_needed ~label:[ label ]
    ~batch_dims:[ n_batches; batch_size; eff_seq_len ]
    ~input_dims:[] ~output_dims:[ vocab_size ] ()

(* === Main === *)

let () =
  Utils.settings.fixed_state_for_init <- Some 3;
  Tensor.unsafe_reinitialize ();

  let train_inputs, train_targets, num_examples = prepare_dataset () in
  let n_batches = num_examples / batch_size in

  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let step_n, bindings = IDX.get_static_symbol bindings in
  let total_tokens = batch_size * eff_seq_len in

  (* === Data tensors: the full dataset on device, batch selected by static index === *)
  let full_inputs = full_one_hot ~label:"train_inputs_oh" train_inputs ~num_examples ~n_batches in
  let full_targets =
    full_one_hot ~label:"train_targets_oh" train_targets ~num_examples ~n_batches
  in
  let%op input_batch = full_inputs @| batch_n in
  let%op target_batch = full_targets @| batch_n in

  (* === Causal mask === *)
  let mask =
    NTDSL.init ~l:"mask" ~prec:Ir.Ops.single ~b:[ eff_seq_len ] ~i:[ eff_seq_len ] ~o:[]
      ~f:(function
        | [| s; t |] -> if s >= t then 1. else 0. | _ -> failwith "unexpected mask indices")
      ()
  in

  (* === Model === Decoder-only transformer: masked self-attention + FFN with residual connections.
     Layer norm is omitted for this small model to keep generated code compact and avoid the
     gradient signal issue noted in fsm_transformer.ml. Uses multi_head_attention with ~mask for
     causal masking. *)
  let open Nn_blocks in
  let mha = multi_head_attention ~label:[ "mha" ] ~num_heads ~d_k ~d_v () in
  let ffn = Nn_blocks.mlp ~label:[ "ffn" ] ~hid_dims:[ d_ff ] () in
  let%op build_model () =
   fun ~train_step ~mask input ->
    let embedded = ({ tok_embed; o = [ d_model ] } * input) + { pos_encoding } in
    let x1 = embedded + mha ~train_step ~mask embedded in
    let x2 = x1 + ffn x1 in
    { w_out } * x2
  in
  let model = build_model () in

  (* === Training computation === *)
  let train_logits = model ~train_step:(Some step_n) ~mask input_batch in
  (* Numerically stable log-softmax cross-entropy: subtract max before exp to avoid overflow. *)
  let%op max_logits = train_logits @^^ "...|... => ...|0" in
  let%op shifted = train_logits - max_logits in
  let%op log_sum_exp = log (exp shifted ++ "...|... => ...|0") in
  let%op log_probs = shifted - log_sum_exp in
  let%op nll = neg ((target_batch *. log_probs) ++ "...|... => ...|0") in
  let%op batch_loss = (nll ++ "...|... => |->0") /. !..total_tokens in

  Train.every_non_literal_materialized batch_loss;
  (* Accumulate the loss on device: the training loop then syncs on the host once per epoch. *)
  let loss_accum = Train.loss_accumulator () in
  let update = Train.grad_update ~accum_loss:loss_accum batch_loss in
  let steps = epochs * n_batches in
  (* Linear lr scaling for the larger batch (0.01 at batch 32 -> 0.04 at batch 128): the same
     per-sample progress with 4x fewer updates; epoch-loss thresholds pass with margin. *)
  let%op learning_rate = 0.04 *. ((1.5 *. !..steps) - !@step_n) /. !..steps in
  let sgd = Train.sgd_update ~learning_rate batch_loss in

  (* === Inference computation (forward-only, shares trained weights) === *)
  let infer_input =
    let open Bigarray in
    let ga = Genarray.create Float32 c_layout [| 1; eff_seq_len; vocab_size |] in
    Bigarray.Genarray.fill ga 0.;
    let nd = Ir.Ndarray.as_array Ir.Ops.Single ga in
    Tensor.term ~init_data:(Reshape nd) ~grad_spec:Prohibit_grad ~label:[ "infer_input" ]
      ~batch_dims:[ 1; eff_seq_len ] ~input_dims:[] ~output_dims:[ vocab_size ] ()
  in
  let counter_n, infer_bindings = IDX.get_static_symbol IDX.empty in
  let%cd infer_logits = model ~train_step:None ~mask infer_input in
  let%cd infer_comp =
    ~~("names infer";
       infer_logits.forward;
       { dice } =: uniform_at !@counter_n)
  in

  (* === Compile === *)
  let ctx = Context.auto () in
  let ctx = Train.init_params ctx bindings batch_loss in
  Train.set_materialized input_batch.value;
  Train.set_materialized target_batch.value;
  Train.set_materialized infer_logits.value;
  Train.set_materialized infer_input.value;

  (* Compile training routine *)
  let train_comp = Asgns.sequence [ update; sgd ] in
  Set.iter
    (snd @@ Asgns.collect_nodes_guess_output train_comp.Asgns.asgns)
    ~f:Train.set_materialized;
  (* Tune the step schedule empirically. [rounds:0] keeps only the preset seed candidates, which all
     preserve reduction order and, since gh-ocannl-1682 on gcc too, per-step rounding (the packed
     seeds' explicit FMA against the serial loop's contracted one; [schedule_pad] pins it) — the
     trained values are schedule-invariant, so this file's expected output stays deterministic no
     matter which seed wins. [timing_ctx] gives the tuner a scratch lineage (with its own freshly
     initialized parameter buffers) to time candidates against, so the timing runs cannot perturb
     the real training state (a step timed on all-zero data inputs poisons parameters with inf/NaN
     through log 0). *)
  (* The crown and the finals outcome go to stderr ([Tune_crown]): a golden diff on another box is
     traced to the seed that won without rerunning under [autotune_log]. *)
  let scratch = Train.init_params (Context.auto ()) bindings batch_loss in
  let ctx, sgd_step =
    Autotune.tune ~rounds:0 ~timing_ctx:scratch
      ~report:(Tune_crown.eprint ~site:"transformer_names")
      ctx train_comp bindings
  in

  (* Compile inference routine *)
  Set.iter
    (snd @@ Asgns.collect_nodes_guess_output infer_comp.Asgns.asgns)
    ~f:Train.set_materialized;
  let infer_comp =
    { infer_comp with Asgns.embedded_nodes = Set.add infer_comp.Asgns.embedded_nodes mask.value }
  in
  let ctx, infer_routine = Context.compile ctx infer_comp infer_bindings in

  let open Operation.At in
  let step_ref = IDX.find_exn sgd_step.Context.bindings step_n in
  step_ref := 0;
  let batch_ref = IDX.find_exn sgd_step.Context.bindings batch_n in
  let counter_ref = IDX.find_exn infer_routine.Context.bindings counter_n in
  counter_ref := 0;
  Train.set_materialized batch_loss.value;

  (* === Training loop === Random baseline: ln(28) ≈ 3.33 per token, epoch sum ≈ 3.33 * n_batches.
     We keep first/middle trajectory checks and judge convergence on the last-ten window. *)
  let epoch_loss_limit_first = 2.0 *. Float.of_int n_batches in
  let epoch_loss_limit_mid = 1.4 *. Float.of_int n_batches in
  let tail_window_size = 10 in
  (* Pre-window sweep logs retained only three threshold verdicts. The 1.5-per-batch window bound
     sits above the observed cc mean (1.258 per batch) without tightening around that one draw, and
     the stderr statistic below gives the daily backend sweep the data for future audits. *)
  let tail_loss_limit = 1.5 *. Float.of_int n_batches in
  let logged_losses = ref [] in
  let all_valid = ref true in
  for epoch = 0 to epochs - 1 do
    for batch = 0 to n_batches - 1 do
      batch_ref := batch;
      let ctx' = Context.run ctx sgd_step in
      ignore (ctx' : Context.t);
      Int.incr step_ref
    done;
    (* The only device sync of the epoch: read the accumulated loss sum, then reset it. *)
    let epoch_loss = ref (Context.get_values ctx loss_accum.Tensor.value).(0) in
    ignore (Context.set_values ctx loss_accum.Tensor.value [| 0. |] : Context.t);
    logged_losses := !epoch_loss :: !logged_losses;
    if not Float.(is_finite !epoch_loss && !epoch_loss >= -1e-3) then all_valid := false;
    eprintf "Epoch %d, loss=%.6f (not part of the golden)\n%!" epoch !epoch_loss;
    if epoch = 0 || epoch = epochs / 2 then
      let limit = if epoch = 0 then epoch_loss_limit_first else epoch_loss_limit_mid in
      Verdict.pf "Epoch %d, loss below threshold" epoch Float.(!epoch_loss < limit)
  done;
  let tail_mean = Training_golden.recent_mean_exn ~count:tail_window_size !logged_losses in
  eprintf "Last %d logged epochs mean loss=%.6f (not part of the golden)\n%!" tail_window_size
    tail_mean;
  Verdict.p "every epoch loss is a finite, nonnegative cross-entropy" !all_valid;
  Verdict.pf "Last %d logged epochs mean loss below %g" tail_window_size tail_loss_limit
    Float.(tail_mean < tail_loss_limit);

  (* === Autoregressive generation === Generate names token-by-token, sampling from the model's
     output distribution. Uses the CDF-based sampling pattern from bigram_mlp.ml. *)
  let set_one_hot_seq context =
    let flat = Array.create ~len:(1 * eff_seq_len * vocab_size) 0. in
    for t = 0 to eff_seq_len - 1 do
      let base = t * vocab_size in
      flat.(base + context.(t)) <- 1.
    done;
    ignore (Context.set_values ctx infer_input.value flat : Context.t)
  in

  let gen_name () =
    let context = Array.create ~len:eff_seq_len pad_idx in
    context.(0) <- bos_idx;
    let rec aux pos =
      if pos >= eff_seq_len then (
        (* Max length reached — extract what we have *)
        let name = Buffer.create 16 in
        for i = 1 to eff_seq_len - 1 do
          let c = List.nth_exn Dataprep.Names.letters_with_dot context.(i) in
          if not (Char.equal c '.' || Char.equal c ' ') then Buffer.add_char name c
        done;
        Buffer.contents name)
      else begin
        set_one_hot_seq context;
        Int.incr counter_ref;
        let _ctx = Context.run ctx infer_routine in
        let dice_value = (ctx, dice).@[0] in

        (* Compute softmax probabilities at position (pos-1) in the output (the model predicts token
           at position pos given input up to pos-1). *)
        let logits =
          Array.init vocab_size ~f:(fun v -> (ctx, infer_logits).@{[| 0; pos - 1; v |]})
        in
        let max_logit = Array.fold logits ~init:Float.neg_infinity ~f:Float.max in
        let exp_logits = Array.map logits ~f:(fun l -> Float.exp (l -. max_logit)) in
        let sum_exp = Array.fold exp_logits ~init:0. ~f:( +. ) in
        let probs = Array.map exp_logits ~f:(fun e -> e /. sum_exp) in

        (* CDF-based sampling *)
        let max_i = vocab_size - 1 in
        let rec sample i acc =
          if i >= max_i then i
          else
            let new_acc = acc +. probs.(i) in
            if Float.(new_acc > dice_value) then i else sample (i + 1) new_acc
        in
        let sampled_idx = sample 0 0. in
        let sampled_char = List.nth_exn Dataprep.Names.letters_with_dot sampled_idx in

        if Char.equal sampled_char '.' && pos > 1 then begin
          (* EOS — extract name from positions 1..pos-1 *)
          let name = Buffer.create 16 in
          for i = 1 to pos - 1 do
            let c = List.nth_exn Dataprep.Names.letters_with_dot context.(i) in
            if not (Char.equal c '.' || Char.equal c ' ') then Buffer.add_char name c
          done;
          Buffer.contents name
        end
        else begin
          context.(pos) <- sampled_idx;
          aux (pos + 1)
        end
      end
    in
    aux 1
  in

  (* Generate very few names because different hardware backends diverge quickly. *)
  let names = Array.init 3 ~f:(fun _ -> gen_name ()) in
  Array.iter names ~f:print_endline
