open Base
open Ocannl
open Stdio
module IDX = Train.IDX
open Nn_blocks.DSL_modules
module CDSL = Train.CDSL
module Asgns = Ir.Assignments

module type Backend = Ir.Backend_intf.Backend

(* Makemore progression — Part 3 (Bengio MLP + BatchNorm on Names).

   Corresponds to Karpathy's "Building makemore Part 3: Activations & Gradients, BatchNorm". Extends
   Part 2 ([mlp_names.ml]) by inserting a [batch_norm1d] between the hidden linear and the [tanh]
   non-linearity. Same data pipeline; see [docs/makemore_tutorial.md]. The output contract differs
   from Part 2's in its last two lines: the sampled names are not portable here (see the comment on
   generation below), so they go to stderr and stdout summarizes them instead.

   Note: [batch_norm1d] inherits [batch_norm2d]'s FIXME — running statistics are not yet
   implemented, so inference falls back to the learned [gamma]/[beta] rather than population
   estimates. Acceptable for this tutorial example; do not rely on inference correctness for
   distribution-shifted inputs. *)

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

(** Fill a per-batch flat one-hot buffer for contexts (shape
    [batch_size * block_size * vocab_size]). *)
let fill_ctx_one_hot buf contexts ~offset =
  Array.fill buf ~pos:0 ~len:(Array.length buf) 0.;
  for i = 0 to batch_size - 1 do
    for t = 0 to block_size - 1 do
      let base = ((i * block_size) + t) * vocab_size in
      buf.(base + contexts.(((offset + i) * block_size) + t)) <- 1.
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
    let ga = Genarray.create Float32 c_layout [| batch_size; block_size; vocab_size |] in
    Bigarray.Genarray.fill ga 0.;
    let nd = Ir.Ndarray.as_array Ir.Ops.Single ga in
    Tensor.term ~init_data:(Reshape nd) ~grad_spec:If_needed ~label:[ label ]
      ~batch_dims:[ batch_size; block_size ] ~input_dims:[] ~output_dims:[ vocab_size ] ()
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

  (* === Model === embed: per-position character embedding. hidden: linear contraction (block_size,
     embed_dim) -> hid_dim, then batch_norm1d, then tanh. The BatchNorm is the Part-3 ingredient.
     logits: final linear projection to [vocab_size]. *)
  let bn1 = Nn_blocks.batch_norm1d ~label:[ "bn1" ] () in
  let%op embed input = { c; o = [ embed_dim ] } * input in
  (* Part 3 uses Kaiming init on the hidden weight w1: standard-normal samples scaled by sqrt(6 /
     fan_in) (the default scale_sq = 6). Note this is in the spirit of, but NOT equal to, Karpathy's
     kaiming_normal with tanh gain: (5/3)/sqrt(fan_in) ~ 1.67/sqrt(fan_in) vs our
     sqrt(6)/sqrt(fan_in) ~ 2.45/sqrt(fan_in) — sqrt(6) is the gain of the kaiming UNIFORM bound.
     Realized stds are measured in test/operations/test_default_init_std.ml. *)
  let%op mk_hidden () ~train_step x =
    tanh
      (bn1 ~train_step
         (embed x
         +* { w1 = kaiming normal1 () } "bs|->e; |se->h => b|->h" [ "s"; "e" ]
         + { b1; o = [ hid_dim ] }))
  in
  let hidden = mk_hidden () in
  let%op mk_logits () ~train_step x = ({ w2 } * hidden ~train_step x) + { b2 } in
  let logits = mk_logits () in

  let train_logits = logits ~train_step:(Some step_n) input_batch in
  (* Numerically stable log-softmax cross-entropy, matching transformer_names.ml. *)
  let%op max_l = train_logits @^^ "... | ... => ... | 0" in
  let%op shifted = train_logits - max_l in
  let%op lse = log (exp shifted ++ "... | ... => ... | 0") in
  let%op log_probs = shifted - lse in
  let%op nll = neg ((target_batch *. log_probs) ++ "... | ... => |->0") in
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

  let ctx, sgd_step = Train.to_routine ctx bindings (Asgns.sequence [ update; sgd ]) in
  let open Operation.At in
  let step_ref = IDX.find_exn sgd_step.Context.bindings step_n in
  Train.set_materialized batch_loss.value;

  let ctx_buf = Array.create ~len:(batch_size * block_size * vocab_size) 0. in
  let tgt_buf = Array.create ~len:(batch_size * vocab_size) 0. in

  (* === Training === *)
  (* Preserve early-trajectory guards, then judge convergence on a window: a single late epoch is a
     noisier statistic than the ten losses the run already computes and logs (gh-ocannl-854). *)
  let early_epoch_loss_limit epoch = if epoch = 0 then 5.0 else 3.0 in
  let tail_window_size = 10 in
  let logged_losses = ref [] in
  (* Two-sided, because every relocated claim here is an upper bound and an upper bound is
     one-sided: a dropped negation or a backend sign error yields a FINITE NEGATIVE cross-entropy
     that clears every threshold below, and only the digits this commit moved to stderr would have
     caught it (Codex round 1, P2). Cross-entropy is -log p >= 0 by construction; the tolerance is
     for accumulated rounding at exactly zero. Accumulated across the epochs and the three
     evaluation means, and claimed once after them. *)
  let valid_loss x = Float.(is_finite x && x >= -1e-3) in
  let all_valid = ref true in
  for epoch = 0 to epochs - 1 do
    let epoch_loss = ref 0. in
    for batch = 0 to n_batches - 1 do
      let offset = batch * batch_size in
      fill_ctx_one_hot ctx_buf train_ctx ~offset;
      fill_tgt_one_hot tgt_buf train_tgt ~offset;
      ignore (Context.set_values ctx input_batch.value ctx_buf : Context.t);
      ignore (Context.set_values ctx target_batch.value tgt_buf : Context.t);
      Train.run ctx sgd_step;
      epoch_loss := !epoch_loss +. (ctx, batch_loss).@[0];
      Int.incr step_ref
    done;
    let mean_loss = !epoch_loss /. Float.of_int n_batches in
    (* Exact digits to stderr, threshold claim on stdout (gh-ocannl-725): a trained mean arrives
       through a long floating-point reduction, so its low decimals depend on reduction order --
       backend, SIMD width, worker count -- and NO fixed print precision is portable. The property
       the number was there to show is the bound, and that is what the golden keeps. *)
    if not (valid_loss mean_loss) then all_valid := false;
    logged_losses := mean_loss :: !logged_losses;
    eprintf "Epoch %d, mean train loss=%.4f (not part of the golden)\n%!" epoch mean_loss;
    if epoch < epochs - tail_window_size then
      let limit = early_epoch_loss_limit epoch in
      Verdict.pf "Epoch %d, mean train loss below %g" epoch limit Float.(mean_loss < limit)
  done;
  let tail_mean = Training_golden.recent_mean_exn ~count:tail_window_size !logged_losses in
  let tail_limit = 2.8 in
  eprintf "Last %d logged epochs mean train loss=%.6f (not part of the golden)\n%!" tail_window_size
    tail_mean;
  Verdict.pf "Last %d logged epochs mean train loss below %g" tail_window_size tail_limit
    Float.(tail_mean < tail_limit);

  (* === Evaluation on train/dev/test === Build a separate forward-only subgraph with fresh
     input/target tensors so the trained batch_loss's forward code (already consumed by grad_update)
     isn't re-used. Weights/embeddings are shared because the model is invoked a second time under
     %cd. *)
  let eval_input =
    let open Bigarray in
    let ga = Genarray.create Float32 c_layout [| batch_size; block_size; vocab_size |] in
    Bigarray.Genarray.fill ga 0.;
    let nd = Ir.Ndarray.as_array Ir.Ops.Single ga in
    Tensor.term ~init_data:(Reshape nd) ~grad_spec:Prohibit_grad ~label:[ "eval_input" ]
      ~batch_dims:[ batch_size; block_size ] ~input_dims:[] ~output_dims:[ vocab_size ] ()
  in
  let eval_target =
    let open Bigarray in
    let ga = Genarray.create Float32 c_layout [| batch_size; vocab_size |] in
    Bigarray.Genarray.fill ga 0.;
    let nd = Ir.Ndarray.as_array Ir.Ops.Single ga in
    Tensor.term ~init_data:(Reshape nd) ~grad_spec:Prohibit_grad ~label:[ "eval_target" ]
      ~batch_dims:[ batch_size ] ~input_dims:[] ~output_dims:[ vocab_size ] ()
  in
  let%cd eval_logits = logits ~train_step:None eval_input in
  let%cd eval_max_l = eval_logits @^^ "... | ... => ... | 0" in
  let%cd eval_shifted = eval_logits - eval_max_l in
  let%cd eval_lse = log (exp eval_shifted ++ "... | ... => ... | 0") in
  let%cd eval_log_probs = eval_shifted - eval_lse in
  let%cd eval_nll = neg ((eval_target *. eval_log_probs) ++ "... | ... => |->0") in
  let%cd eval_loss = (eval_nll ++ "... => 0") /. !..batch_size in
  Train.set_materialized eval_loss.value;
  Train.set_materialized eval_input.value;
  Train.set_materialized eval_target.value;
  let%cd eval_comp =
    ~~("mlp_names eval";
       eval_loss.forward)
  in
  let ctx, eval_step = Train.to_routine sgd_step.Context.context IDX.empty eval_comp in
  let mean_loss_over (ctx_arr, tgt_arr, n) =
    let nb = n / batch_size in
    if nb = 0 then 0.0
    else begin
      let acc = ref 0. in
      for batch = 0 to nb - 1 do
        let offset = batch * batch_size in
        fill_ctx_one_hot ctx_buf ctx_arr ~offset;
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
  (* The 2026-08-29/30 sweep showed no backend/day spread to four decimals. These bounds retain
     materially more headroom than that observed spread while excluding an untrained model. *)
  let train_below = 2.8 in
  let dev_below = 2.8 in
  let test_below = 2.8 in
  (* Digits to stderr, bounds on stdout -- see the epoch loop above (gh-ocannl-725). *)
  eprintf "Final losses (not part of the golden): train=%.4f dev=%.4f test=%.4f\n%!" final_train
    final_dev final_test;
  Verdict.pf "Final train loss below %g" train_below Float.(final_train < train_below);
  Verdict.pf "Final dev   loss below %g" dev_below Float.(final_dev < dev_below);
  Verdict.pf "Final test  loss below %g" test_below Float.(final_test < test_below);
  List.iter [ final_train; final_dev; final_test ] ~f:(fun l ->
      if not (valid_loss l) then all_valid := false);
  Verdict.p "every epoch and evaluation mean is a finite, nonnegative cross-entropy" !all_valid;

  (* === Generation === Autoregressive sampling from a rolling [block_size] context. *)
  let infer_input =
    let open Bigarray in
    let ga = Genarray.create Float32 c_layout [| 1; block_size; vocab_size |] in
    Bigarray.Genarray.fill ga 0.;
    let nd = Ir.Ndarray.as_array Ir.Ops.Single ga in
    (* The leading batch axis is a single inference example that must broadcast against the training
       batch (which [batch_norm1d]'s statistics were reduced over). Under the total-basis design an
       explicit user size-1 axis is a non-stretching [default] atom; to opt into broadcasting we tag
       it with the reserved [bcast_if_1] basis (the advertisable affordance). *)
    Tensor.term ~init_data:(Reshape nd) ~grad_spec:Prohibit_grad ~label:[ "infer_input" ]
      ~batch_axes:[ (Row.bcast_if_1, 1); (Row.default_basis, block_size) ]
      ~input_dims:[] ~output_dims:[ vocab_size ] ()
  in
  let counter_n, infer_bindings = IDX.get_static_symbol IDX.empty in
  let%cd infer_logits = logits ~train_step:None infer_input in
  let%cd infer_comp =
    ~~("names infer";
       infer_logits.forward;
       { dice } =: uniform_at !@counter_n)
  in
  Train.set_materialized infer_logits.value;
  Train.set_materialized infer_input.value;
  let ctx, infer_step = Train.to_routine eval_step.Context.context infer_bindings infer_comp in
  let counter_ref = IDX.find_exn infer_step.Context.bindings counter_n in
  counter_ref := 0;

  let dot_idx = Dataprep.Names.char_index '.' in
  let set_ctx_one_hot context =
    let buf = Array.create ~len:(block_size * vocab_size) 0. in
    for t = 0 to block_size - 1 do
      buf.((t * vocab_size) + context.(t)) <- 1.
    done;
    ignore (Context.set_values ctx infer_input.value buf : Context.t)
  in

  (* One inference step: advance the [dice] counter, run the routine, and return the softmax
     distribution over the next character alongside the drawn [dice] value. *)
  let next_probs () =
    Int.incr counter_ref;
    Train.run ctx infer_step;
    let dice_value = (ctx, dice).@[0] in
    let logits_arr = Array.init vocab_size ~f:(fun v -> (ctx, infer_logits).@{[| 0; v |]}) in
    let max_logit = Array.fold logits_arr ~init:Float.neg_infinity ~f:Float.max in
    let exp_logits = Array.map logits_arr ~f:(fun l -> Float.exp (l -. max_logit)) in
    let sum_exp = Array.fold exp_logits ~init:0. ~f:( +. ) in
    (Array.map exp_logits ~f:(fun e -> e /. sum_exp), dice_value)
  in

  let max_len = 20 in
  let gen_name () =
    let context = Array.create ~len:block_size dot_idx in
    let buf = Buffer.create 16 in
    let rec aux steps =
      if steps >= max_len then Buffer.contents buf
      else begin
        set_ctx_one_hot context;
        let probs, dice_value = next_probs () in
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

  (* The sampled text is deliberately NOT part of the golden output. Two reasons, either one fatal
     to a byte-exact expectation:

     1. With a single-example inference batch the BatchNorm above collapses to [beta] regardless of
     input (the running-statistics FIXME in [nn_blocks.ml], spelled out in
     [docs/makemore_tutorial.md]), so the characters are essentially a readout of the [dice] stream
     against a near-constant distribution, not of the model reading its context.

     2. They sit on knife-edge boundaries of the sampling CDF. Under this seed, step 15 of the
     second name draws dice=0.294824 against an 'a'/'b' boundary at 0.294888 — a margin of 6.4e-5. A
     ~2e-4 wiggle in the trained weights, small enough to leave all three printed losses identical
     at 4 decimals, moves the boundary past the dice and flips the character; that is exactly what
     differs between backends, config profiles, and hardware. Re-pinning the text would just
     relocate the failure to the next machine.

     So the names go to stderr for the reader (the [slow] rule captures only stdout), and stdout
     keeps what is portable: that sampling stayed inside the alphabet, and the head of the
     distribution the sampler consumes. *)
  let names = Array.init 3 ~f:(fun _ -> gen_name ()) in
  Array.iter names ~f:(fun name -> eprintf "sampled name (not part of the golden): %s\n%!" name);
  Verdict.p_all
    (Printf.sprintf "Generated %d names, all chars in alphabet" (Array.length names))
    (Array.to_list names) ~f:(String.for_all ~f:Char.is_alpha);

  (* Head of the learned next-character distribution at the start context. The probabilities
     themselves drift by ~3e-4 between builds that disagree on the sampled text, and a fixed print
     precision only moves the boundary at which that drift shows: two decimals of 0.2749 and 0.2750
     differ (gh-ocannl-725). What is robust here is the RANKING and its margins — the gaps between
     the top three characters are ~0.08, some 200x the drift — so the digits go to stderr and stdout
     keeps the ordered letters plus the separation that makes the order meaningful. Both claims
     discriminate: a model that failed to learn the letter-frequency head would rank differently,
     and a collapsed (near-uniform) distribution would fail the margin. *)
  set_ctx_one_hot (Array.create ~len:block_size dot_idx);
  let start_probs, _dice = next_probs () in
  let ranked =
    Array.mapi start_probs ~f:(fun i p -> (p, i))
    |> Array.sorted_copy ~compare:(fun (p1, _) (p2, _) -> Float.descending p1 p2)
  in
  let top3 = Array.sub ranked ~pos:0 ~len:3 in
  let letter_of i = List.nth_exn Dataprep.Names.letters_with_dot i in
  eprintf "Start-context top-3 next chars (not part of the golden): %s\n%!"
    (top3
    |> Array.map ~f:(fun (p, i) -> Printf.sprintf "%c=%.2f" (letter_of i) p)
    |> String.concat_array ~sep:" ");
  let letters =
    top3
    |> Array.map ~f:(fun (_, i) -> String.of_char (letter_of i))
    |> String.concat_array ~sep:" "
  in
  Verdict.p "Start-context top-3 next chars are a, e, i in that order"
    (String.equal letters "a e i");
  let min_gap = 0.05 in
  Verdict.p_alli
    (Printf.sprintf "Start-context top-3 probabilities are separated by more than %g" min_gap)
    (Array.to_list top3) ~f:(fun k (p, _) ->
      k = 0
      ||
      let prev = fst top3.(k - 1) in
      Float.(prev -. p > min_gap));
  (* Ranking and separation are both SHAPE claims: a head of 0.70 / 0.20 / 0.08 satisfies each of
     them while being severely distorted, and the two printed decimals used to rule that out (Codex
     round 1, P2). So keep a magnitude check too, as coarse per-rank bands -- each is roughly a
     factor of two wide around the observed 0.28 / 0.17 / 0.10, some hundreds of times the ~3e-4
     cross-build drift, so it is portable while still rejecting a collapsed or a spiked head. *)
  let bands = [| (0.15, 0.45); (0.08, 0.30); (0.04, 0.20) |] in
  Verdict.p_alli
    "Start-context top-3 probabilities lie in their coarse bands (0.15-0.45, 0.08-0.30, 0.04-0.20)"
    (Array.to_list top3) ~f:(fun k (p, _) ->
      let lo, hi = bands.(k) in
      Float.(p >= lo && p < hi))
