(* OCANNL GPT-2-style runner: pre-LN decoder blocks built from the idiomatic nn_blocks pieces
   (multi_head_attention, layer_norm — fixture weights injected by name) and a tanh-gelu FFN from
   fixture-backed weight tensors. Token embedding is the logical one-hot gather (gh-343); the
   lm_head is tied to wte via an einsum that reads it transposed. Layouts documented in
   gen_fixtures.py build_gpt.

   The step shape follows the fixture's [mode] metadata, like the Python runners:

   - [mode: infer] (gpt2_mini) — forward-only; the parity metric is softmax-CE of the logits against
   fixture target ids, recorded per batch with no updates. - [mode: train] (gpt2_mini_train,
   gh-ocannl-551) — every weight is a parameter, the step is backprop plus plain SGD, and the whole
   gh-ocannl-492 task-5 gate-cost family (BENCH_PRECISION with BENCH_STATIC_SCALE /
   BENCH_GATE_INTERVAL) applies, since the workload now has an optimizer and hence a loss scale to
   gate. The flags themselves live in Bench_harness, shared with bench_mlp. *)

open Base
open Ocannl
module IDX = Train.IDX
open Nn_blocks.DSL_modules
module St = Safetensors
module H = Bench_harness

let cross_entropy_loss = Nn_blocks.cross_entropy_loss
let gelu = Nn_blocks.gelu

let ids_of_gen g =
  let dims = Bigarray.Genarray.dims g in
  Array.init dims.(0) ~f:(fun s ->
      Array.init dims.(1) ~f:(fun i -> Int.of_float (Bigarray.Genarray.get g [| s; i |])))

(* Token ids as a uint32 tensor with explicit [n_batches; batch_size; seq] batch dims, so that [@|
   batch_n] indexes the leading batch axis. *)
let ids_tensor ~label ints ~n_batches ~batch_size ~seq =
  let open Bigarray in
  let g = Genarray.create Int32 c_layout [| n_batches; batch_size; seq |] in
  Array.iteri ints ~f:(fun idx row ->
      let b = idx / batch_size and s = idx % batch_size in
      Array.iteri row ~f:(fun t id -> Genarray.set g [| b; s; t |] (Int32.of_int_trunc id)));
  TDSL.wrap ~l:label ~b:[ n_batches; batch_size; seq ] ~o:[]
    (Ir.Ndarray.as_array Ir.Ops.Uint32 g)
    ()

(* Under a reduced precision the training leg keeps f32 master weights and reads reduced-precision
   cast twins ([Mixed_prec.with_master_weights]) — except the layer-norm gains and biases, which
   stay f32 like the softmax-CE head does (the AMP default, and what the inference leg pins through
   the storage policy). [Mixed_prec] has no per-parameter filter, so install the hook here. *)
let with_master_weights_except_ln ~prec f =
  let saved = !Tensor.param_postprocess in
  let is_ln (p : Tensor.t) =
    List.exists p.Tensor.value.Ir.Tnode.label ~f:(fun l ->
        String.equal l "gamma" || String.equal l "beta")
  in
  (Tensor.param_postprocess :=
     fun p ->
       let p = saved p in
       if is_ln p then p else Mixed_prec.cast_param ~prec p);
  Exn.protect ~f ~finally:(fun () -> Tensor.param_postprocess := saved)

let () =
  let fixture = Stdlib.Sys.getenv "BENCH_FIXTURE" in
  let tune = H.env_flag "BENCH_TUNE" in
  H.install_timing_trace ();
  let materialize = H.env_flag "BENCH_MATERIALIZE" in
  let debug = H.env_flag "BENCH_DEBUG" in
  let st = St.read fixture in
  (* BENCH_PRECISION=bf16|f16 (gh-ocannl-492 task 4). Reduced precision enters differently in the
     two modes, and the difference is the workload's, not a limitation:

     - Forward-only: LOAD-TIME CONVERSION, not cast twins — inference has no optimizer, so there is
     no master copy for a twin to preserve, and keeping f32 storage would pay exactly the weight
     bandwidth the leg measures away (torch's [model.half ()]). Data-backed weights (wte/wpe/ffn)
     convert at wrap; the attention-projection params take the reduced precision through the storage
     policy and [H.inject]'s [set_values] converts the f32 fixture values at load. No loss scaling:
     there are no gradients. - Training: the mixed-precision recipe — f32 masters with
     reduced-precision cast twins, the storage policy over the decoder body's activations and
     gradients, and for f16 the dynamic loss scaling whose gate the task-5 legs measure.

     Layer-norm gains/biases and the softmax-CE head stay f32 in both. *)
  let training = H.is_training st in
  let leg = H.precision_leg ~runner:"bench_gpt" ~training ~st () in
  let mp_prec = leg.H.prec in
  let batch_size = H.meta_int st "batch_size" in
  let n_layer = H.meta_int st "n_layer" in
  let n_head = H.meta_int st "n_head" in
  let d_model = H.meta_int st "d_model" in
  let vocab = H.meta_int st "vocab" in
  let seq = H.meta_int st "seq_len" in
  let d_head = d_model / n_head in
  let ids_all = ids_of_gen (St.to_float32 st "ids") in
  let tgt_all = ids_of_gen (St.to_float32 st "tgt") in
  let total = Array.length ids_all in
  let n_batches = total / batch_size in
  let ids_t = ids_tensor ~label:"ids" ids_all ~n_batches:(total / batch_size) ~batch_size ~seq in
  let tgt_t = ids_tensor ~label:"tgt" tgt_all ~n_batches:(total / batch_size) ~batch_size ~seq in
  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let%op ids_b = ids_t @| batch_n in
  let%op tgt_b = tgt_t @| batch_n in
  (* Training makes every weight a parameter (the optimizer's target, and what the Python runners
     put in their optimizer); inference keeps them data-backed constants, re-precisioned at load. *)
  (* The training parameters [wrap_param] builds already hold their fixture tensor, as a host init
     the step's link uploads; the untuned path does not inject them again (see [H.inject]'s
     [skip] below). *)
  let fixture_backed = Hash_set.create (module Int) in
  let wrap_param ~l ?i ~o nd =
    let t = TDSL.wrap_param ~l ?i ~o nd () in
    Hash_set.add fixture_backed t.Tensor.value.Ir.Tnode.id;
    t
  in
  let wrap name ~i ~o =
    let nd = St.to_ndarray st name in
    if training then wrap_param ~l:name ~i ~o nd
    else TDSL.wrap ~l:name ?prec:mp_prec ~b:[] ~i ~o nd ()
  in
  let mask =
    NTDSL.init ~l:"mask" ~prec:Ir.Ops.single ~b:[ seq ] ~i:[ seq ] ~o:[]
      ~f:(function [| s; t |] -> if s >= t then 1. else 0. | _ -> assert false)
      ()
  in
  let onehot_x = Nn_blocks.one_hot_of_ids ~num_classes:vocab ids_b in
  (* The whole model construction is a thunk so the master-weights hook (which fires at parameter
     creation, i.e. at the blocks' [()] application) covers the block params as well as ours. *)
  let build () =
    let wte = wrap "wte" ~i:[ vocab ] ~o:[ d_model ] in
    (* A parameter has no batch axes (Tensor.param pins them empty), so the trained positional table
       is [seq] x [d_model] output axes and the einsum-add places its seq axis onto the sequence
       batch axis; inference keeps the plain broadcast add over a [seq]-batched table. *)
    let wpe_param =
      if training then Some (wrap_param ~l:"wpe" ~o:[ seq; d_model ] (St.to_ndarray st "wpe"))
      else None
    in
    let%op embedded =
      if training then
        (wte * onehot_x) +++ "... s | d; | s d => ... s | d" (Option.value_exn wpe_param)
      else
        (wte * onehot_x)
        + TDSL.wrap ~l:"wpe" ?prec:mp_prec ~b:[ seq ] ~i:[] ~o:[ d_model ] (St.to_ndarray st "wpe")
            ()
    in
    let layers =
      List.init n_layer ~f:(fun i ->
          let name fmt = Printf.sprintf fmt i in
          let lbl = Printf.sprintf "l%d" i in
          let mha =
            Nn_blocks.multi_head_attention ~label:[ lbl ] ~num_heads:n_head ~d_k:d_head ~d_v:d_head
              ()
          in
          let ln1 = Nn_blocks.layer_norm ~label:[ "ln1"; lbl ] () in
          let ln2 = Nn_blocks.layer_norm ~label:[ "ln2"; lbl ] () in
          let fw1 = wrap (name "l%d_ffn_w1") ~i:[ d_model ] ~o:[ H.meta_int st "d_ff" ] in
          let fb1 = wrap (name "l%d_ffn_b1") ~i:[] ~o:[ H.meta_int st "d_ff" ] in
          let fw2 = wrap (name "l%d_ffn_w2") ~i:[ H.meta_int st "d_ff" ] ~o:[ d_model ] in
          let fb2 = wrap (name "l%d_ffn_b2") ~i:[] ~o:[ d_model ] in
          fun x ->
            let%op x1 = x + mha ~train_step:None ~mask (ln1 x) in
            let%op x2 = x1 + ((fw2 * gelu ((fw1 * ln2 x1) + fb1)) + fb2) in
            x2)
    in
    let lnf = Nn_blocks.layer_norm ~label:[ "lnf" ] () in
    let hfinal = lnf (List.fold layers ~init:embedded ~f:(fun x layer -> layer x)) in
    (* Tied lm_head: read wte transposed (contract d, produce v). *)
    let%op logits = wte +* "|v -> d; ... | d => ... | v" hfinal in
    logits
  in
  let logits =
    match mp_prec with
    | Some prec when training -> with_master_weights_except_ln ~prec build
    | _ -> build ()
  in
  let targets = Nn_blocks.one_hot_of_ids ~num_classes:vocab tgt_b in
  let n_positions = batch_size * seq in
  let%op batch_loss =
    cross_entropy_loss ~spec:"...|v" ~normalize_by:!..n_positions () ~logits ~targets
  in
  (* Storage policy for the reduced-precision leg: the decoder body (the logits subtree) computes
     in the reduced precision — while layer-norm gains/biases and the softmax-CE head are PINNED at
     f32 via [except] (merely not assigning them is not enough: precision inference would pull them
     to the reduced precision from their neighbors). Inference re-precisions the parameters here
     too ([param_prec]; injection converts the fixture values at load); training leaves them alone
     ([param_prec = None]) because masters and cast twins already carry Specified precisions, and
     assigns the gradients instead. *)
  (* The attention softmax computes in the reduced precision along with the rest of the body —
     including at f16, since gh-ocannl-548 made the causal mask's fill -inf (representable, unlike
     the -1e9 that the fp16 constant guard rejected) and gh-ocannl-547 took reduction identities
     out of that guard's scope. Both landed while this workload was being built; before them the
     f16 gpt cells could not compile at all. *)
  Option.iter mp_prec ~f:(fun prec ->
      let body = Hash_set.create (module Int) in
      let rec walk t =
        if not (Hash_set.mem body t.Tensor.value.Ir.Tnode.id) then (
          Hash_set.add body t.Tensor.value.Ir.Tnode.id;
          Option.iter t.Tensor.diff ~f:(fun d -> Hash_set.add body d.Tensor.grad.Ir.Tnode.id);
          List.iter t.Tensor.children ~f:(fun c -> walk c.Tensor.subtensor))
      in
      walk logits;
      let is_ln tn =
        List.exists tn.Ir.Tnode.label ~f:(fun l -> String.equal l "gamma" || String.equal l "beta")
      in
      let except tn = (not (Hash_set.mem body tn.Ir.Tnode.id)) || is_ln tn in
      Precision_policy.apply ~except
        {
          Precision_policy.param_prec = (if training then None else Some prec);
          activation_prec = Some prec;
          grad_prec = (if training then Some prec else None);
        }
        batch_loss);
  if materialize then Train.every_non_literal_materialized batch_loss;
  (* Inference is the loss's forward code; training is the leg's step shape (backprop plus plain
     SGD, or one of the loss-scaling shapes). Either consumes the forward code, so only one is
     built. *)
  let step_shape =
    if training then
      let learning_rate = TDSL.O.( !. ) (Float.of_string (H.get_meta st "lr")) in
      `Train (H.train_step_parts ~leg ~learning_rate batch_loss)
    else `Forward (Train.forward batch_loss)
  in
  let ctx = Context.auto () in
  let backend = Context.backend_name ctx in
  let ctx = Train.init_params ctx bindings batch_loss in
  if debug then (
    H.dump_params batch_loss;
    Stdlib.exit 0);
  let mapping =
    ("lnf_g", [ "gamma"; "lnf" ])
    :: ("lnf_b", [ "beta"; "lnf" ])
       (* Training turns the data-backed weights into parameters, so they join the injection mapping
          — [H.inject] requires every parameter to name its fixture entry, and the tuned variant
          re-injects after the search has overwritten them. *)
    :: (if training then [ ("wte", [ "wte" ]); ("wpe", [ "wpe" ]) ] else [])
    @ List.concat_map (List.range 0 n_layer) ~f:(fun i ->
        let l = Printf.sprintf "l%d" i in
        [
          (Printf.sprintf "l%d_wq" i, [ l; "w"; "q" ]);
          (Printf.sprintf "l%d_wk" i, [ l; "w"; "k" ]);
          (Printf.sprintf "l%d_wv" i, [ l; "w"; "v" ]);
          (Printf.sprintf "l%d_wo" i, [ l; "w"; "o" ]);
          (Printf.sprintf "l%d_ln1_g" i, [ l; "gamma"; "ln1" ]);
          (Printf.sprintf "l%d_ln1_b" i, [ l; "beta"; "ln1" ]);
          (Printf.sprintf "l%d_ln2_g" i, [ l; "gamma"; "ln2" ]);
          (Printf.sprintf "l%d_ln2_b" i, [ l; "beta"; "ln2" ]);
        ]
        @
        if training then
          [
            (Printf.sprintf "l%d_ffn_w1" i, [ l; "ffn"; "w1" ]);
            (Printf.sprintf "l%d_ffn_b1" i, [ l; "ffn"; "b1" ]);
            (Printf.sprintf "l%d_ffn_w2" i, [ l; "ffn"; "w2" ]);
            (Printf.sprintf "l%d_ffn_b2" i, [ l; "ffn"; "b2" ]);
          ]
        else [])
  in
  (* Injecting a fixture-backed parameter here would re-upload the tensor it already holds, and
     through [Context.set_values] on a node no routine has linked yet, which allocates it a pool of
     its own: 18 of them in the training step, over Metal's 16-pool binding budget, so the step
     could not link on Metal at all. Left alone, they are uploaded from their host init into the
     step's own pool at its link. The tuned variant's re-injection below covers every parameter. *)
  let ctx =
    H.inject ~skip:(fun tn -> Hash_set.mem fixture_backed tn.Ir.Tnode.id) ctx st batch_loss mapping
  in
  let t0 = Unix.gettimeofday () in
  (* Placement A/B: tune the default (virtual + promotion) graph and the materialize-all graph,
     keep the measured winner. Tuning runs on a scratch lineage, so the repeated candidate
     executions never touch the benchmark's own weights. *)
  (* Both placement arms' crowned candidates go into the emitted result (gh-ocannl-546). *)
  let arms = H.tune_arms () in
  let tuned ctx comp =
    let scratch = Train.init_params (Context.auto ()) bindings batch_loss in
    (* [flip_report] only counts the gh-555 refinement searches toward the result line's [searched]
       (gh-ocannl-644): they run whenever [tune_inline_flips] is configured, whether or not a
       callback is wired, and a flip search loads this process like an arm search. *)
    Train.tune_placements ~report:(H.collect_arm arms) ~flip_report:(H.collect_search arms)
      ~on_ship:(H.collect_ship arms) ~rounds:0 ~timing_ctx:scratch ctx batch_loss comp bindings
  in
  let ctx, routines =
    match step_shape with
    | `Train parts -> H.compile_train_step ~tune ~tuned ctx bindings parts
    | `Forward fwd ->
        let ctx, routine =
          if tune then tuned ctx fwd
          else if Lazy.force Autotune.model_default_enabled then
            (* gh-ocannl-491: the model-picked untuned default (config
               [model_default_schedule=true]). *)
            Autotune.model_default ctx fwd bindings
          else Context.compile ctx fwd bindings
        in
        (ctx, H.Plain routine)
  in
  (* What the timed artifact emitted, off the routines themselves (gh-ocannl-626): a flip refinement
     or a timing_ctx replay fallback ships something no arm report describes. *)
  H.collect_shipped arms routines;
  let compile_s = Unix.gettimeofday () -. t0 in
  H.trace_search_done ~compile_s;
  let ctx = if tune then H.inject ctx st batch_loss mapping else ctx in
  (* The scaled training legs thread the context (Loss_scaler.update overwrites the scale tensors),
     hence the reference. *)
  let ctx_ref = ref ctx in
  let batch_ref = IDX.find_exn (H.train_step_bindings routines) batch_n in
  let step_count = ref 0 in
  let run_step () =
    batch_ref := !step_count % n_batches;
    H.run_train_step routines ctx_ref ~step:!step_count;
    Int.incr step_count
  in
  let open Operation.At in
  ignore
    (H.measure_and_emit ~protocol:(H.protocol_of_st st) ~backend
       ~variant:
         (* Mirror bench_mlp: the scheduling variant alone. orchestrate renders precision as its own
            report column and composes the two axes itself (gh-ocannl-539), so a reduced-precision
            cell is distinguished by the precision field rather than by overloading this one. *)
         (if tune then "tuned" else if materialize then "materialized" else "default")
       ~precision:leg.H.label ~compile_s ~tokens_per_step:(batch_size * seq) ~tune:arms
       ~dominant_kernel:(fun () ->
         H.dominant_kernel ~ctx:!ctx_ref ~bindings (H.step_routines routines))
       ~run_step
       ~read_loss:(fun () -> (!ctx_ref, batch_loss).@[0])
       ~sync:(fun () -> Context.sync !ctx_ref)
       ()
      : string)
