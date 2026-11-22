(* Training-step benchmark for recurring host-to-device batch feeds (gh-ocannl-1854).

   A wide-input 2-layer MLP classifier (BATCH x 3072 -> 64 -> 8, softmax cross-entropy, plain SGD)
   compiled through the untuned default pipeline ([Train.to_routine]), fed every step via
   [Context.from_host] from K=4 pre-built reusable host [Ndarray.t] batch arrays (~24 MiB each at
   the default batch size 2048). Warmup runs two full epochs (8 steps) so every batch array has been
   seen more than once before timing begins.

   Step 0's loss is checked against an independent host forward computation over the exact same
   initial weights and separable batch-0 input (within fp32 tolerance), so every reported timing
   carries a real parity check ([Bench_out.Passed]), catching silent transfer corruption.

   Prints a summary line and two [Bench_out.result] lines ([step] = upload + compute + sync,
   [upload] = [Context.from_host] alone) for [benchmarks/ab.py] and [benchmarks/bin_cells.py].

   Usage: bench_host_feed_step.exe [STEPS] (default 60; STEPS is the first non-[--] argument). *)

open Base
open Ocannl
module IDX = Train.IDX
open Nn_blocks.DSL_modules
module Asgns = Ir.Assignments
module Nd = Ir.Ndarray

let batch_size = 2048
let in_dim = 3072
let hid_dim = 64
let num_classes = 8
let num_batches = 4

(* Separable per-batch pattern: distinct dyadic values across every row, column and batch, while
   letting the host reference compute [x * w1] in O(hid_dim * in_dim) instead of O(batch_size *
   hid_dim * in_dim). *)
let row_scale ~batch_idx row =
  let v = (((row * 17) + (batch_idx * 31) + 5) % 16) - 8 in
  Float.of_int v /. 16.

let col_scale col =
  let v = (((col * 13) + 3) % 8) - 4 in
  Float.of_int v /. 8.

let target_class ~batch_idx row = ((row * 7) + (batch_idx * 3) + 1) % num_classes

let make_input_batch ~batch_idx =
  let open Bigarray in
  let ga = Genarray.create Float32 c_layout [| batch_size; in_dim |] in
  let flat = reshape_1 ga (batch_size * in_dim) in
  let cols = Array.init in_dim ~f:col_scale in
  for r = 0 to batch_size - 1 do
    let rs = row_scale ~batch_idx r in
    let base = r * in_dim in
    for c = 0 to in_dim - 1 do
      Array1.set flat (base + c) (rs *. cols.(c))
    done
  done;
  Nd.as_array Ir.Ops.Single ga

let make_target_batch ~batch_idx =
  let open Bigarray in
  let ga = Genarray.create Float32 c_layout [| batch_size; num_classes |] in
  Genarray.fill ga 0.;
  for r = 0 to batch_size - 1 do
    Genarray.set ga [| r; target_class ~batch_idx r |] 1.
  done;
  Nd.as_array Ir.Ops.Single ga

(* Exact host forward softmax cross-entropy loss on batch 0 from the initial parameter values. *)
let host_loss_batch0 ~w1 ~b1 ~w2 ~b2 =
  let cols = Array.init in_dim ~f:col_scale in
  (* Precompute [proj.(h) = sum_d w1[h, d] * col_scale(d)]. *)
  let proj =
    Array.init hid_dim ~f:(fun h ->
        let base = h * in_dim in
        let acc = ref 0. in
        for d = 0 to in_dim - 1 do
          acc := !acc +. (w1.(base + d) *. cols.(d))
        done;
        !acc)
  in
  let total_loss = ref 0. in
  let hid = Array.create ~len:hid_dim 0. in
  let logits = Array.create ~len:num_classes 0. in
  for r = 0 to batch_size - 1 do
    let rs = row_scale ~batch_idx:0 r in
    for h = 0 to hid_dim - 1 do
      let z = (rs *. proj.(h)) +. b1.(h) in
      hid.(h) <- (if Float.(z > 0.) then z else 0.)
    done;
    let max_l = ref Float.neg_infinity in
    for k = 0 to num_classes - 1 do
      let base = k * hid_dim in
      let acc = ref b2.(k) in
      for h = 0 to hid_dim - 1 do
        acc := !acc +. (w2.(base + h) *. hid.(h))
      done;
      logits.(k) <- !acc;
      if Float.(!acc > !max_l) then max_l := !acc
    done;
    let sum_exp = ref 0. in
    for k = 0 to num_classes - 1 do
      sum_exp := !sum_exp +. Float.exp (logits.(k) -. !max_l)
    done;
    let lse = !max_l +. Float.log !sum_exp in
    let cls = target_class ~batch_idx:0 r in
    total_loss := !total_loss +. (lse -. logits.(cls))
  done;
  !total_loss /. Float.of_int batch_size

let () =
  let steps =
    match
      Sys.get_argv () |> Array.to_list |> List.tl_exn
      |> List.find ~f:(fun s -> not (String.is_prefix s ~prefix:"--"))
    with
    | Some s -> Int.of_string s
    | None -> 60
  in
  Utils.settings.fixed_state_for_init <- Some 42;
  Tensor.unsafe_reinitialize ();
  let input_batches = Array.init num_batches ~f:(fun batch_idx -> make_input_batch ~batch_idx) in
  let target_batches = Array.init num_batches ~f:(fun batch_idx -> make_target_batch ~batch_idx) in
  let input_batch =
    Operation.init ~mutable_:true ~l:"input_batch" ~prec:Ir.Ops.single ~b:[ batch_size ]
      ~o:[ in_dim ]
      ~f:(fun _ -> 0.)
      ~grad_spec:Tensor.Prohibit_grad ()
  in
  let target_batch =
    Operation.init ~mutable_:true ~l:"target_batch" ~prec:Ir.Ops.single ~b:[ batch_size ]
      ~o:[ num_classes ]
      ~f:(fun _ -> 0.)
      ~grad_spec:Tensor.Prohibit_grad ()
  in
  let w1 = TDSL.param "hfs_w1" ~input_dims:[ in_dim ] ~output_dims:[ hid_dim ] () in
  let b1 = TDSL.param "hfs_b1" ~output_dims:[ hid_dim ] () in
  let w2 = TDSL.param "hfs_w2" ~input_dims:[ hid_dim ] ~output_dims:[ num_classes ] () in
  let b2 = TDSL.param "hfs_b2" ~output_dims:[ num_classes ] () in
  let%op hidden = relu ((w1 * input_batch) + b1) in
  let%op logits = (w2 * hidden) + b2 in
  let%op batch_loss =
    Nn_blocks.cross_entropy_loss ~spec:"...|v" ~normalize_by:!..batch_size () ~logits
      ~targets:target_batch
  in
  let update = Train.grad_update batch_loss in
  let learning_rate = TDSL.O.( !. ) 0.01 in
  let sgd = Train.sgd_update ~learning_rate batch_loss in
  let ctx = Context.auto () in
  let backend = Context.backend_name ctx in
  let ctx = Train.init_params ctx IDX.empty batch_loss in
  let ctx, sgd_step = Train.to_routine ctx IDX.empty (Asgns.sequence [ update; sgd ]) in
  (* Read initial weights before step 0 so the host reference evaluates the exact same point. *)
  let ref_loss0 =
    host_loss_batch0
      ~w1:(Context.get_values ctx w1.Tensor.value)
      ~b1:(Context.get_values ctx b1.Tensor.value)
      ~w2:(Context.get_values ctx w2.Tensor.value)
      ~b2:(Context.get_values ctx b2.Tensor.value)
  in
  let ctx_ref = ref ctx in
  let run_step ~batch_idx =
    let t0 = Mtime_clock.counter () in
    ctx_ref := Context.from_host !ctx_ref input_batch.Tensor.value input_batches.(batch_idx);
    ctx_ref := Context.from_host !ctx_ref target_batch.Tensor.value target_batches.(batch_idx);
    let upload_s = Mtime.Span.to_float_ns (Mtime_clock.count t0) /. 1e9 in
    ctx_ref := Context.run !ctx_ref sgd_step;
    Context.sync !ctx_ref;
    let step_s = Mtime.Span.to_float_ns (Mtime_clock.count t0) /. 1e9 in
    (upload_s, step_s)
  in
  (* Step 0: check parity against the independent host reference. *)
  ignore (run_step ~batch_idx:0 : float * float);
  let got_loss0 = (Context.get_values !ctx_ref batch_loss.Tensor.value).(0) in
  let diff0 = Float.abs (got_loss0 -. ref_loss0) in
  let tol = Float.max 1e-4 (1e-4 *. Float.abs ref_loss0) in
  (* Warmup: complete two full epochs over all K batches so recurring arrays are registered. *)
  for i = 1 to (2 * num_batches) - 1 do
    ignore (run_step ~batch_idx:(i % num_batches) : float * float)
  done;
  let upload_times = Array.create ~len:steps 0. in
  let step_times = Array.create ~len:steps 0. in
  for i = 0 to steps - 1 do
    let u, s = run_step ~batch_idx:(i % num_batches) in
    upload_times.(i) <- u;
    step_times.(i) <- s
  done;
  (* Post-warmup untimed parity check on batch 0, whose host array is now registered on CUDA:
     compare the step loss against an independent f64 host reference at the exact post-loop weights,
     so silent corruption on the registered upload path fails [Bench_out.Passed]. *)
  let ref_loss_post =
    host_loss_batch0
      ~w1:(Context.get_values !ctx_ref w1.Tensor.value)
      ~b1:(Context.get_values !ctx_ref b1.Tensor.value)
      ~w2:(Context.get_values !ctx_ref w2.Tensor.value)
      ~b2:(Context.get_values !ctx_ref b2.Tensor.value)
  in
  ignore (run_step ~batch_idx:0 : float * float);
  let final_loss = (Context.get_values !ctx_ref batch_loss.Tensor.value).(0) in
  let diff_post = Float.abs (final_loss -. ref_loss_post) in
  let tol_post = Float.max 1e-4 (1e-4 *. Float.abs ref_loss_post) in
  let check =
    if
      Float.is_finite got_loss0 && Float.is_finite final_loss
      && Float.(diff0 <= tol)
      && Float.(diff_post <= tol_post)
    then Bench_out.Passed
    else
      Bench_out.Failed
        (Printf.sprintf
           "step0 loss got=%.6f ref=%.6f diff=%.6g tol=%.6g; post loss got=%.6f ref=%.6f diff=%.6g \
            tol=%.6g"
           got_loss0 ref_loss0 diff0 tol final_loss ref_loss_post diff_post tol_post)
  in
  let median arr =
    let sorted = Array.sorted_copy arr ~compare:Float.compare in
    sorted.(Array.length sorted / 2)
  in
  let upload_med = median upload_times in
  let step_med = median step_times in
  let input_bytes = Nd.size_in_bytes input_batches.(0) in
  Bench_out.p
    "host_feed_step: backend=%s input_bytes=%d batches=%d steps=%d loss0=%.6f ref0=%.6f \
     final_loss=%.6f upload_median_ms=%.4f step_median_ms=%.4f\n"
    backend input_bytes num_batches steps got_loss0 ref_loss0 final_loss (upload_med *. 1e3)
    (step_med *. 1e3);
  Bench_out.result ~variant:"upload" ~check ~seconds:upload_med;
  Bench_out.result ~variant:"step" ~check ~seconds:step_med;
  Context.release !ctx_ref
