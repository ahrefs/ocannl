(* gh-ocannl-1183 kernel-level A/B: the forward + backprop routine of one dense layer at a
   gpt2_mini_train layer's sizes, whose weight gradient dw[o,i] = sum_{b,s} dy[b,s,o] * x[b,s,i] is
   lowered as backprop lowers every weight gradient of the training step -- the contraction loops
   (b, s) outermost. It measures the kernel the batch-scaling collapse of gh-ocannl-1183 is made of,
   without a fixture or a model around it.

   Usage: bench_wgrad.exe [ocannl configuration flags], with the environment selecting the workload
   and the arm. BENCH_WGRAD_DIMS is "b,s,o,i" (default "256,128,256,1024", the l*_ffn_w2 gradient at
   batch 256). BENCH_TUNE=1 times the tuned schedule (Autotune.tune, no disk cache) instead of the
   untuned default, which goes through Bench_harness.compile_default and so honours the
   model_default_schedule setting. BENCH_REPEATS is the number of timed runs, min taken (default
   20).

   Prints one [wgrad:] line -- the arm, the dims, the w.grad nest's lowered loop extents, the
   min-of-repeats synced time of the whole routine, and for a tuned arm the search's crowned label
   and counters -- then the shipped kernels' isolated times ([Bench_harness.time_shipped_segments]),
   where the [w: wg_w.grad] row is the weight gradient's. Inputs are small positive dyadics, so the
   gradient's checksum is exact whatever the schedule and must agree across arms. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module H = Bench_harness

let () =
  let dims =
    Option.value (Stdlib.Sys.getenv_opt "BENCH_WGRAD_DIMS") ~default:"256,128,256,1024"
    |> String.split ~on:',' |> List.map ~f:Int.of_string
  in
  let nb, ns, no, ni =
    match dims with
    | [ b; s; o; i ] -> (b, s, o, i)
    | _ -> failwith "BENCH_WGRAD_DIMS must be b,s,o,i"
  in
  let tune = Option.equal String.equal (Stdlib.Sys.getenv_opt "BENCH_TUNE") (Some "1") in
  let repeats =
    Option.value_map (Stdlib.Sys.getenv_opt "BENCH_REPEATS") ~default:20 ~f:Int.of_string
  in
  let cyc n m stride = Array.init n ~f:(fun c -> Float.of_int (1 + (c % m)) *. stride) in
  (* The layer as the training step has it: [y = w * x] with a weight of input [i] and output [o]
     over activations of batch [b] and sequence [s]; the loss [sum (y *. dy)] makes [dy] the
     upstream gradient, so backprop's [w.grad] nest is [for b, s, o, i: w.grad[o,i] += dy[b,s,o] *
     x[b,s,i]] -- the shape every weight gradient of gpt2_mini_train has. *)
  let w =
    Operation.init ~l:"wg_w" ~prec:Ir.Ops.single ~i:[ ni ] ~o:[ no ]
      ~f:(fun idx -> Float.of_int (1 + ((idx.(0) + (3 * idx.(1))) % 5)) *. 0.125)
      ~grad_spec:Tensor.Require_grad ()
  in
  let x =
    NTDSL.ndarray
      (cyc (nb * ns * ni) 5 0.125)
      ~label:[ "wg_x" ] ~batch_dims:[ nb; ns ] ~output_dims:[ ni ] ()
  in
  let dy =
    NTDSL.ndarray
      (cyc (nb * ns * no) 7 0.25)
      ~label:[ "wg_dy" ] ~batch_dims:[ nb; ns ] ~output_dims:[ no ] ()
  in
  let%op y = w * x in
  let%op loss = (y *. dy) ++ "... => 0" in
  let wgrad = (Option.value_exn ~here:[%here] w.Tensor.diff).Tensor.grad in
  Train.set_materialized wgrad;
  let comp = Train.grad_update loss in
  let ctx = Context.auto () in
  (* The lowered loop order of the [w.grad] accumulation nest: the point of the workload is that it
     is backprop's, contraction loops outermost. *)
  let loops = ref [] in
  let _ =
    Context.compile_outcome
      ~lowered_transform:(fun opt ->
        let rec writes = function
          | Ir.Low_level.For_loop { body; _ } | Ir.Low_level.If { body; _ } -> writes body
          | Ir.Low_level.Seq (a, b) -> writes a || writes b
          | Ir.Low_level.Set { tn; _ } -> Ir.Tnode.equal tn wgrad
          | _ -> false
        in
        let rec nest acc = function
          | Ir.Low_level.For_loop { to_; body; _ } -> nest ((to_ + 1) :: acc) body
          | _ -> List.rev acc
        in
        List.iter (Ir.Low_level.flat_lines [ opt.Ir.Low_level.llc ]) ~f:(fun stmt ->
            if List.is_empty !loops && writes stmt then loops := nest [] stmt);
        [ opt ])
      ~provenance:Ir.Schedule_outcome.User_schedule ctx comp Ir.Indexing.Empty
  in
  let backend = Context.backend_name ctx in
  let report = ref None in
  let t0 = Unix.gettimeofday () in
  let ctx, routine =
    if tune then
      Autotune.tune ~search:true ~cache_dir:""
        ~report:(fun r -> report := Some r)
        ctx comp Ir.Indexing.Empty
    else H.compile_default ctx comp Ir.Indexing.Empty
  in
  let compile_s = Unix.gettimeofday () -. t0 in
  let ctx = Context.run ctx routine in
  Context.sync ctx;
  let best = ref Float.infinity in
  for _ = 1 to repeats do
    let c0 = Unix.gettimeofday () in
    let ctx = Context.run ctx routine in
    Context.sync ctx;
    best := Float.min !best ((Unix.gettimeofday () -. c0) *. 1000.)
  done;
  let ctx = Context.run ctx routine in
  let checksum = Array.fold (Context.get_values ctx wgrad) ~init:0. ~f:( +. ) in
  let tuned =
    match !report with
    | None -> ""
    | Some r ->
        Printf.sprintf
          " crowned=%s best_ms=%.4f baseline_ms=%.4f sketch_candidates=%d \
           fiss_sketch_candidates=%d timed=%d"
          r.Autotune.best_label r.Autotune.best_ms r.Autotune.baseline_ms
          r.Autotune.sketch_candidates r.Autotune.fiss_sketch_candidates r.Autotune.candidates_timed
  in
  Stdio.printf
    "wgrad: backend=%s arm=%s dims=%d,%d,%d,%d loops=[%s] compile_s=%.2f ms=%.4f checksum=%h%s\n%!"
    backend
    (if tune then "tuned" else "default")
    nb ns no ni
    (String.concat ~sep:"," (List.map !loops ~f:Int.to_string))
    compile_s !best checksum tuned;
  ignore
    (H.time_shipped_segments ~repeats ~ctx ~bindings:Ir.Indexing.Empty [ routine ]
      : (float, string) Result.t list)
