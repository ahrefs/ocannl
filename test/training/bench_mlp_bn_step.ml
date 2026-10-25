(* Diagnostic bench (not a test) for gh-ocannl-1343: the mlp_bn_names training step (Bengio MLP with
   batch norm, test/training/mlp_bn_names.ml: batch 1000, context 3, embedding 10, hidden 200,
   vocabulary 28) compiled through the UNTUNED default pipeline ([Train.to_routine]), timed in
   steady state on fixed synthetic inputs. Prints the step's segment census (segments, and how many
   carry a Grid launch) and one timing line:

   mlp_bn_step: segments=<n> grid_segments=<g> steps=<n> median_ms=<m> mean_ms=<a> min_ms=<x>

   and the median as a [Bench_out.result] line (variant [step]), which is what the benchmark drivers
   (benchmarks/bin_cells.py, benchmarks/ab.py) read.

   Usage: bench_mlp_bn_step.exe [STEPS] (default 200, after 20 warmup steps; STEPS is the first
   argument that is not an --ocannl_ flag, wherever it sits). Run from a directory whose nearest
   ocannl_config names the backend (benchmarks/ does), or pass --ocannl_backend=cc. *)
open Base
open Ocannl
open Stdio
module IDX = Train.IDX
open Nn_blocks.DSL_modules
module Asgns = Ir.Assignments
module LL = Ir.Low_level

let block_size = 3
let embed_dim = 10
let hid_dim = 200
let vocab_size = Dataprep.Names.dict_size
let batch_size = 1000

let () =
  let steps =
    (* The drivers put their pinned flags first, so the count is found, not assumed at argv.(1). *)
    match
      Sys.get_argv () |> Array.to_list |> List.tl_exn
      |> List.find ~f:(fun s -> not (String.is_prefix s ~prefix:"--"))
    with
    | Some s -> Int.of_string s
    | None -> 200
  in
  Utils.settings.fixed_state_for_init <- Some 3;
  Tensor.unsafe_reinitialize ();
  let step_n, bindings = IDX.get_static_symbol IDX.empty in
  let term label ~batch_dims ~output_dims ~f =
    let open Bigarray in
    let dims = Array.of_list (batch_dims @ output_dims) in
    let ga = Genarray.create Float32 c_layout dims in
    let n = Array.fold dims ~init:1 ~f:( * ) in
    let flat = reshape_1 ga n in
    for i = 0 to n - 1 do
      Array1.set flat i (f i)
    done;
    let nd = Ir.Ndarray.as_array Ir.Ops.Single ga in
    Tensor.term ~init_data:(Reshape nd) ~grad_spec:If_needed ~label:[ label ] ~batch_dims
      ~input_dims:[] ~output_dims ()
  in
  (* Fixed synthetic one-hot contexts and targets: a deterministic token per (example, position). *)
  let token i = i * 7919 % vocab_size in
  let input_batch =
    term "input_batch" ~batch_dims:[ batch_size; block_size ] ~output_dims:[ vocab_size ]
      ~f:(fun i -> if i % vocab_size = token (i / vocab_size) then 1. else 0.)
  in
  let target_batch =
    term "target_batch" ~batch_dims:[ batch_size ] ~output_dims:[ vocab_size ] ~f:(fun i ->
        if i % vocab_size = token ((i / vocab_size) + 1) then 1. else 0.)
  in
  let bn1 = Nn_blocks.batch_norm1d ~label:[ "bn1" ] () in
  let%op embed input = { c; o = [ embed_dim ] } * input in
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
  let%op max_l = train_logits @^^ "... | ... => ... | 0" in
  let%op shifted = train_logits - max_l in
  let%op lse = log (exp shifted ++ "... | ... => ... | 0") in
  let%op log_probs = shifted - lse in
  let%op nll = neg ((target_batch *. log_probs) ++ "... | ... => |->0") in
  let%op batch_loss = (nll ++ "... => 0") /. !..batch_size in
  let update = Train.grad_update batch_loss in
  let%op learning_rate = 0.1 *. ((1.5 *. !..steps) - !@step_n) /. !..steps in
  let sgd = Train.sgd_update ~learning_rate batch_loss in
  let ctx = Context.auto () in
  let ctx = Train.init_params ctx bindings batch_loss in
  let ctx, sgd_step = Train.to_routine ctx bindings (Asgns.sequence [ update; sgd ]) in
  (* The kernel segments the compile shipped. *)
  let segments = sgd_step.Context.segments in
  let grid_segments =
    List.count segments ~f:(fun seg ->
        Array.exists (LL.launch_dims seg.LL.llc).grid ~f:(fun g -> g > 1))
  in
  let step_ref = IDX.find_exn sgd_step.Context.bindings step_n in
  step_ref := 0;
  let run () =
    Train.run ctx sgd_step;
    Int.incr step_ref
  in
  for _ = 1 to 20 do
    run ()
  done;
  Context.sync ctx;
  let times =
    Array.init steps ~f:(fun _ ->
        let t0 = Unix.gettimeofday () in
        run ();
        Context.sync ctx;
        (Unix.gettimeofday () -. t0) *. 1000.)
  in
  let sorted = Array.sorted_copy times ~compare:Float.compare in
  let median = sorted.(steps / 2) in
  let mean = Array.fold times ~init:0. ~f:( +. ) /. Float.of_int steps in
  printf
    "mlp_bn_step: backend=%s segments=%d grid_segments=%d steps=%d median_ms=%.4f mean_ms=%.4f \
     min_ms=%.4f\n\
     %!"
    (Context.backend_name ctx) (List.length segments) grid_segments steps median mean sorted.(0);
  Bench_out.result ~variant:"step"
    ~check:
      (Bench_out.Unchecked "a timing of the training step; its loss is not compared to an oracle")
    ~seconds:(median /. 1000.)
