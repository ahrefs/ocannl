(* gh-ocannl-1490: bench_conv_diag's census-only and per-seed parity modes, run on a model
   fabricated in memory (the fixtures are not in a fresh checkout), through the library both modes
   call (Bench_seed_parity).

   A small convnet training step (padded 3x3 conv, relu, 2x2 pool, a dense layer, the logits, cross
   entropy, one SGD update), every non-literal materialized as on the materialize-all bench arm, so
   that fission cuts conv segments. The census must cover the whole routine and the fission
   segments; each seed of each keyed segment the search seeds is executed against the default
   pipeline and must match; and the negative control -- every seeded segment's code perturbed after
   scheduling -- must be reported a mismatch for every perturbed seed that executes. Every seed, not
   only the conv ones: the parity machinery is family-agnostic, and the dense layers' matmul seeds
   keep a backend whose conv seeds all decline from leaving the control vacuous.

   Gated on the seeder's own capability judgment for an f32 site, as conv_merged_gpu_seeds is: a GPU
   that withholds the f32 tensorized capability (CUDA at tf32 off) seeds nothing on GPU and skips
   with the aggregation that lifts it. Per-seed verdicts and the worst differences go to stderr; the
   golden holds the claims. *)

open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims
module IDX = Train.IDX
module Asgns = Ir.Assignments
module P = Bench_seed_parity

let batch = 8
let n_batches = 2
let side = 8
let chans = 8
let classes = 4
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_cpu = Ir.Schedule.backend_is_cpu backend_name

(* Deterministic, RNG-free, and varying with every axis of the images. *)
let wave k = Float.of_int ((k * 37 % 19) - 9) /. 10.
let nd debug dims f = Ir.Ndarray.init_array ~debug Ir.Ops.single ~dims ~padding:None ~f

let claims_census =
  [ "census: the whole routine is censused"; "census: fission segments are censused" ]

let claims_parity =
  [
    "parity: at least one seed of a keyed segment executes";
    "parity: every executed seed matches the default pipeline's step";
    "parity: no seed fails to reach its segment, or raises";
    "negative control: at least one perturbed seed executes";
    "negative control: every executed perturbed seed is reported a mismatch";
    "negative control: no perturbed seed fails to reach its segment, raises, or has nothing to \
     perturb";
  ]

let tensorized_capability =
  lazy
    (let ctx = Context.auto () in
     Ll_test.tensorized_capability ~is_gpu:(not on_cpu) ~is_cpu:on_cpu
       ~limits:(Context.hardware_limits ctx) ~a_prec:Ir.Ops.single ~b_prec:Ir.Ops.single
       ~d_prec:Ir.Ops.single)

let () =
  let total = batch * n_batches in
  let x_nd =
    nd "sp_x" [| total; side; side; chans |] (fun i ->
        wave ((((((i.(0) * side) + i.(1)) * side) + i.(2)) * chans) + i.(3)))
  in
  let y_nd = nd "sp_y" [| total; classes |] (fun i -> if i.(1) = i.(0) % classes then 1. else 0.) in
  let xs = TDSL.rebatch ~l:"xs" x_nd () in
  let ys = TDSL.rebatch ~l:"ys" y_nd () in
  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let%op batch_x = xs @| batch_n in
  let%op batch_y = ys @| batch_n in
  let conv =
    Nn_blocks.conv2d ~label:[ "sp_conv" ] ~kernel_size:3 ~use_padding:true ~out_channels:16 ()
  in
  let pool = Nn_blocks.max_pool2d ~stride:2 ~window_size:2 () in
  let fc = Nn_blocks.mlp_layer ~label:[ "sp_fc" ] ~hid_dim:16 () in
  let%op logits = { w_logits } * fc (pool (relu (conv batch_x))) in
  let%op loss =
    Nn_blocks.cross_entropy_loss ~spec:"...|v" ~normalize_by:!..batch () ~logits ~targets:batch_y
  in
  Train.every_non_literal_materialized loss;
  let step =
    let update = Train.grad_update loss in
    let sgd = Train.sgd_update ~learning_rate:(TDSL.O.( !. ) 0.1) loss in
    Asgns.sequence [ update; sgd ]
  in
  let ctx = Train.init_params (Context.auto ()) bindings loss in
  let static_indices = [ batch_n ] in
  let opt = Bench_harness.capture_lowering ctx step bindings in
  (* The census mode. *)
  let census = P.census_lines ctx ~static_indices opt in
  List.iter census ~f:(fun l -> Stdio.eprintf "%s (not part of the golden)\n" l);
  let censused what = List.exists census ~f:(String.is_substring ~substring:what) in
  p (List.nth_exn claims_census 0) (censused "where=\"tune whole-routine\"");
  p (List.nth_exn claims_census 1) (censused "where=\"tune coarse segment ");
  (* The parity mode, and its negative control. *)
  let gpu_withheld =
    match Lazy.force tensorized_capability with
    | `Withheld aggregation when not on_cpu -> Some aggregation
    | `Withheld _ | `Advertised -> None
  in
  match gpu_withheld with
  | Some aggregation ->
      List.iter claims_parity ~f:(Verdict.skipped ~aggregation ~backend:backend_name)
  | None ->
      let parity ~perturb =
        snd
          (P.parity ~perturb ~select:P.any_seed
             ~on_result:(fun r ->
               Stdio.eprintf "%s%s (not part of the golden)\n"
                 (if perturb then "perturbed " else "")
                 (P.verdict_line r))
             ctx step bindings
             ~bind:(fun r -> IDX.find_exn r.Context.bindings batch_n := 0)
             ~static_indices opt)
      in
      let executed =
        List.filter ~f:(fun r -> match r.P.verdict with Match _ | Mismatch _ -> true | _ -> false)
      in
      let results = parity ~perturb:false in
      let matched = function { P.verdict = Match _; _ } -> true | _ -> false in
      let mismatched = function { P.verdict = Mismatch _; _ } -> true | _ -> false in
      let broken = function
        | { P.verdict = Not_applied | Failed _ | Not_perturbed; _ } -> true
        | _ -> false
      in
      p (List.nth_exn claims_parity 0) (not (List.is_empty (executed results)));
      p_all (List.nth_exn claims_parity 1) (executed results) ~f:matched;
      p_none (List.nth_exn claims_parity 2) results ~f:broken;
      let perturbed = parity ~perturb:true in
      p (List.nth_exn claims_parity 3) (not (List.is_empty (executed perturbed)));
      p_all (List.nth_exn claims_parity 4) (executed perturbed) ~f:mismatched;
      p_none (List.nth_exn claims_parity 5) perturbed ~f:broken
