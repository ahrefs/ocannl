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
   golden holds the claims.

   gh-ocannl-1602: a tf32 arm (CUDA, the [-tf32] rule) rounds a tensorized conv's multiplicands, so
   the parity reference rounds the seed's accumulations the same way. The step is built to make that
   load-bearing: the conv kernel is set to 1/8 and the images to multiples of 1/8, so every sum is
   exact whatever the order, except one pool window of image 0 whose two top cells differ by [3 *
   2^-12] in f32 -- the column only the first reads holds [1 + 2^-12], which tf32 rounds to 1 -- and
   tie in tf32, so the pool's backward routes the gradient to both under a tf32 conv and to one
   under an f32 conv. Every seed must match the rounded reference; against the plain reference some
   tf32-rounded conv seed must mismatch. On an arm that reads f32 as f32 those two claims skip. *)

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

(* Deterministic, RNG-free, and varying with every axis of the images; multiples of 1/8, exact in
   tf32, so that with the 1/8 kernel every conv sum is exact. *)
let wave k = Float.of_int ((k * 37 % 19) - 9) /. 8.
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

let claims_tf32 =
  [
    "tf32: some executed seed's reference is rounded to tf32";
    "tf32 control: against the plain reference, some tf32-rounded conv seed mismatches";
  ]

(* Image 0's near-tie pool window (output rows 2-3, columns 2-3). Output cell (h, w) reads input
   rows h+o..h+o+2 and columns w+o..w+o+2 for the padded conv's offset [o], 0 or -1; the block below
   is laid out to work for either. Rows 1-4 of columns 1-5 hold 1, except columns 1 and 2, which
   hold 1 + 2^-12, and row 5 of columns 0-6 holds 0. The window's top-left cell reads one more
   perturbed column than the top-right one, so in f32 it is strictly larger by 3 * 2^-12 (a bottom
   cell can tie it, never exceed it), while tf32 rounds the perturbation away and the top-right cell
   ties it. The rest of the image is [wave], as the other images are, so the gradient reaching the
   window is not killed by dead activations. *)
let image ~img ~h ~w k =
  if img = 0 && h >= 1 && h <= 4 && w >= 1 && w <= 5 then if w <= 2 then 1. +. (2. **. -12.) else 1.
  else if img = 0 && h = 5 && w <= 6 then 0.
  else wave k

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
        image ~img:i.(0) ~h:i.(1) ~w:i.(2)
          ((((((i.(0) * side) + i.(1)) * side) + i.(2)) * chans) + i.(3)))
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
  (* The conv's kernel at 1/8 and its bias at 0: exact in tf32, so only the images' low bits differ
     between a tf32 and an f32 conv. *)
  let ctx =
    Set.fold loss.Tensor.params ~init:ctx ~f:(fun ctx p ->
        let tn = p.Tensor.value in
        let dn = Ir.Tnode.debug_name tn in
        if String.is_substring dn ~substring:"sp_conv" then
          Context.set_values ctx tn
            (Array.create ~len:(Ir.Tnode.num_elems tn)
               (if String.is_substring dn ~substring:"bias" then 0. else 0.125))
        else ctx)
  in
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
  let limits = Context.hardware_limits ctx in
  let tf32_withheld =
    if P.f32_reads_tf32 limits then None
    else if P.f32_reads_tf32_under_tf32_policy limits then Some `Environment
    else Some `Backend
  in
  match gpu_withheld with
  | Some aggregation ->
      List.iter (claims_parity @ claims_tf32)
        ~f:(Verdict.skipped ~aggregation ~backend:backend_name)
  | None -> (
      let parity ?(round = true) ?(select = P.any_seed) ~perturb () =
        snd
          (P.parity ~perturb ~round ~select
             ~on_result:(fun r ->
               Stdio.eprintf "%s%s%s (not part of the golden)\n"
                 (if perturb then "perturbed " else "")
                 (if round then "" else "plain reference: ")
                 (P.verdict_line r))
             ctx step bindings
             ~bind:(fun r -> IDX.find_exn r.Context.bindings batch_n := 0)
             ~static_indices opt)
      in
      let executed =
        List.filter ~f:(fun r -> match r.P.verdict with Match _ | Mismatch _ -> true | _ -> false)
      in
      let results = parity ~perturb:false () in
      let matched = function { P.verdict = Match _; _ } -> true | _ -> false in
      let mismatched = function { P.verdict = Mismatch _; _ } -> true | _ -> false in
      let broken = function
        | { P.verdict = Not_applied | Unjudged _ | Failed _ | Not_perturbed; _ } -> true
        | _ -> false
      in
      p (List.nth_exn claims_parity 0) (not (List.is_empty (executed results)));
      p_all (List.nth_exn claims_parity 1) (executed results) ~f:matched;
      p_none (List.nth_exn claims_parity 2) results ~f:broken;
      let perturbed = parity ~perturb:true () in
      p (List.nth_exn claims_parity 3) (not (List.is_empty (executed perturbed)));
      p_all (List.nth_exn claims_parity 4) (executed perturbed) ~f:mismatched;
      p_none (List.nth_exn claims_parity 5) perturbed ~f:broken;
      match tf32_withheld with
      | Some aggregation ->
          List.iter claims_tf32 ~f:(Verdict.skipped ~aggregation ~backend:backend_name)
      | None ->
          let rounded r = not (List.is_empty r.P.rounded) in
          p_exists (List.nth_exn claims_tf32 0) (executed results) ~f:rounded;
          let rounded_labels =
            List.filter_map results ~f:(fun r -> if rounded r then Some r.P.label else None)
          in
          let plain = parity ~round:false ~select:P.conv_seed ~perturb:false () in
          p_exists (List.nth_exn claims_tf32 1) plain ~f:(fun r ->
              mismatched r && List.mem rounded_labels r.P.label ~equal:String.equal))
