(* gh-494 waypoint 2 / gh-554: the read-before-write decider. The retired concrete-index tracer
   truncated loops (at the retired [virtualize_max_tracing_dim], default 5), so on a padded-conv
   intermediate the consumer's affine reads (e.g. [oh+kh]) reached positions past the traced write
   range and were classified [Recurrent] — spuriously: every read is in fact covered by prior
   in-routine writes. The affine containment query ([Low_level.reads_covered_query]) is now the
   primary decider and is exact where the tracer sampled.

   Pinned here, on a chain of two padded convs whose intermediate has spatial dims 9x9 (larger than
   the retired tracing truncation, so the concrete tracer used to misclassify it): - the query
   proves coverage: [read_before_write] stays false, so the intermediate is not classified as a
   routine input ([input_and_output_nodes]) — under the truncating tracer it was, forcing
   [On_device] and excluding it from buffer aliasing; - executed parity: the default run's values
   match a run with the intermediate forced materialized (the pre-flip placement) — the placement
   difference must not change computed values.

   Printed facts are booleans/PASS lines so the expected output stays backend-stable. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Tn = Ir.Tnode
open Verdict.Claims

(* Deterministic input and (via [fixed_state_for_init]) deterministic conv params, so the two phases
   compute the same function. *)
let build () =
  Utils.settings.fixed_state_for_init <- Some 42;
  Tensor.unsafe_reinitialize ();
  let x_src =
    NTDSL.init ~l:"rbwf_xsrc" ~prec:Ir.Ops.single ~o:[ 9; 9; 2 ]
      ~f:(Ll_test.weighted ~weights:[| 3; 5; 7 |] ~modulus:11 ~offset:0. ~stride:0.125)
      ()
  in
  (* An [init] data node has its layout committed at creation; the padded conv needs a fresh operand
     to grant the halo (see padding_lifecycle.ml). *)
  let x = NTDSL.O.einsum1 "h, w, c => h, w, c" x_src in
  let conv1 =
    Nn_blocks.conv2d ~label:[ "rbwf_c1" ] ~kernel_size:3 ~use_padding:true ~out_channels:3 ()
  in
  let conv2 =
    Nn_blocks.conv2d ~label:[ "rbwf_c2" ] ~kernel_size:3 ~use_padding:true ~out_channels:2 ()
  in
  let h = conv1 x in
  let y = conv2 h in
  (h, y)

let run ~materialize_h =
  let h, y = build () in
  if materialize_h then Train.set_materialized h.Tensor.value;
  Train.set_materialized y.Tensor.value;
  Tn.set_observable y.Tensor.value;
  let ctx = Context.auto () in
  let ctx = Train.init_params ctx Ir.Indexing.Empty y in
  let captured = ref None in
  let transform (opt : LL.optimized) =
    captured := Some opt;
    opt
  in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun o -> [ transform o ])
      ctx (Train.forward y) Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let yv = Context.get_values ctx y.Tensor.value in
  (yv, Option.value_exn !captured, h)

let () =
  let yv_default, opt, h = run ~materialize_h:false in
  let h_tn = h.Tensor.value in
  (match Hashtbl.find opt.LL.traced_store h_tn with
  | None -> p "default: intermediate traced" false
  | Some traced -> p "default: read_before_write stays false" (not traced.LL.read_before_write));
  let (inputs, _outputs), _merge = LL.input_and_output_nodes opt in
  p "default: intermediate is not a routine input" (not (Set.mem inputs h_tn));
  let yv_mat, opt_mat, h_mat = run ~materialize_h:true in
  (match Hashtbl.find opt_mat.LL.traced_store h_mat.Tensor.value with
  | None -> p "materialized: intermediate traced" false
  | Some traced -> p "materialized: read_before_write stays false" (not traced.LL.read_before_write));
  p "materialized: intermediate stays non-virtual"
    (not (Tn.Placements.known_virtual opt_mat.LL.optimize_ctx.LL.placements h_mat.Tensor.value));
  p "parity: same result length" (Array.length yv_default = Array.length yv_mat);
  let max_diff =
    Array.foldi yv_default ~init:0.0 ~f:(fun i acc v -> Float.max acc (Float.abs (v -. yv_mat.(i))))
  in
  Verdict.pass_fail "parity"
    Float.(max_diff <= 1e-5)
    ~detail:(fun () -> Printf.sprintf "max diff %.8f" max_diff)
