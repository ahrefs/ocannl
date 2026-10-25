(* gh-ocannl-1405: the untuned CUDA/HIP default seeds and sinks narrow-storage sites only under the
   numerics gate [gpu_default_seed_narrow_storage] ([Autotune.seeded_default]'s [seed_narrow]).

   A site whose accumulator precision is wider than its storage precision (a bf16 gradient on CUDA,
   where [accum_prec] widens bf16 to f32) is left to the default pipeline by the seeding and by the
   reduction sink. Backprop lowers a weight or bias gradient with the batch rows outermost, so the
   default reads the bf16 cell, adds one contribution at f32 and rounds back to bf16 once per row.
   The gate admits those sites, with two different outcomes: - a seeded contraction (the weight
   gradient) keeps its result bitwise: the sketch's [Privatize] mints its tile wide only where the
   default's serial rendering widens too (gh-ocannl-1151), so the tile stays at bf16 and narrows
   once per row, as the default does; - a sunk reduction (the bias gradient) does not: below its
   write loops the accumulator is one code generation widens, so the whole reduction runs at f32 and
   rounds once -- the summation order is the default's, only the number of roundings changes,
   towards the exact sum.

   Four parts. 1. Payloads: the [approximate] profile turns the gate on, and [reproducible] pins it
   off, at its default (part 4 executes the unset default). 2. Structure, on every backend: the
   seeding rule refuses a bf16 matmul output under a widening [accum_prec] unless admitted, and
   admitted it seeds the geometry it seeds at storage-width accumulation. 3. Executed: one dense
   layer with bf16 weight and bias gradients, the inputs NOT dyadic, compiled on the default
   pipeline and through the seeded default with the admission off and on. Off, both gradients are
   bitwise the default's on every backend. On, where the seeded default runs and [accum_prec] widens
   bf16 (the backend's own resolution), the weight gradient is seeded and stays bitwise the
   default's, and the bias gradient is sunk, closer to an f64 host reference than the default's, and
   within one bf16 rounding of it in every cell, which the default's is not. Where nothing widens,
   the admission changes nothing. 4. Configuration: the key reaches the seeded default and
   [model_default] when no override is passed. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module Tn = Ir.Tnode
module LL = Ir.Low_level

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let key = "gpu_default_seed_narrow_storage"

(* As in seeded_default: every product accumulation an explicit FMA, so that the bitwise claims are
   about summation order and rounding count, not the GPU compiler's per-kernel contraction. *)
let () = Unix.putenv "OCANNL_FP_CONTRACT" "true"
let bf16 = Ir.Ops.bfloat16

(* {2 1. Payloads} *)

let () =
  let payload name =
    Utils.parse_config_lines ~source:("profile " ^ name)
      (String.split_lines (List.Assoc.find_exn Utils.profile_payloads name ~equal:String.equal))
  in
  let stated name = List.Assoc.find (payload name) key ~equal:String.equal in
  p "the approximate profile turns the gate on"
    (Option.equal String.equal (stated "approximate") (Some "true"));
  p "the reproducible profile pins the gate off, its default"
    (Option.equal String.equal (stated "reproducible") (Some "false"))

(* {2 2. The seeding rule} *)

let scratch (o : LL.optimized) =
  {
    o with
    LL.traced_store = Hashtbl.copy o.LL.traced_store;
    LL.optimize_ctx = LL.copy_optimize_ctx o.LL.optimize_ctx;
  }

let widen p = if Ir.Ops.equal_prec p bf16 then Ir.Ops.single else p

let () =
  let a =
    TDSL.ndarray
      (Array.create ~len:(128 * 64) 1.)
      ~label:[ "sns_a" ] ~input_dims:[ 64 ] ~output_dims:[ 128 ] ()
  in
  let b =
    TDSL.ndarray
      (Array.create ~len:(64 * 128) 1.)
      ~label:[ "sns_b" ] ~input_dims:[ 128 ] ~output_dims:[ 64 ] ()
  in
  let%op c = a * b in
  Tn.update_prec c.Tensor.value bf16;
  let lowered =
    Context.lowered_for_decisions ~name:"sns_matmul" (Context.auto ()) (Train.forward c)
      Ir.Indexing.Empty
  in
  let seeded ?admit_narrow accum_prec =
    Autotune.seeded_params ?admit_narrow ~limits:Ir.Backend_intf.no_hardware_limits ~accum_prec
      ~min_reduction:1 (scratch lowered)
  in
  let at_storage = seeded Fn.id in
  p "a bf16 matmul output accumulated at bf16 is seeded (the control)" (Option.is_some at_storage);
  p "under a widening accumulator the rule refuses it" (Option.is_none (seeded widen));
  p "under a widening accumulator the rule refuses it with the admission off too"
    (Option.is_none (seeded ~admit_narrow:false widen));
  p "admitted, the rule seeds it at the geometry it seeds at storage-width accumulation"
    (Option.is_some at_storage
    && Option.equal
         (fun x y -> String.equal (Autotune.sketch_params_label x) (Autotune.sketch_params_label y))
         (seeded ~admit_narrow:true widen) at_storage)

(* {2 3. Executed: a dense layer with bf16 gradients} *)

let nb = 4
and ns = 64
and no = 64
and ni = 96

(* Deterministic values with full mantissas: [k / 1009 - 1/2] over a stride coprime to 1009. *)
let values ~n ~stride =
  Array.init n ~f:(fun c -> (Float.of_int (c * stride % 1009) /. 1009.) -. 0.5)

let w =
  Operation.init ~l:"sns_w" ~prec:Ir.Ops.single ~i:[ ni ] ~o:[ no ]
    ~f:(fun idx ->
      Ll_test.weighted ~weights:[| 31; 17 |] ~modulus:1013 ~offset:(-506.5) ~stride:(1. /. 1013.)
        idx)
    ~grad_spec:Tensor.Require_grad ()

let b =
  Operation.init ~l:"sns_b" ~prec:Ir.Ops.single ~o:[ no ]
    ~f:(fun idx -> Float.of_int (idx.(0) * 13 % 101) /. 101.)
    ~grad_spec:Tensor.Require_grad ()

let x_values = values ~n:(nb * ns * ni) ~stride:389
let dy_values = values ~n:(nb * ns * no) ~stride:577
let x = NTDSL.ndarray x_values ~label:[ "sns_x" ] ~batch_dims:[ nb; ns ] ~output_dims:[ ni ] ()
let dy = NTDSL.ndarray dy_values ~label:[ "sns_dy" ] ~batch_dims:[ nb; ns ] ~output_dims:[ no ] ()
let%op y = (w * x) + b
let%op loss = (y *. dy) ++ "... => 0"
let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad
let wgrad = grad_of w
let bgrad = grad_of b

let () =
  Tn.update_prec wgrad bf16;
  Tn.update_prec bgrad bf16;
  Train.set_materialized wgrad;
  Train.set_materialized bgrad

let comp = Train.grad_update loss

(* The f64 host references: [y.grad] is [dy], exactly, so [w.grad[o,i] = sum_r dy[r,o] * x[r,i]] and
   [b.grad[o] = sum_r dy[r,o]], over the f32 inputs. *)
let f32 v = Int32.float_of_bits (Int32.bits_of_float v)
let rows = nb * ns

let wgrad_ref =
  Array.init (no * ni) ~f:(fun c ->
      let o = c / ni and i = c % ni in
      let acc = ref 0. in
      for r = 0 to rows - 1 do
        acc := !acc +. (f32 dy_values.((r * no) + o) *. f32 x_values.((r * ni) + i))
      done;
      !acc)

let bgrad_ref =
  Array.init no ~f:(fun o ->
      let acc = ref 0. in
      for r = 0 to rows - 1 do
        acc := !acc +. f32 dy_values.((r * no) + o)
      done;
      !acc)

let bitwise a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)
let error got want = Array.fold2_exn got want ~init:0. ~f:(fun acc g r -> acc +. Float.abs (g -. r))

(* Within one bf16 rounding of the reference: half a bf16 unit in the last place is at most 2^-8 of
   the magnitude (8 significant bits); the f32 accumulation's own error is about 2^-24 per row, so
   [2^-8 * |ref| + 2^-16] bounds a single rounding of a near-exact sum and no more. *)
let within_one_rounding got want = Float.(abs (got -. want) <= (abs want /. 256.) +. (1. /. 65536.))

let run compile =
  let ctx, routine = compile (Context.auto ()) comp Ir.Indexing.Empty in
  let ctx = Context.run ctx routine in
  Context.sync ctx;
  (Context.get_values ctx wgrad, Context.get_values ctx bgrad)

let seeded ?seed_narrow () =
  let report = ref None in
  let values =
    run (fun ctx comp bindings ->
        Autotune.seeded_default ~min_reduction:128 ?seed_narrow
          ~report:(fun r -> report := Some r)
          ctx comp bindings)
  in
  (values, Option.value_exn ~here:[%here] !report)

(* Where the admission can change anything: the seeded default runs here, and the backend's own
   accumulator resolution widens bf16. *)
let narrow_here =
  let ctx = Context.auto () in
  Autotune.seeded_default_backend backend_name
  && not
       (Ir.Ops.equal_prec ((Context.codegen_capabilities ctx).Ir.Backend_intf.accum_prec bf16) bf16)

let census (r : Autotune.seeded_choice) =
  List.map r.sd_seeded ~f:(fun (at, p) ->
      Printf.sprintf "u%d %s" at (Autotune.sketch_params_label p))
  @ List.map r.sd_sunk ~f:(fun (at, n) -> Printf.sprintf "u%d sunk %d" at n)

let () =
  Stdio.eprintf "seeded_narrow_storage: narrow_here=%b on %s (not part of the golden)\n%!"
    narrow_here backend_name;
  let w_default, b_default = run Context.compile in
  let (w_off, b_off), r_off = seeded ~seed_narrow:false () in
  let (w_on, b_on), r_on = seeded ~seed_narrow:true () in
  List.iter (census r_on) ~f:(fun line ->
      Stdio.eprintf "admitted: %s (not part of the golden)\n" line);
  p_exists "the weight gradient is not trivially zero" (Array.to_list w_default) ~f:(fun v ->
      Float.(abs v > 1e-3));
  (* Off: the default's numerics, everywhere. *)
  Verdict.gated ~when_:narrow_here ~on:backend_name
    "where accum_prec widens bf16, with the admission off the seeded default seeds and sinks fewer \
     segments than with it on"
    (List.length r_off.sd_seeded < List.length r_on.sd_seeded
    && List.length r_off.sd_sunk < List.length r_on.sd_sunk);
  p_all2 "with the admission off the weight gradient is bitwise the default pipeline's" w_off
    w_default ~f:bitwise;
  p_all2 "with the admission off the bias gradient is bitwise the default pipeline's" b_off
    b_default ~f:bitwise;
  (* On: the contraction seeded and unchanged, the reduction sunk and more accurate. *)
  Verdict.gated ~when_:narrow_here ~on:backend_name
    "where accum_prec widens bf16, with the admission on the weight gradient is seeded"
    (not (List.is_empty r_on.sd_seeded));
  Verdict.gated ~when_:narrow_here ~on:backend_name
    "where accum_prec widens bf16, with the admission on the bias gradient is sunk"
    (not (List.is_empty r_on.sd_sunk));
  Verdict.gated ~when_:narrow_here ~on:backend_name
    "where accum_prec widens bf16, the admitted routine compiles without falling back"
    (not r_on.sd_fell_back);
  Stdio.eprintf
    "abs error vs f64: weight default %g admitted %g; bias default %g admitted %g (not part of the \
     golden)\n\
     %!"
    (error w_default wgrad_ref) (error w_on wgrad_ref) (error b_default bgrad_ref)
    (error b_on bgrad_ref);
  p_all2 "with the admission on the seeded weight gradient is bitwise the default pipeline's" w_on
    w_default ~f:bitwise;
  Verdict.gated_exists ~when_:narrow_here ~on:backend_name
    "where accum_prec widens bf16, the admission changes the bias gradient in some cell"
    (List.zip_exn (Array.to_list b_on) (Array.to_list b_default))
    ~f:(fun (a, b) -> not (bitwise a b));
  Verdict.gated ~when_:narrow_here ~on:backend_name
    "where accum_prec widens bf16, the admitted bias gradient is closer to the f64 reference"
    Float.(error b_on bgrad_ref < error b_default bgrad_ref);
  Verdict.gated_all ~when_:narrow_here ~on:backend_name
    "where accum_prec widens bf16, every admitted bias-gradient cell is one bf16 rounding of the \
     reference"
    (List.zip_exn (Array.to_list b_on) (Array.to_list bgrad_ref))
    ~f:(fun (g, r) -> within_one_rounding g r);
  Verdict.gated_exists ~when_:narrow_here ~on:backend_name
    "where accum_prec widens bf16, some default bias-gradient cell is not (the control)"
    (List.zip_exn (Array.to_list b_default) (Array.to_list bgrad_ref))
    ~f:(fun (g, r) -> not (within_one_rounding g r));
  Verdict.gated_all ~when_:(not narrow_here) ~on:backend_name
    "where nothing widens, the admission leaves the weight gradient bitwise the default's"
    (List.zip_exn (Array.to_list w_on) (Array.to_list w_default))
    ~f:(fun (a, b) -> bitwise a b);
  Verdict.gated_all ~when_:(not narrow_here) ~on:backend_name
    "where nothing widens, the admission leaves the bias gradient bitwise the default's"
    (List.zip_exn (Array.to_list b_on) (Array.to_list b_default))
    ~f:(fun (a, b) -> bitwise a b);
  (* {2 4. The configuration key} Unset (the test config does not state it), the seeded default
     decides as with the admission off; stated true, as with it on. *)
  let _, r_unset = seeded () in
  Verdict.gated ~when_:narrow_here ~on:backend_name
    "where accum_prec widens bf16, with the key unset the seeded default decides as with the \
     admission off, short of what the admission takes"
    (List.equal String.equal (census r_unset) (census r_off)
    && List.length (census r_unset) < List.length (census r_on));
  Unix.putenv ("OCANNL_" ^ String.uppercase key) "true";
  let _, r_keyed = seeded () in
  Verdict.gated ~when_:narrow_here ~on:backend_name
    "where accum_prec widens bf16, the key stated true seeds what the admission seeds"
    ((not (List.is_empty (census r_on))) && List.equal String.equal (census r_keyed) (census r_on));
  (* [model_default] starts from the seeded plan under the configured admission, and its
     numerics-preserving set admits the narrow contraction with it: whichever scalar blocktile it
     picks, the weight gradient keeps the default's narrowing points. *)
  let w_model, _ = run (fun ctx comp bindings -> Autotune.model_default ctx comp bindings) in
  p_all2 "with the key true the model's weight gradient is bitwise the default pipeline's" w_model
    w_default ~f:bitwise
