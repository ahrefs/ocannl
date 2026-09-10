(* gh-ocannl-498 rematerialization: budget-driven recompute-vs-store, on top of the gh-ocannl-489
   liveness planner. The subject is one training step compiled as ONE routine (forward + zero-grads
   + backprop + SGD) — the whole-step program a memory budget targets.

   Four layers:

   1. Calibration: the routine's scored footprint under the default policy (the baseline) and under
   [Memory_budget.Minimize] (every flip that still relieves anything). Everything below is expressed
   relative to those two numbers, so the test states no byte constants and stays backend-stable —
   statement-granularity (CPU) and segment-granularity (GPU) liveness find different spans, hence
   different absolute footprints and different achievable relief.

   2. The selector's contract: a budget at the baseline takes no flips; a budget between the
   baseline and the minimum is met, with the reported footprint actually under it (the footprint
   assertion); an unreachable budget is best-effort, reporting NOT-within rather than raising or
   forcing illegal flips; tighter budgets never give larger footprints; and planning twice picks
   exactly the same flips (it is a deterministic pass, not a search).

   3. Executed parity across the gate: the loss trajectory of six steps is bitwise identical with
   the budget off and with a budget that demotes materialized activations to recompute-at-use. This
   is the executable half of the structural-vs-executable rule — a flip that changed a value, not
   just a placement, shows up here and nowhere in the plan. The step's actual device footprint also
   has to shrink, not just the modelled one.

   4. Default-off: with no budget argument and no [memory_budget] config key nothing is planned and
   the losses are the ones section 3's off-arm recorded — the gate's off position is the previous
   behavior.

   [buffer_aliasing] is the planner's precondition and is enabled here from the environment (the
   config is read afresh at each compile); a budget requested without it raises, which is layer 0.
   Printed facts are booleans/PASS lines so the expected output stays backend-stable; the concrete
   byte counts go to stderr for local debugging. *)

open Base
open Stdio
open Ocannl
module IDX = Train.IDX
open Nn_blocks.DSL_modules
module Tn = Ir.Tnode
module Asgns = Ir.Assignments
open Verdict.Claims

(* The gate's off position must really be off. An environment setting outranks the copied test
   config, so an ambient OCANNL_MEMORY_BUDGET would make the "budget off" phase plan after all and
   quietly invalidate both the default-off and the footprint-reduction assertions. Pin it here; the
   budgeted phases pass their budget as an argument, not through the environment. *)
let () = Unix.putenv "OCANNL_MEMORY_BUDGET" "0"

(* Same reasoning for the planner's own precondition: layer 0 asserts that a budget without
   [buffer_aliasing] raises, so the gate must start OFF regardless of what the process inherited.
   The later phases switch it on explicitly. *)
let () = Unix.putenv "OCANNL_BUFFER_ALIASING" "false"

(* And the model schedule gate, which the loss-trajectory parity assertion is sensitive to for a
   reason this feature created: a budgeted compile deliberately bypasses [model_default_schedule] to
   use the pipeline its footprint was scored against. With the gate ambiently on, the budget-off
   phase would take the model's pick and the budget-on phase the default one, so the two arms would
   differ in SCHEDULE as well as placement and could diverge on reduction ordering alone -- a
   failure that looks like rematerialization changing values but is not. *)
let () = Unix.putenv "OCANNL_MODEL_DEFAULT_SCHEDULE" "false"

(* The config-key parser, exercised through the environment (values are re-read on each access). The
   suffix scaling is the interesting part: a syntactically valid but absurd setting must be REJECTED
   rather than wrapped into a small or negative target, which the planner would honor as an
   unreachably tight budget. *)
let () =
  let setting s =
    Unix.putenv "OCANNL_MEMORY_BUDGET" s;
    match Train.memory_budget_setting () with
    | None -> "none"
    | Some Memory_budget.Minimize -> "minimize"
    | Some (Memory_budget.Bytes n) -> Int.to_string n
    | exception Utils.User_error _ -> "rejected"
  in
  p "budget parse: 0 is off" (String.equal (setting "0") "none");
  p "budget parse: minimize" (String.equal (setting "minimize") "minimize");
  p "budget parse: plain bytes" (String.equal (setting "4096") "4096");
  p "budget parse: K/M/G suffixes scale by 1024"
    (String.equal (setting "2K") "2048"
    && String.equal (setting "3M") "3145728"
    && String.equal (setting "1G") "1073741824");
  p "budget parse: a suffix that would overflow is rejected"
    (String.equal (setting "5000000000G") "rejected");
  p "budget parse: nonsense is rejected" (String.equal (setting "banana") "rejected");
  Unix.putenv "OCANNL_MEMORY_BUDGET" "0"

(* The relief-per-cost comparator, which ranks the candidates. Numerators are byte counts and CAN be
   negative — inlining a node can lengthen other nodes' spans and cost footprint rather than free it
   — and OCaml's division truncates toward zero, so a continued-fraction descent that assumes
   non-negative numerators silently inverts those comparisons. Checked against float division: the
   wrong tool for the real comparator (not bit-reproducible, and it overflows differently) but a
   fine oracle at these magnitudes. *)
let () =
  let oracle ra ca rb cb =
    Float.compare (Float.of_int ra /. Float.of_int ca) (Float.of_int rb /. Float.of_int cb)
  in
  let cases =
    [
      (* The sign cases a descent over truncating division gets backwards. *)
      (-1, 10, 1, 10);
      (1, 10, -1, 10);
      (0, 1, -1, 5);
      (-1, 5, 0, 1);
      (-5, 2, -1, 2);
      (-1, 2, -5, 2);
      (* Ordinary positives, an exact tie, and pairs close together or far apart. *)
      (3, 4, 5, 6);
      (7, 3, 7, 3);
      (1, 1000000, 1, 999999);
      (32768, 512, 32768, 256);
    ]
  in
  p_all "ratio: agrees with exact division on signs, ties and magnitudes" cases
    ~f:(fun (ra, ca, rb, cb) -> Memory_budget.compare_relief_ratio ra ca rb cb = oracle ra ca rb cb);
  (* The reason it is not a cross-multiplication: these products overflow, the ratios do not. *)
  let big = Int.max_value / 3 in
  p "ratio: survives operands whose cross-products would overflow"
    (Memory_budget.compare_relief_ratio big 2 big 3 > 0
    && Memory_budget.compare_relief_ratio big 3 big 2 < 0)

(* The subject: a 4-hidden-layer MLP whose whole training step is one routine. Depth matters — the
   backprop chain's activation gradients are what the liveness planner staggers and what the budget
   selector then demotes. *)
let build () =
  Utils.settings.fixed_state_for_init <- Some 3;
  Tensor.unsafe_reinitialize ();
  TDSL.default_param_init := NTDSL.xavier ~scale_sq:2.0 TDSL.O.uniform1;
  let ctx = Context.auto () in
  let mem0 = Context.get_used_memory ctx in
  let batch = 32 and d_in = 64 and d_hid = 256 in
  let xs =
    NTDSL.init ~l:"xs" ~prec:Ir.Ops.single ~b:[ batch ] ~o:[ d_in ]
      ~f:(fun idcs -> Float.sin (Float.of_int ((idcs.(0) * d_in) + idcs.(1))))
      ()
  in
  let ys =
    NTDSL.init ~l:"ys" ~prec:Ir.Ops.single ~b:[ batch ] ~o:[ 1 ]
      ~f:(fun idcs -> Float.cos (Float.of_int idcs.(0)))
      ()
  in
  let%op mlp x =
    { w4 }
    * relu
        ({ b3; o = [ d_hid ] }
        + { w3 }
          * relu ({ b2; o = [ d_hid ] } + ({ w2 } * relu ({ b1; o = [ d_hid ] } + ({ w1 } * x)))))
  in
  let%op err = mlp xs - ys in
  let%op scalar_loss = ((err *. err) ++ "...|... => |->0") /. !..batch in
  let update = Train.grad_update scalar_loss in
  let sgd = Train.sgd_update ~learning_rate:(TDSL.O.( !. ) 1e-6) scalar_loss in
  let comp = Asgns.sequence [ update; sgd ] in
  let ctx = Train.init_params ctx IDX.empty scalar_loss in
  (ctx, comp, scalar_loss, mem0)

(* Compile the step under [budget] and run six of them, returning the loss trajectory, the step's
   actual device-footprint delta, and the plan that shipped (if any). *)
let train_phase ?budget ~label () =
  let ctx, comp, scalar_loss, mem0 = build () in
  let plan = ref None in
  let ctx, routine =
    Train.to_routine ctx ?budget ~budget_report:(fun r -> plan := Some r) IDX.empty comp
  in
  let open Operation.At in
  let losses = ref [] in
  for _ = 1 to 6 do
    Train.run ctx routine;
    losses := (ctx, scalar_loss).@[0] :: !losses
  done;
  let losses = List.rev !losses in
  let mem_delta = Context.get_used_memory ctx - mem0 in
  eprintf "%s: losses %s; step footprint: %d bytes\n%!" label
    (String.concat ~sep:", " (List.map losses ~f:(Printf.sprintf "%.6f")))
    mem_delta;
  (losses, mem_delta, !plan)

(* Layer 0: the planner's precondition. Deliberately BEFORE the environment enables
   [buffer_aliasing], so the failure is the one a user gets by setting a budget alone. *)
let () =
  let ctx, comp, _loss, _mem0 = build () in
  match Memory_budget.fit ~budget:(Memory_budget.Bytes 1024) ctx comp IDX.empty with
  | _ -> p "budget without buffer_aliasing raises" false
  | exception Utils.User_error msg ->
      p "budget without buffer_aliasing raises"
        (String.is_substring msg ~substring:"buffer_aliasing")

let () = Unix.putenv "OCANNL_BUFFER_ALIASING" "true"

(* Layers 1 and 2: calibrate, then pin the selector's contract against the two calibrated ends. *)
let baseline, minimum, mid, minimize_flips =
  let ctx, comp, _loss, _mem0 = build () in
  let base = Memory_budget.footprint ctx comp IDX.empty in
  let _ctx, min_plan = Memory_budget.fit ~budget:Memory_budget.Minimize ctx comp IDX.empty in
  let baseline = base.fp_total and minimum = min_plan.bp_final.fp_total in
  eprintf
    "calibration: baseline %d bytes, minimum %d bytes, %d flip(s), dedicated %d, planned %d\n%!"
    baseline minimum (List.length min_plan.bp_flips) base.fp_dedicated base.fp_planned;
  p "calibration: the baseline is the plan's own baseline" (baseline = min_plan.bp_baseline.fp_total);
  p "minimize: relieves footprint" (minimum < baseline);
  p "minimize: takes at least one flip" (not (List.is_empty min_plan.bp_flips));
  p "minimize: reports itself within budget (it has no target)" min_plan.bp_within_budget;
  (* The reported reliefs must account for the whole move: a flip committed as part of a joint group
     carries 0 and the group's relief lands on the flip that closed it, so the SUM is the invariant
     to assert, not per-flip positivity. *)
  p "minimize: the reported reliefs sum to the footprint moved"
    (List.fold min_plan.bp_flips ~init:0 ~f:(fun a (_, relief, _) -> a + relief)
    = baseline - minimum);
  p_all "minimize: no flip reports negative relief" min_plan.bp_flips ~f:(fun (_, relief, _) ->
      relief >= 0);
  (* Strictly between the two ends, so it is reachable but not free. The flips minimize took are the
     population a budgeted plan chooses from: what the baseline plan declines is drawn from them. *)
  (baseline, minimum, (baseline + minimum) / 2, min_plan.bp_flips)

let () =
  let ctx, comp, _loss, _mem0 = build () in
  let plan budget = snd (Memory_budget.fit ~budget ctx comp IDX.empty) in
  let at_baseline = plan (Memory_budget.Bytes baseline) in
  p_empty "budget at the baseline: no flips" ~over:minimize_flips at_baseline.bp_flips;
  p "budget at the baseline: within budget" at_baseline.bp_within_budget;
  let at_mid = plan (Memory_budget.Bytes mid) in
  p "reachable budget: met" at_mid.bp_within_budget;
  (* The footprint assertion: the layout the plan reports is actually under the budget. *)
  p "reachable budget: the scored footprint is under it" (at_mid.bp_final.fp_total <= mid);
  p "reachable budget: it took flips" (not (List.is_empty at_mid.bp_flips));
  (* Acceptance stops as soon as the budget is met, so a loose-enough budget must not go all the way
     to the minimum. *)
  p "reachable budget: stops at the budget rather than minimizing"
    (at_mid.bp_final.fp_total >= minimum);
  let unreachable = plan (Memory_budget.Bytes 1) in
  p "unreachable budget: reports NOT within budget" (not unreachable.bp_within_budget);
  p "unreachable budget: still relieves what it can" (unreachable.bp_final.fp_total = minimum);
  p "monotone: a tighter budget never scores larger"
    (unreachable.bp_final.fp_total <= at_mid.bp_final.fp_total
    && at_mid.bp_final.fp_total <= at_baseline.bp_final.fp_total);
  (* Deterministic, not a search: the same inputs pick the same flips, in the same order, with the
     same reported relief. *)
  let again = plan (Memory_budget.Bytes mid) in
  p "deterministic: replanning picks the same flips"
    (List.equal
       (fun (a, ra, ca) (b, rb, cb) -> Tn.equal a b && ra = rb && ca = cb)
       at_mid.bp_flips again.bp_flips)

(* Layers 3 and 4: executed parity across the gate, and default-off. *)
let () =
  let losses_off, mem_off, plan_off = train_phase ~label:"budget off" () in
  p "default-off: nothing is planned" (Option.is_none plan_off);
  let losses_on, mem_on, plan_on =
    train_phase ~budget:(Memory_budget.Bytes mid) ~label:"budget on" ()
  in
  (match plan_on with
  | None -> p "budget on: a plan shipped" false
  | Some plan ->
      p "budget on: a plan shipped" true;
      p "budget on: it demoted intermediates to recompute-at-use"
        (not (List.is_empty plan.bp_flips));
      p "budget on: the scored footprint is under the budget" (plan.bp_final.fp_total <= mid));
  Verdict.pass_fail "train: loss trajectory parity across the budget gate"
    (List.equal Float.equal losses_off losses_on);
  p "train: loss decreased" Float.(List.last_exn losses_off < List.hd_exn losses_off);
  (* The modelled relief has to show up as real device memory: the demoted nodes lose their
     buffers. *)
  p "train: actual device footprint reduced" (mem_on < mem_off)
