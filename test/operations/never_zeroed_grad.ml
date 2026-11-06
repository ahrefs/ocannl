(* gh-ocannl-1781: a gradient that accumulates across runs with no reset. A routine whose
   computation accumulates into a gradient without zeroing it -- a loss over a parameter whose
   first-constructed consumer carried the zeroing (gh-ocannl-1744), or a [Train.grad_update
   ~accum_steps] micro-step -- sums every run's gradient into the buffer until something resets it.
   [Context.run] refuses such a routine once it has run as many times as its computation allows
   (one, or [k] under [~accum_steps:k]) since the buffer's last reset: a run of a routine zeroing
   the gradient, or a write from outside a run. The refusal names the gradient and what the lineage
   knows about its reset; a refused run dispatches nothing, so the values are unchanged.

   [w = [1; 2; 3]] under [l = w *. w], so one step's gradient is [2w = [2; 4; 6]], and a micro-step
   of two under [~accum_steps:2] contributes [w]. Every value is a small integer, compared
   bitwise. *)
open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims
module LC = Lineage_claims
module IDX = Train.IDX

let one t = Set.singleton (module Tensor) t
let compile ctx comp = Train.to_routine ctx IDX.empty comp
let run_all ctx routines = List.fold routines ~init:ctx ~f:(fun ctx r -> Context.run ctx r)
let zeros = [| 0.; 0.; 0. |]
let two_w = [| 2.; 4.; 6. |]
let gh = "gh-ocannl-1781"
let param prefix = TDSL.param ~values:[| 1.; 2.; 3. |] (prefix ^ "_w") ~output_dims:[ 3 ] ()

(* A first consumer of [w], built and evaluated forward-only: it carries [w]'s zeroing, so a later
   loss over [w] accumulates into [w.grad] without zeroing it. The case holds it past that loss's
   construction: a reclaimed consumer would give the zeroing back (gh-ocannl-1804). *)
let forward_only_consumer ctx w =
  let%op c = w *. 2. in
  (Train.forward_once ctx c, c)

let hold (c : Tensor.t) = ignore (Stdlib.Sys.opaque_identity c : Tensor.t)
let names_grad msg w = LC.has msg (Ir.Tnode.label_name (LC.grad_of w))

let () =
  case "never zeroed anywhere in the lineage" (fun () ->
      let w = param "nz" in
      let ctx, c = forward_only_consumer (Context.auto ()) w in
      let%op l = w *. w in
      hold c;
      let update = Train.grad_update l in
      p "the update accumulates into w's gradient without zeroing it"
        (Set.mem update.accumulates (LC.grad_of w) && not (Set.mem update.zeroes (LC.grad_of w)));
      (* The seed puts the gradient in the context, past the link's "lacks node" hint. *)
      let ctx = Context.set_values ctx (LC.grad_of w) zeros in
      let ctx, step = compile ctx update in
      let ctx = Context.run ctx step in
      LC.exact "one run: w.grad = 2w" (Context.get_values ctx (LC.grad_of w)) two_w;
      let msg = LC.refusal "a second run" (fun () -> Context.run ctx step) in
      p "the second run is refused, naming the gradient, the routine and the missing reset"
        (names_grad msg w && LC.has msg step.name && LC.has msg gh
        && LC.has msg "no routine of this lineage zeroes it"
        && LC.has msg "Train.zero_params_grads"
        && LC.has msg "Train.grad_update ~accum_steps:k");
      LC.exact "the refusal dispatched nothing: w.grad = 2w"
        (Context.get_values ctx (LC.grad_of w))
        two_w;
      ignore (LC.refusal "a second run from a discarded context" (fun () -> Context.run ctx step));
      (* Uploads of other nodes are no reset of this one. *)
      let ctx = Context.set_values ctx w.value [| 1.; 2.; 3. |] in
      ignore (LC.refusal "after an upload of the parameter" (fun () -> Context.run ctx step));
      (* A write from outside a run is a reset. *)
      let cycle ctx =
        let ctx = Context.set_values ctx (LC.grad_of w) zeros in
        Context.run ctx step
      in
      match LC.accepted "an upload of the gradient resets it" (fun () -> cycle ctx) with
      | None -> ()
      | Some ctx ->
          LC.exact "reset by upload, run: w.grad = 2w" (Context.get_values ctx (LC.grad_of w)) two_w;
          let ctx = cycle ctx in
          LC.exact "a second such cycle: the same" (Context.get_values ctx (LC.grad_of w)) two_w);
  case "a zeroing routine compiled and not run" (fun () ->
      let w = param "nr" in
      let ctx, c = forward_only_consumer (Context.auto ()) w in
      let%op l = w *. w in
      hold c;
      let ctx, zero = compile ctx (Train.zero_params_grads ~params:(one w) l) in
      let ctx, step = compile ctx (Train.grad_update l) in
      let ctx = run_all ctx [ zero; step; zero; step ] in
      LC.exact "reset before each step: w.grad = 2w" (Context.get_values ctx (LC.grad_of w)) two_w;
      let msg = LC.refusal "a step without the reset" (fun () -> Context.run ctx step) in
      p "the refusal names the lineage's zeroing routine, which has not run since"
        (names_grad msg w && LC.has msg zero.name && LC.has msg "has not run since");
      let ctx = run_all ctx [ zero; step ] in
      LC.exact "the reset, then the step: w.grad = 2w" (Context.get_values ctx (LC.grad_of w)) two_w);
  case "the accum_steps opt-in" (fun () ->
      let w = param "acc" in
      let%op l = w *. w in
      let micro = Train.grad_update ~accum_steps:2 l in
      p "a micro-step states its span for the leaf gradient"
        (Option.equal Int.equal (Map.find micro.accumulation_runs (LC.grad_of w)) (Some 2));
      p "a plain update states none" (Map.is_empty (Train.grad_update l).accumulation_runs);
      let stating_nothing =
        Ir.Assignments.comp
          ~accumulates:(Set.singleton (module Ir.Tnode) (LC.grad_of w))
          ~embedded:(Set.empty (module Ir.Tnode))
          Ir.Assignments.Noop
      in
      p "sequenced with an accumulator stating nothing, the span is its implicit one"
        (Option.equal Int.equal
           (Map.find (Ir.Assignments.sequence [ micro; stating_nothing ]).accumulation_runs
              (LC.grad_of w))
           (Some 1));
      let ctx = Train.init_params (Context.auto ()) IDX.empty l in
      let ctx, zero = compile ctx (Train.zero_params_grads l) in
      let ctx, m = compile ctx micro in
      let ctx = run_all ctx [ zero; m; m ] in
      LC.exact "a cycle of two micro-steps: w.grad = 2w"
        (Context.get_values ctx (LC.grad_of w))
        two_w;
      let msg = LC.refusal "a third micro-step" (fun () -> Context.run ctx m) in
      p "the third micro-step is refused, naming the declared span"
        (names_grad msg w && LC.has msg "2 run(s) of this routine since its last reset, 2 allowed");
      let ctx = run_all ctx [ zero; m; m ] in
      LC.exact "the next cycle: w.grad = 2w" (Context.get_values ctx (LC.grad_of w)) two_w;
      (* No zeroing routine at all: the micro-steps still take their span, over an uploaded seed. *)
      let w = param "acc2" in
      let%op l = w *. w in
      let micro = Train.grad_update ~accum_steps:2 l in
      let ctx = Train.init_params (Context.auto ()) IDX.empty l in
      let ctx, m = compile ctx micro in
      let ctx = Context.set_values ctx (LC.grad_of w) zeros in
      let ctx = run_all ctx [ m; m ] in
      LC.exact "two micro-steps over the seed: w.grad = 2w"
        (Context.get_values ctx (LC.grad_of w))
        two_w;
      let msg = LC.refusal "a third micro-step, no zeroing routine" (fun () -> Context.run ctx m) in
      p "refused, and no routine of the lineage zeroes it"
        (names_grad msg w && LC.has msg "no routine of this lineage zeroes it"));
  case "timing runs and rollback" (fun () ->
      let w = param "tm" in
      let%op l = w *. w in
      let ctx = Train.init_params (Context.auto ()) IDX.empty l in
      let ctx, zero = compile ctx (Train.zero_params_grads l) in
      let ctx, m = compile ctx (Train.grad_update ~accum_steps:1 l) in
      let ctx = Context.run ctx zero in
      ignore
        (LC.accepted "the autotuner's timing loop repeats an accumulating routine" (fun () ->
             Autotune.time_routine ~timing:Autotune.Isolated ~repeats:2 ctx m));
      ignore (LC.refusal "a run after the timing runs" (fun () -> Context.run ctx m));
      let ctx = run_all ctx [ zero; m ] in
      LC.exact "the reset, then the step: w.grad = 2w" (Context.get_values ctx (LC.grad_of w)) two_w;
      (* A run withdrawn as known not to have written takes its count with it. *)
      Context.rollback_execution ctx m.routine_id;
      ignore (LC.accepted "a run after a withdrawn one" (fun () -> Context.run ctx m)));
  case "a search's state restore is not a reset" (fun () ->
      let w = param "sr" in
      let%op l = w *. w in
      let micro = Train.grad_update ~accum_steps:1 l in
      let ctx = Train.init_params (Context.auto ()) IDX.empty l in
      let ctx_z, zero = compile ctx (Train.zero_params_grads l) in
      let ctx, m = compile ctx_z micro in
      let ctx = run_all ctx [ zero; m ] in
      (* A search of the same update as a sibling of [m]: it binds the same gradient buffer, its
         timing runs write it, and it restores it through uploads (gh-ocannl-1621). *)
      ignore
        (Autotune.with_uncontended_test_windows (fun () ->
             Autotune.tune ~search:true ~beam_width:1 ~rounds:0 ~repeats:1 ~timing:Autotune.Isolated
               ~cache_dir:"" ctx_z micro IDX.empty));
      LC.exact "the search restored the gradient: w.grad = 2w"
        (Context.get_values ctx (LC.grad_of w))
        two_w;
      ignore
        (LC.refusal "the step again after the search, without the reset" (fun () ->
             Context.run ctx m)))
