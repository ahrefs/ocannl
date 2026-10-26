(* gh-ocannl-1645: a training step compiles again in the lineage that already ran it.

   The step is SGD over [y = w * x] with [loss = sum (y *. y)], so [y.grad = 2 y] depends on [w]: a
   stale or uninitialized [y.grad] gives a different update. The first compile commits [w.grad]
   virtual and stores its template, which reads [y.grad] while [y.grad] is still undecided; the
   compile's finalization then places [y.grad] as routine-local scratch ([Local]). A second compile
   of the same step writes [y.grad] before [w.grad] reads it, so the read is covered, but gh-1620's
   rule refused it because [w.grad] cannot fall back to materialization: [Utils.User_error] "... was
   computed as routine-local scratch by an earlier routine ...". The fix admits a covered scratch
   read in an already-virtual node's re-stored template; the template's replays are judged where
   they land, covered or refused, in every routine's emitted code (gh-ocannl-1646).

   Executed parity: the lineage is held, step by step, to a sibling lineage that materializes
   [y.grad] and [w.grad] ([Context.decide_materialized]). Every value is exact by construction --
   [x] holds small integers, [w] eighths, the learning rate is a power of two, and three steps stay
   well inside single precision's mantissa -- so the two agree bitwise whatever schedule computed
   them. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module IDX = Train.IDX

let n = 4

let () =
  Tensor.unsafe_reinitialize ();
  let x =
    TDSL.ndarray
      (Array.init (n * n)
         ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:5 ~offset:(-2.) ~stride:1.))
      ~input_dims:[ n ] ~output_dims:[ n ] ()
  in
  let w_init =
    Array.init (n * n)
      ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:9 ~offset:(-4.) ~stride:0.125)
  in
  let w = TDSL.param ~values:w_init "w" ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op y = w * x in
  let%op loss = (y *. y) ++ "...|...->... => 0" in
  let%op learning_rate = 0.0078125 in
  Train.set_materialized w.Tensor.value;
  Train.set_materialized learning_rate.Tensor.value;
  let comp =
    Ir.Assignments.sequence [ Train.grad_update loss; Train.sgd_update ~learning_rate loss ]
  in
  let grad t = (Option.value_exn ~here:[%here] t.Tensor.diff).grad in
  let y_grad = grad y and w_grad = grad w in
  let read ctx = Context.get_values ctx w.Tensor.value in
  (* One step: [Error] when the compile is refused (the base behaviour under test). *)
  let step ctx =
    match Context.compile ctx comp IDX.empty with
    | ctx, routine ->
        let ctx = Context.run ctx routine in
        Ok (ctx, read ctx)
    | exception Utils.User_error msg -> Error msg
  in
  let fresh () = Train.init_params (Context.auto ()) IDX.empty loss in
  let lineage = fresh () in
  Stdio.eprintf "backend: %s (not part of the golden)\n%!" (Context.backend_name lineage);
  let reference = Context.decide_materialized (fresh ()) [ y_grad; w_grad ] in
  let snapshot = read lineage in
  let rec steps k ~before ctx_l ctx_r =
    if k <= 3 then
      let label = Printf.sprintf "step %d" k in
      match (step ctx_l, step ctx_r) with
      | Ok (ctx_l, got), Ok (ctx_r, expected) ->
          p (label ^ ": compiles in the lineage that ran the earlier steps") true;
          if k = 1 then (
            let plc = Context.placements ctx_l and plc_r = Context.placements ctx_r in
            (* The shape under test: without these, the later steps would pass vacuously. *)
            (* The shape is the CPU backends' placement of this step; a GPU backend's default
               schedule may place [y.grad] otherwise (Metal does at the larger size of
               [autotune_timing_state_restore]), where the parity claims still execute. *)
            let backend = Context.backend_name ctx_l in
            let on_cpu = Ir.Schedule.backend_is_cpu backend in
            gated ~when_:on_cpu ~on:backend "step 1: w.grad is committed virtual"
              (Ir.Tnode.Placements.known_virtual plc w_grad);
            gated ~when_:on_cpu ~on:backend "step 1: y.grad is routine-local scratch"
              (Ir.Tnode.Placements.known_non_virtual plc y_grad
              && Ir.Tnode.Placements.known_not_materialized plc y_grad);
            p_all "step 1: the reference materializes y.grad and w.grad" [ y_grad; w_grad ]
              ~f:(fun tn ->
                Ir.Tnode.Placements.known_non_virtual plc_r tn
                && not (Ir.Tnode.Placements.known_not_materialized plc_r tn)));
          p (label ^ ": changes the parameter") (not (Poly.equal before got));
          p_all2
            (label ^ ": the parameter agrees bitwise with the materialized lineage")
            got expected ~f:Float.equal;
          steps (k + 1) ~before:got ctx_l ctx_r
      | Error msg, _ ->
          p (label ^ ": compiles in the lineage that ran the earlier steps (" ^ msg ^ ")") false
      | _, Error msg -> p (label ^ ": the materialized reference compiles (" ^ msg ^ ")") false
  in
  steps 1 ~before:snapshot lineage reference
