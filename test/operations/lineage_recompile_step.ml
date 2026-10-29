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
  (* gh-ocannl-1724: a lineage refuses a tensor fragment compiled twice. This test's recompiles in
     one lineage exercise lowering the same step again (gh-ocannl-1645/1653), which an unowned copy
     of the program -- the same assignments, no fragment metadata -- still does. *)
  let comp =
    Ir.Assignments.comp ~embedded:comp.Ir.Assignments.embedded_nodes comp.Ir.Assignments.asgns
  in
  let grad t = (Option.value_exn ~here:[%here] t.Tensor.diff).grad in
  let y_grad = grad y and w_grad = grad w in
  let read ctx = Context.get_values ctx w.Tensor.value in
  (* gh-ocannl-1653: what a recompile costs the lineage -- the stored template components every
     later read replays, and the statements of the routine the compile emits. *)
  let code_size (routine : Context.routine) =
    List.sum
      (module Int)
      routine.Context.segments
      ~f:(fun o -> Ll_test.count_stmt ~in_scopes:true ~f:(fun _ -> true) o.Ir.Low_level.llc)
  in
  let sizes = ref [] in
  (* One step: [Error] when the compile is refused (the base behaviour under test). *)
  let step ctx =
    match Context.compile ctx comp IDX.empty with
    | ctx, routine ->
        sizes := code_size routine :: !sizes;
        let ctx = Context.run ctx routine in
        Ok (ctx, read ctx)
    | exception Utils.User_error msg -> Error msg
  in
  let totals = ref [] in
  let w_grad_components = ref 0 in
  let last = ref None in
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
          totals := Context.stored_component_total ctx_l :: !totals;
          Stdio.eprintf
            "%s: %d stored components (w.grad: %d), routine of %d statements (not part of the \
             golden)\n\
             %!"
            label (List.hd_exn !totals)
            (Context.stored_components ctx_l w_grad)
            (List.nth_exn !sizes 1);
          if k = 1 then w_grad_components := Context.stored_components ctx_l w_grad;
          last := Some ctx_l;
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
  steps 1 ~before:snapshot lineage reference;
  (* gh-ocannl-1653: a recompile of the same step replaces the template components its re-trace
     supersedes, so neither the lineage's stored templates nor the recompiled routine grow with the
     number of recompiles. Before, every recompile prepended one more copy of each re-stored
     component, which every later read -- the recompiled routine's own reads included --
     replayed. *)
  let totals = List.rev !totals in
  (let backend = Context.backend_name lineage in
   gated
     ~when_:(Ir.Schedule.backend_is_cpu backend)
     ~on:backend "after step 1, the lineage stores w.grad's template" (!w_grad_components > 0));
  (match totals with
  | first :: rest ->
      p_all "every later step's lineage stores as many template components as step 1's" rest
        ~f:(fun t -> t = first)
  | [] -> p "the steps ran" false);
  (* The lineage's own sizes, newest first, interleaved with the reference's: keep the lineage's. *)
  let lineage_sizes = List.rev !sizes |> List.filteri ~f:(fun i _ -> i % 2 = 0) in
  (match lineage_sizes with
  | _ :: second :: rest ->
      p_all "every recompile after the first emits a routine as large as the second" rest
        ~f:(fun n -> n = second)
  | _ -> p "the steps ran" false);
  (* A longer chain, compile-only (nothing runs, so no value claim needs a tolerance): eight more
     recompiles of the step in the lineage, each from the previous one's context. *)
  match (!last, totals) with
  | Some ctx, first :: _ ->
      let chain =
        List.folding_map (List.range 0 8) ~init:ctx ~f:(fun ctx _ ->
            let ctx, routine = Context.compile ctx comp IDX.empty in
            (ctx, (Context.stored_component_total ctx, code_size routine)))
      in
      Stdio.eprintf
        "compile-only chain (stored components, statements): %s (not part of the golden)\n%!"
        (String.concat ~sep:" " (List.map chain ~f:(fun (t, n) -> Printf.sprintf "(%d, %d)" t n)));
      p_all "eight more recompiles store as many template components as step 1" chain
        ~f:(fun (t, _) -> t = first);
      p_all "eight more recompiles emit a routine as large as the second step's" chain
        ~f:(fun (_, n) -> n = List.nth_exn lineage_sizes 1)
  | _ -> p "the steps ran" false
