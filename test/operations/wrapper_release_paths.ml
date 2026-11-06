(* gh-ocannl-1859: a wrapper that mints a context returns it, hands it to what it returns, or
   releases it, whichever way the call ends. Every case below opens a root, runs one wrapper, and
   asserts that once the caller released what it holds, the process's census is back where it was:
   contexts, working pools and constant pools, exactly. A raising case also asserts that its fault
   fired, that the caller sees that fault and no other, and that the caller's own context still
   compiles and runs. The normal cases run what they got first.

   Faults go through [Ir.Resource_fault_injection], in the order the wrapper reaches them: in
   [Train.run_once], the initialization's compile and run come first, so the second link and the
   second launch are the ones of the routine itself. *)
open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims
module IDX = Train.IDX
module FI = Ir.Resource_fault_injection
module Census = Ir.Alloc_census

let () = Unix.putenv "OCANNL_BUFFER_ALIASING" "true"
let fault = "gh1859 injected failure"

type reading = { contexts : int; working : int; constant : int } [@@deriving equal, sexp_of]

let read () =
  let c = Census.snapshot () in
  {
    contexts = Census.unreleased_contexts c;
    working = c.live_working_pools;
    constant = c.live_constant_pools;
  }

let n = 4
let value i j = Float.of_int ((i + j) % 7)

(* [t = w *. 2.] over a host-backed parameter, as the wrappers' callers build it. *)
let graph =
  let serial = ref 0 in
  fun () ->
    Int.incr serial;
    let i = !serial in
    let w =
      TDSL.param
        ~values:(Array.init n ~f:(value i))
        (Printf.sprintf "wrp_w%d" i) ~output_dims:[ n ] ()
    in
    let%op t = w *. 2. in
    (t, Array.init n ~f:(fun j -> 2. *. value i j))

let computes ctx ((t : Tensor.t), expected) =
  Array.equal Float.equal (Context.get_values ctx t.value) expected

(* The caller's context, after whatever happened: it compiles and runs a new graph. *)
let still_usable root =
  let g = graph () in
  let ctx = Train.forward_once root (fst g) in
  let usable = computes ctx g in
  Context.release ctx;
  usable

let back_at name before =
  pf "%s: nothing the call minted is left" name (equal_reading (read ()) before)

let returned name f =
  let before = read () in
  let root = Context.auto () in
  let ctx, computed = f root in
  p (name ^ ": what it returned computes") computed;
  Context.release ctx;
  p (name ^ ": the caller's context still works") (still_usable root);
  Context.release root;
  back_at name before

(* [f] with [point]'s [at]-th hit raising the fault; [at = 0] raises at every hit. *)
let raising ?(point = FI.Run_before_launch) ?(at = 1) name f =
  let before = read () in
  let root = Context.auto () in
  let hits = ref 0 in
  let seen =
    match
      FI.with_callback
        (fun seen ->
          if FI.equal_point seen point then (
            Int.incr hits;
            if at = 0 || !hits = at then failwith fault))
        ~f:(fun () -> f root)
    with
    | (_ : Context.t) -> None
    | exception exn -> Some (Exn.to_string exn)
  in
  p (name ^ ": the fault fired") (!hits >= max at 1);
  p
    (name ^ ": the caller sees the fault")
    (Option.exists seen ~f:(String.is_substring ~substring:fault));
  p (name ^ ": the caller's context still works") (still_usable root);
  Context.release root;
  back_at name before

let budget = Memory_budget.Bytes (1 lsl 40)
let refusing_report (_ : Memory_budget.plan) = failwith fault

let () =
  returned "forward_once" (fun root ->
      let g = graph () in
      let ctx = Train.forward_once root (fst g) in
      (ctx, computes ctx g));
  returned "to_routine under a budget" (fun root ->
      let t, expected = graph () in
      let ctx = Train.init_params root IDX.empty t in
      let compiled, routine = Train.to_routine ctx ~budget IDX.empty (Train.forward t) in
      let compiled = Context.run compiled routine in
      let computed = computes compiled (t, expected) in
      Context.release compiled;
      (ctx, computed));
  raising "run_once, its run" ~at:2 (fun root -> Train.forward_once root (fst (graph ())));
  raising "run_once, its initialization's run" ~at:1 (fun root ->
      Train.forward_once root (fst (graph ())));
  raising "run_once, its initialization's upload" ~point:From_host_before_copy (fun root ->
      Train.forward_once ~reinit_all:true root (fst (graph ())));
  raising "run_once, its compile" ~point:Link_after_delta ~at:2 (fun root ->
      Train.forward_once root (fst (graph ())));
  raising "run_once, its compile under a budget" ~point:Link_after_delta ~at:2 (fun root ->
      Train.forward_once ~budget root (fst (graph ())));
  raising "init_params, its run" (fun root -> Train.init_params root IDX.empty (fst (graph ())));
  (* The report's own failure: the fault is the report's, so no injection point is involved. *)
  let report_case name f =
    let before = read () in
    let root = Context.auto () in
    let seen =
      match f root with (_ : Context.t) -> None | exception exn -> Some (Exn.to_string exn)
    in
    p
      (name ^ ": the caller sees the fault")
      (Option.exists seen ~f:(String.is_substring ~substring:fault));
    p (name ^ ": the caller's context still works") (still_usable root);
    Context.release root;
    back_at name before
  in
  report_case "run_once, its budget report" (fun root ->
      Train.forward_once ~budget ~budget_report:refusing_report root (fst (graph ())));
  report_case "to_routine, its budget report" (fun root ->
      let t, _ = graph () in
      let ctx = Train.init_params root IDX.empty t in
      Exn.protect
        ~f:(fun () ->
          fst
            (Train.to_routine ctx ~budget ~budget_report:refusing_report IDX.empty (Train.forward t)))
        ~finally:(fun () -> Context.release ctx))

(* A failed run with the caller's own initialized context: it stays the caller's. *)
let () =
  let before = read () in
  let root = Context.auto () in
  let t, expected = graph () in
  let ctx = Train.init_params root IDX.empty t in
  let raised =
    match
      FI.with_callback
        (fun seen -> if FI.equal_point seen Run_before_launch then failwith fault)
        ~f:(fun () -> Train.forward_once ~skip_init:true ctx t)
    with
    | (_ : Context.t) -> false
    | exception Failure m -> String.is_substring m ~substring:fault
  in
  p "skip_init: the fault reached the caller" raised;
  p "skip_init: the caller's initialized context still reads its parameter"
    (let w = Set.choose_exn t.params in
     Array.equal Float.equal (Context.get_values ctx w.value)
       (Array.map expected ~f:(fun v -> v /. 2.)));
  Context.release ctx;
  Context.release root;
  back_at "skip_init" before

(* {1 decide_readings} *)

let () =
  let before = read () in
  let root = Context.auto () in
  let a = (fst (graph ())).value and b = (fst (graph ())).value and c = (fst (graph ())).value in
  let contexts = (read ()).contexts in
  let decided = Context.decide_readings root ([ a ], [ b ], [ c ]) in
  p "decide_readings mints one context for three readings" ((read ()).contexts = contexts + 1);
  p "and adds one lifecycle level"
    (Context.lifecycle_depth decided = Context.lifecycle_depth root + 1);
  p "all three are decided in it" (Context.is_lifecycle_ancestor decided ~ancestor:root);
  p "nothing to decide mints nothing"
    (Context.same_release_unit root (Context.decide_readings root ([], [], [])));
  Context.release decided;
  Context.release root;
  back_at "decide_readings" before

(* {1 Train.tune_placements} *)

let side = 8

(* Materializing the intermediate is a placement decision, and relu reads it once. *)
let tunable =
  let serial = ref 0 in
  fun () ->
    Int.incr serial;
    let i = !serial in
    let square label f =
      TDSL.ndarray
        (Array.init (side * side) ~f)
        ~label:[ label ] ~input_dims:[ side ] ~output_dims:[ side ] ()
    in
    let ma = square "wrp_a" (fun j -> Float.of_int ((j + i) % 7)) in
    let mb = square "wrp_b" (fun j -> Float.of_int ((j + (2 * i)) % 5)) in
    let%op mc = ma * mb in
    let%op t = relu mc in
    ignore mc;
    (t, Train.forward t)

let tune ?(cache_dir = "") ?report ?flip_report ?inline_flips ?ship_arm ?timing_ctx ?on_ship root t
    comp =
  Train.tune_placements ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir ?report ?flip_report
    ?inline_flips ?ship_arm ?timing_ctx ?on_ship root t comp Ir.Indexing.Empty

(* Every cell of the [side]-by-[side] result, whatever the winning placement. *)
let all_finite ctx (t : Tensor.t) =
  Array.count (Context.get_values ctx t.value) ~f:Float.is_finite = side * side

let tune_returned name f =
  returned name (fun root ->
      let (t : Tensor.t), comp = tunable () in
      let ctx, routine = f root t comp in
      let ctx = Context.run ctx routine in
      (ctx, all_finite ctx t))

let cache_dir = "autotune_cache_wrapper_release_paths"

let () =
  if Stdlib.Sys.file_exists cache_dir then
    Array.iter (Stdlib.Sys.readdir cache_dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat cache_dir f));
  tune_returned "tune_placements, the measured winner" (fun root t comp -> tune root t comp);
  tune_returned "tune_placements, arm A forced" (fun root t comp ->
      tune ~ship_arm:Train.Force_arm_a root t comp);
  tune_returned "tune_placements, arm B forced" (fun root t comp ->
      tune ~ship_arm:Train.Force_arm_b root t comp);
  let flips = ref 0 in
  tune_returned "tune_placements, with flips" (fun root t comp ->
      tune ~inline_flips:2 ~flip_report:(fun _ -> Int.incr flips) root t comp);
  p "the flip refinement measured a flip" (!flips > 0);
  (* The caller's timing context stays the caller's. *)
  returned "tune_placements, in a timing lineage" (fun root ->
      let (t : Tensor.t), comp = tunable () in
      let timing_ctx = Context.auto () in
      let ctx, routine = tune ~timing_ctx ~inline_flips:2 root t comp in
      let ctx = Context.run ctx routine in
      let computed = all_finite ctx t in
      p "the timing context stays usable" (still_usable timing_ctx);
      Context.release timing_ctx;
      (ctx, computed));
  (* A second call over one program replays the first's recorded decision. *)
  let t, comp = tunable () in
  let outcomes = ref [] in
  let report r = outcomes := r.Autotune.outcome :: !outcomes in
  returned "tune_placements, recording a decision" (fun root ->
      let ctx, routine = tune ~cache_dir ~report root t (Ll_test.unowned_copy comp) in
      (ctx, all_finite (Context.run ctx routine) t));
  outcomes := [];
  returned "tune_placements, replaying it" (fun root ->
      let ctx, routine = tune ~cache_dir ~report root t (Ll_test.unowned_copy comp) in
      (ctx, all_finite (Context.run ctx routine) t));
  p "the second call replayed a search"
    (List.exists !outcomes ~f:(function Autotune.Cache_replay -> true | _ -> false));
  let tune_raising name f =
    let before = read () in
    let root = Context.auto () in
    let (t : Tensor.t), comp = tunable () in
    let seen = match f root t comp with _ -> None | exception exn -> Some (Exn.to_string exn) in
    p
      (name ^ ": the caller sees the fault")
      (Option.exists seen ~f:(String.is_substring ~substring:fault));
    p (name ^ ": the caller's context still works") (still_usable root);
    Context.release root;
    back_at name before
  in
  tune_raising "tune_placements, on_ship" (fun root t comp ->
      tune ~on_ship:(fun _ -> failwith fault) root t comp);
  tune_raising "tune_placements, on_ship of arm B" (fun root t comp ->
      tune ~ship_arm:Train.Force_arm_b ~on_ship:(fun _ -> failwith fault) root t comp);
  (* Arm B's report: its decided context exists by then. *)
  tune_raising "tune_placements, report" (fun root t comp ->
      let reports = ref 0 in
      tune
        ~report:(fun _ ->
          Int.incr reports;
          if !reports = 2 then failwith fault)
        root t comp);
  tune_raising "tune_placements, every arm failing" (fun root t comp ->
      FI.with_callback
        (fun seen -> if FI.equal_point seen Link_after_delta then failwith fault)
        ~f:(fun () -> tune root t comp))
