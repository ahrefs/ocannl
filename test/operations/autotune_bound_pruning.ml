(* gh-ocannl-514 phase 4b: bound pruning against the measured incumbent in [Autotune.tune], config
   [autotune_bound_pruning]. The dune rule pins the cc backend and a tiny envelope (model_peak_* =
   1e3), making the schedule-invariant roofline floor astronomically larger than any measured time —
   every enumerative sketch candidate is then provably unable to win and is pruned before compile,
   deterministically on any machine, while the baseline and presets are exempt and still timed. The
   tuned routine must still compute correct values (the winner comes from the never-pruned
   candidates). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Asgns = Ir.Assignments
open Verdict.Claims

(* The report's outcome as the questions this test asks of it (gh-ocannl-677): the outcome is a
   variant naming one of five mutually exclusive states, so a claim names the state it means instead
   of combining flags — [not (replayed r)] in particular does NOT say a search ran. *)
let completed (r : Autotune.report) =
  match r.Autotune.outcome with Autotune.Searched -> true | _ -> false

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let () =
  let nn = 64 in
  let av =
    Array.init (nn * nn)
      ~f:(Ll_test.cycle_flat ~dims:[| nn; nn |] ~modulus:13 ~offset:0. ~stride:0.25)
  in
  let bv =
    Array.init (nn * nn)
      ~f:(Ll_test.cycle_flat ~dims:[| nn; nn |] ~modulus:7 ~offset:(-3.) ~stride:0.5)
  in
  let a = TDSL.ndarray av ~label:[ "a" ] ~input_dims:[ nn ] ~output_dims:[ nn ] () in
  let b = TDSL.ndarray bv ~label:[ "b" ] ~input_dims:[ nn ] ~output_dims:[ nn ] () in
  (* Unscheduled twin for the values check. *)
  let%op c0 = a * b in
  let ctx0, r0 =
    Context.compile (Context.auto ()) (named "bp_serial" (Train.forward c0)) Ir.Indexing.Empty
  in
  let ctx0 = Context.run ctx0 r0 in
  let want = Context.get_values ctx0 c0.Tensor.value in
  (* The tuned run, bound pruning on (via the rule's command line). Fresh-search behavior is the
     assertion, so a cache entry left by an earlier run of this binary in the same build tree must
     not replay — clear the cache directory first. *)
  let cache_dir = "autotune_cache_bound_pruning" in
  if Stdlib.Sys.file_exists cache_dir then
    Array.iter (Stdlib.Sys.readdir cache_dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat cache_dir f));
  let%op c1 = a * b in
  let reports = ref [] in
  let ctx, routine =
    Autotune.tune ~rounds:0 ~repeats:1 ~cache_dir
      ~report:(fun r -> reports := r :: !reports)
      (Context.auto ())
      (named "bp_tuned" (Train.forward c1))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx c1.Tensor.value in
  p_close "tuned values match the serial twin" got want ~tol:(fun _ -> 1e-4);
  match !reports with
  | [ r ] ->
      p "the search completed" (completed r);
      p "bound pruning fathomed sketch candidates" (r.Autotune.bound_pruned > 0);
      p "never-pruned candidates were still timed" (r.Autotune.candidates_timed >= 1);
      p "keep-fraction pruning is counted apart" (r.Autotune.model_pruned = 0)
  | _ -> Stdio.printf "expected exactly one report\n"
