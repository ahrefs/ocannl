(* gh-ocannl-1445: a graph-captured routine's instantiated graphs are destroyed when its context is
   released, not only when its capture cache evicts them.

   On CUDA and HIP, [sequence_segments] stream-captures a fissioned routine's segment launches into
   a graph and caches the instantiated graph per capture key (the static-index binding values). The
   cache used to be emptied only by eviction, so a released routine's graphs stayed instantiated for
   as long as its task stayed reachable, and every one still cached at exit outlived the process --
   on HIP, the [SharedSignalPool ... Signals leaked] report at exit ([exit_stream_teardown] pins
   that half). The backends count the graphs their caches hold in [live_graph_execs], in their
   global debug info; this test reads it around a routine's lifetime.

   The routine is a two-matmul chain whose materialized intermediates cut it into segments. Claimed
   where the backend reports the counter and graph capture is on: two runs at one binding
   instantiate exactly one graph (the second replays it), so the count is not vacuously flat, and
   releasing the routine's context brings it back to where it was before the compile. Elsewhere the
   claims are skipped: the CPU backends and Metal cache nothing across runs. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

(* A routine that fissions into several segments: the materialized intermediates cut the chain of
   two matmuls, so a CUDA or HIP backend graph-captures its segment launches. *)
let fissioned_chain () =
  let q = 32 in
  let matrix label ~modulus =
    TDSL.ndarray
      (Array.init (q * q)
         ~f:(Ll_test.cycle_flat ~dims:[| q; q |] ~modulus ~offset:(-2.) ~stride:1.))
      ~label:[ label ] ~input_dims:[ q ] ~output_dims:[ q ] ()
  in
  let qa = matrix "fc_a" ~modulus:11 and qb = matrix "fc_b" ~modulus:7 in
  let qc = matrix "fc_c" ~modulus:5 in
  let%op qd = qa + qb in
  Train.set_materialized qd.Tensor.value;
  let%op qe = qd * qc in
  Train.set_materialized qe.Tensor.value;
  let%op qg = qc * qe in
  Train.forward qg

(* [live_graph_execs] in the backend's global debug info, if it reports one. *)
let live_graph_execs ctx = Ll_test.debug_counter ctx "live_graph_execs"

let () =
  let comp = fissioned_chain () in
  let ctx0 = Context.auto () in
  let backend = Context.backend_name ctx0 in
  Stdio.eprintf "backend: %s (not part of the golden)\n%!" backend;
  let capture_on = Utils.get_global_flag ~default:true ~arg_name:"gpu_graph_capture" in
  let before = live_graph_execs ctx0 in
  let leg =
    Verdict.sub_leg
      (Verdict.leg ~when_:(Option.is_some before) ~on:backend ())
      ~when_:(lazy (capture_on && not (Utils.debug_log_from_routines ())))
      ~aggregation:`Environment ~on:"graph capture off (gpu_graph_capture=false or routine logging)"
  in
  let counts =
    lazy
      (let before = Option.value_exn before in
       let ctx, routine = Context.compile ctx0 comp Ir.Indexing.Empty in
       let ctx = Context.run ctx routine in
       let ctx = Context.run ctx routine in
       Context.sync ctx;
       let after_runs = Option.value (live_graph_execs ctx) ~default:(-1) in
       Context.release ctx;
       let after_release = Option.value (live_graph_execs ctx0) ~default:(-1) in
       Stdio.eprintf
         "live_graph_execs: %d before the compile, %d after two runs, %d after the release (not \
          part of the golden)\n\
          %!"
         before after_runs after_release;
       (before, after_runs, after_release))
  in
  in_leg ~leg "two runs at one binding instantiate exactly one graph" (fun label ->
      let before, after_runs, _ = Lazy.force counts in
      p label (after_runs - before = 1));
  in_leg ~leg "releasing the routine's context destroys its graphs" (fun label ->
      let before, _, after_release = Lazy.force counts in
      p label (after_release = before))
