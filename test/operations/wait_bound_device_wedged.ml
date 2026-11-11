(* gh-ocannl-2134: the production bounded wait ([Context.await_within]) gives up on work still
   running, and the device it gave up on then refuses instead of blocking.

   No kernel hangs: a finite matmul, long enough that it is still running microseconds after its
   dispatch, is waited for with a zero bound, which only completed work could meet. Asserted on
   [multidev_cc] (worker-domain devices, the generic is_idle poll): a generous bound waits it out
   ([true]); a zero bound answers [false]; afterwards [Context.sync] and [Context.release] of a
   context on that device raise [Device_wedged] at once rather than block and return. A GPU leg
   asserts the same on the device: Metal's own bounded wait, CUDA's and HIP's generic stream poll.
   The mark lasts for the process, hence this executable of its own. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let n = 1024

let comp =
  let mav =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:7 ~offset:0. ~stride:0.25)
  in
  let mbv =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:5 ~offset:(-2.) ~stride:1.)
  in
  let ma = TDSL.ndarray mav ~label:[ "wd_ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "wd_mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op mc = ma * mb in
  let%op t = mc * mb * ma in
  ignore mc;
  Ll_test.unowned_copy (Train.forward t)

type observed = {
  control : bool;  (** A generous bound's answer. *)
  expired : bool;  (** A zero bound's answer, negated: [true] when it gave up. *)
  sync_refused : bool;
  release_refused : bool;
}

let wedged = function Ir.Backend_intf.Device_wedged _ -> true | _ -> false

let exercise make_ctx =
  let ctx, routine = Context.compile (make_ctx ()) comp Ir.Indexing.Empty in
  let control =
    let ctx = Context.run ctx routine in
    Context.await_within ctx ~seconds:60.
  in
  let ctx = Context.run ctx routine in
  let started = Mtime_clock.counter () in
  let expired = not (Context.await_within ctx ~seconds:0.) in
  let sync_refused = match Context.sync ctx with () -> false | exception e -> wedged e in
  let release_refused = match Context.release ctx with () -> false | exception e -> wedged e in
  Stdio.eprintf "%s: the refusals came %.1f ms after the dispatch (not part of the golden)\n%!"
    (Context.backend_name ctx)
    (Mtime.Span.to_float_ns (Mtime_clock.count started) /. 1e6);
  { control; expired; sync_refused; release_refused }

let () =
  let o = exercise (fun () -> Context.cpu ~threads:2 ()) in
  p "multidev_cc: a generous bound waits the work out" o.control;
  p "multidev_cc: a zero bound gives up on work still running" o.expired;
  p "multidev_cc: a sync of the marked device raises Device_wedged" o.sync_refused;
  p "multidev_cc: a release on the marked device raises Device_wedged, freeing nothing"
    o.release_refused;
  let ctx = Context.auto () in
  let backend_name = Context.backend_name ctx in
  let leg =
    Verdict.leg
      ~when_:(not (Ir.Backend_intf.renders_on_host_pool (Context.backend_facts ctx)))
      ~on:backend_name ()
  in
  Verdict.in_leg ~leg
    "on a GPU device: a generous bound waits; a zero bound gives up; the marked device refuses \
     sync and release" (fun claim ->
      let o = exercise (fun () -> ctx) in
      p claim (o.control && o.expired && o.sync_refused && o.release_refused))
