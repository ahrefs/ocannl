(* gh-ocannl-1006: the device-footprint high-water mark, and why it is a counter rather than a
   reading.

   The benchmark report's memory column has to say what a workload's steps cost on the device. Every
   backend already exposes [Context.get_used_memory], and none of them can answer that question: it
   is a CURRENT gauge, so a value read once the timed window has closed describes whatever is still
   held at that moment — and on [metal] and [cc] the bytes come back from a GC finalizer, so the
   same run reports different numbers depending on when a collection happened to run.
   {!Ir.Alloc_census.reset_peak} and [peak_pool_bytes] are the counter beside it: raised at the
   allocation site, never lowered by a free, rebased only where a caller brackets a window.

   Two halves. The first drives the real allocator through [Context] on whatever backend is
   configured, which is what pins the WIRING — the shared seam feeding the mark — and it is where
   the mark and the live bytes are made to disagree: after a release the live bytes fall and the
   mark does not. The second half drives {!Ir.Alloc_census.record_pool} / [forget_pool] directly on
   a device id no backend has, which is what pins the ARITHMETIC exactly: a re-recorded pool
   replaces its own bytes rather than adding to them, a free never lowers the mark, and a rebase
   lands on the live bytes and not on zero. The claims are all backend-uniform (byte counts are
   compared with each other, never against a fixed number), so the golden is portable. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module AC = Ir.Alloc_census

let live = AC.live_pool_bytes
let peak c = c.AC.peak_pool_bytes

(* Big enough that the matmul's own buffers dominate whatever else a compile allocates, so "the mark
   rose by at least this workload's bytes" is a claim about the workload rather than about slack. *)
let n = 256
let value_bytes = n * n * 4

let () =
  (* Everything below is a difference against this rebase, so the process's own start-up footprint —
     and this test's reference compile, on a second run of the same executable — cannot be mistaken
     for the workload's. *)
  AC.reset_peak ();
  let before = AC.snapshot () in
  p "the bracket starts with the mark on the bytes already live, not on zero"
    (peak before = live before);

  let av = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 7) *. 0.5) in
  let bv = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 5) -. 2.) in
  let ma = TDSL.ndarray av ~label:[ "acp_ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray bv ~label:[ "acp_mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op mc = ma * mb in
  let ctx, routine = Context.compile (Context.auto ()) (Train.forward mc) Ir.Indexing.Empty in
  let ctx = Context.run ctx routine in
  (* Read the result, so the buffers below are demonstrably a computation's and not a compile's
     bookkeeping that a dead-code pass could have taken away. *)
  let got = Context.get_values ctx mc.Tensor.value in
  p "the workload ran and produced its output" (Array.length got = n * n);

  let during = AC.snapshot () in
  (* The three matrices are 256 KiB each; one of them is the floor, since a backend is free to
     inline, alias or pack the others (gh-ocannl-489 arenas) and none of that is a regression of
     this property. *)
  p "running the workload raises the mark by at least the bytes it computes into"
    (peak during - peak before >= value_bytes);
  p "the mark is never below what is live under it" (peak during >= live during);

  Context.release ctx;
  let after = AC.snapshot () in
  (* The contrast the column rests on, both halves read off the same seam in the same process: a
     gauge sampled here has already given the workload's bytes back, and the mark has not. *)
  p "releasing the context gives the bytes back, so the live gauge falls" (live after < live during);
  p "and the mark stays where the workload put it" (peak after = peak during);
  p "so a reading taken after the window would understate it by the bytes just freed"
    (live after < peak after);

  (* And the bracket itself: the next window starts from what is live now, which is what stops a
     cell's memory column from inheriting the high water of the search that preceded it. *)
  AC.reset_peak ();
  let rebased = AC.snapshot () in
  p "a rebase puts the mark back on the live bytes" (peak rebased = live rebased);
  p "so the window that follows does not inherit the earlier peak" (peak rebased < peak during)

(* The arithmetic, on a device id no backend mints, so these entries cannot collide with the real
   pools the half above left live. *)
let fake_device = 1_000_000

let record ~pool_id ~size_in_bytes =
  AC.record_pool ~device_id:fake_device ~pool_id ~constant:false ~size_in_bytes

let forget ~pool_id = AC.forget_pool ~device_id:fake_device ~pool_id

let () =
  AC.reset_peak ();
  let base = AC.snapshot () in
  record ~pool_id:1 ~size_in_bytes:1000;
  record ~pool_id:2 ~size_in_bytes:2000;
  let both = AC.snapshot () in
  p "the mark takes in every pool live at once, not the largest one" (peak both - peak base = 3000);

  forget ~pool_id:2;
  let one_gone = AC.snapshot () in
  p "a free lowers the live bytes" (live both - live one_gone = 2000);
  p "and leaves the mark untouched" (peak one_gone = peak both);

  (* Re-recording an existing pool is a pool GROWN IN PLACE, which the live table already models by
     replacing the entry. The mark has to follow that reading or it would count a merge slab's every
     growth as a fresh allocation and drift upward without any of it being live. *)
  record ~pool_id:1 ~size_in_bytes:1500;
  let grown = AC.snapshot () in
  p "growing a pool in place adds its growth, not its whole new size"
    (live grown - live one_gone = 500);
  p "the mark rises only where the growth passes the earlier high water" (peak grown = peak both);

  record ~pool_id:3 ~size_in_bytes:5000;
  let higher = AC.snapshot () in
  (* 1500 + 5000 live now, against the 1000 + 2000 the mark had been standing on. *)
  p "a bigger total does raise the mark" (peak higher - peak base = 6500);

  forget ~pool_id:1;
  forget ~pool_id:3;
  let emptied = AC.snapshot () in
  p "the fabricated pools are all given back" (live emptied = live base);
  AC.reset_peak ();
  let rebased = AC.snapshot () in
  p "a rebase after them lands on the live bytes, not on the high water"
    (peak rebased = live rebased && peak rebased < peak higher)
