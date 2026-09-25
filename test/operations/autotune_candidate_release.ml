(* gh-ocannl-550, accumulation half: a schedule search's live artifacts must be bounded by the beam,
   not by candidates processed.

   The failure this pins needs no GPU even though its consequence does. On CUDA the accumulation
   exhausted a 12 GB card a fifth of the way through a cold tf32 gpt2_mini search
   (benchmarks/report-gh550-cuda.md); on [cc] the very same artifacts accumulate — one context and
   one working pool per candidate — and {!Ir.Alloc_census} counts them, so the growth is
   host-observable here while the out-of-memory is not.

   The observation point is [Autotune.on_candidate_attempt], which fires BEFORE a candidate's
   compile and therefore after the previous candidate has been released: exactly "between
   candidates". What is asserted is the shape of the sequence, not an absolute number — pools per
   candidate and contexts per compile are implementation details, but "the k-th sample does not
   exceed the first sample by more than the beam holds" is the property gh-ocannl-550 is about. The
   census deltas are taken against the first sample so that the caller's own contexts and the
   reference compile do not enter the count.

   The two sequences this separates, measured on cc (beam width 2): released, the live working-pool
   delta flattens at the beam's steady state (0,1,2,2,2,2,2,3,4,4,4,... over an 18-candidate
   search); not released, it reads 0,1,2,3,...,17 — exactly one per candidate, to the end.

   WORKING pools specifically, and the size below is chosen so that the distinction is not academic.
   At n = 128 the search seeds hoisted [Stage] candidates (the [hoist] labels) wherever that family
   exists — a CPU one: [Autotune.matmul_seed_params] proposes [sk_hoist] only in its [is_cpu]
   branch, so on a GPU backend the constant class stays empty and the last assertion below reads the
   other side of its equivalence. Each application of that transform mints a FRESH packed-constant
   tnode, which [allocate_delta] routes into the per-device constant pool and registers in
   [constant_buffer_cache] under a unique key. Context release skips every key in that cache —
   deliberately, since constants are shared per device and outlive contexts — so the constant class
   still grows one pool per hoisted candidate: measured 1 -> 109 constant pools (0.1 -> 15.1 MiB)
   over 181 candidates, while working pools stayed within 2-6. Bounding that needs a per-pool purity
   rule in the shared constant cache (a pool mixing a private packed tile with genuinely shared
   weights must not be freed), which is eviction-policy work and belongs to gh-ocannl-565. So this
   test asserts the class gh-ocannl-550 bounds and does not pretend about the other one: asserting
   the constant class "bounded" would be a false green, and asserting its current growth would pin a
   bug. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let n = 128
let beam_width = 2

let clean_cache dir =
  if Stdlib.Sys.file_exists dir && Stdlib.Sys.is_directory dir then
    Array.iter (Stdlib.Sys.readdir dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat dir f))

(* Samples of the census taken between candidates, oldest first, paired with the label of the
   candidate each sample precedes. *)
let sample_between_candidates f =
  let samples = ref [] in
  (Autotune.on_candidate_attempt :=
     fun label -> samples := (label, Ir.Alloc_census.snapshot ()) :: !samples);
  let result = Exn.protect ~f ~finally:(fun () -> Autotune.on_candidate_attempt := fun _ -> ()) in
  (result, List.rev !samples)

let () =
  let mav =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:7 ~offset:0. ~stride:0.5)
  in
  let mbv =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:5 ~offset:(-2.) ~stride:1.)
  in
  let ma = TDSL.ndarray mav ~label:[ "acr_ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "acr_mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op mc = ma * mb in
  let%op t2 = relu mc in
  ignore mc;
  let comp = Train.forward t2 in

  (* Reference values from a plain compile, so the release cannot be "correct" by having broken the
     search: a tuner that freed the winner's buffers would still satisfy every count below. *)
  let ctx_ref, routine_ref = Context.compile (Context.auto ()) comp Ir.Indexing.Empty in
  let ctx_ref = Context.run ctx_ref routine_ref in
  let expected = Context.get_values ctx_ref t2.Tensor.value in

  let cache_dir = "autotune_cache_candidate_release" in
  clean_cache cache_dir;
  let (ctx_t, routine_t), attempts =
    sample_between_candidates (fun () ->
        Autotune.tune ~beam_width ~rounds:1 ~repeats:1 ~cache_dir (Context.auto ()) comp
          Ir.Indexing.Empty)
  in
  let labels = List.map attempts ~f:fst and samples = List.map attempts ~f:snd in
  (* A search that timed one or two candidates would satisfy every bound below vacuously. *)
  p "the search attempted enough candidates for accumulation to show" (List.length samples >= 8);

  let first = List.hd_exn samples in
  let deltas ~f = List.map samples ~f:(fun s -> f s - f first) in
  (* [live_working_pools], not [live_pools]: the latter sums in the constant class, whose growth
     under hoisted staging is a separate, unfixed matter (see the header). Summing them would fail
     this test for a reason it is not about — and, on a workload with no hoisted candidates, would
     pass while proving less than it appears to. *)
  let pool_deltas = deltas ~f:(fun c -> c.Ir.Alloc_census.live_working_pools) in
  let ctx_deltas = deltas ~f:Ir.Alloc_census.unreleased_contexts in
  let max_of = List.fold ~init:0 ~f:max in

  (* The absolute bound: one candidate in flight, plus what the beam retains, plus the running best.
     Deliberately generous — how many pools one compile needs is an implementation detail, and a
     backend needing two per candidate is not a regression of this property. *)
  let bound = 4 * (beam_width + 2) in
  p "live pools between candidates stay bounded" (max_of pool_deltas <= bound);
  p "live contexts between candidates stay bounded" (max_of ctx_deltas <= bound);

  (* And the scale-free form, which is the one that actually discriminates: the second half of the
     search must not hold measurably more than the first half did. Accumulation fails this whatever
     the per-candidate footprint is, and whatever the constant above is loosened to. *)
  let halve l =
    let k = List.length l / 2 in
    (List.take l k, List.drop l k)
  in
  let pools_early, pools_late = halve pool_deltas in
  let ctx_early, ctx_late = halve ctx_deltas in
  let slack = beam_width + 2 in
  p "live pools do not grow across the search" (max_of pools_late <= max_of pools_early + slack);
  p "live contexts do not grow across the search" (max_of ctx_late <= max_of ctx_early + slack);

  (* Freeing has to be actual freeing, not just a dropped reference: the census counts a pool freed
     only where the backend's [free_pool] ran. Phrased against contexts CREATED rather than against
     the sample count, because a candidate declining before link never creates one — with the sample
     count the threshold would silently depend on how many candidates happened to reach link. *)
  let last = List.last_exn samples in
  p "candidates were released, not merely dropped"
    (last.Ir.Alloc_census.pools_freed > 0
    && last.Ir.Alloc_census.contexts_released >= last.Ir.Alloc_census.contexts_created - bound);

  let ctx_t = Context.run ctx_t routine_t in
  let got = Context.get_values ctx_t t2.Tensor.value in
  p_all2 "the winner still ships and computes the right values" got expected ~f:(fun a b ->
      Float.(abs (a - b) < 1e-4));

  (* And the exit sweep: with the search over and its routine in hand, nothing else it compiled is
     still held. Read against the state before the search's first candidate, so the reference
     compile's own footprint does not enter. *)
  let after = Ir.Alloc_census.snapshot () in
  p "the search holds nothing beyond its winner once it has returned"
    (after.Ir.Alloc_census.live_working_pools - first.Ir.Alloc_census.live_working_pools <= bound);

  (* Hoisting coverage is a precondition, not a decoration: without hoisted candidates the
     assertions above hold for a workload that never exercised the working/constant split, and this
     test would be quietly weaker than it reads.

     Whether this workload HAS hoisted candidates is a backend fact, not a workload fact:
     [Autotune.matmul_seed_params] proposes [sk_hoist] only from its [is_cpu] branch — link-time
     packing of a constant operand into a panel is a CPU packing family — so on a GPU backend no
     candidate mints a packed-constant tnode and the constant class cannot grow. Asserting growth
     outright is therefore false on Metal/CUDA, and dropping the assertion would hide the vacuity on
     every backend at once. Asserted as the equivalence instead: the constant class grows exactly
     where the search attempted hoisted candidates. On cc that reads "44 hoist labels, +50 constant
     pools" and pins the split as before; on Metal it reads "no hoist labels, no constant growth"
     and pins that the working-pool bounds above were measured on a search with nothing in the other
     class — which is what makes them the whole story there. Either way a search that stopped
     minting packed constants while still seeding hoisted candidates fails it. *)
  let hoisted_attempted = List.exists labels ~f:(String.is_substring ~substring:" hoist") in
  let constant_growth =
    after.Ir.Alloc_census.constant_pools_allocated - first.Ir.Alloc_census.constant_pools_allocated
  in
  p "the constant class grows exactly where the search seeded hoisted staging"
    (Bool.equal hoisted_attempted (constant_growth > 4))
