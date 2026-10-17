(* gh-ocannl-550 and gh-ocannl-1326, accumulation half: a schedule search must not keep the buffers
   of the candidates it has timed.

   The failure this pins needs no GPU even though its consequence does. On CUDA the accumulation
   exhausted a 12 GB card a fifth of the way through a cold tf32 gpt2_mini search
   (benchmarks/report-gh550-cuda.md); on [cc] the very same artifacts accumulate — one context and
   one working pool per candidate — and {!Ir.Alloc_census} counts them, so the growth is
   host-observable here while the out-of-memory is not. gh-ocannl-550 bounded the growth at the beam
   (plus the running best and the candidate in flight); gh-ocannl-1326 found that bound still too
   high for a step that uses over half the device — a b256 gpt2_mini search on a 12 GiB card timed
   one candidate and declined every later one at link, because the incumbent's 6.6 GiB stayed
   resident — and the tuner now releases every candidate as soon as its timing window is over,
   ranking incumbents by their saved schedules and recompiling the winner at the end.

   The observation point is [Autotune.on_candidate_attempt], which fires BEFORE a candidate's
   compile: exactly "between candidates". Its first firing is the base compile's, before anything of
   the search is allocated, and the census deltas are taken against it, so the caller's own contexts
   and the reference compile do not enter the count. What is asserted is therefore exact: at every
   later firing, no working pool and no context the search compiled is still live. Measured on cc
   (beam width 2, over a ~190-candidate search): all zero released at the timing; the gh-ocannl-550
   beam bound read 0,1,2,2,2,2,2,3,4,4,4,... and no release at all 0,1,2,3,...,17, one per
   candidate.

   WORKING pools specifically, and the size below is chosen so that the distinction is not academic.
   At n = 128 the search seeds hoisted [Stage] candidates (the [hoist] labels) wherever that family
   exists — a CPU one: [Autotune.matmul_seed_params] proposes [sk_hoist] only in its [is_cpu]
   branch, so on a GPU backend the constant class stays empty and the last assertion below reads the
   other side of its equivalence. Each application of that transform mints a FRESH packed-constant
   tnode, which [allocate_delta] routes into the per-device constant pool and registers in
   [constant_buffer_cache] under a unique key. Context release skips every key in that cache —
   deliberately, since constants are shared per device and outlive contexts — so the constant class
   still grows one pool per hoisted candidate: measured 1 -> 109 constant pools (0.1 -> 15.1 MiB)
   over 181 candidates. Bounding that needs a per-pool purity rule in the shared constant cache (a
   pool mixing a private packed tile with genuinely shared weights must not be freed), which is
   eviction-policy work and belongs to gh-ocannl-565. So this test asserts the class the tuner
   releases and does not pretend about the other one: asserting the constant class released would be
   a false green, and asserting its current growth would pin a bug. *)

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
  let result =
    Autotune.Seam.with_value Autotune.on_candidate_attempt
      (fun label -> samples := (label, Ir.Alloc_census.snapshot ()) :: !samples)
      f
  in
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
  (* A search that timed one or two candidates would satisfy every claim below vacuously. *)
  p "the search attempted enough candidates for accumulation to show" (List.length samples >= 8);

  let first = List.hd_exn samples in
  let deltas ~f = List.map samples ~f:(fun s -> f s - f first) in
  (* [live_working_pools], not [live_pools]: the latter sums in the constant class, whose growth
     under hoisted staging is a separate, unfixed matter (see the header). Summing them would fail
     this test for a reason it is not about — and, on a workload with no hoisted candidates, would
     pass while proving less than it appears to. *)
  let pool_deltas = deltas ~f:(fun c -> c.Ir.Alloc_census.live_working_pools) in
  let ctx_deltas = deltas ~f:Ir.Alloc_census.unreleased_contexts in
  p_all "no timed candidate's working pools are live when the next one compiles" pool_deltas
    ~f:(fun d -> d = 0);
  p_all "no candidate's context is left unreleased when the next one compiles" ctx_deltas
    ~f:(fun d -> d = 0);

  (* Freeing has to be actual freeing, not just a dropped reference: the census counts a pool freed
     only where the backend's [free_pool] ran. *)
  let last = List.last_exn samples in
  p "candidates were released, not merely dropped"
    (last.Ir.Alloc_census.pools_freed > first.Ir.Alloc_census.pools_freed);

  let ctx_t = Context.run ctx_t routine_t in
  let got = Context.get_values ctx_t t2.Tensor.value in
  p_all2 "the winner still ships and computes the right values" got expected ~f:(fun a b ->
      Float.(abs (a - b) < 1e-4));

  (* And the exit: with the search over and its routine in hand, nothing else it compiled is still
     held — the returned routine is one context, whose pools are the only live ones the call added.
     Read against the state before the search's first candidate, so the reference compile's own
     footprint does not enter. *)
  let after = Ir.Alloc_census.snapshot () in
  p "the search holds nothing beyond its winner once it has returned"
    (Ir.Alloc_census.unreleased_contexts after - Ir.Alloc_census.unreleased_contexts first = 1);

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
