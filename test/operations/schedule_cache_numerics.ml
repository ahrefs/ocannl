(* gh-ocannl-568: the schedule disk cache never replays a winner across numerics policies.

   The numerics policy ([Ir.Numerics.t]: tf32 matmuls, narrow compute width, native fp16 arithmetic)
   is not a property of the lowered code — it is consulted at codegen and by the autotuner's
   tile-shape choice — so two runs differing only in it lower to byte-identical code with an equal
   canonical digest. Before the fix the disk-cache key was that digest plus the backend name, so a
   default-flags run sharing a cache directory with a tf32-tuned search reported an ordinary cache
   hit and replayed the tf32-tuned tensorized schedule, whose mma rendering degrades to the scalar
   fallback under the stricter numerics: measured at 5.9x slower than not tuning at all. It also
   breaks [Ir.Numerics]'s invariant that the policy is identical across sibling candidates.

   Printed booleans, all backend-independent (the policy flips are no-ops for the cc backend's
   codegen — what is asserted here is cache identity, which must separate the regimes on every
   backend, conservatively, rather than only where the flag currently bites):

   - The cache key is a function of the policy: equal within a regime, different across. - A
   search's entry replays under the policy that wrote it, and does NOT under another. - The two
   entries coexist: crossing back still hits, i.e. the second regime's search did not overwrite the
   first regime's winner. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Sched = Ir.Schedule
module SC = Ir.Schedule_cache

let timing_identity =
  Some
    {
      Ir.Backend_intf.device_signature = "synthetic-device";
      toolchain_signature = "synthetic-compiler";
    }

module Numerics = Ir.Numerics
module Asgns = Ir.Assignments
open Verdict.Claims

(* The report's outcome as the questions this test asks of it (gh-ocannl-677): the outcome is a
   variant naming one of five mutually exclusive states, so a claim names the state it means instead
   of combining flags — [not (replayed r)] in particular does NOT say a search ran. *)
let replayed (r : Autotune.report) =
  match r.Autotune.outcome with Autotune.Cache_replay -> true | _ -> false

let completed (r : Autotune.report) =
  match r.Autotune.outcome with Autotune.Searched -> true | _ -> false

let approx a b = Float.(abs (a - b) < 1e-4)

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let n = 16

let mav =
  Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:7 ~offset:0. ~stride:0.5)

let mbv =
  Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:11 ~offset:(-4.) ~stride:1.)

let mm_expected =
  Array.init (n * n) ~f:(fun idx ->
      let i = idx / n and j = idx % n in
      let acc = ref 0. in
      for k = 0 to n - 1 do
        acc := !acc +. (mav.((i * n) + k) *. mbv.((k * n) + j))
      done;
      !acc)

(* The dune sandbox persists across runs; a stale entry would turn the miss assertions vacuous. *)
let cache_dir = "autotune_cache_numerics"

let clean_cache () =
  if Stdlib.Sys.file_exists cache_dir && Stdlib.Sys.is_directory cache_dir then
    Array.iter (Stdlib.Sys.readdir cache_dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat cache_dir f))

let entry_count () =
  if Stdlib.Sys.file_exists cache_dir && Stdlib.Sys.is_directory cache_dir then
    Array.count (Stdlib.Sys.readdir cache_dir) ~f:(fun f -> String.is_suffix f ~suffix:".sexp")
  else 0

let () =
  clean_cache ();
  let ma = TDSL.ndarray mav ~label:[ "scn_ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "scn_mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op mc = ma * mb in
  let tune_comp = named "scn_matmul" (Train.forward mc) in
  let ctx = Context.auto () in
  let backend = Context.backend_name ctx in
  let cache_available = Option.is_some (Context.timing_identity ctx) in

  (* --- The key is a function of the policy --- *)
  let base = Numerics.get () in
  let canon = ref None in
  let _ctx, _routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        canon := Some (SC.canonicalize ~static_indices:[] opt);
        [ opt ])
      ctx tune_comp Ir.Indexing.Empty
  in
  let canon = Option.value_exn ~here:[%here] !canon in
  let key_of policy =
    Numerics.set_policy policy;
    Option.value_exn
      (SC.cache_key ~timing_identity ~limits:(Context.hardware_limits ctx) canon ~backend)
  in
  let policy_a = { base with Numerics.tf32_matmuls = false } in
  let policy_b = { base with Numerics.tf32_matmuls = true } in
  let key_a = key_of policy_a and key_b = key_of policy_b in
  p "the cache key is stable within one numerics regime" (String.equal key_a (key_of policy_a));
  p "the cache key separates the tf32 regime from the default one" (not (String.equal key_a key_b));
  (* Every field participates, not just the one this issue was reported against: the record is
     open-ended by design, so the tag is derived from its sexp rather than field by field. *)
  p "a policy difference in any other field separates the key too"
    (not (String.equal key_a (key_of { policy_a with Numerics.narrow_compute_f32 = false })));
  p "the key still carries the source digest and the backend"
    (String.is_prefix key_a ~prefix:(SC.digest canon ^ "-" ^ backend));

  (* --- Regime A: a search from cold, then a replay --- *)
  let tune () =
    let report = ref None in
    let ctx = Context.auto () in
    let ctx, routine =
      Autotune.tune ~beam_width:1 ~rounds:1 ~repeats:1 ~cache_dir
        ~report:(fun r -> report := Some r)
        ctx tune_comp Ir.Indexing.Empty
    in
    let ctx = Context.run ctx routine in
    let got = Context.get_values ctx mc.Tensor.value in
    (Option.value_exn ~here:[%here] !report, got)
  in
  Numerics.set_policy policy_a;
  let r, got = tune () in
  p "the cold call ran a search rather than replaying" (completed r);
  p_all2 "the cold search computes correct values" got mm_expected ~f:approx;
  (* A search that timed nothing (gh-ocannl-532), or refused any window for contention
     (gh-ocannl-855), stores nothing, so the replay assertions are conditioned on an entry actually
     having been written. *)
  let stored_a = entry_count () = 1 in
  p "the search stores exactly when it timed a complete candidate set"
    (Bool.equal stored_a
       (cache_available && r.Autotune.candidates_timed > 0 && r.Autotune.timings_contended = 0));
  let r_a2, got = tune () in
  p "the entry replays under the policy that wrote it" (Bool.equal (replayed r_a2) stored_a);
  p_all2 "the replayed routine computes correct values" got mm_expected ~f:approx;
  let stored_a_after_retry = cache_available && (stored_a || r_a2.Autotune.timings_contended = 0) in

  (* --- Regime B: the same code, the same directory, the other policy --- *)
  Numerics.set_policy policy_b;
  let r, got = tune () in
  p "the entry does NOT replay under a different policy (gh-ocannl-568)" (not (replayed r));
  p "the cross-policy run searches instead of replaying"
    ((not stored_a) || r.Autotune.candidates_timed >= 1);
  p_all2 "the cross-policy run computes correct values" got mm_expected ~f:approx;

  (* --- The regimes coexist rather than overwriting each other --- *)
  Numerics.set_policy policy_a;
  let r, _got = tune () in
  p "regime B's search did not overwrite regime A's winner"
    (Bool.equal (replayed r) stored_a_after_retry);
  Numerics.set_policy base
