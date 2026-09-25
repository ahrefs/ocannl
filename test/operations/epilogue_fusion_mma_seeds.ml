(* The gh-ocannl-521 route-1 regression: the GPU mma sketch seeds' fused-epilogue twins must survive
   candidate compile. The seeds annotate the GEMM nest and leave companion nests (relu, bias)
   uncovered; on GPU there is no all-serial fallback, so the non-EP variants fail
   [validate_parallel] on the companion write and the one-companion seed only survives through its
   [Fuse_epilogue] twin. Before this fix the twin's survival path was itself broken two ways —
   "guarded writes of the reduction output are unsupported" (the gh-485 pad masks on a non-dividing
   site's fragment store-back) and "the accumulator is a whole-K Tile_mma target" (the unstaged and
   single-full-K-stage pipelines) — so every mma-labelled candidate failed before being timed, on
   every GPU backend, and no test asserted more than seeding.

   Pinned here, against the REAL seed enumeration ([Autotune.sketch_seed_params] /
   [sketch_schedule]) on the current backend, for a dividing and a non-dividing matmul+bias+relu
   graph: epilogue twins are seeded; at least one applies and passes [validate_parallel] (i.e. would
   reach timing); and every twin that compiles end-to-end computes values matching the untuned
   reference. On cc a few cache-blocked packed twins are correctly rejected (their accumulator is
   genuinely partial per k-block — fusing after it would read partial accumulations); the booleans
   below hold on every backend. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Stdio
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
open Verdict.Claims

(* Zeros compare equal to zeros. A fragment mapping that reads outside the staged block, a kernel
   that never ran, or a reference whose own setup silently collapsed all yield all-zeros, and a
   parity check between two zero arrays passes while covering nothing (gh-ocannl-481 item 3). Every
   reference array is pinned nonzero where it is produced, so the parity claims below have
   content. *)
let nonzero name (a : float array) =
  if not (Array.exists a ~f:(fun x -> Float.(x <> 0.))) then
    failwith (name ^ ": the reference is all zeros — the parity checks against it are vacuous");
  a

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let is_gpu = Sched.backend_is_gpu backend_name
let is_cpu = Sched.backend_is_cpu backend_name
let approx a b = Float.(abs (a - b) < 1e-3 *. (1. +. abs b))

let census tag ~m ~n ~k =
  Tensor.unsafe_reinitialize ();
  let mav =
    Array.init (m * k) ~f:(Ll_test.cycle_flat ~dims:[| m; k |] ~modulus:13 ~offset:0. ~stride:0.25)
  in
  let mbv =
    Array.init (k * n) ~f:(Ll_test.cycle_flat ~dims:[| k; n |] ~modulus:17 ~offset:(-8.) ~stride:1.)
  in
  let bv = Array.init m ~f:(fun i -> Float.of_int (i % 5) -. 2.) in
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ k ] ~output_dims:[ m ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ n ] ~output_dims:[ k ] () in
  let bias = TDSL.ndarray bv ~label:[ "bias" ] ~output_dims:[ m ] () in
  let%op prod = ma * mb in
  let%op mc = relu (prod + bias) in
  Train.set_materialized prod.Tensor.value;
  let ctx = Context.auto () in
  let limits = Context.hardware_limits ctx in
  let fwd = named ("ep521_" ^ tag) (Train.forward mc) in
  let capture = ref None in
  let _ctx, _r =
    Context.compile
      ~lowered_transform:(fun opt ->
        capture := Some opt;
        [ opt ])
      ctx fwd Ir.Indexing.Empty
  in
  let opt = Option.value_exn ~here:[%here] !capture in
  let mma_params =
    List.filter (Autotune.sketch_seed_params ~is_gpu ~is_cpu ~limits opt) ~f:(fun q ->
        q.Autotune.sk_mma)
  in
  let ep_params = List.filter mma_params ~f:(fun q -> q.Autotune.sk_epilogue) in
  (* Vacuous only where the backend advertises no mma format tile for this site's precision at all:
     CUDA/HIP need the tf32 arm (config [tf32_matmuls]) for an f32 site, Metal's simdgroup matrices
     take f32 directly, and cc packs its own tile. Say so on stderr — as [autotune_mma_companion]
     does — so a vacuous pass is not mistaken for coverage, and keep the golden backend-independent
     rather than asserting a bare [true] that no CUDA run at default config can satisfy.

     Deliberately keyed on the mma family as a whole, NOT on [ep_params]: the epilogue twins are
     what this test asserts the existence of, so excusing their absence by their own absence would
     make every assertion below self-satisfying and would mask the regression on a backend that does
     advertise a tile (Metal at f32, CUDA under tf32). Where the family is seeded, the twins are
     required. *)
  let vacuous = is_gpu && List.is_empty mma_params in
  if vacuous then
    Stdio.eprintf "%s: no GPU mma seed for this site on %s — the checks below are vacuous\n" tag
      backend_name;
  (* Structural leg: the twin applies and its scheduled form passes [validate_parallel] — the only
     remaining gate before a tuner would time it is the backend compile itself. *)
  let n_ok =
    List.count ep_params ~f:(fun q ->
        let scratch =
          {
            opt with
            LL.traced_store = Hashtbl.copy opt.LL.traced_store;
            LL.optimize_ctx = LL.copy_optimize_ctx opt.LL.optimize_ctx;
          }
        in
        match Sched.apply (Autotune.sketch_schedule ~p:q scratch) scratch with
        | post -> (
            match LL.validate_parallel post.LL.optimize_ctx.placements post.LL.llc with
            | () -> true
            | exception Invalid_argument _ -> false)
        | exception Invalid_argument _ -> false)
  in
  (* Executed leg: every twin that compiles end-to-end matches the untuned reference. *)
  let rctx = Context.auto () in
  let rctx, rroutine = Context.compile rctx fwd Ir.Indexing.Empty in
  let rctx = Context.run rctx rroutine in
  let want = nonzero (tag ^ "_ref") (Context.get_values rctx mc.Tensor.value) in
  let n_ran = ref 0 and n_correct = ref 0 in
  List.iter ep_params ~f:(fun q ->
      match
        let ctx = Context.auto () in
        let ctx, routine =
          Context.compile
            ~lowered_transform:(fun opt -> [ Sched.apply (Autotune.sketch_schedule ~p:q opt) opt ])
            ctx fwd Ir.Indexing.Empty
        in
        let ctx = Context.run ctx routine in
        Context.get_values ctx mc.Tensor.value
      with
      | got ->
          Int.incr n_ran;
          if Array.for_all2_exn got want ~f:approx then Int.incr n_correct
      | exception _ -> ());
  p (tag ^ ": mma epilogue twins are seeded") (vacuous || List.length ep_params > 0);
  p
    (tag ^ ": at least one epilogue twin applies and validates (reaches timing)")
    (vacuous || n_ok >= 1);
  p (tag ^ ": some epilogue twin compiles and runs end-to-end") (vacuous || !n_ran >= 1);
  p (tag ^ ": every running twin matches the untuned reference") (!n_ran = !n_correct)

let () =
  (* Dividing extents: the whole-K [Tile_mma] site (unstaged and single-full-K-stage pipelines). *)
  census "div" ~m:32 ~n:32 ~k:32;
  (* Non-multiples of the intrinsic tiles: the pad-composition pipelines (gh-485), whose fragment
     store-back carries range masks — the guarded-writes rejection before the fix. *)
  census "nondiv" ~m:24 ~n:40 ~k:56;
  printf "\nDone.\n%!"
