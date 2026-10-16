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
   below hold on every backend.

   The scalar block-tiling twins (gh-ocannl-1325) are pinned harder, on every backend: EVERY one
   applies, validates, fuses (one top-level statement fewer than its unfused sibling) and matches
   the untuned reference -- before, [Fuse_epilogue] refused every GPU one. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Stdio
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
open Verdict.Claims

(* The backend's accumulator residency, which a [Privatize] tile is minted at (gh-ocannl-1116). *)
let accum_prec =
  let caps = lazy (Context.codegen_capabilities (Context.auto ())) in
  fun p -> (Lazy.force caps).Ir.Backend_intf.accum_prec p

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
  (* Vacuous only where the backend advertises no tensorized pipeline for this site's precisions:
     CUDA/HIP need the tf32 arm (config [tf32_matmuls]) for an f32 site, Metal's simdgroup matrices
     take f32 directly, and cc packs its own tile. The gate is that CAPABILITY, derived from what
     the backend advertises ([Ll_test.tensorized_matmul_capability]) and never from the seeds this
     test asserts the existence of (gh-ocannl-1115): keyed on the seeds, even on the mma family as a
     whole, a seeding regression on a backend that does advertise a tile (Metal at f32, CUDA under
     tf32) would read as a vacuous skip. Where the capability holds, the family and its epilogue
     twins are required. *)
  let capability =
    Ll_test.tensorized_matmul_capability ~is_gpu ~is_cpu ~limits ~a:ma.Tensor.value
      ~b:mb.Tensor.value ~d:prod.Tensor.value
  in
  let mma_gated label b =
    match capability with
    | `Advertised -> p label b
    | `Withheld aggregation -> Verdict.skipped ~aggregation ~backend:backend_name label
  in
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
        match Sched.apply (Autotune.sketch_schedule ~accum_prec ~p:q scratch) scratch with
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
  let n_ran = ref 0 and n_correct = ref 0 and n_rendered = ref 0 in
  List.iter ep_params ~f:(fun q ->
      match
        let ctx = Context.auto () in
        let ctx, routine =
          Context.compile
            ~lowered_transform:(fun opt ->
              [ Sched.apply (Autotune.sketch_schedule ~accum_prec ~p:q opt) opt ])
            ctx fwd Ir.Indexing.Empty
        in
        let ctx = Context.run ctx routine in
        (Context.get_values ctx mc.Tensor.value, routine.Context.mma)
      with
      | got, census ->
          Int.incr n_ran;
          if Array.for_all2_exn got want ~f:approx then Int.incr n_correct;
          (* A twin whose [Tile_mma] declined to the scalar fallback runs and matches too; only the
             routine's census says it ran tensorized (gh-ocannl-1283). *)
          if Ll_test.rendered_tensorized census then Int.incr n_rendered
          else
            Stdio.eprintf "%s: an epilogue twin rendered %s\n" tag
              (Ir.C_syntax.mma_summary_string census)
      | exception _ -> ());
  mma_gated (tag ^ ": mma epilogue twins are seeded") (List.length ep_params > 0);
  mma_gated (tag ^ ": at least one epilogue twin applies and validates (reaches timing)") (n_ok >= 1);
  mma_gated (tag ^ ": some epilogue twin compiles and runs end-to-end") (!n_ran >= 1);
  mma_gated
    (tag ^ ": every running epilogue twin renders tensorized per its mma census")
    (!n_ran >= 1 && !n_rendered = !n_ran);
  p (tag ^ ": every running twin matches the untuned reference") (!n_ran = !n_correct);
  (* gh-ocannl-1325: the scalar block-tiling twins, of the backend's own family. [Fuse_epilogue]
     used to refuse every GPU one, on every GPU backend: the [Privatize] store-back sits under the
     register split's [Workgroup] loops ("output loop ... must be Serial or Grid"), and a padded
     geometry's store-back under its pad masks ("guarded writes ... unsupported") -- five wasted
     compiles per segment per search, and no compilable sketch at all for a segment whose unfused
     family is refuted on companion coverage where no tensorized pipeline exists for its triple. *)
  let bt_params =
    List.filter (Autotune.sketch_seed_params ~is_gpu ~is_cpu ~limits opt) ~f:(fun q ->
        (not q.Autotune.sk_mma) && q.Autotune.sk_epilogue)
  in
  let bt_label what = tag ^ ": every blocktile epilogue twin " ^ what in
  let real_stmts (o : LL.optimized) =
    List.count (LL.flat_lines [ o.LL.llc ]) ~f:(function
      | LL.Noop | LL.Comment _ -> false
      | _ -> true)
  in
  let scheduled q =
    let scratch =
      {
        opt with
        LL.traced_store = Hashtbl.copy opt.LL.traced_store;
        LL.optimize_ctx = LL.copy_optimize_ctx opt.LL.optimize_ctx;
      }
    in
    match Sched.apply (Autotune.sketch_schedule ~accum_prec ~p:q scratch) scratch with
    | post -> (
        match LL.validate_parallel post.LL.optimize_ctx.placements post.LL.llc with
        | () -> Ok post
        | exception Invalid_argument msg -> Error msg)
    | exception Invalid_argument msg -> Error msg
  in
  let geometry q =
    Printf.sprintf "%dx%d/%dx%d" q.Autotune.sk_bm q.Autotune.sk_bn q.Autotune.sk_tm q.Autotune.sk_tn
  in
  p (tag ^ ": blocktile epilogue twins are seeded") (not (List.is_empty bt_params));
  p_all (bt_label "applies and validates") bt_params ~f:(fun q ->
      match scheduled q with
      | Ok _ -> true
      | Error msg ->
          Stdio.eprintf "%s: blocktile twin %s refused: %s\n" tag (geometry q) msg;
          false);
  p_all (bt_label "drops the tail statement its unfused sibling keeps") bt_params ~f:(fun q ->
      match (scheduled q, scheduled { q with Autotune.sk_epilogue = false }) with
      | Ok fused, Ok unfused -> real_stmts fused + 1 = real_stmts unfused
      | _ -> false);
  p_all (bt_label "compiles, runs and matches the untuned reference") bt_params ~f:(fun q ->
      match
        let ctx = Context.auto () in
        let ctx, routine =
          Context.compile
            ~lowered_transform:(fun opt ->
              [ Sched.apply (Autotune.sketch_schedule ~accum_prec ~p:q opt) opt ])
            ctx fwd Ir.Indexing.Empty
        in
        Context.get_values (Context.run ctx routine) mc.Tensor.value
      with
      | got -> Array.for_all2_exn got want ~f:approx
      | exception exn ->
          Stdio.eprintf "%s: blocktile twin %s FAILED: %s\n" tag (geometry q) (Exn.to_string exn);
          false)

let () =
  (* Dividing extents: the whole-K [Tile_mma] site (unstaged and single-full-K-stage pipelines). *)
  census "div" ~m:32 ~n:32 ~k:32;
  (* Non-multiples of the intrinsic tiles: the pad-composition pipelines (gh-485), whose fragment
     store-back carries range masks — the guarded-writes rejection before the fix. *)
  census "nondiv" ~m:24 ~n:40 ~k:56;
  printf "\nDone.\n%!"
