(* gh-ocannl-521: an mma-labelled autotune candidate must be TIMED, not merely seeded.

   The workload is the shape lenet's classifier head has: a GEMM whose materialized output feeds an
   elementwise companion nest (bias + relu) in the SAME fission segment, the GEMM's [Zero_out]
   fissioned into its own [`Zeros] segment. On GPU backends the segment's mma seeds annotate only
   the GEMM nest, leaving the companion write uncovered by the kernel's hardware dimensions —
   [Low_level.validate_parallel] then rejects every candidate at compile, so a search could seed
   dozens of tensorized candidates and time none of them. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
open Verdict.Claims

(* The backend's accumulator residency, which a [Privatize] tile is minted at (gh-ocannl-1116). *)
let accum_prec = Ll_test.backend_accum_prec

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let is_gpu = Sched.backend_is_gpu backend_name

let clean_cache dir =
  if Stdlib.Sys.file_exists dir && Stdlib.Sys.is_directory dir then
    Array.iter (Stdlib.Sys.readdir dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat dir f))

let () =
  clean_cache "autotune_cache_mma_companion";
  let n = 64 in
  let xv =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:7 ~offset:0. ~stride:0.25)
  in
  let wv =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:5 ~offset:0. ~stride:0.125)
  in
  let bv = Array.init n ~f:(fun i -> Float.of_int (i % 3) -. 1.) in
  let expected =
    Array.init (n * n) ~f:(fun idx ->
        let i = idx / n and j = idx % n in
        let acc = ref 0. in
        for k = 0 to n - 1 do
          acc := !acc +. (xv.((i * n) + k) *. wv.((k * n) + j))
        done;
        Float.max 0. (!acc +. bv.(j)))
  in
  let x = TDSL.ndarray xv ~label:[ "mc_x" ] ~output_dims:[ n; n ] () in
  let w = TDSL.ndarray wv ~label:[ "mc_w" ] ~output_dims:[ n; n ] () in
  let bias = TDSL.ndarray bv ~label:[ "mc_b" ] ~output_dims:[ n ] () in
  let%op z = x +* "ik;kj=>ij" w in
  Train.set_materialized z.Tensor.value;
  let%op y = relu (z + bias) in
  let comp = named "mc_head" (Train.forward y) in

  (* --- Executable parity, seed by seed. This is the discriminating pin for gh-ocannl-521: these
     are the UNFUSED tensorized seeds, applied whole-routine with the Zero_out and the companion
     nest present. Before the companion nests carried the sketch's own hardware geometry, every one
     of them died at [Low_level.validate_parallel] ("write to materialized node ... not nested under
     annotated loops covering all active hardware dimensions") and only the fused twin could ever
     compile. And a candidate that compiles is not yet a candidate that computes, so the values are
     compared against the unscheduled twin rather than the schedule's shape being inspected. --- *)
  let limits = Context.hardware_limits (Context.auto ()) in
  let seeds_of opt =
    List.filter (Autotune.sketch_seed_params ~is_gpu ~is_cpu:false ~limits opt) ~f:(fun p ->
        p.Autotune.sk_mma && p.Autotune.sk_gpu && not p.Autotune.sk_epilogue)
  in
  let n_seeds =
    let captured = ref [] in
    let _ctx, _r =
      Context.compile
        ~lowered_transform:(fun opt ->
          captured := seeds_of opt;
          [ opt ])
        (Context.auto ()) comp Ir.Indexing.Empty
    in
    List.length !captured
  in
  (* Gated on the seeder's own capability judgment for the site's f32 operands and destination
     (gh-ocannl-1115), never on the seed list under test (gh-ocannl-1283): cc has no GPU tile, CUDA
     needs the tf32 arm (config [tf32_matmuls]) for an f32 site, HIP's rocWMMA has no f32-input
     shape, while Metal's simdgroup matrices take f32 directly. Where the tile is advertised, an
     empty seed list fails both claims rather than skipping them. A seed whose [Tile_mma] declined
     to the scalar fallback computes the same values, so each seed's routine census must also say it
     ran tensorized. *)
  let claim = "mc: every unfused GPU tensorized seed compiles and computes correctly" in
  let rendered_claim =
    "mc: every unfused GPU tensorized seed renders tensorized per its mma census"
  in
  (match
     Ll_test.tensorized_matmul_capability ~is_gpu ~is_cpu:false ~limits ~a:x.Tensor.value
       ~b:w.Tensor.value ~d:z.Tensor.value
   with
  | `Withheld aggregation ->
      Verdict.skipped ~aggregation ~backend:backend_name claim;
      Verdict.skipped ~aggregation ~backend:backend_name rendered_claim
  | `Advertised ->
      let results =
        List.init n_seeds ~f:(fun k ->
            let transform opt =
              let p = List.nth_exn (seeds_of opt) k in
              Sched.apply (Autotune.sketch_schedule ~accum_prec ~p opt) opt
            in
            match
              let sctx, sroutine =
                Context.compile
                  ~lowered_transform:(fun o -> [ transform o ])
                  (Context.auto ()) comp Ir.Indexing.Empty
              in
              let sctx = Context.run sctx sroutine in
              (Context.get_values sctx y.Tensor.value, sroutine.Context.mma)
            with
            | got, census ->
                if not (Ll_test.rendered_tensorized census) then
                  Stdio.eprintf "mc: GPU tensorized seed %d rendered %s\n" k
                    (Ir.C_syntax.mma_summary_string census);
                ( Array.for_all2_exn got expected ~f:(fun a b -> Float.(abs (a - b) < 1e-3)),
                  Ll_test.rendered_tensorized census )
            | exception exn ->
                Stdio.eprintf "mc: GPU tensorized seed %d FAILED %s\n" k (Exn.to_string exn);
                (false, false))
      in
      p_all claim results ~f:fst;
      p_all rendered_claim results ~f:snd);

  let reports = ref [] in
  let ctx = Context.auto () in
  let ctx, routine =
    Autotune.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:"autotune_cache_mma_companion"
      ~timing_ctx:(Context.auto ())
      ~report:(fun r -> reports := r :: !reports)
      ctx comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx y.Tensor.value in
  p_all2 "mc: tuned classifier head matches the reference" got expected ~f:(fun a b ->
      Float.(abs (a - b) < 1e-3));
  match !reports with
  | [ r ] ->
      (* The assertion the arc was missing: SEEDED is not TIMED. Every GPU backend used to seed
         dozens of tensorized candidates per search and time none of them, and no test noticed
         because the family-level counters stayed non-zero on the fused twins alone. *)
      p "mc: no tensorized candidate family is seeded without any of it being timed"
        (r.Autotune.mma_candidates = 0 || r.Autotune.mma_timed > 0)
  | _ -> p "mc: exactly one tuning report" false
