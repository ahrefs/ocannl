(* gh-ocannl-2194: per-file exact counts of premises outside Premise_scan's closed migration
   protocol. Counts may shrink, never grow. Other checked spellings outside that protocol remain
   debt; this is deliberately conservative rather than a proof about arbitrary helpers. *)
open Base
open Stdio
open Verdict.Claims
module Scan = Test_scan.Premise_scan

(* Remaining debt includes the helper's mathematical unit tests (discriminating_values), which are
   not operand premises. Lower a row with each conversion; remove it at zero. *)
let debt =
  [
    ("test/operations/accum_width.ml", 1);
    ("test/operations/autotune_grid_record.ml", 1);
    ("test/operations/back_to_back_runs.ml", 1);
    ("test/operations/conv_twin_packing.ml", 1);
    ("test/operations/default_prefill.ml", 1);
    ("test/operations/deferred_hit_compiles.ml", 1);
    ("test/operations/discriminating_values.ml", 2);
    ("test/operations/fragment_scope_move_legality.ml", 1);
    ("test/operations/fresh_output_fill.ml", 1);
    ("test/operations/frozen_master_pack.ml", 1);
    ("test/operations/gpu_conv_twin_tiles.ml", 1);
    ("test/operations/gpu_lane_reduction.ml", 1);
    ("test/operations/hardware_warp_shuffle.ml", 1);
    ("test/operations/matmul_interchange.ml", 1);
    ("test/operations/mma_bf16_seed_parity.ml", 2);
    ("test/operations/prefix_scan.ml", 2);
    ("test/operations/privatize_recheck.ml", 1);
    ("test/operations/reduction_forms.ml", 1);
    ("test/operations/scan_companion_coverage.ml", 1);
    ("test/operations/schedule_batch_grid.ml", 3);
    ("test/operations/schedule_batched_mma.ml", 1);
    ("test/operations/schedule_contraction_nest.ml", 4);
    ("test/operations/schedule_ldmatrix_matmul.ml", 2);
    ("test/operations/schedule_mma_matmul.ml", 5);
    ("test/operations/schedule_pad.ml", 1);
    ("test/operations/set_values_pool_coalescing.ml", 1);
    ("test/operations/split_reduce_pair.ml", 2);
    ("test/operations/tile_mma_narrow.ml", 2);
    ("test/operations/tune_arm_a_replay.ml", 1);
    ("test/operations/workgroup_memory_accounting.ml", 1);
  ]

let () =
  let argv = Stdlib.Sys.argv in
  let inventory =
    Test_scan.Source_inventory.of_dune_sandbox ~workspace_root:argv.(1)
      ~generated:[ argv.(0); "premise_ratchet.actual"; "ocannl_config" ]
  in
  let sources =
    Test_scan.Source_inventory.select inventory ~f:(fun path ->
        Test_scan.Ll_test_scan.test_source path && String.is_suffix path ~suffix:".ml")
  in
  let rows =
    List.filter_map sources ~f:(fun file ->
        match
          Scan.read_or_refusal ~reader:Scan.read ~source:file.path
            (In_channel.read_all file.on_disk)
        with
        | Ok sites -> Some (file.path, sites)
        | Error refusal ->
            Verdict.fail refusal;
            None)
  in
  p "every test source parses" (List.length rows = List.length sources);
  let findings = Scan.violations ~debt rows in
  List.iter findings ~f:Verdict.fail;
  List.iter rows ~f:(fun (path, sites) ->
      if not (List.is_empty sites) then
        printf "%s: %d uncovered, %d covered\n" path (Scan.uncovered sites)
          (List.count sites ~f:(fun s -> s.covered)));
  p_empty "premise debt has not grown and every row is current" ~over:rows findings;
  p_exists "the census contains exactness premises" rows ~f:(fun (_, sites) ->
      not (List.is_empty sites));
  Test_scan.Refusal_control_manifest.print "premise_ratchet.ml"
