(* gh-ocannl-964: adoption ratchet over the two packages' test sources. *)
open Base
open Stdio
module Scan = Test_utils.Ll_test_scan
module Inventory = Test_utils.Source_inventory

(* Existing debt is deliberately exact by path. Remove a row when its test adopts the harness;
   arrayjit cannot do that until gh-ocannl-954 splits out the package-safe builders. *)
let exemptions =
  [
    ("arrayjit/test/test_cross_cse.ml", "blocked-on-954: arrayjit package cannot link ll_test");
    ( "arrayjit/test/test_local_scope_init.ml",
      "blocked-on-954: arrayjit package cannot link ll_test" );
    ( "arrayjit/test/test_one_hot_gather_rewrite.ml",
      "blocked-on-954: arrayjit package cannot link ll_test" );
    ( "arrayjit/test/test_stage_b_where_debug.ml",
      "blocked-on-954: arrayjit package cannot link ll_test" );
    ( "arrayjit/test/test_vectorized_codegen.ml",
      "blocked-on-954: arrayjit package cannot link ll_test" );
    ("test/operations/affine_extraction.ml", "existing migration debt; adopt ll_test when touched");
    ("test/operations/affine_lowering.ml", "existing migration debt; adopt ll_test when touched");
    ("test/operations/autotune_scope_menu.ml", "existing migration debt; adopt ll_test when touched");
    ("test/operations/autotune_smoke.ml", "existing migration debt; adopt ll_test when touched");
    ("test/operations/cost_model_floor.ml", "existing migration debt; adopt ll_test when touched");
    ("test/operations/cpu_simd_reduction.ml", "existing migration debt; adopt ll_test when touched");
    ("test/operations/fission_schedule.ml", "existing migration debt; adopt ll_test when touched");
    ( "test/operations/hardware_axes_parity.ml",
      "existing migration debt; adopt ll_test when touched" );
    ("test/operations/hip_scratch_budget.ml", "existing migration debt; adopt ll_test when touched");
    ( "test/operations/mma_tensorization_label.ml",
      "existing migration debt; adopt ll_test when touched" );
    ( "test/operations/model_default_fallback.ml",
      "existing migration debt; adopt ll_test when touched" );
    ( "test/operations/narrow_storage_compute.ml",
      "existing migration debt; adopt ll_test when touched" );
    ("test/operations/schedule_conv_gemm.ml", "existing migration debt; adopt ll_test when touched");
    ( "test/operations/schedule_cpu_pack_matmul.ml",
      "existing migration debt; adopt ll_test when touched" );
    ( "test/operations/schedule_epilogue_fusion.ml",
      "existing migration debt; adopt ll_test when touched" );
    ( "test/operations/schedule_ldmatrix_matmul.ml",
      "existing migration debt; adopt ll_test when touched" );
    ( "test/operations/schedule_pack_mma_matmul.ml",
      "existing migration debt; adopt ll_test when touched" );
    ( "test/operations/schedule_pipelined_matmul.ml",
      "existing migration debt; adopt ll_test when touched" );
    ( "test/operations/schedule_register_matmul.ml",
      "existing migration debt; adopt ll_test when touched" );
    ( "test/operations/schedule_swizzle_matmul.ml",
      "existing migration debt; adopt ll_test when touched" );
    ( "test/operations/scratch_value_variance.ml",
      "existing migration debt; adopt ll_test when touched" );
    ( "test/operations/test_bounds_folded_gather.ml",
      "existing migration debt; adopt ll_test when touched" );
    ( "test/operations/test_one_hot_embedding_backward.ml",
      "existing migration debt; adopt ll_test when touched" );
    ( "test/operations/test_one_hot_embedding_lookup.ml",
      "existing migration debt; adopt ll_test when touched" );
    ("test/operations/test_slice_alias.ml", "existing migration debt; adopt ll_test when touched");
    ("test/operations/tile_mma_declines.ml", "existing migration debt; adopt ll_test when touched");
    ("test/operations/tile_mma_geometry.ml", "existing migration debt; adopt ll_test when touched");
    ( "test/support/ll_test.ml",
      "canonical harness implementation owns these builders and traversals" );
  ]

let scan ~exemptions root generated =
  let inventory = Inventory.of_dune_sandbox ~workspace_root:root ~generated in
  let sources = Inventory.select inventory ~f:Scan.test_source in
  let constructors =
    Scan.constructors
      (In_channel.read_all (Stdlib.Filename.concat root "arrayjit/lib/low_level.ml"))
  in
  let rows =
    List.map sources ~f:(fun (file : Inventory.file) ->
        let dir = Stdlib.Filename.dirname file.path in
        let modules =
          List.filter_map sources ~f:(fun (s : Inventory.file) ->
              if String.equal (Stdlib.Filename.dirname s.path) dir then
                Some (Stdlib.Filename.remove_extension (Stdlib.Filename.basename s.path))
              else None)
        in
        let dune = Stdlib.Filename.concat root (dir ^ "/dune") in
        let linked =
          Stdlib.Sys.file_exists dune
          && Scan.linked ~directory_modules:modules
               ~module_name:(Stdlib.Filename.remove_extension (Stdlib.Filename.basename file.path))
               (In_channel.read_all dune)
        in
        let counts = Scan.census ~constructors (In_channel.read_all file.on_disk) in
        if Scan.needs_harness counts && not linked then
          eprintf "%s records=%d traversals=%d\n" file.path counts.records counts.traversals;
        (file.path, counts, linked))
  in
  let problems = Scan.violations ~exemptions rows in
  List.iter problems ~f:Verdict.fail;
  List.iter
    [ ("test/", 200); ("arrayjit/test/", 20) ]
    ~f:(fun (prefix, floor) ->
      let count =
        List.count sources ~f:(fun (file : Inventory.file) -> String.is_prefix file.path ~prefix)
      in
      printf "Source floor: %s >= %d\n" prefix floor;
      if count < floor then Verdict.fail (prefix ^ ": source inventory below floor"));
  printf "Adoption threshold: 3 record constructions or 1 private traversal\n";
  List.iter exemptions ~f:(fun (path, reason) -> printf "%s -- %s\n" path reason)

let () =
  match Array.to_list Stdlib.Sys.argv with
  | [ _; "--fixture"; root ] -> scan ~exemptions:[] root []
  | [ _; "--fixture-exempt"; root ] ->
      scan ~exemptions:[ ("test/new.ml", "control exemption") ] root []
  | _ :: root :: generated -> scan ~exemptions root generated
  | _ -> Stdlib.exit 2
