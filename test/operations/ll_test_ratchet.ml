(* gh-ocannl-964: adoption ratchet over the two packages' test sources. *)
open Base
open Stdio
module Scan = Test_utils.Ll_test_scan
module Inventory = Test_utils.Source_inventory

(* Existing debt is exact by path and capped independently in both detected metrics. Decreases
   remain valid; adoption (or eliminating all detected debt) makes the row stale. The canonical
   harness itself is an intentional permanent exception. Remove a row when its test adopts the
   harness or the package-safe arrayjit.ll_builders: links it AND calls its IR surface. Linking it
   for an operand helper ([Ll_test.cycle]) adopts nothing and retires no row (gh-ocannl-1052). *)
let exemptions =
  [
    ( "test/operations/affine_lowering.ml",
      Scan.Migration { records = 0; traversals = 3 },
      "existing migration debt; adopt ll_test when touched" );
    ( "test/operations/autotune_scope_menu.ml",
      Scan.Migration { records = 0; traversals = 1 },
      "existing migration debt; adopt ll_test when touched" );
    ( "test/operations/autotune_smoke.ml",
      Scan.Migration { records = 3; traversals = 1 },
      "existing migration debt; adopt ll_test when touched" );
    ( "test/operations/buffer_aliasing.ml",
      Scan.Migration { records = 1; traversals = 0 },
      "links ll_test for operand helpers only; still hand-builds Low_level" );
    ( "test/operations/cost_model_selection.ml",
      Scan.Migration { records = 0; traversals = 1 },
      "links ll_test for operand helpers only; still hand-builds Low_level" );
    ( "test/operations/cpu_parallel.ml",
      Scan.Migration { records = 11; traversals = 1 },
      "links ll_test for operand helpers only; still hand-builds Low_level" );
    ( "test/operations/cpu_simd_reduction.ml",
      Scan.Migration { records = 5; traversals = 1 },
      "links ll_test for operand helpers only; still hand-builds Low_level" );
    ( "test/operations/cpu_vectorized.ml",
      Scan.Migration { records = 0; traversals = 1 },
      "links ll_test for operand helpers only; still hand-builds Low_level" );
    ( "test/operations/fission_schedule.ml",
      Scan.Migration { records = 9; traversals = 2 },
      "links ll_test for operand helpers only; still hand-builds Low_level" );
    ( "test/operations/hardware_axes_parity.ml",
      Scan.Migration { records = 4; traversals = 4 },
      "existing migration debt; adopt ll_test when touched" );
    ( "test/operations/hip_scratch_budget.ml",
      Scan.Migration { records = 5; traversals = 0 },
      "existing migration debt; adopt ll_test when touched" );
    ( "test/operations/mma_tensorization_label.ml",
      Scan.Migration { records = 0; traversals = 1 },
      "links ll_test for operand helpers only; still hand-builds Low_level" );
    ( "test/operations/model_default_fallback.ml",
      Scan.Migration { records = 2; traversals = 2 },
      "existing migration debt; adopt ll_test when touched" );
    ( "test/operations/narrow_storage_compute.ml",
      Scan.Migration { records = 0; traversals = 1 },
      "existing migration debt; adopt ll_test when touched" );
    ( "test/operations/op_legality.ml",
      Scan.Migration { records = 4; traversals = 1 },
      "links ll_test for operand helpers only; still hand-builds Low_level" );
    ( "test/operations/schedule_conv_gemm.ml",
      Scan.Migration { records = 0; traversals = 1 },
      "links ll_test for operand helpers only; still hand-builds Low_level" );
    ( "test/operations/schedule_cpu_pack_matmul.ml",
      Scan.Migration { records = 0; traversals = 1 },
      "links ll_test for operand helpers only; still hand-builds Low_level" );
    ( "test/operations/schedule_epilogue_fusion.ml",
      Scan.Migration { records = 4; traversals = 4 },
      "links ll_test for operand helpers only; still hand-builds Low_level" );
    ( "test/operations/schedule_ldmatrix_matmul.ml",
      Scan.Migration { records = 0; traversals = 1 },
      "links ll_test for operand helpers only; still hand-builds Low_level" );
    ( "test/operations/schedule_mma_matmul.ml",
      Scan.Migration { records = 0; traversals = 1 },
      "links ll_test for operand helpers only; still hand-builds Low_level" );
    ( "test/operations/schedule_ops.ml",
      Scan.Migration { records = 4; traversals = 1 },
      "links ll_test for operand helpers only; still hand-builds Low_level" );
    ( "test/operations/schedule_pack_mma_matmul.ml",
      Scan.Migration { records = 0; traversals = 1 },
      "links ll_test for operand helpers only; still hand-builds Low_level" );
    ( "test/operations/schedule_packed_constants.ml",
      Scan.Migration { records = 0; traversals = 1 },
      "links ll_test for operand helpers only; still hand-builds Low_level" );
    ( "test/operations/schedule_pad.ml",
      Scan.Migration { records = 0; traversals = 1 },
      "links ll_test for operand helpers only; still hand-builds Low_level" );
    ( "test/operations/schedule_pipelined_matmul.ml",
      Scan.Migration { records = 3; traversals = 8 },
      "links ll_test for operand helpers only; still hand-builds Low_level" );
    ( "test/operations/schedule_register_matmul.ml",
      Scan.Migration { records = 0; traversals = 1 },
      "links ll_test for operand helpers only; still hand-builds Low_level" );
    ( "test/operations/schedule_smem_matmul.ml",
      Scan.Migration { records = 0; traversals = 1 },
      "links ll_test for operand helpers only; still hand-builds Low_level" );
    ( "test/operations/schedule_split_reduce.ml",
      Scan.Migration { records = 0; traversals = 7 },
      "links ll_test for operand helpers only; still hand-builds Low_level" );
    ( "test/operations/schedule_swizzle_matmul.ml",
      Scan.Migration { records = 0; traversals = 1 },
      "links ll_test for operand helpers only; still hand-builds Low_level" );
    ( "test/operations/scratch_value_variance.ml",
      Scan.Migration { records = 5; traversals = 0 },
      "existing migration debt; adopt ll_test when touched" );
    ( "test/operations/test_bounds_folded_gather.ml",
      Scan.Migration { records = 0; traversals = 2 },
      "existing migration debt; adopt ll_test when touched" );
    ( "test/operations/test_one_hot_embedding_backward.ml",
      Scan.Migration { records = 0; traversals = 4 },
      "existing migration debt; adopt ll_test when touched" );
    ( "test/operations/test_one_hot_embedding_lookup.ml",
      Scan.Migration { records = 0; traversals = 6 },
      "existing migration debt; adopt ll_test when touched" );
    ( "test/operations/test_slice_alias.ml",
      Scan.Migration { records = 0; traversals = 2 },
      "existing migration debt; adopt ll_test when touched" );
    ( "test/operations/tile_mma_declines.ml",
      Scan.Migration { records = 0; traversals = 1 },
      "links ll_test for operand helpers only; still hand-builds Low_level" );
    ( "test/operations/tile_mma_geometry.ml",
      Scan.Migration { records = 0; traversals = 1 },
      "links ll_test for operand helpers only; still hand-builds Low_level" );
    ( "test/operations/tile_mma_narrow.ml",
      Scan.Migration { records = 0; traversals = 1 },
      "links ll_test for operand helpers only; still hand-builds Low_level" );
    ( "test/operations/vec_signed_zero.ml",
      Scan.Migration { records = 0; traversals = 2 },
      "links ll_test for operand helpers only; still hand-builds Low_level" );
    ( "test/support/ll_builders.ml",
      Scan.Permanent,
      "canonical pure IR builders shared by both packages" );
  ]

let scan ~exemptions root generated =
  let inventory = Inventory.of_dune_sandbox ~workspace_root:root ~generated in
  let sources = Inventory.select inventory ~f:Scan.test_source in
  let constructors =
    Scan.constructors
      (In_channel.read_all (Stdlib.Filename.concat root "arrayjit/lib/low_level.ml"))
  in
  let dune_files =
    Inventory.select inventory ~f:(fun path -> String.equal (Stdlib.Filename.basename path) "dune")
    |> List.map ~f:(fun (file : Inventory.file) -> (file.path, In_channel.read_all file.on_disk))
  in
  let is_linked, ownership_problems =
    Scan.ownership
      ~sources:(List.map sources ~f:(fun (file : Inventory.file) -> file.path))
      ~dune_files
  in
  List.iter ownership_problems ~f:Verdict.fail;
  let surface =
    Scan.surface
      (List.map Scan.harness_sources ~f:(fun path ->
           In_channel.read_all (Stdlib.Filename.concat root path)))
  in
  let rows =
    List.map sources ~f:(fun (file : Inventory.file) ->
        let source = In_channel.read_all file.on_disk in
        let counts = Scan.census ~constructors source in
        let adoption : Scan.adoption =
          if not (is_linked file.path) then Unlinked
          else if Scan.uses_surface ~surface source then Adopted
          else Linked_unused
        in
        (match adoption with
        | Adopted -> ()
        | Unlinked | Linked_unused ->
            if Scan.needs_harness counts then
              eprintf "%s records=%d traversals=%d%s\n" file.path counts.records counts.traversals
                (match adoption with Linked_unused -> " (linked, IR surface unused)" | _ -> ""));
        (file.path, counts, adoption))
  in
  let problems = Scan.violations ~exemptions rows in
  if List.is_empty ownership_problems then List.iter problems ~f:Verdict.fail;
  List.iter
    [ ("test/", 200); ("arrayjit/test/", 20) ]
    ~f:(fun (prefix, floor) ->
      let count =
        List.count sources ~f:(fun (file : Inventory.file) -> String.is_prefix file.path ~prefix)
      in
      printf "Source floor: %s >= %d\n" prefix floor;
      if count < floor then Verdict.fail (prefix ^ ": source inventory below floor"));
  printf "Adoption threshold: 1 record construction or 1 private traversal\n";
  (* Derived, so printed: the golden shows which harness values adopt nothing when a test links the
     harness for them alone, and a new helper's class moves it (gh-ocannl-1052). *)
  printf "Harness values outside the IR surface (adopt nothing): %s\n"
    (String.concat ~sep:", " (Set.to_list surface.other));
  List.iter exemptions ~f:(fun (path, kind, reason) ->
      match kind with
      | Scan.Permanent -> printf "%s -- permanent: %s\n" path reason
      | Scan.Migration cap ->
          printf "%s -- records<=%d traversals<=%d: %s\n" path cap.records cap.traversals reason)

let () =
  match Array.to_list Stdlib.Sys.argv with
  | [ _; "--fixture"; root ] -> scan ~exemptions:[] root []
  | [ _; "--fixture-exempt"; root ] ->
      scan
        ~exemptions:
          [ ("test/new.ml", Scan.Migration { records = 2; traversals = 1 }, "control exemption") ]
        root []
  | [ _; "--fixture-permanent"; root ] ->
      scan ~exemptions:[ ("test/new.ml", Scan.Permanent, "control permanent exemption") ] root []
  | _ :: root :: generated -> scan ~exemptions root generated
  | _ -> Stdlib.exit 2
