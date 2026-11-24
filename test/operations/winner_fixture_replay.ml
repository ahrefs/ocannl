(* gh-ocannl-2394: extract a saved autotune winner into a digest-free replay fixture, validate every
   anchor against a fresh lowering before replaying, and refuse stale anchors by name.

   Exercises: 1. CLI extraction ([tools/extract_winner_fixture.exe]) of #2347's saved CUDA TF32
   winner into the committed digest-free fixture, plus CLI refusals on non-winner and malformed
   entries. 2. Validation and replay of #2347's extracted fixture against a fresh lowering:
   structural on-device accumulator placement on every backend, and executed CUDA TF32 parity (exact
   at tolerance 0) where CUDA advertises the TF32 tile. 3. Extraction and replay of whole-routine
   and split-reduce fissioned winners on every backend. 4. Named refusals before replaying for every
   stale-anchor class (out-of-range base loop, ambiguous/shadowed base loop, missing static index,
   unminted / out-of-rank / forward minted axis, out-of-range tensor node, mismatched segmentation
   unit count, mismatched [Zeros] segment kind, consumed loop, and failing structural precondition),
   plus hermeticity of [opt] on refusal.

   5. Single-source minted-role agreement across [Schedule_cache.minted_roles_of_op],
   [Schedule_cache.of_saved], [Schedule_cache.to_saved], and [Winner_fixture.validate] for every
   [saved_optop] constructor (gh-ocannl-2454). *)

open Base
open Stdio
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module Idx = Ir.Indexing
module LL = Ir.Low_level
module Sched = Ir.Schedule
module SC = Ir.Schedule_cache
module Tn = Ir.Tnode
module WF = Winner_fixture

let extractor_exe = (Sys.get_argv ()).(1)
let gh2347_fixture_path = (Sys.get_argv ()).(2)
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")

(* Exact saved winner from gh-ocannl-2347 (~/ocannl-repros/gh1615-tf32/tf32-failing-winner.sexp). *)
let gh2347_saved_winner =
  {|((key
  47b2612ae4ae5aee01293fef2e525508-cuda-nd45597a3-c39c58447-d345a50e95afe8033779b60c63bea66c8-tqueued-v3)
 (payload
  ((version 26) (backend cuda) (numerics d45597a3) (codegen 39c58447)
   (objective queued) (source_digest 47b2612ae4ae5aee01293fef2e525508)
   (saved ())
   (segments
    (((seg_kind Zeros) (seg_units 1)
      (seg_digest 1ed587a68e1a6466f3bf415e822706f8)
      (seg_saved
       ((Expand_zero (tn 0))
        (Retype (axis (Minted 0 (Expand_axis 0))) (ty Grid))
        (Retype (axis (Minted 0 (Expand_axis 1))) (ty Workgroup)))))
     ((seg_kind Normal) (seg_units 2)
      (seg_digest 3cd204684aa7f3d0b7b810456680974e)
      (seg_saved
       ((Split (axis (Base 0)) (factor 16) (outer Grid) (inner Serial))
        (Split (axis (Base 1)) (factor 32) (outer Grid) (inner Serial))
        (Split (axis (Base 2)) (factor 32) (outer Serial) (inner Serial))
        (Swap (outer (Minted 0 Split_inner)) (inner (Minted 1 Split_outer)))
        (Swap (outer (Minted 1 Split_inner)) (inner (Minted 2 Split_outer)))
        (Swap (outer (Minted 0 Split_inner)) (inner (Minted 2 Split_outer)))
        (Stage (source 1)
         (tile_loops ((Minted 0 Split_inner) (Minted 2 Split_inner)))
         (shared true) (cooperative (32)) (hoisted false) (pipeline_depth 2))
        (Stage (source 2)
         (tile_loops ((Minted 2 Split_inner) (Minted 1 Split_inner)))
         (shared true) (cooperative (32)) (hoisted false) (pipeline_depth 2))
        (Tensorize (i (Minted 0 Split_inner)) (j (Minted 1 Split_inner))
         (k (Minted 2 Split_inner)) (simd_width 32))
        (Fuse_epilogue (target 0) (shared true)))))))
   (best_ms 0.00423571875) (baseline_ms INF) (mma_best_ms 0.00423571875)
   (best_spread 1.0009557621935474)
   (best_steps
    ((search_shape
      "search=true beam=2 rounds=0 repeats=1 keep=0x1p+0 split_sites=8 blocks=64,128,256,512||seed=bac8a9208f99a7437b9d58b1cef0cd11")
     (steps
      ((1 0.00431412060546875) (4 0.00430575244140625) (11 0.004236142578125)
       (12 0.00423571875)))))
   (finals
    ((outcome Finals_kept)
     (contenders
      (((contender_label
         "F_swap[mma-gpu 16x32x32 simd32 pd2 ep @3cd20468/89713255:u1; plan bac8a920]")
        (contender_search_ms 0.00423571875)
        (contender_finals_ms 0.0042361494140625))
       ((contender_label
         "F_swap[mma-gpu 16x32x32 simd32 ep @3cd20468/89713255:u1; plan bac8a920]")
        (contender_search_ms 0.004236142578125)
        (contender_finals_ms 0.00423628662109375)))))))))
|}

let with_temp_dir f =
  let dir = Stdlib.Filename.temp_dir "ocannl_wf_test_" "" in
  let rec rm_rf path =
    if Stdlib.Sys.file_exists path then
      if Stdlib.Sys.is_directory path then (
        Array.iter (Stdlib.Sys.readdir path) ~f:(fun name ->
            rm_rf (Stdlib.Filename.concat path name));
        Unix.rmdir path)
      else Stdlib.Sys.remove path
  in
  Exn.protect ~f:(fun () -> f dir) ~finally:(fun () -> rm_rf dir)

let write_file path data =
  Out_channel.with_file ~binary:true path ~f:(fun oc -> Out_channel.output_string oc data)

let is_exited_0 = function Unix.WEXITED 0 -> true | _ -> false
let is_exited_nonzero = function Unix.WEXITED n when n <> 0 -> true | _ -> false

(* --- 1. CLI extraction and fail-closed input refusals --- *)
let () =
  let committed_text = In_channel.read_all gh2347_fixture_path in
  with_temp_dir (fun dir ->
      let winner_path = Stdlib.Filename.concat dir "winner.sexp" in
      let out_path = Stdlib.Filename.concat dir "fixture.sexp" in
      write_file winner_path gh2347_saved_winner;
      let status, _out, _err =
        Fresh_process.run ~exe:extractor_exe [ winner_path; "-o"; out_path ]
      in
      let extracted_text =
        if Stdlib.Sys.file_exists out_path then In_channel.read_all out_path else ""
      in
      p "CLI extracts gh-2347 saved winner to file matching the committed fixture"
        (is_exited_0 status && String.equal extracted_text committed_text);
      let status_stdout, out_stdout, _err = Fresh_process.run ~exe:extractor_exe [ winner_path ] in
      p "CLI stdout mode matches the committed fixture byte-for-byte"
        (is_exited_0 status_stdout && String.equal out_stdout committed_text);
      let forbidden_tokens =
        [
          "source_digest";
          "seg_digest";
          "47b2612a";
          "1ed587a6";
          "3cd20468";
          "version";
          "best_ms";
          "finals";
          "numerics";
          "codegen";
        ]
      in
      p_none "extracted fixture carries no cache digests, versions, or timings" forbidden_tokens
        ~f:(fun needle -> String.is_substring committed_text ~substring:needle);
      let check_cli_refuses label filename content needle =
        let path = Stdlib.Filename.concat dir filename in
        write_file path content;
        let status, _out, err = Fresh_process.run ~exe:extractor_exe [ path ] in
        p label (is_exited_nonzero status && String.is_substring err ~substring:needle)
      in
      check_cli_refuses "CLI refuses placements- cache entries by namespace" "placements.sexp"
        "((key placements-abc) (payload ((version 2))))\n" "placements-";
      check_cli_refuses "CLI refuses abandonment- cache entries by namespace" "abandonment.sexp"
        "((key abandonment-abc) (payload ((version 26))))\n" "abandonment-";
      check_cli_refuses "CLI refuses malformed sexp files" "malformed.sexp" "((not a valid entry\n"
        "Schedule_cache";
      check_cli_refuses "CLI refuses saved entries with an empty segment list" "empty_segs.sexp"
        "((key k) (payload ((version 26) (backend cuda) (numerics n) (source_digest d) (saved ()) \
         (segments ()) (best_ms 1.0) (baseline_ms 2.0))))\n"
        "empty segment list")

(* --- 2. gh-2347 fixture validation & replay (structural on every backend, executed on CUDA TF32)
   --- *)

let m = 192
and n = 32
and k = 32

let prec = Ir.Ops.single

let a_values =
  Ll_test.operand ~b:[] ~o:[ m ] ~i:[ k ]
    (Ll_test.weighted ~weights:[| 3; 1 |] ~modulus:7 ~offset:(-3.) ~stride:1.)

let b_values =
  Ll_test.operand ~b:[] ~o:[ k ] ~i:[ n ]
    (Ll_test.weighted ~weights:[| 1; 2 |] ~modulus:5 ~offset:(-2.) ~stride:1.)

let make_gh2347_graph () =
  let a = NTDSL.init ~l:"wf_fa" ~prec ~o:[ m ] ~i:[ k ] ~f:a_values () in
  let b = NTDSL.init ~l:"wf_fb" ~prec ~o:[ k ] ~i:[ n ] ~f:b_values () in
  let%op prod = a * b in
  let%op out = relu prod in
  (a.Tensor.value, b.Tensor.value, prod.Tensor.value, out)

let gh2347_fixture = WF.load_file_exn gh2347_fixture_path

let () =
  Unix.putenv "OCANNL_BUFFER_ALIASING" "false";
  let _a, _b, prod, out = make_gh2347_graph () in
  let opt = Ll_test.captured_lowering ~name:"wf_gh2347_capture" (Train.forward out) in
  let gpu = Ll_test.synthetic_gpu () in
  p "gh-2347 extracted fixture validates against a fresh lowering"
    (Result.is_ok (WF.validate ~is_gpu:gpu.is_gpu ~limits:gpu.limits gh2347_fixture opt));
  let replayed = WF.replay ~is_gpu:gpu.is_gpu ~limits:gpu.limits gh2347_fixture opt in
  p "gh-2347 extracted fixture replays to two kernels sharing the on-device accumulator"
    (match replayed with
    | Ok [ zero; fused ] ->
        LL.code_touches_tn prod zero.llc && LL.code_touches_tn prod fused.llc
        && (match Tn.Placements.get fused.optimize_ctx.placements prod with
          | Some (Tn.On_device, _) -> true
          | _ -> false)
        && not (Set.mem fused.workgroup_shared prod)
    | Ok _ | Error _ -> false)

let compile_gh2347 ~scheduled name =
  let _a, _b, prod, out = make_gh2347_graph () in
  let ctx = Context.auto () in
  let ctx, lowered_transform =
    if not scheduled then (Context.decide_materialized ctx [ prod ], fun opt -> [ opt ])
    else
      let facts = Context.backend_facts ctx in
      let is_gpu = Ir.Backend_intf.binds_hardware_grid facts in
      let limits = Context.hardware_limits ctx in
      (ctx, WF.replay_exn ~is_gpu ~limits gh2347_fixture)
  in
  let ctx, routine =
    Context.compile ~name ~lowered_transform ctx (Train.forward out) Ir.Indexing.Empty
  in
  Stdio.eprintf "%s: shipped %d segments, mma %s (not part of the golden)\n%!" name
    (List.length routine.Context.segments)
    (Ll_test.census_string (Ll_test.renderings routine.mma));
  (ctx, routine, prod, out.Tensor.value)

let () =
  let cells dims read =
    List.init (Array.fold dims ~init:1 ~f:( * )) ~f:(fun i -> read (Ll_test.unflat ~dims i))
  in
  p_all "gh-2347 fixture operands fit TF32's significand"
    (cells [| m; k |] a_values @ cells [| k; n |] b_values)
    ~f:(fun x ->
      let mantissa, _ = Float.frexp x in
      Float.is_integer (Float.ldexp mantissa 11));
  let products = Ll_test.matmul_products ~m ~n ~k a_values b_values in
  p_all "gh-2347 fixture products have exact partial sums at f32" products
    ~f:(Ll_test.exact_partials ~any_sign:true ~sig_bits:(Ir.Ops.significand_bits prec));
  Test_utils.with_policy
    (fun policy -> { policy with Ir.Numerics.tf32_matmuls = true })
    (fun () ->
      let cuda = String.equal backend_name "cuda" in
      let leg =
        Verdict.leg ~on:backend_name
          ~when_:(cuda && Ll_test.advertised ~backend_name ~a_prec:prec ~b_prec:prec ~d_prec:prec)
          ()
      in
      let reference =
        lazy
          (let ctx, routine, prod, out = compile_gh2347 ~scheduled:false "wf_gh2347_ref" in
           let ran = Context.run ctx routine in
           let values = (Exact_sums.get_stored_values ran prod, Context.get_values ran out) in
           Context.release ctx;
           values)
      in
      let replayed =
        lazy
          (let ctx, routine, _, out = compile_gh2347 ~scheduled:true "wf_gh2347_replay" in
           (* Run twice back-to-back: on the gh-ocannl-2347 defect, kernel 1's uninitialized
              workgroup-shared accumulator can accidentally read zero on a cold GPU page on the
              first launch, then reads nonzero leftovers on the second launch. *)
           let ctx = Ll_test.run_linked (ctx, routine) ~seed:[] in
           let got = Context.get_values (Ll_test.run_linked (ctx, routine) ~seed:[]) out in
           Context.release ctx;
           (got, Ll_test.renderings routine.mma))
      in
      Verdict.in_leg ~leg "gh-2347 unscheduled product matches the host reduction" (fun label ->
          p_all2 label
            (fst (Lazy.force reference))
            (Array.of_list (List.map products ~f:(List.sum (module Float) ~f:Fn.id)))
            ~f:Float.equal);
      Verdict.in_leg ~leg "gh-2347 fixture replay renders CUDA TF32 MMA intrinsics" (fun label ->
          let _, census = Lazy.force replayed in
          p label (Ll_test.all_rendered ~what:"wf_gh2347_replay" Ir.C_syntax.Mma_intrinsics census));
      Verdict.in_leg ~leg "gh-2347 fixture replay matches the unscheduled reference cell-for-cell"
        (fun label ->
          let got, _ = Lazy.force replayed in
          p_close ~min:(m * n) label got (snd (Lazy.force reference)) ~tol:(fun _ -> 0.)))

(* --- 3. Whole-routine and split-reduce fissioned winner round-trips (every backend) --- *)

let () =
  (* 3a. Whole-routine saved winner extracted and replayed on a fresh lowering. *)
  let make_whole () =
    let a =
      NTDSL.init ~l:"wf_wa" ~prec ~o:[ 16 ] ~i:[ 16 ]
        ~f:(Ll_test.weighted ~weights:[| 2; 1 |] ~modulus:5 ~offset:(-2.) ~stride:0.5)
        ()
    in
    let b =
      NTDSL.init ~l:"wf_wb" ~prec ~o:[ 16 ] ~i:[ 16 ]
        ~f:(Ll_test.weighted ~weights:[| 1; 3 |] ~modulus:5 ~offset:(-1.) ~stride:0.5)
        ()
    in
    let%op c = a * b in
    c
  in
  let c_cap = make_whole () in
  let opt_cap = Ll_test.captured_lowering ~name:"wf_whole_cap" (Train.forward c_cap) in
  let row, col =
    match Ll_test.nest_paths opt_cap.llc with
    | [ [ row; col; _ ] ] -> (row, col)
    | _ -> failwith "expected single 3-loop matmul nest"
  in
  let split_i, _io, ii = Sched.split ~axis:row ~factor:4 ~outer:LL.Serial ~inner:LL.Serial in
  let split_j, jo, _ji = Sched.split ~axis:col ~factor:4 ~outer:LL.Serial ~inner:LL.Serial in
  let sched = [ split_i; split_j; Sched.Swap { outer = ii; inner = jo } ] in
  let canon = SC.canonicalize ~with_placements:false opt_cap in
  let saved, _ = SC.to_saved (SC.base_registry canon) sched in
  let fake_entry =
    Test_utils.Cache_plant.entry ~backend:"cc" ~source_digest:"stale-digest-0000" ~saved ()
  in
  let whole_fixture = Result.ok_or_failwith (WF.extract_entry fake_entry) in
  p "whole-routine winner extracts with prelude ops and no segments"
    (Option.is_none whole_fixture.segments && List.length whole_fixture.prelude = 3);
  let c_ref = make_whole () in
  let ctx_ref, r_ref =
    Context.compile ~name:"wf_whole_ref"
      ~lowered_transform:(fun opt -> [ opt ])
      (Context.cpu ()) (Train.forward c_ref) Ir.Indexing.Empty
  in
  let expected = Context.get_values (Context.run ctx_ref r_ref) c_ref.Tensor.value in
  Context.release ctx_ref;
  let c_rep = make_whole () in
  let ctx_rep = Context.cpu () in
  let limits_cpu = Context.hardware_limits ctx_rep in
  let ctx_rep, r_rep =
    Context.compile ~name:"wf_whole_replay"
      ~lowered_transform:(WF.replay_exn ~is_gpu:false ~limits:limits_cpu whole_fixture)
      ctx_rep (Train.forward c_rep) Ir.Indexing.Empty
  in
  let got = Context.get_values (Context.run ctx_rep r_rep) c_rep.Tensor.value in
  Context.release ctx_rep;
  p_close "whole-routine extracted fixture replays on a fresh lowering with matching values" got
    expected ~tol:(fun _ -> 1e-4);

  (* 3b. Split-reduce fissioned winner (non-empty prelude + segments) extracted and replayed. *)
  let make_sr () =
    let a =
      NTDSL.init ~l:"wf_sra" ~prec ~o:[ 8; 32 ]
        ~f:(Ll_test.weighted ~weights:[| 1; 2 |] ~modulus:5 ~offset:(-2.) ~stride:0.5)
        ()
    in
    let%op s = a ++ "ij=>i" in
    s
  in
  let s_cap = make_sr () in
  let opt_sr = Ll_test.captured_lowering ~name:"wf_sr_cap" (Train.forward s_cap) in
  let reduce_loop =
    match Ll_test.nest_paths opt_sr.llc with
    | [ [ _; red ] ] -> red
    | _ -> failwith "expected 2-loop reduction nest"
  in
  let sr_op, _blk, _inn, _comb =
    Sched.split_reduce ~axis:reduce_loop ~target:s_cap.Tensor.value ~num_blocks:4
  in
  let prelude_sched = [ sr_op ] in
  let sr_canon = SC.canonicalize ~with_placements:false opt_sr in
  let prelude_saved, _ = SC.to_saved (SC.base_registry sr_canon) prelude_sched in
  let post_prelude = Sched.apply_classified prelude_sched (Ll_test.hermetic_copy opt_sr) in
  (* Give each split-reduce segment a non-empty schedule so [fission_segmented] retains the
     two-kernel boundary instead of coalescing adjacent unannotated segments into one kernel. *)
  let sr_seg_preset (seg : LL.optimized) =
    match Ll_test.nest_paths seg.llc with
    | (outer :: _) :: _ -> [ Sched.Retype { axis = outer; ty = LL.Unrolled } ]
    | _ -> []
  in
  let segmentation, tuples =
    Sched.fission_segmented ~promote_locals:false ~preset:sr_seg_preset
      ~zero_sched:(fun _ -> [])
      ~static_indices:[] post_prelude
  in
  let saved_segs = List.map (SC.save_segments segmentation tuples) ~f:fst in
  let sr_entry =
    Test_utils.Cache_plant.entry ~backend:"cc" ~source_digest:"stale-sr-digest" ~saved:prelude_saved
      ~segments:saved_segs ()
  in
  let sr_fixture = Result.ok_or_failwith (WF.extract_entry sr_entry) in
  let expected_sr_segments = Option.value_map sr_fixture.segments ~default:0 ~f:List.length in
  p "split-reduce fissioned winner extracts both prelude and a multi-kernel segmentation"
    ((not (List.is_empty sr_fixture.prelude)) && expected_sr_segments >= 2);
  let s_ref = make_sr () in
  let ctx_sr_ref, r_sr_ref =
    Context.compile ~name:"wf_sr_ref"
      ~lowered_transform:(fun opt -> [ opt ])
      (Context.cpu ()) (Train.forward s_ref) Ir.Indexing.Empty
  in
  let expected_sr = Context.get_values (Context.run ctx_sr_ref r_sr_ref) s_ref.Tensor.value in
  Context.release ctx_sr_ref;
  let s_rep = make_sr () in
  let ctx_sr_rep = Context.cpu () in
  let ctx_sr_rep, r_sr_rep =
    Context.compile ~name:"wf_sr_replay"
      ~lowered_transform:(WF.replay_exn ~is_gpu:false ~limits:limits_cpu sr_fixture)
      ctx_sr_rep (Train.forward s_rep) Ir.Indexing.Empty
  in
  let shipped_sr_segments = List.length r_sr_rep.Context.segments in
  let got_sr = Context.get_values (Context.run ctx_sr_rep r_sr_rep) s_rep.Tensor.value in
  Context.release ctx_sr_rep;
  p "split-reduce extracted fixture replay preserves the multi-kernel fission boundary"
    (shipped_sr_segments = expected_sr_segments && shipped_sr_segments >= 2);
  p_close "split-reduce extracted fixture replays on a fresh lowering with matching values" got_sr
    expected_sr ~tol:(fun _ -> 1e-4)

(* --- 4. Stale anchor refusals by name (every anchor class + hermeticity) --- *)

let refused_naming ~loc ~needles = function
  | Ok _ -> false
  | Error (err : WF.anchor_error) ->
      let msg = WF.error_to_string err in
      WF.equal_anchor_location err.location loc
      && (not (List.is_empty needles))
      && List.for_all needles ~f:(fun needle -> String.is_substring msg ~substring:needle)

let () =
  let _a, _b, prod, out = make_gh2347_graph () in
  let base_opt = Ll_test.captured_lowering ~name:"wf_stale_base" (Train.forward out) in
  let gpu = Ll_test.synthetic_gpu () in
  let segs = Option.value_exn gh2347_fixture.segments in
  let seg0, seg1 = match segs with [ s0; s1 ] -> (s0, s1) | _ -> assert false in
  let validate_gpu fixture opt = WF.validate ~is_gpu:gpu.is_gpu ~limits:gpu.limits fixture opt in

  (* 4a. Out-of-range base loop anchor (Base 99). *)
  let stale_base_loop =
    {
      WF.prelude = [];
      segments =
        Some
          [
            seg0;
            {
              seg1 with
              ops =
                SC.Split { axis = SC.Base 99; factor = 16; outer = LL.Grid; inner = LL.Serial }
                :: seg1.ops;
            };
          ];
    }
  in
  p "stale out-of-range base loop anchor is refused by name"
    (refused_naming ~loc:(WF.Segment 1)
       ~needles:[ "Base 99"; "Split"; "does not exist" ]
       (validate_gpu stale_base_loop base_opt));

  (* 4b. Shadowed / ambiguous base loop anchor (same symbol bound by two loops). *)
  let dup_sym = Idx.get_symbol () in
  let mk_tn = Ll_test.node_factory ~first_id:9000 ~dims:[| 4 |] () in
  let dup_tn = mk_tn "dup_out" in
  let dup_loop =
    Ll_test.loop_n dup_sym 4 (Ll_test.set_at dup_tn (Ll_test.iter dup_sym) (LL.Constant 1.))
  in
  let dup_opt =
    Ll_test.optimize_scoped ~materialized:[ dup_tn ] ~name:"wf_dup_loop" ~raw:dup_loop
      (Ll_test.seq dup_loop dup_loop)
  in
  let ambig_fixture =
    { WF.prelude = [ SC.Retype { axis = SC.Base 0; ty = LL.Unrolled } ]; segments = None }
  in
  p "ambiguous shadowed base loop anchor is refused by name"
    (refused_naming ~loc:WF.Prelude
       ~needles:[ "Base 0"; "Retype"; "ambiguous" ]
       (validate_gpu ambig_fixture dup_opt));

  (* 4c. Missing static axis anchor (Static 0 with no static indices). *)
  let stale_static =
    {
      WF.prelude = [];
      segments =
        Some [ seg0; { seg1 with ops = [ SC.Retype { axis = SC.Static 0; ty = LL.Unrolled } ] } ];
    }
  in
  p "stale static axis anchor is refused by name"
    (refused_naming ~loc:(WF.Segment 1)
       ~needles:[ "Static 0"; "Retype"; "does not exist" ]
       (validate_gpu stale_static base_opt));

  (* 4d. Out-of-rank minted axis anchor (Expand_axis 2 on a rank-2 node). *)
  let stale_expand_axis =
    {
      WF.prelude = [];
      segments =
        Some
          [
            {
              seg0 with
              ops =
                [
                  SC.Expand_zero { tn = 0 };
                  SC.Retype { axis = SC.Minted (0, SC.Expand_axis 2); ty = LL.Workgroup };
                ];
            };
            seg1;
          ];
    }
  in
  p "stale out-of-rank minted Expand_axis anchor is refused by name"
    (refused_naming ~loc:(WF.Segment 0)
       ~needles:[ "Minted 0 (Expand_axis 2)"; "Retype"; "Expand_zero" ]
       (validate_gpu stale_expand_axis base_opt));

  (* 4e. Forward / wrong-op minted axis anchor. *)
  let stale_forward_minted =
    {
      WF.prelude = [];
      segments =
        Some
          [
            seg0;
            { seg1 with ops = [ SC.Retype { axis = SC.Minted (5, SC.Split_outer); ty = LL.Grid } ] };
          ];
    }
  in
  p "stale forward minted axis anchor is refused by name"
    (refused_naming ~loc:(WF.Segment 1)
       ~needles:[ "Minted 5 Split_outer"; "Retype"; "has not run" ]
       (validate_gpu stale_forward_minted base_opt));

  (* 4f. Out-of-range tensor-node anchor (source tn 99). *)
  let stale_tn =
    {
      WF.prelude = [];
      segments =
        Some
          [
            seg0;
            {
              seg1 with
              ops =
                [
                  SC.Stage
                    {
                      source = 99;
                      tile_loops = [ SC.Base 0 ];
                      shared = true;
                      cooperative = None;
                      hoisted = false;
                      swizzle = None;
                      pad_stride = None;
                      pipeline_depth = None;
                      tile_prec = None;
                    };
                ];
            };
          ];
    }
  in
  p "stale out-of-range tensor-node anchor is refused by name"
    (refused_naming ~loc:(WF.Segment 1)
       ~needles:[ "tn 99"; "Stage"; "does not exist" ]
       (validate_gpu stale_tn base_opt));

  (* 4g. Stale segmentation unit count (covers 4 units on a 3-unit lowering). *)
  let stale_seg_units =
    { WF.prelude = []; segments = Some [ seg0; { seg1 with units = seg1.units + 1 } ] }
  in
  p "stale segmentation unit count is refused by name"
    (refused_naming ~loc:WF.Prelude
       ~needles:[ "segmentation"; "does not cover" ]
       (validate_gpu stale_seg_units base_opt));

  (* 4h. Stale Zeros segment kind (Zeros segment on a non-Zero_out unit, even with empty ops). *)
  let stale_zeros_kind =
    {
      WF.prelude = [];
      segments =
        Some [ { kind = `Normal; units = 1; ops = [] }; { kind = `Zeros; units = 2; ops = [] } ];
    }
  in
  p "stale Zeros segment kind on non-zero statements is refused by name"
    (refused_naming ~loc:(WF.Segment 1)
       ~needles:[ "segment 1 (Zeros)"; "Zero_out" ]
       (validate_gpu stale_zeros_kind base_opt));

  (* 4i. Consumed live loop anchor (Base 0 split then retyped). *)
  let stale_consumed_loop =
    {
      WF.prelude = [];
      segments =
        Some
          [
            seg0;
            {
              seg1 with
              ops =
                [
                  SC.Split { axis = SC.Base 0; factor = 16; outer = LL.Grid; inner = LL.Serial };
                  SC.Retype { axis = SC.Base 0; ty = LL.Workgroup };
                ];
            };
          ];
    }
  in
  p "consumed loop anchor referenced after Split is refused by name"
    (refused_naming ~loc:(WF.Segment 1) ~needles:[ "op #1"; "Retype"; "Base 0" ]
       (validate_gpu stale_consumed_loop base_opt));

  (* 4j. Structural precondition failure (Expand_zero on a node with no Zero_out in segment 1). *)
  let stale_no_zero_out =
    { WF.prelude = []; segments = Some [ seg0; { seg1 with ops = [ SC.Expand_zero { tn = 0 } ] } ] }
  in
  p "stale Expand_zero anchor with no Zero_out in segment is refused by name"
    (refused_naming ~loc:(WF.Segment 1)
       ~needles:[ "op #0"; "Expand_zero"; "tn 0"; "Zero_out" ]
       (validate_gpu stale_no_zero_out base_opt));

  (* 4k. Hermeticity: a refused replay after a mutating prelude (Split_reduce mints a partials node
     into traced_store before segment 1 fails on a stale anchor) and a refused live precondition
     after segment 0 both leave the caller's lowering traced_store and placements untouched. *)
  let placement_before = Tn.Placements.raw_entry base_opt.optimize_ctx.placements prod in
  let traced_before_base = Hashtbl.length base_opt.traced_store in
  let replay_refused_live =
    WF.replay ~is_gpu:gpu.is_gpu ~limits:gpu.limits stale_no_zero_out base_opt
  in
  let placement_after = Tn.Placements.raw_entry base_opt.optimize_ctx.placements prod in
  let traced_after_base = Hashtbl.length base_opt.traced_store in
  p "refused replay after segment 0 live precondition leaves placements and traced_store untouched"
    (Result.is_error replay_refused_live
    && traced_after_base = traced_before_base
    && Option.equal
         (fun (m1, _) (m2, _) -> Tn.equal_memory_mode m1 m2)
         placement_before placement_after);
  let s_herm =
    let a =
      NTDSL.init ~l:"wf_herm_a" ~prec ~o:[ 8; 32 ]
        ~f:(Ll_test.weighted ~weights:[| 1; 2 |] ~modulus:5 ~offset:(-2.) ~stride:0.5)
        ()
    in
    let%op s = a ++ "ij=>i" in
    s
  in
  let herm_opt = Ll_test.captured_lowering ~name:"wf_herm_sr" (Train.forward s_herm) in
  let herm_reduce =
    match Ll_test.nest_paths herm_opt.llc with
    | [ [ _; red ] ] -> red
    | _ -> failwith "expected 2-loop reduction nest"
  in
  let herm_sr_op, _, _, _ =
    Sched.split_reduce ~axis:herm_reduce ~target:s_herm.Tensor.value ~num_blocks:4
  in
  let herm_prelude_saved, _ =
    SC.to_saved (SC.base_registry (SC.canonicalize ~with_placements:false herm_opt)) [ herm_sr_op ]
  in
  let stale_after_prelude =
    {
      WF.prelude = herm_prelude_saved;
      segments =
        Some [ { kind = `Normal; units = 1; ops = [] }; { kind = `Normal; units = 99; ops = [] } ];
    }
  in
  let traced_before_sr = Hashtbl.length herm_opt.traced_store in
  let replay_refused_sr =
    WF.replay ~is_gpu:false ~limits:Ir.Backend_intf.no_hardware_limits stale_after_prelude herm_opt
  in
  let traced_after_sr = Hashtbl.length herm_opt.traced_store in
  p
    "refused replay after a mutating Split_reduce prelude leaves the caller's traced_store \
     untouched"
    (Result.is_error replay_refused_sr && traced_after_sr = traced_before_sr)

(* --- 5. Single-source minted-role agreement across Schedule_cache and Winner_fixture
   (gh-ocannl-2454) --- *)

let () =
  let module SO = Ir.Schedule_outcome in
  let tn0 = (Ll_test.node_factory ~first_id:9500 ~dims:[||] ()) "mr_r0" in
  let tn1 = (Ll_test.node_factory ~first_id:9501 ~dims:[| 4 |] ()) "mr_r1" in
  let tn2 = (Ll_test.node_factory ~first_id:9502 ~dims:[| 4; 8 |] ()) "mr_r2" in
  let i0 = Idx.get_symbol () and i1 = Idx.get_symbol () and i2 = Idx.get_symbol () in
  let body =
    Ll_test.seq
      (Ll_test.set tn0 [||] (LL.Constant 1.))
      (Ll_test.seq
         (Ll_test.set tn1 [| Ll_test.iter i0 |] (LL.Constant 2.))
         (Ll_test.set tn2 [| Ll_test.iter i0; Ll_test.iter i1 |] (LL.Constant 3.)))
  in
  let nest = Ll_test.loop_n i0 4 (Ll_test.loop_n i1 8 (Ll_test.loop_n i2 4 body)) in
  let synth_opt =
    Ll_test.optimize_scoped ~materialized:[ tn0; tn1; tn2 ] ~name:"wf_minted_roles" ~raw:nest nest
  in
  let canon = SC.canonicalize ~with_placements:false synth_opt in
  let base_reg = SC.base_registry canon in
  let ref0 = Option.value_exn (SC.resolve_tn base_reg tn0) in
  let ref1 = Option.value_exn (SC.resolve_tn base_reg tn1) in
  let ref2 = Option.value_exn (SC.resolve_tn base_reg tn2) in
  let all_ops =
    [
      SC.Split { axis = SC.Base 0; factor = 2; outer = LL.Serial; inner = LL.Serial };
      SC.Swap { outer = SC.Base 0; inner = SC.Base 1 };
      SC.Retype { axis = SC.Base 0; ty = LL.Unrolled };
      SC.Unroll { axis = SC.Base 0; materialize = false };
      SC.Partition { axis = SC.Base 0; breakpoints = [] };
      SC.Partition { axis = SC.Base 0; breakpoints = [ 1; 2 ] };
      SC.Pad { axis = SC.Base 0; to_multiple_of = 4 };
      SC.Coalesce { outer = SC.Base 0; inner = SC.Base 1 };
      SC.Stage
        {
          source = ref2;
          tile_loops = [ SC.Base 0; SC.Base 1 ];
          shared = true;
          cooperative = None;
          hoisted = false;
          swizzle = None;
          pad_stride = None;
          pipeline_depth = None;
          tile_prec = None;
        };
      SC.Privatize { target = ref2; over = SC.Base 0; acc_prec = prec };
      SC.Expand_zero { tn = ref0 };
      SC.Expand_zero { tn = ref1 };
      SC.Expand_zero { tn = ref2 };
      SC.Tensorize { i = SC.Base 0; j = SC.Base 1; k = SC.Base 2; simd_width = 8; tile = None };
      SC.Fuse_epilogue { target = ref2; shared = false };
      SC.Split_reduce { axis = SC.Base 2; target = ref1; num_blocks = 2 };
      SC.Split_reduce { axis = SC.Base 2; target = ref2; num_blocks = 2 };
      SC.Fold_mma { query = SC.Base 0; width = 8 };
    ]
  in
  p_all "Schedule_cache.minted_roles_of_op matches of_saved and to_saved for every saved_optop"
    all_ops ~f:(fun saved_op ->
      let roles = SC.minted_roles_of_op canon saved_op in
      match SC.of_saved canon [ saved_op ] with
      | [ live_op ], _ ->
          let of_bwd_ok =
            List.for_all roles ~f:(fun role ->
                match
                  SC.of_saved canon
                    [ saved_op; SC.Retype { axis = SC.Minted (0, role); ty = LL.Serial } ]
                with
                | [ _; _ ], _ -> true
                | _ -> false
                | exception _ -> false)
          in
          let saved_rt, _ = SC.to_saved base_reg [ live_op ] in
          let to_rt_ok = List.equal SC.equal_saved_optop saved_rt [ saved_op ] in
          of_bwd_ok && to_rt_ok
      | _ -> false);
  let boundary_cases =
    [
      ( SC.Split { axis = SC.Base 0; factor = 2; outer = LL.Serial; inner = LL.Serial },
        SC.Coalesce_merged );
      (SC.Swap { outer = SC.Base 0; inner = SC.Base 1 }, SC.Split_outer);
      (SC.Partition { axis = SC.Base 0; breakpoints = [] }, SC.Partition_seg 1);
      (SC.Partition { axis = SC.Base 0; breakpoints = [ 1; 2 ] }, SC.Partition_seg 3);
      (SC.Coalesce { outer = SC.Base 0; inner = SC.Base 1 }, SC.Split_outer);
      (SC.Expand_zero { tn = ref0 }, SC.Expand_axis 0);
      (SC.Expand_zero { tn = ref2 }, SC.Expand_axis 2);
      ( SC.Tensorize { i = SC.Base 0; j = SC.Base 1; k = SC.Base 2; simd_width = 8; tile = None },
        SC.Split_outer );
      (SC.Fold_mma { query = SC.Base 0; width = 8 }, SC.Split_outer);
      ( SC.Split_reduce { axis = SC.Base 2; target = ref2; num_blocks = 2 },
        SC.Split_reduce_combine 2 );
    ]
  in
  p_all "unminted role boundary is refused identically by of_saved and Winner_fixture.validate"
    boundary_cases ~f:(fun (saved_op, bad_role) ->
      let pair = [ saved_op; SC.Retype { axis = SC.Minted (0, bad_role); ty = LL.Serial } ] in
      let of_saved_refused =
        match SC.of_saved canon pair with
        | _ -> false
        | exception SO.Cause_at (SO.Transform, SO.Illegal_schedule { check; _ }) ->
            String.equal check "Schedule_cache.of_saved"
      in
      let roles = SC.minted_roles_of_op canon saved_op in
      let expected_needles =
        Sexp.to_string (SC.sexp_of_mint_role bad_role)
        ::
        (if List.is_empty roles then [ "mints no symbols" ]
         else List.map roles ~f:(fun r -> Sexp.to_string (SC.sexp_of_mint_role r)))
      in
      let wf_refused =
        match
          WF.validate ~is_gpu:false ~limits:Ir.Backend_intf.no_hardware_limits
            { WF.prelude = pair; segments = None }
            synth_opt
        with
        | Error err ->
            WF.equal_anchor_location err.location WF.Prelude
            && Option.equal Int.equal err.op_index (Some 1)
            && List.for_all expected_needles ~f:(fun needle ->
                String.is_substring (WF.error_to_string err) ~substring:needle)
        | Ok () -> false
      in
      of_saved_refused && wf_refused);
  let oob_tn_ops =
    [
      SC.Expand_zero { tn = -1 };
      SC.Expand_zero { tn = 99 };
      SC.Split_reduce { axis = SC.Base 0; target = -1; num_blocks = 2 };
      SC.Split_reduce { axis = SC.Base 0; target = 99; num_blocks = 2 };
    ]
  in
  p_all "Schedule_cache.minted_roles_of_op raises typed tn_of_ref refusal on out-of-range nodes"
    oob_tn_ops ~f:(fun op ->
      match SC.minted_roles_of_op canon op with
      | _ -> false
      | exception SO.Cause_at (SO.Transform, SO.Illegal_schedule { check; _ }) ->
          String.equal check "Schedule_cache.tn_of_ref");
  p_all "Winner_fixture.validate still refuses out-of-range tn anchors by name before role lookup"
    [
      SC.Expand_zero { tn = 99 }; SC.Split_reduce { axis = SC.Base 0; target = 99; num_blocks = 2 };
    ]
    ~f:(fun op ->
      refused_naming ~loc:WF.Prelude ~needles:[ "tn 99"; "does not exist" ]
        (WF.validate ~is_gpu:false ~limits:Ir.Backend_intf.no_hardware_limits
           { WF.prelude = [ op ]; segments = None }
           synth_opt));
  p
    "Schedule_cache.to_saved rejects a hand-built op whose minted symbol count disagrees with its \
     roles"
    (match SC.to_saved base_reg [ Sched.Expand_zero { tn = tn2; indices = [] } ] with
    | _ -> false
    | exception Invalid_argument msg -> String.is_substring msg ~substring:"minted 0 symbol(s)");
  (* Pin persisted role-to-field bindings independently of [minted_roles_of_op] so swapping two
     roles in the shared helper cannot silently redirect cached references to the wrong live
     loop. *)
  let resolves_to saved_op role expected_sym =
    match
      SC.of_saved canon [ saved_op; SC.Retype { axis = SC.Minted (0, role); ty = LL.Serial } ]
    with
    | [ live_op; Sched.Retype { axis; _ } ], reg_of ->
        let sym = expected_sym live_op in
        let expected_ref = Some (SC.Minted (0, role)) in
        let _, reg_to = SC.to_saved base_reg [ live_op ] in
        Idx.equal_symbol axis sym
        && Option.equal SC.equal_sym_ref (SC.resolve reg_of sym) expected_ref
        && Option.equal SC.equal_sym_ref (SC.resolve reg_to sym) expected_ref
    | _ -> false
  in
  let role_binding_cases =
    [
      ( SC.Split { axis = SC.Base 0; factor = 2; outer = LL.Serial; inner = LL.Serial },
        SC.Split_outer,
        function Sched.Split { outer_index; _ } -> outer_index | _ -> assert false );
      ( SC.Split { axis = SC.Base 0; factor = 2; outer = LL.Serial; inner = LL.Serial },
        SC.Split_inner,
        function Sched.Split { inner_index; _ } -> inner_index | _ -> assert false );
      ( SC.Partition { axis = SC.Base 0; breakpoints = [ 1; 2 ] },
        SC.Partition_seg 0,
        function
        | Sched.Partition { segment_indices; _ } -> List.nth_exn segment_indices 0
        | _ -> assert false );
      ( SC.Partition { axis = SC.Base 0; breakpoints = [ 1; 2 ] },
        SC.Partition_seg 1,
        function
        | Sched.Partition { segment_indices; _ } -> List.nth_exn segment_indices 1
        | _ -> assert false );
      ( SC.Partition { axis = SC.Base 0; breakpoints = [ 1; 2 ] },
        SC.Partition_seg 2,
        function
        | Sched.Partition { segment_indices; _ } -> List.nth_exn segment_indices 2
        | _ -> assert false );
      ( SC.Coalesce { outer = SC.Base 0; inner = SC.Base 1 },
        SC.Coalesce_merged,
        function Sched.Coalesce { merged; _ } -> merged | _ -> assert false );
      ( SC.Expand_zero { tn = ref2 },
        SC.Expand_axis 0,
        function Sched.Expand_zero { indices; _ } -> List.nth_exn indices 0 | _ -> assert false );
      ( SC.Expand_zero { tn = ref2 },
        SC.Expand_axis 1,
        function Sched.Expand_zero { indices; _ } -> List.nth_exn indices 1 | _ -> assert false );
      ( SC.Tensorize { i = SC.Base 0; j = SC.Base 1; k = SC.Base 2; simd_width = 8; tile = None },
        SC.Tensorize_lane,
        function Sched.Tensorize { lane; _ } -> lane | _ -> assert false );
      ( SC.Fold_mma { query = SC.Base 0; width = 8 },
        SC.Fold_mma_lane,
        function Sched.Fold_mma { lane; _ } -> lane | _ -> assert false );
      ( SC.Fold_mma { query = SC.Base 0; width = 8 },
        SC.Fold_mma_block,
        function Sched.Fold_mma { block; _ } -> block | _ -> assert false );
      ( SC.Split_reduce { axis = SC.Base 2; target = ref2; num_blocks = 2 },
        SC.Split_reduce_block,
        function Sched.Split_reduce { block_index; _ } -> block_index | _ -> assert false );
      ( SC.Split_reduce { axis = SC.Base 2; target = ref2; num_blocks = 2 },
        SC.Split_reduce_inner,
        function Sched.Split_reduce { inner_index; _ } -> inner_index | _ -> assert false );
      ( SC.Split_reduce { axis = SC.Base 2; target = ref2; num_blocks = 2 },
        SC.Split_reduce_combine 0,
        function
        | Sched.Split_reduce { combine_indices; _ } -> List.nth_exn combine_indices 0
        | _ -> assert false );
      ( SC.Split_reduce { axis = SC.Base 2; target = ref2; num_blocks = 2 },
        SC.Split_reduce_combine 1,
        function
        | Sched.Split_reduce { combine_indices; _ } -> List.nth_exn combine_indices 1
        | _ -> assert false );
    ]
  in
  p_all
    "persisted mint_role values resolve to their named live op fields independently of role order"
    role_binding_cases ~f:(fun (saved_op, role, expected_sym) ->
      resolves_to saved_op role expected_sym)
