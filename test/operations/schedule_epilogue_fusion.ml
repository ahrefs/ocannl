(* Epilogue fusion (gh-ocannl-486): [Sched.Fuse_epilogue] folds the sole-consumer elementwise tail —
   here relu(prod + bias) after [prod = ma * mb] — into prod's store-back site, executed against the
   unfused two-kernel form on every backend.

   Three fusion sites are exercised: - the plain accumulation nest (the tail slides inside the
   output loops after the serial k loop — classic loop fusion); - the [Privatize] tile store-back of
   the S4 packed pipeline (per-element, after the final write; all-Serial, legal on every backend);
   - the lane-0 fragment store-back synthesized by [Tensorize]'s accumulator contraction (the tail
   becomes a fourth, lane-0-guarded statement of the marked region; with [shared] the accumulator
   moves to workgroup-shared memory so Metal's simdgroup fragment intrinsics keep firing — the
   epilogue then renders after the intrinsic block's trailing barrier).

   Elementwise epilogues never reorder the reduction, so on the C backends every fused form must
   match the two-kernel form BITWISE; GPU fragment paths stay under the usual tolerance (the tile
   reduction reassociates). The negative checks pin the op's pattern discipline. Note that on
   hardware-parallel schedules the UNFUSED whole-routine form is not even expressible as one kernel
   (the tail write is not covered by the hardware axes — [validate_parallel] rejects it); fusion is
   what makes whole-routine sketches apply to matmul+tail graphs at all. *)

(* cc-fp-contract: exact-operands -- quarter-step and integer operands make every product and sum
   exact *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments

(* The backend's accumulator residency, which a [Privatize] tile is minted at (gh-ocannl-1116). *)
let accum_prec = Ll_test.backend_accum_prec
let () = Utils.settings.output_debug_files_in_build_directory <- true

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

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")

(* Intentional dialect identity: the remaining branch pins the exact MSL fragment-store and barrier
   ordering versus the CUDA/HIP emitted spellings. Whether MMA fired is censused. *)
let on_metal = String.is_substring backend_name ~substring:"metal"
let on_gpu = Ir.Backend_intf.binds_hardware_grid (Context.backend_facts_of_name backend_name)

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name
let n = 32
let bm = 16
let simd_width = 32

let mav =
  Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:13 ~offset:0. ~stride:0.25)

let mbv =
  Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:17 ~offset:(-8.) ~stride:1.)

let bv = Array.init n ~f:(fun i -> Float.of_int (i % 5) -. 2.)

(* Each leg builds its own graph (tensors are single-use across routines here to keep the legs
   hermetic), so share the construction. *)
let make_graph () =
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let bias = TDSL.ndarray bv ~label:[ "bias" ] ~output_dims:[ n ] () in
  let%op prod = ma * mb in
  let%op mc = relu (prod + bias) in
  (ma, mb, prod, mc)

(* The readback, and the compile's mma census. *)
let run_census name transform (mc : Tensor.t) =
  let ctx = Context.auto () in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun o -> [ transform o ])
      ctx
      (named name (Train.forward mc))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  (Context.get_values ctx mc.Tensor.value, Ll_test.renderings routine.Context.mma)

let run_with name transform mc = fst (run_census name transform mc)

let () =
  (* --- Two-kernel reference --- *)
  let _, _, _, mc0 = make_graph () in
  let want = nonzero "epf_ref" (run_with "epf_ref" (fun opt -> opt) mc0) in

  (* --- Site 3: the plain accumulation nest --- *)
  let _, _, prod1, mc1 = make_graph () in
  let fused_count = ref (-1) in
  let transform1 (opt : LL.optimized) =
    let opt =
      Sched.apply [ Sched.Fuse_epilogue { target = prod1.Tensor.value; shared = false } ] opt
    in
    let real =
      List.filter (LL.flat_lines [ opt.LL.llc ]) ~f:(function
        | LL.Noop | LL.Comment _ | LL.Zero_out _ -> false
        | _ -> true)
    in
    fused_count := List.length real;
    opt
  in
  let got1 = run_with "epf_plain" transform1 mc1 in
  p "plain-nest fusion leaves a single nest (tail merged)" (!fused_count = 1);
  p_all2 "plain-nest fused values match two-kernel bitwise" got1 want ~f:Float.equal;

  (* --- Site 2: the S4 packed pipeline's Privatize store-back (all-Serial, every backend) --- *)
  let ma2, mb2, prod2, mc2 = make_graph () in
  let transform2 (opt : LL.optimized) =
    let paths = Ll_test.nest_paths opt.LL.llc in
    let i, j, k =
      match List.find_exn paths ~f:(fun p -> List.length p = 3) with
      | [ i; j; k ] -> (i, j, k)
      | _ -> assert false
    in
    let b = 8 in
    let sp_i, _, i_i = Sched.split ~axis:i ~factor:b ~outer:LL.Serial ~inner:LL.Serial in
    let sp_j, j_o, j_i = Sched.split ~axis:j ~factor:b ~outer:LL.Serial ~inner:LL.Serial in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:b ~outer:LL.Serial ~inner:LL.Serial in
    let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner }) in
    let sched =
      [ sp_i; sp_j; sp_k ]
      @ sink i_i [ j_o; j_i; k_o; k_i ]
      @ sink j_i [ k_o; k_i; i_i ]
      @ [
          Sched.Stage
            {
              source = ma2.Tensor.value;
              tile_loops = [ i_i; k_i ];
              shared = false;
              cooperative = None;
              hoisted = false;
              swizzle = None;
              pad_stride = None;
              pipeline_depth = 1;
              tile_prec = None;
            };
          Sched.Stage
            {
              source = mb2.Tensor.value;
              tile_loops = [ k_i; j_i ];
              shared = false;
              cooperative = None;
              hoisted = false;
              swizzle = None;
              pad_stride = None;
              pipeline_depth = 1;
              tile_prec = None;
            };
          Sched.privatize ~accum_prec ~target:prod2.Tensor.value ~over:k_o;
          Sched.Fuse_epilogue { target = prod2.Tensor.value; shared = false };
        ]
    in
    Sched.apply sched opt
  in
  let got2 = run_with "epf_packed" transform2 mc2 in
  p_all2 "packed+privatized fused values match two-kernel bitwise" got2 want ~f:Float.equal;

  (* --- Site 1: the staged tensorized fragment store-back --- *)
  let _, _, prod3, mc3 = make_graph () in
  let has_epilogue_sibling = ref false in
  let transform3 (opt : LL.optimized) =
    let paths = Ll_test.nest_paths opt.LL.llc in
    let i, j, k =
      match List.find_exn paths ~f:(fun p -> List.length p = 3) with
      | [ i; j; k ] -> (i, j, k)
      | _ -> assert false
    in
    let out = prod3.Tensor.value in
    let ez, zsyms = Sched.expand_zero ~tn:out in
    let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
    let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let rz = Sched.Retype { axis = zj; ty = LL.Workgroup } in
    let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bm ~outer:LL.Serial ~inner:LL.Serial in
    let tz, _lane = Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width () in
    let sched =
      [
        ez;
        sp_zi;
        rz;
        sp_i;
        sp_k;
        Sched.Swap { outer = j; inner = k_o };
        Sched.Swap { outer = i_i; inner = k_o };
        tz;
        Sched.Fuse_epilogue { target = out; shared = on_gpu };
      ]
    in
    let opt = Sched.apply sched opt in
    (* Structural pin: a lane-guarded statement writing the tail output immediately follows the
       lane-guarded fragment store-back, inside the same (Grid) loop body — the marked region kept
       its recognizable shape and gained the sibling epilogue. *)
    let is_lane_wg = function
      | LL.For_loop { axis = LL.Workgroup; body = LL.If _; _ } -> true
      | _ -> false
    in
    let rec writes tn = function
      | LL.Set { tn = t; _ } -> Tn.equal t tn
      | LL.Seq (x, y) -> writes tn x || writes tn y
      | LL.For_loop { body; _ } | LL.If { body; _ } -> writes tn body
      | _ -> false
    in
    let rec scan (llc : LL.t) =
      match llc with
      | LL.Seq (a, b) ->
          scan a;
          scan b;
          let b1 = match b with LL.Seq (b1, _) -> b1 | b -> b in
          if
            is_lane_wg a && is_lane_wg b1 && writes prod3.Tensor.value a
            && writes mc3.Tensor.value b1
          then has_epilogue_sibling := true
      | LL.For_loop { body; _ } | LL.If { body; _ } -> scan body
      | _ -> ()
    in
    scan opt.LL.llc;
    opt
  in
  let got3, census3 = run_census "epf_mma" transform3 mc3 in
  p "staged fused: epilogue is a sibling of the fragment store-back" !has_epilogue_sibling;
  p_close "staged fused values match two-kernel" got3 want ~tol:(fun _ -> 1e-2);
  (let label = "staged fused bitwise on C backends" in
   if on_gpu then Verdict.skipped ~backend:backend_name label
   else p_all2 label got3 want ~f:Float.equal);
  (let src = Generated.read "epf_mma" in
   let has s = String.is_substring src ~substring:s in
   (* Which way the [Tile_mma] went is the census's to say, read against the backend's capability
      for the uniform-f32 site (gh-ocannl-1316): the register tiling on the C backends, Metal's
      simdgroup intrinsics against the workgroup-shared accumulator, and the lane-0 scalar fallback
      where the backend advertises no f32-input tile (CUDA outside tf32, HIP). Either way the fused
      epilogue renders in the same routine. *)
   let rendered =
     Ll_test.census_agrees_with_capability ~backend_name ~what:"epf_mma" ~a_prec:Ir.Ops.single
       ~b_prec:Ir.Ops.single ~d_prec:Ir.Ops.single census3
   in
   let ordered =
     (not on_metal)
     ||
     (* Metal's fused epilogue renders after the fragment store and its trailing barrier: an order
        only the source has. *)
     match
       ( String.substr_index src ~pattern:"simdgroup_store(__mma_fragment_",
         String.substr_index src ~pattern:"relu_mc[" )
     with
     | Some store, Some ep -> has "threadgroup float" && store < ep
     | _ -> false
   in
   p "staged fused structure as expected" (rendered && ordered && has "relu_mc["));

  (* --- Autotune seeding: every matmul sketch gets a fused-epilogue twin on this graph, and the
     tuned routine (whichever candidate wins) matches the two-kernel reference. --- *)
  let clean_cache dir =
    if Stdlib.Sys.file_exists dir && Stdlib.Sys.is_directory dir then
      Array.iter (Stdlib.Sys.readdir dir) ~f:(fun f ->
          Stdlib.Sys.remove (Stdlib.Filename.concat dir f))
  in
  clean_cache "autotune_cache_epilogue";
  let _, _, _, mct = make_graph () in
  let reports = ref [] in
  let tctx = Context.auto () in
  let tctx, troutine =
    Autotune.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:"autotune_cache_epilogue"
      ~timing_ctx:(Context.auto ())
      ~report:(fun r -> reports := r :: !reports)
      tctx
      (named "epf_tuned" (Train.forward mct))
      Ir.Indexing.Empty
  in
  let tctx = Context.run tctx troutine in
  let got_t = Context.get_values tctx mct.Tensor.value in
  (* The two flavors are independently judged trees (gh-ocannl-577), so their leaf counts need not
     match -- but the fused flavor's companion demand is the unfused one's minus the tail its fusion
     relocates, so the unfused flavor never seeds more. On a hardware-parallel backend the UNFUSED
     flavor is refuted whole at tree construction — the companion-coverage rule (gh-ocannl-521)
     reports that the accumulation nest's aligned chain was trimmed below its geometry, i.e. exactly
     this file's header: the two-kernel form is not expressible as one kernel here, and fusion is
     what makes whole-routine sketches apply — so every seed is a twin (gh-ocannl-632). On the C
     backends both flavors enumerate; the unfused one loses only the CPU tensorized shapes binding a
     hardware dimension, which it excludes beside the tail as search economy (gh-ocannl-1609), and
     keeps the all-serial ones. *)
  (match !reports with
  | [ r ] ->
      let fused = r.Autotune.epilogue_sketch_candidates in
      let unfused = r.Autotune.sketch_candidates - fused in
      p "fused-epilogue sketch twins seeded"
        (fused > 0 && if on_gpu then unfused = 0 else unfused > 0 && unfused <= fused)
  | _ -> p "fused-epilogue sketch twins seeded" false);
  p_close "tuned matmul+tail matches two-kernel" got_t want ~tol:(fun _ -> 1e-2);

  (* --- Pattern discipline: targeted errors --- *)
  let ma4, _, prod4, mc4 = make_graph () in
  let expect_error name transform mc =
    match
      try
        ignore
          (Context.compile
             ~lowered_transform:(fun o -> [ transform o ])
             (Context.auto ())
             (named name (Train.forward mc))
             Ir.Indexing.Empty
            : Context.t * Context.routine);
        None
      with Invalid_argument msg -> Some msg
    with
    | Some msg ->
        p
          (name ^ " rejected with a targeted error")
          (String.is_substring msg ~substring:"Schedule.Fuse_epilogue")
    | None -> p (name ^ " rejected with a targeted error") false
  in
  expect_error "epf_bad_input"
    (fun opt ->
      Sched.apply [ Sched.Fuse_epilogue { target = ma4.Tensor.value; shared = false } ] opt)
    mc4;
  (* A transposed tail reads the reduction output at non-identity indices. *)
  let ma5 = TDSL.ndarray mav ~label:[ "ma5" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb5 = TDSL.ndarray mbv ~label:[ "mb5" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op prod5 = ma5 * mb5 in
  let%op mc5 = prod5 ++ "ij=>ji" in
  expect_error "epf_bad_transpose"
    (fun opt ->
      Sched.apply [ Sched.Fuse_epilogue { target = prod5.Tensor.value; shared = false } ] opt)
    mc5;
  (* A fragment store-back left inside a surplus serial loop must be rejected: the relocated tail
     would read partial accumulations. [contract_tensorized_accumulator] now contracts across the
     whole chain of qualifying serial loops (gh-ocannl-501) — the double-split-k shape that used to
     leave the store-back inside [k_oo] is contracted across, and fusion legitimately succeeds there
     — so a genuinely partial store-back is hand-built: wrap the marked three-part region in a fresh
     extent-2 loop the store-back does not index (the graph is never run; only the rejection is
     pinned). *)
  let _, _, prod6, mc6 = make_graph () in
  (let transform6 (opt : LL.optimized) =
     let paths = Ll_test.nest_paths opt.LL.llc in
     let i, j, k =
       match List.find_exn paths ~f:(fun p -> List.length p = 3) with
       | [ i; j; k ] -> (i, j, k)
       | _ -> assert false
     in
     let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Serial ~inner:LL.Serial in
     let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:8 ~outer:LL.Serial ~inner:LL.Serial in
     let tz, _lane = Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width () in
     let sched =
       [
         sp_i;
         sp_k;
         Sched.Swap { outer = j; inner = k_o };
         Sched.Swap { outer = i_i; inner = k_o };
         tz;
       ]
     in
     let opt = Sched.apply sched opt in
     (* The contraction leaves [init; k_o reduction; store-back] as the body of the outer output
        loop, the store-back a lane-guarded [Workgroup] nest. Wrap that body in the surplus loop. *)
     let surplus = Ir.Indexing.get_symbol () in
     let wrapped = ref false in
     let rec wrap (llc : LL.t) =
       match llc with
       | LL.For_loop ({ body; _ } as fc) ->
           let parts =
             List.filter (LL.flat_lines [ body ]) ~f:(function
               | LL.Noop | LL.Comment _ -> false
               | _ -> true)
           in
           if
             (not !wrapped)
             && List.exists parts ~f:(function
               | LL.For_loop { axis = LL.Workgroup; _ } -> true
               | _ -> false)
           then (
             wrapped := true;
             LL.For_loop
               {
                 fc with
                 body = LL.For_loop { index = surplus; from_ = 0; to_ = 1; body; axis = LL.Serial };
               })
           else LL.For_loop { fc with body = wrap body }
       | LL.Seq (a, b) ->
           let a = wrap a in
           LL.Seq (a, wrap b)
       | LL.If ({ body; _ } as x) -> LL.If { x with body = wrap body }
       | other -> other
     in
     let opt = { opt with LL.llc = wrap opt.LL.llc } in
     assert !wrapped;
     Sched.apply [ Sched.Fuse_epilogue { target = prod6.Tensor.value; shared = false } ] opt
   in
   match
     try
       ignore
         (Context.compile
            ~lowered_transform:(fun o -> [ transform6 o ])
            (Context.auto ())
            (named "epf_partial_storeback" (Train.forward mc6))
            Ir.Indexing.Empty
           : Context.t * Context.routine);
       None
     with Invalid_argument msg -> Some msg
   with
   | Some msg ->
       p "epf_partial_storeback rejected with a targeted error"
         (String.is_substring msg ~substring:"Schedule.Fuse_epilogue"
         && String.is_substring msg ~substring:"does not index it")
   | None -> p "epf_partial_storeback rejected with a targeted error" false);
  (* gh-ocannl-1712: the k-block [Partition] of the padded packed seeds (gh-ocannl-1670), hand-built
     on their transformer_names geometry: a batched GEMM of 15 rows, padded to one 16-row block, and
     k = 28, padded to two 16-deep blocks and partitioned at the last full block by the seeds' own
     spelling ([Sched.pad_edge_partition]), on the width-1 lane the CPU packed pipelines tensorize
     with. Where the k-block loop sits decides the store-back. Under the row-block loop, the
     accumulator contraction spans the k-block loop and both its segments, and the one store-back
     follows the whole reduction: the fusion is sound, and its values must equal the two-kernel run
     (exactly: every value is a small dyadic). With the k-block loop above the row blocks (the
     packed seeds' order), the accumulator's base indices name the row block bound inside it, so
     nothing contracts across it: each single-trip segment stores its own partial accumulation back,
     inside the batch loop that encloses both, and no enclosing loop of extent > 1 betrays the first
     store-back as partial. That fusion must be refused for the second segment's write. *)
  let b_pad = 2 and m_pad = 15 and k_pad = 28 in
  (* [batched:false] drops the batch axis (gh-ocannl-1733): the k-block segments are then top-level
     statements, the last one the last writer of the accumulator. *)
  let make_padded_graph ?(batched = true) ?(m = m_pad) ?(k = k_pad) tag =
    let rows = (if batched then b_pad else 1) * m in
    let mav =
      Array.init (rows * k)
        ~f:(Ll_test.cycle_flat ~dims:[| rows; k |] ~modulus:13 ~offset:0. ~stride:0.25)
    in
    let mbv =
      Array.init (k * n)
        ~f:(Ll_test.cycle_flat ~dims:[| k; n |] ~modulus:17 ~offset:(-8.) ~stride:1.)
    in
    let mb = TDSL.ndarray mbv ~label:[ "mb" ^ tag ] ~output_dims:[ k; n ] () in
    let bias = TDSL.ndarray bv ~label:[ "bias" ^ tag ] ~output_dims:[ n ] () in
    let ma, prod =
      if batched then
        let ma =
          TDSL.ndarray mav ~label:[ "ma" ^ tag ] ~batch_dims:[ b_pad ] ~output_dims:[ m; k ] ()
        in
        let%op prod = ma +* "b|ik;kj=>b|ij" mb in
        (ma, prod)
      else
        let ma = TDSL.ndarray mav ~label:[ "ma" ^ tag ] ~output_dims:[ m; k ] () in
        let%op prod = ma +* "ik;kj=>ij" mb in
        (ma, prod)
    in
    let%op mc = prod + bias in
    (ma, mb, prod, mc)
  in
  (* [partition] builds the Partition ops from the row-block and k-block loops; the default is the
     packed seeds' own k-block split. *)
  let padded_pipeline
      ?(partition =
        fun ~i_o:_ ~k_o -> Sched.pad_edge_partition ~block_loop:k_o ~extent:k_pad ~block:bm)
      ~k_outermost (ma : Tensor.t) (mb : Tensor.t) (opt : LL.optimized) =
    let paths = Ll_test.nest_paths opt.LL.llc in
    let i, j, k =
      (* The matmul nest: the deepest one (the batched graph's tail nest is three deep too). *)
      let depth = List.fold paths ~init:0 ~f:(fun d p -> Int.max d (List.length p)) in
      match List.find_exn paths ~f:(fun p -> List.length p = depth) with
      | [ _; i; j; k ] | [ i; j; k ] -> (i, j, k)
      | _ -> assert false
    in
    let sp_i, i_o, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Serial ~inner:LL.Serial in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bm ~outer:LL.Serial ~inner:LL.Serial in
    (* The CPU packed pipelines' width-1 lane, as the seeds tensorize. *)
    let tz, _lane = Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width:1 () in
    let stage source tile_loops =
      Sched.Stage
        {
          source;
          tile_loops;
          shared = false;
          cooperative = None;
          hoisted = false;
          swizzle = None;
          pad_stride = None;
          pipeline_depth = 1;
          tile_prec = None;
        }
    in
    Sched.apply
      ([
         Sched.Pad { axis = i; to_multiple_of = bm };
         Sched.Pad { axis = k; to_multiple_of = bm };
         sp_i;
         sp_k;
         Sched.Swap { outer = j; inner = k_o };
         Sched.Swap { outer = i_i; inner = k_o };
       ]
      @ (if k_outermost then [ Sched.Swap { outer = i_o; inner = k_o } ] else [])
      @ [ stage ma.Tensor.value [ i_i; k_i ]; stage mb.Tensor.value [ k_i; j ]; tz ]
      @ partition ~i_o ~k_o)
      opt
  in
  let fuse (prod : Tensor.t) opt =
    Sched.apply [ Sched.Fuse_epilogue { target = prod.Tensor.value; shared = false } ] opt
  in
  let partitioned_k ~k_outermost (ma : Tensor.t) (mb : Tensor.t) (prod : Tensor.t)
      (opt : LL.optimized) =
    fuse prod (padded_pipeline ~k_outermost ma mb opt)
  in
  let want_padded =
    let _, _, _, mc = make_padded_graph "7r" in
    nonzero "epf_padded_ref" (run_with "epf_padded_ref" (fun opt -> opt) mc)
  in
  (let ma, mb, prod, mc = make_padded_graph "7s" in
   let got = run_with "epf_partition_spanned" (partitioned_k ~k_outermost:false ma mb prod) mc in
   p_all2 "epf_partition_spanned (one store-back after both segments) fused values match two-kernel"
     got want_padded ~f:Float.equal);
  (let ma, mb, prod, mc = make_padded_graph "7o" in
   let label =
     "epf_partition_outermost (a store-back per single-trip segment) rejected with a targeted error"
   in
   match
     try
       ignore
         (Context.compile
            ~lowered_transform:(fun o -> [ partitioned_k ~k_outermost:true ma mb prod o ])
            (Context.auto ())
            (named "epf_partition_outermost" (Train.forward mc))
            Ir.Indexing.Empty
           : Context.t * Context.routine);
       None
     with Invalid_argument msg -> Some msg
   with
   | Some msg ->
       Stdio.eprintf "epf_partition_outermost: %s (not part of the golden)\n%!" msg;
       p label
         (String.is_substring msg ~substring:"Schedule.Fuse_epilogue"
         && String.is_substring msg ~substring:"written again")
   | None -> p label false);
  (* gh-ocannl-1721: the store-back recognizer knows the masked fragment by IDENTITY -- membership
     in [masked_fragments], which the contraction fills -- not by the ["fragment"] label head it
     mints. The real masked fragment of the padded pipeline is a member (and the fusion through it
     matches the two-kernel run, [epf_partition_spanned] above). The forgery is end to end
     (post-wave task post-1-task-13): a look-alike -- the fragment's namespace, label, precision and
     shape, a node any client can mint -- substituted for it throughout the scheduled program (its
     init-load, the contraction's accumulation, the store-back), with its traced-store entry and
     placement carried over, so that the program differs from the one that fuses by the node's
     identity alone. The fusion is refused. The control: the same forged program with the look-alike
     ENROLLED in the set fuses -- so the refusal is the identity check and nothing else the
     substitution disturbed. The superseded label rule accepted the forged node. *)
  (let ma, mb, prod, mc = make_padded_graph "7f" in
   let refusal opt =
     match fuse prod opt with
     | (_ : LL.optimized) -> None
     | exception Invalid_argument msg -> Some msg
   in
   let checked = ref false in
   let transform (opt : LL.optimized) =
     let opt = padded_pipeline ~k_outermost:false ma mb opt in
     let masked = Set.to_list opt.LL.masked_fragments in
     p "fragment identity: the padded pipeline registers exactly one masked fragment"
       (List.length masked = 1);
     p_none "fragment identity: the masked fragment is not a simdgroup fragment" masked
       ~f:(Set.mem opt.LL.simdgroup_fragments);
     let real = List.hd_exn masked in
     let look_alike = Ll_test.look_alike ~id:1721999 real in
     let forged = Ll_test.substitute_optimized ~from:real ~to_:look_alike opt in
     p
       "fragment identity: the forged program names the look-alike, labelled [fragment], and never \
        the fragment"
       (Ll_test.count_get forged look_alike > 0
       && Ll_test.count_set forged look_alike > 0
       && Ll_test.count_get forged real = 0
       && Ll_test.count_set forged real = 0
       && match look_alike.Tn.label with "fragment" :: _ -> true | _ -> false);
     (match refusal forged with
     | Some msg ->
         Stdio.eprintf "fragment identity, forged: %s (not part of the golden)\n%!" msg;
         p "fragment identity: a look-alike of the fragment is not recognized as the fragment"
           (String.is_substring msg ~substring:"Schedule.Fuse_epilogue")
     | None ->
         p "fragment identity: a look-alike of the fragment is not recognized as the fragment" false);
     let enrolled =
       { forged with LL.masked_fragments = Set.add forged.LL.masked_fragments look_alike }
     in
     p "fragment identity control: the same program with the node enrolled fuses"
       (Option.is_none (refusal enrolled));
     checked := true;
     (* The unforged pipeline, fused through the real fragment, is what runs. *)
     fuse prod opt
   in
   let got = run_with "epf_fragment_identity" transform mc in
   p "fragment identity: the transform ran" !checked;
   p_all2 "fragment identity: fused through the real masked fragment, values match two-kernel" got
     want_padded ~f:Float.equal);
  (* gh-ocannl-1733: an UNBATCHED k-split. With the k-block loop above the row blocks, each segment
     stores its own partial accumulation back, as in [epf_partition_outermost]; but with no batch
     loop the two segments are top-level statements, and the last one -- a loop starting above 0,
     which nothing re-executes -- is the last writer of the accumulator. Its store-back follows the
     whole reduction, so the tail is fused there: the relocated tail lands inside that segment and
     nowhere else, and its values equal the two-kernel run bitwise (every operand varies with each
     of its axes, so a tail reading the first k block alone differs). *)
  let refusal_of name (mc : Tensor.t) transform =
    match
      ignore
        (Context.compile
           ~lowered_transform:(fun o -> [ transform o ])
           (Context.auto ())
           (named name (Train.forward mc))
           Ir.Indexing.Empty
          : Context.t * Context.routine)
    with
    | () -> None
    | exception Invalid_argument msg ->
        Stdio.eprintf "%s: %s (not part of the golden)\n%!" name msg;
        Some msg
  in
  let fuse_refusal prod opt =
    match fuse prod opt with
    | (_ : LL.optimized) -> None
    | exception Invalid_argument msg -> Some msg
  in
  let sets_of (tn : Tn.t) = function LL.Set { tn = t; _ } -> Tn.equal t tn | _ -> false in
  (let want_unbatched =
     let _, _, _, mc = make_padded_graph ~batched:false "8r" in
     nonzero "epf_unbatched_ref" (run_with "epf_unbatched_ref" (fun opt -> opt) mc)
   in
   let ma, mb, prod, mc = make_padded_graph ~batched:false "8s" in
   let placed = ref None in
   let transform opt =
     let fused = partitioned_k ~k_outermost:true ma mb prod opt in
     let tail_writes = Ll_test.count_stmt fused.LL.llc ~f:(sets_of mc.Tensor.value) in
     let in_later_segment =
       List.sum
         (module Int)
         (Ll_test.loop_sites fused.LL.llc)
         ~f:(fun site ->
           match site.Ll_test.ls_stmt with
           | LL.For_loop { from_; body; _ } when from_ > 0 ->
               Ll_test.count_stmt body ~f:(sets_of mc.Tensor.value)
           | _ -> 0)
     in
     placed := Some (tail_writes > 0 && tail_writes = in_later_segment);
     fused
   in
   let got = run_with "epf_partition_unbatched" transform mc in
   p "epf_partition_unbatched: the tail is relocated into the last k segment, and only there"
     (Option.value ~default:false !placed);
   p_all2 "epf_partition_unbatched (fused at the last segment) fused values match two-kernel" got
     want_unbatched ~f:Float.equal);
  (* A later segment of more than one trip that does not index the store-back re-executes it with
     partial accumulations: k = 70 is five 16-deep blocks, partitioned after the third, so the last
     segment runs blocks 3 and 4. Descended, it is refused by the surplus-loop check. *)
  (let ma, mb, prod, mc = make_padded_graph ~batched:false ~k:70 "8m" in
   let partition ~i_o:_ ~k_o = [ fst (Sched.partition ~axis:k_o ~breakpoints:[ 3 ]) ] in
   let label =
     "epf_partition_multitrip (a later k segment of two trips) rejected with a targeted error"
   in
   match
     refusal_of "epf_partition_multitrip" mc (fun o ->
         fuse prod (padded_pipeline ~partition ~k_outermost:true ma mb o))
   with
   | Some msg ->
       p label
         (String.is_substring msg ~substring:"Schedule.Fuse_epilogue"
         && String.is_substring msg ~substring:"does not index it")
   | None -> p label false);
  (* A later segment of the ROW-block loop covers only the edge rows: its store-back is indexed by
     the segment's own symbol, which starts above 0, and is refused by the cover check (fusing at
     each segment is two-site fusion, not built). 20 rows are two 16-row blocks, partitioned after
     the first; the k-block loop sits under the row block, so each row block has one store-back. *)
  (let ma, mb, prod, mc = make_padded_graph ~batched:false ~m:20 "8e" in
   let partition ~i_o ~k_o:_ = [ fst (Sched.partition ~axis:i_o ~breakpoints:[ 1 ]) ] in
   let label = "epf_partition_rowedge (a later row-block segment) rejected with a targeted error" in
   match
     refusal_of "epf_partition_rowedge" mc (fun o ->
         fuse prod (padded_pipeline ~partition ~k_outermost:false ma mb o))
   with
   | Some msg ->
       p label
         (String.is_substring msg ~substring:"Schedule.Fuse_epilogue"
         && String.is_substring msg ~substring:"do not cover the output space bijectively")
   | None -> p label false);
  (* A later segment is judged only where nothing re-executes the statements before it. Under the
     batch loop, the first segment's write of a later batch iteration would follow the last
     segment's store-back, unseen by the written-again check (which reads only what follows the site
     textually). In real schedules the first segment matches first and is refused as written again
     ([epf_partition_outermost]); here its intrinsic and store-back are made unrecognizable (their
     [Workgroup] lane loops turned [Serial]) while it still writes the accumulator, so the last
     segment must not be descended. The control: the same surgery on the unbatched program, where
     the segments are top-level, fuses. Neither surgered program is run (the unfused pipeline
     is). *)
  let serialize_segment seg (llc : LL.t) =
    let rec serial (llc : LL.t) : LL.t =
      match llc with
      | LL.For_loop fc ->
          LL.For_loop
            {
              fc with
              axis = (match fc.axis with LL.Workgroup -> LL.Serial | a -> a);
              body = serial fc.body;
            }
      | LL.Seq (a, b) ->
          let a = serial a in
          LL.Seq (a, serial b)
      | LL.If x -> LL.If { x with body = serial x.body }
      | other -> other
    in
    let found = ref false in
    let rec find (llc : LL.t) : LL.t =
      match llc with
      | LL.For_loop fc when Ir.Indexing.equal_symbol fc.index seg ->
          found := true;
          serial llc
      | LL.For_loop fc -> LL.For_loop { fc with body = find fc.body }
      | LL.Seq (a, b) ->
          let a = find a in
          LL.Seq (a, find b)
      | LL.If x -> LL.If { x with body = find x.body }
      | other -> other
    in
    let res = find llc in
    assert !found;
    res
  in
  let surgered ~batched tag =
    let ma, mb, prod, mc = make_padded_graph ~batched tag in
    let verdict = ref None in
    let transform opt =
      let first = ref None in
      let partition ~i_o:_ ~k_o =
        let op, segs = Sched.partition ~axis:k_o ~breakpoints:[ 1 ] in
        first := Some (List.hd_exn segs);
        [ op ]
      in
      let opt = padded_pipeline ~partition ~k_outermost:true ma mb opt in
      let seg = Option.value_exn ~here:[%here] !first in
      verdict := Some (fuse_refusal prod { opt with LL.llc = serialize_segment seg opt.LL.llc });
      opt
    in
    ignore (run_with ("epf_partition_surgery_" ^ tag) transform mc : float array);
    Option.value_exn ~here:[%here] !verdict
  in
  (match surgered ~batched:true "8g" with
  | Some msg ->
      Stdio.eprintf "epf_partition_surgery batched: %s (not part of the golden)\n%!" msg;
      p
        "epf_partition_surgery: a later segment under a batch loop, after an unrecognized writer, \
         is not fused"
        (String.is_substring msg ~substring:"Schedule.Fuse_epilogue")
  | None ->
      p
        "epf_partition_surgery: a later segment under a batch loop, after an unrecognized writer, \
         is not fused"
        false);
  p "epf_partition_surgery control: the same surgery on the unbatched program fuses"
    (Option.is_none (surgered ~batched:false "8c"));
  ignore prod4
