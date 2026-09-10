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

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments

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

let nest_paths (llc : LL.t) : Ir.Indexing.symbol list list =
  let strip stmts = List.filter stmts ~f:(function LL.Noop | LL.Comment _ -> false | _ -> true) in
  let rec path (llc : LL.t) : Ir.Indexing.symbol list =
    match llc with
    | LL.For_loop { index; body; _ } ->
        index :: (match strip (LL.flat_lines [ body ]) with [ single ] -> path single | _ -> [])
    | LL.If { body; _ } -> path body
    | _ -> []
  in
  List.filter_map (LL.flat_lines [ llc ]) ~f:(fun stmt ->
      match path stmt with [] -> None | p -> Some p)

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

(* Intentional dialect identity: the remaining branch pins the exact MSL fragment-store and barrier
   ordering versus the CUDA/HIP emitted spellings. Whether MMA fired is censused. *)
let on_metal = String.is_substring backend_name ~substring:"metal"
let on_gpu = Sched.backend_is_gpu backend_name

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name
let n = 32
let bm = 16
let simd_width = 32
let mav = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 13) *. 0.25)
let mbv = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 17) -. 8.)
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

let run_with name transform (mc : Tensor.t) =
  let ctx = Context.auto () in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun o -> [ transform o ])
      ctx
      (named name (Train.forward mc))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  Context.get_values ctx mc.Tensor.value

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
    let paths = nest_paths opt.LL.llc in
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
          Sched.Privatize { target = prod2.Tensor.value; over = k_o };
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
    let paths = nest_paths opt.LL.llc in
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
  let got3 = run_with "epf_mma" transform3 mc3 in
  p "staged fused: epilogue is a sibling of the fragment store-back" !has_epilogue_sibling;
  p_all2 "staged fused values match two-kernel" got3 want ~f:(fun a b -> Float.(abs (a - b) < 1e-2));
  p_all2 "staged fused bitwise on C backends" got3 want ~f:(fun a b -> on_gpu || Float.equal a b);
  (let src = Generated.read "epf_mma" in
   let has s = String.is_substring src ~substring:s in
   let ok =
     if on_metal then
       (* The intrinsic fragment path fires against the workgroup-shared accumulator, and the fused
          epilogue renders after the fragment store and its trailing barrier, in the same kernel. *)
       match
         ( String.substr_index src ~pattern:"simdgroup_store(__mma_fragment_",
           String.substr_index src ~pattern:"relu_mc[" )
       with
       | Some store, Some ep ->
           has "threadgroup float" && has "simdgroup_multiply_accumulate" && store < ep
       | _ -> false
     else if on_gpu then
       (* CUDA/HIP decline uniform f32 to the lane-0 scalar fallback; the epilogue still fuses into
          the same kernel. *)
       has "== 0)" && has "relu_mc["
     else
       (* cc: the register tiling fires on the fragment array; the fused epilogue is in the same
          routine. *)
       has "Tile_mma register tiling" && has "relu_mc["
   in
   p "staged fused structure as expected" ok);

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
     match. On the C backends both enumerate and the twins double the seed list. On a
     hardware-parallel backend the UNFUSED flavor is refuted whole at tree construction — the
     companion-coverage rule (gh-ocannl-521) reports that the accumulation nest's aligned chain was
     trimmed below its geometry, i.e. exactly this file's header: the two-kernel form is not
     expressible as one kernel here, and fusion is what makes whole-routine sketches apply — so
     every seed is a twin (gh-ocannl-632). *)
  let flavors = if on_gpu then 1 else 2 in
  (match !reports with
  | [ r ] ->
      p "fused-epilogue sketch twins seeded"
        (r.Autotune.epilogue_sketch_candidates > 0
        && r.Autotune.epilogue_sketch_candidates * flavors = r.Autotune.sketch_candidates)
  | _ -> p "fused-epilogue sketch twins seeded" false);
  p_all2 "tuned matmul+tail matches two-kernel" got_t want ~f:(fun a b ->
      Float.(abs (a - b) < 1e-2));

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
     let paths = nest_paths opt.LL.llc in
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
  ignore prod4
