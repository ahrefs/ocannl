(* gh-ocannl-697: the convolution sketch family as a refinement tree, the conv sibling of
   sketch_family_tree.

   Before, the conv family was a flat seed list whose fused-epilogue twins were a flag flipped over
   the unfused seeds, whose refusals were free-text notes, and which [model_default] scanned flat
   beside the matmul family's branch-and-bound walk. It is now [Autotune.conv_sketch_tree]: the
   matmul tree's epilogue-fusion root above a [conv-leg] level (CPU, then GPU), the CPU [conv-shape]
   menu and the GPU [conv-staging] / [conv-rows] menus with the pipelined geometries' depth twins.
   What is pinned here, on synthetic limits so every leg is machine-independent (seeding is a pure
   function of the lowering).

   The tree is printed with its verdicts, every refusal at its own decision path. Its leaves ARE the
   seed list [Autotune.sketch_seed_params] the search enumerates, and every leaf's path agrees with
   its parameters, read off the walk (fusion vs [sk_epilogue], the leg vs [sk_gpu], the shape, rows
   and depth vs [sk_grid], [sk_bm] and [sk_depth]).

   Each fusion flavor is judged by its own preconditions. A site with no fusable tail refutes the
   fused flavor with the recognizer's own reason. On a conv with a bias/relu tail, the unfused CPU
   Grid flavor (its Grid loop would leave the tail's materialized write uncovered) and the unfused
   GPU flavor (the tail cannot follow the accumulation nest's chain) are refuted with their
   witnesses, while the fused flavor, whose tail the fusion relocates into the conv nest, keeps its
   leaves. Before, both unfused flavors were proposed and failed at every compile.

   The launch caps refute at the leaf with the gate's sentence: at the cap the geometry is admitted,
   one below it refused.

   Pad waste (gh-ocannl-1393): both conv menus are judged by [judge_pad_waste]. At the default
   threshold nothing is excluded, by construction (row blocks need two or more blocks, so their
   waste stays under 2x, and the GPU channel pads are site-wide, so they cancel): no tree here
   carries a pad-waste exclusion and lifting the policy changes no leaf list. Under a lowered
   threshold a 16-row block on a 20-row site (1.6x against the whole extent's and the 8-row block's
   1.2x) is excluded with the witness, the least-padded geometries stay, and
   [Autotune.lift_pad_waste] re-proposes exactly the excluded leaves and nothing refuted.

   Register tiles (gh-ocannl-697, the matmul family's gh-ocannl-619 level): every CPU conv leaf is
   twinned with the geometries [Ir.Register_tile.alternatives] proposes for its micro-kernel's
   extents (the GEMM row, or the panel's rows, by the out-channel column), "auto" first. The tiled
   leaves are exactly those alternatives, one per alternative per untiled leaf, derived here from
   the same function the tree consults.

   [Autotune.model_family_search] walks the conv tree (it returned [None] for every conv site
   before, when [model_default] scanned conv seeds flat): with an unbeatable incumbent it scores
   leaves, and its pick is the exhaustive first argmin over the same leaves scored the same way;
   with the incumbent at the family's floor it fathoms instead of scoring. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Sspace = Ir.Schedule_space
module FD = Autotune.Family_decision
module CM = Ir.Cost_model

let scratch (o : LL.optimized) =
  {
    o with
    LL.traced_store = Hashtbl.copy o.LL.traced_store;
    LL.optimize_ctx = LL.copy_optimize_ctx o.LL.optimize_ctx;
  }

(* The routine's whole lowering, captured on its way through a compile. *)
let lowering name t =
  let captured = ref None in
  let _ctx, _routine =
    Context.compile ~name
      ~lowered_transform:(fun opt ->
        captured := Some (scratch opt);
        [ opt ])
      (Context.auto ()) (Train.forward t) Ir.Indexing.Empty
  in
  Option.value_exn !captured

let cpu_limits =
  {
    Ir.Backend_intf.no_hardware_limits with
    simd_vector = { Ir.Vector_file.bytes = 32; registers = 16 };
  }

let f32 = Ir.Backend_intf.Mma_f32

(* An 8x8x8 f32 tile, with pipelined staging at depth 2. *)
let gpu_limits =
  {
    Ir.Backend_intf.no_hardware_limits with
    mma =
      Some
        {
          Ir.Backend_intf.minimal_mma_capability with
          mma_tile = (8, 8, 8);
          mma_format_tiles = [ ((f32, f32, f32), (8, 8, 8)) ];
          mma_pipeline_depths = [ 2 ];
        };
  }

(* A 2-D valid conv, [b] images of [h x h x ic] into [oc] channels through a 3x3 window. *)
let make_conv ?(tail = false) tag ~b ~h ~ic ~oc =
  let x =
    NTDSL.init ~l:(tag ^ "x") ~prec:Ir.Ops.single ~b:[ b ] ~o:[ h; h; ic ]
      ~f:(Ll_test.weighted ~weights:[| 1; 1; 2; 3 |] ~modulus:7 ~offset:0. ~stride:1.)
      ()
  in
  let kern =
    NTDSL.init ~l:(tag ^ "k") ~prec:Ir.Ops.single ~i:[ 3; 3; ic ] ~o:[ oc ]
      ~f:(Ll_test.weighted ~weights:[| 2; 1; 1; 3 |] ~modulus:5 ~offset:(-2.) ~stride:1.)
      ()
  in
  let%op y =
    x +* "...| oh<+kh, ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." kern
  in
  if not tail then y
  else
    let bias =
      NTDSL.init ~l:(tag ^ "b") ~prec:Ir.Ops.single ~o:[ oc ]
        ~f:(fun idcs -> Float.of_int (idcs.(0) % 3) -. 1.)
        ()
    in
    let%op z = relu (y + bias) in
    z

let show (p : Autotune.sketch_params) =
  Printf.sprintf "gpu:%b simd:%d bm:%-2d bn:%d bk:%d tm:%d grid:%b epi:%b depth:%d%s"
    p.Autotune.sk_gpu p.Autotune.sk_simd p.Autotune.sk_bm p.Autotune.sk_bn p.Autotune.sk_bk
    p.Autotune.sk_tm p.Autotune.sk_grid p.Autotune.sk_epilogue p.Autotune.sk_depth
    (match p.Autotune.sk_tile with None -> "" | Some t -> " tile:" ^ Ir.Register_tile.to_string t)

let rec print_tree ~indent tree =
  match tree with
  | Sspace.Leaf p -> Stdio.printf "%s* %s\n" indent (show p)
  | Sspace.Choice { level; children } ->
      List.iter children ~f:(fun (d, child) ->
          let label = FD.to_label d in
          match child with
          | Sspace.Child sub ->
              Stdio.printf "%s%s = %s\n" indent level label;
              print_tree ~indent:(indent ^ "  ") (Lazy.force sub)
          | Sspace.Unknown (w, sub) ->
              Stdio.printf "%s%s = %s  [unknown: %s]\n" indent level label w;
              print_tree ~indent:(indent ^ "  ") (Lazy.force sub)
          | Sspace.Excluded (w, _) ->
              Stdio.printf "%s%s = %s  [excluded: %s]\n" indent level label w
          | Sspace.Refuted w -> Stdio.printf "%s%s = %s  [refuted: %s]\n" indent level label w)

let decisions path = List.map path ~f:snd
let on_path d path = List.exists path ~f:(fun (_, d') -> FD.equal d d')

(* Whether a leaf's parameters say what its path committed. Every conv decision the path carries is
   matched against the field it determines; a decision this function does not know fails, so a new
   level cannot slip past it. *)
let agrees path (p : Autotune.sketch_params) =
  p.Autotune.sk_conv
  && List.for_all (decisions path) ~f:(function
    | FD.Fusion `Unfused -> not p.Autotune.sk_epilogue
    | FD.Fusion `Fused -> p.Autotune.sk_epilogue
    | FD.Conv_leg `Cpu -> not p.Autotune.sk_gpu
    | FD.Conv_leg `Gpu -> p.Autotune.sk_gpu
    | FD.Conv_shape `Serial -> p.Autotune.sk_bm = 0 && not p.Autotune.sk_grid
    | FD.Conv_shape `Grid -> p.Autotune.sk_bm = 0 && p.Autotune.sk_grid
    | FD.Conv_shape (`Row_block bm) -> p.Autotune.sk_bm = bm && not p.Autotune.sk_grid
    | FD.Conv_staging `Plain -> p.Autotune.sk_depth = 1
    | FD.Conv_staging `Pipelined -> p.Autotune.sk_depth > 1
    | FD.Conv_rows bm -> p.Autotune.sk_bm = bm
    | FD.Twin (`Depth d) -> p.Autotune.sk_depth = d
    | FD.Register_tile None -> Option.is_none p.Autotune.sk_tile
    | FD.Register_tile (Some t) -> Option.exists p.Autotune.sk_tile ~f:(Poly.equal t)
    | _ -> false)

(* [~refused:true] states the section's passing case is an empty leaf list: every geometry is
   refuted, which the section's own claims then say. *)
let trees = ref []

let section ?(refused = false) name ~is_gpu ~limits opt =
  Stdio.printf "== %s ==\n" name;
  match Autotune.conv_sketch_tree ~is_gpu ~is_cpu:(not is_gpu) ~limits opt with
  | None ->
      p (name ^ ": the conv family claims the site") false;
      None
  | Some tree ->
      trees := tree :: !trees;
      print_tree ~indent:"  " tree;
      let seeds = Autotune.sketch_seed_params ~is_gpu ~is_cpu:(not is_gpu) ~limits opt in
      p
        (name ^ ": the tree's leaves are the seed list")
        (List.equal Poly.equal (Sspace.leaves tree) seeds);
      p
        (name ^ ": the claiming family's tree is the conv tree")
        (Option.exists (Autotune.sketch_family_tree ~is_gpu ~is_cpu:(not is_gpu) ~limits opt)
           ~f:(fun t -> List.equal Poly.equal (Sspace.leaves t) seeds));
      (* The register-tile twins of the CPU leaves, against the alternatives at each leaf's own
         micro-kernel extents (f32 sites: 4-byte elements). *)
      (if not is_gpu then
         let site = Option.value_exn (Autotune.detect_conv opt.LL.llc) in
         let alternatives (q : Autotune.sketch_params) =
           Ir.Register_tile.alternatives ~file:limits.Ir.Backend_intf.simd_vector ~elt_bytes:4
             ~m:(if q.Autotune.sk_bm = 0 then site.Autotune.c_nrow else q.Autotune.sk_bm)
             ~n:site.Autotune.c_noc
         in
         let leaves = Sspace.leaves tree in
         let sorted l = List.sort l ~compare:Poly.compare in
         p
           (name ^ ": the register-tile twins are the alternatives at each leaf's extents")
           (List.equal Poly.equal
              (sorted (List.filter leaves ~f:(fun q -> Option.is_some q.Autotune.sk_tile)))
              (sorted
                 (List.concat_map leaves ~f:(fun q ->
                      if Option.is_some q.Autotune.sk_tile then []
                      else
                        List.map (alternatives q) ~f:(fun t -> { q with Autotune.sk_tile = Some t }))))));
      if refused then
        p_empty (name ^ ": no leaf survives") ~over:(Sspace.refutations tree) (Sspace.leaves tree)
      else
        p_all (name ^ ": every leaf's path agrees with its parameters") (Sspace.enumerate tree)
          ~f:(fun (path, q) -> agrees path q);
      Some tree

let () =
  (* A 9x9 output: the row and the 4-channel reduction pad to the 8-row tile. *)
  let masked = lowering "cft_masked" (make_conv "cm" ~b:2 ~h:11 ~ic:4 ~oc:8) in
  let site = Option.value_exn (Autotune.detect_conv masked.LL.llc) in
  let cpu_tree = section "9x9 conv, CPU" ~is_gpu:false ~limits:cpu_limits masked in
  ignore (section "9x9 conv, GPU (masked)" ~is_gpu:true ~limits:gpu_limits masked);
  (* No tail: the fused flavor carries the recognizer's reason, judged at the root whatever the
     unfused flavor seeds. *)
  p "a site without a fusable tail refutes the fused flavor with the recognizer's reason"
    (match (cpu_tree, Sched.fuse_epilogue_witness ~target:site.Autotune.c_d masked) with
    | Some tree, Some why ->
        List.exists (Sspace.refutations tree) ~f:(fun (path, w) ->
            String.equal w why && List.equal Poly.equal (decisions path) [ FD.Fusion `Fused ])
    | _ -> false);
  (* A 16x16 output over 8 channels: every extent divides the tile, so the pipelined twins of the
     whole-extent and 8-row geometries are proposed. *)
  let dividing = lowering "cft_dividing" (make_conv "cd" ~b:2 ~h:18 ~ic:8 ~oc:8) in
  (match section "16x16 conv, GPU (dividing)" ~is_gpu:true ~limits:gpu_limits dividing with
  | None -> ()
  | Some tree ->
      p_exists "dividing: a pipelined depth-2 twin is a leaf" (Sspace.enumerate tree)
        ~f:(fun (path, q) -> on_path (FD.Twin (`Depth 2)) path && q.Autotune.sk_depth = 2));
  (* 64 out-channels: a micro-kernel column of eight 8-lane vectors, where the register file affords
     other C-tiles than the renderer's own -- the register-tile level appears. *)
  let wide = lowering "cft_wide" (make_conv "cw" ~b:2 ~h:18 ~ic:8 ~oc:64) in
  (match section "16x16 conv into 64 channels, CPU" ~is_gpu:false ~limits:cpu_limits wide with
  | None -> ()
  | Some tree ->
      p_exists "64 channels: a register-tile twin is a leaf" (Sspace.enumerate tree)
        ~f:(fun (path, q) ->
          Option.is_some q.Autotune.sk_tile
          && List.exists path ~f:(fun (_, d) ->
              match d with FD.Register_tile (Some _) -> true | _ -> false)));
  (* With a tail: both flavors build their subtrees. *)
  let tailed = lowering "cft_tail" (make_conv ~tail:true "ct" ~b:2 ~h:11 ~ic:4 ~oc:8) in
  (match section "9x9 conv + bias/relu tail, CPU" ~is_gpu:false ~limits:cpu_limits tailed with
  | None -> ()
  | Some tree ->
      p_exists "tail, CPU: the unfused Grid flavor is refuted on its uncovered companion"
        (Sspace.refutations tree) ~f:(fun (path, w) ->
          on_path (FD.Fusion `Unfused) path
          && on_path (FD.Conv_shape `Grid) path
          && String.is_substring w ~substring:"which validate_parallel rejects");
      p_exists "tail, CPU: the fused Grid flavor is a leaf" (Sspace.enumerate tree)
        ~f:(fun (path, _) -> on_path (FD.Fusion `Fused) path && on_path (FD.Conv_shape `Grid) path));
  (match section "9x9 conv + bias/relu tail, GPU" ~is_gpu:true ~limits:gpu_limits tailed with
  | None -> ()
  | Some tree ->
      p_exists "tail, GPU: the unfused flavor is refuted on companion coverage"
        (Sspace.refutations tree) ~f:(fun (path, w) ->
          on_path (FD.Fusion `Unfused) path && on_path (FD.Conv_leg `Gpu) path
          && String.is_substring w ~substring:"GPU conv companion coverage (gh-521)");
      p_exists "tail, GPU: the fused flavor keeps its leaves" (Sspace.enumerate tree)
        ~f:(fun (path, _) -> on_path (FD.Fusion `Fused) path);
      p_none "tail, GPU: no unfused leaf survives" (Sspace.enumerate tree) ~f:(fun (path, _) ->
          on_path (FD.Fusion `Unfused) path));
  (* Launch caps, at the leaf: a 4-image batch over a 4-row spatial loop puts the batch on grid.y.
     At a 4-wide cap the whole-extent geometry is admitted; at 3 it is refuted with the gate's
     sentence. *)
  let small = lowering "cft_caps" (make_conv "cc" ~b:4 ~h:6 ~ic:8 ~oc:8) in
  let small_site = Option.value_exn (Autotune.detect_conv small.LL.llc) in
  let capped cap = { gpu_limits with Ir.Backend_intf.max_grid_yz = Some cap } in
  (match
     section ~refused:true "4x4 conv over 4 images, GPU, grid cap 3" ~is_gpu:true ~limits:(capped 3)
       small
   with
  | None -> ()
  | Some tree ->
      p "cap 3: the whole-extent geometry is refuted with the gate's sentence"
        (List.exists (Sspace.refutations tree) ~f:(fun (path, w) ->
             on_path (FD.Conv_rows 0) path
             &&
             let geom =
               Autotune.conv_launch_geometry small_site
                 {
                   (List.hd_exn
                      (Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits:gpu_limits
                         small))
                   with
                   Autotune.sk_bm = 0;
                 }
             in
             match Sched.launch_geometry_excess ~limits:(capped 3) geom with
             | Some x -> String.is_substring w ~substring:x.Sched.lx_phrase
             | None -> false)));
  match Autotune.conv_sketch_tree ~is_gpu:true ~is_cpu:false ~limits:(capped 4) small with
  | None -> p "cap 4: the whole-extent geometry is admitted" false
  | Some tree ->
      p_exists "cap 4: the whole-extent geometry is admitted" (Sspace.enumerate tree)
        ~f:(fun (path, _) -> on_path (FD.Conv_rows 0) path)

(* The model search over a conv site, on the CPU leg under stand-in envelope constants: the score is
   the ranking score [model_default] minimizes, of each leaf's scheduled form. *)
let () =
  let limits =
    { cpu_limits with Ir.Backend_intf.peak_flops = Some 1e12; peak_memory_bandwidth = Some 1e11 }
  in
  let base = lowering "cft_model" (make_conv ~tail:true "cmd" ~b:2 ~h:11 ~ic:4 ~oc:8) in
  let rank (o : LL.optimized) =
    CM.ranking_seconds ?peak_flops:limits.peak_flops
      ?peak_memory_bandwidth:limits.peak_memory_bandwidth ~is_global:(CM.global_in o) o.LL.llc
  in
  let score segs =
    List.fold segs ~init:(Some 0.) ~f:(fun acc o -> Option.map2 acc (rank o) ~f:Float.( + ))
  in
  let search ~incumbent =
    Autotune.model_family_search ~capabilities:Ir.Backend_intf.no_codegen_capabilities ~is_gpu:false
      ~is_cpu:true ~limits ~static_indices:[] ~peak_flops:limits.peak_flops
      ~peak_memory_bandwidth:limits.peak_memory_bandwidth ~score ~incumbent base
  in
  (* Exhaustively: every leaf scheduled and scored as the walk does; the first strict minimum (no
     leaf has a tile-MMA load count on cc, so no tie-break key applies). *)
  let exhaustive =
    List.fold (Autotune.sketch_seed_params ~is_gpu:false ~is_cpu:true ~limits base) ~init:None
      ~f:(fun best p ->
        let sc =
          match Sched.apply (Autotune.sketch_schedule ~accum_prec:Fn.id ~p base) (scratch base) with
          | post -> score [ post ]
          | exception _ -> None
        in
        match (sc, best) with
        | Some s, Some (_, b) when Float.(s >= b) -> best
        | Some s, _ -> Some (p, s)
        | None, _ -> best)
  in
  match search ~incumbent:Float.infinity with
  | None -> p "model search: the walk covers the conv site" false
  | Some (pick, stats, bound) -> (
      p "model search: the walk covers the conv site" true;
      p "model search: with an unbeatable incumbent the walk scores leaves"
        (stats.Sspace.st_scored > 0);
      p "model search: the pick is the exhaustive first argmin"
        (Option.equal
           (fun (p1, s1) (p2, s2) -> Poly.equal p1 p2 && Float.equal s1 s2)
           pick exhaustive);
      match bound with
      | None -> p "model search: at the floor the walk fathoms instead of scoring" false
      | Some floor ->
          p "model search: at the floor the walk fathoms instead of scoring"
            (match search ~incumbent:floor with
            | Some (None, st, _) -> st.Sspace.st_scored = 0 && st.Sspace.st_fathomed > 0
            | _ -> false))

(* Pad waste (gh-ocannl-1393). *)
let pad_waste_prefix = "pad waste:"

let () =
  p_none "pad waste: at the default threshold no conv tree above excludes a geometry" !trees
    ~f:(fun tree ->
      List.exists (Sspace.exclusions tree) ~f:(fun (_, w) ->
          String.is_prefix w ~prefix:pad_waste_prefix));
  p_all "pad waste: at the default threshold lifting the policy changes no conv leaf list" !trees
    ~f:(fun tree ->
      List.equal Poly.equal (Sspace.leaves (Autotune.lift_pad_waste tree)) (Sspace.leaves tree));
  let opt = lowering "cft_padwaste" (make_conv "cpw" ~b:2 ~h:22 ~ic:8 ~oc:8) in
  let tree () = Autotune.conv_sketch_tree ~is_gpu:true ~is_cpu:false ~limits:gpu_limits opt in
  let default_tree = tree () in
  Hashtbl.set Utils.config_file_args ~key:"autotune_pad_waste_max" ~data:"1.25";
  p "pad waste: the lowered threshold is in force" Float.(Autotune.pad_waste_max () = 1.25);
  (* The tree is lazy and reads the threshold where a menu is built, so it is walked while the
     threshold is in force. *)
  let lowered = tree () in
  Option.iter lowered ~f:(fun t -> ignore (Sspace.enumerate t : _ list));
  Hashtbl.remove Utils.config_file_args "autotune_pad_waste_max";
  match (default_tree, lowered) with
  | Some default_tree, Some lowered ->
      Stdio.printf "== 20-row conv, GPU, pad-waste threshold 1.25 ==\n";
      print_tree ~indent:"  " lowered;
      let pad_excluded (_, w) = String.is_prefix w ~prefix:pad_waste_prefix in
      p_exists "pad waste: the 16-row block is excluded with the witness"
        (Sspace.exclusions lowered) ~f:(fun (path, w) ->
          pad_excluded (path, w) && on_path (FD.Conv_rows 16) path);
      p_none "pad waste: the least-padded geometries stay proposed" (Sspace.exclusions lowered)
        ~f:(fun (path, w) ->
          pad_excluded (path, w) && (on_path (FD.Conv_rows 0) path || on_path (FD.Conv_rows 8) path));
      let lifted = Autotune.lift_pad_waste lowered in
      p "pad waste: lifting re-proposes exactly the default threshold's leaves"
        (List.equal Poly.equal (Sspace.leaves lifted) (Sspace.leaves default_tree));
      p "pad waste: lifting re-proposes nothing refuted: the refutations are the default's"
        (List.equal Poly.equal (Sspace.refutations lifted) (Sspace.refutations default_tree))
  | _ -> p "pad waste: the conv family claims the 20-row site" false
