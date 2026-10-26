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

(* gh-ocannl-1488: the combined view ([Autotune.sketch_family_tree]) is a [family = matmul | conv]
   root above the claiming family's own tree, each family judged by its own detector on this site.
   On a conv site the matmul child is refuted with the no-matmul-site witness, and the conv child is
   [conv_sketch_tree] unchanged — every path below the root is the per-family tree's, behind the one
   [family=conv] commitment, so its leaves are the seed list. The matmul site's twin of these pins
   is in sketch_family_tree. *)
let family_root_pins name ~is_gpu ~limits opt family =
  let matmul = FD.Family `Matmul and conv = FD.Family `Conv in
  let under d = List.map ~f:(fun (path, x) -> ((FD.level d, d) :: path, x)) in
  match Autotune.sketch_family_tree ~is_gpu ~is_cpu:(not is_gpu) ~limits opt with
  | Some (Sspace.Choice { level; children } as combined) ->
      let verdict = function
        | Sspace.Child _ -> "live"
        | Sspace.Unknown (w, _) -> "unknown: " ^ w
        | Sspace.Excluded (w, _) -> "excluded: " ^ w
        | Sspace.Refuted w -> "refuted: " ^ w
      in
      Stdio.printf "  combined view, root level %s:\n" level;
      List.iter children ~f:(fun (d, c) -> Stdio.printf "    %s: %s\n" (FD.to_label d) (verdict c));
      p
        (name ^ ": the root commits the family, matmul then conv")
        (String.equal level (FD.level conv)
        && List.equal FD.equal (List.map children ~f:fst) [ matmul; conv ]);
      p
        (name ^ ": below the root, the conv child is the conv family's own tree")
        (Poly.equal (Sspace.enumerate combined) (under conv (Sspace.enumerate family)));
      p
        (name ^ ": the refutations are the matmul family's no-site witness, then the conv family's")
        (Poly.equal (Sspace.refutations combined)
           (([ (FD.level matmul, matmul) ], Autotune.no_matmul_site_witness)
           :: under conv (Sspace.refutations family)))
  | _ -> p (name ^ ": a conv site has a family root above its family tree") false

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
      family_root_pins name ~is_gpu ~limits opt tree;
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

(* gh-ocannl-1489: the conv family's certain-traffic increment. Before, [sketch_path_traffic_floor]
   priced every conv path at 0, so the model search's bound was the schedule-invariant floor on
   every conv subtree and nothing below the root could fathom. A committed row geometry now prices
   both staged operand tiles, written and read, at the tiles' own precisions.

   Three things are pinned, over paths walked out of the combined view (the [family] root included,
   as the search sees them): - every leaf's increment is what its OWN sketch parameters imply
   (nothing read off the path), and the increment is monotone along every prefix; - soundness,
   exhaustively on small sites: every leaf's schedule is applied and [Cost_model.analyze]d, its
   freshly minted nodes are charged at least the increment, and for EVERY prefix of its path the
   composed floor stays at or below the leaf's [analyze] bytes and its bound at or below the leaf's
   ranking score; - pruning: on a memory-dominated envelope with the incumbent at the best leaf's
   score, the walk never admits a leaf whose own bound reaches it, at least one such leaf exists,
   and it scores fewer leaves than the same walk under the old zero increment. *)

(* A conv with a stride-2 window, the input tile's row compacted by [Stage]. *)
let make_strided_conv tag ~b ~h ~ic ~oc =
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
    x +* "...| 2*oh<+kh, 2*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." kern
  in
  y

(* Half-precision operands into a single-precision output: the CPU leg packs both operands at the
   wider compute precision ([sk_pack_prec]), so its tiles are twice the operands' width. *)
let make_narrow_conv tag ~b ~h ~ic ~oc =
  let x =
    NTDSL.init ~l:(tag ^ "x") ~prec:Ir.Ops.half ~b:[ b ] ~o:[ h; h; ic ]
      ~f:(Ll_test.weighted ~weights:[| 1; 1; 2; 3 |] ~modulus:7 ~offset:0. ~stride:1.)
      ()
  in
  let kern =
    NTDSL.init ~l:(tag ^ "k") ~prec:Ir.Ops.half ~i:[ 3; 3; ic ] ~o:[ oc ]
      ~f:(Ll_test.weighted ~weights:[| 2; 1; 1; 3 |] ~modulus:5 ~offset:(-2.) ~stride:1.)
      ()
  in
  let%op y =
    x +* "...| oh<+kh, ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." kern
  in
  y

(* The increment a leaf's own parameters imply: the two staged tiles' cells, written and read. The
   row is the panel or block ([sk_bm]), else the whole row — on the GPU leg padded to the intrinsic
   row tile the seed carries in [sk_tm]; the GPU leg pads the out-channel and reduction extents to
   the multiples it carries in [sk_bn]/[sk_bk] (0 = already a multiple). A tile is at the packing
   precision where it differs from its operand's, else at the operand's. *)
let expected_conv_inc (site : Autotune.conv_site) (p : Autotune.sketch_params) =
  let pad n t = if t = 0 then n else (n + t - 1) / t * t in
  let width (tn : Ir.Tnode.t) =
    let sp = Lazy.force tn.Ir.Tnode.storage_prec in
    match p.Autotune.sk_pack_prec with
    | Some q when not (Ir.Ops.equal_prec q sp) -> Ir.Ops.prec_in_bytes q
    | _ -> Ir.Ops.prec_in_bytes sp
  in
  let m, k, n =
    if p.Autotune.sk_gpu then
      ( (if p.Autotune.sk_bm > 0 then p.Autotune.sk_bm else pad site.Autotune.c_nrow p.Autotune.sk_tm),
        pad site.Autotune.c_nred p.Autotune.sk_bk,
        pad site.Autotune.c_noc p.Autotune.sk_bn )
    else
      ( (if p.Autotune.sk_bm > 0 then p.Autotune.sk_bm else site.Autotune.c_nrow),
        site.Autotune.c_nred,
        site.Autotune.c_noc )
  in
  let fragment =
    (* gh-ocannl-1601: the accumulator fragment Tensorize's contraction mints across the
       kernel-window chain: [m x n] at the output's storage width, written and read. *)
    if List.exists site.Autotune.c_axes ~f:(fun cx -> cx.Autotune.cx_nk >= 2) then
      2 * m * n * Ir.Ops.prec_in_bytes (Lazy.force site.Autotune.c_d.Ir.Tnode.storage_prec)
    else 0
  in
  (2 * ((m * k * width site.Autotune.c_a) + (k * n * width site.Autotune.c_b))) + fragment

let prefixes path =
  List.folding_map path ~init:[] ~f:(fun acc entry ->
      let acc = acc @ [ entry ] in
      (acc, acc))

let envelope limits ~peak_flops ~bandwidth =
  {
    limits with
    Ir.Backend_intf.peak_flops = Some peak_flops;
    peak_memory_bandwidth = Some bandwidth;
  }

let rank_with (limits : Ir.Backend_intf.hardware_limits) (o : LL.optimized) =
  CM.ranking_seconds ?peak_flops:limits.peak_flops
    ?peak_memory_bandwidth:limits.peak_memory_bandwidth ~is_global:(CM.global_in o) o.LL.llc

let apply_leaf base p =
  match Sched.apply (Autotune.sketch_schedule ~accum_prec:Fn.id ~p base) (scratch base) with
  | post -> Some post
  | exception _ -> None

let bytes_of (s : CM.summary) = s.CM.read_bytes + s.CM.write_bytes

let traffic_floor_pins name ~is_gpu ~limits base =
  let limits = envelope limits ~peak_flops:1e12 ~bandwidth:1e11 in
  match
    ( Autotune.detect_conv base.LL.llc,
      Autotune.sketch_family_tree ~is_gpu ~is_cpu:(not is_gpu) ~limits base )
  with
  | Some site, Some tree ->
      let inc = Autotune.sketch_path_traffic_floor ~limits base in
      let paths = Sspace.enumerate tree in
      let leg = if is_gpu then FD.Conv_leg `Gpu else FD.Conv_leg `Cpu in
      let on_leg = List.filter paths ~f:(fun (path, _) -> on_path leg path) in
      let floor = CM.completion_floor base.LL.llc in
      let bound_at i =
        CM.roofline_seconds ?peak_flops:limits.peak_flops
          ?peak_memory_bandwidth:limits.peak_memory_bandwidth ~flops:floor.CM.fr_flops
          ~bytes:(floor.CM.fr_bytes + i) ()
      in
      let base_nodes =
        Set.of_list (module Ir.Tnode) (List.map (CM.analyze base.LL.llc).CM.per_node ~f:fst)
      in
      let applied =
        List.filter_map paths ~f:(fun (path, p) ->
            Option.map (apply_leaf base p) ~f:(fun post -> (path, p, post)))
      in
      Stdio.printf
        "== %s: %d leaf paths (%d on the %s leg), %d applied; floor %d bytes; increments %s ==\n"
        name (List.length paths) (List.length on_leg)
        (if is_gpu then "gpu" else "cpu")
        (List.length applied) floor.CM.fr_bytes
        (String.concat ~sep:" "
           (List.dedup_and_sort ~compare:Int.compare (List.map paths ~f:(fun (path, _) -> inc path))
           |> List.map ~f:Int.to_string));
      p_all (name ^ ": every leaf's increment is the traffic its own parameters imply") paths
        ~f:(fun (path, p) -> inc path = expected_conv_inc site p);
      p_exists (name ^ ": a leaf on the target's leg prices above the floor") on_leg
        ~f:(fun (path, _) -> inc path > 0);
      p_all (name ^ ": the increment is monotone along every path's prefixes") paths
        ~f:(fun (path, _) ->
          List.is_sorted (0 :: List.map (prefixes path) ~f:inc) ~compare:Int.compare);
      p (name ^ ": every leaf's schedule applies") (List.length applied = List.length paths);
      p_all (name ^ ": the freshly minted nodes are charged at least the increment") applied
        ~f:(fun (path, _, post) ->
          let fresh =
            List.sum
              (module Int)
              (CM.analyze post.LL.llc).CM.per_node
              ~f:(fun (tn, fp) ->
                if Set.mem base_nodes tn then 0 else fp.CM.fp_read_bytes + fp.CM.fp_write_bytes)
          in
          fresh >= inc path);
      p_all (name ^ ": at every prefix the composed floor is within the leaf's analyzed bytes")
        applied ~f:(fun (path, _, post) ->
          let total = bytes_of (CM.analyze post.LL.llc) in
          List.for_all (prefixes path) ~f:(fun pre -> floor.CM.fr_bytes + inc pre <= total));
      p_all (name ^ ": every leaf's bound is within its ranking score") applied
        ~f:(fun (path, _, post) ->
          match (bound_at (inc path), rank_with limits post) with
          | Some b, Some s -> Float.(b <= s)
          | _ -> false)
  | _ -> p (name ^ ": the conv family claims the site") false

let () =
  let narrow_cpu = { cpu_limits with Ir.Backend_intf.native_fp16_arithmetic = false } in
  let masked = lowering "cft_tf_masked" (make_conv "tm" ~b:2 ~h:11 ~ic:4 ~oc:8) in
  let dividing = lowering "cft_tf_dividing" (make_conv "td" ~b:2 ~h:18 ~ic:8 ~oc:8) in
  let tailed = lowering "cft_tf_tail" (make_conv ~tail:true "tt" ~b:2 ~h:11 ~ic:4 ~oc:8) in
  let strided = lowering "cft_tf_strided" (make_strided_conv "ts" ~b:2 ~h:21 ~ic:8 ~oc:8) in
  let narrow = lowering "cft_tf_narrow" (make_narrow_conv "tn" ~b:2 ~h:18 ~ic:8 ~oc:8) in
  traffic_floor_pins "floor, masked 9-row, cpu" ~is_gpu:false ~limits:cpu_limits masked;
  traffic_floor_pins "floor, masked 9-row, gpu" ~is_gpu:true ~limits:gpu_limits masked;
  traffic_floor_pins "floor, dividing 16-row, cpu" ~is_gpu:false ~limits:cpu_limits dividing;
  traffic_floor_pins "floor, dividing 16-row, gpu" ~is_gpu:true ~limits:gpu_limits dividing;
  traffic_floor_pins "floor, bias/relu tail, cpu" ~is_gpu:false ~limits:cpu_limits tailed;
  traffic_floor_pins "floor, bias/relu tail, gpu" ~is_gpu:true ~limits:gpu_limits tailed;
  traffic_floor_pins "floor, stride 2, cpu" ~is_gpu:false ~limits:cpu_limits strided;
  traffic_floor_pins "floor, stride 2, gpu" ~is_gpu:true ~limits:gpu_limits strided;
  traffic_floor_pins "floor, f16 operands packed at f32, cpu" ~is_gpu:false ~limits:narrow_cpu
    narrow

(* Pruning: the walk on a memory-dominated envelope, with the incumbent strictly between the
   schedule-invariant floor's bound and the largest raised bound ([inc_max - 1] bytes above the
   floor). The increments the incumbent and the raised set are read from are the leaves' own
   ([expected_conv_inc] of their sketch parameters), never the floor under test, so a floor that
   prices nothing fails the fathoming claims themselves. Every leaf's real score sits far above that
   incumbent, so the walk's answer is [None] either way; what moves is the work: the leaves whose
   raised bound reaches the incumbent are fathomed unscored, where the zero increment scored them.
   The incumbent is placed rather than taken from a leaf so that this section isolates the
   increment's contribution (the walk against the same walk under a zero increment); pruning at real
   incumbents, which the windowed-read floor and the fragment term make possible (gh-ocannl-1601),
   is [real_site_pins]' claim below. *)
let pruning_pins name ~is_gpu ~limits base =
  let limits = envelope limits ~peak_flops:1e15 ~bandwidth:1e9 in
  let score segs =
    List.fold segs ~init:(Some 0.) ~f:(fun acc o ->
        Option.map2 acc (rank_with limits o) ~f:Float.( + ))
  in
  let score_p p = Option.bind (apply_leaf base p) ~f:(fun post -> score [ post ]) in
  let floor = CM.completion_floor base.LL.llc in
  let bound_at i =
    CM.roofline_seconds ?peak_flops:limits.peak_flops
      ?peak_memory_bandwidth:limits.peak_memory_bandwidth ~flops:floor.CM.fr_flops
      ~bytes:(floor.CM.fr_bytes + i) ()
  in
  match
    ( Autotune.detect_conv base.LL.llc,
      Autotune.sketch_family_tree ~is_gpu ~is_cpu:(not is_gpu) ~limits base )
  with
  | None, _ | _, None -> p (name ^ ": the conv family claims the site") false
  | Some site, Some tree ->
      let scored =
        List.filter_map (Sspace.enumerate tree) ~f:(fun (path, p) ->
            Option.map (score_p p) ~f:(fun s -> (path, p, s)))
      in
      let inc_max =
        List.fold scored ~init:0 ~f:(fun acc (_, p, _) -> max acc (expected_conv_inc site p))
      in
      (match (bound_at 0, bound_at (inc_max - 1)) with
      | Some b0, Some incumbent when Float.(incumbent > b0) ->
          let admitted = ref [] in
          let r =
            Autotune.model_family_search
              ~admit:(fun p ->
                admitted := p :: !admitted;
                true)
              ~capabilities:Ir.Backend_intf.no_codegen_capabilities ~is_gpu ~is_cpu:(not is_gpu)
              ~limits ~static_indices:[] ~peak_flops:limits.peak_flops
              ~peak_memory_bandwidth:limits.peak_memory_bandwidth ~score ~incumbent base
          in
          let reaches (_, p, _) =
            Option.exists (bound_at (expected_conv_inc site p)) ~f:(fun b -> Float.(b >= incumbent))
          in
          let raised = List.filter scored ~f:reaches in
          (* The old walk: the same tree and scores under a uniform zero increment. *)
          let _, zero_stats =
            Sspace.search ~bound:(fun ~path:_ _ -> bound_at 0) ~incumbent ~score:score_p tree
          in
          let st = match r with Some (_, st, _) -> st.Sspace.st_scored | None -> -1 in
          Stdio.printf
            "== %s: %d scored leaves, %d with a bound at the incumbent; the walk scores %d, the \
             zero-increment walk %d ==\n"
            name (List.length scored) (List.length raised) st zero_stats.Sspace.st_scored;
          p_exists (name ^ ": some leaf's raised bound reaches the incumbent") scored ~f:reaches;
          p_none (name ^ ": no leaf whose bound reaches the incumbent is admitted to scoring")
            raised ~f:(fun (_, p, _) -> List.mem !admitted p ~equal:Poly.equal);
          p
            (name ^ ": the walk scores fewer leaves than under the zero increment")
            (st >= 0 && st < zero_stats.Sspace.st_scored);
          p
            (name ^ ": the pick is the exhaustive answer below the incumbent")
            (match r with
            | Some (pick, _, _) ->
                let below =
                  List.fold scored ~init:None ~f:(fun acc (_, p, s) ->
                      if Float.(s >= incumbent) then acc
                      else match acc with Some (_, b) when Float.(s >= b) -> acc | _ -> Some (p, s))
                in
                Option.equal
                  (fun (p1, s1) (p2, s2) -> Poly.equal p1 p2 && Float.equal s1 s2)
                  pick below
            | None -> false)
      | _ -> p (name ^ ": a leaf prices above the floor, so an incumbent fits between") false);
      (* With an unbeatable incumbent the raised bound still finds the exhaustive first argmin. *)
      let exhaustive =
        List.fold scored ~init:None ~f:(fun acc (_, p, s) ->
            match acc with Some (_, b) when Float.(s >= b) -> acc | _ -> Some (p, s))
      in
      p
        (name ^ ": with an unbeatable incumbent the pick is the exhaustive first argmin")
        (match
           Autotune.model_family_search ~capabilities:Ir.Backend_intf.no_codegen_capabilities
             ~is_gpu ~is_cpu:(not is_gpu) ~limits ~static_indices:[] ~peak_flops:limits.peak_flops
             ~peak_memory_bandwidth:limits.peak_memory_bandwidth ~score ~incumbent:Float.infinity
             base
         with
        | Some (pick, _, _) ->
            Option.equal
              (fun (p1, s1) (p2, s2) -> Poly.equal p1 p2 && Float.equal s1 s2)
              pick exhaustive
        | None -> false)

let () =
  let dividing = lowering "cft_prune" (make_conv "pd" ~b:2 ~h:18 ~ic:8 ~oc:8) in
  pruning_pins "pruning, dividing 16-row, cpu" ~is_gpu:false ~limits:cpu_limits dividing;
  pruning_pins "pruning, dividing 16-row, gpu" ~is_gpu:true ~limits:gpu_limits dividing

(* gh-ocannl-1601: pruning at a real conv site, with real incumbents. cifar_resnet's stage-1 conv
   (benchmarks/runners/ocannl/bench_conv_model.ml's [resnet]: 3x3, 16 -> 16 channels, padded, a
   batch of 64 32x32 images, fed by the stem's relu and feeding its own, both materialized like the
   network's activations), and the segment the tuner itself enumerates candidates for
   ([Ll_test.tuner_segments] under each leg's target). The floor now prices the padded input's
   windowed read ([Affine.image_cardinality_lb]) and the increment the accumulator fragment, so a
   committed row geometry's bound reaches a real leaf's score. On each leg, under the
   memory-dominated stand-in envelope of [pruning_pins] (the GPU leg is the model's, under the 8x8x8
   stand-in tile — no GPU executes): the walk from an infinite incumbent and the walk from the best
   leaf's own score each fathom a subtree below the root, admit fewer leaves to scoring than the
   tree has, and agree with the exhaustive answer — the first argmin, which is the production tie
   rule here: [no_codegen_capabilities] gives no leaf an operand-load key, so a tie keeps the
   earlier leaf. *)
let real_site_pins name ~is_gpu ~limits ~target =
  let limits = envelope limits ~peak_flops:1e15 ~bandwidth:1e9 in
  let x0 =
    NTDSL.init ~l:"rsx" ~prec:Ir.Ops.single ~b:[ 64 ] ~o:[ 32; 32; 16 ]
      ~f:(Ll_test.weighted ~weights:[| 1; 1; 2; 3 |] ~modulus:7 ~offset:0. ~stride:1.)
      ()
  in
  (* The stem's activation, as in the network: computed in the routine, so its buffer takes the
     conv's padding margins. *)
  let%op x = relu x0 in
  Ir.Tnode.update_memory_mode x.Tensor.value Ir.Tnode.On_device (Site "99:test-setup");
  let conv =
    Nn_blocks.conv2d ~label:[ "rs1c1" ] ~kernel_size:3 ~use_padding:true ~out_channels:16 ()
  in
  let%op y = relu (conv x) in
  Ir.Tnode.update_memory_mode y.Tensor.value Ir.Tnode.On_device (Site "99:test-setup");
  let whole =
    (* The conv's kernel and bias are parameters: initialized into the context first, as a training
       run does, so the forward routine links against them. *)
    let captured = ref None in
    let ctx = Train.init_params (Context.auto ()) Ir.Indexing.Empty y in
    let _ctx, _routine =
      Context.compile
        ~name:("cft_resnet_" ^ if is_gpu then "gpu" else "cpu")
        ~lowered_transform:(fun opt ->
          captured := Some (scratch opt);
          [ opt ])
        ctx (Train.forward y) Ir.Indexing.Empty
    in
    Option.value_exn !captured
  in
  let segment =
    List.find_map
      (snd (Ll_test.tuner_segments ~target whole))
      ~f:(fun (_, pre, _, _) ->
        Option.some_if (Option.is_some (Autotune.detect_conv pre.LL.llc)) pre)
  in
  match segment with
  | None -> p (name ^ ": the tuner's segmentation has a conv segment") false
  | Some base -> (
      let score segs =
        List.fold segs ~init:(Some 0.) ~f:(fun acc o ->
            Option.map2 acc (rank_with limits o) ~f:Float.( + ))
      in
      let score_p p = Option.bind (apply_leaf base p) ~f:(fun post -> score [ post ]) in
      match Autotune.sketch_family_tree ~is_gpu ~is_cpu:(not is_gpu) ~limits base with
      | None -> p (name ^ ": the conv family claims the segment") false
      | Some tree ->
          let scored =
            List.filter_map (Sspace.enumerate tree) ~f:(fun (_, p) ->
                Option.map (score_p p) ~f:(fun s -> (p, s)))
          in
          let exhaustive ~incumbent =
            List.fold scored ~init:None ~f:(fun acc (p, s) ->
                if Float.(s >= incumbent) then acc
                else match acc with Some (_, b) when Float.(s >= b) -> acc | _ -> Some (p, s))
          in
          let best = Option.map (exhaustive ~incumbent:Float.infinity) ~f:snd in
          let walk ~incumbent =
            let admitted = ref 0 in
            let r =
              Autotune.model_family_search
                ~admit:(fun _ ->
                  Int.incr admitted;
                  true)
                ~capabilities:Ir.Backend_intf.no_codegen_capabilities ~is_gpu ~is_cpu:(not is_gpu)
                ~limits ~static_indices:[] ~peak_flops:limits.peak_flops
                ~peak_memory_bandwidth:limits.peak_memory_bandwidth ~score ~incumbent base
            in
            (r, !admitted)
          in
          Stdio.printf "== %s: %d leaves, %d scored ==\n" name
            (List.length (Sspace.leaves tree))
            (List.length scored);
          p_exists ~min:2 (name ^ ": the tree's leaves score") scored ~f:(fun _ -> true);
          List.iter
            [
              ("an infinite incumbent", Float.infinity);
              ("the best leaf's score", Option.value_exn best);
            ]
            ~f:(fun (what, incumbent) ->
              let r, admitted = walk ~incumbent in
              match r with
              | None -> p (name ^ ", from " ^ what ^ ": the walk runs") false
              | Some (pick, st, _) ->
                  Stdio.printf "  from %s: %d scored, %d fathomed, %d admitted of %d leaves\n" what
                    st.Sspace.st_scored st.Sspace.st_fathomed admitted (List.length scored);
                  p
                    (name ^ ", from " ^ what ^ ": a subtree below the root fathoms unscored")
                    (st.Sspace.st_fathomed > 0 && admitted < List.length scored);
                  p
                    (name ^ ", from " ^ what ^ ": the pick is the exhaustive answer")
                    (Option.equal
                       (fun (p1, s1) (p2, s2) -> Poly.equal p1 p2 && Float.equal s1 s2)
                       pick (exhaustive ~incumbent))))

let () =
  real_site_pins "cifar_resnet stage-1 conv, cpu" ~is_gpu:false ~limits:cpu_limits
    ~target:{ Ll_test.is_gpu = false; is_cpu = true; limits = cpu_limits };
  real_site_pins "cifar_resnet stage-1 conv, gpu" ~is_gpu:true ~limits:gpu_limits
    ~target:(Ll_test.synthetic_gpu ~limits:gpu_limits ())
