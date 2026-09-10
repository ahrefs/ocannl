(* gh-ocannl-514 phase 1: the matmul sketch family as a refinement tree.

   The flat sections below pin the family's seed enumeration — originally recorded against the
   hand-written enumeration before the tree refactor (which reproduced it list-for-list, order
   included: enumeration order reaches candidate timing order and dedup keep-first); the phase-2
   pre-compile refutations have since deliberately removed statically-doomed entries (e.g. the
   single-row-block whole-triple Grid form), pinned here as behavior. Synthetic limits keep every
   leg machine-independent — seeding is a pure function of the lowering, so the GPU legs enumerate
   (and twin: swizzle, pipeline depth) without any GPU present.

   The site is a 64x64x64 f32 matmul: every tile geometry in the curated menus divides it, so the
   menus enumerate in full. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Asgns = Ir.Assignments

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* Compile [tensor], capturing the base lowering for the (pure) seeding calls. *)
let with_lowering ~name tensor =
  let captured = ref None in
  let _ctx, _routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        captured := Some opt;
        [ opt ])
      (Context.auto ())
      (named name (Train.forward tensor))
      Ir.Indexing.Empty
  in
  Option.value_exn !captured

let cpu_limits = { Ir.Backend_intf.no_hardware_limits with simd_vector_bytes = 32 }
let f32 = Ir.Backend_intf.Mma_f32

let gpu_plain_limits =
  {
    Ir.Backend_intf.no_hardware_limits with
    mma =
      Some
        {
          Ir.Backend_intf.mma_simd_width = 32;
          mma_tile = (8, 8, 8);
          mma_format_tiles = [ ((f32, f32, f32), (8, 8, 8)) ];
          mma_f16_wide_acc_scopes = [];
          mma_staged_layouts = [];
          mma_pipeline_depths = [];
        };
  }

(* Staged-layout and pipelining advertisements switch on the twin seeding: a swizzled twin per
   staged seed whose tile rows span power-of-two 16-byte units, and a depth twin per fully dividing
   staged seed. *)
let gpu_full_limits =
  {
    Ir.Backend_intf.no_hardware_limits with
    mma =
      Some
        {
          Ir.Backend_intf.mma_simd_width = 32;
          mma_tile = (8, 8, 8);
          mma_format_tiles = [ ((f32, f32, f32), (8, 8, 8)) ];
          mma_f16_wide_acc_scopes = [];
          mma_staged_layouts = [ ((f32, f32, f32), Ir.Backend_intf.Mma_swizzled_b128) ];
          mma_pipeline_depths = [ 2 ];
        };
  }

let show (p : Autotune.sketch_params) =
  Printf.sprintf
    "gpu:%b mma:%b simd:%-2d bm:%-3d bn:%-3d bk:%-3d tm:%d tn:%d hoist:%b grid:%b packrest:%b \
     epi:%b swz:%s depth:%d%s"
    p.Autotune.sk_gpu p.Autotune.sk_mma p.Autotune.sk_simd p.Autotune.sk_bm p.Autotune.sk_bn
    p.Autotune.sk_bk p.Autotune.sk_tm p.Autotune.sk_tn p.Autotune.sk_hoist p.Autotune.sk_grid
    p.Autotune.sk_pack_rest p.Autotune.sk_epilogue
    (match p.Autotune.sk_swizzle with
    | None -> "-"
    | Some LL.Swizzle_b128 -> "b128"
    | Some LL.Swizzle_elem -> "elem")
    p.Autotune.sk_depth
    (* Appended only when set, so the f32 sections' lines are unchanged (gh-ocannl-575). *)
    (match p.Autotune.sk_pack_prec with
    | None -> ""
    | Some pr -> " packprec:" ^ Ir.Ops.prec_string pr)
  ^
  (* Likewise (gh-ocannl-619): only a schedule-carried geometry prints. *)
  match p.Autotune.sk_tile with None -> "" | Some t -> " tile:" ^ Ir.Register_tile.to_string t

let section name ~is_gpu ~is_cpu ~limits opt =
  let seeds = Autotune.sketch_seed_params ~is_gpu ~is_cpu ~limits opt in
  Stdio.printf "== %s: %d seeds ==\n" name (List.length seeds);
  List.iteri seeds ~f:(fun i p -> Stdio.printf "%2d  %s\n" i (show p));
  seeds

(* The refinement-tree view of the same family (gh-ocannl-514 phases 1-2): decision levels with
   commitment-dependent domains, whose leaves are exactly the flat enumeration above, and whose
   children carry verdicts — [refuted]/[excluded] branches print their witness and contribute no
   leaves, [unknown] branches never fathom. *)
module Sspace = Ir.Schedule_space
module FD = Autotune.Family_decision

(* Every label printed below is [FD.to_label] of the path's decision datum (gh-ocannl-591): the
   tree's identities are data, the strings are this rendering. *)
let lbl = FD.to_label

let rec print_tree ~indent tree =
  match tree with
  | Sspace.Leaf p -> Stdio.printf "%s* %s\n" indent (show p)
  | Sspace.Choice { level; children } ->
      if List.is_empty children then Stdio.printf "%s%s: infeasible\n" indent level
      else
        List.iter children ~f:(fun (d, child) ->
            let label = lbl d in
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

(* The three verdict collectors: a shape's pre-compilation decline explanations (gh-ocannl-479), the
   policy-suppressed branches a driver could re-propose, and the branches only candidate compilation
   settles. *)
let verdict_reports tree =
  let pp (path, w) = Stdio.printf "  %s: %s\n" (FD.render_path path) w in
  let section name entries =
    if not (List.is_empty entries) then (
      Stdio.printf "-- %s --\n" name;
      List.iter entries ~f:pp)
  in
  section "refuted" (Sspace.refutations tree);
  section "excluded" (Sspace.exclusions tree);
  section "unknown" (Sspace.unknowns tree)

let tree_section name ~is_gpu ~is_cpu ~limits opt seeds =
  match Autotune.matmul_sketch_tree ~is_gpu ~is_cpu ~limits opt with
  | None -> Stdio.printf "== %s tree: no site detected ==\n" name
  | Some tree ->
      Stdio.printf "== %s tree: %d choice nodes, depth %d ==\n" name (Sspace.count_choices tree)
        (Sspace.depth tree);
      print_tree ~indent:"" tree;
      let paths = Sspace.enumerate tree in
      (match List.last paths with
      | Some (path, _) -> Stdio.printf "last leaf's decision path: %s\n" (FD.render_path path)
      | None -> Stdio.printf "no leaves\n");
      Verdict.p "tree leaves = flat enumeration"
        (List.equal (fun a b -> String.equal (show a) (show b)) (Sspace.leaves tree) seeds)

(* gh-ocannl-591: what the certain-traffic increment of a leaf's path MUST be, derived from the
   leaf's own sketch parameters and NOTHING on the path — no decision, no label. Cross-checking it
   against [sketch_path_traffic_floor] over paths walked out of the real tree is what pins the
   decision protocol end to end: a commitment the floor stops reading shows up as a mismatch, where
   the string protocol's silent fall-through showed up as a uniform (sound, useless) zero. The two
   families count differently, exactly as [Cost_model.analyze] charges them: the GPU pipelines stage
   both operand tiles in kernel (written and read), while the CPU packed composition adds traffic
   only where it packs in kernel — every other shape packs at link time or per Grid chunk, replacing
   the original operand's reads rather than adding to them. *)
let expected_inc ~elt_bytes ~n_extent (p : Autotune.sketch_params) =
  let bm = p.Autotune.sk_bm and bk = p.Autotune.sk_bk in
  let bn = if p.Autotune.sk_bn = 0 then n_extent else p.Autotune.sk_bn in
  let tile = ((bm * bk) + (bk * bn)) * elt_bytes in
  if p.Autotune.sk_gpu then if bk > 0 then 2 * tile else 0
  else if
    p.Autotune.sk_mma && bk > 0
    && not (p.Autotune.sk_hoist || p.Autotune.sk_grid || p.Autotune.sk_pack_rest)
  then tile
  else 0

let prefixes path =
  List.folding_map path ~init:[] ~f:(fun acc entry ->
      let acc = acc @ [ entry ] in
      (acc, acc))

(* The traffic floor judged against the tree it is supposed to price (gh-ocannl-591): the paths come
   from [Sspace.enumerate] of the real tree, never from literals. A renamed level or a reworded
   label moves the rendered lines below and no number; a level ADDED, REMOVED or re-parameterized
   moves the counts and the per-leaf agreement. *)
let traffic_pins name ~limits ~elt_bytes ~n_extent tree opt =
  let inc = Autotune.sketch_path_traffic_floor ~limits opt in
  let paths = Sspace.enumerate tree in
  let priced = List.count paths ~f:(fun (path, _) -> inc path > 0) in
  let levels =
    List.dedup_and_sort ~compare:String.compare
      (List.concat_map paths ~f:(fun (path, _) -> List.map path ~f:fst))
  in
  Stdio.printf "== %s: certain-traffic increments over %d leaf paths ==\n" name (List.length paths);
  Stdio.printf "  levels on the leaf paths: %s\n" (String.concat ~sep:", " levels);
  Stdio.printf "  priced above the schedule-invariant floor: %d; max increment %d bytes\n" priced
    (List.fold paths ~init:0 ~f:(fun acc (path, _) -> max acc (inc path)));
  Verdict.p_all (name ^ ": every leaf's increment is the traffic its own parameters imply") paths
    ~f:(fun (path, p) -> inc path = expected_inc ~elt_bytes ~n_extent p);
  Verdict.p_all (name ^ ": the increment is monotone along every path's prefixes") paths
    ~f:(fun (path, _) ->
      let incs = 0 :: List.map (prefixes path) ~f:inc in
      List.is_sorted incs ~compare:Int.compare);
  Verdict.p
    (name ^ ": the staging commitments price above the floor")
    (priced > 0
    && priced = List.count paths ~f:(fun (_, p) -> expected_inc ~elt_bytes ~n_extent p > 0));
  Verdict.p_all (name ^ ": every leaf path commits its pipeline's shape level") paths
    ~f:(fun (path, _) ->
      List.exists path ~f:(fun (_, d) ->
          match d with FD.Geometry _ | FD.Row_block _ -> true | _ -> false));
  (* One representative path per pricing regime, walked out of the tree and rendered here (the
     rendering is all that a label reword can move). *)
  let sample what f =
    match List.find paths ~f:(fun (path, _) -> f path) with
    | None -> Stdio.printf "  %-26s (none in this tree)\n" what
    | Some (path, _) ->
        Stdio.printf "  %-26s %d bytes\n    %s\n" what (inc path) (FD.render_path path)
  in
  sample "staged geometry" (fun path ->
      inc path > 0
      && List.exists path ~f:(fun (_, d) ->
          match d with FD.Geometry (FD.Gpu_mma _ | FD.Cpu_packed _) -> true | _ -> false));
  sample "unpriced geometry" (fun path -> inc path = 0);
  sample "lattice leaf" (fun path ->
      List.exists path ~f:(fun (_, d) -> FD.equal d (FD.Geometry FD.Lattice)));
  (* The refinement chain of one lattice leaf: each interval commitment tightens the corner the box
     is priced at, so the increments grow monotonically down to the singleton. Pinning the chain off
     the tree replaces the hand-written box labels this test used to feed the parser. *)
  match
    (* The LAST lattice leaf: the largest corner, so the chain's tightening is visible — every box
       prices at its own minimum, which the refinement raises step by step. *)
    List.last
      (List.filter paths ~f:(fun (path, _) ->
           List.exists path ~f:(fun (_, d) -> FD.equal d (FD.Geometry FD.Lattice))))
  with
  | None -> ()
  | Some (path, _) ->
      Stdio.printf "  lattice refinement chain:\n";
      List.iter (prefixes path) ~f:(fun pre ->
          match List.last pre with
          | Some (_, d) when match d with FD.Lattice_box _ -> true | _ -> false ->
              Stdio.printf "    %-14s -> %d bytes\n" (FD.to_label d) (inc pre)
          | _ -> ())

(* Awkward sites (phase 2): the tree's verdicts explain, before any compilation, why branches
   propose nothing — where the flat enumeration silently dropped them. Only the collector reports
   print here; the leaves are still the seeds the flat API proposes. *)
let awkward_section name ~is_gpu ~is_cpu ~limits opt =
  match Autotune.matmul_sketch_tree ~is_gpu ~is_cpu ~limits opt with
  | None -> Stdio.printf "== %s: no site detected ==\n" name
  | Some tree ->
      let seeds = Autotune.sketch_seed_params ~is_gpu ~is_cpu ~limits opt in
      Stdio.printf "== %s: %d seeds ==\n" name (List.length seeds);
      Verdict.p "tree leaves = flat enumeration"
        (List.equal (fun a b -> String.equal (show a) (show b)) (Sspace.leaves tree) seeds);
      verdict_reports tree

(* The verdict collectors of a tree this test also prints in full: [tree_section] renders each
   witness inline at its branch, this renders it with the decision path that reached it. *)
let tree_verdicts name ~is_gpu ~is_cpu ~limits opt =
  match Autotune.matmul_sketch_tree ~is_gpu ~is_cpu ~limits opt with
  | None -> Stdio.printf "== %s verdicts: no site detected ==\n" name
  | Some tree ->
      Stdio.printf "== %s verdicts ==\n" name;
      verdict_reports tree

let () =
  let nn = 64 in
  (* A non-hoistable, B hoistable (host-init-backed constant): the exactly-one-hoistable case, so
     the hoisted, hoisted-grid AND mixed grid-outermost ([sk_pack_rest]) packing shapes all
     enumerate. *)
  let av =
    NTDSL.init ~l:"av" ~prec:Ir.Ops.single ~o:[ nn; nn ]
      ~f:(fun idcs -> (Float.of_int (((idcs.(0) * nn) + idcs.(1)) % 13) -. 6.) *. 0.25)
      ()
  in
  let bvv = Array.init (nn * nn) ~f:(fun x -> (Float.of_int (x % 17) -. 8.) *. 0.125) in
  let bv = TDSL.ndarray bvv ~label:[ "bv" ] ~output_dims:[ nn; nn ] () in
  let%op mm = av +* "ik;kj=>ij" bv in
  let opt = with_lowering ~name:"sft_mm" mm in
  let cpu_seeds = section "cpu simd32" ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt in
  let _ = section "gpu plain" ~is_gpu:true ~is_cpu:false ~limits:gpu_plain_limits opt in
  let gpu_seeds =
    section "gpu staged+depth" ~is_gpu:true ~is_cpu:false ~limits:gpu_full_limits opt
  in
  tree_section "cpu simd32" ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt cpu_seeds;
  tree_section "gpu staged+depth" ~is_gpu:true ~is_cpu:false ~limits:gpu_full_limits opt gpu_seeds;
  (* --- Awkward sites: witnesses for what is NOT proposed --- *)
  (* 20^3: no curated geometry divides it, so the pad composition decides which pipeline survives.
     Both GPU pipelines stage both operands and pad past 20 (gh-ocannl-485, gh-ocannl-730) — the
     blocktile family whole, the tensorized one at its staged geometries; the unstaged mma form
     reads its operands in place and is still refuted, as is the CPU blocktile pipeline, which packs
     into stack scratch outside that composition. CPU Grid shapes at bm=64 lack row blocks. *)
  let wa =
    NTDSL.init ~l:"wa" ~prec:Ir.Ops.single ~o:[ 20; 20 ]
      ~f:(fun idcs -> Float.of_int (((idcs.(0) * 20) + idcs.(1)) % 7) *. 0.5)
      ()
  in
  let wb =
    NTDSL.init ~l:"wb" ~prec:Ir.Ops.single ~o:[ 20; 20 ]
      ~f:(fun idcs -> Float.of_int (((idcs.(0) * 20) + idcs.(1)) % 5) -. 2.)
      ()
  in
  let%op awk = wa +* "ik;kj=>ij" wb in
  let opt_awk = with_lowering ~name:"sft_awk" awk in
  awkward_section "awkward 20^3 gpu" ~is_gpu:true ~is_cpu:false ~limits:gpu_full_limits opt_awk;
  awkward_section "awkward 20^3 cpu" ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt_awk;
  (* Half-precision operands (gh-ocannl-575): the seeding pre-filter resolves compute precision
     through the same [Numerics.cpu_compute_prec] the emission asks. Under the default
     [narrow_compute_f32] policy the site computes in f32, so the CPU tensorized families enumerate,
     with the packed seeds minting f32 panels ([sk_pack_prec], the packprec column); with the policy
     off the branch is refuted; with native-fp16 limits plus the [fp16_arithmetic] policy the seeds
     are pure-f16 (no packprec — panels stay half, at twice the lanes). The synthetic GPU capability
     still advertises only the f32 format triple, so the GPU leg stays a witness for "not
     proposed". *)
  let ha =
    NTDSL.init ~l:"ha" ~prec:Ir.Ops.half ~o:[ nn; nn ]
      ~f:(fun idcs -> Float.of_int (((idcs.(0) * nn) + idcs.(1)) % 7) *. 0.5)
      ()
  in
  let hb =
    NTDSL.init ~l:"hb" ~prec:Ir.Ops.half ~o:[ nn; nn ]
      ~f:(fun idcs -> Float.of_int (((idcs.(0) * nn) + idcs.(1)) % 5) -. 2.)
      ()
  in
  let%op hmm = ha +* "ik;kj=>ij" hb in
  let opt_h = with_lowering ~name:"sft_half" hmm in
  let _ = section "half-prec cpu seeds" ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt_h in
  awkward_section "half-prec cpu" ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt_h;
  awkward_section "half-prec gpu" ~is_gpu:true ~is_cpu:false ~limits:gpu_full_limits opt_h;
  let module Numerics = Ir.Numerics in
  let saved_policy = Numerics.get () in
  Numerics.set_policy { saved_policy with narrow_compute_f32 = false };
  awkward_section "half-prec cpu, narrow_compute_f32 off" ~is_gpu:false ~is_cpu:true
    ~limits:cpu_limits opt_h;
  Numerics.set_policy { saved_policy with fp16_arithmetic = Numerics.Fp16_narrow };
  let cpu_native_fp16_limits = { cpu_limits with native_fp16_arithmetic = true } in
  let _ =
    section "half-prec cpu seeds, native fp16" ~is_gpu:false ~is_cpu:true
      ~limits:cpu_native_fp16_limits opt_h
  in
  (* The policy alone must not flip the resolution: on a merely promoted target ([cpu_limits],
     native_fp16_arithmetic = false) fp16 still computes in f32 and the packed seeds keep their f32
     panels. *)
  let _ =
    section "half-prec cpu seeds, fp16 policy on promoted target" ~is_gpu:false ~is_cpu:true
      ~limits:cpu_limits opt_h
  in
  Numerics.set_policy saved_policy;
  (* gh-ocannl-680/836: the wide-f16 seeding gate is per emission scope. A GPU capability
     advertising the uniform-f16 triple seeds every scope under the default policy. Under
     [Fp16_wide], no wide scope removes every uniform-f16 mma seed; the per-statement scope alone
     keeps only eligible single-statement seeds (CUDA sm_80+); both scopes keep the full family (HIP
     since gh-ocannl-789 and Metal since gh-ocannl-837). The mixed [(f16, f16, f32)] triple is
     advertised too, standing in for the f32-storage destinations the gate must NOT touch — the gate
     keys on the DESTINATION's storage precision, which for [opt_h] is f16. *)
  let f16t = Ir.Backend_intf.Mma_f16 in
  let gpu_f16_limits ~wide_scopes =
    {
      Ir.Backend_intf.no_hardware_limits with
      mma =
        Some
          {
            Ir.Backend_intf.mma_simd_width = 32;
            mma_tile = (8, 8, 8);
            mma_format_tiles = [ ((f16t, f16t, f16t), (8, 8, 8)); ((f16t, f16t, f32), (8, 8, 8)) ];
            mma_f16_wide_acc_scopes = wide_scopes;
            mma_staged_layouts = [];
            mma_pipeline_depths = [];
          };
    }
  in
  let has_unstaged_mma seeds =
    List.exists seeds ~f:(fun p -> p.Autotune.sk_mma && p.Autotune.sk_bk = 0)
  in
  let has_staged_mma seeds =
    List.exists seeds ~f:(fun p -> p.Autotune.sk_mma && p.Autotune.sk_bk > 0)
  in
  let no_mma seeds =
    (not (List.is_empty seeds)) && List.for_all seeds ~f:(fun p -> not p.Autotune.sk_mma)
  in
  let no_staged_mma seeds =
    (not (List.is_empty seeds))
    && List.for_all seeds ~f:(fun p -> (not p.Autotune.sk_mma) || p.Autotune.sk_bk = 0)
  in
  let f16_default =
    section "half-prec gpu, f16 tiles, default policy" ~is_gpu:true ~is_cpu:false
      ~limits:(gpu_f16_limits ~wide_scopes:[]) opt_h
  in
  Numerics.set_policy { saved_policy with fp16_arithmetic = Numerics.Fp16_wide };
  let f16_wide_no_arm =
    section "half-prec gpu, f16 tiles, wide policy, no wide-accumulate scope" ~is_gpu:true
      ~is_cpu:false ~limits:(gpu_f16_limits ~wide_scopes:[]) opt_h
  in
  let f16_wide_no_arm_enablement, f16_wide_no_arm_disablement =
    Autotune.placement_enablement ~limits:(gpu_f16_limits ~wide_scopes:[]) ~static_indices:[]
      ~base:opt_h ~allmat:opt_h
  in
  let f16_wide_statement =
    section "half-prec gpu, f16 tiles, wide policy, per-statement scope only" ~is_gpu:true
      ~is_cpu:false
      ~limits:(gpu_f16_limits ~wide_scopes:[ Ir.Backend_intf.Mma_per_statement ])
      opt_h
  in
  let f16_wide_both =
    section "half-prec gpu, f16 tiles, wide policy, both emission scopes" ~is_gpu:true ~is_cpu:false
      ~limits:
        (gpu_f16_limits
           ~wide_scopes:[ Ir.Backend_intf.Mma_per_statement; Ir.Backend_intf.Mma_fragment_scope ])
      opt_h
  in
  (* A zero explicit k-block is only statement-scoped when the contraction has one axis. With an
     inherited outer contraction loop ([m_ko]), the tensorized accumulator survives across that loop
     even though [sk_bk = 0], so it needs the fragment-scope arm too. This multi-axis shape is the
     negative control for the old [bk]-only classification: a per-statement-only capability must not
     admit even its nominally unstaged MMA seeds. *)
  let check_wide_scope_edges () =
    Numerics.set_policy { saved_policy with fp16_arithmetic = Numerics.Fp16_wide };
    let per_statement = Ir.Backend_intf.Mma_per_statement in
    let fragment_scope = Ir.Backend_intf.Mma_fragment_scope in
    let round_tripped_scope =
      Ir.Backend_intf.mma_emission_scope_of_sexp
        (Ir.Backend_intf.sexp_of_mma_emission_scope per_statement)
    in
    let statement_limits = gpu_f16_limits ~wide_scopes:[ Ir.Backend_intf.Mma_per_statement ] in
    let both_limits =
      gpu_f16_limits
        ~wide_scopes:[ Ir.Backend_intf.Mma_per_statement; Ir.Backend_intf.Mma_fragment_scope ]
    in
    let seeds limits opt = Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits opt in
    (* A staged split whose padded block count is one does not extend the accumulator beyond its
       single Tile_mma statement. *)
    let one_block_a =
      NTDSL.init ~l:"one_block_a" ~prec:Ir.Ops.half ~o:[ 32; 16 ] ~f:(fun _ -> 0.25) ()
    in
    let one_block_b =
      NTDSL.init ~l:"one_block_b" ~prec:Ir.Ops.half ~o:[ 16; 32 ] ~f:(fun _ -> 0.25) ()
    in
    let%op one_block = one_block_a +* "ik;kj=>ij" one_block_b in
    let one_block_statement =
      seeds statement_limits (with_lowering ~name:"sft_one_block" one_block)
    in
    let whh = 2 and wee = 8 in
    let wide_outer_w =
      NTDSL.init ~l:"wide_outer_w" ~prec:Ir.Ops.half ~o:[ nn ] ~i:[ whh; wee ] ~f:(fun _ -> 0.25) ()
    in
    let wide_outer_x =
      NTDSL.init ~l:"wide_outer_x" ~prec:Ir.Ops.half ~b:[ 2; 16 ] ~o:[ whh; wee ]
        ~f:(fun _ -> 0.25)
        ()
    in
    let%op wide_outer = wide_outer_w * wide_outer_x in
    let opt_wide_outer = with_lowering ~name:"sft_wide_outer" wide_outer in
    let wide_outer_statement = seeds statement_limits opt_wide_outer in
    let wide_outer_both = seeds both_limits opt_wide_outer in
    Numerics.set_policy saved_policy;
    Verdict.p "MMA emission-scope serialization and comparison distinguish accumulator lifetimes"
      (Ir.Backend_intf.equal_mma_emission_scope round_tripped_scope per_statement
      && Ir.Backend_intf.compare_mma_emission_scope per_statement fragment_scope <> 0);
    Verdict.p "a one-block staged matmul uses the per-statement wide-f16 capability"
      (has_staged_mma one_block_statement);
    Verdict.p
      "a multi-axis contraction requires fragment-scope wide-f16 accumulation even with no \
       explicit k-block"
      (no_mma wide_outer_statement && has_unstaged_mma wide_outer_both
     && has_staged_mma wide_outer_both)
  in
  (* The CPU register tiling has the same wide-policy divergence case: under [Fp16_wide] with
     [narrow_compute_f32 = false] on a native-fp16 target, compute resolves half while the
     accumulator residency ([Numerics.cpu_accum_prec]) is f32, so the C-tile cannot honor it and
     seeding must omit the tensorized candidates — mirroring [C_syntax.try_register_tile]'s
     residency-divergence decline (Codex P1 round 1 on staging PR #477). *)
  Numerics.set_policy
    { saved_policy with fp16_arithmetic = Numerics.Fp16_wide; narrow_compute_f32 = false };
  let f16_cpu_wide_nco =
    section "half-prec cpu seeds, wide policy + narrow_compute_f32 off, native fp16" ~is_gpu:false
      ~is_cpu:true ~limits:cpu_native_fp16_limits opt_h
  in
  Numerics.set_policy saved_policy;
  Verdict.p
    "the wide-f16 policy withholds uniform-f16 mma seeds exactly in unsupported emission scopes"
    (has_unstaged_mma f16_default && has_staged_mma f16_default && no_mma f16_wide_no_arm
    && has_unstaged_mma f16_wide_statement
    && no_staged_mma f16_wide_statement && has_unstaged_mma f16_wide_both
    && has_staged_mma f16_wide_both);
  Verdict.p "placement eligibility omits a wide-f16 site with no applicable MMA scope"
    (Set.is_empty f16_wide_no_arm_enablement && Set.is_empty f16_wide_no_arm_disablement);
  Verdict.p
    "the wide-f16 policy under narrow_compute_f32 off omits the CPU register-tiled candidates \
     (accumulator residency diverges from compute)"
    (no_mma f16_cpu_wide_nco);
  (* Transposed B (k on its minor axis): whole-triple and the hoisted-only Grid shape read B in
     place, which the register tiling statically declines; packing shapes normalize the layout. *)
  let tb =
    NTDSL.init ~l:"tb" ~prec:Ir.Ops.single ~o:[ nn; nn ]
      ~f:(fun idcs -> Float.of_int (((idcs.(0) * nn) + idcs.(1)) % 11) *. 0.25)
      ()
  in
  let%op tmm = av +* "ik;jk=>ij" tb in
  let opt_t = with_lowering ~name:"sft_tb" tmm in
  awkward_section "transposed-B cpu" ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt_t;
  (* A companion nest that cannot follow the site's full arity (gh-ocannl-577): the matmul's output
     feeds a row reduction in the same routine — the lm_head max-logits pattern. The GPU
     companion-coverage rule (gh-521) then fails for {e every} tile completion, a fact decidable
     from the site and the lowering alone, so the tree refutes both pipelines at construction —
     where each candidate previously died at build. The refutation sits above the geometry menus and
     the lattice: lifting the lattice over the refuted family expands nothing and scores nothing
     (the gh-514 phase-6 finding this pins). CPU pipelines carry no kernel-global launch geometry
     and still enumerate. *)
  let%op cred = av +* "ik;kj=>ij" bv ++ "ij=>i" in
  let opt_c = with_lowering ~name:"sft_companion" cred in
  awkward_section "companion-reduction gpu" ~is_gpu:true ~is_cpu:false ~limits:gpu_full_limits opt_c;
  awkward_section "companion-reduction cpu" ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt_c;
  (match Autotune.matmul_sketch_tree ~is_gpu:true ~is_cpu:false ~limits:gpu_full_limits opt_c with
  | None -> Stdio.printf "companion-reduction: no site detected\n"
  | Some tree ->
      let lifted = Autotune.lift_geometry_lattice tree in
      let _best, stats = Sspace.search ~score:(fun _ -> Some Float.infinity) lifted in
      Stdio.printf
        "lifted-lattice search over the refuted family: %d expanded, %d scored, %d refuted\n"
        stats.Sspace.st_expanded stats.Sspace.st_scored stats.Sspace.st_refuted);
  (* A multi-axis contraction (gh-ocannl-683): attention's out projection [d[b,s,j] += w[j,h,e] *
     x[b,s,h,e]], whose weight carries two input axes, so lowering splits the contraction into an
     outer head loop ([m_ko]) above the innermost per-head loop, and only the latter's extent
     ([m_nk]) is what the tile gates judge. Every site above contracts over a single axis, where a
     tile's k-extent and the site's whole K coincide; here they do not, and a refutation says which
     one it means ([Sketch_families.k_extent_label]) — head_dim 12 divides neither blocktile menu's
     k-extents (GPU 8 and 16, CPU 16 and 8), and where the gate still applies its witness names
     "innermost contraction extent k=12 (of K=48 over 2 loops)" rather than a bare "k=12" that reads
     as the site's contraction. Where it applies is now the CPU blocktile pipeline alone: since
     gh-ocannl-730 the GPU blocktile family stages both operands like the tensorized one and PADS
     past the same 12, so both GPU pipelines enumerate here in full — the ten k-gate refutations
     this golden used to record are ten leaves. The unstaged tensorized form reads its operands in
     place and keeps the intrinsic-tile gate. The batch axes give the GPU pipelines the batch-grid
     level ([sk_batch_grid], gh-ocannl-528) that the rank-2 sites above never show. *)
  let bb = 2 and ss = 64 and jj = 64 and hh = 4 and ee = 12 in
  let ov =
    NTDSL.init ~l:"ov" ~prec:Ir.Ops.single ~o:[ jj ] ~i:[ hh; ee ]
      ~f:(fun idcs ->
        (Float.of_int (((((idcs.(0) * hh) + idcs.(1)) * ee) + idcs.(2)) % 11) -. 5.) *. 0.5)
      ()
  in
  let oa =
    NTDSL.init ~l:"oa" ~prec:Ir.Ops.single ~b:[ bb; ss ] ~o:[ hh; ee ]
      ~f:(fun idcs ->
        Float.of_int (((((idcs.(0) * ss) + idcs.(1)) * hh * ee) + (idcs.(2) * ee) + idcs.(3)) % 13)
        *. 0.25)
      ()
  in
  let%op oproj = ov * oa in
  let opt_o = with_lowering ~name:"sft_outproj" oproj in
  let o_cpu_seeds =
    section "out-projection cpu" ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt_o
  in
  let o_gpu_seeds =
    section "out-projection gpu staged+depth" ~is_gpu:true ~is_cpu:false ~limits:gpu_full_limits
      opt_o
  in
  tree_section "out-projection cpu" ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt_o o_cpu_seeds;
  tree_section "out-projection gpu staged+depth" ~is_gpu:true ~is_cpu:false ~limits:gpu_full_limits
    opt_o o_gpu_seeds;
  tree_verdicts "out-projection cpu" ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt_o;
  tree_verdicts "out-projection gpu staged+depth" ~is_gpu:true ~is_cpu:false ~limits:gpu_full_limits
    opt_o;
  (* The epilogue-fusion level (gh-ocannl-613): the family's root decision. Every site above feeds
     its matmul output to nothing fusable, so their fused flavor is refuted AT THE ROOT with the
     fusion recognizer's own reason — the first line of each "-- refuted --" report above, where
     before the flavor left no witness anywhere. Here the output feeds a bias-add + relu tail: the
     fused flavor is feasible and enumerates after every unfused leaf (the seeds-then-twins order
     candidate timing relies on), geometry for geometry, under its own construction-time verdicts —
     one tree, where a flag-flip over the unfused leaves or a second build per flavor used to mint
     the twins outside it. *)
  let fbias =
    TDSL.ndarray
      (Array.init nn ~f:(fun x -> Float.of_int (x % 3) *. 0.5))
      ~label:[ "fbias" ] ~output_dims:[ nn ] ()
  in
  let%op fprod = av +* "ik;kj=>ij" bv in
  let%op fmm = relu (fprod + fbias) in
  Train.set_materialized fprod.Tensor.value;
  let opt_f = with_lowering ~name:"sft_fusable" fmm in
  let fusable_section name ~is_gpu ~is_cpu ~limits opt =
    match Autotune.matmul_sketch_tree ~is_gpu ~is_cpu ~limits opt with
    | None -> Stdio.printf "== %s: no site detected ==\n" name
    | Some tree ->
        let seeds = Autotune.sketch_seed_params ~is_gpu ~is_cpu ~limits opt in
        let unfused, fused = List.partition_tf seeds ~f:(fun p -> not p.Autotune.sk_epilogue) in
        Stdio.printf "== %s: %d seeds (%d unfused + %d fused), %d choice nodes, depth %d ==\n" name
          (List.length seeds) (List.length unfused) (List.length fused) (Sspace.count_choices tree)
          (Sspace.depth tree);
        (match tree with
        | Sspace.Choice { children; _ }
          when List.for_all children ~f:(fun (d, _) ->
                   match d with FD.Fusion _ -> true | _ -> false) ->
            List.iter children ~f:(fun (d, child) ->
                Stdio.printf "fusion = %s%s\n" (lbl d)
                  (match child with
                  | Sspace.Child _ -> ""
                  | Sspace.Unknown (w, _) -> "  [unknown: " ^ w ^ "]"
                  | Sspace.Excluded (w, _) -> "  [excluded: " ^ w ^ "]"
                  | Sspace.Refuted w -> "  [refuted: " ^ w ^ "]"))
        | _ -> Stdio.printf "root is not the fusion level\n");
        Verdict.p "tree leaves = flat enumeration"
          (List.equal (fun a b -> String.equal (show a) (show b)) (Sspace.leaves tree) seeds);
        Verdict.p_alli "every fused leaf follows every unfused leaf" seeds ~f:(fun i p ->
            Bool.equal p.Autotune.sk_epilogue (i >= List.length unfused));
        Verdict.p "the fused flavor twins the unfused leaves geometry for geometry"
          ((not (List.is_empty fused))
          && List.equal
               (fun u f -> String.equal (show { u with Autotune.sk_epilogue = true }) (show f))
               unfused fused);
        (* The fused flavor's own verdicts carry its path — the same refutations and exclusions the
           unfused flavor has, plus nothing, on this fully-dividing site. *)
        let flavor_of (path, _) =
          List.find_map path ~f:(fun (_, d) ->
              match d with FD.Fusion f -> Some (FD.Fusion f) | _ -> None)
        in
        let count flavor entries =
          List.count entries ~f:(fun e -> Option.equal FD.equal (flavor_of e) (Some flavor))
        in
        Stdio.printf "refuted: %d unfused, %d fused; excluded: %d unfused, %d fused\n"
          (count (FD.Fusion `Unfused) (Sspace.refutations tree))
          (count (FD.Fusion `Fused) (Sspace.refutations tree))
          (count (FD.Fusion `Unfused) (Sspace.exclusions tree))
          (count (FD.Fusion `Fused) (Sspace.exclusions tree))
  in
  fusable_section "fusable tail gpu" ~is_gpu:true ~is_cpu:false ~limits:gpu_full_limits opt_f;
  fusable_section "fusable tail cpu" ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt_f;
  (* Tight hardware limits: the staged operand tiles are a sound workgroup-memory floor, so
     geometries whose depth-1 tiles exceed the cap refute outright and dividing geometries whose
     doubled tiles exceed it refute their depth twins; a blocktile geometry's launch size (bm/tm *
     bn/tn threads) is statically known, so the thread cap refutes bm64 bn64 tm4 tn4 (256 threads)
     at 128 — all pre-compile, where Schedule.check_hardware_limits_classified would otherwise
     reject candidate by candidate. 6144 bytes fits bm16 bn32 bk32 (2048 + 4096) exactly at depth
     1. *)
  let gpu_tight_limits =
    {
      gpu_full_limits with
      Ir.Backend_intf.max_workgroup_memory_bytes = Some 6144;
      max_threads_per_workgroup = Some 128;
      (* An advertised depth outside Schedule.apply_stage's implemented 1..2 range refutes rather
         than enumerating twins that fail every candidate compile. *)
      mma =
        Option.map gpu_full_limits.Ir.Backend_intf.mma ~f:(fun m ->
            { m with Ir.Backend_intf.mma_pipeline_depths = [ 2; 3 ] });
    }
  in
  awkward_section "gpu tight smem" ~is_gpu:true ~is_cpu:false ~limits:gpu_tight_limits opt;
  (* A wide output column extent: the unsplit B~ panel (bn = 0) of the (64, 0, 256) packed geometry
     spans 589824 bytes of tiles — above the 256 KiB stack/cache-economy threshold, which EXCLUDES
     (a driver may lift the policy) rather than refutes. *)
  let wn =
    NTDSL.init ~l:"wn" ~prec:Ir.Ops.single ~o:[ 64; 512 ]
      ~f:(fun idcs -> Float.of_int (((idcs.(0) * 512) + idcs.(1)) % 9) *. 0.25)
      ()
  in
  let%op wmm = av +* "ik;kj=>ij" wn in
  let opt_w = with_lowering ~name:"sft_wide" wmm in
  awkward_section "wide-N cpu" ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt_w;
  (* The lift operation: an Excluded child's payload is the same judgment with only that policy
     lifted, still subject to legality — the serial shape's economy-capped geometry recovers a leaf,
     while the Grid shapes' lifted payloads re-refute on the single-row-block rule. *)
  (match Autotune.matmul_sketch_tree ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt_w with
  | None -> Stdio.printf "wide-N: no site\n"
  | Some tree ->
      let rec lift_all t =
        match t with
        | Sspace.Leaf p -> Sspace.Leaf p
        | Sspace.Choice { level; children } ->
            Sspace.Choice
              {
                level;
                children =
                  List.map children ~f:(fun (l, c) ->
                      ( l,
                        match Sspace.lift_excluded c with
                        | Sspace.Child sub -> Sspace.Child (lazy (lift_all (Lazy.force sub)))
                        | c -> c ));
              }
      in
      Stdio.printf "wide-N leaves: %d; with exclusions lifted: %d\n"
        (List.length (Sspace.leaves tree))
        (List.length (Sspace.leaves (lift_all tree))));
  (* --- The B&B driver's contract on a synthetic tree (phase 4): the bound fathoms Child subtrees
     against the tightening threshold (equality fathoms — displacement needs strict improvement),
     Unknown children are NEVER fathomed even when their bound would, Refuted and Excluded children
     are never entered (an excluded 0-cost leaf must not leak into the minimum), and without a bound
     the walk degenerates to the flat first-best scan. --- *)
  let leafv name v = Sspace.Child (lazy (Sspace.Leaf (name, v))) in
  let choice level children = Sspace.Choice { level; children } in
  let syn =
    choice "top"
      [
        ("a", leafv "a" 5.);
        ("b", Sspace.Child (lazy (choice "bsub" [ ("b1", leafv "b1" 3.); ("b2", leafv "b2" 2.) ])));
        ("u", Sspace.Unknown ("compile-settled", lazy (choice "usub" [ ("u1", leafv "u1" 1.) ])));
        ("r", Sspace.Refuted "illegal");
        ("x", Sspace.Excluded ("policy", lazy (leafv "x" 0.)));
        ("c", Sspace.Child (lazy (choice "csub" [ ("c1", leafv "c1" 2.) ])));
      ]
  in
  let bound ~path:_ sub =
    match sub with
    | Sspace.Choice { level = "bsub"; _ } -> Some 2.5
    | Sspace.Choice { level = "usub"; _ } -> Some 100.
    | Sspace.Choice { level = "csub"; _ } -> Some 10.
    | _ -> None
  in
  let score (_, v) = Some v in
  let show_run name (best, stats) =
    Stdio.printf "%s: best=%s stats=%s\n" name
      (match best with Some ((n, _), v) -> Printf.sprintf "%s@%.1f" n v | None -> "none")
      (Sexp.to_string_hum (Sspace.sexp_of_search_stats stats))
  in
  show_run "search with bounds" (Sspace.search ~bound ~incumbent:5.5 ~score syn);
  show_run "search without bounds" (Sspace.search ~incumbent:5.5 ~score syn);
  show_run "ties: first best wins"
    (Sspace.search ~score (choice "tie" [ ("t1", leafv "t1" 2.); ("t2", leafv "t2" 2.) ]));
  (* Cost-fathoming vs verdict-fathoming: a bounded Child subtree CONTAINING an Unknown below is
     correctly fathomed by cost — bound soundness quantifies over every completion independently of
     how verdicts resolve, so a subtree that cannot beat the incumbent even where legal is pruned.
     Only a directly encountered Unknown skips the bound (its dominant unknown is legality, where a
     bound over possibly-nonexistent completions is vacuous). *)
  let nested =
    choice "top"
      [
        ("good", leafv "good" 3.);
        ( "deep",
          Sspace.Child
            (lazy
              (choice "dsub"
                 [ ("du", Sspace.Unknown ("compile-settled", lazy (Sspace.Leaf ("du", 1.)))) ])) );
      ]
  in
  let bound ~path:_ = function Sspace.Choice { level = "dsub"; _ } -> Some 5. | _ -> None in
  show_run "cost bound fathoms above a nested Unknown" (Sspace.search ~bound ~score nested);
  (* Path-dependent bounds (gh-ocannl-514, the placement-space search): the bound receives the
     committed (level, label) vector down to and including the judged child — the partial vector the
     subtree stands for — so a floor that tightens as commitments accumulate can price the exact
     node being judged. Here the bound prices by the committed labels alone: committing "mat" costs
     4 per level, so the mat/mat subtree (floor 8) is fathomed against the incumbent 7 while
     mat/keep (floor 4) and keep/* (floor 0) are entered. (The fathomed leaf scores 1 — this
     synthetic bound is deliberately dishonest about it, pinning the mechanics and the documented
     caveat that an unsound bound prunes true winners.) *)
  let placementish =
    choice "n1"
      [
        ( "mat",
          Sspace.Child (lazy (choice "n2" [ ("mat", leafv "mm" 1.); ("keep", leafv "mk" 6.) ])) );
        ( "keep",
          Sspace.Child (lazy (choice "n2" [ ("mat", leafv "km" 6.5); ("keep", leafv "kk" 6.9) ])) );
      ]
  in
  let bound ~path _sub =
    Some (4. *. Float.of_int (List.count path ~f:(fun (_, label) -> String.equal label "mat")))
  in
  show_run "path-dependent bound fathoms the doubly-committed subtree"
    (Sspace.search ~bound ~incumbent:7. ~score placementish);
  (* --- Phase 5: the tile-size lattice and the non-uniform family bound --- *)
  (* The lattice hides behind its exclusion: the un-lifted tree keeps the curated leaves (the
     seed-list identity above already pinned that), and lifting turns the exclusion into interval
     boxes over every intrinsic-tile multiple — 8 bm values x 8 staged bk values on the 64^3
     site — whose leaves join the enumeration without disturbing the curated ones' order. *)
  let mm2 =
    (* A fresh lowering of the same 64^3 site: [opt] is still live above, but a dedicated name keeps
       this section's output self-contained. *)
    with_lowering ~name:"sft_lattice"
      (let%op l2 = av +* "ik;kj=>ij" bv in
       l2)
  in
  (match Autotune.matmul_sketch_tree ~is_gpu:true ~is_cpu:false ~limits:gpu_full_limits mm2 with
  | None -> Stdio.printf "lattice: no site detected\n"
  | Some tree ->
      let curated = List.length (Sspace.leaves tree) in
      let lifted = Autotune.lift_geometry_lattice tree in
      let lifted_leaves = List.length (Sspace.leaves lifted) in
      Stdio.printf "== lattice (64^3, tile 8x8x8) ==\n";
      Stdio.printf "curated leaves %d, lifted leaves %d (+%d lattice singletons), depth %d -> %d\n"
        curated lifted_leaves (lifted_leaves - curated) (Sspace.depth tree) (Sspace.depth lifted);
      Verdict.p "the lattice exclusion carries the lift instructions"
        (List.exists (Sspace.exclusions tree) ~f:(fun (_, w) ->
             String.equal w Autotune.geometry_lattice_witness));
      (* The certain-traffic increments that make the bound non-uniform: a committed staged geometry
         prices its operand tiles, an unstaged one prices nothing, and a lattice box prices its most
         favorable corner — monotone in refinement. Judged over the LIFTED tree's own paths
         (gh-ocannl-591), so the pin moves with the tree rather than with a list of literals this
         test wrote for itself. *)
      let inc = Autotune.sketch_path_traffic_floor ~limits:gpu_full_limits mm2 in
      traffic_pins "gpu staged lattice" ~limits:gpu_full_limits ~elt_bytes:4 ~n_extent:64 lifted mm2;
      (* Search over the lifted tree with the increment itself as the bound and an incumbent between
         the small and large boxes' floors: large-tile boxes fathom without expansion, so the walk
         scores a fraction of the lattice — the logarithmic-effective regime. The score never
         displaces (infinity), pinning that fathoming alone dispatches the boxes. *)
      let bound ~path _sub = Some (Float.of_int (inc path)) in
      let score _p = Some Float.infinity in
      let _best, stats = Sspace.search ~bound ~incumbent:6000. ~score lifted in
      Stdio.printf
        "lifted search at incumbent 6000: %d expanded, %d scored, %d fathomed, %d refuted, %d \
         excluded\n"
        stats.Sspace.st_expanded stats.Sspace.st_scored stats.Sspace.st_fathomed
        stats.Sspace.st_refuted stats.Sspace.st_excluded);
  (* The CPU side of the same protocol, which no test priced before (gh-ocannl-591): the packed
     composition's traffic depends on the packing shape ABOVE its geometry — only in-kernel [serial]
     packing adds panel bytes, the hoisted and Grid shapes replace or split reads — and the
     blocktile pipeline stages nothing at all. Same tree-walked judgment, so the CPU arms of the
     floor are pinned by the tree rather than by a literal path. *)
  (match Autotune.matmul_sketch_tree ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt with
  | None -> Stdio.printf "cpu traffic: no site detected\n"
  | Some tree -> traffic_pins "cpu simd32" ~limits:cpu_limits ~elt_bytes:4 ~n_extent:64 tree opt);
  check_wide_scope_edges ();
  (* Corner-judged box refutations: a workgroup-memory cap that admits only the smallest staged
     tiles refutes the large-tile half-boxes at their most favorable corner, pre-expansion — the
     "tile-size interval whose minimum footprint exceeds shared memory" fathom of the issue. *)
  let gpu_tiny_smem =
    { gpu_full_limits with Ir.Backend_intf.max_workgroup_memory_bytes = Some 2048 }
  in
  match Autotune.matmul_sketch_tree ~is_gpu:true ~is_cpu:false ~limits:gpu_tiny_smem mm2 with
  | None -> Stdio.printf "lattice under 2KB smem: no site detected\n"
  | Some tree -> (
      let lifted = Autotune.lift_geometry_lattice tree in
      let box_refutations =
        List.filter (Sspace.refutations lifted) ~f:(fun (path, _) ->
            List.exists path ~f:(fun (_, d) ->
                match d with FD.Lattice_box { lb_lo; lb_hi; _ } -> lb_lo <> lb_hi | _ -> false)
            && List.exists path ~f:(fun (_, d) -> FD.equal d (FD.Geometry FD.Lattice)))
      in
      Stdio.printf "== lattice under a 2048-byte workgroup-memory cap ==\n";
      Stdio.printf "surviving lattice leaves %d; box-level refutations %d, e.g.:\n"
        (List.length
           (List.filter (Sspace.enumerate lifted) ~f:(fun (path, _) ->
                List.exists path ~f:(fun (_, d) -> FD.equal d (FD.Geometry FD.Lattice)))))
        (List.length box_refutations);
      List.iter (List.take box_refutations 2) ~f:(fun (path, w) ->
          Stdio.printf "  %s: %s\n" (FD.render_path path) w);
      (* Review round (Codex P1 on PR #327): the lattice minima and the open-corner pricing must
         come from the same per-format tile selection the tree builds with — a canonical [mma_tile]
         coarser than the selected format's (CUDA's TF32 shape) must not inflate the open-axis
         corner. Canonical 16x16x16, f32 format tile 8x8x8: the open-corner increment prices 8s, and
         the lattice enumerates 8-step multiples. *)
      let gpu_coarse_canonical =
        {
          Ir.Backend_intf.no_hardware_limits with
          mma =
            Some
              {
                Ir.Backend_intf.mma_simd_width = 32;
                mma_tile = (16, 16, 16);
                mma_format_tiles = [ ((f32, f32, f32), (8, 8, 8)) ];
                mma_f16_wide_acc_scopes = [];
                mma_staged_layouts = [];
                mma_pipeline_depths = [];
              };
        }
      in
      match
        Autotune.matmul_sketch_tree ~is_gpu:true ~is_cpu:false ~limits:gpu_coarse_canonical mm2
      with
      | None -> Stdio.printf "coarse-canonical: no site detected\n"
      | Some tree ->
          let lifted = Autotune.lift_geometry_lattice tree in
          let lattice_leaves =
            List.filter (Sspace.enumerate lifted) ~f:(fun (path, _) ->
                List.exists path ~f:(fun (_, d) -> FD.equal d (FD.Geometry FD.Lattice)))
          in
          let inc = Autotune.sketch_path_traffic_floor ~limits:gpu_coarse_canonical mm2 in
          Stdio.printf "== coarse canonical mma_tile 16^3, format tile 8^3 ==\n";
          Verdict.pf "lattice leaves %d, all 8-step multiples of both axes"
            (List.length lattice_leaves)
            (List.for_all lattice_leaves ~f:(fun (_, p) ->
                 p.Autotune.sk_bm % 8 = 0 && p.Autotune.sk_bk % 8 = 0)
            && List.exists lattice_leaves ~f:(fun (_, p) -> p.Autotune.sk_bm = 8)
            && List.exists lattice_leaves ~f:(fun (_, p) -> p.Autotune.sk_bk = 8));
          Stdio.printf
            "  open-corner lattice increment (format 8s, not canonical 16s) -> %d bytes\n"
            (inc
               [
                 (FD.level (FD.Pipeline `Tensorized), FD.Pipeline `Tensorized);
                 (FD.level (FD.Geometry FD.Lattice), FD.Geometry FD.Lattice);
               ]))
