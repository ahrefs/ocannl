(* gh-ocannl-1134: a [Scan_loop] nest sharing a GPU sketch's kernel is a companion like any other.

   A GPU matmul sketch builds hardware geometry for its accumulation nest, and launch dimensions are
   kernel-global, so every other nest of the kernel writing a materialized node must be nested under
   loops covering the same hardware slots ([Low_level.validate_parallel]): the companion-coverage
   rule of gh-ocannl-521, whose construction-time verdict ([companion_geometry]) the family tree
   refutes on (gh-ocannl-577). That verdict's walk did not enter a [Scan_loop], so a scan nest's
   writes were missing from its demand: a segment holding a matmul and a conflict-free scan (fission
   merges conflict-free nests) passed the witness and seeded every sketch, and every one of them
   died at [validate_parallel] on the scan's row write. The stage-1 tuned arm of gh-ocannl-1003 lost
   its tensorized score candidates that way, on the two-pass online softmax's [max_vals]. Now the
   scan nest is in the demand. Where its enclosing loops align with the site's chain it takes the
   site's per-position geometry, the scan running serially per thread with its carried state in
   registers. Where they do not -- a scan over the site's own column axis, the attention's -- the
   family is refuted with the scan's node named, and the finer [arity_cuts] fission (gh-ocannl-574)
   cuts the scan into its own kernel.

   Legs; the structural ones use synthetic device limits, so they run on every backend:

   1. Aligned: a matmul [z] and an independent per-cell scan over a [32 x 32] grid share a fission
   segment, and its seeds, tensorized ones among them, all construct and pass [validate_parallel].
   2. Misaligned: the scan's grid is [32 x 8]. The shared segment's family is refuted naming the
   scan's node and seeds nothing; the [arity_cuts] segmentation frees the matmul, whose seeds all
   validate. 3. The two-pass online-softmax attention: the score contraction's default GPU segment
   holds no scan, and its seeds, tensorized ones among them, all validate. Under the legality rules
   alone (no schedule-aware [keep_mapping]) the contraction shares the scan's segment, which is
   refuted on coverage rather than seeded, and the [arity_cuts] segmentation frees it. 4. Executed
   (GPU backends, real limits): every GPU seed of leg 1's shared segment gives [z] and the softmax
   the scan feeds as the untuned pipeline does, and where the backend advertises the tile, its
   tensorized seeds render tensorized. 5. Executed (GPU backends): every GPU seed of the attention's
   score segment gives the untuned forward's output. Device floats stay off the golden: the claims
   are two-sided comparisons, bitwise for leg 4's [z] and within a tolerance for the softmax
   outputs. *)

open Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Tn = Ir.Tnode

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let is_gpu = Sched.backend_is_gpu backend_name

(* The backend's accumulator residency, which a [Privatize] tile is minted at (gh-ocannl-1116). *)
let accum_prec = Ll_test.backend_accum_prec

let scratch (o : LL.optimized) =
  {
    o with
    LL.traced_store = Hashtbl.copy o.LL.traced_store;
    LL.optimize_ctx = LL.copy_optimize_ctx o.LL.optimize_ctx;
  }

(* Synthetic GPU limits, so the structural legs are backend-independent: 8x8x8 tiles in the (f16,
   f16, f32) format every tensorizing GPU backend advertises, for the independent matmul, and in
   f32, for the attention's scores. *)
let synthetic_limits =
  {
    Ir.Backend_intf.no_hardware_limits with
    mma =
      Some
        {
          Ir.Backend_intf.minimal_mma_capability with
          mma_tile = (8, 8, 8);
          mma_format_tiles =
            [
              (Ir.Backend_intf.(Mma_f16, Mma_f16, Mma_f32), (8, 8, 8));
              (Ir.Backend_intf.(Mma_f32, Mma_f32, Mma_f32), (8, 8, 8));
            ];
        };
  }

let has_scan (llc : LL.t) =
  Ll_test.count_stmt ~f:(function LL.Scan_loop _ -> true | _ -> false) llc > 0

let mentions s ~sub = String.is_substring s ~substring:sub

(* The tuner's GPU fission settings ([Autotune.tune]'s segment enumeration): promoted locals, folded
   zeros, and the schedule-aware cut probe unless [~keep_mapping:false] asks for the legality rules
   alone. [preset] schedules each [`Normal] segment. *)
let fission ?(keep_mapping = true) ?(arity_cuts = false) ?preset ~limits opt =
  let preset =
    Option.value preset ~default:(fun seg -> Sched.default_gpu ~min_parallel:1 ~limits seg)
  in
  Sched.fission_scheduled ~promote_locals:true ~fold_zeros:true ~arity_cuts
    ?keep_mapping:(if keep_mapping then Sched.fission_keep_mapping ~is_gpu:true ~limits else None)
    ~preset ~zero_sched:(Sched.zero_expansion ~limits) ~static_indices:[] opt

(* The [`Normal] segments' pre-schedule slices. *)
let normal_segments ?keep_mapping ?arity_cuts ~limits opt =
  List.filter_map
    (fission ?keep_mapping ?arity_cuts ~limits (scratch opt))
    ~f:(fun (kind, pre, _, _) -> match kind with `Normal -> Some pre | `Zeros | `Solo -> None)

let gpu_seeds ~limits seg = Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits seg

(* The matmul site a segment's family claims, with the family tree. *)
let site_of ~limits seg =
  match fst (Autotune.family_seeding ~is_gpu:true ~is_cpu:false ~limits seg) with
  | Autotune.Seeding_matmul { site; tree } -> Some (site, tree)
  | Autotune.Seeding_conv _ | Autotune.Seeding_none -> None

(* The segment whose matmul site writes a node [is_site] accepts. *)
let site_segment ~limits ~is_site segs =
  List.find segs ~f:(fun seg ->
      Option.exists (site_of ~limits seg) ~f:(fun ((site : Autotune.matmul_site), _) ->
          is_site site.Autotune.m_d))

(* Whether a seed's sketch schedule constructs, applies and passes [validate_parallel]; a failure
   goes to stderr. *)
let validates ~label seg sp =
  match
    let s = scratch seg in
    let post = Sched.apply (Autotune.sketch_schedule ~accum_prec ~p:sp s) s in
    LL.validate_parallel post.LL.optimize_ctx.placements post.LL.llc
  with
  | () -> true
  | exception exn ->
      Stdio.eprintf "%s: seed failed: %s\n" label (Exn.to_string exn);
      false

let coverage_refutations seg ~limits =
  Option.value_map (site_of ~limits seg) ~default:[] ~f:(fun (_, tree) ->
      List.filter_map (Ir.Schedule_space.refutations tree) ~f:(fun (_, w) ->
          Option.some_if (mentions w ~sub:"companion coverage (gh-521)") w))

(* The routine's whole lowering, captured on its way through a compile. The attention's parameters
   are not initialized here, so linking may fail after the capture; only the capture matters. *)
let lowering ~name comp =
  let captured = ref None in
  (try
     ignore
       (Context.compile ~name
          ~lowered_transform:(fun opt ->
            captured := Some (scratch opt);
            [ opt ])
          (Context.auto ()) comp Ir.Indexing.Empty)
   with _ -> ());
  Option.value_exn ~message:"the lowering was captured" !captured

let close ~tol g w = Float.(abs (g -. w) <= tol *. max 1. (abs w))

(* Compile [comp] with [transform] (none: the untuned pipeline) and read [outs] back, with the
   routine's mma census. *)
let execute ?transform ~name ~init comp outs =
  let ctx = init (Context.auto ()) in
  let ctx, routine =
    match transform with
    | None -> Context.compile ~name ctx comp Ir.Indexing.Empty
    | Some f -> Context.compile ~name ~lowered_transform:f ctx comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  (List.map outs ~f:(fun (t : Tensor.t) -> Context.get_values ctx t.Tensor.value), routine.mma)

(* The per-seed executed claim's verdict for one seed: [preset] is applied on the target segment
   only, and a run that never reached it is a failure, not a pass. *)
let run_seed ~label ~idx ~limits ~keep_mapping ~is_target ~build ~reference =
  let applied = ref false in
  let transform opt =
    let preset seg =
      if is_target seg then
        match List.nth (gpu_seeds ~limits seg) idx with
        | Some sp ->
            applied := true;
            Autotune.sketch_schedule ~accum_prec ~p:sp seg
        | None -> failwith (Printf.sprintf "%s: seed %d is gone from the target segment" label idx)
      else Sched.default_gpu ~min_parallel:1 ~limits seg
    in
    List.map (fission ~keep_mapping ~preset ~limits opt) ~f:(fun (_, _, _, post) -> post)
  in
  match build ~transform with
  | got, census ->
      let ok =
        !applied
        && List.for_all2_exn got reference ~f:(fun (g, tol) (w, _) ->
            (not (Array.is_empty g)) && Array.for_all2_exn g w ~f:(close ~tol))
      in
      if not ok then (
        Stdio.eprintf "%s: seed %d applied=%b, values differ\n" label idx !applied;
        List.iteri (List.zip_exn got reference) ~f:(fun o ((g, _), (w, _)) ->
            let worst = ref (-1, 0.) in
            Array.iteri g ~f:(fun i gv ->
                let d = Float.abs (gv -. w.(i)) in
                if Float.(d > snd !worst) then worst := (i, d));
            let i, d = !worst in
            if i >= 0 then
              Stdio.eprintf "  output %d: worst cell %d of %d, got %.9g want %.9g (diff %.3g)\n" o i
                (Array.length g) g.(i) w.(i) d));
      (ok, Some census)
  | exception exn ->
      Stdio.eprintf "%s: seed %d FAILED %s\n" label idx (Exn.to_string exn);
      (false, None)

(* === The independent scan: a half-precision matmul [z] (f32 destination) beside a softmax whose
   online-softmax rewrite is one [Scan_loop] per cell of a [32 x cols] grid, writing the row max and
   normalizer it carries. The two share no node, so fission merges their nests unconditionally.
   [z]'s operands are small NONNEGATIVE integers (0..6 against 0..4), so every product and partial
   sum is exact in f32 and of one sign, which is what makes gfx11's WMMA exactly rounded
   (gh-ocannl-1420): leg 4 compares [z] bitwise on every backend (gh-ocannl-1432). *)
let n = 32
let kk = 32
let red = 24

let indep ~tag ~cols =
  Tensor.unsafe_reinitialize ();
  let a =
    NTDSL.init ~l:(tag ^ "_a") ~prec:Ir.Ops.half ~b:[] ~i:[] ~o:[ n; kk ]
      ~f:(Ll_test.weighted ~weights:[| 3; 1 |] ~modulus:7 ~offset:0. ~stride:1.)
      ()
  in
  let b =
    NTDSL.init ~l:(tag ^ "_b") ~prec:Ir.Ops.half ~b:[] ~i:[] ~o:[ kk; n ]
      ~f:(Ll_test.weighted ~weights:[| 1; 2 |] ~modulus:5 ~offset:0. ~stride:1.)
      ()
  in
  let%op z = a +* "ik;kj=>ij" b in
  Tn.update_prec z.Tensor.value Ir.Ops.single;
  Train.set_materialized z.Tensor.value;
  let x =
    NTDSL.init ~l:(tag ^ "_x") ~prec:Ir.Ops.single ~b:[ n ] ~i:[ red ] ~o:[ cols ]
      ~f:(Ll_test.weighted ~weights:[| 5; 3; 1 |] ~modulus:11 ~offset:(-5.) ~stride:0.125)
      ()
  in
  let sm = Nn_blocks.softmax ~spec:" ... | r -> ..." () x in
  (* The scan's row max and normalizer stay buffers whatever kernel boundaries fission draws, as in
     the materialize-all training recipe -- otherwise a segmentation keeping the softmax beside the
     scan leaves them routine-local scratch, which needs no covering. *)
  Train.every_non_literal_materialized sm;
  (Ir.Assignments.sequence [ Train.forward z; Train.forward sm ], a, b, z, sm)

let is_node (t : Tensor.t) tn = Tn.equal tn t.Tensor.value

(* === The two-pass online-softmax attention at a head width above the recompute cap, so the scores
   are stored and read by the scan: the gh-ocannl-1003 stage-1 shape. [seq] is the only extent of
   64, so the score node is the one with two such axes. *)
let batch = 2
let seq = 64
let d_model = 48
let heads = 2
let d_k = 32

let attention () =
  Tensor.unsafe_reinitialize ();
  let x =
    TDSL.range_of_shape ~label:[ "x" ] ~batch_dims:[ batch; seq ] ~input_dims:[]
      ~output_dims:[ d_model ] ()
  in
  let%op x = x /. 8192. in
  let mask =
    NTDSL.init ~l:"mask" ~prec:Ir.Ops.single ~b:[ seq ] ~i:[ seq ] ~o:[]
      ~f:(function [| s; t |] -> if s >= t then 1. else 0. | _ -> assert false)
      ()
  in
  let block = Nn_blocks.multi_head_attention ~label:[ "attn" ] ~num_heads:heads ~d_k ~d_v:d_k () in
  let%op y = x + block ~train_step:None ~mask x in
  y

let is_scores tn = Array.count (Lazy.force tn.Tn.dims) ~f:(fun d -> d = seq) >= 2

let () =
  Stdio.eprintf "backend: %s (not part of the golden)\n%!" backend_name;
  Ir.Online_softmax.set_enabled (Some true);
  Ir.Online_softmax.set_block (Some 0);
  let limits = synthetic_limits in

  Stdio.printf "--- leg 1: a scan aligned with the matmul's chain shares its kernel ---\n";
  let comp, _, _, z, _ = indep ~tag:"al" ~cols:n in
  let opt = lowering ~name:"scan_cov_al" comp in
  let seg = site_segment ~limits ~is_site:(is_node z) (normal_segments ~limits opt) in
  p "aligned: the matmul's segment also holds the scan"
    (Option.exists seg ~f:(fun seg -> has_scan seg.LL.llc));
  let seeds = Option.value_map seg ~default:[] ~f:(gpu_seeds ~limits) in
  p_exists "aligned: the shared segment seeds a tensorized candidate" seeds ~f:(fun sp ->
      sp.Autotune.sk_mma);
  p_all "aligned: every seed of the shared segment passes validate_parallel" seeds ~f:(fun sp ->
      validates ~label:"aligned" (Option.value_exn seg) sp);

  Stdio.printf "--- leg 2: a scan the matmul's chain cannot cover is refuted, then cut ---\n";
  let comp, _, _, z, _ = indep ~tag:"mis" ~cols:8 in
  let opt = lowering ~name:"scan_cov_mis" comp in
  let coarse = site_segment ~limits ~is_site:(is_node z) (normal_segments ~limits opt) in
  p "misaligned: the matmul's segment also holds the scan"
    (Option.exists coarse ~f:(fun seg -> has_scan seg.LL.llc));
  p "misaligned: the shared segment seeds nothing"
    (Option.exists coarse ~f:(fun seg -> List.is_empty (gpu_seeds ~limits seg)));
  let refs = Option.value_map coarse ~default:[] ~f:(coverage_refutations ~limits) in
  List.iter refs ~f:(Stdio.eprintf "misaligned refutation: %s\n");
  p_exists "misaligned: the family is refuted on coverage, naming the scan's max_vals" refs
    ~f:(mentions ~sub:"max_vals");
  let fine = normal_segments ~arity_cuts:true ~limits opt in
  let freed = site_segment ~limits ~is_site:(is_node z) fine in
  p "misaligned: the arity_cuts segmentation cuts the scan from the matmul"
    (Option.exists freed ~f:(fun seg -> not (has_scan seg.LL.llc))
    && List.exists fine ~f:(fun seg -> has_scan seg.LL.llc));
  let seeds = Option.value_map freed ~default:[] ~f:(gpu_seeds ~limits) in
  p_exists "misaligned: the freed matmul seeds a tensorized candidate" seeds ~f:(fun sp ->
      sp.Autotune.sk_mma);
  p_all "misaligned: every seed of the freed matmul passes validate_parallel" seeds ~f:(fun sp ->
      validates ~label:"misaligned" (Option.value_exn freed) sp);

  Stdio.printf "--- leg 3: the two-pass attention's score contraction and its scan ---\n";
  let y = attention () in
  let opt = lowering ~name:"scan_cov_attn" (Train.forward y) in
  let scores = site_segment ~limits ~is_site:is_scores (normal_segments ~limits opt) in
  p "attention: the default GPU segmentation keeps the scan out of the scores' segment"
    (Option.exists scores ~f:(fun seg -> not (has_scan seg.LL.llc)));
  let seeds = Option.value_map scores ~default:[] ~f:(gpu_seeds ~limits) in
  p_exists "attention: the scores' segment seeds a tensorized candidate" seeds ~f:(fun sp ->
      sp.Autotune.sk_mma);
  p_all "attention: every seed of the scores' segment passes validate_parallel" seeds ~f:(fun sp ->
      validates ~label:"attention" (Option.value_exn scores) sp);
  let shared =
    site_segment ~limits ~is_site:is_scores (normal_segments ~keep_mapping:false ~limits opt)
  in
  p "attention: under the legality rules alone the scores share the scan's segment"
    (Option.exists shared ~f:(fun seg -> has_scan seg.LL.llc));
  p "attention: that shared segment seeds nothing"
    (Option.exists shared ~f:(fun seg -> List.is_empty (gpu_seeds ~limits seg)));
  p_exists "attention: the shared segment's family is refuted on companion coverage"
    (Option.value_map shared ~default:[] ~f:(coverage_refutations ~limits))
    ~f:(fun _ -> true);
  let freed =
    site_segment ~limits ~is_site:is_scores
      (normal_segments ~keep_mapping:false ~arity_cuts:true ~limits opt)
  in
  p "attention: the arity_cuts segmentation frees the scores from the scan"
    (Option.exists freed ~f:(fun seg -> not (has_scan seg.LL.llc)));
  p_exists "attention: the freed scores seed a tensorized candidate"
    (Option.value_map freed ~default:[] ~f:(gpu_seeds ~limits))
    ~f:(fun sp -> sp.Autotune.sk_mma);

  Stdio.printf "--- leg 4: executed, every seed of the aligned shared segment ---\n";
  let real_limits = Context.hardware_limits (Context.auto ()) in
  let leg4_holds_scan, leg4 =
    if not is_gpu then (false, [])
    else
      let build ~transform =
        let comp, _, _, z, sm = indep ~tag:"al" ~cols:n in
        let got, census = execute ?transform ~name:"scan_cov_run" ~init:Fn.id comp [ z; sm ] in
        (* [z] bitwise (a tolerance of 0): its cells are sums of nonnegative small-integer products,
           exact through any f32 accumulation, the tensor unit's included. It once held 1e-5 because
           HIP's rocWMMA (f16, f16, f32) seeds returned most cells a few f32 ulps off (gfx1151:
           5.00000191 for 5) -- the mixed-sign residue of the signed operands it had then
           (gh-ocannl-1420, gh-ocannl-1432). The softmax keeps 1e-5: an f32 [exp] chain whose kernel
           shapes differ between the seed and the untuned pipeline, compiled without the contraction
           licence, so a bitwise claim on it would compare two compilers' contraction choices
           (gh-ocannl-998), not coverage. A race or an uncovered companion moves cells by whole
           units. *)
        (List.zip_exn got [ 0.; 1e-5 ], census)
      in
      let reference, _ = build ~transform:None in
      let comp, a, b, z, _ = indep ~tag:"al" ~cols:n in
      let opt = lowering ~name:"scan_cov_al" comp in
      (* The legality rules alone, so the merge the leg is about happens on every device. *)
      let target =
        site_segment ~limits:real_limits ~is_site:(is_node z)
          (normal_segments ~keep_mapping:false ~limits:real_limits opt)
      in
      let seeds = Option.value_map target ~default:[] ~f:(gpu_seeds ~limits:real_limits) in
      let capability =
        Ll_test.tensorized_matmul_capability ~is_gpu ~is_cpu:false ~limits:real_limits
          ~a:a.Tensor.value ~b:b.Tensor.value ~d:z.Tensor.value
      in
      Stdio.eprintf "leg 4 (not part of the golden): %d seeds, %d of them tensorized\n"
        (List.length seeds)
        (List.count seeds ~f:(fun sp -> sp.Autotune.sk_mma));
      (* The run's own target is the segment holding both the site and the scan, so a run whose
         segmentation lost the scan applies no seed and fails. *)
      let is_target seg =
        has_scan seg.LL.llc
        && Option.exists (site_of ~limits:real_limits seg) ~f:(fun (site, _) ->
            String.equal (Tn.debug_name site.Autotune.m_d) (Tn.debug_name z.Tensor.value))
      in
      ( Option.exists target ~f:(fun seg -> has_scan seg.LL.llc),
        List.mapi seeds ~f:(fun idx sp ->
            let ok, census =
              run_seed ~label:"leg 4" ~idx ~limits:real_limits ~keep_mapping:false ~is_target
                ~build:(fun ~transform -> build ~transform:(Some transform))
                ~reference
            in
            let rendered =
              match capability with
              | `Advertised when sp.Autotune.sk_mma ->
                  Option.exists census ~f:Ll_test.rendered_tensorized
              | _ -> true
            in
            ok && rendered) )
  in
  gated ~when_:is_gpu ~on:backend_name
    "aligned, executed: the matmul's segment under the real limits also holds the scan"
    leg4_holds_scan;
  gated_all ~when_:is_gpu ~on:backend_name
    "aligned, executed: every GPU seed of the shared segment computes z and the softmax, \
     tensorized where the tile is advertised"
    leg4 ~f:Fn.id;

  Stdio.printf "--- leg 5: executed, every seed of the attention's score segment ---\n";
  let leg5 =
    if not is_gpu then []
    else
      let build ~transform =
        let y = attention () in
        let init ctx = Train.init_params ctx Ir.Indexing.Empty y in
        let got, census =
          execute ?transform ~name:"scan_cov_attn_run" ~init (Train.forward y) [ y ]
        in
        (List.zip_exn got [ 1e-4 ], census)
      in
      let reference, _ = build ~transform:None in
      let y = attention () in
      let opt = lowering ~name:"scan_cov_attn" (Train.forward y) in
      let target =
        site_segment ~limits:real_limits ~is_site:is_scores
          (normal_segments ~limits:real_limits opt)
      in
      let seeds = Option.value_map target ~default:[] ~f:(gpu_seeds ~limits:real_limits) in
      Stdio.eprintf "leg 5 (not part of the golden): %d seeds, %d of them tensorized\n"
        (List.length seeds)
        (List.count seeds ~f:(fun sp -> sp.Autotune.sk_mma));
      let is_target seg =
        Option.exists (site_of ~limits:real_limits seg) ~f:(fun (site, _) ->
            is_scores site.Autotune.m_d)
      in
      List.mapi seeds ~f:(fun idx _ ->
          fst
            (run_seed ~label:"leg 5" ~idx ~limits:real_limits ~keep_mapping:true ~is_target
               ~build:(fun ~transform -> build ~transform:(Some transform))
               ~reference))
  in
  gated_all ~when_:is_gpu ~on:backend_name
    "attention, executed: every GPU seed of the scores' segment gives the untuned output" leg5
    ~f:Fn.id
