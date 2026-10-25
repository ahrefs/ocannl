(* gh-ocannl-1076: the strided 1x1 conv — [resnet_block]'s downsample shortcut ([lib/nn_blocks.ml])
   — executed through every matmul seed it gets.

   A 1x1 window is a plain GEMM once lowering drops its extent-1 window loops, so [detect_conv]
   refuses it and the MATMUL family seeds it (pinned structurally by [conv_detection_boundary]). At
   stride 2 the A operand is read at [(2*oh, 2*ow)], and the fear (the issue) was that the matmul
   pipelines' packing and staging [Stage]s assume unit-stride rows of A and would compute wrong
   values silently — candidates are timed, not value-checked. So every seed is compiled, run and
   compared here against the unscheduled form, in the style of [schedule_conv_gemm]'s strided legs
   ([cvs2], [cvs2b]).

   Where the stride lands, pinned below from the lowered access maps. The GEMM row has two
   candidates: the batch axis [b], which the input is read at unit coefficient along
   ([Sketch_families.unit_axis]), and the minor stride-2 output axis [ow]. The classifier chooses by
   extent (gh-ocannl-1397): [ow] owns the row where it is strictly taller than [b], or where there
   is no [b] at all (a batch of one, gh-ocannl-1148, where until then no sketch family seeded the
   site); a tie keeps [b].

   With the batch row, the stride-2 axes [oh, ow] are the site's interior batch loops ([m_bi]),
   which the pipelines hoist and iterate but never tile or pack along: the packing copies [row x k]
   tiles at a fixed strided spatial position. With the strided row ([m_row_stride = 2]), every
   pipeline packs A through a compacting [Stage] ({!Ir.Schedule.stage_compaction_factor}): the tile
   is dense, stored and read at the row's quotient, and only the load keeps the stride. The family
   tree refutes the geometries that would read A in place or pack it at link time. Either way the
   legs claim the same things: both flavors seed, every seed runs and matches, and the tensorized
   ones render tensorized. A unit-stride control shows the batch-row predicate discriminates (there
   the row is [ow], read at unit stride).

   The producers discriminate: every operand cell is a small positive integer varying with each of
   its indices ([Ll_test.weighted], distinct weights, none a multiple of the modulus), so every
   output cell is clear of the zero init, every sum is exact in f32 and f16 (hence in any reduction
   order and on TF32 or f16 tensor units), and a unit-stride misread of A changes the result — the
   host oracle below computes both and pins that they differ, so the parity claims cover the stride.
   The tolerance is the sketch tier's, but on exact integers it only admits exact matches.

   Parity cannot say a tensorized ([sk_mma]) seed ran tensorized: a [Tile_mma] that declines at
   codegen renders its scalar fallback, which computes the same values. So each leg also claims,
   over every executed [sk_mma] seed — on cc the pinned 16-byte population's included, which a wider
   device renders at its minted width (gh-ocannl-1317) — that it renders tensorized per its
   routine's mma census (gh-ocannl-1283) — gated on the backend's tensorized capability for the site
   (the register tiling on cc, an advertised fragment format on a GPU), never on the seeds proposed.

   Every seed the tuner would propose on this box is executed, in both of [Autotune.tune]'s flavors:
   whole-routine seeds under the device's own limits (and, on the C backends, also under
   [conv_detection_boundary]'s pinned 16-byte vector width), and the per-segment flavor mirrored
   from the tuner (see {!segment_seeds}). The bare einsum sites must seed both flavors, and every
   seed must run and match. The production construction — [resnet_block]'s body, main branch
   included (see {!make}) — is stated differently, because there the tuner's view is
   backend-dependent: on cc the block does not fission, and the shortcut's whole-routine seeds stage
   [x] while the main branch reads it too, at another index vector, which [Stage] leaves reading the
   source (gh-ocannl-1147). On GPU the block fissions and the shortcut's segment seeds run. Its
   claims are that every proposed seed either declines with a typed cause or runs and matches, and
   that on the C backends every whole-routine seed runs and matches. Seed counts and declines vary
   with the device, so they go to stderr.

   The block's values are not exact: inference-mode batch norm divides by the square root of its
   stored variance, and the main branch's two 3x3 convolutions then sum 576 such terms per cell into
   magnitudes near 2e5. A per-segment candidate also runs every OTHER segment on the default preset,
   whose reduction orders round those sums differently from the unscheduled form — on HIP by a few
   ulps in a third of the cells, the same for every seed (gh-ocannl-1217). So a per-segment seed is
   compared, at the exact tolerance, against its own segmentation on the default preset — the run it
   replaces one segment's schedule of — and that reference is compared against the unscheduled form
   up to f32 rounding. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
module A = Ir.Affine
module Idx = Ir.Indexing
open Verdict.Claims

(* The whole numerics policy is pinned at its defaults, as a literal so a new knob fails to compile
   here rather than leaking in from the environment: the policy selects seeds (tf32 seeds on CUDA;
   an ambient [Fp16_wide] may withhold the f16 tensor-unit seeds, gh-ocannl-1078), and the stanza
   declares none of its keys. *)
let () =
  Ir.Numerics.set_policy
    {
      tf32_matmuls = false;
      narrow_compute_f32 = true;
      fp16_arithmetic = Ir.Numerics.Fp16_auto;
      bf16_arithmetic = Ir.Numerics.Bf16_auto;
    }

(* The block's parameters take the default precision, which [default_prec] would otherwise set from
   the environment or a config file and so reach the operand precisions the seeding reads. *)
let () =
  Tensor.default_value_prec := Ir.Ops.single;
  Tensor.default_grad_prec := Ir.Ops.single

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let skipped = Verdict.skipped ~backend:backend_name
let on_cpu = Sched.backend_is_cpu backend_name

(* Which backend actually ran, on stderr (the golden is backend-uniform, gh-ocannl-622). *)
let () = Stdio.eprintf "schedule_strided_1x1: backend %s (not part of the golden)\n%!" backend_name
let device_limits = Context.hardware_limits (Context.auto ())

(* The backend's accumulator residency, which a [Privatize] tile is minted at (gh-ocannl-1116). *)
let accum_prec = (Context.codegen_capabilities (Context.auto ())).Ir.Backend_intf.accum_prec

(* A 1x1 site: [b] images of [h x w x ic] channels, [oc] out-channels, at [stride]; built either as
   a bare einsum over constant operands, or through the production blocks. *)
type site = {
  prec : Ir.Ops.prec;
  b : int;
  h : int;
  w : int;
  ic : int;
  oc : int;
  stride : int;
  build : [ `Einsum of string | `Resnet_block ];
}

(* The input is indexed [b; h; w; ic]; the kernel ([~i:[1; 1; ic] ~o:[oc]]) is indexed [oc; 0; 0;
   ic]. Values 1..7 and 1..5. *)
let fx = Ll_test.weighted ~weights:[| 1; 2; 3; 5 |] ~modulus:7 ~offset:1. ~stride:1.
let fk = Ll_test.weighted ~weights:[| 2; 1; 1; 3 |] ~modulus:5 ~offset:1. ~stride:1.

(* Deterministic operands, so sibling graphs compute identical values (forward code is consumed by
   compilation, so every run builds its own graph). *)
let make s tag =
  (* The input goes through the padding-aware [reshape] constructor, as [resnet_block]'s input would
     arrive from an upstream layer: its layout commits at first compilation with the padded conv's
     margin, where [init] commits an unpadded layout and the padded spec is rejected. *)
  let x =
    NTDSL.reshape ~l:(tag ^ "x") ~b:[ s.b ] ~o:[ s.h; s.w; s.ic ]
      (Ir.Ndarray.init_array ~debug:(tag ^ "x") s.prec ~dims:[| s.b; s.h; s.w; s.ic |] ~padding:None
         ~f:fx)
      ()
  in
  match s.build with
  | `Einsum spec ->
      let k = NTDSL.init ~l:(tag ^ "k") ~prec:s.prec ~i:[ 1; 1; s.ic ] ~o:[ s.oc ] ~f:fk () in
      (x, NTDSL.O.einsum ~label:[ tag ] spec x k)
  | `Resnet_block ->
      let block =
        Nn_blocks.resnet_block ~label:[ tag ^ "rb" ] ~stride:s.stride ~out_channels:s.oc ()
      in
      (x, block ~train_step:None x)

(* The block's parameters, made deterministic: every one gets small positive values varying with
   each of its indices (the batch norms' [gamma] and [beta] included). *)
let init_params ctx s (y : Tensor.t) =
  match s.build with
  | `Einsum _ -> ctx
  | `Resnet_block ->
      let ctx = Train.init_params ctx Ir.Indexing.Empty y in
      Set.fold y.Tensor.params ~init:ctx ~f:(fun ctx p ->
          let tn = p.Tensor.value in
          let dims = Lazy.force tn.Ir.Tnode.dims in
          let n = Array.fold dims ~init:1 ~f:( * ) in
          Context.set_values ctx tn
            (Array.init n ~f:(Ll_test.cycle_flat ~dims ~modulus:7 ~offset:1. ~stride:0.125)))

(* Output spatial extents: [stride*o + 0 < n] with a 1-wide window. *)
let out_extent s n = ((n - 1) / s.stride) + 1

(* The host oracle, with the A read at [(stride * oh, stride * ow)]; [~stride:1] on a stride-2 site
   is the misread a unit-stride packing of the strided row would produce. *)
let oracle s ~stride =
  let range n = List.range 0 n in
  List.concat_map (range s.b) ~f:(fun b ->
      List.concat_map
        (range (out_extent s s.h))
        ~f:(fun i ->
          List.concat_map
            (range (out_extent s s.w))
            ~f:(fun j ->
              List.map (range s.oc) ~f:(fun o ->
                  List.sum
                    (module Float)
                    (range s.ic)
                    ~f:(fun c -> fx [| b; stride * i; stride * j; c |] *. fk [| o; 0; 0; c |])))))
  |> Array.of_list

(* Every run builds its own graph in its own context, and releases both contexts after the readback,
   leaf first: device buffers live in backend pool tables the GC does not reclaim, and a leg runs
   tens of candidates. It returns the readback with the routine's mma census ([Context.routine.mma],
   gh-ocannl-1283): how its [Tile_mma] statements actually rendered, which a tensorized seed's
   values cannot tell (see {!leg}). *)
let run_with name s ~lowered_transform =
  let _, y = make s name in
  let init = init_params (Context.auto ()) s y in
  let ctx, routine =
    Context.compile ~lowered_transform init (named name (Train.forward y)) Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx y.Tensor.value in
  Context.release ctx;
  Context.release init;
  (got, routine.Context.mma)

let run_plain name s = fst (run_with name s ~lowered_transform:(fun opt -> [ opt ]))

(* The tuner's per-segment flavor, mirrored ([Autotune.tune]'s [enum_fiss_entries] and its
   [F_sketch] candidates): the routine is fissioned with the default preset and the pipeline's own
   settings; when it splits into more than one segment, every [`Normal] segment seeds on its
   pre-schedule form, keyed by its structural digest, and a candidate schedules the segment with
   that key by the seed and every other one by the default preset. The finer [arity_cuts]
   segmentation is enumerated too on GPU, for segments it keys anew. *)
let seg_key seg =
  Ir.Schedule_cache.digest
    (Ir.Schedule_cache.canonicalize ~static_indices:[] ~with_placements:false seg)

let default_preset seg =
  if on_cpu then Sched.default_cpu ~min_work:0 seg
  else Sched.default_gpu ~min_parallel:1 ~limits:device_limits seg

let fission ~arity_cuts ~preset opt =
  let zero_sched tns = if on_cpu then [] else Sched.zero_expansion ~limits:device_limits tns in
  Sched.fission_scheduled ~promote_locals:(not on_cpu) ~arity_cuts ~preset ~zero_sched
    ~static_indices:[] opt

(* The keyed segments of one segmentation, on a hermetic copy of the lowering; [None] when the
   routine does not fission (the tuner then runs no per-segment flavor). *)
let segment_seeds ~seeds_of ~arity_cuts (opt : LL.optimized) =
  let scratch =
    {
      opt with
      LL.traced_store = Hashtbl.copy opt.LL.traced_store;
      LL.optimize_ctx = LL.copy_optimize_ctx opt.LL.optimize_ctx;
    }
  in
  match fission ~arity_cuts ~preset:default_preset scratch with
  | [] | [ _ ] -> None
  | tuples ->
      Some
        (List.filter_map tuples ~f:(fun (kind, pre, _, _) ->
             (* Matmul segments only: a segment of the block's 3x3 convs seeds the conv family, for
                another site (schedule_conv_gemm's). *)
             match kind with
             | `Normal when Option.is_some (Autotune.detect_matmul pre.LL.llc) -> (
                 match seeds_of pre with [] -> None | seeds -> Some (seg_key pre, seeds))
             | _ -> None))

(* Run a candidate as the tuner runs it: a schedule that violates an op's precondition on this graph
   raises a typed [Schedule_outcome.Cause_at] while scheduling, and the tuner declines that
   candidate — skips it, never times it. The cause is caught inside the transform because
   [Context.compile] turns it back into a plain exception at its boundary; the declined candidate is
   reported on stderr and yields [None] (its compile falls back to the unscheduled form, which is
   not looked at). Any other exception stays fatal. *)
let candidate name s transform =
  let declined = ref None in
  let got =
    run_with name s ~lowered_transform:(fun opt ->
        match transform opt with
        | opts -> opts
        | exception Ir.Schedule_outcome.Cause_at (phase, cause) ->
            declined := Some (phase, cause);
            [ opt ])
  in
  match !declined with
  | None -> Some got
  | Some (phase, cause) ->
      Stdio.eprintf "%s: declined (the tuner skips it): %s (not part of the golden)\n%!" name
        (Sexp.to_string_hum
           (Sexp.List
              [ Ir.Schedule_outcome.sexp_of_phase phase; Ir.Schedule_outcome.sexp_of_cause cause ]));
      None

let run_whole_candidate name s q =
  candidate name s (fun opt ->
      [ Sched.apply_classified (Autotune.sketch_schedule ~accum_prec ~p:q opt) opt ])

(* The [F_sketch] candidate scheduling the segment keyed [key] by [q]; also whether a final
   [`Normal] segment carried that key, so a segmentation that drifted from the enumerated one cannot
   pass by replaying every segment on the default preset. *)
let run_seg_candidate name s ~arity_cuts ~key q =
  let hit = ref false in
  Option.map
    (candidate name s (fun opt ->
         let preset seg =
           if String.equal (seg_key seg) key then Autotune.sketch_schedule ~accum_prec ~p:q seg
           else default_preset seg
         in
         let tuples = fission ~arity_cuts ~preset opt in
         hit :=
           List.exists tuples ~f:(fun (kind, pre, _, _) ->
               Poly.equal kind `Normal && String.equal (seg_key pre) key);
         List.map tuples ~f:(fun (_, _, _, post) -> post)))
    ~f:(fun got -> (!hit, got))

(* The same segmentation with every segment on the default preset: what a per-segment candidate
   replaces one segment of. Against it, a seed's run differs only in the segment the seed schedules.
   Against the unscheduled form, it differs also in the default preset's reduction orders over the
   other segments, which round differently once the values are inexact. *)
let run_default_fission name s ~arity_cuts =
  Option.map ~f:fst
    (candidate name s (fun opt ->
         List.map (fission ~arity_cuts ~preset:default_preset opt) ~f:(fun (_, _, _, post) -> post)))

(* The accumulation's read map on the input [x], and its write map: the one read of [x] that shares
   its statement with a read-modify-write — of the matmul site's output when there is a site (the
   block's 3x3 conv accumulates over [x] too). *)
let maps ?(site : Autotune.matmul_site option) (x : Ir.Tnode.t) (llc : LL.t) =
  let accs = LL.affine_accesses llc in
  let acc_write r =
    List.find accs ~f:(fun w ->
        w.A.a_write && w.A.a_rmw && (not w.A.a_whole)
        && A.same_statement w.A.a_path r.A.a_path
        && Option.for_all site ~f:(fun m -> phys_equal w.A.a_tn m.Autotune.m_d))
  in
  match
    List.filter_map accs ~f:(fun r ->
        if (not r.A.a_write) && phys_equal r.A.a_tn x then
          Option.map (acc_write r) ~f:(fun w -> (r.A.a_map, w.A.a_map))
        else None)
  with
  | [ m ] -> m
  | l ->
      failwith
        (Printf.sprintf "expected one accumulation reading the input, found %d" (List.length l))

(* The symbols the input reads at coefficient 2, and the output symbols it reads at unit stride. *)
let strided_syms in_map =
  Array.to_list in_map
  |> List.concat_map ~f:(function
    | Idx.Affine { symbols; _ } ->
        List.filter_map symbols ~f:(fun (c, sym) -> Option.some_if (c = 2) sym)
    | _ -> [])

let unit_out_syms ~in_map ~out_map =
  Array.to_list in_map
  |> List.filter_map ~f:(function
    | Idx.Iterator sym
      when Array.exists out_map ~f:(fun o -> Idx.equal_axis_index o (Idx.Iterator sym)) ->
        Some sym
    | _ -> None)

let same_syms l1 l2 =
  List.equal Idx.equal_symbol
    (List.sort l1 ~compare:Idx.compare_symbol)
    (List.sort l2 ~compare:Idx.compare_symbol)

(* A seed on one stderr line: its tile sizes, then the names of its set flags. *)
let show (q : Autotune.sketch_params) =
  let flags =
    List.filter_map
      [
        ("gpu", q.Autotune.sk_gpu);
        ("mma", q.Autotune.sk_mma);
        ("hoist", q.Autotune.sk_hoist);
        ("grid", q.Autotune.sk_grid);
        ("epilogue", q.Autotune.sk_epilogue);
        ("batch-grid", q.Autotune.sk_batch_grid);
      ]
      ~f:(fun (name, set) -> Option.some_if set name)
  in
  Printf.sprintf "simd %d, bm %d, bn %d, bk %d, tm %d, tn %d; %s" q.Autotune.sk_simd
    q.Autotune.sk_bm q.Autotune.sk_bn q.Autotune.sk_bk q.Autotune.sk_tm q.Autotune.sk_tn
    (String.concat ~sep:" " flags)

let dedup seeds =
  List.fold seeds ~init:[] ~f:(fun acc q ->
      if List.mem acc q ~equal:Poly.equal then acc else acc @ [ q ])

type probed = {
  site : Autotune.matmul_site option;
  in_map : Idx.axis_index array;
  out_map : Idx.axis_index array;
  whole : Autotune.sketch_params list;
  segs : (bool * string * Autotune.sketch_params list) list option;
      (** [(arity_cuts, key, seeds)] per keyed segment; [None] when the routine does not fission. *)
  device_whole : Autotune.sketch_params list;
  device_segs : (bool * string * Autotune.sketch_params list) list option;
      (** The whole-routine and keyed per-segment seeds under the device's own limits alone — what
          [Autotune.tune] proposes; on the C backends the lists above also carry the pinned
          16-byte-vector population. *)
}

(* The probe: the site and whole-routine seeds on the optimized routine, and the keyed per-segment
   seeds — exactly as the tuner's two flavors see them. *)
let probe ~tag s =
  let seed_limits =
    if on_cpu then
      [
        device_limits;
        {
          device_limits with
          Ir.Backend_intf.simd_vector = { device_limits.simd_vector with bytes = 16 };
        };
      ]
    else [ device_limits ]
  in
  let is_gpu = not on_cpu and is_cpu = on_cpu in
  let seeds_of opt =
    dedup
      (List.concat_map seed_limits ~f:(fun limits ->
           Autotune.sketch_seed_params ~is_gpu ~is_cpu ~limits opt))
  in
  let device_seeds_of opt = Autotune.sketch_seed_params ~is_gpu ~is_cpu ~limits:device_limits opt in
  let keyed ~seeds_of opt =
    Option.map (segment_seeds ~seeds_of ~arity_cuts:false opt) ~f:(fun coarse ->
        let fine =
          if on_cpu then []
          else
            Option.value ~default:[] (segment_seeds ~seeds_of ~arity_cuts:true opt)
            |> List.filter ~f:(fun (k, _) ->
                not (List.exists coarse ~f:(fun (c, _) -> String.equal c k)))
        in
        List.map coarse ~f:(fun (k, q) -> (false, k, q))
        @ List.map fine ~f:(fun (k, q) -> (true, k, q)))
  in
  let found = ref None in
  (let x, y = make s (tag ^ "_probe") in
   let init = init_params (Context.auto ()) s y in
   let ctx, _ =
     Context.compile
       ~lowered_transform:(fun opt ->
         let site = Autotune.detect_matmul opt.LL.llc in
         let in_map, out_map = maps ?site x.Tensor.value opt.LL.llc in
         found :=
           Some
             {
               site;
               in_map;
               out_map;
               whole = seeds_of opt;
               segs = keyed ~seeds_of opt;
               device_whole = device_seeds_of opt;
               device_segs = keyed ~seeds_of:device_seeds_of opt;
             };
         [ opt ])
       init
       (named (tag ^ "_probe") (Train.forward y))
       Ir.Indexing.Empty
   in
   Context.release ctx;
   Context.release init);
  let o = Option.value_exn !found in
  Option.iter o.site ~f:(fun m ->
      Stdio.eprintf
        "%s: matmul site ni=%d nj=%d nk=%d, %d interior batch loops (not part of the golden)\n%!"
        tag m.Autotune.m_ni m.Autotune.m_nj m.Autotune.m_nk (List.length m.Autotune.m_bi));
  Stdio.eprintf "%s: %d whole-routine seeds; %s (not part of the golden)\n%!" tag
    (List.length o.whole)
    (match o.segs with
    | None -> "the routine does not fission"
    | Some l ->
        Printf.sprintf "%d keyed segments with %d seeds" (List.length l)
          (List.sum (module Int) l ~f:(fun (_, _, q) -> List.length q)));
  List.iter o.whole ~f:(fun q -> Stdio.eprintf "  %s whole: %s\n%!" tag (show q));
  Option.iter o.segs ~f:(fun l ->
      List.iter l ~f:(fun (fine, _, qs) ->
          List.iter qs ~f:(fun q ->
              Stdio.eprintf "  %s segment%s: %s\n%!" tag (if fine then " (fine)" else "") (show q))));
  o

(* A site loop's extent: the row's, or a batch loop's. *)
let extent (m : Autotune.matmul_site) sym =
  if Idx.equal_symbol sym m.Autotune.m_i then Some m.Autotune.m_ni
  else List.Assoc.find (m.Autotune.m_bo @ m.Autotune.m_bi) ~equal:Idx.equal_symbol sym

(* The tallest of [syms] as site loops, [0] for none; [None] when one of them is no site loop. *)
let tallest m syms =
  let ns = List.filter_map syms ~f:(extent m) in
  Option.some_if (List.length ns = List.length syms) (List.fold ns ~init:0 ~f:Int.max)

(* The batch row: the stride-2 symbols are exactly the site's interior batch loops, the GEMM row is
   an output axis the input reads at unit stride, and no stride-2 axis is taller than it (the extent
   rule, gh-ocannl-1397: a tie keeps the unit-stride row). *)
let stride_on_batch_loops o =
  match (o.site, strided_syms o.in_map) with
  | None, _ | _, [] -> false
  | Some m, strided ->
      same_syms strided (List.map m.Autotune.m_bi ~f:fst)
      && m.Autotune.m_row_stride = 1
      && List.mem
           (unit_out_syms ~in_map:o.in_map ~out_map:o.out_map)
           m.Autotune.m_i ~equal:Idx.equal_symbol
      && Option.value_map (tallest m strided) ~default:false ~f:(fun n -> n <= m.Autotune.m_ni)

(* The strided row: the GEMM row is the minor stride-2 output axis (so no interior batch loop sits
   between the roles), with the site's stride recorded, and it is strictly taller than every output
   axis the input reads at unit stride (none at a batch of one, gh-ocannl-1148; the batch axis at
   batch 2-8, gh-ocannl-1397). *)
let stride_on_row o =
  match (o.site, strided_syms o.in_map) with
  | None, _ | _, [] -> false
  | Some m, strided ->
      m.Autotune.m_row_stride = 2
      && List.mem strided m.Autotune.m_i ~equal:Idx.equal_symbol
      && m.Autotune.m_row_axis = Array.length o.out_map - 2
      && Option.value_map
           (tallest m (unit_out_syms ~in_map:o.in_map ~out_map:o.out_map))
           ~default:false
           ~f:(fun n -> n < m.Autotune.m_ni)

(* The seeder's own judgment of a tensorized pipeline for the site's operands on this backend
   ([Ll_test.tensorized_capability], gh-ocannl-1115): the register tiling on the C backends, an
   advertised fragment format on a GPU. Read off the probed site's nodes, at the leg's precision
   when no site was found (the stride claim then fails on its own). *)
let capability s (o : probed) =
  let is_gpu = not on_cpu and is_cpu = on_cpu and limits = device_limits in
  match o.site with
  | Some m ->
      Ll_test.tensorized_matmul_capability ~is_gpu ~is_cpu ~limits ~a:m.Autotune.m_a
        ~b:m.Autotune.m_b ~d:m.Autotune.m_d
  | None ->
      Ll_test.tensorized_capability ~is_gpu ~is_cpu ~limits ~a_prec:s.prec ~b_prec:s.prec
        ~d_prec:s.prec

(* Whether a run rendered tensorized per its mma census; a seed that did not is named on stderr. *)
let rendered name q census =
  let ok = Ll_test.rendered_tensorized census in
  if not ok then
    Stdio.eprintf "%s (%s): rendered %s (not part of the golden)\n%!" name (show q)
      (Ir.C_syntax.mma_summary_string census);
  ok

(* The structural claim a leg makes about where the stride lands. *)
let structure_claim = function
  | `Batch ->
      "a matmul site whose stride-2 axes are its interior batch loops, the row read at unit stride \
       and at least as tall as each of them"
  | `Strided ->
      "a matmul site whose row is the minor output axis, read at stride 2 and strictly taller than \
       every output axis read at unit stride"

let leg ?(declines = false) ?(row = `Batch) ~what ~tag s =
  let o = probe ~tag s in
  let unscheduled = run_plain (tag ^ "_ref") s in
  (match s.build with
  | `Einsum _ ->
      let want = oracle s ~stride:s.stride in
      p_exists
        (what ^ ": the producers discriminate: a unit-stride read of A changes the result")
        (List.zip_exn (Array.to_list want) (Array.to_list (oracle s ~stride:1)))
        ~f:(fun (a, b) -> Float.(a <> b));
      p_close (what ^ ": the unscheduled form matches the host oracle") unscheduled want
        ~tol:(fun _ -> 1e-3)
  | `Resnet_block ->
      Stdio.eprintf "%s: max |unscheduled| %.6g over %d cells (not part of the golden)\n%!" tag
        (Array.fold unscheduled ~init:0. ~f:(fun m a -> Float.max m (Float.abs a)))
        (Array.length unscheduled));
  p
    (what ^ ": " ^ structure_claim row)
    (match row with `Batch -> stride_on_batch_loops o | `Strided -> stride_on_row o);
  let close ~tol ~want got =
    Array.length got = Array.length want
    && Array.for_all2_exn got want ~f:(fun a b -> Float.(abs (a - b) < tol b))
  in
  let exact _ = 1e-3 in
  (* Where values are inexact (the block's batch norms divide by a square root, and its two 3x3
     convolutions then sum 576 such terms into magnitudes near 2e5, where one f32 ulp is 1/64),
     forms that reorder a reduction differ by a few ulps: a relative 1e-5 admits about 80 of
     them. *)
  let rounding b = 1e-3 +. (1e-5 *. Float.abs b) in
  let matches_ref ~want got = close ~tol:exact ~want got in
  let matches got = matches_ref ~want:unscheduled got in
  (* Which run diverged from which form and by how much, so a red claim names its seed: the
     differing cells, the worst one with both values, and the form's magnitude, on stderr. *)
  let report ?(tol = exact) ~against ~want name what_ran got =
    ignore
      (Verdict.within ~what_ran
         (Printf.sprintf "%s, against the %s form" name against)
         got want ~tol
        : bool)
  in
  let seg_seeds =
    Option.value_map o.segs ~default:[] ~f:(fun l ->
        List.concat_map l ~f:(fun (arity_cuts, key, qs) ->
            List.map qs ~f:(fun q -> (arity_cuts, key, q))))
  in
  (* The tuner's own seeding must propose, whatever the pinned-width population adds on cc. *)
  (let whole_n = List.length o.device_whole
   and seg_n =
     Option.value_map o.device_segs ~default:0
       ~f:(List.sum (module Int) ~f:(fun (_, _, q) -> List.length q))
   in
   if declines then
     p_all
       (what ^ ": the device's own limits seed at least one flavor")
       [ whole_n + seg_n ]
       ~f:(fun n -> n > 0)
   else
     p_all (what ^ ": the device's own limits seed both flavors") [ whole_n; seg_n ] ~f:(fun n ->
         n > 0));
  (* A whole-routine seed reorders the site's reduction against the unscheduled form, which on the
     bare sites' exact integers changes nothing, and in the block may round its inexact sums by a
     few ulps (a misread operand is off by far more). *)
  let whole_tol = if declines then rounding else exact in
  let whole_matches got = close ~tol:whole_tol ~want:unscheduled got in
  (* Declined candidates (see {!candidate}) yield [None]; a run keeps its name and seed for the
     census claims below. *)
  let whole_runs =
    List.mapi o.whole ~f:(fun i q ->
        let name = Printf.sprintf "%s_w%d" tag i in
        let r = run_whole_candidate name s q in
        Option.iter r ~f:(fun (got, _) ->
            report ~tol:whole_tol ~against:"unscheduled" ~want:unscheduled name (show q) got);
        (name, q, r))
  in
  (* The form a per-segment seed is compared against: on the bare sites the unscheduled one, whose
     values are exact integers; in the block, the default-preset run of the seed's own segmentation
     (see {!run_default_fission}), since there the values are not exact. *)
  let seg_baselines =
    if not declines then []
    else
      List.dedup_and_sort ~compare:Bool.compare (List.map seg_seeds ~f:(fun (a, _, _) -> a))
      |> List.map ~f:(fun arity_cuts ->
          let name = Printf.sprintf "%s_f%b" tag arity_cuts in
          let got = run_default_fission name s ~arity_cuts in
          let what_ran = Printf.sprintf "default preset, arity cuts %b" arity_cuts in
          Option.iter got ~f:(fun got ->
              if Array.length got = Array.length unscheduled then
                Stdio.eprintf
                  "%s (%s): max |diff| from the unscheduled form %.3g (not part of the golden)\n%!"
                  name what_ran
                  (Array.fold2_exn got unscheduled ~init:0. ~f:(fun m a b ->
                       Float.max m (Float.abs (a -. b))));
              report ~tol:rounding ~against:"unscheduled" ~want:unscheduled name what_ran got);
          (arity_cuts, got))
  in
  let seg_want arity_cuts =
    if not declines then Some ("unscheduled", unscheduled)
    else
      Option.map
        (Option.join (List.Assoc.find seg_baselines ~equal:Bool.equal arity_cuts))
        ~f:(fun want -> ("default-preset segmentation", want))
  in
  let seg_runs =
    List.mapi seg_seeds ~f:(fun i (arity_cuts, key, q) ->
        let name = Printf.sprintf "%s_s%d" tag i in
        let r = run_seg_candidate name s ~arity_cuts ~key q in
        Option.iter r ~f:(fun (hit, (got, _)) ->
            if not hit then
              Stdio.eprintf "%s (%s): no final segment carried its key (not part of the golden)\n%!"
                name (show q);
            Option.iter (seg_want arity_cuts) ~f:(fun (against, want) ->
                report ~against ~want name (show q) got));
        ( name,
          (key, q),
          Option.map r ~f:(fun (hit, (got, census)) -> (arity_cuts, hit, got, census)) ))
  in
  let seg_ok (arity_cuts, hit, got, _) =
    hit
    && Option.value_map (seg_want arity_cuts) ~default:false ~f:(fun (_, want) ->
        matches_ref ~want got)
  in
  let values runs = List.map runs ~f:(fun (_, _, r) -> r) in
  (* A tensorized seed whose [Tile_mma] declines at codegen renders the scalar fallback, which
     computes the same values, so parity cannot say it ran tensorized: every executed [sk_mma] seed
     also answers for its routine's mma census (gh-ocannl-1283). The block-tile seeds request no
     tensorization. On the C backends the claims cover the pinned 16-byte population too: its
     register-tile geometries are minted for a narrower file than the device's where the device is
     wider (AVX2, AVX-512), and a narrower file's widths render on a wider one
     ([Register_tile.renderable_lanes], gh-ocannl-1317) -- which is what lets a seed or a tuned
     schedule cross to a wider machine without silently losing its tile. Gated on {!capability},
     judged under the device limits, never on the seeds proposed: where the capability holds, a
     flavor that proposes no tensorized seed fails the population floor. *)
  let whole_census = List.map whole_runs ~f:(fun (name, q, r) -> (name, q, Option.map r ~f:snd))
  and seg_census =
    List.map seg_runs ~f:(fun (name, (_, q), r) ->
        (name, q, Option.map r ~f:(fun (_, _, _, census) -> census)))
  in
  let tensorized = List.filter ~f:(fun (_, q, _) -> q.Autotune.sk_mma) in
  let advertised, aggregation =
    match capability s o with `Advertised -> (true, None) | `Withheld a -> (false, Some a)
  in
  let census_claim ~on_decline label runs =
    gated_all ?aggregation ~when_:advertised ~on:backend_name label (tensorized runs)
      ~f:(fun (name, q, census) -> Option.value_map census ~default:on_decline ~f:(rendered name q))
  in
  if not declines then (
    (* The bare sites: each flavor seeds, and every seed runs — a decline fails. *)
    p_all
      (what ^ ": every whole-routine seed runs and matches the unscheduled form")
      (values whole_runs)
      ~f:(Option.value_map ~default:false ~f:(fun (got, _) -> matches got));
    p_all
      (what
     ^ ": every per-segment seed runs, scheduling the segment it was keyed on, and matches the \
        unscheduled form")
      (values seg_runs)
      ~f:(Option.value_map ~default:false ~f:seg_ok);
    census_claim ~on_decline:false
      (what
     ^ ": the whole-routine seeds include tensorized ones, and every one renders tensorized per \
        its routine's mma census")
      whole_census;
    census_claim ~on_decline:false
      (what
     ^ ": the per-segment seeds include tensorized ones, and every one renders tensorized per its \
        routine's mma census")
      seg_census)
  else (
    (* The block: which flavor seeds is backend-dependent (whole-routine on the C backends, which do
       not fission it; per-segment on GPU, which gates the zeroed whole-routine site), and inside it
       a seed may decline with a typed cause. What holds everywhere is that no proposed seed,
       applied alone, computes a wrong value — over a population that is not empty. The tuner's
       recombined per-segment candidates (several keyed segments' best seeds at once, the block's
       conv segments included) are not exercised: they are the fission machinery's composition, not
       this site's seeds. *)
    p_all
      (what
     ^ ": every seed either tuner flavor proposes, applied alone, declines with a typed cause or \
        runs and matches the form it replaces one schedule of: the unscheduled form up to f32 \
        rounding for a whole-routine seed, its segmentation on the default preset for a \
        per-segment one")
      (List.map (values whole_runs)
         ~f:(Option.value_map ~default:true ~f:(fun (got, _) -> whole_matches got))
      @ List.map (values seg_runs) ~f:(Option.value_map ~default:true ~f:seg_ok))
      ~f:Fn.id;
    (* gh-ocannl-1147: the main branch reads [x] too, at its own index vector, and [Stage] leaves
       that read on the source. A whole-routine seed that declined here would leave the C backends,
       where the block does not fission, with no sketch for the tuner to time for the shortcut. *)
    gated_all ~when_:on_cpu ~on:backend_name
      (what
     ^ ": on the C backends, which do not fission the block, every whole-routine seed runs (its \
        operand Stage leaves the main branch's read of the input on the source) and matches the \
        unscheduled form up to f32 rounding")
      (values whole_runs)
      ~f:(Option.value_map ~default:false ~f:(fun (got, _) -> whole_matches got));
    census_claim ~on_decline:true
      (what
     ^ ": every tensorized seed either tuner flavor proposes declines with a typed cause or \
        renders tensorized per its routine's mma census")
      (whole_census @ seg_census);
    (* The per-segment comparison above is only as good as its reference: each default-preset
       segmentation must itself compute the block, up to the f32 rounding of reordered inexact sums.
       On the C backends the block does not fission, so there is no segmentation. *)
    let label =
      what
      ^ ": every segmentation a seed is keyed in, on the default preset, runs and matches the \
         unscheduled form up to f32 rounding"
    in
    if on_cpu then skipped label
    else
      p_all label seg_baselines ~f:(fun (_, got) ->
          Option.value_map got ~default:false ~f:(close ~tol:rounding ~want:unscheduled)));
  o

(* [conv_detection_boundary]'s [cdb_k11s] witness: a valid-mode 1x1 window at stride 2 over a 7x7
   map, 4 in-channels, 8 out-channels. *)
let valid = "...| 2*oh<+kh, 2*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.."
let valid1 = "...| 1*oh<+kh, 1*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.."

(* [Nn_blocks.conv2d ~kernel_size:1 ~stride:2]'s spec (padded mode), as a bare einsum. *)
let padded = "... | 2*oh=+kh, 2*ow=+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ... | oh, ow, ..oc.."

(* The f16 legs: every GPU runs them — the block-tile seeds do not depend on an MMA capability — and
   only the tensorized-seed claims wait on the f16 capability (an advertised f16 fragment format:
   HIP's rocWMMA, CUDA, Metal), where the operands reach the Stage + Tensorize pipelines. The C
   backends' seeding pre-filters to uniform f32/f64, so there every claim is skipped. *)
let f16_leg ~row ~what ~tag s =
  if not on_cpu then ignore (leg ~row ~what ~tag s : probed)
  else
    List.iter
      [
        "the producers discriminate: a unit-stride read of A changes the result";
        "the unscheduled form matches the host oracle";
        structure_claim row;
        "the device's own limits seed both flavors";
        "every whole-routine seed runs and matches the unscheduled form";
        "every per-segment seed runs, scheduling the segment it was keyed on, and matches the \
         unscheduled form";
        "the whole-routine seeds include tensorized ones, and every one renders tensorized per its \
         routine's mma census";
        "the per-segment seeds include tensorized ones, and every one renders tensorized per its \
         routine's mma census";
      ]
      ~f:(fun c -> skipped (what ^ ": " ^ c))

let () =
  let f32 = Ir.Ops.single in
  let run_leg ?declines ?row ~what ~tag s = ignore (leg ?declines ?row ~what ~tag s : probed) in
  (* At batch 2 the strided row [ow] (4 and 16 tall here, 8 in the block below) is taller than the
     batch row, so it owns the GEMM row (gh-ocannl-1397). *)
  run_leg ~row:`Strided ~what:"k11s witness" ~tag:"s11w"
    { prec = f32; b = 2; h = 7; w = 7; ic = 4; oc = 8; stride = 2; build = `Einsum valid };
  run_leg ~row:`Strided ~what:"downsample einsum f32" ~tag:"s11r"
    { prec = f32; b = 2; h = 16; w = 32; ic = 16; oc = 16; stride = 2; build = `Einsum padded };
  (* The tie: at batch 8 over an 8-wide strided row the batch axis keeps the row. *)
  run_leg ~what:"downsample einsum f32, batch as tall as the row" ~tag:"s11t"
    { prec = f32; b = 8; h = 8; w = 16; ic = 16; oc = 16; stride = 2; build = `Einsum padded };
  (* The production construction — the whole block, main branch included — at a ResNet downsample's
     channel width. (At 16 channels the shortcut's conv output is virtualized instead, inlined as a
     local dot product into each of its batch norm's three consumers, and no family sees a site.) *)
  run_leg ~declines:true ~row:`Strided ~what:"resnet block" ~tag:"s11d"
    { prec = f32; b = 2; h = 8; w = 16; ic = 64; oc = 64; stride = 2; build = `Resnet_block };
  (* The unit-stride control: the row is [ow], and the predicate above is false. *)
  (let o =
     probe ~tag:"s11c"
       { prec = f32; b = 2; h = 7; w = 7; ic = 4; oc = 8; stride = 1; build = `Einsum valid1 }
   in
   p "unit-stride control: a matmul site whose row is the minor spatial axis, read at unit stride"
     (match o.site with
     | None -> false
     | Some m ->
         m.Autotune.m_row_axis = Array.length o.out_map - 2
         && List.mem
              (unit_out_syms ~in_map:o.in_map ~out_map:o.out_map)
              m.Autotune.m_i ~equal:Idx.equal_symbol);
   p_empty "unit-stride control: the input is read at no stride-2 axis"
     ~over:(Array.to_list o.in_map) (strided_syms o.in_map);
   p "unit-stride control: the stride predicate is false" (not (stride_on_batch_loops o)));
  (* A batch of one (gh-ocannl-1148): no output axis is read at unit stride, so the strided window
     owns the row, and every seed packs A through a compacting [Stage]. *)
  (let o =
     leg ~row:`Strided ~what:"batch of one" ~tag:"s11o"
       { prec = f32; b = 1; h = 16; w = 32; ic = 16; oc = 16; stride = 2; build = `Einsum padded }
   in
   p_empty "batch of one: the input is read at no unit-stride output axis"
     ~over:(Array.to_list o.in_map)
     (unit_out_syms ~in_map:o.in_map ~out_map:o.out_map));
  (* The same at the valid 7x7 witness's odd extents: a 4-wide strided row that no block size
     divides, so the staged pipelines pad it and the compacted tile's edge guards hold. *)
  run_leg ~row:`Strided ~what:"batch of one, valid 7x7" ~tag:"s11v"
    { prec = f32; b = 1; h = 7; w = 7; ic = 4; oc = 8; stride = 2; build = `Einsum valid };
  f16_leg ~row:`Strided ~what:"downsample einsum f16" ~tag:"s11h"
    {
      prec = Ir.Ops.half;
      b = 2;
      h = 16;
      w = 32;
      ic = 16;
      oc = 16;
      stride = 2;
      build = `Einsum padded;
    };
  f16_leg ~row:`Batch ~what:"downsample einsum f16, batch as tall as the row" ~tag:"s11u"
    {
      prec = Ir.Ops.half;
      b = 8;
      h = 8;
      w = 16;
      ic = 16;
      oc = 16;
      stride = 2;
      build = `Einsum padded;
    };
  f16_leg ~row:`Strided ~what:"batch of one f16" ~tag:"s11p"
    {
      prec = Ir.Ops.half;
      b = 1;
      h = 16;
      w = 32;
      ic = 16;
      oc = 16;
      stride = 2;
      build = `Einsum padded;
    }
