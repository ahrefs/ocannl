(* gh-ocannl-1076: the strided 1x1 conv — [resnet_block]'s downsample shortcut ([lib/nn_blocks.ml])
   — executed through every matmul seed it gets.

   A 1x1 window is a plain GEMM once lowering drops its extent-1 window loops, so [detect_conv]
   refuses it and the MATMUL family seeds it (pinned structurally by [conv_detection_boundary]). At
   stride 2 the A operand is read at [(2*oh, 2*ow)], and the fear (the issue) was that the matmul
   pipelines' packing and staging [Stage]s assume unit-stride rows of A and would compute wrong
   values silently — candidates are timed, not value-checked. So every seed is compiled, run and
   compared here against the unscheduled form, in the style of [schedule_conv_gemm]'s strided legs
   ([cvs2], [cvs2b]).

   Where the stride lands, pinned below from the lowered access maps: NOT on a tiled role. The
   matmul classifier gives a role only to an axis the operand reads at unit coefficient
   ([Sketch_families.unit_axis]), so the stride-2 axes [oh, ow] become the site's interior batch
   loops ([m_bi]), which the pipelines hoist and iterate but never tile or pack along, and the GEMM
   row is the batch axis. The packing copies [row x k] tiles at a fixed strided spatial position.
   The executed claims cover that addressing; the structural claims say why no tile row is strided,
   and a unit-stride control shows the reason predicate discriminates (there the row is [ow]). The
   consequence for a batch of one is pinned too: no unit-stride output axis is left for the row, so
   no sketch family seeds the site at all.

   The producers discriminate: every operand cell is a small positive integer varying with each of
   its indices ([Ll_test.weighted], distinct weights, none a multiple of the modulus), so every
   output cell is clear of the zero init, every sum is exact in f32 and f16 (hence in any reduction
   order and on TF32 or f16 tensor units), and a unit-stride misread of A changes the result — the
   host oracle below computes both and pins that they differ, so the parity claims cover the stride.
   The tolerance is the sketch tier's, but on exact integers it only admits exact matches.

   Every seed the tuner would propose on this box is executed, in both of [Autotune.tune]'s flavors:
   whole-routine seeds under the device's own limits (and, on the C backends, also under
   [conv_detection_boundary]'s pinned 16-byte vector width), and the per-segment flavor mirrored
   from the tuner (see {!segment_seeds}). The bare einsum sites must seed both flavors, and every
   seed must run and match. The production construction — [resnet_block]'s body, main branch
   included (see {!make}) — is stated differently, because there the tuner's view is
   backend-dependent: on cc the block does not fission and every whole-routine seed DECLINES (its
   operand [Stage] meets the main branch's second read of [x]), so the tuner times no sketch for the
   shortcut; on GPU it fissions and the shortcut's segment seeds and runs. Its claim is that every
   proposed seed either declines with a typed cause or runs and matches. Seed counts and declines
   vary with the device, so they go to stderr. *)

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

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let skipped = Verdict.skipped ~backend:backend_name
let on_cpu = Sched.backend_is_cpu backend_name

(* Which backend actually ran, on stderr (the golden is backend-uniform, gh-ocannl-622). *)
let () = Stdio.eprintf "schedule_strided_1x1: backend %s (not part of the golden)\n%!" backend_name
let device_limits = Context.hardware_limits (Context.auto ())

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
      (* [Nn_blocks.resnet_block ~stride]'s body verbatim — main branch, downsample shortcut,
         residual add, final activation — except that every conv pins its out-channels. The
         constructor leaves them to inference, which cannot close them ("You forgot to specify the
         hidden dimension(s)" on the shortcut's bias), so [resnet_block] itself does not compile
         standalone; nothing in the tree calls it. *)
      let label = [ tag ^ "rb" ] and train_step = None in
      let conv name ~kernel_size ~stride =
        Nn_blocks.conv2d ~label:(name :: label) ~kernel_size ~stride ~out_channels:s.oc ()
      in
      let conv1 = conv "conv1" ~kernel_size:3 ~stride:s.stride in
      let bn1 = Nn_blocks.batch_norm2d ~label:("bn1" :: label) () in
      let conv2 = conv "conv2" ~kernel_size:3 ~stride:1 in
      let bn2 = Nn_blocks.batch_norm2d ~label:("bn2" :: label) () in
      let downsample_conv = conv "downsample" ~kernel_size:1 ~stride:s.stride in
      let downsample_bn = Nn_blocks.batch_norm2d ~label:("downsample_bn" :: label) () in
      let identity = downsample_bn ~train_step (downsample_conv x) in
      let out = conv1 x |> bn1 ~train_step |> TDSL.O.relu |> conv2 |> bn2 ~train_step in
      (x, TDSL.O.(relu (out + identity)))

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
   tens of candidates. *)
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
  got

let run_plain name s = run_with name s ~lowered_transform:(fun opt -> [ opt ])

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
  if on_cpu then Sched.default_cpu ~min_parallel:1 seg
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
  candidate name s (fun opt -> [ Sched.apply_classified (Autotune.sketch_schedule ~p:q opt) opt ])

(* The [F_sketch] candidate scheduling the segment keyed [key] by [q]; also whether a final
   [`Normal] segment carried that key, so a segmentation that drifted from the enumerated one cannot
   pass by replaying every segment on the default preset. *)
let run_seg_candidate name s ~arity_cuts ~key q =
  let hit = ref false in
  Option.map
    (candidate name s (fun opt ->
         let preset seg =
           if String.equal (seg_key seg) key then Autotune.sketch_schedule ~p:q seg
           else default_preset seg
         in
         let tuples = fission ~arity_cuts ~preset opt in
         hit :=
           List.exists tuples ~f:(fun (kind, pre, _, _) ->
               Poly.equal kind `Normal && String.equal (seg_key pre) key);
         List.map tuples ~f:(fun (_, _, _, post) -> post)))
    ~f:(fun got -> (!hit, got))

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
  conv : bool;
  in_map : Idx.axis_index array;
  out_map : Idx.axis_index array;
  whole : Autotune.sketch_params list;
  segs : (bool * string * Autotune.sketch_params list) list option;
      (** [(arity_cuts, key, seeds)] per keyed segment; [None] when the routine does not fission. *)
}

(* The probe: the site and whole-routine seeds on the optimized routine, and the keyed per-segment
   seeds — exactly as the tuner's two flavors see them. *)
let probe ~tag s =
  let seed_limits =
    if on_cpu then [ device_limits; { device_limits with Ir.Backend_intf.simd_vector_bytes = 16 } ]
    else [ device_limits ]
  in
  let is_gpu = not on_cpu and is_cpu = on_cpu in
  let seeds_of opt =
    dedup
      (List.concat_map seed_limits ~f:(fun limits ->
           Autotune.sketch_seed_params ~is_gpu ~is_cpu ~limits opt))
  in
  let found = ref None in
  (let x, y = make s (tag ^ "_probe") in
   let init = init_params (Context.auto ()) s y in
   let ctx, _ =
     Context.compile
       ~lowered_transform:(fun opt ->
         let site = Autotune.detect_matmul opt.LL.llc in
         let in_map, out_map = maps ?site x.Tensor.value opt.LL.llc in
         let segs =
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
         found :=
           Some
             {
               site;
               conv = Option.is_some (Autotune.detect_conv opt.LL.llc);
               in_map;
               out_map;
               whole = seeds_of opt;
               segs;
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

(* Why no tile row is strided: the stride-2 symbols are exactly the site's interior batch loops, and
   the GEMM row is an output axis the input reads at unit stride. *)
let stride_on_batch_loops o =
  match (o.site, strided_syms o.in_map) with
  | None, _ | _, [] -> false
  | Some m, strided ->
      same_syms strided (List.map m.Autotune.m_bi ~f:fst)
      && List.mem
           (unit_out_syms ~in_map:o.in_map ~out_map:o.out_map)
           m.Autotune.m_i ~equal:Idx.equal_symbol

let leg ?(want_mma = false) ?(declines = false) ~what ~tag s =
  let o = probe ~tag s in
  let unscheduled = run_plain (tag ^ "_ref") s in
  (match s.build with
  | `Einsum _ ->
      let want = oracle s ~stride:s.stride in
      p_exists
        (what ^ ": the producers discriminate: a unit-stride read of A changes the result")
        (List.zip_exn (Array.to_list want) (Array.to_list (oracle s ~stride:1)))
        ~f:(fun (a, b) -> Float.(a <> b));
      p_all2 (what ^ ": the unscheduled form matches the host oracle") unscheduled want
        ~f:(fun a b -> Float.(abs (a - b) < 1e-3))
  | `Resnet_block -> ());
  p
    (what
   ^ ": a matmul site whose stride-2 axes are its interior batch loops, the row read at unit stride"
    )
    (stride_on_batch_loops o);
  let matches got =
    Array.length got = Array.length unscheduled
    && Array.for_all2_exn got unscheduled ~f:(fun a b -> Float.(abs (a - b) < 1e-3))
  in
  let seg_seeds =
    Option.value_map o.segs ~default:[] ~f:(fun l ->
        List.concat_map l ~f:(fun (arity_cuts, key, qs) ->
            List.map qs ~f:(fun q -> (arity_cuts, key, q))))
  in
  if want_mma then (
    p_exists (what ^ ": the whole-routine seeds include tensorized ones") o.whole ~f:(fun q ->
        q.Autotune.sk_mma);
    p_exists (what ^ ": the per-segment seeds include tensorized ones") seg_seeds
      ~f:(fun (_, _, q) -> q.Autotune.sk_mma));
  (* Declined candidates (see {!candidate}) yield [None]. *)
  let whole_runs =
    List.mapi o.whole ~f:(fun i q -> run_whole_candidate (Printf.sprintf "%s_w%d" tag i) s q)
  in
  let seg_runs =
    List.mapi seg_seeds ~f:(fun i (arity_cuts, key, q) ->
        run_seg_candidate (Printf.sprintf "%s_s%d" tag i) s ~arity_cuts ~key q)
  in
  let seg_ok (hit, got) = hit && matches got in
  if not declines then (
    (* The bare sites: each flavor seeds, and every seed runs — a decline fails. *)
    p_all
      (what ^ ": every whole-routine seed runs and matches the unscheduled form")
      whole_runs
      ~f:(Option.value_map ~default:false ~f:matches);
    p_all
      (what
     ^ ": every per-segment seed runs, scheduling the segment it was keyed on, and matches the \
        unscheduled form")
      seg_runs
      ~f:(Option.value_map ~default:false ~f:seg_ok))
  else
    (* The block: which flavor seeds is backend-dependent (whole-routine on the C backends, which do
       not fission it; per-segment on GPU, which gates the zeroed whole-routine site), and inside it
       a seed may decline — on cc every whole-routine seed does, its operand [Stage] meeting the
       main branch's second read of [x]. What holds is that no proposed seed, applied alone,
       computes a wrong value — over a population that is not empty. The tuner's recombined
       per-segment candidates (several keyed segments' best seeds at once, the block's conv segments
       included) are not exercised: they are the fission machinery's composition, not this site's
       seeds. *)
    p_all
      (what
     ^ ": every seed either tuner flavor proposes, applied alone, declines with a typed cause or \
        runs and matches the unscheduled form")
      (List.map whole_runs ~f:(Option.value_map ~default:true ~f:matches)
      @ List.map seg_runs ~f:(Option.value_map ~default:true ~f:seg_ok))
      ~f:Fn.id

(* [conv_detection_boundary]'s [cdb_k11s] witness: a valid-mode 1x1 window at stride 2 over a 7x7
   map, 4 in-channels, 8 out-channels. *)
let valid = "...| 2*oh<+kh, 2*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.."
let valid1 = "...| 1*oh<+kh, 1*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.."

(* [Nn_blocks.conv2d ~kernel_size:1 ~stride:2]'s spec (padded mode), as a bare einsum. *)
let padded = "... | 2*oh=+kh, 2*ow=+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ... | oh, ow, ..oc.."

let () =
  let f32 = Ir.Ops.single in
  leg ~what:"k11s witness" ~tag:"s11w"
    { prec = f32; b = 2; h = 7; w = 7; ic = 4; oc = 8; stride = 2; build = `Einsum valid };
  leg ~what:"downsample einsum f32" ~tag:"s11r"
    { prec = f32; b = 2; h = 16; w = 32; ic = 16; oc = 16; stride = 2; build = `Einsum padded };
  (* The production construction — the whole block, main branch included — at a ResNet downsample's
     channel width. (At 16 channels the shortcut's conv output is virtualized instead, inlined as a
     local dot product into each of its batch norm's three consumers, and no family sees a site.) *)
  leg ~declines:true ~what:"resnet block" ~tag:"s11d"
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
  (* A batch of one: the stride-2 axes cannot own the row and no other output axis is left. *)
  (let o =
     probe ~tag:"s11o"
       { prec = f32; b = 1; h = 16; w = 32; ic = 16; oc = 16; stride = 2; build = `Einsum padded }
   in
   p "batch of one: not a matmul or conv site, and no sketch family seeds it"
     (Option.is_none o.site && (not o.conv) && List.is_empty o.whole
     && Option.for_all o.segs ~f:List.is_empty);
   p_empty "batch of one: because the input is read at no unit-stride output axis"
     ~over:(Array.to_list o.in_map)
     (unit_out_syms ~in_map:o.in_map ~out_map:o.out_map));
  (* The tensor-unit leg: f16 operands reach the tensorized (Stage + Tensorize) pipelines on the GPU
     backends advertising f16 fragments (HIP's rocWMMA, CUDA, Metal); the C backends' seeding
     pre-filters to uniform f32/f64. *)
  let what = "downsample einsum f16" in
  if
    (not on_cpu)
    && List.exists [ Ir.Backend_intf.Mma_f32; Ir.Backend_intf.Mma_f16 ] ~f:(fun d ->
        Ir.Backend_intf.advertises_mma_format device_limits ~a:Ir.Backend_intf.Mma_f16
          ~b:Ir.Backend_intf.Mma_f16 ~d)
  then
    leg ~want_mma:true ~what ~tag:"s11h"
      {
        prec = Ir.Ops.half;
        b = 2;
        h = 16;
        w = 32;
        ic = 16;
        oc = 16;
        stride = 2;
        build = `Einsum padded;
      }
  else
    List.iter
      [
        "the producers discriminate: a unit-stride read of A changes the result";
        "the unscheduled form matches the host oracle";
        "a matmul site whose stride-2 axes are its interior batch loops, the row read at unit \
         stride";
        "the whole-routine seeds include tensorized ones";
        "the per-segment seeds include tensorized ones";
        "every whole-routine seed runs and matches the unscheduled form";
        "every per-segment seed runs, scheduling the segment it was keyed on, and matches the \
         unscheduled form";
      ] ~f:(fun c -> skipped (what ^ ": " ^ c))
