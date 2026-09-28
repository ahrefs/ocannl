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

   Every seed the tuner would propose on this box is executed: whole-routine seeds under the
   device's own limits (and, on the C backends, also under [conv_detection_boundary]'s pinned
   16-byte vector width), plus the seeds of the unzeroed fission segment, which is what
   [Autotune.tune]'s per-segment flavor substitutes. Seed counts vary with the device, so they go to
   stderr; the stdout claims are that seeds exist and that each one matches. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
module A = Ir.Affine
module Idx = Ir.Indexing
open Verdict.Claims

(* The f16 leg is stated for the default fp16 mode: under an ambient [Fp16_wide] the family may
   withhold its f16 tensor-unit seeds (gh-ocannl-1078), and this file claims they are executed. *)
let () =
  Ir.Numerics.set_policy { (Ir.Numerics.get ()) with fp16_arithmetic = Ir.Numerics.Fp16_auto }

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let skipped = Verdict.skipped ~backend:backend_name
let on_cpu = Sched.backend_is_cpu backend_name

(* Which backend actually ran, on stderr (the golden is backend-uniform, gh-ocannl-622). *)
let () = Stdio.eprintf "schedule_strided_1x1: backend %s (not part of the golden)\n%!" backend_name
let device_limits = Context.hardware_limits (Context.auto ())

(* A 1x1 site: [b] images of [h x w x ic] channels, [oc] out-channels, at [stride]. *)
type site = {
  prec : Ir.Ops.prec;
  b : int;
  h : int;
  w : int;
  ic : int;
  oc : int;
  stride : int;
  spec : string;
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
  let k = NTDSL.init ~l:(tag ^ "k") ~prec:s.prec ~i:[ 1; 1; s.ic ] ~o:[ s.oc ] ~f:fk () in
  (x, NTDSL.O.einsum ~label:[ tag ] s.spec x k)

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

let run_with name s ~lowered_transform =
  let _, y = make s name in
  let ctx = Context.auto () in
  let ctx, routine =
    Context.compile ~lowered_transform ctx (named name (Train.forward y)) Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  Context.get_values ctx y.Tensor.value

let run_plain name s = run_with name s ~lowered_transform:(fun opt -> [ opt ])

(* Through the fission seam: [seg_sched] schedules the segment holding the UNZEROED matmul site (the
   accumulation, whose [Zero_out] fissions into its own [`Zeros] segment); every other segment stays
   unscheduled, zero segments getting the default zero expansion on GPU. *)
let run_fiss name s ~seg_sched =
  let limits = device_limits in
  let transforms (opt : LL.optimized) =
    let preset (seg : LL.optimized) =
      match Autotune.detect_matmul seg.LL.llc with
      | Some site when not site.Autotune.m_zeroed -> seg_sched seg
      | _ -> []
    in
    let zero_sched tns = if on_cpu then [] else Sched.zero_expansion ~limits tns in
    Sched.fission_scheduled ~preset ~zero_sched ~static_indices:[] opt
    |> List.map ~f:(fun (_, _, _, post) -> post)
  in
  run_with name s ~lowered_transform:transforms

(* The accumulation's read map on the input [x], and its write map. *)
let maps (x : Ir.Tnode.t) (llc : LL.t) =
  let accs = LL.affine_accesses llc in
  match List.filter accs ~f:(fun a -> (not a.A.a_write) && phys_equal a.A.a_tn x) with
  | [ r ] -> (
      match
        List.filter accs ~f:(fun w ->
            w.A.a_write && w.A.a_rmw && (not w.A.a_whole) && A.same_statement w.A.a_path r.A.a_path)
      with
      | [ w ] -> (r.A.a_map, w.A.a_map)
      | _ -> failwith "the read of the input is not in one accumulation")
  | l -> failwith (Printf.sprintf "expected one read of the input, found %d" (List.length l))

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
  seg : Autotune.sketch_params list;
}

(* The probe: the site and whole-routine seeds on the optimized routine, the per-segment seeds on
   the unzeroed segment — exactly as the tuner's two flavors see them. *)
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
  let found = ref None and seg = ref [] in
  (let x, y = make s (tag ^ "_probe") in
   ignore
     (Context.compile
        ~lowered_transform:(fun opt ->
          let in_map, out_map = maps x.Tensor.value opt.LL.llc in
          found :=
            Some
              {
                site = Autotune.detect_matmul opt.LL.llc;
                conv = Option.is_some (Autotune.detect_conv opt.LL.llc);
                in_map;
                out_map;
                whole = seeds_of opt;
                seg = [];
              };
          [ opt ])
        (Context.auto ())
        (named (tag ^ "_probe") (Train.forward y))
        Ir.Indexing.Empty));
  ignore
    (run_fiss (tag ^ "_sprobe") s ~seg_sched:(fun pre ->
         seg := seeds_of pre;
         []));
  let o = { (Option.value_exn !found) with seg = !seg } in
  Option.iter o.site ~f:(fun m ->
      Stdio.eprintf
        "%s: matmul site ni=%d nj=%d nk=%d, %d interior batch loops (not part of the golden)\n%!"
        tag m.Autotune.m_ni m.Autotune.m_nj m.Autotune.m_nk (List.length m.Autotune.m_bi));
  Stdio.eprintf "%s: %d whole-routine and %d per-segment seeds (not part of the golden)\n%!" tag
    (List.length o.whole) (List.length o.seg);
  List.iter (o.whole @ o.seg) ~f:(fun q -> Stdio.eprintf "  %s: %s\n%!" tag (show q));
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

let leg ~what ~tag s =
  let o = probe ~tag s in
  let want = oracle s ~stride:s.stride in
  p_exists
    (what ^ ": the producers discriminate: a unit-stride read of A changes the result")
    (List.zip_exn (Array.to_list want) (Array.to_list (oracle s ~stride:1)))
    ~f:(fun (a, b) -> Float.(a <> b));
  p_all2
    (what ^ ": the unscheduled form matches the host oracle")
    (run_plain (tag ^ "_ref") s)
    want
    ~f:(fun a b -> Float.(abs (a - b) < 1e-3));
  p
    (what
   ^ ": a matmul site whose stride-2 axes are its interior batch loops, the row read at unit stride"
    )
    ((not o.conv) && stride_on_batch_loops o);
  let matches got =
    Array.length got = Array.length want
    && Array.for_all2_exn got want ~f:(fun a b -> Float.(abs (a - b) < 1e-3))
  in
  p_alli (what ^ ": every whole-routine seed matches the unscheduled form") o.whole ~f:(fun i q ->
      let name = Printf.sprintf "%s_w%d" tag i in
      matches
        (run_with name s ~lowered_transform:(fun opt ->
             [ Sched.apply (Autotune.sketch_schedule ~p:q opt) opt ])));
  p_alli (what ^ ": every per-segment seed matches the unscheduled form") o.seg ~f:(fun i q ->
      let name = Printf.sprintf "%s_s%d" tag i in
      matches (run_fiss name s ~seg_sched:(fun pre -> Autotune.sketch_schedule ~p:q pre)))

(* [conv_detection_boundary]'s [cdb_k11s] witness: a valid-mode 1x1 window at stride 2 over a 7x7
   map, 4 in-channels, 8 out-channels. *)
let valid = "...| 2*oh<+kh, 2*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.."
let valid1 = "...| 1*oh<+kh, 1*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.."

(* [Nn_blocks.conv2d ~kernel_size:1 ~stride:2]'s spec as [resnet_block]'s downsample builds it
   (padded mode). *)
let padded = "... | 2*oh=+kh, 2*ow=+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ... | oh, ow, ..oc.."

let () =
  let f32 = Ir.Ops.single in
  leg ~what:"k11s witness" ~tag:"s11w"
    { prec = f32; b = 2; h = 7; w = 7; ic = 4; oc = 8; stride = 2; spec = valid };
  leg ~what:"resnet downsample f32" ~tag:"s11r"
    { prec = f32; b = 2; h = 16; w = 32; ic = 16; oc = 16; stride = 2; spec = padded };
  (* The unit-stride control: the row is [ow], and the predicate above is false. *)
  (let o =
     probe ~tag:"s11c"
       { prec = f32; b = 2; h = 7; w = 7; ic = 4; oc = 8; stride = 1; spec = valid1 }
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
       { prec = f32; b = 1; h = 16; w = 32; ic = 16; oc = 16; stride = 2; spec = padded }
   in
   p "batch of one: not a matmul or conv site, and no sketch family seeds it"
     (Option.is_none o.site && (not o.conv) && List.is_empty o.whole && List.is_empty o.seg);
   p_empty "batch of one: because the input is read at no unit-stride output axis"
     ~over:(Array.to_list o.in_map)
     (unit_out_syms ~in_map:o.in_map ~out_map:o.out_map));
  (* The tensor-unit leg: f16 operands reach the tensorized (Stage + Tensorize) pipelines on the GPU
     backends advertising f16 fragments (HIP's rocWMMA, CUDA, Metal); the C backends' seeding
     pre-filters to uniform f32/f64. *)
  let what = "resnet downsample f16" in
  if
    (not on_cpu)
    && List.exists [ Ir.Backend_intf.Mma_f32; Ir.Backend_intf.Mma_f16 ] ~f:(fun d ->
        Ir.Backend_intf.advertises_mma_format device_limits ~a:Ir.Backend_intf.Mma_f16
          ~b:Ir.Backend_intf.Mma_f16 ~d)
  then
    leg ~what ~tag:"s11h"
      { prec = Ir.Ops.half; b = 2; h = 16; w = 32; ic = 16; oc = 16; stride = 2; spec = padded }
  else
    List.iter
      [
        "the producers discriminate: a unit-stride read of A changes the result";
        "the unscheduled form matches the host oracle";
        "a matmul site whose stride-2 axes are its interior batch loops, the row read at unit \
         stride";
        "every whole-routine seed matches the unscheduled form";
        "every per-segment seed matches the unscheduled form";
      ] ~f:(fun c -> skipped (what ^ ": " ^ c))
