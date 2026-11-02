(* gh-ocannl-1609: do the CPU flavors the family tree excludes beside a companion win?

   The whole-triple form, the packed Grid shapes and the conv direct Grid / row-panel shapes are
   valid beside a materialized companion on cc's fork-join model, and the family tree [Excluded]s
   them there as search economy until a measurement says one wins. This is that measurement's
   instrument: on one workload it times, with the tuner's own timer ([Autotune.time_routine], the
   [Queued] objective), the untuned default, every seed the production family tree offers on the
   whole-routine lowering (the conv: its own segment's), and every seed
   [Autotune.lift_cpu_companions] recovers, each checked against the default's output (bit for bit
   on the exact integer matmul, within 1e-4 relative on the conv, whose parameters are random and
   whose seeds reduce in their own order). A recovered seed wins when it beats both the default and
   the best production seed.

   Workloads: - [diag N K]: tune_footprint_default's routine in its materialize-all arm, an [N x K]
   by [K x N] product (materialized) read through its diagonal -- the companion. Defaults 8 32, the
   routine's own geometry. - [conv B]: the cifar_resnet stage-1 conv with its bias + relu tail, [B]
   images of 32x32x16 into 16 channels through a padded 3x3 window (default 64, the workload's
   batch).

   Usage: companion_flavor_bench diag [N K] [repeats] | companion_flavor_bench conv [B] [repeats]
   (repeats default 16). Measure through benchmarks/bin_cells.py; the backend is pinned to cc unless
   one is configured. Every variant prints its [bench-result] line; a failed check exits 1 after all
   have reported. *)

open Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Sspace = Ir.Schedule_space
module Asgns = Ir.Assignments

let p fmt = Bench_out.p fmt

let make_ctx () =
  match Utils.get_global_arg ~arg_name:"backend" ~default:"" with
  | "" -> Context.cpu ()
  | _ -> Context.auto ()

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

type workload = {
  w_name : string;
  w_segment : bool;
      (** Seeds the site's own segment of the tuner's segmentation (the per-segment candidates),
          rather than the whole routine (the whole-routine candidates). *)
  w_comp : Asgns.comp;
  w_out : Tensor.t;
  w_mat : Ir.Tnode.t list;
  w_ctx : Context.t;  (** With the workload's parameters initialized, once for every variant. *)
  w_exact : bool;
      (** Integer operands, every sum exact: a variant must match the default bit for bit. Otherwise
          (the conv's randomly initialized parameters) a variant reduces in its own order, and the
          check is relative, at [1e-4]. *)
}

let ints label ~rows ~cols modulus =
  TDSL.ndarray
    (Array.init (rows * cols) ~f:(fun idx ->
         let i = idx / cols and j = idx % cols in
         Float.of_int (((i + (2 * j)) % modulus) + 1)))
    ~label:[ label ] ~output_dims:[ rows; cols ] ()

let diag ~n ~k =
  let x = ints "cfb_x" ~rows:n ~cols:k 7 in
  let y = ints "cfb_y" ~rows:k ~cols:n 5 in
  let%op a = x +* "ik; kj => ij" y in
  let%op d = a ++ "ii => i" in
  let mat = [ a.Tensor.value ] in
  {
    w_name = Printf.sprintf "diag%dx%d" n k;
    w_segment = false;
    w_comp = Train.forward d;
    w_out = d;
    w_mat = mat;
    w_ctx = Context.decide_materialized (make_ctx ()) mat;
    w_exact = true;
  }

let conv ~b =
  let x0 =
    NTDSL.init ~l:"cfb_x0" ~prec:Ir.Ops.single ~b:[ b ] ~o:[ 32; 32; 16 ]
      ~f:(fun idcs ->
        Float.of_int ((Array.fold idcs ~init:0 ~f:(fun acc i -> (acc * 3) + i) % 7) - 3))
      ()
  in
  (* The stem's activation, computed in the routine as in the network, so its buffer takes the
     conv's padding margins. *)
  let%op x = relu x0 in
  Train.set_materialized x.Tensor.value;
  let block =
    Nn_blocks.conv2d ~label:[ "cfb_s1c1" ] ~kernel_size:3 ~use_padding:true ~out_channels:16 ()
  in
  let%op y = relu (block x) in
  Train.set_materialized y.Tensor.value;
  {
    w_name = Printf.sprintf "conv_b%d" b;
    (* The whole routine computes the conv's input, which the conv pipeline's packing [Stage] cannot
       read in the same kernel: the conv site is seeded per segment, as the tuner seeds it. *)
    w_segment = true;
    w_comp = Train.forward y;
    w_out = y;
    w_mat = [];
    w_ctx = Train.init_params (make_ctx ()) Ir.Indexing.Empty y;
    w_exact = false;
  }

(* What a seed is, for the report. *)
let flavor (sp : Autotune.sketch_params) =
  if sp.sk_conv then if sp.sk_grid || sp.sk_bm > 0 then "conv-direct" else "conv-serial"
  else if sp.sk_mma && sp.sk_bk = 0 then "whole-triple"
  else if sp.sk_mma && sp.sk_grid then "packed-grid"
  else if sp.sk_mma then "packed-serial"
  else "blocktile"

let () =
  let args = Bench_args.create "companion_flavor_bench" in
  let which = Bench_args.string args 0 ~default:"diag" in
  let w, repeats =
    match which with
    | "diag" ->
        let n = Bench_args.int args 1 ~name:"N" ~default:8 in
        let k = Bench_args.int args 2 ~name:"K" ~default:32 in
        (diag ~n ~k, Bench_args.int args 3 ~name:"repeats" ~default:16)
    | "conv" ->
        let b = Bench_args.int args 1 ~name:"B" ~default:64 in
        (conv ~b, Bench_args.int args 2 ~name:"repeats" ~default:16)
    | other -> Bench_args.bad args "unknown workload %S (diag or conv)" other
  in
  let limits = Context.hardware_limits w.w_ctx in
  let accum_prec = Fn.id in
  let seeding opt = Autotune.family_seeding ~is_gpu:false ~is_cpu:true ~limits opt in
  let production opt = snd (seeding opt) in
  let recovered opt =
    let offered = production opt in
    match fst (seeding opt) with
    | Autotune.Seeding_matmul { tree; _ } | Autotune.Seeding_conv { tree; _ } ->
        List.filter
          (Sspace.leaves (Autotune.lift_cpu_companions tree))
          ~f:(fun sp -> not (List.mem offered sp ~equal:Poly.equal))
    | Autotune.Seeding_none -> []
  in
  let hermetic (o : LL.optimized) =
    {
      o with
      LL.traced_store = Hashtbl.copy o.LL.traced_store;
      optimize_ctx = LL.copy_optimize_ctx o.LL.optimize_ctx;
    }
  in
  let has_site (o : LL.optimized) =
    match fst (seeding o) with Autotune.Seeding_none -> false | _ -> true
  in
  (* The tuner's segmentation of [o] under its [`Search] presets, on a hermetic copy, and the index
     of the first segment a family claims. *)
  let segmentation o =
    let plan, segs =
      Autotune.sketch_segmentation ~presets:`Search ~is_gpu:false ~is_cpu:true ~limits
        ~static_indices:[] (hermetic o)
    in
    match List.findi segs ~f:(fun _ (_, pre, _, _) -> has_site pre) with
    | Some (i, (_, pre, _, _)) -> (plan, i, pre)
    | None -> failwith "companion_flavor_bench: no segment of the routine holds a sketch site"
  in
  let base =
    Context.lowered_for_decisions ~materialized:w.w_mat w.w_ctx w.w_comp Ir.Indexing.Empty
  in
  let opt =
    if w.w_segment then
      let _, _, pre = segmentation base in
      pre
    else base
  in
  let prod = production opt and back = recovered opt in
  p "companion_flavor_bench %s: %d production seed(s), %d recovered, %d repeats\n" w.w_name
    (List.length prod) (List.length back) repeats;
  (* One variant: compiled (the seed re-derived by position from the compile's own lowering), run
     once for the check, then timed. *)
  let run ~variant ?enum idx_sp =
    let seed_schedule enum (o : LL.optimized) =
      let idx, (sp : Autotune.sketch_params) = Option.value_exn idx_sp in
      match List.nth (enum o) idx with
      | Some sp' when Poly.equal sp' sp -> Autotune.sketch_schedule ~accum_prec ~p:sp' o
      | _ -> failwith (Printf.sprintf "seed %d is gone from the compile's lowering" idx)
    in
    let transform =
      Option.map enum ~f:(fun enum (o : LL.optimized) ->
          if w.w_segment then
            (* As a per-segment candidate compiles: the site's segment takes the seed, every other
               segment its kind's [`Search] preset. *)
            let plan, ci, _ = segmentation o in
            List.map
              (snd
                 (Autotune.tuner_fission
                    ~replay:
                      (plan, fun i pre -> if i = ci then Some (seed_schedule enum pre) else None)
                    ~is_gpu:false ~limits
                    ~preset:
                      (Autotune.segment_preset ~presets:`Search ~is_gpu:false ~is_cpu:true ~limits)
                    ~static_indices:[] o))
              ~f:(fun (_, _, _, post) -> post)
          else [ Sched.apply (seed_schedule enum o) o ])
    in
    match
      let ctx, routine =
        match transform with
        | None -> Context.compile w.w_ctx (named variant w.w_comp) Ir.Indexing.Empty
        | Some lowered_transform ->
            Context.compile ~lowered_transform w.w_ctx (named variant w.w_comp) Ir.Indexing.Empty
      in
      let ctx = Context.run ctx routine in
      let values = Context.get_values ctx w.w_out.Tensor.value in
      let r = Autotune.time_routine ~timing:Autotune.Queued ~repeats ctx routine in
      Context.release ctx;
      (values, Autotune.admitted_timing_ms r)
    with
    | values, ms -> Ok (values, ms)
    | exception exn -> Error (Exn.to_string exn)
  in
  let failed = ref 0 in
  let report ~variant ~reference result =
    match result with
    | Error why ->
        Int.incr failed;
        p "  %-48s FAILED: %s\n" variant why;
        Bench_out.result ~variant ~check:(Bench_out.Failed why) ~seconds:Float.nan;
        None
    | Ok (values, ms) ->
        let check =
          match reference with
          | None -> Bench_out.Passed
          | Some r ->
              if
                Array.length values = Array.length r
                && Array.for_all2_exn values r ~f:(fun g e ->
                    if w.w_exact then Int64.equal (Int64.bits_of_float g) (Int64.bits_of_float e)
                    else Float.(abs (g - e) <= 1e-4 * (1. + abs e)))
              then Bench_out.Passed
              else
                Bench_out.Failed
                  (if w.w_exact then "output differs from the default's bit for bit"
                   else "output differs from the default's beyond 1e-4 relative")
        in
        (match check with Bench_out.Failed _ -> Int.incr failed | _ -> ());
        let seconds = Option.value_map ms ~default:Float.nan ~f:(fun ms -> ms /. 1000.) in
        p "  %-48s %s\n" variant
          (Option.value_map ms ~default:"timing refused" ~f:(fun ms ->
               Printf.sprintf "%.3f us" (ms *. 1000.)));
        Bench_out.result ~variant ~check ~seconds;
        ms
  in
  let reference, default_ms =
    match run ~variant:"default" None with
    | Ok (values, _) as r -> (Some values, report ~variant:"default" ~reference:None r)
    | Error _ as r -> (None, report ~variant:"default" ~reference:None r)
  in
  let timed ~kind ~enum seeds =
    List.filter_mapi seeds ~f:(fun idx sp ->
        let variant =
          Printf.sprintf "%s %s %s" kind (flavor sp) (Autotune.sketch_params_label sp)
        in
        Option.map
          (report ~variant ~reference (run ~variant ~enum (Some (idx, sp))))
          ~f:(fun ms -> (flavor sp, ms)))
  in
  let prod_ms = timed ~kind:"prod" ~enum:production prod in
  let back_ms = timed ~kind:"rec" ~enum:recovered back in
  let best l = List.min_elt (List.map l ~f:snd) ~compare:Float.compare in
  let show = Option.value_map ~default:"-" ~f:(fun ms -> Printf.sprintf "%.3f us" (ms *. 1000.)) in
  p "summary %s: default %s; best production %s\n" w.w_name (show default_ms) (show (best prod_ms));
  List.iter
    (List.dedup_and_sort ~compare:String.compare (List.map back_ms ~f:fst))
    ~f:(fun cls ->
      p "summary %s: best recovered %s %s\n" w.w_name cls
        (show (best (List.filter back_ms ~f:(fun (c, _) -> String.equal c cls)))));
  if !failed > 0 then (
    p "companion_flavor_bench: %d variant(s) failed\n" !failed;
    Stdlib.exit 1)
