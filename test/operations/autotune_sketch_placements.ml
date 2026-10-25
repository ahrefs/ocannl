(* gh-ocannl-1243: a per-segment sketch is for the segments it was seeded on, found by position.

   [Autotune.tune]'s per-segment sketch candidates ([F_sketch]) seed each matmul segment of the
   routine's fission and time each parameter set alone, every other segment on its default preset.
   They used to find their segment by its structural pre-schedule digest alone, which leaves
   placements out: a sketch seeded on one segment also landed on every segment of the same
   structure, however differently placed, so its window timed (and its label named) a schedule
   nobody seeded. The candidates now replay the segmentation their seeds were enumerated under and
   find their segments in it by position; segments of one structure share a sketch only when their
   placements agree too -- one code, one sketch, as for a model's repeated layers.

   The workload is three layers [y = relu (w * h)] whose products the fission keeps in a segment of
   their own each, the first two of one structure (the twins). What a candidate scheduled is read
   off its schedule digest, one component per segment, against a single seeded on the third layer: a
   twin's component differs from that single's exactly where the candidate sketched it.

   1. [differ]: the second layer's product is materialized and the first's is not, so the twins
   differ in placements -- on a GPU backend, where fission restores an unneeded [Local] promotion.
   On the C backends the first product stays [Local] too, but its whole-node zero opens its own
   accumulation's segment (gh-ocannl-1343), so the two products' segments differ in structure and
   there are no twins: the leg is skipped there. Each twin gets its own singles, and no single
   sketches both.

   The first twin's [Local] product keeps the layer's [relu] in the twin's kernel as a companion the
   unfused sketches cannot cover (GPU companion coverage, gh-ocannl-521), so only the epilogue-fused
   twins seed it. Since gh-ocannl-1325 the scalar block-tiling twins compile too ([Fuse_epilogue]
   used to refuse that pipeline's register-tiled store path, which left the twin unseeded wherever
   the backend advertises no tensorized pipeline for the site's f32 triple: CUDA at the default
   [tf32_matmuls], HIP), so the claim that the first twin gets a single of its own holds on every
   GPU backend. The second twin and the third layer have their products materialized, so the scalar
   pipeline seeds them unfused: the third layer's single is the reference the claims read the twins
   against (a [Local] third product would carry the same companion, so only its fused twins would
   seed it).

   2. [agree]: every product is materialized, so the twins agree in placements as well: they share
   one key, and every single that sketches one twin sketches both.

   Every timing window is scripted ([Autotune.on_candidate_measured], with
   [Autotune.with_uncontended_test_windows]): the claims are about what was compiled, never about
   what was fast.

   On CPU backends the searches run on [Fissioned_chain]'s small menu (the roofline pre-filter
   keeping a fifth of each segment's sketches; fourteen singles per key instead of 38 on cc, at the
   vector width the small menu pins). The claims ask which segments a single sketched, never which
   parameters it carries, so they hold for any single the model keeps; they need a single on each
   key, and [Fissioned_chain.menu_claim] fails loudly if the pruned [agree] search ever keys other
   than two segments or offers fewer than three singles on one. GPU backends keep the whole menu:
   their sketch families are their own, and the [differ] leg they run keys three segments. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module SC = Ir.Schedule_cache
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let is_gpu = Sched.backend_is_gpu backend_name
let is_cpu = Sched.backend_is_cpu backend_name

(* Before the first search: the pre-filter reads its envelope once per process. *)
let () = if is_cpu then Fissioned_chain.small_menu ()
let q = 32

let matrix ~label ~modulus ~offset ~stride =
  TDSL.ndarray
    (Array.init (q * q) ~f:(Ll_test.cycle_flat ~dims:[| q; q |] ~modulus ~offset ~stride))
    ~label:[ label ] ~input_dims:[ q ] ~output_dims:[ q ] ()

(* Three layers [y_k = relu (w_k * y_{k-1})] over a materialized input sum; [materialized] lists the
   layers whose product is materialized. The third layer closes the routine, so its segment carries
   the closing comment and is not a twin of the first two. *)
let chain ~materialized () =
  let qa = matrix ~label:"qa" ~modulus:11 ~offset:0. ~stride:0.125 in
  let qb = matrix ~label:"qb" ~modulus:7 ~offset:(-3.) ~stride:1. in
  let w1 = matrix ~label:"w1" ~modulus:5 ~offset:(-2.) ~stride:0.5 in
  let w2 = matrix ~label:"w2" ~modulus:9 ~offset:(-4.) ~stride:0.25 in
  let w3 = matrix ~label:"w3" ~modulus:3 ~offset:(-1.) ~stride:0.5 in
  let layer k (w : Tensor.t) (y : Tensor.t) =
    let%op h = w * y in
    if List.mem materialized k ~equal:Int.equal then Train.set_materialized h.Tensor.value;
    let%op y = relu h in
    (y, h)
  in
  let%op x = qa + qb in
  Train.set_materialized x.Tensor.value;
  let y1, _ = layer 1 w1 x in
  Train.set_materialized y1.Tensor.value;
  let y2, _ = layer 2 w2 y1 in
  Train.set_materialized y2.Tensor.value;
  Train.forward (fst (layer 3 w3 y2))

(* A hermetic copy: fission promotes placements of the record it is given. *)
let copy (o : LL.optimized) =
  {
    o with
    LL.traced_store = Hashtbl.copy o.LL.traced_store;
    LL.optimize_ctx = LL.copy_optimize_ctx o.LL.optimize_ctx;
  }

(* The segmentation the tuner enumerates its per-segment seeds under -- its fission settings and
   aggressive presets -- with each segment's structural and placement-aware pre-schedule digests. *)
let enumerate comp =
  let captured = ref None in
  let ctx, _routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        captured := Some (copy opt);
        [ opt ])
      (Context.auto ()) comp Ir.Indexing.Empty
  in
  let limits = Context.hardware_limits ctx in
  Context.release ctx;
  let base = Option.value_exn ~here:[%here] !captured in
  let preset o =
    if is_gpu then Sched.default_gpu ~min_parallel:1 ~limits o else Sched.default_cpu ~min_work:0 o
  in
  let zero_sched tns = if is_gpu then Sched.zero_expansion ~limits tns else [] in
  let _plan, tuples =
    Sched.fission_segmented ~promote_locals:is_gpu ~fold_zeros:true
      ?keep_mapping:(Sched.fission_keep_mapping ~is_gpu ~limits)
      ~preset ~zero_sched ~static_indices:[] (copy base)
  in
  List.map tuples ~f:(fun (kind, pre, _, _) ->
      let digest with_placements = SC.digest (SC.canonicalize ~with_placements pre) in
      (kind, digest false, digest true))

(* The twins: the two [`Normal] segments of one structural digest, by position. *)
let twins segments =
  List.find_mapi segments ~f:(fun i (kind, d, _) ->
      match kind with
      | `Zeros | `Solo -> None
      | `Normal ->
          List.find_mapi segments ~f:(fun j (kind', d', _) ->
              Option.some_if (j > i && Poly.equal kind' `Normal && String.equal d d') (i, j)))

(* One search, every window scripted alike; the per-segment singles of the coarse segmentation it
   timed, as (label, schedule digest components). *)
let singles comp =
  let (o : Scripted_search.observed) =
    Scripted_search.search ~ctx:(Context.auto ()) ~beam_width:1 ~log:false ~progress:false
      ~tag:"singles"
      ~measure:(fun _ -> 1.)
      comp
  in
  List.map o.windows ~f:(fun (w, _) -> (w.label, w.digest))
  |> List.filter_map ~f:(fun (label, digest) ->
      if
        String.is_prefix label ~prefix:"F_sketch["
        && (not (String.is_prefix label ~prefix:"F_sketch[fine "))
        && not (String.mem label ',')
      then Some (label, String.split digest ~on:'+')
      else None)

(* The segments a single sketched: where its schedule digest departs from a reference single's,
   seeded on another segment, at the twins' positions -- there the reference keeps the preset. *)
let sketched ~reference (i, j) (_, components) =
  List.filter [ i; j ] ~f:(fun k ->
      match (List.nth components k, List.nth reference k) with
      | Some c, Some r -> not (String.equal c r)
      | _ -> true)

(* One leg's search over the chain, with what its claims read: the twins' positions, whether they
   agree in placements, the timed singles, and the twins each single sketched. *)
let leg ~tag ~materialized ~when_ =
  let segments = if when_ then enumerate (chain ~materialized ()) else [] in
  let found = twins segments in
  gated ~when_ ~on:backend_name
    (tag ^ ": two matmul segments share a structural digest")
    (Option.is_some found);
  let ((i, j) as pair) = Option.value found ~default:(0, 0) in
  let digests k =
    Option.value_map (List.nth segments k) ~default:("", "") ~f:(fun (_, d, p) -> (d, p))
  in
  let singles = if when_ then singles (chain ~materialized ()) else [] in
  Stdio.eprintf "%s singles (not part of the golden): %d\n%!" tag (List.length singles);
  List.iter singles ~f:(fun (l, _) -> Stdio.eprintf "  %s\n%!" l);
  gated_all ~when_ ~on:backend_name (tag ^ ": every single compiles the enumerated segmentation")
    singles ~f:(fun (_, components) -> List.length components = List.length segments);
  let is_twin_single (label, _) =
    String.is_substring label ~substring:("@" ^ String.prefix (fst (digests i)) 8)
  in
  gated_exists ~when_ ~on:backend_name
    (tag ^ ": a single is seeded on another segment")
    singles ~f:(Fn.non is_twin_single);
  let reference =
    Option.value_map ~default:[] ~f:snd (List.find singles ~f:(Fn.non is_twin_single))
  in
  (pair, String.equal (snd (digests i)) (snd (digests j)), singles, sketched ~reference pair)

let () =
  let when_ = is_gpu and on = backend_name in
  let (i, j), placed_alike, singles, sketched = leg ~tag:"differ" ~materialized:[ 2; 3 ] ~when_ in
  gated ~when_ ~on "differ: the twins differ in placements" (not placed_alike);
  gated_exists ~when_ ~on "differ: a single sketches the first twin alone" singles ~f:(fun s ->
      List.equal Int.equal (sketched s) [ i ]);
  gated_exists ~when_ ~on "differ: a single sketches the second twin alone" singles ~f:(fun s ->
      List.equal Int.equal (sketched s) [ j ]);
  gated_all ~when_ ~on "differ: no single sketches both twins" singles ~f:(fun s ->
      List.length (sketched s) < 2)

let () =
  let _, placed_alike, singles, sketched = leg ~tag:"agree" ~materialized:[ 1; 2; 3 ] ~when_:true in
  Fissioned_chain.menu_claim ~when_:is_cpu ~on:backend_name (List.map singles ~f:fst);
  p "agree: the twins agree in placements" placed_alike;
  p_exists "agree: a single sketches the twins" singles ~f:(fun s ->
      not (List.is_empty (sketched s)));
  p_all "agree: every single that sketches one twin sketches both"
    (List.filter singles ~f:(fun s -> not (List.is_empty (sketched s))))
    ~f:(fun s -> List.length (sketched s) = 2)
