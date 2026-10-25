(* gh-ocannl-1343: on the C backends, a scratch intermediate -- a node neither inlined nor
   materialized, so [Local]: a function-scope array -- no longer leaves a matmul segment serial.

   Two defects, fixed together:

   - Fission grouped a scratch node's whole-node [Zero_out] with the PRECEDING segment: a zero of
   scratch conflicts with nothing, so the greedy grouping took it, and then cut before the nest
   accumulating into the node. The node crossed that boundary, was promoted [On_device], and the
   promoted zero -- a bare materialized write -- bailed the annotation of BOTH segments, which
   coalesced back into one serial kernel. Whatever the preceding segment held went serial with it:
   in mlp_bn_names, the first layer's 1000x200x30 matmul, ahead of the batch-norm statistics. -
   [Schedule.default_cpu] took chains only from materialized writes, so a nest writing only scratch
   -- a matmul whose product is a non-materialized intermediate -- stayed serial even in a segment
   of its own (on GPU it runs whole per thread over a per-thread copy, but on the C backends a
   serial nest runs once, on one core).

   Legs, each on an explicit [Context.cpu ()], each executed against an unscheduled (serial) compile
   of the same computation -- bit for bit, since every cell keeps its reduction order -- and each
   checking that the C backend renders every [Grid] loop the schedule placed on its pool:

   1. [stats], through the default pipeline ([Schedule.maybe_default_schedules], the transform
   backend [compile] applies, at the configured thresholds): a 128x128 matmul [h] (materialized: it
   exceeds the stack threshold) followed by row statistics [s] (scratch) and [y = h - s]. The
   matmul's accumulation nest is [Grid]-typed, the statistics stay scratch, and their consumer keeps
   its [Grid]: linking the scratch writer [s[i]] to its reader [y[i, j]] trims the reader's chain to
   [i], below [cpu_schedule_min_parallel], so the preset keeps the plain analysis there.

   1b. [rowstats], under the tuner's presets ([min_parallel] 1): the same shape over a materialized
   32x32 input. The trim to [i] keeps the kernel eligible, so the scratch statistics' nest is
   [Grid]-typed beside its consumer -- comparing chain sizes instead of eligibility would fall back
   and leave it serial.

   2. [chain], through the default pipeline at [min_parallel] 1 (the presets the tuner seeds and
   times its fissioned candidates with): the issue's three layers [y_k = relu (w_k * y_{k-1})] at
   32x32, every [y_k] materialized and only the last product, so the first two products are scratch.
   Every product's accumulation nest is [Grid]-typed and the scratch products stay scratch.

   3. [transposed], same presets: a scratch product its consumer reads transposed cannot be linked
   to it by the alignment rule, so the preset falls back to the plain analysis -- the consumer keeps
   its [Grid] and the producer stays serial, as before the fix. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Tn = Ir.Tnode
module Asgns = Ir.Assignments
module Generated = Test_utils.Generated
open Verdict.Claims

let backend_name = "cc"
let () = Utils.settings.output_debug_files_in_build_directory <- true
let () = Generated.init ~backend_name

(* These legs pin which Grid loops the C backend CAN render on its pool (the safety rules), not
   whether a launch is worth its fork/join: the pool-launch work floor ([cc_grid_min_updates],
   gh-ocannl-1564) would serialize their small nests, so it is disabled for every compile here. *)
let () = Hashtbl.set Utils.config_file_args ~key:"cc_grid_min_updates" ~data:"0"

let matrix ~q ~label ~modulus ~offset ~stride =
  TDSL.ndarray
    (Array.init (q * q) ~f:(Ll_test.cycle_flat ~dims:[| q; q |] ~modulus ~offset ~stride))
    ~label:[ label ] ~input_dims:[ q ] ~output_dims:[ q ] ()

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* The tensor nodes [llc] writes. *)
let written (llc : LL.t) =
  let acc = ref [] in
  Ll_test.walk llc ~on_set:(fun tn -> acc := tn :: !acc);
  !acc

(* Per top-level loop nest of every segment: the axis of its outermost loop and the nodes it
   writes. *)
let nests segments =
  List.concat_map segments ~f:(fun (seg : LL.optimized) ->
      List.filter_map (LL.flat_lines [ seg.LL.llc ]) ~f:(function
        | LL.For_loop { axis; _ } as nest -> Some (axis, written nest)
        | _ -> None))

let is_grid axis = LL.equal_axis_type axis LL.Grid
let grid_loops segments = List.count (nests segments) ~f:(fun (axis, _) -> is_grid axis)

(* The outermost-loop axes of the nests writing [tn]. *)
let writer_axes segments tn =
  List.filter_map (nests segments) ~f:(fun (axis, ws) ->
      Option.some_if (List.mem ws tn ~equal:Tn.equal) axis)

(* A nest writing [tn] is Grid-typed at its outermost loop. *)
let grid_writer segments tn = List.exists (writer_axes segments tn) ~f:is_grid

let pool_rendered src =
  List.length (String.substr_index_all src ~may_overlap:false ~pattern:"Pool-backed Grid rendering")

(* The default pipeline: as configured, or with the presets at [min_parallel] 1. *)
let schedules ~tuner_presets opt =
  if tuner_presets then
    List.map
      (Sched.fission_scheduled
         ~preset:(Sched.default_cpu ~min_parallel:1)
         ~zero_sched:(fun _ -> [])
         ~static_indices:[] opt)
      ~f:(fun (_, _, _, post) -> post)
  else Sched.maybe_default_schedules ~backend_name ~static_indices:[] opt

(* Compiles and runs [comp] twice: scheduled (capturing the segments, and whether each [scratch]
   node is still scratch after scheduling) and unscheduled. Returns the segments, the scratch flags,
   the scheduled compile's generated source and both results. *)
let scheduled_and_serial ~tuner_presets ~name ~scratch ~result comp =
  let segments = ref [] and still_scratch = ref [] in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        let segs = schedules ~tuner_presets opt in
        segments := segs;
        let plc = opt.LL.optimize_ctx.LL.placements in
        still_scratch := List.map scratch ~f:(Tn.Placements.is_local_peek plc);
        segs)
      (Context.cpu ()) (named name comp) Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx result in
  Context.release ctx;
  (* A fissioned routine emits its segments into one [<name>__seg] source. *)
  let src = Generated.read (if List.length !segments > 1 then name ^ "__seg" else name) in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun opt -> [ opt ])
      (Context.cpu ())
      (named (name ^ "_serial") comp)
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let want = Context.get_values ctx result in
  Context.release ctx;
  (!segments, !still_scratch, src, got, want)

let () =
  let q = 128 in
  let w = matrix ~q ~label:"w" ~modulus:13 ~offset:(-6.) ~stride:0.5 in
  let x = matrix ~q ~label:"x" ~modulus:17 ~offset:(-8.) ~stride:0.25 in
  let%op h = w * x in
  let%op s = h ++ "i->o => o" in
  let%op y = h - s in
  Train.set_materialized y.Tensor.value;
  let segments, still_scratch, src, got, want =
    scheduled_and_serial ~tuner_presets:false ~name:"scratch_stats" ~scratch:[ s.Tensor.value ]
      ~result:y.Tensor.value (Train.forward y)
  in
  p "stats: the matmul's accumulation nest is Grid-typed" (grid_writer segments h.Tensor.value);
  p_all "stats: the statistics stay scratch" still_scratch ~f:Fn.id;
  p "stats: the statistics' consumer keeps its Grid" (grid_writer segments y.Tensor.value);
  p "stats: every Grid loop renders on the pool"
    (grid_loops segments > 0 && pool_rendered src = grid_loops segments);
  p_all2 "stats: values equal the unscheduled compile's, bit for bit" got want ~f:Float.equal

let () =
  let q = 32 in
  let xa = matrix ~q ~label:"xa" ~modulus:11 ~offset:0. ~stride:0.125 in
  let xb = matrix ~q ~label:"xb" ~modulus:7 ~offset:(-3.) ~stride:1. in
  let%op x = xa + xb in
  Train.set_materialized x.Tensor.value;
  let%op s = x ++ "i->o => o" in
  let%op y = x - s in
  Train.set_materialized y.Tensor.value;
  let segments, still_scratch, src, got, want =
    scheduled_and_serial ~tuner_presets:true ~name:"scratch_rowstats" ~scratch:[ s.Tensor.value ]
      ~result:y.Tensor.value (Train.forward y)
  in
  p_all "rowstats: the statistics are scratch" still_scratch ~f:Fn.id;
  p "rowstats: the statistics' nest is Grid-typed" (grid_writer segments s.Tensor.value);
  p "rowstats: their consumer keeps its Grid" (grid_writer segments y.Tensor.value);
  p "rowstats: every Grid loop renders on the pool"
    (grid_loops segments > 0 && pool_rendered src = grid_loops segments);
  p_all2 "rowstats: values equal the unscheduled compile's, bit for bit" got want ~f:Float.equal

let () =
  let q = 32 in
  let qa = matrix ~q ~label:"qa" ~modulus:11 ~offset:0. ~stride:0.125 in
  let qb = matrix ~q ~label:"qb" ~modulus:7 ~offset:(-3.) ~stride:1. in
  let w1 = matrix ~q ~label:"w1" ~modulus:5 ~offset:(-2.) ~stride:0.5 in
  let w2 = matrix ~q ~label:"w2" ~modulus:9 ~offset:(-4.) ~stride:0.25 in
  let w3 = matrix ~q ~label:"w3" ~modulus:3 ~offset:(-1.) ~stride:0.5 in
  let layer ~materialize (w : Tensor.t) (y : Tensor.t) =
    let%op h = w * y in
    if materialize then Train.set_materialized h.Tensor.value;
    let%op y = relu h in
    Train.set_materialized y.Tensor.value;
    (y, h)
  in
  let%op x = qa + qb in
  Train.set_materialized x.Tensor.value;
  let y1, h1 = layer ~materialize:false w1 x in
  let y2, h2 = layer ~materialize:false w2 y1 in
  let y3, h3 = layer ~materialize:true w3 y2 in
  let segments, still_scratch, src, got, want =
    scheduled_and_serial ~tuner_presets:true ~name:"scratch_chain"
      ~scratch:[ h1.Tensor.value; h2.Tensor.value ]
      ~result:y3.Tensor.value (Train.forward y3)
  in
  p_all "chain: every product's accumulation nest is Grid-typed"
    [ h1.Tensor.value; h2.Tensor.value; h3.Tensor.value ]
    ~f:(grid_writer segments);
  p_all "chain: the first two products stay scratch" still_scratch ~f:Fn.id;
  p "chain: every Grid loop renders on the pool"
    (grid_loops segments > 0 && pool_rendered src = grid_loops segments);
  p_all2 "chain: values equal the unscheduled compile's, bit for bit" got want ~f:Float.equal

let () =
  let q = 32 in
  let wa = matrix ~q ~label:"wa" ~modulus:13 ~offset:(-6.) ~stride:0.5 in
  let xa = matrix ~q ~label:"xa" ~modulus:17 ~offset:(-8.) ~stride:0.25 in
  let%op h = wa * xa in
  let%op y = relu (h ++ "i->o => o->i") in
  Train.set_materialized y.Tensor.value;
  let segments, still_scratch, src, got, want =
    scheduled_and_serial ~tuner_presets:true ~name:"scratch_transposed" ~scratch:[ h.Tensor.value ]
      ~result:y.Tensor.value (Train.forward y)
  in
  p_all "transposed: the product is scratch" still_scratch ~f:Fn.id;
  p "transposed: the consumer keeps its Grid" (grid_writer segments y.Tensor.value);
  p_all "transposed: the scratch producer stays serial"
    (writer_axes segments h.Tensor.value)
    ~f:(Fn.non is_grid);
  p "transposed: every Grid loop renders on the pool"
    (grid_loops segments > 0 && pool_rendered src = grid_loops segments);
  p_all2 "transposed: values equal the unscheduled compile's, bit for bit" got want ~f:Float.equal

(* 4. [solo], hand-built IR under the tuner's presets: a parallel nest writing [x], then a scratch
   zero, then a unit the analysis can only run serially ([`Solo]: it accumulates [m[0]], a
   materialized write no chain loop covers) that accumulates into the scratch node. The zero must go
   with that unit rather than join the preceding segment, or the node crosses the cut, its promotion
   makes the zero a bare materialized write, and the [x] nest loses its [Grid] (review round 2). *)
let () =
  let n = 32 in
  let node = Ll_test.node_factory ~first_id:13430 ~dims:[| n; n |] () in
  let a = node "sa" and x = node "sx" and m = node ~dims:[| 1 |] "sm" in
  let s = node ~dims:[| n |] "ss" in
  List.iter [ a; x; m ] ~f:Ll_test.materialize;
  let i = Ll_test.sym () and j = Ll_test.sym () and i2 = Ll_test.sym () and j2 = Ll_test.sym () in
  let open Ll_test in
  let program =
    seq
      (loop_n i n
         (loop_n j n (set x [| iter i; iter j |] (add (get a [| iter i; iter j |]) (c 1.)))))
      (seq (zero s)
         (loop_n i2 n
            (seq
               (loop_n j2 n
                  (set s [| iter i2 |] (add (get s [| iter i2 |]) (get x [| iter i2; iter j2 |]))))
               (set m [| fixed 0 |] (add (get m [| fixed 0 |]) (get s [| iter i2 |]))))))
  in
  let o = optimize ~materialized:[ a; x; m ] ~name:"scratch_solo" program in
  p "solo: the accumulated node is scratch"
    (not (Tn.Placements.is_materialized_peek o.LL.optimize_ctx.LL.placements s));
  let segments =
    List.map
      (Sched.fission_scheduled
         ~preset:(Sched.default_cpu ~min_parallel:1)
         ~zero_sched:(fun _ -> [])
         ~static_indices:[] o)
      ~f:(fun (_, _, _, post) -> post)
  in
  p "solo: the preceding nest keeps its Grid" (grid_writer segments x);
  p_all "solo: the serial unit stays serial" (writer_axes segments m) ~f:(Fn.non is_grid)
