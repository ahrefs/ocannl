(* gh-ocannl-1458: the default CPU preset's own cross-nest rule and its per-dispatch work gate.

   The C backends run each pool-parallel [Grid] loop as one synchronous dispatch that joins before
   the next statement, and everything outside those loops once, on the calling thread. So sibling
   nests need no common geometry: [Schedule.default_cpu] proves, per nest, the ONE loop it retypes
   independent over its own iterations, leaves a nest it cannot prove serial without affecting the
   others, and parallelizes a loop only when one run of it does at least [cpu_schedule_min_work]
   scalar updates ([Cost_model.fork_join_updates], the renderer's launch-floor units).
   [Low_level.validate_parallel ~fork_join] admits the resulting mixed kernels: a write outside
   every [Grid] loop runs once.

   Legs 1-6 apply [Sched.default_cpu] (configured threshold, no optional arguments) DIRECTLY to one
   optimized hand-built kernel -- no fission, which would otherwise separate unschedulable or
   conflicting statements before the preset saw them -- and leg 7 runs the default pipeline
   ([Schedule.maybe_default_schedules], fission included), so the change is seen reaching
   production. Every leg executes the scheduled kernel against the unscheduled one, bit for bit:
   every cell keeps its reduction order. Producers are discriminating ([Ll_test.tick] / [tag]), and
   each leg checks the C backend renders every [Grid] loop the schedule placed on its pool, so the
   parity is not that of two serial runs. The affine legality crosscheck is on throughout. *)
open Base
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Tn = Ir.Tnode
module Generated = Test_utils.Generated
open Verdict.Claims

let backend_name = "cc"
let () = Unix.putenv "OCANNL_LEGALITY_CROSSCHECK" "true"
let () = Utils.settings.output_debug_files_in_build_directory <- true
let () = Generated.init ~backend_name
let node = Ll_test.node_factory ~first_id:1458000 ~dims:[| 1 |] ()

let mk ?(observe = true) label dims =
  let tn = node ~dims label in
  if observe then Ll_test.materialize tn;
  tn

open Ll_test

let rev n s = aff [ (-1, s) ] (n - 1)

(* Per top-level loop nest of [llc]: whether any loop in it is [Grid]-typed, and the nodes it
   writes. *)
let nests (llc : LL.t) =
  List.filter_map (LL.flat_lines [ llc ]) ~f:(function
    | LL.For_loop _ as nest ->
        let grid = ref false and ws = ref [] in
        Ll_test.walk nest
          ~on_stmt:(function LL.For_loop { axis = LL.Grid; _ } -> grid := true | _ -> ())
          ~on_set:(fun tn -> ws := tn :: !ws);
        Some (!grid, !ws)
    | _ -> None)

(* Per nest writing [tn]: whether it is [Grid]-typed. *)
let writers llc tn =
  List.filter_map (nests llc) ~f:(fun (g, ws) -> Option.some_if (List.mem ws tn ~equal:Tn.equal) g)

let grid_writer name llc tn = p_exists name (writers llc tn) ~f:Fn.id
let serial_writer name llc tn = p_all ~min:1 name (writers llc tn) ~f:not
let grid_loops llc = List.count (nests llc) ~f:fst

let pool_rendered src =
  List.length (String.substr_index_all src ~may_overlap:false ~pattern:"Pool-backed Grid rendering")

let seed_of tns = List.map tns ~f:(fun (tn, n, f) -> (tn, Array.init n ~f))

(* Optimizes [llc], schedules it directly with the configured default CPU preset, executes the
   scheduled and the unscheduled kernels, and returns the scheduled code, the scheduled compile's
   generated source, and both readbacks. *)
let direct ~name ~materialized ~seed ~read llc =
  let o = Ll_test.optimize ~materialized ~name llc in
  let scheduled = Sched.apply (Sched.default_cpu o) o in
  let got = Ll_test.execute ~ctx:(Context.cpu ()) ~name scheduled ~seed:(seed_of seed) ~read in
  let src = Generated.read name in
  let want =
    Ll_test.execute ~ctx:(Context.cpu ()) ~name:(name ^ "_serial") o ~seed:(seed_of seed) ~read
  in
  (scheduled.LL.llc, src, got, want)

let parity leg got want =
  List.iteri (List.zip_exn got want) ~f:(fun k (g, w) ->
      pf_all2 "%s: readback %d equals the unscheduled compile's, bit for bit" leg k g w
        ~f:Float.equal)

let renders_on_pool leg llc src =
  p
    (leg ^ ": every Grid loop renders on the pool")
    (grid_loops llc > 0 && pool_rendered src = grid_loops llc)

(* 1. A materialized producer read transposed by its consumer: the GPU alignment rule found no
   aligned prefix and serialized the whole kernel. *)
let () =
  let q = 128 in
  let a = mk "a" [| q; q |] and b = mk "b" [| q; q |] in
  let i = sym () and j = sym () and i' = sym () and j' = sym () in
  let llc =
    LL.unflat_lines
      [
        loop_n i q (loop_n j q (set a [| iter i; iter j |] (tag i j)));
        loop_n j' q
          (loop_n i' q (set b [| iter j'; iter i' |] (mul (get a [| iter i'; iter j' |]) (c 3.))));
      ]
  in
  let llc, src, got, want =
    direct ~name:"fj_transposed" ~materialized:[ a; b ] ~seed:[] ~read:[ a; b ] llc
  in
  grid_writer "transposed: the producer's nest is Grid-typed" llc a;
  grid_writer "transposed: the consumer's nest is Grid-typed" llc b;
  renders_on_pool "transposed" llc src;
  parity "transposed" got want

(* 2. Producer and consumer of different extents (the consumer reads two producer cells per
   iteration): no common-extent prefix, so the GPU rule bailed the kernel. *)
let () =
  let n = 40000 in
  let a = mk "a2" [| n |] and b = mk "b2" [| n / 2 |] in
  let i = sym () and k = sym () in
  let llc =
    LL.unflat_lines
      [
        loop_n i n (set_at a (iter i) (tick i));
        loop_n k (n / 2)
          (set_at b (iter k) (add (get a [| aff [ (2, k) ] 0 |]) (get a [| aff [ (2, k) ] 1 |])));
      ]
  in
  let llc, src, got, want =
    direct ~name:"fj_extents" ~materialized:[ a; b ] ~seed:[] ~read:[ a; b ] llc
  in
  grid_writer "extents: the producer's nest is Grid-typed" llc a;
  grid_writer "extents: the consumer's nest is Grid-typed" llc b;
  renders_on_pool "extents" llc src;
  parity "extents" got want

(* 3. Parallel, then serial, then parallel: a materialized whole-node zero and a scalar reduction
   (whose write mentions no loop symbol, so it has no candidate) between two parallel nests. The
   zero bailed the GPU access scan for the whole kernel, and hardware-thread validation rejects both
   serial statements beside a [Grid] loop. *)
let () =
  let n = 32768 in
  let a = mk "a3" [| n |] and s = mk "s3" [| 1 |] and c3 = mk "c3" [| n |] in
  let i = sym () and i2 = sym () and i3 = sym () in
  let llc =
    LL.unflat_lines
      [
        loop_n i n (set_at a (iter i) (tick i));
        zero s;
        loop_n i2 n (set_at s (fixed 0) (add (get s [| fixed 0 |]) (get a [| iter i2 |])));
        loop_n i3 n (set_at c3 (iter i3) (mul (get a [| iter i3 |]) (get s [| fixed 0 |])));
      ]
  in
  let llc, src, got, want =
    direct ~name:"fj_par_serial_par" ~materialized:[ a; s; c3 ] ~seed:[] ~read:[ a; s; c3 ] llc
  in
  grid_writer "par-serial-par: the producer's nest is Grid-typed" llc a;
  serial_writer "par-serial-par: the reduction stays serial" llc s;
  grid_writer "par-serial-par: the consumer's nest is Grid-typed" llc c3;
  renders_on_pool "par-serial-par" llc src;
  parity "par-serial-par" got want

(* 4. RAW, WAR and WAW between siblings through reversed reads: each nest is independent over its
   own iterations, and the joins order the nests -- a reordering would show in the values. *)
let () =
  let n = 16384 in
  let a = mk "a4" [| n |] and b = mk "b4" [| n |] in
  let i1 = sym () and i2 = sym () and i3 = sym () and i4 = sym () in
  let llc =
    LL.unflat_lines
      [
        loop_n i1 n (set_at a (iter i1) (tick i1));
        loop_n i2 n (set_at b (iter i2) (mul (get a [| rev n i2 |]) (c 3.)));
        loop_n i3 n (set_at a (iter i3) (add (get b [| rev n i3 |]) (c 2.)));
        loop_n i4 n (set_at b (iter i4) (add (get a [| iter i4 |]) (get b [| iter i4 |])));
      ]
  in
  let llc, src, got, want =
    direct ~name:"fj_siblings" ~materialized:[ a; b ] ~seed:[] ~read:[ a; b ] llc
  in
  p "siblings: all four nests are Grid-typed" (grid_loops llc = 4);
  renders_on_pool "siblings" llc src;
  parity "siblings" got want

(* 5. A loop-carried stencil beside an independent nest: the stencil fails its own proof and stays
   serial; under the GPU rule its failure serialized the sibling too. *)
let () =
  let n = 20000 in
  let a = mk "a5" [| n + 1 |] and b = mk "b5" [| n |] in
  let i = sym () and j = sym () in
  let llc =
    LL.unflat_lines
      [
        loop_n i n (set_at a (aff [ (1, i) ] 1) (add (get a [| iter i |]) (tick i)));
        loop_n j n (set_at b (iter j) (mul (tick j) (c 2.)));
      ]
  in
  let llc, src, got, want =
    direct ~name:"fj_hazard" ~materialized:[ a; b ]
      ~seed:[ (a, n + 1, fun x -> Float.of_int (x % 7) +. 0.5) ]
      ~read:[ a; b ] llc
  in
  serial_writer "hazard: the stencil stays serial" llc a;
  grid_writer "hazard: its independent sibling is Grid-typed" llc b;
  renders_on_pool "hazard" llc src;
  parity "hazard" got want

(* 6 (and 7). The work gate, over one kernel: equal output extents with a cheap pointwise body and a
   16-long reduction per cell (only the reduction pays for a dispatch, 2048 x 16 updates; the old
   element threshold, 2048 < 16384, refused both), a large copy (zero FLOPs, nonzero work, exactly
   at the threshold), a loop under a serial loop that may not borrow the enclosing iterations, and
   an extent-1 candidate over a heavy inner loop (never parallel). Nests that stay serial sit beside
   parallel ones. *)
let gate_program () =
  let m = 2048 and n = 16384 in
  let cheap = mk "cheap" [| m |] and costly = mk "costly" [| m |] in
  let x = mk "x" [| n |] and copy = mk "copy" [| n |] in
  let f = mk "f" [| 128 |] and g = mk "g" [| 64; 128 |] and h = mk "h" [| 1 |] in
  let ic = sym ()
  and ie = sym ()
  and ix = sym ()
  and r = sym ()
  and ir = sym ()
  and k = sym ()
  and kk = sym ()
  and i1 = sym () in
  let llc =
    LL.unflat_lines
      [
        loop_n ic m (set_at cheap (iter ic) (tick ic));
        zero costly;
        loop_n ie m
          (loop_n kk 16
             (set_at costly (iter ie)
                (add (get costly [| iter ie |]) (mul (get x [| iter kk |]) (tick ie)))));
        loop_n ix n (set_at copy (iter ix) (get x [| iter ix |]));
        zero f;
        loop_n r 64
          (loop_n ir 128
             (set_at f (iter ir) (add (get f [| iter ir |]) (get g [| iter r; iter ir |]))));
        zero h;
        loop_n i1 1
          (loop_n k 100000
             (set_at h (iter i1) (add (get h [| iter i1 |]) (mul (tick k) (c 0.001)))));
      ]
  in
  let seed =
    [
      (x, n, fun v -> Float.of_int (v % 13) -. 6.);
      (g, 64 * 128, fun v -> (Float.of_int (v % 11) *. 0.25) -. 1.);
    ]
  in
  (llc, [ cheap; costly; x; copy; f; g; h ], seed, (cheap, costly, copy, f, h))

let () =
  let llc, materialized, seed, (cheap, costly, copy, f, h) = gate_program () in
  let llc, src, got, want =
    direct ~name:"fj_gate" ~materialized ~seed ~read:[ cheap; costly; copy; f; h ] llc
  in
  serial_writer "gate: the cheap pointwise nest stays serial" llc cheap;
  grid_writer "gate: the reduction over the same output extent is Grid-typed" llc costly;
  grid_writer "gate: the large copy is Grid-typed" llc copy;
  serial_writer "gate: the loop under a serial loop does not borrow its iterations" llc f;
  serial_writer "gate: the extent-1 candidate stays serial" llc h;
  renders_on_pool "gate" llc src;
  parity "gate" got want

(* 7. The same program through the default pipeline: fission may split it, and every segment gets
   the preset. The gate's verdicts survive -- under the old per-kernel threshold the cheap nest was
   retyped beside the copy. *)
let () =
  let llc, materialized, seed, (cheap, costly, copy, f, h) = gate_program () in
  let name = "fj_pipeline" in
  let o = Ll_test.optimize ~materialized ~name llc in
  let segments = ref [] in
  let ctx, routine =
    Context.compile ~name ~prelowered:o
      ~lowered_transform:(fun opt ->
        let segs = Sched.maybe_default_schedules ~backend_name ~static_indices:[] opt in
        segments := segs;
        segs)
      (Context.cpu ()) Ir.Assignments.empty_comp Ir.Indexing.Empty
  in
  let ctx = Ll_test.run_linked (ctx, routine) ~seed:(seed_of seed) in
  let read = [ cheap; costly; copy; f; h ] in
  let got = List.map read ~f:(Context.get_values ctx) in
  let src = Generated.read (if List.length !segments > 1 then name ^ "__seg" else name) in
  let want =
    Ll_test.execute ~ctx:(Context.cpu ()) ~name:(name ^ "_serial") o ~seed:(seed_of seed) ~read
  in
  let all = LL.unflat_lines (List.map !segments ~f:(fun (s : LL.optimized) -> s.LL.llc)) in
  serial_writer "pipeline: the cheap pointwise nest stays serial" all cheap;
  grid_writer "pipeline: the reduction is Grid-typed" all costly;
  grid_writer "pipeline: the large copy is Grid-typed" all copy;
  renders_on_pool "pipeline" all src;
  parity "pipeline" got want
