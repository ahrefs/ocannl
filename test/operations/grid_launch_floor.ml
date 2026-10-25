(* gh-ocannl-1564: every pool-parallel CPU Grid launch must amortize its native fork/join
   ([C_syntax.grid_min_updates], config [cc_grid_min_updates]) -- top-level launches too, not only
   the repeated inner ones gh-ocannl-933 floored. [bin/narrow_gebp_bench]'s [packmma_par] (the
   packed GEBP with Grid row blocks outermost, built here by the bench's own [Bench_gebp]) ran about
   10x slower than its serial twin at n <= 32 on rog's hybrid CPU: two fork/joins per call (the
   zeroing nest and the compute nest) against a microsecond of work.

   At the issue's own cell, 32^3 at 16 x 32, the default floor renders both launches as serial
   loops, while the values stay bitwise the unscheduled kernel's and the [Tile_mma] still renders
   the register tiling. The same schedule with the floor disabled ([cc_grid_min_updates=0]) renders
   pool-parallel: the floor, and no other decline, is what serialized it. At 128^3 the same layout
   clears the floor and stays pool-parallel. And the [Tile_mma] weighting is what puts the small
   compute launch under the floor: its [Tile_mma] MACs alone, charged at par, reach it.

   The operands are the bench's own ([Bench_gebp.a_value]/[b_value]): they vary with every index
   pair and their products are small multiples of 1/8, so every rendering's reduction is exact and
   bitwise is the right claim. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
module LL = Ir.Low_level
module Generated = Test_utils.Generated
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_cpu = Sched.backend_is_cpu backend_name
let () = Utils.settings.output_debug_files_in_build_directory <- true
let () = Generated.init ~backend_name

let has_parallel_construct src =
  String.is_substring src ~substring:"dispatch_apply"
  || String.is_substring src ~substring:"#pragma omp parallel for"

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* The floor is read per compile from the configuration; a leg states it for its own compile and
   restores what was there. *)
let with_floor value f =
  let key = "cc_grid_min_updates" in
  let prior = Hashtbl.find Utils.config_file_args key in
  Hashtbl.set Utils.config_file_args ~key ~data:value;
  Exn.protect ~f ~finally:(fun () ->
      match prior with
      | Some v -> Hashtbl.set Utils.config_file_args ~key ~data:v
      | None -> Hashtbl.remove Utils.config_file_args key)

(* The outermost Grid loops of a scheduled procedure: the launches the floor judges. *)
let grid_launches llc =
  let is_grid (site : Ll_test.loop_site) =
    match site.ls_stmt with LL.For_loop { axis = Grid; _ } -> true | _ -> false
  in
  List.filter_map (Ll_test.loop_sites llc) ~f:(fun site ->
      Option.some_if (is_grid site && not (List.exists site.ls_ancestors ~f:is_grid)) site.ls_stmt)

(* The multiply-accumulates the [Tile_mma] statements of [launch] cover per launch: each tile's [m x
   n x k] times the extents of the loops enclosing it. *)
let launch_macs launch =
  let multiplicity = ref [ 1 ] and total = ref 0 in
  Ll_test.walk launch
    ~on_stmt:(fun s ->
      match s with
      | LL.For_loop { from_; to_; _ } ->
          multiplicity := ((to_ - from_ + 1) * List.hd_exn !multiplicity) :: !multiplicity
      | LL.Tile_mma { m; n; k; _ } -> total := !total + (m * n * k * List.hd_exn !multiplicity)
      | _ -> ())
    ~on_exit:(fun s ->
      match s with LL.For_loop _ -> multiplicity := List.tl_exn !multiplicity | _ -> ());
  !total

(* One compile and run of the [n^3] product: unscheduled ([layout = None]) or under the bench's
   packed builder. Returns the readback, the [Tile_mma] census, the generated source and the
   scheduled procedure. *)
let run ~name ~n ~bm ~bk ~layout =
  let ma =
    NTDSL.init ~l:"ma" ~prec:Ir.Ops.single ~i:[ n ] ~o:[ n ]
      ~f:(fun idcs -> Bench_gebp.a_value idcs.(0) idcs.(1))
      ()
  in
  let mb =
    NTDSL.init ~l:"mb" ~prec:Ir.Ops.single ~i:[ n ] ~o:[ n ]
      ~f:(fun idcs -> Bench_gebp.b_value idcs.(0) idcs.(1))
      ()
  in
  let%op mc = ma * mb in
  let scheduled = ref None in
  let transform opt =
    match layout with
    | None -> opt
    | Some layout ->
        let s =
          Bench_gebp.packed_schedule ~layout ~tile_prec:None ~bm ~bk ~a:ma.Tensor.value
            ~b:mb.Tensor.value ~c:mc.Tensor.value opt
        in
        let opt = Sched.apply s opt in
        scheduled := Some opt.LL.llc;
        opt
  in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun o -> [ transform o ])
      (Context.auto ())
      (named name (Train.forward mc))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  ( Context.get_values ctx mc.Tensor.value,
    Ll_test.renderings routine.Context.mma,
    Generated.read name,
    !scheduled )

let grid = Some Bench_gebp.Grid_outermost

let () =
  if not on_cpu then
    List.iter
      [
        "32^3 at 16x32, default floor: both Grid launches render serial";
        "32^3 at 16x32, default floor: values match the unscheduled kernel bitwise";
        "32^3 at 16x32, default floor: every Tile_mma renders the register tiling";
        "32^3 at 16x32, floor disabled: the launches render pool-parallel";
        "32^3 at 16x32, floor disabled: values match the unscheduled kernel bitwise";
        "128^3 at 16x32, default floor: the launches render pool-parallel";
        "128^3 at 16x32, default floor: values match the unscheduled kernel bitwise";
        "32^3 at 16x32: every Grid launch estimates below the default floor";
        "32^3 at 16x32: with MACs charged at par, the Tile_mma launch reaches the default floor";
      ]
      ~f:(skipped ~backend:backend_name)
  else
    (* Without a native pool (cc_parallel_grid resolved to none) no Grid renders parallel at all, so
       the legs that expect a parallel construct are skipped; which branch ran goes to stderr. *)
    let pool = Context.Cc_backend.pool_parallel_grid () in
    Stdio.eprintf "cc pool-parallel grid: %b (not part of the golden)\n%!" pool;
    let floor = Ir.C_syntax.default_grid_min_updates in
    let want32, _, _, _ = run ~name:"glf_ref32" ~n:32 ~bm:16 ~bk:32 ~layout:None in
    let got32, census32, src32, sched32 = run ~name:"glf_small" ~n:32 ~bm:16 ~bk:32 ~layout:grid in
    p "32^3 at 16x32, default floor: both Grid launches render serial"
      (not (has_parallel_construct src32));
    p_all2 "32^3 at 16x32, default floor: values match the unscheduled kernel bitwise" got32 want32
      ~f:Float.equal;
    p_all "32^3 at 16x32, default floor: every Tile_mma renders the register tiling" census32
      ~f:(fun r -> Poly.equal r Ir.C_syntax.Mma_register_tiled);
    let got32_off, _, src32_off, _ =
      with_floor "0" (fun () -> run ~name:"glf_small_nofloor" ~n:32 ~bm:16 ~bk:32 ~layout:grid)
    in
    gated ~when_:pool ~on:backend_name
      "32^3 at 16x32, floor disabled: the launches render pool-parallel"
      (has_parallel_construct src32_off);
    p_all2 "32^3 at 16x32, floor disabled: values match the unscheduled kernel bitwise" got32_off
      want32 ~f:Float.equal;
    let want128, _, _, _ = run ~name:"glf_ref128" ~n:128 ~bm:16 ~bk:32 ~layout:None in
    let got128, _, src128, _ = run ~name:"glf_large" ~n:128 ~bm:16 ~bk:32 ~layout:grid in
    gated ~when_:pool ~on:backend_name
      "128^3 at 16x32, default floor: the launches render pool-parallel"
      (has_parallel_construct src128);
    p_all2 "128^3 at 16x32, default floor: values match the unscheduled kernel bitwise" got128
      want128 ~f:Float.equal;
    let launches = Option.value_map sched32 ~default:[] ~f:grid_launches in
    p_all "32^3 at 16x32: every Grid launch estimates below the default floor" launches ~f:(fun l ->
        Ir.C_syntax.grid_update_count ~cap:floor l < floor);
    p_all "32^3 at 16x32: with MACs charged at par, the Tile_mma launch reaches the default floor"
      (List.filter launches ~f:(fun l -> launch_macs l > 0))
      ~f:(fun l -> launch_macs l >= floor)
