(* gh-ocannl-679: the pre-driver launch gate covers every hardware dimension of the launch, not only
   the workgroup's thread PRODUCT.

   The gap this pins closed: [max_threads_per_workgroup] caps the product of the block's three
   dimensions, and nothing capped them individually. CUDA's [maxThreadsDim] is (1024, 1024, 64) --
   the [.z] component is 16x smaller than the product cap -- and Metal read only the [width] of its
   3-D [maxThreadsPerThreadgroup]. So a workgroup of 2 x 2 x 128 has a perfectly legal 512-thread
   product, passes every check, compiles, and then fails at the driver with an opaque
   invalid-configuration error, which is exactly the failure mode the gate exists to remove.

   [Workgroup] slots are capped at 3 and the innermost binds [.x], so the outermost annotated loop's
   extent lands on [.z] directly -- no fold, no exotic schedule. No in-tree annotator emits three
   nested [Workgroup] loops (both [Schedule.default_gpu] and [Schedule.zero_expansion] emit exactly
   one per nest), so the geometry is built here directly as [Ir.Low_level.t] through the [Ll_test]
   harness, annotated by hand, and judged against MOCKED [hardware_limits] records. Mocked
   deliberately: the point is a device whose per-dimension cap is below its product cap, and the
   only such device in the fleet is a CUDA one -- gfx1151 and every Apple part report their three
   components equal to the product cap, so a real-limits run on this machine could not tell a
   working gate from a missing one.

   Every claim comes in a pair: the cap that refuses and the cap that accepts. Without the accepting
   arm a gate that refuses everything reads as a pass. *)

open Base
open Ll_test
open Verdict.Claims
module Sched = Ir.Schedule
module SO = Ir.Schedule_outcome
module BI = Ir.Backend_intf

let mk = node_factory ~first_id:9200 ~dims:[| 128; 2; 2 |] ()

(* [Ll_test.loop] builds a [Serial] loop; hardware annotation is what this test is about. *)
let wg_loop ~upto s body : LL.t =
  LL.For_loop { index = s; from_ = 0; to_ = upto; body; axis = LL.Workgroup }

let grid_loop ~upto s body : LL.t =
  LL.For_loop { index = s; from_ = 0; to_ = upto; body; axis = LL.Grid }

(* Threads 2 x 2 x 128: the innermost annotated loop binds [.x], so the extents read [.z], [.y],
   [.x] from the outside in. The value written varies with every symbol (gh-ocannl-589's
   discriminating-producer rule) so the nest is a real kernel and not a constant the optimizer can
   collapse -- a collapsed nest would lose the very loops whose annotation is under test. *)
let block_nest () =
  let o = mk "ldg_out" in
  materialize o;
  let sz = sym () and sy = sym () and sx = sym () in
  let llc =
    wg_loop ~upto:127 sz
      (wg_loop ~upto:1 sy
         (wg_loop ~upto:1 sx
            (set o
               [| iter sz; iter sy; iter sx |]
               (add (tick sz) (add (mul (c 10.) (embed sy)) (mul (c 100.) (embed sx)))))))
  in
  optimize ~materialized:[ o ] ~name:"ldg_block" llc

let cuda_like ~z =
  {
    BI.no_hardware_limits with
    max_threads_per_workgroup = Some 1024;
    max_workgroup_dims = Some (1024, 1024, z);
  }

let accepts ~limits opt =
  match
    Sched.check_hardware_limits_classified ~capabilities:Ir.Backend_intf.no_codegen_capabilities
      ~name:"ldg" ~limits opt
  with
  | () -> true
  | exception _ -> false

(* The typed cause, or [None] when the schedule was accepted. Named rather than matched inline so a
   claim can say WHICH dimension asked too much: a gate that refuses under the wrong resource would
   group an autotune search's declines under the wrong key. *)
let refusal ~limits opt =
  match
    Sched.check_hardware_limits_classified ~capabilities:Ir.Backend_intf.no_codegen_capabilities
      ~name:"ldg" ~limits opt
  with
  | () -> None
  | exception SO.Cause_at (_, SO.Resource_exceeded { resource; requested; limit; _ }) ->
      Some (resource, requested, limit)
  | exception _ -> None

let () =
  let opt = block_nest () in
  let dims = LL.launch_dims opt.LL.llc in
  let block = dims.LL.block in
  let product = Array.fold block ~init:1 ~f:( * ) in

  (* The premise. Without it every claim below could hold vacuously on a nest whose annotations the
     optimizer dropped. *)
  p "geometry: the hand-built nest is a 2 x 2 x 128 workgroup"
    (Array.equal Int.equal block [| 2; 2; 128 |]);
  p "geometry: its thread product is 512, well under a 1024-thread device" (product = 512);

  (* The gap itself: on the product alone -- the whole of the pre-679 gate -- this launch is legal.
     This claim is what makes the refusal below attributable to the per-dimension cap and not to
     some other check tightening. *)
  p "product-only limits accept the 2 x 2 x 128 workgroup (the pre-679 gate)"
    (accepts opt ~limits:{ BI.no_hardware_limits with max_threads_per_workgroup = Some 1024 });

  (* The fix, against a mocked CUDA-shaped record. *)
  p "a .z cap of 64 refuses it as a typed Workgroup_z_extent Resource_exceeded"
    (match refusal opt ~limits:(cuda_like ~z:64) with
    | Some (SO.Workgroup_z_extent, 128, Some 64) -> true
    | _ -> false);
  p "a .z cap at the extent accepts it" (accepts opt ~limits:(cuda_like ~z:128));
  p "a .z cap one below the extent refuses it"
    (match refusal opt ~limits:(cuda_like ~z:127) with
    | Some (SO.Workgroup_z_extent, 128, Some 127) -> true
    | _ -> false);

  (* Each dimension is gated by its own entry and reported as its own resource: a shared variant
     would erase which knob has to shrink. Caps of 1 on [.x] / [.y] leave [.z] slack, so the row
     that fires is the row under test. *)
  let dim_only caps =
    {
      BI.no_hardware_limits with
      max_threads_per_workgroup = Some 1024;
      max_workgroup_dims = Some caps;
    }
  in
  p "an .x cap below the .x extent refuses as Workgroup_x_extent"
    (match refusal opt ~limits:(dim_only (1, 1024, 1024)) with
    | Some (SO.Workgroup_x_extent, 2, Some 1) -> true
    | _ -> false);
  p "a .y cap below the .y extent refuses as Workgroup_y_extent"
    (match refusal opt ~limits:(dim_only (1024, 1, 1024)) with
    | Some (SO.Workgroup_y_extent, 2, Some 1) -> true
    | _ -> false);
  p "caps at every extent accept the whole geometry"
    (accepts opt
       ~limits:
         {
           BI.no_hardware_limits with
           max_threads_per_workgroup = Some 1024;
           max_workgroup_dims = Some (2, 2, 128);
         });

  (* [max_workgroup_dims = None] is what the C backends report, and it must exempt the dimensions
     rather than reject on a missing cap. *)
  p "absent per-dimension caps gate nothing"
    (accepts opt ~limits:{ BI.no_hardware_limits with max_threads_per_workgroup = Some 512 });

  (* The user-facing half: the raising variant names the dimension, which is the whole difference
     between this and a driver error. *)
  p "the raising variant names the .z workgroup extent"
    (match
       Sched.check_hardware_limits ~capabilities:Ir.Backend_intf.no_codegen_capabilities ~name:"ldg"
         ~limits:(cuda_like ~z:64) opt
     with
    | () -> false
    | exception Utils.User_error msg ->
        String.is_substring msg ~substring:".z workgroup extent"
        && String.is_substring msg ~substring:"128"
    | exception _ -> false);

  (* The grid rows still gate after the table rewrite -- both of them, so what #397 established is
     pinned by this file too and not only by [schedule_batch_grid.ml]'s matmul seeds. Slot
     assignment is positional from the inside out ([.x] innermost), so the 300 goes on the OUTER of
     two Grid loops to land on [.y]. *)
  let grid_opt =
    let o = mk "ldg_grid" ~dims:[| 300; 4 |] in
    materialize o;
    let gy = sym () and gx = sym () in
    optimize ~materialized:[ o ] ~name:"ldg_grid"
      (grid_loop ~upto:299 gy (grid_loop ~upto:3 gx (set o [| iter gy; iter gx |] (tick gy))))
  in
  let gdims = LL.launch_dims grid_opt.LL.llc in
  p "geometry: the grid nest is .y = 300 over .x = 4"
    (gdims.LL.grid.(1) = 300 && gdims.LL.grid.(0) = 4 && gdims.LL.grid.(2) = 1);
  p "a max_grid_yz of 300 accepts it"
    (accepts grid_opt ~limits:{ BI.no_hardware_limits with max_grid_yz = Some 300 });
  p "a max_grid_yz of 299 refuses it as Grid_y_extent"
    (match refusal grid_opt ~limits:{ BI.no_hardware_limits with max_grid_yz = Some 299 } with
    | Some (SO.Grid_y_extent, 300, Some 299) -> true
    | _ -> false);
  (* The fold: three Grid loops, so the outermost sits at slot 2 and its extent becomes [grid.(2)].
     [.y] is 2 here, below the same cap, so only the [.z] row can fire and the typed resource is not
     an artifact of check ordering. *)
  let fold_opt =
    let o = mk "ldg_fold" ~dims:[| 4; 2; 8 |] in
    materialize o;
    let gz = sym () and gy = sym () and gx = sym () in
    optimize ~materialized:[ o ] ~name:"ldg_fold"
      (grid_loop ~upto:3 gz
         (grid_loop ~upto:1 gy
            (grid_loop ~upto:7 gx (set o [| iter gz; iter gy; iter gx |] (tick gz)))))
  in
  let fdims = LL.launch_dims fold_opt.LL.llc in
  p "geometry: the fold nest folds 4 onto .z with a .y of 2"
    (fdims.LL.grid.(2) = 4 && fdims.LL.grid.(1) = 2);
  p "a max_grid_yz of 4 accepts the fold"
    (accepts fold_opt ~limits:{ BI.no_hardware_limits with max_grid_yz = Some 4 });
  p "a max_grid_yz of 3 refuses the fold as Grid_z_extent"
    (match refusal fold_opt ~limits:{ BI.no_hardware_limits with max_grid_yz = Some 3 } with
    | Some (SO.Grid_z_extent, 4, Some 3) -> true
    | _ -> false);

  (* The annotator side (gh-ocannl-679's "the gate should be a backstop, not the first line"):
     [Schedule.default_gpu] clamps its block size against the per-dimension [.x] cap as well as the
     thread product. It emits one [Workgroup] loop per nest, so [.x] is the only entry it can
     bind. *)
  let flat_opt =
    let o = mk "ldg_flat" ~dims:[| 4096 |] in
    materialize o;
    let s = sym () in
    optimize ~materialized:[ o ] ~name:"ldg_flat" (loop ~upto:4095 s (set o [| iter s |] (tick s)))
  in
  let annotated ~limits =
    let o = Sched.apply (Sched.default_gpu ~block_size:256 ~limits flat_opt) flat_opt in
    (LL.launch_dims o.LL.llc).LL.block.(0)
  in
  p "annotator: with no per-dimension cap the block keeps the requested 256 threads"
    (annotated ~limits:{ BI.no_hardware_limits with max_threads_per_workgroup = Some 1024 } = 256);
  p "annotator: an .x cap of 8 clamps the block to 8, before the gate ever sees it"
    (annotated
       ~limits:
         {
           BI.no_hardware_limits with
           max_threads_per_workgroup = Some 1024;
           max_workgroup_dims = Some (8, 1024, 1024);
         }
    = 8)
