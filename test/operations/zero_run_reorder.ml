(* gh-ocannl-1250: kernel fission sorts each maximal run of whole-node zeros by the zeroed node's
   shape ([Schedule.collect_units], gh-ocannl-1169). The reorder is sound because the zeros of a run
   write constant zeros, read nothing, and have nothing between them, so they commute. This test
   checks the reorder against an oracle that does not share it.

   [gpu_fission_mapping] case 4 compares two segmentations that both go through the sort, and starts
   from fresh buffers that are already zero, so it can catch neither an error both sides share nor a
   zero that never runs. Here:

   - every node the routine writes is PREFILLED with nonzero values that differ, cell for cell, from
   what the routine leaves there, so a zero (or any write) that does not run leaves a value the
   check sees; - the reference compiles the same optimized code with no transform: one kernel in the
   original statement order, which never meets the sort. A closed form pins the reference itself; -
   the code holds repeated targets (two zeros of [za] in the first run, of [zf] in the second) and
   overlapping storage: [zp] is zeroed whole in the first run and then written through a slice ([zp]
   at fixed row 1), as a slice-alias view's write lowers ([Assignments.resolve_alias] rewrites a
   view to its parent at a fixed leading index, and a view's zero lowers to a [Set] nest, not a
   [Zero_out], gh-ocannl-1475). The runs are cut by that slice write and by the accumulations, which
   neither commute with the zeros nor may be sorted with them.

   Each fission arm (the GPU pipeline schedule-aware, legality-only and with folded zeros, and the
   CPU pipeline) must reorder the runs (the case is not vacuous) and produce the reference's values.
   The folding arm must also fold [zf]'s last zero into the reduction after it: the fold pairs each
   zero with the unit that follows it AFTER the sort, and here the sort is what makes them adjacent.

   Mutants this test kills (gh-ocannl-1250's acceptance): dropping one zero from a sorted run, and
   letting a run's sort cross an overlapping non-zero write. Swapping two zeros of a run is not
   observable in values -- they commute, which is the reorder's own soundness argument -- so no
   executed check can fail on it, and none tries to. *)

open Base
open Stdio
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module L = Ll_test
module LL = Ir.Low_level
module S = Ir.Schedule
module Tn = Ir.Tnode

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

(* The GPU pipeline is built (and executed, its hardware loops rendered serially) on a C backend
   under a GPU backend's name, as in [gpu_fission_mapping]. *)
let gpu_name = if S.backend_is_gpu backend_name then backend_name else "metal"
let limits = { Ir.Backend_intf.no_hardware_limits with Ir.Backend_intf.simdgroup_width = Some 32 }
let node = L.node_factory ~first_id:1250000 ~dims:[| 1 |] ()

(* Ranks 1, 2 and 3: the shape sort moves every run. *)
let za = node ~dims:[| 4; 8 |] "zrr_a"
let zb = node ~dims:[| 16 |] "zrr_b"
let zc = node ~dims:[| 2; 4; 8 |] "zrr_c"
let zd = node ~dims:[| 4; 8 |] "zrr_d"
let ze = node ~dims:[| 16 |] "zrr_e"
let zf = node ~dims:[| 2; 4; 8 |] "zrr_f"
let zp = node ~dims:[| 3; 16 |] "zrr_p"
let nodes = [ za; zb; zc; zd; ze; zf; zp ]
let () = List.iter nodes ~f:L.materialize
let dims tn = Lazy.force tn.Tn.dims
let numel tn = Array.fold (dims tn) ~init:1 ~f:( * )

(* What the routine leaves in each node, by flat index: zero, except the slice write's row of [zp]
   and the two accumulations' targets. *)
let expected tn =
  Array.init (numel tn) ~f:(fun k ->
      let ix = L.unflat ~dims:(dims tn) k in
      if Tn.equal tn zp && ix.(0) = 1 then Float.of_int (1 + ix.(1))
      else if Tn.equal tn zd then Float.of_int (1 + (10 * ix.(0)) + ix.(1))
      else if Tn.equal tn zf then
        Float.of_int ((3 * (1 + (100 * ix.(0)) + (10 * ix.(1)) + ix.(2))) + 3)
      else 0.)

(* The prefill: nonzero, distinct per node, and off every value the routine writes. *)
let prefill =
  List.mapi nodes ~f:(fun n tn ->
      (tn, Array.init (numel tn) ~f:(fun k -> Float.of_int ((1000 * (n + 1)) + 1 + k))))

(* The original order: two zero runs, cut by a slice write into [zp] (overlapping its zero) and
   followed by accumulations into [zf] and [zd]. *)
let program () =
  let i = L.sym () and j = L.sym () and s = L.sym () in
  let run1 = List.map [ zc; za; zb; za; zd; zp ] ~f:L.zero in
  let slice = L.loop_n s 16 (L.set zp [| L.fixed 1; L.iter s |] (L.tick s)) in
  let run2 = List.map [ zf; ze; zf ] ~f:L.zero in
  let acc_f =
    let a' = L.sym () and i' = L.sym () and j' = L.sym () and r = L.sym () in
    let idx = [| L.iter a'; L.iter i'; L.iter j' |] in
    L.loop_n a' 2
      (L.loop_n i' 4
         (L.loop_n j' 8
            (L.loop_n r 3
               (L.set zf idx
                  (L.add (L.get zf idx)
                     (L.add (L.mul (L.c 100.) (L.embed a')) (L.add (L.tag i' j') (L.embed r))))))))
  in
  let acc_d =
    let idx = [| L.iter i; L.iter j |] in
    L.loop_n i 4 (L.loop_n j 8 (L.set zd idx (L.add (L.get zd idx) (L.tag i j))))
  in
  LL.unflat_lines (run1 @ [ slice ] @ run2 @ [ acc_f; acc_d ])

(* The whole-node zeros of a statement list, in order. *)
let zero_order stmts = List.filter_map stmts ~f:(function LL.Zero_out tn -> Some tn | _ -> None)
let original_order = zero_order (LL.flat_lines [ program () ])

(* Each arm optimizes the code afresh (fission promotes placements of the record it is given),
   compiles it with [transform], prefills every node, runs once and reads every node back. *)
let execute ~name ~transform =
  let o = L.optimize ~name (program ()) in
  let ctx, routine =
    Context.compile ~name ~prelowered:o ~lowered_transform:transform (Lazy.force L.base_ctx)
      Ir.Assignments.empty_comp Ir.Indexing.Empty
  in
  let ctx = L.run_linked (ctx, routine) ~seed:prefill in
  (o, List.map nodes ~f:(Context.get_values ctx))

let check_values ~what got =
  List.iter2_exn nodes got ~f:(fun tn values ->
      p_all2
        (Printf.sprintf "%s: %s holds the reference's values" what (Tn.debug_name tn))
        values (expected tn) ~f:Float.equal)

type arm = {
  arm : string;
  folds : bool;  (** The arm folds the reduction's zero, which the sort made adjacent to it. *)
  fission :
    LL.optimized -> ([ `Normal | `Zeros | `Solo ] * LL.optimized * S.schedule * LL.optimized) list;
}

let gpu_fission ?(fold_zeros = false) ~keep o =
  let keep_mapping = if keep then S.fission_keep_mapping ~is_gpu:true ~limits else None in
  S.fission_scheduled ~promote_locals:true ~fold_zeros ?keep_mapping ~preset:(S.default_gpu ~limits)
    ~zero_sched:(S.zero_expansion ~limits) ~static_indices:[] o

let arms =
  [
    { arm = "GPU pipeline, schedule-aware"; folds = false; fission = gpu_fission ~keep:true };
    { arm = "GPU pipeline, legality-only"; folds = false; fission = gpu_fission ~keep:false };
    {
      arm = "GPU pipeline, schedule-aware, folded zeros";
      folds = true;
      fission = gpu_fission ~fold_zeros:true ~keep:true;
    };
    {
      arm = "CPU pipeline";
      folds = false;
      fission =
        S.fission_scheduled ~preset:(S.default_cpu ~min_parallel:1)
          ~zero_sched:(fun _ -> [])
          ~static_indices:[];
    };
  ]

let () =
  eprintf "zero_run_reorder backend: %s, GPU pipeline of %s (not part of the golden)\n%!"
    backend_name gpu_name;
  p_all "the prefill differs from what the routine leaves in every cell" prefill
    ~f:(fun (tn, values) ->
      Array.for_alli values ~f:(fun k v -> not (Float.equal v (expected tn).(k))));
  printf "--- reference: no transform, the original statement order ---\n";
  let o, reference = execute ~name:"zrr_reference" ~transform:(fun o -> [ o ]) in
  p "reference: the optimized code keeps every zero, the repeated one included"
    (List.equal Tn.equal (zero_order (LL.flat_lines [ o.LL.llc ])) original_order);
  check_values ~what:"reference" reference;
  List.iteri arms ~f:(fun n { arm; folds; fission } ->
      printf "--- %s ---\n" arm;
      let segments = ref [] in
      let _, got =
        execute ~name:(Printf.sprintf "zrr_arm%d" n) ~transform:(fun o ->
            let segs = fission o in
            segments := segs;
            List.map segs ~f:(fun (_, _, _, post) -> post))
      in
      let order =
        List.concat_map !segments ~f:(fun (_, pre, _, _) ->
            zero_order (LL.flat_lines [ pre.LL.llc ]))
      in
      let kind = function `Normal -> "N" | `Zeros -> "Z" | `Solo -> "S" in
      eprintf "%s: segments [%s], zero order [%s] (not part of the golden)\n%!" arm
        (String.concat ~sep:" "
           (List.map !segments ~f:(fun (k, pre, _, _) ->
                Printf.sprintf "%s%d" (kind k) (List.length (LL.flat_lines [ pre.LL.llc ])))))
        (String.concat ~sep:" " (List.map order ~f:Tn.debug_name));
      p (arm ^ ": the zero runs are reordered") (not (List.equal Tn.equal order original_order));
      if folds then
        p
          (arm ^ ": a zero folds into the accumulation the sort moved it next to")
          (List.length order < List.length original_order);
      check_values ~what:arm got)
