(* gh-ocannl-1612: every seed the CPU family trees offer is a schedule the cc compile path accepts.

   The tuner times a seed only after its schedule applies and the result validates; a seed refused
   there at every compile is a candidate compile spent on a decline, and noise in the decline
   census. Two classes have been found this way: gh-ocannl-1561's hardware-binding flavors beside a
   companion (since gh-ocannl-1609 valid on the fork-join model, and excluded as search economy),
   and this issue's fused k-blocked packed twins. The packed shapes that sink the row-block loop
   under the reduction-block loops (Serial, Hoisted and the plain Grid shape) write the accumulator
   once per reduction block, so [Fuse_epilogue]'s exactly-once rule refuses the fused twin wherever
   no single store-back follows the whole reduction -- [mma-cpu 16x0x16 ep] and its [grid] twin, at
   every compile. The tree now refutes those geometries in the fused flavor
   ([Autotune.cpu_pack_fused_partial_witness]).

   Seeding uses synthetic limits (a 32-byte vector file), so the offered set is machine-independent.
   Validation is the cc compile path's: [validate_parallel ~fork_join:true] with the online-softmax
   reassociation, as [C_syntax.compile_proc] runs it on a backend binding no hardware [Grid] index.

   1. Seed validity, over representative routines: a matmul with a bias + relu tail (k = 32, two k
   blocks of the 16x0x16 geometry), a 3x3 convolution with a bias tail, and tune_footprint_default's
   routine (an 8x32 by 32x8 product read through its diagonal, materialize-all). Every seed the
   production trees offer, every seed [Autotune.lift_cpu_companions] recovers, and every coalesced
   seed applies and validates; on the tail routine fused tensorized twins are offered, and none is
   refused by [Fuse_epilogue] (the issue's census, red before the refutation).

   2. The refutation against the op, two-sided: on k-split sites (one full k block and an edge,
   batched and not; four full blocks and an edge, batched and not; k of one block and below one; a
   row edge; a column-blocked site; dividing k = 32), every unfused packed seed of the lifted tree
   whose own schedule applies and validates is flipped to its fused twin, and [Fuse_epilogue]
   refuses that twin exactly where the witness says so. Each of the witness's clauses fires
   somewhere, an admitted twin of a concerned shape exists (gh-ocannl-1733's last segment), and no
   witnessed twin is offered by the fused flavor.

   Values cannot change: the refutation only removes candidates that never compiled. The admitted
   fused twins' values are epilogue_fusion_single_trip_sweep's (bitwise) and
   epilogue_fusion_mma_seeds'. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Sspace = Ir.Schedule_space
open Verdict.Claims

let accum_prec = Ll_test.backend_accum_prec

let cpu_limits =
  {
    Ir.Backend_intf.no_hardware_limits with
    simd_vector = { Ir.Vector_file.bytes = 32; registers = 16 };
  }

let scratch (o : LL.optimized) =
  {
    o with
    LL.traced_store = Hashtbl.copy o.LL.traced_store;
    LL.optimize_ctx = LL.copy_optimize_ctx o.LL.optimize_ctx;
  }

let label sp = Autotune.sketch_params_label sp
let seeding opt = Autotune.family_seeding ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt

let tree_of opt =
  match fst (seeding opt) with
  | Autotune.Seeding_matmul { tree; _ } | Autotune.Seeding_conv { tree; _ } -> Some tree
  | Autotune.Seeding_none -> None

let matmul_site opt =
  match fst (seeding opt) with
  | Autotune.Seeding_matmul { site; _ } -> Some site
  | Autotune.Seeding_conv _ | Autotune.Seeding_none -> None

let dedup sps =
  List.rev
    (List.fold sps ~init:[] ~f:(fun acc sp ->
         if List.exists acc ~f:(fun q -> String.equal (label q) (label sp)) then acc else sp :: acc))

(* Every seed a CPU search can reach without lifting legality: the production seeding, the companion
   lift, and the coalesced branch. *)
let population opt =
  let lifted =
    match tree_of opt with
    | None -> []
    | Some tree -> Sspace.leaves (Autotune.lift_cpu_companions tree)
  in
  dedup
    (snd (seeding opt)
    @ lifted
    @ Autotune.coalesced_seed_params ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt)

type verdict = Valid | Fuse_refused of string | Other_refused of string

(* The cc compile path's gate: the schedule applies, and the result passes fork-join validation. *)
let judge opt sp =
  match
    let s = scratch opt in
    let post = Sched.apply (Autotune.sketch_schedule ~accum_prec ~p:sp s) s in
    LL.validate_parallel ~reassociable:Ir.Online_softmax.reassociable_local ~fork_join:true
      post.LL.optimize_ctx.placements post.LL.llc
  with
  | () -> Valid
  | exception Invalid_argument detail when String.is_prefix detail ~prefix:"Schedule.Fuse_epilogue"
    ->
      Fuse_refused detail
  | exception exn -> Other_refused (Exn.to_string exn)

let one_line msg = String.prefix (String.tr ~target:'\n' ~replacement:' ' msg) 300

let report ~tag sp = function
  | Valid -> ()
  | Fuse_refused m | Other_refused m ->
      Stdio.eprintf "%s: %s refused: %s (not part of the golden)\n%!" tag (label sp) (one_line m)

let ints label ~rows ~cols modulus =
  TDSL.ndarray
    (Array.init (rows * cols)
       ~f:(Ll_test.cycle_flat ~dims:[| rows; cols |] ~modulus ~offset:1. ~stride:1.))
    ~label:[ label ] ~output_dims:[ rows; cols ] ()

let lower ?(materialized = []) comp =
  Context.lowered_for_decisions ~materialized (Context.cpu ()) comp Ir.Indexing.Empty

(* Part 1: one routine's seed population, judged. [fused_premise] requires fused tensorized twins in
   the population (the routine has a fusable tail). *)
let validity_leg ?(fused_premise = false) tag opt =
  let offered = population opt in
  let judged = List.map offered ~f:(fun sp -> (sp, judge opt sp)) in
  List.iter judged ~f:(fun (sp, v) -> report ~tag sp v);
  Stdio.eprintf "%s: %d seeds judged (not part of the golden)\n%!" tag (List.length offered);
  p_exists (tag ^ ": premise: the CPU trees offer seeds") offered ~f:(fun _ -> true);
  if fused_premise then
    p_exists (tag ^ ": premise: fused tensorized twins are offered") offered ~f:(fun sp ->
        sp.sk_mma && sp.sk_epilogue);
  p_none (tag ^ ": no offered fused twin is refused by Fuse_epilogue") judged ~f:(fun (_, v) ->
      match v with Fuse_refused _ -> true | Valid | Other_refused _ -> false);
  p_all (tag ^ ": every offered seed applies and passes fork-join validation") judged
    ~f:(fun (_, v) -> match v with Valid -> true | Fuse_refused _ | Other_refused _ -> false)

let () =
  let tag = "tail" in
  let x = ints "tail_x" ~rows:64 ~cols:32 7 in
  let y = ints "tail_y" ~rows:32 ~cols:64 5 in
  let bias =
    TDSL.ndarray
      (Array.init 64 ~f:(fun i -> Float.of_int (i % 5) -. 2.))
      ~label:[ "tail_bias" ] ~output_dims:[ 64 ] ()
  in
  let%op prod = x +* "ik; kj => ij" y in
  let%op out = relu (prod + bias) in
  validity_leg ~fused_premise:true tag
    (lower ~materialized:[ prod.Tensor.value ] (Train.forward out))

let () =
  let tag = "conv3" in
  let x =
    NTDSL.init ~l:"conv3_x" ~prec:Ir.Ops.single ~b:[ 2 ] ~o:[ 11; 11; 4 ]
      ~f:(Ll_test.weighted ~weights:[| 1; 1; 2; 3 |] ~modulus:7 ~offset:0. ~stride:1.)
      ()
  in
  let kern =
    NTDSL.init ~l:"conv3_k" ~prec:Ir.Ops.single ~i:[ 3; 3; 4 ] ~o:[ 8 ]
      ~f:(Ll_test.weighted ~weights:[| 2; 1; 1; 3 |] ~modulus:5 ~offset:(-2.) ~stride:1.)
      ()
  in
  let bias =
    NTDSL.init ~l:"conv3_b" ~prec:Ir.Ops.single ~o:[ 8 ]
      ~f:(fun idcs -> Float.of_int (idcs.(0) % 3) -. 1.)
      ()
  in
  let%op z =
    x +* "...| 1*oh<+kh, 1*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." kern
  in
  let%op out = z + bias in
  let opt = lower ~materialized:[ z.Tensor.value ] (Train.forward out) in
  p
    (tag ^ ": premise: the conv family claims the site")
    (match fst (seeding opt) with
    | Autotune.Seeding_conv _ -> true
    | Autotune.Seeding_matmul _ | Autotune.Seeding_none -> false);
  validity_leg tag opt

(* tune_footprint_default's routine (footprint_diagonal_einsum's): an 8x32 by 32x8 product read only
   through its diagonal, in that test's materialize-all arm, where gh-ocannl-1561's whole-triple
   form was offered beside the diagonal companion. (Its default placement footprint-scopes the
   product, and no family claims a site there.) *)
let () =
  let tag = "diag" in
  let x = ints "diag_x" ~rows:8 ~cols:32 7 in
  let y = ints "diag_y" ~rows:32 ~cols:8 5 in
  let%op a = x +* "ik; kj => ij" y in
  let%op d = a ++ "ii => i" in
  validity_leg tag (lower ~materialized:[ a.Tensor.value ] (Train.forward d))

(* Part 2: the witness against the op. A GEMM [z = x . w] over [b] batch rows ([b = 0]: unbatched)
   with the elementwise tail [z + pos]. *)
let gemm ?(b = 0) ~m ~k ~n tag =
  let rows = Int.max b 1 * m in
  let xv =
    Array.init (rows * k)
      ~f:(Ll_test.cycle_flat ~dims:[| rows; k |] ~modulus:13 ~offset:0. ~stride:0.25)
  in
  let wv =
    Array.init (k * n) ~f:(Ll_test.cycle_flat ~dims:[| k; n |] ~modulus:17 ~offset:(-8.) ~stride:1.)
  in
  let pv = Array.init n ~f:(fun j -> Float.of_int (j % 5) -. 2.) in
  let x =
    if b = 0 then TDSL.ndarray xv ~label:[ tag ^ "_x" ] ~output_dims:[ m; k ] ()
    else TDSL.ndarray xv ~label:[ tag ^ "_x" ] ~batch_dims:[ b ] ~output_dims:[ m; k ] ()
  in
  let w = TDSL.ndarray wv ~label:[ tag ^ "_w" ] ~output_dims:[ k; n ] () in
  let pos = TDSL.ndarray pv ~label:[ tag ^ "_pos" ] ~output_dims:[ n ] () in
  let z =
    if b = 0 then
      let%op z = x +* "ik;kj=>ij" w in
      z
    else
      let%op z = x +* "b|ik;kj=>b|ij" w in
      z
  in
  let%op y = z + pos in
  lower ~materialized:[ z.Tensor.value ] (Train.forward y)

type case = {
  c_tag : string;
  c_twin : Autotune.sketch_params;
  c_witness : string option;
  c_verdict : verdict;
  c_offered : bool;  (** The fused flavor of the lifted tree offers the twin. *)
}

let cases = ref []

let relationship_site tag opt =
  match (matmul_site opt, tree_of opt) with
  | Some site, Some tree ->
      let lifted = Sspace.leaves (Autotune.lift_pad_waste (Autotune.lift_cpu_companions tree)) in
      let unfused_packed =
        List.filter lifted ~f:(fun sp -> sp.sk_mma && sp.sk_bk > 0 && not sp.sk_epilogue)
      in
      let applicable = List.filter unfused_packed ~f:(fun sp -> Poly.equal (judge opt sp) Valid) in
      p_exists (tag ^ ": premise: unfused packed seeds apply") applicable ~f:(fun _ -> true);
      let fused_offered = List.filter lifted ~f:(fun sp -> sp.sk_epilogue) in
      let twins = dedup (List.map applicable ~f:(fun sp -> { sp with sk_epilogue = true })) in
      List.iter twins ~f:(fun twin ->
          let c_verdict = judge opt twin in
          (match (Autotune.cpu_pack_fused_partial_witness site twin, c_verdict) with
          | Some _, Fuse_refused _ | None, Valid -> ()
          | _ -> report ~tag twin c_verdict);
          cases :=
            {
              c_tag = tag;
              c_twin = twin;
              c_witness = Autotune.cpu_pack_fused_partial_witness site twin;
              c_verdict;
              c_offered =
                List.exists fused_offered ~f:(fun q -> String.equal (label q) (label twin));
            }
            :: !cases)
  | _ -> p (tag ^ ": premise: the matmul family claims the site") false

let () =
  relationship_site "k32" (gemm ~m:32 ~k:32 ~n:32 "k32");
  relationship_site "ksplit1" (gemm ~m:16 ~k:20 ~n:32 "ksplit1");
  relationship_site "ksplit1b" (gemm ~b:4 ~m:15 ~k:28 ~n:16 "ksplit1b");
  relationship_site "ksplit4" (gemm ~m:24 ~k:70 ~n:40 "ksplit4");
  relationship_site "ksplit4b" (gemm ~b:2 ~m:32 ~k:70 ~n:32 "ksplit4b");
  relationship_site "kone" (gemm ~m:24 ~k:16 ~n:32 "kone");
  relationship_site "ksub" (gemm ~m:24 ~k:12 ~n:32 "ksub");
  relationship_site "rowedge" (gemm ~m:20 ~k:20 ~n:32 "rowedge");
  relationship_site "colblocks" (gemm ~m:128 ~k:140 ~n:288 "colblocks");
  let all = List.rev !cases in
  let refused c =
    match c.c_verdict with Fuse_refused _ -> true | Valid | Other_refused _ -> false
  in
  let witnessed ~phrase c =
    Option.exists c.c_witness ~f:(fun w -> String.is_substring w ~substring:phrase)
  in
  p_all "relationship: Fuse_epilogue refuses a fused packed twin exactly where the witness says so"
    all ~f:(fun c ->
      match (c.c_witness, c.c_verdict) with
      | Some _, Fuse_refused _ | None, Valid -> true
      | Some _, (Valid | Other_refused _) | None, (Fuse_refused _ | Other_refused _) -> false);
  p_none "relationship: no witnessed twin is offered by the fused flavor" all ~f:(fun c ->
      Option.is_some c.c_witness && c.c_offered);
  p_exists "relationship: the repeating reduction-block clause fires" all
    ~f:(witnessed ~phrase:"once per reduction block");
  p_exists "relationship: the clause for a loop enclosing both k segments fires" all
    ~f:(witnessed ~phrase:"enclose both segments");
  p_exists "relationship: the clause fires on a column-block loop alone" all ~f:(fun c ->
      String.equal c.c_tag "colblocks" && witnessed ~phrase:"the column-block loop" c);
  p_exists "relationship: the row-edge clause fires" all
    ~f:(witnessed ~phrase:"row-block loop's edge");
  p_exists "relationship: a partitioned twin of a row-sinking shape is admitted" all ~f:(fun c ->
      (not
         (Autotune.cpu_pack_grid_outermost ~sk_grid:c.c_twin.sk_grid ~sk_hoist:c.c_twin.sk_hoist
            ~sk_pack_rest:c.c_twin.sk_pack_rest))
      && (not (refused c))
      && List.mem [ "ksplit1"; "ksplit4" ] c.c_tag ~equal:String.equal);
  Stdio.eprintf "relationship: %d twins, %d refused (not part of the golden)\n%!" (List.length all)
    (List.count all ~f:refused)

let () = Stdio.printf "cpu_seed_validity: done\n"
