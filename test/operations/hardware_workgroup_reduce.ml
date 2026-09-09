(* Axis-type annotations, Phase C (docs/proposals/axis-types-for-loops.md §3): an executed
   shared-memory workgroup reduction (tinygrad's GROUP_REDUCE pattern) against the serial reduction.

   The surrogate computation [s = sum v] (64 elements) is compiled twice: once as lowered
   (all-Serial), and once with its lowered body REPLACED via [?lowered_transform] by a hand-built
   kernel: cooperative load of [v] into a workgroup-shared tile, barrier, log2(64) tree-combine
   stages ([If (i < stride)] guarded, barrier after each), and an [If (i == 0)] write of
   [partial[0]] into [s]. The tile is a fresh [Local]-mode node registered in the traced store and
   in [optimized.workgroup_shared], so it is declared with the backend's shared prefix
   ([threadgroup] / [__shared__]) instead of a per-thread stack array.

   On GPU backends (Metal locally, CUDA in CI) the kernel runs with a 64-thread workgroup and its
   result must match the serial sum. The C backends cannot implement barriers or shared placement by
   serialization; they must reject the kernel with a clear [Invalid_argument] -- that clean
   rejection is what this test pins there, so every printed boolean holds on every backend. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Asgns = Ir.Assignments
module L = Ll_test

let () = Utils.settings.output_debug_files_in_build_directory <- true

open Verdict.Claims

let n = 64
let vv = Array.init n ~f:(fun i -> (Float.of_int i *. 0.5) -. 7.)
let expected_sum = Array.fold vv ~init:0. ~f:( +. )
let approx a b = Float.(abs (a - b) < 1e-3)
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let skipped = Verdict.skipped ~backend:backend_name
let has_barriers = Ir.Schedule.backend_is_gpu backend_name

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name
let single = Ir.Ops.single

(* Replace the lowered serial sum with the hand-built workgroup tree reduction. *)
let group_reduce ~(v : Tn.t) ~(s : Tn.t) (opt : LL.optimized) : LL.optimized =
  let partial =
    Tn.create (Tn.Specified single) ~id:999001 ~label:[ "partial_sums" ]
      ~unpadded_dims:(lazy [| n |])
      ~padding:(lazy None)
      ()
  in
  Tn.update_memory_mode partial Tn.Local 991;
  (* Register the tile in the traced store so [compile_proc]'s local-declaration pass sees it. *)
  ignore (LL.get_node opt.traced_store partial : LL.traced_array);
  let it = L.iter in
  let f0 = L.fixed 0 in
  let wg body_f : LL.t =
    let i = L.sym () in
    L.loop_n ~axis:LL.Workgroup_reduce i n (body_f i)
  in
  let load = wg (fun i -> L.set_at partial (it i) (L.get v [| it i |])) in
  let stage stride =
    wg (fun i ->
        L.if_idx
          (L.lt (L.embed i) (L.ic stride))
          (L.set_at partial (it i)
             (L.add (L.get partial [| it i |]) (L.get partial [| L.aff [ (1, i) ] stride |]))))
  in
  let write_out =
    wg (fun i -> L.if_idx (L.eq (L.embed i) (L.ic 0)) (L.set_at s f0 (L.get partial [| f0 |])))
  in
  let strides = [ 32; 16; 8; 4; 2; 1 ] in
  let stmts =
    load :: LL.Workgroup_barrier
    :: List.concat_map strides ~f:(fun st -> [ stage st; LL.Workgroup_barrier ])
    @ [ write_out ]
  in
  { opt with llc = LL.unflat_lines stmts; workgroup_shared = Set.add opt.workgroup_shared partial }

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let () =
  let v = TDSL.ndarray vv ~label:[ "v" ] ~output_dims:[ n ] () in

  (* --- Serial twin --- *)
  let%op s0 = v ++ "i=>0" in
  let serial_comp = named "wgred_sum_serial" (Train.forward s0) in
  let ctx_s = Context.auto () in
  let ctx_s, routine_s = Context.compile ctx_s serial_comp Ir.Indexing.Empty in
  let ctx_s = Context.run ctx_s routine_s in
  let got_s0 = Context.get_values ctx_s s0.Tensor.value in
  p "serial reduction correct" (approx got_s0.(0) expected_sum);

  (* --- Workgroup-shared tree reduction --- *)
  let%op s1 = v ++ "i=>0" in
  let annot_comp = named "sum_wg_reduce" (Train.forward s1) in
  let transform = group_reduce ~v:v.Tensor.value ~s:s1.Tensor.value in
  let ctx_a = Context.auto () in
  if has_barriers then (
    let ctx_a, routine_a =
      Context.compile
        ~lowered_transform:(fun o -> [ transform o ])
        ctx_a annot_comp Ir.Indexing.Empty
    in
    let ctx_a = Context.run ctx_a routine_a in
    let got_s1 = Context.get_values ctx_a s1.Tensor.value in
    p "workgroup reduction parity (GPU) or clean rejection (CPU)" (approx got_s1.(0) expected_sum);
    let src = Generated.read "sum_wg_reduce" in
    let has sub = String.is_substring src ~substring:sub in
    (* This is intentionally dialect identity: the claim pins each language's shared-address-space
       qualifier and workgroup barrier token, not whether the reduction was selected. *)
    let shared_ok =
      if String.is_substring backend_name ~substring:"metal" then
        has "threadgroup float partial_sums[64]" && has "threadgroup_barrier"
      else has "__shared__ float partial_sums[64]" && has "__syncthreads()"
    in
    p "shared tile and barrier rendered (GPU) or rejected (CPU)" shared_ok)
  else (
    (match
       try
         ignore
           (Context.compile
              ~lowered_transform:(fun o -> [ transform o ])
              ctx_a annot_comp Ir.Indexing.Empty
             : Context.t * Context.routine);
         None
       with Invalid_argument msg -> Some msg
     with
    | Some msg ->
        p "workgroup reduction parity (GPU) or clean rejection (CPU)"
          (String.is_substring msg ~substring:"not supported")
    | None -> p "workgroup reduction parity (GPU) or clean rejection (CPU)" false);
    skipped "shared tile and barrier rendered (GPU) or rejected (CPU)")
