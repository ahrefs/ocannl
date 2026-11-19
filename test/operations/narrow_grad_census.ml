(* gh-ocannl-2362: the census of a bf16 decoder block's backward, host-side: every narrow gradient
   nest of plain cells is sunk, and every privatized tile over a narrow gradient is f32.

   [gpt2_mini_train]'s bf16 backward on HIP shipped 61 bf16 [Privatize] tiles (the register tiles of
   the seeded GEMM sketches, gfx1151): backprop lowers a gradient over its forward op's product
   space, so the serial rendering of a weight, input, attention or positional-table gradient
   localized nothing, and the shared width decision kept each tile at storage to match. The
   optimizer now sinks those nests ([Low_level.sink_narrow_reduction_loops]). This pins it on a
   training step shaped like that workload's layer, gradients at bf16 ([Tensor.default_grad_prec]):
   a positional table added through the trained einsum ([... s | d; | s d => ... s | d], the [wpe]
   gradient [wpe.grad[s,d] += x.grad[b,s,d]]), then a decoder block (Q, K, V, scores, softmax, the
   value product, layer norms, the FFN).

   The claims, on the routine every backend lowers (no device fact enters): - the nests: some narrow
   plain-cell accumulation is reduction-outer as lowered ([opt.source], the population the next
   claim needs), every one opens its cell's loops first in the optimized routine ([opt.llc]), and
   the positional table's gradient is one of them, named; - the tiles: the privatized preset over
   the whole routine ([Autotune.extend_with_privatize], each proposal over a target's outermost
   serial reduction loop), under a HIP-like accumulator resolution (narrow floats widen to f32, as
   [bf16_arithmetic=auto] resolves on HIP, CUDA, cc and Metal), mints at least a few [acc] tiles
   over bf16 gradients, and every one is f32. The tiles are read off the applied program, not off
   the ops. *)

open Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Idx = Ir.Indexing
module Tn = Ir.Tnode

let bf16 = Ir.Ops.bfloat16

let seq = 4
and d_model = 16

(* The accumulation precision of a backend whose policy widens narrow floats. *)
let accum_prec p = if Ir.Ops.is_narrow_float p then Ir.Ops.single else p

let input ~label ~b ~o =
  NTDSL.init ~l:label ~prec:Ir.Ops.single ~b ~i:[] ~o
    ~f:(fun idx ->
      Float.of_int ((Array.fold idx ~init:7 ~f:(fun acc i -> (acc * 31) + i) % 13) - 6) /. 8.)
    ()

let saved_grad_prec = !Tensor.default_grad_prec
let () = Tensor.default_grad_prec := bf16

let loss, wpe =
  let x = input ~label:"ngc_x" ~b:[ 2; seq ] ~o:[ d_model ] in
  let wpe =
    Operation.init ~l:"ngc_wpe" ~prec:Ir.Ops.single ~o:[ seq; d_model ]
      ~f:(fun idx -> Float.of_int ((idx.(0) * 5) + idx.(1)) /. 64.)
      ~grad_spec:Tensor.Require_grad ()
  in
  let mask =
    NTDSL.init ~l:"ngc_mask" ~prec:Ir.Ops.single ~b:[ seq ] ~i:[ seq ] ~o:[]
      ~f:(function [| s; t |] -> if s >= t then 1. else 0. | _ -> 0.)
      ()
  in
  let block =
    Nn_blocks.decoder_only_block ~label:[ "ngc_dec" ] ~num_heads:2 ~d_k:8 ~d_v:8 ~d_ff:32 ()
  in
  let%op embedded = x +++ "... s | d; | s d => ... s | d" wpe in
  let y = block ~train_step:None embedded ~mask in
  let%op loss = (y *. y) ++ "...|... => |->0" in
  (loss, wpe)

let opt =
  let step = Train.grad_update loss in
  let comp = { step with asgns = Ir.Assignments.Block_comment ("ngc_step", step.asgns) } in
  let ctx = Train.init_params (Context.auto ()) Idx.Empty loss in
  let captured = ref None in
  let _ctx, _routine =
    Context.compile
      ~lowered_transform:(fun o ->
        captured := Some o;
        [ o ])
      ctx comp Idx.Empty
  in
  Option.value_exn ~here:[%here] !captured

let () = Tensor.default_grad_prec := saved_grad_prec
let narrow tn = Ir.Ops.is_narrow_float (Lazy.force tn.Tn.storage_prec)

type role = Output | Reduction

(* Each top-level nest ending in exactly one accumulating write to a narrow node whose cell is made
   of plain loop iterators (each its own axis's whole index, once): the node and its loops' roles,
   outermost first. *)
let narrow_plain_nests (llc : LL.t) : (Tn.t * (Idx.symbol * role) list) list =
  List.filter_map (LL.flat_lines [ llc ]) ~f:(fun stmt ->
      let writes = ref [] in
      Ll_test.walk stmt ~on_stmt:(fun st ->
          match st with
          | LL.Set { tn; idcs; _ } when LL.has_accumulating_cell st ->
              writes := (tn, idcs) :: !writes
          | _ -> ());
      match (Ll_test.nest_paths stmt, !writes) with
      | [ (_ :: _ :: _ as syms) ], [ (tn, idcs) ] when narrow tn ->
          let mentions s = Array.count idcs ~f:(Idx.axis_index_mentions_symbol s) in
          let plain s =
            mentions s = 0
            || (mentions s = 1 && Array.exists idcs ~f:(Idx.equal_axis_index (Idx.Iterator s)))
          in
          if not (List.for_all syms ~f:plain) then None
          else
            let roles =
              List.map syms ~f:(fun s -> (s, if mentions s > 0 then Output else Reduction))
            in
            if List.exists roles ~f:(fun (_, r) -> Poly.equal r Reduction) then Some (tn, roles)
            else None
      | _ -> None)

let outputs_first nest =
  let rec go seen_reduction = function
    | [] -> true
    | (_, Output) :: _ when seen_reduction -> false
    | (_, Output) :: rest -> go seen_reduction rest
    | (_, Reduction) :: rest -> go true rest
  in
  go false nest

let describe (tn, nest) =
  String.concat ~sep:"_" tn.Tn.label
  ^ " ["
  ^ String.concat ~sep:" "
      (List.map nest ~f:(fun (s, r) ->
           Idx.symbol_ident s ^ match r with Output -> ":out" | Reduction -> ":red"))
  ^ "]"

let () =
  let source = narrow_plain_nests opt.LL.source and llc = narrow_plain_nests opt.LL.llc in
  Stdio.eprintf "narrow_grad_census: source %s (not part of the golden)\n"
    (String.concat ~sep:"; " (List.map source ~f:describe));
  Stdio.eprintf "narrow_grad_census: llc %s (not part of the golden)\n%!"
    (String.concat ~sep:"; " (List.map llc ~f:describe));
  p_exists ~min:4
    "as lowered, some narrow plain-cell gradient nest runs a reduction loop outside its cell" source
    ~f:(fun (_, nest) -> not (outputs_first nest));
  p_all ~min:4
    "the optimized routine opens every narrow plain-cell gradient nest's cell loops first" llc
    ~f:(fun (_, nest) -> outputs_first nest);
  let wpe_grad = (Option.value_exn ~here:[%here] wpe.Tensor.diff).Tensor.grad in
  let wpe_nests nests = List.filter nests ~f:(fun (tn, _) -> Tn.equal tn wpe_grad) in
  p_all "as lowered, the positional table's bf16 gradient runs its batch loop outside its cell"
    (wpe_nests source) ~f:(fun (_, n) -> not (outputs_first n));
  p_all "the optimized routine opens the positional table's bf16 gradient's cell loops first"
    (wpe_nests llc) ~f:(fun (_, n) -> outputs_first n)

(* The tiles: the privatized preset over the whole routine, applied to a hermetic copy. *)
let () =
  let scratch =
    {
      opt with
      LL.traced_store = Hashtbl.copy opt.LL.traced_store;
      LL.optimize_ctx = LL.copy_optimize_ctx opt.LL.optimize_ctx;
    }
  in
  let sched = Autotune.extend_with_privatize ~accum_prec ~static_indices:[] [] scratch in
  let applied = Sched.apply sched scratch in
  let written = Hash_set.create (module Tn) in
  Ll_test.walk applied.LL.llc ~on_stmt:(function
    | LL.Set { tn; _ } -> Hash_set.add written tn
    | _ -> ());
  let narrow_targets =
    List.filter_map sched ~f:(function
      | Sched.Privatize { target; _ } when narrow target -> Some target
      | _ -> None)
  in
  (* A [Privatize] tile is labelled [acc] over its target's label ([Schedule.apply_privatize]). *)
  let tiles =
    Hash_set.to_list written
    |> List.filter ~f:(fun tn ->
        match tn.Tn.label with
        | "acc" :: rest ->
            List.exists narrow_targets ~f:(fun g -> List.equal String.equal g.Tn.label rest)
        | _ -> false)
  in
  Stdio.eprintf
    "narrow_grad_census: %d acc tiles over bf16 gradients: %s (not part of the golden)\n%!"
    (List.length tiles)
    (String.concat ~sep:", "
       (List.map tiles ~f:(fun tn ->
            Printf.sprintf "n%d_%s:%s" tn.Tn.id
              (String.concat ~sep:"_" tn.Tn.label)
              (Ir.Ops.prec_string (Lazy.force tn.Tn.storage_prec)))));
  p_all ~min:4 "every acc tile the privatized preset mints over a bf16 gradient is f32" tiles
    ~f:(fun tn -> Ir.Ops.equal_prec (Lazy.force tn.Tn.storage_prec) Ir.Ops.single)
