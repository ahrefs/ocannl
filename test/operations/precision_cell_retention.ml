(* gh-ocannl-2169: settling storage precision must release the stated-by-operands lazy's ancestor
   nodes. Keep only the result node, not its tensor (which owns its children). No compilation or
   backend is involved: this observes precision-cell reachability. *)
open Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims

let virtual_node (t : Tensor.t) =
  Ir.Tnode.update_memory_mode t.value Virtual (Site "99:test-setup");
  t.value

let[@inline never] settled_chain () =
  let leaf = NTDSL.number 1. in
  let a = NTDSL.O.relu leaf in
  let b = NTDSL.O.relu a in
  let result = NTDSL.O.relu b in
  let ancestors = List.map [ leaf; a; b ] ~f:(fun t -> Gc_witness.witness (virtual_node t)) in
  let node = virtual_node result in
  (* Closing shapes releases the open inference session independently of precision cells. *)
  ignore (Lazy.force node.dims : int array);
  ignore (Lazy.force node.storage_prec : Ir.Ops.prec);
  (node, Gc_witness.witness node, ancestors)

let () =
  let node, live, ancestors = settled_chain () in
  p "settled result node survives collections" (Gc_witness.kept live);
  p_all "settled result releases every virtual ancestor node" ancestors ~f:Gc_witness.reclaimed;
  ignore (Sys.opaque_identity node : Ir.Tnode.t)

(* Close the open-node registry without settling storage precision. *)
let[@inline never] unsettled ~keep () =
  let operand = NTDSL.O.relu (NTDSL.number 1.) in
  let other = NTDSL.number 2. in
  let result = NTDSL.add operand other () in
  ignore (Lazy.force result.value.dims : int array);
  p "operand storage remains unsettled" (not (Lazy.is_val operand.value.storage_prec));
  ((if keep then Some operand else None), Gc_witness.witness result, Gc_witness.witness other.value)

let () =
  let dropped, _, other = unsettled ~keep:false () in
  p "dropped graph releases other value node after closure" (Gc_witness.reclaimed other);
  ignore (Sys.opaque_identity dropped : Tensor.t option);
  let operand, consumer, other = unsettled ~keep:true () in
  p "live operand releases consumer tensor after closure" (Gc_witness.reclaimed consumer);
  p "live operand releases other value node after closure" (Gc_witness.reclaimed other);
  ignore (Sys.opaque_identity operand : Tensor.t option)

(* Keeping only the consumer's node is sufficient to keep a live reverse precision constraint. *)
let[@inline never] live_consumer () =
  let operand = NTDSL.O.relu (NTDSL.number 1.) in
  let result = NTDSL.add operand (NTDSL.number 2.) () in
  Ir.Tnode.update_prec result.value Ir.Ops.double;
  ignore (Lazy.force result.value.dims : int array);
  (operand.value, result.value, Gc_witness.witness result)

let () =
  let operand, result, tensor = live_consumer () in
  p "consumer tensor can die while its value node is held" (Gc_witness.reclaimed tensor);
  p "held consumer value node still widens its operand"
    (Ir.Ops.equal_prec (Lazy.force operand.storage_prec) Ir.Ops.double);
  ignore (Sys.opaque_identity result : Ir.Tnode.t)

let[@inline never] rebuild operand =
  let other = NTDSL.number 2. in
  let result = NTDSL.add operand other () in
  ignore (Lazy.force result.value.dims : int array);
  Gc_witness.witness other.value

let () =
  let operand = NTDSL.O.relu (NTDSL.number 1.) in
  let others = List.init 64 ~f:(fun _ -> rebuild operand) in
  p "shared operand stays unsettled across closed graph rebuilds"
    (not (Lazy.is_val operand.value.storage_prec));
  p_all "every rebuilt graph releases its other value node" others ~f:Gc_witness.reclaimed;
  ignore (Sys.opaque_identity operand : Tensor.t)
