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
