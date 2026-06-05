(* GNN broadcast: forming edge/pair features from node features without expand_dims.

   This is the "GNN broadcast" row of the Rosetta-stone table in the blog post
   "A Shape Is Not Its Index". Star's GNN example notes that adding a (source, target)
   array to a (batch, source, target, feature) tensor forces an [expand_dims] in NumPy,
   and that naming the axes lets you avoid tracking axis positions. OCANNL goes further:
   the named axes in the einsum spec place each operand on its own node axis and
   co-iterate the shared feature axis, so the size-1 insertion never has to be written;
   and the three independent row variables let the batch (graph) row stay polymorphic,
   so the same operator runs on a single graph or a batch of graphs with no reshape.

   To slot into the suite: drop next to [test/einsum_trivia.ml] and convert the
   [Tensor.print] calls below into the assertion style used there (e.g. [%expect]). *)

open Ocannl.Operation.DSL_modules

(* Pairwise message tensor from node features.
   out[i, j, f] = src[i, f] * dst[j, f]
   - [i] appears only on the left, so the left operand broadcasts over [j];
   - [j] appears only on the right, so the right operand broadcasts over [i];
   - [f] is shared, so it is co-iterated (pointwise).
   PyTorch equivalent: src[:, None, :] * dst[None, :, :] -- two expand_dims. *)
let%op pairwise a b = a +* b "i, f; j, f => i, j, f"

(* Same construction, but the batch (graph) row is a row variable [..g..], so the
   operator is polymorphic in the number of leading graph/sample axes: one graph or a
   batch of graphs, no reshape, no expand_dims. *)
let%op pairwise_batched a b = a +* b "..g.. | i, f; ..g.. | j, f => ..g.. | i, j, f"

(* GCN-style neighbourhood aggregation.
   out[v, f] = sum_u adj[v, u] * h[u, f]
   [u] appears in both operands but not in the result, so it is summed (the contraction
   that aggregates each node's neighbourhood). Batched over graphs via [..g..]. *)
let%op aggregate adj h = adj +* h "..g.. | v, u; ..g.. | u, f => ..g.. | v, f"

let () =
  (* Pin sizes with explicit inline-param output dims: nodes = 3, features = 4. *)
  let%op edges = pairwise { src; o = [ 3; 4 ] } { dst; o = [ 3; 4 ] } in
  (* Expected output row: i, j, f = 3, 3, 4 *)
  Tensor.print ~force:true edges;

  (* Batched: 2 graphs of 3 nodes, 4 features. The batch axis is inferred for the
     row variable [..g..] without any reshape on the operands. *)
  let%op edges_b =
    pairwise_batched { src_b; b = [ 2 ]; o = [ 3; 4 ] } { dst_b; b = [ 2 ]; o = [ 3; 4 ] }
  in
  (* Expected: batch g = 2, output i, j, f = 3, 3, 4 *)
  Tensor.print ~force:true edges_b;

  (* Aggregation: adjacency 3x3, features 4, per graph in a batch of 2. *)
  let%op out =
    aggregate { adj; b = [ 2 ]; o = [ 3; 3 ] } { h; b = [ 2 ]; o = [ 3; 4 ] }
  in
  (* Expected: batch g = 2, output v, f = 3, 4 (u = 3 summed out) *)
  Tensor.print ~force:true out
