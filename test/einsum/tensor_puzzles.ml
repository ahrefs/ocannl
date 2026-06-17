(* Sasha Rush's Tensor Puzzles in OCANNL (gh-ocannl-308).

   Source: https://github.com/srush/Tensor-Puzzles -- 21 exercises that reimplement standard
   NumPy/PyTorch functions using only broadcasting, [arange], [where], arithmetic/comparison
   operators and matmul. This file solves the ones that map naturally onto OCANNL's extended einsum
   notation, demonstrates workarounds for the ones that need index tensors / convolution, and
   documents (as comments) the ones OCANNL cannot currently express, naming the missing capability.

   The output is a deterministic, human-checkable trace (tiny inputs) committed as
   [tensor_puzzles.expected]. Run under the sync_cc backend:
     OCANNL_BACKEND=sync_cc dune build test/einsum/tensor_puzzles.exe.output

   ============================================================================================
   CLASSIFICATION (this doubles as the GitHub issue #308 comment source)
   Summary: 8 einsum + 6 op-level workaround = 14 solvable; 7 not expressible today (21 total).
   ============================================================================================

   A. Solvable with OCANNL einsum / %op (pure contraction / pointwise / fixed-index reshaping):
      1  ones        -- broadcast a constant
      2  sum         -- unary reduction: a ++ "i => 0"
      3  outer       -- a +* "i; j => i->j" b
      4  diag        -- repeated-label extraction: m ++ "ii => i"  (einsum1 extracts diagonals)
      9  vstack      -- block-literal stacking: [ a; b ]
      19 heaviside   -- nested pointwise where
      20 repeat      -- block-literal stacking of d copies (broadcast)
      21 bucketize   -- broadcast compare + reduce: (not (v < bnd)) ++ "ij => i"

   B. Solvable with OCANNL ops / workarounds (per-axis index grids, convolution, concatenation):
      5  eye          -- index grids + equality: (r ++ "i=>i0") = (r ++ "j=>0j")
      6  triu         -- index grids + comparison: not ((r ++ "j=>0j") < (r ++ "i=>i0"))
      8  diff         -- finite-difference operator matrix D[i,o]=(i=o+1)-(i=o), contracted: a +* D
      13 pad_to       -- concatenation for padding (fixed sizes); truncation is a gap (see below)
      14 sequence_mask-- column index grid vs per-row length + where
      18 linspace     -- affine arithmetic on a range tensor

   C. Not expressible today (each names the missing primitive):
      7  cumsum     -- loop-carried prefix dependency; needs a SCAN primitive
      10 roll       -- a[(i+1) mod n]; needs GATHER / circular-shift (modular dynamic index)
      11 flip       -- a[n-1-i]; needs negative-stride affine indexing or a REVERSE op
      12 compress   -- prefix-count then place; needs SCAN + SCATTER
      15 bincount   -- out[a[i]] += 1; needs SCATTER (values-as-indices)
      16 scatter_add-- out[link[j]] += values[j]; the canonical SCATTER
      17 flatten    -- reshape across rank; needs a RESHAPE / view op

   Most impactful missing capabilities (gap analysis):
      1. Gather / scatter (dynamic indexing) -- unlocks compress(12), bincount(15), scatter_add(16)
         and a gather route for roll(10); needed for embedding lookup, variable-length masking,
         and many real ML ops. Related: gh-ocannl-293 (sharding & slicing).
      2. Scan / prefix-sum -- unlocks cumsum(7) and compress(12); needed for sequence processing.
      3. Reshape / view -- unlocks flatten(17).
      4. Reverse / modular affine indexing -- unlocks flip(11) and roll(10).

   DISCOVERED LIMITATION (diff): the valid-convolution einsum [a +* "o<+k; k => o" kernel] hangs
   OCANNL's shape inference even on a length-5 vector with a length-2 kernel (no error, just spins).
   The operator-matrix formulation used below avoids it. Worth filing as a shape-inference bug for
   1-D valid convolutions over output-only axes.

   NOTE on semantics: OCANNL is float-valued throughout, so integer-domain puzzles (bincount,
   scatter_add) are doubly out of reach -- they need both dynamic indexing AND integer index
   tensors. Equality comparisons below are only safe because the operands are small literals or
   exact [range] indices.

   NOTE on per-axis index grids: a number in an einsum axis spec is a FIXED INDEX (it selects /
   creates a particular slot), not "axis N". So [range n ++ "i=>i0"] is a [n,1] row grid (value
   varies along axis 0) and [range n ++ "j=>0j"] is a [1,n] column grid (value varies along the
   last axis). Comparing the two broadcasts to an [n,n] grid. *)

open Base
open Ocannl
module Tn = Ir.Tnode
module IDX = Train.IDX
module CDSL = Train.CDSL
open Nn_blocks.DSL_modules

(* Compact, deterministic printer: flatten the tensor's host values and join with spaces. *)
let show ctx name category t =
  let vals = Context.get_values ctx t.Tensor.value in
  let s =
    String.concat ~sep:" " (Array.to_list (Array.map vals ~f:(fun v -> Printf.sprintf "%.4g" v)))
  in
  Stdio.printf "#%-13s [%-13s] %s\n%!" name category s

let () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  Stdio.printf "=== Tensor Puzzles in OCANNL (gh-ocannl-308) ===\n%!";

  (* --- A: solvable with einsum / %op --- *)

  (* 1 ones (i)->[i]: a length-i vector of ones. Tie the length to a range, zero it, add one. *)
  let r4 = TDSL.range 4 in
  let%op p1 = (r4 *. 0.) + 1 in
  let ctx = Train.forward_once ctx p1 in
  show ctx "1 ones" "einsum" p1;

  (* 2 sum ([i])->[1]: contract the axis to a single number. *)
  let%op p2 = r4 ++ "i => 0" in
  let ctx = Train.forward_once ctx p2 in
  show ctx "2 sum" "einsum" p2;

  (* 3 outer ([i],[j])->[i,j]: no shared index => outer product. *)
  let a3 = TDSL.range 3 and b4 = TDSL.range 4 in
  let%op p3 = a3 +* "i; j => i->j" b4 in
  let ctx = Train.forward_once ctx p3 in
  show ctx "3 outer" "einsum" p3;

  (* 4 diag ([i,i])->[i]: repeated-label extraction. m is row-major 0..8 reshaped [3,3]. *)
  let m33 = TDSL.range_of_shape ~output_dims:[ 3; 3 ] () in
  let%op p4 = m33 ++ "ii => i" in
  let ctx = Train.forward_once ctx p4 in
  show ctx "4 diag" "einsum" p4;

  (* 9 vstack ([i],[i])->[2,i]: block-literal stacking introduces a fresh leading axis. *)
  let r3a = TDSL.range 3 in
  let%op v9b = [ 10.; 20.; 30. ] in
  let%op p9 = [ r3a; v9b ] in
  let ctx = Train.forward_once ctx p9 in
  show ctx "9 vstack" "einsum" p9;

  (* 19 heaviside (a,b)->[i]: where(a=0, b, where(a>0, 1, 0)); a>0 is 0<a. *)
  let%op a19 = [ -1.; 0.; 2. ] in
  let%op b19 = [ 5.; 5.; 5. ] in
  let%op p19 = where (a19 = 0) b19 (where (0 < a19) 1 0) in
  let ctx = Train.forward_once ctx p19 in
  show ctx "19 heaviside" "einsum" p19;

  (* 20 repeat ([i])->[d,i]: stack d copies (broadcast along the new axis). *)
  let r3d = TDSL.range 3 in
  let%op p20 = [ r3d; r3d; r3d ] in
  let ctx = Train.forward_once ctx p20 in
  show ctx "20 repeat" "einsum" p20;

  (* 21 bucketize ([i],[j])->[i]: count boundaries <= v. With only (<), v>=b is not(v<b). *)
  let%op v21 = [ 0.5; 2.5; 5.0 ] in
  let%op bnd21 = [ 1.; 3.; 5. ] in
  let%op p21 = not ((v21 ++ "i=>i0") < (bnd21 ++ "j=>0j")) ++ "ij => i" in
  let ctx = Train.forward_once ctx p21 in
  show ctx "21 bucketize" "einsum" p21;

  (* --- B: solvable with workarounds --- *)

  (* 5 eye (j)->[j,j]: row grid [n,1] vs column grid [1,n], equality. *)
  let r3 = TDSL.range 3 in
  let%op p5 = (r3 ++ "i=>i0") = (r3 ++ "j=>0j") in
  let ctx = Train.forward_once ctx p5 in
  show ctx "5 eye" "workaround" p5;

  (* 6 triu (j)->[j,j]: i <= j is not (j < i). *)
  let%op p6 = not ((r3 ++ "j=>0j") < (r3 ++ "i=>i0")) in
  let ctx = Train.forward_once ctx p6 in
  show ctx "6 triu" "workaround" p6;

  (* 8 diff ([i])->[i-1]: out[o] = a[o+1] - a[o]. The natural spelling is a valid (length-shrinking)
     convolution with kernel [-1; 1]: [a +* "o<+k; k => o" kernel]. That currently HANGS OCANNL's
     shape inference on this tiny case (a real limitation, noted in the gap analysis below), so we
     instead build the finite-difference operator matrix D[i,o] = (i = o+1) - (i = o) from index
     grids and contract it -- a pure-einsum formulation. Boundary convention: result length i-1,
     matching PyTorch's [diff] (no first-element padding). *)
  let%op a_diff = [ 0.; 1.; 3.; 6.; 10. ] in
  let ri5 = TDSL.range 5 and ro4 = TDSL.range 4 in
  (* D[i,o] = (i = o+1) - (i = o) is the finite-difference operator [5,4]; then contract i. *)
  let%op diffmat = ((ri5 ++ "i=>i0") = ((ro4 ++ "j=>0j") + 1)) - ((ri5 ++ "i=>i0") = (ro4 ++ "j=>0j")) in
  let%op p8 = a_diff +* "i; i o => o" diffmat in
  let ctx = Train.forward_once ctx p8 in
  show ctx "8 diff" "workaround" p8;

  (* 13 pad_to ([i],j)->[j]: fixed-size padding via concatenation with a zero block.
     a=[1;2;3] padded to length 5 = [1;2;3;0;0]. Truncation to a SMALLER fixed size, and padding
     to a RUNTIME size j, are gaps -- see the comment after this block. *)
  let%op a13 = [ 1.; 2.; 3. ] in
  let%op z13 = [ 0.; 0. ] in
  let%op p13 = (a13, z13) ++^ "i; j => i^j" [ "i"; "j" ] in
  let ctx = Train.forward_once ctx p13 in
  show ctx "13 pad_to" "workaround" p13;

  (* 14 sequence_mask ([i,j],[i])->[i,j]: keep values[i,j] while j < length[i]. *)
  let v14 = TDSL.range_of_shape ~output_dims:[ 2; 3 ] () in
  let%op lens14 = [ 2.; 1. ] in
  let r3c = TDSL.range 3 in
  let%op p14 = where ((r3c ++ "j=>0j") < (lens14 ++ "i=>i0")) v14 0 in
  let ctx = Train.forward_once ctx p14 in
  show ctx "14 seq_mask" "workaround" p14;

  (* 18 linspace (lo,hi,n)->[n]: out[k] = lo + (hi-lo)*k/(n-1). Here lo=2, hi=10, n=5 => step 2.
     The n=1 case is special (the formula divides by n-1=0) and would be handled separately. *)
  let n18 = 5 and lo18 = 2. and hi18 = 10. in
  let r5 = TDSL.range n18 in
  let step18 = (hi18 -. lo18) /. Float.of_int (n18 - 1) in
  let%op p18 = (r5 *. !.step18) + !.lo18 in
  let ctx = Train.forward_once ctx p18 in
  show ctx "18 linspace" "workaround" p18;

  Stdio.printf
    "\n\
     Not expressible today (need new primitives):\n\
    \  7  cumsum     -- needs SCAN (loop-carried prefix dependency)\n\
    \  10 roll       -- needs GATHER / circular shift (modular dynamic index)\n\
    \  11 flip       -- needs negative-stride affine indexing or a REVERSE op\n\
    \  12 compress   -- needs SCAN + SCATTER\n\
    \  15 bincount   -- needs SCATTER (values-as-indices) + integer index tensors\n\
    \  16 scatter_add-- needs SCATTER\n\
    \  17 flatten    -- needs RESHAPE / view across rank\n\
     %!";
  ()

(* 13 pad_to, continued: TRUNCATION to a smaller fixed size is not directly available -- OCANNL's
   slicing operator [@|] selects a single index of the leftmost batch axis, not a contiguous
   prefix range, and there is no range-slice op. It can be emulated by a matmul against a
   rectangular selection matrix (an eye-like [n,k] grid), but that is no longer the puzzle's spirit.
   Padding to a *runtime* size j (the puzzle's signature) is also a gap: concatenation requires the
   pad width to be a statically known shape, whereas j is a runtime integer. *)
