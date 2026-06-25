(* Sasha Rush's Tensor Puzzles in OCANNL (gh-ocannl-308).

   Source: https://github.com/srush/Tensor-Puzzles -- 21 exercises that reimplement standard
   NumPy/PyTorch functions using only broadcasting, [arange], [where], arithmetic/comparison
   operators and matmul. This file solves the ones that map naturally onto OCANNL's extended einsum
   notation, demonstrates workarounds for the ones that need index tensors / convolution, and
   documents (as comments) the ones OCANNL cannot currently express, naming the missing capability.

   The output is a deterministic, human-checkable trace (tiny inputs) committed as
   [tensor_puzzles.expected]. Run under the sync_cc backend: OCANNL_BACKEND=sync_cc dune build
   test/einsum/tensor_puzzles.exe.output

   ============================================================================================
   CLASSIFICATION (this doubles as the GitHub issue #308 comment source) Summary: 8 einsum + 10
   op-level workaround = 18 solvable; 3 not expressible today (21 total). Key principle: any STATIC,
   data-independent index map (permutation, reshape, prefix, convolution) is a dense
   selector/operator matrix S[i,o] built from index-grid comparisons and contracted via einsum --
   O(size^2) but expressible. Only DATA-DEPENDENT index maps (the index is a runtime tensor value)
   genuinely need gather/scatter, which OCANNL lacks.
   ============================================================================================

   A. Solvable with OCANNL einsum / %op (pure contraction / pointwise / fixed-index reshaping): 1
   ones -- broadcast a constant 2 sum -- unary reduction: a ++ "i => 0" 3 outer -- a +* "i; j =>
   i->j" b 4 diag -- repeated-label extraction: m ++ "ii => i" (einsum1 extracts diagonals) 9 vstack
   -- block-literal stacking: [ a; b ] 19 heaviside -- nested pointwise where 20 repeat --
   block-literal stacking of d copies (broadcast) 21 bucketize -- broadcast compare + reduce: (not
   (v < bnd)) ++ "ij => i"

   B. Solvable with OCANNL ops / workarounds (index-grid selector matrices, concatenation) -- all
   for STATICALLY known sizes: 5 eye -- index grids + equality: (r ++ "i=>i0") = (r ++ "j=>0j") 6
   triu -- index grids + comparison: not ((r ++ "j=>0j") < (r ++ "i=>i0")) 7 cumsum --
   lower-triangular selector (i <= o), contracted: a +* "i; i o => o" L 8 diff -- finite-difference
   operator matrix D[i,o]=(i=o+1)-(i=o), contracted: a +* D 10 roll -- circular permutation selector
   (i = o+1, plus wrap), contracted 11 flip -- anti-diagonal selector (i + o = n-1), contracted 13
   pad_to -- pad via concatenation; truncate via rectangular selection matmul (static j) 14
   sequence_mask-- column index grid vs per-row length + where 17 flatten -- reshape selector
   S[i,j,k]=(i*cols+j = k) via range_of_shape, contract i,j 18 linspace -- affine arithmetic on a
   range tensor

   C. Not expressible today -- DATA-DEPENDENT indexing (index = a runtime tensor value): 12 compress
   -- left-align kept entries: output position = prefix-count of mask, then SCATTER 15 bincount --
   out[a[i]] += 1: SCATTER with values-as-indices (+ integer index tensors) 16 scatter_add--
   out[link[j]] += values[j]: the canonical SCATTER

   Most impactful missing capabilities (gap analysis): 1. Gather / scatter (dynamic indexing) -- the
   ONLY fundamental gap; unlocks compress(12), bincount(15), scatter_add(16); needed for embedding
   lookup, variable-length masking, and many real ML ops. Related: gh-ocannl-293 (sharding &
   slicing). 2. Scan / prefix-sum -- not needed for EXPRESSIBILITY (cumsum is the
   triangular-selector matmul above), but the O(n) streaming way; the selector matmul is O(n^2). 3.
   Native reshape/view, reverse, and modular/affine indexing -- ergonomic/efficient replacements for
   the O(size^2) selector-matrix workarounds for flatten/flip/roll.

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
   varies along axis 0) and [range n ++ "j=>0j"] is a [1,n] column grid (value varies along the last
   axis). Comparing the two broadcasts to an [n,n] grid. *)

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
  let%op p21 = (not (v21 ++ "i=>i0" < bnd21 ++ "j=>0j")) ++ "ij => i" in
  let ctx = Train.forward_once ctx p21 in
  show ctx "21 bucketize" "einsum" p21;

  (* --- B: solvable with workarounds --- *)

  (* 5 eye (j)->[j,j]: row grid [n,1] vs column grid [1,n], equality. *)
  let r3 = TDSL.range 3 in
  let%op p5 = r3 ++ "i=>i0" = r3 ++ "j=>0j" in
  let ctx = Train.forward_once ctx p5 in
  show ctx "5 eye" "workaround" p5;

  (* 6 triu (j)->[j,j]: i <= j is not (j < i). *)
  let%op p6 = not (r3 ++ "j=>0j" < r3 ++ "i=>i0") in
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
  let%op diffmat = (ri5 ++ "i=>i0" = ro4 ++ "j=>0j" + 1) - (ri5 ++ "i=>i0" = ro4 ++ "j=>0j") in
  let%op p8 = a_diff +* "i; i o => o" diffmat in
  let ctx = Train.forward_once ctx p8 in
  show ctx "8 diff" "workaround" p8;

  (* 13 pad_to ([i],j)->[j]: both directions, for STATICALLY known target sizes. Pad: a=[1;2;3] ->
     length 5 = [1;2;3;0;0] via concatenation with a zero block. *)
  let%op a13 = [ 1.; 2.; 3. ] in
  let%op z13 = [ 0.; 0. ] in
  let%op p13 = (a13, z13) ++^ "i; j => i^j" [ "i"; "j" ] in
  let ctx = Train.forward_once ctx p13 in
  show ctx "13 pad_to/pad" "workaround" p13;
  (* Truncate: a=[1;2;3;4;5] -> first 3 = [1;2;3] via a rectangular selection matmul, sel[i,o]=(i=o)
     with i in 5, o in 3, contracting i. (A runtime target size j would need a dynamic/range-slice
     op; here the target size is a static shape, so this is the puzzle in the static-shape
     regime.) *)
  let%op a13t = [ 1.; 2.; 3.; 4.; 5. ] in
  let ri5t = TDSL.range 5 and ro3t = TDSL.range 3 in
  let%op sel13 = ri5t ++ "i=>i0" = ro3t ++ "j=>0j" in
  let%op p13t = a13t +* "i; i o => o" sel13 in
  let ctx = Train.forward_once ctx p13t in
  show ctx "13 pad_to/trunc" "workaround" p13t;

  (* 14 sequence_mask ([i,j],[i])->[i,j]: keep values[i,j] while j < length[i]. *)
  let v14 = TDSL.range_of_shape ~output_dims:[ 2; 3 ] () in
  let%op lens14 = [ 2.; 1. ] in
  let r3c = TDSL.range 3 in
  let%op p14 = where (r3c ++ "j=>0j" < lens14 ++ "i=>i0") v14 0 in
  let ctx = Train.forward_once ctx p14 in
  show ctx "14 seq_mask" "workaround" p14;

  (* 18 linspace (lo,hi,n)->[n]: out[k] = lo + (hi-lo)*k/(n-1). Here lo=2, hi=10, n=5 => step 2. The
     n=1 case is special (the formula divides by n-1=0) and would be handled separately. *)
  let n18 = 5 and lo18 = 2. and hi18 = 10. in
  let r5 = TDSL.range n18 in
  let step18 = (hi18 -. lo18) /. Float.of_int (n18 - 1) in
  let%op p18 = (r5 *. !.step18) + !.lo18 in
  let ctx = Train.forward_once ctx p18 in
  show ctx "18 linspace" "workaround" p18;

  (* The next four are STATIC, data-independent index maps -- so they are the same dense
     selector-matrix pattern as diff/pad_to-truncate: build a 0/1 selector S[i,o] from index-grid
     comparisons and contract the input over i. They are O(size^2) (or O(size^3) for flatten), not
     how you'd implement them for real, but they ARE expressible. Only DATA-DEPENDENT index maps
     (compress/bincount/scatter_add) are out of reach. *)

  (* 7 cumsum ([i])->[i]: out[o] = sum_{i<=o} a[i] = lower-triangular ones matrix times a. *)
  let%op a7 = [ 1.; 2.; 3.; 4. ] in
  let ri7 = TDSL.range 4 and co7 = TDSL.range 4 in
  let%op tri7 = not (co7 ++ "j=>0j" < ri7 ++ "i=>i0") in
  (* L[i,o] = (i <= o) *)
  let%op p7 = a7 +* "i; i o => o" tri7 in
  let ctx = Train.forward_once ctx p7 in
  show ctx "7 cumsum" "workaround" p7;

  (* 11 flip ([i])->[i]: out[o] = a[n-1-o] via the anti-diagonal selector (i + o = n-1). *)
  let%op a11 = [ 1.; 2.; 3.; 4. ] in
  let ri11 = TDSL.range 4 and co11 = TDSL.range 4 in
  let%op anti11 = ri11 ++ "i=>i0" + (co11 ++ "j=>0j") = 3 in
  let%op p11 = a11 +* "i; i o => o" anti11 in
  let ctx = Train.forward_once ctx p11 in
  show ctx "11 flip" "workaround" p11;

  (* 10 roll ([i])->[i]: circular shift, out[o] = a[(o+1) mod n]. Static circular selector: S[i,o] =
     (i = o+1) plus the wrap case (i = 0 and o = n-1). *)
  let%op a10 = [ 1.; 2.; 3.; 4. ] in
  let ri10 = TDSL.range 4 and co10 = TDSL.range 4 in
  let%op circ10 =
    (ri10 ++ "i=>i0" = co10 ++ "j=>0j" + 1) + ((ri10 ++ "i=>i0" = 0) *. (co10 ++ "j=>0j" = 3))
  in
  let%op p10 = a10 +* "i; i o => o" circ10 in
  let ctx = Train.forward_once ctx p10 in
  show ctx "10 roll" "workaround" p10;

  (* 17 flatten ([i,j])->[i*j]: static affine reshape out[k] = a[i,j] where k = i*cols + j.
     range_of_shape already fills row-major offsets, so pos[i,j] = i*cols + j directly; the selector
     is S[i,j,k] = (pos[i,j] = k), contracted over i and j. *)
  let%op a17 = [ [ 10.; 20.; 30. ]; [ 40.; 50.; 60. ] ] in
  let pos17 = TDSL.range_of_shape ~output_dims:[ 2; 3 ] () in
  let ck17 = TDSL.range 6 in
  let%op sel17 = pos17 ++ "i j => i j 0" = ck17 ++ "k => 0 0 k" in
  let%op p17 = a17 +* "i j; i j k => k" sel17 in
  let ctx = Train.forward_once ctx p17 in
  show ctx "17 flatten" "workaround" p17;

  Stdio.printf
    "\n\
     Not expressible today -- DATA-DEPENDENT indexing (the index is a runtime tensor value), which\n\
     needs gather/scatter that OCANNL lacks:\n\
    \  12 compress   -- left-align kept entries: position = prefix-count of mask, then SCATTER\n\
    \  15 bincount   -- out[a[i]] += 1: SCATTER with values-as-indices + integer index tensors\n\
    \  16 scatter_add-- out[link[j]] += values[j]: the canonical SCATTER\n\
     %!";
  ()

(* 13 pad_to, note on the runtime-size limitation: both directions above use a STATICALLY known
   target length. The only part not covered is a *runtime* integer target j -- concatenation needs a
   statically known pad width and the selection matmul needs a statically known output axis, so a
   data-dependent j would require a dynamic/range-slice op OCANNL does not yet have. OCANNL's
   slicing operator [@|] selects a single index of the leftmost batch axis, not a contiguous prefix
   range. *)
