open Base
open Ocannl
open Nn_blocks.DSL_modules
open Stdio

(* Block tensor literal syntax tests.
   Disambiguation: first-leaf heuristic.
   - Numeric literal first leaf → ndarray constant (existing behavior).
   - Non-numeric first leaf → block tensor (unsqueeze + concat).
   - Computed-number expressions like [Float.sin 1.0; 2.0] are reclassified as block tensors
     (accepted compatibility break; such patterns don't exist in the project). *)

let () =
  printf "=== Block Tensor Literal Tests ===\n\n%!";
  Tensor.unsafe_reinitialize ();

  (* --- Test 1: List — output axis stacking [x1; x2] --- *)
  printf "--- Test 1: List output axis [x1; x2] ---\n%!";
  let x1 =
    PDSL.ndarray [| 1.0; 2.0; 3.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 3 ] ()
  in
  let x2 =
    PDSL.ndarray [| 4.0; 5.0; 6.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 3 ] ()
  in
  let%op stacked = [x1; x2] in
  let ctx = Context.auto () in
  Train.set_materialized stacked.value;
  let ctx = Train.forward_once ctx stacked in
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ~style:`Default ctx stacked;

  (* --- Test 2: Array — batch axis stacking [|x1; x2|] --- *)
  printf "\n--- Test 2: Array batch axis [|x1; x2|] ---\n%!";
  let x3 =
    PDSL.ndarray [| 10.0; 20.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()
  in
  let x4 =
    PDSL.ndarray [| 30.0; 40.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()
  in
  let%op batched = [|x3; x4|] in
  Train.set_materialized batched.value;
  let ctx = Train.forward_once ctx batched in
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ~style:`Default ctx batched;

  (* --- Test 3: Tuple — input axis stacking (x1, x2) --- *)
  printf "\n--- Test 3: Tuple input axis (x1, x2) ---\n%!";
  let%op input_stack = (x1, x2) in
  Train.set_materialized input_stack.value;
  let ctx = Train.forward_once ctx input_stack in
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ~style:`Default ctx input_stack;

  (* --- Test 4: 3-way list [x1; x2; x3] --- *)
  printf "\n--- Test 4: 3-way list [x1; x2; x3] ---\n%!";
  let x5 =
    PDSL.ndarray [| 7.0; 8.0; 9.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 3 ] ()
  in
  let%op triple = [x1; x2; x5] in
  Train.set_materialized triple.value;
  let ctx = Train.forward_once ctx triple in
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ~style:`Default ctx triple;

  (* --- Test 5: Nested block matrix [[x1; x2]; [x1; x2]] (same-kind 2-level output nest) ---
     Two output-axis concat levels: the result reproduces the nested ndarray literal's shape,
     output axes 2; 2; 3 (two fresh size-2 stack axes ahead of the operand's 3). *)
  printf "\n--- Test 5: Nested [[x1; x2]; [x1; x2]] ---\n%!";
  let%op nested = [ [ x1; x2 ]; [ x1; x2 ] ] in
  Train.set_materialized nested.value;
  let ctx = Train.forward_once ctx nested in
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ~style:`Default ctx nested;

  (* --- Test 5b: Deeper same-kind nest (3 output-axis levels) ---
     [[[x1; x2]; [x1; x2]]; [[x1; x2]; [x1; x2]]] reproduces the nested literal at depth 3:
     output axes 2; 2; 2; 3. *)
  printf "\n--- Test 5b: Deeper nest [[[..]]] (3 levels) ---\n%!";
  let%op nested3 = [ [ [ x1; x2 ]; [ x1; x2 ] ]; [ [ x1; x2 ]; [ x1; x2 ] ] ] in
  Train.set_materialized nested3.value;
  let ctx = Train.forward_once ctx nested3 in
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ~style:`Default ctx nested3;

  (* --- Test 5c: Cross-kind nest, batch over output [| [x1; x2]; [x1; x2] |] ---
     Outer array introduces a new BATCH axis (size 2); inner list a new OUTPUT axis (size 2). The
     two new axes land on different kinds — operand output 3, so the result has batch axis 2 and
     output axes 2; 3 (memory order batch @ output @ input), reproducing the nested literal.
     NB the canonical three-way order puts the input (tuple) axis innermost, but ppx_op only
     recognizes a tuple as an input-axis stack at the top level (not nested), so the input axis is
     exercised separately in Test 5c2 below rather than via a nested tuple. *)
  printf "\n--- Test 5c: Cross-kind batch/output [| [x1; x2]; [x1; x2] |] ---\n%!";
  let%op cross = [| [ x1; x2 ]; [ x1; x2 ] |] in
  Train.set_materialized cross.value;
  let ctx = Train.forward_once ctx cross in
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ~style:`Default ctx cross;

  (* --- Test 5c2: Cross-kind nest, input over output ([x1; x2], [x1; x2]) ---
     Top-level tuple introduces a new INPUT axis (size 2); inner lists a new OUTPUT axis (size 2),
     so the result has output axes 2; 3 and input axis 2. Exercises concat arithmetic on the input
     kind. *)
  printf "\n--- Test 5c2: Cross-kind input/output ([x1; x2], [x1; x2]) ---\n%!";
  let%op cross_in = ([ x1; x2 ], [ x1; x2 ]) in
  Train.set_materialized cross_in.value;
  let ctx = Train.forward_once ctx cross_in in
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ~style:`Default ctx cross_in;

  (* --- Test 5d: Mixed-rank row [[x1; x2]; {c}] (AC #2) ---
     The first row [x1; x2] has output axes 2; 3; the bare sibling [c] is under-specified, so shape
     inference forces it to the sibling's stacked rank — c is inferred with output axes 2; 3 (it
     acquires the leading stack axis), rather than erroring or auto-unsqueezing. *)
  printf "\n--- Test 5d: Mixed-rank [[x1; x2]; {c}] ---\n%!";
  let%op mixed = [ [ x1; x2 ]; { c } ] in
  Train.set_materialized mixed.value;
  let ctx = Train.forward_once ctx mixed in
  printf "mixed (shape 2; 2; 3):\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ~style:`Default ctx mixed;
  printf "c (rank inferred up to output 2; 3):\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ~style:`Default ctx c;

  (* --- Test 6: Single element [x1] — unsqueeze --- *)
  printf "\n--- Test 6: Single element [x1] ---\n%!";
  let%op unsqueezed = [x1] in
  Train.set_materialized unsqueezed.value;
  let ctx = Train.forward_once ctx unsqueezed in
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ~style:`Default ctx unsqueezed;

  (* --- Test 7: Gradient flow (2-way) --- *)
  printf "\n--- Test 7: Gradient flow (2-way) ---\n%!";
  let g1 =
    PDSL.ndarray [| 1.0; 2.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()
  in
  let g2 =
    PDSL.ndarray [| 3.0; 4.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()
  in
  let%op grad_result = sin [g1; g2] in
  let%op loss = grad_result ++ "...|... => 0" in
  Train.set_materialized loss.value;
  Train.set_materialized grad_result.value;
  Train.set_materialized (Option.value_exn ~here:[%here] g1.diff).grad;
  Train.set_materialized (Option.value_exn ~here:[%here] g2.diff).grad;
  ignore (Train.update_once ~output_cd_file:false ctx loss);

  printf "grad_result (sin of stacked):\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ctx grad_result;
  printf "\nGradient of g1 (should be cos of original):\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true ctx g1;
  printf "\nGradient of g2:\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true ctx g2;

  (* --- Test 8: Gradient flow (3-way) --- *)
  printf "\n--- Test 8: Gradient flow (3-way) ---\n%!";
  let h1 =
    PDSL.ndarray [| 0.5; 1.5 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()
  in
  let h2 =
    PDSL.ndarray [| 1.0; 2.0 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()
  in
  let h3 =
    PDSL.ndarray [| 3.0; 0.1 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()
  in
  let%op grad3_result = sin [h1; h2; h3] in
  let%op loss3 = grad3_result ++ "...|... => 0" in
  Train.set_materialized loss3.value;
  Train.set_materialized grad3_result.value;
  Train.set_materialized (Option.value_exn ~here:[%here] h1.diff).grad;
  Train.set_materialized (Option.value_exn ~here:[%here] h2.diff).grad;
  Train.set_materialized (Option.value_exn ~here:[%here] h3.diff).grad;
  ignore (Train.update_once ~output_cd_file:false ctx loss3);

  printf "grad3_result (sin of 3-way stacked):\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ctx grad3_result;
  printf "\nGradient of h1:\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true ctx h1;
  printf "\nGradient of h2:\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true ctx h2;
  printf "\nGradient of h3:\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true ctx h3;

  (* --- Test 8b: Nested gradient flow ([[n1; n2]; [n3; n4]]) ---
     Backward through a 2-level nested stack must reach every leaf operand. Extends the Test 8
     precedent to the nested case: grad of each leaf is cos of its original (sum-of-sin loss). *)
  printf "\n--- Test 8b: Nested gradient flow [[n1; n2]; [n3; n4]] ---\n%!";
  let n1 = PDSL.ndarray [| 0.2; 0.4 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] () in
  let n2 = PDSL.ndarray [| 0.6; 0.8 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] () in
  let n3 = PDSL.ndarray [| 1.0; 1.2 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] () in
  let n4 = PDSL.ndarray [| 1.4; 1.6 |] ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] () in
  let%op grad_nested = sin [ [ n1; n2 ]; [ n3; n4 ] ] in
  let%op loss_nested = grad_nested ++ "...|... => 0" in
  Train.set_materialized loss_nested.value;
  Train.set_materialized grad_nested.value;
  Train.set_materialized (Option.value_exn ~here:[%here] n1.diff).grad;
  Train.set_materialized (Option.value_exn ~here:[%here] n2.diff).grad;
  Train.set_materialized (Option.value_exn ~here:[%here] n3.diff).grad;
  Train.set_materialized (Option.value_exn ~here:[%here] n4.diff).grad;
  let ctx = Train.update_once ~output_cd_file:false ctx loss_nested in
  printf "grad_nested (sin of nested stacked):\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ctx grad_nested;
  printf "\nGradient of n1 (should be cos of [0.2; 0.4] = [0.980; 0.921]):\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true ctx n1;
  printf "\nGradient of n4 (should be cos of [1.4; 1.6] = [0.170; -0.029]):\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true ctx n4;

  (* --- Test 9: ndarray constant tuple (1.0, 2.0) — must NOT become block tensor --- *)
  printf "\n--- Test 9: ndarray constant tuple (1.0, 2.0) ---\n%!";
  let%op nd_tuple = (1.0, 2.0) in
  Train.set_materialized nd_tuple.value;
  let ctx = Train.forward_once ctx nd_tuple in
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ~style:`Default ctx nd_tuple;

  (* --- Test 10: ndarray constant list [1.0; 2.0; 3.0] --- *)
  printf "\n--- Test 10: ndarray constant list [1.0; 2.0; 3.0] ---\n%!";
  let%op nd_list = [1.0; 2.0; 3.0] in
  Train.set_materialized nd_list.value;
  let ctx = Train.forward_once ctx nd_list in
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ~style:`Default ctx nd_list;

  (* --- Test 11: ndarray constant array [|1.0; 2.0|] --- *)
  printf "\n--- Test 11: ndarray constant array [|1.0; 2.0|] ---\n%!";
  let%op nd_array = [|1.0; 2.0|] in
  Train.set_materialized nd_array.value;
  let ctx = Train.forward_once ctx nd_array in
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ~style:`Default ctx nd_array;

  (* --- Test 12: Tuple in function application preserved as OCaml tuple --- *)
  (* [%oc fst] keeps fst as a plain OCaml function; (x1, x2) must remain an
     OCaml tuple, not a block tensor.  Wrapping in sin produces a root tensor. *)
  printf "\n--- Test 12: Tuple in apply preserved ---\n%!";
  let%op preserved_tuple = sin ([%oc fst] (x1, x2)) in
  Train.set_materialized preserved_tuple.value;
  let ctx = Train.forward_once ctx preserved_tuple in
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ~style:`Default ctx preserved_tuple;

  (* --- Test 13: List block tensor with input-dim components --- *)
  (* Verifies output-axis concat spec preserves input axes via broadcast. *)
  printf "\n--- Test 13: List block tensor with input dims ---\n%!";
  let m1 =
    PDSL.ndarray [| 1.0; 2.0; 3.0; 4.0 |] ~batch_dims:[] ~input_dims:[ 2 ] ~output_dims:[ 2 ] ()
  in
  let m2 =
    PDSL.ndarray [| 5.0; 6.0; 7.0; 8.0 |] ~batch_dims:[] ~input_dims:[ 2 ] ~output_dims:[ 2 ] ()
  in
  let%op stacked_mat = [m1; m2] in
  Train.set_materialized stacked_mat.value;
  let ctx = Train.forward_once ctx stacked_mat in
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ~style:`Default ctx stacked_mat;

  (* --- Test 14: Callable stack operation (no PPX block syntax) --- *)
  (* Exercises TDSL.O.stack directly: same output-axis stacking as Test 1, but built by an
     explicit call to the named operation rather than the [x1; x2] desugaring. *)
  printf "\n--- Test 14: Callable TDSL.O.stack `Output [| x1; x2 |] ---\n%!";
  let called_stack = TDSL.O.stack `Output [| x1; x2 |] in
  Train.set_materialized called_stack.value;
  let ctx = Train.forward_once ctx called_stack in
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ~style:`Default ctx called_stack;

  (* --- Test 15: Empty operand array raises Invalid_argument --- *)
  (* The callable API guards against an empty rhs array (which the PPX list/array/tuple syntax
     cannot produce, but a direct caller can), since it would otherwise generate a malformed
     concat spec. *)
  printf "\n--- Test 15: Empty stack rhses raises ---\n%!";
  (try
     let (_ : Tensor.t) = TDSL.O.stack `Output [||] in
     printf "ERROR: expected Invalid_argument, got a tensor\n%!"
   with Invalid_argument msg -> printf "raised Invalid_argument: %s\n%!" msg);

  (* --- Test 16: Tuple-argument elements are still translated --- *)
  (* Regression for the tuple-in-apply path: a tuple passed as a function argument keeps its
     structure (not an input-axis stack), but its elements must still be translated as %op
     expressions. Here [x1 + x2] must become a tensor add; if the element were left as raw OCaml
     this would not even compile (no [+] for Tensor.t). *)
  printf "\n--- Test 16: Tuple arg element translated ([%%oc fst] (x1 + x2, x1)) ---\n%!";
  let%op tuple_elem_translated = sin ([%oc fst] (x1 + x2, x1)) in
  Train.set_materialized tuple_elem_translated.value;
  let ctx = Train.forward_once ctx tuple_elem_translated in
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ~style:`Default ctx tuple_elem_translated;

  (* --- Test 17: Inline style on a beg_dims shape (regression for crash) --- *)
  (* stacked has a non-empty beg_dims row (output-axis concat), which previously caused
     Invalid_argument("index out of bounds") in to_doc_inline. *)
  printf "\n--- Test 17: Inline style on beg_dims shape (was crash) ---\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ~style:`Inline ctx stacked;

  printf "\n=== Block Tensor Literal Tests Complete ===\n%!"
