open! Base
open! Ocannl
open! Ocannl.Operation.DSL_modules
open Stdio

(** The padding neutral element is part of a padded tensor's identity: the margins permanently hold
    a single committed value. Padded max-family windows do not participate: the clamped-window
    lowering (gh-504) range-guards the window to the operand's valid region instead of demanding
    [-inf] margins, so the max-pool never touches (nor commits) margins at all.

    Pinned here:
    - Composition: a padded max-pool (clamped) and a padded conv (0-neutral margins) reading the
      SAME tensor compose without a copy — the conv commits 0-margins on the shared buffer, the pool
      never reads them (all-negative input: a 0-margin max would corrupt the corners, a
      [-inf]-margin conv would produce [-inf]). Before gh-504 this was rejected ("Conflicting
      padding neutral elements") with a materialize-a-copy remedy.
    - The copy pattern keeps working: routing the conv through a materialized copy leaves the pool
      operand entirely unpadded, and the copy commits the conv's 0 neutral.
    - Single-operation pooling: a padded max-pool as sole consumer leaves its operand unpadded,
      results stable across repeated runs (the clamp is part of the compiled code, not buffer
      state). *)

let padding_elem_to_string (sh : Shape.t) =
  match sh.padding_elem with
  | None -> "unknown"
  | Some None -> "conflicting"
  | Some (Some v) -> Float.to_string v

(* Executed value checks (input is -16..-1, 3x3 windows, left/right margins 1/2): pooled[0,0] =
   max(-16,-15,-12,-11) = -11 — a 0-margin max would wrongly give 0; pooled[1,1] = -6 (window fully
   inside); pooled[3,3] = max(-6,-5,-2,-1) = -1; conv[0,0] = -16-15-12-11 = -54; conv[1,1] = sum of
   the top-left 3x3 = -99 — a -inf-margin conv would give -inf at the edges. *)
let check_values ctx pooled conv_out =
  let pooled_v = Context.get_values ctx pooled.Tensor.value in
  let conv_v = Context.get_values ctx conv_out.Tensor.value in
  printf "pooled[0,0]=%g pooled[1,1]=%g pooled[3,3]=%g\n%!" pooled_v.(0) pooled_v.(5) pooled_v.(15);
  printf "conv[0,0]=%g conv[1,1]=%g\n%!" conv_v.(0) conv_v.(5);
  Verdict.p "pooled values correct (windows clamped)"
    Float.(pooled_v.(0) = -11. && pooled_v.(5) = -6. && pooled_v.(15) = -1.);
  Verdict.p "conv values correct (margins 0)" Float.(conv_v.(0) = -54. && conv_v.(5) = -99.);
  (pooled_v, conv_v)

let test_shared_operand () =
  printf "Testing a clamped max-pool and a 0-margin conv sharing one operand...\n%!";
  Tensor.unsafe_reinitialize ();
  (* 4x4 input with values -16..-1: negative values expose wrong margin contents. *)
  let%op input = TDSL.range_of_shape ~output_dims:[ 4; 4 ] () - 16. in
  (* Padded conv on the SAME input as the pool: commits 0-margins on the shared buffer. Created (and
     compiled) first, embedding [input]'s computation. *)
  let%op conv_out = input +* "oh=+kh, ow=+kw; kh, kw => oh, ow" [ "kh"; "kw" ] (1.0 + 0.0) in
  Shape.set_dim kh 3;
  Shape.set_dim kw 3;
  (* Padded max-pool: clamped windows, no margin demand (gh-504). *)
  let%op pooled = input @^+ "oh=+wh, ow=+ww; wh, ww => oh, ow" [ "wh"; "ww" ] (stretch 0.0) in
  Shape.set_dim wh 3;
  Shape.set_dim ww 3;
  let ctx = Context.auto () in
  Train.set_materialized input.value;
  Train.set_materialized pooled.value;
  Train.set_materialized conv_out.value;
  (* [conv_out] embeds [input]'s computation and compiles first, committing the conv's margins on
     the shared buffer; the pool then compiles against the committed (0-neutral, padded) layout,
     reading it with clamped windows. *)
  let ctx = Train.forward_once ctx conv_out in
  let ctx = Train.forward_once ctx pooled in
  printf "shared operand's committed neutral (the conv's): %s\n%!"
    (padding_elem_to_string input.shape);
  ignore (check_values ctx pooled conv_out)

let test_separate_copies () =
  printf "\nTesting the copy pattern: conv through a materialized copy...\n%!";
  Tensor.unsafe_reinitialize ();
  let%op input = TDSL.range_of_shape ~output_dims:[ 4; 4 ] () - 16. in
  let%op pooled = input @^+ "oh=+wh, ow=+ww; wh, ww => oh, ow" [ "wh"; "ww" ] (stretch 0.0) in
  Shape.set_dim wh 3;
  Shape.set_dim ww 3;
  (* The copy is read by the conv; [input] itself is read only by the max-pool. *)
  let%op input_c = input ++ "a, b => a, b" in
  let%op conv_out = input_c +* "oh=+kh, ow=+kw; kh, kw => oh, ow" [ "kh"; "kw" ] (1.0 + 0.0) in
  Shape.set_dim kh 3;
  Shape.set_dim kw 3;
  let ctx = Context.auto () in
  Train.set_materialized input.value;
  Train.set_materialized input_c.value;
  Train.set_materialized pooled.value;
  Train.set_materialized conv_out.value;
  let ctx = Train.init_params ctx Train.IDX.empty conv_out in
  let ctx = Train.init_params ctx Train.IDX.empty pooled in
  let fwd_pooled = Train.forward pooled in
  let fwd_conv = Train.forward conv_out in
  let combined = Ir.Assignments.sequence [ fwd_pooled; fwd_conv ] in
  let ctx, routine = Context.compile ctx combined Train.IDX.empty in
  (* The clamped pool leaves its operand unpadded; the copy commits the conv's neutral. *)
  Verdict.p "pool operand stays unpadded"
    (match Lazy.force input.value.Ir.Tnode.padding with None -> true | Some _ -> false);
  printf "conv operand's committed neutral: %s\n%!" (padding_elem_to_string input_c.shape);
  let ctx = Context.run ctx routine in
  let pooled_v, conv_v = check_values ctx pooled conv_out in
  (* Second run: results stable. *)
  let ctx = Context.run ctx routine in
  let pooled_v2 = Context.get_values ctx pooled.value in
  let conv_v2 = Context.get_values ctx conv_out.value in
  Verdict.p_all2 "second-pass pooled values identical" pooled_v pooled_v2 ~f:Float.equal;
  Verdict.p_all2 "second-pass convolution values identical" conv_v conv_v2 ~f:Float.equal

(** Input used with a single padded max-pool: the operand stays unpadded (no margins to initialize),
    results stable across repeated runs. *)
let test_single_operation_padding () =
  printf "\n\n========================================\n%!";
  printf "Testing single-operation pooling (max-pool only)...\n%!";
  Tensor.unsafe_reinitialize ();

  (* Create a 4x4 input with negative values: -16..-1 *)
  let%op input = TDSL.range_of_shape ~output_dims:[ 4; 4 ] () - 16. in

  (* Only max-pool operation on input - no other operations use this input. *)
  let%op pooled = input @^+ "oh=+wh, ow=+ww; wh, ww => oh, ow" [ "wh"; "ww" ] (stretch 0.0) in
  Shape.set_dim wh 3;
  Shape.set_dim ww 3;

  let ctx = Context.auto () in
  Train.set_materialized input.value;
  Train.set_materialized pooled.value;

  let ctx = Train.init_params ctx Train.IDX.empty pooled in
  let fwd_pooled = Train.forward pooled in
  let ctx, routine = Context.compile ctx fwd_pooled Train.IDX.empty in
  Verdict.p "sole-consumer pool leaves the operand unpadded"
    (match Lazy.force input.value.Ir.Tnode.padding with None -> true | Some _ -> false);

  printf "\n=== First forward pass (single max-pool operation) ===\n%!";
  let ctx = Context.run ctx routine in
  let pooled_v = Context.get_values ctx pooled.value in
  printf "pooled[0,0]=%g pooled[1,1]=%g pooled[3,3]=%g\n%!" pooled_v.(0) pooled_v.(5) pooled_v.(15);
  Verdict.p "pooled values correct (windows clamped)"
    Float.(pooled_v.(0) = -11. && pooled_v.(5) = -6. && pooled_v.(15) = -1.);

  (* Run second pass - results stable *)
  let ctx = Context.run ctx routine in
  let pooled_v2 = Context.get_values ctx pooled.value in
  Verdict.p_all2 "second pass identical" pooled_v pooled_v2 ~f:Float.equal

let () =
  Verdict.case "test_shared_operand" (fun () ->
      test_shared_operand ();
      printf "\nShared-operand composition test completed!\n%!");
  Verdict.case "test_separate_copies" (fun () ->
      test_separate_copies ();
      printf "\nCopy-pattern test completed!\n%!");
  Verdict.case "test_single_operation_padding" (fun () ->
      test_single_operation_padding ();
      printf "\nSingle-operation pooling test completed!\n%!")
