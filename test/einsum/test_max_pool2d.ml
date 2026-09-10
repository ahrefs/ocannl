open! Base
open! Ocannl
open! Nn_blocks.DSL_modules
open Stdio

let max_pool2d = Nn_blocks.max_pool2d

(** Test basic max_pool2d operation with default parameters.

    Default: stride=2, window_size=2 For 4x4 input, output should be 2x2. *)
let test_max_pool2d_basic () =
  printf "Testing max_pool2d with default parameters (stride=2, window=2)...\n%!";
  Tensor.unsafe_reinitialize ();

  (* Create a 4x4 input with 1 channel *)
  let input = TDSL.range_of_shape ~output_dims:[ 4; 4; 1 ] () in

  (* Apply max_pool2d with default params *)
  let%op output = max_pool2d () input in

  let ctx = Context.auto () in
  Train.set_materialized input.value;
  Train.set_materialized output.value;
  let ctx = Train.forward_once ctx output in

  printf "Input shape: 4x4x1\n%!";
  printf "Window size: 2x2\n%!";
  printf "Stride: 2\n%!";
  printf "Expected output spatial dims: 2x2\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ctx input;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ctx output;
  printf "\n%!"

(** Test max_pool2d with stride=2, window=3.

    For 7x7 input with stride=2, window=3 (no padding): Output size = (7 - 3) / 2 + 1 = 3. Valid
    convolution requires: input = stride * (output - 1) + window = 2 * 2 + 3 = 7 *)
let test_max_pool2d_window3 () =
  printf "Testing max_pool2d with stride=2, window=3...\n%!";
  Tensor.unsafe_reinitialize ();

  (* Create a 7x7 input with 2 channels *)
  let input = TDSL.range_of_shape ~output_dims:[ 7; 7; 2 ] () in

  (* Apply max_pool2d with window_size=3 *)
  let%op output = max_pool2d ~stride:2 ~window_size:3 () input in

  let ctx = Context.auto () in
  Train.set_materialized input.value;
  Train.set_materialized output.value;
  let ctx = Train.forward_once ctx output in

  printf "Input shape: 7x7x2\n%!";
  printf "Window size: 3x3\n%!";
  printf "Stride: 2\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ctx input;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ctx output;
  printf "\n%!"

(** Test max_pool2d with output dimension 1.

    For 3x3 input with stride=2, window=3 (no padding): Output size = (3 - 3) / 2 + 1 = 1. This
    tests the edge case where the kernel exactly covers the input. *)
let test_max_pool2d_output_dim_1 () =
  printf "Testing max_pool2d with output dimension 1...\n%!";
  Tensor.unsafe_reinitialize ();

  (* Create a 3x3 input with 1 channel *)
  let input = TDSL.range_of_shape ~output_dims:[ 3; 3; 1 ] () in

  (* Apply max_pool2d with window_size=3, stride=2 *)
  let%op output = max_pool2d ~stride:2 ~window_size:3 () input in

  let ctx = Context.auto () in
  Train.set_materialized input.value;
  Train.set_materialized output.value;
  let ctx = Train.forward_once ctx output in

  printf "Input shape: 3x3x1\n%!";
  printf "Window size: 3x3\n%!";
  printf "Stride: 2\n%!";
  printf "Expected output spatial dims: 1x1\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ctx input;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ctx output;
  printf "\n%!"

(** Test max_pool2d preserves channels.

    Channel dimension should pass through unchanged. *)
let test_max_pool2d_channels () =
  printf "Testing max_pool2d preserves channel dimension...\n%!";
  Tensor.unsafe_reinitialize ();

  (* Create a 4x4 input with 3 channels *)
  let input = TDSL.range_of_shape ~output_dims:[ 4; 4; 3 ] () in

  let%op output = max_pool2d () input in

  let ctx = Context.auto () in
  Train.set_materialized input.value;
  Train.set_materialized output.value;
  let ctx = Train.forward_once ctx output in

  printf "Input shape: 4x4x3\n%!";
  printf "Expected output shape: 2x2x3 (channels preserved)\n%!";
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ctx output;
  printf "\n%!"

(** Test backpropagation for max_pool2d.

    This tests that shape inference and gradient propagation work correctly during backpropagation
    for strided max pooling (which never uses padding).

    The input uses a pattern where maximum values are in different positions within each 2x2 window:
    - Top-left window: max at position (0,0) = 9
    - Top-right window: max at position (1,1) = 8
    - Bottom-left window: max at position (0,1) = 7
    - Bottom-right window: max at position (1,0) = 6

    Expected gradient behavior: For max pooling, gradients should flow back ONLY to positions that
    achieved the maximum in their respective windows. In proper max-pool backprop:
    - Position (0,0) should have grad=1 (max in top-left window)
    - Position (1,3) should have grad=1 (max in top-right window)
    - Position (2,1) should have grad=1 (max in bottom-left window)
    - Position (3,2) should have grad=1 (max in bottom-right window)
    - All other positions should have grad=0

    This is now correctly implemented using the [_rhs1/_rhs2] naming convention for intermediate
    tensors in tropical backprop. The [=:\@^] (max-reduce) assignment computes the max at each input
    position, and the naming convention causes ppx_cd to assign the RHS1/RHS2 projection slots,
    giving the condition tensors input shape instead of output shape. *)
let test_max_pool2d_backprop () =
  printf "\nTesting backprop for max_pool2d...\n%!";
  Tensor.unsafe_reinitialize ();

  (* Create a 4x4 input with 1 channel using a parameter (requires grad). Design: each 2x2 window
     has its max in a different position: Window positions: (row within window, col within window) -
     Top-left window [0-1, 0-1]: max 9 at (0,0) - Top-right window [0-1, 2-3]: max 8 at (1,1) -
     Bottom-left window [2-3, 0-1]: max 7 at (0,1) - Bottom-right window [2-3, 2-3]: max 6 at
     (1,0) *)
  let%op input =
    {
      x =
        [
          (* Row 0 *)
          [ [ 9 ]; [ 1 ]; [ 2 ]; [ 3 ] ];
          (* Row 1 *)
          [ [ 4 ]; [ 5 ]; [ 0 ]; [ 8 ] ];
          (* Row 2 *)
          [ [ 1 ]; [ 7 ]; [ 2 ]; [ 3 ] ];
          (* Row 3 *)
          [ [ 4 ]; [ 5 ]; [ 6 ]; [ 0 ] ];
        ];
    }
  in

  (* Apply max_pool2d with default params (stride=2, window=2) *)
  let%op output = max_pool2d () input in
  (* Sum to scalar for backprop *)
  let%op loss = output ++ "...|... => |->0" in

  let ctx = Context.auto () in
  Train.set_materialized loss.value;
  Train.set_materialized (Option.value_exn ~here:[%here] input.diff).grad;
  ignore (Train.update_once ~output_cd_file:false ctx loss);

  printf "Input shape: 4x4x1\n%!";
  printf "Window size: 2x2\n%!";
  printf "Stride: 2\n%!";
  printf "Expected max pool output: [[9, 8], [7, 6]]\n%!";
  printf "Expected loss (sum): 9 + 8 + 7 + 6 = 30\n%!";
  printf "Backprop completed successfully!\n%!";
  Train.printf ~here:[%here] ~with_code:false ctx loss;
  Train.printf ~here:[%here] ~with_code:false ~with_grad:true ctx input

(* 4x4 input with values -16..-1. All-negative inputs expose wrong margins: 0-margins would make the
   edge maxima 0 instead of the valid-range maxima. *)
let make_negative_input () =
  TDSL.init ~l:"input" ~prec:Ir.Ops.single ~o:[ 4; 4; 1 ]
    ~f:(fun idcs -> Float.of_int ((4 * idcs.(0)) + idcs.(1)) -. 16.)
    ()

let padding_to_string (tn : Ir.Tnode.t) =
  if not (Lazy.is_val tn.Ir.Tnode.padding) then "UNFORCED"
  else
    match Lazy.force tn.Ir.Tnode.padding with
    | None -> "None"
    | Some (arr, elem) ->
        Printf.sprintf "[%s] elem=%s"
          (String.concat ~sep:"; "
             (Array.to_list arr
             |> List.map ~f:(fun Ir.Ops.{ left; right } -> Printf.sprintf "%d/%d" left right)))
          (Float.to_string elem)

(* Expected padded pooling of [make_negative_input] at stride=2, window=3: the window at output (oh,
   ow) covers input rows 2*oh-1 .. 2*oh+1 (left margin 1, right 2), clamped to the valid range:
   out[0,0] = max rows{0,1} cols{0,1} = -11; out[0,1] = max rows{0,1} cols{1..3} = -9; out[1,0] =
   max rows{1..3} cols{0,1} = -3; out[1,1] = max rows{1..3} cols{1..3} = -1. *)
let check_padded_values ctx output =
  let v = Context.get_values ctx output.Tensor.value in
  printf "padded pooled = [%g %g; %g %g]\n%!" v.(0) v.(1) v.(2) v.(3);
  Verdict.p "padded max-pool values correct (clamped windows)"
    Float.(v.(0) = -11. && v.(1) = -9. && v.(2) = -3. && v.(3) = -1.)

(** Test max_pool2d with use_padding=true ("same" pooling): output spatial dims = input/stride. The
    pool lowers with clamped window bounds (gh-504): per output position the window is range-guarded
    to the valid input range, so the operand needs no margins at all — no -inf demand, no padding
    commitment. The operand here is a plain broadcast result (open trailing-dims row), exercising
    the mixed-anchoring row unification. *)
let test_max_pool2d_padded () =
  printf "Testing max_pool2d with use_padding=true (stride=2, window=3)...\n%!";
  Tensor.unsafe_reinitialize ();

  let input = make_negative_input () in
  let%op pre = input *. 1.0 in
  let%op output = max_pool2d ~stride:2 ~window_size:3 ~use_padding:true () pre in

  (* No private copy: the pool tensor reads [pre] as a direct child. *)
  Verdict.p "pool reads its operand directly (no copy)"
    (List.exists output.Tensor.children ~f:(fun ch -> phys_equal ch.Tensor.subtensor pre));
  Verdict.p "operand padding unforced before compilation"
    (not (Lazy.is_val pre.value.Ir.Tnode.padding));
  let ctx = Context.auto () in
  Train.set_materialized pre.value;
  Train.set_materialized output.value;
  let ctx = Train.forward_once ctx output in
  (* The clamp replaces the margin demand: the operand commits no padding. *)
  printf "operand committed padding = %s\n%!" (padding_to_string pre.value);
  check_padded_values ctx output

(** A data node created via [TDSL.init] wraps an eagerly allocated array: its unpadded layout is
    committed at creation. The clamped-window lowering (gh-504) demands no margins, so the padded
    max-pool reads it directly — before gh-504 this was rejected ("Operation padding exceeds
    resolved padding") with [max_pool2d_copy] as the remedy; the copy variant is kept working. *)
let test_max_pool2d_padded_locked_data () =
  printf "Testing max_pool2d use_padding=true on an init data node (locked layout)...\n%!";
  Tensor.unsafe_reinitialize ();

  let input = make_negative_input () in
  Verdict.p "data node layout committed at creation" (Lazy.is_val input.value.Ir.Tnode.padding);
  let%op output = max_pool2d ~stride:2 ~window_size:3 ~use_padding:true () input in
  let ctx = Context.auto () in
  Train.set_materialized input.value;
  Train.set_materialized output.value;
  let ctx = Train.forward_once ctx output in
  Verdict.p "clamped pool on a locked-layout data node accepted, stays unpadded"
    (match Lazy.force input.value.Ir.Tnode.padding with None -> true | Some _ -> false);
  check_padded_values ctx output;

  (* The materialized-copy variant keeps working (no margins on the copy either). *)
  Tensor.unsafe_reinitialize ();
  let input = make_negative_input () in
  let%op output = Nn_blocks.max_pool2d_copy ~stride:2 ~window_size:3 ~use_padding:true () input in
  let ctx = Context.auto () in
  Train.set_materialized input.value;
  Train.set_materialized output.value;
  let ctx = Train.forward_once ctx output in
  Verdict.p "max_pool2d_copy: data node stays unpadded"
    (match Lazy.force input.value.Ir.Tnode.padding with None -> true | Some _ -> false);
  check_padded_values ctx output

(** Inception-style sharing: one tensor feeding both a padded 0-neutral conv and a padded max-pool.
    The conv commits 0-margins on the shared buffer; the pool reads the same buffer with clamped
    windows (gh-504) and never sees the margins, so plain [max_pool2d] composes without a copy —
    before gh-504 this was rejected ("Conflicting padding neutral elements") with [max_pool2d_copy]
    as the remedy; the copy variant is kept working. *)
let test_max_pool2d_conflicting_consumers () =
  printf "Testing Inception-style sharing: padded conv + padded max-pool...\n%!";
  Tensor.unsafe_reinitialize ();

  let make_graph pool_block =
    let input = TDSL.range_of_shape ~output_dims:[ 4; 4; 2 ] () in
    let%op shared = input *. 1.0 in
    let conv =
      Nn_blocks.conv2d ~label:[ "incp" ] ~kernel_size:3 ~stride:1 ~use_padding:true ~out_channels:2
        ()
    in
    let conv_branch = conv shared in
    let pool_branch = pool_block shared in
    (shared, conv_branch, pool_branch)
  in
  (* shared(h, w, c) = 8h + 2w + c; padded max windows per output cell (see [check_padded_values]
     for window coverage): out(oh, ow, c) row-major. *)
  let expected_pool = [| 10.; 11.; 14.; 15.; 26.; 27.; 30.; 31. |] in
  let run_shared tag shared conv_branch pool_branch =
    let ctx = Context.auto () in
    Train.set_materialized shared.Tensor.value;
    Train.set_materialized conv_branch.Tensor.value;
    Train.set_materialized pool_branch.Tensor.value;
    (* [conv_branch] embeds [shared]'s computation, so it compiles (and runs) first. *)
    let ctx = Train.forward_once ctx conv_branch in
    let ctx = Train.forward_once ctx pool_branch in
    printf "%s: shared tensor carries the conv's committed margins: %s\n%!" tag
      (padding_to_string shared.Tensor.value);
    let v = Context.get_values ctx pool_branch.Tensor.value in
    printf "%s: pooled shared = [%g %g %g %g %g %g %g %g]\n%!" tag v.(0) v.(1) v.(2) v.(3) v.(4)
      v.(5) v.(6) v.(7);
    Verdict.pf_all2 "%s: pooled shared values correct (0-margins never read)" tag v expected_pool
      ~f:(fun a b -> Float.(a = b))
  in
  let shared, conv_branch, pool_branch =
    make_graph (fun x -> max_pool2d ~stride:2 ~window_size:3 ~use_padding:true () x)
  in
  run_shared "max_pool2d" shared conv_branch pool_branch;

  Tensor.unsafe_reinitialize ();
  let shared, conv_branch, pool_branch =
    make_graph (fun x -> Nn_blocks.max_pool2d_copy ~stride:2 ~window_size:3 ~use_padding:true () x)
  in
  run_shared "max_pool2d_copy" shared conv_branch pool_branch

let () =
  test_max_pool2d_basic ();
  test_max_pool2d_window3 ();
  test_max_pool2d_output_dim_1 ();
  test_max_pool2d_channels ();
  test_max_pool2d_backprop ();
  test_max_pool2d_padded ();
  test_max_pool2d_padded_locked_data ();
  test_max_pool2d_conflicting_consumers ();
  printf "\nAll max_pool2d tests completed!\n%!"
