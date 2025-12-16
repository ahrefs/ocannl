open Base
open Ocannl.Nn_blocks.DSL_modules
open Stdio

(** {1 Random Number Generation Tests}

    IMPORTANT: Understanding OCANNL's counter-based PRNG architecture:

    The [uniform_at], [normal_at], [kaiming_at], [xavier_at] functions use a counter-based PRNG
    (Threefry). The [counter] argument is NOT meant to determine the output shape! It is a "mix-in"
    to bifurcate randomness across different counter values.

    The architecture: 1. [counter] should be scalar or small (dimension-1) so it broadcasts to any
    result shape 2. [Range_over_offsets] generates indices over the result shape for mixing 3.
    [uint4x32_to_prec_uniform] reshapes from the uint4x32 backbone to the target shape 4. The output
    shape is determined by shape inference from how the result is used

    For [kaiming] and [xavier] operations:
    - The result tensor's shape determines fan_in/fan_out through einsum dimension capture
    - The counter is just for randomness bifurcation (e.g., different steps in training) *)

let create_histogram values ~num_bins ~min_val ~max_val =
  let bins = Array.create ~len:num_bins 0 in
  let bin_width = (max_val -. min_val) /. Float.of_int num_bins in
  Array.iter values ~f:(fun x ->
      let bin_idx =
        Int.min (num_bins - 1) (Int.max 0 (Float.to_int ((x -. min_val) /. bin_width)))
      in
      bins.(bin_idx) <- bins.(bin_idx) + 1);
  bins

let print_histogram bins ~title ~max_width =
  printf "\n%s\n" title;
  printf "%s\n" (String.make (String.length title) '=');
  let max_count = Array.max_elt bins ~compare:Int.compare |> Option.value ~default:0 in
  let total = Array.fold bins ~init:0 ~f:( + ) in
  Array.iteri bins ~f:(fun i count ->
      let bar_width = count * max_width / max_count in
      let bar = String.make bar_width '#' in
      let percentage = Float.of_int count /. Float.of_int total *. 100.0 in
      printf "Bin %2d: %s %4d (%.1f%%)\n" i bar count percentage)

(** Test uniform_at with a SCALAR counter, letting shape be inferred from usage. This is the correct
    way to use uniform_at - counter is for randomness bifurcation, not for determining the output
    shape. *)
let test_uniform_at_with_shape () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let module O = TDSL.O in
  (* Scalar counter - just for randomness bifurcation *)
  let counter = NTDSL.number 44.0 in

  (* Create a target tensor with the desired shape to drive shape inference *)
  let num_values = 10000 in
  let target = TDSL.range num_values in

  (* uniform_at with scalar counter, shape inferred from pointwise operation with target *)
  let%op uniform_values = O.uniform_at counter + (target *. 0.0) in
  Ir.Tnode.update_prec uniform_values.value Ir.Ops.single;

  (* Compile and run *)
  Ocannl.Train.set_hosted uniform_values.value;
  ignore (Ocannl.Train.forward_once ctx uniform_values);
  let result = Ir.Tnode.get_values uniform_values.value in

  printf "Uniform Distribution Test (shape-inferred, scalar counter)\n";
  printf "===========================================================\n";
  printf "Generated %d values with scalar counter\n" (Array.length result);

  (* Create and print histogram *)
  let num_bins = 20 in
  let bins = create_histogram result ~num_bins ~min_val:0.0 ~max_val:1.0 in
  print_histogram bins ~title:"Uniform Distribution [0, 1) Histogram" ~max_width:40;

  (* Statistical tests *)
  let mean = Array.fold result ~init:0.0 ~f:( +. ) /. Float.of_int (Array.length result) in
  let variance =
    Array.fold result ~init:0.0 ~f:(fun acc x -> acc +. ((x -. mean) *. (x -. mean)))
    /. Float.of_int (Array.length result)
  in
  let std_dev = Float.sqrt variance in

  printf "\nStatistics:\n";
  printf "  Mean: %.4f (expected: ~0.5)\n" mean;
  printf "  Std Dev: %.4f (expected: ~%.4f)\n" std_dev (Float.sqrt (1.0 /. 12.0));
  printf "  Min: %.4f\n" (Array.min_elt result ~compare:Float.compare |> Option.value ~default:0.0);
  printf "  Max: %.4f\n" (Array.max_elt result ~compare:Float.compare |> Option.value ~default:0.0);

  (* Check uniformity with chi-square test *)
  let expected_per_bin = Float.of_int (Array.length result) /. Float.of_int num_bins in
  let chi_square =
    Array.fold bins ~init:0.0 ~f:(fun acc observed ->
        let diff = Float.of_int observed -. expected_per_bin in
        acc +. (diff *. diff /. expected_per_bin))
  in
  printf "  Chi-square statistic: %.2f (df=%d, critical value at 0.05: ~%.2f)\n" chi_square
    (num_bins - 1) 30.14;

  (* Check if all values are in range *)
  let all_in_range = Array.for_all result ~f:(fun x -> Float.(x >= 0.0 && x < 1.0)) in
  printf "  All values in [0, 1) range: %b\n" all_in_range

(** Test normal_at1 which works pointwise (one output per uint4x32 input).

    NOTE: normal_at internally uses box_muller which creates TWO uniform random tensors. The non-1
    variants have shape constraints from uint4x32. Use normal_at1 which works pointwise, combined
    with a target tensor to drive shape inference. *)
let test_normal_at_with_shape () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let module O = TDSL.O in
  (* Scalar counter for randomness bifurcation *)
  let counter = NTDSL.number 123.0 in

  (* Use normal_at1 (pointwise) with shape from target tensor *)
  let num_values = 10000 in
  let target = TDSL.range num_values in
  let%op normal_values = O.normal_at1 counter + (target *. 0.0) in
  Ir.Tnode.update_prec normal_values.value Ir.Ops.single;

  (* Compile and run *)
  Ocannl.Train.set_hosted normal_values.value;
  ignore (Ocannl.Train.forward_once ctx normal_values);
  let result = Ir.Tnode.get_values normal_values.value in

  (* Calculate statistics *)
  let n = Array.length result in
  let mean = Array.fold result ~init:0.0 ~f:( +. ) /. Float.of_int n in
  let variance =
    Array.fold result ~init:0.0 ~f:(fun acc x -> acc +. ((x -. mean) *. (x -. mean)))
    /. Float.of_int n
  in
  let std_dev = Float.sqrt variance in
  let min_val = Array.min_elt result ~compare:Float.compare |> Option.value ~default:0.0 in
  let max_val = Array.max_elt result ~compare:Float.compare |> Option.value ~default:0.0 in

  (* Check what percentage falls within standard deviations *)
  let within_1_std = Array.count result ~f:(fun x -> Float.(abs x <= 1.0)) in
  let within_2_std = Array.count result ~f:(fun x -> Float.(abs x <= 2.0)) in
  let within_3_std = Array.count result ~f:(fun x -> Float.(abs x <= 3.0)) in

  let pct_1_std = Float.of_int within_1_std /. Float.of_int n *. 100.0 in
  let pct_2_std = Float.of_int within_2_std /. Float.of_int n *. 100.0 in
  let pct_3_std = Float.of_int within_3_std /. Float.of_int n *. 100.0 in

  (* Normality test using skewness and kurtosis *)
  let skewness =
    let sum_cubed =
      Array.fold result ~init:0.0 ~f:(fun acc x ->
          let diff = x -. mean in
          acc +. (diff *. diff *. diff))
    in
    sum_cubed /. (Float.of_int n *. std_dev *. std_dev *. std_dev)
  in

  let kurtosis =
    let sum_fourth =
      Array.fold result ~init:0.0 ~f:(fun acc x ->
          let diff = x -. mean in
          let diff2 = diff *. diff in
          acc +. (diff2 *. diff2))
    in
    (sum_fourth /. (Float.of_int n *. std_dev *. std_dev *. std_dev *. std_dev)) -. 3.0
  in

  printf "\nNormal Distribution N(0,1) Statistical Test\n";
  printf "============================================\n";
  printf "Generated %d values with scalar counter\n" n;

  (* Verify statistical properties - only print PASS/FAIL to avoid machine-specific output *)
  let check name value expected tolerance =
    let passed = Float.(abs (value -. expected) <= tolerance) in
    printf "  %s (expected: ~%.1f, tolerance: %.2f): %s\n" name expected tolerance
      (if passed then "PASS" else Printf.sprintf "FAIL (got %.4f)" value);
    passed
  in

  let check_bound name value bound is_lower =
    let passed = if is_lower then Float.(value < bound) else Float.(value > bound) in
    let op = if is_lower then "<" else ">" in
    printf "  %s (should be %s %.1f): %s\n" name op bound
      (if passed then "PASS" else Printf.sprintf "FAIL (got %.4f)" value);
    passed
  in

  let all_passed =
    check "Mean" mean 0.0 0.1 && check "Std Dev" std_dev 1.0 0.1
    && check "Within 1 std dev %%" pct_1_std 68.3 3.0
    && check "Within 2 std dev %%" pct_2_std 95.4 2.0
    && check "Within 3 std dev %%" pct_3_std 99.7 1.0
    && check "Skewness" skewness 0.0 0.15
    && check "Excess Kurtosis" kurtosis 0.0 0.15
    && check_bound "Min" min_val (-3.0) true
    && check_bound "Max" max_val 3.0 false
  in

  printf "\nOverall: %s\n" (if all_passed then "ALL TESTS PASSED" else "SOME TESTS FAILED")

(** Test that different counter values produce different random sequences. This demonstrates the
    counter's purpose: bifurcating randomness. *)
let test_counter_bifurcation () =
  printf "\nCounter Bifurcation Test\n";
  printf "========================\n";
  printf "Testing that different counter values produce different random streams\n\n";

  (* Use a small target shape for easy comparison *)
  let num_values = 100 in

  let get_values counter_val =
    Tensor.unsafe_reinitialize ();
    let ctx = Context.auto () in
    let module O = TDSL.O in
    let counter = NTDSL.number (Float.of_int counter_val) in
    let target = TDSL.range num_values in
    let%op uniform_values = O.uniform_at counter + (target *. 0.0) in
    Ir.Tnode.update_prec uniform_values.value Ir.Ops.single;
    Ocannl.Train.set_hosted uniform_values.value;
    ignore (Ocannl.Train.forward_once ctx uniform_values);
    Ir.Tnode.get_values uniform_values.value
  in

  let values_0 = get_values 0 in
  let values_1 = get_values 1 in
  let values_0_again = get_values 0 in

  (* Check that counter=0 and counter=1 produce different values *)
  let diff_count = ref 0 in
  for i = 0 to num_values - 1 do
    if Float.(abs (values_0.(i) -. values_1.(i)) > 1e-6) then Int.incr diff_count
  done;
  printf "Counter 0 vs Counter 1: %d/%d values differ (expected: ~100%%)\n" !diff_count num_values;

  (* Check that same counter produces same values (deterministic) *)
  let same_count = ref 0 in
  for i = 0 to num_values - 1 do
    if Float.(abs (values_0.(i) -. values_0_again.(i)) < 1e-6) then Int.incr same_count
  done;
  printf "Counter 0 vs Counter 0 (repeat): %d/%d values same (expected: 100%%)\n" !same_count
    num_values;

  if !diff_count > 90 && !same_count = num_values then printf "\nBifurcation test: PASS\n"
  else printf "\nBifurcation test: FAIL\n"

(** Test kaiming_at with proper shape structure. The result tensor needs input dimensions for
    kaiming to extract fan_in.

    This test demonstrates specifying dimensions explicitly via TDSL (not TDSL.O). The counter is
    scalar (for randomness bifurcation), and output shape is given directly to uniform_at via
    ~input_dims and ~output_dims. *)
let test_kaiming_at_with_proper_shape () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in

  (* Define weight matrix dimensions *)
  let fan_in = 100 in
  let fan_out = 40 in

  (* Scalar counter for randomness bifurcation *)
  let counter = NTDSL.number 45.0 in

  (* Use TDSL.uniform_at (not TDSL.O.uniform_at) to specify dimensions explicitly. This is an
     alternative to shape inference from a target tensor. *)
  let kaiming_values =
    TDSL.kaiming_at ~input_dims:[ fan_in ] ~output_dims:[ fan_out ] TDSL.O.uniform_at counter ()
  in
  Ir.Tnode.update_prec kaiming_values.value Ir.Ops.single;

  (* Compile and run *)
  Ocannl.Train.set_hosted kaiming_values.value;
  ignore (Ocannl.Train.forward_once ctx kaiming_values);
  let result = Ir.Tnode.get_values kaiming_values.value in

  (* Expected: uniform [0,1) scaled by sqrt(6/fan_in) = sqrt(6/100) ≈ 0.245 So values should be in
     [0, 0.245) with mean ≈ 0.122 *)
  let expected_scale = Float.sqrt (6.0 /. Float.of_int fan_in) in

  printf "Kaiming Initialization Test (fan_in=%d, fan_out=%d)\n" fan_in fan_out;
  printf "====================================================\n";
  printf "Generated %d values (shape [%d; %d])\n" (Array.length result) fan_out fan_in;
  printf "Expected scale: sqrt(6/%d) = %.4f\n" fan_in expected_scale;

  (* Calculate statistics *)
  let n = Array.length result in
  let mean = Array.fold result ~init:0.0 ~f:( +. ) /. Float.of_int n in
  let variance =
    Array.fold result ~init:0.0 ~f:(fun acc x -> acc +. ((x -. mean) *. (x -. mean)))
    /. Float.of_int n
  in
  let std_dev = Float.sqrt variance in
  let min_val = Array.min_elt result ~compare:Float.compare |> Option.value ~default:0.0 in
  let max_val = Array.max_elt result ~compare:Float.compare |> Option.value ~default:0.0 in

  printf "  Mean: %.4f (expected: ~%.4f)\n" mean (expected_scale /. 2.0);
  printf "  Std Dev: %.4f\n" std_dev;
  printf "  Min: %.4f\n" min_val;
  printf "  Max: %.4f (expected: <%.4f)\n" max_val expected_scale;

  (* Create and print histogram *)
  let num_bins = 20 in
  let bins =
    create_histogram result ~num_bins ~min_val:(min_val -. 0.01) ~max_val:(max_val +. 0.01)
  in
  print_histogram bins ~title:"Kaiming Distribution Histogram" ~max_width:40

(** Test xavier_at with proper shape structure. Xavier needs both input and output dimensions for
    scaling.

    Similar to kaiming test, uses TDSL with explicit dimensions. *)
let test_xavier_at_with_proper_shape () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in

  (* Define weight matrix dimensions *)
  let fan_in = 100 in
  let fan_out = 40 in

  (* Scalar counter for randomness bifurcation *)
  let counter = NTDSL.number 43.0 in

  (* Use TDSL.uniform_at with explicit dimensions *)
  let xavier_values =
    TDSL.xavier_at ~input_dims:[ fan_in ] ~output_dims:[ fan_out ] TDSL.O.uniform_at counter ()
  in
  Ir.Tnode.update_prec xavier_values.value Ir.Ops.single;

  (* Compile and run *)
  Ocannl.Train.set_hosted xavier_values.value;
  ignore (Ocannl.Train.forward_once ctx xavier_values);
  let result = Ir.Tnode.get_values xavier_values.value in

  (* Expected: uniform [0,1) scaled by sqrt(6/(fan_in + fan_out)) = sqrt(6/150) ≈ 0.2 *)
  let expected_scale = Float.sqrt (6.0 /. Float.of_int (fan_in + fan_out)) in

  printf "Xavier Initialization Test (fan_in=%d, fan_out=%d)\n" fan_in fan_out;
  printf "===================================================\n";
  printf "Generated %d values (shape [%d; %d])\n" (Array.length result) fan_out fan_in;
  printf "Expected scale: sqrt(6/%d) = %.4f\n" (fan_in + fan_out) expected_scale;

  (* Calculate statistics *)
  let n = Array.length result in
  let mean = Array.fold result ~init:0.0 ~f:( +. ) /. Float.of_int n in
  let variance =
    Array.fold result ~init:0.0 ~f:(fun acc x -> acc +. ((x -. mean) *. (x -. mean)))
    /. Float.of_int n
  in
  let std_dev = Float.sqrt variance in
  let min_val = Array.min_elt result ~compare:Float.compare |> Option.value ~default:0.0 in
  let max_val = Array.max_elt result ~compare:Float.compare |> Option.value ~default:0.0 in

  printf "  Mean: %.4f (expected: ~%.4f)\n" mean (expected_scale /. 2.0);
  printf "  Std Dev: %.4f\n" std_dev;
  printf "  Min: %.4f\n" min_val;
  printf "  Max: %.4f (expected: <%.4f)\n" max_val expected_scale;

  (* Create and print histogram *)
  let num_bins = 20 in
  let bins =
    create_histogram result ~num_bins ~min_val:(min_val -. 0.01) ~max_val:(max_val +. 0.01)
  in
  print_histogram bins ~title:"Xavier Distribution Histogram" ~max_width:40

let () =
  test_uniform_at_with_shape ();
  printf "\n";
  test_normal_at_with_shape ();
  printf "\n";
  test_counter_bifurcation ();
  printf "\n";
  test_kaiming_at_with_proper_shape ();
  printf "\n";
  test_xavier_at_with_proper_shape ()
