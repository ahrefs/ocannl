open Base
open Ocannl.Nn_blocks.DSL_modules
open Stdio

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

let test_uniform_at_histogram () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let module O = TDSL.O in
  (* Generate a large batch of random numbers using uniform_at *)
  (* Note: uniform_at produces 4 values per counter input (from uint4x32) *)
  let num_counters = 2500 in
  let counter = TDSL.range num_counters in

  (* Generate uniform random values using uniform_at *)
  let uniform_values = O.uniform_at counter in
  Ir.Tnode.update_prec uniform_values.value Ir.Ops.single;

  (* Compile and run *)
  Ocannl.Train.set_hosted uniform_values.value;
  ignore (Ocannl.Train.forward_once ctx uniform_values);
  let result = Ir.Tnode.get_values uniform_values.value in

  printf "Generated %d values from %d counters (%.1fx expansion)\n" (Array.length result)
    num_counters
    (Float.of_int (Array.length result) /. Float.of_int num_counters);

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

let test_normal_at_histogram () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let module O = TDSL.O in
  (* Generate a large batch of random numbers using normal_at *)
  (* Note: normal_at also produces 4 values per counter input *)
  let num_counters = 2500 in
  let counter = TDSL.range num_counters in

  (* Generate normal random values using normal_at *)
  let normal_values = O.normal_at counter in
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

  (* Note: Box-Muller transformation uses transcendental functions (log, cos) which may
     produce slightly different results across different CPU architectures and math libraries.
     We only verify statistical properties are within acceptable bounds, not exact values. *)
  printf "\nNormal Distribution N(0,1) Statistical Test\n";
  printf "============================================\n";
  printf "Generated %d values\n" n;

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
    check "Mean" mean 0.0 0.1
    && check "Std Dev" std_dev 1.0 0.1
    && check "Within 1 std dev %%" pct_1_std 68.3 3.0
    && check "Within 2 std dev %%" pct_2_std 95.4 2.0
    && check "Within 3 std dev %%" pct_3_std 99.7 1.0
    && check "Skewness" skewness 0.0 0.15
    && check "Excess Kurtosis" kurtosis 0.0 0.15
    && check_bound "Min" min_val (-3.0) true
    && check_bound "Max" max_val 3.0 false
  in

  printf "\nOverall: %s\n" (if all_passed then "ALL TESTS PASSED" else "SOME TESTS FAILED")

let test_batched_generation_consistency () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let module O = TDSL.O in
  (* Test that batched generation gives consistent results *)
  let batch_size = 100 in
  let num_batches = 10 in

  printf "\nBatched Generation Consistency Test\n";
  printf "====================================\n";

  (* Generate values in batches and check they don't repeat across batches *)
  let all_uniform_values = ref [||] in
  let all_normal_values = ref [||] in

  for _batch = 0 to num_batches - 1 do
    (* Each batch uses its own counter range - values are just seeds *)
    let counter = TDSL.range batch_size in

    (* Generate uniform batch *)
    let uniform_batch = O.uniform_at counter in
    Ir.Tnode.update_prec uniform_batch.value Ir.Ops.single;
    Ocannl.Train.set_hosted uniform_batch.value;
    ignore (Ocannl.Train.forward_once ctx uniform_batch);
    let uniform_result = Ir.Tnode.get_values uniform_batch.value in
    all_uniform_values := Array.append !all_uniform_values uniform_result;

    (* Generate normal batch *)
    let normal_batch = O.normal_at counter in
    Ir.Tnode.update_prec normal_batch.value Ir.Ops.single;
    Ocannl.Train.set_hosted normal_batch.value;
    ignore (Ocannl.Train.forward_once ctx normal_batch);
    let normal_result = Ir.Tnode.get_values normal_batch.value in
    all_normal_values := Array.append !all_normal_values normal_result
  done;

  (* Check for uniqueness (with small tolerance for floating point) *)
  let count_unique arr =
    let sorted = Array.copy arr in
    Array.sort sorted ~compare:Float.compare;
    let unique = ref 1 in
    for i = 1 to Array.length sorted - 1 do
      let diff = Float.abs (sorted.(i) -. sorted.(i - 1)) in
      if Float.(diff > 1e-7) then unique := !unique + 1
    done;
    !unique
  in

  let total_values = batch_size * num_batches in
  let unique_uniform = count_unique !all_uniform_values in
  let unique_normal = count_unique !all_normal_values in

  printf "Generated %d values in %d batches of %d\n" total_values num_batches batch_size;
  printf "Uniform values: %d unique out of %d (%.1f%%)\n" unique_uniform total_values
    (Float.of_int unique_uniform /. Float.of_int total_values *. 100.0);
  printf "Normal values: %d unique out of %d (%.1f%%)\n" unique_normal total_values
    (Float.of_int unique_normal /. Float.of_int total_values *. 100.0);

  (* Verify batch consistency of statistical properties *)
  let batch_means_uniform = Array.create ~len:num_batches 0.0 in
  let batch_means_normal = Array.create ~len:num_batches 0.0 in

  for batch = 0 to num_batches - 1 do
    let start_idx = batch * batch_size in
    let uniform_batch = Array.sub !all_uniform_values ~pos:start_idx ~len:batch_size in
    let normal_batch = Array.sub !all_normal_values ~pos:start_idx ~len:batch_size in

    batch_means_uniform.(batch) <-
      Array.fold uniform_batch ~init:0.0 ~f:( +. ) /. Float.of_int batch_size;
    batch_means_normal.(batch) <-
      Array.fold normal_batch ~init:0.0 ~f:( +. ) /. Float.of_int batch_size
  done;

  let mean_of_means_uniform =
    Array.fold batch_means_uniform ~init:0.0 ~f:( +. ) /. Float.of_int num_batches
  in
  let mean_of_means_normal =
    Array.fold batch_means_normal ~init:0.0 ~f:( +. ) /. Float.of_int num_batches
  in

  let std_of_means_uniform =
    let diff_sum =
      Array.fold batch_means_uniform ~init:0.0 ~f:(fun acc x ->
          let diff = x -. mean_of_means_uniform in
          acc +. (diff *. diff))
    in
    Float.sqrt (diff_sum /. Float.of_int num_batches)
  in
  let std_of_means_normal =
    let diff_sum =
      Array.fold batch_means_normal ~init:0.0 ~f:(fun acc x ->
          let diff = x -. mean_of_means_normal in
          acc +. (diff *. diff))
    in
    Float.sqrt (diff_sum /. Float.of_int num_batches)
  in

  printf "\nBatch means consistency:\n";
  printf "  Uniform: mean of batch means = %.4f, std = %.4f\n" mean_of_means_uniform
    std_of_means_uniform;
  printf "  Normal: mean of batch means = %.4f, std = %.4f\n" mean_of_means_normal
    std_of_means_normal

let () =
  test_uniform_at_histogram ();
  printf "\n";
  test_normal_at_histogram ();
  printf "\n";
  test_batched_generation_consistency ()
