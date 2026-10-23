open Base
open Ocannl.Nn_blocks.DSL_modules

(* gh-509: packed [uniform] is total over shapes. The result's element count no longer needs to be a
   multiple of the 128-bit block width (16 / bytes-per-element): shape inference rounds the counter
   extent up ([Row.Strided_var] with [round_up]) and lowering peels the last counter iteration into
   a shorter [Set_from_vec] store. Pins: - odd sizes across precisions produce exactly [n] values,
   all in [0, 1) -- fp8 included since gh-ocannl-1486, whose draw used to be raw e5m2 bit patterns
   (negatives, infinities, NaNs); - the reference draw is not one repeated value; - prefix
   stability: the first [n] values are bitwise identical to those of a larger tensor built
   identically (after [Tensor.unsafe_reinitialize] the ids, and thus the threefry keys, line up); -
   the printed values are golden: they enforce parity of the packed stream across backends. *)

let run_uniform ~prec ?input_dims output_dims =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let t = TDSL.uniform () ?input_dims ~output_dims () in
  Ir.Tnode.update_prec t.value prec;
  Ocannl.Train.set_materialized t.value;
  let ctx = Ocannl.Train.forward_once ctx t in
  Context.get_values ctx t.value

let print_prefix values k =
  for i = 0 to Int.min (k - 1) (Array.length values - 1) do
    Stdio.printf " %s" (Ir.Ndarray.concise_float ~prec:4 values.(i))
  done;
  Stdio.printf "\n"

let test_prec ~name prec ns =
  Verdict.case name @@ fun () ->
  let lanes = 16 / Ir.Ops.prec_in_bytes prec in
  Stdio.printf "=== %s: %d lanes per block ===\n" name lanes;
  let max_n = List.fold ns ~init:0 ~f:Int.max in
  let full = (max_n + lanes - 1) / lanes * lanes in
  let reference = run_uniform ~prec [ full ] in
  Stdio.printf "reference (size %d) first values:" full;
  print_prefix reference 5;
  (* A conversion that collapsed every draw to one value (0, say) would pass the range claims. *)
  Verdict.p_exists (name ^ ": the reference draws at least two distinct values")
    (Array.to_list reference) ~f:(fun x -> not (Float.equal x reference.(0)));
  List.iter ns ~f:(fun n ->
      let vs = run_uniform ~prec [ n ] in
      let count_ok = Array.length vs = n in
      let range_holds =
        (not (Array.is_empty vs)) && Array.for_all vs ~f:(fun x -> Float.(x >= 0.0 && x < 1.0))
      in
      let prefix_stable =
        count_ok
        && (not (Array.is_empty vs))
        && Array.for_alli vs ~f:(fun i x -> Float.equal x reference.(i))
      in
      (* A three-property census row whose shape is the point, so the claims sit beside it on the
         same [let]-bound booleans, each naming the precision and the size it is about. *)
      Stdio.printf "n=%d: count %b, range ok %b, prefix-stable %b\n" n count_ok range_holds
        prefix_stable;
      Verdict.claimf "%s n=%d: exactly n values" name n count_ok;
      Verdict.claimf "%s n=%d: all values in [0,1)" name n range_holds;
      Verdict.claimf "%s n=%d: prefix bitwise-stable against the reference" name n prefix_stable)

(* A multi-axis shape with a non-divisible total: the value stream depends only on the flat element
   index, so the flattened tensor must still be a prefix of the 1-D reference. *)
let test_multi_axis () =
  Stdio.printf "=== single precision, multi-axis 5->3 (15 elements) ===\n";
  let reference = run_uniform ~prec:Ir.Ops.single [ 16 ] in
  let vs = run_uniform ~prec:Ir.Ops.single ~input_dims:[ 5 ] [ 3 ] in
  Stdio.printf "got %d elements\n" (Array.length vs);
  let count_ok = Array.length vs = 15 in
  let in_range =
    (not (Array.is_empty vs)) && Array.for_all vs ~f:(fun x -> Float.(x >= 0.0 && x < 1.0))
  in
  let prefix_stable = count_ok && Array.for_alli vs ~f:(fun i x -> Float.equal x reference.(i)) in
  Stdio.printf "count %b, in [0,1) %b, prefix-stable %b\n" count_ok in_range prefix_stable;
  Verdict.claim "multi-axis 5->3: exactly 15 values" count_ok;
  Verdict.claim "multi-axis 5->3: all values in [0,1)" in_range;
  Verdict.claim "multi-axis 5->3: prefix bitwise-stable against the reference" prefix_stable;
  Stdio.printf "values:";
  print_prefix vs 15

let () =
  test_prec ~name:"single" Ir.Ops.single [ 1; 2; 3; 5; 7; 10 ];
  test_prec ~name:"half" Ir.Ops.half [ 1; 3; 9; 12 ];
  test_prec ~name:"double" Ir.Ops.double [ 1; 3 ];
  (* fp8 is the stress case: 16 lanes per block. (The test is disabled on Metal for the double case
     above, not for this one.) *)
  test_prec ~name:"fp8" Ir.Ops.fp8 [ 1; 7; 15; 17; 33 ];
  Verdict.case "multi-axis" test_multi_axis
