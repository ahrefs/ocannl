open Base
open Stdio

let test_single_char () =
  printf "Testing single-char mode:\n";

  (* Test 1: Simple single-char *)
  let spec1 = "abc" in
  let labels1 = Einsum_parser.axis_labels_of_spec spec1 in
  printf "  'abc' -> %d output axes\n" (List.length labels1.given_output);

  (* Test 2: With batch and input *)
  let spec2 = "b|i->o" in
  let labels2 = Einsum_parser.axis_labels_of_spec spec2 in
  printf "  'b|i->o' -> batch:%d input:%d output:%d\n"
    (List.length labels2.given_batch)
    (List.length labels2.given_input)
    (List.length labels2.given_output);

  (* Test 3: Einsum spec *)
  let spec3 = "ij;jk=>ik" in
  let l1, l2_opt, l3 = Einsum_parser.einsum_of_spec spec3 in
  let l2 = Option.value_exn l2_opt in
  printf "  'ij;jk=>ik' -> (%d,%d);(%d,%d)=>(%d,%d)\n"
    (List.length l1.given_input)
    (List.length l1.given_output)
    (List.length l2.given_input)
    (List.length l2.given_output)
    (List.length l3.given_input)
    (List.length l3.given_output);

  printf "\n"

let test_multichar () =
  printf "Testing multichar mode:\n";

  (* Test 1: Comma-separated *)
  let spec1 = "a, b, c" in
  let labels1 = Einsum_parser.axis_labels_of_spec spec1 in
  printf "  'a, b, c' -> %d output axes\n" (List.length labels1.given_output);

  (* Test 2: Trailing comma *)
  let spec2 = "a, b," in
  let labels2 = Einsum_parser.axis_labels_of_spec spec2 in
  printf "  'a, b,' -> %d output axes\n" (List.length labels2.given_output);

  (* Test 3: Conv expression *)
  let spec3 = "2*o+k" in
  let labels3 = Einsum_parser.axis_labels_of_spec spec3 in
  printf "  '2*o+k' -> %d output axes\n" (List.length labels3.given_output);

  (* Test 4: Mixed conv with regular *)
  let spec4 = "2*o+3*k, x" in
  let labels4 = Einsum_parser.axis_labels_of_spec spec4 in
  printf "  '2*o+3*k, x' -> %d output axes\n" (List.length labels4.given_output);

  printf "\n"

let test_mode_detection () =
  printf "Testing mode detection:\n";

  let test_spec spec expected_mode =
    let is_multi = Einsum_parser.is_multichar spec in
    let mode_str = if is_multi then "multichar" else "single-char" in
    let expected_str = if expected_mode then "multichar" else "single-char" in
    let status = if Bool.equal is_multi expected_mode then "✓" else "✗" in
    printf "  %s '%s' -> %s (expected %s)\n" status spec mode_str expected_str
  in

  test_spec "abc" false;
  test_spec "a,b,c" true;
  test_spec "2*a+b" true;
  test_spec "a+b" true;
  test_spec "a*b" true;
  test_spec "a^b" true;
  test_spec "a&b" true;
  test_spec "a|b->c" false;
  test_spec "...a..b" false;

  printf "\n"

let () =
  test_mode_detection ();
  test_single_char ();
  test_multichar ();
  printf "All tests passed!\n"
