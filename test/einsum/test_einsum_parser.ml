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
  printf "  'b|i->o' -> batch:%d input:%d output:%d\n" (List.length labels2.given_batch)
    (List.length labels2.given_input)
    (List.length labels2.given_output);

  (* Test 3: Einsum spec *)
  let spec3 = "ij;jk=>ik" in
  let rhs_list, l3 = Einsum_parser.einsum_of_spec spec3 in
  (match rhs_list with
  | [ l1; l2 ] ->
      printf "  'ij;jk=>ik' -> (%d,%d);(%d,%d)=>(%d,%d)\n" (List.length l1.given_input)
        (List.length l1.given_output) (List.length l2.given_input) (List.length l2.given_output)
        (List.length l3.given_input) (List.length l3.given_output)
  | _ -> printf "  'ij;jk=>ik' -> unexpected number of RHSes: %d\n" (List.length rhs_list));

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

let test_row_variable_spellings () =
  printf "Testing row variable spellings:\n";
  let row_var = Option.value ~default:"<none>" in
  let show_labels spec =
    match Einsum_parser.axis_labels_of_spec spec with
    | labels ->
        printf "  '%s' -> batch row: %s, output row: %s\n" spec (row_var labels.bcast_batch)
          (row_var labels.bcast_output)
    | exception Einsum_parser.Parse_error msg -> printf "  '%s' -> %s\n" spec msg
  in
  let show_einsum spec =
    match Einsum_parser.einsum_of_spec spec with
    | rhses, result ->
        printf "  '%s' -> %d RHSes, result batch row: %s\n" spec (List.length rhses)
          (row_var result.bcast_batch)
    | exception Einsum_parser.Parse_error msg -> printf "  '%s' -> %s\n" spec msg
  in
  (* The named row variable, and its misspelling with the context ellipsis' three dots. *)
  show_labels "..batch.., seq | heads";
  show_labels "...batch.., seq | heads";
  show_labels "..b..|x";
  show_labels "...b..|x";
  show_labels "... batch , .., x";
  (* The attention spec of the shapes slides, both ways (lukstafi/ocannl-staging#1061). *)
  show_einsum
    "..batch.., seq | heads, ..dims..; ..batch.., time | heads, ..dims.. => ..batch.., seq | time \
     -> heads";
  show_einsum
    "...batch.., seq | heads, ..dims..; ...batch.., time | heads, ..dims.. => ...batch.., seq | \
     time -> heads";
  (* The context ellipsis itself, and an unrelated error, carry no hint. *)
  show_labels "...|...->x";
  show_labels "a|b|c";
  printf "\n"

let () =
  test_mode_detection ();
  test_single_char ();
  test_multichar ();
  test_row_variable_spellings ();
  printf "All tests passed!\n"
