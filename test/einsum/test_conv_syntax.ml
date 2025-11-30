open Base
open Stdio

let test_conv_parsing () =
  printf "Testing conv syntax parsing...\n%!";

  (* Test 1: Basic conv expression with coefficients (multichar - requires commas) *)
  let spec1 = "2*o+3*k" in
  let labels1 = Einsum_parser.axis_labels_of_spec spec1 in
  printf "Test 1: Parsed '%s' successfully\n%!" spec1;
  printf "  Structure: %s\n\n%!" (Sexp.to_string_hum (Einsum_parser.sexp_of_parsed_axis_labels labels1));

  (* Test 2: Simple conv expression without coefficients (multichar - requires commas) *)
  let spec2 = "o+k" in
  let labels2 = Einsum_parser.axis_labels_of_spec spec2 in
  printf "Test 2: Parsed '%s' successfully\n%!" spec2;
  printf "  Structure: %s\n\n%!" (Sexp.to_string_hum (Einsum_parser.sexp_of_parsed_axis_labels labels2));

  (* Test 3: Mixed spec with comma (multichar mode) *)
  let spec3 = "a, 2*b+c" in
  let labels3 = Einsum_parser.axis_labels_of_spec spec3 in
  printf "Test 3: Parsed '%s' successfully\n%!" spec3;
  printf "  Structure: %s\n\n%!" (Sexp.to_string_hum (Einsum_parser.sexp_of_parsed_axis_labels labels3));

  (* Test 4: Conv expression with multiple identifiers (multichar - requires commas) *)
  let spec4 = "i, o+k, j" in
  let labels4 = Einsum_parser.axis_labels_of_spec spec4 in
  printf "Test 4: Parsed '%s' successfully (multichar mode)\n%!" spec4;
  printf "  Structure: %s\n\n%!" (Sexp.to_string_hum (Einsum_parser.sexp_of_parsed_axis_labels labels4));

  (* Test 5: Conv expression with multi-char identifiers (multichar) *)
  let spec5 = "a+bc" in
  let labels5 = Einsum_parser.axis_labels_of_spec spec5 in
  printf "Test 5: Parsed '%s' successfully (multichar mode)\n%!" spec5;
  printf "  Structure: %s\n\n%!" (Sexp.to_string_hum (Einsum_parser.sexp_of_parsed_axis_labels labels5));

  (* Test 6: Test in einsum notation with multichar conv *)
  let spec6 = "i, j -> 2*i+j" in
  let labels6 = Einsum_parser.axis_labels_of_spec spec6 in
  printf "Test 6: Parsed '%s' successfully\n%!" spec6;
  printf "  Structure: %s\n\n%!" (Sexp.to_string_hum (Einsum_parser.sexp_of_parsed_axis_labels labels6));

  (* Test 7: Complex batch-input-output spec with conv *)
  let spec7 = "batch|input->3*output+1*kernel," in
  let labels7 = Einsum_parser.axis_labels_of_spec spec7 in
  printf "Test 7: Parsed '%s' successfully\n%!" spec7;
  printf "  Structure: %s\n\n%!" (Sexp.to_string_hum (Einsum_parser.sexp_of_parsed_axis_labels labels7));

  printf "All conv syntax parsing tests passed!\n%!"

let test_strided_iteration_parsing () =
  printf "\nTesting strided iteration syntax parsing...\n%!";

  (* Test 1: Basic strided iteration (multichar mode due to multiplication) *)
  let spec1 = "2*output" in
  let labels1 = Einsum_parser.axis_labels_of_spec spec1 in
  printf "Test 1: Parsed strided iteration '%s' successfully\n%!" spec1;
  printf "  Structure: %s\n\n%!" (Sexp.to_string_hum (Einsum_parser.sexp_of_parsed_axis_labels labels1));

  (* Test 2: Strided iteration with single-char identifier (multichar mode) *)
  let spec2 = "3*i" in
  let labels2 = Einsum_parser.axis_labels_of_spec spec2 in
  printf "Test 2: Parsed strided iteration '%s' successfully\n%!" spec2;
  printf "  Structure: %s\n\n%!" (Sexp.to_string_hum (Einsum_parser.sexp_of_parsed_axis_labels labels2));

  (* Test 3: Strided iteration in einsum context (multichar due to multiplication) *)
  let spec3 = "input -> 2*output" in
  let labels3 = Einsum_parser.axis_labels_of_spec spec3 in
  printf "Test 3: Parsed einsum with strided iteration '%s' successfully\n%!" spec3;
  printf "  Structure: %s\n\n%!" (Sexp.to_string_hum (Einsum_parser.sexp_of_parsed_axis_labels labels3));

  (* Test 4: Mixed regular labels and strided iteration (multichar due to comma) *)
  let spec4 = "regular, 3*strided" in
  let labels4 = Einsum_parser.axis_labels_of_spec spec4 in
  printf "Test 4: Parsed mixed labels with strided iteration '%s' successfully\n%!" spec4;
  printf "  Structure: %s\n\n%!" (Sexp.to_string_hum (Einsum_parser.sexp_of_parsed_axis_labels labels4));

  printf "\nAll strided iteration parsing tests completed!\n%!"

let test_conv_multichar_detection () =
  printf "\nTesting multichar mode detection...\n%!";

  (* These should trigger multichar mode *)
  let multichar_specs =
    [ "a,b"; (* comma *) "2*o+k"; (* multiplication *) "o+k"; (* addition *) "a,2*b+c" (* combo *) ]
  in

  List.iter multichar_specs ~f:(fun spec ->
      try
        let _labels = Einsum_parser.axis_labels_of_spec spec in
        printf "✓ Multichar spec '%s' parsed correctly\n%!" spec
      with exn -> printf "✗ Failed to parse multichar spec '%s': %s\n%!" spec (Exn.to_string exn));

  (* These should work in single-char mode (no multiplication, plus, caret, ampersand, or commas) *)
  let singlechar_specs =
    [
      "abc";
      (* single chars *)
      "ijk";
      (* single chars *)
      "i->j";
      (* single chars with arrow *)
      "...|ij";
      (* ellipsis in batch axes *)
      "j...";
      (* matching on beginning axes *)
      "...|j...->i";
      (* ellipsis, matching on beginning axes *)
      "...|i->1";
      (* ellipsis, fixed index *)
    ]
  in

  List.iter singlechar_specs ~f:(fun spec ->
      try
        let _labels = Einsum_parser.axis_labels_of_spec spec in
        printf "✓ Single-char spec '%s' parsed correctly\n%!" spec
      with exn ->
        printf "✗ Failed to parse single-char spec '%s': %s\n%!" spec (Exn.to_string exn))

let test_single_char_conv_equivalence () =
  printf "\nTesting conv spec parsing...\n%!";

  (* Conv expressions now always trigger multichar mode due to plus or multiplication *)
  let conv_spec = "a+b" in

  let labels = Einsum_parser.axis_labels_of_spec conv_spec in

  printf "Conv spec '%s': %s\n%!" conv_spec
    (Sexp.to_string_hum (Einsum_parser.sexp_of_parsed_axis_labels labels));

  printf "Note: Conv expressions with + or * now always use multichar mode\n%!"

let () =
  test_conv_parsing ();
  test_strided_iteration_parsing ();
  test_conv_multichar_detection ();
  test_single_char_conv_equivalence ();
  printf "\nAll conv syntax tests completed!\n%!"
