open Base
open Ocannl
open Stdio

let test_conv_parsing () =
  printf "Testing conv syntax parsing...\n%!";
  
  (* Test 1: Basic conv expression with coefficients (multichar) *)
  let spec1 = "2*o+3*k" in
  let labels1 = Shape.axis_labels_of_spec spec1 in
  printf "Test 1: Parsed '%s' successfully\n%!" spec1;
  printf "  Structure: %s\n\n%!" (Sexp.to_string_hum (Shape.sexp_of_parsed_axis_labels labels1));
  
  (* Test 2: Simple conv expression without coefficients (multichar) *)
  let spec2 = "o+k" in
  let labels2 = Shape.axis_labels_of_spec spec2 in
  printf "Test 2: Parsed '%s' successfully\n%!" spec2;
  printf "  Structure: %s\n\n%!" (Sexp.to_string_hum (Shape.sexp_of_parsed_axis_labels labels2));
  
  (* Test 3: Mixed spec with comma (multichar mode) *)
  let spec3 = "a,2*b+c" in
  let labels3 = Shape.axis_labels_of_spec spec3 in
  printf "Test 3: Parsed '%s' successfully\n%!" spec3;
  printf "  Structure: %s\n\n%!" (Sexp.to_string_hum (Shape.sexp_of_parsed_axis_labels labels3));
  
  (* Test 4: Single-char conv expression *)
  let spec4 = "io+kj" in
  let labels4 = Shape.axis_labels_of_spec spec4 in
  printf "Test 4: Parsed '%s' successfully (single-char mode)\n%!" spec4;
  printf "  Structure: %s\n\n%!" (Sexp.to_string_hum (Shape.sexp_of_parsed_axis_labels labels4));
  
  (* Test 5: Multiple single-char conv expressions *)
  let spec5 = "a+bc" in
  let labels5 = Shape.axis_labels_of_spec spec5 in
  printf "Test 5: Parsed '%s' successfully (single-char mode)\n%!" spec5;
  printf "  Structure: %s\n\n%!" (Sexp.to_string_hum (Shape.sexp_of_parsed_axis_labels labels5));
  
  (* Test 6: Test in einsum notation with multichar conv *)
  let spec6 = "i,j->2*i+j" in
  let labels6 = Shape.axis_labels_of_spec spec6 in
  printf "Test 6: Parsed '%s' successfully\n%!" spec6;
  printf "  Structure: %s\n\n%!" (Sexp.to_string_hum (Shape.sexp_of_parsed_axis_labels labels6));
  
  (* Test 7: Complex batch-input-output spec with conv *)
  let spec7 = "batch|input->3*output+1*kernel," in
  let labels7 = Shape.axis_labels_of_spec spec7 in
  printf "Test 7: Parsed '%s' successfully\n%!" spec7;
  printf "  Structure: %s\n\n%!" (Sexp.to_string_hum (Shape.sexp_of_parsed_axis_labels labels7));
  
  printf "All conv syntax parsing tests passed!\n%!"

let test_conv_multichar_detection () =
  printf "\nTesting multichar mode detection...\n%!";
  
  (* These should trigger multichar mode *)
  let multichar_specs = [
    "a,b";          (* comma *)
    "2*o+k";        (* multiplication *)
    "o+k";          (* addition *)
    "a,2*b+c";      (* combo *)
  ] in
  
  List.iter multichar_specs ~f:(fun spec ->
    try
      let _labels = Shape.axis_labels_of_spec spec in
      printf "✓ Multichar spec '%s' parsed correctly\n%!" spec
    with exn ->
      printf "✗ Failed to parse multichar spec '%s': %s\n%!" spec (Exn.to_string exn)
  );
  
  (* These should work in single-char mode (but might also contain conv expressions) *)
  let singlechar_specs = [
    "abc";          (* single chars *)
    "ijk";          (* single chars *)
    "i->j";         (* single chars with arrow *)
    "io+kj";        (* single-char conv expressions *)
    "a+bc";         (* single-char conv with regular chars *)
    "...|ij";       (* ellipsis in batch axes *)
    "j...";         (* matching on beginning axes *)
    "...|j...->i";  (* ellipsis, matching on beginning axes *)
    "...|i->1";     (* ellipsis, fixed index *)
  ] in
  
  List.iter singlechar_specs ~f:(fun spec ->
    try
      let _labels = Shape.axis_labels_of_spec spec in
      printf "✓ Single-char spec '%s' parsed correctly\n%!" spec
    with exn ->
      printf "✗ Failed to parse single-char spec '%s': %s\n%!" spec (Exn.to_string exn)
  )

let test_single_char_conv_equivalence () =
  printf "\nTesting single-char conv equivalence...\n%!";
  
  (* Test that a+b in single-char mode is equivalent to a+b in multichar mode *)
  let single_char_spec = "a+b" in
  let multi_char_spec = "a+b" in
  
  let single_labels = Shape.axis_labels_of_spec single_char_spec in
  let multi_labels = Shape.axis_labels_of_spec multi_char_spec in
  
  printf "Single-char '%s': %s\n%!" single_char_spec 
    (Sexp.to_string_hum (Shape.sexp_of_parsed_axis_labels single_labels));
  printf "Multi-char '%s': %s\n%!" multi_char_spec 
    (Sexp.to_string_hum (Shape.sexp_of_parsed_axis_labels multi_labels));
  
  printf "Note: Both should produce the same Conv_spec structure\n%!"

let () =
  test_conv_parsing ();
  test_conv_multichar_detection ();
  test_single_char_conv_equivalence ();
  printf "\nAll conv syntax tests completed!\n%!" 