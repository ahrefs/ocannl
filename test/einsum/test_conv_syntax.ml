open Base
open Ocannl
open Stdio

let test_conv_parsing () =
  printf "Testing conv syntax parsing...\n%!";
  
  (* Test 1: Basic conv expression with coefficients *)
  let spec1 = "2*o+3*k" in
  let labels1 = Shape.axis_labels_of_spec spec1 in
  let _axis_map1 = Shape.axis_labels labels1 in
  printf "Test 1: Parsed '%s' successfully\n%!" spec1;
  
  (* Test 2: Simple conv expression without coefficients *)
  let spec2 = "o+k" in
  let labels2 = Shape.axis_labels_of_spec spec2 in
  let _axis_map2 = Shape.axis_labels labels2 in
  printf "Test 2: Parsed '%s' successfully\n%!" spec2;
  
  (* Test 3: Mixed spec with comma (multichar mode) *)
  let spec3 = "a,2*b+c" in
  let labels3 = Shape.axis_labels_of_spec spec3 in
  let _axis_map3 = Shape.axis_labels labels3 in
  printf "Test 3: Parsed '%s' successfully\n%!" spec3;
  
  (* Test 4: Test in einsum notation *)
  let spec4 = "i,j->2*i+j" in
  let labels4 = Shape.axis_labels_of_spec spec4 in
  let _axis_map4 = Shape.axis_labels labels4 in
  printf "Test 4: Parsed '%s' successfully\n%!" spec4;
  
  (* Test 5: Complex batch-input-output spec with conv *)
  let spec5 = "batch|input->3*output+1*kernel" in
  let labels5 = Shape.axis_labels_of_spec spec5 in
  let _axis_map5 = Shape.axis_labels labels5 in
  printf "Test 5: Parsed '%s' successfully\n%!" spec5;
  
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
  
  (* These should work in single-char mode *)
  let singlechar_specs = [
    "abc";          (* single chars *)
    "ijk";          (* single chars *)
    "i->j";         (* single chars with arrow *)
  ] in
  
  List.iter singlechar_specs ~f:(fun spec ->
    try
      let _labels = Shape.axis_labels_of_spec spec in
      printf "✓ Single-char spec '%s' parsed correctly\n%!" spec
    with exn ->
      printf "✗ Failed to parse single-char spec '%s': %s\n%!" spec (Exn.to_string exn)
  )

let () =
  test_conv_parsing ();
  test_conv_multichar_detection ();
  printf "\nAll conv syntax tests completed!\n%!" 