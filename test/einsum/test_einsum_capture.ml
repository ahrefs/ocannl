open Base
open Ocannl

let capture_for_computation () =
  let open Operation.DSL_modules in
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let%op x = { x = uniform1 (); o = [ 2; 3 ] } in
  let%op y = { y = uniform1 (); o = [ 3; 4 ] } in
  let%op z = x +* "ab;bc=>ac" [ "a"; "b"; "c" ] y in

  (* Trigger shape inference by accessing the tensor node *)
  let ctx = Train.forward_once ctx z in

  (* Check if dimensions were captured *)
  Stdio.printf "Dimension a: %s\n"
    (match a.var_ref.solved_dim with Some d -> Int.to_string d | None -> "not resolved");
  Stdio.printf "Dimension b: %s\n"
    (match b.var_ref.solved_dim with Some d -> Int.to_string d | None -> "not resolved");
  Stdio.printf "Dimension c: %s\n"
    (match c.var_ref.solved_dim with Some d -> Int.to_string d | None -> "not resolved");

  let%op x2 = { x2 = uniform1 (); o = [ 5; 7 ] } in
  (* Manually call einsum1 with capture_dims for now *)
  let%op y2 = x2 ++ "ij=>ji" [ "i"; "j" ] in

  (* Trigger shape inference by accessing the tensor node *)
  let ctx = Train.forward_once ctx y2 in

  (* Check if dimensions were captured *)
  Stdio.printf "Dimension i: %s\n"
    (match i.var_ref.solved_dim with Some d -> Int.to_string d | None -> "not resolved");
  Stdio.printf "Dimension j: %s\n"
    (match j.var_ref.solved_dim with Some d -> Int.to_string d | None -> "not resolved");

  (* Test capturing row variables *)
  let%op x3 = { x3 = uniform1 (); o = [ 2; 3; 4 ] } in
  let%op y3 = { y3 = uniform1 (); o = [ 3; 4; 5 ] } in
  let%op z3 = x3 +* "a..r..;..r..b=>ab" [ "r" ] y3 in

  (* Trigger shape inference *)
  let ctx = Train.forward_once ctx z3 in

  (* Check if row variable was captured *)
  Stdio.printf "Row variable r (product of dims): %s\n"
    (match r.var_ref.solved_dim with Some d -> Int.to_string d | None -> "not resolved");

  let%op dim_calc = dim a + dim j + dim r in
  let _ctx = Train.forward_once ctx dim_calc in

  Train.printf ~here:[%here] ~with_code:false ~with_grad:false dim_calc

let test_set_dim_and_set_equal () =
  let open Operation.DSL_modules in
  Tensor.unsafe_reinitialize ();
  Stdio.printf "\n=== Testing set_dim and set_equal functionality ===\n";

  (* Test 1: set_dim functionality *)
  let var1 = Shape.get_variable_ref "test_var1" in
  let var2 = Shape.get_variable_ref "test_var2" in

  Shape.set_dim var1 42;
  Stdio.printf "Test 1 - set_dim: var1 set to 42: %s\n"
    (match var1.var_ref.solved_dim with Some d -> Int.to_string d | None -> "not resolved");

  (* Test 2: set_equal with one solved, one unsolved *)
  Shape.set_equal var2 var1;
  Stdio.printf "Test 2 - set_equal (one solved): var2 should now be 42: %s\n"
    (match var2.var_ref.solved_dim with Some d -> Int.to_string d | None -> "not resolved");

  (* Test 3: set_equal with both solved and equal *)
  let var3 = Shape.get_variable_ref "test_var3" in
  Shape.set_dim var3 42;
  Shape.set_equal var1 var3;
  (* Should succeed since both are 42 *)
  Stdio.printf "Test 3 - set_equal (both solved, equal): Success - no exception\n";

  (* Test 4: Error case - set_equal with different solved values *)
  let var6 = Shape.get_variable_ref "test_var6" in
  let var7 = Shape.get_variable_ref "test_var7" in
  Shape.set_dim var6 50;
  Shape.set_dim var7 75;

  (try
     Shape.set_equal var6 var7;
     Stdio.printf "Test 4 - ERROR: Should have thrown exception for different values\n"
   with
  | Row.Shape_error (msg, _) ->
      Stdio.printf "Test 4 - set_equal error case: Correctly caught exception: %s\n" msg
  | _ -> Stdio.printf "Test 4 - ERROR: Unexpected exception type\n");

  (* Test 5: Using captured variables in actual einsum operations *)
  let ctx = Context.auto () in
  let%op x_test = { x_test = uniform1 (); o = [ 3; 4 ] } in
  let%op y_test = { y_test = uniform1 (); o = [ 4; 5 ] } in
  let%op z_test = x_test +* "pq;qr=>pr" [ "p"; "q"; "r" ] y_test in

  (* Don't set equality constraint - just test capturing works *)
  let ctx = Train.forward_once ctx z_test in

  Stdio.printf "Test 5 - einsum variable capture:\n";
  Stdio.printf "  Dimension p: %s\n"
    (match p.var_ref.solved_dim with Some d -> Int.to_string d | None -> "not resolved");
  Stdio.printf "  Dimension q: %s\n"
    (match q.var_ref.solved_dim with Some d -> Int.to_string d | None -> "not resolved");
  Stdio.printf "  Dimension r: %s\n"
    (match r.var_ref.solved_dim with Some d -> Int.to_string d | None -> "not resolved");

  (* Verify expected dimensions *)
  let p_val = match p.var_ref.solved_dim with Some d -> d | None -> -1 in
  let q_val = match q.var_ref.solved_dim with Some d -> d | None -> -1 in
  let r_val = match r.var_ref.solved_dim with Some d -> d | None -> -1 in
  Stdio.printf "  Expected dimensions (p=3, q=4, r=5): p=%d, q=%d, r=%d\n" p_val q_val r_val;

  (* Test 6: Row variable test with set_equal *)
  let%op x_row = { x_row = uniform1 (); o = [ 2; 6 ] } in
  let%op y_row = { y_row = uniform1 (); o = [ 6; 3 ] } in
  let%op z_row = x_row +* "a..s..;..s..b=>ab" [ "s" ] y_row in

  (* Create a dimension variable and set it equal to the row variable *)
  let dim_var = Shape.get_variable_ref "test_dim" in
  Shape.set_equal s dim_var;

  (* s is a row variable, dim_var is a dimension variable *)
  let _ctx = Train.forward_once ctx z_row in

  Stdio.printf "Test 6 - row-dimension equality:\n";
  Stdio.printf "  Row variable s (product): %s\n"
    (match s.var_ref.solved_dim with Some d -> Int.to_string d | None -> "not resolved");
  Stdio.printf "  Dimension variable test_dim: %s\n"
    (match dim_var.var_ref.solved_dim with Some d -> Int.to_string d | None -> "not resolved");

  (* Verify they are equal *)
  let s_val = match s.var_ref.solved_dim with Some d -> d | None -> -1 in
  let dim_val = match dim_var.var_ref.solved_dim with Some d -> d | None -> -1 in
  Stdio.printf "  s == test_dim constraint satisfied: %b (both should be 6)\n"
    (s_val = dim_val && s_val = 6);

  Stdio.printf "=== All tests completed ===\n"

let capture_for_shape_validation () =
  let open Operation.DSL_modules in
  Tensor.unsafe_reinitialize ();
  Shape.unsafe_reinitialize ();
  let ctx = Context.auto () in
  Stdio.printf "\n=== Testing shape validation integration with equality constraints ===\n";

  (* Test 1: Einsum with equality constraints - demonstrate constraint validation *)
  let%op a1 = { a1 = uniform1 (); o = [ 4; 6 ] } in
  let%op b1 = { b1 = uniform1 (); o = [ 6; 4 ] } in
  (* Make k=4 so i=k constraint can work *)
  let%op c1 = a1 +* "ij;jk=>ik" [ "i"; "j"; "k" ] b1 in

  (* Add constraint that i should equal k - this should work since both are 4 *)
  Shape.set_equal i k;

  let ctx = Train.forward_once ctx c1 in

  Stdio.printf "Test 1 - Constraint i=k in matrix multiply:\n";
  Stdio.printf "  Input a1 shape: %s\n" (Shape.to_string_hum a1.shape);
  Stdio.printf "  Input b1 shape: %s\n" (Shape.to_string_hum b1.shape);
  Stdio.printf "  Output c1 shape: %s\n" (Shape.to_string_hum c1.shape);
  Stdio.printf "  Dimension i: %s, k: %s (should be equal)\n"
    (match i.var_ref.solved_dim with Some d -> Int.to_string d | None -> "not resolved")
    (match k.var_ref.solved_dim with Some d -> Int.to_string d | None -> "not resolved");

  (* Test 2: Complex einsum with multiple constraints *)
  let%op x2 = { x2 = uniform1 (); o = [ 3; 5; 7 ] } in
  let%op y2 = { y2 = uniform1 (); o = [ 5; 7; 3 ] } in
  (* Make d=3 so a=d constraint can work *)
  let%op z2 = x2 +* "abc;bcd=>ad" [ "a"; "b"; "c"; "d" ] y2 in

  (* Add constraints: a should equal d, and c should have specific value *)
  Shape.set_equal a d;
  let fixed_c = Shape.get_variable_ref "fixed_c" in
  Shape.set_dim fixed_c 7;
  Shape.set_equal c fixed_c;

  let ctx = Train.forward_once ctx z2 in

  Stdio.printf "\nTest 2 - Multiple constraints (a=d, c=7):\n";
  Stdio.printf "  Input x2 shape: %s\n" (Shape.to_string_hum x2.shape);
  Stdio.printf "  Input y2 shape: %s\n" (Shape.to_string_hum y2.shape);
  Stdio.printf "  Output z2 shape: %s\n" (Shape.to_string_hum z2.shape);
  Stdio.printf "  Dimensions: a=%s, b=%s, c=%s, d=%s\n"
    (match a.var_ref.solved_dim with Some d -> Int.to_string d | None -> "?")
    (match b.var_ref.solved_dim with Some d -> Int.to_string d | None -> "?")
    (match c.var_ref.solved_dim with Some d -> Int.to_string d | None -> "?")
    (match d.var_ref.solved_dim with Some d -> Int.to_string d | None -> "?");

  (* Test 3: Row variables with constraints *)
  let%op r1 = { r1 = uniform1 (); o = [ 2; 3; 4 ] } in
  let%op r2 = { r2 = uniform1 (); o = [ 3; 5 ] } in
  let%op r3 = r1 +* "a, ..row1.., b; ..row2.., c => a, ..row1.., c" [ "row1"; "row2" ] r2 in

  (* Constraint: row1 and row2 should have same total elements *)
  Shape.set_equal row1 row2;

  let ctx = Train.forward_once ctx r3 in

  Stdio.printf "\nTest 3 - Row variable constraints (row1=row2 total elements):\n";
  Stdio.printf "  Input r1 shape: %s\n" (Shape.to_string_hum r1.shape);
  Stdio.printf "  Input r2 shape: %s\n" (Shape.to_string_hum r2.shape);
  Stdio.printf "  Output r3 shape: %s\n" (Shape.to_string_hum r3.shape);
  Stdio.printf "  Row1 total: %s, Row2 total: %s (should be equal)\n"
    (match row1.var_ref.solved_dim with Some d -> Int.to_string d | None -> "?")
    (match row2.var_ref.solved_dim with Some d -> Int.to_string d | None -> "?");

  (* Test 4: Mixed row and dimension constraints *)
  let%op m1 = { m1 = uniform1 (); o = [ 2; 4; 8 ] } in
  let%op m2 = m1 ++ "..mix.., n => n, ..mix.." [ "mix"; "n" ] in
  Shape.set_equal mix n;

  (* Row variable mix should have total elements = p *)
  let _ctx = Train.forward_once ctx m2 in

  Stdio.printf "\nTest 4 - Mixed row-dimension constraints:\n";
  Stdio.printf "  Input m1 shape: %s\n" (Shape.to_string_hum m1.shape);
  Stdio.printf "  Output m2 shape: %s\n" (Shape.to_string_hum m2.shape);
  Stdio.printf "  Dimension n: %s\n"
    (match n.var_ref.solved_dim with Some d -> Int.to_string d | None -> "?");
  Stdio.printf "  Row mix total: %s\n"
    (match mix.var_ref.solved_dim with Some d -> Int.to_string d | None -> "?");

  (* Test 5: Constraint propagation through multiple operations *)
  let%op chain1 = { chain1 = uniform1 (); o = [ 4; 5 ] } in
  let%op chain2 = chain1 ++ "a, b => b, a" [ "a"; "b" ] in
  let%op chain3 = { chain3 = uniform1 (); o = [ 5; 6 ] } in
  let%op chain4 = chain2 +* "e, d; e, f => d, f" [ "d"; "e"; "f" ] chain3 in

  (* Link variables across operations *)
  Shape.set_equal a d;
  Shape.set_equal b e;
  let final_size = Shape.get_variable_ref "final_size" in
  Shape.set_dim final_size 4;
  Shape.set_equal d final_size;

  (* This should propagate back to a *)
  let _ctx = Train.forward_once ctx chain4 in

  Stdio.printf "\nTest 5 - Constraint propagation across operations:\n";
  Stdio.printf "  Chain1 shape: %s\n" (Shape.to_string_hum chain1.shape);
  Stdio.printf "  Chain2 shape: %s\n" (Shape.to_string_hum chain2.shape);
  Stdio.printf "  Chain3 shape: %s\n" (Shape.to_string_hum chain3.shape);
  Stdio.printf "  Chain4 shape: %s\n" (Shape.to_string_hum chain4.shape);
  Stdio.printf "  Variables: a=%s, b=%s, a_chain=%s, b_chain=%s, c_chain=%s\n"
    (match a.var_ref.solved_dim with Some d -> Int.to_string d | None -> "?")
    (match b.var_ref.solved_dim with Some d -> Int.to_string d | None -> "?")
    (match d.var_ref.solved_dim with Some d -> Int.to_string d | None -> "?")
    (match e.var_ref.solved_dim with Some d -> Int.to_string d | None -> "?")
    (match f.var_ref.solved_dim with Some d -> Int.to_string d | None -> "?");

  Stdio.printf "=== Shape inference integration tests completed ===\n"

let capture_for_shape_inference () =
  let open Operation.DSL_modules in
  Tensor.unsafe_reinitialize ();
  Shape.unsafe_reinitialize ();
  let ctx = Context.auto () in
  Stdio.printf "\n=== Testing pure shape inference with equality constraints ===\n";

  (* Test 1: Pure matrix multiply with constraint-driven shape inference *)
  let%op m1 = { m1 = uniform1 () } in
  (* No shape specified *)
  let%op m2 = { m2 = uniform1 () } in
  (* No shape specified *)
  let%op result1 = m1 +* "ij;jk=>ik" [ "i"; "j"; "k" ] m2 in

  (* Set constraints to drive shape inference *)
  let i_size = Shape.get_variable_ref "i_size" in
  let j_size = Shape.get_variable_ref "j_size" in
  let k_size = Shape.get_variable_ref "k_size" in

  Shape.set_dim i_size 3;
  Shape.set_dim j_size 4;
  Shape.set_dim k_size 5;
  Shape.set_equal i i_size;
  Shape.set_equal j j_size;
  Shape.set_equal k k_size;

  let ctx = Train.forward_once ctx result1 in

  Stdio.printf "Test 1 - Matrix multiply with constraint-driven shapes:\n";
  Stdio.printf "  m1 inferred shape: %s\n" (Shape.to_string_hum m1.shape);
  Stdio.printf "  m2 inferred shape: %s\n" (Shape.to_string_hum m2.shape);
  Stdio.printf "  result1 inferred shape: %s\n" (Shape.to_string_hum result1.shape);
  Stdio.printf "  Captured dimensions: i=%s, j=%s, k=%s\n"
    (match i.var_ref.solved_dim with Some d -> Int.to_string d | None -> "?")
    (match j.var_ref.solved_dim with Some d -> Int.to_string d | None -> "?")
    (match k.var_ref.solved_dim with Some d -> Int.to_string d | None -> "?");

  (* Test 2: Chain of operations with constraint propagation *)
  let%op base = { base = uniform1 () } in
  (* No shape specified *)
  let%op transposed = base ++ "ab=>ba" [ "a"; "b" ] in
  let%op multiplied = { mult_input = uniform1 () } in
  (* No shape specified *)
  let%op final = transposed +* "ba;bc=>ac" [ "a"; "b"; "c" ] multiplied in

  (* Set dimensions directly on the captured variables *)
  let base_height = Shape.get_variable_ref "base_height" in
  let base_width = Shape.get_variable_ref "base_width" in
  let mult_depth = Shape.get_variable_ref "mult_depth" in

  Shape.set_dim base_height 6;
  Shape.set_dim base_width 8;
  Shape.set_dim mult_depth 10;

  Shape.set_equal a base_height;
  (* This should propagate through the chain *)
  Shape.set_equal b base_width;
  (* This should propagate through the chain *)
  Shape.set_equal c mult_depth;

  let ctx = Train.forward_once ctx final in

  Stdio.printf "\nTest 2 - Chain operations with constraint propagation:\n";
  Stdio.printf "  base inferred shape: %s\n" (Shape.to_string_hum base.shape);
  Stdio.printf "  transposed inferred shape: %s\n" (Shape.to_string_hum transposed.shape);
  Stdio.printf "  multiplied inferred shape: %s\n" (Shape.to_string_hum multiplied.shape);
  Stdio.printf "  final inferred shape: %s\n" (Shape.to_string_hum final.shape);
  Stdio.printf "  Constraint propagation: a=%s, b=%s, c=%s\n"
    (match a.var_ref.solved_dim with Some d -> Int.to_string d | None -> "?")
    (match b.var_ref.solved_dim with Some d -> Int.to_string d | None -> "?")
    (match c.var_ref.solved_dim with Some d -> Int.to_string d | None -> "?");

  (* Test 3: Simple 3-tensor einsum with pure inference *)
  let%op tensor1 = { tensor1 = uniform1 () } in
  (* No shape specified *)
  let%op tensor2 = { tensor2 = uniform1 () } in
  (* No shape specified *)
  let%op result3 = tensor1 +* "xy;yz=>xz" [ "x"; "y"; "z" ] tensor2 in

  (* Set up constraints for shape inference *)
  let x_size = Shape.get_variable_ref "x_size" in
  let y_size = Shape.get_variable_ref "y_size" in
  let z_size = Shape.get_variable_ref "z_size" in

  Shape.set_dim x_size 7;
  Shape.set_dim y_size 8;
  Shape.set_dim z_size 9;

  Shape.set_equal x x_size;
  Shape.set_equal y y_size;
  Shape.set_equal z z_size;

  let ctx = Train.forward_once ctx result3 in

  Stdio.printf "\nTest 3 - Simple 3-tensor einsum with pure inference:\n";
  Stdio.printf "  tensor1 inferred shape: %s\n" (Shape.to_string_hum tensor1.shape);
  Stdio.printf "  tensor2 inferred shape: %s\n" (Shape.to_string_hum tensor2.shape);
  Stdio.printf "  result3 inferred shape: %s\n" (Shape.to_string_hum result3.shape);
  Stdio.printf "  Constraints: x=%s, y=%s, z=%s\n"
    (match x.var_ref.solved_dim with Some d -> Int.to_string d | None -> "?")
    (match y.var_ref.solved_dim with Some d -> Int.to_string d | None -> "?")
    (match z.var_ref.solved_dim with Some d -> Int.to_string d | None -> "?");

  (* Test 4: Complex inference with mixed constraints *)
  let%op complex1 = { complex1 = uniform1 () } in
  (* No shape specified *)
  let%op complex2 = { complex2 = uniform1 () } in
  (* No shape specified *)
  let%op complex_result = complex1 +* "pqr;rst=>pqst" [ "p"; "q"; "r"; "s"; "t" ] complex2 in

  (* Set up interdependent constraints *)
  let size_constraint = Shape.get_variable_ref "size_constraint" in
  Shape.set_dim size_constraint 4;

  (* Make p = t = size_constraint, and q = s *)
  Shape.set_equal p size_constraint;
  Shape.set_equal t size_constraint;
  Shape.set_equal q s;

  (* Set r to specific value *)
  let r_size = Shape.get_variable_ref "r_size" in
  Shape.set_dim r_size 6;
  Shape.set_equal r r_size;

  (* Set q to a different specific value *)
  let q_size = Shape.get_variable_ref "q_size" in
  Shape.set_dim q_size 5;
  Shape.set_equal q q_size;

  (* This will also constrain s=5 due to q=s *)
  let _ctx = Train.forward_once ctx complex_result in

  Stdio.printf "\nTest 4 - Complex interdependent constraints:\n";
  Stdio.printf "  complex1 inferred shape: %s\n" (Shape.to_string_hum complex1.shape);
  Stdio.printf "  complex2 inferred shape: %s\n" (Shape.to_string_hum complex2.shape);
  Stdio.printf "  complex_result inferred shape: %s\n" (Shape.to_string_hum complex_result.shape);
  Stdio.printf "  Constraint resolution: p=%s, q=%s, r=%s, s=%s, t=%s\n"
    (match p.var_ref.solved_dim with Some d -> Int.to_string d | None -> "?")
    (match q.var_ref.solved_dim with Some d -> Int.to_string d | None -> "?")
    (match r.var_ref.solved_dim with Some d -> Int.to_string d | None -> "?")
    (match s.var_ref.solved_dim with Some d -> Int.to_string d | None -> "?")
    (match t.var_ref.solved_dim with Some d -> Int.to_string d | None -> "?");
  Stdio.printf "  Expected: p=4, q=5, r=6, s=5, t=4 (with q=s and p=t constraints satisfied)\n";

  Stdio.printf "=== Pure shape inference tests completed ===\n"

let () =
  capture_for_computation ();
  test_set_dim_and_set_equal ();
  capture_for_shape_validation ();
  capture_for_shape_inference ()
