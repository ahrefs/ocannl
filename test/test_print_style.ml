let test_print_styles () =
  let open Ocannl.Row in
  
  Stdio.printf "Testing print_style functionality:\n\n";
  
  (* Create a solved dimension with all possible attributes *)
  let solved_dim_full = { 
    d = 28; 
    padding = Some 2; 
    label = Some "height"; 
    proj_id = None
  } in
  
  (* Create a dimension with projection by using fresh_row_proj *)
  let row_with_dim = { 
    dims = [get_dim ~d:32 ~label:"width" ()]; 
    bcast = Broadcastable; 
    id = row_id ~sh_id:1 ~kind:`Output 
  } in
  let row_with_proj = fresh_row_proj row_with_dim in
  let solved_dim_with_proj = match row_with_proj.dims with
    | [Dim sd] -> sd
    | _ -> failwith "Expected single dimension"
  in
  
  (* Create a solved dimension with minimal attributes *)
  let solved_dim_minimal = { 
    d = 64; 
    padding = None; 
    label = None; 
    proj_id = None 
  } in
  
  (* Create a solved dimension with only padding *)
  let solved_dim_padding = { 
    d = 32; 
    padding = Some 3; 
    label = Some "width"; 
    proj_id = None 
  } in
  
  (* Create a variable dimension *)
  let var_dim_labeled = get_var ~label:"channels" () in
  let var_dim_unlabeled = get_var () in
  
  Stdio.printf "=== Testing solved_dim_to_string ===\n";
  Stdio.printf "Full attributes (d=28, padding=2, label=height, proj_id):\n";
  Stdio.printf "  Only_labels: %s\n" (solved_dim_to_string Only_labels solved_dim_full);
  Stdio.printf "  Axis_size: %s\n" (solved_dim_to_string Axis_size solved_dim_full);
  Stdio.printf "  Axis_number_and_size: %s\n" (solved_dim_to_string Axis_number_and_size solved_dim_full);
  Stdio.printf "  Projection_and_size: %s\n" (solved_dim_to_string Projection_and_size solved_dim_full);
  
  Stdio.printf "\nMinimal attributes (d=64, no padding, no label, no proj_id):\n";
  Stdio.printf "  Only_labels: %s\n" (solved_dim_to_string Only_labels solved_dim_minimal);
  Stdio.printf "  Axis_size: %s\n" (solved_dim_to_string Axis_size solved_dim_minimal);
  Stdio.printf "  Projection_and_size: %s\n" (solved_dim_to_string Projection_and_size solved_dim_minimal);
  
  Stdio.printf "\nWith padding only (d=32, padding=3, label=width, no proj_id):\n";
  Stdio.printf "  Axis_size: %s\n" (solved_dim_to_string Axis_size solved_dim_padding);
  Stdio.printf "  Projection_and_size: %s\n" (solved_dim_to_string Projection_and_size solved_dim_padding);
  
  Stdio.printf "\nWith projection (d=32, label=width, proj_id):\n";
  Stdio.printf "  Axis_size: %s\n" (solved_dim_to_string Axis_size solved_dim_with_proj);
  Stdio.printf "  Projection_and_size: %s\n" (solved_dim_to_string Projection_and_size solved_dim_with_proj);
  
  Stdio.printf "\n=== Testing dim_to_string ===\n";
  Stdio.printf "Solved dimensions:\n";
  Stdio.printf "  Only_labels (full): %s\n" (dim_to_string Only_labels (Dim solved_dim_full));
  Stdio.printf "  Axis_size (full): %s\n" (dim_to_string Axis_size (Dim solved_dim_full));
  Stdio.printf "  Projection_and_size (full): %s\n" (dim_to_string Projection_and_size (Dim solved_dim_full));
  Stdio.printf "  Only_labels (minimal): %s\n" (dim_to_string Only_labels (Dim solved_dim_minimal));
  Stdio.printf "  Axis_size (minimal): %s\n" (dim_to_string Axis_size (Dim solved_dim_minimal));
  
  Stdio.printf "\nVariable dimensions:\n";
  Stdio.printf "  Only_labels (labeled var): %s\n" (dim_to_string Only_labels (Var var_dim_labeled));
  Stdio.printf "  Axis_size (labeled var): %s\n" (dim_to_string Axis_size (Var var_dim_labeled));
  Stdio.printf "  Projection_and_size (labeled var): %s\n" (dim_to_string Projection_and_size (Var var_dim_labeled));
  Stdio.printf "  Only_labels (unlabeled var): %s\n" (dim_to_string Only_labels (Var var_dim_unlabeled));
  Stdio.printf "  Axis_size (unlabeled var): %s\n" (dim_to_string Axis_size (Var var_dim_unlabeled))

let test_shape_to_string () =
  let open Ocannl in
  
  Stdio.printf "\n=== Testing Shape.to_string_hum ===\n";
  
  (* Create a simple shape *)
  let shape = Shape.make 
    ~batch_dims:[1] 
    ~input_dims:[784] 
    ~output_dims:[10; 5] 
    ~debug_name:"test_shape" 
    ~id:42 
    () in
  
  Stdio.printf "Shape with batch=[1], input=[784], output=[10,5]:\n";
  Stdio.printf "  Only_labels: %s\n" (Shape.to_string_hum ~style:Row.Only_labels shape);
  Stdio.printf "  Axis_size: %s\n" (Shape.to_string_hum ~style:Row.Axis_size shape);
  Stdio.printf "  Axis_number_and_size: %s\n" (Shape.to_string_hum ~style:Row.Axis_number_and_size shape);
  Stdio.printf "  Projection_and_size: %s\n" (Shape.to_string_hum ~style:Row.Projection_and_size shape);
  
  (* Test default style *)
  Stdio.printf "  Default style: %s\n" (Shape.to_string_hum shape)

let () =
  test_print_styles ();
  test_shape_to_string () 