open Ocannl
open Operation

(* Example demonstrating the new interface ergonomics with the uniform operation *)

(* Old interface would require specifying all parameters at once *)
(* let t1 = uniform ~label:["t1"] ~grad_spec:If_needed ~batch_dims:[10] ~output_dims:[5; 3] () *)

(* New interface allows partial application and configuration *)
let uniform_generator = uniform ~label:["my_uniform"]

(* Can configure grad_spec and batch dimensions, then reuse *)
let uniform_with_grad = uniform_generator ~grad_spec:Tensor.Require_grad ~batch_dims:[10]

(* Can create multiple tensors with same configuration but different output shapes *)
let tensor1 = uniform_with_grad ~output_dims:[5; 3] ()
let tensor2 = uniform_with_grad ~output_dims:[7; 4] ()

(* Can also create specialized generators *)
let matrix_uniform = uniform_generator ~grad_spec:Tensor.If_needed ~batch_dims:[]
let matrix_5x5 = matrix_uniform ~output_dims:[5; 5] ()
let matrix_3x7 = matrix_uniform ~output_dims:[3; 7] ()

(* The configurability is preserved through partial application *)
let () =
  Stdio.printf "Created tensors with configurable shapes using the new interface!\n";
  Stdio.printf "tensor1 shape: %s\n" (Tensor.debug_name @@ tensor1 ());
  Stdio.printf "tensor2 shape: %s\n" (Tensor.debug_name @@ tensor2 ());
  Stdio.printf "matrix_5x5 shape: %s\n" (Tensor.debug_name @@ matrix_5x5 ());
  Stdio.printf "matrix_3x7 shape: %s\n" (Tensor.debug_name @@ matrix_3x7 ())