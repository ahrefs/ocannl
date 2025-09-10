(** Re-export tensor framework modules for backward compatibility *)

module Row = Ocannl_tensor.Row
module Shape = Ocannl_tensor.Shape  
module Tensor = Ocannl_tensor.Tensor
module Operation = Ocannl_tensor.Operation
module PrintBox_utils = Ocannl_tensor.PrintBox_utils

(** User-facing modules *)
module Train = Train
module Nn_blocks = Nn_blocks