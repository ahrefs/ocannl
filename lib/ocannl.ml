(** Re-export tensor framework modules for backward compatibility *)

module Row = Ocannl_tensor.Row
module Shape = Ocannl_tensor.Shape
module Tensor = Ocannl_tensor.Tensor
module Operation = Ocannl_tensor.Operation
module PrintBox_utils = Ocannl_tensor.PrintBox_utils

module Train = Train
(** User-facing modules *)

module Nn_blocks = Nn_blocks
