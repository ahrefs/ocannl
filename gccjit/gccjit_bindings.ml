(* The MIT License (MIT)

   Copyright (c) 2015 Nicolas Ojeda Bar <n.oje.bar@gmail.com>

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE. *)

open Ctypes
open Gccjit_bindings_types

module Functions (F : Ctypes.FOREIGN) = struct
  module E = Types_generated

  let gcc_jit_context_acquire = F.foreign "gcc_jit_context_acquire" F.(void @-> returning gcc_jit_context)
  let gcc_jit_context_release = F.foreign "gcc_jit_context_release" F.(gcc_jit_context @-> returning void)

  let gcc_jit_context_set_str_option =
    F.foreign "gcc_jit_context_set_str_option"
      F.(gcc_jit_context @-> E.gcc_jit_str_option @-> string @-> returning void)

  let gcc_jit_context_set_int_option =
    F.foreign "gcc_jit_context_set_int_option"
      F.(gcc_jit_context @-> E.gcc_jit_int_option @-> int @-> returning void)

  let gcc_jit_context_set_bool_option =
    F.foreign "gcc_jit_context_set_bool_option"
      F.(gcc_jit_context @-> E.gcc_jit_bool_option @-> bool @-> returning void)

  let gcc_jit_context_compile =
    F.foreign "gcc_jit_context_compile" F.(gcc_jit_context @-> returning gcc_jit_result)

  let gcc_jit_context_compile_to_file =
    F.foreign "gcc_jit_context_compile_to_file"
      F.(gcc_jit_context @-> E.gcc_jit_output_kind @-> string @-> returning void)

  let gcc_jit_context_dump_to_file =
    F.foreign "gcc_jit_context_dump_to_file" F.(gcc_jit_context @-> string @-> int @-> returning void)

  let gcc_jit_context_set_logfile =
    F.foreign "gcc_jit_context_set_logfile"
      F.(gcc_jit_context @-> ptr void @-> int @-> int @-> returning void)

  let gcc_jit_context_get_first_error =
    F.foreign "gcc_jit_context_get_first_error" F.(gcc_jit_context @-> returning string_opt)

  let gcc_jit_context_get_last_error =
    F.foreign "gcc_jit_context_get_last_error" F.(gcc_jit_context @-> returning string_opt)

  let gcc_jit_result_get_code =
    F.foreign "gcc_jit_result_get_code" F.(gcc_jit_result @-> string @-> returning (ptr void))

  let gcc_jit_result_get_global =
    F.foreign "gcc_jit_result_get_global" F.(gcc_jit_result @-> string @-> returning (ptr void))

  let gcc_jit_result_release = F.foreign "gcc_jit_result_release" F.(gcc_jit_result @-> returning void)

  let gcc_jit_object_get_context =
    F.foreign "gcc_jit_object_get_context" F.(gcc_jit_object @-> returning gcc_jit_context)

  let gcc_jit_object_get_debug_string =
    F.foreign "gcc_jit_object_get_debug_string" F.(gcc_jit_object @-> returning string)
  (* CHECK string NULL ? *)

  let gcc_jit_context_new_location =
    F.foreign "gcc_jit_context_new_location"
      F.(gcc_jit_context @-> string @-> int @-> int @-> returning gcc_jit_location)

  let gcc_jit_location_as_object =
    F.foreign "gcc_jit_location_as_object" F.(gcc_jit_location @-> returning gcc_jit_object)

  let gcc_jit_type_as_object =
    F.foreign "gcc_jit_type_as_object" F.(gcc_jit_type @-> returning gcc_jit_object)

  let gcc_jit_context_get_type =
    F.foreign "gcc_jit_context_get_type" F.(gcc_jit_context @-> E.gcc_jit_types @-> returning gcc_jit_type)

  let gcc_jit_context_get_int_type =
    F.foreign "gcc_jit_context_get_int_type" F.(gcc_jit_context @-> int @-> int @-> returning gcc_jit_type)

  let gcc_jit_type_get_pointer =
    F.foreign "gcc_jit_type_get_pointer" F.(gcc_jit_type @-> returning gcc_jit_type)

  let gcc_jit_type_get_const = F.foreign "gcc_jit_type_get_const" F.(gcc_jit_type @-> returning gcc_jit_type)

  let gcc_jit_type_get_volatile =
    F.foreign "gcc_jit_type_get_volatile" F.(gcc_jit_type @-> returning gcc_jit_type)

  let gcc_jit_context_new_array_type =
    F.foreign "gcc_jit_context_new_array_type"
      F.(gcc_jit_context @-> gcc_jit_location @-> gcc_jit_type @-> int @-> returning gcc_jit_type)

  let gcc_jit_context_new_field =
    F.foreign "gcc_jit_context_new_field"
      F.(gcc_jit_context @-> gcc_jit_location @-> gcc_jit_type @-> string @-> returning gcc_jit_field)

  let gcc_jit_field_as_object =
    F.foreign "gcc_jit_field_as_object" F.(gcc_jit_field @-> returning gcc_jit_object)

  let gcc_jit_context_new_struct_type =
    F.foreign "gcc_jit_context_new_struct_type"
      F.(
        gcc_jit_context @-> gcc_jit_location @-> string @-> int @-> ptr gcc_jit_field
        @-> returning gcc_jit_struct)

  let gcc_jit_context_new_opaque_struct =
    F.foreign "gcc_jit_context_new_opaque_struct"
      F.(gcc_jit_context @-> gcc_jit_location @-> string @-> returning gcc_jit_struct)

  let gcc_jit_struct_as_type =
    F.foreign "gcc_jit_struct_as_type" F.(gcc_jit_struct @-> returning gcc_jit_type)

  let gcc_jit_struct_set_fields =
    F.foreign "gcc_jit_struct_set_fields"
      F.(gcc_jit_struct @-> gcc_jit_location @-> int @-> ptr gcc_jit_field @-> returning void)

  let gcc_jit_context_new_union_type =
    F.foreign "gcc_jit_context_new_union_type"
      F.(
        gcc_jit_context @-> gcc_jit_location @-> string @-> int @-> ptr gcc_jit_field
        @-> returning gcc_jit_type)

  let gcc_jit_context_new_function_ptr_type =
    F.foreign "gcc_jit_context_new_function_ptr_type"
      F.(
        gcc_jit_context @-> gcc_jit_location @-> gcc_jit_type @-> int @-> ptr gcc_jit_type @-> int
        @-> returning gcc_jit_type)

  let gcc_jit_context_new_param =
    F.foreign "gcc_jit_context_new_param"
      F.(gcc_jit_context @-> gcc_jit_location @-> gcc_jit_type @-> string @-> returning gcc_jit_param)

  let gcc_jit_param_as_object =
    F.foreign "gcc_jit_param_as_object" F.(gcc_jit_param @-> returning gcc_jit_object)

  let gcc_jit_param_as_lvalue =
    F.foreign "gcc_jit_param_as_lvalue" F.(gcc_jit_param @-> returning gcc_jit_lvalue)

  let gcc_jit_param_as_rvalue =
    F.foreign "gcc_jit_param_as_rvalue" F.(gcc_jit_param @-> returning gcc_jit_rvalue)

  let gcc_jit_context_new_function =
    F.foreign "gcc_jit_context_new_function"
      F.(
        gcc_jit_context @-> gcc_jit_location @-> E.gcc_jit_function_kind @-> gcc_jit_type @-> string @-> int
        @-> ptr gcc_jit_param @-> int @-> returning gcc_jit_function)

  let gcc_jit_context_get_builtin_function =
    F.foreign "gcc_jit_context_get_builtin_function"
      F.(gcc_jit_context @-> string @-> returning gcc_jit_function)

  let gcc_jit_function_as_object =
    F.foreign "gcc_jit_function_as_object" F.(gcc_jit_function @-> returning gcc_jit_object)

  let gcc_jit_function_get_param =
    F.foreign "gcc_jit_function_get_param" F.(gcc_jit_function @-> int @-> returning gcc_jit_param)

  let gcc_jit_function_dump_to_dot =
    F.foreign "gcc_jit_function_dump_to_dot" F.(gcc_jit_function @-> string @-> returning void)

  let gcc_jit_function_new_block =
    F.foreign "gcc_jit_function_new_block" F.(gcc_jit_function @-> string_opt @-> returning gcc_jit_block)

  let gcc_jit_block_as_object =
    F.foreign "gcc_jit_block_as_object" F.(gcc_jit_block @-> returning gcc_jit_object)

  let gcc_jit_block_get_function =
    F.foreign "gcc_jit_block_get_function" F.(gcc_jit_block @-> returning gcc_jit_function)

  let gcc_jit_context_new_global =
    F.foreign "gcc_jit_context_new_global"
      F.(
        gcc_jit_context @-> gcc_jit_location @-> E.gcc_jit_global_kind @-> gcc_jit_type @-> string
        @-> returning gcc_jit_lvalue)

  let gcc_jit_lvalue_as_object =
    F.foreign "gcc_jit_lvalue_as_object" F.(gcc_jit_lvalue @-> returning gcc_jit_object)

  let gcc_jit_lvalue_as_rvalue =
    F.foreign "gcc_jit_lvalue_as_rvalue" F.(gcc_jit_lvalue @-> returning gcc_jit_rvalue)

  let gcc_jit_rvalue_as_object =
    F.foreign "gcc_jit_rvalue_as_object" F.(gcc_jit_rvalue @-> returning gcc_jit_object)

  let gcc_jit_rvalue_get_type =
    F.foreign "gcc_jit_rvalue_get_type" F.(gcc_jit_rvalue @-> returning gcc_jit_type)

  let gcc_jit_context_new_rvalue_from_int =
    F.foreign "gcc_jit_context_new_rvalue_from_int"
      F.(gcc_jit_context @-> gcc_jit_type @-> int @-> returning gcc_jit_rvalue)
  (* CHECK int *)

  let gcc_jit_context_new_rvalue_from_long =
    F.foreign "gcc_jit_context_new_rvalue_from_long"
      F.(gcc_jit_context @-> gcc_jit_type @-> int @-> returning gcc_jit_rvalue)
  (* CHECK int *)

  let gcc_jit_context_zero =
    F.foreign "gcc_jit_context_zero" F.(gcc_jit_context @-> gcc_jit_type @-> returning gcc_jit_rvalue)

  let gcc_jit_context_one =
    F.foreign "gcc_jit_context_one" F.(gcc_jit_context @-> gcc_jit_type @-> returning gcc_jit_rvalue)

  let gcc_jit_context_new_rvalue_from_double =
    F.foreign "gcc_jit_context_new_rvalue_from_double"
      F.(gcc_jit_context @-> gcc_jit_type @-> float @-> returning gcc_jit_rvalue)

  let gcc_jit_context_new_rvalue_from_ptr =
    F.foreign "gcc_jit_context_new_rvalue_from_ptr"
      F.(gcc_jit_context @-> gcc_jit_type @-> ptr void @-> returning gcc_jit_rvalue)

  let gcc_jit_context_null =
    F.foreign "gcc_jit_context_null" F.(gcc_jit_context @-> gcc_jit_type @-> returning gcc_jit_rvalue)

  let gcc_jit_context_new_string_literal =
    F.foreign "gcc_jit_context_new_string_literal" F.(gcc_jit_context @-> string @-> returning gcc_jit_rvalue)

  let gcc_jit_context_new_unary_op =
    F.foreign "gcc_jit_context_new_unary_op"
      F.(
        gcc_jit_context @-> gcc_jit_location @-> E.gcc_jit_unary_op @-> gcc_jit_type @-> gcc_jit_rvalue
        @-> returning gcc_jit_rvalue)

  let gcc_jit_context_new_binary_op =
    F.foreign "gcc_jit_context_new_binary_op"
      F.(
        gcc_jit_context @-> gcc_jit_location @-> E.gcc_jit_binary_op @-> gcc_jit_type @-> gcc_jit_rvalue
        @-> gcc_jit_rvalue @-> returning gcc_jit_rvalue)

  let gcc_jit_context_new_comparison =
    F.foreign "gcc_jit_context_new_comparison"
      F.(
        gcc_jit_context @-> gcc_jit_location @-> E.gcc_jit_comparison @-> gcc_jit_rvalue @-> gcc_jit_rvalue
        @-> returning gcc_jit_rvalue)

  let gcc_jit_context_new_call =
    F.foreign "gcc_jit_context_new_call"
      F.(
        gcc_jit_context @-> gcc_jit_location @-> gcc_jit_function @-> int @-> ptr gcc_jit_rvalue
        @-> returning gcc_jit_rvalue)

  let gcc_jit_context_new_call_through_ptr =
    F.foreign "gcc_jit_context_new_call_through_ptr"
      F.(
        gcc_jit_context @-> gcc_jit_location @-> gcc_jit_rvalue @-> int @-> ptr gcc_jit_rvalue
        @-> returning gcc_jit_rvalue)

  let gcc_jit_context_new_cast =
    F.foreign "gcc_jit_context_new_cast"
      F.(
        gcc_jit_context @-> gcc_jit_location @-> gcc_jit_rvalue @-> gcc_jit_type @-> returning gcc_jit_rvalue)

  let gcc_jit_context_new_array_access =
    F.foreign "gcc_jit_context_new_array_access"
      F.(
        gcc_jit_context @-> gcc_jit_location @-> gcc_jit_rvalue @-> gcc_jit_rvalue
        @-> returning gcc_jit_lvalue)

  let gcc_jit_lvalue_access_field =
    F.foreign "gcc_jit_lvalue_access_field"
      F.(gcc_jit_lvalue @-> gcc_jit_location @-> gcc_jit_field @-> returning gcc_jit_lvalue)

  let gcc_jit_rvalue_access_field =
    F.foreign "gcc_jit_rvalue_access_field"
      F.(gcc_jit_rvalue @-> gcc_jit_location @-> gcc_jit_field @-> returning gcc_jit_rvalue)

  let gcc_jit_rvalue_dereference_field =
    F.foreign "gcc_jit_rvalue_dereference_field"
      F.(gcc_jit_rvalue @-> gcc_jit_location @-> gcc_jit_field @-> returning gcc_jit_lvalue)

  let gcc_jit_rvalue_dereference =
    F.foreign "gcc_jit_rvalue_dereference"
      F.(gcc_jit_rvalue @-> gcc_jit_location @-> returning gcc_jit_lvalue)

  let gcc_jit_lvalue_get_address =
    F.foreign "gcc_jit_lvalue_get_address"
      F.(gcc_jit_lvalue @-> gcc_jit_location @-> returning gcc_jit_rvalue)

  let gcc_jit_function_new_local =
    F.foreign "gcc_jit_function_new_local"
      F.(gcc_jit_function @-> gcc_jit_location @-> gcc_jit_type @-> string @-> returning gcc_jit_lvalue)

  let gcc_jit_block_add_eval =
    F.foreign "gcc_jit_block_add_eval"
      F.(gcc_jit_block @-> gcc_jit_location @-> gcc_jit_rvalue @-> returning void)

  let gcc_jit_block_add_assignment =
    F.foreign "gcc_jit_block_add_assignment"
      F.(gcc_jit_block @-> gcc_jit_location @-> gcc_jit_lvalue @-> gcc_jit_rvalue @-> returning void)

  let gcc_jit_block_add_assignment_op =
    F.foreign "gcc_jit_block_add_assignment_op"
      F.(
        gcc_jit_block @-> gcc_jit_location @-> gcc_jit_lvalue @-> E.gcc_jit_binary_op @-> gcc_jit_rvalue
        @-> returning void)

  let gcc_jit_block_add_comment =
    F.foreign "gcc_jit_block_add_comment" F.(gcc_jit_block @-> gcc_jit_location @-> string @-> returning void)

  let gcc_jit_block_end_with_conditional =
    F.foreign "gcc_jit_block_end_with_conditional"
      F.(
        gcc_jit_block @-> gcc_jit_location @-> gcc_jit_rvalue @-> gcc_jit_block @-> gcc_jit_block
        @-> returning void)

  let gcc_jit_block_end_with_jump =
    F.foreign "gcc_jit_block_end_with_jump"
      F.(gcc_jit_block @-> gcc_jit_location @-> gcc_jit_block @-> returning void)

  let gcc_jit_block_end_with_return =
    F.foreign "gcc_jit_block_end_with_return"
      F.(gcc_jit_block @-> gcc_jit_location @-> gcc_jit_rvalue @-> returning void)

  let gcc_jit_block_end_with_void_return =
    F.foreign "gcc_jit_block_end_with_void_return" F.(gcc_jit_block @-> gcc_jit_location @-> returning void)

  let gcc_jit_context_new_child_context =
    F.foreign "gcc_jit_context_new_child_context" F.(gcc_jit_context @-> returning gcc_jit_context)

  let gcc_jit_context_dump_reproducer_to_file =
    F.foreign "gcc_jit_context_dump_reproducer_to_file" F.(gcc_jit_context @-> string @-> returning void)

  external int_of_fd : Unix.file_descr -> int = "%identity"

  let fdopen = F.foreign "fdopen" F.(int @-> string @-> returning (ptr_opt void))

  (*let gcc_jit_context_enable_dump =
    F.foreign "gcc_jit_context_enable_dump" (gcc_jit_context @-> string @-> ptr char @-> returning void)*)
end
