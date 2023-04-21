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

type gcc_jit_context_t
type gcc_jit_result_t
type gcc_jit_object_t
type gcc_jit_location_t
type gcc_jit_type_t
type gcc_jit_field_t
type gcc_jit_struct_t
type gcc_jit_param_t
type gcc_jit_lvalue_t
type gcc_jit_rvalue_t
type gcc_jit_function_t
type gcc_jit_block_t
type gcc_jit_str_option = GCC_JIT_STR_OPTION_PROGNAME
type gcc_jit_int_option = GCC_JIT_INT_OPTION_OPTIMIZATION_LEVEL

type gcc_jit_bool_option =
  | GCC_JIT_BOOL_OPTION_DEBUGINFO
  | GCC_JIT_BOOL_OPTION_DUMP_INITIAL_TREE
  | GCC_JIT_BOOL_OPTION_DUMP_INITIAL_GIMPLE
  | GCC_JIT_BOOL_OPTION_DUMP_GENERATED_CODE
  | GCC_JIT_BOOL_OPTION_DUMP_SUMMARY
  | GCC_JIT_BOOL_OPTION_DUMP_EVERYTHING
  | GCC_JIT_BOOL_OPTION_SELFCHECK_GC
  | GCC_JIT_BOOL_OPTION_KEEP_INTERMEDIATES

type gcc_jit_output_kind =
  | GCC_JIT_OUTPUT_KIND_ASSEMBLER
  | GCC_JIT_OUTPUT_KIND_OBJECT_FILE
  | GCC_JIT_OUTPUT_KIND_DYNAMIC_LIBRARY
  | GCC_JIT_OUTPUT_KIND_EXECUTABLE

type gcc_jit_types =
  | GCC_JIT_TYPE_VOID
  | GCC_JIT_TYPE_VOID_PTR
  | GCC_JIT_TYPE_BOOL
  | GCC_JIT_TYPE_CHAR
  | GCC_JIT_TYPE_SIGNED_CHAR
  | GCC_JIT_TYPE_UNSIGNED_CHAR
  | GCC_JIT_TYPE_SHORT
  | GCC_JIT_TYPE_UNSIGNED_SHORT
  | GCC_JIT_TYPE_INT
  | GCC_JIT_TYPE_UNSIGNED_INT
  | GCC_JIT_TYPE_LONG
  | GCC_JIT_TYPE_UNSIGNED_LONG
  | GCC_JIT_TYPE_LONG_LONG
  | GCC_JIT_TYPE_UNSIGNED_LONG_LONG
  | GCC_JIT_TYPE_FLOAT
  | GCC_JIT_TYPE_DOUBLE
  | GCC_JIT_TYPE_LONG_DOUBLE
  | GCC_JIT_TYPE_CONST_CHAR_PTR
  | GCC_JIT_TYPE_SIZE_T
  | GCC_JIT_TYPE_FILE_PTR
  | GCC_JIT_TYPE_COMPLEX_FLOAT
  | GCC_JIT_TYPE_COMPLEX_DOUBLE
  | GCC_JIT_TYPE_COMPLEX_LONG_DOUBLE

type gcc_jit_function_kind =
  | GCC_JIT_FUNCTION_EXPORTED
  | GCC_JIT_FUNCTION_INTERNAL
  | GCC_JIT_FUNCTION_IMPORTED
  | GCC_JIT_FUNCTION_ALWAYS_INLINE

type gcc_jit_global_kind = GCC_JIT_GLOBAL_EXPORTED | GCC_JIT_GLOBAL_INTERNAL | GCC_JIT_GLOBAL_IMPORTED

type gcc_jit_unary_op =
  | GCC_JIT_UNARY_OP_MINUS
  | GCC_JIT_UNARY_OP_BITWISE_NEGATE
  | GCC_JIT_UNARY_OP_LOGICAL_NEGATE
  | GCC_JIT_UNARY_OP_ABS

type gcc_jit_binary_op =
  | GCC_JIT_BINARY_OP_PLUS
  | GCC_JIT_BINARY_OP_MINUS
  | GCC_JIT_BINARY_OP_MULT
  | GCC_JIT_BINARY_OP_DIVIDE
  | GCC_JIT_BINARY_OP_MODULO
  | GCC_JIT_BINARY_OP_BITWISE_AND
  | GCC_JIT_BINARY_OP_BITWISE_XOR
  | GCC_JIT_BINARY_OP_BITWISE_OR
  | GCC_JIT_BINARY_OP_LOGICAL_AND
  | GCC_JIT_BINARY_OP_LOGICAL_OR
  | GCC_JIT_BINARY_OP_LSHIFT
  | GCC_JIT_BINARY_OP_RSHIFT

type gcc_jit_comparison =
  | GCC_JIT_COMPARISON_EQ
  | GCC_JIT_COMPARISON_NE
  | GCC_JIT_COMPARISON_LT
  | GCC_JIT_COMPARISON_LE
  | GCC_JIT_COMPARISON_GT
  | GCC_JIT_COMPARISON_GE

type gcc_jit_context = gcc_jit_context_t structure ptr
type gcc_jit_result = gcc_jit_result_t structure ptr
type gcc_jit_object = gcc_jit_object_t structure ptr
type gcc_jit_location = gcc_jit_location_t structure ptr
type gcc_jit_type = gcc_jit_type_t structure ptr
type gcc_jit_field = gcc_jit_field_t structure ptr
type gcc_jit_struct = gcc_jit_struct_t structure ptr
type gcc_jit_param = gcc_jit_param_t structure ptr
type gcc_jit_lvalue = gcc_jit_lvalue_t structure ptr
type gcc_jit_rvalue = gcc_jit_rvalue_t structure ptr
type gcc_jit_function = gcc_jit_function_t structure ptr
type gcc_jit_block = gcc_jit_block_t structure ptr

let gcc_jit_context : gcc_jit_context typ = ptr (structure "gcc_jit_context")
let gcc_jit_result : gcc_jit_result typ = ptr (structure "gcc_jit_result")
let gcc_jit_object : gcc_jit_object typ = ptr (structure "gcc_jit_object")
let gcc_jit_location : gcc_jit_location typ = ptr (structure "gcc_jit_location")
let gcc_jit_type : gcc_jit_type typ = ptr (structure "gcc_jit_type")
let gcc_jit_field : gcc_jit_field typ = ptr (structure "gcc_jit_field")
let gcc_jit_struct : gcc_jit_struct typ = ptr (structure "gcc_jit_struct")
let gcc_jit_param : gcc_jit_param typ = ptr (structure "gcc_jit_param")
let gcc_jit_lvalue : gcc_jit_lvalue typ = ptr (structure "gcc_jit_lvalue")
let gcc_jit_rvalue : gcc_jit_rvalue typ = ptr (structure "gcc_jit_rvalue")
let gcc_jit_function : gcc_jit_function typ = ptr (structure "gcc_jit_function")
let gcc_jit_block : gcc_jit_block typ = ptr (structure "gcc_jit_block")

module Types (T : Ctypes.TYPE) = struct
  let gcc_jit_str_option_progname = T.constant "GCC_JIT_STR_OPTION_PROGNAME" T.int64_t

  let gcc_jit_str_option =
    T.enum "gcc_jit_str_option" [ (GCC_JIT_STR_OPTION_PROGNAME, gcc_jit_str_option_progname) ]

  let gcc_jit_int_option_optimization_level = T.constant "GCC_JIT_INT_OPTION_OPTIMIZATION_LEVEL" T.int64_t

  let gcc_jit_int_option =
    T.enum "gcc_jit_int_option"
      [ (GCC_JIT_INT_OPTION_OPTIMIZATION_LEVEL, gcc_jit_int_option_optimization_level) ]

  let gcc_jit_bool_option_debuginfo = T.constant "GCC_JIT_BOOL_OPTION_DEBUGINFO" T.int64_t
  let gcc_jit_bool_option_dump_initial_tree = T.constant "GCC_JIT_BOOL_OPTION_DUMP_INITIAL_TREE" T.int64_t
  let gcc_jit_bool_option_dump_initial_gimple = T.constant "GCC_JIT_BOOL_OPTION_DUMP_INITIAL_GIMPLE" T.int64_t
  let gcc_jit_bool_option_dump_generated_code = T.constant "GCC_JIT_BOOL_OPTION_DUMP_GENERATED_CODE" T.int64_t
  let gcc_jit_bool_option_dump_summary = T.constant "GCC_JIT_BOOL_OPTION_DUMP_SUMMARY" T.int64_t
  let gcc_jit_bool_option_dump_everything = T.constant "GCC_JIT_BOOL_OPTION_DUMP_EVERYTHING" T.int64_t
  let gcc_jit_bool_option_selfcheck_gc = T.constant "GCC_JIT_BOOL_OPTION_SELFCHECK_GC" T.int64_t
  let gcc_jit_bool_option_keep_intermediates = T.constant "GCC_JIT_BOOL_OPTION_KEEP_INTERMEDIATES" T.int64_t

  let gcc_jit_bool_option =
    T.enum "gcc_jit_bool_option"
      [
        (GCC_JIT_BOOL_OPTION_DEBUGINFO, gcc_jit_bool_option_debuginfo);
        (GCC_JIT_BOOL_OPTION_DUMP_INITIAL_TREE, gcc_jit_bool_option_dump_initial_tree);
        (GCC_JIT_BOOL_OPTION_DUMP_INITIAL_GIMPLE, gcc_jit_bool_option_dump_initial_gimple);
        (GCC_JIT_BOOL_OPTION_DUMP_GENERATED_CODE, gcc_jit_bool_option_dump_generated_code);
        (GCC_JIT_BOOL_OPTION_DUMP_SUMMARY, gcc_jit_bool_option_dump_summary);
        (GCC_JIT_BOOL_OPTION_DUMP_EVERYTHING, gcc_jit_bool_option_dump_everything);
        (GCC_JIT_BOOL_OPTION_KEEP_INTERMEDIATES, gcc_jit_bool_option_keep_intermediates);
      ]

  let gcc_jit_output_kind_assembler = T.constant "GCC_JIT_OUTPUT_KIND_ASSEMBLER" T.int64_t
  let gcc_jit_output_kind_object_file = T.constant "GCC_JIT_OUTPUT_KIND_OBJECT_FILE" T.int64_t
  let gcc_jit_output_kind_dynamic_library = T.constant "GCC_JIT_OUTPUT_KIND_DYNAMIC_LIBRARY" T.int64_t
  let gcc_jit_output_kind_executable = T.constant "GCC_JIT_OUTPUT_KIND_EXECUTABLE" T.int64_t

  let gcc_jit_output_kind =
    T.enum "gcc_jit_output_kind"
      [
        (GCC_JIT_OUTPUT_KIND_ASSEMBLER, gcc_jit_output_kind_assembler);
        (GCC_JIT_OUTPUT_KIND_OBJECT_FILE, gcc_jit_output_kind_object_file);
        (GCC_JIT_OUTPUT_KIND_DYNAMIC_LIBRARY, gcc_jit_output_kind_dynamic_library);
        (GCC_JIT_OUTPUT_KIND_EXECUTABLE, gcc_jit_output_kind_executable);
      ]

  let gcc_jit_type_void = T.constant "GCC_JIT_TYPE_VOID" T.int64_t
  let gcc_jit_type_void_ptr = T.constant "GCC_JIT_TYPE_VOID_PTR" T.int64_t
  let gcc_jit_type_bool = T.constant "GCC_JIT_TYPE_BOOL" T.int64_t
  let gcc_jit_type_char = T.constant "GCC_JIT_TYPE_CHAR" T.int64_t
  let gcc_jit_type_signed_char = T.constant "GCC_JIT_TYPE_SIGNED_CHAR" T.int64_t
  let gcc_jit_type_unsigned_char = T.constant "GCC_JIT_TYPE_UNSIGNED_CHAR" T.int64_t
  let gcc_jit_type_short = T.constant "GCC_JIT_TYPE_SHORT" T.int64_t
  let gcc_jit_type_unsigned_short = T.constant "GCC_JIT_TYPE_UNSIGNED_SHORT" T.int64_t
  let gcc_jit_type_int = T.constant "GCC_JIT_TYPE_INT" T.int64_t
  let gcc_jit_type_unsigned_int = T.constant "GCC_JIT_TYPE_UNSIGNED_INT" T.int64_t
  let gcc_jit_type_long = T.constant "GCC_JIT_TYPE_LONG" T.int64_t
  let gcc_jit_type_unsigned_long = T.constant "GCC_JIT_TYPE_UNSIGNED_LONG" T.int64_t
  let gcc_jit_type_long_long = T.constant "GCC_JIT_TYPE_LONG_LONG" T.int64_t
  let gcc_jit_type_unsigned_long_long = T.constant "GCC_JIT_TYPE_UNSIGNED_LONG_LONG" T.int64_t
  let gcc_jit_type_float = T.constant "GCC_JIT_TYPE_FLOAT" T.int64_t
  let gcc_jit_type_double = T.constant "GCC_JIT_TYPE_DOUBLE" T.int64_t
  let gcc_jit_type_long_double = T.constant "GCC_JIT_TYPE_LONG_DOUBLE" T.int64_t
  let gcc_jit_type_const_char_ptr = T.constant "GCC_JIT_TYPE_CONST_CHAR_PTR" T.int64_t
  let gcc_jit_type_size_t = T.constant "GCC_JIT_TYPE_SIZE_T" T.int64_t
  let gcc_jit_type_file_ptr = T.constant "GCC_JIT_TYPE_FILE_PTR" T.int64_t
  let gcc_jit_type_complex_float = T.constant "GCC_JIT_TYPE_COMPLEX_FLOAT" T.int64_t
  let gcc_jit_type_complex_double = T.constant "GCC_JIT_TYPE_COMPLEX_DOUBLE" T.int64_t
  let gcc_jit_type_complex_long_double = T.constant "GCC_JIT_TYPE_COMPLEX_LONG_DOUBLE" T.int64_t

  let gcc_jit_types =
    T.enum "gcc_jit_types"
      [
        (GCC_JIT_TYPE_VOID, gcc_jit_type_void);
        (GCC_JIT_TYPE_VOID_PTR, gcc_jit_type_void_ptr);
        (GCC_JIT_TYPE_BOOL, gcc_jit_type_bool);
        (GCC_JIT_TYPE_CHAR, gcc_jit_type_char);
        (GCC_JIT_TYPE_SIGNED_CHAR, gcc_jit_type_signed_char);
        (GCC_JIT_TYPE_UNSIGNED_CHAR, gcc_jit_type_unsigned_char);
        (GCC_JIT_TYPE_SHORT, gcc_jit_type_short);
        (GCC_JIT_TYPE_UNSIGNED_SHORT, gcc_jit_type_unsigned_short);
        (GCC_JIT_TYPE_INT, gcc_jit_type_int);
        (GCC_JIT_TYPE_UNSIGNED_INT, gcc_jit_type_unsigned_int);
        (GCC_JIT_TYPE_LONG, gcc_jit_type_long);
        (GCC_JIT_TYPE_UNSIGNED_LONG, gcc_jit_type_unsigned_long);
        (GCC_JIT_TYPE_LONG_LONG, gcc_jit_type_long_long);
        (GCC_JIT_TYPE_UNSIGNED_LONG_LONG, gcc_jit_type_unsigned_long_long);
        (GCC_JIT_TYPE_FLOAT, gcc_jit_type_float);
        (GCC_JIT_TYPE_DOUBLE, gcc_jit_type_double);
        (GCC_JIT_TYPE_LONG_DOUBLE, gcc_jit_type_long_double);
        (GCC_JIT_TYPE_CONST_CHAR_PTR, gcc_jit_type_const_char_ptr);
        (GCC_JIT_TYPE_SIZE_T, gcc_jit_type_size_t);
        (GCC_JIT_TYPE_FILE_PTR, gcc_jit_type_file_ptr);
        (GCC_JIT_TYPE_COMPLEX_FLOAT, gcc_jit_type_complex_float);
        (GCC_JIT_TYPE_COMPLEX_DOUBLE, gcc_jit_type_complex_double);
        (GCC_JIT_TYPE_COMPLEX_LONG_DOUBLE, gcc_jit_type_complex_long_double);
      ]

  let gcc_jit_function_exported = T.constant "GCC_JIT_FUNCTION_EXPORTED" T.int64_t
  let gcc_jit_function_internal = T.constant "GCC_JIT_FUNCTION_INTERNAL" T.int64_t
  let gcc_jit_function_imported = T.constant "GCC_JIT_FUNCTION_IMPORTED" T.int64_t
  let gcc_jit_function_always_inline = T.constant "GCC_JIT_FUNCTION_ALWAYS_INLINE" T.int64_t

  let gcc_jit_function_kind =
    T.enum "gcc_jit_function_kind"
      [
        (GCC_JIT_FUNCTION_EXPORTED, gcc_jit_function_exported);
        (GCC_JIT_FUNCTION_INTERNAL, gcc_jit_function_internal);
        (GCC_JIT_FUNCTION_IMPORTED, gcc_jit_function_imported);
        (GCC_JIT_FUNCTION_ALWAYS_INLINE, gcc_jit_function_always_inline);
      ]

  let gcc_jit_global_exported = T.constant "GCC_JIT_GLOBAL_EXPORTED" T.int64_t
  let gcc_jit_global_internal = T.constant "GCC_JIT_GLOBAL_INTERNAL" T.int64_t
  let gcc_jit_global_imported = T.constant "GCC_JIT_GLOBAL_IMPORTED" T.int64_t

  let gcc_jit_global_kind =
    T.enum "gcc_jit_global_kind"
      [
        (GCC_JIT_GLOBAL_EXPORTED, gcc_jit_global_exported);
        (GCC_JIT_GLOBAL_INTERNAL, gcc_jit_global_internal);
        (GCC_JIT_GLOBAL_IMPORTED, gcc_jit_global_imported);
      ]

  let gcc_jit_unary_op_minus = T.constant "GCC_JIT_UNARY_OP_MINUS" T.int64_t
  let gcc_jit_unary_op_bitwise_negate = T.constant "GCC_JIT_UNARY_OP_BITWISE_NEGATE" T.int64_t
  let gcc_jit_unary_op_logical_negate = T.constant "GCC_JIT_UNARY_OP_LOGICAL_NEGATE" T.int64_t
  let gcc_jit_unary_op_abs = T.constant "GCC_JIT_UNARY_OP_ABS" T.int64_t

  let gcc_jit_unary_op =
    T.enum "gcc_jit_unary_op"
      [
        (GCC_JIT_UNARY_OP_MINUS, gcc_jit_unary_op_minus);
        (GCC_JIT_UNARY_OP_BITWISE_NEGATE, gcc_jit_unary_op_bitwise_negate);
        (GCC_JIT_UNARY_OP_LOGICAL_NEGATE, gcc_jit_unary_op_logical_negate);
        (GCC_JIT_UNARY_OP_ABS, gcc_jit_unary_op_abs);
      ]

  let gcc_jit_binary_op_plus = T.constant "GCC_JIT_BINARY_OP_PLUS" T.int64_t
  let gcc_jit_binary_op_minus = T.constant "GCC_JIT_BINARY_OP_MINUS" T.int64_t
  let gcc_jit_binary_op_mult = T.constant "GCC_JIT_BINARY_OP_MULT" T.int64_t
  let gcc_jit_binary_op_divide = T.constant "GCC_JIT_BINARY_OP_DIVIDE" T.int64_t
  let gcc_jit_binary_op_modulo = T.constant "GCC_JIT_BINARY_OP_MODULO" T.int64_t
  let gcc_jit_binary_op_bitwise_and = T.constant "GCC_JIT_BINARY_OP_BITWISE_AND" T.int64_t
  let gcc_jit_binary_op_bitwise_xor = T.constant "GCC_JIT_BINARY_OP_BITWISE_XOR" T.int64_t
  let gcc_jit_binary_op_bitwise_or = T.constant "GCC_JIT_BINARY_OP_BITWISE_OR" T.int64_t
  let gcc_jit_binary_op_logical_and = T.constant "GCC_JIT_BINARY_OP_LOGICAL_AND" T.int64_t
  let gcc_jit_binary_op_logical_or = T.constant "GCC_JIT_BINARY_OP_LOGICAL_OR" T.int64_t
  let gcc_jit_binary_op_lshift = T.constant "GCC_JIT_BINARY_OP_LSHIFT" T.int64_t
  let gcc_jit_binary_op_rshift = T.constant "GCC_JIT_BINARY_OP_RSHIFT" T.int64_t

  let gcc_jit_binary_op =
    T.enum "gcc_jit_binary_op"
      [
        (GCC_JIT_BINARY_OP_PLUS, gcc_jit_binary_op_plus);
        (GCC_JIT_BINARY_OP_MINUS, gcc_jit_binary_op_minus);
        (GCC_JIT_BINARY_OP_MULT, gcc_jit_binary_op_mult);
        (GCC_JIT_BINARY_OP_DIVIDE, gcc_jit_binary_op_divide);
        (GCC_JIT_BINARY_OP_MODULO, gcc_jit_binary_op_modulo);
        (GCC_JIT_BINARY_OP_BITWISE_AND, gcc_jit_binary_op_bitwise_and);
        (GCC_JIT_BINARY_OP_BITWISE_XOR, gcc_jit_binary_op_bitwise_xor);
        (GCC_JIT_BINARY_OP_BITWISE_OR, gcc_jit_binary_op_bitwise_or);
        (GCC_JIT_BINARY_OP_LOGICAL_AND, gcc_jit_binary_op_logical_and);
        (GCC_JIT_BINARY_OP_LOGICAL_OR, gcc_jit_binary_op_logical_or);
        (GCC_JIT_BINARY_OP_LSHIFT, gcc_jit_binary_op_lshift);
        (GCC_JIT_BINARY_OP_RSHIFT, gcc_jit_binary_op_rshift);
      ]

  let gcc_jit_comparison_eq = T.constant "GCC_JIT_COMPARISON_EQ" T.int64_t
  let gcc_jit_comparison_ne = T.constant "GCC_JIT_COMPARISON_NE" T.int64_t
  let gcc_jit_comparison_lt = T.constant "GCC_JIT_COMPARISON_LT" T.int64_t
  let gcc_jit_comparison_le = T.constant "GCC_JIT_COMPARISON_LE" T.int64_t
  let gcc_jit_comparison_gt = T.constant "GCC_JIT_COMPARISON_GT" T.int64_t
  let gcc_jit_comparison_ge = T.constant "GCC_JIT_COMPARISON_GE" T.int64_t

  let gcc_jit_comparison =
    T.enum "gcc_jit_comparison"
      [
        (GCC_JIT_COMPARISON_EQ, gcc_jit_comparison_eq);
        (GCC_JIT_COMPARISON_NE, gcc_jit_comparison_ne);
        (GCC_JIT_COMPARISON_LT, gcc_jit_comparison_lt);
        (GCC_JIT_COMPARISON_LE, gcc_jit_comparison_le);
        (GCC_JIT_COMPARISON_GT, gcc_jit_comparison_gt);
        (GCC_JIT_COMPARISON_GE, gcc_jit_comparison_ge);
      ]
end
