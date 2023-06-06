open Ctypes

type nvrtc_program_t
type nvrtc_program = nvrtc_program_t structure ptr
let nvrtc_program : nvrtc_program typ = ptr (structure "nvrtcProgram")

type nvrtc_result =
  | NVRTC_SUCCESS
  | NVRTC_ERROR_OUT_OF_MEMORY
  | NVRTC_ERROR_PROGRAM_CREATION_FAILURE
  | NVRTC_ERROR_INVALID_INPUT
  | NVRTC_ERROR_INVALID_PROGRAM
  | NVRTC_ERROR_INVALID_OPTION
  | NVRTC_ERROR_COMPILATION
  | NVRTC_ERROR_BUILTIN_OPERATION_FAILURE
  | NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
  | NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
  | NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID
  | NVRTC_ERROR_INTERNAL_ERROR
  (* Only available in recent versions of nvrtc.h: *)
  (* | NVRTC_ERROR_TIME_FILE_WRITE_FAILED *)
  | NVRTC_ERROR_UNCATEGORIZED of int64

module Types (T : Ctypes.TYPE) = struct
  let nvrtc_result_success = T.constant "NVRTC_SUCCESS" T.int64_t
  let nvrtc_result_error_out_of_memory = T.constant "NVRTC_ERROR_OUT_OF_MEMORY" T.int64_t

  let nvrtc_result_error_program_creation_failure =
    T.constant "NVRTC_ERROR_PROGRAM_CREATION_FAILURE" T.int64_t

  let nvrtc_result_error_invalid_input = T.constant "NVRTC_ERROR_INVALID_INPUT" T.int64_t
  let nvrtc_result_error_invalid_program = T.constant "NVRTC_ERROR_INVALID_PROGRAM" T.int64_t
  let nvrtc_result_error_invalid_option = T.constant "NVRTC_ERROR_INVALID_OPTION" T.int64_t
  let nvrtc_result_error_compilation = T.constant "NVRTC_ERROR_COMPILATION" T.int64_t

  let nvrtc_result_error_builtin_operation_failure =
    T.constant "NVRTC_ERROR_BUILTIN_OPERATION_FAILURE" T.int64_t

  let nvrtc_result_error_no_name_expressions_after_compilation =
    T.constant "NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION" T.int64_t

  let nvrtc_result_error_no_lowered_names_before_compilation =
    T.constant "NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION" T.int64_t

  let nvrtc_result_error_name_expression_not_valid =
    T.constant "NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID" T.int64_t

  let nvrtc_result_error_internal_error = T.constant "NVRTC_ERROR_INTERNAL_ERROR" T.int64_t

  (* Only available in recent versions of nvrtc.h: *)
  (* let nvrtc_result_error_time_file_write_failed = T.constant "NVRTC_ERROR_TIME_FILE_WRITE_FAILED" T.int64_t *)

  let nvrtc_result =
    T.enum ~typedef:true
      ~unexpected:(fun error_code -> NVRTC_ERROR_UNCATEGORIZED error_code)
      "nvrtcResult"
      [
        (NVRTC_SUCCESS, nvrtc_result_success);
        (NVRTC_ERROR_OUT_OF_MEMORY, nvrtc_result_error_out_of_memory);
        (NVRTC_ERROR_PROGRAM_CREATION_FAILURE, nvrtc_result_error_program_creation_failure);
        (NVRTC_ERROR_INVALID_INPUT, nvrtc_result_error_invalid_input);
        (NVRTC_ERROR_INVALID_PROGRAM, nvrtc_result_error_invalid_program);
        (NVRTC_ERROR_INVALID_OPTION, nvrtc_result_error_invalid_option);
        (NVRTC_ERROR_COMPILATION, nvrtc_result_error_compilation);
        (NVRTC_ERROR_BUILTIN_OPERATION_FAILURE, nvrtc_result_error_builtin_operation_failure);
        ( NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION,
          nvrtc_result_error_no_name_expressions_after_compilation );
        ( NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION,
          nvrtc_result_error_no_lowered_names_before_compilation );
        (NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID, nvrtc_result_error_name_expression_not_valid);
        (NVRTC_ERROR_INTERNAL_ERROR, nvrtc_result_error_internal_error);
        (* Only available in recent versions of nvrtc.h: *)
        (* (NVRTC_ERROR_TIME_FILE_WRITE_FAILED, nvrtc_result_error_time_file_write_failed); *)
      ]
end
