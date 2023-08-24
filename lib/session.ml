(** Managing a computation session. *)

open Base
module Nd = Arrayjit.Ndarray
module LA = Arrayjit.Low_level.Lazy_array
module CDSL = Arrayjit.Low_level.CDSL
open Arrayjit

(** *** Session management. *** *)
type backend = Gccjit | Cuda [@@deriving sexp, equal]

let exec = ref Exec_as_gccjit.jit
let cleanup_executor_session = ref Exec_as_gccjit.cleanup_session

let set_executor = function
  | Gccjit ->
      exec := Exec_as_gccjit.jit;
      cleanup_executor_session := Exec_as_gccjit.cleanup_session
  | Cuda ->
      exec := Exec_as_cuda.jit;
      cleanup_executor_session := Exec_as_cuda.cleanup_session

let compile_routine ~name code =
  !exec ~name @@ Low_level.compile_proc ~name ~for_step_update:false @@ High_level.to_low_level code


let value_1d_points ?from_axis ~xdim t =
  Option.value_map ~default:[||] ~f:(fun arr -> Nd.retrieve_1d_points ?from_axis ~xdim arr)
  @@ Lazy.force t.Tensor.value.array

let value_2d_points ?from_axis ~xdim ~ydim t =
  Option.value_map ~default:[||] ~f:(fun arr -> Nd.retrieve_2d_points ?from_axis ~xdim ~ydim arr)
  @@ Lazy.force t.Tensor.value.array

let grad_1d_points ?from_axis ~xdim t =
  match t.Tensor.diff with
  | None -> [||]
  | Some diff ->
      Option.value_map ~default:[||] ~f:(fun arr -> Nd.retrieve_1d_points ?from_axis ~xdim arr)
      @@ Lazy.force diff.grad.array

let grad_2d_points ?from_axis ~xdim ~ydim t =
  match t.Tensor.diff with
  | None -> [||]
  | Some diff ->
      Option.value_map ~default:[||] ~f:(fun arr -> Nd.retrieve_2d_points ?from_axis ~xdim ~ydim arr)
      @@ Lazy.force diff.grad.array

let set_value t = Nd.set_from_float @@ Option.value_exn @@ Lazy.force t.Tensor.value.array
let get_value t = Nd.get_as_float @@ Option.value_exn @@ Lazy.force t.Tensor.value.array

module O = struct
  (** Get the value at the given indices. *)
  let ( .@{} ) = get_value

  (** Set the value at the given indices. *)
  let ( .@{}<- ) = set_value

  (** Get the value at the given index from a single-axis shape tensor. *)
  let ( .@[] ) t indx = get_value t [| indx |]

  (** Set the value at the given index for a single-axis shape tensor. *)
  let ( .@[]<- ) t indx = set_value t [| indx |]
end

module SDSL = struct
  type nonrec backend = backend = Gccjit | Cuda

  module O = O

  let set_executor = set_executor

  (*
  let refresh_session = refresh_session
  let drop_session = drop_session
  let drop_all_sessions = drop_all_sessions
  let close_session = close_session
  let session_params = session_params
  let minus_learning_rate = minus_learning_rate
  *)

  let compile_routine = compile_routine

  let max_sublabel_length = Tensor.max_sublabel_length
  let print_tensor = Tensor.print
  let print_forward_roots = Tensor.print_forward_roots
  let print_tensor_preamble t = Stdio.print_endline @@ Tensor.header t
  let print_decimals_precision = Nd.print_decimals_precision
  let set_values t cs = Nd.(init (Constant_fill cs) @@ Option.value_exn @@ Lazy.force t.Tensor.value.array)

  let set_fully_on_host t =
    t.Tensor.value.never_virtual <- true;
    t.Tensor.value.never_device_only <- true;
    Option.iter t.diff ~f:(fun diff ->
        diff.grad.never_virtual <- true;
        diff.grad.never_device_only <- true)

  let enable_all_debugs ?(trace_interpreter = false) ?(hosted_only = true) () =
    Low_level.CDSL.with_debug := true;
    Low_level.CDSL.keep_files_in_run_directory := true;
    if hosted_only then Low_level.CDSL.virtualize_settings.enable_device_only <- false;
    if trace_interpreter then Low_level.CDSL.debug_verbose_trace := true

  let disable_all_debugs ?(restore_defaults = false) () =
    Low_level.CDSL.debug_verbose_trace := false;
    Low_level.CDSL.with_debug := false;
    Low_level.CDSL.keep_files_in_run_directory := false;
    if restore_defaults then Low_level.CDSL.virtualize_settings.enable_device_only <- true

  let default_value_prec = Tensor.default_value_prec
  let default_grad_prec = Tensor.default_grad_prec
end
