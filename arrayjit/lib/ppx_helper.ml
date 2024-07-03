open Base
open Ppxlib

type li = longident

let rec collect_list accu = function
  | [%expr [%e? hd] :: [%e? tl]] -> collect_list (hd :: accu) tl
  | [%expr []] -> List.rev accu
  | expr -> List.rev (expr :: accu)

let dim_spec_to_string = function
  | `Input_dims dim -> "input (tuple) of dim " ^ Int.to_string dim
  | `Output_dims dim -> "output (list) of dim " ^ Int.to_string dim
  | `Batch_dims dim -> "batch (array) of dim " ^ Int.to_string dim

let ndarray_constant expr =
  let loc = expr.pexp_loc in
  (* Traverse the backbone of the ndarray to collect the dimensions. *)
  let rec loop_dims accu = function
    | { pexp_desc = Pexp_tuple (exp :: _ as exps); _ } ->
        loop_dims (`Input_dims (List.length exps) :: accu) exp
    | { pexp_desc = Pexp_array (exp :: _ as exps); _ } ->
        loop_dims (`Batch_dims (List.length exps) :: accu) exp
    | { pexp_desc = Pexp_tuple []; _ } -> `Input_dims 0 :: accu
    | { pexp_desc = Pexp_array []; _ } -> `Batch_dims 0 :: accu
    | { pexp_desc = Pexp_construct ({ txt = Lident "::"; _ }, _); _ } as expr -> (
        let exps = collect_list [] expr in
        match exps with
        | exp :: _ -> loop_dims (`Output_dims (List.length exps) :: accu) exp
        | [] -> `Output_dims 0 :: accu)
    | _ -> accu
  in
  let dims_spec = Array.of_list_rev @@ loop_dims [] expr in
  let open Ast_builder.Default in
  let rec loop_values depth accu expr =
    if depth >= Array.length dims_spec then
      match expr with
      | { pexp_desc = Pexp_constant (Pconst_float _); _ } -> expr :: accu
      | { pexp_desc = Pexp_constant (Pconst_integer _); _ } ->
          [%expr Float.of_int [%e expr]] :: accu
      | { pexp_desc = Pexp_tuple _; pexp_loc = loc; _ } ->
          (pexp_extension ~loc
          @@ Location.error_extensionf ~loc
               "Arrayjit: ndarray literal found input axis (tuple), expected number")
          :: accu
      | { pexp_desc = Pexp_array _; pexp_loc = loc; _ } ->
          (pexp_extension ~loc
          @@ Location.error_extensionf ~loc
               "Arrayjit: ndarray literal found batch axis (array), expected number")
          :: accu
      | { pexp_desc = Pexp_construct ({ txt = Lident "::"; _ }, _); _ } ->
          (pexp_extension ~loc
          @@ Location.error_extensionf ~loc
               "Arrayjit: ndarray literal found output axis (list), expected number")
          :: accu
      | expr -> expr :: accu (* it either computes a number, or becomes a type error *)
    else
      match expr with
      | { pexp_desc = Pexp_tuple exps; _ } -> (
          match dims_spec.(depth) with
          | `Input_dims dim when dim = List.length exps ->
              List.fold_left exps ~init:accu ~f:(loop_values @@ (depth + 1))
          | dim_spec ->
              (pexp_extension ~loc
              @@ Location.error_extensionf ~loc
                   "Arrayjit: ndarray literal axis mismatch, got %s, expected %s"
                   (dim_spec_to_string @@ `Input_dims (List.length exps))
                   (dim_spec_to_string dim_spec))
              :: accu)
      | { pexp_desc = Pexp_array exps; _ } -> (
          match dims_spec.(depth) with
          | `Batch_dims dim when dim = List.length exps ->
              List.fold_left exps ~init:accu ~f:(loop_values @@ (depth + 1))
          | dim_spec ->
              (pexp_extension ~loc
              @@ Location.error_extensionf ~loc
                   "Arrayjit: ndarray literal axis mismatch, got %s, expected %s"
                   (dim_spec_to_string @@ `Batch_dims (List.length exps))
                   (dim_spec_to_string dim_spec))
              :: accu)
      | { pexp_desc = Pexp_construct ({ txt = Lident "::"; _ }, _); _ } -> (
          let exps = collect_list [] expr in
          match dims_spec.(depth) with
          | `Output_dims dim when dim = List.length exps ->
              List.fold_left exps ~init:accu ~f:(loop_values @@ (depth + 1))
          | dim_spec ->
              (pexp_extension ~loc
              @@ Location.error_extensionf ~loc
                   "Arrayjit: ndarray literal axis mismatch, got %s, expected %s"
                   (dim_spec_to_string @@ `Output_dims (List.length exps))
                   (dim_spec_to_string dim_spec))
              :: accu)
      | { pexp_loc = loc; _ } ->
          (pexp_extension ~loc
          @@ Location.error_extensionf ~loc
               "Arrayjit: ndarray literal: expected an axis (tuple, list or array)")
          :: accu
  in
  let result = loop_values 0 [] expr in
  let values = { expr with pexp_desc = Pexp_array (List.rev result) } in
  let batch_dims, output_dims, input_dims =
    Array.fold dims_spec ~init:([], [], []) ~f:(fun (batch_dims, output_dims, input_dims) ->
      function
      | `Input_dims dim -> (batch_dims, output_dims, eint ~loc dim :: input_dims)
      | `Output_dims dim -> (batch_dims, eint ~loc dim :: output_dims, input_dims)
      | `Batch_dims dim -> (eint ~loc dim :: batch_dims, output_dims, input_dims))
  in
  (values, List.rev batch_dims, List.rev output_dims, List.rev input_dims)
