open Base
open Ppxlib

type li = longident

let rec collect_list accu = function
  | [%expr [%e? hd] :: [%e? tl]] -> collect_list (hd :: accu) tl
  | [%expr []] -> List.rev accu
  | expr ->
      let error_expr =
        Ast_builder.Default.pexp_extension ~loc:expr.pexp_loc
        @@ Location.error_extensionf ~loc:expr.pexp_loc
             "Arrayjit: expected a list literal, cannot parse a non-literal list value"
      in
      List.rev (error_expr :: accu)

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
    Array.fold dims_spec ~init:([], [], [])
      ~f:(fun (batch_dims, output_dims, input_dims) -> function
      | `Input_dims dim -> (batch_dims, output_dims, eint ~loc dim :: input_dims)
      | `Output_dims dim -> (batch_dims, eint ~loc dim :: output_dims, input_dims)
      | `Batch_dims dim -> (eint ~loc dim :: batch_dims, output_dims, input_dims))
  in
  (values, List.rev batch_dims, List.rev output_dims, List.rev input_dims)

(** Convert a string containing patterns like "identifier*" to an OCaml expression that substitutes
    the identifiers with their runtime values. Identifiers match the pattern [a-z_][a-z0-9_]* and
    must directly precede '*'.

    Example usage: [substitute_identifiers_in_string ~loc "a *x + b * y"] generates an expression
    equivalent to: [String.concat "" [Int.to_string a; " *x + "; Int.to_string b; " * y"]]

    So if [a = 2] and [b = 3], the result would be ["2 *x + 3 * y"]. Whitespace between identifiers
    and '*' is preserved. *)
let substitute_identifiers_in_einsum_spec ~loc str_input =
  let multichar = String.contains str_input ',' in
  let open Ast_builder.Default in
  (* Helper to check if character is valid for identifier start *)
  let is_identifier_start c = Char.is_alpha c || Char.equal c '_' in

  (* Helper to check if character is valid for identifier continuation *)
  let is_identifier_char c = Char.is_alphanum c || Char.equal c '_' in

  (* Find all identifier* patterns and their positions using forward scanning *)
  let len = String.length str_input in
  let substitutions = ref [] in

  let i = ref 0 in
  while !i < len do
    let c = str_input.[!i] in
    if is_identifier_start c then (
      (* Found start of potential identifier *)
      let start_pos = !i in
      (* Scan forward to find end of identifier *)
      while !i < len && is_identifier_char str_input.[!i] && (multichar || !i = start_pos) do
        i := !i + 1
      done;
      let end_pos = !i - 1 in

      (* Skip any whitespace after identifier *)
      while !i < len && List.mem ~equal:Char.equal [ ' '; '\t'; '\n'; '\r' ] str_input.[!i] do
        i := !i + 1
      done;

      (* Check if followed by '*' *)
      if !i < len && Char.equal str_input.[!i] '*' then
        let identifier = String.sub str_input ~pos:start_pos ~len:(end_pos - start_pos + 1) in
        substitutions := (start_pos, end_pos, identifier) :: !substitutions)
    else i := !i + 1
  done;

  let substitutions = List.rev !substitutions in

  (* Build segments by splitting the string at substitution boundaries *)
  let segments = ref [] in
  let pos = ref 0 in

  List.iter substitutions ~f:(fun (start_pos, end_pos, identifier) ->
      (* Add literal segment before substitution *)
      (if start_pos > !pos then
         let literal = String.sub str_input ~pos:!pos ~len:(start_pos - !pos) in
         segments := estring ~loc literal :: !segments);

      (* Add substitution marker *)
      segments :=
        [%expr Int.to_string [%e pexp_ident ~loc (Located.mk ~loc (Lident identifier))]]
        :: !segments;

      (* Move position past the '*' *)
      pos := end_pos + 1);

  (* Add final literal segment *)
  (if !pos < len then
     let literal = String.sub str_input ~pos:!pos ~len:(len - !pos) in
     segments := estring ~loc literal :: !segments);

  let segments = List.rev !segments in

  (* Generate expression to concatenate all segments *)
  match segments with
  | [] -> estring ~loc ""
  | [ single ] -> single
  | multiple -> [%expr String.concat ~sep:"" [%e elist ~loc multiple]]
