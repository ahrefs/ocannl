open Base
open Ppxlib

type li = longident

let pat2string pat =
  let rec lident = function Lident s | Ldot (_, s) -> s | Lapply (_, i) -> lident i in
  let rec loop pat =
    match pat.ppat_desc with
    | Ppat_open (_, pat) | Ppat_lazy pat | Ppat_constraint (pat, _) -> loop pat
    | Ppat_alias (_, ident) -> ident.txt
    | Ppat_var ident -> ident.txt
    | Ppat_any -> "_"
    | Ppat_variant (s, _)
    | Ppat_constant (Pconst_string (s, _, _))
    | Ppat_constant (Pconst_integer (s, _))
    | Ppat_constant (Pconst_float (s, _)) ->
        s
    | Ppat_constant (Pconst_char c) -> Char.to_string c
    | Ppat_tuple pats -> "(" ^ String.concat ~sep:", " (List.map ~f:loop pats) ^ ")"
    | Ppat_array pats -> "[|" ^ String.concat ~sep:", " (List.map ~f:loop pats) ^ "|]"
    | Ppat_construct (c, _) -> lident c.txt
    | Ppat_interval (_, _)
    | Ppat_record (_, _)
    | Ppat_or (_, _)
    | Ppat_type _ | Ppat_unpack _ | Ppat_exception _ | Ppat_extension _ ->
        ""
  in
  Ast_helper.Exp.constant @@ Pconst_string (loop pat, pat.ppat_loc, None)

let opt_pat2string ~loc = function None -> [%expr None] | Some pat -> [%expr Some [%e pat2string pat]]
let opt_expr ~loc = function None -> [%expr None] | Some expr -> [%expr Some [%e expr]]

let rec collect_list accu = function
  | [%expr [%e? hd] :: [%e? tl]] -> collect_list (hd :: accu) tl
  | [%expr []] -> List.rev accu
  | expr -> List.rev (expr :: accu)

let rec pat2expr pat =
  let module Ast = Ast_builder.Default in
  let loc = pat.ppat_loc in
  match pat.ppat_desc with
  | Ppat_constraint (pat', typ) -> Ast.pexp_constraint ~loc (pat2expr pat') typ
  | Ppat_alias (_, ident) | Ppat_var ident -> Ast.pexp_ident ~loc { ident with txt = Lident ident.txt }
  | Ppat_variant (ident, e_opt) -> Ast.pexp_variant ~loc ident @@ Option.map e_opt ~f:pat2expr
  | Ppat_constant c -> Ast.pexp_constant ~loc c
  | Ppat_construct (c, None) -> Ast.pexp_construct ~loc c None
  | Ppat_construct (c, Some ([], args)) -> Ast.pexp_construct ~loc c @@ Some (pat2expr args)
  | Ppat_record (fields, Asttypes.Closed) ->
      Ast.pexp_record ~loc (List.map fields ~f:(fun (label, field) -> (label, pat2expr field))) None
  | Ppat_tuple pats -> Ast.pexp_tuple ~loc @@ List.map pats ~f:pat2expr
  | Ppat_array pats -> Ast.pexp_array ~loc @@ List.map pats ~f:pat2expr
  | _ ->
      Ast.pexp_extension ~loc
      @@ Location.error_extensionf ~loc
           "ppx_ocannl does not recognize/support the pattern; maybe try using an `as` alias."

let dim_spec_to_string = function
  | `Input_dims dim -> "input (tuple) of dim " ^ Int.to_string dim
  | `Output_dims dim -> "output (list) of dim " ^ Int.to_string dim
  | `Batch_dims dim -> "batch (array) of dim " ^ Int.to_string dim

let alphanum_regexp = Str.regexp "^[^a-zA-Z0-9]+$"
let is_operator ident = Str.string_match alphanum_regexp ident 0

let is_assignment ident =
  String.length ident > 1
  && Char.equal ident.[0] '='
  && (not @@ List.mem [ "=="; "==="; "=>"; "==>"; "=>>" ] ident ~equal:String.equal)

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
      | { pexp_desc = Pexp_constant (Pconst_integer _); _ } -> [%expr Float.of_int [%e expr]] :: accu
      | { pexp_desc = Pexp_tuple _; pexp_loc = loc; _ } ->
          (pexp_extension ~loc
          @@ Location.error_extensionf ~loc
               "OCANNL: ndarray literal found input axis (tuple), expected number")
          :: accu
      | { pexp_desc = Pexp_array _; pexp_loc = loc; _ } ->
          (pexp_extension ~loc
          @@ Location.error_extensionf ~loc
               "OCANNL: ndarray literal found batch axis (array), expected number")
          :: accu
      | { pexp_desc = Pexp_construct ({ txt = Lident "::"; _ }, _); _ } ->
          (pexp_extension ~loc
          @@ Location.error_extensionf ~loc
               "OCANNL: ndarray literal found output axis (list), expected number")
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
              @@ Location.error_extensionf ~loc "OCANNL: ndarray literal axis mismatch, got %s, expected %s"
                   (dim_spec_to_string @@ `Input_dims (List.length exps))
                   (dim_spec_to_string dim_spec))
              :: accu)
      | { pexp_desc = Pexp_array exps; _ } -> (
          match dims_spec.(depth) with
          | `Batch_dims dim when dim = List.length exps ->
              List.fold_left exps ~init:accu ~f:(loop_values @@ (depth + 1))
          | dim_spec ->
              (pexp_extension ~loc
              @@ Location.error_extensionf ~loc "OCANNL: ndarray literal axis mismatch, got %s, expected %s"
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
              @@ Location.error_extensionf ~loc "OCANNL: ndarray literal axis mismatch, got %s, expected %s"
                   (dim_spec_to_string @@ `Output_dims (List.length exps))
                   (dim_spec_to_string dim_spec))
              :: accu)
      | { pexp_loc = loc; _ } ->
          (pexp_extension ~loc
          @@ Location.error_extensionf ~loc "OCANNL: ndarray literal: expected an axis (tuple, list or array)"
          )
          :: accu
  in
  let result = loop_values 0 [] expr in
  let values = { expr with pexp_desc = Pexp_array (List.rev result) } in
  let batch_dims, output_dims, input_dims =
    Array.fold dims_spec ~init:([], [], []) ~f:(fun (batch_dims, output_dims, input_dims) -> function
      | `Input_dims dim -> (batch_dims, output_dims, eint ~loc dim :: input_dims)
      | `Output_dims dim -> (batch_dims, eint ~loc dim :: output_dims, input_dims)
      | `Batch_dims dim -> (eint ~loc dim :: batch_dims, output_dims, input_dims))
  in
  let to_dim dims = List.rev_map dims ~f:(fun d -> [%expr Shape.Dim [%e d]]) in
  (values, to_dim batch_dims, to_dim output_dims, to_dim input_dims)

let convert_dsl_dims dims =
  List.map dims ~f:(function
    | { pexp_desc = Pexp_constant (Pconst_integer _); pexp_loc = loc; _ } as i -> [%expr Shape.Dim [%e i]]
    | { pexp_desc = Pexp_ident ({txt = Lident "parallel"; loc}); pexp_loc = _; _ } -> [%expr Shape.Parallel]
    | e -> e)

let let_opt ~loc vbs expr =
  if Map.is_empty vbs then expr else Ast_helper.Exp.let_ ~loc Nonrecursive (Map.data vbs) expr

let no_vbs = Map.empty (module String)
let reduce_vbss = List.reduce_exn ~f:(Map.merge_skewed ~combine:(fun ~key:_ _v1 v2 -> v2))
