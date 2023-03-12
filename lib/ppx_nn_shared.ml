open Base

open Ppxlib

let rec pat2expr pat =
  let loc = pat.ppat_loc in
  match pat.ppat_desc with
  | Ppat_constraint (pat', typ) ->
    Ast_builder.Default.pexp_constraint ~loc (pat2expr pat') typ
  | Ppat_alias (_, ident)
  | Ppat_var ident ->
    Ast_builder.Default.pexp_ident ~loc {ident with txt = Lident ident.txt}
  | _ ->
     Ast_builder.Default.pexp_extension ~loc @@ Location.error_extensionf ~loc
       "ppx_ocannl requires a pattern identifier here: try using an `as` alias."

let rec pat2pat_ref pat =
  let loc = pat.ppat_loc in
  match pat.ppat_desc with
  | Ppat_constraint (pat', _) -> pat2pat_ref pat'
  | Ppat_alias (_, ident)
  | Ppat_var ident -> Ast_builder.Default.ppat_var ~loc {ident with txt = ident.txt ^ "__ref"}
  | _ ->
    Ast_builder.Default.ppat_extension ~loc @@ Location.error_extensionf ~loc
      "ppx_ocannl requires a pattern identifier here: try using an `as` alias."

let rec collect_list accu = function
  | [%expr [%e? hd] :: [%e? tl]] -> collect_list (hd::accu) tl
  | [%expr []] -> List.rev accu
  | expr -> List.rev (expr::accu)

let dim_spec_to_string = function
| `Input_dims dim -> "input (tuple) of dim "^Int.to_string dim
| `Output_dims dim -> "output (list) of dim "^Int.to_string dim
| `Batch_dims dim -> "batch (array) of dim "^Int.to_string dim

let ndarray_constant ?axis_labels ?label expr =
  let loc = expr.pexp_loc in
  (* Traverse the backbone of the ndarray to collect the dimensions. *)
  let rec loop_dims accu = function
    | { pexp_desc = Pexp_tuple (exp::_ as exps); _ } -> loop_dims (`Input_dims (List.length exps)::accu) exp
    | { pexp_desc = Pexp_array (exp::_ as exps); _ } -> loop_dims (`Batch_dims (List.length exps)::accu) exp
    | { pexp_desc = Pexp_tuple []; _ } -> `Input_dims 0::accu
    | { pexp_desc = Pexp_array []; _ } -> `Batch_dims 0::accu
    | { pexp_desc = Pexp_construct ({txt=Lident "::"; _}, _); _ } as expr ->
      let exps = collect_list [] expr in
      (match exps with
       | exp::_ -> loop_dims (`Output_dims (List.length exps)::accu) exp
       | [] -> `Output_dims 0::accu)
    | _ -> accu in
  let dims_spec = Array.of_list_rev @@ loop_dims [] expr in
  let open Ast_builder.Default in
  let rec loop_values depth accu expr =
    if depth >= Array.length dims_spec
    then match expr with
      | { pexp_desc = Pexp_constant (Pconst_float _); _ } -> expr::accu
      | { pexp_desc = Pexp_constant (Pconst_integer _); _ } ->
        [%expr Float.of_int [%e expr]]::accu
      | { pexp_desc = Pexp_tuple _; pexp_loc=loc; _ } ->
        (pexp_extension ~loc
         @@ Location.error_extensionf ~loc
           "OCaNNL: ndarray literal found input axis (tuple), expected number")::accu
      | { pexp_desc = Pexp_array _; pexp_loc=loc; _ } -> 
        (pexp_extension ~loc
         @@ Location.error_extensionf ~loc
           "OCaNNL: ndarray literal found batch axis (array), expected number")::accu
      | { pexp_desc = Pexp_construct ({txt=Lident "::"; _}, _); _ } ->
        (pexp_extension ~loc
         @@ Location.error_extensionf ~loc
           "OCaNNL: ndarray literal found output axis (list), expected number")::accu
      | expr -> expr::accu (* it either computes a number, or becomes a type error *)
    else match expr with
      | { pexp_desc = Pexp_tuple exps; _ } ->
        (match dims_spec.(depth) with
         | `Input_dims dim when dim = List.length exps ->
           List.fold_left exps ~init:accu ~f:(loop_values @@ depth + 1)
         | dim_spec ->
           (pexp_extension ~loc
            @@ Location.error_extensionf ~loc
              "OCaNNL: ndarray literal axis mismatch, got %s, expected %s"
              (dim_spec_to_string @@ `Input_dims (List.length exps)) (dim_spec_to_string dim_spec))
           ::accu)
      | { pexp_desc = Pexp_array exps; _ } ->
        (match dims_spec.(depth) with
         | `Batch_dims dim when dim = List.length exps ->
           List.fold_left exps ~init:accu ~f:(loop_values @@ depth + 1)
         | dim_spec ->
           (pexp_extension ~loc
            @@ Location.error_extensionf ~loc
              "OCaNNL: ndarray literal axis mismatch, got %s, expected %s"
              (dim_spec_to_string @@ `Batch_dims (List.length exps)) (dim_spec_to_string dim_spec))
           ::accu)
      | { pexp_desc = Pexp_construct ({txt=Lident "::"; _}, _); _ } ->
        let exps = collect_list [] expr in
        (match dims_spec.(depth) with
         | `Output_dims dim when dim = List.length exps ->
           List.fold_left exps ~init:accu ~f:(loop_values @@ depth + 1)
         | dim_spec ->
           (pexp_extension ~loc
            @@ Location.error_extensionf ~loc
              "OCaNNL: ndarray literal axis mismatch, got %s, expected %s"
              (dim_spec_to_string @@ `Output_dims (List.length exps)) (dim_spec_to_string dim_spec))
           ::accu)
      | { pexp_loc=loc; _ } ->
        (pexp_extension ~loc
         @@ Location.error_extensionf ~loc
           "OCaNNL: ndarray literal: expected an axis (tuple, list or array)")::accu in
  let result = loop_values 0 [] expr in
  let values = {expr with pexp_desc = Pexp_array (List.rev result)} in
  let batch_dims, output_dims, input_dims =
    Array.fold dims_spec ~init:([], [], []) ~f:(fun (batch_dims, output_dims, input_dims) -> 
      function
      | `Input_dims dim -> batch_dims, output_dims, eint ~loc dim::input_dims
      | `Output_dims dim -> batch_dims, eint ~loc dim::output_dims, input_dims
      | `Batch_dims dim -> eint ~loc dim::batch_dims, output_dims, input_dims) in
  let edims dims = elist ~loc @@ List.rev dims in
  let op =
    match axis_labels, label with
    | None, None -> [%expr Operation.ndarray]
    | Some axis_labels, None -> [%expr Operation.ndarray ?axis_labels:[%e axis_labels]]
    | None, Some label -> [%expr Operation.ndarray ?label:[%e label]]
    | Some axis_labels, Some label ->
      [%expr Operation.ndarray ?axis_labels:[%e axis_labels] ?label:[%e label]] in
  [%expr Network.return_term
      ([%e op] ~batch_dims:[%e edims batch_dims] ~input_dims:[%e edims input_dims]
         ~output_dims:[%e edims output_dims] [%e values])]

let let_opt ~loc vbs expr =
  if Map.is_empty vbs then expr
  else Ast_helper.Exp.let_ ~loc Nonrecursive (Map.data vbs) expr

let no_vbs = Map.empty (module String)
let reduce_vbss = List.reduce_exn ~f:(Map.merge_skewed ~combine:(fun ~key:_ _v1 v2 -> v2))
