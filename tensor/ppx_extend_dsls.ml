open Base
open Ppxlib
open Ppx_shared

(* Helper to transform a structure item by opening the DSL operators *)
let transform_dsl_binding ~loc ~dsl_name binding =
  let transform_expr expr =
    let vbs, result =
      Ppx_op.translate ~no_grads_for_inline_defs:true
      @@ add_module_qualifier_to_applied_function ~module_name:dsl_name expr
    in
    if List.is_empty vbs then result
    else
      Ast_builder.Default.pexp_extension ~loc
      @@ Location.error_extensionf ~loc
           "%%extend_dsls functions with inline definitions must take a unit parameter"
  in
  let params, pvb_expr =
    match binding.pvb_expr with
    | { pexp_desc = Pexp_function (params, _, _); _ } as expr -> (params, transform_expr expr)
    | _ -> ([], transform_expr binding.pvb_expr)
  in
  (params, { binding with pvb_expr })

(* Module-level expansion: create module bindings for TDSL, NTDSL, PDSL *)
let str_expander ~loc:pstr_loc ~path:_ str_items =
  let transform_op_binding params binding =
    let loc = binding.pvb_loc in
    let label_p =
      { pparam_loc = loc; pparam_desc = Pparam_val (Optional "label", None, [%pat? label]) }
    in
    let f = function
      | { pparam_desc = Pparam_val (label, _, pat); _ } -> (label, pat2expr pat)
      | _ -> assert false
    in
    let args = List.map params ~f in
    let body =
      Ast_helper.Exp.apply ~loc (pat2expr binding.pvb_pat)
        (args @ [ (Optional "label", [%expr label]); (Nolabel, [%expr ()]) ])
    in
    let pvb_expr =
      {
        binding.pvb_expr with
        pexp_desc = Pexp_function (label_p :: params, None, Pfunction_body body);
      }
    in
    { binding with pvb_expr }
  in
  let items_for_dsl dsl_name =
    let item_bindings, op_item_bindings =
      List.unzip
      @@ List.concat_map str_items ~f:(function
        | { pstr_desc = Pstr_value (Nonrecursive, bindings); pstr_loc = loc; _ } ->
            List.map bindings ~f:(fun binding ->
                let params, binding = transform_dsl_binding ~loc ~dsl_name binding in
                let op_binding = transform_op_binding params binding in
                (binding, op_binding))
        | { pstr_loc = loc; _ } ->
            let pat = Ast_helper.Pat.var ~loc { txt = "syntax_error"; loc } in
            let v =
              Ast_builder.Default.pexp_extension ~loc
              @@ Location.error_extensionf ~loc
                   "ppx_extend_dsls: currently only non-recursive value bindings are supported"
            in
            [ (Ast_helper.Vb.mk ~loc pat v, Ast_helper.Vb.mk ~loc pat v) ])
    in
    let item = { pstr_desc = Pstr_value (Nonrecursive, item_bindings); pstr_loc } in
    let op_item = { pstr_desc = Pstr_value (Nonrecursive, op_item_bindings); pstr_loc } in
    (item, op_item)
  in
  let loc = pstr_loc in
  let item_TDSL, op_item_TDSL = items_for_dsl "TDSL" in
  let item_NTDSL, op_item_NTDSL = items_for_dsl "NTDSL" in
  let item_PDSL, op_item_PDSL = items_for_dsl "PDSL" in
  [%stri
    module DSL_modules = struct
      module Ir = Ir
      module Shape = Shape
      module Tensor = Tensor

      module TDSL = struct
        include TDSL

        [%%i item_TDSL]

        module O = struct
          include TDSL.O

          [%%i op_item_TDSL]
        end
      end

      module NTDSL = struct
        include NTDSL

        [%%i item_NTDSL]

        module O = struct
          include NTDSL.O

          [%%i op_item_NTDSL]
        end
      end

      module PDSL = struct
        include PDSL

        [%%i item_PDSL]

        module O = struct
          include PDSL.O

          [%%i op_item_PDSL]
        end
      end
    end]
