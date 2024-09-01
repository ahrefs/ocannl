[@@@ocaml.ppx.context
  {
    tool_name = "ppx_driver";
    include_dirs = [];
    load_path = [];
    open_modules = [];
    for_package = None;
    debug = false;
    use_threads = false;
    use_vmthreads = false;
    recursive_types = false;
    principal = false;
    transparent_modules = false;
    unboxed_types = false;
    unsafe_string = false;
    cookies = [("library-name", "ppx_ocannl")]
  }]
let () =
  Ppx_module_timer_runtime.record_start Ppx_module_timer_runtime.__MODULE__
let () = Ppx_bench_lib.Benchmark_accumulator.Current_libname.set "ppx_ocannl"
let () =
  Ppx_expect_runtime.Current_file.set
    ~filename_rel_to_project_root:"lib/ppx_op.ml"
let () = Ppx_inline_test_lib.set_lib_and_partition "ppx_ocannl" "ppx_op.ml"
open Base
open Ppxlib
open Ppx_arrayjit.Ppx_helper
open Ppx_shared
let ndarray_op ?label  ?axis_labels  expr =
  let loc = expr.pexp_loc in
  let (values, batch_dims, output_dims, input_dims) = ndarray_constant expr in
  let edims dims = Ast_builder.Default.elist ~loc dims in
  let op =
    match axis_labels with
    | None ->
        ({
           pexp_desc =
             (Pexp_ident { txt = (Ldot ((Lident "TDSL"), "ndarray")); loc });
           pexp_loc = loc;
           pexp_loc_stack = [];
           pexp_attributes = []
         } : Ppxlib_ast.Ast.expression)
    | Some axis_labels ->
        ({
           pexp_desc =
             (Pexp_apply
                ({
                   pexp_desc =
                     (Pexp_ident
                        { txt = (Ldot ((Lident "TDSL"), "ndarray")); loc });
                   pexp_loc = loc;
                   pexp_loc_stack = [];
                   pexp_attributes = []
                 },
                  [((Labelled "axis_labels"),
                     (axis_labels : Ppxlib_ast.Ast.expression))]));
           pexp_loc = loc;
           pexp_loc_stack = [];
           pexp_attributes = []
         } : Ppxlib_ast.Ast.expression) in
  ({
     pexp_desc =
       (Pexp_apply
          ((op : Ppxlib_ast.Ast.expression),
            [((Optional "label"),
               (opt_expr ~loc label : Ppxlib_ast.Ast.expression));
            ((Labelled "batch_dims"),
              (edims batch_dims : Ppxlib_ast.Ast.expression));
            ((Labelled "input_dims"),
              (edims input_dims : Ppxlib_ast.Ast.expression));
            ((Labelled "output_dims"),
              (edims output_dims : Ppxlib_ast.Ast.expression));
            (Nolabel, (values : Ppxlib_ast.Ast.expression))]));
     pexp_loc = loc;
     pexp_loc_stack = [];
     pexp_attributes = []
   } : Ppxlib_ast.Ast.expression)
let make_p ~has_config  ~loc  =
  if has_config
  then
    ({
       pexp_desc =
         (Pexp_apply
            ({
               pexp_desc =
                 (Pexp_ident { txt = (Ldot ((Lident "TDSL"), "param")); loc });
               pexp_loc = loc;
               pexp_loc_stack = [];
               pexp_attributes = []
             },
              [((Labelled "more_label"),
                 {
                   pexp_desc =
                     (Pexp_field
                        ({
                           pexp_desc =
                             (Pexp_ident { txt = (Lident "config"); loc });
                           pexp_loc = loc;
                           pexp_loc_stack = [];
                           pexp_attributes = []
                         }, { txt = (Lident "label"); loc }));
                   pexp_loc = loc;
                   pexp_loc_stack = [];
                   pexp_attributes = []
                 })]));
       pexp_loc = loc;
       pexp_loc_stack = [];
       pexp_attributes = []
     } : Ppxlib_ast.Ast.expression)
  else
    ({
       pexp_desc =
         (Pexp_ident { txt = (Ldot ((Lident "TDSL"), "param")); loc });
       pexp_loc = loc;
       pexp_loc_stack = [];
       pexp_attributes = []
     } : Ppxlib_ast.Ast.expression)
let make_vb ?value  ~has_config  ~loc  ~str_loc  ~ident  string =
  let pat = Ast_helper.Pat.var ~loc { loc = str_loc; txt = ident } in
  let value =
    match value with
    | Some c ->
        ({
           pexp_desc =
             (Pexp_construct
                ({ txt = (Lident "Some"); loc },
                  (Some (c : Ppxlib_ast.Ast.expression))));
           pexp_loc = loc;
           pexp_loc_stack = [];
           pexp_attributes = []
         } : Ppxlib_ast.Ast.expression)
    | None ->
        ({
           pexp_desc =
             (Pexp_construct ({ txt = (Lident "None"); loc }, None));
           pexp_loc = loc;
           pexp_loc_stack = [];
           pexp_attributes = []
         } : Ppxlib_ast.Ast.expression) in
  let v =
    ({
       pexp_desc =
         (Pexp_apply
            ((make_p ~has_config ~loc : Ppxlib_ast.Ast.expression),
              [((Optional "values"), (value : Ppxlib_ast.Ast.expression));
              (Nolabel, (string : Ppxlib_ast.Ast.expression))]));
       pexp_loc = loc;
       pexp_loc_stack = [];
       pexp_attributes = []
     } : Ppxlib_ast.Ast.expression) in
  let vb = Ast_helper.Vb.mk ~loc pat v in (pat, vb)
let make_vb_dims ~has_config  ~loc  ~str_loc  ~ident  ~dims  ~dims_loc 
  string =
  let pat = Ast_helper.Pat.var ~loc { loc = str_loc; txt = ident } in
  let dims =
    let loc = dims_loc in
    List.fold_right dims
      ~init:({
               pexp_desc =
                 (Pexp_construct ({ txt = (Lident "[]"); loc }, None));
               pexp_loc = loc;
               pexp_loc_stack = [];
               pexp_attributes = []
             } : Ppxlib_ast.Ast.expression)
      ~f:(fun d ->
            fun ds ->
              ({
                 pexp_desc =
                   (Pexp_construct
                      ({ txt = (Lident "::"); loc },
                        (Some
                           {
                             pexp_desc =
                               (Pexp_tuple
                                  [(d : Ppxlib_ast.Ast.expression);
                                  (ds : Ppxlib_ast.Ast.expression)]);
                             pexp_loc = loc;
                             pexp_loc_stack = [];
                             pexp_attributes = []
                           })));
                 pexp_loc = loc;
                 pexp_loc_stack = [];
                 pexp_attributes = []
               } : Ppxlib_ast.Ast.expression)) in
  let v =
    ({
       pexp_desc =
         (Pexp_apply
            ((make_p ~has_config ~loc : Ppxlib_ast.Ast.expression),
              [((Labelled "output_dims"), (dims : Ppxlib_ast.Ast.expression));
              (Nolabel, (string : Ppxlib_ast.Ast.expression))]));
       pexp_loc = loc;
       pexp_loc_stack = [];
       pexp_attributes = []
     } : Ppxlib_ast.Ast.expression) in
  let vb = Ast_helper.Vb.mk ~loc pat v in (pat, vb)
let make_vb_nd ~has_config  ~loc  ~str_loc  ?axis_labels  ~ident  ~init_nd 
  string =
  let pat = Ast_helper.Pat.var ~loc { loc = str_loc; txt = ident } in
  let (values, batch_dims, output_dims, input_dims) =
    ndarray_constant init_nd in
  let v =
    if not @@ (List.is_empty batch_dims)
    then
      (Ast_builder.Default.pexp_extension ~loc) @@
        (Location.error_extensionf ~loc
           "ppx_ocannl param cannot have batch dims: define a constant or remove the array syntax.")
    else
      (let edims dims = Ast_builder.Default.elist ~loc dims in
       let op =
         match axis_labels with
         | None -> make_p ~has_config ~loc
         | Some axis_labels ->
             ({
                pexp_desc =
                  (Pexp_apply
                     ((make_p ~has_config ~loc : Ppxlib_ast.Ast.expression),
                       [((Labelled "axis_labels"),
                          (axis_labels : Ppxlib_ast.Ast.expression))]));
                pexp_loc = loc;
                pexp_loc_stack = [];
                pexp_attributes = []
              } : Ppxlib_ast.Ast.expression) in
       ({
          pexp_desc =
            (Pexp_apply
               ((op : Ppxlib_ast.Ast.expression),
                 [((Labelled "input_dims"),
                    (edims input_dims : Ppxlib_ast.Ast.expression));
                 ((Labelled "output_dims"),
                   (edims output_dims : Ppxlib_ast.Ast.expression));
                 ((Labelled "values"), (values : Ppxlib_ast.Ast.expression));
                 (Nolabel, (string : Ppxlib_ast.Ast.expression))]));
          pexp_loc = loc;
          pexp_loc_stack = [];
          pexp_attributes = []
        } : Ppxlib_ast.Ast.expression)) in
  let vb = Ast_helper.Vb.mk ~loc pat v in (pat, vb)
let lift_config_vb ~loop  ~num_configs  ?label  ~expr1  ~c_expr  arg_exprs =
  let (vbs1, e1) = loop ?label expr1 in
  let (vbss, es) = List.unzip @@ (List.map arg_exprs ~f:loop) in
  let ident = "config_block__" ^ (Int.to_string (!num_configs)) in
  Int.incr num_configs;
  (let loc = expr1.pexp_loc in
   let pat = Ast_helper.Pat.var ~loc { loc = (c_expr.pexp_loc); txt = ident } in
   let v =
     ({
        pexp_desc =
          (Pexp_apply
             ((e1 : Ppxlib_ast.Ast.expression),
               [((Labelled "config"), (c_expr : Ppxlib_ast.Ast.expression))]));
        pexp_loc = loc;
        pexp_loc_stack = [];
        pexp_attributes = []
      } : Ppxlib_ast.Ast.expression) in
   let vb = Ast_helper.Vb.mk ~loc pat v in
   (((Map.add_exn ~key:ident ~data:vb) @@ (reduce_vbss (vbs1 :: vbss))),
     (match es with
      | [] ->
          ((pat2expr pat : Ppxlib_ast.Ast.expression) : Ppxlib_ast.Ast.expression)
      | e2::[] ->
          ({
             pexp_desc =
               (Pexp_apply
                  ((pat2expr pat : Ppxlib_ast.Ast.expression),
                    [(Nolabel, (e2 : Ppxlib_ast.Ast.expression))]));
             pexp_loc = loc;
             pexp_loc_stack = [];
             pexp_attributes = []
           } : Ppxlib_ast.Ast.expression)
      | e2::e3::[] ->
          ({
             pexp_desc =
               (Pexp_apply
                  ((pat2expr pat : Ppxlib_ast.Ast.expression),
                    [(Nolabel, (e2 : Ppxlib_ast.Ast.expression));
                    (Nolabel, (e3 : Ppxlib_ast.Ast.expression))]));
             pexp_loc = loc;
             pexp_loc_stack = [];
             pexp_attributes = []
           } : Ppxlib_ast.Ast.expression)
      | _ -> assert false)))
let rec translate ~num_configs  ~is_toplevel  ~has_config  ?label  expr =
  let loc = expr.pexp_loc in
  let loop = translate ~num_configs ~is_toplevel:false ~has_config in
  match expr with
  | { pexp_desc = Pexp_constant (Pconst_float _);_} ->
      (no_vbs,
        ({
           pexp_desc =
             (Pexp_apply
                ({
                   pexp_desc =
                     (Pexp_ident
                        { txt = (Ldot ((Lident "TDSL"), "number")); loc });
                   pexp_loc = loc;
                   pexp_loc_stack = [];
                   pexp_attributes = []
                 },
                  [((Optional "label"),
                     (opt_expr ~loc label : Ppxlib_ast.Ast.expression));
                  (Nolabel, (expr : Ppxlib_ast.Ast.expression))]));
           pexp_loc = loc;
           pexp_loc_stack = [];
           pexp_attributes = []
         } : Ppxlib_ast.Ast.expression))
  | { pexp_desc = Pexp_constant (Pconst_integer _);_} ->
      (no_vbs,
        ({
           pexp_desc =
             (Pexp_apply
                ({
                   pexp_desc =
                     (Pexp_ident
                        { txt = (Ldot ((Lident "TDSL"), "number")); loc });
                   pexp_loc = loc;
                   pexp_loc_stack = [];
                   pexp_attributes = []
                 },
                  [(Nolabel,
                     {
                       pexp_desc =
                         (Pexp_apply
                            ({
                               pexp_desc =
                                 (Pexp_ident
                                    {
                                      txt =
                                        (Ldot ((Lident "Float"), "of_int"));
                                      loc
                                    });
                               pexp_loc = loc;
                               pexp_loc_stack = [];
                               pexp_attributes = []
                             },
                              [(Nolabel, (expr : Ppxlib_ast.Ast.expression))]));
                       pexp_loc = loc;
                       pexp_loc_stack = [loc];
                       pexp_attributes = []
                     })]));
           pexp_loc = loc;
           pexp_loc_stack = [];
           pexp_attributes = []
         } : Ppxlib_ast.Ast.expression))
  | ({
       pexp_desc = Pexp_apply
         (({ pexp_desc = Pexp_constant (Pconst_char ch); pexp_loc;_} :
            Ppxlib_ast.Ast.expression),
          (Nolabel,
           (({ pexp_desc = Pexp_constant (Pconst_float _);_} as f) :
             Ppxlib_ast.Ast.expression))::[]);
       pexp_loc = _; pexp_loc_stack = _; pexp_attributes = _ }
      : Ppxlib_ast.Ast.expression) ->
      let axis =
        Ast_helper.Exp.constant ~loc:pexp_loc
          (Pconst_string ((String.of_char ch), pexp_loc, None)) in
      (no_vbs,
        ({
           pexp_desc =
             (Pexp_apply
                ({
                   pexp_desc =
                     (Pexp_ident
                        { txt = (Ldot ((Lident "TDSL"), "number")); loc });
                   pexp_loc = loc;
                   pexp_loc_stack = [];
                   pexp_attributes = []
                 },
                  [((Optional "label"),
                     (opt_expr ~loc label : Ppxlib_ast.Ast.expression));
                  ((Labelled "axis_label"),
                    (axis : Ppxlib_ast.Ast.expression));
                  (Nolabel, (f : Ppxlib_ast.Ast.expression))]));
           pexp_loc = loc;
           pexp_loc_stack = [];
           pexp_attributes = []
         } : Ppxlib_ast.Ast.expression))
  | ({
       pexp_desc = Pexp_apply
         (({ pexp_desc = Pexp_constant (Pconst_char ch); pexp_loc;_} :
            Ppxlib_ast.Ast.expression),
          (Nolabel,
           (({ pexp_desc = Pexp_constant (Pconst_integer _);_} as i) :
             Ppxlib_ast.Ast.expression))::[]);
       pexp_loc = _; pexp_loc_stack = _; pexp_attributes = _ }
      : Ppxlib_ast.Ast.expression) ->
      let axis =
        Ast_helper.Exp.constant ~loc:pexp_loc
          (Pconst_string ((String.of_char ch), pexp_loc, None)) in
      (no_vbs,
        ({
           pexp_desc =
             (Pexp_apply
                ({
                   pexp_desc =
                     (Pexp_ident
                        { txt = (Ldot ((Lident "TDSL"), "number")); loc });
                   pexp_loc = loc;
                   pexp_loc_stack = [];
                   pexp_attributes = []
                 },
                  [((Optional "label"),
                     (opt_expr ~loc label : Ppxlib_ast.Ast.expression));
                  ((Labelled "axis_label"),
                    (axis : Ppxlib_ast.Ast.expression));
                  (Nolabel,
                    {
                      pexp_desc =
                        (Pexp_apply
                           ({
                              pexp_desc =
                                (Pexp_ident
                                   {
                                     txt =
                                       (Ldot ((Lident "Float"), "of_int"));
                                     loc
                                   });
                              pexp_loc = loc;
                              pexp_loc_stack = [];
                              pexp_attributes = []
                            }, [(Nolabel, (i : Ppxlib_ast.Ast.expression))]));
                      pexp_loc = loc;
                      pexp_loc_stack = [loc];
                      pexp_attributes = []
                    })]));
           pexp_loc = loc;
           pexp_loc_stack = [];
           pexp_attributes = []
         } : Ppxlib_ast.Ast.expression))
  | ({
       pexp_desc = Pexp_apply
         ({ pexp_desc = Pexp_ident { txt = Lident "*+"; loc = _ };
            pexp_loc = _; pexp_loc_stack = _; pexp_attributes = _ },
          (Nolabel, (expr1 : Ppxlib_ast.Ast.expression))::(Nolabel,
                                                           {
                                                             pexp_desc =
                                                               Pexp_apply
                                                               ((({
                                                                    pexp_desc
                                                                    =
                                                                    Pexp_constant
                                                                    (Pconst_string
                                                                    (spec_str,
                                                                    _, _));_}
                                                                    as spec)
                                                                  :
                                                                  Ppxlib_ast.Ast.expression),
                                                                (Nolabel,
                                                                 (expr2 :
                                                                   Ppxlib_ast.Ast.expression))::[]);
                                                             pexp_loc = _;
                                                             pexp_loc_stack =
                                                               _;
                                                             pexp_attributes
                                                               = _
                                                             })::[]);
       pexp_loc = _; pexp_loc_stack = _; pexp_attributes = _ }
      : Ppxlib_ast.Ast.expression) when String.contains spec_str '>' ->
      let (vbs1, e1) = loop expr1 in
      let (vbs2, e2) = loop expr2 in
      ((reduce_vbss [vbs1; vbs2]),
        ({
           pexp_desc =
             (Pexp_apply
                ({
                   pexp_desc =
                     (Pexp_ident
                        { txt = (Ldot ((Lident "TDSL"), "einsum")); loc });
                   pexp_loc = loc;
                   pexp_loc_stack = [];
                   pexp_attributes = []
                 },
                  [((Optional "label"),
                     (opt_expr ~loc label : Ppxlib_ast.Ast.expression));
                  (Nolabel, (spec : Ppxlib_ast.Ast.expression));
                  (Nolabel, (e1 : Ppxlib_ast.Ast.expression));
                  (Nolabel, (e2 : Ppxlib_ast.Ast.expression))]));
           pexp_loc = loc;
           pexp_loc_stack = [];
           pexp_attributes = []
         } : Ppxlib_ast.Ast.expression))
  | ({
       pexp_desc = Pexp_apply
         ({ pexp_desc = Pexp_ident { txt = Lident "++"; loc = _ };
            pexp_loc = _; pexp_loc_stack = _; pexp_attributes = _ },
          (Nolabel, (expr1 : Ppxlib_ast.Ast.expression))::(Nolabel,
                                                           (({
                                                               pexp_desc =
                                                                 Pexp_constant
                                                                 (Pconst_string
                                                                 (spec_str,
                                                                  _, _));_}
                                                               as spec)
                                                             :
                                                             Ppxlib_ast.Ast.expression))::[]);
       pexp_loc = _; pexp_loc_stack = _; pexp_attributes = _ }
      : Ppxlib_ast.Ast.expression) when String.contains spec_str '>' ->
      let (vbs1, e1) = loop expr1 in
      (vbs1,
        ({
           pexp_desc =
             (Pexp_apply
                ({
                   pexp_desc =
                     (Pexp_ident
                        { txt = (Ldot ((Lident "TDSL"), "einsum1")); loc });
                   pexp_loc = loc;
                   pexp_loc_stack = [];
                   pexp_attributes = []
                 },
                  [((Optional "label"),
                     (opt_expr ~loc label : Ppxlib_ast.Ast.expression));
                  (Nolabel, (spec : Ppxlib_ast.Ast.expression));
                  (Nolabel, (e1 : Ppxlib_ast.Ast.expression))]));
           pexp_loc = loc;
           pexp_loc_stack = [];
           pexp_attributes = []
         } : Ppxlib_ast.Ast.expression))
  | ({
       pexp_desc = Pexp_apply
         ((({
              pexp_desc = Pexp_constant (Pconst_string (ident, str_loc, _));_}
              as s)
            : Ppxlib_ast.Ast.expression),
          (Nolabel,
           (({ pexp_desc = Pexp_constant (Pconst_integer _);
               pexp_loc = dims_loc;_}
             | { pexp_desc = Pexp_ident _; pexp_loc = dims_loc;_}
             | { pexp_desc = Pexp_field _; pexp_loc = dims_loc;_} as d) :
             Ppxlib_ast.Ast.expression))::[]);
       pexp_loc = _; pexp_loc_stack = _; pexp_attributes = _ }
      : Ppxlib_ast.Ast.expression) ->
      let (pat, vb) =
        make_vb_dims ~has_config ~loc ~str_loc ~ident ~dims:[d] ~dims_loc s in
      ((Map.singleton (module String) ident vb), (pat2expr pat))
  | ({
       pexp_desc = Pexp_apply
         ((({
              pexp_desc = Pexp_constant (Pconst_string (ident, str_loc, _));_}
              as s)
            : Ppxlib_ast.Ast.expression),
          (Nolabel,
           (({ pexp_desc = Pexp_array _;_}
             | { pexp_desc = Pexp_construct ({ txt = Lident "::";_}, _);_} as
               init_nd)
             : Ppxlib_ast.Ast.expression))::[]);
       pexp_loc = _; pexp_loc_stack = _; pexp_attributes = _ }
      : Ppxlib_ast.Ast.expression) ->
      let (pat, vb) = make_vb_nd ~has_config ~loc ~str_loc ~ident ~init_nd s in
      ((Map.singleton (module String) ident vb), (pat2expr pat))
  | ({
       pexp_desc = Pexp_apply
         ((({
              pexp_desc = Pexp_constant (Pconst_string (ident, str_loc, _));_}
              as s)
            : Ppxlib_ast.Ast.expression),
          (Nolabel,
           ({ pexp_desc = Pexp_tuple dims; pexp_loc = dims_loc;_} :
             Ppxlib_ast.Ast.expression))::[]);
       pexp_loc = _; pexp_loc_stack = _; pexp_attributes = _ }
      : Ppxlib_ast.Ast.expression) ->
      let (pat, vb) =
        make_vb_dims ~has_config ~loc ~str_loc ~ident ~dims ~dims_loc s in
      ((Map.singleton (module String) ident vb), (pat2expr pat))
  | { pexp_desc = Pexp_constant (Pconst_string (ident, str_loc, _));_} ->
      let (pat, vb) = make_vb ~has_config ~loc ~str_loc ~ident expr in
      ((Map.singleton (module String) ident vb), (pat2expr pat))
  | { pexp_desc = Pexp_array _;_}
    | { pexp_desc = Pexp_construct ({ txt = Lident "::";_}, _);_} ->
      (no_vbs, (ndarray_op ?label expr))
  | ({
       pexp_desc = Pexp_apply
         ({ pexp_desc = Pexp_ident { txt = Lident "**."; loc = _ };
            pexp_loc = _; pexp_loc_stack = _; pexp_attributes = _ },
          (Nolabel, (expr1 : Ppxlib_ast.Ast.expression))::(Nolabel,
                                                           (({
                                                               pexp_desc =
                                                                 Pexp_constant
                                                                 (Pconst_integer
                                                                 _);_}
                                                               as i)
                                                             :
                                                             Ppxlib_ast.Ast.expression))::[]);
       pexp_loc = _; pexp_loc_stack = _; pexp_attributes = _ }
      : Ppxlib_ast.Ast.expression) ->
      let (vbs, e1) = loop expr1 in
      (vbs,
        ({
           pexp_desc =
             (Pexp_apply
                ({
                   pexp_desc =
                     (Pexp_ident
                        {
                          txt = (Ldot ((Ldot ((Lident "TDSL"), "O")), "**."));
                          loc
                        });
                   pexp_loc = loc;
                   pexp_loc_stack = [];
                   pexp_attributes = []
                 },
                  [((Optional "label"),
                     (opt_expr ~loc label : Ppxlib_ast.Ast.expression));
                  (Nolabel, (e1 : Ppxlib_ast.Ast.expression));
                  (Nolabel,
                    {
                      pexp_desc =
                        (Pexp_apply
                           ({
                              pexp_desc =
                                (Pexp_ident
                                   {
                                     txt =
                                       (Ldot ((Lident "Float"), "of_int"));
                                     loc
                                   });
                              pexp_loc = loc;
                              pexp_loc_stack = [];
                              pexp_attributes = []
                            }, [(Nolabel, (i : Ppxlib_ast.Ast.expression))]));
                      pexp_loc = loc;
                      pexp_loc_stack = [loc];
                      pexp_attributes = []
                    })]));
           pexp_loc = loc;
           pexp_loc_stack = [];
           pexp_attributes = []
         } : Ppxlib_ast.Ast.expression))
  | ({
       pexp_desc = Pexp_apply
         ({ pexp_desc = Pexp_ident { txt = Lident "**."; loc = _ };
            pexp_loc = _; pexp_loc_stack = _; pexp_attributes = _ },
          (Nolabel, (expr1 : Ppxlib_ast.Ast.expression))::(Nolabel,
                                                           (expr2 :
                                                             Ppxlib_ast.Ast.expression))::[]);
       pexp_loc = _; pexp_loc_stack = _; pexp_attributes = _ }
      : Ppxlib_ast.Ast.expression) ->
      let (vbs, e1) = loop expr1 in
      (vbs,
        ({
           pexp_desc =
             (Pexp_apply
                ({
                   pexp_desc =
                     (Pexp_ident
                        {
                          txt = (Ldot ((Ldot ((Lident "TDSL"), "O")), "**."));
                          loc
                        });
                   pexp_loc = loc;
                   pexp_loc_stack = [];
                   pexp_attributes = []
                 },
                  [((Optional "label"),
                     (opt_expr ~loc label : Ppxlib_ast.Ast.expression));
                  (Nolabel, (e1 : Ppxlib_ast.Ast.expression));
                  (Nolabel, (expr2 : Ppxlib_ast.Ast.expression))]));
           pexp_loc = loc;
           pexp_loc_stack = [];
           pexp_attributes = []
         } : Ppxlib_ast.Ast.expression))
  | ({
       pexp_desc = Pexp_apply
         ((expr1 : Ppxlib_ast.Ast.expression),
          (Labelled "config", (c_expr : Ppxlib_ast.Ast.expression))::
          (Nolabel, (expr2 : Ppxlib_ast.Ast.expression))::(Nolabel,
                                                           (expr3 :
                                                             Ppxlib_ast.Ast.expression))::[]);
       pexp_loc = _; pexp_loc_stack = _; pexp_attributes = _ }
      : Ppxlib_ast.Ast.expression) ->
      lift_config_vb ~loop ~num_configs ?label ~expr1 ~c_expr [expr2; expr3]
  | ({
       pexp_desc = Pexp_apply
         ((expr1 : Ppxlib_ast.Ast.expression),
          (Labelled "config", (c_expr : Ppxlib_ast.Ast.expression))::
          (Nolabel, (expr2 : Ppxlib_ast.Ast.expression))::[]);
       pexp_loc = _; pexp_loc_stack = _; pexp_attributes = _ }
      : Ppxlib_ast.Ast.expression) ->
      lift_config_vb ~loop ~num_configs ?label ~expr1 ~c_expr [expr2]
  | ({
       pexp_desc = Pexp_apply
         ((expr1 : Ppxlib_ast.Ast.expression),
          (Labelled "config", (c_expr : Ppxlib_ast.Ast.expression))::[]);
       pexp_loc = _; pexp_loc_stack = _; pexp_attributes = _ }
      : Ppxlib_ast.Ast.expression) ->
      lift_config_vb ~loop ~num_configs ?label ~expr1 ~c_expr []
  | ({
       pexp_desc = Pexp_apply
         ((expr1 : Ppxlib_ast.Ast.expression),
          (Nolabel, (expr2 : Ppxlib_ast.Ast.expression))::(Nolabel,
                                                           (expr3 :
                                                             Ppxlib_ast.Ast.expression))::[]);
       pexp_loc = _; pexp_loc_stack = _; pexp_attributes = _ }
      : Ppxlib_ast.Ast.expression) ->
      let (vbs1, e1) = loop ?label expr1 in
      let (vbs2, e2) = loop expr2 in
      let (vbs3, e3) = loop expr3 in
      ((reduce_vbss [vbs1; vbs2; vbs3]),
        ({
           pexp_desc =
             (Pexp_apply
                ((e1 : Ppxlib_ast.Ast.expression),
                  [(Nolabel, (e2 : Ppxlib_ast.Ast.expression));
                  (Nolabel, (e3 : Ppxlib_ast.Ast.expression))]));
           pexp_loc = loc;
           pexp_loc_stack = [];
           pexp_attributes = []
         } : Ppxlib_ast.Ast.expression))
  | ({
       pexp_desc = Pexp_apply
         ((expr1 : Ppxlib_ast.Ast.expression),
          (Nolabel, (expr2 : Ppxlib_ast.Ast.expression))::[]);
       pexp_loc = _; pexp_loc_stack = _; pexp_attributes = _ }
      : Ppxlib_ast.Ast.expression) ->
      let (vbs1, e1) = loop ?label expr1 in
      let (vbs2, e2) = loop expr2 in
      ((reduce_vbss [vbs1; vbs2]),
        ({
           pexp_desc =
             (Pexp_apply
                ((e1 : Ppxlib_ast.Ast.expression),
                  [(Nolabel, (e2 : Ppxlib_ast.Ast.expression))]));
           pexp_loc = loc;
           pexp_loc_stack = [];
           pexp_attributes = []
         } : Ppxlib_ast.Ast.expression))
  | ({
       pexp_desc = Pexp_fun
         (Labelled "config", None,
          { ppat_desc = Ppat_var { txt = "config"; loc = _ }; ppat_loc = _;
            ppat_loc_stack = _; ppat_attributes = _ },
          (body : Ppxlib_ast.Ast.expression));
       pexp_loc = _; pexp_loc_stack = _; pexp_attributes = _ }
      : Ppxlib_ast.Ast.expression) ->
      let (vbs, body) =
        translate ~num_configs ~is_toplevel:true ~has_config:true ?label body in
      (no_vbs,
        ({
           pexp_desc =
             (Pexp_fun
                ((Labelled "config"), None,
                  {
                    ppat_desc = (Ppat_var { txt = "config"; loc });
                    ppat_loc = loc;
                    ppat_loc_stack = [];
                    ppat_attributes = []
                  }, (let_opt ~loc vbs body : Ppxlib_ast.Ast.expression)));
           pexp_loc = loc;
           pexp_loc_stack = [];
           pexp_attributes = []
         } : Ppxlib_ast.Ast.expression))
  | ({
       pexp_desc = Pexp_fun
         (Labelled "config", None,
          {
            ppat_desc = Ppat_constraint
              ({ ppat_desc = Ppat_var { txt = "config"; loc = _ };
                 ppat_loc = _; ppat_loc_stack = _; ppat_attributes = _ },
               {
                 ptyp_desc = Ptyp_extension
                   ({ txt = "typ"; loc = _ }, PPat
                    ({ ppat_desc = Ppat_var { txt = "config_ty"; loc = _ };
                       ppat_loc = _; ppat_loc_stack = _; ppat_attributes = _
                       },
                     None));
                 ptyp_loc = _; ptyp_loc_stack = _; ptyp_attributes = _ });
            ppat_loc = _; ppat_loc_stack = _; ppat_attributes = _ },
          (body : Ppxlib_ast.Ast.expression));
       pexp_loc = _; pexp_loc_stack = _; pexp_attributes = _ }
      : Ppxlib_ast.Ast.expression) ->
      let (vbs, body) =
        translate ~num_configs ~is_toplevel:true ~has_config:true ?label body in
      (no_vbs,
        ({
           pexp_desc =
             (Pexp_fun
                ((Labelled "config"), None,
                  {
                    ppat_desc =
                      (Ppat_constraint
                         ({
                            ppat_desc = (Ppat_var { txt = "config"; loc });
                            ppat_loc = loc;
                            ppat_loc_stack = [];
                            ppat_attributes = []
                          },
                           {
                             ptyp_desc =
                               (Ptyp_extension
                                  ({ txt = "typ"; loc },
                                    (PStr
                                       [{
                                          pstr_desc =
                                            (Pstr_eval
                                               ({
                                                  pexp_desc =
                                                    (Pexp_ident
                                                       {
                                                         txt = (Lident "ty");
                                                         loc
                                                       });
                                                  pexp_loc = loc;
                                                  pexp_loc_stack = [];
                                                  pexp_attributes = []
                                                }, []));
                                          pstr_loc = loc
                                        }])));
                             ptyp_loc = loc;
                             ptyp_loc_stack = [];
                             ptyp_attributes = []
                           }));
                    ppat_loc = loc;
                    ppat_loc_stack = [];
                    ppat_attributes = []
                  }, (let_opt ~loc vbs body : Ppxlib_ast.Ast.expression)));
           pexp_loc = loc;
           pexp_loc_stack = [];
           pexp_attributes = []
         } : Ppxlib_ast.Ast.expression))
  | ({
       pexp_desc = Pexp_fun
         (Nolabel, None, (pat : Ppxlib_ast.Ast.pattern),
          (body : Ppxlib_ast.Ast.expression));
       pexp_loc = _; pexp_loc_stack = _; pexp_attributes = _ }
      : Ppxlib_ast.Ast.expression) when is_toplevel ->
      let input_label =
        let loc = pat.ppat_loc in
        ({
           pexp_desc =
             (Pexp_field
                ({
                   pexp_desc =
                     (Pexp_field
                        ((pat2expr pat : Ppxlib_ast.Ast.expression),
                          { txt = (Ldot ((Lident "Tensor"), "value")); loc }));
                   pexp_loc = loc;
                   pexp_loc_stack = [];
                   pexp_attributes = []
                 },
                  {
                    txt =
                      (Ldot ((Ldot ((Lident "Arrayjit"), "Tnode")), "label"));
                    loc
                  }));
           pexp_loc = loc;
           pexp_loc_stack = [];
           pexp_attributes = []
         } : Ppxlib_ast.Ast.expression) in
      let label =
        match label with
        | None -> input_label
        | Some label ->
            let loc = pat.ppat_loc in
            ({
               pexp_desc =
                 (Pexp_apply
                    ({
                       pexp_desc = (Pexp_ident { txt = (Lident "@"); loc });
                       pexp_loc = loc;
                       pexp_loc_stack = [];
                       pexp_attributes = []
                     },
                      [(Nolabel, (label : Ppxlib_ast.Ast.expression));
                      (Nolabel, (input_label : Ppxlib_ast.Ast.expression))]));
               pexp_loc = loc;
               pexp_loc_stack = [];
               pexp_attributes = []
             } : Ppxlib_ast.Ast.expression) in
      let (vbs, body) = loop ~label body in
      (vbs,
        ({
           pexp_desc =
             (Pexp_fun
                (Nolabel, None, (pat : Ppxlib_ast.Ast.pattern),
                  (body : Ppxlib_ast.Ast.expression)));
           pexp_loc = loc;
           pexp_loc_stack = [];
           pexp_attributes = []
         } : Ppxlib_ast.Ast.expression))
  | ({
       pexp_desc = Pexp_fun
         (Nolabel, None, (pat : Ppxlib_ast.Ast.pattern),
          (body : Ppxlib_ast.Ast.expression));
       pexp_loc = _; pexp_loc_stack = _; pexp_attributes = _ }
      : Ppxlib_ast.Ast.expression) ->
      let (vbs, body) = loop ?label body in
      (vbs,
        ({
           pexp_desc =
             (Pexp_fun
                (Nolabel, None, (pat : Ppxlib_ast.Ast.pattern),
                  (body : Ppxlib_ast.Ast.expression)));
           pexp_loc = loc;
           pexp_loc_stack = [];
           pexp_attributes = []
         } : Ppxlib_ast.Ast.expression))
  | ({
       pexp_desc = Pexp_while
         ((test_expr : Ppxlib_ast.Ast.expression),
          (body_expr : Ppxlib_ast.Ast.expression));
       pexp_loc = _; pexp_loc_stack = _; pexp_attributes = _ }
      : Ppxlib_ast.Ast.expression) ->
      let (vbs, body) = loop ?label body_expr in
      (vbs,
        ({
           pexp_desc =
             (Pexp_while
                ((test_expr : Ppxlib_ast.Ast.expression),
                  (body : Ppxlib_ast.Ast.expression)));
           pexp_loc = loc;
           pexp_loc_stack = [];
           pexp_attributes = []
         } : Ppxlib_ast.Ast.expression))
  | ({
       pexp_desc = Pexp_for
         ((pat : Ppxlib_ast.Ast.pattern), (init : Ppxlib_ast.Ast.expression),
          (final : Ppxlib_ast.Ast.expression), Upto,
          (body_expr : Ppxlib_ast.Ast.expression));
       pexp_loc = _; pexp_loc_stack = _; pexp_attributes = _ }
      : Ppxlib_ast.Ast.expression) ->
      let (vbs, body) = loop ?label body_expr in
      (vbs,
        ({
           pexp_desc =
             (Pexp_for
                ((pat : Ppxlib_ast.Ast.pattern),
                  (init : Ppxlib_ast.Ast.expression),
                  (final : Ppxlib_ast.Ast.expression), Upto,
                  (body : Ppxlib_ast.Ast.expression)));
           pexp_loc = loc;
           pexp_loc_stack = [];
           pexp_attributes = []
         } : Ppxlib_ast.Ast.expression))
  | ({
       pexp_desc = Pexp_for
         ((pat : Ppxlib_ast.Ast.pattern), (init : Ppxlib_ast.Ast.expression),
          (final : Ppxlib_ast.Ast.expression), Downto,
          (body_expr : Ppxlib_ast.Ast.expression));
       pexp_loc = _; pexp_loc_stack = _; pexp_attributes = _ }
      : Ppxlib_ast.Ast.expression) ->
      let (vbs, body) = loop ?label body_expr in
      (vbs,
        ({
           pexp_desc =
             (Pexp_for
                ((pat : Ppxlib_ast.Ast.pattern),
                  (init : Ppxlib_ast.Ast.expression),
                  (final : Ppxlib_ast.Ast.expression), Downto,
                  (body : Ppxlib_ast.Ast.expression)));
           pexp_loc = loc;
           pexp_loc_stack = [];
           pexp_attributes = []
         } : Ppxlib_ast.Ast.expression))
  | ({
       pexp_desc = Pexp_sequence
         ((expr1 : Ppxlib_ast.Ast.expression),
          (expr2 : Ppxlib_ast.Ast.expression));
       pexp_loc = _; pexp_loc_stack = _; pexp_attributes = _ }
      : Ppxlib_ast.Ast.expression) ->
      let (vbs1, e1) = loop expr1 in
      let (vbs2, e2) = loop ?label expr2 in
      ((reduce_vbss [vbs1; vbs2]),
        ({
           pexp_desc =
             (Pexp_sequence
                ((e1 : Ppxlib_ast.Ast.expression),
                  (e2 : Ppxlib_ast.Ast.expression)));
           pexp_loc = loc;
           pexp_loc_stack = [];
           pexp_attributes = []
         } : Ppxlib_ast.Ast.expression))
  | ({
       pexp_desc = Pexp_ifthenelse
         ((expr1 : Ppxlib_ast.Ast.expression),
          (expr2 : Ppxlib_ast.Ast.expression), Some
          (expr3 : Ppxlib_ast.Ast.expression));
       pexp_loc = _; pexp_loc_stack = _; pexp_attributes = _ }
      : Ppxlib_ast.Ast.expression) ->
      let (vbs2, e2) = loop ?label expr2 in
      let (vbs3, e3) = loop ?label expr3 in
      ((reduce_vbss [vbs2; vbs3]),
        ({
           pexp_desc =
             (Pexp_ifthenelse
                ((expr1 : Ppxlib_ast.Ast.expression),
                  (e2 : Ppxlib_ast.Ast.expression),
                  (Some (e3 : Ppxlib_ast.Ast.expression))));
           pexp_loc = loc;
           pexp_loc_stack = [];
           pexp_attributes = []
         } : Ppxlib_ast.Ast.expression))
  | ({
       pexp_desc = Pexp_ifthenelse
         ((expr1 : Ppxlib_ast.Ast.expression),
          (expr2 : Ppxlib_ast.Ast.expression), None);
       pexp_loc = _; pexp_loc_stack = _; pexp_attributes = _ }
      : Ppxlib_ast.Ast.expression) ->
      let (vbs2, e2) = loop ?label expr2 in
      (vbs2,
        ({
           pexp_desc =
             (Pexp_ifthenelse
                ((expr1 : Ppxlib_ast.Ast.expression),
                  (e2 : Ppxlib_ast.Ast.expression), None));
           pexp_loc = loc;
           pexp_loc_stack = [];
           pexp_attributes = []
         } : Ppxlib_ast.Ast.expression))
  | { pexp_desc = Pexp_match (expr1, cases);_} ->
      let (vbss, cases) =
        List.unzip @@
          (List.map cases
             ~f:(fun ({ pc_rhs;_} as c) ->
                   let (vbs, pc_rhs) = loop ?label pc_rhs in
                   (vbs, { c with pc_rhs }))) in
      ((reduce_vbss vbss),
        { expr with pexp_desc = (Pexp_match (expr1, cases)) })
  | { pexp_desc = Pexp_let (recflag, bindings, body);_} ->
      let (vbss1, bindings) =
        List.unzip @@
          (List.map bindings
             ~f:(fun binding ->
                   let (vbs, pvb_expr) =
                     loop
                       ~label:({
                                 pexp_desc =
                                   (Pexp_construct
                                      ({ txt = (Lident "::"); loc },
                                        (Some
                                           {
                                             pexp_desc =
                                               (Pexp_tuple
                                                  [(pat2string
                                                      binding.pvb_pat : 
                                                  Ppxlib_ast.Ast.expression);
                                                  {
                                                    pexp_desc =
                                                      (Pexp_construct
                                                         ({
                                                            txt =
                                                              (Lident "[]");
                                                            loc
                                                          }, None));
                                                    pexp_loc = loc;
                                                    pexp_loc_stack = [];
                                                    pexp_attributes = []
                                                  }]);
                                             pexp_loc = loc;
                                             pexp_loc_stack = [];
                                             pexp_attributes = []
                                           })));
                                 pexp_loc = loc;
                                 pexp_loc_stack = [];
                                 pexp_attributes = []
                               } : Ppxlib_ast.Ast.expression)
                       binding.pvb_expr in
                   (vbs, { binding with pvb_expr }))) in
      let (vbs2, body) = loop ?label body in
      let all_bindings =
        (Map.data @@ (reduce_vbss vbss1)) @ (bindings @ (Map.data vbs2)) in
      (no_vbs,
        { expr with pexp_desc = (Pexp_let (recflag, all_bindings, body)) })
  | { pexp_desc = Pexp_open (decl, body);_} ->
      let (vbs, body) = loop ?label body in
      (vbs, { expr with pexp_desc = (Pexp_open (decl, body)) })
  | { pexp_desc = Pexp_letmodule (name, module_expr, body);_} ->
      let (vbs, body) = loop ?label body in
      (vbs,
        { expr with pexp_desc = (Pexp_letmodule (name, module_expr, body)) })
  | { pexp_desc = Pexp_ident { txt = Lident op_ident;_};_} when
      is_operator op_ident ->
      (no_vbs,
        ({
           pexp_desc =
             (Pexp_apply
                ((expr : Ppxlib_ast.Ast.expression),
                  [((Optional "label"),
                     (opt_expr ~loc label : Ppxlib_ast.Ast.expression))]));
           pexp_loc = loc;
           pexp_loc_stack = [];
           pexp_attributes = []
         } : Ppxlib_ast.Ast.expression))
  | expr -> (no_vbs, expr)
let translate ?ident_label  expr =
  let (vbs, expr) =
    translate ~num_configs:(ref 0) ~is_toplevel:true ~has_config:false
      ~label:(opt_pat2string_list ~loc:(expr.pexp_loc) ident_label) expr in
  let loc = expr.pexp_loc in
  (vbs,
    (match ident_label with
     | Some
         ({ ppat_desc = Ppat_any; ppat_loc = _; ppat_loc_stack = _;
            ppat_attributes = _ }
           : Ppxlib_ast.Ast.pattern)
         ->
         ({
            pexp_desc =
              (Pexp_apply
                 ({
                    pexp_desc =
                      (Pexp_ident
                         {
                           txt =
                             (Ldot
                                ((Lident "Tensor"), "with_unchanged_roots"));
                           loc
                         });
                    pexp_loc = loc;
                    pexp_loc_stack = [];
                    pexp_attributes = []
                  },
                   [((Labelled "f"),
                      {
                        pexp_desc =
                          (Pexp_fun
                             (Nolabel, None,
                               {
                                 ppat_desc =
                                   (Ppat_construct
                                      ({ txt = (Lident "()"); loc }, None));
                                 ppat_loc = loc;
                                 ppat_loc_stack = [];
                                 ppat_attributes = []
                               },
                               {
                                 pexp_desc =
                                   (Pexp_open
                                      ({
                                         popen_expr =
                                           {
                                             pmod_desc =
                                               (Pmod_ident
                                                  {
                                                    txt =
                                                      (Ldot
                                                         ((Lident "TDSL"),
                                                           "O"));
                                                    loc
                                                  });
                                             pmod_loc = loc;
                                             pmod_attributes = []
                                           };
                                         popen_override = Override;
                                         popen_loc = loc;
                                         popen_attributes = []
                                       }, (expr : Ppxlib_ast.Ast.expression)));
                                 pexp_loc = loc;
                                 pexp_loc_stack = [];
                                 pexp_attributes = []
                               }));
                        pexp_loc = loc;
                        pexp_loc_stack = [loc];
                        pexp_attributes = []
                      })]));
            pexp_loc = loc;
            pexp_loc_stack = [];
            pexp_attributes = []
          } : Ppxlib_ast.Ast.expression)
     | _ ->
         ({
            pexp_desc =
              (Pexp_open
                 ({
                    popen_expr =
                      {
                        pmod_desc =
                          (Pmod_ident
                             { txt = (Ldot ((Lident "TDSL"), "O")); loc });
                        pmod_loc = loc;
                        pmod_attributes = []
                      };
                    popen_override = Override;
                    popen_loc = loc;
                    popen_attributes = []
                  }, (expr : Ppxlib_ast.Ast.expression)));
            pexp_loc = loc;
            pexp_loc_stack = [];
            pexp_attributes = []
          } : Ppxlib_ast.Ast.expression)))
let expr_expander ~loc  ~path  =
  expr_expander_with_punning translate ~loc ~path
let str_expander ~loc  ~path  =
  str_expander_with_punning translate ~loc ~path
let () = Ppx_inline_test_lib.unset_lib "ppx_ocannl"
let () = Ppx_expect_runtime.Current_file.unset ()
let () = Ppx_bench_lib.Benchmark_accumulator.Current_libname.unset ()
let () =
  Ppx_module_timer_runtime.record_until Ppx_module_timer_runtime.__MODULE__
