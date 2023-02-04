open Base

open Ppxlib

let pat2expr pat = ignore pat; failwith "NOT LOOKED UP YET"

let rec translate expr =
  let loc = expr.pexp_loc in
  match expr with
  | [%expr [%e? expr1] [%e? expr2] [%e? expr3] ] ->
    [%expr Network.apply (Network.apply [%e expr1] [%e translate expr2]) [%e translate expr3]]

  | [%expr [%e? expr1] [%e? expr2] ] ->
    [%expr Network.apply [%e expr1] [%e translate expr2]]

  | [%expr fun ~config [%p? pat1] [%p? pat2] -> [%e? body] ] ->
    (* TODO(38): generalize config to any number of labeled arguments with any labels. *)
    [%expr
      fun ~config ->
        let inp1 = ref None in let [%p pat1] = Network.return (Network.Placeholder inp1) in
        let inp2 = ref None in let [%p pat2] = Network.return (Network.Placeholder inp2) in
        let body = [%e translate body] in
        fun [%p pat1] [%p pat2] ->
          inp1 := [%e (pat2expr pat1)]; inp2 := [%e (pat2expr pat2)];
          Network.unpack body
    ]

  | [%expr fun ~config [%p? pat] -> [%e? body] ] ->
    (* TODO(38): generalize config to any number of labeled arguments with any labels. *)
    [%expr
      fun ~config ->
        let inp = ref None in let [%p pat] = Network.return (Network.Placeholder inp) in
        let body = [%e translate body] in
        fun [%p pat] ->
          inp := [%e (pat2expr pat)];
          Network.unpack body
    ]

  | [%expr fun [%p? pat1] [%p? pat2] -> [%e? body] ] ->
    [%expr
      let inp1 = ref None in let [%p pat1] = Network.return (Network.Placeholder inp1) in
      let inp2 = ref None in let [%p pat2] = Network.return (Network.Placeholder inp2) in
      let body = [%e translate body] in
      fun [%p pat1] [%p pat2] ->
        inp1 := [%e (pat2expr pat1)]; inp2 := [%e (pat2expr pat2)];
        Network.unpack body
    ]

  | [%expr fun [%p? pat] -> [%e? body] ] ->
    [%expr
      let inp = ref None in let [%p pat] = Network.return (Network.Placeholder inp) in
      let body = [%e translate body] in
      fun [%p pat] ->
        inp := [%e (pat2expr pat)];
        Network.unpack body
    ]

  | [%expr while [%e? test_expr] do [%e? body_expr] done ] ->
    [%expr while [%e test_expr] do [%e translate body_expr] done ]

  | [%expr for [%p? pat] = [%e? init] to [%e? final] do [%e? body] done ] ->
    [%expr for [%p pat] = [%e init] to [%e final] do [%e translate body] done ]

  | [%expr for [%p? pat] = [%e? init] downto [%e? final] do [%e? body] done ] ->
    [%expr for [%p pat] = [%e init] downto [%e final] do [%e translate body] done ]

  | [%expr [%e? expr1] ; [%e? expr2] ] ->
    [%expr [%e translate expr1] ; [%e translate expr2]]

  | [%expr if [%e? expr1] then [%e? expr2] else [%e? expr3]] ->
    [%expr if [%e expr1] then [%e translate expr2] else [%e translate expr3]]

  | [%expr if [%e? expr1] then [%e? expr2]] ->
    [%expr if [%e expr1] then [%e translate expr2]]

  | { pexp_desc = Pexp_match (expr1, cases); _ } ->
    let cases =
       List.map cases
         ~f:(fun ({pc_rhs; _} as c) ->
            {c with pc_rhs=translate pc_rhs}) in
     { expr with pexp_desc = Pexp_match (expr1, cases) }

  | { pexp_desc = Pexp_let (recflag, bindings, body); _ } ->
     let body = translate body in
     let bindings = List.map bindings ~f:(fun binding ->
         {binding with pvb_expr = translate binding.pvb_expr}) in
     {expr with pexp_desc=Pexp_let (recflag, bindings, body)}

  | { pexp_desc = Pexp_open (decl, body); _ } ->
     {expr with pexp_desc=Pexp_open (decl, translate body)}

  | { pexp_desc = Pexp_letmodule (name, module_expr, body); _ } ->
     {expr with pexp_desc=Pexp_letmodule (name, module_expr, translate body)}

  | expr ->
    expr

let expander ~loc:_ ~path:_ payload =
  translate payload

let flatten_str ~loc ~path:_ items =
  match items with
  | [x] -> x
  | _ ->
    Ast_helper.Str.include_ {
       pincl_mod = Ast_helper.Mod.structure items
     ; pincl_loc = loc
     ; pincl_attributes = [] }

let translate_str ({pstr_desc; _} as str) =
  match pstr_desc with
  | Pstr_eval (expr, attrs) -> {str with pstr_desc=Pstr_eval (translate expr, attrs)}
  | Pstr_value (recf, vbl) ->
    {str with pstr_desc=Pstr_value (recf, List.map vbl ~f:(fun vb -> {vb with pvb_expr=translate vb.pvb_expr}))}
  | _ -> str
     
let str_expander ~loc ~path (payload: structure_item list) =
  flatten_str ~loc ~path @@ List.map payload ~f:translate_str

let rules =
  [Ppxlib.Context_free.Rule.extension  @@
   Extension.declare "ocannl" Extension.Context.expression Ast_pattern.(single_expr_payload __) expander;
   Ppxlib.Context_free.Rule.extension  @@
   Extension.declare "ocannl" Extension.Context.structure_item Ast_pattern.(pstr __) str_expander;
   ]

let () =
  Driver.register_transformation
    ~rules
    "ppx_ocannl"
