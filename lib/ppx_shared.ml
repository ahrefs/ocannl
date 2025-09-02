open Base
open Ppxlib
open Ppx_arrayjit.Ppx_helper

type li = longident

let string_expr ~loc s = Ast_helper.Exp.constant @@ Pconst_string (s, loc, None)

let string_of_pat pat =
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
  loop pat

let pat2string pat =
  let loc = pat.ppat_loc in
  string_expr ~loc @@ string_of_pat pat

let collect_pat_idents pat =
  let one = Set.singleton (module String) in
  let none = Set.empty (module String) in
  let rec loop pat =
    let all pats = Set.union_list (module String) @@ List.map ~f:loop pats in
    match pat.ppat_desc with
    | Ppat_open (_, pat) | Ppat_lazy pat | Ppat_constraint (pat, _) -> loop pat
    | Ppat_alias (_, ident) -> one ident.txt
    | Ppat_var ident -> one ident.txt
    | Ppat_any -> none
    | Ppat_variant (_, None) -> none
    | Ppat_variant (_, Some pat) -> loop pat
    | Ppat_constant _ -> none
    | Ppat_tuple pats | Ppat_array pats -> all pats
    | Ppat_construct (_, None) -> none
    | Ppat_construct (_, Some (_, pat)) -> loop pat
    | Ppat_interval (_, _) -> none
    | Ppat_record (lpats, _) -> all @@ List.map ~f:snd lpats
    | Ppat_or (p1, p2) -> all [ p1; p2 ]
    | Ppat_type _ | Ppat_unpack _ | Ppat_exception _ | Ppat_extension _ -> none
  in
  loop pat

let expr2string_or_empty expr =
  let rec lident = function
    | Lident s -> s
    | Ldot (li, s) -> lident li ^ "." ^ s
    | Lapply (_, i) -> lident i
  in
  let rec loop expr =
    match expr.pexp_desc with
    | Pexp_open (_, expr) | Pexp_lazy expr | Pexp_constraint (expr, _) -> loop expr
    | Pexp_ident ident -> lident ident.txt
    | Pexp_variant (s, _)
    | Pexp_constant (Pconst_string (s, _, _))
    | Pexp_constant (Pconst_integer (s, _))
    | Pexp_constant (Pconst_float (s, _)) ->
        s
    | Pexp_constant (Pconst_char c) -> Char.to_string c
    | Pexp_tuple exprs -> "(" ^ String.concat ~sep:", " (List.map ~f:loop exprs) ^ ")"
    | Pexp_array exprs -> "[|" ^ String.concat ~sep:", " (List.map ~f:loop exprs) ^ "|]"
    | Pexp_construct (c, _) -> lident c.txt
    | _ -> ""
  in
  string_expr ~loc:expr.pexp_loc @@ loop expr

let opt_pat2string ~loc = function
  | None -> [%expr None]
  | Some pat -> [%expr Some [%e pat2string pat]]

let opt_pat2string_list ~loc = function
  | None -> [%expr []]
  | Some pat -> [%expr [ [%e pat2string pat] ]]

let opt_expr ~loc = function None -> [%expr None] | Some expr -> [%expr Some [%e expr]]

let rec pat2expr pat =
  let module Ast = Ast_builder.Default in
  let loc = pat.ppat_loc in
  match pat.ppat_desc with
  | Ppat_constraint (pat', typ) -> Ast.pexp_constraint ~loc (pat2expr pat') typ
  | Ppat_alias (_, ident) | Ppat_var ident ->
      Ast.pexp_ident ~loc { ident with txt = Lident ident.txt }
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

let non_alphanum_regexp = Str.regexp "^[^a-zA-Z0-9]+$"
let is_operator ident = Str.string_match non_alphanum_regexp ident 0

let is_assignment ident =
  String.length ident > 1
  && Char.equal ident.[0] '='
  && (not @@ List.mem [ "=="; "==="; "=>"; "==>"; "=>>" ] ident ~equal:String.equal)

(** Binary primitive ops, both infix operator and function name variants. *)
let binary_ops =
  Hashtbl.of_alist_exn
    (module String)
    [
      ("-@>", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Arg1]));
      ("fst", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Arg1]));
      ("-/>", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Arg2]));
      ("snd", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Arg2]));
      ("+", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Add]));
      ("add", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Add]));
      ("-", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Sub]));
      ("sub", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Sub]));
      ( "*",
        fun loc ->
          ( Ast_builder.Default.pexp_extension ~loc
            @@ Location.error_extensionf ~loc
                 "No default compose type for binary `*`, try e.g. ~logic:\".\" for pointwise, %s"
                 "~logic:\"@\" for matrix multiplication",
            [%expr Ir.Ops.Mul] ) );
      ("mul", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Mul]));
      ( "/",
        fun loc ->
          ( Ast_builder.Default.pexp_extension ~loc
            @@ Location.error_extensionf ~loc
                 "For clarity, no default compose type for binary `/`, use ~logic:\".\" for \
                  pointwise division",
            [%expr Ir.Ops.Div] ) );
      ("div", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Div]));
      ("**", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.ToPowOf]));
      ("pow", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.ToPowOf]));
      ("-?/", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Relu_gate]));
      ("relu_gate", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Relu_gate]));
      ("-?^", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Satur01_gate]));
      ("sat01_gate", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Satur01_gate]));
      ("<", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Cmplt]));
      ("lt", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Cmplt]));
      ("=", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Cmpeq]));
      ("eq", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Cmpeq]));
      ("<>", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Cmpne]));
      ("ne", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Cmpne]));
      ("||", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Or]));
      ("or_", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Or]));
      ("&&", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.And]));
      ("and_", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.And]));
      ("%", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Mod]));
      ("mod_", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Mod]));
      ("@^", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Max]));
      ("max", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Max]));
      ("@-", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Min]));
      ("min", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Min]));
      ("^^^^", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Threefry4x32_crypto]));
      ( "threefry4x32_crypto",
        fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Threefry4x32_crypto]) );
      ("^^", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Threefry4x32_light]));
      ( "threefry4x32_light",
        fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Threefry4x32_light]) );
    ]

(** Unary primitive ops. *)
let unary_ops =
  Hashtbl.of_alist_exn
    (module String)
    [
      ("id", fun loc -> ([%expr Shape.Pointwise_un], [%expr Ir.Ops.Identity]));
      ("relu", fun loc -> ([%expr Shape.Pointwise_un], [%expr Ir.Ops.Relu]));
      ("sat01", fun loc -> ([%expr Shape.Pointwise_un], [%expr Ir.Ops.Satur01]));
      ("exp", fun loc -> ([%expr Shape.Pointwise_un], [%expr Ir.Ops.Exp]));
      ("log", fun loc -> ([%expr Shape.Pointwise_un], [%expr Ir.Ops.Log]));
      ("exp2", fun loc -> ([%expr Shape.Pointwise_un], [%expr Ir.Ops.Exp2]));
      ("log2", fun loc -> ([%expr Shape.Pointwise_un], [%expr Ir.Ops.Log2]));
      ("sin", fun loc -> ([%expr Shape.Pointwise_un], [%expr Ir.Ops.Sin]));
      ("cos", fun loc -> ([%expr Shape.Pointwise_un], [%expr Ir.Ops.Cos]));
      ("sqrt", fun loc -> ([%expr Shape.Pointwise_un], [%expr Ir.Ops.Sqrt]));
      ("recip", fun loc -> ([%expr Shape.Pointwise_un], [%expr Ir.Ops.Recip]));
      ("recip_sqrt", fun loc -> ([%expr Shape.Pointwise_un], [%expr Ir.Ops.Recip_sqrt]));
      ("neg", fun loc -> ([%expr Shape.Pointwise_un], [%expr Ir.Ops.Neg]));
      ("tanh", fun loc -> ([%expr Shape.Pointwise_un], [%expr Ir.Ops.Tanh_approx]));
      ("not", fun loc -> ([%expr Shape.Pointwise_un], [%expr Ir.Ops.Not]));
      ( "uint4x32_to_prec_uniform1",
        fun loc -> ([%expr Shape.Pointwise_un], [%expr Ir.Ops.Uint4x32_to_prec_uniform1]) );
    ]

(** Vector unary primitive ops. *)
let vec_unary_ops =
  Hashtbl.of_alist_exn
    (module String)
    [
      ( "uint4x32_to_prec_uniform",
        fun loc -> ([%expr Shape.Uint4x32_to_prec], [%expr Ir.Ops.Uint4x32_to_prec_uniform]) );
    ]

(** Ternary primitive ops. *)
let ternary_ops =
  Hashtbl.of_alist_exn
    (module String)
    [
      ("where", fun loc -> ([%expr Shape.Pointwise_tern], [%expr Ir.Ops.Where]));
      ("fma", fun loc -> ([%expr Shape.Compose_accumulate], [%expr Ir.Ops.FMA]));
    ]

(** Assignment binary ops, and whether assignment reduction is zero-initialized. *)
let assignment_ops =
  (* This should stay in sync with Ir.Ops.assign_op_cd_syntax. *)
  Hashtbl.of_alist_exn
    (module String)
    [
      ("=:", fun loc -> (false, [%expr Ir.Ops.Arg2]));
      ("=+", fun loc -> (false, [%expr Ir.Ops.Add]));
      ("=-", fun loc -> (false, [%expr Ir.Ops.Sub]));
      ("=*", fun loc -> (false, [%expr Ir.Ops.Mul]));
      ("=/", fun loc -> (false, [%expr Ir.Ops.Div]));
      ("=**", fun loc -> (false, [%expr Ir.Ops.ToPowOf]));
      ("=?/", fun loc -> (false, [%expr Ir.Ops.Relu_gate]));
      ("=?^", fun loc -> (false, [%expr Ir.Ops.Satur01_gate]));
      ("=||", fun loc -> (false, [%expr Ir.Ops.Or]));
      ("=&&", fun loc -> (false, [%expr Ir.Ops.And]));
      ("=@^", fun loc -> (false, [%expr Ir.Ops.Max]));
      ("=@-", fun loc -> (false, [%expr Ir.Ops.Min]));
      ("=^^^^", fun loc -> (false, [%expr Ir.Ops.Threefry4x32]));
      ("=:+", fun loc -> (true, [%expr Ir.Ops.Add]));
      ("=:-", fun loc -> (true, [%expr Ir.Ops.Sub]));
      ("=:*", fun loc -> (true, [%expr Ir.Ops.Mul]));
      ("=:/", fun loc -> (true, [%expr Ir.Ops.Div]));
      ("=:**", fun loc -> (true, [%expr Ir.Ops.ToPowOf]));
      ("=:?/", fun loc -> (true, [%expr Ir.Ops.Relu_gate]));
      ("=:?^", fun loc -> (true, [%expr Ir.Ops.Satur01_gate]));
      ("=:||", fun loc -> (true, [%expr Ir.Ops.Or]));
      ("=:&&", fun loc -> (true, [%expr Ir.Ops.And]));
      ("=:@^", fun loc -> (true, [%expr Ir.Ops.Max]));
      ("=:@-", fun loc -> (true, [%expr Ir.Ops.Min]));
      ("=:^^^^", fun loc -> (true, [%expr Ir.Ops.Threefry4x32]));
    ]

let is_primitive_op op_ident =
  List.exists ~f:(Fn.flip Hashtbl.mem op_ident) [ ternary_ops; unary_ops; binary_ops ]

let let_opt ~loc vbs expr =
  if Map.is_empty vbs then expr else Ast_helper.Exp.let_ ~loc Nonrecursive (Map.data vbs) expr

let no_vbs = Map.empty (module String)
let reduce_vbss = List.reduce_exn ~f:(Map.merge_skewed ~combine:(fun ~key:_ _v1 v2 -> v2))

let expr_expander_with_punning translate ~loc ~path:_ payload =
  match payload with
  | { pexp_desc = Pexp_let (recflag, bindings, body); _ } ->
      (* We are at the %op/%cd annotation level: do not tranlsate the body. *)
      let vbss, bindings =
        List.unzip
        @@ List.map bindings ~f:(fun vb ->
               let vbs, v = translate ?ident_label:(Some vb.pvb_pat) vb.pvb_expr in
               (vbs, { vb with pvb_expr = v }))
      in
      let expr = { payload with pexp_desc = Pexp_let (recflag, bindings, body) } in
      let_opt ~loc (reduce_vbss vbss) expr
  | expr ->
      let vbs, expr = translate ?ident_label:None expr in
      let_opt ~loc vbs expr

let flatten_str ~loc ~path:_ items =
  match items with
  | [ x ] -> x
  | _ ->
      Ast_helper.Str.include_
        { pincl_mod = Ast_helper.Mod.structure items; pincl_loc = loc; pincl_attributes = [] }

let translate_str translate ({ pstr_desc; pstr_loc = loc; _ } as str) =
  match pstr_desc with
  | Pstr_eval (expr, attrs) ->
      let expr = expr_expander_with_punning translate ~loc ~path:() expr in
      { str with pstr_desc = Pstr_eval (expr, attrs) }
  | Pstr_value (recf, bindings) ->
      let f vb =
        let loc = vb.pvb_loc in
        let vbs, v = translate ?ident_label:(Some vb.pvb_pat) vb.pvb_expr in
        let v = let_opt ~loc vbs v in
        { vb with pvb_expr = v }
      in
      { str with pstr_desc = Pstr_value (recf, List.map bindings ~f) }
  | _ -> str

let str_expander_with_punning translate ~loc ~path (payload : structure_item list) =
  flatten_str ~loc ~path @@ List.map payload ~f:(translate_str translate)

let ndarray_op ?axis_labels ?label expr =
  let loc = expr.pexp_loc in
  let values, batch_dims, output_dims, input_dims = ndarray_constant expr in
  let edims dims = Ast_builder.Default.elist ~loc dims in
  let w_val = [%expr ndarray [%e values]] in
  let op =
    match (axis_labels, label) with
    | None, None -> w_val
    | Some axis_labels, None -> [%expr [%e w_val] ~axis_labels:[%e axis_labels]]
    | None, Some label -> [%expr [%e w_val] ~label:[%e label]]
    | Some axis_labels, Some label ->
        [%expr [%e w_val] ~axis_labels:[%e axis_labels] ~label:[%e label]]
  in
  [%expr
    [%e op] ~batch_dims:[%e edims batch_dims] ~input_dims:[%e edims input_dims]
      ~output_dims:[%e edims output_dims] ()]

let collect_capture_labels ~loc head rest =
  let capture_labels = head :: collect_list [] rest in
  let capture_labels, errors =
    List.partition_map capture_labels ~f:(function
      | { pexp_desc = Pexp_constant (Pconst_string (label, _, _)); pexp_loc; _ } ->
          Either.First (pexp_loc, label)
      | expr ->
          Either.Second
            (Ast_builder.Default.pexp_extension ~loc:expr.pexp_loc
            @@ Location.error_extensionf ~loc:expr.pexp_loc
                 "ppx_ocannl %%op: expected a string literal"))
  in
  let capture_refs, capture_bindings =
    List.map capture_labels ~f:(fun (loc, label) ->
        let ref_expr =
          [%expr
            {
              Ir.Indexing.ref_label = [%e Ast_builder.Default.estring ~loc label];
              solved_dim = None;
            }]
        in
        let binding =
          Ast_builder.Default.value_binding ~loc
            ~pat:(Ast_builder.Default.pvar ~loc label)
            ~expr:ref_expr
        in
        (Ast_builder.Default.evar ~loc label, (label, binding)))
    |> List.unzip
  in
  let capture_dims_expr = Ast_builder.Default.elist ~loc (errors @ capture_refs) in
  let capture_vbs = Map.of_alist_exn (module String) capture_bindings in
  (capture_vbs, capture_dims_expr)
