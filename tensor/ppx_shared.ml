open Base
open Ppxlib

(* {1 Coupling to ppxlib's selected AST}

   ppxlib parses with the compiler's own parser and then migrates to its SELECTED ast, which is
   Astlib.Ast_502 for every ppxlib this project permits. Where a shape is fixed, the four ppx
   modules state it as a quotation -- %expr lhs, %expr [%e? _] :: [%e? _], %expr [], %expr [||],
   %expr (), %pat? (), %expr [%oc [%e? e]] -- because metaquot is regenerated for whichever ast is
   selected and so cannot go stale. Write any new fixed shape that way.

   What is left names constructors, and this is the map of it, so that raising the ppxlib bound in
   [dune-project] is a known quantity rather than a discovery. Measured against Ast_503..Ast_506,
   which are the candidates a selection move can land on:

   - {b Literal values} ([Pconst_string], [Pconst_float], [Pconst_integer], [Pconst_char]) -- about
   fifty occurrences, mostly in [ppx_cd] and [ppx_op]. {b Moves at Ast_503}, where [constant] stops
   being a flat variant and becomes [{ pconst_desc; pconst_loc }]; the payloads themselves are
   unchanged. Not expressible: a quotation pins one literal, and there is no way to write "any
   string literal, and give me its text", which is what these arms need; [Ast_pattern.estring] says
   it but raises on an attributed node rather than declining to match (see [string_literal] below).
   Reads that stand alone are funnelled through [string_literal] and [string_of_constant]; the rest
   sit inside quotation-shaped arms in [ppx_cd]/[ppx_op]. Per-site fix: [Pconst_X (..)] becomes [{
   pconst_desc = Pconst_X (..); _ }]. Mechanical.

   - {b Tuple payloads} ([Pexp_tuple], [Ppat_tuple]) -- about fifteen occurrences. {b Moves at
   Ast_504}, which adds labelled tuples: [Pexp_tuple of (string option * expression) list] and
   [Ppat_tuple of (string option * pattern) list * closed_flag]. Not expressible: all but one bind a
   variable-length element list, and no quotation states one. Construction already goes through
   [Ast_builder.Default.pexp_tuple] where it can ([pat2expr]). Per-site fix: pair each element with
   [None] -- and decide, per site, whether a labelled tuple should be accepted as an axis container
   or refused. That is a design question, not a rename.

   - {b [Pexp_open] and [Pexp_letmodule]} -- ten occurrences. {b Moves at Ast_505}, which deletes
   both: [let open M in e] and [let module M = .. in e] become [Pexp_struct_item (Pstr_open
   ../Pstr_module .., e)]. Not expressible: the arms match a binder and rebuild it around a
   translated body, and neither a quotation nor a builder states "the body of whatever binder this
   is". This is the one residue class whose fix is a restructure rather than a rename;
   lukstafi/ocannl-staging#408 met the same deletion from the compiler side, in [Cache_dir_scan].

   - {b "Any node of this kind" tests} -- [Pexp_ident _], [Pexp_array _], [Pexp_apply (fn, args)],
   [Pexp_record (..)], [Pexp_function (..)], [Pstr_value], [Ppat_var] and friends. A quotation pins
   a shape; it cannot say "any identifier" or "an array of any length". These are coupling to a grep
   and not to a version: every one of them is byte-identical from Ast_502 through Ast_506,
   [Pexp_function]'s [params * constraint * body] triple included. *)

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
  | `Input_dims (dim, _) -> "input (tuple) of dim " ^ Int.to_string dim
  | `Output_dims (dim, _) -> "output (list) of dim " ^ Int.to_string dim
  | `Batch_dims (dim, _) -> "batch (array) of dim " ^ Int.to_string dim

(* Extract a tensor-literal axis-label basis from a type annotation: only a bare type-constructor
   name (e.g. [rgb], [heads]) is a label; anything else (qualified / parameterised / arrow / tuple /
   variant) is not, and the annotation is ignored. *)
let axis_basis_of_type (typ : core_type) : string option =
  match typ.ptyp_desc with Ptyp_constr ({ txt = Lident b; _ }, []) -> Some b | _ -> None

(* Reads of a literal outside the quotation-shaped arms of [ppx_cd] and [ppx_op] go through the two
   helpers below, so that ppxlib moving its selected AST off [Astlib.Ast_502] -- where [constant]
   stops being a flat variant and becomes a record [{ pconst_desc; pconst_loc }], from [Ast_503] on
   -- costs these two definitions rather than every call site.

   [Ast_pattern.estring] would say "a string literal" without naming a constructor at all, but its
   generated combinators assert the node carries no attributes and RAISE when it does, where a plain
   match simply does not select the arm. That would turn several friendly "expected a string
   literal" error extensions into hard failures, so it is declined here. *)
let string_literal expr =
  match expr.pexp_desc with Pexp_constant (Pconst_string (s, _, _)) -> Some s | _ -> None

let string_of_constant = function
  | Pconst_string (s, _, _) | Pconst_integer (s, _) | Pconst_float (s, _) -> s
  | Pconst_char c -> Char.to_string c

let ndarray_constant expr =
  let loc = expr.pexp_loc in
  (* Traverse the backbone of the ndarray to collect the dimensions and any per-axis basis labels. A
     [(container : tag)] type-annotation labels that axis's dimension basis: [([…] : rgb)] for an
     output (list) axis, [((…) : feat)] for an input (tuple) axis, [([|…|] : batch)] for a batch
     (array) axis. [?basis] carries the label of the container currently being matched. *)
  let rec loop_dims ?basis accu = function
    | { pexp_desc = Pexp_constraint (inner, typ); _ } ->
        loop_dims ?basis:(axis_basis_of_type typ) accu inner
    | { pexp_desc = Pexp_tuple (exp :: _ as exps); _ } ->
        loop_dims (`Input_dims (List.length exps, basis) :: accu) exp
    | { pexp_desc = Pexp_array (exp :: _ as exps); _ } ->
        loop_dims (`Batch_dims (List.length exps, basis) :: accu) exp
    | { pexp_desc = Pexp_tuple []; _ } -> `Input_dims (0, basis) :: accu
    | [%expr [||]] -> `Batch_dims (0, basis) :: accu
    | [%expr [%e? _] :: [%e? _]] as expr -> (
        let exps = collect_list [] expr in
        match exps with
        | exp :: _ -> loop_dims (`Output_dims (List.length exps, basis) :: accu) exp
        | [] -> `Output_dims (0, basis) :: accu)
    | _ -> accu
  in
  let dims_spec = Array.of_list_rev @@ loop_dims [] expr in
  let open Ast_builder.Default in
  let rec loop_values depth accu expr =
    (* See through an axis-label type-annotation to the container it wraps. *)
    match expr with
    | { pexp_desc = Pexp_constraint (inner, _); _ } -> loop_values depth accu inner
    | _ -> (
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
          | [%expr [%e? _] :: [%e? _]] ->
              (pexp_extension ~loc
              @@ Location.error_extensionf ~loc
                   "Arrayjit: ndarray literal found output axis (list), expected number")
              :: accu
          | expr -> expr :: accu (* it either computes a number, or becomes a type error *)
        else
          match expr with
          | { pexp_desc = Pexp_tuple exps; _ } -> (
              match dims_spec.(depth) with
              | `Input_dims (dim, _) when dim = List.length exps ->
                  List.fold_left exps ~init:accu ~f:(loop_values @@ (depth + 1))
              | dim_spec ->
                  (pexp_extension ~loc
                  @@ Location.error_extensionf ~loc
                       "Arrayjit: ndarray literal axis mismatch, got %s, expected %s"
                       (dim_spec_to_string @@ `Input_dims (List.length exps, None))
                       (dim_spec_to_string dim_spec))
                  :: accu)
          | { pexp_desc = Pexp_array exps; _ } -> (
              match dims_spec.(depth) with
              | `Batch_dims (dim, _) when dim = List.length exps ->
                  List.fold_left exps ~init:accu ~f:(loop_values @@ (depth + 1))
              | dim_spec ->
                  (pexp_extension ~loc
                  @@ Location.error_extensionf ~loc
                       "Arrayjit: ndarray literal axis mismatch, got %s, expected %s"
                       (dim_spec_to_string @@ `Batch_dims (List.length exps, None))
                       (dim_spec_to_string dim_spec))
                  :: accu)
          | [%expr [%e? _] :: [%e? _]] -> (
              let exps = collect_list [] expr in
              match dims_spec.(depth) with
              | `Output_dims (dim, _) when dim = List.length exps ->
                  List.fold_left exps ~init:accu ~f:(loop_values @@ (depth + 1))
              | dim_spec ->
                  (pexp_extension ~loc
                  @@ Location.error_extensionf ~loc
                       "Arrayjit: ndarray literal axis mismatch, got %s, expected %s"
                       (dim_spec_to_string @@ `Output_dims (List.length exps, None))
                       (dim_spec_to_string dim_spec))
                  :: accu)
          | { pexp_loc = loc; _ } ->
              (pexp_extension ~loc
              @@ Location.error_extensionf ~loc
                   "Arrayjit: ndarray literal: expected an axis (tuple, list or array)")
              :: accu)
  in
  let result = loop_values 0 [] expr in
  let values = { expr with pexp_desc = Pexp_array (List.rev result) } in
  (* Each kind's axes are returned as [(dim_expr, basis option)] pairs (outer-to-inner order). *)
  let batch_dims, output_dims, input_dims =
    Array.fold dims_spec ~init:([], [], [])
      ~f:(fun (batch_dims, output_dims, input_dims) -> function
      | `Input_dims (dim, basis) -> (batch_dims, output_dims, (eint ~loc dim, basis) :: input_dims)
      | `Output_dims (dim, basis) -> (batch_dims, (eint ~loc dim, basis) :: output_dims, input_dims)
      | `Batch_dims (dim, basis) -> ((eint ~loc dim, basis) :: batch_dims, output_dims, input_dims))
  in
  (values, List.rev batch_dims, List.rev output_dims, List.rev input_dims)

(** Convert an einsum spec string to an OCaml expression that constructs the runtime string.

    This function parses the einsum spec using the Einsum_parser, then reconstructs a runtime string
    expression, handling:
    - stride and dilation values: if they look like integer literals, emit them directly; otherwise
      emit [Int.to_string identifier] to convert at runtime
    - use_padding: if unspecified (legacy syntax), emit [if use_padding then "=" else "<"] to read
      the value from [Row.use_padding] at runtime

    Example: ["stride*x=+k; y => z"] where [stride] is a variable, generates an expression that
    evaluates to e.g. ["2*x=+k; y => z"] if [stride = 2]. *)
let substitute_identifiers_in_einsum_spec ~loc str_input =
  let open Ast_builder.Default in
  let open Einsum_parser in
  (* Helper to check if a string is an integer literal *)
  let is_int_literal s =
    try
      ignore (Int.of_string s);
      true
    with _ -> false
  in
  (* Convert a string that might be an identifier to a runtime int expression *)
  let int_value_expr s =
    if is_int_literal s then estring ~loc s
    else [%expr Int.to_string [%e pexp_ident ~loc (Located.mk ~loc (Lident s))]]
  in
  (* Convert use_padding_spec to a string expression *)
  let use_padding_expr = function
    | `True -> estring ~loc "="
    | `False -> estring ~loc "<"
    | `Unspecified -> [%expr if use_padding then "=" else "<"]
  in
  (* Convert a conv_spec to string segments *)
  let conv_to_segments conv =
    let dilation_expr = int_value_expr conv.dilation in
    let padding_expr = use_padding_expr conv.use_padding in
    if String.equal conv.dilation "1" then
      [ padding_expr; estring ~loc "+"; estring ~loc conv.kernel_label ]
    else
      [
        padding_expr;
        estring ~loc "+";
        dilation_expr;
        estring ~loc "*";
        estring ~loc conv.kernel_label;
      ]
  in
  (* Convert axis_spec to string segments *)
  let axis_spec_to_segments spec =
    match spec with
    | Label s -> [ estring ~loc s ]
    | Fixed_index i -> [ estring ~loc (Int.to_string i) ]
    | Affine_spec { stride; over_label; conv; stride_offset } ->
        let stride_expr = int_value_expr stride in
        let base_segments =
          if String.equal stride "1" then [ estring ~loc over_label ]
          else [ stride_expr; estring ~loc "*"; estring ~loc over_label ]
        in
        let offset_segments =
          if stride_offset = 0 then []
          else [ estring ~loc "+"; estring ~loc (Int.to_string stride_offset) ]
        in
        let conv_segments = match conv with None -> [] | Some c -> conv_to_segments c in
        base_segments @ conv_segments @ offset_segments
    | Concat_spec labels ->
        (* Join labels with " ^ " separator *)
        List.concat_mapi labels ~f:(fun i label ->
            let prefix = if i > 0 then [ estring ~loc " ^ " ] else [] in
            prefix @ [ estring ~loc label ])
  in
  (* Convert a list of axis_spec to segments with comma separators *)
  let axes_to_segments axes =
    List.concat_mapi axes ~f:(fun i spec ->
        let prefix = if i > 0 then [ estring ~loc ", " ] else [] in
        prefix @ axis_spec_to_segments spec)
  in
  (* Convert a row (bcast, given, given_beg) to segments *)
  let row_to_segments ~kind bcast given_beg given =
    let bcast_segment =
      match bcast with
      | None -> []
      | Some s ->
          if String.equal s ("..." ^ kind) then [ estring ~loc "..." ]
          else [ estring ~loc ".."; estring ~loc s; estring ~loc ".." ]
    in
    let beg_segments = axes_to_segments given_beg in
    let end_segments = axes_to_segments given in
    let comma_before_bcast =
      if List.is_empty beg_segments || List.is_empty bcast_segment then []
      else [ estring ~loc ", " ]
    in
    let comma_after_bcast =
      if List.is_empty bcast_segment || List.is_empty end_segments then []
      else [ estring ~loc ", " ]
    in
    beg_segments @ comma_before_bcast @ bcast_segment @ comma_after_bcast @ end_segments
  in
  (* Convert parsed_axis_labels to segments *)
  let parsed_to_segments parsed =
    let batch_segments =
      row_to_segments ~kind:"batch" parsed.bcast_batch parsed.given_beg_batch parsed.given_batch
    in
    let input_segments =
      row_to_segments ~kind:"input" parsed.bcast_input parsed.given_beg_input parsed.given_input
    in
    let output_segments =
      row_to_segments ~kind:"output" parsed.bcast_output parsed.given_beg_output parsed.given_output
    in
    (* Minimal-form rendering: a row parsed as implicit (omitted together with its kind separator)
       is rendered as omitted again — reparsing restores it. Any explicitly written row, including a
       closed (empty) one and an explicit ellipsis, must keep its separator so that the
       reconstructed string preserves the closed-vs-implicit distinction. *)
    (if parsed.implicit_batch then [] else batch_segments @ [ estring ~loc "|" ])
    @ (if parsed.implicit_input then [] else input_segments @ [ estring ~loc "->" ])
    @ output_segments
  in
  (* Fold the segments into one literal when every one of them is a literal, so the common case
     costs nothing at run time. *)
  let concat_segments segments =
    match Option.all (List.map segments ~f:string_literal) with
    | Some literals -> estring ~loc (String.concat literals)
    | None -> [%expr String.concat ~sep:"" [%e elist ~loc segments]]
  in
  (* Try to parse as einsum spec *)
  try
    let rhs_labels, labels_r = einsum_of_spec str_input in
    let rhs_segments =
      List.mapi rhs_labels ~f:(fun i labels ->
          let segs = parsed_to_segments labels in
          if i = 0 then segs else [ estring ~loc "; " ] @ segs)
      |> List.concat
    in
    let segments_r = [ estring ~loc " => " ] @ parsed_to_segments labels_r in
    let all_segments = rhs_segments @ segments_r in
    concat_segments all_segments
  with Parse_error _ -> (
    (* If parsing fails, try as axis_labels_spec *)
    try
      let parsed = axis_labels_of_spec str_input in
      concat_segments (parsed_to_segments parsed)
    with Parse_error msg ->
      (* Fall back to returning the original string with an error note *)
      pexp_extension ~loc @@ Location.error_extensionf ~loc "Failed to parse einsum spec: %s" msg)

let string_expr ~loc s = Ast_helper.Exp.constant @@ Ast_helper.Const.string ~loc s

let string_of_pat pat =
  let rec lident = function Lident s | Ldot (_, s) -> s | Lapply (_, i) -> lident i in
  let rec loop pat =
    match pat.ppat_desc with
    | Ppat_open (_, pat) | Ppat_lazy pat | Ppat_constraint (pat, _) -> loop pat
    | Ppat_alias (_, ident) -> ident.txt
    | Ppat_var ident -> ident.txt
    | Ppat_any -> "_"
    | Ppat_variant (s, _) -> s
    | Ppat_constant c -> string_of_constant c
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
    | Pexp_variant (s, _) -> s
    | Pexp_constant c -> string_of_constant c
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
      ("<=", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Cmple]));
      ("le", fun loc -> ([%expr Shape.Pointwise_bin], [%expr Ir.Ops.Cmple]));
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
      ("mul3", fun loc -> ([%expr Shape.Pointwise_tern], [%expr Ir.Ops.Mul3]));
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

let einsum_binary_ops =
  Hashtbl.of_alist_exn
    (module String)
    (List.map Einsum_parser.binary_operators_with_generated_specs ~f:(function
      | "+*" -> ("+*", fun loc -> [%expr einsum])
      | "@^+" -> ("@^+", fun loc -> [%expr tropical])
      | "+++" -> ("+++", fun loc -> [%expr outer_sum])
      | operator -> failwith ("missing PPX binary einsum handler for " ^ operator)))

let einsum_unary_ops =
  Hashtbl.of_alist_exn
    (module String)
    (List.map Einsum_parser.unary_operators_with_generated_specs ~f:(function
      | "++" -> ("++", fun loc -> [%expr einsum1])
      | "@^^" -> ("@^^", fun loc -> [%expr einmax1])
      | operator -> failwith ("missing PPX unary einsum handler for " ^ operator)))

let is_primitive_op op_ident =
  List.exists ~f:(Fn.flip Hashtbl.mem op_ident) [ ternary_ops; unary_ops; binary_ops ]

let let_opt ~loc vbs expr =
  (* Check for duplicates and create nested let bindings preserving definition order *)
  let seen = Hashtbl.create (module String) in
  List.fold_right vbs ~init:expr ~f:(fun vb acc ->
      let name = match vb.pvb_pat.ppat_desc with Ppat_var { txt; _ } -> txt | _ -> "_" in
      match Hashtbl.add seen ~key:name ~data:() with
      | `Ok -> Ast_helper.Exp.let_ ~loc Nonrecursive [ vb ] acc
      | `Duplicate ->
          let loc = vb.pvb_loc in
          let error_expr =
            Ast_builder.Default.pexp_extension ~loc
            @@ Location.error_extensionf ~loc
                 "ppx_ocannl: name clash for inline definition or variable capture '%s' - the name \
                  is already defined"
                 name
          in
          Ast_helper.Exp.let_ ~loc Nonrecursive [ { vb with pvb_expr = error_expr } ] acc)

let no_vbs = []
let reduce_vbss vbss = List.concat vbss

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

(* Build the [(arg-name, expr)] for one axis kind from its [(dim_expr, basis option)] pairs. If any
   axis carries a tensor-literal basis label, emit [<kind>_axes = [(basis, dim); …]] (unlabelled
   axes in that kind take the reserved ["default"] tag); otherwise [<kind>_dims = [dim; …]]. *)
let axes_or_dims_named ~loc ~kind (pairs : (expression * string option) list) : string * expression
    =
  let open Ast_builder.Default in
  if List.exists pairs ~f:(fun (_, b) -> Option.is_some b) then
    let axes =
      List.map pairs ~f:(fun (dim, b) ->
          let basis = Option.value b ~default:"default" in
          [%expr [%e estring ~loc basis], [%e dim]])
    in
    (kind ^ "_axes", elist ~loc axes)
  else (kind ^ "_dims", elist ~loc (List.map pairs ~f:fst))

let axes_or_dims_arg ~loc ~kind pairs : Ppxlib.arg_label * expression =
  let name, expr = axes_or_dims_named ~loc ~kind pairs in
  (Labelled name, expr)

let ndarray_op ?axis_labels ?label ~ndarray_fn expr =
  let loc = expr.pexp_loc in
  let values, batch, output, input = ndarray_constant expr in
  let w_val = [%expr [%e ndarray_fn] [%e values]] in
  let op =
    match (axis_labels, label) with
    | None, None -> w_val
    | Some axis_labels, None -> [%expr [%e w_val] ~axis_labels:[%e axis_labels]]
    | None, Some label -> [%expr [%e w_val] ~label:[%e label]]
    | Some axis_labels, Some label ->
        [%expr [%e w_val] ~axis_labels:[%e axis_labels] ~label:[%e label]]
  in
  Ast_builder.Default.pexp_apply ~loc op
    [
      axes_or_dims_arg ~loc ~kind:"batch" batch;
      axes_or_dims_arg ~loc ~kind:"input" input;
      axes_or_dims_arg ~loc ~kind:"output" output;
      (Nolabel, [%expr ()]);
    ]

let collect_capture_labels ~loc head rest =
  let capture_labels = head :: collect_list [] rest in
  let capture_labels, errors =
    List.partition_map capture_labels ~f:(fun expr ->
        match string_literal expr with
        | Some label -> Either.First (expr.pexp_loc, label)
        | None ->
            Either.Second
              (Ast_builder.Default.pexp_extension ~loc:expr.pexp_loc
              @@ Location.error_extensionf ~loc:expr.pexp_loc
                   "ppx_ocannl %%op: expected a string literal"))
  in
  let capture_labels, more_errors =
    List.fold_left capture_labels ~init:([], []) ~f:(fun (labels, errors) ((loc, label) as arg) ->
        if List.mem labels arg ~equal:(fun (_, a) (_, b) -> String.equal a b) then
          ( labels,
            (Ast_builder.Default.pexp_extension ~loc
            @@ Location.error_extensionf ~loc "ppx_ocannl %%op: repeated variable capture '%s'"
                 label)
            :: errors )
        else (arg :: labels, errors))
  in
  let capture_refs, capture_bindings =
    List.map capture_labels ~f:(fun (loc, label) ->
        let ref_expr = [%expr Shape.get_variable_ref [%e Ast_builder.Default.estring ~loc label]] in
        let binding =
          Ast_builder.Default.value_binding ~loc
            ~pat:(Ast_builder.Default.pvar ~loc label)
            ~expr:ref_expr
        in
        (Ast_builder.Default.evar ~loc label, (label, binding)))
    |> List.unzip
  in
  let capture_dims_expr = Ast_builder.Default.elist ~loc (more_errors @ errors @ capture_refs) in
  let capture_vbs = List.map capture_bindings ~f:snd in
  (capture_vbs, capture_dims_expr)

let operators =
  (* TODO: Auto-generate this list from Operation.Make_DSL.O. *)
  Hashtbl.of_alist_exn
    (module String)
    [
      ("*", "matmul");
      ("*.", "pointmul");
      ("+", "add");
      ("**.", "pointpow");
      ("!.", "number");
      ("!..", "number_int");
      ("!%", "bits");
      ("!@", "embed_symbol");
      ("dim", "embed_dim");
      ("-", "sub");
      ("~-", "num_neg");
      ("/.", "pointdiv");
      ("@|", "slice");
      ("<", "lt");
      ("=", "eq");
      ("<>", "ne");
    ]

let add_module_qualifier_to_applied_function ?(module_name = "NTDSL") expr =
  let qualify_if_needed fn =
    match fn.pexp_desc with
    | Pexp_ident { txt = Lident name; loc } ->
        let op_name = Option.value ~default:name @@ Hashtbl.find operators name in
        Ast_builder.Default.pexp_ident ~loc { txt = Ldot (Lident module_name, op_name); loc }
    | _ -> fn
  in
  let rec decompose_app expr acc =
    match expr.pexp_desc with
    | Pexp_apply (fn, args) -> decompose_app fn (args @ acc)
    | _ -> (expr, acc)
  in
  let rec process_expr expr =
    let loc = expr.pexp_loc in
    match expr.pexp_desc with
    | Pexp_apply (_, _) ->
        let fn, args = decompose_app expr [] in
        let qualified_fn = qualify_if_needed fn in
        Ast_builder.Default.pexp_apply ~loc qualified_fn args
    | Pexp_ifthenelse (cond, then_expr, else_expr) ->
        let processed_then = process_expr then_expr in
        let processed_else = Option.map else_expr ~f:process_expr in
        Ast_builder.Default.pexp_ifthenelse ~loc cond processed_then processed_else
    | Pexp_sequence (expr1, expr2) ->
        let processed_expr2 = process_expr expr2 in
        Ast_builder.Default.pexp_sequence ~loc expr1 processed_expr2
    | Pexp_let (recflag, bindings, body) ->
        let processed_body = process_expr body in
        Ast_builder.Default.pexp_let ~loc recflag bindings processed_body
    | Pexp_open (decl, expr) ->
        let processed_expr = process_expr expr in
        Ast_builder.Default.pexp_open ~loc decl processed_expr
    | Pexp_function (params, cnstr, Pfunction_body body) ->
        let body = process_expr body in
        Ast_builder.Default.pexp_function ~loc params cnstr (Pfunction_body body)
    | Pexp_function (params, cnstr, Pfunction_cases (cases, loc, attrs)) ->
        let cases =
          List.map cases ~f:(fun case -> { case with pc_rhs = process_expr case.pc_rhs })
        in
        Ast_builder.Default.pexp_function ~loc params cnstr (Pfunction_cases (cases, loc, attrs))
    | Pexp_extension ({ txt = "oc"; _ }, PStr [ { pstr_desc = Pstr_eval (e, _); _ } ]) -> e
    | _ -> expr
  in
  process_expr expr
