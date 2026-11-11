(** Helpers shared by the [%op], [%cd] and [%extend_dsls] translators: AST-to-string conversions,
    the primitive and einsum operator tables, ndarray literals, and inline-binding management. *)

open Base
open Ppxlib

val collect_list : expression list -> expression -> expression list
(** The elements of list literal [e] after [List.rev acc]; a non-literal tail is an error node. *)

val ndarray_constant :
  expression ->
  expression
  * (expression * string option) list
  * (expression * string option) list
  * (expression * string option) list
(** Parses an ndarray literal into its flat values and batch, output, input [(dim, basis)] axes. *)

val substitute_identifiers_in_einsum_spec : loc:location -> string -> expression
(** An expression building an einsum or axis-labels spec, reading identifier strides at run time. *)

val einsum_spec_identifiers : string -> string list
(** The identifiers [substitute_identifiers_in_einsum_spec] reads as integers at run time: the
    strides and dilations that are not integer literals. Empty for a spec that does not parse. *)

val string_expr : loc:location -> string -> expression
(** The string literal expression [s] at [loc]. *)

val pat2string : pattern -> expression
(** A string literal naming a pattern: its variable, alias, constant or constructor, else [""]. *)

val collect_pat_idents : pattern -> Set.M(String).t
(** The variables a pattern binds; under an [as] alias, only the alias name is collected. *)

val expr2string_or_empty : expression -> expression
(** A string literal rendering an identifier, constant, constructor, tuple or array; else [""]. *)

val opt_pat2string_list : loc:location -> pattern option -> expression
(** The list expression holding {!pat2string} of the pattern, or [[]] for [None]. *)

val opt_expr : loc:location -> expression option -> expression
(** Lifts an optional expression into the expression [Some e] or [None]. *)

val pat2expr : pattern -> expression
(** The expression mirroring a pattern ([x as y] gives [y]); unsupported patterns give an error. *)

val is_operator : string -> bool
(** Whether an identifier is an operator: non-empty, with no ASCII letter or digit. *)

val is_assignment : string -> bool
(** Whether an identifier is [=] followed by more, excluding [==], [===], [=>], [==>] and [=>>]. *)

val binary_ops : (string, location -> expression * expression) Hashtbl.t
(** Binary primitives, by operator and by name, to their [Shape] logic and [Ir.Ops] operator. *)

val unary_ops : (string, location -> expression * expression) Hashtbl.t
(** Unary primitives, by name, to their [Shape] logic and [Ir.Ops] operator. *)

val vec_unary_ops : (string, location -> expression * expression) Hashtbl.t
(** Vector unary primitives, by name, to their [Shape] logic and [Ir.Ops] vector operator. *)

val ternary_ops : (string, location -> expression * expression) Hashtbl.t
(** Ternary primitives, by name, to their [Shape] logic and [Ir.Ops] operator. *)

val assignment_ops : (string, location -> bool * expression) Hashtbl.t
(** Assignment operators to whether the target starts at the op's neutral element, and the op. *)

val einsum_binary_ops : (string, location -> expression) Hashtbl.t
(** Binary einsum operators ([+*], [@^+], [+++]) to the DSL function they apply. *)

val einsum_unary_ops : (string, location -> expression) Hashtbl.t
(** Unary einsum operators ([++], [@^^]) to the DSL function they apply. *)

val is_named_spec_function : expression -> bool
(** Whether the expression names a named einsum form ([Einsum_parser.named_spec_functions],
    unqualified). *)

val named_form_spec : expression -> expression option
(** The spec of an application of a named einsum form ([Einsum_parser.named_spec_functions],
    unqualified): its first positional argument. *)

val substitute_named_form_spec : expression -> expression
(** The application of a named einsum form with its literal spec substituted as an operator's is
    ({!substitute_identifiers_in_einsum_spec}), when the spec reads a runtime value: a stride or
    dilation identifier, or a plain [+] window reading [use_padding] (gh-ocannl-2060). Anything
    else, a spec reading no runtime value or one that does not parse included, is returned as it is.
*)

val is_primitive_op : string -> bool
(** Whether a name is in {!binary_ops}, {!unary_ops} or {!ternary_ops}. *)

val let_opt : loc:location -> value_binding list -> expression -> expression
(** Binds [vbs] around [e] as nested [let]s in order; a repeated or capture-hiding name errors. *)

val bind_captures :
  loc:location -> value_binding list -> expression -> value_binding list * expression
(** Binds the dimension captures in [vbs] atop a function body; returns the rest, plus markers. *)

val collect_capture_labels :
  loc:location -> spec:string -> expression -> expression -> value_binding list * expression
(** For capture list [head :: rest]: a [Shape.get_variable_ref] binding per label, and the list.
    Refuses a label absent from the literal source [spec], at the capture literal's location. *)

val no_vbs : 'a list
(** The empty list of collected bindings. *)

val reduce_vbss : 'a list list -> 'a list
(** Concatenates lists of collected bindings. *)

val in_source_order :
  loc:location ->
  prefix:string ->
  expression list ->
  f:(expression list -> expression) ->
  expression
(** [f exprs'] under [let]s that run the effects of [exprs] in source order: every member but a
    variable, a constant or the last is bound to [prefix ^ "<position>"], [prefix] extended with
    underscores until no identifier in [exprs] starts with it. *)

val expr_expander_with_punning :
  (?ident_label:pattern -> expression -> value_binding list * expression) ->
  loc:location ->
  path:'a ->
  expression ->
  expression
(** An expression expander from [translate]; a [let] payload's bindings are labelled by pattern. *)

val str_expander_with_punning :
  (?ident_label:pattern -> expression -> value_binding list * expression) ->
  loc:location ->
  path:'a ->
  structure_item list ->
  structure_item
(** A structure expander from [translate] over [let] bindings (labelled by pattern) and [eval]s. *)

val axes_or_dims_named :
  loc:location -> kind:string -> (expression * string option) list -> string * expression
(** One axis kind's argument: [<kind>_axes] of [(basis, dim)] if labelled, else [<kind>_dims]. *)

val ndarray_op :
  ?axis_labels:expression -> ?label:expression -> ndarray_fn:expression -> expression -> expression
(** Applies [ndarray_fn] to an ndarray literal's values, then batch/input/output axes and [()]. *)

val add_module_qualifier_to_applied_function : ?module_name:string -> expression -> expression
(** Qualifies tail applications' functions by [module_name] (default [NTDSL]), [*] as [matmul]. *)
