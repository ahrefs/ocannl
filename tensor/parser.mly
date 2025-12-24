%{
(** Parser for einsum notation.

    Supports parsing of:
    - Single axis specifications (labels, fixed indices, conv expressions)
    - Row specifications (with optional ellipsis)
    - Full shape specifications (batch|input->output)
    - Einsum specifications (spec1; spec2 => result)
    - Permute specifications (spec => result)

    The lexer determines whether to use multichar or single-char mode.
*)

open Base
open Einsum_types

(* Helper functions *)

let make_axes_map ~in_axes ~from_end specs =
  let f pos spec = (({ in_axes; pos; from_end } : AxisKey.t), spec) in
  let n = List.length specs in
  let indexed =
    if from_end then
      List.mapi specs ~f:(fun i spec -> f (n - i) spec)
    else
      List.mapi specs ~f:(fun i spec -> f (i + 1) spec)
  in
  Map.of_alist_exn (module AxisKey) indexed

let merge_maps m1 m2 =
  Map.merge_skewed m1 m2 ~combine:(fun ~key:_ _v1 _v2 ->
    failwith "Duplicate axis key")

let make_row_spec ~kind in_axes (beg_axes, row_var_spec, end_axes) =
  let from_beg = make_axes_map ~in_axes ~from_end:false beg_axes in
  let from_end = make_axes_map ~in_axes ~from_end:true end_axes in
  ( Option.map row_var_spec ~f:(fun rv -> if String.equal rv "..." then kind else rv),
    end_axes,
    beg_axes,
    merge_maps from_beg from_end )

let default_row = (None, [], [], Map.empty (module AxisKey))

let make_parsed_labels batch_opt input_opt output =
  let (bcast_batch, given_batch, given_beg_batch, batch_labels) =
    Option.value batch_opt ~default:default_row
  in
  let (bcast_input, given_input, given_beg_input, input_labels) =
    Option.value input_opt ~default:default_row
  in
  let (bcast_output, given_output, given_beg_output, output_labels) = output in
  let labels = merge_maps input_labels (merge_maps output_labels batch_labels) in
  {
    bcast_batch;
    bcast_input;
    bcast_output;
    given_batch;
    given_input;
    given_output;
    given_beg_batch;
    given_beg_input;
    given_beg_output;
    labels;
  }

%}

/* Token declarations */
%token <string> IDENT
%token <int> INT
%token COMMA
%token PIPE
%token ARROW        /* -> */
%token DOUBLE_ARROW /* => */
%token SEMICOLON
%token PLUS
%token STAR
%token CARET        /* ^ */
%token AMPERSAND    /* & - reserved for future use */
%token UNDERSCORE   /* _ */
%token ELLIPSIS     /* ... */
%token DOT_DOT      /* .. */
%token EQUALS       /* = - use_padding=true */
%token LESS_THAN    /* < - use_padding=false */
%token EOF

/* Start symbols */
%start <parsed_axis_labels> axis_labels_spec
%start <parsed_axis_labels list * parsed_axis_labels> einsum_spec

%%

/* Single axis specification */
axis_spec:
  | i = INT
    { Fixed_index i }
  | id = IDENT
    { Label id }
  | UNDERSCORE
    { Label "_" }
  | affine_expr
    { $1 }

/* Helper for converting int to string *)
%inline int_as_string:
  | n = INT { Int.to_string n }

/* Helper for stride value (int or identifier) */
%inline stride_value:
  | n = INT { Int.to_string n }
  | id = IDENT { id }

/* Helper for use_padding specification */
%inline use_padding_marker:
  | EQUALS { `True }
  | LESS_THAN { `False }

/* Affine expression: [stride*]over[=|<][+offset][+[dilation*]kernel]
   Supports various combinations of stride, offset, and convolution components.
   = after over means use_padding=true, < means use_padding=false.
   The underscore (_) can be used as a placeholder in convolution syntax.
   stride and dilation can be int literals or identifiers. */
affine_expr:
  /* Full form with use_padding: stride*over[=|<]+offset+dilation*kernel */
  | stride = stride_value; STAR; over = IDENT; padding = use_padding_marker; PLUS; offset = INT; PLUS; dilation = stride_value; STAR; kernel = IDENT
    { Affine_spec { stride; over_label = over; stride_offset = offset;
                    conv = Some { dilation; kernel_label = kernel; use_padding = padding } } }
  /* stride*over[=|<]+offset+kernel (dilation=1) */
  | stride = stride_value; STAR; over = IDENT; padding = use_padding_marker; PLUS; offset = INT; PLUS; kernel = IDENT
    { Affine_spec { stride; over_label = over; stride_offset = offset;
                    conv = Some { dilation = "1"; kernel_label = kernel; use_padding = padding } } }
  /* over[=|<]+offset+dilation*kernel (stride=1) */
  | over = IDENT; padding = use_padding_marker; PLUS; offset = INT; PLUS; dilation = stride_value; STAR; kernel = IDENT
    { Affine_spec { stride = "1"; over_label = over; stride_offset = offset;
                    conv = Some { dilation; kernel_label = kernel; use_padding = padding } } }
  /* over[=|<]+offset+kernel (stride=1, dilation=1) */
  | over = IDENT; padding = use_padding_marker; PLUS; offset = INT; PLUS; kernel = IDENT
    { Affine_spec { stride = "1"; over_label = over; stride_offset = offset;
                    conv = Some { dilation = "1"; kernel_label = kernel; use_padding = padding } } }
  /* stride*over[=|<]+dilation*kernel (no offset) */
  | stride = stride_value; STAR; over = IDENT; padding = use_padding_marker; PLUS; dilation = stride_value; STAR; kernel = IDENT
    { Affine_spec { stride; over_label = over; stride_offset = 0;
                    conv = Some { dilation; kernel_label = kernel; use_padding = padding } } }
  /* stride*over[=|<]+kernel (no offset, dilation=1) */
  | stride = stride_value; STAR; over = IDENT; padding = use_padding_marker; PLUS; kernel = IDENT
    { Affine_spec { stride; over_label = over; stride_offset = 0;
                    conv = Some { dilation = "1"; kernel_label = kernel; use_padding = padding } } }
  /* over[=|<]+dilation*kernel (stride=1, no offset) */
  | over = IDENT; padding = use_padding_marker; PLUS; dilation = stride_value; STAR; kernel = IDENT
    { Affine_spec { stride = "1"; over_label = over; stride_offset = 0;
                    conv = Some { dilation; kernel_label = kernel; use_padding = padding } } }
  /* over[=|<]+kernel (stride=1, dilation=1, no offset) */
  | over = IDENT; padding = use_padding_marker; PLUS; kernel = IDENT
    { Affine_spec { stride = "1"; over_label = over; stride_offset = 0;
                    conv = Some { dilation = "1"; kernel_label = kernel; use_padding = padding } } }
  /* Unspecified use_padding (legacy syntax without = or <, implies no convolution or use_padding=`Unspecified) */
  /* Full form: stride*over+offset+dilation*kernel (unspecified use_padding) */
  | stride = stride_value; STAR; over = IDENT; PLUS; offset = INT; PLUS; dilation = stride_value; STAR; kernel = IDENT
    { Affine_spec { stride; over_label = over; stride_offset = offset;
                    conv = Some { dilation; kernel_label = kernel; use_padding = `Unspecified } } }
  /* stride*over+offset+kernel (dilation=1, unspecified use_padding) */
  | stride = stride_value; STAR; over = IDENT; PLUS; offset = INT; PLUS; kernel = IDENT
    { Affine_spec { stride; over_label = over; stride_offset = offset;
                    conv = Some { dilation = "1"; kernel_label = kernel; use_padding = `Unspecified } } }
  /* over+offset+dilation*kernel (stride=1, unspecified use_padding) */
  | over = IDENT; PLUS; offset = INT; PLUS; dilation = stride_value; STAR; kernel = IDENT
    { Affine_spec { stride = "1"; over_label = over; stride_offset = offset;
                    conv = Some { dilation; kernel_label = kernel; use_padding = `Unspecified } } }
  /* over+offset+kernel (stride=1, dilation=1, unspecified use_padding) */
  | over = IDENT; PLUS; offset = INT; PLUS; kernel = IDENT
    { Affine_spec { stride = "1"; over_label = over; stride_offset = offset;
                    conv = Some { dilation = "1"; kernel_label = kernel; use_padding = `Unspecified } } }
  /* stride*over+dilation*kernel (no offset, unspecified use_padding) */
  | stride = stride_value; STAR; over = IDENT; PLUS; dilation = stride_value; STAR; kernel = IDENT
    { Affine_spec { stride; over_label = over; stride_offset = 0;
                    conv = Some { dilation; kernel_label = kernel; use_padding = `Unspecified } } }
  /* stride*over+kernel (no offset, dilation=1, unspecified use_padding) */
  | stride = stride_value; STAR; over = IDENT; PLUS; kernel = IDENT
    { Affine_spec { stride; over_label = over; stride_offset = 0;
                    conv = Some { dilation = "1"; kernel_label = kernel; use_padding = `Unspecified } } }
  /* over+dilation*kernel (stride=1, no offset, unspecified use_padding) */
  | over = IDENT; PLUS; dilation = stride_value; STAR; kernel = IDENT
    { Affine_spec { stride = "1"; over_label = over; stride_offset = 0;
                    conv = Some { dilation; kernel_label = kernel; use_padding = `Unspecified } } }
  /* over+kernel (stride=1, dilation=1, no offset, unspecified use_padding) */
  | over = IDENT; PLUS; kernel = IDENT
    { Affine_spec { stride = "1"; over_label = over; stride_offset = 0;
                    conv = Some { dilation = "1"; kernel_label = kernel; use_padding = `Unspecified } } }
  /* stride*over+offset (no conv) */
  | stride = stride_value; STAR; over = IDENT; PLUS; offset = INT
    { Affine_spec { stride; over_label = over; stride_offset = offset; conv = None } }
  /* stride*over (no offset, no conv) */
  | stride = stride_value; STAR; over = IDENT
    { Affine_spec { stride; over_label = over; stride_offset = 0; conv = None } }
  | head = IDENT; CARET; tail = separated_nonempty_list(CARET, IDENT)
    { Concat_spec (head :: tail) }

/* List of axis specifications - can be empty, allows trailing comma */
axes_spec:
  | /* empty */ { [] }
  | l = axes_list { l }

axes_list:
  | x = axis_spec { [x] }
  | x = axis_spec; COMMA; xs = axes_spec { x :: xs }

/* Ellipsis specification */
ellipsis_spec:
  | ELLIPSIS
    { "..." }
  | DOT_DOT; id = IDENT; option(COMMA); DOT_DOT
    { id }

/* Row specification with optional ellipsis (for one kind of axes) */
row_spec:
  /* beg_axes ellipsis end_axes */
  | beg = axes_spec; option(COMMA); ell = ellipsis_spec; option(COMMA); end_ = axes_spec; option(COMMA)
    { (beg, Some ell, end_) }
  /* beg_axes ellipsis */
  | beg = axes_spec; option(COMMA); ell = ellipsis_spec; option(COMMA)
    { (beg, Some ell, []) }
  /* ellipsis end_axes */
  | ell = ellipsis_spec; option(COMMA); end_ = axes_spec; option(COMMA)
    { ([], Some ell, end_) }
  /* just ellipsis */
  | ell = ellipsis_spec; option(COMMA)
    { ([], Some ell, []) }
  /* just axes (no ellipsis) */
  | specs = axes_spec; option(COMMA)
    { ([], None, specs) }

/* Shape specification: [batch|][input->]output */
shape_spec:
  /* batch|input->output */
  | batch = row_spec; PIPE; input = row_spec; ARROW; output = row_spec
    { (Some batch, Some input, output) }
  /* batch|output */
  | batch = row_spec; PIPE; output = row_spec
    { (Some batch, None, output) }
  /* input->output */
  | input = row_spec; ARROW; output = row_spec
    { (None, Some input, output) }
  /* just output */
  | output = row_spec
    { (None, None, output) }

/* Axis labels specification (entry point) */
axis_labels_spec:
  | spec = shape_spec; EOF
    { let (batch, input, output) = spec in
      let batch_res = Option.map batch ~f:(make_row_spec ~kind:"batch" `Batch) in
      let input_res = Option.map input ~f:(make_row_spec ~kind:"input" `Input) in
      let output_res = make_row_spec ~kind:"output" `Output output in
      make_parsed_labels batch_res input_res output_res }

/* Helper to convert a shape_spec to parsed_axis_labels */
%inline make_labels(spec):
  | spec = shape_spec
    { let batch, input, output = spec in
      make_parsed_labels
        (Option.map batch ~f:(make_row_spec ~kind:"batch" `Batch))
        (Option.map input ~f:(make_row_spec ~kind:"input" `Input))
        (make_row_spec ~kind:"output" `Output output) }

/* List of RHS specs separated by semicolons */
rhs_specs:
  | spec = make_labels(shape_spec) { [spec] }
  | spec = make_labels(shape_spec); SEMICOLON; rest = rhs_specs { spec :: rest }

/* Einsum specification: spec1[; spec2; ...] => result */
einsum_spec:
  /* specs => result */
  | specs = rhs_specs; DOUBLE_ARROW; result = make_labels(shape_spec); EOF
    { (specs, result) }

%%
