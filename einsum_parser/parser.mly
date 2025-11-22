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
  let f pos spec = ({ in_axes; pos; from_end }, spec) in
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
    List.length end_axes,
    List.length beg_axes,
    merge_maps from_beg from_end )

let default_row = (None, 0, 0, Map.empty (module AxisKey))

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
%token CARET        /* ^ - reserved for future use */
%token AMPERSAND    /* & - reserved for future use */
%token UNDERSCORE   /* _ */
%token ELLIPSIS     /* ... */
%token DOT_DOT      /* .. */
%token EOF

/* Start symbols */
%start <parsed_axis_labels> axis_labels_spec
%start <parsed_axis_labels * parsed_axis_labels option * parsed_axis_labels> einsum_spec

%%

/* Single axis specification */
axis_spec:
  | i = INT
    { Fixed_index i }
  | id = IDENT
    { Label id }
  | UNDERSCORE
    { Label "_" }
  | conv_expr
    { $1 }

/* Convolution expression: [stride*]output[+[dilation*]kernel] */
conv_expr:
  | stride = INT; STAR; output = IDENT; PLUS; dilation = INT; STAR; kernel = IDENT
    { Conv_spec { stride; output_label = output; dilation; kernel_label = kernel } }
  | stride = INT; STAR; output = IDENT; PLUS; dilation = INT; STAR; UNDERSCORE
    { Conv_spec { stride; output_label = output; dilation; kernel_label = "_offset_only" } }
  | stride = INT; STAR; output = IDENT; PLUS; offset = INT
    { Conv_spec { stride; output_label = output; dilation = offset; kernel_label = "_offset_only" } }
  | output = IDENT; PLUS; dilation = INT; STAR; kernel = IDENT
    { Conv_spec { stride = 1; output_label = output; dilation; kernel_label = kernel } }
  | output = IDENT; PLUS; dilation = INT; STAR; UNDERSCORE
    { Conv_spec { stride = 1; output_label = output; dilation; kernel_label = "_offset_only" } }
  | output = IDENT; PLUS; offset = INT
    { Conv_spec { stride = 1; output_label = output; dilation = offset; kernel_label = "_offset_only" } }
  | stride = INT; STAR; output = IDENT; PLUS; kernel = IDENT
    { Conv_spec { stride; output_label = output; dilation = 1; kernel_label = kernel } }
  | stride = INT; STAR; output = IDENT; PLUS; UNDERSCORE
    { Conv_spec { stride; output_label = output; dilation = 1; kernel_label = "_offset_only" } }
  | stride = INT; STAR; output = IDENT
    { Conv_spec { stride; output_label = output; dilation = 0; kernel_label = "_stride_only" } }

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

/* Einsum specification: spec1[; spec2] => result */
einsum_spec:
  /* spec1; spec2 => result */
  | spec1 = shape_spec; SEMICOLON; spec2 = shape_spec; DOUBLE_ARROW; result = shape_spec; EOF
    { let batch1, input1, output1 = spec1 in
      let batch2, input2, output2 = spec2 in
      let batch_r, input_r, output_r = result in
      let labels1 = make_parsed_labels
        (Option.map batch1 ~f:(make_row_spec ~kind:"batch" `Batch))
        (Option.map input1 ~f:(make_row_spec ~kind:"input" `Input))
        (make_row_spec ~kind:"output" `Output output1)
      in
      let labels2 = make_parsed_labels
        (Option.map batch2 ~f:(make_row_spec ~kind:"batch" `Batch))
        (Option.map input2 ~f:(make_row_spec ~kind:"input" `Input))
        (make_row_spec ~kind:"output" `Output output2)
      in
      let labels_r = make_parsed_labels
        (Option.map batch_r ~f:(make_row_spec ~kind:"batch" `Batch))
        (Option.map input_r ~f:(make_row_spec ~kind:"input" `Input))
        (make_row_spec ~kind:"output" `Output output_r)
      in
      (labels1, Some labels2, labels_r) }
  /* spec1 => result (permute) */
  | spec1 = shape_spec; DOUBLE_ARROW; result = shape_spec; EOF
    { let batch1, input1, output1 = spec1 in
      let batch_r, input_r, output_r = result in
      let labels1 = make_parsed_labels
        (Option.map batch1 ~f:(make_row_spec ~kind:"batch" `Batch))
        (Option.map input1 ~f:(make_row_spec ~kind:"input" `Input))
        (make_row_spec ~kind:"output" `Output output1)
      in
      let labels_r = make_parsed_labels
        (Option.map batch_r ~f:(make_row_spec ~kind:"batch" `Batch))
        (Option.map input_r ~f:(make_row_spec ~kind:"input" `Input))
        (make_row_spec ~kind:"output" `Output output_r)
      in
      (labels1, None, labels_r) }

%%
