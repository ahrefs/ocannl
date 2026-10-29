(** PrintBox helpers shared by tensor printing, the tutorials and the benchmarks: the [dag]
    pre-layout that {!Tensor.print} renders, a wrapper over [printbox-ext-plot] plots, and the
    benchmark comparison table. The module exports what those callers use; the sexp converters
    behind the layout types had no caller and are not derived (gh-ocannl-915). *)

type box = PrintBox.t

type dag =
  [ `Empty
  | `Pad of dag
  | `Frame of dag
  | `Align of [ `Left | `Center | `Right ] * [ `Top | `Center | `Bottom ] * dag
  | `Text of string
  | `Box of box
  | `Vlist of bool * dag list
  | `Hlist of bool * dag list
  | `Table of dag array array
  | `Tree of dag * dag list
  | `Embed_subtree_ID of string
  | `Subtree_with_ID of string * dag ]
(** A tree-shaped layout that {!boxify} flattens level by level before {!dag_to_box} renders it. A
    [`Subtree_with_ID] marks a subtree that an [`Embed_subtree_ID] elsewhere refers back to; the
    identifier is shown only where the reference exists. *)

val boxify : int -> dag -> dag
(** [boxify depth b] rewrites the outermost [depth] levels of [`Tree] nodes into vertical lists of
    the node over a horizontal list of its children. *)

val dag_to_box : dag -> PrintBox.t
(** Renders a [dag] into a box, resolving subtree references by identifier. *)

val reformat_dag : int -> dag -> PrintBox.t
(** [reformat_dag depth b] is [boxify depth b |> dag_to_box]. *)

val plot :
  ?as_canvas:bool ->
  ?x_label:string ->
  ?y_label:string ->
  ?axes:bool ->
  ?size:int * int ->
  ?small:bool ->
  ?legend:(string * string) list ->
  PrintBox_ext_plot.plot_spec list ->
  PrintBox.t
(** A plot box over [printbox-ext-plot]'s defaults. With axes (the default), both axes carry tick
    labels: the two ends of the data window and the multiples of a round step between them, each
    labeled row and column marked on the canvas frame; the y-axis label sits above the plot and the
    x-axis label below it. [~legend] lists [(marker, description)] pairs, rendered as a box below
    the plot, right-aligned. [~as_canvas] drops the labels and axes; [~small] quarters the size in
    both directions. The layout has text, Markdown and HTML renderings (gh-ocannl-103). *)

type table_row_spec =
  | Benchmark of {
      bench_title : string;
      time_in_sec : float;
      mem_in_bytes : int;
      result_label : string;
      result : Base.Sexp.t;
    }

val table : table_row_spec list -> PrintBox.t
(** Groups the rows by [result_label], preserving first-seen label order, and renders each group
    with speedup and memory-gain columns relative to the group's slowest and largest rows. *)
