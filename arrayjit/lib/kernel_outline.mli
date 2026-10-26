(** Outlining a large kernel (gh-ocannl-1603).

    A C compiler's cost grows faster than linearly in a function's size: the serial baseline of a
    [gpt2_mini_train] search is one function of every statement of the routine, and clang -O3 took
    127 s over it, against 30 s for the same statements as 62 [noinline] functions. A host C backend
    ({!C_syntax.C_syntax_config.outline_part_weight}) therefore renders a large kernel OUTLINED: its
    top-level statements packed in order into parts, each a function the kernel calls in turn, in
    program order -- the same statements, so the same results. *)

val groups : part:int -> Low_level.t list -> Low_level.t list list option
(** [groups ~part stmts]: a kernel's top-level statements ([Low_level.flat_lines]) packed in order
    into parts of weight at least [part] each -- a statement weighs its statement and expression
    nodes, about a line of C each -- or [None] when the kernel stays whole: it weighs no more than
    sixteen parts (a few parts' worth compiles quickly whole, and a fissioned segment, which a
    candidate compiles by the hundred, stays one function), it would make fewer than two parts, or a
    top-level statement's text is opaque to the renderer (a staged callback) or synchronizes (a
    barrier). A part never ends where something spans the boundary: a scope local named on both
    sides of it, or a zero initializer, which the renderer may forward into the statement after it.
*)

val part_name : kernel:string -> int -> string
(** The function name of the kernel's part [k]: spelled after the kernel's own, so unique within a
    library. *)

val part_attribute : string
(** The macro every part's declaration and definition carries: never inlined back into its kernel --
    which would rebuild the one large function outlining avoids -- and visible to the other
    translation units of its library only. *)

val part_attribute_definition : string
(** {!part_attribute}'s definition, guarded so that every translation unit can carry it; PE/COFF has
    no hidden visibility. *)

val trailing_ident : string -> string
(** The identifier a parameter declaration ([uint32_t *restrict n826]) declares: its last token. *)
