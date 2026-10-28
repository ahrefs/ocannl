(** gh-ocannl-1526: one lowering of a routine, reused by the many compiles of an autotuning search.

    Every candidate of a search compiles the same [comp] with the same bindings from the same search
    context, and builds its code from the one base lowering the search captured; the lowering each
    compile used to redo contributed nothing but a fresh fork of the lineage state, at about 1.7 s a
    candidate on HIP once the kernel cache made device compilation cheap. A cell is created per
    search. The first compile handed the cell lowers as usual and leaves a hermetic snapshot of what
    [lower_assignments] returned (before zero sinking, so every later stage runs as on a fresh
    lowering); each later compile gets a private deep copy of that snapshot instead of lowering
    again.

    The contract. Under unchanged lowering inputs, a reused lowering equals the fresh lowering the
    compile would have made, in every field the compile consumes;
    [test/operations/ autotune_lowering_reuse] checks that per candidate through {!with_audit}.
    Lowering inputs can change between the snapshot and a candidate in one way: a link proposes node
    value bounds, and a later fresh lowering folds guards the base kept (the drift of PR #140 /
    gh-ocannl-1421). The candidates' code is the base's either way; the reused state is the one that
    code was decided under, where the fresh one was decided under other code. Bounds are therefore
    not an invalidation input, by design.

    The cell's owner -- the source lineage table, the assignments, the bindings, the name -- is
    compared by physical identity. A compile handing the cell a different owner raises
    [Invalid_argument]: the cell is scoped to its search, and the caller routes a compile into
    another lineage around it. The owner check does not detect a lineage table mutated IN PLACE;
    within a search nothing does that (every compile forks it, see {!Low_level.copy_optimize_ctx}).
*)

type t
(** A search's lowering cell: empty, or a snapshot under its owner. *)

val create : unit -> t
(** An empty cell, for one search. *)

val lower_or_reuse :
  t ->
  lineage:Low_level.optimize_ctx ->
  asgns:Assignments.t ->
  bindings:Indexing.unit_bindings ->
  name:string option ->
  lower:(unit -> string * Low_level.optimized) ->
  string * Low_level.optimized
(** What [Backend.compile] calls in place of [lower]: on an empty cell, [lower ()], whose result is
    also kept (a private copy) under the owner [(lineage, asgns, bindings, name)]; on a filled cell,
    a private copy of the snapshot, the owner checked by physical identity ([Invalid_argument] on a
    mismatch). The resolved routine name travels with the snapshot. *)

val reseed : t -> Low_level.optimized -> unit
(** Replaces a filled cell's snapshot with (a private copy of) the given lowering, under the same
    owner: the search swapped its base for another lowering of the same routine from the same
    context (the bounds-blind base of gh-ocannl-1614), and its candidates' lineage state must come
    from the code they are rebased on. The lowering is taken as [lower_assignments] returns it,
    before zero sinking. [Invalid_argument] on an empty cell. *)

val private_copy : Low_level.optimized -> Low_level.optimized
(** A private copy of a lowering: the traced store's table and records
    ({!Low_level.copy_traced_store}), and the lineage tables ({!Low_level.copy_optimize_ctx}). Every
    other field of {!Low_level.optimized} is immutable. *)

val snapshot : t -> Low_level.optimized option
(** The snapshot itself, for tests checking that candidates left it untouched. Read it, never mutate
    it. *)

val stats : unit -> int * int
(** [(reused, lowered)]: cumulative over the process, how many compiles given a cell took a copy of
    its snapshot, and how many lowered (filling an empty cell). For tests and diagnostics. *)

(** {2 Test seams}

    Domain-local: each holds only while its body runs in the calling domain. *)

val with_audit :
  (reused:Low_level.optimized -> fresh:Low_level.optimized -> unit) -> (unit -> 'a) -> 'a
(** Every reuse ALSO lowers fresh, and the check gets a copy of the reused lowering and the fresh
    one; the compile proceeds with the reused one. *)

val with_created_observer : (t -> unit) -> (unit -> 'a) -> 'a
(** The observer gets every cell {!create} makes (one per search), so a test can read its
    {!snapshot} afterwards. *)

val with_disabled : (unit -> 'a) -> 'a
(** Cells are ignored and every compile lowers, as before gh-ocannl-1526: the reference the reuse
    path is checked against. *)
