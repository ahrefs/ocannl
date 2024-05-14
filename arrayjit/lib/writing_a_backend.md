# The Anatomy of an OCANNL Backend

Currently, OCANNL integrates new backends via code in [backends.ml](backends.ml). `Backends` introduces the context-specific `routine` type, and has a helper `lower_assignments` that wraps `Assignments.lower` and `Low_level.optimize_proc`. The interface to a `Backend` has `compile` functions, to allow some backends handle assignments directly, instead of using the optimized C-like representation `Low_level.t`.

```ocaml
type 'context routine = {
  context : 'context;
  schedule : unit -> Tnode.work;
  bindings : Indexing.lowered_bindings;
  name : string;
}
```

Backends need to provide the `code` (for compilation result) and `context` types. `code` is some intermediate state between assignments and `context routine`. A backend my postpone doing anything specific until linking, e.g. `code = Low_level.optimized`, or linking may be a no-op, effectively `code = routine`, usually it will fall somewhere in between and depend on whether `~shared:true` is passed.

```ocaml
module type No_device_backend = sig
  type code
  type context
  type nonrec routine = context routine
  ...
end
```

