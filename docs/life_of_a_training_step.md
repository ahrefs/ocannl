# Life of a Training Step: OCANNL's Compilation Pipeline End to End

This tutorial walks one small program through the whole compilation stack — from a `%op`
expression to machine code executing on a device — naming every stage, the source files and
functions that implement it, and the debug artifact it leaves behind. It is the connective
tissue between the deep-dive documents, each of which covers one segment:

- [Tensors and Contexts](tensors_and_contexts.md) — the user-facing objects (stage 0 and stage 7 here).
- [Syntax extensions](syntax_extensions.md) — what `%op`/`%cd` expand to.
- [Shape inference](shape_inference.md) — the constraint system behind projections (stage 2).
- [Lowering and inlining](lowering_and_inlining.md) — the `Low_level` optimization pipeline (stages 3–4).
- [Schedules and autotuning](schedules_and_autotuning.md) — the loop-transform layer and its search (stage 5).
- [The compilation manifesto](compilation_manifesto.md) — why the pipeline is shaped this way.

Like those documents, this one references code by module, function, and constructor name
rather than line number, so it stays valid as files evolve. Quoted artifacts were captured in
August 2026; regenerate them with the companion program below if they drift.

## The pipeline at a glance

```
      %op / %cd user code                                  tensor/operation.ml, ppx
              │
              ▼
 0.  Tensor.t construction: each tensor carries its        tensor/tensor.ml (Tensor.op)
     forward and backprop code as Assignments.comp
              │
              ▼
 1.  A whole-step comp: Train.grad_update sequences        lib/train.ml
     forward + zero-grads + backprop (+ optimizer)
              │
              ▼          Context.compile ctx comp bindings
              ▼
 2.  Shape inference completes; projections derived        tensor/shape.ml, row.ml
     (forced by lowering through the lazy in Accum_op)
              │
              ▼
 3.  Lowering: Assignments.to_low_level builds loop        arrayjit/lib/assignments.ml
     nests from projections → Low_level.t          [.cd, -unoptimized.ll artifacts]
              │
              ▼
 4.  Backend-independent optimization: analysis,           arrayjit/lib/low_level.ml
     placement, virtualization/inlining, simplify,
     CSE → Low_level.optimized                     [.ll artifact]
              │
              ▼
 5.  Schedules: default parallelization annotations        arrayjit/lib/schedule.ml
     and kernel fission, or explicit/autotuned
     transforms at the lowered_transform seam
              │
              ▼
 6.  Codegen: C_syntax renders each kernel as C /          arrayjit/lib/c_syntax.ml,
     CUDA C++ / HIP / MSL; backend compiles it     cc_backend.ml, cuda_backend.ml,
     (cc → dlopen, nvrtc/hiprtc, MSL at link)      hip_backend.ml, metal_backend.ml
                                                   [.c/.cu/.hip/.metal]
              │
              ▼
 7.  Link and run: buffers allocated into a child          arrayjit/lib/backends.ml,
     context, kernels become a Task.t; Context.run         context.ml, task.ml
     dispatches it on the device's stream
```

Stages 2–7 all happen inside one call to `Context.compile` (+ `Context.run`); stages 0–1 are
pure code construction that happens as your OCaml program builds tensor expressions.

## The companion program

Everything below refers to this program — a dense layer with a relu and a sum-everything
loss, deliberately tiny so every artifact fits on a screen:

```ocaml
(* tutorial_example.ml *)
open Base
open Stdio
open Ocannl
module IDX = Train.IDX
open Ocannl.Operation.DSL_modules

let () =
  let ctx = Context.auto () in
  (* A fixed input: batch of 2, 3 features. *)
  let x =
    TDSL.ndarray
      [| 1.; 2.; 3.; 4.; 5.; 6. |]
      ~label:[ "x" ] ~batch_dims:[ 2 ] ~output_dims:[ 3 ] ()
  in
  (* One dense layer with a relu, and a sum-everything loss. *)
  let%op y = relu (({ w; o = [ 2 ] }) * x + { b }) in
  let%op loss = y ++ "...|... => |->0" in
  (* Build the forward + backprop assignments, then compile them as one routine. *)
  let update = Train.grad_update loss in
  let ctx = Train.init_params ctx IDX.empty loss in
  let ctx, routine = Context.compile ctx update IDX.empty in
  let ctx = Context.run ctx routine in
  printf "loss = %.4f\n" (Context.get_values ctx loss.Tensor.value).(0);
  Train.printf_tree ctx ~with_grad:true loss
```

Build it against the `ocannl` library and run it with debug artifacts enabled:

```bash
OCANNL_BACKEND=cc OCANNL_OUTPUT_DEBUG_FILES_IN_BUILD_DIRECTORY=true ./tutorial_example.exe
```

The run prints `loss = 1.6019` and leaves, under `build_files/tutorial_example/`, four files
per compiled routine — the artifacts quoted throughout this tutorial:

| file | stage | contents |
|---|---|---|
| `<routine>.cd` | 1 | the `Assignments.comp`, printed in `%cd` syntax |
| `<routine>-unoptimized.ll` | 3 | the freshly lowered `Low_level.t` |
| `<routine>.ll` | 4 | the optimized `Low_level.t` |
| `<routine>.c` / `.cu` / `.hip` / `.metal` | 6 | the generated backend source |

Two routines get compiled: `init_params_for_loss` (parameter initialization; see
`Train.init_params` below) and `loss_forward_and_gradient_update` — the training step, our
protagonist. One detail worth savoring before we start: the same program run with
`OCANNL_BACKEND=metal` prints the same `loss = 1.6019`. That is not luck, but it is also not
an unconditional cross-backend invariant: here both backends run the same serial default
schedule, and it is the *schedule* that pins the numerics — a reduction-reassociating
schedule (split reductions, tensorized paths) makes the computed bits a function of the
schedule, so results are bitwise-reproducible as long as the same schedule replays, and
every relaxation beyond that is a named, opt-in policy (see the
[manifesto](compilation_manifesto.md), §4; the `approximate` profile planned for v1.1,
gh-719, will be the packaged one).

## Stage 0: tensors carry their code

*Code: `tensor/tensor.ml` (`Tensor.op`, `binop`/`unop`/`term`), `tensor/operation.ml`.*

OCANNL has no separate "tracing" or "graph capture" step: a `Tensor.t` *is* the graph. Each
tensor records, alongside its `value : Tnode.t` and `shape : Shape.t`:

- `forward : Assignments.comp` — the code that computes this tensor *and all subtensors it
  consumed*;
- `diff : diff option`, where `diff` holds `grad : Tnode.t`, `zero_grads : Assignments.comp`,
  and `backprop : Assignments.comp`.

Every operation — `+`, `*`, `relu`, einsum — funnels into the internal constructor
`Tensor.op`, parameterized by two code-building closures written in `%cd` syntax in
`tensor/operation.ml`. For addition:

```ocaml
let%cd op_asn ~t ~t1 ~t2 ~projections = v =:+ v1 + v2 in
let%cd grad_asn ~t:_ ~g ~t1 ~t2 ~projections = g1 =+ g; g2 =+ g in
...
```

`op_asn` emits the forward assignment, `grad_asn` the gradient accumulations. `Tensor.op`
sequences the forward comps of the operands that are still *forward roots* — a shared
operand already consumed by an earlier-constructed consumer is not re-spliced — followed by
this operation's `op_asn` (an `Assignments.sequence`), and prepends this operation's
`grad_asn` to the operands' backprop comps — backprop code accumulates in reverse construction order, with a topological
re-ordering pass (gh-461) making sure a fragment that accumulates into a gradient runs before
the fragment that embeds that node's backprop code.

Two bookkeeping devices matter downstream:

- **Forward roots.** `session_state.forward_roots` tracks tensors whose forward code has not
  yet been consumed by a consumer expression. When `y`'s construction consumes `x`, `x` stops
  being a root and its code is spliced into `y.forward`. `Tensor.consume_forward_code` /
  `consume_backprop_code` are the explicit consumption points used by `Train`.
- **Embedded nodes.** `Assignments.comp` is not bare code: it pairs `asgns : Assignments.t`
  with `embedded_nodes : Set.M(Tnode).t` — the nodes this comp *owns* (creates and computes).
  A node mentioned by the code but not embedded is expected from the context the comp is
  compiled against — with two exemptions applied at link time: a node with registered
  host-init data self-initializes from `Ir.Host_inits`, and a node an earlier routine in the
  lineage committed `Virtual` is served by splicing its stored computation, no buffer
  needed. This ownership set is what lets the compile step distinguish "allocate this here"
  from "expect this from a parent context" (`from_prior_context` in
  `arrayjit/lib/backends.ml`, with exactly those exemptions in `verify_prior_context`).

The `Assignments.t` type itself (`arrayjit/lib/assignments.ml`) is a short imperative
language of *accumulating assignments*:

```ocaml
type t =
  | Noop
  | Seq of t * t
  | Block_comment of string * t
  | Accum_op of { initialize_neutral : bool; accum : Ops.binop; lhs : Tn.t;
                  rhs : accum_rhs;                    (* Ternop/Binop/Unop/Block/Rev_sides *)
                  projections : Indexing.projections Lazy.t;
                  projections_debug : string }
  | Set_vec_unop of { ... }                           (* vectorized fills, e.g. threefry *)
  | Fetch of { array : Tn.t; fetch_op : fetch_op; dims : int array Lazy.t }
```

An `Accum_op` says: accumulate (`accum` ∈ `Add`, `Max`, ... — `=:+` means "initialize
neutral, then add-accumulate") the right-hand-side expression into `lhs`, iterating over the
loop structure described by `projections` — a *lazy* value, and that laziness is the hinge of
stage 2. `Fetch` initializes a node from a constant, a fill array, a random-seed embed, etc.
`Block_comment` is not cosmetic: `Assignments.get_name_exn` derives the routine's name (and
hence its artifact file names and kernel names) from the outermost block comment.

## Stage 1: one comp for the whole step

*Code: `lib/train.ml` (`grad_update`, `sgd_update`, `init_params`, `to_routine`).*

`Train.grad_update loss` builds the entire training step as a single comp — this is OCANNL's
"schedule the step, not the kernel" decision (manifesto §1). It is itself just a `%cd`
quotation: consume `loss`'s forward code, sequence the gradient zeroing, seed
`loss.grad =: 1`, and splice `loss`'s backprop code, all wrapped in named block comments. An
optimizer step (`Train.sgd_update`, also plain `%cd` code built per parameter) can be
sequenced into the same comp — forward, backward, and update then compile as one program, and
the optimizer's reads of gradients are ordinary dataflow visible to every later optimization.

The `.cd` artifact shows the result for our example — this is the *entire* high-level
program, pretty-printed by `Assignments.to_doc`:

```
loss_forward_and_gradient_update (): # "loss forward and gradient update";
  x =: constant_fill([1., 2., 3., 4., 5., 6.]);
  n22 =:+ w * x ~logic:"@";
  n24 =:+ n22 + b;
  relu_y =:+ relu n24;
  loss =:+ relu_y ~logic:"...|... => |->0";
  # "loss zero grads and backprop";
  b.grad =: 0.;
  w.grad =: 0.;
  n22.grad =: 0.;
  n24.grad =: 0.;
  relu_y.grad =: 0.;
  loss.grad =: 0.;
  _1 =: 1.;
  loss.grad =: _1;
  relu_y.grad =+ loss.grad ~logic:"...|... => |->0 [lhs←rhs1, rhs1←lhs]";
  n24.grad =+ n24 -?/ relu_y.grad ~logic:". [lhs←rhs1, rhs2←lhs]";
  n22.grad =+ n24.grad ~logic:". [lhs←rhs1, rhs1←lhs]";
  b.grad =+ n24.grad ~logic:". [lhs←rhs2, rhs1←lhs]";
  w.grad =+ n22.grad * x ~logic:"@ [lhs←rhs1, rhs1←lhs]";
```

Reading it: `n22`, `n24`, `relu_y` are the anonymous intermediates (`w * x`, `+ b`, the
relu); `~logic` records each assignment's projection spec — `"@"` is matrix multiply,
`"...|... => |->0"` the sum-everything einsum from the source (the bare `|->` closes the
result's batch and input rows, so nothing broadcasts through), `.` pointwise; the bracketed suffixes
on backprop assignments record how the gradient projection was derived from the forward one
(swap lhs with the rhs being differentiated). `-?/` is `relu_gate`. Note what is *absent*:
shapes. Nothing here has committed to dimensions yet.

Parameter initialization is a separate comp compiled the same way: `Train.init_params`
gathers the `forward` code of `loss.params` (the transitively-reachable parameters — a set
each tensor also carries) into an `init_params_for_loss` routine. Its `.cd` artifact is a
nice bonus read: OCANNL's counter-based PRNG means `w`'s "uniform init" is itself ordinary
compiled code (`threefry4x32` applied to `range_over_offsets`, then bit-converted and
affine-mapped), not a host-side RNG call.

`Train`'s convenience wrappers (`to_routine`, `forward_once`, `update_once`, `run_once`) all
reduce to: build a comp as above, then call `Context.compile` — the door to everything below.

## Stage 2: shape inference completes, projections appear

*Code: `tensor/shape.ml` (`propagate_shapes`, `finish_inference`, `derive_projections`),
`tensor/row.ml` (the solver); consumed via `arrayjit/lib/indexing.ml`.*

While tensors were being constructed, each operation eagerly registered its shape constraints
(`Shape.propagate_shapes`, called from `Tensor.make_projections`, which serves both `Tensor.op`
and the `Tensor.raw_*` accumulations) and ran
the cheap first solver stage. But nothing forced final answers: the `projections` field of
every `Accum_op` is a `lazy` closing over that operation's `Shape.update_step`.

The forcing happens the first time *anything* demands a final answer: `Assignments.to_low_level`
forces a `Fetch`'s lazy `dims` or an `Accum_op`'s lazy `projections`, either of which calls
into `Shape.finish_inference` (via `Shape.to_dims` / `Shape.get_projections`) — and *that*
runs the remaining solver stages over all constraints registered so far, resolves every
remaining row and dimension variable (unsolved ones become an error or default per the
solver's rules), then calls `Shape.derive_projections` for each pending update step.
Completion is global and one-shot per batch of registered steps, so in the companion program
it actually happens inside the *earlier* `init_params_for_loss` compile; by the time the
training step lowers, the answers are already there. This is what "shape inference
completion is forced by lowering" means concretely; you can watch it in any backtrace of a
shape error raised from inside `Context.compile` — the frames run from
`Assignments.to_low_level` through `CamlinternalLazy.force` into `Shape.finish_inference`
and `Row.solve_inequalities`.

What pops out, per assignment, is an `Indexing.projections` record
(`arrayjit/lib/indexing.ml`) — the sole interface between shape inference and code
generation. Conceptually it answers: *what is the iteration space of this assignment, and how
does each tensor index into it?*

```ocaml
type projections = {
  components : (int * symbol) list array; (* the iteration space: one entry per product
                                             component, pairing each of its segments'
                                             dimension with its iterator symbol (lists ≠
                                             singleton only for concatenation) *)
  lhs_dims : int array;
  rhs_dims : int array array;
  project_lhs : axis_index array;        (* product-space point → LHS index, per axis *)
  project_rhs : axis_index array array;  (* same per RHS argument *)
  extent_syms : ...; debug_info : ...;
}
```

where `axis_index` is `Fixed_idx of int` (pinned positions, and any non-iterated dim-1
axis), `Iterator of symbol` (the common case; also how a static-symbol batch slice reads),
`Affine of { symbols : (int * symbol) list; offset : int }` (convolutions:
`stride·i + dilation·k − padding`), `Sub_axis` (member of a multi-axis flattened access),
and `Concat of symbol list` (concatenated axes; eliminated during lowering).

For our matmul assignment `n22 =:+ w * x ~logic:"@"`, with `n22 : 2×2`, `w : 2×3`,
`x : 2×3`, the derived record is (symbols renamed for readability):

```
components        = [| [(2, i)]; [(2, j)]; [(3, k)] |]  (* batch row i, out col j, in k *)
project_lhs       = [| Iterator i; Iterator j |]                        (* n22[i, j] *)
project_rhs       = [| [| Iterator j; Iterator k |];                    (* w[j, k]   *)
                       [| Iterator i; Iterator k |] |]                  (* x[i, k]   *)
```

Reductions are implicit: an axis that appears in the product space but not in `project_lhs`
(here `k`) is a reduction axis — that is all a reduction is, which is why
`initialize_neutral` plus the accumulation operator fully determine the loop's semantics.
This representation is also why OCANNL has no layout/reshape subsystem: every access
pattern — transposes, broadcasts, convolutions, slicing — is index arithmetic in
`project_*`, and einsum projections are the sole loop-nest generator.

An aside on broadcasting, because it is a *constraint-generation* choice, not a projections
feature: operations with implicit semantics — pointwise ops, and compose (`*`, the `"@"`
logic) — emit subtyping constraints (`Row_ineq` in `Shape.get_inequalities`: the result's
rows broadcast-cover each operand's), which is what lets a scalar add to a matrix with no
spec written anywhere. Explicit einsum specs instead emit equations (`Row_eq`, via
`einsum_n_constraints`): what the spec *names* unifies exactly, and all flexibility flows
through row variables — written explicitly (`...`, `..v..`), or created implicitly for the
rows a terse spec leaves open (an omitted row kind reads as the context ellipsis shared
between argument and result, and an axis sequence still admits untracked axes to its left),
so `ijk => kji` broadcasts batch axes through without anyone writing `...`. Two consequences
follow: a shared row variable forces the rows it appears in *equal* (not merely compatible —
`ij;jk=>ik` pairs the operands' batch rows), and *closing* a row is what must be said
explicitly (the `|->` in our loss spec — omit it and the batch row flows through, quietly
turning the "scalar" loss into a per-example one). This is a design choice — the specs could generate subtyping
constraints on the spec-to-LHS side — but equations propagate information in both
directions, so relaxing them would make inference both weaker (an einsum would no longer
pin its operands' shapes) and more complex.

Because the whole training step lowers at once, one `finish_inference` serves all
assignments; einsum specs, convolution strides (`Affine` indices with
`stride·i + dilation·j − padding` symbol combinations), and broadcast row variables have all
been settled by the same constraint solve. See [shape_inference.md](shape_inference.md) for
the solver itself and [syntax_extensions.md](syntax_extensions.md) for the einsum notation
that generates the constraints.

## Stage 3: lowering to Low_level

*Code: `arrayjit/lib/assignments.ml` (`lower`, `to_low_level`).*

`Assignments.lower` (called from `Backends.lower_assignments`,
`arrayjit/lib/backends.ml`) translates the comp into `Low_level.t` — a C-like mini-language
of nested `For_loop`s over scalar `Set`/`Get` operations (the full grammar is at the top of
[lowering_and_inlining.md](lowering_and_inlining.md)). The recipe for each `Accum_op`:

1. every segment of every entry of `projections.components` becomes a `For_loop` with a
   **fresh** symbol (product iterators may be shared between operations, so lowering
   α-renames; the substitution also rewrites symbols inside `Affine` indices);
2. the loop body is a `Set` of `lhs` at `project_lhs`, whose right-hand side combines
   `accum` with the `Get`s of the rhs buffers at their `project_rhs` indices — unless
   `initialize_neutral` holds *and* the projection is provably injective
   (`Affine.is_injective`), in which case there is nothing to accumulate with and the
   read-modify-write collapses to a plain store;
3. if `initialize_neutral` is set and the projection isn't surjective-and-injective, a
   `Zero_out` (or neutral-element fill) precedes the nest;
4. concatenation axes (`Concat` indices, from `Block`/`Rev_sides` assignments) are
   eliminated here, by emitting one loop nest per component in sequence.

The `-unoptimized.ll` artifact for our example (excerpt):

```
  x[0, 0] := 1.0;
  ...
  x[1, 2] := 6.0;
  zero_out n22;
  for i31 = 0 to 1 {
    for i32 = 0 to 1 {
      for i33 = 0 to 2 {
        n22[i31, i32] := (n22[i31, i32] + (w[i32, i33] * x[i31, i33]));
      }
    }
  }
  for i34 = 0 to 1 {
    for i35 = 0 to 1 { n24[i34, i35] := (n22[i34, i35] + b[i35]); }
  }
  ...
  zero_out loss;
  for i38 = 0 to 1 {
    for i39 = 0 to 1 { loss[0] := (loss[0] + relu_y[i38, i39]); }
  }
  ...
  for i48 = 0 to 1 {
    for i49 = 0 to 1 {
      for i50 = 0 to 2 {
        w.grad[i49, i50] := (w.grad[i49, i50] + (n22.grad[i48, i49] * x[i48, i50]));
      }
    }
  }
```

Everything is now concrete: shapes are integers, the matmul is three nested loops, every
intermediate is written to what is still nominally its own array. The `Constant_fill` fetch
of `x` unrolled into six scalar stores. This is the maximally naive program — stage 4 exists
to un-naive it.

## Stage 4: backend-independent optimization

*Code: `arrayjit/lib/low_level.ml` (`optimize` = `analyze_proc` + `specialize_proc`); the
per-pass detail is [lowering_and_inlining.md](lowering_and_inlining.md).*

`Low_level.optimize` runs the pipeline summarized as:

```
analyze_proc:      trace_node_facts   (structural facts per tensor node)
                   affine access metrics (exact-or-upper read-multiplicity/coverage queries)
specialize_proc:   decide_placements  (virtual? local? on-device? — per compile lineage)
                   virtual_llc + inline_computation   (inlining, with legality checks)
                   cleanup_virtual_llc  (drop dead materialized writes)
                   simplify_llc       (constant folding, algebra, FMA)
                   rewrite_one_hot_reductions
                   eliminate_common_subexpressions + hoist_cross_statement_cse
```

The analysis/specialization split (gh-555) is what makes search affordable: the expensive
decision-independent analysis is computed once per routine (and memoized in a process-global
cache keyed by a structural digest), while placement decisions replay cheaply per candidate.
The **default state of a computed tensor node is `Virtual`** — no bytes anywhere, consumers
recompute the defining expression inline. More precisely: nodes start *undecided*; a node
the routine only reads resolves `On_device` (it must be an input buffer), explicit intent
(parameters, `Train.set_materialized` — which `Train.grad_update` applies to the loss's
value, which is why our loss has storage) wins outright, and it is
the written-and-still-undecided nodes that default to `Virtual` at cleanup. Materialization
of those must be earned (visit-count and recompute-cost caps) or forced by user intent. Observability —
a declared intent to read the node's values, `Tnode.set_observable` — deliberately does
*not* force materialization: a virtual node stays observable by recomputation. It only
steers placement away from `Local`, the sole unobservable class (routine-scoped scratch
whose computation is not tracked): an observable node resolves `On_device` where an
unobservable one would default to `Local`, and an observable node still *undecided* at a
forcing point materializes only because nothing was stored to recompute it from. What
actually breaks observation is a recomputation that transitively depends on a `Local` node
with no materialized node shielding the dependency. (Implementation status: the placement
guards are in; recompute-on-read exists today as the best-effort `Train.ensure_printable`
path — a `for_print` copy compiled and registered as a read proxy, used by `Train.printf` /
`printf_tree` — while bare `Context.get_values` raises for a node with no buffer, host-init
data, or proxy; the full contract is the manifesto's stated goal state.) Decisions land in
`Placements` on the `optimize_ctx`, which is *forked per compile* (`copy_optimize_ctx` in
`Backends.lower_assignments`): the same tensor can be virtual in one routine and on-device in
another.

The optimized `.ll` for our example repays close reading:

```
  zero_out loss;
  for i38 = 0 to 1 {
    for i39 = 0 to 1 {
      loss[0] := (loss[0] + relu((v27_n22 {
        v27_n22 := 0.0;
        for i55 = 0 to 2 {
          v27_n22 := fma(w[i39, i55], x[i38, i55], v27_n22);
        }
      } + b[i39])));
    }
  }
  /* loss zero grads and backprop */
  zero_out b.grad;
  zero_out w.grad;
  zero_out n22.grad;
  for i44 = 0 to 1 {
    for i45 = 0 to 1 {
      n22.grad[i44, i45] := (n22.grad[i44, i45] + relu_gate((v27_n22 { ... }
        + b[i45]), 1.0));
    }
  }
  ...
```

What happened, pass by pass:

- **`n22`, `n24`, `relu_y` are gone as arrays.** All three stayed `Virtual`; the matmul
  re-appears as a `Local_scope` (`v27_n22 { ... }`) *inline at each consumption site* — the
  forward-pass loop nest and both backprop nests each recompute the row-dot-product they
  need. Recompute-over-materialize is the default trade; the caps in `virtualize_settings`
  and the transitive fan-in guard bound how far it goes.
- **Their gradients mostly vanished too.** `relu_y.grad` and `loss.grad` were inlined down to
  the constant `1.0` inside `relu_gate(..., 1.0)` — constant propagation through the
  virtualized seed (`loss.grad =: _1; _1 =: 1.`). Only `n22.grad` survives as storage. Why it
  and nothing else: the visit cap (`virtualize_max_visits`, default 1) counts *per-cell read
  multiplicity*, computed from the affine access relations as an exact-or-upper bound
  (where disjointness cannot be proven, sites are conservatively summed — erring toward
  materialization), and a read at its own
  statement's write position (accumulation-style read-modify-write) is exempt. Every other
  intermediate is read once per cell, or only at its own write position, or is a scalar
  constant expression (the seed `_1`, `loss.grad`), which inlines regardless; but the `w.grad`
  nest reads each `n22.grad[i, j]` cell once per element of the `k` loop — multiplicity 3 —
  so recomputing it inline would triple its cost, and the policy materializes it
  (provenance `Never_virtual 1`).
- **`x`'s six stores vanished** — but differently: `x` is materialized (it's read
  everywhere), and a materialized node whose only writes are literal constants registered
  with `Host_inits` has its in-kernel initialization *moved to link time*
  (`hosted_constant_inits_to_link_time`, gh-633): the values upload at the first allocation
  on a device — read-only constants dedup through the per-device `constant_buffer_cache`,
  so later contexts on that device reuse the buffer — instead of re-executing on every step.
- **FMA formation** (`simplify_llc`): the multiply-accumulate became `fma(...)`.
- **Zeroing survives only where storage survives**: of the six zero-grad statements in the
  `.cd`, three remain (`b.grad`, `w.grad`, `n22.grad`) and three died with their arrays
  (`n24.grad`, `relu_y.grad`, `loss.grad`). The surviving `zero_out loss` is not one of
  them — it is the forward accumulation's neutral init.

The result record, `Low_level.optimized`, carries more than the code `llc`: the
`traced_store` of per-node facts (which stage 6 uses to derive kernel parameters), the
`optimize_ctx` to thread into the produced context, and schedule-related sets
(`workgroup_shared`, `zero_fringe`, ...) that are empty until stage 5 populates them. It also
reports `flip_candidates` — the inlining decisions a search may flip, which is how
placement became a searchable schedule dimension rather than a fixed heuristic (gh-555).

## Stage 5: schedules — parallelism as a separate concern

*Code: `arrayjit/lib/schedule.ml`; the full story is
[schedules_and_autotuning.md](schedules_and_autotuning.md).*

Everything so far decided *what* to compute and *where values live*; nothing yet decided how
loops map onto hardware. That is the schedule layer, a
`Low_level.optimized -> Low_level.optimized` transformation applied at the
`?lowered_transform` seam of backend `compile` (reachable from user code via
`Context.compile ~lowered_transform`). The separation is not absolute: virtualization
decisions are never re-run, but schedule transforms do adjust memory in schedule-owned
ways — staging and split reductions mint fresh scratch/partials nodes (populating
`workgroup_shared`, `simdgroup_fragments`), and fission promotes a `Local` node whose live
range crosses a kernel cut to `On_device` (`Placements.promote_local_to_device`, the one
sanctioned placement override). When no explicit transform is given —
our example's case — `Schedule.maybe_default_schedules` applies the default annotators plus
kernel fission:

- **Default GPU/CPU presets**: for each loop nest whose parallelism is provable from the code
  (every materialized write covered by the loop index; conservative race analysis, backed by
  the `Ir.Affine` relational queries), retype loops to hardware axes — `Grid` and
  `Workgroup` on GPU, an outer `Grid` loop rendered onto a thread pool on CPU. Reduction
  loops stay serial: under the deterministic regime — the default, and today the only one —
  atomics are forbidden, so an ordinary loop reduction parallelizes only via the explicit
  `Split_reduce` two-pass transform (a fixed combine tree, bitwise-reproducible per
  schedule); the other atomics-free mechanism is a tensorized contraction, where an
  accepted `Tile_mma` intrinsic rendering executes the tile's reduction cooperatively. This rule is
  profile-conditional by design: the `approximate` profile planned for v1.1 (gh-719) will
  admit numerics-relaxing parallelizations as a named policy beside the deterministic
  default.
- **Kernel fission** (`Schedule.fission_scheduled`): the whole-step program is cut into
  segments at materialized producer→consumer edges that cannot share a grid launch; each
  segment compiles as its own kernel, chained by device events. This is the fission-not-fusion
  inversion (manifesto §1): kernel boundaries are *discovered* in the fused program, not
  assembled from an operator graph.

In our toy example every extent is below `gpu_schedule_min_parallel`/
`cpu_schedule_min_parallel`, so the nests stay serial and no fission cut is needed — the
whole step ships as one kernel even on Metal. On real models this stage is where the
matmul-tiling, staging, tensorization, and split-reduction transforms apply, either as
autotuner-searched compositions (`Autotune.tune`, with winners persisted in a schedule cache
keyed by a structural digest of the optimized code, and `Train.tune_placements`' placement
decisions persisted beside them, keyed by the raw program's — gh-786) or — for `Train`'s convenience wrappers,
when `model_default_schedule=true` (off by default; plain `Context.compile` never does
this) — as a model-ranked pick among the default-schedule flavors with zero timing runs.
Transform *application* is strict: an invalid composition (a missing loop symbol, a
non-positive split factor, an illegal interchange) fails loudly at apply time rather than
falling back. Rendering an applied schedule then never produces *wrong* code, and for the
constructs that carry fallbacks — `Tile_mma`'s decline ladder down to the scalar arm,
hardware-axis loops serializing where the backend has no binding — it degrades gracefully;
a few schedule-minted constructs without a fallback (e.g. a pipelined tile read outside its
rotor loop) instead decline with a classified `Backend_codegen` error, which the autotuner
treats as a candidate decline. Performance, never correctness, is what's being searched.

## Stage 6: code generation

*Code: `arrayjit/lib/c_syntax.ml` (the shared functor), `cc_backend.ml`,
`cuda_backend.ml`, `hip_backend.ml`, `metal_backend.ml`, `builtins.c`/`builtins_*.ml`.*

All current backends emit C-family text through the `C_syntax` functor, which turns
`Low_level.t` into a PPrint document given a small per-backend vocabulary (type names,
builtin spellings, how to render precision conversions, how a `Grid` loop binds to the
hardware index, ...). The cc backend takes the defaults nearly verbatim; CUDA/HIP override
what NVIDIA/AMD dialects need (e.g. `__float2half`, wmma intrinsics, `cp.async` staging);
Metal overrides for MSL (`simdgroup` matrices, threadgroup memory).

The functor's entry point is `C_syntax.compile_proc`, called from each backend's
`compile`/`compile_batch`. It derives the kernel's parameter list, then renders the body
(`pp_ll`, a structural recursion over `Low_level.t`):

- `For_loop { axis = Serial }` → a plain `for`; `Grid`/`Workgroup` axes → the backend's
  hardware index binding (`blockIdx`/`threadIdx` on CUDA, `gid`/`lid` on Metal) — and on
  cc, which has no hardware indices, an eligible `Grid` loop renders as a chunked
  `dispatch_apply`/OpenMP thread-pool loop and everything else stays serial;
  `Vectorized`/`Unrolled` axes try SIMD/unrolled renderings with serial fallbacks.
- `Set`/`Get` → array stores/loads with Horner-form row-major offsets; `Local_scope` → a
  scoped local variable; `Zero_out` → a zeroing nest (or elided into the local array's
  `= {0}` initializer); `Tile_mma` → the decline ladder from intrinsics (wmma /
  `simdgroup_matrix` / rocWMMA) through register-tiled C down to the scalar fallback, with
  every outcome recorded in the `mma_census`.

The generated `.c` for our example (excerpt):

```c
void loss_forward_and_gradient_update(
    float *restrict b,
    float *restrict b_grad,
    float *restrict loss,
    float *restrict w,
    float *restrict w_grad,
    float *restrict x) {

  /* Local declarations and initialization. */
  float n22_grad[4] __attribute__((aligned(32))) = {0};

  loss[0] = (float)(0.0);
  {
    float v42_loss;
    v42_loss = loss[0];
    for (int32_t i38 = 0; i38 <= 1; ++i38) {
      for (int32_t i39 = 0; i39 <= 1; ++i39) {
        {
          float v27_n22;
          v27_n22 = (float)(0.0);
          for (int32_t i55 = 0; i55 <= 2; ++i55) {
            v27_n22 = fmaf(w[(i39) * 3 + i55], x[(i38) * 3 + i55], v27_n22);
          }
          v42_loss = (v42_loss + fmaxf(0.0, (v27_n22 + b[i39])));
        }
      }
    }
    loss[0] = v42_loss;
  }
  ...
}
```

Observations that generalize:

- **The buffer parameters are exactly the materialized context nodes** (derived from the
  `traced_store`): `b`, `w`, `x`, `loss`, and the two surviving gradients. (The full ABI can
  hold more than buffers: one `int` parameter per static-index binding, a merge-buffer
  pointer when the routine reads one, and a log parameter under routine logging — our
  example has none of these.) Virtual nodes
  never appear; `n22_grad` resolved to `Local` — routine-scoped scratch, here a stack array
  inside the function (a node over `stack_threshold_in_bytes` resolves `On_device`
  instead). Kernels
  are context-independent: the same compiled function can link against different contexts'
  buffers.
- **`Local_scope`s became scoped local variables** (`v27_n22`, and `v42_loss` — the latter a
  read-modify-write contraction of the `loss` accumulation, hoisted around both loops).
- **Multi-dimensional indices flattened to row-major arithmetic** (`(i39) * 3 + i55`).
- On Metal the same body appears inside a `kernel void ... [[buffer(n)]]` function whose
  parameters are *memory pools plus a slot table* rather than one buffer per node — Metal's
  argument-binding limit forces the pooled ABI (`ptr_param_style = Pooled`). CUDA and HIP
  also pool device allocations internally, but keep the per-node-pointer ABI: their linkers
  resolve each node's pool base + offset to an individual kernel argument before launch.
- Numeric builtins that C lacks (`relu_gate` variants, precision conversions, the
  `threefry4x32` PRNG, bf16/fp8 arithmetic) come from the per-backend builtins tables —
  `builtins_cc.ml`, `builtins_cuda.ml`, `builtins_hip.ml`, `builtins_metal.ml` — whose
  definitions are token-matched against the kernel and prepended to its source. `builtins.c`
  is the *host-side* twin, compiled into the OCaml binary for `Ops`/`Ndarray`, kept
  numerically in sync with the kernel-side tables.

Then each backend turns text into an executable object — how much of that happens in
`compile` vs. `link` differs: `cc_backend` does it all at compile time — writes the `.c`
file, invokes the configured C compiler into a shared object, and `dlopen`s it (CPU code
can load eagerly; there is one program memory); `cuda_backend` compiles through NVRTC to
PTX at compile time but defers module loading to link time (`cuModuleLoadDataEx` loads into
a device-specific context), and `hip_backend` mirrors this through HIPRTC (`.hip` source to
a code object); `metal_backend` defers even source compilation to link time,
compiling the MSL per device (`Me.Library.on_device`) and validating the launch against the
device's limits there. This is why the backend interface distinguishes `compile` (produce
the artifact) from `link` (bind it to a device context) — and batches both
(`compile_batch`/`link_batch`) for the segment kernels of one fissioned routine, so the
segments share one compiler invocation and one loaded module.

## Stage 7: link, contexts, and running

*Code: `arrayjit/lib/backends.ml` (`Raise_backend.link`, `allocate_delta`),
`arrayjit/lib/context.ml`, `arrayjit/lib/task.ml`; user-facing story in
[tensors_and_contexts.md](tensors_and_contexts.md).*

`Context.compile` returns a `routine` — and behind that word sits the last stretch of the
pipeline, `Raise_backend.link`:

1. **Verify** that every node the code expects from a prior context is actually there
   (`from_prior_context` vs. the context's buffers), and that a merge-buffer consumer is
   linked against the context of the transfer that produced it (static check, gh-288).
2. **Allocate the buffer delta**: materialized nodes not yet present in the parent context
   get device memory now — packed into pools (with liveness-based aliasing of disjoint-lived
   buffers when `buffer_aliasing` is on), constants deduplicated through a per-device cache.
   Nodes with registered host-init data (our `x`) are uploaded here — the link-time half of
   the gh-633 constant-init move from stage 4. (`w` and `b` need no upload: the
   `init_params_for_loss` routine already computed them on-device, in this same lineage.)
3. **Bind the kernel** to the buffers and to the static-index bindings, producing a
   `Task.t` — a plain thunk wrapper. Fissioned segments become one task chained by device
   events; per backend, `sequence_segments` can do better — CUDA *and* HIP capture the
   segment launches into a graph replayed as one unit (when `gpu_graph_capture=true` and
   routine logging is off; ordinary chained launches otherwise), Metal encodes them into a
   single command buffer.
4. **Make a child context**: contexts form a lineage; the routine's context extends its
   parent with the new buffers and the forked `optimize_ctx` (so a node this compile decided
   to keep virtual stays virtual for every later compile in this lineage, and its stored
   computation can be inlined into later routines).

Back in `context.ml`, the routine is registered in the lineage's execution ledger, and its
`execution_deps` are derived from read/write hazards against previously compiled routines.
`Context.run` then validates — execution dependencies satisfied, input buffers present and
framework-initialized (a buffer-availability check, not a data check: an input you forgot
to fill can still run with the allocator's zeros), bindings in range —
and dispatches the task onto the device's stream — a FIFO queue with events; on the default
`cc` backend the "stream" degenerates to synchronous execution, on `multidev_cc` it is a
worker domain, on GPUs a real device stream. Runtime-varying integers enter through the
`bindings` (`Train.IDX.get_static_symbol`): each static symbol lowers to an `int` kernel
parameter, so re-running with a new value is just writing an `int ref`, no recompilation.
The symbol's role varies: a minibatch position (the `@|` slice) is an index selector — it
picks which batch row the projections read; an embedded step counter is just a scalar value;
and only a symbol used as a *symbolic extent* (`used_as_extent`, gh-490) changes an
iteration range — rendered as a maximum-extent loop whose body is guarded by
`i < the bound value`.

Reading results back is explicit and context-mediated: `Context.get_values ctx tn` performs
an on-demand device-to-host transfer into a fresh temporary host buffer. What gh-333
removed is the *persistent* host mirror tensors used to carry — nothing keeps host copies in
sync anymore, and a literal's registered `Host_inits` data can serve a read without touching
the device at all. Which is where our program prints its `loss = 1.6019` — the end of the
line.

## Recap: where to look when

| you want to... | look at |
|---|---|
| see the high-level code of a routine | `.cd` artifact; `Assignments.to_doc`; `Train.to_routine ~output_cd_file:true` |
| see loop nests before/after optimization | `-unoptimized.ll` / `.ll` artifacts (`Low_level.to_doc`) |
| understand why a node materialized | `Tnode.debug_memory_mode`, provenance codes in [lowering_and_inlining.md](lowering_and_inlining.md) |
| see the kernel source | `.c`/`.cu`/`.hip`/`.metal` artifacts (config `output_debug_files_in_build_directory=true`) |
| see what a kernel computed at runtime | config `debug_log_from_routines=true` (see the debug-tracing skill / docs) |
| trace scheduling decisions | `schedule_log_declines`, `schedule_log_launches`; `routine.mma` |
| understand compile failures by phase | `Schedule_outcome` classification (`Transform` / `Hardware_limits` / `Backend_codegen` / `Backend_compile` / `Backend_link`, plus `Launch` / `Sync` at run time) |

And the one-sentence summary of the whole pipeline: **tensors carry `%cd` code; `Train`
sequences a whole step of it; lowering forces shape inference and mints loop nests from
projections; `Low_level.optimize` decides what deserves memory and inlines the rest;
schedules retype loops onto hardware and cut kernels at materialization edges; `C_syntax`
renders each kernel to a C dialect the backend compiles; and linking binds kernels to
per-context device buffers that `Context.run` dispatches on a stream.**
