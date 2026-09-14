# What remains for OCANNL

*A guide to the open design work, September 14, 2026.*

OCANNL already has the ingredients of a substantial deep-learning compiler: a tensor language,
shape inference, automatic differentiation, several CPU and GPU backends, and a search system
that can choose how a computation runs. The interesting question now is what kind of system
those ingredients can become together.

The remaining work has several different ambitions. Some of it makes the implementation more
elegant. Some tries to turn expressive compiler machinery into measured speed. Some makes
experiments easier to conduct and understand. The largest architectural proposals ask whether
tensor programs can become reusable units of inference, and whether the compiler underneath
them can stand on its own.

This guide explains those ambitions and the gaps between them and the current implementation.
It is an OCANNL-focused companion to the operational sequencing plan, incorporating the
September 14 milestone reshuffle. It leaves machine assignments, daily waves, review histories
and other repositories' backlogs in that plan. [ROADMAP.md](../../ROADMAP.md) owns the release
calendar; [CHANGES.md](../../CHANGES.md) records what has landed. Issue links below point to
`ahrefs/ocannl`, while implementation PRs live in `lukstafi/ocannl-staging`. The descriptions and
issue directory are a dated snapshot, not a second live tracker.

The six themes are:

- [Compiler elegance](#1-give-shared-ideas-one-expression)
- [Measured performance](#2-turn-compiler-expressiveness-into-useful-speed)
- [Memory and reuse](#3-compute-and-store-the-part-that-is-actually-needed)
- [Maintainable evidence](#4-make-evidence-easier-to-maintain)
- [Complete experiments](#5-turn-capabilities-into-complete-experiments)
- [Reusable programs and standalone ArrayJIT](#6-make-programs-reusable-and-the-compiler-independently-useful)

## A short tour of the system we are extending

An OCANNL computation begins with tensors and relationships between their axes. Generalized
einsum notation describes more than matrix multiplication: it also expresses reductions,
broadcasting, permutations and convolution-like indexing. Shape inference works out which
axes and extents can satisfy those relationships. Automatic differentiation constructs the
corresponding backward computation.

Below that language, the compiler works with explicit loops, reads, writes and reductions.
A *schedule* describes choices such as tiling, staging data in faster memory, parallelizing
loops, or using matrix instructions. *Fission* introduces kernel boundaries where needed.
*Autotuning* compares candidate implementations by running them. Placement decides when a
value needs storage and when its defining computation can be inserted at a read site.

These layers explain why apparently unrelated issues belong together. An attention rewrite
needs a way to express evolving state inside a loop. A new loop construct needs analyses that
understand its implicit reads and writes. A fast candidate needs both a reliable timing and a
clear owner for its allocated resources. A reusable tensor function needs a shape solver whose
state can be separated from one particular application.

For the detailed architecture, see [the life of a training step](../life_of_a_training_step.md)
and [the compilation manifesto](../compilation_manifesto.md). Here the focus is on what remains.

## 1. Give shared ideas one expression

The compiler-elegance work retained in **v1.0.2** is worth doing for its own sake. If two
backends implement the same rule, there is value in writing that rule once and making their
differences explicit. Easier feature development may follow, but a predicted speedup or a
proven dependency on the next feature is not the admission test for this work.

The recent renderer refactor made each render own its traversal state. The next step is to
make the shared structure of backend compilation equally visible. CUDA and HIP still repeat
much of their syntax configuration; several backends repeat the outline of compilation and
conditional header inclusion. The goal is a common description with explicit backend
variation, supported by checks that compile the real vendor implementations
([#770](https://github.com/ahrefs/ocannl/issues/770), [#794](https://github.com/ahrefs/ocannl/issues/794)). This is a structural change, not a claim that CUDA and HIP are interchangeable.

A deeper form of duplication appears in IR traversal. Several analyses independently answer
questions such as “which operands are actually evaluated?” and “does this loop run at all?”
An expression may contain an operand that its operator discards. A loop may have no iterations.
A recurrence also performs initialization and state rotation that are not ordinary statements
in its written body. A shared traversal should express these semantics once while letting
analyses choose the information they collect ([#630](https://github.com/ahrefs/ocannl/issues/630)). The challenge is to share meaning,
not merely shorten pattern matches.

The same idea appears at smaller scales. The host stubs and generated CPU kernels have copies
of common C builtins ([#656](https://github.com/ahrefs/ocannl/issues/656)). Precision enumeration should belong to the operations layer
rather than be reconstructed by renderers and tests ([#917](https://github.com/ahrefs/ocannl/issues/917)). Affine-index normalization
should establish its invariant at construction ([#774](https://github.com/ahrefs/ocannl/issues/774)). Configuration precedence should use
one resolver, with bootstrap restrictions represented explicitly ([#604](https://github.com/ahrefs/ocannl/issues/604)). The renderer and
its shuffle simulator should consume one description of the reduction stages ([#875](https://github.com/ahrefs/ocannl/issues/875)).

Together with vendor-arm compilation, these are the eight retained issues. The two items
shared with tests, precision enumeration and shuffle stages, concern an implementation concept
that tests also consume. General testing-infrastructure refactorings move to v1.1.1. That
boundary preserves an opportunity to improve the compiler's form without making the whole
maintenance backlog a prerequisite to performance work.

## 2. Turn compiler expressiveness into useful speed

The **v1.1** performance milestone asks a demanding question: which ideas improve an actual
computation, and under which numerical and hardware conditions? Adding a candidate to the
search space is only a beginning. It must render the intended implementation, survive numerical
comparison, and win against an appropriate alternative. A well-measured null result is also
useful: it tells us where another mechanism or a different workload is needed.

### Attention needs a different computation

Consider attention over a sequence of length *n*. The straightforward composition builds an
*n × n* score matrix, normalizes its rows, and combines the normalized scores with the value
vectors. That intermediate grows quadratically. Making its loops faster does not remove the
cost of constructing and moving it.

Online softmax offers a different route. As blocks of scores arrive, a running maximum and
normalization sum evolve, and previously accumulated contributions are rescaled. The compiler
must represent a recurrence whose new state depends on several pieces of old state. That is
why the recently landed `Low_level.Scan_loop` matters: it provides the recurrence vocabulary.
The attention-pattern recognition and substitution are still to be written ([#483](https://github.com/ahrefs/ocannl/issues/483)).

The intended result is an attention implementation that avoids materializing the full score
intermediate, with numerical agreement inside a stated tolerance and measurements across
sequence lengths. “The IR can express the recurrence” and “the compiler rewrites attention
into it” are different achievements. The former has landed; the latter has not.

Cumulative operations and top-k raise a related but separate question: how should a person
express recurrences through the tensor API? The current proposal favors dedicated primitives
before introducing a general scan into the assignments language ([#952](https://github.com/ahrefs/ocannl/issues/952)). Those operations
can eventually support recurrent models, but that API decision does not block a rewrite that
works directly on lowered attention code.

Winograd convolution is the other major algebraic proposal ([#505](https://github.com/ahrefs/ocannl/issues/505)). It transforms eligible
3×3 convolutions into a different arrangement of arithmetic. Fewer multiplications alone do
not establish a win: transforms, memory traffic, shapes and hardware all matter. Its meaningful
comparison is the existing implicit-GEMM implementation, not an artificially weak baseline.
Like online softmax, it changes rounding behavior and needs a numerical policy.

### Numerical policy makes comparisons intelligible

The `approximate` profile and benchmark regime column have landed. The fleet acceptance work
has not finished ([#719](https://github.com/ahrefs/ocannl/issues/719)). The sequencing plan records tuned CUDA attempts that exceeded the
cell budget and a separate approximate-CPU library-load failure. Neither observation is a
successful performance result, and neither means the profile itself is still unimplemented.

The important distinction is between a configuration being available and a claim about it being
supported by measurements. The benchmark expansion ([#720](https://github.com/ahrefs/ocannl/issues/720)) gives those claims a setting:
sequence and batch scaling, convolution workloads, memory use, and comparisons against other
frameworks under declared regimes. New algebraic rewrites should join that same numerical
contract rather than acquire an unrelated definition of “close enough.”

Matrix instructions add another distinction: input storage, accumulation and output storage
can have different precisions. Some useful combinations remain unsupported by particular
renderings—Metal half inputs with float output storage, CUDA half destinations after persistent
wide accumulation, and HIP wide accumulation for bf16 inputs ([#923](https://github.com/ahrefs/ocannl/issues/923), [#925](https://github.com/ahrefs/ocannl/issues/925), [#838](https://github.com/ahrefs/ocannl/issues/838)). These
are specific gaps, not a single switch called “mixed precision.”

### Geometry, search and measurement must agree

Register-tile geometry is now a schedule decision the tuner can compare. The remaining work
is to explain and improve how it behaves: model the cost of scalar remainders, inspect spills
for the geometries actually proposed, and support useful schedules when extents do not divide
the blocking exactly ([#947](https://github.com/ahrefs/ocannl/issues/947), [#948](https://github.com/ahrefs/ocannl/issues/948), [#620](https://github.com/ahrefs/ocannl/issues/620), [#627](https://github.com/ahrefs/ocannl/issues/627)). A tile that looks attractive in its
interior may lose badly at its edges.

Convolution has a similar set of questions. Its candidate families need a more structured
search representation; padding may enable a fast tile while also multiplying the work; and
some convolution shapes fall outside current recognition ([#697](https://github.com/ahrefs/ocannl/issues/697), [#740](https://github.com/ahrefs/ocannl/issues/740), [#741](https://github.com/ahrefs/ocannl/issues/741), [#912](https://github.com/ahrefs/ocannl/issues/912)).
These are opportunities to improve what search considers, rather than simply considering more.

Measurement itself must answer the question we intend. Timing a candidate in isolation and
timing it in a queue can choose different winners. The queued default needs comparisons on
CUDA and Metal, and its whole-session cost needs measurement ([#833](https://github.com/ahrefs/ocannl/issues/833), [#834](https://github.com/ahrefs/ocannl/issues/834)). Calibration
rows should account for admitted timings, and cache identity should distinguish devices when
the evidence is device-specific ([#922](https://github.com/ahrefs/ocannl/issues/922), [#594](https://github.com/ahrefs/ocannl/issues/594)). Persisting placement decisions would also
avoid repeatedly rediscovering choices whose schedule winners already survive across runs
([#786](https://github.com/ahrefs/ocannl/issues/786)).

Finally, the search's failure paths remain part of its semantics. A candidate can be timed
successfully and then encounter an exception in a callback. The suspected nonwinning-candidate
ownership gap needs an experiment before it can be called a leak ([#975](https://github.com/ahrefs/ocannl/issues/975)). Similarly, the
Metal device-memory fence has been fixed, while general analysis of cross-statement conflicts
between barrier regions remains open ([#963](https://github.com/ahrefs/ocannl/issues/963)). A faster schedule is useful only when its
ownership and synchronization obligations are understood.

## 3. Compute and store the part that is actually needed

Memory questions connect performance to the larger architecture. OCANNL can keep a value
virtual, recomputing its definition at reads, or materialize it. But the useful unit of storage
is not always the entire tensor.

Suppose a consumer only needs the diagonal of an expensive *n × n* virtual matrix. Computing
the full matrix uses quadratic storage and work even though only *n* entries are wanted.
Repeatedly inlining each needed computation can also be expensive if it is read many times.
A third possibility is to compute just the diagonal into private scratch and reuse it.

That is the motivation for footprint-scoped materialization ([#616](https://github.com/ahrefs/ocannl/issues/616)). The proposed scratch is
a derived value associated with the consumer's access pattern; it does not change the original
tensor's placement for every other consumer. The hard questions concern guards, overlapping
reader footprints, reuse and cost. This proposal is more specific than “add another cache.”

At runtime, device-memory pressure introduces a different problem ([#565](https://github.com/ahrefs/ocannl/issues/565)). Device memory can
be scarce while the OCaml heap is small, and the pool tables hold references that ordinary
garbage collection cannot release. Explicit release already handles known candidate lifetimes.
General pressure management goes further: it raises choices about eviction, recomputation, host
spill and pool policy. Captured GPU graphs and pointer-based identities constrain what can move or
be freed. The issue is deliberately a design space, not an already selected eviction algorithm.

Loading and staging form two further boundaries. Mapping a checkpoint into host memory has
landed; letting backend constant pools refer to that mapping without a further copy has not
([#585](https://github.com/ahrefs/ocannl/issues/585), now **v1.1**). Asynchronous staging has a foundation, but backend-specific copy paths
and deeper pipelines still need capability checks and a workload that rewards them ([#576](https://github.com/ahrefs/ocannl/issues/576)).
More buffering is not automatically more overlap or more speed.

Gemma 3 is proposed as a real-weights, longer-context demonstration and benchmark target
([#570](https://github.com/ahrefs/ocannl/issues/570)). Its role is to put attention, memory and architectural support under a meaningful
workload. The exact model configuration and feasible measurement envelope still need to be
established; a model name on the roadmap is not an implementation or a capacity result.

## 4. Make evidence easier to maintain

The **v1.1.1** consolidation queue concerns much of the machinery used to establish and explain
correctness. Deferring it does not make it unimportant. It separates testing-side refactoring
from the compiler elegance deliberately retained in v1.0.2, and gives the performance work a
chance to reveal which maintenance costs matter most.

One recurring design question is how much of OCaml a source scanner should understand. To
recognize a use through aliases, opens, shadowing and functors, a scanner can gradually acquire
its own approximation of the language's naming rules. Two alternatives are to use proper typing
information, or to ask the source to follow a small structural contract that needs no name
resolution. The guard-key, lifecycle-membership and emitted-code scans each face this choice
([#797](https://github.com/ahrefs/ocannl/issues/797), [#798](https://github.com/ahrefs/ocannl/issues/798), [#799](https://github.com/ahrefs/ocannl/issues/799)). Their answers need not be identical, but building three independent
partial resolvers would miss the point of the exercise.

Other work is less conceptual and no less useful: share the repository source inventory,
represent configuration parsing outcomes explicitly, test scanners against synthetic trees,
and compare inferred generated references with actual PPX expansion ([#911](https://github.com/ahrefs/ocannl/issues/911), [#916](https://github.com/ahrefs/ocannl/issues/916), [#603](https://github.com/ahrefs/ocannl/issues/603),
[#913](https://github.com/ahrefs/ocannl/issues/913), [#914](https://github.com/ahrefs/ocannl/issues/914), [#915](https://github.com/ahrefs/ocannl/issues/915)). A scanner should make its own limits legible as well as find defects.

Golden files pose a different question: which differences should be visible? A changed tensor
node number or a shifted source line may alter a golden without altering the behavior it was
meant to illustrate. Removing that noise requires deciding what identity and location information
is genuinely part of the example ([#642](https://github.com/ahrefs/ocannl/issues/642), [#672](https://github.com/ahrefs/ocannl/issues/672)). It is not a license to normalize away useful
diagnostics. Likewise, a skipped claim that is supposedly checked elsewhere should name an
actual coverage owner, and a cache-key classification should agree with what changes the key
([#926](https://github.com/ahrefs/ocannl/issues/926), [#596](https://github.com/ahrefs/ocannl/issues/596)).

The queue also includes shared child-process testing, safer harness retention, better CI timing
analysis, compiled documentation examples and a reading aid for editorial API changes ([#910](https://github.com/ahrefs/ocannl/issues/910),
[#607](https://github.com/ahrefs/ocannl/issues/607), [#918](https://github.com/ahrefs/ocannl/issues/918), [#660](https://github.com/ahrefs/ocannl/issues/660), [#946](https://github.com/ahrefs/ocannl/issues/946)). Hosted Metal CI is an evaluation of available coverage, not a
replacement for performance measurements on physical hardware ([#942](https://github.com/ahrefs/ocannl/issues/942)).

Some deferred issues concern production contracts rather than tests. What happens to cells
past a bound symbolic extent? The forward and gradient behavior should express one decision
([#928](https://github.com/ahrefs/ocannl/issues/928), [#929](https://github.com/ahrefs/ocannl/issues/929)). How do we distinguish an intentionally assigned launch value of zero from a
binding the caller forgot to set ([#940](https://github.com/ahrefs/ocannl/issues/940))? How should a placement diagnostic explain the reason
for a decision ([#609](https://github.com/ahrefs/ocannl/issues/609))? Their placement after v1.1 is a scope decision; an experiment that
needs one of these contracts can still bring the relevant work forward.

## 5. Turn capabilities into complete experiments

The consumers milestone is now **v1.1.2**. It asks what it is like to build, inspect, interrupt,
resume and share an experiment using the compiler.

Resumable training is a good example of the distance between a primitive and a complete
experience. Saving tensors or mapping a checkpoint does not by itself preserve everything
needed to continue a training run. The checkpointing issue needs a current specification of
that state—parameters, optimizer and training progress, and relevant randomness—rather than
literal implementation of the legacy module names in its original description ([#96](https://github.com/ahrefs/ocannl/issues/96)).
Experiment tracking and readable plots make that state visible over time ([#122](https://github.com/ahrefs/ocannl/issues/122), [#103](https://github.com/ahrefs/ocannl/issues/103)).

Two APIs expose unfinished behavior especially plainly. Batch normalization's running-stat
momentum and `mobile_cnn`'s channel-width multiplier are currently marked as placeholder options.
Implementing them means defining their behavior and applying it throughout the model, not
merely accepting an argument ([#879](https://github.com/ahrefs/ocannl/issues/879), [#880](https://github.com/ahrefs/ocannl/issues/880)).

Persistent optimizer state needs an equally explicit vocabulary. The current SGD fix recognizes
its state through a label convention; a declaration of cross-invocation state would make the
intent part of the model of memory ([#793](https://github.com/ahrefs/ocannl/issues/793)). Computed-value observability addresses another
unfinished promise: a virtual tensor should be inspectable through recomputation in a suitable
context, rather than relying on a best-effort printing proxy ([#777](https://github.com/ahrefs/ocannl/issues/777)). Together, these changes
would make more of the system's existing behavior expressible and explainable.

The model backlog supplies different kinds of pressure. LSTM and Bonsai networks exercise
recurrence. Digit addition is a reproduction with a compact target behavior. BERT/ModernBERT
exercises another transformer family; model surgery asks whether adapting an existing model is
straightforward ([#60](https://github.com/ahrefs/ocannl/issues/60), [#182](https://github.com/ahrefs/ocannl/issues/182), [#427](https://github.com/ahrefs/ocannl/issues/427), [#297](https://github.com/ahrefs/ocannl/issues/297), [#33](https://github.com/ahrefs/ocannl/issues/33)). These are invitations to develop complete
examples, not interchangeable boxes to check.

Deployment and integration are also exploratory: standalone inference artifacts, tabular data,
low-communication distributed training, and studies of other small frameworks ([#97](https://github.com/ahrefs/ocannl/issues/97), [#219](https://github.com/ahrefs/ocannl/issues/219),
[#278](https://github.com/ahrefs/ocannl/issues/278), [#435](https://github.com/ahrefs/ocannl/issues/435), [#277](https://github.com/ahrefs/ocannl/issues/277)). Each needs a concrete question or demonstration to justify its eventual
scope. The milestone is paced by interest; it is not a promise to reproduce every model listed.

## 6. Make programs reusable, and the compiler independently useful

The proposed **v1.2** theme is *reusable tensor programs on a standalone ArrayJIT compiler*.
Three architectural issues give that theme substance. They are substantial enough that their
small issue count should not be mistaken for a small release.

### A compiler that stands on its own

The package boundary proposed in [#852](https://github.com/ahrefs/ocannl/issues/852) puts loop-level IR, optimization, scheduling,
autotuning and lowered backends in ArrayJIT. Tensor-oriented assignments and context orchestration
belong with the neural-network library. Autotuning needs an abstract driver boundary so that
moving orchestration does not drag the compiler back into a dependency on the tensor layer.

The outcome should be understandable without a neural-network example: construct a loop
program, compile it, run it, and compare alternative implementations. The recently published
`arrayjit.ll_builders` is useful groundwork, but shared builders alone do not establish an
independent compiler. A reference driver, a deliberate public surface and a non-neural example
would make the distinction concrete.

### A failed construction should not poison the next one

Shape inference currently mutates more than a single environment value. A failed tensor
construction can leave substitutions and shared operand state partly updated. Resetting the
whole session is a recovery mechanism, but it is a poor foundation for trying alternatives or
interactively developing a model.

The transaction proposal ([#903](https://github.com/ahrefs/ocannl/issues/903)) asks for session-owned state and an explicit recovery boundary.
Imagine constructing an invalid layer application, reporting its error, and then successfully
constructing a valid one using the same earlier tensors. Making that ordinary is a stronger
criterion than restoring one global variable. The design choice between a mutation trail and
persistent state remains open; both must account for everything a construction changes.

Renderer state isolation is helpful progress elsewhere in the stack. It does not prove that
shape inference or compilation as a whole is reentrant. Those ownership boundaries need their
own designs and demonstrations.

### A tensor function should carry reusable shape knowledge

A shape scheme records relationships among a function's arguments and results, rather than
just the concrete shape of one application. Applying the function again would freshen the
scheme's variables and connect them to the new arguments ([#404](https://github.com/ahrefs/ocannl/issues/404)).

For example, a reusable block might preserve its input's batch axes while relating its input
and output feature dimensions. The interesting artifact is that relationship. A second
application at another compatible shape should not need to rediscover it from scratch, and
an incompatible application should produce a useful error. Expected benefits include less
repeated inference and earlier detection of some errors, while finalization may still be needed
to resolve others.

Session ownership and recovery matter because freshening a scheme and tentatively applying it
must not accidentally share mutable inference state with another application. A compelling
v1.2 demonstration would combine reuse at several shapes with an invalid application followed
by a valid one. That makes the architecture visible through behavior.

PoPE can be an additional reusable-block example ([#444](https://github.com/ahrefs/ocannl/issues/444)). CUDA pinned host buffers and constant
memory have their own ownership and measurement questions ([#170](https://github.com/ahrefs/ocannl/issues/170), [#195](https://github.com/ahrefs/ocannl/issues/195)). CDNA MFMA needs
appropriate hardware; the existing RDNA-oriented HIP coverage does not establish it ([#477](https://github.com/ahrefs/ocannl/issues/477)).
These remain worthwhile companions, but the proposed architectural theme is coherent without
requiring every hardware-dependent feature to finish at the same time.

## The reshuffled milestones at a glance

The calendar is backward-chained from landing v1.2 around early November. Compiler-side
deduplication stays before performance; testing-side consolidation follows it; the former
consumer milestone moves from v1.1.1 to v1.1.2. Version depth describes scope in this project,
not semantic-versioning compatibility.

| Milestone | Soft target | Role |
|-----------|-------------|------|
| v1.0.2 | September 16 | Landed robustness plus the eight retained compiler-elegance and coverage issues. |
| v1.1 | October 2 | Performance improvements supported by meaningful comparisons. |
| v1.1.1 | October 10 | Deferred consolidation, especially testing-side refactoring. |
| v1.1.2 | October 18 | Training experience, models, reproductions and integrations. |
| v1.2 | November 3 | Proposed architectural focus: standalone ArrayJIT, recoverable inference sessions and shape schemes. |

The dates are end-of-period targets. A milestone can leave work open without invalidating the
release's contribution. If scope exceeds the available time, the early-November anchor calls
for a coherent smaller result and an explicit deferral, rather than silently moving every later
date. For the current calendar and assignments, follow the roadmap and linked issues.

## Issue directory

This directory covers all 107 open OCANNL issues in the September 14 snapshot, after the
reshuffle. It is a map from subjects to the detailed discussions, not an implementation order.
Completed foundations mentioned above do not appear in this open-issue directory.

### v1.0.2 — 8 open issues

| Subject | Issues |
|---------|--------|
| Backend structure and actual vendor compilation | [#770](https://github.com/ahrefs/ocannl/issues/770), [#794](https://github.com/ahrefs/ocannl/issues/794) |
| Shared IR semantics and affine structure | [#630](https://github.com/ahrefs/ocannl/issues/630), [#774](https://github.com/ahrefs/ocannl/issues/774) |
| Builtins, precision, configuration and shuffle stages | [#656](https://github.com/ahrefs/ocannl/issues/656), [#917](https://github.com/ahrefs/ocannl/issues/917), [#604](https://github.com/ahrefs/ocannl/issues/604), [#875](https://github.com/ahrefs/ocannl/issues/875) |

### v1.1 — 37 open issues

| Subject | Issues |
|---------|--------|
| Numerical regime, workloads and algebraic rewrites | [#719](https://github.com/ahrefs/ocannl/issues/719), [#720](https://github.com/ahrefs/ocannl/issues/720), [#483](https://github.com/ahrefs/ocannl/issues/483), [#505](https://github.com/ahrefs/ocannl/issues/505), [#952](https://github.com/ahrefs/ocannl/issues/952), [#570](https://github.com/ahrefs/ocannl/issues/570) |
| Backend-specific matrix accumulation | [#923](https://github.com/ahrefs/ocannl/issues/923), [#925](https://github.com/ahrefs/ocannl/issues/925), [#838](https://github.com/ahrefs/ocannl/issues/838) |
| Register geometry, remainders and head-axis tiling | [#947](https://github.com/ahrefs/ocannl/issues/947), [#948](https://github.com/ahrefs/ocannl/issues/948), [#620](https://github.com/ahrefs/ocannl/issues/620), [#627](https://github.com/ahrefs/ocannl/issues/627), [#728](https://github.com/ahrefs/ocannl/issues/728) |
| Convolution recognition, search and padding | [#503](https://github.com/ahrefs/ocannl/issues/503), [#697](https://github.com/ahrefs/ocannl/issues/697), [#740](https://github.com/ahrefs/ocannl/issues/740), [#741](https://github.com/ahrefs/ocannl/issues/741), [#912](https://github.com/ahrefs/ocannl/issues/912), [#939](https://github.com/ahrefs/ocannl/issues/939) |
| Placement persistence and reduction/search evidence | [#786](https://github.com/ahrefs/ocannl/issues/786), [#717](https://github.com/ahrefs/ocannl/issues/717), [#724](https://github.com/ahrefs/ocannl/issues/724) |
| Cost models, benchmark protocol and timing evidence | [#636](https://github.com/ahrefs/ocannl/issues/636), [#637](https://github.com/ahrefs/ocannl/issues/637), [#743](https://github.com/ahrefs/ocannl/issues/743), [#819](https://github.com/ahrefs/ocannl/issues/819), [#833](https://github.com/ahrefs/ocannl/issues/833), [#834](https://github.com/ahrefs/ocannl/issues/834), [#922](https://github.com/ahrefs/ocannl/issues/922) |
| Materialization, memory pressure, staging and cache identity | [#616](https://github.com/ahrefs/ocannl/issues/616), [#565](https://github.com/ahrefs/ocannl/issues/565), [#576](https://github.com/ahrefs/ocannl/issues/576), [#585](https://github.com/ahrefs/ocannl/issues/585), [#594](https://github.com/ahrefs/ocannl/issues/594) |
| Barrier-region legality and candidate ownership | [#963](https://github.com/ahrefs/ocannl/issues/963), [#975](https://github.com/ahrefs/ocannl/issues/975) |

### v1.1.1 — 38 open issues

| Subject | Issues |
|---------|--------|
| Scanner resolution: typed information or structural contracts | [#797](https://github.com/ahrefs/ocannl/issues/797), [#798](https://github.com/ahrefs/ocannl/issues/798), [#799](https://github.com/ahrefs/ocannl/issues/799) |
| Source inventory, configuration parsing and coverage | [#911](https://github.com/ahrefs/ocannl/issues/911), [#916](https://github.com/ahrefs/ocannl/issues/916), [#920](https://github.com/ahrefs/ocannl/issues/920), [#603](https://github.com/ahrefs/ocannl/issues/603), [#596](https://github.com/ahrefs/ocannl/issues/596) |
| Shell/opam lexical context and citation recognition | [#907](https://github.com/ahrefs/ocannl/issues/907), [#921](https://github.com/ahrefs/ocannl/issues/921), [#932](https://github.com/ahrefs/ocannl/issues/932) |
| Generated references and the dead-export census | [#913](https://github.com/ahrefs/ocannl/issues/913), [#914](https://github.com/ahrefs/ocannl/issues/914), [#915](https://github.com/ahrefs/ocannl/issues/915) |
| Test support, coverage ownership, goldens and GPU-free execution | [#910](https://github.com/ahrefs/ocannl/issues/910), [#926](https://github.com/ahrefs/ocannl/issues/926), [#642](https://github.com/ahrefs/ocannl/issues/642), [#672](https://github.com/ahrefs/ocannl/issues/672), [#778](https://github.com/ahrefs/ocannl/issues/778), [#678](https://github.com/ahrefs/ocannl/issues/678) |
| Diagnostics and remaining tensor/IR contracts | [#609](https://github.com/ahrefs/ocannl/issues/609), [#641](https://github.com/ahrefs/ocannl/issues/641), [#707](https://github.com/ahrefs/ocannl/issues/707), [#625](https://github.com/ahrefs/ocannl/issues/625), [#818](https://github.com/ahrefs/ocannl/issues/818), [#940](https://github.com/ahrefs/ocannl/issues/940), [#928](https://github.com/ahrefs/ocannl/issues/928), [#929](https://github.com/ahrefs/ocannl/issues/929) |
| Documentation, historical evidence, CI and harness retention | [#660](https://github.com/ahrefs/ocannl/issues/660), [#918](https://github.com/ahrefs/ocannl/issues/918), [#919](https://github.com/ahrefs/ocannl/issues/919), [#934](https://github.com/ahrefs/ocannl/issues/934), [#942](https://github.com/ahrefs/ocannl/issues/942), [#946](https://github.com/ahrefs/ocannl/issues/946), [#607](https://github.com/ahrefs/ocannl/issues/607) |
| PPX compatibility and constructor-surface discipline | [#695](https://github.com/ahrefs/ocannl/issues/695), [#705](https://github.com/ahrefs/ocannl/issues/705) |
| Dated retirement of legacy test-run locks | [#966](https://github.com/ahrefs/ocannl/issues/966) |

### v1.1.2 — 17 open issues

| Subject | Issues |
|---------|--------|
| Training state, observability, tracking and plots | [#96](https://github.com/ahrefs/ocannl/issues/96), [#793](https://github.com/ahrefs/ocannl/issues/793), [#777](https://github.com/ahrefs/ocannl/issues/777), [#122](https://github.com/ahrefs/ocannl/issues/122), [#103](https://github.com/ahrefs/ocannl/issues/103) |
| Batch-normalization momentum and model width | [#879](https://github.com/ahrefs/ocannl/issues/879), [#880](https://github.com/ahrefs/ocannl/issues/880) |
| Models, reproductions and model surgery | [#33](https://github.com/ahrefs/ocannl/issues/33), [#60](https://github.com/ahrefs/ocannl/issues/60), [#182](https://github.com/ahrefs/ocannl/issues/182), [#297](https://github.com/ahrefs/ocannl/issues/297), [#427](https://github.com/ahrefs/ocannl/issues/427) |
| Inference artifacts, data/distributed integrations and framework studies | [#97](https://github.com/ahrefs/ocannl/issues/97), [#219](https://github.com/ahrefs/ocannl/issues/219), [#278](https://github.com/ahrefs/ocannl/issues/278), [#277](https://github.com/ahrefs/ocannl/issues/277), [#435](https://github.com/ahrefs/ocannl/issues/435) |

### v1.2 — 7 open issues

| Subject | Issues |
|---------|--------|
| Standalone compiler and reusable inference | [#852](https://github.com/ahrefs/ocannl/issues/852), [#903](https://github.com/ahrefs/ocannl/issues/903), [#404](https://github.com/ahrefs/ocannl/issues/404) |
| Polar position embeddings | [#444](https://github.com/ahrefs/ocannl/issues/444) |
| CUDA memory and CDNA matrix hardware | [#170](https://github.com/ahrefs/ocannl/issues/170), [#195](https://github.com/ahrefs/ocannl/issues/195), [#477](https://github.com/ahrefs/ocannl/issues/477) |

