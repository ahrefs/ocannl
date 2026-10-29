# Backend dialects, identifiers and bindings

Where one GPU dialect differs from another, what a generated name may collide with, and how the
backend config functor binds its overrides.

Part of the agent notes; the [index](../agent-notes.md) carries the scope discipline and the other
files.

- Metal shader compiler miscompiles serial accumulations, and API validation hides it
  (`MTL_SHADER_VALIDATION=1` makes it vanish). Two spellings are affected: the original
  `acc[k] = acc[k] + f(i)` device-memory RMW can leave only the last iteration (fingerprint: loss
  ≈ correct/batch_size), and gh-ocannl-731 showed that codegen's replacement scope-local
  accumulator can instead diverge by a data-independent additive constant.
  `volatile_serial_accumulation` in `arrayjit/lib/c_syntax.ml` (Metal sets it) therefore renders
  device reads inside the accumulating update and its controlling guards through expression-level
  `volatile` pointer casts; accumulator declarations, opening reads, vectorized/packed paths and MMA
  reads stay plain (gh-ocannl-820). The volatility census (`Context.routine.volatility`) reports per
  routine which localized sites and device-memory RMWs received that form. Standalone repros:
  `bench_metal_bug.ml` (RMW form) and `bench_metal_bug_local.ml` (localized form, plus a
  one-factor-at-a-time matrix and the tax measurement) under `benchmarks/runners/ocannl/`; executed
  guards: `scalar_rmw_accumulation.ml`, `reduction_accumulator_residency.ml`, and `rope_test.ml`.
  Suspect this class first for a Metal-only accumulation bug that disappears under shader
  validation.
- A test that reads emitted text and counts node accesses must count BOTH spellings of a device
  read: `ident[idx]`, and the cast form `ident)[idx]` that expression-level `volatile` produces. The
  failure direction is a false pass, so it does not announce itself on the CPU backends that never
  emit the cast: an uncounted read costs a read-modify-write its second subscript, and a statement
  still hammering the device cell every step then reads as one that touches it once. Both existing
  readers normalize -- `reduction_accumulator_residency.ml` counts the two patterns
  (`count_node_accesses`), `reduction_forms.ml` erases the cast before any counting
  (`strip_volatile_casts`) so its several readings cannot disagree about what an access is. Metal is
  off the per-PR path, so such a test can only be wrong on the daily sweep; a normalizer checked
  directly on both spellings, as `reduction_forms.ml` does, holds on every backend instead.
- What that matrix refuted is worth knowing before proposing a narrower predicate (gh-ocannl-782,
  M4 Max / macOS 26): the pooled slot table is NOT the trigger — building the pointers from literal
  offsets straight off a kernel parameter, with no dynamic load anywhere, miscompiles identically —
  and neither is `__restrict`, nor a device memory barrier, nor where the preceding device store
  lands (moving it to an unrelated cell keeps the defect; removing it altogether is what stops it,
  which no kernel can rely on). The two things that do stop it are the `volatile` accumulator and a
  `volatile` READ pointer — in either spelling, the pointer's declaration or the per-read cast the
  backend actually emits (`volatile-source`, `volatile-source-expr`); an accumulating loop that
  dereferences no node pointer at all never miscompiled. A read reached only through the update's
  controlling guard sits on that same safe side: `guard-read-only` — the update reading nothing
  while its `if` reads the node — computes correctly unqualified. So the reproducer-backed
  narrowing axis excludes accumulations whose accumulating expression reads no device memory, which
  are rare and usually constant-folded. The shipped rule is deliberately wider than that axis and
  qualifies a recognized accumulation's controlling guards too (gh-ocannl-820, Codex P1 round 3 on
  staging#553): one 4x16 shape on one toolchain not reproducing is not a shape proven safe, and
  confining the context to the update alone would leave a guard-only accumulation with no
  workaround at all.
  Measured, the tax is 1.06x on a memory-bound per-thread reduction, 2.15x accumulator-bound, 4.1x
  on a single-threaded scalar-loss reduction. The `volatile`-source form costs 1.03x on that last
  shape; gh-ocannl-820 adopted its expression-level equivalent after the matrix stayed green
  row-for-row.
- Metal `Where` must stay a short-circuiting ternary: MSL `select` is a function call that
  evaluates BOTH branches, so any range guard's deliberately out-of-range read (clamped windows,
  inlined-concat component guards) would still be evaluated. Codegen pins:
  `test_where_precision.metal.expected`, `test_metal_guarded_gather_codegen`.
- Metal has no `double`, and f64 stays rejected in `typ_of_prec` — a declared double buffer must
  fail rather than be silently degraded; only scalar expression casts render as `float`. So a
  backend-agnostic test must never reach for `double` just to have a second precision:
  `digest_identity_flips` flipped `default_value_prec` to f64 to probe a code-borne cache-key knob,
  and aborted the whole metal `test/operations` run (gh-ocannl-632). Half is the portable choice.

- Metal buffer binding is the pooled slot-table (`__pools` + `__pool_slots`); raw `gpuAddress`
  casts segfault at dispatch and argument encoders don't fit the binding model. Same-queue
  command buffers overlap over untracked resources: back-to-back runs of the SAME routine need
  the FIFO wait, pipelined (no-sync) timing is unreliable, and `get_values`/`set_values` do FULL
  awaits by design.
- What a kernel lists from its node store (parameters, Metal's pool-slot table, local declarations)
  goes in `C_syntax.store_order`: first-touch order in the code (`Low_level.touched_tns_in_order`),
  never a traversal of the uid-keyed `traced_store` hash table and never a sort by spelling — node
  identifiers and static-index names carry minted numbers (`n9999_w`, `i12`), and a string sort of
  those flips at a power of ten. Two fresh builds of one program therefore emit one kernel up to
  those numbers; `Test_utils.Generated.normalize_minted` renames them away, and
  `test/operations/fresh_build_kernel_text` (gh-ocannl-1473) pins it, ids straddling 10^4 included.
- **A `restrict` kernel pointer is a promise no other THREAD hands values through that buffer
  within the kernel, and barriers do not take it back** (gh-ocannl-1598). The device compilers
  read the no-alias qualifier as "this thread's memory alone": on CUDA's pad-masked tensorized
  conv seeds, where lane 0 stores the shared fragment back for its whole workgroup and a companion
  then reads the cells per lane, nvrtc's PTX for `compute_80`..`compute_90` (the tf32 wmma floor
  OCANNL targets) kept each other lane's own zeroing store of the last zeroed row in a register
  across `__syncthreads()` and lane 0's stores, so that row read back as 0; `compute_100`+ PTX and
  the same kernel with `pr` unqualified computed it right. `C_syntax.lane_restrict` therefore
  leaves `Low_level.lane_selected_writes` (nodes written under an `if lane == 0` of a `Workgroup`
  loop) unqualified in both parameter styles; every other pointer keeps the qualifier the
  register promotion of gh-ocannl-1348 relies on. Reproducing a codegen-flag-dependent wrong value
  outside OCANNL: an nvrtc + driver harness that compiles the emitted `.cu` with the backend's
  options (`Compiler_options.nvrtc`, the `--gpu-architecture=` from `Cuda_mma.gpu_arch_options`)
  and launches it at `Low_level.launch_dims`; `nvcc -arch=native` compiles for the device's own
  arch and hides this class.
- **CUDA ships floor PTX, and a value can depend on which target nvrtc compiled for**
  (gh-ocannl-1674). `Cuda_mma.with_ptx_target Native` compiles every kernel in its extent at the
  device's own `compute_<cc>` instead of the marker floor (or nvrtc's default arch for an unmarked
  kernel), and returns the `.target` each program's PTX actually carries; the kernel code cache
  keys on the option list, so the two targets never share a kernel.
  `test/operations/ptx_target_parity` compares the two bitwise on the cm8w conv seeds under tf32
  and caught the restrict hazard above with no reference of its own (512 of 8192 cells, with
  `lane_restrict` reverted). To cover another kernel class, compile it under both targets the same
  way.
- Query optional `MTLCompileOptions` properties on an initialized object, not through the nominal
  class's `instancesRespondToSelector:`. On macOS 26.6.2 that class-level query returned false for
  the macOS-15 math properties even though the object accepted them, silently selecting the macOS-14
  fallback. `Metal_math_api_runtime.get` owns the runtime query shared by backend and probe; the
  selector-to-branch decision stays injectable in `Compiler_options.metal_math_api` (gh-ocannl-882).

- Emitted float constants go through `C_syntax.c_float_literal`, the single site that turns a
  `Low_level.Constant` into kernel text, and it owes three things at once (gh-ocannl-623): a radix
  point or exponent, so the token is a *floating* literal and not an integer one the cast happens to
  convert (`%.16g (-0.)` was `"-0"`, hence `+0.0`, a live corruption on every C-family backend);
  enough digits to round-trip, since 16 do not recover every double (`0.1 +. 0.2` and `max_float`
  both print as a *different* double at `%.16g`, and hosted constant inits inline arbitrary host
  values through here); and spellings for `INFINITY` / `(-INFINITY)` / `NAN`, which MSL provides and
  the CUDA and HIP preludes `#ifndef`-define. What it deliberately does NOT emit is a precision
  suffix: the literal stays double-typed and `convert_precision`'s cast narrows it, which is one
  rounding of the exact host double — an `f`-suffixed decimal rounds the decimal straight to float
  and disagrees on a value sitting on a float tie. The one place that reasoning has a hole is worth
  knowing, because it is invisible from the C backends: the cast only IS a narrowing where the
  dialect has a `double`, and **MSL does not** — Metal rounds the emitted decimal itself, at parse
  time. Round-tripping is not exactness, so "round the value to f32" and "round a decimal near it
  to f32" can differ, and they differ exactly on an f32 tie (host takes ties-to-even; the dialect
  follows whichever side the decimal fell on), which makes the emitted digit count decide the
  value. Hence `is_f32_tie` -> a `%h` hexadecimal literal, exact by construction so neither reading
  has anything to round; C99, CUDA, HIP and MSL all accept the form, and only ties are spelled that
  way. Guard: `test/operations/float_literal_forms`, bitwise at f64/f32/f16 plus the emitted
  tokens, and its three tie cases fail on Metal without the hex spelling.
- Two of those three obligations are not C's: an IR dump wants a floating literal that round-trips
  for the same reasons a kernel does, and the `.ll`/`.cd` printers (`Low_level.to_doc_cstyle`,
  `Low_level.to_doc`) used to render a `Constant` with a bare `%.16g` of their own — so `2.` showed
  as `2`, `-0.` as the integer `-0`, and a constant whose 17th digit mattered was displayed as its
  16-digit neighbour. That is the debug surface you reach for when chasing this exact class of bug,
  which is what made it worth more than tidiness (gh-ocannl-713). The radix-forcing and the
  `%.17g` retry now live in `Utils.decimal_float_literal`, shared by `c_float_literal` and both
  printers; the C-dialect spellings (`INFINITY`, the tie's `%h`) stay in `c_float_literal`, since a
  dump is not C and its specials keep `%.16g`'s `inf`/`nan` words. Guard:
  `test/operations/ll_printer_constants`, the dump-side twin of `float_literal_forms`.
- Reduced-precision *literals* are dialect-specific and do not transpose between backends. `0.0h`
  is a clang extension and valid MSL, but not CUDA C++ — nvrtc rejects it with "user-defined
  literal operator not found" (gh-ocannl-518, the half `Relu_gate`). On CUDA/HIP write the zero as
  `__ushort_as_half((unsigned short)0x0000U)` (bf16: `__ushort_as_bfloat16`), and prefer the
  intrinsic comparisons (`__hgt`/`__hlt`) over operators: mixing a `__half`/`__nv_bfloat16` with a
  literal of another arithmetic type is separately ambiguous under nvrtc/hiprtc, since the type's
  implicit conversion operators make the overload sequences indistinguishable (see the bf16
  comments in `cuda_backend.ml`/`hip_backend.ml`). Same family as the MSL `bfloat` trap below —
  a reduced-precision literal or overload that is fine in one dialect is a hard error, or worse a
  silent truncation, in another. Such bugs only surface with that vendor's hardware attached; the
  executed guards are `test/operations/half_ops.ml`, `test/operations/bf16_ops.ml` (operand
  ambiguity) and `test/operations/bf16_builtins.ml` (builtin return types), plus
  `test/training/mixed_prec_parity.ml`.
- MSL's math library has **no `bfloat` overload of any builtin** — `sqrt`, `exp`, `log`, `pow`,
  `fmax`, `fmin`, `fmod`, `trunc`, `rsqrt`, `tanh`, `fma` all promote to `float` and return
  `float`, and unlike C, MSL then rejects the narrowing assignment back to a `bfloat` destination
  ("assigning to 'bfloat' from incompatible type 'float'"). So the bridge belongs on the whole
  math-builtin family, not per operator: `metal_backend.ml`'s `bf16_from_builtin` casts the result
  back (gh-ocannl-549). What is *not* affected, and needs no bridge: arithmetic operators,
  comparisons, the ternary, `!`, and the `0.0bf` literal suffix — MSL's `bfloat` is a native scalar
  type. `half` has the full overload set, so f16 has no such gap. Verify claims like these by
  compiling one-line kernels through `Metal.Library.on_device` rather than by reasoning about the
  spec; there is no `xcrun metal` without full Xcode.
- The same bf16 emission fails *differently* per GPU dialect, which is why one backend's evidence
  misleads about another's (gh-ocannl-549). A float-returning builtin at bf16 is the single root
  site; where the dialect complains depends on where that float lands. MSL rejects the assignment,
  so every placement fails. CUDA/HIP accept it (`__nv_bfloat16`/`__hip_bfloat16` have an implicit
  converting constructor from float), so the materialized placement — which stores each result in
  its own bf16 node — compiles, and only the placement that *inlines* the builtin into a consuming
  bf16 binop fails, on the operand: nvrtc reports a mixed-operand `__hadd` (its bf16 `Add` arm is
  `func "__hadd"`), hiprtc reports `operator '+' is ambiguous ('__hip_bfloat16' and 'float')` (its
  `Add` falls through to plain `+`). A placement-dependent bf16 compile error is therefore a clue
  about *inlining*, not about a fission-introduced mixed type — nothing introduces a float, the op
  table's own `expf`/`sqrtf`/... arms return one.
- The MSL bf16 trap's older half: an *untyped* literal does not fail loudly. `max(0, v)` makes an
  integer overload unambiguous, so it compiles and silently truncates every sub-unit activation to
  0, whereas `max((bfloat)0.0, v)` is a clean "call to 'max' is ambiguous" error. Fingerprint of
  the silent form: loss pinned at exactly ln(#classes) with NO batch-to-batch variation (a
  frozen-weights bug would still vary per batch; an input-independent forward does not). Found by
  the gh-ocannl-476 sweep; `Relu` at `Bfloat16_prec` had fallen through to a catch-all commented
  `Byte_prec, Void_prec`. When adding a precision, audit every `unop_syntax`/`binop_syntax`
  catch-all arm.
- Tensor-node debug names become identifiers verbatim in the emitted kernel, so anything the
  backend also emits as a *name* must be reserved (`ident_blacklist`). Reserve it from the
  backend's own syntax functions, never from the C spellings: `C_syntax.op_syntax_idents` renders
  every (precision, operator) pair over a placeholder and harvests the identifiers, so an override
  cannot drift out of the list. Deriving from `Ops.*_c_syntax` instead described C only and left
  MSL's unsuffixed `tanh`/`exp`/`log`/`sqrt`/`sin`/`cos`/`trunc` free — and those are exactly the
  `Tensor.unop ~op_label` labels, so a GPT-2 gelu declared `device float *__restrict tanh` and the
  call on the next line resolved to the pointer (gh-ocannl-553). A backend's builtins-table keys
  belong in the list too: a node taking one shadows the definition *and* drags it into a kernel
  that never calls it, since `filter_and_prepend_builtins` selects entries by searching the
  rendered kernel for their key. The collision only bites when one kernel holds both the
  declaration and the call, so which backend it fires on depends on fissioning — the guard is
  `test/operations/test_ident_blacklist.ml`, and its section 3 only has teeth under
  `OCANNL_BACKEND=metal` (C spells these with an `f` suffix, so no C compile can exhibit it).
  The GPU dialects reserve the C++ keywords on top of the C ones: CUDA, HIP and MSL are all parsed
  by a C++ front end, so `C_syntax.cpp_keywords` (one shared table, added to all three) covers
  plausible labels like `class`, `new`, `operator`, `bool` and `this` that plain C leaves free.
- On the GPU dialects every node identifier also carries a fixed prefix (`node_ident_prefix`, `t_`,
  from `C_syntax.node_ident_prefix_for_backend`), because their vendor headers declare global types
  and macros outside the reserved namespace, and a blacklist would have to track headers the runtime
  compilers carry built in (gh-ocannl-1338). The measured case: Metal stores a uint4x32 node as
  `uint4`, so a node labelled `uint4` made the next `device uint4* ...` of its kernel fail to parse;
  a node labelled after a macro (`M_PI_F`, `CUDARTAPI`) is expanded, not declared. The prefix
  is on the EMITTED identifier only (`get_ident`); the code name (`get_code_ident`: `Tn.code_name`,
  debug output, routine-log message text) and scope locals (`v<scope>_<name>`) keep the bare name,
  though a log line echoing kernel code (`# local ... := <code>`) shows the prefixed one. A claim
  pinning a node's emitted identifier spells it `Generated.node_ident`, never a literal. On every
  backend a label in the reserved namespace (`__LINE__`, `_Bool`, `__CUDA_ARCH__`) or a predefined
  system macro (`linux`, `unix`) takes the `n<id>_` form instead; cc keeps bare names otherwise, its
  only headers being the C library's, which `ident_blacklist` covers. Guard:
  `test/operations/vendor_node_names.ml`, executed per backend (on base its macro labels fail on
  every backend, `uint4` on Metal).
- Routine names are the *other* identifier class reaching the emitted source, and they do not go
  through the node-name machinery at all — a node's colliding label is forced into `n<id>_<label>`
  and every local carries a `v<scope>_` / `wred_` / `__rmw_` prefix, but a routine name was emitted
  verbatim, so `Block_comment ("asm", ...)` produced `void asm(` and the compiler's rejection
  surfaced as "this is a bug in OCANNL … file an issue with the generated .c file", naming
  everything except the name that caused it (gh-ocannl-686). `C_syntax.kernel_ident` is the fix and
  the rule: apply it once at each backend's `compile`/`compile_batch` entry, *before* anything
  derives a file name or a symbol, so the emitted header, the `dlsym` / `cuModuleGetFunction` /
  `hipModuleGetFunction` / `new_function_with_name` lookup and the `.c`/`.cu`/`.hip`/`.metal`
  artifact all name the same thing; `compile_proc` refuses a name its caller did not mangle rather
  than emitting one. It is the identity on any legal non-reserved identifier, which is what keeps
  schedule-cache identities and existing goldens from churning. The name a routine is *known* by —
  the routine record, the `.cd`/`.ll` sources, calibration rows — deliberately keeps the caller's
  spelling; only the C-family artifacts carry the mangled one. Guard:
  `test/operations/reserved_routine_names.ml` (executes each colliding routine, not just parses it).
- A third identifier class is the renderers' own fixed-name temporaries (`vec_result`,
  `new_set_v`, `new_set_local_v`, the register tile's `tmma_*`, the GPU arms' `__mma_*`, the
  vectorized loops' counter-named `vset1__`...). Two statements that declare one, rendered side by
  side in one C scope, redeclare it, and the kernel fails to compile. The rule: the statement that
  declares a fixed name braces itself on EVERY path, or the name is declared once per kernel (a
  parameter, the pool or log prologue). Bracing "only when there are local definitions" is the trap:
  `Set_from_vec` did that until a simplification left two packed draws with none (gh-ocannl-1498).
  The gh-ocannl-1553 audit found no other open site. The register tile sits inside a second scope,
  the `lane0_guarded` `if (lane == 0) { }`, so removing its literal brace alone shows nothing. Guard:
  `test/operations/fixed_name_temporaries.ml`, which builds two of each family as siblings,
  compiles and executes them, with and without routine logging.
- Beyond the C floor (gh-ocannl-707), a routine name meets the vendor headers, and only some of
  their names collide with a kernel DEFINITION: global types, macros, dialect keywords and
  C-linkage functions do; ordinary C++ functions do not, since overloading lets a kernel coexist
  with them (on Metal every `metal::` function compiled as a kernel name). Three layers: the
  implementation's reserved namespace (`__x`, `_X`, where vendors put most intrinsics) is closed by
  rule, `C_syntax.implementation_reserved`; each backend's `vendor_routine_idents` holds the rest,
  Metal's harvested by compiling every identifier token of the toolchain's `include/metal` headers
  as a kernel name (582 failed; kept as the grammar they instantiate), CUDA's and HIP's GENERATED
  by the same experiment (`tools/vendor_idents.exe`, below); and whatever a later SDK adds fails
  through `C_syntax.attribute_routine_name_failure`, which turns a compiler report quoting a
  routine's `void <name>(` line, or an error line quoting the name, into a `Utils.User_error`
  naming the routine. Only an ERROR diagnostic's lines count (`C_syntax.diagnostic_scopes` assigns
  each echoed source line to the header before it): clang's `note: to match this '('` quotes the
  declaration line for an error anywhere in the parameter list, and blaming the name there sent a
  macro-expanded node identifier to the user as a routine-name collision. A Metal compile error is a `Failure` carrying the whole source, where every
  declaration appears, so attribution there reads the compiler's message alone. The attributed
  message quotes the report less its "This is a bug in OCANNL. Please file an issue with …" line,
  which cc's report carries for the unattributed case. cc builds that line with
  `C_syntax.codegen_bug_claim`, and the attribution drops exactly that line; a backend whose
  compile-time report gains the sentence must build it the same way. (CUDA's copy lives in its
  exception classifier, which only sees an nvrtc failure attribution let through.)
- The CUDA and HIP routine-name tables are `arrayjit/lib/vendor_idents_{cuda,hip}.ml`, generated by
  `tools/vendor_idents.exe` on a box with the toolkit (`tools/test-run.sh run exec
  tools/vendor_idents.exe -- cuda`; about ten minutes for HIP, whose rocWMMA prelude dominates each
  hiprtc compile): it compiles every identifier token of the headers the prelude reaches as a kernel
  name with the backend's own compiler, settles each failing batch by the names its report blames
  (each confirmed ALONE) or by bisection, and prints the measurement, the missing and superfluous
  entries, and the module to paste. Exit 1 says the checked-in table no longer matches the
  toolchain; `--names a,b` spot-checks entries. Reading the headers had been wrong in both
  directions (gh-ocannl-1338): nvrtc accepts kernels named `float4`, `uint4`, `char4` (CUDA's
  vector types are structs, which a same-named function hides -- legal, though it would break a
  later use of the type in the same source; the emitter spells none in CUDA kernel bodies) and
  `CUDART_INF_FP16` (undefined under nvrtc), yet rejects `rsqrtf`, `norm` and some 900 runtime-API
  macros and enumerators (`CUDARTAPI`, `CUDA_C_16BF`); hiprtc rejects `float4`, `char4`, `FLT_MAX`,
  `HIPRT_INF_FP16` but accepts `rsqrtf`. Metal's runtime compiler carries its headers inside the
  framework, so without the Metal toolchain installed the tool measures only the table's own
  entries (`--include-dir` points it at an `include/metal` tree).
- `test/config/ocannl_config` pins `backend=cc`, so `dune runtest` never exercises GPU codegen —
  a Metal/CUDA-only rendering bug passes a fully green suite. The bf16 bug above was already
  covered by `test/training/mixed_prec_parity.ml` (its "loss trajectory parity within 0.1" check
  would have caught a zeroed forward); it had simply never run on a GPU backend. Run
  `OCANNL_BACKEND=metal dune runtest` (the env var is an explicit dune dependency, so it re-runs)
  before trusting a backend-specific codegen change.
- Parallel-codegen work often lands Metal → cc → CUDA/HIP, but that is a default reflecting
  which machine is booted first and used most (the Mac Studio), not a rule — tasks can start on
  CUDA or HIP for load balancing across machines. The durable part: codegen snapshots for a
  backend whose hardware isn't attached (`.cu.expected` etc.) go stale until that hardware next
  runs the suite — expect re-promotes.
- A routine-log statement's printf conversions have to track the width of `B.loop_index_type`, which
  `large_models` widens to 64 bits. `C_syntax_config.log_index_arg` is the seam: it returns the
  conversion specification and the cast for one loop-index-typed argument, defaulting to
  `%lld` / `(long long)` — the widest pair C's variadic promotions name without `<inttypes.h>`,
  accepted by gcc, clang, the MSVC runtime, nvrtc and hiprtc alike. Metal overrides it, because MSL
  has no `long long` at all: its 64-bit signed scalar is `long` (`%ld`), and `os_log` checks the
  format string against the argument types at shader-compile time, so a merely-wide-enough
  conversion is a compile error there rather than silent UB. A backend spelling the loop index as
  its own type overrides both together. Every array offset in a value-log line goes through the
  same seam, including nested `debug_float` accessors and vector-store offsets (gh-ocannl-953), so
  a wide routine never pairs a 64-bit offset argument with a narrow conversion.
- Metal routine logs arrive on Metal's own dispatch threads, through the `MTLLogState` handler,
  so no OCaml may run there. The backend installs ocaml-metal's native
  `LogState.add_buffered_log_handler`, which copies each message into a C-side buffer, and `await`
  drains it on the host thread with `drain_log_buffer` into `Utils.log_debug_routine_logs`
  (gh-ocannl-1594). Under logging, `await` also waits for its own signalling command buffer to
  complete before draining, since no Metal document promises delivery by either point. A message
  that arrives late lands in the next `await`. The trap behind it is general: never hand Metal a
  ctypes `Runtime.Block` whose body is OCaml. The old `add_log_handler` did, and its block literal
  lived in ctypes-managed memory that Metal kept pointing at (a global-flagged block's
  `Block_copy` returns the same pointer), so the next log message after a GC reused that memory
  crashed the host (SEGV, bus error, TRAP; 6 of 6 under GC churn). ocaml-metal's other block users
  (`add_completed_handler`, `notify_listener`, the no-copy buffer deallocator) share the shape, and
  OCANNL uses none of them. `test/operations/metal_routine_log_delivery` pins delivery: exactly one
  copy of each store line by the next sync, on cc and Metal.

## Choosing a backend-query seam

The gh-ocannl-822 census sorted every production test that inferred a C-syntax capability by reading
`backend_name` as a string. Use the same three-way question for the next site: a stable backend
capability belongs on the backend's queryable codegen-capability record (like `hardware_limits`); a
rendering or scheduling decision belongs in the routine/report census that records what actually
happened; only a test of the emitted dialect itself should branch on the dialect name. Default an
unclear site to the decision census, which is stronger than predicting a decision from
configuration.

| Bin | Production test | Fact being read and chosen seam |
| --- | --- | --- |
| capability | `accum_width.ml` | Resolved bf16 accumulator width; query `codegen_capabilities.accum_prec`. |
| capability | `float_literal_forms.ml` | Whether f64 storage can reach this dialect; query `codegen_capabilities.supports_f64`. |
| capability | `hardware_warp_shuffle.ml` | CPU family uses `Schedule.backend_is_cpu`; resolved bf16 accumulator width uses `codegen_capabilities.accum_prec`. |
| capability | `reduction_forms.ml` | Resolved accumulator residency for each storage precision; query `codegen_capabilities.accum_prec`, after checking it against the test's independent exact-backend policy table (the table is an oracle, not a capability access path). |
| capability | `schedule_batched_mma.ml` | Whether the device advertises the uniform-bf16 tile the tensorized execution leg needs; query `Backend_intf.advertises_mma_format`, and report the leg skipped when it answers no. |
| capability | `schedule_conv_gemm.ml` | Which typed MMA formats the device advertises; derive the f32/tf32 legs with `Backend_intf.advertises_mma_format`, never an open-coded `mma_format_tiles` scan (which is easy to write with the accumulator dropped from the key). |
| capability | `schedule_contraction_nest.ml` | Same uniform-bf16 gate as `schedule_batched_mma.ml`, on the multi-axis site's tensorized leg; query `Backend_intf.advertises_mma_format`. Its intrinsic census and shared-memory token are separate checks below. |
| capability | `schedule_mma_matmul.ml` | The tf32-only leg; derive it from the advertised tf32 format via `Backend_intf.advertises_mma_format` rather than the CUDA spelling. Other sites in this file are dialect checks below. |
| capability | `schedule_pad.ml` | Whether the padded extents fit the advertised MMA tile; compare `hardware_limits.mma_tile` with the extents, then check the routine census. |
| capability | `schedule_pipelined_matmul.ml` | Whether asynchronous staging can render; query `codegen_capabilities.asynchronous_staging_copy`. Its declaration and barrier tokens remain dialect checks below. |
| capability | `test_fp8_codec_parity.ml` | Whether f64 storage can reach this dialect; query `codegen_capabilities.supports_f64`. |
| decision outcome | `autotune_fission_sketch.ml` | Whether tensorized candidates reached candidate compilation; read `Autotune.report.mma_candidates`, or its `fiss_mma_candidates` subcounter for the per-segment family, not a backend-specific seed-count floor. |
| decision outcome | `schedule_batched_mma.ml` | Whether each sampled candidate rendered a tensor-core intrinsic; read `Context.routine.mma`. |
| decision outcome | `schedule_contraction_nest.ml` | Whether each sampled tensorized contraction rendered an intrinsic; read `Context.routine.mma`. Its shared-memory token stays a dialect check below. |
| dialect identity | `hardware_axes_parity.ml` | The test pins MSL `gid`/`lid`, CUDA/HIP `blockIdx`/`threadIdx`, and C serial-loop spellings. |
| dialect identity | `hardware_workgroup_reduce.ml` | The test pins MSL `threadgroup`/`threadgroup_barrier` versus CUDA/HIP `__shared__`/`__syncthreads`. |
| dialect identity | `schedule_batch_grid.ml` | The test pins the emitted folded batch-axis register, MSL `gid.z` versus CUDA/HIP `blockIdx.z`. |
| dialect identity | `schedule_contraction_nest.ml` | The source assertion pins MSL `threadgroup` versus CUDA/HIP `__shared__`; the intrinsic decision is censused above. |
| dialect identity | `schedule_epilogue_fusion.ml` | The test pins per-dialect fragment store and fused-epilogue ordering in emitted source. |
| dialect identity | `schedule_ldmatrix_matmul.ml` | The test deliberately pins CUDA inline-PTX `ldmatrix` spellings and the non-CUDA decline form, cross-checked with the MMA census. |
| dialect identity | `schedule_mma_matmul.ml` | The remaining sites pin CUDA WMMA/PTX, HIP rocWMMA, MSL simdgroup, and their documented hardware-specific tolerances. |
| dialect identity | `schedule_pipelined_matmul.ml` | The test counts MSL versus CUDA/HIP shared-array and barrier tokens; the asynchronous-copy capability is queried above. |
| dialect identity | `schedule_register_matmul.ml` | The test pins MSL `threadgroup` versus CUDA/HIP `__shared__` declarations. |
| dialect identity | `schedule_smem_matmul.ml` | The test pins MSL shared declarations/barriers versus CUDA/HIP spellings. |
| dialect identity | `schedule_swizzle_matmul.ml` | The test pins the MSL and CUDA/HIP shared declarations and intrinsic names around swizzled tiles. |

- A backend's `C_syntax_config` binds what it inherits at `include Pure_C_config` time, and that has
  bitten three ways: emission code defined ABOVE the backend's `typ_of_prec` override captures
  Pure_C's C spelling (half renders as `HALF_T`); a module-level function whose name matches a config
  field is shadowed by the include (hence `cc_backend.ml`'s `_setting` suffix convention); and
  overriding one member of a paired default (`compute_prec` without `accum_prec`) silently keeps the
  other's default pairing. Define overrides before the code that reads them, and restate both halves
  of a pair.

- **`dispatch_apply` traps at high fork/join rates on macOS, and the defect is the platform's**
  (gh-ocannl-870). The cc backend renders pool-backed `Grid` loops with `dispatch_apply` wherever
  `cc_parallel_grid` probes to `dispatch`, which is every Apple box. Past a few million calls the
  process dies with `SIGTRAP` / `EXC_BREAKPOINT` and
  `BUG IN CLIENT OF LIBMALLOC: memory corruption of free block`, faulting inside
  `_xzm_xzone_malloc_freelist_outlined` under `_dispatch_calloc_typed` under `dispatch_apply` --
  with a generated kernel in the frame below, which is what makes it read as a codegen or buffer
  lifetime bug. It is not: `benchmarks/runners/ocannl/dispatch_apply_stress.c` reproduces the
  identical signature from a plain C loop containing no OCANNL code, 10 of 40 runs at five million
  applies of extent 6 (M4 Max, macOS 26.6.2 build 25G83, 2026-09-05).
- The exposure is the CALL RATE, not any one kernel: a `Grid` loop nested under a serial loop is one
  fork/join per outer iteration, so an autotune candidate whose parallel loop has extent 6 under a
  serial 128 pays 128 of them per launch, and the queued objective launches each candidate
  thousands of times. `test/operations/autotune_split_reduce`'s bias-gradient arm is pinned to
  `Autotune.Isolated` for exactly this reason, and `--ocannl_cc_parallel_grid=none` is the control
  that separates the platform trap from anything of OCANNL's: 5 of 10 runs trapped with `dispatch`
  rendering against 0 of 10 interleaved runs with `none`.
- Two plausible OCANNL-side causes were refuted before the platform was, and re-refuting them costs
  a suite run each. Unloading kernels is not it: with the `Gc.finalise` `Dl.dlclose` in
  `cc_backend.ml`'s `c_compile_and_load` disabled, 5 of 8 runs still trapped. Nor is the trapping
  kernel's indexing: under `--ocannl_output_dlls_in_build_directory=true` the cc backend keeps its
  compiled `.so` under the name the crash report prints (by default it unlinks the file once
  loaded, next bullet), so `objdump -d --disassemble-symbols=_<routine>` on that exact file shows
  what ran -- here a guarded `p[chunk] += 1.0f` over six cells of a six-cell buffer. Reach for that
  disassembly rather than for the `build_files/` source, whose same-named copy is overwritten by the
  next candidate; the kept library is not, since its name carries the pid and the run id.
- A cc compilation leaves nothing in the system temp directory on any path, success and compiler
  rejection included (gh-ocannl-1197, `test/operations/cc_temp_ownership`): its temporary source,
  log and library are removed, and what a setting asks for is kept in `build_files/<exe>/`
  instead. Library output (`output_dlls_in_build_directory`) and debug files
  (`output_debug_files_in_build_directory`) alike keep the library there, as
  `<kernel>_pid<P>_n<nonce>_run<N>.so` (`Utils.kept_library_file`, gh-ocannl-1247, gh-ocannl-1335);
  under debug files the compiler reads a private copy of the source under the same name with `.c`
  (`Utils.owned_artifact_file`), removed once the library loads and kept on any failure as the
  source the report names -- the informational `build_files/<exe>/<kernel>.c` can be rewritten
  mid-compile by a concurrent same-named kernel. The pid, because concurrent processes of one
  executable share that directory -- the OS id (`Utils.os_process_id`), never `Unix.getpid`, which
  on Windows answers a process-local handle -- and the per-process random nonce because an OS id is
  not unique across PID namespaces. The kernel name in such a name, and in the temporary source's,
  is cut within `Utils.library_kernel_name_limit` bytes at a character boundary and suffixed with a
  digest of the whole name, so two long names sharing their opening bytes stay apart
  (`Utils.library_kernel_name`): a routine name has no length limit, a file-name component does,
  and clang names its own temporaries after the source. Tests list kept libraries with
  `Test_utils.Generated.kept_libraries` (`test/operations/cc_kept_library`). The library is unlinked right after `dlopen`: the kernel
  runs from the mapping, which POSIX keeps alive past the unlink. Windows refuses to delete a
  mapped DLL, so a failed removal is retried after the finalizer's `dlclose`, never before it. OCaml
  runs no finalisers at exit and the OpenMP arm never unloads, so on Windows only routines
  collected mid-run (autotune candidates, mostly) get their library removed; the libraries of
  routines alive at exit, and of every OpenMP kernel, stay past exit. So a temporary library's name
  records its owner too, `ocannl_cc_<kernel>_pid<P>_n<nonce>_run_id_<N>_<random>` (the marker
  keeps another program's same-shaped file in the shared directory out of the sweep)
  (`Utils.temp_library_prefix`/
  `parse_temp_library_name`), and every process's startup (the `Utils` cleanup block, ungated) removes the temp directory's libraries whose owner is gone, with
  the same liveness test that spares the kept libraries, once they are older than
  `Utils.temp_library_sweep_min_age` (gh-ocannl-1268, `test/operations/cc_temp_library_sweep`,
  whose end-to-end leg is gated to Windows). The age guard is what makes an ungated sweep of a
  SHARED directory safe: liveness answers only within this PID namespace, and a container's
  process may even report this process's own id (two PID-1 entrypoints), but a library is at risk
  only between its link and its `dlopen`. Unlinking a
  still-mapped library frees its name, and a loader answers a `dlopen` of a path it has mapped with that mapping without reading the file, which is why every library path
  carries the process-unique run id: drop it and a later `temp_file` landing on the freed name runs
  the earlier kernel. Only a FAILED removal is retried (a path already gone counts as removed),
  because a freed name may already belong to another process. Profilers and crash symbolication
  that read a kernel's `.so` after the process (`perf report`, macOS crash reports) need
  `output_dlls_in_build_directory=true` too.
- The startup cleanup of `build_files/<exe>/` and `log_files/<exe>/` cannot tell a running sibling's
  ownerless artifacts (debug sources, logs) from a finished run's, so it asks a lock
  (gh-ocannl-1335): a process takes a SHARED `Unix.lockf` on `<base>/<exe>.ocannl-lock` the first
  time it names a path to write there (`Utils.build_file`, `Utils.diagn_log_file`, so in practice
  every process holds the `log_files` one), for its lifetime -- one per lock file's absolute path,
  so a process that changes directory holds each directory's -- and a starting process tries it
  EXCLUSIVELY without waiting. Held means a sibling runs, and the cleanup then removes NOTHING: it
  cannot tell the sibling's ownerless artifacts from a finished run's, and not even a gone-looking
  owner-named one is safe, since the lock reaches across PID namespaces where liveness
  (`Utils.process_alive`) does not; the stale ones go at the next startup that finds no sibling.
  Without a sibling, the kept libraries of a RUNNING process (`Utils.parse_kept_library_name`; a
  source copy is never read for an owner, since a routine may be NAMED owner-shaped) still stay, and the root is emptied, never
  removed, since a sibling may hold a path inside it it has yet to write. A lock dies with its
  process, which is why it is not a marker file. Two traps: POSIX drops all of a process's locks on
  a file when ANY descriptor of it closes, so nothing else may open the lock file (it sits beside
  the subdirectory, out of every listing; under the flat layout the cleanup skips it by its
  suffix, and it is created 0666 so every account sharing the directory can lock it -- one that
  cannot open it for writing reads it as held); and two dune rules running ONE executable share its subdirectory and so sweep each
  other, which `cc_kept_library`'s rules serialize with a dune lock. POSIX does not carry record
  locks into a forked child, so the table of held locks is keyed by OS pid too and a child
  re-takes them; that, and holding each directory's lock across `chdir`, are pinned by
  `test/operations/artifacts_lock_lifecycle` (gh-ocannl-1462), which observes every lock from a
  separate process (record locks never conflict within one). A long-running holder blocks every
  later cleanup: `force_clean_up_artifacts_on_startup=true` empties a held directory anyway and
  names a holder pid on stderr (`fcntl(F_GETLK)`, `Utils.lock_holder_pid`: one of possibly several
  shared holders; unknown on Windows). It overrides the lock's gate only: the lock file stays
  (unlinking it would let later processes lock different inodes) and so do a running owner's
  owner-named artifacts.

- A failed cc kernel compile blames the generated code only after a trivial kernel compiles under
  the same command, flags and link flags (gh-ocannl-1350, `Cc_backend.toolchain_fault`,
  `test/operations/cc_toolchain_attribution`). When it does not, the failure is a
  `Backend_rejected` at `Schedule_outcome.toolchain_stage` with severity `Configuration`:
  uncontainable, since a toolchain that compiles nothing rejects every candidate alike, and
  rendered as `Utils.User_error`, never with `C_syntax.codegen_bug_claim`, and
  `C_syntax.attribute_routine_name_failure` passes it through untouched (a configuration-wide
  `-Werror` echoes the routine's declaration, which would otherwise read as a name collision). It names the setting
  whose flag fails on its own (`Cc_backend.kernel_compiler_flag_settings` pairs each flag with its
  setting), the compiler command when even the bare command fails, and every flag setting when
  only their combination does. Probing rather than matching compiler stderr keeps it independent of
  the compiler's wording; it runs only on a failure, once per process per configuration.

- Every pool-parallel CPU `Grid` launch, top-level or repeated, requires at least
  `cc_grid_min_updates` (default 16,384) estimated scalar updates per native dispatch
  (`C_syntax.grid_update_count`; repeated launches since gh-ocannl-933, top-level ones since
  gh-ocannl-1564, where `narrow_gebp_bench`'s `packmma_par` at n <= 32 lost 10x to serial on rog).
  The estimate multiplies only the Grid and its body extents, includes scalar `Local_scope` work,
  charges a `Tile_mma` its MACs over `C_syntax.tile_mma_macs_per_update` (16: it renders as a
  vector micro-kernel; MACs at par let a 2 x 16x32x32 launch through), and saturates before
  overflow; enclosing serial iterations cannot amortize their own fork/joins. The floor is fixed
  across machines (emitted code must not depend on the pool), is applied after safety checks, and
  does not claim all larger regions are profitable. Default-schedule Grids already carry
  `cpu_schedule_min_parallel` (16,384) parallel iterations, so the floor bites on hand-built and
  tuned schedules and the tuner's `min_parallel:1` presets. It is read per compile: a test that pins
  which Grid loops CAN render on the pool (eligibility, binding legality, fine-grained stress)
  states `cc_grid_min_updates=0` for its compiles (`cpu_parallel`'s `top_six_nofloor`,
  `cpu_scratch_matmul_grid`, `schedule_pack_mma_matmul`); `grid_launch_floor` pins the floor itself.

- **Whether a Grid loop ran on the pool is a field of the routine: `Context.routine.grid`**
  (gh-ocannl-1668). A declined pool-parallel Grid renders a correct serial loop, so neither the
  values nor (without `schedule_log_declines`) anything else said so: `narrow_gebp_bench`'s
  `packmma_par` at n=512 was timed slower than serial `packmma` under its parallel label because
  its per-chunk tiles were over `cc_grid_private_bytes_cap` (staging#1457). `collect_parallel_grid`
  records every outermost Grid loop of two or more iterations as `Grid_pool { private_bytes }` or
  `Grid_serial why` (`Grid_private_over_cap { private_bytes; cap }`, `Grid_below_min_updates`, the
  sharing/privatization rules, an offset range), bracketed by `C_syntax.with_grid_census` inside
  `Context.compile` beside the mma census. `pool = false` means no entry was recorded: no pool
  backend, one chunk, or `debug_log_from_routines`; on a GPU a Grid is a hardware grid. The cap is
  checked once, on the whole per-chunk footprint (every privatized local, the C fragment included),
  so the bytes are deterministic. `grid_summary_string` is the bench line (`narrow_gebp_bench` and
  `schedule_bench` print it beside the mma label; `narrow_gebp_bench` warns when `packmma_par`
  rendered serially). `tile_mma_declines` (cap, both sides) and `grid_launch_floor` (floor) pin it.

- `C_syntax.render_ctx` owns the state of one emission, including volatility observers, zero seeds,
  thread-binding analysis and codegen-minted accumulator scopes (gh-ocannl-769). `create_render_ctx`
  requires an optimized procedure; body-only `compile_main` callers pass that context explicitly.
  The loop strategies are siblings of `pp_ll`; `derive_kparams` preserves alias-candidate restrictions
  independently of emission. Census brackets are domain-local and additive within one domain.
  `arrayjit/test/test_render_context.ml` injects nested compilation and an emission exception;
  its shared-zero-state negative control demonstrates the old lost-rezero failure.

- `Cuda_like_config.Make` owns the CUDA/HIP scalar operator table (gh-ocannl-770); its live
  `Cuda` and `Hip` descriptors retain bf16 arithmetic/ReLU, approximate-tanh and fp8 narrowing
  differences. Vendor conversion overloads, accumulator residency and MMA hooks remain local.
  `C_syntax.Compile_driver` owns single/batch naming, binding order and source assembly; Metal
  retains source until link time, while cc/CUDA/HIP compile once per batch. Compiler callbacks
  keep artifact ownership and typed exceptions; `with_compiler_options` attaches effective flags.
- Conditional vendor headers and builtin selection share `C_syntax.Source_tokens` (gh-ocannl-770).
  Qualified markers match across whitespace/comments, while longer identifiers and quoted literals
  cannot activate a header. Direct CUDA/HIP compiler seams prepare headers idempotently too, so
  the HIP compile-failure probe still exercises the production options and exception constructor.
- Every `hipEventRecord` costs one `Signal 0x... time stamps may be invalid.` on the C runtime's
  stderr wherever ROCr was built with assertions on — the Ubuntu 26.04 `libhsa-runtime64`
  7.1.0+dfsg-0ubuntu9 is (it imports `__assert_fail`, so ROCr's `debug_print` is live), and the
  whole native-Linux fleet shares that build. ROCclr's `hip::EventMarker` sets
  `profilingInfo_.enabled_ = true` for every event marker however the event was created, so
  `hipEventDisableTiming` does NOT avoid it: ROCr then reads back dispatch timestamps the hardware
  never wrote for a barrier packet and reports the zero pair. Nothing in `hip_backend.ml` or
  `Delimited_event` can avoid it and no runtime switch reaches it — `GPU_FORCE_QUEUE_PROFILING=0`
  and `AMD_DIRECT_DISPATCH=0` change nothing, `=1` makes it worse — which is why the answer lives
  at the stream seam instead (`Utils.c_stderr_detached`, gh-ocannl-1031). A 30-line pure-HIP
  program reproduces it in one line per recorded event, with none when no event is recorded; reach
  for that shape before suspecting OCANNL's event discipline. It is an APU symptom: on gfx1151
  (minix) one autotuning test prints 268,172 (Dune kept ~2,000 of them), while the discrete card on tuf, with the identical
  package, writes valid timestamps and prints none. Detaching kept it out of goldens but not off the
  run log, where Dune truncates a long action output from the middle and the test's own SKIPPED
  lines went with it, so since gh-ocannl-1370 the HIP backend, when `get_device` opens a device (before its stream), also forces
  `Utils.c_stderr_noise_collapsed` (`collapse_foreign_stderr_noise`, default true): the C stream
  becomes a cookie stream (`fopencookie`/`funopen`) over the same descriptor that writes the first
  such line, one note, and a count from an exit-time destructor, and writes every other byte
  through at once. The destructor runs from `_dl_fini`, after every `atexit` handler including the
  stream teardown below, which is when ROCclr prints most of them, but before the shared libraries'
  own destructors: a line printed there is written through, uncounted. `fileno(stderr)` reads -1 once it
  is installed, and a cookie stream cannot be wide-oriented, so an already wide-oriented C stderr
  is left alone; an `fclose(stderr)` reports the count at the close (the close callback), so the
  destructor never touches a freed FILE. Only the HIP backend installs it, and only on opening a
  device, so a process that probed HIP and fell back keeps its C stream.
- `Warning: Resource leak detected by SharedSignalPool, 77 Signals leaked.` on a hip run exit is
  the per-device HIP stream never being destroyed, not leaked events: the same pure-HIP probe
  reports exactly 77 whatever the iteration count when it skips `hipStreamDestroy`, and a count
  equal to the number of undestroyed events when it skips `hipEventDestroy` instead. OCANNL shows
  the fixed 77, so its `Delimited_event` release discipline holds and the gap was process-exit
  teardown (`Gc.finalise finalize_device` does not run at exit, and would not destroy the stream if
  it did). Since gh-ocannl-1036 `hip_backend.ml` registers, at module initialization (so every
  handler a program registers runs before it), an `at_exit` that destroys each device's stream
  through `Utils.bounded_exit_teardown`: it polls every device's `H.Stream.is_ready` under ONE
  shared bound of `exit_stream_teardown_timeout` seconds and calls `H.Stream.destroy` (which
  synchronizes unboundedly) only once idle, so hung devices cannot hold the process. `at_exit`
  runs on an uncaught exception too, which is why the bound, not an exit-status test, does that
  job. A reappearing 77 means the teardown was skipped: `0` in the config, or a busy stream at exit
  (stderr says `still busy`). An instantiated graph that outlives the process brings it back too
  (`seeded_default` on minix printed it with `gpu_graph_capture` on and not off, gh-ocannl-1370),
  so the same teardown first destroys the `hipGraphExec`s graph-captured routines still cache
  (gh-ocannl-1445, backend-memory.md). CUDA stays without it: the driver reports nothing and cudajit
  (0.8.0) exposes no eager `Stream.destroy`. `test/operations/exit_stream_teardown` pins both.
