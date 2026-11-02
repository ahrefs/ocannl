(** OCANNL's C-family code generator, shared by the cc, CUDA, HIP and Metal backends: renders
    lowered routines as C, CUDA, HIP or MSL source through a backend's {!C_syntax_config}, and
    reports per-compile censuses of the renderings it chose. *)

open Base

type t = PPrint.document
(** A rendered C-family source document. *)

val main_logic_marker : string
(** The comment opening every kernel's main logic: its statements, or an outlined kernel's part
    calls. *)

val warp_shuffle_stages : width:Int.t -> Int.t list
(** The ordered XOR lane masks of one warp-shuffle reduction over the power-of-two [width]. *)

val integer_power_operands : Float.t -> int64 * int64 * Int.t * bool
(** An [ocannl_powi_*] exponent's operands: [|exponent|] as [(lo, hi)] words, a [shift], its sign.
*)

val pool_slot_is_64 : unit -> bool
(** Whether Metal's pooled slot table holds 64-bit entries, as it does under [large_models]. *)

val pool_slot_msl_typ : unit -> string
(** The MSL element type of the pooled slot table: [ulong] or [uint], per {!pool_slot_is_64}. *)

val routine_stack_locals_bytes : Int.t
(** Stack budget for a routine's function-scope [Local] arrays; heap-capable backends spill the
    rest. *)

val chunk_stack_locals_bytes : Int.t
(** Stack budget for one pool-parallel [Grid] chunk's privatized tiles; past it they go to the heap.
*)

val per_chunk_private_bytes_cap : Int.t lazy_t
(** The resolved [cc_grid_private_bytes_cap]: a [Grid] loop needing more per chunk stays serial. *)

val default_grid_min_updates : int
(** The default of [cc_grid_min_updates], the work floor of a pool-parallel [Grid] launch. *)

val grid_min_updates : unit -> Int.t
(** The resolved [cc_grid_min_updates]: a launch estimated below it renders serially; [0] admits
    all. *)

val tile_mma_macs_per_update : int
(** How many [Tile_mma] multiply-accumulates count as one scalar update ({!Cost_model}'s constant).
*)

val grid_update_count : ?cap:int -> Low_level.t -> int
(** Estimated scalar updates of one launch of a [Grid] body, saturating at [cap]. *)

(** How one [Tile_mma] statement rendered: intrinsics, a register tile, or the scalar fallback. *)
type mma_rendering =
  | Mma_intrinsics
  | Mma_intrinsics_ldmatrix
      (** The intrinsic arms fed by warp-cooperative [ldmatrix] loads over a [Swizzle_b128] staged
          tile (gh-ocannl-481 item 3) instead of per-lane gathers. Recorded distinctly because the
          gh-476 sweep must be able to pin which of the two load paths a timing measured. *)
  | Mma_register_tiled
  | Mma_scalar_fallback
[@@deriving sexp_of, equal]

val mma_census : unit -> (string * mma_rendering) list ref
(** This domain's [Tile_mma] census: (kernel name, rendering) entries, most recent first. *)

val mma_scope_census : unit -> string list ref
(** This domain's census of persistent-fragment scopes the fragment hook rendered, one kernel name
    each. *)

val is_tensorized_rendering : mma_rendering -> bool
(** Whether a rendering is a real tensor-core or SIMD-register-tile emission, not the scalar
    fallback. *)

(** Whether a routine's [Tile_mma] statements tensorized: none emitted, any honoured, or all fell
    back. *)
type tensorization =
  | Not_requested
      (** Codegen emitted no [Tile_mma] statement at all: nothing about this routine claims tensor
          cores, and a timing of it is not a tensorized timing. *)
  | Tensorized
      (** At least one [Tile_mma] statement rendered to a real tensor-core or SIMD-register-tile
          emission ({!Mma_intrinsics}, {!Mma_intrinsics_ldmatrix}, {!Mma_register_tiled}). The C
          backends' register-tiled rendering counts: it is the SIMD-tile emission those backends
          have, and the point of the label is fallback-vs-not, not which ISA. *)
  | Scalar_fallback
      (** [Tile_mma] statements were emitted and {e every} one of them declined to the lane-0 scalar
          fallback. A timing labeled tensorized in this state measured scalar code. *)
[@@deriving sexp_of, equal]

val tensorization_name : tensorization -> string
(** The label a census line prints for a {!tensorization} state. *)

type mma_summary = {
  renderings : (string * mma_rendering) list;
      (** The census entries of one compile, in emission order (kernel name, rendering). Fissioned
          segments of one routine contribute their kernels to the same summary. *)
  tensorization : tensorization;
  statements : int;  (** [Tile_mma] statements emitted: [List.length renderings]. *)
  scalar_fallbacks : int;  (** Of those, how many rendered as the lane-0 scalar fallback. *)
  fragment_scopes : int;
      (** gh-ocannl-1707: persistent-fragment regions ([Low_level.fragment_region]) the backend's
          fragment hook rendered, the accumulator register-resident across the whole reduction. Not
          counted: a region whose hook declined (its [Tile_mma]s alias the target per statement, or
          fall back), and a pad-masked contraction's fragment, which is never marked as one. *)
}
[@@deriving sexp_of]
(** What a compile's {!mma_census} says about the routine it produced. *)

val summarize_census : ?fragment_scopes:int -> (string * mma_rendering) List.t -> mma_summary
(** Summarize census entries and the [fragment_scopes] count into an {!mma_summary}. *)

val empty_mma_summary : mma_summary
(** The summary of a compile that emitted no [Tile_mma] statement. *)

val merge_mma_summaries : mma_summary List.t -> mma_summary
(** One summary of several compiles' census entries and fragment scopes together. *)

val mma_summary_string : mma_summary -> string
(** The one-line census a timing report prints: the {!tensorization} label, then per-rendering
    counts. *)

val with_census : (unit -> 'a) -> 'a * mma_summary
(** Run [f] collecting the [Tile_mma] census, nesting additively; return its result and summary. *)

(** Why a site's width decision declined a base the peel reached: {!Low_level.accum_skip}
    re-exported. *)
type peel_skip = Low_level.accum_skip = Skip_debug_logging | Skip_dead_level | Skip_accum_pinned
[@@deriving sexp_of, equal]

type peel_verdict = {
  levels : int;
      (** Loop levels the localized scope spans: the level being rendered, plus those
          {!Low_level.peel_accum_nest} took below it. One for a reduction localized at its own
          level; two where the accumulated cell let the peel swallow the enclosing level too. *)
  guards : Low_level.peel_guard_verdict list;
      (** The verdict of each [If] among them, outermost first. *)
}
[@@deriving sexp_of, equal]
(** What a localizing site decided: the loop levels its scope spans and its guards' verdicts. *)

(** One reduction site's peel outcome: localized, ceded to a wide rendering, refused, or not
    attempted. *)
type peel_site =
  | Peel_localized of peel_verdict  (** The site localized, deciding this. *)
  | Peel_ceded of peel_verdict
      (** The width decision reached a base the serial rendering would have localized, deciding this
          — and the site rendered a form that holds the accumulator at that same residency somewhere
          other than a minted scope: the warp-shuffle tree or the SIMD accumulator grid
          (gh-ocannl-754). Not {!Peel_localized}, since no scope was minted; not a skip, since the
          width question was asked and answered wide. A test can therefore pin that the shuffle or
          the grid took its width from the one shared decision rather than from a recognizer of its
          own. *)
  | Peel_refused of Low_level.peel_refusal  (** [Low_level.peel_accum_nest] refused, with why. *)
  | Peel_not_attempted of peel_skip
      (** The peel reached a base, and localization was not attempted, for this reason. Every
          rendering of the level then keeps the accumulator at storage width (gh-ocannl-754). *)
[@@deriving sexp_of, equal]

val peel_census_enabled : unit -> bool ref
(** This domain's switch for collecting the {!peel_census}. *)

val peel_census : unit -> (string * peel_site) list ref
(** This domain's reduction-peel census: (kernel name, site) entries, most recent first. *)

val is_localized_peel : peel_site -> bool
(** Whether a site localized its accumulator into a minted scope. *)

type peel_summary = {
  sites : (string * peel_site) list;
      (** The census entries of one compile, in emission order (kernel name, site). Fissioned
          segments of one routine contribute their kernels to the same summary. *)
  localized : int;
  ceded : int;
      (** Sites whose width decision was wide and whose rendering held the accumulator outside a
          scope (gh-ocannl-754). *)
  declined : int;  (** Sites that did not localize or cede, refused and not-attempted together. *)
}
[@@deriving sexp_of]
(** What a compile's {!peel_census} says about the routine it produced. *)

(** How one reduction level's accumulator is held, decided once for every rendering of that level.
*)
type accum_width =
  | Accum_not_a_nest of Low_level.peel_refusal
      (** {!Low_level.peel_accum_nest} reached no accumulation base under the level. *)
  | Accum_base of {
      tn : Tnode.t;
      idcs : Indexing.axis_index array;
      base : [ `Update of Low_level.scalar_t | `Scope of Low_level.scope_id * Low_level.t list ];
      debug : string;
      rebuild : Low_level.t -> Low_level.t;
      verdict : peel_verdict;
          (** What the peel decided, the level being rendered counted in ([levels >= 1]). *)
      pinned : peel_skip option;
          (** [Some why]: the serial rendering keeps the accumulator in its storage cell, narrowing
              every step, so no rendering may hold it wider. [None]: the serial rendering localizes
              it at the residency, and so may any other. *)
    }

type statement_accum = {
  sa_tn : Tnode.t;
  sa_idcs : Indexing.axis_index array;
  sa_op : Ops.binop;
  sa_contrib : Low_level.scalar_t;
  sa_verdict : peel_verdict;
  sa_pinned : peel_skip option;
}
(** An {!accum_width} base that is a single accumulation statement at the level itself. *)

val peel_summary_string : peel_summary -> string
(** The one-line peel census: how many sites localized, ceded or declined, and the distinct
    verdicts. *)

val with_peel_census : (unit -> 'a) -> 'a * peel_summary
(** Run [f] collecting the {!peel_census}, nesting additively; return its result and summary. *)

(** One serial accumulation's volatility-census entry, by the emitted identifier. *)
type volatility_site =
  | Volatile_accumulation_reads of string
      (** A reduction-shaped scope local for which the backend selected expression-level [volatile]
          pointer casts on any device reads in its accumulating loop, by the local's emitted
          identifier: the localized accumulation form (gh-ocannl-731, gh-ocannl-820). *)
  | Plain_accumulator of string
      (** A reduction-shaped scope local whose reads stay plain — either this backend does not
          request the workaround, or no materialized device read was emitted in its accumulating
          loop. The accumulator itself stays register-resident in both cases. *)
  | Volatile_rmw_reads of string
      (** A device-memory read-modify-write at an address invariant across an enclosing serial loop,
          whose device reads use expression-level [volatile] pointer casts. *)
[@@deriving sexp_of, equal]

type volatility_summary = {
  entries : (string * volatility_site) list;
      (** The census entries of one compile, in emission order (kernel name, site). Fissioned
          segments of one routine contribute their kernels to the same summary. *)
  requested : bool;
      (** Whether the backend this compile ran on asks for the workaround
          ({!C_syntax_config.volatile_serial_accumulation}). [false] makes every localized site a
          {!Plain_accumulator} and suppresses the device-memory RMW arm entirely. *)
  volatile_accumulations : int;
  plain_accumulations : int;
  volatile_rmw_reads : int;
}
[@@deriving sexp_of]
(** What a compile's volatility census says about the routine it produced. *)

val volatility_census_enabled : unit -> bool ref
(** This domain's switch for collecting the {!volatility_census}. *)

val volatility_census : unit -> (string * volatility_site) list ref
(** This domain's volatility census: (kernel name, site) entries, most recent first. *)

val volatility_requested : unit -> bool ref
(** Whether the backend compiling on this domain requests the volatile-read workaround. *)

val volatility_summary_string : volatility_summary -> string
(** The one-line volatility census: accumulations, and how many use the volatile-read workaround. *)

val with_volatility_census : (unit -> 'a) -> 'a * volatility_summary
(** Run [f] collecting the volatility census, nesting additively; return its result and summary. *)

(** Why an outermost [Grid] loop rendered serially on a pool-backed backend. *)
type grid_serial_reason =
  | Grid_opaque_body  (** An opaque statement or a barrier sits under the loop body. *)
  | Grid_scope_local_escapes
      (** A scope local is set under the loop but declared outside it, so its function-scope storage
          would race across chunks. *)
  | Grid_local_accessed_outside of string
      (** This local is written grid-variantly (it fails the shared rule) and is also accessed
          outside the loop body (it fails the privatization rule). *)
  | Grid_local_not_covering of string
      (** This local fails the shared rule, and its first access per iteration is not a standalone
          covering write (it fails the privatization rule). *)
  | Grid_private_over_cap of { private_bytes : int; cap : int }
      (** Every local the loop needs privatized could be, but their combined per-chunk footprint
          [private_bytes] exceeds [cap], the resolved [cc_grid_private_bytes_cap]
          ({!per_chunk_private_bytes_cap}). *)
  | Grid_below_min_updates of { updates : int; floor : int }
      (** The launch's estimated scalar updates ({!grid_update_count}, saturating at [floor]) fall
          below [floor], the resolved [cc_grid_min_updates]. *)
  | Grid_offset_range of { from_ : int; to_ : int }
      (** The loop does not start at 0, which the chunked rendering assumes. *)
[@@deriving sexp_of, equal]

(** One outermost [Grid] loop's decision: rendered on the pool, or serial and why. *)
type grid_site =
  | Grid_pool of { private_bytes : int }
      (** Rendered on the pool, each chunk privatizing [private_bytes] of tiles (0 when none). *)
  | Grid_serial of grid_serial_reason
[@@deriving sexp_of, equal]

type grid_entry = { kernel : string; loop : string; site : grid_site } [@@deriving sexp_of]
(** One outermost [Grid] loop: the kernel it is in, its loop index's identifier, the decision. *)

type grid_summary = {
  loops : grid_entry list;
      (** The census entries of one compile, in emission order. Fissioned segments of one routine
          contribute their kernels to the same summary. *)
  pool : bool;
      (** Whether any kernel of this compile was rendered with pool-parallel [Grid] loops available
          at all: a pool-backed backend ([cc_parallel_grid] not [none]) with more than one chunk and
          no [debug_log_from_routines]. [false] means no entry was recorded, and on a GPU backend
          that a [Grid] loop is a hardware grid, not a serial loop. *)
  pool_rendered : int;
  serial : int;
}
[@@deriving sexp_of]
(** What a compile's Grid census says about the routine it produced. *)

val grid_summary_string : grid_summary -> string
(** The one-line Grid census: [Grid] loops rendered on the pool, and why each other one is serial.
*)

val with_grid_census : (unit -> 'a) -> 'a * grid_summary
(** Run [f] collecting the Grid census, nesting additively; return its result and summary. *)

type mma_space = [ `Device | `Shared | `Thread | `Fragment of string ]
(** The address space of a tile-MMA operand as the emission hooks see it. *)

type mma_layout = [ `Plain | `Swizzled_b128 ]
(** The physical layout of a tile-MMA operand's storage, which the emission hooks trust as given. *)

type mma_operand = PPrint.document * int * mma_space * mma_layout
(** A tile-MMA operand: tile-base pointer, leading-dimension stride, address space, layout. *)

type mma_source = int * mma_space * mma_layout
(** A tile-MMA input operand described but not addressed: stride, address space, layout. *)

type mma_emission = a_ptr:PPrint.document -> b_ptr:PPrint.document -> PPrint.document
(** The emitting half of an accepted {!C_syntax_config.mma_syntax} call, awaiting the a/b addresses.
*)

type async_copy_syntax = {
  ac_copy : dst:PPrint.document -> src:PPrint.document -> bytes:int -> PPrint.document;
      (** One element-sized asynchronous global→workgroup-shared copy statement; [dst] and [src] are
          element addresses ([&ident[offset]]) and [bytes] the element size — 4 or 8 today. The
          hardware also copies 16, but a 16-byte copy requires a 16-byte-aligned destination, and
          plain workgroup-shared declarations align to the element type only (4 for [uint4x32_t]) —
          so 16 stays out of the per-element eligibility until a rendering guarantees the alignment
          (Codex P2 on PR #317). The copy is byte-for-byte: eligibility (same storage precision on
          both sides, no value transformation) is the caller's check. *)
  ac_wait_all : string;
      (** Statement completing every asynchronous copy issued so far by the calling thread,
          committed or not (CUDA [cp.async.wait_all]). Cross-thread visibility still needs the
          workgroup barrier the caller emits right after. *)
}
(** Asynchronous global-to-workgroup-shared copies for staging software-pipelined tiles. *)

type select_blend =
  | Blend_truth of (PPrint.document -> PPrint.document -> PPrint.document -> PPrint.document)
  | Blend_over_locals of string
      (** How a backend renders a select evaluating both arms: a truth blend, or arms bound to
          locals. *)

type param_guard_dialect =
  | C_guards
  | Cxx_guards
      (** How a dialect spells a builtin's parameter guard: GNU C static assertion, or C++ braced
          init. *)

val blend_local_a : string
(** The local a [Blend_over_locals] blend binds its first arm to; tests find blends by it. *)

(** What a backend supplies to the C-family renderer: its dialect's spellings, capabilities and
    hooks. *)
module type C_syntax_config = sig
  val procs : Low_level.t array
  (** The low-level procedures this functor application will render: one per kernel of the
      compilation unit (a singleton for a plain compile, one entry per routine for a batch). The
      functor reads them for whole-unit analyses -- the identifier census, the scope-local verdicts
      -- while each kernel's own rendering goes through {!compile_proc}, which takes the full
      {!Low_level.optimized} record. *)

  val main_kernel_prefix : string
  val kernel_prep_line : string

  val outline_part_weight : unit -> int option
  (** gh-ocannl-1603: [Some w] when a kernel whose top-level statements weigh more than [16 * w]
      ({!Kernel_outline.groups}) is rendered OUTLINED -- its statements packed in order into parts
      of about [w] each, every part a [noinline] function the kernel calls in turn -- because a C
      compiler's cost grows faster than linearly in a function's size. [None] renders every kernel
      whole. Only a host C backend can outline: a part is an ordinary function call. *)

  val buffer_prefix : string
  val buffer_suffix : pos:int -> string
  val arg_int_prefix : string
  val loop_index_type : string
  val extra_args : string list
  val typ_of_prec : Ops.prec -> string

  val supports_f64 : bool
  (** Whether [typ_of_prec] accepts f64 tensor storage. Explicit so capability queries need not
      deliberately raise through a rendering function. *)

  val vec_typ_of_prec : length:int -> Ops.prec -> string
  val ident_blacklist : string list

  val vendor_routine_idents : string list
  (** File-scope names the backend's own vendor headers declare (or its compiler predefines) that a
      kernel DEFINITION cannot take, beyond the C floor {!c_stdlib_idents} already covers: global
      type names, macros, dialect keywords, and C-linkage functions (gh-ocannl-707). Constrains
      routine names only ({!C_syntax.kernel_ident}), for the reason {!c_stdlib_idents} does: a
      parameter or a local may shadow these, a file-scope definition may not. Names in the
      implementation's reserved namespace ([__x], [_X]) need no entry: {!implementation_reserved}
      closes that class by rule. [[]] for a dialect whose headers stop at the C floor. *)

  val node_ident_prefix : string
  (** Prepended to every identifier a tensor node takes in an emitted kernel -- a parameter, a
      pool-pointer local, a kernel-scope or shared array -- and to nothing else: not to the node's
      code name ({!Tnode.t.code_name}, what debug output and error messages show), not to the text
      of a routine log line, not to scope locals ([v<scope>_<name>] already carries a prefix)
      (gh-ocannl-1338).

      A GPU dialect's vendor headers declare global types ([float4], [uint4], [half2]) and macros
      ([M_PI_F], [MAXFLOAT], [CUDARTAPI]) outside the reserved namespace, and the kernel body spells
      the types: a parameter named [uint4] shadows the type for every later [device uint4*] of the
      kernel, and a node named after a macro is macro-expanded before it is declared. A fixed prefix
      closes that class by construction, where a blacklist would have to track every header the
      toolchain's runtime compiler pulls in, including built-in ones no file holds. [""] for cc,
      whose only headers are the C library's: their macros and the types the renderer spells are
      already in [ident_blacklist], and the compiler's predefined macros are closed by rule
      ({!implementation_reserved}, {!system_macro_idents}). *)

  val ptr_param_style : [ `Per_param | `Pooled of int ]
  (** How materialized in-context tensor nodes are passed to the kernel. [`Per_param] (the default,
      used by C and CUDA) emits one typed pointer parameter per node, whose host side binds
      [pool_base + offset] -- byte-identical to the pre-pooling codegen. [`Pooled n] (Metal) emits a
      fixed [n] byte-pointer pool parameters plus one (pool_index, byte_offset) slot table, and a
      kernel prologue that forms each node's typed pointer by casting (pools at slot.pool) + offset.
      This collapses O(num_nodes) buffer bindings to [n] + 1, which Metal needs to stay under its
      ~31 argument-buffer binding limit. *)

  val float_log_style : string
  (** Format specifier for printing floating point numbers in debug logs. *)

  val styled_log_arg : PPrint.document -> PPrint.document
  (** Function to convert potentially floating-point numeric values for logging. *)

  val uint4x32_log_lane : PPrint.document -> int -> PPrint.document
  (** Lane [i] (0 to 3) of a [uint4x32] value, as a [%u] routine-log argument. A [uint4x32] value is
      a struct ([uint4x32_t], C/CUDA/HIP) or a vector ([uint4], Metal), never a number, so neither
      {!float_log_style} nor {!styled_log_arg} applies to it: a log statement prints its four lanes
      as unsigned integers instead. Coupled to [typ_of_prec]'s spelling of [Uint4x32_prec]. *)

  val log_index_arg : PPrint.document -> string * PPrint.document
  (** How a value of {!loop_index_type} is passed to a log statement: the printf conversion
      specification to splice into the format string, and the argument document, cast where the
      backend's variadic logging call needs a different type than the loop index's own.

      A loop index is [int32_t]/[int] normally but 64-bit under [large_models], and the conversion
      has to track that width: passing a 64-bit argument to [%d] is a variadic type mismatch --
      undefined behaviour on the printf paths, a compile error on Metal's [os_log], and wrong digits
      for values past 32 bits either way.

      Coupled to [loop_index_type] and to [pp_log_statement]: a backend that spells the loop index
      as its own type, or logs through a call whose conversions are not C's, overrides this
      alongside. *)

  val ternop_syntax :
    Ops.prec ->
    Ops.ternop ->
    PPrint.document ->
    PPrint.document ->
    PPrint.document ->
    PPrint.document

  val binop_syntax : Ops.prec -> Ops.binop -> PPrint.document -> PPrint.document -> PPrint.document

  val builtin_param_guards : param_guard_dialect option
  (** The dialect of the builtin parameter guards a kernel's prelude carries under the test seam
      {!Param_guards.with_guards} (gh-ocannl-1768); [None] (the {!Pure_C_config} default) guards
      nothing. *)

  val select_blend_syntax : Ops.prec -> select_blend option
  (** A branchless rendering of a value select at this compute precision (gh-ocannl-1622), in one of
      the {!select_blend} forms: it must evaluate both arms unconditionally — each rendered before
      any [?:], [&&] or [||] ({!operand_conditionality_violations} checks it) — and have exactly the
      C type and value of the short-circuiting spelling it replaces, bit for bit. [pp_scalar_op]
      uses it for a [Where], [Relu_gate] or [Satur01_gate] that {!Low_level.speculable_select}
      admits, keeping the condition exactly as the short-circuiting spelling tests it.

      Why: C forbids evaluating the unselected arm of [?:], so where an arm holds a load the
      compiler cannot prove dereferenceable it emits a conditional branch, and a step's time follows
      the data (cifar_resnet's inlined ReLU and max-pool backward selects on cc, gh-ocannl-1557).
      With both arms already values, the compiler if-converts. AMDGPU branches too (gh-ocannl-1695),
      so HIP blends over [auto] locals. [None] (the {!Pure_C_config} default) keeps the
      short-circuiting [ternop_syntax] / [binop_syntax] spelling: CUDA's, which ptxas if-converts
      itself, and Metal's, whose compiler no tool here can read. *)

  val unop_syntax : Ops.prec -> Ops.unop -> PPrint.document -> PPrint.document
  val vec_unop_syntax : Ops.prec -> Ops.vec_unop -> PPrint.document -> PPrint.document
  val convert_precision : from:Ops.prec -> to_:Ops.prec -> string * string

  val compute_prec : Ops.prec -> Ops.prec
  (** The precision the {e arithmetic} over a node runs at, given the precision the node is
      {e stored} at (gh-ocannl-517). Identity by default: a backend with native arithmetic at every
      storage width computes where it stores, which is what the GPU backends do ([__nv_bfloat16],
      MSL's [bfloat]/[half], and the 16-bit tensor-core shapes that consume them).

      The CPU backends have no 16-bit arithmetic, so they map the narrow floats to f32 (subject to
      {!Ir.Numerics.narrow_compute_f32}). Reads then widen once at the load and the result narrows
      once at the store, instead of every operator round-tripping through f32 and rounding its
      result to the narrow format — the "16-bit storage, f32 compute" of gh-ocannl-517.

      Only the register precision of an assignment's intermediates is at stake: this function is
      never consulted for a declaration, a kernel parameter, or a buffer's element type, which
      always take the storage precision. It must be a function of the storage precision alone —
      identical across sibling autotune candidates — or schedule transforms would stop being
      numerics-preserving. *)

  val accum_prec : Ops.prec -> Ops.prec
  (** The precision a reduction {e accumulator} resides at across its whole nest, given the
      precision the accumulated node is stored at (gh-ocannl-663). Where it differs from the storage
      precision, every serial-rendered form of a recognized accumulation — the plain serial
      fallback, unrolled and partitioned nests, and reduction-shaped scope locals — holds the
      accumulator here and narrows once at the store, so a reduction's effective accumulation width
      is a per-backend policy, never a property of which schedule or rendering ran (gh-ocannl-639's
      guarantee, extended beyond [compute_prec]).

      On the CPU backends this {e is} [compute_prec]: the accumulator is an assignment intermediate
      like any other. The GPU backends compute where they store ([compute_prec] is the identity on
      their native narrow arithmetic) but accumulate per their tensor-unit format triples, and this
      hook is where a backend mirrors those: CUDA's bf16 mma legs hold f32 per-lane registers across
      the whole [k] extent (the hardware has no bf16 accumulate), so its serial legs must widen bf16
      the same way, while Metal's uniform-bf16 tile accumulates in bf16 fragments, so its serial
      legs keep bf16 residency — as HIP's do only under [Numerics.Bf16_narrow]: under the default
      [Bf16_auto] HIP swaps its uniform-bf16 arm to an f32 accumulator fragment and widens its
      serial legs to match (gh-ocannl-1051). fp8 has an accumulator format on no backend and follows
      the CPU policy (f32) everywhere; f16 accumulates natively at f16 in every seeded GPU triple
      and stays put.

      The recognized accumulation update renders {e wholly} at this precision, contribution included
      — gh-ocannl-639's rendering shape, kept on GPU deliberately: operand widenings are exact, a
      narrow-by-narrow product is exact at the wider precision (the same full-precision-product
      semantics the tensor units apply per element), and the FMA form stays a single fused
      operation. Statements outside recognized accumulations keep {!compute_prec}.

      Must resolve at least as wide as {!compute_prec} (asserted at codegen setup: narrowing an
      intermediate below its own arithmetic precision would round-trip every update), and like it
      must be a function of the storage precision alone. When overriding {!compute_prec}, override
      this together with it — the two are bound at [include] time, so a stale pairing does not track
      the override. *)

  val vector_prec_ok : Ops.prec -> bool
  (** Whether the explicit vector-arithmetic renderings ([`Vec_extensions] [Vectorized] loops and
      reductions) can operate at this {e compute} precision. f32 and f64 everywhere; fp16
      additionally on CPU targets with native 16-bit arithmetic (gh-ocannl-516), which is exactly
      where {!compute_prec} leaves [Half_prec] alone. A storage precision this rejects can still be
      vectorized when {!compute_prec} maps it to one this accepts -- that is gh-ocannl-517's
      convert-on-load/store. The [`Packed_struct] rendering does no vector arithmetic and does not
      consult this: its own gate admits every float precision with a full-width [vec_typ_of_prec]
      pack type that is also the storage precision of the nodes it packs (gh-ocannl-1295). *)

  val hardware_index : kind:[ `Grid | `Workgroup ] -> slot:int -> string option
  (** The hardware register expression an annotated loop's index binds to (e.g. ["blockIdx.x"],
      ["gid.y"]), or [None] when the backend cannot bind this axis in hardware — the loop then
      renders as a serial [for] (a legal implementation absent barriers; see
      docs/proposals/axis-types-for-loops.md §2/§5). Slots are positional: 0 = [.x], 1 = [.y], 2 =
      [.z]. *)

  val barrier_syntax : string option
  (** Workgroup barrier statement ([__syncthreads();] / [threadgroup_barrier(...);]); [None] makes
      [Workgroup_barrier] a compile-time error (serialization cannot implement a barrier). *)

  val async_copy : async_copy_syntax option
  (** gh-ocannl-487 phase 2: asynchronous global→workgroup-shared copies (CUDA [cp.async]) for the
      staging loads of software-pipelined tiles ({!Low_level.optimized.pipelined}). When provided,
      an eligible staging [Set] (a raw same-precision copy of a materialized global into an
      async-eligible pipelined tile) renders as [ac_copy] instead of a load+store through registers,
      so the prefetch issued for iteration [k+1] genuinely overlaps the compute of [k]. Completion
      is uniform, not per-group: the rotor loop's body is prefixed with [ac_wait_all] followed by a
      workgroup barrier (re-inserting, for the async arm, exactly the phase opener that
      {!Schedule.elide_staged_barriers} elides for synchronous stores — those are published by the
      previous iteration's trailing bracket, an async copy needs its wait BEFORE the publishing
      barrier). Per-statement eligibility is opportunistic: an ineligible staging statement
      (precision conversion, a surviving fringe ternary, a non-global source) keeps the plain store,
      which the same barrier publishes — correctness never depends on which statements the arm
      accepted. [None] (the default, and every backend but CUDA today) keeps the portable
      synchronous rendering everywhere. *)

  val parallel_grid_syntax : [ `None | `Dispatch | `Openmp ]
  (** Pool-backed [Grid] rendering (docs/proposals/gh-ocannl-164.md): how to render an eligible
      outermost [Grid] loop when [hardware_index] does not bind it. [`Dispatch] emits libdispatch's
      [dispatch_apply] over contiguous chunks (macOS; blocks extension), [`Openmp] a
      [#pragma omp parallel for] over the chunk loop; both runtimes own a single process-global
      thread pool, so no pool state lives in the compiled kernel. [`None] keeps the serial fallback.
      Eligibility is decided per loop by [compile_proc] (see [parallel_grid_safe]); [Workgroup]
      loops always stay serial inside a chunk. *)

  val parallel_grid_chunks : int
  (** Target chunk count for [parallel_grid_syntax] (e.g. a small multiple of the core count); the
      actual count is capped by the loop extent. Values [<= 1] disable parallel rendering. *)

  val shared_decl_prefix : string option
  (** Declaration prefix for workgroup-shared placements ([__shared__ ] / [threadgroup ]); [None]
      makes a non-empty [workgroup_shared] set a compile-time error. *)

  val volatile_serial_accumulation : bool
  (** Workaround for a Metal shader-compiler miscompilation of serial accumulations (observed on
      macOS 15/Metal 3.1-3.2 through macOS 26/Metal 4), in both spellings a reduction can take.
      Named for what it covers rather than for either spelling: it was [volatile_scalar_rmw] while
      only the read-modify-write arm existed (gh-ocannl-782 renamed it).

      The original manifestation: a serial loop accumulating into a loop-invariant address of a
      kernel-parameter-derived pointer — [acc[k] = acc[k] + f(i)] with [k] free of [i] — can execute
      as if the load were hoisted above the loop and the store sunk below it {e without} carrying
      the accumulation, leaving only the last iteration's contribution (scalar losses collapsed to
      the last sample's CE; [w.grad] accumulated only the last batch element). Standalone repro:
      [benchmarks/runners/ocannl/bench_metal_bug.ml].

      The localized manifestation (gh-ocannl-731): after the serial-reduction localizer the
      accumulator is a scope local and the node is stored once, and the same pass corrupts THAT form
      instead — by a data-independent additive constant. Standalone repro and one-factor-at-a-time
      matrix: [benchmarks/runners/ocannl/bench_metal_bug_local.ml], whose findings are why the rule
      keys on statement shape rather than on anything finer. Neither the pooled slot table nor
      [__restrict] nor the placement of the preceding device store is the trigger: dropping every
      dynamic load (pointers built from literal offsets straight off a kernel parameter) miscompiles
      identically, and so does moving the preceding store to an unrelated cell. What does remove it,
      besides the qualifier, is having the accumulating loop read no device pointer at all.

      When [true]: device reads in reduction-shaped scope-local updates and their controlling
      guards, and in [Set] statements that read the written node at an index invariant across at
      least one enclosing serial [for] loop, use expression-level [volatile] pointer casts. The cast
      exists only while rendering the accumulating expression: accumulator declarations, opening
      reads, vectorized/packed paths and MMA reads stay plain. Pointwise updates and non-accumulator
      locals stay untouched. Both decisions are reported per routine by the volatility census
      ({!volatility_summary}).

      The form matters: qualifying the accumulator itself cost 1.06x on a memory-bound per-thread
      reduction, 2.15x on an accumulator-bound dependency chain and 4.1x on a long single-threaded
      scalar-loss reduction on an M4 Max. The confined volatile-read form measured 1.03x on that
      loss shape while keeping the standalone reproducer matrix correct row-for-row
      ([bench_metal_bug_local], gh-ocannl-820). *)

  val restrict_keyword : string option
  (** No-alias qualifier for kernel pointer parameters and, in the pooled style, for the derived
      per-node pointers ([restrict] / [__restrict__] / [__restrict]); [None] emits no qualifier.
      Sound because kernel parameters are buffer-owning roots addressing disjoint (sub-)ranges:
      alias views are rewritten to parent accesses at assignments lowering and never reach
      [compile_proc]'s parameter list (asserted there; gh-ocannl-164). The merge buffer stays
      unqualified — a streaming merge mode could point it at a live same-device buffer. *)

  val vectorize_pragma : string list
  (** Lines emitted verbatim before a [Vectorized]-typed loop's [for] statement (gh-ocannl-164),
      e.g. guarded [#pragma clang loop vectorize(enable)] / [#pragma GCC ivdep]. An empty list
      renders the loop as a plain serial [for] — the legal fallback, mirroring
      [hardware_index = None]. Used when explicit vector emission ([vector_file.bytes]) is disabled
      or the loop body is ineligible for it. *)

  val vector_file : Vector_file.t
  (** The vector register file of the explicit-SIMD renderings, its width and its register count as
      one value (gh-ocannl-1399), so that a backend overriding the width cannot inherit a count of
      [0] and silently lose the register tile.

      The width ([vector_file.bytes]): vector register width in bytes for explicit SIMD rendering of
      [Vectorized] loops via GCC/Clang vector extensions (the [Vectorized] codegen follow-up of
      gh-ocannl-164 / docs/proposals/watch-ocannl-README-md-347818d3.md): eligible loop bodies emit
      vector-typed loads, arithmetic and stores in [lanes = bytes / element size] chunks plus a
      serial remainder loop, instead of relying on the compiler's auto-vectorizer (which e.g. cannot
      reassociate strict-FP reductions — the [Vectorized] retype carries that permission, like
      [Swap]). A recognized accumulation body renders as independent accumulator chains with a
      horizontal reduce at loop exit (gh-ocannl-468; [`Vec_extensions] only). [0] disables explicit
      emission ([vectorize_pragma] fallback).

      The count ([vector_file.registers], gh-ocannl-1285): what the register-tiled [Tile_mma]'s
      column cap {!Register_tile.rn_cap} fits to, the same file the backend reports as
      [hardware_limits.simd_vector] so the seeding and this renderer agree. [0] where the width is
      [0], and on the [`Packed_struct] backends, which render no register tile. *)

  val vector_style : [ `Vec_extensions | `Packed_struct ]
  (** How eligible [Vectorized] loops emit explicit vector code when [vector_file.bytes > 0].
      [`Vec_extensions] (CPU): GCC/Clang [vector_size] types, unaligned [__builtin_memcpy]
      loads/stores, vector-infix arithmetic. [`Packed_struct] (GPU, gh-ocannl-463; llm.c's
      [Packed128], llmc/cuda_utils.cuh): the backend's [vec_typ_of_prec] aggregate is loaded and
      stored through [reinterpret_cast] at guaranteed-aligned offsets — the 128-bit LDG/STS
      transactions that bandwidth-bound kernels need — while the arithmetic stays scalar in a
      per-lane loop over the pack's [.v] payload (on GPU the payoff is memory transactions, not SIMD
      ALUs; per-lane [fmaf]/[fma] also matches the serial path's rounding exactly). [`Packed_struct]
      eligibility additionally requires every vector-accessed node to be materialized (device
      buffers and pool offsets are [Ops.buffer_alignment]-aligned, stack and workgroup-shared arrays
      only element-aligned) and every access's non-loop offset contribution to be a lane multiple.
  *)

  val aligned_local_attr : string option
  (** Declaration suffix aligning stack-allocated local arrays for SIMD access, e.g.
      [__attribute__((aligned(32)))] (gh-ocannl-164). Applies to the plain stack-array branch only,
      never to workgroup-shared placements. *)

  val heap_local_arrays : bool
  (** Whether a [Local]-placement array past the stack budget is declared on the heap
      (gh-ocannl-1669): the routine's function-scope arrays beyond [routine_stack_locals_bytes], a
      pool-parallel [Grid] chunk's privatized tiles beyond [chunk_stack_locals_bytes], smallest
      first on the stack. A heap array is [malloc]ed ([calloc]ed where its declaration
      zero-initializes) [Ops.buffer_alignment]-aligned where the stack array would be declared, and
      freed at the end of the same scope. [false] (the {!Pure_C_config} default) for the GPU
      backends, whose thread-private arrays have no heap to go to. *)

  val warp_size : int
  (** SIMD-group (warp) width for the warp-shuffle rendering of [Workgroup_reduce] accumulation
      loops (gh-ocannl-462; llm.c's [warpReduceSum]/[blockReduce] idiom). Backends setting this to a
      nonzero power of two must define [ocannl_shfl_xor(value, lane_mask)] overloads in their
      builtins for the supported accumulator precisions (single, and double where it exists), bind
      workgroup slot 0 in [hardware_index], and provide [barrier_syntax] plus [shared_decl_prefix]
      (needed by the two-phase multi-warp form). Those two overloads are all the rendering asks for
      at any storage width: the shuffled value resides at [accum_prec] of the storage precision
      (gh-ocannl-682), and a storage precision whose residency is neither f32 nor f64 is refused —
      as is an RNG-bearing contribution wherever that residency is wider than storage, since the
      serial rendering of one keeps a narrow accumulator ({!Low_level.accum_skip}). [0] disables the
      rendering: [Workgroup_reduce] loops render like [Workgroup] — hardware binding, or the serial
      fallback (which is the correct meaning of a recognized accumulation body on CPU backends). *)

  val mma_syntax :
    (d_prec:Ops.prec ->
    a_prec:Ops.prec ->
    b_prec:Ops.prec ->
    ta:bool ->
    tb:bool ->
    m:int ->
    n:int ->
    k:int ->
    d:mma_operand ->
    a:mma_source ->
    b:mma_source ->
    mma_emission option)
    option
  (** Cooperative tile-MMA emission for [Low_level.Tile_mma] (docs/proposals/tensorize-mma.md §4):
      given the per-operand precisions (the backend decides which combinations its units support —
      Metal advertises uniform storage triples but can select an f32 accumulator internally for its
      wide-f16 arm; CUDA wmma's flagship combination is mixed f16×f16→f32), the transposed-storage
      flags [ta]/[tb] (the operand's stored layout is the transpose of its role — load tiles with
      the hardware transpose flag and swapped offset arithmetic), the covered block extents
      [m]/[n]/[k], and per operand its leading-dimension stride in elements, its address space and
      its physical layout ({!type-mma_layout}) — plus, for [d], a pointer expression to the tile
      base (already offset) — emit the intrinsic sequence (fragment declarations / loads / mma steps
      / stores) executed by every lane of the enclosing lane loop. Return [None] to decline a
      particular call (unsupported precision combination, extents not multiples of the intrinsic
      tile, thread-space operand, a swizzled layout the arm has no load form for) — the caller then
      renders the scalar [fallback] under an [if (lane == 0)] guard, which is also the path when the
      whole hook is [None] (cc, and any backend until wired).

      Acceptance is thus decided without the a/b tile addresses: an accepting arm returns an
      {!type-mma_emission} that the caller applies to them once it stands where they are renderable.
      Callers that only need to know whether a call is supported — the fragment scope deciding
      whether to alias its accumulator back to the backing target — test the outer option and never
      apply the emission.

      Accepting a [`Swizzled_b128] operand is a promise that it was consumed through a swizzle-aware
      load. {!mma_operand_loads} distinguishes that instruction from swizzle-aware gathers in the
      rendering census. An arm without a swizzle-aware load must decline the call. *)

  val mma_operand_loads :
    a_prec:Ops.prec ->
    b_prec:Ops.prec ->
    d_prec:Ops.prec ->
    scope:Backend_intf.mma_emission_scope ->
    ta:bool ->
    tb:bool ->
    a:Backend_intf.mma_operand_place ->
    b:Backend_intf.mma_operand_place ->
    bool * bool
  (** Which of an accepted {!mma_syntax} call's [a] and [b] operands load through a warp-cooperative
      matrix load ([ldmatrix]) rather than per-lane gathers. Shared-swizzled operands can also use
      byte gathers, so layout alone cannot establish [ldmatrix] emission: derive it from the rule
      the emission itself applies. Read twice, so the two cannot drift: by the rendering census
      ([Mma_intrinsics_ldmatrix] when either is [true]) and, published as
      {!Backend_intf.codegen_capabilities.mma_operand_loads}, by the cost model's load tie-break
      (gh-ocannl-1254). Backends without that instruction inherit [(false, false)]. *)

  val mma_operand_fragments :
    a_prec:Ops.prec ->
    b_prec:Ops.prec ->
    d_prec:Ops.prec ->
    scope:Backend_intf.mma_emission_scope ->
    Backend_intf.mma_fragments option
  (** How an accepted call for these storage precisions, in this emission scope, loads its operand
      fragments ({!Backend_intf.mma_fragments}); published as
      {!Backend_intf.codegen_capabilities.mma_operand_fragments} (gh-ocannl-1254). State it beside
      the loops the hook emits. Default [None]: undescribed, so the cost model counts no loads. *)

  val mma_arm :
    a_prec:Ops.prec ->
    b_prec:Ops.prec ->
    d_prec:Ops.prec ->
    scope:Backend_intf.mma_emission_scope ->
    Backend_intf.mma_arm option
  (** The arm {!mma_syntax} ([Mma_per_statement]) or {!mma_fragment_syntax} ([Mma_fragment_scope])
      selects for these storage precisions under the current numerics policy (gh-ocannl-1153),
      published as {!Backend_intf.codegen_capabilities.mma_arm} and so into the schedule cache's
      identity. Derive it from the very combination table and floors those hooks dispatch on — a
      second copy would let a code change to the hooks' arm choice leave the identity alone. Where
      several arms compete for a triple (CUDA), resolve them once and have the hooks match on the
      result ([Cuda_mma.resolve_arm], gh-ocannl-1232) rather than restate a dispatch order here.
      [None] for a combination neither hook has an arm for, and everywhere on a backend without
      them. *)

  val mma_fragment_syntax :
    (d_prec:Ops.prec ->
    a_prec:Ops.prec ->
    b_prec:Ops.prec ->
    m:int ->
    n:int ->
    k:int ->
    fragment:string ->
    target:mma_operand ->
    a:mma_source ->
    b:mma_source ->
    body:(unit -> PPrint.document) ->
    PPrint.document option)
    option
  (** Rendering of a marked cross-reduction accumulator lifetime. The callback receives the backing
      target and one representative [Tile_mma]'s operands/shape, so it can decline before forcing
      [body]. When accepted, forcing [body] renders its [Tile_mma] with [d] identified as
      [`Fragment fragment], allowing the backend to emit update-only MMA steps between one outer
      fragment load and store.

      Only [target] carries a pointer: it is the one operand this emission addresses (the fragment
      load and store bracketing the reduction). [a] and [b] arrive as {!type-mma_source} — extents,
      space and layout to decide acceptance by, no address — because they are addressed by the
      nested [Tile_mma]s, each at its own position inside the reduction loop. *)

  val mma_kernel_prologue : triples:(Ops.prec * Ops.prec * Ops.prec) list -> string list
  (** Kernel-scope lines — declarations and their initialization — that the tile-MMA emissions of
      one kernel share, given the [(d, a, b)] storage triples of the emissions the hooks accepted in
      it (gh-ocannl-1255). [compile_proc] places them once, after the local declarations and before
      the main logic; every emission scope reading them opens with a barrier, which orders the
      initialization before its reads. Metal's converted destination boundary is the one non-empty
      case: its coordinate table, the same 0..63 for every scope, lives in [threadgroup] memory, so
      one copy per kernel serves all of them. Default: none. *)

  val mma_kernel_workgroup_bytes : triples:(Ops.prec * Ops.prec * Ops.prec) list -> int
  (** The workgroup-shared bytes {!mma_kernel_prologue} declares for these triples, derived from the
      same predicate, so [Schedule.check_hardware_limits_classified]'s estimate cannot drift from
      the kernel. The estimate passes the triples of EVERY [Tile_mma] statement of the kernel —
      including those a hook then declines (extents, spaces, layouts), which the renderer does not
      pass — so it is an upper bound. Default 0. *)

  val workgroup_allocation_alignment : int
  (** The granule, in bytes, the device's allocator rounds each [shared_decl_prefix] array up to
      (gh-ocannl-1330), published as
      {!Backend_intf.codegen_capabilities.workgroup_allocation_alignment} for the schedule-level
      estimate to round each allocation by. Metal: 4 (measured). Default 1: the declared sizes add
      up exactly. *)

  val kernel_log_param : (string * string) option
  (** Kernel parameter for logging, if any. E.g., (Some ("int", "log_id")) or (Some ("const char*",
      "log_file_name")). *)

  val log_involves_file_management : bool
  (** Whether the logging setup involves opening/closing a FILE* (e.g., for fprintf). *)

  val pp_log_statement :
    log_param_c_expr_doc:PPrint.document option ->
    base_message_literal:string ->
    args_docs:PPrint.document list ->
    PPrint.document
  (** Generates a C log statement.
      - [log_param_c_expr_doc]: Document for the C expression of the log parameter (e.g.,
        [string "log_id"] or [string "log_file_name"]), if [kernel_log_param] is Some).
      - [base_message_literal]: The raw, unescaped, unquoted base printf-style format string (e.g.,
        "index %s = %d\n").
      - [args_docs]: Documents for the C expressions of the arguments to the format string. The
        implementation should handle quoting [base_message_literal], choosing the log function
        (printf, fprintf, os_log), and prepending any necessary prefixes (like a log_id or
        captured_log_prefix) to the format string and arguments. *)
end

val codegen_capabilities : (module C_syntax_config) -> Backend_intf.codegen_capabilities
(** The codegen capabilities a backend's config publishes, derived from its own hooks. *)

val default_binop_syntax :
  Ops.prec -> Ops.binop -> PPrint.document -> PPrint.document -> PPrint.document
(** The C-family layout of a binop's {!Ops.binop_c_syntax} spelling, shared by every backend. *)

val rng_binop_syntax : backend:string -> call:(String.t -> 'a) -> Ops.prec -> Ops.binop -> 'a
(** Render an RNG binop (the Threefry variants, the uniform lane conversion) as its builtin's call.
*)

val all_precs : Ops.prec List.t
(** Every storage-bearing precision: an alias of {!Ops.storage_precs}. *)

val operand_conditionality_violations :
  ternop_syntax:
    (Ops.prec ->
    Ops.ternop ->
    PPrint.document ->
    PPrint.document ->
    PPrint.document ->
    PPrint.ToBuffer.document) ->
  binop_syntax:
    (Ops.prec -> Ops.binop -> PPrint.document -> PPrint.document -> PPrint.ToBuffer.document) ->
  select_blend_syntax:(Ops.prec -> select_blend option) ->
  string List.t
(** Where a backend's operator renderings break the operand-evaluation contract; [[]] when nowhere.
*)

val builtin_idents : ('a * 'b * 'c) List.t -> 'a List.t
(** The names a builtins table defines (its entries' keys), for a backend's [ident_blacklist]. *)

val builtin_definitions :
  (String.t * 'a * String.t List.t) List.t -> keys:String.t List.t -> 'a List.t
(** The definitions [keys] need from a builtins table, dependencies included, in table order. *)

val vec_widen_macro : store_prec:Ops.prec -> prec:Ops.prec -> lanes:int -> String.t option
(** The cc builtin widening a 16-bit-storage vector to [lanes] lanes of [prec]; [None] otherwise. *)

val vec_narrow_macro : store_prec:Ops.prec -> prec:Ops.prec -> lanes:int -> String.t option
(** The cc builtin narrowing [lanes] lanes of [prec] to 16-bit storage; [None] otherwise. *)

val partial_staging_idents : string * string
(** The [(bits, conv)] locals a register tile's partial bf16/fp16 column is staged through. *)

val derived_name : String.t * String.t -> String.t -> String.t
(** The identifier [prefix ^ base ^ suffix] a derivation template spells from [base]. *)

val mem_alias_derivation : string * string
(** Derivation template of a local array's alias pointer. *)

val heap_local_derivation : string * string
(** Derivation template of the allocation a heap-declared local array points into. *)

val grid_chunk_derivations : (string * string) * (string * string) * (string * string)
(** Derivation templates of a pool-chunked [Grid] loop's chunk index, lower and upper bounds. *)

val lane_reduce_derivations : (string * string) * (string * string)
(** Derivation templates of a lane all-reduce's value and partials. *)

val warp_reduce_derivations : (string * string) * (string * string)
(** Derivation templates of a warp-shuffle reduction's value and partials. *)

val mma_fragment_derivation : string * string
(** Derivation template of an MMA fragment register, applied to its node's uid. *)

val minted_derivations : (string * string) list
(** Every derivation template the emitter mints names with, for the kernel code cache's renaming. *)

val vec_widen_rows_macros :
  store_prec:Ops.prec -> prec:Ops.prec -> lanes:int -> (string * string) option
(** The [(pack, row)] cc builtins a register tile's A column widens through; [None]: per-row
    widening. *)

val cpp_keywords : string list
(** The words C++ reserves beyond C's, for the CUDA, HIP and MSL backends' blacklists. *)

val implementation_reserved : String.t -> bool
(** Whether [name] lies in the implementation-reserved namespace: [__x] or [_X]. *)

val system_macro_idents : string list
(** Predefined compiler macros outside the reserved namespace ([linux], [unix], ...). *)

type routine_name_probe = {
  prelude : string;
      (** The source every kernel follows: the unconditional includes, with every conditional
          include ahead of them, so a name any kernel's headers declare is measured. *)
  kernel_definition : string -> string;
      (** A minimal kernel DEFINITION of the given name, one line, as the backend declares one. *)
  compile : string -> (unit, string) Result.t;
      (** The backend's runtime compiler on a whole source; [Error] carries its report. *)
  kernel_ident : string -> string;
      (** The backend's routine-name mangling ({!C_syntax.kernel_ident}). *)
  routine_floor_idents : string list;
      (** What routine names avoid besides [vendor_routine_idents]: the node-name blacklist and the
          C library's names. The reserved namespace ({!implementation_reserved}) is a rule, not a
          list. *)
  vendor_routine_idents : string list;  (** The backend's harvested table. *)
}
(** What [tools/vendor_idents.exe] needs from a backend to harvest its toolchain's file-scope names.
*)

val vector_type_names :
  prefixes:String.t List.t -> scalars:String.t List.t -> widths:Int.t List.t -> String.t List.t
(** Every [<prefix><scalar><width>] spelling: GPU dialects' vector-type names, for name blacklists.
*)

val diagnostic_scopes : String.t -> ([> `Error | `Other | `Unscoped ] * String.t) List.t
(** Each line of a compiler report paired with its diagnostic's severity, or [`Unscoped]. *)

val routine_name_in_diagnostics : routine_names:String.t List.t -> String.t -> String.t option
(** The first of [routine_names] whose declaration a compiler's error report blames, if any. *)

val codegen_bug_claim_prefix : string
(** The opening of the sentence a compile-failure report blames OCANNL's codegen with. *)

val codegen_bug_claim : artifact:String.t -> String.t
(** The one-line codegen-bug sentence, asking for [artifact]. *)

val missing_include : String.t -> bool
(** Whether a failed compile's report says a header could not be opened: a toolchain fault. *)

val attribute_routine_name_failure :
  ?report:string ->
  routine_names:(String.t * String.t) List.t ->
  exn ->
  Stdlib.Printexc.raw_backtrace ->
  'a
(** Re-raise a failed compile, as a [Utils.User_error] naming the routine when its name caused it.
*)

val printf_support_unless_uniform : unit -> bool
(** [full_printf_support] for a [%g]-capable dialect: true unless [prefer_backend_uniformity] is
    set. *)

val source_mentions : marker:string -> string -> bool
(** Whether [source]'s code tokens (not comments or literals) contain [marker]'s tokens in sequence.
*)

val prepend_conditional_includes :
  conditional_includes:(string * string) List.t -> string -> String.t
(** Prepend each header whose marker [source] mentions and that [source] does not already include.
*)

val with_compiler_options :
  compiler:String.t ->
  options:string list ->
  enrich:(exn -> suffix:String.t -> exn Option.t) ->
  (unit -> 'a) ->
  'a
(** Run [compile], attaching the compiler's options to an exception [enrich] recognizes. *)

(** The [Blend_over_locals] renderings, shared by the emitter and
    {!operand_conditionality_violations}. *)
module Over_locals : sig
  val blend_where :
    decl:string ->
    ternop_syntax:
      (Ops.prec ->
      Ops.ternop ->
      PPrint.document ->
      PPrint.document ->
      PPrint.document ->
      PPrint.document) ->
    Ops.prec ->
    PPrint.document ->
    PPrint.document ->
    PPrint.document ->
    PPrint.document

  val blend_gate :
    decl:string ->
    binop_syntax:(Ops.prec -> Ops.binop -> PPrint.document -> PPrint.document -> PPrint.document) ->
    Ops.prec ->
    Ops.binop ->
    PPrint.document ->
    PPrint.document ->
    PPrint.document
end

(** Builtin parameter guards: a test seam checking builtin calls pass arguments without narrowing.
*)
module Param_guards : sig
  val parameter_types : key:string -> string -> (string list, string) Result.t
  (** The parameter types of the function [key] that a table entry's [definition] defines, read off
      its signature, or why there are none to guard: {!a_macro} and {!not_a_function} for an entry
      that is no function, another reason for a function the guard skips. *)

  val a_macro : string
  (** The reason given for a macro. *)

  val not_a_function : string
  (** The reason given for an entry that names [key] nowhere as a call or a definition: a type, a
      constant, a header. *)

  val prelude : param_guard_dialect -> string
  (** The helpers the guards of {!guard} expand to, emitted once before them. *)

  val guard : param_guard_dialect -> key:string -> string list -> string
  (** The macro shadowing [key], whose parameters have these types. *)

  val with_guards : ?guard:bool -> ?overrides:(string * string) list -> (unit -> 'a) -> 'a
  (** [with_guards ?guard ?overrides f] runs [f] with every kernel prelude of a backend that
      declares a {!param_guard_dialect} guarding the functions the kernel calls (unless [guard] is
      [false]), and with each [(key, definition)] of [overrides] replacing that key's definition in
      the backend's table -- how a test plants a defect, and shows both that it compiles unguarded
      and that the guards catch it. A test seam: kernels are never guarded outside it. An override
      naming a key the table lacks fails the prelude's assembly. *)

  val active : unit -> (bool * (string * string) list) option
  (** [Some (guard, overrides)] inside {!with_guards}, [None] outside it. *)

  val note_emitted : int -> unit

  val emitted : unit -> int
  (** How many guards the preludes assembled in this process carried: zero where no backend declares
      a dialect, which is how a test sees that its guards were never emitted. *)
end

(** C source tokens for builtin and header selection: comments and literals are not uses. *)
module Source_tokens : sig
  type t

  val scan : string -> t
  val mentions : t -> string -> bool
end

(** The plain-C defaults of {!C_syntax_config}, the cc backend's, which other backends override. *)
module Pure_C_config (_ : sig
  val procs : Low_level.t array
  val full_printf_support : bool
end) : sig
  include C_syntax_config

  val for_log_trace_tree : bool
  (** Whether log lines mark newlines with [$] for trace-tree logs: [debug_log_to_stream_files]. *)
end

(** The C-family renderer for one backend's config: routine naming, prelude assembly, kernel
    emission. *)
module C_syntax (_ : C_syntax_config) : sig
  val kernel_ident : String.t -> String.t
  (** The C identifier a routine's kernel is emitted under: sanitized and clear of reserved names.
  *)

  val routine_name_probe :
    prelude:string -> compile:(string -> (unit, string) Result.t) -> routine_name_probe
  (** This dialect's half of a {!type-routine_name_probe}; the backend supplies prelude and
      compiler. *)

  type render_ctx
  (** One routine's rendering state. *)

  val filter_and_prepend_builtins :
    ?conditional_includes:(string * string) List.t ->
    routine_names:String.t List.t ->
    includes:string ->
    builtins:(String.t * string * String.t List.t) List.t ->
    proc_doc:PPrint.ToBuffer.document ->
    unit ->
    String.t
  (** Kernel source: [includes], the builtin definitions the kernels use, then [proc_doc]'s kernels.
  *)

  val doc_to_string : PPrint.ToBuffer.document -> string
  (** Render a document compactly to a string. *)

  val vec_acc_fma :
    prec:Ops.prec -> lanes:int -> dst:string -> a:string -> b:string -> PPrint.document
  (** Render [dst = fma(a, b, dst)] over [lanes]-lane vectors, rounding once as the scalar path
      does. *)

  val compile_main : render_ctx -> Low_level.t -> PPrint.document
  (** Render a statement under the given rendering context. *)

  val create_render_ctx : name:string -> Low_level.optimized -> render_ctx
  (** A fresh rendering context for routine [name], from its optimized lowering. *)

  val zero_out_seen : render_ctx -> int Hash_set.t
  (** The uids whose [Zero_out] has rendered; exposed for [test_render_context]'s white-box checks.
  *)

  val peel_census_enabled : render_ctx -> bool ref
  (** The context's peel-census switch; exposed for [test_render_context]'s white-box checks. *)

  val compile_proc :
    name:String.t ->
    Indexing.static_symbol List.t ->
    Low_level.optimized ->
    (string * Backend_intf.kparam_source) list * PPrint.document * Low_level.launch_dims
  (** Render a kernel definition: its parameters with their sources, the code, its launch geometry.
  *)
end

(** Renders and compiles routines for a backend, attributing name-caused compile failures to the
    name. *)
module Compile_driver
    (_ : functor
      (_ : sig
         val procs : Low_level.t array
       end)
      -> C_syntax_config) : sig
  type 'kparams rendered = {
    entries : ('kparams * string * Low_level.launch_dims) array;
        (** Per kernel, in batch order: its parameters, its emitted name and its launch geometry. *)
    assemble : int list -> string;
        (** The source of one translation unit holding the kernels at these indices, in the given
            order: the includes, the builtins those kernels use and their definitions, conditional
            includes applied. [assemble] of every index in order is the batch source
            {!compile_batch} hands its compiler; [assemble [i]] is kernel [i]'s standalone source,
            which a kernel code cache keys on (gh-ocannl-1314). *)
    assemble_units : int list -> max_units:int -> min_unit_bytes:int -> string list;
        (** gh-ocannl-1603: the same kernels as {!assemble} as several translation units, to be
            compiled concurrently and linked into one library: their units -- a kernel, or an
            outlined kernel's parts and the kernel -- packed by size into at most [max_units]
            sources of at least about [min_unit_bytes] each ([0]: no minimum), each with its own
            prelude. One unit is [[assemble idxs]] exactly. *)
    attribute : 'a. int list -> exn -> Stdlib.Printexc.raw_backtrace -> 'a;
        (** Re-raises a failed compile of [assemble idxs] as {!compile_batch} would: attributed to a
            routine's name when the report points at one of those kernels' declarations. *)
    key_text : int -> string;
        (** Kernel [i]'s standalone source as a kernel code cache keys it (gh-ocannl-1525): its
            prelude verbatim, its definition through {!Kernel_code_cache.canonical_kernel} -- the
            names it owns (symbols, scope locals, its nodes' identifiers, the emitter's
            {!minted_derivations} of them) renamed by first occurrence and its segment index
            dropped, so that an unchanged kernel keys alike whatever names the rest of the batch
            made it take. The backend adds the compiler options and looks the code up by the stored
            symbol. *)
  }
  (** A batch rendered but not compiled: per-kernel entries, and source assemblers over kernel
      indices. *)

  val render_batch :
    names:string array ->
    'a Indexing.bindings ->
    Low_level.optimized array ->
    includes:string ->
    builtins:(string * string * string list) list ->
    ?conditional_includes:(string * string) list ->
    unit ->
    (string * Backend_intf.kparam_source) list rendered
  (** Render a batch with one syntax instance over all of it, without compiling. *)

  val drive :
    batch:bool ->
    names:string array ->
    'a Indexing.bindings ->
    Low_level.optimized array ->
    includes:string ->
    builtins:(string * string * string list) list ->
    ?conditional_includes:(string * string) list ->
    ?compiles_source:bool ->
    compile_source:(name:string -> string -> 'b) ->
    unit ->
    'b * ((string * Backend_intf.kparam_source) list * string * Low_level.launch_dims) array
  (** Render kernels into one source and compile it, attributing name-caused failures to the name.
  *)

  val compile :
    name:string ->
    'a Indexing.bindings ->
    Low_level.optimized ->
    includes:string ->
    builtins:(string * string * string list) list ->
    ?conditional_includes:(string * string) list ->
    ?compiles_source:bool ->
    compile_source:(name:string -> string -> 'b) ->
    unit ->
    'b * (string * Backend_intf.kparam_source) list * string * Low_level.launch_dims
  (** {!drive} for one routine: the compile's result, its parameters, emitted name and launch
      geometry. *)

  val compile_batch :
    names:string array ->
    'a Indexing.bindings ->
    Low_level.optimized array ->
    includes:string ->
    builtins:(string * string * string list) list ->
    ?conditional_includes:(string * string) list ->
    ?compiles_source:bool ->
    compile_source:(name:string -> string -> 'b) ->
    unit ->
    'b * ((string * Backend_intf.kparam_source) list * string * Low_level.launch_dims) array
  (** {!drive} for a batch, compiled as one source named by the routines' common prefix. *)
end
