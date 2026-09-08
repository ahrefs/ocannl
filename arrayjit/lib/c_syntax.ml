open Base
module Lazy = Utils.Lazy
open Backend_intf

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_C_SYNTAX=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_C_SYNTAX"]

module Tn = Tnode

type t = PPrint.document

(* gh-ocannl-344: integer width of the Metal pooled slot table (pool_index, byte_offset per node).
   With [large_models] the per-pool 4 GB cap is lifted (see {!Backends.plan_pool_segments}), so a
   byte offset can exceed [UINT32_MAX] and the slot table -- together with the MSL type the shader
   declares -- must be 64-bit to avoid silent truncation; otherwise 32-bit suffices (offsets are
   capped under 4 GB). This is the single source of truth shared by the codegen (the [const ... *
   __pool_slots] MSL type) and the backend (the [Ctypes] element type), so the two cannot drift.
   [large_models] is a startup-fixed global, read identically when the source is generated and when
   the table is filled. *)
let pool_slot_is_64 () = Utils.settings.large_models
let pool_slot_msl_typ () = if pool_slot_is_64 () then "ulong" else "uint"

(* Opt-in rendering-decline diagnostics (config [schedule_log_declines]; gh-ocannl-474 /
   gh-ocannl-479): one stderr line per decline, naming the kernel, the construct, and the rule that
   failed — when a [Grid] loop fails pool-parallel eligibility ([parallel_grid_safe]) or a
   [Tile_mma] statement falls back from the intrinsic / register-tiled rendering. Complements
   [schedule_log_launches]: that one says what geometry a compile launched, this one says why a
   requested rendering was NOT used (declines are silent by design — the fallback is correct — but
   indistinguishable from the fast path in timings, which already cost one wrong conclusion:
   docs/proposals/tensorize-mma.md's unverified-T3 episode). *)
let log_declines = lazy (Utils.get_global_flag ~default:false ~arg_name:"schedule_log_declines")

(* Per-chunk private tiles live on the pool workers' stacks; libdispatch workers get 512KB by
   default, so cap their combined footprint per Grid loop (declining just keeps the loop serial).
   Config [cc_grid_private_bytes_cap] (gh-ocannl-474): raise it when the pool's worker stacks are
   known larger (e.g. under [OMP_STACKSIZE]) — for instance to let a grid-outermost packed GEMM
   privatize a whole B~ panel per chunk (gh-ocannl-475). At the top level, not inside {!C_syntax},
   because it is also one of the codegen knobs the cc backend's cache-identity tag covers
   (gh-ocannl-572) — the resolved cap must be one value, not two spellings of a default. *)
let per_chunk_private_bytes_cap =
  lazy
    (match
       Int.of_string
         (String.strip
            (Utils.get_global_arg ~arg_name:"cc_grid_private_bytes_cap" ~default:"262144"))
     with
    | n when n > 0 -> n
    | _ -> 256 * 1024
    | exception _ -> 256 * 1024)

(* Census of [Tile_mma] statement renderings, collected during codegen while [mma_census_enabled]
   (gh-ocannl-479): "the tensorized candidate lost" and "the tensorized candidate never ran
   tensorized" must be distinguishable in tuning logs. Entries are (kernel name, rendering), most
   recent first.

   Consumers do not touch these refs: {!with_census} brackets them, {!Context.compile} calls it
   around every routine's codegen, and the resulting {!mma_summary} is a field of the compiled
   routine (gh-ocannl-626). Opt-in collection made the DEFAULT for a new timing harness "report a
   variant name that may be unrelated to what rendered", which is exactly the false perf number the
   census exists to prevent. *)
type mma_rendering =
  | Mma_intrinsics
  | Mma_intrinsics_ldmatrix
      (** The intrinsic arms fed by warp-cooperative [ldmatrix] loads over a [Swizzle_b128] staged
          tile (gh-ocannl-481 item 3) instead of per-lane gathers. Recorded distinctly because the
          gh-476 sweep must be able to pin which of the two load paths a timing measured. *)
  | Mma_register_tiled
  | Mma_scalar_fallback
[@@deriving sexp, compare, equal]

let mma_census_enabled = ref false
let mma_census : (string * mma_rendering) list ref = ref []

let is_tensorized_rendering = function
  | Mma_intrinsics | Mma_intrinsics_ldmatrix | Mma_register_tiled -> true
  | Mma_scalar_fallback -> false

(** Whether a compiled routine's tensorization request was honoured (gh-ocannl-626). Three states,
    not a boolean, because "this routine has no tensor-core work in it" and "this routine asked for
    tensor-core work and got the scalar fallback" are opposite readings of a timing and the boolean
    that collapses them is the defect: an [Tile_mma]-free routine is not a failure, a wholly
    declined one is. Mixed renderings — some statements honoured, some declined — read as
    {!Tensorized}, and the counts on {!mma_summary} say by how much; the label answers "did any
    tensor-core / SIMD-tile emission happen", the counts answer "how much of what was asked for". *)
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
[@@deriving sexp, compare, equal]

let tensorization_name = function
  | Not_requested -> "not-requested"
  | Tensorized -> "tensorized"
  | Scalar_fallback -> "scalar-fallback"

type mma_summary = {
  renderings : (string * mma_rendering) list;
      (** The census entries of one compile, in emission order (kernel name, rendering). Fissioned
          segments of one routine contribute their kernels to the same summary. *)
  tensorization : tensorization;
  statements : int;  (** [Tile_mma] statements emitted: [List.length renderings]. *)
  scalar_fallbacks : int;  (** Of those, how many rendered as the lane-0 scalar fallback. *)
}
[@@deriving sexp_of]
(** What a compile's {!mma_census} says about the routine it produced (gh-ocannl-626). Derived once,
    where the routine is compiled, so that "did this tensorize" is a property of the compiled
    routine rather than of whichever call site remembered to bracket the global. *)

let summarize_census renderings =
  let statements = List.length renderings in
  let scalar_fallbacks =
    List.count renderings ~f:(fun (_, r) -> equal_mma_rendering r Mma_scalar_fallback)
  in
  let tensorization =
    if statements = 0 then Not_requested
    else if scalar_fallbacks = statements then Scalar_fallback
    else Tensorized
  in
  { renderings; tensorization; statements; scalar_fallbacks }

let empty_mma_summary = summarize_census []

let merge_mma_summaries summaries =
  summarize_census (List.concat_map summaries ~f:(fun s -> s.renderings))

(** The one-line census a timing report prints beside a measurement (gh-ocannl-626): the
    {!tensorization} label first — a reader scanning a table for a mismatch reads that word, not a
    histogram — then the per-rendering counts. Shared so that every harness that names a variant
    after a rendering says "did this tensorize" the same way; the two bench copies had disagreed
    about the predicate (one counted the fallback, the other tested equality with
    {!Mma_register_tiled}, which false-warns on any GPU backend). *)
let mma_summary_string summary =
  match summary.renderings with
  | [] -> tensorization_name summary.tensorization
  | rs ->
      let counted =
        List.map
          (List.dedup_and_sort (List.map rs ~f:snd) ~compare:compare_mma_rendering)
          ~f:(fun r ->
            Printf.sprintf "%s x%d"
              (Sexp.to_string (sexp_of_mma_rendering r))
              (List.count rs ~f:(fun (_, r') -> equal_mma_rendering r r')))
      in
      Printf.sprintf "%s: %s"
        (tensorization_name summary.tensorization)
        (String.concat ~sep:", " counted)

(** Run [f] with the {!mma_census} collecting, and return its result alongside the summary of what
    rendered during it (gh-ocannl-626). Both census refs are saved and restored, so calls nest and
    an inner compile does not disturb an outer collection; the summary covers only [f]'s own
    entries. This replaces the hand-rolled [Exn.protect] bracket that had been copied to six call
    sites — and, applied inside {!Context.compile}, makes collecting the census the default rather
    than something a timing harness must remember to do.

    The census is a process global, as it always was: compiles are sequential on the main domain,
    and nothing here makes concurrent compiles from several domains attribute their renderings
    correctly. *)
let with_census f =
  let saved_enabled = !mma_census_enabled and saved_census = !mma_census in
  mma_census_enabled := true;
  mma_census := [];
  (* Nesting is additive, not shadowing: an enclosing collection (a bench bracketing several
     compiles, each of which collects its own summary) must still see the inner entries, or wrapping
     the compile path in this helper would silently empty every outer bracket. *)
  let restore () =
    let inner = !mma_census in
    mma_census_enabled := saved_enabled;
    mma_census := if saved_enabled then inner @ saved_census else saved_census
  in
  let result =
    match f () with
    | r -> r
    | exception e ->
        restore ();
        raise e
  in
  let renderings = List.rev !mma_census in
  restore ();
  (result, summarize_census renderings)

(* {2 The peel census (gh-ocannl-733)}

   The [Tile_mma] census answers "did this routine tensorize". This one answers the same kind of
   question for the reduction peel — "which DECISION produced this kernel" — which the emitted form
   cannot answer on its own: a nest whose accumulated cell is free of the enclosing index peels BOTH
   levels under a [Confined_to_peel] guard, while one whose cell mentions it peels the inner level
   only, under a [Lane_private_if_separated] guard admitted because the cell separates the lanes
   (gh-ocannl-721). The two render the same localized kernel — one closing store, two node
   subscripts, no foreign store — so a test classifying the emitted code passes over either, whether
   or not the code path it is named for ran.

   Collected exactly where the decision is consumed — [try_localize_serial_reduce],
   [try_warp_reduce] and [try_vectorize_reduce], which all take it from the one [decide_accum_width]
   call of their level (gh-ocannl-754) — and only at sites where localization is a live question
   ([Low_level.has_accumulating_cell] of the level's body — a [Set] reading the very cell it
   writes): a loop with no such recurrence was never a candidate, and censusing it would bury the
   reductions in noise. Not [has_accumulation], which counts every [Local_scope] conservatively and
   so records an ordinary virtualized computation inside a loop as a declined reduction site. The
   refs are bracketed by {!with_peel_census}, which {!Context.compile} calls around every routine's
   codegen, so the summary is a field of the compiled routine rather than something a caller must
   remember to collect. *)

type peel_skip =
  | Skip_debug_logging
      (** [debug_log_from_routines]: the per-step [Set] form is kept so the trace survives. *)
  | Skip_dead_level  (** The level being rendered is dead ([to_ < from_]). *)
  | Skip_accum_pinned
      (** The peel reached a base whose update renders at storage precision (the rng carve-out,
          gh-ocannl-517), so localizing it would change the draw rather than move it. *)
[@@deriving sexp, equal, compare]

type peel_verdict = {
  levels : int;
      (** Loop levels the localized scope spans: the level being rendered, plus those
          {!Low_level.peel_accum_nest} took below it. One for a reduction localized at its own
          level; two where the accumulated cell let the peel swallow the enclosing level too. *)
  guards : Low_level.peel_guard_verdict list;
      (** The verdict of each [If] among them, outermost first. *)
}
[@@deriving sexp, equal, compare]
(** What a localizing site DECIDED (gh-ocannl-733). A named record rather than an inline one so a
    consumer can hold one: comparing the verdicts of two compiles is the whole use. *)

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
[@@deriving sexp, equal, compare]

let peel_census_enabled = ref false
let peel_census : (string * peel_site) list ref = ref []

let is_localized_peel = function
  | Peel_localized _ -> true
  | Peel_ceded _ | Peel_refused _ | Peel_not_attempted _ -> false

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
(** What a compile's {!peel_census} says about the routine it produced (gh-ocannl-733). *)

let summarize_peel_census sites =
  let localized = List.count sites ~f:(fun (_, s) -> is_localized_peel s) in
  let ceded =
    List.count sites ~f:(fun (_, s) -> match s with Peel_ceded _ -> true | _ -> false)
  in
  { sites; localized; ceded; declined = List.length sites - localized - ceded }

(** How the accumulator of one reduction level is held — decided ONCE, and consumed by every
    rendering of that level (gh-ocannl-754). A [Workgroup_reduce] level renders as the warp-shuffle
    tree, a [Vectorized] level as the SIMD accumulator grid, and every kind falls back through the
    localizing serial form; each of the three holds the accumulator somewhere other than its storage
    cell, and until this decision existed each recognized the accumulation through an analysis of
    its own — the shuffle and the grid through a single-statement recognizer, the serial form
    through {!Low_level.peel_accum_nest} plus the storage-pinning carve-out. Wherever the two
    disagreed on a body, a retype changed the accumulation WIDTH and so the value (gh-ocannl-639 on
    [Unrolled], gh-ocannl-682 on RNG-bearing shuffles). Now the peel and the carve-out are asked
    here, once, and a rendering that holds the accumulator wide may do so exactly where the serial
    form localizes. *)
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
(** The base of an {!accum_width} that is a single accumulation STATEMENT at the level itself —
    [acc[idcs] = op(acc[idcs], contrib)] or its FMA form, no level or guard between — which is the
    shape the warp-shuffle and SIMD renderings can render. *)

let empty_peel_summary = summarize_peel_census []

(** The one-line peel census a report prints beside a routine (gh-ocannl-733): how many reduction
    sites localized, and the distinct verdicts they earned. *)
let peel_summary_string summary =
  match summary.sites with
  | [] -> "no reduction peel site"
  | sites ->
      let counted =
        List.map
          (List.dedup_and_sort (List.map sites ~f:snd) ~compare:compare_peel_site)
          ~f:(fun s ->
            Printf.sprintf "%s x%d"
              (Sexp.to_string (sexp_of_peel_site s))
              (List.count sites ~f:(fun (_, s') -> equal_peel_site s s')))
      in
      Printf.sprintf "%d localized, %d ceded, %d declined: %s" summary.localized summary.ceded
        summary.declined (String.concat ~sep:", " counted)

(** Run [f] with the {!peel_census} collecting, and return its result alongside the summary of what
    the reduction peel decided during it (gh-ocannl-733). Nests additively and restores both refs,
    exactly as {!with_census} does for the [Tile_mma] census — and for the same reason: an enclosing
    collection must still see an inner compile's sites. *)
let with_peel_census f =
  let saved_enabled = !peel_census_enabled and saved_census = !peel_census in
  peel_census_enabled := true;
  peel_census := [];
  let restore () =
    let inner = !peel_census in
    peel_census_enabled := saved_enabled;
    peel_census := if saved_enabled then inner @ saved_census else saved_census
  in
  let result =
    match f () with
    | r -> r
    | exception e ->
        restore ();
        raise e
  in
  let sites = List.rev !peel_census in
  restore ();
  (result, summarize_peel_census sites)

(* {2 The volatility census (gh-ocannl-782, gh-ocannl-820)}

   Which serial accumulations in a routine carry the {!C_syntax_config.volatile_serial_accumulation}
   workaround, in which of its two forms, and which were left alone. The emitted text answers "does
   this expression contain a volatile read"; it does not answer "how many accumulations did this
   routine have, and did any of them escape the workaround" — and that second question is the one a
   residency or performance investigation asks.

   Collected where each decision is made (the [Local_scope]/[Declare_local] declarations and their
   rendered [Set_local] reads for the localized form, [pp_ll]'s [Set] case for the read-modify-write
   form) and bracketed by {!with_volatility_census}, which {!Context.compile} calls around every
   routine's codegen, so the summary is a field of the compiled routine.

   The two arms are censused asymmetrically, and deliberately. A reduction-shaped scope local is
   recorded on EVERY backend — the classification it needs is computed anyway, so a routine compiled
   for a C backend still reports "three accumulations, none protected" — while the device-memory RMW
   is recorded only where the backend requests the workaround, because elsewhere its structural
   predicate is never evaluated (the capability short-circuits it) and there is no decision to
   record. {!field-volatility_summary.requested} is what tells the two situations apart. *)

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
[@@deriving sexp, equal, compare]

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
(** What a compile's volatility census says about the routine it produced (gh-ocannl-782). *)

(* Set by [compile_proc] before any analysis or rendering: the kernel name, for the decline
   diagnostics ([log_declines]) and for every census entry. Module level, not functor level, for the
   same reason the census refs are: compiles are sequential on the main domain, and a census
   collected around one compile must name the kernels of whichever backend ran it. *)
let current_kernel_name = ref ""
let volatility_census_enabled = ref false
let volatility_census : (string * volatility_site) list ref = ref []

(* Set by the backend functor at each [compile_proc], read by the summary: the capability is a
   per-backend source constant, and a summary that did not carry it could not tell "this backend
   leaves accumulators alone" from "this routine had none". *)
let volatility_requested = ref false

let summarize_volatility_census ~requested entries =
  let count f = List.count entries ~f:(fun (_, s) -> f s) in
  {
    entries;
    requested;
    volatile_accumulations = count (function Volatile_accumulation_reads _ -> true | _ -> false);
    plain_accumulations = count (function Plain_accumulator _ -> true | _ -> false);
    volatile_rmw_reads = count (function Volatile_rmw_reads _ -> true | _ -> false);
  }

let empty_volatility_summary = summarize_volatility_census ~requested:false []

(** The one-line volatility census a report prints beside a routine (gh-ocannl-782/820): how many
    serial accumulations this routine carries and how many selected the volatile-read workaround. *)
let volatility_summary_string summary =
  let accumulations = summary.volatile_accumulations + summary.plain_accumulations in
  if not summary.requested then
    Printf.sprintf "%d accumulation(s), workaround not requested by this backend" accumulations
  else
    Printf.sprintf
      "%d of %d accumulation(s) use the volatile-read workaround, %d volatile rmw read(s)"
      summary.volatile_accumulations accumulations summary.volatile_rmw_reads

(** Run [f] with the volatility census collecting, and return its result alongside the summary of
    what the accumulation workaround decided during it (gh-ocannl-782). Nests additively and
    restores the refs, exactly as {!with_census} and {!with_peel_census} do. *)
let with_volatility_census f =
  let saved_enabled = !volatility_census_enabled
  and saved_census = !volatility_census
  and saved_requested = !volatility_requested in
  volatility_census_enabled := true;
  volatility_census := [];
  volatility_requested := false;
  let restore () =
    let inner = !volatility_census in
    volatility_census_enabled := saved_enabled;
    volatility_census := if saved_enabled then inner @ saved_census else saved_census;
    volatility_requested := saved_requested
  in
  let result =
    match f () with
    | r -> r
    | exception e ->
        restore ();
        raise e
  in
  let entries = List.rev !volatility_census and requested = !volatility_requested in
  restore ();
  (result, summarize_volatility_census ~requested entries)

type mma_space = [ `Device | `Shared | `Thread | `Fragment of string ]
(** The address space of a tile-MMA operand as the emission hooks see it. *)

type mma_layout = [ `Plain | `Swizzled_b128 ]
(** The physical layout of a tile-MMA operand's storage (gh-ocannl-481 item 3, D2). This is the
    whole [Stage] -> emission contract: the emission never re-derives the layout, it trusts the
    component it is handed.

    [`Swizzled_elem] operands never reach a hook — the [Tile_mma] rendering declines them centrally,
    since no intrinsic load form matches an element-granularity permutation. [`Swizzled_b128] is
    only passed when the access is reconstructible from [(ptr, ld)] alone: the pointer is the tile
    origin of a rank-2 node whose minor dim is [ld], so the element at [(row, col)] sits at
    [row*ld + (((col/u) lxor (row land (ld/u - 1))) * u)] with [u = 16 / prec_in_bytes] — everything
    else declines. *)

type mma_operand = PPrint.document * int * mma_space * mma_layout

type mma_source = int * mma_space * mma_layout
(** A tile-MMA input operand described but not addressed: leading-dimension stride in elements,
    address space, physical layout — no pointer. This is how the a and b operands reach both
    emission hooks, because a caller does not always have an address to give: a software-pipelined
    tile's live copy rotates per k-block (gh-ocannl-487), so its pointer only exists inside the
    rotor loop, and {!C_syntax_config.mma_fragment_syntax} runs outside it.

    Everything an arm decides acceptance by is here — the pointer never was a criterion. *)

type mma_emission = a_ptr:PPrint.document -> b_ptr:PPrint.document -> PPrint.document
(** The emitting half of an accepted {!C_syntax_config.mma_syntax} call: the arm has committed, and
    only the a/b tile addresses are still outstanding. Splitting the hook this way makes
    [Option.is_some (mma_syntax ...)] a support predicate usable wherever a call is known but not
    yet addressed — with no separate acceptance implementation that could drift from the emitting
    one. *)

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
(** gh-ocannl-487 phase 2: asynchronous staging copies for software-pipelined tiles (see
    {!C_syntax_config.async_copy}). *)

module type C_syntax_config = sig
  val procs : Low_level.t array
  (** The low-level procedures this functor application will render: one per kernel of the
      compilation unit (a singleton for a plain compile, one entry per routine for a batch). The
      functor reads them for whole-unit analyses -- the identifier census, the scope-local verdicts
      -- while each kernel's own rendering goes through {!compile_proc}, which takes the full
      {!Low_level.optimized} record. *)

  val main_kernel_prefix : string
  val kernel_prep_line : string
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
      the same way, while HIP's and Metal's uniform-bf16 tiles accumulate in bf16 fragments, so
      their serial legs keep bf16 residency. fp8 has an accumulator format on no backend and follows
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
  (** Whether the explicit vector renderings ([Vectorized] loops) can operate at this {e compute}
      precision. f32 and f64 everywhere; fp16 additionally on CPU targets with native 16-bit
      arithmetic (gh-ocannl-516), which is exactly where {!compute_prec} leaves [Half_prec] alone. A
      storage precision this rejects can still be vectorized when {!compute_prec} maps it to one
      this accepts -- that is gh-ocannl-517's convert-on-load/store. *)

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
      [hardware_index = None]. Used when explicit vector emission ([vector_bytes]) is disabled or
      the loop body is ineligible for it. *)

  val vector_bytes : int
  (** Vector register width in bytes for explicit SIMD rendering of [Vectorized] loops via GCC/Clang
      vector extensions (the [Vectorized] codegen follow-up of gh-ocannl-164 /
      docs/proposals/watch-ocannl-README-md-347818d3.md): eligible loop bodies emit vector-typed
      loads, arithmetic and stores in [lanes = vector_bytes / element size] chunks plus a serial
      remainder loop, instead of relying on the compiler's auto-vectorizer (which e.g. cannot
      reassociate strict-FP reductions — the [Vectorized] retype carries that permission, like
      [Swap]). A recognized accumulation body renders as independent accumulator chains with a
      horizontal reduce at loop exit (gh-ocannl-468; [`Vec_extensions] only). [0] disables explicit
      emission ([vectorize_pragma] fallback). *)

  val vector_style : [ `Vec_extensions | `Packed_struct ]
  (** How eligible [Vectorized] loops emit explicit vector code when [vector_bytes > 0].
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
      serial rendering of one keeps a narrow accumulator (see [accum_pinned_to_storage_prec]). [0]
      disables the rendering: [Workgroup_reduce] loops render like [Workgroup] — hardware binding,
      or the serial fallback (which is the correct meaning of a recognized accumulation body on CPU
      backends). *)

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
      load: the caller records the call as {!Mma_intrinsics_ldmatrix} on that basis. An arm without
      such a load form must decline the call. *)

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

let codegen_capabilities (module Config : C_syntax_config) =
  {
    Backend_intf.supports_f64 = Config.supports_f64;
    accum_prec = Config.accum_prec;
    asynchronous_staging_copy = Option.is_some Config.async_copy;
  }

(** Whether [c] lies exactly halfway between two adjacent f32 values, so that narrowing it to f32 is
    a tie that IEEE-754 breaks to even -- and any decimal near but not equal to [c] would instead
    break by whichever side it fell on.

    Computed against the two neighbours rather than by masking mantissa bits, so that the f32
    subnormal range (where the retained-bit count shrinks, and a bit mask would need a second case)
    is covered by the same three lines. Values that overflow f32 have no neighbours to sit between
    and are not ties. *)
let is_f32_tie c =
  Float.is_finite c
  && (Float.(abs c = 0x1.ffffffp+127)
     (* The OVERFLOW midpoint, which the neighbour walk below cannot see: halfway between the
        largest finite f32 and the (unrepresentable) 2^128 that would follow it. IEEE-754 rounds it
        to even, and the even candidate is the one that overflows, so the host answers an infinity
        while a decimal spelling of it -- necessarily just under the midpoint, since the midpoint's
        exact decimal is longer -- answers the largest finite f32. The tie has exactly one
        magnitude, so naming it is complete rather than a first case of many. *)
     ||
     let f = Int32.float_of_bits (Int32.bits_of_float c) in
     Float.is_finite f
     && Float.(f <> c)
     &&
     let bits = Int32.bits_of_float f in
     (* One f32 step from [f] towards [c]: away from zero when they share a direction, towards it
        otherwise, since the bit pattern's order tracks magnitude rather than value. *)
     let step =
       if Float.(c > f) then if Float.(f >= 0.) then Int32.(bits + one) else Int32.(bits - one)
       else if Float.(f > 0.) then Int32.(bits - one)
       else Int32.(bits + one)
     in
     let g = Int32.float_of_bits step in
     Float.is_finite g && Float.(abs (c - f) = abs (g - c)))

(** A C source literal for the floating-point constant [c], as a {e double}-typed floating literal
    that parses back to exactly [c] on every C-family backend.

    Three properties, each of which [%.16g] alone gets wrong (gh-ocannl-623). Two of them are not
    C's alone — a debug dump wants a floating literal that round-trips for the same reasons a kernel
    does — so they live in {!Utils.decimal_float_literal}, which the IR printers share
    (gh-ocannl-713):

    - {b It is a floating literal, not an integer one.} In C the consequence is direct: [2.] came
      out as the integer literal [2], value-preserving in the cast contexts the constants happen to
      sit in today — but [-0.] came out as ["-0"], the integer zero, hence [+0.0] once cast, so a
      [-0.0] in host data silently reached kernels as [+0.0] wherever the constant fill inlines as
      scalar stores (gh-ocannl-615 fixed that one value). The rest of the class is latent rather
      than inert: an integer literal divides as an integer against another one, promotes as an
      integer, and would overflow its type once the digits outgrow [long long].
    - {b It round-trips.} That reaches emitted code for real: hosted constant inits (gh-ocannl-633)
      inline arbitrary host values as scalar stores, and [0.1 +. 0.2] at 16 digits is a
      {e different} double. Values that already round-trip at 16 digits keep their exact previous
      spelling, which is why no codegen golden moves except by the appended [.0].
    - {b The specials are spelled, not printed.} This one is C's own, and is why the shared
      rendering is called only after they are ruled out: it leaves them as [%.16g]'s words ([inf],
      [nan]), which no C dialect parses. [INFINITY] and [NAN] are C99 [math.h] macros that MSL also
      provides; the CUDA and HIP preludes define them under [#ifndef] since nvrtc/hiprtc supply no
      standard headers. [(-INFINITY)] is parenthesized so it cannot glue into [--INFINITY] beside a
      subtraction. A NaN's payload and sign do not survive this (no dialect spells a payload
      portably); nothing in OCANNL depends on them, and a bit-exact constant has
      {!Low_level.Constant_bits} available.

    Deliberately {e not} carried here: a precision suffix. The literal is always double-typed and
    the narrowing is [B.convert_precision]'s cast, which is a single rounding of the exact host
    double — the same conversion the host performs when it stores the value. An [f]-suffixed decimal
    would instead round the decimal straight to float. The cast is present for every non-double
    target: [Ops.c_convert_precision] and each backend's override return [("", "")] only when [from]
    and [to_] are the same precision.

    That last argument has a hole, and {!is_f32_tie} is what fills it: the cast only {e is} a
    narrowing where the dialect has a [double]. MSL does not, so Metal rounds the decimal itself, at
    parse time, to f32. A decimal that round-trips as a double is not thereby exact, so the two
    readings — round [c] to f32, versus round a decimal near [c] to f32 — can differ, and they
    differ exactly when [c] sits on an f32 tie: the host takes ties-to-even while the dialect
    follows whichever side the decimal happens to land on. Which digit count is emitted then decides
    the value, so this is not a hazard the retry above introduced but one it {e moves}. A
    hexadecimal literal takes it away instead of moving it: it is exact by construction, so there is
    no parse-time rounding for either reading to disagree about, and both round [c] itself. It is
    spelled only for ties — a handful of values that were otherwise a coin toss — so the ordinary
    constant keeps its readable decimal, and C99, CUDA, HIP and MSL all accept the form. *)
let c_float_literal c =
  if Float.(c = infinity) then "INFINITY"
  else if Float.(c = neg_infinity) then "(-INFINITY)"
  else if Float.is_nan c then "NAN"
  else if is_f32_tie c then Printf.sprintf "%h" c
  else Utils.decimal_float_literal c

(** The C-family rendering of a binary operation, from {!Ops.binop_c_syntax}: prefix, first operand,
    infix, second operand (breaking after the operator), suffix.

    Outside {!Pure_C_config} because the GPU backends, which shadow [binop_syntax] wholesale (most
    ops need target-specific intrinsics or precision bridging), delegate here for the ops that are
    spelled the same in C, CUDA, HIP and MSL -- the comparisons and the logical connectives. Those
    then have one spelling ({!Ops.binop_c_syntax}) and one layout (here) across all backends,
    instead of a copy per backend. *)
let default_binop_syntax prec op v1 v2 =
  let op_prefix, op_infix, op_suffix = Ops.binop_c_syntax prec op in
  let open PPrint in
  group
    (string op_prefix ^^ v1 ^^ string op_infix
    ^^ ifflat (space ^^ v2) (nest 2 (break 1 ^^ v2))
    ^^ string op_suffix)

(** The RNG binops -- the two Threefry variants and the per-lane uniform conversion -- rendered as a
    call to the builtin of that name. Every C-family backend provides the same three builtins
    ([builtins.c], {!Builtins_cuda}, {!Builtins_metal}) under the same precision contract: the
    Threefry ops produce a uint4x32 block, and the lane conversion consumes one to produce the
    target precision, so it is the one binop that rejects uint4x32 (its builtin already yields the
    target precision, bypassing the generic bfloat16/fp8 compute-in-single wrapping).

    Backends pass their own two-argument [call] renderer -- the layouts differ in where a line break
    may fall -- and the [backend] name for the diagnostics. [op] must be one of the three ops above.
*)
let rng_binop_syntax ~backend ~call prec op =
  let uint4x32_only ~op_name ~builtin =
    match prec with
    | Ops.Uint4x32_prec _ -> call builtin
    | _ ->
        raise
        @@ Utils.User_error
             (Printf.sprintf "%s backend: %s requires target precision to be uint4x32, but got %s"
                backend op_name (Ops.prec_string prec))
  in
  match op with
  | Ops.Threefry4x32_crypto ->
      uint4x32_only ~op_name:"Threefry4x32_crypto" ~builtin:"arrayjit_threefry4x32_crypto"
  | Ops.Threefry4x32_light ->
      uint4x32_only ~op_name:"Threefry4x32_light" ~builtin:"arrayjit_threefry4x32_light"
  | Ops.Uint4x32_to_prec_uniform_lane -> (
      match prec with
      | Ops.Uint4x32_prec _ ->
          raise
          @@ Utils.User_error
               (Printf.sprintf
                  "%s backend: Uint4x32_to_prec_uniform_lane not supported for Uint4x32 target \
                   precision"
                  backend)
      | _ -> call ("uint4x32_to_" ^ Ops.prec_string prec ^ "_uniform_lane"))
  | _ -> invalid_arg "C_syntax.rng_binop_syntax: not a random number generation operator"

(** All maximal identifier-like substrings of [s] -- a run of alphanumerics and underscores starting
    at a letter or underscore. This decomposes a composite rendering like ["(fabsf(floorf("] into
    [["fabsf"; "floorf"]] rather than the concatenation ["fabsffloorf"].

    A run whose first character is preceded by a digit or a [.] is the tail of a numeric literal
    ([f] in ["0.0f"], [h] in ["1.0h"]), not a name, and is skipped -- otherwise every single-letter
    literal suffix a backend emits would be reserved. *)
let extract_idents s =
  let n = String.length s in
  let result = ref [] in
  let i = ref 0 in
  while !i < n do
    if Char.is_alpha s.[!i] || Char.equal s.[!i] '_' then begin
      let j = ref !i in
      while !j < n && (Char.is_alphanum s.[!j] || Char.equal s.[!j] '_') do
        Int.incr j
      done;
      let literal_suffix = !i > 0 && (Char.is_digit s.[!i - 1] || Char.equal s.[!i - 1] '.') in
      if not literal_suffix then result := String.sub s ~pos:!i ~len:(!j - !i) :: !result;
      i := !j
    end
    else Int.incr i
  done;
  !result

(** The precisions a backend's renderers are exercised over -- by {!op_syntax_idents} and by
    {!operand_conditionality_violations}. The operator sweeps use [Ops]' derived enumerations
    ([Ops.all_of_binop] and friends), so a newly added operator joins both checks automatically;
    [Ops.prec] cannot be derived the same way -- its constructors carry the phantom-typed
    [precision] witness -- so the exhaustive match below stands in for it: adding a precision is a
    build error here, and the fix is to extend this list (or to leave the precision out
    deliberately, as [Void_prec] is, since every renderer rejects it). *)
let all_precs =
  Ops.[ byte; uint16; int32; uint32; int64; uint64; uint4x32; half; bfloat16; fp8; single; double ]

let _all_precs_is_complete : Ops.prec -> unit = function
  | Void_prec | Byte_prec _ | Uint16_prec _ | Int32_prec _ | Uint32_prec _ | Int64_prec _
  | Uint64_prec _ | Uint4x32_prec _ | Half_prec _ | Bfloat16_prec _ | Fp8_prec _ | Single_prec _
  | Double_prec _ ->
      ()

(** Every function and type name a backend's operator rendering can emit, obtained by rendering each
    (precision, operator) pair over a placeholder operand and harvesting the identifiers.

    Rendering the syntax functions is what makes this correct per backend: reading the names off
    {!Ops.unop_c_syntax} instead would describe *C*, and the GPU backends shadow those functions
    wholesale. MSL spells [Tanh_approx] as [tanh] where C spells it [tanhf], so the C-derived list
    left ["tanh"] free for a tensor node -- and {!Tensor.unop}'s [~op_label] makes that an ordinary
    name, minted by every [Operation.tanh]. The Metal kernel then declared
    [device float *__restrict tanh] and the call on the next line resolved to that pointer
    (gh-ocannl-553). The same holds for [exp], [log], [sqrt], [sin], [cos], [trunc]: unsuffixed in
    MSL, suffixed in C.

    Backends reject some (precision, operator) pairs by raising, either while selecting the renderer
    or while applying it; those pairs contribute no names. Any escaping exception is swallowed
    rather than failing the compilation this list only guards. *)
let op_syntax_idents ~ternop_syntax ~binop_syntax ~unop_syntax ~vec_unop_syntax ~convert_precision =
  let names = ref (Set.empty (module String)) in
  let add_string s = List.iter (extract_idents s) ~f:(fun name -> names := Set.add !names name) in
  let arg = PPrint.string "?" in
  let add_doc f =
    try
      let buf = Buffer.create 256 in
      (* A width no rendering reaches keeps identifiers off line boundaries. *)
      PPrint.ToBuffer.pretty 1.0 1_000_000 buf (f ());
      add_string (Buffer.contents buf)
    with _ -> ()
  in
  List.iter all_precs ~f:(fun prec ->
      List.iter Ops.all_of_ternop ~f:(fun op ->
          add_doc (fun () -> ternop_syntax prec op arg arg arg));
      List.iter Ops.all_of_binop ~f:(fun op -> add_doc (fun () -> binop_syntax prec op arg arg));
      List.iter Ops.all_of_unop ~f:(fun op -> add_doc (fun () -> unop_syntax prec op arg));
      List.iter Ops.all_of_vec_unop ~f:(fun op -> add_doc (fun () -> vec_unop_syntax prec op arg));
      List.iter all_precs ~f:(fun to_ ->
          try
            let prefix, suffix = convert_precision ~from:prec ~to_ in
            add_string prefix;
            add_string suffix
          with _ -> ()));
  Set.to_list !names

(** What a backend's operator renderings actually emit, checked against the operand-evaluation
    contract they are supposed to implement ({!Ops.binop_conditionality} /
    {!Ops.ternop_conditionality}) -- returns one message per disagreement, [[]] when they agree
    (gh-ocannl-582).

    The classifier is what {!Low_level.affine_accesses} and both {!Cost_model} walks consume, and
    the renderers are the ground truth it describes; checking rather than consulting is deliberate,
    since a renderer cannot ask "am I allowed to short-circuit here" -- it either emits a
    conditional or it does not. Each operator is rendered over distinguishable placeholder operands
    at every precision, and the emitted text is read for the only C-family constructs that can skip
    an operand: [?:], [&&] and [||]. So:

    - a projection ([Only_first] / [Only_second]) must not mention its discarded operand -- every
      backend rejects one outright, which satisfies this trivially, and {!C_syntax.pp_scalar} emits
      the selected operand alone before [binop_syntax] is ever reached;
    - [Gated_second] must place one of those constructs before its second operand;
    - [Both_operands] / [All_three] must emit none of them anywhere -- an [Ops.Max] spelled
      [(a > b ? a : b)] would evaluate [a] twice and [b] conditionally, and the cost model's floor
      would keep charging both;
    - [Cond_and_one_arm] must put its arms on the two sides of a [?:] alternative, which is what
      rules out MSL's [select] (a call: it evaluates both arms, so a range guard lowered to [Where]
      would still perform its out-of-range read).

    The operator sweep is [Ops]' derived enumeration, so an operator added to {!Ops.binop} or
    {!Ops.ternop} is checked here without anyone remembering to list it. A (precision, operator)
    pair the backend rejects by raising contributes nothing, as in {!op_syntax_idents}. *)
let operand_conditionality_violations ~ternop_syntax ~binop_syntax =
  let violations = ref [] in
  let report what msg = violations := Printf.sprintf "%s: %s" what msg :: !violations in
  let a1 = PPrint.string "$1" and a2 = PPrint.string "$2" and a3 = PPrint.string "$3" in
  let render f =
    try
      let buf = Buffer.create 256 in
      (* A width no rendering reaches keeps the placeholders off line boundaries. *)
      PPrint.ToBuffer.pretty 1.0 1_000_000 buf (f ());
      Some (Buffer.contents buf)
    with _ -> None
  in
  let at s ph = String.substr_index s ~pattern:ph in
  let gated_in s =
    String.exists s ~f:(Char.equal '?')
    || String.is_substring s ~substring:"&&"
    || String.is_substring s ~substring:"||"
  in
  let gated_before s pos = gated_in (String.prefix s pos) in
  List.iter all_precs ~f:(fun prec ->
      let at_prec op_name = Printf.sprintf "%s at %s" op_name (Ops.prec_string prec) in
      List.iter Ops.all_of_binop ~f:(fun op ->
          match render (fun () -> binop_syntax prec op a1 a2) with
          | None -> ()
          | Some s -> (
              let what = at_prec (Ops.binop_cd_fallback_syntax op) in
              let i1 = at s "$1" and i2 = at s "$2" in
              match Ops.binop_conditionality op with
              | Ops.Only_first ->
                  if Option.is_some i2 then
                    report what
                      "classified as a projection onto the first operand, but the rendering \
                       mentions the second"
              | Ops.Only_second ->
                  if Option.is_some i1 then
                    report what
                      "classified as a projection onto the second operand, but the rendering \
                       mentions the first"
              | (Ops.Both_operands | Ops.Gated_second) as cond -> (
                  if Option.is_none i1 then report what "the first operand is not rendered";
                  match (cond, i2) with
                  | _, None -> report what "the second operand is not rendered"
                  | Ops.Gated_second, Some i2 ->
                      if not (gated_before s i2) then
                        report what
                          "classified as gating its second operand, but the rendering reaches it \
                           without a [?], [&&] or [||]"
                  | _, Some _ ->
                      if gated_in s then
                        report what
                          "classified as always evaluating both operands, but the rendering \
                           short-circuits ([?], [&&] or [||])")));
      List.iter Ops.all_of_ternop ~f:(fun op ->
          match render (fun () -> ternop_syntax prec op a1 a2 a3) with
          | None -> ()
          | Some s -> (
              let what = at_prec (Ops.ternop_cd_syntax op) in
              match (at s "$1", at s "$2", at s "$3") with
              | None, _, _ | _, None, _ | _, _, None -> report what "not every operand is rendered"
              | Some _, Some i2, Some i3 -> (
                  match Ops.ternop_conditionality op with
                  | Ops.All_three ->
                      if gated_in s then
                        report what
                          "classified as always evaluating all three operands, but the rendering \
                           short-circuits ([?], [&&] or [||])"
                  | Ops.Cond_and_one_arm ->
                      if not (gated_before s i2) then
                        report what
                          "classified as evaluating one arm, but the rendering reaches the first \
                           arm without a [?]";
                      if
                        i3 <= i2
                        || not
                             (String.is_substring
                                (String.sub s ~pos:i2 ~len:(i3 - i2))
                                ~substring:":")
                      then
                        report what
                          "classified as evaluating one arm, but its arms are not the two sides of \
                           a [?:] alternative -- a call form (MSL's [select]) evaluates both"))));
  List.rev !violations

(** The names defined by a backend's builtins table (the keys of its
    [(name, definition, dependencies)] entries), for its [ident_blacklist]. A node taking one of
    these names both shadows the definition and, since {!filter_and_prepend_builtins} selects
    entries by searching the rendered kernel for their key, drags the definition into a kernel that
    never calls it. *)
let builtin_idents builtins = List.map builtins ~f:(fun (key, _, _) -> key)

(** The words the C language itself reserves, plus the scaffolding names this module's rendering
    emits unconditionally. Shared by every C-family backend through {!Pure_C_config}, and by
    {!C_syntax.kernel_ident}: a routine and a tensor node are both plain identifiers in the emitted
    source, so they are unsafe on exactly the same words.

    [asm] is in the list even though C89/C99 reserve it only as a common extension: every compiler
    OCANNL emits for (gcc, clang, nvrtc, hiprtc, the Metal front end) treats it as a keyword, which
    is what made [Block_comment "asm"] emit [void asm(] and fail as an "internal" codegen bug
    (gh-ocannl-686). *)
let c_keywords =
  [
    (* C89 keywords *)
    "auto";
    "break";
    "case";
    "char";
    "const";
    "continue";
    "default";
    "do";
    "double";
    "else";
    "enum";
    "extern";
    "float";
    "for";
    "goto";
    "if";
    "int";
    "long";
    "register";
    "return";
    "short";
    "signed";
    "sizeof";
    "static";
    "struct";
    "switch";
    "typedef";
    "union";
    "unsigned";
    "void";
    "volatile";
    "while";
    (* C99 additions *)
    "inline";
    "restrict";
    "_Bool";
    "_Complex";
    "_Imaginary";
    (* C11 additions *)
    "_Alignas";
    "_Alignof";
    "_Atomic";
    "_Generic";
    "_Noreturn";
    "_Static_assert";
    "_Thread_local";
    (* Keywords every compiler in the toolchain set accepts as an extension, or C23 promotes *)
    "asm";
    "typeof";
    (* Scaffolding names emitted by generated code that must not clash with variable names *)
    "log_file";
    "log_file_name";
    "uint32_t";
    "uint64_t";
  ]

(** Macro names the C-family preludes' unconditional includes define ([<stdio.h>], [<math.h>],
    [<string.h>], [<stdlib.h>], [<stdint.h>]). Macros are the half of the standard library that
    NOTHING can shadow -- a parameter or a local named [NAN] expands mid-declaration and the kernel
    fails to parse -- so these constrain every identifier, tensor-node names included, and belong in
    the shared [ident_blacklist] (gh-ocannl-686). *)
let c_stdlib_macros =
  [
    "NULL";
    "EOF";
    "BUFSIZ";
    "SEEK_SET";
    "SEEK_CUR";
    "SEEK_END";
    "FILENAME_MAX";
    "FOPEN_MAX";
    "TMP_MAX";
    "L_tmpnam";
    "stdin";
    "stdout";
    "stderr";
    "NAN";
    "INFINITY";
    "HUGE_VAL";
    "HUGE_VALF";
    "HUGE_VALL";
    "M_PI";
    "M_E";
    "M_SQRT2";
    "FP_NAN";
    "FP_INFINITE";
    "FP_ZERO";
    "FP_SUBNORMAL";
    "FP_NORMAL";
    "FP_ILOGB0";
    "FP_ILOGBNAN";
    "MATH_ERRNO";
    "MATH_ERREXCEPT";
    "fpclassify";
    "isfinite";
    "isinf";
    "isnan";
    "isnormal";
    "signbit";
    "isgreater";
    "isgreaterequal";
    "isless";
    "islessequal";
    "islessgreater";
    "isunordered";
    "RAND_MAX";
    "EXIT_SUCCESS";
    "EXIT_FAILURE";
    "MB_CUR_MAX";
    "INT8_MAX";
    "INT16_MAX";
    "INT32_MAX";
    "INT64_MAX";
    "INT8_MIN";
    "INT16_MIN";
    "INT32_MIN";
    "INT64_MIN";
    "UINT8_MAX";
    "UINT16_MAX";
    "UINT32_MAX";
    "UINT64_MAX";
    "INTMAX_MAX";
    "INTMAX_MIN";
    "UINTMAX_MAX";
    "SIZE_MAX";
    "PTRDIFF_MAX";
    "PTRDIFF_MIN";
    "assert";
    "offsetof";
    "errno";
    "va_start";
    "va_arg";
    "va_end";
    "va_copy";
  ]

(** Function and type names the same unconditional includes declare. Unlike the macros above these
    are ordinary file-scope declarations, and C scoping makes the two cases genuinely different:

    - a routine DEFINITION takes file scope, so [void printf(...)] after [<stdio.h>] is a
      conflicting declaration and [void exp(...)] after [<math.h>] likewise -- an error regardless
      of whether the kernel ever calls either;
    - a parameter or a local legally SHADOWS them, so a tensor node named [exp] is well-formed C and
      always has been.

    So this table constrains routine names only ({!C_syntax.kernel_ident}) and is deliberately kept
    out of [ident_blacklist]: adding it there would rename tensor nodes -- [Tensor.unop]'s
    [~op_label]s are exactly these words ([exp], [log], [sqrt], [tanh]) -- churning goldens to
    prevent a collision that cannot happen. The names a rendering actually CALLS are a separate
    concern already covered by {!op_syntax_idents}, which is what stops a node from shadowing a
    callee the same kernel invokes. *)
let c_stdlib_idents =
  [
    "FILE";
    "fpos_t";
    "fopen";
    "freopen";
    "fclose";
    "fflush";
    "setbuf";
    "setvbuf";
    "fread";
    "fwrite";
    "fprintf";
    "printf";
    "sprintf";
    "snprintf";
    "vprintf";
    "vfprintf";
    "vsprintf";
    "vsnprintf";
    "fscanf";
    "scanf";
    "sscanf";
    "vscanf";
    "vfscanf";
    "vsscanf";
    "fgetc";
    "getc";
    "getchar";
    "fgets";
    "fputc";
    "putc";
    "putchar";
    "fputs";
    "puts";
    "ungetc";
    "fseek";
    "ftell";
    "rewind";
    "fgetpos";
    "fsetpos";
    "remove";
    "rename";
    "tmpfile";
    "tmpnam";
    "clearerr";
    "feof";
    "ferror";
    "perror";
    "acos";
    "asin";
    "atan";
    "atan2";
    "cos";
    "sin";
    "tan";
    "acosh";
    "asinh";
    "atanh";
    "cosh";
    "sinh";
    "tanh";
    "exp";
    "exp2";
    "expm1";
    "frexp";
    "ldexp";
    "log";
    "log10";
    "log1p";
    "log2";
    "logb";
    "ilogb";
    "modf";
    "scalbn";
    "scalbln";
    "cbrt";
    "fabs";
    "hypot";
    "pow";
    "sqrt";
    "erf";
    "erfc";
    "lgamma";
    "tgamma";
    "ceil";
    "floor";
    "nearbyint";
    "rint";
    "lrint";
    "llrint";
    "round";
    "lround";
    "llround";
    "trunc";
    "fmod";
    "remainder";
    "remquo";
    "copysign";
    "nan";
    "nextafter";
    "nexttoward";
    "fdim";
    "fmax";
    "fmin";
    "fma";
    "acosf";
    "asinf";
    "atanf";
    "atan2f";
    "cosf";
    "sinf";
    "tanf";
    "coshf";
    "sinhf";
    "tanhf";
    "expf";
    "exp2f";
    "logf";
    "log10f";
    "log2f";
    "fabsf";
    "hypotf";
    "powf";
    "sqrtf";
    "cbrtf";
    "ceilf";
    "floorf";
    "roundf";
    "truncf";
    "fmodf";
    "copysignf";
    "fmaxf";
    "fminf";
    "fmaf";
    "erff";
    "tgammaf";
    "lgammaf";
    "rintf";
    "memcpy";
    "memmove";
    "memset";
    "memcmp";
    "memchr";
    "strcpy";
    "strncpy";
    "strcat";
    "strncat";
    "strcmp";
    "strncmp";
    "strcoll";
    "strxfrm";
    "strchr";
    "strrchr";
    "strspn";
    "strcspn";
    "strpbrk";
    "strstr";
    "strtok";
    "strlen";
    "strerror";
    "malloc";
    "calloc";
    "realloc";
    "free";
    "aligned_alloc";
    "abort";
    "exit";
    "_Exit";
    "atexit";
    "system";
    "getenv";
    "bsearch";
    "qsort";
    "abs";
    "labs";
    "llabs";
    "div";
    "ldiv";
    "lldiv";
    "rand";
    "srand";
    "atoi";
    "atol";
    "atoll";
    "atof";
    "strtol";
    "strtoul";
    "strtoll";
    "strtoull";
    "strtod";
    "strtof";
    "strtold";
    "div_t";
    "ldiv_t";
    "lldiv_t";
    "size_t";
    "ptrdiff_t";
    "wchar_t";
    "intptr_t";
    "uintptr_t";
    "intmax_t";
    "uintmax_t";
    "int8_t";
    "int16_t";
    "int32_t";
    "int64_t";
    "uint8_t";
    "uint16_t";
  ]

(** The words C++ reserves on top of {!c_keywords}. CUDA, HIP and MSL are all C++ dialects -- their
    kernels are parsed by a C++ front end even where the emitted body is plain C -- so their
    backends extend the blacklist with this table. Several entries are entirely plausible tensor
    labels or routine names ([class], [new], [operator], [bool], [this], [template]), and a
    collision there is the same misattributed "bug in OCANNL" failure as gh-ocannl-686's [asm]. *)
let cpp_keywords =
  [
    "alignas";
    "alignof";
    "and";
    "and_eq";
    "bitand";
    "bitor";
    "bool";
    "catch";
    "char8_t";
    "char16_t";
    "char32_t";
    "class";
    "compl";
    "concept";
    "consteval";
    "constexpr";
    "constinit";
    "const_cast";
    "co_await";
    "co_return";
    "co_yield";
    "decltype";
    "delete";
    "dynamic_cast";
    "explicit";
    "export";
    "false";
    "friend";
    "mutable";
    "namespace";
    "new";
    "noexcept";
    "not";
    "not_eq";
    "nullptr";
    "operator";
    "or";
    "or_eq";
    "private";
    "protected";
    "public";
    "reinterpret_cast";
    "requires";
    "static_assert";
    "static_cast";
    "template";
    "this";
    "thread_local";
    "throw";
    "true";
    "try";
    "typeid";
    "typename";
    "using";
    "virtual";
    "wchar_t";
    "xor";
    "xor_eq";
  ]

(** {!C_syntax_config.full_printf_support} for a dialect whose [printf] does support [%g] — every
    backend except Metal, whose MSL [printf] does not. Such a backend still gives the scaled-integer
    rendering up when asked for backend uniformity, so that its routine logs read the same as
    Metal's: the setting is what decides, and it is read here once rather than restated per backend.
*)
let printf_support_unless_uniform () =
  not @@ Utils.get_global_flag ~default:false ~arg_name:"prefer_backend_uniformity"

module Pure_C_config (Input : sig
  val procs : Low_level.t array
  val full_printf_support : bool
end) =
struct
  let procs = Input.procs
  let main_kernel_prefix = ""
  let kernel_prep_line = ""
  let buffer_prefix = ""
  let buffer_suffix = fun ~pos:_ -> ""

  (* Signed index arithmetic (docs/proposals/signed-index-precision.md); the width tracks
     [Ops.index_prec ()]. *)
  let arg_int_prefix = if Utils.settings.large_models then "const int64_t " else "const int32_t "
  let loop_index_type = if Utils.settings.large_models then "int64_t " else "int32_t "
  let extra_args = []
  let typ_of_prec = Ops.c_typ_of_prec
  let supports_f64 = true
  let vec_typ_of_prec = Ops.c_vec_typ_of_prec
  let ptr_param_style = `Per_param

  (* Plain C backends bind no hardware axes: annotated loops fall back to serial [for] loops (sound
     absent barriers), and barriers / shared placements are compile-time errors. *)
  let hardware_index ~kind:_ ~slot:_ = None
  let barrier_syntax = None
  let async_copy = None
  let parallel_grid_syntax = `None
  let parallel_grid_chunks = 1
  let vector_bytes = 0
  let vector_style = `Vec_extensions
  let shared_decl_prefix = None
  let restrict_keyword = Some "restrict"
  let volatile_serial_accumulation = false

  (* Clang defines both [__clang__] and [__GNUC__], so test [__clang__] first. *)
  let vectorize_pragma =
    [
      "#if defined(__clang__)";
      "#pragma clang loop vectorize(enable) interleave(enable)";
      "#elif defined(__GNUC__)";
      "#pragma GCC ivdep";
      "#endif";
    ]

  let aligned_local_attr = Some (Printf.sprintf "__attribute__((aligned(%d)))" Ops.buffer_alignment)

  (* No shuffle intrinsics on plain C backends: a [Workgroup_reduce] accumulation loop renders as
     the serial fallback, which is exactly its serial meaning. *)
  let warp_size = 0

  (* No tile-MMA units on plain C backends: [Tile_mma] renders its scalar fallback under the [lane
     == 0] guard. *)
  let mma_syntax = None
  let mma_fragment_syntax = None
  let float_log_style = if Input.full_printf_support then "%g" else "%de-3"

  let styled_log_arg doc =
    if Input.full_printf_support then doc
    else
      let open PPrint in
      string "(int)(" ^^ doc ^^ string " * 1000.0)"

  (* [long long] is the widest type C's variadic promotions name portably, and [%lld] its conversion
     -- available on glibc, musl, the MSVC runtime, nvrtc and hiprtc alike, with no [<inttypes.h>]
     and no [PRId64] string-concatenation dance. The cast is a no-op where [loop_index_type] is
     already [long long] (CUDA, HIP) and a widening no-op where it is [int64_t] (the C backends). *)
  let log_index_arg doc =
    if Utils.settings.large_models then ("%lld", PPrint.(string "(long long)" ^^ doc))
    else ("%d", doc)

  let ternop_syntax prec op v1 v2 v3 =
    let op_prefix, op_infix1, op_infix2, op_suffix = Ops.ternop_c_syntax prec op in
    let open PPrint in
    group
      (string op_prefix ^^ v1 ^^ string op_infix1
      ^^ ifflat (space ^^ v2) (nest 2 (break 1 ^^ v2))
      ^^ string op_infix2
      ^^ ifflat (space ^^ v3) (nest 2 (break 1 ^^ v3))
      ^^ string op_suffix)

  let binop_syntax = default_binop_syntax

  let unop_syntax prec op v =
    let op_prefix, op_suffix = Ops.unop_c_syntax prec op in
    let open PPrint in
    group (string op_prefix ^^ v ^^ string op_suffix)

  let vec_unop_syntax prec op v =
    let op_prefix, op_suffix = Ops.vec_unop_c_syntax prec op in
    let open PPrint in
    group (string op_prefix ^^ v ^^ string op_suffix)

  let convert_precision = Ops.c_convert_precision

  (* Compute where you store. The backends that override this are the ones without native narrow
     arithmetic; see the signature. *)
  let compute_prec prec = prec

  (* Accumulate where you compute. Every backend overrides at least one of the pair (see the
     signature's coupling note); this default only serves a hypothetical backend whose reductions
     have no wider residency than its arithmetic. *)
  let accum_prec = compute_prec
  let vector_prec_ok = function Ops.Single_prec _ | Ops.Double_prec _ -> true | _ -> false

  (* The names the *language* reserves and the scaffolding this module emits. The names an operator
     rendering emits are not restated here: {!C_syntax} derives them from the backend's own syntax
     functions ({!op_syntax_idents}), so an override cannot drift out of the list. What [c_names]
     adds is the plain-C rendering's share of that -- a floor every C-family backend keeps even
     where it overrides the op to something else, so that a node's code name does not depend on
     which arms a backend happens to shadow. *)
  let ident_blacklist =
    let c_names =
      op_syntax_idents ~ternop_syntax ~binop_syntax ~unop_syntax ~vec_unop_syntax ~convert_precision
    in
    c_keywords @ c_stdlib_macros @ c_names

  let kernel_log_param = Some ("const char*", "log_file_name")
  let log_involves_file_management = true

  let for_log_trace_tree =
    Utils.get_global_flag ~default:false ~arg_name:"debug_log_to_stream_files"

  let pp_log_statement ~log_param_c_expr_doc:_ ~base_message_literal ~args_docs =
    let open PPrint in
    let log_file_check =
      match kernel_log_param with
      | Some (_, lname) -> string ("if (" ^ lname ^ " && log_file) ")
      | None ->
          string "if (log_file) " (* Should not happen if log_involves_file_management is true *)
    in
    let base_message_literal =
      let with_ = if for_log_trace_tree then "$" else "\\n" in
      let res = String.substr_replace_all base_message_literal ~pattern:"\n" ~with_ in
      if for_log_trace_tree && String.is_suffix res ~suffix:"$" then
        String.drop_suffix res 1 ^ "\\n"
      else res
    in
    log_file_check
    ^^ group
         (string "fprintf(log_file, "
         ^^ dquotes (string base_message_literal)
         ^^ (if List.is_empty args_docs then empty
             else comma ^^ nest 4 (break 1 ^^ separate (comma ^^ break 1) args_docs))
         ^^ rparen ^^ semi)
end

module C_syntax (B : C_syntax_config) = struct
  (* The backend's renderings must implement the operand-evaluation contract that
     [Low_level.affine_accesses] and both [Cost_model] walks read off {!Ops.binop_conditionality} /
     {!Ops.ternop_conditionality} -- checked here, once per functor application, rather than in a
     test, because the GPU backends' syntax configs are sealed inside their [Impl] modules and only
     reachable on the hardware that has them (gh-ocannl-582). A mismatch is an internal
     inconsistency, not a user error: fix the classifier or the rendering. *)
  let () =
    match
      operand_conditionality_violations ~ternop_syntax:B.ternop_syntax ~binop_syntax:B.binop_syntax
    with
    | [] -> ()
    | violations ->
        invalid_arg
          ("C_syntax: operator renderings disagree with Ops.binop_conditionality / \
            Ops.ternop_conditionality:\n"
          ^ String.concat ~sep:"\n" violations)

  (* Identifiers a tensor node's code name must not take: what the backend declares reserved (the
     language's keywords, its builtins, its intrinsic globals) plus what its own operator rendering
     actually emits -- see {!op_syntax_idents} for why the second half has to be derived from [B]
     and not from the C spellings. *)
  let ident_blacklist =
    B.ident_blacklist
    @ op_syntax_idents ~ternop_syntax:B.ternop_syntax ~binop_syntax:B.binop_syntax
        ~unop_syntax:B.unop_syntax ~vec_unop_syntax:B.vec_unop_syntax
        ~convert_precision:B.convert_precision

  let get_ident =
    Low_level.get_ident_within_code ~no_dots:true ~blacklist:ident_blacklist @@ B.procs

  (* What a ROUTINE name must avoid: everything a node name must ({!ident_blacklist}) plus the
     standard-library functions and types the preludes' unconditional includes declare. The extra
     table applies here and not to node names because a routine definition takes FILE scope, where
     [void exp(...)] after [<math.h>] is a conflicting declaration, while a parameter or local named
     [exp] merely shadows it and is well-formed C (gh-ocannl-686 review round 1). *)

  (** [kernel_ident name] is [name] made safe to emit as a kernel function's C identifier
      (gh-ocannl-686).

      Routine names come from user-facing surfaces -- an {!Assignments.Block_comment} label, the
      [~name] argument of [Context.compile] and of the autotune drop-ins (gh-ocannl-669) -- and
      reached the emitted [void <name>(] unexamined. A name that happens to be a reserved word or a
      builtin therefore produced a source the backend's compiler rejects, and every backend reports
      that rejection as "this is a bug in OCANNL", naming a generated file rather than the name that
      caused it.

      Tensor nodes and scope locals need no equivalent: a node's code name is minted by
      {!Low_level.get_ident_within_code} with [ident_blacklist] pre-seeded, so a colliding label is
      forced into the disambiguated [n<id>_<label>] form, and every local is minted with a
      [v<scope>_] / [wred_] / [__rmw_] prefix. Routine names were the one identifier class entering
      the emitted source verbatim.

      The scheme is deterministic and, on a name that is already a safe C identifier, the identity
      -- so nothing about a non-colliding routine changes: not its emitted symbol, not its
      schedule-cache identity, not its generated-source goldens. On a colliding one:

      - characters C does not admit in an identifier become [_], and a name starting with a digit
        (or empty after that pass) gains a [k_] prefix, so the result is always well-formed;
      - a result still equal to a reserved word or builtin gains a [__] suffix, repeatedly until it
        is clear of [ident_blacklist] -- which terminates, the list being finite.

      The name a routine is KNOWN by is untouched: [Context.routine]'s [name] field, the [.cd] and
      [.ll] sources (written by {!Backends} before any backend sees the code), and the calibration
      rows all keep what the caller asked for. Only the C-family artifacts downstream of this
      function -- the emitted symbol, the [.c] / [.cu] / [.hip] / [.metal] source and what the
      backend looks the symbol up by -- carry the mangled spelling, and they carry it consistently.

      The table is {!routine_ident_blacklist}: everything the node names avoid ([ident_blacklist] --
      the language's keywords, C plus C++ for the GPU dialects, the backend's intrinsic globals, its
      builtins table's keys, and the names its own operator renderings emit) plus
      {!c_stdlib_idents}, the standard-library functions and types the preludes' unconditional
      includes declare, which constrain a file-scope definition and not a parameter.

      Two things it deliberately does not do. Uniqueness {e among} the routines of one batch is not
      its business and never was -- two routines given the same name collide as duplicate C symbols
      with or without mangling. And it cannot see what a BACKEND-SPECIFIC header declares beyond the
      C standard floor ([<cuda_fp16.h>], [metal_stdlib]'s namespace): those names are covered only
      where a rendering emits them, so a routine named after an unused vendor intrinsic still
      reaches the vendor compiler. That failure is an ordinary compile error naming the conflict,
      which is the state this function put the reserved words in -- not the misattributed "bug in
      OCANNL" the issue was about. *)
  let routine_ident_blacklist = ident_blacklist @ c_stdlib_idents

  let kernel_ident name =
    let sanitized =
      String.map name ~f:(fun c -> if Char.is_alphanum c || Char.equal c '_' then c else '_')
    in
    let sanitized =
      if String.is_empty sanitized || Char.is_digit sanitized.[0] then "k_" ^ sanitized
      else sanitized
    in
    let rec avoid s =
      if List.mem routine_ident_blacklist s ~equal:String.equal then avoid (s ^ "__") else s
    in
    avoid sanitized

  (* {3 Storage precision vs. compute precision (gh-ocannl-517)}

     A [Low_level] precision is always a {e storage} precision: it comes off a tensor node, and
     names the element type of a buffer, a stack array, or a scope-local scalar. The precision an
     expression is {e rendered} at is a different thing -- it decides which operator spellings and
     which conversions appear -- and on backends without native narrow arithmetic the two diverge.
     [comp_prec] is the one-way map between them, and the rule is: declarations and buffer element
     types take the storage precision, rendered arithmetic takes [comp_prec] of it.

     A corollary (gh-ocannl-639): a reduction accumulator RESIDES at [acc_prec] across its whole
     reduction nest and narrows once at the store, in every rendering — virtual scopes
     ([scope_prec_of]), [try_vectorize_reduce]'s epilogue, [try_register_tile]'s C-tile, and the
     plain serial fallback ([try_localize_serial_reduce]) — so the effective accumulation width is
     the numerics policy's, never an artifact of which rendering or schedule ran. On the CPU
     backends [acc_prec] IS [comp_prec]; the GPU backends resolve accumulators wider than their
     (native, identity) compute precision where their tensor-unit triples do (gh-ocannl-663). *)

  let comp_prec = B.compute_prec
  let acc_prec = B.accum_prec

  (* The signature's coupling contract: an accumulator resides at least as wide as the arithmetic
     that feeds it. The failure mode this catches is an [include]-time stale pairing — a backend
     overriding [compute_prec] without restating [accum_prec] keeps the default bound to the
     PRE-override compute precision, which would silently narrow every reduction accumulator. *)
  let () =
    List.iter [ Ops.half; Ops.bfloat16; Ops.fp8; Ops.single; Ops.double ] ~f:(fun p ->
        if Ops.prec_in_bytes (acc_prec p) < Ops.prec_in_bytes (comp_prec p) then
          invalid_arg
            (Printf.sprintf
               "C_syntax: accum_prec (%s) resolves narrower than compute_prec (%s) for storage %s \
                — accum_prec must be overridden together with compute_prec"
               (Ops.prec_string (acc_prec p))
               (Ops.prec_string (comp_prec p))
               (Ops.prec_string p)))

  (* The RNG lane conversions pick both their result type and which of the 128 random bits they
     consume from the precision they are rendered at ([uint4x32_to_fp8_uniform_lane] is a different
     generator from [uint4x32_to_single_uniform_lane], not a rounding of it). Their rendering
     precision is therefore pinned to the target's storage precision, and they are outside the
     storage/compute split -- [test_uniform_virtual_lane]'s virtual-vs-materialized parity is what
     this protects. *)
  let is_rng_conversion (llsc : Low_level.scalar_t) =
    match llsc with
    | Low_level.Binop (Ops.Uint4x32_to_prec_uniform_lane, _, _)
    | Unop (Ops.Uint4x32_to_prec_uniform1, _) ->
        true
    | _ -> false

  (* Anywhere in the expression, not just at its root: a virtualized narrow uniform is routinely
     consumed by further arithmetic (the default centered-scaled parameter initializer is exactly
     that shape), and the generator is selected by the precision the {e conversion} renders at,
     which [pp_scalar] inherits from its enclosing operator. An assignment that mentions one
     therefore renders wholly at the storage precision -- forgoing the wide-compute benefit for that
     statement, which is the cheap side of the trade. Not descended into [Local_scope]: its body
     renders at its own scope precision, decided by [scope_prec_of]. *)
  let rec mentions_rng_conversion (llsc : Low_level.scalar_t) =
    is_rng_conversion llsc
    ||
    match llsc with
    | Low_level.Ternop (_, (a, _), (b, _), (c, _)) ->
        mentions_rng_conversion a || mentions_rng_conversion b || mentions_rng_conversion c
    | Binop (_, (a, _), (b, _)) -> mentions_rng_conversion a || mentions_rng_conversion b
    | Unop (_, (a, _)) -> mentions_rng_conversion a
    | Local_scope _ | Get _ | Get_local _ | Get_dynamic _ | Get_merge_buffer _ | Constant _
    | Constant_bits _ | Embed_index _ ->
        false

  (* gh-ocannl-682: whether a recognized accumulation's contribution pins its accumulator to the
     target's STORAGE precision. One of the declines of the accumulator-width decision every
     rendering of a reduction level consumes ([decide_accum_width], gh-ocannl-754), so no rendering
     can drift on which bodies get the wide accumulator.

     An RNG conversion picks both its result type and which random bits it consumes from the
     precision it renders at (gh-ocannl-517), so [try_localize_serial_reduce] declines to localize
     an RNG-bearing update: its serial rendering accumulates directly in the narrow cell, narrowing
     on EVERY iteration. A rendering that instead accumulated the whole reduction at [accum_prec]
     and narrowed once would therefore change the accumulator's WIDTH, not merely the association --
     the exact property gh-ocannl-682 exists to preserve -- so the warp-shuffle rendering refuses
     such a body wherever the residency is wider than storage. Where the two coincide (f32/f64
     storage, and every backend that does not widen) there is no divergence to guard against and
     RNG-bearing reductions render exactly as they did before gh-ocannl-682. *)
  let accum_pinned_to_storage_prec (llsc : Low_level.scalar_t) = mentions_rng_conversion llsc

  (* Whether the value of a [Set] renders directly at the target's storage precision, bypassing
     [comp_prec]. True when the rendering contains no operator, so there is no intermediate to keep
     wide -- a copy, a constant, a scope-local read -- and whenever an RNG conversion appears
     anywhere in it. Rendering [x = <narrow y read at f32>] through the store's narrowing conversion
     would be bitwise the same (widening is exact and narrowing an exactly-representable value is
     the identity), but it spells a copy loop -- the very shape narrow storage exists to speed up --
     as a round-trip through f32 that no C compiler folds away. *)
  let rec renders_at_store_prec (llsc : Low_level.scalar_t) =
    match llsc with
    | Low_level.Get _ | Get_dynamic _ | Get_merge_buffer _ | Get_local _ | Local_scope _
    | Constant _ | Constant_bits _ | Embed_index _ ->
        true
    | _ when mentions_rng_conversion llsc -> true
    | Unop (Ops.Identity, (v, _)) -> renders_at_store_prec v
    | Binop (Ops.Arg1, (v, _), _) | Binop (Ops.Arg2, _, (v, _)) -> renders_at_store_prec v
    | Unop _ | Binop _ | Ternop _ -> false

  (* The precision a [Set]/[Set_dynamic] value expression renders at, and the conversion wrapping it
     back to the target's storage precision (empty when they coincide). *)
  let store_precs ~store_prec llsc =
    let prec = if renders_at_store_prec llsc then store_prec else comp_prec store_prec in
    (prec, B.convert_precision ~from:prec ~to_:store_prec)

  (* gh-ocannl-696: the two assignments a [Scan_loop] renders for one carried scalar besides the
     body's own -- the init [prev = init] ahead of the loop and the rotation [prev = next] at the
     end of every iteration. Every per-local census below walks them as the [Set_local]s they render
     as (through [pp_ll]'s [Set_local] arm), so a classification a census would give an explicit
     assignment is given the implicit one too. *)
  let scan_implicit_set_locals (c : Low_level.carried) : Low_level.t list =
    [ Set_local (c.prev, c.init); Set_local (c.prev, Get_local c.next) ]

  (* gh-ocannl-696: the carried ids of every [Scan_loop] in the procs. Carried state resides at its
     node's storage precision -- the contract says the node TYPES the state, and a recurrence that
     rounds per step at fp16 is a different function from one kept wide -- so [scope_prec_of] pins
     these like the rng carve-out, ahead of the accumulator widening. The arithmetic feeding an
     update still runs at compute precision; only the carried value is narrowed per iteration. Keyed
     by the full scope id (node and integer), the identity the rendered C name carries, so a
     hand-minted local over another node sharing the integer is not pinned by association. Populated
     by the rng census below, which already visits every scan. *)
  let carried_state_scope_ids : Low_level.Scope_id.t Hash_set.t =
    Hash_set.create (module Low_level.Scope_id)

  (* Scope-local scalars an RNG conversion writes. Their declaration, their assignments and their
     reads all have to agree on a precision, and only a whole-proc scan sees all three (a
     [Declare_local] carries no value), so the exclusion is resolved once here rather than per
     statement. A superset would only forgo an optimization, never mis-render. Keyed per scope id --
     the full [Scope_id], node and integer, the identity the rendered C name carries -- not per
     tnode (Codex P2 round 2 on PR #396; full ids since gh-ocannl-696): virtualization copies of an
     rng-consuming node each carry their own rng-mentioning [Set_local], so every copy is marked on
     its own, while an UNRELATED scope over the same tnode — a schedule-minted accumulation into a
     node that elsewhere consumes rng — keeps its accumulator residency instead of being pinned to
     storage by the shared uid. A [Tile_mma]'s scalar fallback renders through the same
     [scope_prec_of], so the scan descends into it. *)
  let rng_scope_ids =
    let acc = Hash_set.create (module Low_level.Scope_id) in
    let rec scan_sc (llsc : Low_level.scalar_t) =
      match llsc with
      | Low_level.Local_scope { body; _ } -> scan body
      | Get _ | Get_local _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ ->
          ()
      | Get_dynamic { dyn_value = v, _; _ } -> scan_sc v
      | Ternop (_, (a, _), (b, _), (c, _)) ->
          scan_sc a;
          scan_sc b;
          scan_sc c
      | Binop (_, (a, _), (b, _)) ->
          scan_sc a;
          scan_sc b
      | Unop (_, (a, _)) -> scan_sc a
    and scan (llc : Low_level.t) =
      match llc with
      | Low_level.Seq (a, b) ->
          scan a;
          scan b
      | For_loop { body; _ } -> scan body
      | Scan_loop { carried; body; _ } ->
          (* gh-ocannl-696: the census sees exactly the statements the renderer emits for a scan --
             the implicit init [prev = init] and rotation [prev = next] are [Set_local]s to it --
             and records the carried ids for the storage-precision pin. *)
          List.iter carried ~f:(fun c ->
              Hash_set.add carried_state_scope_ids c.Low_level.prev;
              Hash_set.add carried_state_scope_ids c.Low_level.next;
              List.iter (scan_implicit_set_locals c) ~f:scan);
          scan body
      | If { cond = c, _; body } ->
          scan_sc c;
          scan body
      | Set_local (id, v) ->
          if mentions_rng_conversion v then Hash_set.add acc id;
          scan_sc v
      | Set { llsc; _ } -> scan_sc llsc
      | Set_dynamic { dyn_value = v, _; llsc; _ } ->
          scan_sc v;
          scan_sc llsc
      | Set_from_vec { arg = a, _; _ } -> scan_sc a
      | Tile_mma { fallback; _ } -> scan fallback
      | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Declare_local _ | Workgroup_barrier
        ->
          ()
    in
    Array.iter B.procs ~f:scan;
    acc

  (* Scope locals that are reduction ACCUMULATORS (gh-ocannl-663): every [Set_local] to the scope is
     either an opening init (its value free of the local) or a reduce-shaped update
     ([Low_level.accum_local_update_parts]'s grammar, the FMA form counting as Add), all updates
     under ONE reduction operator — a general recurrence or a mixed-operator sequence keeps its
     per-iteration narrowing by the source's own semantics and disqualifies the scope. Such locals
     take [acc_prec] instead of [comp_prec] in [scope_prec_of], so a virtualized or schedule-minted
     accumulator (Unroll ~materialize / Partition) resides at the same width as the widened serial
     fallback below ([try_localize_serial_reduce], which registers its codegen-minted scopes here) —
     on the backends where the two precisions coincide this census is inert. Classified per
     [scope_id] over the scope's own updates wherever they sit, which covers [Local_scope] values,
     the [Declare_local]-lifted form of [hoist_cross_statement_cse], and scopes inside a
     [Tile_mma]'s scalar fallback (rendered on decline through the same [scope_prec_of]).

     The residency must be observation-neutral, and it is only while no control flow reads the local
     mid-reduction: a guard observing the accumulator (e.g. [If (local < c)] around [local += x])
     would execute a DIFFERENT set of iterations under a widened local, since the wide value reaches
     sums the per-step-narrowed one rounds away. So any local read by an [If] condition is
     disqualified (Codex P1 round 2 on PR #396) — a stronger condition than the hoisting license
     ([scope_updates_reduce_op] demands guard PURITY, because hoisting moves the init and store
     across the guard; residency moves nothing, so a data-dependent guard that does not read the
     local stays eligible). A nested [Local_scope] inside a condition is not a read of ITS local at
     this level: the condition observes only that scope's once-narrowed result, which is
     residency-neutral, and the main walk classifies its interior. *)
  let accum_scope_ids =
    let verdicts = Hashtbl.create (module Int) in
    let classify (id : Low_level.scope_id) v =
      (* [accum_local_update_op] rather than [accum_local_update_parts]: virtualization's
         guarded-read updates — [Where (index-only cond, update, Get_local id)] — are reductions for
         residency purposes (Codex P1 round 3 on PR #396); treating their self-read as a recurrence
         left a virtualized reduction narrow while its materialized serial twin widened, a
         placement-dependent width. *)
      match Low_level.accum_local_update_op ~id v with
      | Some op ->
          Hashtbl.update verdicts id.scope_id ~f:(function
            | None -> `Accum op
            | Some (`Accum op0) when Ops.equal_binop op0 op -> `Accum op
            | Some _ -> `Bad)
      | None ->
          if Low_level.scalar_reads_scope ~id v then
            Hashtbl.set verdicts ~key:id.scope_id ~data:`Bad
    in
    let rec guarding_reads_sc (llsc : Low_level.scalar_t) =
      match llsc with
      | Low_level.Get_local id -> Hashtbl.set verdicts ~key:id.Low_level.scope_id ~data:`Bad
      | Local_scope _ | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ ->
          ()
      | Get_dynamic { dyn_value = v, _; _ } -> guarding_reads_sc v
      | Ternop (_, (a, _), (b, _), (c, _)) ->
          guarding_reads_sc a;
          guarding_reads_sc b;
          guarding_reads_sc c
      | Binop (_, (a, _), (b, _)) ->
          guarding_reads_sc a;
          guarding_reads_sc b
      | Unop (_, (a, _)) -> guarding_reads_sc a
    in
    let rec scan_sc (llsc : Low_level.scalar_t) =
      match llsc with
      | Low_level.Local_scope { body; _ } -> scan body
      | Get _ | Get_local _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ ->
          ()
      | Get_dynamic { dyn_value = v, _; _ } -> scan_sc v
      | Ternop (_, (a, _), (b, _), (c, _)) ->
          scan_sc a;
          scan_sc b;
          scan_sc c
      | Binop (_, (a, _), (b, _)) ->
          scan_sc a;
          scan_sc b
      | Unop (_, (a, _)) -> scan_sc a
    and scan (llc : Low_level.t) =
      match llc with
      | Low_level.Seq (a, b) ->
          scan a;
          scan b
      | For_loop { body; _ } -> scan body
      | Scan_loop { carried; body; _ } ->
          (* gh-ocannl-696: the implicit init and rotation are classified like the [Set_local]s they
             render as. A carried update reads [prev] and writes [next], so the classifier sees no
             self-update; the state's residency is decided by [carried_state_scope_ids] (its node's
             storage precision), not here. *)
          List.iter carried ~f:(fun c -> List.iter (scan_implicit_set_locals c) ~f:scan);
          scan body
      | If { cond = c, _; body } ->
          guarding_reads_sc c;
          scan_sc c;
          scan body
      | Set_local (id, v) ->
          classify id v;
          scan_sc v
      | Set { llsc; _ } -> scan_sc llsc
      | Set_dynamic { dyn_value = v, _; llsc; _ } ->
          scan_sc v;
          scan_sc llsc
      | Set_from_vec { arg = a, _; _ } -> scan_sc a
      | Tile_mma { fallback; _ } -> scan fallback
      | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Declare_local _ | Workgroup_barrier
        ->
          ()
    in
    Array.iter B.procs ~f:scan;
    let acc = Hash_set.create (module Int) in
    Hashtbl.iteri verdicts ~f:(fun ~key ~data ->
        match data with `Accum _ -> Hash_set.add acc key | `Bad -> ());
    acc

  (* The precision a scope-local scalar is declared, written and read at. The rng carve-out takes
     precedence: a scope whose value consumes an RNG conversion is pinned to the storage precision
     wholesale (gh-ocannl-517), reduction-shaped or not -- and so is a scan's carried state
     (gh-ocannl-696), whose node's precision is the recurrence's semantics. *)
  let scope_prec_of (id : Low_level.scope_id) =
    let p = Lazy.force id.tn.Tn.storage_prec in
    if Hash_set.mem rng_scope_ids id || Hash_set.mem carried_state_scope_ids id then p
    else if Hash_set.mem accum_scope_ids id.scope_id then acc_prec p
    else comp_prec p

  (* The emitted name of a scope local. One definition, so that the volatility census below names
     the same identifier the declaration carries rather than a second spelling of the convention. *)
  let scope_local_ident Low_level.{ scope_id; tn } =
    "v" ^ Int.to_string scope_id ^ "_" ^ get_ident tn

  (* The Metal compiler bug also reaches the localized spelling of a scalar accumulation
     (gh-ocannl-731): instead of a device-memory RMW, the loop updates a scope local and stores it
     once. Shader validation hides both manifestations. The accumulator stays register-resident;
     [Set_local] below confines the workaround to expression-level casts on the accumulating loop's
     device reads (gh-ocannl-820). The decision is censused on EVERY backend (gh-ocannl-782) — an
     accumulation left plain because the backend asks for nothing is exactly what a residency
     investigation wants to see, and the classification it needs was computed for the precision
     decision anyway. *)
  let scope_decl_type (id : Low_level.scope_id) = B.typ_of_prec (scope_prec_of id)

  let wrap_conversion (pre, post) doc =
    let open PPrint in
    if String.is_empty pre && String.is_empty post then doc
    else group (string pre ^^ doc ^^ string post)

  (* Set by [compile_proc]: the per-compilation-lineage placement resolution
     (docs/proposals/context-scoped-memory-modes.md). Codegen both consults and settles placements
     here -- never on the tnode. *)
  let current_placements : Tn.Placements.t option ref = ref None

  let placements () =
    Option.value_exn ~message:"C_syntax: placements consulted outside compile_proc"
      !current_placements

  (* Whether scalar device loads at the current expression-rendering point must use the Metal
     accumulation workaround. Bracketed narrowly by the two recognized accumulating-statement arms
     below; in particular the opening read of a localized accumulator is outside it. *)
  let volatile_accumulation_reads = ref false
  let volatile_accumulation_scope_stack : int list ref = ref []
  let volatile_read_observers : bool ref list ref = ref []
  let volatile_accumulation_scopes = Hash_set.create (module Int)
  let rendered_accumulation_scopes = Hash_set.create (module Int)

  type volatility_event = Accumulation of Low_level.scope_id | Site of volatility_site

  (* Reverse emission order, like [volatility_census] itself. Accumulations stay symbolic until the
     whole kernel body has rendered: a [Declare_local] precedes the lifted update that decides
     whether it emitted a volatile read, while an inline [Local_scope] encloses that update. Keeping
     one event stream preserves their order relative to device-memory RMW sites in both forms. *)
  let volatility_events : volatility_event list ref = ref []

  let record_accumulation_scope (id : Low_level.scope_id) =
    if
      !volatility_census_enabled
      && Hash_set.mem accum_scope_ids id.scope_id
      && not (Hash_set.mem rendered_accumulation_scopes id.scope_id)
    then begin
      Hash_set.add rendered_accumulation_scopes id.scope_id;
      volatility_events := Accumulation id :: !volatility_events
    end

  let with_volatile_accumulation_reads ?(scope_ids = []) enabled f =
    let saved = !volatile_accumulation_reads in
    let saved_scopes = !volatile_accumulation_scope_stack in
    let saved_observers = !volatile_read_observers in
    let emitted = ref false in
    volatile_accumulation_reads := saved || enabled;
    if enabled then begin
      volatile_accumulation_scope_stack := scope_ids @ saved_scopes;
      volatile_read_observers := emitted :: saved_observers
    end;
    let restore () =
      volatile_accumulation_reads := saved;
      volatile_accumulation_scope_stack := saved_scopes;
      volatile_read_observers := saved_observers
    in
    let result = Exn.protect ~f ~finally:restore in
    (result, !emitted)

  let record_volatile_read () =
    List.iter !volatile_read_observers ~f:(fun seen -> seen := true);
    List.iter !volatile_accumulation_scope_stack ~f:(Hash_set.add volatile_accumulation_scopes)

  (* Accumulator locals whose updates are controlled by a statement subtree. A materialized read in
     an [If] condition around [Set_local (id, id + constant)] is the accumulating loop's ONLY device
     read, so it needs the same expression-level cast as a read in the update itself. Walk nested
     scalar scopes too: an [If] can control a statement whose value owns the accumulator. *)
  let controlled_accumulation_scope_ids body =
    let seen = Hash_set.create (module Int) in
    let ids = ref [] in
    let add (id : Low_level.scope_id) value =
      if
        Hash_set.mem accum_scope_ids id.scope_id
        && Option.is_some (Low_level.accum_local_update_op ~id value)
        && not (Hash_set.mem seen id.scope_id)
      then begin
        Hash_set.add seen id.scope_id;
        ids := id.scope_id :: !ids
      end
    in
    let rec scalar = function
      | Low_level.Local_scope { body; _ } -> stmt body
      | Get_dynamic { dyn_value = value, _; _ } -> scalar value
      | Ternop (_, (a, _), (b, _), (c, _)) ->
          scalar a;
          scalar b;
          scalar c
      | Binop (_, (a, _), (b, _)) ->
          scalar a;
          scalar b
      | Unop (_, (a, _)) -> scalar a
      | Get _ | Get_local _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ ->
          ()
    and stmt = function
      | Low_level.Seq (a, b) ->
          stmt a;
          stmt b
      | For_loop { body; _ } | If { body; _ } -> stmt body
      | Scan_loop { carried; body; _ } ->
          List.iter carried ~f:(fun c -> List.iter (scan_implicit_set_locals c) ~f:stmt);
          stmt body
      | Set_local (id, value) ->
          add id value;
          scalar value
      | Set { llsc; _ } -> scalar llsc
      | Set_dynamic { dyn_value = value, _; llsc; _ } ->
          scalar value;
          scalar llsc
      | Set_from_vec { arg = value, _; _ } -> scalar value
      | Tile_mma { fallback; _ } -> stmt fallback
      | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Declare_local _ | Workgroup_barrier
        ->
          ()
    in
    stmt body;
    List.rev !ids

  let pp_device_read_ptr tn ident_doc =
    let open PPrint in
    if !volatile_accumulation_reads && Tn.Placements.is_materialized_force (placements ()) tn 820
    then begin
      record_volatile_read ();
      let typ =
        "(" ^ B.buffer_prefix ^ "volatile " ^ B.typ_of_prec (Lazy.force tn.Tn.storage_prec) ^ "*)"
      in
      parens (string typ ^^ ident_doc)
    end
    else ident_doc

  let in_ctx tn = Tn.Placements.is_in_context_force (placements ()) tn 46

  (* [routine_names]: the kernel function names in [proc_doc] — their declarations (and the
     name-echoing comments) are token occurrences the usage scan below must not count as builtin
     uses. Since gh-ocannl-686 the names arriving here are {!kernel_ident}-mangled, so a routine
     name can no longer BE a builtin key: a colliding one was renamed before it reached codegen, and
     the kernel's own token no longer matches the key. The exclusion is kept as the backstop for
     that guarantee — a routine named exactly like a builtin cannot genuinely use it (the definition
     would be a duplicate C symbol), so dropping the name from the scan degrades a pathological
     collision from silent helper injection — which on CUDA could raise the architecture floor past
     the device (Codex P2 on PR #317, round 4) — into an ordinary compile error naming the
     conflict. *)
  let filter_and_prepend_builtins ~routine_names ~includes ~builtins ~proc_doc =
    let doc_buffer = Buffer.create 4096 in
    PPrint.ToBuffer.pretty 1.0 110 doc_buffer proc_doc;
    let doc_string = Buffer.contents doc_buffer in
    let result_buffer = Buffer.create 4096 in
    Buffer.add_string result_buffer includes;
    Buffer.add_string result_buffer "\n";

    (* Collect all needed keys, including dependencies. A key is "used" when it occurs as a whole
       C identifier token in the KERNEL source, not as an arbitrary substring: only direct uses
       appear there (intra-builtin needs are the explicit dependency lists), and every direct use
       — a call, a type name — is a token. Substring matching over-included on longer identifiers
       containing a key, which was harmless noise until a key gained an architecture floor: a
       ROUTINE named [ocannl_cp_async4_probe] would inject the sm_80 [cp.async] helper into a
       pre-Ampere kernel and [gpu_arch_options] would then emit PTX that device cannot load
       (Codex P2 on PR #317, round 3; routine names are not covered by the tensor-identifier
       blacklist). *)
    (* Comments and string literals are not uses either: [Comment] statements render arbitrary
       text as [/* ... */] and the debug-log mode renders node names into printf format literals,
       so a stray helper token there would activate a builtin — and, on CUDA, its architecture
       floor — without any call (Codex P2 on PR #317, round 5). Strip both before scanning;
       genuine uses are code tokens and survive. Removed regions become a single space so tokens
       cannot concatenate across them. *)
    let scannable =
      let s = doc_string in
      let n = String.length s in
      let b = Buffer.create n in
      let rec go i state =
        if i < n then
          match state with
          | `Code ->
              if i + 1 < n && Char.equal s.[i] '/' && Char.equal s.[i + 1] '*' then (
                Buffer.add_char b ' ';
                go (i + 2) `Block)
              else if i + 1 < n && Char.equal s.[i] '/' && Char.equal s.[i + 1] '/' then
                go (i + 2) `Line
              else if Char.equal s.[i] '"' then (
                Buffer.add_char b ' ';
                go (i + 1) `Str)
              else (
                Buffer.add_char b s.[i];
                go (i + 1) `Code)
          | `Block ->
              if i + 1 < n && Char.equal s.[i] '*' && Char.equal s.[i + 1] '/' then go (i + 2) `Code
              else go (i + 1) `Block
          | `Line ->
              if Char.equal s.[i] '\n' then (
                Buffer.add_char b '\n';
                go (i + 1) `Code)
              else go (i + 1) `Line
          | `Str ->
              if Char.equal s.[i] '\\' then go (i + 2) `Str
              else if Char.equal s.[i] '"' then go (i + 1) `Code
              else go (i + 1) `Str
      in
      go 0 `Code;
      Buffer.contents b
    in
    let is_ident_char c = Char.is_alphanum c || Char.equal c '_' in
    let mentions_token key =
      let klen = String.length key in
      let dlen = String.length scannable in
      let rec scan pos =
        match String.substr_index ~pos scannable ~pattern:key with
        | None -> false
        | Some i ->
            let pre_ok = i = 0 || not (is_ident_char scannable.[i - 1]) in
            let post_ok = i + klen >= dlen || not (is_ident_char scannable.[i + klen]) in
            if pre_ok && post_ok then true else scan (i + 1)
      in
      scan 0
    in
    let needed_keys = ref (Set.empty (module String)) in
    List.iter builtins ~f:(fun (key, _, _) ->
        if (not (List.mem routine_names key ~equal:String.equal)) && mentions_token key then
          needed_keys := Set.add !needed_keys key);

    (* Add dependencies recursively *)
    let processed_keys = ref (Set.empty (module String)) in
    let rec add_dependencies key =
      if not (Set.mem !processed_keys key) then (
        processed_keys := Set.add !processed_keys key;
        needed_keys := Set.add !needed_keys key;
        match List.find builtins ~f:(fun (k, _, _) -> String.equal k key) with
        | Some (_, _, deps) -> List.iter deps ~f:add_dependencies
        | None -> ())
    in
    Set.iter !needed_keys ~f:add_dependencies;

    (* Add the builtins in order *)
    List.iter builtins ~f:(fun (key, definition, _) ->
        if Set.mem !needed_keys key then (
          Buffer.add_string result_buffer definition;
          Buffer.add_string result_buffer "\n"));
    Buffer.add_string result_buffer doc_string;
    Buffer.contents result_buffer

  open Indexing
  open Doc_helpers

  (* An embedded index that renders as a sum (more than one term) must be parenthesized when spliced
     into an operator context; single-term indices bind at least as tightly as [/], [%] and
     friends. *)
  let affine_needs_parens = function
    | Indexing.Affine { symbols; offset } -> (List.length symbols + if offset = 0 then 0 else 1) > 1
    | _ -> false

  (* The row-major flat offset of an access: Horner over the axes, where [axis_doc i] renders the
     [i]-th index. Axes whose index renders empty (a [Sub_axis] folded into its neighbour) drop out
     of the sum while still contributing their stride. *)
  let pp_offset_by ~axis_doc dims n =
    let open PPrint in
    if n = 0 then string "0"
    else begin
      let doc = ref (axis_doc 0) in
      for i = 1 to n - 1 do
        let idx_doc = axis_doc i in
        if PPrint.is_empty !doc then doc := idx_doc
        else if PPrint.is_empty idx_doc then
          doc := parens !doc ^^ string (" * " ^ Int.to_string dims.(i))
        else doc := parens !doc ^^ string (" * " ^ Int.to_string dims.(i) ^ " + ") ^^ idx_doc
      done;
      !doc
    end

  let pp_array_offset (idcs, dims) =
    pp_offset_by ~axis_doc:(fun i -> pp_axis_index idcs.(i)) dims (Array.length idcs)

  (* gh-343: like [pp_array_offset] but at [dyn_axis] splices [dyn_idx_doc] (an integer expression
     derived from a runtime value) instead of the static [axis_index]. Used for [Get_dynamic]. *)
  let pp_array_offset_dyn (idcs, dims) ~dyn_axis ~dyn_idx_doc =
    pp_offset_by
      ~axis_doc:(fun i -> if i = dyn_axis then dyn_idx_doc else pp_axis_index idcs.(i))
      dims (Array.length idcs)

  let doc_to_string doc =
    let buf = Buffer.create 128 in
    PPrint.ToBuffer.compact buf doc;
    Buffer.contents buf

  let array_offset_to_string (idcs, dims) = doc_to_string @@ pp_array_offset (idcs, dims)

  let pp_local_defs (local_defs : (int * PPrint.document) list) =
    let open PPrint in
    List.dedup_and_sort local_defs ~compare:(fun (a, _) (b, _) -> Int.compare a b)
    |> List.map ~f:snd |> separate hardline

  let pp_scope_id Low_level.{ scope_id; tn } =
    let open PPrint in
    string ("v" ^ Int.to_string scope_id ^ "_" ^ get_ident tn)

  (* {3 Vector-extension accumulator grids (gh-ocannl-468 / gh-ocannl-469)}

     Emission helpers for grids of vector-register accumulators in the [`Vec_extensions] style. The
     SIMD reduction rendering of [Vectorized] accumulation loops (gh-ocannl-468; ggml's
     [ggml_vec_dot_f32] pattern) uses a 1×N grid — N independent accumulator chains folded at loop
     exit — and the register-tiled [Tile_mma] micro-kernel (gh-ocannl-469) will hold its C-tile as
     an RM×RN grid of vector registers across the fused k-loop. *)

  (* Broadcast the scalar VARIABLE [src] across every lane of [vtyp], as an initializer holding
     [lanes] copies of it. The vector extensions splat a scalar operand of a binary operator but
     have no cast or intrinsic form that does it, and the two arithmetic spellings that look like
     they would both change bits the scalar twin keeps — while a remainder loop, a peeled edge and
     the serial fallback all consume the element itself, so the rendering's BITWISE-equality promise
     covers every input, not merely the well-behaved ones:

     - [((vtyp){0} + x)] normalizes a negative-zero [x] to [+0.0], since IEEE-754 has [(+0.0) +
     (-0.0) = +0.0] (gh-ocannl-615); - [(x - (vtyp){0})], its replacement, is the identity on both
     zeros — but it is still ARITHMETIC, so it quiets a signaling NaN: verified as [0x7f800001 ->
     0x7fc00001] under gcc [-fsignaling-nans], and by inspection at clang [-O0]. That it usually
     survives is only the optimizer folding the subtraction into a broadcast because nothing
     requires it to respect sNaN — the same "the compiler will probably do the right thing"
     dependency that [vec_fma_builtin] refuses for [a * b + c] (Codex P2 on PR #360).

     An initializer performs no arithmetic — each lane is a copy — so it preserves every bit pattern
     unconditionally. Taking a variable rather than an expression is what keeps that cheap: the
     callers bind the scalar once (the operand can be a memory load under a [convert_precision]
     wrapper), so the [lanes] repetitions below are repetitions of an identifier. *)
  let vec_splat ~vtyp ~lanes src =
    let open PPrint in
    string
      (Printf.sprintf "(%s){%s}" vtyp (String.concat ~sep:", " (List.init lanes ~f:(fun _ -> src))))

  (* The vector-extension typedef shared by the explicit-SIMD renderings: [lanes] elements of the
     compute precision (f32, f64, or -- where the target has native 16-bit arithmetic, gh-ocannl-516
     -- fp16, whose element type [HALF_T] is [_Float16] exactly when that probe passed). *)
  let vec_ext_typ ~prec ~lanes =
    let vtyp =
      Printf.sprintf "ocannl_vec%d%s" lanes
        (match prec with Ops.Double_prec _ -> "d" | Ops.Half_prec _ -> "h" | _ -> "f")
    in
    ( vtyp,
      PPrint.string
        (Printf.sprintf "typedef %s %s __attribute__((vector_size(%d)));" (B.typ_of_prec prec) vtyp
           (lanes * Ops.prec_in_bytes prec)) )

  (* The widest vector this loop can fill, [None] where even the narrowest cannot: see
     {!Backend_intf.simd_lanes_for} for why the width degrades per loop instead of being one number
     per machine, and for what the floor protects. *)
  let vec_lanes_for ~prec ~extent =
    simd_lanes_for ~vector_bytes:B.vector_bytes ~elt_bytes:(Ops.prec_in_bytes prec) ~extent

  (* {4 Convert-on-load / convert-on-store for narrow storage (gh-ocannl-517)}

     A node stored at a narrow float precision but computed in f32 ({!C_syntax_config.compute_prec})
     reaches a vector register through a conversion. The lane geometry is keyed off the {e compute}
     vector — [lanes] f32 elements, so the narrow side is a half-width vector of [lanes] elements —
     which is the whole point: converting per lane inside the loop body would keep the loop scalar
     and give back the traffic win.

     [vec_bridge] returns the [load]/[store] statement builders for one (storage, compute) pair. It
     is the identity memcpy when they coincide (the pre-gh-517 f32/f64 path, unchanged byte for
     byte), and otherwise:

     - bf16 is the top 16 bits of an f32, so widening is a zero-extend and a shift, and narrowing is
     [single_to_bfloat16]'s round-to-nearest-even done with vector arithmetic — bitwise what the
     scalar path computes, by construction. - fp16 converts in one instruction on [_Float16]
     targets. Whether the type exists is a C-preprocessor fact the renderer cannot see
     (gh-ocannl-516's design problem), so both arms are emitted under [#if HAS_NATIVE_FLOAT16]. Only
     the conversion is duplicated, never the arithmetic, so this stays a few lines rather than a
     second copy of the kernel body. - everything else (fp8), and every fallback arm, converts per
     lane through the backend's own [convert_precision] — the scalar path's conversion, so parity is
     not something to verify — while the arithmetic around it stays vectorized. *)

  let vec_typedef_doc ~ctyp ~name ~bytes =
    PPrint.string (Printf.sprintf "typedef %s %s __attribute__((vector_size(%d)));" ctyp name bytes)

  (* The auxiliary typedefs a bridge registered, in name order: a hash table's traversal order would
     make the emitted source depend on hashing, and generated sources are snapshot-compared. *)
  let registered_typedefs tbl =
    Hashtbl.to_alist tbl
    |> List.sort ~compare:(fun (a, _) (b, _) -> String.compare a b)
    |> List.map ~f:snd

  (* [mem] is an element access document ([x[offset]]); [&x[offset]] is the base of the narrow run
     it starts. The conversions themselves are backend builtins ([OCANNL_VEC_WIDEN_BFLOAT16] and
     friends in {!Builtins_cc}) rather than inline preprocessor arms, so a kernel body stays one
     line per load. *)
  let vec_bridge ~store_prec ~prec ~lanes ~vtyp ~need_typedef ~fresh:_ =
    let open PPrint in
    let base mem = string "&" ^^ mem in
    let call fn args = string (fn ^ "(") ^^ separate (string ", ") args ^^ string ");" in
    let per_lane_load ~dst ~mem =
      let pre, post = B.convert_precision ~from:store_prec ~to_:prec in
      string
        (Printf.sprintf "for (int ocannl_l__ = 0; ocannl_l__ < %d; ++ocannl_l__) %s[ocannl_l__] = "
           lanes dst)
      ^^ string pre
      ^^ parens (base mem)
      ^^ string "[ocannl_l__]" ^^ string post ^^ semi
    in
    let per_lane_store ~src ~mem =
      let pre, post = B.convert_precision ~from:prec ~to_:store_prec in
      string (Printf.sprintf "for (int ocannl_l__ = 0; ocannl_l__ < %d; ++ocannl_l__) " lanes)
      ^^ parens (base mem)
      ^^ string "[ocannl_l__] = " ^^ string pre
      ^^ string (Printf.sprintf "%s[ocannl_l__]" src)
      ^^ string post ^^ semi
    in
    if Ops.equal_prec store_prec prec then
      ( (fun ~dst ~mem ->
          string (vtyp ^ " " ^ dst ^ ";")
          ^^ hardline
          ^^ string ("__builtin_memcpy(&" ^ dst ^ ", &")
          ^^ mem
          ^^ string (", sizeof(" ^ dst ^ "));")),
        fun ~src ~mem ->
          string "__builtin_memcpy(&" ^^ mem
          ^^ string (Printf.sprintf ", &%s, sizeof(%s));" src src) )
    else
      match store_prec with
      | Ops.Bfloat16_prec _ ->
          let u16 = Printf.sprintf "ocannl_vec%du16" lanes in
          let u32 = Printf.sprintf "ocannl_vec%du32" lanes in
          need_typedef u16 (vec_typedef_doc ~ctyp:"unsigned short" ~name:u16 ~bytes:(lanes * 2));
          need_typedef u32 (vec_typedef_doc ~ctyp:"uint32_t" ~name:u32 ~bytes:(lanes * 4));
          let args = [ string u16; string u32; OCaml.int lanes ] in
          ( (fun ~dst ~mem ->
              string (vtyp ^ " " ^ dst ^ ";")
              ^^ hardline
              ^^ call "OCANNL_VEC_WIDEN_BFLOAT16" (args @ [ string dst; base mem ])),
            fun ~src ~mem -> call "OCANNL_VEC_NARROW_BFLOAT16" (args @ [ base mem; string src ]) )
      | Ops.Half_prec _ ->
          let h = Printf.sprintf "ocannl_vec%dh" lanes in
          (* [_Float16] exists only where the C preprocessor says so, and this typedef is the one
             place that needs the type name itself rather than a macro argument. *)
          need_typedef h
            (string "#if HAS_NATIVE_FLOAT16" ^^ hardline
            ^^ vec_typedef_doc ~ctyp:"_Float16" ~name:h ~bytes:(lanes * 2)
            ^^ hardline ^^ string "#endif");
          ( (fun ~dst ~mem ->
              string (vtyp ^ " " ^ dst ^ ";")
              ^^ hardline
              ^^ call "OCANNL_VEC_WIDEN_HALF"
                   [ string vtyp; string h; OCaml.int lanes; string dst; base mem ]),
            fun ~src ~mem ->
              call "OCANNL_VEC_NARROW_HALF" [ string h; OCaml.int lanes; base mem; string src ] )
      | _ ->
          (* fp8 and any other narrow format: the arithmetic still vectorizes, only the conversion
             is per lane -- through the scalar path's own conversion, so parity is by
             construction. *)
          ( (fun ~dst ~mem -> string (vtyp ^ " " ^ dst ^ ";") ^^ hardline ^^ per_lane_load ~dst ~mem),
            fun ~src ~mem -> per_lane_store ~src ~mem )

  (* The names of a [rows]×[cols] grid of vector accumulator registers. *)
  let vec_acc_grid ~prefix ~rows ~cols : string array array =
    Array.init rows ~f:(fun r ->
        Array.init cols ~f:(fun c -> Printf.sprintf "%s_%d_%d__" prefix r c))

  (* [dst = op(dst, src)] on whole vector registers ([src] must also name a register): vector infix
     arithmetic for the ring operators, a COMPARE-AND-SELECT for [Max]/[Min].

     [Max]/[Min] have no vector infix and no builtin with the right semantics -- the ISA's
     [__builtin_ia32_maxps] family and clang's [__builtin_elementwise_max] return their second
     operand when either is NaN, where [fmaxf] returns the non-NaN one -- so the rendering used to
     be a fixed-trip per-lane loop calling the scalar [fmaxf]/[fminf], relying on SLP to reassemble
     it. gh-ocannl-649 measured that, and SLP never gets the chance: gcc will not contract
     [fmax]/[fmaxf] into [maxsd]/[maxss] without [-ffinite-math-only] (those instructions have the
     wrong NaN behaviour), so with the default numerics each lane compiled to a LIBRARY CALL --- at
     every [-march] from [x86-64] to [x86-64-v4] and at both optimization levels, as
     [test/operations/cc_march_census] measures. At -O3 the loop was not merely calling out but
     fully scalarized besides: at [x86-64-v3], 64-byte width, f32, its innermost loop censused 259
     instructions, 0 vector operations, 190 scalar ones, 64 libm calls and 126 stack references,
     against the FMA accumulator loop's 20 / 16 / 0 / 0 / 0 in the same kernel. That is worse than
     the scalarization gh-ocannl-614/gh-ocannl-621 found for the FMA update (which at least stayed
     in registers), and the register-pressure argument this comment used to make against a
     whole-vector form was beside the point: an opaque call cannot be vectorized at any grid size,
     so look for CALLS before reasoning about allocation.

     The faithful whole-vector form is the C definition of [fmax] written out: [fmax(a, b)] is [a]
     when [a >= b] or when [b] is NaN, and [b] otherwise. GNU C vector comparisons yield an
     all-ones/all-zeros integer vector of the element width, so the select is a mask blend, spelled
     through [__typeof__] so that no auxiliary typedef has to be registered. It implements what C
     specifies of [fmax]/[fmin]: the larger (smaller) operand, the non-NaN one when exactly one is
     NaN, a NaN when both are. Compared against glibc over every ordered pair of {0, -0, +-1,
     +-inf, +-NaN, +-denormal, ...} at f32 and f64, the two agree bitwise everywhere except the two
     cases C leaves UNSPECIFIED: which of [+0]/[-0] a tie returns, and which payload a both-NaN pair
     returns. Both are order-dependent under a [Vectorized] retype regardless -- splitting the fold
     into [chains * lanes] independent chains already changes which of several equal-comparing
     operands reaches the final combine -- so neither is a behaviour this could preserve if it tried.

     Two deviations that ARE real, and are accepted rather than unnoticed. A SIGNALING NaN differs
     in both value and flag: glibc's [fmax] propagates it, this selects the numeric operand. It is
     reachable, contrary to what this comment claimed until gh-ocannl-649 review round 1 -- an OCaml
     float carries whatever payload it was built with and [Ndarray.set_from_float] stores it into a
     [Float64] bigarray unnarrowed, so no conversion quiets it on the way to an f64 kernel. And
     [>=]/[<=] are SIGNALING comparisons (C's relational operators are, unlike [!=]), so a quiet NaN
     raises [FE_INVALID] here where [fmax] would not. Neither is chased because GNU C has no quiet
     whole-vector comparison to chase them with -- x86's quiet predicate is reachable only through
     per-width [__builtin_ia32_cmp*] arms carrying an [_CMP_*_OQ] immediate, and the aarch64 arm
     above sidesteps the question entirely since [FMAXNM] does not signal -- and because a kernel
     runs with FP traps disabled, where a raised flag nothing reads costs nothing. A build that
     enables invalid-operation trapping and feeds NaNs to a [Max] reduction is the case this would
     hurt; it does not exist here, and this note is what makes it findable if it comes to.

     Only float precisions reach here at all ([vector_prec_ok]: f32, f64, and fp16 where the target
     has native 16-bit arithmetic), so the blend never has to mean anything on an integer vector.
     [test/operations/cpu_simd_reduction] executes the NaN case, which no structural check can
     see. *)
  (* The target builtins spelling a WHOLE-VECTOR [fmax]/[fmin] for this (compute precision, lane
     count), in the shape and with the guard discipline of {!vec_fma_builtin} next door: an arm here
     is preferred over the portable mask blend below, and a compiler that spells it differently
     falls through to that blend rather than failing to compile.

     Only aarch64 has one. gcc's internal NEON [fmax]/[fmin] builtins render as a single [FMAXNM] /
     [FMINNM], which is IEEE 754 [maxNum]/[minNum] and therefore exactly C's [fmax]/[fmin] --
     [fmaxnm v0.4s, v0.4s, v1.4s] against the blend's compare, or and bit-select. gcc's NaN-
     propagating [FMAX] is a DIFFERENT builtin ([__builtin_aarch64_fmax_nan*]), so the naming does
     not silently hand back the wrong one. x86 has no counterpart worth an arm: the
     [__builtin_ia32_maxps] family and clang's [__builtin_elementwise_max] both return their second
     operand when either is NaN, which is not what [fmax] means, and the blend already compiles to
     the two packed instructions those would be.

     The fp16 arm exists at every [-march]: [__has_builtin(__builtin_aarch64_fmaxv8hf)] is 1 even
     under plain [armv8-a], where the instruction is unavailable, so it takes the same
     [__ARM_FEATURE_FP16_VECTOR_ARITHMETIC] guard the fp16 FMA arm does. It is also the arm that
     MATTERS: gcc scalarizes a 16-bit vector COMPARISON on aarch64 whatever type it is spelled in
     ([_Float16] and [__fp16] alike, at -O2 and -O3), widening every lane to float, so the blend
     that costs three vector instructions at f32/f64 costs 172 scalar ones at fp16 -- which is the
     one cell of [test/operations/cc_march_census]'s matrix this arm exists to close. *)
  let vec_minmax_builtin ~prec ~lanes ~op : (string * (dst:string -> src:string -> string)) list =
    let name = match op with Ops.Max -> "fmax" | _ -> "fmin" in
    let builtin suffix = Printf.sprintf "__builtin_aarch64_%s%s" name suffix in
    let neon suffix =
      ( Printf.sprintf "defined(__aarch64__) && __has_builtin(%s)" (builtin suffix),
        fun ~dst ~src -> Printf.sprintf "%s = %s(%s, %s);" dst (builtin suffix) dst src )
    in
    (* Typed in [__fp16], a distinct type from the [_Float16] the emitted vectors carry, so the arm
       casts both ways through a block-local typedef -- the same shape [neon_half_render] uses for
       the fp16 FMA builtin, and for the same reason (gh-ocannl-621). *)
    let neon_half suffix ~bytes =
      ( Printf.sprintf
          "HAS_NATIVE_FLOAT16 && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && __has_builtin(%s)"
          (builtin suffix),
        fun ~dst ~src ->
          Printf.sprintf
            "{ typedef __fp16 ocannl_hfv__ __attribute__((vector_size(%d))); %s = \
             (__typeof__(%s))%s((ocannl_hfv__)%s, (ocannl_hfv__)%s); }"
            bytes dst dst (builtin suffix) dst src )
    in
    match (prec, lanes) with
    | Ops.Single_prec _, 4 -> [ neon "v4sf" ]
    | Ops.Double_prec _, 2 -> [ neon "v2df" ]
    | Ops.Half_prec _, 8 -> [ neon_half "v8hf" ~bytes:16 ]
    | _ -> []

  let vec_acc_combine ~prec ~lanes ~op ~dst ~src =
    match op with
    | Ops.Add | Ops.Sub | Ops.Mul | Ops.Div ->
        let inf = match op with Ops.Add -> " + " | Sub -> " - " | Mul -> " * " | _ -> " / " in
        PPrint.string (Printf.sprintf "%s = %s%s%s;" dst dst inf src)
    | Ops.Max | Ops.Min -> (
        let open PPrint in
        let cmp = match op with Ops.Max -> ">=" | _ -> "<=" in
        let blend =
          string
            (Printf.sprintf
               "{ __typeof__(%s %s %s) ocannl_m__ = (%s %s %s) | (%s != %s); %s = \
                (__typeof__(%s))((ocannl_m__ & (__typeof__(ocannl_m__))%s) | (~ocannl_m__ & \
                (__typeof__(ocannl_m__))%s)); }"
               dst cmp src dst cmp src src src dst dst dst src)
        in
        match vec_minmax_builtin ~prec ~lanes ~op with
        | [] -> blend
        | arms ->
            let first = ref true in
            let chain =
              List.map arms ~f:(fun (guard, render) ->
                  let kw = if !first then "#if " else "#elif " in
                  first := false;
                  string (kw ^ guard) ^^ hardline ^^ string (render ~dst ~src) ^^ hardline)
              |> concat
            in
            chain ^^ string "#else" ^^ hardline ^^ blend ^^ hardline ^^ string "#endif")
    | _ -> invalid_arg "C_syntax.vec_acc_combine: not an accumulation operator"

  (* The target builtins spelling a WHOLE-VECTOR fused multiply-add for this (compute precision,
     lane count), most preferred first: each is the preprocessor condition under which the arm
     exists, paired with a rendering of the statement [dst = fma(a, b, dst)]. Empty where nothing is
     known; the guards within one entry are mutually exclusive, so the list becomes a chain of
     [#elif]s.

     Why this table exists at all: gcc has no generic vector [fma] builtin (clang's
     [__builtin_elementwise_fma], the first arm of [vec_acc_fma], is a clang extension), so on gcc
     the accumulator update falls to the per-lane loop — which gcc mis-compiles in two distinct
     ways, both measured per width on the emitted micro-kernel (gh-ocannl-614, gh-ocannl-621):

     - it SPILLS. -O3 unrolls the k-loop and puts the C-tile in memory: at f32/8 lanes, 18
     instructions / 8 vector FMAs / 0 stack references per k step become 200 / 4 vector + 48 scalar
     / 8, which [bin/narrow_gebp_bench] measures as 12.6 GFLOP/s against 128 (gh-ocannl-614). The
     cause is the lane subscripting itself — straight-line lane stores, a lane-indexed scratch
     array, a [(vtyp){...}] constructor of per-lane [fmaf] calls and [#pragma GCC unroll 1] all
     spill the same way, while ANY form reaching gcc as one vector operation costs the -O2
     instruction count at -O3. Worst at f32/4 lanes on aarch64: 294 instructions / 8 vector + 48
     scalar FMAs / 58 stack references against the builtin's 26 / 16 / 0. - or it SCALARIZES. SLP
     does not reassemble the lane loop at all, so one lanes-wide FMA becomes [lanes] scalar ones — a
     lane-count arithmetic loss with no spill to give it away. gcc does this to f64/2 lanes on x86
     AND aarch64 at both -O2 and -O3, so [cc_backend_optimization_level=2] does not diagnose it the
     way it diagnoses a spill, and to f64/8 lanes under AVX-512 at -O3.

     Which is why these are builtins and not the obvious [dst = a * b + dst]: that reaches gcc as
     one vector operation too, and allocates perfectly, but it is only MAYBE contracted into an FMA
     — under [cc_backend_fp_contract=off] it measurably is not (a mul and an add, two roundings,
     verified against these builtins), and then each accumulator UPDATE in the vector body would
     round twice where the scalar peel and the serial fallback round once. Be precise about which
     promise that breaks: it is not whole-result bit equality, which never held for a reduction
     anyway. Vector accumulation reassociates -- [vec_acc_grid_fold] folds the register grid into
     one accumulator and [vec_acc_lane_fold] then chains that vector's lanes into a scalar, so the
     lanes accumulate independently and are combined at the end, a different summation order from
     the serial fallback's (which is why the [reproducible] profile pins [cc_vector_bytes=0]). What
     the three paths do promise each other is the rounding of each individual UPDATE: an FMA rounds
     once, and all three spell that same single-rounded operation, so they differ by summation order
     alone and not by the semantics of each step. Every builtin here is fused by definition, so that
     per-update promise holds under every flag; each was checked to render exactly one fused
     instruction at [-ffp-contract=off], computing [a * b + dst]. The masked forms are spelled the
     way gcc's own [<immintrin.h>] spells them: with an all-ones mask and [_MM_FROUND_CUR_DIRECTION]
     (= 4, i.e. obey MXCSR rather than an embedded rounding mode) these calls are character for
     character what [_mm512_fmadd_ps], [_mm512_fmadd_pd], [_mm_fmadd_ph], [_mm256_fmadd_ph] and
     [_mm512_fmadd_ph] expand to, so their semantics are the ISA's rather than something inferred
     here.

     The AVX-512, AVX512-FP16 and aarch64 rows could not be RUN on the machine they were added from
     (an Arrow Lake-HX part, where AVX-512 is fused off entirely; QEMU's TCG implements neither
     AVX-512 nor AVX512-FP16), so they carry a second guard the AVX/AVX2 rows do not:
     [__has_builtin], which on gcc tracks the enabled target features exactly.

     It cannot catch a wrong signature — that is what the compile checks are for — but it turns
     "this compiler spells it differently" into a fall-through to the per-lane arm rather than a
     failed kernel compile. The cost is that a gcc older than 10, which has no [__has_builtin],
     takes the per-lane arm at those widths (the prelude shims the macro to 0).

     The AVX-512 f32 x 16 row has since been executed, on Zen 5 (gcc 15, which has no
     [__builtin_elementwise_fma], so the chain reaches this arm): correct to the bit against the
     32-byte rendering, and 225.7 against 130.5 GFLOP/s on the packed f32 GEBP once
     [cc_vector_bytes] stopped capping the width at 32. The AVX512-FP16 rows remain unexecuted and
     will stay that way here — no AMD part implements AVX512-FP16, and this fleet's only native-fp16
     target is ARM, which takes the NEON rows instead. *)
  let vec_fma_builtin ~prec ~lanes : (string * (dst:string -> a:string -> b:string -> string)) list
      =
    let call name extra ~dst ~a ~b =
      Printf.sprintf "%s = %s(%s, %s, %s%s);" dst name a b dst extra
    in
    let has name = Printf.sprintf "__has_builtin(%s)" name in
    (* [__FMA__] is x86-only, so these arms self-select away on every other target. The AVX/AVX2
       widths are the shipped, hardware-measured ones (gh-ocannl-614). *)
    let x86 name = ("defined(__FMA__)", call name "") in
    (* AVX-512F: the 512-bit forms take an all-ones write mask and a rounding mode
       ([_MM_FROUND_CUR_DIRECTION] = 4). [__mmask16]/[__mmask8] are [<immintrin.h>] names for plain
       integer types, which is what a kernel writes — it includes no intrinsics header. *)
    let avx512 name mask =
      ( Printf.sprintf "defined(__AVX512F__) && %s" (has name),
        call name (Printf.sprintf ", (%s)-1, 4" mask) )
    in
    (* AVX512-FP16: the VL (128/256-bit) forms take a mask and no rounding argument, the 512-bit one
       takes both. *)
    let avx512fp16 ?(vl = true) ?(round = false) name mask =
      ( Printf.sprintf "HAS_NATIVE_FLOAT16 && defined(__AVX512FP16__)%s && %s"
          (if vl then " && defined(__AVX512VL__)" else "")
          (has name),
        call name (Printf.sprintf ", (%s)-1%s" mask (if round then ", 4" else "")) )
    in
    (* aarch64: gcc's internal NEON [fma] builtins. Undocumented, which is the whole reason for the
       [__has_builtin] guard — [<arm_neon.h>]'s [vfmaq_f32] would be the documented spelling but the
       header cannot be included (its native vector typedefs collide with the emitted pack-struct
       typedefs; see [Builtins_cc.includes]). Inline [fmla] asm was the other candidate and censused
       three instructions per k step worse, so the builtin wins on both counts. *)
    let neon name =
      ( Printf.sprintf "defined(__aarch64__) && defined(__ARM_FEATURE_FMA) && %s" (has name),
        call name "" )
    in
    (* The fp16 NEON builtin is typed in [__fp16], a DISTINCT type from the [_Float16] the emitted
       vectors carry, so its arm casts both ways through a block-local typedef rather than being a
       plain call. (The f32/f64 builtins take the element types the emission already uses.) *)
    let neon_half_render name ~bytes ~dst ~a ~b =
      Printf.sprintf
        "{ typedef __fp16 ocannl_hfv__ __attribute__((vector_size(%d))); %s = \
         (__typeof__(%s))%s((ocannl_hfv__)%s, (ocannl_hfv__)%s, (ocannl_hfv__)%s); }"
        bytes dst dst name a b dst
    in
    let neon_half name ~bytes =
      ( Printf.sprintf "HAS_NATIVE_FLOAT16 && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && %s"
          (has name),
        neon_half_render name ~bytes )
    in
    match (prec, lanes) with
    | Ops.Single_prec _, 4 -> [ x86 "__builtin_ia32_vfmaddps"; neon "__builtin_aarch64_fmav4sf" ]
    | Ops.Single_prec _, 8 -> [ x86 "__builtin_ia32_vfmaddps256" ]
    | Ops.Single_prec _, 16 -> [ avx512 "__builtin_ia32_vfmaddps512_mask" "unsigned short" ]
    | Ops.Double_prec _, 2 -> [ x86 "__builtin_ia32_vfmaddpd"; neon "__builtin_aarch64_fmav2df" ]
    | Ops.Double_prec _, 4 -> [ x86 "__builtin_ia32_vfmaddpd256" ]
    | Ops.Double_prec _, 8 -> [ avx512 "__builtin_ia32_vfmaddpd512_mask" "unsigned char" ]
    | Ops.Half_prec _, 8 ->
        [
          avx512fp16 "__builtin_ia32_vfmaddph128_mask" "unsigned char";
          neon_half "__builtin_aarch64_fmav8hf" ~bytes:16;
        ]
    | Ops.Half_prec _, 16 -> [ avx512fp16 "__builtin_ia32_vfmaddph256_mask" "unsigned short" ]
    | Ops.Half_prec _, 32 ->
        [ avx512fp16 ~vl:false ~round:true "__builtin_ia32_vfmaddph512_mask" "unsigned int" ]
    | _ -> []

  (* [dst = fma(a, b, dst)] elementwise-fused on vector registers (single rounding, matching the
     scalar path's [fmaf]/[fma]): clang's [__builtin_elementwise_fma] where available, else the
     target's whole-vector builtins ({!vec_fma_builtin}), else the per-lane fused loop. *)
  let vec_acc_fma ~prec ~lanes ~dst ~a ~b =
    let open PPrint in
    (* At fp16 the per-lane fallback must be the *same* fused multiply-add the scalar path emits, or
       the arms of the [#if] -- and the vector rendering and its serial remainder -- would round
       differently: promoting to [fmaf] rounds at float and again at fp16, while a native fp16 FMA
       rounds once. [OCANNL_HALF_FMA] switches on the same target facts as the arms below (clang's
       elementwise builtin, then a native [__builtin_fmaf16]), so every configuration agrees by
       construction (gh-ocannl-516, gh-ocannl-621). *)
    let fma_fn =
      match prec with
      | Ops.Double_prec _ -> "fma"
      | Ops.Half_prec _ -> "OCANNL_HALF_FMA"
      | _ -> "fmaf"
    in
    let builtin_arms =
      vec_fma_builtin ~prec ~lanes
      |> List.map ~f:(fun (guard, render) ->
          string ("#elif " ^ guard) ^^ hardline ^^ string (render ~dst ~a ~b) ^^ hardline)
      |> concat
    in
    string "#if OCANNL_HAS_ELEMENTWISE_FMA"
    ^^ hardline
    ^^ string (Printf.sprintf "%s = __builtin_elementwise_fma(%s, %s, %s);" dst a b dst)
    ^^ hardline ^^ builtin_arms ^^ string "#else" ^^ hardline
    ^^ string
         (Printf.sprintf
            "for (int ocannl_l__ = 0; ocannl_l__ < %d; ++ocannl_l__) %s[ocannl_l__] = \
             %s(%s[ocannl_l__], %s[ocannl_l__], %s[ocannl_l__]);"
            lanes dst fma_fn a b dst)
    ^^ hardline ^^ string "#endif"

  (* Fold every register of the grid into [grid.(0).(0)] (a statement list; empty for a 1×1
     grid). *)
  let vec_acc_grid_fold ~prec ~lanes ~op (grid : string array array) : PPrint.document list =
    let dst = grid.(0).(0) in
    Array.concat_map grid ~f:Fn.id |> Array.to_list
    |> List.filter ~f:(fun s -> not (String.equal s dst))
    |> List.map ~f:(fun src -> vec_acc_combine ~prec ~lanes ~op ~dst ~src)

  (* Horizontal reduction of one vector register into the fresh scalar [out] (a statement list): a
     linear lane chain through the scalar combine — per-element semantics identical to the serial
     loop's. *)
  let vec_acc_lane_fold ~prec ~lanes ~op ~vname ~out : PPrint.document list =
    let open PPrint in
    string (Printf.sprintf "%s %s = %s[0];" (B.typ_of_prec prec) out vname)
    :: List.init (lanes - 1) ~f:(fun l ->
        string (out ^ " = ")
        ^^ B.binop_syntax prec op (string out) (string (Printf.sprintf "%s[%d]" vname (l + 1)))
        ^^ semi)

  (* Set by [compile_proc] so that [pp_ll] can consult per-node tracing info (e.g. to elide a
     [Zero_out] loop that the declaration's [= {0}] already covers). *)
  let current_traced_store : Low_level.traced_store option ref = ref None

  (* Set by [compile_proc]: the kernel's hardware-annotated loops with their positional slots
     (docs/proposals/axis-types-for-loops.md §1/§5), consulted by [pp_ll]'s [For_loop] case to
     render hardware index bindings. *)
  let current_hardware_axes : Low_level.hardware_axis_info list ref = ref []

  (* Every symbol bound by a loop in the routine, with its iteration range. The accumulator peel
     needs the COMPLEMENT: a guard symbol outside this set is bound outside every loop -- a static
     index parameter, a runtime extent -- so it cannot select among an enclosing level's iterations,
     which is what keeps gh-490's runtime-extent guard ([i < s]) peelable. The ranges are what lets
     it ask whether the accumulated cell tells two enclosing instances apart ([Affine.separates],
     gh-ocannl-721). *)
  let current_loop_bounds : (Indexing.symbol * (int * int)) list ref = ref []

  (* Set by [compile_proc]: nodes placed in workgroup-shared memory. Their declarations carry
     [shared_decl_prefix] and cannot use [= {0}] (not allowed for [__shared__]/[threadgroup]), so
     their [Zero_out] is never elided. *)
  let current_workgroup_shared : Set.M(Tn).t ref = ref (Set.empty (module Tn))

  (* Marked local accumulator tiles and the one currently being rendered by a backend fragment
     scope. Outside such a scope they retain ordinary local-array semantics. *)
  let current_simdgroup_fragments : Set.M(Tn).t ref = ref (Set.empty (module Tn))
  let rendered_simdgroup_fragments : Set.M(Tn).t ref = ref (Set.empty (module Tn))

  (* Set by [compile_proc]: nodes stored XOR-swizzled ([Schedule.Stage ~swizzle], see
     {!Low_level.optimized.swizzled}). Scalar element accesses go through [pp_tn_offset] below;
     renderings that assume row-major storage (contiguous vector loads/stores, [Tile_mma] intrinsic
     / register-tiled / fragment paths) must decline these nodes. *)
  let current_swizzled : Low_level.swizzle_kind Map.M(Tn).t ref = ref (Map.empty (module Tn))
  let swizzle_of tn = Map.find !current_swizzled tn
  let is_swizzled tn = Option.is_some (swizzle_of tn)

  (* Set by [compile_proc]: software-pipelined staged tiles ([Schedule.Stage ~pipeline_depth], see
     {!Low_level.optimized.pipelined}), allocated as [pt_depth] rotating copies with every element
     access offset by a buffer-selection term ([pp_pipelined_rotation] below). Renderings that
     assume single-copy storage (contiguous vector loads/stores, the register-tiled [Tile_mma] path)
     must decline these nodes; the intrinsic [Tile_mma] arms are fine — their operand pointers carry
     the rotation term. *)
  let current_pipelined : Low_level.pipelined_tile Map.M(Tn).t ref = ref (Map.empty (module Tn))
  let is_pipelined tn = Map.mem !current_pipelined tn

  (* gh-487 phase 2: the pipelined tiles whose staging copies render asynchronously this proc
     ([B.async_copy] provided, no kernel logging, element size the hardware can copy). Decided once
     by [compile_proc]; membership drives both the [Set] case's copy emission and the rotor loop's
     wait+barrier prefix, so the two can never disagree about whether a wait is needed. *)
  let current_async_tiles : Set.M(Tn).t ref = ref (Set.empty (module Tn))

  (* Elements per 16-byte unit for [Low_level.Swizzle_b128]: [u] and its log, with [units] the
     (power-of-two, checked by [Schedule.Stage]) count of units in one row of [c] elements. *)
  let b128_units ~prec ~c =
    let u = 16 / Ops.prec_in_bytes prec in
    (u, Int.floor_log2 u, c / u)

  (* The flat-offset rendering for an element access of [tn]'s buffer: row-major [pp_array_offset],
     except for swizzled nodes, where the minor-axis index is permuted per row against the low bits
     of the linearized row prefix [P] — a bijection, so all-element traversals (zeroing) and matched
     read/write pairs are unaffected while same-column accesses from consecutive rows land in
     distinct shared-memory banks. Two granularities (gh-ocannl-481 item 3, D1):

     - [Swizzle_elem]: [P*C + col] becomes [P*C + (col ^ (P & (C-1)))], [C] the minor dim. -
     [Swizzle_b128]: the column's 16-byte-unit index is XORed instead, the offset within the unit
     untouched — [P*C + (((col/u ^ (P & (U-1))) * u) + col%u)] with [u] elements per unit and [U]
     units per row. Whole 16-byte units stay contiguous and 16-byte-aligned, which is what makes the
     layout simultaneously bank-de-conflicted and [ldmatrix]-loadable.

     The prefix expression is emitted twice; downstream C compilers CSE it. *)
  let pp_tn_offset tn (idcs, dims) =
    let open PPrint in
    let n = Array.length idcs in
    match swizzle_of tn with
    | None -> pp_array_offset (idcs, dims)
    | Some _ when n < 2 -> pp_array_offset (idcs, dims)
    | Some kind -> (
        let c = dims.(n - 1) in
        let prefix_doc =
          pp_array_offset (Array.sub idcs ~pos:0 ~len:(n - 1), Array.sub dims ~pos:0 ~len:(n - 1))
        in
        let col_doc = pp_axis_index idcs.(n - 1) in
        let col_doc = if PPrint.is_empty col_doc then string "0" else col_doc in
        if PPrint.is_empty prefix_doc then col_doc
        else
          let row_base = parens prefix_doc ^^ string (" * " ^ Int.to_string c ^ " + ") in
          match kind with
          | Low_level.Swizzle_elem ->
              row_base
              ^^ parens
                   (parens col_doc ^^ string " ^ "
                   ^^ parens (parens prefix_doc ^^ string (" & " ^ Int.to_string (c - 1))))
          | Low_level.Swizzle_b128 ->
              let u, ushift, units = b128_units ~prec:(Lazy.force tn.Tn.storage_prec) ~c in
              (* C binds [+] tighter than [<<], so the shifted unit index needs its own parens. *)
              let unit_idx =
                parens
                  (parens (parens col_doc ^^ string (" >> " ^ Int.to_string ushift))
                  ^^ string " ^ "
                  ^^ parens (parens prefix_doc ^^ string (" & " ^ Int.to_string (units - 1))))
              in
              row_base
              ^^ parens
                   (parens (unit_idx ^^ string (" << " ^ Int.to_string ushift))
                   ^^ string " + "
                   ^^ parens (parens col_doc ^^ string (" & " ^ Int.to_string (u - 1)))))

  (* Whether an operand's storage layout is one the intrinsic arms can be handed: [`Plain] and
     [`Swizzled_b128] are (the latter only when the access is reconstructible from the pointer and
     leading dimension alone — see {!type-mma_layout}), everything else names its decline reason. *)
  let operand_layout tn ~ld ~idcs ~dims : [ mma_layout | `Decline of string ] =
    match swizzle_of tn with
    | None -> `Plain
    | Some Low_level.Swizzle_elem ->
        `Decline "element-granularity swizzle (no intrinsic load form matches it)"
    | Some Low_level.Swizzle_b128 ->
        if Array.length dims <> 2 then `Decline "b128-swizzled operand of rank <> 2"
        else if ld <> dims.(1) then
          `Decline "b128-swizzled operand whose leading dimension is not its minor dim"
        else if not (Array.for_all idcs ~f:(function Indexing.Fixed_idx 0 -> true | _ -> false))
        then `Decline "b128-swizzled operand not accessed from the tile origin"
        else `Swizzled_b128

  (* An operand tuple carrying a decline reason cannot be handed to an emission hook. *)
  let narrow_operand (ptr, ld, space, layout) : mma_operand option =
    match layout with
    | `Decline _ -> None
    | (`Plain | `Swizzled_b128) as layout -> Some (ptr, ld, space, layout)

  let narrow_source (ld, space, layout) : mma_source option =
    match layout with
    | `Decline _ -> None
    | (`Plain | `Swizzled_b128) as layout -> Some (ld, space, layout)

  (* An addressed operand splits into what the hooks decide by and the address itself. *)
  let operand_source ((_, ld, space, layout) : mma_operand) : mma_source = (ld, space, layout)
  let operand_ptr ((ptr, _, _, _) : mma_operand) = ptr

  let operand_decline (_, _, _, layout) =
    match layout with `Decline reason -> Some reason | `Plain | `Swizzled_b128 -> None

  type active_mma_accumulator =
    | Active_fragment of Tn.t * string
    | Active_target of Tn.t * mma_operand

  let active_mma_accumulator : active_mma_accumulator option ref = ref None

  (* Set by [compile_proc]: outermost [Grid] loops eligible for pool-backed parallel rendering
     (docs/proposals/gh-ocannl-164.md), identified by index symbol. Empty unless
     [B.parallel_grid_syntax] renders in parallel. *)
  let current_parallel_grid : Set.M(Indexing.Symbol).t ref =
    ref (Set.empty (module Indexing.Symbol))

  (* Set by [compile_proc] alongside [current_parallel_grid]: per eligible [Grid] loop, the
     [Local]-placement tnodes privatized to per-chunk block-scope declarations inside that loop's
     chunk body (gh-ocannl-469: in-kernel packed operand tiles under a pool-parallel Grid). These
     are excluded from [compile_proc]'s function-scope [local_decls]. *)
  let current_grid_private : Tn.t list Map.M(Indexing.Symbol).t ref =
    ref (Map.empty (module Indexing.Symbol))

  (* Set by [compile_proc] alongside [current_parallel_grid]: [Local]-placement tnodes kept at
     function scope but accessed inside a pool-parallel Grid loop rendered as a blocks-extension
     [dispatch_apply] ([`Dispatch]). Blocks cannot refer to a declaration with an array type, but
     they capture pointers by value, so [local_decls] declares these behind a [const] pointer alias.
     Always empty for [`Openmp]/[`None]. *)
  let current_local_ptr_alias : Set.M(Tn).t ref = ref (Set.empty (module Tn))

  let declinef fmt =
    Printf.ksprintf
      (fun s ->
        if Lazy.force log_declines then
          Stdlib.Printf.eprintf "declined: %s: %s\n%!" !current_kernel_name s)
      fmt

  (* Shared traversal for the pool-parallel Grid analyses below: fires [access] for every
     tensor-node access event in program order (a [Set]'s right-hand side fires before its write).
     [Tile_mma] is traversed through its scalar [fallback]: every rendering of the statement
     (intrinsics, register tiling, lane-0 fallback) touches exactly the fallback's tensors over the
     fallback's index ranges, so the fallback IS the statement's access footprint. [on_stmt]
     observes the statement-level events the locals analysis keys on (opaque statements, scope
     declarations). [kind] tells the affine queries how to interpret the index vector: [`Whole] for
     accesses whose cells are not statically known ([Zero_out]'s every-cell write, data-dependent
     [Set_dynamic]/[Get_dynamic] slots, fired with empty indices), [`Vec] for vectorized writes
     whose last component is the base of a minor-axis run. *)
  let iter_local_accesses ~access ~on_stmt (root : Low_level.t) : unit =
    let rec go (llc : Low_level.t) =
      match llc with
      | Low_level.Noop | Comment _ -> ()
      | Staged_compilation _ | Workgroup_barrier -> on_stmt `Opaque
      | Tile_mma { fallback; _ } -> go fallback
      | Seq (a, b) ->
          go a;
          go b
      | For_loop { body; _ } -> go body
      | Scan_loop { carried; body; _ } ->
          (* The carried pair is declared by the construct itself, so under a pool-rendered [Grid]
             body it is per-chunk storage like a [Declare_local] there. *)
          List.iter carried ~f:(fun c ->
              on_stmt (`Declare_local c.Low_level.prev);
              on_stmt (`Declare_local c.Low_level.next);
              go_sc c.Low_level.init);
          go body
      | If { cond = c, _; body } ->
          go_sc c;
          go body
      | Zero_out tn -> access ~write:true ~kind:`Whole tn [||]
      | Set { tn; idcs; llsc; _ } ->
          go_sc llsc;
          access ~write:true ~kind:`Exact tn idcs
      | Set_dynamic { tn; dyn_value = v, _; llsc; _ } ->
          go_sc v;
          go_sc llsc;
          access ~write:true ~kind:`Whole tn [||]
      | Set_from_vec { tn; idcs; arg = a, _; _ } ->
          go_sc a;
          access ~write:true ~kind:`Vec tn idcs
      | Set_local (id, llsc) ->
          on_stmt (`Set_local id);
          go_sc llsc
      | Declare_local { id; _ } -> on_stmt (`Declare_local id)
    and go_sc (llsc : Low_level.scalar_t) =
      match llsc with
      | Local_scope { id; body; _ } ->
          on_stmt (`Declare_local id);
          go body
      | Get_local _ -> ()
      | Get (tn, idcs) -> access ~write:false ~kind:`Exact tn idcs
      | Get_dynamic { tn; dyn_value = v, _; _ } ->
          access ~write:false ~kind:`Whole tn [||];
          go_sc v
      | Get_merge_buffer _ -> ()
      | Ternop (_, (a, _), (b, _), (c, _)) ->
          go_sc a;
          go_sc b;
          go_sc c
      | Binop (_, (a, _), (b, _)) ->
          go_sc a;
          go_sc b
      | Unop (_, (a, _)) -> go_sc a
      | Constant _ | Constant_bits _ | Embed_index _ -> ()
    in
    go root

  (* Per-local access info under a candidate pool-parallel Grid loop's body. *)
  type grid_local_info = {
    gl_tn : Tn.t;
    mutable gl_written : bool;
    mutable gl_accs : (bool * [ `Exact | `Whole | `Vec ] * Indexing.axis_index array) list;
    mutable gl_count : int;
  }

  (* The privatization rule's write-dominance check: whether [tn]'s FIRST access (program order)
     under [body] is a standalone covering write — a [Zero_out], or an unguarded [Set] nest that
     rewrites the whole array and COMPLETES before any other access to [tn] executes. The completion
     requirement is structural (Codex P2 on PR #159): the covering [Set]'s per-axis coverage may
     only use loops that enclose NO other access to [tn] — descending from the body root, a loop
     stays usable while every access to [tn] in scope sits inside one child statement, and once
     siblings share the level (e.g. the pack nest followed by its consumer), the loops collected so
     far are discarded and coverage must come from the write's own nest. A write like [for x {
     tmp[x] = ..; use tmp[y] }] therefore declines: its only coverage loop [x] also interleaves the
     reads, which under per-chunk storage would observe missing prior iterations at chunk
     boundaries. Coverage per axis: a fresh usable-loop symbol of matching extent, or a mixed-radix
     affine combination of such symbols, each symbol used once across the index vector — the nest
     then enumerates every cell. Extra enclosing usable loops merely repeat the covering nest
     (required non-degenerate, or the write never executes). *)
  let first_access_standalone_covering (tn : Tn.t) (body : Low_level.t) : bool =
    let count_accesses (llc : Low_level.t) =
      let c = ref 0 in
      iter_local_accesses llc
        ~access:(fun ~write:_ ~kind:_ tn2 _ -> if tn2.Tn.uid = tn.Tn.uid then Int.incr c)
        ~on_stmt:(fun _ -> ());
      !c
    in
    let touches llc = count_accesses llc > 0 in
    let dims = Lazy.force tn.Tn.dims in
    (* Coverage by affine query ({!Affine.covers_box}): the nest enumerates every cell exactly once.
       Only the usable [loops] participate as the box environment — coverage through a loop that
       interleaves other accesses to [tn] must not count (see the completion requirement above). *)
    let covering ~loops (idcs : Indexing.axis_index array) =
      let range s =
        List.find_map loops ~f:(fun (s', from_, to_) ->
            if Indexing.equal_symbol s s' then Some (from_, to_) else None)
      in
      Affine.covers_box ~range ~dims idcs
    in
    let rec stmt (llc : Low_level.t) ~loops =
      match llc with
      | Low_level.Seq _ -> level (Low_level.flat_lines [ llc ]) ~loops
      | For_loop { index; from_; to_; body; _ } ->
          to_ >= from_ && level (Low_level.flat_lines [ body ]) ~loops:((index, from_, to_) :: loops)
      | Zero_out tn2 -> tn2.Tn.uid = tn.Tn.uid
      | Set { tn = tn2; idcs; _ } ->
          (* [count_accesses = 1]: the write itself, so the right-hand side does not read [tn] (a
             read-modify-write is not a covering first access). *)
          tn2.Tn.uid = tn.Tn.uid && count_accesses llc = 1 && covering ~loops idcs
      (* [If] = guarded (partial coverage); dynamic/vector writes, [Tile_mma] operand traffic and a
         scan's serial trajectory are conservatively never covering. *)
      | Noop | Comment _ | Staged_compilation _ | Workgroup_barrier | If _ | Set_dynamic _
      | Set_from_vec _ | Set_local _ | Declare_local _ | Tile_mma _ | Scan_loop _ ->
          false
    and level stmts ~loops =
      match List.filter stmts ~f:touches with
      | [] -> false
      | [ only ] -> stmt only ~loops
      | first :: _ :: _ ->
          (* Later siblings access [tn] too: they run after [first] completes, but the loops
             collected so far re-run them interleaved with [first] — coverage must come from
             [first]'s own nest. *)
          stmt first ~loops:[]
    in
    level (Low_level.flat_lines [ body ]) ~loops:[]

  (* Whether the body of an outermost [Grid] loop over [sym] tolerates its iterations being
     partitioned into chunks that execute on parallel CPU threads. Materialized accesses are already
     safe: [validate_parallel] requires every materialized write to cover [sym], so chunks write
     disjoint elements, and nests execute as separate parallel loops with a join in between
     (stronger than the GPU's single launch). The hazards are the stack arrays of [Local]-placement
     nodes: on GPU each thread gets a private copy, so a GPU-valid kernel may legally write them
     grid-invariantly (identical values per iteration) -- under one shared function-scope array that
     is a data race. A local written under the loop must satisfy one of:

     - Shared rule: every access to it (read or write) mentions [sym], and all accesses agree on
     every index component that mentions [sym] -- the same agreement rule as the default annotator's
     hazard analysis (mere mention is not enough, e.g. a stencil write [tmp[i]] + read [tmp[i-1]]
     both mention [sym] but reach across iterations). Distinct iterations then touch disjoint cells
     of one function-scope array. - Privatization rule: ALL of the node's accesses in the kernel sit
     inside this loop's body, and its first access per iteration is a standalone covering write -- a
     whole-array rewrite that completes before any other access to it executes (the write-dominance
     check of [first_access_standalone_covering]) -- so no value flows between iterations and each
     chunk can own a block-scope copy. This is what in-kernel packing [Stage] tiles satisfy
     (gh-ocannl-469): the pack nest fully rewrites the tile before the micro-kernel reads it. The
     combined per-chunk footprint is capped by [per_chunk_private_bytes_cap] (pool worker stacks).

     Under [`Dispatch] (the blocks extension), a block cannot refer to a declaration with an array
     type at all -- even read-only -- but it captures pointers by value, so every non-privatized
     local accessed under the loop is recorded for a pointer-alias declaration
     ([current_local_ptr_alias]). [Set_local] scope locals must have their declaration within the
     loop body (block scope = per-chunk storage). Opaque statements and barriers disqualify.

     Returns [Some (privatized, ptr_aliased)] when the loop can render in parallel. *)
  let parallel_grid_safe ~sym ~grid_range ~(global_counts : int Hashtbl.M(Int).t)
      (body : Low_level.t) : (Tn.t list * Tn.t list) option =
    let plc = placements () in
    let is_local tn =
      (not (Tn.Placements.is_virtual_force plc tn 431))
      && not (Tn.Placements.is_materialized_force plc tn 432)
    in
    let mentions_comp = Indexing.axis_index_mentions_symbol sym in
    let loop_ident = Indexing.symbol_ident sym in
    let locals : grid_local_info Hashtbl.M(Int).t = Hashtbl.create (module Int) in
    (* Keyed by the full scope id -- node and integer, the identity the rendered C name carries. *)
    let declared_scopes = Hash_set.create (module Low_level.Scope_id) in
    let ok = ref true in
    let opaque = ref false in
    let escaped_scope = ref false in
    let access ~write ~kind tn idcs =
      if is_local tn then (
        let info =
          Hashtbl.find_or_add locals tn.Tn.uid ~default:(fun () ->
              { gl_tn = tn; gl_written = false; gl_accs = []; gl_count = 0 })
        in
        info.gl_count <- info.gl_count + 1;
        info.gl_written <- info.gl_written || write;
        info.gl_accs <- (write, kind, idcs) :: info.gl_accs)
    in
    iter_local_accesses body ~access ~on_stmt:(function
      | `Opaque ->
          ok := false;
          opaque := true
      | `Declare_local id -> Hash_set.add declared_scopes id
      | `Set_local id ->
          if not (Hash_set.mem declared_scopes id) then (
            ok := false;
            escaped_scope := true));
    if not !ok then (
      if !opaque then
        declinef "Grid loop %s stays serial: opaque statement or barrier under the loop body"
          loop_ident;
      if !escaped_scope then
        declinef
          "Grid loop %s stays serial: a scope local is set under the loop but declared outside it \
           (function-scope storage would race across chunks)"
          loop_ident;
      None)
    else
      let privatized = ref [] and ptr_aliased = ref [] in
      let private_bytes = ref 0 in
      (* The box environment for the affine conflict query: the Grid loop itself plus every loop
         bound under its body. Loops enclosing the Grid loop are shared across chunks (chunks of one
         dispatch execute under the same outer-iteration values, with a join before the next), which
         is exactly the query's treatment of unlisted symbols. *)
      let env = (sym, grid_range) :: Low_level.loop_bounds body in
      let range s = List.Assoc.find env s ~equal:Indexing.equal_symbol in
      let dup s = List.Assoc.mem env s ~equal:Indexing.equal_symbol in
      let feasible =
        Hashtbl.for_all locals ~f:(fun info ->
            (* The legacy procedural shared rule (every access mentions [sym] and all accesses agree
               on every mentioning component), kept for [legality_crosscheck]. *)
            let procedural_shared_ok () =
              (not info.gl_written)
              ||
              let maps = List.map info.gl_accs ~f:(fun (_, _, m) -> m) in
              List.for_all maps ~f:(fun idcs -> Array.exists idcs ~f:mentions_comp)
              &&
              let rank = List.fold maps ~init:0 ~f:(fun m a -> max m (Array.length a)) in
              let agree = ref true in
              for p = 0 to rank - 1 do
                let comps =
                  List.map maps ~f:(fun a ->
                      if p < Array.length a then a.(p) else Indexing.Fixed_idx 0)
                in
                if List.exists comps ~f:mentions_comp then
                  match comps with
                  | [] -> ()
                  | c0 :: rest ->
                      if not (List.for_all rest ~f:(Indexing.equal_axis_index c0)) then
                        agree := false
              done;
              !agree
            in
            (* Shared rule by affine query: chunks share one function-scope array, so every access
               pair involving a write must have its conflicts confined to a single chunk of [sym]
               ([Same_thread]) or be disjoint outright — which also admits patterns the agreement
               rule could only decline (constant-offset or strided-disjoint cells). *)
            let shared_ok =
              (not info.gl_written)
              ||
              let interp (kind, idcs) =
                match kind with
                | `Whole -> None
                | `Exact -> Some idcs
                | `Vec ->
                    if Array.is_empty idcs then None
                    else
                      let m = Array.copy idcs in
                      m.(Array.length m - 1) <- Indexing.Sub_axis;
                      Some m
              in
              let witness = ref "" in
              let q =
                List.for_all info.gl_accs ~f:(fun (wx, kx, mx) ->
                    List.for_all info.gl_accs ~f:(fun (wy, ky, my) ->
                        (not (wx || wy))
                        ||
                        match (interp (kx, mx), interp (ky, my)) with
                        | Some l, Some r -> (
                            match
                              Affine.pair_conflict ~range ~dup_left:dup ~dup_right:dup
                                ~pairs:[ (sym, sym) ]
                                ~left:l ~right:r
                            with
                            | Affine.Disjoint | Affine.Same_thread -> true
                            | Affine.Cross_thread wit ->
                                witness := wit;
                                false)
                        | _ ->
                            witness := "statically unknown cells (whole-node or dynamic access)";
                            false))
              in
              Affine.crosscheck ~site:"cc pool-parallel shared rule"
                ~context:(Tn.debug_name info.gl_tn ^ " under Grid loop " ^ loop_ident)
                ~procedural_safe:procedural_shared_ok ~query_safe:q ~witness:!witness;
              q
            in
            if shared_ok then (
              (match B.parallel_grid_syntax with
              | `Dispatch -> ptr_aliased := info.gl_tn :: !ptr_aliased
              | `Openmp | `None -> ());
              true)
            else
              let all_inside =
                match Hashtbl.find global_counts info.gl_tn.Tn.uid with
                | Some total -> total = info.gl_count
                | None -> false
              in
              if not all_inside then (
                declinef
                  "Grid loop %s stays serial: local %s is written grid-variantly (fails the shared \
                   rule) and is also accessed outside the loop body (fails the privatization rule)"
                  loop_ident (Tn.debug_name info.gl_tn);
                false)
              else if not (first_access_standalone_covering info.gl_tn body) then (
                declinef
                  "Grid loop %s stays serial: local %s fails the shared rule, and its first access \
                   per iteration is not a standalone covering write (fails the privatization rule)"
                  loop_ident (Tn.debug_name info.gl_tn);
                false)
              else (
                private_bytes :=
                  !private_bytes
                  + Tn.num_elems info.gl_tn
                    * Ops.prec_in_bytes (Lazy.force info.gl_tn.Tn.storage_prec);
                privatized := info.gl_tn :: !privatized;
                let fits = !private_bytes <= Lazy.force per_chunk_private_bytes_cap in
                if not fits then
                  declinef
                    "Grid loop %s stays serial: privatizing local %s brings the combined per-chunk \
                     tile footprint to %d bytes, over cc_grid_private_bytes_cap = %d"
                    loop_ident (Tn.debug_name info.gl_tn) !private_bytes
                    (Lazy.force per_chunk_private_bytes_cap);
                fits))
      in
      if feasible then Some (!privatized, !ptr_aliased) else None

  (* The outermost [Grid] loops safe to render in parallel, with each loop's privatized locals and
     (under [`Dispatch]) the pointer-aliased function-scope locals. Nested [Grid] loops render
     serially inside a chunk (still correct: write coverage holds per grid index). Runtime kernel
     logging writes to a shared FILE, so parallel rendering is skipped under
     [debug_log_from_routines]. *)
  let collect_parallel_grid (llc : Low_level.t) :
      Set.M(Indexing.Symbol).t * Tn.t list Map.M(Indexing.Symbol).t * Set.M(Tn).t =
    let empty =
      (Set.empty (module Indexing.Symbol), Map.empty (module Indexing.Symbol), Set.empty (module Tn))
    in
    if
      Poly.equal B.parallel_grid_syntax `None
      || B.parallel_grid_chunks <= 1 || Utils.debug_log_from_routines ()
    then empty
    else
      let plc = placements () in
      let is_local tn =
        (not (Tn.Placements.is_virtual_force plc tn 431))
        && not (Tn.Placements.is_materialized_force plc tn 432)
      in
      (* Whole-kernel access counts per local, so [parallel_grid_safe] can tell when a local's
         accesses all sit inside one loop's body (the privatization rule). Same traversal, so the
         counts are comparable by construction. *)
      let global_counts : int Hashtbl.M(Int).t = Hashtbl.create (module Int) in
      iter_local_accesses llc
        ~access:(fun ~write:_ ~kind:_ tn _ ->
          if is_local tn then Hashtbl.incr global_counts tn.Tn.uid)
        ~on_stmt:(fun _ -> ());
      let syms = ref (Set.empty (module Indexing.Symbol)) in
      let privs = ref (Map.empty (module Indexing.Symbol)) in
      let aliases = ref (Set.empty (module Tn)) in
      let rec go (llc : Low_level.t) =
        match llc with
        | Low_level.For_loop { axis = Grid; index; from_; to_; body; _ } -> (
            if from_ = 0 && to_ >= 1 then
              match parallel_grid_safe ~sym:index ~grid_range:(from_, to_) ~global_counts body with
              | Some (privatized, ptr_aliased) ->
                  syms := Set.add !syms index;
                  if not (List.is_empty privatized) then
                    privs := Map.set !privs ~key:index ~data:privatized;
                  aliases := List.fold ptr_aliased ~init:!aliases ~f:Set.add
              | None -> ())
        | For_loop { body; _ } -> go body
        | If { body; _ } -> go body
        | Seq (a, b) ->
            go a;
            go b
        | _ -> ()
      in
      go llc;
      (!syms, !privs, !aliases)

  (* Renders a [Local]-placement (routine-scope scratch) array declaration; shared by
     [compile_proc]'s function-scope [local_decls] and the per-chunk declarations of pool-parallel
     [Grid] loops. With [alias_ptr], the array gets a mangled name and a [const] pointer alias under
     the node's ident: [`Dispatch] blocks cannot refer to array declarations but capture pointers by
     value, and array indexing syntax is unchanged through the pointer. *)
  let local_array_decl ?(alias_ptr = false) ~zero_init (tn : Tn.t) : PPrint.document =
    let open PPrint in
    let typ = B.typ_of_prec @@ Lazy.force tn.Tn.storage_prec in
    let ident = get_ident tn in
    let arr_name = if alias_ptr then ident ^ "_mem__" else ident in
    let align_doc =
      (* SIMD alignment for plain stack arrays (gh-ocannl-164). *)
      match B.aligned_local_attr with
      | Some attr -> string (" " ^ attr)
      | None -> empty
    in
    let init_doc = if zero_init then string " = {0}" else empty in
    let decl =
      string typ ^^ space ^^ string arr_name
      ^^ brackets (OCaml.int (Tn.num_elems tn))
      ^^ align_doc ^^ init_doc ^^ semi
    in
    if alias_ptr then
      decl ^^ hardline ^^ string (Printf.sprintf "%s * const %s = %s;" typ ident arr_name)
    else decl

  (* A [Zero_out] loop is redundant when the array's declaration already initializes it with [=
     {0}]. That happens for local (non-virtual, non-materialized) declarations whose traced node has
     [zero_initialized_by_code = true]; see [compile_proc]'s [local_decls]. Materialized (on-device)
     nodes do NOT get [= {0}] (allocation handles zeroing, and is skipped exactly when
     [zero_initialized_by_code] is true), so their [Zero_out] loop must be kept. *)
  let zero_out_loop_redundant tn =
    match !current_traced_store with
    | None -> false
    | Some traced_store -> (
        match Hashtbl.find traced_store tn with
        | Some node ->
            let plc = placements () in
            node.Low_level.zero_initialized_by_code
            && (not
                  (Tn.Placements.is_virtual_force plc tn 337
                  || Tn.Placements.is_materialized_force plc tn 338))
            && not (Set.mem !current_workgroup_shared tn)
        | None -> false)

  (* Tensor node ids whose [Zero_out] has already been encountered during the current [pp_ll]
     traversal. Only the *first-touch* [Zero_out tn] is made redundant by the declaration's [= {0}];
     any later [Zero_out tn] (e.g. in [Zero_out tn; Set tn; Zero_out tn], or any [Zero_out] reached
     inside a loop body) is a genuine re-zero and must still emit its loop. Cleared per
     [compile_proc]. *)
  let zero_out_seen : int Hash_set.t = Hash_set.create (module Int)

  (* A whole-node zero immediately before a statement can be forwarded into that statement's
     localized serial accumulator when the statement owns every cell it closes. The affine check
     below establishes the dead-store side of the rewrite; [try_localize_serial_reduce] marks the
     seed consumed only after the localization itself has succeeded. Keeping those two decisions
     separate is load-bearing: a vector/SIMD rendering or any localizer refusal still needs the
     original [Zero_out] and opening node read. *)
  type localized_zero_seed = {
    lzs_tn : Tn.t;
    lzs_idcs : Indexing.axis_index array;
    lzs_repeated : Indexing.symbol list;
    mutable lzs_consumed : bool;
  }

  let current_localized_zero_seed : localized_zero_seed option ref = ref None

  (* Whether [body] contains an effect whose buffer accesses or ordering are deliberately opaque to
     the affine access list. A zero store may not move through either one. [Tile_mma] is rejected as
     a unit even though its fallback has an affine footprint: the selected intrinsic need not
     execute that fallback's scalar closing store. *)
  let has_opaque_zero_forwarding_effect (body : Low_level.t) =
    let rec stmt = function
      | Low_level.Staged_compilation _ | Workgroup_barrier | Tile_mma _ -> true
      | Seq (a, b) -> stmt a || stmt b
      | For_loop { body; _ } | If { body; _ } -> stmt body
      | Scan_loop { carried; body; _ } ->
          List.exists carried ~f:(fun c -> scalar c.Low_level.init) || stmt body
      | Set { llsc; _ } | Set_local (_, llsc) -> scalar llsc
      | Set_dynamic { dyn_value = value, _; llsc; _ } -> scalar value || scalar llsc
      | Set_from_vec { arg = value, _; _ } -> scalar value
      | Noop | Comment _ | Zero_out _ | Declare_local _ -> false
    and scalar = function
      | Low_level.Local_scope { body; _ } -> stmt body
      | Get_dynamic { dyn_value = value, _; _ } -> scalar value
      | Ternop (_, (a, _), (b, _), (c, _)) -> scalar a || scalar b || scalar c
      | Binop (_, (a, _), (b, _)) -> scalar a || scalar b
      | Unop (_, (a, _)) -> scalar a
      | Get _ | Get_local _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ ->
          false
    in
    stmt body

  (* The principled DSE condition for forwarding [Zero_out tn] into [next]: [next]'s only accesses
     of [tn] are one unconditional, same-statement, same-cell read-modify-write pair, and that
     statement's affine write map covers the whole node over its enclosing loop box. Loops omitted
     from the map are recorded as repeated-cell dimensions; the localizer remains the final
     authority and may consume the seed only when its accepted scope includes all of them. A dead
     enclosing loop is rejected because it executes no closing store. *)
  let localized_zero_seed_candidate (tn : Tn.t) (next : Low_level.t) : localized_zero_seed option =
    if has_opaque_zero_forwarding_effect next then None
    else
      let same_map = Array.equal Indexing.equal_axis_index in
      let accesses =
        Low_level.affine_accesses next
        |> List.filter ~f:(fun access -> Tn.equal access.Affine.a_tn tn)
      in
      match accesses with
      | [ ({ Affine.a_write = false; _ } as read); ({ a_write = true; _ } as write) ]
        when write.a_rmw && (not read.a_guarded) && (not write.a_guarded) && (not read.a_dynamic)
             && (not write.a_dynamic) && (not read.a_whole) && (not write.a_whole)
             && (not read.a_vec_last) && (not write.a_vec_last)
             && Affine.same_statement read.a_path write.a_path
             && Option.exists read.a_stmt_write ~f:(same_map write.a_map)
             && same_map read.a_map write.a_map ->
          let range symbol =
            List.find_map write.a_loops ~f:(fun (bound, range) ->
                if Indexing.equal_symbol symbol bound then Some range else None)
          in
          if List.exists write.a_loops ~f:(fun (_, (lo, hi)) -> hi < lo) then None
          else if Affine.covers_box ~range ~dims:(Lazy.force tn.Tn.dims) write.a_map then
            let repeated =
              List.filter_map write.a_loops ~f:(fun (symbol, (lo, hi)) ->
                  Option.some_if
                    (hi > lo
                    && not
                         (Array.exists write.a_map ~f:(Indexing.axis_index_mentions_symbol symbol))
                    )
                    symbol)
            in
            Some
              { lzs_tn = tn; lzs_idcs = write.a_map; lzs_repeated = repeated; lzs_consumed = false }
          else None
      | _ -> None

  (* Take one top-level statement without flattening the suffix. Optimized programs are commonly
     right-associated [Seq] trees, so this is constant work there; a left-associated prefix costs
     only the depth of that prefix and the residual tree retains the original suffix wholesale. *)
  let rec take_first_statement = function
    | Low_level.Seq (left, right) ->
        let first, left_rest = take_first_statement left in
        let rest =
          match left_rest with None -> right | Some left_rest -> Low_level.Seq (left_rest, right)
        in
        (first, Some rest)
    | statement -> (statement, None)

  (* Symbols of the serial [for] loops enclosing the current [pp_ll] rendering point (innermost
     first): maintained by [serial_loop] below, consulted by the [Set] case's
     [volatile_serial_accumulation] rule and by [pp_pipelined_rotation]. *)
  let serial_loop_stack : Indexing.symbol list ref = ref []

  (* gh-487: the buffer-selection term of a software-pipelined tile, prepended to the intra-copy
     offset ([pp_tn_offset] / [pp_array_offset]) at every access site. Reads select the copy the
     schedule loaded for the current rotor iteration ([rotor % depth]); writes select the copy being
     loaded for the next one ([(rotor + 1) % depth]) — the schedule emits the in-loop load nest one
     iteration ahead — except the prologue load before the rotor loop, which fills copy 0 (matching
     the first iteration's read of [from_ % depth = 0]; the rendering point's position relative to
     the rotor loop is exactly [serial_loop_stack] membership, [from_ = 0] by the tile loops'
     validation). This is the renderer half of the transform's bitwise-identity argument: the IR
     keeps single-copy indices, and every read resolves to the copy holding exactly the values the
     unpipelined form would read. *)
  let pp_pipelined_rotation ~is_write tn =
    let open PPrint in
    match Map.find !current_pipelined tn with
    | None -> empty
    | Some { Low_level.pt_depth; pt_rotor } ->
        if List.mem !serial_loop_stack pt_rotor ~equal:Indexing.equal_symbol then
          let counter =
            if is_write then parens (pp_symbol pt_rotor ^^ string " + 1") else pp_symbol pt_rotor
          in
          parens (counter ^^ string (" % " ^ Int.to_string pt_depth))
          ^^ string (" * " ^ Int.to_string (Tn.num_elems tn) ^ " + ")
        else if is_write then (* The prologue load: copy 0 at offset 0. *) empty
        else
          (* No copy of the rotating buffer is the right one here: outside the rotor loop there is
             no rotor value to select with. A schedule that puts such a read there is one this
             renderer cannot express, so it is a typed decline rather than a bare [invalid_arg]:
             reached as a tuner candidate (a pipelined twin surviving into a beam round), an untyped
             exception is classified [Fatal] under [strict_failure_classification] and ends the
             whole search. At the public [Context.compile] boundary [Schedule_outcome.raise_cause]
             still renders it as the same [Invalid_argument] carrying this message. *)
          raise
            (Schedule_outcome.Cause_at
               ( Schedule_outcome.Backend_codegen,
                 Schedule_outcome.Unsupported
                   {
                     feature = "pipelined tile read outside its rotor loop";
                     detail =
                       "C_syntax: read of pipelined tile " ^ Tn.debug_name tn
                       ^ " outside its rotor loop (the schedule only remaps reads within the \
                          staging scope)";
                   } ))

  (* Recognize the exact scalar fallback region synthesized by
     [Schedule.contract_tensorized_accumulator]:

     lane { if lane==0 { fragment <- target } }; for k_o { ... Tile_mma(d=fragment) ... }; lane { if
     lane==0 { target <- fragment } }.

     The marker on [optimized] makes this structural, not a proof over arbitrary user IR. A backend
     hook may replace the region; declining leaves the ordinary local-array rendering untouched. *)
  let render_mma_fragment_scope ~render (c : Low_level.t) : PPrint.document option =
    let open Low_level in
    let open PPrint in
    let nonempty =
      List.filter (flat_lines [ c ]) ~f:(function Noop | Comment _ -> false | _ -> true)
    in
    let is_lane0_guard lane = function
      | Binop (Ops.Cmpeq, (Embed_index (Indexing.Iterator guard_lane), _), (Constant zero, _))
      | Binop (Ops.Cmpeq, (Constant zero, _), (Embed_index (Indexing.Iterator guard_lane), _)) ->
          Indexing.equal_symbol lane guard_lane && Float.equal zero 0.
      | _ -> false
    in
    let unwrap_transfer ~into_fragment = function
      | For_loop
          { index = lane; from_ = 0; to_; axis = Workgroup; body = If { cond = cond, _; body }; _ }
        when is_lane0_guard lane cond ->
          let rec descend syms = function
            | For_loop { index; axis = Serial; body; _ } -> descend (index :: syms) body
            | Set { tn = fragment; llsc = Get (target, target_idcs); _ } when into_fragment ->
                Some (lane, to_ + 1, fragment, target, target_idcs, syms)
            | Set { tn = target; llsc = Get (fragment, _); idcs = target_idcs; _ }
              when not into_fragment ->
                Some (lane, to_ + 1, fragment, target, target_idcs, syms)
            | _ -> None
          in
          descend [] body
      | _ -> None
    in
    let rec collect_fragment_tiles fragment acc = function
      | Tile_mma { d = d, _; _ } as tm when Tn.equal d fragment -> tm :: acc
      | Seq (a, b) -> collect_fragment_tiles fragment (collect_fragment_tiles fragment acc a) b
      | For_loop { body; _ } | If { body; _ } -> collect_fragment_tiles fragment acc body
      | _ -> acc
    in
    let zero_symbols syms idx =
      let bound s = List.mem syms s ~equal:Indexing.equal_symbol in
      match idx with
      | Indexing.Iterator s when bound s -> Indexing.Fixed_idx 0
      | Indexing.Affine { symbols; offset } -> (
          let symbols = List.filter symbols ~f:(fun (_, s) -> not (bound s)) in
          match (symbols, offset) with
          | [], k -> Indexing.Fixed_idx k
          | [ (1, s) ], 0 -> Indexing.Iterator s
          | _ -> Indexing.Affine { symbols; offset })
      | other -> other
    in
    let operand_space tn =
      if Set.mem !current_workgroup_shared tn then `Shared
      else if Tn.Placements.is_materialized_force (placements ()) tn 441 then `Device
      else `Thread
    in
    (* The a/b operands of a fragment scope are described, not addressed: no pointer is rendered for
       them here. Their addresses belong to the [Tile_mma] sites inside the reduction loop, which
       re-render them where a pipelined tile's buffer rotation is in scope (gh-ocannl-487) — at this
       scope-level position, outside the rotor loop, it is not renderable at all. *)
    let source ld (tn, idcs) =
      let dims = Lazy.force tn.Tn.dims in
      let prec = Lazy.force tn.Tn.storage_prec in
      (prec, (ld, operand_space tn, operand_layout tn ~ld ~idcs ~dims))
    in
    (* The accumulator target, in contrast, is addressed at scope level: the fragment load and store
       bracketing the reduction read and write it. It is never a pipelined tile. *)
    let operand ld (tn, idcs) =
      let dims = Lazy.force tn.Tn.dims in
      let prec = Lazy.force tn.Tn.storage_prec in
      let ptr = parens (string (get_ident tn) ^^ string " + " ^^ pp_array_offset (idcs, dims)) in
      (prec, (ptr, ld, operand_space tn, operand_layout tn ~ld ~idcs ~dims))
    in
    (* The target's leading-dimension stride: the stride of the axis carrying the transfer nest's
       outermost (row) copy symbol — the minor dim in the plain case, larger when interior batch
       axes sit between the tile roles (gh-ocannl-528). The transfer nests are synthesized by
       [Schedule.contract_tensorized_accumulator], so the outermost serial copy loop is the
       fragment's row by construction. *)
    let target_ld_of ~(row_sym : Indexing.symbol option) (tn, idcs) =
      let dims = Lazy.force tn.Tn.dims in
      let rank = Array.length dims in
      let default = if rank >= 1 then dims.(rank - 1) else 1 in
      match row_sym with
      | None -> default
      | Some s -> (
          match Array.findi idcs ~f:(fun _ idx -> Indexing.axis_index_mentions_symbol s idx) with
          | Some (p, _) ->
              let ld = ref 1 in
              for x = p + 1 to rank - 1 do
                ld := !ld * dims.(x)
              done;
              !ld
          | None -> default)
    in
    match nonempty with
    | init :: reduction :: store :: rest -> (
        (* [rest] is any statements following the marked region in the same body — in particular the
           lane-0 epilogue statement fused by [Schedule.Fuse_epilogue] (gh-ocannl-486). They render
           after the region's rendering: the backend hooks end with the visibility barrier that the
           epilogue's re-reads of the just-stored target need. *)
        match
          (unwrap_transfer ~into_fragment:true init, unwrap_transfer ~into_fragment:false store)
        with
        | ( Some (lane1, width1, fragment, target, init_idcs, init_syms),
            Some (lane2, width2, fragment2, target2, store_idcs, _) )
          when Set.mem !current_simdgroup_fragments fragment
               && Tn.equal fragment fragment2 && Tn.equal target target2 && width1 = width2
               && Indexing.equal_symbol lane1 lane2
               && Array.equal Indexing.equal_axis_index init_idcs store_idcs -> (
            match collect_fragment_tiles fragment [] reduction with
            | [ Tile_mma { a; b; ta; tb; m; n; k; lda; ldb; _ } ] -> (
                let target_base = Array.map init_idcs ~f:(zero_symbols init_syms) in
                (* [init_syms] is innermost-first, so its last element is the outermost (row) copy
                   symbol; [init_idcs] still carries it. *)
                let t_ld = target_ld_of ~row_sym:(List.last init_syms) (target, init_idcs) in
                let d_prec, target_raw = operand t_ld (target, target_base) in
                let a_prec, a_raw = source lda a in
                let b_prec, b_raw = source ldb b in
                (* Operands whose layout no intrinsic load form matches (the element-granularity
                   swizzle; a b128 tile not addressable from its origin) decline the fragment hooks;
                   falling through to [None] keeps the ordinary rendering, whose [Tile_mma]s decline
                   to the swizzle-aware scalar fallback. A [`Swizzled_b128] staged tile does NOT
                   decline here (gh-ocannl-481 item 3, D3): the per-call hooks below judge it, and
                   an accumulator-resident wmma scope that cannot feed from [ldmatrix] declines
                   there — leaving the ordinary [mma_syntax] path, which can. *)
                match (narrow_operand target_raw, narrow_source a_raw, narrow_source b_raw) with
                | Some target_op, Some a_src, Some b_src -> (
                    let fragment_name = Printf.sprintf "__mma_fragment_%d" fragment.Tn.uid in
                    let render_with active =
                      let old = !active_mma_accumulator in
                      active_mma_accumulator := Some active;
                      Exn.protect
                        ~f:(fun () -> render reduction)
                        ~finally:(fun () -> active_mma_accumulator := old)
                    in
                    let fragment_doc =
                      if Utils.debug_log_from_routines () then None
                      else
                        Option.bind B.mma_fragment_syntax ~f:(fun emit ->
                            emit ~d_prec ~a_prec ~b_prec ~m ~n ~k ~fragment:fragment_name
                              ~target:target_op ~a:a_src ~b:b_src ~body:(fun () ->
                                render_with (Active_fragment (fragment, fragment_name))))
                    in
                    let rendered =
                      match fragment_doc with
                      | Some _ as doc -> doc
                      | None when Utils.debug_log_from_routines () -> None
                      | None ->
                          Option.bind B.mma_syntax ~f:(fun emit ->
                              (* When the fragment hook declined this call (HIP until its
                                 persistent-fragment mapping lands, CUDA's fp8 combination,
                                 unsupported precisions), preserve the existing per-[k_o] intrinsic
                                 path by aliasing the marked local back to the original target when
                                 that exact MMA call is supported. Unsupported calls retain the
                                 explicit lane-0 local-array fallback.

                                 This is [mma_syntax] as a support predicate: the emission it
                                 returns is never applied here — the reduction is re-rendered below,
                                 and each [Tile_mma] applies its own, in a position where the a/b
                                 addresses (a pipelined tile's rotating copy among them,
                                 gh-ocannl-487) exist. *)
                              match
                                emit ~d_prec ~a_prec ~b_prec ~ta ~tb ~m ~n ~k ~d:target_op ~a:a_src
                                  ~b:b_src
                              with
                              | Some _ -> Some (render_with (Active_target (fragment, target_op)))
                              | None -> None)
                    in
                    match rendered with
                    | Some doc ->
                        rendered_simdgroup_fragments :=
                          Set.add !rendered_simdgroup_fragments fragment;
                        Some (separate hardline (doc :: List.map rest ~f:render))
                    | None ->
                        (* Scalar/local fallback: lane 0 performs the synthesized transfers.
                           Hardware backends need the same outer visibility barriers as the ordinary
                           Tile_mma load/store path, so the init observes sibling zeroing and later
                           statements observe the final store. Serial C renderers need no
                           barriers. *)
                        let body_doc =
                          separate hardline
                            (List.map (init :: reduction :: store :: rest) ~f:render)
                        in
                        Some
                          (match B.barrier_syntax with
                          | Some barrier ->
                              string barrier ^^ hardline ^^ body_doc ^^ hardline ^^ string barrier
                          | None -> body_doc))
                | _ -> None)
            | _ -> None)
        | _ -> None)
    | _ -> None

  let try_mma_fragment_scope ~render c =
    if Set.is_empty !current_simdgroup_fragments then None else render_mma_fragment_scope ~render c

  let rec pp_ll ?(log_set_locals = true) ?(in_loop = false) (c : Low_level.t) : PPrint.document =
    let open PPrint in
    match c with
    | Low_level.Noop -> empty
    | Seq (c1, c2) -> (
        (* Note (gh-ocannl-639): a MATERIALIZED unroll of a reduction never reaches this arm as a
           run of adjacent same-cell [Set]s — [Sched.Unroll ~materialize:true] rewrites a recognized
           accumulation nest into the scope-local form at unroll time, where the provenance lives. A
           codegen-side run collapse here was tried and rejected: it could not tell unrolled copies
           of one source assignment from two user-authored adjacent accumulations, whose separate
           stores (and separate narrowings) are their semantics. *)
        match
          try_mma_fragment_scope ~render:(fun body -> pp_ll ~log_set_locals ~in_loop body) c
        with
        | Some doc -> doc
        | None ->
            let render_in_order left right =
              (* Rendering mutates occurrence-level state such as [zero_out_seen]; tuple component
                 evaluation order is unspecified, so spell the program order explicitly. *)
              let left_doc = pp_ll ~log_set_locals ~in_loop left in
              let right_doc = pp_ll ~log_set_locals ~in_loop right in
              (left_doc, right_doc)
            in
            let d1, d2 =
              match c1 with
              | Zero_out tn -> (
                  let next, rest = take_first_statement c2 in
                  match localized_zero_seed_candidate tn next with
                  | None -> render_in_order c1 c2
                  | Some seed ->
                      let saved = !current_localized_zero_seed in
                      current_localized_zero_seed := Some seed;
                      let next_doc =
                        Exn.protect
                          ~f:(fun () -> pp_ll ~log_set_locals ~in_loop next)
                          ~finally:(fun () -> current_localized_zero_seed := saved)
                      in
                      let zero_doc =
                        if seed.lzs_consumed then begin
                          (* Preserve the occurrence-level first-touch fact: a later [Zero_out] is a
                             genuine re-zero even though this one emitted no statement. *)
                          Hash_set.add zero_out_seen tn.Tn.uid;
                          empty
                        end
                        else pp_ll ~log_set_locals ~in_loop c1
                      in
                      let rest_doc =
                        Option.value_map rest ~default:empty ~f:(pp_ll ~log_set_locals ~in_loop)
                      in
                      let tail =
                        if PPrint.is_empty next_doc then rest_doc
                        else if PPrint.is_empty rest_doc then next_doc
                        else next_doc ^^ hardline ^^ rest_doc
                      in
                      (zero_doc, tail))
              | _ -> render_in_order c1 c2
            in
            (* Avoid extra hardlines if one side is empty *)
            if PPrint.is_empty d1 then d2
            else if PPrint.is_empty d2 then d1
            else d1 ^^ hardline ^^ d2)
    | For_loop { index = i; from_; to_; body; axis } -> (
        (* Rendering phase of docs/proposals/axis-types-for-loops.md (§5): [Serial] loops render as
           C [for] statements; [Grid]/[Workgroup]/[Workgroup_reduce] loops bind their index to the
           backend's hardware register (at the signed [loop_index_type] width, with an explicit cast
           from the unsigned register) when [B.hardware_index] provides one, and fall back to a
           serial loop otherwise (legal absent barriers); [Vectorized] loops render serially,
           prefixed with [B.vectorize_pragma] when non-empty; [Unrolled] loops emit the repeated
           body with the index bound as a per-block constant. *)
        let body_doc ?(body = body) () =
          let doc = ref (pp_ll ~log_set_locals ~in_loop:true body) in
          (if Utils.debug_log_from_routines () then
             let log_doc =
               let spec, arg_doc = B.log_index_arg (pp_symbol i) in
               let base_message = Printf.sprintf "index %s = %s\n" (symbol_ident i) spec in
               let log_param_doc =
                 Option.map B.kernel_log_param ~f:(fun (_, name) -> string name)
               in
               B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
                 ~base_message_literal:base_message ~args_docs:[ arg_doc ]
             in
             doc := log_doc ^^ hardline ^^ !doc);
          !doc
        in
        let serial_loop () =
          (* gh-490 guard-fusion peephole: a body-wrapping symbolic-extent guard [if (i < s)] (with
             [s] a kernel parameter, not an enclosing loop index) hoists into the loop header as [i
             <= to_ && i < s]. The iteration variable is monotone, so once the guard fails it stays
             false: exiting the loop is equivalent to skipping the remaining iterations. *)
          let fused =
            match body with
            | If
                {
                  cond =
                    ( Binop
                        ( Ops.Cmplt,
                          (Embed_index (Indexing.Iterator i'), _),
                          (Embed_index (Indexing.Iterator s), _) ),
                      _ );
                  body = inner;
                }
              when Indexing.equal_symbol i' i
                   && not (List.mem !serial_loop_stack s ~equal:Indexing.equal_symbol) ->
                Some (s, inner)
            | _ -> None
          in
          let guard_doc =
            match fused with
            | None -> empty
            | Some (s, _) -> string " && " ^^ pp_symbol i ^^ string " < " ^^ pp_symbol s
          in
          let header =
            string ("for (" ^ B.loop_index_type)
            ^^ pp_symbol i ^^ string " = " ^^ PPrint.OCaml.int from_ ^^ semi ^^ space ^^ pp_symbol i
            ^^ string " <= " ^^ PPrint.OCaml.int to_ ^^ guard_doc ^^ semi ^^ space ^^ string "++"
            ^^ pp_symbol i ^^ string ")"
          in
          serial_loop_stack := i :: !serial_loop_stack;
          let body_ir = body in
          let body =
            Exn.protect
              ~f:(fun () -> body_doc ?body:(Option.map fused ~f:snd) ())
              ~finally:(fun () -> serial_loop_stack := List.tl_exn !serial_loop_stack)
          in
          (* gh-487 phase 2: the rotor loop of async-staged pipelined tiles opens each iteration
             with wait-then-barrier — the calling thread's outstanding copies (the prefetch issued
             one iteration back, or the prologue) complete, then the barrier publishes them to the
             workgroup before the compute's reads. When the IR body still opens with its own
             [Workgroup_barrier] (un-elided form), only the wait is prepended; when
             [Schedule.elide_staged_barriers] dropped that opener against the previous iteration's
             trailing [Tile_mma] bracket — sound for synchronous stores, which that bracket
             publishes — the async arm re-inserts it, since a barrier BEFORE the wait publishes
             nothing (and the intrinsic's leading bracket cannot be relied on: the fragment-scope
             form opens it once outside the loop, not per iteration). *)
          let async_prefix =
            match B.async_copy with
            | Some ac
              when Map.existsi !current_pipelined ~f:(fun ~key ~data ->
                       Set.mem !current_async_tiles key
                       && Indexing.equal_symbol data.Low_level.pt_rotor i) ->
                let rec first_real = function
                  | (Low_level.Noop | Low_level.Comment _) :: tl -> first_real tl
                  | hd :: _ -> Some hd
                  | [] -> None
                in
                let has_leading_barrier =
                  match first_real (Low_level.flat_lines [ body_ir ]) with
                  | Some Low_level.Workgroup_barrier -> true
                  | _ -> false
                in
                string ac.ac_wait_all ^^ hardline
                ^^
                if has_leading_barrier then empty
                else string (Option.value_exn B.barrier_syntax) ^^ hardline
            | _ -> empty
          in
          group
            (header ^^ space ^^ lbrace
            ^^ nest 2 (hardline ^^ async_prefix ^^ body)
            ^^ hardline ^^ rbrace)
        in
        (* [fallback] renders the loop when the backend binds no hardware register for it — the
           dispatch passes the gh-ocannl-639 widening-then-serial fallback, so a hardware-annotated
           reduction serialized for lack of a hardware index (cc's [Workgroup_reduce] among others)
           keeps the same accumulator width as the [Serial] spelling of the same loop. *)
        let hardware_binding ?(fallback = serial_loop) kind =
          let slot =
            match
              List.find !current_hardware_axes ~f:(fun a ->
                  Indexing.equal_symbol a.Low_level.ha_index i)
            with
            | Some a -> a.Low_level.ha_slot
            | None ->
                invalid_arg
                  ("C_syntax.pp_ll: hardware-annotated loop " ^ symbol_ident i
                 ^ " missing from the slot table (pp_ll called outside compile_proc?)")
          in
          (* Grid slots [>= 2] fold onto the hardware [.z] register (gh-ocannl-643, the [Low_level]
             hardware-axis section comment): the loop binds [(z / stride) % cap], with the
             divisor/modulo omitted where trivial — a lone slot-2 loop renders the bare register
             exactly as before the fold existed. The backend is only ever asked for slots it names
             ([0..2]). *)
          let hw_slot, fold =
            match kind with
            | `Grid when slot >= 2 -> (2, Some (Low_level.grid_fold !current_hardware_axes ~slot))
            | _ -> (slot, None)
          in
          match B.hardware_index ~kind ~slot:hw_slot with
          | None -> fallback ()
          | Some reg ->
              let cast = "(" ^ String.strip B.loop_index_type ^ ")" in
              let expr =
                match fold with
                | None | Some (1, None) -> cast ^ reg
                | Some (stride, cap) ->
                    let e = cast ^ reg in
                    let e = if stride > 1 then e ^ " / " ^ Int.to_string stride else e in
                    Option.value_map cap ~default:e ~f:(fun c -> e ^ " % " ^ Int.to_string c)
              in
              let binding =
                string ("const " ^ B.loop_index_type) ^^ pp_symbol i ^^ string (" = " ^ expr ^ ";")
              in
              group
                (lbrace
                ^^ nest 2 (hardline ^^ binding ^^ hardline ^^ body_doc ())
                ^^ hardline ^^ rbrace)
        in
        let parallel_grid_loop () =
          (* Pool-backed Grid rendering (gh-ocannl-164): contiguous chunks of the grid extent
             execute on the process-global native pool ([dispatch_apply] / OpenMP); [Workgroup]
             loops and nested [Grid] loops stay serial inside a chunk. Eligibility (including [from_
             = 0]) was established by [collect_parallel_grid]. *)
          let extent = to_ + 1 in
          let target = min B.parallel_grid_chunks extent in
          let grain = (extent + target - 1) / target in
          let nchunks = (extent + grain - 1) / grain in
          let it = B.loop_index_type in
          let ident = symbol_ident i in
          let chunk = ident ^ "_chunk" and lo = ident ^ "_lo" and hi = ident ^ "_hi" in
          let decls =
            string
              (Printf.sprintf "const %s%s = (%s)(%s * %d);" it lo (String.strip it) chunk grain)
            ^^ hardline
            ^^ string
                 (Printf.sprintf "const %s%s = %s + %d <= %d ? %s + %d : %d;" it hi lo grain extent
                    lo grain extent)
          in
          (* Locals privatized to this loop (see [parallel_grid_safe]): block-scope arrays inside
             the chunk body, one copy per chunk -- iterations rewrite them wholly before reading, so
             per-chunk storage matches the serial semantics. *)
          let decls =
            match Map.find !current_grid_private i with
            | None | Some [] -> decls
            | Some tns ->
                let zero_init tn =
                  match !current_traced_store with
                  | Some ts ->
                      Hashtbl.find ts tn
                      |> Option.value_map ~default:false ~f:(fun node ->
                          node.Low_level.zero_initialized_by_code)
                  | None -> false
                in
                List.fold tns ~init:decls ~f:(fun acc tn ->
                    acc ^^ hardline ^^ local_array_decl ~zero_init:(zero_init tn) tn)
          in
          let inner =
            string (Printf.sprintf "for (%s%s = %s; %s < %s; ++%s)" it ident lo ident hi ident)
            ^^ space ^^ lbrace
            ^^ nest 2 (hardline ^^ body_doc ())
            ^^ hardline ^^ rbrace
          in
          let comment =
            string
              (Printf.sprintf "/* Pool-backed Grid rendering: %d chunks of up to %d. */" nchunks
                 grain)
          in
          match B.parallel_grid_syntax with
          | `Dispatch ->
              comment ^^ hardline
              ^^ string
                   (Printf.sprintf "dispatch_apply((size_t)%d, DISPATCH_APPLY_AUTO, ^(size_t %s) {"
                      nchunks chunk)
              ^^ nest 2 (hardline ^^ decls ^^ hardline ^^ inner)
              ^^ hardline ^^ string "});"
          | `Openmp ->
              comment ^^ hardline
              ^^ string "#pragma omp parallel for schedule(static)"
              ^^ hardline
              ^^ string
                   (Printf.sprintf "for (%s%s = 0; %s < %d; ++%s)" it chunk chunk nchunks chunk)
              ^^ space ^^ lbrace
              ^^ nest 2 (hardline ^^ decls ^^ hardline ^^ inner)
              ^^ hardline ^^ rbrace
          | `None -> assert false
        in
        (* --- Shared analysis for the explicit-SIMD ([Vectorized]) and warp-shuffle
           ([Workgroup_reduce]) renderings below. --- *)
        let mentions_comp = Indexing.axis_index_mentions_symbol i in
        let nonempty_stmts body =
          List.filter (Low_level.flat_lines [ body ]) ~f:(function
            | Low_level.Noop | Comment _ -> false
            | _ -> true)
        in
        (* --- The accumulator-width decision of this level (gh-ocannl-754), shared by the
           warp-shuffle ([Workgroup_reduce]), SIMD-grid ([Vectorized]) and localizing-serial
           renderings below. ONE analysis: {!Low_level.peel_accum_nest} over the level's body, then
           the declines the serial rendering applies to the base it reaches. A rendering that holds
           the accumulator at the residency across the level may do so exactly where this says the
           serial rendering localizes ([pinned = None]), and must keep the storage cell's width
           where it says the serial rendering declines. The previous arrangement — a
           single-statement recognizer for the shuffle and the grid, the peel for the serial form,
           and one shared predicate patching the one disagreement that had been noticed — kept the
           two agreeing by maintenance; deriving the shuffle's and the grid's answer from the same
           call makes the agreement structural. *)
        (* A hardware-annotated reduction loop this backend serializes (no hardware index for its
           slot) is a serial level like any other — without this, retyping an INNER reduction axis
           to [Workgroup_reduce] on cc would stop the peel and narrow the accumulator once per
           remaining outer iteration. *)
        let serialized_hardware index = function
          | Low_level.Workgroup_reduce -> (
              match
                List.find !current_hardware_axes ~f:(fun a ->
                    Indexing.equal_symbol a.Low_level.ha_index index)
              with
              | Some a -> Option.is_none (B.hardware_index ~kind:`Workgroup ~slot:a.ha_slot)
              | None -> false)
          | _ -> false
        in
        (* The peel census (gh-ocannl-733) records what this site DECIDED, not merely what it
           rendered. Only accumulating levels are censused: elsewhere localization was never a
           question, and the entries would drown the reductions. *)
        let censusing = !peel_census_enabled && Low_level.has_accumulating_cell body in
        let record site =
          if censusing then peel_census := (!current_kernel_name, site) :: !peel_census
        in
        (* The localized rendering re-renders the peeled levels INSIDE the scope it just minted, and
           those re-visits refuse (the base is a [Set_local] by then) — censusing them would report
           refusals for levels that localized. Collection is suspended for the recursive render, and
           nothing genuine hides behind that: the peel descends single-statement levels down to the
           accumulation base, so the scope body holds no other site. *)
        let without_census f =
          let saved = !peel_census_enabled in
          peel_census_enabled := false;
          Exn.protect ~f ~finally:(fun () -> peel_census_enabled := saved)
        in
        (* [?body]: the warp-shuffle rendering asks about the body with its synthetic launch guard
           stripped, which is vacuous with respect to the level's own iteration space. *)
        let decide_accum_width ?(body = body) () : accum_width =
          let report = ref None in
          match
            Low_level.peel_accum_nest ~extra_level:serialized_hardware
              ~report:(fun r -> report := Some r)
              ~loop_bounds:!current_loop_bounds ~free_of:[ i ] body
          with
          | None ->
              Accum_not_a_nest
                (match !report with
                | Some { Low_level.refusal = Some refusal; _ } -> refusal
                | _ ->
                    (* [peel_accum_nest] reports exactly once, and a [None] result carries a
                       refusal; this arm exists only so the census cannot invent a verdict. *)
                    Low_level.Refused_not_a_nest)
          | Some (tn, idcs, base, debug, rebuild) ->
              let verdict =
                match !report with
                | Some { Low_level.levels; guards; refusal = _ } ->
                    (* [+ 1]: the peel is asked of this level's BODY, so a scope spans one more
                       level than it reports — the one being rendered here. *)
                    { levels = levels + 1; guards }
                | None -> { levels = 1; guards = [] }
              in
              (* The declines, in the order the serial rendering has always applied them. Under
                 [debug_log_from_routines] the per-step [Set] form is kept — a [Local_scope] body
                 renders with [log_set_locals:false], so a rewrite would silence the per-iteration
                 trace (the SIMD and shuffle renderings refuse under logging for the same reason). A
                 dead level ([to_ < from_]) performs no accesses; see [peel_accum_nest]'s refusal,
                 which covers the levels BELOW this one — this is the same refusal for the level
                 being rendered, whose bounds the peel never sees. And the storage-pinned base
                 ([accum_pinned_to_storage_prec]): an update mentioning an RNG conversion renders at
                 the storage precision, so localizing it would change the draw, not merely move it —
                 the serial rendering accumulates it in the narrow cell, narrowing every iteration,
                 and every other rendering must do the same or change the width (gh-ocannl-682). *)
              let pinned =
                if Utils.debug_log_from_routines () then Some Skip_debug_logging
                else if to_ < from_ then Some Skip_dead_level
                else
                  match base with
                  | `Update llsc when accum_pinned_to_storage_prec llsc -> Some Skip_accum_pinned
                  | `Update _ | `Scope _ -> None
              in
              Accum_base { tn; idcs; base; debug; rebuild; verdict; pinned }
        in
        (* The single accumulation statement at THIS level, as the warp-shuffle and SIMD renderings
           need it: the peel reached a raw update with no level or guard in between. *)
        let statement_accumulation (decision : accum_width) : statement_accum option =
          match decision with
          | Accum_base
              {
                tn;
                idcs;
                base = `Update llsc;
                verdict = { levels = 1; guards = [] } as verdict;
                pinned;
                _;
              } ->
              Option.map (Low_level.accum_update_parts ~tn ~idcs llsc) ~f:(fun (op, contrib) ->
                  {
                    sa_tn = tn;
                    sa_idcs = idcs;
                    sa_op = op;
                    sa_contrib = contrib;
                    sa_verdict = verdict;
                    sa_pinned = pinned;
                  })
          | Accum_base _ | Accum_not_a_nest _ -> None
        in
        (* Whether a rendering that holds a statement's accumulator at the residency may take it
           even though the decision pinned it to storage: only where the residency IS the storage
           width, so there is no width to change — at f32/f64 storage, and on every backend that
           does not widen this precision, an RNG-bearing reduction shuffles or vectorizes exactly as
           it did before gh-ocannl-682. *)
        let residency_is_storage tn =
          let store_prec = Lazy.force tn.Tn.storage_prec in
          Ops.equal_prec (acc_prec store_prec) store_prec
        in
        (* What a rendering that took the statement records: the width DECISION, not the form. *)
        let width_site (sa : statement_accum) =
          match sa.sa_pinned with
          | None -> Peel_ceded sa.sa_verdict
          | Some skip -> Peel_not_attempted skip
        in
        (* Eligibility bail-out of the explicit-SIMD renderings ([try_vectorize] /
           [try_vectorize_reduce]) back to the pragma/serial fallbacks. *)
        let exception Bail in
        let rec scalar_mentions (llsc : Low_level.scalar_t) =
          match llsc with
          | Low_level.Get (_, idcs) | Get_merge_buffer (_, idcs) ->
              Array.exists idcs ~f:mentions_comp
          | Get_dynamic { idcs; dyn_value = v, _; _ } ->
              Array.exists idcs ~f:mentions_comp || scalar_mentions v
          (* Scope-local bodies could bind or mention the index in statement position;
             conservatively ineligible. *)
          | Local_scope _ | Get_local _ -> raise Bail
          | Embed_index idx -> mentions_comp idx
          | Ternop (_, (a, _), (b, _), (c, _)) ->
              scalar_mentions a || scalar_mentions b || scalar_mentions c
          | Binop (_, (a, _), (b, _)) -> scalar_mentions a || scalar_mentions b
          | Unop (_, (a, _)) -> scalar_mentions a
          | Constant _ | Constant_bits _ -> false
        in
        let contiguous idcs =
          let n = Array.length idcs in
          n > 0
          && Array.for_alli idcs ~f:(fun p idx -> p = n - 1 || not (mentions_comp idx))
          &&
          match idcs.(n - 1) with
          | Indexing.Iterator s -> Indexing.equal_symbol s i
          | Indexing.Affine { symbols; _ } ->
              List.for_all symbols ~f:(fun (c, s) -> (not (Indexing.equal_symbol s i)) || c = 1)
              && List.count symbols ~f:(fun (_, s) -> Indexing.equal_symbol s i) = 1
          | _ -> false
        in
        let check_read ~written tn idcs =
          match Hashtbl.find written tn.Tn.uid with
          | Some w_idcs ->
              if not (Array.equal Indexing.equal_axis_index w_idcs idcs) then raise Bail
          | None -> ()
        in
        let rec no_written_reads ~written (llsc : Low_level.scalar_t) =
          match llsc with
          | Low_level.Get (tn, _) | Get_merge_buffer (tn, _) | Get_dynamic { tn; _ } ->
              if Hashtbl.mem written tn.Tn.uid then raise Bail
          | Local_scope _ | Get_local _ -> raise Bail
          | Embed_index _ | Constant _ | Constant_bits _ -> ()
          | Ternop (_, (a, _), (b, _), (c, _)) ->
              no_written_reads ~written a;
              no_written_reads ~written b;
              no_written_reads ~written c
          | Binop (_, (a, _), (b, _)) ->
              no_written_reads ~written a;
              no_written_reads ~written b
          | Unop (_, (a, _)) -> no_written_reads ~written a
        in
        let uniform_scalar ~written prec llsc =
          (* Uniform across lanes: a read of a stored node cannot equal its (index-mentioning) store
             vector, so reject those; then render as a plain scalar (vector-scalar arithmetic
             splats; in the packed style the scalar participates per lane). *)
          no_written_reads ~written llsc;
          let local_defs, sdoc = pp_scalar prec llsc in
          if not (List.is_empty local_defs) then raise Bail;
          parens sdoc
        in
        (* The [`Vec_extensions] expression renderer shared by the elementwise ([try_vectorize]) and
           reduction ([try_vectorize_reduce]) renderings. Emitted binding statements accumulate
           through [emit]; [written] maps written nodes to their store index vectors — every read of
           a written node must use that exact vector (vector semantics evaluates all lanes' loads
           before the store, so cross-lane flow would reorder against the serial loop). *)
        let vec_ext_machinery ~prec ~lanes ~vtyp ~written ~emit ~fresh ~need_typedef =
          let vload tn idcs =
            (* A swizzled layout breaks row-major contiguity within a row. *)
            if is_swizzled tn || is_pipelined tn then raise Bail;
            if not (contiguous idcs) then raise Bail;
            check_read ~written tn idcs;
            let name = fresh "vget" in
            let offset = pp_array_offset (idcs, Lazy.force tn.Tn.dims) in
            let store_prec = Lazy.force tn.Tn.storage_prec in
            let load, _store = vec_bridge ~store_prec ~prec ~lanes ~vtyp ~need_typedef ~fresh in
            emit (load ~dst:name ~mem:(string (get_ident tn) ^^ brackets offset));
            string name
          in
          let rec vec_expr (llsc : Low_level.scalar_t) (p : Ops.prec) : PPrint.document =
            if not (scalar_mentions llsc) then uniform_scalar ~written prec llsc
            else if not (Ops.equal_prec (comp_prec p) prec) then raise Bail
            else
              match llsc with
              | Low_level.Get (tn, idcs) ->
                  (* Narrow storage is admissible: [vload] widens it into the compute vector. What
                     is not is a node whose arithmetic would run at another width. *)
                  if not (Ops.equal_prec (comp_prec (Lazy.force tn.Tn.storage_prec)) prec) then
                    raise Bail;
                  vload tn idcs
              | Binop (op, (a, pa), (b, pb)) ->
                  let inf =
                    match op with
                    | Ops.Add -> " + "
                    | Sub -> " - "
                    | Mul -> " * "
                    | Div -> " / "
                    | _ -> raise Bail
                  in
                  parens (vec_expr a pa ^^ string inf ^^ vec_expr b pb)
              | Ternop (Ops.FMA, (a, pa), (b, pb), (c, pc)) ->
                  (* Fused, matching the scalar path's [fmaf]/[fma] single rounding (the simplifier
                     synthesizes [FMA] from mul-add trees, so this is the hot case). A plain [a * b
                     + c] would be only maybe-contracted, so vector lanes could differ from the
                     serial remainder loop and twin. Operands bind to vector temps (lane-uniform
                     ones splat explicitly: vector = scalar init is invalid). *)
                  let bind llsc p =
                    let name = fresh "vfop" in
                    emit (string (vtyp ^ " " ^ name ^ " = ") ^^ vec_operand llsc p ^^ semi);
                    name
                  in
                  let na = bind a pa and nb = bind b pb and nc = bind c pc in
                  let nr = fresh "vfma" in
                  emit (string (vtyp ^ " " ^ nr ^ " = ") ^^ string nc ^^ semi);
                  emit (vec_acc_fma ~prec ~lanes ~dst:nr ~a:na ~b:nb);
                  string nr
              | Unop (Ops.Identity, (a, pa)) -> vec_expr a pa
              | Unop (Ops.Neg, (a, pa)) -> parens (string "-" ^^ vec_expr a pa)
              | _ -> raise Bail
          and vec_operand (llsc : Low_level.scalar_t) (p : Ops.prec) : PPrint.document =
            (* A vector-typed rendering even for lane-uniform values: initializers and builtin
               arguments need a vector, where the implicit vector-scalar splat of binary operators
               does not apply. *)
            if scalar_mentions llsc then vec_expr llsc p
            else
              (* Bound to a scalar temp first: [vec_splat] repeats its argument per lane, and this
                 one can be a memory load. Emitted here, hence before the vector declaration the
                 caller wraps around this expression. *)
              let sname = fresh "vunif" in
              emit
                (string (B.typ_of_prec prec ^ " " ^ sname ^ " = ")
                ^^ uniform_scalar ~written prec llsc ^^ semi);
              vec_splat ~vtyp ~lanes sname
          in
          (vec_expr, vec_operand)
        in
        (* Explicit SIMD rendering of a [Vectorized] loop via GCC/Clang vector extensions (portable
           across gcc/clang and AVX2/NEON; the [Vectorized]-codegen follow-up of gh-ocannl-164). The
           loop must start at 0 and its body must be a sequence of plain [Set] statements over one
           floating precision, with every access that mentions the loop index contiguous in it (the
           index appears only in the last component, with coefficient 1 — the flat offset then
           advances by exactly 1 per iteration). Index-free subexpressions render as scalars
           (vector-scalar arithmetic splats across lanes); vector subexpressions allow
           [Add]/[Sub]/[Mul]/[Div]/[Neg] and fused [FMA] (matching the scalar path's [fmaf]/[fma]
           rounding; see the note in [vec_expr]). At most one store per node, and every read of a
           stored node must use the store's exact index vector — vector semantics evaluates all
           lanes' loads before the store, so cross-lane flow would reorder against the serial loop.
           The main loop advances by [lanes]; a serial remainder loop reuses [body_doc]. Anything
           else falls back to [vectorize_pragma] / serial. *)
        let try_vectorize () : PPrint.document option =
          try
            if B.vector_bytes < 8 || from_ <> 0 || Utils.debug_log_from_routines () then raise Bail;
            let extent = to_ + 1 in
            let stmts = nonempty_stmts body in
            let sets =
              List.map stmts ~f:(function
                | Low_level.Set { tn; idcs; llsc; _ } -> (tn, idcs, llsc)
                | _ -> raise Bail)
            in
            if List.is_empty sets then raise Bail;
            (* The lane geometry is keyed off the *compute* precision (gh-ocannl-517): narrow
               storage pairs a half-width memory vector with a full-width f32 register, and the
               conversion rides the load and the store. *)
            let prec =
              let tn, _, _ = List.hd_exn sets in
              comp_prec (Lazy.force tn.Tn.storage_prec)
            in
            if not (B.vector_prec_ok prec) then raise Bail;
            let lanes = match vec_lanes_for ~prec ~extent with Some l -> l | None -> raise Bail in
            let written = Hashtbl.create (module Int) in
            List.iter sets ~f:(fun (tn, idcs, _) ->
                if not (Ops.equal_prec (comp_prec (Lazy.force tn.Tn.storage_prec)) prec) then
                  raise Bail;
                match Hashtbl.add written ~key:tn.Tn.uid ~data:idcs with
                | `Ok -> ()
                | `Duplicate -> raise Bail);
            let stmts_docs = ref [] in
            let emit d = stmts_docs := d :: !stmts_docs in
            let extra_typedefs = Hashtbl.create (module String) in
            let need_typedef name doc = Hashtbl.set extra_typedefs ~key:name ~data:doc in
            let fresh =
              let ctr = ref 0 in
              fun pfx ->
                Int.incr ctr;
                Printf.sprintf "%s%d__" pfx !ctr
            in
            let prelude =
              match B.vector_style with
              | `Vec_extensions ->
                  let vtyp, typedef_doc = vec_ext_typ ~prec ~lanes in
                  let _vec_expr, vec_operand =
                    vec_ext_machinery ~prec ~lanes ~vtyp ~written ~emit ~fresh ~need_typedef
                  in
                  List.iter sets ~f:(fun (tn, idcs, llsc) ->
                      if is_swizzled tn || is_pipelined tn then raise Bail;
                      if not (contiguous idcs) then raise Bail;
                      let rhs = vec_operand llsc prec in
                      let vname = fresh "vset" in
                      emit (string (vtyp ^ " " ^ vname ^ " = ") ^^ rhs ^^ semi);
                      let store_prec = Lazy.force tn.Tn.storage_prec in
                      let _load, store =
                        vec_bridge ~store_prec ~prec ~lanes ~vtyp ~need_typedef ~fresh
                      in
                      emit
                        (store ~src:vname
                           ~mem:
                             (string (get_ident tn)
                             ^^ brackets (pp_array_offset (idcs, Lazy.force tn.Tn.dims)))));
                  separate hardline (typedef_doc :: registered_typedefs extra_typedefs) ^^ hardline
              | `Packed_struct ->
                  (* GPU 128-bit packed loads/stores (gh-ocannl-463; llm.c's Packed128): the
                     backend's aligned pack aggregate is loaded/stored via [reinterpret_cast] — one
                     128-bit memory transaction — while the arithmetic stays scalar in a per-lane
                     loop over the pack's [.v] payload (per-lane [fmaf]/[fma] keeps the serial
                     path's rounding). Sound only at provably lane-aligned offsets of
                     device-resident buffers, hence the extra eligibility checks. *)
                  let vtyp =
                    match B.vec_typ_of_prec ~length:lanes prec with
                    | s -> s
                    | exception _ -> raise Bail
                  in
                  (* The flat offset must stay a lane multiple whenever the loop index is one:
                     components before the last contribute stride multiples of [dims.(n - 1)], so
                     the last dimension must be a lane multiple (unless the access is 1-D), and the
                     last component's constant offset and non-index coefficients must be lane
                     multiples. Buffer bases and pool offsets are [Ops.buffer_alignment >= 16]
                     aligned, so lane-multiple element offsets are 16-byte-aligned addresses. *)
                  let lane_aligned tn idcs =
                    let dims = Lazy.force tn.Tn.dims in
                    let n = Array.length idcs in
                    n > 0
                    && (n = 1 || dims.(n - 1) % lanes = 0)
                    &&
                    match idcs.(n - 1) with
                    | Indexing.Iterator _ -> true
                    | Indexing.Affine { symbols; offset } ->
                        offset % lanes = 0
                        && List.for_all symbols ~f:(fun (c, s) ->
                            Indexing.equal_symbol s i || c % lanes = 0)
                    | Indexing.Fixed_idx _ | Indexing.Sub_axis | Indexing.Concat _ -> false
                  in
                  let eligible tn idcs =
                    contiguous idcs && lane_aligned tn idcs
                    && Tn.Placements.is_materialized_force (placements ()) tn 463
                    (* Stack and workgroup-shared arrays are only element-aligned. *)
                    && (not (Set.mem !current_workgroup_shared tn))
                    && (not (is_swizzled tn))
                    && not (is_pipelined tn)
                  in
                  let vload tn idcs =
                    if not (eligible tn idcs) then raise Bail;
                    check_read ~written tn idcs;
                    let name = fresh "vget" in
                    emit
                      (string
                         (Printf.sprintf "const %s %s = *reinterpret_cast<%sconst %s*>(&" vtyp name
                            B.buffer_prefix vtyp)
                      ^^ string (get_ident tn)
                      ^^ brackets (pp_array_offset (idcs, Lazy.force tn.Tn.dims))
                      ^^ string ");");
                    name
                  in
                  let lane_var = "ocannl_l__" in
                  let rec lane_expr (llsc : Low_level.scalar_t) (p : Ops.prec) : PPrint.document =
                    if not (scalar_mentions llsc) then uniform_scalar ~written prec llsc
                    else if not (Ops.equal_prec p prec) then raise Bail
                    else
                      match llsc with
                      | Low_level.Get (tn, idcs) ->
                          if not (Ops.equal_prec (Lazy.force tn.Tn.storage_prec) prec) then
                            raise Bail;
                          string (vload tn idcs ^ ".v[" ^ lane_var ^ "]")
                      | Binop (((Ops.Add | Ops.Sub | Ops.Mul | Ops.Div) as op), (a, pa), (b, pb)) ->
                          B.binop_syntax prec op (lane_expr a pa) (lane_expr b pb)
                      | Ternop (Ops.FMA, (a, pa), (b, pb), (c, pc)) ->
                          B.ternop_syntax prec Ops.FMA (lane_expr a pa) (lane_expr b pb)
                            (lane_expr c pc)
                      | Unop (Ops.Identity, (a, pa)) -> lane_expr a pa
                      | Unop (Ops.Neg, (a, pa)) -> parens (string "-" ^^ lane_expr a pa)
                      | _ -> raise Bail
                  in
                  List.iter sets ~f:(fun (tn, idcs, llsc) ->
                      if not (eligible tn idcs) then raise Bail;
                      let rhs = lane_expr llsc prec in
                      let vname = fresh "vset" in
                      emit (string (vtyp ^ " " ^ vname ^ ";"));
                      emit
                        (string
                           (Printf.sprintf "for (int %s = 0; %s < %d; ++%s) { %s.v[%s] = " lane_var
                              lane_var lanes lane_var vname lane_var)
                        ^^ rhs ^^ string "; }");
                      emit
                        (string (Printf.sprintf "*reinterpret_cast<%s%s*>(&" B.buffer_prefix vtyp)
                        ^^ string (get_ident tn)
                        ^^ brackets (pp_array_offset (idcs, Lazy.force tn.Tn.dims))
                        ^^ string (") = " ^ vname ^ ";")));
                  empty
            in
            let ivar = symbol_ident i in
            let it = B.loop_index_type in
            let body_vec = separate hardline (List.rev !stmts_docs) in
            Some
              (string
                 (Printf.sprintf "{ /* Vectorized rendering: %d lanes of %s. */" lanes
                    (B.typ_of_prec prec))
              ^^ nest 2
                   (hardline ^^ prelude
                   ^^ string (Printf.sprintf "%s%s = 0;" it ivar)
                   ^^ hardline
                   ^^ string
                        (Printf.sprintf "for (; %s + %d <= %d; %s += %d) {" ivar lanes extent ivar
                           lanes)
                   ^^ nest 2 (hardline ^^ body_vec)
                   ^^ hardline ^^ string "}" ^^ hardline
                   ^^ string (Printf.sprintf "for (; %s <= %d; ++%s) {" ivar to_ ivar)
                   ^^ nest 2 (hardline ^^ body_doc ())
                   ^^ hardline ^^ string "}")
              ^^ hardline ^^ string "}")
          with Bail -> None
        in
        (* SIMD reduction rendering of a [Vectorized] accumulation loop (gh-ocannl-468; ggml's
           [ggml_vec_dot_f32] pattern, ggml/src/ggml-cpu/vec.h). A recognized accumulation body
           [acc[idcs] = op(acc[idcs], contrib(i))] renders as a 1×[chains] grid of independent
           vector accumulator registers — splitting the loop-carried dependency into [chains *
           lanes] independent chains is exactly the strict-FP reassociation the [Vectorized] retype
           licenses — updated in a fused main loop advancing by [chains * lanes], then folded
           register-wise, lane-wise, and finally into the accumulator cell; a serial tail loop
           reuses the scalar body. [chains] defaults to 4 (ggml's [GGML_F32_ARR]: enough independent
           chains to cover the FMA latency-throughput gap), halved until the first-block
           initialization fits the extent. Initializing the chains from the first [chains] blocks of
           contributions avoids needing an identity constant, so [Max]/[Min] reductions work
           unchanged. [`Vec_extensions] only: on GPU backends reductions parallelize via
           [Workgroup_reduce] warp shuffles instead, and a bailed-out accumulation falls back to a
           plain serial loop (never to [vectorize_pragma], which would assert iteration
           independence). *)
        let try_vectorize_reduce () : PPrint.document option =
          try
            (match B.vector_style with `Vec_extensions -> () | `Packed_struct -> raise Bail);
            if B.vector_bytes < 8 || from_ <> 0 || Utils.debug_log_from_routines () then raise Bail;
            (* Two accumulator targets: a memory cell (the ordinary form), or the scope LOCAL of a
               widened reduction (gh-ocannl-639: the loop sits inside the accumulator's
               [Local_scope], its update is a [Set_local] — recognizing it here is what lets a
               vectorized inner reduction axis keep the whole nest's compute-precision residency,
               its chains folding into the local with no storage round-trip at all). *)
            let acc_target, op, contrib, decided =
              match statement_accumulation (decide_accum_width ()) with
              | Some sa ->
                  (* gh-ocannl-754: the chains hold the accumulator across the level, so the grid
                     may take the statement only where the shared decision lets the serial rendering
                     localize it — or where the residency is the storage width and the decline
                     changes nothing. A pinned body at a wider residency (an RNG-bearing
                     contribution on a widening backend) bails to the serial fallback, which keeps
                     the per-step narrowing the pin asks for. *)
                  (match sa.sa_pinned with
                  | None -> ()
                  | Some Skip_accum_pinned -> if not (residency_is_storage sa.sa_tn) then raise Bail
                  | Some (Skip_debug_logging | Skip_dead_level) -> raise Bail);
                  (`Cell (sa.sa_tn, sa.sa_idcs), sa.sa_op, sa.sa_contrib, Some sa)
              | None -> (
                  match nonempty_stmts body with
                  | [ Low_level.Set_local (id, llsc) ] -> (
                      match Low_level.accum_local_update_parts ~id llsc with
                      | Some (op, contrib) -> (`Local id, op, contrib, None)
                      | None -> raise Bail)
                  | _ -> raise Bail)
            in
            let prec =
              match acc_target with
              | `Cell (tn, _) ->
                  let store_prec = Lazy.force tn.Tn.storage_prec in
                  let p = comp_prec store_prec in
                  (* The chains and the direct-cell fold hold the accumulator at [p], so the
                     direct-cell form honors the accumulator-width contract only where [p] IS the
                     residency. Under [Fp16_wide] with [narrow_compute_f32 = false] on a native-fp16
                     target, [acc_prec] resolves an f16 cell to f32 while [comp_prec] stays half —
                     half register chains would round narrowly while the serial schedule localizes
                     at f32 (Codex P1 round 2 on staging PR #477). Bail: the dispatch then reaches
                     [try_localize_serial_reduce], whose scope carries [acc_prec]. The [Vectorized]
                     level rides into the scope and this rendering is attempted again on the
                     [`Local] target at the residency, but [vec_expr]'s compute-width gates decline
                     the contribution there too (its operands' own compute width is half), so this
                     policy corner renders the localized SERIAL form — width-correct, SIMD
                     forfeited; [accum_width.ml]'s vec leg pins that outcome. *)
                  if not (Ops.equal_prec (acc_prec store_prec) p) then raise Bail;
                  p
              | `Local id -> scope_prec_of id
            in
            if not (B.vector_prec_ok prec) then raise Bail;
            (* A loop-invariant contribution deserves strength reduction, not chains. *)
            if not (scalar_mentions contrib) then raise Bail;
            let extent = to_ + 1 in
            (* Not [vec_lanes_for]: this rendering pays a horizontal fold as long as the lane count,
               so the width is ranked against the update-plus-epilogue cost — see
               {!Backend_intf.simd_reduce_lanes_for}. *)
            let lanes =
              match
                simd_reduce_lanes_for ~vector_bytes:B.vector_bytes
                  ~elt_bytes:(Ops.prec_in_bytes prec) ~extent
              with
              | Some l -> l
              | None -> raise Bail
            in
            let chains = simd_reduce_chains ~lanes ~extent in
            let step = chains * lanes in
            let vtyp, typedef_doc = vec_ext_typ ~prec ~lanes in
            let extra_typedefs = Hashtbl.create (module String) in
            let need_typedef name doc = Hashtbl.set extra_typedefs ~key:name ~data:doc in
            (* The accumulator is not vector-loaded ([contrib] cannot touch it, per the recognizer),
               so nothing is [written] from the vector expressions' viewpoint. *)
            let written = Hashtbl.create (module Int) in
            let stmts_docs = ref [] in
            let emit d = stmts_docs := d :: !stmts_docs in
            let take () =
              let docs = List.rev !stmts_docs in
              stmts_docs := [];
              docs
            in
            let fresh =
              let ctr = ref 0 in
              fun pfx ->
                Int.incr ctr;
                Printf.sprintf "%s%d__" pfx !ctr
            in
            let _vec_expr, vec_operand =
              vec_ext_machinery ~prec ~lanes ~vtyp ~written ~emit ~fresh ~need_typedef
            in
            (* Chain [c] consumes the flat-offset window shifted by [c * lanes]: under the
               contiguity rule the shift is a constant added to each access's last (loop-index)
               component. *)
            let shift_idx ~by (idx : Indexing.axis_index) =
              match idx with
              | Indexing.Iterator s when Indexing.equal_symbol s i ->
                  Indexing.Affine { symbols = [ (1, s) ]; offset = by }
              | Indexing.Affine { symbols; offset }
                when List.exists symbols ~f:(fun (_, s) -> Indexing.equal_symbol s i) ->
                  Indexing.Affine { symbols; offset = offset + by }
              | _ -> idx
            in
            let rec shift ~by (llsc : Low_level.scalar_t) : Low_level.scalar_t =
              if by = 0 then llsc
              else
                match llsc with
                | Low_level.Get (tn, idcs) -> Low_level.Get (tn, Array.map idcs ~f:(shift_idx ~by))
                | Binop (op, (a, pa), (b, pb)) -> Binop (op, (shift ~by a, pa), (shift ~by b, pb))
                | Ternop (op, (a, pa), (b, pb), (c, pc)) ->
                    Ternop (op, (shift ~by a, pa), (shift ~by b, pb), (shift ~by c, pc))
                | Unop (op, (a, pa)) -> Unop (op, (shift ~by a, pa))
                (* Any other index-mentioning form bails inside [vec_ext_machinery]. *)
                | _ -> llsc
            in
            let ivar = symbol_ident i in
            let grid = vec_acc_grid ~prefix:("vred_" ^ ivar) ~rows:1 ~cols:chains in
            let acc_regs = grid.(0) in
            (* Chain initialization from the first [chains] blocks, read at [i = 0]. *)
            Array.iteri acc_regs ~f:(fun c name ->
                let rhs = vec_operand (shift ~by:(c * lanes) contrib) prec in
                emit (string (vtyp ^ " " ^ name ^ " = ") ^^ rhs ^^ semi));
            let init_docs = take () in
            (* The fused main-loop body: one independent update per chain. *)
            Array.iteri acc_regs ~f:(fun c name ->
                match (op, contrib) with
                | Ops.Add, Low_level.Binop (Ops.Mul, (a, pa), (b, pb)) ->
                    (* The dot-product case (also reached from the recognizer's FMA form):
                       fused-multiply-accumulate straight into the chain register. *)
                    let bind llsc p =
                      let nm = fresh "vfop" in
                      emit
                        (string (vtyp ^ " " ^ nm ^ " = ")
                        ^^ vec_operand (shift ~by:(c * lanes) llsc) p
                        ^^ semi);
                      nm
                    in
                    let na = bind a pa in
                    let nb = bind b pb in
                    emit (vec_acc_fma ~prec ~lanes ~dst:name ~a:na ~b:nb)
                | _ ->
                    let nm = fresh "vsrc" in
                    emit
                      (string (vtyp ^ " " ^ nm ^ " = ")
                      ^^ vec_operand (shift ~by:(c * lanes) contrib) prec
                      ^^ semi);
                    emit (vec_acc_combine ~prec ~lanes ~op ~dst:name ~src:nm));
            let update_docs = take () in
            let total = "vred_total_" ^ ivar ^ "__" in
            (* The accumulator cell itself stays at its storage precision: the fold reads it widened
               and narrows the combined value once, exactly as the scalar path's [Set] does
               (gh-ocannl-517); a scope-local target is already at [prec] and takes the combined
               value with no conversion. The scalar REMAINDER (a non-multiple extent's tail) folds
               into the compute-precision [total] BEFORE that single store (gh-ocannl-639): the
               original per-step body would narrow the vector partial mid-way and then narrow again
               per tail step, splitting this rendering from the widened serial baseline on exactly
               the non-dividing extents. *)
            let folds =
              vec_acc_grid_fold ~prec ~lanes ~op grid
              @ vec_acc_lane_fold ~prec ~lanes ~op ~vname:acc_regs.(0) ~out:total
            in
            let tail_defs, tail_update =
              match (op, contrib) with
              | Ops.Add, Low_level.Binop (Ops.Mul, (a, _), (b, _)) ->
                  (* Mirror the widened serial baseline's fused update ([pp_scalar]'s homogeneous
                     FMA), or the tail's mul-then-add would round differently from the serial
                     candidate's fmaf on the same steps. *)
                  let da, ea = pp_scalar prec a in
                  let db, eb = pp_scalar prec b in
                  ( da @ db,
                    string total ^^ string " = "
                    ^^ B.ternop_syntax prec Ops.FMA ea eb (string total)
                    ^^ semi )
              | _ ->
                  let dc, ec = pp_scalar prec contrib in
                  ( dc,
                    string total ^^ string " = " ^^ B.binop_syntax prec op (string total) ec ^^ semi
                  )
            in
            let tail_body = pp_local_defs tail_defs ^^ tail_update in
            let store =
              match acc_target with
              | `Cell (tn, idcs) ->
                  let store_prec = Lazy.force tn.Tn.storage_prec in
                  let widen = B.convert_precision ~from:store_prec ~to_:prec in
                  let narrow = B.convert_precision ~from:prec ~to_:store_prec in
                  let cell () =
                    string (get_ident tn)
                    ^^ brackets (pp_array_offset (idcs, Lazy.force tn.Tn.dims))
                  in
                  cell () ^^ string " = "
                  ^^ wrap_conversion narrow
                       (B.binop_syntax prec op (wrap_conversion widen (cell ())) (string total))
                  ^^ semi
              | `Local id ->
                  pp_scope_id id ^^ string " = "
                  ^^ B.binop_syntax prec op (pp_scope_id id) (string total)
                  ^^ semi
            in
            let it = B.loop_index_type in
            (* Past every bail-out: the grid renders, and the census says which decision it took its
               width from. A scope-local target sits inside a scope that was censused when it was
               minted, so it records nothing here. *)
            Option.iter decided ~f:(fun sa -> record (width_site sa));
            Some
              (string
                 (Printf.sprintf
                    "{ /* Vectorized reduction rendering: %d chain(s) of %d lanes of %s. */" chains
                    lanes (B.typ_of_prec prec))
              ^^ nest 2
                   (hardline
                   ^^ separate hardline (typedef_doc :: registered_typedefs extra_typedefs)
                   ^^ hardline
                   ^^ string (Printf.sprintf "%s%s = 0;" it ivar)
                   ^^ hardline ^^ separate hardline init_docs ^^ hardline
                   ^^ string
                        (Printf.sprintf "for (%s = %d; %s + %d <= %d; %s += %d) {" ivar step ivar
                           step extent ivar step)
                   ^^ nest 2 (hardline ^^ separate hardline update_docs)
                   ^^ hardline ^^ string "}" ^^ hardline ^^ separate hardline folds ^^ hardline
                   ^^ string (Printf.sprintf "for (; %s <= %d; ++%s) {" ivar to_ ivar)
                   ^^ nest 2 (hardline ^^ tail_body)
                   ^^ hardline ^^ string "}" ^^ hardline ^^ store)
              ^^ hardline ^^ string "}")
          with Bail -> None
        in
        (* Warp-shuffle rendering of a [Workgroup_reduce] accumulation loop (gh-ocannl-462; llm.c's
           [warpReduceSum] / [blockReduce] idiom, llmc/cuda_utils.cuh). Recognizes a body that is a
           single accumulation statement [acc[idcs] = op(acc[idcs], contrib)] (or its FMA form [acc
           = FMA(a, b, acc)]) where [idcs] does not mention the loop index and [op] is an
           associative-commutative reduction — such a body IS the loop's serial meaning, so backends
           without shuffle support ([warp_size = 0]) render it with the ordinary fallbacks. With
           shuffle support the loop renders as: every thread computes its contribution, a log2(warp)
           [ocannl_shfl_xor] tree reduces within each warp, then (for multi-warp extents) lane 0 of
           each warp stages one value in a workgroup-shared slot, a barrier, and the first warp
           shuffle-reduces the per-warp partials — thread 0 finally folds the total into the
           accumulator (reassociation is the annotation's license, like [Vectorized]). This halves
           the shared-memory traffic and barrier count of the explicitly staged tree, which remains
           supported: unrecognized bodies keep the [Workgroup]-style hardware binding and their own
           staging and barriers.

           The multi-warp phase needs no identity constant: [num_warps] must be a power of two, and
           XOR with offsets [< num_warps] maps lanes [< num_warps] onto themselves, so the garbage
           held by lanes [>= num_warps] never mixes into the reduced prefix.

           A recognized accumulation that cannot be rendered (extent not covering whole warps,
           reduce axis not at workgroup slot 0, ...) raises: binding the index like a plain
           [Workgroup] axis would make every thread race the read-modify-write. *)
        let try_warp_reduce () : PPrint.document option =
          if B.warp_size <= 0 then None
          else
            let stmts = nonempty_stmts body in
            (* When this loop's extent is smaller than its slot's launch dimension,
               [guard_annotated_extents] has already wrapped the body in the synthetic launch guard
               [If (i < extent)]. Strip exactly that shape — it is vacuous with respect to the
               loop's own iteration space — so a guarded accumulation is still recognized, and then
               rejected by the extent-coverage check below, instead of silently racing under the
               hardware-binding fallback (PR #119 review). *)
            let stmts =
              match stmts with
              | [
               Low_level.If
                 {
                   cond =
                     Binop (Ops.Cmplt, (Embed_index (Indexing.Iterator s), _), (Constant c, _)), _;
                   body = guarded;
                 };
              ]
                when Indexing.equal_symbol s i && Float.equal c (Float.of_int (to_ - from_ + 1)) ->
                  nonempty_stmts guarded
              | _ -> stmts
            in
            let fail msg =
              invalid_arg
                ("C_syntax.pp_ll: Workgroup_reduce loop " ^ symbol_ident i
               ^ " is a recognized accumulation, but the warp-shuffle rendering requires " ^ msg
               ^ " (a plain hardware binding would race the accumulator update)")
            in
            let decision = decide_accum_width ~body:(Low_level.unflat_lines stmts) () in
            match statement_accumulation decision with
            | None -> (
                match decision with
                | Accum_base { verdict = { guards = []; _ }; _ } ->
                    (* The shared decision found an accumulation into a cell every lane shares — a
                       nest of levels below this one, or a schedule-minted scope — with no guard
                       selecting among the lanes. The shuffle cannot render it, and binding the
                       index would have every lane read-modify-write the one cell; before
                       gh-ocannl-754 this fell through to that binding silently. *)
                    fail
                      "a single accumulation statement as its body: the accumulation found under \
                       this level (a nest of inner levels, or a schedule-minted scope) is one the \
                       shuffle cannot render, and it is unguarded, so every lane would update the \
                       same cell"
                | Accum_base _ | Accum_not_a_nest _ -> (
                    (* Not an accumulation the shuffle owns. The hardware binding is the correct
                       rendering of an explicitly staged tree (lane-selecting guards, per-lane
                       cells) and of a per-lane update — and a silent race for a body that
                       read-modify-writes a cell every lane shares: a level with a sibling
                       statement, a data-dependent guard, or an inner nest, which the peel refuses
                       as [Accum_not_a_nest] and which fell through to the binding until
                       gh-ocannl-950. The race criterion, not the nest criterion, is what tells the
                       two apart: [Low_level.racing_lane_invariant_update]. *)
                    (* Shared across the lanes: device-resident (materialized) or workgroup-shared
                       storage. A per-thread local array is one array per lane and cannot race. A
                       level of extent one binds lane 0 alone and cannot race either. *)
                    let shared tn =
                      Set.mem !current_workgroup_shared tn
                      || Tn.Placements.is_materialized_force (placements ()) tn 950
                    in
                    match
                      Low_level.racing_lane_invariant_update ~lane:i ~shared
                        (Low_level.unflat_lines stmts)
                    with
                    | None -> None
                    | Some (tn, idcs) ->
                        let cell =
                          Tn.debug_name tn ^ "["
                          ^ String.concat ~sep:", "
                              (Array.to_list idcs
                              |> List.map ~f:(fun idx ->
                                  Sexp.to_string_hum (Indexing.sexp_of_axis_index idx)))
                          ^ "]"
                        in
                        (* A typed schedule cause, not a bare [Invalid_argument]: under the
                           autotuner this is one candidate's decline, and the strict default
                           classification would make an unclassified exception fatal to the whole
                           search. A hand-written schedule still sees [Invalid_argument] at the
                           [Context.compile] boundary ([Schedule_outcome.exception_of_cause]). *)
                        raise
                          (Schedule_outcome.Cause_at
                             ( Schedule_outcome.Backend_codegen,
                               Schedule_outcome.Illegal_schedule
                                 {
                                   check = "workgroup_reduce_race";
                                   detail =
                                     "C_syntax.pp_ll: Workgroup_reduce loop " ^ symbol_ident i
                                     ^ " binds the lane index, and its body updates " ^ cell
                                     ^ " in every lane: the statement reads the cell it writes and \
                                        the cell does not depend on the lane index, so a hardware \
                                        binding would race the read-modify-write across lanes (a \
                                        guard selecting one lane does not help: every block, and \
                                        every coordinate of another bound axis, has that lane), \
                                        and the body is not a single accumulation the warp shuffle \
                                        can render (a sibling statement, a data-dependent guard, \
                                        or an inner nest). Keep the level Serial, or stage the \
                                        reduction explicitly with per-lane cells and a plain final \
                                        store (gh-ocannl-950)";
                                 } ))))
            | Some ({ sa_tn = tn; sa_idcs = idcs; sa_op = op; sa_contrib = contrib; _ } as sa) ->
                let warp = B.warp_size in
                assert (warp > 1 && Int.is_pow2 warp);
                let extent = to_ - from_ + 1 in
                (* gh-ocannl-682: the shuffle stages the accumulator at the backend's accumulator
                   RESIDENCY ([acc_prec], gh-ocannl-663), not at the node's storage precision — the
                   same residency every serial-rendered form of a recognized accumulation holds
                   (gh-ocannl-639), so a [Workgroup_reduce] retype does not change the width a
                   reduction accumulates at. [vname], the per-warp staging slots and the shuffle
                   stages all live at [prec]; the narrow cell is read widened and written narrowed
                   once, in [fold_total]. The gate is on the RESIDENCY, not on storage: where a
                   backend's accumulators stay narrow (bf16/f16 on HIP and Metal, f16 on CUDA) there
                   is no wider value to shuffle and no [ocannl_shfl_xor] overload to shuffle it
                   with, so those keep the loud refusal rather than gaining an untested
                   narrow-shuffle path. *)
                let store_prec = Lazy.force tn.Tn.storage_prec in
                let prec = acc_prec store_prec in
                (match prec with
                | Ops.Single_prec _ | Ops.Double_prec _ -> ()
                | _ ->
                    fail
                      (Printf.sprintf
                         "a single- or double-precision accumulator residency (accum_prec resolves \
                          %s storage to %s)"
                         (Ops.prec_string store_prec) (Ops.prec_string prec)));
                (* gh-ocannl-754: the shuffle may widen only where the serial rendering widens, and
                   the shared decision is what says so. A storage-pinned statement (an RNG-bearing
                   contribution, gh-ocannl-682) is refused only where the residency is actually
                   wider: at f32/f64 storage the two coincide, and such a reduction shuffles exactly
                   as it did before gh-ocannl-682. *)
                (match sa.sa_pinned with
                | None -> ()
                | Some Skip_debug_logging -> fail "debug_log_from_routines to be disabled"
                | Some Skip_dead_level -> fail "a live extent (the level is dead: to_ < from_)"
                | Some Skip_accum_pinned ->
                    if not (residency_is_storage tn) then
                      fail
                        (Printf.sprintf
                           "a contribution free of RNG conversions when the accumulator residency \
                            (%s) is wider than storage (%s): the serial rendering of an \
                            RNG-bearing accumulation declines localization and narrows its \
                            accumulator every iteration, so shuffling this one at the residency \
                            would change the accumulation width rather than only its association"
                           (Ops.prec_string prec) (Ops.prec_string store_prec)));
                if extent % warp <> 0 then
                  fail
                    (Printf.sprintf "the extent (%d) to be a multiple of the warp size (%d)" extent
                       warp);
                let num_warps = extent / warp in
                let axes = !current_hardware_axes in
                (match
                   List.find axes ~f:(fun a -> Indexing.equal_symbol a.Low_level.ha_index i)
                 with
                | Some a when a.Low_level.ha_slot = 0 -> ()
                | Some _ ->
                    fail
                      "the reduce axis at workgroup slot 0 (warp lanes are consecutive .x threads)"
                | None ->
                    invalid_arg
                      ("C_syntax.pp_ll: hardware-annotated loop " ^ symbol_ident i
                     ^ " missing from the slot table (pp_ll called outside compile_proc?)"));
                let slot_max =
                  List.fold axes ~init:1 ~f:(fun m a ->
                      match a.Low_level.ha_kind with
                      | `Workgroup when a.Low_level.ha_slot = 0 -> max m a.Low_level.ha_extent
                      | _ -> m)
                in
                if extent <> slot_max then
                  fail
                    "the extent to cover the whole workgroup .x dimension (a smaller sibling \
                     extent would diverge the shuffles)";
                let reg =
                  match B.hardware_index ~kind:`Workgroup ~slot:0 with
                  | Some reg -> reg
                  | None -> fail "the backend to bind workgroup slot 0"
                in
                if num_warps > 1 then (
                  if not (Int.is_pow2 num_warps) then
                    fail "a power-of-two number of warps (for the identity-free second phase)";
                  if num_warps > warp then
                    fail "at most warp-size warps (one second-phase slot per warp)";
                  if Option.is_none B.barrier_syntax || Option.is_none B.shared_decl_prefix then
                    fail "barrier and workgroup-shared support";
                  if
                    List.exists axes ~f:(fun a ->
                        match a.Low_level.ha_kind with
                        | `Workgroup -> not (Indexing.equal_symbol a.Low_level.ha_index i)
                        | `Grid -> false)
                  then
                    fail
                      "the reduce axis to be the only workgroup axis (the per-warp staging slots \
                       are not replicated per sibling workgroup thread)");
                let ident = symbol_ident i in
                let ctyp = B.typ_of_prec prec in
                let cast = "(" ^ String.strip B.loop_index_type ^ ")" in
                let vname = "wred_v_" ^ ident ^ "__" in
                let combine a b = B.binop_syntax prec op a b in
                let rec halvings n = if n < 1 then [] else n :: halvings (n / 2) in
                let shuffle_stage off =
                  string (vname ^ " = ")
                  ^^ combine (string vname)
                       (string (Printf.sprintf "ocannl_shfl_xor(%s, %d)" vname off))
                  ^^ semi
                in
                let acc_doc =
                  string (get_ident tn) ^^ brackets (pp_array_offset (idcs, Lazy.force tn.Tn.dims))
                in
                (* The cell keeps its storage precision (a declaration or a buffer element type is
                   never [acc_prec]'d), so the one place the residency meets storage is this fold:
                   read widened, combined at [prec], narrowed once. Empty spellings where the two
                   coincide, which is every backend at f32/f64 storage. *)
                let widen = B.convert_precision ~from:store_prec ~to_:prec in
                let narrow = B.convert_precision ~from:prec ~to_:store_prec in
                let fold_total =
                  group
                    (string "if (" ^^ pp_symbol i ^^ string " == 0) " ^^ lbrace
                    ^^ nest 2
                         (hardline ^^ acc_doc ^^ string " = "
                         ^^ wrap_conversion narrow
                              (combine (wrap_conversion widen acc_doc) (string vname))
                         ^^ semi)
                    ^^ hardline ^^ rbrace)
                in
                let tail =
                  if num_warps = 1 then fold_total
                  else
                    let pname = "wred_partials_" ^ ident ^ "__" in
                    let barrier = string (Option.value_exn B.barrier_syntax) in
                    string
                      (Printf.sprintf "%s%s %s[%d];"
                         (Option.value_exn B.shared_decl_prefix)
                         ctyp pname num_warps)
                    ^^ hardline
                    ^^ string
                         (Printf.sprintf "if ((%s & %d) == 0) { %s[%s >> %d] = %s; }" ident
                            (warp - 1) pname ident (Int.ceil_log2 warp) vname)
                    ^^ hardline ^^ barrier ^^ hardline
                    ^^ group
                         (string (Printf.sprintf "if (%s < %d) " ident warp)
                         ^^ lbrace
                         ^^ nest 2
                              (hardline
                              ^^ string
                                   (Printf.sprintf "if (%s < %d) { %s = %s[%s]; }" ident num_warps
                                      vname pname ident)
                              ^^ hardline
                              ^^ separate hardline
                                   (List.map (halvings (num_warps / 2)) ~f:shuffle_stage)
                              ^^ hardline ^^ fold_total)
                         ^^ hardline ^^ rbrace)
                    ^^ hardline ^^ barrier
                in
                (* The contribution renders at the residency, deliberately: operand widenings are
                   exact, so a narrow-times-narrow product is exact at [prec] — the same
                   full-precision-product-into-f32 semantics the serial legs and the tensor cores
                   apply (gh-ocannl-663's note in [Cuda_backend.accum_prec]). *)
                let local_defs, contrib_doc = pp_scalar prec contrib in
                let local_defs = pp_local_defs local_defs in
                let binding =
                  string ("const " ^ B.loop_index_type)
                  ^^ pp_symbol i
                  ^^ string (" = " ^ cast ^ reg ^ ";")
                in
                (* Past every refusal: the tree renders, and the census says which decision it took
                   its width from. *)
                record (width_site sa);
                Some
                  (string
                     (Printf.sprintf
                        "{ /* Workgroup_reduce warp-shuffle rendering: extent %d = %d simdgroup(s) \
                         of %d. */"
                        extent num_warps warp)
                  ^^ nest 2
                       (hardline ^^ binding ^^ hardline
                       ^^ (if PPrint.is_empty local_defs then empty else local_defs ^^ hardline)
                       ^^ string (ctyp ^ " " ^ vname ^ " = ")
                       ^^ contrib_doc ^^ semi ^^ hardline
                       ^^ separate hardline (List.map (halvings (warp / 2)) ~f:shuffle_stage)
                       ^^ hardline ^^ tail)
                  ^^ hardline ^^ rbrace)
        in
        (* gh-ocannl-639 / gh-ocannl-693: the plain-serial fallback of a reduction nest holds its
           accumulator in a scope LOCAL across the whole nest and stores once, after the nest — the
           same residency as [try_vectorize_reduce]'s epilogue, [try_register_tile]'s C-tile and
           virtual scopes ([scope_prec_of]). Two properties fall out of the one rewrite, and they
           are independent:

           - {b Width} (gh-ocannl-639): the local resides at the backend's accumulator precision
           ([acc_prec], gh-ocannl-663 — on CPU that is the compute precision) and narrows once at
           the store, so a reduction's effective accumulation width is set by the numerics policy,
           never by which schedule happened to place the accumulator in a register. - {b Residency}
           (gh-ocannl-693): the accumulator leaves the node's storage, so the nest performs one load
           and one store instead of a global read-modify-write per step. This holds at EVERY
           precision, the identity ones included — f32/f64/integers, [narrow_compute_f32 = false],
           native fp16, and the GPU backends' 16-bit precisions whose tensor units accumulate at
           storage width. At those the widening half is vacuous (the local's precision IS the
           storage precision) and the rewrite is exactly value-neutral, which is why it is
           unconditional: leaving it precision-gated made residency "whichever schedule happened to
           place it" at f32, and on Metal [volatile_serial_accumulation] pinned the resulting RMW to
           device memory by construction.

           Implemented as a local rewrite into exactly the [Local_scope] form virtualization gives
           virtual accumulators, rendered recursively: [scope_prec_of] (the minted scope is
           registered in [accum_scope_ids]) and the [Set_local]/[Get_local]/[Local_scope] arms
           already carry the residency, the widening and the single narrowing, so nothing new is
           emitted. Codegen is the ONLY place a materialized accumulator is localized ([optimize]
           rejects a [Local_scope] over a materialized node, gh-ocannl-681).

           The nest is peeled through Serial/[Unrolled]/[Vectorized] single-statement loop levels by
           [Low_level.peel_accum_nest], outermost-first ([pp_ll] recurses top-down), so the store
           lands above every enclosing level whose symbols the accumulator's cell is free of. A
           guard ([If]) that is not pure-index, a sibling statement, or any other inner loop stops
           the peel at that level (the recognizer then fails or the localization shrinks to that
           sub-nest). Two more declines, both unchanged by gh-ocannl-693: under
           [debug_log_from_routines] the per-step [Set] form is kept — a [Local_scope] body renders
           with [log_set_locals:false], so the rewrite would silence the per-iteration trace; the
           SIMD renderings bail there for the same reason, and logged narrow runs already differ
           numerically from plain runs (every tensorized rendering declines under logging). And an
           update mentioning an RNG conversion is never localized: the conversion picks its result
           type AND which random bits it consumes from the precision it renders at (gh-ocannl-517's
           carve-out, [renders_at_store_prec]), so rendering it inside a scope's precision would
           change the draw, not just move it.

           Interaction with [volatile_serial_accumulation] (Metal) has two forms. Localization lifts
           the node [Set] out of exactly the invariant-address loops, so the device-memory RMW
           predicate is false at a fully localized site. But gh-ocannl-731 showed the same shader
           compiler pass corrupting the replacement scope-local accumulation when its contribution
           reads through a pooled pointer; [Set_local] therefore renders those device reads through
           confined volatile pointer casts. Where the peel is blocked at an outer level, the
           device-memory RMW remains and its reads receive the same expression-level cast. *)
        let try_localize_serial_reduce () : PPrint.document option =
          (* The decision is the shared one above (gh-ocannl-754); this arm only renders what it
             says and records it. *)
          match decide_accum_width () with
          | Accum_not_a_nest refusal ->
              record (Peel_refused refusal);
              None
          | Accum_base { pinned = Some skip; _ } ->
              (* The peel reached a base and the decision declined it — under logging, at a dead
                 level, or the storage-pinned accumulator. A census that recorded the peel's success
                 here would credit the site with a rewrite the kernel does not contain. *)
              record (Peel_not_attempted skip);
              None
          | Accum_base { tn; idcs; base; debug; rebuild; verdict; pinned = None } ->
              let doc =
                let id, update_code =
                  match base with
                  | `Update llsc ->
                      let id = Low_level.get_scope tn in
                      (* Codegen-minted, so the census over [B.procs] never saw it: register the
                         scope as an accumulator or [scope_prec_of] would resolve it at [comp_prec]
                         and defeat the widening on the backends where the two differ
                         (gh-ocannl-663). Fresh ids per mint, so no collision with a censused
                         verdict. *)
                      Hash_set.add accum_scope_ids id.Low_level.scope_id;
                      (id, Low_level.Set_local (id, Low_level.subst_accum_read ~tn ~idcs ~id llsc))
                  | `Scope (id, rest) ->
                      (* The scope-form base [Sched.Unroll ~materialize:true] minted (or a previous
                         level of this very rewrite): hoist it through the enclosing reduction
                         levels by moving its init above them and keeping its updates inside —
                         otherwise a partially materialized nest would store and narrow the
                         accumulator once per remaining outer iteration. The scope id is reused, so
                         the rng census's storage-precision marking still applies; at storage
                         precision the hoist is value-neutral. *)
                      (id, Low_level.unflat_lines rest)
                in
                (* The level being rendered is re-wrapped around the rebuilt nest here; the peel
                   never saw its bounds, which is why the dead-level decline is the decision's and
                   not the peel's. *)
                let rebuild_hook b = Low_level.For_loop { index = i; from_; to_; body = b; axis } in
                let rec loop_symbols = function
                  | Low_level.For_loop { index; body; _ } | Scan_loop { index; body; _ } ->
                      index :: loop_symbols body
                  | If { body; _ } -> loop_symbols body
                  | Seq (left, right) -> loop_symbols left @ loop_symbols right
                  | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Set _ | Set_local _
                  | Set_dynamic _ | Set_from_vec _ | Declare_local _ | Workgroup_barrier
                  | Tile_mma _ ->
                      []
                in
                let localized_symbols = i :: loop_symbols (rebuild Low_level.Noop) in
                let scope_body =
                  let opening =
                    match !current_localized_zero_seed with
                    | Some seed
                      when Tn.equal seed.lzs_tn tn
                           && Array.equal Indexing.equal_axis_index seed.lzs_idcs idcs ->
                        if
                          List.for_all seed.lzs_repeated ~f:(fun repeated ->
                              List.mem localized_symbols repeated ~equal:Indexing.equal_symbol)
                        then begin
                          seed.lzs_consumed <- true;
                          Low_level.Constant 0.0
                        end
                        else Low_level.Get (tn, idcs)
                    | _ -> Low_level.Get (tn, idcs)
                  in
                  Low_level.Seq
                    (Low_level.Set_local (id, opening), rebuild_hook (rebuild update_code))
                in
                without_census (fun () ->
                    pp_ll ~log_set_locals ~in_loop
                      (Low_level.Set
                         {
                           tn;
                           idcs;
                           llsc =
                             Local_scope
                               {
                                 id;
                                 body = scope_body;
                                 orig_indices = idcs;
                                 mint = Schedule_minted;
                               };
                           debug;
                         }))
              in
              record (Peel_localized verdict);
              Some doc
        in
        let localize_or_serial () =
          match try_localize_serial_reduce () with Some doc -> doc | None -> serial_loop ()
        in
        match axis with
        | Low_level.Serial -> localize_or_serial ()
        | Grid when Set.mem !current_parallel_grid i -> parallel_grid_loop ()
        | Grid -> hardware_binding `Grid
        | Workgroup -> hardware_binding ~fallback:localize_or_serial `Workgroup
        | Workgroup_reduce -> (
            match try_warp_reduce () with
            | Some doc -> doc
            | None -> hardware_binding ~fallback:localize_or_serial `Workgroup)
        | Vectorized -> (
            match try_vectorize_reduce () with
            | Some doc -> doc
            | None -> (
                match try_vectorize () with
                | Some doc -> doc
                | None -> (
                    if
                      (* gh-ocannl-164: a serial loop annotated with the backend's vectorization
                         pragmas; without them the plain serial loop is the legal fallback (same
                         discipline as unbound [Grid]/[Workgroup] axes). The pragmas assert
                         iteration independence, which a loop-carried accumulation does not satisfy
                         — an accumulating body that no explicit rendering accepted always falls
                         back to the plain serial loop (gh-ocannl-468: this is what lets the
                         autotune menu propose Retype-[Vectorized] over reductions). *)
                      Low_level.has_accumulation body
                    then
                      match try_localize_serial_reduce () with
                      | Some doc -> doc
                      | None -> serial_loop ()
                    else
                      match B.vectorize_pragma with
                      | [] -> serial_loop ()
                      | lines -> separate_map hardline string lines ^^ hardline ^^ serial_loop ())))
        | Unrolled -> (
            (* An [Unrolled] reduction axis holds the wide local across the repeated bodies
               (gh-ocannl-639): autotune proposes [Unroll] over any small Serial loop, reduction
               loops included, and without this hook the unrolled candidate would round-trip the
               accumulator through storage on every repetition while the serial baseline widens —
               numerics-divergent candidates in one search. *)
            match try_localize_serial_reduce () with
            | Some doc -> doc
            | None ->
                separate hardline
                (* [max 0]: a DEAD level ([to_ < from_]) unrolls to zero repetitions, and rendering
                   nothing is exactly its access-free meaning. The clamp is load-bearing rather than
                   defensive — [Base.List.init] RAISES on a negative length, so without it a dead
                   [Unrolled] level aborts codegen. gh-ocannl-693 widened the reach of that: the
                   localizer used to peel such a level and never fall through here, and now declines
                   it (a dead level must not be peeled), so this arm receives what the peel
                   refused. *)
                @@ List.init
                     (max 0 (to_ - from_ + 1))
                     ~f:(fun k ->
                       let binding =
                         string ("const " ^ B.loop_index_type)
                         ^^ pp_symbol i
                         ^^ string (" = " ^ Int.to_string (from_ + k) ^ ";")
                       in
                       group
                         (lbrace
                         ^^ nest 2 (hardline ^^ binding ^^ hardline ^^ body_doc ())
                         ^^ hardline ^^ rbrace))))
    | Scan_loop { index = i; from_; to_; direction; carried; body } ->
        (* gh-ocannl-696: the carried state renders as two locals per entry, declared and
           initialized ahead of a serial loop -- counting down for [Backward] -- whose body ends
           with the rotation [prev = next]. Declarations, inits and the rotation go through the
           [Declare_local] / [Set_local] arms, so the state's type, precision conversions and
           runtime logging are exactly a scope local's. The scan index joins [serial_loop_stack]
           like a serial loop's, since a symbolic-extent guard inside the body must not take it for
           a kernel parameter. *)
        let decls =
          List.concat_map carried ~f:(fun { Low_level.prev; next; init } ->
              [
                pp_ll ~log_set_locals ~in_loop (Declare_local { id = prev; needs_init = false });
                pp_ll ~log_set_locals ~in_loop (Declare_local { id = next; needs_init = false });
                pp_ll ~log_set_locals ~in_loop (Set_local (prev, init));
              ])
        in
        let header =
          let ty = string ("for (" ^ B.loop_index_type) in
          match direction with
          | Forward ->
              ty ^^ pp_symbol i ^^ string " = " ^^ PPrint.OCaml.int from_ ^^ semi ^^ space
              ^^ pp_symbol i ^^ string " <= " ^^ PPrint.OCaml.int to_ ^^ semi ^^ space
              ^^ string "++" ^^ pp_symbol i ^^ string ")"
          | Backward ->
              ty ^^ pp_symbol i ^^ string " = " ^^ PPrint.OCaml.int to_ ^^ semi ^^ space
              ^^ pp_symbol i ^^ string " >= " ^^ PPrint.OCaml.int from_ ^^ semi ^^ space
              ^^ string "--" ^^ pp_symbol i ^^ string ")"
        in
        serial_loop_stack := i :: !serial_loop_stack;
        let body_doc =
          Exn.protect
            ~f:(fun () ->
              let doc = pp_ll ~log_set_locals ~in_loop:true body in
              let rotation =
                List.map carried ~f:(fun { Low_level.prev; next; _ } ->
                    pp_ll ~log_set_locals ~in_loop:true (Set_local (prev, Get_local next)))
              in
              let doc = separate hardline (doc :: rotation) in
              if Utils.debug_log_from_routines () then
                (* Identical to the [For_loop] arm's trace line, index width included: the two loop
                   kinds' per-iteration traces are read side by side. *)
                let spec, arg_doc = B.log_index_arg (pp_symbol i) in
                let base_message = Printf.sprintf "index %s = %s\n" (symbol_ident i) spec in
                let log_param_doc =
                  Option.map B.kernel_log_param ~f:(fun (_, name) -> string name)
                in
                B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
                  ~base_message_literal:base_message ~args_docs:[ arg_doc ]
                ^^ hardline ^^ doc
              else doc)
            ~finally:(fun () -> serial_loop_stack := List.tl_exn !serial_loop_stack)
        in
        let loop_doc =
          group (header ^^ space ^^ lbrace ^^ nest 2 (hardline ^^ body_doc) ^^ hardline ^^ rbrace)
        in
        lbrace
        ^^ nest 2 (hardline ^^ separate hardline decls ^^ hardline ^^ loop_doc)
        ^^ hardline ^^ rbrace
    | Zero_out tn ->
        let first_touch = not (Hash_set.mem zero_out_seen tn.Tn.uid) in
        Hash_set.add zero_out_seen tn.Tn.uid;
        if first_touch && (not in_loop) && zero_out_loop_redundant tn then
          (* First-touch, executed once at function scope: the declaration's [= {0}] already covers
             it. A later [Zero_out tn], or one reached inside a loop, is a real re-zero and falls
             through to emit the zeroing loop below. *)
          empty
        else
          pp_ll ~log_set_locals ~in_loop
            (Low_level.loop_over_dims (Lazy.force tn.dims) ~body:(fun idcs ->
                 Set { tn; idcs; llsc = Constant 0.0; debug = get_ident tn ^ " := 0" }))
    | Set { tn; idcs; llsc; debug } -> (
        let ident_doc = string (get_ident tn) in
        let dims = Lazy.force tn.dims in
        let store_prec = Lazy.force tn.storage_prec in
        (* See {!C_syntax_config.volatile_serial_accumulation}: identify a per-iteration
           read-modify-write whose target address is invariant across an enclosing serial loop. The
           old workaround shadowed the target pointer for the whole statement; gh-ocannl-820 instead
           brackets [pp_scalar] so only device reads in the accumulating expression acquire an
           expression-level volatile pointer cast. *)
        let rmw_volatile_reads =
          B.volatile_serial_accumulation
          && List.exists !serial_loop_stack ~f:(fun s ->
              not (Array.exists idcs ~f:(Indexing.axis_index_mentions_symbol s)))
          (* Only kernel-parameter-derived device pointers: routine-local scratch is declared as a
             plain local array (not address-castable, and compiler-visible anyway). *)
          && Tn.Placements.is_materialized_force (placements ()) tn 433
          &&
          let rec reads_tn (llsc : Low_level.scalar_t) =
            match llsc with
            | Low_level.Get (tn2, _) -> Tn.equal tn tn2
            | Get_dynamic { tn = tn2; dyn_value = v, _; _ } -> Tn.equal tn tn2 || reads_tn v
            | Local_scope { body; _ } -> body_reads_tn body
            | Get_local _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ ->
                false
            | Ternop (_, (a, _), (b, _), (c, _)) -> reads_tn a || reads_tn b || reads_tn c
            | Binop (_, (a, _), (b, _)) -> reads_tn a || reads_tn b
            | Unop (_, (a, _)) -> reads_tn a
          and body_reads_tn (body : Low_level.t) =
            match body with
            | Low_level.Seq (a, b) -> body_reads_tn a || body_reads_tn b
            | For_loop { body; _ } -> body_reads_tn body
            | Scan_loop { carried; body; _ } ->
                List.exists carried ~f:(fun c -> reads_tn c.Low_level.init) || body_reads_tn body
            | If { cond = c, _; body } -> reads_tn c || body_reads_tn body
            | Set { llsc; _ } | Set_local (_, llsc) -> reads_tn llsc
            | Set_dynamic { dyn_value = v, _; llsc; _ } -> reads_tn v || reads_tn llsc
            | Set_from_vec { arg = a, _; _ } -> reads_tn a
            | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Declare_local _
            | Workgroup_barrier | Tile_mma _ ->
                false
          in
          reads_tn llsc
        in
        (* gh-487 phase 2: a staging copy into an async-eligible pipelined tile renders as the
           backend's asynchronous copy — the same address arithmetic on both sides (the write-side
           buffer rotation included), byte-for-byte, so the bitwise-identity invariant is untouched;
           completion is the rotor loop's wait+barrier prefix. The eligibility pattern is exact: a
           raw [Get] of a materialized (global) node at the tile's own storage precision. Anything
           else — a precision conversion, a surviving zero-fringe ternary, a non-global source —
           falls through to the plain synchronous store, which the same barrier publishes. *)
        let async_copy_doc =
          if not (Set.mem !current_async_tiles tn) then None
          else
            match (B.async_copy, llsc) with
            | Some ac, Low_level.Get (src, src_idcs)
              when Ops.equal_prec (Lazy.force src.Tn.storage_prec) store_prec
                   && Tn.Placements.is_materialized_force (placements ()) src 487 ->
                let offset_doc =
                  pp_pipelined_rotation ~is_write:true tn ^^ pp_tn_offset tn (idcs, dims)
                in
                let src_offset_doc = pp_tn_offset src (src_idcs, Lazy.force src.Tn.dims) in
                Some
                  (ac.ac_copy
                     ~dst:(string "&" ^^ ident_doc ^^ brackets offset_doc)
                     ~src:(string "&" ^^ string (get_ident src) ^^ brackets src_offset_doc)
                     ~bytes:(Ops.prec_in_bytes store_prec))
            | _ -> None
        in
        match async_copy_doc with
        | Some doc -> doc
        | None ->
            let prec, narrowing = store_precs ~store_prec llsc in
            let (local_defs, val_doc), volatile_read_emitted =
              with_volatile_accumulation_reads rmw_volatile_reads (fun () -> pp_scalar prec llsc)
            in
            let val_doc = wrap_conversion narrowing val_doc in
            let local_defs = pp_local_defs local_defs in
            let offset_doc =
              pp_pipelined_rotation ~is_write:true tn ^^ pp_tn_offset tn (idcs, dims)
            in
            (* The volatility census (gh-ocannl-782) records this arm only where it fires. Unlike
               the accumulator arm, its predicate is not evaluated at all on a backend that requests
               nothing — the capability short-circuits it — so there is no decision there to report,
               and [volatility_summary.requested] is what says so. *)
            if volatile_read_emitted && !volatility_census_enabled then
              volatility_events := Site (Volatile_rmw_reads (get_ident tn)) :: !volatility_events;
            let assignment =
              group
                (ident_doc ^^ brackets offset_doc ^^ string " ="
                ^^ ifflat (space ^^ val_doc) (nest 4 (hardline ^^ val_doc))
                ^^ semi)
            in
            if Utils.debug_log_from_routines () then
              (* [val_doc] is already narrowed to the storage precision, so the temporary carrying
                 it into the log statement takes the storage type. *)
              let num_typ = string (B.typ_of_prec store_prec) in
              let new_var = string "new_set_v" in
              let decl = num_typ ^^ space ^^ new_var ^^ string " = " ^^ val_doc ^^ semi in
              let debug_val_doc, debug_args_docs = debug_float prec llsc in
              let debug_val_str = doc_to_string debug_val_doc in
              let pp_args_docs =
                List.map debug_args_docs ~f:(function
                  | `Accessor idx -> pp_array_offset idx
                  | `Value v_doc -> B.styled_log_arg v_doc)
              in
              let log_args_for_printf =
                offset_doc
                :: B.styled_log_arg (ident_doc ^^ brackets offset_doc)
                :: B.styled_log_arg new_var :: pp_args_docs
              in
              let log_doc =
                let log_param_doc =
                  Option.map B.kernel_log_param ~f:(fun (_, name) -> string name)
                in
                let comment_base_msg = "# " ^ debug ^ "\n" in
                let value_base_msg =
                  Printf.sprintf "%s[%%u]{=%s} = %s = %s\n" (get_ident tn) B.float_log_style
                    B.float_log_style debug_val_str
                in
                let comment_log =
                  B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
                    ~base_message_literal:comment_base_msg ~args_docs:[]
                in
                let value_log =
                  B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
                    ~base_message_literal:value_base_msg ~args_docs:log_args_for_printf
                in
                let flush_log =
                  if B.log_involves_file_management then string "fflush(log_file);" else empty
                in
                comment_log ^^ hardline ^^ value_log ^^ hardline ^^ flush_log
              in
              let assignment' =
                ident_doc ^^ brackets offset_doc ^^ string " = " ^^ new_var ^^ semi
              in
              let block_content =
                if PPrint.is_empty local_defs then
                  decl ^^ hardline ^^ log_doc ^^ hardline ^^ assignment'
                else
                  local_defs ^^ hardline ^^ decl ^^ hardline ^^ log_doc ^^ hardline ^^ assignment'
              in
              lbrace ^^ nest 2 (hardline ^^ block_content) ^^ hardline ^^ rbrace
            else if PPrint.is_empty local_defs then assignment
            else
              let block_content = local_defs ^^ hardline ^^ assignment in
              lbrace ^^ nest 2 (hardline ^^ block_content) ^^ hardline ^^ rbrace)
    | Set_dynamic { tn; idcs; dyn_axis; dyn_value = iv, iprec; llsc; debug } ->
        (* gh-466: the scatter counterpart of the [Get_dynamic] gather — the write offset splices
           the runtime index (cast to [Ops.index_prec ()], mirroring the gather) at [dyn_axis]. The
           enclosing [If] guard (when interval analysis has not discharged it) guarantees the index
           is in range before this statement executes. *)
        if is_swizzled tn || is_pipelined tn then
          invalid_arg
            ("C_syntax: Set_dynamic targets swizzled or pipelined node " ^ Tn.debug_name tn
           ^ " (dynamic offsets are not swizzle-remapped)");
        let ident_doc = string (get_ident tn) in
        let dims = Lazy.force tn.dims in
        let store_prec = Lazy.force tn.storage_prec in
        let prec, narrowing = store_precs ~store_prec llsc in
        let dyn_defs, dyn_expr = pp_scalar iprec iv in
        let idx_typ = B.typ_of_prec (Ops.index_prec ()) in
        let dyn_idx_doc = string ("((" ^ idx_typ ^ ")(") ^^ dyn_expr ^^ string "))" in
        let offset_doc = pp_array_offset_dyn (idcs, dims) ~dyn_axis ~dyn_idx_doc in
        let val_defs, val_doc = pp_scalar prec llsc in
        let val_doc = wrap_conversion narrowing val_doc in
        let local_defs = pp_local_defs (dyn_defs @ val_defs) in
        let assignment =
          group
            (ident_doc ^^ brackets offset_doc ^^ string " ="
            ^^ ifflat (space ^^ val_doc) (nest 4 (hardline ^^ val_doc))
            ^^ semi)
        in
        if Utils.debug_log_from_routines () then
          (* [val_doc] is already narrowed to the storage precision, so the temporary carrying it
             into the log statement takes the storage type. *)
          let num_typ = string (B.typ_of_prec store_prec) in
          let new_var = string "new_set_v" in
          let decl = num_typ ^^ space ^^ new_var ^^ string " = " ^^ val_doc ^^ semi in
          let debug_val_doc, debug_args_docs = debug_float prec llsc in
          let debug_val_str = doc_to_string debug_val_doc in
          let pp_args_docs =
            List.map debug_args_docs ~f:(function
              | `Accessor idx -> pp_array_offset idx
              | `Value v_doc -> B.styled_log_arg v_doc)
          in
          let log_args_for_printf =
            offset_doc
            :: B.styled_log_arg (ident_doc ^^ brackets offset_doc)
            :: B.styled_log_arg new_var :: pp_args_docs
          in
          let log_doc =
            let log_param_doc = Option.map B.kernel_log_param ~f:(fun (_, name) -> string name) in
            let comment_base_msg = "# " ^ debug ^ "\n" in
            let value_base_msg =
              Printf.sprintf "%s[%%u]{=%s} = %s = %s\n" (get_ident tn) B.float_log_style
                B.float_log_style debug_val_str
            in
            let comment_log =
              B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
                ~base_message_literal:comment_base_msg ~args_docs:[]
            in
            let value_log =
              B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
                ~base_message_literal:value_base_msg ~args_docs:log_args_for_printf
            in
            let flush_log =
              if B.log_involves_file_management then string "fflush(log_file);" else empty
            in
            comment_log ^^ hardline ^^ value_log ^^ hardline ^^ flush_log
          in
          let assignment' = ident_doc ^^ brackets offset_doc ^^ string " = " ^^ new_var ^^ semi in
          let block_content =
            if PPrint.is_empty local_defs then
              decl ^^ hardline ^^ log_doc ^^ hardline ^^ assignment'
            else local_defs ^^ hardline ^^ decl ^^ hardline ^^ log_doc ^^ hardline ^^ assignment'
          in
          lbrace ^^ nest 2 (hardline ^^ block_content) ^^ hardline ^^ rbrace
        else if PPrint.is_empty local_defs then assignment
        else
          let block_content = local_defs ^^ hardline ^^ assignment in
          lbrace ^^ nest 2 (hardline ^^ block_content) ^^ hardline ^^ rbrace
    | Comment message ->
        if Utils.debug_log_from_routines () then
          let base_message = "COMMENT: " ^ message ^ "\n" in
          let log_param_doc = Option.map B.kernel_log_param ~f:(fun (_, name) -> string name) in
          B.pp_log_statement ~log_param_c_expr_doc:log_param_doc ~base_message_literal:base_message
            ~args_docs:[]
        else string "/* " ^^ string message ^^ string " */"
    | Staged_compilation callback -> callback ()
    | Set_from_vec { tn; idcs; length; vec_unop; arg = arg, arg_prec; debug } ->
        (* Multi-element consecutive-offset write: incompatible with a swizzled layout, and staged
           tiles are only ever written by the Stage-minted scalar load nest — fail loudly rather
           than miscompile if that invariant is ever broken. *)
        if is_swizzled tn || is_pipelined tn then
          invalid_arg
            ("C_syntax: Set_from_vec targets swizzled or pipelined node " ^ Tn.debug_name tn
           ^ " (row-major multi-element write into an XOR-swizzled layout)");
        let ident_doc = string (get_ident tn) in
        let dims = Lazy.force tn.dims in
        let prec = Lazy.force tn.storage_prec in
        (* Determine argument precision based on operation homogeneity *)
        let arg_prec =
          if Ops.is_homogeneous_prec_vec_unop vec_unop then prec
            (* Homogeneous: argument uses result precision *)
          else arg_prec
        in
        let local_defs, arg_doc = pp_scalar arg_prec arg in
        let local_defs = pp_local_defs local_defs in
        (* Generate the function call *)
        let result_doc = B.vec_unop_syntax prec vec_unop arg_doc in
        (* The vector temporary always has the full 128-bit block width: a tail store ([length] <
           block lanes) stores only the leading lanes of the fully computed block. *)
        let block_lanes = Ops.vec_unop_lanes prec in
        (* Generate assignments for each output element *)
        let open PPrint in
        let vec_var = string "vec_result" in
        let vec_typ = string (B.vec_typ_of_prec ~length:block_lanes prec) in
        let vec_decl = vec_typ ^^ space ^^ vec_var ^^ string " = " ^^ result_doc ^^ semi in
        let assignments =
          let elem_assigns =
            List.init length ~f:(fun i ->
                let offset_doc =
                  match idcs.(Array.length idcs - 1) with
                  | Fixed_idx idx ->
                      (* For Fixed_idx, update the index and compute offset normally *)
                      let elem_idcs = Array.copy idcs in
                      elem_idcs.(Array.length elem_idcs - 1) <- Fixed_idx (idx + i);
                      pp_array_offset (elem_idcs, dims)
                  | _ ->
                      (* For non-Fixed_idx (Iterator, etc), add i to the computed offset *)
                      pp_array_offset (idcs, dims) ^^ string (" + " ^ Int.to_string i)
                in
                let value_doc =
                  if block_lanes = 1 then
                    (* A single-lane block has a scalar type, so no .v[] access *)
                    vec_var
                  else
                    (* When the block has multiple lanes, access the vector element *)
                    vec_var ^^ string (".v[" ^ Int.to_string i ^ "]")
                in
                ident_doc ^^ brackets offset_doc ^^ string " = " ^^ value_doc ^^ semi)
          in
          separate hardline elem_assigns
        in
        if Utils.debug_log_from_routines () then
          let open PPrint in
          let log_param_doc = Option.map B.kernel_log_param ~f:(fun (_, name) -> string name) in
          let comment_base_msg = "# " ^ debug ^ "\n" in
          let comment_log =
            B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
              ~base_message_literal:comment_base_msg ~args_docs:[]
          in
          let value_logs =
            List.init length ~f:(fun i ->
                let elem_idcs = Array.copy idcs in
                (match elem_idcs.(Array.length elem_idcs - 1) with
                | Fixed_idx idx -> elem_idcs.(Array.length elem_idcs - 1) <- Fixed_idx (idx + i)
                | _ -> ());
                let offset_doc =
                  let base_offset = pp_array_offset (elem_idcs, dims) in
                  match elem_idcs.(Array.length elem_idcs - 1) with
                  | Fixed_idx _ -> base_offset
                  | _ -> base_offset ^^ string (" + " ^ Int.to_string i)
                in
                let value_base_msg =
                  Printf.sprintf "%s[%%u]{=%s} = vec_result.v[%d] = %s\n" (get_ident tn)
                    B.float_log_style i B.float_log_style
                in
                let log_args =
                  [
                    offset_doc;
                    B.styled_log_arg (ident_doc ^^ brackets offset_doc);
                    B.styled_log_arg (string ("vec_result.v[" ^ Int.to_string i ^ "]"));
                  ]
                in
                B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
                  ~base_message_literal:value_base_msg ~args_docs:log_args)
          in
          let flush_log =
            if B.log_involves_file_management then string "fflush(log_file);" else empty
          in
          let log_docs =
            comment_log ^^ hardline ^^ separate hardline value_logs ^^ hardline ^^ flush_log
          in
          let block_content =
            if PPrint.is_empty local_defs then
              vec_decl ^^ hardline ^^ log_docs ^^ hardline ^^ assignments
            else
              local_defs ^^ hardline ^^ vec_decl ^^ hardline ^^ log_docs ^^ hardline ^^ assignments
          in
          lbrace ^^ nest 2 (hardline ^^ block_content) ^^ hardline ^^ rbrace
        else if PPrint.is_empty local_defs then vec_decl ^^ hardline ^^ assignments
        else
          let block_content = local_defs ^^ hardline ^^ vec_decl ^^ hardline ^^ assignments in
          lbrace ^^ nest 2 (hardline ^^ block_content) ^^ hardline ^^ rbrace
    | Set_local (({ tn = { storage_prec = _; _ }; _ } as id), value) ->
        (* A scope-local scalar is declared at [scope_prec_of] its node (see [Declare_local] and
           [pp_scalar]'s [Local_scope]), so a self-referential update renders there too: an inlined
           narrow intermediate keeps its residency's mantissa across the whole scope instead of
           being rounded at every assignment to it.

           A [Set_local] that does NOT read its own local is an INIT — the inlined image of a
           separate source assignment — and renders as that assignment's store would: at [comp_prec]
           (the rng carve-out rides [scope_prec_of]'s storage pin), converted once into the scope's
           residency. Where scope and compute precision coincide (the CPU backends, non-accumulator
           scopes) this is the identity; where an accumulator resides wider than the arithmetic
           (gh-ocannl-663), it preserves the initializer's own per-assignment rounding — e.g. a
           virtual bf16 node initialized by [a + b] and then reduced narrows the init to bf16
           exactly as its materialized twin's store does, instead of leaking f32 residency into a
           different source assignment's semantics (Codex P2 round 3 on PR #396; the provenance
           boundary of gh-ocannl-639's adjacent-accumulations rule). *)
        let prec = scope_prec_of id in
        let value_prec =
          if Low_level.scalar_reads_scope ~id value then prec
          else if Hash_set.mem rng_scope_ids id then prec
          else comp_prec (Lazy.force id.Low_level.tn.Tn.storage_prec)
        in
        let volatile_reads =
          B.volatile_serial_accumulation
          && Hash_set.mem accum_scope_ids id.Low_level.scope_id
          && (not (List.is_empty !serial_loop_stack))
          && Option.is_some (Low_level.accum_local_update_op ~id value)
        in
        let (local_defs, value_doc), _ =
          with_volatile_accumulation_reads ~scope_ids:[ id.Low_level.scope_id ] volatile_reads
            (fun () -> pp_scalar value_prec value)
        in
        let value_doc =
          wrap_conversion (B.convert_precision ~from:value_prec ~to_:prec) value_doc
        in
        let local_defs = pp_local_defs local_defs in
        let assignment = pp_scope_id id ^^ string " = " ^^ value_doc ^^ semi in
        if Utils.debug_log_from_routines () && log_set_locals then
          let new_var = string "new_set_local_v" in
          let num_typ = string (B.typ_of_prec prec) in
          let decl = num_typ ^^ space ^^ new_var ^^ string " = " ^^ value_doc ^^ semi in
          let debug_val_doc, debug_args_docs = debug_float value_prec value in
          let debug_val_str = doc_to_string debug_val_doc in
          let pp_args_docs =
            List.map debug_args_docs ~f:(function
              | `Accessor idx -> pp_array_offset idx
              | `Value v_doc -> B.styled_log_arg v_doc)
          in
          let log_doc =
            let log_param_doc = Option.map B.kernel_log_param ~f:(fun (_, name) -> string name) in
            let scope_doc = pp_scope_id id in
            let comment_base_msg =
              "# local " ^ doc_to_string scope_doc ^ " := " ^ doc_to_string value_doc ^ "\n"
            in
            let value_base_msg =
              Printf.sprintf "%s{=%s} = %s = %s\n" (doc_to_string scope_doc) B.float_log_style
                B.float_log_style debug_val_str
            in
            let comment_log =
              B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
                ~base_message_literal:comment_base_msg ~args_docs:[]
            in
            let value_log =
              B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
                ~base_message_literal:value_base_msg
                ~args_docs:(B.styled_log_arg scope_doc :: B.styled_log_arg new_var :: pp_args_docs)
            in
            let flush_log =
              if B.log_involves_file_management then string "fflush(log_file);" else empty
            in
            comment_log ^^ hardline ^^ value_log ^^ hardline ^^ flush_log
          in
          let assignment' = pp_scope_id id ^^ string " = " ^^ new_var ^^ semi in
          let block_content =
            if PPrint.is_empty local_defs then
              decl ^^ hardline ^^ log_doc ^^ hardline ^^ assignment'
            else local_defs ^^ hardline ^^ decl ^^ hardline ^^ log_doc ^^ hardline ^^ assignment'
          in
          lbrace ^^ nest 2 (hardline ^^ block_content) ^^ hardline ^^ rbrace
        else if PPrint.is_empty local_defs then assignment
        else
          let block_content = local_defs ^^ hardline ^^ assignment in
          lbrace ^^ nest 2 (hardline ^^ block_content) ^^ hardline ^^ rbrace
    | Declare_local { id = { tn = { storage_prec = _; _ }; _ } as id; needs_init } ->
        record_accumulation_scope id;
        let scope_prec = scope_prec_of id in
        let num_typ = string (scope_decl_type id) in
        let init_zero =
          (* Runtime instrumentation prints both the old and new values for [Set_local]. Even when
             the computation itself writes this local before reading it ([needs_init = false]), the
             debug print therefore observes the declaration's value first. Initialize in debug
             builds so instrumentation does not introduce an undefined read (and backend-specific
             garbage) into an otherwise well-defined computation. *)
          if needs_init || Utils.debug_log_from_routines () then
            let prefix, postfix = B.convert_precision ~from:Ops.int32 ~to_:scope_prec in
            string " = " ^^ string prefix ^^ string "0" ^^ string postfix
          else empty
        in
        num_typ ^^ space ^^ pp_scope_id id ^^ init_zero ^^ semi
    | Workgroup_barrier -> (
        match B.barrier_syntax with
        | Some s -> string s
        | None ->
            invalid_arg
              "C_syntax.pp_ll: Workgroup_barrier not supported by this backend (serialization \
               cannot implement a barrier)")
    | Tile_mma { d; a; b; ta; tb; m; n; k; ldd; lda; ldb; lane; tile; fallback } -> (
        (* Cooperative tile-MMA (docs/proposals/tensorize-mma.md §4). Backends with an [mma_syntax]
           hook emit the intrinsic sequence on every lane; everywhere else (including per-call
           declines and logged runs, which must stay serial and deterministic) the scalar fallback
           runs once per simdgroup, guarded on lane 0 — the lane loop still supplies the launch's
           threads. *)
        let d_tn = fst d in
        let fragment_is_active =
          match !active_mma_accumulator with
          | Some (Active_fragment (fragment, _)) | Some (Active_target (fragment, _)) ->
              Tn.equal d_tn fragment
          | None -> false
        in
        if Set.mem !current_simdgroup_fragments d_tn && not fragment_is_active then
          declinef
            "marked simdgroup fragment %s rendered outside its recognized fragment scope; falling \
             back from the intended persistent intrinsic path"
            (Tn.debug_name d_tn);
        (* [ld] is the statement's recorded leading-dimension stride for this operand
           ([Tile_mma.ldd]/[lda]/[ldb]) — the minor dim of the tnode's last axis in the plain case,
           a larger stride when interior batch axes sit between the tile roles (gh-ocannl-528). *)
        let operand ld (tn, idcs) =
          let dims = Lazy.force tn.Tn.dims in
          let prec = Lazy.force tn.Tn.storage_prec in
          let ptr_doc, ld, space, layout =
            match !active_mma_accumulator with
            | Some (Active_fragment (fragment, name)) when Tn.equal tn fragment ->
                (string name, ld, `Fragment name, `Plain)
            | Some (Active_target (fragment, (ptr, target_ld, space, layout)))
              when Tn.equal tn fragment ->
                (ptr, target_ld, space, (layout :> [ mma_layout | `Decline of string ]))
            | _ ->
                let space =
                  if Set.mem !current_workgroup_shared tn then `Shared
                  else if Tn.Placements.is_materialized_force (placements ()) tn 440 then `Device
                  else `Thread
                in
                ( parens
                    (string (get_ident tn)
                    ^^ string " + "
                    (* A pipelined operand tile's pointer carries the read-side buffer rotation
                       (gh-487) — the intrinsic loads then read the current iteration's copy. *)
                    ^^ pp_pipelined_rotation ~is_write:false tn
                    ^^ pp_array_offset (idcs, dims)),
                  ld,
                  space,
                  operand_layout tn ~ld ~idcs ~dims )
          in
          (prec, (ptr_doc, ld, space, layout))
        in
        let lane0_guarded body_doc =
          let guarded =
            group
              (string "if (" ^^ pp_symbol lane ^^ string " == 0) " ^^ lbrace
              ^^ nest 2 (hardline ^^ body_doc)
              ^^ hardline ^^ rbrace)
          in
          (* On backends that bind the lane loop in hardware, sibling statements (e.g. the zeroing
             of [d]) execute lane-partitioned: bracket the single-lane fallback in barriers so lane
             0 observes the other lanes' writes, and they its. Uniform by the barrier-strength
             validation [Tile_mma] inherits. Serial renderers need no ordering. *)
          match B.barrier_syntax with
          | Some s -> string s ^^ hardline ^^ guarded ^^ hardline ^^ string s
          | None -> guarded
        in
        let record rendering =
          if !mma_census_enabled then mma_census := (!current_kernel_name, rendering) :: !mma_census
        in
        (* Shape facts for the decline diagnostics: enough to identify the statement and check every
           statically-checkable emission rule by eye. *)
        let describe () =
          let space_str = function
            | `Shared -> "shared"
            | `Device -> "device"
            | `Thread -> "thread"
            | `Fragment _ -> "fragment"
          in
          let layout_str = function
            | `Plain -> ""
            | `Swizzled_b128 -> ",swz128"
            | `Decline reason -> ",unsupported layout: " ^ reason
          in
          let op_str role ld (tn, idcs) =
            let prec, (_, ld, space, layout) = operand ld (tn, idcs) in
            Printf.sprintf "%s=%s:%s[ld=%d,%s%s]" role (Tn.debug_name tn) (Ops.prec_string prec) ld
              (space_str space) (layout_str layout)
          in
          Printf.sprintf "%dx%dx%d ta=%b tb=%b %s %s %s" m n k ta tb (op_str "d" ldd d)
            (op_str "a" lda a) (op_str "b" ldb b)
        in
        let fallback_doc () =
          record Mma_scalar_fallback;
          lane0_guarded (pp_ll ~log_set_locals ~in_loop:true fallback)
        in
        (* Register-tiled CPU rendering (gh-ocannl-469; tinyBLAS/llamafile's [mnpack] and the S4
           micro-kernel shape): the C-tile lives in an RM×RN grid of vector-extension registers
           across the ENTIRE k-loop — per k step: RN B-row vector loads, RM A-element splats, and
           RM×RN fused-FMA updates — loaded from [d] at block entry (the statement's [+=] semantics)
           and stored back at block exit. The geometry — RM rows, RN vector columns, the lane count
           — is the schedule's to carry ([Tile_mma.tile], gh-ocannl-619) and otherwise the
           renderer's [Register_tile.default]: RM = 4; RN up to 3 vector columns on AVX2-class
           16-register files ([vector_bytes = 32]), 6 on NEON/AVX-512-class 32-register files —
           RM×RN + RM + RN live registers, tinyBLAS's budget. Edge tiles are peeled into scalar
           loops, not masked.

           Like the [Vectorized] renderings, the register geometry is keyed off the *compute*
           precision (gh-ocannl-517/575): narrow-storage operands cross the memory boundary through
           [vec_bridge] (vectors) and [convert_precision] (the A splats and the scalar peel), and
           the C-tile accumulates at [comp_prec] across the whole k-loop, narrowing once at the
           store — the same once-per-cell narrowing as [try_vectorize_reduce]'s epilogue. Where
           storage and compute precisions coincide (including pure-fp16 on native-arithmetic
           targets, where [comp_prec] leaves [Half_prec] alone) the bridges are the identity
           memcpys, and for each output element the k-chain runs in serial order with the same fused
           rounding, so the rendering is BITWISE equal to the scalar fallback. A narrow-storage
           rendering instead rounds strictly less often than the fallback's per-k-step narrowing
           (better, never bitwise on inexact values — the gh-ocannl-545 accumulator precedent), so
           parity tests pick narrow-exact inputs. The plain-add (non-FMA) fallback form is declined
           — its [a * b + c] arithmetic is only maybe-contracted, so a vector twin could not promise
           the equality. Emitted under the same lane-0 guard as the fallback ([`Vec_extensions]
           backends render the lane loop serially; GPU backends never take this path). *)
        let try_register_tile () : PPrint.document option =
          (* [cond] names a rule violation: decline (with the per-rule diagnostic, gh-ocannl-479)
             when it holds. *)
          let no_test ~reason cond =
            if cond then (
              declinef "Tile_mma register tiling declined (%s): %s" reason (describe ());
              None)
            else Some ()
          in
          let ( let* ) o f = Option.bind o ~f in
          let* () =
            no_test ~reason:"packed-struct vector style"
              (match B.vector_style with `Vec_extensions -> false | `Packed_struct -> true)
          in
          let* () =
            no_test
              ~reason:(Printf.sprintf "vector_bytes = %d < 8" B.vector_bytes)
              (B.vector_bytes < 8)
          in
          let* () =
            no_test ~reason:"debug_log_from_routines (logged runs stay serial and deterministic)"
              (Utils.debug_log_from_routines ())
          in
          (* A transposed B ([tb]: [b[j*ldb+l]]) would turn the per-k row vector loads into
             stride-[ldb] gathers — decline it (the scalar fallback handles it; a packing [Stage]
             with [tile_loops] in micro-kernel order normalizes the layout instead). A transposed A
             ([ta]: [a[l*lda+i]]) costs nothing — the A feeds are scalar element loads (splats)
             either way — so only the index arithmetic swaps. *)
          let* () = no_test ~reason:"transposed B storage (tb)" tb in
          (* The C-tile pointers stream rows at the raw leading-dimension stride; a swizzled
             operand's elements are not where row-major arithmetic expects them. *)
          let* () =
            no_test ~reason:"swizzled operand layout"
              (List.exists [ fst d; fst a; fst b ] ~f:is_swizzled)
          in
          (* The register-tiled pointers stream rows from the raw base at row-major arithmetic; a
             pipelined operand's live copy rotates per k-block (gh-487). The scalar fallback and the
             intrinsic arms handle it — their accesses carry the rotation term. *)
          let* () =
            no_test ~reason:"pipelined operand layout (rotating buffer copies)"
              (List.exists [ fst d; fst a; fst b ] ~f:is_pipelined)
          in
          let d_tn = fst d in
          let d_store_prec = Lazy.force d_tn.Tn.storage_prec in
          let a_store_prec = Lazy.force (fst a).Tn.storage_prec in
          let b_store_prec = Lazy.force (fst b).Tn.storage_prec in
          let prec = comp_prec d_store_prec in
          (* The C-tile holds the accumulator at [prec] for the whole k extent, so the rendering
             honors the accumulator-width contract only where that IS the residency. Under
             [Fp16_wide] with [narrow_compute_f32 = false], [acc_prec] resolves an f16 destination
             to f32 while [comp_prec] leaves it at half — rendering the tile would accumulate
             narrowly while every serial rendering holds f32, resurrecting the schedule-dependent
             width gh-ocannl-680 removes (Codex P1 round 1 on staging PR #477). Decline: the serial
             fallback then localizes at [acc_prec]. The seeding pre-filter asks the same question
             through [Numerics.cpu_accum_prec], so such a candidate is never timed under a
             tensorized label either. *)
          let* () =
            no_test
              ~reason:
                (Printf.sprintf "accumulator residency %s diverges from compute precision %s"
                   (Ops.prec_string (acc_prec d_store_prec))
                   (Ops.prec_string prec))
              (not (Ops.equal_prec (acc_prec d_store_prec) prec))
          in
          let* () =
            no_test
              ~reason:
                (Printf.sprintf "compute precision %s is not vector-capable here"
                   (Ops.prec_string prec))
              (not (B.vector_prec_ok prec))
          in
          let* () =
            no_test ~reason:"mixed operand compute precisions"
              (not
                 (Ops.equal_prec (comp_prec a_store_prec) prec
                 && Ops.equal_prec (comp_prec b_store_prec) prec))
          in
          (* The fallback carries the arithmetic form; require the fused one (see above). *)
          let rec innermost_set (llc : Low_level.t) =
            match llc with
            | Low_level.For_loop { body; _ } | If { body; _ } -> innermost_set body
            | Seq (x, y) -> (
                match innermost_set x with Some _ as r -> r | None -> innermost_set y)
            | Set { tn; llsc; _ } -> Some (tn, llsc)
            | _ -> None
          in
          let* () =
            no_test ~reason:"accumulation is not in fused (FMA) form"
              (match innermost_set fallback with
              | Some (tn, Low_level.Ternop (Ops.FMA, _, _, (Low_level.Get (tn2, _), _))) ->
                  not (Tn.equal tn d_tn && Tn.equal tn2 d_tn)
              | _ -> true)
          in
          (* The C-tile geometry (gh-ocannl-619): [rm] rows of [rn] vectors of [lanes]. A schedule
             that carries one ([tile]) is honoured exactly or declined — never silently replaced, so
             a candidate the tuner times under a geometry label ran that geometry or the scalar
             fallback (the census says which). Without one, the renderer's own ranking model picks,
             {!Register_tile.default}: chosen against the ACTUAL [n], not fixed at the
             register-pressure cap (gh-ocannl-575) — the columns [bw = rn * lanes] does not cover
             are peeled to the scalar fallback, and a scalar column is roughly a whole vector slot's
             worth of work, so a cap that leaves a fat remainder loses far more than the extra
             A-reuse it buys (on NEON at n = 512 the pure-fp16 [bw = 48] peels 32 of 512 columns and
             runs 3.6x slower than the peel-free [bw = 32]). The same model ranks the lane count
             over the ladder of widths the file renders, stepping down exactly when the narrower
             vector's smaller peel outweighs its extra issues (n = 40: 16 lanes peel 8 where 8 lanes
             divide); the register-pressure cap stays keyed on the MACHINE's width, since stepping
             down does not shrink the register file. The model, its fitted constant and the fit
             rules a request must pass live in [Register_tile], where the sketch seeding consults
             the same functions to propose the alternatives it times. *)
          let elt_bytes = Ops.prec_in_bytes prec in
          let vector_bytes = B.vector_bytes in
          let* { Register_tile.rm; rn; lanes } =
            match tile with
            | Some t -> (
                match Register_tile.check ~vector_bytes ~elt_bytes ~m ~n t with
                | Ok () -> Some t
                | Error why ->
                    declinef "Tile_mma register tiling declined (requested geometry %s: %s): %s"
                      (Register_tile.to_string t) why (describe ());
                    None)
            | None -> (
                match Register_tile.default ~vector_bytes ~elt_bytes ~m ~n with
                | Some t -> Some t
                | None ->
                    declinef
                      "Tile_mma register tiling declined (n = %d below the vector width (lanes = \
                       %d)): %s"
                      n
                      (vector_bytes / max 1 elt_bytes)
                      (describe ());
                    None)
          in
          let bw = rn * lanes in
          let m_full = m - (m % rm) in
          let n_full = n - (n % bw) in
          let vtyp, typedef_doc = vec_ext_typ ~prec ~lanes in
          let ctyp = B.typ_of_prec prec in
          let it = B.loop_index_type in
          (* The fused scalar step at the compute precision. fp16 must go through the same
             [#if]-selected macro as [vec_acc_fma]'s per-lane arm and the scalar rendering, or the
             peel and the vector body would round differently (gh-ocannl-516). *)
          let fma_fn =
            match prec with
            | Ops.Double_prec _ -> "fma"
            | Ops.Half_prec _ -> "OCANNL_HALF_FMA"
            | _ -> "fmaf"
          in
          let _, (d_ptr, ldd, _, _) = operand ldd d in
          let _, (a_ptr, lda, _, _) = operand lda a in
          let _, (b_ptr, ldb, _, _) = operand ldb b in
          (* The A element at (row expression, k expression), honoring [ta]'s storage order. *)
          let a_at ~row ~l =
            if ta then Printf.sprintf "tmma_a__[%s * %d + %s]" l lda row
            else Printf.sprintf "tmma_a__[%s * %d + %s]" row lda l
          in
          (* Scalar memory-boundary conversions (empty when storage = compute): the same
             [convert_precision] spellings the scalar fallback renders through, so the peel's
             arithmetic matches it by construction. *)
          let scal ~from ~to_ expr =
            let pre, post = B.convert_precision ~from ~to_ in
            pre ^ expr ^ post
          in
          let a_elt ~row ~l = scal ~from:a_store_prec ~to_:prec (a_at ~row ~l) in
          let b_elt idx = scal ~from:b_store_prec ~to_:prec (Printf.sprintf "tmma_b__[%s]" idx) in
          (* Vector memory-boundary bridges: identity memcpys when storage = compute, the
             gh-ocannl-517 widen/narrow conversions otherwise. *)
          let extra_typedefs = Hashtbl.create (module String) in
          let need_typedef name doc = Hashtbl.set extra_typedefs ~key:name ~data:doc in
          let fresh pfx = pfx ^ "__" in
          let d_load, d_store =
            vec_bridge ~store_prec:d_store_prec ~prec ~lanes ~vtyp ~need_typedef ~fresh
          in
          let b_load, _ =
            vec_bridge ~store_prec:b_store_prec ~prec ~lanes ~vtyp ~need_typedef ~fresh
          in
          let stmts = separate hardline in
          (* Scalar peel of rows [i_lo, i_hi) × cols [j_lo, j_hi): same fmaf chain per element. *)
          let scalar_peel ~i_lo ~i_hi ~j_lo ~j_hi =
            if i_lo >= i_hi || j_lo >= j_hi then []
            else
              [
                string
                  (Printf.sprintf
                     "for (%stmma_i__ = %d; tmma_i__ < %d; ++tmma_i__) { for (%stmma_j__ = %d; \
                      tmma_j__ < %d; ++tmma_j__) {"
                     it i_lo i_hi it j_lo j_hi)
                ^^ nest 2
                     (hardline
                     ^^ string
                          (Printf.sprintf "%s tmma_acc__ = %s;" ctyp
                             (scal ~from:d_store_prec ~to_:prec
                                (Printf.sprintf "tmma_d__[tmma_i__ * %d + tmma_j__]" ldd)))
                     ^^ hardline
                     ^^ string
                          (Printf.sprintf
                             "for (%stmma_l__ = 0; tmma_l__ < %d; ++tmma_l__) tmma_acc__ = %s(%s, \
                              %s, tmma_acc__);"
                             it k fma_fn
                             (a_elt ~row:"tmma_i__" ~l:"tmma_l__")
                             (b_elt (Printf.sprintf "tmma_l__ * %d + tmma_j__" ldb)))
                     ^^ hardline
                     ^^ string
                          (Printf.sprintf "tmma_d__[tmma_i__ * %d + tmma_j__] = %s;" ldd
                             (scal ~from:prec ~to_:d_store_prec "tmma_acc__")))
                ^^ hardline ^^ string "} }";
              ]
          in
          let grid = vec_acc_grid ~prefix:"tmma_c" ~rows:rm ~cols:rn in
          let d_mem r c =
            string
              (Printf.sprintf "tmma_d__[(tmma_i__ + %d) * %d + tmma_j__ + %d]" r ldd (c * lanes))
          in
          let full_blocks =
            if m_full = 0 || n_full = 0 then []
            else
              let k_body =
                List.init rn ~f:(fun c ->
                    b_load ~dst:(Printf.sprintf "tmma_b_%d__" c)
                      ~mem:
                        (string
                           (Printf.sprintf "tmma_b__[tmma_l__ * %d + tmma_j__ + %d]" ldb (c * lanes))))
                @ List.concat
                    (List.init rm ~f:(fun r ->
                         (string
                            (Printf.sprintf "%s tmma_as_%d__ = %s;" ctyp r
                               (a_elt ~row:(Printf.sprintf "(tmma_i__ + %d)" r) ~l:"tmma_l__"))
                         ^^ hardline
                         ^^ string (Printf.sprintf "%s tmma_a_%d__ = " vtyp r)
                         ^^ vec_splat ~vtyp ~lanes (Printf.sprintf "tmma_as_%d__" r)
                         ^^ semi)
                         :: List.init rn ~f:(fun c ->
                             vec_acc_fma ~prec ~lanes
                               ~dst:grid.(r).(c)
                               ~a:(Printf.sprintf "tmma_a_%d__" r)
                               ~b:(Printf.sprintf "tmma_b_%d__" c))))
              in
              let per_cell f =
                List.concat (List.init rm ~f:(fun r -> List.init rn ~f:(fun c -> f r c)))
              in
              [
                string
                  (Printf.sprintf "for (%stmma_i__ = 0; tmma_i__ + %d <= %d; tmma_i__ += %d) {" it
                     rm m rm)
                ^^ nest 2
                     (hardline
                     ^^ string
                          (Printf.sprintf
                             "for (%stmma_j__ = 0; tmma_j__ + %d <= %d; tmma_j__ += %d) {" it bw n
                             bw)
                     ^^ nest 2
                          (hardline
                          ^^ stmts (per_cell (fun r c -> d_load ~dst:grid.(r).(c) ~mem:(d_mem r c)))
                          ^^ hardline
                          ^^ string
                               (Printf.sprintf "for (%stmma_l__ = 0; tmma_l__ < %d; ++tmma_l__) {"
                                  it k)
                          ^^ nest 2 (hardline ^^ stmts k_body)
                          ^^ hardline ^^ string "}" ^^ hardline
                          ^^ stmts
                               (per_cell (fun r c -> d_store ~src:grid.(r).(c) ~mem:(d_mem r c))))
                     ^^ hardline ^^ string "}")
                ^^ hardline ^^ string "}";
              ]
          in
          let body =
            (typedef_doc :: registered_typedefs extra_typedefs)
            @ [
                (* Operand pointers keep their storage element type -- the bridges and the scalar
                   conversions cross to the compute precision at each access. *)
                string (Printf.sprintf "%s *tmma_d__ = " (B.typ_of_prec d_store_prec))
                ^^ d_ptr ^^ semi;
                string (Printf.sprintf "const %s *tmma_a__ = " (B.typ_of_prec a_store_prec))
                ^^ a_ptr ^^ semi;
                string (Printf.sprintf "const %s *tmma_b__ = " (B.typ_of_prec b_store_prec))
                ^^ b_ptr ^^ semi;
              ]
            @ full_blocks
            @ scalar_peel ~i_lo:0 ~i_hi:m_full ~j_lo:n_full ~j_hi:n
            @ scalar_peel ~i_lo:m_full ~i_hi:m ~j_lo:0 ~j_hi:n
          in
          let narrow_note =
            let note role sp =
              if Ops.equal_prec sp prec then None
              else Some (Printf.sprintf "%s:%s" role (Ops.prec_string sp))
            in
            match
              List.filter_opt
                [ note "d" d_store_prec; note "a" a_store_prec; note "b" b_store_prec ]
            with
            | [] -> ""
            | notes -> "; narrow storage bridged: " ^ String.concat ~sep:" " notes
          in
          let geometry_note =
            (* Which decision produced this geometry, for the artifact readers (the schedule's or
               the renderer's default); absent for the default so pre-gh-619 goldens stand. *)
            if Option.is_some tile then "; geometry from the schedule" else ""
          in
          Some
            (string
               (Printf.sprintf
                  "{ /* Tile_mma register tiling: %dx%d C-tile of %d-lane %s held across the \
                   k-loop (full blocks %dx%d of %dx%d)%s%s. */"
                  rm rn lanes ctyp m_full n_full m n narrow_note geometry_note)
            ^^ nest 2 (hardline ^^ stmts body)
            ^^ hardline ^^ string "}")
        in
        let fallback_or_tiled () =
          match try_register_tile () with
          | Some doc ->
              record Mma_register_tiled;
              lane0_guarded doc
          | None ->
              declinef "Tile_mma renders the lane-0 scalar fallback: %s" (describe ());
              fallback_doc ()
        in
        match B.mma_syntax with
        | None -> fallback_or_tiled ()
        | Some _ when Utils.debug_log_from_routines () ->
            declinef
              "Tile_mma intrinsics skipped (debug_log_from_routines: logged runs stay serial and \
               deterministic): %s"
              (describe ());
            fallback_doc ()
        | Some emit -> (
            let d_prec, d_raw = operand ldd d in
            let a_prec, a_raw = operand lda a in
            let b_prec, b_raw = operand ldb b in
            (* Layouts the intrinsic loads have no form for (the element-granularity swizzle; a
               b128-swizzled tile not addressable from its origin) never reach the hook: the scalar
               fallback reads elementwise through the swizzle-aware offsets and stays correct. A
               [`Swizzled_b128] tile addressed from its origin DOES reach it — the arms that own
               [ldmatrix] consume it, the others decline per call (gh-ocannl-481 item 3, D2). *)
            match List.find_map [ d_raw; a_raw; b_raw ] ~f:operand_decline with
            | Some reason ->
                declinef "Tile_mma intrinsics declined (%s): %s" reason (describe ());
                fallback_or_tiled ()
            | None -> (
                let d_op = Option.value_exn ~here:[%here] (narrow_operand d_raw) in
                let a_op = Option.value_exn ~here:[%here] (narrow_operand a_raw) in
                let b_op = Option.value_exn ~here:[%here] (narrow_operand b_raw) in
                (* Here — and only here — the a/b tile addresses are renderable, so this site both
                   asks the hook and applies the emission it returns. *)
                let a_ptr_doc, a_src = (operand_ptr a_op, operand_source a_op) in
                let b_ptr_doc, b_src = (operand_ptr b_op, operand_source b_op) in
                match emit ~d_prec ~a_prec ~b_prec ~ta ~tb ~m ~n ~k ~d:d_op ~a:a_src ~b:b_src with
                | Some emission ->
                    (* Accepting a swizzled operand is a promise that it was read through a
                       swizzle-aware load; no other reading of that layout is correct. *)
                    let swizzled (_, _, _, layout) =
                      match layout with `Swizzled_b128 -> true | `Plain -> false
                    in
                    record
                      (if List.exists [ d_op; a_op; b_op ] ~f:swizzled then Mma_intrinsics_ldmatrix
                       else Mma_intrinsics);
                    emission ~a_ptr:a_ptr_doc ~b_ptr:b_ptr_doc
                | None ->
                    declinef "Tile_mma intrinsics declined by the backend hook: %s" (describe ());
                    fallback_or_tiled ())))
    | If { cond = c, cprec; body } ->
        (* Guarded statement (axis-types proposal §2): [body] executes iff [cond] is nonzero -- C's
           [if] tests exactly that. A guard controlling a recognized scope-local accumulation is
           part of its accumulating loop: when the guard is its only materialized read, excluding it
           would remove the Metal workaround altogether (gh-ocannl-820, Codex P1 round 3 on #553).
           Keep the context expression-local, and only while already inside a serial loop. *)
        let controlled_scopes =
          if B.volatile_serial_accumulation && not (List.is_empty !serial_loop_stack) then
            controlled_accumulation_scope_ids body
          else []
        in
        let (local_defs, cond_doc), _ =
          with_volatile_accumulation_reads ~scope_ids:controlled_scopes
            (not (List.is_empty controlled_scopes))
            (fun () -> pp_scalar (comp_prec cprec) c)
        in
        let local_defs = pp_local_defs local_defs in
        let body_doc =
          match !current_localized_zero_seed with
          | Some seed when not seed.lzs_consumed ->
              (* An [If] reached before localization conditionally guards the closing store, so it
                 cannot consume a whole-node zero. Guards *inside* a recognized reduction are
                 rebuilt only after the enclosing loop has already consumed the seed. *)
              let saved = !current_localized_zero_seed in
              current_localized_zero_seed := None;
              Exn.protect
                ~f:(fun () -> pp_ll ~log_set_locals ~in_loop:true body)
                ~finally:(fun () -> current_localized_zero_seed := saved)
          | None | Some _ -> pp_ll ~log_set_locals ~in_loop:true body
        in
        let if_doc =
          group
            (string "if (" ^^ cond_doc ^^ string ") " ^^ lbrace
            ^^ nest 2 (hardline ^^ body_doc)
            ^^ hardline ^^ rbrace)
        in
        if PPrint.is_empty local_defs then if_doc
        else lbrace ^^ nest 2 (hardline ^^ local_defs ^^ hardline ^^ if_doc) ^^ hardline ^^ rbrace

  and pp_scalar (prec : Ops.prec) (vcomp : Low_level.scalar_t) :
      (int * PPrint.document) list * PPrint.document =
    (* Returns (local definitions, value expression) *)
    let open PPrint in
    match vcomp with
    | Local_scope
        { id = { tn = { storage_prec = _; _ }; scope_id } as id; body; orig_indices = _; mint = _ }
      ->
        record_accumulation_scope id;
        let scope_prec = scope_prec_of id in
        let num_typ = string (scope_decl_type id) in
        let init_zero =
          if Low_level.reads_scope_before_set id body then
            let prefix, postfix = B.convert_precision ~from:Ops.int32 ~to_:scope_prec in
            string " = " ^^ string prefix ^^ string "0" ^^ string postfix
          else empty
        in
        let decl = num_typ ^^ space ^^ pp_scope_id id ^^ init_zero ^^ semi in
        (* A [Local_scope] body is a nested sub-computation; conservatively treat it like a loop
           body so a [Zero_out] reached through it is never mistaken for the function-scope
           first-touch that the declaration's [= {0}] covers. *)
        let body_doc = pp_ll ~log_set_locals:false ~in_loop:true body in
        let def_doc = decl ^^ hardline ^^ body_doc in
        let prefix, postfix = B.convert_precision ~from:scope_prec ~to_:prec in
        let expr = string prefix ^^ pp_scope_id id ^^ string postfix in
        ([ (scope_id, def_doc) ], expr)
    | Get_local id ->
        let scope_prec = scope_prec_of id in
        let prefix, postfix = B.convert_precision ~from:scope_prec ~to_:prec in
        let expr = string prefix ^^ pp_scope_id id ^^ string postfix in
        ([], expr)
    | Get_merge_buffer (source, idcs) ->
        let tn = source in
        let dims = Lazy.force tn.dims in
        let from_prec = Lazy.force tn.storage_prec in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        let offset_doc = pp_array_offset (idcs, dims) in
        let ptr_doc =
          if !volatile_accumulation_reads then begin
            record_volatile_read ();
            parens
              (string
                 ("(" ^ B.buffer_prefix ^ "volatile " ^ B.typ_of_prec from_prec ^ "*)merge_buffer"))
          end
          else string "merge_buffer"
        in
        let expr = string prefix ^^ ptr_doc ^^ brackets offset_doc ^^ string postfix in
        ([], expr)
    | Get (tn, idcs) ->
        let ident_doc = string (get_ident tn) in
        let dims = Lazy.force tn.dims in
        let from_prec = Lazy.force tn.storage_prec in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        let offset_doc = pp_pipelined_rotation ~is_write:false tn ^^ pp_tn_offset tn (idcs, dims) in
        let expr =
          string prefix ^^ pp_device_read_ptr tn ident_doc ^^ brackets offset_doc ^^ string postfix
        in
        ([], expr)
    | Get_dynamic { tn; idcs; dyn_axis; dyn_value = iv, iprec } ->
        (* gh-343: a guarded dynamic gather. The dynamic index is spliced into the row-major offset
           at [dyn_axis] as an integer; the enclosing [Where] guard (when interval analysis has not
           discharged it against proven bounds) guarantees it is in range before this load is
           evaluated (C ternary short-circuits). Cast to [Ops.index_prec ()] so the index tracks the
           same width as loop counters (signed int32 normally, int64 under large_models), preventing
           truncation for very large table/vocabulary axes. *)
        if is_swizzled tn || is_pipelined tn then
          invalid_arg
            ("C_syntax: Get_dynamic reads swizzled or pipelined node " ^ Tn.debug_name tn
           ^ " (dynamic offsets are not swizzle-remapped)");
        let ident_doc = string (get_ident tn) in
        let dims = Lazy.force tn.dims in
        let from_prec = Lazy.force tn.storage_prec in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        let dyn_defs, dyn_expr = pp_scalar iprec iv in
        let idx_typ = B.typ_of_prec (Ops.index_prec ()) in
        let dyn_idx_doc = string ("((" ^ idx_typ ^ ")(") ^^ dyn_expr ^^ string "))" in
        let offset_doc = pp_array_offset_dyn (idcs, dims) ~dyn_axis ~dyn_idx_doc in
        let expr =
          string prefix ^^ pp_device_read_ptr tn ident_doc ^^ brackets offset_doc ^^ string postfix
        in
        (dyn_defs, expr)
    | Constant c ->
        let from_prec = Ops.double in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        let c_str = c_float_literal c in
        let expr =
          (* The sign BIT, not [c < 0.]: an unparenthesized [-0.0] beside a subtraction would render
             as [--0.0]. *)
          if String.is_empty prefix && Float.ieee_negative c && not Float.(c = neg_infinity) then
            string "(" ^^ string c_str ^^ string ")" ^^ string postfix
          else string prefix ^^ string c_str ^^ string postfix
        in
        ([], expr)
    | Constant_bits i ->
        let from_prec = Ops.int64 in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        let expr = string prefix ^^ string (Printf.sprintf "%LdLL" i) ^^ string postfix in
        ([], expr)
    | Embed_index idx ->
        let from_prec = Ops.index_prec () in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        let idx_doc = pp_axis_index idx in
        let idx_doc = if PPrint.is_empty idx_doc then string "0" else idx_doc in
        (* A multi-term affine renders as an unparenthesized sum; parenthesize so embedding into an
           operator context (e.g. the [/] and [%] of the gh-509 lane-extract form) cannot rebind
           terms via C precedence. *)
        let idx_doc = if affine_needs_parens idx then parens idx_doc else idx_doc in
        let expr = string prefix ^^ idx_doc ^^ string postfix in
        ([], expr)
    | Ternop (Ops.FMA, _, _, _) when not (Ops.is_float prec) ->
        (* gh-ocannl-873: hand-built low-level IR can bypass the simplifier's float-only FMA rewrite
           guard. Reject it at the shared C-family scalar-emission seam before a backend's syntax
           table can turn integer arithmetic into [fma]/[fmaf]. The explicit SIMD and MMA renderings
           are already gated by [vector_prec_ok], which admits only float-capable compute
           precisions. *)
        invalid_arg
          (Printf.sprintf "C_syntax.pp_scalar: FMA requires floating-point precision, got %s"
             (Ops.prec_string prec))
    | Ternop (op, (v1, v1_prec), (v2, v2_prec), (v3, v3_prec)) ->
        (* A heterogeneous argument keeps its own precision, which -- like every precision reaching
           here from [Low_level] -- is a storage precision; the rendering takes [comp_prec] of
           it. *)
        let v1_prec = comp_prec v1_prec and v2_prec = comp_prec v2_prec in
        let v3_prec = comp_prec v3_prec in
        let d1, e1, d2, e2, d3, e3 =
          if Ops.is_homogeneous_prec_ternop op then
            (* Homogeneous: all arguments use result precision *)
            let d1, e1 = pp_scalar prec v1 in
            let d2, e2 = pp_scalar prec v2 in
            let d3, e3 = pp_scalar prec v3 in
            (d1, e1, d2, e2, d3, e3)
          else
            (* Heterogeneous: arguments keep their natural precision *)
            match op with
            | Ops.Where ->
                (* For Where: condition keeps its precision, then/else use result precision *)
                (* Note: we evaluate condition without precision conversion, but then/else
                   need to match the result precision for the final assignment *)
                let d1, e1 = pp_scalar v1_prec v1 in
                (* condition: no conversion *)
                let d2, e2 = pp_scalar prec v2 in
                (* then: result precision *)
                let d3, e3 = pp_scalar prec v3 in
                (* else: result precision *)
                (d1, e1, d2, e2, d3, e3)
            | _ ->
                (* Other heterogeneous ternary ops would go here *)
                let d1, e1 = pp_scalar v1_prec v1 in
                let d2, e2 = pp_scalar v2_prec v2 in
                let d3, e3 = pp_scalar v3_prec v3 in
                (d1, e1, d2, e2, d3, e3)
        in
        let defs = List.concat [ d1; d2; d3 ] in
        let expr = group (B.ternop_syntax prec op e1 e2 e3) in
        (defs, expr)
    | Binop (op, (v1, v1_prec), (v2, v2_prec)) -> (
        match Ops.binop_conditionality op with
        (* A projection emits its selected operand alone: the operator has no spelling of its own
           (every backend's [binop_syntax] rejects one), and the discarded operand is not rendered
           at all -- which is what lets [Low_level.affine_accesses] and [Cost_model] drop it. *)
        | Ops.Only_first -> pp_scalar prec v1
        | Ops.Only_second -> pp_scalar prec v2
        | Ops.Both_operands | Ops.Gated_second ->
            let v1_prec = comp_prec v1_prec and v2_prec = comp_prec v2_prec in
            let d1, e1, d2, e2 =
              if Ops.is_homogeneous_prec_binop op then
                (* Homogeneous: both arguments use result precision *)
                let d1, e1 = pp_scalar prec v1 in
                let d2, e2 = pp_scalar prec v2 in
                (d1, e1, d2, e2)
              else
                (* Heterogeneous: arguments keep their natural precision *)
                (* Currently all binops are homogeneous, but this is here for future extension *)
                let d1, e1 = pp_scalar v1_prec v1 in
                let d2, e2 = pp_scalar v2_prec v2 in
                (d1, e1, d2, e2)
            in
            let defs = List.concat [ d1; d2 ] in
            let expr = group (B.binop_syntax prec op e1 e2) in
            (defs, expr))
    | Unop (op, (v, v_prec)) ->
        let arg_prec =
          if Ops.is_homogeneous_prec_unop op then prec
            (* Homogeneous: argument uses result precision *)
          else comp_prec v_prec
        in
        let defs, expr_v = pp_scalar arg_prec v in
        let expr = group (B.unop_syntax prec op expr_v) in
        (defs, expr)

  and debug_float ?guard (prec : Ops.prec) (value : Low_level.scalar_t) :
      PPrint.document
      * [ `Accessor of Indexing.axis_index array * int array | `Value of PPrint.document ] list =
    (* Returns (value expression doc, list of arguments for printf).

       [guard] (task-9658aac9): when [Some cond_c] (a real C boolean expression), any array
       dereference produced here is only safe when [cond_c] holds, so its printf [`Value] argument
       is short-circuited as [(cond_c ? read : 0)]. This mirrors the runtime [Where] ternary's
       short-circuiting and closes the only debug-logging path that would otherwise dereference a
       conditionally-evaluated branch (e.g. the Stage B unit-solve then-branch producer read) out of
       bounds -- the same hazard gh-343 fixed for the sibling [Get_dynamic] gather. Only the
       dereferencing argument is gated; the displayed expression doc is unchanged. *)
    let open PPrint in
    let guarded_value access_doc =
      match guard with
      | None -> access_doc
      | Some g -> parens (g ^^ string " ? " ^^ access_doc ^^ string " : 0")
    in
    match value with
    | Local_scope { id; _ } ->
        (* Not printing the inlined definition: (1) code complexity; (2) don't overload the debug
           logs. *)
        debug_float prec @@ Get_local id
    | Get_local id ->
        let scope_prec = scope_prec_of id in
        let prefix, postfix = B.convert_precision ~from:scope_prec ~to_:prec in
        let v_doc = string prefix ^^ pp_scope_id id ^^ string postfix in
        (v_doc ^^ braces (string ("=" ^ B.float_log_style)), [ `Value v_doc ])
    | Get_merge_buffer (source, idcs) ->
        let tn = source in
        let dims = Lazy.force tn.dims in
        let from_prec = Lazy.force tn.storage_prec in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        let offset_doc = pp_array_offset (idcs, dims) in
        let access_doc =
          string prefix ^^ string "merge_buffer" ^^ brackets offset_doc ^^ string postfix
        in
        let expr_doc =
          string prefix ^^ string "merge_buffer"
          ^^ brackets (string "%u")
          ^^ string postfix
          ^^ braces (string ("=" ^ B.float_log_style))
        in
        (expr_doc, [ `Accessor (idcs, dims); `Value (guarded_value access_doc) ])
    | Get (tn, idcs) ->
        let ident_doc = string (get_ident tn) in
        let dims = Lazy.force tn.dims in
        let from_prec = Lazy.force tn.storage_prec in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        let offset_doc = pp_pipelined_rotation ~is_write:false tn ^^ pp_tn_offset tn (idcs, dims) in
        let access_doc = string prefix ^^ ident_doc ^^ brackets offset_doc ^^ string postfix in
        let expr_doc =
          string prefix ^^ ident_doc
          ^^ brackets (string "%u")
          ^^ string postfix
          ^^ braces (string ("=" ^ B.float_log_style))
        in
        (expr_doc, [ `Accessor (idcs, dims); `Value (guarded_value access_doc) ])
    | Get_dynamic { tn; dyn_value = iv, iprec; _ } ->
        (* gh-343: do NOT dereference the table in debug logs. A [Where]'s [debug_float] collects
           all three branch values as printf arguments evaluated unconditionally, so returning the
           raw [table[((idx_typ)(idx))]] access here would read out of bounds for ids the
           surrounding guard is meant to exclude. Log the (always-safe) dynamic index value
           instead. *)
        let prefix, postfix = B.convert_precision ~from:iprec ~to_:prec in
        let _defs, idx_e = pp_scalar iprec iv in
        let idx_doc = string prefix ^^ idx_e ^^ string postfix in
        let label = string (get_ident tn ^ "@dyn_idx") in
        (label ^^ braces (string ("=" ^ B.float_log_style)), [ `Value idx_doc ])
    | Constant c ->
        let from_prec = Ops.double in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        (string prefix ^^ string (c_float_literal c) ^^ string postfix, [])
    | Constant_bits i ->
        let from_prec = Ops.int64 in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        let expr = string prefix ^^ string (Printf.sprintf "%LdLL" i) ^^ string postfix in
        (expr, [])
    | Embed_index idx ->
        let idx_doc = pp_axis_index idx in
        let idx_doc = if PPrint.is_empty idx_doc then string "0" else idx_doc in
        (* Parenthesize multi-term affines, mirroring [pp_scalar]. *)
        ((if affine_needs_parens idx then parens idx_doc else idx_doc), [])
    | Ternop (op, (v1, v1_prec), (v2, v2_prec), (v3, v3_prec)) ->
        let v1_prec = comp_prec v1_prec and v2_prec = comp_prec v2_prec in
        let v3_prec = comp_prec v3_prec in
        let v1_doc, idcs1, v2_doc, idcs2, v3_doc, idcs3 =
          if Ops.is_homogeneous_prec_ternop op then
            (* Homogeneous: all arguments use result precision *)
            let v1_doc, idcs1 = debug_float ?guard prec v1 in
            let v2_doc, idcs2 = debug_float ?guard prec v2 in
            let v3_doc, idcs3 = debug_float ?guard prec v3 in
            (v1_doc, idcs1, v2_doc, idcs2, v3_doc, idcs3)
          else
            (* Heterogeneous: handle based on operation *)
            match op with
            | Ops.Where ->
                (* task-9658aac9: short-circuit array-read leaves on the live [Where] condition so
                   debug value-logging never dereferences a conditionally-evaluated branch out of
                   bounds (the Stage B unit-solve then-branch producer read; gh-343's hazard class
                   for the sibling [Get_dynamic]). The displayed ternary is unchanged -- only the
                   dereferencing printf arguments are gated: then-reads under [cond], else-reads
                   under [!cond], each AND-composed with any enclosing [guard] so nested/triangular
                   range guards compose. [cond_c] is rendered twice -- via [pp_scalar] for the real
                   C guard expression here, and via [debug_float] for the annotated display
                   below. *)
                let _cond_defs, cond_c = pp_scalar v1_prec v1 in
                (* Conditions reaching a Where here are pure index/value comparisons (range guards,
                   in_range) with no Local_scope, so [_cond_defs] is empty. Even if a future
                   condition inlined a Local_scope, dropping the redundant re-definition is safe:
                   the local is already bound by the assignment computation that precedes the log
                   statement (the same invariant debug_float's Local_scope arm relies on). *)
                let compose outer c =
                  match outer with None -> c | Some g -> parens (g ^^ string " && " ^^ c)
                in
                let then_guard = compose guard cond_c in
                let else_guard = compose guard (parens (string "!" ^^ parens cond_c)) in
                let v1_doc, idcs1 = debug_float ?guard v1_prec v1 in
                (* condition: no precision conversion. It is evaluated whenever the enclosing branch
                   is reached, so its array reads (a nested [Where] condition may contain a [Get])
                   are gated by the incoming [guard] -- not by [cond_c], which the condition itself
                   computes. At the top level [guard = None], so this is a no-op for the common
                   pure-index-comparison case. *)
                let v2_doc, idcs2 = debug_float ~guard:then_guard prec v2 in
                (* then: result precision, gated by [cond] *)
                let v3_doc, idcs3 = debug_float ~guard:else_guard prec v3 in
                (* else: result precision, gated by [!cond] *)
                (v1_doc, idcs1, v2_doc, idcs2, v3_doc, idcs3)
            | _ ->
                let v1_doc, idcs1 = debug_float ?guard v1_prec v1 in
                let v2_doc, idcs2 = debug_float ?guard v2_prec v2 in
                let v3_doc, idcs3 = debug_float ?guard v3_prec v3 in
                (v1_doc, idcs1, v2_doc, idcs2, v3_doc, idcs3)
        in
        (B.ternop_syntax prec op v1_doc v2_doc v3_doc, idcs1 @ idcs2 @ idcs3)
    | Binop (op, (v1, v1_prec), (v2, v2_prec)) -> (
        match Ops.binop_conditionality op with
        (* A projection displays its selected operand alone, exactly as [pp_scalar] emits it. *)
        | Ops.Only_first -> debug_float ?guard prec v1
        | Ops.Only_second -> debug_float ?guard prec v2
        | Ops.Both_operands | Ops.Gated_second ->
            let v1_prec = comp_prec v1_prec and v2_prec = comp_prec v2_prec in
            let v1_doc, idcs1, v2_doc, idcs2 =
              if Ops.is_homogeneous_prec_binop op then
                (* Homogeneous: both arguments use result precision *)
                let v1_doc, idcs1 = debug_float ?guard prec v1 in
                let v2_doc, idcs2 = debug_float ?guard prec v2 in
                (v1_doc, idcs1, v2_doc, idcs2)
              else
                (* Heterogeneous: arguments keep their natural precision *)
                let v1_doc, idcs1 = debug_float ?guard v1_prec v1 in
                let v2_doc, idcs2 = debug_float ?guard v2_prec v2 in
                (v1_doc, idcs1, v2_doc, idcs2)
            in
            (B.binop_syntax prec op v1_doc v2_doc, idcs1 @ idcs2))
    | Unop (op, (v, v_prec)) ->
        let arg_prec =
          if Ops.is_homogeneous_prec_unop op then prec
            (* Homogeneous: argument uses result precision *)
          else comp_prec v_prec
        in
        let v_doc, idcs = debug_float ?guard arg_prec v in
        (B.unop_syntax prec op v_doc, idcs)

  let compile_main llc : PPrint.document = pp_ll llc

  let compile_proc ~name idx_params
      Low_level.
        {
          traced_store;
          llc;
          merge_node;
          optimize_ctx;
          workgroup_shared;
          simdgroup_fragments;
          swizzled;
          pipelined;
          zero_fringe = _;
          flip_candidates = _;
          spliced_rbw = _;
        } : (string * kparam_source) list * PPrint.document * Low_level.launch_dims =
    let open PPrint in
    (if not (Set.is_empty workgroup_shared) then
       match B.shared_decl_prefix with
       | Some _ -> ()
       | None ->
           (* The local-declaration pass below would silently emit workgroup-shared nodes as
              per-thread stack arrays -- wrong sharing semantics. *)
           invalid_arg
             "C_syntax.compile_proc: workgroup-shared placement not supported by this backend");
    (* gh-ocannl-686: the routine name reaches the emitted [void <name>(] verbatim, so it must
       already be a legal, non-reserved identifier. Mangling it HERE would leave the caller's symbol
       lookup and artifact names pointing at the pre-mangling spelling, so the caller owns the
       normalization ({!kernel_ident}, applied once at each backend's [compile] entry) and this is
       the check that it happened -- a backend that skips it fails here, naming the routine, instead
       of handing its compiler a source it rejects as an OCANNL bug. *)
    if not (String.equal name (kernel_ident name)) then
      invalid_arg
        (Printf.sprintf
           "C_syntax.compile_proc: routine name %S is not a legal C identifier for this backend \
            (reserved word, builtin, or illegal character) -- the backend must pass it through \
            C_syntax.kernel_ident first, which would give %S"
           name (kernel_ident name));
    current_kernel_name := name;
    (* The volatility census's [requested] field (gh-ocannl-782): the capability is a per-backend
       source constant, and the census is collected around a compile that does not know which
       backend ran it. Set here, where the backend is the functor's own. *)
    volatility_requested := B.volatile_serial_accumulation;
    current_placements := Some optimize_ctx.Low_level.placements;
    volatile_accumulation_reads := false;
    volatile_accumulation_scope_stack := [];
    volatile_read_observers := [];
    Hash_set.clear volatile_accumulation_scopes;
    Hash_set.clear rendered_accumulation_scopes;
    volatility_events := [];
    (* gh-ocannl-584: scope purity, the pipeline's EXIT gate ([Low_level.optimize_proc] is the entry
       one) — this catches what a schedule transform constructs, which the entry gate cannot see.
       Checked ahead of the schedule validation and NOT transported as an [Illegal_schedule]: an
       impure scope body is malformed IR that no schedule choice can rescue, so declining candidates
       around it would only hide the defect. *)
    Low_level.validate_scope_bodies llc;
    Low_level.validate_scan_loops optimize_ctx.Low_level.placements llc;
    Low_level.validate_parallel_classified optimize_ctx.Low_level.placements llc;
    (* Launch-extent guards (construct-then-fold, axis-types proposal §2), only for kinds this
       backend binds in hardware -- the serial fallback iterates the true extent. *)
    let llc =
      Low_level.guard_annotated_extents
        ~should_guard:(fun kind -> Option.is_some (B.hardware_index ~kind ~slot:0))
        llc
    in
    let launch = Low_level.launch_dims llc in
    current_hardware_axes := Low_level.hardware_axes llc;
    current_loop_bounds := Low_level.loop_bounds llc;
    (let parallel_grid, grid_private, local_ptr_alias = collect_parallel_grid llc in
     current_parallel_grid := parallel_grid;
     current_grid_private := grid_private;
     current_local_ptr_alias := local_ptr_alias);
    current_workgroup_shared := workgroup_shared;
    current_simdgroup_fragments := simdgroup_fragments;
    current_swizzled := swizzled;
    current_pipelined := pipelined;
    current_localized_zero_seed := None;
    (* gh-487 phase 2: which pipelined tiles stage asynchronously — backend hook present, no kernel
       logging (logged [Set]s read the written value back, which an in-flight copy cannot provide),
       and an element size the hardware copies at the alignment plain shared declarations guarantee
       (4/8 bytes: element-type alignment; sub-4-byte tiles keep the portable form, and 16-byte
       elements are excluded until a rendering guarantees 16-byte destination alignment — see
       {!type-async_copy_syntax}). Per-tile, not per-statement: the rotor loop's wait+barrier prefix
       keys on the same set, so a tile with only ineligible statements merely pays a redundant
       wait. *)
    (current_async_tiles :=
       match B.async_copy with
       | Some _ when not (Utils.debug_log_from_routines ()) ->
           Map.keys pipelined
           |> List.filter ~f:(fun tn ->
               match Ops.prec_in_bytes (Lazy.force tn.Tn.storage_prec) with
               | 4 | 8 -> true
               | _ -> false)
           |> Set.of_list (module Tn)
       | _ -> Set.empty (module Tn));
    (* gh-487 sanity: the rotation renders off the rotor loop's serial counter; a schedule that
       later retyped the rotor to a hardware axis would otherwise silently freeze the buffer
       selection at copy 0 ([serial_loop_stack] only tracks serial loops). Typed for the same reason
       as the read-position check in [pp_pipelined_rotation]: both say "this renderer cannot express
       that pipelined tile", and a candidate composing its way into one is a decline, not a fatal
       that ends the search around it. *)
    (if not (Map.is_empty pipelined) then
       let check_rotor index axis =
         Map.iteri pipelined ~f:(fun ~key:tn ~data:{ Low_level.pt_rotor; _ } ->
             if Indexing.equal_symbol index pt_rotor then
               match axis with
               | Low_level.Serial -> ()
               | _ ->
                   raise
                     (Schedule_outcome.Cause_at
                        ( Schedule_outcome.Backend_codegen,
                          Schedule_outcome.Unsupported
                            {
                              feature = "pipelined tile whose rotor loop is not Serial";
                              detail =
                                "C_syntax.compile_proc: the rotor loop of pipelined tile "
                                ^ Tn.debug_name tn ^ " is no longer Serial";
                            } )))
       in
       let rec scan (llc : Low_level.t) =
         match llc with
         | For_loop { index; axis; body; _ } ->
             check_rotor index axis;
             scan body
         | Seq (a, b) ->
             scan a;
             scan b
         | If { body; _ } -> scan body
         | _ -> ()
       in
       scan llc);
    rendered_simdgroup_fragments := Set.empty (module Tn);
    current_traced_store := Some traced_store;
    Hash_set.clear zero_out_seen;
    (* The materialized in-context nodes, in deterministic [traced_store] order, with their
       per-param pointer declaration (used by the [`Per_param] style). *)
    let ptr_params : (string * Tn.t) list =
      List.rev
      @@ Hashtbl.fold traced_store ~init:[] ~f:(fun ~key:tn ~data:_ acc ->
          let backend_info, is_param =
            let plc = placements () in
            if Tn.Placements.is_virtual_force plc tn 334 then ("Virt", false)
            else if in_ctx tn then ("Ctx", true)
            else if Tn.Placements.is_materialized_force plc tn 335 then ("Global", true)
            else if Tn.Placements.known_not_materialized plc tn then ("Local", false)
            else assert false
          in
          let backend_info = Sexp.Atom backend_info in
          if not @@ Utils.sexp_mem ~elem:backend_info tn.backend_info then
            tn.backend_info <- Utils.sexp_append ~elem:backend_info tn.backend_info;
          if is_param then (
            (* Assignments lowering rewrites every alias-view access to a parent access, so alias
               tnodes never reach this parameter list — but hand-built [Low_level.t] (schedule
               layer, tests) could mint one, and with [restrict_keyword] an aliased parameter pair
               is a miscompile rather than a redundant pointer. Fail loudly (gh-ocannl-164). *)
            if Tn.is_alias tn then
              invalid_arg
                ("C_syntax.compile_proc: alias view " ^ Tn.debug_name tn
               ^ " as a kernel parameter: accesses must be rewritten to the buffer-owning parent \
                  (aliased parameters would falsify the restrict qualifier)");
            (* gh-ocannl-489: an aliasing candidate may be placed at bytes overlapping another
               parameter's by the link-time liveness planner, so its pointer must not promise
               [restrict] -- whether a candidate pair actually shares bytes is unknowable at codegen
               time. *)
            let restrict_ =
              match B.restrict_keyword with
              | Some kw when not (Hash_set.mem optimize_ctx.Low_level.alias_candidates tn) ->
                  kw ^ " "
              | _ -> ""
            in
            (B.typ_of_prec (Lazy.force tn.Tn.storage_prec) ^ " *" ^ restrict_ ^ get_ident tn, tn)
            :: acc)
          else acc)
    in
    (* [`Per_param]: one typed pointer param per node (C/CUDA, byte-identical to before). [`Pooled
       n]: [n] byte-pointer pool params + one slot table (Metal binding fix). *)
    let kparams : (string * kparam_source) list =
      match B.ptr_param_style with
      | `Per_param -> List.map ptr_params ~f:(fun (decl, tn) -> (decl, Kparam_ptr tn))
      | `Pooled n_pools -> (
          match ptr_params with
          | [] -> []
          | _ ->
              let slabs =
                List.init n_pools ~f:(fun i ->
                    (Printf.sprintf "char* __pool%d" i, Kparam_pool_slab i))
              in
              let slots =
                ( Printf.sprintf "const %s* __pool_slots" (pool_slot_msl_typ ()),
                  Kparam_pool_slots (List.map ptr_params ~f:snd) )
              in
              slabs @ [ slots ])
    in
    let idx_params =
      List.map idx_params ~f:(fun s ->
          (B.arg_int_prefix ^ Indexing.symbol_ident s.Indexing.static_symbol, Static_idx s))
    in
    let log_file_param =
      if Utils.debug_log_from_routines () then
        match B.kernel_log_param with
        | Some (typ, name) -> [ (typ ^ " " ^ name, Log_file_name) ]
        | None -> []
      else []
    in
    let merge_param =
      Option.(
        to_list
        @@ map merge_node ~f:(fun tn ->
            ("const " ^ B.typ_of_prec (Lazy.force tn.storage_prec) ^ " *merge_buffer", Merge_buffer)))
    in
    let all_params = log_file_param @ merge_param @ idx_params @ kparams in
    let sorted_params =
      List.sort all_params ~compare:(fun (p1_name, _) (p2_name, _) ->
          compare_string p1_name p2_name)
    in
    let args_docs =
      List.mapi sorted_params ~f:(fun pos (name, _) ->
          string (B.buffer_prefix ^ name ^ B.buffer_suffix ~pos))
      @ List.map B.extra_args ~f:string
    in
    let func_header =
      string B.main_kernel_prefix ^^ space ^^ string "void" ^^ space ^^ string name
      ^^ nest 4 (lparen ^^ hardline ^^ separate (comma ^^ hardline) args_docs ^^ rparen)
    in
    let body = ref empty in
    if not (String.is_empty B.kernel_prep_line) then
      body := !body ^^ string B.kernel_prep_line ^^ semi ^^ hardline;

    if Utils.debug_log_from_routines () && B.log_involves_file_management then
      let log_file_var_name =
        match B.kernel_log_param with
        | Some (_, name) -> name
        | None -> "log_file_name" (* Should ideally not be reached if management is true *)
      in
      body :=
        !body ^^ string "FILE* log_file = NULL;" ^^ hardline
        ^^ string ("if (" ^ log_file_var_name ^ ") ")
        ^^ lbrace
        ^^ nest 2 (hardline ^^ string ("log_file = fopen(" ^ log_file_var_name ^ ", \"w\");"))
        ^^ hardline ^^ rbrace ^^ hardline
    else body := !body ^^ hardline;

    (if Utils.debug_log_from_routines () then
       let debug_init_doc =
         string "/* Debug initial parameter state. */"
         ^^ hardline
         ^^ separate_map hardline
              (fun (p_name_and_type, source) ->
                let log_param_doc =
                  Option.map B.kernel_log_param ~f:(fun (_, name) -> string name)
                in
                match source with
                | Merge_buffer ->
                    let merge_tn = Option.value_exn ~here:[%here] merge_node in
                    let base_msg =
                      Printf.sprintf "%s &[%d] = %%p\n" p_name_and_type (Tnode.num_elems merge_tn)
                    in
                    B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
                      ~base_message_literal:base_msg
                      ~args_docs:[ string @@ "(" ^ B.buffer_prefix ^ "void*)merge_buffer" ]
                | Log_file_name -> empty (* Already handled by fopen or if it's just an ID *)
                | Kparam_ptr tn ->
                    let base_msg =
                      Printf.sprintf "%s &[%d] = %%p\n" p_name_and_type (Tnode.num_elems tn)
                    in
                    let ident_doc = string (get_ident tn) in
                    B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
                      ~base_message_literal:base_msg
                      ~args_docs:[ string ("(" ^ B.buffer_prefix ^ "void*)") ^^ ident_doc ]
                | Static_idx s ->
                    let base_msg = Printf.sprintf "%s = %%d\n" p_name_and_type in
                    let ident_doc = pp_symbol s.static_symbol in
                    B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
                      ~base_message_literal:base_msg ~args_docs:[ ident_doc ]
                | Kparam_pool_slab _ | Kparam_pool_slots _ ->
                    (* Pooled (Metal): the per-node pointers are materialized as locals in the pool
                       prologue and logged at their first [Set]; the pool bases / slot table
                       themselves are not separately logged here. *)
                    empty)
              sorted_params
       in
       body := !body ^^ debug_init_doc ^^ hardline);

    (* Pooled (Metal) prologue: build the local pool-base array from the bound pool params, then
       form each materialized node's typed pointer from its slot. Indices match the
       [Kparam_pool_slots] order (= [ptr_params]). The rest of the body indexes [get_ident tn]
       exactly as in the per-param style. *)
    (match B.ptr_param_style with
    | `Per_param -> ()
    | `Pooled n_pools when not (List.is_empty ptr_params) ->
        let pools_decl =
          Printf.sprintf "%schar* __pools[%d] = { %s };" B.buffer_prefix n_pools
            (String.concat ~sep:", " (List.init n_pools ~f:(Printf.sprintf "__pool%d")))
        in
        let defs =
          (* The derived per-node pointers address disjoint slab sub-ranges, and the kernel body
             accesses nodes only through them (never through the pool bases), which is all the
             restrict qualifier asserts (gh-ocannl-164). The liveness planner (gh-ocannl-489)
             preserves the disjointness for pooled backends: they use segment-granularity liveness,
             under which any two nodes accessed by ONE kernel have overlapping live spans and are
             therefore never placed at overlapping offsets. *)
          let restrict_ = match B.restrict_keyword with Some kw -> kw ^ " " | None -> "" in
          List.mapi ptr_params ~f:(fun k (_decl, tn) ->
              let typ = B.typ_of_prec (Lazy.force tn.Tn.storage_prec) in
              Printf.sprintf "%s%s* %s%s = (%s%s*)(__pools[__pool_slots[%d]] + __pool_slots[%d]);"
                B.buffer_prefix typ restrict_ (get_ident tn) B.buffer_prefix typ (2 * k)
                ((2 * k) + 1))
        in
        body :=
          !body
          ^^ string "/* Pool base pointers. */"
          ^^ hardline
          ^^ separate_map hardline string (pools_decl :: defs)
          ^^ hardline
    | `Pooled _ -> ());

    (* Render before declarations so accepted marked-fragment regions can suppress their otherwise
       dead scalar local arrays. Declining/debug paths leave the set empty and keep the arrays. *)
    let main_logic_doc = compile_main llc in
    if !volatility_census_enabled then
      volatility_census :=
        List.map !volatility_events ~f:(function
          | Accumulation id ->
              ( !current_kernel_name,
                if Hash_set.mem volatile_accumulation_scopes id.scope_id then
                  Volatile_accumulation_reads (scope_local_ident id)
                else Plain_accumulator (scope_local_ident id) )
          | Site site -> (!current_kernel_name, site))
        @ !volatility_census;
    let grid_privatized =
      Map.fold !current_grid_private
        ~init:(Set.empty (module Tn))
        ~f:(fun ~key:_ ~data acc -> List.fold data ~init:acc ~f:Set.add)
    in
    let local_decls =
      string "/* Local declarations and initialization. */"
      ^^ hardline
      ^^ separate_map empty
           (fun (tn, node) ->
             let plc = placements () in
             if
               (not
                  (Tn.Placements.is_virtual_force plc tn 333
                  || Tn.Placements.is_materialized_force plc tn 336))
               (* Privatized to a pool-parallel [Grid] loop: declared per chunk inside that loop's
                  body instead (see [parallel_grid_loop]). *)
               && (not (Set.mem grid_privatized tn))
               && not (Set.mem !rendered_simdgroup_fragments tn)
             then
               let is_shared = Set.mem workgroup_shared tn in
               if is_shared then
                 (* Workgroup-shared placement (axis-types proposal §3): one tile per workgroup
                    instead of one per thread. [= {0}] is not allowed for shared declarations, so
                    zero-initialization stays as explicit [Zero_out] code (never elided for shared
                    nodes; see [zero_out_loop_redundant]); shared placements otherwise keep the
                    backend's default layout (no [aligned_local_attr]).

                    A [Swizzle_b128] tile is the exception: its layout contract is stated in 16-byte
                    units, and the warp-cooperative loads that consume it ([ldmatrix], gh-ocannl-481
                    item 3) require every row-group address to be 16-byte aligned. Row starts are
                    16-byte multiples by the [Stage] validation, so aligning the base is what makes
                    all of them aligned. The GNU attribute spelling is understood by every compiler
                    that has a shared address space here (nvcc, hipcc, MSL's clang). *)
                 (match swizzle_of tn with
                   | Some Low_level.Swizzle_b128 -> string "__attribute__((aligned(16))) "
                   | Some Low_level.Swizzle_elem | None -> empty)
                 ^^ string (Option.value_exn ~here:[%here] B.shared_decl_prefix)
                 ^^ string (B.typ_of_prec @@ Lazy.force tn.storage_prec)
                 ^^ space
                 ^^ string (get_ident tn)
                 (* A pipelined tile is [pt_depth] rotating copies (gh-487); the accesses select the
                    copy via [pp_pipelined_rotation]. [Schedule.check_hardware_limits] accounts the
                    same multiplier. *)
                 ^^ brackets
                      (OCaml.int
                         (Tn.num_elems tn
                         *
                         match Map.find !current_pipelined tn with
                         | Some { Low_level.pt_depth; _ } -> pt_depth
                         | None -> 1))
                 ^^ semi ^^ hardline
               else
                 local_array_decl
                   ~alias_ptr:(Set.mem !current_local_ptr_alias tn)
                   ~zero_init:node.Low_level.zero_initialized_by_code tn
                 ^^ hardline
             else empty)
           (Hashtbl.to_alist traced_store)
    in
    body := !body ^^ local_decls ^^ hardline;

    let main_logic = string "/* Main logic. */" ^^ hardline ^^ main_logic_doc in
    body := !body ^^ main_logic;

    if Utils.debug_log_from_routines () && B.log_involves_file_management then
      body :=
        !body ^^ hardline
        ^^ string "if (log_file) { fclose(log_file); log_file = NULL; }"
        ^^ hardline;

    let func_doc =
      func_header ^^ space ^^ lbrace ^^ nest 2 (hardline ^^ !body) ^^ hardline ^^ rbrace
    in
    (sorted_params, func_doc, launch)
end
