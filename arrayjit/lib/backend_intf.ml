(** {1 The interface types for backends}

    The shared backend-interface types: the user-facing API ({!Backend}, {!routine}, {!buffer_loc})
    together with the interface pieces the implementation layers assemble from (marked
    implementation-facing where applicable). Implementation-only components live in {!Backend_impl}.
*)

open Base

type buffer_loc = { pool_id : int; offset : int } [@@deriving sexp, compare, equal]
(** A backend-agnostic, deterministic per-device buffer location: a [pool_id] into the device's
    backend-private [pool_id -> 'base] pool table, plus a byte [offset] within that pool. The
    concrete backend pointer ([Metal.Buffer.t] / [CUdeviceptr] / [void*]) lives only in that private
    table -- it never appears in any type of this shared interface -- so [buffer_loc] (pure
    integers) is stable across runs, diffable, and meaningful in logs and [.expected] files. Phase-1
    policy is one pool per tnode at [offset = 0], byte-for-byte equivalent to per-tnode allocation.
    An alias (future work) is the parent's [{ pool_id; offset = offset + delta }]. *)

type ctx_buffers = buffer_loc Map.M(Tnode).t [@@deriving sexp_of]

exception Backend_unavailable of { backend : string; detail : string }
(** Device discovery established that this backend cannot be used on this machine: its library is
    not linked in, or the driver reports no devices. This is deliberately narrow — it is the only
    failure {!Context.auto} treats as "try the next backend" (gh-ocannl-536 landing step 5). A
    driver that is present but fails to initialize is {e not} this: that is a real problem with a
    real installation, and silently selecting another backend would hide it. *)

let () =
  Stdlib.Printexc.register_printer (function
    | Backend_unavailable { backend; detail } ->
        Some (Printf.sprintf "Backend %s unavailable: %s" backend detail)
    | _ -> None)

(** Element formats tensor-core instructions accept for their multiplicand operands, and (reusing
    the same constructors) for their accumulator. This is deliberately NOT [Ops.prec]: formats like
    tf32 have no byte layout of their own, so they must never appear as a tensor node's storage
    precision. *)
type mma_input_format =
  | Mma_f32  (** Genuine f32 multiply-accumulate (Metal [simdgroup_float8x8]). *)
  | Mma_tf32
      (** f32 storage computed with a 10-bit mantissa (CUDA wmma [precision::tf32], sm_80+). Not a
          storage precision — data lives in memory as ordinary f32; only tensor-core loads truncate.
          Gated by {!Numerics.t.tf32_matmuls}. *)
  | Mma_f16
  | Mma_bf16
  | Mma_fp8_e5m2
      (** OCANNL's single fp8 today ([Ops.Fp8_prec], e5m2). An e4m3 constructor slots in here when
          the precision exists (gh-ocannl-481 item 2); descriptor entries are keyed per operand
          pair, so mixed e5m2×e4m3 combinations need no interface change. *)
[@@deriving sexp, compare, equal]

(** A physical layout the backend's tensor-core loads can consume for a cooperatively staged operand
    tile, beyond the plain row-major one (gh-ocannl-481 item 3, D3). *)
type mma_staged_layout =
  | Mma_swizzled_b128
      (** {!Low_level.Swizzle_b128}: the CUDA inline-PTX [mma.sync] arms read it with
          [ldmatrix.sync.aligned.m8n8]. Metal banks too but has no [ldmatrix] analogue; a later
          [simdgroup]-era entry would reuse this type. *)
[@@deriving sexp, compare, equal]

(** Where a tensor-unit rendering keeps its accumulator. The distinction is observable whenever a
    reduction is split into outer [k] blocks: {!Mma_per_statement} crosses only the intrinsic's own
    [k] extent, while {!Mma_fragment_scope} crosses the enclosing serial reduction. *)
type mma_emission_scope =
  | Mma_per_statement
      (** One {!Low_level.Tile_mma} emitted by the backend's [mma_syntax] hook. In a staged schedule
          this form reloads and stores [d] at every outer [k] block. *)
  | Mma_fragment_scope
      (** A persistent accumulator emitted by [mma_fragment_syntax], loaded before and stored after
          the enclosing serial reduction. *)
[@@deriving sexp, compare, equal]

type mma_capability = {
  mma_simd_width : int;
      (** Threads cooperating in one tile-MMA instruction (CUDA warp / Metal simdgroup width). *)
  mma_tile : int * int * int;
      (** The canonical intrinsic tile shape [(m, n, k)] (8×8×8 for Metal [simdgroup_matrix],
          16×16×16 for CUDA wmma), used where schedule construction has no typed operand site; a
          {!Low_level.t.Tile_mma}'s block extents must be multiples of the tile of the format
          actually emitted. Typed matmul/conv sketch seeds use [mma_format_tiles] below. *)
  mma_format_tiles :
    ((mma_input_format * mma_input_format * mma_input_format) * (int * int * int)) list;
      (** Per (a-operand, b-operand, accumulator) format intrinsic tile shapes, for formats whose
          tile diverges from [mma_tile] as well as the ones matching it (e.g. CUDA fp8 16×8×32, tf32
          16×16×8). Typed autotune seeds use the matching entry for divisibility; whether a given
          call ultimately emits is still decided by the backend's [mma_syntax] hook plus the
          {!Numerics} policy.

          The accumulator format is part of the key because it is NOT free to choose: the operand
          pair that a backend supports against an f32 accumulator is generally not the pair it
          supports against a narrow one. CUDA is the case that made this explicit (gh-ocannl-545):
          [nvcuda::wmma] pairs bf16 operands with a [float] accumulator only, so keying on the
          operands alone made the autotuner seed — and time, and rank — 36 candidates per arm on a
          uniformly-bf16 network that every one of them rendered as the lane-0 scalar fallback. *)
  mma_f16_wide_acc_scopes : mma_emission_scope list;
      (** The emission scopes in which the backend's uniform-f16 arm — an f16-storage destination
          with f16 operands — holds the accumulator in f32 and converts once at that scope's [d]
          boundary, as {!Numerics.Fp16_wide} requires (gh-ocannl-680, gh-ocannl-836).

          CUDA sm_80+ advertises only {!Mma_per_statement}: its inline-PTX [mma.sync.m16n8k16] arm
          is wide, but [nvcuda::wmma] has no uniform-f16 wide fragment arm, so staged seeds that
          would require {!Mma_fragment_scope} are withheld. HIP advertises both scopes since
          gh-ocannl-789 (rocWMMA's [(f16, f16, f32)] fragments and converted boundary), and Metal
          advertises both since gh-ocannl-837 (mixed-type [simdgroup_multiply_accumulate] plus a
          [thread_elements()] boundary copy). Withholding only the unsupported scope preserves legal
          tensorized schedules without letting an outer [k] split introduce extra f16 narrowing
          boundaries. *)
  mma_staged_layouts :
    ((mma_input_format * mma_input_format * mma_input_format) * mma_staged_layout) list;
      (** Format triples whose cooperatively staged operand tiles the backend can read in a
          non-row-major layout, and which layout (gh-ocannl-481 item 3, D3). Autotune's {e staged}
          mma sketches seed a swizzled twin per staged seed exactly for the advertised triples — the
          tuner, not a heuristic, then decides whether the bank-conflict fix beats the plain tile.

          Keyed by format triple for the same reason as [mma_format_tiles], and pre-filtered for the
          same reason (gh-ocannl-479): eligibility is per operand AND per orientation, and the
          orientation the staged sketches mint is each role's own. CUDA's fp8 arm, for instance, can
          feed A from [ldmatrix] in that orientation but not B — 4 fp8 bytes of a B register are
          strided there — so a swizzled fp8 twin would be timed and ranked as a tensorized candidate
          while rendering the scalar fallback. Empty everywhere the question does not arise. *)
  mma_pipeline_depths : int list;
      (** Software-pipelining depths beyond the unpipelined 1 that autotune's {e staged} mma/conv
          sketches propose as twins of each staged seed ([Schedule.Stage ~pipeline_depth],
          gh-ocannl-487) — a list, not a flag, so the search has a dimension. The portable
          double-buffered rendering is backend-generic, but a depth is advertised only where the arm
          has been validated on hardware: Metal ([[2]], phase 1, portable form) and CUDA ([[2]] on
          sm_80+, phase 2, the [cp.async] arm); HIP stays empty until its LDS async-copy arm lands.
          Empty on CPU backends (cooperative staging is not renderable there). *)
}
[@@deriving sexp, compare, equal]
(** Tensor-core capability descriptor (docs/proposals/tensorize-mma.md §6). Which operand precisions
    are supported is decided per call by the backend's [mma_syntax] hook (the emission is the source
    of truth); this record carries what schedule construction needs. *)

type hardware_limits = {
  max_threads_per_workgroup : int option;
      (** Upper bound on the number of threads in one workgroup (CUDA thread block / Metal
          threadgroup); [None] when the backend imposes no limit (the C backends render annotated
          loops serially). *)
  max_workgroup_memory_bytes : int option;
      (** Capacity in bytes of the workgroup-shared memory (CUDA [__shared__] / Metal
          [threadgroup]); [None] when the backend imposes no limit. *)
  max_workgroup_dims : (int * int * int) option;
      (** Per-dimension upper bounds on the launch's workgroup shape — the caps on [.x], [.y] and
          [.z] of {!Low_level.launch_dims}' [block], in that order. Beside, not instead of,
          {!field-max_threads_per_workgroup}: that one caps the {e product}, and the two are not the
          same fact. CUDA's [maxThreadsDim] is [(1024, 1024, 64)] — the [.z] component is 16x
          smaller than the product cap — so a workgroup of [2 x 2 x 128] has a perfectly legal
          512-thread product and is still an invalid launch configuration (gh-ocannl-679).
          [Workgroup] slots are capped at 3 and the innermost binds [.x], so the outermost annotated
          loop's extent lands on [.z] directly; no fold is involved.

          Unlike {!field-max_grid_yz} this carries all three bounds rather than one shared one,
          because here the dimensions genuinely differ: on CUDA [.z] is the odd one out, while HIP
          (queried [max_threads_dim]) and Metal ([maxThreadsPerThreadgroup]'s three components)
          report three equal values. [None] on the C backends, which render annotated loops
          serially.

          A tuple rather than an [int array], on two independent grounds. It is the shape
          {!field-mma_tile} already uses for a 3-D quantity here; and it is {e immutable}, which
          this record needs from every field it has. The GPU backends memoize their
          [hardware_limits] behind a [lazy] and [Context.hardware_limits] returns that record
          itself, so one mutable cell anywhere in it would let a caller deriving tighter limits for
          a custom schedule write through into the process-wide singleton — after which compiles
          reject legal kernels or admit illegal ones, with nothing to point at. A tuple also makes
          the arity exactly the three [Workgroup] slots, so no reader has to bounds-check a length
          the type does not promise.

          Enforced pre-driver by [Schedule.check_hardware_limits_classified] (as
          [Schedule_outcome.Workgroup_x_extent] / [_y_] / [_z_extent]); [Schedule.default_gpu] and
          [Schedule.zero_expansion] clamp their block size against the [.x] entry too, so the gate
          is a backstop rather than the first line of defence. *)
  max_grid_yz : int option;
      (** Upper bound on {e each} of the launch's [.y] and [.z] grid dimensions: the row-block count
          ([grid.(1)]) and the folded batch-extent product ([grid.(2)], the dimension [Grid] slots
          [>= 2] fold onto — gh-ocannl-643, [Low_level]'s hardware-axis section comment). One field
          rather than two because it is one hardware fact: CUDA and HIP cap [gridDim.y] and
          [gridDim.z] at the same 65535 while [gridDim.x] is 2^31-scale (on CUDA an architectural
          constant, unlike the queried per-device limits above; HIP queries it, conservatively as
          the smaller of the two components). What a caller does about an excess differs per
          dimension, and that distinction lives in the typed cause instead
          ([Schedule_outcome.Grid_y_extent] vs. [Grid_z_extent]). [None] when the backend imposes no
          such limit (Metal's threadgroups-per-grid dimensions are not 16-bit, and the C backends
          render annotated loops serially). Both dimensions are checked pre-driver by
          [Schedule.check_hardware_limits_classified]; the autotune batch-grid twins also consult
          the [.z] reading at seeding so an over-cap candidate is never proposed. *)
  mma : mma_capability option;
      (** Tile-MMA units ([simdgroup_matrix] / tensor cores); [None] when the backend has none wired
          — [Tile_mma] statements then render their scalar fallback. *)
  simd_vector_bytes : int;
      (** Vector register width in bytes used by the C backends' explicit vector-extension
          renderings ([Vectorized] loops, the register-tiled [Tile_mma] micro-kernel); [0] when the
          backend does no such rendering (GPU backends bind hardware axes instead). Carried here so
          schedule construction (autotune's seeding pre-filter, gh-ocannl-479) can statically rule
          out candidates the renderer must decline, e.g. a micro-kernel column extent below one
          vector's lane count. *)
  peak_flops : float option;
      (** Advisory peak arithmetic throughput in FLOP/s (single-precision, FMA counted as two), the
          hardware envelope of the analytic cost model (gh-ocannl-491): rough documented constants
          or cheap device queries — the model ranks candidate schedules, it does not predict
          runtimes. Never gates compilation and never overrides a measured timing; [None] when the
          backend offers no estimate. Known single-precision bias (gh-ocannl-575): a pure-fp16
          kernel on a {!native_fp16_arithmetic} target has twice this ceiling, so its roofline flops
          leg over-estimates — harmless for ranking because a site's candidates all share one
          policy-resolved compute precision (footprint widths, by contrast, are exact: they come off
          each node's own storage precision). *)
  peak_memory_bandwidth : float option;
      (** Advisory peak main-memory bandwidth in bytes/s, the other leg of the roofline envelope
          (gh-ocannl-491). Same contract as [peak_flops] — advisory, rough, never load-bearing for
          correctness; [None] when the backend offers no estimate — with one bias requirement
          (gh-ocannl-578): the value must be a {e class ceiling}, at least what any machine of the
          backend's class can sustain. Streaming kernels with exact byte counts are routine now (the
          calibration pass, packed initializations), and each one achieving more than the advisory
          trips the gh-514 agreement warning — while under [autotune_bound_pruning] an understated
          leg over-prunes. Overstatement only loosens an advisory bound; calibrated [model_peak_*]
          overrides beat these wherever fidelity matters. *)
  native_fp16_arithmetic : bool;
      (** Whether 16-bit float arithmetic executes natively at twice f32's lane count
          (gh-ocannl-516: ARMv8.2-FP16, AVX512-FP16). [false] covers both "no [_Float16] on this
          target" and the middle case that matters for ranking: the type exists and computes
          correctly, but the compiler implements it by promoting to float, so the lane count does
          {e not} double and candidates must not be seeded as if it did. Whether the type exists at
          all is a separate, purely textual question the emitted C answers for itself
          ([HAS_NATIVE_FLOAT16]); this field is about throughput.

          Always [false] on the GPU backends, whose 16-bit story is their native types and
          tensor-core shapes rather than a CPU vector width. *)
  worker_pool_tag : string option;
      (** Compact signature of the worker pool timings execute on ([w8P], [w24], ...), filled by the
          CPU backends from the pool-uniformity policy (gh-ocannl-530). Enters the autotune
          disk-cache key the way the numerics tag does: schedules crowned on one pool do not
          transfer to another, so a policy flip or a different external pinning must re-tune rather
          than replay. [None] (GPU backends) leaves the cache key unchanged. *)
  codegen_tag : string option;
      (** Compact signature of this backend's {e codegen} configuration: the settings the backend
          consults when rendering and compiling a kernel, which are therefore invisible to the
          canonical digest of the lowered code (gh-ocannl-572). Same contract as
          {!field-worker_pool_tag} — it enters the autotune disk-cache key, so a knob flip re-tunes
          instead of replaying a winner crowned in another codegen regime, which is the hazard
          gh-ocannl-568 measured at 5.9x. Fill it from resolved values, not raw settings: what
          ["auto"] resolves to is a per-machine fact, and crowns do not transfer across machines
          either. [None] where the backend has no such knobs. *)
}
[@@deriving sexp, compare, equal]

(** Equality of the [(a-operand, b-operand, accumulator)] format triple that keys
    {!field-mma_capability.mma_format_tiles} and {!field-mma_capability.mma_staged_layouts}. All
    three components are part of the key, for the reason those fields document (gh-ocannl-545):
    which operand pair a backend supports is a function of the accumulator it is paired against. *)
let equal_mma_format_triple (a1, b1, d1) (a2, b2, d2) =
  equal_mma_input_format a1 a2 && equal_mma_input_format b1 b2 && equal_mma_input_format d1 d2

(** Whether the backend's tensor-core unit advertises an intrinsic tile for this [(a, b, d)] format
    triple -- [false] when it advertises the triple's operands only against another accumulator, and
    [false] when the backend has no such unit at all ([mma = None]).

    The typed way to interrogate the MMA seam ({!field-hardware_limits.mma}, gh-ocannl-822): callers
    ask the descriptor rather than re-deriving the [mma_format_tiles] lookup, which is easy to
    open-code with the accumulator dropped from the key. Whether a given call ultimately emits is
    still decided per call by the backend's [mma_syntax] hook plus the {!Ir.Numerics} policy; this
    answers what schedule construction and its tests can know statically. *)
let advertises_mma_format (limits : hardware_limits) ~a ~b ~d =
  match limits.mma with
  | None -> false
  | Some mma -> List.Assoc.mem mma.mma_format_tiles (a, b, d) ~equal:equal_mma_format_triple

type codegen_capabilities = {
  supports_f64 : bool;
      (** Whether the backend dialect can represent f64 tensor storage. This is explicit rather than
          an exception probe against [typ_of_prec]. *)
  accum_prec : Ops.prec -> Ops.prec;
      (** The resolved accumulator precision for a storage precision, from the same
          [C_syntax_config.accum_prec] function code generation uses. *)
  asynchronous_staging_copy : bool;
      (** Whether eligible pipelined staging copies use a dialect-specific asynchronous copy arm. A
          portable synchronous depth-2 pipeline is not this capability. *)
}
(** Stable code-generation facts callers need before compiling. Actual rendering decisions stay on
    the compiled routine's censuses. *)

(** Conservative implementation-facing defaults for missing and mock backends. A real C-family
    backend derives this record from its {!Ir.C_syntax.C_syntax_config}. *)
let no_codegen_capabilities =
  { supports_f64 = false; accum_prec = Fn.id; asynchronous_staging_copy = false }

let no_hardware_limits =
  {
    max_threads_per_workgroup = None;
    max_workgroup_memory_bytes = None;
    max_workgroup_dims = None;
    max_grid_yz = None;
    mma = None;
    simd_vector_bytes = 0;
    peak_flops = None;
    peak_memory_bandwidth = None;
    native_fp16_arithmetic = false;
    worker_pool_tag = None;
    codegen_tag = None;
  }

type device_dump = {
  group : string;  (** The group atom naming the dump, e.g. ["cuda_devices"]. *)
  devices : (string * Sexp.t) list list;
      (** One [(key, value)] assoc per device, in ordinal order. *)
}
[@@deriving sexp_of]
(** A parsed {!Backend_device_common.static_properties} dump: see {!parse_static_properties}. *)

(** Reads a {!Backend_device_common.static_properties} dump per its contract (gh-ocannl-710), or
    answers [None] when the sexp is not a device dump at all.

    The contract, which every backend that enumerates devices honors and this function is the single
    reader of:

    - The dump is [(<group> <entry> ...)]: an atom naming the group, then the entries. [<group>]
      ends in ["_devices"] -- and is [<backend name>_devices] -- exactly when the dump enumerates
      devices, so a dump that describes something else (an unlinked backend's
      [(<backend>_missing (error ...))], see [Lowered_backend_missing]) is distinguishable without
      guessing.
    - Every entry of a [_devices] dump is a device, and is [Sexp.message]-shaped: the atom [device]
      followed by [(key value)] pairs, ONE nesting level -- no list-of-pairs wrapper around the
      pairs. There is one entry per device, in ordinal order.
    - Every device carries at least [device_name] and [device_ordinal], and all the devices of one
      dump carry the same keys, so an entry indexes uniformly and two machines' dumps diff line by
      line.
    - The device COUNT is not a child of its own: it is the number of entries. Neither is any other
      backend-level fact -- a child that is not a device is what made a generic reader reproduce the
      backend's [num_devices] as a second, fictitious device (gh-ocannl-710).

    A dump violating the shape answers [None] rather than a partial reading: a reader that invents
    structure is worse than one that says it does not recognize the shape. Uniform keys and the
    ordinal sequence are contract too, but are not enforced here -- they are what
    [test/operations/static_properties_contract.ml] checks against each backend's real dump. *)
let parse_static_properties (props : Sexp.t) : device_dump option =
  let entry = function
    | Sexp.List (Sexp.Atom "device" :: (_ :: _ as fields)) ->
        List.map fields ~f:(function
          | Sexp.List [ Sexp.Atom key; value ] -> Some (key, value)
          | _ -> None)
        |> Option.all
    | _ -> None
  in
  match props with
  | Sexp.List (Sexp.Atom group :: entries) when String.is_suffix group ~suffix:"_devices" ->
      Option.map (Option.all (List.map entries ~f:entry)) ~f:(fun devices -> { group; devices })
  | _ -> None

let simd_lane_ladder = Register_tile.simd_lane_ladder

let simd_lanes_for ~vector_bytes ~elt_bytes ~extent =
  simd_lane_ladder ~vector_bytes ~elt_bytes
  |> List.filter ~f:(fun lanes -> extent >= lanes)
  (* Loop trips: [extent / lanes] vector steps plus [extent mod lanes] scalar remainder iterations.
     Both issue one instruction per body operation, so the sum is comparable across widths with no
     fitted constant -- and it is what makes the choice extent-shaped rather than width-shaped: it
     takes the full width on a long loop (517 at 16 lanes is 32 + 5 trips against 8 lanes' 64 + 5),
     and steps down where the wider vector would leave a remainder the narrower one divides away (40
     is 2 + 8 against 5 + 0). *)
  |> List.min_elt ~compare:(fun a b ->
      let trips lanes = (extent / lanes) + (extent % lanes) in
      match compare_int (trips a) (trips b) with
      | 0 -> compare_int b a (* equal trips: the wider vector *)
      | c -> c)

let simd_reduce_chains ~lanes ~extent =
  if 4 * lanes <= extent then 4 else if 2 * lanes <= extent then 2 else 1

let simd_reduce_lanes_for ~vector_bytes ~elt_bytes ~extent =
  let cost lanes =
    let chains = simd_reduce_chains ~lanes ~extent in
    let step = chains * lanes in
    let steps = extent / step in
    (* One vector update per chain per step (the first step is the chains' initialization), the
       leftover iterations serially, then the epilogue: [chains - 1] whole-vector combines and the
       [lanes - 1] scalar operations of the horizontal fold. *)
    (steps * chains) + (extent - (steps * step)) + (chains - 1) + (lanes - 1)
  in
  simd_lane_ladder ~vector_bytes ~elt_bytes
  |> List.filter ~f:(fun lanes -> extent >= lanes)
  |> List.min_elt ~compare:(fun a b ->
      match compare_int (cost a) (cost b) with
      | 0 -> compare_int b a (* equal cost: the wider vector *)
      | c -> c)

(** The lane count for an ACCUMULATING [Vectorized] loop, which {!simd_lanes_for} would get wrong at
    short extents: that rendering ends in a horizontal fold whose length is the lane count itself,
    so a wider vector buys fewer updates and pays a longer dependent tail. At an f32 extent of 64,
    16 lanes save four vector updates over 8 and add eight operations to the fold — the wider width
    losing on a loop the elementwise metric would hand it. The cost mirrors the emission term for
    term, so the two cannot drift: chains and step as {!C_syntax} computes them, the serial
    leftover, and the epilogue's vector combines and scalar fold. *)

(** The lane count an explicit-SIMD rendering should use for a loop of [extent] iterations over
    [elt_bytes]-wide elements on a [vector_bytes]-wide register file: the width of
    {!simd_lane_ladder} that minimizes loop trips, [None] where even the narrowest exceeds the
    extent. (The register-tiled micro-kernel searches the ladder itself: its peel is a scalar column
    loop rather than a remainder of the same body, and it has a fitted cost model for that.)

    A single width would make a wider machine emit {e less} vector code than a narrower one — the
    renderings decline outright below one full vector, so widening the auto [cc_vector_bytes] from
    32 to 64 on an AVX-512 target (gh-ocannl-621 follow-up) would drop every f32 loop of extent
    8..15 to the serial fallback, and an accumulating loop loses reassociation with it, which is the
    whole point of the [Vectorized] retype. Nor is "the widest that fits" enough: at extent 40 a
    16-lane vector covers 32 columns and leaves 8 to scalar code, where 8 lanes divide the extent
    evenly — a wider register file made to run slower. Hence a ladder and a choice, not a number.

    The floor is [min vector_bytes 32] — never narrower than the width the machine used before any
    widening. Degrading past it would newly vectorize loops that used to render serially, and a
    vector accumulation reassociates, so that is a numerics change rather than a scheduling one and
    does not belong in a width default.

    Shared by the renderer ({!C_syntax}) and autotune's seeding pre-filter ({!Sketch_families}) so
    the two agree on which extents are vectorizable: a stricter seeding rule would withhold
    candidates the renderer would in fact tile. *)

(** The backend slab allocator, replacing the per-tnode [Alloc_buffer] interface. The shared
    allocator seam (see {!Backends}) mints deterministic per-device [pool_id]s and calls these
    int-in / int-out primitives; the backend keeps the [pool_id -> 'base] table private. The
    [pool_id -> 'base] resolution (then [base + offset]) stays inside the backend. *)
module type Slab_alloc = sig
  type device

  val alloc_pool :
    ?mode:Tnode.memory_mode -> device -> pool_id:int -> size_in_bytes:int -> alignment:int -> unit
  (** Allocates the slab for [pool_id] on [device]. The optional [?mode] carries the tnode's memory
      mode so backends can pick a storage mode (Metal private vs. shared); backends that do not care
      ignore it. *)

  val free_pool : (device -> pool_id:int -> unit) option
  (** Frees the slab for [pool_id] and drops its table entry. [None] for backends that rely on GC.
  *)

  val memset_zero : device -> pool_id:int -> offset:int -> size_in_bytes:int -> unit
  (** Zero-initializes [size_in_bytes] at [base_of pool_id + offset]. *)
end

type merge_buffer_use = No | Copy [@@deriving sexp_of]

(** Kernel-parameter sources: the codegen <-> backend contract for a compiled routine's parameters.
    Implementation-facing (consumed by {!C_syntax} and the backends' link steps); it lives in this
    file because the shared {!Backend_impl.Lowered_no_device_backend} signature mentions it. *)
type kparam_source =
  | Log_file_name
  | Merge_buffer
  | Kparam_ptr of Tnode.t
  | Kparam_pool_slab of int
      (** gh-ocannl-344: the [i]-th pool base-pointer parameter of a pooled kernel (Metal). A fixed
          number of these is emitted; at link the backend binds slab [i] to the pool assigned index
          [i] (or a duplicate of an in-use pool for the unused tail). Lets a kernel reach hundreds
          of tensor nodes through a handful of bound pools, staying under Metal's ~31 binding limit.
      *)
  | Kparam_pool_slots of Tnode.t list
      (** gh-ocannl-344: the per-routine slot table accompanying {!Kparam_pool_slab}. For the [k]-th
          tnode in this list the backend writes (pool_index, byte_offset); the shader reads it to
          form the typed pointer by casting (pools at pool_index) + byte_offset. Emitted only by
          pooled (Metal) codegen; per-tnode pointer backends (C, CUDA) never produce it. *)
  | Static_idx of Indexing.static_symbol
[@@deriving sexp_of]

(** The link-time impossibility every [`Per_param] backend shares: pooled kparams are emitted only
    by pooled codegen, so a backend whose {!C_syntax_config.ptr_param_style} is [`Per_param] can
    never be handed one. Named here, once, next to the constructors whose invariant it states. *)
let unexpected_pooled_kparam ~backend =
  invalid_arg (backend ^ ".link: unexpected pooled kparam (" ^ backend ^ " uses per-tnode pointers)")

type 'context routine = {
  context : 'context;
  schedule : Task.t;
  bindings : Indexing.lowered_bindings;
  name : string;
  inputs : Set.M(Tnode).t;
      (** The materialized read-only and read-before-write (within the routine) non-constant nodes.
          They are inputs in a broad sense, as they could be recurrent nodes or parameters. *)
  merge_buffer_input : Tnode.t option;
      (** Similar to {!field-inputs}, for the merge buffer. The execution-dependency ledger consumes
          this as a read edge on the transfer that filled the transient slab, but it is not an
          ordinary context input requiring initialization. *)
  outputs : Set.M(Tnode).t;  (** All the materialized nodes written-to by the routine. *)
}
[@@deriving sexp_of]

module type Device_config_common = sig
  type dev [@@deriving sexp_of]
  (** Interface to a device driver. *)

  type runner [@@deriving sexp_of]
  (** Interface to a stream driver. *)

  type event [@@deriving sexp_of]
  (** An event tracks if a device's runner finished computing past a particular point in its
      schedule. These values are used internally for scheduling across devices/queues of the
      backend, and can be used for explicit scheduling. *)

  val name : string
end

type ('dev, 'runner, 'event) device = {
  dev : 'dev;
  ordinal : int;
      (** The number of the represented backend's device, in the range from 0 to the number of the
          backend's devices - 1. *)
  device_id : int;
      (** A unique identifier among all device instances of all backends. Note that multiple
          [device_id] (distinct device instances) might refer to the same physical device. *)
  runner : 'runner;
  merge_buffer : buffer_loc option ref;
      (** The merge buffer's reserved single-tenant pool location, or [None] if not yet allocated.
          The slab can be reused (grown in place) for nodes that fit. *)
  mutable merge_buffer_capacity : int;
      (** Byte capacity of the reserved merge-buffer pool; drives the grow decision. *)
  updating_for : 'event Hashtbl.M(Tnode).t;
      (** The completion event for the most recent updating (writing to) a node via this device. *)
  mutable updating_for_merge_buffer : (Tnode.t * 'event option) option;
      (** The tensor node that was most recently scheduled to be in the device's merge buffer. See
          also {!field-updating_for}. *)
  constant_buffer_cache : buffer_loc Hashtbl.M(Tnode).t;
      (** Per-device cache for read-only/constant buffer allocations. *)
  mutable next_pool_id : int;
      (** Deterministic per-device pool-id counter, advanced by the shared allocator seam in tnode
          iteration order. Pool id 0 is reserved for the merge buffer; tnode pools start at 1. *)
}
(** A device bundles its single compute [runner] with the associated buffer and event tracking: the
    [merge_buffer], the [updating_for] writer events (used for cross-device coherence by
    {!Backend.device_to_device}), and the deterministic pool-id counter. The design is
    forward-compatible with a future fixed-role prefetch/transfer runner. *)

let sexp_of_device _ _ _ device = [%sexp_of: string * int] ("device_id", device.device_id)
let equal_device d1 d2 = d1.device_id = d2.device_id

(** Pool id 0 on every device is reserved for the (single-tenant) merge buffer. *)
let merge_buffer_pool_id = 0

(** Invalidate every software ownership claim for the reserved merge slab as one transaction.
    [remove_slab_claim] drops the backend-private pool-table entry and must not perform the fallible
    raw free: callers free the previously-found slab only after this function has made it
    unreachable. The writer is recommitted separately, after the replacement copy is scheduled. *)
let invalidate_merge_slab device ~remove_slab_claim =
  remove_slab_claim ();
  device.merge_buffer := None;
  device.merge_buffer_capacity <- 0;
  device.updating_for_merge_buffer <- None

(** Commit a successfully allocated reserved merge slab and its backend-private table entry as one
    transaction. The writer marker stays invalid until the copy into this slab is scheduled. *)
let commit_merge_slab device ~size_in_bytes ~install_slab_claim =
  install_slab_claim ();
  device.merge_buffer := Some { pool_id = merge_buffer_pool_id; offset = 0 };
  device.merge_buffer_capacity <- size_in_bytes

type ('dev, 'runner, 'event) context = {
  device : ('dev, 'runner, 'event) device;
  parent : ('dev, 'runner, 'event) context option;
  ctx_buffers : ctx_buffers;
      (** This map contains the deterministic buffer locations used in this context or an ancestor
          context. *)
  finalized : Utils.atomic_bool;
  mutable released_pool_ids : Set.M(Int).t;
      (** Pools this context has already released. Retained across a failed-finalize retry so a
          cleanup that freed some pools before raising never calls the backend free twice. *)
  optimize_ctx : Low_level.optimize_ctx;
      (** The optimization context threaded through compilation: all OCANNL backends compile through
          the {!Low_level} IR, so this is concretely {!Low_level.optimize_ctx} (the abstraction for
          hypothetical assignments-level backends was retired; the [Assignments.comp -> code] seam
          can be reintroduced if such a backend ever materializes). *)
  merge_buffer_node : Tnode.t option;
      (** The tensor node that a {!Backend.device_to_device} transfer with [into_merge_buffer:Copy]
          placed (or will place) into this context's device's merge buffer. It is a static,
          immutably-chained fact carried producer -> consumer: linking a consumer whose code expects
          a merge-buffer node verifies it against this field at link time. A transfer with
          [into_merge_buffer:No] does not touch the merge buffer and inherits the parent's value. *)
}
[@@deriving sexp_of]

(** The one constructor for evolving a context in place of itself with a newly allocated buffer: the
    result supersedes [ctx] as the lineage leaf while deliberately keeping its lifecycle identity —
    the same {!field:context.parent} link and the {e same} {!field:context.finalized} flag, so at
    most one of the pair can ever free the pools they share — and no context creation is counted in
    [Alloc_census]. The buffer-allocating transfer entry points ([init_from_host],
    [init_from_device]) go through this; a compile/link result is a new lifecycle node and goes
    through [Device.make_child] instead. *)
let evolve_with_buffer ctx tn loc =
  { ctx with ctx_buffers = Map.add_exn ctx.ctx_buffers ~key:tn ~data:loc }

module type Device_types = sig
  include Device_config_common

  type nonrec device = (dev, runner, event) device [@@deriving sexp_of]
  type nonrec context = (dev, runner, event) context [@@deriving sexp_of]
end

module type Device = sig
  include Device_types
  include Slab_alloc with type device := device

  (* [pool_id -> base] resolution is intentionally NOT part of this shared signature: the concrete
     backend pointer never appears in a shared type. Resolution lives backend-side (see
     {!Backend_impl.Make_slab} / each backend's private [Slab]). *)

  val make_device : dev -> runner -> ordinal:int -> device

  val make_context :
    ?ctx_buffers:ctx_buffers -> ?optimize_ctx:Low_level.optimize_ctx -> device -> context
  (** Returns a context without a parent. *)

  val make_child :
    ?ctx_buffers:ctx_buffers ->
    ?optimize_ctx:Low_level.optimize_ctx ->
    ?merge_buffer_node:Tnode.t option ->
    context ->
    context
  (** Returns a context with the same {!field:Backend_intf.context.device},
      {!field:Backend_intf.context.ctx_buffers}, {!field:Backend_intf.context.optimize_ctx},
      {!field:Backend_intf.context.merge_buffer_node} if omitted, as the given context's, which is
      also the {!field:Backend_intf.context.parent}. *)

  val get_name : device -> string
end

(** The device, event and synchronization part of the backend interface, shared by the user-facing
    {!Backend} and the implementation-facing {!Backend_impl.Lowered_backend}. Does not include:
    compilation and linking (they differ between the user-facing and lowered interfaces); copying
    and tensor-node-level synchronization (copying is different for user-facing and
    implementation-facing APIs, synchronization is provided by a component outside of backend
    implementations). *)
module type Backend_device_common = sig
  include Device

  val sync : event -> unit
  (** Blocks till the event completes, if it's not done already.

      It is rarely needed to call [sync] explicitly, because it should always be called internally
      when necessary, in particular before extracting values from host. *)

  val is_done : event -> bool
  (** Whether the event completed. *)

  val will_wait_for : context -> event -> unit
  (** Schedules waiting for the given event on the context's device.

      NOTE: it should rarely be needed to call [will_wait_for] explicitly, because it should always
      be called internally when necessary. *)

  val static_properties : unit -> Sexp.t
  (** Returns a sexp description of the properties of all devices, in the shape {!device_dump}
      documents and {!parse_static_properties} reads:
      [(<backend>_devices (device (key value) ...) (device (key value) ...) ...)] -- a group atom
      naming the dump, then exactly one [Sexp.message]-shaped entry per device, in ordinal order.
      See {!parse_static_properties} for the full contract (gh-ocannl-710).

      A function so that computing it (device enumeration) does not run at backend-module
      initialization: singleton backends instantiate eagerly at program startup, where touching a
      driver could fail runs that never use the backend. *)

  val hardware_limits : unit -> hardware_limits
  (** Conservative per-workgroup device limits: on multi-device backends the minimum across the
      devices, so code compiled once (compilation is not per-device) is valid wherever it links.
      All-[None] for backends that do not bind hardware axes. A function for the same reason as
      {!static_properties}: computing it (device enumeration) must not run at backend-module
      initialization. *)

  val codegen_capabilities : unit -> codegen_capabilities
  (** Facts from the backend's C-syntax configuration. A function, like {!hardware_limits}, so
      policy-dependent fields reflect the current process configuration. *)

  val classify_failure : Schedule_outcome.phase -> exn -> Schedule_outcome.classified_cause option
  (** Recognizes backend-owned failures at a tagged boundary. Returning [None] leaves the common
      policy to treat the exception as unclassified. *)

  val get_used_memory : device -> int
  (** Returns (an upper bound of) the memory used for arrays, in bytes. *)

  val get_global_debug_info : unit -> Sexp.t
  (** Global debug information; backend-specific and might evolve independently on the backends. *)

  val get_debug_info : device -> Sexp.t
  (** Per-device debug information; backend-specific and might evolve independently on the backends
  *)

  val await : device -> unit
  (** Blocks till the device becomes idle, i.e. synchronizes the device's runner. *)

  val all_work : device -> event
  (** Returns the event indicating if any currently running or scheduled computations on the device
      have completed. *)

  val is_idle : device -> bool
  (** Whether the device's runner is currently waiting for work. *)

  val get_device : ordinal:int -> device
  val num_devices : unit -> int
end

module type With_buffer_retrieval_and_syncing = sig
  type device
  type context
  type event

  val from_host : context -> Tnode.t -> Ndarray.t -> bool
  (** [from_host ctx tn src] schedules a copy of the explicit host buffer [src] into [tn]'s
      in-context device buffer and returns true, or returns false if the node is not in context.
      After [gh-ocannl-333] the host buffer is supplied by the caller (e.g. {!Context.set_values});
      it is no longer read from the tensor node. *)

  val init_from_host : context -> Tnode.t -> Ndarray.t -> context
  (** Schedules a copy from the explicit host buffer to context: a variant of {!from_host} that
      requires the input context to not contain the tensor node, and outputs the context with the
      tensor node. *)

  val to_host : context -> Tnode.t -> Ndarray.t -> bool
  (** [to_host ctx tn dst] schedules a copy of [tn]'s in-context device buffer into the explicit
      host buffer [dst] and returns true, or returns false if the node is not in context. After
      [gh-ocannl-333] the destination buffer is supplied by the caller (e.g. {!Context.to_host}); it
      is no longer the tensor node's own array. *)

  val device_to_device :
    Tnode.t ->
    into_merge_buffer:merge_buffer_use ->
    dst:context ->
    src:context ->
    context routine option
  (** [device_to_device tn ~into_merge_buffer ~dst ~src] builds a transfer {e routine} instead of
      scheduling the copy directly. The caller schedules it (e.g. via [Task.run r.schedule]) or
      links a consumer against [r.context]. It returns:
      - [None] if there is nothing to transfer: the node is absent from [src]; or, for
        [into_merge_buffer=No], the node is absent from [dst] or the source and destination buffers
        are physically the same.
      - [Some r] otherwise. Running [r.schedule] waits for writing into the tensor node on [src] to
        finish, then performs the copy and updates the writer event.
      - For [into_merge_buffer=No], the copy goes from [src] to [dst]; [r.context] is a child of
        [dst] inheriting its {!field:Backend_intf.context.merge_buffer_node}.
      - For [into_merge_buffer=Copy], the copy goes from [src] to the merge buffer of [dst]'s
        stream; [r.context] is a child of [dst] with [merge_buffer_node = Some tn], so that linking
        a consumer of the merge buffer against [r.context] statically verifies the node. *)

  val init_from_device : Tnode.t -> dst:context -> src:context -> context
  (** Schedules a copy from [src] to [dst]: a variant of {!device_to_device} with
      [into_merge_buffer=No] that requires the input [src] context to not contain the tensor node,
      and outputs the [dst] context with the tensor node. *)

  val sync_device : device -> unit
  (** Synchronizes all the streams on a device, and cleans up (removes) all associated events. *)
end

module type Backend = sig
  type code [@@deriving sexp_of]

  val empty_optimize_ctx : unit -> Low_level.optimize_ctx

  val compile :
    Low_level.optimize_ctx ->
    ?name:string ->
    ?lowered_transform:(Low_level.optimized -> Low_level.optimized list) ->
    ?prelowered:Low_level.optimized ->
    Indexing.unit_bindings ->
    Assignments.comp ->
    code
  (** [name] is used to derive names for compilation artifacts. If omitted, it's derived via
      {!Assignments.get_name_exn}. [lowered_transform] is applied to the optimized lowered code
      before backend compilation — the seam where schedule transforms (and hand-annotating tests)
      rewrite loops with hardware axis types, barriers and shared placements
      (docs/proposals/axis-types-for-loops.md). It returns the routine's KERNEL SEGMENTS: a
      whole-routine transform returns a singleton ([fun o -> [ f o ]]), while a transform that
      splits the routine into several kernels (fission, {!Schedule.fission_scheduled}) returns one
      element per segment. The segments compile as one fissioned routine and run back-to-back on the
      routine's stream with a device-side event chained at each boundary, exactly as
      {!Schedule.maybe_default_schedules}' segments do. It must return a non-empty list.

      [prelowered] (gh-ocannl-562, a test seam) replaces this compile's own lowering of [comp] with
      the given optimized code: it drives codegen, I/O classification, liveness planning and
      context-node settlement alike, so a hand-built {!Low_level.optimized} becomes executable
      rather than only analyzable. [comp] is then consulted for nothing but the routine's name (when
      [name] is omitted) and its context-node/embedded-node bookkeeping — {!Assignments.empty_comp}
      with an explicit [name] is the usual choice; a caller that wants the routine's nodes settled
      against a prior context supplies a comp naming them. The record's [optimize_ctx] becomes the
      linked context's lineage state in place of the fork a real compile would make, so the caller
      owns its provenance. Everything downstream of lowering is unchanged, including the default
      schedule annotator — pass [~lowered_transform:(fun o -> [ o ])] to keep hand-built code
      exactly as written. *)

  include Backend_device_common

  val link : context -> code -> context routine
  (** Returns the routine for the code's procedure, in a new context derived from the given context.
  *)

  include
    With_buffer_retrieval_and_syncing
      with type device := device
       and type context := context
       and type event := event
end
