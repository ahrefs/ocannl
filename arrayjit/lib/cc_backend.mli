include Ir.Backend_impl.Lowered_no_device_backend

val compiler_command : unit -> string
(** The C compiler command this backend builds kernels with: [cc_backend_compiler_command] when set,
    else the toolchain's own C compiler as reported by [ocamlc -config]. Exposed for the
    generated-kernel census (gh-ocannl-650), which must compile the emitted sources with the SAME
    toolchain that will build them for real -- a census run against a different compiler would
    describe guarded arms nothing here selects. *)

val compiler_executable_identity : string -> string
(** Resolves every executable token in a compiler command and fingerprints its path, size and
    modification time. Exposed for the generated-kernel census's persistent listing identity: an
    in-place toolchain replacement must not replay assembly from the previous executable. *)

val pool_parallel_grid : unit -> bool
(** Whether this backend renders the outermost [Grid] loops it proves safe as concurrent chunks of
    the native pool — the gate of [C_syntax.collect_parallel_grid]: a pool syntax was probed, more
    than one chunk is configured, and no kernel logging is on. Exposed so a test can say what a
    bound grid axis means on cc (gh-ocannl-959): a binding the legality check judges where this
    holds, a serial loop where it does not. *)

val vector_bytes_setting : unit -> int
(** The vector register width in bytes for the explicit SIMD renderings (config [cc_vector_bytes];
    auto-probed when unset). Exposed for [Schedulers.cpu_mma_limits]'s [simd_vector_bytes]. *)

val effective_pool_width : unit -> int
(** The worker-pool width after the pool policy: the restricted class's logical CPU count when the
    restriction fired, the process's affinity-respecting CPU count otherwise. Sizes the auto
    [cc_parallel_chunks]. *)

val pool_tag : unit -> string
(** Compact pool signature ([w8P], [w24], ...) identifying the pool the process executes on. Exposed
    for [Schedulers.cpu_mma_limits]'s [worker_pool_tag]: schedules crowned on one pool do not
    transfer to another (gh-ocannl-530), so the tag enters the autotune disk-cache key. *)

val codegen_tag : unit -> string
(** A short digest of this backend's resolved codegen configuration: the compiler command and its
    flags, the vector width, the fp16-arithmetic support, the parallel-grid syntax and chunking, and
    the per-chunk privatization cap. Exposed for [Schedulers.cpu_mma_limits]'s
    {!Ir.Backend_intf.hardware_limits.codegen_tag}: these settings are consulted at codegen,
    {e after} the lowered code the canonical digest names, so they enter the autotune disk-cache key
    (gh-ocannl-572) and a knob flip re-tunes rather than replaying a winner from another codegen
    regime. *)

val has_native_fp16_arithmetic : unit -> bool
(** Whether the configured C compiler and target execute [_Float16] arithmetic natively, at twice
    f32's lane count (ARMv8.2-FP16, AVX512-FP16) -- as opposed to lacking the type, or having it
    with every operation promoted to float (correct, but no throughput win). Probed once per process
    by test-compiling; overridable with [cc_fp16_arithmetic]. Exposed for
    [Schedulers.cpu_mma_limits]'s [native_fp16_arithmetic] and for the compute-precision decision in
    [CC_syntax_config]. *)
