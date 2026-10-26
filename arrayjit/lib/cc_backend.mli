include Ir.Backend_impl.Lowered_no_device_backend

val compiler_command : unit -> string
(** The C compiler command this backend builds kernels with: [cc_backend_compiler_command] when set,
    else the toolchain's own C compiler as reported by [ocamlc -config]. Exposed for the
    generated-kernel census (gh-ocannl-650), which must compile the emitted sources with the SAME
    toolchain that will build them for real -- a census run against a different compiler would
    describe guarded arms nothing here selects. *)

val kernel_compiler_flags : unit -> string
(** The flags a kernel compile passes after {!compiler_command}: optimization level, arch and SIMD
    flags, fast math, contraction and the parallel-grid runtime, all resolved. Exposed for
    [test/operations/bridge_arm_coverage] (gh-ocannl-1145), which preprocesses retained kernel
    sources under exactly these flags to read which storage-bridge arm each one compiled into. *)

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
    auto-probed when unset). Read by {!vector_file_setting}, and alone wherever only the width
    matters. *)

val auto_vector_bytes : unit -> int
(** What {!vector_bytes_setting} answers when [cc_vector_bytes] is unset: the widest vector the
    configured target holds (64 with AVX-512 F/BW/VL, 32 with AVX2, else 16). Exposed for
    [test/operations/bridge_arm_coverage], which forces every width up to this one. *)

val vector_file_setting : unit -> Ir.Vector_file.t
(** The vector register file of the explicit SIMD renderings, width and count as one value
    (gh-ocannl-1399): the {!vector_bytes_setting} width, and how many registers of that width the
    kernels' target has (config [cc_vector_registers]; auto-probed when unset) -- the native file's
    count, 32 on AArch64 and AVX-512, 16 on other x86 targets, probed under the kernel flags --
    divided where the width is wider than a native register; [0] registers when explicit SIMD is
    off. Exposed for [Schedulers.cpu_mma_limits]'s [simd_vector], which the register tile's
    [Register_tile.rn_cap] reads in the seeding, as the renderer does here (gh-ocannl-1285). *)

val effective_pool_width : unit -> int
(** The worker-pool width after the pool policy: the restricted class's logical CPU count when the
    restriction fired, the process's affinity-respecting CPU count otherwise. Sizes the auto
    [cc_parallel_chunks]. *)

val pool_tag : unit -> string
(** Compact pool signature ([w8P], [w24], ...) identifying the pool the process executes on. Exposed
    for [Schedulers.cpu_mma_limits]'s [worker_pool_tag]: schedules crowned on one pool do not
    transfer to another (gh-ocannl-530), so the tag enters the autotune disk-cache key. *)

val timing_identity : unit -> Ir.Backend_intf.timing_identity option
(** CPU model and host partition, plus observed compiler target/executable metadata and macOS
    hardware UUID/OS build when available. OpenMP and non-macOS persistence remain supported.
    Executable provenance retains the probe cache's path/size/mtime contract; opaque wrappers and
    separately updated runtime libraries are not completely fingerprinted. *)

val codegen_tag : unit -> string
(** A short digest of this backend's resolved codegen configuration: the compiler command and its
    flags, the vector width, the fp16-arithmetic support, the parallel-grid syntax and chunking, and
    the per-chunk privatization cap. Exposed for [Schedulers.cpu_mma_limits]'s
    {!Ir.Backend_intf.hardware_limits.codegen_tag}: these settings are consulted at codegen,
    {e after} the lowered code the canonical digest names, so they enter the autotune disk-cache key
    (gh-ocannl-572) and a knob flip re-tunes rather than replaying a winner from another codegen
    regime. *)

val host_capabilities : unit -> (string * string) list
(** Which kind of host this process runs on, as the kernels see it (gh-ocannl-1371): the CPU model,
    the logical CPU count and pool tag, the ISA the SIMD probes found under the kernel flags
    ([avx2+fma], [avx512f+bw+vl], [neon]), the resolved {!vector_bytes_setting} beside
    {!auto_vector_bytes}, the register count of {!vector_file_setting} and the native register file,
    the fp16 arithmetic support, and the compiler with the arch and SIMD flags those were resolved
    under. Read from the same probes and settings the renderer reads, so it cannot disagree with
    them. [tools/host_caps.exe] prints it as the header line of every CI build job: GitHub's x86
    runners come with or without AVX-512, and the auto-probed width changes what the autotune
    pre-filter keeps. *)

val has_native_fp16_arithmetic : unit -> bool
(** Whether the configured C compiler and target execute [_Float16] arithmetic natively, at twice
    f32's lane count (ARMv8.2-FP16, AVX512-FP16) -- as opposed to lacking the type, or having it
    with every operation promoted to float (correct, but no throughput win). Probed once per process
    by test-compiling; overridable with [cc_fp16_arithmetic]. Exposed for
    [Schedulers.cpu_mma_limits]'s [native_fp16_arithmetic] and for the compute-precision decision in
    [CC_syntax_config]. *)

val kernel_link_flags : string Lazy.t
(** Platform-specific kernel shared-library flags. In particular, ELF kernels bind their own builtin
    calls locally even when the host stubs use a different native-half ABI. Runtime compiler probes
    use this same policy rather than restating the linker flags. *)

val compile_failure_detail :
  exit_code:int -> source_note:string -> command:string -> compiler_output:string -> string
(** The detail of the [Backend_rejected] a failed C compile raises: the exit code, the
    {!Ir.C_syntax.codegen_bug_claim} asking for [source_note]'s file, the command and the compiler's
    output. Exposed so a test can hand {!Ir.C_syntax.attribute_routine_name_failure} the report this
    backend actually raises: attributed to a routine's name, it is quoted without that claim; not
    attributed, it propagates with the claim intact. *)

type split_settings = { part_weight : int; jobs : int; unit_min_bytes : int }
(** gh-ocannl-1603: how a cc compile is split. [part_weight]: a kernel weighing more than sixteen
    parts of this weight is rendered outlined, into [noinline] parts it calls in order ([0]: never;
    config [cc_kernel_part_weight]). [jobs]: a batch is compiled as up to this many translation
    units, by concurrent compiler processes, linked into one library ([cc_compile_jobs]; always 1
    off Unix). [unit_min_bytes]: the least source a unit holds ([cc_compile_unit_min_bytes]). *)

val split_settings : unit -> split_settings
(** The settings in force: the configuration, or {!with_split_settings}'s. *)

val with_split_settings : split_settings -> (unit -> 'a) -> 'a
(** Runs [f] with these settings instead of the configured ones, then restores the previous ones:
    for a test comparing a split compile with a whole one in one process. Not domain-local. *)

val translation_units_compiled : unit -> int
(** How many translation units this process has compiled as parts of a split batch (each unit of a
    batch split into several counts; an unsplit compile counts none). *)
