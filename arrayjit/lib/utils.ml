open Base

type settings = {
  mutable log_level : int;
  mutable debug_log_from_routines : bool;
      (** If the [debug_log_from_routines] flag is true _and_ the flag [log_level > 1], backends
          should generate code (e.g. fprintf statements) to log the execution, and arrange for the
          logs to be emitted via ppx_minidebug. *)
  mutable output_debug_files_in_build_directory : bool;
      (** Writes compilation related files in the [build_files] subdirectory of the run directory
          (additional files, or files that would otherwise be in temp directory). When both
          [output_debug_files_in_build_directory = true] and [log_level > 1], compilation should
          also preserve debug and line information for runtime debugging. *)
  mutable fixed_state_for_init : int option;
  mutable print_decimals_precision : int;
      (** When rendering arrays etc., outputs this many decimal digits. *)
  mutable check_half_prec_constants_cutoff : float option;
      (** If given, generic code optimization should fail if a half precision FP16 constant exceeds
          the cutoff. *)
  mutable default_prng_variant : string;
      (** The default variant of threefry4x32 PRNG to use. Options: "crypto" (20 rounds) or "light"
          (2 rounds). Defaults to "light" for better performance. *)
  mutable large_models : bool;
      (** If true, use uint64 for indexing arithmetic. If false, use uint32 for indexing arithmetic.
          This affects all backends' kernel index parameters and local index variables, and gates
          the per-pool offset width (uint32 caps a pool at 4 GB; see the pool allocator). Decoupled
          in intent from element indexing within a single tensor, though both currently follow it.
      *)
}
[@@deriving sexp]

let settings =
  {
    log_level = 0;
    debug_log_from_routines = false;
    output_debug_files_in_build_directory = false;
    fixed_state_for_init = None;
    print_decimals_precision = 2;
    check_half_prec_constants_cutoff = Some (2. **. 14.);
    default_prng_variant = "light";
    large_models = false;
  }

let accessed_global_args = Hash_set.create (module String)
let str_nonempty ~f s = if String.is_empty s then None else Some (f s)
let pair a b = (a, b)

let known_config_keys =
  Set.of_list
    (module String)
    [
      (* Bootstrap keys (read before config file via read_cmdline_or_env_var directly) *)
      "suppress_welcome_message";
      "no_config_file";
      "log_config_sourcing";
      (* The preset-bundle picker (gh-ocannl-559); resolved before any ordinary key *)
      "profile";
      (* Utils.settings *)
      "log_level";
      "debug_log_from_routines";
      "output_debug_files_in_build_directory";
      "fixed_state_for_init";
      "print_decimals_precision";
      "check_half_prec_constants_cutoff";
      "default_prng_variant";
      "large_models";
      "big_models" (* deprecated alias for large_models *);
      (* Cleanup / startup *)
      "build_files_prefix";
      "clean_up_build_files_on_startup";
      "clean_up_log_files_on_startup";
      "never_capture_stdout";
      (* ppx_minidebug *)
      "snapshot_every_sec";
      "time_tagged";
      "elapsed_times";
      "location_format";
      "debug_backend";
      "hyperlink_prefix";
      "logs_print_scope_ids";
      "logs_verbose_scope_ids";
      "log_main_domain_to_stdout";
      "log_file_stem";
      "prev_run_prefix";
      "toc_entry_minimal_depth";
      "toc_entry_minimal_size";
      "toc_entry_minimal_span";
      "debug_highlights";
      "debug_highlight_pcre";
      "diff_ignore_pattern_pcre";
      "diff_max_distance_factor";
      "debug_scope_id_pairs";
      "debug_log_truncate_children";
      "debug_log_prune_upto";
      "debug_log_to_stream_files";
      (* Backends *)
      "backend";
      "prefer_backend_uniformity";
      "cc_backend_optimization_level";
      "cc_backend_compiler_command";
      "cc_backend_arch_flags";
      "cc_backend_simd_flags";
      "cc_backend_fp_contract";
      "cc_backend_probe_cache";
      "cc_backend_fast_math";
      "cc_backend_post_compile_timeout";
      "cc_backend_verify_codesign";
      "output_dlls_in_build_directory";
      "cuda_printf_fifo_size";
      "hip_printf_fifo_size";
      "hip_scratch_validation";
      "gpu_graph_capture";
      "multidev_num_devices";
      "buffer_aliasing";
      "log_buffer_aliasing";
      "memory_budget";
      "log_memory_budget";
      (* Low-level / optimization *)
      "virtualize_max_visits";
      "virtualize_max_inline_reduction";
      "virtualize_max_inline_fanin";
      "enable_device_only";
      "inline_scalar_constexprs";
      "inline_simple_computations";
      "inline_complex_computations";
      "output_prec_in_ll_files";
      "stack_threshold_in_bytes";
      (* Schedule layer (docs/proposals/schedule-ir-optops.md §6) *)
      "automatic_gpu_schedule";
      "gpu_schedule_block_size";
      "gpu_schedule_min_parallel";
      "automatic_cpu_schedule";
      "cpu_schedule_min_parallel";
      "schedule_fission";
      "schedule_log_launches";
      "schedule_log_declines";
      "legality_crosscheck";
      "cc_parallel_grid";
      "cc_parallel_chunks";
      "cc_pool_core_class";
      "cc_grid_private_bytes_cap";
      "cc_vector_bytes";
      "cc_fp16_arithmetic";
      (* Autotuning (Autotune.tune) *)
      "autotune_search";
      "autotune_beam_width";
      "autotune_rounds";
      "autotune_repeats";
      "autotune_timing";
      "autotune_cache_dir";
      "autotune_split_reduce_max_sites";
      "autotune_log";
      "tune_inline_flips";
      "tune_flip_ordering";
      "tune_flip_profit_margin";
      "tune_ship_arm";
      "strict_failure_classification";
      (* Analytic cost model (gh-ocannl-491) *)
      "autotune_keep_fraction";
      "autotune_bound_pruning";
      "autotune_calibration_file";
      "model_default_schedule";
      "model_default_placements";
      "model_default_geometry_lattice";
      "model_peak_flops";
      "model_peak_memory_bandwidth";
      (* Numerics policy (gh-ocannl-478) *)
      "tf32_matmuls";
      "narrow_compute_f32";
      "fp16_arithmetic";
      (* Identifiers and other *)
      "ll_ident_style";
      "cd_ident_style";
      "default_prec";
      "limit_constant_fill_size";
      "max_shape_error_origins";
      "checkpoint_load_mmap";
    ]

(** {2 Cache-identity classification of the configuration keys (gh-ocannl-572)} *)

(** What a configuration key does to the identity a compilation cache keys on. A cache replays a
    schedule crowned by an earlier process onto freshly lowered code, and the only thing between
    "replay" and "replay from another regime" is that identity: gh-ocannl-568 measured a
    default-flags run replaying a tf32-tuned winner at 5.9x slower than not tuning at all, because
    the numerics policy was absent from the key. So every key is classified here, and
    [test/operations/digest_completeness] fails on one that is not — adding a knob forces the
    question "does this change what a cached schedule means?" while the answer is still fresh. *)
type config_key_class =
  | Aggregate
      (** Selects values for other keys rather than acting itself; those keys carry the
          classification. *)
  | Code_borne
      (** Changes the lowered code or its placements, hence the canonical digest
          ([Ir.Schedule_cache.digest], the ["digest"] key component) that every cache key starts
          with. Nothing to add: the code {e is} the identity. *)
  | Keyed of string
      (** Invisible to the lowered code, so it has to be carried explicitly — by this named
          component of [Ir.Schedule_cache.key_components]. This is the class gh-ocannl-568 was an
          omission from. *)
  | Search_shaping
      (** Shapes which schedule a search proposes, times or crowns, but not what a crowned one means
          or how fast it then runs: a saved schedule carries its own ops and replay re-derives
          nothing from these. Two processes differing only here may find different winners; each is
          a valid winner for the other. *)
  | Execution_neutral
      (** Host-side behavior only: logging, debug artifacts, directories, validation and error
          reporting, allocation layout, launch mechanics. Nothing a kernel does depends on it. *)

(** The classification of every key in {!known_config_keys}, grouped by class and by reason. The
    reason is the reviewable part: the class alone does not say why, and why is what a reviewer has
    to check when a key is added to a group or moved between two. *)
let config_key_classification : (config_key_class * string * string list) list =
  [
    ( Aggregate,
      "picks the values other keys take; each of those carries its own classification",
      [ "profile"; "no_config_file" ] );
    ( Code_borne,
      "an optimizer decision: it changes the code that comes out of lowering",
      [
        "virtualize_max_visits";
        "virtualize_max_inline_reduction";
        "virtualize_max_inline_fanin";
        "enable_device_only";
        "inline_scalar_constexprs";
        "inline_simple_computations";
        "inline_complex_computations";
        "memory_budget";
      ] );
    ( Code_borne,
      "it changes the assignments the front end builds, hence the code they lower to",
      [ "default_prec"; "default_prng_variant"; "limit_constant_fill_size" ] );
    ( Code_borne,
      "it decides a tensor node's placement class, which the canonical digest carries: identical \
       code over [Local] scratch and over an [On_device] buffer generates different kernels",
      [ "stack_threshold_in_bytes" ] );
    ( Keyed "backend",
      "it decides which backend compiles, and the backend name is that component",
      [ "backend" ] );
    ( Keyed "numerics",
      "the numerics policy is consulted at codegen and by the autotune tile-shape choice, never in \
       the lowered code (gh-ocannl-568)",
      [ "tf32_matmuls"; "narrow_compute_f32"; "fp16_arithmetic" ] );
    ( Keyed "pool",
      "it decides the worker pool timings execute on, and CPU crowns do not transfer across pools \
       (gh-ocannl-530)",
      [ "cc_pool_core_class" ] );
    ( Keyed "codegen",
      "a cc-backend codegen knob: rendering or compiling a kernel reads it, after the lowered code \
       the digest names",
      [
        "cc_backend_optimization_level";
        "cc_backend_compiler_command";
        "cc_backend_arch_flags";
        "cc_backend_simd_flags";
        "cc_backend_fp_contract";
        "cc_backend_fast_math";
        "cc_vector_bytes";
        "cc_fp16_arithmetic";
        "cc_parallel_grid";
        "cc_parallel_chunks";
        "cc_grid_private_bytes_cap";
      ] );
    ( Keyed "codegen",
      "a codegen knob whose effect is not a property of the lowered code. The two debug gates bite \
       only at log_level > 1, so what the tag hashes is the effective predicates -- \
       [Utils.debug_log_from_routines], which rewrites the kernel and disables the parallel-grid, \
       vectorized and mma renderings, for every backend; [Utils.with_runtime_debug], which \
       switches the CUDA and HIP compilers to debug compilation, in those two backends' own tags, \
       since no other compiler reads it. That is also why log_level belongs here, without an \
       ordinary verbosity bump churning cache keys. prefer_backend_uniformity does not pick a \
       backend: it picks how the C-family backends spell their logging expressions, so it is \
       hashed here only once logging actually reaches the kernel -- that is its only effect on \
       emitted code, since gh-ocannl-647 made HIP's guarded float-to-fp8 conversion unconditional \
       rather than a thing this flag chooses",
      [
        "large_models";
        "big_models";
        "log_level";
        "debug_log_from_routines";
        "debug_log_to_stream_files";
        "output_debug_files_in_build_directory";
        "prefer_backend_uniformity";
      ] );
    ( Keyed "codegen",
      "it changes the emitted code or the mechanics a search measures in, without being a property \
       of the lowered code: an aliasing candidate's kernel parameter drops its [restrict] \
       qualifier (the liveness planner may overlap its bytes, gh-ocannl-489), and GPU graph \
       capture fires only for multi-segment routines, so it moves a fissioned candidate's launch \
       overhead relative to a whole-routine one",
      [ "buffer_aliasing"; "gpu_graph_capture" ] );
    ( Search_shaping,
      "it defines the untuned default pipeline, which a search seeds from and reports against; a \
       cached winner carries its own ops and replays without consulting it \
       ([Schedule.default_schedule_fingerprint] records these for the [default_ms] diagnostic)",
      [
        "automatic_gpu_schedule";
        "gpu_schedule_block_size";
        "gpu_schedule_min_parallel";
        "automatic_cpu_schedule";
        "cpu_schedule_min_parallel";
        "schedule_fission";
      ] );
    ( Search_shaping,
      "it steers the search: how wide, how long, what is proposed, what is pruned, how candidates \
       are ranked, and when a candidate's failure ends the search",
      [
        "autotune_search";
        "autotune_beam_width";
        "autotune_rounds";
        "autotune_repeats";
        "autotune_split_reduce_max_sites";
        "autotune_keep_fraction";
        "autotune_bound_pruning";
        "autotune_calibration_file";
        "model_default_schedule";
        "model_default_placements";
        "model_default_geometry_lattice";
        "model_peak_flops";
        "model_peak_memory_bandwidth";
        "strict_failure_classification";
      ] );
    ( Keyed "timing",
      "it picks the OBJECTIVE a candidate is timed against -- the latency of a lone dispatch, or \
       what the kernel sustains under queue depth -- and the two crown different candidates, \
       measured (gh-ocannl-755). Not Search_shaping, which is where it started and where the \
       neighbouring autotune_* knobs correctly sit: those change how carefully the SAME quantity \
       is measured, so either process's winner answers the other's question, whereas an entry \
       crowned under isolated timing is the answer to a question a queued search did not ask -- \
       and its stored best_ms, baseline_ms and mma_best_ms are readings of a different quantity, \
       which a replay copies into the reading process's report under that process's label. Left \
       unkeyed, a warm cache would also defeat the new default outright, replaying \
       isolated-crowned winners forever (Codex P1 on PR #512)",
      [ "autotune_timing" ] );
    ( Search_shaping,
      "it makes the tuner try alternative inlining decisions; each alternative is a different \
       program and keys on its own digest",
      [ "tune_inline_flips"; "tune_flip_ordering"; "tune_flip_profit_margin" ] );
    ( Search_shaping,
      "it decides which of the two searched placement arms ships, overriding the measured \
       comparison rather than changing either arm: each arm is a different program keyed on its \
       own digest, each arm's crown is cached under that digest either way, and a schedule crowned \
       under one setting is a valid crown under the other (gh-ocannl-638)",
      [ "tune_ship_arm" ] );
    ( Execution_neutral,
      "startup and configuration-sourcing chatter",
      [ "suppress_welcome_message"; "log_config_sourcing"; "never_capture_stdout" ] );
    ( Execution_neutral,
      "debug artifacts and where they go: the files are written beside the run, the kernels are \
       what they would have been",
      [
        "output_dlls_in_build_directory";
        "build_files_prefix";
        "clean_up_build_files_on_startup";
        "clean_up_log_files_on_startup";
        "output_prec_in_ll_files";
        "autotune_cache_dir";
        "autotune_log";
        "log_buffer_aliasing";
        "log_memory_budget";
        "schedule_log_launches";
        "schedule_log_declines";
      ] );
    ( Execution_neutral,
      "ppx_minidebug logging of the library's own execution",
      [
        "snapshot_every_sec";
        "time_tagged";
        "elapsed_times";
        "location_format";
        "debug_backend";
        "hyperlink_prefix";
        "logs_print_scope_ids";
        "logs_verbose_scope_ids";
        "log_main_domain_to_stdout";
        "log_file_stem";
        "prev_run_prefix";
        "toc_entry_minimal_depth";
        "toc_entry_minimal_size";
        "toc_entry_minimal_span";
        "debug_highlights";
        "debug_highlight_pcre";
        "diff_ignore_pattern_pcre";
        "diff_max_distance_factor";
        "debug_scope_id_pairs";
        "debug_log_truncate_children";
        "debug_log_prune_upto";
      ] );
    ( Execution_neutral,
      "a check that can refuse or report, but never changes what is emitted",
      [
        "check_half_prec_constants_cutoff";
        "legality_crosscheck";
        "hip_scratch_validation";
        "cc_backend_verify_codesign";
        "cc_backend_post_compile_timeout";
        "max_shape_error_origins";
      ] );
    ( Execution_neutral,
      "identifier spelling in generated code and in printouts",
      [ "ll_ident_style"; "cd_ident_style"; "print_decimals_precision" ] );
    ( Execution_neutral,
      "host-side execution mechanics: where buffers sit, how many devices a scheduler opens, how \
       launches are issued, how initialization is seeded, how checkpoint payloads reach host \
       memory. The kernels are unchanged, and every candidate of a search meets the same mechanics \
       as every other",
      [
        "multidev_num_devices";
        "cuda_printf_fifo_size";
        "hip_printf_fifo_size";
        "fixed_state_for_init";
        "checkpoint_load_mmap";
      ] );
    ( Execution_neutral,
      "memoization of toolchain probes: it changes how long resolving a setting takes, not what it \
       resolves to -- and the resolved values are what the codegen tag records",
      [ "cc_backend_probe_cache" ] );
  ]

(** The class and the reason recorded for [key], or [None] when it is unclassified (which
    [test/operations/digest_completeness] rejects for every known key). *)
let classify_config_key key =
  List.find_map config_key_classification ~f:(fun (cls, why, keys) ->
      if List.mem keys key ~equal:String.equal then Some (cls, why) else None)

let bool_of_config_string ~arg_name s =
  match String.lowercase @@ String.strip s with
  | "true" | "1" -> true
  | "false" | "0" -> false
  | _ -> invalid_arg @@ "ocannl_" ^ arg_name ^ " setting should be a boolean; found: " ^ s

(** Whether to print where each configuration setting comes from (commandline, environment, config
    file, or the hard-coded default). It gates the logging of the config-reading functions
    themselves, so it is bootstrapped directly rather than via {!get_global_arg}: the initial value
    only reflects the commandline and the environment, and the config file setting is applied once
    the config file is read. Can also be set programmatically, e.g. to trace configs a test reads.

    Opt-in (gh-ocannl-595): the trace is a debugging tool -- some eighty lines on a run that reads a
    config file -- and it shares stderr with the unknown-config-key warning, the one startup message
    that means the user made a mistake. Enabling it traces every key, without a second dependence on
    [log_level]: it says "tell me where the configuration came from", and answering that for one key
    is not a useful reading of it. *)
let log_config_sourcing = ref false

(** The environment spelling of a config key: [OCANNL_] followed by the key in uppercase, and
    nothing else. The prefix is mandatory here (unlike on the commandline and in a config file) so
    that OCANNL does not read an unrelated tool's variable.

    ONE spelling (gh-ocannl-652). A lowercase [ocannl_<key>] used to be read as well, and won over
    the uppercase one besides -- which [test/operations/profiles/dune] had to document as a real
    precedence trap. It bought nothing anyone could name: no caller in this repository spelled a
    variable that way and no documentation recommended it, while every dune rule that has to declare
    the ambient variables it is invalidated by paid two lines per key, 228 of them, forever
    (gh-ocannl-628). The reserved namespaces this file introduced alongside them ([OCANNL_TOOL_...],
    [OCANNL_LOG_LEVEL_<MODULE>]) were uppercase-only from the start, for the same reason: uppercase
    is what the shell convention is unambiguous about.

    Dropping a spelling someone may have exported is a silent demotion, so the spellings this file
    no longer reads are not merely ignored: {!classify_env_var} reports one as
    {!Env_unread_spelling}, and the check at the foot of this file makes it FATAL when it names a
    known key. See {!unread_env_vars}.

    Dashes are not a spelling either: [ocannl-log_level] and its uppercase form were dropped in
    gh-ocannl-605, documented nowhere and used by nobody, while costing every such dune rule four
    spellings per key -- of which the natural-looking all-dashed [ocannl-log-level] was never one.
    Dashes remain idiomatic on the commandline, where {!cmdline_var_names} accepts them. *)
let env_var_name n = "OCANNL_" ^ String.uppercase n

(** The [ocannl]-prefixed environment namespaces that are deliberately NOT configuration, so that a
    name in one of them is never reported as a misspelt key (gh-ocannl-629).

    Two of them, and each is a namespace someone else reads:

    - [ocannl_tool_…] belongs to this repository's own tooling and test harnesses --
      [tools/sweep.sh]'s state directory, [tools/test-run.sh]'s time cap, the hardware hook in
      [test/operations/test_cpu_topology.ml]. None of them is a library setting, and every one of
      them is exported into the environment of processes that link OCANNL.
    - [ocannl_log_level_…] is read by ppx_minidebug at PREPROCESSING time: the per-module tracing
      gates, one [%%global_debug_log_level_from_env_var] at the top of [tensor/row.ml] and of its
      eighteen siblings. The name after the prefix is a module, not a setting, and it is consumed
      before this file's initialization exists to have an opinion.

    A reserved prefix rather than a list of exempt names: a list is a second place to update when a
    tool grows a variable, nothing forces the update, and the failure mode is a warning the reader
    cannot act on -- which trains people to ignore the warning that matters, the exact outcome
    gh-ocannl-595 was fixed to prevent. A prefix costs the tooling a rename once and nothing
    thereafter. *)
let env_var_reserved_prefixes = [ "ocannl_tool_"; "ocannl_log_level_" ]

(** Whether the platform resolves environment variable names case-insensitively. On Windows
    [ocannl_Log_level] and [OCANNL_LOG_LEVEL] are ONE variable, so a spelling this file would call
    unread there is in fact read, and reporting it would be wrong on that platform only.

    Native Windows only, NOT Cygwin (Codex P1 on PR #389): a Cygwin runtime sets [Sys.cygwin] rather
    than [Sys.win32], and its POSIX environment is case-SENSITIVE -- [getenv "OCANNL_BACKEND"] does
    not find [ocannl_backend] there. Folding case on that runtime would call the lowercase spelling
    read while {!read_env_var} found nothing, which is the silent demotion this whole check exists
    to prevent, delivered by the check itself. What matters is what the runtime's own [getenv] does,
    not whether the host kernel is Windows.

    The DEFAULT of a parameter, rather than the only answer (gh-ocannl-661). Baked in from
    [Sys.win32] at definition, it made the Windows reading of {!classify_env_var} unreachable on
    every other host: Linux and macOS CI could not execute that branch, so a change collapsing it
    stayed green on the ordinary PR path and surfaced days later on a scheduled Windows run -- which
    is how the dashed spellings came to be classified as read there in the first place. Every caller
    in this file still takes the platform's answer; what the parameter buys is that
    [test/operations/config_var_spellings] pins BOTH readings on EVERY host. *)
let env_names_case_insensitive = Stdlib.Sys.win32

(** What an environment variable name is, to OCANNL. The classification is shared by the startup
    warning at the foot of this file and by [test/operations/env_var_deps], which asks it of every
    [(env_var …)] a dune file declares -- so a name a rule tracks and a name a run warns about are
    decided by one function. *)
type env_var_class =
  | Env_not_addressed  (** not [ocannl]-prefixed: someone else's variable entirely *)
  | Env_reserved of string  (** in a reserved non-configuration namespace, named by its prefix *)
  | Env_config_key of string  (** a spelling {!read_env_var} reads, of that key *)
  | Env_unread_spelling of string
      (** a known key, spelled in a way nothing reads: dashed, or not fully uppercase where case
          matters. A configuration ERROR rather than a warning -- see {!unread_env_vars}. *)
  | Env_unread_reserved of string
      (** in a reserved namespace, in a casing its reader does not consult *)
  | Env_unknown_key of string  (** addressed to the configuration, naming no key *)

(** Which family a name is in, and then -- for both families alike -- whether THIS spelling of it is
    one its reader actually consults. The second question is the one that carries the feature: an
    unread spelling is invisible in exactly the way a typo is, and answering it for keys while
    waving reserved names through would suppress the warning precisely where the name looks most
    like it should work (Codex P2 on PR #371). Each family has its own reader, so each answers it
    its own way -- {!env_var_name} for a key, and uppercase for a reserved name, which is what the
    shell scripts and the [%%global_debug_log_level_from_env_var] arguments spell. Since
    gh-ocannl-652 those two answers are the same one: uppercase, everywhere under the prefix.

    Case, and only case, collapses on Windows: the environment folds [ocannl_backend] onto
    [OCANNL_BACKEND] there, so the same name is read on one platform and not the other. It does NOT
    fold punctuation -- [getenv "OCANNL_LOG_LEVEL"] finds nothing set as [ocannl-log_level] on any
    platform -- so {!same_env_name} compares the whole name rather than answering "true" to every
    candidate wherever the environment is case-insensitive (Codex P1 on PR #389). The earlier form
    classified every dashed spelling as read on Windows, which is where the dashed spellings this
    file's goldens lean on would have stopped being unread.

    A key's separators are normalized before the known-key lookup, and only for it:
    [OCANNL_PRINT-DECIMALS-PRECISION] names [print_decimals_precision] recognizably, so it is an
    unread SPELLING of a known key -- fatal, per {!unread_env_vars} -- rather than an unknown key
    that would warn and let the run continue on the default (Codex P2 on PR #389). The dashes are
    idiomatic on the commandline, where {!cmdline_var_names} reads exactly this shape, which is why
    someone writes one here. What the normalization does not do is make it a spelling: the canonical
    name it is compared against is still the undashed one. *)

(** Whether two environment variable names denote the same variable on this platform: case-folded
    where the environment is, and otherwise exactly. [case_insensitive] defaults to
    {!env_names_case_insensitive} and is passed explicitly only by the test that pins both readings
    on every host -- see there for why the platform's answer is not the only one reachable. *)
let same_env_name ?(case_insensitive = env_names_case_insensitive) a b =
  if case_insensitive then String.Caseless.equal a b else String.equal a b

let classify_env_var ?(case_insensitive = env_names_case_insensitive) name =
  let same_env_name = same_env_name ~case_insensitive in
  let lower = String.lowercase name in
  let addressed =
    List.find_map [ "ocannl_"; "ocannl-" ] ~f:(fun prefix -> String.chop_prefix lower ~prefix)
  in
  match addressed with
  | None -> Env_not_addressed
  | Some key -> (
      match
        List.find env_var_reserved_prefixes ~f:(fun prefix -> String.is_prefix lower ~prefix)
      with
      (* A reserved name has no canonical spelling of its own -- the module or tool after the prefix
         names it -- so the question is whether THIS name is the uppercase its reader spells.
         Vacuously yes on Windows, which is the right answer there. *)
      | Some prefix ->
          if same_env_name name (String.uppercase name) then Env_reserved prefix
          else Env_unread_reserved (String.uppercase prefix)
      | None -> (
          let known =
            List.find
              [ key; String.tr key ~target:'-' ~replacement:'_' ]
              ~f:(Set.mem known_config_keys)
          in
          match known with
          (* Reported under the name as written, the normalization having recognized nothing. *)
          | None -> Env_unknown_key key
          | Some key ->
              if same_env_name name (env_var_name key) then Env_config_key key
              else Env_unread_spelling key))

(** Every environment variable that addresses OCANNL's configuration and that nothing reads: the
    name, whether it is fatal, and the reason, in the order the names sort.

    {b Fatal} for a known key spelled in a way nothing reads, and a warning for everything else
    (gh-ocannl-652). The distinction is whether a VALUE that was meant to decide something is being
    dropped: [ocannl_backend=cuda] names a real key, so somebody wrote it to choose a backend, and
    silently running on the default instead is the failure mode this whole check exists to prevent
    -- it is also what dropping the lowercase spelling would otherwise have inflicted on anyone who
    had exported one. A name that matches no key ([OCANNL_BACKEDN]) never decided anything to begin
    with, and a lowercase name in a reserved namespace belongs to a tool rather than to the
    configuration; both stay warnings, as they were.

    An EMPTY value is not reported at all, in either class: "" counts as unset at every source (see
    {!read_env_var}), a dune rule clears a variable by setting it empty, and a launcher expanding
    [$OCANNL_BACKEND] with nothing set must not thereby abort the run.

    Separate from the loop that consumes it (at the foot of this file) so that the walk is a value
    rather than an effect: what is reported is then a list something else -- a test, a tool refusing
    to run under a misconfigured environment -- can also ask for. Sorted so that a stream capturing
    the messages does not depend on the order the C library hands the environment over. *)
let unread_env_vars () =
  Unix.environment () |> Array.to_list
  |> List.filter_map ~f:(fun binding ->
      match String.lsplit2 binding ~on:'=' with
      | Some (_, "") -> None
      | Some (name, _) -> Some name
      | None -> Some binding)
  |> List.dedup_and_sort ~compare:String.compare
  |> List.filter_map ~f:(fun name ->
      match classify_env_var name with
      | Env_not_addressed | Env_reserved _ | Env_config_key _ -> None
      | Env_unknown_key _ -> Some (name, false, "names no configuration key")
      | Env_unread_spelling key ->
          Some
            ( name,
              true,
              "is not a spelling OCANNL reads; the environment spelling of " ^ key ^ " is "
              ^ env_var_name key )
      | Env_unread_reserved prefix ->
          Some
            ( name,
              false,
              "is not a spelling anything reads; " ^ prefix ^ " names are read in uppercase only" ))

(* The same check on the other silent source (gh-ocannl-629). Of the three ways to set a
   configuration key, the environment was the one that said nothing about a mistake: a config file
   names the unknown key it holds (the warning at {!config_file_args}, which
   [test/operations/startup_streams] pins), the commandline warns just above, and
   `OCANNL_BACKEDN=cuda dune runtest` ran the whole suite on the default backend and reported
   success. It is also the source people reach for in CI and in one-off shell invocations, where a
   typo has no reviewer.

   Reading it costs one walk over the environment, and what makes the walk safe is
   {!env_var_reserved_prefixes}: without a namespace for the variables that are addressed to OCANNL
   without being configuration -- the tooling's, and ppx_minidebug's per-module gates -- this check
   would fire on every run of every OCANNL executable under `tools/sweep.sh`.

   Since gh-ocannl-652 the strongest case is fatal rather than merely reported, and the exit is here
   rather than inside {!unread_env_vars} so that the walk stays a value: every offending name is
   printed before the process ends, so one run fixes the whole environment instead of one variable
   per attempt. `Stdlib.exit` rather than an exception, this being the user's mistake and not a bug
   to hand a backtrace for -- the same shape as the fatal unknown profile name, whose `invalid_arg`
   predates the policy.

   Placed HERE, immediately after the walk it consumes and before every configuration-driven effect
   this file performs, rather than at the foot of the file where it started (Codex P2 on PR #389). A
   run that is going to abort must abort before it has DONE anything: the startup cleanup below
   deletes `log_files/` and `build_files/`, and it reads whether to do so through the ordinary
   config path -- so a rejected `ocannl_clean_up_build_files_on_startup=false` was ignored in favour
   of the default, the artifacts the user asked to keep were deleted, and only then did the run stop
   to say the spelling was wrong. The check needs nothing but the environment and the key set, so
   nothing forced it to be late. The commandline's unknown-argument warning stays at the foot: it is
   not fatal and destroys nothing, so its only cost is that the two warnings no longer share a
   neighbourhood in the output. *)
let () =
  let fatal = ref false in
  List.iter (unread_env_vars ()) ~f:(fun (name, is_fatal, reason) ->
      Stdio.eprintf "OCANNL %s: environment variable %S %s\n%!"
        (if is_fatal then "error" else "warning")
        name reason;
      fatal := !fatal || is_fatal);
  if !fatal then (
    Stdio.eprintf
      "OCANNL: aborting -- a configuration key set under a spelling nothing reads decides nothing, \
       and continuing would run on the default as if it had never been set\n\
       %!";
    Stdlib.exit 1)

(** The commandline spellings of a config key, up to the value separator: the [ocannl_]-qualified
    ones -- then, unless [qualified_only], the prefix-free ones.

    Every spelling carries a leading dash, one or two. A bare [ocannl_log_level=1] used to be read
    as well, and gh-ocannl-605 dropped it (Codex P2 on PR #363): a bare argument is a host
    application's positional, and an OCANNL-linked tool taking a path -- [ocannl_config] is the
    obvious one -- was one key name away from having it eaten. It also left the unknown-argument
    warning with a spelling it could not diagnose, since a bare argument is exactly what it must NOT
    claim to know about.

    The dashing is two independent choices, not one per separator: the prefix separator dashes on
    its own, and the key's own separators dash TOGETHER. For [log_level] that is [ocannl_log_level],
    [ocannl_log-level], [ocannl-log_level] and [ocannl-log-level] (each in lowercase and in
    uppercase, each with one leading dash or two). A key dashed halfway
    ([ocannl-print_decimals-precision]) is not a spelling -- and, since {!cmdline_var_prefixes} is
    also what the unknown-argument warning matches, it is reported as unknown rather than silently
    ignored, which is what makes the narrower contract safe to have (Codex P2 on PR #363).
    Enumerating every separator independently is the alternative, at 2^separators spellings per key;
    nothing asked for it.

    [qualified_only] exists because OCANNL is a library: it scans the host executable's [Sys.argv],
    so a prefix-free key claims an application's own option of that name. That is tolerable for keys
    nobody else would spell ([--virtualize_max_visits]) and not for [--profile], which is a common
    application flag and which OCANNL treats as fatal when it does not name a known bundle -- a host
    passing [--profile=prod] would die during module initialization (Codex P2 on PR #291). *)
let cmdline_var_names ?(qualified_only = false) n =
  let n_dash = String.tr ~target:'_' ~replacement:'-' n in
  let keys = if String.equal n n_dash then [ n ] else [ n; n_dash ] in
  (* Prefixed commandline variants first (backward compat), then prefix-free. *)
  let qualified =
    List.concat_map [ "ocannl_"; "ocannl-" ] ~f:(fun prefix ->
        List.concat_map keys ~f:(fun k ->
            let name = prefix ^ k in
            List.concat_map [ name; String.uppercase name ] ~f:(fun n -> [ "-" ^ n; "--" ^ n ])))
  in
  let unqualified =
    if qualified_only then [] else List.concat_map keys ~f:(fun k -> [ "--" ^ k; "-" ^ k ])
  in
  qualified @ unqualified

(** What an argument setting [n] begins with: a spelling from {!cmdline_var_names} followed by the
    value separator, which is [_], [-], [=] or nothing at all. Whatever remains of the argument is
    the value.

    This is the single source of truth for "an argument OCANNL reads", and the unknown-argument
    warning at the bottom of this file matches against it rather than parsing arguments a second
    way. It used to parse: split on [=], normalize every dash to an underscore, look the result up
    -- which accepted spellings the reader ignored (`--ocannl-log-level=1`, before gh-ocannl-605
    made it real) and rejected ones the reader honoured (`--ocannl_log_level_1`, whose separator is
    not an [=], so the key came out as `log_level_1`). Both directions are silent contradictions:
    one applies nothing while saying nothing, the other applies the setting while warning that it is
    unknown. One table cannot disagree with itself. *)
let cmdline_var_prefixes ?qualified_only n =
  List.concat_map (cmdline_var_names ?qualified_only n) ~f:(fun n ->
      [ n ^ "_"; n ^ "-"; n ^ "="; n ])

(** The registry-independent grammatical shape of a configuration token. This is deliberately a
    lexer for scanners and documentation tooling, not another configuration lookup: the runtime
    readers below still resolve a literal key through {!cmdline_var_prefixes}, {!env_var_name}, and
    the registered key set.

    A documentation assignment has no namespace prefix to distinguish it from another API or from
    mathematical notation, so its bare name must look like an OCANNL key: lowercase snake case with
    at least one underscore. One-word keys remain discoverable under the explicit [ocannl_] spelling
    (for example [ocannl_backend=cc]) and through the unambiguous command-line/environment forms. *)
type config_token_shape =
  | Command_line_token
  | Environment_assignment_token
  | Documentation_assignment_token

type config_token = { token_shape : config_token_shape; token_key : string }

(** Prefixes that can begin a qualified command-line token. Exposed with the parser so a scanner can
    discover candidate tokens without reconstructing the command-line spelling grammar. *)
let config_token_command_line_prefixes =
  let sentinel = "configtokengrammarkey" in
  cmdline_var_names ~qualified_only:true sentinel
  |> List.filter_map ~f:(fun name ->
      let suffix =
        if String.equal name (String.uppercase name) then String.uppercase sentinel else sentinel
      in
      String.chop_suffix name ~suffix)
  |> List.dedup_and_sort ~compare:String.compare

(** Classify one observed token without consulting {!known_config_keys}. Command-line and
    environment assignments are always recognized; [documentation:true] additionally admits the
    narrowed bare-assignment grammar documented on {!config_token_shape}. A command-line token must
    use one uniform case and one separator style, as {!cmdline_var_names} does; its key is the
    maximal name before [=], normalized to underscores. Alternate value separators make shorter
    readings inherently ambiguous without a key registry; consumers that need one of those readings
    keep an explicit site judgment. *)
let parse_config_token ?(documentation = false) token =
  let config_token_name_char c =
    Char.is_alpha c || Char.is_digit c || Char.equal c '_' || Char.equal c '-'
  in
  let command_line_config_token token =
    let name = Option.value_map (String.lsplit2 token ~on:'=') ~default:token ~f:fst in
    List.find_map config_token_command_line_prefixes ~f:(fun prefix ->
        Option.bind (String.chop_prefix name ~prefix) ~f:(fun raw_key ->
            let uppercase = String.equal prefix (String.uppercase prefix) in
            let uniform_case =
              String.for_all raw_key ~f:(fun c ->
                  (not (Char.is_alpha c))
                  || if uppercase then Char.is_uppercase c else Char.is_lowercase c)
            in
            let uniform_separator =
              not (String.contains raw_key '_' && String.contains raw_key '-')
            in
            if
              String.is_empty raw_key
              || (not (String.for_all raw_key ~f:config_token_name_char))
              || (not uniform_case) || not uniform_separator
            then None
            else
              let key = String.lowercase raw_key |> String.tr ~target:'-' ~replacement:'_' in
              Some { token_shape = Command_line_token; token_key = key }))
  in
  let environment_config_token token =
    Option.bind (String.lsplit2 token ~on:'=') ~f:(fun (name, _value) ->
        Option.bind (String.chop_prefix name ~prefix:"OCANNL_") ~f:(fun raw_key ->
            Option.some_if
              ((not (String.is_empty raw_key))
              && String.for_all raw_key ~f:(fun c ->
                  Char.is_uppercase c || Char.is_digit c || Char.equal c '_'))
              { token_shape = Environment_assignment_token; token_key = String.lowercase raw_key }))
  in
  let documentation_config_token token =
    Option.bind (String.lsplit2 token ~on:'=') ~f:(fun (raw_name, _value) ->
        let name = String.strip raw_name in
        let config_looking =
          (not (String.is_empty name))
          && Char.is_lowercase name.[0]
          && String.for_all name ~f:(fun c ->
              Char.is_lowercase c || Char.is_digit c || Char.equal c '_')
          && String.contains name '_'
        in
        let key = Option.value (String.chop_prefix name ~prefix:"ocannl_") ~default:name in
        Option.some_if
          (config_looking && not (String.is_empty key))
          { token_shape = Documentation_assignment_token; token_key = key })
  in
  let token = String.strip token in
  Option.first_some (command_line_config_token token)
    (Option.first_some (environment_config_token token)
       (if documentation then documentation_config_token token else None))

(* Keys whose prefix-free command-line spellings are never claimed: common application flags a host
   executable is likely to own ({!cmdline_var_names}' [qualified_only] doc; Codex P2 on PR #291).
   The single source of the policy — {!read_cmdline_var}'s default and {!cmdline_arg_is_config_key}
   both derive from it, so a spelling the resolver would ignore is never mistaken for a claimed
   argument (gh-ocannl-578). *)
let qualified_only_config_keys = Set.of_list (module String) [ "profile" ]

(** The commandline sublevel of {!get_global_arg}: returns the setting's value and the [Sys.argv]
    element it came from. Pure -- the sourcing log lives at the resolution seam, which is the only
    place that knows which sublevel actually won.

    [qualified_only] defaults per key from {!qualified_only_config_keys}, so a caller need not
    remember which keys renounce their prefix-free spellings. *)
let read_cmdline_var ?qualified_only n =
  let qualified_only =
    Option.value qualified_only ~default:(Set.mem qualified_only_config_keys n)
  in
  let cmd_variants = cmdline_var_prefixes ~qualified_only n in
  Array.find_map Stdlib.Sys.argv ~f:(fun arg ->
      List.find_map cmd_variants ~f:(fun p ->
          Option.some_if (String.is_prefix ~prefix:p arg)
            (String.drop_prefix arg (String.length p), arg)))

(** Whether a raw command-line argument addresses a known configuration key under {e any} spelling
    {!read_cmdline_var} accepts — prefixed or prefix-free, dashed or underscored, any separator. For
    executables that parse their own flags (tools/): such an argument belongs to the config
    machinery and should be passed over rather than rejected as unknown, while an argument matching
    no known key under any spelling can still be flagged as a probable typo. *)
let cmdline_arg_is_config_key arg =
  Set.exists known_config_keys ~f:(fun k ->
      let qualified_only = Set.mem qualified_only_config_keys k in
      List.exists (cmdline_var_prefixes ~qualified_only k) ~f:(fun p ->
          String.is_prefix arg ~prefix:p))

(** The environment sublevel of {!get_global_arg}: returns the setting's value and the variable it
    came from. An empty value counts as unset. *)
let read_env_var n =
  let env_n = env_var_name n in
  match Option.(join @@ map (Stdlib.Sys.getenv_opt env_n) ~f:(str_nonempty ~f:(pair env_n))) with
  | None | Some (_, "") -> None
  | Some (env_n, result) -> Some (result, env_n)

(** The bootstrap reader: the few keys that are consulted before the config file exists (and hence
    before profiles are resolved) come from the commandline or the environment only.

    Silent, deliberately. These keys are read before {!log_config_sourcing} is resolved -- one of
    the reads IS that resolution -- so nothing here can know whether anyone asked for a trace, and
    each is read more than once besides (three call sites consult [suppress_welcome_message]). Their
    provenance is reported once, in full, by [bootstrap_config_report] below. *)
let read_cmdline_or_env_var n =
  match read_cmdline_var n with
  | Some (result, _arg) -> Some result
  | None -> Option.map (read_env_var n) ~f:fst

(* Originally from the library core.filename_base. *)
let filename_parts filename =
  let rec loop acc filename =
    match (Stdlib.Filename.dirname filename, Stdlib.Filename.basename filename) with
    | ("." as base), "." -> base :: acc
    | ("/" as base), "/" -> base :: acc
    | disk, base when String.is_suffix disk ~suffix:":\\" -> disk :: base :: acc
    | rest, dir -> loop (dir :: acc) rest
  in
  loop [] filename

(* Originally from the library core.filename_base. *)
let filename_of_parts = function
  | [] -> invalid_arg "Utils.filename_of_parts: empty parts list"
  | root :: rest -> List.fold rest ~init:root ~f:Stdlib.Filename.concat

let log_config_sourcing_arg = read_cmdline_or_env_var "log_config_sourcing"

let () =
  Option.iter log_config_sourcing_arg ~f:(fun v ->
      log_config_sourcing := bool_of_config_string ~arg_name:"log_config_sourcing" v)

(** Parses the [ocannl_config] syntax: one [key=value] per line, [#] and [~~] lines are comments,
    empty values mean "unset", the [ocannl_] key prefix is optional and keys are case-insensitive.
    Shared by the config file and by the embedded profile payloads (which are literally partial
    config files); [source] names the origin in error messages. *)
let parse_config_lines ~source lines =
  lines
  |> List.filter ~f:(fun l ->
      not (String.is_prefix ~prefix:"~~" l || String.is_prefix ~prefix:"#" l))
  |> List.map ~f:(String.split ~on:'=')
  |> List.filter_map ~f:(function
    | [] -> None
    | [ s ] when String.is_empty (String.strip s) -> None
    | key :: [ v ] ->
        let key =
          String.(lowercase @@ strip ~drop:(fun c -> equal_char '-' c || equal_char ' ' c) key)
        in
        let key =
          if String.is_prefix key ~prefix:"ocannl" then
            String.drop_prefix key 6 |> String.strip ~drop:(equal_char '_')
          else key
        in
        str_nonempty ~f:(pair key) v
    | l ->
        failwith @@ "OCANNL: invalid syntax in " ^ source
        ^ ", should have a single '=' on each non-empty line, found: " ^ String.concat l)

let config_table_of_lines ~source lines =
  match Hashtbl.of_alist (module String) (parse_config_lines ~source lines) with
  | `Ok h -> h
  | `Duplicate_key key -> failwith @@ "OCANNL: duplicate key in " ^ source ^ ": " ^ key

let config_file_args =
  let suppress_welcome_message () =
    Option.value_map ~default:false ~f:Bool.of_string
    @@ read_cmdline_or_env_var "suppress_welcome_message"
  in
  match read_cmdline_or_env_var "no_config_file" with
  | None | Some "false" ->
      let read = Stdio.In_channel.read_lines in
      let fname, config_lines =
        let rev_dirs = List.rev @@ filename_parts @@ Stdlib.Sys.getcwd () in
        let rec find_up = function
          | [] ->
              if not (suppress_welcome_message ()) then
                Stdio.eprintf
                  "\nWelcome to OCANNL! No ocannl_config file found along current path.\n%!";
              ("", [])
          | _ :: tl as rev_dirs -> (
              let fname = filename_of_parts (List.rev @@ ("ocannl_config" :: rev_dirs)) in
              try (fname, read fname) with Sys_error _ -> find_up tl)
        in
        find_up rev_dirs
      in
      let result = config_table_of_lines ~source:("the config file " ^ fname) config_lines in
      if String.length fname > 0 then
        Hashtbl.iter_keys result ~f:(fun key ->
            if not (Set.mem known_config_keys key) then
              Stdio.eprintf "OCANNL warning: unknown config key %S in %s\n%!" key fname);
      if
        String.length fname > 0
        && (not (suppress_welcome_message ()))
        && not
             (Option.value_map ~default:false ~f:Bool.of_string
             @@ Hashtbl.find result "suppress_welcome_message")
      then Stdio.eprintf "\nWelcome to OCANNL! Reading configuration defaults from %s.\n%!" fname;
      result
  | Some _ ->
      if not (suppress_welcome_message ()) then
        Stdio.eprintf "\nWelcome to OCANNL! Configuration defaults file is disabled.\n%!";
      Hashtbl.create (module String)

let () =
  (* The commandline and the environment take precedence, and were already applied above. *)
  if Option.is_none log_config_sourcing_arg then
    Option.iter (Hashtbl.find config_file_args "log_config_sourcing") ~f:(fun v ->
        log_config_sourcing := bool_of_config_string ~arg_name:"log_config_sourcing" v)

(** {2 Configuration profiles (gh-ocannl-559)} *)

(** The source levels a setting can come from, in decreasing priority. Each level splits into two
    sublevels: the keys stated explicitly at that level, then the payload of a profile {e picked} at
    that level -- so a specific setting always beats an aggregate one of equal immediacy, and a
    profile named on the commandline still overrides an exhaustive config file. *)
type config_level = Cmdline_level | Env_level | Config_file_level

let equal_config_level l1 l2 =
  match (l1, l2) with
  | Cmdline_level, Cmdline_level | Env_level, Env_level | Config_file_level, Config_file_level ->
      true
  | (Cmdline_level | Env_level | Config_file_level), _ -> false

let describe_config_level = function
  | Cmdline_level -> "the commandline"
  | Env_level -> "the environment"
  | Config_file_level -> "the config file"

(** Where {!get_global_arg} found a value; reported by the [log_config_sourcing] trace. *)
type config_source =
  | From_cmdline of string  (** the matching [Sys.argv] element *)
  | From_env of string  (** the matching environment variable *)
  | From_config_file
  | From_profile of config_level * string  (** the level that picked the profile, and its name *)
  | From_default

(** A short provenance tag, e.g. ["profile 'reproducible' via the commandline"]. *)
let config_source_label = function
  | From_cmdline _ -> "commandline"
  | From_env _ -> "environment"
  | From_config_file -> "config file"
  | From_profile (level, name) ->
      Printf.sprintf "profile '%s' via %s" name (describe_config_level level)
  | From_default -> "default"

let describe_config_source ~value ~default = function
  | From_cmdline arg -> Printf.sprintf "Found %s, commandline %s" value arg
  | From_env var -> Printf.sprintf "Found %s, environment %s" value var
  | From_config_file -> Printf.sprintf "Found %s, in the config file" value
  | From_profile (level, name) ->
      Printf.sprintf "Found %s, in the profile %S picked via %s" value name
        (describe_config_level level)
  | From_default -> Printf.sprintf "Not found, using default %s" default

(** The precedence walk, factored out of {!get_global_arg} so it can be exercised on synthetic
    sources (see test/operations/config_profiles.ml). The lookups are pure; logging happens at the
    call site, which is the only place that knows which sublevel won. *)
let resolve_config_value ~cmdline ~env ~file ~profile ~default ~arg_name =
  let from_profile level =
    match profile with
    | Some (l, name, lookup) when equal_config_level l level ->
        Option.map (lookup arg_name) ~f:(fun v -> (v, From_profile (level, name)))
    | _ -> None
  in
  let sublevels =
    [
      (fun () -> Option.map (cmdline arg_name) ~f:(fun (v, arg) -> (v, From_cmdline arg)));
      (fun () -> from_profile Cmdline_level);
      (fun () -> Option.map (env arg_name) ~f:(fun (v, var) -> (v, From_env var)));
      (fun () -> from_profile Env_level);
      (fun () -> Option.map (file arg_name) ~f:(fun v -> (v, From_config_file)));
      (fun () -> from_profile Config_file_level);
    ]
  in
  Option.value (List.find_map sublevels ~f:(fun f -> f ())) ~default:(default, From_default)

(** The keys a profile payload may not set: they are read before profiles are resolved (or would
    make the resolution recursive). *)
let profile_ineligible_keys =
  Set.of_list (module String) [ "profile"; "no_config_file"; "log_config_sourcing" ]

(* The payloads are embedded rather than installed as files: the config search walks up from the
   working directory and would find the USER's config, so a shipped preset file would need
   share-directory machinery -- painful on Windows, and wrong under `dune runtest` where the working
   directory is _build/default/<test dir>. Embedded text costs no distribution machinery and stays
   inspectable: it is reproduced verbatim in ocannl_config.reference and its provenance is reported
   by the log_config_sourcing trace. *)

let reproducible_profile_payload =
  {|# Deterministic, and identical across machines wherever reasonable. Cross-BACKEND
# reproducibility is explicitly out of scope: this profile does not try to make the cc
# and the cuda backends agree bit for bit.

# The largest cross-machine determinism leak: schedule identity pins numerics (loop order,
# split reductions, tensorization), so two machines crowning two schedules compute two
# results. Replaying stays allowed -- a pinned schedule is deterministic -- but only from a
# CHOSEN autotune_cache_dir: with the search off the built-in default is treated as no cache,
# so an earlier local search's leftovers in ./autotune_cache cannot silently pin a schedule.
autotune_search=false
# The analytic cost model picks schedules from per-backend (and, once calibrated, per-machine)
# envelope constants -- the same leak without the timing runs.
model_default_schedule=false
# The greedy inlining refinement of Train.tune_placements accepts a flip when the timing
# improves, so which nodes end up inlined is machine-dependent -- and inlining is not
# numerics-neutral where storage precision is narrower than compute precision: a materialized
# node rounds to its storage precision, an inlined one does not.
tune_inline_flips=0

# `-mcpu=native` / `-march=native` changes which FMA and vector instructions the compiler may
# use, hence the results, per machine. "none" passes no architecture flag.
cc_backend_arch_flags=none
# Whether the SIMD probe fires depends on what the toolchain's DEFAULT target already exposes,
# which is itself a per-machine fact -- so pin the flags off rather than reason about which of
# them could have changed a result.
cc_backend_simd_flags=none
# Contraction the codegen did not ask for (the explicit `fmaf` selections stay): whether a*b+c
# becomes one fused op is compiler- and target-discretionary.
cc_backend_fp_contract=off
cc_backend_fast_math=false
# Explicit SIMD rendering reassociates strict-FP reductions into per-lane accumulator chains,
# so the summation order would follow the probed vector width. 0 keeps the serial order.
cc_vector_bytes=0

# The numerics gates at their exact defaults, so that a profile switch never silently changes
# what math is being done (they are the orthogonal axis; see the issue). fp16_arithmetic=auto
# resolves deterministically per backend (gh-ocannl-680), so pinning the default keeps the
# principle: this profile changes no math, and same-machine runs stay reproducible.
tf32_matmuls=false
fp16_arithmetic=auto
narrow_compute_f32=true
|}

let performance_profile_payload =
  {|# The fastest configuration AT UNCHANGED SEMANTICS. Result-changing gates (tf32_matmuls,
# fp16_arithmetic's accuracy-for-throughput trade is the one exception below, and the
# algebraic-rewrite tiers) belong to the numerics axis, not to this one.

# Empirical schedule search on, with a wider beam and more rounds than the everyday defaults.
autotune_search=true
autotune_beam_width=4
autotune_rounds=4
# Untuned compiles still get the cost model's pick instead of the plain default pipeline.
model_default_schedule=true

# Target the host CPU, and let the SIMD probe add what that target already supports.
cc_backend_arch_flags=auto
cc_backend_simd_flags=auto

# Native 16-bit arithmetic where the target has it; ignored on targets that promote fp16 to
# float, and on the GPU backends (whose f16 arithmetic and accumulators are native narrow
# under this setting, same as under auto -- gh-ocannl-680).
fp16_arithmetic=true
|}

let approximate_profile_payload =
  {|# The fastest configuration AT TOLERANCE-BOUNDED SEMANTICS (gh-ocannl-719): the `performance`
# payload plus every knob that changes numerics for throughput. Results differ from the exact
# profiles by the tolerance the benchmark parity envelope names (PARITY_TOL_APPROX in
# benchmarks/orchestrate.py), never in shape or in which operations run. Each key below
# defaults off for the right reason -- PyTorch-style opt-in -- and this profile is the one
# word that flips them together, so that a benchmark cell can name the regime in one flag
# and a report can say which regime a number came from. The policy gates of the algebraic
# rewrites (online-softmax attention, gh-ocannl-483; Winograd convolution, gh-ocannl-505)
# join this payload as they land, each in the PR that adds its key.

# The `performance` payload's keys, restated (test_config_consistency checks they agree).
autotune_search=true
autotune_beam_width=4
autotune_rounds=4
model_default_schedule=true
cc_backend_arch_flags=auto
cc_backend_simd_flags=auto
fp16_arithmetic=true

# f32 matmul operands computed at tf32 (10-bit mantissa, f32 accumulation) on the backends with
# a tf32 tile shape (CUDA sm_80+); a no-op elsewhere.
tf32_matmuls=true
# The C compiler's licence to reassociate (fast-math) and to contract a*b+c into one rounding
# across statements (fp-contract=fast). Both change results per compiler and target, which is
# why `reproducible` pins them off and `performance` leaves them at their defaults.
cc_backend_fast_math=true
cc_backend_fp_contract=fast
# The greedy inlining refinement of Train.tune_placements is not numerics-neutral where storage
# is narrower than compute (a materialized node rounds to its storage precision, an inlined one
# does not), so `performance` leaves it alone. Two flips: each costs a full search, so this
# bounds the extra cost at two searches per arm while still exercising the refinement.
tune_inline_flips=2
|}

(** The embedded profile payloads, by name. Each is literally a partial [ocannl_config] file: same
    syntax, same parser, setting only the keys where the profiles' goals disagree. *)
let profile_payloads =
  [
    ("reproducible", reproducible_profile_payload);
    ("performance", performance_profile_payload);
    ("approximate", approximate_profile_payload);
  ]

let parse_profile_payload ~name text =
  let source = Printf.sprintf "the built-in profile %S" name in
  let table = config_table_of_lines ~source (String.split_lines text) in
  Hashtbl.iter_keys table ~f:(fun key ->
      if Set.mem profile_ineligible_keys key then
        failwith @@ "OCANNL: " ^ source ^ " sets the key " ^ key
        ^ ", which is resolved before profiles are"
      else if not (Set.mem known_config_keys key) then
        failwith @@ "OCANNL: " ^ source ^ " sets the unknown config key " ^ key);
  table

(** The profile picked for this run, if any: its level (which decides the priority of its payload),
    its name, and the parsed payload. *)
let active_profile =
  (* An EMPTY value is unset, at each level independently: everywhere else in the configuration ""
     means "as if absent", and a launcher expanding [--ocannl_profile=$PROFILE] with an unset
     variable must not thereby disable a profile the environment or the config file names (Codex P2
     on PR #291). So the fall-through tests each level's value, not just its presence. *)
  let normalize name = str_nonempty ~f:Fn.id (String.lowercase (String.strip name)) in
  let picked =
    List.find_map
      [
        (* [--ocannl_profile=...], not [--profile=...]: see [read_cmdline_var]'s
           [qualified_only]. *)
        (Cmdline_level, Option.map (read_cmdline_var ~qualified_only:true "profile") ~f:fst);
        (Env_level, Option.map (read_env_var "profile") ~f:fst);
        (Config_file_level, Hashtbl.find config_file_args "profile");
      ]
      ~f:(fun (level, value) ->
        Option.map (Option.bind value ~f:normalize) ~f:(fun name -> (level, name)))
  in
  Option.map picked ~f:(fun (level, name) ->
      match List.Assoc.find profile_payloads name ~equal:String.equal with
      | None ->
          invalid_arg @@ "OCANNL: unknown ocannl_profile " ^ name ^ " (picked via "
          ^ describe_config_level level ^ "); known profiles: "
          ^ String.concat ~sep:", " (List.map profile_payloads ~f:fst)
      | Some text ->
          if !log_config_sourcing then
            Stdio.eprintf "\nOCANNL: using the configuration profile %S, picked via %s.\n%!" name
              (describe_config_level level);
          (level, name, parse_profile_payload ~name text))

(** The provenance of the settings that resolve before {!get_global_arg_with_source} can report
    them, which is exactly the four read directly above: the three bootstrap keys and [profile].
    Everything else in OCANNL goes through that function and is traced as it goes.

    They cannot report themselves as they go. Each bootstrap key is read before
    {!log_config_sourcing} is settled -- one of the reads settles it -- and each is read more than
    once; [profile] resolves before the trace has a place to put a "not picked" line. So the report
    is assembled here instead, walking the same sources in the same order, and it covers the
    DEFAULTED cases: a run that sets none of the four still says so, which is what makes enabling
    the trace in a config file (the common way) report every setting the run read. Reporting only
    what was found is where rounds 1 and 2 of Codex's review on PR #348 went wrong twice.

    Recomputed rather than remembered: the lookups are pure functions of [Sys.argv], the environment
    and the config table, so one place holding the whole precedence walk cannot drift from the
    resolution the way scattered logging did. Two asymmetries are real, not oversights: a config
    file cannot supply [no_config_file] (it is what decides whether the file is read at all), and no
    profile can supply any of the bootstrap keys -- {!profile_ineligible_keys} rejects them,
    profiles being resolved later still. The bootstrap keys all default to false, [profile] to
    unset. *)
let () =
  if !log_config_sourcing then (
    Stdio.eprintf
      "\nOCANNL: settings resolved before the ordinary per-key trace could report them:\n%!";
    let report n line =
      Stdio.eprintf "Retrieving commandline, environment, or config file variable ocannl_%s\n%!" n;
      Stdio.eprintf "%s\n%!" line
    in
    List.iter [ "log_config_sourcing"; "no_config_file"; "suppress_welcome_message" ] ~f:(fun n ->
        let from_file =
          if equal_string n "no_config_file" then None else Hashtbl.find config_file_args n
        in
        let value, source =
          match read_cmdline_var n with
          | Some (value, arg) -> (value, From_cmdline arg)
          | None -> (
              match read_env_var n with
              | Some (value, var) -> (value, From_env var)
              | None -> (
                  match from_file with
                  | Some value -> (value, From_config_file)
                  | None -> ("false", From_default)))
        in
        report n (describe_config_source ~value ~default:"false" source));
    (* Taken from the resolved profile rather than re-walked: the walk above it normalizes empty
       values and falls through per level, and a second copy of that rule could disagree with the
       one that decides. The banner it prints on the way is about the payload taking effect; this
       line is about where the setting came from, and only it appears when no profile is picked. *)
    report "profile"
      (match active_profile with
      | Some (level, name, _) -> Printf.sprintf "Found %s, in %s" name (describe_config_level level)
      | None -> describe_config_source ~value:"" ~default:"" From_default))

let profile_lookup =
  Option.map active_profile ~f:(fun (level, name, table) ->
      (level, name, fun key -> Hashtbl.find table key))

(** Retrieves the [arg_name] setting from the commandline, the environment, the config file, or the
    payload of the profile picked at one of those levels; returns [default] if none has it, together
    with where the value came from. *)
let get_global_arg_with_source ~default ~arg_name:n =
  let with_debug = !log_config_sourcing && not (Hash_set.mem accessed_global_args n) in
  if with_debug then
    Stdio.eprintf "Retrieving commandline, environment, or config file variable ocannl_%s\n%!" n;
  let result, source =
    resolve_config_value ~cmdline:read_cmdline_var ~env:read_env_var
      ~file:(Hashtbl.find config_file_args) ~profile:profile_lookup ~default ~arg_name:n
  in
  if with_debug then Stdio.eprintf "%s\n%!" (describe_config_source ~value:result ~default source);
  Hash_set.add accessed_global_args n;
  (result, source)

let get_global_arg ~default ~arg_name = fst (get_global_arg_with_source ~default ~arg_name)

let get_global_flag ~default ~arg_name:n =
  bool_of_config_string ~arg_name:n
  @@ get_global_arg ~default:(if default then "true" else "false") ~arg_name:n

(* Defaults to 0 (gh-ocannl-595): every [ocannl_config] in this repository chose 0, which is the
   measure of a suspect default. Level 1 is a verbosity the user asks for -- it adds the backend
   info to {!Tnode.header} and raises the ppx_minidebug runtime's threshold; the gates that change
   what a kernel computes ([with_runtime_debug], [debug_log_from_routines]) sit at level 2 and are
   unaffected. *)
let original_log_level =
  let log_level =
    let s = String.strip @@ get_global_arg ~default:"0" ~arg_name:"log_level" in
    match Int.of_string_opt s with
    | Some ll -> ll
    | None -> invalid_arg @@ "ocannl_log_level setting should be an integer; found: " ^ s
  in
  settings.log_level <- log_level;
  log_level

(* Originally from the library core.filename_base. *)
let filename_concat p1 p2 =
  if String.is_empty p1 then
    invalid_arg
    @@ "Utils.filename_concat called with an empty string as its first argument, second argument: "
    ^ p2;
  let rec collapse_trailing s =
    match String.rsplit2 s ~on:'/' with
    | Some ("", ("." | "")) -> ""
    | Some (s, ("." | "")) -> collapse_trailing s
    | None | Some _ -> s
  in
  let rec collapse_leading s =
    match String.lsplit2 s ~on:'/' with
    | Some (("." | ""), s) -> collapse_leading s
    | Some _ | None -> s
  in
  collapse_trailing p1 ^ "/" ^ collapse_leading p2

let clean_filename fname =
  let fname = String.strip fname in
  let fname =
    (* Beyond the path separators, the rest of the set Windows reserves: a routine named after a
       diagnostic can carry any of them (gh-ocannl-481's "... unit count > 1" reached [open_out] as
       a filename and killed the test process on Windows with [Invalid argument], while passing on
       POSIX where only '/' is special). Replaced on every platform rather than under [Sys.win32],
       so that a debug artifact has one name everywhere and a rule naming it cannot be
       platform-dependent. *)
    String.map
      ~f:(fun c ->
        if
          List.exists ~f:(equal_char c) [ '/'; '\\'; ':'; '<'; '>'; '"'; '|'; '?'; '*' ]
          || Char.to_int c < 0x20
        then '-'
        else c)
      fname
  in
  (* Reject bare "."/".." (and the empty string): otherwise filename_concat "build_files" cleaned
     can resolve to build_files/.. and the startup cleanup (clean_up_build_files_on_startup=true)
     would recursively delete the parent directory. *)
  match fname with
  | "" | "." | ".." -> "_"
  | _ -> fname

(* Concurrently running programs (e.g. dune tests) share the working directory, so a flat
   [build_files/] (or [log_files/]) would race on same-named artifacts. Unless overridden by
   [build_files_prefix], every process gets its own subdirectory derived from the executable name;
   the sentinel value "." restores the flat legacy layout. *)
let artifacts_subdir () =
  match get_global_arg ~default:"" ~arg_name:"build_files_prefix" with
  | "" ->
      Some
        (clean_filename @@ Stdlib.Filename.remove_extension
        @@ Stdlib.Filename.basename Stdlib.Sys.executable_name)
  | "." -> None
  | prefix -> Some (clean_filename prefix)

(* Tolerates mkdir races between concurrently running tests. *)
let ensure_artifacts_dir base subdir =
  let dir = match subdir with None -> base | Some p -> filename_concat base p in
  (try assert (Stdlib.Sys.is_directory dir)
   with Stdlib.Sys_error _ | Assert_failure _ -> (
     (try assert (Stdlib.Sys.is_directory base)
      with Stdlib.Sys_error _ | Assert_failure _ -> (
        try Stdlib.Sys.mkdir base 0o777 with Stdlib.Sys_error _ -> ()));
     if Option.is_some subdir then try Stdlib.Sys.mkdir dir 0o777 with Stdlib.Sys_error _ -> ()));
  dir

(** The directory generated-code debug files are written to (created if missing):
    [build_files/<prefix>/], where the prefix defaults to the executable's base name. *)
let build_files_dir () = ensure_artifacts_dir "build_files" (artifacts_subdir ())

let build_file fname = filename_concat (build_files_dir ()) @@ clean_filename fname

(** The directory diagnostic and routine-debug logs are written to (created if missing):
    [log_files/<prefix>/], sharing the prefix resolution of {!build_files_dir}. *)
let log_files_dir () = ensure_artifacts_dir "log_files" (artifacts_subdir ())

let diagn_log_file fname = filename_concat (log_files_dir ()) @@ clean_filename fname

let () =
  (* Cleanup needs to happen before get_local_debug_runtime (or any other code is run). *)
  (* Use Unix.lstat to distinguish symlinks from real directories: Sys.is_directory
     follows symlinks, so a symlink inside build_files/log_files pointing outside
     the tree would cause recursion to delete unrelated files. We unlink the link
     itself instead of descending. *)
  let lstat_kind path = try Some (Unix.lstat path).st_kind with Unix.Unix_error _ -> None in
  let rec remove_dir_if_exists dirname =
    match lstat_kind dirname with
    | None -> ()
    | Some Unix.S_DIR -> (
        try
          Array.iter (Stdlib.Sys.readdir dirname) ~f:(fun fname ->
              let path = Stdlib.Filename.concat dirname fname in
              match lstat_kind path with
              | Some Unix.S_DIR -> remove_dir_if_exists path
              | Some _ | None -> (
                  (* Regular file, symlink (to file or dir), socket, etc.: unlink the entry, do not
                     follow. *)
                  try Stdlib.Sys.remove path with Stdlib.Sys_error _ -> ()));
          Stdlib.Sys.rmdir dirname
        with exn ->
          Stdio.eprintf "Failed to delete directory %s: %s\n%!" dirname (Exn.to_string exn))
    | Some _ -> (
        (* Symlink (even to a dir), regular file, etc.: unlink the entry. *)
        try Stdlib.Sys.remove dirname
        with exn ->
          Stdio.eprintf "Failed to delete %s (expected a directory): %s\n%!" dirname
            (Exn.to_string exn))
  in
  (* Cleanup is scoped to this process's own subdirectory (see [artifacts_subdir]), so a starting
     test cannot delete the in-flight artifacts of a concurrently running one. If the artifact root
     itself is a symlink, skip the scoped cleanup rather than follow it: appending the subdirectory
     would resolve through the link, deleting files outside the working tree. *)
  let remove_scoped_dir base =
    match artifacts_subdir () with
    | None -> remove_dir_if_exists base
    | Some p -> (
        match lstat_kind base with
        | Some Unix.S_DIR -> remove_dir_if_exists (filename_concat base p)
        | Some _ | None -> ())
  in
  let clean_up_log_files_on_startup =
    get_global_flag ~default:true ~arg_name:"clean_up_log_files_on_startup"
  in
  if clean_up_log_files_on_startup then remove_scoped_dir "log_files";
  let clean_up_build_files_on_startup =
    get_global_flag ~default:true ~arg_name:"clean_up_build_files_on_startup"
  in
  if clean_up_build_files_on_startup then remove_scoped_dir "build_files"

let get_local_debug_runtime =
  let snapshot_every_sec =
    Option.join
    @@ str_nonempty ~f:Float.of_string_opt
    @@ get_global_arg ~default:"" ~arg_name:"snapshot_every_sec"
  in
  let time_tagged =
    match String.lowercase @@ get_global_arg ~default:"elapsed" ~arg_name:"time_tagged" with
    | "not_tagged" -> Minidebug_runtime.Not_tagged
    | "clock" -> Clock
    | "elapsed" -> Elapsed
    | s -> invalid_arg @@ "ocannl_time_tagged setting should be none, clock or elapsed; found: " ^ s
  in
  let elapsed_times =
    match String.lowercase @@ get_global_arg ~default:"not_reported" ~arg_name:"elapsed_times" with
    | "not_reported" -> Minidebug_runtime.Not_reported
    | "seconds" -> Seconds
    | "milliseconds" -> Milliseconds
    | "microseconds" -> Microseconds
    | "nanoseconds" -> Nanoseconds
    | s ->
        invalid_arg
        @@ "ocannl_elapsed_times setting should be not_reported, seconds or milliseconds, \
            microseconds or nanoseconds; found: " ^ s
  in
  let location_format =
    match String.lowercase @@ get_global_arg ~default:"beg_pos" ~arg_name:"location_format" with
    | "no_location" -> Minidebug_runtime.No_location
    | "file_only" -> File_only
    | "beg_line" -> Beg_line
    | "beg_pos" -> Beg_pos
    | "range_line" -> Range_line
    | "range_pos" -> Range_pos
    | s ->
        invalid_arg
        @@ "ocannl_location_format setting should be one of: no_location, file_only, beg_line, \
            beg_pos, range_line, range_pos; found: " ^ s
  in
  let backend, toc_flame_graph, printbox_backend =
    match
      String.lowercase @@ String.strip @@ get_global_arg ~default:"db" ~arg_name:"debug_backend"
    with
    | "text" -> (`Printbox, false, `Text)
    | "html" -> (`Printbox, true, `Html Minidebug_runtime.default_html_config)
    | "markdown" -> (`Printbox, false, `Markdown Minidebug_runtime.default_md_config)
    | "flushing" -> (`Flushing, true, `Text)
    | "db" -> (`Db, false, `Text)
    | s ->
        invalid_arg
        @@ "ocannl_debug_backend setting should be text, html, markdown, flushing or db; found: "
        ^ s
  in
  let hyperlink = get_global_arg ~default:"./" ~arg_name:"hyperlink_prefix" in
  let print_scope_ids = get_global_flag ~default:false ~arg_name:"logs_print_scope_ids" in
  let verbose_scope_ids = get_global_flag ~default:false ~arg_name:"logs_verbose_scope_ids" in
  let log_main_domain_to_stdout =
    get_global_flag ~default:false ~arg_name:"log_main_domain_to_stdout"
  in
  let file_stem =
    if log_main_domain_to_stdout then None
    else Some (get_global_arg ~default:"debug" ~arg_name:"log_file_stem")
  in
  let filename = Option.map file_stem ~f:diagn_log_file in
  let prev_run_file =
    let prefix = str_nonempty ~f:Fn.id @@ get_global_arg ~default:"" ~arg_name:"prev_run_prefix" in
    Option.map2 prefix file_stem ~f:(fun prefix stem -> diagn_log_file @@ prefix ^ stem ^ ".raw")
  in
  let toc_entry_minimal_depth =
    let arg = get_global_arg ~default:"" ~arg_name:"toc_entry_minimal_depth" in
    if String.is_empty arg then [] else [ Minidebug_runtime.Minimal_depth (Int.of_string arg) ]
  in
  let toc_entry_minimal_size =
    let arg = get_global_arg ~default:"" ~arg_name:"toc_entry_minimal_size" in
    if String.is_empty arg then [] else [ Minidebug_runtime.Minimal_size (Int.of_string arg) ]
  in
  let toc_entry_minimal_span =
    let arg = get_global_arg ~default:"" ~arg_name:"toc_entry_minimal_span" in
    if String.is_empty arg then []
    else
      let arg, period = (String.prefix arg (String.length arg - 2), String.suffix arg 2) in
      let period =
        match period with
        | "ns" -> Mtime.Span.ns
        | "us" -> Mtime.Span.us
        | "ms" -> Mtime.Span.ms
        | _ ->
            invalid_arg
            @@ "ocannl_toc_entry_minimal_span setting should end with one of: ns, us, ms; found: "
            ^ period
      in
      [ Minidebug_runtime.Minimal_span Mtime.Span.(Int.of_string arg * period) ]
  in
  let toc_entry =
    Minidebug_runtime.And (toc_entry_minimal_depth @ toc_entry_minimal_size @ toc_entry_minimal_span)
  in
  let debug_highlights =
    let arg = get_global_arg ~default:"" ~arg_name:"debug_highlights" in
    if String.is_empty arg then [] else String.split arg ~on:'|'
  in
  let highlight_re =
    let arg = get_global_arg ~default:"" ~arg_name:"debug_highlight_pcre" in
    Option.to_list @@ str_nonempty ~f:Re.Pcre.re arg
  in
  let highlight_terms = Re.(alt (highlight_re @ List.map debug_highlights ~f:str)) in
  let diff_ignore_pattern =
    str_nonempty ~f:Re.Pcre.re @@ get_global_arg ~default:"" ~arg_name:"diff_ignore_pattern_pcre"
  in
  let max_distance_factor =
    str_nonempty ~f:Int.of_string @@ get_global_arg ~default:"" ~arg_name:"diff_max_distance_factor"
  in
  let scope_id_pairs =
    let pairs_str = get_global_arg ~default:"" ~arg_name:"debug_scope_id_pairs" in
    if String.is_empty pairs_str then []
    else
      String.split pairs_str ~on:';'
      |> List.filter_map ~f:(fun pair_str ->
          match String.split pair_str ~on:',' with
          | [ id1; id2 ] ->
              Option.try_with (fun () ->
                  (Int.of_string (String.strip id1), Int.of_string (String.strip id2)))
          | _ -> None)
  in
  let truncate_children =
    let arg = get_global_arg ~default:"" ~arg_name:"debug_log_truncate_children" in
    if String.is_empty arg then None else Some (Int.of_string arg)
  in
  let prune_upto =
    let arg = get_global_arg ~default:"" ~arg_name:"debug_log_prune_upto" in
    if String.is_empty arg then None else Some (Int.of_string arg)
  in
  let name = get_global_arg ~default:"debug" ~arg_name:"log_file_stem" in
  match (backend, filename) with
  | `Flushing, None ->
      Minidebug_runtime.prefixed_runtime_flushing ~time_tagged ~elapsed_times ~location_format
        ~print_scope_ids ~verbose_scope_ids ~global_prefix:name ~for_append:false
        ~log_level:original_log_level ()
  | `Flushing, Some filename ->
      Minidebug_runtime.local_runtime_flushing ~time_tagged ~elapsed_times ~location_format
        ~print_scope_ids ~verbose_scope_ids ~global_prefix:name ~for_append:false
        ~log_level:original_log_level filename
  | `Printbox, None ->
      Minidebug_runtime.prefixed_runtime ~time_tagged ~elapsed_times ~location_format
        ~print_scope_ids ~verbose_scope_ids ~global_prefix:name ~toc_entry
        ~toc_specific_hyperlink:"" ~highlight_terms ?truncate_children
        ~exclude_on_path:Re.(str "env")
        ~log_level:original_log_level ?snapshot_every_sec ()
  | `Printbox, Some filename ->
      Minidebug_runtime.local_runtime ~time_tagged ~elapsed_times ~location_format ~print_scope_ids
        ~verbose_scope_ids ~global_prefix:name ~toc_flame_graph ~flame_graph_separation:50
        ~toc_entry ~for_append:false ~max_inline_sexp_length:120 ~hyperlink
        ~toc_specific_hyperlink:"" ~highlight_terms ?truncate_children
        ~exclude_on_path:Re.(str "env")
        ?prune_upto ~backend:printbox_backend ~log_level:original_log_level ?snapshot_every_sec
        ?prev_run_file ?diff_ignore_pattern ?max_distance_factor ~scope_id_pairs filename
  | `Db, _ ->
      let filename = Option.value ~default:(diagn_log_file name) filename in
      let db =
        Minidebug_db.debug_db_file ~time_tagged ~elapsed_times ~print_scope_ids ~verbose_scope_ids
          ~run_name:name ~log_level:original_log_level filename
      in
      fun () -> db

let _get_local_debug_runtime = get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_UTILS=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_UTILS"]

(* [%%global_debug_interrupts { max_nesting_depth = 100; max_num_children = 1000 }] *)

let%diagn_sexp set_log_level level =
  settings.log_level <- level;
  [%log
    "Set log_level to",
    (Debug_runtime.log_level := level;
     level
      : int)]

let restore_settings () =
  set_log_level original_log_level;
  settings.debug_log_from_routines <-
    get_global_flag ~default:false ~arg_name:"debug_log_from_routines";
  settings.output_debug_files_in_build_directory <-
    get_global_flag ~default:false ~arg_name:"output_debug_files_in_build_directory";
  settings.fixed_state_for_init <-
    (let seed = get_global_arg ~arg_name:"fixed_state_for_init" ~default:"" in
     if String.is_empty seed then None else Some (Int.of_string seed));
  settings.print_decimals_precision <-
    Int.of_string @@ get_global_arg ~arg_name:"print_decimals_precision" ~default:"2";
  settings.check_half_prec_constants_cutoff <-
    Float.of_string_opt
    @@ get_global_arg ~arg_name:"check_half_prec_constants_cutoff" ~default:"16384.0";
  settings.default_prng_variant <- get_global_arg ~default:"light" ~arg_name:"default_prng_variant";
  (* [big_models] is the pre-gh-ocannl-344 name; read it as a fallback so existing configs / CLI /
     OCANNL_BIG_MODELS still enable 64-bit index and pool-offset widths when [large_models] is
     unset. *)
  settings.large_models <-
    get_global_flag
      ~default:(get_global_flag ~default:false ~arg_name:"big_models")
      ~arg_name:"large_models"

let () = restore_settings ()

(* An argument that ADDRESSES OCANNL -- it carries the prefix -- and that no known key would be read
   from. Only the qualified spellings are eligible: a prefix-free `--verbose` belongs to the host
   application, and OCANNL has no standing to call it unknown.

   The test is "would some key read this argument", asked of {!cmdline_var_prefixes}, which is what
   `read_cmdline_var` itself scans -- so the warning cannot disagree with the reader about what a
   spelling means. It costs one pass over argv per known key, at module initialization. *)
let () =
  (* The leading dash is what makes an argument addressed rather than positional, which is why
     `cmdline_var_names` no longer reads bare qualified spellings: these four prefixes now cover
     every spelling it emits, so nothing is read that cannot also be diagnosed (Codex P2 on PR
     #363). A bare `ocannl_config` on a host tool's commandline is a path, and stays one.

     Case-folded, because the uppercase spellings are read too: `--OCANNL_LOG_LEVEL=1` is a setting,
     `--OCANNL_NOT_A_KEY=1` is a mistake, and both address OCANNL. *)
  let ocannl_prefixes = [ "--ocannl_"; "--ocannl-"; "-ocannl_"; "-ocannl-" ] in
  let addresses_ocannl arg =
    let lower = String.lowercase arg in
    List.exists ocannl_prefixes ~f:(fun p -> String.is_prefix ~prefix:p lower)
  in
  (* On an [ocannl]-prefixed argument the per-key [qualified_only] of {!cmdline_arg_is_config_key}
     coincides with matching every key's qualified spellings — unqualified spellings cannot match a
     prefixed argument — so the warning and the tools' pass-through share one predicate and cannot
     drift apart. *)
  Array.iter Stdlib.Sys.argv ~f:(fun arg ->
      if addresses_ocannl arg && not (cmdline_arg_is_config_key arg) then
        Stdio.eprintf "OCANNL warning: unknown commandline argument %S\n%!" arg)

let with_runtime_debug () = settings.output_debug_files_in_build_directory && settings.log_level > 1
let debug_log_from_routines () = settings.debug_log_from_routines && settings.log_level > 1
let never_capture_stdout () = get_global_flag ~default:false ~arg_name:"never_capture_stdout"

let enable_runtime_debug () =
  settings.output_debug_files_in_build_directory <- true;
  set_log_level @@ max 2 settings.log_level

(** A fresh non-negative integer per call, wrapping to 0 on overflow. Backends use it to tell one
    run of a kernel from the next: cc names each dynamically loaded library by it (there can be only
    one library of a given name in a process), and the GPU backends prefix a launch's routine logs
    with it. Process-wide rather than per backend, so that ids never collide across the backends a
    process links. *)
let get_global_run_id =
  let next_id = ref 0 in
  fun () ->
    Int.incr next_id;
    if !next_id < 0 then next_id := 0;
    !next_id

let rec union_find ~equal map ~key ~rank =
  match Map.find map key with
  | None -> (key, rank)
  | Some data ->
      if equal key data then (key, rank) else union_find ~equal map ~key:data ~rank:(rank + 1)

let union_add ~equal map k1 k2 =
  if equal k1 k2 then map
  else
    let root1, rank1 = union_find ~equal map ~key:k1 ~rank:0
    and root2, rank2 = union_find ~equal map ~key:k2 ~rank:0 in
    if rank1 < rank2 then Map.update map root1 ~f:(fun _ -> root2)
    else Map.update map root2 ~f:(fun _ -> root1)

(** Filters the list keeping the first occurrence of each element. *)
let unique_keep_first ~equal l =
  let rec loop acc = function
    | [] -> List.rev acc
    | hd :: tl -> if List.mem acc hd ~equal then loop acc tl else loop (hd :: acc) tl
  in
  loop [] l

(** Returns the multiset difference of [l1] and [l2], where [l1] and [l2] must be sorted in
    increasing order. *)
let sorted_diff ~compare l1 l2 =
  let rec loop acc l1 l2 =
    match (l1, l2) with
    | [], _ -> List.rev acc
    | l1, [] -> List.rev_append acc l1
    | h1 :: t1, h2 :: t2 -> (
        match compare h1 h2 with
        | c when c < 0 -> loop (h1 :: acc) t1 l2
        | 0 ->
            (* Depending on this line this can be either a set diff or a multiset: currently it's
               multiset diff. *)
            loop acc t1 t2
        | _ -> loop acc l1 t2)
  in
  (loop [] l1 l2 [@nontail])

(** Removes the first occurrence of an element from the list that is equal to the given element. *)
let remove_elem ~equal elem l =
  let rec loop acc = function
    | [] -> List.rev acc
    | hd :: tl -> if equal elem hd then List.rev_append acc tl else loop (hd :: acc) tl
  in
  loop [] l

(** [parallel_merge merge num_devices] progressively invokes the pairwise [merge] callback,
    converging on the 0th position, with [from] ranging from [1] to [num_devices - 1], and
    [to_ < from]. *)
let parallel_merge merge (num_devices : int) =
  let rec loop (upper : int) : unit =
    let is_even = (upper + 1) % 2 = 0 in
    let lower = if is_even then 0 else 1 in
    let half : int = (upper - (lower - 1)) / 2 in
    if half > 0 then (
      let midpoint : int = half + lower - 1 in
      for i = lower to midpoint do
        (* Maximal [from] is [2 * half + lower - 1 = upper]. *)
        merge ~from:(half + i) ~to_:i
      done;
      loop midpoint)
  in
  loop (num_devices - 1)

let ( !@ ) = Atomic.get

type atomic_bool = bool Atomic.t

let sexp_of_atomic_bool flag = sexp_of_bool @@ Atomic.get flag

type atomic_int = int Atomic.t

let sexp_of_atomic_int flag = sexp_of_int @@ Atomic.get flag

let sexp_append ~elem = function
  | Sexp.List l -> Sexp.List (elem :: l)
  | Sexp.Atom _ as e2 -> Sexp.List [ elem; e2 ]

let sexp_mem ~elem = function
  | Sexp.Atom _ as e2 -> Sexp.equal elem e2
  | Sexp.List l -> Sexp.(List.mem ~equal l elem)

let rec sexp_deep_mem ~elem = function
  | Sexp.Atom _ as e2 -> Sexp.equal elem e2
  | Sexp.List l -> Sexp.(List.mem ~equal l elem) || List.exists ~f:(sexp_deep_mem ~elem) l

let split_with_seps sep s =
  let tokens = Re.split_full sep s in
  List.map tokens ~f:(function `Text tok -> tok | `Delim sep -> Re.Group.get sep 0)

(** {2 Rendering a double as text (gh-ocannl-623, gh-ocannl-713)} *)

(** [s] with any exponent's leading zeros removed, so that the rendering does not depend on which C
    runtime formatted it.

    OCaml's [%g] goes through the platform's [snprintf], and the Windows runtimes pad the exponent
    to three digits ([1e+020] where glibc writes [1e+20]) -- which would make generated kernels, IR
    dumps, and any golden quoting one, differ by platform. Both spellings denote the same number, so
    this is about the artifact being reproducible rather than about the value. Digits that are not
    padding are kept: [1e-300] stays itself. *)
let normalize_exponent s =
  match String.findi s ~f:(fun _ ch -> Char.(ch = 'e' || ch = 'E')) with
  | None -> s
  | Some (i, _) ->
      let mantissa = String.prefix s i and exp = String.drop_prefix s (i + 1) in
      let sign, digits =
        match String.chop_prefix exp ~prefix:"+" with
        | Some d -> ("+", d)
        | None -> (
            match String.chop_prefix exp ~prefix:"-" with Some d -> ("-", d) | None -> ("", exp))
      in
      if String.is_empty digits || not (String.for_all digits ~f:Char.is_digit) then s
      else
        let stripped = String.lstrip digits ~drop:(Char.equal '0') in
        let stripped = if String.is_empty stripped then "0" else stripped in
        mantissa ^ "e" ^ sign ^ stripped

(** The decimal spelling of the double [c]: a floating literal -- a radix point or an exponent is
    always present -- that parses back to exactly [c].

    Two properties [%.16g] alone gets wrong, and both of them matter wherever a constant is written
    down: in emitted kernel text ({!C_syntax.c_float_literal}, gh-ocannl-623) and in the IR dumps
    ({!Low_level.to_doc} and {!Low_level.to_doc_cstyle}, gh-ocannl-713).

    - {b It is a floating literal, not an integer one.} [%.16g] of a value with no fractional part
      has no radix point, so [2.] comes out as ["2"] and [-0.] as ["-0"] -- the integer zero, hence
      [+0.0] once read as C, and indistinguishable from [+0.] to a reader of a dump. That is the one
      distinction the gh-ocannl-615 chase was about: asking whether a [-0.0] in host data survives
      to the kernel, and being answered [:= 0] either way. Forcing the radix point closes the whole
      class at once.
    - {b It round-trips.} 16 significant digits do not recover every double: [0.1 +. 0.2] prints as
      ["0.3"], which is a {e different} double, and so a constant whose 17th digit matters displays
      identically to one whose does not. The rendering therefore retries at [%.17g] -- the width
      IEEE-754 guarantees -- whenever the 16-digit spelling does not parse back to [c]. Values that
      already round-trip keep their exact previous spelling, which is why goldens move only by the
      appended [.0].

    Non-finite values get [%.16g]'s words ([inf], [-inf], [nan]): they have no decimal spelling at
    all, and appending a radix point to one would produce a token that is neither. A caller whose
    dialect needs its own spelling handles them before calling -- {!C_syntax.c_float_literal} spells
    C's [INFINITY] and [NAN], while an IR dump, which is not C, keeps the words. *)
let decimal_float_literal c =
  if not (Float.is_finite c) then Printf.sprintf "%.16g" c
  else
    let s =
      let s16 = Printf.sprintf "%.16g" c in
      if Float.equal (Float.of_string s16) c then s16 else Printf.sprintf "%.17g" c
    in
    (* [%g] emits a radix point or an exponent for everything else, and either one makes the token a
       floating literal. *)
    if String.exists s ~f:(function '.' | 'e' | 'E' -> true | _ -> false) then normalize_exponent s
    else s ^ ".0"

module Lazy = struct
  include Lazy

  let sexp_of_t = Minidebug_runtime.sexp_of_lazy_t
  let sexp_of_lazy_t = Minidebug_runtime.sexp_of_lazy_t
end

type requirement =
  | Skip
  | Required
  | Optional of { callback_if_missing : unit -> unit [@sexp.opaque] [@compare.ignore] }
[@@deriving compare, sexp]

let default_indent = ref 2

let doc_of_sexp sexp =
  let open Sexp in
  let open Int in
  let module Bytes = Stdlib.Bytes in
  let must_escape str =
    let len = String.length str in
    len = 0
    ||
    let rec loop str ix =
      match str.[ix] with
      | '"' | '(' | ')' | ';' | '\\' -> true
      | '|' ->
          ix > 0
          &&
          let next = ix - 1 in
          Char.equal str.[next] '#' || loop str next
      | '#' ->
          ix > 0
          &&
          let next = ix - 1 in
          Char.equal str.[next] '|' || loop str next
      | '\000' .. '\032' | '\127' .. '\255' -> true
      | _ -> ix > 0 && loop str (ix - 1)
    in
    loop str (len - 1)
  in

  let escaped s =
    let n = ref 0 in
    for i = 0 to String.length s - 1 do
      n :=
        !n
        +
        match String.unsafe_get s i with
        | '\"' | '\\' | '\n' | '\t' | '\r' | '\b' -> 2
        | ' ' .. '~' -> 1
        | _ -> 4
    done;
    if !n = String.length s then s
    else
      let s' = Bytes.create !n in
      n := 0;
      for i = 0 to String.length s - 1 do
        (match String.unsafe_get s i with
        | ('\"' | '\\') as c ->
            Bytes.unsafe_set s' !n '\\';
            incr n;
            Bytes.unsafe_set s' !n c
        | '\n' ->
            Bytes.unsafe_set s' !n '\\';
            incr n;
            Bytes.unsafe_set s' !n 'n'
        | '\t' ->
            Bytes.unsafe_set s' !n '\\';
            incr n;
            Bytes.unsafe_set s' !n 't'
        | '\r' ->
            Bytes.unsafe_set s' !n '\\';
            incr n;
            Bytes.unsafe_set s' !n 'r'
        | '\b' ->
            Bytes.unsafe_set s' !n '\\';
            incr n;
            Bytes.unsafe_set s' !n 'b'
        | ' ' .. '~' as c -> Bytes.unsafe_set s' !n c
        | c ->
            let a = Stdlib.Char.code c in
            Bytes.unsafe_set s' !n '\\';
            incr n;
            Bytes.unsafe_set s' !n (Stdlib.Char.chr (48 + (a / 100)));
            incr n;
            Bytes.unsafe_set s' !n (Stdlib.Char.chr (48 + (a / 10 % 10)));
            incr n;
            Bytes.unsafe_set s' !n (Stdlib.Char.chr (48 + (a % 10))));
        incr n
      done;
      Bytes.unsafe_to_string s'
  in

  let esc_str str =
    let estr = escaped str in
    let elen = String.length estr in
    let res = Bytes.create (elen + 2) in
    Bytes.blit_string estr 0 res 1 elen;
    Bytes.unsafe_set res 0 '"';
    Bytes.unsafe_set res (elen + 1) '"';
    Bytes.unsafe_to_string res
  in

  let index_of_newline str start = Stdlib.String.index_from_opt str start '\n' in

  let get_substring str index end_pos_opt =
    let end_pos = match end_pos_opt with None -> String.length str | Some end_pos -> end_pos in
    String.sub str ~pos:index ~len:(end_pos - index)
  in

  let is_one_line str =
    match index_of_newline str 0 with
    | None -> true
    | Some index -> Int.(index + 1 = String.length str)
  in

  let open PPrint in
  let doc_maybe_esc_str str =
    if not (must_escape str) then string str
    else if is_one_line str then string (esc_str str)
    else
      let rec loop index acc =
        let next_newline = index_of_newline str index in
        let next_line = get_substring str index next_newline in
        let acc = acc ^^ string (escaped next_line) in
        match next_newline with
        | None -> acc
        | Some newline_index ->
            loop (newline_index + 1) (acc ^^ string "\\" ^^ hardline ^^ string "\\n")
      in
      (* the leading space is to line up the lines *)
      string " \"" ^^ loop 0 empty ^^ string "\""
  in

  let rec doc_of_sexp_indent indent = function
    | Atom str -> doc_maybe_esc_str str
    | List (h :: t) ->
        group (string "(" ^^ nest indent (doc_of_sexp_indent indent h ^^ doc_of_sexp_rest indent t))
    | List [] -> string "()"
  and doc_of_sexp_rest indent = function
    | h :: t -> space ^^ doc_of_sexp_indent indent h ^^ doc_of_sexp_rest indent t
    | [] -> string ")"
  in

  doc_of_sexp_indent !default_indent sexp

let output_to_build_file ~fname =
  if settings.output_debug_files_in_build_directory then
    let f = Stdio.Out_channel.create @@ build_file fname in
    let print doc =
      PPrint.ToChannel.pretty 0.7 100 f doc;
      Stdio.Out_channel.flush f
    in
    Some print
  else None

let get_debug_output_channel ~fname =
  if settings.output_debug_files_in_build_directory then
    Some (Stdio.Out_channel.create @@ build_file fname)
  else None

exception User_error of string

let header_sep =
  let open Re in
  compile (seq [ str " "; opt any; str "="; str " " ])

let%diagn_sexp log_trace_tree _logs =
  [%log_block
    "trace tree";
    let sep s = String.concat ~sep:"\n" @@ String.split ~on:'$' s in
    let rec loop = function
      | [] -> []
      | line :: more when String.is_empty line -> loop more
      | "COMMENT: end" :: more -> more
      | comment :: more when String.is_prefix comment ~prefix:"COMMENT: " ->
          let more =
            [%log_entry
              sep @@ String.chop_prefix_exn ~prefix:"COMMENT: " comment;
              loop more]
          in
          loop more
      | source :: trace :: more when String.is_prefix source ~prefix:"# " ->
          (let source = sep @@ String.chop_prefix_exn ~prefix:"# " source in
           match split_with_seps header_sep @@ sep trace with
           | [] | [ "" ] -> [%log source]
           | header1 :: assign1 :: header2 :: body ->
               let header = String.concat [ header1; assign1; header2 ] in
               let body = String.concat body in
               let _message = Sexp.(List [ Atom header; Atom source; Atom body ]) in
               [%log (_message : Sexp.t)]
           | _ -> [%log source, trace]);
          loop more
      | _line :: more ->
          [%log sep _line];
          loop more
    in
    let rec loop_logs logs =
      let output = loop logs in
      if not (List.is_empty output) then
        [%log_block
          "TRAILING LOGS:";
          loop_logs output]
    in
    loop_logs _logs]

include Datatypes
module Cpu_topology = Cpu_topology

(* Re-exported rather than left as a bare sibling: [utils.ml] is the library's interface module, so
   [Utils.Atomic_file] is the only spelling a dependant can reach it by. *)
module Atomic_file = Atomic_file

type build_file_channel = { f_path : string; oc : Stdlib.out_channel; finalize : unit -> unit }

let open_build_file ~base_name ~extension : build_file_channel =
  let f_path =
    if settings.output_debug_files_in_build_directory then build_file @@ base_name ^ extension
    else Stdlib.Filename.temp_file (base_name ^ "_") extension
  in
  (* (try Stdlib.Sys.remove f_path with _ -> ()); *)
  let oc = Out_channel.open_text f_path in
  let finalize () =
    Stdio.Out_channel.flush oc;
    Stdio.Out_channel.close oc
  in
  { f_path; oc; finalize }

let captured_log_prefix = ref "!@#"

type captured_log_processor = { log_processor_prefix : string; process_logs : string list -> unit }

let captured_log_processors : captured_log_processor list ref = ref []

let add_log_processor ~prefix process_logs =
  captured_log_processors :=
    { log_processor_prefix = prefix; process_logs } :: !captured_log_processors

external input_scan_line : Stdlib.in_channel -> int = "caml_ml_input_scan_line"
external flush_c_streams : unit -> unit = "ocannl_flush_c_streams"

let input_line chan =
  let n = input_scan_line chan in
  if n = 0 then raise End_of_file;
  let line = Stdlib.really_input_string chan (abs n) in
  ( n > 0,
    String.chop_suffix_if_exists ~suffix:"\n" @@ String.chop_suffix_if_exists line ~suffix:"\r\n" )

let capture_stdout_logs arg =
  if never_capture_stdout () || not (debug_log_from_routines ()) then arg ()
  else (
    Stdlib.flush Stdlib.stdout;
    flush_c_streams ();
    (* Ensure previous stdout is flushed *)
    let original_stdout_fd = Unix.dup Unix.stdout in

    let pipe_read_fd, pipe_write_fd = Unix.pipe ~cloexec:true () in
    Unix.dup2 pipe_write_fd Unix.stdout;

    (* pipe_write_fd is now the new Stdlib.stdout, do not close it in parent until done. *)
    (* The reader domain will close pipe_read_fd. *)
    let collected_logs_ref = ref [] in
    let reader_domain_failed = Atomic.make false in

    let reader_domain_logic () =
      let in_channel = Unix.in_channel_of_descr pipe_read_fd in
      (* Create an output channel to the original stdout for immediate passthrough *)
      let orig_out = Unix.out_channel_of_descr (Unix.dup original_stdout_fd) in
      try
        while true do
          let _is_endlined, line = input_line in_channel in
          match String.chop_prefix ~prefix:!captured_log_prefix line with
          | Some logline -> collected_logs_ref := logline :: !collected_logs_ref
          | None ->
              (* Forward non-log lines to original stdout immediately *)
              Stdlib.output_string orig_out (line ^ "\n");
              Stdlib.flush orig_out
        done;
        Stdlib.close_out_noerr orig_out;
        Stdlib.close_in_noerr in_channel (* This closes pipe_read_fd *)
      with
      | End_of_file -> () (* Normal termination of the reader *)
      | exn ->
          Stdlib.close_out_noerr orig_out;
          Atomic.set reader_domain_failed true;
          Stdio.eprintf "Exception in stdout reader domain: %s\\nBacktrace:\\n%s\\n%!"
            (Exn.to_string exn)
            (Stdlib.Printexc.get_backtrace ());
          Stdlib.close_in_noerr in_channel (* This closes pipe_read_fd *);
          Stdlib.Printexc.raise_with_backtrace exn (Stdlib.Printexc.get_raw_backtrace ())
    in

    let reader_domain = Domain.spawn reader_domain_logic in

    let result =
      try arg ()
      with exn ->
        (* Ensure cleanup even if arg() fails *)
        Stdlib.flush Stdlib.stdout;
        flush_c_streams ();
        (* Flush to pipe_write_fd *)
        Unix.close pipe_write_fd;
        (* Signal EOF to reader domain *)
        (* Restore stdout before waiting for the reader domain so that the write end of the pipe is
           effectively closed (both the explicit [pipe_write_fd] descriptor above _and_ the
           descriptor 1 obtained via [dup2] earlier). Otherwise the reader domain would never see an
           EOF and [Domain.join] would block indefinitely. *)
        Unix.dup2 original_stdout_fd Unix.stdout;
        (* Restore stdout *)
        Unix.close original_stdout_fd;

        (* Now that all write descriptors for the pipe are closed, we can wait for the reader domain
           to finish. *)
        (try Domain.join reader_domain
         with e ->
           Stdio.eprintf "Exception while joining reader domain (arg failed): %s\\n%!"
             (Exn.to_string e));

        (if not (Atomic.get reader_domain_failed) then
           let captured_output = List.rev !collected_logs_ref in
           List.iter (List.rev !captured_log_processors)
             ~f:(fun { log_processor_prefix; process_logs } ->
               process_logs
               @@ List.filter_map captured_output
                    ~f:(String.chop_prefix ~prefix:log_processor_prefix)));
        captured_log_processors := [];
        (* Clear processors *)
        Stdlib.Printexc.raise_with_backtrace exn (Stdlib.Printexc.get_raw_backtrace ())
    in

    (* Normal path: arg() completed successfully *)
    Stdlib.flush Stdlib.stdout;
    flush_c_streams ();
    (* Flush to pipe_write_fd *)
    Unix.close pipe_write_fd;

    (* Signal EOF to reader domain *)

    (* Restore stdout before waiting for the reader domain so that the write end of the pipe is
       effectively closed and the reader can finish properly. *)
    Unix.dup2 original_stdout_fd Unix.stdout;
    (* Restore stdout *)
    Unix.close original_stdout_fd;

    (try Domain.join reader_domain
     with e ->
       Stdio.eprintf "Exception while joining reader domain (arg succeeded): %s\\n%!"
         (Exn.to_string e);
       if Atomic.get reader_domain_failed then
         Stdlib.Printexc.raise_with_backtrace e (Stdlib.Printexc.get_raw_backtrace ()));

    if not (Atomic.get reader_domain_failed) then
      let captured_output = List.rev !collected_logs_ref in
      Exn.protect
        ~f:(fun () ->
          (* Process captured logs by processors first. *)
          List.iter (List.rev !captured_log_processors)
            ~f:(fun { log_processor_prefix; process_logs } ->
              process_logs
              @@ List.filter_map captured_output
                   ~f:(String.chop_prefix ~prefix:log_processor_prefix)))
        ~finally:(fun () -> captured_log_processors := [])
    else captured_log_processors := [] (* Clear processors if reader failed *);
    result)

let log_debug_routine_logs ~log_contents ~stream_name =
  if get_global_flag ~default:false ~arg_name:"debug_log_to_stream_files" then
    let stream_file_name = diagn_log_file @@ stream_name ^ ".log" in
    Stdio.Out_channel.with_file stream_file_name ~append:true ~f:(fun oc ->
        List.iter log_contents ~f:(fun line -> Stdio.Out_channel.output_line oc line))
  else log_trace_tree log_contents

let log_debug_routine_file ~log_file_name ~stream_name =
  let log_contents = Stdio.In_channel.read_lines log_file_name in
  log_debug_routine_logs ~log_contents ~stream_name;
  Stdlib.Sys.remove log_file_name

let gcd a b =
  let rec loop a b = if b = 0 then a else loop b (a % b) in
  loop (abs a) (abs b)
