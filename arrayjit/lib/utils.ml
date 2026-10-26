open Base

(* gh-ocannl-1440: completed backend compiles on this domain -- each code generation a [Context]
   compile completes, and each device compile a backend defers to link time (Metal's: a code value's
   first link on a device, which compiles its kernels or takes them from the kernel code cache,
   gh-ocannl-1605) -- the work a relink of retained code skips. Read through
   [Context.backend_compiles]. *)
let count_backend_compile, backend_compiles =
  let key = Stdlib.Domain.DLS.new_key (fun () -> ref 0) in
  ((fun () -> Int.incr (Stdlib.Domain.DLS.get key)), fun () -> !(Stdlib.Domain.DLS.get key))

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
[@@deriving sexp_of]

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
      (* Bootstrap keys (resolved without profiles, before ordinary per-key tracing) *)
      "suppress_welcome_message";
      "no_config_file";
      "log_config_sourcing";
      "log_config_json";
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
      "force_clean_up_artifacts_on_startup";
      "clean_up_log_files_on_startup";
      "never_capture_stdout";
      "detach_c_stderr";
      "collapse_foreign_stderr_noise";
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
      "exit_stream_teardown_timeout";
      "device_init_deadline";
      "hip_scratch_validation";
      "hip_kernel_cache_mib";
      "cuda_kernel_cache_mib";
      "cc_kernel_cache_mib";
      "cc_kernel_part_weight";
      "cc_compile_jobs";
      "cc_compile_unit_min_bytes";
      "metal_kernel_cache_mib";
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
      "virtualize_footprint_materialization";
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
      "gpu_schedule_workgroup_fill";
      "gpu_lane_preamble_reduction";
      "gpu_lane_all_reduce_simdgroups";
      "gpu_serial_lanes";
      "automatic_cpu_schedule";
      "cpu_schedule_min_work";
      "schedule_fission";
      "gpu_fission_keep_mapping";
      "gpu_elide_covered_zeros";
      "gpu_default_sketch_min_reduction";
      "gpu_default_split_reduce_min_serial";
      "gpu_default_reduction_sink";
      "gpu_default_seed_narrow_storage";
      "schedule_log_launches";
      "schedule_log_declines";
      "legality_crosscheck";
      "debug_check_upload_arenas";
      "cc_parallel_grid";
      "cc_parallel_chunks";
      "cc_pool_core_class";
      "cc_grid_private_bytes_cap";
      "cc_grid_min_updates";
      "cc_vector_bytes";
      "cc_vector_registers";
      "cc_fp16_arithmetic";
      (* Autotuning (Autotune.tune) *)
      "autotune_search";
      "autotune_beam_width";
      "autotune_rounds";
      "autotune_repeats";
      "autotune_timing";
      "autotune_cache_dir";
      "autotune_split_reduce_max_sites";
      "autotune_register_tile_rm_twin";
      "autotune_pad_waste_max";
      "autotune_composite_playoff_width";
      "autotune_composite_playoff_margin";
      "autotune_staffed_singles";
      "autotune_round_expansion";
      "autotune_budget_seconds";
      "autotune_log";
      "autotune_progress";
      "autotune_log_seeding";
      "autotune_strict_replay";
      "tune_inline_flips";
      "tune_call_budget_seconds";
      "tune_flip_ordering";
      "tune_flip_profit_margin";
      "tune_ship_arm";
      "tune_placement_store";
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
      "model_cache_capacity";
      "model_cache_bandwidth";
      "model_issue_rate";
      "model_dram_bandwidth";
      "model_mma_rate";
      "model_device_gather_rate";
      "model_multiprocessors";
      "model_fill_threads";
      (* Numerics policy (gh-ocannl-478) *)
      "tf32_matmuls";
      "narrow_compute_f32";
      "fp16_arithmetic";
      "bf16_arithmetic";
      (* Licences of the floating-point algebra over lowered code (gh-ocannl-998) *)
      "fp_contract";
      "fp_reassociate";
      "fp_finite_math";
      (* Algebraic rewrites over lowered code *)
      "online_softmax";
      "online_softmax_backward";
      "online_softmax_block";
      "winograd_conv";
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
      (** Not fully captured by the lowered code, so it has to be carried explicitly — by this named
          component of [Ir.Schedule_cache.key_components]. This is the class gh-ocannl-568 was an
          omission from. *)
  | Search_shaping
      (** Shapes which schedule a search proposes, times or crowns, but not what a crowned one means
          or how fast it then runs: a saved schedule carries its own ops and replay re-derives
          nothing from these. Two processes differing only here may find different winners; each is
          a valid winner for the other. *)
  | Placement_shaping
      (** Steers [Train.tune_placements] above any single search: which placement programs it
          searches (each a different program, keyed on its own digest) and which of them ships. No
          single [Autotune.tune] call reads it, so it is neither in the cache key nor in the search
          trajectory's identity ({!config_class_fingerprint} of [Search_shaping]): a search's timed
          record is the same record whatever the level above it was told (gh-ocannl-1192). *)
  | Untimed_model
      (** Steers the untuned regime's untimed pick: [Autotune.model_default], the pick by the
          analytic cost model -- whether a recipe's untuned compile routes through it, and how
          widely it searches. It times nothing and neither replays nor stores a schedule-cache
          entry, and no [Autotune.tune] call reads these keys, so they are neither in the cache key
          nor in the search trajectory's identity ({!config_class_fingerprint} of [Search_shaping]):
          a search's timed record is the same record whatever the untuned regime beside it was told
          (gh-ocannl-1241). [Autotune.seeded_default]'s keys started here and are [Search_shaping]
          since the search seeds its plan first (gh-ocannl-1388). *)
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
        "virtualize_footprint_materialization";
        "enable_device_only";
        "inline_scalar_constexprs";
        "inline_simple_computations";
        "inline_complex_computations";
        "memory_budget";
        "online_softmax";
        "online_softmax_backward";
        "online_softmax_block";
        "winograd_conv";
      ] );
    ( Code_borne,
      "a licence of the floating-point algebra, applied once by the rewrite tier on the raw \
       lowering (gh-ocannl-998): the rewritten code carries it, and the post-scheduling \
       simplification is exact, so no transform consults it",
      [ "fp_contract"; "fp_reassociate"; "fp_finite_math" ] );
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
      [ "tf32_matmuls"; "narrow_compute_f32"; "fp16_arithmetic"; "bf16_arithmetic" ] );
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
        "cc_kernel_part_weight";
        "cc_vector_bytes";
        "cc_vector_registers";
        "cc_fp16_arithmetic";
        "cc_parallel_grid";
        "cc_parallel_chunks";
        "cc_grid_private_bytes_cap";
        "cc_grid_min_updates";
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
    ( Keyed "codegen",
      "it rewrites a GPU routine's kernel list after the transform that produced it -- the zeros \
       it launches and the accumulations they fed (gh-ocannl-1257) -- so it is not a property of \
       the lowered code the digest names, and a search times every candidate in the form it ships \
       in",
      [ "gpu_elide_covered_zeros" ] );
    ( Search_shaping,
      "it defines the untuned default pipeline -- its schedules and, on GPU backends, its fission \
       segmentation (since gh-ocannl-1126 the merge decision compares the default GPU schedule's \
       per-statement mappings) -- which a search seeds from and reports against; a cached winner \
       carries its own ops, and a fissioned one its segmentation and every segment's schedule, \
       zero expansions included, so it replays without consulting these (gh-ocannl-1164; \
       [Schedule.default_schedule_fingerprint] records them for the [default_ms] diagnostic)",
      [
        "automatic_gpu_schedule";
        "automatic_cpu_schedule";
        "cpu_schedule_min_work";
        "schedule_fission";
        "gpu_fission_keep_mapping";
        "gpu_schedule_block_size";
        "gpu_schedule_min_parallel";
        "gpu_schedule_workgroup_fill";
        "gpu_lane_preamble_reduction";
        "gpu_lane_all_reduce_simdgroups";
        "gpu_serial_lanes";
      ] );
    ( Search_shaping,
      "they define the seeded untuned plan -- which large CUDA/HIP contractions the recipes' \
       untuned compile seeds with the matmul family's sketch (Autotune.seeded_default, \
       gh-ocannl-1324), which long reductions it splits before segmenting (gh-ocannl-1325), \
       whether it sinks the reduction loops of the segments it does not seed (gh-ocannl-1348) -- \
       which a search seeds as its FIRST candidate and floor (gh-ocannl-1388), so they decide what \
       it times first; a cached winner carries its own ops and segmentation and replays without \
       consulting them",
      [
        "gpu_default_sketch_min_reduction";
        "gpu_default_split_reduce_min_serial";
        "gpu_default_reduction_sink";
      ] );
    ( Keyed "codegen",
      "a numerics gate on the seeded untuned plan (gh-ocannl-1405): it admits narrow-storage \
       sites, and a sunk narrow reduction rounds to storage once where the default rounds every \
       step, so a winner crowned from that plan is not the default regime's answer and must not \
       replay under it; AGENTS.md's numerics-gate rule. Hashed into the codegen tag on CUDA and \
       HIP, where the plan runs, when on",
      [ "gpu_default_seed_narrow_storage" ] );
    ( Search_shaping,
      "it steers the search: how wide, how long, what is proposed, what is pruned, how candidates \
       are ranked, and when a candidate's failure ends the search",
      [
        "autotune_search";
        "autotune_beam_width";
        "autotune_rounds";
        "autotune_repeats";
        "autotune_split_reduce_max_sites";
        "autotune_register_tile_rm_twin";
        "autotune_pad_waste_max";
        "autotune_composite_playoff_width";
        "autotune_composite_playoff_margin";
        "autotune_staffed_singles";
        "autotune_round_expansion";
        "autotune_keep_fraction";
        "autotune_bound_pruning";
        "autotune_calibration_file";
        "model_peak_flops";
        "model_peak_memory_bandwidth";
        "strict_failure_classification";
      ] );
    ( Search_shaping,
      "a wall-clock budget on the search (gh-ocannl-1409): it decides how many candidates are \
       attempted before the search crowns its incumbent, never what a crowned schedule means or \
       how fast it runs. The crown of a search it cut is stored marked as cut \
       (Schedule_cache.entry's budget_cut), so only a call that also has a budget replays it -- an \
       incomplete measurement set must not become every later call's answer, the rule that keeps a \
       contended search out of the cache",
      [ "autotune_budget_seconds" ] );
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
    ( Placement_shaping,
      "it makes the tuner try alternative inlining decisions; each alternative is a different \
       program and keys on its own digest",
      [ "tune_inline_flips"; "tune_flip_ordering"; "tune_flip_profit_margin" ] );
    ( Placement_shaping,
      "a wall-clock budget on the whole Train.tune_placements call (gh-ocannl-1440), split among \
       its searches: no Autotune.tune reads the key -- a search receives only its share \
       (Autotune.call_budget), which cuts it by autotune_budget_seconds' rule and marks a cut \
       crown budget_cut, so only a budgeted call replays it. Out of the trajectory identity \
       because the share changes with the wall clock, and a cut trajectory is a prefix of the same \
       candidate order, which early stopping reads conservatively (a slower incumbent abandons \
       fewer flips)",
      [ "tune_call_budget_seconds" ] );
    ( Placement_shaping,
      "it decides which of the two searched placement arms ships, overriding the measured \
       comparison rather than changing either arm: each arm is a different program keyed on its \
       own digest, each arm's crown is cached under that digest either way, and a schedule crowned \
       under one setting is a valid crown under the other (gh-ocannl-638)",
      [ "tune_ship_arm" ] );
    ( Placement_shaping,
      "it decides whether the placement decision is replayed from the placement-decision store or \
       re-derived by comparing the two arms (gh-ocannl-1020): each arm and each flip is a \
       different program keyed on its own digest and cached either way, the store holds only which \
       of them shipped, and a decision re-derived from the arms is as valid a winner as a replayed \
       one",
      [ "tune_placement_store" ] );
    ( Untimed_model,
      "it steers Autotune.model_default -- whether the recipes' untuned compiles route through it, \
       how many placement levels it branch-and-bounds, whether its family search lifts the \
       tile-size lattice -- which picks by the analytic model without timing anything and never \
       reads or writes the schedule cache; Autotune.tune reads none of them. The envelope keys \
       (model_peak_flops, model_peak_memory_bandwidth) stay search-shaping: tune's keep-fraction \
       ranking and bound pruning read them too",
      [ "model_default_schedule"; "model_default_placements"; "model_default_geometry_lattice" ] );
    ( Untimed_model,
      "they are the cache capacity, cache bandwidth, scalar issue rate, DRAM rate, tile-MMA rate, \
       device-gather rate and launch-fill constants of the analytic model's ranking score \
       (gh-ocannl-1253, gh-ocannl-1442, gh-ocannl-1479, gh-ocannl-1495), which only the untimed \
       choosers read (Autotune.model_default's pick, and the auto seeding gate of \
       Autotune.seeded_default) -- like the backends' advisory cache_capacity_bytes, \
       cache_bandwidth, issue_rate, dram_bandwidth, mma_rate, device_gather_rate, multiprocessors \
       and fill_threads they override, Untimed fields of the device description and outside the \
       cache key for the same reason: a stored winner is a measured schedule, which they change \
       neither the compilation nor the speed of. Autotune.tune's bound pruning and keep-fraction \
       ranking read the distinct-cell roofline; the search reaches them only through the seeded \
       plan it seeds first, which joins the trajectory identity as its content digest and, for a \
       staffed start, as a gate tag naming these readings",
      [
        "model_cache_capacity";
        "model_cache_bandwidth";
        "model_issue_rate";
        "model_dram_bandwidth";
        "model_mma_rate";
        "model_device_gather_rate";
        "model_multiprocessors";
        "model_fill_threads";
      ] );
    ( Execution_neutral,
      "startup and configuration-sourcing chatter, which descriptor foreign C-library chatter \
       follows, and whether its known per-event noise is collapsed into a count",
      [
        "suppress_welcome_message";
        "log_config_sourcing";
        "log_config_json";
        "never_capture_stdout";
        "detach_c_stderr";
        "collapse_foreign_stderr_noise";
      ] );
    ( Execution_neutral,
      "debug artifacts and where they go: the files are written beside the run, the kernels are \
       what they would have been",
      [
        "output_dlls_in_build_directory";
        "build_files_prefix";
        "clean_up_build_files_on_startup";
        "force_clean_up_artifacts_on_startup";
        "clean_up_log_files_on_startup";
        "output_prec_in_ll_files";
        "autotune_cache_dir";
        "autotune_log";
        "autotune_progress";
        "autotune_log_seeding";
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
        "autotune_strict_replay";
        "debug_check_upload_arenas";
        "hip_scratch_validation";
        "cc_backend_verify_codesign";
        "cc_backend_post_compile_timeout";
        "max_shape_error_origins";
        "device_init_deadline";
      ] );
    ( Execution_neutral,
      "reuse of compiled kernel code within a process, keyed by the kernel's exact source and \
       compiler options, so a reused kernel was compiled from the same text under the same \
       options: only the compile's cost changes",
      [
        "hip_kernel_cache_mib";
        "cuda_kernel_cache_mib";
        "cc_kernel_cache_mib";
        "metal_kernel_cache_mib";
      ] );
    ( Execution_neutral,
      "how a cc batch is split into translation units and how many compiler processes build them \
       at once (gh-ocannl-1603): every function is compiled alone, never inlined across units, so \
       only the compile's cost changes",
      [ "cc_compile_jobs"; "cc_compile_unit_min_bytes" ] );
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
        "exit_stream_teardown_timeout";
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

(** How a configuration key's value parses, for telling two spellings of one value apart from two
    values (gh-ocannl-1272) and a value from text that is none (gh-ocannl-1332). Each kind is THE
    parse of its keys: {!canonical_config_value} is both what the trajectory identity compares and
    what the typed getters ({!registered_int} and its siblings) read, so two spellings a kind calls
    equal are two the program cannot tell apart either, and text a kind does not parse is refused at
    every read. *)
type config_value_kind =
  | Flag  (** {!bool_of_config_string}: [true]/[1] and [false]/[0], any case, stripped *)
  | Integer  (** [Int.of_string] of the stripped text *)
  | Float_number  (** [Float.of_string] of the stripped text, NaN excluded *)
  | Keyword
      (** a mode name, stripped and lowercased, such as [auto]; which names are modes is the read
          site's [~parse] ({!registered_keyword}), so the kind itself refuses nothing *)
  | Text  (** stripped text, such as a path *)
  | Positive_float_or_unset
      (** an override that only a positive float sets: empty and non-positive numbers mean "not
          overridden", as [Autotune.peak_override] reads it *)
  | Float_at_least of { floor : float }
      (** a float clamped up to [floor], NaN excluded: every spelling the read site cannot tell
          apart from another canonicalizes alike, so the read ([Sketch_families.pad_waste_max]) and
          the fingerprint share one owner of the clamp *)

(** The value [s] has under [kind], spelled canonically, or [None] when [s] does not parse as one.
*)
let canonical_config_value kind s =
  let s = String.strip s in
  let number s = Option.filter (Float.of_string_opt s) ~f:(fun f -> not (Float.is_nan f)) in
  match kind with
  | Flag -> (
      match String.lowercase s with
      | "true" | "1" -> Some "true"
      | "false" | "0" -> Some "false"
      | _ -> None)
  | Integer -> Option.map (Int.of_string_opt s) ~f:Int.to_string
  | Float_number -> Option.map (number s) ~f:Float.to_string
  | Keyword -> Some (String.lowercase s)
  | Text -> Some s
  | Positive_float_or_unset -> (
      if String.is_empty s then Some ""
      else
        match number s with
        | Some f when Float.(f > 0.) -> Some (Float.to_string f)
        | Some _ -> Some ""
        | None -> None)
  | Float_at_least { floor } ->
      Option.map (number s) ~f:(fun f -> Float.to_string (Float.max floor f))

(** The built-in default of every key whose RESOLVED value an identity renders -- today the
    [Search_shaping] class, which {!config_class_fingerprint} renders into [Autotune.tune]'s search
    trajectory identity -- with the kind its value parses as. The fingerprint has to know the
    default to leave out a key stated at it (gh-ocannl-1272), and this row is the default's only
    copy: a key listed here is read through the typed getter of its kind ({!registered_int} and its
    siblings, gh-ocannl-1332), which takes no [~default:] and refuses a value its kind does not
    parse, and {!get_global_arg_with_source} refuses any other read of it.
    [test/operations/config_class_fingerprint] holds the [Search_shaping] class equal to this list's
    keys, and [test/operations/test_config_consistency] every read of a listed key to its kind's
    getter. *)
let config_key_defaults : (string * (config_value_kind * string)) list =
  [
    ("automatic_gpu_schedule", (Flag, "true"));
    ("automatic_cpu_schedule", (Flag, "true"));
    ("cpu_schedule_min_work", (Integer, "16384"));
    ("schedule_fission", (Flag, "true"));
    ("gpu_fission_keep_mapping", (Flag, "true"));
    ("gpu_schedule_block_size", (Integer, "256"));
    ("gpu_schedule_min_parallel", (Integer, "64"));
    ("gpu_schedule_workgroup_fill", (Integer, "256"));
    ("gpu_lane_preamble_reduction", (Keyword, "auto"));
    ("gpu_lane_all_reduce_simdgroups", (Keyword, "auto"));
    ("gpu_serial_lanes", (Keyword, "auto"));
    ("gpu_default_sketch_min_reduction", (Keyword, "auto"));
    ("gpu_default_split_reduce_min_serial", (Integer, "1024"));
    ("gpu_default_reduction_sink", (Flag, "true"));
    ("autotune_search", (Flag, "true"));
    ("autotune_beam_width", (Integer, "2"));
    ("autotune_rounds", (Integer, "2"));
    ("autotune_repeats", (Integer, "3"));
    ("autotune_split_reduce_max_sites", (Integer, "8"));
    ("autotune_register_tile_rm_twin", (Flag, "false"));
    ("autotune_pad_waste_max", (Float_at_least { floor = 1. }, "2."));
    ("autotune_composite_playoff_width", (Integer, "2"));
    ("autotune_composite_playoff_margin", (Float_number, "0.005"));
    ("autotune_staffed_singles", (Flag, "false"));
    ("autotune_round_expansion", (Keyword, "differing"));
    ("autotune_budget_seconds", (Positive_float_or_unset, ""));
    ("autotune_keep_fraction", (Float_number, "1."));
    ("autotune_bound_pruning", (Flag, "false"));
    ("autotune_calibration_file", (Text, ""));
    ("model_peak_flops", (Positive_float_or_unset, ""));
    ("model_peak_memory_bandwidth", (Positive_float_or_unset, ""));
    ("strict_failure_classification", (Flag, "true"));
  ]

let registered_config_default key = List.Assoc.find config_key_defaults key ~equal:String.equal

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

(** Whether to print every resolved setting to stderr as one JSON line (gh-ocannl-1566): the
    machine-readable sibling of {!log_config_sourcing}, carrying the same records {!resolved_config}
    returns, rendered by {!config_record_to_json}. Bootstrapped the same way, and opt-in for the
    same reason. *)
let log_config_json = ref false

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

(** Whether the test suite's ambient record carries the variable [name] (gh-ocannl-1149): every name
    {!classify_env_var} addresses to the configuration -- a key, an unknown key, an unread spelling
    -- except [OCANNL_BACKEND], whose invalidation each dune stanza states itself, and the reserved
    tooling and tracing-gate namespaces. [test/config/ocannl_ambient_env.ml] writes the record from
    this answer and [test/operations/env_var_deps] refuses an [(env_var …)] declaration of a name it
    is [true] of, so the two cannot disagree on which variables a declaration would duplicate
    (gh-ocannl-1425). *)
let env_var_in_ambient_record ?case_insensitive name =
  match classify_env_var ?case_insensitive name with
  | Env_not_addressed | Env_reserved _ -> false
  | Env_config_key "backend" -> false
  | Env_config_key _ | Env_unknown_key _ | Env_unread_spelling _ | Env_unread_reserved _ -> true

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

(** Candidate prefixes for an argument setting [n]: a spelling from {!cmdline_var_names} followed by
    [_], [-], [=] or nothing. {!cmdline_arg_value} decides whether a candidate belongs to [n] or to
    a longer registered key before taking the remainder as its value. *)
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

(** Why an observed token has no configuration reading. These reasons describe syntax only; registry
    membership is always the consumer's decision. *)
type config_token_rejection =
  | Not_configuration_syntax
  | Empty_qualified_key
  | Invalid_command_line_key
  | Invalid_environment_key
  | Invalid_documentation_key

type config_token_parse =
  | Accepted of config_token
  | Ambiguous of {
      explicit : config_token option;
      candidates : config_token list;
      fallback : config_token;
    }
  | Rejected of { reason : config_token_rejection; qualified : config_token option }

(** Classify a token without consulting {!known_config_keys}. A uniform command-line name before [=]
    is the preferred explicit reading, but alternate separators and concatenated values can also
    give shorter keys. [Ambiguous] carries all those runtime-shaped candidates, longest first, plus
    the normalized full name for diagnosing removed or invalid keys. Consumers select a candidate
    through registry membership or an explicit site judgment; they need not re-lex it.
    [documentation:true] also admits lowercase snake-case assignments. Lowercase one-word names are
    ambiguous with ordinary prose/API assignments, so their single candidate needs a judgment.
    Runtime configuration readers are unchanged. *)
let parse_config_token ?(documentation = false) token =
  let token = String.strip token in
  let normalize key = String.lowercase key |> String.tr ~target:'-' ~replacement:'_' in
  let make token_shape token_key = { token_shape; token_key } in
  let reject ?qualified reason = Rejected { reason; qualified } in
  let name = Option.value_map (String.lsplit2 token ~on:'=') ~default:token ~f:fst in
  match
    List.find_map config_token_command_line_prefixes ~f:(fun prefix ->
        Option.map (String.chop_prefix name ~prefix) ~f:(fun raw_key -> (prefix, raw_key)))
  with
  | Some (prefix, raw_key) -> (
      let uppercase = String.equal prefix (String.uppercase prefix) in
      let valid key =
        (not (String.is_empty key))
        && String.for_all key ~f:(fun c ->
            Char.is_digit c || Char.equal c '_' || Char.equal c '-'
            || if uppercase then Char.is_uppercase c else Char.is_lowercase c)
        && not (String.contains key '_' && String.contains key '-')
      in
      let fallback = make Command_line_token (normalize raw_key) in
      if String.is_empty raw_key then reject Empty_qualified_key
      else
        let explicit = Option.some_if (valid raw_key) fallback in
        (* Derive candidates from the same prefixes the runtime reader consumes, including its
           legacy concatenated-value spelling. The scan only resolves these candidates. *)
        let candidates =
          List.init (String.length raw_key) ~f:(fun index ->
              let raw = String.prefix raw_key (index + 1) in
              if not (valid raw) then None
              else
                let key = normalize raw in
                let matches =
                  cmdline_var_prefixes ~qualified_only:true key
                  |> List.exists ~f:(fun candidate ->
                      String.is_prefix token ~prefix:candidate
                      && String.length token > String.length candidate)
                in
                Option.some_if matches (make Command_line_token key))
          |> List.filter_opt
          |> List.dedup_and_sort ~compare:(fun left right ->
              let length =
                Int.compare (String.length right.token_key) (String.length left.token_key)
              in
              if Int.equal length 0 then String.compare left.token_key right.token_key else length)
          |> List.filter ~f:(fun candidate ->
              not (String.equal candidate.token_key fallback.token_key))
        in
        match (explicit, candidates) with
        | Some parsed, [] -> Accepted parsed
        | _, _ :: _ -> Ambiguous { explicit; candidates; fallback }
        | None, [] -> reject ~qualified:fallback Invalid_command_line_key)
  | None -> (
      match String.lsplit2 token ~on:'=' with
      | None -> reject Not_configuration_syntax
      | Some (raw_name, _value) -> (
          match String.chop_prefix raw_name ~prefix:"OCANNL_" with
          | Some raw_key ->
              if String.is_empty raw_key then reject Empty_qualified_key
              else if
                String.for_all raw_key ~f:(fun c ->
                    Char.is_uppercase c || Char.is_digit c || Char.equal c '_')
              then Accepted (make Environment_assignment_token (String.lowercase raw_key))
              else reject Invalid_environment_key
          | None when documentation ->
              let name = String.strip raw_name in
              let key = Option.value (String.chop_prefix name ~prefix:"ocannl_") ~default:name in
              let valid =
                (not (String.is_empty name))
                && Char.is_lowercase name.[0]
                && String.for_all name ~f:(fun c ->
                    Char.is_lowercase c || Char.is_digit c || Char.equal c '_')
                && not (String.is_empty key)
              in
              if not valid then reject Invalid_documentation_key
              else
                let parsed = make Documentation_assignment_token key in
                if String.contains name '_' then Accepted parsed
                else Ambiguous { explicit = None; candidates = [ parsed ]; fallback = parsed }
          | None -> reject Not_configuration_syntax))

(* Keys whose prefix-free command-line spellings are never claimed: common application flags a host
   executable is likely to own ({!cmdline_var_names}' [qualified_only] doc; Codex P2 on PR #291).
   The single source of the policy — {!read_cmdline_var}'s default and {!cmdline_arg_is_config_key}
   both derive from it, so a spelling the resolver would ignore is never mistaken for a claimed
   argument (gh-ocannl-578). *)
let qualified_only_config_keys = Set.of_list (module String) [ "profile" ]

(** Read one argument for [n]. A longer registered key owns its spelling, even when the shorter key
    could interpret its suffix as a value separated by [_], [-], or nothing. [=] ends the name
    unambiguously. Unregistered suffixes retain the legacy value grammar.

    This is the single source of truth for reading an argument's value. For registered keys under
    their default qualification policy, the unknown-argument classifier uses the union of
    {!cmdline_var_prefixes}: ownership cannot change that union, because an argument a shorter key
    gives up is still claimed by its longest registered extension. Both paths use the same spelling
    table, so they cannot disagree about whether an argument is read. *)
let cmdline_arg_value ?qualified_only n =
  let qualified_only =
    Option.value qualified_only ~default:(Set.mem qualified_only_config_keys n)
  in
  let normalized = String.tr n ~target:'-' ~replacement:'_' in
  let longer_names =
    Set.to_list known_config_keys
    |> List.filter ~f:(fun key ->
        String.length key > String.length normalized && String.is_prefix key ~prefix:normalized)
    |> List.concat_map ~f:(fun key ->
        cmdline_var_names ~qualified_only:(Set.mem qualified_only_config_keys key) key)
  in
  let prefixes = cmdline_var_prefixes ~qualified_only n in
  fun arg ->
    if List.exists longer_names ~f:(fun longer -> String.is_prefix arg ~prefix:longer) then None
    else List.find_map prefixes ~f:(fun prefix -> String.chop_prefix arg ~prefix)

(** The commandline sublevel of {!get_global_arg}: returns the setting's value and the [Sys.argv]
    element it came from. Pure -- the sourcing log lives at the resolution seam, which is the only
    place that knows which sublevel actually won.

    [qualified_only] defaults per key from {!qualified_only_config_keys}, so a caller need not
    remember which keys renounce their prefix-free spellings. The first argument for that exact key
    wins, independent of arguments for longer registered keys. *)
let read_cmdline_var ?qualified_only n =
  let value_of_arg = cmdline_arg_value ?qualified_only n in
  Array.find_map Stdlib.Sys.argv ~f:(fun arg ->
      Option.map (value_of_arg arg) ~f:(fun value -> (value, arg)))

(** Whether a raw command-line argument addresses a known configuration key under {e any} spelling
    {!read_cmdline_var} accepts — prefixed or prefix-free, dashed or underscored, any separator. For
    executables that parse their own flags (tools/): such an argument belongs to the config
    machinery and should be passed over rather than rejected as unknown, while an argument matching
    no known key under any spelling can still be flagged as a probable typo. *)
let cmdline_arg_is_config_key arg =
  Set.exists known_config_keys ~f:(fun k ->
      let qualified_only = Set.mem qualified_only_config_keys k in
      List.exists (cmdline_var_prefixes ~qualified_only k) ~f:(fun prefix ->
          String.is_prefix arg ~prefix))

(** The environment sublevel of {!get_global_arg}: returns the setting's value and the variable it
    came from. An empty value counts as unset. *)
let read_env_var n =
  let env_n = env_var_name n in
  match Option.(join @@ map (Stdlib.Sys.getenv_opt env_n) ~f:(str_nonempty ~f:(pair env_n))) with
  | None | Some (_, "") -> None
  | Some (env_n, result) -> Some (result, env_n)

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

(** Bootstrap resolution uses the ordinary precedence walk, with no profile and, until the file
    exists, no file source. Keep the returned provenance for the deferred startup trace. *)
let resolve_bootstrap_config ?(file = fun _ -> None) ~default ~arg_name () =
  resolve_config_value ~cmdline:read_cmdline_var ~env:read_env_var ~file ~profile:None ~default
    ~arg_name

let no_config_file_resolution =
  resolve_bootstrap_config ~default:"false" ~arg_name:"no_config_file" ()

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

let early_log_config_sourcing_resolution =
  resolve_bootstrap_config ~default:"false" ~arg_name:"log_config_sourcing" ()

let () =
  log_config_sourcing :=
    bool_of_config_string ~arg_name:"log_config_sourcing" (fst early_log_config_sourcing_resolution);
  log_config_json :=
    bool_of_config_string ~arg_name:"log_config_json"
      (fst (resolve_bootstrap_config ~default:"false" ~arg_name:"log_config_json" ()))

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

let config_file_args, config_file_path, suppress_welcome_message_resolution =
  let early_suppression =
    resolve_bootstrap_config ~default:"false" ~arg_name:"suppress_welcome_message" ()
  in
  let suppress_welcome_message () = Bool.of_string (fst early_suppression) in
  match fst no_config_file_resolution with
  | "false" ->
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
      let suppression =
        resolve_bootstrap_config ~file:(Hashtbl.find result) ~default:"false"
          ~arg_name:"suppress_welcome_message" ()
      in
      if String.length fname > 0 && not (Bool.of_string (fst suppression)) then
        Stdio.eprintf "\nWelcome to OCANNL! Reading configuration defaults from %s.\n%!" fname;
      (result, fname, suppression)
  | _ ->
      if not (suppress_welcome_message ()) then
        Stdio.eprintf "\nWelcome to OCANNL! Configuration defaults file is disabled.\n%!";
      (Hashtbl.create (module String), "", early_suppression)

let log_config_sourcing_resolution =
  resolve_bootstrap_config ~file:(Hashtbl.find config_file_args) ~default:"false"
    ~arg_name:"log_config_sourcing" ()

let log_config_json_resolution =
  resolve_bootstrap_config ~file:(Hashtbl.find config_file_args) ~default:"false"
    ~arg_name:"log_config_json" ()

let () =
  log_config_sourcing :=
    bool_of_config_string ~arg_name:"log_config_sourcing" (fst log_config_sourcing_resolution);
  log_config_json :=
    bool_of_config_string ~arg_name:"log_config_json" (fst log_config_json_resolution)

(** {2 Configuration profiles (gh-ocannl-559)} *)

(** The keys a profile payload may not set: they are read before profiles are resolved (or would
    make the resolution recursive). *)
let profile_ineligible_keys =
  Set.of_list
    (module String)
    [
      "profile";
      "no_config_file";
      "log_config_sourcing";
      "log_config_json";
      "suppress_welcome_message";
    ]

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
# The lowered code's own licences of the same kinds (gh-ocannl-998), pinned at their defaults: it
# contracts no a*b+c into an fma, reassociates no constant, sum or product, and folds no zero
# identity that infinities, NaNs or the sign of zero would contradict. What remains is the
# arithmetic the program wrote, rounded where it wrote it (explicit FMAs, such as a tensorized
# accumulation's, stay).
fp_contract=false
fp_reassociate=false
fp_finite_math=false
cc_backend_fast_math=false
# Explicit SIMD rendering reassociates strict-FP reductions into per-lane accumulator chains,
# so the summation order would follow the probed vector width. 0 keeps the serial order.
cc_vector_bytes=0

# The numerics gates at their exact defaults, so that a profile switch never silently changes
# what math is being done (they are the orthogonal axis; see the issue). fp16_arithmetic=auto
# resolves deterministically per backend (gh-ocannl-680), so pinning the default keeps the
# principle: this profile changes no math, and same-machine runs stay reproducible.
# bf16_arithmetic=auto likewise (gh-ocannl-838).
tf32_matmuls=false
fp16_arithmetic=auto
bf16_arithmetic=auto
narrow_compute_f32=true
# The algebraic-rewrite gates likewise at their defaults: no rewrite reassociates a reduction
# under this profile.
online_softmax=false
online_softmax_backward=false
online_softmax_block=0
winograd_conv=false
# The untuned GPU default's split reductions (gh-ocannl-1325) likewise: deterministic and chosen
# from the routine alone, but they reassociate long sums into a tree over blocks, which this
# profile keeps serial.
gpu_default_split_reduce_min_serial=0
# The untuned GPU default's narrow-storage admission (gh-ocannl-1405) at its default, off: a sunk
# narrow-storage reduction keeps its accumulator wide across the whole reduction, where the default
# pipeline rounds it to storage after every step, so admitting one changes its result.
gpu_default_seed_narrow_storage=false
|}

let performance_profile_payload =
  {|# The fastest configuration AT UNCHANGED SEMANTICS. Result-changing gates (tf32_matmuls,
# fp16_arithmetic's accuracy-for-throughput trade is the one exception below, and the
# algebraic-rewrite tiers) belong to the numerics axis, not to this one.

# Empirical schedule search on, with a wider beam and more rounds than the everyday defaults.
# The benchmark runners (benchmarks/runners/ocannl) pass ~rounds:0, so their tuned cells get
# this beam width but not these rounds: in a partial gpt2_mini/CUDA probe, one round added 150%
# to the one arm that finished it, which kept its seed winner (gh-ocannl-1137).
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

# The floating-point licences (fp_contract, fp_reassociate, fp_finite_math; gh-ocannl-998) stay
# at their defaults, off: they change results, which is the `approximate` profile's axis.
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
# As there, the benchmark runners' tuned cells run none of the rounds (gh-ocannl-1137).
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
# The narrow side of the bf16 accumulator trade (gh-ocannl-838), symmetric with fp16_arithmetic
# above: never the strict f32 residency `false` requests. It resolves as auto everywhere except
# HIP, where auto went wide (gh-ocannl-1051) and this keeps gfx11's bf16-accumulate WMMA.
bf16_arithmetic=true
# The online-softmax attention rewrite (gh-ocannl-483): the softmax normalizer's summation is
# reassociated into a per-row scan, and the probabilities are never materialized.
online_softmax=true
# Its fused backward (gh-ocannl-1002): the attention's score gradient is recomputed per key from
# the forward's row state, so the [seq, seq] probability and score gradients are never stored.
online_softmax_backward=true
# Its single-pass block fold (gh-ocannl-1003): the scores are read once per key block into a tile
# and the value contraction is reassociated as `(sum p * v) / l`. Auto keeps block 16 on
# Metal and CPU, and the two-pass rewrite on CUDA, HIP and unmeasured targets: the block fold
# loses on the measured CUDA and HIP devices (gh-ocannl-1194 and gh-ocannl-1171).
online_softmax_block=auto
# The Winograd F(2x2, 3x3) convolution rewrite (gh-ocannl-505): a 3x3 stride-1 convolution is
# computed in a transformed domain, through 16 channel contractions per 4x4 input tile.
winograd_conv=true
# The C compiler's licence to reassociate (fast-math) and to contract a*b+c into one rounding
# across statements (fp-contract=fast). Both change results per compiler and target, which is
# why `reproducible` pins them off and `performance` leaves them at their defaults.
cc_backend_fast_math=true
cc_backend_fp_contract=fast
# The lowered code's own licences of the same kinds (gh-ocannl-998), on every backend: contract
# a*b+c into an fma, reassociate constants, sums and products (and unroll known integer
# powers), and fold the zero identities x+0 and 0*x as if no operand were infinite, NaN or a
# negative zero. Each applies within one assignment of the raw lowering, never across an
# inlining boundary.
fp_contract=true
fp_reassociate=true
fp_finite_math=true
# The greedy inlining refinement of Train.tune_placements is not numerics-neutral where storage
# is narrower than compute (a materialized node rounds to its storage precision, an inlined one
# does not), so `performance` leaves it alone. Two flips: each costs a full search, so this
# bounds the extra cost at two searches per arm while still exercising the refinement.
tune_inline_flips=2
# The untuned CUDA/HIP default also seeds and sinks the sites whose accumulator is wider than their
# storage (gh-ocannl-1405): a bf16 weight gradient on CUDA, fp8 on CUDA and HIP. A seeded one keeps
# its result; a sunk one (a bias gradient) keeps its summation order but rounds to storage once
# instead of after every step.
gpu_default_seed_narrow_storage=true
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

(** Pick a normalized profile name through ordinary source precedence, without consulting a profile
    payload recursively. *)
let resolve_profile_selection ~cmdline ~env ~file =
  (* Normalize independently BEFORE resolving: an empty commandline profile must fall through to the
     environment or file, including when it was expanded from an unset shell variable. *)
  let normalize name = str_nonempty ~f:Fn.id (String.lowercase (String.strip name)) in
  let normalize_tagged lookup key =
    Option.bind (lookup key) ~f:(fun (value, tag) ->
        Option.map (normalize value) ~f:(fun value -> (value, tag)))
  in
  resolve_config_value ~cmdline:(normalize_tagged cmdline) ~env:(normalize_tagged env)
    ~file:(fun key -> Option.bind (file key) ~f:normalize)
    ~profile:None ~default:"" ~arg_name:"profile"

let profile_selection =
  resolve_profile_selection ~cmdline:read_cmdline_var ~env:read_env_var
    ~file:(Hashtbl.find config_file_args)

(** The profile picked for this run, if any: its level, name, and parsed payload. *)
let active_profile =
  let name, source = profile_selection in
  let picked =
    match source with
    | From_cmdline _ -> Some (Cmdline_level, name)
    | From_env _ -> Some (Env_level, name)
    | From_config_file -> Some (Config_file_level, name)
    | From_default -> None
    | From_profile _ -> assert false (* Profile selection never consults a payload. *)
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

(** {2 Resolved-configuration records (gh-ocannl-1566)} *)

type config_record = { key : string; value : string; source : config_source }
(** One resolution of one setting: the key, the raw string the precedence walk settled on (before a
    typed getter canonicalizes or refuses it), and where it came from. *)

(** A JSON string literal: the double quote, the backslash and the control characters below 0x20
    escaped, every other byte as it is (so UTF-8 passes intact). The same escaping as
    [Bench_out.json_string]. *)
let json_string_literal text =
  let buf = Buffer.create (String.length text + 10) in
  Buffer.add_char buf '"';
  String.iter text ~f:(fun c ->
      match c with
      | '"' -> Buffer.add_string buf "\\\""
      | '\\' -> Buffer.add_string buf "\\\\"
      | c when Char.to_int c < 0x20 ->
          Buffer.add_string buf (Printf.sprintf "\\u%04x" (Char.to_int c))
      | c -> Buffer.add_char buf c);
  Buffer.add_char buf '"';
  Buffer.contents buf

(** One record as the one-line JSON object the [log_config_json] stream carries -- schema version 1,
    every field always present, in this order:
    [{"ocannl_config":1,"key":K,"value":V,"source":S,"layer":L,"profile":P,"origin":O}]. [source] is
    the sublevel kind: [commandline], [environment], [config_file], [profile] or [default]. [layer]
    is the precedence level the value came from: [source] itself, except for a profile value, whose
    layer is the level that picked the profile. [profile] is that profile's name, else [null].
    [origin] is the concrete place that stated the value -- the argv element, the environment
    variable's name, the config file's path, and for a profile value the place that picked the
    profile -- or [null] for a default. The leading field is both the tag a reader picks the lines
    out of stderr by and the schema version. *)
let config_record_to_json ({ key; value; source } : config_record) =
  let level_tag = function
    | Cmdline_level -> "commandline"
    | Env_level -> "environment"
    | Config_file_level -> "config_file"
  in
  let source_tag = function
    | From_cmdline _ -> "commandline"
    | From_env _ -> "environment"
    | From_config_file -> "config_file"
    | From_profile _ -> "profile"
    | From_default -> "default"
  in
  (* The concrete place a non-profile source names: the argv element, the variable, the file. *)
  let explicit_origin = function
    | From_cmdline arg -> Some arg
    | From_env var -> Some var
    | From_config_file -> Some config_file_path
    | From_profile _ | From_default -> None
  in
  let layer, profile, origin =
    match source with
    | From_profile (level, name) ->
        (level_tag level, Some name, explicit_origin (snd profile_selection))
    | _ -> (source_tag source, None, explicit_origin source)
  in
  let opt = function None -> "null" | Some s -> json_string_literal s in
  Printf.sprintf
    {|{"ocannl_config":1,"key":%s,"value":%s,"source":%s,"layer":%s,"profile":%s,"origin":%s}|}
    (json_string_literal key) (json_string_literal value)
    (json_string_literal (source_tag source))
    (json_string_literal layer) (opt profile) (opt origin)

(** [record_config_resolution ~key (value, source)] records one resolution, and prints it when
    [log_config_json] is on. A resolution identical to the key's last record adds nothing; one that
    differs (the sources changed under the process) is a new record, so the change stays visible.
    The bookkeeping and the line share one lock, so the stream's order is the records' order; each
    line goes out in one write.

    [resolved_config ()] is every setting this process has resolved so far, in the order it was
    first resolved (a changed re-resolution appended where it happened): exactly the records the
    [log_config_json] stream printed, whether or not the stream is on. Observational -- it looks
    nothing up and prints nothing. It covers every read through this module's getters, beginning
    with the bootstrap settings; a setting never read is absent, since most defaults belong to their
    read sites. *)
let record_config_resolution, resolved_config =
  let mutex = Stdlib.Mutex.create () in
  let records_rev : config_record list ref = ref [] in
  let last_record = Hashtbl.create (module String) in
  let equal_source s1 s2 =
    match (s1, s2) with
    | From_cmdline a1, From_cmdline a2 | From_env a1, From_env a2 -> String.equal a1 a2
    | From_config_file, From_config_file | From_default, From_default -> true
    | From_profile (l1, n1), From_profile (l2, n2) -> equal_config_level l1 l2 && String.equal n1 n2
    | (From_cmdline _ | From_env _ | From_config_file | From_profile _ | From_default), _ -> false
  in
  let record ~key (value, source) =
    Stdlib.Mutex.protect mutex (fun () ->
        match Hashtbl.find last_record key with
        | Some (last : config_record)
          when String.equal last.value value && equal_source last.source source ->
            ()
        | _ ->
            let r = { key; value; source } in
            Hashtbl.set last_record ~key ~data:r;
            records_rev := r :: !records_rev;
            if !log_config_json then (
              Stdio.Out_channel.output_string Stdio.stderr (config_record_to_json r ^ "\n");
              Stdio.Out_channel.flush Stdio.stderr))
  in
  let resolved () = Stdlib.Mutex.protect mutex (fun () -> List.rev !records_rev) in
  (record, resolved)

(** Bootstrap values cannot trace themselves until the sourcing switches are settled. Report the
    sourced resolutions already used above, without another precedence walk. [no_config_file]
    deliberately has no file source, and every bootstrap read has no profile source. *)
let () =
  let bootstrap =
    [
      ("log_config_sourcing", "false", log_config_sourcing_resolution);
      ("log_config_json", "false", log_config_json_resolution);
      ("no_config_file", "false", no_config_file_resolution);
      ("suppress_welcome_message", "false", suppress_welcome_message_resolution);
      ("profile", "", profile_selection);
    ]
  in
  if !log_config_sourcing then (
    Stdio.eprintf
      "\nOCANNL: settings resolved before the ordinary per-key trace could report them:\n%!";
    List.iter bootstrap ~f:(fun (n, default, (value, source)) ->
        Stdio.eprintf "Retrieving commandline, environment, or config file variable ocannl_%s\n%!" n;
        Stdio.eprintf "%s\n%!" (describe_config_source ~value ~default source)));
  List.iter bootstrap ~f:(fun (key, _, resolution) -> record_config_resolution ~key resolution)

let profile_lookup =
  Option.map active_profile ~f:(fun (level, name, table) ->
      (level, name, fun key -> Hashtbl.find table key))

(* The lookup itself, for {!get_global_arg_with_source} and the typed getters below it. *)
let unchecked_global_arg_with_source ~default ~arg_name:n =
  let with_debug = !log_config_sourcing && not (Hash_set.mem accessed_global_args n) in
  if with_debug then
    Stdio.eprintf "Retrieving commandline, environment, or config file variable ocannl_%s\n%!" n;
  let result, source =
    resolve_config_value ~cmdline:read_cmdline_var ~env:read_env_var
      ~file:(Hashtbl.find config_file_args) ~profile:profile_lookup ~default ~arg_name:n
  in
  if with_debug then Stdio.eprintf "%s\n%!" (describe_config_source ~value:result ~default source);
  record_config_resolution ~key:n (result, source);
  Hash_set.add accessed_global_args n;
  (result, source)

let registered_getter_of_kind = function
  | Flag -> "registered_flag"
  | Integer -> "registered_int"
  | Float_number -> "registered_float"
  | Keyword -> "registered_keyword"
  | Text -> "registered_text"
  | Positive_float_or_unset -> "registered_positive_float_or_unset"
  | Float_at_least _ -> "registered_float_at_least"

(** Retrieves the [arg_name] setting from the commandline, the environment, the config file, or the
    payload of the profile picked at one of those levels; returns [default] if none has it, together
    with where the value came from. A key with a {!config_key_defaults} row is refused: its read
    goes through the typed getter of its kind, which owns its default (gh-ocannl-1332). *)
let get_global_arg_with_source ~default ~arg_name:n =
  Option.iter (registered_config_default n) ~f:(fun (kind, _) ->
      invalid_arg
        (Printf.sprintf
           "OCANNL: %s has a Utils.config_key_defaults row, so it is read through Utils.%s \
            ~arg_name:%S, which takes its default from that row and refuses a value its kind does \
            not parse"
           n (registered_getter_of_kind kind) n));
  unchecked_global_arg_with_source ~default ~arg_name:n

let get_global_arg ~default ~arg_name = fst (get_global_arg_with_source ~default ~arg_name)

(** {2 Typed getters of the registered keys (gh-ocannl-1332)} *)

exception Invalid_config_value of { key : string; value : string; expected : string }
(** A setting whose value does not parse as its key's kind. One shape for every registered key, so a
    caller that wants to report a bad setting matches one exception, and [key] names it: a silent
    fallback ([autotune_rounds=two] running as 2) would leave a run timing, and keying its cached
    trajectory by, a value nobody stated. *)

let () =
  Stdlib.Printexc.register_printer (function
    | Invalid_config_value { key; value; expected } ->
        Some
          (Printf.sprintf "OCANNL: invalid setting ocannl_%s=%S: expected %s" key
             (String.strip value) expected)
    | _ -> None)

(* The registered key [arg_name]'s value, canonical under its kind, read with the registered
   default. [getter] names the caller, which must be its kind's getter. *)
let registered_canonical ~getter ~arg_name =
  match registered_config_default arg_name with
  | None ->
      invalid_arg
        (Printf.sprintf
           "OCANNL: Utils.%s ~arg_name:%S: the key has no Utils.config_key_defaults row" getter
           arg_name)
  | Some (kind, default) -> (
      if not (String.equal getter (registered_getter_of_kind kind)) then
        invalid_arg
          (Printf.sprintf
             "OCANNL: Utils.%s ~arg_name:%S: the key's registered kind is read by Utils.%s" getter
             arg_name (registered_getter_of_kind kind));
      let value = fst (unchecked_global_arg_with_source ~default ~arg_name) in
      match canonical_config_value kind value with
      | Some canonical -> canonical
      | None ->
          (* What a value of [kind] looks like. *)
          let expected =
            match kind with
            | Flag -> "a boolean: true, false, 1 or 0"
            | Integer -> "an integer"
            | Float_number -> "a number"
            | Keyword -> "a mode name"
            | Text -> "text"
            | Positive_float_or_unset -> "a number, or nothing to leave it unset"
            | Float_at_least { floor } ->
                Printf.sprintf "a number (below %g it means %g)" floor floor
          in
          raise (Invalid_config_value { key = arg_name; value; expected }))

let registered_flag ~arg_name =
  String.equal "true" (registered_canonical ~getter:"registered_flag" ~arg_name)

let registered_int ~arg_name =
  Int.of_string (registered_canonical ~getter:"registered_int" ~arg_name)

let registered_float ~arg_name =
  Float.of_string (registered_canonical ~getter:"registered_float" ~arg_name)

let registered_text ~arg_name = registered_canonical ~getter:"registered_text" ~arg_name

let registered_positive_float_or_unset ~arg_name =
  match registered_canonical ~getter:"registered_positive_float_or_unset" ~arg_name with
  | "" -> None
  | positive -> Some (Float.of_string positive)

let registered_float_at_least ~arg_name =
  Float.of_string (registered_canonical ~getter:"registered_float_at_least" ~arg_name)

(** A [Keyword] key's mode, by the read site's own [parse] of the stripped, lowercased value; a
    value [parse] rejects is refused like any other, with [expected] listing the modes. *)
let registered_keyword ~arg_name ~expected ~parse =
  let value = registered_canonical ~getter:"registered_keyword" ~arg_name in
  match parse value with
  | Some mode -> mode
  | None -> raise (Invalid_config_value { key = arg_name; value; expected })

(** [profile_key_source] resolves one key with its source, and [profile_payload_sources] maps it
    over a payload: where each key of the named built-in profile's payload resolves from in this
    process, and to what unless it is the built-in default (whose value the read sites own, not this
    table): [(key, None)] for a key nothing sets, [(key, Some (value, source))] otherwise. The
    benchmark runners report this for the [approximate] payload (gh-ocannl-719): a cell's numerics
    regime is the resolution of these keys, not the name of the profile alone -- an ambient
    [OCANNL_TF32_MATMULS=true] makes an exact cell approximate with no profile picked, and an
    explicit [--ocannl_tf32_matmuls=false] beside [--ocannl_profile=approximate] makes an
    approximate cell neither regime. The key list is the payload's own, so a gate that adds a key to
    the profile is reported without a second list anywhere. Not logged and not recorded as an
    access: the read sites report their own reads. *)
let profile_key_source key =
  match
    resolve_config_value ~cmdline:read_cmdline_var ~env:read_env_var
      ~file:(Hashtbl.find config_file_args) ~profile:profile_lookup ~default:"" ~arg_name:key
  with
  | _, From_default -> None
  | value, source -> Some (value, source)

(** Every key [classify_config_key] puts in [cls] whose resolved value is not its default, as
    [key=value;] in the classification's order: an identity for the resolved settings of one class.
    [Autotune.tune] keys a cached search trajectory by the [Search_shaping] class this way
    (gh-ocannl-1110), so a key classified into it later joins the identity with no second list to
    update.

    What is compared is the VALUE, not whether some source states it (gh-ocannl-1272): a key stated
    at its default -- in any spelling its read sites parse to the same value
    ({!canonical_config_value} of its kind in {!config_key_defaults}) -- renders exactly as a key
    nothing states, and a rendered value is spelled canonically. The reference point is the built-in
    default, not "what the key would resolve to without this statement": the identity is then a
    function of the resolved values alone, so a profile that moves a key off its default renders it
    (as a statement of the same value would), and a statement that restores the built-in default
    under such a profile renders nothing (as the profile-less process it then behaves like does).
    The profile's name is in no identity.

    [resolve] maps a key's canonical value to the value the device it runs on resolves it to, for a
    default that is itself device-dependent: [auto] lane modes, which [Ir.Schedule] resolves from
    the hardware limits. The value it receives and returns is canonical; the default is resolved the
    same way before the comparison, so [cooperative] stated where [auto] resolves to [cooperative]
    is at the default. A key with no registered default renders whenever some source sets it, as
    every key did before gh-ocannl-1272: conservative, never under-keyed.

    [reads] says whether the process the identity is for reads [key] at all (gh-ocannl-1331): a key
    it never reads -- [gpu_schedule_block_size] on a cc search -- renders nothing whatever it is set
    to, unparsable text included, since nothing that process does depends on it. By default every
    key is read. Not logged and not recorded as an access, like [profile_key_source]. *)
let config_class_fingerprint ?(resolve = fun ~key:_ value -> value) ?(reads = fun _ -> true) cls =
  List.concat_map config_key_classification ~f:(fun (c, _, keys) ->
      if Poly.equal c cls then keys else [])
  |> List.filter ~f:reads
  |> List.filter_map ~f:(fun key ->
      Option.bind (profile_key_source key) ~f:(fun (value, _) ->
          match registered_config_default key with
          | None -> Some (key ^ "=" ^ value ^ ";")
          | Some (kind, default) -> (
              match canonical_config_value kind value with
              (* Unparsable: no value to compare, so it renders as stated. Its read refuses it
                 (gh-ocannl-1332), so no search runs under this identity; rendering it keeps a
                 caller that compares identities without reading from equating it with the
                 default. *)
              | None -> Some (key ^ "=" ^ value ^ ";")
              | Some canonical ->
                  let resolved = resolve ~key canonical in
                  let default_resolved =
                    Option.value_map
                      (canonical_config_value kind default)
                      ~default ~f:(resolve ~key)
                  in
                  if String.equal resolved default_resolved then None
                  else Some (key ^ "=" ^ resolved ^ ";"))))
  |> String.concat

let profile_payload_sources name =
  match List.Assoc.find profile_payloads name ~equal:String.equal with
  | None -> invalid_arg ("OCANNL: unknown profile " ^ name)
  | Some text ->
      parse_config_lines ~source:("profile " ^ name) (String.split_lines text)
      |> List.map ~f:(fun (key, _) -> (key, profile_key_source key))

(** Every configuration key this process INHERITED a value for from its environment, with where the
    key actually resolved from (gh-ocannl-1337): [(variable, key, resolution)] per non-empty
    environment variable that {!classify_env_var} calls a spelling of a key, sorted by variable, the
    resolution as {!profile_key_source} gives it. Inherited is not applied: the commandline and a
    commandline-picked profile's payload both outrank the environment, so a variable resolved from
    anything but [From_env] was in the environment and decided nothing. The benchmark runners report
    this beside [regime_knobs], so the driver's stamp of which inherited variables a cell's command
    line outranks -- a prediction from the argv it built -- is checked against the process that ran.
    A variable in a reserved namespace, or naming no key, is no setting and is left out; an unread
    spelling of a key never gets here, being fatal at startup. Not logged and not recorded as an
    access, like [profile_key_source]. *)
let inherited_env_key_sources () =
  Unix.environment () |> Array.to_list
  |> List.filter_map ~f:(fun binding ->
      match String.lsplit2 binding ~on:'=' with
      | Some (_, "") | None -> None
      | Some (name, _) -> Some name)
  |> List.dedup_and_sort ~compare:String.compare
  |> List.filter_map ~f:(fun name ->
      match classify_env_var name with
      | Env_config_key key -> Some (name, key, profile_key_source key)
      | Env_not_addressed | Env_reserved _ | Env_unread_spelling _ | Env_unread_reserved _
      | Env_unknown_key _ ->
          None)

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

external os_process_id : unit -> int = "ocannl_os_process_id"
(** The operating system's id of this process. Not [Unix.getpid], which on Windows answers a
    process-local handle that need not differ between two processes. *)

(* Concurrently running processes of one executable share its artifact subdirectories, and each
   process's startup empties them (the cleanup block below). So a process that writes into one first
   takes a SHARED advisory lock on a lock file beside it, held until the process ends, and the
   startup cleanup tries the lock EXCLUSIVELY, without waiting, for its own duration. When that
   fails a sibling is running, and the cleanup removes only the artifacts whose names record an
   owner that is gone (gh-ocannl-1335). The operating system releases a lock when its process ends,
   killed or not, so a crashed sibling never holds a later cleanup back -- which is why this is a
   lock and not a marker file.

   The lock file sits beside the subdirectory, [build_files/<prefix>.ocannl-lock], not in it, so
   nothing that lists or empties the subdirectory meets it; under the flat layout it is
   [build_files/.ocannl-lock], which the cleanup skips by its suffix. POSIX drops a process's lock
   on a file when ANY descriptor of that file is closed, so the descriptor stays open for the life
   of the process and nothing else opens the file. *)
module Artifacts_lock = struct
  let suffix = ".ocannl-lock"

  (* Absolute, so that a process that changes its working directory between writes holds the lock of
     each directory it writes in, not only the first one's. *)
  let path base =
    let path =
      match artifacts_subdir () with
      | Some p -> filename_concat base (p ^ suffix)
      | None -> filename_concat base suffix
    in
    if Stdlib.Filename.is_relative path then Stdlib.Filename.concat (Stdlib.Sys.getcwd ()) path
    else path

  (* World-writable before the umask: every account that writes artifacts here needs to open it. *)
  let open_ base = Unix.openfile (path base) [ Unix.O_RDWR; Unix.O_CREAT; Unix.O_CLOEXEC ] 0o666

  (* Best effort: where the lock cannot be taken (a file system without locks) the process goes
     unprotected, as every process did before gh-ocannl-1335. Blocking: a starting sibling holds the
     lock exclusively only while it empties the directory, and what this process is about to write
     there must not land in the middle of that. The descriptors are kept, not just the fact: an
     unreachable descriptor would be free to close. *)
  let hold =
    let mutex = Stdlib.Mutex.create () in
    let holds : (string, Unix.file_descr option) Hashtbl.t = Hashtbl.create (module String) in
    fun base ->
      (* With this process's id: POSIX does not carry record locks across [fork], so a forked child,
         inheriting the table, takes the lock afresh. *)
      let key = Printf.sprintf "%d:%s" (os_process_id ()) (path base) in
      Stdlib.Mutex.protect mutex (fun () ->
          if not (Hashtbl.mem holds key) then
            let fd =
              try
                let fd =
                  (* A shared lock needs only read access: an account that cannot write another's
                     lock file still holds it. *)
                  try open_ base
                  with Unix.Unix_error ((Unix.EACCES | Unix.EPERM), _, _) ->
                    Unix.openfile (path base) [ Unix.O_RDONLY; Unix.O_CLOEXEC ] 0
                in
                try
                  Unix.lockf fd Unix.F_RLOCK 0;
                  Some fd
                with Unix.Unix_error _ ->
                  Unix.close fd;
                  None
              with Unix.Unix_error _ -> None
            in
            Hashtbl.set holds ~key ~data:fd)
end

(** The directory generated-code debug files are written to (created if missing):
    [build_files/<prefix>/], where the prefix defaults to the executable's base name. *)
let build_files_dir () = ensure_artifacts_dir "build_files" (artifacts_subdir ())

(** [build_file fname] is the path of the artifact [fname] in {!build_files_dir}. Asking for it
    declares that this process writes there: from then on, the startup cleanup of a concurrently
    starting process of the same executable leaves this process's artifacts alone. *)
let build_file fname =
  let dir = build_files_dir () in
  Artifacts_lock.hold "build_files";
  filename_concat dir @@ clean_filename fname

(** The file extension of a dynamically loaded library on this platform. *)
let dynamic_library_extension = if Sys.win32 then ".dll" else ".so"

external lock_holder_pid : Unix.file_descr -> int = "ocannl_lock_holder_pid"
(** The OS id of a process holding a lock on the descriptor's file that would block an exclusive
    one, by [fcntl(F_GETLK)]: [0] when none does, [-1] when it cannot tell (Windows, or an error).
    With several shared holders it names one of them. *)

external process_alive : int -> bool = "ocannl_process_alive"
(** Whether a process with the given OS id is running now; one that exists but cannot be queried
    counts as running. *)

(** A random token drawn once per process, which tells apart two processes that report the same OS
    id: processes in different PID namespaces (containers) or on different hosts sharing a checkout,
    or a process reusing the id of one that is gone. *)
let process_nonce =
  let nonce =
    lazy
      (let state = Stdlib.Random.State.make_self_init () in
       Printf.sprintf "%04x%04x"
         (Stdlib.Random.State.bits state land 0xffff)
         (Stdlib.Random.State.bits state land 0xffff))
  in
  fun () -> Lazy.force nonce

(** The most bytes of a kernel's name that the file name of an artifact recording its owner carries:
    a kept library's ({!kept_library_name}), a temporary library's ({!temp_library_prefix}), the
    private source copy a debug compile keeps when it fails. The kernel name is informational there
    (the owner fields and the run id make the name unique), and a routine's name has no length
    limit, while a file-name component has one (255 bytes on the common file systems). *)
let library_kernel_name_limit = 64

(** [library_kernel_name base_name] is the kernel name such a file name carries:
    {!clean_filename}[ base_name] when it fits in {!library_kernel_name_limit} bytes; otherwise a
    prefix of it cut at a UTF-8 character boundary, then ['-'] and 8 hex digits of the whole name's
    digest, within the same limit. The digest keeps apart two long names sharing their opening
    bytes, so that a reader matching artifacts by kernel name never takes one routine's for
    another's. *)
let library_kernel_name base_name =
  let name = clean_filename base_name in
  if String.length name <= library_kernel_name_limit then name
  else
    let digest = String.prefix (Stdlib.Digest.to_hex (Stdlib.Digest.string name)) 8 in
    (* Back off from a UTF-8 continuation byte, so no character is split. *)
    let rec cut i = if i > 0 && Char.to_int name.[i] land 0xC0 = 0x80 then cut (i - 1) else i in
    String.prefix name (cut (library_kernel_name_limit - 1 - String.length digest)) ^ "-" ^ digest

(** [owned_artifact_name ~pid ~nonce ~run_id ~extension base_name] is the file name of an artifact
    of the kernel [base_name]'s compile [run_id] that the process [pid] (with {!process_nonce}
    [nonce]) keeps in {!build_files_dir}. The pid and the nonce keep apart the processes that share
    that directory -- concurrently running ones of one executable, and those in other PID namespaces
    -- and the pid is what the startup cleanup asks about; the run id keeps the compiles of one
    process apart. *)
let owned_artifact_name ~pid ~nonce ~run_id ~extension base_name =
  Printf.sprintf "%s_pid%d_n%s_run%d%s" (library_kernel_name base_name) pid nonce run_id extension

(** [kept_library_name ~pid ~nonce ~run_id base_name] is the {!owned_artifact_name} of the library
    the process keeps under [output_dlls_in_build_directory] (gh-ocannl-1247) or
    [output_debug_files_in_build_directory] (gh-ocannl-1335). {!parse_kept_library_name} inverts it.
*)
let kept_library_name ~pid ~nonce ~run_id base_name =
  owned_artifact_name ~pid ~nonce ~run_id ~extension:dynamic_library_extension base_name

type kept_library = {
  kernel_name : string;
  owner_pid : int;
  owner_nonce : string;
  library_run_id : int;
}
(** The owner and the compile a library's file name records: a kept library's
    ([parse_kept_library_name]) or a temporary one's ([parse_temp_library_name]). *)

(** [temp_library_prefix ~pid ~nonce ~run_id base_name] is the [Filename.temp_file] prefix of the
    temporary library that the process [pid] (with {!process_nonce} [nonce]) names for its compile
    [run_id] of the kernel [base_name], and removes once it is loaded (gh-ocannl-1197). Like a kept
    library's name, it records the owner, because Windows refuses to delete a loaded DLL: a library
    whose routine is alive at exit stays in the temp directory, and a later process's startup
    removes it once the owner is gone (gh-ocannl-1268). The run id keeps the path unique within the
    process even after the file is gone, and [temp_file]'s random suffix keeps the name
    unpredictable. [parse_temp_library_name] inverts the name.

    The temp directory is shared with every other program, so the name opens with
    {!temp_library_marker}, which the sweep requires: an owner-shaped name another program chose is
    never taken for OCANNL's. The kernel name is {!library_kernel_name}'s. *)
let temp_library_marker = "ocannl_cc_"

let temp_library_prefix ~pid ~nonce ~run_id base_name =
  Printf.sprintf "%s%s_pid%d_n%s_run_id_%d_" temp_library_marker (library_kernel_name base_name) pid
    nonce run_id

(** How old, in seconds since its last write, a temporary library of a process that is gone must be
    before a startup removes it (gh-ocannl-1268). Liveness alone cannot protect a library whose
    owner is invisible to this process -- one in another PID namespace sharing the temp directory,
    which may even report this process's own OS id -- and such a library is at risk only between its
    compiler finishing and its [dlopen]. An hour is far past that window, macOS's first-load malware
    scan of a fresh library included, while a Windows leftover still goes at a later startup. *)
let temp_library_sweep_min_age = 3600.

(** [parse_kept_library_name fname] inverts {!kept_library_name} on a file name, and
    [parse_temp_library_name fname] reads the owner and the compile from the name of a temporary
    library that {!temp_library_prefix} named; each is [None] for any other file name. Only
    libraries are read so, never an owned source copy: the informational debug source of a routine
    whose NAME is owner-shaped would otherwise pass for another process's. The two shapes are
    disjoint: a kept name ends in [_run<N>], a temporary one in [temp_file]'s hex suffix. Fields are
    read from the right, so a kernel name may itself contain underscores and field-like words. *)
let parse_kept_library_name, parse_temp_library_name =
  let number s =
    if (not (String.is_empty s)) && String.for_all s ~f:Char.is_digit then Int.of_string_opt s
    else None
  in
  let hex s =
    if
      (not (String.is_empty s))
      && String.for_all s ~f:(fun c -> Char.is_digit c || Char.between c ~low:'a' ~high:'f')
    then Some s
    else None
  in
  let field ~prefix ~f s = Option.bind (String.chop_prefix s ~prefix) ~f in
  (* [<kernel>_pid<P>_n<nonce>], the owner part every name shares. *)
  let owner ~library_run_id owner =
    match String.rsplit2 owner ~on:'_' with
    | None -> None
    | Some (rest, nonce) -> (
        match String.rsplit2 rest ~on:'_' with
        | Some (kernel, pid) when not (String.is_empty kernel) -> (
            match (field ~prefix:"pid" ~f:number pid, field ~prefix:"n" ~f:hex nonce) with
            | Some owner_pid, Some owner_nonce ->
                Some { kernel_name = kernel; owner_pid; owner_nonce; library_run_id }
            | _ -> None)
        | Some _ | None -> None)
  in
  let owned_stem stem =
    Option.bind (String.rsplit2 stem ~on:'_') ~f:(fun (rest, run) ->
        Option.bind (field ~prefix:"run" ~f:number run) ~f:(fun library_run_id ->
            owner ~library_run_id rest))
  in
  let kept fname : kept_library option =
    Option.bind (String.chop_suffix fname ~suffix:dynamic_library_extension) ~f:owned_stem
  in
  let temp fname : kept_library option =
    Option.bind (String.chop_prefix fname ~prefix:temp_library_marker) ~f:(fun name ->
        String.chop_suffix name ~suffix:dynamic_library_extension)
    |> Option.bind ~f:(fun stem ->
        Option.bind (String.rsplit2 stem ~on:'_') ~f:(fun (rest, random) ->
            Option.bind (hex random) ~f:(fun _ ->
                Option.bind (String.rsplit2 rest ~on:'_') ~f:(fun (rest, run) ->
                    Option.bind (number run) ~f:(fun library_run_id ->
                        Option.bind
                          (String.chop_suffix rest ~suffix:"_run_id")
                          ~f:(owner ~library_run_id))))))
  in
  (kept, temp)

(** [owned_artifact_file ~run_id ~extension base_name] is where this process keeps an artifact of
    [base_name]'s compile number [run_id]: beside [base_name]'s debug source, under the
    {!owned_artifact_name} no other compile, in this process or a concurrent one, uses. *)
let owned_artifact_file ~run_id ~extension base_name =
  build_file
    (owned_artifact_name ~pid:(os_process_id ()) ~nonce:(process_nonce ()) ~run_id ~extension
       base_name)

(** [kept_library_file ~run_id base_name] is where this process keeps the library of [base_name]'s
    compile number [run_id] under [output_dlls_in_build_directory] or
    [output_debug_files_in_build_directory]. *)
let kept_library_file ~run_id base_name =
  owned_artifact_file ~run_id ~extension:dynamic_library_extension base_name

(** The directory diagnostic and routine-debug logs are written to (created if missing):
    [log_files/<prefix>/], sharing the prefix resolution of {!build_files_dir}. *)
let log_files_dir () = ensure_artifacts_dir "log_files" (artifacts_subdir ())

(** [diagn_log_file fname] is the path of the log [fname] in {!log_files_dir}; like {!build_file},
    asking for it protects this process's logs from a concurrently starting sibling's cleanup. *)
let diagn_log_file fname =
  let dir = log_files_dir () in
  Artifacts_lock.hold "log_files";
  filename_concat dir @@ clean_filename fname

let () =
  (* Cleanup needs to happen before get_local_debug_runtime (or any other code is run). *)
  (* Use Unix.lstat to distinguish symlinks from real directories: Sys.is_directory
     follows symlinks, so a symlink inside build_files/log_files pointing outside
     the tree would cause recursion to delete unrelated files. We unlink the link
     itself instead of descending. *)
  let lstat_kind path = try Some (Unix.lstat path).st_kind with Unix.Unix_error _ -> None in
  (* An artifact whose name records its owner (a kept library, gh-ocannl-1247) goes by that owner's
     liveness: another process that is still running may be between choosing the path and linking or
     loading it, or a profiler may still need it. At startup, one under this process's own OS id was
     left by a predecessor that had the id. *)
  let library_owner_running { owner_pid; _ } =
    owner_pid <> os_process_id () && process_alive owner_pid
  in
  let owner_running fname =
    Option.exists (parse_kept_library_name fname) ~f:library_owner_running
  in
  (* What the cleanup keeps: the lock files, and the kept libraries of a running owner. *)
  let keep fname = String.is_suffix fname ~suffix:Artifacts_lock.suffix || owner_running fname in
  (* Answers whether [dirname] is gone: a directory keeping something stays. The cleanup's own root
     is emptied but kept ([~keep_root]): a concurrent process of the same executable may already
     hold a path inside it that it has yet to write -- a kept library its compiler is about to link
     (gh-ocannl-1247), a debug source -- and removing the directory would fail that write. *)
  let rec remove_dir_if_exists ?(keep_root = false) dirname =
    match lstat_kind dirname with
    | None -> true
    | Some Unix.S_DIR -> (
        try
          let emptied =
            Array.fold (Stdlib.Sys.readdir dirname) ~init:true ~f:(fun emptied fname ->
                let path = Stdlib.Filename.concat dirname fname in
                match lstat_kind path with
                | Some Unix.S_DIR -> remove_dir_if_exists path && emptied
                | Some Unix.S_REG when keep fname -> false
                | Some _ | None -> (
                    (* Regular file, symlink (to file or dir), socket, etc.: unlink the entry, do
                       not follow. *)
                    try
                      Stdlib.Sys.remove path;
                      emptied
                    with Stdlib.Sys_error _ -> emptied))
          in
          if emptied && not keep_root then Stdlib.Sys.rmdir dirname;
          emptied && not keep_root
        with exn ->
          Stdio.eprintf "Failed to delete directory %s: %s\n%!" dirname (Exn.to_string exn);
          false)
    | Some _ -> (
        (* Symlink (even to a dir), regular file, etc.: unlink the entry. *)
        try
          Stdlib.Sys.remove dirname;
          true
        with exn ->
          Stdio.eprintf "Failed to delete %s (expected a directory): %s\n%!" dirname
            (Exn.to_string exn);
          false)
  in
  (* Empties [root] under the exclusive side of [base]'s artifact lock, taken without waiting. A
     lock that is held means a running sibling, and then nothing is removed (gh-ocannl-1335): the
     cleanup cannot tell the sibling's ownerless artifacts -- debug sources, logs -- from a finished
     run's, and not even an owner-named artifact whose owner looks gone is safe, since the lock
     reaches across PID namespaces (containers sharing the directory) where liveness does not. The
     stale ones go at the next startup that finds no sibling. A lock that cannot be taken at all --
     no lock file can be created, or the file system has no locks -- leaves the cleanup as it was
     before: owner-named artifacts of running owners stay, everything else goes. *)
  (* gh-ocannl-1462: [force_clean_up_artifacts_on_startup] empties a held directory anyway -- for a
     long-running process that would otherwise hold every later cleanup back -- and names a holder
     on stderr. It overrides the lock's gate only: the lock file stays (removing it would let later
     processes lock different files), and so do the owner-named artifacts of running owners, as in
     an unlocked cleanup. *)
  let force_clean_up =
    get_global_flag ~default:false ~arg_name:"force_clean_up_artifacts_on_startup"
  in
  let remove_root ~base root =
    (* Tried whenever [base] is a directory, [root] or not: a root absent now may be created, and
       written into under its lock, by a sibling before the removal below reads it. *)
    let lock =
      match lstat_kind base with
      | None ->
          (* Nothing to remove; whatever appears here from now on is a sibling's. *)
          `Absent
      | Some Unix.S_DIR -> (
          match Artifacts_lock.open_ base with
          | exception Unix.Unix_error ((Unix.EACCES | Unix.EPERM), _, _) ->
              (* A lock file another account created, which this one cannot open for writing:
                 whoever shares the directory may hold it, so take it as held. *)
              `Held (-1)
          | exception Unix.Unix_error _ -> `Unlockable
          | fd -> (
              match Unix.lockf fd Unix.F_TLOCK 0 with
              | () -> `Exclusive fd
              | exception Unix.Unix_error ((Unix.EAGAIN | Unix.EACCES | Unix.EWOULDBLOCK), _, _) ->
                  let holder = if force_clean_up then lock_holder_pid fd else -1 in
                  Unix.close fd;
                  `Held holder
              | exception Unix.Unix_error _ ->
                  Unix.close fd;
                  `Unlockable))
      | Some _ -> `Unlockable
    in
    Exn.protect
      ~f:(fun () ->
        match lock with
        | `Held holder when force_clean_up ->
            Stdio.eprintf
              "OCANNL: force_clean_up_artifacts_on_startup: emptying %s although a running process \
               holds its lock %s (%s); that process's artifacts there are removed.\n\
               %!"
              root (Artifacts_lock.path base)
              (if holder > 0 then
                 Printf.sprintf "held by pid %d, one of possibly several holders" holder
               else "holder pid unknown");
            ignore (remove_dir_if_exists ~keep_root:true root : bool)
        | `Held _ | `Absent -> ()
        | `Exclusive _ | `Unlockable -> ignore (remove_dir_if_exists ~keep_root:true root : bool))
      ~finally:(fun () ->
        match lock with
        | `Exclusive fd -> ( try Unix.close fd with Unix.Unix_error _ -> ())
        | `Held _ | `Absent | `Unlockable -> ())
  in
  (* Cleanup is scoped to this process's own subdirectory (see [artifacts_subdir]), so a starting
     test cannot delete the in-flight artifacts of a concurrently running one. If the artifact root
     itself is a symlink, skip the scoped cleanup rather than follow it: appending the subdirectory
     would resolve through the link, deleting files outside the working tree. *)
  let remove_scoped_dir base =
    match artifacts_subdir () with
    | None -> remove_root ~base base
    | Some p -> (
        match lstat_kind base with
        | Some Unix.S_DIR -> remove_root ~base (filename_concat base p)
        | Some _ | None -> ())
  in
  let clean_up_log_files_on_startup =
    get_global_flag ~default:true ~arg_name:"clean_up_log_files_on_startup"
  in
  if clean_up_log_files_on_startup then remove_scoped_dir "log_files";
  let clean_up_build_files_on_startup =
    get_global_flag ~default:true ~arg_name:"clean_up_build_files_on_startup"
  in
  if clean_up_build_files_on_startup then remove_scoped_dir "build_files";
  (* Temporary libraries of processes that are gone (gh-ocannl-1268). A cc compile removes its
     temporary library once it is loaded, but Windows refuses to delete a loaded DLL, so the library
     of a routine alive at exit stays behind (POSIX has already removed it, so this finds none
     there). Not gated by a setting: these are files their owner meant to remove, not artifacts
     anyone asked for. A running owner's library stays, as in the cleanup above; at startup, one
     under this process's own OS id was left by a predecessor that had the id. So does a library
     written less than [temp_library_sweep_min_age] ago: its owner may be running where this process
     cannot see it, between linking the library and loading it. *)
  let temp_dir = Stdlib.Filename.get_temp_dir_name () in
  let past_load_window path =
    try Float.(Unix.gettimeofday () - (Unix.stat path).Unix.st_mtime > temp_library_sweep_min_age)
    with Unix.Unix_error _ -> false
  in
  match Stdlib.Sys.readdir temp_dir with
  | exception Stdlib.Sys_error _ -> ()
  | entries ->
      Array.iter entries ~f:(fun fname ->
          let path = Stdlib.Filename.concat temp_dir fname in
          match parse_temp_library_name fname with
          | Some owner when (not (library_owner_running owner)) && past_load_window path -> (
              try Stdlib.Sys.remove path with Stdlib.Sys_error _ -> ())
          | Some _ | None -> ())

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

   The test is "would some key read this argument", asked of {!cmdline_arg_is_config_key}. Its union
   of candidate prefixes is unchanged by the longest-key ownership in {!cmdline_arg_value}, so the
   warning and reader agree about which arguments are read. *)
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

external detach_c_stderr : unit -> bool = "ocannl_detach_c_stderr"
(** Move the C runtime's [stderr] onto its own descriptor, duplicated from the process's real
    stderr, so that foreign C libraries' diagnostics follow the real stderr rather than whatever
    later takes over fd 2 (gh-ocannl-1031). Reports whether it took effect.

    The chatter that forced it is ROCm's: Ubuntu 26.04 ships an assertions-enabled libhsa-runtime64
    7.1.0, whose live [debug_print] costs one [Signal 0x... time stamps may be invalid.] per
    [hipEventRecord] -- unavoidably, since ROCclr profiles every event marker regardless of
    [hipEventDisableTiming] and no runtime switch reaches the print. That belongs on stderr and the
    project already treats stderr as the chatter channel, but ppx_expect redirects fd 1 AND fd 2
    into the file it diffs, so the chatter lands in a [%%expect] golden. Nothing is filtered here:
    the lines still reach the real stderr and the run log, and OCaml's own [Stdlib.stderr] keeps
    writing to fd 2, so the library's own diagnostics are unchanged. *)

(* At module initialization, so the descriptor is duplicated from the real stderr before any test
   harness or capture wrapper has had a chance to redirect fd 2. *)
let c_stderr_detached =
  get_global_flag ~default:true ~arg_name:"detach_c_stderr" && detach_c_stderr ()

external c_stderr_replaceable : unit -> bool = "ocannl_c_stderr_replaceable"
(** Whether this C runtime's [stderr] can be replaced at all (glibc, Darwin): the predicate both
    {!detach_c_stderr} and {!collapse_c_stderr_noise} are compiled under, so a test can tell a host
    that cannot from a stub that failed. *)

external collapse_c_stderr_noise : unit -> bool = "ocannl_collapse_c_stderr_noise"
(** Replace the C runtime's [stderr] by a stream that writes through to the same descriptor but
    collapses the ROCm runtime's [Signal 0x... time stamps may be invalid.] lines into a count
    (gh-ocannl-1370): the first is written, with a note that later ones are counted, and the count
    is reported in one line at exit. Every other byte is written through at once and in order.
    Reports whether it took effect (glibc and Darwin only). Idempotent.

    {!c_stderr_detached} keeps the line out of a ppx_expect golden, but not off the run log, where
    an assertions-enabled ROCr prints one per recorded event -- 268,172 for one autotuning test --
    and Dune truncates a long action output from the middle, which is where the test's own
    diagnostics were. *)

(** Whether the C runtime's [stderr] collapses the ROCm runtime's per-event chatter, installing the
    filter on first force when [collapse_foreign_stderr_noise] (default true) asks for it. The HIP
    backend forces it when it opens a device, before creating its stream; nothing else does, so no
    other process's C stream is replaced. *)
let c_stderr_noise_collapsed =
  lazy
    (get_global_flag ~default:true ~arg_name:"collapse_foreign_stderr_noise"
    && collapse_c_stderr_noise ())

(** What {!bounded_exit_teardown} did with one resource. *)
type exit_teardown_outcome =
  | Torn_down
  | Disabled  (** [exit_stream_teardown_timeout] is [0] or negative. *)
  | Still_busy  (** Not idle within the bound: left to the driver, never waited on further. *)
  | Failed of exn  (** [is_idle] or [teardown] raised; the exception is swallowed. *)

(** The [exit_stream_teardown_timeout] in effect when the setting is absent or not a finite number.
*)
let default_exit_stream_teardown_timeout = 2.0

type exit_teardown_resource = { what : string; is_idle : unit -> bool; teardown : unit -> unit }
(** One resource for {!bounded_exit_teardown}: [what] names it in diagnostics. *)

(** Process-exit teardown of device resources (a GPU backend's streams), under ONE bound shared by
    all of them (gh-ocannl-1036). Meant to run from [Stdlib.at_exit]: polls every resource's
    [is_idle] in turn, calls a resource's [teardown] as soon as it is idle, and stops at
    [exit_stream_teardown_timeout] seconds after the call, leaving whatever is still busy to the
    driver -- so a process whose devices are hung is delayed at exit by at most the bound, however
    many there are, and is never held. [teardown] itself must therefore not wait for the device
    beyond what [is_idle] already established. Returns one outcome per resource, in order.

    Nothing escapes, diagnostics included (stderr may be closed or broken by exit time): an
    exception at exit would replace the program's own exit status or uncaught-exception report. A
    process that dies by a fatal signal never runs [at_exit] at all; one that ends on an uncaught
    exception does (the runtime runs [at_exit] before reporting it), which is why the bound, not an
    exit-status test, is what keeps a failing process from being held. *)
let bounded_exit_teardown (resources : exit_teardown_resource list) : exit_teardown_outcome list =
  let note fmt =
    Printf.ksprintf (fun line -> try Stdio.eprintf "OCANNL: %s\n%!" line with _ -> ()) fmt
  in
  let default = default_exit_stream_teardown_timeout in
  let setting =
    try
      Some
        (get_global_arg ~default:(Float.to_string default) ~arg_name:"exit_stream_teardown_timeout")
    with _ -> None
  in
  (* A value that is not a finite number would break the bound ([inf] polls forever) or raise
     outside the handlers below; it falls back to the default instead. *)
  let timeout =
    match Option.bind setting ~f:(fun s -> Float.of_string_opt (String.strip s)) with
    | Some t when Float.is_finite t -> t
    | _ ->
        note "exit_stream_teardown_timeout=%s is not a finite number of seconds; using %g"
          (Option.value setting ~default:"<unreadable>")
          default;
        default
  in
  if Float.(timeout <= 0.) then List.map resources ~f:(fun _ -> Disabled)
  else
    let started = Mtime_clock.counter () in
    let elapsed () = Mtime.Span.to_float_ns (Mtime_clock.count started) /. 1e9 in
    let resources = Array.of_list resources in
    let outcomes = Array.map resources ~f:(fun _ -> None) in
    let attempt i { what; is_idle; teardown } =
      match is_idle () with
      | false -> ()
      | true -> (
          match teardown () with
          | () -> outcomes.(i) <- Some Torn_down
          | exception e ->
              note "tearing down %s at process exit failed: %s" what (Exn.to_string e);
              outcomes.(i) <- Some (Failed e))
      | exception e ->
          note "polling %s at process exit failed: %s" what (Exn.to_string e);
          outcomes.(i) <- Some (Failed e)
    in
    let rec poll () =
      Array.iteri resources ~f:(fun i r -> if Option.is_none outcomes.(i) then attempt i r);
      if Array.exists outcomes ~f:Option.is_none then
        if Float.(elapsed () < timeout) then (
          (* A signal can interrupt the sleep (EINTR); the deadline, not the sleep, ends the
             wait. *)
          (try Unix.sleepf 0.001 with _ -> ());
          poll ())
        else
          Array.iteri resources ~f:(fun i { what; _ } ->
              if Option.is_none outcomes.(i) then (
                note
                  "%s still busy after %.3fs of process exit; left for the driver to reclaim \
                   (exit_stream_teardown_timeout=%g)"
                  what (elapsed ()) timeout;
                outcomes.(i) <- Some Still_busy))
    in
    poll ();
    Array.to_list outcomes |> List.map ~f:(Option.value ~default:Still_busy)

external device_init_deadline_supported : unit -> bool = "ocannl_device_init_deadline_supported"
(** Whether this build can arm the device-init deadline at all (not on Windows, which has no
    SIGALRM): the predicate {!with_device_init_deadline}'s stubs are compiled under. *)

(** The fixed start of the diagnostic the deadline writes. [tools/test-run.sh]'s digest recognizes a
    wedged device by it, and [test/operations/device_init_deadline] checks that the two agree. *)
let device_init_deadline_tag = "OCANNL device_init_deadline:"

(** Test-only: when set, {!with_device_init_deadline} runs it inside the guarded region, before the
    guarded initialization -- so a test can simulate a stuck initialization (a sleep) and, by
    raising, skip the real one. *)
let device_init_test_stall : (unit -> unit) option ref = ref None

(** The pieces of {!with_device_init_deadline}: its C stubs ([utils_stubs.c]) and its setting. *)
module Device_init_deadline_internals = struct
  external arm : float -> string -> int = "ocannl_device_init_deadline_arm"
  external disarm : unit -> unit = "ocannl_device_init_deadline_disarm"

  let default = 120.0

  (* Read once, at module initialization, while the process has one domain: the configuration reader
     records every access in a table no two domains may touch at once, and a guarded call can come
     from any domain, beside any other domain's configuration reads. *)
  let seconds =
    let setting =
      get_global_arg ~default:(Float.to_string default) ~arg_name:"device_init_deadline"
    in
    match Float.of_string_opt (String.strip setting) with
    | Some t when Float.is_finite t -> t
    | _ ->
        Stdio.eprintf
          "OCANNL: device_init_deadline=%s is not a finite number of seconds; using %g\n%!" setting
          default;
        default
end

(** [with_device_init_deadline ~backend_name f] runs [f] -- a GPU backend's device discovery, device
    query or device opening -- under a deadline of [device_init_deadline] seconds (gh-ocannl-1281;
    [0] or negative disables; 120 when the setting is absent or not a finite number): if [f] has not
    returned by then, the process writes a diagnostic opening with {!device_init_deadline_tag},
    naming the backend and the key, and dies by SIGALRM (status 142), since a wedged driver never
    returns and nothing in-process can unwind out of it. The bound is a watchdog thread of the C
    stubs ([utils_stubs.c]), not a signal handler or the real-time timer, so the program's own
    timers, handlers and signal masks are left alone. The deadline is disarmed when [f] returns or
    raises, so it is never armed across user code, and it is not armed at all when SIGALRM does not
    have its default disposition (killing by it would have to override the program's own) or on
    Windows.

    Guarded regions are serialized process-wide, in the stubs: the watchdog holds one deadline at a
    time, and a region entered while another thread's is armed waits for it, bounded by its
    deadline. A region entered from inside the same thread's armed one runs under the outer
    deadline.

    A healthy device initializes in well under a second (the harness's device probe, a whole fresh
    process, answers in 0.2-0.6 s), so the default is generous for cold starts on loaded, WSL2 or
    freshly installed boxes. *)
let with_device_init_deadline ~backend_name f =
  let module Stub = Device_init_deadline_internals in
  let seconds = Stub.seconds in
  let armed =
    Float.(seconds > 0.)
    &&
    match
      Stub.arm seconds
        (Printf.sprintf
           "\n\
            %s %s device initialization or query did not finish within %gs; the device or driver \
            is likely wedged. Dying by SIGALRM. The bound is the config key device_init_deadline \
            (seconds; 0 disables; gh-ocannl-1281).\n"
           device_init_deadline_tag backend_name seconds)
    with
    | 1 -> true
    | 0 -> false
    | _ ->
        (* Fail closed: a configured bound that cannot be armed is not "no bound". *)
        failwith
          (Printf.sprintf
             "OCANNL: cannot arm the %gs device_init_deadline before a %s driver call: its \
              watchdog thread could not be started (device_init_deadline=0 proceeds without the \
              bound; gh-ocannl-1281)"
             seconds backend_name)
  in
  Exn.protect
    ~f:(fun () ->
      Option.iter !device_init_test_stall ~f:(fun stall -> stall ());
      f ())
    ~finally:(fun () -> if armed then Stub.disarm ())

let enable_runtime_debug () =
  settings.output_debug_files_in_build_directory <- true;
  set_log_level @@ max 2 settings.log_level

(** A fresh non-negative integer per call, wrapping to 0 on overflow. Backends use it to tell one
    run of a kernel from the next: cc names each dynamically loaded library by it (there can be only
    one library of a given name in a process), and the GPU backends prefix a launch's routine logs
    with it. Process-wide rather than per backend, so that ids never collide across the backends a
    process links. *)
let get_global_run_id =
  (* Atomic: cc library paths rely on these ids being unique within the process (gh-ocannl-1197). *)
  let next_id = Atomic.make 0 in
  fun () -> Atomic.fetch_and_add next_id 1 + 1

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
  | Optional of { callback_if_missing : unit -> unit [@sexp.opaque] }
[@@deriving sexp_of]

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

(* Re-exported for the same reason (gh-ocannl-1558): the autotuner's co-tenant signal. *)
module Host_load = Host_load

type build_file_channel = { f_path : string; oc : Stdlib.out_channel; finalize : unit -> unit }

let open_build_file ~base_name ~extension : build_file_channel =
  let f_path =
    if settings.output_debug_files_in_build_directory then build_file @@ base_name ^ extension
    else
      (* The kernel name cut as an owned artifact's is: a routine's name has no length limit, and a
         compiler names its own temporaries after the source, so an uncut name fails the compile
         with "File name too long" (gh-ocannl-1335). *)
      Stdlib.Filename.temp_file (library_kernel_name base_name ^ "_") extension
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

(* How many [capture_stdout_logs] calls are running. A logged GPU launch inside one leaves its
   device-side printf to that window; outside all of them, [routine_log_window] opens one of its own
   (gh-ocannl-1636). *)
let stdout_capture_depth = Atomic.make 0

let capture_stdout_logs_window arg =
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

    (* The reader's own duplicate of the original stdout, for immediate passthrough. Made here,
       before the reader is spawned: the reader made it itself, so when [arg] returned before the
       reader had run, [original_stdout_fd] was already closed, the [dup] failed with [EBADF] and
       the window lost every line it caught (gh-ocannl-1636). *)
    let orig_out_fd = Unix.dup ~cloexec:true original_stdout_fd in
    let reader_domain_logic () =
      let in_channel = Unix.in_channel_of_descr pipe_read_fd in
      let orig_out = Unix.out_channel_of_descr orig_out_fd in
      try
        while true do
          let _is_endlined, line = input_line in_channel in
          match String.chop_prefix ~prefix:!captured_log_prefix line with
          | Some logline -> collected_logs_ref := logline :: !collected_logs_ref
          | None ->
              (* Forward non-log lines to original stdout immediately *)
              Stdlib.output_string orig_out (line ^ "\n");
              Stdlib.flush orig_out
        done
      with
      | End_of_file ->
          (* Normal termination of the reader: the loop above only ends here, so both channels are
             closed here, or each window leaks two descriptors -- one per logged launch under
             [routine_log_window] (gh-ocannl-1636). *)
          Stdlib.close_out_noerr orig_out;
          Stdlib.close_in_noerr in_channel (* This closes pipe_read_fd *)
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

let capture_stdout_logs arg =
  Atomic.incr stdout_capture_depth;
  Exn.protect
    ~finally:(fun () -> Atomic.decr stdout_capture_depth)
    ~f:(fun () -> capture_stdout_logs_window arg)

(** [routine_log_window ~drain launch] runs a logged CUDA or HIP routine's [launch] -- which
    registers the routine's log processor and launches its kernel -- so that, while
    [debug_log_to_stream_files] is on, the kernel's device-side [printf] never reaches stdout
    (gh-ocannl-1636). Those runtimes write a kernel's [printf] to process stdout at whichever
    synchronization point, later launch or blocking copy comes next, and only a
    [capture_stdout_logs] window hands the lines to the processor that writes the stream file. So
    outside every window it opens one around [launch] followed by [drain], the synchronization that
    empties the device's printf buffer, and every line of this launch is in the stream file when it
    returns. Inside a caller's window it only runs [launch]: that window collects the lines, as
    before. With the key off the lines are relogged through the debug trace, which needs the
    caller's window, as before; [never_capture_stdout] still turns every capture off. *)
let routine_log_window ~drain launch =
  if
    Atomic.get stdout_capture_depth > 0
    || not (get_global_flag ~default:false ~arg_name:"debug_log_to_stream_files")
  then launch ()
  else
    capture_stdout_logs (fun () ->
        launch ();
        drain ())

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

let macos_platform_uuid registry =
  String.split_lines registry
  |> List.find_map ~f:(fun line ->
      match String.lsplit2 line ~on:'=' with
      | Some (name, value) when String.is_suffix (String.strip name) ~suffix:"\"IOPlatformUUID\"" ->
          Option.bind
            (String.chop_prefix (String.strip value) ~prefix:"\"")
            ~f:(fun value ->
              Option.bind (String.chop_suffix value ~suffix:"\"") ~f:(fun uuid ->
                  let uuid = String.lowercase uuid in
                  let parts = String.split uuid ~on:'-' in
                  if
                    Poly.equal (List.map parts ~f:String.length) [ 8; 4; 4; 4; 12 ]
                    && List.for_all parts
                         ~f:
                           (String.for_all ~f:(fun c ->
                                Char.is_digit c || Char.(c >= 'a' && c <= 'f')))
                    && String.exists uuid ~f:(fun c -> not (Char.equal c '0' || Char.equal c '-'))
                  then Some (Stdlib.Digest.to_hex (Stdlib.Digest.string uuid))
                  else None))
      | _ -> None)

(* Registry IDs are machine-scoped. macOS hardware UUID separates hosts with identical device names;
   the OS build adds observed metadata for the bundled Metal/dispatch runtime. *)
let macos_timing_environment =
  lazy
    (let query path args =
       try
         let ch = Unix.open_process_args_in path args in
         let status = ref None in
         let output =
           Exn.protect
             ~f:(fun () -> Stdio.In_channel.input_all ch)
             ~finally:(fun () -> status := Some (Unix.close_process_in ch))
         in
         match !status with
         | Some (Unix.WEXITED 0) when not (String.is_empty (String.strip output)) ->
             Some (String.strip output)
         | _ -> None
       with Unix.Unix_error _ | Stdlib.Sys_error _ -> None
     in
     if not (Stdlib.Sys.file_exists "/usr/sbin/ioreg") then None
     else
       Option.bind
         (query "/usr/sbin/ioreg" [| "ioreg"; "-rd1"; "-c"; "IOPlatformExpertDevice" |])
         ~f:(fun registry ->
           Option.map (macos_platform_uuid registry) ~f:(fun host ->
               (host, query "/usr/bin/sw_vers" [| "sw_vers"; "-buildVersion" |]))))

(** gh-ocannl-1038: the headers a C or C++ text includes by [#include <...>], in order -- the files
    {!capability_dir} fingerprints, read off the prelude a backend compiles with. *)
let included_headers text =
  String.split_lines text
  |> List.filter_map ~f:(fun line ->
      match String.chop_prefix (String.strip line) ~prefix:"#include" with
      | None -> None
      | Some rest -> (
          match String.lsplit2 (String.strip rest) ~on:'>' with
          | Some (spec, _) -> String.chop_prefix spec ~prefix:"<"
          | None -> None))

(** gh-ocannl-1038: a found directory as a host-capability record states it. Three things can move
    while the path a probe answered stays the same, and each must move the record:

    - where it resolves, appended after [=>] when that differs: a toolkit or SDK reached through a
      version-neutral symlink ([/usr/local/cuda], [/opt/rocm]) is upgraded by repointing the link;
    - whether the headers the backend compiles against are there at all, since some probes answer a
      configured path without looking inside it ([CUDA_PATH]): [(missing:<file>,...)] names the
      absent ones;
    - which headers they are, when all are present: a [#] and a digest of each one's size and
      modification time, since a distro package upgrades its headers in place under [/usr].

    [none] when the probe found nothing. *)
let capability_dir ~requires = function
  | None -> "none"
  | Some dir -> (
      let located =
        match Unix.realpath dir with
        | resolved when not (String.equal resolved dir) -> dir ^ "=>" ^ resolved
        | _ -> dir
        | exception (Unix.Unix_error _ | Stdlib.Sys_error _) -> dir
      in
      let stat file =
        try
          let st = Unix.stat (Stdlib.Filename.concat dir file) in
          Some (Printf.sprintf "%s:%d:%.0f" file st.Unix.st_size st.Unix.st_mtime)
        with Unix.Unix_error _ | Stdlib.Sys_error _ -> None
      in
      match List.filter requires ~f:(fun file -> Option.is_none (stat file)) with
      | [] ->
          let stamps = List.filter_map requires ~f:stat in
          located ^ "#"
          ^ String.prefix
              (Stdlib.Digest.to_hex (Stdlib.Digest.string (String.concat ~sep:"\n" stamps)))
              8
      | missing -> located ^ "(missing:" ^ String.concat ~sep:"," missing ^ ")")
