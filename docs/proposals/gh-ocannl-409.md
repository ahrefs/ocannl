# Proposal: Relax `ocannl_` prefix for commandline arguments and validate config keys

**Issue**: https://github.com/ahrefs/ocannl/issues/409
**Task**: gh-ocannl-409

## Goal

Improve OCANNL configuration usability by (1) accepting commandline arguments without the `ocannl_` prefix, (2) validating config file keys and commandline args against a known set, and (3) fixing discrepancies between `ocannl_config.example` and actual code usage.

## Acceptance Criteria

- Commandline arguments work without the `ocannl_` prefix (e.g., `--backend=multicore_cc`)
- Commandline arguments still work WITH the `ocannl_` prefix for backward compatibility
- Unknown config file keys produce a warning at load time
- Commandline arguments matching `--ocannl_*` that don't correspond to any known key produce a warning
- `ocannl_config.example` is a superset of all `get_global_arg`/`get_global_flag` call sites -- verified by a build-time or test-time check
- All discrepancies in `ocannl_config.example` and `ocannl_config.for_debug` are fixed
- No regression in existing tests
- Environment variables retain the `OCANNL_` prefix (to avoid namespace collisions)

## Context

### Current state

Configuration resolution in `arrayjit/lib/utils.ml`:

1. **Commandline** (`read_cmdline_or_env_var`, lines 53-79): generates 12 variants per key, all requiring the `ocannl_` prefix. Scans `Sys.argv` for prefix matches.
2. **Environment variables**: checks `ocannl_<name>` and `OCANNL_<NAME>` variants.
3. **Config file** (`config_file_args`, lines 97-158): walks directory tree for `ocannl_config`, already strips the `ocannl` prefix from keys.
4. **Defaults**: hard-coded at each `get_global_arg`/`get_global_flag` call site.

The config file parser already accepts keys without the prefix. Only the commandline parser requires it.

### Known discrepancies (as of current HEAD)

| File | Issue |
|------|-------|
| `ocannl_config.example` line 69 | `bacend=multicore_cc` -- typo, should be `backend` |
| `ocannl_config.example` line 136 | `randomness_lib=stdlib` -- no corresponding `get_global_arg` call in code |
| `ocannl_config.example` | Missing: `default_prng_variant`, `cd_ident_style`, `never_capture_stdout` (partially -- it's at line 77 but the key used in code is `never_capture_stdout`) |
| `ocannl_config.for_debug` line 11 | `output_debug_files_in_run_directory=true` -- stale name, should be `output_debug_files_in_build_directory` |
| `ocannl_config.for_debug` line 14 | `randomness_lib=for_tests` -- unused key |

### All known config keys from code

Extracted from all `get_global_arg`/`get_global_flag` call sites (54 unique keys):

**Settings (utils.ml)**: `log_level`, `debug_log_from_routines`, `output_debug_files_in_build_directory`, `fixed_state_for_init`, `print_decimals_precision`, `check_half_prec_constants_cutoff`, `automatic_host_transfers`, `default_prng_variant`, `big_models`

**Cleanup/bootstrap (utils.ml)**: `suppress_welcome_message`, `no_config_file`, `clean_up_log_files_on_startup`, `clean_up_build_files_on_startup`, `never_capture_stdout`

**ppx_minidebug (utils.ml)**: `snapshot_every_sec`, `time_tagged`, `elapsed_times`, `location_format`, `debug_backend`, `hyperlink_prefix`, `logs_print_scope_ids`, `logs_verbose_scope_ids`, `log_main_domain_to_stdout`, `log_file_stem`, `toc_entry_minimal_depth`, `toc_entry_minimal_size`, `toc_entry_minimal_span`, `debug_highlights`, `debug_highlight_pcre`, `prev_run_prefix`, `diff_ignore_pattern_pcre`, `diff_max_distance_factor`, `debug_scope_id_pairs`, `debug_log_truncate_children`, `debug_log_prune_upto`, `debug_log_to_stream_files`

**Backends**: `backend`, `prefer_backend_uniformity`, `cc_backend_optimization_level`, `cc_backend_compiler_command`, `cc_backend_arch_flags`, `cc_backend_fast_math`, `cc_backend_post_compile_timeout`, `cc_backend_verify_codesign`, `output_dlls_in_build_directory`, `cuda_printf_fifo_size`

**Low-level/optimization**: `virtualize_max_visits`, `virtualize_max_tracing_dim`, `enable_device_only`, `inline_scalar_constexprs`, `inline_simple_computations`, `inline_complex_computations`, `output_prec_in_ll_files`, `stack_threshold_in_bytes`

**Other**: `ll_ident_style`, `cd_ident_style`, `default_prec`, `limit_constant_fill_size`, `max_shape_error_origins`

### Approach

1. **Add prefix-free commandline variants** in `read_cmdline_or_env_var`: generate additional variants without the `ocannl_` prefix (e.g., `--backend=`, `-backend=`, `backend=`). Keep all existing prefixed variants for backward compatibility.

2. **Build a known-keys registry** as a static `Set` in `utils.ml` listing all valid config key names. This is explicit and gives immediate validation at runtime.

3. **Validate config file keys** at load time: after parsing `ocannl_config`, warn on any key not in the known set.

4. **Validate commandline args** at initialization: scan `Sys.argv` for entries matching `--ocannl_*` or `--ocannl-*` patterns that don't match any known key, and warn.

5. **Fix all discrepancies** in `ocannl_config.example` and `ocannl_config.for_debug` (see table above). Add missing keys (`default_prng_variant`, `cd_ident_style`) to the example file.

6. **Add a consistency test** that extracts `arg_name` strings from `get_global_arg`/`get_global_flag` call sites and verifies they all appear in `ocannl_config.example`, and vice versa (bidirectional check).

7. **Rename consideration**: The issue suggests renaming `ocannl_config.example`. Since the actual config file name `ocannl_config` is searched for by that exact name in the directory walk, only the example/reference file could be renamed. A reasonable choice is `ocannl_config.reference` to clarify it is the canonical documentation of all keys. This is a minor point and can be decided during implementation.
