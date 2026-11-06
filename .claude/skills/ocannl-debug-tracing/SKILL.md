---
name: ocannl-debug-tracing
description: How to get debug output out of OCANNL: kernel/routine runtime logs, generated code artifacts, and ppx_minidebug tracing of library internals such as the shape/row solver. Use when debugging a wrong-value or wrong-shape result and you need to see generated code, per-kernel logs, or the solver's decisions.
---

# Debugging and Logging

- Set `debug_log_from_routines=true` in config for kernel/routine-level debugging; it needs `log_level > 1` (`Utils.debug_log_from_routines ()` tests both) and is silent below that. Add `debug_log_to_stream_files=true` to read the lines: with the key off they go to `Utils.log_trace_tree`, which preprocesses to a no-op unless the build sets `OCANNL_LOG_LEVEL_UTILS`, so cc and Metal drop them. The first batch dropped prints one stderr notice per process (`Utils.routine_logs_dropped_notice`, gh-ocannl-1999; pinned by `test/operations/routine_log_dropped_notice.ml`); the kernel is compiled for logging either way (serial, unvectorized), which tests use as a codegen switch
- Use `log_level=2` for verbose ppx_minidebug output
- CUDA and HIP routine logs are device-side `printf`, written to process stdout at a sync point. With `debug_log_to_stream_files=true` the backend keeps them off stdout itself: outside any `Utils.capture_stdout_logs` window it opens one around each logged launch and synchronizes the device inside it (gh-ocannl-1636; pinned by `test/operations/routine_log_stream_files.ml`). With the key off they stay on stdout (and in any `.expected` golden) unless the caller opens a capture window: wrap the runs in `Utils.capture_stdout_logs` and `Context.sync` before leaving it. That window hands them to the same `Utils.log_trace_tree` as on cc and Metal, so a default build drops them there too, with the same one notice; only a build with `OCANNL_LOG_LEVEL_UTILS` relogs them into the debug trace. `never_capture_stdout=true` turns both captures off
- Debug files generated in `log_files/<exe-name>/` (this process's subdirectory is cleaned on startup by default)
- Runtime logs from kernel execution are written to `<backend>-<device>-<stream>.log` (e.g., `cuda-0-0.log`) inside that subdirectory
- Generated code files in `build_files/<exe-name>/` show high-level `.cd`, intermediate `.ll`, and backend-specific `.c`/`.cu` files

See the `Important Debug Settings` list in `AGENTS.md` for the config keys that enable these artifacts.

## Tracing library internals with ppx_minidebug (e.g. the shape/row solver)

- High-level `%debug5_sexp`/`%track5_sexp` etc. log statements are stripped at COMPILE time; each module has its own gate env var: `[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_ROW"]` at the top of `row.ml`, similarly `..._SHAPE`, `..._TNODE`, `..._ASSIGNMENTS`, `..._CC_BACKEND`, etc. (grep for `global_debug_log_level_from_env_var`)
- Every gate is declared as a `preprocessor_deps` `(env_var ...)` of the library whose modules read it (gh-ocannl-628), so setting one re-preprocesses those modules by itself: `OCANNL_LOG_LEVEL_ROW=9 OCANNL_LOG_LEVEL_SHAPE=9 dune build ...`. Adding a gate to a module means adding it to that library's `preprocessor_deps` too, which `test/operations/env_var_deps` checks
- At runtime, pass `--ocannl_log_level=9` and run from a directory with an `ocannl_config` on its ancestor path (e.g. `test/config/`); logs land in `log_files/`
- Backend choice: the default `debug_backend=db` writes a database supporting structured queries (e.g. print a subtree containing two given terms); `--ocannl_debug_backend=flushing` writes greppable flat text to `log_files/<exe>/debug.log` and survives crashes mid-trace. Under `test/config/ocannl_config` (`log_main_domain_to_stdout=true`) this text goes to stdout instead and no file appears, so a test's trace lands in its `_build/default/<dir>/<name>.exe.output`, and its alias reports a golden diff that is the trace itself: read it, never promote it
- Useful flat-text queries: `stage = StageN` lines segment `finish_inference`'s solver passes; track a variable by grepping `Row_var N` / `(id N)` for its transition to `Solved_row`/`Solved_dim`, then inspect the surrounding `unify_row`/`solve_dim_ineq` scope
- A row trace is too long to read whole, even from a small test: start from its digest, `tools/row-trace-compact.py TRACE`, which prints one line per distinct `solve_row_ineq` (trace line, stage, `res >= opnd` as `beg,..v..,dims`) plus template mints, the trace line leading back to the full scope. To capture one, `OCANNL_LOG_LEVEL_ROW=9 OCANNL_DEBUG_BACKEND=flushing OCANNL_LOG_LEVEL=9 tools/test-run.sh run build @test/operations/runtest-<name>`, then `tools/row-trace-compact.py _build/default/test/operations/<name>.exe.output` (a `debug.log` digests the same)
