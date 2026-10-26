---
name: ocannl-debug-tracing
description: How to get debug output out of OCANNL: kernel/routine runtime logs, generated code artifacts, and ppx_minidebug tracing of library internals such as the shape/row solver. Use when debugging a wrong-value or wrong-shape result and you need to see generated code, per-kernel logs, or the solver's decisions.
---

# Debugging and Logging

- Set `debug_log_from_routines=true` in config for kernel/routine-level debugging
- Use `log_level=2` for verbose ppx_minidebug output
- CUDA and HIP routine logs are device-side `printf`, written to process stdout at a sync point. With `debug_log_to_stream_files=true` the backend keeps them off stdout itself: outside any `Utils.capture_stdout_logs` window it opens one around each logged launch and synchronizes the device inside it (gh-ocannl-1636; pinned by `test/operations/routine_log_stream_files.ml`). With the key off they are relogged into the debug trace only through a capture window the caller opens: wrap the runs in `Utils.capture_stdout_logs` and `Context.sync` before leaving it, or they reach stdout (and any `.expected` golden). `never_capture_stdout=true` turns both captures off
- Debug files generated in `log_files/<exe-name>/` (this process's subdirectory is cleaned on startup by default)
- Runtime logs from kernel execution are written to `<backend>-<device>-<stream>.log` (e.g., `cuda-0-0.log`) inside that subdirectory
- Generated code files in `build_files/<exe-name>/` show high-level `.cd`, intermediate `.ll`, and backend-specific `.c`/`.cu` files

See the `Important Debug Settings` list in `AGENTS.md` for the config keys that enable these artifacts.

## Tracing library internals with ppx_minidebug (e.g. the shape/row solver)

- High-level `%debug5_sexp`/`%track5_sexp` etc. log statements are stripped at COMPILE time; each module has its own gate env var: `[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_ROW"]` at the top of `row.ml`, similarly `..._SHAPE`, `..._TNODE`, `..._ASSIGNMENTS`, `..._CC_BACKEND`, etc. (grep for `global_debug_log_level_from_env_var`)
- Every gate is declared as a `preprocessor_deps` `(env_var ...)` of the library whose modules read it (gh-ocannl-628), so setting one re-preprocesses those modules by itself: `OCANNL_LOG_LEVEL_ROW=9 OCANNL_LOG_LEVEL_SHAPE=9 dune build ...`. Adding a gate to a module means adding it to that library's `preprocessor_deps` too, which `test/operations/env_var_deps` checks
- At runtime, pass `--ocannl_log_level=9` and run from a directory with an `ocannl_config` on its ancestor path (e.g. `test/config/`); logs land in `log_files/`
- Backend choice: the default `debug_backend=db` writes a database supporting structured queries (e.g. print a subtree containing two given terms); `--ocannl_debug_backend=flushing` writes greppable flat text to `log_files/debug.log` and survives crashes mid-trace
- Useful flat-text queries: `stage = StageN` lines segment `finish_inference`'s solver passes; track a variable by grepping `Row_var N` / `(id N)` for its transition to `Solved_row`/`Solved_dim`, then inspect the surrounding `unify_row`/`solve_dim_ineq` scope
