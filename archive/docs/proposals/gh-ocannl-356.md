# Proposal: Rename param/params to kparam/kparams

**Task**: [gh-ocannl-356](https://github.com/ahrefs/ocannl/issues/356)
**Status**: ready

## Goal

Eliminate the naming ambiguity between ML-level tensor parameters (`Tensor.params` -- trainable weights/biases) and kernel/routine parameters (`param_source`, `Param_ptr`, `procedure.params` -- buffer pointers and static indices passed to compiled compute kernels). Rename all kernel-parameter identifiers from `param`/`params` to `kparam`/`kparams` in the arrayjit backend layer.

## Acceptance Criteria

- [ ] `param_source` type renamed to `kparam_source` in `backend_intf.ml`
- [ ] `Param_ptr` variant renamed to `Kparam_ptr`
- [ ] `params` field renamed to `kparams` in all `procedure` record types (`cc_backend.ml`, `cuda_backend.ml`, `metal_backend.ml`)
- [ ] `params_and_names` renamed to `kparams_and_names` in `cuda_backend.ml` `code_batch`
- [ ] All local variable bindings, function parameters, and pattern matches referencing kernel params updated consistently across `c_syntax.ml`, `cc_backend.ml`, `cuda_backend.ml`, `metal_backend.ml`
- [ ] `Tensor.params`, `Tensor.init_params`, and all tensor-layer naming is **unchanged**
- [ ] `kernel_log_param`, `log_param_c_expr_doc`, `idx_params` (Indexing symbols), and other unrelated uses of "param" are **unchanged** -- only kernel buffer/source parameter identifiers are renamed
- [ ] Comments referencing renamed identifiers are updated
- [ ] Project compiles cleanly with no type errors
- [ ] Existing test suite passes with no regressions

## Context

The codebase uses "params" for two distinct concepts:

1. **Tensor parameters** (`Tensor.params`): trainable descendants whose `diff` is not `None` -- the ML sense of "parameters."
2. **Kernel parameters** (`param_source`, `Param_ptr`, `procedure.params`): arguments passed to compiled compute kernels -- buffer pointers, merge buffers, log file names, static indices.

Reading code like `List.iter params ~f:(fun (name, Param_ptr tn) -> ...)` is ambiguous. The `k` prefix ("kernel parameter") disambiguates concisely.

### Scope

The rename is confined to the arrayjit backend layer. Affected files:

| File | Key changes |
|------|-------------|
| `arrayjit/lib/backend_intf.ml` | `param_source` -> `kparam_source`, `Param_ptr` -> `Kparam_ptr` |
| `arrayjit/lib/c_syntax.ml` | `compile_proc` return type, local `params` bindings, `Param_ptr` patterns (~15 locations) |
| `arrayjit/lib/cc_backend.ml` | `procedure.params` -> `kparams`, link function `params` arg, `Param_ptr` pattern |
| `arrayjit/lib/cuda_backend.ml` | `procedure.params` -> `kparams`, `code_batch.params_and_names` -> `kparams_and_names`, `link_proc` signature, `Param_ptr` pattern |
| `arrayjit/lib/metal_backend.ml` | `procedure.params` -> `kparams`, `code_batch.funcs` tuples, `link_proc` signature, `Param_ptr` patterns |

### What is NOT renamed

- `kernel_log_param` / `log_param_c_expr_doc` / `log_file_param` / `merge_param` -- these are about specific kernel parameter *kinds*, not the `param_source` type. They could optionally be prefixed but the task issue focuses on the `param`/`params` -> `kparam`/`kparams` rename for disambiguation from `Tensor.params`.
- `idx_params` (bound symbols from `Indexing`) -- these are indexing symbols, not kernel parameter sources.
- `CAMLparam*` in `builtins.c` -- OCaml C FFI macros, unrelated.
- Any `Tensor`-layer code.

### Approach

This is a mechanical rename. OCaml's static type system will catch any missed references at compile time. The recommended workflow is:

1. Rename the type and variant in `backend_intf.ml`
2. Fix all compilation errors file by file
3. Update comments referencing old names
4. Run the test suite

A single commit preserves git blame clarity.
