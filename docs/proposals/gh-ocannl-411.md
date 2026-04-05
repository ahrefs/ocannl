# Proposal: HIP Backend (AMD Hardware)

**Task**: gh-ocannl-411
**Issue**: https://github.com/ahrefs/ocannl/issues/411

## Goal

Add AMD GPU support to OCANNL by creating (1) an `ocaml-hipjit` standalone bindings package for HIP driver API and hipRTC, and (2) a `hip_backend` in OCANNL that mirrors the existing CUDA backend. HIP is source-compatible with CUDA, so both the bindings and backend are largely mechanical translations of their CUDA counterparts.

## Acceptance Criteria

- [ ] `ocaml-hipjit` package provides OCaml bindings to HIP driver API (Device, Context, Deviceptr, Stream, Event, Module) and hipRTC (runtime compilation), following `ocaml-cudajit` structure
- [ ] `hip_backend.ml` implements the full `Lowered_backend` module type, mirroring `cuda_backend.ml` with HIP API calls replacing CUDA calls
- [ ] HIP backend is optional: builds succeed without ROCm/HIP installed (dune `(optional)` + select directive for `hip_backend_impl.ml`)
- [ ] Backend is registered as `"hip"` in `backends.ml`'s `fresh_backend` function
- [ ] HIP-specific precision types are handled: `_Float16` for half, `__hip_bfloat16` for bfloat16
- [ ] hipRTC runtime compilation produces device code that loads and executes correctly
- [ ] Existing test suite passes with `OCANNL_BACKEND=hip` (on AMD hardware or via HIP's CUDA platform fallback)
- [ ] No regressions in existing backends (CUDA, Metal, CC)

## Context

### Architecture

OCANNL's backend system is modular. Each GPU backend is an optional dune library that implements `Lowered_backend` (defined in `backend_intf.ml`), wrapped by `Raise_backend` in `backends.ml`. The selection mechanism uses dune's `(select ...)` directive: when the optional library is available, a real implementation is used; when absent, a stub (`lowered_backend_missing.ml`) raises an error.

Current GPU backends:
- **CUDA** (`cuda_backend` library): uses `ocaml-cudajit` bindings, compiles CUDA C via NVRTC to PTX, loads via `Cu.Module.load_data_ex`
- **Metal** (`metal_backend` library): uses `ocaml-metal` bindings, compiles Metal Shading Language via system compiler

Both share the `C_syntax` code generation layer (`c_syntax.ml`) via the `C_syntax_config` module type, which parameterizes kernel syntax (attributes, type names, intrinsics).

### HIP-CUDA correspondence

HIP is designed as a near drop-in for CUDA. Key differences for code generation:
- Half precision: `_Float16` (not `__half`)
- BFloat16: `__hip_bfloat16` (not `__nv_bfloat16`)
- Compilation: hipRTC (`hiprtcCompileProgram`) produces device binary (not PTX), loaded via `hipModuleLoadData`
- Context model: HIP supports explicit contexts (`hipCtxSetCurrent`) matching CUDA's pattern

All kernel attributes (`__global__`, `__shared__`, `extern "C"`), math intrinsics, and launch configuration APIs are source-compatible.

### Current state of `ocaml-hipjit`

The repo exists at `~/ocaml-hipjit/` with LICENSE and README only -- no code yet. It needs to be built from scratch following the `ocaml-cudajit` structure:
- `hip_ffi/` -- ctypes FFI bindings to HIP driver API
- `hiprtc_ffi/` -- ctypes FFI bindings to hipRTC
- `src/hip.ml` + `hip.mli` -- high-level OCaml wrapper (mirroring `cuda.mli` ~836 lines)
- `src/hiprtc.ml` + `hiprtc.mli` -- runtime compiler wrapper

### Code pointers

| File | Role |
|------|------|
| `arrayjit/lib/backend_intf.ml:357-377` | `Backend` module type |
| `arrayjit/lib/cuda_backend.ml` | Primary reference (~1055 lines) |
| `arrayjit/lib/cuda_backend_impl.cudajit.ml` | 1-line: `include Cuda_backend` |
| `arrayjit/lib/cuda_backend_impl.missing.ml` | Stub using `Lowered_backend_missing` |
| `arrayjit/lib/backends.ml:643-665` | `fresh_backend` registration |
| `arrayjit/lib/dune:62-76` | `cuda_backend` optional library definition |
| `arrayjit/lib/dune:94-142` | `context` library with select directives |
| `arrayjit/lib/c_syntax.ml:16` | `C_syntax_config` module type |
| `~/ocaml-cudajit/` | Bindings package to mirror |

### Testing strategy

- Development can proceed using `HIP_PLATFORM=nvidia` which runs HIP code on NVIDIA GPUs via CUDA, enabling testing without AMD hardware
- ROCm 5.x+ is the target for mature hipRTC support
- The CUDA backend's single-threaded kernel launch configuration (`grid_dim=1, block_dim=1`) applies identically to HIP

### Build integration

New files needed in OCANNL:
- `arrayjit/lib/hip_backend.ml` -- backend implementation
- `arrayjit/lib/hip_backend.mli` -- interface
- `arrayjit/lib/builtins_hip.ml` -- AMD-specific builtins (likely near-identical to CUDA)
- `arrayjit/lib/hip_backend_impl.hipjit.ml` -- `include Hip_backend`
- `arrayjit/lib/hip_backend_impl.missing.ml` -- stub
- `arrayjit/lib/hip_backend_impl.mli` -- interface

Dune additions: new `hip_backend` optional library, new select directive in `context` library, `"hip"` case in `fresh_backend`.
