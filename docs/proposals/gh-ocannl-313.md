# Proposal: Get MSVC to work on the C backend for native Windows

**Task**: gh-ocannl-313
**Issue**: https://github.com/ahrefs/ocannl/issues/313

## Goal

Enable OCANNL's C backend (`cc_backend.ml`) to compile generated C code using MSVC (`cl.exe`) on native Windows, load the resulting DLL, and execute compiled functions -- without regressing GCC/Clang compilation on Unix.

## Acceptance Criteria

- MSVC (`cl.exe`) can compile OCANNL-generated C code into a DLL
- The compiled DLL is loaded and functions are called successfully via ctypes
- MSVC compiler flags are correctly formatted (`/O2 /LD` not `-O3 -shared`)
- GCC/Clang compilation on Unix is not affected (no regressions)
- At least one OCANNL test passes end-to-end on Windows with MSVC

## Context

### Current compilation pipeline

The C backend lives in two files:
- `arrayjit/lib/cc_backend.ml` -- compiler invocation, dynamic loading, linking
- `arrayjit/lib/c_syntax.ml` -- C code generation (AST to PPrint documents)

The compilation flow:
1. `compile` / `compile_batch` generates a `.c` file via `C_syntax.compile_proc`
2. `c_compile_and_load` shells out to the system C compiler (detected via `ocamlc -config`)
3. The resulting `.so` / `.dll` is loaded with `Dl.dlopen` + `RTLD_NOW`
4. Functions are looked up via `Foreign.foreign ~from:lib name`

### What already works on Windows

- File extension handling: `.dll` on `Sys.win32` (lines 71, 74)
- Null device: `nul` on `Sys.win32` (line 140)
- OS type detection: `Sys.os_type` match includes `"Win32"` and `"Cygwin"` cases (line 83)

### What does NOT work with MSVC

**1. Compiler flags (cc_backend.ml lines 86-101)**
The current flag construction is entirely GCC-style:
- `-O3` optimization (MSVC uses `/O2`)
- `-march=native` arch flags (MSVC uses `/arch:AVX2` or similar)
- `-ffast-math` (MSVC uses `/fp:fast`)
- `-shared` / `-bundle` link flags (MSVC uses `/LD`)
- `-o output.dll` (MSVC uses `/Fe:output.dll`)

**2. DLL symbol export**
GCC with `-shared` exports all symbols by default. MSVC does NOT -- functions must be marked `__declspec(dllexport)` or listed in a `.def` file. Without this, `GetProcAddress` (used by ctypes' `Foreign.foreign`) cannot find the compiled functions.

The function header is generated in `c_syntax.ml` line 803-805:
```ocaml
let func_header =
  string B.main_kernel_prefix ^^ space ^^ string "void" ^^ space ^^ string name
```
For the CC backend, `main_kernel_prefix` is `""` (set in `Pure_C_config`). This needs to become `__declspec(dllexport)` on MSVC/Windows.

**3. Dynamic library loading**
`Dl.dlopen` (line 150) wraps POSIX `dlopen`. On Windows, ctypes' `Dl` module maps to `LoadLibrary`/`GetProcAddress`, which should work if ctypes was compiled with Windows support. This needs verification but is likely a non-issue.

### Generated C code portability

The generated C code is standard C99 with no GCC extensions (`__attribute__`, `__builtin_*`, inline assembly). MSVC 2015+ supports the required C99 subset. The builtins in `builtins_cc.ml` use portable constructs (bit manipulation for half/bfloat16/fp8 emulation, standard math functions). No changes to generated C code are expected beyond the export decoration.

### Key code locations

| What | File | Lines |
|------|------|-------|
| Compiler command detection | `cc_backend.ml` | 23-41 |
| Compilation + loading | `cc_backend.ml` | 62-153 |
| Platform-specific link flags | `cc_backend.ml` | 77-85 |
| Compiler flags (optimization, arch, fast-math) | `cc_backend.ml` | 87-97 |
| Command line assembly | `cc_backend.ml` | 98-101 |
| Dynamic loading | `cc_backend.ml` | 150 |
| Function header generation | `c_syntax.ml` | 803-805 |
| `main_kernel_prefix` (empty for CC) | `cc_backend.ml` | 160 (via `Pure_C_config`) |
| C includes / builtins | `builtins_cc.ml` | 1-10 |

## Approach

### 1. Detect MSVC vs GCC-compatible compiler

Add a `compiler_family` type and detection in `cc_backend.ml`. Detection based on `ocamlc -config` output: if the C compiler path contains `cl` or `cl.exe` (without a preceding `-` or `/`), it is MSVC. Expose as a function so other parts can query it.

### 2. Branch compiler flag construction

In `c_compile_and_load`, build the command line differently for MSVC:

| GCC/Clang | MSVC |
|-----------|------|
| `-O3` | `/O2` (max is `/O2`; `/Ox` for full) |
| `-shared -fPIC` | `/LD` |
| `-o output.dll` | `/Fe:output.dll` |
| `-march=native` | `/arch:AVX2` (or empty) |
| `-ffast-math` | `/fp:fast` |
| `-bundle -undefined dynamic_lookup` | N/A (macOS only) |

### 3. Add `__declspec(dllexport)` to generated functions on Windows

Modify `main_kernel_prefix` in the CC backend's `Pure_C_config` to return `"__declspec(dllexport)"` when `Sys.win32` is true, or `""` otherwise. This is the minimal change -- the field already exists and is concatenated before `void` in the function signature.

### 4. Verify `Dl.dlopen` on Windows

OCaml ctypes' `Dl` module should map to `LoadLibrary` on Windows. If it does not, provide a fallback using `Ctypes.Foreign` or direct FFI to `LoadLibraryA`/`GetProcAddress`. This is a verification step -- likely no code change needed.

### 5. Handle MSVC-specific edge cases

- MSVC generates `.lib` and `.exp` files alongside `.dll` -- temp file cleanup should handle these
- MSVC debug builds produce `.pdb` files -- not relevant for optimized compilation
- Minimum version: MSVC 2015 (v14.0) for C99 compound literal support
