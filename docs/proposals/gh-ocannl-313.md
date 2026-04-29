# Proposal: Get MSVC to work on the C backend for native Windows

**Task**: gh-ocannl-313
**Issue**: https://github.com/ahrefs/ocannl/issues/313

## Goal

Enable OCANNL's C backend (`cc_backend.ml`) to compile generated C code using MSVC (`cl.exe`) on native Windows, load the resulting DLL, and execute compiled functions -- without regressing GCC/Clang compilation on Unix.

## Acceptance Criteria

- MSVC (`cl.exe`) can compile OCANNL-generated C code into a DLL
- The compiled DLL is loaded and functions are called successfully via ctypes
- MSVC compiler flags are correctly formatted (`/O2 /LD` not `-O3 -shared`)
- Dynamic library loading works on Windows (ctypes `Dl.dlopen` maps to `LoadLibrary`)
- GCC/Clang compilation on Unix is not affected (no regressions)
- At least one OCANNL test passes end-to-end on Windows with MSVC
- Generated function signatures include `__declspec(dllexport)` on Windows so symbols are visible to `GetProcAddress`

## Context

### Current compilation pipeline

The C backend lives in three key files:
- `arrayjit/lib/cc_backend.ml` -- compiler invocation, dynamic loading, linking
- `arrayjit/lib/c_syntax.ml` -- C code generation (PPrint-based AST to C text)
- `arrayjit/lib/builtins_cc.ml` -- C builtins (float16/bfloat16/fp8 emulation, threefry PRNG, vector types)

The compilation flow:
1. `compile` / `compile_batch` generates a `.c` file via `C_syntax.compile_proc` + `filter_and_prepend_builtins`
2. `c_compile_and_load` (cc_backend.ml:62-153) shells out to the system C compiler
3. Compiler detected via `ocamlc -config` field `c_compiler:` (cc_backend.ml:23-41)
4. The resulting `.so`/`.dll` is loaded with `Dl.dlopen ~flags:[RTLD_NOW]` (line 150)
5. Functions are looked up via `Foreign.foreign ~from:lib name` (line 391)

### What already works on Windows

- File extension: `.dll` on `Sys.win32` (lines 71, 74)
- Null device: `nul` on `Sys.win32` (line 140)
- OS type detection: `Sys.os_type` match includes `"Win32"` and `"Cygwin"` cases (line 83), but uses GCC-style `-shared` flag

### What does NOT work with MSVC

**1. Compiler flags (cc_backend.ml:86-101)**
The entire flag construction is GCC-specific:
- `compiler_flags` (line 88-96): `-O3`, `-march=native`, `-ffast-math` -- all GCC syntax
- `kernel_link_flags` (line 77-85): `-shared -fPIC` on Linux, `-bundle` on macOS, `-shared` on Win32 -- the Win32 case uses GCC/MinGW syntax, not MSVC
- `cmdline` (line 98-100): `compiler f_path flags -o libname link_flags` -- MSVC uses different argument ordering and `/` prefix flags

**2. DLL symbol export (c_syntax.ml:803-805)**
GCC with `-shared` exports all symbols by default. MSVC does NOT. Functions must be marked `__declspec(dllexport)` or a `.def` file must be provided. The function header is generated as:
```ocaml
string B.main_kernel_prefix ^^ space ^^ string "void" ^^ space ^^ string name
```
For CC backend, `main_kernel_prefix` is `""` (set in `Pure_C_config`, c_syntax.ml:90). On Windows/MSVC this needs to become `"__declspec(dllexport)"`.

**3. Dynamic library loading (cc_backend.ml:150)**
`Dl.dlopen` wraps POSIX `dlopen`. The OCaml ctypes library's `Dl` module maps to `LoadLibrary`/`GetProcAddress` on Windows when ctypes is compiled with Windows support. This needs verification but is likely a non-issue since ctypes is actively maintained cross-platform.

### Generated C code portability assessment

The generated C code is standard C99 -- no GCC extensions found:
- No `__attribute__`, `__builtin_*`, inline assembly, or VLAs
- Standard headers only: `stdio.h`, `math.h`, `stdint.h`, `string.h`, `stdlib.h`
- Math functions used (`fmax`, `fmin`, `ldexpf`, `fabsf`, `isinf`, `isnan`) all available in MSVC
- Type punning via `memcpy` and pointer cast (`*((float *)&u32)`) works on MSVC
- Half/BFloat16/FP8 emulation uses portable bit manipulation

**One caveat in `builtins_cc.ml`**: The `HALF_TO_UINT16` and `UINT16_TO_HALF` macros (lines 73, 82) use GCC statement expressions (`({ ... })`), but only inside `#if HAS_NATIVE_FLOAT16` guards. Since MSVC does not define `__FLT16_MAX__`, `HAS_NATIVE_FLOAT16` will be 0, so these macros are never compiled on MSVC. No change needed.

### Key code locations

| What | File | Lines |
|------|------|-------|
| Compiler command detection | `cc_backend.ml` | 23-41 |
| Compilation + loading | `cc_backend.ml` | 62-153 |
| Platform-specific link flags | `cc_backend.ml` | 77-85 |
| Compiler flags (optimization, arch, fast-math) | `cc_backend.ml` | 87-97 |
| Command line assembly | `cc_backend.ml` | 98-101 |
| Dynamic loading (`Dl.dlopen`) | `cc_backend.ml` | 150 |
| Function linking (`Foreign.foreign`) | `cc_backend.ml` | 391 |
| Function header generation | `c_syntax.ml` | 803-805 |
| `main_kernel_prefix` (empty for CC) | `c_syntax.ml` | 90 |
| C includes/builtins | `builtins_cc.ml` | 1-10 |
| `builtins.c` (C stubs, same float16 macros) | `arrayjit/lib/builtins.c` | 1-32 |
| Dune config (foreign_stubs) | `arrayjit/lib/dune` | 37-39 |

## Approach

### 1. Add compiler family detection (cc_backend.ml)

Add a `compiler_family` type and detection function:

```ocaml
type compiler_family = GCC_compatible | MSVC

let detect_compiler_family () =
  let cmd = compiler_command () in
  (* MSVC toolchain: cl.exe or cl *)
  if String.is_substring cmd ~substring:"cl.exe"
     || (String.is_suffix cmd ~suffix:"cl"
         && not (String.is_substring cmd ~substring:"clang")) then
    MSVC
  else GCC_compatible
```

Expose this so `Pure_C_config` can also use it for setting `main_kernel_prefix`.

### 2. Branch compiler flag construction (cc_backend.ml:77-101)

Replace the current monolithic flag+cmdline construction with a compiler-family branch:

**Link flags:**
- GCC/Clang: `-shared -fPIC` (Linux), `-bundle -undefined dynamic_lookup` (macOS)
- MSVC: `/LD` (create DLL)

**Compiler flags:**
| GCC/Clang | MSVC | Notes |
|-----------|------|-------|
| `-O3` | `/O2` | MSVC max optimization; `/Ox` for full |
| `-march=native` | `/arch:AVX2` or empty | Config-driven, default empty for MSVC |
| `-ffast-math` | `/fp:fast` | |
| `-fPIC` | (not needed) | Windows DLLs always position-independent |

**Command line format:**
- GCC: `cc source.c -O3 -march=native -o output.so -shared -fPIC > log 2>&1`
- MSVC: `cl.exe /O2 /LD /Fe:output.dll source.c > log 2>&1`

### 3. Add `__declspec(dllexport)` to generated functions (c_syntax.ml)

Modify `Pure_C_config` to set `main_kernel_prefix`:

```ocaml
let main_kernel_prefix =
  if Sys.win32 then "__declspec(dllexport)" else ""
```

This uses the existing `main_kernel_prefix` mechanism (already used by CUDA for `extern "C" __global__` and Metal for `kernel`). The field is concatenated before `void` in the function signature at c_syntax.ml:804.

### 4. Handle MSVC auxiliary file cleanup (cc_backend.ml)

MSVC generates `.lib` and `.exp` files alongside `.dll`. Add cleanup of these files:
- After successful compilation, remove `base_name.lib` and `base_name.exp`
- On compilation failure, include these in cleanup too

### 5. Add MSVC arch flag configuration (cc_backend.ml)

The `arch_flags` setting (line 20) defaults to `-march=native` which is GCC-only. For MSVC, default to empty string (MSVC auto-detects) or allow `/arch:AVX2` via the existing `cc_backend_arch_flags` config:

```ocaml
let arch_flags () =
  let default = match detect_compiler_family () with
    | GCC_compatible -> "-march=native"
    | MSVC -> ""  (* MSVC auto-detects; user can set /arch:AVX2 *)
  in
  Utils.get_global_arg ~default ~arg_name:"cc_backend_arch_flags"
```

### 6. Verify and test

- Verify `Dl.dlopen` maps to `LoadLibrary` on Windows (read ctypes source or test)
- If ctypes `Dl` does not support Windows, provide a fallback (unlikely needed)
- Run existing tests on Windows with MSVC toolchain
- Ensure `builtins.c` (the OCaml C stubs, not the generated builtins) compiles with MSVC via dune -- dune handles this via `(foreign_stubs)` which respects `ocamlc -config` compiler

### Edge cases

- **MinGW vs MSVC on Windows**: If OCaml is built with MinGW-GCC, `ocamlc -config` returns `gcc` and existing code works fine. The MSVC path only activates when OCaml uses MSVC toolchain.
- **MSVC version**: Require MSVC 2015 (v14.0)+ for C99 compound literal and mixed declaration support.
- **Clang-cl**: Microsoft's `clang-cl` accepts both `/` and `-` flags. The GCC-compatible path should work but may need minor adjustments. This is a follow-up concern.
- **`builtins.c` GCC statement expressions**: Lines 20-21 use `({ ... })` inside `#if HAS_NATIVE_FLOAT16` which MSVC won't compile. But MSVC never defines `__FLT16_MAX__` so `HAS_NATIVE_FLOAT16=0` and those macros are not compiled. Same applies to `builtins_cc.ml` lines 73, 82.
