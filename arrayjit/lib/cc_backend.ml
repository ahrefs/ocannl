open Base
module Lazy = Utils.Lazy
open Ir

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_CC_BACKEND=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_CC_BACKEND"]

include Backend_impl.No_device_buffer_and_copying ()
open Backend_intf

let name = "cc"

let optimization_level () =
  Int.of_string @@ Utils.get_global_arg ~default:"3" ~arg_name:"cc_backend_optimization_level"

let fast_math_enabled () = Utils.get_global_flag ~default:false ~arg_name:"cc_backend_fast_math"

(* The flag as passed to the compiler, shared by the compile line and the codegen tag so that the
   schedule cache's identity follows the flag's EXPANSION, not the boolean: [-ffast-math] minus
   [-ffinite-math-only]. The licence is reassociation, contraction, reciprocal approximation and no
   trapping -- never "assume no infinities". The lowering itself emits [-inf] (the [Max] neutral
   element, the default attention mask fill), and the online-softmax recurrence (gh-ocannl-483)
   rests on IEEE infinity arithmetic; under finite-math-only clang folds and reorders those into NaN
   (a masked-prefix row went NaN in the review of staging PR #737, reproduced on a five-line
   probe). *)
let fast_math_flag () =
  if fast_math_enabled () then Some "-ffast-math -fno-finite-math-only" else None

(* Toolchain probing -- resolving the compiler command, [arch_flags], [simd_flags],
   [parallel_grid_syntax_setting], [fp16_arithmetic_support] -- costs the better part of a dozen
   subprocesses per process: one [ocamlc -config], plus compile and link probes for each of the
   rest. Each is memoized by a [lazy], so the memoization is process-local, while a [dune build]
   runs a couple hundred test executables. On Windows, where process creation is an order of
   magnitude costlier than on Unix, that probing dominates short tests -- measured on one Windows
   box, test/operations/hello_world_dim1x1 runs in 1.52s probing against 1.00s cached.

   So memoize across processes too, in files under [probe_cache_dir]. The key covers everything that
   can change an answer without changing a file's identity: the settings feeding the probes, plus a
   digest of PATH standing in for which toolchain is active (an opam switch change, a different
   compiler in front). A stale entry would silently select wrong ISA flags, so the key errs towards
   re-probing; [cc_backend_probe_cache=false] bypasses the files entirely.

   Every failure degrades to probing in-process rather than raising: an unwritable temp directory, a
   torn or truncated file, a rename losing a race. Dune runs these executables in parallel, so
   several processes can probe at once and race to publish -- harmless, since they compute the same
   answer, and one file per probe keeps a publication from ever being interleaved with another's. *)
let probe_cache_marker = "ocannl-cc-probe-v1\n"

(* The cached values are interpolated straight into [Sys.command], so where they are stored is a
   privilege boundary, not a detail: a predictable name under a world-writable POSIX /tmp would let
   any local account pre-create a file bearing our marker and thereby choose the compiler command
   OCANNL runs. Hence a per-user directory created 0700, refused unless it really is a directory we
   own with no group/other access -- a squatted path (or a symlink, which is why this is [lstat] and
   not [stat]) fails the test and probing simply stays in-process, as it was before this cache.
   Ownership and permissions are checked on POSIX only: Windows has no equivalent exposure, its temp
   directory being per-user already, and it reports both halves synthetically -- [getuid] answers 1
   where [lstat] answers uid 0, and the mode always reads 0o777 -- so the test would reject OCANNL's
   own directory and quietly disable the cache on the platform that needs it most. That the path is
   a directory rather than a planted symlink is still checked everywhere. *)
let probe_cache_dir =
  lazy
    (let dir =
       Stdlib.Filename.concat
         (Stdlib.Filename.get_temp_dir_name ())
         (Printf.sprintf "ocannl_cc_probes_%d" (Unix.getuid ()))
     in
     try
       (try Unix.mkdir dir 0o700 with Unix.Unix_error (Unix.EEXIST, _, _) -> ());
       let st = Unix.lstat dir in
       let ours =
         match st.Unix.st_kind with
         | Unix.S_DIR ->
             Sys.win32 || (st.Unix.st_uid = Unix.getuid () && st.Unix.st_perm land 0o077 = 0)
         | _ -> false
       in
       Option.some_if ours dir
     with _ -> None)

(* Forward reference to [compiler_command] below: resolving the command is itself a cached probe, so
   the key of the probes that RUN the compiler is built from it, not the other way round. *)
let compiler_command_ref : (unit -> string) ref = ref (fun () -> "")

(* Whitespace-separated tokens, honoring quotes: the command is spelled for a shell, so a path
   containing spaces arrives quoted and splitting on the first space would fingerprint a
   fragment. *)
let shell_tokens command =
  let tokens = ref [] and current = Buffer.create 32 and quote = ref None in
  let flush () =
    if Buffer.length current > 0 then (
      tokens := Buffer.contents current :: !tokens;
      Buffer.clear current)
  in
  String.iter command ~f:(fun c ->
      match (!quote, c) with
      | Some q, _ when Char.equal q c -> quote := None
      | Some _, _ -> Buffer.add_char current c
      | None, ('"' | '\'') -> quote := Some c
      | None, (' ' | '\t') -> flush ()
      | None, _ -> Buffer.add_char current c);
  flush ();
  List.rev !tokens

let resolve_executable token =
  let candidates =
    if String.is_empty token then []
    else if String.exists token ~f:(fun c -> Char.equal c '/' || Char.equal c '\\') then [ token ]
    else
      let path = Option.value (Stdlib.Sys.getenv_opt "PATH") ~default:"" in
      let dirs = String.split path ~on:(if Sys.win32 then ';' else ':') in
      let exts = if Sys.win32 then [ ""; ".exe"; ".bat"; ".cmd" ] else [ "" ] in
      List.concat_map dirs ~f:(fun dir ->
          List.map exts ~f:(fun ext -> Stdlib.Filename.concat dir (token ^ ext)))
  in
  (* Executable, not merely present (Codex P1 on PR #337): a PATH search skips a non-executable file
     of the right name and runs a later one, so accepting the first regular file would key on a file
     the compile never runs -- and then an upgrade of the one it does run would go unnoticed. [X_OK]
     is meaningless on Windows, where OCaml's [access] does not implement it and the extension list
     above carries the same information. *)
  let is_executable path =
    Sys.win32
    ||
      try
        Unix.access path [ Unix.X_OK ];
        true
      with _ -> false
  in
  List.find_map candidates ~f:(fun path ->
      match try Some (Unix.stat path) with _ -> None with
      | Some ({ Unix.st_kind = Unix.S_REG; _ } as st) when is_executable path ->
          (* [stat], not [lstat]: /usr/bin/cc is a symlink, and it is the TARGET whose size and
             mtime move when the toolchain is upgraded. *)
          Some (Printf.sprintf "%s:%d:%.0f" path st.Unix.st_size st.Unix.st_mtime)
      | _ -> None)

(* The identity of the compiler EXECUTABLE, for the probe-cache key below (Codex P1 on PR #337). An
   in-place toolchain upgrade leaves the command name, the flags and PATH unchanged, so keyed on
   those alone every cached probe would keep answering for the old compiler indefinitely — the
   target fingerprint most absurdly of all, since noticing exactly this is its job. Resolved by
   walking PATH in process: asking anything would spend the subprocess the cache exists to save.
   Size and mtime, not a version string, for the same reason; an unresolvable command degrades to
   its spelling, which is what the key held before. *)

(* EVERY token that names an executable, not just the first (Codex P1 on PR #337): the command can
   be a wrapper — [ccache clang] — where the first token is the one that never changes, and flags
   simply do not resolve. The residual is a wrapper that names no compiler on its command line (a
   shell script calling one internally): its own file is fingerprinted, but not what it invokes,
   which nothing short of running it could see — [cc_backend_probe_cache=false] is the escape hatch
   there. *)
let compiler_executable_identity command =
  match List.filter_map (shell_tokens command) ~f:resolve_executable with
  | [] -> "unresolved:" ^ String.strip command
  | identities -> String.concat ~sep:"|" identities

let probe_cache_path =
  let key_digest =
    lazy
      (* Raw settings, not resolved values: [arch_flags] is itself one of the probes cached here, so
         keying on its result would both invert the dependency and spend the probes the key exists
         to avoid. Nothing is lost -- what it resolves to is a function of the compiler and the
         target, and both are already pinned by the entries below.

         Every setting a cached probe CONSULTS has to be here (gh-ocannl-572, the same completeness
         rule as the schedule cache's): the fp16 probe compiles under [arch_flags () ^ simd_flags
         ()], so an explicit [cc_backend_simd_flags] can change its answer -- a target whose
         [_Float16] vector arithmetic only appears under the added flags -- and without the setting
         in the key the first run's answer would be served to the other configuration. *)
      (let key =
         String.concat ~sep:"\000"
           [
             Utils.get_global_arg ~default:"" ~arg_name:"cc_backend_compiler_command";
             Utils.get_global_arg ~default:"auto" ~arg_name:"cc_backend_arch_flags";
             Utils.get_global_arg ~default:"auto" ~arg_name:"cc_backend_simd_flags";
             Option.value (Stdlib.Sys.getenv_opt "PATH") ~default:"";
             (* The identity of [ocamlc] itself: with the command setting unset, the "compiler"
                probe caches what [ocamlc -config] reports, so an OCaml installation upgraded in
                place would keep naming the previous [c_compiler] -- or one that no longer exists
                (Codex P1 on PR #337). This is the base key's own executable, the one exemption the
                two-stage scheme leaves; downstream probes additionally key on the C compiler. *)
             compiler_executable_identity "ocamlc";
           ]
       in
       String.prefix (Stdlib.Digest.to_hex (Stdlib.Digest.string key)) 16)
  in
  (* Two stages, because the compiler command is itself one of the probes: with the setting unset it
     comes from [ocamlc -config], so a key that named the executable could not be built before that
     probe answered. The "compiler" probe keeps the settings-only key; every probe that RUNS the
     compiler keys on the executable it will run. *)
  let full_digest =
    lazy
      (let key =
         String.concat ~sep:"\000"
           [
             Lazy.force key_digest; compiler_executable_identity (compiler_command_ref.contents ());
           ]
       in
       String.prefix (Stdlib.Digest.to_hex (Stdlib.Digest.string key)) 16)
  in
  fun ~name ->
    Option.map (Lazy.force probe_cache_dir) ~f:(fun dir ->
        let digest =
          if String.equal name "compiler" then Lazy.force key_digest else Lazy.force full_digest
        in
        Stdlib.Filename.concat dir (Printf.sprintf "ocannl_cc_probe_%s_%s.txt" name digest))

(* [compute] must be deterministic given the key above: its result is what every later process on
   this machine will use. [validate] rejects a cached value the caller cannot interpret, so that a
   foreign file carrying our marker re-probes instead of propagating a parse failure. *)
let cached_probe ?(validate = fun _ -> true) ~name ~compute () =
  match
    if Utils.get_global_flag ~default:true ~arg_name:"cc_backend_probe_cache" then
      probe_cache_path ~name
    else None
  with
  | None -> compute ()
  | Some path -> (
      (* Before the read, not after a miss: a hit is the common path, so sweeping only on the miss
         side would leave a staging file abandoned by a killed writer sitting in the probe directory
         for as long as the entry it was meant to replace stays valid (Codex P2, round 4). Once per
         directory per process either way. *)
      (try Utils.Atomic_file.cleanup_stale_once (Stdlib.Filename.dirname path) with _ -> ());
      let cached =
        try
          let data = Stdio.In_channel.read_all path in
          (* The marker separates "the probe answered with the empty string" -- which [simd_flags]
             legitimately does -- from a half-written or foreign file. *)
          if String.is_prefix data ~prefix:probe_cache_marker then
            let value = String.drop_prefix data (String.length probe_cache_marker) in
            Option.some_if (validate value) value
          else None
        with _ -> None
      in
      match cached with
      | Some value -> value
      | None ->
          let value = compute () in
          (* Publish by rename so a concurrent reader sees either the old file or the whole new one,
             never a partial write. [Atomic_file] stages inside the same (private) directory, which
             keeps the rename on one volume — it requires that — and keeps the staging file
             unreadable to anyone else too. The probe cache is an optimization, so every failure
             here is swallowed: the value has already been computed. *)
          (try Utils.Atomic_file.write_all ~path ~data:(probe_cache_marker ^ value) ()
           with _ -> ());
          value)

let compiler_command =
  let resolved =
    (* TODO: there's a direct way to get the compiler command from the OCaml compiler. *)
    lazy
      (cached_probe ~name:"compiler"
         ~compute:(fun () ->
           let ic = Unix.open_process_in "ocamlc -config" in
           let rec find_compiler () =
             match In_channel.input_line ic with
             | None -> "cc" (* Default fallback *)
             | Some line ->
                 if String.is_prefix line ~prefix:"c_compiler: " then
                   String.drop_prefix line 12 (* Length of "c_compiler: " *)
                 else find_compiler ()
           in
           let compiler = find_compiler () in
           ignore (Unix.close_process_in ic);
           compiler)
         ())
  in
  fun () ->
    (* Resolve lazily rather than passing [Lazy.force resolved] as [~default]: that spawns [ocamlc
       -config] even when the setting makes the answer irrelevant. An empty setting means unset --
       which is how [ocannl_config.reference] spells it. *)
    match Utils.get_global_arg ~default:"" ~arg_name:"cc_backend_compiler_command" with
    | "" -> Lazy.force resolved
    | command -> command

let () = compiler_command_ref := compiler_command

(* gh-ocannl-164: explicit SIMD flags appended to the compiler invocation. The "auto" default
   probes in two stages (once per process, and once per machine via [cached_probe]). Stage 1
   test-compiles a translation unit that #errors
   unless the target selected by [arch_flags] alone already defines __AVX2__ and __FMA__ — so the
   explicit flags never escalate the ISA beyond the configured target (a non-AVX2 x86 CPU under
   -march=native, or any ARM target, gets no flags and the generated code runs wherever the target
   does). Stage 2 verifies the compiler accepts the candidate flags themselves, preferring the
   variant with -ftree-vectorize (explicit, though -O3 usually implies it). Any other config value
   is passed through verbatim (empty disables). *)
(* Probe whether the configured compiler accepts [flags] on the translation unit [data].
   [`Compile] is compile-only ([-c]); [`Link] builds a full executable ([data] must define
   [main]), catching runtimes that the compiler accepts flags for but cannot link -- e.g. clang
   accepting [-fopenmp] at compile time without libomp installed. Shared by the SIMD-flags and
   parallel-grid probes. *)
let probe_compiles ?(mode = `Compile) ~flags ~data () =
  let src = Stdlib.Filename.temp_file "ocannl_cc_probe_" ".c" in
  let out = Stdlib.Filename.temp_file "ocannl_cc_probe_" ".out" in
  let log = Stdlib.Filename.temp_file "ocannl_cc_probe_" ".log" in
  let ok =
    try
      Stdio.Out_channel.write_all src ~data;
      let compile_only = match mode with `Compile -> "-c " | `Link -> "" in
      (* Quoted: [temp_file] inherits the system temp directory, which can contain whitespace (a
         custom TMPDIR, and routinely on Windows), and an unquoted path would split into arguments
         and fail the probe -- silently, since a failed probe is indistinguishable from a negative
         answer here (Codex P1 on PR #337). *)
      let cmd =
        Printf.sprintf "%s %s %s%s -o %s > %s 2>&1" (compiler_command ()) flags compile_only
          (Stdlib.Filename.quote src) (Stdlib.Filename.quote out) (Stdlib.Filename.quote log)
      in
      Stdlib.Sys.command cmd = 0
    with _ -> false
  in
  List.iter [ src; out; log ] ~f:(fun f -> try Stdlib.Sys.remove f with _ -> ());
  ok

(* The flag that tells the compiler to target the host CPU. Its spelling is architecture-specific,
   and getting it wrong is worse than passing nothing: Apple clang accepts [-march=native] on arm64
   and *downgrades* the target with it -- 22 [__ARM_FEATURE_*] macros against 26 with no flag and 33
   with [-mcpu=native], losing [__ARM_FEATURE_FP16_VECTOR_ARITHMETIC] among them, so a machine with
   native 16-bit arithmetic looked like one without (gh-ocannl-516). On x86 the mistake runs the
   other way: [-mcpu=] is an alias for [-mtune=] there, which selects scheduling but not the ISA, so
   it would silently forgo AVX2.

   "auto" (the default) therefore asks the target which family it is in and then probes that
   family's spelling, falling back to no flag -- the compiler's own default target, which is the
   host on Apple and a portable baseline elsewhere. Any other config value is passed through
   verbatim, so [cc_backend_arch_flags=-march=native] restores the old behavior exactly. *)
let arch_flags =
  let probed =
    lazy
      (cached_probe ~name:"arch"
         ~compute:(fun () ->
           let is_arm =
             probe_compiles ~flags:""
               ~data:
                 "#if !defined(__aarch64__) && !defined(__arm__) && !defined(_M_ARM64)\n\
                  #error \"not ARM\"\n\
                  #endif\n\
                  int ocannl_arch_probe;\n"
               ()
           in
           let trivial = "int ocannl_arch_probe;\n" in
           let candidate = if is_arm then "-mcpu=native" else "-march=native" in
           if probe_compiles ~flags:candidate ~data:trivial () then candidate else "")
         ())
  in
  fun () ->
    match Utils.get_global_arg ~default:"auto" ~arg_name:"cc_backend_arch_flags" with
    | "auto" -> Lazy.force probed
    (* The portable baseline, spelled as a word because a config source cannot carry an empty value
       (an empty setting means "unset"), and the `reproducible` profile has to be able to pin it. *)
    | "none" -> ""
    | flags -> flags

(* Floating-point contraction beyond what the codegen selects explicitly (it emits `fmaf` where it
   wants an FMA): whether the compiler is free to fuse a*b+c on its own is compiler- and
   target-discretionary, so a reproducible run pins it off. "auto" (the default) passes no flag,
   leaving the compiler's own default -- which is what every OCANNL release before gh-ocannl-559
   did. *)
let fp_contract_flag () =
  match
    String.lowercase
      (String.strip (Utils.get_global_arg ~default:"auto" ~arg_name:"cc_backend_fp_contract"))
  with
  | "auto" -> None
  | ("off" | "on" | "fast") as mode -> Some ("-ffp-contract=" ^ mode)
  | other -> invalid_arg ("cc_backend_fp_contract: expected auto | off | on | fast, got " ^ other)

let simd_flags =
  let probed =
    lazy
      (cached_probe ~name:"simd"
         ~compute:(fun () ->
           let compile ~flags ~data = probe_compiles ~flags ~data () in
           let guard =
             "#if !defined(__AVX2__) || !defined(__FMA__)\n\
              #error \"target lacks AVX2/FMA\"\n\
              #endif\n\
              int ocannl_simd_probe;\n"
           in
           let trivial = "int ocannl_simd_probe;\n" in
           if not (compile ~flags:(String.strip (arch_flags ())) ~data:guard) then ""
           else if compile ~flags:"-mavx2 -mfma -ftree-vectorize" ~data:trivial then
             "-mavx2 -mfma -ftree-vectorize"
           else if compile ~flags:"-mavx2 -mfma" ~data:trivial then "-mavx2 -mfma"
           else "")
         ())
  in
  fun () ->
    match Utils.get_global_arg ~default:"auto" ~arg_name:"cc_backend_simd_flags" with
    | "auto" -> Lazy.force probed
    (* No SIMD flags, spelled as a word for the same reason as [arch_flags]' "none": a config source
       cannot carry an empty value. Whether the probe fires at all depends on what the toolchain's
       default target already exposes, which is a per-machine fact, so a run that must be
       machine-independent pins this rather than reasoning about which of the added flags could have
       changed a result. *)
    | "none" -> ""
    | flags -> flags

(* Pool-backed Grid rendering (docs/proposals/gh-ocannl-164.md): eligible outermost [Grid] loops
   render as chunked parallel loops on a process-global native pool -- libdispatch's
   [dispatch_apply] on macOS (blocks; no pool state in the compiled kernel), OpenMP elsewhere
   (libgomp/libomp is likewise process-global). The worker threads run pure C on raw kernel
   arguments; the OCaml runtime is never involved. "auto" probes what the configured compiler
   accepts, once per process (and once per machine, via [cached_probe]). *)
let parallel_grid_of_string = function
  | "dispatch" -> Some `Dispatch
  | "openmp" -> Some `Openmp
  | "none" | "false" -> Some `None
  | _ -> None

let parallel_grid_syntax_setting =
  let probed =
    lazy
      (let probe () =
         (* Probes link full executables: a compiler can accept the flags yet lack the runtime
            library at link time (clang without libomp), and the kernel command compiles and links
            in one step. *)
         let dispatch_src =
           "#include <dispatch/dispatch.h>\n\
            int main(void) {\n\
           \  dispatch_apply(1, DISPATCH_APPLY_AUTO, ^(size_t i) { (void)i; });\n\
           \  return 0;\n\
            }\n"
         in
         let omp_src =
           "int main(void) {\n\
           \  float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};\n\
            #pragma omp parallel for\n\
           \  for (int i = 0; i < 4; ++i) a[i] += 1.0f;\n\
           \  return (int)a[0] - 1;\n\
            }\n"
         in
         if probe_compiles ~mode:`Link ~flags:"" ~data:dispatch_src () then "dispatch"
         else if probe_compiles ~mode:`Link ~flags:"-fopenmp" ~data:omp_src () then "openmp"
         else "none"
       in
       let cached =
         cached_probe ~name:"parallel_grid"
           ~validate:(fun s -> Option.is_some (parallel_grid_of_string s))
           ~compute:probe ()
       in
       (* [validate] already rejected anything unparseable, so the probe's own vocabulary is the
          only remaining source. *)
       Option.value_exn ~message:"Cc_backend: unexpected parallel-grid probe result"
         (parallel_grid_of_string cached))
  in
  fun () ->
    let setting =
      String.lowercase (Utils.get_global_arg ~default:"auto" ~arg_name:"cc_parallel_grid")
    in
    if String.equal setting "auto" then Lazy.force probed
    else
      match parallel_grid_of_string setting with
      | Some mode -> mode
      | None ->
          invalid_arg ("cc_parallel_grid: expected auto | dispatch | openmp | none, got " ^ setting)

(* gh-ocannl-516: whether the target has native fp16 arithmetic. Three states, and the middle one is
   the reason a boolean will not do:

   - [`None]: no [_Float16] at all. Half is emulated -- stored as uint16, computed through float. -
   [`Promoted]: the type exists and its arithmetic is correct, but the compiler implements it by
   promoting to float (x86-64 without AVX512-FP16). Better than bf16's explicit round-trips because
   the compiler owns the promotion and can keep values in registers, but there is no lane-count win,
   so the vector renderings and the cost model must not expect one. - [`Native]: genuine 16-bit
   vector arithmetic (ARMv8.2-FP16, AVX512-FP16) -- twice f32's lanes.

   All three are C-preprocessor facts, resolved when [cc] compiles a kernel, while the renderer
   decides what to emit in OCaml well before that. Probing the compiler once per process and
   carrying the answer in [hardware_limits] puts a target capability where target capabilities
   already live, and keeps a half kernel's body single-armed. *)
let fp16_arithmetic_of_string = function
  | "none" | "false" -> Some `None
  | "promoted" -> Some `Promoted
  | "native" | "true" -> Some `Native
  | _ -> None

let fp16_arithmetic_support =
  let probed =
    lazy
      (let cached =
         cached_probe ~name:"fp16"
           ~validate:(fun s -> Option.is_some (fp16_arithmetic_of_string s))
           ~compute:(fun () ->
             let flags = String.strip (arch_flags () ^ " " ^ simd_flags ()) in
             (* Vector arithmetic, not just the type: a target can define [__FLT16_MAX__] and still
                reject [_Float16] in a [vector_size] type. *)
             let typed =
               "#ifndef __FLT16_MAX__\n\
                #error \"no _Float16\"\n\
                #endif\n\
                typedef _Float16 ocannl_v8h __attribute__((vector_size(16)));\n\
                ocannl_v8h ocannl_fp16_probe(ocannl_v8h a, ocannl_v8h b, ocannl_v8h c) { return a \
                * b + c; }\n"
             in
             let native =
               "#if !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(__AVX512FP16__)\n\
                #error \"fp16 arithmetic is promoted to float, not native\"\n\
                #endif\n\
                int ocannl_fp16_native_probe;\n"
             in
             if not (probe_compiles ~flags ~data:typed ()) then "none"
             else if probe_compiles ~flags ~data:native () then "native"
             else "promoted")
           ()
       in
       Option.value_exn ~message:"Cc_backend: unexpected fp16-arithmetic probe result"
         (fp16_arithmetic_of_string cached))
  in
  fun () ->
    let setting =
      String.lowercase (Utils.get_global_arg ~default:"auto" ~arg_name:"cc_fp16_arithmetic")
    in
    if String.equal setting "auto" then Lazy.force probed
    else
      match fp16_arithmetic_of_string setting with
      | Some support -> support
      | None ->
          invalid_arg
            ("cc_fp16_arithmetic: expected auto | none | promoted | native, got " ^ setting)

let has_native_fp16_arithmetic () =
  match fp16_arithmetic_support () with `Native -> true | `Promoted | `None -> false

(* Whether the target the kernels are compiled for has 512-bit vectors, probed the same way as
   [simd_flags]' first stage: under the flags a kernel actually gets, and asserting rather than
   requesting, so the answer never escalates the ISA beyond the configured target.

   [BW] and [VL] alongside [F] because a 64-byte vector here is routinely a vector of 16-bit
   elements (narrow storage bridges through [ocannl_vec32h]/[ocannl_vec32u16]), which is what BW
   makes a single-instruction affair; VL is what the 128/256-bit forms of the same instructions
   need, and the emission mixes widths within one kernel. Every AVX-512 part that has ever shipped
   outside Xeon Phi has all three, so the conjunction costs nothing real and keeps the width from
   claiming hardware the narrow paths would then find missing. *)
let avx512_target =
  let probed =
    lazy
      (String.equal "true"
         (cached_probe ~name:"avx512"
            ~validate:(fun s -> String.equal s "true" || String.equal s "false")
            ~compute:(fun () ->
              let flags = String.strip (arch_flags () ^ " " ^ simd_flags ()) in
              let guard =
                "#if !defined(__AVX512F__) || !defined(__AVX512BW__) || !defined(__AVX512VL__)\n\
                 #error \"target lacks AVX-512 F/BW/VL\"\n\
                 #endif\n\
                 int ocannl_avx512_probe;\n"
              in
              if probe_compiles ~flags ~data:guard () then "true" else "false")
            ()))
  in
  fun () -> Lazy.force probed

(* Explicit SIMD width for [Vectorized] loops (gh-ocannl-164 follow-up): vector register bytes for
   the GCC/Clang vector-extension rendering in [C_syntax]. Auto (-1 or unset): 64 bytes where the
   target has AVX-512 (gh-ocannl-621 follow-up), 32 where the SIMD probe found AVX2, else 16 (NEON
   width; clang/gcc lower 16-byte vectors natively on ARM). 0 disables explicit emission
   (auto-vectorization pragmas remain).

   The 512-bit rung is measured, not inferred: on Zen 5 (Ryzen AI Max+ 395, gcc 15) the packed f32
   GEBP of [bin/narrow_gebp_bench] at n = 512 runs 225.7 GFLOP/s at 64 bytes against 130.5 at 32,
   f16 189.6 against 108.1 and bf16 140.6 against 95.1, with identical checksums. It is a default
   rather than a hint because the alternative -- a machine that has the registers using half of them
   -- is the more surprising of the two. Where 512-bit operation is not wanted (Intel server parts
   clock down under sustained 512-bit work in a way Zen 5 does not), pin [cc_vector_bytes=32]; and
   note the width is part of what a schedule-cache entry keys on, so flipping it re-tunes rather
   than replaying a crown chosen at the other width.

   Widening a *default* cannot narrow what already vectorizes: a loop too short to fill the wider
   vector degrades to the width it can fill rather than declining, see [vec_lanes_for] in
   [C_syntax]. *)
let auto_vector_bytes () =
  (* Gated on the AVX2 stage having fired, so that [cc_backend_simd_flags=none] -- pinned by a run
     that must not depend on what this machine happens to have -- keeps answering 16 rather than
     acquiring a wider vector through the arch flag's back door. *)
  if not (String.is_substring (simd_flags ()) ~substring:"avx2") then 16
  else if avx512_target () then 64
  else 32

let vector_bytes_setting () =
  match Int.of_string @@ Utils.get_global_arg ~default:"-1" ~arg_name:"cc_vector_bytes" with
  | n when n >= 0 -> n
  | _ -> auto_vector_bytes ()

(* The register file the kernels' target allocates vectors from (gh-ocannl-1285): its register count
   and native register width -- 32 x 64 bytes on AVX-512 (F and VL: VL is what gives the narrower
   forms the same 32), 16 x 32 on AVX/AVX2, 32 x 16 on AArch64 (NEON), 16 x 16 on any other target
   (SSE, and whatever the probe does not recognize, which then caps the register tile at tinyBLAS's
   AVX2 4 x 3 rather than risking a spill). Probed like [avx512_target], under the flags a kernel
   actually gets, so it describes the target the compiler allocates for rather than the host. *)
let native_vector_file =
  let probed =
    lazy
      (let parse s =
         match String.split s ~on:'x' with
         | [ regs; bytes ] -> (
             match (Int.of_string_opt regs, Int.of_string_opt bytes) with
             | Some r, Some b when r > 0 && b > 0 -> Some (r, b)
             | _ -> None)
         | _ -> None
       in
       let answer =
         cached_probe ~name:"vector_file"
           ~validate:(fun s -> Option.is_some (parse s))
           ~compute:(fun () ->
             let flags = String.strip (arch_flags () ^ " " ^ simd_flags ()) in
             let has cond =
               probe_compiles ~flags
                 ~data:
                   (Printf.sprintf
                      "#if !(%s)\n#error \"no\"\n#endif\nint ocannl_vector_file_probe;\n" cond)
                 ()
             in
             if has "defined(__AVX512F__) && defined(__AVX512VL__)" then "32x64"
             else if has "defined(__AVX__)" then "16x32"
             else if has "defined(__aarch64__) || defined(_M_ARM64)" then "32x16"
             else "16x16")
           ()
       in
       Option.value (parse answer) ~default:(16, 16))
  in
  fun () -> Lazy.force probed

(* The vector register file of the explicit-SIMD renderings: the configured [vector_bytes_setting]
   width, and how many registers of that width the kernels have -- config [cc_vector_registers], or
   (auto) the native count, scaled down where the configured vector is wider than a native register
   (the compiler splits it across [vector_bytes / native] registers, so a 32-byte vector on NEON
   costs two of its 32 and the file holds 16 of them); 0 when explicit SIMD is off. One value
   (gh-ocannl-1399), read by [hardware_limits.simd_vector] in the seeding and
   [C_syntax_config.vector_file] in the renderer, where [Register_tile.rn_cap] fits the register
   tile's columns to the count. Not a function of the width alone: a 16-byte file is 32 registers on
   NEON and 16 on SSE, and keying the cap on the width gave SSE NEON's 4 x 6 tile, which spills
   there (gh-ocannl-1285). *)
let vector_file_setting () =
  let bytes = vector_bytes_setting () in
  let registers =
    match Int.of_string @@ Utils.get_global_arg ~default:"-1" ~arg_name:"cc_vector_registers" with
    | n when n > 256 ->
        (* No vector register file has more than 64 registers; a count past that is a typo, and the
           register-tile arithmetic over it would overflow. *)
        invalid_arg
          (Printf.sprintf "cc_vector_registers: expected -1 (auto) or a count from 0 to 256, got %d"
             n)
    | n when n >= 0 -> n
    | _ ->
        if bytes <= 0 then 0
        else
          let registers, native_bytes = native_vector_file () in
          registers * native_bytes / Int.max native_bytes bytes
  in
  { Ir.Vector_file.bytes; registers }

(* Whether the kernel .so is a macOS bundle or an ELF/PE shared object. Distinguishing the two BSDs
   from Darwin needs [uname], and this used to shell out once per kernel compile -- on a machine
   compiling thousands of kernels per build, an entirely redundant `sh -c` plus `uname` plus `grep`
   each time. The answer cannot change within a process. *)
(* [-Wl,-Bsymbolic] on ELF (gh-ocannl-754): a kernel's references to its OWN builtins bind to its
   own definitions. Without it, a [-fPIC] call to a global function goes through the PLT and the
   dynamic linker resolves it FIRST against the executable — and the OCaml executable carries a host
   copy of the builtins ([arrayjit/lib/builtins.c], the [ir] library's foreign stubs) under the
   same names. The two copies share their source text but not their compilation: the host copy is
   built without the kernel's flags, and a builtin whose return type depends on the target's fp16
   mode ([HALF_T]) comes back through an integer register from one and a float register from the
   other, so the kernel reads whatever [xmm0] held — a half [uniform1] draw read as [-0.0] on the
   Linux CI leg while every C-side call of the very same [.so] was correct. macOS has no such
   interposition (two-level namespaces bind a bundle's references to the bundle), which is why it
   passed there; Windows DLLs resolve per module too.

   [-lm] on ELF (gh-ocannl-1045): the kernel names its math library instead of borrowing the host
   process's. Scalar [expf] resolves against the libm the OCaml executable links, but under
   [__FAST_MATH__] glibc declares [expf], [logf], ... as SIMD-callable and gcc vectorizes those
   calls into libmvec entry points ([_ZGVdN8v_expf]) -- a library nothing in the process loads, so
   the kernel died at dlopen on an undefined symbol. glibc's [libm.so] is a linker script,
   [GROUP ( libm.so.6 AS_NEEDED ( libmvec.so.1 ) )], so the kernel gains a dependency on libmvec
   exactly when it references a vector variant. That [cc_backend_fast_math]'s
   [-fno-finite-math-only] currently keeps gcc from defining the macro is an accident of spelling,
   not a guard: [test/operations/fast_math_libmvec] forces it back. Not for Darwin, where libm is
   part of libSystem and [-undefined dynamic_lookup] resolves against it, nor for Windows. Link
   flags reach no codegen identity: the emitted code is the same. *)
let kernel_link_flags =
  lazy
    (match Sys.os_type with
    | "Unix" ->
        if Stdlib.Sys.command "uname -s | grep -q Darwin" = 0 then
          "-bundle -undefined dynamic_lookup"
        else "-shared -fPIC -Wl,-Bsymbolic -lm"
    | "Win32" | "Cygwin" -> "-shared"
    | _ -> "-shared -fPIC -Wl,-Bsymbolic -lm")

(* gh-ocannl-530 (docs/proposals/gh-ocannl-530-pool-uniformity.md): on hybrid CPUs, one pool mixing
   two core speeds costs the tuned schedules 20-31% -- chunked Grid loops end at a barrier, so the
   step is set by the slowest worker -- while restricting the pool to a uniform core class recovers
   25-34% of tuned time at no measured cost to the default (untuned) baselines. So [cc] restricts
   its worker pool to the highest-performance core class by default, on hybrid, native
   (non-virtualized) topologies; everything else -- libdispatch pools, fabricated guest topologies
   (WSL2), externally pinned processes, uniform machines -- is a conservative no-op. *)
let pool_core_class_of_string = function
  | "auto" -> Some `Auto
  | "all" -> Some `All
  | "performance" -> Some `Performance
  | "efficiency" -> Some `Efficiency
  | _ -> None

let pool_restriction =
  lazy
    (let open Utils.Cpu_topology in
     let setting_str =
       String.lowercase (Utils.get_global_arg ~default:"auto" ~arg_name:"cc_pool_core_class")
     in
     let setting =
       match pool_core_class_of_string setting_str with
       | Some s -> s
       | None ->
           invalid_arg
             ("cc_pool_core_class: expected auto | all | performance | efficiency, got "
            ^ setting_str)
     in
     let openmp =
       match parallel_grid_syntax_setting () with `Openmp -> true | `Dispatch | `None -> false
     in
     let effective = effective_cpu_count () in
     let affinity_mask = current_affinity_mask () in
     let decision =
       (* The class and hypervisor probes only matter where a restriction is possible at all; skip
          their syscalls when the outcome is structurally "keep". *)
       if (not openmp) || Poly.equal setting `All then
         unrestricted_decision ~effective ~affinity_mask
       else
         decide_pool_restriction ~openmp ~setting ~classes:(core_classes ())
           ~hypervisor:(hypervisor_present ()) ~effective ~affinity_mask
     in
     match decision.pool_restrict with
     | None -> decision
     | Some cls -> (
         match restrict_process_to_mask cls.mask with
         | Ok () -> decision
         | Error msg ->
             (* Degrade to the unrestricted pool; failing to pin must not fail the run. *)
             Stdlib.Printf.eprintf "OCANNL cc backend: cc_pool_core_class not applied: %s\n%!" msg;
             unrestricted_decision ~effective:(effective_cpu_count ())
               ~affinity_mask:(current_affinity_mask ())))

let effective_pool_width () = (Lazy.force pool_restriction).Utils.Cpu_topology.pool_width
let pool_tag () = (Lazy.force pool_restriction).Utils.Cpu_topology.pool_tag

let parallel_grid_chunks_setting () =
  match Int.of_string @@ Utils.get_global_arg ~default:"0" ~arg_name:"cc_parallel_chunks" with
  | n when n > 0 -> n
  (* Auto: a small multiple of the worker-pool width -- enough chunks that uneven per-chunk cost
     load-balances, few enough that per-chunk overhead stays negligible. The width is the
     pool-policy result (gh-ocannl-530), which is affinity-respecting -- unlike
     [Domain.recommended_domain_count], which on Windows reports the full machine under any pinning,
     silently mis-sizing the grid decomposition of a pinned run. *)
  | _ -> 4 * effective_pool_width ()

(* gh-ocannl-572: the codegen knobs of this backend, as one signature for the autotune disk-cache
   key. They are consulted when a kernel is rendered and compiled -- after the lowered code the
   canonical digest names -- so two processes differing only in them digest identically while
   emitting different kernels, which is precisely the shape gh-ocannl-568 measured at 5.9x when a
   tf32-tuned winner replayed under default numerics. [cc_vector_bytes=0] degrades a [Vectorized]
   winner to serial lanes, [cc_parallel_grid=none] a Grid-parallel one to a serial loop, and the
   compiler flags decide what the emitted C becomes; a schedule crowned under one such regime is a
   measurement from another machine's worth of evidence.

   Resolved values, not raw settings: "auto" means a different thing per machine, and the resolvers
   are already forced by any compile that reaches here (the pool tag alone forces the parallel-grid
   probe). The pool signature itself is a separate key component and is not repeated here. *)
(* The flags say what was ASKED for; this says what the toolchain will actually do with it (Codex
   P1 on PR #337). [cc_backend_arch_flags=auto] resolves to the spelling [-mcpu=native], and
   [compiler_command ()] usually to the word [cc] -- so two machines sharing a cache directory can
   hold identical keys while targeting different microarchitectures, or the same one through
   different compiler versions, and one machine's winner poisons the other's measurements. The
   compiler's own predefined macros under the configured flags fingerprint both at once
   ([__clang_major__], [__AVX2__], [__ARM_FEATURE_FP16_VECTOR_ARITHMETIC], ...): one subprocess,
   cached per machine like every other probe. A toolchain that cannot dump them degrades to the
   flag spelling, which is what the tag had before. *)
let pool_parallel_grid () =
  (not (Poly.equal (parallel_grid_syntax_setting ()) `None))
  && parallel_grid_chunks_setting () > 1
  && not (Utils.debug_log_from_routines ())

let target_fingerprint =
  lazy
    (cached_probe ~name:"target"
       ~compute:(fun () ->
         let src = Stdlib.Filename.temp_file "ocannl_cc_probe_" ".c" in
         let out = Stdlib.Filename.temp_file "ocannl_cc_probe_" ".txt" in
         let log = Stdlib.Filename.temp_file "ocannl_cc_probe_" ".log" in
         let flags = String.strip (arch_flags () ^ " " ^ simd_flags ()) in
         let dumped =
           try
             Stdio.Out_channel.write_all src ~data:"";
             let cmd =
               Printf.sprintf "%s %s -dM -E %s > %s 2> %s" (compiler_command ()) flags
                 (Stdlib.Filename.quote src) (Stdlib.Filename.quote out) (Stdlib.Filename.quote log)
             in
             Stdlib.Sys.command cmd = 0
           with _ -> false
         in
         let fingerprint =
           if not dumped then None
           else
             try
               let data = Stdio.In_channel.read_all out in
               (* An empty dump is a compiler that accepted the flags and printed nothing: no
                  fingerprint, not a fingerprint of nothing. *)
               if String.is_empty (String.strip data) then None
               else Some (String.prefix (Stdlib.Digest.to_hex (Stdlib.Digest.string data)) 16)
             with _ -> None
         in
         List.iter [ src; out; log ] ~f:(fun f -> try Stdlib.Sys.remove f with _ -> ());
         Option.value fingerprint ~default:"no-target-fingerprint")
       ())

let cpu_model =
  lazy
    (try
       if Stdlib.Sys.file_exists "/proc/cpuinfo" then
         Stdio.In_channel.read_lines "/proc/cpuinfo"
         |> List.filter ~f:(fun line ->
             String.is_prefix line ~prefix:"model name" || String.is_prefix line ~prefix:"Hardware")
         |> List.dedup_and_sort ~compare:String.compare
         |> String.concat ~sep:";"
       else if String.equal Stdlib.Sys.os_type "Win32" then
         Option.value (Stdlib.Sys.getenv_opt "PROCESSOR_IDENTIFIER") ~default:""
       else
         let ch =
           Unix.open_process_args_in "/usr/sbin/sysctl"
             [| "sysctl"; "-n"; "machdep.cpu.brand_string" |]
         in
         let value = Stdio.In_channel.input_all ch |> String.strip in
         match Unix.close_process_in ch with Unix.WEXITED 0 -> value | _ -> ""
     with Unix.Unix_error _ | Stdlib.Sys_error _ -> "")

let timing_identity () =
  try
    let model = Lazy.force cpu_model in
    if String.is_empty model then None
    else
      let host, os_build =
        match Lazy.force Utils.macos_timing_environment with
        | Some environment -> environment
        | None -> (Unix.gethostname (), None)
      in
      let target = Lazy.force target_fingerprint in
      let target = if String.equal target "no-target-fingerprint" then None else Some target in
      let executable = compiler_executable_identity (compiler_command ()) in
      let executable =
        if String.is_prefix executable ~prefix:"unresolved:" then None else Some executable
      in
      Some
        {
          Ir.Backend_intf.device_signature =
            Sexp.to_string ([%sexp_of: string * string] (host, model));
          toolchain_signature =
            (if Option.is_none target && Option.is_none executable && Option.is_none os_build then
               None
             else
               Some
                 (Sexp.to_string
                    ([%sexp_of: string option * string option * string option]
                       (target, executable, os_build))));
        }
  with Unix.Unix_error _ | Stdlib.Sys_error _ -> None

(* {2 Split compilation (gh-ocannl-1603)}

   A C compiler's cost grows faster than linearly in a function's size: the serial baseline a cc
   search compiles first is one function of every statement of the routine, and on a
   [gpt2_mini_train] step clang -O3 took 127 s over it, against 30 s for the same statements as 62
   [noinline] functions in one translation unit, and 8 s for those spread over 8 units compiled
   concurrently. So a large kernel is rendered OUTLINED
   ([C_syntax.C_syntax_config.outline_part_weight], [cc_kernel_part_weight]), and a large batch is
   compiled as up to [cc_compile_jobs] translation units of at least [cc_compile_unit_min_bytes]
   each, by as many compiler processes, linked into one library. Neither changes what a kernel
   computes. Windows compiles one unit: the concurrent compile is verified on the Unix toolchains
   only. *)
type split_settings = { part_weight : int; jobs : int; unit_min_bytes : int }

let non_negative_int ~key value =
  match Int.of_string_opt (String.strip value) with
  | Some n when n >= 0 -> n
  | _ ->
      raise
      @@ Utils.User_error (Printf.sprintf "%s must be a non-negative integer; found: %s" key value)

let split_override : split_settings option ref = ref None

(* Read apart from the other two: it is the one a cache key hashes ([codegen_tag]), and computing a
   key reads no key outside the [Keyed] class. *)
let part_weight_setting () =
  match !split_override with
  | Some split -> split.part_weight
  | None ->
      non_negative_int ~key:"cc_kernel_part_weight"
        (Utils.get_global_arg ~default:"256" ~arg_name:"cc_kernel_part_weight")

let split_settings () =
  match !split_override with
  | Some split -> split
  | None ->
      let jobs =
        match
          String.lowercase
            (String.strip (Utils.get_global_arg ~default:"auto" ~arg_name:"cc_compile_jobs"))
        with
        | "auto" -> Int.min 8 (Stdlib.Domain.recommended_domain_count ())
        | value -> non_negative_int ~key:"cc_compile_jobs" value
      in
      let unit_min_bytes =
        non_negative_int ~key:"cc_compile_unit_min_bytes"
          (Utils.get_global_arg ~default:"65536" ~arg_name:"cc_compile_unit_min_bytes")
      in
      let jobs = if String.equal Sys.os_type "Unix" then Int.max 1 jobs else 1 in
      { part_weight = part_weight_setting (); jobs; unit_min_bytes }

let with_split_settings split f =
  let saved = !split_override in
  split_override := Some split;
  Exn.protect ~f ~finally:(fun () -> split_override := saved)

module Translation_units = struct
  let count = Stdlib.Atomic.make 0
  let record n = ignore (Stdlib.Atomic.fetch_and_add count n : int)
end

let translation_units_compiled () = Stdlib.Atomic.get Translation_units.count

let codegen_tag () =
  let parts =
    [
      compiler_command ();
      Lazy.force target_fingerprint;
      Int.to_string (optimization_level ());
      Option.value (fast_math_flag ()) ~default:"fast-math-off";
      Option.value (fp_contract_flag ()) ~default:"fp-contract-default";
      arch_flags ();
      simd_flags ();
      Int.to_string (vector_file_setting ()).bytes;
      Int.to_string (vector_file_setting ()).registers;
      (match fp16_arithmetic_support () with
      | `None -> "fp16-none"
      | `Promoted -> "fp16-promoted"
      | `Native -> "fp16-native");
      (* gh-ocannl-1603: where a large kernel's function boundaries fall. *)
      "part-weight=" ^ Int.to_string (part_weight_setting ());
    ]
    (* The pool-parallel Grid rendering, and only when there IS one: with the grid syntax resolved
       to [`None], [C_syntax.collect_parallel_grid] returns before the chunk count or the
       privatization cap can reach the emitted code, so hashing them would retune identical serial
       kernels (Codex P2 on PR #337). *)
    @
    match parallel_grid_syntax_setting () with
    | `None -> [ "grid-none" ]
    | (`Dispatch | `Openmp) as mode ->
        let runtime_controls =
          match mode with
          | `Dispatch -> []
          | `Openmp ->
              (* The OpenMP runtime's own controls decide the team every Grid loop executes on, and
                 they move nothing the pool tag is derived from -- that is process affinity (Codex
                 P1 on PR #337). OMP_NUM_THREADS=1 against 16 is the difference between a serial and
                 a parallel machine, which is the whole ranking; OMP_STACKSIZE bounds what a
                 privatized candidate may put on a worker's stack, so a winner crowned under a
                 raised stack must not replay under the default one; the wait policy decides whether
                 idle workers spin, which is most of a small kernel's repeat cost; and the runtimes'
                 own affinity variables place the team on cores the process mask does not name. Read
                 only here: libdispatch reads none of them.

                 A complete list is not on offer -- an OpenMP runtime has dozens of variables, and
                 vendor ones keep arriving -- so this covers the standard team, stack, wait and
                 affinity controls plus libgomp's and the Intel runtime's common equivalents
                 (including [GOMP_STACKSIZE], which libgomp honors when [OMP_STACKSIZE] is unset:
                 Codex P1 on PR #337). An unlisted variable degrades to what everything did before
                 this component existed: a shared key across two timing regimes. What is
                 deliberately absent is [OMP_SCHEDULE]: the emitted pragma is [schedule(static)],
                 which the variable cannot reach, so keying on it would only retune identical
                 loops. *)
              List.map
                [
                  "OMP_NUM_THREADS";
                  "OMP_DYNAMIC";
                  "OMP_THREAD_LIMIT";
                  "OMP_PROC_BIND";
                  "OMP_PLACES";
                  "OMP_MAX_ACTIVE_LEVELS";
                  "OMP_STACKSIZE";
                  "OMP_WAIT_POLICY";
                  "GOMP_STACKSIZE";
                  "GOMP_CPU_AFFINITY";
                  "GOMP_SPINCOUNT";
                  "KMP_AFFINITY";
                  "KMP_BLOCKTIME";
                ] ~f:(fun var -> var ^ "=" ^ Option.value (Stdlib.Sys.getenv_opt var) ~default:"")
        in
        (match mode with `Dispatch -> "grid-dispatch" | `Openmp -> "grid-openmp")
        :: runtime_controls
        @ [
            Int.to_string (parallel_grid_chunks_setting ());
            Int.to_string (Lazy.force C_syntax.per_chunk_private_bytes_cap);
            "grid-min-updates=" ^ Int.to_string (C_syntax.grid_min_updates ());
            "mma-macs-per-update=" ^ Int.to_string C_syntax.tile_mma_macs_per_update;
          ]
  in
  String.prefix (Stdlib.Digest.to_hex (Stdlib.Digest.string (String.concat ~sep:"\000" parts))) 8

(* gh-ocannl-1371: which kind of host a run had, from the probes and settings the renderer itself
   reads, so a log can say it without anything restating the probe logic. The ISA is what those
   probes found under the kernel flags -- the AVX2/FMA stage of [simd_flags], [avx512_target], and
   [native_vector_file]'s AArch64 answer, which is NEON's register file -- so it names the target
   the kernels are compiled for, which with the default [-march=native] is the host. *)
let host_capabilities () =
  let registers, native_bytes = native_vector_file () in
  let isa =
    List.filter_opt
      [
        Option.some_if (String.is_substring (simd_flags ()) ~substring:"avx2") "avx2+fma";
        Option.some_if (avx512_target ()) "avx512f+bw+vl";
        Option.some_if (registers = 32 && native_bytes = 16) "neon";
      ]
  in
  (* [cpu_model] keeps /proc/cpuinfo's [model name\t: ...] lines whole, as the timing identity wants
     them; here only the values. *)
  let cpu =
    String.split (Lazy.force cpu_model) ~on:';'
    |> List.map ~f:(fun entry ->
        match String.lsplit2 entry ~on:':' with
        | Some (key, value)
          when String.is_prefix key ~prefix:"model name" || String.is_prefix key ~prefix:"Hardware"
          ->
            String.strip value
        | _ -> String.strip entry)
    |> String.concat ~sep:"; "
  in
  [
    ("cpu", cpu);
    ("cpus", Int.to_string (Utils.Cpu_topology.effective_cpu_count ()));
    ("pool", pool_tag ());
    ("isa", if List.is_empty isa then "none-probed" else String.concat ~sep:"," isa);
    ("cc_vector_bytes", Int.to_string (vector_bytes_setting ()));
    ("auto_vector_bytes", Int.to_string (auto_vector_bytes ()));
    ("cc_vector_registers", Int.to_string (vector_file_setting ()).registers);
    ("native_vector_file", Printf.sprintf "%dx%d" registers native_bytes);
    ( "fp16",
      match fp16_arithmetic_support () with
      | `None -> "none"
      | `Promoted -> "promoted"
      | `Native -> "native" );
    (* gh-ocannl-1038: which runtime the pool-parallel Grid loops render against -- the probe that
       LINKS a dispatch or OpenMP program, so a libomp installed or removed under an unchanged
       compiler moves this answer, and with it every parallel kernel. *)
    ( "parallel_grid",
      match parallel_grid_syntax_setting () with
      | `None -> "none"
      | `Dispatch -> "dispatch"
      | `Openmp -> "openmp" );
    ("compiler", compiler_command ());
    ("arch_flags", arch_flags ());
    ("simd_flags", simd_flags ());
    (* gh-ocannl-1038: the digest [hardware_limits] carries of everything codegen consults after
       lowering -- the OpenMP runtime's own controls among them, which no OCANNL setting names. With
       it every field of this backend's [hardware_limits] is a function of this list. *)
    ("codegen_tag", codegen_tag ());
  ]

module Tn = Tnode

type library = {
  lib : (Dl.library[@sexp.opaque]);
  libname : string;
  bytes : int;
      (** The library file's size as compiled, its weight in the kernel code cache (gh-ocannl-1525).
      *)
}
[@@deriving sexp_of]

type procedure = {
  bindings : Indexing.unit_bindings;
  name : string;
  result : library;
  symbol : string;
      (** What [result] exports this kernel as: [name] unless the kernel code cache answered with
          code compiled under another name (gh-ocannl-1525); [name] stays the one diagnostics use.
      *)
  kparams : (string * kparam_source) list;
}
[@@deriving sexp_of]

let compile_failure_detail ~exit_code ~source_note ~command ~compiler_output =
  Printf.sprintf
    "OCANNL cc backend: generated code failed to compile (exit code %d).\n\
     %s\n\
     Compilation command: %s\n\
     Compiler output:\n\
     %s"
    exit_code
    (C_syntax.codegen_bug_claim ~artifact:source_note)
    command compiler_output

(* The flags every kernel compile passes after the compiler command (output, link and log
   redirections aside), each paired with the setting it comes from. One definition, because
   [test/operations/bridge_arm_coverage] preprocesses the retained kernel sources under them to read
   which bridge arm each kernel compiled into (gh-ocannl-1145): a reading under any other flags
   would describe a target no kernel was built for; and because a failed compile names the setting
   whose flag the toolchain rejects (gh-ocannl-1350, [toolchain_fault]). *)
let kernel_compiler_flag_settings () =
  let optimization_flag = "-O" ^ Int.to_string (optimization_level ()) in
  let arch_flag = String.strip (arch_flags ()) in
  let simd_flag = String.strip (simd_flags ()) in
  (* [-fopenmp] must also reach the link step (this command compiles and links); harmless for
     kernels without parallel Grid loops. *)
  let parallel_flag =
    match parallel_grid_syntax_setting () with
    | `Openmp -> Some "-fopenmp"
    | `Dispatch | `None -> None
  in
  [
    Some ("cc_backend_optimization_level", optimization_flag);
    Option.some_if (not (String.is_empty arch_flag)) ("cc_backend_arch_flags", arch_flag);
    Option.some_if (not (String.is_empty simd_flag)) ("cc_backend_simd_flags", simd_flag);
    Option.map (fast_math_flag ()) ~f:(fun flag -> ("cc_backend_fast_math", flag));
    (* AFTER [-ffast-math], which itself sets contraction to fast: clang documents that the last of
       the two wins, so the explicit knob has to come last or [cc_backend_fp_contract=off] would
       silently do nothing in the one combination where it is load-bearing (Codex P2 on PR #291). *)
    Option.map (fp_contract_flag ()) ~f:(fun flag -> ("cc_backend_fp_contract", flag));
    Option.map parallel_flag ~f:(fun flag -> ("cc_parallel_grid", flag));
  ]
  |> List.filter_opt

let kernel_compiler_flags () =
  String.concat ~sep:" " (List.map (kernel_compiler_flag_settings ()) ~f:snd)

(* gh-ocannl-1350: a kernel compile can fail for a reason that has nothing to do with the generated
   code -- an explicit [cc_backend_arch_flags] the compiler does not know, a
   [cc_backend_compiler_command] that names no compiler, a broken local toolchain -- and the report
   would still call it a bug in OCANNL. So a failed compile first compiles a trivial kernel the same
   way: the same command, flags and link flags. If that fails too, the configuration is at fault,
   and narrowing it down costs a few more probes: the command alone with the link flags, then each
   setting's flag on its own. Asking the toolchain rather than reading its stderr keeps this
   independent of which compiler is installed and how it words its errors. Failures are the rare
   path, but an autotune search can fail many candidates, so the diagnosis is computed once per
   process per configuration. *)
type toolchain_probe = Probe_compiles | Probe_fails of { command : string; output : string }

let probe_kernel_toolchain ~flags =
  (* Every temporary is allocated inside the guard, and only the allocated ones are removed: a temp
     directory that fails here (full, gone) must leave the original rejection standing. *)
  let allocated = ref [] in
  let temp suffix =
    let path = Stdlib.Filename.temp_file "ocannl_cc_toolchain_" suffix in
    allocated := path :: !allocated;
    path
  in
  let result =
    try
      let src = temp ".c" in
      let lib = temp Utils.dynamic_library_extension in
      let log = temp ".log" in
      (* The command must write its output, not merely exit 0, so the placeholder goes first. *)
      Stdlib.Sys.remove lib;
      Stdio.Out_channel.write_all src ~data:"int ocannl_toolchain_probe(void) { return 0; }\n";
      let command =
        Printf.sprintf "%s %s %s -o %s %s" (compiler_command ()) (Stdlib.Filename.quote src) flags
          (Stdlib.Filename.quote lib) (Lazy.force kernel_link_flags)
      in
      if
        Stdlib.Sys.command (Printf.sprintf "%s > %s 2>&1" command (Stdlib.Filename.quote log)) = 0
        && Stdlib.Sys.file_exists lib
      then Some Probe_compiles
      else
        let output =
          try Stdio.In_channel.read_all log with _ -> "(unable to read probe output)"
        in
        Some (Probe_fails { command; output })
    with _ -> None
  in
  List.iter !allocated ~f:(fun f -> try Stdlib.Sys.remove f with _ -> ());
  result

type toolchain_fault = {
  culprits : (string * string) list;
      (** The settings the probes isolate, each with the value the compile passed for it. *)
  alone : bool;  (** Whether each culprit fails on its own, rather than only in combination. *)
  probe_command : string;
  probe_output : string;
}

let diagnose_toolchain () =
  let flag_settings = kernel_compiler_flag_settings () in
  let fails ~flags =
    match probe_kernel_toolchain ~flags with Some (Probe_fails _) -> true | _ -> false
  in
  (* A probe that could not even run (no temp directory) attributes nothing: the generated code
     keeps the blame, as it did before. *)
  match probe_kernel_toolchain ~flags:(kernel_compiler_flags ()) with
  | None | Some Probe_compiles -> None
  | Some (Probe_fails { command = probe_command; output = probe_output }) ->
      let culprits, alone =
        if fails ~flags:"" then
          let command =
            match Utils.get_global_arg ~default:"" ~arg_name:"cc_backend_compiler_command" with
            | "" -> compiler_command () ^ " (unset: resolved from `ocamlc -config`)"
            | command -> command
          in
          ([ ("cc_backend_compiler_command", command) ], true)
        else
          match List.filter flag_settings ~f:(fun (_, flag) -> fails ~flags:flag) with
          | [] -> (flag_settings, false)
          | culprits -> (culprits, true)
      in
      Some { culprits; alone; probe_command; probe_output }

let toolchain_fault =
  let mutex = Stdlib.Mutex.create () in
  let diagnoses = Hashtbl.create (module String) in
  fun () ->
    let key =
      String.concat ~sep:"\000"
        [
          compiler_command ();
          (* What the command resolves to: an executable replaced in place, or a PATH change, is
             another toolchain under the same spelling. *)
          compiler_executable_identity (compiler_command ());
          kernel_compiler_flags ();
          Lazy.force kernel_link_flags;
        ]
    in
    Stdlib.Mutex.protect mutex (fun () ->
        Hashtbl.find_or_add diagnoses key ~default:diagnose_toolchain)

let toolchain_failure_detail { culprits; alone; probe_command; probe_output } ~exit_code ~command
    ~compiler_output =
  let named =
    String.concat ~sep:", "
      (List.map culprits ~f:(fun (setting, value) -> Printf.sprintf "%s (%S)" setting value))
  in
  Printf.sprintf
    "OCANNL cc backend: the C toolchain configuration fails to compile even a trivial kernel, so \
     the generated code is not what failed (exit code %d).\n\
     %s: %s\n\
     Probe command: %s\n\
     Probe output:\n\
     %s\n\
     Kernel compilation command: %s\n\
     Kernel compiler output:\n\
     %s"
    exit_code
    (if alone then "The setting the toolchain rejects"
     else "No single setting fails alone; the toolchain rejects these together")
    named probe_command probe_output command compiler_output

let%track7_sexp c_compile_and_load ~name:(base_name : string) ~run_id ~f_paths =
  (* The pool restriction (gh-ocannl-530) must be in force before the first [-fopenmp] kernel is
     dlopened, not merely before its first parallel region: libgomp computes its default team size
     from the affinity mask in its ELF/PE constructor, which runs at dlopen. Forced unconditionally
     — the decision itself no-ops under [`Dispatch]/[`None], and re-reading the syntax setting here
     would re-scan argv per kernel compile just to guard an already-memoized force. *)
  ignore (Lazy.force pool_restriction : Utils.Cpu_topology.pool_decision);
  (* File ownership (gh-ocannl-1197). [f_path] is handed over by [compile_source]: a file that only
     the compiler reads -- a temporary, or under debug files a private copy in [build_files/] (see
     [compilation_copy]). This compilation also creates the compiler log and the library. The log is
     always removed. The source is removed once the library is loaded, and on every failure path too
     unless debug files are requested: then a failed compile keeps it, as the source its report
     names. The library is a temporary removed once it is loaded, unless a setting asks for it:
     library output or debug files keep it in [build_files/] (gh-ocannl-1247, gh-ocannl-1335), so a
     debug run leaves nothing in the system temp directory. The informational [build_files/] copy of
     the source is a debug output written elsewhere, never one of these. *)
  let retain_debug = Utils.settings.output_debug_files_in_build_directory in
  let library_requested =
    Utils.get_global_flag ~default:false ~arg_name:"output_dlls_in_build_directory"
  in
  let library_owned = not (retain_debug || library_requested) in
  (* A path that is already gone counts as removed: a retry could only hit whoever holds the freed
     name now. *)
  let removed path =
    try
      Unix.unlink path;
      true
    with
    | Unix.Unix_error (Unix.ENOENT, _, _) -> true
    | Unix.Unix_error _ -> false
  in
  let remove path = ignore (removed path : bool) in
  let libname =
    if library_owned then
      (* Named after its owner: where the removal below fails -- Windows, for a routine alive at
         exit -- a later process's startup removes it once this process is gone (gh-ocannl-1268). *)
      Stdlib.Filename.temp_file
        (Utils.temp_library_prefix ~pid:(Utils.os_process_id ()) ~nonce:(Utils.process_nonce ())
           ~run_id base_name)
        Utils.dynamic_library_extension
    else
      (* Kept beside the kernel's [build_files/] source, named apart per process as well as per
         compile, since concurrent processes of one executable share that directory
         (gh-ocannl-1247). *)
      Utils.kept_library_file ~run_id base_name
  in
  (try Stdlib.Sys.remove libname with _ -> ());
  let kernel_link_flags = Lazy.force kernel_link_flags in
  let source_note_of f_path =
    if retain_debug then "the generated .c file at " ^ f_path
    else
      "the generated .c file (this compile removed its temporary copy; rerun with \
       --ocannl_output_debug_files_in_build_directory=true to keep it)"
  in
  (* The compile: its exit code, the command a failure reports, the log holding its output, and the
     source a failure names. *)
  let _rc, _cmdline, temp_log, source_note =
    match f_paths with
    | [ f_path ] ->
        let temp_log = Stdlib.Filename.temp_file "ocannl_cc_" ".log" in
        let _cmdline : string =
          (* Quoted for the same reason as the probes': these paths sit under the build or the
             system temp directory, either of which can contain whitespace, and an unquoted one
             splits into arguments and fails every compile with a "this is a bug in OCANNL" report
             against a command the shell never saw whole. *)
          Printf.sprintf "%s %s %s -o %s %s > %s 2>&1" (compiler_command ())
            (Stdlib.Filename.quote f_path) (kernel_compiler_flags ())
            (Stdlib.Filename.quote libname) kernel_link_flags (Stdlib.Filename.quote temp_log)
        in
        (* Debug: log the command if debugging is enabled *)
        [%log3 "command", _cmdline];
        let _rc : int = Stdlib.Sys.command _cmdline in
        (* The compiler was the source's only reader: without debug files nothing else needs it. *)
        if not retain_debug then remove f_path;
        (_rc, _cmdline, temp_log, source_note_of f_path)
    | _ -> (
        (* gh-ocannl-1603: several translation units, each compiled to an object by its own compiler
           process, all at once, then linked into the one library. Everything a process needs is
           prepared before the first starts; the objects and the compile logs are this compilation's
           temporaries, removed on every path. *)
        let units =
          List.map f_paths ~f:(fun f_path ->
              let obj = Stdlib.Filename.temp_file "ocannl_cc_" ".o" in
              let log = Stdlib.Filename.temp_file "ocannl_cc_" ".log" in
              let command =
                Printf.sprintf "%s -c %s %s -fPIC -o %s > %s 2>&1" (compiler_command ())
                  (Stdlib.Filename.quote f_path) (kernel_compiler_flags ())
                  (Stdlib.Filename.quote obj) (Stdlib.Filename.quote log)
              in
              [%log3 "command", command];
              (f_path, obj, log, command))
        in
        let started =
          List.map units ~f:(fun (_, _, _, command) ->
              try Ok (Unix.open_process_out command) with exn -> Error exn)
        in
        let exits =
          List.map started ~f:(function
            | Error _ -> 127
            | Ok process -> (
                match Unix.close_process_out process with
                | Unix.WEXITED rc -> rc
                | Unix.WSIGNALED _ | Unix.WSTOPPED _ -> 255
                | exception Unix.Unix_error _ -> 255))
        in
        Translation_units.record (List.length units);
        if not retain_debug then List.iter units ~f:(fun (f_path, _, _, _) -> remove f_path);
        let remove_objects () = List.iter units ~f:(fun (_, obj, _, _) -> remove obj) in
        let failed = List.find (List.zip_exn units exits) ~f:(fun (_, rc) -> rc <> 0) in
        match failed with
        | Some ((f_path, _, log, command), rc) ->
            List.iter units ~f:(fun (_, _, other, _) ->
                if not (String.equal other log) then remove other);
            remove_objects ();
            (rc, command, log, source_note_of f_path)
        | None ->
            List.iter units ~f:(fun (_, _, log, _) -> remove log);
            let temp_log = Stdlib.Filename.temp_file "ocannl_cc_" ".log" in
            let command =
              Printf.sprintf "%s %s %s -o %s %s > %s 2>&1" (compiler_command ())
                (String.concat ~sep:" "
                   (List.map units ~f:(fun (_, obj, _, _) -> Stdlib.Filename.quote obj)))
                (kernel_compiler_flags ()) (Stdlib.Filename.quote libname) kernel_link_flags
                (Stdlib.Filename.quote temp_log)
            in
            [%log3 "command", command];
            let rc = Stdlib.Sys.command command in
            remove_objects ();
            (rc, command, temp_log, source_note_of (List.hd_exn f_paths)))
  in
  if _rc <> 0 then (
    let compiler_output =
      try Stdio.In_channel.read_all temp_log with _ -> "(unable to read compiler output)"
    in
    remove temp_log;
    (* A compiler can exit non-zero after writing part of its output. *)
    if library_owned then remove libname;
    (* gh-ocannl-1350: before blaming the generated code, ask whether this toolchain configuration
       compiles anything at all. *)
    match toolchain_fault () with
    | Some fault ->
        raise
          (Schedule_outcome.Cause_at
             ( Schedule_outcome.Backend_compile,
               Schedule_outcome.Backend_rejected
                 {
                   backend = name;
                   stage = Schedule_outcome.toolchain_stage;
                   severity = Schedule_outcome.Configuration;
                   detail =
                     toolchain_failure_detail fault ~exit_code:_rc ~command:_cmdline
                       ~compiler_output;
                 } ))
    | None ->
        (* A header the compiler could not open is the toolchain's fault, and fails every kernel
           alike: it propagates untyped rather than as a decline of this kernel
           (ahrefs/ocannl#1387). Asked only after the probe, which can name a setting that injects
           the header (an [-include] in the flags); a header only the generated code includes is
           this case. *)
        if C_syntax.missing_include compiler_output then
          failwith
            (Printf.sprintf
               "OCANNL cc backend: the compiler could not open a header (exit code %d); check the \
                C compiler installation.\n\
                Compilation command: %s\n\
                Compiler output:\n\
                %s"
               _rc _cmdline compiler_output);
        let detail =
          compile_failure_detail ~exit_code:_rc ~source_note ~command:_cmdline ~compiler_output
        in
        raise
          (Schedule_outcome.Cause_at
             ( Schedule_outcome.Backend_compile,
               Schedule_outcome.Backend_rejected
                 {
                   backend = name;
                   stage = "compiler";
                   severity = Schedule_outcome.Compiler_bug;
                   detail;
                 } )))
  else remove temp_log;
  (* All three post-compile host-boundary failures are typed, uncontainable link rejections. *)
  let reject_link stage detail =
    (* Cleanup must not replace the typed rejection: [remove] swallows its own failure. *)
    if library_owned then remove libname;
    raise
      (Schedule_outcome.Cause_at
         ( Schedule_outcome.Backend_link,
           Schedule_outcome.Backend_rejected
             { backend = name; stage; severity = Schedule_outcome.Compiler_bug; detail } ))
  in
  (* A successful compiler must produce its artifact. A missing library says nothing about the
     candidate's schedule, so keep it fatal under permissive classification (gh-ocannl-1142). Wait a
     moment for the file to be fully written on success. *)
  let start_time = Unix.gettimeofday () in
  let timeout =
    Float.of_string
    @@ Utils.get_global_arg ~default:"10.0" ~arg_name:"cc_backend_post_compile_timeout"
  in
  while not (Stdlib.Sys.file_exists libname) do
    let elapsed = Unix.gettimeofday () -. start_time in
    if Float.(elapsed > timeout) then
      reject_link Schedule_outcome.artifact_missing_stage
        (Printf.sprintf
           "Cc_backend.c_compile_and_load: compiled library %s not found after successful \
            compilation (timeout %g seconds).\n\
            Compilation command: %s"
           libname timeout _cmdline);
    Unix.sleepf 0.001
  done;
  (* Signing is a host/toolchain prerequisite, not a schedule constraint (gh-ocannl-1142). Expected
     to succeed on MacOS only. *)
  let verify_codesign =
    Utils.get_global_flag ~default:false ~arg_name:"cc_backend_verify_codesign"
  in
  (if verify_codesign then
     let null_device = if Sys.win32 then "nul" else "/dev/null" in
     let command =
       Printf.sprintf "codesign -s - %s > %s 2>&1" (Stdlib.Filename.quote libname) null_device
     in
     let rc = Stdlib.Sys.command command in
     if rc <> 0 then
       reject_link Schedule_outcome.codesign_stage
         (Printf.sprintf
            "Cc_backend.c_compile_and_load: codesign failed with exit code %d for library %s\n\
             Codesign command: %s\n\
             Compilation command: %s"
            rc libname command _cmdline));
  (* Note: RTLD_DEEPBIND not available on MacOS. A load failure is typed at [Backend_link]
     (gh-ocannl-1077): the object compiled, so what the loader refuses — typically a symbol neither
     the link line nor the process supplies — is OCANNL's link bug, and [Schedule_outcome.protect]
     makes it fatal rather than a candidate decline. *)
  let lib =
    try Dl.dlopen ~filename:libname ~flags:[ RTLD_NOW ]
    with Dl.DL_error dlerror ->
      let detail =
        Printf.sprintf
          "OCANNL cc backend: the compiled kernel failed to load (dlopen).\n\
           This is a bug in OCANNL: the kernel references a symbol that neither its link line nor \
           the process supplies. Please file an issue with %s\n\
           dlerror: %s\n\
           Compilation command: %s"
          source_note dlerror _cmdline
      in
      reject_link Schedule_outcome.dlopen_stage detail
  in
  (* Loaded: a debug run's private source copy has served its purpose, the informational
     [build_files/] copy being the one a reader looks at. *)
  if retain_debug then List.iter f_paths ~f:remove;
  let bytes = try (Unix.stat libname).st_size with Unix.Unix_error _ -> 0 in
  let result = { lib; libname; bytes } in
  (* The kernel now executes from the mapping, not from the path. POSIX unlinks a mapped file's name
     and keeps the mapping alive; Windows refuses to delete a mapped DLL, and then the removal waits
     for the unload below, if one happens before exit. Either outcome is safe, so the code does not
     ask which platform it is on. Only a removal that failed is retried: once the name is gone,
     another process's [temp_file] may own it. *)
  let removal_pending = library_owned && not (removed libname) in
  Alloc_census.count_module_loaded ();
  (* gh-ocannl-550: counted here, next to the unload the OpenMP arm deliberately does not perform,
     so the census reports the mapping as live for as long as it really is. *)
  (match parallel_grid_syntax_setting () with
  | `Openmp ->
      (* Never dlclose kernels built with -fopenmp: unloading an object whose parallel regions
         executed is a documented GOMP restriction -- libgomp's pool threads can retain references
         into it, and the dlclose can drop libgomp itself (loaded only as this object's dependency)
         under its parked workers. Observed as a SIGSEGV shortly after a routine was collected
         (ubuntu CI, PR #97). Leak the mapping instead; kernels are small. A Windows library whose
         removal is pending therefore stays on disk until a later process's startup sweeps it
         (gh-ocannl-1268): nothing unmaps it before exit. *)
      ()
  | `Dispatch | `None ->
      let%track7_sexp finalize (lib : library) : unit =
        Dl.dlclose ~handle:lib.lib;
        (* After the unload, never before: Windows deletes a DLL only once nothing maps it. OCaml
           runs no finalisers at exit, so on Windows only a routine collected before exit gets its
           library removed here; the rest wait for a later process's startup sweep
           (gh-ocannl-1268). *)
        if removal_pending then remove lib.libname;
        Alloc_census.count_module_unloaded ()
      in
      Stdlib.Gc.finalise finalize result);
  result

module CC_syntax_config (Procs : sig
  val procs : Low_level.t array
end) =
struct
  include C_syntax.Pure_C_config (struct
    let procs = Procs.procs
    let full_printf_support = C_syntax.printf_support_unless_uniform ()
  end)

  (* The prelude's linkage macro and the outlined parts' attribute (gh-ocannl-1603) are macros: a
     node or routine spelled like one would expand. *)
  let ident_blacklist =
    ident_blacklist
    @ C_syntax.builtin_idents Builtins_cc.builtins
    @ [ "ARRAYJIT_BUILTIN_LINKAGE"; Kernel_outline.part_attribute ]

  let parallel_grid_syntax = parallel_grid_syntax_setting ()
  let parallel_grid_chunks = parallel_grid_chunks_setting ()

  (* gh-ocannl-1669: a CPU thread's stack is small and fixed, its heap is not. *)
  let heap_local_arrays = true
  let vector_file = vector_file_setting ()
  let outline_part_weight () = match part_weight_setting () with 0 -> None | weight -> Some weight

  (* gh-ocannl-517: a CPU has no 16-bit arithmetic -- the narrow arms of the operator renderings
     below are widen/op/narrow round-trips through f32, and [_Float16] arithmetic, where the type
     exists at all, is promoted to float by the compiler on targets without AVX512-FP16. So the
     arithmetic runs in f32 and the narrow format is a storage format: one widen per load, one
     narrow per store. Under [narrow_compute_f32 = false] the narrow arms below take over again and
     every operator rounds to the target's storage precision (the pre-gh-517 semantics). *)
  (* gh-ocannl-516: fp16 is the one narrow format a CPU can execute natively, and where it does,
     computing in it doubles the lane count against f32. Opt-in, and only where the arithmetic is
     genuinely 16-bit -- on a target that merely promotes to float there is no throughput to win,
     so the policy is ignored rather than silently costing mantissa. The resolution itself lives in
     [Numerics.cpu_compute_prec] so autotune's seeding shares it verbatim (gh-ocannl-575). *)
  let compute_prec prec =
    Numerics.cpu_compute_prec ~native_fp16_arithmetic:(has_native_fp16_arithmetic ()) prec

  (* On CPU a reduction accumulator is an assignment intermediate like any other: it resides at the
     compute precision (gh-ocannl-639) — except under [Fp16_wide], whose contract is f32 f16
     residency on every backend unconditionally, [narrow_compute_f32 = false] included
     (gh-ocannl-680). Restated next to the [compute_prec] override because the pair binds at
     [include] time (see the signature's coupling note, gh-ocannl-663); the resolution itself lives
     in [Numerics.cpu_accum_prec] so autotune's seeding shares it verbatim, exactly as
     [compute_prec] shares [cpu_compute_prec]. *)
  let accum_prec prec =
    Numerics.cpu_accum_prec ~native_fp16_arithmetic:(has_native_fp16_arithmetic ()) prec

  (* The explicit vector renderings work at the compute precision, so admitting fp16 here is
     admitting native 16-bit vector arithmetic -- [vec_ext_typ] mints a [HALF_T] vector and the lane
     count doubles.

     The condition is the target's, not the policy's, and the two are not the same question.
     Arriving here at [Half_prec] does not mean [fp16_arithmetic] chose it: [narrow_compute_f32 =
     false] also leaves half alone, on any target, and there [HALF_T] is [uint16_t] -- a vector of
     those would do integer arithmetic on raw half bit patterns and quietly corrupt the loop. So ask
     the probe directly. Declining on a merely [`Promoted] target costs nothing that existed before
     gh-ocannl-516, when half never vectorized at all. *)
  let vector_prec_ok prec =
    match prec with
    | Ops.Single_prec _ | Ops.Double_prec _ -> true
    | Ops.Half_prec _ -> has_native_fp16_arithmetic ()
    | _ -> false

  (* Override operation syntax to handle special precision types *)
  let ternop_syntax prec op v1 v2 v3 =
    match (prec, op) with
    (* gh-ocannl-516: at fp16 compute precision the FMA goes through the shared macro, so the scalar
       path and the vector rendering's per-lane fallback are the same operation -- see
       [C_syntax.vec_acc_fma]. *)
    | Ops.Half_prec _, Ops.FMA ->
        let open PPrint in
        group
          (string "OCANNL_HALF_FMA(" ^^ v1 ^^ string ","
          ^^ ifflat (space ^^ v2) (nest 2 (break 1 ^^ v2))
          ^^ string ","
          ^^ ifflat (space ^^ v3) (nest 2 (break 1 ^^ v3))
          ^^ string ")")
    | _ -> (
        match prec with
        | Ops.Bfloat16_prec _ ->
            (* For BFloat16, perform operations in float precision *)
            let open PPrint in
            let float_v1 = string "bfloat16_to_single(" ^^ v1 ^^ string ")" in
            let float_v2 = string "bfloat16_to_single(" ^^ v2 ^^ string ")" in
            let float_v3 = string "bfloat16_to_single(" ^^ v3 ^^ string ")" in
            let op_prefix, op_infix1, op_infix2, op_suffix = Ops.ternop_c_syntax Ops.single op in
            let float_result =
              group
                (string op_prefix ^^ float_v1 ^^ string op_infix1
                ^^ ifflat (space ^^ float_v2) (nest 2 (break 1 ^^ float_v2))
                ^^ string op_infix2
                ^^ ifflat (space ^^ float_v3) (nest 2 (break 1 ^^ float_v3))
                ^^ string op_suffix)
            in
            string "single_to_bfloat16(" ^^ float_result ^^ string ")"
        | Ops.Half_prec _ ->
            (* For Half, perform operations in float precision on non-native systems *)
            let open PPrint in
            let float_v1 = string "HALF_TO_FP(" ^^ v1 ^^ string ")" in
            let float_v2 = string "HALF_TO_FP(" ^^ v2 ^^ string ")" in
            let float_v3 = string "HALF_TO_FP(" ^^ v3 ^^ string ")" in
            let op_prefix, op_infix1, op_infix2, op_suffix = Ops.ternop_c_syntax Ops.single op in
            let float_result =
              group
                (string op_prefix ^^ float_v1 ^^ string op_infix1
                ^^ ifflat (space ^^ float_v2) (nest 2 (break 1 ^^ float_v2))
                ^^ string op_infix2
                ^^ ifflat (space ^^ float_v3) (nest 2 (break 1 ^^ float_v3))
                ^^ string op_suffix)
            in
            (* [FLOAT_TO_HALF], not [FP_TO_HALF]: the latter is the *identity* on a native
               [_Float16] target, which would leave the f32 result of a library call ([expf],
               [sqrtf], [fmaxf]) at f32 -- and C's usual arithmetic conversions then keep every
               enclosing operator at f32 too, all the way to the store. The fp16-arithmetic policy
               promises fp16 intermediates, 10-bit mantissa and a 65504 ceiling included, so the
               narrowing has to be a real cast (gh-ocannl-516 review). A no-op where the ring
               operators already compute in [_Float16], and unchanged on the emulated path, where
               both macros are [float_to_half_emulated]. Same reasoning at the two [FLOAT_TO_HALF]
               sites below. *)
            string "FLOAT_TO_HALF(" ^^ float_result ^^ string ")"
        | Ops.Fp8_prec _ ->
            (* For FP8, perform operations in float precision *)
            let open PPrint in
            let float_v1 = string "fp8_to_single(" ^^ v1 ^^ string ")" in
            let float_v2 = string "fp8_to_single(" ^^ v2 ^^ string ")" in
            let float_v3 = string "fp8_to_single(" ^^ v3 ^^ string ")" in
            let op_prefix, op_infix1, op_infix2, op_suffix = Ops.ternop_c_syntax Ops.single op in
            let float_result =
              group
                (string op_prefix ^^ float_v1 ^^ string op_infix1
                ^^ ifflat (space ^^ float_v2) (nest 2 (break 1 ^^ float_v2))
                ^^ string op_infix2
                ^^ ifflat (space ^^ float_v3) (nest 2 (break 1 ^^ float_v3))
                ^^ string op_suffix)
            in
            string "single_to_fp8(" ^^ float_result ^^ string ")"
        | _ ->
            let op_prefix, op_infix1, op_infix2, op_suffix = Ops.ternop_c_syntax prec op in
            let open PPrint in
            group
              (string op_prefix ^^ v1 ^^ string op_infix1
              ^^ ifflat (space ^^ v2) (nest 2 (break 1 ^^ v2))
              ^^ string op_infix2
              ^^ ifflat (space ^^ v3) (nest 2 (break 1 ^^ v3))
              ^^ string op_suffix))

  let binop_syntax prec op v1 v2 =
    match op with
    | (Ops.Threefry4x32_crypto | Ops.Threefry4x32_light | Ops.Uint4x32_to_prec_uniform_lane) as op
      ->
        let call fn v1 v2 =
          let open PPrint in
          group (string (fn ^ "(") ^^ v1 ^^ string ", " ^^ v2 ^^ string ")")
        in
        C_syntax.rng_binop_syntax ~backend:"CC" ~call prec op v1 v2
    | _ -> (
        match prec with
        | Ops.Bfloat16_prec _ ->
            (* For BFloat16, perform all operations in float precision *)
            let open PPrint in
            let float_v1 = string "bfloat16_to_single(" ^^ v1 ^^ string ")" in
            let float_v2 = string "bfloat16_to_single(" ^^ v2 ^^ string ")" in
            let op_prefix, op_infix, op_suffix = Ops.binop_c_syntax Ops.single op in
            let float_result =
              group
                (string op_prefix ^^ float_v1 ^^ string op_infix
                ^^ ifflat (space ^^ float_v2) (nest 2 (break 1 ^^ float_v2))
                ^^ string op_suffix)
            in
            string "single_to_bfloat16(" ^^ float_result ^^ string ")"
        | Ops.Fp8_prec _ ->
            (* For FP8, perform all operations in float precision *)
            let open PPrint in
            let float_v1 = string "fp8_to_single(" ^^ v1 ^^ string ")" in
            let float_v2 = string "fp8_to_single(" ^^ v2 ^^ string ")" in
            let op_prefix, op_infix, op_suffix = Ops.binop_c_syntax Ops.single op in
            let float_result =
              group
                (string op_prefix ^^ float_v1 ^^ string op_infix
                ^^ ifflat (space ^^ float_v2) (nest 2 (break 1 ^^ float_v2))
                ^^ string op_suffix)
            in
            string "single_to_fp8(" ^^ float_result ^^ string ")"
        | Ops.Half_prec _ ->
            (* For Half, perform all operations in float precision on non-native systems *)
            let open PPrint in
            let float_v1 = string "HALF_TO_FP(" ^^ v1 ^^ string ")" in
            let float_v2 = string "HALF_TO_FP(" ^^ v2 ^^ string ")" in
            let op_prefix, op_infix, op_suffix = Ops.binop_c_syntax Ops.single op in
            let float_result =
              group
                (string op_prefix ^^ float_v1 ^^ string op_infix
                ^^ ifflat (space ^^ float_v2) (nest 2 (break 1 ^^ float_v2))
                ^^ string op_suffix)
            in
            string "FLOAT_TO_HALF(" ^^ float_result ^^ string ")"
        | _ ->
            let op_prefix, op_infix, op_suffix = Ops.binop_c_syntax prec op in
            let open PPrint in
            group
              (string op_prefix ^^ v1 ^^ string op_infix
              ^^ ifflat (space ^^ v2) (nest 2 (break 1 ^^ v2))
              ^^ string op_suffix))

  (* gh-ocannl-1622: a value select renders as a GNU statement expression that evaluates both arms
     into [__auto_type] locals before selecting between them. The arms are then plain values, so the
     compiler if-converts ([fcsel] / [bsl] on AArch64, [cmov] / blends on x86) instead of branching
     around an arm's load, as it must for a short-circuiting [?:]. The locals take each arm's own C
     type and the final [?:] applies the usual arithmetic conversions to them, so the blend has
     EXACTLY the type and value of the [?:] spelling it replaces -- a gate's double [0.0] literal
     included, which an f32 helper function would round away (review round 1). It is a true select,
     not a multiply by a mask: NaN, infinities and signed zeros pass unchanged. Inner blends shadow
     the names in their own block, and no arm can mention them. Only at the two compute precisions
     whose [?:] spellings above are the plain [Ops] ones; the bridged narrow precisions keep
     [?:]. *)
  let select_blend_syntax prec =
    let blend truth a b =
      let open PPrint in
      group
        (string "({ __auto_type ocannl_sel_a__ ="
        ^^ nest 2 (break 1 ^^ a)
        ^^ string ";" ^^ break 1
        ^^ string "__auto_type ocannl_sel_b__ ="
        ^^ nest 2 (break 1 ^^ b)
        ^^ string ";" ^^ break 1 ^^ string "(" ^^ truth
        ^^ string ") ? ocannl_sel_a__ : ocannl_sel_b__; })")
    in
    match prec with Ops.Single_prec _ | Ops.Double_prec _ -> Some blend | _ -> None

  let unop_syntax prec op v =
    match op with
    | Ops.Uint4x32_to_prec_uniform1 ->
        (* Heterogeneous op: the argument is uint4x32 whatever the result precision, so the
           per-precision float-bridging below must not wrap it (e.g. [fp8_to_single] on a uint4x32_t
           would not even compile); the builtin already returns the result precision's storage
           type. *)
        let op_prefix, op_suffix = Ops.unop_c_syntax prec op in
        let open PPrint in
        group (string op_prefix ^^ v ^^ string op_suffix)
    | _ -> (
        match prec with
        | Ops.Bfloat16_prec _ ->
            (* For BFloat16, perform operations in float precision *)
            let open PPrint in
            let float_v = string "bfloat16_to_single(" ^^ v ^^ string ")" in
            let op_prefix, op_suffix = Ops.unop_c_syntax Ops.single op in
            let float_result = group (string op_prefix ^^ float_v ^^ string op_suffix) in
            string "single_to_bfloat16(" ^^ float_result ^^ string ")"
        | Ops.Fp8_prec _ ->
            (* For FP8, perform operations in float precision *)
            let open PPrint in
            let float_v = string "fp8_to_single(" ^^ v ^^ string ")" in
            let op_prefix, op_suffix = Ops.unop_c_syntax Ops.single op in
            let float_result = group (string op_prefix ^^ float_v ^^ string op_suffix) in
            string "single_to_fp8(" ^^ float_result ^^ string ")"
        | Ops.Half_prec _ ->
            (* For Half, perform operations in float precision on non-native systems *)
            let open PPrint in
            let float_v = string "HALF_TO_FP(" ^^ v ^^ string ")" in
            let op_prefix, op_suffix = Ops.unop_c_syntax Ops.single op in
            let float_result = group (string op_prefix ^^ float_v ^^ string op_suffix) in
            string "FLOAT_TO_HALF(" ^^ float_result ^^ string ")"
        | _ ->
            let op_prefix, op_suffix = Ops.unop_c_syntax prec op in
            let open PPrint in
            group (string op_prefix ^^ v ^^ string op_suffix))
end

let codegen_capabilities () =
  let module Config = CC_syntax_config (struct
    let procs = [||]
  end) in
  C_syntax.codegen_capabilities (module Config)

(* Under [output_debug_files_in_build_directory] the generated source lives at the predictable
   shared path [build_files/<name>.c], which a concurrently running process compiling a same-named
   kernel can rewrite while our C compiler reads it — macOS CI hit this as an "extraneous closing
   brace" in a torn sum_serial.c (two tests, one base name). Compile from a private copy, named
   after this process and this compile beside the kept library ([Utils.owned_artifact_file]), so
   that a failed compile can keep it as the exact source its report names, and the debug run leaves
   nothing in the system temp directory (gh-ocannl-1335); the [build_files/] copy stays purely
   informational. Without debug files, [open_build_file] already returned a unique temp path — use
   it directly. *)
let compilation_copy ~name ~run_id (build_file : Utils.build_file_channel) filtered_code =
  if Utils.settings.output_debug_files_in_build_directory then (
    let copy = Utils.owned_artifact_file ~run_id ~extension:".c" name in
    Stdio.Out_channel.write_all copy ~data:filtered_code;
    copy)
  else build_file.f_path

module Compile = C_syntax.Compile_driver (CC_syntax_config)

(* [sources]: one translation unit, compiled as [<name>.c]; or several (gh-ocannl-1603), compiled as
   [<name>.tu<k>.c] and linked into one library. *)
let compile_source ~name sources =
  (* The run id makes every library path unique within the process, even once the file is gone. A
     dynamic loader (glibc, dyld, LoadLibrary) answers a [dlopen] of a path it already has mapped
     with that mapping, without reading the file; the library is unlinked while still mapped and
     [Dl.dlclose] need not unload it, so a later [temp_file] landing on the freed name would
     otherwise run the earlier kernel. *)
  let run_id = Utils.get_global_run_id () in
  let write ~name source =
    let build_file = Utils.open_build_file ~base_name:name ~extension:".c" in
    Out_channel.output_string build_file.oc source;
    build_file.finalize ();
    compilation_copy ~name ~run_id build_file source
  in
  let f_paths =
    match sources with
    | [ source ] -> [ write ~name source ]
    | _ ->
        List.mapi sources ~f:(fun k source -> write ~name:(name ^ ".tu" ^ Int.to_string k) source)
  in
  c_compile_and_load ~name ~run_id ~f_paths

(* {2 The kernel code cache (gh-ocannl-1525)}

   The C compiler's cost is per kernel, and large: a [gpt2_mini_train] search candidate of 16
   kernels spent 10-37 s in it on mac-studio, so a cc search could time a handful of candidates in a
   budget. As on HIP and CUDA ([Kernel_code_cache], gh-ocannl-1314/1529), a batch compiles only the
   kernels no earlier compile in this process produced -- its misses, one library of them per batch
   -- and reuses the rest. A kernel's key is its canonical standalone source ([render_batch]'s
   [key_text]: the names it owns renamed, its segment index dropped) plus the toolchain it compiles
   under: the compiler command, the executable it resolves to, the kernel compiler flags and the
   link flags, the same identity the toolchain diagnosis keys on. A reused kernel is looked up in
   its library by the symbol it was compiled under. Process-local, bounded by [cc_kernel_cache_mib]
   (library file bytes); 0 disables it, and every batch compiles as one library of all its kernels,
   as before. Bypassed under device debugging, as on the GPUs. A library the cache holds stays
   loaded while it does; under OpenMP a library is never unloaded anyway. *)
let kernel_cache_capacity_bytes =
  lazy
    (let mib = Utils.get_global_arg ~default:"256" ~arg_name:"cc_kernel_cache_mib" in
     match Int.of_string_opt (String.strip mib) with
     | Some m when m >= 0 -> m * 1024 * 1024
     | _ ->
         raise
         @@ Utils.User_error
              ("cc_kernel_cache_mib must be a non-negative integer (MiB); found: " ^ mib))

let kernel_cache : (library * string) Kernel_code_cache.t =
  Kernel_code_cache.create ~capacity_bytes:(fun () -> Lazy.force kernel_cache_capacity_bytes)

(* Held from the lookups of a batch's misses to their publication, so two domains compiling the same
   kernels compile them once. *)
let kernel_compile_lock = Stdlib.Mutex.create ()

let write_source_artifact ~name source =
  if Utils.settings.output_debug_files_in_build_directory then (
    let build_file = Utils.open_build_file ~base_name:name ~extension:".c" in
    Out_channel.output_string build_file.oc source;
    build_file.finalize ())

let compile_kernels ~batch ~names bindings (lowereds : Low_level.optimized array) : procedure array
    =
  Compile_cost.record_reuse_capable ();
  let r =
    Compile.render_batch ~names bindings lowereds ~includes:Builtins_cc.includes
      ~builtins:Builtins_cc.builtins ()
  in
  let n = Array.length r.entries in
  Compile_cost.record_kernels ~rendered:n ~compiled:0;
  let all = List.init n ~f:Fn.id in
  let kernel_names = Array.map r.entries ~f:(fun (_, name, _) -> name) in
  (* The artifacts' name, as [Compile_driver.drive] derives it. *)
  let name =
    if batch then
      String.(strip ~drop:(equal_char '_') @@ common_prefix (Array.to_list kernel_names))
    else kernel_names.(0)
  in
  (* Not under device debugging, as on the GPUs, nor when the caller asked for the routine's library
     as an artifact ([output_dlls_in_build_directory]): what is kept is then the routine's own whole
     library, compiled as before. *)
  let enabled =
    Kernel_code_cache.enabled kernel_cache
    && (not (Utils.with_runtime_debug ()))
    && not (Utils.get_global_flag ~default:false ~arg_name:"output_dlls_in_build_directory")
  in
  (* One library of [idxs]. A library of every kernel in batch order compiles the routine's own
     source as [<name>.c]; any other is compiled as [<name>.part<k>.c], and the routine's full
     source is still written as [<name>.c] (what tests read through [Generated] does not depend on
     what the cache held). [.part<k>] is a spelling no routine name takes. *)
  let compile_group ~group ~n_groups idxs =
    let whole = n_groups = 1 && List.equal Int.equal idxs all in
    let artifact = if whole then name else name ^ ".part" ^ Int.to_string group in
    (* Before the call: a failed compile was still given these kernels. *)
    Compile_cost.record_kernels ~rendered:0 ~compiled:(List.length idxs);
    let lib =
      let { part_weight = _; jobs; unit_min_bytes } = split_settings () in
      let sources =
        if jobs <= 1 then [ r.assemble idxs ]
        else r.assemble_units idxs ~max_units:jobs ~min_unit_bytes:unit_min_bytes
      in
      (* Several units: the routine's full source is still [<name>.c] (read through [Generated]). *)
      if List.length sources > 1 && whole then write_source_artifact ~name (r.assemble all);
      try Compile_cost.time_device (fun () -> compile_source ~name:artifact sources)
      with exn ->
        let backtrace = Stdlib.Printexc.get_raw_backtrace () in
        r.attribute idxs exn backtrace
    in
    (lib, lib.bytes)
  in
  let resolved =
    if not enabled then
      let lib, _ = compile_group ~group:0 ~n_groups:1 all in
      Array.map kernel_names ~f:(fun symbol -> (lib, symbol))
    else
      let toolchain =
        [
          compiler_command ();
          compiler_executable_identity (compiler_command ());
          kernel_compiler_flags ();
          Lazy.force kernel_link_flags;
        ]
      in
      let keys =
        Array.init n ~f:(fun i -> Kernel_code_cache.key ~parts:(toolchain @ [ r.key_text i ]))
      in
      let resolved =
        Kernel_code_cache.resolve kernel_cache ~lock:kernel_compile_lock ~keys
          ~options:(Array.create ~len:n toolchain) ~symbols:kernel_names
          ~compile_group:(fun ~group ~n_groups _ idxs -> compile_group ~group ~n_groups idxs)
          ()
      in
      (* The routine's own full source as [<name>.c], whatever was compiled (a whole-batch library
         wrote the same text there already). *)
      write_source_artifact ~name (r.assemble all);
      resolved
  in
  Array.mapi r.entries ~f:(fun i (kparams, name, _launch) ->
      let result, symbol = resolved.(i) in
      { result; kparams; bindings; name; symbol })

let%diagn_sexp compile ~(name : string) bindings (lowered : Low_level.optimized) : procedure =
  (compile_kernels ~batch:false ~names:[| name |] bindings [| lowered |]).(0)

let%diagn_sexp compile_batch ~names bindings (lowereds : Low_level.optimized array) :
    procedure array =
  compile_kernels ~batch:true ~names bindings lowereds

let%track3_sexp link_compiled ?lowered_bindings ~merge_buffer ~resolve ~runner_label ctx_buffers
    (code : procedure) =
  let _name : string = code.name in
  let log_file_name = Utils.diagn_log_file [%string "debug-%{runner_label}-%{code.name}.log"] in
  (* When [lowered_bindings] is given (batch linking, e.g. fissioned segments of one routine), the
     static-index refs are shared: looked up by the [Static_idx] param's symbol rather than freshly
     minted, so one bindings assoc drives every procedure of the batch. *)
  let idx_ref s =
    match lowered_bindings with
    | Some lb -> Indexing.find_exn lb s
    | None -> Indexing.fresh_launch_param ()
  in
  let run_variadic =
    [%log_level
      0;
      let rec link :
          'a 'b 'idcs.
          'idcs Indexing.bindings ->
          kparam_source list ->
          ('a -> 'b) Ctypes.fn ->
          ('a -> 'b, 'idcs, 'p1, 'p2) Indexing.variadic =
       fun (type a b idcs) (binds : idcs Indexing.bindings) kparams (cs : (a -> b) Ctypes.fn) ->
        match (binds, kparams) with
        | Empty, [] -> Indexing.Result (Foreign.foreign ~from:code.result.lib code.symbol cs)
        | Bind _, [] -> invalid_arg "Cc_backend.link: too few static index params"
        | Bind (_, bs), Static_idx s :: ps -> Param_idx (idx_ref s, link bs ps Ctypes.(int @-> cs))
        | Empty, Static_idx _ :: _ -> invalid_arg "Cc_backend.link: too many static index params"
        | bs, Log_file_name :: ps ->
            Param_1 (ref (Some log_file_name), link bs ps Ctypes.(string @-> cs))
        | bs, Merge_buffer :: ps ->
            (* The device's merge buffer is a [buffer_loc] set (lazily) by a transfer routine;
               resolve it to the backend pointer at execution time. *)
            Param_2f (resolve, merge_buffer, link bs ps Ctypes.(ptr void @-> cs))
        | bs, Kparam_ptr tn :: ps ->
            let c_ptr =
              match Map.find ctx_buffers tn with
              | Some loc -> resolve loc
              | None ->
                  (* After gh-ocannl-333 there is no host array to fall back on: every in-context
                     node must be present in [ctx_buffers], put there by [Backends.allocate_delta]
                     before this link. The [buffer_loc -> base] resolution is backend-private (the
                     shared layer hands us locations, never pointers). *)
                  raise
                  @@ Utils.User_error
                       [%string
                         "Cc_backend.link_compiled: node %{Tn.debug_name tn} missing from context: \
                          %{Tn.debug_memory_mode tn.Tn.memory_mode_intent}"]
            in
            Param_2 (ref (Some c_ptr), link bs ps Ctypes.(ptr void @-> cs))
        | _, (Kparam_pool_slab _ | Kparam_pool_slots _) :: _ ->
            Backend_intf.unexpected_pooled_kparam ~backend:"Cc_backend"
      in
      (* Reverse the input order because [Indexing.apply] will reverse it again. Important:
         [code.bindings] are traversed in the wrong order but that's OK because [link] only uses
         them to check the number of indices. *)
      let kparams = List.rev_map code.kparams ~f:(fun (_, p) -> p) in
      link code.bindings kparams Ctypes.(void @-> returning void)]
  in
  let bindings =
    match lowered_bindings with
    | Some lb -> lb
    | None -> Indexing.lowered_bindings code.bindings run_variadic
  in
  let%diagn_sexp work () : unit =
    [%log_result _name];
    (* Launch-side validation of the values the kernel is about to read, mirroring the GPU backends'
       [Static_idx] checks: a task scheduled without [Context.run]'s pre-dispatch check must not
       launch on an unassigned parameter (gh-ocannl-940). *)
    Indexing.validate_lowered_bindings ~width64:Utils.settings.large_models bindings;
    Indexing.apply run_variadic ();
    if Utils.debug_log_from_routines () then
      Utils.log_debug_routine_file ~log_file_name ~stream_name:runner_label
  in
  ( bindings,
    Task.Task
      {
        (* In particular, keep code alive so it doesn't get unloaded. *)
        context_lifetime = (ctx_buffers, code);
        description = "executes " ^ code.name ^ " on " ^ runner_label;
        work;
      } )
