(* The GPU half of the e5m2 codec verification (gh-ocannl-657): OCANNL's software codec against the
   vendor's fp8 type, over every input rather than a sample.

   Why this is a tool and not a test: it needs a GPU, and it sweeps 2^32 float bit patterns and
   17.2e9 doubles. The pure-CPU half -- which pins the codec against a rounding oracle and pins the
   two codecs against each other -- is test/operations/fp8_codec_exhaustive.ml, on the `slow` alias.
   What only hardware can answer is whether the codec still agrees with `__nv_fp8_e5m2` /
   `__hip_fp8_e5m2`, which is the property gh-ocannl-646 established by changing OUR rounding to
   match THEIRS, and which a new CUDA or ROCm release can take away.

   The comparison matters because both conversions live inside one tensor: a kernel narrows with the
   vendor cast, while every host-side write of the same fp8 node goes through [Ops.single_to_fp8] /
   [Ops.double_to_fp8]. So the host side here IS the shipped codec (builtins.c, reached through
   fp8_soak_stubs.c) and the device side IS what the backend emits.

   WHICH SPELLING is a question only ROCm makes interesting (gh-ocannl-757). CUDA has one: the bare
   [(__nv_fp8_e5m2)x] the backend emits. HIP has two, because its bare cast is BROKEN for tiny
   magnitudes (gh-ocannl-647) and every OCANNL narrowing there goes through a guarded helper
   instead. So the default sweeps what the backend actually emits -- guarded on HIP, the cast
   everywhere else -- and that run is a pass/fail gate on every box: 0 disagreements. The raw cast
   is then an opt-in probe ([--spelling=raw]), and it does NOT claim agreement, because on an
   affected ROCm it cannot have it; it claims LOCALIZATION -- that every disagreement lies inside
   the window the guard covers -- which is true today, stays true (vacuously) once the platform is
   fixed, and whose disagreement count reaching 0 is the trigger to delete the guard.

   Usage: dune exec tools/fp8_soak.exe -- every arm the box has, both sweeps dune exec
   tools/fp8_soak.exe -- --arm=cuda --sweep=f32 dune exec tools/fp8_soak.exe -- --arm=hip
   --spelling=both Runs in a couple of minutes on an RTX 5070 Ti; see
   docs/agent-notes/backend-precision-and-simd.md.

   Exit status is a verdict: nonzero if any claim failed, so it can gate a release check.

   WHAT LIVES WHERE (gh-ocannl-758): this module is linked on every box, the per-vendor arms only
   where their jit library is installed (every box still type-checks them against the stub libraries
   of arrayjit/vendor_stubs, gh-ocannl-1688), so everything that is not a call into a vendor's jit
   library lives here -- the [vendor] records below hold each platform's name, C type, narrowing
   spellings and labels, and the reading of what its kernel macros mean. An arm holds its kernel
   source, its compile/load/launch/copy calls, and data extractors; its [last_compiled] says where
   and when it last saw a compiler, and the run header prints it. *)

open Base

type bytes_buf =
  (int, Stdlib.Bigarray.int8_unsigned_elt, Stdlib.Bigarray.c_layout) Stdlib.Bigarray.Array1.t

type counts_buf =
  (int64, Stdlib.Bigarray.int64_elt, Stdlib.Bigarray.c_layout) Stdlib.Bigarray.Array1.t

external soak_f32 : int64 -> int64 -> bytes_buf -> counts_buf -> unit = "ocannl_fp8_soak_f32"

external soak_f64 : int64 -> int64 -> int array -> bytes_buf -> counts_buf -> unit
  = "ocannl_fp8_soak_f64"

type spelling = [ `Raw | `Guarded ]
type arch_policy = [ `Device | `Backend ]

(* An arm narrows inputs with ONE vendor's fp8 type and says nothing about what the answer should
   be. Adding a vendor is adding a module of this shape, a [vendor] record below, and one `select`
   clause in tools/dune -- not a second program, which is how the CUDA sweep and the HIP sweep
   drifted apart the first time.

   THE SIGNATURE IS DELIBERATELY THIN (gh-ocannl-758). An arm file is linked and run only on a box
   that has its vendor's jit library: every box type-checks it against the stub libraries of
   arrayjit/vendor_stubs (gh-ocannl-1688), but whether it still drives the vendor correctly is first
   learned by whichever session next runs on the right hardware. So everything an arm could hold but
   does not have to CALL THE JIT LIBRARY FOR lives here instead: the vendor's name and C type, its
   narrowing spellings and their labels, the probe and header wording, the thresholds that decide
   which conversion path a macro reading means, and every claim label. What remains in an arm is:
   the vendor kernel source, the compile/load/launch/copy calls, and data extractors that read a
   vendor struct field or a kernel's own report. A change to how the soak BEHAVES then never touches
   an arm; a change to an arm is a change to how this vendor is driven. *)
module type ARM = sig
  val last_compiled : string
  (** Where and when this arm's source was last really linked and run: the box, the date, and where
      the evidence lives — the PR that carried it rather than a commit, since the sha a session
      verifies does not survive the rebase before merge. Printed in the run header, and updated by
      whoever runs it on a box that has the vendor library: type-checking happens on every box
      through arrayjit/vendor_stubs, so this is what tells the next editor how long the arm has gone
      without a real run. *)

  val built : bool
  (** [false] in the [.missing] stub that the `select` picks where the vendor's jit library is
      absent, [true] in a real arm. Nothing else in the stub is ever called: the selection below
      refuses an unbuilt arm before asking it anything. *)

  val device_count : unit -> (int, string) Result.t
  (** How many devices the vendor's runtime reports, with the exception text when initializing it
      raised. Asked because "the library is linked" is not "this box can run the arm": a switch
      carrying both `cudajit` and `hipjit` compiles both arms, and on a machine with one kind of GPU
      the default selection must skip the other rather than raise partway through a run, possibly
      after the first vendor's several-minute sweep has already completed. *)

  val set_arch_policy : arch_policy -> unit
  (** Which architecture the vendor kernel is compiled for, and therefore WHAT IS MEASURED.
      [cuda_fp8.hpp] guards its conversions with [#if __CUDA_ARCH__ >= 890]: at or above sm_89 the
      cast is the hardware conversion instruction, below it the header's software emulation. So
      [`Device] — this GPU's own capability — is the setting that verifies the codec against the
      HARDWARE, which is what gh-ocannl-646 established the codec's rules against, while [`Backend]
      reproduces [Cuda_backend]'s marker-driven policy, which for a source with no tensor-core
      markers passes no architecture at all and lands on the software path. Both are real questions;
      the default is [`Device]. Must be called before the arm's first sweep. An arm whose platform
      has nothing to select between (HIP: hiprtc compiles for the current device either way) takes
      the setting and ignores it. *)

  val device_report : unit -> (string * string) list
  (** What the vendor's device query says about this box, as pairs the run header formats. One key
      is looked up by name and must be present: ["device"], the accelerator's own name; further
      pairs are printed in order, under labels that say what they are FACTS ABOUT. A device
      capability is not the architecture the kernel compiled for — under [--arch=backend] a
      marker-free source gets nvrtc's default target, measured at [__CUDA_ARCH__ = 750] on a
      compute-12.0 device — so that architecture is the vendor record's [target], derived from what
      the compiled kernel reports, and never a device-query field wearing the name (Codex P2 round 1
      on PR #490). *)

  val compile_options : unit -> string list
  (** What the vendor's runtime compiler was given, so a run record says what it compiled. These
      come from the BACKEND (include discovery, arch policy), never from a local guess: one nvrtc
      caller disagreeing with the backend about which options a source needs is how a soak comes to
      measure something the backend never emits. *)

  val kernel_macros : unit -> (string * int) list
  (** What the COMPILED KERNEL reports about itself, keyed by the macro's own C spelling: CUDA's
      [__CUDA_ARCH__], HIP's [HIP_FP8_CVT_FAST_PATH] and [HIP_FP8_TYPE_OCP]. Which conversion those
      readings MEAN is decided here in the shared module, not in the arm, and rides in every claim's
      label so no run can be mistaken for the other kind afterwards (Codex P2 round 4 on PR #463).
  *)

  val narrow_f32 : spelling:spelling -> base:int -> count:int -> bytes_buf -> unit
  (** Fills [out.{i}] with the code the given spelling narrows the f32 with bit pattern [base + i]
      to. Never asked for a spelling outside the vendor record's [spellings]. *)

  val narrow_f64 : spelling:spelling -> base:int -> count:int -> lows:int array -> bytes_buf -> unit
  (** Fills [out.{4*i + k}] with the code the given spelling narrows the double with bit pattern
      [(base + i) << 32 | lows.(k)] to. *)
end

(* Everything about a vendor that is not a call into its jit library, and therefore everything that
   is compiled on every box (gh-ocannl-758). *)
type vendor = {
  name : string;  (** The [--arm=] selector, and the arm's name in every message. *)
  library : string;  (** The optional jit library the `select` in tools/dune keys off. *)
  runtime : string;  (** How the vendor's runtime is named when a probe fails. *)
  compiler : string;  (** The runtime compiler whose options the header reports. *)
  vendor_type : string;  (** The C type the kernels narrow to; goes in every claim. *)
  spellings : spelling list;
      (** Which narrowing spellings this vendor has, the DEFAULT first. [`Raw] is the platform's own
          cast, which is what every backend emits except HIP's, where it is broken for tiny
          magnitudes (gh-ocannl-647) and [`Guarded] — the [ocannl_*_to_fp8_uniform] helpers — is
          what is emitted instead. An arm is never asked for a spelling outside this list. *)
  spelling_label : spelling -> string;
      (** How the swept narrowing is written in an emitted kernel, for the claim labels. *)
  target : report:(string * string) list -> macros:(string * int) list -> string;
      (** The architecture the kernels were COMPILED FOR, as this vendor names it — which is not the
          same question as what the device is capable of, and the two answer differently exactly
          where the run record matters: under [--arch=backend] a marker-free source passes no
          [--gpu-architecture] and gets nvrtc's default target, measured at [__CUDA_ARCH__ = 750] on
          a compute-12.0 device. So CUDA derives this from the compiled kernel's own macro, while
          HIP takes the device's gcn arch — there hiprtc compiles for the current default device
          when given no [--offload-arch], so the device's architecture IS the compile target. *)
  macro_facts : macros:(string * int) list -> (string * string) list;
      (** Header facts that only the compiled kernel's macro readings can answer. *)
  conversion_path : target:string -> macros:(string * int) list -> string;
      (** Which conversion the vendor kernel actually got, as the COMPILED KERNEL reports it rather
          than as the options we passed imply — CUDA's [cuda_fp8.hpp] is the hardware instruction at
          [__CUDA_ARCH__ >= 890] and its own software emulation below, so a device whose capability
          is under sm_89 lands on the software path however honestly [--arch=device] asked for its
          own architecture. *)
}

(* Both lookups fail loudly rather than defaulting: a missing key means an arm and this module
   disagree about what the arm reports, which is exactly the drift the thin surface exists to make
   impossible to ship silently -- so it should stop the run on the box that can see it. *)
let field report name =
  match List.Assoc.find report name ~equal:String.equal with
  | Some v -> v
  | None -> failwith (Printf.sprintf "fp8_soak: the arm's device report has no %S entry" name)

let macro macros name =
  match List.Assoc.find macros name ~equal:String.equal with
  | Some v -> v
  | None -> failwith (Printf.sprintf "fp8_soak: the arm did not report %s" name)

(* cuda_fp8.hpp: `#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)` selects the hardware
   conversion; below it the header emulates in software. *)
let fp8_hardware_arch = 890
let cuda_arch_macro = "__CUDA_ARCH__"

let cuda_vendor =
  {
    name = "cuda";
    library = "cudajit";
    runtime = "the CUDA driver";
    compiler = "nvrtc";
    vendor_type = "__nv_fp8_e5m2";
    (* One spelling, and it is the platform's own cast: [Cuda_backend] emits [(__nv_fp8_e5m2)x] with
       nothing wrapped around it, because nothing on this platform needs guarding — gh-ocannl-647's
       defect is ROCm's. *)
    spellings = [ `Raw ];
    spelling_label =
      (function
      | `Raw -> "(__nv_fp8_e5m2)x"
      (* Unreachable through [spellings], and named rather than [assert false] so that a future
         guard on this platform fails to compile here instead of mislabelling a sweep. *)
      | `Guarded -> "no guarded spelling on CUDA");
    (* From the compiled kernel, never from the device: [--arch=backend] asks [Cuda_backend] for
       this marker-free source's options, which include no [--gpu-architecture] at all, so nvrtc's
       default target applies and a compute-12.0 box compiles for 7.5. The device's own capability
       is reported too, under a label that says that is what it is. *)
    target =
      (fun ~report:_ ~macros ->
        match macro macros cuda_arch_macro with
        | 0 -> Printf.sprintf "unknown (%s undefined)" cuda_arch_macro
        | arch -> Printf.sprintf "compute capability %d.%d" (arch / 100) (arch / 10 % 10));
    macro_facts = (fun ~macros:_ -> []);
    conversion_path =
      (fun ~target:_ ~macros ->
        let arch = macro macros cuda_arch_macro in
        if arch >= fp8_hardware_arch then
          Printf.sprintf "hardware cvt (%s = %d)" cuda_arch_macro arch
        else if arch = 0 then Printf.sprintf "unknown (%s undefined)" cuda_arch_macro
        else
          Printf.sprintf "header software path (%s = %d < %d)" cuda_arch_macro arch
            fp8_hardware_arch);
  }

(* [amd_hip_fp8.h]: `#if (defined(__gfx942__) || __gfx1200__ || __gfx1201__ || __gfx950__ ||
   __gfx1250__) && __HIP_DEVICE_COMPILE__` sets HIP_FP8_CVT_FAST_PATH to 1, and every conversion
   entry point branches on it; at 0 the header's own software [cast_to_f8] runs, which is the
   function gh-ocannl-647 is about. A sweep that silently took the fast path would report the bug
   fixed, so the reading goes into every claim's label. HIP_FP8_TYPE_OCP rides along because it
   selects which fp8 INTERPRETATION the type has; e5m2 is the OCP one, and a build where only FNUZ
   is available would be narrowing to a different format entirely. The arm reports the macros; which
   value means what is decided here. 2 is the arm's "the header did not define it". *)
let hip_fast_path_macro = "HIP_FP8_CVT_FAST_PATH"
let hip_ocp_macro = "HIP_FP8_TYPE_OCP"

let hip_vendor =
  {
    name = "hip";
    library = "hipjit";
    runtime = "the HIP runtime";
    compiler = "hiprtc";
    vendor_type = "__hip_fp8_e5m2";
    (* The raw cast and the guarded helpers are different conversions on ROCm, so the sweep has to
       say which one it swept. [`Guarded] first: it is what OCANNL emits, hence the default. *)
    spellings = [ `Guarded; `Raw ];
    spelling_label =
      (function `Raw -> "(__hip_fp8_e5m2)x" | `Guarded -> "ocannl_{single,double}_to_fp8_uniform");
    macro_facts =
      (fun ~macros ->
        [
          ( "fp8 interpretation",
            match macro macros hip_ocp_macro with 1 -> "OCP" | 0 -> "FNUZ only" | _ -> "unknown" );
        ]);
    (* hiprtc given no [--offload-arch] compiles for the current default device, so the device's own
       gcn arch IS what the kernels were built for -- unlike CUDA, where the two can differ. *)
    target = (fun ~report ~macros:_ -> field report "gcn arch");
    conversion_path =
      (fun ~target ~macros ->
        (* The gcn arch travels with the answer because it is what SELECTS the side, but the value
           reported is the macro the kernel compiled with, not a name matched against a list
           here. *)
        match macro macros hip_fast_path_macro with
        | 1 -> Printf.sprintf "hardware cvt (%s = 1 on %s)" hip_fast_path_macro target
        | 0 -> Printf.sprintf "header software cast_to_f8 (%s = 0 on %s)" hip_fast_path_macro target
        | _ -> Printf.sprintf "unknown (%s undefined, on %s)" hip_fast_path_macro target);
  }

let arms : (vendor * (module ARM)) list =
  [ (cuda_vendor, (module Fp8_soak_cuda)); (hip_vendor, (module Fp8_soak_hip)) ]

(* Whether THIS BOX can run the arm, with the reason when it cannot -- the vendor's jit library
   compiled in AND its runtime reporting a device. An explicit [--arm] runs regardless of the
   default selection, so a box whose hardware is missing or misconfigured still gets this diagnosis
   instead of a silent skip. *)
let probe v (module A : ARM) =
  if not A.built then
    Error (Printf.sprintf "not built: the %s arm needs the %s library" v.name v.library)
  else
    match A.device_count () with
    | Ok 0 -> Error (Printf.sprintf "%s is linked, but %s reports no device" v.library v.runtime)
    | Ok _ -> Ok ()
    | Error e ->
        Error
          (Printf.sprintf "%s is linked, but %s initialization failed: %s" v.library v.runtime e)

let target v (module A : ARM) = v.target ~report:(A.device_report ()) ~macros:(A.kernel_macros ())

let conversion_path v arm =
  let (module A : ARM) = arm in
  v.conversion_path ~target:(target v arm) ~macros:(A.kernel_macros ())

(* The compiler options are part of the answer, not decoration: on CUDA which [--gpu-architecture]
   the kernel was built for decides whether the cast became a hardware conversion or the header's
   software fallback, and a soak whose record does not say which was measured is asking to be
   re-derived later. *)
let describe v (module A : ARM) =
  let report = A.device_report () in
  let macros = A.kernel_macros () in
  let target = v.target ~report ~macros in
  let facts =
    (* The compile target leads, because it is what the rest of the run is relative to; then the
       arm's own device facts under their own labels, then what only the macros can say. A device
       fact the target already states verbatim is not repeated -- on HIP the gcn arch IS the compile
       target, on CUDA the device's capability and the target are two different numbers. *)
    ("target", target)
    :: List.filter report ~f:(fun (k, x) ->
        (not (String.equal k "device")) && not (String.equal x target))
    @ v.macro_facts ~macros
    |> List.map ~f:(fun (k, x) -> Printf.sprintf "%s: %s" k x)
  in
  Printf.sprintf "%s (%s); %s options: %s" (field report "device") (String.concat ~sep:", " facts)
    v.compiler
    (match A.compile_options () with [] -> "(none)" | os -> String.concat ~sep:" " os)

(* The file the `select` in tools/dune picks for this vendor where its library is present -- i.e.
   the file whose [last_compiled] is being reported, so the reader knows which one to update. *)
let arm_source v = Printf.sprintf "tools/fp8_soak_%s.%s.ml" v.name v.library

(* Mirrors the S_* offsets in fp8_soak_stubs.c. *)
let s_finite = 0
let s_inf = 1
let s_nan = 2
let s_inf_seen = 3
let s_nan_seen = 4
let s_inf_codes = 5
let s_nan_codes = 9
let s_all_codes = 13
let s_unguarded = 17
let s_dis_exps = 18
let s_reported = 50
let s_records = 51
let s_max_records = 8
let s_len = s_records + (3 * s_max_records)
let two_pow_32 = 0x1_0000_0000

(* Half the smallest e5m2 subnormal: below this magnitude every correctly rounded narrowing answers
   a signed zero, which is why the HIP guard's clamp is exact and not an approximation. The stub
   classifies each finite disagreement against it. *)
let guard_threshold_exp = -17

(* The gh-ocannl-647 window, and it is not a range: ROCm's defect RECURS WITH PERIOD 64 in the
   input's binary exponent, which is the signature of the defect itself. [cast_to_f8] computes an
   [exponent_diff] that reaches 85 for an f32 source and far more for an f64 one, then shifts a
   64-bit value by it -- and a shift by >= 64 is taken mod 64, so every 64th binary exponent
   reproduces the same wrong shift amount and the same wrong answer. The window is therefore a
   residue class: magnitudes 2^m with m <= -78 and (-78 - m) mod 64 < 4.

   Measured exhaustively on gfx1151 / ROCm 7.14.60850 (gh-ocannl-757). The F32 sweep can only reach
   the topmost member -- f32 exponent fields 46..49, i.e. 2^-81..2^-77 -- and a sweep restricted to
   f32-exact doubles reaches only one more, the f32-subnormal magnitudes ~4.5e-44..1.8e-43, which is
   why gh-ocannl-647 recorded "two windows". The full 17.2e9-double sweep here reaches all FIFTEEN,
   down to 2^-977, and the period is what ties them together.

   A run that finds a disagreement outside this residue class has found something new, and says so:
   the claim below is one-sided containment, so it stays true (vacuously) on a repaired platform. *)
let window_top = -78
let window_period = 64
let window_width = 4
let in_documented_window m = m <= window_top && (window_top - m) % window_period < window_width

(* The four low halves the f64 sweep crosses every top half with: zero, one ulp up, the mantissa's
   own midpoint bit, and all ones. The midpoint is the point -- gh-ocannl-648 was a double sitting
   just off an f32 tie, and a codec that narrows via f32 first moves it ONTO the tie. *)
let low_halves = [| 0x00000000; 0x00000001; 0x80000000; 0xFFFFFFFF |]

let fresh_counts () : counts_buf =
  let b = Stdlib.Bigarray.Array1.create Stdlib.Bigarray.int64 Stdlib.Bigarray.c_layout s_len in
  Stdlib.Bigarray.Array1.fill b 0L;
  b

let get c i = Int64.to_int_exn c.{i}

(* A bitmap the stub filled, as the list of set indices. [~size] because two of them are swept: the
   256 e5m2 codes, and the 2048 possible biased exponent fields of an f64 input. *)
let bit_set (c : counts_buf) slot ~size =
  List.filter (List.range 0 size) ~f:(fun i ->
      not (Int64.equal 0L (Int64.bit_and c.{slot + (i / 64)} (Int64.shift_left 1L (i % 64)))))

let code_set (c : counts_buf) slot = bit_set c slot ~size:256

let show_codes codes =
  if List.is_empty codes then "none"
  else String.concat ~sep:" " (List.map codes ~f:(fun c -> Printf.sprintf "0x%02x" c))

(* Progress on stderr, and only to a terminal: piped into a log the carriage returns would be one
   line per percent of a sweep that already reports its wall clock. *)
let interactive = Unix.isatty Unix.stderr
let progress = ref (-1)

let show_progress label done_ total =
  let pct = 100 * done_ / total in
  if interactive && pct > !progress then (
    progress := pct;
    Stdio.eprintf "\r%s: %d%%%!" label pct)

let end_progress () =
  progress := -1;
  if interactive then Stdio.eprintf "\r%!"

(* Chunk sizes, in device bytes: 64 MiB each, so the soak keeps one modest allocation on a GPU that
   may be shared, and the transfer is large enough that neither launch nor copy latency shows. *)
let f32_chunk = 1 lsl 26
let f64_chunk = 1 lsl 24 (* top halves; four bytes of output each *)

let report_records (c : counts_buf) ~arm ~vendor =
  for k = 0 to get c s_reported - 1 do
    let base = s_records + (3 * k) in
    Stdio.eprintf "%s: input 0x%Lx -> codec 0x%02x, %s 0x%02x\n" arm c.{base}
      (get c (base + 1))
      vendor
      (get c (base + 2))
  done

let elapsed since = Unix.gettimeofday () -. since

(* Two landmark narrowings, printed once: a tie that must go to even (1.125 -> 1.0, code 0x3C) and
   the tie between zero and the smallest subnormal (2^-17 -> +0). They are also what puts builtins.o
   on the link line -- the sweep reaches [single_to_fp8] and [double_to_fp8] through an `extern`
   declaration in C, and a member of the [ir] library's stub archive is only pulled in once
   something has referenced it, which the OCaml externals here do. *)
let codec_landmarks () =
  Printf.sprintf "single_to_fp8 1.125 -> 0x%02x, double_to_fp8 2^-17 -> 0x%02x"
    (Ir.Ops.single_to_fp8 1.125)
    (Ir.Ops.double_to_fp8 (Float.( ** ) 2. (-17.)))

let run_f32 v (module A : ARM) ~spelling =
  let counts = fresh_counts () in
  let buf : bytes_buf =
    Stdlib.Bigarray.Array1.create Stdlib.Bigarray.int8_unsigned Stdlib.Bigarray.c_layout f32_chunk
  in
  let started = Unix.gettimeofday () in
  let base = ref 0 in
  while !base < two_pow_32 do
    let count = Int.min f32_chunk (two_pow_32 - !base) in
    A.narrow_f32 ~spelling ~base:!base ~count buf;
    soak_f32 (Int64.of_int !base) (Int64.of_int count) buf counts;
    base := !base + count;
    show_progress (v.name ^ " f32") !base two_pow_32
  done;
  end_progress ();
  (counts, elapsed started)

let run_f64 v (module A : ARM) ~spelling =
  let counts = fresh_counts () in
  let buf : bytes_buf =
    Stdlib.Bigarray.Array1.create Stdlib.Bigarray.int8_unsigned Stdlib.Bigarray.c_layout
      (4 * f64_chunk)
  in
  let started = Unix.gettimeofday () in
  let base = ref 0 in
  while !base < two_pow_32 do
    let count = Int.min f64_chunk (two_pow_32 - !base) in
    A.narrow_f64 ~spelling ~base:!base ~count ~lows:low_halves buf;
    soak_f64 (Int64.of_int !base) (Int64.of_int count) low_halves buf counts;
    base := !base + count;
    show_progress (v.name ^ " f64") !base two_pow_32
  done;
  end_progress ();
  (counts, elapsed started)

(* Every finite e5m2 code, both signs: 0x00-0x7B and 0x80-0xFB. A sweep of all 2^32 float patterns
   must reach all of them, and a kernel that silently wrote nothing would not. *)
let all_finite_codes = List.filter (List.range 0 256) ~f:(fun c -> c land 0x7F <= 0x7B)

(* An input's base-2 magnitude exponent from its biased exponent FIELD. A zero field is subnormal
   and its magnitude is only bounded above, by 2^(1-bias) -- which is what is returned, so a
   subnormal disagreement is judged against the window by the largest magnitude it could have and
   therefore never sneaks INTO one. *)
let magnitude_exp ~bias e = if e = 0 then 1 - bias else e - bias

let show_window () =
  Printf.sprintf "2^%d..2^%d and every 2^-%d below that, %d binary exponents wide"
    (window_top - window_width + 1)
    (window_top + 1) window_period window_width

(* The disagreeing exponent fields as CONSECUTIVE RUNS with their magnitudes, because the runs are
   the finding: ROCm's defect comes in groups of four adjacent exponents repeating every 64, and a
   flat list of sixty numbers hides exactly the structure that identifies the mod-64 shift. *)
let show_exps exps ~bias =
  let runs =
    List.fold_right exps ~init:[] ~f:(fun e acc ->
        match acc with (lo, hi) :: rest when lo = e + 1 -> (e, hi) :: rest | _ -> (e, e) :: acc)
  in
  String.concat ~sep:", "
    (List.map runs ~f:(fun (lo, hi) ->
         if lo = hi then Printf.sprintf "%d (2^%d)" lo (magnitude_exp ~bias lo)
         else
           Printf.sprintf "%d-%d (2^%d..2^%d)" lo hi (magnitude_exp ~bias lo)
             (magnitude_exp ~bias hi)))

let report v arm ~sweep ~inputs ~spelling ~bias ~exp_size (counts, seconds) =
  let vendor = v.vendor_type in
  (* Both halves of "what was measured" ride in every claim: which side of the vendor header's
     compile-time split the kernel compiled on, and which of the vendor's narrowing spellings it
     called. Neither is inferable from the options this program passed. *)
  let via =
    Printf.sprintf "%s, narrowing with %s" (conversion_path v arm) (v.spelling_label spelling)
  in
  let finite = get counts s_finite in
  let exps = bit_set counts s_dis_exps ~size:exp_size in
  Stdio.printf "\n%s %s sweep, %s: %d inputs, %.1fs\n" v.name sweep (v.spelling_label spelling)
    inputs seconds;
  Stdio.printf "  non-finite inputs: %d infinite, %d NaN\n" (get counts s_inf_seen)
    (get counts s_nan_seen);
  Stdio.printf "  %s codes on infinite inputs: %s (%d disagree with the codec)\n" vendor
    (show_codes (code_set counts s_inf_codes))
    (get counts s_inf);
  Stdio.printf "  %s codes on NaN inputs: %s (%d disagree with the codec)\n" vendor
    (show_codes (code_set counts s_nan_codes))
    (get counts s_nan);
  Stdio.printf "  distinct %s codes produced overall: %d\n" vendor
    (List.length (code_set counts s_all_codes));
  (* Descriptive, not a claim: on the raw HIP spelling this number is the FINDING (gh-ocannl-647),
     and it is the guard's removal check -- when a ROCm release brings it to 0, the guarded
     narrowing in [Builtins_hip] can go. Where the number must be zero, the claim below says so. *)
  Stdio.printf "  finite-input disagreements: %d, on %d exponent field(s)%s\n" finite
    (List.length exps)
    (if List.is_empty exps then "" else ": " ^ show_exps exps ~bias);
  report_records counts ~arm:v.name ~vendor;
  (* The raw cast is known-defective on ROCm, so on an arm that HAS a guarded spelling the raw probe
     asserts LOCALIZATION rather than agreement: asserting agreement there would be a claim that
     fails by design on affected hardware, which makes the tool unusable as a gate exactly where it
     is most needed. Everywhere else -- CUDA, and HIP's guarded spelling, which is what OCANNL emits
     -- the claim is the full one. *)
  let defective_by_design =
    (match spelling with `Raw -> true | `Guarded -> false)
    && List.exists v.spellings ~f:(function `Guarded -> true | `Raw -> false)
  in
  if defective_by_design then (
    (* Both of these are VACUOUSLY true at zero disagreements, and that is the intended end state,
       not an unguarded quantifier: a fixed ROCm has nothing to localize, and the descriptive count
       above is what distinguishes "the window still holds" from "the window is empty". So they use
       [Verdict.pf] rather than the non-emptiness-guarded [p_all]/[p_none], which would fail a run
       on a platform that had been repaired. *)
    Verdict.pf
      "every %s %s disagreement with the codec is one the %s guarded narrowing closes, i.e. lies \
       inside the |x| < 2^%d it clamps to a signed zero, via the %s"
      vendor sweep v.name guard_threshold_exp via
      (get counts s_unguarded = 0);
    Verdict.pf
      "the %s %s disagreements are confined to the documented gh-ocannl-647 magnitude window (%s), \
       via the %s"
      vendor sweep (show_window ()) via
      (List.for_all exps ~f:(fun e -> in_documented_window (magnitude_exp ~bias e))))
  else
    Verdict.pf
      "the software codec and %s agree on every finite %s input the sweep covers, via the %s" vendor
      sweep via (finite = 0);
  let produced = code_set counts s_all_codes in
  Verdict.p_all
    (Printf.sprintf "the %s %s sweep produced every signed finite e5m2 code, via the %s" vendor
       sweep via) all_finite_codes ~min:248 ~f:(fun c -> List.mem produced c ~equal:Int.equal)

let usage () =
  Stdio.printf
    "fp8_soak: sweep OCANNL's e5m2 codec against a GPU vendor's fp8 type.\n\
    \  --arm=cuda|hip         which vendor (default: every arm this build has)\n\
    \  --sweep=f32|f64|both   which input set (default: both)\n\
    \  --spelling=default|raw|guarded|both\n\
    \                         which narrowing to sweep. default: what the backend emits --\n\
    \                         the guarded ocannl_*_to_fp8_uniform helpers on HIP, the bare\n\
    \                         vendor cast elsewhere; raw: the bare cast, which on ROCm is\n\
    \                         gh-ocannl-647's defect and claims localization, not agreement\n\
    \  --arch=device|backend  device: this GPU's own capability, so the vendor cast is the\n\
    \                         hardware instruction (default, and what verifies the codec against\n\
    \                         the hardware); backend: the repo's marker-driven arch policy, which\n\
    \                         for a marker-free source lands on the header's software path.\n\
    \                         CUDA only: ROCm's split is keyed off the target architecture\n\
    \                         macro, which hiprtc takes from the device either way\n"

let () =
  let arm_filter = ref None in
  let sweep = ref "both" in
  let spelling = ref "default" in
  let arch = ref `Device in
  Array.iteri Stdlib.Sys.argv ~f:(fun i s ->
      if i > 0 then
        match String.lsplit2 s ~on:'=' with
        | Some ("--arm", v) -> arm_filter := Some (String.lowercase v)
        | Some ("--sweep", v) -> sweep := String.lowercase v
        | Some ("--spelling", (("default" | "raw" | "guarded" | "both") as v)) -> spelling := v
        | Some ("--arch", "device") -> arch := `Device
        | Some ("--arch", "backend") -> arch := `Backend
        | _ ->
            usage ();
            Stdlib.exit (if String.equal s "--help" then 0 else 2));
  (* The default takes the arms that probe clean and says out loud why it passed over each of the
     others. An arm named EXPLICITLY is never skipped silently: a box whose driver is missing or
     whose GPU is not visible gets the probe's own diagnosis as a failed verdict — which carries the
     nonzero exit an uncaught exception would, and the reason without the backtrace. *)
  let selected =
    match !arm_filter with
    | Some n ->
        List.filter arms ~f:(fun (v, arm) ->
            String.equal n v.name
            &&
            match probe v arm with
            | Ok () -> true
            | Error why ->
                Verdict.fail (Printf.sprintf "the %s arm can run on this box (%s)" v.name why);
                false)
    | None ->
        List.filter arms ~f:(fun (v, arm) ->
            match probe v arm with
            | Ok () -> true
            | Error why ->
                Stdio.eprintf "fp8_soak: skipping the %s arm: %s\n%!" v.name why;
                false)
  in
  (match (selected, !arm_filter) with
  | [], Some _ when Verdict.any_failed () -> () (* the probe already said why, as a verdict *)
  | [], _ ->
      Stdio.eprintf "fp8_soak: no arm selected; this build has: %s\n%!"
        (String.concat ~sep:", "
           (List.map arms ~f:(fun (v, arm) ->
                match probe v arm with
                | Ok () -> Printf.sprintf "%s (ready)" v.name
                | Error why -> Printf.sprintf "%s (%s)" v.name why)));
      Verdict.fail "at least one GPU arm can run on this box"
  | _ -> ());
  List.iter selected ~f:(fun (v, arm) ->
      let (module A : ARM) = arm in
      A.set_arch_policy !arch;
      (* Flushed, so the arm header cannot land after a skip notice on the other stream. *)
      Stdio.printf "%s arm: %s\n%!" v.name (describe v arm);
      (* Which file was swept and when it last really ran (gh-ocannl-758): this arm's source is
         type-checked everywhere through arrayjit/vendor_stubs but linked only where its vendor
         library is, so a run is also the record that the file in the repository still drives the
         vendor somewhere. *)
      Stdio.printf "  arm source: %s, last compiled %s\n" (arm_source v) A.last_compiled;
      Stdio.printf "  codec: builtins.c single_to_fp8 / double_to_fp8 (%s); vendor: %s\n"
        (codec_landmarks ()) v.vendor_type;
      (* Printed before any sweep, and repeated inside every claim: an [--arch=device] run on a
         pre-sm_89 GPU asks honestly for the device's own architecture and still gets the header's
         software conversion, so "device mode" is not by itself a statement about hardware. *)
      Stdio.printf "  conversion swept: %s\n%!" (conversion_path v arm);
      (* [`Guarded] is asked for only where the vendor has it -- [v.spellings] is that menu, with
         the default at the head -- so [--spelling=guarded] on CUDA sweeps nothing and says so
         rather than reaching a [narrow_f32] that would reject it. *)
      let available = v.spellings in
      let wanted =
        match !spelling with
        | "default" -> [ List.hd_exn available ]
        | "raw" -> [ `Raw ]
        | "guarded" -> [ `Guarded ]
        | _ -> available
      in
      let explicit = String.(!spelling = "raw" || !spelling = "guarded") in
      let wanted =
        List.filter wanted ~f:(fun sp ->
            List.mem available sp ~equal:Poly.equal
            ||
            (* Asked for by name and not there: a verdict, on the same reasoning as an explicit
               [--arm] whose hardware is missing. Under [--spelling=both] it is only a note, since
               "both" means "both of the ones this arm has". *)
            (if explicit then
               Verdict.fail
                 (Printf.sprintf "the %s arm has the %s narrowing that was asked for" v.name
                    !spelling)
             else
               Stdio.eprintf "fp8_soak: the %s arm has no %s narrowing; skipping that spelling\n%!"
                 v.name !spelling;
             false))
      in
      List.iter wanted ~f:(fun spelling ->
          if String.(!sweep = "f32" || !sweep = "both") then
            report v arm ~sweep:"f32" ~inputs:two_pow_32 ~spelling ~bias:127 ~exp_size:256
              (run_f32 v arm ~spelling);
          if String.(!sweep = "f64" || !sweep = "both") then
            report v arm ~sweep:"f64" ~inputs:(4 * two_pow_32) ~spelling ~bias:1023 ~exp_size:2048
              (run_f64 v arm ~spelling)))
