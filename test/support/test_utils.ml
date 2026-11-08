(** Portable output helpers for OCANNL tests.

    The Windows and glibc C runtimes format floats differently: Windows prints 3-digit exponents
    ([e+018] vs. Linux's [e+18]) and rounds representable decimal ties away from zero where glibc
    rounds to even ([%.1f] of [2.25] prints [2.3] on Windows, [2.2] on Linux). Tests that print
    floats with raw [%g]/[%e]/[%f] into [.expected] files therefore fail across platforms. Use these
    printers instead — they are portable by construction. *)

open Base
open Stdio

module Generated = Generated
(** Freshness-checked reads of the generated kernels under [build_files/], for tests that assert on
    emitted code. Artifacts outlive the run that wrote them, so a read that does not establish
    provenance can keep asserting on a kernel that is no longer emitted at all. *)

module Asm_census = Asm_census
(** The [-march] compile matrix and the innermost-loop instruction census (gh-ocannl-650): compiling
    an emitted kernel under a target the build host cannot run is what makes a guarded arm checkable
    at all, and counting its innermost loop is what separates "gcc accepted the arm" from "gcc kept
    it in registers as one vector operation". *)

module Cache_plant = Cache_plant
(** Planting schedule-cache entries by hand from a few named arguments, and the waiver for claims
    resting on cache I/O the filesystem refused. *)

(** [concise_float ~prec v] formats [v] with [prec] decimals, normalizing exponent digits portably.
    Re-export of [Ir.Ndarray.concise_float]. *)
let concise_float = Ir.Ndarray.concise_float

(** [hex_float v] formats [v] with OCaml's [%h] hex-float notation: bit-exact on every platform,
    sidestepping decimal-tie rounding divergence entirely. *)
let hex_float v = Printf.sprintf "%h" v

(** [uint4x32_words cell] is the four 32-bit words of a [uint4x32] cell as {!Ir.Ndarray} stores it,
    a [Complex.t] (gh-ocannl-1704): the real part's bits hold words 0 and 1, the imaginary part's
    words 2 and 3, each pair little-endian -- the order the [Ir.Ops.<prec>_to_uint4x32] host stubs
    return them in. *)
let uint4x32_words (cell : Stdlib.Complex.t) =
  let lo = Int64.bits_of_float cell.re and hi = Int64.bits_of_float cell.im in
  let word bits k =
    Int64.to_int_exn (Int64.bit_and (Int64.shift_right_logical bits (32 * k)) 0xFFFF_FFFFL)
  in
  [| word lo 0; word lo 1; word hi 0; word hi 1 |]

(** [hex_words words] spells [uint4x32_words]'s result as four 8-digit hex words. *)
let hex_words words = String.concat_array ~sep:" " (Array.map words ~f:(Printf.sprintf "%08x"))

(** Prints [v] via [concise_float]. *)
let print_float ?(prec = 6) v = printf "%s" (concise_float ~prec v)

(** Prints [v] via [concise_float], followed by a newline. *)
let print_float_ln ?(prec = 6) v = printf "%s\n" (concise_float ~prec v)

(** Prints [vs] separated by [sep] (default a single space) via [concise_float]. *)
let print_floats ?(prec = 6) ?(sep = " ") vs =
  printf "%s" (String.concat ~sep (List.map vs ~f:(concise_float ~prec)))

(** Puts stdout in binary mode. Required when echoing a golden [.expected] file byte-for-byte (e.g.
    in [.missing.ml] backend stubs): text-mode stdout on Windows rewrites ["\n"] to ["\r\n"],
    corrupting the comparison. *)
let set_binary_stdout () = Out_channel.set_binary_mode stdout true

(** [with_policy adjust f] runs [f] under the numerics policy [adjust] makes of the ambient one, and
    restores the ambient policy after, exceptions included. A leg claiming the behaviour of a named
    policy — the default one included — pins it this way rather than inheriting it: a stanza that
    declares [OCANNL_FP16_ARITHMETIC] or [OCANNL_BF16_ARITHMETIC] runs under whatever mode the
    caller exported, and a default-labelled claim would then test another policy under the default's
    name (gh-ocannl-1053, gh-ocannl-1078). The policy is read while compiling, so wrapping the
    compile suffices. *)
let with_policy adjust f =
  let saved = Ir.Numerics.get () in
  Ir.Numerics.set_policy (adjust saved);
  Exn.protect ~f ~finally:(fun () -> Ir.Numerics.set_policy saved)

(** [with_fp16_auto f] is [with_policy] pinning [Fp16_auto], the default fp16 mode. *)
let with_fp16_auto f =
  with_policy (fun p -> { p with Ir.Numerics.fp16_arithmetic = Ir.Numerics.Fp16_auto }) f
