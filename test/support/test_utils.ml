(** Portable output helpers for OCANNL tests.

    The Windows and glibc C runtimes format floats differently: Windows prints 3-digit exponents
    ([e+018] vs. Linux's [e+18]) and rounds representable decimal ties away from zero where glibc
    rounds to even ([%.1f] of [2.25] prints [2.3] on Windows, [2.2] on Linux). Tests that print
    floats with raw [%g]/[%e]/[%f] into [.expected] files therefore fail across platforms. Use these
    printers instead — they are portable by construction. *)

open Base
open Stdio

module Config_key_scan = Config_key_scan
(** Scanning OCaml sources for the config keys they read, shared by the configuration-consistency
    tests. *)

module Dune_stanza_scan = Dune_stanza_scan

(* Which backends a dune argv can reach a stanza naming (gh-ocannl-1004, gh-ocannl-1066): the
   reachability half of tools/batch-backends.sh's resolution of a batch's backends. *)
module Slot_kind = Slot_kind
(** Reading dune files for the stanzas that run a test executable, and whether they declare the
    shared [ocannl_config]. *)

module Cache_dir_scan = Cache_dir_scan
(** Scanning OCaml sources for the autotune schedule cache directories they name, so that the one
    root [.gitignore] glob over their shared prefix covers all of them. *)

module Refused_action_scan = Refused_action_scan
module Refusal_control_scan = Refusal_control_scan

module Refusal_control_manifest = Refusal_control_manifest
(** Extracting static refusal-diagnostic fragments from repository scanners and relating them to
    permanent control goldens. *)

module Optional_arg_scan = Optional_arg_scan
(** Classifying whether [lib/] optional arguments affect their function bodies or are deliberately
    unimplemented, in which case the caller-visible label must begin with an underscore. *)

module Verdict_scan = Verdict_scan
(** Scanning test sources for claims a test decides itself and prints outside [Verdict], where a
    failing one is [dune promote]-able into the golden. *)

module Verdict_provenance = Verdict_provenance
(** Scope-and-polarity provenance of the Boolean a [Verdict] claim receives: which quantifiers it
    can rest on, through bindings, helpers, wrappers, matches and modules, and which populations are
    witnessed non-empty when it holds. *)

module Agent_notes_scan = Agent_notes_scan
(** Reading [docs/agent-notes.md] and [docs/agent-notes/] as structure: bullet integrity, index-hook
    agreement, table shape, reachability from the index, and repetition across files. *)

module Dead_export_scan = Dead_export_scan

module Api_drift = Api_drift
(** Enumerating source-declared values in modules without interfaces and conservatively counting
    external qualified, aliased, opened, and included references to them. *)

module Codegen_text_scan = Codegen_text_scan
(** Deciding what pins the TEXT of generated code: goldens holding emitted kernel or IR source, and
    test sources asserting on it from a string literal. *)

module Source_inventory = Source_inventory
(** A source-only repository inventory derived from a clean Dune sandbox. Repository scans select
    their corpus from this shared set instead of maintaining recursive source-root lists. *)

module Scan_argv = Scan_argv
(** Response-file expansion for the repository-wide scans: a [@<path>] argument stands for the words
    in that file. What a scan reads is handed to it on the command line, and Windows caps a command
    line at 32,767 characters. *)

module Scan_floors = Scan_floors
(** Floors over a scanned census: the tripwire that keeps a scanning test from passing vacuously,
    shared by the scans that glob the repository. *)

module Generated = Generated
(** Freshness-checked reads of the generated kernels under [build_files/], for tests that assert on
    emitted code. Artifacts outlive the run that wrote them, so a read that does not establish
    provenance can keep asserting on a kernel that is no longer emitted at all. *)

module Asm_census = Asm_census
(** The [-march] compile matrix and the innermost-loop instruction census (gh-ocannl-650): compiling
    an emitted kernel under a target the build host cannot run is what makes a guarded arm checkable
    at all, and counting its innermost loop is what separates "gcc accepted the arm" from "gcc kept
    it in registers as one vector operation". *)

(** [concise_float ~prec v] formats [v] with [prec] decimals, normalizing exponent digits portably.
    Re-export of [Ir.Ndarray.concise_float]. *)
let concise_float = Ir.Ndarray.concise_float

(** [hex_float v] formats [v] with OCaml's [%h] hex-float notation: bit-exact on every platform,
    sidestepping decimal-tie rounding divergence entirely. *)
let hex_float v = Printf.sprintf "%h" v

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

module Lexical_scope = Lexical_scope
(** The lexical scope model the source scans resolve names over: which binding an identifier reaches
    where it is spelled, parameterised over what a name denotes (gh-ocannl-1079). *)

module Ll_test_scan = Ll_test_scan

module Operand_key_scan = Operand_key_scan
(** Finding operand fixtures that mint values from a hand-rolled modulus of a multi-axis key instead
    of going through [Ll_test.cycle]'s blind-axis guard (gh-ocannl-1018). *)

module Provenance_tag_scan = Provenance_tag_scan
(** Where the virtualizer's rejection codes are minted, which files name them, and the boundary
    test's phase table checked against the minting functions (gh-ocannl-1015). *)

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
