(* gh-ocannl-593 and gh-ocannl-595: what an OCANNL-linked executable puts on each stream before it
   reaches its own [main].

   The convention (gh-ocannl-581) is that stdout belongs to the program and every library diagnostic
   goes to stderr -- which is what lets `tools/fit_envelope.exe` and the benchmark runners make
   stdout a data channel. Nothing tested it: every other test config sets
   `suppress_welcome_message=true` and `log_config_sourcing=false`, so the whole suite would stay
   green with the [Stdio.eprintf] calls in `arrayjit/lib/utils.ml` flipped back to [Stdio.printf].

   This executable has no output of its own, which is what makes it usable as a subject: whatever a
   rule captures came from the library. The two rules next to it capture the two halves of the
   contract.

   - The stdout half: with the chatter turned all the way UP, stdout is still empty. - The stderr
   half: on a DEFAULT run, stderr is short enough that a warning on it is legible. That is the
   acceptance test of gh-ocannl-595, which is why `log_config_sourcing` defaults to false and
   `log_level` to 0. The golden is the welcome banner plus the unknown-config-key warning that this
   directory's `ocannl_config` provokes on purpose -- three lines, not eighty-four.

   The modes are argument-selected rather than three executables because the stdout half wants a
   subject that links the library and does nothing else, and the other two are a few lines each. *)

open Base
open Stdio

(* Keys that would rewrite the stderr golden if they arrived from the ambient environment, which
   outranks this directory's config file. As in `profiles/`, the executable cannot PREVENT this (the
   settings are read during [Utils]'s initialization) but detecting it is enough: a rule that exits
   nonzero writes no golden, so a mystifying diff becomes a named failure. The dune rule declares
   the same keys as env_var dependencies, which is what gets this guard RUN when one of them changes
   -- and [test/operations/env_var_deps] pairs THIS list with those declarations rather than leaving
   the two to agree by hand (gh-ocannl-749). Only the stderr rule guards; the stdout rule's claim is
   stream-level and holds under any of these.

   The startup-cleanup keys are not here: they are pinned on the rule's commandline, which no
   environment variable can outrank -- the honest option wherever it is available, a guard being
   only the second best. It is not available for these five: pinning them on the commandline is
   pinning the very defaults under test. *)
let stderr_shaping_keys =
  [ "suppress_welcome_message"; "log_config_sourcing"; "log_level"; "no_config_file"; "profile" ]

(* On STDOUT, deliberately: the rule captures this process's stderr into its target, so a diagnostic
   written there would be swallowed by the very redirection it is explaining, leaving dune to report
   a bare "Command exited with code 1". Stdout is where the reason survives. *)
let guard () =
  List.iter stderr_shaping_keys ~f:(fun arg_name ->
      Option.iter (Utils.read_env_var arg_name) ~f:(fun (value, var) ->
          printf
            "startup_streams: %s=%s is set in the environment and would rewrite this test's \
             expected stderr; unset it to run the test.\n"
            var value;
          Stdlib.exit 1))

(* The captured stderr names the config file by the absolute path the walk-up search built, so the
   golden gets the basename instead. The subject ran in this same directory (dune runs both actions
   of the rule there), which is what lets the prefix be stripped as a substring: tokenizing on
   spaces would mangle the path under a checkout whose own path contains one (Codex P2 on PR #348).
   Both separators are covered because the path is built with [Filename.concat] on the platform's,
   while the capture may be read anywhere. Also strips a trailing CR: on Windows the capture is
   text-mode, and the golden is pinned to LF by .gitattributes. *)
let directory_prefixes () =
  let cwd = Stdlib.Sys.getcwd () in
  let slashed = String.tr ~target:'\\' ~replacement:'/' cwd in
  let backslashed = String.tr ~target:'/' ~replacement:'\\' cwd in
  [ slashed ^ "/"; backslashed ^ "\\" ]

let normalize () =
  Out_channel.set_binary_mode stdout true;
  let prefixes = directory_prefixes () in
  In_channel.iter_lines In_channel.stdin ~f:(fun line ->
      let line = String.rstrip ~drop:(Char.equal '\r') line in
      List.fold prefixes ~init:line ~f:(fun line prefix ->
          String.substr_replace_all line ~pattern:prefix ~with_:"")
      |> printf "%s\n")

(* The conflicting file sets suppression true; the commandline sets false. Both the banner and its
   deferred provenance must reflect the commandline value, not an OR of source levels. *)
let bootstrap_check () =
  let text = In_channel.input_all In_channel.stdin in
  let contains substring = String.is_substring text ~substring in
  Verdict.p "explicit false overrides file welcome suppression"
    (contains "Welcome to OCANNL! Reading configuration defaults from");
  Verdict.p "bootstrap trace carries the commandline suppression source"
    (contains "Found false, commandline --ocannl_suppress_welcome_message=false");
  Verdict.p "bootstrap trace carries the commandline file-search source"
    (contains "Found false, commandline --ocannl_no_config_file=false");
  Verdict.p "profile trace carries the normalized commandline selection"
    (contains "Found reproducible, commandline --ocannl_profile= RePROducible ")

let () =
  (* The configuration flags a rule passes are addressed to the library, which read them during
     initialization; the mode is whatever else is on the commandline. *)
  match Bench_args.positional (Bench_args.create "startup_streams") with
  (* The stdout half: the library has already spoken by the time this runs. *)
  | [] -> ()
  | "guard" :: _ -> guard ()
  | "normalize" :: _ -> normalize ()
  | "bootstrap-check" :: _ -> bootstrap_check ()
  | arg :: _ ->
      eprintf "startup_streams: unknown mode %S (expected none, \"guard\" or \"normalize\")\n" arg;
      Stdlib.exit 1
