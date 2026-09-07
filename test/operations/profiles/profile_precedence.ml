(* gh-ocannl-559: profile resolution end to end, through the real config sources.

   The `ocannl_config` next to this test is the "exhaustive config file" scenario: it picks the
   `reproducible` profile AND states three settings explicitly, two of them keys that payload also
   sets. The dune rules run this executable four times -- with no argument, with a
   commandline-picked profile, with an environment-picked one, and with a commandline-picked profile
   plus one explicitly overridden key -- so the four .expected files are the precedence claims of
   the issue, checked against what the library actually resolves. *)

open Base
open Stdio

(* Keys where the built-in payloads disagree, plus the three the config file states explicitly
   (autotune_rounds, cc_vector_bytes, fp16_arithmetic) and one no source mentions. *)
let keys =
  [
    ("autotune_search", "true");
    ("autotune_rounds", "2");
    ("autotune_beam_width", "2");
    ("model_default_schedule", "false");
    ("tune_inline_flips", "0");
    ("cc_backend_arch_flags", "auto");
    ("cc_backend_simd_flags", "auto");
    ("cc_backend_fp_contract", "auto");
    ("cc_backend_fast_math", "false");
    ("cc_vector_bytes", "-1");
    ("fp16_arithmetic", "auto");
    ("tf32_matmuls", "false");
    ("virtualize_max_visits", "1");
  ]

(* `no_config_file` is consulted during [Utils]'s own initialization, before this executable reaches
   [main], and would discard the ocannl_config the whole test rests on. Nothing here can PREVENT
   that -- but detecting it is enough, since a rule that exits nonzero writes no golden, and
   detection is one list entry where clearing it in dune would be a `setenv` per rule (Codex P2 on
   PR #291; two spellings x six rules before gh-ocannl-652 dropped the lowercase one, four before
   gh-ocannl-605 dropped the dash-prefixed ones). The startup chatter keys
   (`suppress_welcome_message`, `log_config_sourcing`) need no guard: that output goes to stderr
   (gh-ocannl-581) and the rules capture stdout only. *)
let guarded_keys = List.map keys ~f:fst @ [ "no_config_file" ]

let () =
  (* An OCANNL variable in the ambient environment outranks this directory's config file, so it
     would rewrite the golden -- and dune tracks no environment variable but OCANNL_BACKEND, so a
     stale output could be reused besides (Codex P2 on PR #291). Fail with the variable's name
     instead of producing a mystifying diff. `profile` is exempt: the env-picked rule sets it. *)
  List.iter guarded_keys ~f:(fun arg_name ->
      Option.iter (Utils.read_env_var arg_name) ~f:(fun (value, var) ->
          eprintf
            "profile_precedence: %s=%s is set in the environment and would outrank this test's \
             ocannl_config; unset it to run the test.\n"
            var value;
          Stdlib.exit 1));
  (match Utils.active_profile with
  | None -> printf "%-24s = %-14s (%s)\n" "profile" "" "unset"
  | Some (level, name, _) ->
      printf "%-24s = %-14s (picked via %s)\n" "profile" name (Utils.describe_config_level level));
  List.iter keys ~f:(fun (arg_name, default) ->
      let value, source = Utils.get_global_arg_with_source ~default ~arg_name in
      printf "%-24s = %-14s (%s)\n" arg_name value (Utils.config_source_label source))
