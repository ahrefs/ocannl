(* gh-ocannl-559: the six-sublevel precedence walk, on synthetic sources.

   Every source level splits into two sublevels -- the keys stated explicitly at that level, then
   the payload of a profile PICKED at that level -- so that a specific setting always beats an
   aggregate one of equal immediacy, and a profile named on the commandline still overrides an
   exhaustive config file. The table below is that claim: the same four sources, resolved four
   times, once per level the profile could have been picked at. *)

open Base
open Stdio

let alist_lookup alist key = List.Assoc.find alist key ~equal:String.equal

(* [k_cmd] is stated everywhere, [k_env] everywhere below the commandline, and so on; [k_prof] only
   by the payload, [k_none] nowhere. *)
let cmdline = [ ("k_cmd", "cmdline") ]
let env = [ ("k_cmd", "env"); ("k_env", "env") ]
let file = [ ("k_cmd", "file"); ("k_env", "file"); ("k_file", "file") ]

let payload =
  [ ("k_cmd", "payload"); ("k_env", "payload"); ("k_file", "payload"); ("k_prof", "payload") ]

let cmdline_lookup key =
  Option.map (alist_lookup cmdline key) ~f:(fun v -> (v, "--ocannl_" ^ key ^ "=" ^ v))

let env_lookup key =
  Option.map (alist_lookup env key) ~f:(fun v -> (v, "OCANNL_" ^ String.uppercase key))

let file_lookup key = alist_lookup file key

(* The table below is pure -- synthetic sources, embedded payloads, no real configuration is read --
   but linking arrayjit.utils runs the config machinery at module initialization, and an unknown
   ambient OCANNL_PROFILE aborts the process outright. (The startup chatter itself is harmless here:
   it goes to stderr, gh-ocannl-581.) Same stray-variable guard as
   test/operations/profiles/profile_precedence.ml (Codex P2 on PR #291), duplicated rather than
   shared: the two tests link only arrayjit.utils between them, and test_utils would drag in the
   whole IR for eight lines. A third copy is the signal to factor it out. *)
let () =
  Option.iter (Utils.read_env_var "profile") ~f:(fun (value, var) ->
      eprintf
        "config_profiles: %s=%s is set in the environment; an unknown profile name aborts this \
         test during module initialization. Unset it to run the test.\n"
        var value;
      Stdlib.exit 1)

let () =
  let levels =
    [
      (None, "nowhere (no profile)");
      (Some Utils.Cmdline_level, "the commandline");
      (Some Utils.Env_level, "the environment");
      (Some Utils.Config_file_level, "the config file");
    ]
  in
  List.iter levels ~f:(fun (level, description) ->
      let profile = Option.map level ~f:(fun l -> (l, "demo", alist_lookup payload)) in
      printf "Profile picked via %s:\n" description;
      List.iter [ "k_cmd"; "k_env"; "k_file"; "k_prof"; "k_none" ] ~f:(fun arg_name ->
          let value, source =
            Utils.resolve_config_value ~cmdline:cmdline_lookup ~env:env_lookup ~file:file_lookup
              ~profile ~default:"builtin" ~arg_name
          in
          printf "  %-6s = %-8s (%s)\n" arg_name value (Utils.config_source_label source));
      printf "\n");
  (* The payloads are ordinary partial config files; that they parse, and to what, is part of the
     documented contract (they are quoted verbatim in ocannl_config.reference). *)
  List.iter Utils.profile_payloads ~f:(fun (name, text) ->
      printf "Profile %S sets:\n" name;
      List.iter
        (Utils.parse_config_lines ~source:("profile " ^ name) (String.split_lines text))
        ~f:(fun (key, value) -> printf "  %s=%s\n" key value);
      printf "\n")

(* gh-ocannl-719: the `approximate` payload is the `performance` payload plus the numerics-changing
   knobs. Pinned as a relationship between the two parsed payloads rather than as a second copy of
   the performance keys: every key performance sets, approximate sets to the same value, and it sets
   more. *)
let () =
  let parsed name =
    Utils.parse_config_lines ~source:("profile " ^ name)
      (String.split_lines (List.Assoc.find_exn Utils.profile_payloads name ~equal:String.equal))
  in
  let performance = parsed "performance" and approximate = parsed "approximate" in
  Verdict.p_all ~min:5
    "every key the performance profile sets, the approximate profile sets to the same value"
    performance ~f:(fun (key, value) ->
      Option.equal String.equal (List.Assoc.find approximate key ~equal:String.equal) (Some value));
  Verdict.p "the approximate profile sets keys the performance profile does not"
    (List.length approximate > List.length performance)
