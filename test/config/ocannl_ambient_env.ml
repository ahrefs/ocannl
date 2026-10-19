(* gh-ocannl-1149: the environment half of every test's configuration, as a file dune can digest.

   A test resolves its configuration from three places: the `ocannl_config` beside it, the
   environment, and its own command line. Dune sees the first and the last -- one is a declared
   dependency, the other is written in the rule -- but the environment only where a stanza names a
   variable with `(env_var OCANNL_<KEY>)`, and nothing tied those names to the keys the linked
   pipeline reads. They were written by hand, review found one missing key per round, and an
   undeclared key served the previous run's result as a pass whenever it changed.

   This program prints the part of the environment addressed to OCANNL's configuration, so that the
   file it writes is the dependency instead: `test/config/dune` runs it under `(universe)` -- once
   per dune invocation, a few milliseconds -- and appends its output to the shared `ocannl_config`
   every test directory copies. When the environment is unchanged the output is byte-identical and
   dune's early cutoff stops there; when a variable changes, so does every `ocannl_config` copy, and
   every rule depending on one reruns. No key is listed anywhere: the set is whatever the
   environment holds, classified by `Utils.classify_env_var` -- the function the startup check and
   `env_var_deps` already share -- so a key the pipeline will start reading tomorrow is covered
   today.

   What is left out, and why:

   - Names not addressed to OCANNL ([Env_not_addressed]), and the reserved namespaces
   ([Env_reserved]): `OCANNL_TOOL_...` is the tooling's -- tools/test-run.sh exports a fresh run
   directory in it on every run, which would invalidate the whole suite each time -- and
   `OCANNL_LOG_LEVEL_<MODULE>` gates are read while preprocessing, which `env_var_deps` pairs with
   the libraries that read them. - `OCANNL_BACKEND`. It is the one key whose invalidation each
   stanza already states, and states exactly: `(env_var OCANNL_BACKEND)` or a backend marker,
   enforced by `env_var_deps` (gh-ocannl-659). Tracking it here too would rerun every backend-pinned
   stanza on every switch of backend -- what a sweep does between legs -- for no change in any
   result.

   Unknown keys and unread spellings ARE included: each makes every OCANNL program print a warning
   (or abort) on stderr, which some goldens capture.

   The lines are comments to the configuration reader, which skips `#` lines: the values reach the
   run through the environment exactly as before, and the copy only records them. Sorted, so the
   order the C library hands the environment over cannot change the digest; escaped, so a value
   holding a newline cannot end the comment. *)

open Base

let () =
  Stdlib.set_binary_mode_out Stdlib.stdout true;
  let tracked =
    Unix.environment () |> Array.to_list
    |> List.filter_map ~f:(fun binding ->
        match String.lsplit2 binding ~on:'=' with
        (* An empty value is unset at every source ([Utils.read_env_var]). *)
        | None | Some (_, "") -> None
        | Some (name, value) -> (
            match Utils.classify_env_var name with
            | Utils.Env_not_addressed | Utils.Env_reserved _ -> None
            | Utils.Env_config_key "backend" -> None
            | Utils.Env_config_key _ | Utils.Env_unknown_key _ | Utils.Env_unread_spelling _
            | Utils.Env_unread_reserved _ ->
                Some (name, value)))
    |> List.sort ~compare:(fun (a, _) (b, _) -> String.compare a b)
  in
  Stdio.print_string
    "# gh-ocannl-1149: the ambient OCANNL environment these tests run under, recorded by\n\
     # test/config/ocannl_ambient_env.ml so that a change of it reruns every rule depending on this\n\
     # file. Comments only: the run still reads the values from the environment.\n";
  List.iter tracked ~f:(fun (name, value) -> Stdio.printf "# %s=%s\n" name (String.escaped value))
