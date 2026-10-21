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
   environment holds, filtered by `Utils.env_var_in_ambient_record` -- which `env_var_deps` asks of
   every `(env_var …)` a dune file declares, refusing a declaration the record already carries
   (gh-ocannl-1425) -- so a key the pipeline will start reading tomorrow is covered today.

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
   holding a newline cannot end the comment. The record opens with an empty line, so a configuration
   it is appended to that lacks a final newline keeps its last value intact.

   Written to the file [record] names, never to stdout: `Utils`' own initialization may print to
   stdout under an ambient tracing setting (`OCANNL_LOG_LEVEL_UTILS` and friends), and that must not
   become part of every test's configuration.

   gh-ocannl-1038: the host is the other half of the ambient world a verdict can depend on and dune
   cannot see -- a rocWMMA tree appearing at a path the HIP backend already searches, an OpenMP
   runtime installed under an unchanged compiler. `--host-record=PATH` names the record
   `tools/host_caps.exe --record` wrote under its own `(universe)` rule, and it is appended after
   the environment, so `ocannl_ambient` -- the one file every test's configuration already carries
   or lists -- changes when either does. Its lines must all be comments: anything else would be a
   setting smuggled into every test's configuration, so one that is not is refused. *)

open Base

let record = "ocannl_ambient"

let () =
  let host_record =
    Array.find_map Stdlib.Sys.argv ~f:(fun arg -> String.chop_prefix arg ~prefix:"--host-record=")
  in
  let tracked =
    Unix.environment () |> Array.to_list
    |> List.filter_map ~f:(fun binding ->
        match String.lsplit2 binding ~on:'=' with
        (* An empty value is unset at every source ([Utils.read_env_var]). *)
        | None | Some (_, "") -> None
        | Some (name, value) -> Option.some_if (Utils.env_var_in_ambient_record name) (name, value))
    |> List.sort ~compare:(fun (a, _) (b, _) -> String.compare a b)
  in
  Stdio.Out_channel.with_file ~binary:true record ~f:(fun out ->
      Stdio.Out_channel.output_string out
        "\n\
         # gh-ocannl-1149: the ambient OCANNL environment these tests run under, recorded by\n\
         # test/config/ocannl_ambient_env.ml so that a change of it reruns every rule depending on \
         this\n\
         # file. Comments only: the run still reads the values from the environment.\n";
      List.iter tracked ~f:(fun (name, value) ->
          Stdio.Out_channel.fprintf out "# %s=%s\n" name (String.escaped value));
      Option.iter host_record ~f:(fun path ->
          let lines = Stdio.In_channel.read_lines path in
          List.iter lines ~f:(fun line ->
              if not (String.is_prefix line ~prefix:"#") then
                failwith
                  (Printf.sprintf
                     "ocannl_ambient_env: %s holds a line that is not a comment, which would \
                      become a setting of every test's configuration: %S"
                     path line);
              Stdio.Out_channel.output_string out (line ^ "\n"))))
