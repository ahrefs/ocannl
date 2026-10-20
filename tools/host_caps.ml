(* gh-ocannl-1371: one line saying which kind of host this is, as the backends' own probes see it:
   {!Context.host_capabilities} -- the CPU, the ISA the cc backend's SIMD probes find, the resolved
   [cc_vector_bytes] and register count, the worker pool, the parallel-Grid runtime, and the
   compiler and flags they were resolved under, then the host-side probes of each linked GPU backend
   (gh-ocannl-1038). Every CI build job prints it before building, because GitHub's x86 runners come
   with or without AVX-512 and the auto-probed vector width changes what the autotune pre-filter
   keeps; a red whose cause is the host is then visible from the log alone.

   Usage: run it from the directory whose [ocannl_config] the answer should resolve under (CI uses
   test/config, the configuration every test copies), e.g.

   {v cd test/config && dune exec --root ../.. ./tools/host_caps.exe v}

   [--save=PATH] also writes the line to PATH (CI saves it for the failure listing that ends a red
   job, tools/ci-failed-rules.sh).

   [--record=PATH] writes the same capabilities to PATH as configuration comments, one [# host
   key=value] line each (gh-ocannl-1038): test/config/dune runs this under [(universe)] and folds
   the file into the ambient record every test's configuration carries, so a capability that moves
   reruns the tests instead of leaving dune to serve a verdict computed on the host as it was.
   Comments, because the configuration reader skips them: the record only makes the host a
   dependency, the run still probes for itself. One key per line so a change reads as a one-line
   diff. *)

open Base

let quote value =
  if
    String.is_empty value
    || String.exists value ~f:(fun c -> Char.is_whitespace c || Char.equal c '"')
  then "\"" ^ String.substr_replace_all value ~pattern:"\"" ~with_:"\\\"" ^ "\""
  else value

let () =
  let flag prefix =
    Array.find_map (Sys.get_argv ()) ~f:(fun arg -> String.chop_prefix arg ~prefix)
  in
  let capabilities = Context.host_capabilities () in
  let line =
    "OCANNL host: "
    ^ String.concat ~sep:" "
        (List.map capabilities ~f:(fun (key, value) -> key ^ "=" ^ quote value))
  in
  Stdio.print_endline line;
  Option.iter (flag "--save=") ~f:(fun path -> Stdio.Out_channel.write_all path ~data:(line ^ "\n"));
  Option.iter (flag "--record=") ~f:(fun path ->
      Stdio.Out_channel.with_file ~binary:true path ~f:(fun out ->
          Stdio.Out_channel.output_string out
            "# gh-ocannl-1038: the host capabilities the backends' own probes found, recorded by\n\
             # tools/host_caps.exe so that a change of them reruns every rule depending on this \
             file.\n";
          List.iter capabilities ~f:(fun (key, value) ->
              Stdio.Out_channel.fprintf out "# host %s=%s\n" key (String.escaped value))))
