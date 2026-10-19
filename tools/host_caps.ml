(* gh-ocannl-1371: one line saying which kind of host this is, as the cc backend's kernels see it:
   {!Context.Cc_backend.host_capabilities} -- the CPU, the ISA its SIMD probes find, the resolved
   [cc_vector_bytes] and register count, the worker pool, and the compiler and flags they were
   resolved under. Every CI build job prints it before building, because GitHub's x86 runners come
   with or without AVX-512 and the auto-probed vector width changes what the autotune pre-filter
   keeps; a red whose cause is the host is then visible from the log alone.

   Usage: run it from the directory whose [ocannl_config] the answer should resolve under (CI uses
   test/config, the configuration every test copies), e.g.

   {v cd test/config && dune exec --root ../.. ./tools/host_caps.exe v}

   [--save=PATH] also writes the line to PATH (CI saves it for the failure listing that ends a red
   job, tools/ci-failed-rules.sh). *)

open Base

let quote value =
  if
    String.is_empty value
    || String.exists value ~f:(fun c -> Char.is_whitespace c || Char.equal c '"')
  then "\"" ^ String.substr_replace_all value ~pattern:"\"" ~with_:"\\\"" ^ "\""
  else value

let () =
  let save =
    Array.find_map (Sys.get_argv ()) ~f:(fun arg -> String.chop_prefix arg ~prefix:"--save=")
  in
  let line =
    "OCANNL host: "
    ^ String.concat ~sep:" "
        (List.map (Context.Cc_backend.host_capabilities ()) ~f:(fun (key, value) ->
             key ^ "=" ^ quote value))
  in
  Stdio.print_endline line;
  Option.iter save ~f:(fun path -> Stdio.Out_channel.write_all path ~data:(line ^ "\n"))
