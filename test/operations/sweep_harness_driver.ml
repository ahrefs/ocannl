let windows_git_roots getenv =
  let under name suffix =
    match getenv name with
    | Some dir when not (String.equal dir "") -> [ List.fold_left Filename.concat dir suffix ]
    | _ -> []
  in
  under "ProgramFiles" [ "Git" ] @ under "ProgramW6432" [ "Git" ]
  @ under "ProgramFiles(x86)" [ "Git" ]
  @ under "LOCALAPPDATA" [ "Programs"; "Git" ]
  @ [ {|C:\Progra~1\Git|} ]

let git_bash_candidates getenv =
  List.map
    (fun root -> Filename.concat (Filename.concat root "bin") "bash.exe")
    (windows_git_roots getenv)

let resolve_bash ~win32 ~git_bashes ~available =
  if win32 then List.find_opt available git_bashes else Some "bash"

let path_available ~win32 ~stat_kind ~x_ok path =
  match stat_kind path with Some Unix.S_REG -> win32 || x_ok path | _ -> false

(* On macOS, pin [awk] to [mawk] (gh-ocannl-2414) so Dune-run shell tests exercise Linux mawk
   semantics rather than macOS's BWK [/usr/bin/awk]. Never fall back to BWK awk when [mawk] is
   missing. Non-Darwin hosts keep their default [awk] untouched. *)
let darwin_mawk_fallbacks = [ "/opt/homebrew/bin/mawk"; "/usr/local/bin/mawk" ]

let missing_mawk_message =
  "sweep_harness_driver: macOS requires mawk to pin awk for Dune shell tests (macOS /usr/bin/awk \
   is BWK awk and hides Linux mawk defects; no fallback). Install it with: brew install mawk"

let mawk_candidates ~path_env ~fallbacks ~getcwd =
  let from_path =
    match path_env with
    | None -> []
    | Some path ->
        String.split_on_char ':' path
        |> List.map (fun dir ->
            let abs_dir =
              if String.equal dir "" then getcwd ()
              else if Filename.is_relative dir then Filename.concat (getcwd ()) dir
              else dir
            in
            Filename.concat abs_dir "mawk")
  in
  from_path @ fallbacks

let resolve_darwin_mawk ~darwin ~path_env ~fallbacks ~getcwd ~available =
  if not darwin then `Not_needed
  else
    match List.find_opt available (mawk_candidates ~path_env ~fallbacks ~getcwd) with
    | Some mawk -> `Pinned mawk
    | None -> `Missing

let prepend_path ~dir env =
  let replaced = ref false in
  let env =
    Array.map
      (fun entry ->
        if String.starts_with ~prefix:"PATH=" entry then (
          replaced := true;
          let rest = String.sub entry 5 (String.length entry - 5) in
          if String.equal rest "" then "PATH=" ^ dir else "PATH=" ^ dir ^ ":" ^ rest)
        else entry)
      env
  in
  if !replaced then env else Array.append env [| "PATH=" ^ dir |]

let rec waitpid_eintr pid =
  try snd (Unix.waitpid [] pid) with Unix.Unix_error (Unix.EINTR, _, _) -> waitpid_eintr pid

let cleanup_awk_dir dir =
  (try Unix.unlink (Filename.concat dir "awk") with Unix.Unix_error _ -> ());
  try Unix.rmdir dir with Unix.Unix_error _ -> ()

(* These controls run on every host, keeping both the Windows Git Bash lookup and the macOS mawk pin
   host-independent. In particular, an executable bare [bash] on Windows is the WSL launcher in
   System32 on the runner image and must not win; on macOS, an executable [/usr/bin/awk] must never
   become a fallback when [mawk] is absent. *)
let check_resolution () =
  let program_files = "fixture-program-files" in
  let getenv name = if String.equal name "ProgramFiles" then Some program_files else None in
  let git_bash =
    Filename.concat (Filename.concat (Filename.concat program_files "Git") "bin") "bash.exe"
  in
  let raw_bash =
    Filename.concat
      (Filename.concat (Filename.concat (Filename.concat program_files "Git") "usr") "bin")
      "bash.exe"
  in
  let git_bashes = git_bash_candidates getenv in
  let available path = String.equal path "bash" || String.equal path git_bash in
  let expect name got want =
    if not (Option.equal String.equal got want) then
      failwith
        (Printf.sprintf "%s: got %s, want %s" name
           (Option.value ~default:"<none>" got)
           (Option.value ~default:"<none>" want))
  in
  expect "Windows selects Git Bash even when bare bash is executable"
    (resolve_bash ~win32:true ~git_bashes ~available)
    (Some git_bash);
  expect "Windows never falls back to bare or raw usr/bin bash"
    (resolve_bash ~win32:true ~git_bashes ~available:(fun path ->
         String.equal path "bash" || String.equal path raw_bash))
    None;
  expect "non-Windows keeps PATH lookup"
    (resolve_bash ~win32:false ~git_bashes:[] ~available)
    (Some "bash");
  let regular _ = Some Unix.S_REG in
  let directory _ = Some Unix.S_DIR in
  if
    not
      (path_available ~win32:true ~stat_kind:regular
         ~x_ok:(fun _ -> failwith "Windows must not call Unix.access X_OK")
         git_bash)
  then failwith "Windows rejects a regular Git Bash executable";
  if path_available ~win32:true ~stat_kind:directory ~x_ok:(fun _ -> true) git_bash then
    failwith "Windows accepts a Git Bash directory";
  if path_available ~win32:false ~stat_kind:regular ~x_ok:(fun _ -> false) git_bash then
    failwith "non-Windows accepts a non-executable Git Bash file";
  let expect_mawk name got want =
    let show = function
      | `Not_needed -> "Not_needed"
      | `Pinned p -> "Pinned " ^ p
      | `Missing -> "Missing"
    in
    if got <> want then failwith (Printf.sprintf "%s: got %s, want %s" name (show got) (show want))
  in
  let getcwd () = "/fixture/cwd" in
  let fallbacks = [ "/opt/homebrew/bin/mawk"; "/usr/local/bin/mawk" ] in
  expect_mawk "non-Darwin leaves awk unpinned even when mawk is absent"
    (resolve_darwin_mawk ~darwin:false ~path_env:(Some "/usr/bin:/bin") ~fallbacks ~getcwd
       ~available:(fun _ -> false))
    `Not_needed;
  expect_mawk "Darwin selects PATH mawk ahead of fallbacks and absolutizes relative dirs"
    (resolve_darwin_mawk ~darwin:true ~path_env:(Some "rel/bin:/usr/bin") ~fallbacks ~getcwd
       ~available:(fun p ->
         String.equal p "/fixture/cwd/rel/bin/mawk" || String.equal p "/opt/homebrew/bin/mawk"))
    (`Pinned "/fixture/cwd/rel/bin/mawk");
  expect_mawk "Darwin treats empty PATH component as current working directory"
    (resolve_darwin_mawk ~darwin:true ~path_env:(Some ":/usr/bin:/bin") ~fallbacks ~getcwd
       ~available:(fun p ->
         String.equal p "/fixture/cwd/mawk" || String.equal p "/opt/homebrew/bin/mawk"))
    (`Pinned "/fixture/cwd/mawk");
  expect_mawk "Darwin treats empty PATH string as current working directory"
    (resolve_darwin_mawk ~darwin:true ~path_env:(Some "") ~fallbacks ~getcwd ~available:(fun p ->
         String.equal p "/fixture/cwd/mawk" || String.equal p "/opt/homebrew/bin/mawk"))
    (`Pinned "/fixture/cwd/mawk");
  expect_mawk "Darwin falls back to Homebrew mawk when PATH lacks mawk"
    (resolve_darwin_mawk ~darwin:true ~path_env:(Some "/usr/bin:/bin") ~fallbacks ~getcwd
       ~available:(fun p -> String.equal p "/opt/homebrew/bin/mawk"))
    (`Pinned "/opt/homebrew/bin/mawk");
  expect_mawk "Darwin refuses when mawk is absent and never falls back to /usr/bin/awk"
    (resolve_darwin_mawk ~darwin:true ~path_env:(Some "/usr/bin:/bin") ~fallbacks ~getcwd
       ~available:(fun p -> String.equal p "/usr/bin/awk" || String.equal p "awk"))
    `Missing;
  expect_mawk "Darwin ignores non-regular or non-executable mawk entries"
    (resolve_darwin_mawk ~darwin:true ~path_env:(Some "/dir/bin:/noexec/bin") ~fallbacks ~getcwd
       ~available:
         (path_available ~win32:false
            ~stat_kind:(fun p ->
              if String.equal p "/dir/bin/mawk" then Some Unix.S_DIR
              else if String.equal p "/noexec/bin/mawk" then Some Unix.S_REG
              else None)
            ~x_ok:(fun _ -> false)))
    `Missing;
  let contains_substring ~haystack ~needle =
    let hlen = String.length haystack and nlen = String.length needle in
    let rec loop i =
      if i + nlen > hlen then false
      else if String.equal (String.sub haystack i nlen) needle then true
      else loop (i + 1)
    in
    loop 0
  in
  if not (contains_substring ~haystack:missing_mawk_message ~needle:"brew install mawk") then
    failwith "missing_mawk_message must name brew install mawk";
  let expect_env name got want =
    if got <> want then
      failwith
        (Printf.sprintf "%s: got [%s], want [%s]" name
           (String.concat "; " (Array.to_list got))
           (String.concat "; " (Array.to_list want)))
  in
  expect_env "prepend_path prefixes existing PATH"
    (prepend_path ~dir:"/pin" [| "HOME=/h"; "PATH=/usr/bin:/bin" |])
    [| "HOME=/h"; "PATH=/pin:/usr/bin:/bin" |];
  expect_env "prepend_path avoids trailing colon on empty PATH"
    (prepend_path ~dir:"/pin" [| "PATH=" |])
    [| "PATH=/pin" |];
  expect_env "prepend_path adds PATH when absent"
    (prepend_path ~dir:"/pin" [| "HOME=/h" |])
    [| "HOME=/h"; "PATH=/pin" |]

let executable path =
  try
    Unix.access path [ Unix.X_OK ];
    true
  with Unix.Unix_error _ -> false

let () =
  let check_mawk_only = Array.length Sys.argv = 2 && String.equal Sys.argv.(1) "--check-mawk" in
  if (not check_mawk_only) && Array.length Sys.argv <> 2 && Array.length Sys.argv <> 5 then (
    prerr_endline
      "usage: sweep_harness_driver (--check-mawk | HARNESS [SWEEP AGGREGATE_SKIPS VERDICT_PROBE])";
    exit 2);
  check_resolution ();
  let stat_kind path = try Some (Unix.stat path).st_kind with Unix.Unix_error _ -> None in
  let awk_pin =
    resolve_darwin_mawk ~darwin:Utils.host_is_darwin ~path_env:(Sys.getenv_opt "PATH")
      ~fallbacks:darwin_mawk_fallbacks ~getcwd:Sys.getcwd
      ~available:(path_available ~win32:false ~stat_kind ~x_ok:executable)
  in
  (if check_mawk_only then
     match awk_pin with
     | `Not_needed ->
         print_endline "non-darwin";
         exit 0
     | `Missing ->
         prerr_endline missing_mawk_message;
         exit 2
     | `Pinned mawk ->
         let resolved = try Unix.realpath mawk with Unix.Unix_error _ -> mawk in
         let digest =
           try Digest.to_hex (Digest.file resolved)
           with Sys_error msg ->
             prerr_endline ("sweep_harness_driver: cannot fingerprint mawk: " ^ msg);
             exit 2
         in
         Printf.printf "%s\t%s\t%s\n" mawk resolved digest;
         exit 0);
  let bash =
    resolve_bash ~win32:Sys.win32
      ~git_bashes:(git_bash_candidates Sys.getenv_opt)
      ~available:(path_available ~win32:Sys.win32 ~stat_kind ~x_ok:executable)
  in
  let bash =
    match bash with
    | Some bash -> bash
    | None ->
        Verdict.skipped ~aggregation:`Environment ~backend:"Git Bash unavailable"
          "sweep harness requires Git Bash on Windows";
        exit 0
  in
  let metal_options =
    Ir.Compiler_options.metal ~routine_logging:false ~math_api:Modern_split
    |> Ir.Compiler_options.render_metal
  in
  let hip_options =
    Ir.Compiler_options.hiprtc ~target_archs:[] ~hip_include_options:[] ~rocwmma_include_options:[]
      ~uses_rocwmma:false ~fast_math:true ~with_debug:false ()
    |> Ir.Compiler_options.render
  in
  (* The two slots [Cuda_backend] discovers at compile time (the CUDA_PATH include directory and
     [gpu_arch_options]' architecture target) are sentinels here, as in
     arrayjit/test/test_cuda_compile_options.ml: the harness pins that the fingerprint carries the
     rendered vector whole, and a value no installed toolkit produces cannot match by
     coincidence. *)
  let nvrtc_options =
    Ir.Compiler_options.nvrtc ~cuda_include_options:[ "-I/sentinel/cuda/include" ]
      ~arch_options:[ "--gpu-architecture=compute_999" ]
      ~fast_math:true ~with_device_debug:false ()
    |> Ir.Compiler_options.render
  in
  let argv =
    if Array.length Sys.argv = 2 then [| bash; Sys.argv.(1) |]
    else
      [|
        bash;
        Sys.argv.(1);
        Sys.argv.(2);
        Sys.argv.(3);
        Sys.argv.(4);
        metal_options;
        hip_options;
        nvrtc_options;
      |]
  in
  let code =
    match awk_pin with
    | `Missing ->
        prerr_endline missing_mawk_message;
        2
    | `Not_needed -> (
        let pid = Unix.create_process bash argv Unix.stdin Unix.stdout Unix.stderr in
        match waitpid_eintr pid with
        | Unix.WEXITED code -> code
        | Unix.WSIGNALED signal | Unix.WSTOPPED signal -> 128 + signal)
    | `Pinned mawk ->
        let dir = Filename.temp_dir "sweep-harness-awk-" "" in
        Fun.protect
          ~finally:(fun () -> cleanup_awk_dir dir)
          (fun () ->
            Unix.symlink mawk (Filename.concat dir "awk");
            let env = prepend_path ~dir (Unix.environment ()) in
            let pid = Unix.create_process_env bash argv env Unix.stdin Unix.stdout Unix.stderr in
            match waitpid_eintr pid with
            | Unix.WEXITED code -> code
            | Unix.WSIGNALED signal | Unix.WSTOPPED signal -> 128 + signal)
  in
  exit code
