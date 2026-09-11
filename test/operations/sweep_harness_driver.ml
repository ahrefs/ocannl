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
  List.concat_map
    (fun root ->
      [
        Filename.concat (Filename.concat root "bin") "bash.exe";
        Filename.concat (Filename.concat (Filename.concat root "usr") "bin") "bash.exe";
      ])
    (windows_git_roots getenv)

let resolve_bash ~win32 ~git_bashes ~available =
  if win32 then List.find_opt available git_bashes else Some "bash"

(* This test runs on Windows only in the extended matrix, so keep the lookup's defining negative
   control host-independent. In particular, an executable bare [bash] on Windows is the WSL launcher
   in System32 on the runner image; it must not win, and must not become a fallback when Git Bash is
   absent. *)
let check_resolution () =
  let git_bash = "fixture-git-bash.exe" in
  let available path = String.equal path "bash" || String.equal path git_bash in
  let expect name got want =
    if not (Option.equal String.equal got want) then
      failwith
        (Printf.sprintf "%s: got %s, want %s" name
           (Option.value ~default:"<none>" got)
           (Option.value ~default:"<none>" want))
  in
  expect "Windows selects Git Bash even when bare bash is executable"
    (resolve_bash ~win32:true ~git_bashes:[ git_bash ] ~available)
    (Some git_bash);
  expect "Windows never falls back to bare bash"
    (resolve_bash ~win32:true ~git_bashes:[] ~available)
    None;
  expect "non-Windows keeps PATH lookup"
    (resolve_bash ~win32:false ~git_bashes:[] ~available)
    (Some "bash")

let executable path =
  try
    Unix.access path [ Unix.X_OK ];
    true
  with Unix.Unix_error _ -> false

let () =
  if Array.length Sys.argv <> 5 then (
    prerr_endline "usage: sweep_harness_driver HARNESS SWEEP AGGREGATE_SKIPS VERDICT_PROBE";
    exit 2);
  check_resolution ();
  let bash =
    resolve_bash ~win32:Sys.win32 ~git_bashes:(git_bash_candidates Sys.getenv_opt)
      ~available:(fun path -> Sys.file_exists path && executable path)
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
    Ir.Compiler_options.hiprtc ~hip_include_options:[] ~rocwmma_include_options:[]
      ~uses_rocwmma:false ~with_debug:false
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
      ~with_device_debug:false
    |> Ir.Compiler_options.render
  in
  let argv =
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
  let pid = Unix.create_process bash argv Unix.stdin Unix.stdout Unix.stderr in
  match snd (Unix.waitpid [] pid) with
  | Unix.WEXITED code -> exit code
  | Unix.WSIGNALED signal | Unix.WSTOPPED signal -> exit (128 + signal)
