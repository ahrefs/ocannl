(* gh-ocannl-1197: which temporary files one cc compilation leaves in the system temp directory, on
   a successful compile and on a compile the C compiler rejects.

   A compilation owns three temporaries there: the source it hands the compiler (written by
   [Utils.open_build_file], or under debug files a private copy beside the informational
   [build_files/] one), the compiler log, and the library the compiler writes. The log was always
   removed; this test pins that the source and the library are too, unless a setting asks for them:
   [output_debug_files_in_build_directory] retains the source and the library on every path, and
   [output_dlls_in_build_directory] retains the library. The [build_files/] copy is a debug output,
   never a temporary, and stays.

   The test runs in a private temp directory holding an unrelated sentinel per artifact suffix, and
   counts what a compile adds there. The success mode then RUNS the routine and checks its values:
   the library is unlinked as soon as it is mapped, and the mapping, not the path, is what the
   kernel executes from. The [compiler] mode's rule renames [float] to an undeclared type, so the
   compiler rejects the generated code; strict classification makes that rejection a typed fatal.

   Windows refuses to delete a mapped DLL: there the library is removed only after an unload, which
   the OpenMP arm never performs, so the success-mode library count is gated off Windows (the source
   count, the sentinel, and the compiler-rejection counts are not). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module Outcome = Ir.Schedule_outcome

let mode = Stdlib.Sys.argv.(1)
let () = if not (List.mem [ "success"; "compiler" ] mode ~equal:String.equal) then invalid_arg mode
let library_suffix = if Sys.win32 then ".dll" else ".so"
let sentinels = [ "unrelated.c"; "unrelated" ^ library_suffix; "unrelated.log" ]

let cause_of_failure = function
  | Outcome.Classified { cause; _ } -> Some cause
  | Outcome.Fatal { cause; _ } -> cause

let run directory =
  let snapshot () =
    Stdlib.Sys.readdir directory |> Array.to_list |> List.sort ~compare:String.compare
  in
  let before = snapshot () in
  let debug = Utils.settings.output_debug_files_in_build_directory in
  let dll_output =
    Utils.get_global_flag ~default:false ~arg_name:"output_dlls_in_build_directory"
  in
  Stdio.printf "mode: %s\n" mode;
  Stdio.printf "library artifacts requested: %s\n" (if dll_output then "yes" else "no");
  Stdio.printf "debug artifacts requested: %s\n" (if debug then "yes" else "no");
  let ctx = Context.auto () in
  Stdio.printf "backend: %s\n" (Context.backend_name ctx);
  let name = "gh1197_ownership" in
  let x = TDSL.range 16 in
  let%op y = (2. *. x) + 1. in
  Train.set_materialized y.value;
  let comp = Train.forward y in
  let outcome =
    Context.compile_outcome ~name ~provenance:Outcome.Candidate ~candidate:"ownership probe" ctx
      comp Ir.Indexing.Empty
  in
  (* Snapshot right after the compile: nothing has run, so this is what the compile itself left. *)
  let after = snapshot () in
  let added = List.filter after ~f:(fun f -> not (List.mem before f ~equal:String.equal)) in
  let removed = List.filter before ~f:(fun f -> not (List.mem after f ~equal:String.equal)) in
  let count suffix = List.count added ~f:(String.is_suffix ~suffix) in
  let sources = count ".c" and libraries = count library_suffix and logs = count ".log" in
  Stdio.eprintf "added after the compile (not part of the golden): %s\n%!"
    (String.concat ~sep:" " added);
  let rejection =
    match outcome with
    | Ok _ -> None
    | Error failure -> (
        match cause_of_failure failure with
        | Some (Outcome.Backend_rejected { stage; detail; _ }) ->
            (* The compiler output that follows is long; its first lines identify the run. *)
            Stdio.eprintf "rejection detail head (not part of the golden):\n%s\n%!"
              (String.concat ~sep:"\n" (List.take (String.split_lines detail) 3));
            Some (stage, detail)
        | Some _ | None ->
            Stdio.eprintf "unexpected failure shape\n%!";
            None)
  in
  let values =
    match outcome with
    | Ok (ctx, routine) ->
        let ctx = Context.run ctx routine in
        Some (Context.get_values ctx y.value)
    | Error _ -> None
  in
  let success = String.equal mode "success" in
  if success then (
    p "the compile succeeded" (Option.is_some values);
    p_alli
      (if debug || dll_output then "the routine computes 2x + 1 from its retained library"
       else "the routine computes 2x + 1 from its unlinked library")
      (Option.value values ~default:[||] |> Array.to_list)
      ~f:(fun i v -> Float.equal v (Float.of_int ((2 * i) + 1)))
      ~min:16)
  else (
    p "the C compiler rejected the generated code"
      (Option.exists rejection ~f:(fun (stage, _) -> String.equal stage "compiler"));
    p "the rejection says the source was removed exactly when debug files are not requested"
      (Option.exists rejection ~f:(fun (_, detail) ->
           Bool.equal (not debug)
             (String.is_substring detail ~substring:"removed its temporary copy"))));
  p "the compiler log is removed" (logs = 0);
  p "the source is retained exactly when debug files are requested"
    (sources = if debug then 1 else 0);
  (* A rejected compile produced no library; a successful one's is retained exactly when asked. *)
  let expected_libraries = if success && (debug || dll_output) then 1 else 0 in
  gated ~aggregation:`Environment
    ~when_:(not (success && Sys.win32))
    ~on:"Windows, which refuses to delete a mapped DLL"
    "the library is retained exactly when requested" (libraries = expected_libraries);
  p "nothing else is added" (List.length added = sources + libraries + logs);
  p_empty "no file present before the compile is removed" ~over:before removed;
  p_all "every unrelated sentinel keeps its content" sentinels ~f:(fun f ->
      String.equal (Stdio.In_channel.read_all (Stdlib.Filename.concat directory f)) f)

let () =
  let original = Stdlib.Filename.get_temp_dir_name () in
  let dir = Stdlib.Filename.temp_file "ocannl-gh1197-" "" in
  Stdlib.Sys.remove dir;
  Unix.mkdir dir 0o700;
  Stdlib.Filename.set_temp_dir_name dir;
  List.iter sentinels ~f:(fun f ->
      Stdio.Out_channel.write_all (Stdlib.Filename.concat dir f) ~data:f);
  Exn.protect
    ~f:(fun () -> run dir)
    ~finally:(fun () ->
      Stdlib.Filename.set_temp_dir_name original;
      Array.iter (Stdlib.Sys.readdir dir) ~f:(fun file ->
          try Stdlib.Sys.remove (Stdlib.Filename.concat dir file) with Sys_error _ -> ());
      try Unix.rmdir dir with Unix.Unix_error _ -> ())
