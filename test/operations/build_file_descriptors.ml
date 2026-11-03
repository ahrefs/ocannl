(* Compiling with debug artifacts on holds no descriptor past the compile (gh-ocannl-1784). Every
   lowering writes [<name>-unoptimized.ll], [<name>.ll] and [<name>.cd] through
   [Utils.output_to_build_file], whose channel was opened eagerly and never closed: three
   descriptors per routine, which a long test such as [schedule_mma_matmul] ran out of under a
   256-descriptor limit (an ssh shell on macOS). The claim counts this process's open descriptors
   across N compiles of one routine shape; the negative control leaks one descriptor per step
   through the same counter, so a counter that cannot see growth cannot pass the claim. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Asgns = Ir.Assignments

let () = Utils.settings.output_debug_files_in_build_directory <- true

open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let compiles = 8

(* [/dev/fd] lists the calling process's descriptors on macOS and Linux; the listing's own
   descriptor is counted in every reading alike. *)
let open_descriptors () = Array.length (Stdlib.Sys.readdir "/dev/fd")

(* One routine shape, a fresh routine (and so fresh artifact files) per call; answers its name. *)
let compile_one ctx i =
  let x = TDSL.range 7 ~label:[ Printf.sprintf "fd_x%d" i ] in
  let%op y = (x *. 2.) + 1. in
  let comp = Train.forward y in
  let name = Asgns.get_name_exn comp.Asgns.asgns in
  let ctx, (_ : Context.routine) = Context.compile ctx comp Ir.Indexing.Empty in
  (ctx, name)

let () =
  Stdio.eprintf "build_file_descriptors: backend=%s (not part of the golden)\n%!" backend_name;
  let leg =
    Verdict.leg ~aggregation:`Environment ~when_:(not Stdlib.Sys.win32) ~on:"windows (no /dev/fd)"
      ()
  in
  let open_descriptors () = if Stdlib.Sys.win32 then 0 else open_descriptors () in
  Test_utils.Generated.init ~backend_name;
  (* Warm-up: backend start-up, the artifact directory's lock and the first compile's lazily opened
     driver state stay open for the process's lifetime, and are not per-compile growth. *)
  let ctx, (_ : string) = compile_one (Context.auto ()) 0 in
  let before = open_descriptors () in
  let names = ref [] in
  let (_ : Context.t) =
    List.fold
      (List.range 1 (compiles + 1))
      ~init:ctx
      ~f:(fun ctx i ->
        let ctx, name = compile_one ctx i in
        names := name :: !names;
        ctx)
  in
  let growth = open_descriptors () - before in
  Stdio.eprintf "%d compiles grew open descriptors by %d (not part of the golden)\n%!" compiles
    growth;
  in_legf ~leg "%d compiles with debug artifacts on grow open descriptors by fewer than %d" compiles
    compiles (fun label -> p label (growth < compiles));
  (* The fix must not stop the artifacts being written: each one holds its routine's code. *)
  p_all "every compile's optimized .ll artifact names its routine" !names ~f:(fun name ->
      String.is_substring (Test_utils.Generated.read ~ext:".ll" name) ~substring:name);
  p_all "every compile's .cd artifact is written" !names ~f:(fun name ->
      not (String.is_empty (Test_utils.Generated.read ~ext:".cd" name)));
  (* Negative control: the counter sees a writer that keeps one channel per step open. *)
  in_legf ~leg "negative control: %d leaking writes grow open descriptors by at least %d" compiles
    compiles (fun label ->
      let before = open_descriptors () in
      let leaked = List.init compiles ~f:(fun (_ : int) -> Stdio.Out_channel.create "/dev/null") in
      let leak_growth = open_descriptors () - before in
      List.iter leaked ~f:Stdio.Out_channel.close;
      p label (leak_growth >= compiles))
