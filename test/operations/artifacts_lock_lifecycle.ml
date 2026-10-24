(* gh-ocannl-1462: [Utils.Artifacts_lock] across [fork] and [chdir], and the forced startup cleanup.

   gh-ocannl-1335's lock is a shared [Unix.lockf] a process takes on [<base>/<prefix>.ocannl-lock]
   when it first writes artifacts there, held by an open descriptor until it exits; a starting
   process's cleanup tries it exclusively and leaves a held directory alone. Covered here, each
   observed from a separate process (record locks never conflict within one process): - [fork]:
   POSIX does not carry record locks into a child, so a forked child that writes artifacts takes the
   lock afresh although it inherits the table of held locks -- after the process that forked it
   exits, the child alone still holds it, and once the child exits the lock is free; - [chdir]: a
   process that writes artifacts from two working directories holds both locks; -
   [force_clean_up_artifacts_on_startup]: a starting process's cleanup of a held directory leaves it
   alone, unless forced, when it empties the directory and names the holder on stderr.

   The probes and the starting processes are this executable, re-run in a mode named by its first
   argument. Coverage of existing behavior (fork, chdir) and of the new key (force). *)

open Base
open Verdict.Claims

let exe = Stdlib.Sys.executable_name
let prefix = Stdlib.Filename.remove_extension (Stdlib.Filename.basename exe)

(* Mode [probe <lock path>]: exit 0 when an exclusive lock on the file can be taken now, 1 when a
   lock someone holds blocks it, 2 when the file cannot be opened. *)
let probe_mode path =
  match Unix.openfile path [ Unix.O_RDWR ] 0 with
  | exception Unix.Unix_error _ -> Stdlib.exit 2
  | fd -> (
      match Unix.lockf fd Unix.F_TLOCK 0 with
      | () -> Stdlib.exit 0
      | exception Unix.Unix_error _ -> Stdlib.exit 1)

let () =
  match Array.to_list Stdlib.Sys.argv with
  | _ :: "probe" :: path :: _ -> probe_mode path
  | _ :: "startup" :: _ ->
      (* The startup cleanup ran while this module's dependencies initialized. *)
      Stdlib.exit 0
  | _ -> ()

let wait_status pid = match snd (Unix.waitpid [] pid) with Unix.WEXITED n -> n | _ -> -1

(* Runs this executable with [args] from the original working directory, [stderr] to [err]. *)
let run_self ?(err = Unix.stderr) args =
  let pid = Unix.create_process exe (Array.of_list (exe :: args)) Unix.stdin Unix.stdout err in
  wait_status pid

let held path = run_self [ "probe"; path ] = 1
let free path = run_self [ "probe"; path ] = 0
let home = Stdlib.Sys.getcwd ()
let made_dirs = ref []

let fresh_dir tag =
  let dir =
    Stdlib.Filename.concat
      (Stdlib.Filename.get_temp_dir_name ())
      (Printf.sprintf "ocannl_%s_%d_%s" prefix (Utils.os_process_id ()) tag)
  in
  (try Stdlib.Sys.mkdir dir 0o777 with Stdlib.Sys_error _ -> ());
  made_dirs := dir :: !made_dirs;
  dir

(* Best effort: on Windows this process's own open lock descriptors keep their files. *)
let rec remove_tree path =
  match Unix.lstat path with
  | exception Unix.Unix_error _ -> ()
  | { Unix.st_kind = Unix.S_DIR; _ } -> (
      Array.iter (Stdlib.Sys.readdir path) ~f:(fun f -> remove_tree (Stdlib.Filename.concat path f));
      try Unix.rmdir path with Unix.Unix_error _ -> ())
  | _ -> ( try Stdlib.Sys.remove path with Stdlib.Sys_error _ -> ())

(* Writes an artifact from [dir] (taking its lock), returning the lock's and the artifact's
   paths. *)
let write_artifact_in dir name =
  Stdlib.Sys.chdir dir;
  let path = Utils.diagn_log_file name in
  Stdio.Out_channel.write_all path ~data:"artifact\n";
  let lock = Utils.Artifacts_lock.path "log_files" in
  Stdlib.Sys.chdir home;
  (lock, Stdlib.Filename.concat dir path)

let fork_claims =
  [
    "fork: the forked child wrote its artifacts";
    "fork: the forking holder exited";
    "fork: with the forking holder gone, its child still holds the lock";
    "fork: the holder query names the forked child";
    "fork: once the child exits, the lock is free";
  ]

let () =
  Verdict.case "fork" @@ fun () ->
  if Stdlib.Sys.win32 then
    (* Windows has no fork: each claim is announced as skipped, keeping the golden uniform. *)
    List.iter fork_claims ~f:(Verdict.skipped ~aggregation:`Environment ~backend:"windows")
  else
    let dir = fresh_dir "fork" in
    let lock = Stdlib.Filename.concat dir (Printf.sprintf "log_files/%s.ocannl-lock" prefix) in
    let ready_r, ready_w = Unix.pipe () and release_r, release_w = Unix.pipe () in
    Stdlib.flush_all ();
    match Unix.fork () with
    | 0 -> (
        (* The forking holder: takes the lock, forks, and exits at once. *)
        Unix.close ready_r;
        Unix.close release_w;
        Stdlib.Sys.chdir dir;
        ignore (Utils.diagn_log_file "holder" : string);
        match Unix.fork () with
        | 0 ->
            (* Its child: inherits the table naming the parent's lock, writes artifacts, says so,
               and holds on until released (or until the test process is gone). *)
            ignore (Utils.diagn_log_file "child" : string);
            let pid = Printf.sprintf "%d\n" (Utils.os_process_id ()) in
            ignore (Unix.write_substring ready_w pid 0 (String.length pid) : int);
            (try ignore (Unix.read release_r (Bytes.create 1) 0 1 : int)
             with Unix.Unix_error _ -> ());
            Unix._exit 0
        | _ -> Unix._exit 0)
    | holder ->
        Unix.close ready_w;
        Unix.close release_r;
        let child_pid =
          let buf = Bytes.create 32 in
          let n = Unix.read ready_r buf 0 32 in
          Int.of_string_opt (String.strip (Bytes.To_string.sub buf ~pos:0 ~len:n))
        in
        pf "fork: the forked child wrote its artifacts" (Option.is_some child_pid);
        pf "fork: the forking holder exited" (wait_status holder = 0);
        pf "fork: with the forking holder gone, its child still holds the lock" (held lock);
        (* This process holds no lock on that file, so [F_GETLK] reports the child's. *)
        let queried =
          let fd = Unix.openfile lock [ Unix.O_RDONLY ] 0 in
          Exn.protect ~f:(fun () -> Utils.lock_holder_pid fd) ~finally:(fun () -> Unix.close fd)
        in
        pf "fork: the holder query names the forked child"
          (Option.equal Int.equal (Some queried) child_pid);
        Unix.close release_w;
        (* The child exits on EOF; the probe waits until its lock is gone, a bounded number of
           times. *)
        let rec settle n =
          if free lock then true
          else if n = 0 then false
          else (
            Unix.sleepf 0.05;
            settle (n - 1))
        in
        pf "fork: once the child exits, the lock is free" (settle 200)

let () =
  Verdict.case "chdir" @@ fun () ->
  let dir_a = fresh_dir "chdir_a" and dir_b = fresh_dir "chdir_b" in
  let lock_a, _ = write_artifact_in dir_a "in_a" in
  let lock_b, _ = write_artifact_in dir_b "in_b" in
  pf "chdir: the two working directories have distinct lock files"
    (not (String.equal lock_a lock_b));
  pf "chdir: the lock of the first directory is held" (held lock_a);
  pf "chdir: the lock of the second directory is held too" (held lock_b)

let () =
  Verdict.case "force" @@ fun () ->
  let dir = fresh_dir "force" in
  let _lock, artifact = write_artifact_in dir "victim.c" in
  let err_file = Stdlib.Filename.concat dir "startup.stderr" in
  (* A process of this executable starting in [dir], whose startup cleanup meets this process's
     lock. *)
  let start ~force =
    let err = Unix.openfile err_file [ Unix.O_WRONLY; Unix.O_CREAT; Unix.O_TRUNC ] 0o644 in
    Stdlib.Sys.chdir dir;
    let status =
      Exn.protect
        ~f:(fun () ->
          run_self ~err
            ([ "startup"; "--ocannl_clean_up_log_files_on_startup=true" ]
            @ if force then [ "--ocannl_force_clean_up_artifacts_on_startup=true" ] else []))
        ~finally:(fun () ->
          Stdlib.Sys.chdir home;
          Unix.close err)
    in
    (status, Stdio.In_channel.read_all err_file)
  in
  let status, _ = start ~force:false in
  pf "force: an unforced start ran" (status = 0);
  pf "force: unforced, the held directory's artifact stays" (Stdlib.Sys.file_exists artifact);
  let status, err = start ~force:true in
  pf "force: a forced start ran" (status = 0);
  pf "force: forced, the held directory's artifact is removed"
    (not (Stdlib.Sys.file_exists artifact));
  pf "force: the forced cleanup says it emptied a held directory"
    (String.is_substring err ~substring:"force_clean_up_artifacts_on_startup: emptying");
  pf "force: it names the holder where the platform can tell"
    (String.is_substring err
       ~substring:
         (if Stdlib.Sys.win32 then "holder pid unknown"
          else Printf.sprintf "held by pid %d," (Utils.os_process_id ())));
  pf "force: the lock file itself stays"
    (Stdlib.Sys.file_exists
       (Stdlib.Filename.concat dir (Printf.sprintf "log_files/%s.ocannl-lock" prefix)))

let () = List.iter !made_dirs ~f:remove_tree
