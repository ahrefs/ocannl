(* gh-ocannl-884: a regime transition and a store in different processes share the permanent record
   lock. The transition child pauses after sweeping the old entries and staging the new stamp, but
   before publishing that stamp; the writer child then attempts a store and reports from its
   entry-commit hook. It must not reach that hook until the transition is released, and its entry
   must remain after both cache opens finish.

   The rendezvous makes the interleaving deliberate rather than scheduler-dependent. Every wait is
   bounded in monotonic seconds, so a missing hook or a child that stops making progress fails
   instead of hanging the test. The lock's negative control is the production defect this exists to
   catch: removing [Unix.lockf fd Unix.F_LOCK 0] from [Schedule_cache.with_cache_open] makes the
   parent's direct lock-state probe report no exclusion at the writer's attempt boundary, and this
   test exits 1. *)

open Base
module FI = Ir.Resource_fault_injection
module SC = Ir.Schedule_cache
open Verdict.Claims

let cache_dir = "autotune_cache_regime_race"
let old_key = "old"
let writer_key = "concurrent-writer"
let transition_mode = "--regime-transition-child"
let writer_mode = "--regime-writer-child"

let entry backend : SC.entry =
  {
    version = SC.entry_version;
    backend;
    numerics = SC.numerics_tag ();
    codegen = None;
    objective = None;
    source_digest = "gh884-source";
    saved = [];
    segments = None;
    finer_fission = None;
    best_ms = 1.;
    baseline_ms = 2.;
    default_ms = None;
    mma_best_ms = None;
    default_fingerprint = None;
  }

let stamp_file = Stdlib.Filename.concat cache_dir SC.regime_stamp_filename
let lock_file = Stdlib.Filename.concat cache_dir SC.regime_lock_filename
let entry_file key = Stdlib.Filename.concat cache_dir (key ^ ".sexp")
let marker name = Stdlib.Filename.concat cache_dir ("rendezvous-" ^ name)

let clean_dir () =
  if Stdlib.Sys.file_exists cache_dir && Stdlib.Sys.is_directory cache_dir then (
    Array.iter (Stdlib.Sys.readdir cache_dir) ~f:(fun name ->
        Stdlib.Sys.remove (Stdlib.Filename.concat cache_dir name));
    Stdlib.Sys.rmdir cache_dir)

let write_stamp version = Stdio.Out_channel.write_all stamp_file ~data:(Int.to_string version ^ "\n")

let write_entry key value =
  Stdio.Out_channel.write_all (entry_file key) ~data:(Sexp.to_string_hum (SC.sexp_of_entry value))

let touch name = Stdio.Out_channel.write_all (marker name) ~data:"ready\n"
let since counter = Mtime.Span.to_float_ns (Mtime_clock.count counter) /. 1e9

let await ?(seconds = 10.) predicate =
  let started = Mtime_clock.counter () in
  let rec loop () =
    if predicate () then true
    else if Float.(since started > seconds) then false
    else (
      Unix.sleepf 0.002;
      loop ())
  in
  loop ()

let await_marker ?seconds name = await ?seconds (fun () -> Stdlib.Sys.file_exists (marker name))
let child_exit code = Stdlib.exit code

let transition_child () =
  let reached = ref false in
  let opened =
    FI.with_callback
      (fun point ->
        if FI.equal_point point FI.Schedule_cache_before_regime_commit then (
          reached := true;
          touch "stamp-ready";
          if not (await_marker "transition-release") then child_exit 2))
      ~f:(fun () -> SC.lookup ~dir:cache_dir ~key:(Some old_key))
  in
  if !reached && Option.is_none opened then (
    touch "transition-done";
    child_exit 0)
  else child_exit 3

let writer_child () =
  touch "writer-ready";
  if not (await_marker "writer-go") then child_exit 2;
  let reached_lock = ref false in
  let reached_commit = ref false in
  FI.with_callback
    (fun point ->
      if FI.equal_point point FI.Schedule_cache_before_lock then (
        reached_lock := true;
        touch "writer-at-lock";
        if not (await_marker "writer-lock-go") then child_exit 2)
      else if FI.equal_point point FI.Schedule_cache_before_commit then (
        reached_commit := true;
        touch "entry-commit";
        if not (await_marker "writer-release") then child_exit 2))
    ~f:(fun () -> SC.store ~dir:cache_dir ~key:(Some writer_key) (entry "writer"));
  if !reached_lock && !reached_commit then (
    touch "writer-done";
    child_exit 0)
  else child_exit 3

let () =
  match Array.to_list Stdlib.Sys.argv with
  | _ :: mode :: _ when String.equal mode transition_mode -> transition_child ()
  | _ :: mode :: _ when String.equal mode writer_mode -> writer_child ()
  | _ -> ()

type child = { pid : int; mutable status : Unix.process_status option }

let spawn mode =
  let exe = Stdlib.Sys.executable_name in
  let pid = Unix.create_process exe [| exe; mode |] Unix.stdin Unix.stdout Unix.stderr in
  { pid; status = None }

let poll_child child =
  match child.status with
  | Some _ -> true
  | None ->
      let waited, status = Unix.waitpid [ Unix.WNOHANG ] child.pid in
      if waited = 0 then false
      else (
        child.status <- Some status;
        true)

let reap child =
  ignore (await (fun () -> poll_child child) : bool);
  child.status

let terminate child =
  if not (poll_child child) then (
    (try Unix.kill child.pid Stdlib.Sys.sigkill with Unix.Unix_error _ -> ());
    ignore (reap child : Unix.process_status option))

let exited_zero child =
  match reap child with
  | Some (Unix.WEXITED 0) -> true
  | Some (Unix.WEXITED _ | Unix.WSIGNALED _ | Unix.WSTOPPED _) | None -> false

let record_lock_held_by_other_process () =
  let fd = Unix.openfile lock_file [ Unix.O_RDWR ] 0o600 in
  Exn.protect
    ~finally:(fun () -> Unix.close fd)
    ~f:(fun () ->
      match Unix.lockf fd Unix.F_TEST 0 with
      | () -> false
      | exception Unix.Unix_error ((Unix.EACCES | Unix.EAGAIN), _, _) -> true
      | exception exn ->
          Stdio.eprintf "record-lock probe failed: %s\n" (Exn.to_string exn);
          false)

let () =
  clean_dir ();
  Stdlib.Sys.mkdir cache_dir 0o755;
  write_entry old_key (entry "old");
  write_stamp (SC.cache_regime_version - 1);
  let transition = spawn transition_mode in
  let writer = ref None in
  Exn.protect
    ~finally:(fun () ->
      Option.iter !writer ~f:terminate;
      terminate transition;
      clean_dir ())
    ~f:(fun () ->
      p "the transition reaches the window before publishing its new stamp"
        (await_marker "stamp-ready");
      p "the paused transition has swept the old entry"
        (not (Stdlib.Sys.file_exists (entry_file old_key)));
      p "the paused transition has not published the new stamp"
        (String.equal
           (String.strip (Stdio.In_channel.read_all stamp_file))
           (Int.to_string (SC.cache_regime_version - 1)));

      let concurrent = spawn writer_mode in
      writer := Some concurrent;
      p "the concurrent writer is ready before it attempts the cache open"
        (await_marker "writer-ready");
      touch "writer-go";
      p "the concurrent writer reaches the record-lock attempt while stamping is paused"
        (await_marker "writer-at-lock");
      p "the transition's record lock excludes the writer at its attempt boundary"
        (record_lock_held_by_other_process ());
      p "the waiting writer has not published its entry"
        (not (Stdlib.Sys.file_exists (entry_file writer_key)));

      touch "writer-lock-go";
      touch "transition-release";
      p "the released transition finishes publishing the current stamp"
        (await_marker "transition-done");
      p "the writer reaches entry commit after the transition releases the lock"
        (await_marker "entry-commit");
      p "the writer retains the record lock through its entry commit"
        (record_lock_held_by_other_process ());
      touch "writer-release";
      p "the released writer finishes its store" (await_marker "writer-done");
      p "both cache-opening processes exit successfully"
        (exited_zero transition && exited_zero concurrent);
      p "the writer's subsequent entry survives the completed transition"
        (match SC.lookup ~dir:cache_dir ~key:(Some writer_key) with
        | Some value -> String.equal value.SC.backend "writer"
        | None -> false))
