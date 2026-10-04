(* The provenance guarantees of [Test_utils.Generated] (gh-ocannl-655), pinned executably.

   That module is what ~30 tests' structural assertions now rest on, so its preconditions are
   load-bearing in a way that is easy to break silently: every one of them fires only when something
   has ALREADY gone wrong, which is precisely the kind of code that rots unobserved. Two review
   rounds on the introducing PR found three such defects by inspection alone -- a swallowed deletion
   failure, a directory-scoping test that misread a valid configuration, and a sweep that was not
   this process's to perform. This probe exists so that the next one fails a test instead.

   Each mode is one guarantee. The modes that must SUCCEED are run directly by the dune rule and
   claim normally, so that a module which refused everything could not pass them. The modes that
   must be REFUSED are run by the [refusals] mode, as child processes whose streams it captures --
   and it claims on what it captured.

   Capturing is not incidental (gh-ocannl-692). A refusal reports itself through [Verdict], which
   prints [FAIL: ...] on stdout and stderr and [FAILED: n checks did not hold.] at teardown. Those
   words are this repository's failure marker by convention -- the one [verdict_ratchet] enforces at
   the source level, and the one [grep FAIL] over a suite log rests on -- so a child that inherited
   the suite's streams put four of them into the log of a GREEN run, and the only way to tell them
   from a real failure was to read the line after each. A designed refusal and a blessed regression
   must not be spelled the same way; that is the same argument as gh-ocannl-601, one level up.

   The capture also buys the stronger check, which is why it beat the alternative of a marker
   reserved for expected-failure children. An exit status alone says only that the child failed:
   under [with-accepted-exit-codes] a mode whose refusal had stopped firing still passed as long as
   something else exited 1 -- a mistyped mode name, a missing config, a failure in the mode's own
   setup. Each refusal is now asserted BY ITS MESSAGE, so it is this guarantee that is being
   observed and not merely this process's misfortune.

   The artifacts are written by hand rather than compiled: what is under test is how this module
   decides whether a file on disk belongs to this run, and a real backend would only make the setup
   slower and the failure modes harder to stage. *)

open Base
module Generated = Test_utils.Generated

let backend_name = "cc"
let dir () = Utils.build_files_dir ()
let path routine = Stdlib.Filename.concat (dir ()) (routine ^ ".c")

(* The symlink modes below must NOT call [dir ()] first: it creates the directory whose place the
   link has to take. They resolve the same path the way Generated does instead. *)
let artifacts_root = "build_files"

let scoped_dir () =
  Stdlib.Filename.concat artifacts_root
    (Stdlib.Filename.remove_extension (Stdlib.Filename.basename Stdlib.Sys.executable_name))

let link_target () = Stdlib.Filename.concat artifacts_root "gp_symlink_target"
let precious () = Stdlib.Filename.concat (link_target ()) "gp_precious.c"
let ignore_unix f x = try f x with Unix.Unix_error _ -> ()

(* Stands in for the backend emitting a kernel. *)
let emit routine contents = Stdio.Out_channel.write_all (path routine) ~data:contents
let run_child ?(args = []) mode = Fresh_process.run (mode :: args)

let child_refused ~message child =
  let ok = Fresh_process.matches ~exit:1 ~contains:[ message ] child in
  if not ok then Fresh_process.report ~label:("refusal: " ^ message) child;
  ok

let refused claim ~message child = Verdict.p claim (child_refused ~message child)

let () =
  let mode =
    match Array.to_list Stdlib.Sys.argv with
    | _ :: m :: _ when not (String.is_prefix m ~prefix:"--") -> m
    | _ -> failwith "generated_provenance: expected a mode argument"
  in
  match mode with
  (* === Must succeed === *)

  (* The happy path. Pins that the refusals below are not vacuous: a module that refused every read
     would satisfy all of them. *)
  | "fresh" ->
      Generated.init ~backend_name;
      emit "gp_fresh" "kernel body: fresh\n";
      Verdict.p "a freshly emitted artifact reads back"
        (String.is_substring (Generated.read "gp_fresh") ~substring:"fresh")
  (* Arming is what turns per-candidate attribution into a checked fact: the same two-kernel
     sequence that is refused unarmed below must go through when each compile is armed. *)
  | "armed_overwrite" ->
      Generated.init ~backend_name;
      emit "gp_twice" "kernel body: candidate 1\n";
      ignore (Generated.read "gp_twice" : string);
      Generated.arm "gp_twice";
      emit "gp_twice" "kernel body: candidate 2\n";
      Verdict.p "an armed second reading is attributed to its own compile"
        (String.is_substring (Generated.read "gp_twice") ~substring:"candidate 2")
  (* === The driver for the modes that must be refused ===

     One claim per guarantee, each naming the refusal it requires. The order is the order the dune
     rule used to run them in: these modes share one artifact directory and several stage a state
     for the next, so it is a sequence rather than a set. *)
  | "refusals" ->
      refused "an artifact left by an earlier run is swept rather than read as this run's"
        ~message:"no generated source for routine gp_stale" (run_child "stale");
      refused "a routine this run never emitted is refused rather than answered"
        ~message:"no generated source for routine gp_never_emitted" (run_child "missing");
      refused "an unarmed second reading under one routine name is refused as an overwrite"
        ~message:"generated source for routine gp_twice changed between reads"
        (run_child "overwrite");
      refused "an armed compile that emitted nothing is refused rather than credited"
        ~message:"no generated source for routine gp_twice" (run_child "armed_no_emission");
      refused "a read before init is refused rather than answered from the directory"
        ~message:
          "Generated.read gp_anything: Test_utils.Generated.init ~backend_name must be called \
           before any compile"
        (run_child "uninitialized");
      (* Both spellings: the flat layout and a named prefix are one refusal in the source, and a
         reader of this list should not have to know that to see that both are covered. *)
      refused "a flat configured build_files_prefix is refused"
        ~message:"Generated.init: build_files_prefix is set"
        (run_child "explicit_prefix" ~args:[ "--ocannl_build_files_prefix=." ]);
      refused "a named configured build_files_prefix is refused"
        ~message:"Generated.init: build_files_prefix is set"
        (run_child "explicit_prefix" ~args:[ "--ocannl_build_files_prefix=gp_shared_dir" ])
  (* === Must be refused (run by the mode above, which asserts the refusal) === *)
  (* The issue's own failure: an artifact left by an EARLIER run, which the readers this replaced
     accepted because it happened to exist. Written before init, so init's sweep must remove it. *)
  | "stale" ->
      let stale = path "gp_stale" in
      Stdio.Out_channel.write_all stale ~data:"kernel body: from a previous run\n";
      Generated.init ~backend_name;
      Stdio.printf "read after sweep: %S\n" (Generated.read "gp_stale")
  (* A kernel that stopped being emitted at all: folded to a constant, renamed, fissioned. *)
  | "missing" ->
      Generated.init ~backend_name;
      Stdio.printf "read of a never-emitted routine: %S\n" (Generated.read "gp_never_emitted")
  (* Two kernels read under one routine name, unarmed: the same-process overwrite. *)
  | "overwrite" ->
      Generated.init ~backend_name;
      emit "gp_twice" "kernel body: candidate 1\n";
      ignore (Generated.read "gp_twice" : string);
      emit "gp_twice" "kernel body: candidate 2\n";
      ignore (Generated.read "gp_twice" : string);
      Stdio.printf "second unarmed reading completed\n"
  (* Armed, but the compile emitted nothing -- the case [arm] exists for. The previous candidate's
     kernel is gone, so nothing can be credited to this one. *)
  | "armed_no_emission" ->
      Generated.init ~backend_name;
      emit "gp_twice" "kernel body: candidate 1\n";
      ignore (Generated.read "gp_twice" : string);
      Generated.arm "gp_twice";
      Stdio.printf "read after an armed compile emitted nothing: %S\n" (Generated.read "gp_twice")
  (* Before init the directory may hold anything, so no read may be answered. *)
  | "uninitialized" -> Stdio.printf "read before init: %S\n" (Generated.read "gp_anything")
  (* === The symlinked artifact directory ===

     [Sys.is_directory] follows symlinks, so a symlinked build_files/<exe>/ reads as an ordinary
     directory and a sweep would unlink real files in the link's TARGET -- outside the artifact tree
     entirely. Two things make this mode's shape what it is.

     Refusal ABORTS the process, so [init] cannot be called in-process and observed; it runs in a
     child, whose exit status carries the refusal. And symlink creation is a privilege on Windows
     (Developer Mode, or SeCreateSymbolicLink) that the CI matrix does not necessarily have, so the
     staging is attempted rather than assumed: where links cannot be made the leg is reported as
     skipped instead of failing for the host's permissions. Both halves are then checked here -- the
     child refused, AND the planted file in the link target is still there -- so this stays a single
     mode that must SUCCEED. *)
  | "symlink" -> (
      ignore_unix (fun d -> Unix.mkdir d 0o777) artifacts_root;
      ignore_unix (fun d -> Unix.mkdir d 0o777) (link_target ());
      Stdio.Out_channel.write_all (precious ()) ~data:"a file that is not this test's to delete\n";
      let scoped = scoped_dir () in
      (* Whatever an earlier mode left here has to go, so the link can take its place. *)
      ignore_unix Unix.unlink scoped;
      (try
         Array.iter (Stdlib.Sys.readdir scoped) ~f:(fun e ->
             ignore_unix Unix.unlink (Stdlib.Filename.concat scoped e));
         ignore_unix Unix.rmdir scoped
       with Stdlib.Sys_error _ -> ());
      (* Relative, so it resolves inside build_files/ wherever the tree is checked out.
         [~to_dir:true] is ignored on Unix and load-bearing on Windows, where a link created without
         it is a FILE symlink that would not stand in for a directory. *)
      match Unix.symlink ~to_dir:true "gp_symlink_target" scoped with
      | exception Unix.Unix_error (err, _, _) ->
          Stdio.eprintf "symbolic links unavailable here (%s)\n" (Unix.error_message err);
          Verdict.skipped ~aggregation:`Environment ~backend:backend_name
            "a symlinked artifact directory is refused rather than followed into"
      | () ->
          let child =
            (* The child inherits this process's environment, in which the dune rule has cleared
               OCANNL_BUILD_FILES_PREFIX. That matters: init refuses every configured prefix, so a
               child that inherited an ambient one would exit 1 without ever looking at the staged
               link. Requiring the SYMLINK refusal by its message would now catch that -- but the
               environment stays cleared rather than left to be diagnosed, since a probe that has to
               fail to be right about its own setup is one nobody can read. An empty
               --ocannl_build_files_prefix= argument would NOT do instead: an empty command-line
               value reads as "not given" and falls through to the environment. *)
            run_child "symlink_child"
          in
          let link_refused =
            child_refused
              ~message:
                "is a symbolic link; refusing to use it, since following it would delete files \
                 outside the artifact tree"
              child
          in
          let survived = Stdlib.Sys.file_exists (precious ()) in
          (* Restore a real directory for whatever runs next. *)
          ignore_unix Unix.unlink scoped;
          Verdict.p "a symlinked artifact directory is refused rather than followed into"
            (link_refused && survived))
  (* Run by the mode above, in a child process, because refusing a directory aborts.

     The arm attempt after init is the point: an exit status alone cannot tell a refusal that
     STOPPED the run from one that merely recorded a failure and returned, and the difference is
     whether the deletions that follow go through the link. Under a terminating refusal this line is
     never reached; under a non-terminating one it unlinks the planted file in the link target, and
     the parent's survival check fails. That regression shipped once -- a lost edit left the branch
     calling Verdict.fail -- so the probe discriminates it rather than trusting the exit code. *)
  | "symlink_child" ->
      Generated.init ~backend_name;
      Generated.arm "gp_precious";
      Stdio.printf "init returned on a symlinked artifact directory\n"
  (* A configured prefix names a directory this process does not own: another executable can be
     given the same one. Deleting there could take a running test's kernel out from under it, and
     without deletion there is no write signal independent of timestamp granularity -- a
     deterministic compile re-emits byte-identical bytes, possibly within one mtime quantum. So the
     configuration is refused rather than supported approximately, and that refusal is pinned here
     for both spellings. *)
  | "explicit_prefix" ->
      Generated.init ~backend_name;
      Stdio.printf "init returned under a configured build_files_prefix\n"
  | m -> failwith ("generated_provenance: unknown mode " ^ m)
