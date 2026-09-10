(* Concurrent writers against {!Utils.Atomic_file} (gh-ocannl-780).

   The property under test is the one the schedule cache and the checkpoint writer depend on: a
   reader of a published path observes a COMPLETE payload — the previous one or a new one — never a
   torn one, and never a missing file. The hazards are three, and there is a leg for each: two
   writers sharing one staging artifact (uniqueness), a writer that fails between staging and commit
   (failure cleanup), and a writer killed in that window (crash-stale cleanup).

   The WORK here is a fixed number of publications and reads; every place a domain waits for another
   is bounded in seconds, under an explicit cap that FAILS the run rather than hanging it. Counting
   turns of a sleeping loop instead was the same bug in three places: [Unix.sleepf] on Windows
   truncates a sub-millisecond request to no sleep at all and rounds everything else up to a 15.6 ms
   timer tick, so a cap of "20,000 turns of half a millisecond" was ten seconds on Linux and macOS
   and a fraction of one there. Two things make the stress leg able to fail rather than merely pass:

   - Its readers are compared against the EXACT set of payloads the writers published, not against a
   shape. A shape check accepted a torn read whose truncation happened to land on the length lattice
   (Codex P2, round 2); the payloads are self-describing now, and the shape check is kept only to
   control that counterexample explicitly. - Readers and writers are made to overlap by a rendezvous
   rather than by hoping the scheduler interleaves them. On one core the readers could otherwise
   finish every read against the seeded file before a writer ran at all, and pass while publication
   truncated the target in place (Codex P2, round 2). Each reader now reads once before any writer
   proceeds past its first round, and keeps reading until every writer has recorded that it
   finished; every reader records that exact exit condition.

   The stress leg is controlled executably in the other direction too: a child deliberately
   truncates the published path, then runs the SAME gated-read claim. The parent requires that claim
   to fail, so making the readers fair to the writer scheduler cannot make a broken publication
   pass. *)

open Base
module AF = Utils.Atomic_file
module Ignore = Test_utils.Cache_dir_scan

let dir = "atomic_file_race_dir"
let target = Stdlib.Filename.concat dir "published.bin"

let listing () =
  match Stdlib.Sys.readdir dir with
  | exception _ -> []
  | entries -> List.sort ~compare:String.compare (Array.to_list entries)

let staging_leftovers () = List.filter (listing ()) ~f:AF.is_staging_file
let generated_staging_names = ref []

let rec remove_tree path =
  match Unix.lstat path with
  | { Unix.st_kind = Unix.S_DIR; _ } ->
      Array.iter (Stdlib.Sys.readdir path) ~f:(fun entry ->
          remove_tree (Stdlib.Filename.concat path entry));
      Unix.rmdir path
  | _ -> Unix.unlink path
  | exception Unix.Unix_error _ -> ()

(* Recursively, because not every fixture here is a file: the publication leg builds a staging TREE
   and publishes it as one, and the sweep leg creates a directory of its own. A run interrupted
   between planting one of those and removing it leaves a DIRECTORY behind, and a file-only reset
   walked past it -- so the next run died at [Unix.mkdir] with EEXIST, or at [publish_staged]'s
   rename into a non-empty target, instead of starting from the empty directory every leg here
   assumes (gh-ocannl-803). Removal is best-effort per entry, so one entry the filesystem refuses
   does not leave the rest of the leftovers in place. *)
let reset_dir () =
  AF.ensure_dir dir;
  List.iter (listing ()) ~f:(fun name ->
      try remove_tree (Stdlib.Filename.concat dir name) with _ -> ())

(* At startup, before any leg runs: whatever the previous run left, this one begins from an empty
   directory. The per-leg resets below keep the legs independent of each other; this one keeps the
   RUN independent of the last one. *)
let () = reset_dir ()

(* The child half of the rerun control below. What recovers a rerun is the initialization directly
   above -- what happens when the PROCESS starts -- and a control that calls [reset_dir] itself pins
   the HELPER instead: it would stay green with that startup call deleted, or moved after the first
   leg, because the per-leg resets mask its absence (Codex P2, round 1). So the control plants
   leftovers and starts a fresh process, which lands here: this argument makes a run do its startup
   and nothing else, reporting where it ran and what the scratch directory held once the startup
   reset had run.

   An argv marker rather than an environment variable, so nothing ambient can put a run into this
   mode -- OCANNL's commandline scan leaves an argument it is not addressed by alone, and there is
   no undeclared variable for dune to serve a stale result across. It must sit immediately after the
   startup reset and before every leg, or what it reports would be some later leg's reset. *)
let startup_probe_arg = "--startup-probe"

let () =
  if Array.exists Stdlib.Sys.argv ~f:(String.equal startup_probe_arg) then (
    (* The cwd first: it is what lets the parent claim this process cleared ITS scratch directory,
       so that "the leftovers are gone" cannot be satisfied by a child that ran somewhere else and
       truthfully saw nothing. *)
    Stdio.printf "%s\n" (Unix.getcwd ());
    List.iter (listing ()) ~f:(Stdio.printf "%s\n");
    Stdio.Out_channel.flush Stdio.stdout;
    Stdlib.exit 0)

(* Start a fresh process over whatever the scratch directory currently holds, and report what its
   startup made of it. The child inherits this process's working directory, so it resolves [dir] to
   the same place -- which it says out loud, above. *)
let run_startup_probe () =
  let read_fd, write_fd = Unix.pipe () in
  let exe = Stdlib.Sys.executable_name in
  let pid = Unix.create_process exe [| exe; startup_probe_arg |] Unix.stdin write_fd Unix.stderr in
  Unix.close write_fd;
  let ic = Unix.in_channel_of_descr read_fd in
  let output = Stdio.In_channel.input_all ic in
  Stdlib.close_in ic;
  let _, status = Unix.waitpid [] pid in
  match String.split_lines output with
  | cwd :: entries -> (status, Some cwd, List.filter entries ~f:(Fn.non String.is_empty))
  | [] -> (status, None, [])

(* A payload is SELF-DESCRIBING: it names its writer, its round and its own body length, and ends
   with a terminator. Length alone is not enough — a run of one character truncated by a multiple of
   the step is a shorter round's payload exactly (Codex P2, round 2) — but a body that disagrees
   with its declared length, or a payload missing its terminator, is torn however it was cut. Mixing
   two writers' bytes shows up in the body's characters. *)
let base_length = 997
let length_step = 41

let payload ~writer ~round =
  let tag = Char.of_int_exn writer in
  let body_length = base_length + (length_step * round) in
  Printf.sprintf "%c|%d|%d|%s|END" tag round body_length (String.make body_length tag)

let is_well_formed data =
  match String.split data ~on:'|' with
  | [ tag; round; declared; body; "END" ] -> (
      String.length tag = 1
      &&
      match (Stdlib.int_of_string_opt round, Stdlib.int_of_string_opt declared) with
      | Some round, Some declared ->
          round >= 0
          && declared = base_length + (length_step * round)
          && String.length body = declared
          && String.for_all body ~f:(Char.equal tag.[0])
      | _ -> false)
  | _ -> false

(* Bounded wait: the deterministic-interleaving leg needs one domain to reach a known state before
   the other proceeds, and a wait that cannot end must fail rather than hang the suite.

   Bounded in SECONDS, not in turns of a loop. `20_000` turns of a 0.5 ms sleep reads as ten seconds
   and is ten seconds on Linux and macOS, but [Unix.sleepf] on Windows cannot sleep for less than
   the system timer tick and truncates a sub-millisecond request to nothing at all -- measured
   there, 0.5 ms returns immediately while everything from 1 ms up costs a full 15.6 ms. So on
   Windows the same loop was a bare busy-spin that gave up in well under a second, while burning the
   core its peer domain needed to make the progress it was waiting for. That is how "the blocked
   writer waited rather than timing out" failed on Windows and nowhere else: the writer did not time
   out of anything the test meant to give it. *)
let wait_seconds = 10.0

(* Long enough to be a real sleep on Windows (one timer tick) and short enough to be a fine-grained
   one everywhere else. *)
let spin_pause = 0.002

(* Monotonic, for the reason `publish_staged` gives: every bound here is a DURATION, and a wall
   clock an NTP step moves would cut one short or extend it by the size of the adjustment. *)
let since counter = Mtime.Span.to_float_ns (Mtime_clock.count counter) /. 1e9

let await ?(seconds = wait_seconds) predicate =
  let started = Mtime_clock.counter () in
  let rec loop () =
    if predicate () then true
    else if Float.(since started > seconds) then false
    else (
      Unix.sleepf spin_pause;
      loop ())
  in
  loop ()

let await_flag flag = await (fun () -> Atomic.get flag)

(* What one read of the published path found -- three outcomes, because they are three different
   facts and `try Some (read_all target) with _ -> None` reported them as one.

   [Absent] is a failure of the property this whole module exists for: the publish let the path
   cease to exist. A refusal says nothing about the bytes at all. On Windows the C runtime opens
   files without FILE_SHARE_DELETE, so an open and a commit that overlap refuse EACH OTHER --
   `atomic_file.ml` documents the writer's half of that, and this is the reader's. Conflated, every
   claim below read "the path went missing" on Windows whenever a read merely collided with a
   rename, which is what they did.

   Opened through [Unix.openfile] rather than [In_channel.read_all] so that the refusal can be TOLD
   from the absence: [Sys_error] carries only a message, while the [Unix_error] says [ENOENT] or
   does not, which is the whole distinction. *)
type observation = Bytes_read of string | Absent | Refused of string

let open_published () =
  match Unix.openfile target [ Unix.O_RDONLY ] 0o400 with
  | fd ->
      let ic = Unix.in_channel_of_descr fd in
      Stdlib.set_binary_mode_in ic true;
      Exn.protect
        ~finally:(fun () -> Stdlib.close_in_noerr ic)
        ~f:(fun () -> Bytes_read (Stdio.In_channel.input_all ic))
  | exception Unix.Unix_error (Unix.ENOENT, _, _) -> Absent
  | exception Unix.Unix_error (error, _, _) -> Refused (Unix.error_message error)

(* How long a reader keeps trying past a refusal. A reader of a concurrently published file has to
   retry one on Windows, and this reader stands in for a real one -- but the retry is COUNTED and
   claimed on below rather than assumed away, so a refusal that never resolves is still a reported
   failure and not a silent wait. *)
let read_retry_seconds = 1.0
let refusals_retried = Atomic.make 0

let observe_published () =
  let started = lazy (Mtime_clock.counter ()) in
  let rec attempt () =
    match open_published () with
    | (Bytes_read _ | Absent) as found -> found
    | Refused _ as refused ->
        Atomic.incr refusals_retried;
        if Float.(since (force started) > read_retry_seconds) then refused
        else (
          Unix.sleepf spin_pause;
          attempt ())
  in
  attempt ()

(* The sequential legs read at moments they arranged, where the only interesting answer is which
   bytes are there. *)
let read_published () =
  match observe_published () with Bytes_read data -> Some data | Absent | Refused _ -> None

let gated_read_claim = "every reader's gated read saw a whole payload while one was half-written"

let observation_is_published published = function
  | Bytes_read data -> Hash_set.mem published data
  | Absent | Refused _ -> false

(* The negative control runs in a child because the condition it exists to demonstrate must FAIL a
   real [Verdict] claim. Capturing its streams keeps that designed [FAIL:] out of this passing run's
   log, while checking the claim text keeps an unrelated setup failure from satisfying the control.

   The broken writer truncates the target and leaves only half a payload before the read. Closing it
   first makes the counterexample portable to Windows, where an open writer handle can affect which
   reader opens are permitted; the defect under test is the published path containing those partial
   bytes, not a platform sharing rule. *)
let broken_publication_arg = "--broken-publication-probe"

let () =
  if Array.exists Stdlib.Sys.argv ~f:(String.equal broken_publication_arg) then (
    reset_dir ();
    let seed = payload ~writer:(Char.to_int 'a') ~round:0 in
    let whole = payload ~writer:(Char.to_int 'b') ~round:1 in
    AF.write_all ~path:target ~data:seed ();
    Stdio.Out_channel.write_all target ~data:(String.prefix whole (String.length whole / 2));
    let published = Hash_set.of_list (module String) [ seed; whole ] in
    Verdict.p_all gated_read_claim [ observe_published () ] ~f:(observation_is_published published);
    Stdlib.exit 0)

let describe_status = function
  | Unix.WEXITED n -> Printf.sprintf "exited %d" n
  | Unix.WSIGNALED n -> Printf.sprintf "was killed by signal %d" n
  | Unix.WSTOPPED n -> Printf.sprintf "was stopped by signal %d" n

let ignore_unix f x = try f x with Unix.Unix_error _ -> ()

let run_broken_publication_probe () =
  let exe = Stdlib.Sys.executable_name in
  let capture suffix = Stdlib.Filename.temp_file "atomic_file_broken" suffix in
  let out_path = capture ".out" and err_path = capture ".err" in
  let open_capture path = Unix.openfile path [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
  let out = open_capture out_path and err = open_capture err_path in
  let pid = Unix.create_process exe [| exe; broken_publication_arg |] Unix.stdin out err in
  let _, status = Unix.waitpid [] pid in
  Unix.close out;
  Unix.close err;
  let stdout_text = Stdio.In_channel.read_all out_path in
  let stderr_text = Stdio.In_channel.read_all err_path in
  ignore_unix Unix.unlink out_path;
  ignore_unix Unix.unlink err_path;
  (status, stdout_text, stderr_text)

let () =
  let status, stdout_text, stderr_text = run_broken_publication_probe () in
  let failure_line = gated_read_claim ^ ": false" in
  let controlled =
    (match status with
      | Unix.WEXITED 1 -> true
      | Unix.WEXITED _ | Unix.WSIGNALED _ | Unix.WSTOPPED _ -> false)
    && String.is_substring stdout_text ~substring:failure_line
    && String.is_substring stderr_text ~substring:("FAIL: " ^ failure_line)
  in
  if not controlled then
    Stdio.eprintf
      "the broken-publication child %s without failing the gated-read claim. Its capture:\n%s%s\n"
      (describe_status status) stdout_text stderr_text;
  Verdict.p "a publication that truncates the target fails the gated-read claim (negative control)"
    controlled

(* Control: the reader's verdict discriminates. Without this, "every read was complete" is a claim
   about a predicate nobody checked. *)
let () =
  let good = payload ~writer:(Char.to_int 'a') ~round:3 in
  Verdict.p "the completeness test accepts a whole payload" (is_well_formed good);
  Verdict.p "the completeness test rejects a truncated payload"
    (not (is_well_formed (String.prefix good 512)));
  (* The counterexample the length-lattice shape check accepted: round 3's payload cut to exactly
     round 1's length. Under a shape check keyed on length alone the two are indistinguishable. *)
  Verdict.p "the completeness test rejects a truncation aligned to the length lattice"
    (not
       (is_well_formed
          (String.prefix good (String.length (payload ~writer:(Char.to_int 'a') ~round:1)))));
  Verdict.p "the completeness test rejects a payload missing its terminator"
    (not (is_well_formed (String.drop_suffix good 4)));
  Verdict.p "the completeness test rejects a payload whose body mixes two writers"
    (not
       (is_well_formed (String.prefix good (String.length good - 8) ^ String.make 4 'b' ^ "|END")));
  Verdict.p "the completeness test rejects an empty file" (not (is_well_formed ""))

(* A staged-but-uncommitted writer is invisible, and it obstructs nobody: while one publish sits in
   its commit window, a whole second publish to the same path succeeds, and the first still commits
   its own complete payload afterwards. This is the deterministic form of the race the stress leg
   below runs at volume. *)
let () =
  reset_dir ();
  let seed = payload ~writer:(Char.to_int 'a') ~round:0 in
  AF.write_all ~path:target ~data:seed ();
  let staged = Atomic.make false in
  let release = Atomic.make false in
  let blocked = payload ~writer:(Char.to_int 'b') ~round:1 in
  let waited = ref false in
  let writer =
    Domain.spawn (fun () ->
        AF.write_all ~path:target ~data:blocked
          ~before_commit:(fun () ->
            Atomic.set staged true;
            waited := await_flag release)
          ())
  in
  Verdict.p "the blocked writer reaches its commit window" (await_flag staged);
  let during_stage = read_published () in
  Verdict.p "a staged payload is invisible until it commits"
    (Option.equal String.equal during_stage (Some seed));
  Verdict.p "exactly the blocked writer's staging file is present"
    (List.length (staging_leftovers ()) = 1);
  let overtaking = payload ~writer:(Char.to_int 'c') ~round:2 in
  AF.write_all ~path:target ~data:overtaking ();
  Verdict.p "a competing publish commits while another writer holds its staging file"
    (Option.equal String.equal (read_published ()) (Some overtaking));
  Verdict.p "the competing publish did not disturb the blocked writer's staging file"
    (List.length (staging_leftovers ()) = 1);
  Atomic.set release true;
  Domain.join writer;
  Verdict.p "the released writer commits its own complete payload"
    (Option.equal String.equal (read_published ()) (Some blocked));
  Verdict.p "the blocked writer waited rather than timing out" !waited;
  Verdict.p_none "no staging file survives the interleaving" (listing ()) ~f:AF.is_staging_file

(* The stress leg. Fixed counts and a rendezvous, so the interleaving is arranged rather than hoped
   for: every writer publishes its first round and then WAITS for every reader to have read, and
   every reader keeps reading until the writers are done. Both halves are needed — the first makes
   readers observe a directory under publication, the second keeps them reading through it. *)
let writers = 4
let rounds = 150
let readers = 2
let reads_per_reader = 400

(* A ceiling on the reader loop, so a starved writer domain cannot turn "read until the writers
   finish" into an unbounded run. In SECONDS rather than in reads, for the reason [await] gives and
   one more: how many reads a reader gets through before the writers finish is a ratio between two
   speeds, and on Windows a publish that has to wait out a sharing violation is orders of magnitude
   slower than a read, so a ceiling counted in reads is a different amount of patience on every
   platform. Reaching it is a pathology, and the overlap claim below is what then fails. *)
let reader_ceiling_seconds = 120.0
let total_publications = writers * rounds

type reader_stop =
  | Writers_finished of { reads : int; concluded : int; committed : int }
  | Reader_ceiling of { reads : int; writers_finished : int; concluded : int; committed : int }
  | Reader_not_started

type reader_report = { during_write : observation; seen : observation list; stop : reader_stop }

let () =
  reset_dir ();
  let seed = payload ~writer:(Char.to_int 'a') ~round:0 in
  AF.write_all ~path:target ~data:seed ();
  (* Every payload any writer will publish, plus the seed: what a reader observes must be one of
     these EXACTLY, not merely a string shaped like one. *)
  let published = Hash_set.create (module String) in
  Hash_set.add published seed;
  for i = 0 to writers - 1 do
    for round = 0 to rounds - 1 do
      Hash_set.add published (payload ~writer:(Char.to_int 'a' + i) ~round)
    done
  done;
  let refusals = Array.create ~len:writers 0 in
  (* What a reader domain that never wrote its slot leaves behind. Chosen so it fails every claim it
     reaches rather than reading as a satisfied one. *)
  let no_reads =
    {
      during_write = Refused "the reader domain recorded nothing";
      seen = [];
      stop = Reader_not_started;
    }
  in
  let observations = Array.create ~len:readers no_reads in
  let committed = Atomic.make 0 in
  (* Publications that have ENDED, either way. This, and not [committed], is what a reader records
     beside each read: "the writers are done" is a fact about the work being over, not about all of
     it having succeeded, and a publication the platform refused (see the liveness claims below)
     never advances [committed] -- so keying a reader on that counter made a Windows refusal fail a
     claim about the READER, in a leg where refusals are expected and separately accounted for. *)
  let concluded = Atomic.make 0 in
  let readers_read = Atomic.make 0 in
  let writers_finished = Atomic.make 0 in
  let half_written = Atomic.make false in
  let writers_met = Array.create ~len:writers false in
  let readers_met = Array.create ~len:readers false in
  (* Why a publish was refused, kept alongside the count. The golden carries the claim alone, so
     without this a red run says a writer lost a round and nothing about what refused it -- and on
     Windows the candidates (a sharing violation on the commit, an exhausted staging name, a path
     the host cannot hold) are different bugs that look identical from the count. First refusal per
     writer; each domain writes only its own slot. *)
  let refusal_notes = Array.create ~len:writers "" in
  let refused i exn =
    refusals.(i) <- refusals.(i) + 1;
    if String.is_empty refusal_notes.(i) then refusal_notes.(i) <- Stdlib.Printexc.to_string exn
  in
  let publish_round i writer round =
    (try
       AF.write_all ~path:target ~data:(payload ~writer ~round) ();
       Atomic.incr committed
     with exn -> refused i exn);
    Atomic.incr concluded
  in
  (* The gate. One writer stops HALFWAY THROUGH ITS PAYLOAD -- not between two publications -- and
     every reader takes a read before it continues. Pausing between publications was not enough
     (Codex P2, round 5): on one core the readers could take every read while the writers waited, so
     nothing was read during a mutation and a truncate-in-place implementation would have passed.
     Stopped here, the bytes of a partial payload exist on disk; an implementation that wrote them
     into the target would be caught by the exact-set claim below, on any number of cores. *)
  let publish_gated i writer round =
    let whole = payload ~writer ~round in
    let half = String.length whole / 2 in
    (try
       AF.with_channel ~path:target () ~f:(fun oc ->
           Stdlib.output_string oc (String.prefix whole half);
           Stdlib.flush oc;
           Atomic.set half_written true;
           writers_met.(i) <- await (fun () -> Atomic.get readers_read >= readers);
           Stdlib.output_string oc (String.drop_prefix whole half));
       Atomic.incr committed
     with exn -> refused i exn);
    Atomic.incr concluded
  in
  let writer_domains =
    Array.init writers ~f:(fun i ->
        Domain.spawn (fun () ->
            let writer = Char.to_int 'a' + i in
            if i = 0 then publish_gated i writer 0
            else (
              publish_round i writer 0;
              writers_met.(i) <- await (fun () -> Atomic.get readers_read >= readers));
            for round = 1 to rounds - 1 do
              publish_round i writer round
            done;
            Atomic.incr writers_finished))
  in
  let reader_domains =
    Array.init readers ~f:(fun i ->
        Domain.spawn (fun () ->
            readers_met.(i) <- await (fun () -> Atomic.get half_written);
            (* Taken with a payload half-written on disk. *)
            let seen = ref [ observe_published () ] in
            let during_write = List.hd_exn !seen in
            Atomic.incr readers_read;
            let reading_since = Mtime_clock.counter () in
            let snapshot reads =
              (reads, Atomic.get concluded, Atomic.get committed, Atomic.get writers_finished)
            in
            let rec loop n =
              let reads, concluded, committed, finished = snapshot n in
              (* The variant records the SAME [finished] sample this branch tested. The lifetime
                 claim below asks only which branch each reader took; it never reconstructs that
                 fact from [concluded] or [committed] after the domains have stopped. *)
              if n >= reads_per_reader && finished >= writers then
                Writers_finished { reads; concluded; committed }
              else if Float.(since reading_since > reader_ceiling_seconds) then
                Reader_ceiling { reads; writers_finished = finished; concluded; committed }
              else (
                seen := observe_published () :: !seen;
                (* A zero-gap reopen on Windows can deny every rename attempt for the entire commit
                   deadline. That measures which domain the host scheduled, not Atomic_file. The
                   gated half-write above already forces real overlap; this pause merely gives the
                   writer scheduler a turn between observations, and is one real timer tick
                   there. *)
                Unix.sleepf spin_pause;
                loop (n + 1))
            in
            let stop = loop 1 in
            (* One last read after the recorded exit condition, so the bytes claims cover the
               boundary observation as well as the concurrent body. *)
            seen := observe_published () :: !seen;
            observations.(i) <- { during_write; seen = !seen; stop }))
  in
  Array.iter writer_domains ~f:Domain.join;
  Array.iter reader_domains ~f:Domain.join;
  let per_reader = Array.to_list observations in
  let seen = List.concat_map per_reader ~f:(fun report -> report.seen) in
  Array.iteri observations ~f:(fun i report ->
      (match report.stop with
      | Writers_finished { reads; concluded; committed } ->
          if concluded <> total_publications then
            Stdio.eprintf
              "reader %d recorded every writer finished after %d reads, but publications concluded \
               %d/%d and committed %d (harness counter mismatch; not part of the golden)\n"
              i reads concluded total_publications committed
      | Reader_ceiling { reads; writers_finished; concluded; committed } ->
          Stdio.eprintf
            "reader %d hit its %.0fs harness ceiling after %d reads: writers finished %d/%d, \
             publications concluded %d/%d, committed %d (not part of the golden)\n"
            i reader_ceiling_seconds reads writers_finished writers concluded total_publications
            committed
      | Reader_not_started ->
          Stdio.eprintf "reader %d recorded no report (harness failure; not part of the golden)\n" i);
      match List.find report.seen ~f:(Fn.non (observation_is_published published)) with
      | None -> ()
      | Some (Bytes_read data) ->
          Stdio.eprintf
            "reader %d observed unpublished bytes: length %d, self-describing=%b (Atomic_file \
             regression; not part of the golden)\n"
            i (String.length data) (is_well_formed data)
      | Some Absent ->
          Stdio.eprintf
            "reader %d observed the published path missing (Atomic_file regression; not part of \
             the golden)\n"
            i
      | Some (Refused reason) ->
          Stdio.eprintf
            "reader %d remained refused after its retry deadline: %s (platform/harness refusal; \
             not part of the golden)\n"
            i reason);
  Verdict.p_all ~min:writers "every writer met the readers at the rendezvous"
    (Array.to_list writers_met) ~f:Fn.id;
  Verdict.p_all ~min:readers "every reader saw a payload go half-written"
    (Array.to_list readers_met) ~f:Fn.id;
  (* The overlap claim, and the one that makes this leg independent of how many cores run it: the
     read below was taken while a writer sat inside its payload with half of it already on disk. *)
  Verdict.p_all ~min:readers gated_read_claim per_reader ~f:(fun report ->
      observation_is_published published report.during_write);
  Verdict.p_all ~min:readers "every reader kept reading until the writers were done" per_reader
    ~f:(fun report -> match report.stop with Writers_finished _ -> true | _ -> false);
  (* The platform's refusals are neither ignored nor folded into the two claims about the BYTES:
     they get one of their own, and the reads they cost are what the count on stderr names. Every
     read is therefore accounted for by exactly one of the three. *)
  Stdio.eprintf "reads refused and retried by the platform: %d (not part of the golden)\n"
    (Atomic.get refusals_retried);
  Verdict.p_all ~min:(readers * reads_per_reader)
    "every read under concurrent publication resolved rather than staying refused" seen ~f:(function
    | Refused _ -> false
    | Bytes_read _ | Absent -> true);
  Verdict.p_all ~min:(readers * reads_per_reader)
    "every read under concurrent publication observes a payload some writer published"
    (List.filter_map seen ~f:(function Bytes_read data -> Some data | Absent | Refused _ -> None))
    ~f:(Hash_set.mem published);
  Verdict.p_none ~min:(readers * reads_per_reader)
    "no read under concurrent publication finds the path missing" seen ~f:(function
    | Absent -> true
    | Bytes_read _ | Refused _ -> false);
  Array.iteri refusal_notes ~f:(fun i note ->
      if not (String.is_empty note) then
        Stdio.eprintf "writer %d lost %d of %d rounds; first refusal: %s (not part of the golden)\n"
          i refusals.(i) rounds note);
  (* Liveness, and this is as much of it as the module HAS on Windows. OCaml's opens carry no
     FILE_SHARE_DELETE, so a commit can be refused on every poll for the whole second
     [publish_staged] waits when readers reopen with no gap. Nothing in [atomic_file.ml] can prevent
     that; the sharing rule belongs to the platform, and the module's contract says a publish may be
     refused ([atomic_file.mli], "Windows"). The readers above pause for one scheduler turn after a
     read rather than manufacturing that denial loop; the gated half-write and its executed broken
     control retain deterministic overlap and the ability to catch torn publication.

     The only scheduled Windows run carrying this metric before that change (2026-09-02) measured 24
     refusals in 600 publications. One loaded-host sample is not a basis for moving either the
     one-second library deadline or this test's 1% line, so both stay unchanged (gh-ocannl-900).
     "Every writer committed every round" remains a claim about the host rather than about the
     module: every publication must end in exactly one contract outcome, none may go missing, and
     refusal must stay rare. The claims that carry the module's actual promise -- nothing torn,
     nothing absent, no staging file left behind -- are unchanged and unweakened. *)
  Verdict.p "every publication either committed or was refused"
    (Atomic.get committed + Array.fold refusals ~init:0 ~f:( + ) = total_publications);
  (* Which failure it was, not merely that there was one. `publish_staged` propagates `Sys.rename`'s
     bare message; `open_staging` formats its own around the staging path, and a path this host
     cannot hold arrives as a third thing again -- so pinning the exact text is what keeps a lost
     round attributable to the commit rather than to whichever of them happened. A different message
     is a different fact and should fail here; the note is on stderr above to read it by. *)
  Verdict.p_all ~min:writers "every writer that lost a round lost it to a refused commit"
    (Array.to_list (Array.mapi refusals ~f:(fun i n -> (n, refusal_notes.(i)))))
    ~f:(fun (n, note) -> n = 0 || String.equal note {|Sys_error("Permission denied")|});
  (* A hundredth is the line between "the platform refused a commit" and "publication is broken":
     four writers publishing 150 rounds each may lose six between them and no more. *)
  Verdict.p "publication was refused for fewer than one round in a hundred"
    (100 * Array.fold refusals ~init:0 ~f:( + ) < total_publications);
  Verdict.p "the file left by the race is a complete payload"
    (Option.value_map (read_published ()) ~default:false ~f:is_well_formed);
  Verdict.p_none "the race leaves no staging file behind" (listing ()) ~f:AF.is_staging_file;
  Verdict.p "the race leaves only the published file"
    (List.equal String.equal (listing ()) [ "published.bin" ])

(* Whether a basename at the per-component limit can exist in [dir] on this host at all.

   A component limit and a PATH limit are different budgets, and Windows has both: 255 bytes for one
   component, and 260 characters (MAX_PATH) for the whole thing. The build tree spends about 65 of
   those before the fixture's 244 are added, so every open of the long fixtures below fails there
   with ENOENT -- and did, invisibly, because the exception reached the runtime after [Verdict]'s
   at_exit had already exited 1 over the failures this file's Windows legs were also producing.

   Asked by TRYING rather than by [Sys.win32], because the budget belongs to the path this run
   actually got: a Windows checkout near the drive root would have the room, and a deeply nested
   POSIX one still has no such cap. The probe is a name of the fixtures' own length, so it answers
   the question the fixtures ask. *)
let long_names_fit_here =
  lazy
    (let probe = Stdlib.Filename.concat dir (String.make 240 'p' ^ ".probe") in
     match Unix.openfile probe [ Unix.O_WRONLY; Unix.O_CREAT; Unix.O_EXCL ] 0o600 with
     | fd ->
         Unix.close fd;
         (try Stdlib.Sys.remove probe with _ -> ());
         true
     | exception Unix.Unix_error _ -> false)

let path_budget_gate = "a host that caps the whole path (MAX_PATH)"

(* A staging name is a DERIVED name, and the two things derivation must guarantee are that it fits
   where the target fits and that no two attempts pick the same one. Both are observed through the
   commit window, which is the only moment a staging file exists. *)
let () =
  reset_dir ();
  (* A basename close to the per-component limit: with the old four-character suffix it fit, and
     with a naive suffix it would not (Codex P2, round 2). *)
  let long_name = String.make 240 'n' ^ ".bin" in
  let long_target = Stdlib.Filename.concat dir long_name in
  (* A long name whose characters are multibyte: a byte-wise cut can land inside one, and the
     malformed name that results is refused by Windows and by APFS alike (Codex P2, round 4). The
     3-byte character makes the budget's cut fall mid-character. 70 of them, not 90: a component
     limit is 255 BYTES on ext4 while macOS counts UTF-16 units, so 90 snowmen are a legal target
     name here and an illegal one on Linux, where this leg would have failed before reaching an
     assertion (Codex P1, round 7). The claim below is what pins that -- a fixture has to be a name
     every filesystem accepts, or it tests the filesystem instead. *)
  let utf8_name = String.concat (List.init 70 ~f:(fun _ -> "\xe2\x98\x83")) ^ ".bin" in
  let utf8_target = Stdlib.Filename.concat dir utf8_name in
  (* About the FIXTURES and not about this host, so it is asked wherever the file runs -- including
     where the two claims' subjects cannot be created. *)
  Verdict.p_all "every fixture name is a valid component on a 255-byte filesystem"
    [ long_name; utf8_name ] ~f:(fun name -> String.length name <= 255);
  (* Every remaining claim of this leg goes through one of these two: evaluated where the host has
     the path budget for the fixtures, announced as skipped where it does not. [Verdict.skipped]
     prints the line [p] would, so the golden stays uniform across hosts while a reader of the run's
     stderr still sees exactly what was not checked. The collections are thunks because under the
     gate nothing published, and a quantified claim over the empty list they would then be is a
     refusal rather than a skip. *)
  let fits = force long_names_fit_here in
  let long_claim name b =
    if fits then Verdict.p name (b ())
    else Verdict.skipped ~aggregation:`Environment ~backend:path_budget_gate name
  in
  let long_claim_all ?min name items ~f =
    if fits then Verdict.p_all ?min name (items ()) ~f
    else Verdict.skipped ~aggregation:`Environment ~backend:path_budget_gate name
  in
  let staged = ref [] in
  let utf8_staged = ref [] in
  if fits then (
    let capture () = staged := staging_leftovers () @ !staged in
    AF.write_all ~path:long_target ~data:"payload" ~before_commit:capture ();
    AF.write_all ~path:utf8_target ~data:"payload"
      ~before_commit:(fun () -> utf8_staged := staging_leftovers () @ !utf8_staged)
      ());
  long_claim "a target named to the filesystem's limit publishes" (fun () ->
      Stdlib.Sys.file_exists long_target);
  long_claim_all "every staging name fits one filesystem component"
    (fun () -> !staged)
    ~f:(fun name -> String.length name <= 255);
  long_claim_all "a long target's staging file is recognized as that target's"
    (fun () -> !staged)
    ~f:(AF.is_staging_file_for ~path:long_target);
  (* Both fixtures must actually reach the truncating path, or they test the short-name branch under
     a long-looking name: a stem that fit would appear in the staging name verbatim. *)
  long_claim_all "the long fixtures exercise the truncating stem"
    (fun () -> !staged @ !utf8_staged)
    ~f:(fun name ->
      not (String.is_prefix name ~prefix:long_name || String.is_prefix name ~prefix:utf8_name));
  long_claim "a multibyte target name publishes" (fun () -> Stdlib.Sys.file_exists utf8_target);
  long_claim_all "every staging name of a multibyte target is itself valid UTF-8"
    (fun () -> !utf8_staged)
    ~f:Stdlib.String.is_valid_utf_8;
  long_claim_all "a multibyte target's staging file fits one filesystem component"
    (fun () -> !utf8_staged)
    ~f:(fun name -> String.length name <= 255);
  long_claim_all "a multibyte target's staging file is recognized as that target's"
    (fun () -> !utf8_staged)
    ~f:(AF.is_staging_file_for ~path:utf8_target);
  (* Case: on Windows and on a default macOS volume these two spellings are one file. *)
  let shouting = Stdlib.Filename.concat dir (String.uppercase long_name) in
  long_claim_all "a differently-cased spelling of the target claims its staging files"
    (fun () -> !staged)
    ~f:(AF.is_staging_file_for ~path:shouting);
  (* Uniqueness within a process, observed rather than assumed: each attempt's staging file is
     captured in its own commit window, and no name repeats. Uniqueness ACROSS processes rests on
     exclusive creation, which no single-process test can exercise. *)
  reset_dir ();
  let names = ref [] in
  for round = 0 to 49 do
    AF.write_all ~path:target
      ~data:(payload ~writer:(Char.to_int 'a') ~round)
      ~before_commit:(fun () -> names := staging_leftovers () @ !names)
      ()
  done;
  generated_staging_names := !names;
  Verdict.p "all 50 attempts expose exactly one staging path" (List.length !names = 50);
  Verdict.p_all ~min:50 "every staging_path output round-trips through is_staging_file" !names
    ~f:AF.is_staging_file;
  Verdict.p "no two staging names repeat"
    (List.length (List.dedup_and_sort !names ~compare:String.compare) = List.length !names);
  Verdict.p_all ~min:50 "a differently-cased short target claims its staging files too" !names
    ~f:(AF.is_staging_file_for ~path:(Stdlib.Filename.concat dir "PUBLISHED.BIN"));
  Verdict.p "the repeated publishing left only the published file"
    (List.equal String.equal (listing ()) [ "published.bin" ])

(* Publishing a privately built TREE: the caller assembles a directory under a staging name and
   commits the whole thing with one rename. Factored out because the rerun control below runs the
   very same sequence a second time -- what an interrupted run leaves behind must not change what
   this sequence does. *)
let publish_directory_tree () =
  let staged_dir = Stdlib.Filename.concat dir "staged-directory" in
  let published_dir = Stdlib.Filename.concat dir "published-directory" in
  match
    Unix.mkdir staged_dir 0o755;
    Stdio.Out_channel.write_all (Stdlib.Filename.concat staged_dir "complete") ~data:"tree";
    AF.publish_staged ~staging:staged_dir ~path:published_dir;
    Stdlib.Sys.file_exists (Stdlib.Filename.concat published_dir "complete")
  with
  | published ->
      remove_tree published_dir;
      published
  | exception exn ->
      (* Reported rather than propagated, so the claim that wanted this sequence to work says
         [false] on its own line -- with the refusal named on stderr for whoever reads why. *)
      Stdio.eprintf "directory publication refused: %s\n%!" (Stdlib.Printexc.to_string exn);
      false

(* A writer that fails in its commit window leaves the previous entry and no artifact — for a
   payload held in memory and for one streamed through a channel alike. *)
let () =
  reset_dir ();
  let seed = payload ~writer:(Char.to_int 'a') ~round:0 in
  AF.write_all ~path:target ~data:seed ();
  let failed =
    try
      AF.write_all ~path:target
        ~data:(payload ~writer:(Char.to_int 'b') ~round:1)
        ~before_commit:(fun () -> failwith "gh780 injected commit failure")
        ();
      false
    with _ -> true
  in
  Verdict.p "a publish that fails in its commit window raises" failed;
  Verdict.p "a failed publish leaves the previous payload"
    (Option.equal String.equal (read_published ()) (Some seed));
  Verdict.p_none "a failed publish removes its own staging file" (listing ()) ~f:AF.is_staging_file;
  let streamed = payload ~writer:(Char.to_int 'c') ~round:2 in
  AF.with_channel ~path:target () ~f:(fun oc -> Stdlib.output_string oc streamed);
  Verdict.p "a streamed publish commits its whole payload"
    (Option.equal String.equal (read_published ()) (Some streamed));
  let stream_failed =
    try
      let () =
        AF.with_channel ~path:target () ~f:(fun oc ->
            Stdlib.output_string oc "half";
            (failwith "gh780 injected stream failure" : unit))
      in
      false
    with _ -> true
  in
  Verdict.p "a streamed publish that raises mid-write raises" stream_failed;
  Verdict.p "a failed streamed publish leaves the previous payload"
    (Option.equal String.equal (read_published ()) (Some streamed));
  Verdict.p_none "a failed streamed publish removes its own staging file" (listing ())
    ~f:AF.is_staging_file;
  Verdict.p "a privately built directory tree publishes as one path" (publish_directory_tree ())

(* The rerun control (gh-ocannl-803). Every leg above starts from an empty scratch directory, and a
   run cut short -- Ctrl-C, a failed claim's exit, a killed suite -- does not get to finish its own
   cleanup. So plant, by hand, exactly what an interruption in this file can leave behind: the
   staging tree built but not yet published, the published tree not yet removed, the sweep leg's
   directory not yet rmdir'd, and a stale published file. Each tree gets a nested subtree, so that
   clearing them is claimed to RECURSE rather than merely to try [rmdir] once.

   Then hand that state to an ACTUAL rerun -- a fresh process, whose startup is the thing under test
   -- and re-run the sequence the leftovers obstruct. Under a file-only reset the directories
   survive the child's startup, [Unix.mkdir] raises EEXIST and [publish_staged] renames into a
   non-empty target; with the startup reset gone, the child reports the leftovers it was supposed to
   have cleared. *)
let () =
  AF.ensure_dir dir;
  let plant_tree name =
    let path = Stdlib.Filename.concat dir name in
    AF.ensure_dir path;
    Stdio.Out_channel.write_all (Stdlib.Filename.concat path "complete") ~data:"tree";
    let nested = Stdlib.Filename.concat path "nested" in
    AF.ensure_dir nested;
    Stdio.Out_channel.write_all (Stdlib.Filename.concat nested "deep") ~data:"deep"
  in
  List.iter [ "staged-directory"; "published-directory"; "not_created_yet" ] ~f:plant_tree;
  Stdio.Out_channel.write_all target ~data:"leftover from an interrupted run";
  let planted = listing () in
  Verdict.p "the planted leftovers are what an interrupted run leaves behind"
    (List.equal String.equal planted
       [ "not_created_yet"; "published-directory"; "published.bin"; "staged-directory" ]);
  let status, probe_cwd, seen_after_startup = run_startup_probe () in
  Verdict.p "a fresh process starts over the leftovers and exits cleanly"
    (match status with Unix.WEXITED 0 -> true | _ -> false);
  Verdict.p "the fresh process ran in this test's own scratch directory"
    (Option.value_map probe_cwd ~default:false ~f:(String.equal (Unix.getcwd ())));
  Verdict.p_empty "a fresh process's startup clears an interrupted run's directory fixtures"
    ~over:planted seen_after_startup;
  Verdict.p_empty "the rerun's clearing is visible to this process too" ~over:planted (listing ());
  Verdict.p "a rerun over an interrupted run's leftovers publishes its directory tree"
    (publish_directory_tree ());
  Verdict.p_empty "the rerun left the scratch directory as it found it" ~over:planted (listing ())

(* One exception type for filesystem refusals, whatever refused. A best-effort writer -- the
   schedule cache treats a refusal as a future miss rather than a failed tuning run -- needs one
   handler, not a taxonomy of the operations the helper happens to use internally; the exclusive
   open raises [Unix_error], and letting that escape would have walked straight past such a handler
   (Codex P2, round 3). What [f] and [before_commit] raise is the caller's own and must NOT be
   converted. *)
let classify f =
  match f () with
  | () -> `Returned
  | exception Stdlib.Sys_error _ -> `Sys_error
  | exception Failure msg when String.is_substring msg ~substring:"gh780" -> `Caller
  | exception _ -> `Other

let () =
  reset_dir ();
  let refusals =
    [
      (* The directory does not exist. *)
      Stdlib.Filename.concat (Stdlib.Filename.concat dir "no_such_dir") "x.bin";
      (* A path component that is a file rather than a directory. *)
      Stdlib.Filename.concat target "x.bin";
    ]
  in
  AF.write_all ~path:target ~data:"seed" ();
  Verdict.p_all "every filesystem refusal reaches the caller as Sys_error" refusals ~f:(fun path ->
      Poly.equal `Sys_error (classify (fun () -> AF.write_all ~path ~data:"payload" ())));
  Verdict.p "an exception from the caller's own writer is not converted"
    (Poly.equal `Caller
       (classify (fun () ->
            AF.with_channel ~path:target () ~f:(fun _oc -> (failwith "gh780 caller failure" : unit)))));
  Verdict.p "an exception from the commit hook is not converted"
    (Poly.equal `Caller
       (classify (fun () ->
            AF.write_all ~path:target ~data:"payload"
              ~before_commit:(fun () -> failwith "gh780 hook failure")
              ())));
  Verdict.p_none "a refused publish leaves no staging file" (listing ()) ~f:AF.is_staging_file

(* Crash-stale cleanup: the writer that dies in its commit window cannot clean up after itself, so
   the sweep must — by age, and only over this module's own artifacts. *)
let age_seconds = 3600.
let hex_alphabet = "abcdef0123456789"
let hex_field width = String.init width ~f:(fun i -> hex_alphabet.[i % String.length hex_alphabet])
let nonce = hex_field AF.nonce_width
let field value = Printf.sprintf "%0*x" AF.field_width value

let staged_name ~target ~counter =
  Printf.sprintf "%s%s%s.%s.%s" target AF.staging_infix (field 4242) (field counter) nonce

let staging_name_of_fields ~stem = function
  | [ pid; counter; nonce ] -> Printf.sprintf "%s%s%s.%s.%s" stem AF.staging_infix pid counter nonce
  | _ -> invalid_arg "staging_name_of_fields: expected pid, counter and nonce"

let valid_fields = [ hex_field AF.field_width; hex_field AF.field_width; nonce ]

let replace_field fields at replacement =
  List.mapi fields ~f:(fun i field -> if i = at then replacement else field)

(* Each of the three generated fields is perturbed in every direction that previously drifted by
   hand: wider, narrower, outside lowercase hex, and case-flipped. This matrix replaces the list of
   reviewer-found impostors, so a change to a width or to the field count expands from the same
   constants as generation rather than requiring another literal name. *)
let field_near_misses =
  List.mapi valid_fields ~f:(fun at original ->
      let perturbed =
        [
          original ^ "0";
          String.drop_suffix original 1;
          "g" ^ String.drop_prefix original 1;
          String.uppercase original;
        ]
      in
      List.map perturbed ~f:(fun replacement ->
          staging_name_of_fields ~stem:"report" (replace_field valid_fields at replacement)))
  |> List.concat

let empty_stem_near_miss = AF.staging_infix ^ String.concat ~sep:"." valid_fields

let overlong_stem_near_miss =
  String.make 192 's' ^ AF.staging_infix ^ String.concat ~sep:"." valid_fields

let missing_field_near_miss =
  "report" ^ AF.staging_infix ^ String.concat ~sep:"." (List.drop_last_exn valid_fields)

let surplus_field_near_miss = staging_name_of_fields ~stem:"report" valid_fields ^ ".extra"

let structural_near_misses =
  [
    (* Empty stems are not generated. *)
    empty_stem_near_miss;
    (* Every field is valid, but generation never emits a stem beyond its 191-byte budget. *)
    overlong_stem_near_miss;
    (* Missing and surplus fields cannot be generator outputs either. *)
    missing_field_near_miss;
    surplus_field_near_miss;
  ]

let near_misses = field_near_misses @ structural_near_misses

let rec gitignore_files dir =
  Array.to_list (Stdlib.Sys.readdir dir)
  |> List.concat_map ~f:(fun entry ->
      let path = Stdlib.Filename.concat dir entry in
      if String.equal entry ".gitignore" then [ path ]
      else if String.equal entry ".git" || String.equal entry "_build" then []
      else
        match Stdlib.Sys.is_directory path with
        | true -> gitignore_files path
        | false -> []
        | exception Stdlib.Sys_error _ -> [])

(* The ignore rule is the third description of the name scheme. Derive it from the generator's
   constants, then compare against the one committed line instead of pinning another copy. The one
   known bound remains: a glob cannot express the recognizer's MAXIMUM stem length. It can therefore
   hide an overlong near-miss from [git status], but that is leak-or-hide only, never
   wrong-deletion: the destructive sweep consults [is_staging_file], which rejects that name. *)
let () =
  let field_glob width = String.concat (List.init width ~f:(fun _ -> "[0-9a-f]")) in
  let expected =
    Printf.sprintf "?*%s%s.%s.%s" AF.staging_infix (field_glob AF.field_width)
      (field_glob AF.field_width) (field_glob AF.nonce_width)
  in
  let gitignore = Stdio.In_channel.read_all "../../.gitignore" in
  let committed =
    String.split_lines gitignore
    |> List.filter ~f:(fun line ->
        (not (String.is_prefix line ~prefix:"#"))
        && String.is_substring line ~substring:AF.staging_infix)
  in
  Verdict.p "the ignore file carries exactly one Atomic_file staging rule"
    (List.length committed = 1);
  Verdict.p_all "the committed staging rule equals the scheme derived from Atomic_file" committed
    ~f:(String.equal expected);
  let matches_committed_rule name =
    List.exists committed ~f:(fun pattern -> Ignore.glob_matches pattern name)
  in
  (* The generator is the authority on separators and field layout. These are actual names captured
     while [with_channel] held them in the commit window, so changing [staging_path] and its
     recognizer together cannot leave this relationship green against an old ignore rule. The exact
     derived rule contains no slash, so Git applies it to the basename at every depth. Under Git's
     ordered last-match-wins semantics, reject every later negation that has a wildcard/class or
     literally names the staging infix. A backslash is refused too: it can hide an escaped literal
     from the raw spelling. Exact unrelated re-inclusions (the committed file has two for [.claude])
     cannot match a generated staging basename; any general or escaped negation is refused
     conservatively. This needs neither a [.git] directory nor a Git executable in package builds
     (Codex P2, rounds 4-6 and 8). *)
  Verdict.p_all ~min:50 "the committed basename rule matches every actual staging_path output"
    !generated_staging_names ~f:matches_committed_rule;
  let patterns = Ignore.ignore_patterns gitignore in
  let rec after_committed_rule = function
    | [] -> []
    | { Ignore.pattern; negated = false } :: rest when String.equal pattern expected -> rest
    | _ :: rest -> after_committed_rule rest
  in
  let could_expose_staging { Ignore.pattern; negated } =
    negated
    && (String.exists pattern ~f:(fun c ->
            Char.equal c '*' || Char.equal c '?' || Char.equal c '[' || Char.equal c '\\')
       || String.is_substring pattern ~substring:AF.staging_infix)
  in
  Verdict.p_none "no later Git negation can match an Atomic_file staging path"
    (after_committed_rule patterns) ~f:could_expose_staging;
  let ignore_root = Stdlib.Filename.concat ".." ".." in
  let nested_ignores =
    gitignore_files ignore_root
    |> List.filter ~f:(fun path -> not (String.equal (Stdlib.Filename.dirname path) ignore_root))
  in
  let nested_patterns =
    List.concat_map nested_ignores ~f:(fun path ->
        Ignore.ignore_patterns (Stdio.In_channel.read_all path)
        |> List.map ~f:(fun pattern -> (path, pattern)))
  in
  Verdict.p_exists "the nested ignore corpus is present" nested_ignores ~f:(fun _ -> true);
  Verdict.p_none "no nested Git negation can match an Atomic_file staging path" nested_patterns
    ~f:(fun (_, pattern) -> could_expose_staging pattern);
  Verdict.p_none ~min:15 "the committed rule rejects every expressible generated near-miss"
    (field_near_misses @ [ empty_stem_near_miss; missing_field_near_miss; surplus_field_near_miss ])
    ~f:matches_committed_rule;
  Verdict.p_all "the glob-only overlong-stem residue is hidden but never recognized for deletion"
    [ overlong_stem_near_miss ] ~f:(fun name ->
      matches_committed_rule name && not (AF.is_staging_file name))

let plant_staging ~name ~age =
  let path = Stdlib.Filename.concat dir name in
  Stdio.Out_channel.write_all path ~data:"abandoned";
  let stamp = Unix.time () -. age in
  Unix.utimes path stamp stamp;
  path

let () =
  reset_dir ();
  let seed = payload ~writer:(Char.to_int 'a') ~round:0 in
  AF.write_all ~path:target ~data:seed ();
  let stale = plant_staging ~name:(staged_name ~target:"published.bin" ~counter:0) ~age:7200. in
  let fresh = plant_staging ~name:(staged_name ~target:"published.bin" ~counter:1) ~age:0. in
  let bystander = plant_staging ~name:"unrelated.bin" ~age:7200. in
  (* The overlong-stem near-miss is a 240-character name BY CONSTRUCTION -- a stem past the
     generator's 191-byte budget is what it is -- so on a host that caps the whole path there is no
     directory it fits in at all (see [long_names_fit_here]). It is dropped from the PLANTED corpus
     there and from nothing else: the two recognizer claims above are about the name, and they ask
     it wherever this runs. Named rather than filtered by a length, so that a corpus that grows a
     second long member fails here instead of being silently trimmed. *)
  let plantable =
    if force long_names_fit_here then near_misses
    else (
      Stdio.eprintf
        "not planted, and this host has no directory it would fit in: the %d-character \
         overlong-stem near-miss (not part of the golden)\n"
        (String.length overlong_stem_near_miss);
      List.filter near_misses ~f:(Fn.non (String.equal overlong_stem_near_miss)))
  in
  let planted_near_misses = List.map plantable ~f:(fun name -> plant_staging ~name ~age:7200.) in
  Verdict.p_all "every planted staging file is recognized as one" [ stale; fresh ] ~f:(fun path ->
      AF.is_staging_file (Stdlib.Filename.basename path));
  Verdict.p "the bystander is not recognized as a staging file"
    (not (AF.is_staging_file (Stdlib.Filename.basename bystander)));
  Verdict.p_none ~min:16 "no systematically perturbed near-miss is recognized as a staging file"
    near_misses ~f:AF.is_staging_file;
  (* The narrow scope: whose staging file it is, not merely that it is one. *)
  Verdict.p_all "the published file's own staging files are recognized as its" [ stale; fresh ]
    ~f:(fun path -> AF.is_staging_file_for ~path:target (Stdlib.Filename.basename path));
  let other_target = plant_staging ~name:(staged_name ~target:"other.bin" ~counter:9) ~age:7200. in
  Verdict.p "another target's staging file is not recognized as this one's"
    (not (AF.is_staging_file_for ~path:target (Stdlib.Filename.basename other_target)));
  AF.cleanup_stale_for ~max_age_seconds:age_seconds target;
  Verdict.p "the narrow sweep removes this target's abandoned staging file"
    (not (Stdlib.Sys.file_exists stale));
  Verdict.p "the narrow sweep spares another target's staging file"
    (Stdlib.Sys.file_exists other_target);
  let stale = plant_staging ~name:(staged_name ~target:"published.bin" ~counter:0) ~age:7200. in
  AF.cleanup_stale ~max_age_seconds:age_seconds dir;
  Verdict.p "the sweep removes an abandoned staging file" (not (Stdlib.Sys.file_exists stale));
  Verdict.p "the sweep removes another target's abandoned staging file too"
    (not (Stdlib.Sys.file_exists other_target));
  Verdict.p "the sweep spares a staging file young enough to be in flight"
    (Stdlib.Sys.file_exists fresh);
  Verdict.p "the sweep spares the published file" (Stdlib.Sys.file_exists target);
  Verdict.p "the sweep spares an aged file that is not a staging artifact"
    (Stdlib.Sys.file_exists bystander);
  (* Fifteen: the corpus is sixteen, and the one member a path-capped host cannot plant is one. *)
  Verdict.p_all ~min:15 "the sweep spares every aged systematically perturbed near-miss"
    planted_near_misses ~f:Stdlib.Sys.file_exists;
  Verdict.p "the published file still reads as it was written"
    (Option.equal String.equal (read_published ()) (Some seed));
  (* A cache's reader calls the once-per-process sweep before its first writer has created the
     directory, so a missing directory must not consume the one sweep. *)
  let absent = Stdlib.Filename.concat dir "not_created_yet" in
  AF.cleanup_stale_once ~max_age_seconds:age_seconds absent;
  AF.ensure_dir absent;
  let planted_after_creation =
    let path = Stdlib.Filename.concat absent (staged_name ~target:"late.bin" ~counter:0) in
    Stdio.Out_channel.write_all path ~data:"abandoned";
    let stamp = Unix.time () -. 7200. in
    Unix.utimes path stamp stamp;
    path
  in
  AF.cleanup_stale_once ~max_age_seconds:age_seconds absent;
  Verdict.p "a directory that did not exist yet does not consume its one sweep"
    (not (Stdlib.Sys.file_exists planted_after_creation));
  (try Stdlib.Sys.rmdir absent with _ -> ());
  (* The once-per-process form sweeps this directory exactly once, however many writers call it: a
     staging file abandoned after that first sweep survives until the next process. *)
  AF.cleanup_stale_once ~max_age_seconds:age_seconds dir;
  let after_once =
    plant_staging ~name:(staged_name ~target:"published.bin" ~counter:2) ~age:7200.
  in
  AF.cleanup_stale_once ~max_age_seconds:age_seconds dir;
  Verdict.p "the once-per-process sweep does not run a second time"
    (Stdlib.Sys.file_exists after_once);
  AF.cleanup_stale ~max_age_seconds:age_seconds dir;
  Verdict.p "the unconditional sweep still removes it" (not (Stdlib.Sys.file_exists after_once));
  reset_dir ()
