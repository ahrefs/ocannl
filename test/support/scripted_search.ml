(* The scripted-search harness of the autotune tests: a search whose ranking is the test's.

   A test that claims what a search attempts, times, crowns or stores cannot let the machine rank
   the candidates, so it scripts every admitted window through [Autotune.on_candidate_measured]
   (gh-ocannl-1027) and opts the windows out of host contention
   ([Autotune.with_uncontended_test_windows], gh-ocannl-1156). Around that it records what the
   search did: each candidate attempt ([Autotune.on_candidate_attempt]), each admitted window with
   the time the script gave it, and, for a full search, the schedule-cache operations
   ([Ir.Schedule_cache.recording_cache_io]); a budget test reads a virtual clock off the attempts
   ([Autotune.budget_elapsed], gh-ocannl-1409). Configuration a test states for one search is set at
   the config-file level for that search alone.

   [scripted] is that scope around any body -- an [Autotune.tune] call with arguments of its own, or
   [Train.tune_placements]; [search] is the common body: one [Autotune.tune] on a computation,
   optionally running the shipped routine to read an output. The other seams a test injects through
   ([on_candidate_preflight], [on_staffed_start], [on_candidate_timed], the timing-call observer) it
   sets around the call itself: the harness sets only the three above, and calls the attempt hook it
   finds in force, so an enclosing fault injection still fires. *)
open Base
open Ocannl.Operation.DSL_modules
module SC = Ir.Schedule_cache

(* One admitted window, as the script sees it: its ordinal among the scope's admitted windows
   (1-based), the candidate's label and schedule digest, and the window's real reading in ms. *)
type window = { ordinal : int; label : string; digest : string; reading : float }

type record = {
  attempts : string list;  (** Every candidate attempt's label, in order, the baseline's first. *)
  windows : (window * float) list;
      (** Every admitted window, in order, with the time the script gave it. *)
}

(* The admitted windows' labels, in order. *)
let measured windows = List.map windows ~f:(fun (w, _) -> w.label)

(* [with_settings settings f] runs [f] with each [(key, value)] of [settings] stated at the
   config-file level, restoring what the file level held before on return or exception. *)
let with_settings settings f =
  let saved =
    List.map settings ~f:(fun (key, _) -> (key, Hashtbl.find Utils.config_file_args key))
  in
  Exn.protect
    ~finally:(fun () ->
      List.iter (List.rev saved) ~f:(fun (key, old) ->
          match old with
          | Some data -> Hashtbl.set Utils.config_file_args ~key ~data
          | None -> Hashtbl.remove Utils.config_file_args key))
    ~f:(fun () ->
      List.iter settings ~f:(fun (key, data) -> Hashtbl.set Utils.config_file_args ~key ~data);
      f ())

(* [scripted ~measure body] runs [body] with every admitted window measuring [measure window] and
   returns its result beside the scope's [record]. [?settings] are stated for the scope alone;
   [~attempt_clock:true] makes a budget check read the attempts so far, the baseline's included, as
   the elapsed seconds, so a budget of [n] lets the first [n] attempts start. A raise from [body]
   propagates and the record is dropped: a test that stops a search by an exception catches it
   inside [body]. *)
let scripted ?(settings = []) ?(attempt_clock = false) ~measure body =
  let attempts = ref [] and windows = ref [] in
  let enclosing = Autotune.Seam.get Autotune.on_candidate_attempt in
  let result =
    with_settings settings @@ fun () ->
    Autotune.Seam.with_value Autotune.on_candidate_attempt (fun label ->
        attempts := label :: !attempts;
        enclosing label)
    @@ fun () ->
    Autotune.Seam.with_value Autotune.on_candidate_measured (fun ~label ~digest reading ->
        let window = { ordinal = List.length !windows + 1; label; digest; reading } in
        let ms = measure window in
        windows := (window, ms) :: !windows;
        ms)
    @@ fun () ->
    let clocked f =
      if attempt_clock then
        Autotune.Seam.with_value Autotune.budget_elapsed
          (fun _real -> Float.of_int (List.length !attempts))
          f
      else f ()
    in
    clocked @@ fun () -> Autotune.with_uncontended_test_windows body
  in
  (result, { attempts = List.rev !attempts; windows = List.rev !windows })

type observed = {
  report : Autotune.report;
  attempts : string list;  (** Every candidate attempt's label, in order, the baseline's first. *)
  windows : (window * float) list;
      (** Every admitted window, in order, with the time the script gave it. *)
  measured : string list;  (** Every admitted window's label, in order. *)
  output : float array;  (** What the shipped routine computed into [?output]; empty without. *)
  ios : SC.cache_io list;  (** The search's schedule-cache operations. *)
  identity : bool;  (** Whether the search's context had a timing identity to key the cache on. *)
}

(* One [Autotune.tune] of [comp] in [?ctx] (a fresh [Context.cpu ()] by default) under [scripted],
   at one repeat, with [?beam_width] (2), [?rounds] (0) and [?cache_dir] (none); [?log] and
   [?progress] are passed through. With [?output] the shipped routine is run and that tensor's
   values read; without, the routine is released unrun. [tag] names the search in a failure. *)
let search ?settings ?attempt_clock ?ctx ?output ?(cache_dir = "") ?(beam_width = 2) ?(rounds = 0)
    ?log ?progress ~tag ~measure comp =
  let report = ref None in
  let ctx = match ctx with Some ctx -> ctx | None -> Context.cpu () in
  let identity = Option.is_some (Context.timing_identity ctx) in
  let (output, ios), record =
    scripted ?settings ?attempt_clock ~measure (fun () ->
        let (ctx, routine), ios =
          SC.recording_cache_io (fun () ->
              Autotune.tune ~beam_width ~rounds ~repeats:1 ~cache_dir ?log ?progress
                ~report:(fun r -> report := Some r)
                ctx comp Ir.Indexing.Empty)
        in
        match output with
        | None ->
            Context.release ctx;
            ([||], ios)
        | Some (t : Tensor.t) ->
            let ctx = Context.run ctx routine in
            let values = Context.get_values ctx t.value in
            Context.release ctx;
            (values, ios))
  in
  match !report with
  | Some report ->
      {
        report;
        attempts = record.attempts;
        windows = record.windows;
        measured = measured record.windows;
        output;
        ios;
        identity;
      }
  | None -> failwith (tag ^ ": the search delivered no report")

(* The untuned compile of [comp] in [?ctx] (a fresh [Context.cpu ()] by default), run: [output]'s
   values, the reference a crowned routine is compared with. *)
let untuned ?ctx (output : Tensor.t) comp =
  let ctx = match ctx with Some ctx -> ctx | None -> Context.cpu () in
  let ctx, routine = Context.compile ctx comp Ir.Indexing.Empty in
  let ctx = Context.run ctx routine in
  let values = Context.get_values ctx output.value in
  Context.release ctx;
  values

(* Empties a schedule-cache directory: an earlier run's entries would turn a cold search into a
   replay. *)
let clear_cache_dir dir =
  if Stdlib.Sys.file_exists dir && Stdlib.Sys.is_directory dir then
    Array.iter (Stdlib.Sys.readdir dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat dir f))

(* Runs [f] with stderr routed into a file and returns its result with the lines it wrote, for a
   claim on a search's [autotune:] or [autotune-progress:] lines. The captured text is echoed back
   to stderr once stderr is restored, whether [f] returned or raised (an exception is re-raised with
   its original backtrace after the echo), so nothing the run wrote is hidden; the file is removed
   on every path. *)
let with_stderr_captured f =
  let file = Stdlib.Filename.temp_file "scripted_search" ".stderr" in
  Exn.protect
    ~finally:(fun () -> try Stdlib.Sys.remove file with Sys_error _ -> ())
    ~f:(fun () ->
      Stdio.Out_channel.flush Stdio.stderr;
      let saved = Unix.dup Unix.stderr in
      let outcome =
        Exn.protect
          ~finally:(fun () -> Unix.close saved)
          ~f:(fun () ->
            let fd = Unix.openfile file [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
            Exn.protect ~finally:(fun () -> Unix.close fd) ~f:(fun () -> Unix.dup2 fd Unix.stderr);
            let outcome =
              match f () with
              | result -> Ok result
              | exception exn -> Error (exn, Stdlib.Printexc.get_raw_backtrace ())
            in
            Stdio.Out_channel.flush Stdio.stderr;
            Unix.dup2 saved Unix.stderr;
            outcome)
      in
      let text = Stdio.In_channel.read_all file in
      Stdio.eprintf "%s%!" text;
      match outcome with
      | Ok result -> (result, String.split_lines text)
      | Error (exn, backtrace) -> Stdlib.Printexc.raise_with_backtrace exn backtrace)
