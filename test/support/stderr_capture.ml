(** What a test reads back from its own stderr (gh-ocannl-1296): the capture, and the
    [autotune-progress:] lines the tuner writes there.

    Five tests carried a capture of their own, and only one of them survived an exception inside it:
    the others left fd 2 pointing at a temporary file the run then deleted, so every later line --
    the exception's own report, Verdict's teardown -- went nowhere, and a crash read as a silent
    exit. Three of them parsed the progress lines, each a little differently. This module is the one
    copy of both.

    Base, stdio and unix only, like {!Fresh_process}: reading stderr is not a reason to link a
    backend. *)

open Base

(** [capture ?temp_dir ?echo f] runs [f] with file descriptor 2 routed into a temporary file, and
    answers [f]'s result with everything written to fd 2 meanwhile -- by OCaml's [stderr] channel
    and by C code alike, since the redirection is of the descriptor (the one ppx_expect performs,
    which [foreign_c_stderr] relies on).

    On every path out, whether [f] returned or raised: stderr is flushed and restored; the captured
    text is written back to the restored stderr when [echo] (the default), so nothing the run wrote
    is hidden from its log; the temporary file is removed; and the descriptors this opened are
    closed, even when setting the redirection up failed half-way. An exception from [f] is re-raised
    AFTER all of that, with its original backtrace, so it is reported on the real stderr with the
    captured text before it. [?temp_dir] places the temporary file (default: the system's), which a
    test of this module uses to see that the file is gone. *)
let capture ?temp_dir ?(echo = true) f =
  let file = Stdlib.Filename.temp_file ?temp_dir "ocannl_stderr_capture" ".txt" in
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
            (* From here fd 2 is the file: whatever [f] does, it is pointed back before [saved] is
               closed. *)
            let outcome =
              match f () with
              | result -> Ok result
              | exception exn -> Error (exn, Stdlib.Printexc.get_raw_backtrace ())
            in
            (* A flush that fails (a full temporary filesystem) must not skip the restoration: fd 2
               is pointed back first, and the flush's exception reported after, unless [f]'s own is
               already on its way. *)
            let flushed =
              match Stdio.Out_channel.flush Stdio.stderr with
              | () -> Ok ()
              | exception exn -> Error (exn, Stdlib.Printexc.get_raw_backtrace ())
            in
            Unix.dup2 saved Unix.stderr;
            match (outcome, flushed) with
            | Ok _, Error failure -> Error failure
            | _, (Ok () | Error _) -> outcome)
      in
      let text = Stdio.In_channel.read_all file in
      if echo then Stdio.eprintf "%s%!" text;
      match outcome with
      | Ok result -> (result, text)
      | Error (exn, backtrace) -> Stdlib.Printexc.raise_with_backtrace exn backtrace)

(** [lines ?temp_dir ?echo f] is {!capture} with the captured text split into lines. *)
let lines ?temp_dir ?echo f =
  let result, text = capture ?temp_dir ?echo f in
  (result, String.split_lines text)

(** The tuner's progress lines, as [arrayjit/lib/autotune.mli] documents them (gh-ocannl-1061):

    {v autotune-progress: wall_s=<s> event=<event> <key>=<value> ... v}

    A value is a bare word, or an OCaml-quoted ([%S]) string, which may hold spaces -- an arm label
    does. *)
module Progress = struct
  let prefix = "autotune-progress: "

  type line = {
    event : string;
    wall_s : float;
    fields : (string * string) list;
        (** Every [key=value] of the line, [wall_s] and [event] included, in order; a quoted value
            unquoted and unescaped. *)
  }

  (* The [key=value] fields of [rest], or [None] where it breaks the format: a key that is empty or
     holds a space, a bare value that is empty, a quoted one never closed or not a valid OCaml
     string literal. *)
  let parse_fields rest =
    let len = String.length rest in
    let rec go i acc =
      if i >= len then Some (List.rev acc)
      else
        match String.index_from rest i '=' with
        | None -> None
        | Some eq ->
            let key = String.sub rest ~pos:i ~len:(eq - i) in
            if String.is_empty key || String.mem key ' ' then None
            else if eq + 1 < len && Char.equal rest.[eq + 1] '"' then
              (* The closing quote is the first one no backslash escapes. *)
              let rec close j =
                if j >= len then None
                else if Char.equal rest.[j] '\\' then close (j + 2)
                else if Char.equal rest.[j] '"' then Some j
                else close (j + 1)
              in
              match close (eq + 2) with
              | None -> None
              | Some j -> (
                  match
                    Stdlib.Scanf.unescaped (String.sub rest ~pos:(eq + 2) ~len:(j - eq - 2))
                  with
                  | value ->
                      let next = j + 1 in
                      if next < len && not (Char.equal rest.[next] ' ') then None
                      else go (next + 1) ((key, value) :: acc)
                  | exception (Stdlib.Scanf.Scan_failure _ | Failure _ | End_of_file) -> None)
            else
              let stop = Option.value (String.index_from rest eq ' ') ~default:len in
              let value = String.sub rest ~pos:(eq + 1) ~len:(stop - eq - 1) in
              if String.is_empty value then None else go (stop + 1) ((key, value) :: acc)
    in
    go 0 []

  (** [parse line] is the progress line [line] read, or [None] when it is not one or breaks the
      documented shape: [wall_s] (a number) and [event] must lead, in that order. *)
  let parse line =
    Option.bind (String.chop_prefix line ~prefix) ~f:(fun rest ->
        match parse_fields rest with
        | Some (("wall_s", w) :: ("event", event) :: _ as fields) ->
            Option.map (Float.of_string_opt w) ~f:(fun wall_s -> { event; wall_s; fields })
        | _ -> None)

  (** Whether [line] starts as a progress line, whatever follows. *)
  let is_progress line = String.is_prefix line ~prefix

  (** The progress lines among [lines] that parse, in order. *)
  let of_lines lines = List.filter_map lines ~f:parse

  (** [events lines name] are the progress lines among [lines] of event [name]. *)
  let events lines name = List.filter (of_lines lines) ~f:(fun line -> String.equal line.event name)

  let field line key = List.Assoc.find line.fields key ~equal:String.equal
  let float_field line key = Option.bind (field line key) ~f:Float.of_string_opt
  let int_field line key = Option.bind (field line key) ~f:Int.of_string_opt

  (** [count lines pairs] counts the progress lines among [lines] carrying every [(key, value)] of
      [pairs]. *)
  let count lines pairs =
    List.count (of_lines lines) ~f:(fun line ->
        List.for_all pairs ~f:(fun (key, value) ->
            Option.equal String.equal (field line key) (Some value)))
end
