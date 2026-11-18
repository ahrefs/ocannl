open Base

type t = Unix.process_status * string * string

let absolute path =
  if Stdlib.Filename.is_relative path then Stdlib.Filename.concat (Stdlib.Sys.getcwd ()) path
  else path

let executable () = absolute Stdlib.Sys.executable_name

let rec wait pid =
  match Unix.waitpid [] pid with
  | _, status -> status
  | exception Unix.Unix_error (Unix.EINTR, _, _) -> wait pid

(* Dune sets these in every action's environment: [INSIDE_DUNE] changes how a Dune reading it finds
   its root, configuration and output paths, and a Dune exiting with [DUNE_ACTION_TRACE_DIR] set
   copies its own trace into that directory, which the parent collects as the action's trace. *)
let dune_action_variables = [ "INSIDE_DUNE"; "DUNE_ACTION_TRACE_DIR" ]

let nested_dune_environment ?(env = Unix.environment ()) () =
  Array.filter env ~f:(fun binding ->
      not
        (List.exists dune_action_variables ~f:(fun var ->
             String.is_prefix binding ~prefix:(var ^ "="))))

let run ?exe ?cwd ?temp_dir ?env ?(stderr = `Capture) args =
  let exe = absolute (Option.value exe ~default:(executable ())) in
  let paths = ref [] and descriptors = ref [] in
  Exn.protect
    ~finally:(fun () ->
      let errors = ref [] in
      let clean f x = try f x with exn -> errors := exn :: !errors in
      List.iter !descriptors ~f:(clean Unix.close);
      List.iter !paths ~f:(clean Unix.unlink);
      match List.rev !errors with [] -> () | exn :: _ -> Stdlib.raise exn)
    ~f:(fun () ->
      let capture suffix =
        let path = Stdlib.Filename.temp_file ?temp_dir "ocannl-child-" suffix in
        paths := path :: !paths;
        let fd = Unix.openfile path [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
        descriptors := fd :: !descriptors;
        Unix.set_close_on_exec fd;
        (path, fd)
      in
      let out_path, out = capture ".out" in
      let err_path, err =
        match stderr with
        | `Capture ->
            let path, fd = capture ".err" in
            (Some path, fd)
        | `Discard ->
            let fd = Unix.openfile Stdlib.Filename.null [ Unix.O_WRONLY ] 0 in
            descriptors := fd :: !descriptors;
            Unix.set_close_on_exec fd;
            (None, fd)
        | `Into fd -> (None, fd)
      in
      let argv = Array.of_list (exe :: args) in
      let spawn () =
        match env with
        | None -> Unix.create_process exe argv Unix.stdin out err
        | Some env -> Unix.create_process_env exe argv env Unix.stdin out err
      in
      let pid =
        match cwd with
        | None -> spawn ()
        | Some root ->
            let here = Stdlib.Sys.getcwd () in
            Exn.protect
              ~finally:(fun () -> Unix.chdir here)
              ~f:(fun () ->
                Unix.chdir root;
                spawn ())
      in
      List.iter !descriptors ~f:Unix.close;
      descriptors := [];
      let status = wait pid in
      ( status,
        Stdio.In_channel.read_all out_path,
        Option.value_map err_path ~default:"" ~f:Stdio.In_channel.read_all ))

let output (_, stdout, stderr) = stdout ^ stderr

let matches ?(stream = `Both) ~exit ~contains (status, stdout, stderr) =
  Poly.equal status (Unix.WEXITED exit)
  && List.for_all contains ~f:(fun substring ->
      let in_stream text = String.is_substring text ~substring in
      match stream with
      | `Stdout -> in_stream stdout
      | `Stderr -> in_stream stderr
      | `Both -> in_stream stdout || in_stream stderr)

let describe_status = function
  | Unix.WEXITED n -> Printf.sprintf "exited %d" n
  | Unix.WSIGNALED n -> Printf.sprintf "was killed by signal %d" n
  | Unix.WSTOPPED n -> Printf.sprintf "was stopped by signal %d" n

let prefixed text =
  String.split text ~on:'\n'
  |> List.map ~f:(fun line -> "  child | " ^ line)
  |> String.concat ~sep:"\n"

let report ~label (status, stdout, stderr) =
  Stdio.eprintf "%s: child %s. stdout:\n%s\nstderr:\n%s\n" label (describe_status status)
    (prefixed stdout) (prefixed stderr)
