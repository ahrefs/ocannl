(* gh-ocannl-866: the PPX expectation goldens and [.ocamlformat-ignore] stay in correspondence.

   A PPX output fixture is deliberately hostile to ocamlformat: formatting it changes the golden,
   its test promotes the original text back, and `dune build @fmt` -- the CI gate every PR passes
   through -- stays red until the file is ignored. The ignore entry used to be a prose-only
   obligation. When an entry was appended to an ignore file lacking its final newline, two paths
   became one nonexistent path and both goldens silently stopped being ignored.

   This scan holds both directions directly: every [test/ppx/*_expected.ml] visible in dune's clean
   declared-input sandbox is an entry, and every entry names a file there. It also requires one
   nonempty path per line and a final newline, so the append failure is refused at the boundary that
   enabled it.

   The live tree is normally valid, so its refusals are otherwise dead code. [control] builds a
   synthetic tree, invokes this executable as a child, and asserts both exit status and diagnostic
   text for every refusal. Capturing the streams keeps the designed children's [FAIL:] markers out
   of a green suite log, following [generated_provenance]'s pattern. *)

open Base
open Stdio

let printf = Test_utils.Refusal_control_manifest.printf

module Inventory = Test_utils.Source_inventory

type line = { number : int; entry : string }

let is_ppx_golden path =
  match String.chop_prefix path ~prefix:"test/ppx/" with
  | Some basename ->
      (not (String.mem basename '/'))
      && (not (String.mem basename '\\'))
      && String.is_suffix basename ~suffix:"_expected.ml"
  | None -> false

let strip_cr line = Option.value (String.chop_suffix line ~suffix:"\r") ~default:line

let lines_of content =
  let pieces = String.split content ~on:'\n' in
  let pieces =
    if String.is_suffix content ~suffix:"\n" then List.drop_last_exn pieces else pieces
  in
  List.mapi pieces ~f:(fun index entry -> { number = index + 1; entry = strip_cr entry })

let report_details details = List.iter details ~f:(eprintf "%s\n")

let scan ~path_root ~ignore_file ~generated =
  let inventory = Inventory.of_dune_sandbox ~workspace_root:path_root ~generated in
  let goldens =
    Inventory.select inventory ~f:is_ppx_golden
    |> List.map ~f:(fun (file : Inventory.file) -> file.path)
  in
  let declared_file = Inventory.mem inventory in
  let content = In_channel.read_all ignore_file in
  let lines = lines_of content in
  let nonempty, empty =
    List.partition_tf lines ~f:(fun { entry; _ } -> not (String.is_empty entry))
  in
  let entries = List.map nonempty ~f:(fun { entry; _ } -> entry) in
  let entry_set = Set.of_list (module String) entries in
  let trailing_newline = (not (String.is_empty content)) && String.is_suffix content ~suffix:"\n" in
  if not trailing_newline then
    eprintf
      ".ocamlformat-ignore does not end in a newline; appending a path would concatenate it with \
       the final entry\n";
  Verdict.p ".ocamlformat-ignore ends in a newline" trailing_newline;
  report_details
    (List.map empty ~f:(fun { number; _ } ->
         Printf.sprintf
           ".ocamlformat-ignore:%d: blank line; each line must contain exactly one path" number));
  Verdict.p_empty "every .ocamlformat-ignore line contains exactly one path" ~over:lines empty;
  let missing_entries = List.filter nonempty ~f:(fun { entry; _ } -> not (declared_file entry)) in
  report_details
    (List.map missing_entries ~f:(fun { number; entry } ->
         Printf.sprintf ".ocamlformat-ignore:%d: listed path `%s` is not a declared source file"
           number entry));
  Verdict.p_all "every .ocamlformat-ignore entry names an existing file" nonempty
    ~f:(fun { entry; _ } -> declared_file entry);
  let unlisted = List.filter goldens ~f:(fun path -> not (Set.mem entry_set path)) in
  report_details
    (List.map unlisted ~f:(fun path ->
         Printf.sprintf "%s is a ppx-expectation golden missing from .ocamlformat-ignore" path));
  Verdict.p_all "every ppx-expectation golden is listed in .ocamlformat-ignore" goldens
    ~f:(Set.mem entry_set);
  eprintf "Scanned %d ignore entries and %d ppx-expectation goldens.\n" (List.length entries)
    (List.length goldens)

let write_file path data =
  let rec mkdirs dir =
    if not (String.equal dir Stdlib.Filename.current_dir_name || Stdlib.Sys.file_exists dir) then (
      mkdirs (Stdlib.Filename.dirname dir);
      try Unix.mkdir dir 0o755 with Unix.Unix_error (Unix.EEXIST, _, _) -> ())
  in
  mkdirs (Stdlib.Filename.dirname path);
  Out_channel.write_all path ~data

let rec remove_tree path =
  match Unix.lstat path with
  | { Unix.st_kind = Unix.S_DIR; _ } ->
      Array.iter (Stdlib.Sys.readdir path) ~f:(fun entry ->
          remove_tree (Stdlib.Filename.concat path entry));
      Unix.rmdir path
  | _ -> Unix.unlink path
  | exception Unix.Unix_error _ -> ()

let describe_status = function
  | Unix.WEXITED n -> Printf.sprintf "exited %d" n
  | Unix.WSIGNALED n -> Printf.sprintf "was killed by signal %d" n
  | Unix.WSTOPPED n -> Printf.sprintf "was stopped by signal %d" n

let run_child ~root ~exe ~excluded =
  let capture suffix = Stdlib.Filename.temp_file "fmt_ignore_control" suffix in
  let out_path = capture ".out" and err_path = capture ".err" in
  let open_capture path = Unix.openfile path [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
  let out = open_capture out_path and err = open_capture err_path in
  let ignore_file = Stdlib.Filename.concat root ".ocamlformat-ignore" in
  (* Exercise the shipping scan over a declared-input root with no Git metadata. [--scan-only]
     suppresses only the parent's control driver; without it every child would recursively stage
     another generation of controls. *)
  let argv = [| exe; "--scan-only"; root; ignore_file; excluded |] in
  let pid = Unix.create_process exe argv Unix.stdin out err in
  let _, status = Unix.waitpid [] pid in
  Unix.close out;
  Unix.close err;
  let text = In_channel.read_all out_path ^ In_channel.read_all err_path in
  (try Unix.unlink out_path with Unix.Unix_error _ -> ());
  (try Unix.unlink err_path with Unix.Unix_error _ -> ());
  (status, text)

let control () =
  let exe =
    let name = Stdlib.Sys.executable_name in
    if Stdlib.Filename.is_relative name then Stdlib.Filename.concat (Stdlib.Sys.getcwd ()) name
    else name
  in
  let fixture = Stdlib.Filename.temp_dir "fmt ignore control " "" in
  let root = Stdlib.Filename.concat fixture "declared tree" in
  let a = "test/ppx/a_expected.ml" and b = "test/ppx/b_expected.ml" in
  let extra = "fixtures/format_hostile.ml" in
  let undeclared = "build/stale_expected.ml" in
  let ignore = ".ocamlformat-ignore" in
  List.iter [ a; b ] ~f:(fun path -> write_file (Stdlib.Filename.concat root path) "golden\n");
  write_file (Stdlib.Filename.concat root extra) "fixture\n";
  let excluded = Stdlib.Filename.concat root undeclared in
  (* Simulate a generated action file that Dune places beside the declared source tree. *)
  write_file excluded "stale artifact\n";
  let run content =
    write_file (Stdlib.Filename.concat root ignore) content;
    run_child ~root ~exe ~excluded
  in
  let report label (status, text) =
    eprintf "the %s control %s. Its captured output:\n%s\n" label (describe_status status) text
  in
  let passed label (status, text) =
    let ok = match status with Unix.WEXITED 0 -> true | _ -> false in
    if not ok then report label (status, text);
    ok
  in
  let refused label ~messages (status, text) =
    let ok =
      (match status with Unix.WEXITED 1 -> true | _ -> false)
      && List.for_all messages ~f:(fun message -> String.is_substring text ~substring:message)
    in
    if not ok then report label (status, text);
    ok
  in
  let legitimate = run (a ^ "\n" ^ b ^ "\n" ^ extra ^ "\n") in
  let concatenated = run (a ^ b ^ "\n") in
  let unterminated = run (a ^ "\n" ^ b) in
  let blank_line = run (a ^ "\n\n" ^ b ^ "\n") in
  let stale_artifact = run (a ^ "\n" ^ b ^ "\n" ^ undeclared ^ "\n") in
  printf
    "Synthetic controls invoke the shipping scanner over a complete fixture and over each\n\
     malformed shape; refusal output is captured and matched below.\n\n";
  Verdict.p "a complete fixture with an extra existing path and no Git metadata passes"
    (passed "legitimate" legitimate);
  Verdict.p
    "a concatenated append is refused in both directions, naming the nonexistent entry and omitted \
     golden"
    (refused "concatenated append"
       ~messages:
         [
           "listed path `test/ppx/a_expected.mltest/ppx/b_expected.ml` is not a declared source \
            file";
           "test/ppx/b_expected.ml is a ppx-expectation golden missing from .ocamlformat-ignore";
         ]
       concatenated);
  Verdict.p "a missing trailing newline is refused with its append-corruption diagnostic"
    (refused "unterminated file"
       ~messages:[ ".ocamlformat-ignore does not end in a newline" ]
       unterminated);
  Verdict.p "a blank line is refused as a line containing no path"
    (refused "blank line"
       ~messages:[ ".ocamlformat-ignore:2: blank line; each line must contain exactly one path" ]
       blank_line);
  Verdict.p "an undeclared file present in the build root is refused as a stale artifact"
    (refused "stale artifact"
       ~messages:[ "listed path `build/stale_expected.ml` is not a declared source file" ]
       stale_artifact);
  Test_utils.Refusal_control_manifest.print "ocamlformat_ignore_scan.ml";
  remove_tree fixture

let usage () =
  eprintf
    "Usage: %s <declared-source root> <.ocamlformat-ignore> <generated exclusions...> | --control\n"
    Stdlib.Sys.argv.(0);
  Stdlib.exit 2

let () =
  match Array.to_list Stdlib.Sys.argv with
  | [ _; "--control" ] -> control ()
  | _ :: "--scan-only" :: path_root :: ignore_file :: generated ->
      scan ~path_root ~ignore_file ~generated
  | _ :: path_root :: ignore_file :: generated ->
      scan ~path_root ~ignore_file ~generated;
      control ()
  | _ -> usage ()
