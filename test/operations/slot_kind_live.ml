(** What {!Test_utils.Slot_kind} says of the repository's own tree (gh-ocannl-1227): the checks that
    keep its two configuration-free answers true hold -- no alias on the configuration-free list
    reaches a configuration reader, a plain compilation reaches no backend, every reader reads a
    configuration the runner resolves (gh-ocannl-1249), and the inventory is complete -- and the
    answers they guard are the ones tools/test-run.sh gets. A stanza that breaks a check fails here,
    in [@test/operations/scans], rather than passing as a batch the runner takes to read nothing.
    How each answer and each refusal is decided, on fixture trees built to separate the cases, is
    [slot_kind_cases]'s.

    It reads exactly the files the rule hands it -- the dune files, the [ocannl_config] files, and
    the metadata the runner's walk refuses ([dune-project]s, a [dune-workspace], a [dune-file], a
    [.t] cram test) -- and never walks the build tree, which other tests' outputs share with it
    while it runs: what it reads is what dune invalidates it on. *)

open Base
open Stdio
open Verdict.Claims
module Scan = Test_utils.Dune_stanza_scan
module Slot_kind = Test_utils.Slot_kind

let () =
  if Array.length Stdlib.Sys.argv < 2 then (
    eprintf "Usage: %s <workspace_root> <declared input...>\n" Stdlib.Sys.argv.(0);
    Stdlib.exit 1);
  (* Paths arrive relative to the test's directory deep in the build tree; [Scan.base_dir] and
     [Scan.repo_relative] turn them into repository-relative ones. *)
  let base = Scan.base_dir Stdlib.Sys.argv.(1) in
  let handed = Array.to_list (Array.subo Stdlib.Sys.argv ~pos:2) in
  let opened = ref [] in
  let read path =
    opened := path :: !opened;
    In_channel.read_all path
  in
  let paths =
    List.map handed ~f:(fun path -> (Scan.repo_relative base path, path))
    |> List.dedup_and_sort ~compare:(fun (a, _) (b, _) -> String.compare a b)
  in
  let of_basename name =
    List.filter paths ~f:(fun (path, _) -> String.equal (Stdlib.Filename.basename path) name)
  in
  let dir_of path = match Stdlib.Filename.dirname path with "." -> "" | dir -> dir in
  let dune_files =
    List.map (of_basename "dune") ~f:(fun (path, on_disk) -> (dir_of path, read on_disk))
  in
  let config_files = List.map (of_basename Scan.config_file) ~f:fst in
  (* What dune reads beside the dune files, which the runner's own walk refuses
     ([Slot_kind.dune_files]): every dune-project's dialect, a workspace naming more than its
     language, the alternative dune-file name, and an implicitly discovered cram test. Declared too,
     so the scan refuses what would make the runner answer every backend (Codex review round 2 on
     staging#1318). *)
  let checked f path on_disk =
    match f ~path (read on_disk) with () -> None | exception Failure why -> Some why
  in
  let metadata =
    (if List.exists paths ~f:(fun (path, _) -> String.equal path "dune-project") then []
     else [ "no root dune-project among the declared inputs" ])
    @ List.filter_map paths ~f:(fun (path, on_disk) ->
        match Stdlib.Filename.basename path with
        | "dune-project" -> checked Slot_kind.check_project path on_disk
        | "dune-workspace" -> checked Slot_kind.check_workspace path on_disk
        | "dune-file" -> Some (path ^ " is an alternative dune-file")
        | base when String.is_suffix base ~suffix:".t" -> Some (path ^ " is a cram test")
        | _ -> None)
  in
  (* The counts move with the tree, so they go to stderr, which the golden does not record. *)
  eprintf "read %d dune files and %d configurations\n" (List.length dune_files)
    (List.length config_files);
  let refusals = Slot_kind.refusals ~config_files dune_files @ metadata in
  List.iter refusals ~f:(fun r -> printf "refused: %s\n" r);
  List.iter
    [
      "build @test/operations/scans";
      "build @metal-codegen";
      "build bin/projection_shape_bench.exe";
      "build ./test/operations/test_vec_simple.exe";
      "build @lifecycle";
      "build @test/operations/runtest-env_var_deps";
    ] ~f:(fun argv ->
      let answer = Slot_kind.answer ~dune_files (String.split argv ~on:' ') in
      let why =
        match answer with
        | Reaches { reads_config = Some why; _ } -> why
        | Reaches { reads_config = None; _ } -> "<the configuration does not count>"
        | Unknown why -> "unknown: " ^ why
      in
      printf "%-45s %s\n    %s\n" argv (Slot_kind.summary answer) why);
  let summary argv = Slot_kind.summary (Slot_kind.answer ~dune_files (String.split argv ~on:' ')) in
  p "the live tree is a dune-file inventory, not an empty glob" (List.length dune_files > 10);
  p_empty "the live tree breaks no check that keeps the configuration-free answers true"
    ~over:dune_files refusals;
  p "@test/operations/scans holds no backend"
    (String.equal (summary "build @test/operations/scans") "names nothing");
  p "@metal-codegen holds Metal and nothing else"
    (String.equal (summary "build @metal-codegen") "names metal");
  p "a plain build of a bin/ executable holds no backend"
    (String.equal (summary "build bin/projection_shape_bench.exe") "names nothing");
  p "a plain build in bisect's spelling holds no backend"
    (String.equal (summary "build ./test/operations/test_vec_simple.exe") "names nothing");
  p "@lifecycle, off the list, counts the configuration"
    (String.is_suffix (summary "build @lifecycle") ~suffix:" + reads config");
  p "a single none-marked test off the list counts the configuration"
    (String.equal
       (summary "build @test/operations/runtest-env_var_deps")
       "names nothing + reads config");
  p_all "every file read is one the rule declared" !opened ~f:(fun path ->
      List.mem handed path ~equal:String.equal)
