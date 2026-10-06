(** What {!Test_utils.Slot_kind} says a dune argv reaches on the repository's own tree (its dune
    files, copied beside this test by the stanza's deps; gh-ocannl-1095): the batch the issue is
    about, [@test/operations/scans], holds no backend, and the one review round 1 found reading the
    configuration through the [.actual] its diff consumes still does. A new stanza that breaks the
    proof for [scans] -- a construct not modelled exactly, anywhere compilation reaches -- shows
    here, rather than as scans quietly taking a GPU token again. How each construct is read, on
    fixture trees built to separate the cases, is [slot_kind_cases]'s. *)

open Base
open Stdio
open Verdict.Claims
module Slot_kind = Test_utils.Slot_kind

let () =
  let live = Slot_kind.dune_files ~workspace_root:false ~root:"../.." () in
  List.iter
    [
      ("build @test/operations/scans", "names nothing");
      ("build @test/operations/runtest-bandwidth_calibration", "names nothing + reads config");
    ]
    ~f:(fun (argv, want) ->
      let answer = Slot_kind.answer ~dune_files:live (String.split argv ~on:' ') in
      let shown = Slot_kind.summary answer in
      let why =
        match answer with
        | Reaches { reads_config = Some why; _ } -> why
        | Reaches { reads_config = None; _ } -> "<the configuration does not count>"
        | Unknown why -> "unknown: " ^ why
      in
      printf "%-55s %s\n    %s\n" argv shown why;
      p (Printf.sprintf "%s answers %s" argv want) (String.equal shown want))
