(** [Ll_test]'s IR dump: hand-built IR leaves [<name>-unoptimized.ll] and [<name>.ll] under the gate
    a backend's own lowering uses, and nothing with it off. [optimize_scoped] optimizes [raw] and
    only then swaps in [scoped], so its dump is [raw]'s, as its documentation says. *)

open Base
module Ll = Ll_test
module Generated = Test_utils.Generated
open Verdict.Claims

let node = Ll.node_factory ~first_id:17230 ~dims:[| 1 |] ()
let out = node "out"
let store v = Ll.set out [| Ll.fixed 0 |] v
let optimize ~name llc = Ll.optimize ~materialized:[ out ] ~name llc

let with_gate on f =
  let saved = Utils.settings.output_debug_files_in_build_directory in
  Exn.protect
    ~finally:(fun () -> Utils.settings.output_debug_files_in_build_directory <- saved)
    ~f:(fun () ->
      Utils.settings.output_debug_files_in_build_directory <- on;
      f ())

let dumped ~ext name = Stdlib.Sys.file_exists (Generated.path ~ext name)
let holds ~ext name text = String.is_substring (Generated.read ~ext name) ~substring:text

let () =
  Ll.materialize out;
  Generated.init ~backend_name:"cc";
  (* [2 + 3] folds, so the two files of one run differ in whether the sum is still spelled. *)
  let sum = store (Ll.add (Ll.c 2.) (Ll.c 3.)) in
  case "off" (fun () ->
      let name = "ltd_off" in
      with_gate false (fun () -> ignore (optimize ~name sum : Ir.Low_level.optimized));
      p "gate off: no .ll" (not (dumped ~ext:".ll" name));
      p "gate off: no -unoptimized.ll" (not (dumped ~ext:"-unoptimized.ll" name)));
  case "on" (fun () ->
      let name = "ltd_on" in
      with_gate true (fun () -> ignore (optimize ~name sum : Ir.Low_level.optimized));
      p "gate on: -unoptimized.ll is the IR as handed in"
        (holds ~ext:"-unoptimized.ll" name "out[0] := (2.0 + 3.0)");
      p "gate on: .ll is the IR the pipeline left" (holds ~ext:".ll" name "out[0] := 5.0"));
  case "scoped" (fun () ->
      let name = "ltd_scoped" in
      with_gate true (fun () ->
          ignore
            (Ll.optimize_scoped ~materialized:[ out ] ~name ~raw:(store (Ll.c 7.)) (store (Ll.c 9.))
              : Ir.Low_level.optimized));
      p "optimize_scoped: .ll is raw's, not the scoped program"
        (holds ~ext:".ll" name "out[0] := 7.0"))
