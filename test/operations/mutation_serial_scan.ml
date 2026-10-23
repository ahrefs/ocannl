(* gh-ocannl-1506: every write to a module-level mutable binding of the inference-session modules
   advances Row.mutation_serial, or is classified in place as not needing to. The reader, its closed
   lists and what it deliberately does not read are in test/support/mutation_serial_scan.ml's
   header; the classification lives on the bindings themselves, as [@@mutation_serial.*] attributes
   in tensor/row.ml and tensor/shape.ml.

   Live: [mutation_serial_scan.exe <root>] reads <root>/tensor/{row,shape}.ml with their .mli.
   Control: [mutation_serial_scan.exe --fixture <root>] reads the same layout under a fixture tree,
   each file with a [.fixture] suffix so that no other scan or build reads it as a source, and ends
   as a negative control: its golden is the findings, and it must exit 1. *)
open Base
open Stdio
module Scan = Test_utils.Mutation_serial_scan

(* The corpus is a policy, not a census: the two modules holding the inference session's state. *)
let corpus = [ "tensor/row.ml"; "tensor/shape.ml" ]

let scan_file ~root ~suffix file =
  let read path =
    let full = Stdlib.Filename.concat root path ^ suffix in
    if Stdlib.Sys.file_exists full then Some (In_channel.read_all full) else None
  in
  match read file with
  | None ->
      Verdict.fail (Printf.sprintf "%s: a corpus file is missing -- the scan read nothing" file);
      None
  | Some source -> (
      let interface = read (file ^ "i") in
      match Test_utils.Scan_attempt.run (fun () -> Scan.scan ~file ?interface source) with
      | Scanned report -> Some report
      | Unparsed why ->
          Verdict.fail (Printf.sprintf "%s does not parse: %s" file why);
          None
      | Raised exn ->
          Verdict.fail (Printf.sprintf "the mutation-serial scan raised on %s: %s" file exn);
          None)

let run ?(suffix = "") root =
  let reports = List.filter_map corpus ~f:(scan_file ~root ~suffix) in
  List.iter reports ~f:(fun (report : Scan.report) -> List.iter report.findings ~f:Verdict.fail);
  reports

let () =
  match Array.to_list Stdlib.Sys.argv with
  | [ _; "--fixture"; root ] ->
      ignore (run ~suffix:".fixture" root : Scan.report list);
      Verdict.exit_negative_control ()
  | [ _; root ] ->
      let reports = run root in
      let inventory = List.concat_map reports ~f:(fun report -> report.inventory) in
      List.iter inventory ~f:print_endline;
      (* Canaries: a scan gone blind and an intact tree print the same nothing. These name a binding
         whose writes are noted, one exempt and one fingerprinted, so a reader that stopped tracking
         any of the three classes fails here. *)
      let has prefix = List.exists inventory ~f:(String.is_prefix ~prefix) in
      Verdict.p "the scan tracks a noted table (row.ml's safe_to_guess)"
        (has "tensor/row.ml: safe_to_guess (Hash_set) -- accounted");
      Verdict.p "the scan tracks an exempt counter (row.ml's uid)"
        (has "tensor/row.ml: uid (ref) -- exempt");
      Verdict.p "the scan tracks a fingerprinted ref (shape.ml's state)"
        (has "tensor/shape.ml: state (ref) -- fingerprinted")
  | _ -> Stdlib.exit 2
