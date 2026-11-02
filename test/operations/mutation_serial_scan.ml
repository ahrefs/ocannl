(* gh-ocannl-1506: every write to a module-level mutable binding of the inference-session modules
   advances Row.mutation_serial, or is classified in place as not needing to. The reader, its closed
   lists and what it deliberately does not read are in test/support/mutation_serial_scan.ml's
   header; the classification lives on the bindings themselves, as [@@mutation_serial.*] attributes
   in tensor/row.ml, tensor/shape.ml and tensor/tensor.ml (gh-ocannl-1573 added tensor.ml, whose
   session records are classified field by field).

   Live: [mutation_serial_scan.exe <root>] reads <root>/tensor/{row,shape,tensor}.ml with their
   .mli. Control: [mutation_serial_scan.exe --fixture <root>] reads the same layout under a fixture
   tree, each file with a [.fixture] suffix so that no other scan or build reads it as a source, and
   ends as a negative control: its golden is the findings, and it must exit 1. *)
open Base
open Stdio
module Scan = Test_utils.Mutation_serial_scan

(* The corpus is a policy, not a census: the three modules holding the inference session's state. *)
let corpus = [ "tensor/row.ml"; "tensor/shape.ml"; "tensor/tensor.ml" ]

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
      match
        Test_utils.Scan_attempt.run (fun () ->
            Scan.scan ~file
              ~serial_owner:(String.is_suffix file ~suffix:"row.ml")
              ~require_capture:true ?interface source)
      with
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
      Verdict.p "the scan tracks a noted ref, captured by capture_globals (row.ml's safe_to_guess)"
        (has
           "tensor/row.ml: safe_to_guess (ref) -- accounted: every write noted or at a boundary; \
            captured by capture_globals");
      (* gh-ocannl-903: a registry a rewind extends rather than restores. *)
      Verdict.p "the scan tracks a kept registry (tensor.ml's discarded_nodes_table)"
        (has "tensor/tensor.ml: discarded_nodes_table (ref) -- accounted; kept by rewinds");
      Verdict.p "the scan tracks an exempt counter (row.ml's uid)"
        (has "tensor/row.ml: uid (ref) -- exempt");
      Verdict.p "the scan tracks a fingerprinted ref (shape.ml's state)"
        (has "tensor/shape.ml: state (ref) -- fingerprinted");
      (* gh-ocannl-1573: module-level record values, judged field by field. *)
      Verdict.p
        "the scan tracks an accounted record field (tensor.ml's session_state.forward_roots)"
        (has "tensor/tensor.ml: session_state.forward_roots (mutable field) -- accounted");
      Verdict.p "the scan tracks an exempt record field (tensor.ml's session_state.next_id)"
        (has "tensor/tensor.ml: session_state.next_id (mutable field) -- exempt");
      Verdict.p "the scan tracks a kept record field (tensor.ml's handouts.fwd_out)"
        (has "tensor/tensor.ml: handouts.fwd_out (mutable field) -- accounted; kept by rewinds")
  | _ -> Stdlib.exit 2
