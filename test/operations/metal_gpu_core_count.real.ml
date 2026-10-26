(* gh-ocannl-1638: Metal publishes no GPU core count, so the backend reads the launch fill's
   multiprocessor count from the IORegistry: the AGX accelerator's "gpu-core-count", found by the
   Metal device's registry ID through IOKit. This test reads the same property a second, independent
   way -- `ioreg -rc AGXAccelerator -d1`, the tool's text -- and pins the relationship: the
   backend's [multiprocessors] is ioreg's count (both absent on a GPU with no AGX entry, as in a
   paravirtual VM). Where ioreg reports a count, the ranking score's launch fill also resolves with
   no configuration, on that count; on a machine with none that leg is skipped (a host capability,
   not the backend).

   Negative control: before gh-ocannl-1638 the backend reported [None], so on an Apple-silicon Mac
   both claims are false. *)

open! Base
open Verdict.Claims

(* Every ["gpu-core-count" = N] line ioreg prints for the AGX accelerators, minimum over them, the
   reading the backend takes over devices; [None] when there are none. *)
let ioreg_core_count () =
  let ch =
    Unix.open_process_args_in "/usr/sbin/ioreg" [| "ioreg"; "-rc"; "AGXAccelerator"; "-d1" |]
  in
  let text = Stdio.In_channel.input_all ch in
  (match Unix.close_process_in ch with
  | Unix.WEXITED 0 -> ()
  | _ -> fail "ioreg -rc AGXAccelerator -d1 did not exit 0");
  String.split_lines text
  |> List.filter_map ~f:(fun line ->
      match String.lsplit2 (String.strip line) ~on:'=' with
      | Some (key, count) when String.equal (String.strip key) "\"gpu-core-count\"" ->
          Some (Int.of_string (String.strip count))
      | _ -> None)
  |> List.min_elt ~compare:Int.compare

let claim_count = "Metal's multiprocessors is the IORegistry's gpu-core-count as ioreg prints it"

let claim_fill =
  "with no configuration, the launch fill resolves on the IORegistry's gpu-core-count"

let () =
  let ioreg = ioreg_core_count () in
  let limits = Context.hardware_limits (Context.metal ()) in
  let show = Option.value_map ~default:"none" ~f:Int.to_string in
  Stdio.eprintf "ioreg gpu-core-count: %s; backend multiprocessors: %s (not part of the golden)\n%!"
    (show ioreg) (show limits.multiprocessors);
  p claim_count (Option.equal Int.equal limits.multiprocessors ioreg);
  match ioreg with
  | None -> skipped ~aggregation:`Environment ~backend:"metal" claim_fill
  | Some count ->
      p claim_fill
        (match Autotune.ranking_fill_model ~limits with
        | Some fm -> fm.Ir.Cost_model.fm_multiprocessors = count && fm.fm_fill_threads > 0
        | None -> false)
