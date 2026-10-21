(* tools/refusal-manifest-sync.sh's engine (gh-ocannl-1429): rewrites the stale rows, keys and
   labels of test/support/refusal_control_manifest.ml from the scanners' current diagnostics, and
   names the goldens to re-run. Run from the repository root; the wrapper does that, reformats the
   manifest and runs the goldens. Usage: refusal_manifest_sync.exe [--dry-run] [--base REV]
   [SOURCE.ml ...] *)

open Base
open Stdio
module Sync = Test_utils.Refusal_manifest_sync
module Scan = Test_utils.Refusal_control_scan
module Stanza = Test_utils.Dune_stanza_scan

let manifest_path = "test/support/refusal_control_manifest.ml"
let operations = "test/operations"

let git args =
  let channel = Unix.open_process_args_in "git" (Array.of_list ("git" :: args)) in
  let output = In_channel.input_all channel in
  match Unix.close_process_in channel with Unix.WEXITED 0 -> Some output | _ -> None

let diagnostics_of source =
  let path = Stdlib.Filename.concat operations source in
  (* Read only when present: a row whose source is gone is the plan's UNRESOLVED item, not a
     crash. *)
  if Stdlib.Sys.file_exists path then Some (Scan.diagnostics (In_channel.read_all path)) else None

let refuse fmt =
  Printf.ksprintf
    (fun message ->
      eprintf "refusal-manifest-sync: %s\n" message;
      Stdlib.exit 2)
    fmt

(* The control claims reworded since [base] in the test sources the working tree changed, and the
   old ones nothing replaced one for one. A [base] git cannot diff against refuses: read as "nothing
   changed", it would report a stale label as nothing to sync. A source new since [base] has no old
   claims; one that does not parse on either side contributes nothing (the scans report it). *)
let claim_changes base =
  let changed =
    match git [ "diff"; "--name-only"; base; "--"; "test/operations"; "test/support" ] with
    | None -> refuse "git cannot diff against --base %s" base
    | Some listing ->
        String.split_lines listing
        |> List.filter ~f:(fun path ->
            String.is_suffix path ~suffix:".ml" && Stdlib.Sys.file_exists path)
  in
  let renamed, unpaired =
    List.map changed ~f:(fun path ->
        match git [ "show"; base ^ ":" ^ path ] with
        | None -> ([], [])
        | Some before -> (
            try Sync.claim_renames ~before ~after:(In_channel.read_all path) with _ -> ([], [])))
    |> List.unzip
  in
  (List.concat renamed, List.concat unpaired)

(* Each catalogued source's goldens: the refusal_control_scan_cases stanza's argument pairs. *)
let catalogue () =
  let stanzas = Stanza.stanzas (In_channel.read_all (Stdlib.Filename.concat operations "dune")) in
  let audit =
    List.find stanzas ~f:(fun stanza ->
        List.mem (Stanza.names_of stanza) "refusal_control_scan_cases" ~equal:String.equal)
  in
  let arguments =
    Option.bind audit ~f:(fun stanza -> Stanza.field stanza "action")
    |> Option.value ~default:[] |> List.concat_map ~f:Stanza.atoms
    |> List.drop_while ~f:(Fn.non (String.equal "%{test}"))
    |> Fn.flip List.drop 1
  in
  let rec pairs = function
    | source :: goldens :: rest -> (source, String.split goldens ~on:',') :: pairs rest
    | _ -> []
  in
  pairs arguments

let () =
  let dry_run = ref false and base = ref "HEAD" and named = ref [] in
  let rec parse = function
    | "--dry-run" :: rest ->
        dry_run := true;
        parse rest
    | "--base" :: rev :: rest ->
        base := rev;
        parse rest
    | source :: rest when not (String.is_prefix source ~prefix:"-") ->
        named := Stdlib.Filename.basename source :: !named;
        parse rest
    | [] -> ()
    | argument :: _ ->
        eprintf "refusal-manifest-sync: unknown argument %s\n" argument;
        Stdlib.exit 2
  in
  parse (List.tl_exn (Array.to_list (Sys.get_argv ())));
  (* Every argument is checked before anything is written: a refusal leaves the tree as it was. *)
  let catalogue = catalogue () in
  List.iter !named ~f:(fun source ->
      if not (List.Assoc.mem catalogue source ~equal:String.equal) then
        refuse "%s is no source the refusal_control_scan_cases stanza catalogues" source);
  let renamed_claims, unpaired_claims = claim_changes !base in
  let text = In_channel.read_all manifest_path in
  let plan = Sync.plan ~text ~diagnostics_of ~renamed_claims ~unpaired_claims () in
  List.iter plan.changes ~f:(function
    | Sync.Row { source; added; removed } ->
        printf "row %s: %d marker(s) added, %d removed\n" source (List.length added)
          (List.length removed);
        List.iter removed ~f:(printf "  - %s\n");
        List.iter added ~f:(printf "  + %s\n")
    | Sync.Rekeyed { table; old_key; new_key } ->
        printf "re-keyed in %s: %s -> %s\n" table old_key new_key
    | Sync.Relabeled { key; old_label; new_label } ->
        printf "relabeled %s:\n  - %s\n  + %s\n" key old_label new_label);
  List.iter plan.unresolved ~f:(printf "UNRESOLVED: %s\n");
  if (not !dry_run) && not (List.is_empty plan.edits) then
    Out_channel.write_all manifest_path ~data:(Sync.apply text plan.edits);
  let sources = List.dedup_and_sort (Sync.touched_sources plan @ !named) ~compare:String.compare in
  (* The goldens the rewording's text reaches, rewritten in place unless this is a dry run. *)
  let rewritten =
    Stdlib.Sys.readdir operations |> Array.to_list |> List.sort ~compare:String.compare
    |> List.filter ~f:(fun file ->
        String.is_suffix file ~suffix:".expected"
        &&
        let path = Stdlib.Filename.concat operations file in
        let golden = In_channel.read_all path in
        let updated = Sync.rewrite_golden plan golden in
        (not (String.equal golden updated))
        &&
        (if not !dry_run then Out_channel.write_all path ~data:updated;
         printf "golden %s: %s its replaced markers and reworded claim rows\n" file
           (if !dry_run then "would rewrite" else "rewrote");
         true))
  in
  let goldens =
    List.concat_map sources ~f:(fun source ->
        Option.value (List.Assoc.find catalogue source ~equal:String.equal) ~default:[])
    @ rewritten
    @
    if List.is_empty sources && List.is_empty rewritten then []
    else [ "refusal_control_scan_cases.expected" ]
  in
  List.dedup_and_sort goldens ~compare:String.compare
  |> List.iter ~f:(fun golden ->
      printf "rerun: @%s/runtest-%s\n" operations
        (String.chop_suffix_exn golden ~suffix:".expected"));
  if List.is_empty plan.changes && List.is_empty !named then printf "nothing to sync\n";
  if not (List.is_empty plan.unresolved) then Stdlib.exit 1
