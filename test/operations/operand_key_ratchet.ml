(* gh-ocannl-1018: operand fixtures that hand-roll a multi-axis key under a modulus, bypassing
   Ll_test.cycle's blind-axis guard. The reader, its boundary and what it deliberately does not see
   are in test/support/operand_key_scan.ml's header. *)
open Base
open Stdio
module Scan = Test_utils.Operand_key_scan
module Inventory = Test_utils.Source_inventory

(* A [Site] row absorbs ONE site: the file, the site's source text with whitespace collapsed (what
   the refusal prints), and why it is not an operand the guard should check. A [File] row absorbs
   every site of its file. A row nothing answers to is stale and fails, so converting a site deletes
   its row in the same commit. *)
let exemptions =
  let sin_scaled =
    "the remainder only scales a sin/cos term that already varies with every cell; the value is \
     not a function of the key"
  in
  [
    ( "test/support/ll_test.ml",
      Scan.File,
      "the guard itself: its remainders are the arithmetic it checks, not an operand" );
    ( "test/operations/discriminating_values.ml",
      Scan.Site "(((idcs.(0) * 20) + idcs.(1)) % 7)",
      "the hand-written idiom cycle is pinned equal to: an oracle, not an operand" );
    ( "test/operations/discriminating_values.ml",
      Scan.Site "((idcs.(0) + idcs.(1) + (2 * idcs.(2)) + (3 * idcs.(3))) % 7)",
      "the hand-written idiom weighted is pinned equal to: an oracle, not an operand" );
    ( "test/operations/fission_equivalence.ml",
      Scan.Site "(idcs.(0) + idcs.(1)) % classes",
      "a one-hot label: the remainder picks the class a batch row is labelled with" );
    ( "test/operations/bench_checksum_discrimination.ml",
      Scan.Site "(Bc.mix ~salt:0x00A5 (t / n) (t % n) % 97)",
      "the key is Bench_checksum.mix, the aperiodic mixer; the remainder only bounds its range" );
    ("test/operations/autotune_split_reduce.ml", Scan.Site "(x % 3)", sin_scaled);
    ("test/operations/autotune_split_reduce.ml", Scan.Site "(x % 4)", sin_scaled);
    ("test/operations/autotune_split_reduce.ml", Scan.Site "(x % 3)", sin_scaled);
    ("test/operations/autotune_split_reduce.ml", Scan.Site "(x % 3)", sin_scaled);
    ("test/operations/schedule_split_reduce.ml", Scan.Site "(x % 5)", sin_scaled);
    ("test/operations/wrap_prec_convert.ml", Scan.Site "(i % 5)", sin_scaled);
  ]

let scan ~exemptions root generated =
  let inventory = Inventory.of_dune_sandbox ~workspace_root:root ~generated in
  let sources = Inventory.select inventory ~f:Test_utils.Ll_test_scan.test_source in
  let rows =
    List.map sources ~f:(fun (file : Inventory.file) ->
        (file.path, Scan.sites (In_channel.read_all file.on_disk)))
  in
  eprintf "operand-key sites read: %d\n"
    (List.sum (module Int) rows ~f:(fun (_, sites) -> List.length sites));
  List.iter (Scan.violations ~exemptions rows) ~f:Verdict.fail;
  List.iter
    [ ("test/", 200); ("arrayjit/test/", 20) ]
    ~f:(fun (prefix, floor) ->
      let count =
        List.count sources ~f:(fun (file : Inventory.file) -> String.is_prefix file.path ~prefix)
      in
      printf "Source floor: %s >= %d\n" prefix floor;
      if count < floor then Verdict.fail (prefix ^ ": source inventory below floor"));
  List.iter exemptions ~f:(fun (path, kind, reason) ->
      match kind with
      | Scan.File -> printf "%s -- every site: %s\n" path reason
      | Scan.Site key -> printf "%s -- `%s`: %s\n" path key reason)

let () =
  match Array.to_list Stdlib.Sys.argv with
  | [ _; "--fixture"; root ] -> scan ~exemptions:[] root []
  | [ _; "--fixture-exempt"; root ] ->
      scan
        ~exemptions:
          [ ("test/new.ml", Scan.Site "(((idcs.(0) * 20) + idcs.(1)) % 5)", "control exemption") ]
        root []
  | _ :: root :: generated -> scan ~exemptions root generated
  | _ -> Stdlib.exit 2
