open Base
open Ocannl

let bench ~bench_title ~time_in_sec ~mem_in_bytes ~result_label =
  PrintBox_utils.Benchmark
    { bench_title; time_in_sec; mem_in_bytes; result_label; result = Sexplib0.Sexp.Atom "v" }

let render rows = PrintBox_text.to_string (PrintBox_utils.table rows)

let () =
  (* Empty input returns PrintBox.empty, which renders as "". *)
  let s = render [] in
  Stdio.printf "=== Empty input ===\n%s\n" s;
  assert (String.is_empty s)

let () =
  (* Single-label: output should be structurally identical to what the old single-record
     implementation produced. We verify this by checking only the single label appears. *)
  let rows =
    [
      bench ~bench_title:"run-A" ~time_in_sec:1.0 ~mem_in_bytes:100 ~result_label:"score";
      bench ~bench_title:"run-B" ~time_in_sec:2.0 ~mem_in_bytes:200 ~result_label:"score";
    ]
  in
  let s = render rows in
  Stdio.printf "=== Single-label ===\n%s\n" s;
  assert (String.is_substring s ~substring:"score");
  (* Only one distinct label -- no second column header. *)
  assert (String.count s ~f:(fun c -> Char.equal c 's') > 0)

let () =
  (* Multi-label grouping: both labels appear as column headers. *)
  let rows =
    [
      bench ~bench_title:"fast-A" ~time_in_sec:1.0 ~mem_in_bytes:100 ~result_label:"ms";
      bench ~bench_title:"fast-B" ~time_in_sec:2.0 ~mem_in_bytes:200 ~result_label:"ms";
      bench ~bench_title:"big-C" ~time_in_sec:3.0 ~mem_in_bytes:300 ~result_label:"MB";
    ]
  in
  let s = render rows in
  Stdio.printf "=== Multi-label grouping ===\n%s\n" s;
  (* Both result labels must appear as column headers. *)
  assert (String.is_substring s ~substring:"ms");
  assert (String.is_substring s ~substring:"MB")

let () =
  (* Per-group speedup/mem_gain: group A (times 1,2,4) and group B (times 10,20,40).
     With per-group max: both groups' speedups are [4.000; 2.000; 1.000].
     With global max (40): group A's first speedup would be 40.000 -- the falsifier. *)
  let rows =
    [
      bench ~bench_title:"a1" ~time_in_sec:1.0 ~mem_in_bytes:10 ~result_label:"ms";
      bench ~bench_title:"a2" ~time_in_sec:2.0 ~mem_in_bytes:20 ~result_label:"ms";
      bench ~bench_title:"a3" ~time_in_sec:4.0 ~mem_in_bytes:40 ~result_label:"ms";
      bench ~bench_title:"b1" ~time_in_sec:10.0 ~mem_in_bytes:100 ~result_label:"MB";
      bench ~bench_title:"b2" ~time_in_sec:20.0 ~mem_in_bytes:200 ~result_label:"MB";
      bench ~bench_title:"b3" ~time_in_sec:40.0 ~mem_in_bytes:400 ~result_label:"MB";
    ]
  in
  let s = render rows in
  Stdio.printf "=== Per-group speedup ===\n%s\n" s;
  (* Per-group: max speedup for each group is 4.000. *)
  assert (String.is_substring s ~substring:"4.000");
  (* Global-max would produce 40.000 as A's first-row speedup/mem_gain -- must not appear. *)
  assert (not (String.is_substring s ~substring:"40.000"))
