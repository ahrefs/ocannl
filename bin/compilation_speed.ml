open Base
open Ocannl
module FDSL = Operation.FDSL
module NFDSL = Operation.NFDSL
module CDSL = Code.CDSL
module SDSL = Session.SDSL

let recompiling_graph executor opti () =
  let open SDSL.O in
  SDSL.disable_all_debugs ();
  let bench_title = Sexp.to_string_hum ([%sexp_of: Session.backend * int] (executor, opti)) in
  Stdio.prerr_endline @@ "\n\n****** Benchmarking " ^ bench_title ^ " ******";
  let () = SDSL.set_executor executor in
  Exec_as_gccjit.optimization_level := opti;
  (* let open Operation.FDSL in *)
  SDSL.drop_all_sessions ();
  Random.init 0;
  let init_time = Time_now.nanoseconds_since_unix_epoch () in
  let%nn_op f = (3 *. ("x" [ 5 ] **. 2)) - (4 *. x) + 5 in
  SDSL.refresh_session ();
  let xs = Array.init 100 ~f:Float.(fun i -> of_int i - 50.) in
  let ys =
    Array.map xs ~f:(fun v ->
        SDSL.compile_routine [%nn_cd x =: !.v] ();
        SDSL.refresh_session ();
        f.@[0])
  in
  let plot_box =
    let open PrintBox_utils in
    plot ~size:(75, 35) ~x_label:"x" ~y_label:"f(x)"
      [ Scatterplot { points = Array.zip_exn xs ys; pixel = "#" } ]
  in
  let final_time = Time_now.nanoseconds_since_unix_epoch () in
  let time_in_sec = Int63.(to_float @@ (final_time - init_time)) /. 1000_000_000. in
  let result =
    PrintBox_utils.Benchmark { bench_title; time_in_sec; total_size_in_bytes = SDSL.global_size_in_bytes () }
  in
  PrintBox_text.output Stdio.stdout plot_box;
  Exec_as_gccjit.optimization_level := 3;
  Stdio.print_endline "\n";
  result

let benchmarks =
  [
    recompiling_graph Interpreter 3;
    recompiling_graph OCaml 3;
    recompiling_graph Gccjit 0;
    recompiling_graph Gccjit 1;
    recompiling_graph Gccjit 2;
    recompiling_graph Gccjit 3;
  ]

let _suspended () = recompiling_graph Gccjit 3 ()

let () =
  List.map benchmarks ~f:(fun bench -> bench ()) |> PrintBox_utils.table |> PrintBox_text.output Stdio.stdout

(* Example output, before the single-use-not-memorized aka. virtual nodes optimization,
       when still using core_bench:

   ┌─────────────┬─────────────┬────────────┬────────────┬──────────┬────────────┐
   │ Name        │    Time/Run │    mWd/Run │   mjWd/Run │ Prom/Run │ Percentage │
   ├─────────────┼─────────────┼────────────┼────────────┼──────────┼────────────┤
   │ Interpreter │      3.08ms │   970.34kw │   109.17kw │ 108.13kw │      0.02% │
   │ OCaml       │ 12_698.98ms │ 1_082.35kw │ 1_205.24kw │  97.03kw │    100.00% │
   │ gccjit O0   │ 10_561.48ms │   786.42kw │   100.28kw │ 100.28kw │     83.17% │
   │ gccjit O1   │ 10_699.58ms │   786.42kw │   100.28kw │ 100.28kw │     84.26% │
   │ gccjit O2   │ 10_683.29ms │   786.42kw │   100.28kw │ 100.28kw │     84.13% │
   │ gccjit O3   │ 11_478.59ms │   786.42kw │   100.28kw │ 100.28kw │     90.39% │
   └─────────────┴─────────────┴────────────┴────────────┴──────────┴────────────┘

   More recent:
   ┌───────────────┬────────────┬───────────────┬─────────────────┬──────────────────┐
   │Benchmarks     │Time in sec │Memory in bytes│Speedup vs. worst│Mem gain vs. worst│
   ├───────────────┼────────────┼───────────────┼─────────────────┼──────────────────┤
   │(Interpreter 3)│0.002409401 │500            │4846.12966501    │1.                │
   ├───────────────┼────────────┼───────────────┼─────────────────┼──────────────────┤
   │(OCaml 3)      │11.582041125│500            │1.00813574524    │1.                │
   ├───────────────┼────────────┼───────────────┼─────────────────┼──────────────────┤
   │(Gccjit 0)     │9.868432101 │500            │1.18319400098    │1.                │
   ├───────────────┼────────────┼───────────────┼─────────────────┼──────────────────┤
   │(Gccjit 1)     │9.941047054 │500            │1.17455129199    │1.                │
   ├───────────────┼────────────┼───────────────┼─────────────────┼──────────────────┤
   │(Gccjit 2)     │9.850038416 │500            │1.18540346422    │1.                │
   ├───────────────┼────────────┼───────────────┼─────────────────┼──────────────────┤
   │(Gccjit 3)     │11.676269661│500            │1.               │1.                │
   └───────────────┴────────────┴───────────────┴─────────────────┴──────────────────┘
*)
