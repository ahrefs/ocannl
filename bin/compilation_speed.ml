open Base
open Core_bench
open Ocannl
module FDSL = Operation.FDSL
module NFDSL = Operation.NFDSL
module CDSL = Code.CDSL

let recompiling_graph executor opti () =
  Code.CDSL.with_debug := false;
  Stdio.prerr_endline @@ "\n\n****** Benchmarking "
  ^ Sexp.to_string_hum (Session.sexp_of_backend executor)
  ^ " ******";
  let () = Session.SDSL.set_executor executor in
  Exec_as_gccjit.optimization_level := opti;
  (* let open Operation.FDSL in *)
  let open Session.SDSL in
  drop_all_sessions ();
  Random.init 0;
  let%nn_op f = (3 *. ("x" [ 5 ] **. 2)) - (4 *. x) + 5 in
  refresh_session ();
  let xs = Array.init 100 ~f:Float.(fun i -> of_int i - 50.) in
  let ys =
    Array.map xs ~f:(fun v ->
        let setval = compile_routine [%nn_cd x =: !.v] in
        setval ();
        refresh_session ();
        (value_1d_points ~xdim:0 f).(0))
  in
  let plot_box =
    let open PrintBox_utils in
    plot ~size:(75, 35) ~x_label:"x" ~y_label:"f(x)"
      [ Scatterplot { points = Array.zip_exn xs ys; pixel = "#" } ]
  in
  PrintBox_text.output Stdio.stdout plot_box;
  Stdio.print_endline "\n"

let benchmarks =
  [
    ("Interpreter", recompiling_graph Interpreter 3);
    ("OCaml", recompiling_graph OCaml 3);
    ("gccjit O0", recompiling_graph Gccjit 0);
    ("gccjit O1", recompiling_graph Gccjit 1);
    ("gccjit O2", recompiling_graph Gccjit 2);
    ("gccjit O3", recompiling_graph Gccjit 3);
  ]

let _suspended () = recompiling_graph Gccjit 3 ()

let () =
  List.map benchmarks ~f:(fun (name, test) -> Bench.Test.create ~name test)
  |> Bench.make_command |> Command_unix.run

(* Example output, before the single-use-not-memorized aka. virtual nodes optimization:

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
*)
