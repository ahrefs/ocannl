open Base
open Core_bench
open Ocannl
module FDSL = Operation.FDSL
module NFDSL = Operation.NFDSL
module CDSL = Code.CDSL

let classify_moons ~virtualize executor ~opti_level () =
  Code.CDSL.with_debug := false;
  Stdio.prerr_endline @@ "\n\n****** Benchmarking virtualized: "
  ^ Sexp.to_string_hum ([%sexp_of: bool * Session.backend * int] (virtualize, executor, opti_level))
  ^ " ******";
  Code.CDSL.virtualize_settings.virtualize <- virtualize;
  Session.SDSL.set_executor executor;
  Exec_as_gccjit.optimization_level := opti_level;
  (* let open Operation.FDSL in *)
  let open Session.SDSL in
  drop_all_sessions ();
  Random.init 0;
  (* let hid1 = 64 in *)
  let len = 400 in
  let batch = 10 in
  let epochs = 10000 in
  let steps = epochs * 2 * len / batch in
  let noise () = Random.float_range (-0.1) 0.1 in
  let moons_flat =
    Array.concat_map (Array.create ~len ())
      ~f:
        Float.(
          fun () ->
            let i = Random.int len in
            let v = of_int i * pi / of_int len in
            let c = cos v and s = sin v in
            [| c + noise (); s + noise (); 1.0 - c + noise (); 0.5 - s + noise () |])
  in
  let moons_flat = FDSL.init_const ~l:"moons_flat" ~b:[ epochs; batch ] ~o:[ 2 ] moons_flat in
  let moons_classes = Array.init (len * 2) ~f:(fun i -> if i % 2 = 0 then 1. else -1.) in
  let moons_classes = FDSL.init_const ~l:"moons_classes" ~b:[ epochs; batch ] ~o:[ 1 ] moons_classes in
  let%nn_op mlp x =
      "b6" 1
      + "w6"
        * !/("b4" 4
            + "w4"
              * !/("b2" 8
                  + ("w2" * !/("b1" 16 + ("w1" * x)))
                  + "b3" 8
                  + ("w3" * !/(b2 + (w2 * !/(b1 + (w1 * x))))))
            + ("b5" 4 + ("w5" * !/(b4 + (w4 * !/(b3 + (w3 * !/(b2 + (w2 * !/(b1 + (w1 * x)))))))))))
    in
  (* let%nn_op mlp x =
    "b6" 1
    + "w6"
      * !/("b5" 4
          + ("w5" * !/("b4" 4 + ("w4" * !/("b3" 8 + ("w3" * !/("b2" 8 + ("w2" * !/("b1" 16 + ("w1" * x)))))))))
          )
  in *)
  let%nn_dt session_step ~output_dims:[ 1 ] = n =+ 1 in
  let%nn_dt minus_lr ~output_dims:[ 1 ] = n =: -0.1 *. (!..steps - session_step) /. !..steps in
  minus_learning_rate := Some minus_lr;
  let%nn_op moons_input = moons_flat @.| session_step in
  let%nn_op moons_class = moons_classes @.| session_step in
  let points1 = ref [] in
  let points2 = ref [] in
  let losses = ref [] in
  let log_losses = ref [] in
  let learning_rates = ref [] in
  let%nn_op margin_loss = !/(1 - (moons_class *. mlp moons_input)) in
  let%nn_op ssq w = (w **. 2) ++ "...|...->... => 0" in
  let reg_loss =
    List.map ~f:ssq [ w1; w2; w3; w4; w5; w6; b1; b2; b3; b4; b5; b6 ] |> List.reduce_exn ~f:FDSL.O.( + )
  in
  let%nn_op total_loss = ((margin_loss ++ "...|... => 0") /. !..batch) + (0.001 *. reg_loss) in
  for step = 1 to steps do
    refresh_session ();
    if step <= len then (
      let points = value_2d_points ~xdim:0 ~ydim:1 moons_input in
      let classes = value_1d_points ~xdim:0 moons_class in
      let npoints1, npoints2 = Array.partitioni_tf points ~f:Float.(fun i _ -> classes.(i) > 0.) in
      points1 := npoints1 :: !points1;
      points2 := npoints2 :: !points2);
    if step % 1000 = 0 then (
      learning_rates := ~-.(minus_lr.@[0]) :: !learning_rates;
      losses := total_loss.@[0] :: !losses;
      log_losses := Float.log total_loss.@[0] :: !log_losses;
      if step % 50000 = 0 then (
        Stdio.printf "Minus learning rate over batch for step %d: %f\n%!" step minus_lr.@[0];
        Stdio.printf "Loss over batch for step %d: %f\n%!" step total_loss.@[0];
        Stdio.printf "Step index at step %d: %f\n%!" step session_step.@[0]))
  done;
  close_session ();
  let%nn_op point = [ 0; 0 ] in
  let mlp_result = mlp point in
  refresh_session ();
  let callback (x, y) =
    set_values point [| x; y |];
    refresh_session ();
    Float.(mlp_result.@[0] >= 0.)
  in
  let plot_moons =
    let open PrintBox_utils in
    plot ~size:(120, 40) ~x_label:"ixes" ~y_label:"ygreks"
      [
        Scatterplot { points = Array.concat !points1; pixel = "#" };
        Scatterplot { points = Array.concat !points2; pixel = "%" };
        Boundary_map { pixel_false = "."; pixel_true = "*"; callback };
      ]
  in
  Stdio.printf "\nHalf-moons scatterplot and decision boundary:\n%!";
  PrintBox_text.output Stdio.stdout plot_moons;
  Stdio.printf "\nLoss curve:\n%!";
  let plot_loss =
    let open PrintBox_utils in
    plot ~size:(120, 30) ~x_label:"step" ~y_label:"loss"
      [ Line_plot { points = Array.of_list_rev !losses; pixel = "-" } ]
  in
  PrintBox_text.output Stdio.stdout plot_loss;
  Stdio.printf "\nLog-loss, for better visibility:\n%!";
  let plot_loss =
    let open PrintBox_utils in
    plot ~size:(120, 30) ~x_label:"step" ~y_label:"log loss"
      [ Line_plot { points = Array.of_list_rev !log_losses; pixel = "-" } ]
  in
  PrintBox_text.output Stdio.stdout plot_loss;
  Stdio.printf "\nLearning rate:\n%!";
  let plot_lr =
    let open PrintBox_utils in
    plot ~size:(120, 30) ~x_label:"step" ~y_label:"learning rate"
      [ Line_plot { points = Array.of_list_rev !learning_rates; pixel = "-" } ]
  in
  PrintBox_text.output Stdio.stdout plot_lr;
  Exec_as_gccjit.optimization_level := 3;
  Stdio.printf "\n%!"

let benchmarks =
  [
    (* ("non-virt. Interpreter", classify_moons ~virtualize:false Interpreter 3); *)
    (* ("non-virt. OCaml", classify_moons ~virtualize:false OCaml 3); *)
    ("non-virt. gccjit O0", classify_moons ~virtualize:false Gccjit ~opti_level:0);
    ("non-virt. gccjit O1", classify_moons ~virtualize:false Gccjit ~opti_level:1);
    ("non-virt. gccjit O2", classify_moons ~virtualize:false Gccjit ~opti_level:2);
    ("non-virt. gccjit O3", classify_moons ~virtualize:false Gccjit ~opti_level:3);
    (* ("virtualized Interpreter", classify_moons ~virtualize:true Interpreter 3); *)
    (* ("virtualized OCaml", classify_moons ~virtualize:true OCaml 3); *)
    ("virtualized gccjit O0", classify_moons ~virtualize:true Gccjit ~opti_level:0);
    ("virtualized gccjit O1", classify_moons ~virtualize:true Gccjit ~opti_level:1);
    ("virtualized gccjit O2", classify_moons ~virtualize:true Gccjit ~opti_level:2);
    ("virtualized gccjit O3", classify_moons ~virtualize:true Gccjit ~opti_level:3);
  ]

let () = classify_moons ~virtualize:true Gccjit ~opti_level:3 ()

let _suspended () =
  List.map benchmarks ~f:(fun (name, test) -> Bench.Test.create ~name test)
  |> Bench.make_command |> Command_unix.run

(* Example output, before monolithic update overhaul and before the virtual nodes optimization:

    ┌─────────────┬───────────┬────────────────┬──────────────┬──────────────┬────────────┐
    │ Name        │  Time/Run │        mWd/Run │     mjWd/Run │     Prom/Run │ Percentage │
    ├─────────────┼───────────┼────────────────┼──────────────┼──────────────┼────────────┤
    │ Interpreter │ 3_705.02s │ 1_316_342.55Mw │ 178_312.43kw │ 178_176.42kw │    100.00% │
    │ OCaml       │ 1_405.22s │   384_153.02Mw │  14_864.72kw │  14_658.17kw │     37.93% │
    │ gccjit      │    10.76s │       286.06Mw │     737.77kw │     703.35kw │      0.29% │
    └─────────────┴───────────┴────────────────┴──────────────┴──────────────┴────────────┘

   Run after the transition to monolithic step update code (single routine call per step):

    ┌────────┬───────────┬──────────────┬────────────┬────────────┬────────────┐
    │ Name   │  Time/Run │      mWd/Run │   mjWd/Run │   Prom/Run │ Percentage │
    ├────────┼───────────┼──────────────┼────────────┼────────────┼────────────┤
    │ OCaml  │ 1_464.26s │ 384_100.19Mw │ 2_024.75kw │ 1_882.76kw │    100.00% │
    │ gccjit │     9.38s │      44.96Mw │   780.57kw │   746.15kw │      0.64% │
    └────────┴───────────┴──────────────┴────────────┴────────────┴────────────┘
    ┌───────────┬──────────┬─────────┬──────────┬──────────┬────────────┐
    │ Name      │ Time/Run │ mWd/Run │ mjWd/Run │ Prom/Run │ Percentage │
    ├───────────┼──────────┼─────────┼──────────┼──────────┼────────────┤
    │ gccjit O0 │   47.07s │ 44.72Mw │ 757.03kw │ 722.61kw │    100.00% │
    │ gccjit O1 │   14.11s │ 44.72Mw │ 756.40kw │ 721.98kw │     29.97% │
    │ gccjit O2 │    9.65s │ 44.96Mw │ 780.57kw │ 746.15kw │     20.50% │
    │ gccjit O3 │    9.26s │ 44.96Mw │ 780.57kw │ 746.15kw │     19.67% │
    └───────────┴──────────┴─────────┴──────────┴──────────┴────────────┘

   Same, without shared subexpressions (the commented-out mlp):

    ┌───────────┬──────────┬─────────┬──────────┬──────────┬────────────┐
    │ Name      │ Time/Run │ mWd/Run │ mjWd/Run │ Prom/Run │ Percentage │
    ├───────────┼──────────┼─────────┼──────────┼──────────┼────────────┤
    │ gccjit O0 │   19.73s │ 44.30Mw │ 738.84kw │ 704.42kw │    100.00% │
    │ gccjit O1 │    6.16s │ 44.55Mw │ 761.39kw │ 726.97kw │     31.21% │
    │ gccjit O2 │    4.75s │ 44.55Mw │ 761.41kw │ 726.99kw │     24.07% │
    │ gccjit O3 │    4.65s │ 44.55Mw │ 761.41kw │ 726.99kw │     23.58% │
    └───────────┴──────────┴─────────┴──────────┴──────────┴────────────┘
*)
