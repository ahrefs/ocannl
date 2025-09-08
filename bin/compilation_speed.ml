open Base
open Ocannl
module Tn = Ir.Tnode
module IDX = Train.IDX
module CDSL = Train.CDSL
open Operation.DSL_modules

(* FIXME: expose backend by name from Context *)

let benchmark_overhead _backend_name () =
  let n_data = 20 in
  let ctx = Context.auto () in
  CDSL.disable_all_debugs ();
  Stdio.prerr_endline @@ "\n\n****** Benchmarking " ^ Context.backend_name ctx ^ " ******";
  let init_time = Time_now.nanoseconds_since_unix_epoch () in
  let%op f = (3 *. ({ x; o = [ 5 ] } **. 2)) - (4 *. x) + 5 in
  Train.set_hosted f.value;

  let update_f = Train.grad_update f in
  let ctx = Train.init_params ctx IDX.empty f in
  let f_routine = Train.to_routine ctx IDX.empty update_f in
  Train.printf_tree ~with_grad:true ~depth:9 f;

  let xs = Array.init n_data ~f:Float.(fun i -> of_int i - (of_int n_data /. 2.)) in
  let open Operation.At in
  (* Note: this compiles entirely fresh code for each step of the loop. *)
  let ys =
    Array.map xs ~f:(fun v ->
        let%cd update_x = ~~("update_x"; x =: !.v )in
        let assign_x =
          Train.to_routine (Context.context f_routine) IDX.empty update_x
        in
        Train.run ctx assign_x;
        Train.run ctx f_routine;
        f.@[0])
  in
  let plot_box =
    PrintBox_utils.plot ~small:true ~x_label:"x" ~y_label:"f(x)"
      [ Scatterplot { points = Array.zip_exn xs ys; content = PrintBox.line "#" } ]
  in
  let final_time = Time_now.nanoseconds_since_unix_epoch () in
  let time_in_sec = Int63.(to_float @@ (final_time - init_time)) /. 1000_000_000. in
  let result =
    PrintBox_utils.Benchmark
      {
        bench_title = Context.backend_name ctx ^ " overhead";
        time_in_sec;
        mem_in_bytes = 0;
        result_label = "x, f(x)";
        result =
          [%sexp_of: (float * float) list]
          @@ [ (xs.(0), ys.(0)); (xs.(n_data / 2), ys.(n_data / 2)) ];
      }
  in
  PrintBox_text.output Stdio.stdout plot_box;
  Stdio.print_endline "\n";
  result

let benchmarks =
  [
    (* benchmark_overhead (fresh_backend "gccjit" ()); *)
    benchmark_overhead "backend";
    (* benchmark_overhead (fresh_backend ~backend_name:"cuda" ()); *)
  ]

let () =
  List.map benchmarks ~f:(fun bench -> bench ())
  |> PrintBox_utils.table |> PrintBox_text.output Stdio.stdout
