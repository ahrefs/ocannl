open Base
open Ocannl
module Tn = Arrayjit.Tnode
module IDX = Train.IDX
module CDSL = Train.CDSL
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module Rand = Arrayjit.Rand.Lib

module type Backend = Arrayjit.Backend_intf.Backend

let fresh_backend = Arrayjit.Backends.fresh_backend

let benchmark_overhead backend () =
  let n_data = 20 in
  let module Backend = (val backend : Backend) in
  (* Utils.settings.with_debug <- true; *)
  (* Utils.settings.output_debug_files_in_build_directory <- true; *)
  (* Utils.settings.debug_log_from_routines <- true; *)
  CDSL.disable_all_debugs ();
  Stdio.prerr_endline @@ "\n\n****** Benchmarking " ^ Backend.name ^ " ******";
  Rand.init 0;
  let init_time = Time_now.nanoseconds_since_unix_epoch () in
  let%op f = (3 *. ("x" [ 5 ] **. 2)) - (4 *. x) + 5 in
  Train.set_hosted f.value;

  (* Train.every_non_literal_on_host f; *)
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  let init_mem = Backend.(get_used_memory stream.device) in
  let update_f = Train.grad_update f in
  (* Initialize the context with a mock update of x to ensure that it is not optimized as a
     constant. *)
  let%cd mock_update_x = x =: 42 in
  let init_assign_x =
    Train.to_routine (module Backend) ctx ~name:"init_assign_x" IDX.empty mock_update_x
  in
  let f_routine =
    Train.to_routine (module Backend) init_assign_x.context IDX.empty update_f.fwd_bprop
  in
  Tensor.print_tree ~with_grad:true ~with_backend_info:true ~depth:9 f;
  Tensor.iter_embedded f ~f:(fun a -> ignore (Backend.from_host f_routine.context a : bool));

  let xs = Array.init n_data ~f:Float.(fun i -> of_int i - (of_int n_data /. 2.)) in
  let open Operation.At in
  (* Note: this compiles entirely fresh code for each step of the loop. *)
  let ys =
    Array.map xs ~f:(fun v ->
        let%cd update_x = x =: !.v in
        let assign_x =
          Train.to_routine (module Backend) f_routine.context ~name:"assign_x" IDX.empty update_x
        in
        Train.run assign_x;
        Train.run f_routine;
        f.@[0])
  in
  let plot_box =
    let open PrintBox_utils in
    plot ~size:(40, 25) ~x_label:"x" ~y_label:"f(x)"
      [ Scatterplot { points = Array.zip_exn xs ys; pixel = "#" } ]
  in
  let final_time = Time_now.nanoseconds_since_unix_epoch () in
  let time_in_sec = Int63.(to_float @@ (final_time - init_time)) /. 1000_000_000. in
  let mem_in_bytes = Backend.(get_used_memory stream.device) - init_mem in
  let result =
    PrintBox_utils.Benchmark
      {
        bench_title = Backend.name ^ " overhead";
        time_in_sec;
        mem_in_bytes;
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
    benchmark_overhead (fresh_backend ~backend_name:"cc" ());
    benchmark_overhead (fresh_backend ~backend_name:"cuda" ());
  ]

let () =
  List.map benchmarks ~f:(fun bench -> bench ())
  |> PrintBox_utils.table |> PrintBox_text.output Stdio.stdout
