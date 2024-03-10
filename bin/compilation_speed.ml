open Base
open Ocannl
module Tn = Arrayjit.Tnode
module IDX = Arrayjit.Indexing.IDX
module CDSL = Arrayjit.Low_level.CDSL
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL

let benchmark_overhead backend () =
  let n_data = 20 in
  Arrayjit.Backends.reinitialize backend;
  let open (val backend : Arrayjit.Backends.Backend) in
  (* Utils.settings.with_debug <- true; *)
  (* Utils.settings.output_debug_files_in_run_directory <- true; *)
  (* Utils.settings.debug_log_jitted <- true; *)
  CDSL.disable_all_debugs ();
  Stdio.prerr_endline @@ "\n\n****** Benchmarking " ^ name ^ " ******";
  Random.init 0;
  let init_time = Time_now.nanoseconds_since_unix_epoch () in
  let%op f = (3 *. ("x" [ 5 ] **. 2)) - (4 *. x) + 5 in
  Train.set_hosted f.value;
  (* Train.every_non_literal_on_host f; *)

  let device = get_device ~ordinal:0 in
  let ctx = init device in
  let update_f = Train.grad_update f in
  (* Initialize the context with a mock update of x to ensure that it is not optimized as a constant. *)
  let%cd mock_update_x = x =: 42 in
  let init_jitted_x = jit ~name:"init_assign_x" ctx IDX.empty mock_update_x in
  let jitted_f = jit init_jitted_x.context IDX.empty update_f.fwd_bprop in
  Tensor.print_tree ~with_grad:true ~with_backend_info:true ~depth:9 f;
  Tensor.iter_embedded_arrays f ~f:(fun a ->
      if from_host jitted_f.context a then Stdio.printf "Sent array %s.\n%!" @@ Tn.name a);

  let xs = Array.init n_data ~f:Float.(fun i -> of_int i - (of_int n_data /. 2.)) in
  let open Tensor.O in
  let ys =
    Array.map xs ~f:(fun v ->
        let%cd update_x = x =: !.v in
        let jitted_x = jit ~name:"assign_x" jitted_f.context IDX.empty update_x in
        Train.run jitted_x;
        await device;
        Train.run jitted_f;
        await device;
        ignore (to_host jitted_f.context f.value : bool);
        f.@[0])
  in
  let plot_box =
    let open PrintBox_utils in
    plot ~size:(40, 25) ~x_label:"x" ~y_label:"f(x)"
      [ Scatterplot { points = Array.zip_exn xs ys; pixel = "#" } ]
  in
  let final_time = Time_now.nanoseconds_since_unix_epoch () in
  let time_in_sec = Int63.(to_float @@ (final_time - init_time)) /. 1000_000_000. in
  let result =
    PrintBox_utils.Benchmark
      {
        bench_title = name ^ " overhead";
        time_in_sec;
        (* FIXME: global mem consumption *)
        mem_in_bytes = 0;
        result_label = "x, f(x)";
        result = [%sexp_of: (float * float) list] @@ [ (xs.(0), ys.(0)); (xs.(n_data / 2), ys.(n_data / 2)) ];
      }
  in
  PrintBox_text.output Stdio.stdout plot_box;
  Stdio.print_endline "\n";
  result

let benchmarks =
  [
    benchmark_overhead (module Arrayjit.Backends.Gccjit_backend);
    benchmark_overhead (module Arrayjit.Backends.Cuda_backend);
  ]

let () =
  List.map benchmarks ~f:(fun bench -> bench ()) |> PrintBox_utils.table |> PrintBox_text.output Stdio.stdout
