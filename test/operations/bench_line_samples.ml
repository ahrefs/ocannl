(* gh-ocannl-1767: lines the benchmark harness's own emitters print, for [bench_line_readers.py] to
   run benchmarks/bench_output.py's readers over. Nothing here formats a line: each mode drives the
   code that prints it, so a change to an emitter reaches the readers' check the day it lands.

   - [harness]: the dominant-kernel instrument ([Bench_harness.dominant_kernel]) over four small
   compiled routines, with its per-kernel table on ([BENCH_KERNEL_TABLE=1], from the rule) on stderr
   -- one kernel through the instrument's own [time_kernel] seam as declined, one as refused, so the
   table holds every kind of row -- and the launch-geometry trace of their compiles
   ([schedule_log_launches], from the command line) beside it. A CPU backend ships a routine as one
   kernel, hence four routines. - [trace]: the timing trace ([BENCH_TIMING_TRACE=1], from the rule)
   of a seed-only search of a small matmul, then the end-of-search marker; the summary prints at
   exit. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module H = Bench_harness

let harness () =
  let ctx = ref (Context.auto ()) in
  let routines =
    List.init 4 ~f:(fun k ->
        let a =
          NTDSL.range_of_shape ~label:[ Printf.sprintf "bls_a%d" k ] ~output_dims:[ 8; 8 ] ()
        in
        let%op c = a +* "ij; jk => ik" a in
        Train.set_materialized c.Tensor.value;
        let ctx', routine =
          Context.compile ~name:(Printf.sprintf "bls_k%d" k) !ctx (Train.forward c)
            Ir.Indexing.Empty
        in
        Train.run ctx' routine;
        ctx := ctx';
        routine)
  in
  let time_kernel ~i timing =
    match i with
    | 0 -> Error (H.Declined "injected by bench_line_samples:\na detail over two lines")
    | 1 ->
        Error
          (H.Untimed
             {
               Bench_retime.ms = Float.nan;
               lo_ms = Float.nan;
               hi_ms = Float.nan;
               admitted = 0;
               refused = 4;
             })
    | _ -> timing ()
  in
  (* The record is the result line's [dominant_kernel] object; the rows are what is read here. *)
  ignore (H.dominant_kernel ~calls:4 ~time_kernel ~ctx:!ctx ~bindings:Train.IDX.empty routines)

let trace () =
  H.install_timing_trace ();
  let a = NTDSL.range_of_shape ~label:[ "bls_a" ] ~output_dims:[ 16; 16 ] () in
  let b = NTDSL.range_of_shape ~label:[ "bls_b" ] ~output_dims:[ 16; 16 ] () in
  let%op c = a +* "ik; kj => ij" b in
  Train.set_materialized c.Tensor.value;
  let ctx = Context.auto () in
  let t0 = Unix.gettimeofday () in
  let _ctx, _routine =
    Autotune.tune ~search:true ~rounds:0 ~cache_dir:"" ~name:"bls_matmul" ctx (Train.forward c)
      Ir.Indexing.Empty
  in
  H.trace_search_done ~compile_s:(Unix.gettimeofday () -. t0)

let () =
  (* The mode comes first; the [--ocannl_*] settings after it are the configuration's. *)
  match Array.to_list (Sys.get_argv ()) with
  | _ :: "harness" :: _ -> harness ()
  | _ :: "trace" :: _ -> trace ()
  | _ -> failwith "usage: bench_line_samples.exe harness|trace [--ocannl_<key>=<value> ...]"
