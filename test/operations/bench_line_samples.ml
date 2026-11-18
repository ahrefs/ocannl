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
   exit. - [trace-raise]: the same search with one failure injected (gh-ocannl-2246): the first
   candidate after the baseline to start its timing call's validation raises
   [Device_memory_exhausted] from its first batch ([Autotune.on_batch_wall], inside the tagged call
   and past its validation) -- the candidate's classified decline, which the search survives (an
   expired wait would abandon the device, gh-ocannl-2133). The run prints the injected candidate's
   label and exception on stdout, [%S]-quoted as the trace quotes them, for the reader to find in
   the trace's [raised] line. *)

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

(* Installed after the trace, so the trace's own seams run first and count what this injects. *)
let inject_one_raise () =
  let current = ref "" and armed = ref false and injected = ref false in
  let prev_attempt = Autotune.Seam.get Autotune.on_candidate_attempt in
  Autotune.Seam.set Autotune.on_candidate_attempt (fun label ->
      prev_attempt label;
      current := label);
  let prev_preflight = Autotune.Seam.get Autotune.on_candidate_preflight in
  Autotune.Seam.set Autotune.on_candidate_preflight (fun name ->
      prev_preflight name;
      armed := (not !injected) && not (String.equal !current "baseline"));
  let prev_wall = Autotune.Seam.get Autotune.on_batch_wall in
  Autotune.Seam.set Autotune.on_batch_wall (fun ~depth ~ordinal wall ->
      let wall = prev_wall ~depth ~ordinal wall in
      if not !armed then wall
      else begin
        armed := false;
        injected := true;
        let exn =
          Ir.Schedule_outcome.Device_memory_exhausted
            {
              backend = "cc";
              requested = 4096;
              free = None;
              detail = Printf.sprintf "injected by bench_line_samples at depth %d" depth;
            }
        in
        Stdio.printf "injected: %S %S\n%!" !current (Stdlib.Printexc.to_string exn);
        raise exn
      end)

let trace ~inject =
  H.install_timing_trace ();
  if inject then inject_one_raise ();
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
  | _ :: "trace" :: _ -> trace ~inject:false
  | _ :: "trace-raise" :: _ -> trace ~inject:true
  | _ ->
      failwith
        "usage: bench_line_samples.exe harness|trace|trace-raise [--ocannl_<key>=<value> ...]"
