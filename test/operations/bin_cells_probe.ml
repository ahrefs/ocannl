(* A stand-in bench for the control of benchmarks/bin_cells.py (gh-ocannl-743): the two things the
   driver reads off a real bench, produced by the real code that produces them there. The
   configuration trace is the library's own -- this links [arrayjit.utils] and reads settings
   through the same getters a bench does, so the driver's trace parser is held to what [Utils]'s
   [log_config_sourcing] trace actually prints rather than to a copy of its wording -- and the
   result lines are [Bench_out.result]'s, the printer every bin/ timing bench uses.

   Its one positional is the treatment: a time in seconds to report (as [probe], and half of it as
   [probe_half]), or a misbehaviour -- [fail] exits 1, [silent] reports nothing, [twice] reports one
   variant twice, [sleep] outlives any cap the control sets, [nonfinite] reports a time JSON cannot
   carry, [refused] reports nothing and exits 3 as a bench whose every timing window was refused for
   host contention does (projection_shape_bench), [env] reports the seconds a stated
   [BENCH_PROBE_SECONDS] carries, and [record:<ms>] prints a runner's result line
   ([Bench_json.result_line], the printer bench_gpt uses) with that p50 step time. No backend is
   linked: nothing is compiled. *)

open Base

let () =
  let args = Bench_args.create "bin_cells_probe" in
  let mode = Bench_args.string args 0 ~default:"0.001" in
  (* What a bench consults: an unpinnable search gate, a mode setting, the backend. *)
  let _pruning : bool = Utils.registered_flag ~arg_name:"autotune_bound_pruning" in
  let _timing : string = Utils.get_global_arg ~arg_name:"autotune_timing" ~default:"queued" in
  let _backend : string = Utils.get_global_arg ~arg_name:"backend" ~default:"cc" in
  match mode with
  | "fail" -> Stdlib.exit 1
  | "silent" -> ()
  | "twice" ->
      Bench_out.result ~variant:"probe" ~check:Bench_out.Passed ~seconds:0.001;
      Bench_out.result ~variant:"probe" ~check:Bench_out.Passed ~seconds:0.002
  | "wrong" ->
      Bench_out.result ~variant:"probe" ~check:(Bench_out.Failed "differs from the oracle")
        ~seconds:0.001
  | "unchecked" ->
      Bench_out.result ~variant:"probe" ~check:(Bench_out.Unchecked "no oracle for this one")
        ~seconds:0.001
  | "sleep" -> Unix.sleepf 120.
  | "nonfinite" -> Bench_out.result ~variant:"probe" ~check:Bench_out.Passed ~seconds:Float.nan
  | "refused" -> Stdlib.exit 3
  | "env" ->
      let seconds = Float.of_string (Stdlib.Sys.getenv "BENCH_PROBE_SECONDS") in
      Bench_out.result ~variant:"probe" ~check:Bench_out.Passed ~seconds
  | s when String.is_prefix s ~prefix:"record:" ->
      let p50 = Float.of_string (String.chop_prefix_exn s ~prefix:"record:") in
      Stdio.print_endline
        (Bench_json.result_line ~backend:"cc" ~variant:"default" ~precision:"f32" ~profile:None
           ~regime_knobs:[] ~ambient_resolved:[] ~workload:"probe" ~compile_s:0.5 ~searched:false
           ~p10:(p50 /. 2.) ~p50 ~p90:(p50 *. 2.) ~queued_ms:p50 ~timed_steps:4 ~peak_memory:None
           ~losses:[| 2.5; 1.25 |] ())
  | s ->
      let seconds = Float.of_string s in
      Bench_out.result ~variant:"probe" ~check:Bench_out.Passed ~seconds;
      Bench_out.result ~variant:"probe_half" ~check:Bench_out.Passed ~seconds:(seconds /. 2.)
