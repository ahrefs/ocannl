(* A stand-in bench for the control of benchmarks/bin_cells.py (gh-ocannl-743): the two things the
   driver reads off a real bench, produced by the real code that produces them there. The
   configuration trace is the library's own -- this links [arrayjit.utils] and reads settings
   through the same getters a bench does, so the driver's trace parser is held to what [Utils]'s
   [log_config_sourcing] trace actually prints rather than to a copy of its wording -- and the
   result lines are [Bench_out.result]'s, the printer every bin/ timing bench uses.

   Its one positional is the treatment: a time in seconds to report (as [probe], and half of it as
   [probe_half]), or a misbehaviour -- [fail] exits 1, [sleep] outlives any cap the control sets,
   [nonfinite] reports a time JSON cannot carry. No backend is linked: nothing is compiled. *)

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
  | "sleep" -> Unix.sleepf 120.
  | "nonfinite" -> Bench_out.result ~variant:"probe" ~seconds:Float.nan
  | s ->
      let seconds = Float.of_string s in
      Bench_out.result ~variant:"probe" ~seconds;
      Bench_out.result ~variant:"probe_half" ~seconds:(seconds /. 2.)
