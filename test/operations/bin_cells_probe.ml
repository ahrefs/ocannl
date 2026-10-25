(* A stand-in bench for the control of benchmarks/bin_cells.py (gh-ocannl-743): the two things the
   driver reads off a real bench, produced by the real code that produces them there. The
   configuration records are the library's own -- this links [arrayjit.utils] and reads settings
   through the same getters a bench does, so the driver's record reader is held to what [Utils]'s
   [log_config_json] stream actually prints rather than to a copy of its schema -- and the result
   lines are [Bench_out.result]'s, the printer every bin/ timing bench uses.

   Its one positional is the treatment: a time in seconds to report (as [probe], and half of it as
   [probe_half]), or a misbehaviour -- [fail] exits 1, [silent] reports nothing, [twice] reports one
   variant twice, [sleep] outlives any cap the control sets, [nonfinite] reports a time JSON cannot
   carry -- or [records], which prints the process's resolved-configuration records on stdout. No
   backend is linked: nothing is compiled. *)

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
  | "records" ->
      (* The function's records, on stdout, for the control to hold equal to the stream the same run
         printed on stderr under [log_config_json] (gh-ocannl-1566) -- after one read more, made
         here, past every read the other modes make. *)
      let _late : string = Utils.get_global_arg ~arg_name:"debug_highlight_pcre" ~default:"" in
      List.iter (Utils.resolved_config ()) ~f:(fun r ->
          Stdlib.print_endline (Utils.config_record_to_json r))
  | "sleep" -> Unix.sleepf 120.
  | "nonfinite" -> Bench_out.result ~variant:"probe" ~check:Bench_out.Passed ~seconds:Float.nan
  | s ->
      let seconds = Float.of_string s in
      Bench_out.result ~variant:"probe" ~check:Bench_out.Passed ~seconds;
      Bench_out.result ~variant:"probe_half" ~check:Bench_out.Passed ~seconds:(seconds /. 2.)
