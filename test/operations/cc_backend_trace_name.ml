open Base
open Ocannl
open Ocannl.Operation.DSL_modules

let claim = "the cc work trace reports the compiled routine name"
let trace_gate = "OCANNL_LOG_LEVEL_CC_BACKEND"
let routine_name = "gh859_cc_trace_name"
let trace_file = "cc_backend_trace_name.log"

let compiled_trace_level () =
  Option.bind (Stdlib.Sys.getenv_opt trace_gate) ~f:Int.of_string_opt |> Option.value ~default:0

(* A flushing trace writes a bare result as the first line inside [work]. Keep the assertion on that
   scope: [link_compiled]'s argument dump also contains [code.name], and accepting an occurrence
   anywhere in the file would let the wrong-binding regression pass. *)
let work_results lines =
  let rec collect acc = function
    | begin_line :: result :: rest
      when String.is_substring (String.strip begin_line) ~substring:" work begin " ->
        collect (String.strip result :: acc) rest
    | _ :: rest -> collect acc rest
    | [] -> List.rev acc
  in
  collect [] lines

let run_trace_smoke () =
  Tensor.unsafe_reinitialize ();
  let x = TDSL.ndarray [| 1.0 |] ~label:[ "trace_input" ] ~output_dims:[ 1 ] () in
  let%op y = x + 1.0 in
  let ctx, routine =
    Context.compile ~name:routine_name (Context.cpu ()) (Train.forward y) Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  Context.sync ctx;
  let results = Stdio.In_channel.read_lines (Utils.diagn_log_file trace_file) |> work_results in
  Verdict.p_exists claim results ~f:(String.equal routine_name)

let () =
  if compiled_trace_level () >= 3 then run_trace_smoke ()
  else
    (* The Ubuntu compiler-trace CI job executes this exact claim at level 3. The ordinary fleet
       sweep cannot infer that configuration-only execution from its default-config box logs, so
       keep the skip visible without presenting it as missing sweep coverage, naming the step that
       owns it (sweep_owner_scan holds the step to building this test's alias). *)
    Verdict.skipped
      ~aggregation:
        (`Outside_sweep
           (Verdict.Ci_step
              {
                workflow = "ci.yml";
                step = "Check non-default module log level and trace semantics";
                alias = "test/operations/runtest-cc_backend_trace_name";
              }))
      ~backend:(trace_gate ^ "<3") claim
