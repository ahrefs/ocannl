open Base
module Lazy = Utils.Lazy

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

type t =
  | Task : { context_lifetime : ('a[@sexp.opaque]); description : string; work : unit -> unit } -> t
[@@deriving sexp_of]

let describe (Task task) = task.description

let run (Task task) : unit =
  (* [%log_result "run", task.description]; *)
  task.work ()

let prepend ~work (Task task) =
  Task
    {
      task with
      work =
        (fun () ->
          work ();
          task.work ());
    }

let append ~work (Task task) =
  Task
    {
      task with
      work =
        (fun () ->
          task.work ();
          work ());
    }

let enschedule ~schedule_task ~get_stream_name stream (Task { description; _ } as task) =
  (* [%log_result "enschedule", description, "on", get_stream_name stream]; *)
  let work () = schedule_task stream task in
  Task
    {
      context_lifetime = ();
      description = "schedules {" ^ description ^ "} on " ^ get_stream_name stream;
      work;
    }
