(* gh-ocannl-1281: a GPU backend's device discovery and device opening run under a deadline, and a
   stuck one kills its process by SIGALRM with a diagnostic, rather than hanging it forever.

   On 2026-10-04 a faulted gfx1102 left tuf's HIP runtime wedged: every later process that touched
   it spun in device initialization and never returned (gh-ocannl-1211). The harness bounds that
   from outside; [Utils.with_device_init_deadline] bounds it inside every process, around the entry
   points [Backends.With_device_init_deadline] guards. A wedged driver cannot be produced on demand,
   so the stuck initialization is simulated by [Utils.device_init_test_stall], a test-only hook that
   sleeps INSIDE the guarded region and then raises before any driver contact -- so the stalled
   calls touch no device, whichever backend they name (the one real call, in the [again] case, is to
   metal: a missing backend's stub off macOS). Death, and the deadline's restraint, are observed
   from outside: each case is a fresh [device_init_deadline_child], with the deadline pinned on its
   commandline.

   The cases: every GPU backend's discovery, and each other guarded entry point, dies within the
   deadline by SIGALRM with the diagnostic; [device_init_deadline=0] lets the same stall run out; a
   program's own SIGALRM handler is left installed and in charge, and its own timer fires on time; a
   pending timer under the default disposition fires on time, unextended, without the diagnostic
   (the negative control: an implementation that armed over it would survive); disarming restores
   the default disposition and an idle timer; an entry point stays guarded after a call that
   returned; a SIGALRM blocked in the thread, or one already pending, does not hide the deadline; a
   SIGALRM from anything else meets the program's own disposition, and a timer the program starts
   during a region is left in place; the diagnostic's write blocked on a full fd 2 holds the death
   at most 2 s measured on the clock, even with every nap oversleeping; overlapping regions on two
   domains are serialized, so a healthy one's disarm does not leave a wedged one unbounded; a
   process exec'd under an ancestor that caught SIGALRM with SA_SIGINFO, which on macOS reads the
   default disposition with that flag still set (GitHub's macOS runner agent is such an ancestor),
   is armed; a region a fork child inherited takes back only its own deadline, not one a domain of
   the child armed meanwhile; and the cc backends are not guarded at all. Last, the diagnostic's tag
   is the one [tools/test-run.sh]'s digest reads. *)

open Base
open Stdio
open Verdict.Claims
module Backends = Context.Backends

let argv = Sys.get_argv ()
let child_exe = argv.(1)
let test_run_sh = argv.(2)
let tag = Utils.device_init_deadline_tag
let now () = Int64.to_float (Mtime_clock.now_ns ()) /. 1e9

(* A gate on the predicate the deadline is compiled under: no SIGALRM, no deadline (Windows). *)
let when_ = Utils.device_init_deadline_supported ()
let on = "a platform without SIGALRM"

type outcome = {
  status : Unix.process_status;
  ended : float;  (** The parent's monotonic reading as the child was reaped. *)
  stdout_lines : string list;
  stderr_lines : string list;
}

let run_child ~deadline args =
  let ((status, out, err) as child) =
    Fresh_process.run ~exe:child_exe (("--ocannl_device_init_deadline=" ^ deadline) :: args)
  in
  let ended = now () in
  eprintf "child %s (deadline %s) %s (not part of the golden):\n%s\n%!"
    (String.concat ~sep:" " args) deadline
    (Fresh_process.describe_status status)
    (Fresh_process.prefixed (out ^ err));
  ignore child;
  {
    status;
    ended;
    stdout_lines = String.split_lines out;
    stderr_lines = String.split_lines (String.filter err ~f:(fun c -> not (Char.equal c '\r')));
  }

(* [run_child] for a child that may hang for good (the negative control of the full-stderr case): it
   is reaped by polling and SIGKILLed past [cap] seconds. Its stderr is not read: the case replaces
   it. *)
let run_child_capped ~cap ~deadline args =
  let out = Stdlib.Filename.temp_file "ocannl-child-" ".out" in
  Exn.protect
    ~finally:(fun () -> Stdlib.Sys.remove out)
    ~f:(fun () ->
      let fd = Unix.openfile out [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
      let argv =
        Array.of_list (child_exe :: ("--ocannl_device_init_deadline=" ^ deadline) :: args)
      in
      let pid = Unix.create_process child_exe argv Unix.stdin fd Unix.stderr in
      Unix.close fd;
      let started = now () in
      let rec reap () =
        match Unix.waitpid [ Unix.WNOHANG ] pid with
        | 0, _ when Float.(now () -. started > cap) ->
            Unix.kill pid Stdlib.Sys.sigkill;
            (snd (Unix.waitpid [] pid), true)
        | 0, _ ->
            Unix.sleepf 0.05;
            reap ()
        | _, status -> (status, false)
        | exception Unix.Unix_error (Unix.EINTR, _, _) -> reap ()
      in
      let status, capped = reap () in
      let ended = now () in
      let text = In_channel.read_all out in
      eprintf "child %s (deadline %s) %s%s (not part of the golden):\n%s\n%!"
        (String.concat ~sep:" " args) deadline
        (Fresh_process.describe_status status)
        (if capped then ", SIGKILLed at the cap" else "")
        (Fresh_process.prefixed text);
      ({ status; ended; stdout_lines = String.split_lines text; stderr_lines = [] }, capped))

let field o key =
  List.find_map o.stdout_lines ~f:(fun l -> String.chop_prefix l ~prefix:(key ^ " "))

let float_field o key = Option.bind (field o key) ~f:Float.of_string_opt
let says o line = List.mem o.stdout_lines line ~equal:String.equal
let by_sigalrm o = match o.status with Unix.WSIGNALED s -> s = Stdlib.Sys.sigalrm | _ -> false
let exited_0 o = match o.status with Unix.WEXITED 0 -> true | _ -> false

(* The hook was reached: without this, a case the guard never ran would pass as one the deadline
   left alone. *)
let stalled o = Option.is_some (field o "stall-began")
let diagnostic o = List.find o.stderr_lines ~f:(fun l -> String.is_prefix l ~prefix:tag)
let since o key = Option.map (float_field o key) ~f:(fun t -> o.ended -. t)
let within lo hi = Option.value_map ~default:false ~f:(fun s -> Float.(lo <= s && s <= hi))

(* The deadline's own death: SIGALRM, within the 0.5 s deadline of the stall's start (and not before
   it), with the diagnostic naming the backend and the bound -- and the stall did not run out. *)
let died_by_deadline name o =
  by_sigalrm o
  && (not (says o "survived"))
  && within 0.4 10.0 (since o "stall-began")
  && Option.value_map (diagnostic o) ~default:false ~f:(fun l ->
      String.is_prefix l ~prefix:(tag ^ " " ^ name ^ " device initialization")
      && String.is_substring l ~substring:"within 0.5s"
      && String.is_substring l ~substring:"likely wedged"
      && String.is_substring l ~substring:"device_init_deadline")

(* Which backends are guarded is a design decision, not a derived set: an exhaustive match, so a new
   backend cannot be added without deciding. *)
let guarded = function
  | Backends.Cuda | Backends.Hip | Backends.Metal -> true
  | Backends.Cc | Backends.Multidev_cc -> false

let gpu_backends =
  List.filter Backends.all_of_backend ~f:guarded |> List.map ~f:Backends.backend_name

let cpu_backends =
  List.filter Backends.all_of_backend ~f:(Fn.non guarded) |> List.map ~f:Backends.backend_name

let () =
  let run ~deadline args = if when_ then Some (run_child ~deadline args) else None in
  let holds o ~f = Option.value_map o ~default:false ~f in
  let discovery =
    List.map gpu_backends ~f:(fun name ->
        (name, run ~deadline:"0.5" [ "stall"; name; "num_devices"; "30" ]))
  in
  gated_all ~when_ ~on ~aggregation:`Environment
    "every GPU backend's device discovery stalled past the deadline dies by SIGALRM within it, \
     with the diagnostic naming the backend and the bound"
    discovery ~f:(fun (name, o) -> holds o ~f:(died_by_deadline name));
  let entries =
    List.map [ "get_device"; "static_properties"; "hardware_limits" ] ~f:(fun entry ->
        (entry, run ~deadline:"0.5" [ "stall"; "cuda"; entry; "30" ]))
  in
  gated_all ~when_ ~on ~aggregation:`Environment
    "opening a device, and reading the properties or limits, are guarded the same way" entries
    ~f:(fun (_, o) -> holds o ~f:(died_by_deadline "cuda"));
  let tiny = run ~deadline:"0.000001" [ "stall"; "cuda"; "num_devices"; "30" ] in
  gated ~when_ ~on ~aggregation:`Environment
    "a deadline of one microsecond, which can expire before arming returns, still reads as the \
     deadline: SIGALRM with the diagnostic"
    (holds tiny ~f:(fun o ->
         by_sigalrm o && (not (says o "survived")) && Option.is_some (diagnostic o)));
  let disabled = run ~deadline:"0" [ "stall"; "cuda"; "num_devices"; "1.5" ] in
  gated ~when_ ~on ~aggregation:`Environment
    "with device_init_deadline=0 a stall three times the deadline above runs out and the process \
     exits normally"
    (holds disabled ~f:(fun o ->
         exited_0 o && stalled o && says o "survived" && Option.is_none (diagnostic o)));
  let handler_only = run ~deadline:"0.5" [ "own-handler"; "cuda"; "no-timer" ] in
  gated ~when_ ~on ~aggregation:`Environment
    "under the program's own SIGALRM handler no deadline is armed: the stall runs out, the handler \
     never runs and is still installed"
    (holds handler_only ~f:(fun o ->
         exited_0 o && stalled o && says o "survived" && says o "handler-fired 0"
         && says o "handler-intact true"
         && Option.is_none (diagnostic o)));
  let handler_timer = run ~deadline:"0.5" [ "own-handler"; "cuda"; "timer" ] in
  gated ~when_ ~on ~aggregation:`Environment
    "the program's own handler and its own 0.3 s timer: the handler runs once, on time, during the \
     stall, and is still installed after it"
    (holds handler_timer ~f:(fun o ->
         exited_0 o && stalled o && says o "survived" && says o "handler-fired 1"
         && within 0.25 1.2 (float_field o "handler-at")
         && says o "handler-intact true"
         && Option.is_none (diagnostic o)));
  let pending = run ~deadline:"60" [ "pending"; "cuda" ] in
  gated ~when_ ~on ~aggregation:`Environment
    "a pending 1 s timer under the default disposition fires on time during a 3 s stall, killing \
     the process by SIGALRM without the deadline's diagnostic"
    (holds pending ~f:(fun o ->
         by_sigalrm o && stalled o
         && (not (says o "survived"))
         && within 0.8 2.9 (since o "timer-set")
         && Option.is_none (diagnostic o)));
  let restored = run ~deadline:"0.5" [ "restored"; "cuda" ] in
  gated ~when_ ~on ~aggregation:`Environment
    "after a guarded call returns, SIGALRM has its default disposition again, the timer is idle, \
     and the process outlives the deadline"
    (holds restored ~f:(fun o ->
         exited_0 o && says o "hook-ran" && says o "disposition-default true"
         && says o "timer-idle true" && says o "survived"));
  let again = run ~deadline:"0.5" [ "again"; "metal" ] in
  gated ~when_ ~on ~aggregation:`Environment
    "a guarded entry point stays guarded after a call that returned: a later stall in it dies by \
     the deadline"
    (holds again ~f:(fun o -> says o "first-call returned" && died_by_deadline "metal" o));
  let blocked = run ~deadline:"0.5" [ "blocked"; "cuda" ] in
  gated ~when_ ~on ~aggregation:`Environment
    "with SIGALRM blocked in the process's only OCaml thread, the deadline still fires"
    (holds blocked ~f:(died_by_deadline "cuda"));
  let blocked_pending = run ~deadline:"0.5" [ "blocked-pending"; "cuda" ] in
  gated ~when_ ~on ~aggregation:`Environment
    "a blocked SIGALRM already pending does not hide the deadline: its diagnostic is written, and \
     the process dies by SIGALRM"
    (holds blocked_pending ~f:(died_by_deadline "cuda"));
  let externals =
    List.map [ "kill"; "raise" ] ~f:(fun how ->
        (how, run ~deadline:"60" [ "external"; "cuda"; how ]))
  in
  gated_all ~when_ ~on ~aggregation:`Environment
    "a SIGALRM sent or raised by anything else during a region meets the program's own \
     disposition: the default action kills, and nothing of the deadline's is written"
    externals ~f:(fun (_, o) ->
      holds o ~f:(fun o ->
          by_sigalrm o && stalled o
          && (not (says o "survived"))
          && not
               (List.exists o.stderr_lines
                  ~f:(String.is_substring ~substring:"device_init_deadline"))));
  let timer_in_region = run ~deadline:"60" [ "timer-in-region"; "cuda" ] in
  gated ~when_ ~on ~aggregation:`Environment
    "a real-time timer the program starts during a region is left in place: it fires on time after \
     the region, killing by SIGALRM without the deadline's diagnostic"
    (holds timer_in_region ~f:(fun o ->
         by_sigalrm o && stalled o
         && (not (says o "survived"))
         && within 0.8 2.9 (since o "timer-set")
         && Option.is_none (diagnostic o)));
  let full_stderr =
    if when_ then Some (run_child_capped ~cap:20.0 ~deadline:"0.5" [ "full-stderr"; "cuda" ])
    else None
  in
  gated ~when_ ~on ~aggregation:`Environment
    "with fd 2 a full pipe nobody reads, the deadline still kills the process by SIGALRM, its \
     backstop alarm firing while the diagnostic's write is blocked"
    (Option.value_map full_stderr ~default:false ~f:(fun (o, capped) ->
         (not capped) && by_sigalrm o
         && (not (says o "survived"))
         && within 1.5 10.0 (since o "stall-began")));
  let full_stderr_throttled =
    if when_ then
      Some (run_child_capped ~cap:20.0 ~deadline:"0.5" [ "full-stderr"; "cuda"; "throttled" ])
    else None
  in
  gated ~when_ ~on ~aggregation:`Environment
    "the backstop's 2 s are measured, not counted in naps: with the process's timers throttled so \
     every nap oversleeps by far (as on a loaded macOS VM), the full-pipe case still dies by \
     SIGALRM within 10 s of the stall's start"
    (Option.value_map full_stderr_throttled ~default:false ~f:(fun (o, capped) ->
         says o "throttled true" && (not capped) && by_sigalrm o
         && (not (says o "survived"))
         && within 1.5 10.0 (since o "stall-began")));
  let forked = run ~deadline:"0.5" [ "fork-in-region"; "cuda" ] in
  gated ~when_ ~on ~aggregation:`Environment
    "a child forked from inside an armed region gets a deadline of its own: its stall dies by \
     SIGALRM, and the parent's region returns"
    (holds forked ~f:(fun o -> says o "fork-child sigalrm" && exited_0 o && says o "survived"));
  let fork_disarm = run ~deadline:"0.5" [ "fork-disarm"; "hip" ] in
  gated ~when_ ~on ~aggregation:`Environment
    "a region a fork child inherited, returning after a new domain of the child armed its own, \
     leaves that deadline armed: the domain's wedged stall still kills the child by SIGALRM with \
     the diagnostic (gh-ocannl-1446)"
    (holds fork_disarm ~f:(fun o ->
         says o "fork-child sigalrm" && exited_0 o && says o "survived"
         && Option.exists (diagnostic o) ~f:(String.is_prefix ~prefix:(tag ^ " hip device"))));
  let concurrent = run ~deadline:"0.5" [ "concurrent"; "cuda"; "hip" ] in
  gated ~when_ ~on ~aggregation:`Environment
    "guarded regions are serialized: a wedged one entered while a healthy one held the timer is \
     still killed by its own deadline after the healthy one returns"
    (holds concurrent ~f:(died_by_deadline "hip"));
  let inherited = run ~deadline:"0.5" [ "inherited-siginfo"; "cuda" ] in
  gated ~when_ ~on ~aggregation:`Environment
    "a SIGALRM an ancestor caught with SA_SIGINFO reaches an exec'd process as its default \
     disposition, and the deadline is armed there (on macOS the flag survives the exec)"
    (holds inherited ~f:(died_by_deadline "cuda"));
  let cpu =
    List.map cpu_backends ~f:(fun name -> (name, run ~deadline:"0.5" [ "unguarded"; name ]))
  in
  gated_all ~when_ ~on ~aggregation:`Environment
    "the cc backends' entry points are not guarded: the stall hook never runs" cpu ~f:(fun (_, o) ->
      holds o ~f:(fun o -> exited_0 o && says o "returned" && not (says o "hook-ran")));
  (* Pin the relationship: the digest recognizes a wedged device by this tag, spelled in its own
     shell variable. *)
  p "tools/test-run.sh's digest reads the diagnostic's tag as it is written"
    (String.is_substring (In_channel.read_all test_run_sh)
       ~substring:("device_init_deadline_tag='" ^ tag ^ "'"))
