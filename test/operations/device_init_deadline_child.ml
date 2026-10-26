(* gh-ocannl-1281: the child process of [device_init_deadline], which reads what it prints and how
   it ended. The parent pins [device_init_deadline] on this child's COMMANDLINE, which no
   environment variable or configuration file outranks. The modes, after the configuration flags,
   follow.

   [stall BACKEND ENTRY SECONDS]: the test stall hook sleeps SECONDS inside BACKEND's guarded ENTRY
   (num_devices, get_device, static_properties or hardware_limits), then raises before any driver
   contact; prints [survived] if the stall ran out. Every stall prints [stall-began <t>] first, so a
   case that the hook never reached cannot pass as one the deadline left alone.

   [own-handler BACKEND (timer|no-timer)]: an OCaml SIGALRM handler is installed first, and with
   [timer] the real-time timer is started 0.3 s ahead; then a 1.5 s stall. Prints how often and when
   the handler ran, and whether it is still the installed disposition.

   [pending BACKEND]: SIGALRM keeps its default disposition and the real-time timer is started 1 s
   ahead (printing [timer-set <t>]); then a 3 s stall.

   [restored BACKEND]: one guarded call whose hook prints [hook-ran] and returns at once; then
   prints whether SIGALRM's disposition is the default again and the timer idle, sleeps 1 s, and
   prints [survived].

   [unguarded BACKEND]: one guarded call whose hook would print [hook-ran]; prints [returned].

   [inherited-siginfo BACKEND]: catches SIGALRM with an [SA_SIGINFO] handler, then [exec]s this
   executable, with the same configuration flags, as [exec-stall BACKEND]: as under GitHub's macOS
   runner agent, whose caught SIGALRM reaches every descendant as [SIG_DFL] with [SA_SIGINFO] still
   set on macOS.

   [exec-stall BACKEND]: prints [siginfo-inherited <whether SIGALRM reads as SIG_DFL with
   SA_SIGINFO>], then a 30 s stall.

   [again BACKEND]: one real [num_devices] call (on a missing backend's stub, or on a device that
   answers), then a 30 s stall in the same entry point.

   [blocked BACKEND]: SIGALRM is blocked in the (only) thread first; then a 5 s stall.

   [blocked-pending BACKEND]: SIGALRM is blocked and then sent to the process, so it is pending;
   then a 30 s stall.

   [timer-in-region BACKEND]: the program starts its own 1 s real-time timer from inside a region
   (printing [timer-set <t>]) that returns 0.2 s later; then it waits 3 s.

   [external BACKEND (kill|raise)]: another domain sends the process a SIGALRM 0.3 s into a 5 s
   stall.

   [full-stderr BACKEND]: fd 2 is replaced by a full pipe nobody reads; then a 30 s stall.
   [full-stderr BACKEND throttled] first throttles the process's timers (every nap oversleeps by
   far, as on a loaded macOS VM), printing [throttled <whether it applied>].

   [fork-in-region BACKEND]: forks from inside an armed region; the child stalls 5 s in a region of
   its own, and the parent prints [fork-child <how it ended>].

   [fork-disarm BACKEND]: forks from inside an armed region. In the child, a new domain enters a
   region of its own, under BACKEND, whose stall is 3 s; once that region is armed, the child's main
   thread leaves the region it inherited, and so runs that region's disarm. The parent prints
   [fork-child <how it ended>].

   [concurrent FIRST SECOND]: a second domain enters FIRST's guarded [num_devices], whose stall is
   healthy (0.3 s); once that stall has begun, this thread enters SECOND's, whose stall is 5 s.

   Times are seconds on the monotonic clock, which is system-wide, so the parent can subtract its
   own reading from them. *)

open Base
open Stdio
module Backends = Context.Backends

exception Stall_done

external raise_sigalrm : unit -> unit = "ocannl_test_raise_sigalrm"
external catch_sigalrm_siginfo : unit -> unit = "ocannl_test_catch_sigalrm_siginfo"
external sigalrm_default_with_siginfo : unit -> bool = "ocannl_test_sigalrm_default_with_siginfo"
external throttle_timers : unit -> bool = "ocannl_test_throttle_timers"

let now () = Int64.to_float (Mtime_clock.now_ns ()) /. 1e9
let say fmt = Printf.ksprintf (fun line -> printf "%s\n%!" line) fmt

let backend_module name : (module Ir.Backend_intf.Backend) =
  match
    List.find Backends.all_of_backend ~f:(fun b -> String.equal (Backends.backend_name b) name)
  with
  | Some Backends.Cc -> (module Backends.Cc_b)
  | Some Backends.Multidev_cc -> (module Backends.Multidev_cc_b)
  | Some Backends.Cuda -> (module Backends.Cuda_b)
  | Some Backends.Hip -> (module Backends.Hip_b)
  | Some Backends.Metal -> (module Backends.Metal_b)
  | None -> failwith ("device_init_deadline_child: unknown backend " ^ name)

let call_entry name entry =
  let (module B) = backend_module name in
  match entry with
  | "num_devices" -> ignore (B.num_devices () : int)
  | "get_device" -> ignore (B.get_device ~ordinal:0 : B.device)
  | "static_properties" -> ignore (B.static_properties () : Sexp.t)
  | "hardware_limits" -> ignore (B.hardware_limits () : Ir.Backend_intf.hardware_limits)
  | _ -> failwith ("device_init_deadline_child: unknown entry " ^ entry)

(* Runs [entry] with the stall hook set to [stall]; a [Stall_done] it raises ends the call. *)
let guarded_call ?(entry = "num_devices") name stall =
  Utils.device_init_test_stall := Some stall;
  (try call_entry name entry with Stall_done -> ());
  Utils.device_init_test_stall := None

let stall_for seconds () =
  say "stall-began %.6f" (now ());
  Unix.sleepf seconds;
  raise Stall_done

(* Waits for [pid] under a 15 s cap: a child stuck on state the fork copied (an OCaml mutex its
   owner thread took along, gh-ocannl-1281 review round 8) hangs for good, and is reported. *)
let reap_child pid =
  let started = now () in
  let rec reap () =
    match Unix.waitpid [ Unix.WNOHANG ] pid with
    | 0, _ when Float.(now () -. started > 15.0) ->
        Unix.kill pid Stdlib.Sys.sigkill;
        ignore (Unix.waitpid [] pid : int * Unix.process_status);
        "hung"
    | 0, _ ->
        Unix.sleepf 0.05;
        reap ()
    | _, Unix.WSIGNALED s when s = Stdlib.Sys.sigalrm -> "sigalrm"
    | _, Unix.WEXITED n -> "exited-" ^ Int.to_string n
    | _ -> "other"
  in
  reap ()

let start_timer seconds =
  ignore
    (Unix.setitimer Unix.ITIMER_REAL { Unix.it_interval = 0.; it_value = seconds }
      : Unix.interval_timer_status)

let () =
  let argv = Sys.get_argv () in
  let flags, args =
    Array.to_list argv |> List.tl_exn |> List.partition_tf ~f:(String.is_prefix ~prefix:"--")
  in
  match args with
  | [ "stall"; name; entry; seconds ] ->
      guarded_call ~entry name (stall_for (Float.of_string seconds));
      say "survived"
  | [ "own-handler"; name; timer ] ->
      let fired = ref [] in
      let on_alarm _ = fired := now () :: !fired in
      Stdlib.Sys.set_signal Stdlib.Sys.sigalrm (Stdlib.Sys.Signal_handle on_alarm);
      let t0 = now () in
      if String.equal timer "timer" then start_timer 0.3;
      guarded_call name (stall_for 1.5);
      let intact =
        let current = Stdlib.Sys.signal Stdlib.Sys.sigalrm Stdlib.Sys.Signal_default in
        Stdlib.Sys.set_signal Stdlib.Sys.sigalrm current;
        (* [Sys.signal] answers a fresh [Signal_handle] block: the closure is what is compared. *)
        match current with
        | Stdlib.Sys.Signal_handle f -> phys_equal f on_alarm
        | _ -> false
      in
      say "handler-fired %d" (List.length !fired);
      List.iter (List.rev !fired) ~f:(fun t -> say "handler-at %.6f" (t -. t0));
      say "handler-intact %b" intact;
      say "survived"
  | [ "pending"; name ] ->
      say "timer-set %.6f" (now ());
      start_timer 1.0;
      guarded_call name (stall_for 3.0);
      say "survived"
  | [ "restored"; name ] ->
      guarded_call name (fun () ->
          say "hook-ran";
          raise Stall_done);
      let default =
        match Stdlib.Sys.signal Stdlib.Sys.sigalrm Stdlib.Sys.Signal_default with
        | Stdlib.Sys.Signal_default -> true
        | _ -> false
      in
      let idle = Float.((Unix.getitimer Unix.ITIMER_REAL).Unix.it_value = 0.) in
      say "disposition-default %b" default;
      say "timer-idle %b" idle;
      Unix.sleepf 1.0;
      say "survived"
  | [ "again"; name ] ->
      (match call_entry name "num_devices" with
      | () -> say "first-call returned"
      | exception exn -> say "first-call raised %s" (Exn.to_string exn));
      guarded_call name (stall_for 30.0);
      say "survived"
  | [ "blocked"; name ] ->
      ignore (Unix.sigprocmask Unix.SIG_BLOCK [ Stdlib.Sys.sigalrm ] : int list);
      guarded_call name (stall_for 5.0);
      say "survived"
  | [ "blocked-pending"; name ] ->
      ignore (Unix.sigprocmask Unix.SIG_BLOCK [ Stdlib.Sys.sigalrm ] : int list);
      Unix.kill (Unix.getpid ()) Stdlib.Sys.sigalrm;
      guarded_call name (stall_for 30.0);
      say "survived"
  | [ "timer-in-region"; name ] ->
      (* The program starts its own 1 s timer from inside the region, 0.2 s before the region
         returns; then it waits 3 s outside. *)
      guarded_call name (fun () ->
          say "stall-began %.6f" (now ());
          say "timer-set %.6f" (now ());
          start_timer 1.0;
          Unix.sleepf 0.2;
          raise Stall_done);
      Unix.sleepf 3.0;
      say "survived"
  | [ "external"; name; how ] ->
      (* Another domain sends the process a SIGALRM 0.3 s into a 5 s stall under a deadline far
         away: [kill] (si_code SI_USER) or [raise] in that domain's thread (SI_TKILL on Linux). *)
      let sender =
        Stdlib.Domain.spawn (fun () ->
            Unix.sleepf 0.3;
            match how with
            | "kill" -> Unix.kill (Unix.getpid ()) Stdlib.Sys.sigalrm
            | _ -> raise_sigalrm ())
      in
      guarded_call name (stall_for 5.0);
      Stdlib.Domain.join sender;
      say "survived"
  | "full-stderr" :: name :: (([] | [ "throttled" ]) as throttled) ->
      if not (List.is_empty throttled) then say "throttled %b" (throttle_timers ());
      (* fd 2 becomes a pipe that is full and that nobody reads, so a blocking write to it never
         returns. *)
      let r, w = Unix.pipe () in
      Unix.set_nonblock w;
      let fill chunk =
        let b = Bytes.make chunk 'x' in
        try
          while true do
            ignore (Unix.write w b 0 chunk : int)
          done
        with Unix.Unix_error ((Unix.EAGAIN | Unix.EWOULDBLOCK), _, _) -> ()
      in
      fill 65536;
      fill 1;
      Unix.clear_nonblock w;
      Unix.dup2 w Unix.stderr;
      guarded_call name (stall_for 30.0);
      say "survived";
      Unix.close r
  | [ "fork-in-region"; name ] ->
      (* Forks from inside an armed region, which the parent then leaves at once. The child, whose
         watchdog thread did not come along, runs a 5 s stall in a region of its own; the parent
         reports, after its region, how the child ended. *)
      let child = ref None in
      guarded_call name (fun () ->
          match Unix.fork () with
          | 0 ->
              Utils.device_init_test_stall := Some (stall_for 5.0);
              (try call_entry name "num_devices" with Stall_done -> ());
              Stdlib.exit 0
          | pid ->
              child := Some pid;
              raise Stall_done);
      Option.iter !child ~f:(fun pid -> say "fork-child %s" (reap_child pid));
      say "survived"
  | [ "fork-disarm"; name ] ->
      (* The parent's region is armed when it forks; the child's copy of it is still on the child's
         stack. A domain the child spawns arms a region of its own, wedged for 3 s; only then does
         the inherited region return, running its disarm. That disarm must leave the domain's
         deadline armed (gh-ocannl-1446): the child dies by it, rather than outliving the stall. *)
      let child = ref None and wedged = ref None in
      guarded_call name (fun () ->
          match Unix.fork () with
          | 0 ->
              let armed = Atomic.make false in
              Utils.device_init_test_stall :=
                Some
                  (fun () ->
                    Atomic.set armed true;
                    stall_for 3.0 ());
              wedged :=
                Some
                  (Stdlib.Domain.spawn (fun () ->
                       try call_entry name "num_devices" with Stall_done -> ()));
              while not (Atomic.get armed) do
                Unix.sleepf 0.01
              done;
              (* Leaves the inherited region: [guarded_call] catches it, past the region's
                 disarm. *)
              raise Stall_done
          | pid ->
              child := Some pid;
              raise Stall_done);
      (match (!child, !wedged) with
      | Some pid, _ -> say "fork-child %s" (reap_child pid)
      | None, Some domain ->
          (* The child, out of the region it inherited: it outlives the wedged stall only if that
             region's disarm took the domain's deadline back. *)
          Stdlib.Domain.join domain;
          Stdlib.exit 0
      | None, None -> ());
      say "survived"
  | [ "concurrent"; first; second ] ->
      (* The first region to take the hook is the healthy one: it returns 0.3 s in. This thread
         enters SECOND's region only once that hook has started, i.e. once the other domain holds
         the region, so SECOND's is the wedged one (5 s) whatever the scheduling. *)
      let calls = Atomic.make 0 and holding = Atomic.make false in
      Utils.device_init_test_stall :=
        Some
          (fun () ->
            if Atomic.fetch_and_add calls 1 = 0 then (
              Atomic.set holding true;
              Unix.sleepf 0.3;
              raise Stall_done)
            else stall_for 5.0 ());
      let healthy =
        Stdlib.Domain.spawn (fun () -> try call_entry first "num_devices" with Stall_done -> ())
      in
      while not (Atomic.get holding) do
        Unix.sleepf 0.01
      done;
      (try call_entry second "num_devices" with Stall_done -> ());
      Stdlib.Domain.join healthy;
      say "survived"
  | [ "inherited-siginfo"; name ] ->
      catch_sigalrm_siginfo ();
      let exe = Stdlib.Sys.executable_name in
      Unix.execv exe (Array.of_list ((exe :: flags) @ [ "exec-stall"; name ]))
  | [ "exec-stall"; name ] ->
      say "siginfo-inherited %b" (sigalrm_default_with_siginfo ());
      guarded_call name (stall_for 30.0);
      say "survived"
  | [ "unguarded"; name ] ->
      guarded_call name (fun () ->
          say "hook-ran";
          raise Stall_done);
      say "returned"
  | _ -> failwith ("device_init_deadline_child: bad arguments: " ^ String.concat ~sep:" " args)
