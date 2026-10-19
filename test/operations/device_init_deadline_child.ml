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

   [again BACKEND]: one real [num_devices] call (on a missing backend's stub, or on a device that
   answers), then a 30 s stall in the same entry point.

   [blocked BACKEND]: SIGALRM is blocked in the (only) thread first; then a 5 s stall.

   [blocked-pending BACKEND]: SIGALRM is blocked and then sent to the process, so it is pending;
   then a 1.5 s stall, and whether SIGALRM is still blocked afterwards.

   [external BACKEND (kill|raise)]: another domain sends the process a SIGALRM 0.3 s into a 5 s
   stall.

   [full-stderr BACKEND]: fd 2 is replaced by a full pipe nobody reads; then a 30 s stall.

   [concurrent FIRST SECOND]: a second domain enters FIRST's guarded [num_devices], whose stall is
   healthy (0.3 s); once that stall has begun, this thread enters SECOND's, whose stall is 5 s.

   Times are seconds on the monotonic clock, which is system-wide, so the parent can subtract its
   own reading from them. *)

open Base
open Stdio
module Backends = Context.Backends

exception Stall_done

external raise_sigalrm : unit -> unit = "ocannl_test_raise_sigalrm"

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

let start_timer seconds =
  ignore
    (Unix.setitimer Unix.ITIMER_REAL { Unix.it_interval = 0.; it_value = seconds }
      : Unix.interval_timer_status)

let () =
  let args =
    Array.to_list (Sys.get_argv ())
    |> List.tl_exn
    |> List.filter ~f:(fun a -> not (String.is_prefix a ~prefix:"--"))
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
      guarded_call name (stall_for 1.5);
      let mask = Unix.sigprocmask Unix.SIG_BLOCK [] in
      say "still-blocked %b" (List.mem mask Stdlib.Sys.sigalrm ~equal:Int.equal);
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
  | [ "full-stderr"; name ] ->
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
  | [ "unguarded"; name ] ->
      guarded_call name (fun () ->
          say "hook-ran";
          raise Stall_done);
      say "returned"
  | _ -> failwith ("device_init_deadline_child: bad arguments: " ^ String.concat ~sep:" " args)
