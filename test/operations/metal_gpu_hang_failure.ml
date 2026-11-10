(* gh-ocannl-2083: a command buffer the GPU fails is a raised failure, with Metal's own error text,
   on the wait that covers it and on every later use of the device.

   The kernel below never ends: Metal has no way to fail a command buffer on demand. Wild accesses
   complete with no error, even 1 TiB past a buffer and under shader validation, and neither does a
   purgeable pool emptied under a launch, so a stall the OS kills is the only failure there is. On a
   bare event wait that case was a silent success: Metal delivers the event signals of the buffers
   discarded behind a hang, so the wait returned and the host read the stale prefill. A later
   failure was the hang gh-ocannl-2083 was filed on: Metal then ignores the process's submissions,
   and an ignored signal never fires. The device refusing work after the first failure keeps a run
   away from that state, so no claim here can reach it.

   Whether the OS kills the kernel at all is the platform's call, not the backend's
   (gh-ocannl-2082). On mac-studio (M4 Max, macOS 27.2) the watchdog killed it about 3 s in on
   2026-11-07, with "Caused GPU Hang Error". On macbook-pro (M4 Pro, macOS 27.2) on 2026-11-10 it
   was still Scheduled after 120 s, with no GPU recovery and other processes' kernels running beside
   it. A legitimate 30 s kernel completes there too. So the wait is given [kill_bound_s]. Whether a
   kill happened is read from the IORegistry's GPU [recoveryCount], never from the backend under
   test. The claims about the report are evaluated only where the count rose, and skipped otherwise.
   The claim that the wait never returned success holds either way. Ending the process ends the
   kernel at once, so a run that gives up leaves the GPU free for the next one.

   Gated behind `slow`: each run costs up to [kill_bound_s] of GPU stall, and a GPU recovery where
   the OS kills. Evaluated on Metal only; the watchdog and the status reads this pins are that
   backend's. *)

open Base
open Verdict.Claims

let n = 8
let ctx = Context.auto ()
let backend_name = Context.backend_name ctx
let leg = Verdict.leg ~when_:(String.equal backend_name "metal") ~on:backend_name ()

(* How long the wait over the never-ending kernel is given for the OS to kill it: several times the
   ~3 s a kill took where the watchdog fired (gh-ocannl-2082). *)
let kill_bound_s = 20

(* The GPU recoveries the IORegistry has counted since boot, summed over its accelerators. A kill of
   a hung kernel is one. *)
let recovery_count () =
  let ic = Unix.open_process_in "/usr/sbin/ioreg -r -c IOAccelerator -l" in
  let rec sum acc =
    match Stdio.In_channel.input_line ic with
    | None -> acc
    | Some line -> (
        match String.substr_index line ~pattern:"\"recoveryCount\"=" with
        | None -> sum acc
        | Some i ->
            let rest = String.drop_prefix line (i + 16) in
            let len =
              Option.value ~default:(String.length rest)
                (String.lfindi rest ~f:(fun _ c -> not (Char.is_digit c)))
            in
            sum (acc + Option.value (Int.of_string_opt (String.prefix rest len)) ~default:0))
  in
  let count = sum 0 in
  ignore (Unix.close_process_in ic : Unix.process_status);
  count

(* Runs a [+1] routine, whose output starts blank, and says whether it read back right. *)
let plain ~first_id ~name ctx =
  let mk = Ll_test.node_factory ~first_id ~dims:[| n |] () in
  let a = mk (name ^ "_src") and x = mk name in
  List.iter [ a; x ] ~f:Ll_test.materialize;
  let i = Ll_test.sym () in
  let at = Ll_test.iter i in
  let llc =
    Ll_test.loop_n i n (Ll_test.set_at x at (Ll_test.add (Ll_test.get a [| at |]) (Ll_test.c 1.)))
  in
  let o = Ll_test.optimize ~materialized:[ a; x ] ~name llc in
  let av = Array.init n ~f:Float.of_int in
  let ctx, _routine = Ll_test.run_routine ~ctx ~name o ~seed:[ (a, av); (x, Ll_test.blank n) ] in
  Context.sync ctx;
  Array.equal Float.equal (Context.get_values ctx x) (Array.map av ~f:(fun v -> v +. 1.))

(* Far past any watchdog: 10^15 read-modify-writes the compiler cannot fold away. *)
let stalling ~first_id ~name =
  let x = Ll_test.node_factory ~first_id ~dims:[| n |] () name in
  Ll_test.materialize x;
  let cell = [| Ll_test.fixed 0 |] in
  let i = Ll_test.sym () and j = Ll_test.sym () and k = Ll_test.sym () in
  let nest s body = Ll_test.loop_n s 100_000 body in
  let llc =
    nest i (nest j (nest k (Ll_test.set x cell (Ll_test.add (Ll_test.get x cell) (Ll_test.c 1.)))))
  in
  (x, Ll_test.optimize ~materialized:[ x ] ~name llc)

type outcome = {
  before : bool;
  bounded : bool;  (** The wait was still waiting at [kill_bound_s]. *)
  killed : bool;  (** The GPU recovery count rose while the kernel was in flight. *)
  report : string option;
  read : string option;
  later : string option;
}

exception Still_waiting

let raised f =
  match f () with
  | _ -> None
  | exception Failure message -> Some message
  | exception e -> Some (Exn.to_string e)

(* A hard bound on the whole scenario that needs no OCaml code to run: a wait stuck in a native call
   defers the SIGALRM handler below for as long as it lasts. A separate process sends the default-
   action SIGTERM, which ends this one and fails the run; [at_exit] retires it on a normal end. *)
let arm_watchdog ~seconds =
  let null = Unix.openfile "/dev/null" [ O_RDWR ] 0 in
  let watchdog =
    Unix.create_process "/bin/sh"
      [| "sh"; "-c"; Printf.sprintf "sleep %d; kill -TERM %d" seconds (Unix.getpid ()) |]
      null null null
  in
  Unix.close null;
  Stdlib.at_exit (fun () -> try Unix.kill watchdog Stdlib.Sys.sigkill with _ -> ())

let scenario () =
  arm_watchdog ~seconds:(kill_bound_s + 100);
  let before = plain ~first_id:9_000 ~name:"before" ctx in
  let x, o = stalling ~first_id:9_100 ~name:"stalling" in
  let recoveries = recovery_count () in
  let run_ctx, _routine =
    Ll_test.run_routine ~ctx ~name:"stalling" o ~seed:[ (x, Ll_test.blank n) ]
  in
  (* The wait polls the device between slices of OCaml code, so the handler's exception ends it. *)
  Stdlib.Sys.set_signal Stdlib.Sys.sigalrm (Signal_handle (fun _ -> raise Still_waiting));
  ignore (Unix.alarm kill_bound_s : int);
  let waited = Unix.gettimeofday () in
  let report, bounded =
    match Context.sync run_ctx with
    | () -> (None, false)
    | exception Still_waiting -> (None, true)
    | exception Failure message -> (Some message, false)
    | exception e -> (Some (Exn.to_string e), false)
  in
  ignore (Unix.alarm 0 : int);
  Stdlib.Sys.set_signal Stdlib.Sys.sigalrm Signal_default;
  let waited = Unix.gettimeofday () -. waited in
  (* The count can trail the failed status; a kernel nobody killed costs the full poll. *)
  let rec rose tries =
    recovery_count () > recoveries
    || tries > 0
       &&
       (Unix.sleepf 0.1;
        rose (tries - 1))
  in
  let killed = rose 20 in
  Stdio.eprintf "the wait %s after %.1f s; GPU recoveries %s (not part of the golden)\n%!"
    (if bounded then "was still waiting" else "ended")
    waited
    (if killed then "rose" else "did not rise");
  Option.iter report ~f:(Stdio.eprintf "the wait raised: %s (not part of the golden)\n%!");
  (* Behind a kernel still running, both of these would wait for it. *)
  let read, later =
    if bounded then (None, None)
    else
      ( raised (fun () -> Context.get_values run_ctx x),
        (* A fresh lineage, so that this reads the device and not the failed lineage's poison. *)
        raised (fun () -> plain ~first_id:9_200 ~name:"later" (Context.auto ())) )
  in
  { before; bounded; killed; report; read; later }

let outcome = lazy (scenario ())

(* Open where the OS killed the kernel: read off the GPU recovery count, not off the backend. *)
let killed_leg =
  Verdict.sub_leg ~aggregation:`Environment leg
    ~when_:(lazy (Lazy.force outcome).killed)
    ~on:"a macOS that did not kill the never-ending kernel within the bound (gh-ocannl-2082)"

let claim ?(leg = leg) label read =
  Verdict.in_leg ~leg label (fun label -> p label (read (Lazy.force outcome)))

let () =
  claim "a routine before the fault ran and read back" (fun o -> o.before);
  claim "the wait over the never-ending kernel never returned success" (fun o ->
      o.bounded || Option.is_some o.report);
  claim ~leg:killed_leg "the wait over the killed kernel raised" (fun o -> Option.is_some o.report);
  claim ~leg:killed_leg "the report names the kernel" (fun o ->
      Option.exists o.report ~f:(String.is_substring ~substring:"kernel stalling"));
  claim ~leg:killed_leg "the report carries Metal's error text" (fun o ->
      Option.exists o.report ~f:(fun report ->
          String.is_substring report ~substring:"failed: "
          && not (String.is_substring report ~substring:"no error text")));
  claim ~leg:killed_leg "reading the kernel's output raised, not returned stale cells" (fun o ->
      Option.is_some o.read);
  claim ~leg:killed_leg "a fresh lineage on the failed device raised the same report" (fun o ->
      match (o.report, o.later) with
      | Some report, Some later -> String.is_substring later ~substring:report
      | _ -> false)
