(* What the two real-stall Metal tests read off the OS rather than off the backend under test
   (gh-ocannl-2082, gh-ocannl-2139): whether macOS killed a hung kernel, and a bound on a run that
   no OCaml code in the process can enforce. Shared by metal_gpu_hang_failure and
   metal_wait_bound_real_hang, which each choose a leg on [recovery_count], and by the hand-run
   probe test/metal_probe/metal_hang_probe.ml, so the probe and the tests cannot disagree on what a
   recovery is. *)

open Base

(* The GPU recoveries the IORegistry has counted since boot, summed over its accelerators. A kill of
   a hung kernel is one. Fails rather than reading 0 when [ioreg] fails, names no counter or prints
   one that does not parse: a leg chosen on a 0 that was never read would claim the OS did not
   kill. *)
let recovery_count () =
  let ic = Unix.open_process_in "/usr/sbin/ioreg -r -c IOAccelerator -l" in
  let samples = ref 0 in
  let rec sum acc =
    match Stdio.In_channel.input_line ic with
    | None -> acc
    | Some line -> (
        match String.substr_index line ~pattern:"\"recoveryCount\"=" with
        | None -> sum acc
        | Some i -> (
            let rest = String.drop_prefix line (i + 16) in
            let len =
              Option.value ~default:(String.length rest)
                (String.lfindi rest ~f:(fun _ c -> not (Char.is_digit c)))
            in
            match Int.of_string_opt (String.prefix rest len) with
            | Some v ->
                Int.incr samples;
                sum (acc + v)
            | None -> failwith ("unparseable recoveryCount in: " ^ line)))
  in
  let count = sum 0 in
  match Unix.close_process_in ic with
  | Unix.WEXITED 0 when !samples > 0 -> count
  | _ -> failwith "ioreg gave no GPU recoveryCount: whether the OS killed the kernel is unknown"

(* A hard bound on the whole run that needs no OCaml code to run: a wait stuck in a native call
   defers a SIGALRM handler for as long as it lasts. A separate process sends the default-action
   SIGTERM, which ends this one and fails the run; [at_exit] retires it on a normal end. *)
let arm_watchdog ~seconds =
  let null = Unix.openfile "/dev/null" [ O_RDWR ] 0 in
  let watchdog =
    Unix.create_process "/bin/sh"
      [| "sh"; "-c"; Printf.sprintf "sleep %d; kill -TERM %d" seconds (Unix.getpid ()) |]
      null null null
  in
  Unix.close null;
  Stdlib.at_exit (fun () -> try Unix.kill watchdog Stdlib.Sys.sigkill with _ -> ())
