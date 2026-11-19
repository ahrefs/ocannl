(** This process's open descriptors, and the bound a run of compiles holds them to (gh-ocannl-1784,
    gh-ocannl-1882).

    A debug-artifact writer that keeps its descriptor past its write leaks one or more per compile,
    and only a long run under a low limit (macOS's 256 in a non-interactive ssh shell) notices.
    [build_file_descriptors] pins the contract on a trivial routine, and [schedule_mma_matmul], the
    longest compile-heavy test, holds every writer its routines reach to it. Both read the same
    counter and claim the same bound. Each test's negative control leaks a channel count the test
    writes as its own literal, independent of {!bound}, and claims the bound rejects it: so a
    counter that cannot see growth fails the control, and so does a bound loosened past that count.
    [schedule_mma_matmul] also claims the bound, its baseline and its control fit under
    {!low_shell_limit}.

    Each test makes its claims in its own [Verdict.leg], closed on Windows, which has no [/dev/fd]
    (built at the claim's site, where [verdict_ratchet]'s gate reader follows it): the readings
    there are 0, and only the skipped claims would read them. *)

open Base

(** The number of this process's open descriptors. [/dev/fd] lists them on macOS and Linux; the
    listing's own descriptor is counted in every reading alike. *)
let count () = if Stdlib.Sys.win32 then 0 else Array.length (Stdlib.Sys.readdir "/dev/fd")

(** The descriptor limit of a non-interactive ssh shell on macOS, under which [schedule_mma_matmul]
    once ran out of descriptors (gh-ocannl-1784). *)
let low_shell_limit = 256

(** The most new descriptors any number of compiles may leave open: the room for one-time state
    opened on demand after the warm-up compile -- driver state a routine's first compile or launch
    of a new kind opens, and that stays open for the process's lifetime. On cc and Metal that state
    measured nothing (the tests print their growth on stderr); the room is for backends where it was
    not measured (CUDA and HIP), and it stays far below {!low_shell_limit}. *)
let allowance = 16

(** [bound ~compiles] is the number of new open descriptors [compiles] compiles, read after a
    warm-up compile, must stay below: one per compile up to {!allowance}. Every artifact writer runs
    on every compile of its backend, so a writer that keeps even one descriptor breaks it once the
    run has [allowance] compiles, and a short run breaks it from its first leaked descriptor per
    compile; a leak only some compiles reach breaks it once it has leaked [allowance]. *)
let bound ~compiles = Int.min compiles allowance

(** [within_bound ~compiles growth] is [growth < bound ~compiles]. *)
let within_bound ~compiles growth = growth < bound ~compiles

(** [leak_growth n] keeps [n] channels open at once, reads the growth through {!count} and closes
    them: the negative control's measurement. [n] must be positive, since a control that leaks
    nothing shows nothing. *)
let leak_growth n =
  if n < 1 then invalid_arg (Printf.sprintf "Descriptors.leak_growth: %d channels leak nothing" n);
  let before = count () in
  let leaked = List.init n ~f:(fun (_ : int) -> Stdio.Out_channel.create "/dev/null") in
  let growth = count () - before in
  List.iter leaked ~f:Stdio.Out_channel.close;
  growth
