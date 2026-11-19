(** This process's open descriptors, and the bound a run of compiles holds them to (gh-ocannl-1784,
    gh-ocannl-1882).

    A debug-artifact writer that keeps its descriptor past its write leaks one or more per compile,
    and only a long run under a low limit (macOS's 256 in a non-interactive ssh shell) notices.
    [build_file_descriptors] pins the contract on a trivial routine, and [schedule_mma_matmul], the
    longest compile-heavy test, holds every writer its routines reach to it. Both read the same
    counter, claim the same bound, and run the same leaking control through them, so a counter that
    cannot see growth, or a bound loose enough to admit a per-compile leak, cannot pass either test.

    Each test makes its claims in its own [Verdict.leg], closed on Windows, which has no [/dev/fd]
    (built at the claim's site, where [verdict_ratchet]'s gate reader follows it): the readings
    there are 0, and only the skipped claims would read them. *)

open Base

(** The number of this process's open descriptors. [/dev/fd] lists them on macOS and Linux; the
    listing's own descriptor is counted in every reading alike. *)
let count () = if Stdlib.Sys.win32 then 0 else Array.length (Stdlib.Sys.readdir "/dev/fd")

(** The most new descriptors any number of compiles may leave open: the room for one-time state
    opened on demand after the warm-up compile -- driver state a routine's first compile or launch
    of a new kind opens, and that stays open for the process's lifetime. Measured on cc and Metal
    (mac-studio, M4 Max), that state is nothing: [schedule_mma_matmul] grew by 0 across 41 and 300
    compiles. The room is for backends where it was not measured (CUDA and HIP), and it stays far
    below the 256-descriptor shell that killed [schedule_mma_matmul] (gh-ocannl-1784), whose
    baseline after its first compile is 10 to 14. *)
let allowance = 16

(** [bound ~compiles] is the number of new open descriptors [compiles] compiles, read after a
    warm-up compile, must stay below: one per compile up to {!allowance}. Every artifact writer runs
    on every compile of its backend, so a writer that keeps even one descriptor breaks it once the
    run has [allowance] compiles, and a short run breaks it from its first leaked descriptor per
    compile; a leak only some compiles reach breaks it once it has leaked [allowance]. *)
let bound ~compiles = Int.min compiles allowance

(** [within_bound ~compiles growth] is [growth < bound ~compiles]. *)
let within_bound ~compiles growth = growth < bound ~compiles

(** [leak_breaks_bound ~compiles] keeps one channel open per compile up to the bound, reads the
    growth through {!count} and closes them: whether {!within_bound} rejects that leak -- the
    negative control of the claim beside it. At most {!allowance} channels, so the control cannot
    itself run a 256-descriptor shell out of descriptors. *)
let leak_breaks_bound ~compiles =
  let steps = bound ~compiles in
  let before = count () in
  let leaked = List.init steps ~f:(fun (_ : int) -> Stdio.Out_channel.create "/dev/null") in
  let growth = count () - before in
  List.iter leaked ~f:Stdio.Out_channel.close;
  not (within_bound ~compiles growth)
