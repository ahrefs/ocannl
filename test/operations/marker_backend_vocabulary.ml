(* The one relationship the dune-marker grammar rests on and nothing used to state (gh-ocannl-689).

   [Dune_stanza_scan.marker_backends] is the closed vocabulary of the [; ocannl-backend: <word> --
   <reason>] stanza marker, and its closedness is load-bearing: it is what makes [; ocannl-backend:
   metl -- ...] fail instead of reading as a truthful exemption. But the list is written out as
   text, deliberately -- the scanning tests link [arrayjit.utils] and the source scanners, and
   dragging the backend closure into a check that reads dune files would be a bad trade -- so
   nothing in the scanner relates it to the backends OCANNL actually has.

   Both drifts are silent, and the one that matters is not the one it looks like. Remove or rename a
   backend and markers naming the dead one keep passing. Add one, and the CORRECT marker for it is
   rejected as malformed; the author's fix is then to reach for [none], which is a lie the grammar
   accepts. So the failure mode is not a red build someone fixes by editing a list, it is pressure
   toward the least honest classification available.

   This test is where the two lists meet, and it exists as its own executable because that is the
   whole cost being managed: it links the backend closure so that no scanning test has to. It starts
   no context, opens no device and compiles nothing -- [Backends.all_of_backend] is a derived
   constant.

   It has since become the place for every relationship that needs the closure, for the same reason:
   the second one below relates the backends to [Ir.Schedule]'s CPU/GPU predicates. *)

open Base
module Backends = Context.Backends
module Scan = Test_utils.Dune_stanza_scan
module Sched = Ir.Schedule

let () =
  let sorted l = List.sort l ~compare:String.compare in
  (* Plus [none], which is not a backend: it says the run does not depend on the configured backend
     at all. Every other word must name one. *)
  let expected = sorted ("none" :: List.map Backends.all_of_backend ~f:Backends.backend_name) in
  let admitted = sorted Scan.marker_backends in
  (* Compared as sorted lists rather than as sets, so a word repeated on either side is a mismatch
     too: a vocabulary with a duplicate in it is not the set it is standing in for. *)
  let agree = List.equal String.equal admitted expected in
  if not agree then
    (* The claim is a bare boolean so that the golden stays fixed as backends come and go; the two
       lists go to stderr, where a failing run shows which way they parted. *)
    Stdio.eprintf "the marker admits [%s]; the backends OCANNL has, plus none, are [%s]\n"
      (String.concat ~sep:"; " admitted)
      (String.concat ~sep:"; " expected);
  Verdict.p "the ocannl-backend marker vocabulary is exactly OCANNL's backends plus none" agree

(* The second relationship, gh-ocannl-706. [Ir.Schedule.backend_is_gpu] and [backend_is_cpu] decide
   which legs a test may evaluate on the configured backend -- whether a kernel has barriers and
   shared memory, whether Grid loops render on the CPU pool. Both are substring tests over the
   backend's name, so a backend the predicates were never taught reads as neither, and every caller
   phrased as [if on_gpu then ... else ...] takes the CPU branch for it: the leg goes green having
   evaluated the wrong half, which is the failure a skip is supposed to make visible.

   Nothing said the two predicates had to cover the backends. Now the covering is the claim, and
   exactly-one rather than at-least-one, since a name both predicates claim is as undecided as one
   neither claims. This is also what makes the test-side restatements -- 29 files spelled the
   substring test out instead of asking, before this change -- unnecessary rather than merely
   discouraged: there is one classification, and adding a backend fails here until it has one. *)
let () =
  let classify name = (Sched.backend_is_cpu name, Sched.backend_is_gpu name) in
  let undecided name = match classify name with true, false | false, true -> false | _ -> true in
  let describe name =
    match classify name with
    | false, false -> "claimed by neither backend_is_cpu nor backend_is_gpu"
    | _ -> "claimed by both backend_is_cpu and backend_is_gpu"
  in
  let backends = List.map Backends.all_of_backend ~f:Backends.backend_name in
  let stray = List.filter backends ~f:undecided in
  List.iter stray ~f:(fun name -> Stdio.eprintf "%s is %s\n" name (describe name));
  Verdict.p_empty
    "every backend OCANNL has is classified CPU or GPU, by exactly one of the predicates"
    ~over:backends stray;
  (* Put to names that are not backends, because every backend today satisfies the rule: a control
     drawn from the corpus would encode the absence of the violating shape, which a rule deciding
     nothing satisfies just as well. Both violating shapes, since the claim above rejects both. *)
  Verdict.p "a name neither predicate knows is undecided" (undecided "vulkan");
  Verdict.p "a name both predicates claim is undecided too" (undecided "cc_metal");
  Verdict.p "and a backend name proper is decided" (not (undecided "metal"))
