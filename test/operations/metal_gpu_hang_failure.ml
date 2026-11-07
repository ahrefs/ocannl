(* gh-ocannl-2083: a command buffer the GPU fails is a raised failure, with Metal's own error text,
   on the wait that covers it and on every later use of the device.

   The kernel below runs until the OS watchdog kills it: Metal has no way to fail a command buffer
   on demand. Wild accesses complete with no error, and neither does a purgeable pool emptied under
   a launch, so a stall is the failure that can be had. On a bare event wait this case is a silent
   success: Metal delivers the event signals of the buffers discarded behind a hang, so the wait
   returned and the host read the stale prefill. A later failure was the hang gh-ocannl-2083 was
   filed on: Metal then ignores the process's submissions, and an ignored signal never fires. The
   device refusing work after the first failure keeps a run away from that state, so no claim here
   can reach it.

   Gated behind `slow`: each run costs seconds of GPU stall and one GPU recovery. Evaluated on Metal
   only; the watchdog and the status reads this pins are that backend's. *)

open Base
open Verdict.Claims

let n = 8
let ctx = Context.auto ()
let backend_name = Context.backend_name ctx
let leg = Verdict.leg ~when_:(String.equal backend_name "metal") ~on:backend_name ()

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
  report : string option;
  read : string option;
  later : string option;
}

let raised f =
  match f () with
  | _ -> None
  | exception Failure message -> Some message
  | exception e -> Some (Exn.to_string e)

let scenario () =
  (* A regressed wait must end the run, and no OCaml can be relied on to run during one: the binding
     of the event wait keeps the runtime lock. *)
  ignore (Unix.alarm 300 : int);
  let before = plain ~first_id:9_000 ~name:"before" ctx in
  let x, o = stalling ~first_id:9_100 ~name:"stalling" in
  let run_ctx, _routine =
    Ll_test.run_routine ~ctx ~name:"stalling" o ~seed:[ (x, Ll_test.blank n) ]
  in
  let report = raised (fun () -> Context.sync run_ctx) in
  let read = raised (fun () -> Context.get_values run_ctx x) in
  (* A fresh lineage, so that this reads the device and not the failed lineage's poison. *)
  let later = raised (fun () -> plain ~first_id:9_200 ~name:"later" (Context.auto ())) in
  Option.iter report ~f:(Stdio.eprintf "the wait raised: %s (not part of the golden)\n%!");
  { before; report; read; later }

let outcome = lazy (scenario ())
let claim label read = Verdict.in_leg ~leg label (fun label -> p label (read (Lazy.force outcome)))

let () =
  claim "a routine before the fault ran and read back" (fun o -> o.before);
  claim "the wait over the killed kernel raised" (fun o -> Option.is_some o.report);
  claim "the report names the kernel" (fun o ->
      Option.exists o.report ~f:(String.is_substring ~substring:"kernel stalling"));
  claim "the report carries Metal's error text" (fun o ->
      Option.exists o.report ~f:(fun report ->
          String.is_substring report ~substring:"failed: "
          && not (String.is_substring report ~substring:"no error text")));
  claim "reading the kernel's output raised, not returned stale cells" (fun o ->
      Option.is_some o.read);
  claim "a fresh lineage on the failed device raised the same report" (fun o ->
      match (o.report, o.later) with
      | Some report, Some later -> String.is_substring later ~substring:report
      | _ -> false)
