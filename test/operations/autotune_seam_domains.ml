(* The search's test seams are per domain ([Autotune.Seam], post-wave task 13 of #1263's hand-back):
   a seam set in one domain is what that domain's searches call, a search running in another domain
   neither reaches it nor disturbs it, and a fresh domain starts with the default.

   Two searches run in two domains -- this one and a spawned one -- each with its own value set in
   EVERY seam the search calls ([on_candidate_attempt], [on_candidate_preflight],
   [on_candidate_timed], [on_candidate_measured], [on_candidate_callback],
   [with_uncontended_test_windows]) and its own timing-call observer, and with BOTH domains' seams
   set throughout both searches: the spawned domain sets its values before this one searches, and
   keeps them while it searches itself. Each observer tags what it sees with the search that was
   running when it fired; the claims are that every observer saw events (so none of them holds
   vacuously), only its own search's, and as many as its search's report accounts for.

   The two searches take turns rather than overlapping. What is under test is where a seam's value
   lives, which the turns expose exactly as well: with process-wide seams, the spawned domain's
   settings would replace this domain's, so this domain's search would reach the spawned domain's
   observers and its own would see nothing. Whether two searches can share the compile stack at the
   same instant is a separate question, about state this task does not touch.

   The measurement seam is also the INJECTION direction: each domain's measured seam reads every
   admitted window as its own constant (1 ms here, 2 ms there), so each report's [best_ms] says
   whose seam ranked that search -- a search that reached the other domain's value reports the other
   constant.

   Pinned to cc: the property is the OCaml runtime's and the search's own code, while a GPU backend
   binds device state per thread, which would make a search from a second domain a question about
   the backend rather than about the seams. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

type searcher = Main | Spawned

let searcher_name = function Main -> "main" | Spawned -> "spawned"

(* Which search is running, if any. Read by both domains' observers, written by the domain that is
   about to search; the turn hand-off below orders every write before the other domain's reads. *)
let running : searcher option Stdlib.Atomic.t = Stdlib.Atomic.make None

type seen = {
  attempts : searcher option list ref;
  preflights : searcher option list ref;
  timed : (searcher option * int) list ref;
  measured : searcher option list ref;
  callbacks : searcher option list ref;
  calls : searcher option list ref;
}

let fresh_seen () =
  {
    attempts = ref [];
    preflights = ref [];
    timed = ref [];
    measured = ref [];
    callbacks = ref [];
    calls = ref [];
  }

let record r = r := Stdlib.Atomic.get running :: !r

(* Sets every seam, and adds a timing-call observer, in the CALLING domain for the rest of its life;
   [ms] is what this domain's measurement seam reads every admitted window as. *)
let install seen ~ms =
  Autotune.Seam.set Autotune.on_candidate_attempt (fun _label -> record seen.attempts);
  Autotune.Seam.set Autotune.on_candidate_preflight (fun _routine -> record seen.preflights);
  Autotune.Seam.set Autotune.on_candidate_timed (fun _routine ~timed_so_far ->
      seen.timed := (Stdlib.Atomic.get running, timed_so_far) :: !(seen.timed));
  Autotune.Seam.set Autotune.on_candidate_measured (fun ~label:_ ~digest:_ _ms ->
      record seen.measured;
      ms);
  Autotune.Seam.set Autotune.on_candidate_callback (fun _ ~candidate_ms:_ ~incumbent_ms:_ ->
      record seen.callbacks);
  Autotune.install_timing_call_observer (fun _call -> record seen.calls)

let matmul ~n ~modulus =
  let a =
    TDSL.ndarray
      (Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus ~offset:0. ~stride:1.))
      ~input_dims:[ n ] ~output_dims:[ n ] ()
  in
  let b =
    TDSL.ndarray
      (Array.init (n * n)
         ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:(modulus + 2) ~offset:0. ~stride:1.))
      ~input_dims:[ n ] ~output_dims:[ n ] ()
  in
  let%op product = a * b in
  Train.forward product

(* One search, in the calling domain, with the turn marked for the observers. The windows are opted
   out of host contention (per domain, like the seams), so every admitted window reaches the
   measurement seam whatever else the machine is doing. *)
let search who comp =
  let report = ref None in
  Stdlib.Atomic.set running (Some who);
  Exn.protect
    ~finally:(fun () -> Stdlib.Atomic.set running None)
    ~f:(fun () ->
      let ctx, _routine =
        Autotune.with_uncontended_test_windows (fun () ->
            Autotune.tune ~search:true ~beam_width:1 ~rounds:0 ~repeats:1 ~timing:Autotune.Isolated
              ~cache_dir:"" ~log:false ~progress:false
              ~report:(fun r -> report := Some r)
              (Context.cpu ()) comp Ir.Indexing.Empty)
      in
      Context.release ctx);
  Option.value_exn ~message:(searcher_name who ^ ": the search reported nothing") !report

(* A one-way turn signal between the two domains. *)
type signal = { mutex : Stdlib.Mutex.t; cond : Stdlib.Condition.t; mutable raised : bool }

let signal () =
  { mutex = Stdlib.Mutex.create (); cond = Stdlib.Condition.create (); raised = false }

let raise_signal s =
  Stdlib.Mutex.protect s.mutex (fun () ->
      s.raised <- true;
      Stdlib.Condition.broadcast s.cond)

let await_signal s =
  Stdlib.Mutex.protect s.mutex (fun () ->
      while not s.raised do
        Stdlib.Condition.wait s.cond s.mutex
      done)

let () =
  let comp_main = matmul ~n:16 ~modulus:7 and comp_spawned = matmul ~n:16 ~modulus:5 in
  let main_seen = fresh_seen () and spawned_seen = fresh_seen () in
  install main_seen ~ms:1.;
  let main_attempt = Autotune.Seam.get Autotune.on_candidate_attempt in
  let spawned_ready = signal () and spawned_turn = signal () in
  let spawned =
    Stdlib.Domain.spawn (fun () ->
        (* A fresh domain starts with the defaults, whatever its parent set: the identity
           measurement, not the parent's constant. *)
        let inherited_ms =
          Autotune.Seam.get Autotune.on_candidate_measured ~label:"" ~digest:"" 5.
        in
        install spawned_seen ~ms:2.;
        raise_signal spawned_ready;
        await_signal spawned_turn;
        let report = search Spawned comp_spawned in
        (inherited_ms, report))
  in
  (* The spawned domain's seams are set from here on, through both searches. *)
  await_signal spawned_ready;
  let report_main =
    Exn.protect
      ~f:(fun () -> search Main comp_main)
      ~finally:(fun () ->
        (* Even if this search fails, the spawned domain must be released to finish. *)
        raise_signal spawned_turn)
  in
  let inherited_ms, report_spawned = Stdlib.Domain.join spawned in
  p "a fresh domain starts with the default measurement seam, not its parent's"
    (Float.equal inherited_ms 5.);
  p "the spawned domain's searching left this domain's seam in place"
    (phys_equal (Autotune.Seam.get Autotune.on_candidate_attempt) main_attempt);
  let check who seen (report : Autotune.report) ~ms =
    let name = searcher_name who in
    let own tag = Option.equal Poly.equal tag (Some who) in
    let claim_own seam events =
      p_all
        (Printf.sprintf "%s domain: every %s event came from its own search" name seam)
        events ~f:own
    in
    Stdio.eprintf
      "(not part of the golden) %s: %d attempts, %d preflights, %d timed, %d measured, %d \
       callbacks, %d timing calls; report: %d timed, best %g ms\n"
      name (List.length !(seen.attempts)) (List.length !(seen.preflights))
      (List.length !(seen.timed)) (List.length !(seen.measured)) (List.length !(seen.callbacks))
      (List.length !(seen.calls)) report.candidates_timed report.best_ms;
    claim_own "attempt" !(seen.attempts);
    claim_own "preflight" !(seen.preflights);
    claim_own "timed" (List.map !(seen.timed) ~f:fst);
    claim_own "measured" !(seen.measured);
    claim_own "callback" !(seen.callbacks);
    claim_own "timing-call" !(seen.calls);
    (* Complete, not merely clean: the timed seam saw every admission its own report counts. *)
    p
      (name ^ " domain: the timed seam saw as many admissions as its search's report counts")
      (List.length !(seen.timed) = report.candidates_timed
      && List.exists !(seen.timed) ~f:(fun (_, so_far) -> so_far = report.candidates_timed));
    p
      (name ^ " domain: its search was ranked by its own measurement seam")
      (Float.equal report.best_ms ms)
  in
  check Main main_seen report_main ~ms:1.;
  check Spawned spawned_seen report_spawned ~ms:2.
