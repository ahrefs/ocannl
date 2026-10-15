(* [Ir.Resource_fault_injection]'s callback is per domain (post-wave task 5 of wave-20261014, after
   the [Autotune.Seam] treatment of #1142): a callback installed in one domain is what that domain's
   operations call, an operation running in another domain neither reaches it nor disturbs it, and a
   fresh domain starts with the no-op. Per domain is only sound because every injection point fires
   on the domain that CALLED the operation, never on a scheduler's worker domain; the second half of
   this test pins that on [multidev_cc], whose tasks run on worker domains.

   The scenario reaches every injection point an uninjected run can: a compile (the delta pools and
   the link), a run and a readback, a fresh host upload, the release of both contexts, and a
   schedule-cache store and lookup in a fresh directory. Only the failure-cleanup point
   ([Transfer_cleanup_before_await]) and the same-device copy ([Init_from_device_before_await]) stay
   out, being reachable only through an injected failure or a second context lineage.

   Isolation, on [cc]: this domain installs a callback for the whole test and runs the scenario
   once; a spawned domain then runs it BEFORE installing anything (it must not reach this domain's
   callback), installs its own and waits; this domain runs the scenario again while the spawned
   domain's callback is installed (it must reach only its own); then the spawned domain runs its
   scenario under its own callback. Every callback tags what it sees with the domain whose turn it
   is. With a process-wide callback the spawned domain's installation would replace this domain's,
   so this domain's second run would land in the spawned domain's callback and the spawned domain's
   uninstalled run in this one's. The turns never overlap: what is under test is where the callback
   lives, which the turns expose exactly as well.

   Worker domains, on [multidev_cc]: the same scenario, from this domain, must show this domain's
   callback the same set of points as on [cc], where everything runs on the calling domain. A point
   moved into a task the scheduler enqueues (the kernel task [link] hands to [Task.enschedule], say)
   would fire on the worker domain, whose callback is the no-op, and drop out of the set; moving one
   into the routine-level schedule would not, since that wrapper only enqueues, on the caller.

   The [cc] leg names its backend through [Context.cpu ()], the [multidev_cc] leg through
   [Context.cpu ~threads:2 ()]. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module FI = Ir.Resource_fault_injection
module SC = Ir.Schedule_cache
open Verdict.Claims
module Cache_plant = Test_utils.Cache_plant

type turn = Main | Spawned

(* Whose turn it is. Written only by the domain about to run; the hand-offs below order every write
   before the other domain's reads. *)
let running : turn option Stdlib.Atomic.t = Stdlib.Atomic.make None

type event = { turn : turn option; point : FI.point }

let recorder events point = events := { turn = Stdlib.Atomic.get running; point } :: !events

let points events =
  List.map events ~f:(fun e -> e.point) |> List.dedup_and_sort ~compare:Poly.compare

let clean_dir dir =
  if Stdlib.Sys.file_exists dir && Stdlib.Sys.is_directory dir then (
    Array.iter (Stdlib.Sys.readdir dir) ~f:(fun name ->
        Stdlib.Sys.remove (Stdlib.Filename.concat dir name));
    Stdlib.Sys.rmdir dir)

let cache_dir = "autotune_cache_resource_fault_injection_domains"

type fixture = { comp : Ir.Assignments.comp; y : Tensor.t; w : Tensor.t; nd : Ir.Ndarray.t }

(* Built in this domain before any other domain exists: tensor construction is not under test. *)
let fixture name =
  let values = [| -2.; -1.; 3.; 4. |] in
  let x = TDSL.ndarray values ~label:[ name ^ "_x" ] ~output_dims:[ 4 ] () in
  let w = TDSL.ndarray values ~label:[ name ^ "_w" ] ~output_dims:[ 4 ] () in
  let%op y = relu x in
  let nd =
    Ir.Ndarray.init_array ~debug:(name ^ " upload") Ir.Ops.single ~dims:[| 4 |] ~padding:None
      ~f:(fun i -> values.(i.(0)))
  in
  { comp = Train.forward y; y; w; nd }

(* One uninjected scenario in the calling domain, as [who]'s turn, against contexts from [root]. *)
let scenario ?who ~root ~name f =
  Stdlib.Atomic.set running who;
  Exn.protect
    ~finally:(fun () -> Stdlib.Atomic.set running None)
    ~f:(fun () ->
      let ctx, routine = Context.compile (root ()) f.comp Ir.Indexing.Empty in
      let ctx = Context.run ctx routine in
      let readback = Context.get_values ctx f.y.Tensor.value in
      Context.release ctx;
      let uploaded = Context.from_host (root ()) f.w.Tensor.value f.nd in
      Context.release uploaded;
      (* One directory for every scenario: the turns never overlap, and each starts it fresh. *)
      clean_dir cache_dir;
      Stdlib.Sys.mkdir cache_dir 0o755;
      Exn.protect
        ~finally:(fun () -> clean_dir cache_dir)
        ~f:(fun () ->
          SC.store ~dir:cache_dir ~key:(Some name)
            (Cache_plant.entry ~backend:name ~source_digest:"rfi-domains" ~best_ms:1.
               ~baseline_ms:2. ());
          let found = SC.lookup ~dir:cache_dir ~key:(Some name) in
          (* A scenario that silently did nothing would make every claim below vacuous. *)
          Array.length readback = 4
          && Option.exists found ~f:(fun e -> String.equal e.SC.backend name)))

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

let cc () = Context.cpu ()

let () =
  let main_fixture = fixture "rfid_main" and spawned_fixture = fixture "rfid_spawned" in
  let multidev_fixture = fixture "rfid_multidev" in
  let main_events = ref [] and spawned_events = ref [] in
  let spawned_ready = signal () and spawned_turn = signal () in
  let ( first_ok,
        second_ok,
        (uninstalled_ok, spawned_ok),
        multidev_ok,
        multidev_events,
        multidev_backend ) =
    FI.with_callback (recorder main_events) ~f:(fun () ->
        let first_ok = scenario ~who:Main ~root:cc ~name:"rfid_main" main_fixture in
        let spawned =
          Stdlib.Domain.spawn (fun () ->
              (* Before installing anything: a fresh domain calls the no-op, not its parent's. *)
              let uninstalled_ok =
                scenario ~who:Spawned ~root:cc ~name:"rfid_spawned_bare" spawned_fixture
              in
              FI.with_callback (recorder spawned_events) ~f:(fun () ->
                  raise_signal spawned_ready;
                  await_signal spawned_turn;
                  ( uninstalled_ok,
                    scenario ~who:Spawned ~root:cc ~name:"rfid_spawned" spawned_fixture )))
        in
        (* The spawned domain's callback is installed from here until its own scenario is done. *)
        await_signal spawned_ready;
        let second_ok =
          Exn.protect
            ~f:(fun () -> scenario ~who:Main ~root:cc ~name:"rfid_main_again" main_fixture)
            ~finally:(fun () ->
              (* Even if this scenario fails, the spawned domain must be released to finish. *)
              raise_signal spawned_turn)
        in
        let spawned_results = Stdlib.Domain.join spawned in
        (* The worker-domain leg, under a callback of its own in this same domain. *)
        let multidev_events = ref [] in
        let multidev_backend = ref "" in
        let multidev_ok =
          FI.with_callback (recorder multidev_events) ~f:(fun () ->
              scenario ~who:Main
                ~root:(fun () ->
                  let ctx = Context.cpu ~threads:2 () in
                  multidev_backend := Context.backend_name ctx;
                  ctx)
                ~name:"rfid_multidev" multidev_fixture)
        in
        (first_ok, second_ok, spawned_results, multidev_ok, !multidev_events, !multidev_backend))
  in
  let main_events = !main_events and spawned_events = !spawned_events in
  Stdio.eprintf
    "(not part of the golden) main saw %d hits over %d points, spawned %d over %d; %s saw %d hits \
     over %d points\n"
    (List.length main_events)
    (List.length (points main_events))
    (List.length spawned_events)
    (List.length (points spawned_events))
    multidev_backend (List.length multidev_events)
    (List.length (points multidev_events));
  p "every scenario ran to its readback and cache lookup"
    (first_ok && second_ok && uninstalled_ok && spawned_ok && multidev_ok);
  p "the multidev leg ran on multidev_cc" (String.equal multidev_backend "multidev_cc");
  let own who (e : event) = Option.equal Poly.equal e.turn (Some who) in
  p_all "main domain: every hit its callback saw came from its own scenarios" main_events
    ~f:(own Main);
  p_all "spawned domain: every hit its callback saw came from its own scenario" spawned_events
    ~f:(own Spawned);
  p "main domain: its callback saw its scenarios' hits" (not (List.is_empty main_events));
  (* The same scenario reaches the same points wherever it runs: a spawned domain's callback is
     complete, not merely clean. *)
  p "spawned domain: its callback saw every point the main domain's did"
    (List.equal FI.equal_point (points spawned_events) (points main_events));
  p "multidev_cc: the calling domain's callback saw every point the cc scenario reaches"
    (List.equal FI.equal_point (points multidev_events) (points main_events))
