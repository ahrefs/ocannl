(* gh-ocannl-1352: searches run one at a time per process. The compile path a search drives keeps
   process-wide state never made safe for two domains at once, so [Autotune.tune] started in one
   domain while another domain's search runs raises [Autotune.Concurrent_search] before it does
   anything; searches in different domains taking turns stay supported ([autotune_seam_domains]).

   The overlap is made, not hoped for: this domain's search blocks in its first candidate attempt
   ([Autotune.on_candidate_attempt], per domain) until the spawned domain has tried its own search
   and come back. With the refusal in place it comes back at once, refused; without it, the spawned
   search runs to completion inside this one's -- which is what the refusal claims fail on. The
   spawned search's arguments, its context included, are built here beforehand, so the spawned
   domain does nothing before the call but install its own counting seam, and the claims that the
   refused call called no seam and delivered no report say it did nothing after it either.

   This domain's search runs nested in its own [Autotune.exclusive_search] span, as
   [Train.tune_placements]' arms do, and that span exits by an exception: the spawned domain's later
   turn, a whole search admitted and completed, says both the nested and the outer release happened.

   Pinned to cc: both searches run on an explicit [Context.cpu ()]; the property is the search
   entry's, not a backend's. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

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

let tune ~report ctx comp =
  Autotune.with_uncontended_test_windows (fun () ->
      Autotune.tune ~search:true ~beam_width:1 ~rounds:0 ~repeats:1 ~timing:Autotune.Isolated
        ~cache_dir:"" ~log:false ~progress:false
        ~report:(fun r -> report := Some r)
        ctx comp Ir.Indexing.Empty)

let completed = function
  | Some { Autotune.outcome = Autotune.Searched; candidates_timed; _ } -> candidates_timed > 0
  | Some _ | None -> false

(* A one-way signal between the two domains. *)
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

type attempt = Refused of { holder : int; caller : int } | Admitted | Failed of string

exception Span_exit

let () =
  let comp_main = matmul ~n:16 ~modulus:7 and comp_spawned = matmul ~n:16 ~modulus:5 in
  let ctx_spawned = Context.cpu () in
  let main_id = (Stdlib.Domain.self () :> int) in
  let main_searching = signal () and spawned_tried = signal () and main_done = signal () in
  let spawned =
    Stdlib.Domain.spawn (fun () ->
        let self = (Stdlib.Domain.self () :> int) in
        let seam_calls = ref 0 and report = ref None in
        Autotune.Seam.set Autotune.on_candidate_attempt (fun _ -> Int.incr seam_calls);
        await_signal main_searching;
        let attempt =
          Exn.protect
            ~finally:(fun () -> raise_signal spawned_tried)
            ~f:(fun () ->
              match tune ~report ctx_spawned comp_spawned with
              | ctx, _ ->
                  Context.release ctx;
                  Admitted
              | exception Autotune.Concurrent_search { holder; caller } ->
                  Refused { holder; caller }
              | exception exn -> Failed (Exn.to_string exn))
        in
        let refused_seam_calls = !seam_calls and refused_report = Option.is_some !report in
        (* The later turn, once this domain's search has returned. *)
        await_signal main_done;
        let later_report = ref None in
        let later =
          match tune ~report:later_report (Context.cpu ()) comp_spawned with
          | ctx, _ ->
              Context.release ctx;
              Admitted
          | exception Autotune.Concurrent_search { holder; caller } -> Refused { holder; caller }
          | exception exn -> Failed (Exn.to_string exn)
        in
        (self, attempt, refused_seam_calls, refused_report, later, !later_report))
  in
  let main_report = ref None in
  let first_attempt = ref true in
  let old_attempt = Autotune.Seam.get Autotune.on_candidate_attempt in
  Exn.protect
    ~finally:(fun () ->
      Autotune.Seam.set Autotune.on_candidate_attempt old_attempt;
      (* Whatever happened here, the spawned domain must be released to finish. *)
      raise_signal main_searching;
      raise_signal main_done)
    ~f:(fun () ->
      ( Autotune.Seam.set Autotune.on_candidate_attempt @@ fun _label ->
        if !first_attempt then (
          first_attempt := false;
          raise_signal main_searching;
          await_signal spawned_tried) );
      try
        Autotune.exclusive_search (fun () ->
            let ctx, _ = tune ~report:main_report (Context.cpu ()) comp_main in
            Context.release ctx;
            raise Span_exit)
      with Span_exit -> ());
  let spawned_id, attempt, refused_seam_calls, refused_report, later, later_report =
    Stdlib.Domain.join spawned
  in
  let describe = function
    | Refused { holder; caller } -> Printf.sprintf "refused (holder %d, caller %d)" holder caller
    | Admitted -> "admitted"
    | Failed msg -> "failed: " ^ msg
  in
  Stdio.eprintf
    "(not part of the golden) main domain %d, spawned %d: overlapping attempt %s; later turn %s\n%!"
    main_id spawned_id (describe attempt) (describe later);
  p "this domain's search, nested in its own exclusive span, completed" (completed !main_report);
  p "the spawned domain's search, started during this domain's, is refused"
    (match attempt with Refused _ -> true | Admitted | Failed _ -> false);
  p "the refusal names the holding domain and the refused one"
    (match attempt with
    | Refused { holder; caller } -> holder = main_id && caller = spawned_id
    | Admitted | Failed _ -> false);
  p "the refused search called no seam" (refused_seam_calls = 0);
  p "the refused search delivered no report" (not refused_report);
  p "the spawned domain's later turn is admitted" (match later with Admitted -> true | _ -> false);
  p "the spawned domain's later search completed" (completed later_report)
