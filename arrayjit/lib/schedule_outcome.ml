open Base

type phase =
  | Transform
  | Hardware_limits
  | Backend_codegen
  | Backend_compile
  | Backend_link
  | Preflight
  | Launch
  | Sync
[@@deriving sexp_of, compare, equal]

type resource =
  | Workgroup_threads
  | Workgroup_memory
  | Thread_scratch
  | Workgroup_x_extent
  | Workgroup_y_extent
  | Workgroup_z_extent
  | Grid_y_extent
  | Grid_z_extent
[@@deriving sexp_of, compare, equal]

type severity = Expected | Compiler_bug [@@deriving sexp_of, compare, equal]

type execution_effect = No_device_writes | Writes_may_have_occurred
[@@deriving sexp_of, compare, equal]

type cause =
  | Illegal_schedule of { check : string; detail : string }
  | Unsupported of { feature : string; detail : string }
  | Resource_exceeded of {
      resource : resource;
      requested : int;
      limit : int option;
      detail : string;
    }
  | Backend_rejected of { backend : string; stage : string; severity : severity; detail : string }
  | Unclassified of { phase : phase; exn_constructor : string; detail : string }
  | Seed_evicted of { family : string; detail : string }
  | Not_dispatched of { origin : string; detail : string }
[@@deriving sexp_of, equal]

type rejection_key =
  | Illegal_schedule_key of string
  | Unsupported_key of string
  | Resource_exceeded_key of resource
  | Backend_rejected_key of string * string * severity
  | Unclassified_key of phase * string
  | Seed_evicted_key of string
  | Not_dispatched_key of string
[@@deriving sexp_of, compare, equal]

let key_of_cause = function
  | Illegal_schedule { check; _ } -> Illegal_schedule_key check
  | Unsupported { feature; _ } -> Unsupported_key feature
  | Resource_exceeded { resource; _ } -> Resource_exceeded_key resource
  | Backend_rejected { backend; stage; severity; _ } ->
      Backend_rejected_key (backend, stage, severity)
  | Unclassified { phase; exn_constructor; _ } -> Unclassified_key (phase, exn_constructor)
  | Seed_evicted { family; _ } -> Seed_evicted_key family
  | Not_dispatched { origin; _ } -> Not_dispatched_key origin

let dlopen_stage = "dlopen"
let artifact_missing_stage = "artifact_missing"
let codesign_stage = "codesign"

let uncontainable phase cause =
  match (phase, cause) with
  | Backend_link, Backend_rejected { stage; _ } ->
      List.mem [ dlopen_stage; artifact_missing_stage; codesign_stage ] stage ~equal:String.equal
  | _ -> false

type fatal = {
  exn : exn;
  backtrace : Stdlib.Printexc.raw_backtrace;
  phase : phase;
  candidate : string option;
  cause : cause option;
}

type classified_cause = { phase : phase; cause : cause; execution_effect : execution_effect }
[@@deriving sexp_of, equal]

type failure = Classified of classified_cause | Fatal of fatal
type 'a outcome = ('a, failure) Result.t

exception Cause_at of phase * cause
exception Raised_at of phase * exn * Stdlib.Printexc.raw_backtrace

type provenance = Candidate | Cache_replay | Advisory | User_schedule
[@@deriving sexp_of, compare, equal]

let strict_failure_classification =
  lazy (Utils.get_global_flag ~default:true ~arg_name:"strict_failure_classification")

let is_oom_or_interrupt = function Out_of_memory | Stdlib.Sys.Break -> true | _ -> false
let is_assert_or_stack = function Assert_failure _ | Stack_overflow -> true | _ -> false

let is_compile_side = function
  | Transform | Hardware_limits | Backend_codegen | Backend_compile | Backend_link -> true
  | Preflight | Launch | Sync -> false

(* gh-ocannl-564: this phase precedes the dispatch, so its failures prove the device did nothing —
   no partial write to bound, no lineage to condemn. The one phase always containable, strict or
   not. *)
let is_pre_dispatch = function
  | Preflight -> true
  | Transform | Hardware_limits | Backend_codegen | Backend_compile | Backend_link | Launch | Sync
    ->
      false

let unclassified phase exn =
  {
    phase;
    cause =
      Unclassified
        { phase; exn_constructor = Stdlib.Printexc.exn_slot_name exn; detail = Exn.to_string exn };
    execution_effect = No_device_writes;
  }

let classify_raw ~strict ~classify_backend ~provenance ~phase ~candidate exn backtrace =
  let fatal () = Fatal { exn; backtrace; phase; candidate; cause = None } in
  if is_oom_or_interrupt exn then fatal ()
  else if is_assert_or_stack exn && not (equal_provenance provenance Cache_replay) then fatal ()
  else if is_pre_dispatch phase then
    (* Not routed through the backend (gh-ocannl-564): this validation is host-side and precedes
       every backend call, so a classifier answering [Writes_may_have_occurred] for an error it does
       not recognize would escalate a failure that provably wrote nothing. The [Fatal] default the
       other phases take is what condemned the lineage for a fixable mistake wherever
       [classify_failure] answers [None] — every C backend. *)
    Classified (unclassified phase exn)
  else
    match classify_backend phase exn with
    (* The backend is handed the phase, so the phase it reports back is not independent evidence:
       pin it to where the failure was actually raised. *)
    | Some classified -> Classified { classified with phase }
    | None ->
        if equal_provenance provenance Cache_replay && is_assert_or_stack exn then
          Classified (unclassified phase exn)
        else if
          (not strict) && is_compile_side phase && not (equal_provenance provenance User_schedule)
        then Classified (unclassified phase exn)
        else fatal ()

let detail_of_cause = function
  | Illegal_schedule { detail; _ }
  | Unsupported { detail; _ }
  | Resource_exceeded { detail; _ }
  | Backend_rejected { detail; _ }
  | Unclassified { detail; _ }
  | Seed_evicted { detail; _ }
  | Not_dispatched { detail; _ } ->
      detail

let exception_of_cause cause =
  match cause with
  | Resource_exceeded _ -> Utils.User_error (detail_of_cause cause)
  | Illegal_schedule _ | Unsupported _ | Backend_rejected _ | Unclassified _ | Seed_evicted _
  | Not_dispatched _ ->
      Invalid_argument (detail_of_cause cause)

let protect ?strict ~classify_backend ~provenance ~phase ?candidate f =
  let strict = Option.value strict ~default:(Lazy.force strict_failure_classification) in
  match f () with
  | result -> Ok result
  | exception Cause_at (cause_phase, cause) ->
      if uncontainable cause_phase cause then
        (* Rendered into the public exception, as {!fatal_of_classified} does, but with the raise
           site's backtrace: nothing contained it first. *)
        let backtrace = Stdlib.Printexc.get_raw_backtrace () in
        Error
          (Fatal
             {
               exn = exception_of_cause cause;
               backtrace;
               phase = cause_phase;
               candidate;
               cause = Some cause;
             })
      else Error (Classified { phase = cause_phase; cause; execution_effect = No_device_writes })
  | exception Raised_at (raised_phase, exn, backtrace) ->
      Error
        (classify_raw ~strict ~classify_backend ~provenance ~phase:raised_phase ~candidate exn
           backtrace)
  | exception exn ->
      let backtrace = Stdlib.Printexc.get_raw_backtrace () in
      Error (classify_raw ~strict ~classify_backend ~provenance ~phase ~candidate exn backtrace)

let tag phase f =
  match f () with
  | result -> result
  | exception (Cause_at _ as exn) ->
      let backtrace = Stdlib.Printexc.get_raw_backtrace () in
      Stdlib.Printexc.raise_with_backtrace exn backtrace
  | exception (Raised_at _ as exn) ->
      let backtrace = Stdlib.Printexc.get_raw_backtrace () in
      Stdlib.Printexc.raise_with_backtrace exn backtrace
  | exception exn ->
      let backtrace = Stdlib.Printexc.get_raw_backtrace () in
      Stdlib.Printexc.raise_with_backtrace (Raised_at (phase, exn, backtrace)) backtrace

let raise_cause cause = raise (exception_of_cause cause)

let fatal_of_classified ?candidate (classified : classified_cause) =
  let exn = exception_of_cause classified.cause in
  let backtrace = Stdlib.Printexc.get_callstack 16 in
  { exn; backtrace; phase = classified.phase; candidate; cause = Some classified.cause }

let raise_failure = function
  | Classified { cause; _ } -> raise_cause cause
  | Fatal { exn; backtrace; _ } -> Stdlib.Printexc.raise_with_backtrace exn backtrace
