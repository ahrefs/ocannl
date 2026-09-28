open Base
open Ir.Schedule_outcome

let no_backend_classification _phase _exn = None

let expect_classified = function
  | Error (Classified classified) -> classified
  | Ok _ | Error (Fatal _) -> failwith "expected a classified failure"

let expect_fatal = function
  | Error (Fatal fatal) -> fatal
  | Ok _ | Error (Classified _) -> failwith "expected a fatal failure"

let () =
  let illegal_1 = Illegal_schedule { check = "Schedule.apply"; detail = "missing loop i" } in
  let illegal_2 = Illegal_schedule { check = "Schedule.apply"; detail = "missing loop j" } in
  assert (equal_rejection_key (key_of_cause illegal_1) (key_of_cause illegal_2));
  let resource_1 =
    Resource_exceeded
      {
        resource = Workgroup_threads;
        requested = 1_024;
        limit = Some 512;
        detail = "too many threads";
      }
  in
  let resource_2 =
    Resource_exceeded
      {
        resource = Workgroup_threads;
        requested = 2_048;
        limit = Some 1_024;
        detail = "still too many threads";
      }
  in
  assert (equal_rejection_key (key_of_cause resource_1) (key_of_cause resource_2));
  let unsupported_1 = Unsupported { feature = "mma"; detail = "precision f64" } in
  let unsupported_2 = Unsupported { feature = "mma"; detail = "extent 7" } in
  assert (equal_rejection_key (key_of_cause unsupported_1) (key_of_cause unsupported_2));
  let backend_rejection ?(backend = "cc") ?(stage = "compiler") ?(severity = Compiler_bug) detail =
    Backend_rejected { backend; stage; severity; detail }
  in
  let backend_key = key_of_cause (backend_rejection "diagnostic one") in
  assert (equal_rejection_key backend_key (key_of_cause (backend_rejection "diagnostic two")));
  assert (
    not
      (equal_rejection_key backend_key
         (key_of_cause (backend_rejection ~backend:"cuda" "diagnostic one"))));
  assert (
    not
      (equal_rejection_key backend_key
         (key_of_cause (backend_rejection ~stage:"linker" "diagnostic one"))));
  assert (
    not
      (equal_rejection_key backend_key
         (key_of_cause (backend_rejection ~severity:Expected "diagnostic one"))));
  let unclassified_1 =
    Unclassified { phase = Transform; exn_constructor = "Failure"; detail = "one" }
  in
  let unclassified_2 =
    Unclassified { phase = Transform; exn_constructor = "Failure"; detail = "two" }
  in
  assert (equal_rejection_key (key_of_cause unclassified_1) (key_of_cause unclassified_2));
  assert (
    not
      (equal_rejection_key (key_of_cause unclassified_1)
         (key_of_cause
            (Unclassified { phase = Backend_compile; exn_constructor = "Failure"; detail = "one" }))));
  let typed =
    protect ~strict:true ~classify_backend:no_backend_classification ~provenance:Candidate
      ~phase:Transform (fun () -> raise (Cause_at (Transform, illegal_1)))
    |> expect_classified
  in
  assert (equal_cause typed.cause illegal_1);
  (* gh-ocannl-1077: typed, yet fatal. A kernel the host loader refuses is an OCANNL link bug that a
     search must not absorb as a decline, under any provenance or strictness; the fatal failure
     keeps its cause and renders it into the public exception. *)
  let dlopen_rejection = backend_rejection ~stage:dlopen_stage "undefined symbol: sym" in
  List.iter [ Candidate; Cache_replay; Advisory; User_schedule ] ~f:(fun provenance ->
      List.iter [ true; false ] ~f:(fun strict ->
          let fatal =
            protect ~strict ~classify_backend:no_backend_classification ~provenance ~phase:Transform
              (fun () -> raise (Cause_at (Backend_link, dlopen_rejection)))
            |> expect_fatal
          in
          assert (equal_phase fatal.phase Backend_link);
          assert (Option.equal equal_cause fatal.cause (Some dlopen_rejection));
          assert (
            match fatal.exn with
            | Invalid_argument detail -> String.equal detail "undefined symbol: sym"
            | _ -> false)));
  (* The escalation is the loader's stage at [Backend_link], not every link-time rejection: the
     compiler stage at link, and the loader stage raised at another phase, stay declines. *)
  let link_compiler_rejection = backend_rejection "declined at link" in
  let contained_at_link =
    protect ~strict:true ~classify_backend:no_backend_classification ~provenance:Candidate
      ~phase:Transform (fun () -> raise (Cause_at (Backend_link, link_compiler_rejection)))
    |> expect_classified
  in
  assert (equal_cause contained_at_link.cause link_compiler_rejection);
  let contained_elsewhere =
    protect ~strict:true ~classify_backend:no_backend_classification ~provenance:Candidate
      ~phase:Transform (fun () -> raise (Cause_at (Backend_compile, dlopen_rejection)))
    |> expect_classified
  in
  assert (equal_cause contained_elsewhere.cause dlopen_rejection);
  let strict_unknown =
    protect ~strict:true ~classify_backend:no_backend_classification ~provenance:Candidate
      ~phase:Backend_compile (fun () -> failwith "compiler vanished")
    |> expect_fatal
  in
  assert (equal_phase strict_unknown.phase Backend_compile);
  let permissive_unknown =
    protect ~strict:false ~classify_backend:no_backend_classification ~provenance:Candidate
      ~phase:Backend_compile (fun () -> failwith "compiler vanished")
    |> expect_classified
  in
  (match permissive_unknown.cause with
  | Unclassified { phase = Backend_compile; exn_constructor; _ } ->
      assert (String.is_suffix exn_constructor ~suffix:"Failure")
  | _ -> failwith "expected an unclassified compile failure");
  ignore
    (protect ~strict:false ~classify_backend:no_backend_classification ~provenance:Candidate
       ~phase:Launch (fun () -> failwith "launch failed")
    |> expect_fatal);
  ignore
    (protect ~strict:false ~classify_backend:no_backend_classification ~provenance:User_schedule
       ~phase:Transform (fun () -> failwith "user transform failed")
    |> expect_fatal);
  ignore
    (protect ~strict:false ~classify_backend:no_backend_classification ~provenance:Candidate
       ~phase:Transform (fun () -> assert false)
    |> expect_fatal);
  let cached_assert =
    protect ~strict:true ~classify_backend:no_backend_classification ~provenance:Cache_replay
      ~phase:Transform (fun () -> assert false)
    |> expect_classified
  in
  (match cached_assert.cause with
  | Unclassified { phase = Transform; _ } -> ()
  | _ -> failwith "expected an unclassified cache-replay assertion");
  let compiler_rejection =
    {
      phase = Backend_compile;
      cause =
        Backend_rejected
          { backend = "test"; stage = "compiler"; severity = Expected; detail = "rejected" };
      execution_effect = No_device_writes;
    }
  in
  let classified =
    protect ~strict:true
      ~classify_backend:(fun phase _exn ->
        Option.some_if (equal_phase phase Backend_compile) compiler_rejection)
      ~provenance:Candidate ~phase:Backend_compile
      (fun () -> failwith "backend exception")
    |> expect_classified
  in
  assert (equal_classified_cause classified compiler_rejection);
  (* A post-link resource rejection (Metal's per-pipeline maxTotalThreadsPerThreadgroup and static
     threadgroup-memory checks) must be BOTH: an ordinary decline for a tuner candidate, and the
     unchanged public [Utils.User_error] for a hand-written schedule. Untyped it would be neither —
     strict classification makes an unrecognized link failure fatal, so one register-heavy candidate
     would end the search. *)
  let link_resource =
    Resource_exceeded
      {
        resource = Workgroup_threads;
        requested = 1_024;
        limit = Some 640;
        detail = "Metal: threadgroup size 1024 for k exceeds maxTotalThreadsPerThreadgroup 640";
      }
  in
  let link_declined =
    protect ~strict:true ~classify_backend:no_backend_classification ~provenance:Candidate
      ~phase:Transform (fun () ->
        tag Backend_link (fun () -> raise (Cause_at (Backend_link, link_resource))))
    |> expect_classified
  in
  assert (equal_cause link_declined.cause link_resource);
  assert (equal_execution_effect link_declined.execution_effect No_device_writes);
  (* The reported phase is where it was raised, not where [protect] was installed. *)
  assert (equal_phase link_declined.phase Backend_link);
  assert (
    equal_rejection_key (key_of_cause link_declined.cause) (Resource_exceeded_key Workgroup_threads));
  (match raise_failure (Classified link_declined) with
  | _ -> failwith "expected the classified cause to be rendered"
  | exception Utils.User_error msg ->
      assert (String.is_substring msg ~substring:"maxTotalThreadsPerThreadgroup"));
  Stdlib.Printexc.record_backtrace true;
  let tagged =
    protect ~strict:true ~classify_backend:no_backend_classification ~provenance:Candidate
      ~phase:Transform (fun () -> tag Backend_link (fun () -> failwith "link failed"))
    |> expect_fatal
  in
  assert (equal_phase tagged.phase Backend_link);
  assert (Stdlib.Printexc.raw_backtrace_length tagged.backtrace > 0);
  (* gh-ocannl-564: the same untyped failure at [Preflight] instead. Nothing was dispatched there,
     so it is contained under [strict] and under the [Launch] boundary the tuner installs — where,
     tagged [Launch], it would be the fatal above and would condemn the lineage. *)
  let preflight_declined =
    protect ~strict:true ~classify_backend:no_backend_classification ~provenance:Candidate
      ~phase:Launch (fun () -> tag Preflight (fun () -> failwith "unexecuted dependencies"))
    |> expect_classified
  in
  assert (equal_phase preflight_declined.phase Preflight);
  assert (equal_execution_effect preflight_declined.execution_effect No_device_writes);
  assert (
    equal_rejection_key
      (key_of_cause preflight_declined.cause)
      (Unclassified_key (Preflight, "Failure")));
  (* And not by asking the backend: a classifier guessing [Writes_may_have_occurred] for an error it
     does not recognize must not escalate a failure that provably wrote nothing. *)
  let preflight_not_backend_judged =
    protect ~strict:true
      ~classify_backend:(fun phase _exn ->
        Some
          {
            phase;
            cause =
              Backend_rejected
                { backend = "test"; stage = "driver"; severity = Expected; detail = "guessed" };
            execution_effect = Writes_may_have_occurred;
          })
      ~provenance:Candidate ~phase:Launch
      (fun () -> tag Preflight (fun () -> failwith "unexecuted dependencies"))
    |> expect_classified
  in
  assert (equal_execution_effect preflight_not_backend_judged.execution_effect No_device_writes);
  (* The process-level and compiler-invariant classes stay fatal wherever they are raised. *)
  ignore
    (protect ~strict:true ~classify_backend:no_backend_classification ~provenance:Candidate
       ~phase:Launch (fun () -> tag Preflight (fun () -> assert false))
    |> expect_fatal)
