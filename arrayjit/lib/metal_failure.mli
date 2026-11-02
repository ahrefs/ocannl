(** What a failed Metal compile or pipeline creation reports, read off the text the [metal] binding
    raises it with ([Failure]): the classes the backend types, and the cause it types a compiler
    crash as.

    They live here, outside the [metal]-gated [Metal_backend], so they are tested wherever OCANNL
    builds, with no device ([test/operations/metal_failure_classes], gh-ocannl-1436): neither class
    can be provoked on demand -- the threadgroup overflow sits behind the schedule-level gate
    (gh-ocannl-1330), and the compiler crash is a defect of Apple's compiler on particular kernels.
*)

type pipeline_failure =
  | Threadgroup_overflow of { requested : int; limit : int }
      (** The driver refused a kernel whose threadgroup memory is over the device limit
          (gh-ocannl-1226), with its own byte counts. *)
  | Compiler_service_crash
      (** {!compiler_service_crash}: the compiler service died compiling the kernel. *)
  | Unrecognized  (** Anything else: nothing here establishes that it is the kernel's. *)
[@@deriving sexp_of, equal]

val compiler_service_crash : string -> bool
(** Whether a failure reports that Metal's compiler service ([MTLCompilerService], an XPC process)
    went away while it compiled the request: [XPC_ERROR_CONNECTION_INTERRUPTED] (gh-ocannl-1436).
    Metal relaunches the service and retries the request itself before it reports this ("This error
    occurred after multiple retries"), so a failure that reaches OCANNL survived those retries.
    Measured on an M4 Max (macOS 27.2): the unparallelized whole-routine [gpt2_mini_train] kernel
    aborted the service in AGX instruction emission ([llvm::report_fatal_error] under
    [MCObjectStreamer::emitInstruction]) on every one of four attempts, about 40 s each -- a
    compiler crash on that kernel, not an environment fault. [XPC_ERROR_CONNECTION_INVALID], a
    service that could not be reached at all, is not this class: it fails every kernel alike. *)

val classify_pipeline_failure : string -> pipeline_failure
(** The class of a failed [newComputePipelineStateWithFunction:]. *)

val compiler_crash_stage : string
(** ["compiler_crash"]: the [Backend_rejected] stage of a compiler-service crash, apart from
    ["compiler"] (the compiler rejected the generated source with a diagnostic) so the blocker
    census tells a vendor compiler defect from OCANNL's codegen. *)

val compiler_service_crash_cause :
  backend:string -> request:string -> kernel:string -> string -> Schedule_outcome.cause
(** [compiler_service_crash_cause ~backend ~request ~kernel report] types a compiler-service crash
    reported as [report] while compiling [request] (["the library"], ["the pipeline"]) for [kernel]:
    a [Compiler_bug] rejection at {!compiler_crash_stage}, a decline of that candidate or kernel.
    Its detail says whose compiler crashed and where the crash report is, not the generated source:
    several declined candidates must not each print a whole kernel. *)
