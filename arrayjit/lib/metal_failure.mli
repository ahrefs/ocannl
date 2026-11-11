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

(** {2 Failed command buffers (gh-ocannl-2083, gh-ocannl-2089)} *)

type command_buffer_failure = {
  what : string;  (** The buffer's work, as a report names it (["kernel k"], ["a zero fill"]). *)
  runs_kernels : bool;
      (** Whether the buffer dispatched kernels: a kernel's own buffer or a fused segment batch, not
          a fill, copy, wait or signal. *)
  error : string option;  (** Metal's error description of the buffer, when it gave one. *)
  first_report : bool;
      (** Whether this raise is the failure's first. A device keeps a failure for the rest of the
          process and re-raises it on every later use; only the first raise comes from the wait
          covering the failed buffer, so only it says anything about the work being waited for. *)
}
[@@deriving sexp_of, equal]

exception Command_buffer_failed of command_buffer_failure
(** What [Metal_backend] raises for a command buffer it read in status [Error]: on the first wait
    that covers it, then again on every later use of the device. Printed as
    {!command_buffer_report}. *)

val command_buffer_report : command_buffer_failure -> string
(** ["Metal: the command buffer of <what> failed: <error>. The device takes no more work in this
     process."], the same for the first report and every later one. *)

module For_testing : sig
  val with_reap_error : (what:string -> string option option) -> f:(unit -> 'a) -> 'a
  (** Scoped, domain-local command-buffer observation seam for [Metal_backend.reap]. The callback
      sees the tracked work's name. [None] reads the real driver status; [Some error] substitutes
      status [Error] and its error text ([Some None] means no text). Only finite, normally submitted
      work is needed. The runner's resulting failure stays latched after this scope ends. Restores
      the prior callback on return or exception; does not alter Metal's actual command buffers. *)

  val error_for_reap : what:string -> string option option
  (** Backend-side read of the test seam; [None] outside a test scope. *)
end

val gpu_hang_stage : string
(** ["gpu_hang"]: the [Backend_rejected] stage of a kernel the GPU watchdog killed. *)

val reports_gpu_hang : string -> bool
(** Whether Metal's error description carries the driver's hang code
    ([kIOGPUCommandBufferCallbackErrorHang], observed on macOS 27.2). The fail-closed allowlist has
    that one token: the code, not the localized sentence; never elapsed time, which says nothing
    about a buffer still making progress (gh-ocannl-2082). *)

val classify_command_buffer_failure :
  backend:string -> Schedule_outcome.phase -> exn -> Schedule_outcome.classified_cause option
(** Metal's [classify_failure] (gh-ocannl-2089). A GPU hang is the candidate's decline: an
    [Expected] [Backend_rejected] at {!gpu_hang_stage} that may have written device buffers, the
    class CUDA gives [CUDA_ERROR_LAUNCH_TIMEOUT]. A search counts it and still ends: the effect
    poisons the timing lineage, and the device takes no more work. Answered only for the FIRST
    report of a buffer that ran kernels, raised at [Launch] or [Sync], whose error
    {!reports_gpu_hang}. Every other failure answers [None], so it stays unclassified ([Fatal] under
    the common policy) as before: a re-report (not the waited-for work's fault), a fill or copy, an
    ignored submission, a discarded victim, a missing or unrecognized error text, and the same text
    in a plain [Failure]. *)
