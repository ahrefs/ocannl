(** Where a compile's host wall time goes (gh-ocannl-1314).

    A domain-local census of the phases of {!Backends}' compile and link, bracketed like the codegen
    censuses ({!C_syntax.with_census}): each phase is timed where it runs and added to the innermost
    enclosing {!with_census}, and a bracket's totals are added to the bracket around it when it
    closes, so an outer bracket counts what an inner one saw exactly once. Outside any bracket the
    recording functions do nothing but run their thunk.

    What each field covers:

    - [lower_s]: the assignments' lowering and its optimization ([Backends.lower_assignments]).
    - [transform_s]: the [?lowered_transform] (a search candidate's schedule, fission and the
      search's pre-codegen gate), or the default schedules when none is given.
    - [backend_compile_s]: the backend's [compile] / [compile_batch] call as a whole: rendering the
      kernels to source, then the vendor compiler on that source.
    - [device_s]: the share of [backend_compile_s] spent in the backend's [compile_source] callback:
      the vendor compiler (gcc for [cc], nvrtc, hiprtc) together with what the backend does around
      it -- writing debug artifacts, and on [cc] signing and loading the shared object. It is an
      upper bound on the vendor compiler's time, not that time alone. [None] when no compile in the
      bracket timed one: Metal compiles its library at link, so there [link_s] carries it.
    - [link_s]: [Backend.link] (module loading, buffer allocation, the per-kernel checks).
    - [kernels]: the kernels the compiles rendered; [kernels_compiled]: the ones the vendor compiler
      was given, counted just before it runs, so a compile it refused still counts (on Metal, inside
      link). They differ only on a backend that reuses compiled kernels ([reuse_capable]).
    - [kernels_reused]: the rendered kernels whose code an EARLIER compile published -- the hits of
      {!Kernel_code_cache.resolve}'s lookups, before this compile's own programs are compiled. Not
      the kernels that share a program with a twin of their own batch (equal keys compile once), so
      [kernels_compiled + kernels_reused <= kernels], and a cold compile reuses none even when it
      compiles fewer kernels than it rendered (gh-ocannl-1667: what tells a compile the cache served
      from the cold one that filled it).
    - [pipelines_created]: the per-kernel pipeline objects the links created rather than reused
      (Metal's compute pipeline states, gh-ocannl-1605), counted at the creation call; 0 on a
      backend that has no such object.
    - [reuse_capable]: whether a compile in the bracket went through a backend's kernel code cache
      ({!Kernel_code_cache}; HIP's, CUDA's, cc's, Metal's), whether or not the cache was enabled:
      with it disabled ([hip_kernel_cache_mib=0]) such a compile compiles every kernel, which is
      what a test gated on this flag must then see fail.

    A host wall-clock reading for diagnosis, not a timing claim. *)

type t = {
  lower_s : float;
  transform_s : float;
  backend_compile_s : float;
  device_s : float option;
  link_s : float;
  kernels : int;
  kernels_compiled : int;
  kernels_reused : int;
  pipelines_created : int;
  reuse_capable : bool;
}
[@@deriving sexp_of]

val empty : t

val codegen_s : t -> float
(** [backend_compile_s] less [device_s] (taken as 0 when [None]): rendering and the backend's own
    bookkeeping around the vendor compile. *)

val total_s : t -> float
(** The sum of the disjoint phases: [lower_s +. transform_s +. backend_compile_s +. link_s]. *)

val add : t -> t -> t
(** Field-wise sum; [device_s] is [None] only when both are, [reuse_capable] is the disjunction. *)

val with_census : (unit -> 'a) -> 'a * t
(** [with_census f] runs [f] with a fresh census and returns what it recorded. The enclosing census,
    if any, is restored and also receives these totals. When [f] raises, the enclosing census still
    receives them and the exception propagates. *)

val time_lower : (unit -> 'a) -> 'a
val time_transform : (unit -> 'a) -> 'a
val time_backend_compile : (unit -> 'a) -> 'a
val time_device : (unit -> 'a) -> 'a
val time_link : (unit -> 'a) -> 'a

val record_kernels : rendered:int -> compiled:int -> unit
(** Adds a compile's kernel counts. *)

val record_kernels_reused : int -> unit
(** Adds kernels a compile took from code an earlier compile published ([kernels_reused]). *)

val record_pipeline_created : unit -> unit
(** Counts one pipeline object created at link ([pipelines_created]). *)

val record_reuse_capable : unit -> unit
(** Marks the bracket as having compiled through a kernel code cache, enabled or not. *)
