(** What the device compiler made of a routine's kernels, read back from the compiled code
    (gh-ocannl-1530).

    The schedule layer predicts a kernel's static workgroup-shared allocation
    ([Schedule.workgroup_memory_bytes], the charge the seeding floors reproduce); this census holds
    the device compiler's own figure for the same kernel, so a test can relate the two. It is a
    domain-local, opt-in bracket, like {!Compile_cost.with_census}: a backend's link records one
    entry per kernel it links while a bracket is open on the linking domain, and nothing otherwise.
    It is opt-in, rather than a field of the routine beside its codegen censuses, because one
    backend's reading is not free: cudajit binds no function-attribute query, so the CUDA backend
    loads the kernel's PTX a second time to ask the driver, which an ordinary compile must not pay.

    Which backends answer: Metal ([staticThreadgroupMemoryLength] of the compute pipeline state it
    creates at link), CUDA ([CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES], when the driver library exports
    the entry points and runtime debugging is off: a debug link's JIT options are not reproduced by
    the second load) and HIP ([HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES]). The C backends declare no
    workgroup memory and record nothing, so an empty census is "no answer", never "zero bytes". A
    relink of retained code ([Context.relink_outcome]) outside the bracket records nothing. *)

type record = {
  kernel : string;  (** The kernel's name, as the backend links it. *)
  static_workgroup_bytes : int;
      (** The static workgroup-shared (threadgroup / [__shared__]) bytes the compiled kernel
          allocates, as the driver reports them. *)
}
[@@deriving sexp_of]

val with_census : (unit -> 'a) -> 'a * record list
(** [with_census f] runs [f] with the census armed on this domain and returns its result with the
    records of the kernels linked meanwhile, in link order. A nested bracket's records are added to
    the enclosing one's as well when it closes. *)

val armed : unit -> bool
(** Whether a bracket is open on this domain: a backend asks before a reading that costs. *)

val record : kernel:string -> static_workgroup_bytes:int -> unit
(** Adds a record to the innermost open bracket; a no-op outside any. *)
