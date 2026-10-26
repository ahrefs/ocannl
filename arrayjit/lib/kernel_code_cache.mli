(** A bounded, process-local cache of compiled kernel code, keyed by exact source identity
    (gh-ocannl-1314).

    A backend whose vendor compiler is costly per program (hiprtc: about 0.14-0.34 s of fixed cost
    per program on top of about 20 ms per kernel, measured on gfx1151 and gfx1102, and serialized
    within a process) compiles a batch's cache misses together in ONE program and publishes the
    result as a {e group}: the keys of the kernels it compiled, sharing that program's code. A later
    batch whose kernel has the same key reuses the code instead of recompiling it.

    Contracts:
    - A key is the caller's digest of everything the compiled code depends on (see {!key}); this
      module never derives one. A hit is therefore only as sound as the key, and two keys are only
      equal when their inputs are.
    - Only a successful compile publishes: a caller publishes after its compiler returned, so a
      failed program leaves no entry, and the next compile of the same kernels compiles afresh.
    - Bounded by bytes, evicted oldest group first. Eviction drops the cache's own reference and
      nothing else: a value already handed out stays valid for whoever holds it (a linked routine, a
      retained compiled routine). A group larger than the whole capacity is not retained.
    - A capacity of 0 disables the cache: {!find} answers [None] and {!publish} retains nothing.
    - Safe to share between domains (one mutex). *)

type 'v t

val create : capacity_bytes:(unit -> int) -> 'v t
(** [capacity_bytes] is read on every {!publish} and {!find}, so a configuration read lazily by the
    caller applies from first use. *)

val key : parts:string list -> string
(** The hex digest of [parts], each length-prefixed, so no two different lists collide by
    concatenation. *)

val enabled : 'v t -> bool
val find : 'v t -> string -> 'v option

val publish : 'v t -> entries:(string * 'v) list -> bytes:int -> unit
(** [publish c ~entries ~bytes] retains each value under its key, as one group of [bytes] (the
    caller's weight: compiled bytes where the backend can measure them), evicting the oldest groups
    while the retained total exceeds the capacity. The values of one group typically share one
    compiled program and differ in the symbol each kernel was compiled under. A key already present
    is re-pointed; the older group then no longer owns it, and its eviction leaves it be. *)

type stats = { n_groups : int; n_entries : int; n_bytes : int; n_evicted_groups : int }

val stats : 'v t -> stats

(** {2 Concurrent programs (gh-ocannl-1603)}

    A cold compile -- a search's first candidate, every kernel a miss -- compiles the whole batch as
    one program. Where the vendor compiler costs per kernel and runs in parallel across programs
    (nvrtc: a [gpt2_mini_train] step's 269 kernels took 4.73 s as one program, 2.14-2.18 s as 8
    programs compiled on 8 threads, and 7.22 s as 269 programs on 8 threads), {!resolve} can split a
    large group of misses into a few programs and compile them at once. *)

type split = {
  max_programs : int;  (** At most this many programs per option list; [<= 1] never splits. *)
  min_kernels : int;  (** At least this many kernels per program ([0] counts as [1]). *)
}

val programs_for : split -> int -> int
(** [programs_for split n] is how many programs [n] kernels of one option list compile as:
    [min max_programs (n / min_kernels)], at least 1, at most [n]. *)

val chunks : int -> int list -> int list list
(** [chunks k idxs] is [idxs] cut into [k] contiguous runs, balanced by count (the first [t * n / k]
    .. [(t + 1) * n / k] for each [t]), empty runs dropped. *)

val with_split : split -> (unit -> 'a) -> 'a
(** Runs [f] with [split] in place of whatever a backend's configuration would give
    ({!current_split_override}), for tests that compare a split compile with a whole one in one
    process. *)

val current_split_override : unit -> split option

val record_split_capable : unit -> unit
(** A backend that can split its compiles records each compile, whether or not it split -- with its
    kernel code cache disabled too -- so a test gates its split claims on {!split_capable_compiles}
    rather than on a backend name. *)

val split_capable_compiles : unit -> int

val programs_compiled_concurrently : unit -> int
(** Programs {!resolve} has compiled through its concurrent path in this process. *)

type ('p, 'w) concurrent = {
  split : split;
  prepare : group:int -> n_groups:int -> string list -> int list -> unit -> 'w;
      (** On the calling domain, in batch order, before any compile starts: whatever must be done
          there (artifacts, census records); it answers the compile itself, which runs on a domain
          of its own and must touch no domain-local or unsynchronized state. *)
  finish : group:int -> 'w -> 'p * int;
      (** On the calling domain, after every compile returned, in batch order, for each one that
          succeeded: the program and its weight, as [compile_group] answers them. *)
  phase : (unit -> unit) -> unit;
      (** Wraps the concurrent phase on the calling domain (a timer: the census is domain-local). *)
}

val resolve :
  ('p * string) t ->
  lock:Stdlib.Mutex.t ->
  keys:string array ->
  options:string list array ->
  symbols:string array ->
  ?always_lock:bool ->
  ?before_compile:(n_compiled:int -> n_groups:int -> unit) ->
  ?concurrent:('p, 'w) concurrent ->
  compile_group:(group:int -> n_groups:int -> string list -> int list -> 'p * int) ->
  unit ->
  ('p * string) array
(** One batch's kernels, by index: [keys.(i)] is kernel [i]'s key, [options.(i)] the compiler option
    list it compiles under, [symbols.(i)] the name the current rendering gives it. Answers, per
    kernel, the program holding its code and the symbol to look it up by -- the symbol it was
    compiled under, which a hit may have taken from another name (gh-ocannl-1525: the key does not
    depend on a segment kernel's positional index, {!kernel_token}).

    Every kernel is looked up first without [lock]; if any missed, the misses are looked up again
    under it (another domain may just have published them) and the remaining ones compiled there:
    one representative per distinct key (the first in batch order; the others share its program and
    symbol), grouped by option list in batch order, [compile_group ~group:k ~n_groups opts idxs]
    once per group, returning the program and its weight. A group is published only after its
    compile returned, so one that raises publishes nothing (earlier groups stay published) and the
    exception propagates; the next compile of those kernels compiles them afresh. [before_compile]
    runs under the lock ahead of the compiles, also when nothing is left to compile -- which happens
    under the lock only with [always_lock] (a caller that writes per-compile artifacts even for an
    all-hit batch).

    With [concurrent] (gh-ocannl-1603), each option list's representatives are cut into
    [programs_for concurrent.split] {!chunks}. When none is cut, the compile is the one above,
    through [compile_group]. When any is, every program of the batch -- numbered in batch order,
    [n_groups] counting them all -- is prepared, then compiled at once (the first on the calling
    domain, the others on domains of their own, every one joined before [resolve] returns or
    raises), still under [lock]; then each one that succeeded is finished and published in batch
    order, and the first failure in batch order propagates with its backtrace. So the partition
    changes neither a key nor a published value's meaning: a kernel's program holds its own code
    under its own option list, beside fewer companions. *)

val segment_kernel_name : routine:string -> int -> string
(** The name a fissioned routine's segment kernel [i] is given ([Backends]: [<routine>__seg<i>]). *)

val kernel_token : string -> string
(** What a kernel's own name contributes to its canonical text: the name without the positional
    index {!segment_kernel_name} appends ([<routine>__seg<i>] becomes [<routine>__seg]); any other
    name as it is. The routine's name stays in the key because it can decide whether the text
    compiles at all -- a routine named after a declaration the prelude's headers make is refused --
    and a hit must not answer for text that would not compile; the segment index never can, and a
    candidate that adds or removes a segment shifts every later one's (gh-ocannl-1525). *)

val canonical_kernel :
  ?kernel_name:string ->
  ?keep:string list ->
  ?bound:string list ->
  ?derivations:(string * string) list ->
  ?free:string list ->
  string ->
  string
(** [canonical_kernel kernel_text] is the text a kernel's key is taken over: one kernel's C text
    with the names it OWNS renamed to [ocannl_k<n>], numbered by first occurrence, and its own name
    (when [kernel_name] is given) replaced by its {!kernel_token}. The renamed tokens are:
    - a loop or static-index symbol as {!Ir.Indexing.symbol_ident} spells it ([i] then digits);
    - a scope-local ([v], digits, [_], anything), suffix included;
    - a name in [bound]: what the renderer declares for the kernel's tensor nodes (parameters,
      locals), bare or disambiguated;
    - a name derived from one of those by a [derivations] template [(prefix, suffix)] --
      [prefix ^ base ^ suffix] with [base] in [bound], a symbol, or digits -- the emitter's own
      derived locals ([C_syntax.minted_derivations]).

    Never renamed: a name in [keep] or [free]; an identifier after [.], [->] or [::] (a member or a
    qualified name); anything inside [[[...]]] (an attribute), a preprocessor line, or a string or
    character literal. If [bound] meets [free], or holds a name in the implementation's reserved
    namespace ([__x], [_X]) -- a node declared under a name that is also a language, builtin, vendor
    or predefined-macro name, whose other occurrences might be its free meaning, as with a macro,
    which no declaration shadows -- no [bound] name is renamed: such a kernel may not compile at
    all, and its key must not reach the code of a twin that did.

    Why this is sound: every renamed name is bound in the kernel (declared there, and a declaration
    shadows any outer meaning of its name throughout its scope; parameters are passed by position;
    the kernel is looked up through the symbol the cache stores), and distinct tokens get distinct
    names, so two kernels whose canonical texts are equal are equal up to a renaming of their own
    bindings, which neither their meaning nor their machine code depends on. Comments are renamed
    like code.

    Why (gh-ocannl-1314, gh-ocannl-1525): symbols, scope ids and minted tile nodes come from
    process-wide counters, node names are disambiguated over the whole batch ([n<id>_<label>]
    whenever a label repeats), and segment kernels are named by position, so two candidates of one
    search render an unchanged kernel with other names. A difference in anything else costs a miss,
    never a wrong hit. *)
