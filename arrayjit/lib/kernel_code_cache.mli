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

val resolve :
  ('p * string) t ->
  lock:Stdlib.Mutex.t ->
  keys:string array ->
  options:string list array ->
  symbols:string array ->
  ?always_lock:bool ->
  ?before_compile:(n_compiled:int -> n_groups:int -> unit) ->
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
    all-hit batch). *)

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
    character literal. If [bound] meets [free] -- a node declared under a name that is also a
    language, builtin or vendor name, whose other occurrences might be its free meaning, as with a
    macro, which no declaration shadows -- no [bound] name is renamed.

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
