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

val publish : 'v t -> keys:string list -> bytes:int -> 'v -> unit
(** [publish c ~keys ~bytes v] retains [v] under every key in [keys], as one group of [bytes] bytes,
    evicting the oldest groups while the retained total exceeds the capacity. A key already present
    is re-pointed to [v]; the older group then no longer owns it, and its eviction leaves it be. *)

type stats = { n_groups : int; n_entries : int; n_bytes : int; n_evicted_groups : int }

val stats : 'v t -> stats

val canonical_symbols : ?keep:string list -> string -> string
(** [canonical_symbols kernel_text] renames, in one kernel's C text, every identifier token the
    lowering mints fresh -- a loop or static-index symbol as {!Ir.Indexing.symbol_ident} spells it
    ([i] then digits) and a scope-local ([v] then digits, [_], the node's identifier) -- to
    [ocannl_k<n>] (a local keeping its [_<ident>] suffix), numbered by first occurrence. A bijection
    on those tokens, so the kernel's meaning is unchanged: they are the kernel's own loop variables,
    locals and parameters, and parameters are passed by position. String and character literals are
    left alone; comments are renamed like code.

    Why (gh-ocannl-1314): symbols and scope ids come from process-wide counters that every lowering
    and every loop-splitting schedule op advance, so two candidates of one search -- or two compiles
    of one computation -- render an unchanged kernel with other numbers, and a cache keyed on raw
    text would never hit. Only a whole token is renamed: an identifier that merely contains such a
    spelling keeps it, and costs a miss, never a wrong hit. *)
