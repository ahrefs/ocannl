(** Backend-agnostic sighting table for page-locking recurring host arrays (gh-ocannl-1854).

    Tracks host arrays by their byte range from [addr] to [addr + bytes]:
    - Transfers completely contained inside an already-registered range return [true] immediately
      without re-registering (even when smaller than [min_bytes ()] or when [min_bytes () = 0]), so
      the caller still synchronises the stream on a sub-slice of a pinned parent.
    - Transfers that partially overlap an already-registered range first unregister the overlapping
      registration (marking it declined so alternating slice/parent uploads do not thrash), ensuring
      the driver never sees a copy spanning registered and unregistered host memory.
    - Otherwise, transfers smaller than [min_bytes ()], or when [min_bytes () <= 0], never sight or
      register new arrays.
    - The first eligible transfer sights the range and attaches a GC finaliser to its underlying
      [Bigarray.Genarray.t]. The second transfer of that live array calls [register ~addr ~bytes];
      if it returns [Some token], the entry becomes registered and [note_transfer] returns [true]
      until finalisation, overlap eviction, or {!unregister_all}. If [register] returns [None], the
      entry is marked declined and never retried.
    - Unregistration follows the fallible-cleanup rule: [live_bytes] and [unregistered] update only
      after [unregister] succeeds. If [unregister] raises inside GC finalisation, overlap eviction,
      or {!unregister_all}, the token, backing storage (when finalised), and live accounting stay
      queued for retry on the next {!note_transfer} or {!unregister_all}. The GC finaliser takes no
      lock, spins on nothing, and mutates no table. *)

type 'token t

type stats = {
  registered : int;
  unregistered : int;
  declined : int;
  unregister_failures : int;
  live_bytes : int;
}
[@@deriving sexp_of]

exception Overlap_eviction_failed of { addr : nativeint; bytes : int }
(** Raised by {!with_transfer} / {!note_transfer} when a transfer partially overlaps an active
    registration whose [unregister] failed, refusing to let the backend submit a mixed
    registered/unregistered copy. *)

val create :
  min_bytes:(unit -> int) ->
  register:(addr:nativeint -> bytes:int -> 'token option) ->
  unregister:('token -> unit) ->
  'token t

val with_transfer : 'token t -> Ndarray.t -> (registered:bool -> 'a) -> 'a
(** Records a host-to-device transfer of [nd] and executes the transfer callback while guarding its
    byte range against overlapping pin-state mutations, passing [~registered:true] iff [nd] is
    covered by a live registration for this transfer (including a registration whose rollback failed
    and remains pinned in the driver). When [min_bytes () <= 0] and no array is registered, takes a
    zero-lock fast path; otherwise briefly acquires the table mutex to check/update range coverage
    and register the active transfer range, releases the mutex before running the callback (so
    disjoint multi-device transfers run concurrently through copy and stream synchronisation), and
    serialises only transfers or teardowns whose byte ranges overlap. Raises
    {!Overlap_eviction_failed} if a partially overlapping registration could not be unregistered. *)

val note_transfer : 'token t -> Ndarray.t -> bool
(** Convenience wrapper returning [with_transfer t nd (fun ~registered -> registered)]. *)

val unregister_all : 'token t -> unit
(** Retries any pending failed unregistrations and unregisters every currently live registered
    entry. Entries whose [unregister] succeeds are marked dead and removed; entries whose
    [unregister] raises remain retryable with their live accounting intact. Safe to call at process
    exit after device streams are idle. *)

val stats : _ t -> stats
