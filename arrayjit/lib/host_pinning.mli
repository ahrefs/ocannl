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

val create :
  min_bytes:(unit -> int) ->
  register:(addr:nativeint -> bytes:int -> 'token option) ->
  unregister:('token -> unit) ->
  'token t

val note_transfer : 'token t -> Ndarray.t -> bool
(** Records a host-to-device transfer of [nd]. Returns [true] iff [nd] is covered by a live
    registration for this transfer (or an overlapping registration whose unregistration failed). *)

val unregister_all : 'token t -> unit
(** Retries any pending failed unregistrations and unregisters every currently live registered
    entry. Entries whose [unregister] succeeds are marked dead and removed; entries whose
    [unregister] raises remain retryable with their live accounting intact. Safe to call at process
    exit after device streams are idle. *)

val stats : _ t -> stats
