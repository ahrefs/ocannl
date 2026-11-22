(** Backend-agnostic sighting table for page-locking recurring host arrays (gh-ocannl-1854).

    Tracks host arrays by their base address:
    - Transfers smaller than [min_bytes ()], or when [min_bytes () <= 0], never sight or register
      new arrays.
    - The first eligible transfer sights the array and attaches a GC finaliser to its underlying
      [Bigarray.Genarray.t].
    - The second transfer of a still-live sighted array calls [register ~addr ~bytes]; if it returns
      [Some token], the entry becomes registered and [note_transfer] returns [true] on this and all
      subsequent transfers until finalisation or {!unregister_all}. If [register] returns [None],
      the array is marked declined and never retried.
    - Finalisation (or {!unregister_all}) atomically claims the token so [unregister] runs at most
      once, even when finalisation races with registration or teardown. The GC finaliser takes no
      lock and mutates no table, so it cannot deadlock if GC runs inside {!note_transfer}. *)

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
(** Records a host-to-device transfer of [nd]. Returns [true] iff [nd] is registered for this
    transfer (either newly registered on its second sighting or already registered from an earlier
    transfer). *)

val unregister_all : 'token t -> unit
(** Unregisters every currently live registered entry and clears the table. Safe to call at process
    exit after device streams are idle; any GC finaliser that runs afterward sees the entry as
    already dead and is a no-op. *)

val stats : _ t -> stats
