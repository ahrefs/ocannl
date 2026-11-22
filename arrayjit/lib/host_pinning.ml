open Base

type 'token state = Seen | Registered of { token : 'token; bytes : int } | Declined | Dead
type 'token entry = { addr : nativeint; state : 'token state Atomic.t }

type 'token t = {
  min_bytes : unit -> int;
  register : addr:nativeint -> bytes:int -> 'token option;
  unregister : 'token -> unit;
  lock : Stdlib.Mutex.t;
  table : (nativeint, 'token entry) Hashtbl.t;
  retired : 'token entry list Atomic.t;
  registered : int Atomic.t;
  unregistered : int Atomic.t;
  declined : int Atomic.t;
  unregister_failures : int Atomic.t;
  live_registered : int Atomic.t;
  live_bytes : int Atomic.t;
}

type stats = {
  registered : int;
  unregistered : int;
  declined : int;
  unregister_failures : int;
  live_bytes : int;
}
[@@deriving sexp_of]

let create ~min_bytes ~register ~unregister =
  {
    min_bytes;
    register;
    unregister;
    lock = Stdlib.Mutex.create ();
    table = Hashtbl.create (module Nativeint);
    retired = Atomic.make [];
    registered = Atomic.make 0;
    unregistered = Atomic.make 0;
    declined = Atomic.make 0;
    unregister_failures = Atomic.make 0;
    live_registered = Atomic.make 0;
    live_bytes = Atomic.make 0;
  }

let rec push_retired t e =
  let head = Atomic.get t.retired in
  if not (Atomic.compare_and_set t.retired head (e :: head)) then push_retired t e

let drain_retired_locked t =
  let retired = Atomic.exchange t.retired [] in
  List.iter retired ~f:(fun e ->
      match Hashtbl.find t.table e.addr with
      | Some cur when phys_equal cur e -> Hashtbl.remove t.table e.addr
      | _ -> ())

(* Runs from GC finalisation: must take no lock and touch no table so an allocation inside
   [note_transfer] cannot deadlock on [t.lock]. Captures [t] and [e], never the array itself. *)
let finalize_entry t e =
  match Atomic.exchange e.state Dead with
  | Registered { token; bytes } ->
      Atomic.decr t.live_registered;
      ignore (Atomic.fetch_and_add t.live_bytes (-bytes) : int);
      (try
         t.unregister token;
         Atomic.incr t.unregistered
       with _ -> Atomic.incr t.unregister_failures);
      push_retired t e
  | Seen | Declined -> push_retired t e
  | Dead -> ()

let sight_new_locked t ~addr nd =
  let e = { addr; state = Atomic.make Seen } in
  Hashtbl.set t.table ~key:addr ~data:e;
  Ndarray.apply { f = (fun arr -> Stdlib.Gc.finalise (fun _ -> finalize_entry t e) arr) } nd;
  false

let note_transfer t nd =
  let bytes = Ndarray.size_in_bytes nd in
  let threshold = t.min_bytes () in
  if threshold <= 0 || bytes < threshold then
    if Atomic.get t.live_registered = 0 then false
    else
      let addr = Ndarray.to_native nd in
      Stdlib.Mutex.protect t.lock (fun () ->
          drain_retired_locked t;
          match Hashtbl.find t.table addr with
          | Some e -> (
              match Atomic.get e.state with Registered _ -> true | Seen | Declined | Dead -> false)
          | None -> false)
  else
    let addr = Ndarray.to_native nd in
    Stdlib.Mutex.protect t.lock (fun () ->
        drain_retired_locked t;
        match Hashtbl.find t.table addr with
        | None -> sight_new_locked t ~addr nd
        | Some e -> (
            match Atomic.get e.state with
            | Registered _ -> true
            | Declined -> false
            | Dead ->
                (* Prior owner already finalized; treat as a fresh sighting for this new array. *)
                sight_new_locked t ~addr nd
            | Seen -> (
                match t.register ~addr ~bytes with
                | None ->
                    ignore (Atomic.compare_and_set e.state Seen Declined : bool);
                    Atomic.incr t.declined;
                    false
                | Some token ->
                    Atomic.incr t.registered;
                    if Atomic.compare_and_set e.state Seen (Registered { token; bytes }) then (
                      Atomic.incr t.live_registered;
                      ignore (Atomic.fetch_and_add t.live_bytes bytes : int);
                      ignore (Stdlib.Sys.opaque_identity nd);
                      true)
                    else (
                      (* A finalizer won the race and moved [e.state] to [Dead] while [t.register]
                         ran: undo the registration at once so unregister is called exactly once. *)
                      (try
                         t.unregister token;
                         Atomic.incr t.unregistered
                       with _ -> Atomic.incr t.unregister_failures);
                      false))))

let unregister_all t =
  Stdlib.Mutex.protect t.lock (fun () ->
      drain_retired_locked t;
      let entries = Hashtbl.data t.table in
      Hashtbl.clear t.table;
      List.iter entries ~f:(fun e ->
          match Atomic.exchange e.state Dead with
          | Registered { token; bytes } -> (
              Atomic.decr t.live_registered;
              ignore (Atomic.fetch_and_add t.live_bytes (-bytes) : int);
              try
                t.unregister token;
                Atomic.incr t.unregistered
              with _ -> Atomic.incr t.unregister_failures)
          | Seen | Declined | Dead -> ()))

let stats (t : _ t) =
  {
    registered = Atomic.get t.registered;
    unregistered = Atomic.get t.unregistered;
    declined = Atomic.get t.declined;
    unregister_failures = Atomic.get t.unregister_failures;
    live_bytes = Atomic.get t.live_bytes;
  }
