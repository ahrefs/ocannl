open Base

module Range_key = struct
  type t = nativeint * int [@@deriving compare, hash, sexp_of]
end

type keep_alive = Keep : _ -> keep_alive

type 'token state =
  | Seen
  | Registered of { token : 'token; bytes : int }
  | Unregistering
  | Retry_unregister of { token : 'token; bytes : int }
  | Declined
  | Dead

type 'token entry = {
  addr : nativeint;
  bytes : int;
  end_addr : nativeint;
  state : 'token state Atomic.t;
  (* Count of in-flight [with_transfer] calls currently relying on this registration. Checked by
     lock-free GC finalisation after claiming [Unregistering] (Dekker ordering paired with
     [acquire_covering_pinned_locked]) so finalizing a super-slice wrapper cannot unpin pages while
     a sub-slice transfer is still running. *)
  active_users : int Atomic.t;
  (* Populated only if GC finalisation ran while [unregister] was in flight or failed, keeping the
     underlying Bigarray alive until a retry succeeds so [caml_ba_finalize] never frees still-pinned
     pages. [None] during normal lifetime, so normal GC reclamation is unaffected. *)
  finalized_keep : keep_alive option Atomic.t;
}

type active_transfer = { xfer_addr : nativeint; xfer_end : nativeint }

type 'token t = {
  min_bytes : unit -> int;
  register : addr:nativeint -> bytes:int -> 'token option;
  unregister : 'token -> unit;
  lock : Stdlib.Mutex.t;
  cond : Stdlib.Condition.t;
  table : (Range_key.t, 'token entry) Hashtbl.t;
  (* Active [Registered] / [Retry_unregister] entries, guarded by [lock], for O(#registered) range
     containment and overlap checks. *)
  mutable active_pinned : 'token entry list;
  (* In-flight [with_transfer] byte ranges, guarded by [lock]. Transfers whose ranges are disjoint
     run their callbacks concurrently without holding [lock]; overlapping transfers and overlapping
     retry/teardown unregistrations wait on [cond]. *)
  mutable active_transfers : active_transfer list;
  retired : 'token entry list Atomic.t;
  retry_queue : 'token entry list Atomic.t;
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

exception Overlap_eviction_failed of { addr : nativeint; bytes : int }

let () =
  Stdlib.Printexc.register_printer (function
    | Overlap_eviction_failed { addr; bytes } ->
        Some
          (Printf.sprintf
             "Host_pinning.Overlap_eviction_failed: could not unregister an active host \
              registration overlapping [%s, +%d bytes); refusing mixed-range transfer"
             (Nativeint.Hex.to_string addr) bytes)
    | _ -> None)

let ule a b =
  Int.(Nativeint.compare Nativeint.(bit_xor a min_value) Nativeint.(bit_xor b min_value) <= 0)

let ult a b =
  Int.(Nativeint.compare Nativeint.(bit_xor a min_value) Nativeint.(bit_xor b min_value) < 0)

let contains ~outer_addr ~outer_end ~inner_addr ~inner_end =
  ule outer_addr inner_addr && ule inner_end outer_end

let overlaps ~a_addr ~a_end ~b_addr ~b_end = ult a_addr b_end && ult b_addr a_end

let create ~min_bytes ~register ~unregister =
  {
    min_bytes;
    register;
    unregister;
    lock = Stdlib.Mutex.create ();
    cond = Stdlib.Condition.create ();
    table = Hashtbl.create (module Range_key);
    active_pinned = [];
    active_transfers = [];
    retired = Atomic.make [];
    retry_queue = Atomic.make [];
    registered = Atomic.make 0;
    unregistered = Atomic.make 0;
    declined = Atomic.make 0;
    unregister_failures = Atomic.make 0;
    live_registered = Atomic.make 0;
    live_bytes = Atomic.make 0;
  }

let rec push_atomic list_atom x =
  let head = Atomic.get list_atom in
  if not (Atomic.compare_and_set list_atom head (x :: head)) then push_atomic list_atom x

let push_retired t e = push_atomic t.retired e
let push_retry t e = push_atomic t.retry_queue e

let complete_unregister_success t e ~bytes ~next_state =
  Atomic.decr t.live_registered;
  ignore (Atomic.fetch_and_add t.live_bytes (-bytes) : int);
  Atomic.incr t.unregistered;
  (* Publish [next_state] before reading [finalized_keep] (Dekker ordering paired with
     [retain_and_recheck]): if GC finalisation ran concurrently and stored [Some keep], either this
     exchange clears it and moves [Declined -> Dead], or [retain_and_recheck] observes
     [Dead]/[Declined] and clears [finalized_keep] itself. *)
  Atomic.set e.state next_state;
  let was_finalized = Option.is_some (Atomic.exchange e.finalized_keep None) in
  if phys_equal next_state Dead then push_retired t e
  else if was_finalized && Atomic.compare_and_set e.state Declined Dead then push_retired t e

let retain_and_recheck t e ~keep =
  Atomic.set e.finalized_keep (Some keep);
  match Atomic.get e.state with
  | Dead -> ignore (Atomic.exchange e.finalized_keep None : keep_alive option)
  | Declined ->
      ignore (Atomic.exchange e.finalized_keep None : keep_alive option);
      if Atomic.compare_and_set e.state Declined Dead then push_retired t e
  | Retry_unregister _ -> push_retry t e
  | Unregistering | Registered _ | Seen -> ()

(* Runs from GC finalisation: takes no lock, spins on nothing, and touches no table so an allocation
   inside [note_transfer] cannot deadlock. Captures [t] and [e], never the array itself. *)
let rec finalize_entry t e ~keep =
  match Atomic.get e.state with
  | Dead -> ()
  | Unregistering | Retry_unregister _ ->
      (* Unregistration is either in flight or queued for retry on another domain. Publish [keep]
         and recheck [e.state] so a concurrent recovery to [Dead]/[Declined] cannot leave [keep]
         permanently rooted in [finalized_keep]. *)
      retain_and_recheck t e ~keep
  | (Seen | Declined) as prev ->
      if Atomic.compare_and_set e.state prev Dead then push_retired t e
      else finalize_entry t e ~keep
  | Registered { token; bytes } as prev -> (
      if not (Atomic.compare_and_set e.state prev Unregistering) then finalize_entry t e ~keep
      else if Atomic.get e.active_users > 0 then (
        (* A transfer covered by [e] is currently in flight: keep the backing buffer pinned and
           defer unregistration until the transfer finishes. *)
        Atomic.set e.finalized_keep (Some keep);
        Atomic.set e.state (Retry_unregister { token; bytes });
        push_retry t e)
      else
        try
          t.unregister token;
          complete_unregister_success t e ~bytes ~next_state:Dead
        with _ ->
          Atomic.incr t.unregister_failures;
          (* Preserve [token], backing array ([keep]), and live accounting until a retry
             succeeds. *)
          Atomic.set e.finalized_keep (Some keep);
          Atomic.set e.state (Retry_unregister { token; bytes });
          push_retry t e)

let attempt_unregister_locked t e ~next_state =
  match Atomic.get e.state with
  | (Registered { token; bytes } | Retry_unregister { token; bytes }) as prev -> (
      if not (Atomic.compare_and_set e.state prev Unregistering) then false
      else
        try
          t.unregister token;
          complete_unregister_success t e ~bytes ~next_state;
          true
        with _ ->
          Atomic.incr t.unregister_failures;
          Atomic.set e.state (Retry_unregister { token; bytes });
          push_retry t e;
          false)
  | Dead | Declined | Seen -> true
  | Unregistering -> false

let drain_retired_locked t =
  let retired = Atomic.exchange t.retired [] in
  List.iter retired ~f:(fun e ->
      let key = (e.addr, e.bytes) in
      match Hashtbl.find t.table key with
      | Some cur when phys_equal cur e -> Hashtbl.remove t.table key
      | _ -> ())

let is_pinned_state = function
  | Registered _ | Retry_unregister _ | Unregistering -> true
  | Seen | Declined | Dead -> false

let prune_active_pinned_locked t =
  t.active_pinned <- List.filter t.active_pinned ~f:(fun e -> is_pinned_state (Atomic.get e.state))

let overlaps_active_transfer_locked t ~addr ~end_addr =
  List.exists t.active_transfers ~f:(fun x ->
      overlaps ~a_addr:x.xfer_addr ~a_end:x.xfer_end ~b_addr:addr ~b_end:end_addr)

let acquire_covering_pinned_locked t ~addr ~end_addr =
  List.find t.active_pinned ~f:(fun r ->
      if contains ~outer_addr:r.addr ~outer_end:r.end_addr ~inner_addr:addr ~inner_end:end_addr then (
        Atomic.incr r.active_users;
        match Atomic.get r.state with
        | Registered _ | Retry_unregister _ -> true
        | Unregistering | Seen | Declined | Dead ->
            Atomic.decr r.active_users;
            false)
      else false)

let is_covered_by_pinned_locked t ~addr ~end_addr =
  List.exists t.active_pinned ~f:(fun r ->
      match Atomic.get r.state with
      | Registered _ | Retry_unregister _ ->
          contains ~outer_addr:r.addr ~outer_end:r.end_addr ~inner_addr:addr ~inner_end:end_addr
      | _ -> false)

(* True if starting a transfer on [addr, end_addr) must wait for an in-flight transfer: either an
   active transfer overlaps [addr, end_addr) directly, or this transfer would evict an active
   registration [r] that is currently serving an active transfer elsewhere inside [r.addr,
   r.end_addr). Because a thread only waits before adding itself to [active_transfers], wait depth
   is bounded by 1 (deadlock-free). *)
let blocked_by_active_transfer_locked t ~addr ~end_addr =
  overlaps_active_transfer_locked t ~addr ~end_addr
  || (not (is_covered_by_pinned_locked t ~addr ~end_addr))
     && List.exists t.active_pinned ~f:(fun r ->
         is_pinned_state (Atomic.get r.state)
         && overlaps ~a_addr:r.addr ~a_end:r.end_addr ~b_addr:addr ~b_end:end_addr
         && overlaps_active_transfer_locked t ~addr:r.addr ~end_addr:r.end_addr)

let drain_retries_locked t =
  let queued = Atomic.exchange t.retry_queue [] in
  let seen = ref [] in
  List.iter queued ~f:(fun e ->
      if not (List.exists !seen ~f:(fun x -> phys_equal x e)) then (
        seen := e :: !seen;
        if overlaps_active_transfer_locked t ~addr:e.addr ~end_addr:e.end_addr then
          (* Defer unregistration while an overlapping transfer is in flight on another domain. *)
          push_retry t e
        else
          let next_state =
            if Option.is_some (Atomic.get e.finalized_keep) then Dead else Declined
          in
          ignore (attempt_unregister_locked t e ~next_state : bool)))

let sight_new_locked t ~addr ~bytes nd =
  let end_addr = Nativeint.(addr + of_int bytes) in
  let e =
    {
      addr;
      bytes;
      end_addr;
      state = Atomic.make Seen;
      active_users = Atomic.make 0;
      finalized_keep = Atomic.make None;
    }
  in
  Hashtbl.set t.table ~key:(addr, bytes) ~data:e;
  Ndarray.apply
    { f = (fun arr -> Stdlib.Gc.finalise (fun arr -> finalize_entry t e ~keep:(Keep arr)) arr) }
    nd;
  None

(* Returns [Some e] (with [e.active_users] incremented) when the transfer is covered by a pinned
   registration [e], or [None] when the transfer is pageable. *)
let decide_transfer_locked t ~addr ~bytes ~end_addr ~threshold nd =
  drain_retries_locked t;
  drain_retired_locked t;
  prune_active_pinned_locked t;
  (* 1. If an already-registered range completely covers [addr, end_addr), the copy is already
     pinned and needs no new registration (even for a sub-slice smaller than [threshold]). *)
  match acquire_covering_pinned_locked t ~addr ~end_addr with
  | Some _ as covered -> covered
  | None -> (
      (* 2. If any active registration partially overlaps [addr, end_addr), evict it first so the
         driver never sees a copy spanning registered and unregistered host memory. *)
      let still_overlapping =
        List.fold t.active_pinned ~init:false ~f:(fun acc r ->
            if overlaps ~a_addr:r.addr ~a_end:r.end_addr ~b_addr:addr ~b_end:end_addr then
              let evicted = attempt_unregister_locked t r ~next_state:Declined in
              acc || not evicted
            else acc)
      in
      drain_retired_locked t;
      prune_active_pinned_locked t;
      if still_overlapping then
        (* A partially overlapping registration could not be unpinned: refuse to let the backend
           submit a mixed registered/unregistered copy range. *)
        raise (Overlap_eviction_failed { addr; bytes })
      else if threshold <= 0 || bytes < threshold then None
      else
        let key = (addr, bytes) in
        match Hashtbl.find t.table key with
        | None -> sight_new_locked t ~addr ~bytes nd
        | Some e -> (
            match Atomic.get e.state with
            | Registered _ | Retry_unregister _ -> (
                Atomic.incr e.active_users;
                match Atomic.get e.state with
                | Registered _ | Retry_unregister _ -> Some e
                | Unregistering | Seen | Declined | Dead ->
                    Atomic.decr e.active_users;
                    None)
            | Declined -> None
            | Dead -> sight_new_locked t ~addr ~bytes nd
            | Unregistering -> raise (Overlap_eviction_failed { addr; bytes })
            | Seen -> (
                match t.register ~addr ~bytes with
                | None ->
                    ignore (Atomic.compare_and_set e.state Seen Declined : bool);
                    Atomic.incr t.declined;
                    None
                | Some token ->
                    Atomic.incr t.registered;
                    Atomic.incr e.active_users;
                    if Atomic.compare_and_set e.state Seen (Registered { token; bytes }) then (
                      Atomic.incr t.live_registered;
                      ignore (Atomic.fetch_and_add t.live_bytes bytes : int);
                      t.active_pinned <- e :: t.active_pinned;
                      ignore (Stdlib.Sys.opaque_identity nd);
                      Some e)
                    else (
                      Atomic.decr e.active_users;
                      (* A finalizer won the race and moved [e.state] to [Dead] while [t.register]
                         ran: undo the registration immediately; if undo fails, keep it retryable
                         with [nd] alive and honest live accounting. *)
                      try
                        t.unregister token;
                        Atomic.incr t.unregistered;
                        None
                      with _ ->
                        Atomic.incr t.unregister_failures;
                        Atomic.incr t.live_registered;
                        ignore (Atomic.fetch_and_add t.live_bytes bytes : int);
                        let retry_entry =
                          {
                            addr;
                            bytes;
                            end_addr;
                            state = Atomic.make (Retry_unregister { token; bytes });
                            active_users = Atomic.make 1;
                            finalized_keep = Atomic.make (Some (Keep nd));
                          }
                        in
                        t.active_pinned <- retry_entry :: t.active_pinned;
                        push_retry t retry_entry;
                        (* The driver still holds the pin until retry succeeds, so report pinned to
                           keep stream synchronisation safe. *)
                        Some retry_entry))))

let with_transfer t nd f =
  let bytes = Ndarray.size_in_bytes nd in
  let threshold = t.min_bytes () in
  if bytes <= 0 then f ~registered:false
  else if threshold <= 0 && Atomic.get t.live_registered = 0 then f ~registered:false
  else
    let addr = Ndarray.to_native nd in
    let end_addr = Nativeint.(addr + of_int bytes) in
    let xfer = { xfer_addr = addr; xfer_end = end_addr } in
    let pinned_entry =
      Stdlib.Mutex.protect t.lock (fun () ->
          while
            prune_active_pinned_locked t;
            blocked_by_active_transfer_locked t ~addr ~end_addr
          do
            Stdlib.Condition.wait t.cond t.lock
          done;
          let pinned_entry = decide_transfer_locked t ~addr ~bytes ~end_addr ~threshold nd in
          t.active_transfers <- xfer :: t.active_transfers;
          pinned_entry)
    in
    let registered = Option.is_some pinned_entry in
    Exn.protect
      ~finally:(fun () ->
        ignore (Stdlib.Sys.opaque_identity nd);
        Stdlib.Mutex.protect t.lock (fun () ->
            t.active_transfers <-
              List.filter t.active_transfers ~f:(fun x -> not (phys_equal x xfer));
            Option.iter pinned_entry ~f:(fun e -> Atomic.decr e.active_users);
            Stdlib.Condition.broadcast t.cond))
      ~f:(fun () -> f ~registered)

let note_transfer t nd = with_transfer t nd (fun ~registered -> registered)

let unregister_all t =
  Stdlib.Mutex.protect t.lock (fun () ->
      while not (List.is_empty t.active_transfers) do
        Stdlib.Condition.wait t.cond t.lock
      done;
      drain_retries_locked t;
      drain_retired_locked t;
      prune_active_pinned_locked t;
      let pinned = t.active_pinned in
      List.iter pinned ~f:(fun e -> ignore (attempt_unregister_locked t e ~next_state:Dead : bool));
      drain_retired_locked t;
      prune_active_pinned_locked t;
      (* Also clear non-pinned [Seen]/[Declined] table entries so exit leaves the table empty, while
         keeping any [Retry_unregister] entries whose unregistration still needs retry. *)
      Hashtbl.filter_inplace t.table ~f:(fun e -> is_pinned_state (Atomic.get e.state)))

let stats (t : _ t) =
  {
    registered = Atomic.get t.registered;
    unregistered = Atomic.get t.unregistered;
    declined = Atomic.get t.declined;
    unregister_failures = Atomic.get t.unregister_failures;
    live_bytes = Atomic.get t.live_bytes;
  }
