open Base

(* The live table mirrors the backends' private pool tables, keyed the same way, because the shared
   allocator seam is the only place that knows a pool's byte size AND is backend-independent: the
   per-backend [Slab] tables are private by design (the concrete pointer type never appears in a
   shared signature), and [free_pool] is passed only a [pool_id].

   A mutex, like {!Backend_impl.Make_slab}'s [pools_mutex] and for the same reason: with the
   [Multidev] scheduler the allocator sites run on worker domains concurrently with link-time
   allocation on the main domain, and a Base hashtable is not domain-safe. *)
let live : (int * int, int * bool) Hashtbl.Poly.t = Hashtbl.Poly.create ()
let live_mutex = Stdlib.Mutex.create ()
let with_live f = Stdlib.Mutex.protect live_mutex f
let working_pools_allocated : Utils.atomic_int = Atomic.make 0
let constant_pools_allocated : Utils.atomic_int = Atomic.make 0
let pools_freed : Utils.atomic_int = Atomic.make 0
let contexts_created : Utils.atomic_int = Atomic.make 0
let contexts_released : Utils.atomic_int = Atomic.make 0
let modules_loaded : Utils.atomic_int = Atomic.make 0
let modules_unloaded : Utils.atomic_int = Atomic.make 0

(* The live pool bytes, and their high-water mark since the last {!reset_peak} (gh-ocannl-1006).
   Both guarded by [live_mutex] like the table they summarize.

   [live_bytes] is maintained INCREMENTALLY, by the delta against the entry a [record_pool]
   replaces. Folding the table instead would be one fewer thing to keep true, but it makes each
   allocation O(live pools) under the lock -- quadratic over a process that allocates many of them,
   and with the [Multidev] scheduler it serializes the worker domains behind a scan that grows as
   they work. The invariant that buys back the simplicity is checked rather than argued:
   [test/operations/alloc_census_peak] asserts this total against the fold [snapshot] reports as
   [live_pool_bytes] after a mixed sequence of records, in-place growths and frees. *)
let live_bytes = ref 0
let peak_bytes = ref 0

let record_pool ~device_id ~pool_id ~constant ~size_in_bytes =
  if constant then ignore (Atomic.fetch_and_add constant_pools_allocated 1 : int)
  else ignore (Atomic.fetch_and_add working_pools_allocated 1 : int);
  with_live (fun () ->
      let key = (device_id, pool_id) in
      (* Replacing an entry is a pool GROWN IN PLACE, so what it contributes is the difference, not
         the whole new size: the shared seam re-records the merge slab at each new capacity. *)
      let replaced = match Hashtbl.find live key with Some (bytes, _) -> bytes | None -> 0 in
      Hashtbl.set live ~key ~data:(size_in_bytes, constant);
      live_bytes := !live_bytes + size_in_bytes - replaced;
      (* Only allocation can raise the mark, so this is the one place it moves up: a free lowers the
         live bytes and must leave the peak where it was, which is the whole difference between a
         high-water counter and the sampled gauges the backends expose. *)
      if !live_bytes > !peak_bytes then peak_bytes := !live_bytes)

let forget_pool ~device_id ~pool_id =
  with_live (fun () ->
      let key = (device_id, pool_id) in
      match Hashtbl.find live key with
      | None -> ()
      | Some (bytes, _) ->
          Hashtbl.remove live key;
          live_bytes := !live_bytes - bytes;
          ignore (Atomic.fetch_and_add pools_freed 1 : int))

let reset_peak () = with_live (fun () -> peak_bytes := !live_bytes)
let count_context_created () = ignore (Atomic.fetch_and_add contexts_created 1 : int)
let count_context_released () = ignore (Atomic.fetch_and_add contexts_released 1 : int)
let count_module_loaded () = ignore (Atomic.fetch_and_add modules_loaded 1 : int)
let count_module_unloaded () = ignore (Atomic.fetch_and_add modules_unloaded 1 : int)

type t = {
  live_working_pools : int;
  live_working_bytes : int;
  live_constant_pools : int;
  live_constant_bytes : int;
  working_pools_allocated : int;
  constant_pools_allocated : int;
  pools_freed : int;
  contexts_created : int;
  contexts_released : int;
  modules_loaded : int;
  modules_unloaded : int;
  peak_pool_bytes : int;
}
[@@deriving sexp_of, equal]

let snapshot () =
  let live_working_pools, live_working_bytes, live_constant_pools, live_constant_bytes, peak =
    with_live (fun () ->
        let wp, wb, cp, cb =
          Hashtbl.fold live ~init:(0, 0, 0, 0)
            ~f:(fun ~key:_ ~data:(bytes, constant) (wp, wb, cp, cb) ->
              if constant then (wp, wb, cp + 1, cb + bytes) else (wp + 1, wb + bytes, cp, cb))
        in
        (wp, wb, cp, cb, !peak_bytes))
  in
  {
    live_working_pools;
    live_working_bytes;
    live_constant_pools;
    live_constant_bytes;
    working_pools_allocated = Atomic.get working_pools_allocated;
    constant_pools_allocated = Atomic.get constant_pools_allocated;
    pools_freed = Atomic.get pools_freed;
    contexts_created = Atomic.get contexts_created;
    contexts_released = Atomic.get contexts_released;
    modules_loaded = Atomic.get modules_loaded;
    modules_unloaded = Atomic.get modules_unloaded;
    peak_pool_bytes = peak;
  }

let live_pools c = c.live_working_pools + c.live_constant_pools
let live_pool_bytes c = c.live_working_bytes + c.live_constant_bytes
let unreleased_contexts c = c.contexts_created - c.contexts_released
let live_modules c = c.modules_loaded - c.modules_unloaded
let mib bytes = Float.of_int bytes /. 1048576.

let to_string c =
  Printf.sprintf
    "pools %d live = %d working (%.1f MiB) + %d constant (%.1f MiB); %d allocated, %d freed; peak \
     %.1f MiB | contexts %d unreleased (%d created, %d released) | modules %d live (%d loaded, %d \
     unloaded)"
    (live_pools c) c.live_working_pools (mib c.live_working_bytes) c.live_constant_pools
    (mib c.live_constant_bytes)
    (c.working_pools_allocated + c.constant_pools_allocated)
    c.pools_freed (mib c.peak_pool_bytes) (unreleased_contexts c) c.contexts_created
    c.contexts_released (live_modules c) c.modules_loaded c.modules_unloaded
