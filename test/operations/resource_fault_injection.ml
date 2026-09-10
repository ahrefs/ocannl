(* gh-ocannl-571: release/cleanup code is accepted by its error paths. This GPU-free harness drives
   the shared resource-owning seams on [cc], injects one failure at a time, and pairs every injected
   scenario with an uninjected control. Counts are exact: a failure either commits no ownership or
   releases each committed working pool once. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module FI = Ir.Resource_fault_injection
module AC = Ir.Alloc_census
module SC = Ir.Schedule_cache
open Verdict.Claims

let injected ?(at = 1) point f =
  let hits = ref 0 in
  let raised =
    match
      FI.with_callback
        (fun seen ->
          if FI.equal_point seen point then (
            Int.incr hits;
            if !hits = at then failwith "gh571 injected resource failure"))
        ~f
    with
    | _ -> false
    | exception Failure msg -> String.is_substring msg ~substring:"gh571 injected"
  in
  (raised, !hits)

let working_allocated before after =
  after.AC.working_pools_allocated - before.AC.working_pools_allocated

let pools_freed before after = after.AC.pools_freed - before.AC.pools_freed
let live_working_delta before after = after.AC.live_working_pools - before.AC.live_working_pools
let contexts_released before after = after.AC.contexts_released - before.AC.contexts_released

(* Non-empty by construction (gh-ocannl-746): [Array.for_all2_exn] answers [true] on two empty
   arrays, and a readback and the reference it is compared against go empty TOGETHER -- the
   reference went through the same path. {!Verdict.p_all2} is the claim-shaped form; the sites
   reached through this helper keep a boolean, so the guard lives here. *)
let approx_array a b =
  (not (Array.is_empty a))
  && Array.length a = Array.length b
  && Array.for_all2_exn a b ~f:(fun x y -> Float.(abs (x - y) < 1e-5))

let cache_entry backend best_ms : SC.entry =
  {
    version = SC.entry_version;
    backend;
    numerics = SC.numerics_tag ();
    codegen = None;
    objective = None;
    source_digest = "gh571-source";
    saved = [];
    segments = None;
    finer_fission = None;
    best_ms;
    baseline_ms = best_ms +. 1.;
    mma_best_ms = None;
    default_ms = None;
    default_fingerprint = None;
  }

let resource_cache_dir = "autotune_cache_resource_fault_injection"

let cache_backend key =
  Option.map (SC.lookup ~dir:resource_cache_dir ~key) ~f:(fun e -> e.SC.backend)

let clean_dir dir =
  if Stdlib.Sys.file_exists dir && Stdlib.Sys.is_directory dir then
    Array.iter (Stdlib.Sys.readdir dir) ~f:(fun name ->
        Stdlib.Sys.remove (Stdlib.Filename.concat dir name))

let () =
  let values = [| -2.; -1.; 3.; 4. |] in
  let x = TDSL.ndarray values ~label:[ "rfi_x" ] ~output_dims:[ 4 ] () in
  let z = TDSL.ndarray values ~label:[ "rfi_z" ] ~output_dims:[ 4 ] () in
  let%op y = relu x in
  let comp = Train.forward y in
  let expected = [| 0.; 0.; 3.; 4. |] in

  (* Partial allocation: the pool is already in the backend table and census when the injected
     failure fires, but [allocate_delta]'s local minted set can still unwind it. *)
  let before = AC.snapshot () in
  let raised, hits =
    injected FI.Delta_pool_allocated (fun () ->
        Context.compile (Context.cpu ()) comp Ir.Indexing.Empty)
  in
  let after = AC.snapshot () in
  p "allocate-delta injection fired" (raised && hits = 1);
  p "allocate-delta failure freed its one partial working pool"
    (working_allocated before after = 1
    && pools_freed before after = 1
    && live_working_delta before after = 0);
  let before = AC.snapshot () in
  let control_ctx, control_routine = Context.compile (Context.cpu ()) comp Ir.Indexing.Empty in
  let after_compile = AC.snapshot () in
  Context.release control_ctx;
  let after_release = AC.snapshot () in
  p "allocate-delta control commits then releases exactly one working pool"
    (working_allocated before after_compile = 1
    && live_working_delta before after_compile = 1
    && pools_freed after_compile after_release = 1
    && live_working_delta before after_release = 0);
  ignore control_routine;

  (* Backend link: all delta locations and constant-cache insertions exist, but no child context
     exists yet. [with_delta] must give the unreachable working pool back. *)
  let before = AC.snapshot () in
  let raised, hits =
    injected FI.Link_after_delta (fun () -> Context.compile (Context.cpu ()) comp Ir.Indexing.Empty)
  in
  let after = AC.snapshot () in
  p "post-allocation link injection fired" (raised && hits = 1);
  p "link failure freed the unowned delta exactly once"
    (working_allocated before after = 1
    && pools_freed before after = 1
    && live_working_delta before after = 0);
  let before = AC.snapshot () in
  let linked_ctx, linked_routine = Context.compile (Context.cpu ()) comp Ir.Indexing.Empty in
  let linked_ctx = Context.run linked_ctx linked_routine in
  let after_run = AC.snapshot () in
  p "link control runs and computes the reference"
    (approx_array (Context.get_values linked_ctx y.Tensor.value) expected);
  Context.release linked_ctx;
  let after_release = AC.snapshot () in
  p "link control releases its committed pool exactly once"
    (working_allocated before after_run = 1
    && pools_freed after_run after_release = 1
    && live_working_delta before after_release = 0);

  (* A fresh host-transfer destination is the second shared allocation site. Failure after allocate
     but before copy must not leave the pool-table entry rooted. *)
  let nd =
    Ir.Ndarray.init_array ~debug:"gh571 transfer" Ir.Ops.single ~dims:[| 4 |] ~padding:None
      ~f:(fun i -> values.(i.(0)))
  in
  let before = AC.snapshot () in
  let raised, hits =
    injected FI.Transfer_pool_allocated (fun () ->
        Context.from_host (Context.cpu ()) x.Tensor.value nd)
  in
  let after = AC.snapshot () in
  p "transfer-allocation injection fired" (raised && hits = 1);
  p "failed transfer freed its fresh destination pool exactly once"
    (working_allocated before after = 1
    && pools_freed before after = 1
    && live_working_delta before after = 0);
  let before = AC.snapshot () in
  let transfer_ctx = Context.from_host (Context.cpu ()) x.Tensor.value nd in
  let after_copy = AC.snapshot () in
  p "transfer control copies the requested values"
    (approx_array (Context.get_values transfer_ctx x.Tensor.value) values);
  Context.release transfer_ctx;
  let after_release = AC.snapshot () in
  p "transfer control releases its destination exactly once"
    (working_allocated before after_copy = 1
    && pools_freed after_copy after_release = 1
    && live_working_delta before after_release = 0);

  (* Copy failures are separate from allocation failures: the transfer wrapper must unwind the same
     pool, while a readback failure must retain the context-owned source for a retry. *)
  let before = AC.snapshot () in
  let raised, hits =
    injected FI.From_host_before_copy (fun () ->
        Context.from_host (Context.cpu ()) x.Tensor.value nd)
  in
  let after = AC.snapshot () in
  p "from-host copy injection fired" (raised && hits = 1);
  p "from-host copy failure freed the fresh pool"
    (working_allocated before after = 1
    && pools_freed before after = 1
    && live_working_delta before after = 0);
  (* Shared/GPU uploads may report their error only when the stream is synchronized. The fresh pool
     must remain guarded through that await because no updated context reaches the caller when it
     raises. The callback models the first await reporting that queued failure; cleanup's retry
     await then succeeds and permits the exact free. *)
  let before = AC.snapshot () in
  let upload_await_hits = ref 0 in
  let cleanup_await_hits = ref 0 in
  let raised =
    match
      FI.with_callback
        (fun point ->
          if FI.equal_point point FI.From_host_before_await then (
            Int.incr upload_await_hits;
            failwith "gh571 injected persistent upload await failure")
          else if FI.equal_point point FI.Transfer_cleanup_before_await then (
            Int.incr cleanup_await_hits;
            failwith "gh571 injected persistent cleanup await failure"))
        ~f:(fun () -> Context.from_host (Context.cpu ()) x.Tensor.value nd)
    with
    | _ -> false
    | exception Failure msg -> String.is_substring msg ~substring:"gh571 injected persistent"
  in
  let after = AC.snapshot () in
  p "persistent from-host await injection fired for upload and cleanup"
    (raised && !upload_await_hits = 1 && !cleanup_await_hits = 1);
  p "persistent from-host await failure freed the unreachable fresh pool"
    (working_allocated before after = 1
    && pools_freed before after = 1
    && live_working_delta before after = 0);
  let read_ctx = Context.from_host (Context.cpu ()) x.Tensor.value nd in
  let held = AC.snapshot () in
  let raised, hits =
    injected FI.From_host_before_copy (fun () -> Context.from_host read_ctx x.Tensor.value nd)
  in
  let after_failed_overwrite = AC.snapshot () in
  p "existing from-host copy injection fired" (raised && hits = 1);
  p "existing from-host failure retains the destination for retry"
    (working_allocated held after_failed_overwrite = 0
    && pools_freed held after_failed_overwrite = 0
    && live_working_delta held after_failed_overwrite = 0);
  let read_ctx = Context.from_host read_ctx x.Tensor.value nd in
  p "existing from-host control retries in place"
    (approx_array (Context.get_values read_ctx x.Tensor.value) values);
  let held = AC.snapshot () in
  let raised, hits =
    injected FI.To_host_before_copy (fun () -> Context.to_host read_ctx x.Tensor.value)
  in
  let after_failed_read = AC.snapshot () in
  p "to-host copy injection fired" (raised && hits = 1);
  p "to-host failure retained the source for retry"
    (live_working_delta held after_failed_read = 0
    && pools_freed held after_failed_read = 0
    && approx_array (Context.get_values read_ctx x.Tensor.value) values);
  Context.release read_ctx;

  (* Await is the first fallible release action. A failure there commits neither the finalized flag
     nor any free, and the uninjected retry performs the one cleanup. *)
  let await_ctx = Context.from_host (Context.cpu ()) x.Tensor.value nd in
  let before_await_release = AC.snapshot () in
  let raised, hits = injected FI.Finalize_before_await (fun () -> Context.release await_ctx) in
  let after_failed_await = AC.snapshot () in
  p "finalize-await injection fired" (raised && hits = 1);
  p "failed finalize await commits no cleanup state"
    (pools_freed before_await_release after_failed_await = 0
    && live_working_delta before_await_release after_failed_await = 0
    && contexts_released before_await_release after_failed_await = 0);
  Context.release await_ctx;
  let after_await_retry = AC.snapshot () in
  p "finalize-await control retry releases exactly once"
    (pools_freed before_await_release after_await_retry = 1
    && live_working_delta before_await_release after_await_retry = -1
    && contexts_released before_await_release after_await_retry = 1);

  (* Retryable finalization with two transfer pools: fail before the second free. The first pool id
     is remembered in the context, so retry calls the backend only for the remaining pool. *)
  let nd_z =
    Ir.Ndarray.init_array ~debug:"gh571 transfer z" Ir.Ops.single ~dims:[| 4 |] ~padding:None
      ~f:(fun i -> values.(i.(0)) +. 10.)
  in
  let release_ctx = Context.from_host (Context.cpu ()) x.Tensor.value nd in
  let release_ctx = Context.from_host release_ctx z.Tensor.value nd_z in
  let before_release = AC.snapshot () in
  let raised, hits =
    injected ~at:2 FI.Finalize_before_free (fun () -> Context.release release_ctx)
  in
  let after_failed_release = AC.snapshot () in
  p "finalize injection fired after one successful free" (raised && hits = 2);
  p "failed finalize reports no release and records its one completed free"
    (pools_freed before_release after_failed_release = 1
    && live_working_delta before_release after_failed_release = -1
    && contexts_released before_release after_failed_release = 0);
  let retry_hits = ref 0 in
  FI.with_callback
    (fun point -> if FI.equal_point point FI.Finalize_before_free then Int.incr retry_hits)
    ~f:(fun () -> Context.release release_ctx);
  let after_retry = AC.snapshot () in
  p "finalize retry calls free only for the unreleased pool"
    (!retry_hits = 1
    && pools_freed before_release after_retry = 2
    && live_working_delta before_release after_retry = -2
    && contexts_released before_release after_retry = 1);
  let second_hits = ref 0 in
  FI.with_callback (fun _ -> Int.incr second_hits) ~f:(fun () -> Context.release release_ctx);
  let after_second = AC.snapshot () in
  p "finalize control is idempotent after success"
    (!second_hits = 0 && AC.equal after_retry after_second);

  (* Schedule-cache ownership is filesystem ownership: a failed writer must preserve the previous
     committed entry and clean only its own unique staging file; a failed/corrupt replay is a
     miss. *)
  let cache_key = "gh571-cache" in
  clean_dir resource_cache_dir;
  SC.store ~dir:resource_cache_dir ~key:cache_key (cache_entry "old" 2.);
  p "cache-store control commits a readable entry"
    (Option.equal String.equal (cache_backend cache_key) (Some "old"));
  let raised, hits =
    injected FI.Schedule_cache_before_commit (fun () ->
        SC.store ~dir:resource_cache_dir ~key:cache_key (cache_entry "new" 1.))
  in
  (* Quantified over the directory's entries, which the committed entry keeps non-empty. The staging
     naming scheme belongs to the helper that creates these files, so ask it rather than spelling an
     infix again here: a change to the scheme must not quietly turn this into a predicate that
     matches nothing. *)
  let entries = Array.to_list (Stdlib.Sys.readdir resource_cache_dir) in
  p "cache-store injection fired" (raised && hits = 1);
  let old_preserved = Option.equal String.equal (cache_backend cache_key) (Some "old") in
  p_all "failed cache commit preserves the old entry and removes its staging file" entries
    ~f:(fun entry -> old_preserved && not (Utils.Atomic_file.is_staging_file entry));
  SC.store ~dir:resource_cache_dir ~key:cache_key (cache_entry "new" 1.);
  p "cache-store retry commits the replacement"
    (Option.equal String.equal (cache_backend cache_key) (Some "new"));
  let replay, replay_hits =
    let hits = ref 0 in
    let result =
      FI.with_callback
        (fun point ->
          if FI.equal_point point FI.Schedule_cache_before_replay then (
            Int.incr hits;
            failwith "gh571 injected replay failure"))
        ~f:(fun () -> SC.lookup ~dir:resource_cache_dir ~key:cache_key)
    in
    (result, !hits)
  in
  p "cache-replay injection becomes an honest miss" (replay_hits = 1 && Option.is_none replay);
  p "cache-replay control reads the committed replacement"
    (Option.equal String.equal (cache_backend cache_key) (Some "new"));
  Stdio.Out_channel.write_all
    (Stdlib.Filename.concat resource_cache_dir (cache_key ^ ".sexp"))
    ~data:"not a schedule cache entry";
  p "corrupt cache replay is an honest miss"
    (Option.is_none (SC.lookup ~dir:resource_cache_dir ~key:cache_key))
