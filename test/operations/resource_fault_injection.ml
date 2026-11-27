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
module SO = Ir.Schedule_outcome
open Verdict.Claims
module Cache_plant = Test_utils.Cache_plant

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

let constant_allocated before after =
  after.AC.constant_pools_allocated - before.AC.constant_pools_allocated

let live_constant_delta before after = after.AC.live_constant_pools - before.AC.live_constant_pools

(* The absolute tolerance every readback here is held to against its reference. *)
let tol _ = 1e-5

let cache_entry backend best_ms =
  Cache_plant.entry ~backend ~source_digest:"gh571-source" ~best_ms ~baseline_ms:(best_ms +. 1.) ()

let resource_cache_dir = "autotune_cache_resource_fault_injection"

let cache_backend key =
  Option.map (SC.lookup ~dir:resource_cache_dir ~key:(Some key)) ~f:(fun e -> e.SC.backend)

let clean_dir dir =
  if Stdlib.Sys.file_exists dir && Stdlib.Sys.is_directory dir then
    Array.iter (Stdlib.Sys.readdir dir) ~f:(fun name ->
        Stdlib.Sys.remove (Stdlib.Filename.concat dir name))

let () =
  let values = [| -2.; -1.; 3.; 4. |] in
  let x = TDSL.ndarray values ~label:[ "rfi_x" ] ~output_dims:[ 4 ] () in
  let%op y = relu x in
  let comp = Train.forward y in
  (* Upload fixtures are mutable data, not the forward's literal (gh-ocannl-2021). *)
  let upload_target name =
    Operation.init ~mutable_:true ~l:name ~prec:Ir.Ops.single ~o:[ 4 ]
      ~f:(fun i -> values.(i.(0)))
      ~grad_spec:Tensor.Prohibit_grad ()
  in
  let x = upload_target "rfi_up" in
  let z = upload_target "rfi_z" in
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
  (* Its literal's constant pool, which this compile allocated, goes with its last binder's release
     too (gh-ocannl-1762). *)
  p "allocate-delta control commits then releases exactly one working pool"
    (working_allocated before after_compile = 1
    && live_working_delta before after_compile = 1
    && constant_allocated before after_compile = 1
    && pools_freed after_compile after_release = 2
    && live_working_delta before after_release = 0
    && live_constant_delta before after_release = 0);
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
    && constant_allocated before after = 1
    && pools_freed before after = 2
    && live_working_delta before after = 0
    && live_constant_delta before after = 0);
  let before = AC.snapshot () in
  let linked_ctx, linked_routine = Context.compile (Context.cpu ()) comp Ir.Indexing.Empty in
  let linked_ctx = Context.run linked_ctx linked_routine in
  let after_run = AC.snapshot () in
  p_close "link control runs and computes the reference"
    (Context.get_values linked_ctx y.Tensor.value)
    expected ~tol;
  Context.release linked_ctx;
  let after_release = AC.snapshot () in
  p "link control releases its committed pool exactly once"
    (working_allocated before after_run = 1
    && constant_allocated before after_run = 1
    && pools_freed after_run after_release = 2
    && live_working_delta before after_release = 0
    && live_constant_delta before after_release = 0);

  (* gh-ocannl-1555: a refused free part way through [allocate_delta]'s unwind strands no other
     pool. A fresh constant (its own label, so no earlier compile cached it) gives the delta two
     pools: the working one first, then the constant one. The second pool's allocation raises, and
     then so does the unwind's first free -- before [free_pool] runs, so that working pool stays
     live by construction. The unwind must still free the constant pool, and the caller must see the
     allocation failure, not the refused free. The two-pool shape is checked by the control below,
     which is uninjected. *)
  let ux = TDSL.ndarray values ~label:[ "rfi_unwind_x" ] ~output_dims:[ 4 ] () in
  let%op uy = relu ux in
  let unwind_comp = Train.forward uy in
  let before = AC.snapshot () in
  let alloc_hits = ref 0 and free_hits = ref 0 in
  let raised =
    match
      FI.with_callback
        (fun point ->
          match point with
          | FI.Delta_pool_allocated ->
              Int.incr alloc_hits;
              if !alloc_hits = 2 then failwith "gh1555 injected allocation failure"
          | FI.Delta_unwind_before_free ->
              Int.incr free_hits;
              if !free_hits = 1 then failwith "gh1555 injected free failure"
          | _ -> ())
        ~f:(fun () -> Context.compile (Context.cpu ()) unwind_comp Ir.Indexing.Empty)
    with
    | _ -> None
    | exception Failure msg -> Some msg
  in
  let after = AC.snapshot () in
  p "two-pool unwind: the caller sees the allocation failure, not the refused free"
    (Option.equal String.equal raised (Some "gh1555 injected allocation failure"));
  p "two-pool unwind: both pools were allocated, and a free was attempted for each"
    (!alloc_hits = 2 && !free_hits = 2);
  p "two-pool unwind: the pool after the refused free was still freed"
    (working_allocated before after = 1
    && after.AC.constant_pools_allocated - before.AC.constant_pools_allocated = 1
    && pools_freed before after = 1
    && after.AC.live_constant_pools - before.AC.live_constant_pools = 0
    && live_working_delta before after = 1);
  let before = AC.snapshot () in
  let unwind_ctx, unwind_routine = Context.compile (Context.cpu ()) unwind_comp Ir.Indexing.Empty in
  let unwind_ctx = Context.run unwind_ctx unwind_routine in
  let after_run = AC.snapshot () in
  p "two-pool unwind control allocates one working and one constant pool"
    (working_allocated before after_run = 1
    && after_run.AC.constant_pools_allocated - before.AC.constant_pools_allocated = 1);
  p_close "two-pool unwind control computes the reference"
    (Context.get_values unwind_ctx uy.Tensor.value)
    expected ~tol;
  Context.release unwind_ctx;

  p_all "the test pool cap must be a positive power of two" [ -64; 0; 4097 ] ~f:(fun cap ->
      match FI.with_pool_cap cap ~f:(fun () -> ()) with
      | () -> false
      | exception Invalid_argument _ -> true);

  (* Independent observable outputs belong to the same working group, and each fits in one small
     pool. *)
  let elements = 16 in
  let pool_bytes = elements * Ir.Ops.prec_in_bytes Ir.Ops.single in
  let pool_output label values =
    let x = TDSL.ndarray ~prec:Ir.Ops.single values ~label:[ label ] ~output_dims:[ elements ] () in
    let%op y = relu x in
    (y, Train.forward y)
  in
  let left_values = Array.init elements ~f:(fun i -> Float.of_int (i - 8)) in
  let right_values = Array.init elements ~f:(fun i -> Float.of_int (5 - i)) in
  let left, left_code = pool_output "rfi_pool_left" left_values in
  let right, right_code = pool_output "rfi_pool_right" right_values in
  let pool_comp = Ir.Assignments.sequence [ left_code; right_code ] in
  let free = 7 in
  let allocations = ref 0 in
  let at_refusal = ref None in
  let before = AC.snapshot () in
  let refused =
    let inject = function
      | FI.Delta_pool_before_allocate ->
          Int.incr allocations;
          if !allocations = 2 then (
            at_refusal := Some (AC.snapshot ());
            raise
              (SO.Device_memory_exhausted
                 {
                   backend = "simulated";
                   requested = pool_bytes;
                   free = Some free;
                   detail = "test device full";
                 }))
      | _ -> ()
    in
    FI.with_pool_cap pool_bytes ~f:(fun () ->
        FI.with_callback inject ~f:(fun () ->
            Context.compile_outcome ~name:"rfi_split_refusal" ~provenance:SO.User_schedule
              (Context.cpu ()) pool_comp Ir.Indexing.Empty))
  in
  let after = AC.snapshot () in
  p "the second working pool was refused while exactly one pool of its group was live"
    (Option.exists !at_refusal ~f:(fun held ->
         working_allocated before held = 1
         && live_working_delta before held = 1
         && held.AC.live_working_bytes - before.AC.live_working_bytes = pool_bytes));
  p "the link refusal preserves the device-memory cause and identifies its group and earlier pool"
    (match refused with
    | Error
        (SO.Classified
           {
             phase = Backend_link;
             cause = Resource_exceeded { resource = Device_memory; requested; limit; detail };
             execution_effect = No_device_writes;
           }) ->
        requested = pool_bytes
        && Option.equal Int.equal limit (Some free)
        && String.is_substring detail ~substring:"test device full"
        && String.is_substring detail
             ~substring:
               (Printf.sprintf "pool 2 of 2 of rfi_split_refusal's working delta of %d bytes"
                  (2 * pool_bytes))
        && String.is_substring detail
             ~substring:
               (Printf.sprintf "1 earlier pool(s), %d bytes, were still allocated" pool_bytes)
    | Error _ -> false
    | Ok (ctx, _) ->
        Context.release ctx;
        false);
  p "the refusal freed exactly the one pool it held"
    (working_allocated before after = 1
    && pools_freed before after = 1
    && live_working_delta before after = 0
    && after.AC.live_working_bytes = before.AC.live_working_bytes);

  let before = AC.snapshot () in
  let pool_ctx, pool_routine =
    FI.with_pool_cap pool_bytes ~f:(fun () ->
        Context.compile ~name:"rfi_split_delta" (Context.cpu ()) pool_comp Ir.Indexing.Empty)
  in
  let pool_ctx = Context.run pool_ctx pool_routine in
  let after = AC.snapshot () in
  p "small cap splits the working group into two pools"
    (working_allocated before after = 2
    && after.AC.live_working_bytes - before.AC.live_working_bytes = 2 * pool_bytes);
  let relu_values = Array.map ~f:(Float.max 0.) in
  p_close "split delta computes the left output"
    (Context.get_values pool_ctx left.Tensor.value)
    (relu_values left_values) ~tol;
  p_close "split delta computes the right output"
    (Context.get_values pool_ctx right.Tensor.value)
    (relu_values right_values) ~tol;
  Context.release pool_ctx;
  let before = AC.snapshot () in
  let default_pool_ctx, _ =
    Context.compile ~name:"rfi_default_delta" (Context.cpu ()) pool_comp Ir.Indexing.Empty
  in
  let after = AC.snapshot () in
  p "outside the test scope the working group uses the ordinary cap"
    (working_allocated before after = 1);
  Context.release default_pool_ctx;

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
  p_close "transfer control copies the requested values"
    (Context.get_values transfer_ctx x.Tensor.value)
    values ~tol;
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
  p_close "existing from-host control retries in place"
    (Context.get_values read_ctx x.Tensor.value)
    values ~tol;
  let held = AC.snapshot () in
  let raised, hits =
    injected FI.To_host_before_copy (fun () -> Context.to_host read_ctx x.Tensor.value)
  in
  let after_failed_read = AC.snapshot () in
  p "to-host copy injection fired" (raised && hits = 1);
  p "to-host failure retained the source for retry"
    (live_working_delta held after_failed_read = 0
    && pools_freed held after_failed_read = 0
    && Verdict.within "to-host failure retained the source for retry"
         (Context.get_values read_ctx x.Tensor.value)
         values ~tol);
  Context.release read_ctx;

  (* gh-ocannl-1125: uploads of nodes the context does not hold share upload arenas. The first arena
     is exact and the second targets the capacity minted so far, so [x] and [z] take one pool each
     and [w] mints a doubled arena with room for [v]. A failed upload INTO that arena minted
     nothing, so it must free nothing -- least of all the slab [w] lives in -- and must give its
     bytes back: the uninjected retry lands in the same arena, with no pool of its own. *)
  let arena_nd shift =
    Ir.Ndarray.init_array ~debug:"gh1125 arena" Ir.Ops.single ~dims:[| 4 |] ~padding:None
      ~f:(fun i -> values.(i.(0)) +. shift)
  in
  let w = upload_target "rfi_w" in
  let v = upload_target "rfi_v" in
  let nd_w = arena_nd 20. and nd_v = arena_nd 30. in
  let before_arena = AC.snapshot () in
  let arena_ctx = Context.from_host (Context.cpu ()) x.Tensor.value nd in
  let arena_ctx = Context.from_host arena_ctx z.Tensor.value (arena_nd 10.) in
  let arena_ctx = Context.from_host arena_ctx w.Tensor.value nd_w in
  let held = AC.snapshot () in
  let raised, hits =
    injected FI.From_host_before_copy (fun () -> Context.from_host arena_ctx v.Tensor.value nd_v)
  in
  let after_failed_append = AC.snapshot () in
  p "arena-append copy injection fired" (raised && hits = 1);
  p "failed arena append allocates and frees nothing"
    (working_allocated held after_failed_append = 0
    && pools_freed held after_failed_append = 0
    && live_working_delta held after_failed_append = 0);
  let reads what tn nd =
    Verdict.within ~what_ran:what "failed arena append leaves its pool-mates' values intact"
      (Context.get_values arena_ctx tn)
      (Ir.Ndarray.retrieve_flat_values nd)
      ~tol
  in
  p "failed arena append leaves its pool-mates' values intact"
    (reads "x" x.Tensor.value nd && reads "w" w.Tensor.value nd_w);
  let arena_ctx = Context.from_host arena_ctx v.Tensor.value nd_v in
  let after_retry_append = AC.snapshot () in
  p "arena-append retry lands in the arena without a pool of its own"
    (working_allocated before_arena after_retry_append = 3
    && Verdict.within ~what_ran:"v"
         "arena-append retry lands in the arena without a pool of its own"
         (Context.get_values arena_ctx v.Tensor.value)
         (Ir.Ndarray.retrieve_flat_values nd_v)
         ~tol
    && Verdict.within ~what_ran:"w"
         "arena-append retry lands in the arena without a pool of its own"
         (Context.get_values arena_ctx w.Tensor.value)
         (Ir.Ndarray.retrieve_flat_values nd_w)
         ~tol);
  Context.release arena_ctx;
  let after_arena_release = AC.snapshot () in
  p "arena control releases each of its three pools exactly once"
    (pools_freed after_retry_append after_arena_release = 3
    && live_working_delta before_arena after_arena_release = 0);

  (* gh-ocannl-1172: device copies share the upload arena and its unwind, including a failure first
     observed after the queued copy. Fresh destinations free once; append failures restore the bump
     without freeing pool-mates, and the ordinary retry must reuse the same pool. *)
  let source = Context.from_host (Context.cpu ()) x.Tensor.value nd in
  let source = Context.from_host source z.Tensor.value (arena_nd 10.) in
  let source = Context.from_host source w.Tensor.value nd_w in
  let source = Context.from_host source v.Tensor.value nd_v in
  List.iter [ FI.Transfer_pool_allocated; FI.Init_from_device_before_await ] ~f:(fun point ->
      let label = Sexp.to_string (FI.sexp_of_point point) in
      let before = AC.snapshot () in
      let raised, hits =
        injected point (fun () -> Context.copy x.Tensor.value ~src:source ~dst:(Context.cpu ()))
      in
      let after = AC.snapshot () in
      pf "%s: fresh device-copy injection fires" label (raised && hits = 1);
      pf "%s: failed device copy frees exactly its fresh pool" label
        (working_allocated before after = 1
        && pools_freed before after = 1
        && live_working_delta before after = 0));
  let before = AC.snapshot () in
  let destination = Context.copy x.Tensor.value ~src:source ~dst:(Context.cpu ()) in
  let destination = Context.from_host destination z.Tensor.value (arena_nd 10.) in
  let destination = Context.copy w.Tensor.value ~src:source ~dst:destination in
  let held = AC.snapshot () in
  p "mixed host and device transfers commit exactly three arenas"
    (working_allocated before held = 3 && live_working_delta before held = 3);
  List.iter [ FI.Transfer_pool_allocated; FI.Init_from_device_before_await ] ~f:(fun point ->
      let label = Sexp.to_string (FI.sexp_of_point point) in
      let raised, hits =
        injected point (fun () -> Context.copy v.Tensor.value ~src:source ~dst:destination)
      in
      let after = AC.snapshot () in
      pf "%s: device-copy arena append injection fires" label (raised && hits = 1);
      pf "%s: failed device-copy append allocates and frees nothing" label
        (working_allocated held after = 0
        && pools_freed held after = 0
        && live_working_delta held after = 0);
      p_all "failed device-copy append preserves every existing pool-mate"
        [ (x.Tensor.value, nd); (z.Tensor.value, arena_nd 10.); (w.Tensor.value, nd_w) ]
        ~f:(fun (tn, nd) ->
          Verdict.within
            ~what_ran:(label ^ " " ^ Ir.Tnode.label_name tn)
            "failed device-copy append preserves every existing pool-mate"
            (Context.get_values destination tn)
            (Ir.Ndarray.retrieve_flat_values nd)
            ~tol));
  let destination = Context.copy v.Tensor.value ~src:source ~dst:destination in
  let after = AC.snapshot () in
  p "device-copy retry reuses the restored arena bump"
    (working_allocated held after = 0
    && pools_freed held after = 0
    && Verdict.within "device-copy retry reuses the restored arena bump"
         (Context.get_values destination v.Tensor.value)
         (Ir.Ndarray.retrieve_flat_values nd_v)
         ~tol);
  Context.release source;
  p_close "copied destination survives releasing its source"
    (Context.get_values destination v.Tensor.value)
    (Ir.Ndarray.retrieve_flat_values nd_v)
    ~tol;
  let held = AC.snapshot () in
  Context.release destination;
  let after = AC.snapshot () in
  p "mixed transfer destination releases all three pools exactly once"
    (pools_freed held after = 3 && live_working_delta held after = -3);

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
  p "both uploads are recorded as written before the release"
    (Context.buffer_written release_ctx x.Tensor.value
    && Context.buffer_written release_ctx z.Tensor.value);
  let raised, hits =
    injected ~at:2 FI.Finalize_before_free (fun () -> Context.release release_ctx)
  in
  let after_failed_release = AC.snapshot () in
  p "finalize injection fired after one successful free" (raised && hits = 2);
  p "failed finalize reports no release and records its one completed free"
    (pools_freed before_release after_failed_release = 1
    && live_working_delta before_release after_failed_release = -1
    && contexts_released before_release after_failed_release = 0);
  (* gh-ocannl-1738: the ledger's written records follow the frees exactly. Pools are freed in id
     order, so the one completed free is [x]'s, minted first. *)
  p "failed finalize retires the written record of the freed pool only"
    ((not (Context.buffer_written release_ctx x.Tensor.value))
    && Context.buffer_written release_ctx z.Tensor.value);
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
  p "finalize retry retires the remaining written record"
    (not (Context.buffer_written release_ctx z.Tensor.value));
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
  SC.store ~dir:resource_cache_dir ~key:(Some cache_key) (cache_entry "old" 2.);
  p "cache-store control commits a readable entry"
    (Option.equal String.equal (cache_backend cache_key) (Some "old"));
  let raised, hits =
    injected FI.Schedule_cache_before_commit (fun () ->
        SC.store ~dir:resource_cache_dir ~key:(Some cache_key) (cache_entry "new" 1.))
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
  SC.store ~dir:resource_cache_dir ~key:(Some cache_key) (cache_entry "new" 1.);
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
        ~f:(fun () -> SC.lookup ~dir:resource_cache_dir ~key:(Some cache_key))
    in
    (result, !hits)
  in
  p "cache-replay injection becomes an honest miss" (replay_hits = 1 && Option.is_none replay);
  p "cache-replay control reads the committed replacement"
    (Option.equal String.equal (cache_backend cache_key) (Some "new"));
  Stdio.Out_channel.write_all
    (SC.cache_file ~dir:resource_cache_dir ~key:cache_key)
    ~data:"not a schedule cache entry";
  p "corrupt cache replay is an honest miss"
    (Option.is_none (SC.lookup ~dir:resource_cache_dir ~key:(Some cache_key)));

  (* gh-ocannl-778: every fault event names both its ownership phase and its concrete artifact
     identity. Capture an uninjected compile/run/release + cache store/lookup trace and verify exact
     phase classification and matching pool/routine/cache-file artifacts. *)
  let traced = ref [] in
  let trace_comp = comp in
  FI.with_event_callback
    (fun ev -> traced := ev :: !traced)
    ~f:(fun () ->
      let ctx, routine =
        Context.compile ~name:"rfi_trace" (Context.cpu ()) trace_comp Ir.Indexing.Empty
      in
      let ctx = Context.run ctx routine in
      ignore (Context.get_values ctx y.Tensor.value : float array);
      Context.release ctx;
      SC.store ~dir:resource_cache_dir ~key:(Some "gh778-trace") (cache_entry "traced" 1.);
      ignore (SC.lookup ~dir:resource_cache_dir ~key:(Some "gh778-trace") : SC.entry option));
  let traced = List.rev !traced in
  p_all "every traced fault event carries its point's derived ownership phase" traced ~f:(fun ev ->
      FI.equal_ownership_phase ev.phase (FI.phase_of_point ev.point));
  let working_allocated_pools =
    List.filter_map traced ~f:(fun ev ->
        match (ev.point, ev.artifact) with
        | FI.Delta_pool_allocated, FI.Pool { pool_id; constant = Some false; _ } -> Some pool_id
        | _ -> None)
  in
  let finalized_pools =
    List.filter_map traced ~f:(fun ev ->
        match (ev.point, ev.artifact) with
        | FI.Finalize_before_free, FI.Pool { pool_id; _ } -> Some pool_id
        | _ -> None)
  in
  let saw_routine_artifact =
    List.exists traced ~f:(fun ev ->
        match (ev.point, ev.artifact) with
        | FI.Link_after_delta, FI.Routine { name } -> String.equal name "rfi_trace"
        | _ -> false)
  in
  let saw_cache_file_artifact =
    List.exists traced ~f:(fun ev ->
        match (ev.point, ev.artifact) with
        | FI.Schedule_cache_before_commit, FI.Cache_file { path } ->
            String.is_substring path ~substring:resource_cache_dir
        | _ -> false)
  in
  p "traced events identify the same pool id from delta allocation to finalize free"
    ((not (List.is_empty working_allocated_pools))
    && List.equal Int.equal working_allocated_pools finalized_pools);
  p "traced events identify routine and cache-file artifacts"
    (saw_routine_artifact && saw_cache_file_artifact);

  (* gh-ocannl-778: sticky asynchronous failures are the baseline acceptance path for
     synchronization cleanup (matching persistent worker/stream errors); one-shot exceptions above
     remain as negative controls. *)
  let latched_target = Context.from_host (Context.cpu ()) x.Tensor.value nd in
  let before = AC.snapshot () in
  let ( sticky_raised,
        sticky_hits,
        sticky_triggered,
        cross_seam_raised,
        before_cross,
        after_cross,
        after_clear_release_ok,
        after_clear_upload_ok ) =
    FI.with_sticky_async_fault ~trigger:FI.From_host_before_await (fun fault ->
        let raised =
          match Context.from_host (Context.cpu ()) x.Tensor.value nd with
          | _ -> false
          | exception Failure msg -> String.is_substring msg ~substring:"sticky async"
        in
        (* While latched on [From_host_before_await], a DIFFERENT async await seam
           ([Finalize_before_await]) must also raise the sticky exception and commit nothing. *)
        let before_cross = AC.snapshot () in
        let cross_seam_raised =
          match Context.release latched_target with
          | () -> false
          | exception Failure msg -> String.is_substring msg ~substring:"sticky async"
        in
        let after_cross = AC.snapshot () in
        let hits = FI.sticky_fault_hits fault in
        let triggered = FI.sticky_fault_triggered fault in
        FI.clear_sticky_fault fault;
        let after_clear_release_ok =
          match Context.release latched_target with () -> true | exception _ -> false
        in
        let after_clear_upload_ok =
          match Context.from_host (Context.cpu ()) x.Tensor.value nd with
          | fresh ->
              Context.release fresh;
              true
          | exception _ -> false
        in
        ( raised,
          hits,
          triggered,
          cross_seam_raised,
          before_cross,
          after_cross,
          after_clear_release_ok,
          after_clear_upload_ok ))
  in
  let after = AC.snapshot () in
  p "sticky from-host await latches and fails cleanup await persistently"
    (sticky_raised && sticky_triggered
    && List.exists sticky_hits ~f:(fun ev -> FI.equal_point ev.point FI.From_host_before_await)
    && List.exists sticky_hits ~f:(fun ev ->
        FI.equal_point ev.point FI.Transfer_cleanup_before_await));
  p "latched sticky fault raises at a different await seam and clears cleanly"
    (cross_seam_raised
    && pools_freed before_cross after_cross = 0
    && live_working_delta before_cross after_cross = 0
    && List.exists sticky_hits ~f:(fun ev -> FI.equal_point ev.point FI.Finalize_before_await)
    && after_clear_release_ok && after_clear_upload_ok);
  p "sticky from-host await still frees its unreachable pool and cleared contexts once"
    (working_allocated before after = 2
    && pools_freed before after = 3
    && live_working_delta before after = -1);

  let sticky_src = Context.from_host (Context.cpu ()) x.Tensor.value nd in
  let before = AC.snapshot () in
  let sticky_copy_raised, sticky_copy_hits =
    FI.with_sticky_async_fault ~trigger:FI.Init_from_device_before_await (fun fault ->
        let raised =
          match Context.copy x.Tensor.value ~src:sticky_src ~dst:(Context.cpu ()) with
          | _ -> false
          | exception Failure msg -> String.is_substring msg ~substring:"sticky async"
        in
        (raised, FI.sticky_fault_hits fault))
  in
  let after = AC.snapshot () in
  p "sticky init-from-device await latches through cleanup await and frees its fresh pool once"
    (sticky_copy_raised
    && List.exists sticky_copy_hits ~f:(fun ev ->
        FI.equal_point ev.point FI.Transfer_cleanup_before_await)
    && working_allocated before after = 1
    && pools_freed before after = 1
    && live_working_delta before after = 0);
  Context.release sticky_src;

  (* Sticky finalize await: repeated releases fail while latched and commit nothing; once cleared,
     retry frees the pool once. Also verified on [multidev_cc] on the calling domain. *)
  List.iter
    [ ("cc", fun () -> Context.cpu ()); ("multidev_cc", fun () -> Context.cpu ~threads:2 ()) ]
    ~f:(fun (label, mk_root) ->
      let ctx = Context.from_host (mk_root ()) x.Tensor.value nd in
      let before = AC.snapshot () in
      let first_failed, second_failed, still_latched, after_latched =
        FI.with_sticky_async_fault ~trigger:FI.Finalize_before_await (fun fault ->
            let try_release () =
              match Context.release ctx with
              | () -> false
              | exception Failure msg -> String.is_substring msg ~substring:"sticky async"
            in
            let r1 = try_release () in
            let r2 = try_release () in
            let snap = AC.snapshot () in
            let latched = FI.sticky_fault_triggered fault in
            FI.clear_sticky_fault fault;
            Context.release ctx;
            (r1, r2, latched, snap))
      in
      let after_clear = AC.snapshot () in
      pf "sticky finalize await on %s: repeated retries raise and commit nothing while latched"
        label
        (first_failed && second_failed && still_latched
        && pools_freed before after_latched = 0
        && live_working_delta before after_latched = 0
        && contexts_released before after_latched = 0);
      pf "sticky finalize await on %s: clearing the latch lets retry release exactly once" label
        (pools_freed before after_clear = 1
        && live_working_delta before after_clear = -1
        && contexts_released before after_clear = 1));

  (* gh-ocannl-778: partial-batch link cleanup on a 2-segment fissioned routine. Failing at segment
     index 1 (after segment 0 linked) must unwind the entire delta cleanly; uninjected retry runs
     and computes the exact reference. *)
  let seg_a = TDSL.ndarray values ~label:[ "rfi_seg_a" ] ~output_dims:[ 4 ] () in
  let%op seg_mid = relu seg_a in
  Train.set_materialized seg_mid.Tensor.value;
  let%op seg_out = seg_mid + 1. in
  Train.set_materialized seg_out.Tensor.value;
  let seg_comp = Train.forward seg_out in
  let two_segments (o : Ir.Low_level.optimized) =
    let stmts =
      List.filter (Ir.Low_level.flat_lines [ o.llc ]) ~f:(function
        | Ir.Low_level.Comment _ | Ir.Low_level.Noop -> false
        | _ -> true)
    in
    let n = List.length stmts in
    let first = Int.max 1 (n / 2) in
    let _, tuples =
      Ir.Schedule.fission_segmented
        ~replay:([ (`Solo, first); (`Solo, n - first) ], fun _ _ -> None)
        ~preset:(fun _ -> [])
        ~zero_sched:(fun _ -> [])
        ~static_indices:[] o
    in
    List.map tuples ~f:(fun (_, _, _, post) -> post)
  in
  let seg_hits = ref [] in
  let before = AC.snapshot () in
  let partial_batch_failed =
    match
      FI.with_event_callback
        (fun ev ->
          match (ev.point, ev.artifact) with
          | FI.Link_batch_before_segment, FI.Routine_segment { index; count; _ } ->
              seg_hits := (index, count) :: !seg_hits;
              if index = 1 then failwith "gh778 injected partial-batch failure"
          | _ -> ())
        ~f:(fun () ->
          Context.compile ~name:"rfi_partial_batch" ~lowered_transform:two_segments (Context.cpu ())
            seg_comp Ir.Indexing.Empty)
    with
    | _ -> false
    | exception Failure msg -> String.is_substring msg ~substring:"gh778 injected partial-batch"
  in
  let after = AC.snapshot () in
  p "partial-batch link failure fired on segment 1 of 2"
    (partial_batch_failed
    && List.equal
         (fun (a1, b1) (a2, b2) -> a1 = a2 && b1 = b2)
         (List.rev !seg_hits)
         [ (0, 2); (1, 2) ]);
  p "partial-batch link failure freed the whole delta with zero live growth"
    (working_allocated before after >= 1
    && live_working_delta before after = 0
    && live_constant_delta before after = 0);
  let before = AC.snapshot () in
  let seg_ctx, seg_routine =
    Context.compile ~name:"rfi_partial_batch_ok" ~lowered_transform:two_segments (Context.cpu ())
      seg_comp Ir.Indexing.Empty
  in
  let seg_ctx = Context.run seg_ctx seg_routine in
  p_close "partial-batch control links and computes exact values"
    (Context.get_values seg_ctx seg_out.Tensor.value)
    (Array.map expected ~f:(fun v -> v +. 1.))
    ~tol;
  Context.release seg_ctx;
  let after = AC.snapshot () in
  p "partial-batch control releases cleanly with zero live growth"
    (live_working_delta before after = 0 && live_constant_delta before after = 0);

  (* gh-ocannl-778: centralized merge-slab invalidate/commit seam on cc. *)
  let before_merge = AC.snapshot () in
  let small_vals = Array.init 4 ~f:(fun i -> Float.of_int (i + 1)) in
  let large_vals = Array.init 16 ~f:(fun i -> Float.of_int (i + 10)) in
  let small_t =
    Operation.init ~mutable_:true ~l:"rfi_merge_small" ~prec:Ir.Ops.single ~o:[ 4 ]
      ~f:(fun i -> small_vals.(i.(0)))
      ~grad_spec:Tensor.Prohibit_grad ()
  in
  let large_t =
    Operation.init ~mutable_:true ~l:"rfi_merge_large" ~prec:Ir.Ops.single ~o:[ 16 ]
      ~f:(fun i -> large_vals.(i.(0)))
      ~grad_spec:Tensor.Prohibit_grad ()
  in
  let m_src_small = Context.set_values (Context.cpu ()) small_t.Tensor.value small_vals in
  let m_src = Context.set_values m_src_small large_t.Tensor.value large_vals in
  let m_dst = Context.cpu () in
  let m_dst =
    Context.copy ~into_merge_buffer:Ir.Backend_intf.Copy ~src:m_src ~dst:m_dst small_t.Tensor.value
  in
  let before_grow = Context.merge_slab_state m_dst in
  p "merge slab holds all four claims after initial fill on cc"
    (before_grow.has_slab && before_grow.has_location
    && before_grow.capacity = 4 * 4
    && before_grow.has_writer);
  let merge_hit_size = ref None in
  let callback_state = ref None in
  let grow_failed =
    match
      FI.with_event_callback
        (fun ev ->
          match (ev.point, ev.artifact) with
          | FI.Merge_slab_before_allocate, FI.Merge_slab { size_in_bytes; _ } ->
              merge_hit_size := Some size_in_bytes;
              (* Inspecting [merge_slab_state] inside the callback must not re-enter [pools_mutex]
                 and must already observe all four claims invalidated. *)
              callback_state := Some (Context.merge_slab_state m_dst);
              failwith "gh778 injected merge-slab allocation failure"
          | _ -> ())
        ~f:(fun () ->
          Context.copy ~into_merge_buffer:Ir.Backend_intf.Copy ~src:m_src ~dst:m_dst
            large_t.Tensor.value)
    with
    | _ -> false
    | exception Failure msg -> String.is_substring msg ~substring:"gh778 injected merge-slab"
  in
  let after_failed_grow = Context.merge_slab_state m_dst in
  let callback_saw_invalid =
    match !callback_state with
    | Some s -> (not s.has_slab) && (not s.has_location) && s.capacity = 0 && not s.has_writer
    | None -> false
  in
  p "failed merge-slab grow invalidates all four claims on cc"
    (grow_failed && callback_saw_invalid
    && Option.equal Int.equal !merge_hit_size (Some (16 * 4))
    && (not after_failed_grow.has_slab)
    && (not after_failed_grow.has_location)
    && after_failed_grow.capacity = 0 && not after_failed_grow.has_writer);
  let m_dst =
    Context.copy ~into_merge_buffer:Ir.Backend_intf.Copy ~src:m_src ~dst:m_dst large_t.Tensor.value
  in
  let after_retry_grow = Context.merge_slab_state m_dst in
  p "merge-slab retry restores all four claims with grown capacity on cc"
    (after_retry_grow.has_slab && after_retry_grow.has_location
    && after_retry_grow.capacity = 16 * 4
    && after_retry_grow.has_writer);
  let m_dst_upload = Context.set_values m_dst large_t.Tensor.value (Array.create ~len:16 1.) in
  let merge_consume = [%cd large_t =+ large_t.merge] in
  let m_dst_compiled, consume_r =
    Context.compile ~name:"rfi_merge_consume" m_dst_upload merge_consume Ir.Indexing.Empty
  in
  let m_dst_compiled = Context.run m_dst_compiled consume_r in
  p_close "merge-slab retry consumer computes exact values on cc"
    (Context.get_values m_dst_compiled large_t.Tensor.value)
    (Array.map large_vals ~f:(fun v -> v +. 1.))
    ~tol;
  Context.release m_dst_compiled;
  Context.release m_dst_upload;
  Context.release m_src;
  Context.release m_src_small;
  let after_merge = AC.snapshot () in
  p "merge-slab probe releases all upload and compiled contexts with zero live growth on cc"
    (live_working_delta before_merge after_merge = 0)
