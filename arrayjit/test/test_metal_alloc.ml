(* Regression test for gh-ocannl-320 + the buffer-addressing slab API: exercises the *real* Metal
   allocator paths (not just the pure classifier in test_metal_storage_mode.ml).

   It allocates pools through [alloc_pool] and asserts the storage mode of the resulting slab via
   [storage_mode_of_pool] (per-pool storage mode replaces the old concrete [Metal.Buffer.t]
   exposure). If any allocation call site stopped threading [?mode], or [resource_options_for_mode]
   stopped being used, a GPU-only pool would come back [Shared] and the printed line would change.

   It also drives data through a private-mode pool end to end ([from_host] in via [resolve_pool],
   [to_host] back out) and a [memset_zero]'d private pool, verifying the actual values. A broken
   private-buffer blit path (or a private zero-fill that silently no-ops) would change the [true]
   verdicts below. *)

open Base
module Tn = Ir.Tnode
module Nd = Ir.Ndarray
module Ops = Ir.Ops
module SM = Metal.Resource.StorageMode
module B = Metal_backend.Fresh ()

let sm_str = function
  | SM.Shared -> "Shared"
  | SM.Private -> "Private"
  | SM.Managed -> "Managed"
  | SM.Memoryless -> "Memoryless"

let prec = Ops.single
let dims = [| 4 |]
let size_in_bytes = Array.fold dims ~init:1 ~f:( * ) * Ops.prec_in_bytes prec

let () =
  let device = B.get_device ~ordinal:0 in
  (* Deterministic, test-managed pool ids (pool 0 is the device's reserved merge buffer). *)
  let next_pool = ref 1 in
  let fresh_pool ?mode () =
    let pool_id = !next_pool in
    Int.incr next_pool;
    B.alloc_pool ?mode device ~pool_id ~size_in_bytes ~alignment:1;
    pool_id
  in

  (* --- alloc_pool: storage mode follows the tnode memory mode --- *)
  let check_pool label mode =
    let pool_id = fresh_pool ?mode () in
    Stdio.printf "alloc_pool %-13s -> %s\n" label (sm_str (B.storage_mode_of_pool device ~pool_id))
  in
  check_pool "Local" (Some Tn.Local);
  check_pool "Device_only" (Some Tn.Device_only);
  check_pool "On_device" (Some Tn.On_device);
  check_pool "Effectively_constant" (Some Tn.Effectively_constant);
  check_pool "Materialized" (Some Tn.Materialized);
  check_pool "(no mode)" None;

  (* --- memset_zero: GPU-only pools still get private storage --- *)
  let zeros_priv = fresh_pool ~mode:Tn.Device_only () in
  B.memset_zero device ~pool_id:zeros_priv ~offset:0 ~size_in_bytes;
  Stdio.printf "memset_zero Device_only  -> %s\n" (sm_str (B.storage_mode_of_pool device ~pool_id:zeros_priv));

  (* --- from_host / to_host round-trip through a private-mode pool --- *)
  let ctx = B.make_context device in
  let src = Nd.create_array ~debug:"src" prec ~dims ~padding:None in
  for i = 0 to 3 do
    Nd.set_from_float src [| i |] (Float.of_int (i + 1))
  done;
  let dev_pool = fresh_pool ~mode:Tn.Device_only () in
  let dev_ptr = B.resolve_pool device { pool_id = dev_pool; offset = 0 } in
  B.from_host ~dst_ptr:dev_ptr ~dst:ctx src;
  let dst = Nd.create_array ~debug:"dst" prec ~dims ~padding:None in
  B.to_host ~src_ptr:dev_ptr ~src:ctx dst;
  B.await device;
  let roundtrip_ok =
    Array.for_all [| 0; 1; 2; 3 |] ~f:(fun i ->
        Float.equal (Nd.get_as_float dst [| i |]) (Float.of_int (i + 1)))
  in
  Stdio.printf "from_host/to_host private round-trip preserves values -> %b\n" roundtrip_ok;

  (* --- a private memset_zero'd pool actually reads back as zeros --- *)
  let zdst = Nd.create_array ~debug:"zdst" prec ~dims ~padding:None in
  for i = 0 to 3 do
    Nd.set_from_float zdst [| i |] 99.0
  done;
  B.to_host ~src_ptr:(B.resolve_pool device { pool_id = zeros_priv; offset = 0 }) ~src:ctx zdst;
  B.await device;
  let zeros_ok =
    Array.for_all [| 0; 1; 2; 3 |] ~f:(fun i -> Float.equal (Nd.get_as_float zdst [| i |]) 0.0)
  in
  Stdio.printf "memset_zero private pool reads back zeros -> %b\n" zeros_ok
