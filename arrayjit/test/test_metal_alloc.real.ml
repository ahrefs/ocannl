(* Regression test for gh-ocannl-344: exercises the *real* Metal allocator paths (not just a
   classifier). It allocates pools through [alloc_pool] and asserts the storage mode of the resulting
   slab via [storage_mode_of_pool] -- after gh-ocannl-344 every pool is [Shared], regardless of the
   tnode memory mode.

   It then drives data through a *multi-tenant* pool (two 4-element regions in one slab) end to end:
   [from_host] in and [to_host] back out of the region at byte offset > 0, plus a [memset_zero] at
   that same non-zero offset. This pins the offset-aware blit paths ([from_host]/[to_host] honoring
   [buffer_loc.offset], [memset_zero] filling at [offset]) that pooling depends on: a path that
   ignored the offset (the pre-pooling behavior) would read/write the wrong region and flip the
   [true] verdicts below to [false]. *)

open Base
module Tn = Ir.Tnode
module Nd = Ir.Ndarray
module Ops = Ir.Ops
module SM = Metal.Resource.StorageMode
module BI = Ir.Backend_intf
module B = Metal_backend.Fresh ()

let sm_str = function
  | SM.Shared -> "Shared"
  | SM.Private -> "Private"
  | SM.Managed -> "Managed"
  | SM.Memoryless -> "Memoryless"

let prec = Ops.single
let dims = [| 4 |]
let region_bytes = Array.fold dims ~init:1 ~f:( * ) * Ops.prec_in_bytes prec

let () =
  let device = B.get_device ~ordinal:0 in
  (* Deterministic, test-managed pool ids (pool 0 is the device's reserved merge buffer). *)
  let next_pool = ref 1 in
  let fresh_pool ?mode ~size_in_bytes () =
    let pool_id = !next_pool in
    Int.incr next_pool;
    B.alloc_pool ?mode device ~pool_id ~size_in_bytes ~alignment:1;
    pool_id
  in

  (* --- alloc_pool: storage mode is always Shared now --- *)
  let check_pool label mode =
    let pool_id = fresh_pool ?mode ~size_in_bytes:region_bytes () in
    Stdio.printf "alloc_pool %-13s -> %s\n" label (sm_str (B.storage_mode_of_pool device ~pool_id))
  in
  check_pool "Local" (Some Tn.Local);
  check_pool "Device_only" (Some Tn.Device_only);
  check_pool "On_device" (Some Tn.On_device);
  check_pool "Effectively_constant" (Some Tn.Effectively_constant);
  check_pool "Materialized" (Some Tn.Materialized);
  check_pool "(no mode)" None;

  (* --- A multi-tenant pool: region A at offset 0, region B at offset [region_bytes]. --- *)
  let ctx = B.make_context device in
  let pool = fresh_pool ~mode:Tn.Device_only ~size_in_bytes:(2 * region_bytes) () in
  let loc_a = { BI.pool_id = pool; offset = 0 } in
  let loc_b = { BI.pool_id = pool; offset = region_bytes } in

  (* memset_zero at the non-zero offset must not touch region A. Seed A with non-zero data first. *)
  let a_src = Nd.create_array ~debug:"a" prec ~dims ~padding:None in
  for i = 0 to 3 do
    Nd.set_from_float a_src [| i |] (Float.of_int (i + 1))
  done;
  B.from_host ~dst:ctx ~dst_loc:loc_a a_src;

  (* from_host / to_host round-trip through region B (offset > 0). *)
  let b_src = Nd.create_array ~debug:"b" prec ~dims ~padding:None in
  for i = 0 to 3 do
    Nd.set_from_float b_src [| i |] (Float.of_int (10 + i))
  done;
  B.from_host ~dst:ctx ~dst_loc:loc_b b_src;
  let b_dst = Nd.create_array ~debug:"b_dst" prec ~dims ~padding:None in
  B.to_host ~src:ctx ~src_loc:loc_b b_dst;
  B.await device;
  let b_roundtrip_ok =
    Array.for_all [| 0; 1; 2; 3 |] ~f:(fun i ->
        Float.equal (Nd.get_as_float b_dst [| i |]) (Float.of_int (10 + i)))
  in
  Stdio.printf "from_host/to_host at offset>0 preserves values -> %b\n" b_roundtrip_ok;

  (* memset_zero region B at offset>0, then confirm B reads back zeros AND A is untouched. *)
  B.memset_zero device ~pool_id:pool ~offset:region_bytes ~size_in_bytes:region_bytes;
  let b_after = Nd.create_array ~debug:"b_after" prec ~dims ~padding:None in
  for i = 0 to 3 do
    Nd.set_from_float b_after [| i |] 99.0
  done;
  B.to_host ~src:ctx ~src_loc:loc_b b_after;
  let a_check = Nd.create_array ~debug:"a_check" prec ~dims ~padding:None in
  B.to_host ~src:ctx ~src_loc:loc_a a_check;
  B.await device;
  let b_zeros_ok =
    Array.for_all [| 0; 1; 2; 3 |] ~f:(fun i -> Float.equal (Nd.get_as_float b_after [| i |]) 0.0)
  in
  let a_intact =
    Array.for_all [| 0; 1; 2; 3 |] ~f:(fun i ->
        Float.equal (Nd.get_as_float a_check [| i |]) (Float.of_int (i + 1)))
  in
  Stdio.printf "memset_zero at offset>0 reads back zeros -> %b\n" b_zeros_ok;
  Stdio.printf "offset>0 ops leave region A (offset 0) intact -> %b\n" a_intact
