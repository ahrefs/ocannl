(* Regression test for gh-ocannl-344: the Metal pool allocator commits to a single storage mode --
   [Shared] -- for every pool, regardless of the tnode memory mode passed to [alloc_pool]. The
   per-tnode private/shared classifier (gh-ocannl-320's [storage_mode_for_memory_mode]) is retired;
   one device-wide storage mode is what lets a context delta pack into one pool without splitting
   across modes, and lets host transfers be a direct memcpy.

   The invariant pinned here: [storage_mode_of_pool] returns [Shared] for a pool allocated under ANY
   memory mode (including the GPU-only modes that previously selected [Private]). If a backend edit
   reintroduced per-mode selection, the GPU-only lines below would print [Private] and the test
   would fail. *)

open Base
module Tn = Ir.Tnode
module SM = Metal.Resource.StorageMode
module B = Metal_backend.Fresh ()

let string_of_storage_mode = function
  | SM.Shared -> "Shared"
  | SM.Private -> "Private"
  | SM.Managed -> "Managed"
  | SM.Memoryless -> "Memoryless"

let () =
  let device = B.get_device ~ordinal:0 in
  let next_pool =
    ref 1
    (* pool 0 is the reserved merge buffer *)
  in
  let check label (mode : Tn.memory_mode option) =
    let pool_id = !next_pool in
    Int.incr next_pool;
    B.alloc_pool ?mode device ~pool_id ~size_in_bytes:16 ~alignment:1;
    Stdio.printf "%-22s -> %s\n" label
      (string_of_storage_mode (B.storage_mode_of_pool device ~pool_id))
  in
  (* GPU-only modes that used to select Private now allocate Shared like everything else. *)
  check "Local" (Some Tn.Local);
  check "Device_only" (Some Tn.Device_only);
  check "On_device" (Some Tn.On_device);
  check "Materialized" (Some Tn.Materialized);
  check "Effectively_constant" (Some Tn.Effectively_constant);
  check "Never_virtual" (Some Tn.Never_virtual);
  check "Virtual" (Some Tn.Virtual);
  check "None" None
