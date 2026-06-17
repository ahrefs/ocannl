(* Regression test for gh-ocannl-320: the Metal backend must allocate GPU-only buffers with
   private storage and CPU-accessible buffers with shared storage.

   This test pins the [storage_mode_for_memory_mode] classifier, which is the single point that
   decides private vs. shared. If the mapping regresses -- e.g. a GPU-only mode falls back to
   shared, or a host-accessible mode is mistakenly promoted to private (which would make
   [from_host] / [to_host] read or write inaccessible memory) -- the printed mode below changes
   and the test fails. *)

open Base
module Tn = Ir.Tnode
module SM = Metal.Resource.StorageMode

let string_of_storage_mode = function
  | SM.Shared -> "Shared"
  | SM.Private -> "Private"
  | SM.Managed -> "Managed"
  | SM.Memoryless -> "Memoryless"

let check label (mode : Tn.memory_mode option) =
  let sm = Metal_backend.storage_mode_for_memory_mode mode in
  Stdio.printf "%-22s -> %s\n" label (string_of_storage_mode sm)

let () =
  (* GPU-only modes: the CPU never touches these buffers, so they must be private. *)
  check "Local" (Some Tn.Local);
  check "Device_only" (Some Tn.Device_only);
  check "On_device" (Some Tn.On_device);
  (* Materialization-request / host-initialized modes: the CPU may initialize these, so they stay
     shared. After gh-ocannl-333 the [Hosted] mode is gone. *)
  check "Materialized" (Some Tn.Materialized);
  check "Effectively_constant" (Some Tn.Effectively_constant);
  (* Partially-resolved and absent modes: conservative shared default. *)
  check "Never_virtual" (Some Tn.Never_virtual);
  check "Virtual" (Some Tn.Virtual);
  check "None" None
