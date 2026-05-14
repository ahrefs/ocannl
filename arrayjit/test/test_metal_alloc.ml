(* Regression test for gh-ocannl-320: exercises the *real* Metal allocator paths (not just the
   pure classifier in test_metal_storage_mode.ml).

   It allocates buffers through [alloc_array] / [alloc_zeros] / [alloc_buffer] and asserts the
   storage mode of the resulting [Metal.Buffer.t] via [Metal.Resource.get_storage_mode]. If any
   allocation call site stopped threading [?mode], or [resource_options_for_mode] stopped being
   used, a GPU-only buffer would come back [Shared] and the printed line would change.

   It also drives data through a private-mode buffer end to end: [from_host] in, [to_host] back
   out, and a private [alloc_zeros], verifying the actual values. A broken private-buffer blit
   path (or a private zero-fill that silently no-ops) would change the [true] verdicts below. *)

open Base
module Tn = Ir.Tnode
module Nd = Ir.Ndarray
module Ops = Ir.Ops
module Me = Metal
module SM = Me.Resource.StorageMode
module B = Metal_backend.Fresh ()

let sm_of_ptr ptr = Me.Resource.get_storage_mode (Me.Buffer.super ptr)

let sm_str = function
  | SM.Shared -> "Shared"
  | SM.Private -> "Private"
  | SM.Managed -> "Managed"
  | SM.Memoryless -> "Memoryless"

let prec = Ops.single
let dims = [| 4 |]

let () =
  let dev = B.get_device ~ordinal:0 in
  let stream = B.new_stream dev in

  (* --- alloc_array: storage mode follows the tnode memory mode --- *)
  let check_array label mode =
    Stdio.printf "alloc_array %-13s -> %s\n" label
      (sm_str (sm_of_ptr (B.alloc_array ?mode prec ~dims stream)))
  in
  check_array "Local" (Some Tn.Local);
  check_array "Device_only" (Some Tn.Device_only);
  check_array "On_device" (Some Tn.On_device);
  check_array "Hosted" (Some (Tn.Hosted Tn.Nonconstant));
  check_array "Materialized" (Some Tn.Materialized);
  check_array "(no mode)" None;

  (* --- alloc_zeros: GPU-only modes still get private storage --- *)
  let zeros_priv = B.alloc_zeros ~mode:Tn.Device_only prec ~dims stream in
  Stdio.printf "alloc_zeros Device_only   -> %s\n" (sm_str (sm_of_ptr zeros_priv));

  (* --- alloc_buffer: storage mode + reuse guard --- *)
  let shared_buf = B.alloc_buffer ~mode:(Tn.Hosted Tn.Nonconstant) ~size_in_bytes:64 stream in
  let priv_buf = B.alloc_buffer ~mode:Tn.Device_only ~size_in_bytes:64 stream in
  Stdio.printf "alloc_buffer Hosted       -> %s\n" (sm_str (sm_of_ptr shared_buf.ptr));
  Stdio.printf "alloc_buffer Device_only  -> %s\n" (sm_str (sm_of_ptr priv_buf.ptr));
  (* Reuse with a matching storage mode: the old buffer is handed back. *)
  let reuse_match =
    B.alloc_buffer ~old_buffer:priv_buf ~mode:Tn.Device_only ~size_in_bytes:32 stream
  in
  Stdio.printf "alloc_buffer reuse same-mode keeps ptr     -> %b\n"
    (phys_equal reuse_match.ptr priv_buf.ptr);
  (* Reuse with a mismatched storage mode: a shared buffer must NOT be handed back for a
     private-mode request; a fresh private buffer is allocated instead. *)
  let reuse_mismatch =
    B.alloc_buffer ~old_buffer:shared_buf ~mode:Tn.Device_only ~size_in_bytes:32 stream
  in
  Stdio.printf "alloc_buffer reuse mismatched-mode is fresh -> %b\n"
    (not (phys_equal reuse_mismatch.ptr shared_buf.ptr));
  Stdio.printf "alloc_buffer reuse mismatched-mode storage  -> %s\n"
    (sm_str (sm_of_ptr reuse_mismatch.ptr));

  (* --- from_host / to_host round-trip through a private-mode buffer --- *)
  let ctx = B.make_context stream in
  let src = Nd.create_array ~debug:"src" prec ~dims ~padding:None in
  for i = 0 to 3 do
    Nd.set_from_float src [| i |] (Float.of_int (i + 1))
  done;
  let dev_buf = B.alloc_array ~mode:Tn.Device_only prec ~dims stream in
  B.from_host ~dst_ptr:dev_buf ~dst:ctx src;
  let dst = Nd.create_array ~debug:"dst" prec ~dims ~padding:None in
  B.to_host ~src_ptr:dev_buf ~src:ctx dst;
  B.await stream;
  let roundtrip_ok =
    Array.for_all [| 0; 1; 2; 3 |] ~f:(fun i ->
        Float.equal (Nd.get_as_float dst [| i |]) (Float.of_int (i + 1)))
  in
  Stdio.printf "from_host/to_host private round-trip preserves values -> %b\n" roundtrip_ok;

  (* --- a private alloc_zeros buffer actually reads back as zeros --- *)
  let zdst = Nd.create_array ~debug:"zdst" prec ~dims ~padding:None in
  for i = 0 to 3 do
    Nd.set_from_float zdst [| i |] 99.0
  done;
  B.to_host ~src_ptr:zeros_priv ~src:ctx zdst;
  B.await stream;
  let zeros_ok =
    Array.for_all [| 0; 1; 2; 3 |] ~f:(fun i -> Float.equal (Nd.get_as_float zdst [| i |]) 0.0)
  in
  Stdio.printf "alloc_zeros private buffer reads back zeros -> %b\n" zeros_ok
