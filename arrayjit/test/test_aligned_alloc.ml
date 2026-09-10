(* gh-ocannl-164: CPU pool slabs must start on an [Ops.buffer_alignment] boundary (AVX/NEON vector
   loads). [Ctypes.allocate_n] (calloc) only guarantees the ABI's ~16 bytes; [alloc_pool_raw]
   over-allocates and advances to the boundary, preserving ctypes' managed-root GC semantics. Odd
   sizes exercise the padding math; allocating several buffers back-to-back makes it overwhelmingly
   unlikely that all bases are aligned by accident. *)

open Base
module M = Ir.Backend_impl.No_device_buffer_and_copying ()

let () =
  let align = Ir.Ops.buffer_alignment in
  let aligned size_in_bytes =
    let ptr = M.alloc_pool_raw ~size_in_bytes in
    let addr = Ctypes.raw_address_of_ptr ptr in
    Nativeint.(addr % of_int align = 0n)
  in
  let sizes = [ 1; 3; 8; 31; 32; 33; 100; 1023; 4096; 65537 ] in
  Verdict.p_all (Printf.sprintf "all pool bases %d-byte aligned" align) sizes ~f:aligned
