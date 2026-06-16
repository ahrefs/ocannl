(* Regression test for the reserved merge-pool grow path (AC6/AC7): when [alloc_pool] is called for a
   pool id that already has a slab (only the reserved merge pool, id 0, is ever re-allocated in
   place), the previous backend allocation must be freed before it is replaced -- otherwise device
   memory grows without bound on every merge-buffer grow.

   The CUDA backend ([cuda_backend.ml] [Slab.alloc_pool]) and the shared [Make_slab.alloc_pool] use
   the identical free-on-overwrite pattern. CUDA is not buildable in this harness (no cudajit), so we
   pin the invariant through [Make_slab] with a mock raw backend whose [free_pool_raw] is [Some]
   (i.e. a backend that owns explicitly-freed pointers, like CUDA). The assertion
   "grow freed the old pool = true" would print [false] if [alloc_pool] overwrote the table entry
   without freeing -- the exact bug this fixes. A unique tnode pool id (never pre-existing) must free
   nothing. *)

open Base
module Backend_impl = Ir.Backend_impl
module Backend_intf = Ir.Backend_intf

(* A raw backend whose "pointers" are integer ids and whose [free_pool_raw] records frees -- standing
   in for a backend (like CUDA) that owns explicitly-released device pointers. *)
module Mock_raw = struct
  type buffer_ptr = int

  let sexp_of_buffer_ptr = Int.sexp_of_t

  include Backend_impl.Buffer_types (struct
    type nonrec buffer_ptr = buffer_ptr

    let sexp_of_buffer_ptr = sexp_of_buffer_ptr
  end)

  let use_host_memory = None
  let get_used_memory () = 0
  let next = ref 0
  let freed : int list ref = ref []

  let alloc_pool_raw ~size_in_bytes:_ =
    Int.incr next;
    !next

  let free_pool_raw = Some (fun ptr -> freed := ptr :: !freed)
  let memset_zero_raw _ptr ~offset:_ ~size_in_bytes:_ = ()
  let buffer_to_buffer ~dst:_ ~src:_ ~size_in_bytes:_ = ()
  let host_to_buffer _nd ~dst:_ = ()
  let buffer_to_host _nd ~src:_ = ()
end

module Mock_config = struct
  type dev = unit
  type runner = unit
  type event = unit

  let sexp_of_dev = Base.sexp_of_unit
  let sexp_of_runner = Base.sexp_of_unit
  let sexp_of_event = Base.sexp_of_unit
  let name = "mock"
end

module Mock_dt = Backend_impl.Device_types_ll (Mock_config)
module Mock_slab = Backend_impl.Make_slab (Mock_dt) (Mock_raw)
module Mock_dev = Backend_impl.Device (Mock_dt) (Mock_slab)

(* A raw backend that relies on GC (no explicit deallocator), like the sync/multicore CPU backends. *)
module Mock_raw_gc = struct
  type buffer_ptr = int

  let sexp_of_buffer_ptr = Int.sexp_of_t

  include Backend_impl.Buffer_types (struct
    type nonrec buffer_ptr = buffer_ptr

    let sexp_of_buffer_ptr = sexp_of_buffer_ptr
  end)

  let use_host_memory = None
  let get_used_memory () = 0
  let next = ref 100

  let alloc_pool_raw ~size_in_bytes:_ =
    Int.incr next;
    !next

  let free_pool_raw = None (* relies on GC + the dropped table entry *)
  let memset_zero_raw _ptr ~offset:_ ~size_in_bytes:_ = ()
  let buffer_to_buffer ~dst:_ ~src:_ ~size_in_bytes:_ = ()
  let host_to_buffer _nd ~dst:_ = ()
  let buffer_to_host _nd ~src:_ = ()
end

module Mock_gc_slab = Backend_impl.Make_slab (Mock_dt) (Mock_raw_gc)
module Mock_gc_dev = Backend_impl.Device (Mock_dt) (Mock_gc_slab)

let loc pool_id : Backend_intf.buffer_loc = { pool_id; offset = 0 }

let () =
  let device = Mock_dev.make_device () () ~ordinal:0 in
  (* Reserved merge pool (id 0): allocate, then grow it in place (re-allocate the same key). *)
  Mock_slab.alloc_pool device ~pool_id:0 ~size_in_bytes:16 ~alignment:1;
  let p1 = Mock_slab.resolve_pool device (loc 0) in
  Mock_slab.alloc_pool device ~pool_id:0 ~size_in_bytes:32 ~alignment:1;
  let p2 = Mock_slab.resolve_pool device (loc 0) in
  Stdio.printf "grow freed the old pool = %b\n" (List.mem !Mock_raw.freed p1 ~equal:Int.equal);
  Stdio.printf "grow installed a new pool = %b\n" (not (p1 = p2));
  Stdio.printf "freed count after grow = %d\n" (List.length !Mock_raw.freed);
  (* A unique tnode pool id never pre-exists, so allocating it frees nothing. *)
  Mock_slab.alloc_pool device ~pool_id:1 ~size_in_bytes:16 ~alignment:1;
  Stdio.printf "freed count after unique-id alloc = %d\n" (List.length !Mock_raw.freed);

  (* free_pool must drop the private table entry even for a GC-reliant backend (free_pool_raw =
     None), so the strong reference is released and the buffer can be reclaimed. If free_pool were
     [None] (the bug), [finalize] would never remove these entries. *)
  let gc_device = Mock_gc_dev.make_device () () ~ordinal:0 in
  Mock_gc_slab.alloc_pool gc_device ~pool_id:7 ~size_in_bytes:16 ~alignment:1;
  Stdio.printf "gc backend exposes free_pool (not None) = %b\n"
    (Option.is_some Mock_gc_slab.free_pool);
  let present_before =
    try
      ignore (Mock_gc_slab.resolve_pool gc_device (loc 7) : int);
      true
    with _ -> false
  in
  Option.iter Mock_gc_slab.free_pool ~f:(fun free -> free gc_device ~pool_id:7);
  let present_after =
    try
      ignore (Mock_gc_slab.resolve_pool gc_device (loc 7) : int);
      true
    with _ -> false
  in
  Stdio.printf "gc backend entry present before free = %b, after free = %b\n" present_before
    present_after
