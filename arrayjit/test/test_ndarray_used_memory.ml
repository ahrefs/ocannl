(* [Ndarray.get_used_memory] is a LIVE gauge of host-array bytes, not a cumulative allocation total:
   [create_array] adds the array's size and the finalizer it registers gives those bytes back when
   the array is collected. This pins both halves.

   Both used to be wrong. The finalizer added instead of subtracting, so the gauge only ever grew
   and each collected array was counted twice; and [init_array] added a second time (and registered
   a second finalizer) on top of the [create_array] it delegates to, so its arrays were
   double-counted. Either bug makes "gauge returned to baseline after GC" print [false].

   GC finalizers are asynchronous in principle, so the post-drop reading is taken after a
   [Gc.full_major] and compared coarsely -- within one array's worth of bytes -- rather than pinned
   exactly. *)

open Base
module Nd = Ir.Ndarray
module Ops = Ir.Ops

(* Allocates [n] arrays with [alloc], reads the gauge while they are held, drops them, and reads it
   again after a full major collection. Returns (delta while held, residual after collection). *)
let alloc_hold_drop ~n ~alloc =
  let before = Nd.get_used_memory () in
  let arrays = ref (List.init n ~f:(fun _ -> alloc ())) in
  let while_held = Nd.get_used_memory () in
  arrays := [];
  Stdlib.Gc.full_major ();
  Stdlib.Gc.full_major ();
  let after = Nd.get_used_memory () in
  (while_held - before, after - before)

let check name ~n ~bytes_per_array ~alloc =
  Verdict.case name @@ fun () ->
  Stdlib.Gc.full_major ();
  Stdlib.Gc.full_major ();
  let held, residual = alloc_hold_drop ~n ~alloc in
  Stdio.printf "%s: %d arrays of %d bytes each\n" name n bytes_per_array;
  Verdict.p "  delta while held = allocated bytes" (held = n * bytes_per_array);
  (* Coarse: one array's worth of slack, so that a straggling finalizer is not a failure but a
     double-counted or never-subtracted allocation still is. *)
  Verdict.p "  gauge returned to baseline after GC" (abs residual <= bytes_per_array)

let () =
  let prec = Ops.single in
  let bytes_of dims = Array.fold dims ~init:1 ~f:( * ) * Ops.prec_in_bytes prec in
  let create_dims = [| 128; 1024 |] in
  check "create_array" ~n:4 ~bytes_per_array:(bytes_of create_dims) ~alloc:(fun () ->
      Nd.create_array ~debug:"used_memory" prec ~dims:create_dims ~padding:None);
  (* [init_array] is slow (unboxing at each index), hence the smaller arrays. *)
  let init_dims = [| 32; 256 |] in
  check "init_array" ~n:3 ~bytes_per_array:(bytes_of init_dims) ~alloc:(fun () ->
      Nd.init_array ~debug:"used_memory" prec ~dims:init_dims ~padding:None ~f:(fun _ -> 1.0));
  (* A reshaped view shares the source's bytes, so accounting must outlive the source wrapper: the
     bytes are still held while only the view is reachable. [Tnode.create_with_reshape] is exactly
     that situation -- it hands out the view and drops the array it reshaped. *)
  Verdict.case "reshape" (fun () ->
      Stdlib.Gc.full_major ();
      Stdlib.Gc.full_major ();
      let view_dims = [| 64; 512 |] in
      let view_bytes = bytes_of view_dims in
      let make_view () =
        (* [source] is dead once this returns, so only the view keeps the bytes reachable. *)
        let source = Nd.create_array ~debug:"used_memory" prec ~dims:[| 32768 |] ~padding:None in
        Nd.reshape source view_dims
      in
      let before = Nd.get_used_memory () in
      let cell = ref (Some (make_view ())) in
      Stdlib.Gc.full_major ();
      Stdlib.Gc.full_major ();
      (* The view must be read {e after} the collection, not merely stored: a value whose last use
         has passed is dead to the GC (OCaml's liveness ignores a [ref] that is only assigned from
         here on), and collecting the view would make this case pass vacuously. *)
      let view_rank = match !cell with Some view -> Array.length (Nd.dims view) | None -> 0 in
      let held = Nd.get_used_memory () - before in
      Stdio.printf "reshape: 1 array of %d bytes\n" view_bytes;
      Verdict.p "  still counted while only the view is held"
        (view_rank = Array.length view_dims && held = view_bytes);
      cell := None;
      Stdlib.Gc.full_major ();
      Stdlib.Gc.full_major ();
      Verdict.p "  gauge returned to baseline after GC"
        (abs (Nd.get_used_memory () - before) <= view_bytes))
