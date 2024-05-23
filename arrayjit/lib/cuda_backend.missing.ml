open Base

type context = Unimplemented [@@deriving sexp_of]
type code = Indexing.unit_bindings [@@deriving sexp_of]

let initialize () = ()
let is_initialized () = true
let finalize _context = ()
let compile ?name:_ bindings _optimized = bindings

let link (Unimplemented : context) code =
  let lowered_bindings = List.map ~f:(fun s -> (s, ref 0)) @@ Indexing.bound_symbols code in
  let work () = Tnode.Work (fun _debug_runtime () -> ()) in
  ((Unimplemented : context), lowered_bindings, work)

let unsafe_cleanup ?unsafe_shutdown:_ () = ()
let from_host ?rt:_ _context _arr = ()
let to_host ?rt:_ _context _arr = ()
let device_to_device ?rt:_ _arr ~into_merge_buffer:_ ~dst:_ ~src:_ = ()
let physical_merge_buffers = false

type device = Unimplemented [@@deriving sexp_of]

let init (Unimplemented : device) : context = Unimplemented
let await _device = ()
let acknowledge _device = ()
let is_idle _device = true
let is_booked _device = false
let num_devices () = 0
let get_device ~ordinal:_ = failwith "CUDA missing: install cudajit"
let get_ctx_device (Unimplemented : context) : device = Unimplemented
let to_ordinal (Unimplemented : device) : int = failwith "CUDA missing: install cudajit"
