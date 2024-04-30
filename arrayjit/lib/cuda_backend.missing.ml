open Base

type context = Unimplemented [@@deriving sexp_of]
type code = Indexing.unit_bindings [@@deriving sexp_of]

let initialize () = ()
let is_initialized () = true
let finalize _context = ()

let compile ?name:_ bindings _optimized = bindings

let link context code =
  let compiled_bindings = List.map ~f:(fun s -> (s, ref 0)) @@ Indexing.bound_symbols code in
  let work () = Tnode.Work (fun _debug_runtime () -> ()) in
  (context, compiled_bindings, work)

let unsafe_cleanup ?unsafe_shutdown:_ () = ()
let from_host _context _arr = false
let to_host _context _arr = false
let merge ?name_suffix:_ _arr ~dst:_ ~accum:_ ~src:_ _bindings = None

type device = Unimplemented [@@deriving sexp_of]

let init _device = failwith "CUDA missing: install cudajit"
let await _device = ()
let num_devices () = 1
let get_device ~ordinal:_ = failwith "CUDA missing: install cudajit"
let get_ctx_device _context = failwith "CUDA missing: install cudajit"
let to_ordinal _device = 0
