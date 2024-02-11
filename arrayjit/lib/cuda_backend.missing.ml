open Base

type context

let initialize () = ()
let is_initialized () = true
let finalize _context = ()
let sexp_of_context _context = failwith "CUDA missing: install cudajit"

let jit ?name:_ context bindings _asgns =
  let jitted_bindings = List.map ~f:(fun s -> (s, ref 0)) @@ Indexing.bound_symbols bindings in
  (context, jitted_bindings, fun () -> ())

let unsafe_cleanup ?unsafe_shutdown:_ () = ()
let from_host _context _arr = false
let to_host _context _arr = false
let merge ?name_suffix:_ _arr ~dst:_ ~accum:_ ~src:_ _bindings = None

type device

let init _device = failwith "CUDA missing: install cudajit"
let await _device = ()
let sexp_of_device _device = failwith "CUDA missing: install cudajit"
let num_devices () = 1
let get_device ~ordinal:_ = failwith "CUDA missing: install cudajit"
let get_ctx_device _context = failwith "CUDA missing: install cudajit"
let to_ordinal _device = 0
