type buffer_ptr

let use_host_memory = false
let initialize _config = failwith "Backend missing -- install the corresponding library"
let is_initialized () = failwith "Backend missing -- install the corresponding library"
let name = "Backend missing"

type procedure

let sexp_of_procedure _procedure = failwith "Backend missing -- install the corresponding library"

let compile ~name:_ ~opt_ctx_arrays:_ _unit_bindings _optimized =
  failwith "Backend missing -- install the corresponding library"

let compile_batch ~names:_ ~opt_ctx_arrays:_ _unit_bindings _optimizeds =
  failwith "Backend missing -- install the corresponding library"

let link_compiled ~merge_buffer:_ ~runner_label:_ _ctx_arrays _procedure =
  failwith "Backend missing -- install the corresponding library"

let sexp_of_buffer_ptr _buffer_ptr = failwith "Backend missing -- install the corresponding library"
let c_ptr_to_string = None

type nonrec buffer = buffer_ptr Backend_intf.buffer

let sexp_of_buffer _buffer = failwith "Backend missing -- install the corresponding library"

type nonrec ctx_arrays = buffer_ptr Backend_intf.ctx_arrays

let sexp_of_ctx_arrays _ctx_arrays = failwith "Backend missing -- install the corresponding library"

let alloc_buffer ?old_buffer:_ ~size_in_bytes:_ () =
  failwith "Backend missing -- install the corresponding library"

let alloc_zero_init_array _prec ~dims:_ () =
  failwith "Backend missing -- install the corresponding library"

let free_buffer = None
let get_used_memory () = failwith "Backend missing -- install the corresponding library"

let buffer_to_buffer ~dst:_ ~src:_ ~size_in_bytes:_ =
  failwith "Backend missing -- install the corresponding library"

let host_to_buffer _nd = failwith "Backend missing -- install the corresponding library"
let buffer_to_host _nd = failwith "Backend missing -- install the corresponding library"
