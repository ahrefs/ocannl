type procedure = Unimplemented_proc [@@deriving sexp_of]

include Backend_impl.No_device_buffer_and_copying ()

let expected_merge_node Unimplemented_proc =
  failwith "gcc backend missing: install the optional dependency gccjit"

let unified_memory = true

let to_buffer _tn ~dst:_ ~src:_ =
  failwith "gcc backend missing: install the optional dependency gccjit"

let host_to_buffer _src ~dst:_ =
  failwith "gcc backend missing: install the optional dependency gccjit"

let buffer_to_host _dst ~src:_ =
  failwith "gcc backend missing: install the optional dependency gccjit"

let alloc_buffer ?old_buffer:_ ~size_in_bytes:_ () =
  failwith "gcc backend missing: install the optional dependency gccjit"

let alloc_zero_init_array _prec ~dims:_ () =
  failwith "gcc backend missing: install the optional dependency gccjit"

let compile ~name:_ ~opt_ctx_arrays:_ _bindings _code = Unimplemented_proc

let compile_batch ~names:_ ~opt_ctx_arrays:_ _bindings _codes =
  failwith "gcc backend missing: install the optional dependency gccjit"

let link_compiled ~merge_buffer:_ _ctx_arrays Unimplemented_proc =
  failwith "gcc backend missing: install the optional dependency gccjit"

let device_to_device _tn ~into_merge_buffer:_ ~dst:_ ~src:_ =
  failwith "gcc backend missing: install the optional dependency gccjit"

let physical_merge_buffers = false
let name = "gccjit"
let initialize () = ()
let is_initialized () = true
