type context = Unimplemented_ctx [@@deriving sexp_of]
type procedure = Unimplemented_proc [@@deriving sexp_of]
type ctx_array = Ndarray.t [@@deriving sexp_of]
type ctx_arrays = ctx_array Base.Map.M(Tnode).t [@@deriving sexp_of]
type buffer_ptr = ctx_array [@@deriving sexp_of]

let ctx_arrays Unimplemented_ctx = Base.Map.empty (module Tnode)
let buffer_ptr ctx_array = ctx_array

let expected_merge_node Unimplemented_proc =
  failwith "gcc backend missing: install the optional dependency gccjit"

let is_in_context _node = failwith "gcc backend missing: install the optional dependency gccjit"

let to_buffer _tn ~dst:_ ~src:_ =
  failwith "gcc backend missing: install the optional dependency gccjit"

let host_to_buffer _src ~dst:_ =
  failwith "gcc backend missing: install the optional dependency gccjit"

let buffer_to_host _dst ~src:_ =
  failwith "gcc backend missing: install the optional dependency gccjit"

let alloc_buffer ?old_buffer:_ ~size_in_bytes:_ () =
  failwith "gcc backend missing: install the optional dependency gccjit"

let compile ~name:_ ~opt_ctx_arrays:_ _bindings _code = Unimplemented_proc

let compile_batch ~names:_ ~opt_ctx_arrays:_ _bindings _codes =
  failwith "gcc backend missing: install the optional dependency gccjit"

let link_compiled ~merge_buffer:_ Unimplemented_ctx Unimplemented_proc =
  failwith "gcc backend missing: install the optional dependency gccjit"

let from_host Unimplemented_ctx _tn =
  failwith "gcc backend missing: install the optional dependency gccjit"

let to_host Unimplemented_ctx _tn =
  failwith "gcc backend missing: install the optional dependency gccjit"

let device_to_device _tn ~into_merge_buffer:_ ~dst:_ ~src:_ =
  failwith "gcc backend missing: install the optional dependency gccjit"

let physical_merge_buffers = false
let name = "gccjit"
let initialize () = ()
let is_initialized () = true
let init _ = Unimplemented_ctx
let finalize Unimplemented_ctx = ()
