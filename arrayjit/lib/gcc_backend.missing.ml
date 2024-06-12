type context = Unimplemented_ctx [@@deriving sexp_of]
type procedure = Unimplemented_proc [@@deriving sexp_of]
type ctx_arrays = Ndarray.t Base.Map.M(Tnode).t [@@deriving sexp_of]

type config = [ `Physical_devices_only | `For_parallel_copying | `Most_parallel_devices ]
[@@deriving equal, sexp, variants]

let ctx_arrays Unimplemented_ctx = Base.Map.empty (module Tnode)
let compile ~name:_ ~opt_ctx_arrays:_ _bindings _code = Unimplemented_proc

let compile_batch ~names:_ ~opt_ctx_arrays:_ _bindings _codes =
  failwith "backend missing: install the corresponding optional dependency"

let link_compiled Unimplemented_ctx Unimplemented_proc =
  failwith "backend missing: install the corresponding optional dependency"

let from_host ?rt:_ Unimplemented_ctx _tn =
  failwith "backend missing: install the corresponding optional dependency"

let to_host ?rt:_ Unimplemented_ctx _tn =
  failwith "backend missing: install the corresponding optional dependency"

let device_to_device ?rt:_ _tn ~into_merge_buffer:_ ~dst:_ ~src:_ =
  failwith "backend missing: install the corresponding optional dependency"

let physical_merge_buffers = false
let name = "gccjit"
let initialize () = ()
let is_initialized () = true
let init ~label:_ = Unimplemented_ctx
let finalize Unimplemented_ctx = ()
let unsafe_cleanup ?unsafe_shutdown:_ () = ()
