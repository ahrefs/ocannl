open Ir

type buffer_ptr
type dev
type runner
type event

let use_host_memory = None
let sexp_of_dev _dev = failwith "Backend missing -- install the corresponding library"
let sexp_of_runner _runner = failwith "Backend missing -- install the corresponding library"
let sexp_of_event _event = failwith "Backend missing -- install the corresponding library"
let name = "Backend missing"

type nonrec device = (buffer_ptr, dev, runner, event) Backend_intf.device

let sexp_of_device _device = failwith "Backend missing -- install the corresponding library"

type nonrec stream = (buffer_ptr, dev, runner, event) Backend_intf.stream

let sexp_of_stream _stream = failwith "Backend missing -- install the corresponding library"

type nonrec context = (buffer_ptr, stream) Backend_intf.context

let sexp_of_context _context = failwith "Backend missing -- install the corresponding library"

let alloc_buffer ?old_buffer:_ ~size_in_bytes:_ _stream =
  failwith "Backend missing -- install the corresponding library"

let alloc_zero_init_array _prec ~dims:_ _stream =
  failwith "Backend missing -- install the corresponding library"

let free_buffer = None
let make_device _dev = failwith "Backend missing -- install the corresponding library"
let make_stream _device = failwith "Backend missing -- install the corresponding library"

let make_context ?ctx_arrays:_ _stream =
  failwith "Backend missing -- install the corresponding library"

let make_child ?ctx_arrays:_ _context =
  failwith "Backend missing -- install the corresponding library"

let get_name _stream = failwith "Backend missing -- install the corresponding library"
let sexp_of_buffer_ptr _buffer_ptr = failwith "Backend missing -- install the corresponding library"

type nonrec buffer = buffer_ptr Backend_intf.buffer

let sexp_of_buffer _buffer = failwith "Backend missing -- install the corresponding library"

type nonrec ctx_arrays = buffer_ptr Backend_intf.ctx_arrays

let sexp_of_ctx_arrays _ctx_arrays = failwith "Backend missing -- install the corresponding library"
let sync _event = failwith "Backend missing -- install the corresponding library"
let is_done _event = failwith "Backend missing -- install the corresponding library"
let will_wait_for _context = failwith "Backend missing -- install the corresponding library"
let get_used_memory _device = failwith "Backend missing -- install the corresponding library"
let get_global_debug_info () = failwith "Backend missing -- install the corresponding library"
let get_debug_info _stream = failwith "Backend missing -- install the corresponding library"
let await _stream = failwith "Backend missing -- install the corresponding library"
let all_work _stream = failwith "Backend missing -- install the corresponding library"
let is_idle _stream = failwith "Backend missing -- install the corresponding library"
let get_device ~ordinal:_ = failwith "Backend missing -- install the corresponding library"
let num_devices () = 0
let suggested_num_streams _device = failwith "Backend missing -- install the corresponding library"
let new_stream _device = failwith "Backend missing -- install the corresponding library"

let from_host ~dst_ptr:_ ~dst:_ _nd =
  failwith "Backend missing -- install the corresponding library"

let to_host ~src_ptr:_ ~src:_ _nd = failwith "Backend missing -- install the corresponding library"
let device_to_device _tn = failwith "Backend missing -- install the corresponding library"

type code

let sexp_of_code _code = failwith "Backend missing -- install the corresponding library"

type code_batch

let sexp_of_code_batch _code_batch = failwith "Backend missing -- install the corresponding library"

let compile ~name:_ _unit_bindings _optimized =
  failwith "Backend missing -- install the corresponding library"

let compile_batch ~names:_ _unit_bindings _optimizeds =
  failwith "Backend missing -- install the corresponding library"

let link _context _code = failwith "Backend missing -- install the corresponding library"

let link_batch _context _code_batch =
  failwith "Backend missing -- install the corresponding library"
