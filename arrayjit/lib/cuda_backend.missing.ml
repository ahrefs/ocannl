open Base

type buffer_ptr = Unimplemented_buffer_ptr [@@deriving sexp_of]
type context = Unimplemented_ctx [@@deriving sexp_of]
type code = Indexing.unit_bindings [@@deriving sexp_of]
type code_batch = Indexing.unit_bindings array [@@deriving sexp_of]
type ctx_array = | [@@deriving sexp_of]
type ctx_arrays = ctx_array Map.M(Tnode).t [@@deriving sexp_of]
type event = unit

let sync () = ()
let is_done () = true
let work_for _ctx _tn = Some ()
let will_wait_for _ctx () = ()
let initialize (_config : Backend_types.Types.config) = ()
let is_initialized () = true
let finalize _context = ()
let compile ~name:_ bindings _optimized = bindings

let compile_batch ~names:_ (bindings : Indexing.unit_bindings) optimized : code_batch =
  Array.map optimized ~f:(fun _ -> bindings)

let is_in_context _traced_array = false
let ctx_arrays Unimplemented_ctx = Map.empty (module Tnode)

let link (Unimplemented_ctx : context) (code : code) =
  let lowered_bindings = List.map ~f:(fun s -> (s, ref 0)) @@ Indexing.bound_symbols code in
  let task =
    Task.Task
      {
        context_lifetime = ();
        description = "CUDA missing: install cudajit";
        work = (fun () -> ());
      }
  in
  ((Unimplemented_ctx : context), lowered_bindings, task)

let link_batch (Unimplemented_ctx : context) (code_batch : code_batch) =
  let lowered_bindings =
    if Array.is_empty code_batch then []
    else List.map ~f:(fun s -> (s, ref 0)) @@ Indexing.bound_symbols code_batch.(0)
  in
  let task =
    Array.map code_batch ~f:(fun _ ->
        Some
          (Task.Task
             {
               context_lifetime = ();
               description = "CUDA missing: install cudajit";
               work = (fun () -> ());
             }))
  in
  ((Unimplemented_ctx : context), lowered_bindings, task)

let unsafe_cleanup () = ()
let from_host _context _tn = false
let to_host _context _tn = false
let device_to_device _tn ~into_merge_buffer:_ ~dst:_ ~src:_ = false

type stream = Unimplemented_stream [@@deriving sexp_of]
type device = Unimplemented_device [@@deriving sexp_of]

let init Unimplemented_stream = Unimplemented_ctx
let alloc_buffer ?old_buffer:_ ~size_in_bytes:_ Unimplemented_stream = Unimplemented_buffer_ptr
let await _stream = ()
let is_idle _stream = true
let all_work _stream = ()
let get_device ~ordinal:_ = failwith "CUDA missing: install cudajit"
let new_stream Unimplemented_device = Unimplemented_stream
let get_stream_device Unimplemented_stream = Unimplemented_device
let num_devices () = 0
let suggested_num_streams Unimplemented_device = 0
let get_ctx_stream Unimplemented_ctx = Unimplemented_stream
let get_name Unimplemented_stream : string = failwith "CUDA missing: install cudajit"
let to_ordinal _stream = 0
let to_subordinal _stream = 0
let name = "cuda"
