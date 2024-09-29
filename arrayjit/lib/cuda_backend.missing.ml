open Base

type buffer_ptr = Unimplemented_buffer_ptr [@@deriving sexp_of]
type context = Unimplemented_ctx [@@deriving sexp_of]
type code = Indexing.unit_bindings [@@deriving sexp_of]
type code_batch = Indexing.unit_bindings array [@@deriving sexp_of]
type ctx_array = | [@@deriving sexp_of]
type event = unit

let sync () = ()
let is_done () = true
let work_for _ctx _tn = Some ()
let will_wait_for _ctx () = ()
let initialize (_config : Backend_utils.Types.config) = ()
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
    Tnode.Task
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
          (Tnode.Task
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

type device = Unimplemented_dev [@@deriving sexp_of]
type physical_device = Unimplemented_phys_dev [@@deriving sexp_of]

let init Unimplemented_dev = Unimplemented_ctx
let alloc_buffer ?old_buffer:_ ~size_in_bytes:_ Unimplemented_dev = Unimplemented_buffer_ptr
let await _device = ()
let is_idle _device = true
let all_work _device = ()
let get_device ~ordinal:_ = failwith "CUDA missing: install cudajit"
let new_virtual_device Unimplemented_phys_dev = Unimplemented_dev
let get_physical_device Unimplemented_dev = Unimplemented_phys_dev
let num_physical_devices () = 0
let suggested_num_virtual_devices Unimplemented_phys_dev = 0
let get_ctx_device Unimplemented_ctx = Unimplemented_dev
let get_name Unimplemented_dev : string = failwith "CUDA missing: install cudajit"
let to_ordinal _device = 0
let to_subordinal _device = 0
let to_buffer _tn ~dst:_ ~src:_ = failwith "CUDA missing: install cudajit"
let host_to_buffer _tn ~dst:_ = failwith "CUDA missing: install cudajit"
let buffer_to_host _tn ~src:_ = failwith "CUDA missing: install cudajit"
let get_buffer _tn _context = failwith "CUDA missing: install cudajit"
