open Ir

module Missing (Config : sig
  val name : string
end) =
struct
  type buffer_ptr

  let use_host_memory = None
  let name = Config.name

  type procedure

  let sexp_of_procedure _procedure = failwith @@ "Backend " ^ Config.name ^ " missing (no device)"

  let compile ~name:_ _unit_bindings _optimized =
    failwith @@ "Backend " ^ Config.name ^ " missing (no device)"

  let compile_batch ~names:_ _unit_bindings _optimizeds =
    failwith @@ "Backend " ^ Config.name ^ " missing (no device)"

  let link_compiled ~merge_buffer:_ ~resolve:_ ~runner_label:_ _ctx_arrays _procedure =
    failwith @@ "Backend " ^ Config.name ^ " missing (no device)"

  let sexp_of_buffer_ptr _buffer_ptr = failwith @@ "Backend " ^ Config.name ^ " missing (no device)"

  type nonrec buffer = buffer_ptr Backend_intf.buffer

  let sexp_of_buffer _buffer = failwith @@ "Backend " ^ Config.name ^ " missing (no device)"

  let alloc_pool_raw ~size_in_bytes:_ =
    failwith @@ "Backend " ^ Config.name ^ " missing (no device)"

  let free_pool_raw = None

  let memset_zero_raw _ptr ~offset:_ ~size_in_bytes:_ =
    failwith @@ "Backend " ^ Config.name ^ " missing (no device)"

  let get_used_memory () = failwith @@ "Backend " ^ Config.name ^ " missing (no device)"

  let buffer_to_buffer ~dst:_ ~src:_ ~size_in_bytes:_ =
    failwith @@ "Backend " ^ Config.name ^ " missing (no device)"

  let host_to_buffer _nd = failwith @@ "Backend " ^ Config.name ^ " missing (no device)"
  let buffer_to_host _nd = failwith @@ "Backend " ^ Config.name ^ " missing (no device)"
end
