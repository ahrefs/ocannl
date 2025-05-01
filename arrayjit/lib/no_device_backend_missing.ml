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

  let link_compiled ~merge_buffer:_ ~runner_label:_ _ctx_arrays _procedure =
    failwith @@ "Backend " ^ Config.name ^ " missing (no device)"

  let sexp_of_buffer_ptr _buffer_ptr = failwith @@ "Backend " ^ Config.name ^ " missing (no device)"

  type nonrec buffer = buffer_ptr Backend_intf.buffer

  let sexp_of_buffer _buffer = failwith @@ "Backend " ^ Config.name ^ " missing (no device)"

  type nonrec ctx_arrays = buffer_ptr Backend_intf.ctx_arrays

  let sexp_of_ctx_arrays _ctx_arrays = failwith @@ "Backend " ^ Config.name ^ " missing (no device)"

  let alloc_buffer ?old_buffer:_ ~size_in_bytes:_ () =
    failwith @@ "Backend " ^ Config.name ^ " missing (no device)"

  let alloc_zero_init_array _prec ~dims:_ () =
    failwith @@ "Backend " ^ Config.name ^ " missing (no device)"

  let free_buffer = None
  let get_used_memory () = failwith @@ "Backend " ^ Config.name ^ " missing (no device)"

  let buffer_to_buffer ~dst:_ ~src:_ ~size_in_bytes:_ =
    failwith @@ "Backend " ^ Config.name ^ " missing (no device)"

  let host_to_buffer _nd = failwith @@ "Backend " ^ Config.name ^ " missing (no device)"
  let buffer_to_host _nd = failwith @@ "Backend " ^ Config.name ^ " missing (no device)"
end
