open Base

let optimization_level = ref 3

(* let session_context = *)

type sync_properties =
  | Thread_only  (** Thread-local tensor. *)
  | Block_only  (** Shared memory tensor. *)
  | Update_globally_for_thread
      (** All assignments are update assignments. They happen directly in global memory, simultaneously
          syncing the tensor's cell value for a thread. *)
  | Update_globally_for_block
      (** All assignments are update assignments. They happen directly in global memory, simultaneously
          syncing the tensor's cell value for a block (via a shared array). *)
  | Constant
      (** This tensor is accessed directly in the global memory but is not modified by the step update. *)
  | Thread_parallel
      (** Each thread computes a slice of the tensor, independently transferring to global memory. *)
  | Block_parallel  (** Each block computes a slice of the tensor. *)
  | Global  (** This tensor is accessed directly in the global memory, we did not manage to optimize it. *)
[@@deriving sexp, equal, compare, variants]

type tensor = {
  hosted : Node.ndarray option;  (** Pointer to the first value of the associated [Bigarray], if hosted. *)
  global : string option;  (** A global device array, if any. *)
  global_ptr : Cudajit.deviceptr option;
  local : string option;  (** Either a thread-local or shared (block-local) memory, if any. *)
  sync : sync_properties;
  host_dims : int array;
      (** Dimensions (shape) of the tensor as a whole, or an empty array if [hosted_ptr]
                              is [None]. *)
  device_dims : int array;  (** Dimensions (shape) of the per-task slice of the tensor. *)
  host_size_in_bytes : int;  (** Size of the full host's tensor. *)
  device_size_in_bytes : int;  (** Size of the per-task slice of the tensor. *)
  device_length : int;  (** Number of elements in the per-task slice of the tensor. *)
  host_offset : (unit -> int) option;
      (** The offset of the device slice wrt. to the beginning of the host, in number of elements. If [None],
          the device tensor is the full host tensor. *)
  local_is_slice_of_host : bool;
      (** If true, the local tensor is a slice of the host tensor. If false, the local tensor needs to be
          iterated with a jitted code to copy it to/from the host tensor. *)
  num_typ : string;
      (** The type of the stored values: [char] (corresponds to precision [Byte]),
      [short] (precision [Half]), [float] (precision [Single]), [double] (precision [Double]). *)
  is_double : bool;
}

(* let session_results = ref [] *)
let hoist_dynamic_indices = ref false

type session_state = {
  mutable ctx : Cudajit.context option;
  tensors : (NodeUI.tensor_ptr, tensor) Hashtbl.Poly.t;
  mutable last_module : Cudajit.module_ option;
  mutable num_blocks : int;
  mutable num_threads : int;
}

let session_state =
  { ctx = None; tensors = Hashtbl.Poly.create (); last_module = None; num_blocks = 1; num_threads = 1 }

let pp_semi ppf () = Caml.Format.fprintf ppf ";@ "
let pp_comma ppf () = Caml.Format.fprintf ppf ",@ "
let pp_symbol ppf (Shape.Symbol s) = Caml.Format.fprintf ppf "i%d" s
let pp_index ppf sym = Caml.Format.fprintf ppf "%s" @@ Shape.symbol_ident sym

let pp_index_axis ?provider_dim ppf =
  let open Shape in
  function
  | Iterator it | Frozen_recipient it | Dynamic_recipient it -> pp_index ppf it
  | Fixed_idx i -> Caml.Format.fprintf ppf "%d" i
  | Dynamic_provider _ -> Caml.Format.fprintf ppf "%d" @@ Option.value_exn provider_dim

let pp_get_ptr ppf tensor =
  match (tensor.global, tensor.local) with
  | _, Some lv -> Caml.Format.pp_print_string ppf lv
  | Some rv, _ -> Caml.Format.pp_print_string ppf rv
  | None, None -> assert false

let prec_to_c_type = function
  | NodeUI.Void_prec -> "void"
  | Byte_as_int_prec _ -> "char"
  | Half_as_int_prec _ -> "short"
  | Single_prec _ -> "float"
  | Double_prec _ -> "double"

let prec_is_double = function NodeUI.Double_prec _ -> true | _ -> false

exception Unknown_synchronization

let compute_array_offset ~idcs ~dims =
  Array.fold2_exn idcs dims ~init:0 ~f:(fun offset idx dim -> idx + (offset * dim))

let get_tensor ~traced_store ?force_sync ~jit_code ~dyn_env ~host_idcs ptr : tensor =
  let { tensors; _ } = session_state in
  Hashtbl.find_or_add tensors ptr ~default:(fun () ->
      let n = NodeUI.(get ptr.id) in
      let tn = Code.(get_node traced_store ptr) in
      let host_size_in_bytes = NodeUI.host_size_in_bytes ptr in
      let axes = Shape.to_dims n.shape in
      let device_dims = axes |> Array.map ~f:(fun d -> d.dim) in
      let device_length = Array.fold ~init:1 ~f:( * ) device_dims in
      let arr = Option.value_exn @@ NodeUI.get_tensor ptr in
      let hosted = if Array.is_empty @@ Node.dims arr then None else Some arr in
      let device_size_in_bytes = device_length * Node.precision_in_bytes arr in
      let local_is_slice_of_host =
        Array.fold_until axes ~init:true
          ~f:
            (fun was_frozen -> function
              (* Currently for Cuda, parallel dims are part of a slice. *)
              | Shape.{ special = Frozen; _ } -> if was_frozen then Continue true else Stop false
              | _ -> Continue false)
          ~finish:(fun _ -> true)
      in
      let tensor prec is_double arr =
        let num_typ = prec_to_c_type prec in
        let is_block_parallel =
          Array.exists ~f:Shape.(function { special = Dedicated Task_id; _ } -> true | _ -> false)
          @@ Shape.to_dims n.shape
        in
        let is_thread_parallel =
          Array.exists ~f:Shape.(function { special = Dedicated Sample_num; _ } -> true | _ -> false)
          @@ Shape.to_dims n.shape
        in
        let can_be_replicated = tn.is_replicable in
        let computed_directly_across_blocks =
          List.exists tn.rhses ~f:(Code.check_dedicated_dep Shape.Task_id ~cached_dedicated:(fun _ -> false))
        in
        (* tn.is_replicable is the negation of: computed (directly or) indirectly across blocks. *)
        let computed_directly_across_threads =
          List.exists tn.rhses
            ~f:(Code.check_dedicated_dep Shape.Sample_num ~cached_dedicated:(fun _ -> false))
        in
        let update_globally =
          (not is_block_parallel) && (not tn.is_replicable) && tn.read_before_write && tn.reduced_racyness
        in
        let sync =
          Option.value_or_thunk force_sync ~default:(fun () ->
              if
                Option.is_none hosted
                && (is_block_parallel || not computed_directly_across_blocks)
                && (is_thread_parallel || not computed_directly_across_threads)
              then Thread_only
              else if Option.is_none hosted && (is_block_parallel || not computed_directly_across_blocks) then
                Block_only
              else if is_block_parallel && is_thread_parallel then Thread_parallel
              else if is_block_parallel then Block_parallel
              else if update_globally && not is_thread_parallel then Update_globally_for_thread
              else if update_globally then Update_globally_for_block
              else if can_be_replicated then Global
              else (
                if !Code.with_debug then
                  Caml.Format.printf "\nWARNING: No sync for tensor: %a@ node: %a\n%!" Sexp.pp_hum
                    ([%sexp_of: Code.traced_tensor] tn)
                    Sexp.pp_hum
                    ([%sexp_of: NodeUI.t] n);
                raise Unknown_synchronization))
        in
        let global_ptr = Some (Cudajit.mem_alloc ~byte_size:device_size_in_bytes) in
        let global = Some (NodeUI.tensor_ptr_name ptr) in
        let host_dims = Bigarray.Genarray.dims arr in
        let host_offset =
          Option.bind hosted ~f:(fun _ ->
              if local_is_slice_of_host then
                Some
                  (fun () ->
                    let offset_idcs =
                      try
                        Array.map2_exn host_idcs axes ~f:(fun idx -> function
                          | Shape.{ special = Frozen; _ } -> Code.lookup_dyn_ind dyn_env idx | _ -> 0)
                      with e ->
                        Caml.Format.printf "\nMismatch host_idcs axes = %a\n%!" Sexp.pp_hum
                          ([%sexp_of: Shape.dim array] axes);
                        raise e
                    in
                    compute_array_offset ~idcs:offset_idcs ~dims:host_dims)
              else (
                ignore jit_code;
                failwith "Exec_as_gccjit: non-slice hosted: NOT IMPLEMENTED YET"))
        in
        let backend_info =
          (if NodeUI.equal_data_kind ptr.field Value then "v:" else "g:")
          ^ (Sexp.to_string_hum @@ sexp_of_sync_properties sync)
          ^ ";"
        in
        if not @@ String.is_substring n.backend_info ~substring:backend_info then
          n.backend_info <- n.backend_info ^ backend_info;
        {
          hosted;
          local = None;
          sync;
          host_dims;
          device_dims;
          host_size_in_bytes;
          device_size_in_bytes;
          device_length;
          local_is_slice_of_host;
          host_offset;
          num_typ;
          is_double;
          global;
          global_ptr;
        }
      in
      match arr with
      | Byte_as_int_nd arr -> tensor NodeUI.byte_as_int false arr
      | Half_as_int_nd arr -> tensor NodeUI.half_as_int false arr
      | Single_nd arr -> tensor NodeUI.single false arr
      | Double_nd arr -> tensor NodeUI.double true arr)

let pp_array_offset ppf ~idcs ~dims =
  let open Caml.Format in
  let offset = ref 0 in
  for i = 0 to Array.length idcs - 1 do
    offset := dims.(i) * !offset;
    if i < Array.length idcs - 1 then fprintf ppf "@[(%s +@ %d *@ " idcs.(i) !offset
    else fprintf ppf "%s" idcs.(i)
  done;
  for _ = 0 to Array.length idcs - 2 do
    fprintf ppf "@])"
  done

let pp_indices ?provider_dim () ppf idcs =
  Caml.Format.pp_print_list ~pp_sep:pp_comma (pp_index_axis ?provider_dim) ppf @@ Array.to_list idcs

let jit_code (ppf : Caml.Format.formatter) ~traced_store llc : unit =
  let open Code in
  let open Caml.Format in
  (* let lookup ?provider_dim ?(example_only = false) ~on_host indices =
       Array.map indices ~f:(function
         | Shape.Fixed_idx i -> Int.to_string i
         | Iterator it | Dynamic_recipient it -> Shape.symbol_ident it
         | Dynamic_provider _ when example_only -> Int.to_string 0
         | Dynamic_provider _ -> Int.to_string @@ Option.value_exn provider_dim
         | Frozen_recipient it when on_host -> Shape.symbol_ident it
         | Frozen_recipient _ -> Int.to_string 0)
     in *)
  let locals = ref Map.Poly.empty in
  let rec pp_ll ~dyn_env (ppf : formatter) (c : unit low_level) =
    match c with
    | Lines [||] -> ()
    | Lines lines ->
        (* Note: no separator. *)
        (pp_print_list (pp_ll ~dyn_env) ppf @@ Array.to_list lines : unit)
    | For_loop { index = i; from_; to_; body; trace_it = _ } when Shape.task_id_sym i ->
        assert (from_ = 0);
        session_state.num_blocks <- to_ + 1;
        fprintf ppf "@[<2>{@ size_t %a = blockIdx.x;@ " pp_index i;
        pp_ll ~dyn_env ppf body;
        fprintf ppf "@]}@ "
    | For_loop { index = i; from_; to_; body; trace_it = _ } when Shape.sample_num_sym i ->
        assert (from_ = 0);
        session_state.num_threads <- to_ + 1;
        fprintf ppf "@[<2>{@ size_t %a = threadIdx.x;@ " pp_index i;
        pp_ll ~dyn_env ppf body;
        fprintf ppf "@]}@ "
    | For_loop { index = i; from_; to_; body; trace_it = _ } ->
        fprintf ppf "@[<2>for (int@ %a = %d;@ %a <= %d;@ ++%a) {@ %a@]}@ " pp_index i from_ pp_index i to_
          pp_index i (pp_ll ~dyn_env) body
    | Rebalance (s, lines) ->
        pp_ll ~dyn_env ppf
        @@ Lines (Array.append (Option.to_array @@ Option.map s ~f:(fun s -> Comment s)) lines)
    | If_task_id_is _ -> ()
    | Zero_out ptr ->
        if Hashtbl.mem session_state.tensors ptr then
          failwith
            ("exec_as_cuda: Non-initialization zeroing-out NOT IMPLEMENTED YET: " ^ Sexp.to_string_hum
            @@ [%sexp_of: NodeUI.tensor_ptr] ptr);
        let tn = Code.(get_node traced_store ptr) in
        assert tn.zero_initialized
        (* The initialization will be emitted by get_tensor. *)
    | Set (ptr, idcs, v) ->
        (* let host_idcs = lookup ~on_host:true idcs in *)
        let tensor = get_tensor ~traced_store ~jit_code:(pp_ll ~dyn_env) ~dyn_env ~host_idcs:idcs ptr in
        let old_locals = !locals in
        let num_closing_braces = pp_top_locals ~dyn_env ppf v in
        fprintf ppf "@[<2>%a[@,%a] =@ %a;@]@ " pp_get_ptr tensor (pp_indices ()) idcs
          (pp_float ~dyn_env ~num_typ:tensor.num_typ ~is_double:tensor.is_double)
          v;
        for _ = 1 to num_closing_braces do
          fprintf ppf "@]}@ "
        done;
        locals := old_locals
    | Dynamic_indices { tensor; tensor_idcs; dynamic_idcs; target_dims; body; slice = _ } ->
        jit_dynamic_indices ~dyn_env tensor ~tensor_idcs ~dynamic_idcs ~target_dims body
    | Comment message -> fprintf ppf "/* %s */@ " message
    | Set_local (({ scope_id; _ } as id), value) ->
        let num_typ, is_double = Map.find_exn !locals id in
        let old_locals = !locals in
        let num_closing_braces = pp_top_locals ~dyn_env ppf value in
        fprintf ppf "@[<2>v%d =@ %a;@]@ " scope_id ((pp_float ~dyn_env) ~num_typ ~is_double) value;
        for _ = 1 to num_closing_braces do
          fprintf ppf "@]}@ "
        done;
        locals := old_locals
  and pp_top_locals ~dyn_env ppf vcomp =
    match vcomp with
    | Local_scope { id = { scope_id = i; _ } as id; prec; body; orig_indices = _ } ->
        let typ = prec_to_c_type prec in
        (* Tensors are initialized to 0 by default. However, there is typically an explicit
           initialization for virtual nodes. *)
        fprintf ppf "@[{%s v%d = 0;@ " typ i;
        locals := Map.update !locals id ~f:(fun _ -> (typ, prec_is_double prec));
        pp_ll ~dyn_env ppf body;
        1
    | Get_local _ | Get_global _ | Get _ | Constant _ -> 0
    | Binop (Arg1, v1, _v2) -> pp_top_locals ~dyn_env ppf v1
    | Binop (Arg2, _v1, v2) -> pp_top_locals ~dyn_env ppf v2
    | Binop (_, v1, v2) -> pp_top_locals ~dyn_env ppf v1 + pp_top_locals ~dyn_env ppf v2
    | Unop (_, v) -> pp_top_locals ~dyn_env ppf v
  and pp_float ~dyn_env ~num_typ ~is_double ppf value =
    let loop = pp_float ~dyn_env ~num_typ ~is_double in
    match value with
    | Local_scope { id; _ } ->
        (* Embedding of Local_scope is done by pp_top_locals. *)
        loop ppf @@ Get_local id
    | Get_local id ->
        let typ, _local_is_double = Map.find_exn !locals id in
        if not @@ String.equal num_typ typ then fprintf ppf "(%s)" num_typ;
        fprintf ppf "v%d" id.scope_id
    | Get_global _ -> failwith "Exec_as_cuda: Get_global / FFI NOT IMPLEMENTED YET"
    | Get (ptr, idcs) ->
        (* let host_idcs = lookup ~on_host:true idcs in *)
        let tensor = get_tensor ~traced_store ~jit_code:(pp_ll ~dyn_env) ~dyn_env ~host_idcs:idcs ptr in
        fprintf ppf "@[<2>%a[%a@]]" pp_get_ptr tensor (pp_indices ()) idcs
    | Constant c -> fprintf ppf "(%f)" c
    | Binop (Arg1, v1, _v2) -> loop ppf v1
    | Binop (Arg2, _v1, v2) -> loop ppf v2
    | Binop (Add, v1, v2) -> fprintf ppf "@[<2>((%a) +@ (%a)@]@,)" loop v1 loop v2
    | Binop (Mul, v1, v2) -> fprintf ppf "@[<2>((%a) *@ (%a)@]@,)" loop v1 loop v2
    | Binop (Code.ToPowOf, v1, v2) when is_double -> fprintf ppf "@[<2>pow(%a,@ %a@]@,)" loop v1 loop v2
    | Binop (ToPowOf, v1, v2) -> fprintf ppf "@[<2>powf(%a,@ %a@]@,)" loop v1 loop v2
    | Binop (Relu_gate, v1, v2) ->
        fprintf ppf "@[<2>(%a > 0.0 ?@ %a : 0.0@])" loop v1 loop v2
        (* fprintf ppf "@[<2>((int)(%a > 0.0) *@ (%a)@])" (pp_ll ~dyn_env) v1 (pp_ll ~dyn_env) v2 *)
    | Unop (Identity, v) -> loop ppf v
    | Unop (Relu, v) ->
        (* FIXME: don't recompute v *)
        fprintf ppf "@[<2>(%a > 0.0 ?@ %a : 0.0@])" loop v loop v
  and jit_dynamic_indices ~dyn_env ptr ~tensor_idcs ~dynamic_idcs ~target_dims body =
    (* let host_idcs = lookup ~on_host:true ~example_only:true tensor_idcs in *)
    let tensor = get_tensor ~traced_store ~jit_code:(pp_ll ~dyn_env) ~dyn_env ~host_idcs:tensor_idcs ptr in
    let dyn_env =
      Array.foldi dynamic_idcs ~init:dyn_env ~f:(fun provider_dim dyn_env sym ->
          let target_dim = target_dims.(provider_dim).dim in
          fprintf ppf "size_t %a =@ (size_t)(%a[%a]) %% %d;@ " pp_symbol sym pp_get_ptr tensor
            (pp_indices ~provider_dim ()) tensor_idcs target_dim;
          Map.add_exn dyn_env ~key:sym ~data:(ptr, provider_dim, tensor_idcs, target_dim))
    in
    pp_ll ~dyn_env ppf body
    (* fprintf ppf "%a" (pp_ll ~dyn_env ~env) body *)
  in
  pp_ll ~dyn_env:Code.empty_env ppf llc

let new_context ?(device_num = 0) () =
  let num_devices = Cudajit.device_get_count () in
  if num_devices <= device_num then None
  else
    let device = Cudajit.device_get ~ordinal:device_num in
    Some (Cudajit.ctx_create ~flags:0 device)

let cleanup_session () =
  if Option.is_none session_state.ctx then Cudajit.init ();
  Option.iter session_state.last_module ~f:Cudajit.module_unload;
  session_state.last_module <- None;
  Hashtbl.iter session_state.tensors ~f:(fun tensor -> Option.iter tensor.global_ptr ~f:Cudajit.mem_free);
  Option.iter session_state.ctx ~f:Cudajit.ctx_destroy;
  (* For now we stick with device 0. *)
  session_state.ctx <- new_context ()

let error_message ~name:_ ~prefix:_ ?extra_error_msg:_ ~contents:_ _exc = ""

let jit_func ~name (traced_store, llc) =
  let module Cu = Cudajit in
  Option.iter session_state.last_module ~f:Cu.module_unload;
  session_state.last_module <- None;
  if Option.is_none session_state.ctx then cleanup_session ();
  if Option.is_none session_state.ctx then invalid_arg "Exec_as_cuda: no device found";
  ignore @@ Caml.Format.flush_str_formatter ();
  jit_code Caml.Format.str_formatter ~traced_store llc;
  let cu_body = Caml.Format.flush_str_formatter () in
  let tensors = Hashtbl.to_alist session_state.tensors in
  let params =
    List.filter_map tensors ~f:(fun (_, tn) ->
        Option.map tn.global ~f:(fun t_name -> tn.num_typ ^ " *" ^ t_name))
  in
  let cu_src =
    [%string
      {|
      extern "C" __global__ void %{name}(%{String.concat ~sep:", " params}) {
        %{cu_body}
      }|}]
  in
  let ptx = Cu.compile_to_ptx ~cu_src ~name ~options:[ "--use_fast_math" ] ~with_debug:!Code.with_debug in
  let module_ = Cu.module_load_data_ex ptx [] in
  session_state.last_module <- Some module_;
  let func = Cu.module_get_function module_ ~name in
  let args = List.filter_map tensors ~f:(fun (_, tn) -> Option.map tn.global_ptr ~f:(fun p -> Cu.Tensor p)) in
  fun () ->
    List.iter tensors ~f:(function
      | ptr, { hosted = Some ndarray; global_ptr = Some dst; host_offset; device_length; _ } ->
          let host_offset = Option.map host_offset ~f:(fun f -> f ()) in
          let tn = Code.(get_node traced_store ptr) in
          if tn.read_before_write then
            let f src = Cu.memcpy_H_to_D ?host_offset ~length:device_length ~dst ~src () in
            Node.map_as_bigarray { f } ndarray
      | _ -> ());
    List.iter tensors ~f:(function
      | ptr, { global_ptr = Some device; device_size_in_bytes; _ } ->
          let tn = Code.(get_node traced_store ptr) in
          if tn.zero_initialized then Cu.memset_d8 device Unsigned.UChar.zero ~length:device_size_in_bytes
      | _ -> ());
    Cu.launch_kernel func ~grid_dim_x:session_state.num_blocks ~block_dim_x:session_state.num_threads
      ~shared_mem_bytes:0 Cu.no_stream args;
    List.iter tensors ~f:(function
      | ptr, { hosted = Some ndarray; global_ptr = Some src; host_offset; device_length; _ } ->
          let host_offset = Option.map host_offset ~f:(fun f -> f ()) in
          let tn = Code.(get_node traced_store ptr) in
          if not tn.read_only then
            let f dst = Cu.memcpy_D_to_H ?host_offset ~length:device_length ~dst ~src () in
            Node.map_as_bigarray { f } ndarray
      | _ -> ())

let jit_step_func = jit_func
let jit_unit_func = jit_func
