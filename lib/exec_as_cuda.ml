open Base

let optimization_level = ref 3

(* let session_context = *)

type sync_properties =
  | Thread_only
  | Block_only
  | Update_globally_for_thread
      (** All assignments are update assignments. They happen directly in global memory, simultaneously
          syncing the tensor's cell value for a thread. *)
  | Update_globally_for_block
      (** All assignments are update assignments. They happen directly in global memory, simultaneously
          syncing the tensor's cell value for a block. *)
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
  local : string option;  (** Either a thread-local or shared (block-local) memory, if any. *)
  sync : sync_properties;
  host_dims : int array;
      (** Dimensions (shape) of the tensor as a whole, or an empty array if [hosted_ptr]
                              is [None]. *)
  device_dims : int array;  (** Dimensions (shape) of the per-task slice of the tensor. *)
  host_size_in_bytes : int;  (** Size of the full host's tensor. *)
  device_size_in_bytes : int;  (** Size of the per-task slice of the tensor. *)
  local_is_slice_of_host : bool;
      (** If true, the local tensor is a slice of the host tensor. If false, the local tensor needs to be
          iterated with a jitted code to copy it to/from the host tensor. *)
  num_typ : Gccjit.type_;
      (** The type of the stored values: [signed char] (corresponds to precision [Byte]),
      [short] (precision [Half]), [float] (precision [Single]), [double] (precision [Double]). *)
  is_double : bool;
}

(* let session_results = ref [] *)
let hoist_dynamic_indices = ref false

type state = {
  ctx : Cudajit.context;
  func : Cudajit.func;
  tensors : (NodeUI.tensor_ptr, tensor) Hashtbl.Poly.t;
  traced_store : Code.traced_store;
  mutable device_to_host : (unit -> unit) list;
  mutable host_to_device : (unit -> unit) list;
}

let pp_semi ppf () = Caml.Format.fprintf ppf ";@ "
let pp_symbol ppf (Shape.Symbol s) = Caml.Format.fprintf ppf "i%d" s

let pp_index ppf = function
  | Shape.Symbol s as sym when Shape.task_id_sym sym -> Caml.Format.fprintf ppf "task_id_%d" s
  | Shape.Symbol s as sym when Shape.sample_num_sym sym -> Caml.Format.fprintf ppf "sample_num_%d" s
  | Shape.Symbol s -> Caml.Format.fprintf ppf "i%d" s

let pp_index_axis ?provider_dim ppf =
  let open Shape in
  function
  | Iterator (Symbol s) -> Caml.Format.fprintf ppf "i%d" s
  | Special_iterator (Task_id, Symbol s) -> Caml.Format.fprintf ppf "task_id_%d" s
  | Special_iterator (Sample_num, Symbol s) -> Caml.Format.fprintf ppf "sample_num_%d" s
  | Frozen_recipient (Symbol s) -> Caml.Format.fprintf ppf "i%d" s
  | Dynamic_recipient (Symbol s) -> Caml.Format.fprintf ppf "i%d" s
  | Fixed_idx i -> Caml.Format.fprintf ppf "%d" i
  | Dynamic_provider _ -> Caml.Format.fprintf ppf "%d" @@ Option.value_exn provider_dim

let pp_get_ptr ppf tensor =
  match (tensor.global, tensor.local) with
  | _, Some lv -> Caml.Format.pp_print_string ppf lv
  | Some rv, _ -> Caml.Format.pp_print_string ppf rv
  | None, None -> assert false
(*
let get_tensor { ctx; func; tensors; traced_store; device_to_host; host_to_device } ?force_sync ~jit_code
    ~host_idcs ptr : tensor =
  Hashtbl.find_or_add tensors ptr ~default:(fun () ->
      let n = NodeUI.(get ptr.id) in
      let tn = Code.(get_node traced_store ptr) in
      let host_size_in_bytes = NodeUI.host_size_in_bytes ptr in
      let axes = Shape.to_dims n.shape in
      let device_dims = axes |> Array.map ~f:(fun d -> d.dim) in
      let device_size = Array.fold ~init:1 ~f:( * ) device_dims in
      let arr = Option.value_exn @@ NodeUI.get_tensor ptr in
      let device_size_in_bytes = device_size * Node.precision_in_bytes arr in
      let local_is_slice_of_host =
        Array.fold_until axes ~init:true
          ~f:
            (fun was_frozen -> function
              | Shape.{ special = Frozen | Dedicated Task_id; _ } ->
                  if was_frozen then Continue true else Stop false
              | _ -> Continue false)
          ~finish:(fun _ -> true)
      in
      let tensor c_typ is_double arr =
        let num_typ = Type.(get ctx c_typ) in
        let hosted_ptr =
          if Array.is_empty @@ Node.A.dims arr then None
          else Some (RValue.ptr ctx (Type.pointer num_typ) @@ Ctypes.bigarray_start Ctypes_static.Genarray arr)
        in
        let is_parallel =
          Array.exists ~f:Shape.(function { special = Dedicated Task_id; _ } -> true | _ -> false)
          @@ Shape.to_dims n.shape
        in
        let can_be_replicated = tn.is_replicable in
        let update_on_host =
          (not is_parallel) && tn.read_before_write && tn.reduced_racyness && Option.is_some hosted_ptr
        in
        let sync =
          Option.value_or_thunk force_sync ~default:(fun () ->
              if Option.is_none hosted_ptr then Device_only
              else if is_parallel then Parallel_dim
              else if update_on_host then Update_on_host
              else if can_be_replicated then Replicated
              else (
                if !Code.with_debug then
                  Caml.Format.printf "\nWARNING: No sync for tensor: %a@ node: %a\n%!" Sexp.pp_hum
                    ([%sexp_of: Code.traced_tensor] tn)
                    Sexp.pp_hum
                    ([%sexp_of: NodeUI.t] n);
                raise Unknown_synchronization))
        in
        let arr_typ = Type.array ctx num_typ device_size in
        let local = Function.local func arr_typ @@ NodeUI.tensor_ptr_name ptr in
        let host_dims = Bigarray.Genarray.dims arr in
        let cast_void rv = RValue.cast ctx rv c_void_ptr in
        if tn.zero_initialized && not update_on_host then
          Block.eval task_init_block
          @@ RValue.call ctx (Function.builtin ctx "memset")
               [
                 cast_void @@ LValue.address local;
                 RValue.zero ctx c_int;
                 RValue.int ctx c_index device_size_in_bytes;
               ];
        Option.iter hosted_ptr ~f:(fun hosted_ptr ->
            if local_is_slice_of_host then (
              let offset_idcs =
                try
                  Array.map2_exn host_idcs axes ~f:(fun idx -> function
                    | Shape.{ special = Frozen | Dedicated Task_id; _ } -> idx | _ -> RValue.zero ctx c_int)
                with e ->
                  Caml.Format.printf "\nMismatch host_idcs axes = %a\n%!" Sexp.pp_hum
                    ([%sexp_of: Shape.dim array] axes);
                  raise e
              in
              let offset = jit_array_offset ctx ~idcs:offset_idcs ~dims:host_dims in
              let lhs = LValue.access_array hosted_ptr offset in
              if tn.zero_initialized && update_on_host then
                Block.eval task_init_block
                @@ RValue.call ctx (Function.builtin ctx "memset")
                     [
                       cast_void @@ LValue.address lhs;
                       RValue.zero ctx c_int;
                       RValue.int ctx c_index device_size_in_bytes;
                     ];
              if tn.read_before_write then
                Block.eval task_init_block
                @@ RValue.call ctx (Function.builtin ctx "memcpy")
                     [
                       cast_void @@ LValue.address local;
                       cast_void @@ LValue.address lhs;
                       RValue.int ctx c_index device_size_in_bytes;
                     ];
              if (not tn.read_only) && not update_on_host then
                (* let f = Function.builtin ctx "printf" in
                   let print_args =
                     [
                       RValue.string_literal ctx
                         ("\nDEBUG: copy to host "
                         ^ Sexp.to_string_hum ([%sexp_of: NodeUI.tensor_ptr] ptr)
                         ^ " -- index: %d; size: %d: \n");
                         offset;
                         RValue.int ctx c_index device_size_in_bytes;
                     ]
                   in
                   Block.eval task_finalize_block @@ RValue.call ctx f print_args; *)
                Block.eval (if is_replicated sync then replicated_finalize_block else task_finalize_block)
                @@ RValue.call ctx (Function.builtin ctx "memcpy")
                     [
                       cast_void @@ LValue.address lhs;
                       cast_void @@ LValue.address local;
                       RValue.int ctx c_index device_size_in_bytes;
                     ])
            else (
              ignore jit_code;
              failwith "Exec_as_gccjit: non-slice hosted: NOT IMPLEMENTED YET"));
        let backend_info =
          (if NodeUI.equal_data_kind ptr.field Value then "v:" else "g:")
          ^ (Sexp.to_string_hum @@ sexp_of_sync_properties sync)
          ^ ";"
        in
        if not @@ String.is_substring n.backend_info ~substring:backend_info then
          n.backend_info <- n.backend_info ^ backend_info;
        {
          hosted_ptr;
          local = Some local;
          sync;
          host_dims;
          device_dims;
          host_size_in_bytes;
          device_size_in_bytes;
          local_is_slice_of_host;
          num_typ;
          is_double;
        }
      in
      match arr with
      | Byte_as_int_nd arr -> tensor Type.Signed_char false arr
      | Half_as_int_nd arr -> tensor Type.Short false arr
      | Single_nd arr -> tensor Type.Float false arr
      | Double_nd arr -> tensor Type.Double true arr)
*)
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

let jit_code ~name ({ ctx; func; _ } as state) ~as_toplevel (ppf : Caml.Format.formatter) (type a)
    (c : a Code.low_level) : unit =
    ignore (name, ctx, func, state, as_toplevel, ppf, c)
    (*
  let open Code in
  let open Caml.Format in
  let lookup ?provider_dim ?(example_only = false) ~on_host indices =
    try
      Array.map indices ~f:(function
        | Shape.Fixed_idx i -> Int.to_string i
        | Iterator (Symbol s) -> "i" ^ Int.to_string s
        | Special_iterator (Task_id, _) when on_host -> "task_id"
        | Special_iterator Task_id -> Int.to_string 0
        | Special_iterator Sample_num -> snd @@ Option.value_exn env.sample_num
        | Dynamic_recipient s -> Map.find_exn env.dyn_env s
        | Dynamic_provider _ when example_only -> Int.to_string 0
        | Dynamic_provider _ -> Int.to_string @@ Option.value_exn provider_dim
        | Frozen_recipient s when on_host -> Map.find_exn env.dyn_env s
        | Frozen_recipient _ -> Int.to_string 0)
    with e ->
      Caml.Format.eprintf "exec_as_gccjit: missing index from@ %a@ among environment keys:@ %a\n%!"
        Sexp.pp_hum
        ([%sexp_of: Code.sym_index Shape.axis_index array] indices)
        Sexp.pp_hum
        ([%sexp_of: Sexp.t list] @@ Code.environment_keys env);
      raise e
  in
  let rec pp_ll (ppf : formatter) (c : unit low_level) =
    (* FIXME: performance bug, bind the nodes [(get %d)] at the start of the program. *)
    match c with
    | Lines [||] -> ()
    | Lines lines ->
        (pp_print_list ~pp_sep:pp_semi pp_ll ppf @@ Array.to_list lines : unit);
        pp_semi ppf ()
    | For_loop { index = i; from_; to_; body; trace_it = _ } ->
        fprintf ppf "@[<2>for (int@ %a = %d;@ %a <= %d;@ ++%a) {@ %a@]}" pp_index i from_ pp_index i to_
          pp_index i pp_ll body
    | Rebalance (s, lines) ->
        pp_ll ppf @@ Lines (Array.append (Option.to_array @@ Option.map s ~f:(fun s -> Comment s)) lines)
    | If_task_id_is _ -> ()
    | Zero_out ptr ->
        if Hashtbl.mem state.tensors ptr then
          failwith
            ("exec_as_cuda: Non-initialization zeroing-out NOT IMPLEMENTED YET: " ^ Sexp.to_string_hum
            @@ [%sexp_of: NodeUI.tensor_ptr] ptr);
        let tn = Code.(get_node state.traced_store ptr) in
        assert tn.zero_initialized
        (* The initialization will be emitted by get_tensor. *)
    | Set (ptr, idcs, v) ->
        let host_idcs = lookup ~on_host:true env idcs in
        let tensor = get_tensor state ~jit_code:pp_ll ~host_idcs ptr in
        fprintf ppf "@[<2>%a[@,%a] =@ %a@]" pp_get_ptr tensor pp_idcs idcs
          (pp_float ~name ~env ~num_typ:tensor.num_typ ~is_double:tensor.is_double)
          v
    | Dynamic_indices { tensor; tensor_idcs; dynamic_idcs; target_dims; body; slice = _ } ->
        jit_dynamic_indices ~name ~env tensor ~tensor_idcs ~dynamic_idcs ~target_dims body
        (* Array.iteri dynamic_idcs ~f:(fun provider_dim sym ->
               let target_dim =
                 match target_dims.(provider_dim) with Parallel -> !Shape.num_parallel_tasks | Dim d -> d
               in
               fprintf ppf "let@ %a = Int.(@[<2>(get_as_int %a@ (%a)) %% %d@]) in@ " pp_symbol sym pp_data_node
                 tensor (pp_indices ~provider_dim) tensor_idcs target_dim);
           pp_ll ppf body *)
    | Comment message -> fprintf ppf "/* %s */" message
    | Set_local ({ scope_id; _ }, value) -> fprintf ppf "@[<2>v%d :=@ %a@]" scope_id pp_ll value
  and pp_float ~name ~env ~num_typ ~is_double ppf value =
    match value with
    | Local_scope { id = { scope_id; tensor }; prec = _; body; orig_indices } ->
        (* Note: we could support precisions, but it's not worth it. *)
        if !Code.debug_virtual_nodes then
          fprintf ppf "@[<2>let v%d =@ ref 0.0 in@ (%a;@ set_from_float %a@ (%a)@ !v%d; !v%d)@]" scope_id
            pp_ll body pp_data_node tensor pp_idcs orig_indices scope_id scope_id
        else fprintf ppf "@[<2>let v%d =@ ref 0.0 in@ %a;@ !v%d@]" scope_id pp_ll body scope_id
    | Get_local { scope_id; _ } -> fprintf ppf "!v%d" scope_id
    | Get_global Task_id -> fprintf ppf "(Float.of_int task_id)"
    | Get_global (C_function _) -> failwith "OCaml backend: C FFI NOT IMPLEMENTED YET"
    | Get (tensor, indices) -> fprintf ppf "@[<2>get_as_float %a@ (%a)@]" pp_data_node tensor pp_idcs indices
    | Constant c -> fprintf ppf "(%f)" c
    | Binop (Arg1, v1, _v2) -> pp_ll ppf v1
    | Binop (Arg2, _v1, v2) -> pp_ll ppf v2
    | Binop (Add, v1, v2) -> fprintf ppf "(@[<2>(%a) +@ (%a)@]@,)" pp_ll v1 pp_ll v2
    | Binop (Mul, v1, v2) -> fprintf ppf "(@[<2>(%a) *@ (%a)@]@,)" pp_ll v1 pp_ll v2
    | Binop (Code.ToPowOf, v1, v2) when is_double -> fprintf ppf "pow(@[<2>%a,@ %a@]@,)" pp_ll v1 pp_ll v2
    | Binop (ToPowOf, v1, v2) -> fprintf ppf "pow(@[<2>%a,@ %a@]@,)" pp_ll v1 pp_ll v2
    | Binop (Relu_gate, v1, v2) ->
        fprintf ppf "(@[<2>%a > 0.0 ?@ %a : 0.0@])" pp_ll v1 pp_ll v2
        (* fprintf ppf "(@[<2>(int)(%a > 0.0) *@ (%a)@])" pp_ll v1 pp_ll v2 *)
    | Unop (Identity, v) -> pp_ll ppf v
    | Unop (Relu, v) -> fprintf ppf "(@[<2>%a > 0.0 ?@ %a : 0.0@])" pp_ll v pp_ll v
  and jit_dynamic_indices ~name ~env tensor ~tensor_idcs ~dynamic_idcs ~target_dims body =
    let host_idcs = lookup ~on_host:true ~example_only:true env tensor_idcs in
    let tensor = get_tensor state ~jit_code:pp_ll ~host_idcs tensor in
    let env =
      Array.foldi dynamic_idcs ~init:env ~f:(fun provider_dim env (Symbol s as key) ->
          let target_dim = target_dims.(provider_dim).dim in
          let idcs = lookup ~provider_dim ~on_host:false env tensor_idcs in
          let device_prov_offset = pp_array_offset ctx ~idcs ~dims:tensor.device_dims in
          let dyn_index = RValue.lvalue @@ LValue.access_array (get_ptr tensor) device_prov_offset in
          let dyn_index = RValue.cast ctx dyn_index c_index in
          let dyn_index = RValue.binary_op ctx Modulo c_index dyn_index target_dim in
          let data =
            if !hoist_dynamic_indices then (
              let sym_index = Function.local func c_index ("i" ^ Int.to_string s) in
              Block.assign !current_block sym_index dyn_index;
              RValue.lvalue sym_index)
            else dyn_index
          in
          { env with dyn_env = Map.add_exn ~key ~data env.dyn_env })
    in
    Array.iteri dynamic_idcs ~f:(fun provider_dim sym ->
        let target_dim =
          match target_dims.(provider_dim) with Parallel -> !Shape.num_parallel_tasks | Dim d -> d
        in
        fprintf ppf "let@ %a = Int.(@[<2>(get_as_int %a@ (%a)) %% %d@]) in@ " pp_symbol sym pp_data_node
          tensor (pp_indices ~provider_dim) tensor_idcs target_dim);
    pp_ll ppf body
    (* fprintf ppf "%a" (pp_ll ~name ~env) body *)
  in
  (match c with
  | Lines toplevel ->
      if as_toplevel then
        fprintf ppf "@[<2>let () =@ %a@]"
          (pp_print_list ~pp_sep:(fun p () -> fprintf p "@]@ @[<2>let () =@ ") pp_ll)
        @@ Array.to_list toplevel
      else fprintf ppf "(@[<2>%a@]@,)" (pp_print_list ~pp_sep:pp_semi pp_ll) @@ Array.to_list toplevel
  | c -> pp_ll ppf c);
  fprintf ppf "@]"
*)
let cleanup_session () = ()
let error_message ~name:_ ~prefix:_ ?extra_error_msg:_ ~contents:_ _exc = ""
let jit_step_func ~name:_ (_store, _llc) () = ()
let jit_unit_func ~name:_ (_store, _llc) () = ()
