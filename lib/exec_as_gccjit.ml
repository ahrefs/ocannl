open Base

let optimization_level = ref 3

let session_context =
  let open Gccjit.Context in
  let ctx = create () in
  set_option ctx Optimization_level !optimization_level;
  ref ctx

type sync_properties =
  | Device_only  (** The tensor is only needed for a task-local computation and does not exist on host. *)
  | Update_on_host
      (** All assignments are update assignments. They happen directly on the host, simultaneously
          syncing the tensor's cell value. *)
  | Parallel_dim
      (** The shape of the tensor has a [Parallel] dimension. Each task computes a slice of this dimension,
          independently transferring to the host. *)
  | Replicated
      (** If true, the tensor computation happens on-device in all tasks, but result is transferred to host
          on only one task ([task_id = 0]). *)
[@@deriving sexp, equal, compare, variants]

type tensor = {
  hosted_ptr : Gccjit.rvalue option;
      (** Pointer to the first value of the associated [Bigarray], if hosted. Usually it does not correspond
          to the local tensor (e.g. if task id > 0). *)
  local : Gccjit.lvalue option;  (** A local array, if any. *)
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

let session_results : Gccjit.result list ref = ref []
let hoist_dynamic_indices = ref false

type state = {
  ctx : Gccjit.context;
  func : Gccjit.function_;
  tensors : (NodeUI.tensor_ptr, tensor) Hashtbl.Poly.t;
  traced_store : Code.traced_store;
  task_init_block : Gccjit.block;
  task_finalize_block : Gccjit.block;
  replicated_finalize_block : Gccjit.block;
}

let jit_array_offset ctx ~idcs ~dims =
  let open Gccjit in
  let c_index = Type.get ctx Type.Int in
  Array.fold2_exn idcs dims ~init:(RValue.zero ctx c_index) ~f:(fun offset idx dim ->
      RValue.binary_op ctx Plus c_index idx
      @@ RValue.binary_op ctx Mult c_index offset (RValue.int ctx c_index dim))

let get_tensor
    { ctx; func; tensors; traced_store; task_init_block; task_finalize_block; replicated_finalize_block }
    ~jit_code ~host_idcs ptr : tensor =
  let open Gccjit in
  Hashtbl.find_or_add tensors ptr ~default:(fun () ->
      let n = NodeUI.(get ptr.id) in
      let tn = Code.(get_node traced_store ptr) in
      let host_size_in_bytes = NodeUI.host_size_in_bytes ptr in
      let axes = Shape.to_dims n.shape in
      let device_dims = axes |> Array.map ~f:(function Shape.Parallel -> 1 | Frozen _ -> 1 | Dim d -> d) in
      let device_size = Array.fold ~init:1 ~f:( * ) device_dims in
      let arr = Option.value_exn @@ NodeUI.get_tensor ptr in
      let device_size_in_bytes = device_size * Node.precision_in_bytes arr in
      let local_is_slice_of_host =
        Array.fold_until axes ~init:true
          ~f:
            (fun was_frozen -> function
              | Shape.Parallel | Frozen _ -> if was_frozen then Continue true else Stop false
              | _ -> Continue false)
          ~finish:(fun _ -> true)
      in
      let c_void_ptr = Type.(get ctx Type.Void_ptr) in
      let c_index = Type.get ctx Type.Size_t in
      let c_int = Type.get ctx Type.Int in
      let tensor c_typ is_double arr =
        let num_typ = Type.(get ctx c_typ) in
        let hosted_ptr =
          if Array.is_empty @@ Node.A.dims arr then None
          else Some (RValue.ptr ctx (Type.pointer num_typ) @@ Ctypes.bigarray_start Ctypes_static.Genarray arr)
        in
        let arr_typ = Type.array ctx num_typ device_size in
        let local = Function.local func arr_typ @@ NodeUI.tensor_ptr_name ptr in
        let host_dims = Bigarray.Genarray.dims arr in
        let is_parallel = Array.exists ~f:Shape.is_parallel @@ Shape.to_dims n.shape in
        let can_be_replicated =
          (* TODO: currently we do not check for gradient tensors, since their computation dependencies are
             different than the node dependencies. *)
          NodeUI.(equal_data_kind ptr.field Value && (not @@ has_parallel_deps n))
        in
        let update_on_host =
          (not is_parallel) && tn.read_before_write && tn.reduced_racyness && Option.is_some hosted_ptr
        in
        let sync =
          if Option.is_none hosted_ptr then Device_only
          else if is_parallel then Parallel_dim
          else if update_on_host then Update_on_host
          else if can_be_replicated || !Shape.num_parallel_tasks <= 1 then Replicated
          else failwith "exec_as_gccjit: synchronization pattern NOT IMPLEMENTED YET"
        in
        Option.iter hosted_ptr ~f:(fun hosted_ptr ->
            if local_is_slice_of_host then (
              let offset_idcs =
                try
                  Array.map2_exn host_idcs axes ~f:(fun idx -> function
                    | Frozen _ | Parallel -> idx | Dim _ -> RValue.zero ctx c_int)
                with e ->
                  Caml.Format.printf "\nMismatch host_idcs axes = %a\n%!" Sexp.pp_hum
                    ([%sexp_of: Shape.dim array] axes);
                  raise e
              in
              let offset = jit_array_offset ctx ~idcs:offset_idcs ~dims:host_dims in
              let lhs = LValue.access_array hosted_ptr offset in
              let cast_void rv = RValue.cast ctx rv c_void_ptr in
              if tn.read_before_write then
                Block.eval task_init_block
                @@ RValue.call ctx (Function.builtin ctx "memcpy")
                     [
                       cast_void @@ LValue.address local;
                       cast_void @@ LValue.address lhs;
                       RValue.int ctx c_index device_size_in_bytes;
                     ];
              if is_parallel || not update_on_host then
                Block.eval (if is_parallel then task_finalize_block else replicated_finalize_block)
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
          ^ (if is_parallel then "parallelized"
             else if update_on_host then "sync-on-update"
             else if Option.is_none hosted_ptr then "device-only"
             else "replicated")
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

let cleanup_session () =
  let open Gccjit in
  List.iter !session_results ~f:Result.release;
  Context.release !session_context;
  session_context := Context.create ();
  Context.set_option !session_context Optimization_level !optimization_level;
  session_results := []

let prec_to_kind prec =
  let open Gccjit in
  match prec with
  | NodeUI.Void_prec -> Type.Void
  | Byte_as_int_prec _ -> Type.Signed_char
  | Half_as_int_prec _ -> Type.Short
  | Single_prec _ -> Type.Float
  | Double_prec _ -> Type.Double

let prec_is_double = function NodeUI.Double_prec _ -> true | _ -> false

let is_builtin_op = function
  | Code.Add | Code.Mul -> true
  | Code.ToPowOf | Code.Relu_gate | Code.Arg2 | Code.Arg1 -> false

let builtin_op = function
  | Code.Add -> Gccjit.Plus
  | Code.Mul -> Gccjit.Mult
  | Code.ToPowOf | Code.Relu_gate | Code.Arg2 | Code.Arg1 ->
      invalid_arg "Exec_as_gccjit.builtin_op: not a builtin"

let get_ptr tensor =
  match (tensor.hosted_ptr, tensor.local) with
  | _, Some lv -> Gccjit.RValue.lvalue lv
  | Some rv, _ -> rv
  | None, None -> assert false

let get_sync_ptr tensor =
  match (tensor.hosted_ptr, tensor.local) with
  | Some rv, _ when is_update_on_host tensor.sync -> rv
  | _, Some lv -> Gccjit.RValue.lvalue lv
  | Some rv, _ -> rv
  | None, None -> assert false

let jit_code ~name ~env ~task_id ({ ctx; func; _ } as state) initial_block (body : unit Code.low_level) :
    Gccjit.block =
  let open Gccjit in
  let c_int = Type.get ctx Type.Int in
  let c_index = c_int in
  let lookup ?provider_dim ?(example_only = false) ~on_host (env, dyn_env) indices =
    let open Gccjit in
    Array.map indices ~f:(function
      | Shape.Fixed_idx i -> RValue.int ctx c_index i
      | Iterator s -> Map.find_exn env s
      | Task_id when on_host -> RValue.param task_id
      | Task_id -> RValue.zero ctx c_index
      | Dynamic_recipient s -> Map.find_exn dyn_env s
      | Dynamic_provider _ when example_only -> RValue.zero ctx c_index
      | Dynamic_provider _ -> Option.value_exn provider_dim
      | Frozen_recipient s when on_host -> Map.find_exn dyn_env s
      | Frozen_recipient _ -> RValue.zero ctx c_index)
  in
  let c_float = Type.get ctx Type.Float in
  let c_double = Type.get ctx Type.Double in
  let cast_bool num_typ v = RValue.cast ctx (RValue.cast ctx v c_int) num_typ in
  (* Source of unique identifiers. E.g. local scope ids can be non-unique due to inlining.
     We also need unique ids for computation ordering lvalues. *)
  let uid = ref 0 in
  let get_uid () =
    let id =
      Int.incr uid;
      !uid
    in
    Int.to_string id
  in
  let locals = ref Map.Poly.empty in
  let current_block = ref initial_block in
  let loop_binop op ~num_typ ~is_double ~v1 ~v2 =
    match op with
    | Code.Add -> RValue.binary_op ctx Plus num_typ v1 v2
    | Code.Mul -> RValue.binary_op ctx Mult num_typ v1 v2
    | Code.ToPowOf when is_double ->
        let base = RValue.cast ctx v1 c_double in
        let expon = RValue.cast ctx v2 c_double in
        RValue.cast ctx (RValue.call ctx (Function.builtin ctx "pow") [ base; expon ]) num_typ
    | Code.ToPowOf ->
        let base = RValue.cast ctx v1 c_float in
        let expon = RValue.cast ctx v2 c_float in
        RValue.cast ctx (RValue.call ctx (Function.builtin ctx "powf") [ base; expon ]) num_typ
    | Code.Relu_gate ->
        let cmp = cast_bool num_typ @@ RValue.comparison ctx Lt (RValue.zero ctx num_typ) v1 in
        RValue.binary_op ctx Mult num_typ cmp @@ v2
    | Code.Arg2 -> v2
    | Code.Arg1 -> v1
  in
  let log_comment c =
    (if !Code.with_debug && !Code.executor_print_comments then
       let f = Function.builtin ctx "printf" in
       Block.eval !current_block
       @@ RValue.call ctx f
            [ RValue.string_literal ctx ("\nComment for task %d: " ^ c ^ "\n"); RValue.param task_id ]);
    Block.comment !current_block c
  in
  let rec loop_proc ~env ~name (body : unit Code.low_level) : unit =
    let loop = loop_proc ~env in
    match body with
    | Code.Lines lines ->
        Array.iteri lines ~f:(fun i line -> loop ~name:(name ^ "_at_line_" ^ Int.to_string i) line)
    | For_loop { index; from_; to_; body; trace_it = _ } ->
        jit_for_loop ~env index ~from_ ~to_ (Either.First body)
    | Rebalance (_, cs) ->
        (* This backend does not implement a relevant form of fine-grain parallelism. *)
        Array.iteri cs ~f:(fun i line -> loop ~name:(name ^ "_at_par_line_" ^ Int.to_string i) line)
    | If_task_id_is { for_task_id = _; body } when !Shape.num_parallel_tasks <= 1 -> loop ~name body
    | If_task_id_is { for_task_id; body } ->
        let open Gccjit in
        let id = get_uid () in
        let b_if_body =
          Block.create ~name:("body_if_task_id_is_" ^ Int.to_string for_task_id ^ "_" ^ id) func
        in
        let b_after_if =
          Block.create ~name:("after_if_task_id_is_" ^ Int.to_string for_task_id ^ "_" ^ id) func
        in
        let guard = RValue.comparison ctx Eq (RValue.param task_id) (RValue.int ctx c_index for_task_id) in
        Block.cond_jump !current_block guard b_if_body (* on true *) b_after_if (* on false *);
        current_block := b_if_body;
        loop ~name body;
        Block.jump !current_block b_after_if;
        current_block := b_after_if
    | Set (_, _, Binop (Arg2, Get (_, _), _)) -> assert false
    | Set (tensor, idcs, Binop (op, Get (tensor2, idcs2), c2))
      when NodeUI.equal_tensor_ptr tensor tensor2 && [%equal: Code.index array] idcs idcs2 && is_builtin_op op
      ->
        (* FIXME: maybe it's not worth it? *)
        let host_idcs = lookup ~on_host:true env idcs in
        let tensor = get_tensor state ~jit_code:loop_proc ~host_idcs tensor in
        let value = loop_float ~name ~env ~num_typ:tensor.num_typ ~is_double:tensor.is_double c2 in
        let idcs = lookup ~on_host:(is_update_on_host tensor.sync) env idcs in
        let device_offset = jit_array_offset ctx ~idcs ~dims:tensor.device_dims in
        let device_lhs = LValue.access_array (get_ptr tensor) device_offset in
        if is_update_on_host tensor.sync then (
          let host_offset = jit_array_offset ctx ~idcs:host_idcs ~dims:tensor.host_dims in
          let host_lhs = LValue.access_array (get_sync_ptr tensor) host_offset in
          Block.assign_op !current_block host_lhs (builtin_op op) value;
          Block.assign !current_block device_lhs (RValue.lvalue host_lhs))
        else Block.assign_op !current_block device_lhs (builtin_op op) value
    | Set (tensor, idcs, Binop (op, Get (tensor2, idcs2), c2))
      when NodeUI.equal_tensor_ptr tensor tensor2 && [%equal: Code.index array] idcs idcs2 ->
        let host_idcs = lookup ~on_host:true env idcs in
        let tensor = get_tensor state ~jit_code:loop_proc ~host_idcs tensor in
        let value = loop_float ~name ~env ~num_typ:tensor.num_typ ~is_double:tensor.is_double c2 in
        let idcs = lookup ~on_host:(is_update_on_host tensor.sync) env idcs in
        let device_offset = jit_array_offset ctx ~idcs ~dims:tensor.device_dims in
        let device_lhs = LValue.access_array (get_ptr tensor) device_offset in
        if is_update_on_host tensor.sync then (
          let host_offset = jit_array_offset ctx ~idcs:host_idcs ~dims:tensor.host_dims in
          let host_lhs = LValue.access_array (get_sync_ptr tensor) host_offset in
          let result =
            loop_binop op ~num_typ:tensor.num_typ ~is_double:tensor.is_double ~v1:(RValue.lvalue host_lhs)
              ~v2:value
          in
          Block.assign !current_block host_lhs result;
          Block.assign !current_block device_lhs result)
        else
          let rhs =
            loop_binop op ~num_typ:tensor.num_typ ~is_double:tensor.is_double ~v1:(RValue.lvalue device_lhs)
              ~v2:value
          in
          Block.assign !current_block device_lhs rhs
    | Set (tensor, idcs, Binop (op, c2, Get (tensor2, idcs2)))
      when NodeUI.equal_tensor_ptr tensor tensor2 && [%equal: Code.index array] idcs idcs2 ->
        let host_idcs = lookup ~on_host:true env idcs in
        let tensor = get_tensor state ~jit_code:loop_proc ~host_idcs tensor in
        let value = loop_float ~name ~env ~num_typ:tensor.num_typ ~is_double:tensor.is_double c2 in
        let idcs = lookup ~on_host:(is_update_on_host tensor.sync) env idcs in
        let device_offset = jit_array_offset ctx ~idcs ~dims:tensor.device_dims in
        let device_lhs = LValue.access_array (get_ptr tensor) device_offset in
        if is_update_on_host tensor.sync then (
          let host_offset = jit_array_offset ctx ~idcs:host_idcs ~dims:tensor.host_dims in
          let host_lhs = LValue.access_array (get_sync_ptr tensor) host_offset in
          let result =
            loop_binop op ~num_typ:tensor.num_typ ~is_double:tensor.is_double ~v1:value
              ~v2:(RValue.lvalue host_lhs)
          in
          Block.assign !current_block host_lhs result;
          Block.assign !current_block device_lhs result)
        else
          let rhs =
            loop_binop op ~num_typ:tensor.num_typ ~is_double:tensor.is_double ~v1:value
              ~v2:(RValue.lvalue device_lhs)
          in
          Block.assign !current_block device_lhs rhs
    | Set (ptr, idcs, value) ->
        let host_idcs = lookup ~on_host:true env idcs in
        let tensor = get_tensor state ~jit_code:loop_proc ~host_idcs ptr in
        let value = loop_float ~name ~env ~num_typ:tensor.num_typ ~is_double:tensor.is_double value in
        let idcs = lookup ~on_host:false env idcs in
        let device_offset = jit_array_offset ctx ~idcs ~dims:tensor.device_dims in
        let device_lhs = LValue.access_array (get_ptr tensor) device_offset in
        Block.assign !current_block device_lhs value
    | Set_local (id, value) ->
        let lhs, num_typ, is_double = Map.find_exn !locals id in
        let value = loop_float ~name ~env ~num_typ ~is_double value in
        Block.assign !current_block lhs value
    | Comment c -> log_comment c
    | Dynamic_indices { tensor; tensor_idcs; dynamic_idcs; target_dims; body } ->
        jit_dynamic_indices ~name ~env tensor ~tensor_idcs ~dynamic_idcs ~target_dims body
  and loop_float ~name ~env ~num_typ ~is_double value : rvalue =
    let loop = loop_float ~name ~env ~num_typ ~is_double in
    match value with
    | Local_scope { id = { scope_id = i; _ } as id; prec; body; orig_indices = _ } ->
        let typ = Type.get ctx @@ prec_to_kind prec in
        (* Scope ids can be non-unique due to inlining. *)
        let v_name = Int.("v" ^ to_string i ^ "_" ^ get_uid ()) in
        let lvalue = Function.local func typ v_name in
        (* Tensors are initialized to 0 by default. However, there is typically an explicit
           initialization for virtual nodes. *)
        Block.assign !current_block lvalue @@ RValue.zero ctx typ;
        let old_locals = !locals in
        locals := Map.update !locals id ~f:(fun _ -> (lvalue, typ, prec_is_double prec));
        loop_proc ~env ~name:(name ^ "_at_" ^ v_name) body;
        locals := old_locals;
        RValue.lvalue lvalue
    | Get_local id ->
        let lvalue, _typ, _local_is_double = Map.find_exn !locals id in
        (* FIXME: Convert according to local_is_double ?= is_double. *)
        RValue.lvalue lvalue
    | Get_global Task_id -> RValue.cast ctx (RValue.param task_id) num_typ
    | Get_global (C_function f_name) ->
        (* TODO: this is too limiting. *)
        let f = Function.builtin ctx f_name in
        RValue.call ctx f []
    | Get (tensor, idcs) ->
        let host_idcs = lookup ~on_host:true env idcs in
        let tensor = get_tensor state ~jit_code:loop_proc ~host_idcs tensor in
        let idcs = lookup ~on_host:false env idcs in
        let device_offset = jit_array_offset ctx ~idcs ~dims:tensor.device_dims in
        RValue.lvalue @@ LValue.access_array (get_ptr tensor) device_offset
    | Binop (Code.Arg2, _, c2) -> loop c2
    | Binop (Code.Arg1, c1, _) -> loop c1
    | Binop (op, c1, c2) -> loop_binop op ~num_typ ~is_double ~v1:(loop c1) ~v2:(loop c2)
    | Unop (Code.Identity, c) -> loop c
    | Unop (Code.Relu, c) ->
        let cmp = cast_bool num_typ @@ RValue.comparison ctx Lt (RValue.zero ctx num_typ) @@ loop c in
        RValue.binary_op ctx Mult num_typ cmp @@ loop c
    | Constant v -> RValue.double ctx num_typ v
  and jit_for_loop ~env (Code.{ sym = Shape.Symbol s; uid } as key) ~from_ ~to_ body : unit =
    let open Gccjit in
    let i = "i" ^ Int.to_string s ^ "_" ^ Int.to_string uid in
    let index = Function.local func c_index i in
    let env = (Map.add_exn ~key ~data:(RValue.lvalue index) @@ fst env, snd env) in
    let b_loop_cond = Block.create ~name:("loop_cond_" ^ i) func in
    let b_loop_body = Block.create ~name:("loop_body_" ^ i) func in
    let b_after_loop = Block.create ~name:("after_loop_" ^ i) func in
    Block.assign !current_block index (RValue.int ctx c_index from_);
    Block.jump !current_block b_loop_cond;
    let guard = RValue.comparison ctx Gt (RValue.lvalue index) (RValue.int ctx c_index to_) in
    Block.cond_jump b_loop_cond guard b_after_loop (* on true *) b_loop_body (* on false *);
    current_block := b_loop_body;
    (match body with
    | Either.First body -> loop_proc ~env ~name body
    | Second callback -> callback (RValue.lvalue index));
    Block.assign_op !current_block index Plus (RValue.one ctx c_index);
    Block.jump !current_block b_loop_cond;
    current_block := b_after_loop
  and jit_dynamic_indices ~name ~env tensor ~tensor_idcs ~dynamic_idcs ~target_dims body =
    let host_idcs = lookup ~on_host:true ~example_only:true env tensor_idcs in
    let tensor = get_tensor state ~jit_code:loop_proc ~host_idcs tensor in
    let env =
      Array.foldi dynamic_idcs ~init:env ~f:(fun provider_dim env (Symbol s as key) ->
          let target_dim =
            RValue.int ctx c_int
              (match target_dims.(provider_dim) with
              | Shape.Dim d | Frozen d -> d
              | Parallel -> !Shape.num_parallel_tasks)
          in
          let provider_dim = RValue.int ctx c_int provider_dim in
          let idcs = lookup ~provider_dim ~on_host:false env tensor_idcs in
          let device_prov_offset = jit_array_offset ctx ~idcs ~dims:tensor.device_dims in
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
          (fst env, Map.add_exn ~key ~data @@ snd env))
    in
    loop_proc ~name ~env body
  in

  loop_proc ~name ~env body;
  !current_block

let jit_func ~name ctx (traced_store, proc) =
  let open Gccjit in
  let fkind = Function.Exported in
  let env = (Map.Poly.empty, Map.Poly.empty) in
  let task_id = Param.create ctx Type.(get ctx Int) "task_id" in
  let func = Function.create ctx fkind (Type.get ctx Void) name [ task_id ] in
  let task_init_block = Block.create ~name:("init_" ^ name) func in
  let task_finalize_block = Block.create ~name:("finalize_" ^ name) func in
  let replicated_finalize_block = Block.create ~name:("finalize_replicated_" ^ name) func in
  let main_block = Block.create ~name func in
  let state =
    {
      ctx;
      func;
      traced_store;
      task_init_block;
      task_finalize_block;
      replicated_finalize_block;
      tensors = Hashtbl.Poly.create ();
    }
  in
  let after_proc = jit_code ~name ~env ~task_id state main_block proc in
  Block.jump task_init_block main_block;
  Block.jump after_proc task_finalize_block;
  let c_index = Type.get ctx Type.Int in
  let b_after_if = Block.create ~name:("after_finalize_replicated_" ^ name) func in
  let guard = RValue.comparison ctx Eq (RValue.param task_id) (RValue.zero ctx c_index) in
  Block.cond_jump task_finalize_block guard replicated_finalize_block (* on true *) b_after_if (* on false *);
  Block.jump replicated_finalize_block b_after_if;
  Block.return_void b_after_if;
  if !Code.with_debug then
    let suf = "-gccjit-debug.c" in
    let f_name =
      if !Code.keep_files_in_run_directory then name ^ suf else Caml.Filename.temp_file (name ^ "-") suf
    in
    Context.dump_to_file ctx ~update_locs:true f_name

let error_message ~name ~prefix ?extra_error_msg ~contents exc =
  let backtrace = Caml.Printexc.get_backtrace () in
  let exc_str = Caml.Printexc.to_string exc in
  let message = Buffer.create (String.length contents + String.length backtrace + String.length exc_str) in
  let msg = Buffer.add_string message in
  msg name;
  msg ": ";
  msg prefix;
  msg exc_str;
  msg "\n";
  msg backtrace;
  (match extra_error_msg with
  | None -> ()
  | Some extra ->
      msg "\nIn the context of:\n";
      msg extra);
  (* let from_pos, to_pos = first_file_span ~contents ~message:(Buffer.contents message) in
     let from_pos = Int.min from_pos @@ String.length contents - 1 in
     let to_pos = Int.min to_pos @@ String.length contents - 1 in
     msg "\nIn code span ";
     msg error_opening_delimiter; msg "..."; msg error_closing_delimiter; msg ":\n";
     msg @@ String.sub contents ~pos:0 ~len:from_pos;
     msg error_opening_delimiter;
     msg @@ String.sub contents ~pos:from_pos ~len:(to_pos - from_pos);
     msg error_closing_delimiter;
     msg @@ String.sub contents ~pos:to_pos ~len:(String.length contents - to_pos); *)
  msg contents;
  Buffer.contents message

let jit_task_id_func ~name compiled =
  let open Gccjit in
  let ctx = Context.create_child !session_context in
  Context.set_option ctx Context.Optimization_level !optimization_level;
  (*
  if !Code.with_debug && !Code.keep_files_in_run_directory then (
    Context.set_option ctx Context.Keep_intermediates true;
    Context.set_option ctx Context.Dump_everything true);
  *)
  jit_func ~name ctx compiled;
  let result = Context.compile ctx in
  session_results := result :: !session_results;
  let routine = Result.code result name Ctypes.(int @-> returning void) in
  Context.release ctx;
  fun ~task_id -> routine task_id

let jit_unit_func ~name compiled =
  let open Gccjit in
  let ctx = Context.create_child !session_context in
  Context.set_option ctx Context.Optimization_level !optimization_level;
  (*
  if !Code.with_debug && !Code.keep_files_in_run_directory then (
    Context.set_option ctx Context.Keep_intermediates true;
    Context.set_option ctx Context.Dump_everything true);
  *)
  jit_func ~name ctx compiled;
  let result = Context.compile ctx in
  session_results := result :: !session_results;
  (* FIXME(159): adapt to not emitting task-specific code. *)
  (* let routine = Result.code result name Ctypes.(void @-> returning void) in *)
  let routine = Result.code result name Ctypes.(int @-> returning void) in
  Context.release ctx;
  (* routine *)
  fun () -> routine 0
