open Base
module Ops = Ir.Ops
module Tn = Ir.Tnode
module Nd = Ir.Ndarray
module NTDSL = Operation.NTDSL
module Asgns = Ir.Assignments
module Idx = Ir.Indexing
module Task = Ir.Task
module Rand = Ir.Rand.Lib
module BT = Ir.Backend_intf

module type Backend = Ir.Backend_intf.Backend

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

module CDSL = struct
  let half = Ir.Ops.half
  let single = Ir.Ops.single
  let double = Ir.Ops.double
  let virtualize_settings = Ir.Low_level.virtualize_settings

  let enable_all_debugs ?(debug_logs = false) ?(hosted_only = true) () =
    Utils.set_log_level @@ max 2 @@ Utils.settings.log_level;
    Utils.settings.output_debug_files_in_build_directory <- true;
    if hosted_only then virtualize_settings.enable_device_only <- false;
    if debug_logs then Utils.settings.debug_log_from_routines <- true

  let disable_all_debugs ?(restore_defaults = false) () =
    Utils.settings.debug_log_from_routines <- false;
    Utils.set_log_level 0;
    Utils.settings.output_debug_files_in_build_directory <- false;
    if restore_defaults then virtualize_settings.enable_device_only <- true
end

module IDX = struct
  let empty = Idx.Empty
  let get_static_symbol = Idx.get_static_symbol
  let find_exn = Idx.find_exn
end

let run jitted = Task.run jitted.BT.schedule

(* let save_params t = let is_grad, ident = Tn.no_grad_ident_label t.Tensor.value in assert (not
   is_grad); let file_name = Option.value_or_thunk ~default:(fun () -> invalid_arg
   "Train.save_params: root tensor is not named") ident in let with_name p = let is_grad, ident =
   Tn.no_grad_ident_label p.Tensor.value in assert (not is_grad); ( p.Tensor.value,
   Option.value_or_thunk ~default:(fun () -> invalid_arg @@ "Train.save_params: parameter is not
   named: " ^ Tn.debug_name p.Tensor.value) ident ) in let with_names = get_params t |> Set.elements
   |> List.map ~f:with_name in let out_file = Npy.Npz.open_out file_name in List.iter with_names
   ~f:(fun (v, name) -> let f arr = Npy.Npz.write out_file name arr in Nd.map { f } @@
   Option.value_exn ~here:[%here] @@ Lazy.force v.array) *)

(* let restore_params t = let is_grad, ident = Tn.no_grad_ident_label t.Tensor.value in assert (not
   is_grad); let file_name = Option.value_or_thunk ~default:(fun () -> invalid_arg
   "Train.restore_params: root tensor is not named") ident in let with_name p = let is_grad, ident =
   Tn.no_grad_ident_label p.Tensor.value in assert (not is_grad); ( p.Tensor.value,
   Option.value_or_thunk ~default:(fun () -> invalid_arg @@ "Train.restore_params: parameter is not
   named: " ^ Tn.debug_name p.Tensor.value) ident ) in let with_names = get_params t |> Set.elements
   |> List.map ~f:with_name in let in_file = Npy.Npz.open_in file_name in List.iter with_names
   ~f:(fun (v, name) -> let f arr = Npy.Npz.restore in_file name arr in Nd.map { f } @@
   Option.value_exn ~here:[%here] @@ Lazy.force v.array) *)
let set_on_host ?(from_device = true) (a : Tn.t) =
  let memtype = if from_device then Tn.(Changed_on_devices Unset) else Volatile in
  Tn.update_memory_mode a (Hosted memtype) 27

let set_materialized (a : Tn.t) = Tn.update_memory_mode a Materialized 28

let set_hosted (a : Tn.t) =
  if Tn.known_constant a then Tn.update_memory_mode a (Hosted Constant) 411
  else Tn.update_memory_mode a (Hosted (Changed_on_devices Unset)) 412

(** Sets the tensor's value as "fully on host", returns the tensor's forward code with a
    label-derived comment. *)
let forward ?(disable_rootness_check = false) t =
  let fwd = if disable_rootness_check then t.Tensor.forward else Tensor.consume_forward_code t in
  set_hosted t.Tensor.value;
  let label = Tn.debug_name t.value in
  { fwd with asgns = Asgns.Block_comment (label ^ " fwd", fwd.asgns) }

let diff_or_error t provenance =
  Option.value_or_thunk t.Tensor.diff ~default:(fun () ->
      raise @@ Tensor.Session_error (provenance ^ ": tensor is not differentiable", Some t))

let grad_update_nochecks loss =
  let diff = diff_or_error loss "Train.grad_update_nochecks" in
  let fwd_bprop =
    [%cd
      ~~(loss "gradient update";
         ~~(loss "fwd";
            loss.forward);
         ~~(loss "zero grads";
            Asgns.to_comp diff.zero_grads);
         loss.grad =: 1;
         ~~(loss "bprop";
            diff.backprop))]
  in
  fwd_bprop

(** Returns the tensor's forward, zeroing gradients, and backprop code wrapped with label-derived
    comments. Sets the tensor's value as "fully on host". If [setup_for_parallel] is true (false by
    default), sets the parameters and their gradients as "non-local" (on-device). *)
let grad_update ?(disable_rootness_check = false) ?(setup_for_parallel = false) loss =
  set_hosted loss.Tensor.value;
  if setup_for_parallel then
    Set.iter loss.Tensor.params ~f:(fun p ->
        set_materialized (Option.value_exn ~here:[%here] p.diff).grad);
  let fwd =
    if disable_rootness_check then loss.Tensor.forward else Tensor.consume_forward_code loss
  in
  let diff = diff_or_error loss "Train.grad_update" in
  let zero_grads, bprop =
    if disable_rootness_check then (diff.zero_grads, diff.backprop)
    else Tensor.consume_backprop_code loss
  in
  (* Note: the %cd syntax for [loss.grad] does not modify roots. *)
  [%cd
    ~~(loss "gradient update";
       ~~(loss "fwd";
          fwd);
       ~~(loss "zero grads";
          Asgns.to_comp zero_grads);
       loss.grad =: 1;
       ~~(loss "bprop";
          bprop))]

(** See: https://github.com/tinygrad/tinygrad/blob/master/tinygrad/nn/optim.py *)
let sgd_one ~learning_rate ?(momentum = 0.0) ?(weight_decay = 0.0) ?(nesterov = false) p =
  if Option.is_none p.Tensor.diff then
    raise @@ Tensor.Session_error ("Train.sgd_one: not differentiable", Some p);
  [%cd
    ~~(p "param sgd step";
       "sgd_delta" =: p.grad + (!.weight_decay *. p);
       if Float.(momentum > 0.0) then (
         "sgd_momentum" =: (!.momentum *. sgd_momentum) + sgd_delta;
         if nesterov then sgd_delta =+ !.momentum *. sgd_momentum else sgd_delta =: sgd_momentum);
       p =- learning_rate * sgd_delta ~logic:".")]

let sgd_update ~learning_rate ?momentum ?weight_decay ?nesterov loss =
  let code =
    loss.Tensor.params |> Set.to_list
    |> List.map ~f:(sgd_one ~learning_rate ?momentum ?weight_decay ?nesterov)
    |> Asgns.sequence
  in
  [%cd
    ~~(loss "sgd update";
       code)]

(** All and only bindings with associated ranges are iterated, with the binding's initial value
    lost. Bindings without ranges remain at their initial values. *)
let%track3_sexp sequential_loop ~f lowered_bindings =
  let rec loop = function
    | [] -> f ()
    | ({ Idx.static_range = None; static_symbol = _ }, _) :: more -> loop more
    | ({ Idx.static_range = Some range; static_symbol = _ }, idx) :: more ->
        let old_idx = !idx in
        for i = 0 to range - 1 do
          idx := i;
          loop more
        done;
        idx := old_idx
  in
  loop lowered_bindings

(** Distributes iterated indices to workers in a round-robin fashion. All and only bindings with
    associated ranges are iterated, with the binding's initial value lost. Bindings without ranges
    remain at their initial values. [sync] is called after each round of calling all workers, and at
    the end if needed, with the number of workers called during the round. *)
let%track3_sexp round_robin fs parallel_jitbs jitbs ~sync : unit =
  let num_streams : int = Array.length fs in
  assert (Array.length parallel_jitbs = num_streams);
  let pos = ref 0 in
  let rec loop = function
    | [] ->
        fs.(!pos % num_streams) ();
        Int.incr pos;
        if !pos % num_streams = 0 then sync num_streams
    | ({ Idx.static_range = None; static_symbol = _ }, _) :: more -> loop more
    | (({ Idx.static_range = Some range; static_symbol = _ } as s), idx)
      :: ({ Idx.static_range = None; static_symbol = _ }, _)
      :: more
    | (({ Idx.static_range = Some range; static_symbol = _ } as s), idx) :: more ->
        for i = 0 to range - 1 do
          idx := i;
          if List.is_empty more then Idx.find_exn parallel_jitbs.(!pos % num_streams) s := i
          else Array.iter parallel_jitbs ~f:(fun jb -> Idx.find_exn jb s := i);
          loop more
        done
  in
  loop jitbs;
  if !pos % num_streams <> 0 then sync (!pos % num_streams)

let%track3_sexp round_robin_dry_run ~num_streams jitbs ~dry_sync : unit =
  let pos = ref 0 in
  let rec loop = function
    | [] ->
        Int.incr pos;
        if !pos % num_streams = 0 then dry_sync num_streams
    | ({ Idx.static_range = None; static_symbol = _ }, _) :: more -> loop more
    | ({ Idx.static_range = Some range; static_symbol = _ }, idx)
      :: ({ Idx.static_range = None; static_symbol = _ }, _)
      :: more
    | ({ Idx.static_range = Some range; static_symbol = _ }, idx) :: more ->
        for i = 0 to range - 1 do
          idx := i;
          loop more
        done
  in
  loop jitbs;
  if !pos % num_streams <> 0 then dry_sync (!pos % num_streams)

let set_virtual (a : Tn.t) = Tn.update_memory_mode a Virtual 29

let every_non_literal_on_host =
  Tensor.iter_embedded ~f:(fun a ->
      if Tn.mode_is_unspecified a && not (Tn.known_constant a) then set_hosted a)

module Lazy = Utils.Lazy

(** Performs one optimization step, potentially in parallel (if [grad_updates] are linked with
    different streams or devices). All jitted code must have the same bindings. Iterates over
    bindings with ranges, calling one of [grad_updates] in a round-robin fashion, and performs the
    following synchronization each time all [grad_updates] have been called:
    - merges all gradients into the device of [grad_updates.(0)],
    - calls [sgd_update],
    - copies all parameters from the [grad_updates.(0)] device to the other devices, if needed,
    - calls [post_sync] with the number of devices synced since the previous sync.

    All and only bindings with associated ranges are iterated, with the binding's initial value
    lost. Bindings without ranges remain at their initial values. *)
let%track3_sexp parallel_update (type buffer_ptr dev runner event optimize_ctx)
    (module Backend : Backend
      with type buffer_ptr = buffer_ptr
       and type dev = dev
       and type runner = runner
       and type optimize_ctx = optimize_ctx
       and type event = event) ~(grad_updates : Backend.context BT.routine array)
    ~(sgd_update : Backend.context BT.routine) ~copy_to_merge ~post_sync loss =
  assert (not @@ Array.is_empty grad_updates);
  let num_streams : int = Array.length grad_updates in
  let bindings : Idx.static_symbol list = List.map ~f:fst sgd_update.bindings in
  let occupancies_dst_src =
    Array.init num_streams ~f:(fun _ -> Array.create ~len:num_streams false)
  in
  (* to_, from positions correspond to the contexts (and devices) of grad_updates at the
     position. *)
  let dry_merge ~from ~to_ = occupancies_dst_src.(to_).(from) <- true in
  let dry_sync devices_to_sync = Utils.parallel_merge dry_merge devices_to_sync in
  round_robin_dry_run ~num_streams sgd_update.bindings ~dry_sync;
  [%debug_notrace
    assert (
      Array.for_all grad_updates ~f:(fun upd ->
          [%equal: Idx.static_symbol list] bindings @@ List.map ~f:fst upd.bindings))];
  let all_params : Tensor.t array = Set.to_array loss.Tensor.params in
  if Array.is_empty all_params then
    raise @@ Tensor.Session_error ("Train.parallel_update: no parameters", Some loss);
  let _occupancies_debug : bool array array = occupancies_dst_src in
  let ctxs = [%debug_notrace Array.map grad_updates ~f:(fun upd -> upd.context)] in
  let occupancy_dst ~dst_n = Array.exists ~f:Fn.id occupancies_dst_src.(dst_n) in
  let grad_merges =
    Array.map all_params ~f:(fun p ->
        [%cd
          ~~("merging gradient of" p;
             p.grad =+ p.grad.merge)])
  in
  let grad_merges_to : Backend.context BT.routine option array array =
    (* For now, we need all params on all devices. *)
    let occupancy ~name:_ ~src_n:_ = true in
    Array.mapi ctxs ~f:(fun dst_n ctx ->
        if occupancy_dst ~dst_n then
          snd
          @@ Backend.(
               link_batch ctx @@ compile_batch ctx.optimize_ctx ~occupancy Idx.Empty grad_merges)
        else [||])
  in
  (* We can cache scheduling, because merging and copying does not depend on static indexing. *)
  let loss_merge =
    Backend.(
      link sgd_update.BT.context
      @@ compile sgd_update.context.optimize_ctx Idx.Empty
           [%cd
             ~~("merging" loss;
                loss.value =+ loss.value.merge)])
  in
  let mbuf_use sched = if copy_to_merge then (BT.Copy, false) else (BT.Streaming_for sched, true) in
  (* Since each device has its own queue, we can iterate over devices in the outer loop. *)
  let merge_grads ~(from : int) ~(to_ : int) : unit =
    Array.iteri all_params ~f:(fun i p ->
        let grad_merge =
          Option.value_exn ~here:[%here] ~message:(Tn.debug_name p.value) grad_merges_to.(to_).(i)
        in
        let into_merge_buffer, streaming = mbuf_use grad_merge.schedule in
        assert (
          Backend.device_to_device (Option.value_exn ~here:[%here] p.diff).grad ~into_merge_buffer
            ~dst:ctxs.(to_) ~src:ctxs.(from));
        if not streaming then Task.run grad_merge.schedule)
  in
  let merge_loss ~src =
    let into_merge_buffer, streaming = mbuf_use loss_merge.schedule in
    assert (Backend.device_to_device loss.value ~into_merge_buffer ~dst:sgd_update.context ~src);
    if not streaming then Task.run loss_merge.schedule
  in
  (* FIXME: missing device-to-host? *)
  let%track3_sexp sync (devices_to_sync : int) : unit =
    Utils.parallel_merge merge_grads devices_to_sync;
    Task.run sgd_update.schedule;
    Array.iteri ctxs ~f:(fun i src -> if i <> 0 then merge_loss ~src);
    (* We will need to update params on all devices! Not only the ones that computed gradients. *)
    for to_ = 1 to num_streams - 1 do
      Array.iter all_params ~f:(fun p ->
          (* Allow the params to be shared across streams. *)
          ignore
            (Backend.device_to_device p.value ~into_merge_buffer:No ~dst:ctxs.(to_)
               ~src:sgd_update.context))
    done;
    post_sync ~num_synced_devices:devices_to_sync
  in
  let lowered_bindings = [%debug_notrace Array.map grad_updates ~f:(fun upd -> upd.bindings)] in
  let fs = [%debug_notrace Array.map grad_updates ~f:(fun upd () -> Task.run upd.schedule)] in
  fun () -> round_robin fs lowered_bindings sgd_update.bindings ~sync

(* Note: this type signature looks ugly, but it will get simple again with modular explicits. *)
let get_all_suggested_streams ?(max_num_streams : int option) (type buffer_ptr dev runner event)
    (module Backend : Backend
      with type buffer_ptr = buffer_ptr
       and type dev = dev
       and type runner = runner
       and type event = event) =
  let max_num_streams = Option.value max_num_streams ~default:Int.max_value_30_bits in
  let num_devices = min max_num_streams @@ Backend.num_devices () in
  let devices = Array.init num_devices ~f:(fun ordinal -> Backend.get_device ~ordinal) in
  let result =
    Array.folding_mapi devices ~init:0 ~f:(fun ordinal num_collected device ->
        let remaining_devices = num_devices - ordinal - 1 in
        let max_current = Backend.suggested_num_streams device in
        let take_current = min max_current @@ (max_num_streams - remaining_devices) in
        ( num_collected + take_current,
          Array.init take_current ~f:(fun _stream_no -> Backend.new_stream device) ))
    |> Array.concat_map ~f:Fn.id
  in
  (devices, result)

let to_routine (type buffer_ptr dev runner event optimize_ctx)
    (module Backend : Backend
      with type buffer_ptr = buffer_ptr
       and type dev = dev
       and type runner = runner
       and type event = event
       and type optimize_ctx = optimize_ctx) (context : Backend.context) ?(hosted = true) ?name
    bindings comp =
  if hosted then Set.iter (Asgns.guess_output_nodes comp.Asgns.asgns) ~f:set_hosted;
  Backend.link context @@ Backend.compile context.optimize_ctx ?name bindings comp

(** [init_params] initializes the parameters of [t], via running their forward code or copying from
    the host as appropriate. If [reinit_all] is true, all parameters are reinitialized, otherwise
    only the parameters that are not in [ctx.ctx_arrays] are initialized. *)
let init_params (type buffer_ptr dev runner event optimize_ctx)
    (module Backend : Backend
      with type buffer_ptr = buffer_ptr
       and type dev = dev
       and type runner = runner
       and type event = event
       and type optimize_ctx = optimize_ctx) ?(ctx : Backend.context option) ?(reinit_all = false)
    ?hosted ?name bindings t =
  let ctx =
    match ctx with
    | Some ctx -> ctx
    | None -> Backend.make_context @@ Backend.new_stream @@ Backend.get_device ~ordinal:0
  in
  let skip = if reinit_all then None else Some ctx.ctx_arrays in
  let comp = Tensor.init_params ?skip t in
  let init = to_routine (module Backend) ctx ?hosted ?name bindings comp in
  let ctx =
    Set.fold comp.Asgns.embedded_nodes ~init:init.context ~f:(fun ctx tn ->
        if not (Map.mem ctx.ctx_arrays tn) then Backend.init_from_host ctx tn else ctx)
  in
  run init;
  ctx

type example_train_result = {
  inputs : Tensor.t;
  outputs : Tensor.t;
  model_result : Tensor.t;  (** Do not use [model_result] for deriving gradients. *)
  infer_callback : float array -> float array;
      (** Computes the output for the given input via the [model_result] tensor. Note:
          [infer_callback] is inefficient as it is not batched. *)
  rev_batch_losses : float list;
  rev_epoch_losses : float list;
  learning_rates : float list;
  used_memory : int;
}

let example_train_loop ?(disable_rootness_check = false) ~seed ~batch_size ~init_lr ?lr_schedule
    ?(copy_to_merge = false) ?max_num_streams ~data_len ~epochs ~inputs ~outputs ~model ~loss_fn
    ~weight_decay ?per_batch_callback ?per_epoch_callback ?(per_epoch_debug_streams = false)
    (module Backend : Backend) () =
  let module TDSL = Operation.TDSL in
  let module NTDSL = Operation.NTDSL in
  Rand.init seed;
  let devices, streams = get_all_suggested_streams ?max_num_streams (module Backend) in
  let num_streams = Array.length streams in
  let contexts = Array.map streams ~f:(Backend.make_context ?ctx_arrays:None) in
  let init_mem = Array.fold devices ~init:0 ~f:(fun acc dev -> acc + Backend.get_used_memory dev) in
  let minibatch_size = batch_size / num_streams in
  let n_minibatches = data_len / minibatch_size in
  if n_minibatches <= 0 then
    invalid_arg
      [%string
        "Train.example_train_loop: too little data: %{data_len#Int} for minibatch size: \
         %{minibatch_size#Int} = %{batch_size#Int} / %{num_streams#Int} streams"];
  assert (epochs > 0);
  let inputs = inputs ~b:[ n_minibatches; minibatch_size ] in
  let outputs = outputs ~b:[ n_minibatches; minibatch_size ] in
  (* This is the joint number of steps done by the round-robin scheduler across devices. *)
  let steps = epochs * n_minibatches in
  Utils.settings.fixed_state_for_init <- Some seed;
  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_minibatches IDX.empty in
  let step_n, bindings = IDX.get_static_symbol bindings in
  let%op input = inputs @| batch_n in
  let%op expectation = outputs @| batch_n in
  let rev_batch_losses = ref [] in
  let rev_epoch_losses = ref [] in
  let learning_rates = ref [] in
  let%op loss_tensor = loss_fn ~output:(model input) ~expectation in
  let%op scalar_loss = (loss_tensor ++ "...|... => 0") /. !..batch_size in
  let update = grad_update ~disable_rootness_check ~setup_for_parallel:true scalar_loss in
  (* Define learning_rate after scalar_loss is compiled, to not trigger rootness sanitizer. *)
  let%op learning_rate =
    match lr_schedule with
    | None -> !.init_lr *. ((2 *. !..steps) - !@step_n) /. !..steps
    | Some schedule -> schedule ~batch_n ~step_n
  in
  (* Note: constants at default half-prec are automatically upcasted when they exceed
     Utils.settings.check_half_prec_constants_cutoff, no need to upcast learning_rate.value. *)
  set_hosted learning_rate.value;
  let sgd = sgd_update ~learning_rate ~weight_decay scalar_loss in
  (* We initialize params on stream 0, and copy them to the other streams. *)
  let ctx0 = init_params (module Backend) ~ctx:contexts.(0) bindings scalar_loss in
  let grad_update = Backend.compile ctx0.optimize_ctx ?name:None bindings update in
  let contexts =
    Array.mapi contexts ~f:(fun to_ init ->
        if to_ = 0 then ctx0
        else
          Set.fold scalar_loss.Tensor.params ~init ~f:(fun dst p ->
              Backend.init_from_device p.value ~dst ~src:ctx0))
  in
  let grad_updates = Array.map contexts ~f:(fun ctx -> Backend.link ctx grad_update) in
  let sgd_update = to_routine (module Backend) grad_updates.(0).context ?name:None bindings sgd in
  Tensor.log_debug_info ~from_log_level:2 inputs;
  Tensor.log_debug_info ~from_log_level:2 outputs;
  let open Operation.At in
  let epoch_loss = ref 0. in
  let step_ref = IDX.find_exn sgd_update.bindings step_n in
  let batch_ref = IDX.find_exn sgd_update.bindings batch_n in
  let update =
    parallel_update
      (module Backend)
      ~grad_updates ~sgd_update scalar_loss ~copy_to_merge
      ~post_sync:(fun ~num_synced_devices ->
        step_ref := !step_ref + num_synced_devices;
        let batch_loss = scalar_loss.@[0] in
        epoch_loss := !epoch_loss +. batch_loss;
        rev_batch_losses := batch_loss :: !rev_batch_losses;
        Option.iter per_batch_callback ~f:(fun f ->
            f ~at_batch:!batch_ref ~at_step:!step_ref ~learning_rate:learning_rate.@[0] ~batch_loss
              ~epoch_loss:!epoch_loss))
  in
  if Utils.settings.log_level > 1 then (
    Stdlib.Printf.printf "\nTraining...\n%!";
    Tn.log_accessible_headers ());
  for epoch = 0 to epochs - 1 do
    epoch_loss := 0.;
    Utils.capture_stdout_logs update;
    learning_rates := learning_rate.@[0] :: !learning_rates;
    rev_epoch_losses := !epoch_loss :: !rev_epoch_losses;
    Option.iter per_epoch_callback ~f:(fun f ->
        f ~at_step:!step_ref ~at_epoch:epoch ~learning_rate:learning_rate.@[0]
          ~epoch_loss:!epoch_loss);
    let _debug_at pos =
      Array.iter streams ~f:(fun s ->
          Stdlib.Format.printf "Stream %d debug %s:@ %a\n%!" s.stream_id pos Sexp.pp_hum
          @@ Backend.get_debug_info s)
    in
    if per_epoch_debug_streams then _debug_at "before sync";
    (* TODO: there should be nothing pending left to sync. And it offers only a slight speed up. *)
    Array.iter devices ~f:Backend.(fun d -> sync_device d)
    (* This is now cleaned up by await. *)
    (* if per_epoch_debug_streams then _debug_at "after sync" *)
  done;
  (* Using %cd instead of %op to avoid being asked to initialize [infer]. *)
  let%cd model_result = model "infer_input" in
  let infer_fwd =
    if disable_rootness_check then model_result.Tensor.forward
    else Tensor.consume_forward_code model_result
  in
  if not disable_rootness_check then Tensor.remove_bprop_root model_result;
  set_on_host model_result.Tensor.value;
  (* By using sgd_update.context, maybe we don't need to copy the parameters back to the host. *)
  let routine =
    Backend.(
      link sgd_update.context
      @@ compile sgd_update.context.optimize_ctx IDX.empty
           [%cd
             ~~("infer " model_result;
                infer_fwd)])
  in
  let infer_callback values =
    Tn.set_values infer_input.value values;
    (* For the gccjit backend, infer is only on host, not on device. For cuda, this will be
       needed. *)
    Utils.capture_stdout_logs @@ fun () ->
    run routine;
    Tn.get_values model_result.value
  in
  let used_memory =
    Array.fold devices ~init:0 ~f:(fun acc dev -> acc + Backend.get_used_memory dev) - init_mem
  in
  {
    inputs;
    outputs;
    model_result;
    infer_callback;
    rev_batch_losses = !rev_batch_losses;
    rev_epoch_losses = !rev_epoch_losses;
    learning_rates = !learning_rates;
    used_memory;
  }

(** [forward_and_ctx] is a wrapper around {!init_params} that additionally runs code of [t] and
    returns the context. If [skip_init] is true (false by default), no initialization is performmed.
    If [reinit_all] is true (false by default), all parameters are reinitialized, otherwise only the
    parameters that are not in [ctx.ctx_arrays] are initialized. *)
let%track3_sexp forward_and_ctx ?(hosted = true) ?(skip_init = false) ?reinit_all
    ?(disable_rootness_check = false) (type buffer_ptr dev runner event optimize_ctx)
    (module Backend : Backend
      with type buffer_ptr = buffer_ptr
       and type dev = dev
       and type runner = runner
       and type optimize_ctx = optimize_ctx
       and type event = event) ctx ?(bindings = IDX.empty) t =
  (* TODO: this will get nicer with modular explicits. *)
  if hosted then set_hosted t.Tensor.value;
  let ctx =
    if skip_init || Set.is_empty t.params then ctx
    else init_params (module Backend) ~ctx ~hosted ?reinit_all bindings t
  in
  let routine =
    Backend.(link ctx @@ compile ctx.optimize_ctx bindings @@ forward ~disable_rootness_check t)
  in
  if not disable_rootness_check then Tensor.remove_bprop_root t;
  Task.run routine.schedule;
  routine.context

(** [forward_and_force] is a wrapper around {!forward_and_ctx} that additionally forces the tensor's
    value and ensures it is transferred back to host as needed, see the setting
    {!Utils.settings.automatic_host_transfers}. If [skip_init] is true (false by default), no
    initialization is performmed. The resulting context is ignored.

    Note: [Tensor.print ~force:true] also has this effect, so: using [forward_and_force] you don't
    need to pass [~force:true], and if you need the context and also to print the result, you can
    combine {!forward_and_ctx} and [Tensor.print ~force:true]. *)
let forward_and_force ?hosted ?skip_init ?reinit_all ?disable_rootness_check backend ctx ?bindings t
    =
  (* FIXME: to properly forget we need to free the incrementally-allocated memory! *)
  ignore
  @@ forward_and_ctx ?hosted ?skip_init ?reinit_all ?disable_rootness_check backend ctx ?bindings t;
  ignore (Lazy.force t.value.array);
  Tn.do_read t.value
