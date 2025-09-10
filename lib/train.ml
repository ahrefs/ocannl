open Base
module Ops = Ir.Ops
module Tn = Ir.Tnode
module Nd = Ir.Ndarray
open Ocannl_tensor.Operation.DSL_modules
module Asgns = Ir.Assignments
module Idx = Ir.Indexing
module Task = Ir.Task
module BT = Ir.Backend_intf

module type Backend = Ir.Backend_intf.Backend

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_TRAIN=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_TRAIN"]

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

let run ctx routine = ignore (Context.run ctx routine)

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
let forward t =
  let fwd = Tensor.consume_forward_code t in
  set_hosted t.Tensor.value;
  let label = Tn.debug_name t.value in
  { fwd with asgns = Asgns.Block_comment (label ^ " fwd", fwd.asgns) }

(** Returns the tensor's forward, zeroing gradients, and backprop code wrapped with label-derived
    comments. Sets the tensor's value as "fully on host". If [setup_for_parallel] is true (false by
    default), sets the parameters and their gradients as "non-local" (on-device). *)
let grad_update ?(setup_for_parallel = false) loss =
  set_hosted loss.Tensor.value;
  if setup_for_parallel then
    Set.iter loss.Tensor.params ~f:(fun p ->
        set_materialized (Option.value_exn ~here:[%here] p.diff).grad);
  (* Note: the %cd syntax for [loss.grad] does not modify roots. *)
  [%cd
    ~~(loss "forward and gradient update";
       loss.forward;
       ~~(loss "zero grads and backprop";
          loss.zero_grads;
          loss.grad =: 1;
          loss.backprop))]

(** See: https://github.com/tinygrad/tinygrad/blob/master/tinygrad/nn/optim.py *)
let sgd_one ~learning_rate ?(momentum = 0.0) ?(weight_decay = 0.0) ?(nesterov = false) p =
  if Option.is_none p.Tensor.diff then
    raise @@ Tensor.Session_error ("Train.sgd_one: not differentiable", Some p);
  [%cd
    ~~(p "param sgd step";
       { sgd_delta } =: p.grad + (!.weight_decay *. p);
       if Float.(momentum > 0.0) then (
         { sgd_momentum } =: (!.momentum *. sgd_momentum) + sgd_delta;
         if nesterov then sgd_delta =+ !.momentum *. sgd_momentum else sgd_delta =: sgd_momentum);
       p =- learning_rate * sgd_delta ~logic:".")]

let sgd_update ~learning_rate ?momentum ?weight_decay ?nesterov loss =
  let f = sgd_one ~learning_rate ?momentum ?weight_decay ?nesterov in
  let comp = Set.to_list loss.Tensor.params |> List.map ~f |> Asgns.sequence in
  { comp with asgns = Asgns.Block_comment ("sgd_update", comp.asgns) }

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

let to_routine (ctx : Context.t) ?(hosted = true) bindings comp =
  if hosted then Set.iter (snd @@ Asgns.collect_nodes_guess_output comp.Asgns.asgns) ~f:set_hosted;
  let _ctx, routine = Context.compile ctx comp bindings in
  (* Return just the routine for backward compatibility - ctx is discarded here *)
  routine

(** [init_params] initializes the parameters of [t], via running their forward code or copying from
    the host as appropriate. If [reinit_all] is true, all parameters are reinitialized, otherwise
    only the parameters that are not in [ctx.ctx_arrays] are initialized. *)
let init_params ?(reinit_all = false) ?(hosted = true) ctx bindings t =
  let comp = 
    if reinit_all then Tensor.init_params t
    else 
      (* Check which params are already initialized *)
      let skip = Map.empty (module Tn) in
      Set.fold t.Tensor.params ~init:skip ~f:(fun skip p ->
        if Context.is_initialized ctx p.Tensor.value then
          Map.set skip ~key:p.Tensor.value ~data:()
        else skip)
      |> fun skip -> Tensor.init_params ~skip t
  in
  if hosted then Set.iter (snd @@ Asgns.collect_nodes_guess_output comp.Asgns.asgns) ~f:set_hosted;
  (* Compile and run the initialization *)
  let ctx, routine = Context.compile ctx comp bindings in
  let ctx = Context.run ctx routine in
  (* Mark embedded nodes as initialized via init_from_host *)
  Set.fold comp.Asgns.embedded_nodes ~init:ctx ~f:(fun ctx tn ->
    if not (Context.is_initialized ctx tn) then 
      Context.init_from_host_deprecated ctx tn
    else ctx)

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

(** [run_once] is a wrapper around {!init_params} that additionally runs code of [f t] and returns
    the context. If [skip_init] is true (false by default), no initialization is performmed. If
    [reinit_all] is true (false by default), all parameters are reinitialized, otherwise only the
    parameters that are not in [ctx.ctx_arrays] are initialized. *)
let%track3_sexp run_once ?(hosted = true) ?(skip_init = false) ?reinit_all
    ?(bindings = IDX.empty) ~f ctx t =
  if hosted then set_hosted t.Tensor.value;
  (* Compute the update early, to ensure the shape inference is done. *)
  let update = f t in
  let ctx =
    if skip_init || Set.is_empty t.params then ctx
    else init_params ?reinit_all ~hosted ctx bindings t
  in
  let ctx, routine = Context.compile ctx update bindings in
  Context.run ctx routine

(** Context-based versions of training functions for the new simplified API *)

(** [forward_once] is a wrapper around {!run_once} that runs the forward code of [t]. *)
let forward_once ?(hosted = true) ?(skip_init = false) ?reinit_all ?(bindings = IDX.empty) ctx t =
  let ctx = run_once ~hosted ~skip_init ?reinit_all ~bindings ~f:forward ctx t in
  (* FIXME: this is going away soon. *)
  Tensor.remove_bprop_root t;
  ctx

(** [update_once] is a wrapper around {!run_once} that runs the gradient update code of [t]: both
    forward and backprop. *)
let update_once ?(hosted = true) ?(skip_init = false) ?reinit_all ?(bindings = IDX.empty) ctx t =
  run_once ~hosted ~skip_init ?reinit_all ~bindings ~f:grad_update ctx t

(** [printf] is a wrapper around {!Tensor.print} that assumes [~force:true], and by default sets
    [~with_code:false], [~with_grad:true], and [~style:`Default]. *)
let printf ?here ?(with_grad = true) ?(with_code = false) ?(with_low_level = false)
    ?(style = `Default) t =
  Tensor.print ?here ~force:true ~with_grad ~with_code ~with_low_level style t

(** [printf_tree] is a wrapper around {!Tensor.print_tree} that assumes [~force:true], and by
    default sets [~with_value:true], [~with_grad:true], and [~depth:9]. *)
let printf_tree ?here ?with_value ?(with_grad = true) ?(depth = 9) t =
  Tensor.print_tree ?here ~force:true ?with_value ~with_grad ~depth t
