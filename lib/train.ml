open Base
module Ops = Ir.Ops
module Tn = Ir.Tnode
module Nd = Ir.Ndarray
module Asgns = Ir.Assignments
module Idx = Ir.Indexing
module Task = Ir.Task
open Ocannl_tensor.Operation.DSL_modules

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

(* Parameter persistence now lives in {!Persistence} (gh-ocannl-373) and is context-mediated; the
   old hosted-array-based save/restore helpers were removed with the hosted memory mode
   (gh-ocannl-333). *)

let set_materialized (a : Tn.t) = Tn.update_memory_mode a Materialized 28

(** Sets the tensor's value as materialized (device-resident, inspectable on demand via the
    context), and returns the tensor's forward code with a label-derived comment. *)
let forward t =
  let fwd = Tensor.consume_forward_code t in
  set_materialized t.Tensor.value;
  let label = Tn.debug_name t.value in
  { fwd with asgns = Asgns.Block_comment (label ^ " fwd", fwd.asgns) }

(** Returns the tensor's forward, zeroing gradients, and backprop code wrapped with label-derived
    comments. Sets the tensor's value as materialized. If [setup_for_parallel] is true (false by
    default), sets the parameters and their gradients as "non-local" (on-device). *)
let grad_update ?(setup_for_parallel = false) loss =
  set_materialized loss.Tensor.value;
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

let set_virtual (a : Tn.t) = Tn.update_memory_mode a Virtual 29

(** Materializes every non-literal embedded tensor node of [t] (so its value is inspectable on
    demand via the context). Replaces the old [every_non_literal_on_host] now that there is no
    hosted memory mode (gh-ocannl-333). *)
let every_non_literal_materialized =
  Tensor.iter_embedded ~f:(fun a ->
      if Tn.mode_is_unspecified a && not (Tn.known_constant a) then set_materialized a)

module Lazy = Utils.Lazy

let%track7_sexp to_routine (ctx : Context.t) ?(output_cd_file = false) bindings comp =
  if output_cd_file then (
    let name = Asgns.get_name_exn comp.Asgns.asgns in
    if not Utils.settings.output_debug_files_in_build_directory then
      raise
      @@ Utils.User_error
           "Train.to_routine: output_cd_file is true, but output_debug_files_in_build_directory is \
            false";
    let cd_source = Utils.output_to_build_file ~fname:(name ^ "-debug.cd") in
    let static_indices = Idx.bound_symbols bindings in
    match cd_source with
    | None -> ()
    | Some callback -> callback (Asgns.to_doc ~name ~static_indices () comp.Asgns.asgns));
  (* Materialize the guessed output nodes so they persist across calls and are inspectable on
     demand via the context (gh-ocannl-333). *)
  Set.iter (snd @@ Asgns.collect_nodes_guess_output comp.Asgns.asgns) ~f:set_materialized;
  let _ctx, routine = Context.compile ctx comp bindings in
  (* Return just the routine for backward compatibility - ctx is discarded here *)
  routine

(** [init_params] initializes the parameters of [t], via running their forward code or copying from
    the host as appropriate. If [reinit_all] is true, all parameters are reinitialized, otherwise
    only the parameters that are not in [ctx.ctx_arrays] are initialized. *)
let init_params ?(reinit_all = false) ctx bindings t =
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
  (* Materialize the parameters being initialized so they persist and are inspectable on demand. *)
  Set.iter (snd @@ Asgns.collect_nodes_guess_output comp.Asgns.asgns) ~f:set_materialized;
  (* Compile and run the initialization. Literal/ndarray-backed embedded nodes are uploaded into the
     context automatically at link time from [Host_inits] (gh-ocannl-333); there is no longer a
     separate host-array copy step here. *)
  let ctx, routine = Context.compile ctx comp bindings in
  Context.run ctx routine

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
    parameters that are not in [ctx.ctx_arrays] are initialized.

    If [output_cd_file] is true, the global setting [output_debug_files_in_build_directory] must be
    true, and the update code is output to a file before shape inference potentially crashes at
    [init_params]. *)
let%track3_sexp run_once ?(output_cd_file = false) ?(skip_init = false) ?reinit_all
    ?(bindings = IDX.empty) ~f ctx (t : Tensor.t) : Context.t =
  set_materialized t.Tensor.value;
  (* Compute the update early, to ensure the shape inference is done. *)
  let update = f t in
  if output_cd_file then (
    let name = Asgns.get_name_exn update.Asgns.asgns in
    if not Utils.settings.output_debug_files_in_build_directory then
      raise
      @@ Utils.User_error
           "Train.run_once: output_cd_file is true, but output_debug_files_in_build_directory is \
            false";
    let cd_source = Utils.output_to_build_file ~fname:(name ^ "-debug.cd") in
    let static_indices = Idx.bound_symbols bindings in
    match cd_source with
    | None -> ()
    | Some callback -> callback (Asgns.to_doc ~name ~static_indices () update.Asgns.asgns));
  let ctx =
    if skip_init || Set.is_empty t.params then ctx
    else init_params ?reinit_all ctx bindings t
  in
  let ctx, routine = Context.compile ctx update bindings in
  Context.run ctx routine

(** Context-based versions of training functions for the new simplified API *)

(** [forward_once] is a wrapper around {!run_once} that runs the forward code of [t]. *)
let forward_once ?output_cd_file ?(skip_init = false) ?reinit_all ?(bindings = IDX.empty) ctx t =
  let ctx = run_once ?output_cd_file ~skip_init ?reinit_all ~bindings ~f:forward ctx t in
  (* FIXME: this is going away soon. *)
  Tensor.remove_bprop_root t;
  ctx

(** [update_once] is a wrapper around {!run_once} that runs the gradient update code of [t]: both
    forward and backprop. *)
let update_once ?output_cd_file ?(skip_init = false) ?reinit_all ?(bindings = IDX.empty) ctx t =
  run_once ?output_cd_file ~skip_init ?reinit_all ~bindings ~f:grad_update ctx t

(* For-print cache (gh-ocannl-333 AC 5): the [%cd "for_print" =: t] trick. When a tensor's value is
   not materialized in the printing context, we compile and run a copy of it ([for_print = t + 0])
   into a fresh device-resident node, and register that node as a for-print proxy so the printer can
   read the tensor's value through it. The copy tensor is cached by source-node id so the for-print
   node is reused across repeated prints (the proposal's "cache of for-print tensor nodes"). *)
let for_print_cache : (int, Tensor.t) Hashtbl.t = Hashtbl.create (module Int)

let ensure_printable (ctx : Context.t) (t : Tensor.t) : Context.t =
  if Context.mem ctx t.Tensor.value then ctx
  else begin
    let for_print =
      Hashtbl.find_or_add for_print_cache t.Tensor.value.Tn.id ~default:(fun () ->
          let%op for_print = t + 0 in
          for_print)
    in
    let ctx = forward_once ctx for_print in
    Context.register_for_print ~src:t.Tensor.value ~proxy:for_print.Tensor.value;
    ctx
  end

(** [printf] is a wrapper around {!Tensor.print} that assumes [~force:true], and by default sets
    [~with_code:false], [~with_grad:true], and [~style:`Default]. It takes an explicit context and
    retrieves values on demand (gh-ocannl-333). If the tensor's value is not already materialized in
    [ctx], it is recomputed via the [for_print] copy trick so real values are still shown. *)
let%debug7_sexp printf ?here ?(with_grad = true) ?(with_code = false) ?(with_low_level = false)
    ?(style = `Default) (ctx : Context.t) (t : Tensor.t) : unit =
  let ctx = ensure_printable ctx t in
  Tensor.print ?here ~force:true ~ctx ~with_grad ~with_code ~with_low_level style t

(** [printf_tree] is a wrapper around {!Tensor.print_tree} that assumes [~force:true], and by
    default sets [~with_value:true], [~with_grad:true], and [~depth:9]. It takes an explicit context
    and retrieves values on demand (recomputing via [for_print] if not already materialized). *)
let printf_tree ?here ?with_value ?(with_grad = true) ?(depth = 9) (ctx : Context.t) t =
  let ctx = ensure_printable ctx t in
  Tensor.print_tree ?here ~force:true ~ctx ?with_value ~with_grad ~depth t
