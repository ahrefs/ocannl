open Base
module Tn = Ir.Tnode
module Nd = Ir.Ndarray
module Asgns = Ir.Assignments
module Idx = Ir.Indexing
module Task = Ir.Task
module Backends = Context.Backends_deprecated
open Ocannl_tensor.Operation.DSL_modules

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(** Gradient reduction mode used by {!data_parallel} when all-reducing parameter gradients across
    data-parallel shards. [Sum] adds the per-shard gradients; [Mean] additionally divides by the
    shard count. *)
type reduction = Sum | Mean

(* Splits a hosted tensor's host array into [n_shards] contiguous sub-arrays along [axis], copying
   the data (copy-on-shard; alias views are subtask 293a and out of scope here). Each sub-array is
   wrapped as a fresh batch-major tensor term so that every shard owns distinct tnodes. *)
let shard_along ~axis ~n_shards (t : Tensor.t) : Tensor.t array =
  if n_shards <= 0 then invalid_arg "Parallel.shard_along: n_shards must be > 0";
  if axis <> 0 then
    invalid_arg
      (Printf.sprintf "Parallel.shard_along: only axis=0 (leftmost batch axis) supported, got %d"
         axis);
  let tn = t.Tensor.value in
  let dims = Lazy.force tn.Tn.dims in
  if Array.length dims = 0 then
    invalid_arg "Parallel.shard_along: scalar tensor has no batch axis to shard";
  let batch = dims.(0) in
  if batch % n_shards <> 0 then
    invalid_arg
      (Printf.sprintf "Parallel.shard_along: batch size %d not divisible by n_shards %d" batch
         n_shards);
  let sub = batch / n_shards in
  let prec = Lazy.force tn.Tn.prec in
  let get = Tn.get_value tn in
  let rest = Array.subo dims ~pos:1 in
  Array.init n_shards ~f:(fun k ->
      let sub_dims = Array.append [| sub |] rest in
      let nd =
        Nd.init_array ~debug:(Printf.sprintf "shard%d" k) prec ~dims:sub_dims ~padding:None
          ~f:(fun idx ->
            let src_idx = Array.copy idx in
            src_idx.(0) <- idx.(0) + (k * sub);
            get src_idx)
      in
      TDSL.rebatch ~l:(Printf.sprintf "%s_shard%d" (Tn.debug_name tn) k) nd ())

(* Inverse of {!shard_along}: concatenates the shards' host values along [axis] into a fresh tensor
   (copy-on-gather). Built at the host level; a graph-level concat via [Operation.concat] is the
   eventual zero-copy form once slice-as-view (293a) lands. *)
let gather ~axis (shards : Tensor.t array) : Tensor.t =
  if Array.length shards = 0 then invalid_arg "Parallel.gather: empty shards array";
  if axis <> 0 then
    invalid_arg
      (Printf.sprintf "Parallel.gather: only axis=0 (leftmost batch axis) supported, got %d" axis);
  let n = Array.length shards in
  let tn0 = shards.(0).Tensor.value in
  let dims0 = Lazy.force tn0.Tn.dims in
  let prec = Lazy.force tn0.Tn.prec in
  let rest = Array.subo dims0 ~pos:1 in
  let sub = dims0.(0) in
  Array.iteri shards ~f:(fun i s ->
      let d = Lazy.force s.Tensor.value.Tn.dims in
      if Array.length d = 0 || not (Array.equal Int.equal (Array.subo d ~pos:1) rest) then
        invalid_arg
          (Printf.sprintf "Parallel.gather: shard %d has incompatible shape for axis-0 gather" i));
  let total = sub * n in
  let gets = Array.map shards ~f:(fun s -> Tn.get_value s.Tensor.value) in
  let nd =
    Nd.init_array ~debug:"gather" prec ~dims:(Array.append [| total |] rest) ~padding:None
      ~f:(fun idx ->
        let k = idx.(0) / sub in
        let local = Array.copy idx in
        local.(0) <- idx.(0) % sub;
        gets.(k) local)
  in
  TDSL.rebatch ~l:"gathered" nd ()

(** A handle to a live data-parallel training session. All raw-backend state (the shared backend
    module, the per-shard streams/contexts, the parameter replicas, and the compiled per-shard /
    gradient-sync / optimizer / broadcast routines) is captured by these closures; there is no hidden
    global tensor-to-context lookup. Obtain one via {!data_parallel}. *)
type handle = {
  n_shards : int;
  step : unit -> unit;
      (** Run one synchronized optimizer step: every shard's forward+backward, an all-reduce of the
          parameter gradients across shards via merge-buffer transfer routines, one optimizer update
          on the owner shard, then a broadcast of the updated parameters back to the other shards. *)
  grad_sync : unit -> unit;
      (** All-reduce the parameter gradients across shards onto the owner via merge-buffer transfer
          routines (with the configured {!reduction}). Run after the shards' backward passes and
          before the optimizer step; {!step} already calls it, but it is exposed for custom training
          loops. *)
  set_batch : inputs:Tensor.t -> targets:Tensor.t -> unit;
      (** Scatter a fresh logical batch across the shards (re-shards along the batch axis and copies
          into the per-shard input buffers) for multi-step training. *)
  owner_loss_value : unit -> float;  (** The owner shard's scalar loss after the latest {!step}. *)
  sync_params_to_host : unit -> unit;  (** Copy the owner shard's parameters to their host arrays. *)
  owner_params : Tensor.t array;
      (** The owner shard's parameter tensors (in stable order). Read their host values after
          {!sync_params_to_host}. *)
}

let schedule (r : _ Ir.Backend_intf.routine) = Task.run r.Ir.Backend_intf.schedule

(* [data_parallel] splits one logical batch ([inputs]/[targets]) along the batch axis across
   [n_shards] *fully independent* per-shard backend contexts (one stream/queue/domain per shard, each
   owning disjoint tnode buffers), and drives synchronized data-parallel SGD.

   The model is supplied as [loss_of input target]; it is rebuilt once per shard over that shard's
   slice, so each shard has its own parameter tnodes (single-device unified memory means a single
   shared tnode could not hold distinct per-shard data). The replicas are kept identical: parameters
   are broadcast from the owner after init and after every optimizer step, and the per-shard
   gradients are all-reduced onto the owner between backward and the optimizer step. Gradients and
   parameters move across shards only through per-stream merge buffers
   ([device_to_device ~into_merge_buffer:Copy], internally gated by [wait_for_ready]) -- the one
   cross-stream channel that survived gh-ocannl-341.

   Per-shard RNG streams diverge by default: shard i's graph is built after
   [set_random_seed ~seed:(base_seed + i)], so randomized ops (dropout, [uniform_at]) draw
   differently per shard while the data slices are themselves distinct. *)
let data_parallel ?backend_name ?(reduction = Mean) ?(weight_decay = 0.0) ?(momentum = 0.0)
    ?(base_seed = 0) ~n_shards ~(bindings : Idx.unit_bindings) ~(learning_rate : Tensor.t)
    ~(inputs : Tensor.t) ~(targets : Tensor.t) ~(loss_of : Tensor.t -> Tensor.t -> Tensor.t) ~f () =
  if n_shards <= 0 then invalid_arg "Parallel.data_parallel: n_shards must be > 0";
  let backend = Backends.fresh_backend ?backend_name () in
  let module Backend = (val backend : Ir.Backend_intf.Backend) in
  let num_devices = Backend.num_devices () in
  let xs = shard_along ~axis:0 ~n_shards inputs in
  let ys = shard_along ~axis:0 ~n_shards targets in
  Array.iter xs ~f:(fun x -> Train.set_hosted x.Tensor.value);
  Array.iter ys ~f:(fun y -> Train.set_hosted y.Tensor.value);
  Train.set_hosted learning_rate.Tensor.value;
  (* Per-shard loss graphs: distinct parameter/input tnodes, distinct RNG seed. *)
  let losses =
    Array.init n_shards ~f:(fun i ->
        Tensor.set_random_seed ~seed:(base_seed + i) ();
        loss_of xs.(i) ys.(i))
  in
  Tensor.set_random_seed ~seed:base_seed ();
  (* Parameters of each shard, in a stable order (Set order = ascending tnode id = creation order),
     so [params.(i).(k)] is shard i's replica of the k-th parameter. *)
  let params = Array.map losses ~f:(fun l -> Array.of_list (Set.to_list l.Tensor.params)) in
  let n_params = Array.length params.(0) in
  Array.iter params ~f:(fun ps ->
      if Array.length ps <> n_params then
        invalid_arg "Parallel.data_parallel: shards disagree on the parameter count");
  Array.iter params ~f:(Array.iter ~f:(fun p -> Train.set_hosted p.Tensor.value));
  (* Build the forward+backward comps before the parameter-init comps, matching {!Train.run_once}:
     [grad_update] consumes the forward/backprop roots, and [init_params] must run against the state
     it leaves. *)
  let updates = Array.map losses ~f:(Train.grad_update ~setup_for_parallel:true) in
  let init_comps = Array.map losses ~f:Tensor.init_params in
  let streams =
    Array.init n_shards ~f:(fun i ->
        Backend.new_stream (Backend.get_device ~ordinal:(i % num_devices)))
  in
  let owner_stream = streams.(0) in
  (* Per shard: a fresh context, parameters initialized and the input slice staged, then the linked
     forward+backward routine. *)
  let grad_routines =
    Array.init n_shards ~f:(fun i ->
        let opt = Backend.empty_optimize_ctx () in
        let ctx = Backend.make_context ~optimize_ctx:opt streams.(i) in
        let init_routine = Backend.link ctx (Backend.compile opt bindings init_comps.(i)) in
        schedule init_routine;
        Backend.await streams.(i);
        let ctx = init_routine.Ir.Backend_intf.context in
        let ctx = Backend.init_from_host ctx xs.(i).Tensor.value in
        let ctx = Backend.init_from_host ctx ys.(i).Tensor.value in
        Backend.await streams.(i);
        Backend.link ctx (Backend.compile ctx.optimize_ctx bindings updates.(i)))
  in
  let shard_ctx = Array.map grad_routines ~f:(fun r -> r.Ir.Backend_intf.context) in
  let owner_ctx = shard_ctx.(0) in
  (* A merge-buffer transfer of [tn] from shard [i] into [dst], followed by running [consumer] (which
     reads [tn]'s merge buffer) on [dst]'s stream. Returns whether anything was transferred. *)
  let merge_transfer ~dst ~src tn consumer_code =
    match Backend.device_to_device tn ~into_merge_buffer:Copy ~dst ~src with
    | None -> ()
    | Some tr ->
        schedule tr;
        schedule (Backend.link tr.Ir.Backend_intf.context consumer_code)
  in
  (* Broadcast: copy the owner's k-th parameter value into every other shard's replica via the merge
     buffer. [bcast_codes.(i).(k)] reads the merge buffer and assigns shard i's replica. *)
  let bcast_codes =
    Array.init n_shards ~f:(fun i ->
        Array.init n_params ~f:(fun k ->
            let dst_p = params.(i).(k) and owner_p = params.(0).(k) in
            let code = [%cd dst_p =: owner_p.merge] in
            Backend.compile shard_ctx.(i).optimize_ctx
              ~name:(Printf.sprintf "param_bcast_%d_%d" i k) bindings code))
  in
  let broadcast_params () =
    for k = 0 to n_params - 1 do
      for i = 1 to n_shards - 1 do
        merge_transfer ~dst:shard_ctx.(i) ~src:owner_ctx params.(0).(k).Tensor.value
          bcast_codes.(i).(k)
      done
    done;
    Array.iter streams ~f:Backend.await
  in
  (* Replicate the owner's freshly-initialized parameters to the other shards. *)
  broadcast_params ();
  (* Gradient all-reduce: for each parameter, accumulate every other shard's gradient into the
     owner's gradient through the merge buffer. [accum_codes.(i).(k)] computes
     [owner_grad += src_shard_grad.merge]. *)
  let accum_codes =
    Array.init n_shards ~f:(fun i ->
        Array.init n_params ~f:(fun k ->
            let owner_p = params.(0).(k) and src_p = params.(i).(k) in
            let code = [%cd owner_p.grad =+ src_p.grad.merge] in
            Backend.compile owner_ctx.optimize_ctx
              ~name:(Printf.sprintf "grad_allreduce_%d_%d" i k) bindings code))
  in
  let mean_codes =
    match reduction with
    | Sum -> None
    | Mean ->
        let inv_n = 1.0 /. Float.of_int n_shards in
        Some
          (Array.init n_params ~f:(fun k ->
               let owner_p = params.(0).(k) in
               let code = [%cd owner_p.grad =* !.inv_n] in
               Backend.compile owner_ctx.optimize_ctx ~name:(Printf.sprintf "grad_mean_%d" k)
                 bindings code))
  in
  let grad_of p = (Option.value_exn ~here:[%here] p.Tensor.diff).Tensor.grad in
  let grad_sync () =
    for k = 0 to n_params - 1 do
      for i = 1 to n_shards - 1 do
        merge_transfer ~dst:owner_ctx ~src:shard_ctx.(i) (grad_of params.(i).(k))
          accum_codes.(i).(k)
      done
    done;
    Backend.await owner_stream;
    Option.iter mean_codes ~f:(fun codes ->
        Array.iter codes ~f:(fun code -> schedule (Backend.link owner_ctx code));
        Backend.await owner_stream)
  in
  (* The optimizer step runs only on the owner's synchronized gradients; updated parameters are then
     broadcast back to the other shards. *)
  let owner_loss = losses.(0) in
  let sgd_comp =
    Train.sgd_update ~learning_rate ~weight_decay
      ?momentum:(if Float.(momentum > 0.0) then Some momentum else None)
      owner_loss
  in
  let owner_ctx = Backend.init_from_host owner_ctx learning_rate.Tensor.value in
  let sgd_routine =
    Backend.link owner_ctx (Backend.compile owner_ctx.optimize_ctx bindings sgd_comp)
  in
  let step () =
    ignore (Backend.from_host owner_ctx learning_rate.Tensor.value : bool);
    Array.iter grad_routines ~f:schedule;
    Array.iter streams ~f:Backend.await;
    grad_sync ();
    schedule sgd_routine;
    Backend.await owner_stream;
    broadcast_params ()
  in
  let set_batch ~inputs ~targets =
    let nxs = shard_along ~axis:0 ~n_shards inputs in
    let nys = shard_along ~axis:0 ~n_shards targets in
    for i = 0 to n_shards - 1 do
      Tn.set_values xs.(i).Tensor.value (Tn.get_values nxs.(i).Tensor.value);
      Tn.set_values ys.(i).Tensor.value (Tn.get_values nys.(i).Tensor.value);
      ignore (Backend.from_host shard_ctx.(i) xs.(i).Tensor.value : bool);
      ignore (Backend.from_host shard_ctx.(i) ys.(i).Tensor.value : bool);
      Backend.await streams.(i)
    done
  in
  let owner_loss_value () =
    ignore (Backend.to_host owner_ctx owner_loss.Tensor.value : bool);
    Backend.await owner_stream;
    Tn.get_value owner_loss.Tensor.value [| 0 |]
  in
  let sync_params_to_host () =
    Array.iter params.(0) ~f:(fun p ->
        ignore (Backend.to_host owner_ctx p.Tensor.value : bool));
    Backend.await owner_stream
  in
  f
    {
      n_shards;
      step;
      grad_sync;
      set_batch;
      owner_loss_value;
      sync_params_to_host;
      owner_params = params.(0);
    }
