open Base
module Tn = Ir.Tnode
module Nd = Ir.Ndarray
module Asgns = Ir.Assignments
module Idx = Ir.Indexing
module Task = Ir.Task
module Backends = Context.Backends
open Ocannl_tensor.Operation.DSL_modules

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

type reduction = Sum | Mean

(* Host-side accessor for ndarray-backed (literal) tensors. After gh-ocannl-333 tensor data lives
   only on devices; the host-level sharding helpers below therefore operate on literal tensors whose
   initialization buffer is registered in [Host_inits]. *)
let host_get (tn : Tn.t) =
  match Ir.Host_inits.find tn with
  | Some nd ->
      let nd = Lazy.force nd in
      let padding = Option.map ~f:fst (Lazy.force tn.Tn.padding) in
      fun idx -> Nd.get_as_float ?padding nd idx
  | None ->
      invalid_arg
        (Printf.sprintf
           "Parallel: tensor %s has no host-side data; only ndarray-backed (literal) tensors can \
            be sharded/gathered at the host level"
           (Tn.debug_name tn))

let host_values (t : Tensor.t) : float array =
  let tn = t.Tensor.value in
  match Ir.Host_inits.find tn with
  | Some nd ->
      let nd = Lazy.force nd in
      let padding = Option.map ~f:fst (Lazy.force tn.Tn.padding) in
      Nd.retrieve_flat_values ?padding nd
  | None ->
      invalid_arg
        (Printf.sprintf "Parallel.host_values: tensor %s has no host-side data" (Tn.debug_name tn))

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
  let prec = Lazy.force tn.Tn.storage_prec in
  let get = host_get tn in
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

let gather ~axis (shards : Tensor.t array) : Tensor.t =
  if Array.length shards = 0 then invalid_arg "Parallel.gather: empty shards array";
  if axis <> 0 then
    invalid_arg
      (Printf.sprintf "Parallel.gather: only axis=0 (leftmost batch axis) supported, got %d" axis);
  let n = Array.length shards in
  let tn0 = shards.(0).Tensor.value in
  let dims0 = Lazy.force tn0.Tn.dims in
  let prec = Lazy.force tn0.Tn.storage_prec in
  let rest = Array.subo dims0 ~pos:1 in
  let sub = dims0.(0) in
  Array.iteri shards ~f:(fun i s ->
      let d = Lazy.force s.Tensor.value.Tn.dims in
      if Array.length d = 0 || not (Array.equal Int.equal (Array.subo d ~pos:1) rest) then
        invalid_arg
          (Printf.sprintf "Parallel.gather: shard %d has incompatible shape for axis-0 gather" i));
  let total = sub * n in
  let gets = Array.map shards ~f:(fun s -> host_get s.Tensor.value) in
  let nd =
    Nd.init_array ~debug:"gather" prec ~dims:(Array.append [| total |] rest) ~padding:None
      ~f:(fun idx ->
        let k = idx.(0) / sub in
        let local = Array.copy idx in
        local.(0) <- idx.(0) % sub;
        gets.(k) local)
  in
  TDSL.rebatch ~l:"gathered" nd ()

type handle = {
  n_shards : int;
  step : unit -> unit;
  grad_sync : unit -> unit;
  set_batch : inputs:Tensor.t -> targets:Tensor.t -> unit;
  owner_loss_value : unit -> float;
  sync_params_to_host : unit -> unit;
  read_values : Tensor.t -> float array;
  owner_params : Tensor.t array;
  shard_seeds : int array;
  set_launch_value : Idx.static_symbol -> int -> unit;
}

let data_parallel ?backend_name ?(reduction = Mean) ?(weight_decay = 0.0) ?(momentum = 0.0)
    ?(base_seed = 0) ?(launch_values = []) ~n_shards ~(bindings : Idx.unit_bindings)
    ~(learning_rate : Tensor.t) ~(inputs : Tensor.t) ~(targets : Tensor.t)
    ~(loss_of : Tensor.t -> Tensor.t -> Tensor.t) ~f () =
  if n_shards <= 0 then invalid_arg "Parallel.data_parallel: n_shards must be > 0";
  let backend = Backends.backend_module (Backends.get_backend ?backend_name ()) in
  let module Backend = (val backend : Ir.Backend_intf.Backend) in
  let num_devices = Backend.num_devices () in
  let xs = shard_along ~axis:0 ~n_shards inputs in
  let ys = shard_along ~axis:0 ~n_shards targets in
  Array.iter xs ~f:(fun x -> Train.set_materialized x.Tensor.value);
  Array.iter ys ~f:(fun y -> Train.set_materialized y.Tensor.value);
  Train.set_materialized learning_rate.Tensor.value;
  (* Distinct RNG seed per shard so randomized ops diverge across shards: shard i uses
     [shard_seeds.(i) = base_seed + i]. The same value is passed to [set_random_seed] and recorded
     in [shard_seeds], so the recorded seeds are exactly the ones the shards were built with. The
     mutation is scoped: [with_saved_random_seed] restores the caller's global random-seed singleton
     afterwards, so a caller-selected seed is not perturbed; each built loss keeps referencing the
     seed tensor that was current when it was constructed. *)
  let shard_seeds = Array.init n_shards ~f:(fun i -> base_seed + i) in
  (* Per-shard loss graphs: distinct parameter/input tnodes. *)
  let losses =
    Tensor.with_saved_random_seed (fun () ->
        Array.mapi shard_seeds ~f:(fun i seed ->
            Tensor.set_random_seed ~seed ();
            loss_of xs.(i) ys.(i)))
  in
  (* [data_parallel] hands no routine back, so its routines' launch parameters live here, in one
     cell per bound symbol: the caller's [launch_values], an unranged parameter left out at 0, and a
     ranged one left out refused (gh-ocannl-1629) -- once the shards' graphs exist, so that a
     batch-slice role is known, and before anything is compiled. [schedule] copies them into each
     routine it runs, so [set_launch_value] reaches every routine from the next one scheduled. *)
  let width64 = Utils.settings.large_models in
  let launch =
    Idx.resolve_launch_values ~caller:"Parallel.data_parallel"
      ~hint:"pass each one's value through ~launch_values; the handle's set_launch_value changes it"
      ~ranged:Train.ranged_launch_param launch_values (Idx.bound_symbols bindings)
  in
  let schedule (r : _ Ir.Backend_intf.routine) =
    Idx.copy_launch_values ~from:launch r.Ir.Backend_intf.bindings;
    Task.run r.Ir.Backend_intf.schedule
  in
  let set_launch_value =
    Idx.set_launch_value ~width64 ~caller:"Parallel.data_parallel: set_launch_value" launch
  in
  (* Parameters of each shard, in a stable order (Set order = ascending tnode id = creation order),
     so [params.(i).(k)] is shard i's replica of the k-th parameter. *)
  let params = Array.map losses ~f:(fun l -> Array.of_list (Set.to_list l.Tensor.params)) in
  let n_params = Array.length params.(0) in
  Array.iter params ~f:(fun ps ->
      if Array.length ps <> n_params then
        invalid_arg "Parallel.data_parallel: shards disagree on the parameter count");
  Array.iter params ~f:(Array.iter ~f:(fun p -> Train.set_materialized p.Tensor.value));
  (* The state set also owns non-differentiable buffers (e.g. BatchNorm running moments) and frozen
     parameters. Only the subset written by backprop participates in gradient reduction. *)
  let trainable = Array.map losses ~f:Train.trainable_params in
  let grad_masks = Array.mapi params ~f:(fun i ps -> Array.map ps ~f:(Set.mem trainable.(i))) in
  Array.iter grad_masks ~f:(fun mask ->
      if not (Array.equal Bool.equal mask grad_masks.(0)) then
        invalid_arg "Parallel.data_parallel: shards disagree on the trainable parameter set");
  let grad_params =
    Array.mapi params ~f:(fun i ps -> Array.filteri ps ~f:(fun k _ -> grad_masks.(i).(k)))
  in
  let n_grad_params = Array.length grad_params.(0) in
  (* Build the forward+backward comps before the parameter-init comps, matching {!Train.run_once}:
     [grad_update] consumes the forward/backprop roots, and [init_params] must run against the state
     it leaves. *)
  let updates = Array.map losses ~f:(Train.grad_update ~setup_for_parallel:true) in
  let init_comps = Array.map losses ~f:Tensor.init_params in
  let streams = Array.init n_shards ~f:(fun i -> Backend.get_device ~ordinal:(i % num_devices)) in
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
        (* Stage the shard's input/target slices: upload the literal shards' host data into the
           freshly-allocated device buffers (gh-ocannl-333). *)
        let init_literal ctx (t : Tensor.t) =
          let nd =
            Lazy.force (Option.value_exn ~here:[%here] (Ir.Host_inits.find t.Tensor.value))
          in
          Backend.init_from_host ctx t.Tensor.value nd
        in
        let ctx = init_literal ctx xs.(i) in
        let ctx = init_literal ctx ys.(i) in
        Backend.await streams.(i);
        Backend.link ctx (Backend.compile ctx.optimize_ctx bindings updates.(i)))
  in
  let shard_ctx = Array.map grad_routines ~f:(fun r -> r.Ir.Backend_intf.context) in
  let owner_ctx = shard_ctx.(0) in
  (* A merge-buffer transfer of [tn] from shard [i] into [dst], followed by running [consumer]
     (which reads [tn]'s merge buffer) on [dst]'s stream. Returns whether anything was
     transferred. *)
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
              ~name:(Printf.sprintf "param_bcast_%d_%d" i k)
              bindings code))
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
     owner's gradient through the merge buffer. [accum_codes.(i).(k)] computes [owner_grad +=
     src_shard_grad.merge]. *)
  let accum_codes =
    Array.init n_shards ~f:(fun i ->
        Array.init n_grad_params ~f:(fun k ->
            let owner_p = grad_params.(0).(k) and src_p = grad_params.(i).(k) in
            let code = [%cd owner_p.grad =+ src_p.grad.merge] in
            Backend.compile owner_ctx.optimize_ctx
              ~name:(Printf.sprintf "grad_allreduce_%d_%d" i k)
              bindings code))
  in
  let mean_codes =
    match reduction with
    | Sum -> None
    | Mean ->
        let inv_n = 1.0 /. Float.of_int n_shards in
        Some
          (Array.init n_grad_params ~f:(fun k ->
               let owner_p = grad_params.(0).(k) in
               let code = [%cd owner_p.grad =* !.inv_n] in
               Backend.compile owner_ctx.optimize_ctx ~name:(Printf.sprintf "grad_mean_%d" k)
                 bindings code))
  in
  let grad_of p = (Option.value_exn ~here:[%here] p.Tensor.diff).Tensor.grad in
  let grad_sync () =
    for k = 0 to n_grad_params - 1 do
      for i = 1 to n_shards - 1 do
        merge_transfer ~dst:owner_ctx ~src:shard_ctx.(i)
          (grad_of grad_params.(i).(k))
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
  (* After gh-ocannl-333 there is no host array on a tensor node; host access goes through explicit
     device transfers with a caller-supplied [Ndarray]. These helpers wrap that for the raw
     per-shard backend contexts. *)
  let host_buffer_of (tn : Tn.t) =
    Nd.create_array
      ~debug:("parallel host buffer for " ^ Tn.debug_name tn)
      (Lazy.force tn.Tn.storage_prec) ~dims:(Lazy.force tn.Tn.dims)
      ~padding:(Lazy.force tn.Tn.padding)
  in
  let read_ctx_values ~ctx ~stream (tn : Tn.t) =
    let nd = host_buffer_of tn in
    ignore (Backend.to_host ctx tn nd : bool);
    Backend.await stream;
    let padding = Option.map ~f:fst (Lazy.force tn.Tn.padding) in
    Nd.retrieve_flat_values ?padding nd
  in
  (* Upload a literal (ndarray-backed) tensor's host data into an existing device buffer. *)
  let upload_literal ~ctx ~stream ~(dst : Tn.t) (src : Tensor.t) =
    let src_nd =
      Lazy.force (Option.value_exn ~here:[%here] (Ir.Host_inits.find src.Tensor.value))
    in
    (* Interval analysis, Phase B: every host-write path proposes/validates bounds. *)
    Tn.propose_bounds_from_host dst src_nd;
    ignore (Backend.from_host ctx dst src_nd : bool);
    Backend.await stream
  in
  (* The learning rate is a small constant; initialize it on the owner via its own init code (its
     [Constant] fetch runs on-device), rather than uploading a host array. *)
  let owner_ctx =
    let init_comp = Ocannl_tensor.Tensor.init_params learning_rate in
    let r = Backend.link owner_ctx (Backend.compile owner_ctx.optimize_ctx bindings init_comp) in
    schedule r;
    Backend.await owner_stream;
    r.Ir.Backend_intf.context
  in
  let sgd_routine =
    Backend.link owner_ctx (Backend.compile owner_ctx.optimize_ctx bindings sgd_comp)
  in
  let step () =
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
      upload_literal ~ctx:shard_ctx.(i) ~stream:streams.(i) ~dst:xs.(i).Tensor.value nxs.(i);
      upload_literal ~ctx:shard_ctx.(i) ~stream:streams.(i) ~dst:ys.(i).Tensor.value nys.(i)
    done
  in
  let owner_loss_value () =
    (read_ctx_values ~ctx:owner_ctx ~stream:owner_stream owner_loss.Tensor.value).(0)
  in
  let read_values (t : Tensor.t) =
    read_ctx_values ~ctx:owner_ctx ~stream:owner_stream t.Tensor.value
  in
  (* Retained for API compatibility; reading happens on demand via {!read_values}. *)
  let sync_params_to_host () = Backend.await owner_stream in
  f
    {
      n_shards;
      step;
      grad_sync;
      set_batch;
      owner_loss_value;
      sync_params_to_host;
      read_values;
      owner_params = params.(0);
      shard_seeds;
      set_launch_value;
    }
