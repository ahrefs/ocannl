(** Data-parallel training over independent per-shard backend contexts, and the host-level sharding
    helpers it is built from. *)

open Ocannl_tensor

(** Gradient reduction mode used by {!data_parallel} when all-reducing parameter gradients across
    data-parallel shards. [Sum] adds the per-shard gradients; [Mean] additionally divides by the
    shard count. *)
type reduction = Sum | Mean

val host_values : Tensor.t -> float array
(** Reads all (unpadded) values of an ndarray-backed (literal) tensor from its host initialization
    buffer. Use for inspecting host-level shard/gather results. *)

val shard_along : axis:int -> n_shards:int -> Tensor.t -> Tensor.t array
(** Splits a literal tensor's host buffer into [n_shards] contiguous sub-arrays along [axis],
    copying the data (copy-on-shard; alias views are subtask 293a and out of scope here). Each
    sub-array is wrapped as a fresh batch-major tensor term so that every shard owns distinct
    tnodes. *)

val gather : axis:int -> Tensor.t array -> Tensor.t
(** Inverse of {!shard_along}: concatenates the shards' host values along [axis] into a fresh tensor
    (copy-on-gather). Built at the host level; a graph-level concat via [Operation.concat] is the
    eventual zero-copy form once slice-as-view (293a) lands. *)

type handle = {
  n_shards : int;
  step : unit -> unit;
      (** Run one synchronized optimizer step: every shard's forward+backward, an all-reduce of the
          parameter gradients across shards via merge-buffer transfer routines, one optimizer update
          on the owner shard, then a broadcast of the updated parameters back to the other shards.
      *)
  grad_sync : unit -> unit;
      (** All-reduce the parameter gradients across shards onto the owner via merge-buffer transfer
          routines (with the configured {!reduction}). Run after the shards' backward passes and
          before the optimizer step; {!step} already calls it, but it is exposed for custom training
          loops. *)
  set_batch : inputs:Tensor.t -> targets:Tensor.t -> unit;
      (** Scatter a fresh logical batch across the shards (re-shards along the batch axis and copies
          into the per-shard input buffers) for multi-step training. *)
  owner_loss_value : unit -> float;  (** The owner shard's scalar loss after the latest {!step}. *)
  sync_params_to_host : unit -> unit;
      (** Awaits the owner stream. Retained for API compatibility; after gh-ocannl-333 host values
          are read on demand via {!read_values} rather than copied to a host array. *)
  read_values : Tensor.t -> float array;
      (** Reads a tensor's current values from the owner shard's context via an on-demand
          device-to-host transfer. Use for the owner's parameters / loss. *)
  owner_params : Tensor.t array;
      (** The owner shard's parameter tensors (in stable order). Read their values via
          {!read_values}. *)
  shard_seeds : int array;
      (** The RNG seed assigned to each shard ([shard_seeds.(i) = base_seed + i]). These are the
          exact values passed to [set_random_seed] when building the shards, so distinct entries
          witness that the per-shard RNG seeding diverges. *)
}
(** A handle to a live data-parallel training session. All raw-backend state (the shared backend
    module, the per-shard streams/contexts, the parameter replicas, and the compiled per-shard /
    gradient-sync / optimizer / broadcast routines) is captured by these closures; there is no
    hidden global tensor-to-context lookup. Obtain one via {!data_parallel}. *)

val data_parallel :
  ?backend_name:string ->
  ?reduction:reduction ->
  ?weight_decay:float ->
  ?momentum:float ->
  ?base_seed:int ->
  n_shards:int ->
  bindings:Ir.Indexing.unit_bindings ->
  learning_rate:Tensor.t ->
  inputs:Tensor.t ->
  targets:Tensor.t ->
  loss_of:(Tensor.t -> Tensor.t -> Tensor.t) ->
  f:(handle -> 'a) ->
  unit ->
  'a
(** [data_parallel] splits one logical batch ([inputs]/[targets]) along the batch axis across
    [n_shards] *fully independent* per-shard backend contexts (one stream/queue/domain per shard,
    each owning disjoint tnode buffers), and drives synchronized data-parallel SGD.

    The model is supplied as [loss_of input target]; it is rebuilt once per shard over that shard's
    slice, so each shard has its own parameter tnodes (single-device unified memory means a single
    shared tnode could not hold distinct per-shard data). The replicas are kept identical:
    parameters are broadcast from the owner after init and after every optimizer step, and the
    per-shard gradients are all-reduced onto the owner between backward and the optimizer step.
    Gradients and parameters move across shards only through per-stream merge buffers
    ([device_to_device ~into_merge_buffer:Copy], internally gated by [wait_for_ready]) -- the one
    cross-stream channel that survived gh-ocannl-341.

    Per-shard RNG streams diverge by default: shard i's graph is built after
    [set_random_seed ~seed:(base_seed + i)], so randomized ops (dropout, [uniform_at]) draw
    differently per shard while the data slices are themselves distinct. The seed mutation is scoped
    by [Tensor.with_saved_random_seed], so the caller's global random seed is restored on return.

    No routine is handed back, so every launch parameter of [bindings] runs at 0 (gh-ocannl-940). *)
