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

(** Gradient reduction mode used by {!grad_sync} when all-reducing parameter gradients across
    data-parallel shards. *)
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

let _ = (Sum, Mean)
let _ = Asgns.sequence
let _ = Idx.Empty
let _ = Task.run
let _ = Backends.fresh_backend
