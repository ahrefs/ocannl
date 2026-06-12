open Base
open Ocannl
open Operation.DSL_modules
module Tn = Ir.Tnode
module IDX = Train.IDX

(* Regression test for the data-parallel training driver (task-2445dd1c, subtask 293c).

   A small linear-regression step is run two ways and the resulting parameters are compared:
   - n_shards = 1: the whole logical batch on a single shard (the single-shard baseline);
   - n_shards = 2: the same logical batch split along the batch axis across two shards, with the
     per-shard gradients all-reduced via merge-buffer transfer routines before one optimizer step.

   With a sum-over-batch loss and Sum reduction, the all-reduced gradient over the two half-batches
   equals the full-batch gradient exactly, so the two runs must land on identical parameters. The
   test would fail if a shard ran the full batch (the split would be wrong) or if the reduction
   dropped/double-counted a shard. Parameters (not just the loss) are compared. *)

let make_batch label rows =
  (* [rows] is a list of rows, each a float array (the output axis); batch axis is leftmost. *)
  let open Bigarray in
  let n = List.length rows in
  let width = Array.length (List.hd_exn rows) in
  let ga = Genarray.create Float32 c_layout [| n; width |] in
  List.iteri rows ~f:(fun i row -> Array.iteri row ~f:(fun j v -> Genarray.set ga [| i; j |] v));
  let nd = Ir.Ndarray.as_array Ir.Ops.Single ga in
  TDSL.rebatch ~l:label nd ()

(* One logical batch of four examples of the line y = 2x. *)
let inputs () = make_batch "inputs" [ [| 1. |]; [| 2. |]; [| 3. |]; [| 4. |] ]
let targets () = make_batch "targets" [ [| 2. |]; [| 4. |]; [| 6. |]; [| 8. |] ]

let run ~n_shards : float array =
  Tensor.unsafe_reinitialize ();
  Utils.settings.fixed_state_for_init <- Some 1;
  (* Deterministic, id-independent parameter init so the two runs start identically regardless of
     how sharding changes tnode creation order. *)
  let learning_rate = NTDSL.param ~value:0.05 "lr" () in
  Tn.set_values learning_rate.Tensor.value [| 0.05 |];
  (* Sum-over-batch squared error. [loss_of] creates its own parameter per call, so each shard gets a
     distinct (but identically-initialized) replica, as the driver requires. *)
  let loss_of x y =
    let w = TDSL.param ~values:[| 0.5 |] "w" ~output_dims:[ 1 ] () in
    [%op (((w *. x) - y) *. ((w *. x) - y)) ++ "...|... => 0"]
  in
  Parallel.data_parallel ~backend_name:"sync_cc" ~reduction:Parallel.Sum ~n_shards
    ~bindings:IDX.empty ~learning_rate ~inputs:(inputs ()) ~targets:(targets ()) ~loss_of
    ~f:(fun h ->
      h.Parallel.step ();
      Stdio.printf "n_shards=%d: loss=%.4f\n" n_shards (h.Parallel.owner_loss_value ());
      h.Parallel.sync_params_to_host ();
      Array.concat_map h.Parallel.owner_params ~f:(fun p -> Tn.get_values p.Tensor.value))
    ()

let () =
  let p1 = run ~n_shards:1 in
  let p2 = run ~n_shards:2 in
  Stdio.printf "w after 1-shard step  = [%s]\n"
    (String.concat ~sep:" " (Array.to_list (Array.map p1 ~f:(Printf.sprintf "%.6f"))));
  Stdio.printf "w after 2-shard step  = [%s]\n"
    (String.concat ~sep:" " (Array.to_list (Array.map p2 ~f:(Printf.sprintf "%.6f"))));
  let close = Array.for_all2_exn p1 p2 ~f:(fun a b -> Float.(abs (a - b) < 1e-4)) in
  Stdio.printf "data-parallel parity with single-shard baseline = %b\n" close
