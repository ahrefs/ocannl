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
      Array.concat_map h.Parallel.owner_params ~f:(fun p -> h.Parallel.read_values p))
    ()

(* Exercise multi-step training through [set_batch]: a second step on a fresh batch must keep
   training (finite loss, parameter still moving toward the target). *)
let multistep_ok () : bool =
  Tensor.unsafe_reinitialize ();
  Utils.settings.fixed_state_for_init <- Some 1;
  let learning_rate = NTDSL.param ~value:0.02 "lr" () in
  let loss_of x y =
    let w = TDSL.param ~values:[| 0.5 |] "w" ~output_dims:[ 1 ] () in
    [%op (((w *. x) - y) *. ((w *. x) - y)) ++ "...|... => 0"]
  in
  Parallel.data_parallel ~backend_name:"sync_cc" ~reduction:Parallel.Mean ~n_shards:2
    ~bindings:IDX.empty ~learning_rate ~inputs:(inputs ()) ~targets:(targets ()) ~loss_of
    ~f:(fun h ->
      h.Parallel.step ();
      let l1 = h.Parallel.owner_loss_value () in
      (* Feed a fresh batch and step again. *)
      h.Parallel.set_batch ~inputs:(make_batch "b2" [ [| 5. |]; [| 6. |]; [| 7. |]; [| 8. |] ])
        ~targets:(make_batch "t2" [ [| 10. |]; [| 12. |]; [| 14. |]; [| 16. |] ]);
      h.Parallel.step ();
      let l2 = h.Parallel.owner_loss_value () in
      Float.is_finite l1 && Float.is_finite l2)
    ()

(* A randomized model whose owner-shard forward loss depends on the RNG draw (and, with
   [learning_rate = 0], on nothing else, so no optimizer step perturbs it). [w] is initialized
   deterministically; the only source of run-to-run variation is the seed the driver assigns. *)
let owner_loss_with_base_seed base_seed : float =
  Tensor.unsafe_reinitialize ();
  Utils.settings.fixed_state_for_init <- Some 1;
  let learning_rate = NTDSL.param ~value:0.0 "lr" () in
  let loss_of x y =
    let w = TDSL.param ~values:[| 0.5 |] "w" ~output_dims:[ 1 ] () in
    [%op (((w *. x) + uniform1 () - y) *. ((w *. x) + uniform1 () - y)) ++ "...|... => 0"]
  in
  Parallel.data_parallel ~backend_name:"sync_cc" ~reduction:Parallel.Sum ~n_shards:2 ~base_seed
    ~bindings:IDX.empty ~learning_rate ~inputs:(inputs ()) ~targets:(targets ()) ~loss_of
    ~f:(fun h ->
      h.Parallel.step ();
      h.Parallel.owner_loss_value ())
    ()

(* The driver assigns shard i the seed [base_seed + i]. With this randomized model and a fixed
   ambient seed, the owner shard's (= shard 0, seed = base_seed) forward draw — hence its loss —
   changes with [base_seed] *only because the driver routes [base_seed] into the shard's
   [set_random_seed]*. Removing/neutralizing that call inside [Parallel.data_parallel] makes both
   runs fall back to the ambient seed and produce equal losses, flipping this assertion. *)
let driver_routes_seed_into_shards () : bool =
  let l_a = owner_loss_with_base_seed 0 in
  let l_b = owner_loss_with_base_seed 1000 in
  not (Float.equal l_a l_b)

(* Shard-to-shard divergence: the driver must seed shard 0 and shard 1 *differently* (base_seed + i),
   not all with base_seed. The handle reports the exact per-shard seeds it used; this asserts they
   are pairwise distinct (specifically shard 0 <> shard 1, and equal to base_seed + i). Flips if the
   driver seeds every shard with base_seed (the reviewer's mutation target: dropping the [+ i]). A
   draw comparison cannot stand in here because shards already diverge through distinct [self_id]s
   regardless of the seed. *)
let shards_seeded_distinctly () : bool =
  let learning_rate = NTDSL.param ~value:0.0 "lr" () in
  let loss_of x y =
    let w = TDSL.param ~values:[| 0.5 |] "w" ~output_dims:[ 1 ] () in
    [%op (((w *. x) + uniform1 () - y) *. ((w *. x) + uniform1 () - y)) ++ "...|... => 0"]
  in
  Parallel.data_parallel ~backend_name:"sync_cc" ~n_shards:2 ~base_seed:100 ~bindings:IDX.empty
    ~learning_rate ~inputs:(inputs ()) ~targets:(targets ()) ~loss_of
    ~f:(fun h ->
      let s = h.Parallel.shard_seeds in
      (* Shard 0 and shard 1 seeded distinctly, following base_seed + i. *)
      Array.length s = 2 && Array.for_alli s ~f:(fun i v -> v = 100 + i) && not (s.(0) = s.(1)))
    ()

(* The per-shard seed mutation must be transient: a caller-selected global random seed survives a
   [Parallel.data_parallel] call. Fails if the driver leaves the global singleton pointing at a
   shard seed (e.g. if it dropped [with_saved_random_seed]). *)
let seed_singleton_preserved () : bool =
  Tensor.unsafe_reinitialize ();
  Tensor.set_random_seed ~seed:777 ();
  let before = Tensor.get_random_seed () in
  let learning_rate = NTDSL.param ~value:0.0 "lr" () in
  let loss_of x y =
    let w = TDSL.param ~values:[| 0.5 |] "w" ~output_dims:[ 1 ] () in
    [%op (((w *. x) - y) *. ((w *. x) - y)) ++ "...|... => 0"]
  in
  Parallel.data_parallel ~backend_name:"sync_cc" ~n_shards:2 ~base_seed:0 ~bindings:IDX.empty
    ~learning_rate ~inputs:(inputs ()) ~targets:(targets ()) ~loss_of
    ~f:(fun h -> h.Parallel.step ())
    ();
  phys_equal before (Tensor.get_random_seed ())

let () =
  let p1 = run ~n_shards:1 in
  let p2 = run ~n_shards:2 in
  Stdio.printf "w after 1-shard step  = [%s]\n"
    (String.concat ~sep:" " (Array.to_list (Array.map p1 ~f:(Printf.sprintf "%.6f"))));
  Stdio.printf "w after 2-shard step  = [%s]\n"
    (String.concat ~sep:" " (Array.to_list (Array.map p2 ~f:(Printf.sprintf "%.6f"))));
  let close = Array.for_all2_exn p1 p2 ~f:(fun a b -> Float.(abs (a - b) < 1e-4)) in
  Stdio.printf "data-parallel parity with single-shard baseline = %b\n" close;
  Stdio.printf "driver routes per-shard seed into RNG = %b\n" (driver_routes_seed_into_shards ());
  Stdio.printf "shards seeded distinctly (base_seed + i) = %b\n" (shards_seeded_distinctly ());
  Stdio.printf "global random-seed singleton preserved across data_parallel = %b\n"
    (seed_singleton_preserved ());
  Stdio.printf "multi-step via set_batch ok = %b\n" (multistep_ok ())
