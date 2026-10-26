open Base
open Ocannl
open Operation.DSL_modules
module Tn = Ir.Tnode
module IDX = Train.IDX

(* Regression test for the data-parallel training driver (task-2445dd1c, subtask 293c).

   A small linear-regression step is run two ways and the resulting parameters are compared: -
   n_shards = 1: the whole logical batch on a single shard (the single-shard baseline); - n_shards =
   2: the same logical batch split along the batch axis across two shards, with the per-shard
   gradients all-reduced via merge-buffer transfer routines before one optimizer step.

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

let run ?(momentum = 0.0) ?(weight_decay = 0.0) ?(steps = 1) ?(print_loss = true) ~n_shards () :
    float array =
  Tensor.unsafe_reinitialize ();
  Utils.settings.fixed_state_for_init <- Some 1;
  (* Deterministic, id-independent parameter init so the two runs start identically regardless of
     how sharding changes tnode creation order. *)
  let learning_rate = NTDSL.param ~value:0.05 "lr" () in
  (* Sum-over-batch squared error. [loss_of] creates its own parameter per call, so each shard gets
     a distinct (but identically-initialized) replica, as the driver requires. *)
  let loss_of x y =
    let w = TDSL.param ~values:[| 0.5 |] "w" ~output_dims:[ 1 ] () in
    [%op (((w *. x) - y) *. ((w *. x) - y)) ++ "...|... => |->0"]
  in
  Parallel.data_parallel ~backend_name:"cc" ~reduction:Parallel.Sum ~momentum ~weight_decay
    ~n_shards ~bindings:IDX.empty ~learning_rate ~inputs:(inputs ()) ~targets:(targets ()) ~loss_of
    ~f:(fun h ->
      for _ = 1 to steps do
        h.Parallel.step ()
      done;
      if print_loss then
        Stdio.printf "n_shards=%d: loss=%.4f\n" n_shards (h.Parallel.owner_loss_value ());
      h.Parallel.sync_params_to_host ();
      Array.concat_map h.Parallel.owner_params ~f:(fun p -> h.Parallel.read_values p))
    ()

let only_value = function
  | [| value |] -> value
  | values ->
      failwith (Printf.sprintf "expected one optimizer parameter, got %d" (Array.length values))

(* For this model, d(loss)/dw = 2 * sum(x^2) * (w - 2) = 60 * (w - 2). Keeping the host recurrence
   beside the executed wrapper test makes a dropped forwarding option fail against an independent
   value, not merely against another invocation of the same code. *)
let optimizer_oracle ~momentum ~weight_decay ~steps =
  let w = ref 0.5 in
  let buffer = ref 0.0 in
  for _ = 1 to steps do
    let delta = (60.0 *. (!w -. 2.0)) +. (weight_decay *. !w) in
    let delta =
      if Float.(momentum > 0.0) then (
        buffer := (momentum *. !buffer) +. delta;
        !buffer)
      else delta
    in
    w := !w -. (0.05 *. delta)
  done;
  !w

let optimizer_option_case label ~momentum ~weight_decay ~baseline =
  let steps = 2 in
  let actual = run ~momentum ~weight_decay ~steps ~print_loss:false ~n_shards:1 () |> only_value in
  let expected = optimizer_oracle ~momentum ~weight_decay ~steps in
  let oracle_error = Float.abs (actual -. expected) in
  let effect_size = Float.abs (actual -. baseline) in
  Stdio.eprintf
    "Parallel.data_parallel %s (not part of the golden): expected %.8g, got %.8g; oracle error \
     %.3e; option effect %.3e\n\
     %!"
    label expected actual oracle_error effect_size;
  Verdict.pass_fail
    ("data_parallel forwards " ^ label ^ ": matches the host oracle")
    Float.(oracle_error < 1e-4)
    ~detail:(fun () -> Printf.sprintf "absolute error %.3e" oracle_error);
  Verdict.pass_fail
    ("data_parallel forwards " ^ label ^ ": non-default changes the result")
    Float.(effect_size > 1e-3)
    ~detail:(fun () -> Printf.sprintf "absolute difference %.3e" effect_size)

(* Exercise multi-step training through [set_batch]: a second step on a fresh batch must keep
   training (finite loss, parameter still moving toward the target). *)
let multistep_ok () : bool =
  Tensor.unsafe_reinitialize ();
  Utils.settings.fixed_state_for_init <- Some 1;
  let learning_rate = NTDSL.param ~value:0.02 "lr" () in
  let loss_of x y =
    let w = TDSL.param ~values:[| 0.5 |] "w" ~output_dims:[ 1 ] () in
    [%op (((w *. x) - y) *. ((w *. x) - y)) ++ "...|... => |->0"]
  in
  (* Run on multidev_cc: with 2 shards the shards land on devices 0 and 1, exercising the
     cross-device merge-buffer broadcast and gradient all-reduce paths. *)
  Parallel.data_parallel ~backend_name:"multidev_cc" ~reduction:Parallel.Mean ~n_shards:2
    ~bindings:IDX.empty ~learning_rate ~inputs:(inputs ()) ~targets:(targets ()) ~loss_of
    ~f:(fun h ->
      h.Parallel.step ();
      let l1 = h.Parallel.owner_loss_value () in
      (* Feed a fresh batch and step again. *)
      h.Parallel.set_batch
        ~inputs:(make_batch "b2" [ [| 5. |]; [| 6. |]; [| 7. |]; [| 8. |] ])
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
    [%op (((w *. x) + uniform1 () - y) *. ((w *. x) + uniform1 () - y)) ++ "...|... => |->0"]
  in
  Parallel.data_parallel ~backend_name:"cc" ~reduction:Parallel.Sum ~n_shards:2 ~base_seed
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

(* Shard-to-shard divergence: the driver must seed shard 0 and shard 1 *differently* (base_seed +
   i), not all with base_seed. The handle reports the exact per-shard seeds it used; this asserts
   they are pairwise distinct (specifically shard 0 <> shard 1, and equal to base_seed + i). Flips
   if the driver seeds every shard with base_seed (the reviewer's mutation target: dropping the [+
   i]). A draw comparison cannot stand in here because shards already diverge through distinct
   [self_id]s regardless of the seed. *)
let shards_seeded_distinctly () : bool =
  let learning_rate = NTDSL.param ~value:0.0 "lr" () in
  let loss_of x y =
    let w = TDSL.param ~values:[| 0.5 |] "w" ~output_dims:[ 1 ] () in
    [%op (((w *. x) + uniform1 () - y) *. ((w *. x) + uniform1 () - y)) ++ "...|... => |->0"]
  in
  Parallel.data_parallel ~backend_name:"cc" ~n_shards:2 ~base_seed:100 ~bindings:IDX.empty
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
    [%op (((w *. x) - y) *. ((w *. x) - y)) ++ "...|... => |->0"]
  in
  Parallel.data_parallel ~backend_name:"cc" ~n_shards:2 ~base_seed:0 ~bindings:IDX.empty
    ~learning_rate ~inputs:(inputs ()) ~targets:(targets ()) ~loss_of
    ~f:(fun h -> h.Parallel.step ())
    ();
  phys_equal before (Tensor.get_random_seed ())

(* gh-ocannl-1629: [~bindings] works. Each shard slices its two rows with a batch index [i]: a
   ranged parameter given no value is refused, naming it, before anything is built; given a value
   through [~launch_values] every shard's routines run at it, and the handle's [set_launch_value]
   moves it between steps. The rows a step trained on are read off [w]: with lr = 1/16 and the
   shards' rows x = (1, 2) and (3, 4) of y = 2x, a step from w at row i moves w by -(1/16) * 2 *
   (x_a^2 + x_b^2) * (w - 2) over the two shards' row-i inputs -- 40 for i = 1, 20 for i = 0. Dyadic
   throughout, so the pins are exact: w = 0.5 -> 4.25 (i = 1) -> 1.4375 (i = 0); a run that ignored
   the value (slice 0 both times: 2.375, 1.90625) or the move (slice 1 both times: -1.375) misses
   them. *)
let sliced_sym = ref None

let sliced_run ?launch_values ~second () =
  Tensor.unsafe_reinitialize ();
  Utils.settings.fixed_state_for_init <- Some 1;
  let i, bindings = IDX.get_static_symbol ~static_range:2 IDX.empty in
  sliced_sym := Some i;
  let learning_rate = NTDSL.param ~value:0.0625 "lr" () in
  let loss_of x y =
    let w = TDSL.param ~values:[| 0.5 |] "w" ~output_dims:[ 1 ] () in
    [%op (((w *. (x @| i)) - (y @| i)) *. ((w *. (x @| i)) - (y @| i))) ++ "...|... => |->0"]
  in
  Parallel.data_parallel ~backend_name:"cc" ~reduction:Parallel.Sum ~n_shards:2 ~bindings
    ?launch_values:(Option.map launch_values ~f:(fun v -> [ (i, v) ]))
    ~learning_rate ~inputs:(inputs ()) ~targets:(targets ()) ~loss_of
    ~f:(fun h ->
      let read () = only_value (h.Parallel.read_values h.Parallel.owner_params.(0)) in
      h.Parallel.step ();
      let w1 = read () in
      h.Parallel.set_launch_value i second;
      h.Parallel.step ();
      (i, w1, read ()))
    ()

let sliced_case () =
  (match sliced_run ~second:0 () with
  | _ -> Verdict.p "data_parallel with a ranged batch index given no value is refused" false
  | exception Utils.User_error msg ->
      Stdio.eprintf "refused: %s (not part of the golden)\n%!" msg;
      Verdict.p "data_parallel with a ranged batch index given no value is refused" true;
      Verdict.p "that refusal names the batch index, as one"
        (String.is_substring msg ~substring:"(batch-slice index,"
        && Option.exists !sliced_sym ~f:(fun i ->
            String.is_substring msg
              ~substring:(Ir.Indexing.symbol_ident i.Ir.Indexing.static_symbol ^ " "))));
  let _, w1, w2 = sliced_run ~launch_values:1 ~second:0 () in
  Stdio.eprintf "sliced data_parallel: w1=%g w2=%g (not part of the golden)\n%!" w1 w2;
  Verdict.p "data_parallel runs every shard at the launch value given: w = 4.25" Float.(w1 = 4.25);
  Verdict.p "set_launch_value moves the next step to its value: w = 1.4375" Float.(w2 = 1.4375)

let () =
  let p1 = run ~n_shards:1 () in
  let p2 = run ~n_shards:2 () in
  (* Exact parameter digits to stderr (gh-ocannl-725): the two runs reach w by summing the same
     gradient terms in DIFFERENT orders -- that is the whole point of the comparison -- so the last
     of six printed decimals of a single-precision value near 5.0 is within one ulp of flipping.
     stdout keeps the tolerance claim below, which is what the two lines were there to show. *)
  let show p = String.concat ~sep:" " (Array.to_list (Array.map p ~f:(Printf.sprintf "%.6f"))) in
  Stdio.eprintf "w after 1-shard step  = [%s] (not part of the golden)\n%!" (show p1);
  Stdio.eprintf "w after 2-shard step  = [%s] (not part of the golden)\n%!" (show p2);
  (* An absolute pin to go with the parity claim, so the pair cannot both hold on a broken run that
     computes the same wrong thing twice: one step of lr=0.05 on w=0.5 against the closed-form
     gradient -sum 2*(w*x-y)*x = -90 lands w on 0.5 + 4.5. *)
  let expected_w = 5.0 in
  Verdict.pf "1-shard step lands w on the closed-form %g (within 1e-4)" expected_w
    (Array.length p1 = 1 && Float.(abs (p1.(0) - expected_w) < 1e-4));
  Verdict.p_close "data-parallel parity with single-shard baseline" p1 p2 ~tol:(fun _ -> 1e-4);
  let optimizer_baseline = run ~steps:2 ~print_loss:false ~n_shards:1 () |> only_value in
  optimizer_option_case "momentum" ~momentum:0.9 ~weight_decay:0.0 ~baseline:optimizer_baseline;
  optimizer_option_case "weight decay" ~momentum:0.0 ~weight_decay:0.1 ~baseline:optimizer_baseline;
  Verdict.p "driver routes per-shard seed into RNG" (driver_routes_seed_into_shards ());
  Verdict.p "shards seeded distinctly (base_seed + i)" (shards_seeded_distinctly ());
  Verdict.p "global random-seed singleton preserved across data_parallel"
    (seed_singleton_preserved ());
  Verdict.p "multi-step via set_batch ok" (multistep_ok ());
  sliced_case ()
