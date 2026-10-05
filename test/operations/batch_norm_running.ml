open! Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module Train = Ocannl.Train
module IDX = Train.IDX

let epsilon = 0.125
let channels = 2

let check label actual expected =
  let error =
    Array.foldi actual ~init:0. ~f:(fun i e v -> Float.max e (Float.abs (v -. expected.(i))))
  in
  Stdio.eprintf "%s (not part of the golden): max error %.3e\n%!" label error;
  p label Float.(error < 2e-5)

let values observations shift =
  Array.init (observations * channels) ~f:(fun i ->
      let c = i % channels in
      shift +. Float.of_int ((i / channels * (c + 1)) + (c * 7)))

let moments data =
  let count = Array.length data / channels in
  let mean =
    Array.init channels ~f:(fun c ->
        Array.foldi data ~init:0. ~f:(fun i sum v -> if i % channels = c then sum +. v else sum)
        /. Float.of_int count)
  in
  let variance =
    Array.init channels ~f:(fun c ->
        Array.foldi data ~init:0. ~f:(fun i sum v ->
            if i % channels = c then sum +. Float.square (v -. mean.(c)) else sum)
        /. Float.of_int count)
  in
  (mean, variance)

let normalize data mean variance =
  Array.mapi data ~f:(fun i v ->
      let c = i % channels in
      (v -. mean.(c)) /. Float.sqrt (variance.(c) +. epsilon))

let trajectory spatial momentum =
  Tensor.unsafe_reinitialize ();
  let observations, b, o, make =
    if spatial then (8, [ 2 ], [ 2; 2; channels ], Ocannl.Nn_blocks.batch_norm2d)
    else (4, [ 2; 2 ], [ channels ], Ocannl.Nn_blocks.batch_norm1d)
  in
  let name = if spatial then "2d" else "1d" in
  let first = values observations 0. in
  let x = NTDSL.init ~l:"input" ~prec:Ir.Ops.single ~b ~o ~f:(fun _ -> 0.) () in
  Train.set_materialized x.Tensor.value;
  let layer = make ~label:[ name ] ~epsilon ?momentum () in
  let eval_o = if spatial then [ 1; 1; channels ] else [ channels ] in
  let eval_x =
    NTDSL.init ~l:"eval_input" ~prec:Ir.Ops.single ~b:[ 1 ] ~o:eval_o ~f:(fun _ -> 0.) ()
  in
  Train.set_materialized eval_x.Tensor.value;
  (* Fresh inference must resolve without a training graph. This layer later trains on a larger
     batch, and larger spatial dimensions in 2d. Only the channel axes belong to its state. *)
  let inference = layer ~train_step:None eval_x in
  Train.set_materialized inference.Tensor.value;
  let ctx = Train.init_params (Context.cpu ()) IDX.empty inference in
  let ctx, testing =
    Context.compile ~name:"bn_inference" ctx (Tensor.consume_forward_code inference) IDX.empty
  in
  let y = layer ~train_step:(Some 0) x in
  Train.set_materialized y.Tensor.value;
  let ctx, training =
    Context.compile ~name:"bn_training" ctx (Tensor.consume_forward_code y) IDX.empty
  in
  p
    (name ^ ": four state tensors, two trainable")
    (Set.length y.params = 4 && Set.length (Train.trainable_params y) = 2);
  p_all (name ^ ": every state tensor has exactly the channel shape") (Set.to_list y.params)
    ~f:(fun p -> Array.equal Int.equal (Lazy.force p.Tensor.value.dims) [| channels |]);
  let mean = ref [| 0.; 0. |] and variance = ref [| 1.; 1. |] in
  let ctx = ref ctx in
  let infer data label =
    ctx := Context.set_values !ctx eval_x.value data;
    ctx := Context.run !ctx testing;
    let actual = Context.get_values !ctx inference.value in
    check (name ^ ": " ^ label) actual (normalize data !mean !variance);
    actual
  in
  ignore (infer (values 1 0.) "inference initializes to mean zero / variance one" : float array);
  let retention = Option.value momentum ~default:0.9 in
  List.iter
    [ first; values observations 5. ]
    ~f:(fun data ->
      ctx := Context.set_values !ctx x.value data;
      ctx := Context.run !ctx training;
      let bm, bv = moments data in
      check
        (name ^ ": training uses batch statistics")
        (Context.get_values !ctx y.value) (normalize data bm bv);
      mean := Array.mapi !mean ~f:(fun c v -> (retention *. v) +. ((1. -. retention) *. bm.(c)));
      variance :=
        Array.mapi !variance ~f:(fun c v -> (retention *. v) +. ((1. -. retention) *. bv.(c))));
  let shifted = values 1 17. in
  let result = infer shifted "inference follows two-step running-stat recurrence" in
  ignore (infer (values 1 (-9.)) "inference leaves statistics unchanged" : float array);
  ignore (infer shifted "repeated inference leaves statistics unchanged" : float array);
  ctx := Train.init_params ~reinit_all:true !ctx IDX.empty inference;
  mean := [| 0.; 0. |];
  variance := [| 1.; 1. |];
  ignore (infer shifted "explicit reinitialization resets stored statistics" : float array);
  Context.release !ctx;
  result

let gradient spatial =
  Tensor.unsafe_reinitialize ();
  let observations, b, o, make =
    if spatial then (8, [ 2 ], [ 2; 2; channels ], Ocannl.Nn_blocks.batch_norm2d)
    else (4, [ 2; 2 ], [ channels ], Ocannl.Nn_blocks.batch_norm1d)
  in
  let data = values observations 3. in
  let weights = Array.mapi data ~f:(fun i _ -> Float.of_int (((i * i) + 3) % 7)) in
  let lookup values ij =
    values.(Array.fold2_exn ij (Array.of_list (b @ o)) ~init:0 ~f:(fun i j d -> (i * d) + j))
  in
  let x =
    Ocannl.Operation.init ~l:"gradient_input" ~prec:Ir.Ops.single ~b ~o ~f:(lookup data)
      ~grad_spec:Tensor.Require_grad ()
  in
  let w = NTDSL.init ~l:"weights" ~prec:Ir.Ops.single ~b ~o ~f:(lookup weights) () in
  let layer = make ~label:[ "gradient" ] ~epsilon ~momentum:0.5 () in
  let y = layer ~train_step:(Some 0) x in
  let%op loss = (y *. w) ++ "... | ... => | ->0" in
  let dx = (Option.value_exn x.Tensor.diff).grad in
  Train.set_materialized dx;
  let update = Train.grad_update loss in
  let ctx = Train.init_params (Context.cpu ()) IDX.empty loss in
  let ctx, routine = Context.compile ctx update IDX.empty in
  let ctx = Context.run ctx routine in
  let mean, variance = moments data in
  let yhat = normalize data mean variance in
  let mg, _ = moments weights in
  let mgy, _ = moments (Array.mapi weights ~f:(fun i v -> v *. yhat.(i))) in
  let expected =
    Array.mapi weights ~f:(fun i v ->
        let c = i % channels in
        (v -. mg.(c) -. (yhat.(i) *. mgy.(c))) /. Float.sqrt (variance.(c) +. epsilon))
  in
  check
    (if spatial then "2d: input gradient matches batch-norm oracle"
     else "1d: input gradient matches batch-norm oracle")
    (Context.get_values ctx dx) expected;
  Context.release ctx

let () =
  List.iter [ false; true ] ~f:(fun spatial ->
      Verdict.case
        (if spatial then "2d" else "1d")
        (fun () ->
          gradient spatial;
          let latest = trajectory spatial (Some 0.) in
          let averaged = trajectory spatial (Some 0.5) in
          p_exists "momentum changes executed inference"
            (Array.to_list (Array.mapi averaged ~f:(fun i v -> Float.abs (v -. latest.(i)))))
            ~f:(fun difference -> Float.(difference > 0.1));
          ignore (trajectory spatial (Some 1.) : float array);
          let explicit = trajectory spatial (Some 0.9) in
          let default = trajectory spatial None in
          check "default retains 0.9 of previous statistics" default explicit))
