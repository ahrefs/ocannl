open! Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module Train = Ocannl.Train
module IDX = Train.IDX
module Row = Ocannl.Row
module Nn = Ocannl.Nn_blocks

(* gh-ocannl-1955: [batch_norm2d] with a [train_step] closes in a fresh session, on an input whose
   rows no earlier compile has closed. Compiling an evaluation pass first hid both failures this
   pins, so every leg here starts from a reinitialized session and goes straight to training:

   - [h, w, ..c.. = ..a.., d1, d2, d3], the spec against an operation's result, whose row is open on
   the left. The overlap of the flanks waits for the rank, and the equation must not meanwhile read
   the remainder of one side as the value of the other's row variable. - A statistic [..c..] is used
   by a tensor over [h, w, ..c..]: true as a bound, impossible as a value, so the statistic's row
   must not close to it. On 3x4 feature maps with 12 channels the use site's axes divide the channel
   total, so the total alone would not have refused it.

   [input] says what "sized" means: a leaf closes the row at once, a result only bounds it. *)

let batch = 2
let in_channels = 3
let dims t = Array.to_list (Shape.to_dims t.Tensor.shape)

let leaf ~h ~w =
  TDSL.range_of_shape ~label:[ "images" ] ~batch_dims:[ batch ] ~output_dims:[ h; w; in_channels ]
    ()

let result ~h ~w = TDSL.O.(leaf ~h ~w *. !.0.5)
let inputs = [ ("a leaf", leaf); ("an operation's result", result) ]

(* A value any backend computes, and a golden that cannot say which one did. *)
let context () =
  let ctx = Context.auto () in
  Stdio.eprintf "(not part of the golden) backend: %s\n%!" (Context.backend_name ctx);
  ctx

let named t prefix =
  Set.find_exn t.Tensor.params ~f:(fun p ->
      String.is_prefix (Ir.Tnode.label_name p.Tensor.value) ~prefix)

(* The statistics of one [batch_norm2d] directly on [input]. *)
let layer_state ~input =
  Tensor.unsafe_reinitialize ();
  let x = input ~h:4 ~w:4 in
  let bn = Nn.batch_norm2d ~label:[ "bn" ] () in
  let y = bn ~train_step:(Some x) x in
  (dims y, List.map (Set.to_list y.Tensor.params) ~f:dims)

let () =
  List.iter inputs ~f:(fun (name, input) ->
      Verdict.case ("batch_norm2d in training on " ^ name) (fun () ->
          let closed, state = layer_state ~input in
          p "the result keeps the input's shape" (List.equal Int.equal closed [ batch; 4; 4; 3 ]);
          p_all "every state tensor has the channel shape" state
            ~f:(List.equal Int.equal [ in_channels ]);
          (* [add_rank_edge] and [rank_edge_refuted] fail on any verdict the walk does not reach. *)
          let closed', state' = Row.with_unbounded_rank_search (fun () -> layer_state ~input) in
          p "under the reference rank walk it closes the same"
            (List.equal Int.equal closed closed' && List.equal (List.equal Int.equal) state state')))

(* [running_mean] and [running_variance] after one step: [0.9 * init + 0.1 * batch], by channel. *)
let () =
  Verdict.case "one training step on an operation's result, against a host oracle" (fun () ->
      Tensor.unsafe_reinitialize ();
      let h, w = (2, 4) in
      let x = result ~h ~w in
      let bn = Nn.batch_norm2d ~label:[ "bn" ] () in
      let y = bn ~train_step:(Some x) x in
      Train.set_materialized y.Tensor.value;
      let ctx = Train.forward_once (context ()) y in
      let observations = batch * h * w in
      let data = Array.init (observations * in_channels) ~f:(fun i -> 0.5 *. Float.of_int i) in
      let of_channel c = Array.filteri data ~f:(fun i _ -> i % in_channels = c) in
      let mean v = Array.fold v ~init:0. ~f:( +. ) /. Float.of_int observations in
      let variance v =
        let m = mean v in
        mean (Array.map v ~f:(fun x -> Float.square (x -. m)))
      in
      let by_channel f = Array.init in_channels ~f:(fun c -> f (of_channel c)) in
      let tol _ = 0.01 in
      p_close "running_mean moved by a tenth of the channel means"
        (Context.get_values ctx (named y "running_mean").value)
        (by_channel (fun v -> 0.1 *. mean v))
        ~tol;
      p_close "running_variance moved by a tenth of the way to the channel variances"
        (Context.get_values ctx (named y "running_variance").value)
        (by_channel (fun v -> 0.9 +. (0.1 *. variance v)))
        ~tol;
      Context.release ctx)

(* A compiled and executed training step of [conv_bn_relu]. *)
let () =
  List.iter inputs ~f:(fun (name, input) ->
      List.iter
        [
          (* The use site's axes exceed the channel total, *)
          (4, 4, 4);
          (* divide it, *)
          (3, 4, 12);
          (* and equal it. *)
          (2, 2, 4);
        ]
        ~f:(fun (h, w, channels) ->
          let title = Printf.sprintf "conv_bn_relu, %dx%d to %d channels on %s" h w channels name in
          Verdict.case title (fun () ->
              Tensor.unsafe_reinitialize ();
              let x = input ~h ~w in
              let block = Nn.conv_bn_relu ~label:[ "b" ] ~out_channels:channels () in
              let y = block ~train_step:(Some x) x in
              let%op loss = y ++ "... | ... => | ->0" in
              let update = Train.grad_update loss in
              let learning_rate = NTDSL.O.(!.0.01) in
              let sgd = Train.sgd_update ~learning_rate loss in
              let ctx = Train.init_params (context ()) IDX.empty loss in
              let ctx, step =
                Context.compile ctx (Ir.Assignments.sequence [ update; sgd ]) IDX.empty
              in
              let ctx = Context.run ctx step in
              let variance = Context.get_values ctx (named loss "running_variance").value in
              p "the result has the feature map's shape"
                (List.equal Int.equal (dims y) [ batch; h; w; channels ]);
              p "the running statistics have one value per channel"
                (Array.length variance = channels);
              p_all "the step ran: every running variance moved and is finite"
                (Array.to_list variance) ~f:(fun v -> Float.is_finite v && not Float.(v = 1.));
              Context.release ctx)))
