open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

(* Every architecture in [Demo_models] builds, infers its shapes and runs a forward pass on a tiny
   input (gh-ocannl-880). Before this, three of them had no caller in the tree, and neither
   [mobile_cnn] nor [sokoban_cnn] could run at all: [global_avg_pool2d] kept its pooled axes as
   fixed-index axes the classifier could not consume, and [sokoban_cnn]'s convs had no channel
   counts. The dead-export census (test/operations/dead_export_scan.ml) keeps a demo added later
   from going without a caller; this test is where such a caller belongs. What it checks is the
   build and the result's shape, with every value finite -- not what any demo computes. The cases
   pool feature maps of different sizes (6x6 for [sokoban_cnn], 1x1 for [mobile_cnn]), so a pooling
   block whose captured dimensions were bound once, at its definition, fails here.

   Where a demo leaves its output width to the caller ([lenet] infers it from training targets,
   [multi_head_att_workshop]'s output projection from its use), the case pins it the way a caller
   would, with a spec'd add of a fixed-shape tensor (a plain [+] would broadcast instead). *)

let dims_of t = Array.to_list (Lazy.force t.Tensor.value.Ir.Tnode.dims)
let dims_string dims = String.concat ~sep:"x" (List.map dims ~f:Int.to_string)

(* [outputs] builds the demo and returns the tensors to forward ([roots]: no root embeds another,
   and roots share no computation) and the tensors to read back. *)
let smoke name ~outputs =
  Verdict.case name (fun () ->
      Tensor.unsafe_reinitialize ();
      let roots, reads = outputs () in
      List.iter reads ~f:(fun (_, t) ->
          Train.set_materialized t.Tensor.value;
          Ir.Tnode.set_observable t.Tensor.value);
      let ctx = List.fold roots ~init:(Context.auto ()) ~f:Train.forward_once in
      List.iter reads ~f:(fun (what, t) ->
          Stdio.printf "%s %s dims: %s\n" name what (dims_string (dims_of t));
          p_all
            (Printf.sprintf "%s %s: every value is finite" name what)
            (Array.to_list (Context.get_values ctx t.Tensor.value))
            ~f:Float.is_finite);
      Context.release ctx)

(* Inputs in [0, 1): flattened offsets over the input's size. *)
let images ~h ~w ~c =
  let scale = 1. /. Float.of_int (2 * h * w * c) in
  let t = TDSL.range_of_shape ~label:[ "images" ] ~batch_dims:[ 2 ] ~output_dims:[ h; w; c ] () in
  TDSL.O.(t *. !.scale)

(* [t] with its single output axis pinned to [k]. *)
let pin_width t k =
  let pin = TDSL.range_of_shape ~label:[ "pin" ] ~output_dims:[ k ] () in
  let%op pinned = t +++ "... | c; | c => ... | c" pin in
  pinned

let () =
  smoke "multi_head_att_workshop" ~outputs:(fun () ->
      let x = TDSL.range_of_shape ~label:[ "tokens" ] ~batch_dims:[ 2; 3 ] ~output_dims:[ 8 ] () in
      let att = Demo_models.multi_head_att_workshop ~num_heads:2 ~d_k:4 ~d_v:4 () in
      let y = pin_width (att x) 8 in
      ([ y ], [ ("output", y) ]));
  smoke "transformer_with_loss" ~outputs:(fun () ->
      let vocab = 5 and seq = 3 in
      let model =
        Demo_models.transformer ~label:[ "t" ] ~num_encoder_layers:1 ~num_decoder_layers:1
          ~num_heads:2 ~d_enc:8 ~d_dec:8 ~d_ff:16 ()
      in
      let one_hot l ~shift =
        NTDSL.init ~l ~prec:Ir.Ops.single ~b:[ 2; seq ] ~i:[] ~o:[ vocab ]
          ~f:(function [| b; s; v |] -> if v = (b + s + shift) % vocab then 1. else 0. | _ -> 0.)
          ()
      in
      let mask =
        NTDSL.init ~l:"mask" ~prec:Ir.Ops.single ~b:[ seq ] ~i:[ seq ] ~o:[]
          ~f:(function [| s; t |] -> if s >= t then 1. else 0. | _ -> 0.)
          ()
      in
      let loss, logits =
        Demo_models.transformer_with_loss ~label:[ "tl" ] ~model () ~train_step:None
          ~src:(one_hot "src" ~shift:0) ~tgt_input:(one_hot "tgt_input" ~shift:1)
          ~tgt_target:(one_hot "tgt_target" ~shift:2) ~mask
      in
      (* The loss embeds the logits, so it alone is forwarded. *)
      ([ loss ], [ ("logits", logits); ("loss", loss) ]));
  smoke "lenet" ~outputs:(fun () ->
      let logits = pin_width (Demo_models.lenet () ~train_step:None (images ~h:8 ~w:8 ~c:1)) 10 in
      ([ logits ], [ ("logits", logits) ]));
  smoke "sokoban_cnn" ~outputs:(fun () ->
      let action_logits, value =
        Demo_models.sokoban_cnn ~label:[ "s" ] () ~train_step:None
          ~grid_state:(images ~h:6 ~w:6 ~c:3)
      in
      (* The two heads share the trunk, so one root embeds both, as a loss over them would: the
         value, read at its one index, is added to every action. *)
      let%op heads = action_logits +++ "... | a; ... | 0 => ... | a" value in
      ([ heads ], [ ("action_logits", action_logits); ("value", value) ]));
  smoke "mobile_cnn" ~outputs:(fun () ->
      let logits =
        Demo_models.mobile_cnn ~label:[ "m" ] ~num_classes:3 ~width_mult:0.25 () ~train_step:None
          (images ~h:8 ~w:8 ~c:3)
      in
      ([ logits ], [ ("logits", logits) ]))
