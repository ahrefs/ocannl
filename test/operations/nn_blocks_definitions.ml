(* gh-ocannl-1751: the definition-check convention, applied to the library. A [%op] block definition
   is checked early by [Tensor.Scheme.check_definition], which runs a trial application at a default
   configuration in a window that is always rewound: the [()] traces run there, and with them any
   shape error that holds for every argument. [lib/nn_blocks.ml] does not do this at load time
   (module initialization is no place for a trace); this test does it for every schemed block. *)

open Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module Nn = Ocannl.Nn_blocks
module Row = Ocannl.Row

let ignore_ f () = ignore (f () : _)

(* The variant labels of a block whose returned function takes [~(train_step : Tensor.t option)]
   (gh-ocannl-1792): one scheme per presence pattern. *)
let train_eval name = [ name ^ " [train_step]"; name ^ " [no train_step]" ]

(* Each schemed block, by the name its scheme is traced under, with a trial configuration and the
   labels its schemes are traced under: every presence pattern of its option-valued parameters. *)
let blocks =
  let plain name trial = (name, trial, [ name ]) in
  let variants name trial = (name, trial, train_eval name) in
  let mha = Nn.multi_head_attention ~label:[ "t" ] ~num_heads:2 ~d_k:4 ~d_v:4 in
  [
    plain "mlp_layer" (ignore_ (fun () -> Nn.mlp_layer ~label:[ "t" ] ~hid_dim:8 ()));
    plain "mlp" (ignore_ (fun () -> Nn.mlp ~label:[ "t" ] ~hid_dims:[ 8; 8 ] ()));
    plain "softmax" (ignore_ (fun () -> Nn.softmax ~spec:"... | v" ()));
    plain "cross_entropy_loss" (ignore_ (fun () -> Nn.cross_entropy_loss ~spec:"... | v" ()));
    plain "layer_norm" (ignore_ (fun () -> Nn.layer_norm ~label:[ "t" ] ()));
    plain "conv2d" (ignore_ (fun () -> Nn.conv2d ~label:[ "t" ] ~out_channels:4 ()));
    plain "depthwise_separable_conv2d"
      (ignore_ (fun () -> Nn.depthwise_separable_conv2d ~label:[ "t" ] ~out_channels:4 ()));
    plain "max_pool2d" (ignore_ (fun () -> Nn.max_pool2d ()));
    plain "max_pool2d_copy" (ignore_ (fun () -> Nn.max_pool2d_copy ()));
    plain "avg_pool2d" (ignore_ (fun () -> Nn.avg_pool2d ()));
    variants "dropout" (ignore_ (fun () -> Nn.dropout ~rate:0.0 ()));
    ( "multi_head_attention",
      ignore_ (fun () -> mha ()),
      [
        "multi_head_attention [train_step, mask]";
        "multi_head_attention [train_step, no mask]";
        "multi_head_attention [no train_step, mask]";
        "multi_head_attention [no train_step, no mask]";
      ] );
    variants "cross_attention"
      (ignore_ (fun () -> Nn.cross_attention ~label:[ "t" ] ~num_heads:2 ~d_k:4 ~d_v:4 ()));
    variants "transformer_encoder_block"
      (ignore_ (fun () ->
           Nn.transformer_encoder_block ~label:[ "t" ] ~num_heads:2 ~d_k:4 ~d_v:4 ~d_ff:16 ()));
    variants "decoder_only_block"
      (ignore_ (fun () ->
           Nn.decoder_only_block ~label:[ "t" ] ~num_heads:2 ~d_k:4 ~d_v:4 ~d_ff:16 ()));
    variants "transformer_decoder_block"
      (ignore_ (fun () ->
           Nn.transformer_decoder_block ~label:[ "t" ] ~num_heads:2 ~d_k:4 ~d_v:4 ~d_ff:16 ()));
    variants "batch_norm1d" (ignore_ (fun () -> Nn.batch_norm1d ~label:[ "t" ] ()));
    variants "batch_norm2d" (ignore_ (fun () -> Nn.batch_norm2d ~label:[ "t" ] ()));
    variants "conv_bn_relu" (ignore_ (fun () -> Nn.conv_bn_relu ~label:[ "t" ] ~out_channels:4 ()));
    variants "resnet_block"
      (ignore_ (fun () -> Nn.resnet_block ~label:[ "t" ] ~stride:2 ~out_channels:4 ()));
    variants "vgg_block" (ignore_ (fun () -> Nn.vgg_block ~label:[ "t" ] ~num_convs:2 ()));
  ]

(* A block whose body contradicts itself whatever its argument. *)
let%op inconsistent () x = x + ({ c3 = 0.; o = [ 3 ] } + { c4 = 0.; o = [ 4 ] })

(* The blocks the [%op] expansion of [lib/nn_blocks.ml] schemes at a [()] stage, derived from the
   expansion itself: the list above is held to it, so a new schemed block needs its trial. *)
let schemed_at_unit =
  let text = Stdio.In_channel.read_all "../../lib/nn_blocks.ml" in
  Ppx_ocannl.schemed_definitions (Ppxlib.Parse.implementation (Lexing.from_string text))
  |> List.filter_map ~f:(fun (name, at_unit) -> if at_unit then Some name else None)
  |> List.dedup_and_sort ~compare:String.compare

let () =
  Verdict.case "the trials cover exactly the blocks the expansion schemes at ()" (fun () ->
      let trials =
        List.dedup_and_sort ~compare:String.compare (List.map blocks ~f:(fun (n, _, _) -> n))
      in
      Stdio.eprintf "(not part of the golden) schemed at (): %s\n%!"
        (String.concat ~sep:" " schemed_at_unit);
      p_exists "the expansion schemes some block at its () stage" schemed_at_unit ~f:(fun _ -> true);
      p "the trial list is the expansion's list" (List.equal String.equal trials schemed_at_unit));
  Verdict.case "every schemed nn_blocks block passes its definition check" (fun () ->
      Tensor.unsafe_reinitialize ();
      let ck = Tensor.checkpoint ~label:"before the checks" () in
      let next_id = Tensor.get_next_id () in
      List.iter blocks ~f:(fun (name, trial, labels) ->
          let claim = name ^ ": every presence pattern's scheme was traced" in
          match Tensor.Scheme.check_definition ~name trial with
          | traced -> p_all claim labels ~f:(fun l -> List.mem traced l ~equal:String.equal)
          | exception (Row.Shape_error (msg, _) | Tensor.Session_error (msg, _)) ->
              Stdio.eprintf "(not part of the golden) %s refused: %s\n%!" name msg;
              p claim false);
      p "the checks allocated no session id" (Tensor.get_next_id () = next_id);
      p_empty "the checks left the session as it was" ~over:Tensor.For_testing.restore_categories
        (Tensor.For_testing.checkpoint_differences ck));
  Verdict.case "live dropout: the training pattern alone is left unschemed" (fun () ->
      (* With [rate > 0] the training pattern draws random numbers, which a trace refuses
         (gh-ocannl-1791); the evaluation pattern keeps its scheme. *)
      Tensor.unsafe_reinitialize ();
      let unschemed = ref [] in
      let traced =
        Tensor.For_testing.observe_schemes
          (fun () ->
            Tensor.Scheme.check_definition ~name:"dropout"
              (ignore_ (fun () -> Nn.dropout ~rate:0.5 ())))
          ~on_event:(function
            | Tensor.Scheme.Unschemed (l, reason) -> unschemed := (l, reason) :: !unschemed
            | _ -> ())
      in
      List.iter !unschemed ~f:(fun (l, reason) ->
          Stdio.eprintf "(not part of the golden) %s unschemed: %s\n%!" l reason);
      p "the evaluation pattern was traced"
        (List.mem traced "dropout [no train_step]" ~equal:String.equal);
      p "the training pattern was not traced"
        (not (List.mem traced "dropout [train_step]" ~equal:String.equal));
      p "the training pattern was left unschemed"
        (List.Assoc.mem !unschemed "dropout [train_step]" ~equal:String.equal));
  Verdict.case "a step counter nothing reads stays a forward root of its own" (fun () ->
      (* At dropout rate 0 (attention's default) the training pattern draws nothing: the caller's
         [Some counter] is a tensor no graph consumes. It must not get in the way of the graph. *)
      Tensor.unsafe_reinitialize ();
      let step_n, _ = Ocannl.Train.IDX.get_static_symbol Ocannl.Train.IDX.empty in
      let counter = NTDSL.embed_symbol step_n in
      let mha = Nn.multi_head_attention ~label:[ "c" ] ~num_heads:2 ~d_k:4 ~d_v:4 () in
      let x =
        Tensor.term ~grad_spec:Prohibit_grad ~label:[ "x" ] ~batch_dims:[ 2; 3 ] ~output_dims:[ 8 ]
          ()
      in
      let y = mha ~train_step:(Some counter) x in
      p "the result's forward code is consumed, the counter beside it"
        (match Tensor.consume_forward_code y with (_ : Ir.Assignments.comp) -> true);
      p "the result is no longer a forward root" (not (Tensor.is_fwd_root y));
      p "the counter stays a forward root, unconsumed" (Tensor.is_fwd_root counter));
  Verdict.case "control: an inconsistent definition is reported, naming it" (fun () ->
      Tensor.unsafe_reinitialize ();
      let ck = Tensor.checkpoint ~label:"before the check" () in
      (match Tensor.Scheme.check_definition ~name:"inconsistent" (ignore_ inconsistent) with
      | _ -> p "the inconsistent definition is refused" false
      | exception Row.Shape_error (msg, _) ->
          Stdio.eprintf "(not part of the golden) refused: %s\n%!" msg;
          p "the inconsistent definition is refused, naming the definition and its scheme"
            (String.is_substring msg
               ~substring:
                 "checking the definition inconsistent: tracing the shape scheme inconsistent"));
      p_empty "the failed check left the session as it was"
        ~over:Tensor.For_testing.restore_categories
        (Tensor.For_testing.checkpoint_differences ck);
      p "the session is usable after the failed check"
        (match Ocannl.Nn_blocks.mlp_layer ~label:[ "after" ] ~hid_dim:2 () with
        | (_ : Tensor.t -> Tensor.t) -> true
        | exception _ -> false))
