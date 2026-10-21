open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

(* gh-ocannl-880: [Demo_models.mobile_cnn ?width_mult] scales every internal channel width -- 32,
   then 64, 128, 128, 256 -- to the nearest multiple of 8 (ties up, at least 8), and leaves the
   external dimensions alone: the input's channels and the [num_classes] logits. *)

(* A multiplier is written as an integer fraction [num/den]: [num /. den] is the same double as the
   decimal literal, and the label prints the integers, keeping float formatting out of the
   golden. *)
let mult ~num ~den = Float.of_int num /. Float.of_int den

(* The rounding rule on its own, against hand-computed widths: [channels *. width_mult] is in the
   comment of each row. *)
let () =
  let rows =
    [
      (* 32.0, 64.0: unscaled widths are already multiples of 8. *)
      (1, 1, 32, 32);
      (1, 1, 64, 64);
      (* 24.0, 48.0, 96.0: exact multiples. *)
      (3, 4, 32, 24);
      (3, 4, 64, 48);
      (3, 4, 128, 96);
      (* 19.2 rounds down to 16, 38.4 up to 40, 76.8 up to 80, 153.6 down to 152. *)
      (3, 5, 32, 16);
      (3, 5, 64, 40);
      (3, 5, 128, 80);
      (3, 5, 256, 152);
      (* 36.0 and 108.0 are ties, an odd multiple of 4: they round up. *)
      (9, 8, 32, 40);
      (9, 8, 96, 112);
      (* 3.2, 6.4 and even 0.32 reach the floor of 8; 12.8 is past it. *)
      (1, 10, 32, 8);
      (1, 10, 64, 8);
      (1, 100, 32, 8);
      (1, 10, 128, 16);
      (* Widening: 41.6 rounds down to 40, 332.8 up to 336. *)
      (13, 10, 32, 40);
      (13, 10, 256, 336);
    ]
  in
  List.iter rows ~f:(fun (num, den, channels, want) ->
      let got = Demo_models.scale_channels ~width_mult:(mult ~num ~den) channels in
      pf "scale_channels ~width_mult:%d/%d %d = %d" num den channels want (got = want))

(* An invalid multiplier is refused, by [scale_channels] and by [mobile_cnn] itself before it
   creates any tensor: the session's next tensor id does not move. *)
let () =
  List.iter
    [ ("0", 0.); ("-0.5", -0.5); ("nan", Float.nan); ("infinity", Float.infinity) ]
    ~f:(fun (name, width_mult) ->
      let refused f = match f () with _ -> false | exception Invalid_argument _ -> true in
      pf "scale_channels refuses width_mult %s" name
        (refused (fun () -> ignore (Demo_models.scale_channels ~width_mult 32 : int)));
      let next_id = Tensor.get_next_id () in
      pf "mobile_cnn refuses width_mult %s before creating a tensor" name
        (refused (fun () ->
             ignore (Demo_models.mobile_cnn ~label:[ "refused" ] ~width_mult () : train_step:_ -> _))
        && Tensor.get_next_id () = next_id))

let num_classes = 5
let in_channels = 3
let dims_of t = Array.to_list (Lazy.force t.Tensor.value.Ir.Tnode.dims)

(* The parameter of [model] whose debug name is [name] followed by the model's label. *)
let param model name =
  let full = name ^ "_m" in
  match
    List.filter (Set.to_list model.Tensor.params) ~f:(fun p ->
        String.equal (Ir.Tnode.debug_name p.Tensor.value) full)
  with
  | [ p ] -> p
  | ps -> failwith (Printf.sprintf "expected one parameter %s, found %d" full (List.length ps))

(* The value each parameter is set to for the executed forward pass: zero kernels make every conv's
   output its bias, so with unit biases and identity batch-norm statistics every feature map is a
   constant whatever the padding, and the classifier sums [c4] copies of one value. *)
let oracle_value name =
  let prefixed p = String.is_prefix name ~prefix:p in
  if prefixed "kernel_" || prefixed "dw_kernel_" || prefixed "pw_kernel_" then 0.
  else if prefixed "bias_" || prefixed "gamma_" || prefixed "running_variance_" then 1.
  else if prefixed "beta_" || prefixed "running_mean_" || prefixed "b_classifier" then 0.
  else if prefixed "w_classifier" then 1.
  else failwith ("no oracle value for parameter " ^ name)

let run ~num ~den ~widths =
  let width_mult = mult ~num ~den in
  let tag = Printf.sprintf "width_mult %d/%d" num den in
  Verdict.case tag (fun () ->
      Tensor.unsafe_reinitialize ();
      (* 16x16 RGB images: the two stride-2 stages after the stride-2 stem leave 2x2 feature maps,
         so the global pool really averages. *)
      let x =
        TDSL.range_of_shape ~label:[ "images" ] ~batch_dims:[ 2 ]
          ~output_dims:[ 16; 16; in_channels ] ()
      in
      let model = Demo_models.mobile_cnn ~label:[ "m" ] ~num_classes ~width_mult () in
      let y = model ~train_step:None x in
      Train.set_materialized y.Tensor.value;
      Ir.Tnode.set_observable y.Tensor.value;
      let ctx = Context.auto () in
      let ctx = Train.init_params ctx Ir.Indexing.Empty y in
      let c0, c1, c2, c3, c4 =
        match widths with
        | [ c0; c1; c2; c3; c4 ] -> (c0, c1, c2, c3, c4)
        | _ -> failwith "five widths"
      in
      Stdio.printf "%s: channels %s\n" tag
        (String.concat ~sep:", " (List.map widths ~f:Int.to_string));
      (* Every scalable boundary, as the layer dims put it (output axes, then input axes): the stem
         reads the unscaled input channels, each block's depthwise kernel reads the previous width
         and its pointwise kernel maps it to the block's own, and the classifier maps the last width
         to the unscaled [num_classes]. *)
      let shapes =
        [
          ("kernel_conv_conv_init", [ c0; 3; 3; in_channels ]);
          ("bias_conv_conv_init", [ c0 ]);
          ("gamma_bn_conv_init", [ c0 ]);
          ("running_mean_bn_conv_init", [ c0 ]);
          ("dw_kernel_dw1", [ c0; 3; 3 ]);
          ("pw_kernel_dw1", [ c1; c0 ]);
          ("bias_dw1", [ c1 ]);
          ("dw_kernel_dw2", [ c1; 3; 3 ]);
          ("pw_kernel_dw2", [ c2; c1 ]);
          ("bias_dw2", [ c2 ]);
          ("dw_kernel_dw3", [ c2; 3; 3 ]);
          ("pw_kernel_dw3", [ c3; c2 ]);
          ("bias_dw3", [ c3 ]);
          ("dw_kernel_dw4", [ c3; 3; 3 ]);
          ("pw_kernel_dw4", [ c4; c3 ]);
          ("bias_dw4", [ c4 ]);
          ("gamma_bn_final", [ c4 ]);
          ("running_variance_bn_final", [ c4 ]);
          ("w_classifier", [ num_classes; c4 ]);
          ("b_classifier", [ num_classes ]);
        ]
      in
      p_all (tag ^ ": every scalable channel boundary has its scaled width") shapes
        ~f:(fun (name, want) ->
          let got = dims_of (param y name) in
          let ok = List.equal Int.equal got want in
          if not ok then
            Stdio.eprintf "%s: %s has dims [%s] (not part of the golden)\n" tag name
              (String.concat ~sep:"; " (List.map got ~f:Int.to_string));
          ok);
      p_all (tag ^ ": the listed shapes cover every parameter") (Set.to_list y.Tensor.params)
        ~f:(fun p ->
          let name = Ir.Tnode.debug_name p.Tensor.value in
          List.exists shapes ~f:(fun (listed, _) -> String.equal name (listed ^ "_m"))
          || List.mem
               [
                 "beta_bn_conv_init_m";
                 "running_variance_bn_conv_init_m";
                 "beta_bn_final_m";
                 "running_mean_bn_final_m";
               ]
               name ~equal:String.equal);
      pf "%s: the logits are [batch 2 | %d classes]" tag num_classes
        (List.equal Int.equal (dims_of y) [ 2; num_classes ]);
      let ctx =
        Set.fold y.Tensor.params ~init:ctx ~f:(fun ctx p ->
            let tn = p.Tensor.value in
            let count = List.fold (dims_of p) ~init:1 ~f:( * ) in
            Context.set_values ctx tn
              (Array.create ~len:count (oracle_value (Ir.Tnode.debug_name tn))))
      in
      let ctx, routine = Context.compile ctx (Train.forward y) Ir.Indexing.Empty in
      let ctx = Context.run ctx routine in
      let got = Context.get_values ctx y.Tensor.value in
      (* Every batch-norm output is [1 / sqrt (1 + epsilon)] (epsilon = 1e-5), the depthwise blocks
         reset their maps to the bias 1, and the classifier sums the [c4] pooled channels. *)
      let s = 1. /. Float.sqrt (1. +. 1e-5) in
      let want = Array.create ~len:(2 * num_classes) (Float.of_int c4 *. s) in
      pf_close "%s: the executed logits are the last width's channel sum" tag got want
        ~tol:(fun w -> 1e-5 *. Float.abs w);
      Context.release ctx)

let () =
  run ~num:1 ~den:1 ~widths:[ 32; 64; 128; 128; 256 ];
  (* Exact multiples of 8. *)
  run ~num:3 ~den:4 ~widths:[ 24; 48; 96; 96; 192 ];
  (* 19.2, 38.4, 76.8, 76.8, 153.6: rounding both down and up. *)
  run ~num:3 ~den:5 ~widths:[ 16; 40; 80; 80; 152 ];
  (* 36 is a tie, rounding up; 72, 144, 144, 288 are exact. *)
  run ~num:9 ~den:8 ~widths:[ 40; 72; 144; 144; 288 ];
  (* 3.2, 6.4, 12.8, 12.8, 25.6: the floor of 8 for the first two. *)
  run ~num:1 ~den:10 ~widths:[ 8; 8; 16; 16; 24 ]
