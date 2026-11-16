open! Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims

let default_lone_param () =
  Stdio.printf "Testing default lone parameter -- corner case\n";
  Tensor.unsafe_reinitialize ();
  (* This should raise an error because we have a parameter with unspecified dimensions *)
  try
    let w_o = TDSL.param "w_o" () in
    let _ctx : Context.t = Train.init_params (Context.auto ()) Train.IDX.empty w_o in
    Train.printf _ctx w_o;
    Verdict.fail "should have refused an unsized lone parameter"
  with Row.Shape_error (msg, _) -> Stdio.printf "Got acceptable error: %s\n" msg

let lone_param_1d () =
  Stdio.printf "Testing lone parameter 1D -- corner case\n";
  Tensor.unsafe_reinitialize ();
  (* This should raise an error because we have a parameter with unspecified dimensions *)
  try
    let%op w_o = { w_o = uniform1 () } in
    let _ctx : Context.t = Train.init_params (Context.auto ()) Train.IDX.empty w_o in
    Train.printf _ctx w_o;
    Verdict.fail "should have refused an unsized lone parameter"
  with Row.Shape_error (msg, _) -> Stdio.printf "Got acceptable error: %s\n" msg

let default_linear_op () =
  Stdio.printf "Testing default linear operation\n";
  Tensor.unsafe_reinitialize ();
  (* This should raise an error because we have a parameter with unspecified dimensions *)
  try
    let%op w_o = { w = uniform1 () } * [ 3; 4; 5 ] in
    let _ctx : Context.t = Train.init_params (Context.auto ()) Train.IDX.empty w_o in
    Train.printf _ctx w;
    Verdict.fail "should have refused an unsized parameter"
  with Row.Shape_error (msg, _) -> Stdio.printf "Got expected error: %s\n" msg

let default_affine_op_propagated () =
  Stdio.printf "Testing default affine operation with propagated dimensions\n";
  Tensor.unsafe_reinitialize ();
  (* No error: the sum's [1; 2] sizes w's output through the product, so w is [3 -> 2]. *)
  try
    let%op w_o = [ 1; 2 ] + ({ w = uniform1 () } * [ 3; 4; 5 ]) in
    let _ctx : Context.t = Train.init_params (Context.auto ()) Train.IDX.empty w_o in
    Train.printf _ctx w;
    ()
  with Row.Shape_error (msg, _) -> Verdict.fail ("sized parameter was refused: " ^ msg)

let default_affine_op_unknown_input () =
  Stdio.printf "Testing default affine operation with unknown input dimensions\n";
  Tensor.unsafe_reinitialize ();
  (* An error: x is a parameter and nothing sizes its output, the hidden axis between w and x,
     however w's output is sized. Until gh-ocannl-2126 this closed that axis empty: a guess equated
     w's input row with x's output row, and w's input row may close empty. *)
  try
    let%op w_o = [ 1; 2 ] + ({ w = uniform1 () } * { x = uniform1 () }) in
    let _ctx : Context.t = Train.init_params (Context.auto ()) Train.IDX.empty w_o in
    Train.printf _ctx w;
    Verdict.fail "should have refused an unsized parameter"
  with Row.Shape_error (msg, _) -> Stdio.printf "Got expected error: %s\n" msg

let default_bias_param () =
  Stdio.printf "Testing default bias parameter\n";
  Tensor.unsafe_reinitialize ();
  (* This should raise an error because we have a parameter with unspecified dimensions *)
  try
    let%op w_o = { x } + { y } in
    let _ctx : Context.t = Train.init_params (Context.auto ()) Train.IDX.empty w_o in
    Train.printf _ctx w_o;
    Verdict.fail "should have raised an exception"
  with Row.Shape_error (msg, _) -> Stdio.printf "Got expected error: %s\n" msg

let default_bias_param_1d () =
  Stdio.printf "Testing bias parameter 1D\n";
  Tensor.unsafe_reinitialize ();
  (* This should raise an error because we have a parameter with unspecified dimensions *)
  try
    let%op w_o = { x = uniform1 () } + { y = uniform1 () } in
    let _ctx : Context.t = Train.init_params (Context.auto ()) Train.IDX.empty w_o in
    Train.printf _ctx w_o;
    Verdict.fail "should have refused an unsized parameter"
  with Row.Shape_error (msg, _) -> Stdio.printf "Got expected error: %s\n" msg

(* gh-ocannl-1508: a parameter whose input axes are a fixed-index result of an einsum -- the shape
   [Nn_blocks.global_avg_pool2d] had before gh-ocannl-880 -- is not a forgotten hidden dimension: a
   fixed index only bounds the axis from below. The error names the spec instead. *)
let fixed_index_into_param () =
  Stdio.printf "Testing fixed-index axes consumed by a parameter\n";
  Tensor.unsafe_reinitialize ();
  try
    let x = TDSL.range_of_shape ~batch_dims:[ 2 ] ~output_dims:[ 3; 3; 4 ] () in
    let%op pooled = x ++ "... | h, w, ..c.. => ... | 0, 0, ..c.." in
    let%op logits = ({ w } * pooled) + { b = 0.; o = [ 5 ] } in
    let _ctx : Context.t = Train.init_params (Context.auto ()) Train.IDX.empty logits in
    Verdict.fail "should have raised an exception"
  with Row.Shape_error (msg, _) -> Stdio.printf "Got expected error: %s\n" msg

(* The initializer fills the parameter; only its uses may size it. Each refusal is checked at
   closure, including the direct and wrapped variants of the same declaration. *)
let parameter init ?input_dims ?output_dims () =
  let param_init =
    match init with
    | "default" -> None
    | "uniform1" -> Some (NTDSL.uniform1 ())
    | "normal" -> Some (NTDSL.normal ())
    | "zero" -> Some (Tensor.term_init ~grad_spec:Prohibit_grad [| 0. |])
    | _ -> assert false
  in
  TDSL.param "p" ?param_init ?input_dims ?output_dims ()

let initializers = [ "default"; "uniform1"; "normal"; "zero" ]

let parameter_use_refusals () =
  List.iter initializers ~f:(fun init ->
      List.iter
        [ "alone"; "neg"; "constant"; "empty output use"; "fixed index"; "wrapped fixed index" ]
        ~f:(fun use ->
          let label = init ^ " parameter: " ^ use in
          Verdict.case label (fun () ->
              Tensor.unsafe_reinitialize ();
              let refused =
                try
                  let p = parameter init () in
                  let root =
                    match use with
                    | "alone" -> p
                    | "neg" -> TDSL.neg p ()
                    | "constant" ->
                        TDSL.add
                          (TDSL.range_of_shape ~batch_dims:[ 5 ] ~input_dims:[ 3 ] ~output_dims:[]
                             ())
                          p ()
                    | "empty output use" ->
                        TDSL.matmul (TDSL.range_of_shape ~input_dims:[] ~output_dims:[ 5 ] ()) p ()
                    | "fixed index" -> TDSL.einsum1 "...|3 => ...|0" (TDSL.pointmul p p ()) ()
                    | "wrapped fixed index" ->
                        TDSL.einsum1 "...|3 => ...|0" (TDSL.neg (TDSL.pointmul p p ()) ()) ()
                    | _ -> assert false
                  in
                  ignore (Shape.to_dims root.Tensor.shape : int array);
                  false
                with Row.Shape_error (msg, _) ->
                  Stdio.eprintf "%s: %s (not part of the golden)\n" label msg;
                  true
              in
              p label refused)))

let parameter_use_acceptances () =
  List.iter initializers ~f:(fun init ->
      List.iter [ "sized"; "explicit unit axis"; "omitted row"; "schemed omitted row" ]
        ~f:(fun use ->
          let label = init ^ " parameter: " ^ use in
          Verdict.case label (fun () ->
              Tensor.unsafe_reinitialize ();
              let param, root, want =
                match use with
                | "explicit unit axis" ->
                    let p = parameter init ~input_dims:[] ~output_dims:[ 1 ] () in
                    (p, p, [| 1 |])
                | "sized" ->
                    let p = parameter init () in
                    let c = TDSL.range_of_shape ~input_dims:[] ~output_dims:[ 3 ] () in
                    (p, TDSL.add c p (), [| 3 |])
                | "omitted row" ->
                    let p = parameter init () in
                    let root = TDSL.einsum1 "i => i" p ~output_dims:[ 3 ] () in
                    (p, root, [| 3 |])
                | "schemed omitted row" ->
                    let%op omitted () x = x ++ "i => i" in
                    let f = omitted () in
                    let p = parameter init () in
                    let c = TDSL.range_of_shape ~input_dims:[] ~output_dims:[ 3 ] () in
                    (p, TDSL.add c (f p) (), [| 3 |])
                | _ -> assert false
              in
              ignore (Shape.to_dims root.Tensor.shape : int array);
              p label (Array.equal Int.equal want (Shape.to_dims param.Tensor.shape)))))

let () =
  default_lone_param ();
  lone_param_1d ();
  default_linear_op ();
  default_affine_op_propagated ();
  default_affine_op_unknown_input ();
  default_bias_param ();
  default_bias_param_1d ();
  fixed_index_into_param ();
  parameter_use_refusals ();
  parameter_use_acceptances ()
