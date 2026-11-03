(* gh-ocannl-1820: a shape-scheme trace solves symbolically what inline construction solves
   numerically. cifar_resnet's downsampling residual block (bench_conv_model.ml's [resnet]: a valid
   2x2 stride-2 pool on the main path, a padded stride-2 1x1 shortcut, a residual add) relates the
   pool's and the shortcut's output widths through [Row.unify_dim]'s mixed padded/valid affine arm,
   which reduces the pair to a stride-1 valid window of kernel 1 -- the identity. On a tracer's
   unsolved width that identity met the residual add's equation as [o = conv(o+1)], and the occurs
   check refused it, so the default [op_shape_schemes=check] killed the program where [off] ran.
   Pinned here: the schemed block traces and checks, agrees with [off] on shapes, values and
   gradients, and the solver's identity collapse holds without weakening the occurs check. *)

(* cc-fp-contract: same-shape -- check and off build the same graph, so both arms compile the same
   kernel code and contract alike *)

open Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module Row = Ocannl.Row
module Nn_blocks = Ocannl.Nn_blocks

let check_mode = Some `Check
let () = Tensor.For_testing.set_op_shape_schemes check_mode

let outcome f =
  match f () with
  | x -> Ok x
  | exception Row.Shape_error (msg, _) -> Error msg
  | exception Tensor.Session_error (msg, _) -> Error msg

let events f =
  let log = ref [] in
  let result = Tensor.For_testing.observe_schemes f ~on_event:(fun e -> log := e :: !log) in
  (result, List.rev !log)

let traced name = function Tensor.Scheme.Traced n -> String.equal n name | _ -> false
let checked name = function Tensor.Scheme.Checked n -> String.equal n name | _ -> false
let unschemed name = function Tensor.Scheme.Unschemed (n, _) -> String.equal n name | _ -> false

(* The s2 block of [Bench_conv_model.resnet]. The residual [+] evaluates its shortcut first, as the
   benchmark's [residual] does. *)
let make_s2_block ~name =
  let pool = Nn_blocks.max_pool2d ~stride:2 ~window_size:2 () in
  let conv ~part ?(kernel_size = 3) ?(stride = 1) () =
    Nn_blocks.conv2d
      ~label:[ name ^ part ]
      ~kernel_size ~stride ~use_padding:true ~out_channels:4 ()
  in
  let c1 = conv ~part:"c1" () and c2 = conv ~part:"c2" () in
  let sc = conv ~part:"sc" ~kernel_size:1 ~stride:2 () in
  fun x -> TDSL.O.(relu (c2 (relu (c1 (pool x))) + sc x))

(* The networks under test are [%op] functions without a [()] stage, as the benchmark's [model] is:
   each traces at its first application, on a tracer whose spatial widths are unsolved. One block,
   and two stacked (cifar_resnet stacks three): the second block composes the first's widths into
   the mixed padded/valid arm again. *)
let make_net ~depth =
  let b1 = make_s2_block ~name:"s2" in
  if depth = 1 then
    let%op s2_block x = b1 x in
    ("s2_block", s2_block)
  else
    let b2 = make_s2_block ~name:"s3" in
    let%op s2_s3_blocks x = b2 (b1 x) in
    ("s2_s3_blocks", s2_s3_blocks)

(* Non-square, batch 2, 3 channels: 8x12 for one block, 16x24 for two. *)
let input ~depth =
  let k = if depth = 1 then 1 else 2 in
  TDSL.range_of_shape ~label:[ "input" ] ~batch_dims:[ 2 ] ~output_dims:[ 8 * k; 12 * k; 3 ] ()

let run ~depth ~mode =
  Tensor.For_testing.set_op_shape_schemes (Some mode);
  Tensor.unsafe_reinitialize ();
  Exn.protect
    ~finally:(fun () -> Tensor.For_testing.set_op_shape_schemes check_mode)
    ~f:(fun () ->
      let _, net = make_net ~depth in
      let x = input ~depth in
      let built, log = events (fun () -> outcome (fun () -> net x)) in
      ( log,
        Result.map built ~f:(fun out ->
            let%op loss = out ++ "...|... => 0" in
            Set.iter loss.Tensor.params ~f:(fun t ->
                Option.iter t.Tensor.diff ~f:(fun d -> Ocannl.Train.set_materialized d.Tensor.grad));
            let ctx = Ocannl.Train.update_once (Context.auto ()) loss in
            let params =
              List.sort (Set.to_list loss.Tensor.params) ~compare:(fun a b ->
                  String.compare
                    (Ir.Tnode.label_name a.Tensor.value)
                    (Ir.Tnode.label_name b.Tensor.value))
            in
            let values t = Array.to_list (Context.get_values ctx t.Tensor.value) in
            let grads t =
              Option.value_map t.Tensor.diff ~default:[] ~f:(fun d ->
                  Array.to_list (Context.get_values ctx d.Tensor.grad))
            in
            ( Ir.Tnode.dims_to_string out.Tensor.value,
              values out :: values loss
              :: List.concat_map params ~f:(fun t -> [ values t; grads t ]) )) ))

let () =
  List.iter [ 1; 2 ] ~f:(fun depth ->
      let name = fst (make_net ~depth) in
      Verdict.case (name ^ ": traces, checks and agrees with off") (fun () ->
          let log_check, check = run ~depth ~mode:`Check in
          let log_off, off = run ~depth ~mode:`Off in
          (match check with
          | Error msg -> Stdio.eprintf "(not part of the golden) check-mode error: %s\n%!" msg
          | Ok _ -> ());
          p "under check, the network builds without a shape error" (Result.is_ok check);
          p "control (off): the network builds" (Result.is_ok off);
          p "its scheme is traced" (List.count log_check ~f:(traced name) = 1);
          p "and its application is checked against the scheme"
            (List.count log_check ~f:(checked name) = 1);
          p_none "no scheme event leaves it unschemed" log_check ~f:(unschemed name);
          p_empty "control (off): no scheme event, where check had some" ~over:log_check log_off;
          match (check, off) with
          | Ok (dims_check, vals_check), Ok (dims_off, vals_off) ->
              Stdio.printf "output dims: %s\n" dims_off;
              Stdio.eprintf "(not part of the golden) loss under check: %s, under off: %s\n%!"
                (String.concat ~sep:" " (List.map (List.nth_exn vals_check 1) ~f:Float.to_string))
                (String.concat ~sep:" " (List.map (List.nth_exn vals_off 1) ~f:Float.to_string));
              p "check and off give the same output dims" (String.equal dims_check dims_off);
              p "check and off give the same output, loss, parameters and gradients, bitwise"
                (List.equal (List.equal Float.equal) vals_check vals_off)
          | _ -> ()));
  (* A genuine self-reference stays loud: a valid stride-1 3x3 conv residual-added to its own input
     relates [w = o + 2] and [o = w], which no width satisfies. *)
  Verdict.case "a valid 3x3 conv added to its own input is still refused" (fun () ->
      Tensor.unsafe_reinitialize ();
      let c =
        Nn_blocks.conv2d ~label:[ "valid3" ] ~kernel_size:3 ~use_padding:false ~out_channels:3 ()
      in
      let%op self_res x = c x + x in
      p "the scheme or the body raises a shape error"
        (Result.is_error (outcome (fun () -> self_res (input ~depth:1)))))

(* {2 The solver's identity collapse, directly} *)

let origin : Row.constraint_origin list =
  [
    {
      lhs_name = "test";
      lhs_kind = `Output;
      rhs_name = "test";
      rhs_kind = `Output;
      operation = None;
    };
  ]

let dim d = Row.get_dim ~d ~basis:Row.default_basis ()
let eq d1 d2 : Row.constraint_ = Row.Dim_eq { d1; d2; origin }

let solve ?(env = Row.empty_env) cs =
  match Row.solve_inequalities ~stage:Stage4 cs env with
  | _, env -> Ok env
  | exception Row.Shape_error (msg, _) -> Error msg

let valid ~stride ?(dilation = 1) ~kernel over =
  Row.Affine
    {
      stride;
      over;
      conv = Some { dilation; kernel = dim kernel; use_padding = false };
      stride_offset = 0;
    }

let padded ~stride over =
  Row.Affine
    {
      stride;
      over;
      conv = Some { dilation = 1; kernel = dim 1; use_padding = true };
      stride_offset = 0;
    }

let () =
  Verdict.case "a stride-1 valid window of span 1 is the identity" (fun () ->
      let v = Row.get_var () in
      p "v = conv(v + kernel 1)"
        (Result.is_ok (solve [ eq (Var v) (valid ~stride:1 ~kernel:1 (Var v)) ]));
      p "conv(v + kernel 1) = v"
        (Result.is_ok (solve [ eq (valid ~stride:1 ~kernel:1 (Var v)) (Var v) ]));
      p "v = conv(v + dilation 3, kernel 1)"
        (Result.is_ok (solve [ eq (Var v) (valid ~stride:1 ~dilation:3 ~kernel:1 (Var v)) ]));
      p "control: v = conv(v + kernel 3) is still refused"
        (Result.is_error (solve [ eq (Var v) (valid ~stride:1 ~kernel:3 (Var v)) ]));
      p "control: v = conv(v + dilation 2, kernel 2) is still refused"
        (Result.is_error (solve [ eq (Var v) (valid ~stride:1 ~dilation:2 ~kernel:2 (Var v)) ]));
      (* Only the self-reference is the tautology: a window over a solved extent still derives its
         basis from the kernel's, so [bcast_if_1] over a [default] kernel reads as [default]. *)
      let bcast_4 = Row.get_dim ~d:4 ~basis:Row.bcast_if_1 () in
      p "a solved window still merges its kernel's basis: default 4 = conv(bcast 4 + kernel 1)"
        (Result.is_ok (solve [ eq (dim 4) (valid ~stride:1 ~kernel:1 bcast_4) ])));
  Verdict.case "the pool/shortcut relation solved generically, then concretely" (fun () ->
      (* [w = 2*o_sc] (padded stride 2) and [w = 2*(o_p - 1) + 2] (valid stride-2 2x2 pool). *)
      let pair_then ~first ~then_ =
        let w = Row.get_var () and o_sc = Row.get_var () and o_p = Row.get_var () in
        let rel =
          [
            eq (Var w) (padded ~stride:2 (Var o_sc));
            eq (Var w) (valid ~stride:2 ~kernel:2 (Var o_p));
          ]
        in
        let rel = if first then rel else List.rev rel in
        Result.bind (solve rel) ~f:(fun env -> solve ~env (then_ ~w ~o_sc ~o_p))
        |> Result.map ~f:(fun env -> (Row.get_dim_val env o_sc, Row.get_dim_val env o_p))
      in
      List.iter [ true; false ] ~f:(fun first ->
          let order = if first then "shortcut first" else "pool first" in
          p
            (order ^ ": the residual equation o_p = o_sc holds on unsolved widths")
            (Result.is_ok
               (pair_then ~first ~then_:(fun ~w:_ ~o_sc ~o_p -> [ eq (Var o_p) (Var o_sc) ])));
          p
            (order ^ ": then w = 12 gives both outputs 6, as solving concretely does")
            (match pair_then ~first ~then_:(fun ~w ~o_sc:_ ~o_p:_ -> [ eq (Var w) (dim 12) ]) with
            | Ok (Some 6, Some 6) -> true
            | _ -> false);
          p
            (order ^ ": control: different concrete outputs are refused")
            (Result.is_error
               (pair_then ~first ~then_:(fun ~w:_ ~o_sc ~o_p ->
                    [ eq (Var o_sc) (dim 6); eq (Var o_p) (dim 7) ])))))
