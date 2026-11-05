(* gh-ocannl-1820: a shape-scheme trace solves symbolically what inline construction solves
   numerically. cifar_resnet's downsampling residual block (bench_conv_model.ml's [resnet]: a valid
   2x2 stride-2 pool on the main path, a padded stride-2 1x1 shortcut, a residual add) relates the
   pool's and the shortcut's output widths through [Row.unify_dim]'s mixed padded/valid affine arm,
   which reduces the pair to a stride-1 valid window of kernel 1 -- the identity. On a tracer's
   unsolved width that identity met the residual add's equation as [o = conv(o+1)], and the occurs
   check refused it, so the default [op_shape_schemes=check] killed the program where [off] ran.
   Pinned here: the schemed block traces and checks, agrees with [off] on shapes, values and
   gradients, and the solver's identity collapse holds without weakening the occurs check.

   gh-ocannl-1823: that reduction read the padded side's span as 1, not its stride. Equal strides
   hid it (the truncating division happened to give the identity helper); a padded stride-4 shortcut
   against the stride-2 pool derived [o_p = 2*o_sc - 1] where concrete solving gives [2*o_sc]. The
   arm now equates the two equal-stride outputs directly. Pinned by the [s4_shortcut] network
   against [off] and by the solver claims on unequal strides at the end. *)

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

let traced name = function
  | Tensor.Scheme.Traced s -> String.equal (Tensor.Scheme.label s) name
  | _ -> false

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

(* gh-ocannl-1823: unequal strides across the mixed arm. A padded stride-4 1x1 shortcut on [x] is
   residual-added to a valid 2x2 stride-2 pool followed by a padded stride-2 conv, so the pool's
   width is twice the shortcut's: [4*o_sc = 2*(o_p - 1) + 2]. *)
let make_s4_shortcut () =
  let pool = Nn_blocks.max_pool2d ~stride:2 ~window_size:2 () in
  let conv ~part ~kernel_size ~stride =
    Nn_blocks.conv2d
      ~label:[ "s4" ^ part ]
      ~kernel_size ~stride ~use_padding:true ~out_channels:4 ()
  in
  let c = conv ~part:"c" ~kernel_size:3 ~stride:2
  and sc = conv ~part:"sc" ~kernel_size:1 ~stride:4 in
  fun x -> TDSL.O.(relu (c (pool x) + sc x))

(* The networks under test are [%op] functions without a [()] stage, as the benchmark's [model] is:
   each traces at its first application, on a tracer whose spatial widths are unsolved. One block,
   and two stacked (cifar_resnet stacks three): the second block composes the first's widths into
   the mixed padded/valid arm again. *)
let make_net net =
  let b1 = make_s2_block ~name:"s2" in
  match net with
  | `S2 ->
      let%op s2_block x = b1 x in
      ("s2_block", s2_block)
  | `S2_s3 ->
      let b2 = make_s2_block ~name:"s3" in
      let%op s2_s3_blocks x = b2 (b1 x) in
      ("s2_s3_blocks", s2_s3_blocks)
  | `S4 ->
      let b = make_s4_shortcut () in
      let%op s4_shortcut x = b x in
      ("s4_shortcut", s4_shortcut)

(* Non-square, batch 2, 3 channels: 8x12 for one block (and for the stride-4 shortcut), 16x24 for
   two. *)
let input net =
  let k = match net with `S2 | `S4 -> 1 | `S2_s3 -> 2 in
  TDSL.range_of_shape ~label:[ "input" ] ~batch_dims:[ 2 ] ~output_dims:[ 8 * k; 12 * k; 3 ] ()

let run net ~mode =
  Tensor.For_testing.set_op_shape_schemes (Some mode);
  Tensor.unsafe_reinitialize ();
  Exn.protect
    ~finally:(fun () -> Tensor.For_testing.set_op_shape_schemes check_mode)
    ~f:(fun () ->
      let _, f = make_net net in
      let x = input net in
      let built, log = events (fun () -> outcome (fun () -> f x)) in
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
  List.iter [ `S2; `S2_s3; `S4 ] ~f:(fun net ->
      let name = fst (make_net net) in
      Verdict.case (name ^ ": traces, checks and agrees with off") (fun () ->
          let log_check, check = run net ~mode:`Check in
          let log_off, off = run net ~mode:`Off in
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
        (Result.is_error (outcome (fun () -> self_res (input `S2)))))

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

(* {2 Unequal strides and spans across the mixed arm, directly (gh-ocannl-1823)} *)

let () =
  (* Two windows over one width [w], related in both orders, then a concrete [then_]. *)
  let pair_then ~first ~w_o1 ~w_o2 then_ =
    let w = Row.get_var () and o1 = Row.get_var () and o2 = Row.get_var () in
    let rel = [ eq (Var w) (w_o1 (Row.Var o1)); eq (Var w) (w_o2 (Row.Var o2)) ] in
    let rel = if first then rel else List.rev rel in
    Result.bind (solve rel) ~f:(fun env -> solve ~env (then_ ~w ~o1 ~o2))
    |> Result.map ~f:(fun env -> (Row.get_dim_val env o1, Row.get_dim_val env o2))
  in
  let both_orders ~w_o1 ~w_o2 claim =
    List.iter [ true; false ] ~f:(fun first ->
        claim
          ~order:(if first then "o1's window first" else "o2's window first")
          (pair_then ~first ~w_o1 ~w_o2))
  in
  let solves_to expected = function
    | Ok (Some a, Some b) -> List.equal Int.equal [ a; b ] expected
    | _ -> false
  in
  Verdict.case "a padded stride-4 window against a valid stride-2 window of 2: o2 = 2*o1" (fun () ->
      (* [4*o1 = 2*(o2 - 1) + 2]: the padded side's span is 4, so [o2 = 2*o1]. Read as 1, it was [o2
         = 2*o1 - 1]. *)
      both_orders ~w_o1:(padded ~stride:4) ~w_o2:(valid ~stride:2 ~kernel:2) (fun ~order solved ->
          p (order ^ ": o1 = 4 gives o2 = 8")
            (solves_to [ 4; 8 ] (solved (fun ~w:_ ~o1 ~o2:_ -> [ eq (Var o1) (dim 4) ])));
          p (order ^ ": o2 = 8 gives o1 = 4")
            (solves_to [ 4; 8 ] (solved (fun ~w:_ ~o1:_ ~o2 -> [ eq (Var o2) (dim 8) ])));
          p
            (order ^ ": w = 16 gives o1 = 4 and o2 = 8, as solving concretely does")
            (solves_to [ 4; 8 ] (solved (fun ~w ~o1:_ ~o2:_ -> [ eq (Var w) (dim 16) ])));
          p
            (order ^ ": control: o1 = 4 with o2 = 7 is refused")
            (Result.is_error
               (solved (fun ~w:_ ~o1 ~o2 -> [ eq (Var o1) (dim 4); eq (Var o2) (dim 7) ])))));
  Verdict.case "a valid stride-4 window against a padded stride-2 window: o2 = 2*o1 + 1" (fun () ->
      (* The valid side carries the larger stride: [4*(o1 - 1) + 6 = 2*o2], so [o2 = 2*o1 + 1]. A
         control: there the old reading's [(6 - 1) / 2 + 1] truncated onto the right span, 3, as it
         does whenever the padded stride divides the valid span. *)
      both_orders ~w_o1:(valid ~stride:4 ~kernel:6) ~w_o2:(padded ~stride:2) (fun ~order solved ->
          p (order ^ ": o1 = 3 gives o2 = 7")
            (solves_to [ 3; 7 ] (solved (fun ~w:_ ~o1 ~o2:_ -> [ eq (Var o1) (dim 3) ])));
          p
            (order ^ ": w = 14 gives o1 = 3 and o2 = 7")
            (solves_to [ 3; 7 ] (solved (fun ~w ~o1:_ ~o2:_ -> [ eq (Var w) (dim 14) ])))));
  Verdict.case "a pair with no integer solution is refused, not truncated" (fun () ->
      (* [2*o1 = 2*(o2 - 1) + 3]: an even width equal to an odd one. *)
      both_orders ~w_o1:(padded ~stride:2) ~w_o2:(valid ~stride:2 ~kernel:3) (fun ~order solved ->
          p
            (order ^ ": padded stride 2 against a valid stride-2 window of 3")
            (Result.is_error (solved (fun ~w:_ ~o1:_ ~o2:_ -> []))));
      (* [2*(o1 - 1) + 3 = 2*(o2 - 1) + 2], the both-valid arm's counterpart. *)
      both_orders ~w_o1:(valid ~stride:2 ~kernel:3) ~w_o2:(valid ~stride:2 ~kernel:2)
        (fun ~order solved ->
          p
            (order ^ ": valid stride-2 windows of 3 and of 2")
            (Result.is_error (solved (fun ~w:_ ~o1:_ ~o2:_ -> [])))));
  Verdict.case "controls the old reading already solved" (fun () ->
      (* Both valid, unequal strides: [4*(o1 - 1) + 4 = 2*(o2 - 1) + 2], so [o2 = 2*o1]. *)
      both_orders ~w_o1:(valid ~stride:4 ~kernel:4) ~w_o2:(valid ~stride:2 ~kernel:2)
        (fun ~order solved ->
          p
            (order ^ ": both valid: o1 = 4 gives o2 = 8")
            (solves_to [ 4; 8 ] (solved (fun ~w:_ ~o1 ~o2:_ -> [ eq (Var o1) (dim 4) ]))));
      (* Stride 1, where a padded side's span is 1 either way: [o1 = o2 + 2], a helper of span
         -1. *)
      both_orders ~w_o1:(padded ~stride:1) ~w_o2:(valid ~stride:1 ~kernel:3) (fun ~order solved ->
          p
            (order ^ ": padded stride 1 against a valid window of 3: o1 = 7 gives o2 = 5")
            (solves_to [ 7; 5 ] (solved (fun ~w:_ ~o1 ~o2:_ -> [ eq (Var o1) (dim 7) ])));
          p
            (order ^ ": and o2 = 5 gives o1 = 7")
            (solves_to [ 7; 5 ] (solved (fun ~w:_ ~o1:_ ~o2 -> [ eq (Var o2) (dim 5) ])))))
