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
   against [off] and by the solver claims on unequal strides at the end.

   gh-ocannl-1972: strides that do not divide each other (2 and 3) relate no output to the other as
   a function, and the same reduction refused them, so [check] killed a program [off] ran. The pair
   now waits for one side. Pinned by the [windows_2_3] network against [off], by [late_sized] on a
   width arriving after the windows, and by the solver claims.

   gh-ocannl-1968: the solved arms derived a window's basis three ways, so acceptance depended on
   which constraint arrived first. Pinned by the matrix at the end and the [whole_width] network,
   whose outputs wait on that rule. *)

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

(* gh-ocannl-1972: strides that do not divide each other over one input. On a tracer's unsolved
   widths the pair [2*o2 = 3*o3] relates no single output to the other: it waits for an application
   to size the input. *)
let make_windows_2_3 () =
  let conv ~stride =
    Nn_blocks.conv2d
      ~label:[ "w" ^ Int.to_string stride ]
      ~kernel_size:stride ~stride ~use_padding:false ~out_channels:4 ()
  in
  (conv ~stride:2, conv ~stride:3)

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
  | `Windows_2_3 ->
      let c2, c3 = make_windows_2_3 () in
      let%op windows_2_3 x = c2 x ++ "...|... => 0" + (c3 x ++ "...|... => 0") in
      ("windows_2_3", windows_2_3)
  | `Whole_width ->
      let c =
        Nn_blocks.conv2d ~label:[ "whole" ] ~kernel_size:8 ~use_padding:false ~out_channels:4 ()
      in
      let%op whole_width x = relu (c x) ++ "...|... => 0" in
      ("whole_width", whole_width)

(* Non-square, batch 2, 3 channels: 8x12 for one block (and for the stride-4 shortcut), 16x24 for
   two. [Whole_width]'s kernel covers its square input, so the window's outputs are the width-1 ones
   whose tag the equation leaves open (gh-ocannl-1968). *)
let input net =
  let k = match net with `S2 | `S4 | `Whole_width -> 1 | `S2_s3 -> 2 | `Windows_2_3 -> 3 in
  let width = match net with `Whole_width -> 8 | _ -> 12 * k in
  TDSL.range_of_shape ~label:[ "input" ] ~batch_dims:[ 2 ] ~output_dims:[ 8 * k; width; 3 ] ()

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
            Ocannl.Train.set_materialized out.Tensor.value;
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
  List.iter [ `S2; `S2_s3; `S4; `Windows_2_3; `Whole_width ] ~f:(fun net ->
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
      (* The kernel's numeric size is held until closure (gh-ocannl-2131), so registration may defer
         this contradiction. It must be refused before the tensor can be compiled. *)
      p "the invalid residual is refused by shape-inference closure"
        (Result.is_error
           (outcome (fun () ->
                let y = self_res (input `S2) in
                Lazy.force y.Tensor.value.dims))))

(* gh-ocannl-1972: an input whose width is not known when the windows meet it, under [off] too, so
   the application itself leaves the pair waiting. The width comes from a later operation, or never.
   The pair's value counts each window's cells of an all-ones input, so it says the sizes were
   solved, not only accepted: [width / 2 + width / 3], and the later operation's [sum (0 .. width -
   1)] on top. *)
let late_sized ~mode ~width =
  Tensor.For_testing.set_op_shape_schemes (Some mode);
  Tensor.unsafe_reinitialize ();
  Exn.protect
    ~finally:(fun () -> Tensor.For_testing.set_op_shape_schemes check_mode)
    ~f:(fun () ->
      outcome (fun () ->
          let%op pair x =
            x ++ "2*o1 => o1" ++ "...|... => 0" + (x ++ "3*o2 => o2" ++ "...|... => 0")
          in
          let x = TDSL.term ~label:[ "x" ] ~fetch_op:(Constant 1.) () in
          let paired = pair x in
          let out =
            match width with
            | None -> paired
            | Some width ->
                let y = TDSL.range_of_shape ~label:[ "y" ] ~output_dims:[ width ] () in
                let%op out = paired + ((x *. y) ++ "...|... => 0") in
                out
          in
          Ocannl.Train.set_materialized out.Tensor.value;
          let ctx = Ocannl.Train.forward_once (Context.auto ()) out in
          Array.to_list (Context.get_values ctx out.Tensor.value)))

let () =
  List.iter
    [ (`Off, "off"); (`Check, "check") ]
    ~f:(fun (mode, name) ->
      Verdict.case (name ^ ": a width arriving after both windows solves the waiting pair")
        (fun () ->
          let report label r =
            Stdio.eprintf "(not part of the golden) %s %s: %s\n%!" name label
              (match r with
              | Ok v -> String.concat ~sep:" " (List.map v ~f:Float.to_string)
              | Error msg -> "refused: " ^ msg);
            r
          in
          let got = report "width 12" (late_sized ~mode ~width:(Some 12)) in
          p "width 12 gives outputs 6 and 4, over the input and y's 66"
            (Result.equal (List.equal Float.equal) String.equal got (Ok [ 76. ]));
          let says r part =
            match r with Error msg -> String.is_substring msg ~substring:part | Ok _ -> false
          in
          let width_10 = report "width 10" (late_sized ~mode ~width:(Some 10)) in
          p "control: width 10 is no multiple of 3" (Result.is_error width_10);
          p "control: and its refusal is not the unsized pair's" (not (says width_10 "the windows"));
          let no_width = report "no width" (late_sized ~mode ~width:None) in
          p "with no width anywhere, closing refuses" (Result.is_error no_width);
          (* gh-ocannl-2040: by the windows and the widths that fit, not as a bare [mismatch]. *)
          p "the refusal names both windows" (says no_width "the windows 2*o1 and 3*o2");
          p "and the sizes closing gave them"
            (says no_width "o1 at 1 and o2 at 1, which read 2 and 3 cells");
          p "and asks for a width, naming those both windows fit"
            (says no_width "The widths both fit are 6 and every 6 after it")))

(* The width reaching the pair through one window's output instead, as a broadcast's bound, which
   the closing stages must spend before they close the other output at 1. *)
let () =
  Verdict.case "a bound on one output solves the waiting pair before the other closes" (fun () ->
      Tensor.unsafe_reinitialize ();
      let got =
        outcome (fun () ->
            let x = TDSL.term ~label:[ "x" ] ~fetch_op:(Constant 1.) () in
            let y = TDSL.range_of_shape ~label:[ "y" ] ~output_dims:[ 4 ] () in
            let%op out =
              x ++ "2*o1 => o1" ++ "...|... => 0" + (((x ++ "3*o2 => o2") *. y) ++ "...|... => 0")
            in
            Ocannl.Train.set_materialized out.Tensor.value;
            let ctx = Ocannl.Train.forward_once (Context.auto ()) out in
            Array.to_list (Context.get_values ctx out.Tensor.value))
      in
      (match got with
      | Ok v ->
          Stdio.eprintf "(not part of the golden) a bound: %s\n%!"
            (String.concat ~sep:" " (List.map v ~f:Float.to_string))
      | Error msg -> Stdio.eprintf "(not part of the golden) a bound: refused: %s\n%!" msg);
      p "y's 4 gives o2 = 4, so o1 = 6: 6 cells of ones and 0 + 1 + 2 + 3"
        (Result.equal (List.equal Float.equal) String.equal got (Ok [ 12. ])))

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

(* Solve [batches] in order, carrying the environment and whatever the solver deferred into the next
   step. Refusing and deferring are different answers, so neither is read as the other: a claim that
   a system SOLVES needs [is_solved], which a constraint still deferred at the end fails, and a
   claim that it is REFUSED needs [is_refused], which only the solver raising passes. *)
let solve_batches batches =
  let step (deferred, env) cs =
    match Row.solve_inequalities ~stage:Stage4 (deferred @ cs) env with
    | result -> Ok result
    | exception Row.Shape_error (msg, _) -> Error (`Refused msg)
  in
  List.fold batches
    ~init:(Ok ([], Row.empty_env))
    ~f:(fun acc batch -> Result.bind acc ~f:(fun acc -> step acc batch))
  |> Result.bind ~f:(function [], env -> Ok env | _ :: _, _ -> Error `Left_deferred)

let solve cs = solve_batches [ cs ]

let solve_strict ~incremental cs =
  solve_batches (if incremental then List.map cs ~f:(fun c -> [ c ]) else [ cs ])

let is_solved = Result.is_ok
let is_refused = function Error (`Refused _) -> true | _ -> false

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
        (is_solved (solve [ eq (Var v) (valid ~stride:1 ~kernel:1 (Var v)) ]));
      p "conv(v + kernel 1) = v"
        (is_solved (solve [ eq (valid ~stride:1 ~kernel:1 (Var v)) (Var v) ]));
      p "v = conv(v + dilation 3, kernel 1)"
        (is_solved (solve [ eq (Var v) (valid ~stride:1 ~dilation:3 ~kernel:1 (Var v)) ]));
      p "control: v = conv(v + kernel 3) is still refused"
        (is_refused (solve [ eq (Var v) (valid ~stride:1 ~kernel:3 (Var v)) ]));
      p "control: v = conv(v + dilation 2, kernel 2) is still refused"
        (is_refused (solve [ eq (Var v) (valid ~stride:1 ~dilation:2 ~kernel:2 (Var v)) ]));
      (* The identity holds over a solved extent too, basis included (gh-ocannl-1824): a [default]
         kernel does not restamp [bcast_if_1] as [default], which the collapse over an unsolved
         extent could never enforce. *)
      let bcast_4 = Row.get_dim ~d:4 ~basis:Row.bcast_if_1 () in
      p "a solved identity window keeps its extent's basis: bcast 4 = conv(bcast 4 + kernel 1)"
        (is_solved (solve [ eq bcast_4 (valid ~stride:1 ~kernel:1 bcast_4) ]));
      p "control: default 4 = conv(bcast 4 + kernel 1) is refused"
        (is_refused (solve [ eq (dim 4) (valid ~stride:1 ~kernel:1 bcast_4) ])));
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
        solve_batches [ rel; then_ ~w ~o_sc ~o_p ]
        |> Result.map ~f:(fun env -> (Row.get_dim_val env o_sc, Row.get_dim_val env o_p))
      in
      List.iter [ true; false ] ~f:(fun first ->
          let order = if first then "shortcut first" else "pool first" in
          (* The outputs merge, and the pair still owes its tags: a padded window passes its
             output's tag on, a valid one derives it, and they differ at [1_(bcast_if_1)]
             (gh-ocannl-2038). *)
          p
            (order
           ^ ": the residual equation o_p = o_sc holds on unsolved widths, holding only the pair")
            (let w = Row.get_var () and o_sc = Row.get_var () and o_p = Row.get_var () in
             let rel =
               [
                 eq (Var w) (padded ~stride:2 (Var o_sc));
                 eq (Var w) (valid ~stride:2 ~kernel:2 (Var o_p));
               ]
             in
             match
               Row.solve_inequalities ~stage:Stage4
                 ((if first then rel else List.rev rel) @ [ eq (Var o_p) (Var o_sc) ])
                 Row.empty_env
             with
             | [ Dim_eq { d1 = Affine _; d2 = Affine _; _ } ], env ->
                 let merged a b =
                   Option.equal Row.equal_dim (Row.get_dim_from_env env a) (Some (Var b))
                 in
                 merged o_p o_sc || merged o_sc o_p
             | _ | (exception Row.Shape_error _) -> false);
          p
            (order ^ ": then w = 12 gives both outputs 6, as solving concretely does")
            (match pair_then ~first ~then_:(fun ~w ~o_sc:_ ~o_p:_ -> [ eq (Var w) (dim 12) ]) with
            | Ok (Some 6, Some 6) -> true
            | _ -> false);
          p
            (order ^ ": control: different concrete outputs are refused")
            (is_refused
               (pair_then ~first ~then_:(fun ~w:_ ~o_sc ~o_p ->
                    [ eq (Var o_sc) (dim 6); eq (Var o_p) (dim 7) ])))))

(* {2 Unequal strides and spans across the mixed arm, directly (gh-ocannl-1823)} *)

(* Two windows over one width [w], related in both orders, then a concrete [then_]. *)
let pair_then ~first ~w_o1 ~w_o2 then_ =
  let w = Row.get_var () and o1 = Row.get_var () and o2 = Row.get_var () in
  let rel = [ eq (Var w) (w_o1 (Row.Var o1)); eq (Var w) (w_o2 (Row.Var o2)) ] in
  let rel = if first then rel else List.rev rel in
  solve_batches [ rel; then_ ~w ~o1 ~o2 ]
  |> Result.map ~f:(fun env -> (Row.get_dim_val env o1, Row.get_dim_val env o2))

let both_orders ~w_o1 ~w_o2 claim =
  List.iter [ true; false ] ~f:(fun first ->
      claim
        ~order:(if first then "o1's window first" else "o2's window first")
        (pair_then ~first ~w_o1 ~w_o2))

let solves_to expected = function
  | Ok (Some a, Some b) -> List.equal Int.equal [ a; b ] expected
  | _ -> false

let () =
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
            (is_refused (solved (fun ~w:_ ~o1 ~o2 -> [ eq (Var o1) (dim 4); eq (Var o2) (dim 7) ])))));
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
            (is_refused (solved (fun ~w:_ ~o1:_ ~o2:_ -> []))));
      (* [2*(o1 - 1) + 3 = 2*(o2 - 1) + 2], the both-valid arm's counterpart. *)
      both_orders ~w_o1:(valid ~stride:2 ~kernel:3) ~w_o2:(valid ~stride:2 ~kernel:2)
        (fun ~order solved ->
          p
            (order ^ ": valid stride-2 windows of 3 and of 2")
            (is_refused (solved (fun ~w:_ ~o1:_ ~o2:_ -> [])))));
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

(* {2 Strides that do not divide each other, directly (gh-ocannl-1972)} *)

let sym_dim () =
  Row.get_sym_dim (fst (Ocannl.Train.IDX.get_static_symbol ~static_range:12 Ocannl.Train.IDX.empty))

let () =
  Verdict.case "windows of strides 2 and 3 wait for one side, which decides the pair" (fun () ->
      let v ~stride ~kernel over = valid ~stride ~kernel over in
      let cases =
        [
          (* [2*o1 = 3*o2] *)
          ("padded 2, padded 3", padded ~stride:2, padded ~stride:3, 12, 6, 4);
          (* [2*(o1 - 1) + 2 = 3*(o2 - 1) + 3], the same equation. *)
          ("valid 2 of 2, valid 3 of 3", v ~stride:2 ~kernel:2, v ~stride:3 ~kernel:3, 12, 6, 4);
          (* [2*o1 = 3*(o2 - 1) + 5] *)
          ("padded 2, valid 3 of 5", padded ~stride:2, v ~stride:3 ~kernel:5, 14, 7, 4);
          (* [2*(o1 - 1) + 3 = 3*(o2 - 1) + 4] *)
          ("valid 2 of 3, valid 3 of 4", v ~stride:2 ~kernel:3, v ~stride:3 ~kernel:4, 13, 6, 4);
          (* Common divisor 2: [4*(o1 - 1) + 4 = 6*(o2 - 1) + 2]. *)
          ("valid 4 of 4, valid 6 of 2", v ~stride:4 ~kernel:4, v ~stride:6 ~kernel:2, 20, 5, 4);
        ]
      in
      List.iter cases ~f:(fun (name, w_o1, w_o2, width, o1_val, o2_val) ->
          both_orders ~w_o1 ~w_o2 (fun ~order solved ->
              let claim text = name ^ ", " ^ order ^ ": " ^ text in
              p
                (claim "with nothing solved the pair stays deferred, neither solved nor refused")
                (let held = solved (fun ~w:_ ~o1:_ ~o2:_ -> []) in
                 (not (is_solved held)) && not (is_refused held));
              p
                (claim "the width gives both outputs")
                (solves_to [ o1_val; o2_val ]
                   (solved (fun ~w ~o1:_ ~o2:_ -> [ eq (Var w) (dim width) ])));
              p
                (claim "one output gives the other")
                (solves_to [ o1_val; o2_val ]
                   (solved (fun ~w:_ ~o1 ~o2:_ -> [ eq (Var o1) (dim o1_val) ])));
              p (claim "and the other way round")
                (solves_to [ o1_val; o2_val ]
                   (solved (fun ~w:_ ~o1:_ ~o2 -> [ eq (Var o2) (dim o2_val) ])));
              p
                (claim "control: outputs of different widths are refused")
                (is_refused
                   (solved (fun ~w:_ ~o1 ~o2 ->
                        [ eq (Var o1) (dim o1_val); eq (Var o2) (dim (o2_val + 1)) ]))))));
  Verdict.case "a solved side no other side can match is refused when it arrives" (fun () ->
      both_orders ~w_o1:(padded ~stride:2) ~w_o2:(padded ~stride:3) (fun ~order solved ->
          p
            (order ^ ": [2*5 = 3*o2]: 10 is no multiple of 3")
            (is_refused (solved (fun ~w:_ ~o1 ~o2:_ -> [ eq (Var o1) (dim 5) ])));
          p
            (order ^ ": w = 10 is no multiple of 3 either")
            (is_refused (solved (fun ~w ~o1:_ ~o2:_ -> [ eq (Var w) (dim 10) ])))));
  Verdict.case "strides with a common divisor the spans do not share are refused at once" (fun () ->
      (* [4*(o1 - 1) + 3 = 6*(o2 - 1) + 2]: an odd width equal to an even one, whatever the sizes.
         Neither stride divides the other, so this is the case waiting would only postpone. *)
      both_orders ~w_o1:(valid ~stride:4 ~kernel:3) ~w_o2:(valid ~stride:6 ~kernel:2)
        (fun ~order solved ->
          p
            (order ^ ": valid 4 of 3, valid 6 of 2")
            (is_refused (solved (fun ~w:_ ~o1:_ ~o2:_ -> [])))));
  Verdict.case "a pair on two rigid symbols is refused: nothing could solve a side" (fun () ->
      p "2*n = 3*m"
        (is_refused (solve [ eq (padded ~stride:2 (sym_dim ())) (padded ~stride:3 (sym_dim ())) ]));
      let v = Row.get_var () in
      let held = solve [ eq (padded ~stride:2 (sym_dim ())) (padded ~stride:3 (Var v)) ] in
      p "control: a symbol against an unsolved output waits"
        ((not (is_solved held)) && not (is_refused held)));
  Verdict.case "a helper window's output is at least 1, as the concrete window demands" (fun () ->
      (* [2*o1 = 2*(o2 - 1) + 6]: a helper of span -1, [o2 = o1 - 2]. *)
      both_orders ~w_o1:(padded ~stride:2) ~w_o2:(valid ~stride:2 ~kernel:6) (fun ~order solved ->
          p (order ^ ": o1 = 3 gives o2 = 1")
            (solves_to [ 3; 1 ] (solved (fun ~w:_ ~o1 ~o2:_ -> [ eq (Var o1) (dim 3) ])));
          List.iter [ 1; 2 ] ~f:(fun o1_val ->
              let msg =
                Printf.sprintf ": o1 = %d is refused: width %d cannot hold 6" o1_val (2 * o1_val)
              in
              p (order ^ msg)
                (is_refused (solved (fun ~w:_ ~o1 ~o2:_ -> [ eq (Var o1) (dim o1_val) ])));
              p
                (order ^ msg ^ ", solved through the width as well")
                (is_refused (solved (fun ~w ~o1:_ ~o2:_ -> [ eq (Var w) (dim (2 * o1_val)) ])))));
      (* Unequal strides, where no orientation avoids the helper: [4*(o1 - 1) + 2 = 2*(o2 - 1) + 4],
         so [o2 = 2*(o1 - 1) + 0]. *)
      both_orders ~w_o1:(valid ~stride:4 ~kernel:2) ~w_o2:(valid ~stride:2 ~kernel:4)
        (fun ~order solved ->
          p (order ^ ": o1 = 2 gives o2 = 2")
            (solves_to [ 2; 2 ] (solved (fun ~w:_ ~o1 ~o2:_ -> [ eq (Var o1) (dim 2) ])));
          p
            (order ^ ": o1 = 1 is refused: width 2 cannot hold 4")
            (is_refused (solved (fun ~w:_ ~o1 ~o2:_ -> [ eq (Var o1) (dim 1) ])));
          p (order ^ ": as w = 2 is")
            (is_refused (solved (fun ~w ~o1:_ ~o2:_ -> [ eq (Var w) (dim 2) ])))))

(* {2 An identity window's basis does not depend on constraint order (gh-ocannl-1824)}

   Over an unsolved extent, the stride-1 span-1 window collapses ([unify_dim (over, other)]), so the
   extent takes the other side's basis. Over a solved one, the solved arms used to stamp
   [merge_derived_basis over.basis kernel.basis] on it: the kernel's tag whenever the extent carried
   the other reserved one. So [conv(v + kernel 1) = bcast 4] then [v = bcast 4] solved, and the
   reverse order refused "different bases". *)

let tagged d basis = Row.get_dim ~d ~basis ()
let bcast d = tagged d Row.bcast_if_1

let identity_window ?(dilation = 1) ?(kernel = tagged 1 Row.default_basis) over =
  Row.Affine
    { stride = 1; over; conv = Some { dilation; kernel; use_padding = false }; stride_offset = 0 }

let resolves_to expected v = function
  | Ok env -> (
      match Row.get_dim_from_env env v with
      | Some (Dim { d; basis; _ }) -> (
          match expected with Row.Dim e -> d = e.d && String.equal basis e.basis | _ -> false)
      | _ -> false)
  | Error _ -> false

let rec permutations = function
  | [] -> [ [] ]
  | xs ->
      List.concat_mapi xs ~f:(fun i x ->
          let rest = List.filteri xs ~f:(fun j _ -> j <> i) in
          List.map (permutations rest) ~f:(fun perm -> x :: perm))

let () =
  Verdict.case "the issue's identity window resolves to the extent's basis in every order"
    (fun () ->
      List.iter [ true; false ] ~f:(fun window_first ->
          List.iter [ true; false ] ~f:(fun window_left ->
              List.iter [ false; true ] ~f:(fun incremental ->
                  let v = Row.get_var () in
                  let window_eq =
                    if window_left then eq (identity_window (Var v)) (bcast 4)
                    else eq (bcast 4) (identity_window (Var v))
                  in
                  let extent_eq = eq (Var v) (bcast 4) in
                  let cs =
                    if window_first then [ window_eq; extent_eq ] else [ extent_eq; window_eq ]
                  in
                  p
                    (String.concat ~sep:", "
                       [
                         (if window_first then "window first" else "extent first");
                         (if window_left then "window on the left" else "window on the right");
                         (if incremental then "one at a time" else "one batch");
                       ]
                    ^ ": v = 4 of bcast_if_1")
                    (resolves_to (bcast 4) v (solve_strict ~incremental cs))))));
  Verdict.case "every reserved tag pair, size and dilation, in both orders" (fun () ->
      let reserved = [ Row.default_basis; Row.bcast_if_1 ] in
      let other b = if String.equal b Row.bcast_if_1 then Row.default_basis else Row.bcast_if_1 in
      let cells =
        List.concat_map reserved ~f:(fun extent_basis ->
            List.concat_map reserved ~f:(fun kernel_basis ->
                List.concat_map [ 1; 4 ] ~f:(fun size ->
                    List.map [ 1; 3 ] ~f:(fun dilation ->
                        (extent_basis, kernel_basis, size, dilation)))))
      in
      let describe (eb, kb, size, dilation) =
        Printf.sprintf "extent %d of %s, kernel of %s, dilation %d" size eb kb dilation
      in
      List.iter [ true; false ] ~f:(fun extent_first ->
          let order = if extent_first then "extent first" else "window first" in
          let run ~target (eb, kb, size, dilation) =
            let v = Row.get_var () and w = Row.get_var () in
            let extent_eq = eq (Var v) (tagged size eb) in
            let window_eqs =
              [
                eq (Var w) (identity_window ~dilation ~kernel:(tagged 1 kb) (Var v));
                eq (Var w) (target ~size ~eb);
              ]
            in
            ( w,
              solve_strict ~incremental:true
                (if extent_first then extent_eq :: window_eqs else window_eqs @ [ extent_eq ]) )
          in
          p_all ~min:16 (order ^ ": the window resolves to its extent, size and basis") cells
            ~f:(fun ((eb, _, size, _) as cell) ->
              let w, result = run ~target:(fun ~size ~eb -> tagged size eb) cell in
              let ok = resolves_to (tagged size eb) w result in
              if not ok then
                Stdio.eprintf "%s: %s fails (not part of the golden)\n" order (describe cell);
              ok);
          p_all ~min:16 (order ^ ": control: the other reserved tag at the same size is refused")
            cells ~f:(fun cell ->
              let _, result = run ~target:(fun ~size ~eb -> tagged size (other eb)) cell in
              let ok = is_refused result in
              if not ok then
                Stdio.eprintf "%s: %s accepts the other tag (not part of the golden)\n" order
                  (describe cell);
              ok)));
  Verdict.case "a kernel solved later goes through substitution, in every order" (fun () ->
      (* With its kernel a variable the window is no identity yet: [unify_dim] defers it, and
         [s_dim_one] reduces it once the kernel is solved -- a fix confined to [unify_dim] fails
         here. With the extent and [w] solved first, [unify_dim] infers the kernel from the sizes
         alone; the control fails unless that arm re-checks the equation's bases. *)
      let run cs_of =
        let v = Row.get_var () and k = Row.get_var () and w = Row.get_var () in
        List.map
          (permutations (cs_of ~v ~k ~w))
          ~f:(fun named ->
            ( String.concat ~sep:"; " (List.map named ~f:fst),
              w,
              solve_strict ~incremental:true (List.map named ~f:snd) ))
      in
      let report claim ok (order, _, _) =
        if not ok then
          Stdio.eprintf "%s: fails in the order %s (not part of the golden)\n" claim order;
        ok
      in
      let window ~v ~k ~w =
        ("w = conv(v + kernel k)", eq (Var w) (identity_window ~kernel:(Var k) (Var v)))
      in
      let kernel ~k = ("k = 1", eq (Var k) (dim 1))
      and extent ~v = ("v = bcast 4", eq (Var v) (bcast 4)) in
      let claim = "w = conv(v + kernel k), k = 1, v = 4 of bcast_if_1: w = 4 of bcast_if_1" in
      p_all ~min:6 claim
        (run (fun ~v ~k ~w -> [ window ~v ~k ~w; kernel ~k; extent ~v ]))
        ~f:(fun ((_, w, result) as run) -> report claim (resolves_to (bcast 4) w result) run);
      let claim = "control: and w = 4 of default besides is refused" in
      p_all ~min:24 claim
        (run (fun ~v ~k ~w ->
             [ window ~v ~k ~w; kernel ~k; extent ~v; ("w = 4", eq (Var w) (dim 4)) ]))
        ~f:(fun ((_, _, result) as run) -> report claim (is_refused result) run));
  Verdict.case "controls: what the identity does not cover is unchanged" (fun () ->
      let named_kernel = tagged 1 "h" in
      p "a named span-1 kernel still converts: 4 of h = conv(default 4 + kernel 1 of h)"
        (is_solved (solve [ eq (tagged 4 "h") (identity_window ~kernel:named_kernel (dim 4)) ]));
      p "and default 4 = conv(default 4 + kernel 1 of h) is refused"
        (is_refused (solve [ eq (dim 4) (identity_window ~kernel:named_kernel (dim 4)) ]));
      p
        "a size-changing window still derives the kernel's tag: default 3 = conv(bcast 1 + kernel \
         3)"
        (is_solved (solve [ eq (dim 3) (valid ~stride:1 ~kernel:3 (bcast 1)) ]));
      p "a padded stride-1 window over bcast 4 is bcast 4"
        (is_solved (solve [ eq (bcast 4) (padded ~stride:1 (bcast 4)) ]));
      List.iter [ true; false ] ~f:(fun window_first ->
          let v = Row.get_var () in
          let cs = [ eq (padded ~stride:1 (Var v)) (bcast 4); eq (Var v) (bcast 4) ] in
          p
            ((if window_first then "window first" else "extent first")
            ^ ": a padded stride-1 window resolves v = 4 of bcast_if_1")
            (resolves_to (bcast 4) v
               (solve_strict ~incremental:true (if window_first then cs else List.rev cs))));
      (* The one claim on a deferred result, which is the point of it: what the solver holds back is
         neither of the two answers the claims above read. *)
      let v = Row.get_var () and k = Row.get_var () and w = Row.get_var () in
      let held = solve [ eq (Var w) (identity_window ~kernel:(Var k) (Var v)) ] in
      p "a window with nothing solved stays deferred: neither solved nor refused"
        ((not (is_solved held)) && not (is_refused held));
      let v = Row.get_var () in
      p "a different size is refused: v = bcast 4, conv(v + kernel 1) = bcast 5"
        (is_refused
           (solve_strict ~incremental:true
              [ eq (Var v) (bcast 4); eq (identity_window (Var v)) (bcast 5) ])))

(* {2 A window's basis does not depend on constraint order (gh-ocannl-1968)}

   [staged] drives a system the way construction and closure do: each constraint arrives at Stage1,
   what was deferred stays pending, and the closing stages run last. A valid window derives its
   input's basis from its extent's and its kernel's. That rule is not injective, so an output or a
   kernel the equation leaves open waits for another constraint, and takes [default] once none will
   come. [~close] appends one output row over the given variables, the [Shape_row] a tensor's shape
   contributes, so closure sizes them together. *)

let stages_closing = [ Row.Stage2; Stage3; Stage4; Stage5; Stage6a; Stage6b; Stage7 ]

let staged ?(closing = stages_closing) ?(close = []) cs =
  let cs =
    if List.is_empty close then cs
    else
      let row : Row.t =
        {
          beg_dims = [];
          dims = List.map close ~f:(fun v -> Row.Var v);
          bcast = Broadcastable;
          prov = Row.provenance ~sh_id:2038 ~kind:`Output;
        }
      in
      cs @ [ Row.Shape_row (row, origin) ]
  in
  let step stage acc cs =
    Result.bind acc ~f:(fun (deferred, env) ->
        match Row.solve_inequalities ~stage (deferred @ cs) env with
        | result -> Ok result
        | exception Row.Shape_error (msg, _) -> Error (`Refused msg))
  in
  let arrived =
    List.fold cs ~init:(Ok ([], Row.empty_env)) ~f:(fun acc c -> step Stage1 acc [ c ])
  in
  List.fold closing ~init:arrived ~f:(fun acc stage -> step stage acc [])

let finished = Result.bind ~f:(function [], env -> Ok env | _ :: _, _ -> Error `Left_deferred)

let window ?(stride = 1) ?(dilation = 1) ~kernel over =
  Row.Affine
    { stride; over; conv = Some { dilation; kernel; use_padding = false }; stride_offset = 0 }

let padded_window ~stride ~kernel over =
  Row.Affine
    { stride; over; conv = Some { dilation = 1; kernel; use_padding = true }; stride_offset = 0 }

let tag_of env v =
  match Row.get_dim_from_env env v with Some (Dim { d; basis; _ }) -> Some (d, basis) | _ -> None

let solves_tag (v, d, basis) = function
  | Error _ -> false
  | Ok env -> (
      match tag_of env v with Some (got, b) -> got = d && String.equal b basis | None -> false)

(* The rule, stated apart from the solver so that two orders cannot agree on a wrong answer. *)
let rule ~o:(ob, od) ~k:(kb, kd) =
  let free basis d =
    String.equal basis Row.default_basis || (String.equal basis Row.bcast_if_1 && d = 1)
  in
  match (free ob od, free kb kd) with
  | _ when String.equal ob kb -> Some ob
  | true, true -> Some Row.default_basis
  | true, false -> Some kb
  | false, true -> Some ob
  | false, false -> None

let tags = [ Row.default_basis; Row.bcast_if_1; "h"; "j" ]

let tag_triples =
  List.concat_map tags ~f:(fun o ->
      List.concat_map tags ~f:(fun k -> List.map tags ~f:(fun i -> (o, k, i))))

let () =
  Verdict.case "every tag triple, window shape and constraint order agrees with the rule" (fun () ->
      (* stride, dilation, kernel size, output size. The first is the issue's: no output but 1 is
         contested at a reserved kernel. *)
      let shapes =
        [
          (1, 1, 3, 1);
          (2, 1, 1, 1);
          (2, 1, 1, 3);
          (1, 1, 3, 4);
          (1, 2, 2, 1);
          (2, 1, 3, 4);
          (1, 1, 1, 4);
          (1, 1, 1, 1);
        ]
      in
      let cells = List.cartesian_product shapes tag_triples in
      let orders = permutations [ 0; 1; 2; 3 ] in
      let agrees (((stride, dilation, ksize, osize), (ob, kb, ib)) as _cell) order =
        let isize = (stride * (osize - 1)) + (dilation * (ksize - 1)) + 1 in
        let v = Row.get_var () and k = Row.get_var () and w = Row.get_var () in
        let cs =
          [|
            eq (Var w) (window ~stride ~dilation ~kernel:(Var k) (Var v));
            eq (Var v) (tagged osize ob);
            eq (Var k) (tagged ksize kb);
            eq (tagged isize ib) (Var w);
          |]
        in
        let result = finished (staged (List.map order ~f:(Array.get cs))) in
        let is_identity =
          stride = 1
          && (dilation * (ksize - 1)) + 1 = 1
          && (String.equal kb Row.default_basis || String.equal kb Row.bcast_if_1)
        in
        let expected =
          if is_identity then String.equal ob ib
          else Option.equal String.equal (rule ~o:(ob, osize) ~k:(kb, ksize)) (Some ib)
        in
        let ok =
          if expected then
            solves_tag (v, osize, ob) result
            && solves_tag (k, ksize, kb) result
            && solves_tag (w, isize, ib) result
          else is_refused result
        in
        if not ok then
          Stdio.eprintf
            "stride %d dilation %d kernel %d of %s output %d of %s input %d of %s, order %s: %s \
             (not part of the golden)\n"
            stride dilation ksize kb osize ob isize ib
            (String.concat ~sep:"" (List.map order ~f:Int.to_string))
            (if expected then "should solve" else "should be refused");
        ok
      in
      p_all ~min:512 "the output, kernel and input keep their tags or the system is refused" cells
        ~f:(fun cell -> List.for_all orders ~f:(agrees cell)));
  Verdict.case "a padded window keeps its extent's basis, whatever the kernel" (fun () ->
      let cells = List.cartesian_product tags tags in
      let orders = permutations [ 0; 1; 2 ] in
      p_all ~min:16 "the input is the extent's, in every order" cells ~f:(fun (ob, kb) ->
          List.for_all orders ~f:(fun order ->
              let v = Row.get_var () and k = Row.get_var () and w = Row.get_var () in
              let cs =
                [|
                  eq (Var w) (padded_window ~stride:2 ~kernel:(Var k) (Var v));
                  eq (Var v) (tagged 4 ob);
                  eq (Var k) (tagged 3 kb);
                |]
              in
              solves_tag (w, 8, ob) (finished (staged (List.map order ~f:(Array.get cs))))));
      p "and agrees with solving the same window over solved dims in place"
        (is_solved
           (finished
              (staged [ eq (dim 8) (padded_window ~stride:2 ~kernel:(tagged 3 "h") (dim 4)) ])));
      let v = Row.get_var () and k = Row.get_var () and w = Row.get_var () in
      p "a padded window does not wait for its kernel"
        (solves_tag (w, 8, Row.default_basis)
           (Result.map ~f:snd
              (staged ~closing:[]
                 [
                   eq (Var w) (padded_window ~stride:2 ~kernel:(Var k) (Var v)); eq (Var v) (dim 4);
                 ]))));
  Verdict.case "an open output waits, then settles on default" (fun () ->
      let conv3 v = eq (window ~kernel:(dim 3) (Var v)) (dim 3) in
      let v = Row.get_var () in
      let held = staged ~closing:[] [ conv3 v ] in
      (match held with
      | Ok (deferred, env) ->
          p "with nothing else about it, conv3(v) = 3 holds the equation back"
            (List.length deferred = 1);
          p "and binds nothing" (Option.is_none (tag_of env v));
          let again, env = Row.solve_inequalities ~stage:Stage1 deferred env in
          p "retried, it is still held and binds nothing"
            (List.length again = 1 && Option.is_none (tag_of env v));
          let rest, env =
            Row.solve_inequalities ~stage:Stage1 (again @ [ eq (Var v) (bcast 1) ]) env
          in
          p "a later constraint decides it" (is_solved (finished (Ok (rest, env))));
          p "and keeps its tag" (solves_tag (v, 1, Row.bcast_if_1) (Ok env))
      | Error _ -> p "with nothing else about it, conv3(v) = 3 holds the equation back" false);
      let v = Row.get_var () in
      p "with nothing else at all, closing settles it on default"
        (solves_tag (v, 1, Row.default_basis) (finished (staged [ conv3 v ])));
      let v = Row.get_var () in
      p "control, the gate against over-deferral: a default output above 1 solves at Stage1"
        (solves_tag (v, 4, Row.default_basis)
           (Result.map ~f:snd (staged ~closing:[] [ eq (window ~kernel:(dim 3) (Var v)) (dim 6) ])));
      (* A constraint first read at the closing stage, as a trailing setter's is: settling at a
         stage fixpoint rather than in the pass lets it land first, wherever the list has it. *)
      List.iter [ true; false ] ~f:(fun window_first ->
          let v = Row.get_var () in
          let cs =
            [ eq (window ~kernel:(tagged 3 "h") (Var v)) (tagged 3 "h"); eq (Var v) (tagged 1 "h") ]
          in
          let cs = if window_first then cs else List.rev cs in
          p
            ((if window_first then "window" else "extent") ^ " first in one closing pass: v keeps h")
            (solves_tag (v, 1, "h")
               (Result.map ~f:snd
                  (match Row.solve_inequalities ~stage:Stage2 cs Row.empty_env with
                  | result -> Ok result
                  | exception Row.Shape_error (msg, _) -> Error (`Refused msg)))));
      let v = Row.get_var () in
      let w x = window ~kernel:(tagged 3 "h") x in
      p "over a nested window, settling keeps to a tag that window can derive"
        (solves_tag (v, 3, Row.default_basis)
           (finished (staged [ eq (w (w (Var v))) (tagged 7 "h") ])));
      (* A nested window waiting on its own kernel settles nothing, so closing ends and the equation
         stays with it; a settlement that binds nothing would repeat forever. *)
      let nested k v = eq (w (window ~kernel:k (Var v))) (tagged 7 "h") in
      let v = Row.get_var () and k = Row.get_var () in
      let held = staged [ nested (Var k) v ] in
      p "a nested window with an unsolved kernel stays deferred: neither solved nor refused"
        ((not (is_solved (finished held))) && not (is_refused held));
      p "and solves once the kernel arrives"
        (solves_tag (v, 3, Row.default_basis)
           (finished (staged [ nested (Var k) v; eq (Var k) (dim 3) ])));
      (* What an open output's bounds say comes with it: [3 of h] below [v] rules [default] out. *)
      List.iter [ true; false ] ~f:(fun bound_first ->
          let v = Row.get_var () in
          let bound : Row.constraint_ =
            Dim_ineq { res = tagged 3 "h"; opnd = Var v; from_ = Sexp.Atom "test"; origin }
          and open_window = eq (w (Var v)) (tagged 5 "h") in
          p
            ((if bound_first then "bound first" else "window first")
            ^ ": settling an output keeps to its bound")
            (solves_tag (v, 3, "h")
               (finished
                  (staged (if bound_first then [ bound; open_window ] else [ open_window; bound ])))));
      (* Two open windows over one output, with kernels that conflict: [default] is the choice both
         allow, in either order. *)
      List.iter [ true; false ] ~f:(fun h_first ->
          let v = Row.get_var () in
          let conv b = eq (window ~kernel:(tagged 3 b) (Var v)) (tagged 3 b) in
          let cs = if h_first then [ conv "h"; conv "j" ] else [ conv "j"; conv "h" ] in
          p
            ((if h_first then "h first" else "j first")
            ^ ": two open windows of different kernels settle on default")
            (solves_tag (v, 1, Row.default_basis) (finished (staged cs)))));
  Verdict.case "what the rule refuses or infers, by name" (fun () ->
      let v () = Row.get_var () in
      let both_ways name ~extent ~window ~input ~want =
        List.iter [ true; false ] ~f:(fun extent_first ->
            List.iter [ true; false ] ~f:(fun window_left ->
                let o = v () in
                let window_eq =
                  if window_left then eq (window (Row.Var o)) input
                  else eq input (window (Row.Var o))
                and extent_eq = eq (Var o) extent in
                p
                  (name
                  ^ (if extent_first then ", extent first" else ", window first")
                  ^ if window_left then ", window on the left" else ", window on the right")
                  (solves_tag
                     (o, fst want, snd want)
                     (finished
                        (staged
                           (if extent_first then [ extent_eq; window_eq ]
                            else [ window_eq; extent_eq ]))))))
      in
      both_ways "the issue: v = 1 of bcast_if_1, conv3(v) = 3 of default" ~extent:(bcast 1)
        ~window:(window ~kernel:(dim 3))
        ~input:(dim 3) ~want:(1, Row.bcast_if_1);
      both_ways "a named span-1 kernel: v = 4 of default, conv(v, 1 of h) = 4 of h" ~extent:(dim 4)
        ~window:(window ~kernel:(tagged 1 "h"))
        ~input:(tagged 4 "h") ~want:(4, Row.default_basis);
      p "the reverse arm refuses what forward refuses: conv(v, 3 of h) = 3 of default"
        (is_refused
           (staged [ eq (window ~kernel:(tagged 3 "h") (Var (v ()))) (dim 3) ] |> finished));
      p "a kernel closed to the top makes no claim, solved"
        (is_solved (finished (staged [ eq (window ~stride:2 ~kernel:(bcast 1) (dim 3)) (dim 5) ])));
      let o = v () in
      p "a kernel closed to the top makes no claim, reversed"
        (solves_tag (o, 3, Row.default_basis)
           (finished (staged [ eq (window ~stride:2 ~kernel:(bcast 1) (Var o)) (dim 5) ])));
      List.iter [ true; false ] ~f:(fun window_first ->
          let o = v () and k = v () in
          let cs = [ eq (window ~kernel:(Var k) (Var o)) (tagged 5 "x"); eq (Var o) (dim 2) ] in
          p
            ((if window_first then "window" else "extent")
            ^ " first: a kernel inferred under 5 of x over 2 of default is 4 of x")
            (solves_tag (k, 4, "x") (finished (staged (if window_first then cs else List.rev cs))))));
  Verdict.case "a window pair's size equation does not drop a kernel's claim" (fun () ->
      let refused_both name ~k1 ~k2 =
        List.iter [ true; false ] ~f:(fun pair_first ->
            let o1 = Row.get_var () and o2 = Row.get_var () in
            let pair = eq (window ~kernel:k1 (Var o1)) (window ~kernel:k2 (Var o2))
            and outputs = [ eq (Var o1) (dim 4); eq (Var o2) (dim 4) ] in
            p
              (name ^ if pair_first then " (pair first)" else " (outputs first)")
              (is_refused
                 (finished (staged (if pair_first then pair :: outputs else outputs @ [ pair ])))))
      in
      refused_both "kernels of h and j have no common input" ~k1:(tagged 3 "h") ~k2:(tagged 3 "j");
      refused_both "kernels of h and default disagree over a default extent" ~k1:(tagged 3 "h")
        ~k2:(dim 3);
      (* A pair held over a reduction that itself waits holds one equation, not a growing list. *)
      let x = Row.get_var () and y = Row.get_var () and k = Row.get_var () in
      let w x = window ~kernel:(tagged 3 "h") x in
      let held_pair = eq (w (window ~kernel:(Var k) (Var x))) (w (Var y)) in
      p "a held pair over an unsolved inner kernel ends: neither solved nor refused"
        (let result = staged [ held_pair ] in
         (not (is_solved (finished result))) && not (is_refused result));
      p "and solves once its sizes arrive"
        (solves_tag (y, 4, Row.default_basis)
           (finished (staged [ held_pair; eq (Var k) (dim 3); eq (Var x) (dim 2) ])));
      let o1 = Row.get_var () and o2 = Row.get_var () in
      p "control: the same named kernel solves, and the held pair is discharged"
        (is_solved
           (finished
              (staged
                 [
                   eq
                     (window ~kernel:(tagged 3 "h") (Var o1))
                     (window ~kernel:(tagged 3 "h") (Var o2));
                   eq (Var o1) (dim 4);
                 ]))))

(* {2 A window pair relates its outputs' sizes, not their tags (gh-ocannl-2038)}

   Two windows over one extent say that their inputs coincide, in size and tag. Their outputs need
   not: the basis rule is not injective, so [1 of bcast_if_1] and [1 of default] under [default]
   kernels both derive a [default] input. The pair used to equate its outputs (or one output with a
   helper window over the other) at once, which refused that pair-first and accepted it
   solved-first, and the helper's [default] kernel carried a tag across sizes the windows do not: an
   inert [2 of bcast_if_1] became the claim-free top at size 1, and a pair the windows refuse was
   accepted pair-first. The oracle is the per-window rule, stated apart from the solver. *)

type side = Valid of { stride : int; ksize : int } | Padded of { stride : int }

let side_window side ~kernel over =
  match side with
  | Valid { stride; _ } -> window ~stride ~kernel over
  | Padded { stride } -> padded_window ~stride ~kernel over

let side_ksize = function Valid { ksize; _ } -> ksize | Padded _ -> 1

(* The input a window derives from its solved output and kernel. *)
let side_input side ~o:(od, ob) ~k:(kd, kb) =
  match side with
  | Padded { stride } -> Some (stride * od, ob)
  | Valid { stride; _ } ->
      let size = (stride * (od - 1)) + kd in
      let identity =
        stride = 1 && kd = 1 && (String.equal kb Row.default_basis || String.equal kb Row.bcast_if_1)
      in
      if identity then Some (size, ob)
      else Option.map (rule ~o:(ob, od) ~k:(kb, kd)) ~f:(fun b -> (size, b))

(* Each pair with the output sizes its arithmetic relates. *)
let pair_shapes =
  [
    ( "equal valid windows",
      Valid { stride = 2; ksize = 3 },
      Valid { stride = 2; ksize = 3 },
      [ (1, 1); (3, 3) ] );
    ( "stride 2 against stride 1",
      Valid { stride = 2; ksize = 3 },
      Valid { stride = 1; ksize = 3 },
      [ (1, 1); (2, 3) ] );
    ( "padded against valid",
      Padded { stride = 2 },
      Valid { stride = 2; ksize = 2 },
      [ (1, 1); (3, 3) ] );
    ( "a helper of span 0",
      Valid { stride = 2; ksize = 1 },
      Valid { stride = 2; ksize = 3 },
      [ (2, 1); (4, 3) ] );
  ]

let kernel_tags side =
  match side with
  | Padded _ -> [ Row.default_basis ]
  | Valid { ksize = 1; _ } -> [ Row.default_basis; "h"; Row.bcast_if_1 ]
  | Valid _ -> [ Row.default_basis; "h" ]

let output_tags = [ Row.default_basis; Row.bcast_if_1; "h" ]

let pair_cells ~sizes_of =
  List.concat_map pair_shapes ~f:(fun (name, s1, s2, sizes) ->
      List.concat_map (sizes_of sizes) ~f:(fun (d1, d2) ->
          List.concat_map (kernel_tags s1) ~f:(fun k1 ->
              List.concat_map (kernel_tags s2) ~f:(fun k2 ->
                  List.concat_map output_tags ~f:(fun t1 ->
                      List.map output_tags ~f:(fun t2 -> (name, s1, s2, (d1, t1), (d2, t2), k1, k2)))))))

let consistent (_, s1, s2, o1, o2, k1, k2) =
  match
    (side_input s1 ~o:o1 ~k:(side_ksize s1, k1), side_input s2 ~o:o2 ~k:(side_ksize s2, k2))
  with
  | Some (n1, b1), Some (n2, b2) -> n1 = n2 && String.equal b1 b2
  | _ -> false

let describe (name, _, _, (d1, t1), (d2, t2), k1, k2) =
  Printf.sprintf "%s, kernels of %s and %s, o1 = %d of %s, o2 = %d of %s" name k1 k2 d1 t1 d2 t2

(* Every order of [cs], the pair written either way round. *)
let pair_orders ~pair_left ~pair_right rest =
  List.concat_map [ pair_left; pair_right ] ~f:(fun pair -> permutations (pair :: rest))

let () =
  Verdict.case "a window pair's verdict and tags do not depend on constraint order" (fun () ->
      let agrees ~late_kernels ((_, s1, s2, (d1, t1), (d2, t2), k1, k2) as cell) =
        let o1 = Row.get_var () and o2 = Row.get_var () in
        let kv1 = Row.get_var () and kv2 = Row.get_var () in
        let kernel1, kernel2 =
          if late_kernels then (Row.Var kv1, Row.Var kv2)
          else (tagged (side_ksize s1) k1, tagged (side_ksize s2) k2)
        in
        let w1 = side_window s1 ~kernel:kernel1 (Var o1)
        and w2 = side_window s2 ~kernel:kernel2 (Var o2) in
        let rest =
          [ eq (Var o1) (tagged d1 t1); eq (Var o2) (tagged d2 t2) ]
          @
          if late_kernels then
            [ eq (Var kv1) (tagged (side_ksize s1) k1); eq (Var kv2) (tagged (side_ksize s2) k2) ]
          else []
        in
        let expected = consistent cell in
        let orders = pair_orders ~pair_left:(eq w1 w2) ~pair_right:(eq w2 w1) rest in
        (not (List.is_empty orders))
        && List.for_all orders ~f:(fun order ->
            let result = finished (staged order) in
            let ok =
              if expected then solves_tag (o1, d1, t1) result && solves_tag (o2, d2, t2) result
              else is_refused result
            in
            if not ok then
              Stdio.eprintf "%s%s: %s in some order (not part of the golden)\n" (describe cell)
                (if late_kernels then ", kernels late" else "")
                (if expected then "should solve, keeping both tags" else "should be refused");
            ok)
      in
      let cells = pair_cells ~sizes_of:Fn.id in
      p_all ~min:200 "solved kernels: both outputs keep their tags, or the pair is refused" cells
        ~f:(agrees ~late_kernels:false);
      p_exists "and the matrix holds pairs the windows accept with different output tags" cells
        ~f:(fun ((_, _, _, (_, t1), (_, t2), _, _) as cell) ->
          consistent cell && not (String.equal t1 t2));
      p_all ~min:100 "kernels arriving after the pair: the same" ~f:(agrees ~late_kernels:true)
        (pair_cells ~sizes_of:(fun sizes -> [ List.hd_exn sizes ])
        |> List.filter ~f:(fun (_, s1, s2, _, _, k1, k2) ->
            (match (s1, s2) with Valid _, Valid _ -> true | _ -> false)
            && (not (String.equal k1 Row.bcast_if_1))
            && not (String.equal k2 Row.bcast_if_1)));
      (* By name: the issue's case, and the one the helper accepted in one order only. *)
      let named = pair_cells ~sizes_of:Fn.id in
      let find name' (d1', t1') (d2', t2') k =
        List.find_exn named ~f:(fun (name, _, _, (d1, t1), (d2, t2), k1, k2) ->
            String.equal name name' && d1 = d1' && d2 = d2' && String.equal t1 t1'
            && String.equal t2 t2' && String.equal k1 k && String.equal k2 Row.default_basis)
      in
      p "the issue: o1 = 1 of bcast_if_1, o2 = 1 of default under default kernels solves"
        (agrees ~late_kernels:false
           (find "equal valid windows" (1, Row.bcast_if_1) (1, Row.default_basis) Row.default_basis));
      p "a helper of span 0: o1 = 2 of bcast_if_1 (inert), o2 = 1 of bcast_if_1 is refused"
        (let cell =
           find "a helper of span 0" (2, Row.bcast_if_1) (1, Row.bcast_if_1) Row.default_basis
         in
         (not (consistent cell)) && agrees ~late_kernels:false cell);
      p "control: 1 of default = 1 of bcast_if_1 as dims is still refused"
        (is_refused (finished (staged [ eq (dim 1) (bcast 1) ]))));
  Verdict.case "a pair over a solved input: the same answer in every order" (fun () ->
      let cells =
        List.concat_map pair_shapes ~f:(fun (name, s1, s2, sizes) ->
            List.concat_map sizes ~f:(fun (d1, _) ->
                List.concat_map (kernel_tags s1) ~f:(fun k1 ->
                    List.concat_map (kernel_tags s2) ~f:(fun k2 ->
                        List.map output_tags ~f:(fun ib ->
                            let isize =
                              fst
                                (Option.value_exn
                                   (side_input s1 ~o:(d1, Row.default_basis)
                                      ~k:(side_ksize s1, Row.default_basis)))
                            in
                            (name, s1, s2, k1, k2, (isize, ib)))))))
      in
      p_all ~min:50 "one verdict, the same solved outputs, and they derive the input" cells
        ~f:(fun (name, s1, s2, k1, k2, (isize, ib)) ->
          let answers =
            List.map
              (permutations [ 0; 1; 2 ])
              ~f:(fun order ->
                let x = Row.get_var () and o1 = Row.get_var () and o2 = Row.get_var () in
                let cs =
                  [|
                    eq (Var x) (side_window s1 ~kernel:(tagged (side_ksize s1) k1) (Var o1));
                    eq (Var x) (side_window s2 ~kernel:(tagged (side_ksize s2) k2) (Var o2));
                    eq (Var x) (tagged isize ib);
                  |]
                in
                match finished (staged (List.map order ~f:(Array.get cs))) with
                | Ok env -> Some (tag_of env o1, tag_of env o2)
                | Error _ -> None)
          in
          let same = List.for_all answers ~f:(Poly.equal (List.hd_exn answers)) in
          let derives =
            match List.hd_exn answers with
            | Some (Some o1, Some o2) ->
                Poly.equal (side_input s1 ~o:o1 ~k:(side_ksize s1, k1)) (Some (isize, ib))
                && Poly.equal (side_input s2 ~o:o2 ~k:(side_ksize s2, k2)) (Some (isize, ib))
            | Some _ -> false
            | None -> true
          in
          if not (same && derives) then
            Stdio.eprintf "%s, kernels of %s and %s, input %d of %s: %s (not part of the golden)\n"
              name k1 k2 isize ib
              (if same then "outputs that do not derive the input"
               else
                 "orders disagree: "
                 ^ String.concat ~sep:" | "
                     (List.map answers ~f:(function
                       | None -> "refused"
                       | Some (a, b) ->
                           let show = function
                             | Some (d, b) -> Printf.sprintf "%d of %s" d b
                             | None -> "open"
                           in
                           show a ^ ", " ^ show b)));
          same && derives));
  Verdict.case "a pair with neither output solved" (fun () ->
      let pair o1 o2 =
        eq (window ~stride:2 ~kernel:(dim 3) (Var o1)) (window ~stride:1 ~kernel:(dim 2) (Var o2))
      in
      let o1 = Row.get_var () and o2 = Row.get_var () in
      (match staged ~closing:[] [ pair o1 o2 ] with
      | Ok (held, env) ->
          let again, env = Row.solve_inequalities ~stage:Stage1 held env in
          let again, _ = Row.solve_inequalities ~stage:Stage1 again env in
          p "retried at Stage1, the pair is one held equation, not a growing list"
            (List.length held = 1 && List.length again = 1)
      | Error _ -> p "retried at Stage1, the pair is one held equation, not a growing list" false);
      (* Closure sizes each open output on its own; [2 * (o1 - 1) + 3 = (o2 - 1) + 2] closed at 1
         and 1 is a mismatch, so the outputs are related before then. *)
      let o1 = Row.get_var () and o2 = Row.get_var () in
      let result = finished (staged ~close:[ o1; o2 ] [ pair o1 o2 ]) in
      let shown v =
        match result with
        | Ok env -> (
            match tag_of env v with Some (d, b) -> Printf.sprintf "%d of %s" d b | None -> "open")
        | Error `Left_deferred -> "left deferred"
        | Error (`Refused msg) -> "refused: " ^ msg
      in
      let ok =
        match result with
        | Ok env -> (
            match (tag_of env o1, tag_of env o2) with
            | Some (1, _), Some (2, _) -> true
            | _ -> false)
        | Error _ -> false
      in
      if not ok then
        Stdio.eprintf "closing: o1 %s, o2 %s (not part of the golden)\n" (shown o1) (shown o2);
      p "closing sizes the outputs together: o1 = 1, o2 = 2" ok;
      (* Outputs that are windows themselves: the size equation reduces to a nested pair, which
         relates [y = 2 * x] at closing as well. *)
      let x = Row.get_var () and y = Row.get_var () in
      let w s k v = window ~stride:s ~kernel:(dim k) v in
      let result =
        finished (staged ~close:[ x; y ] [ eq (w 2 3 (w 2 3 (Var x))) (w 2 3 (w 1 2 (Var y))) ])
      in
      let closes_1_2 x y = function
        | Ok env -> (
            match (tag_of env x, tag_of env y) with Some (1, _), Some (2, _) -> true | _ -> false)
        | Error _ -> false
      in
      p "a nested pair closes together too: x = 1, y = 2" (closes_1_2 x y result);
      (* And through the rewrites between: two stride-2 padded windows reduce to the identity over
         the inner pair. *)
      let x = Row.get_var () and y = Row.get_var () in
      let pad v = padded_window ~stride:2 ~kernel:(dim 1) v in
      p "a nested pair under padded windows closes together: x = 1, y = 2"
        (closes_1_2 x y
           (finished
              (staged ~close:[ x; y ]
                 [ eq (w 2 3 (pad (w 2 3 (Var x)))) (w 2 3 (pad (w 1 2 (Var y)))) ]))))
