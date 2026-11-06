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
          p
            (order ^ ": the residual equation o_p = o_sc holds on unsolved widths")
            (is_solved
               (pair_then ~first ~then_:(fun ~w:_ ~o_sc ~o_p -> [ eq (Var o_p) (Var o_sc) ])));
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

let () =
  (* Two windows over one width [w], related in both orders, then a concrete [then_]. *)
  let pair_then ~first ~w_o1 ~w_o2 then_ =
    let w = Row.get_var () and o1 = Row.get_var () and o2 = Row.get_var () in
    let rel = [ eq (Var w) (w_o1 (Row.Var o1)); eq (Var w) (w_o2 (Row.Var o2)) ] in
    let rel = if first then rel else List.rev rel in
    solve_batches [ rel; then_ ~w ~o1 ~o2 ]
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
