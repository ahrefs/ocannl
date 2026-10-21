open Base
open Ocannl
module Idx = Ir.Indexing
module Tn = Ir.Tnode

(* gh-490 v1: symbolic extents in shape inference. A static-indexing symbol with a declared range
   plugs into shape inference as a rigid symbolic dimension via [Shape.set_sym_dim]; it is
   materialized at its maximum extent (the range) for Tnode dims and projections. *)

let dims_to_string tn =
  Lazy.force tn.Tn.dims |> Array.to_list |> List.map ~f:Int.to_string |> String.concat ~sep:"x"

let test_symbolic_axis_forward () =
  let open Nn_blocks.DSL_modules in
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  Stdio.printf "=== Symbolic axis: capture, solve, materialize at range ===\n";
  let seq, _bindings = Idx.get_static_symbol ~static_range:6 Idx.Empty in
  let%op x = { x = 0.5 } in
  let%op y = (2. *. x) ++ "s=>s" [ "s" ] in
  Shape.set_sym_dim s seq;
  let ctx = Train.forward_once ctx y in
  Stdio.printf "Captured dim s solved_dim: %s\n"
    (match s.var_ref.solved_dim with Some d -> Int.to_string d | None -> "none (symbolic)");
  Stdio.printf "Captured dim s solved_sym: %s\n"
    (match s.var_ref.solved_sym with
    | Some { Idx.static_symbol; _ } -> Idx.symbol_ident static_symbol
    | None -> "none");
  Stdio.printf "y dims: %s\n" (dims_to_string y.Tensor.value);
  Stdio.printf "x dims: %s\n" (dims_to_string x.Tensor.value);
  Stdio.printf "y shape: %s\n" (Shape.to_string_hum ~style:Axis_size y.Tensor.shape);
  (* Executed at the materialized extent: the whole 6-element buffer is computed. *)
  let ys = Context.get_values ctx y.Tensor.value in
  Stdio.printf "y values: %s\n"
    (String.concat ~sep:" " (Array.to_list (Array.map ys ~f:(Printf.sprintf "%.2f"))));
  (* A reduction over the symbolic axis iterates the materialized extent. *)
  let%op total = y ++ "... => 0" in
  let ctx = Train.forward_once ctx total in
  let total_v = (Context.get_values ctx total.Tensor.value).(0) in
  Stdio.printf "sum over symbolic axis: %.2f\n" total_v;
  (* Embedding the captured dimension as a scalar value materializes the declared range. *)
  let%op sdim = dim s in
  let ctx = Train.forward_once ctx sdim in
  Stdio.printf "embedded dim s (materialized range): %.1f\n"
    (Context.get_values ctx sdim.Tensor.value).(0)

let test_broadcast_and_reuse () =
  let open Nn_blocks.DSL_modules in
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  Stdio.printf "\n=== Same symbol on two tensors unifies; scalars broadcast to it ===\n";
  let seq, _bindings = Idx.get_static_symbol ~static_range:4 Idx.Empty in
  let%op a = { a = 0.25 } in
  let%op b = { b = 0.5 } in
  let%op a2 = a ++ "s=>s" [ "s" ] in
  let%op b2 = b ++ "t=>t" [ "t" ] in
  Shape.set_sym_dim s seq;
  Shape.set_sym_dim t seq;
  (* Pointwise addition requires the two symbolic axes to unify (same symbol: OK), and broadcasts
     the scalar constant. *)
  let%op c = a2 + b2 + 1. in
  let ctx = Train.forward_once ctx c in
  Stdio.printf "c dims: %s\n" (dims_to_string c.Tensor.value);
  let cv = Context.get_values ctx c.Tensor.value in
  Stdio.printf "c values (a + b + 1 = 1.75 each): %s\n"
    (String.concat ~sep:" " (Array.to_list (Array.map cv ~f:(Printf.sprintf "%.2f"))))

(* A case whose failure left the session tainted (gh-ocannl-1301) is rewound, so the next case
   starts from the last closure. *)
let expect_shape_error name f =
  (try
     f ();
     Stdio.printf "%s: ERROR - expected Shape_error, got none\n" name
   with
  | Row.Shape_error (msg, _) -> Stdio.printf "%s: correctly raised: %s\n" name msg
  | Utils.User_error msg -> Stdio.printf "%s: correctly raised: %s\n" name msg);
  if Option.is_some (Shape.taint_provenance ()) then Tensor.rewind_to_last_closure ()

let test_error_cases () =
  let open Nn_blocks.DSL_modules in
  Tensor.unsafe_reinitialize ();
  Stdio.printf "\n=== Error cases ===\n";
  (* Missing range. *)
  let no_range, _bindings = Idx.get_static_symbol Idx.Empty in
  let ref1 = Shape.get_variable_ref "ref1" in
  expect_shape_error "set_sym_dim without static_range" (fun () -> Shape.set_sym_dim ref1 no_range);
  (* Conflicts between concrete and symbolic settings on the same reference. *)
  let sym4, _bindings = Idx.get_static_symbol ~static_range:4 Idx.Empty in
  let sym8, _bindings = Idx.get_static_symbol ~static_range:8 Idx.Empty in
  let ref2 = Shape.get_variable_ref "ref2" in
  Shape.set_dim ref2 5;
  expect_shape_error "set_sym_dim after set_dim" (fun () -> Shape.set_sym_dim ref2 sym4);
  let ref3 = Shape.get_variable_ref "ref3" in
  Shape.set_sym_dim ref3 sym4;
  expect_shape_error "set_dim after set_sym_dim" (fun () -> Shape.set_dim ref3 7);
  expect_shape_error "set_sym_dim with a different symbol" (fun () -> Shape.set_sym_dim ref3 sym8);
  Shape.set_sym_dim ref3 sym4;
  Stdio.printf "set_sym_dim with the same symbol again: OK\n";
  (* A reference whose captured axis is already solved concrete rejects a symbolic setting. *)
  let ctx = Context.auto () in
  let seq, _bindings = Idx.get_static_symbol ~static_range:3 Idx.Empty in
  let%op u = { u = 0.5; o = [ 3 ] } in
  let%op _v = u ++ "s=>s" [ "s" ] in
  expect_shape_error "set_sym_dim on an axis already solved concrete" (fun () ->
      Shape.set_sym_dim s seq);
  (* Solver-level mismatch: a rigid symbolic dimension never unifies with a concrete dimension, even
     one equal to its range. *)
  let%op u2 = { u2 = 0.5 } in
  let%op v2 = u2 ++ "s=>s" [ "s" ] in
  let%op k = { k = 1.0; o = [ 3 ] } in
  expect_shape_error "symbolic vs concrete axis" (fun () ->
      Shape.set_sym_dim s seq;
      let%op vk = v2 + k in
      ignore (Train.forward_once ctx vk));
  (* Two distinct symbols on axes that must unify. *)
  let sym_a, _bindings = Idx.get_static_symbol ~static_range:5 Idx.Empty in
  let sym_b, _bindings = Idx.get_static_symbol ~static_range:5 Idx.Empty in
  let%op e = { e = 0.5 } in
  let%op f = { f = 0.5 } in
  let%op e2 = e ++ "p=>p" [ "p" ] in
  let%op f2 = f ++ "q=>q" [ "q" ] in
  expect_shape_error "two distinct symbols must not unify" (fun () ->
      Shape.set_sym_dim p sym_a;
      Shape.set_sym_dim q sym_b;
      let%op g = e2 + f2 in
      ignore (Train.forward_once ctx g));
  (* Row variables cannot be set to symbolic dimensions (shapes left free so the row variable is
     still unsolved at the call). *)
  let sym6, _bindings = Idx.get_static_symbol ~static_range:6 Idx.Empty in
  let%op m = { m = 0.5 } in
  let%op n = { n = 0.5 } in
  let%op _mn = m +* "a..r..;..r..b=>ab" [ "r" ] n in
  expect_shape_error "set_sym_dim on a row variable" (fun () -> Shape.set_sym_dim r sym6);
  (* Total-elements constraints meeting a symbolic dimension fail fast (no Sym_denom in v1): a
     dim-var/row-var equality routes through a Total_elems constraint. *)
  let%op w = { w = 0.5 } in
  let%op w2 = w ++ "s=>s" [ "s" ] in
  let%op wm = { wm = 0.5 } in
  let%op _wr = wm ++ "a..z..=>a..z.." [ "z" ] in
  expect_shape_error "Total_elems over a symbolic dimension" (fun () ->
      (* Emit the Total_elems (row total = dim variable) constraint first, then solve the dim
         variable to a symbolic dimension: the constraint's substitution fails fast. *)
      Shape.set_equal z s;
      Shape.set_sym_dim s sym6;
      ignore (Train.forward_once ctx w2));
  (* Mixed slice-index / extent uses of one binding are rejected in both orders: a slice index
     validates strictly in [0, range) while an extent validates inclusively in [0, range]. *)
  let slice_sym, _bindings = Idx.get_static_symbol ~static_range:2 Idx.Empty in
  let mb = TDSL.range_of_shape ~label:[ "mb" ] ~batch_dims:[ 2 ] ~output_dims:[ 3 ] () in
  let%op _sl = mb @| slice_sym in
  let%op wa = { wa = 0.5 } in
  let%op _wa2 = wa ++ "u=>u" [ "u" ] in
  expect_shape_error "extent after slice use" (fun () -> Shape.set_sym_dim u slice_sym);
  let mixed_sym, _bindings = Idx.get_static_symbol ~static_range:2 Idx.Empty in
  let%op wb = { wb = 0.5 } in
  let%op _wb2 = wb ++ "u=>u" [ "u" ] in
  Shape.set_sym_dim u mixed_sym;
  let mc = TDSL.range_of_shape ~label:[ "mc" ] ~batch_dims:[ 2 ] ~output_dims:[ 3 ] () in
  expect_shape_error "slice after extent use" (fun () ->
      let%op _sl2 = mc @| mixed_sym in
      ())

let () =
  test_symbolic_axis_forward ();
  test_broadcast_and_reuse ();
  test_error_cases ()
