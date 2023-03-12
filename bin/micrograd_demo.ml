open Base
open Ocannl

let test_executor = `OCaml

let _suspended () =
  (* let open Operation.CLI in *)
  let open Session.CLI in
  drop_session();
  Random.init 0;
  set_executor test_executor;
  let%nn_op c = "a" (-4) + "b" 2 in
  (* TODO: exponentiation operator *)
  let%nn_op d = a * b + b * b * b in
  (* TODO: figure out how to have [let%nn_op c += c + 1] etc. *)
  let%nn_op c = c + c + 1 in
  let%nn_op c = c + 1 + c + ~-a in
  let%nn_op d = d + d * 2 + !/ (b + a) in
  let%nn_op d = d + 3 * d + !/ (b - a) in
  let%nn_op e = c - d in
  let%nn_op f = e * e in
  let%nn_op g = f / 2 in
  let%nn_op g = g + 10. / f in

  refresh_session ();
  print_formula ~with_code:false ~with_grad:false `Default g

let () =
  (* let open Operation.CLI in *)
  let open Session.CLI in
  drop_session();
  Random.init 0;
  set_executor test_executor;
  let%nn_op c = "a" (-4) + "b" 2 in
  (* TODO: exponentiation operator *)
  let%nn_op d = a *. b + b *. b *. b in
  (* TODO: figure out how to have [let%nn_op c += c + 1] etc. *)
  let%nn_op c = c + c + 1 in
  let%nn_op c = c + 1 + c + ~-a in
  let%nn_op d = d + d *. 2 + !/ (b + a) in
  let%nn_op d = d + 3 *. d + !/ (b - a) in
  let%nn_op e = c - d in
  let%nn_op f = e *. e in
  let%nn_op g = f /. 2 in
  let%nn_op g = g + 10. /. f in

  refresh_session ();
  print_preamble ();
  for i = !Formula.first_session_id to Ocannl_runtime.Node.global.unique_id - 1 do
    let n = Ocannl_runtime.Node.get i in
    let h = NodeUI.node_header n in
    Stdio.printf "Value for [%d] -- %s:\n%!" i h;
    NodeUI.pp_tensor_inline Caml.Format.std_formatter ~num_batch_axes:0 ~num_input_axes:0 ~num_output_axes:1 n.value;
    Caml.Format.print_newline();
    let open Ocannl_runtime in
    if Array.length (Node.dims n.grad) = 1 then (
      Stdio.printf "Gradient for [%d]:\n%!" i;
      NodeUI.pp_tensor_inline Caml.Format.std_formatter ~num_batch_axes:0 ~num_input_axes:0 ~num_output_axes:1 n.grad
    );
    Caml.Format.print_newline()
  done;
  print_formula ~with_tree:9 ~with_code:false ~with_grad:false `Default @@ g;
  print_formula ~with_tree:9 ~with_code:false ~with_grad:true `Default @@ a;
  print_formula ~with_tree:9 ~with_code:false ~with_grad:true `Default @@ b
