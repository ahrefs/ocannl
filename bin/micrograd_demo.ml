open Base
open Ocannl
module CLI = Operation.CLI

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
  print_node_tree ~with_grad:true ~depth:99 g.node_id;
  Stdio.print_endline "";
  print_formula ~with_code:true ~with_grad:true `Default @@ a;
  Stdio.print_endline "";
  print_formula ~with_code:true ~with_grad:true `Default @@ b
