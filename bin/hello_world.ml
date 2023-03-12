open Base
open Ocannl

let() =
  Session.drop_session();
  Random.init 0;
  let open Operation.CLI in
  let open Session.CLI in
  set_executor `OCaml;
  (* Hey is inferred to be a matrix. *)
  let hey =
    range_of_shape ~batch_dims:[7] ~input_dims:[9; 10; 11] ~output_dims:[13; 14] () in
  let%nn_op hoo = (1 + 1) * hey - 10 in
  refresh_session ();
  print_formula ~with_tree:9 ~with_code:false ~with_grad:false `Default hoo
  (* Disable line wrapping for viewing the output. In VSCode: `View: Toggle Word Wrap`. *)
