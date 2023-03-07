open Base
open Ocannl

let() =
  Operation.drop_session();
  Random.init 0;
  let open Operation.CLI in
  set_executor `OCaml;
  (* Hey is inferred to be a matrix. *)
  let hey = Network.return_term @@
    range_of_shape ~batch_dims:[7] ~input_dims:[9; 10; 11] ~output_dims:[13; 14] () in
  let%ocannl hoo = (1 + 1) * hey - 10 in
  let hoo_f = Network.unpack hoo in
  refresh_session ();
  (* print_formula ~with_code:false ~with_grad:false `Inline hey;
     [%expect {| |}]; *)
  print_formula ~with_code:false ~with_grad:false `Default hoo_f
  (* Disable line wrapping for viewing the output. In VSCode: `View: Toggle Word Wrap`. *)
