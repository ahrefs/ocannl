open Base
open Ocannl

let test_executor = `OCaml

let%expect_test "Micrograd README basic example" =
  let open Operation.CLI in
  drop_session();
  Random.init 0;
  set_executor test_executor;
  let%ocannl c = "a" (-4) + "b" 2 in
  (* TODO: exponentiation operator *)
  let%ocannl d = a *. b + b *. b *. b in
  (* TODO: figure out how to have [let%ocannl c += c + 1] etc. *)
  let%ocannl c = c + c + 1 in
  let%ocannl c = c + 1 + c + ~-a in
  let%ocannl d = d + d *. 2 + !/ (b + a) in
  let%ocannl d = d + 3 *. d + !/ (b - a) in
  let%ocannl e = c - d in
  let%ocannl f = e *. e in
  let%ocannl g = f /. 2 in
  let%ocannl g = g + 10. /. f in

  let g_f = Network.unpack g in
  let a_f = Network.unpack a in
  let b_f = Network.unpack b in
  refresh_session ();
  print_formula ~with_code:false ~with_grad:false `Default @@ g_f;
  [%expect {|
    ┌────────────────────────────────────────────────────┐
    │[211] ((10*.**-1(#107))+(#208*.**-1(2))): shape 0:1 │
    │┌┬─────────┐                                        │
    │││axis 0   │                                        │
    │├┼─────────┼─────────────────────────────────────── │
    │││ 2.47e+1 │                                        │
    │└┴─────────┘                                        │
    └────────────────────────────────────────────────────┘ |}];
  print_formula ~with_code:false ~with_grad:true `Default @@ a_f;
  [%expect {| |}];
  print_formula ~with_code:false ~with_grad:true `Default @@ b_f;
  [%expect {| |}]

