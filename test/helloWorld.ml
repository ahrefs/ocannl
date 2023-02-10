open Base
open Ocannl

let%expect_test "Hello World" =
  Stdio.printf "Hello World!\n";
  [%expect {| Hello World! |}]

let%expect_test "Pointwise multiplication dims 1" =
  Random.init 0;
  (* "Hey" is inferred to be a scalar.
     Note the pointwise multiplication means "hey" does not have any input axes. *)
  let%ocannl y = 2 *. "hey" in
  let y_f = Network.unpack y in
  let open Operation.CLI in
  refresh_session ();
  print_formula ~with_code:false ~with_grad:false `Default @@ y_f;
  [%expect.unreachable]
[@@expect.uncaught_exn {|
  (* CR expect_test_collector: This test expectation appears to contain a backtrace.
     This is strongly discouraged as backtraces are fragile.
     Please change this test to not include a backtrace. *)

  ( "Node hey #1 dims 1x1;\
   \nNode v1p #2 dims val 1 grad -;\
   \nNode v2p #3 dims val 1 grad -;\
   \nNode heyv2p #4 dims 1;\
   \nNode v1ptheyv2p #5 dims 1;\
   \nFor #1: Forward init error: Compilation error:\
   \nFailure(\"\\027[1mFile \\\"/tmp/build_cb87ed_dune/nnrun3b3c8a.ml\\\", line 7, characters 44-45\\027[0m:\\n7 | let () = (get 1).forward <- Some (fun () -> )\\n                                                \\027[1;31m^\\027[0m\\n\\027[1;31mError\\027[0m: Syntax error\\n\")\
   \nRaised at Stdlib__Buffer.add_channel in file \"buffer.ml\", line 211, characters 18-35\
   \nCalled from Stdio__In_channel.input_all.loop in file \"src/in_channel.ml\", line 40, characters 4-47\
   \nCalled from Stdio__In_channel.input_all in file \"src/in_channel.ml\", line 43, characters 6-13\
   \n\
   \nIn code span  \027[1;31m{$# ... #$}\027[0m :\
   \nopen Base\
   \nopen Ocannl_runtime\
   \nopen Node\
   \nopen Base.Float\
   \nlet () = (get 1).value <- create_array [|1|] `Unspecified\
   \nlet () = reset_array ((get 1).value) (`ConstantOfValue 0.000000)\
   \nlet () = (get 1).forward <- Some (fun () ->  \027[1;31m{$# ) #$}\027[0m ")
  Raised at Ocannl__Operation.refresh_session.(fun) in file "lib/operation.ml", line 364, characters 8-44
  Called from Stdlib__List.iter in file "list.ml", line 110, characters 12-15
  Called from Base__List0.iter in file "src/list0.ml" (inlined), line 25, characters 16-35
  Called from Ocannl__Operation.refresh_session in file "lib/operation.ml", line 336, characters 2-1023
  Called from Tutorials__HelloWorld.(fun) in file "test/helloWorld.ml", line 15, characters 2-20
  Called from Expect_test_collector.Make.Instance_io.exec in file "collector/expect_test_collector.ml", line 262, characters 12-19

  Trailing output
  ---------------

  (* CR expect_test_collector: This test expectation appears to contain a backtrace.
     This is strongly discouraged as backtraces are fragile.
     Please change this test to not include a backtrace. *)

  Node v0p #1 dims -;
  Node hey #2 dims -;
  Node v2p #3 dims -;
  Node heyv2p #4 dims -;
  Compilation error:
  Failure("\027[1mFile \"/tmp/build_cb87ed_dune/nnrun3b3c8a.ml\", line 7, characters 44-45\027[0m:\n7 | let () = (get 1).forward <- Some (fun () -> )\n                                                \027[1;31m^\027[0m\n\027[1;31mError\027[0m: Syntax error\n")
  Raised at Stdlib__Buffer.add_channel in file "buffer.ml", line 211, characters 18-35
  Called from Stdio__In_channel.input_all.loop in file "src/in_channel.ml", line 40, characters 4-47
  Called from Stdio__In_channel.input_all in file "src/in_channel.ml", line 43, characters 6-13

  In code span  [1;31m{$# ... #$}[0m :
  open Base
  open Ocannl_runtime
  open Node
  open Base.Float
  let () = (get 1).value <- create_array [|1|] `Unspecified
  let () = reset_array ((get 1).value) (`ConstantOfValue 0.000000)
  let () = (get 1).forward <- Some (fun () ->  [1;31m{$# ) #$}[0m |}]

let%expect_test "Matrix multiplication dims 1x1" =
  Operation.drop_session();
  Random.init 0;
  (* Hey is inferred to be a matrix. *)
  let%ocannl hey = "hey" in
  let%ocannl y = "q" 2.0 * hey + "p" 1.0 in
  let y_f = Network.unpack y in
  let hey_f = Network.unpack hey in
  let open Operation.CLI in
  refresh_session ();
  print_formula ~with_code:false ~with_grad:false `Default @@ hey_f;
  [%expect {|
    [1] hey: shape q:1->p:1 layout: 0:1 x 1:1
    │0@p=0
    │q=1
    ┼────────────────────────────────────────
    │0.134 |}];
  print_formula ~with_code:false ~with_grad:false `Default @@ y_f;
  [%expect {|
    [5] v1ptheyv2p: shape p:1 layout: 0:1
    │p=0
    ┼────────────────────────────────────
    │1.267 |}]
