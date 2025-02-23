open Base
open Ocannl
module Tn = Arrayjit.Tnode
module IDX = Train.IDX
module CDSL = Train.CDSL
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module Utils = Arrayjit.Utils
module Rand = Arrayjit.Rand.Lib

module type Backend = Arrayjit.Backend_intf.Backend

let plot_unop ~f =
  Tensor.unsafe_reinitialize ();
  Rand.init 0;
  let module Backend = (val Arrayjit.Backends.fresh_backend ()) in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  let open Operation.At in
  CDSL.virtualize_settings.enable_device_only <- false;
  let size = 100 in
  let xs = Array.init size ~f:Float.(fun i -> (of_int i / 10.) - 5.) in
  let x_flat =
    Tensor.term ~grad_spec:Require_grad ~label:[ "x_flat" ]
      ~init_op:(Constant_fill { values = xs; strict = true })
      ()
  in
  let step_sym, bindings = IDX.get_static_symbol ~static_range:size IDX.empty in
  let%op x = x_flat @| step_sym in
  let%op fx = f x in
  Train.set_hosted x.value;
  Train.set_hosted (Option.value_exn ~here:[%here] x.Tensor.diff).grad;
  let update = Train.grad_update fx in
  let fx_routine = Train.to_routine (module Backend) ctx bindings update.fwd_bprop in
  let step_ref = IDX.find_exn fx_routine.bindings step_sym in
  let ys, dys =
    Array.unzip
    @@ Array.mapi xs ~f:(fun i _ ->
           step_ref := i;
           Train.run fx_routine;
           (fx.@[0], x.@%[0]))
  in
  (* It is fine to loop around the data: it's "next epoch". We redo the work though. *)
  PrintBox_utils.plot ~x_label:"x" ~y_label:"f(x)"
    [
      Scatterplot { points = Array.zip_exn xs dys; content = PrintBox.line "*" };
      Scatterplot { points = Array.zip_exn xs ys; content = PrintBox.line "#" };
      Line_plot { points = Array.create ~len:20 0.; content = PrintBox.line "-" };
    ]

let%expect_test "relu" =
  let%op f x = relu x in
  let plot_box = plot_unop ~f in
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect
    {|
    ┌────┬────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │ 4.9│                                                                                                   #│
    │    │                                                                                                  # │
    │    │                                                                                                #   │
    │    │                                                                                                #   │
    │    │                                                                                              ##    │
    │    │                                                                                             #      │
    │    │                                                                                           #        │
    │    │                                                                                           #        │
    │    │                                                                                         ##         │
    │    │                                                                                        #           │
    │    │                                                                                      #             │
    │    │                                                                                      #             │
    │    │                                                                                    ##              │
    │    │                                                                                   #                │
    │    │                                                                                 #                  │
    │    │                                                                                 #                  │
    │    │                                                                               ##                   │
    │    │                                                                             #                      │
    │f   │                                                                             #                      │
    │(   │                                                                            #                       │
    │x   │                                                                          ##                        │
    │)   │                                                                        #                           │
    │    │                                                                        #                           │
    │    │                                                                       #                            │
    │    │                                                                     ##                             │
    │    │                                                                   #                                │
    │    │                                                                   #                                │
    │    │                                                                  #                                 │
    │    │                                                                ##                                  │
    │    │                                                              #                                     │
    │    │                                                              #                                     │
    │    │                                                            #                                       │
    │    │                                                  * * ** * ** * **** **** **** *** **** **** **** **│
    │    │                                                         #                                          │
    │    │                                                         #                                          │
    │    │                                                       #                                            │
    │    │                                                      ##                                            │
    │    │                                                    #                                               │
    │    │                                                    #                                               │
    │ 0  │* * ** * ***** *** **** ***** **** **** * **** ****    -    -    -    -    -    -    -    -    -    │
    ├────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │    │-5                                                                                               4.9│
    │    │                                                 x                                                  │
    └────┴────────────────────────────────────────────────────────────────────────────────────────────────────┘
    |}]

let%expect_test "sat01" =
  let%op f x = sat01 x in
  let plot_box = plot_unop ~f in
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect
    {|
    ┌──┬────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │ 1│                                        * **** **** * ** * *# # #### #### #### ### #### #### #### ##│
    │  │                                                                                                    │
    │  │                                                                                                    │
    │  │                                                                                                    │
    │  │                                                           #                                        │
    │  │                                                                                                    │
    │  │                                                                                                    │
    │  │                                                                                                    │
    │  │                                                         #                                          │
    │  │                                                                                                    │
    │  │                                                                                                    │
    │  │                                                                                                    │
    │  │                                                         #                                          │
    │  │                                                                                                    │
    │  │                                                                                                    │
    │  │                                                                                                    │
    │  │                                                       #                                            │
    │  │                                                                                                    │
    │f │                                                                                                    │
    │( │                                                                                                    │
    │x │                                                       #                                            │
    │) │                                                                                                    │
    │  │                                                                                                    │
    │  │                                                                                                    │
    │  │                                                      #                                             │
    │  │                                                                                                    │
    │  │                                                                                                    │
    │  │                                                                                                    │
    │  │                                                    #                                               │
    │  │                                                                                                    │
    │  │                                                                                                    │
    │  │                                                                                                    │
    │  │                                                    #                                               │
    │  │                                                                                                    │
    │  │                                                                                                    │
    │  │                                                                                                    │
    │  │                                                  #                                                 │
    │  │                                                                                                    │
    │  │                                                                                                    │
    │ 0│* * ** * ***** *** **** ***** **** **** * #### ####    -    * * **** **** **** *** **** **** **** **│
    ├──┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │  │-5                                                                                               4.9│
    │  │                                                 x                                                  │
    └──┴────────────────────────────────────────────────────────────────────────────────────────────────────┘
    |}]
let%expect_test "exp(x)" =
  let%op f x = exp x in
  let plot_box = plot_unop ~f in
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect {|
    ┌────────┬────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │ 1.3e+02│                                                                                                   *│
    │        │                                                                                                    │
    │        │                                                                                                    │
    │        │                                                                                                    │
    │        │                                                                                                  * │
    │        │                                                                                                    │
    │        │                                                                                                    │
    │        │                                                                                                    │
    │        │                                                                                                *   │
    │        │                                                                                                    │
    │        │                                                                                                    │
    │        │                                                                                                *   │
    │        │                                                                                                    │
    │        │                                                                                               *    │
    │        │                                                                                                    │
    │        │                                                                                                    │
    │        │                                                                                              *     │
    │        │                                                                                                    │
    │f       │                                                                                             *      │
    │(       │                                                                                                    │
    │x       │                                                                                           *        │
    │)       │                                                                                                    │
    │        │                                                                                           *        │
    │        │                                                                                                    │
    │        │                                                                                          *         │
    │        │                                                                                         *          │
    │        │                                                                                                    │
    │        │                                                                                        *           │
    │        │                                                                                      *             │
    │        │                                                                                      *             │
    │        │                                                                                     *              │
    │        │                                                                                    *               │
    │        │                                                                                 * *                │
    │        │                                                                                 *                  │
    │        │                                                                               **                   │
    │        │                                                                             *                      │
    │        │                                                                          ***                       │
    │        │                                                                      ***                           │
    │        │                                                              * **** *                              │
    │ 0      │* * ** * ***** *** **** ***** **** **** * **** **** * ** * ** *  -    -    -    -    -    -    -    │
    ├────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │        │-5                                                                                               4.9│
    │        │                                                 x                                                  │
    └────────┴────────────────────────────────────────────────────────────────────────────────────────────────────┘
    |}]

let%expect_test "log(x)" =
  let%op f x = log x in 
  let plot_box = plot_unop ~f in
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect {|
    ┌─────┬────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │ 1.6 │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │f    │                                                                                                    │
    │(    │                                                                                                    │
    │x    │                                                                                                    │
    │)    │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │ -inf│* * ** * ***** *** **** ***** **** **** * **** **** * ** * ** * **** **** **** *** **** **** **** **│
    ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │     │-5                                                                                               4.9│
    │     │                                                 x                                                  │
    └─────┴────────────────────────────────────────────────────────────────────────────────────────────────────┘
    |}]

let%expect_test "log2(x)" =
  let%op f x = log2 x in
  let plot_box = plot_unop ~f in
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect {|
    ┌─────┬────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │ 2.3 │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │f    │                                                                                                    │
    │(    │                                                                                                    │
    │x    │                                                                                                    │
    │)    │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │     │                                                                                                    │
    │ -inf│* * ** * ***** *** **** ***** **** **** * **** **** * ** * ** * **** **** **** *** **** **** **** **│
    ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │     │-5                                                                                               4.9│
    │     │                                                 x                                                  │
    └─────┴────────────────────────────────────────────────────────────────────────────────────────────────────┘
    |}]

let%expect_test "sin(x)" =
  let%op f x = sin x in
  let plot_box = plot_unop ~f in
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect.unreachable]
[@@expect.uncaught_exn {|
  (* CR expect_test_collector: This test expectation appears to contain a backtrace.
     This is strongly discouraged as backtraces are fragile.
     Please change this test to not include a backtrace. *)
  "Assert_failure arrayjit/lib/c_syntax.ml:47:4"
  Raised at Arrayjit__C_syntax.C_syntax.pp_array_offset in file "arrayjit/lib/c_syntax.ml", line 47, characters 4-39
  Called from Stdlib__Format.output_acc in file "format.ml", line 1393, characters 32-48
  Called from Stdlib__Format.output_acc in file "format.ml", line 1383, characters 4-20
  Called from Stdlib__Format.output_acc in file "format.ml", line 1396, characters 32-48
  Called from Stdlib__Format.output_acc in file "format.ml", line 1395, characters 32-48
  Called from Stdlib__Format.output_acc in file "format.ml", line 1383, characters 4-20
  Called from Stdlib__Format.output_acc in file "format.ml", line 1383, characters 4-20
  Called from Stdlib__Format.kfprintf.(fun) in file "format.ml", line 1444, characters 16-34
  Called from Arrayjit__C_syntax.C_syntax.compile_main.pp_ll in file "arrayjit/lib/c_syntax.ml", line 136, characters 12-96
  Called from Stdlib__List.iter in file "list.ml", line 112, characters 12-15
  Called from Stdlib__Format.output_acc in file "format.ml", line 1383, characters 4-20
  Called from Stdlib__Format.kfprintf.(fun) in file "format.ml", line 1444, characters 16-34
  Called from Stdlib__List.iter in file "list.ml", line 112, characters 12-15
  Called from Stdlib__Format.output_acc in file "format.ml", line 1383, characters 4-20
  Called from Stdlib__Format.kfprintf.(fun) in file "format.ml", line 1444, characters 16-34
  Called from Stdlib__List.iter in file "list.ml", line 112, characters 12-15
  Called from Stdlib__Format.output_acc in file "format.ml", line 1383, characters 4-20
  Called from Stdlib__Format.kfprintf.(fun) in file "format.ml", line 1444, characters 16-34
  Called from Arrayjit__C_syntax.C_syntax.compile_proc in file "arrayjit/lib/c_syntax.ml", line 374, characters 4-38
  Re-raised at Arrayjit__C_syntax.C_syntax.compile_proc in file "arrayjit/lib/c_syntax.ml", lines 278-376, characters 31-10
  Called from Arrayjit__Cc_backend.compile in file "arrayjit/lib/cc_backend.ml", line 101, characters 15-71
  Called from Arrayjit__Backends.Add_device.compile in file "arrayjit/lib/backends.ml", line 246, characters 15-45
  Called from Arrayjit__Backends.Raise_backend.compile in file "arrayjit/lib/backends.ml", line 350, characters 29-59
  Re-raised at Arrayjit__Backends.Raise_backend.compile in file "arrayjit/lib/backends.ml", line 350, characters 4-59
  Re-raised at Arrayjit__Backends.Raise_backend.compile in file "arrayjit/lib/backends.ml", lines 346-354, characters 26-99
  Called from Ocannl__Train.to_routine in file "lib/train.ml", line 399, characters 26-61
  Called from Tutorials__Primitive_ops.plot_unop in file "test/primitive_ops.ml", line 34, characters 19-82
  Called from Tutorials__Primitive_ops.(fun) in file "test/primitive_ops.ml", line 314, characters 17-29
  Called from Ppx_expect_runtime__Test_block.Configured.dump_backtrace in file "runtime/test_block.ml", line 142, characters 10-28
  |}]

let%expect_test "cos(x)" =
  let%op f x = cos x in
  let plot_box = plot_unop ~f in
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect {|
    ┌───┬────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │ 1 │                                                  #                                                 │
    │   │                                ** ***         #### #                                         *** **│
    │   │                              **      *      #        #                                      *      │
    │   │                            *         *      #         #                                   *        │
    │   │                           *            *   #          #                                   *        │
    │   │                           *            *  #             #                                *         │
    │   │                                          *              #                                          │
    │   │                          *                *                                             *          │
    │   │                         *              #                  #                            *           │
    │   │                                        #   *               #                                       │
    │   │                        *                                                             *             │
    │   │                                      #      *              #                         *             │
    │   │                      *                      *                                                      │
    │   │                      *               #                       #                      *              │
    │   │#                                              *                                                    │
    │   │                     *               #                        #                     *               │
    │   │#                                               *                                                  #│
    │   │                    *               #                           #                  *                │
    │f  │  #                                              *                                                # │
    │(  │                   *               #                             #               *                  │
    │x  │- #  -    -    -    -    -    -    -    -    -    *    -    -    -    -    -    -    -    -    -#   │
    │)  │                 *               #                                #              *                  │
    │   │    #                                             *                                             #   │
    │   │                 *               #                                 #            *                   │
    │   │     #                          #                   *              #                           #    │
    │   │                *                                                              *                    │
    │   │     #                         #                    *                #                        #     │
    │   │               *                                                             *                      │
    │   │       #                      #                       *               #      *               #      │
    │   │             *                                         *                                            │
    │   │       #    *               #                                          #    *              #        │
    │   │         #                 #                           *                #                  #        │
    │   │            *                                                              *                        │
    │   │          #*               #                             *              # *               #         │
    │   │           #              #                              *                #              #          │
    │   │          * #                                              *            *               #           │
    │   │         *               #                                  *           *  #                        │
    │   │       *    ##          #                                   *          *    #         #             │
    │   │       *       #      #                                       *      **      #       #              │
    │ -1│* * **          ## ###                                        * ****           ### ##               │
    ├───┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │   │-5                                                                                               4.9│
    │   │                                                 x                                                  │
    └───┴────────────────────────────────────────────────────────────────────────────────────────────────────┘
    |}]

let%expect_test "neg(x)" =
  let%op f x = neg x in
  let plot_box = plot_unop ~f in
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect {|
    ┌─────┬────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │ 5   │#                                                                                                   │
    │     │# #                                                                                                 │
    │     │  # ##                                                                                              │
    │     │     # #                                                                                            │
    │     │       # ##                                                                                         │
    │     │           ##                                                                                       │
    │     │            ## #                                                                                    │
    │     │                ##                                                                                  │
    │     │                 # ##                                                                               │
    │     │                     ##                                                                             │
    │     │                      # ##                                                                          │
    │     │                          ##                                                                        │
    │     │                           ## #                                                                     │
    │     │                               ###                                                                  │
    │     │                                 # #                                                                │
    │     │                                    ###                                                             │
    │     │                                      # #                                                           │
    │     │                                        # ##                                                        │
    │f    │                                            ##                                                      │
    │(    │                                             # ##                                                   │
    │x    │-    -    -    -    -    -    -    -    -    -   ##    -    -    -    -    -    -    -    -    -    │
    │)    │                                                  # #                                               │
    │     │                                                      ##                                            │
    │     │                                                       # #                                          │
    │     │* * ** * ***** *** **** ***** **** **** * **** **** * ** * ** * **** **** **** *** **** **** **** **│
    │     │                                                            # #                                     │
    │     │                                                                ###                                 │
    │     │                                                                   #                                │
    │     │                                                                     ###                            │
    │     │                                                                        #                           │
    │     │                                                                          ###                       │
    │     │                                                                             #                      │
    │     │                                                                               ###                  │
    │     │                                                                                 # #                │
    │     │                                                                                    ###             │
    │     │                                                                                      # #           │
    │     │                                                                                         ###        │
    │     │                                                                                           # #      │
    │     │                                                                                              ###   │
    │ -4.9│                                                                                                # ##│
    ├─────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │     │-5                                                                                               4.9│
    │     │                                                 x                                                  │
    └─────┴────────────────────────────────────────────────────────────────────────────────────────────────────┘
    |}]

let%expect_test "fma(x, 2, 1)" =
  let%op f x = fma x !.2. !.1. in
  let plot_box = plot_unop ~f in
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect {|
    ┌───┬────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │ 11│                                                                                                   #│
    │   │                                                                                                # # │
    │   │                                                                                              ###   │
    │   │                                                                                           # #      │
    │   │                                                                                         ###        │
    │   │                                                                                      # #           │
    │   │                                                                                    ###             │
    │   │                                                                                 # #                │
    │   │                                                                               ###                  │
    │   │                                                                             #                      │
    │   │                                                                          ###                       │
    │   │                                                                        #                           │
    │   │                                                                     ###                            │
    │   │                                                                   #                                │
    │   │                                                                ###                                 │
    │   │                                                            # #                                     │
    │   │                                                           ##                                       │
    │   │                                                       # #                                          │
    │f  │* * ** * ***** *** **** ***** **** **** * **** **** * ** * ** * **** **** **** *** **** **** **** **│
    │(  │                                                  # #                                               │
    │x  │                                                 ##                                                 │
    │)  │                                             # ##                                                   │
    │   │-    -    -    -    -    -    -    -    -   ##    -    -    -    -    -    -    -    -    -    -    │
    │   │                                        # ##                                                        │
    │   │                                      # #                                                           │
    │   │                                    ###                                                             │
    │   │                                 # #                                                                │
    │   │                               ###                                                                  │
    │   │                           ## #                                                                     │
    │   │                          ##                                                                        │
    │   │                      # ##                                                                          │
    │   │                     ##                                                                             │
    │   │                 # ##                                                                               │
    │   │                ##                                                                                  │
    │   │            ## #                                                                                    │
    │   │           ##                                                                                       │
    │   │       # ##                                                                                         │
    │   │     # #                                                                                            │
    │   │  # ##                                                                                              │
    │ -9│# #                                                                                                 │
    ├───┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
    │   │-5                                                                                               4.9│
    │   │                                                 x                                                  │
    └───┴────────────────────────────────────────────────────────────────────────────────────────────────────┘
    |}]
