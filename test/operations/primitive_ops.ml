open Base
open Ocannl
module Tn = Ir.Tnode
module IDX = Train.IDX
module CDSL = Train.CDSL
open Nn_blocks.DSL_modules

module type Backend = Ir.Backend_intf.Backend

let plot_unop ?(x_min = -5.) ?(x_max = 5.) ~f () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let open Operation.At in
  CDSL.virtualize_settings.enable_device_only <- false;
  let size = 100 in
  let xs =
    Array.init size ~f:Float.(fun i -> x_min + (of_int i * (x_max - x_min) / (of_int size - 1.)))
  in
  let x_flat = Tensor.term_init xs ~label:[ "x_flat" ] ~grad_spec:Require_grad () in
  let step_sym, bindings = IDX.get_static_symbol ~static_range:size IDX.empty in
  let%op x = x_flat @| step_sym in
  let%op fx = f x in
  Train.set_materialized x.value;
  Train.set_materialized (Option.value_exn ~here:[%here] x.Tensor.diff).grad;
  (* There actually are no params! Stress test the empty comp case. *)
  let ctx = Train.init_params ctx IDX.empty fx in
  let update = Train.grad_update fx in
  let ctx, fx_routine = Train.to_routine ctx bindings update in
  let step_ref = IDX.find_exn fx_routine.Context.bindings step_sym in
  step_ref := 0;
  let ctx = Context.run ctx fx_routine in
  let ys, dys =
    Array.unzip
    @@ Array.mapi xs ~f:(fun i _ ->
        step_ref := i;
        let ctx = Context.run ctx fx_routine in
        ((ctx, fx).@[0], (ctx, x).@%[0]))
  in
  (* It is fine to loop around the data: it's "next epoch". We redo the work though. *)
  PrintBox_utils.plot ~x_label:"x" ~y_label:"f(x)"
    ~legend:[ ("#", "f(x)"); ("*", "f'(x)"); ("-", "0") ]
    [
      Scatterplot { points = Array.zip_exn xs dys; content = PrintBox.line "*" };
      Scatterplot { points = Array.zip_exn xs ys; content = PrintBox.line "#" };
      Line_plot { points = Array.create ~len:20 0.; content = PrintBox.line "-" };
    ]

let%expect_test "relu" =
  let%op f x = relu x in
  let plot_box = plot_unop ~f () in
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect
    {|
    f(x)
       ┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
    5.0┤                                                                                                   #│
       │                                                                                                  # │
       │                                                                                                 #  │
       │                                                                                                #   │
    4.5┤                                                                                              #     │
       │                                                                                             #      │
       │                                                                                            #       │
       │                                                                                           #        │
    4.0┤                                                                                         ##         │
       │                                                                                        #           │
       │                                                                                      #             │
       │                                                                                      #             │
    3.5┤                                                                                    ##              │
       │                                                                                   #                │
       │                                                                                 #                  │
       │                                                                                ##                  │
    3.0┤                                                                               #                    │
       │                                                                              #                     │
       │                                                                             #                      │
       │                                                                           ##                       │
    2.5┤                                                                          #                         │
       │                                                                         #                          │
       │                                                                        #                           │
       │                                                                      ##                            │
    2.0┤                                                                     #                              │
       │                                                                    #                               │
       │                                                                   #                                │
       │                                                                 ##                                 │
    1.5┤                                                                #                                   │
       │                                                               #                                    │
       │                                                             ##                                     │
       │                                                            #                                       │
    1.0┤                                                 * ******************************* **** ******* ****│
       │                                                          #                                         │
       │                                                        ##                                          │
       │                                                       #                                            │
    0.5┤                                                      #                                             │
       │                                                     #                                              │
       │                                                   ##                                               │
    0.0┤**** *** *** *** ********-*************** ********-    -    -    -    -    -    -    -    -    -    │
       └┬────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬┘
        -5      -4        -3        -2        -1         0         1         2         3         4         5
                                                         x
                                                                                                    ┌───────┐
                                                                                                    │# f(x) │
                                                                                                    │* f'(x)│
                                                                                                    │- 0    │
                                                                                                    └───────┘
    |}]

let%expect_test "sat01" =
  let%op f x = sat01 x in
  let plot_box = plot_unop ~f () in
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect
    {|
    f(x)
       ┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
    1.0┤                                                 * *********###################### #### ####### ####│
       │                                                                                                    │
       │                                                           #                                        │
       │                                                                                                    │
    0.9┤                                                                                                    │
       │                                                                                                    │
       │                                                          #                                         │
       │                                                                                                    │
    0.8┤                                                                                                    │
       │                                                                                                    │
       │                                                         #                                          │
       │                                                                                                    │
    0.7┤                                                                                                    │
       │                                                                                                    │
       │                                                        #                                           │
       │                                                                                                    │
    0.6┤                                                                                                    │
       │                                                                                                    │
       │                                                       #                                            │
       │                                                                                                    │
    0.5┤                                                                                                    │
       │                                                                                                    │
       │                                                      #                                             │
       │                                                                                                    │
    0.4┤                                                                                                    │
       │                                                                                                    │
       │                                                     #                                              │
       │                                                                                                    │
    0.3┤                                                                                                    │
       │                                                                                                    │
       │                                                    #                                               │
       │                                                                                                    │
    0.2┤                                                                                                    │
       │                                                                                                    │
       │                                                   #                                                │
       │                                                                                                    │
    0.1┤                                                                                                    │
       │                                                                                                    │
       │                                                 #                                                  │
    0.0┤**** *** *** *** ********-*************** ********-    -    ********************** **** *******-****│
       └┬────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬┘
        -5      -4        -3        -2        -1         0         1         2         3         4         5
                                                         x
                                                                                                    ┌───────┐
                                                                                                    │# f(x) │
                                                                                                    │* f'(x)│
                                                                                                    │- 0    │
                                                                                                    └───────┘
    |}]

let%expect_test "exp(x)" =
  let%op f x = exp x in
  let plot_box = plot_unop ~f ~x_max:1.0 () in
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect
    {|
    f(x)
        ┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
    2.72┤                                                                                                   *│
        │                                                                                                    │
     2.6┤                                                                                                    │
        │                                                                                                  * │
        │                                                                                                    │
     2.4┤                                                                                                 *  │
        │                                                                                                    │
        │                                                                                                *   │
     2.2┤                                                                                                    │
        │                                                                                               *    │
        │                                                                                                    │
     2.0┤                                                                                              *     │
        │                                                                                             *      │
        │                                                                                                    │
     1.8┤                                                                                            *       │
        │                                                                                           *        │
        │                                                                                                    │
     1.6┤                                                                                          *         │
        │                                                                                         *          │
     1.4┤                                                                                        *           │
        │                                                                                                    │
        │                                                                                       *            │
     1.2┤                                                                                      *             │
        │                                                                                     *              │
        │                                                                                    *               │
     1.0┤                                                                                   *                │
        │                                                                                 **                 │
        │                                                                                *                   │
     0.8┤                                                                               *                    │
        │                                                                             **                     │
        │                                                                           **                       │
     0.6┤                                                                        **                          │
        │                                                                       *                            │
        │                                                                     **                             │
     0.4┤                                                                  ***                               │
        │                                                              ****                                  │
        │                                                         *****                                      │
     0.2┤                                                  *******                                           │
        │                                       ***********                                                  │
     0.0┤********* ************ ************** * -    -    -    -    -    -    -    -    -    -    -    -    │
        └┬───────┬───────┬───────┬────────┬───────┬───────┬───────┬────────┬───────┬───────┬───────┬────────┬┘
         -5.0  -4.5    -4.0    -3.5     -3.0    -2.5    -2.0    -1.5     -1.0    -0.5     0.0     0.5     1.0
                                                          x
                                                                                                     ┌───────┐
                                                                                                     │# f(x) │
                                                                                                     │* f'(x)│
                                                                                                     │- 0    │
                                                                                                     └───────┘
    |}]

let%expect_test "log(x)" =
  let%op f x = log x in
  let plot_box = plot_unop ~f ~x_min:0.1 ~x_max:5.0 () in
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect
    {|
    f(x)
         ┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
       10┤*                                                                                                   │
         │                                                                                                    │
         │                                                                                                    │
         │                                                                                                    │
        9┤                                                                                                    │
         │                                                                                                    │
         │                                                                                                    │
        8┤                                                                                                    │
         │                                                                                                    │
         │                                                                                                    │
        7┤                                                                                                    │
         │ *                                                                                                  │
         │                                                                                                    │
        6┤                                                                                                    │
         │                                                                                                    │
         │                                                                                                    │
        5┤  *                                                                                                 │
         │                                                                                                    │
         │                                                                                                    │
         │   *                                                                                                │
        4┤                                                                                                    │
         │                                                                                                    │
         │    *                                                                                               │
        3┤     *                                                                                              │
         │      *                                                                                             │
         │       *                                                                                            │
        2┤        **                                                                                          │
         │          **                                                                           #############│
         │            ***                                                ################### ####             │
        1┤                ******                        ######### #### ##                                     │
         │                      ******** *****##### ####                                                      │
         │                        ###### ##   ***** ************* **** ********************* *******          │
        0┤-    -    -    - ####### -    -    -    -    -    -    -    -    -    -    -    -    -    **********│
         │            ### #                                                                                   │
         │        ####                                                                                        │
       -1┤      ##                                                                                            │
         │    ##                                                                                              │
         │  ##                                                                                                │
         │ #                                                                                                  │
    -2.30┤#                                                                                                   │
         └┬───────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬──────────┬┘
          0.100  0.5       1.0       1.5       2.0       2.5       3.0       3.5       4.0       4.5       5.0
                                                           x
                                                                                                      ┌───────┐
                                                                                                      │# f(x) │
                                                                                                      │* f'(x)│
                                                                                                      │- 0    │
                                                                                                      └───────┘
    |}]

let%expect_test "log2(x)" =
  let%op f x = log2 x in
  let plot_box = plot_unop ~f ~x_min:0.1 ~x_max:5.0 () in
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect
    {|
    f(x)
         ┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
     14.4┤*                                                                                                   │
         │                                                                                                    │
         │                                                                                                    │
         │                                                                                                    │
       13┤                                                                                                    │
         │                                                                                                    │
       12┤                                                                                                    │
         │                                                                                                    │
       11┤                                                                                                    │
         │                                                                                                    │
       10┤                                                                                                    │
         │ *                                                                                                  │
        9┤                                                                                                    │
         │                                                                                                    │
         │                                                                                                    │
        8┤                                                                                                    │
         │  *                                                                                                 │
        7┤                                                                                                    │
         │                                                                                                    │
        6┤   *                                                                                                │
         │                                                                                                    │
        5┤                                                                                                    │
         │    *                                                                                               │
        4┤     *                                                                                              │
         │      *                                                                                             │
         │       *                                                                                            │
        3┤        **                                                                                          │
         │          **                                                                           #############│
        2┤            ***                                                ################### ####             │
         │                ******                        ######### #### ##                                     │
        1┤                      ******** *****##### ####                                                      │
         │                        ###### ##   ***** ************* **** ********************* *******          │
        0┤-    -    -    - ####### -    -    -    -    -    -    -    -    -    -    -    -    -    **********│
         │            ### #                                                                                   │
       -1┤        ####                                                                                        │
         │      ##                                                                                            │
         │    ##                                                                                              │
       -2┤  ##                                                                                                │
         │ #                                                                                                  │
    -3.32┤#                                                                                                   │
         └┬───────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬──────────┬┘
          0.100  0.5       1.0       1.5       2.0       2.5       3.0       3.5       4.0       4.5       5.0
                                                           x
                                                                                                      ┌───────┐
                                                                                                      │# f(x) │
                                                                                                      │* f'(x)│
                                                                                                      │- 0    │
                                                                                                      └───────┘
    |}]

let%expect_test "sin(x)" =
  let%op f x = sin x in
  let plot_box = plot_unop ~f () in
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect
    {|
    f(x)
          ┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
     1.000┤                                                                 #                                  │
          │#### ##                                        *** **         ### ###                               │
          │      #                                      **      **      #       #                              │
          │       #                                    *          *    #         #                             │
       0.8┤         #                                                 #           #                            │
          │          #                                *            * #             #                           │
          │                                          *              *               #                          │
          │           #                            *                #*                                         │
       0.6┤           #                                            #                 #                         │
          │                                        *                  *                                        │
          │             #                         *               #    *              #                        │
          │                                                      #                     #                       │
       0.4┤              #                       *                      *                                      │
          │               #                                     #                       #                      │
          │*                                    *                        *                                    *│
          │               #                                    #                         #                     │
       0.2┤ *                                  *                          *                                  * │
          │                 #                                 #                           #                    │
          │  *                                *                            *                                *  │
          │                  #                              #                              #                   │
       0.0┤- *  -    -    -    -    -    -   *-    -    -    -    -    -    *    -    -    -    -    -    -*   │
          │                   #                             #                               #                  │
          │   *                             *                                *                           *     │
          │                    #                           #                                #                  │
      -0.2┤     *                          *                                  *                          *     │
          │                     #                         #                                   #                │
          │      *                        *                                    *                        *      │
          │                      #                       #                                     #               │
      -0.4┤      *                       *                                      *               #      *       │
          │                       #     *               #                        *                             │
          │       *                #                   #                                         #    *        │
          │         *                  *                                          *                  *         │
      -0.6┤                        #                  #                                          #             │
          │          *                *              #                             *               #*          │
          │           *              *                                              *              *           │
          │                        *  #            #                                 *              #          │
      -0.8┤           *            *   #           #                                  *          *   #         │
          │             *         *     #         #                                    *         *    #        │
          │              **      *       #       #                                      *      **      #       │
    -1.000┤               * *****         #######                                        **** *         ## ####│
          └┬────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬┘
           -5      -4        -3        -2        -1         0         1         2         3         4         5
                                                            x
                                                                                                       ┌───────┐
                                                                                                       │# f(x) │
                                                                                                       │* f'(x)│
                                                                                                       │- 0    │
                                                                                                       └───────┘
    |}]

let%expect_test "cos(x)" =
  let%op f x = cos x in
  let plot_box = plot_unop ~f () in
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect
    {|
    f(x)
          ┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
     1.000┤                                  *                                                                 │
          │                               *** ***         ### ##                                        ** ****│
          │                              *       *      ##      ##                                     *       │
          │                             *         *    #          #                                   *        │
       0.8┤                            *           *                                                 *         │
          │                           *            *  #            #                                *          │
          │                          *               #              #                                          │
          │                                        # *               #                             *           │
       0.6┤                        *                  *                                          *             │
          │                                        #                  #                                        │
          │                        *              #    *               #                         *             │
          │                       *                     *                                                      │
       0.4┤                                      #                      #                       *              │
          │                      *                       *                                     *               │
          │#                                    #                        #                                    #│
          │                     *                         *                                   *                │
       0.2┤ #                                  #                          #                                  # │
          │                    *                           *                                *                  │
          │  #                                #                            #                                #  │
          │                   *                             *                               *                  │
       0.0┤- #  -    -    -    -    -    -   #-    -    -    -    -    -    #    -    -    -    -    -    -#   │
          │                  *                              *                              *                   │
          │   #                             #                                #                           #     │
          │                 *                                 *                           *                    │
      -0.2┤     #                          #                                  #                          #     │
          │               *                                    *                         *                     │
          │      #                        #                                    #                        #      │
          │               *                                     *                       *                      │
      -0.4┤      #       *               #                                      #                      #       │
          │                             #                        *               #     *                       │
          │       #     *                                         *                   *               #        │
          │         #                  #                                          #                  #         │
      -0.6┤           *                                            *                 *                         │
          │          #*               #                             *              #                #          │
          │           #              #                                              *              #           │
          │          *             #                                 *             * #                         │
      -0.8┤         * #            #                                  *           *   #          #             │
          │       *     #         #                                    *         *     #         #             │
          │      *       ##      #                                      *       *       #      ##              │
    -1.000┤**** **        # #####                                        *******         #### #                │
          └┬────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬┘
           -5      -4        -3        -2        -1         0         1         2         3         4         5
                                                            x
                                                                                                       ┌───────┐
                                                                                                       │# f(x) │
                                                                                                       │* f'(x)│
                                                                                                       │- 0    │
                                                                                                       └───────┘
    |}]

let%expect_test "neg(x)" =
  let%op f x = neg x in
  let plot_box = plot_unop ~f () in
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect
    {|
    f(x)
      ┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
     5┤#                                                                                                   │
      │ ##                                                                                                 │
      │  ## #                                                                                              │
      │      #                                                                                             │
     4┤       # ##                                                                                         │
      │           #                                                                                        │
      │             ###                                                                                    │
      │               # #                                                                                  │
     3┤                  ###                                                                               │
      │                     ##                                                                             │
      │                       ##                                                                           │
      │                          ##                                                                        │
     2┤                            ###                                                                     │
      │                               ##                                                                   │
      │                                 ###                                                                │
      │                                    ###                                                             │
     1┤                                       ##                                                           │
      │                                        # ##                                                        │
      │                                            ##                                                      │
      │                                              ###                                                   │
     0┤-    -    -    -    -    -    -    -    -    -   #-    -    -    -    -    -    -    -    -    -    │
      │                                                   ###                                              │
      │                                                      ##                                            │
      │                                                        ###                                         │
    -1┤**** *** *** *** ******** *************** ******** ******************************* **** ******* ****│
      │                                                             ###                                    │
      │                                                                ###                                 │
      │                                                                   ##                               │
    -2┤                                                                     ###                            │
      │                                                                        ##                          │
      │                                                                          ###                       │
      │                                                                             ##                     │
    -3┤                                                                               ###                  │
      │                                                                                 # #                │
      │                                                                                    ###             │
      │                                                                                      # #           │
    -4┤                                                                                         ###        │
      │                                                                                            ##      │
      │                                                                                              # #   │
    -5┤                                                                                                 ###│
      └┬────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬┘
       -5      -4        -3        -2        -1         0         1         2         3         4         5
                                                        x
                                                                                                   ┌───────┐
                                                                                                   │# f(x) │
                                                                                                   │* f'(x)│
                                                                                                   │- 0    │
                                                                                                   └───────┘
    |}]

let%expect_test "fma(x, 2, 1)" =
  let%op f x = fma x !.2. !.1. in
  let plot_box = plot_unop ~f () in
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect
    {|
    f(x)
         ┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
     11.0┤                                                                                                   #│
         │                                                                                                 ## │
       10┤                                                                                              # #   │
         │                                                                                            ##      │
         │                                                                                         ###        │
         │                                                                                      # #           │
        8┤                                                                                    ###             │
         │                                                                                 # #                │
         │                                                                               ###                  │
         │                                                                             ##                     │
        6┤                                                                          ###                       │
         │                                                                        ##                          │
         │                                                                     ###                            │
         │                                                                   ##                               │
        4┤                                                                ###                                 │
         │                                                             ###                                    │
         │                                                           ##                                       │
         │                                                        ###                                         │
        2┤**** *** *** *** ******** *************** ******** ******************************* **** ******* ****│
         │                                                   ###                                              │
         │                                                 #                                                  │
         │                                              ###                                                   │
        0┤-    -    -    -    -    -    -    -    -   ##    -    -    -    -    -    -    -    -    -    -    │
         │                                        # ##                                                        │
         │                                       ##                                                           │
         │                                    ###                                                             │
       -2┤                                 ###                                                                │
         │                               ##                                                                   │
         │                            ###                                                                     │
         │                          ##                                                                        │
       -4┤                       ##                                                                           │
         │                     ##                                                                             │
         │                  ###                                                                               │
         │               # #                                                                                  │
       -6┤             ###                                                                                    │
         │           #                                                                                        │
         │       # ##                                                                                         │
         │      #                                                                                             │
         │  ## #                                                                                              │
    -9.00┤###                                                                                                 │
         └┬────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬┘
          -5      -4        -3        -2        -1         0         1         2         3         4         5
                                                           x
                                                                                                      ┌───────┐
                                                                                                      │# f(x) │
                                                                                                      │* f'(x)│
                                                                                                      │- 0    │
                                                                                                      └───────┘
    |}]

let%expect_test "sqrt(x)" =
  let%op f x = sqrt x in
  let plot_box = plot_unop ~f ~x_min:0.1 ~x_max:5.0 () in
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect
    {|
    f(x)
        ┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
    2.24┤                                                                                                   #│
        │                                                                                              ##### │
        │                                                                                         #####      │
        │                                                                                     ####           │
        │                                                                                ## ##               │
     2.0┤                                                                           #####                    │
        │                                                                       ####                         │
        │                                                                  #####                             │
     1.8┤                                                              ####                                  │
        │                                                          ## #                                      │
        │                                                      # ##                                          │
        │                                                   ###                                              │
     1.6┤*                                              ####                                                 │
        │                                           ####                                                     │
        │                                        # #                                                         │
     1.4┤                                     ###                                                            │
        │                                  ###                                                               │
        │ *                             ###                                                                  │
        │                            ##                                                                      │
     1.2┤                         ###                                                                        │
        │  *                   ###                                                                           │
        │                    ##                                                                              │
     1.0┤   *              ##                                                                                │
        │              # ##                                                                                  │
        │    *        ##                                                                                     │
        │     *     ##                                                                                       │
     0.8┤      **  #                                                                                         │
        │        *#                                                                                          │
        │       # **                                                                                         │
     0.6┤     ##    ***                                                                                      │
        │    #         * *                                                                                   │
        │   #             *****                                                                              │
        │  #                   ********                                                                      │
     0.4┤ #                           * **********                                                           │
        │#                                       * ************* ****                                        │
        │                                                           * ********************* ************     │
     0.2┤                                                                                               *****│
        │                                                                                                    │
        │                                                                                                    │
     0.0┤-    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    │
        └┬───────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬──────────┬┘
         0.100  0.5       1.0       1.5       2.0       2.5       3.0       3.5       4.0       4.5       5.0
                                                          x
                                                                                                     ┌───────┐
                                                                                                     │# f(x) │
                                                                                                     │* f'(x)│
                                                                                                     │- 0    │
                                                                                                     └───────┘
    |}]

let%expect_test "recip(x)" =
  let%op f x = recip x in
  let plot_box = plot_unop ~f ~x_min:0.1 ~x_max:5.0 () in
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect
    {|
    f(x)
        ┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
      10┤#                                                                                                   │
        │                                                                                                    │
        │ ##                                                                                                 │
        │   #########                                                                                        │
       0┤-    -    - ###-**************-********** *************-****-********************* *****************│
        │        *******                                                                                     │
        │      **                                                                                            │
        │     *                                                                                              │
     -10┤    *                                                                                               │
        │                                                                                                    │
        │   *                                                                                                │
     -20┤                                                                                                    │
        │                                                                                                    │
        │  *                                                                                                 │
        │                                                                                                    │
     -30┤                                                                                                    │
        │                                                                                                    │
        │                                                                                                    │
     -40┤                                                                                                    │
        │                                                                                                    │
        │ *                                                                                                  │
        │                                                                                                    │
     -50┤                                                                                                    │
        │                                                                                                    │
        │                                                                                                    │
     -60┤                                                                                                    │
        │                                                                                                    │
        │                                                                                                    │
        │                                                                                                    │
     -70┤                                                                                                    │
        │                                                                                                    │
        │                                                                                                    │
     -80┤                                                                                                    │
        │                                                                                                    │
        │                                                                                                    │
        │                                                                                                    │
     -90┤                                                                                                    │
        │                                                                                                    │
        │                                                                                                    │
    -100┤*                                                                                                   │
        └┬───────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬──────────┬┘
         0.100  0.5       1.0       1.5       2.0       2.5       3.0       3.5       4.0       4.5       5.0
                                                          x
                                                                                                     ┌───────┐
                                                                                                     │# f(x) │
                                                                                                     │* f'(x)│
                                                                                                     │- 0    │
                                                                                                     └───────┘
    |}]

let%expect_test "recip_sqrt(x)" =
  let%op f x = recip_sqrt x in
  let plot_box = plot_unop ~f ~x_min:0.1 ~x_max:5.0 () in
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect
    {|
    f(x)
         ┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
     3.16┤#                                                                                                   │
         │                                                                                                    │
         │ ##                                                                                                 │
        2┤   ##                                                                                               │
         │     #######                                                                                        │
         │            ### ############## #####                                                                │
         │                                    ##### ############# #### ##################### #################│
        0┤-    -    -    -    -    -    -********** *************-****-********************* *****************│
         │              * **************                                                                      │
         │          ****                                                                                      │
         │       ***                                                                                          │
       -2┤      *                                                                                             │
         │     *                                                                                              │
         │    *                                                                                               │
         │                                                                                                    │
       -4┤   *                                                                                                │
         │                                                                                                    │
         │                                                                                                    │
         │                                                                                                    │
       -6┤  *                                                                                                 │
         │                                                                                                    │
         │                                                                                                    │
         │                                                                                                    │
       -8┤                                                                                                    │
         │                                                                                                    │
         │ *                                                                                                  │
         │                                                                                                    │
         │                                                                                                    │
      -10┤                                                                                                    │
         │                                                                                                    │
         │                                                                                                    │
         │                                                                                                    │
      -12┤                                                                                                    │
         │                                                                                                    │
         │                                                                                                    │
         │                                                                                                    │
      -14┤                                                                                                    │
         │                                                                                                    │
         │                                                                                                    │
    -15.8┤*                                                                                                   │
         └┬───────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬──────────┬┘
          0.100  0.5       1.0       1.5       2.0       2.5       3.0       3.5       4.0       4.5       5.0
                                                           x
                                                                                                      ┌───────┐
                                                                                                      │# f(x) │
                                                                                                      │* f'(x)│
                                                                                                      │- 0    │
                                                                                                      └───────┘
    |}]

let%expect_test "tanh(x)" =
  let%op f x = tanh x in
  let plot_box = plot_unop ~f () in
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect
    {|
    f(x)
          ┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
     1.000┤                                                                                                   #│
          │                                                ** *                ############## #### ####### ### │
          │                                               *    *           ####                                │
          │                                              *      *        ##                                    │
       0.8┤                                             *        *      #                                      │
          │                                            *          *   ##                                       │
          │                                                          #                                         │
          │                                           *            *                                           │
       0.6┤                                          *              *                                          │
          │                                                        #                                           │
          │                                        *              #  *                                         │
          │                                        *                  *                                        │
       0.4┤                                                      #                                             │
          │                                       *             #      *                                       │
          │                                      *                      *                                      │
          │                                     *              #         *                                     │
       0.2┤                                   **                          **                                   │
          │                                 **                #             **                                 │
          │                               **                                  **                               │
          │                        * *****                  #                   ******                         │
       0.0┤**** *** *** *** ********-    -    -    -    -    -    -    -    -    -    ******* **** *******-****│
          │                                                 #                                                  │
          │                                                                                                    │
          │                                                #                                                   │
      -0.2┤                                                                                                    │
          │                                               #                                                    │
          │                                                                                                    │
          │                                              #                                                     │
      -0.4┤                                             #                                                      │
          │                                                                                                    │
          │                                            #                                                       │
          │                                           #                                                        │
      -0.6┤                                          #                                                         │
          │                                                                                                    │
          │                                        #                                                           │
          │                                       ##                                                           │
      -0.8┤                                      #                                                             │
          │                                    ##                                                              │
          │                                ####                                                                │
    -1.000┤#### ### ### ### ######## ######                                                                    │
          └┬────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬┘
           -5      -4        -3        -2        -1         0         1         2         3         4         5
                                                            x
                                                                                                       ┌───────┐
                                                                                                       │# f(x) │
                                                                                                       │* f'(x)│
                                                                                                       │- 0    │
                                                                                                       └───────┘
    |}]

let%expect_test "uint4x32_to_prec_uniform(x)" =
  let%op f x = uint4x32_to_prec_uniform x in
  let plot_box = plot_unop ~f () in
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect
    {|
    f(x)
         ┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
    0.752┤#                                                                                                   │
         │ ### ### ### ### ######## ############### ########                                                  │
         │                                                                                                    │
     0.70┤                                                                                                    │
         │                                                                                                    │
         │                                                                                                    │
     0.65┤                                                                                                    │
         │                                                                                                    │
     0.60┤                                                                                                    │
         │                                                                                                    │
         │                                                                                                    │
     0.55┤                                                                                                    │
         │                                                                                                    │
         │                                                                                                    │
     0.50┤                                                                                                    │
         │                                                                                                    │
     0.45┤                                                                                                    │
         │                                                                                                    │
         │                                                                                                    │
     0.40┤                                                                                                    │
         │                                                                                                    │
     0.35┤                                                                                                    │
         │                                                                                                    │
         │                                                                                                    │
     0.30┤                                                                                                    │
         │                                                                                                    │
         │                                                                              #### #### ####### ####│
     0.25┤                                                 # ###########################                      │
         │                                                                                                    │
     0.20┤                                                                                                    │
         │                                                                                                    │
         │                                                                                                    │
     0.15┤                                                                                                    │
         │                                                                                                    │
     0.10┤                                                                                                    │
         │                                                                                                    │
         │                                                                                                    │
     0.05┤                                                                                                    │
         │                                                                                                    │
     0.00┤**** *** *** *** ********-*************** ********-******************************* **** *******-****│
         └┬────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬┘
          -5      -4        -3        -2        -1         0         1         2         3         4         5
                                                           x
                                                                                                      ┌───────┐
                                                                                                      │# f(x) │
                                                                                                      │* f'(x)│
                                                                                                      │- 0    │
                                                                                                      └───────┘
    |}]

let%expect_test "where(x < 0, sin(x), cos(x))" =
  let%op f x = where (x < !.0.) (sin x) (cos x) in
  let plot_box = plot_unop ~f () in
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect
    {|
    f(x)
          ┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
     1.000┤  #                                                                                             *   │
          │#### ##                                        *** ##                                        **  ***│
          │      #                                      **      ##                                     *       │
          │       #                                    *          #                                   *        │
       0.8┤         #                                                                                *         │
          │          #                                *            #                                *          │
          │                                          *              #                                          │
          │           #                            *                 #                             *           │
       0.6┤           #                                                                          *             │
          │                                        *                  #                                        │
          │             #                         *                    #                         *             │
          │                                                                                                    │
       0.4┤              #                       *                      #                       *              │
          │               #                                                                    *               │
          │*                                    *                        #                                    #│
          │               #                                                                   *                │
       0.2┤ *                                  *                          #                                  # │
          │                 #                                                               *                  │
          │  *                                *                            #                                #  │
          │                  #                                                              *                  │
       0.0┤- *  -    -    -    -    -    -   *-    -    -    -    -    -    #    -    -    -    -    -    -#   │
          │                   #                             *                              *                   │
          │   *                             *                                #                           #     │
          │                    #                           #  *                           *                    │
      -0.2┤     *                          *                                  #                          #     │
          │                     #                         #    *                         *                     │
          │      *                        *                                    #                        #      │
          │                      #                       #      *                       *                      │
      -0.4┤      *                       *                                      #                      #       │
          │                       #     *               #        *               #     *                       │
          │       *                #                   #          *                   *               #        │
          │         *                  *                                          #                  #         │
      -0.6┤                        #                  #            *                 *                         │
          │          *                *              #              *              #                #          │
          │           *              *                                              *              #           │
          │                        *  #            #                 *             * #                         │
      -0.8┤           *            *   #           #                  *           *   #          #             │
          │             *         *     #         #                    *         *     #         #             │
          │              **      *       #       #                      *       *       #      ##              │
    -1.000┤               * *****         #######                        *******         #### #                │
          └┬────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬┘
           -5      -4        -3        -2        -1         0         1         2         3         4         5
                                                            x
                                                                                                       ┌───────┐
                                                                                                       │# f(x) │
                                                                                                       │* f'(x)│
                                                                                                       │- 0    │
                                                                                                       └───────┘
    |}]
