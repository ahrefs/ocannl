open Base
open Ocannl
module FDSL = Operation.FDSL

let () = Session.SDSL.set_executor OCaml

let%expect_test "Micrograd README basic example" =
  (* let open Operation.FDSL in *)
  let open Session.SDSL in
  drop_session();
  Random.init 0;
  let%nn_op c = "a" [-4] + "b" [2] in
  let%nn_op d = a *. b + b **. 3 in
  let%nn_op c = c + c + 1 in
  let%nn_op c = c + 1 + c + ~-a in
  let%nn_op d = d + d *. 2 + !/ (b + a) in
  let%nn_op d = d + 3 *. d + !/ (b - a) in
  let%nn_op e = c - d in
  let%nn_op f = e *. e in
  let%nn_op g = f /. 2 in
  let%nn_op g = g + 10. /. f in

  refresh_session ();
  print_formula ~with_code:false ~with_grad:false `Default @@ g;
  [%expect {|
    ┌────────────────┐
    │[49]: shape 0:1 │
    │┌┬─────────┐    │
    │││axis 0   │    │
    │├┼─────────┼─── │
    │││ 2.47e+1 │    │
    │└┴─────────┘    │
    └────────────────┘ |}];
  print_formula ~with_code:false ~with_grad:true `Default @@ a;
  [%expect {|
    ┌───────────────┐
    │[1]: shape 0:1 │
    │┌┬──────────┐  │
    │││axis 0    │  │
    │├┼──────────┼─ │
    │││ -4.00e+0 │  │
    │└┴──────────┘  │
    └───────────────┘
    ┌─────────────────────────┐
    │[1]: shape 0:1  Gradient │
    │┌┬─────────┐             │
    │││axis 0   │             │
    │├┼─────────┼──────────── │
    │││ 1.39e+2 │             │
    │└┴─────────┘             │
    └─────────────────────────┘ |}];
  print_formula ~with_code:false ~with_grad:true `Default @@ b;
  [%expect {|
    ┌───────────────┐
    │[2]: shape 0:1 │
    │┌┬─────────┐   │
    │││axis 0   │   │
    │├┼─────────┼── │
    │││ 2.00e+0 │   │
    │└┴─────────┘   │
    └───────────────┘
    ┌─────────────────────────┐
    │[2]: shape 0:1  Gradient │
    │┌┬─────────┐             │
    │││axis 0   │             │
    │├┼─────────┼──────────── │
    │││ 6.46e+2 │             │
    │└┴─────────┘             │
    └─────────────────────────┘ |}]


let%expect_test "Micrograd half-moons example" =
  (* let open Operation.FDSL in *)
  let open Session.SDSL in
  drop_session();
  Random.init 0;
  let len = 100 in
  let batch = 10 in
  let noise() = Random.float_range (-0.1) 0.1 in
  let moons_flat = Array.concat_mapi (Array.create ~len ()) ~f:Float.(fun i () ->
    let v = of_int i * pi / of_int len in
    let c = cos v and s = sin v in
    [|c + noise(); s + noise(); 1.0 - c + noise(); 0.5 - s + noise()|]) in
  let moons_classes = Array.init (len*2) ~f:(fun i -> if i % 2 = 0 then 1. else (-1.)) in
  let moons_input = FDSL.data ~label:"moons_input" ~batch_dims:[batch] ~output_dims:[2]
      (Init_op (Fixed_constant moons_flat)) in
  let moons_class = FDSL.data ~label:"moons_class" ~batch_dims:[batch] ~output_dims:[1]
      (Init_op (Fixed_constant moons_classes)) in
  let points1 = ref [] in
  let points2 = ref [] in
  for _step = 1 to 2 * len/batch do
    refresh_session ();
    let points = NodeUI.retrieve_2d_points ~xdim:0 ~ydim:1 moons_input.node.node.value in
    let classes = NodeUI.retrieve_1d_points ~xdim:0 moons_class.node.node.value in
    let npoints1, npoints2 = Array.partitioni_tf points ~f:Float.(fun i _ -> classes.(i) > 0.) in
    points1 := npoints1 :: !points1;
    points2 := npoints2 :: !points2;
  done;
  let plot_box = 
    let open PrintBox_utils in
    plot ~size:(75, 35) ~x_label:"ixes" ~y_label:"ygreks"
      [Scatterplot {points=Array.concat !points1; pixel="#"}; 
       Scatterplot {points=Array.concat !points2; pixel="%"}] in
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect {|
     1.091e+0 │                            #
              │                    #
              │                 ##   ##   #      #
              │              #   #     #  #
              │            ##     #   #     # ##  # #
              │          #   ##    ##          #   # #   #
              │       #  #  #                   # #  #
              │         #  #                      #   #   #
              │        #                              ## #
              │     #   #
              │    #  #                                 #
              │      ##                                  ##
              │   #                   %                   # #
              │                                           #  #                           %
    y         │  # ##                 %  %                                          %
    g         │ #                      %%                  #                            %
    r         │ ##                      %%%                  ###                     %
    e         │# #                     %                        #                    %
    k         │   #                                         #  #                   %%%%%
    s         │ #                         %                     #                 %% %
              │   #                    %                      ## #
              │   #                       %                                        % %
              │##                       %% %%                   #                %
              │                           %   %               #                   %%
              │                          %    %                                 %%
              │                                %                              %     %
              │                                 %%                              %   %
              │                               %  %                           %  %
              │                              %
              │                                     %                     %  %%
              │                                  %  % % %              %  %
              │                                   %%   % %  %  %%  %      %   %
              │                                   %  %%      %    % % %%  %
              │                                       %        % % %    %
     -5.843e-1│                                             %%    %  %
    ──────────┼───────────────────────────────────────────────────────────────────────────
              │-1.068e+0                                                          2.095e+0
              │                                   ixes |}];

  let%nn_op mlp x = "b3" 1 + "w3" * !/ ("b2" 16 + "w2" * !/ ("b1" 16 + "w1" * x)) in
  let minus_lr = FDSL.data ~label:"minus_lr" ~batch_dims:[] ~output_dims:[1]
      Float.(Compute_point (fun ~session_step ~dims:_ ~idcs:_ ->
          0.9 * of_int session_step / 100. - 1.)) in
  (* Although [mlp] is not yet applied to anything, we can already compile the weight updates,
     because the parameters are already created by parameter punning. *)
  let update_weights = update_params ~minus_lr () in
  let%nn_op reg_loss = w1 **. 2 + w2 **. 2 + w3 **. 2 + b1 **. 2 + b2 **. 2 + b3 **. 2 in
  let%nn_op margin_loss = !/ (1 - moons_class *. mlp moons_input) in
  let%nn_op _total_loss = margin_loss + 0.0001 *. reg_loss in
  for step = 1 to 2 * len/batch do
    refresh_session (); update_weights ();
  done;
  close_session ();
  let point = [|0.; 0.|] in
  let point_input = FDSL.data ~label:"point_input" ~batch_dims:[1] ~output_dims:[2]
      (Compute_point (fun ~session_step:_ ~dims:_ ~idcs -> point.(idcs.(1)))) in
  let mlp_result = mlp point_input in
  let callback (x, y) =
    point.(0) <- x; point.(1) <- y;
    refresh_session ();
    let result = NodeUI.retrieve_1d_points ~xdim:0 mlp_result.node.node.value in
    Float.(result.(0) >= 0.) in
  let plot_box = 
    let open PrintBox_utils in
    plot ~size:(75, 35) ~x_label:"ixes" ~y_label:"ygreks"
      [Scatterplot {points=Array.concat !points1; pixel="#"}; 
       Scatterplot {points=Array.concat !points2; pixel="%"};
       Boundary_map {pixel_false="."; pixel_true="*"; callback}] in
  PrintBox_text.output Stdio.stdout plot_box;
  [%expect {| |}]
