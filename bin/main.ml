open Base
(* open Matplotlib *)
module L = Lacaml.S

module F = Caml.Format
module LF = Lacaml.Io

let () =
  let m = 5 in
  let n = 3 in

  let data_mat = L.Mat.create m n in
  let data_mat_copy = L.Mat.create m n in
  let res_len = max 1 (max m n) in
  let res_mat = L.Mat.create_mvec res_len in
  let res_mat_copy = L.Mat.create_mvec res_len in

  for i = 1 to m do
    let v_ref = ref 0.0 in

    for j = 1 to n do
      let randf = Random.float 200.0 -. 100.0 in
      v_ref := !v_ref +. Float.of_int j *. randf;
      data_mat.{i, j} <- randf;
      data_mat_copy.{i, j} <- randf;
    done;

    let v = !v_ref in
    res_mat.{i, 1} <- v;
    res_mat_copy.{i, 1} <- v;
  done;

  F.printf
    "\
      @[<2>Predictor variables:\n\
        @\n\
        %a@]\n\
      @\n\
      @[<2>Response variable:\n\
        @\n\
        %a@]@\n\
      @\n"
    LF.pp_fmat data_mat
    LF.pp_rfvec (L.Mat.col res_mat 1);

  let rank = L.gelsd data_mat res_mat in

  F.printf
    "\
      @[<2>Regression weights:\n\
        @\n\
        %a@]\n\
      @\n\
      Rank: %d@\n@\n"
    LF.pp_rfvec (L.Mat.col res_mat 1)
    rank;

  let y = L.gemv data_mat_copy (L.Mat.col res_mat 1) in
  let b = L.Mat.col res_mat_copy 1 in

  F.printf
    "\
      @[<2>Check result (must be close to 0):\n\
        @\n\
        %a@]@\n"
    LF.pp_rfvec (L.Vec.sub y b)

(* 
(* Copy of https://github.com/LaurentMazare/ocaml-matplotlib/blob/master/examples/pyplot.ml *)
let () =
  let xs = List.init 120 ~f:Float.of_int in
  let ys1 = List.map xs ~f:(fun i -> Float.sin (i /. 20.)) in
  let ys2 = List.map xs ~f:(fun i -> Float.cos (i /. 12.)) in
  let xs = Array.of_list xs in
  let ys1 = Array.of_list ys1 in
  let ys2 = Array.of_list ys2 in
  Pyplot.xlabel "x";
  Pyplot.ylabel "y";
  Pyplot.grid true;
  Pyplot.plot ~color:Red ~xs ys1;
  Pyplot.plot ~color:Green ~linestyle:Dotted ~linewidth:2. ~xs ys2;
  Pyplot.fill_between ~alpha:0.3
    xs ys1 (Array.create ~len:(Array.length ys1) 0.);
  Pyplot.legend ~labels:[|"$y=\\sin(x/20)$"; "$y=\\cos(x/12)$"|] ();
  Mpl.savefig "test.png";
  let data = Mpl.plot_data `png in
  Stdio.Out_channel.write_all "test2.png" ~data;
  (* Mpl.set_backend (Mpl.Backend.Other "Qt5Agg"); *)
  Mpl.set_backend (Mpl.Backend.Default);
  Mpl.show ()
   *)