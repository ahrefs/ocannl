open Base

open Matplotlib

module E = Ocannl.Engine

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
