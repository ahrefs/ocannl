(** Losses and the training loop. *)

let hinge_loss m y = Model.O.(m >>| fun score -> !/(!.1.0 - y * score))


(* 
~/ocannl$ dune utop

open Base
#load "_build/default/lib/ocannl.cma"
open Ocannl
module F = Formula
let d = [|3; 3|]
let o = [|3; 1|]
let res_mlp3 = let open Model in O.(
    nonlinear ~w:(!~"w1" d) ~b:(!~"b1" d) %+> nonlinear ~w:(!~"w2" d) ~b:(!~"b2" d) %+>
    linear ~w:(!~"w3" o) ~b:(!~"b3" o))
let loss = Model.O.(Train.hinge_loss res_mlp3 !.7.0 @@ !~"x" d)
let () = Stdio.print_endline @@ fst @@ F.sprint loss.toplevel_forward
let () = Stdio.print_endline @@ fst @@ F.sprint loss.toplevel_backprop
*)