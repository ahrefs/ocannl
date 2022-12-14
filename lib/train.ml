(** Losses and the training loop. *)
open Base
open Model

let hinge_loss m x y = O.(!/(!.1.0 - y * (m @@ x)))

let sum_over_params m ~f = Set.sum (module Formula.Summable) m.params ~f

let l2_reg_loss ~alpha m = O.(!.alpha * sum_over_params m ~f:(fun p -> p * p))


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
let loss = Model.O.(l2_reg_loss ~alpha:1e-4 res_mlp3 + (Train.hinge_loss res_mlp3 !.3.0 !.7.0))
let () = Stdio.print_endline @@ fst @@ F.sprint loss.toplevel_forward
let () = Stdio.print_endline @@ fst @@ F.sprint loss.toplevel_backprop
*)