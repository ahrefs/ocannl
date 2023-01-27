(** A network (module) is a function that takes a [Formula.t] and outputs a [Formula.t],
    while maintaining an index of trainable parameters. *)
open Base

type t = {
  apply: Formula.t -> Formula.t;
  params: (Formula.t, Formula.comparator_witness) Set.t;
  promote_precision: Ndarray.precision option;
  (** The precision at which the network's computation should happen, regardless of the precisions
      of the inputs and results. *)
}
(* FIXME(28): implement [promote_precision] effects. *)

module FO = Operation.O

let linear ?promote_precision ~w ~b =
  let apply x = FO.(w*x + b) in
  let params = Set.of_list (module Formula) [w; b] in
  {apply; params; promote_precision}

let nonlinear ?promote_precision ~w ~b =
  let apply x = FO.(!/(w*x + b)) in
  let params = Set.of_list (module Formula) [w; b] in
  {apply; params; promote_precision}

let compose ?promote_precision m1 m2 =
  let apply x = m1.apply @@ m2.apply x in
  let params = Set.union m1.params m2.params in
  {apply; params; promote_precision}

let residual_compose ?promote_precision m1 m2 =
  let apply x = let z = m2.apply x in FO.(m1.apply z + z)  in
  let params = Set.union m1.params m2.params in
  {apply; params; promote_precision}

let bind_ret m f =
  let apply x = f @@ m.apply x in
  {apply; params=m.params; promote_precision=m.promote_precision}

module O = struct
  include Operation.O
  let (@@) m x = m.apply x
  let ( % ) = compose
  let (%+) = residual_compose
  (* This is like [(>>>)] from arrows, but [(%>)] is neater. *)
  let ( %> ) m1 m2 = compose m2 m1
  let (%+>) m1 m2 = residual_compose m2 m1
  let (>>|) = bind_ret
end

(* 
~/ocannl$ dune utop

open Base
#load "_build/default/lib/ocannl.cma"
open Ocannl
module F = Formula
let d = [|3; 3|]
let res_mlp3 = let open Network in O.(
    nonlinear ~w:(!~"w1" d) ~b:(!~"b1" d) %+> nonlinear ~w:(!~"w2" d) ~b:(!~"b2" d) %+>
    linear ~w:(!~"w3" d) ~b:(!~"b3" d))
let y = Network.O.(res_mlp3 @@ !~"x" d)
let () = Stdio.print_endline @@ fst @@ F.sprint y.toplevel_forward
let () = Stdio.print_endline @@ fst @@ F.sprint y.toplevel_backprop
*)