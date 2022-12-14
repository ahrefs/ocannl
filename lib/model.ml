(** A model (module) is a function that takes one or more [Formula.t]s and outputs a [Formula.t],
    while maintaining an index of trainable parameters.
    
    When need arises we can make the type of [t.apply] more general. *)
open Base

type t = {
  apply: Formula.t -> Formula.t;
  params: (Formula.t, Formula.comparator_witness) Set.t;
}
let linear ~w ~b =
  let apply x = Formula.O.(w*x + b) in
  let params = Set.of_list (module Formula) [w; b] in
  {apply; params}

let nonlinear ~w ~b =
  let apply x = Formula.O.(!/(w*x + b)) in
  let params = Set.of_list (module Formula) [w; b] in
  {apply; params}

let compose m1 m2 =
  let apply x = m1.apply @@ m2.apply x in
  let params = Set.union m1.params m2.params in
  {apply; params}

let residual_compose m1 m2 =
  let apply x = let z = m2.apply x in Formula.O.(m1.apply z + z)  in
  let params = Set.union m1.params m2.params in
  {apply; params}

let bind_ret m f =
  let apply x = f @@ m.apply x in
  {apply; params=m.params}

module O = struct
  include Formula.O
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
let res_mlp3 = let open Model in O.(
    nonlinear ~w:(!~"w1" d) ~b:(!~"b1" d) %+> nonlinear ~w:(!~"w2" d) ~b:(!~"b2" d) %+>
    linear ~w:(!~"w3" d) ~b:(!~"b3" d))
let y = Model.O.(res_mlp3 @@ !~"x" d)
let () = Stdio.print_endline @@ fst @@ F.sprint y.toplevel_forward
let () = Stdio.print_endline @@ fst @@ F.sprint y.toplevel_backprop
*)