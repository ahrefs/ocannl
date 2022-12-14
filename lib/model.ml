(** A model (module) is a function that takes one or more [Formula.t]s and outputs a [Formula.t],
    while maintaining an index of trainable parameters.
    
    When need arises we can make the type of [t.apply] more general. *)
open Base

type t = {
  apply: Formula.t -> Formula.t;
  params: Formula.t Sequence.t;
}
let linear ~w ~b =
  let apply x = Formula.O.(w*x + b) in
  let params = Sequence.of_list [w; b] in
  {apply; params}

let nonlinear ~w ~b =
  let apply x = Formula.O.(!/(w*x + b)) in
  let params = Sequence.of_list [w; b] in
  {apply; params}

let compose m1 m2 =
  let apply x = m1.apply @@ m2.apply x in
  let params = Sequence.merge_deduped_and_sorted m1.params m2.params
      ~compare:(fun f1 f2 -> String.compare f1.comp_node.label f2.comp_node.label) in
  {apply; params}

let residual_compose m1 m2 =
  let apply x = let z = m2.apply x in Formula.O.(m1.apply z + z)  in
  let params = Sequence.merge_deduped_and_sorted m1.params m2.params
      ~compare:(fun f1 f2 -> String.compare f1.comp_node.label f2.comp_node.label) in
  {apply; params}

let bind m f =
  let apply x = f @@ m.apply x in
  {apply; params=m.params}


module O = struct
  include Formula.O
  let (@@) m x = m.apply x
  let ( % ) = compose
  let (%+) = residual_compose
  (* This is [(>>>)] from arrows, but [(%>)] is neater. *)
  let ( %> ) m1 m2 = compose m2 m1
  let (%+>) m1 m2 = residual_compose m2 m1
  let (@>) = bind
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