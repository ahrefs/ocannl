(** A network (module) is a [Formula.t], or a function that takes [Formula.t]s and outputs a [Formula.t],
    while maintaining an index of trainable parameters. *)
open Base

(* FIXME(28): implement [promote_precision] effects. *)

(** Composable network components, with parameter tracking. *)
type 'a t = {
  apply: 'a apply;
  params: (Formula.t, Formula.comparator_witness) Set.t;
  promote_precision: Ndcode.precision option;
  (** The precision at which the network's computation should happen, regardless of the precisions
      of the inputs and results, unless otherwise specified in subnetworks. *)
}
and _ apply =
| Nullary: Formula.t -> Formula.t apply
| Unary: (Formula.t -> Formula.t) -> (Formula.t -> Formula.t) apply
| Binary: (Formula.t -> Formula.t -> Formula.t) -> (Formula.t -> Formula.t -> Formula.t) apply

let apply_val (type a) (n : a t): a =
  match n.apply with
  | Nullary f -> f
  | Unary f -> f
  | Binary f -> f

let partial (type a) (n : (Formula.t -> a) t) x: a t =
  match n.apply with
  | Unary f -> {n with apply=Nullary (f x)}
  | Binary f -> {n with apply=Unary (f x)}

let apply_nullary ?promote_precision (type a) (f: (Formula.t -> a) t) (x: Formula.t t): a t =
  let apply = 
    match f.apply, x.apply with
    | Unary f, Nullary x -> (Nullary (f x): a apply)
    | Binary f, Nullary x -> Unary (f x) in
  let params = Set.union f.params x.params in
  {apply; params; promote_precision}

let compose ?promote_precision (type a) (f: (Formula.t -> a) t) (g: (Formula.t -> a) t): (Formula.t -> a) t =
  let apply = 
    match f.apply, g.apply with
    | Unary f, Unary g -> (Unary (fun x -> f (g x)): (Formula.t -> a) apply)
    | Binary f, Binary g -> (Binary (fun x y -> f (g x y) y): (Formula.t -> a) apply) in
  let params = Set.union f.params g.params in
  {apply; params; promote_precision}

let swap (type a) (f: (Formula.t -> Formula.t -> a) t) =
  let apply = 
    match f.apply with
    | Binary f -> (Binary (fun x y -> f y x): (Formula.t -> Formula.t -> a) apply) in
  {f with apply}

let bind_ret (type a) (m: a t) (f: Formula.t -> a) =
  let apply = 
    match m.apply with
    | Nullary g -> (Nullary (f g): a apply)
    | Unary g -> Unary (fun x -> f (g x) x)
    | Binary g -> Binary (fun x y -> f (g x y) x y) in
  {m with apply}

module FO = Operation.O

let residual_compose ?promote_precision (type a) (f: (Formula.t -> a) t) (g: (Formula.t -> a) t): (Formula.t -> a) t =
  let apply = 
    match f.apply, g.apply with
    | Unary f, Unary g -> (Unary FO.(fun x -> let y = g x in f y + y): (Formula.t -> a) apply)
    | Binary f, Binary g -> (Binary FO.(fun x y -> let z = g x y in f z y + z): (Formula.t -> a) apply) in
  let params = Set.union f.params g.params in
  {apply; params; promote_precision}

let sum_over_params n ~f = Set.sum (module Operation.Summable) n.params ~f

module O = struct
  include Operation.O
  let (@@) m x = apply_nullary m x
  let ( % ) = compose
  let (%+) = residual_compose
  (* This is like [(>>>)] from arrows, but [(%>)] is neater. *)
  let ( %> ) m1 m2 = compose m2 m1
  let (%+>) m1 m2 = residual_compose m2 m1
  let (>>|) = bind_ret
end
