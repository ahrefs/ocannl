(** A network (module) is a [F.t], or a function that takes [F.t]s and outputs a [F.t],
    while maintaining an index of trainable parameters. *)
open Base
module F = Formula

(* FIXME(28): implement [promote_precision] effects. *)

(** Composable network components, with parameter tracking. *)
type 'a t = {
  comp: 'a comp;
  (** The parametric computation. *)
  params: (F.t, F.comparator_witness) Set.t;
  mutable promote_precision: Ndcode.precision option;
  (** The precision at which the network's computation should happen, regardless of the precisions
      of the inputs and results, unless otherwise specified in subnetworks. *)
}
and _ comp =
| Nullary: F.t -> F.t comp
| Unary: (F.t -> F.t) -> (F.t -> F.t) comp
| Binary: (F.t -> F.t -> F.t) -> (F.t -> F.t -> F.t) comp

let unpack (type a) (n : a t): a =
  match n.comp with
  | Nullary f -> f
  | Unary f -> f
  | Binary f -> f

let apply (type a) (f: (F.t -> a) t) (x: F.t t): a t =
  let comp = 
    match f.comp, x.comp with
    | Unary f, Nullary x -> (Nullary (f x): a comp)
    | Binary f, Nullary x -> Unary (f x) in
  let params = Set.union f.params x.params in
  {comp; params; promote_precision=None}

(** Note that [apply_two f x1 x2 = apply (apply f x1) x2]. *)
let apply_two (type a) (f: (F.t -> F.t -> a) t) (x1: F.t t) (x2: F.t t): a t =
  let comp = 
    match f.comp, x1.comp, x2.comp with
    | Binary f, Nullary x1, Nullary x2 -> (Nullary (f x1 x2): a comp) in
  let params = Set.union_list (module F) [f.params; x1.params; x2.params] in
  {comp; params; promote_precision=None}

let compose (type a) (f: (F.t -> a) t) (g: (F.t -> a) t): (F.t -> a) t =
  let comp = 
    match f.comp, g.comp with
    | Unary f, Unary g -> (Unary (fun x -> f (g x)): (F.t -> a) comp)
    | Binary f, Binary g -> (Binary (fun x y -> f (g x y) y): (F.t -> a) comp) in
  let params = Set.union f.params g.params in
  {comp; params; promote_precision=None}

let swap (type a) (f: (F.t -> F.t -> a) t) =
  let comp = 
    match f.comp with
    | Binary f -> (Binary (fun x y -> f y x): (F.t -> F.t -> a) comp) in
  {f with comp}

let bind_ret (type a) (m: a t) (f: F.t -> a) =
  let comp = 
    match m.comp with
    | Nullary g -> (Nullary (f g): a comp)
    | Unary g -> Unary (fun x -> f (g x) x)
    | Binary g -> Binary (fun x y -> f (g x y) x y) in
  {m with comp}

(** Caution! [return_term] should only be used with terminals. *)
let return_term x =
  let params =
    if x.F.needs_gradient then Set.singleton (module F) x else Set.empty (module F) in
  {comp=Nullary x; params; promote_precision=None}

let return c =
  {comp=c; params=Set.empty (module F); promote_precision=None}
  
module FO = Operation.O

let residual_compose (type a) (f: (F.t -> a) t) (g: (F.t -> a) t): (F.t -> a) t =
  let comp = 
    match f.comp, g.comp with
    | Unary f, Unary g -> (Unary FO.(fun x -> let y = g x in f y + y): (F.t -> a) comp)
    | Binary f, Binary g -> (Binary FO.(fun x y -> let z = g x y in f z y + z): (F.t -> a) comp) in
  let params = Set.union f.params g.params in
  {comp; params; promote_precision=None}

let sum_over_params n ~f = Set.sum (module Operation.Summable) n.params ~f

module O = struct
  include Operation.O
  let (@@) m x = apply m x
  let ( % ) = compose
  let (%+) = residual_compose
  (* This is like [(>>>)] from arrows, but [(%>)] is neater. *)
  let ( %> ) m1 m2 = compose m2 m1
  let (%+>) m1 m2 = residual_compose m2 m1
  let (>>|) = bind_ret
end
