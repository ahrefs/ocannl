(** A network (module) is a [F.t], or a function that takes [F.t]s and outputs a [F.t],
    while maintaining an index of trainable parameters. *)
open Base
module F = Formula

(* FIXME(28): implement [promote_precision] effects. *)

(** Composable network components, with parameter tracking. A single instance of [Network.t] can be
    reused multiple times in the same model (i.e. parameter sharing), or can be used in multiple
    simultaneusly or consecutively trained models (i.e. model surgery), but it carries a single instance
    of parameters. *)
type 'a t = {
  comp: 'a comp;
  (** The parametric computation. *)
  params: (F.t, F.comparator_witness) Set.t;
  mutable promote_precision: Code.precision option;
  (** The precision at which the network's computation should happen, regardless of the precisions
      of the inputs and results, unless otherwise specified in subnetworks. *)
}
and _ comp =
| Placeholder: F.t option ref -> F.t comp
(** A placeholder is an "unfulfilled input" for constructing a network. Placeholders are lifted into
    function arguments. *)
| Suspended: F.t Lazy.t -> F.t comp 
| Nullary: F.t -> F.t comp
| Unary: (F.t -> F.t) -> (F.t -> F.t) comp
| Binary: (F.t -> F.t -> F.t) -> (F.t -> F.t -> F.t) comp

let unpack (type a) (n : a t): a =
  match n.comp with
  | Placeholder {contents=None} -> invalid_arg "Network.unpack: encountered an empty placeholder"
  | Placeholder {contents=Some m} -> m
  | Suspended f -> Lazy.force f
  | Nullary f -> f
  | Unary f -> f
  | Binary f -> f

let apply (type a) (f: (F.t -> a) t) (x: F.t t): a t =
  let comp = 
    match f.comp, x.comp with
    | Unary f, Nullary x -> (Nullary (f x): a comp)
    | Unary f, Placeholder x -> (Suspended (lazy (f (Option.value_exn !x))): a comp)
    | Unary f, Suspended x -> (Suspended (lazy (f (Lazy.force x))): a comp)
    | Binary f, Nullary x -> Unary (f x)
    | Binary f, Placeholder x -> (Unary (fun y -> f (Option.value_exn !x) y): a comp)
    | Binary f, Suspended x -> (Unary (fun y -> f (Lazy.force x) y): a comp) in
  let params = Set.union f.params x.params in
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
    | Placeholder x -> (Suspended (lazy (f (Option.value_exn !x))): a comp)
    | Suspended x -> (Suspended (lazy (f (Lazy.force x))): a comp)
    | Nullary x -> (Nullary (f x): a comp)
    | Unary g -> Unary (fun x -> f (g x) x)
    | Binary g -> Binary (fun x y -> f (g x y) x y) in
  {m with comp}

(** Caution! [return_term] should only be used with terminals, to not treat computation results
    as parameters. *)
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
  let ( * ) = return (Binary Operation.matmul)
  let ( *. ) = return (Binary Operation.pointmul)
  let (+) = return (Binary Operation.add)
  let (!/) = return (Unary Operation.relu)
  let (-) = return (Binary Operation.O.(-))

  let (@@) m x = apply m x
  let ( % ) = compose
  let (%+) = residual_compose
  (* This is like [(>>>)] from arrows, but [(%>)] is neater. *)
  let ( %> ) m1 m2 = compose m2 m1
  let (%+>) m1 m2 = residual_compose m2 m1
  let (>>|) = bind_ret
end
