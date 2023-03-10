(** A network (module) is a [F.t], or a function that takes [F.t]s and outputs a [F.t],
    while maintaining an index of trainable parameters. *)
open Base
module F = Formula

(* FIXME(#28): implement [promote_precision] effects. *)
(** The computation payload of a network component. The computations are suspended, so that as we build
    a network component, we partially evaluate it to collect its parameters. *)
type _ comp =
| Placeholder: F.t list ref -> F.t comp
(** A placeholder is an "unfulfilled input" for constructing a network. Placeholders are lifted into
    function arguments. The list type enables building network components recursively (like in an
    unrolled-for-backprop RNN or a fixed-depth universal transformer). *)
| Suspended: (unit -> F.t) -> F.t comp
(** Suspension to enable building a network component before applying it to potentially multiple sets
    of inputs (replication with shared parameters). *)
| Nullary: F.t -> F.t comp
| Unary: (F.t -> F.t) -> (F.t -> F.t) comp
| Binary: (F.t -> F.t -> F.t) -> (F.t -> F.t -> F.t) comp

(** Composable network components, with parameter tracking. A single instance of [Network.t] can be
    reused multiple times in the same model (i.e. parameter sharing), or can be used in multiple
    simultaneusly or consecutively trained models (i.e. model surgery), but it carries a single instance
    of parameters. *)
type 'a t = {
  comp: 'a comp;
  (** The parametric computation. *)
  params: (F.t, F.comparator_witness) Set.t;
  (* mutable promote_precision: Ocannl_runtime.Node.precision option; *)
  (* * The precision at which the network's computation should happen, regardless of the precisions
      of the inputs and results, unless otherwise specified in subnetworks. *)
}

let unpack (type a) (n : a t): a =
  match n.comp with
  | Placeholder {contents=[]} -> invalid_arg "Network.unpack: encountered an empty placeholder"
  | Placeholder {contents=m::_} -> m
  | Suspended f -> f ()
  | Nullary f -> f
  | Unary f -> f
  | Binary f -> f

let apply (type a) (f: (F.t -> a) t) (x: F.t t): a t =
  let comp = 
    match f.comp, x.comp with
    | Unary f, Nullary x -> (Suspended (fun () -> f x): a comp) (* [f] might be binary+placeholder. *)
    | Unary f, Placeholder x -> (Suspended (fun () -> f (List.hd_exn !x)): a comp)
    | Unary f, Suspended x -> (Suspended (fun () -> f (x ())): a comp)
    | Binary f, Nullary x -> Unary (f x)
    | Binary f, Placeholder x -> (Unary (fun y -> f (List.hd_exn !x) y): a comp)
    | Binary f, Suspended x -> (Unary (fun y -> f (x ()) y): a comp) in
  let params = Set.union f.params x.params in
  {comp; params}

let compose (type a) (f: (F.t -> a) t) (g: (F.t -> a) t): (F.t -> a) t =
  let comp = 
    match f.comp, g.comp with
    | Unary f, Unary g -> (Unary (fun x -> f (g x)): (F.t -> a) comp)
    | Binary f, Binary g -> (Binary (fun x y -> f (g x y) y): (F.t -> a) comp) in
  let params = Set.union f.params g.params in
  {comp; params}

let swap (type a) (f: (F.t -> F.t -> a) t) =
  let comp = 
    match f.comp with
    | Binary f -> (Binary (fun x y -> f y x): (F.t -> F.t -> a) comp) in
  {f with comp}

let bind_ret (type a) (m: a t) (f: F.t -> a) =
  let comp = 
    match m.comp with
    | Placeholder x -> (Suspended ((fun () -> f (List.hd_exn !x))): a comp)
    | Suspended x -> (Suspended ((fun () -> f (x ()))): a comp)
    | Nullary x -> (Nullary (f x): a comp)
    | Unary g -> Unary (fun x -> f (g x) x)
    | Binary g -> Binary (fun x y -> f (g x y) x y) in
  {m with comp}

(** Caution! [return_term] should only be used with terminals, to not treat computation results
    as parameters. *)
let return_term x =
  let params =
    if x.F.needs_gradient then Set.singleton (module F) x else Set.empty (module F) in
  {comp=Nullary x; params}

let return c =
  {comp=c; params=Set.empty (module F)}
  
module FO = Operation.O

let residual_compose (type a) (f: (F.t -> a) t) (g: (F.t -> a) t): (F.t -> a) t =
  let comp = 
    match f.comp, g.comp with
    | Unary f, Unary g -> (Unary FO.(fun x -> let y = g x in f y + y): (F.t -> a) comp)
    | Binary f, Binary g -> (Binary FO.(fun x y -> let z = g x y in f z y + z): (F.t -> a) comp) in
  let params = Set.union f.params g.params in
  {comp; params}

let sum_over_params n ~f = Set.sum (module Operation.Summable) n.params ~f

module O = struct
  let ( * ) = return (Binary Operation.matmul)
  let ( *. ) = return (Binary Operation.pointmul)
  let (+) = return (Binary Operation.add)
  let (!/) = return (Unary Operation.relu)
  let (-) = return (Binary Operation.O.(-))
  let (~-) = return (Unary Operation.O.(~-))
  let (/) = return (Binary Operation.O.(/))
  let (/.) = return (Binary Operation.O.(/.))

  let (@@) m x = apply m x
  let ( % ) = compose
  let (%+) = residual_compose
  (* This is like [(>>>)] from arrows, but [(%>)] is neater. *)
  let ( %> ) m1 m2 = compose m2 m1
  let (%+>) m1 m2 = residual_compose m2 m1
  let (>>|) = bind_ret
end
