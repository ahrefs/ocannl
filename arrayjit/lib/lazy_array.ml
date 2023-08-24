open Base

module Nd = Ndarray

type t = {
  array : Nd.t option Lazy.t;
  prec : Nd.prec;
  dims : int array Lazy.t;
  id : int;
  label : string;  (** An optional display information. *)
  literal : bool;
  materialized : bool ref;
  mutable never_virtual : bool;
  mutable never_device_only : bool;
  mutable backend_info : string;
}

let name { id; _ } = "#" ^ Int.to_string id
let compare a1 a2 = compare_int a1.id a2.id
let sexp_of_t a = Sexp.Atom (name a)

include Comparator.Make (struct
  type nonrec t = t

  let compare = compare
  let sexp_of_t = sexp_of_t
end)

let equal a1 a2 = equal_int a1.id a2.id
let hash nd = Int.hash nd.id
let hash_fold_t acc nd = hash_fold_int acc nd.id
let hash_t = hash

let get_exn a =
  match a.array with
  | (lazy (Some nd)) -> nd
  | _ -> invalid_arg @@ "Lazy_array.get_exn: array " ^ name a ^ " is not materialized"

let has a = match a.array with (lazy (Some _)) -> true | _ -> false

let create prec ~id ~label ~dims ?(literal = false) init_op =
  let materialized = ref false in
  let array =
    lazy (if !materialized then Some (Nd.create_array prec ~dims:(Lazy.force dims) init_op) else None)
  in
  {
    array;
    prec;
    id;
    label;
    literal;
    materialized;
    never_virtual = false;
    never_device_only = false;
    backend_info = "";
    dims;
  }
