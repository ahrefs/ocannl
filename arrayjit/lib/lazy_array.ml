open Base
module Nd = Ndarray

type t = {
  array : Nd.t option Lazy.t;
  prec : Ops.prec;
  dims : int array Lazy.t;
  id : int;
  label : string;  (** An optional display information. *)
  literal : bool;
  hosted : bool ref;
  mutable virtual_ : (bool * int) option;
      (** If true, this array is never materialized, its computations are inlined on a per-scalar basis.
          A array that is hosted will not be virtual. *)
  mutable device_only : (bool * int) option;
      (** If true, this node is only materialized on the devices it is computed on, it is not persisted
          outside of a step update. It is marked as [not !(nd.hosted)]. *)
  mutable backend_info : string;
}

let is_false opt = not @@ Option.value ~default:true @@ Option.map ~f:fst opt
let is_true opt = Option.value ~default:false @@ Option.map ~f:fst opt
let isnt_false opt = Option.value ~default:true @@ Option.map ~f:fst opt
let isnt_true opt = not @@ Option.value ~default:false @@ Option.map ~f:fst opt

let name { id; _ } = "n" ^ Int.to_string id
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
  | _ -> invalid_arg @@ "Lazy_array.get_exn: array " ^ name a ^ " is not hosted"

let has a = match a.array with (lazy (Some _)) -> true | _ -> false

let dims_to_string ?(with_axis_numbers = false) arr =
  let dims_s =
    if Lazy.is_val arr.dims then Nd.int_dims_to_string ~with_axis_numbers @@ Lazy.force arr.dims
    else "<not-in-yet>"
  in
  Ops.prec_string arr.prec ^ " prec " ^ dims_s

let header arr =
  let mem_size =
    if Lazy.is_val arr.array then
      match arr.array with
      | (lazy None) -> "<not-hosted>"
      | (lazy (Some nd)) -> Int.to_string_hum @@ Nd.size_in_bytes nd
    else "<not-in-yet>"
  in
  String.concat [ name arr; " "; arr.label; ": "; dims_to_string arr; "; mem in bytes: "; mem_size ]

module Registry = Core.Weak.Make (struct
  type nonrec t = t

  let equal = equal
  let hash = hash
end)

let registry = Registry.create 16

let create prec ~id ~label ~dims ?(literal = false) init_op =
  let hosted = ref false in
  let array = lazy (if !hosted then Some (Nd.create_array prec ~dims:(Lazy.force dims) init_op) else None) in
  let arr =
    { array; prec; id; label; literal; hosted; virtual_ = None; device_only = None; backend_info = ""; dims }
  in
  Registry.add registry arr;
  arr

let print_accessible_headers () =
  Stdio.printf "Lazy_array: collecting accessible arrays...%!\n";
  Core.Gc.full_major ();
  Registry.iter (fun arr -> Stdio.print_endline @@ header arr) registry;
  Stdio.printf "Lazy_array: Finished printing headers.%!\n"
