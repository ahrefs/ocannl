open Base
module Nd = Ndarray

type t = {
  array : Nd.t option Lazy.t;
  prec : Ops.prec;
  dims : int array Lazy.t;
  id : int;
  label : string list;
      (** Display information. It is better if the last element of the list is the most narrow
          or alphanumeric, e.g. an identifier. *)
  hosted : bool option ref;
  mutable virtual_ : (bool * int) option;
      (** If true, this array is never materialized, its computations are inlined on a per-scalar basis.
          A array that is hosted will not be virtual. *)
  mutable device_only : (bool * int) option;
      (** If true, this node is only materialized on the devices it is computed on.
          It is marked as [not !(nd.hosted)]. *)
  mutable backend_info : string;
}

let is_false opt = not @@ Option.value ~default:true @@ Option.map ~f:fst opt
let is_true opt = Option.value ~default:false @@ Option.map ~f:fst opt
let isnt_false opt = Option.value ~default:true @@ Option.map ~f:fst opt
let isnt_true opt = not @@ Option.value ~default:false @@ Option.map ~f:fst opt
let name { id; _ } = "n" ^ Int.to_string id
let label a = String.concat ~sep:"_" a.label
let compare a1 a2 = compare_int a1.id a2.id
let sexp_of_t a = Sexp.Atom (name a ^ "_" ^ label a)

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
  String.concat [ name arr; " "; label arr; ": "; dims_to_string arr; "; mem in bytes: "; mem_size ]

let ident_label arr =
  let is_alphanum_ = String.for_all ~f:(fun c -> Char.equal c '_' || Char.is_alphanum c) in
  let components = List.filter arr.label ~f:(fun i -> is_alphanum_ i && not (String.equal i "grad")) in
  if List.is_empty components then None else Some (String.concat ~sep:"_" components)

let styled_ident ~repeating_idents style arr =
  let n = name arr in
  match style with
  | `Name_only -> n
  | `Name_and_label ->
      let label = label arr in
      if String.is_empty label then n else [%string "%{n}_%{label}"]
  | `Heuristic_ocannl -> (
      let is_grad = List.mem ~equal:String.equal arr.label "grad" in
      let opt_grad = if is_grad then ".grad" else "" in
      match ident_label arr with
      | Some ident ->
          if Hashtbl.mem repeating_idents ident then
            if is_grad then [%string "n%{arr.id - 1#Int}_%{ident}%{opt_grad}"]
            else [%string "n%{arr.id#Int}_%{ident}"]
          else [%string "%{ident}%{opt_grad}"]
      | None when is_grad -> [%string "n%{arr.id - 1#Int}%{opt_grad}"]
      | None -> n)

module Registry = Core.Weak.Make (struct
  type nonrec t = t

  let equal = equal
  let hash = hash
end)

let registry = Registry.create 16

let create prec ~id ~label ~dims init_op =
  let hosted = ref None in
  let array =
    lazy
      (if Option.value_exn !hosted then Some (Nd.create_array prec ~dims:(Lazy.force dims) init_op) else None)
  in
  let arr =
    { array; prec; id; label; hosted; virtual_ = None; device_only = None; backend_info = ""; dims }
  in
  Registry.add registry arr;
  arr

let print_accessible_headers () =
  Stdio.printf "Lazy_array: collecting accessible arrays...%!\n";
  Core.Gc.full_major ();
  Registry.iter (fun arr -> Stdio.print_endline @@ header arr) registry;
  Stdio.printf "Lazy_array: Finished printing headers.%!\n"
