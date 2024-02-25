open Base
module Nd = Ndarray

type memory_mode =
  | Virtual  (** The tensor node's computations are inlined on a per-scalar basis. *)
  | Never_virtual  (** One of: [Local], [On_device], [Hosted]. *)
  | Local
      (** The full tensor node is cached for the duration of a computation but not persisted across
          calls to jitted functions. It is not available for merging across devices. *)
  | Device_only  (** One of: [Local], [On_device]. *)
  | On_device
      (** The tensor node is stored on the devices that compute with it and persisted across function
          calls. It is available for merging across devices (for devices that support merging / P2P),
          but not (directly) for visualization or storing to disk. *)
  | Materialized  (** One of: [On_device], [Hosted]. *)
  | Hosted
      (** The tensor node is stored in a globally addressable memory, in addition to on devices
          where it is computed with (or as part of one of them, if "hosting on device").
          It is available for all operations, and visible to OCaml programs as an {!Ndarray}
          (the optional [array] of {!t}). *)
[@@deriving sexp, compare, equal]

type t = {
  array : (Nd.t option Lazy.t[@sexp.opaque]);
  prec : Ops.prec;
  dims : (int array Lazy.t[@sexp.opaque]);
  id : int;
  label : string list;
      (** Display information. It is better if the last element of the list is the most narrow
          or alphanumeric, e.g. an identifier. *)
  mutable memory_mode : (memory_mode * int) option;
  mutable backend_info : Sexp.t;
}
[@@deriving sexp_of]

let name { id; _ } = "n" ^ Int.to_string id
let label a = String.concat ~sep:"_" a.label
let compare a1 a2 = compare_int a1.id a2.id

let is_hosted_exn tn =
  match tn.memory_mode with
  | None -> invalid_arg @@ "Tnode.is_hosted_exn: memory_mode for " ^ label tn ^ " not inferred yet"
  | Some ((Virtual | Local | Device_only | On_device), _) -> false
  | Some (Hosted, _) -> true
  | Some ((Never_virtual | Materialized), _) ->
      invalid_arg @@ "Tnode.is_hosted_exn: memory_mode for " ^ label tn ^ " not fully inferred"

let known_not_materialized tn = match tn.memory_mode with Some ((Virtual | Local), _) -> true | _ -> false
let known_non_virtual tn = match tn.memory_mode with None | Some (Virtual, _) -> false | _ -> true

let update_memory_mode tn mode provenance =
  match (tn.memory_mode, mode) with
  | None, _ -> tn.memory_mode <- Some (mode, provenance)
  | Some (m1, _), m2 when equal_memory_mode m1 m2 -> ()
  | Some (Never_virtual, prov2), Virtual ->
      raise
      @@ Ndarray.User_error
           [%string
             "Tnode.update_memory_mode: update %{prov2#Int} -> %{provenance#Int} for %{name tn}: cannot be \
              virtual"]
  | Some (Never_virtual, _), mode -> tn.memory_mode <- Some (mode, provenance)
  | Some (Virtual, prov2), Never_virtual ->
      raise
      @@ Ndarray.User_error
           [%string
             "Tnode.update_memory_mode: update %{prov2#Int} -> %{provenance#Int} for %{name tn} is already \
              virtual"]
  | Some (_, _), Never_virtual -> ()
  | Some (Device_only, _), (Local | On_device) -> tn.memory_mode <- Some (mode, provenance)
  | Some (Materialized, _), (On_device | Hosted) -> tn.memory_mode <- Some (mode, provenance)
  | Some ((Local | On_device), _), Device_only -> ()
  | Some ((On_device | Hosted), _), Materialized -> ()
  | Some (Device_only, _), Materialized | Some (Materialized, _), Device_only ->
      tn.memory_mode <- Some (On_device, provenance)
  | Some (_, prov2), _ ->
      invalid_arg
        [%string
          "Tnode.update_memory_mode: update %{prov2#Int} -> %{provenance#Int} inconsistent for %{name tn}"]

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
  | _ -> invalid_arg @@ "Tnode.get_exn: array " ^ name a ^ " is not hosted"

let has a = match a.array with (lazy (Some _)) -> true | _ -> false

let dims_to_string ?(with_axis_numbers = false) arr =
  let dims_s =
    if Lazy.is_val arr.dims then Nd.int_dims_to_string ~with_axis_numbers @@ Lazy.force arr.dims
    else "<not-in-yet>"
  in
  Ops.prec_string arr.prec ^ " prec " ^ dims_s

let ident_label arr =
  let is_alphanum_ = String.for_all ~f:(fun c -> Char.equal c '_' || Char.is_alphanum c) in
  let components = List.filter arr.label ~f:(fun i -> is_alphanum_ i && not (String.equal i "grad")) in
  if List.is_empty components then None else Some (String.concat ~sep:"_" components)

let debug_name ~id ~label =
  let n = "n" ^ Int.to_string id in
  let ident_label =
    let is_alphanum_ = String.for_all ~f:(fun c -> Char.equal c '_' || Char.is_alphanum c) in
    let components = List.filter label ~f:(fun i -> is_alphanum_ i && not (String.equal i "grad")) in
    if List.is_empty components then None else Some (String.concat ~sep:"_" components)
  in
  let is_grad = List.mem ~equal:String.equal label "grad" in
  let opt_grad = if is_grad then ".grad" else "" in
  match ident_label with
  | Some ident -> [%string "%{ident}%{opt_grad}"]
  | None when is_grad -> [%string "n%{id - 1#Int}%{opt_grad}"]
  | None -> n

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

let header arr =
  let mem_size =
    if Lazy.is_val arr.array then
      match arr.array with
      | (lazy None) -> "<not-hosted>"
      | (lazy (Some nd)) -> Int.to_string_hum @@ Nd.size_in_bytes nd
    else "<not-in-yet>"
  in
  let repeating_idents = Hashtbl.create ~size:1 (module String) in
  [%string
    {|%{name arr} %{label arr} as %{
      styled_ident ~repeating_idents `Heuristic_ocannl arr
    }: %{dims_to_string arr}; mem in bytes: %{mem_size}|}]

module Registry = Core.Weak.Make (struct
  type nonrec t = t

  let equal = equal
  let hash = hash
end)

let registry = Registry.create 16

let create prec ~id ~label ~dims init_op =
  let rec array =
    lazy (if is_hosted_exn tn then Some (Nd.create_array prec ~dims:(Lazy.force dims) init_op) else None)
  and tn = { array; prec; id; label; memory_mode = None; backend_info = Sexp.List []; dims } in
  Registry.add registry tn;
  tn

let print_accessible_headers () =
  Stdio.printf "Tnode: collecting accessible arrays...%!\n";
  Core.Gc.full_major ();
  Registry.iter (fun arr -> Stdio.print_endline @@ header arr) registry;
  Stdio.printf "Tnode: Finished printing headers.%!\n"

module Debug_runtime = Utils.Debug_runtime

let%debug_sexp log_accessible_headers () =
  Core.Gc.full_major ();
  Registry.iter (fun arr -> [%log header arr]) registry
