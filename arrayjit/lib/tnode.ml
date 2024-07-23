open Base
module Lazy = Utils.Lazy
module Nd = Ndarray
module Debug_runtime = Utils.Debug_runtime

[%%global_debug_log_level Nothing]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

type task = {
  description : string;
  work : (module Minidebug_runtime.Debug_runtime) -> unit -> unit;
}
[@@deriving sexp_of]

let run debug_runtime task =
  let module Debug_runtime = (val debug_runtime : Minidebug_runtime.Debug_runtime) in
  [%diagn_sexp
    [%log_entry
      task.description;
      task.work debug_runtime ()]]

type memory_type =
  | Constant  (** The tensor node does not change after initialization. *)
  | Nonconstant  (** One of: [Changed_on_devices], [Volatile]. *)
  | Changed_on_devices  (** The tensor node will only change on host via a [to_host] call. *)
  | Volatile
      (** The tensor node will only change on any device via a [from_host] call possibly followed by
          [device_to_device]. *)
[@@deriving sexp, compare, equal]

type memory_mode =
  | Effectively_constant  (** Either [Hosted Constant], or a subset of [Virtual]. *)
  | Virtual  (** The tensor node's computations are inlined on a per-scalar basis. *)
  | Never_virtual  (** One of: [Local], [On_device], [Hosted]. *)
  | Local
      (** The full tensor node is cached for the duration of a computation but not persisted across
          calls to compiled functions. It is not available for merging across devices. *)
  | Device_only  (** One of: [Local], [On_device]. *)
  | On_device
      (** The tensor node is stored on the devices that compute with it and persisted across
          function calls. It is available for merging across devices (for devices that support
          merging / P2P), but not (directly) for visualization or storing to disk. *)
  | Materialized  (** One of: [On_device], [Hosted]. *)
  | Hosted of memory_type
      (** The tensor node is stored in a globally addressable memory, in addition to on devices
          where it is computed with (or as part of one of them, if "hosting on device", or only on
          the host and not on devices, for some backends). It is available for all operations, and
          visible to OCaml programs as an {!Ndarray} (the optional [array] of {!t}). *)
[@@deriving sexp, compare, equal]

type t = {
  array : (Nd.t option Lazy.t[@sexp.opaque]);
  prec : Ops.prec;
  dims : (int array Lazy.t[@sexp.opaque]);
  id : int;
  label : string list;
      (** Display information. It is better if the last element of the list is the most narrow or
          alphanumeric, e.g. an identifier. *)
  mutable memory_mode : (memory_mode * int) option;
  mutable backend_info : Sexp.t;
  mutable code_name : string option;
}
[@@deriving sexp_of]

let compare a1 a2 = compare_int a1.id a2.id

let num_elems tn =
  let dims = Lazy.force tn.dims in
  if Array.is_empty dims then 0 else Array.reduce_exn dims ~f:( * )

let size_in_bytes tn = num_elems tn * Ops.prec_in_bytes tn.prec
let id { id; _ } = "n" ^ Int.to_string id
let label a = String.concat ~sep:"_" a.label
let is_alphanum_ = String.for_all ~f:(fun c -> Char.equal c '_' || Char.is_alphanum c)

let get_debug_name ?code_name ~id ~label () =
  match code_name with
  | Some code_name -> (
      match String.chop_suffix code_name ~suffix:"_grad" with
      | None -> code_name
      | Some ident -> ident ^ ".grad")
  | None -> (
      let components = List.filter ~f:is_alphanum_ label in
      let components, is_grad =
        match List.rev components with
        | "grad" :: components -> (List.rev components, true)
        | _ -> (components, false)
      in
      let ident_label =
        if List.is_empty components then None else Some (String.concat ~sep:"_" components)
      in
      let opt_grad = if is_grad then ".grad" else "" in
      match ident_label with
      | Some ident -> [%string "%{ident}%{opt_grad}"]
      | None when is_grad -> [%string "n%{id - 1#Int}%{opt_grad}"]
      | None -> "n" ^ Int.to_string id)

let debug_name tn =
  let id = tn.id and label = tn.label and code_name = tn.code_name in
  get_debug_name ?code_name ~id ~label ()

let default_to_most_local tn provenance =
  match tn.memory_mode with
  | None | Some (Effectively_constant, _) -> tn.memory_mode <- Some (Virtual, provenance)
  | Some (Never_virtual, _) -> tn.memory_mode <- Some (Local, provenance)
  | Some (Device_only, _) -> tn.memory_mode <- Some (Local, provenance)
  | Some (Materialized, _) -> tn.memory_mode <- Some (On_device, provenance)
  | Some ((Virtual | Local | On_device | Hosted _), _) -> ()

let is_virtual_force tn provenance =
  default_to_most_local tn provenance;
  match tn.memory_mode with Some (Virtual, _) -> true | _ -> false

let is_hosted_force ?specifically tn provenance =
  default_to_most_local tn provenance;
  match (tn.memory_mode, specifically) with
  | None, _ -> assert false
  | Some ((Virtual | Local | Device_only | On_device), _), _ -> false
  | Some (Hosted _, _), None -> true
  | Some (Hosted memtyp, _), Some query -> equal_memory_type memtyp query
  | Some ((Never_virtual | Materialized | Effectively_constant), _), _ -> assert false

let is_materialized_force tn provenance =
  default_to_most_local tn provenance;
  match tn.memory_mode with
  | None -> assert false
  | Some ((Virtual | Local), _) -> false
  | Some ((On_device | Hosted _ | Materialized), _) -> true
  | Some ((Never_virtual | Device_only | Effectively_constant), _) -> assert false

let known_not_materialized tn =
  match tn.memory_mode with Some ((Virtual | Local), _) -> true | _ -> false

let known_constant tn =
  match tn.memory_mode with
  | Some ((Effectively_constant | Hosted Constant), _) -> true
  | _ -> false

let known_non_virtual tn =
  match tn.memory_mode with None | Some ((Virtual | Effectively_constant), _) -> false | _ -> true

let known_not_param tn =
  match tn.memory_mode with
  | Some
      ( ( Virtual | Local | Effectively_constant | Device_only | On_device
        | Hosted (Constant | Volatile) ),
        _ ) ->
      true
  | _ -> false

let mode_is_unspecified tn =
  match tn.memory_mode with
  | None | Some ((Never_virtual | Effectively_constant), _) -> true
  | _ -> false

let update_memory_mode tn mode provenance =
  match (tn.memory_mode, mode) with
  | None, _ -> tn.memory_mode <- Some (mode, provenance)
  | Some (m1, _), m2 when equal_memory_mode m1 m2 -> ()
  | Some (Never_virtual, prov2), Virtual ->
      raise
      @@ Utils.User_error
           [%string
             "Tnode.update_memory_mode: update %{prov2#Int} -> %{provenance#Int} for %{debug_name \
              tn}: cannot be virtual"]
  | Some ((Virtual | Hosted Constant), _), Effectively_constant -> ()
  | Some ((Never_virtual | Materialized), _), Effectively_constant
  | Some (Effectively_constant, _), (Never_virtual | Materialized | Hosted Constant) ->
      tn.memory_mode <- Some (Hosted Constant, provenance)
  | Some (Effectively_constant, _), Virtual -> tn.memory_mode <- Some (mode, provenance)
  | Some (Hosted Nonconstant, _), Hosted (Changed_on_devices | Volatile) ->
      tn.memory_mode <- Some (mode, provenance)
  | Some (Hosted (Changed_on_devices | Volatile), _), Hosted Nonconstant -> ()
  | Some (Never_virtual, _), mode -> tn.memory_mode <- Some (mode, provenance)
  | Some (Virtual, prov2), Never_virtual ->
      raise
      @@ Utils.User_error
           [%string
             "Tnode.update_memory_mode: update %{prov2#Int} -> %{provenance#Int} for %{debug_name \
              tn} is already virtual"]
  | Some (_, _), Never_virtual -> ()
  | Some (Device_only, _), (Local | On_device) -> tn.memory_mode <- Some (mode, provenance)
  | Some (Materialized, _), (On_device | Hosted _) -> tn.memory_mode <- Some (mode, provenance)
  | Some ((Local | On_device), _), Device_only -> ()
  | Some ((On_device | Hosted _), _), Materialized -> ()
  | Some (Device_only, _), Materialized | Some (Materialized, _), Device_only ->
      tn.memory_mode <- Some (On_device, provenance)
  | Some (_, prov2), _ ->
      invalid_arg
        [%string
          "Tnode.update_memory_mode: update %{prov2#Int} -> %{provenance#Int} inconsistent for \
           %{debug_name tn}"]

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
  | _ -> invalid_arg @@ "Tnode.get_exn: array " ^ debug_name a ^ " is not hosted"

let has a = match a.array with (lazy (Some _)) -> true | _ -> false

let dims_to_string ?(with_axis_numbers = false) arr =
  let dims_s =
    if Lazy.is_val arr.dims then Nd.int_dims_to_string ~with_axis_numbers @@ Lazy.force arr.dims
    else "<not-in-yet>"
  in
  Ops.prec_string arr.prec ^ " prec " ^ dims_s

let ident_label tn =
  let components =
    List.filter tn.label ~f:(fun i -> is_alphanum_ i && not (String.equal i "grad"))
  in
  if List.is_empty components then None else Some (String.concat ~sep:"_" components)

let styled_ident ~repeating_nograd_idents ~repeating_grad_idents style arr =
  let n = id arr in
  match style with
  | `Name_only -> n
  | `Name_and_label ->
      let label = label arr in
      if String.is_empty label then n else [%string "%{n}_%{label}"]
  | `Heuristic_ocannl grad_sep -> (
      let is_grad = List.mem ~equal:String.equal arr.label "grad" in
      let opt_grad =
        match (grad_sep, is_grad) with
        | `Dot_grad, true -> ".grad"
        | `Under_grad, true -> "_grad"
        | (`Dot_grad | `Under_grad), _ -> ""
      in
      match ident_label arr with
      | Some ident ->
          if Hashtbl.mem (if is_grad then repeating_grad_idents else repeating_nograd_idents) ident
          then
            if is_grad then [%string "n%{arr.id - 1#Int}_%{ident}%{opt_grad}"]
            else [%string "n%{arr.id#Int}_%{ident}"]
          else [%string "%{ident}%{opt_grad}"]
      | None when is_grad -> [%string "n%{arr.id - 1#Int}%{opt_grad}"]
      | None -> n)

let update_code_name tn ident =
  match tn.code_name with
  | None -> tn.code_name <- Some ident
  | Some old_name ->
      if String.length ident > String.length old_name || String.is_prefix ~prefix:(id tn) ident then
        tn.code_name <- Some ident

let get_style ?(arg_name = "ll_ident_style") ?(no_dots = false) () =
  match Utils.get_global_arg ~arg_name ~default:"heuristic" with
  | "heuristic" -> `Heuristic_ocannl (if no_dots then `Under_grad else `Dot_grad)
  | "name_and_label" -> `Name_and_label
  | "name_only" -> `Name_only
  | _ ->
      invalid_arg @@ "Wrong " ^ arg_name ^ ", must be one of: heuristic, name_and_label, name_only"

let header arr =
  let mem_size =
    if Lazy.is_val arr.array then
      match arr.array with
      | (lazy None) -> "<not-hosted>"
      | (lazy (Some nd)) -> Int.to_string_hum @@ Nd.size_in_bytes nd
    else "<not-in-yet>"
  in
  let repeating_nograd_idents = Hashtbl.create ~size:1 (module String) in
  let repeating_grad_idents = Hashtbl.create ~size:1 (module String) in
  [%string
    {|%{id arr} %{label arr} as %{
      styled_ident ~repeating_nograd_idents ~repeating_grad_idents (`Heuristic_ocannl `Dot_grad) arr
    }: %{dims_to_string arr}; mem in bytes: %{mem_size}|}]

module Registry = Core.Weak.Make (struct
  type nonrec t = t

  let equal = equal
  let hash = hash
end)

let registry = Registry.create 16

let create prec ~id ~label ~dims init_op =
  let rec array =
    lazy
      (if is_hosted_force tn 30 then Some (Nd.create_array prec ~dims:(Lazy.force dims) init_op)
       else None)
  and tn =
    {
      array;
      prec;
      dims;
      id;
      label;
      memory_mode = None;
      backend_info = Sexp.List [];
      code_name = None;
    }
  in
  Registry.add registry tn;
  tn

let find =
  let mock =
    {
      array = lazy None;
      prec = Ops.single;
      dims = lazy [||];
      id = -1;
      label = [];
      memory_mode = None;
      backend_info = Sexp.List [];
      code_name = None;
    }
  in
  fun ~id -> Registry.find_opt registry { mock with id }

let print_accessible_headers () =
  Stdio.printf "Tnode: collecting accessible arrays...%!\n";
  Core.Gc.full_major ();
  Registry.iter (fun arr -> Stdio.print_endline @@ header arr) registry;
  Stdio.printf "Tnode: Finished printing headers.%!\n"

let%debug_sexp log_accessible_headers () =
  Core.Gc.full_major ();
  Registry.iter (fun _arr -> [%log header _arr]) registry;
  ()
