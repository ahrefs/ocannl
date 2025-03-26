open Base
module Lazy = Utils.Lazy
module Nd = Ndarray

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

(** A possible algorithm for deciding sharing within a single device:
    - If a tensor node is read-only for a context, and not otherwise recorded, it is stored as a
      cross-stream sharing candidate.
    - If a cross-stream sharing candidate is read-only for another context, whose parent does not
      have the corresponding array (i.e. it is a different stream), it is recorded as cross-stream
      shared, and the same array is reused.
    - If a tensor node is writable by a context, and it is not cross-stream shared, it is marked as
      non-cross-stream, the array is removed from cross-stream sharing candidates if present. If it
      is cross-stream shared, it is recorded as owned by the corresponding stream. It is an error if
      the node was already owned by a different stream.

    If a tensor node is shared cross-stream, within-device copying is a NOOP as source and
    destination pointers are in that case identical. *)
type sharing =
  | Unset  (** One of: [Per_stream], [Shared_cross_streams]. *)
  | Per_stream  (** The tensor node has separate arrays for each stream. *)
  | Shared_cross_streams
      (** The tensor node has a single array per device that can appear in multiple contexts, except
          for backends with [Option.is_some use_host_memory] and nodes with memory mode already
          [Hosted (Changed_on_devices Shared_cross_streams)] before first linking on a device, where
          it only has the on-host array. In that case the on-host array is registered in the
          context, to avoid misleading behavior from `device_to_device`. *)
[@@deriving sexp, compare, equal]

type memory_type =
  | Constant  (** The tensor node does not change after initialization. *)
  | Nonconstant  (** One of: [Changed_on_devices], [Volatile]. *)
  | Changed_on_devices of sharing
      (** The tensor node will only change on host via a [to_host] call. *)
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
  | On_device of sharing
      (** The tensor node is stored on the devices that compute with it and persisted across
          function calls. It is available for merging across devices (for devices that support
          merging / P2P), but not (directly) for visualization or storing to disk. *)
  | Materialized  (** One of: [On_device], [Hosted]. *)
  | Hosted of memory_type
      (** The tensor node is stored in a globally addressable memory, in addition to on devices
          where it is computed with (or only on the host and not on the device, for some backends).
          It is available for all operations, and visible to OCaml programs as an {!Ndarray} (the
          optional [array] of {!t}). *)
[@@deriving sexp, compare, equal]

type delayed_prec = Not_specified | Default_spec of Ops.prec Lazy.t | Specified of Ops.prec
[@@deriving sexp, equal]

type prepare = { is_done : unit -> bool; sync : unit -> unit; transfer : unit -> unit }
[@@deriving sexp_of]

type t = {
  array : Nd.t option Lazy.t;
  prec : Ops.prec Lazy.t;
  dims : int array Lazy.t;
  size_in_bytes : int Lazy.t;
  id : int;
  label : string list;
      (** Display information. It is better if the last element of the list is the most narrow or
          alphanumeric, e.g. an identifier. *)
  mutable delayed_prec_unsafe : delayed_prec;
      (** Participates in the computation of {!field-prec}. *)
  mutable memory_mode : (memory_mode * int) option;
  mutable backend_info : Sexp.t;
  mutable code_name : string option;
  mutable prepare_read : prepare option;
  mutable prepare_write : prepare option;
  mutable host_read_by_devices : Hash_set.M(Int).t;
      (** The unique ids of devices that read the most recent modification of the host array. *)
}
[@@deriving sexp_of]

let compare a1 a2 = compare_int a1.id a2.id

let num_elems tn =
  let dims = Lazy.force tn.dims in
  if Array.is_empty dims then 0 else Array.reduce_exn dims ~f:( * )

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
        match components with "grad" :: components -> (components, true) | _ -> (components, false)
      in
      let ident_label =
        if List.is_empty components then None else Some (String.concat ~sep:"_" components)
      in
      let opt_grad = if is_grad then ".grad" else "" in
      match ident_label with
      | Some ident -> [%string "%{ident}%{opt_grad}"]
      | None when is_grad -> [%string "n%{id - 1#Int}%{opt_grad}"]
      | None -> "n" ^ Int.to_string id)

let prepare ~is_done ~sync ~transfer old =
  match old with
  | None -> { is_done; sync; transfer }
  | Some old ->
      if old.is_done () then { is_done; sync; transfer }
      else
        {
          is_done = (fun () -> old.is_done () && is_done ());
          sync =
            (fun () ->
              old.sync ();
              sync ());
          transfer;
        }

let prepare_read ~is_done ~sync ~transfer tn =
  tn.prepare_read <- Some (prepare ~is_done ~sync ~transfer tn.prepare_read)

let prepare_write ~is_done ~sync tn =
  tn.prepare_write <- Some (prepare ~is_done ~sync ~transfer:(fun () -> ()) tn.prepare_write)

let debug_name tn =
  let id = tn.id and label = tn.label and code_name = tn.code_name in
  get_debug_name ?code_name ~id ~label ()

let debug_memory_mode = function
  | None -> "unknown"
  | Some (mem, prov) ->
      (match mem with
      | Effectively_constant -> "Const"
      | Virtual -> "Virt"
      | Never_virtual -> "Non-virt"
      | Local -> "Local"
      | Device_only -> "Dev"
      | Materialized -> "Material"
      | On_device Unset -> "On-dev"
      | On_device Shared_cross_streams -> "Dev-shared"
      | On_device Per_stream -> "Dev-stream"
      | Hosted Constant -> "Host-const"
      | Hosted Nonconstant -> "Host-non-const"
      | Hosted Volatile -> "Hosted"
      | Hosted (Changed_on_devices Unset) -> "Host&dev"
      | Hosted (Changed_on_devices Per_stream) -> "Host&stream"
      | Hosted (Changed_on_devices Shared_cross_streams) -> "Host&shared")
      ^ "/" ^ Int.to_string prov

let log_debug_info ~from_log_level tn =
  [%debug_sexp
    [%logN_block
      from_log_level (debug_name tn);
      [%log
        "id:",
        (tn.id : int),
        "label:",
        (tn.label : string list),
        "mem:",
        debug_memory_mode tn.memory_mode,
        "backends:",
        (tn.backend_info : Sexp.t)];
      if Lazy.is_val tn.array then
        match tn.array with
        | (lazy None) -> [%log "<not-on-host>"]
        | (lazy (Some nd)) -> Nd.log_debug_info ~from_log_level nd
      else [%log "<not-in-yet>"]]]

(** The one exception to "most local" is that the sharing property is kept at [Unset]. *)
let default_to_most_local tn provenance =
  match tn.memory_mode with
  | None | Some (Effectively_constant, _) -> tn.memory_mode <- Some (Virtual, provenance)
  | Some (Never_virtual, _) -> tn.memory_mode <- Some (Local, provenance)
  | Some (Device_only, _) -> tn.memory_mode <- Some (Local, provenance)
  | Some (Materialized, _) -> tn.memory_mode <- Some (On_device Unset, provenance)
  | Some ((Virtual | Local | On_device _ | Hosted _), _) -> ()

let is_virtual_force tn provenance =
  match tn.memory_mode with
  | Some (Virtual, _) -> true
  | None | Some (Effectively_constant, _) ->
      tn.memory_mode <- Some (Virtual, provenance);
      true
  | _ -> false

let rec is_hosted_force tn provenance =
  match tn.memory_mode with
  | Some ((Virtual | Local | Device_only | On_device _), _) -> false
  | Some (Hosted _, _) -> true
  | None | Some ((Never_virtual | Materialized | Effectively_constant), _) ->
      default_to_most_local tn provenance;
      is_hosted_force tn provenance

let rec is_materialized_force tn provenance =
  match tn.memory_mode with
  | None -> assert false
  | Some ((Virtual | Local), _) -> false
  | Some ((On_device _ | Hosted _ | Materialized), _) -> true
  | Some ((Never_virtual | Device_only | Effectively_constant), _) ->
      default_to_most_local tn provenance;
      is_materialized_force tn provenance

let%debug3_sexp rec is_in_context_force ~(use_host_memory : 'a option) (tn : t) (provenance : int) :
    bool =
  match tn.memory_mode with
  | Some (Hosted (Changed_on_devices Per_stream), _) -> true
  | Some ((Materialized | Hosted Nonconstant), _) when Option.is_none use_host_memory -> true
  | Some (Hosted (Constant | Volatile), _) when Option.is_some use_host_memory -> false
  | Some (Hosted _, _) -> true
  | Some ((Virtual | Local), _) -> false
  | None | Some ((Materialized | Effectively_constant | Never_virtual | Device_only), _) ->
      default_to_most_local tn provenance;
      is_in_context_force ~use_host_memory tn provenance
  | Some (On_device _, _) -> true

let known_not_materialized tn =
  match tn.memory_mode with Some ((Virtual | Local), _) -> true | _ -> false

let known_constant tn =
  match tn.memory_mode with
  | Some ((Effectively_constant | Hosted Constant), _) -> true
  | _ -> false

let known_volatile tn = match tn.memory_mode with Some (Hosted Volatile, _) -> true | _ -> false

let known_non_virtual tn =
  match tn.memory_mode with None | Some ((Virtual | Effectively_constant), _) -> false | _ -> true

let known_not_param tn =
  match tn.memory_mode with
  | Some
      ( ( Virtual | Local | Effectively_constant | Device_only | On_device _
        | Hosted (Constant | Volatile) ),
        _ ) ->
      true
  | _ -> false

let known_shared_cross_streams tn =
  match tn.memory_mode with
  | Some
      ( ( On_device Shared_cross_streams
        | Hosted (Constant | Volatile | Changed_on_devices Shared_cross_streams) ),
        _ ) ->
      true
  | _ -> false

let known_non_cross_stream tn =
  match tn.memory_mode with
  | Some ((On_device Per_stream | Hosted (Changed_on_devices Per_stream)), _) -> true
  | _ -> false

let potentially_cross_stream tn = not (known_not_materialized tn || known_non_cross_stream tn)

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
  | Some (Hosted Nonconstant, _), Hosted (Changed_on_devices _ | Volatile) ->
      tn.memory_mode <- Some (mode, provenance)
  | Some (Hosted (Changed_on_devices _ | Volatile), _), Hosted Nonconstant -> ()
  | Some (Never_virtual, _), mode -> tn.memory_mode <- Some (mode, provenance)
  | Some (Virtual, prov2), Never_virtual ->
      raise
      @@ Utils.User_error
           [%string
             "Tnode.update_memory_mode: update %{prov2#Int} -> %{provenance#Int} for %{debug_name \
              tn} is already virtual"]
  | Some (_, _), Never_virtual -> ()
  | Some (Device_only, _), (Local | On_device _) -> tn.memory_mode <- Some (mode, provenance)
  | Some (Materialized, _), (On_device _ | Hosted _) -> tn.memory_mode <- Some (mode, provenance)
  | Some ((Local | On_device _), _), Device_only -> ()
  | Some ((On_device _ | Hosted _), _), Materialized -> ()
  | Some (Device_only, _), Materialized | Some (Materialized, _), Device_only ->
      tn.memory_mode <- Some (On_device Unset, provenance)
  | Some (_, prov2), _ ->
      invalid_arg
        [%string
          "Tnode.update_memory_mode: update %{prov2#Int} -> %{provenance#Int} inconsistent for \
           %{debug_name tn}"]

(** [update_memory_sharing tn sharing provenance] preserves the memory mode of [tn] while updating
    the cross-stream sharing property, except that [Hosted Nonconstant] is further specialized to
    [Hosted (Changed_on_devices sharing)]. *)
let update_memory_sharing tn sharing provenance =
  match (tn.memory_mode, sharing) with
  | None, _ -> tn.memory_mode <- Some (On_device sharing, provenance)
  | Some (On_device Shared_cross_streams, _), Shared_cross_streams
  | Some (On_device Per_stream, _), Per_stream ->
      ()
  | Some ((On_device Unset | Device_only | Materialized), _), _ ->
      tn.memory_mode <- Some (On_device sharing, provenance)
  | Some (Hosted (Constant | Volatile), prov2), Per_stream ->
      raise
      @@ Utils.User_error
           [%string
             "Tnode.update_memory_sharing: update %{prov2#Int} -> %{provenance#Int} for \
              %{debug_name tn} (hosted) -- currently hosted nodes not changed on devices must be \
              shared cross-stream"]
  | Some (Hosted (Changed_on_devices Shared_cross_streams), _), Shared_cross_streams
  | Some (Hosted (Changed_on_devices Per_stream), _), Per_stream ->
      ()
  | Some (Hosted (Constant | Volatile), _), Shared_cross_streams -> ()
  | Some (Hosted (Nonconstant | Changed_on_devices Unset), _), _ ->
      tn.memory_mode <- Some (Hosted (Changed_on_devices sharing), provenance)
  | Some (_, prov2), Unset ->
      invalid_arg
        [%string
          "Tnode.update_memory_sharing: update %{prov2#Int} -> %{provenance#Int} for %{debug_name \
           tn} -- currently unsetting of sharing not allowed"]
  | (Some (_, prov2) as mem_mode), _ ->
      invalid_arg
        [%string
          "Tnode.update_memory_sharing: update %{prov2#Int} -> %{provenance#Int} inconsistent for \
           %{debug_name tn}: old mode %{debug_memory_mode mem_mode}, new sharing \
           %{Sexp.to_string_hum @@ sexp_of_sharing sharing}"]

let update_prec ?only_if tn prec =
  let do_update =
    match only_if with
    | None -> false
    | Some cond -> (
        match tn.delayed_prec_unsafe with
        | Specified old_prec -> cond old_prec
        | Default_spec old_prec when Lazy.is_val old_prec -> cond @@ Lazy.force old_prec
        | _ -> true)
  in
  if do_update then
    if Lazy.is_val tn.prec then (
      if not @@ Ops.equal_prec (Lazy.force tn.prec) prec then
        raise
        @@ Utils.User_error
             (String.concat
                [
                  "Tnode.update_prec: setting precision ";
                  Ops.prec_string prec;
                  " for ";
                  debug_name tn;
                  " but the settled precision is ";
                  Ops.prec_string (Lazy.force tn.prec);
                ]))
    else
      match (tn.delayed_prec_unsafe, only_if) with
      | Specified old_prec, _ when not @@ Ops.equal_prec old_prec prec ->
          raise
          @@ Utils.User_error
               (String.concat
                  [
                    "Tnode.update_prec: setting precision ";
                    Ops.prec_string prec;
                    " for ";
                    debug_name tn;
                    ", but the precision is already set to ";
                    Ops.prec_string (Lazy.force tn.prec);
                  ])
      | Default_spec old_prec, Some cond when not @@ Lazy.is_val old_prec ->
          tn.delayed_prec_unsafe <-
            Default_spec
              (lazy
                (let old = Lazy.force old_prec in
                 if cond old then prec else old))
      | _ -> tn.delayed_prec_unsafe <- Specified prec

let exceeds_fp16_cutoff tn c =
  match Utils.settings.check_half_prec_constants_cutoff with
  | None -> false
  | Some cutoff ->
      (* Only force if needed. *)
      Float.(abs c >= cutoff)
      &&
      let prec =
        if Lazy.is_val tn.prec then Lazy.force tn.prec
        else
          match tn.delayed_prec_unsafe with
          | Specified prec -> prec
          | Default_spec prec -> Lazy.force prec
          | Not_specified -> Lazy.force tn.prec
      in
      Ops.is_up_to_fp16 prec

include Comparator.Make (struct
  type nonrec t = t

  let compare = compare
  let sexp_of_t = sexp_of_t
end)

let equal a1 a2 = equal_int a1.id a2.id
let hash nd = Int.hash nd.id
let hash_fold_t acc nd = hash_fold_int acc nd.id
let hash_t = hash

module Comp = struct
  type nonrec t = t
  type nonrec comparator_witness = comparator_witness
end

type t_set = Set.M(Comp).t

let sexp_of_t_set s = [%sexp_of: t Sequence.t] @@ Set.to_sequence s

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
  Ops.prec_string (Lazy.force arr.prec) ^ " prec " ^ dims_s

let no_grad_ident_label tn =
  match List.filter tn.label ~f:(fun i -> is_alphanum_ i) with
  | [] -> (false, None)
  | [ "grad" ] -> (true, None)
  | "grad" :: components -> (true, Some (String.concat ~sep:"_" components))
  | components -> (false, Some (String.concat ~sep:"_" components))

let styled_ident ~repeating_nograd_idents ~repeating_grad_idents style arr =
  let n = id arr in
  match style with
  | `Name_only -> n
  | `Name_and_label ->
      let label = label arr in
      if String.is_empty label then n else [%string "%{n}_%{label}"]
  | `Heuristic_ocannl grad_sep -> (
      let is_grad, ident = no_grad_ident_label arr in
      let opt_grad =
        match (grad_sep, is_grad) with
        | `Dot_grad, true -> ".grad"
        | `Under_grad, true -> "_grad"
        | (`Dot_grad | `Under_grad), _ -> ""
      in
      match ident with
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
      if
        String.length ident > String.length old_name
        && not (String.is_prefix ~prefix:(id tn) old_name)
        || String.is_prefix ~prefix:(id tn) ident
      then tn.code_name <- Some ident

let get_style ?(arg_name = "ll_ident_style") ?(no_dots = false) () =
  match Utils.get_global_arg ~arg_name ~default:"heuristic" with
  | "heuristic" -> `Heuristic_ocannl (if no_dots then `Under_grad else `Dot_grad)
  | "name_and_label" -> `Name_and_label
  | "name_only" -> `Name_only
  | _ ->
      invalid_arg @@ "Wrong " ^ arg_name ^ ", must be one of: heuristic, name_and_label, name_only"

let header tn =
  let debug = Utils.settings.log_level > 0 in
  let mem_size =
    if Lazy.is_val tn.array then
      match tn.array with
      | (lazy None) -> "<not-hosted>"
      | (lazy (Some nd)) ->
          let size = Int.to_string_hum @@ Nd.size_in_bytes nd in
          if debug then size ^ " @ " ^ Nd.ptr_to_string_hum nd else size
    else "<not-in-yet>"
  in
  let repeating_nograd_idents = Hashtbl.create ~size:1 (module String) in
  let repeating_grad_idents = Hashtbl.create ~size:1 (module String) in
  [%string
    {|%{id tn} %{label tn} as %{
      styled_ident ~repeating_nograd_idents ~repeating_grad_idents (`Heuristic_ocannl `Dot_grad) tn
    }: %{debug_memory_mode tn.memory_mode}; %{dims_to_string tn}; mem in bytes: %{mem_size}%{
    if debug then "; debug: " ^ Sexp.to_string_hum tn.backend_info else ""}|}]

module Registry = Stdlib.Weak.Make (struct
  type nonrec t = t

  let equal = equal
  let hash = hash
end)

let registry = Registry.create 16

let create ?default_prec ~id ~label ~dims init_op =
  let debug = "Host array for " ^ get_debug_name ~id ~label () in
  let rec array =
    lazy
      (if is_hosted_force tn 30 then
         Some (Nd.create_array ~debug (Lazy.force prec) ~dims:(Lazy.force dims) init_op)
       else None)
  and prec =
    lazy
      (match tn.delayed_prec_unsafe with
      | Specified prec | Default_spec (lazy prec) -> prec
      | Not_specified ->
          raise @@ Utils.User_error "Tnode.update_prec: precision is not specified yet")
  and size_in_bytes = lazy (num_elems tn * Ops.prec_in_bytes (Lazy.force tn.prec))
  and tn =
    let delayed_prec_unsafe =
      match default_prec with None -> Not_specified | Some prec -> Default_spec prec
    in
    {
      array;
      delayed_prec_unsafe;
      prec;
      dims;
      size_in_bytes;
      id;
      label;
      memory_mode = None;
      backend_info = Sexp.List [];
      code_name = None;
      prepare_read = None;
      prepare_write = None;
      host_read_by_devices = Hash_set.create (module Int);
    }
  in
  (* Note: if tensor nodes get non-trivial finalizers, remember to either add an is_finalized flag
     that is checked in the find function, or to convert it to a find_exn function that should never
     be called on potentially GCed nodes. *)
  Registry.add registry tn;
  tn

let find =
  let mock =
    {
      array = lazy None;
      prec = lazy Ops.single;
      delayed_prec_unsafe = Specified Ops.single;
      dims = lazy [||];
      size_in_bytes = lazy 0;
      id = -1;
      label = [];
      memory_mode = None;
      backend_info = Sexp.List [];
      code_name = None;
      prepare_read = None;
      prepare_write = None;
      host_read_by_devices = Hash_set.create (module Int);
    }
  in
  fun ~id -> Registry.find_opt registry { mock with id }

(** {2 Accessors} *)

let do_read tn =
  Option.iter
    ~f:(fun p ->
      p.sync ();
      if Utils.settings.automatic_host_transfers then p.transfer ())
    tn.prepare_read;
  tn.prepare_read <- None

let do_write tn =
  Option.iter ~f:(fun p -> p.sync ()) tn.prepare_write;
  tn.prepare_write <- None;
  Hash_set.clear tn.host_read_by_devices

let points_1d ?from_axis ~xdim tn =
  do_read tn;
  Option.value_map ~default:[||] ~f:(fun arr -> Nd.retrieve_1d_points ?from_axis ~xdim arr)
  @@ Lazy.force tn.array

let points_2d ?from_axis ~xdim ~ydim tn =
  do_read tn;
  Option.value_map ~default:[||] ~f:(fun arr -> Nd.retrieve_2d_points ?from_axis ~xdim ~ydim arr)
  @@ Lazy.force tn.array

let set_value tn =
  do_write tn;
  Nd.set_from_float @@ Option.value_exn ~here:[%here] @@ Lazy.force tn.array

let get_value tn =
  do_read tn;
  Nd.get_as_float @@ Option.value_exn ~here:[%here] @@ Lazy.force tn.array

let set_values tn values =
  do_write tn;
  Nd.(
    reset (Constant_fill { values; strict = false })
    @@ Option.value_exn ~here:[%here]
    @@ Lazy.force tn.array)

let get_values tn =
  do_read tn;
  Nd.(retrieve_flat_values @@ Option.value_exn ~here:[%here] @@ Lazy.force tn.array)

let print_accessible_headers () =
  Stdio.printf "Tnode: collecting accessible arrays...%!\n";
  Stdlib.Gc.full_major ();
  Registry.iter (fun arr -> Stdio.print_endline @@ header arr) registry;
  Stdio.printf "Tnode: Finished printing headers.%!\n"

let%debug_sexp log_accessible_headers () =
  Stdlib.Gc.full_major ();
  Registry.iter (fun _arr -> [%log header _arr]) registry;
  ()
