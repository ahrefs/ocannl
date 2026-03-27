open Base
module Tn = Ir.Tnode
module Nd = Ir.Ndarray
module Ops = Ir.Ops

(** {1 Checkpoint file format for tensor persistence} *)

(** Metadata for a single tensor in a checkpoint file. *)
type tensor_meta = {
  id : int;
  namespace : string;  (** Reserved for #372, always [""] for now. *)
  label : string list;
  prec : Ops.prec;
  dims : int array;  (** Padded (buffer) dimensions. *)
  padding : (Ops.axis_padding array * float option) option;
  offset : int;  (** Byte offset in the data section. *)
  byte_length : int;  (** Bytes in the data section for this tensor. *)
}

type checkpoint_header = {
  version : int;  (** Currently 1. *)
  tensors : tensor_meta list;
}

(** {2 S-expression serialization for checkpoint types} *)

(* Manual sexp conversion for tensor_meta since Ops.prec uses manual sexp *)
let sexp_of_tensor_meta m =
  Sexp.List
    [
      Sexp.List [ Sexp.Atom "id"; Sexp.Atom (Int.to_string m.id) ];
      Sexp.List [ Sexp.Atom "namespace"; Sexp.Atom m.namespace ];
      Sexp.List
        [
          Sexp.Atom "label";
          Sexp.List (List.map m.label ~f:(fun s -> Sexp.Atom s));
        ];
      Sexp.List [ Sexp.Atom "prec"; Ops.sexp_of_prec m.prec ];
      Sexp.List
        [
          Sexp.Atom "dims";
          Sexp.List (Array.to_list (Array.map m.dims ~f:(fun d -> Sexp.Atom (Int.to_string d))));
        ];
      Sexp.List
        [
          Sexp.Atom "padding";
          (match m.padding with
          | None -> Sexp.Atom "none"
          | Some (padding_arr, pad_val) ->
              Sexp.List
                [
                  Sexp.List
                    (Array.to_list
                       (Array.map padding_arr ~f:(fun Ops.{ left; right } ->
                            Sexp.List
                              [
                                Sexp.Atom (Int.to_string left);
                                Sexp.Atom (Int.to_string right);
                              ])));
                  (match pad_val with
                  | None -> Sexp.Atom "none"
                  | Some v -> Sexp.Atom (Float.to_string v));
                ]);
        ];
      Sexp.List [ Sexp.Atom "offset"; Sexp.Atom (Int.to_string m.offset) ];
      Sexp.List [ Sexp.Atom "byte_length"; Sexp.Atom (Int.to_string m.byte_length) ];
    ]

let tensor_meta_of_sexp sexp =
  let fields =
    match sexp with
    | Sexp.List fields ->
        List.map fields ~f:(function
          | Sexp.List [ Sexp.Atom key; value ] -> (key, value)
          | _ -> failwith "tensor_meta_of_sexp: expected (key value) pair")
    | _ -> failwith "tensor_meta_of_sexp: expected list"
  in
  let find key =
    match List.Assoc.find fields key ~equal:String.equal with
    | Some v -> v
    | None -> failwith ("tensor_meta_of_sexp: missing field " ^ key)
  in
  let id =
    match find "id" with Sexp.Atom s -> Int.of_string s | _ -> failwith "bad id"
  in
  let namespace =
    match find "namespace" with
    | Sexp.Atom s -> s
    | _ -> failwith "bad namespace"
  in
  let label =
    match find "label" with
    | Sexp.List atoms ->
        List.map atoms ~f:(function
          | Sexp.Atom s -> s
          | _ -> failwith "bad label element")
    | _ -> failwith "bad label"
  in
  let prec = Ops.prec_of_sexp (find "prec") in
  let dims =
    match find "dims" with
    | Sexp.List atoms ->
        Array.of_list
          (List.map atoms ~f:(function
            | Sexp.Atom s -> Int.of_string s
            | _ -> failwith "bad dim"))
    | _ -> failwith "bad dims"
  in
  let padding =
    match find "padding" with
    | Sexp.Atom "none" -> None
    | Sexp.List [ Sexp.List padding_sexps; pad_val_sexp ] ->
        let padding_arr =
          Array.of_list
            (List.map padding_sexps ~f:(function
              | Sexp.List [ Sexp.Atom l; Sexp.Atom r ] ->
                  Ops.{ left = Int.of_string l; right = Int.of_string r }
              | _ -> failwith "bad padding entry"))
        in
        let pad_val =
          match pad_val_sexp with
          | Sexp.Atom "none" -> None
          | Sexp.Atom s -> Some (Float.of_string s)
          | _ -> failwith "bad padding value"
        in
        Some (padding_arr, pad_val)
    | _ -> failwith "bad padding"
  in
  let offset =
    match find "offset" with
    | Sexp.Atom s -> Int.of_string s
    | _ -> failwith "bad offset"
  in
  let byte_length =
    match find "byte_length" with
    | Sexp.Atom s -> Int.of_string s
    | _ -> failwith "bad byte_length"
  in
  { id; namespace; label; prec; dims; padding; offset; byte_length }

let sexp_of_checkpoint_header h =
  Sexp.List
    [
      Sexp.List [ Sexp.Atom "version"; Sexp.Atom (Int.to_string h.version) ];
      Sexp.List
        [
          Sexp.Atom "tensors";
          Sexp.List (List.map h.tensors ~f:sexp_of_tensor_meta);
        ];
    ]

let checkpoint_header_of_sexp sexp =
  let fields =
    match sexp with
    | Sexp.List fields ->
        List.map fields ~f:(function
          | Sexp.List [ Sexp.Atom key; value ] -> (key, value)
          | _ -> failwith "checkpoint_header_of_sexp: expected (key value) pair")
    | _ -> failwith "checkpoint_header_of_sexp: expected list"
  in
  let find key =
    match List.Assoc.find fields key ~equal:String.equal with
    | Some v -> v
    | None -> failwith ("checkpoint_header_of_sexp: missing field " ^ key)
  in
  let version =
    match find "version" with
    | Sexp.Atom s -> Int.of_string s
    | _ -> failwith "bad version"
  in
  let tensors =
    match find "tensors" with
    | Sexp.List metas -> List.map metas ~f:tensor_meta_of_sexp
    | _ -> failwith "bad tensors"
  in
  { version; tensors }

(** {2 File I/O helpers} *)

let write_header oc header =
  let sexp = sexp_of_checkpoint_header header in
  let header_str = Sexp.to_string_hum sexp in
  let len = String.length header_str in
  Stdlib.output_binary_int oc len;
  Stdlib.output_string oc header_str

let read_header ic =
  let len =
    try Stdlib.input_binary_int ic
    with End_of_file -> failwith "read_header: unexpected end of file (header length)"
  in
  if len < 0 || len > 100_000_000 then
    failwith ("read_header: invalid header length: " ^ Int.to_string len);
  let buf = Bytes.create len in
  (try Stdlib.really_input ic buf 0 len
   with End_of_file -> failwith "read_header: unexpected end of file (header data)");
  let header_str = Bytes.to_string buf in
  let sexp = Sexplib.Sexp.of_string header_str in
  checkpoint_header_of_sexp sexp

let validate_header header =
  if header.version <> 1 then
    failwith ("unsupported checkpoint version: " ^ Int.to_string header.version);
  (* Check for duplicate IDs *)
  let ids = List.map header.tensors ~f:(fun m -> m.id) in
  let unique_ids = Set.of_list (module Int) ids in
  if Set.length unique_ids <> List.length ids then
    failwith "checkpoint contains duplicate tensor IDs"

(** Compute the byte length for a tensor's logical payload. *)
let compute_byte_length prec dims padding =
  let n_elems =
    if Array.is_empty dims then 1
    else
      Array.foldi dims ~init:1 ~f:(fun axis acc d ->
          match padding with
          | None -> acc * d
          | Some (padding_arr, _) when axis < Array.length padding_arr ->
              acc * (d - padding_arr.(axis).Ops.left - padding_arr.(axis).Ops.right)
          | Some _ -> acc * d)
  in
  n_elems * Ops.prec_in_bytes prec

(** {2 Public API} *)

let save ~appending t_set path =
  let tn_list = Set.to_list t_set in
  (* Validate all tnodes have hosted arrays and sync from device *)
  List.iter tn_list ~f:(fun tn ->
      Tn.do_read tn;
      match Lazy.force tn.Tn.array with
      | None ->
          failwith
            ("save: tensor " ^ Int.to_string tn.Tn.id ^ " has no hosted array")
      | Some _ -> ());
  (* Collect current tensor data *)
  let new_entries =
    List.map tn_list ~f:(fun tn ->
        let prec = Lazy.force tn.Tn.prec in
        let dims = Lazy.force tn.Tn.dims in
        let padding = Lazy.force tn.Tn.padding in
        let byte_length = compute_byte_length prec dims padding in
        let meta =
          {
            id = tn.Tn.id;
            namespace = "";
            label = tn.Tn.label;
            prec;
            dims;
            padding;
            offset = 0;  (* Will be computed later *)
            byte_length;
          }
        in
        (meta, tn))
  in
  (* If appending, read existing file and merge *)
  let all_entries =
    if appending && Stdlib.Sys.file_exists path then begin
      let ic = Stdlib.open_in_bin path in
      let existing_header = read_header ic in
      validate_header existing_header;
      (* Read the data offset for seeking *)
      let data_start = Stdlib.pos_in ic in
      (* Read existing binary payloads for non-overlapping tensors *)
      let new_ids =
        Set.of_list (module Int) (List.map new_entries ~f:(fun (m, _) -> m.id))
      in
      let kept_entries =
        List.filter_map existing_header.tensors ~f:(fun meta ->
            if Set.mem new_ids meta.id then None
            else begin
              (* Read the existing payload *)
              Stdlib.seek_in ic (data_start + meta.offset);
              let payload = Bytes.create meta.byte_length in
              (try Stdlib.really_input ic payload 0 meta.byte_length
               with End_of_file ->
                 failwith
                   ("save: failed to read existing payload for tensor "
                  ^ Int.to_string meta.id));
              Some (`Existing (meta, payload))
            end)
      in
      Stdlib.close_in ic;
      let new_tagged =
        List.map new_entries ~f:(fun (m, tn) -> `New (m, tn))
      in
      kept_entries @ new_tagged
    end
    else List.map new_entries ~f:(fun (m, tn) -> `New (m, tn))
  in
  (* Compute sequential offsets *)
  let _, entries_with_offsets =
    List.fold all_entries ~init:(0, []) ~f:(fun (offset, acc) entry ->
        let byte_length =
          match entry with
          | `Existing (m, _) -> m.byte_length
          | `New (m, _) -> m.byte_length
        in
        let entry_with_offset =
          match entry with
          | `Existing (m, payload) ->
              `Existing ({ m with offset }, payload)
          | `New (m, tn) -> `New ({ m with offset }, tn)
        in
        (offset + byte_length, entry_with_offset :: acc))
  in
  let entries_with_offsets = List.rev entries_with_offsets in
  (* Write to temp file, then rename for atomicity *)
  let tmp_path = path ^ ".tmp" in
  let oc = Stdlib.open_out_bin tmp_path in
  (match
     let header =
       {
         version = 1;
         tensors =
           List.map entries_with_offsets ~f:(function
             | `Existing (m, _) -> m
             | `New (m, _) -> m);
       }
     in
     write_header oc header;
     List.iter entries_with_offsets ~f:(function
       | `Existing (_, payload) -> Stdlib.output_bytes oc payload
       | `New (_, tn) ->
           let nd = Option.value_exn (Lazy.force tn.Tn.array) in
           let padding =
             Option.map ~f:fst (Lazy.force tn.Tn.padding)
           in
           let _n = Nd.write_payload_to_channel ?padding nd oc in
           ())
   with
  | () ->
      Stdlib.close_out oc;
      Stdlib.Sys.rename tmp_path path
  | exception exn ->
      Stdlib.close_out_noerr oc;
      (try Stdlib.Sys.remove tmp_path with _ -> ());
      raise exn)

let load ?prefix_namespace path =
  (match prefix_namespace with
  | None | Some "" -> ()
  | Some _ ->
      failwith
        "load: prefix_namespace is not yet supported (requires #372 namespaces)");
  let ic = Stdlib.open_in_bin path in
  let result =
    match
      let header = read_header ic in
      validate_header header;
      let data_start = Stdlib.pos_in ic in
      (* Pre-check: verify no ID clashes before creating anything *)
      List.iter header.tensors ~f:(fun meta ->
          match Tn.find ~id:meta.id with
          | Some _ ->
              failwith
                ("load: tensor with id " ^ Int.to_string meta.id
               ^ " already exists in registry")
          | None -> ());
      let max_id = ref (-1) in
      let loaded =
        List.map header.tensors ~f:(fun meta ->
            (* Create ndarray *)
            let nd =
              Nd.create_array ~debug:"loaded" meta.prec ~dims:meta.dims
                ~padding:meta.padding
            in
            (* Seek and read payload *)
            Stdlib.seek_in ic (data_start + meta.offset);
            let padding = Option.map ~f:fst meta.padding in
            Nd.read_payload_from_channel ?padding nd ic meta.byte_length;
            (* Create tnode *)
            let tn =
              Tn.create_from_padded ~id:meta.id ~label:meta.label ~ndarray:nd
                ~padding:meta.padding ()
            in
            if meta.id > !max_id then max_id := meta.id;
            tn)
      in
      (* Bump session ID floor *)
      if !max_id >= 0 then Ocannl_tensor.Tensor.bump_next_id !max_id;
      Set.of_list (module Tn) loaded
    with
    | result ->
        Stdlib.close_in ic;
        result
    | exception exn ->
        Stdlib.close_in_noerr ic;
        raise exn
  in
  result

let restore t_set path =
  if Set.is_empty t_set then ()
  else begin
    let ic = Stdlib.open_in_bin path in
    (match
       let header = read_header ic in
       validate_header header;
       let data_start = Stdlib.pos_in ic in
       (* Build lookup map *)
       let file_tensors =
         Map.of_alist_exn
           (module Int)
           (List.map header.tensors ~f:(fun m -> (m.id, m)))
       in
       Set.iter t_set ~f:(fun tn ->
           match Map.find file_tensors tn.Tn.id with
           | None ->
               failwith
                 ("restore: tensor " ^ Int.to_string tn.Tn.id
                ^ " not found in checkpoint")
           | Some meta ->
               (* Verify precision matches *)
               let tn_prec = Lazy.force tn.Tn.prec in
               if not (Ops.equal_prec tn_prec meta.prec) then
                 failwith
                   ("restore: precision mismatch for tensor "
                  ^ Int.to_string tn.Tn.id);
               (* Verify padded dims match *)
               let tn_dims = Lazy.force tn.Tn.dims in
               if not (Array.equal Int.equal tn_dims meta.dims) then
                 failwith
                   ("restore: dimension mismatch for tensor "
                  ^ Int.to_string tn.Tn.id);
               (* Verify padding matches *)
               let tn_padding = Lazy.force tn.Tn.padding in
               let padding_equal =
                 match (tn_padding, meta.padding) with
                 | None, None -> true
                 | Some (p1, v1), Some (p2, v2) ->
                     Array.equal Ops.equal_axis_padding p1 p2
                     && Option.equal Float.equal v1 v2
                 | _ -> false
               in
               if not padding_equal then
                 failwith
                   ("restore: padding mismatch for tensor "
                  ^ Int.to_string tn.Tn.id);
               (* Get existing ndarray *)
               let nd =
                 match Lazy.force tn.Tn.array with
                 | Some nd -> nd
                 | None ->
                     failwith
                       ("restore: tensor " ^ Int.to_string tn.Tn.id
                      ^ " has no hosted array")
               in
               (* Seek and read payload *)
               Stdlib.seek_in ic (data_start + meta.offset);
               let padding = Option.map ~f:fst meta.padding in
               Nd.read_payload_from_channel ?padding nd ic meta.byte_length;
               (* Mark host as authoritative:
                  - Clear prepare_read to prevent stale device-to-host transfers
                  - Call do_write to clear prepare_write and devices_not_lagging_host *)
               tn.Tn.prepare_read <- None;
               Tn.do_write tn)
     with
    | () -> Stdlib.close_in ic
    | exception exn ->
        Stdlib.close_in_noerr ic;
        raise exn)
  end
