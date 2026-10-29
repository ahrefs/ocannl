open Base
module Tn = Ir.Tnode
module Nd = Ir.Ndarray
module Ops = Ir.Ops

(** {1 Checkpoint file format for tensor persistence} *)

type tensor_meta = {
  id : int;
  namespace : string;
      (** The tnode's namespace (gh-ocannl-372). The empty string is a legacy encoding of
          {!Ir.Tnode.default_namespace}, produced by pre-namespace checkpoints. *)
  label : string list;
  prec : Ops.prec;
  dims : int array;  (** Padded (buffer) dimensions. *)
  padding : (Ops.axis_padding array * float) option;
  offset : int;  (** Byte offset in the data section. *)
  byte_length : int;  (** Bytes in the data section for this tensor. *)
}
(** Metadata for a single tensor in a checkpoint file. *)

type checkpoint_header = {
  version : int;  (** Currently 1. *)
  alignment : int;
      (** The boundary every payload offset is a multiple of, counted from the start of the file
          (gh-ocannl-467): the data section itself starts on such a boundary, and the offsets are
          relative to it. Files written before the field existed are read as [alignment = 1], i.e.
          payloads packed back to back. Compare GGUF's [general.alignment]. *)
  tensors : tensor_meta list;
  metadata : (string * string) list;
      (** Free-form key-value pairs riding with the tensors (gh-ocannl-96): what {!Checkpoint} keeps
          of a training run's host-side progress, so that it is published in the same atomic write
          as the state it describes. Written only when non-empty, so a checkpoint without metadata
          is byte-identical to one written before the field existed; readers predating it ignore the
          field. *)
}

(** {2 S-expression serialization for checkpoint types} *)

(* Manual sexp conversion for tensor_meta since Ops.prec uses manual sexp *)
let sexp_of_tensor_meta m =
  Sexp.List
    [
      Sexp.List [ Sexp.Atom "id"; Sexp.Atom (Int.to_string m.id) ];
      Sexp.List [ Sexp.Atom "namespace"; Sexp.Atom m.namespace ];
      Sexp.List [ Sexp.Atom "label"; Sexp.List (List.map m.label ~f:(fun s -> Sexp.Atom s)) ];
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
                              [ Sexp.Atom (Int.to_string left); Sexp.Atom (Int.to_string right) ])));
                  Sexp.Atom (Float.to_string pad_val);
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
  let id = match find "id" with Sexp.Atom s -> Int.of_string s | _ -> failwith "bad id" in
  let namespace = match find "namespace" with Sexp.Atom s -> s | _ -> failwith "bad namespace" in
  let label =
    match find "label" with
    | Sexp.List atoms ->
        List.map atoms ~f:(function Sexp.Atom s -> s | _ -> failwith "bad label element")
    | _ -> failwith "bad label"
  in
  let prec = Ops.prec_of_sexp (find "prec") in
  let dims =
    match find "dims" with
    | Sexp.List atoms ->
        Array.of_list
          (List.map atoms ~f:(function Sexp.Atom s -> Int.of_string s | _ -> failwith "bad dim"))
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
          | Sexp.Atom "none" ->
              (* Pre-neutral-commit files: the margins' neutral was undetermined (managed by
                 per-operation resets, which no longer exist). *)
              failwith
                "tensor_meta_of_sexp: legacy padding without a committed neutral element; re-save \
                 the checkpoint"
          | Sexp.Atom s -> Float.of_string s
          | _ -> failwith "bad padding value"
        in
        Some (padding_arr, pad_val)
    | _ -> failwith "bad padding"
  in
  let offset =
    match find "offset" with Sexp.Atom s -> Int.of_string s | _ -> failwith "bad offset"
  in
  let byte_length =
    match find "byte_length" with Sexp.Atom s -> Int.of_string s | _ -> failwith "bad byte_length"
  in
  { id; namespace; label; prec; dims; padding; offset; byte_length }

let sexp_of_checkpoint_header h =
  Sexp.List
    ([
       Sexp.List [ Sexp.Atom "version"; Sexp.Atom (Int.to_string h.version) ];
       Sexp.List [ Sexp.Atom "alignment"; Sexp.Atom (Int.to_string h.alignment) ];
       Sexp.List [ Sexp.Atom "tensors"; Sexp.List (List.map h.tensors ~f:sexp_of_tensor_meta) ];
     ]
    @
    if List.is_empty h.metadata then []
    else
      [
        Sexp.List
          [
            Sexp.Atom "metadata";
            Sexp.List
              (List.map h.metadata ~f:(fun (k, v) -> Sexp.List [ Sexp.Atom k; Sexp.Atom v ]));
          ];
      ])

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
    match find "version" with Sexp.Atom s -> Int.of_string s | _ -> failwith "bad version"
  in
  (* Absent in checkpoints written before the field existed: they pack payloads back to back. *)
  let alignment =
    match List.Assoc.find fields "alignment" ~equal:String.equal with
    | None -> 1
    | Some (Sexp.Atom s) -> Int.of_string s
    | Some _ -> failwith "bad alignment"
  in
  let tensors =
    match find "tensors" with
    | Sexp.List metas -> List.map metas ~f:tensor_meta_of_sexp
    | _ -> failwith "bad tensors"
  in
  (* Absent in checkpoints written without metadata, and in every one written before the field
     existed. *)
  let metadata =
    match List.Assoc.find fields "metadata" ~equal:String.equal with
    | None -> []
    | Some (Sexp.List pairs) ->
        List.map pairs ~f:(function
          | Sexp.List [ Sexp.Atom k; Sexp.Atom v ] -> (k, v)
          | _ -> failwith "bad metadata entry")
    | Some _ -> failwith "bad metadata"
  in
  { version; alignment; tensors; metadata }

(** {2 File I/O helpers} *)

(** The payload alignment written by default (gh-ocannl-467): SIMD-friendly data pointers, and the
    same value GGUF defaults [general.alignment] to. Mapping does not need it -- {!Unix.map_file}
    handles an arbitrary offset by mapping from the enclosing page boundary. *)
let default_alignment = 32

let round_up n alignment = if alignment <= 1 then n else (n + alignment - 1) / alignment * alignment

(* The header is padded with spaces up to a multiple of the alignment, so that the data section --
   which starts wherever the header ends -- begins on an aligned boundary and the payload offsets
   relative to it are absolute file alignments too. Padding the header rather than the gap after it
   keeps "the data starts at the first byte past the header" true, so a reader that predates the
   alignment field still finds the payloads where their offsets say. *)
let write_header oc header =
  let sexp = sexp_of_checkpoint_header header in
  let header_str = Sexp.to_string_hum sexp in
  let len = String.length header_str in
  (* [output_binary_int] contributes the 4 bytes preceding the header string. *)
  let padded_len = round_up (4 + len) header.alignment - 4 in
  Stdlib.output_binary_int oc padded_len;
  Stdlib.output_string oc header_str;
  Stdlib.output_string oc (String.make (padded_len - len) ' ')

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
  let header_str = String.strip (Bytes.to_string buf) in
  let sexp = Sexplib.Sexp.of_string header_str in
  checkpoint_header_of_sexp sexp

(* Pre-namespace checkpoints wrote namespace = "". The namespace charset excludes ':', so this key
   encoding is injective. *)
let meta_namespace m = if String.is_empty m.namespace then Tn.default_namespace else m.namespace
let meta_key m = meta_namespace m ^ ":" ^ Int.to_string m.id
let meta_name m = Tn.ident_prefix (meta_namespace m) ^ Int.to_string m.id

let validate_header header =
  if header.version <> 1 then
    failwith ("unsupported checkpoint version: " ^ Int.to_string header.version);
  if header.alignment < 1 then
    failwith ("invalid checkpoint payload alignment: " ^ Int.to_string header.alignment);
  (* Check for duplicate (namespace, id) pairs *)
  let ids = List.map header.tensors ~f:meta_key in
  let unique_ids = Set.of_list (module String) ids in
  if Set.length unique_ids <> List.length ids then
    failwith "checkpoint contains duplicate tensor IDs";
  if List.contains_dup header.metadata ~compare:(fun (k1, _) (k2, _) -> String.compare k1 k2) then
    failwith "checkpoint contains duplicate metadata keys"

(** {2 Payload ingestion: mapped or decoded (gh-ocannl-467)} *)

(* Whether a payload is wrapped as a mapping of the file instead of being decoded into a fresh host
   buffer. On by default everywhere, Windows included (gh-ocannl-588). Windows was the platform this
   was expected to break on: a mapped view keeps the file object referenced, so [save]'s replacing
   rename over a path it has loaded from should fail with a sharing violation. Measured instead of
   assumed, it does not -- the rename succeeds, and the live mapping goes on reading the bytes it
   was taken from, exactly as a POSIX rename leaves the mapped inode alone -- presumably because the
   descriptor the section was created from carries delete sharing, though what was measured is the
   behaviour, not the mechanism. Test 15 of test/operations/test_tensor_persistence.ml is that
   measurement, and it makes the same claims on every platform. The setting stays, for a filesystem
   that does refuse. *)
let mmap_by_default = lazy (Utils.get_global_flag ~default:true ~arg_name:"checkpoint_load_mmap")
let use_mmap = function Some flag -> flag | None -> Lazy.force mmap_by_default

type payload_reader = {
  path : string;  (** For error messages only: the payloads are addressed through [ic]. *)
  ic : Stdlib.in_channel;
  data_start : int;
  mmap : bool;
}

let open_reader ?mmap path =
  let ic = Stdlib.open_in_bin path in
  { path; ic; data_start = 0; mmap = use_mmap mmap }

(* The payload extents are checked against the file size up front: mapping past the end of a file is
   an error worth reporting against the checkpoint rather than a partially-read buffer. *)
let validate_extents reader header =
  let file_size = Stdlib.in_channel_length reader.ic in
  List.iter header.tensors ~f:(fun meta ->
      if
        meta.offset < 0 || meta.byte_length < 0
        || reader.data_start + meta.offset + meta.byte_length > file_size
      then
        failwith
          ("checkpoint " ^ reader.path ^ ": the payload of tensor " ^ meta_name meta
         ^ " extends past the end of the file"))

(** Ingests one tensor's payload through {!Nd.ingest_payload}: a mapping of the file where that is
    byte-equivalent, otherwise a fresh buffer decoded from the channel. The format's half of that
    question is what gets passed in. Padding is the only format-level disqualifier: a padded node's
    payload holds just the logical region, which the decoding path scatters into the padded buffer.
    The recorded [byte_length] going in is byte-compatibility bookkeeping -- the payload has to be
    exactly the buffer, which also rejects a header claiming a byte length its dimensions and
    precision do not add up to. Alignment is not automatic either, [?alignment] being the writer's
    choice: at [~alignment:1] -- the packed layout of pre-alignment-field checkpoints -- a byte
    payload ahead of a float one leaves the latter at an odd offset, and that one decodes.

    The channel is the one the header was read through, which is what {!Nd.ingest_payload} maps
    from: a concurrent atomic save would put a different inode at [reader.path]. *)
let ingest_payload reader ~debug meta =
  let byte_offset = reader.data_start + meta.offset in
  let nd, (_ : Nd.ingestion) =
    Nd.ingest_payload ?padding:meta.padding ~mmap:reader.mmap ~debug meta.prec ~dims:meta.dims
      ~byte_offset ~nbytes:meta.byte_length reader.ic
  in
  nd

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

let save ~ctx ~appending ?(alignment = default_alignment) ?(metadata = []) t_set path =
  if alignment < 1 then
    invalid_arg ("Persistence.save: alignment must be positive, got " ^ Int.to_string alignment);
  if List.contains_dup metadata ~compare:(fun (k1, _) (k2, _) -> String.compare k1 k2) then
    invalid_arg "Persistence.save: duplicate metadata keys";
  let tn_list = Set.to_list t_set in
  (* Retrieve each tnode's data from its device buffer on demand (gh-ocannl-333). *)
  let host_of = Hashtbl.create (module Int) in
  List.iter tn_list ~f:(fun tn ->
      match try Some (Context.to_host ctx tn) with _ -> None with
      | None -> failwith ("save: tensor " ^ Tn.id tn ^ " is not present in the given context")
      | Some nd -> Hashtbl.set host_of ~key:tn.Tn.uid ~data:nd);
  (* Collect current tensor data *)
  let new_entries =
    List.map tn_list ~f:(fun tn ->
        let prec = Lazy.force tn.Tn.storage_prec in
        let dims = Lazy.force tn.Tn.dims in
        let padding = Lazy.force tn.Tn.padding in
        let byte_length = compute_byte_length prec dims padding in
        let meta =
          {
            id = tn.Tn.id;
            namespace = tn.Tn.namespace;
            label = tn.Tn.label;
            prec;
            dims;
            padding;
            offset = 0;
            (* Will be computed later *)
            byte_length;
          }
        in
        (meta, tn))
  in
  (* If appending, read existing file and merge *)
  let existing_metadata = ref [] in
  let all_entries =
    if appending && Stdlib.Sys.file_exists path then begin
      let ic = Stdlib.open_in_bin path in
      let existing_header = read_header ic in
      validate_header existing_header;
      existing_metadata := existing_header.metadata;
      (* Read the data offset for seeking *)
      let data_start = Stdlib.pos_in ic in
      (* Read existing binary payloads for non-overlapping tensors *)
      let new_ids =
        Set.of_list (module String) (List.map new_entries ~f:(fun (m, _) -> meta_key m))
      in
      let kept_entries =
        List.filter_map existing_header.tensors ~f:(fun meta ->
            if Set.mem new_ids (meta_key meta) then None
            else begin
              (* Read the existing payload *)
              Stdlib.seek_in ic (data_start + meta.offset);
              let payload = Bytes.create meta.byte_length in
              (try Stdlib.really_input ic payload 0 meta.byte_length
               with End_of_file ->
                 failwith ("save: failed to read existing payload for tensor " ^ meta_name meta));
              Some (`Existing (meta, payload))
            end)
      in
      Stdlib.close_in ic;
      let new_tagged = List.map new_entries ~f:(fun (m, tn) -> `New (m, tn)) in
      kept_entries @ new_tagged
    end
    else List.map new_entries ~f:(fun (m, tn) -> `New (m, tn))
  in
  (* Compute sequential offsets, each rounded up to the declared alignment. *)
  let _, entries_with_offsets =
    List.fold all_entries ~init:(0, []) ~f:(fun (cursor, acc) entry ->
        let offset = round_up cursor alignment in
        let byte_length =
          match entry with `Existing (m, _) -> m.byte_length | `New (m, _) -> m.byte_length
        in
        let entry_with_offset =
          match entry with
          | `Existing (m, payload) -> `Existing ({ m with offset }, payload)
          | `New (m, tn) -> `New ({ m with offset }, tn)
        in
        (offset + byte_length, entry_with_offset :: acc))
  in
  let entries_with_offsets = List.rev entries_with_offsets in
  (* A save killed between streaming and commit cannot run its own cleanup, and what it abandons is
     the size of the model. The next save of that checkpoint is the event that reclaims it — but
     only over ITS OWN staging files: a checkpoint sits among the user's other files, where another
     process's publication is none of this call's business. *)
  Utils.Atomic_file.cleanup_stale_for path;
  (* Stream into a staging sibling, then rename over [path]. [Atomic_file] owns the whole dance: the
     staging name is unique per writer, so two processes checkpointing the same path no longer
     stream into one file, and the staging artifact is removed on every failing path — including a
     rename a filesystem refuses because this process still has the target mapped. *)
  Utils.Atomic_file.with_channel ~path () ~f:(fun oc ->
      let header =
        {
          version = 1;
          alignment;
          tensors =
            List.map entries_with_offsets ~f:(function `Existing (m, _) -> m | `New (m, _) -> m);
          (* Like the tensors: an appending save replaces the keys it names and keeps the rest. *)
          metadata =
            List.filter !existing_metadata ~f:(fun (k, _) ->
                not (List.Assoc.mem metadata k ~equal:String.equal))
            @ metadata;
        }
      in
      write_header oc header;
      let cursor = ref 0 in
      List.iter entries_with_offsets ~f:(fun entry ->
          let meta = match entry with `Existing (m, _) -> m | `New (m, _) -> m in
          (* Fill the alignment gap ahead of this payload. *)
          Stdlib.output_string oc (String.make (meta.offset - !cursor) '\000');
          (match entry with
          | `Existing (_, payload) -> Stdlib.output_bytes oc payload
          | `New (_, tn) ->
              let nd = Hashtbl.find_exn host_of tn.Tn.uid in
              let padding = Option.map ~f:fst (Lazy.force tn.Tn.padding) in
              let _n = Nd.write_payload_to_channel ?padding nd oc in
              ());
          cursor := meta.offset + meta.byte_length))

let metadata path =
  let ic = Stdlib.open_in_bin path in
  Exn.protect
    ~finally:(fun () -> Stdlib.close_in_noerr ic)
    ~f:(fun () ->
      let header = read_header ic in
      validate_header header;
      header.metadata)

let load ~ctx ?prefix_namespace ?mmap path =
  let prefix_namespace =
    match prefix_namespace with
    | None | Some "" -> None
    | Some p ->
        Tn.validate_namespace p;
        Some p
  in
  (* [prefix_namespace] preserves the internal namespace structure of a multi-namespace file:
     corresponding s_ids stay equal, keeping [Embed_self_id] semantics invariant. *)
  let target_namespace meta =
    match prefix_namespace with
    | None -> meta_namespace meta
    | Some p -> p ^ "__" ^ meta_namespace meta
  in
  let reader = open_reader ?mmap path in
  let ic = reader.ic in
  let result =
    match
      let header = read_header ic in
      validate_header header;
      let reader = { reader with data_start = Stdlib.pos_in ic } in
      validate_extents reader header;
      (* Pre-check: verify no ID clashes before creating anything *)
      List.iter header.tensors ~f:(fun meta ->
          match Tn.find_namespaced ~namespace:(target_namespace meta) ~id:meta.id with
          | Some _ ->
              failwith
                ("load: tensor with id "
                ^ Tn.ident_prefix (target_namespace meta)
                ^ Int.to_string meta.id ^ " already exists in registry")
          | None -> ());
      let max_id = ref (-1) in
      let loaded =
        List.map header.tensors ~f:(fun meta ->
            let nd = ingest_payload reader ~debug:"loaded" meta in
            (* Create the tnode (no host data is stored on it); register the loaded buffer so it is
               uploaded into the context below (gh-ocannl-333). *)
            let tn, init =
              Tn.create_from_padded ~namespace:(target_namespace meta) ~id:meta.id ~label:meta.label
                ~ndarray:nd ~padding:meta.padding ()
            in
            Ir.Host_inits.register tn init;
            (* Only nodes landing in the ambient namespace can collide with future session ids. *)
            if
              String.equal (target_namespace meta) (Tn.get_current_namespace ())
              && meta.id > !max_id
            then max_id := meta.id;
            (tn, nd))
      in
      (* Bump session ID floor *)
      if !max_id >= 0 then Ocannl_tensor.Tensor.bump_next_id !max_id;
      (* Upload each loaded node into the context, returning the updated context. *)
      let ctx = List.fold loaded ~init:ctx ~f:(fun ctx (tn, nd) -> Context.from_host ctx tn nd) in
      (ctx, Set.of_list (module Tn) (List.map loaded ~f:fst))
    with
    | result ->
        Stdlib.close_in ic;
        result
    | exception exn ->
        Stdlib.close_in_noerr ic;
        raise exn
  in
  result

let restore ~ctx ?mmap ?check_metadata t_set path =
  if Set.is_empty t_set && Option.is_none check_metadata then ctx
  else begin
    let reader = open_reader ?mmap path in
    let ic = reader.ic in
    match
      let header = read_header ic in
      validate_header header;
      (* Read through the same channel as the payloads: a concurrent atomic save replaces the
         directory entry, not this open file, so the metadata and the tensors are one snapshot. *)
      Option.iter check_metadata ~f:(fun check -> check header.metadata);
      let reader = { reader with data_start = Stdlib.pos_in ic } in
      validate_extents reader header;
      (* Build lookup map *)
      let file_tensors =
        Map.of_alist_exn (module String) (List.map header.tensors ~f:(fun m -> (meta_key m, m)))
      in
      let tn_key tn = tn.Tn.namespace ^ ":" ^ Int.to_string tn.Tn.id in
      Set.fold t_set ~init:ctx ~f:(fun ctx tn ->
          match Map.find file_tensors (tn_key tn) with
          | None -> failwith ("restore: tensor " ^ Tn.id tn ^ " not found in checkpoint")
          | Some meta ->
              (* Verify precision matches *)
              let tn_prec = Lazy.force tn.Tn.storage_prec in
              if not (Ops.equal_prec tn_prec meta.prec) then
                failwith ("restore: precision mismatch for tensor " ^ Tn.id tn);
              (* Verify padded dims match *)
              let tn_dims = Lazy.force tn.Tn.dims in
              if not (Array.equal Int.equal tn_dims meta.dims) then
                failwith ("restore: dimension mismatch for tensor " ^ Tn.id tn);
              (* Verify padding matches *)
              let tn_padding = Lazy.force tn.Tn.padding in
              let padding_equal =
                match (tn_padding, meta.padding) with
                | None, None -> true
                | Some (p1, v1), Some (p2, v2) ->
                    Array.equal Ops.equal_axis_padding p1 p2 && Float.equal v1 v2
                | _ -> false
              in
              if not padding_equal then failwith ("restore: padding mismatch for tensor " ^ Tn.id tn);
              (* Ingest the payload as a host buffer and upload it into the context's device buffer
                 (gh-ocannl-333). *)
              let nd = ingest_payload reader ~debug:"restored" meta in
              Context.from_host ctx tn nd)
    with
    | ctx ->
        Stdlib.close_in ic;
        ctx
    | exception exn ->
        Stdlib.close_in_noerr ic;
        raise exn
  end
