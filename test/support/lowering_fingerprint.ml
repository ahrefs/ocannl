(* gh-ocannl-1526: a structural fingerprint of a lowered routine, for comparing a search's reused
   lowering with a fresh lowering of the same routine.

   Two lowerings of one routine share the routine's tensor nodes but mint their own loop binders,
   local-scope ids and index symbols, so the fingerprint names nodes by [uid] and alpha-renames the
   rest ([Ir.Low_level.alpha_digest]). It covers every field of [Ir.Low_level.optimized] a compile
   consumes, each in its own named section so a mismatch says where: the code and its source, every
   traced-store record field, every placement (mode and provenance, nodes absent from the code
   included), the stored computations (keys, and each template's index tuple and code), the
   alias-candidate and preference sets, and the companion sets, maps and lists.

   Nodes a compile mints afresh -- schedule tiles, privatized partials, footprint scratch -- differ
   by [uid] between two compiles: given [~known], the nodes it does not hold are named by first
   occurrence instead ("m0", "m1", ... over the code first, then by creation order), so two compiles
   of one schedule agree. Without [~known] every node is named by [uid]. A node's own mutable fields
   (bounds, code names) are not part of the fingerprint. *)
open Base
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Idx = Ir.Indexing

(* The node names of one fingerprint: [uid] for a known node, first occurrence for the rest. *)
let namer ?known () =
  let minted = Hashtbl.create (module Int) in
  fun (tn : Tn.t) ->
    match known with
    | Some known when not (known tn) -> (
        match Hashtbl.find minted tn.Tn.uid with
        | Some tok -> tok
        | None ->
            let tok = "m" ^ Int.to_string (Hashtbl.length minted) in
            Hashtbl.set minted ~key:tn.Tn.uid ~data:tok;
            tok)
    | _ -> "t" ^ Int.to_string tn.Tn.uid

(* Destructured exhaustively, so a field added to the record fails to compile here rather than
   silently escaping the comparison. *)
let traced ~uids
    ({
       tn = _;
       has_assignment;
       zero_initialized_by_code;
       zeroed_out;
       read_before_write;
       read_only;
       is_scalar_constexpr;
       is_accessing;
       is_complex;
       prefers_virtual_one_hot;
       has_non_one_hot_setter;
       is_range_producer;
       inline_reduction_extent;
       read_by_other;
       setter_reads;
       inline_fanin;
     } :
      LL.traced_array) =
  Printf.sprintf "%b%b%b%b%b%b%b%b%b%b%b%b/%d/%d/[%s]" has_assignment zero_initialized_by_code
    zeroed_out read_before_write read_only is_scalar_constexpr is_accessing is_complex
    prefers_virtual_one_hot has_non_one_hot_setter is_range_producer read_by_other
    inline_reduction_extent inline_fanin
    (String.concat ~sep:";" (List.map setter_reads ~f:(fun s -> uids (Set.to_list s))))

(* A stored computation: its index tuple's symbols are free in its code, so they are registered
   first, in tuple order, and the tuple renders positionally. *)
let template ~name ((idcs : Idx.axis_index array option), (code : LL.t)) =
  let syms =
    Option.value_map idcs ~default:[] ~f:(fun idcs ->
        Array.to_list idcs
        |> List.concat_map ~f:(function
          | Idx.Iterator s -> [ s ]
          | Idx.Affine { symbols; _ } -> List.map symbols ~f:snd
          | _ -> []))
  in
  let pos s =
    Option.value_map ~default:"?"
      (List.findi syms ~f:(fun _ s' -> Idx.equal_symbol s s'))
      ~f:(fun (i, _) -> "f" ^ Int.to_string i)
  in
  let axis = function
    | Idx.Fixed_idx k -> Int.to_string k
    | Idx.Iterator s -> pos s
    | Idx.Affine { symbols; offset } ->
        String.concat ~sep:"+" (List.map symbols ~f:(fun (c, s) -> Int.to_string c ^ "*" ^ pos s))
        ^ "+" ^ Int.to_string offset
    | other -> Sexp.to_string (Idx.sexp_of_axis_index other)
  in
  Option.value_map idcs ~default:"-" ~f:(fun idcs ->
      String.concat ~sep:"," (Array.to_list (Array.map idcs ~f:axis)))
  ^ ":"
  ^ LL.alpha_digest ~syms ~tn_token:name code

(* Destructured exhaustively, like [traced]. *)
let sections ?known
    ({
       traced_store;
       optimize_ctx =
         { computations; placements; alias_candidates; inline_preferences; footprint_preferences };
       llc;
       merge_node;
       workgroup_shared;
       simdgroup_fragments;
       swizzled;
       pipelined;
       zero_fringe;
       flip_candidates;
       spliced_rbw;
       source;
     } :
      LL.optimized) : (string * string) list =
  let name = namer ?known () in
  (* The code first, so minted nodes are named by their first occurrence in it. *)
  let llc_digest = LL.alpha_digest ~tn_token:name llc in
  let source_digest = LL.alpha_digest ~tn_token:name source in
  (* Every other node in creation order, then each collection sorted by name. *)
  List.iter
    (List.sort ~compare:Tn.compare
       (Hashtbl.keys traced_store
       @ Hashtbl.keys placements.Tn.Placements.table
       @ Hashtbl.keys computations))
    ~f:(fun tn -> ignore (name tn : string));
  let uids tns =
    String.concat ~sep:"," (List.sort ~compare:String.compare (List.map tns ~f:name))
  in
  let entries alist f =
    String.concat ~sep:" "
      (List.sort ~compare:String.compare (List.map alist ~f:(fun (tn, v) -> name tn ^ "=" ^ f v)))
  in
  [
    ("llc", llc_digest);
    ("source", source_digest);
    ("traced_store", entries (Hashtbl.to_alist traced_store) (traced ~uids));
    ( "placements",
      entries (Hashtbl.to_alist placements.Tn.Placements.table) (fun d ->
          Sexp.to_string ([%sexp_of: Tn.memory_mode * Tn.provenance] d)) );
    ( "computations",
      entries (Hashtbl.to_alist computations) (fun ts ->
          "[" ^ String.concat ~sep:";" (List.map ts ~f:(template ~name)) ^ "]") );
    ("alias_candidates", uids (Hash_set.to_list alias_candidates));
    ("inline_preferences", uids (Hash_set.to_list inline_preferences));
    ("footprint_preferences", uids (Hash_set.to_list footprint_preferences));
    ("merge_node", Option.value_map merge_node ~default:"-" ~f:name);
    ("workgroup_shared", uids (Set.to_list workgroup_shared));
    ("simdgroup_fragments", uids (Set.to_list simdgroup_fragments));
    ("swizzled", uids (Map.keys swizzled));
    ("pipelined", uids (Map.keys pipelined));
    ("zero_fringe", uids (Set.to_list zero_fringe));
    ( "flip_candidates",
      String.concat ~sep:" "
        (List.map flip_candidates ~f:(fun (fc : LL.flip_candidate) ->
             name fc.fc_tn ^ "="
             ^ Sexp.to_string
                 ([%sexp_of: LL.reading * LL.flip_alternative list]
                    (fc.fc_default, fc.fc_alternatives)))) );
    ("spliced_rbw", uids (Set.to_list spliced_rbw));
  ]

(* The names of the sections two fingerprints disagree on; empty when they are equal. *)
let differing a b =
  List.filter_map (List.zip_exn a b) ~f:(fun ((name, x), (_, y)) ->
      if String.equal x y then None else Some name)
