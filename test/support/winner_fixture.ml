open Base
open Stdio
module Idx = Ir.Indexing
module LL = Ir.Low_level
module Sched = Ir.Schedule
module SC = Ir.Schedule_cache
module SO = Ir.Schedule_outcome
module Tn = Ir.Tnode

type segment = { kind : Sched.segment_kind; units : int; ops : SC.saved_schedule }
[@@deriving sexp, equal]

type t = { prelude : SC.saved_schedule; segments : segment list option [@sexp.option] }
[@@deriving sexp, equal]

type anchor_location = Prelude | Segment of int [@@deriving sexp, equal]

type anchor_error = {
  location : anchor_location;
  op_index : int option;
  op_name : string option;
  anchor : string;
  detail : string;
}
[@@deriving sexp, equal]

let location_to_string = function
  | Prelude -> "prelude"
  | Segment i -> Printf.sprintf "segment %d" i

let error_to_string err =
  let where =
    match (err.op_index, err.op_name) with
    | Some idx, Some op -> Printf.sprintf "%s op #%d (%s)" (location_to_string err.location) idx op
    | Some idx, None -> Printf.sprintf "%s op #%d" (location_to_string err.location) idx
    | None, Some op -> Printf.sprintf "%s (%s)" (location_to_string err.location) op
    | None, None -> location_to_string err.location
  in
  Printf.sprintf "%s: stale anchor %s: %s" where err.anchor err.detail

let check_structure (fixture : t) : (t, string) Result.t =
  match fixture.segments with
  | None -> Ok fixture
  | Some [] -> Error "fixture has an empty segment list"
  | Some segs -> (
      match List.findi segs ~f:(fun _ s -> s.units <= 0) with
      | Some (i, s) -> Error (Printf.sprintf "segment %d has non-positive unit count %d" i s.units)
      | None -> Ok fixture)

let extract_entry (entry : SC.entry) : (t, string) Result.t =
  let segments =
    Option.map entry.SC.segments ~f:(fun segs ->
        List.map segs ~f:(fun (s : SC.saved_segment) ->
            { kind = s.SC.seg_kind; units = s.SC.seg_units; ops = s.SC.seg_saved }))
  in
  check_structure { prelude = entry.SC.saved; segments }

let placements_prefix = "placements-"
let abandonment_prefix = "abandonment-"

let extract_file path : (t, string) Result.t =
  if not (Stdlib.Sys.file_exists path) then Error (Printf.sprintf "file does not exist: %s" path)
  else
    match SC.read_entry_file path with
    | None ->
        Error "file is not a readable Schedule_cache stored entry ((key <k>) (payload <entry>))"
    | Some (key, payload) -> (
        if String.is_prefix key ~prefix:placements_prefix then
          Error
            (Printf.sprintf
               "cache entry %S is in the %s namespace (placement decision), not a schedule winner"
               key placements_prefix)
        else if String.is_prefix key ~prefix:abandonment_prefix then
          Error
            (Printf.sprintf
               "cache entry %S is in the %s namespace (abandoned search), not a schedule winner" key
               abandonment_prefix)
        else
          match SC.entry_of_sexp payload with
          | entry -> extract_entry entry
          | exception exn ->
              Error
                (Printf.sprintf "payload does not decode as Schedule_cache.entry: %s"
                   (Exn.to_string exn)))

let of_sexp_string s : (t, string) Result.t =
  match Sexplib.Sexp.of_string s |> t_of_sexp with
  | fixture -> check_structure fixture
  | exception exn -> Error (Printf.sprintf "failed to parse winner fixture: %s" (Exn.to_string exn))

let load_file path : (t, string) Result.t =
  match In_channel.read_all path with
  | contents -> of_sexp_string contents
  | exception Sys_error msg -> Error msg

let load_file_exn path =
  match load_file path with Ok fixture -> fixture | Error msg -> invalid_arg msg

let to_sexp_string fixture = Sexp.to_string_hum (sexp_of_t fixture) ^ "\n"

let save_file ~path fixture =
  Out_channel.with_file ~binary:true path ~f:(fun oc ->
      Out_channel.output_string oc (to_sexp_string fixture))

let hermetic_copy (opt : LL.optimized) : LL.optimized =
  {
    opt with
    LL.traced_store = Hashtbl.copy opt.LL.traced_store;
    optimize_ctx = LL.copy_optimize_ctx opt.LL.optimize_ctx;
  }

let saved_optop_name : SC.saved_optop -> string = function
  | SC.Split _ -> "Split"
  | Swap _ -> "Swap"
  | Retype _ -> "Retype"
  | Unroll _ -> "Unroll"
  | Partition _ -> "Partition"
  | Pad _ -> "Pad"
  | Coalesce _ -> "Coalesce"
  | Stage _ -> "Stage"
  | Privatize _ -> "Privatize"
  | Expand_zero _ -> "Expand_zero"
  | Tensorize _ -> "Tensorize"
  | Fuse_epilogue _ -> "Fuse_epilogue"
  | Split_reduce _ -> "Split_reduce"
  | Fold_mma _ -> "Fold_mma"

let sref_to_string sref = Sexp.to_string_hum (SC.sexp_of_sym_ref sref)

let saved_optop_anchors : SC.saved_optop -> string = function
  | SC.Split { axis; _ } -> Printf.sprintf "axis %s" (sref_to_string axis)
  | Swap { outer; inner } ->
      Printf.sprintf "outer %s, inner %s" (sref_to_string outer) (sref_to_string inner)
  | Retype { axis; _ } -> Printf.sprintf "axis %s" (sref_to_string axis)
  | Unroll { axis; _ } -> Printf.sprintf "axis %s" (sref_to_string axis)
  | Partition { axis; _ } -> Printf.sprintf "axis %s" (sref_to_string axis)
  | Pad { axis; _ } -> Printf.sprintf "axis %s" (sref_to_string axis)
  | Coalesce { outer; inner } ->
      Printf.sprintf "outer %s, inner %s" (sref_to_string outer) (sref_to_string inner)
  | Stage { source; tile_loops; _ } ->
      Printf.sprintf "source (tn %d), tile_loops (%s)" source
        (String.concat ~sep:" " (List.map tile_loops ~f:sref_to_string))
  | Privatize { target; over; _ } ->
      Printf.sprintf "target (tn %d), over %s" target (sref_to_string over)
  | Expand_zero { tn } -> Printf.sprintf "tn %d" tn
  | Tensorize { i; j; k; _ } ->
      Printf.sprintf "i %s, j %s, k %s" (sref_to_string i) (sref_to_string j) (sref_to_string k)
  | Fuse_epilogue { target; _ } -> Printf.sprintf "target (tn %d)" target
  | Split_reduce { axis; target; _ } ->
      Printf.sprintf "axis %s, target (tn %d)" (sref_to_string axis) target
  | Fold_mma { query; _ } -> Printf.sprintf "query %s" (sref_to_string query)

let segmentation_anchor (plan : Sched.segmentation) =
  Printf.sprintf "segmentation %s" (Sexp.to_string_hum (Sched.sexp_of_segmentation plan))

let is_zeros_segment_code (llc : LL.t) =
  let has_zero = ref false in
  let all_ok =
    List.for_all (LL.flat_lines [ llc ]) ~f:(function
      | LL.Zero_out _ ->
          has_zero := true;
          true
      | LL.Comment _ | LL.Noop -> true
      | _ -> false)
  in
  !has_zero && all_ok

exception Anchor_refusal of anchor_error

let refuse_anchor err = raise (Anchor_refusal err)

module Minted_key = struct
  module T = struct
    type t = int * SC.mint_role [@@deriving sexp, compare]
  end

  include T
  include Comparator.Make (T)
end

type canonical_census = {
  canon : SC.canonical;
  n_base_loops : int;
  n_statics : int;
  n_tns : int;
  base_ids : (int, Idx.symbol) Hashtbl.t;
  static_ids : (int, Idx.symbol) Hashtbl.t;
  shadowed_statics : int Hash_set.t;
}

let canonical_census ~static_indices (opt : LL.optimized) : canonical_census =
  let canon = SC.canonicalize ~static_indices ~with_placements:false opt in
  let base_reg = SC.base_registry canon in
  let n_tns =
    let rec count i =
      match SC.tn_of_ref canon i with _ -> count (i + 1) | exception SO.Cause_at _ -> i
    in
    count 0
  in
  let binders =
    let hooks =
      {
        (LL.Access_fold.hooks ()) with
        statement =
          (fun _ acc stmt ->
            match stmt with
            | LL.For_loop { index; _ } | LL.Scan_loop { index; _ } ->
                LL.Access_fold.Continue (index :: acc)
            | _ -> LL.Access_fold.Continue acc);
      }
    in
    LL.Access_fold.fold
      ~policy:
        {
          discarded_operands = Visit;
          gated_operands = Visit;
          dead_loops = Visit;
          local_scopes = Visit;
          guards = Ignore;
          scan_implicit = Skip;
        }
      ~hooks ~init:[] opt.LL.llc
  in
  let base_ids = Hashtbl.create (module Int) in
  List.iter binders ~f:(fun sym ->
      match SC.resolve base_reg sym with
      | Some (SC.Base id) -> Hashtbl.set base_ids ~key:id ~data:sym
      | Some (SC.Static _ | SC.Minted _) | None -> ());
  let static_ids = Hashtbl.create (module Int) in
  let shadowed_statics = Hash_set.create (module Int) in
  List.iteri static_indices ~f:(fun k ss ->
      let sym = ss.Idx.static_symbol in
      match SC.resolve base_reg sym with
      | Some (SC.Static k') when k' = k -> Hashtbl.set static_ids ~key:k ~data:sym
      | Some _ | None -> Hash_set.add shadowed_statics k);
  {
    canon;
    n_base_loops = List.length binders;
    n_statics = List.length static_indices;
    n_tns;
    base_ids;
    static_ids;
    shadowed_statics;
  }

let validate_ops ~location (census : canonical_census) (ops : SC.saved_schedule) : unit =
  let op_names = Array.of_list_map ops ~f:saved_optop_name in
  let minted_roles_by_op : SC.mint_role list array = Array.create ~len:(List.length ops) [] in
  let minted_keys = ref (Set.empty (module Minted_key)) in
  let record_minted idx role =
    minted_keys := Set.add !minted_keys (idx, role);
    minted_roles_by_op.(idx) <- minted_roles_by_op.(idx) @ [ role ]
  in
  let fail ~op_index ~op_name ~anchor ~detail =
    refuse_anchor { location; op_index = Some op_index; op_name = Some op_name; anchor; detail }
  in
  let check_sym ~op_index ~op_name ~field (sref : SC.sym_ref) : unit =
    let anchor = Printf.sprintf "%s %s" field (sref_to_string sref) in
    match sref with
    | SC.Base i ->
        if i < 0 || i >= census.n_base_loops then
          fail ~op_index ~op_name ~anchor
            ~detail:
              (Printf.sprintf "base loop %d does not exist (fresh lowering has %d base loop(s))" i
                 census.n_base_loops)
        else if not (Hashtbl.mem census.base_ids i) then
          fail ~op_index ~op_name ~anchor
            ~detail:
              (Printf.sprintf
                 "base loop %d is ambiguous (bound by more than one loop in fresh lowering)" i)
    | SC.Static k ->
        if Hash_set.mem census.shadowed_statics k then
          fail ~op_index ~op_name ~anchor
            ~detail:
              (Printf.sprintf "static index %d is shadowed by a loop binder in fresh lowering" k)
        else if not (Hashtbl.mem census.static_ids k) then
          fail ~op_index ~op_name ~anchor
            ~detail:
              (Printf.sprintf
                 "static index %d does not exist (fresh lowering has %d static index(es))" k
                 census.n_statics)
    | SC.Minted (src_op, role) ->
        if not (Set.mem !minted_keys (src_op, role)) then
          let detail =
            if src_op < 0 || src_op >= op_index then
              Printf.sprintf "minted axis references op #%d, which has not run before op #%d" src_op
                op_index
            else
              let minted_desc =
                match minted_roles_by_op.(src_op) with
                | [] -> "mints no symbols"
                | roles ->
                    "only minted "
                    ^ String.concat ~sep:", "
                        (List.map roles ~f:(fun r -> Sexp.to_string (SC.sexp_of_mint_role r)))
              in
              Printf.sprintf "op #%d (%s) did not mint %s (%s)" src_op op_names.(src_op)
                (Sexp.to_string (SC.sexp_of_mint_role role))
                minted_desc
          in
          fail ~op_index ~op_name ~anchor ~detail
  in
  let check_tn ~op_index ~op_name ~field idx : Tn.t =
    if idx < 0 || idx >= census.n_tns then
      fail ~op_index ~op_name
        ~anchor:(Printf.sprintf "%s (tn %d)" field idx)
        ~detail:
          (Printf.sprintf "tensor node %d does not exist (fresh lowering has %d tensor node(s))" idx
             census.n_tns)
    else SC.tn_of_ref census.canon idx
  in
  List.iteri ops ~f:(fun op_index saved_op ->
      let op_name = saved_optop_name saved_op in
      let sym field sref = check_sym ~op_index ~op_name ~field sref in
      let tn field idx = check_tn ~op_index ~op_name ~field idx in
      match saved_op with
      | SC.Split { axis; _ } ->
          sym "axis" axis;
          record_minted op_index SC.Split_outer;
          record_minted op_index SC.Split_inner
      | Swap { outer; inner } ->
          sym "outer" outer;
          sym "inner" inner
      | Retype { axis; _ } | Unroll { axis; _ } | Pad { axis; _ } -> sym "axis" axis
      | Partition { axis; breakpoints } ->
          sym "axis" axis;
          for j = 0 to List.length breakpoints do
            record_minted op_index (SC.Partition_seg j)
          done
      | Coalesce { outer; inner } ->
          sym "outer" outer;
          sym "inner" inner;
          record_minted op_index SC.Coalesce_merged
      | Stage { source; tile_loops; _ } ->
          ignore (tn "source" source : Tn.t);
          List.iteri tile_loops ~f:(fun k sref -> sym (Printf.sprintf "tile_loops[%d]" k) sref)
      | Privatize { target; over; _ } ->
          ignore (tn "target" target : Tn.t);
          sym "over" over
      | Expand_zero { tn = tn_idx } ->
          let node = tn "tn" tn_idx in
          let rank = Array.length (Lazy.force node.Tn.dims) in
          for j = 0 to rank - 1 do
            record_minted op_index (SC.Expand_axis j)
          done
      | Tensorize { i; j; k; _ } ->
          sym "i" i;
          sym "j" j;
          sym "k" k;
          record_minted op_index SC.Tensorize_lane
      | Fuse_epilogue { target; _ } -> ignore (tn "target" target : Tn.t)
      | Split_reduce { axis; target; _ } ->
          sym "axis" axis;
          let node = tn "target" target in
          let rank = Array.length (Lazy.force node.Tn.dims) in
          record_minted op_index SC.Split_reduce_block;
          record_minted op_index SC.Split_reduce_inner;
          for j = 0 to rank - 1 do
            record_minted op_index (SC.Split_reduce_combine j)
          done
      | Fold_mma { query; _ } ->
          sym "query" query;
          record_minted op_index SC.Fold_mma_lane;
          record_minted op_index SC.Fold_mma_block)

let validate_and_resolve_ops ~location ~static_indices (opt : LL.optimized)
    (ops : SC.saved_schedule) : Sched.schedule =
  let census = canonical_census ~static_indices opt in
  validate_ops ~location census ops;
  (* Delegate live op construction to [Schedule_cache.of_saved] so fixtures and production cache
     replay share one decoder (defaults, minted symbols, and builder behavior stay aligned). *)
  try fst (SC.of_saved census.canon ops)
  with SO.Cause_at (_, cause) ->
    refuse_anchor
      {
        location;
        op_index = None;
        op_name = None;
        anchor = "Schedule_cache.of_saved";
        detail = SO.detail_of_cause cause;
      }

let apply_tracked ~queue ~last_exec f =
  Sched.with_transform_fault
    (fun _op -> match Queue.dequeue queue with Some entry -> last_exec := Some entry | None -> ())
    f

let error_of_cause ~last_exec ?(default_anchor = "schedule") (cause : SO.cause) : anchor_error =
  let detail = SO.detail_of_cause cause in
  match !last_exec with
  | Some (location, op_index, saved_op) ->
      {
        location;
        op_index = Some op_index;
        op_name = Some (saved_optop_name saved_op);
        anchor = saved_optop_anchors saved_op;
        detail;
      }
  | None -> { location = Prelude; op_index = None; op_name = None; anchor = default_anchor; detail }

let apply_fixture ~is_gpu ~limits ~static_indices (fixture : t) (opt : LL.optimized) :
    (LL.optimized list, anchor_error) Result.t =
  match check_structure fixture with
  | Error detail ->
      Error { location = Prelude; op_index = None; op_name = None; anchor = "segmentation"; detail }
  | Ok fixture -> (
      try
        let opt =
          if List.is_empty fixture.prelude then opt
          else
            let prelude_sched =
              validate_and_resolve_ops ~location:Prelude ~static_indices opt fixture.prelude
            in
            let queue = Queue.create () in
            List.iteri fixture.prelude ~f:(fun idx op -> Queue.enqueue queue (Prelude, idx, op));
            let last_exec = ref None in
            try
              apply_tracked ~queue ~last_exec (fun () ->
                  match fixture.segments with
                  | Some _ ->
                      LL.without_node_bounds (fun () ->
                          Sched.apply_classified ~static_indices prelude_sched opt)
                  | None -> Sched.apply_classified ~static_indices prelude_sched opt)
            with SO.Cause_at (_, cause) ->
              refuse_anchor (error_of_cause ~last_exec ~default_anchor:"prelude" cause)
        in
        match fixture.segments with
        | None -> Ok [ opt ]
        | Some segs -> (
            let plan = List.map segs ~f:(fun s -> (s.kind, s.units)) in
            let segs_arr = Array.of_list segs in
            let queue = Queue.create () in
            let last_exec = ref None in
            let schedule i seg_pre =
              let seg = segs_arr.(i) in
              (match seg.kind with
              | `Zeros when not (is_zeros_segment_code seg_pre.LL.llc) ->
                  refuse_anchor
                    {
                      location = Segment i;
                      op_index = None;
                      op_name = None;
                      anchor = Printf.sprintf "segment %d (Zeros)" i;
                      detail =
                        "segment kind is Zeros, but the fresh lowering's segment does not consist \
                         of Zero_out statement(s)";
                    }
              | `Zeros | `Normal | `Solo | `Unfolded -> ());
              let resolved =
                validate_and_resolve_ops ~location:(Segment i) ~static_indices seg_pre seg.ops
              in
              List.iteri seg.ops ~f:(fun idx op -> Queue.enqueue queue (Segment i, idx, op));
              Some resolved
            in
            try
              let _, tuples =
                apply_tracked ~queue ~last_exec (fun () ->
                    Autotune.tuner_fission ~is_gpu ~limits ~replay:(plan, schedule)
                      ~preset:(fun _ -> [])
                      ~static_indices opt)
              in
              Ok (List.map tuples ~f:(fun (_, _, _, post) -> post))
            with
            | Anchor_refusal _ as exn -> raise exn
            | SO.Cause_at (_, SO.Illegal_schedule { check = "Schedule.fission_segmented"; detail })
              ->
                refuse_anchor
                  {
                    location = Prelude;
                    op_index = None;
                    op_name = None;
                    anchor = segmentation_anchor plan;
                    detail;
                  }
            | SO.Cause_at (_, cause) ->
                refuse_anchor
                  (error_of_cause ~last_exec ~default_anchor:(segmentation_anchor plan) cause))
      with Anchor_refusal err -> Error err)

let validate ~is_gpu ~limits ?(static_indices = []) fixture opt =
  Result.map (apply_fixture ~is_gpu ~limits ~static_indices fixture (hermetic_copy opt)) ~f:ignore

let replay ~is_gpu ~limits ?(static_indices = []) fixture opt =
  match validate ~is_gpu ~limits ~static_indices fixture opt with
  | Error _ as err -> err
  | Ok () -> apply_fixture ~is_gpu ~limits ~static_indices fixture opt

let replay_exn ~is_gpu ~limits ?(static_indices = []) fixture opt =
  match replay ~is_gpu ~limits ~static_indices fixture opt with
  | Ok segments -> segments
  | Error err -> invalid_arg (error_to_string err)
