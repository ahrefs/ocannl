open Base
(** The code for operating on n-dimensional arrays. *)

module Lazy = Utils.Lazy
module Tn = Tnode
module Nd = Ndarray

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_ASSIGNMENTS=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_ASSIGNMENTS"]

type init_data =
  | Reshape of Ndarray.t
  | Keep_shape_no_padding of Ndarray.t
  | Padded of { data : Nd.t; padding : Ops.axis_padding array; padded_value : float }
[@@deriving sexp_of, equal]

type buffer = Node of Tn.t | Merge_buffer of Tn.t [@@deriving sexp_of, equal]

(** Resets a array by performing the specified computation or data fetching. *)
type fetch_op =
  | Constant of float
  | Constant_bits of int64  (** Direct bit representation, primarily for uint4x32 *)
  | Constant_fill of float array
      (** Fills in the numbers where the rightmost axis is contiguous. Primes shape inference to
          require the assigned tensor to have the same number of elements as the array, but in case
          of "leaky" shape inference, will loop over the values. This unrolls all assignments and
          should be used only for small arrays. Consider using {!Tnode.set_values} instead for
          larger arrays. *)
  | Range_over_offsets
      (** Fills in the offset number of each cell, i.e. how many cells away it is from the
          beginning, in the logical representation of the tensor node. (The actual in-memory
          positions in a buffer instantiating the node can differ.) *)
  | Slice of { batch_idx : Indexing.static_symbol; sliced : Tn.t }
  | Embed_symbol of Indexing.static_symbol
  | Embed_self_id  (** Embeds the id of the [array] field of the [Fetch] constructor. *)
  | Embed_dim of Indexing.variable_ref
[@@deriving sexp_of, equal]

type accum_rhs =
  | Ternop of { op : Ops.ternop; rhs1 : buffer; rhs2 : buffer; rhs3 : buffer }
  | Binop of { op : Ops.binop; rhs1 : buffer; rhs2 : buffer }
  | Unop of { op : Ops.unop; rhs : buffer }
  | Block of { op : Ops.unop; rhses : buffer array }
      (** [Block] and [Rev_sides] are the only assignment types that allow [Concat] axes in
          projections.

          Similar to [Unop] except it's a projection of potentially multiple tensors, e.g.
          concatenation or block tensor; or with just a single RHS, it can be a slice taking part of
          an argument axis or producing part of a result axis. The corresponding [projections] must
          use [Concat] in such a way that every choice of [Concat] components uses at most one of
          the [rhses] (note: none is allowed). Note: it is also allowed that there is no LHS (i.e.
          no valid LHS projection) for some choices of the [Concat] components. *)
  | Rev_sides of { op : Ops.unop; lhses : buffer array }
      (** Causes the [Accum_op] to completely reverse its semantics: left-hand side and right-hand
          side are swapped. [lhs] becomes the read-from tensor and [rhs], i.e. [lhses] above, become
          the written-to tensors. This is needed in particular for gradients of concatenation. *)
[@@deriving sexp_of, equal]

type t =
  | Noop
  | Seq of t * t
  | Block_comment of string * t  (** Same as the given code, with a comment. *)
  | Accum_op of {
      initialize_neutral : bool;
      accum : Ops.binop;
      lhs : Tn.t;
      rhs : accum_rhs;
      projections : Indexing.projections Lazy.t;
      projections_debug : string;
    }
  | Set_vec_unop of {
      op : Ops.vec_unop;
      lhs : Tn.t;
      rhs : buffer;
      projections : Indexing.projections Lazy.t;
      projections_debug : string;
    }
  | Fetch of { array : Tn.t; fetch_op : fetch_op; dims : int array Lazy.t }
[@@deriving sexp_of]

type comp = {
  asgns : t;
  embedded_nodes : Set.M(Tn).t;
      (** The nodes in {!field-asgns} that are not in [embedded_nodes] need to already be in
          contexts linked with the {!comp}. *)
}
[@@deriving sexp_of]
(** Computations based on assignments. Note: the [arrayjit] library makes use of, but does not
    produce nor verify the {!field-embedded_nodes} associated to some given {!field-asgns}. *)

let to_comp asgns = { asgns; embedded_nodes = Set.empty (module Tnode) }
let empty_comp = to_comp Noop

(** The buffers an [accum_rhs] mentions, in argument order. For [Rev_sides] these are the WRITTEN-TO
    buffers -- the constructor reverses the assignment's roles -- so a caller that cares about the
    direction still matches [Rev_sides], but only for the direction. Destructuring the arities lives
    here and nowhere else: before, four traversals each repeated it, so a new [accum_rhs]
    constructor meant four silent omissions waiting to happen. *)
let buffers_of_accum_rhs : accum_rhs -> buffer list = function
  | Ternop { rhs1; rhs2; rhs3; _ } -> [ rhs1; rhs2; rhs3 ]
  | Binop { rhs1; rhs2; _ } -> [ rhs1; rhs2 ]
  | Unop { rhs; _ } -> [ rhs ]
  | Block { rhses; _ } -> Array.to_list rhses
  | Rev_sides { lhses; _ } -> Array.to_list lhses

(** Whether the assignment's roles are reversed: [Rev_sides]' buffers are written and its enclosing
    [Accum_op]'s [lhs] is read. *)
let is_rev_sides = function Rev_sides _ -> true | _ -> false

(** Folds [f] over the LEAF statements ([Accum_op], [Set_vec_unop], [Fetch]) in execution order,
    skipping the [Noop] / [Seq] / [Block_comment] scaffolding. Descending is what the queries below
    used to each write for themselves, and forgetting to descend through a [Block_comment] loses
    statements silently -- so the recursion is written once. [f] still matches on {!t} (the leaf
    kinds mean different things to different queries), and its scaffolding arm is unreachable. *)
let fold_leaves (asgns : t) ~init ~f =
  let rec loop acc = function
    | Noop -> acc
    | Seq (t1, t2) -> loop (loop acc t1) t2
    | Block_comment (_, t) -> loop acc t
    | (Accum_op _ | Set_vec_unop _ | Fetch _) as leaf -> f acc leaf
  in
  loop init asgns

(** {!fold_leaves} for a [f] that only has effects. *)
let iter_leaves (asgns : t) ~f = fold_leaves asgns ~init:() ~f:(fun () leaf -> f leaf)

let can_skip_accumulation ~projections =
  (* We can skip accumulation (use = instead of +=) only if the projection is injective *)
  Affine.is_injective projections

(** Returns materialized nodes in the sense of {!Tnode.Placements.is_in_context_force}, resolved
    against the given compilation lineage's placements. NOTE: it must be called after compilation
    (when the placements of all involved nodes are settled); otherwise, it will disrupt memory mode
    inference. *)
let%debug3_sexp context_nodes ~(plc : Tn.Placements.t) (asgns : t) : Tn.t_set =
  let open Utils.Set_O in
  let empty = Set.empty (module Tn) in
  let one tn =
    if Tn.Placements.is_in_context_force plc tn 34 then Set.singleton (module Tn) tn else empty
  in
  let of_node = function Node rhs -> one rhs | Merge_buffer _ -> empty in
  fold_leaves asgns ~init:empty ~f:(fun acc leaf ->
      match leaf with
      | Accum_op { lhs; rhs; _ } ->
          Set.union_list
            (module Tn)
            (acc :: one lhs :: List.map (buffers_of_accum_rhs rhs) ~f:of_node)
      | Set_vec_unop { lhs; rhs; _ } -> acc + one lhs + of_node rhs
      (* A slice-alias view's parent must be in context too (it backs the view); the alias itself is
         dropped by [one] via [is_in_context_force] (gh-ocannl-293 293a). *)
      | Fetch { array; fetch_op = Slice { sliced; _ }; _ } -> acc + one array + one sliced
      | Fetch { array; _ } -> acc + one array
      | Noop | Seq _ | Block_comment _ -> acc)

(** In the second set, returns the nodes that are not read from after being written to. In the first
    set, returns the nodes that are ever read from. The second set is also used as the set of nodes
    to materialize; for a [Fetch.Slice] the parent is included there so it is materialized to back a
    potential zero-copy alias view (gh-ocannl-293 293a). *)
let%debug3_sexp collect_nodes_guess_output (asgns : t) : Tn.t_set * Tn.t_set =
  let open Utils.Set_O in
  let empty = Set.empty (module Tn) in
  let one = Set.singleton (module Tn) in
  let of_node = function Node rhs -> one rhs | Merge_buffer _ -> empty in
  let rec loop = function
    | Noop -> (empty, empty)
    | Seq (t1, t2) ->
        let i1, o1 = loop t1 in
        let i2, o2 = loop t2 in
        (i1 + i2, o1 + o2 - (i1 + i2))
    | Block_comment (_, t) -> loop t
    | Accum_op { lhs; rhs; _ } ->
        let buffers =
          List.fold (buffers_of_accum_rhs rhs) ~init:empty ~f:(fun acc buf -> acc + of_node buf)
        in
        (* [Rev_sides] reverses the roles: its buffers are written, and the assignment's [lhs] is
           what it reads. *)
        if is_rev_sides rhs then (one lhs, buffers) else (buffers, one lhs)
    | Set_vec_unop { lhs; rhs; _ } -> (of_node rhs, one lhs)
    (* Materialize the slice parent too, so it can back a zero-copy alias view of [array]; harmless
       in the copy-fallback case where the parent is read by the copy loop (gh-ocannl-293 293a). *)
    | Fetch { array; fetch_op = Slice { sliced; _ }; _ } ->
        (empty, Set.of_list (module Tn) [ array; sliced ])
    | Fetch { array; _ } -> (empty, one array)
  in
  loop asgns

(** All nodes that any assignment writes to (unlike the second set of {!collect_nodes_guess_output},
    nodes also read within [asgns] are included). *)
let collect_written (asgns : t) : Tn.t_set =
  let open Utils.Set_O in
  let empty = Set.empty (module Tn) in
  let one = Set.singleton (module Tn) in
  let of_node = function Node rhs -> one rhs | Merge_buffer _ -> empty in
  fold_leaves asgns ~init:empty ~f:(fun acc leaf ->
      match leaf with
      (* [Rev_sides] reverses the roles: the written-to nodes are its buffers, not the [lhs]. *)
      | Accum_op { rhs = Rev_sides _ as rhs; _ } ->
          List.fold (buffers_of_accum_rhs rhs) ~init:acc ~f:(fun acc buf -> acc + of_node buf)
      | Accum_op { lhs; _ } | Set_vec_unop { lhs; _ } -> acc + one lhs
      | Fetch { array; _ } -> acc + one array
      | Noop | Seq _ | Block_comment _ -> acc)

let sequential l =
  Option.value ~default:Noop @@ List.reduce l ~f:(fun sts another_st -> Seq (sts, another_st))

let sequence l =
  {
    asgns = sequential (List.map l ~f:(fun c -> c.asgns));
    embedded_nodes = Set.union_list (module Tn) (List.map l ~f:(fun c -> c.embedded_nodes));
  }

let collect_neutral_elem (asgns : t) : float option =
  let folded =
    fold_leaves asgns ~init:None ~f:(fun acc leaf ->
        match leaf with
        | Accum_op { accum; _ } -> (
            let neutral = Ops.neutral_elem accum in
            match acc with
            | None -> Some (Some neutral)
            | Some (Some v) when Float.( = ) v neutral -> acc
            | Some (Some _) -> Some None
            | Some None -> acc)
        | Set_vec_unop _ | Fetch _ | Noop | Seq _ | Block_comment _ -> acc)
  in
  match folded with None -> None | Some v -> v

let%track4_sexp to_low_level ?(static_indices = []) code =
  let open Indexing in
  (* gh-490 symbolic extents: wrap a loop body in [index < value] when the loop iterates a
     symbolic-extent axis AND the extent's symbol is among the routine's bindings (so the value is a
     kernel parameter). An unbound extent symbol keeps the maximum-extent semantics: the loop covers
     the whole (max-sized) buffer, exactly as if the extent were written concretely. *)
  let extent_guard ~(projections : Indexing.projections) ~index ~iter body =
    match
      List.Assoc.find projections.extent_syms
        ~equal:(Option.equal Indexing.equal_symbol)
        (Some iter)
    with
    | Some sym when List.mem static_indices sym ~equal:Indexing.equal_static_symbol ->
        let iprec = Ops.index_prec () in
        let cond =
          Low_level.Binop
            ( Ops.Cmplt,
              (Low_level.Embed_index (Indexing.Iterator index), iprec),
              (Low_level.Embed_index (Indexing.Iterator sym.Indexing.static_symbol), iprec) )
        in
        Low_level.If { cond = (cond, iprec); body }
    | _ -> body
  in
  (* Apply left padding offsets to convert from semantic to buffer indices. Semantic indices can be
     negative (e.g., -1 for convolution padding), but buffer indices must be non-negative. Adding
     left_padding converts semantic to buffer space.

     A [Concat] axis cannot arrive here: the loop nest around this access iterates a concatenation's
     segments one at a time, so what reaches an access index is the segment's own iterator. Shifting
     the axis would have to shift every segment's loop bounds instead, which is not a rewrite of one
     index -- hence the explicit refusal rather than a guess. *)
  let apply_padding_offset (tn : Tn.t) (idcs : Indexing.axis_index array) :
      Indexing.axis_index array =
    match Tn.get_padding tn with
    | None -> idcs
    | Some (padding_arr, _) ->
        Array.mapi idcs ~f:(fun i idx ->
            if i >= Array.length padding_arr then idx
            else
              let left_pad = padding_arr.(i).Ops.left in
              if left_pad = 0 then idx
              else
                match idx with
                | Fixed_idx n -> Fixed_idx (n + left_pad)
                | Iterator s -> affine ~symbols:[ (1, s) ] ~offset:left_pad
                | Affine { symbols; offset } -> affine ~symbols ~offset:(offset + left_pad)
                | Sub_axis -> Sub_axis
                | Concat _ ->
                    raise
                    @@ Utils.User_error
                         [%string
                           "Assignments.to_low_level: a concatenated axis cannot carry a padding \
                            shift (node %{Tn.debug_name tn}, axis %{i#Int}); the segments of a \
                            concatenation are iterated one loop each, so an access index is always \
                            a segment's own iterator"])
  in
  let is_padded tn = Option.is_some (Tn.get_padding tn) in
  (* gh-504 clamped windows: a padded ([=]-mode) max-family window spec registers no margin demand
     (see [Row.solve_proj_equations]'s [clamp_padded]), so its semantic indices can escape the
     operand's valid region; the clamp is rendered here as a range guard on the accumulation
     statement — an out-of-range position contributes the accumulation identity, which is the same
     as not visiting it. [index_interval] bounds a semantic index over the loop ranges of its
     symbols; [None] when a symbol's range is unknown (e.g. a static routine binding), leaving the
     access unguarded (conservative: such accesses are in-range by construction). *)
  let index_interval ~sizes (idx : Indexing.axis_index) : (int * int) option =
    match idx with
    | Indexing.Fixed_idx i -> Some (i, i)
    | Indexing.Iterator s -> Option.map (Map.find sizes s) ~f:(fun d -> (0, d - 1))
    | Indexing.Affine { symbols; offset } ->
        List.fold
          (Indexing.coalesce_affine_terms symbols)
          ~init:(Some (offset, offset))
          ~f:(fun acc (c, s) ->
            match (acc, Map.find sizes s) with
            | Some (lo, hi), Some d ->
                if c >= 0 then Some (lo, hi + (c * (d - 1))) else Some (lo + (c * (d - 1)), hi)
            | _ -> None)
    | Indexing.Sub_axis | Indexing.Concat _ -> None
  in
  (* Range-guard conditions for the axes of an access at (semantic, pre-padding-shift) [idcs] that
     can escape the operand's valid region [0, N) — unless the escape is covered by committed
     margins holding this accumulation's neutral element (the physical-halo mechanism: covered
     reads/writes of the margins are intentional). *)
  let clamp_conds ~accum ~sizes (tn : Tn.t) (idcs : Indexing.axis_index array) :
      Low_level.scalar_arg list =
    let padding = Tn.get_padding tn in
    let sem_dims = Tn.dims_without_padding tn in
    let idcs, sem_dims =
      if Array.exists idcs ~f:(function Indexing.Sub_axis -> true | _ -> false) then (
        (* Sub_axis indices describe one flat address, not independently bounded coordinates. The
           renderer retains each axis's stride while dropping Sub_axis's contribution. *)
        (match padding with
        | Some (pads, _) when Array.exists pads ~f:(fun p -> p.Ops.left <> 0 || p.Ops.right <> 0) ->
            raise
            @@ Utils.User_error
                 "Flattened Sub_axis access to a padded tensor has no supported \
                  logical-to-physical layout; materialize an unpadded copy before flattening"
        | _ -> ());
        ( [| Indexing.reflect_projection ~dims:sem_dims ~projection:idcs |],
          [| Array.fold sem_dims ~init:1 ~f:( * ) |] ))
      else (idcs, sem_dims)
    in
    let iprec = Ops.index_prec () in
    let embed idx = (Low_level.Embed_index idx, iprec) in
    Array.to_list
    @@ Array.filter_mapi idcs ~f:(fun i idx ->
        if i >= Array.length sem_dims then None
        else
          match index_interval ~sizes idx with
          | None -> None
          | Some (lo, hi) -> (
              let n = sem_dims.(i) in
              if lo >= 0 && hi < n then None
              else
                let covered =
                  match padding with
                  | Some (pads, v) ->
                      i < Array.length pads
                      && Float.(v = Ops.neutral_elem accum)
                      && lo >= -pads.(i).Ops.left
                      && hi < n + pads.(i).Ops.right
                  | None -> false
                in
                if covered then None
                else
                  (* One canonical shape per role: a lower bound is a direct [0 <= idx], an upper
                     bound the strict [idx < n]. *)
                  let lower =
                    if lo < 0 then
                      Some (Low_level.Binop (Ops.Cmple, embed (Indexing.Fixed_idx 0), embed idx))
                    else None
                  in
                  let upper =
                    if hi >= n then
                      Some (Low_level.Binop (Ops.Cmplt, embed idx, embed (Indexing.Fixed_idx n)))
                    else None
                  in
                  match (lower, upper) with
                  | Some l, Some u -> Some (Low_level.Binop (Ops.And, (l, iprec), (u, iprec)), iprec)
                  | Some c, None | None, Some c -> Some (c, iprec)
                  | None, None -> None))
  in
  let and_all (conds : Low_level.scalar_arg list) : Low_level.scalar_arg =
    List.reduce_exn conds ~f:(fun a b -> (Low_level.Binop (Ops.And, a, b), Ops.index_prec ()))
  in
  (* Redirect a slice-alias view to its parent: a read/write of the alias at [idcs] becomes a
     read/write of the parent at [batch_idx :: idcs] -- exactly the index the materializing copy
     loop used to build for its RHS. Recursive to cover (currently impossible) alias chains. The
     parent is unpadded by alias eligibility, so the downstream padding logic runs correctly against
     it (gh-ocannl-293 293a). *)
  let rec resolve_alias (tn : Tn.t) (idcs : Indexing.axis_index array) :
      Tn.t * Indexing.axis_index array =
    match Tn.alias_of tn with
    | Some (parent, { static_symbol; _ }) ->
        resolve_alias parent (Array.append [| Iterator static_symbol |] idcs)
    | None -> (tn, idcs)
  in
  (* [clamp] (gh-504): when [Some (accum, sizes, conds)], prepend to [conds] the clamp range guards
     of this access, computed on the semantic (pre-padding-shift) indices. *)
  let get ?clamp (buffer : buffer) (idcs : Indexing.axis_index array) : Low_level.scalar_t =
    (* Only [Node] buffers can be slice-alias views; [Merge_buffer] is never redirected. *)
    let buffer, idcs =
      match buffer with
      | Node tn ->
          let parent, idcs = resolve_alias tn idcs in
          (Node parent, idcs)
      | Merge_buffer _ -> (buffer, idcs)
    in
    let tn = match buffer with Node tn -> tn | Merge_buffer tn -> tn in
    let idcs =
      match (idcs, Lazy.force tn.Tn.dims) with
      | [||], [| 1 |] -> [| Fixed_idx 0 |]
      | [| Fixed_idx 0 |], [||] -> idcs
      | idcs, dims when Array.length idcs = Array.length dims -> idcs
      | _ ->
          let dims = Indexing.dims_to_string (Lazy.force tn.Tn.dims) in
          let idcs = Sexp.to_string_hum ([%sexp_of: Indexing.axis_index array] idcs) in
          invalid_arg
            [%string
              "Assignments.to_low_level: indexing mismatch for %{Tn.debug_name tn}: shape %{dims} \
               vs. %{idcs}"]
    in
    Option.iter clamp ~f:(fun (accum, sizes, conds) ->
        conds := clamp_conds ~accum ~sizes tn idcs @ !conds);
    (* The same projection can be used to access a padded or a non-padded tensor. *)
    let idcs = if is_padded tn then apply_padding_offset tn idcs else idcs in
    match buffer with
    | Node tn -> Low_level.Get (tn, idcs)
    | Merge_buffer tn -> Low_level.Get_merge_buffer (tn, idcs)
  in
  let set ?clamp (tn : Tn.t) (idcs : Indexing.axis_index array) (llsc : Low_level.scalar_t) :
      Low_level.t =
    (* Write-through a slice-alias view goes to the parent buffer (gh-ocannl-293 293a). *)
    let tn, idcs = resolve_alias tn idcs in
    let idcs =
      match (idcs, Lazy.force tn.Tn.dims) with
      | [||], [| 1 |] -> [| Fixed_idx 0 |]
      | [| Fixed_idx 0 |], [||] -> idcs
      | idcs, dims when Array.length idcs = Array.length dims -> idcs
      | _ ->
          let dims = Indexing.dims_to_string (Lazy.force tn.Tn.dims) in
          let idcs = Sexp.to_string_hum ([%sexp_of: Indexing.axis_index array] idcs) in
          invalid_arg
            [%string
              "Assignments.to_low_level: indexing mismatch for %{Tn.debug_name tn}: shape %{dims} \
               vs. %{idcs}"]
    in
    Option.iter clamp ~f:(fun (accum, sizes, conds) ->
        conds := clamp_conds ~accum ~sizes tn idcs @ !conds);
    let idcs = if is_padded tn then apply_padding_offset tn idcs else idcs in
    Low_level.Set { tn; idcs; llsc; debug = "" }
  in
  let reset_padding_regions tn neutral_value : Low_level.t list =
    match Tn.get_padding tn with
    | None -> []
    | Some (padding_arr, _) ->
        Low_level.Comment
          ("reset padding margins of " ^ Tnode.debug_name tn ^ " to "
         ^ Float.to_string neutral_value)
        :: [
             Low_level.loop_over_padding_region ~dims:(Lazy.force tn.dims) ~padding:padding_arr
               ~body:(fun idcs ->
                 Low_level.Set
                   {
                     tn;
                     idcs;
                     llsc = Low_level.Constant neutral_value;
                     debug = Tn.debug_name tn ^ " padding := " ^ Float.to_string neutral_value;
                   });
           ]
  in
  let default_padding_before array llc =
    let padding_loops =
      match Tn.get_padding array with Some (_, v) -> reset_padding_regions array v | None -> []
    in
    Low_level.unflat_lines @@ padding_loops @ [ llc ]
  in
  (* gh-ocannl-764: the product-loop machinery, written once. [Empty_block] is raised while
     substituting an index that needs a product iterator the current block did not descend into --
     the index belongs to a different concat segment, so this block contributes no statement for
     it. *)
  let exception Empty_block in
  (* The per-block substitution from product iterators to this block's fresh loop symbols, plus the
     loop widths of those fresh symbols (the gh-504 clamp bounds). Fresh loop symbols are needed
     because product iterators may be shared across different operations/tensors, but each lowered
     operation needs private loop symbols to avoid conflicts in low_level.ml's symbol-to-tensor
     tracking. [block_iters] are the product iterators of the segments descended into, [rev_iters]
     their loop symbols in reverse (innermost-first) order.

     [on_concat] is the one policy the three walkers below legitimately disagree on: the two
     accumulation walkers RESOLVE a [Concat] index to its active segment plus that segment's offset
     (this is what eliminates [Concat] during lowering), while [Set_vec_unop] REJECTS one. *)
  let block_subst ~(iter_sizes : int Map.M(Indexing.Symbol).t) ~all_prod_iters ~on_concat
      ~(block_iters : Indexing.symbol array) ~(rev_iters : Indexing.symbol list) =
    let subst_map =
      let loop_iters = Array.of_list_rev rev_iters in
      Array.map2_exn block_iters loop_iters ~f:(fun block_iter loop_iter ->
          (block_iter, Indexing.Iterator loop_iter))
      |> Array.to_list
      |> Map.of_alist_exn (module Indexing.Symbol)
    in
    (* The flat offset at which [active]'s segment starts within its concatenated axis. *)
    let concat_offset_for syms active =
      let _, offset =
        List.fold syms ~init:(0, None) ~f:(fun (cumul, found) s ->
            let size =
              match Map.find iter_sizes s with
              | Some v -> v
              | None ->
                  raise
                  @@ Utils.User_error
                       ("concat_offset_for: iterator symbol " ^ Indexing.symbol_ident s
                      ^ " absent from projection iter_sizes; a projection component was dropped")
            in
            if Indexing.equal_symbol s active then (cumul + size, Some cumul)
            else (cumul + size, found))
      in
      Option.value ~default:0 offset
    in
    let subst_index = function
      | (Indexing.Fixed_idx _ | Indexing.Sub_axis) as idx -> idx
      | Indexing.Iterator s
        when Set.mem all_prod_iters s && not (Array.mem ~equal:Indexing.equal_symbol block_iters s)
        ->
          raise Empty_block
      | Indexing.Iterator s as idx -> Option.value ~default:idx (Map.find subst_map s)
      | Indexing.Affine { symbols; offset } ->
          (* [subst_map] only ever maps to [Iterator], so the two failing branches are unreachable;
             they are spelled out rather than wildcarded so that widening the map is a build error
             right here. *)
          let symbols =
            List.map symbols ~f:(fun (coeff, s) ->
                match Map.find subst_map s with
                | Some (Indexing.Iterator s') -> (coeff, s')
                | Some (Indexing.Affine _) ->
                    failwith "Affine substitution in Affine index not supported"
                | Some (Indexing.Concat _) ->
                    failwith "Concat substitution in Affine index not supported"
                | Some (Indexing.Fixed_idx _) | Some Indexing.Sub_axis | None -> (coeff, s))
          in
          Indexing.affine ~symbols ~offset
      | Indexing.Concat syms -> (
          match on_concat with
          | `Reject who -> raise @@ Utils.User_error ("Concat indexing not supported in " ^ who)
          | `Resolve role -> (
              (* Find the active segment (the one this block descended into) and resolve to its loop
                 symbol, shifted by the segment's offset within the concatenated axis. *)
              let active =
                List.find_map syms ~f:(fun s ->
                    if Array.mem ~equal:Indexing.equal_symbol block_iters s then
                      match Map.find subst_map s with
                      | Some (Indexing.Iterator s') -> Some (s', concat_offset_for syms s)
                      | _ -> None
                    else None)
              in
              match active with
              | Some (s', 0) -> Indexing.Iterator s'
              | Some (s', offset) -> Indexing.affine ~symbols:[ (1, s') ] ~offset
              | None ->
                  raise
                  @@ Utils.User_error
                       ("Concat index could not be resolved to an active component during " ^ role
                      ^ " lowering")))
    in
    let fresh_sizes =
      Map.fold subst_map
        ~init:(Map.empty (module Indexing.Symbol))
        ~f:(fun ~key ~data acc ->
          match (data, Map.find iter_sizes key) with
          | Indexing.Iterator s', Some d -> Map.set acc ~key:s' ~data:d
          | _ -> acc)
    in
    (subst_index, fresh_sizes)
  in
  (* The product-space loop nest of one assignment: one loop per SEGMENT of each component (a fresh
     symbol each, gh-ocannl-765), extent-guarded, calling [f] at the innermost level of every block
     -- i.e. once per choice of concat segments. [f] receives the block's substitution, the fresh
     symbols' loop widths, and [is_allowed i], which says whether block buffer [i] is the one this
     block selects: a single LHS [Concat] whose segment count matches [arity] selects among the
     buffers, otherwise every buffer participates. [f] raising [Empty_block] -- directly, or from
     [subst_index] -- means this block contributes nothing. *)
  let with_product_loops ~(projections : Indexing.projections) ~lhs ~arity ~role ~f : Low_level.t =
    let all_prod_iters =
      Set.of_list (module Indexing.Symbol) (Indexing.all_iterators projections)
    in
    (* gh-ocannl-878: [extent_guard] above is per-iterator, so a bound extent whose axis has NO
       product iterator gets no guard at all -- the axis is indexed at [Fixed_idx 0] whatever the
       runtime extent is. A maximum-one symbolic axis is exactly that shape (gh-ocannl-817 retains
       its metadata under [None]), and it is reachable from ordinary [%op] code: reducing over a
       maximum-one axis bound to zero yields the operand instead of the accumulation's neutral
       element, a wrong value in an in-extent output cell.

       An enclosing [0 < extent] guard around the nest is NOT a local repair: with no product
       iterator the assignment is surjective and injective, so [can_skip_accumulation] drops the
       neutral-element initialization, and skipping the write would leave the target holding
       whatever preceded it rather than the empty reduction's zero. Making that sound means coupling
       the guard to the accumulation-elision decision, for a declaration whose symbol can only ever
       take the values 0 and 1. Refuse it loudly instead, as [Set_vec_unop] below does for its own
       unguardable extents. An extent absent from [static_indices] keeps the standing maximum-shape
       semantics. *)
    let unguardable =
      List.filter_map projections.extent_syms ~f:(fun (iter, extent) ->
          if
            List.mem static_indices extent ~equal:Indexing.equal_static_symbol
            && not (Option.exists iter ~f:(Set.mem all_prod_iters))
          then Some (Indexing.symbol_ident extent.Indexing.static_symbol)
          else None)
    in
    if not (List.is_empty unguardable) then
      raise
      @@ Utils.User_error
           [%string
             "Accum_op (%{role} lowering) of %{Tn.debug_name lhs}: bound symbolic extent(s) \
              %{String.concat ~sep:\", \" unguardable} have no iterator, so the runtime extent \
              cannot be guarded and the axis is read and written at element zero whatever the \
              extent is bound to. This is the maximum-one symbolic axis: use a concrete dimension \
              of 1, or leave the extent unbound to generate the declared maximum shape"];
    let iter_sizes = Indexing.iterator_sizes projections in
    let concat_syms_opt =
      match
        Array.filter_map projections.project_lhs ~f:(function
          | Indexing.Concat syms -> Some syms
          | _ -> None)
      with
      | [| syms |] when List.length syms = arity -> Some (Array.of_list syms)
      | _ -> None
    in
    let basecase block_iters rev_iters =
      let block_iters = Array.of_list_rev block_iters in
      let subst_index, fresh_sizes =
        block_subst ~iter_sizes ~all_prod_iters ~on_concat:(`Resolve role) ~block_iters ~rev_iters
      in
      let is_allowed i =
        match concat_syms_opt with
        | None -> true
        | Some syms -> Array.mem ~equal:Indexing.equal_symbol block_iters syms.(i)
      in
      try f ~subst_index ~fresh_sizes ~is_allowed with Empty_block -> Low_level.Noop
    in
    let rec for_loop block_iters rev_iters = function
      | [] -> basecase block_iters rev_iters
      | comp :: product ->
          Low_level.unflat_lines
          @@ List.map comp ~f:(fun (d, iter) ->
              (* One fresh symbol per SEGMENT, not per component: a concatenation component's
                 segments become sibling loops with DIFFERENT bounds, and a shared binder makes
                 every flat symbol-keyed scanner misread them -- [def_loop_ranges] keeps only the
                 last segment's width, [affine_accesses] collects two ranges for one symbol, and the
                 canonical render reports the second binder as shadowed, declining the routine for
                 both digest caches (gh-ocannl-765). *)
              let index = Indexing.get_symbol () in
              Low_level.For_loop
                {
                  index;
                  from_ = 0;
                  to_ = d - 1;
                  body =
                    extent_guard ~projections ~index ~iter
                      (for_loop (iter :: block_iters) (index :: rev_iters) product);
                  axis = Serial;
                })
    in
    for_loop [] [] (Array.to_list projections.components)
  in
  let rec loop_accum ~initialize_neutral ~accum ~(op : Ops.op) ~lhs ~rhses projections : Low_level.t
      =
    let projections : Indexing.projections = Lazy.force projections in
    let for_loops =
      with_product_loops ~projections ~lhs ~arity:(Array.length rhses) ~role:"Block"
        ~f:(fun ~subst_index ~fresh_sizes ~is_allowed ->
          let lhs_idcs : Indexing.axis_index array =
            Array.map projections.project_lhs ~f:subst_index
          in
          let open Low_level in
          let lhs_conds = ref [] and rhs_conds = ref [] in
          let lhs_ll = get (Node lhs) lhs_idcs in
          let rhses_ll =
            Array.filter_mapi projections.project_rhs ~f:(fun i rhs_idcs ->
                try
                  if not (is_allowed i) then None
                  else
                    let rhs_idcs = Array.map ~f:subst_index rhs_idcs in
                    Some (get ~clamp:(accum, fresh_sizes, rhs_conds) rhses.(i) rhs_idcs)
                with Empty_block -> None)
          in
          if Array.is_empty rhses_ll then raise Empty_block;
          let rhs2 =
            try apply_op op rhses_ll
            with Invalid_argument _ ->
              raise
              @@ Utils.User_error
                   "Ambiguous indices in concatenation: multiple blocks viable for same position"
          in
          (* Out-of-range reads contribute the accumulation identity: exactly the semantics of a
             padded window spec, so clamping is semantically exact (gh-504). *)
          let rhs2 =
            match !rhs_conds with
            | [] -> rhs2
            | conds ->
                let cond, _ = and_all conds in
                apply_op (Ops.Ternop Ops.Where) [| cond; rhs2; Constant (Ops.neutral_elem accum) |]
          in
          let clamp = (accum, fresh_sizes, lhs_conds) in
          let stmt =
            if initialize_neutral && can_skip_accumulation ~projections then
              set ~clamp lhs lhs_idcs rhs2
            else set ~clamp lhs lhs_idcs @@ apply_op (Ops.Binop accum) [| lhs_ll; rhs2 |]
          in
          (* An out-of-range write target (the transposed clamp of a backward scatter) skips the
             whole statement. *)
          match !lhs_conds with
          | [] -> stmt
          | conds -> Low_level.If { cond = and_all conds; body = stmt })
    in
    (* Need initialization if: initialize_neutral is true AND (not surjective OR not injective)

       Not surjective: some positions never written (need init to avoid garbage)

       Not injective: accumulation needed (need init for first += operation) *)
    let needs_init =
      initialize_neutral && not (Affine.is_surjective projections && Affine.is_injective projections)
    in
    (* The padding neutral element is part of a padded tensor's identity: margins permanently hold
       the committed value (conflicting margin-touching demands are rejected at shape-inference
       time, and valid-window readers never see the margins), so operands need no resets here.
       Establish the committed neutral element in the lhs margins: only hosted / host-initialized
       buffers are creation-filled, device buffers are allocated raw — so the (idempotent) fill
       accompanies every writer of a padded node. *)
    let neutral_value = Ops.neutral_elem accum in
    let padding_resets =
      match Lazy.force lhs.padding with Some (_, v) -> reset_padding_regions lhs v | None -> []
    in
    let for_loops_with_resets =
      if List.is_empty padding_resets then for_loops
      else Low_level.unflat_lines (padding_resets @ [ for_loops ])
    in
    if needs_init then
      let dims = lazy projections.lhs_dims in
      let fetch_op = Constant neutral_value in
      Low_level.Seq (loop (Fetch { array = lhs; fetch_op; dims }), for_loops_with_resets)
    else for_loops_with_resets
  and loop_accum_rev ~initialize_neutral ~accum ~(op : Ops.op) ~lhs ~lhses projections : Low_level.t
      =
    let projections : Indexing.projections = Lazy.force projections in
    let selector =
      match
        Array.filter_map projections.project_lhs ~f:(function
          | Indexing.Concat syms -> Some syms
          | _ -> None)
      with
      | [| syms |] when List.length syms = Array.length lhses -> Some (Array.of_list syms)
      | _ -> None
    in
    let target_projections =
      Array.mapi projections.project_rhs ~f:(fun i project_lhs ->
          (* Keep inactive symbols in the domain: removing one would make the source availability
             proof mistake it for an always-available static parameter. *)
          let selects_target =
            match selector with
            | None -> true
            | Some syms ->
                let selected = syms.(i) in
                Array.exists project_lhs ~f:(Indexing.axis_index_mentions_symbol selected)
                && Array.for_all projections.components ~f:(fun comp ->
                    if List.exists comp ~f:(fun (_, s) -> Indexing.equal_symbol s selected) then
                      let others =
                        List.filter_map comp ~f:(fun (_, s) ->
                            if Indexing.equal_symbol s selected then None else Some s)
                      in
                      not (Array.exists project_lhs ~f:(Indexing.axis_index_mentions_any others))
                    else true)
          in
          if not selects_target then None
          else
            Some
              {
                projections with
                lhs_dims = projections.rhs_dims.(i);
                project_lhs;
                rhs_dims = [| projections.lhs_dims |];
                project_rhs = [| projections.project_lhs |];
              })
    in
    let target_can_skip =
      Array.map target_projections ~f:(fun proj ->
          Option.value_map proj ~default:false ~f:(fun projections ->
              can_skip_accumulation ~projections))
    in
    let target_needs_init =
      Array.map target_projections ~f:(fun proj ->
          initialize_neutral
          && Option.value_map proj ~default:true ~f:(fun proj ->
              not (Affine.is_surjective proj && Affine.is_injective proj)))
    in
    let target_tn_exn = function
      | Node tn -> tn
      | Merge_buffer _ -> raise @@ Utils.User_error "Rev_sides cannot write to merge buffers"
    in
    (* Same walker as [loop_accum], the roles swapped: the assignment's [lhs] is read at
       [project_lhs] and the [lhses] are written at their [project_rhs]. *)
    let for_loops =
      with_product_loops ~projections ~lhs ~arity:(Array.length lhses) ~role:"Rev_sides"
        ~f:(fun ~subst_index ~fresh_sizes ~is_allowed ->
          let rhs_idcs : Indexing.axis_index array =
            Array.map projections.project_lhs ~f:subst_index
          in
          let open Low_level in
          let lhs_conds = ref [] and rhs_conds = ref [] in
          let rhs_ll = get ~clamp:(accum, fresh_sizes, rhs_conds) (Node lhs) rhs_idcs in
          let targets =
            Array.filter_mapi projections.project_rhs ~f:(fun i lhs_idcs ->
                try
                  if not (is_allowed i) then None
                  else
                    let lhs_idcs = Array.map ~f:subst_index lhs_idcs in
                    Some (i, lhses.(i), lhs_idcs)
                with Empty_block -> None)
          in
          if Array.is_empty targets then raise Empty_block;
          if Array.length targets > 1 then
            raise
            @@ Utils.User_error
                 "Ambiguous indices in concatenation: multiple blocks viable for same position";
          let i, target_buf, lhs_idcs = targets.(0) in
          let rhs2 = apply_op op [| rhs_ll |] in
          let rhs2 =
            match !rhs_conds with
            | [] -> rhs2
            | conds ->
                let cond, _ = and_all conds in
                apply_op (Ops.Ternop Ops.Where) [| cond; rhs2; Constant (Ops.neutral_elem accum) |]
          in
          let target_tn = target_tn_exn target_buf in
          let clamp = (accum, fresh_sizes, lhs_conds) in
          let stmt =
            if initialize_neutral && target_can_skip.(i) then set ~clamp target_tn lhs_idcs rhs2
            else
              set ~clamp target_tn lhs_idcs
              @@ apply_op (Ops.Binop accum) [| get target_buf lhs_idcs; rhs2 |]
          in
          match !lhs_conds with
          | [] -> stmt
          | conds -> Low_level.If { cond = and_all conds; body = stmt })
    in
    let neutral_value = Ops.neutral_elem accum in
    (* Establish the committed neutral element in the lhs margins (device buffers are allocated
       raw). *)
    let padding_resets =
      match Lazy.force lhs.padding with Some (_, v) -> reset_padding_regions lhs v | None -> []
    in
    let for_loops_with_resets =
      if List.is_empty padding_resets then for_loops
      else Low_level.unflat_lines (padding_resets @ [ for_loops ])
    in
    let init_ops =
      Array.filter_mapi lhses ~f:(fun i buf ->
          if not target_needs_init.(i) then None
          else
            let array =
              match buf with
              | Node tn -> tn
              | Merge_buffer _ ->
                  raise @@ Utils.User_error "Rev_sides cannot initialize merge buffers"
            in
            Some
              (Fetch
                 { array; fetch_op = Constant neutral_value; dims = lazy projections.rhs_dims.(i) }))
      |> Array.to_list
    in
    if List.is_empty init_ops then for_loops_with_resets
    else Low_level.unflat_lines (List.map init_ops ~f:loop @ [ for_loops_with_resets ])
  and loop (code : t) : Low_level.t =
    match code with
    | Accum_op { initialize_neutral; accum; lhs; rhs; projections; _ } -> (
        let op, rhses =
          match rhs with
          | Unop { op; rhs } -> (Ops.Unop op, [| rhs |])
          | Binop { op; rhs1; rhs2 } -> (Ops.Binop op, [| rhs1; rhs2 |])
          | Ternop { op; rhs1; rhs2; rhs3 } -> (Ops.Ternop op, [| rhs1; rhs2; rhs3 |])
          | Block { op; rhses } -> (Ops.Unop op, rhses)
          | Rev_sides { op; lhses } -> (Ops.Unop op, lhses)
        in
        match rhs with
        | Rev_sides _ -> loop_accum_rev ~initialize_neutral ~accum ~op ~lhs ~lhses:rhses projections
        | _ -> loop_accum ~initialize_neutral ~accum ~op ~lhs ~rhses projections)
    | Set_vec_unop { op; lhs; rhs; projections; _ } ->
        (* Handle vector unary operations *)
        let projections = Lazy.force projections in
        let full_length =
          match op with
          | Ops.Uint4x32_to_prec_uniform ->
              (* Prevent over-eager guard against forcing precision. *)
              ignore (Lazy.force lhs.dims);
              Ops.vec_unop_lanes (Lazy.force lhs.storage_prec)
        in
        (* [Set_from_vec] stores [length] lanes at flat consecutive offsets; a padded (halo) target
           breaks that assumption across rows, and would make the random stream layout-dependent.
           Reject explicitly with a remedy. *)
        (match Tn.get_padding lhs with
        | None -> ()
        | Some (pads, _) when Array.for_all pads ~f:(fun p -> p.Ops.left = 0 && p.Ops.right = 0) ->
            ()
        | Some _ ->
            raise
            @@ Utils.User_error
                 [%string
                   "Set_vec_unop (packed uniform): target %{Tn.debug_name lhs} is padded; \
                    materialize the random tensor into an unpadded node and copy it into the \
                    padded one instead"]);
        (* gh-ocannl-817: a packed vector store cannot be clipped at a runtime extent one lane at a
           time. Shape inference currently rejects this combination earlier because the packed
           sampler's round-up [Total_elems] relation cannot include [Row.Sym], but keep the lowering
           boundary loud: a future projection path must not make the combination reachable as an
           unguarded wrong-write. An extent absent from [static_indices] deliberately retains the
           standing maximum-extent semantics, just as [extent_guard] does above. *)
        let bound_extent_symbols =
          List.filter_map projections.extent_syms ~f:(fun (_, extent) ->
              if List.mem static_indices extent ~equal:Indexing.equal_static_symbol then
                Some (Indexing.symbol_ident extent.static_symbol)
              else None)
        in
        if not (List.is_empty bound_extent_symbols) then
          raise
          @@ Utils.User_error
               [%string
                 "Set_vec_unop (packed uniform): target %{Tn.debug_name lhs} has bound symbolic \
                  extent(s) %{String.concat ~sep:\", \" bound_extent_symbols}; packed vector \
                  stores cannot honor a runtime extent safely. Use a concrete extent or leave the \
                  extent unbound to generate the declared maximum shape"];
        (* Tail peel: when the target's total element count is not a multiple of [full_length], the
           final counter iteration stores only the remaining lanes of its 128-bit block. *)
        let total_elems = Tn.num_elems lhs in
        let rem = total_elems % full_length in
        (* The third product-loop walker (gh-ocannl-764). It reuses [block_subst] -- so an index
           form is substituted in ONE place for all three -- but keeps its own loop nest, and the
           four ways it differs from [with_product_loops] are decisions, not drift:

           1. The nest is SPLIT, not uniform: the counter axis's last iteration is peeled into its
           own loop so the tail store writes only [rem] lanes. A shared walker would have to take a
           per-component split policy to express that, which is the whole of what is specific here.
           2. [Concat] is REJECTED, not resolved ([`Reject]): [Set_from_vec] stores [length] lanes
           at flat consecutive offsets from one base index, which a segment offset would make
           straddle two segments. The non-singleton component is refused by [for_loop] below, and
           this refuses a [Concat] index arriving by any other route. 3. [Empty_block] cannot fire
           here: every component is a singleton (2) and every level is entered, so [block_iters]
           covers all product iterators. Going through the shared policy anyway means an iterator
           that somehow escaped that would fail the lowering rather than survive as a silently
           unsubstituted symbol. 4. No [extent_guard] and no gh-504 clamp: a bound symbolic extent
           is rejected above, while padded-window clamping reaches the accumulation walkers only.
           Wiring either in here would be a behavior change, not a dedup. *)
        let all_prod_iters =
          Set.of_list (module Indexing.Symbol) (Indexing.all_iterators projections)
        in
        let iter_sizes = Indexing.iterator_sizes projections in
        let basecase ~length block_iters rev_iters =
          let subst_index, _fresh_sizes =
            block_subst ~iter_sizes ~all_prod_iters ~on_concat:(`Reject "Set_vec_unop")
              ~block_iters:(Array.of_list_rev block_iters) ~rev_iters
          in
          let lhs_idcs = Array.map projections.project_lhs ~f:subst_index in
          let rhs_idcs = Array.map projections.project_rhs.(0) ~f:subst_index in
          let open Low_level in
          let rhs_ll = get rhs rhs_idcs in
          (* Redirect a vector store through a slice-alias view to the parent, mirroring [set] for
             scalar stores (gh-ocannl-293 293a). Without this the alias [lhs] -- which owns no
             buffer and is excluded from [ctx_buffers] -- would be a write target the backend cannot
             link. The parent is unpadded by alias eligibility, and its precision matches the
             slice's, so [length] (computed from [lhs.storage_prec] above) stays correct. *)
          let lhs, lhs_idcs = resolve_alias lhs lhs_idcs in
          Set_from_vec
            {
              tn = lhs;
              idcs = lhs_idcs;
              length;
              vec_unop = op;
              arg = (rhs_ll, Low_level.scalar_precision rhs_ll);
              debug = "";
            }
        in
        let peel_axis =
          if rem = 0 then -1
          else if total_elems <= full_length then
            (* A single (partial) block: the counter axis is dim-1 and typically has no iterator in
               the projections; every store is the tail store (handled by starting in tail mode
               below). *)
            -1
          else begin
            (* The counter operand is a single axis: exactly one product axis drives the argument's
               projection, and it is the one to peel. *)
            let rhs_symbols =
              Array.to_list projections.project_rhs.(0)
              |> List.concat_map ~f:(function
                | Indexing.Iterator s -> [ s ]
                | Indexing.Affine { symbols; _ } -> List.map symbols ~f:snd
                | Indexing.Fixed_idx _ | Indexing.Sub_axis | Indexing.Concat _ -> [])
            in
            let driving =
              Array.filter_mapi projections.components ~f:(fun i comp ->
                  match comp with
                  | [ (_, s) ] when List.mem rhs_symbols s ~equal:Indexing.Symbol.equal -> Some i
                  | _ -> None)
            in
            match Array.to_list driving with
            | [ i ] ->
                (* The peel below assumes the flat store offset is driven solely by the peeled
                   counter axis: every other product axis must be degenerate. *)
                let others_product =
                  Array.foldi projections.components ~init:1 ~f:(fun j acc comp ->
                      if j = i then acc else List.fold comp ~init:acc ~f:(fun acc (d, _) -> acc * d))
                in
                if others_product <> 1 then
                  raise
                  @@ Utils.User_error
                       [%string
                         "Set_vec_unop (packed uniform): non-divisible target size \
                          %{total_elems#Int} (lanes per block: %{full_length#Int}) requires a 1-D \
                          counter iteration"]
                else i
            | _ ->
                raise
                @@ Utils.User_error
                     [%string
                       "Set_vec_unop (packed uniform): non-divisible target size \
                        %{total_elems#Int} (lanes per block: %{full_length#Int}) but could not \
                        identify the counter axis to peel"]
          end
        in
        let rec for_loop ~tail block_iters rev_iters = function
          | [] -> basecase ~length:(if tail then rem else full_length) block_iters rev_iters
          | (i, [ (d, iter) ]) :: product when i = peel_axis ->
              (* Peel the final counter iteration: interior stores stay full-width and guard-free,
                 the last store writes the [rem] remaining lanes. *)
              let make ~tail ~from_ ~to_ =
                let index = Indexing.get_symbol () in
                Low_level.For_loop
                  {
                    index;
                    from_;
                    to_;
                    body = for_loop ~tail (iter :: block_iters) (index :: rev_iters) product;
                    axis = Serial;
                  }
              in
              let tail_loop = make ~tail:true ~from_:(d - 1) ~to_:(d - 1) in
              if d = 1 then tail_loop
              else Low_level.Seq (make ~tail:false ~from_:0 ~to_:(d - 2), tail_loop)
          | (_, [ (d, iter) ]) :: product ->
              let index = Indexing.get_symbol () in
              Low_level.For_loop
                {
                  index;
                  from_ = 0;
                  to_ = d - 1;
                  body = for_loop ~tail (iter :: block_iters) (index :: rev_iters) product;
                  axis = Serial;
                }
          | _ -> raise @@ Utils.User_error "Concat indexing not supported in Set_vec_unop"
        in
        for_loop
          ~tail:(rem <> 0 && total_elems <= full_length)
          [] []
          (List.mapi ~f:(fun i comp -> (i, comp)) (Array.to_list projections.components))
    | Noop -> Low_level.Noop
    | Block_comment (s, c) -> Low_level.unflat_lines [ Comment s; loop c; Comment "end" ]
    | Seq (c1, c2) ->
        let c1 = loop c1 in
        let c2 = loop c2 in
        Low_level.Seq (c1, c2)
    | Fetch { array; fetch_op = Constant 0.0; dims = _ } ->
        (* [Zero_out] covers the whole buffer including the margins, so a nonzero committed neutral
           element must be re-established after it (a zero neutral is already correct). *)
        let padding_after =
          match Tn.get_padding array with
          | Some (_, v) when Float.( <> ) v 0.0 -> reset_padding_regions array v
          | _ -> []
        in
        Low_level.unflat_lines (Low_level.Zero_out array :: padding_after)
    | Fetch { array; fetch_op = Constant c; dims } ->
        default_padding_before array
        @@ Low_level.loop_over_dims (Lazy.force dims) ~body:(fun idcs ->
            set array idcs @@ Constant c)
    | Fetch { array; fetch_op = Constant_bits i; dims } ->
        default_padding_before array
        @@ Low_level.loop_over_dims (Lazy.force dims) ~body:(fun idcs ->
            set array idcs @@ Constant_bits i)
    | Fetch { array; fetch_op = Slice { batch_idx = { static_symbol = idx; _ }; sliced }; dims } ->
        if Tn.is_alias array then
          (* Zero-copy alias view: no materialization. Reads/writes of [array] are redirected to
             [sliced] (with [idx] prepended) by [get]/[set] (gh-ocannl-293 293a). *)
          Low_level.Noop
        else
          default_padding_before array
          @@ Low_level.loop_over_dims (Lazy.force dims) ~body:(fun idcs ->
              set array idcs @@ get (Node sliced) @@ Array.append [| Iterator idx |] idcs)
    | Fetch { array; fetch_op = Embed_symbol s; dims } ->
        default_padding_before array
        @@ Low_level.loop_over_dims (Lazy.force dims) ~body:(fun idcs ->
            set array idcs @@ Embed_index (Iterator s.static_symbol))
    | Fetch { array; fetch_op = Embed_self_id; dims } ->
        default_padding_before array
        @@ Low_level.loop_over_dims (Lazy.force dims) ~body:(fun idcs ->
            set array idcs @@ Constant_bits (Int64.of_int array.id))
    | Fetch { array; fetch_op = Embed_dim variable_ref; dims } ->
        (* Forcing [dims] first forces shape inference to complete, which is what fills in
           [variable_ref]: it must happen before reading the solved dimension (this fetch may be the
           first statement lowered in the routine). *)
        let dims = Lazy.force dims in
        let dim_value =
          match (variable_ref.Indexing.solved_dim, variable_ref.Indexing.solved_sym) with
          | Some d, _ -> d
          | None, Some { Indexing.static_range = Some range; _ } ->
              (* A symbolic extent (gh-490) materializes at its declared maximum. *)
              range
          | None, Some { Indexing.static_range = None; _ } | None, None ->
              raise
              @@ Utils.User_error
                   ("Embed_dim: variable reference " ^ variable_ref.Indexing.ref_label
                  ^ " has no solved dimension")
        in
        default_padding_before array
        @@ Low_level.loop_over_dims dims ~body:(fun idcs ->
            set array idcs @@ Constant (Float.of_int dim_value))
    | Fetch { array; fetch_op = Range_over_offsets; dims = (lazy dims) } ->
        default_padding_before array
        @@ Low_level.loop_over_dims dims ~body:(fun idcs ->
            let offset = Indexing.reflect_projection ~dims ~projection:idcs in
            set array idcs @@ Embed_index offset)
    | Fetch { array; fetch_op = Constant_fill values; dims = (lazy dims) } ->
        (* TODO: consider failing here and strengthening shape inference. *)
        let size = Array.length values in
        let limit_constant_fill_size =
          Int.of_string @@ Utils.get_global_arg ~default:"16" ~arg_name:"limit_constant_fill_size"
        in
        if size > limit_constant_fill_size then
          raise
          @@ Utils.User_error
               [%string
                 "Constant_fill size is too large to unroll for %{Tn.debug_name array} (size: \
                  %{size#Int}, limit: %{limit_constant_fill_size#Int}), either increase \
                  ocannl_limit_constant_fill_size or use Tnode.set_values instead"];
        default_padding_before array
        @@ Low_level.unroll_dims dims ~body:(fun idcs ~offset ->
            set array idcs @@ Constant values.(offset % size))
  in
  (* Pre-pass: mark alias-eligible [Fetch.Slice]s before lowering, so [get]/[set] redirect them and
     the [Slice] lowering emits no copy loop. Eligibility needs forced shapes, which are available
     here (shape inference is forced before [lower]/[to_low_level]). Idempotent across re-lowerings.
     A slice falls back to the materializing copy loop unless ALL hold: - leading-axis,
     rank-drop-by-one (the only shape [Slice] produces): parent rank = child rank + 1 and the
     trailing dims match elementwise; - parent and child share the same precision: the copy loop
     silently converts precision (e.g. a float buffer sliced then reinterpreted as uint4x32), which
     a shared-storage alias cannot do; - the parent has backing storage (not [Virtual] /
     [Effectively_constant]); - parent and child are both unpadded (aliasing would otherwise break
     the padding contract). (gh-ocannl-293 subtask 293a.) *)
  let slice_alias_eligible ~(array : Tn.t) ~(sliced : Tn.t) : bool =
    let pdims = Lazy.force sliced.Tn.dims and cdims = Lazy.force array.Tn.dims in
    Array.length pdims = Array.length cdims + 1
    && Array.equal Int.equal (Array.subo pdims ~pos:1) cdims
    && Ops.equal_prec (Lazy.force array.Tn.storage_prec) (Lazy.force sliced.Tn.storage_prec)
    (* Alias-ness is a semantic fact settled deterministically at assignments lowering, BEFORE
       per-lineage placement decisions diverge (context-scoped memory modes, category 1). So
       eligibility may consult only lineage-independent facts -- shapes, precision, padding, and the
       parent's DECLARED INTENT -- never a lineage's placements: the alias mark is cached globally
       on the tnode, and a lineage-dependent eligibility input would let one lineage's alias
       redirect accesses to a parent that another lineage virtualized (PR #93 review). Conversely,
       confirming an alias declares [On_device] intent on the parent (see [mark_aliases]), so no
       lineage can resolve the backing buffer away from under the view. *)
    && (not (Tn.known_virtual sliced))
    && (not (Tn.known_constant sliced))
    && Option.is_none (Tn.get_padding sliced)
    && Option.is_none (Tn.get_padding array)
  in
  let mark_aliases (c : t) : unit =
    iter_leaves c ~f:(function
      | Fetch { array; fetch_op = Slice { batch_idx; sliced }; dims = _ } ->
          if slice_alias_eligible ~array ~sliced then (
            (* The view's write semantics (a write through [array] is a write to [sliced]'s
               sub-range, potentially observed by a later routine) require the parent to own a
               persistent buffer in EVERY lineage that lowers this alias. Declare the intent
               globally, like the alias mark itself -- monotone and idempotent; mirrors
               [collect_nodes_guess_output]'s materialization of slice parents. Provenance 27. *)
            Tn.update_memory_mode sliced On_device 27;
            Tn.set_alias_of array ~parent:sliced ~batch_idx)
      | Fetch _ | Accum_op _ | Set_vec_unop _ | Noop | Seq _ | Block_comment _ -> ())
  in
  mark_aliases code;
  loop code

let get_ident_within_code ?no_dots c =
  let ident_style = Tn.get_style ~arg_name:"cd_ident_style" ?no_dots () in
  let nograd_idents = Hashtbl.create (module String) in
  let grad_idents = Hashtbl.create (module String) in
  let visit tn =
    let is_grad, ident = Tn.no_grad_ident_label tn in
    let idents = if is_grad then grad_idents else nograd_idents in
    Option.iter ident
      ~f:
        (Hashtbl.update idents ~f:(fun old ->
             Set.add (Option.value ~default:Utils.no_ints old) tn.uid))
  in
  let tn = function Node tn -> tn | Merge_buffer tn -> tn in
  iter_leaves c ~f:(function
    | Accum_op { lhs; rhs; _ } ->
        List.iter ~f:visit (lhs :: List.map (buffers_of_accum_rhs rhs) ~f:tn)
    | Set_vec_unop { lhs; rhs; _ } -> List.iter ~f:visit [ lhs; tn rhs ]
    | Fetch { array; _ } -> visit array
    | Noop | Seq _ | Block_comment _ -> ());
  let repeating_nograd_idents =
    Hashtbl.filter nograd_idents ~f:(fun ids -> List.length (Set.to_list ids) > 1)
  in
  let repeating_grad_idents =
    Hashtbl.filter grad_idents ~f:(fun ids -> List.length (Set.to_list ids) > 1)
  in
  fun tn ->
    let ident = Tn.styled_ident ~repeating_nograd_idents ~repeating_grad_idents ident_style tn in
    Tn.update_code_name tn ident;
    ident

let to_doc ?name ?static_indices () c =
  let ident = get_ident_within_code c in
  let buffer_ident = function Node tn -> ident tn | Merge_buffer tn -> ident tn ^ ".merge" in

  let open PPrint in
  let doc_of_fetch_op (op : fetch_op) =
    match op with
    | Constant f -> string (Float.to_string f)
    | Constant_bits i -> string (Printf.sprintf "bits(%LdLL)" i)
    | Constant_fill values ->
        let values_str =
          String.concat ~sep:", " (Array.to_list (Array.map values ~f:Float.to_string))
        in
        string ("constant_fill([" ^ values_str ^ "])")
    | Range_over_offsets -> string "range_over_offsets()"
    | Slice { batch_idx; sliced } ->
        string (ident sliced ^ " @| " ^ Indexing.symbol_ident batch_idx.static_symbol)
    | Embed_symbol { static_symbol; static_range = _; used_as_extent = _; used_as_slice = _ } ->
        string ("!@" ^ Indexing.symbol_ident static_symbol)
    | Embed_self_id -> string "self_id()"
    | Embed_dim { ref_label; _ } -> string ("(dim " ^ ref_label ^ ")")
  in

  let rec doc_of_code = function
    | Noop -> empty
    | Seq (c1, c2) -> doc_of_code c1 ^^ doc_of_code c2
    | Block_comment (s, Noop) -> string ("# \"" ^ s ^ "\";") ^^ break 1
    | Block_comment (s, c) -> string ("# \"" ^ s ^ "\";") ^^ break 1 ^^ doc_of_code c
    | Accum_op { initialize_neutral; accum; lhs; rhs; projections_debug; _ } -> (
        let proj_spec = projections_debug in
        match rhs with
        | Ternop { op; rhs1; rhs2; rhs3 } ->
            (* Uncurried syntax for ternary operations. *)
            string (ident lhs)
            ^^ space
            ^^ string (Ops.assign_op_cd_syntax ~initialize_neutral accum)
            ^^ space
            ^^ string (Ops.ternop_cd_syntax op)
            ^^ string "("
            ^^ string (buffer_ident rhs1)
            ^^ string ", "
            ^^ string (buffer_ident rhs2)
            ^^ string ", "
            ^^ string (buffer_ident rhs3)
            ^^ string ")"
            ^^ (if not (String.equal proj_spec ".") then string (" ~logic:\"" ^ proj_spec ^ "\"")
                else empty)
            ^^ string ";" ^^ break 1
        | Binop { op; rhs1; rhs2 } ->
            string (ident lhs)
            ^^ space
            ^^ string (Ops.assign_op_cd_syntax ~initialize_neutral accum)
            ^^ space
            ^^ string (buffer_ident rhs1)
            ^^ space
            ^^ string (Ops.binop_cd_syntax op)
            ^^ space
            ^^ string (buffer_ident rhs2)
            ^^ (if
                  (not (String.equal proj_spec "."))
                  || List.mem ~equal:Ops.equal_binop Ops.[ Mul; Div ] op
                then string (" ~logic:\"" ^ proj_spec ^ "\"")
                else empty)
            ^^ string ";" ^^ break 1
        | Unop { op; rhs } ->
            string (ident lhs)
            ^^ space
            ^^ string (Ops.assign_op_cd_syntax ~initialize_neutral accum)
            ^^ space
            ^^ (if not @@ Ops.equal_unop op Ops.Identity then string (Ops.unop_cd_syntax op ^ " ")
                else empty)
            ^^ string (buffer_ident rhs)
            ^^ (if not (String.equal proj_spec ".") then string (" ~logic:\"" ^ proj_spec ^ "\"")
                else empty)
            ^^ string ";" ^^ break 1
        | Block { op; rhses } ->
            (* TODO: Pretty-print Block operations *)
            string (ident lhs)
            ^^ string (Ops.assign_op_cd_syntax ~initialize_neutral accum)
            ^^ space
            ^^ (if not @@ Ops.equal_unop op Ops.Identity then string (Ops.unop_cd_syntax op ^ " ")
                else empty)
            ^^ brackets
                 (separate (semi ^^ space)
                    (Array.to_list (Array.map rhses ~f:(Fn.compose string buffer_ident))))
            ^^ (if not (String.equal proj_spec ".") then string (" ~logic:\"" ^ proj_spec ^ "\"")
                else empty)
            ^^ string ";" ^^ break 1
        | Rev_sides { op; lhses } ->
            brackets
              (separate (semi ^^ space)
                 (Array.to_list (Array.map lhses ~f:(Fn.compose string buffer_ident))))
            ^^ space
            ^^ string (Ops.assign_op_cd_syntax ~initialize_neutral accum)
            ^^ space
            ^^ (if not @@ Ops.equal_unop op Ops.Identity then string (Ops.unop_cd_syntax op ^ " ")
                else empty)
            ^^ string (ident lhs)
            ^^ (if not (String.equal proj_spec ".") then string (" ~logic:\"" ^ proj_spec ^ "\"")
                else empty)
            ^^ string ";" ^^ break 1)
    | Set_vec_unop { op; lhs; rhs; projections = _; projections_debug } ->
        let proj_spec = projections_debug in
        string (ident lhs)
        ^^ space
        ^^ string (Ops.assign_op_cd_syntax ~initialize_neutral:false Arg2)
        ^^ space
        ^^ string (Ops.vec_unop_cd_syntax op)
        ^^ space
        ^^ string (buffer_ident rhs)
        ^^ (if not (String.equal proj_spec ".") then string (" ~logic:\"" ^ proj_spec ^ "\"")
            else empty)
        ^^ string ";" ^^ break 1
    | Fetch { array; fetch_op; dims = _ } ->
        string (ident array) ^^ string " =: " ^^ doc_of_fetch_op fetch_op ^^ string ";" ^^ break 1
  in

  (* Create the header document *)
  let header_doc =
    match (name, static_indices) with
    | Some n, Some si ->
        string (n ^ " (")
        ^^ separate (comma ^^ space) (List.map si ~f:Indexing.Doc_helpers.pp_static_symbol)
        ^^ string "):" ^^ space
    | Some n, None -> string (n ^ ":") ^^ space
    | _ -> empty
  in

  header_doc ^^ nest 2 (doc_of_code c)

let to_string c =
  let doc = to_doc () c in
  let b = Buffer.create 100 in
  PPrint.ToBuffer.pretty 0.7 100 b doc;
  Buffer.contents b

let get_name_exn asgns =
  let punct_or_sp = Str.regexp "[-@*/:.;, ]" in
  let punct_and_sp = Str.regexp {|[-@*/:.;,]\( |$\)|} in
  let rec loop = function
    | Block_comment (s, _) ->
        Str.global_replace punct_and_sp "" s |> Str.global_replace punct_or_sp "_"
    | Seq (t1, t2) ->
        let n1 = loop t1 and n2 = loop t2 in
        let prefix = String.common_prefix2_length n1 n2 in
        let suffix = String.common_suffix2_length n1 n2 in
        if String.is_empty n1 || String.is_empty n2 then n1 ^ n2
        else String.drop_suffix n1 suffix ^ "_then_" ^ String.drop_prefix n2 prefix
    | _ -> ""
  in
  let result = loop asgns in
  if String.is_empty result then
    invalid_arg ("Assignments.get_name_exn: no comments in code: " ^ to_string asgns)
  else result

let%track6_sexp lower optim_ctx ~unoptim_ll_source ~ll_source ~cd_source ~name static_indices
    (proc : t) : Low_level.optimized =
  (match cd_source with
  | None -> ()
  | Some callback -> callback (to_doc ~name ~static_indices () proc));
  let llc : Low_level.t = to_low_level ~static_indices proc in
  (* The algebraic-rewrite tier over raw lowered code (gh-ocannl-483): ahead of the analyses, so the
     traced store and the placements see the rewritten routine. *)
  let llc = Rewrites.apply llc in
  Low_level.optimize optim_ctx ~unoptim_ll_source ~ll_source ~name static_indices llc
