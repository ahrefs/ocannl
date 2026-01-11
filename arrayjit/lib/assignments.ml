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

let is_total ~initialize_neutral ~projections =
  initialize_neutral && Indexing.is_surjective projections

let can_skip_accumulation ~projections =
  (* We can skip accumulation (use = instead of +=) only if the projection is injective *)
  Indexing.is_injective projections

(** Returns materialized nodes in the sense of {!Tnode.is_in_context_force}. NOTE: it must be called
    after compilation; otherwise, it will disrupt memory mode inference. *)
let%debug3_sexp context_nodes ~(use_host_memory : 'a option) (asgns : t) : Tn.t_set =
  let open Utils.Set_O in
  let empty = Set.empty (module Tn) in
  let one tn =
    if Tn.is_in_context_force ~use_host_memory tn 34 then Set.singleton (module Tn) tn else empty
  in
  let of_node = function Node rhs -> one rhs | Merge_buffer _ -> empty in
  let rec loop = function
    | Noop -> empty
    | Seq (t1, t2) -> loop t1 + loop t2
    | Block_comment (_, t) -> loop t
    | Accum_op { lhs; rhs; _ } ->
        let rhses =
          match rhs with
          | Unop { rhs; _ } -> [ of_node rhs ]
          | Binop { rhs1; rhs2; _ } -> [ of_node rhs1; of_node rhs2 ]
          | Ternop { rhs1; rhs2; rhs3; _ } -> [ of_node rhs1; of_node rhs2; of_node rhs3 ]
          | Block { rhses; _ } -> Array.to_list rhses |> List.map ~f:of_node
          | Rev_sides { lhses; _ } -> Array.to_list lhses |> List.map ~f:of_node
        in
        Set.union_list (module Tn) (one lhs :: rhses)
    | Set_vec_unop { lhs; rhs; _ } -> Set.union (one lhs) (of_node rhs)
    | Fetch { array; _ } -> one array
  in
  loop asgns

(** In the second set, returns the nodes that are not read from after being written to. In the first
    set, returns the nodes that are ever read from. *)
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
        let inputs, outputs =
          match rhs with
          | Unop { rhs; _ } -> (of_node rhs, one lhs)
          | Binop { rhs1; rhs2; _ } -> (of_node rhs1 + of_node rhs2, one lhs)
          | Ternop { rhs1; rhs2; rhs3; _ } -> (of_node rhs1 + of_node rhs2 + of_node rhs3, one lhs)
          | Block { rhses; _ } ->
              (Array.fold rhses ~init:empty ~f:(fun acc buf -> acc + of_node buf), one lhs)
          | Rev_sides { lhses; _ } ->
              (one lhs, Array.fold lhses ~init:empty ~f:(fun acc buf -> acc + of_node buf))
        in
        (inputs, outputs)
    | Set_vec_unop { lhs; rhs; _ } -> (of_node rhs, one lhs)
    | Fetch { array; _ } -> (empty, one array)
  in
  loop asgns

let sequential l =
  Option.value ~default:Noop @@ List.reduce l ~f:(fun sts another_st -> Seq (sts, another_st))

let sequence l =
  Option.value ~default:{ asgns = Noop; embedded_nodes = Set.empty (module Tn) }
  @@ List.reduce l
       ~f:(fun
           { asgns = sts; embedded_nodes = embs } { asgns = another_st; embedded_nodes = emb } ->
         { asgns = Seq (sts, another_st); embedded_nodes = Set.union embs emb })

let collect_neutral_elem (asgns : t) : float option =
  let rec loop acc = function
    | Noop -> acc
    | Seq (t1, t2) -> loop (loop acc t1) t2
    | Block_comment (_, t) -> loop acc t
    | Accum_op { accum; _ } -> (
        let neutral = Ops.neutral_elem accum in
        match acc with
        | None -> Some (Some neutral)
        | Some (Some v) when Float.( = ) v neutral -> acc
        | Some (Some _) -> Some None
        | Some None -> acc)
    | Set_vec_unop _ | Fetch _ -> acc
  in
  match loop None asgns with None -> None | Some v -> v

let%track4_sexp to_low_level code =
  let open Indexing in
  (* Apply left padding offsets to convert from semantic to buffer indices. Semantic indices can be
     negative (e.g., -1 for convolution padding), but buffer indices must be non-negative. Adding
     left_padding converts semantic to buffer space. *)
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
                | Iterator s -> Affine { symbols = [ (1, s) ]; offset = left_pad }
                | Affine { symbols; offset } -> Affine { symbols; offset = offset + left_pad }
                | Sub_axis -> Sub_axis
                | Concat _ -> assert false)
  in
  let is_padded tn = Option.is_some (Tn.get_padding tn) in
  let get (buffer : buffer) (idcs : Indexing.axis_index array) : Low_level.scalar_t =
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
    (* The same projection can be used to access a padded or a non-padded tensor. *)
    let idcs = if is_padded tn then apply_padding_offset tn idcs else idcs in
    match buffer with
    | Node tn -> Low_level.Get (tn, idcs)
    | Merge_buffer tn -> Low_level.Get_merge_buffer (tn, idcs)
  in
  let set (tn : Tn.t) (idcs : Indexing.axis_index array) (llsc : Low_level.scalar_t) : Low_level.t =
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
      match Tn.get_padding array with Some (_, Some v) -> reset_padding_regions array v | _ -> []
    in
    Low_level.unflat_lines @@ padding_loops @ [ llc ]
  in
  let rec loop_accum ~initialize_neutral ~accum ~(op : Ops.op) ~lhs ~rhses projections : Low_level.t
      =
    let projections : Indexing.projections = Lazy.force projections in
    let all_prod_iters =
      Array.to_list projections.product_iterators
      |> List.concat
      |> Set.of_list (module Indexing.Symbol)
    in
    let iter_sizes =
      Array.fold2_exn projections.product_space projections.product_iterators
        ~init:(Map.empty (module Indexing.Symbol))
        ~f:(fun acc ds its ->
          List.fold2_exn ds its ~init:acc ~f:(fun acc d iter ->
              Map.set acc ~key:iter ~data:d))
    in
    let concat_offset_for syms active =
      let _, offset =
        List.fold syms ~init:(0, None) ~f:(fun (cumul, found) s ->
            let size = Map.find iter_sizes s |> Option.value ~default:0 in
            if Indexing.equal_symbol s active then (cumul + size, Some cumul)
            else (cumul + size, found))
      in
      Option.value ~default:0 offset
    in
    let basecase block_iters rev_iters =
      (* Create a substitution from product iterators to loop iterators. Fresh loop symbols are
         needed because product_iterators may be shared across different operations/tensors, but
         each lowered operation needs private loop symbols to avoid conflicts in low_level.ml's
         symbol-to-tensor tracking.
         Concat offsets are computed per Concat index using symbol order. *)
      let exception Empty_block in
      let block_iters = Array.of_list_rev block_iters in
      let subst_map =
        let loop_iters = Array.of_list_rev rev_iters in
        Array.map2_exn block_iters loop_iters ~f:(fun block_iter loop_iter ->
            (block_iter, Indexing.Iterator loop_iter))
        |> Array.to_list
        |> Map.of_alist_exn (module Indexing.Symbol)
      in
      (* Substitute in projections - including inside Affine indices *)
      let subst_index = function
        | (Indexing.Fixed_idx _ | Indexing.Sub_axis) as idx -> idx
        | Iterator s
          when Set.mem all_prod_iters s
               && not (Array.mem ~equal:Indexing.equal_symbol block_iters s) ->
            raise Empty_block
        | Indexing.Iterator s as idx -> Option.value ~default:idx (Map.find subst_map s)
        | Indexing.Affine { symbols; offset } ->
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
            Indexing.Affine { symbols; offset }
        | Indexing.Concat syms ->
            (* For Block lowering: find the active component (in block_iters) and resolve to it
               with the appropriate offset based on Concat symbol order. *)
            let active =
              List.find_mapi syms ~f:(fun _i s ->
                  if Array.mem ~equal:Indexing.equal_symbol block_iters s then
                    match Map.find subst_map s with
                    | Some (Indexing.Iterator s') ->
                        let offset = concat_offset_for syms s in
                        Some (s', offset)
                    | _ -> None
                  else None)
            in
            (match active with
            | Some (s', 0) -> Indexing.Iterator s'
            | Some (s', offset) -> Indexing.Affine { symbols = [ (1, s') ]; offset }
            | None ->
                (* No active component - this shouldn't happen in Block lowering *)
                let syms' =
                  List.map syms ~f:(fun s ->
                      match Map.find subst_map s with
                      | Some (Indexing.Iterator s') -> s'
                      | _ -> s)
                in
                Indexing.Concat syms')
      in
      try
        let lhs_idcs : Indexing.axis_index array =
          Array.map projections.project_lhs ~f:subst_index
        in
        let open Low_level in
        let lhs_ll = get (Node lhs) lhs_idcs in
        let rhses_ll =
          Array.filter_mapi projections.project_rhs ~f:(fun i rhs_idcs ->
              try
                let rhs_idcs = Array.map ~f:subst_index rhs_idcs in
                Some (get rhses.(i) rhs_idcs)
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
        if initialize_neutral && can_skip_accumulation ~projections then set lhs lhs_idcs rhs2
        else set lhs lhs_idcs @@ apply_op (Ops.Binop accum) [| lhs_ll; rhs2 |]
      with Empty_block -> Low_level.Noop
    in
    let rec for_loop block_iters rev_iters = function
      | [] -> basecase block_iters rev_iters
      | (ds, its) :: product ->
          let index = Indexing.get_symbol () in
          Low_level.unflat_lines
          @@ List.map2_exn ds its ~f:(fun d iter ->
              Low_level.For_loop
                {
                  index;
                  from_ = 0;
                  to_ = d - 1;
                  body = for_loop (iter :: block_iters) (index :: rev_iters) product;
                  trace_it = true;
                })
    in
    let for_loops =
      for_loop [] []
        (Array.to_list @@ Array.zip_exn projections.product_space projections.product_iterators)
    in
    (* Need initialization if: initialize_neutral is true AND (not surjective OR not injective)

       Not surjective: some positions never written (need init to avoid garbage)

       Not injective: accumulation needed (need init for first += operation) *)
    let needs_init =
      initialize_neutral
      && not (Indexing.is_surjective projections && Indexing.is_injective projections)
    in
    (* Check if any RHS tensor has padding with None neutral value (needs reset before this op).
       Generate loops that set padding margins to the correct neutral value for this operation. *)
    let neutral_value = Ops.neutral_elem accum in
    let padding_resets =
      Array.filter_map rhses ~f:(fun buf ->
          let tn = match buf with Node tn | Merge_buffer tn -> tn in
          match Lazy.force tn.padding with
          | Some (_, None) ->
              (* Padding exists but neutral value is None - needs reset for this operation. Generate
                 loops to set padding margins to the neutral value. *)
              Some (reset_padding_regions tn neutral_value)
          | Some (_, Some v) when Float.( <> ) v neutral_value ->
              (* Padding exists with different neutral value - also needs reset *)
              Some (reset_padding_regions tn neutral_value)
          | _ -> None)
      |> Array.to_list |> List.concat
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
  and loop_accum_rev ~initialize_neutral ~accum ~(op : Ops.op) ~lhs ~lhses projections :
      Low_level.t =
    let projections : Indexing.projections = Lazy.force projections in
    let all_prod_iters =
      Array.to_list projections.product_iterators
      |> List.concat
      |> Set.of_list (module Indexing.Symbol)
    in
    let target_projections =
      Array.mapi projections.project_rhs ~f:(fun i project_lhs ->
          { projections with lhs_dims = projections.rhs_dims.(i); project_lhs })
    in
    let target_can_skip =
      Array.map target_projections ~f:(fun proj -> can_skip_accumulation ~projections:proj)
    in
    let target_needs_init =
      Array.map target_projections ~f:(fun proj ->
          initialize_neutral
          && not (Indexing.is_surjective proj && Indexing.is_injective proj))
    in
    let iter_sizes =
      Array.fold2_exn projections.product_space projections.product_iterators
        ~init:(Map.empty (module Indexing.Symbol))
        ~f:(fun acc ds its ->
          List.fold2_exn ds its ~init:acc ~f:(fun acc d iter ->
              Map.set acc ~key:iter ~data:d))
    in
    let concat_offset_for syms active =
      let _, offset =
        List.fold syms ~init:(0, None) ~f:(fun (cumul, found) s ->
            let size = Map.find iter_sizes s |> Option.value ~default:0 in
            if Indexing.equal_symbol s active then (cumul + size, Some cumul)
            else (cumul + size, found))
      in
      Option.value ~default:0 offset
    in
    let basecase block_iters rev_iters =
      let exception Empty_block in
      let block_iters = Array.of_list_rev block_iters in
      let subst_map =
        let loop_iters = Array.of_list_rev rev_iters in
        Array.map2_exn block_iters loop_iters ~f:(fun block_iter loop_iter ->
            (block_iter, Indexing.Iterator loop_iter))
        |> Array.to_list
        |> Map.of_alist_exn (module Indexing.Symbol)
      in
      let subst_index = function
        | (Indexing.Fixed_idx _ | Indexing.Sub_axis) as idx -> idx
        | Iterator s
          when Set.mem all_prod_iters s
               && not (Array.mem ~equal:Indexing.equal_symbol block_iters s) ->
            raise Empty_block
        | Indexing.Iterator s as idx -> Option.value ~default:idx (Map.find subst_map s)
        | Indexing.Affine { symbols; offset } ->
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
            Indexing.Affine { symbols; offset }
        | Indexing.Concat syms ->
            (* For Rev_sides lowering: find the active component and resolve with offset *)
            let active =
              List.find_mapi syms ~f:(fun _i s ->
                  if Array.mem ~equal:Indexing.equal_symbol block_iters s then
                    match Map.find subst_map s with
                    | Some (Indexing.Iterator s') ->
                        let offset = concat_offset_for syms s in
                        Some (s', offset)
                    | _ -> None
                  else None)
            in
            (match active with
            | Some (s', 0) -> Indexing.Iterator s'
            | Some (s', offset) -> Indexing.Affine { symbols = [ (1, s') ]; offset }
            | None ->
                let syms' =
                  List.map syms ~f:(fun s ->
                      match Map.find subst_map s with
                      | Some (Indexing.Iterator s') -> s'
                      | _ -> s)
                in
                Indexing.Concat syms')
      in
      let target_tn_exn = function
        | Node tn -> tn
        | Merge_buffer _ ->
            raise @@ Utils.User_error "Rev_sides cannot write to merge buffers"
      in
      try
        let rhs_idcs : Indexing.axis_index array =
          Array.map projections.project_lhs ~f:subst_index
        in
        let open Low_level in
        let rhs_ll = get (Node lhs) rhs_idcs in
        let targets =
          Array.filter_mapi projections.project_rhs ~f:(fun i lhs_idcs ->
              try
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
        let target_tn = target_tn_exn target_buf in
        if initialize_neutral && target_can_skip.(i) then set target_tn lhs_idcs rhs2
        else set target_tn lhs_idcs @@ apply_op (Ops.Binop accum) [| get target_buf lhs_idcs; rhs2 |]
      with Empty_block -> Low_level.Noop
    in
    let rec for_loop block_iters rev_iters = function
      | [] -> basecase block_iters rev_iters
      | (ds, its) :: product ->
          let index = Indexing.get_symbol () in
          Low_level.unflat_lines
          @@ List.map2_exn ds its ~f:(fun d iter ->
              Low_level.For_loop
                {
                  index;
                  from_ = 0;
                  to_ = d - 1;
                  body = for_loop (iter :: block_iters) (index :: rev_iters) product;
                  trace_it = true;
                })
    in
    let for_loops =
      for_loop [] []
        (Array.to_list @@ Array.zip_exn projections.product_space projections.product_iterators)
    in
    let neutral_value = Ops.neutral_elem accum in
    let padding_resets =
      match Lazy.force lhs.padding with
      | Some (_, None) -> reset_padding_regions lhs neutral_value
      | Some (_, Some v) when Float.( <> ) v neutral_value ->
          reset_padding_regions lhs neutral_value
      | _ -> []
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
            Some (Fetch { array; fetch_op = Constant neutral_value; dims = lazy projections.rhs_dims.(i) }))
      |> Array.to_list
    in
    if List.is_empty init_ops then for_loops_with_resets
    else Low_level.unflat_lines (List.map init_ops ~f:loop @ [ for_loops_with_resets ])
  and loop (code : t) : Low_level.t =
    match code with
    | Accum_op { initialize_neutral; accum; lhs; rhs; projections; _ } ->
        let op, rhses =
          match rhs with
          | Unop { op; rhs } -> (Ops.Unop op, [| rhs |])
          | Binop { op; rhs1; rhs2 } -> (Ops.Binop op, [| rhs1; rhs2 |])
          | Ternop { op; rhs1; rhs2; rhs3 } -> (Ops.Ternop op, [| rhs1; rhs2; rhs3 |])
          | Block { op; rhses } -> (Ops.Unop op, rhses)
          | Rev_sides { op; lhses } -> (Ops.Unop op, lhses)
        in
        (match rhs with
        | Rev_sides _ -> loop_accum_rev ~initialize_neutral ~accum ~op ~lhs ~lhses:rhses projections
        | _ -> loop_accum ~initialize_neutral ~accum ~op ~lhs ~rhses projections)
    | Set_vec_unop { op; lhs; rhs; projections; _ } ->
        (* Handle vector unary operations *)
        let projections = Lazy.force projections in
        let basecase rev_iters =
          let subst_map =
            let loop_iters = Array.of_list_rev rev_iters in
            Array.map2_exn loop_iters projections.product_iterators ~f:(fun loop_iter prod_iter ->
                let prod_iter =
                  match prod_iter with
                  | [ prod_iter ] -> prod_iter
                  | _ -> raise @@ Utils.User_error "Concat indexing not supported in Set_vec_unop"
                in
                (prod_iter, Indexing.Iterator loop_iter))
            |> Array.to_list
            |> Map.of_alist_exn (module Indexing.Symbol)
          in
          let subst_index = function
            | Indexing.Concat _ ->
                raise @@ Utils.User_error "Concat indexing not supported in Set_vec_unop"
            | (Fixed_idx _ | Sub_axis) as idx -> idx
            | Iterator s as idx -> Option.value ~default:idx (Map.find subst_map s)
            | Affine { symbols; offset } ->
                (* Substitute symbols in affine index *)
                let subst_symbols =
                  List.map symbols ~f:(fun (coeff, s) ->
                      match Map.find subst_map s with
                      | Some (Indexing.Iterator new_s) -> (coeff, new_s)
                      | _ -> (coeff, s))
                in
                Indexing.Affine { symbols = subst_symbols; offset }
          in
          let lhs_idcs = Array.map projections.project_lhs ~f:subst_index in
          let rhs_idcs = Array.map projections.project_rhs.(0) ~f:subst_index in
          let open Low_level in
          let rhs_ll = get rhs rhs_idcs in
          let length =
            match op with
            | Ops.Uint4x32_to_prec_uniform -> (
                (* Prevent over-eager guard against forcing precision. *)
                ignore (Lazy.force lhs.dims);
                let target_prec = Lazy.force lhs.prec in
                match target_prec with
                | Ops.Byte_prec _ | Ops.Fp8_prec _ -> 16 (* 8-bit values *)
                | Ops.Uint16_prec _ | Ops.Half_prec _ | Ops.Bfloat16_prec _ -> 8 (* 16-bit values *)
                | Ops.Int32_prec _ | Ops.Uint32_prec _ | Ops.Single_prec _ -> 4 (* 32-bit values *)
                | Ops.Double_prec _ | Ops.Int64_prec _ | Ops.Uint64_prec _ -> 2 (* 64-bit values *)
                | Ops.Uint4x32_prec _ -> 1 (* 128-bit value *)
                | Ops.Void_prec -> failwith "Cannot use vector operation with void precision")
          in
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
        let rec for_loop rev_iters = function
          | [] -> basecase rev_iters
          | [ d ] :: product ->
              let index = Indexing.get_symbol () in
              Low_level.For_loop
                {
                  index;
                  from_ = 0;
                  to_ = d - 1;
                  body = for_loop (index :: rev_iters) product;
                  trace_it = true;
                }
          | _ -> raise @@ Utils.User_error "Concat indexing not supported in Set_vec_unop"
        in
        for_loop [] (Array.to_list projections.product_space)
    | Noop -> Low_level.Noop
    | Block_comment (s, c) -> Low_level.unflat_lines [ Comment s; loop c; Comment "end" ]
    | Seq (c1, c2) ->
        let c1 = loop c1 in
        let c2 = loop c2 in
        Low_level.Seq (c1, c2)
    | Fetch { array; fetch_op = Constant 0.0; dims = _ } ->
        default_padding_before array @@ Low_level.Zero_out array
    | Fetch { array; fetch_op = Constant c; dims } ->
        default_padding_before array
        @@ Low_level.loop_over_dims (Lazy.force dims) ~body:(fun idcs ->
            set array idcs @@ Constant c)
    | Fetch { array; fetch_op = Constant_bits i; dims } ->
        default_padding_before array
        @@ Low_level.loop_over_dims (Lazy.force dims) ~body:(fun idcs ->
            set array idcs @@ Constant_bits i)
    | Fetch { array; fetch_op = Slice { batch_idx = { static_symbol = idx; _ }; sliced }; dims } ->
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
        (* Note: we are guaranteed all shape inference is forced before we access variable_ref. *)
        default_padding_before array
        @@ Low_level.loop_over_dims (Lazy.force dims) ~body:(fun idcs ->
            set array idcs @@ Constant (Float.of_int @@ Option.value_exn variable_ref.solved_dim))
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
  loop code

let flatten c =
  let rec loop = function
    | Noop -> []
    | Seq (c1, c2) -> loop c1 @ loop c2
    | Block_comment (s, c) -> Block_comment (s, Noop) :: loop c
    | (Accum_op _ | Set_vec_unop _ | Fetch _) as c -> [ c ]
  in
  loop c

let is_noop c =
  List.for_all ~f:(function Noop | Block_comment (_, Noop) -> true | _ -> false) @@ flatten c

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
             Set.add (Option.value ~default:Utils.no_ints old) tn.id))
  in
  let tn = function Node tn -> tn | Merge_buffer tn -> tn in
  let rec loop (c : t) =
    match c with
    | Noop -> ()
    | Seq (c1, c2) ->
        loop c1;
        loop c2
    | Block_comment (_, c) -> loop c
    | Accum_op { lhs; rhs; _ } ->
        let rhses =
          match rhs with
          | Unop { rhs; _ } -> [ tn rhs ]
          | Binop { rhs1; rhs2; _ } -> [ tn rhs1; tn rhs2 ]
          | Ternop { rhs1; rhs2; rhs3; _ } -> [ tn rhs1; tn rhs2; tn rhs3 ]
          | Block { rhses; _ } -> Array.to_list rhses |> List.map ~f:tn
          | Rev_sides { lhses; _ } -> Array.to_list lhses |> List.map ~f:tn
        in
        List.iter ~f:visit (lhs :: rhses)
    | Set_vec_unop { op = _; lhs; rhs; projections = _; projections_debug = _ } ->
        List.iter ~f:visit [ lhs; tn rhs ]
    | Fetch { array; fetch_op = _; dims = _ } -> visit array
  in
  loop c;
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
    | Embed_symbol { static_symbol; static_range = _ } ->
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
    invalid_arg "Assignments.get_name_exn: no comments in code: " ^ to_string asgns
  else result

let%track6_sexp lower optim_ctx ~unoptim_ll_source ~ll_source ~cd_source ~name static_indices
    (proc : t) : Low_level.optimized =
  (match cd_source with
  | None -> ()
  | Some callback -> callback (to_doc ~name ~static_indices () proc));
  let llc : Low_level.t = to_low_level proc in
  Low_level.optimize optim_ctx ~unoptim_ll_source ~ll_source ~name static_indices llc
