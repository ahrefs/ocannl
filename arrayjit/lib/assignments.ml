open Base
(** The code for operating on n-dimensional arrays. *)

module Lazy = Utils.Lazy
module Tn = Tnode
module Nd = Ndarray

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

type init_data =
  | Reshape of Ndarray.t
  | Keep_shape_no_padding of Ndarray.t
  | Padded of { data : Nd.t; padding : Ops.axis_padding array; padded_value : float }
[@@deriving sexp_of, equal]

type buffer = Node of Tn.t | Merge_buffer of Tn.t [@@deriving sexp_of, equal]

(** Resets a array by performing the specified computation or data fetching. *)
type fetch_op =
  | Constant of float
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
[@@deriving sexp_of, equal]

and t =
  | Noop
  | Seq of t * t
  | Block_comment of string * t  (** Same as the given code, with a comment. *)
  | Accum_ternop of {
      initialize_neutral : bool;
      accum : Ops.binop;
      op : Ops.ternop;
      lhs : Tn.t;
      rhs1 : buffer;
      rhs2 : buffer;
      rhs3 : buffer;
      projections : Indexing.projections Lazy.t;
    }
  | Accum_binop of {
      initialize_neutral : bool;
      accum : Ops.binop;
      op : Ops.binop;
      lhs : Tn.t;
      rhs1 : buffer;
      rhs2 : buffer;
      projections : Indexing.projections Lazy.t;
    }
  | Accum_unop of {
      initialize_neutral : bool;
      accum : Ops.binop;
      op : Ops.unop;
      lhs : Tn.t;
      rhs : buffer;
      projections : Indexing.projections Lazy.t;
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
  if String.is_empty result then invalid_arg "Assignments.get_name: no comments in code" else result

let is_total ~initialize_neutral ~projections =
  initialize_neutral && Indexing.is_bijective projections

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
    | Accum_unop { lhs; rhs; _ } -> Set.union (one lhs) (of_node rhs)
    | Accum_binop { lhs; rhs1; rhs2; _ } ->
        Set.union_list (module Tn) [ one lhs; of_node rhs1; of_node rhs2 ]
    | Accum_ternop { lhs; rhs1; rhs2; rhs3; _ } ->
        Set.union_list (module Tn) [ one lhs; of_node rhs1; of_node rhs2; of_node rhs3 ]
    | Fetch { array; _ } -> one array
  in
  loop asgns

(** Returns the nodes that are not read from after being written to. *)
let%debug3_sexp guess_output_nodes (asgns : t) : Tn.t_set =
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
    | Accum_unop { lhs; rhs; _ } -> (of_node rhs, one lhs)
    | Accum_binop { lhs; rhs1; rhs2; _ } -> (of_node rhs1 + of_node rhs2, one lhs)
    | Accum_ternop { lhs; rhs1; rhs2; rhs3; _ } ->
        (of_node rhs1 + of_node rhs2 + of_node rhs3, one lhs)
    | Fetch { array; _ } -> (empty, one array)
  in
  snd @@ loop asgns

let sequential l =
  Option.value ~default:Noop @@ List.reduce l ~f:(fun sts another_st -> Seq (sts, another_st))

let sequence l =
  Option.value ~default:{ asgns = Noop; embedded_nodes = Set.empty (module Tn) }
  @@ List.reduce l
       ~f:(fun
           { asgns = sts; embedded_nodes = embs } { asgns = another_st; embedded_nodes = emb } ->
         { asgns = Seq (sts, another_st); embedded_nodes = Set.union embs emb })

let%diagn2_sexp to_low_level code =
  let open Indexing in
  let get buffer idcs =
    let tn = match buffer with Node tn -> tn | Merge_buffer tn -> tn in
    if not (Array.length idcs = Array.length (Lazy.force tn.Tn.dims)) then
      [%log
        "get",
        "a=",
        (tn : Tn.t),
        ":",
        Tn.label tn,
        (idcs : Indexing.axis_index array),
        (Lazy.force tn.dims : int array)];
    assert (Array.length idcs = Array.length (Lazy.force tn.Tn.dims));
    match buffer with
    | Node tn -> Low_level.Get (tn, idcs)
    | Merge_buffer tn ->
        (* FIXME: NOT IMPLEMENTED YET - need to handle merge buffer access differently now *)
        Low_level.Get (tn, idcs)
  in
  let set tn idcs llv =
    if not (Array.length idcs = Array.length (Lazy.force tn.Tn.dims)) then
      [%log
        "set",
        "a=",
        (tn : Tn.t),
        ":",
        Tn.label tn,
        (idcs : Indexing.axis_index array),
        (Lazy.force tn.dims : int array)];
    assert (Array.length idcs = Array.length (Lazy.force tn.Tn.dims));
    Low_level.Set { tn; idcs; llv; debug = "" }
  in
  let rec loop_accum ~initialize_neutral ~accum ~op ~lhs ~rhses projections =
    let projections = Lazy.force projections in
    let basecase rev_iters =
      (* Create a substitution from product iterators to loop iterators *)
      let subst_map =
        let loop_iters = Array.of_list_rev rev_iters in
        Array.mapi projections.product_iterators ~f:(fun i prod_iter ->
            (prod_iter, Indexing.Iterator loop_iters.(i)))
        |> Array.to_list
        |> Map.of_alist_exn (module Indexing.Symbol)
      in
      (* Substitute in projections *)
      let subst_index = function
        | Indexing.Fixed_idx _ as idx -> idx
        | Indexing.Iterator s as idx -> Option.value ~default:idx (Map.find subst_map s)
        | Indexing.Affine { symbols; offset } ->
            (* For affine indices, we don't substitute - they should already use the right
               symbols *)
            Indexing.Affine { symbols; offset }
      in
      let lhs_idcs = Array.map projections.project_lhs ~f:subst_index in
      let rhses_idcs = Array.map projections.project_rhs ~f:(Array.map ~f:subst_index) in
      let open Low_level in
      let lhs_ll = get (Node lhs) lhs_idcs in
      let rhses_ll = Array.mapi rhses_idcs ~f:(fun i rhs_idcs -> get rhses.(i) rhs_idcs) in
      let rhs2 = apply_op op rhses_ll in
      if is_total ~initialize_neutral ~projections then set lhs lhs_idcs rhs2
      else set lhs lhs_idcs @@ apply_op (Ops.Binop accum) [| lhs_ll; rhs2 |]
    in
    let rec for_loop rev_iters = function
      | [] -> basecase rev_iters
      | d :: product ->
          let index = Indexing.get_symbol () in
          For_loop
            {
              index;
              from_ = 0;
              to_ = d - 1;
              body = for_loop (index :: rev_iters) product;
              trace_it = true;
            }
    in
    let for_loops =
      try for_loop [] (Array.to_list projections.product_space)
      with e ->
        [%log "projections=", (projections : projections)];
        raise e
    in
    if initialize_neutral && not (is_total ~initialize_neutral ~projections) then
      let dims = lazy projections.lhs_dims in
      let fetch_op = Constant (Ops.neutral_elem accum) in
      Low_level.Seq (loop (Fetch { array = lhs; fetch_op; dims }), for_loops)
    else for_loops
  and loop code =
    match code with
    | Accum_ternop { initialize_neutral; accum; op; lhs; rhs1; rhs2; rhs3; projections } ->
        loop_accum ~initialize_neutral ~accum ~op:(Ops.Ternop op) ~lhs ~rhses:[| rhs1; rhs2; rhs3 |]
          projections
    | Accum_binop { initialize_neutral; accum; op; lhs; rhs1; rhs2; projections } ->
        loop_accum ~initialize_neutral ~accum ~op:(Ops.Binop op) ~lhs ~rhses:[| rhs1; rhs2 |]
          projections
    | Accum_unop { initialize_neutral; accum; op; lhs; rhs; projections } ->
        loop_accum ~initialize_neutral ~accum ~op:(Ops.Unop op) ~lhs ~rhses:[| rhs |] projections
    | Noop -> Low_level.Noop
    | Block_comment (s, c) -> Low_level.unflat_lines [ Comment s; loop c; Comment "end" ]
    | Seq (c1, c2) ->
        let c1 = loop c1 in
        let c2 = loop c2 in
        Seq (c1, c2)
    | Fetch { array; fetch_op = Constant 0.0; dims = _ } -> Zero_out array
    | Fetch { array; fetch_op = Constant c; dims } ->
        Low_level.loop_over_dims (Lazy.force dims) ~body:(fun idcs -> set array idcs @@ Constant c)
    | Fetch { array; fetch_op = Slice { batch_idx = { static_symbol = idx; _ }; sliced }; dims } ->
        (* TODO: doublecheck this always gets optimized away. *)
        Low_level.loop_over_dims (Lazy.force dims) ~body:(fun idcs ->
            set array idcs @@ get (Node sliced) @@ Array.append [| Iterator idx |] idcs)
    | Fetch { array; fetch_op = Embed_symbol s; dims } ->
        Low_level.loop_over_dims (Lazy.force dims) ~body:(fun idcs ->
            set array idcs @@ Embed_index (Iterator s.static_symbol))
    | Fetch { array; fetch_op = Range_over_offsets; dims = (lazy dims) } ->
        Low_level.loop_over_dims dims ~body:(fun idcs ->
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
        Low_level.unroll_dims dims ~body:(fun idcs ~offset ->
            set array idcs @@ Constant values.(offset % size))
  in
  loop code

let flatten c =
  let rec loop = function
    | Noop -> []
    | Seq (c1, c2) -> loop c1 @ loop c2
    | Block_comment (s, c) -> Block_comment (s, Noop) :: loop c
    | (Accum_ternop _ | Accum_binop _ | Accum_unop _ | Fetch _) as c -> [ c ]
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
    | Accum_ternop
        { initialize_neutral = _; accum = _; op = _; lhs; rhs1; rhs2; rhs3; projections = _ } ->
        List.iter ~f:visit [ lhs; tn rhs1; tn rhs2; tn rhs3 ]
    | Accum_binop { initialize_neutral = _; accum = _; op = _; lhs; rhs1; rhs2; projections = _ } ->
        List.iter ~f:visit [ lhs; tn rhs1; tn rhs2 ]
    | Accum_unop { initialize_neutral = _; accum = _; op = _; lhs; rhs; projections = _ } ->
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
    | Constant_fill values ->
        let values_str =
          String.concat ~sep:", " (Array.to_list (Array.map values ~f:Float.to_string))
        in
        string ("constant_fill([" ^ values_str ^ "])")
    | Range_over_offsets -> string "range_over_offsets"
    | Slice { batch_idx; sliced } ->
        string (ident sliced ^ " @| " ^ Indexing.symbol_ident batch_idx.static_symbol)
    | Embed_symbol { static_symbol; static_range = _ } ->
        string ("!@" ^ Indexing.symbol_ident static_symbol)
  in

  let rec doc_of_code = function
    | Noop -> empty
    | Seq (c1, c2) -> doc_of_code c1 ^^ doc_of_code c2
    | Block_comment (s, Noop) -> string ("# \"" ^ s ^ "\";") ^^ break 1
    | Block_comment (s, c) -> string ("# \"" ^ s ^ "\";") ^^ break 1 ^^ doc_of_code c
    | Accum_ternop { initialize_neutral; accum; op; lhs; rhs1; rhs2; rhs3; projections } ->
        let proj_spec =
          if Lazy.is_val projections then (Lazy.force projections).debug_info.spec
          else "<not-in-yet>"
        in
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
    | Accum_binop { initialize_neutral; accum; op; lhs; rhs1; rhs2; projections } ->
        let proj_spec =
          if Lazy.is_val projections then (Lazy.force projections).debug_info.spec
          else "<not-in-yet>"
        in
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
    | Accum_unop { initialize_neutral; accum; op; lhs; rhs; projections } ->
        let proj_spec =
          if Lazy.is_val projections then (Lazy.force projections).debug_info.spec
          else "<not-in-yet>"
        in
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
    | Fetch { array; fetch_op; dims = _ } ->
        string (ident array) ^^ string " := " ^^ doc_of_fetch_op fetch_op ^^ string ";" ^^ break 1
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

let%track6_sexp lower optim_ctx ~unoptim_ll_source ~ll_source ~cd_source ~name static_indices
    (proc : t) : Low_level.optimized =
  let llc : Low_level.t = to_low_level proc in
  (* Generate the low-level code before outputting the assignments, to force projections. *)
  (match cd_source with
  | None -> ()
  | Some callback -> callback (to_doc ~name ~static_indices () proc));
  Low_level.optimize optim_ctx ~unoptim_ll_source ~ll_source ~name static_indices llc
