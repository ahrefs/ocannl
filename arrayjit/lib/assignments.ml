open Base
(** The code for operating on n-dimensional arrays. *)

module Lazy = Utils.Lazy
module Tn = Tnode
module Debug_runtime = Utils.Debug_runtime

[%%global_debug_log_level Nothing]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

type buffer = Node of Tn.t | Merge_buffer of Tn.t [@@deriving sexp_of]

(** Resets a array by performing the specified computation or data fetching. *)
type fetch_op =
  | Constant of float
  | Imported of Ops.global_identifier
  | Slice of { batch_idx : Indexing.static_symbol; sliced : Tn.t }
  | Embed_symbol of Indexing.static_symbol
[@@deriving sexp_of]

and t =
  | Noop
  | Seq of t * t
  | Block_comment of string * t  (** Same as the given code, with a comment. *)
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

let is_noop = function Noop -> true | _ -> false

let get_name_exn asgns =
  let punct_or_sp = Str.regexp "[-@*/:.;, ]" in
  let punct_and_sp = Str.regexp {|[-@*/:.;,]\( |$\)|} in
  let rec loop = function
    | Block_comment (s, _) -> Str.global_replace punct_and_sp "" s |> Str.global_replace punct_or_sp "_"
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

let recurrent_nodes asgns =
  let open Utils.Set_O in
  let empty = Set.empty (module Tn) in
  let single = function Node tn -> Set.singleton (module Tn) tn | Merge_buffer _ -> Set.empty (module Tn) in
  let maybe have lhs = if have then Set.singleton (module Tn) lhs else empty in
  let rec loop = function
    | Noop -> empty
    | Seq (t1, t2) -> loop t1 + (loop t2 - assigned t1)
    | Block_comment (_, t) -> loop t
    | Accum_binop { initialize_neutral; lhs; rhs1; rhs2; _ } ->
        maybe (not initialize_neutral) lhs + single rhs1 + single rhs2
    | Accum_unop { initialize_neutral; lhs; rhs; _ } -> maybe (not initialize_neutral) lhs + single rhs
    | Fetch _ -> empty
  and assigned = function
    | Noop -> Set.empty (module Tn)
    | Seq (t1, t2) -> assigned t1 + assigned t2
    | Block_comment (_, t) -> assigned t
    | Accum_binop { initialize_neutral; lhs; _ } -> maybe initialize_neutral lhs
    | Accum_unop { initialize_neutral; lhs; _ } -> maybe initialize_neutral lhs
    | Fetch { array; _ } -> Set.singleton (module Tn) array
  in
  loop asgns

let sequential l = Option.value ~default:Noop @@ List.reduce l ~f:(fun st sts -> Seq (st, sts))

let%debug_sexp to_low_level code =
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
    | Merge_buffer tn -> Low_level.Get_global (Ops.Merge_buffer { source_node_id = tn.Tn.id }, Some idcs)
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
  let rec loop code =
    match code with
    | Accum_binop { initialize_neutral; accum; op; lhs; rhs1; rhs2; projections } ->
        let projections = Lazy.force projections in
        let lhs_idx =
          derive_index ~product_syms:projections.product_iterators ~projection:projections.project_lhs
        in
        let rhs1_idx =
          derive_index ~product_syms:projections.product_iterators ~projection:projections.project_rhs.(0)
        in
        let rhs2_idx =
          derive_index ~product_syms:projections.product_iterators ~projection:projections.project_rhs.(1)
        in
        let is_assignment = initialize_neutral && Indexing.is_bijective projections in
        let basecase rev_iters =
          let product = Array.of_list_rev_map rev_iters ~f:(fun s -> Indexing.Iterator s) in
          let rhs1_idcs = rhs1_idx ~product in
          let rhs2_idcs = rhs2_idx ~product in
          let lhs_idcs = lhs_idx ~product in
          let open Low_level in
          let lhs_ll = get (Node lhs) lhs_idcs in
          let rhs1_ll = get rhs1 rhs1_idcs in
          let rhs2_ll = get rhs2 rhs2_idcs in
          let rhs2 = binop ~op ~rhs1:rhs1_ll ~rhs2:rhs2_ll in
          if is_assignment then set lhs lhs_idcs rhs2
          else set lhs lhs_idcs @@ binop ~op:accum ~rhs1:lhs_ll ~rhs2
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
        if initialize_neutral && not is_assignment then
          let dims = lazy projections.lhs_dims in
          let fetch_op = Constant (Ops.neutral_elem accum) in
          Low_level.Seq (loop (Fetch { array = lhs; fetch_op; dims }), for_loops)
        else for_loops
    | Accum_unop { initialize_neutral; accum; op; lhs; rhs; projections } ->
        let projections = Lazy.force projections in
        let lhs_idx =
          derive_index ~product_syms:projections.product_iterators ~projection:projections.project_lhs
        in
        let rhs_idx =
          derive_index ~product_syms:projections.product_iterators ~projection:projections.project_rhs.(0)
        in
        let is_assignment = initialize_neutral && Indexing.is_bijective projections in
        let basecase rev_iters =
          let product = Array.of_list_rev_map rev_iters ~f:(fun s -> Indexing.Iterator s) in
          let lhs_idcs = lhs_idx ~product in
          let open Low_level in
          let lhs_ll = get (Node lhs) lhs_idcs in
          let rhs_ll = get rhs @@ rhs_idx ~product in
          let rhs2 = unop ~op ~rhs:rhs_ll in
          if is_assignment then set lhs lhs_idcs rhs2
          else set lhs lhs_idcs @@ binop ~op:accum ~rhs1:lhs_ll ~rhs2
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
        let for_loops = for_loop [] (Array.to_list projections.product_space) in
        if initialize_neutral && not is_assignment then
          let dims = lazy projections.lhs_dims in
          let fetch_op = Constant (Ops.neutral_elem accum) in
          Low_level.Seq (loop (Fetch { array = lhs; fetch_op; dims }), for_loops)
        else for_loops
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
    | Fetch { array; fetch_op = Imported global; dims } ->
        Low_level.loop_over_dims (Lazy.force dims) ~body:(fun idcs ->
            set array idcs @@ Get_global (global, Some idcs))
  in

  loop code

let flatten c =
  let rec loop = function
    | Noop -> []
    | Seq (c1, c2) -> loop c1 @ loop c2
    | Block_comment (s, c) -> Block_comment (s, Noop) :: loop c
    | (Accum_binop _ | Accum_unop _ | Fetch _) as c -> [ c ]
  in
  loop c

let get_ident_within_code ?no_dots c =
  let ident_style = Tn.get_style ~arg_name:"cd_ident_style" ?no_dots () in
  let nograd_idents = Hashtbl.create (module String) in
  let grad_idents = Hashtbl.create (module String) in
  let visit tn =
    let idents = if List.mem ~equal:String.equal tn.Tn.label "grad" then grad_idents else nograd_idents in
    Option.iter (Tn.ident_label tn)
      ~f:(Hashtbl.update idents ~f:(fun old -> Set.add (Option.value ~default:Utils.no_ints old) tn.id))
  in
  let tn = function Node tn -> tn | Merge_buffer tn -> tn in
  let rec loop (c : t) =
    match c with
    | Noop -> ()
    | Seq (c1, c2) ->
        loop c1;
        loop c2
    | Block_comment (_, c) -> loop c
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
  let repeating_grad_idents = Hashtbl.filter grad_idents ~f:(fun ids -> List.length (Set.to_list ids) > 1) in
  Tn.styled_ident ~repeating_nograd_idents ~repeating_grad_idents ident_style

let fprint_hum ?name ?static_indices () ppf c =
  let ident = get_ident_within_code c in
  let buffer_ident = function Node tn -> ident tn | Merge_buffer tn -> "merge " ^ ident tn in
  let open Stdlib.Format in
  let out_fetch_op ppf (op : fetch_op) =
    match op with
    | Constant f -> fprintf ppf "%g" f
    | Imported (Ops.C_function c) -> fprintf ppf "%s()" c
    | Imported (Merge_buffer { source_node_id }) ->
        let tn = Option.value_exn @@ Tn.find ~id:source_node_id in
        fprintf ppf "merge %s" (ident tn)
    | Imported (Ops.External_unsafe { ptr; prec; dims = _ }) -> fprintf ppf "%s" @@ Ops.ptr_to_string ptr prec
    | Slice { batch_idx; sliced } ->
        fprintf ppf "%s @@| %s" (ident sliced) (Indexing.symbol_ident batch_idx.static_symbol)
    | Embed_symbol { static_symbol; static_range = _ } ->
        fprintf ppf "!@@%s" @@ Indexing.symbol_ident static_symbol
  in
  let rec loop = function
    | Noop -> ()
    | Seq (c1, c2) ->
        loop c1;
        loop c2
    | Block_comment (s, Noop) -> fprintf ppf "# \"%s\";@ " s
    | Block_comment (s, c) ->
        fprintf ppf "# \"%s\";@ " s;
        loop c
    | Accum_binop { initialize_neutral; accum; op; lhs; rhs1; rhs2; projections } ->
        let proj_spec =
          if Lazy.is_val projections then (Lazy.force projections).debug_info.spec else "<not-in-yet>"
        in
        fprintf ppf "%s %s %s %s %s%s;@ " (ident lhs)
          (Ops.assign_op_cd_syntax ~initialize_neutral accum)
          (buffer_ident rhs1) (Ops.binop_cd_syntax op) (buffer_ident rhs2)
          (if (not (String.equal proj_spec ".")) || List.mem ~equal:Ops.equal_binop Ops.[ Mul; Div ] op then
             " ~logic:\"" ^ proj_spec ^ "\""
           else "")
    | Accum_unop { initialize_neutral; accum; op; lhs; rhs; projections } ->
        let proj_spec =
          if Lazy.is_val projections then (Lazy.force projections).debug_info.spec else "<not-in-yet>"
        in
        fprintf ppf "%s %s %s%s%s;@ " (ident lhs)
          (Ops.assign_op_cd_syntax ~initialize_neutral accum)
          (if not @@ Ops.equal_unop op Ops.Identity then Ops.unop_cd_syntax op ^ " " else "")
          (buffer_ident rhs)
          (if not (String.equal proj_spec ".") then " ~logic:\"" ^ proj_spec ^ "\"" else "")
    | Fetch { array; fetch_op; dims = _ } -> fprintf ppf "%s := %a;@ " (ident array) out_fetch_op fetch_op
  in
  fprintf ppf "@,@[<v 2>";
  Low_level.fprint_function_header ?name ?static_indices () ppf;
  loop c;
  fprintf ppf "@]"

let%debug_sexp lower_proc ~unoptim_ll_source ~ll_source ~cd_source ~name static_indices (proc : t) :
    Low_level.optimized =
  let llc = to_low_level proc in
  (* Generate the low-level code before outputting the assignments, to force projections. *)
  (match cd_source with
  | None -> ()
  | Some ppf ->
      fprint_hum ~name ~static_indices () ppf proc;
      Stdlib.Format.pp_print_flush ppf ());
  Low_level.optimize_proc ~unoptim_ll_source ~ll_source ~name static_indices llc
