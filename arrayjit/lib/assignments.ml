open Base
(** The code for operating on n-dimensional arrays. *)

module Lazy = Utils.Lazy
module Tn = Tnode
module Debug_runtime = Utils.Debug_runtime

[%%global_debug_log_level Nothing]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

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
      rhs1 : Tn.t;
      rhs2 : Tn.t;
      projections : Indexing.projections Lazy.t;
    }
  | Accum_unop of {
      initialize_neutral : bool;
      accum : Ops.binop;
      op : Ops.unop;
      lhs : Tn.t;
      rhs : Tn.t;
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
  let single = Set.singleton (module Tn) in
  let rec loop = function
    | Noop -> empty
    | Seq (t1, t2) -> loop t1 + (loop t2 - assigned t1)
    | Block_comment (_, t) -> loop t
    | Accum_binop { initialize_neutral; lhs; rhs1; rhs2; _ } ->
        (if initialize_neutral then empty else single lhs) + single rhs1 + single rhs2
    | Accum_unop { initialize_neutral; lhs; rhs; _ } ->
        (if initialize_neutral then empty else single lhs) + single rhs
    | Fetch _ -> empty
  and assigned = function
    | Noop -> Set.empty (module Tn)
    | Seq (t1, t2) -> assigned t1 + assigned t2
    | Block_comment (_, t) -> assigned t
    | Accum_binop { initialize_neutral; lhs; _ } -> if initialize_neutral then single lhs else empty
    | Accum_unop { initialize_neutral; lhs; _ } -> if initialize_neutral then single lhs else empty
    | Fetch { array; _ } -> single array
  in
  loop asgns

let remove_updates array c =
  let rec rm check = function
    | ( Seq ((Accum_binop { lhs; _ } | Accum_unop { lhs; _ }), t)
      | Seq (t, (Accum_binop { lhs; _ } | Accum_unop { lhs; _ })) ) as c
      when check ->
        if Tn.equal array lhs then rm true t else rm false c
    | Seq (t1, t2) -> Seq (rm true t1, rm true t2)
    | (Accum_binop { lhs; _ } | Accum_unop { lhs; _ }) when Tn.equal array lhs -> Noop
    | c -> c
  in
  rm true c

let sequential l = Option.value ~default:Noop @@ List.reduce l ~f:(fun st sts -> Seq (st, sts))

let%debug_sexp to_low_level code =
  let open Indexing in
  let get a idcs =
    if not (Array.length idcs = Array.length (Lazy.force a.Tn.dims)) then
      [%log
        "get",
          "a=",
          (a : Tn.t),
          ":",
          Tn.label a,
          (idcs : Indexing.axis_index array),
          (Lazy.force a.dims : int array)];
    assert (Array.length idcs = Array.length (Lazy.force a.Tn.dims));
    Low_level.Get (a, idcs)
  in
  let set array idcs llv =
    if not (Array.length idcs = Array.length (Lazy.force array.Tn.dims)) then
      [%log
        "set",
          "a=",
          (array : Tn.t),
          ":",
          Tn.label array,
          (idcs : Indexing.axis_index array),
          (Lazy.force array.dims : int array)];
    assert (Array.length idcs = Array.length (Lazy.force array.Tn.dims));
    Low_level.Set { array; idcs; llv; debug = "" }
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
          let lhs_ll = get lhs lhs_idcs in
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
          let lhs_ll = get lhs lhs_idcs in
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
            set array idcs @@ get sliced @@ Array.append [| Iterator idx |] idcs)
    | Fetch { array; fetch_op = Embed_symbol s; dims } ->
        Low_level.loop_over_dims (Lazy.force dims) ~body:(fun idcs ->
            set array idcs @@ Embed_index (Iterator s.static_symbol))
    | Fetch { array = _; fetch_op = Imported _; dims = _ } ->
        failwith "to_low_level: Imported NOT IMPLEMENTED YET"
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

let fprint_hum ?(ident_style = `Heuristic_ocannl) ?name ?static_indices () ppf c =
  let nograd_idents = Hashtbl.create (module String) in
  let grad_idents = Hashtbl.create (module String) in
  let visit la =
    let idents = if List.mem ~equal:String.equal la.Tn.label "grad" then grad_idents else nograd_idents in
    Option.iter (Tn.ident_label la)
      ~f:(Hashtbl.update idents ~f:(fun old -> Set.add (Option.value ~default:Utils.no_ints old) la.id))
  in
  let rec loop (c : t) =
    match c with
    | Noop -> ()
    | Seq (c1, c2) ->
        loop c1;
        loop c2
    | Block_comment (_, c) -> loop c
    | Accum_binop { initialize_neutral = _; accum = _; op = _; lhs; rhs1; rhs2; projections = _ } ->
        List.iter ~f:visit [ lhs; rhs1; rhs2 ]
    | Accum_unop { initialize_neutral = _; accum = _; op = _; lhs; rhs; projections = _ } ->
        List.iter ~f:visit [ lhs; rhs ]
    | Fetch { array; fetch_op = _; dims = _ } -> visit array
  in
  loop c;
  let repeating_nograd_idents =
    Hashtbl.filter nograd_idents ~f:(fun ids -> List.length (Set.to_list ids) > 1)
  in
  let repeating_grad_idents = Hashtbl.filter grad_idents ~f:(fun ids -> List.length (Set.to_list ids) > 1) in
  let ident la = Tn.styled_ident ~repeating_nograd_idents ~repeating_grad_idents ident_style la in
  let open Stdlib.Format in
  let out_fetch_op ppf (op : fetch_op) =
    match op with
    | Constant f -> fprintf ppf "%g" f
    | Imported (Ops.C_function c) -> fprintf ppf "%s()" c
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
          (ident rhs1) (Ops.binop_cd_syntax op) (ident rhs2)
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
          (ident rhs)
          (if not (String.equal proj_spec ".") then " ~logic:\"" ^ proj_spec ^ "\"" else "")
    | Fetch { array; fetch_op; dims = _ } -> fprintf ppf "%s := %a;@ " (ident array) out_fetch_op fetch_op
  in
  fprintf ppf "@,@[<v 2>";
  Low_level.fprint_function_header ?name ?static_indices () ppf;
  loop c;
  fprintf ppf "@]"

let%debug_sexp compile_proc ~unoptim_ll_source ~ll_source ~cd_source ~name static_indices (proc : t) :
    (Tn.t, Low_level.traced_array) Base.Hashtbl.t * Low_level.t =
  let llc = to_low_level proc in
  (* Generate the low-level code before outputting the assignments, to force projections. *)
  (match cd_source with
  | None -> ()
  | Some ppf ->
      let ident_style =
        match Utils.get_global_arg ~arg_name:"cd_ident_style" ~default:"heuristic" with
        | "heuristic" -> `Heuristic_ocannl
        | "name_and_label" -> `Name_and_label
        | "name_only" -> `Name_only
        | _ ->
            invalid_arg
              "Assignments.compile_proc: wrong ocannl_cd_ident_style, must be one of: heuristic, \
               name_and_label, name_only"
      in
      fprint_hum ~name ~static_indices ~ident_style () ppf proc);
  Low_level.compile_proc ~unoptim_ll_source ~ll_source ~name static_indices llc
