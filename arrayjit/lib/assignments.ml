open Base
(** The code for operating on n-dimensional arrays. *)

module LA = Lazy_array

(** Resets a array by performing the specified computation or data fetching. *)
type fetch_op =
  | Constant of float
  | Imported of Ops.global_identifier
  | Slice of { batch_idx : Indexing.static_symbol; sliced : LA.t }
  | Embed_symbol of Indexing.static_symbol
[@@deriving sexp_of]

and t =
  | Noop
  | Seq of t * t
  | Block_comment of string * t  (** Same as the given code, with a comment. *)
  | Accum_binop of {
      zero_out : bool;
      accum : Ops.binop;
      op : Ops.binop;
      lhs : LA.t;
      rhs1 : LA.t;
      rhs2 : LA.t;
      projections : Indexing.projections Lazy.t;
    }
  | Accum_unop of {
      zero_out : bool;
      accum : Ops.binop;
      op : Ops.unop;
      lhs : LA.t;
      rhs : LA.t;
      projections : Indexing.projections Lazy.t;
    }
  | Fetch of { array : LA.t; fetch_op : fetch_op; dims : int array Lazy.t }
[@@deriving sexp_of]

let is_noop = function Noop -> true | _ -> false

let rec get_name =
  let punct_or_sp = Str.regexp "[-@*/:.;, ]" in
  let punct_and_sp = Str.regexp {|[-@*/:.;,]\( |$\)|} in
  function
  | Block_comment (s, _) -> Str.global_replace punct_and_sp "" s |> Str.global_replace punct_or_sp "_"
  | Seq (t1, t2) ->
      let n1 = get_name t1 and n2 = get_name t2 in
      let prefix = String.common_prefix2_length n1 n2 in
      let suffix = String.common_suffix2_length n1 n2 in
      if String.is_empty n1 || String.is_empty n2 then n1 ^ n2
      else String.drop_suffix n1 suffix ^ "_then_" ^ String.drop_prefix n2 prefix
  | _ -> ""

module Nd = Ndarray

let remove_updates array c =
  let rec rm check = function
    | ( Seq ((Accum_binop { lhs; _ } | Accum_unop { lhs; _ }), t)
      | Seq (t, (Accum_binop { lhs; _ } | Accum_unop { lhs; _ })) ) as c
      when check ->
        if LA.equal array lhs then rm true t else rm false c
    | Seq (t1, t2) -> Seq (rm true t1, rm true t2)
    | (Accum_binop { lhs; _ } | Accum_unop { lhs; _ }) when LA.equal array lhs -> Noop
    | c -> c
  in
  rm true c

let sequential = List.fold_right ~init:Noop ~f:(fun st sts -> Seq (st, sts))

let to_low_level (code : t) : Low_level.t =
  let open Indexing in
  let get a idcs =
    if not (Array.length idcs = Array.length (Lazy.force a.LA.dims)) then
      Stdlib.Format.printf "DEBUG: get a=%a: %s@ idcs=%a dims=%a\n%!" Sexp.pp_hum
        ([%sexp_of: LA.t] @@ a)
        (LA.label a) Sexp.pp_hum
        ([%sexp_of: Indexing.axis_index array] @@ idcs)
        Sexp.pp_hum
        ([%sexp_of: int array] @@ Lazy.force a.dims);
    assert (Array.length idcs = Array.length (Lazy.force a.LA.dims));
    Low_level.Get (a, idcs)
  in
  let set a idcs v =
    if not (Array.length idcs = Array.length (Lazy.force a.LA.dims)) then
      Stdlib.Format.printf "DEBUG: set a=%a: %s@ idcs=%a dims=%a\n%!" Sexp.pp_hum
        ([%sexp_of: LA.t] @@ a)
        (LA.label a) Sexp.pp_hum
        ([%sexp_of: Indexing.axis_index array] @@ idcs)
        Sexp.pp_hum
        ([%sexp_of: int array] @@ Lazy.force a.dims);
    assert (Array.length idcs = Array.length (Lazy.force a.LA.dims));
    Low_level.Set (a, idcs, v)
  in
  let rec loop code =
    match code with
    | Accum_binop { zero_out; accum; op; lhs; rhs1; rhs2; projections } ->
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
        let basecase rev_iters =
          let product = Array.of_list_rev_map rev_iters ~f:(fun s -> Indexing.Iterator s) in
          let rhs1_idcs = rhs1_idx ~product in
          let rhs2_idcs = rhs2_idx ~product in
          let lhs_idcs = lhs_idx ~product in
          let open Low_level in
          let lhs_ll = get lhs lhs_idcs in
          let rhs1_ll = get rhs1 rhs1_idcs in
          let rhs2_ll = get rhs2 rhs2_idcs in
          set lhs lhs_idcs @@ binop ~op:accum ~rhs1:lhs_ll ~rhs2:(binop ~op ~rhs1:rhs1_ll ~rhs2:rhs2_ll)
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
            Caml.Format.printf "DEBUG: projections=@ %a\n%!" Sexp.pp_hum
              ([%sexp_of: projections] @@ projections);
            raise e
        in
        if zero_out then
          let dims = lazy projections.lhs_dims in
          Low_level.Seq (loop (Fetch { array = lhs; fetch_op = Constant 0.; dims }), for_loops)
        else for_loops
    | Accum_unop { zero_out; accum; op; lhs; rhs; projections } ->
        let projections = Lazy.force projections in
        let lhs_idx =
          derive_index ~product_syms:projections.product_iterators ~projection:projections.project_lhs
        in
        let rhs_idx =
          derive_index ~product_syms:projections.product_iterators ~projection:projections.project_rhs.(0)
        in
        let basecase rev_iters =
          let product = Array.of_list_rev_map rev_iters ~f:(fun s -> Indexing.Iterator s) in
          let lhs_idcs = lhs_idx ~product in
          let open Low_level in
          let lhs_ll = get lhs lhs_idcs in
          let rhs_ll = get rhs @@ rhs_idx ~product in
          set lhs lhs_idcs @@ binop ~op:accum ~rhs1:lhs_ll ~rhs2:(unop ~op ~rhs:rhs_ll)
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
        if zero_out then
          let dims = lazy projections.lhs_dims in
          Low_level.Seq (loop (Fetch { array = lhs; fetch_op = Constant 0.; dims }), for_loops)
        else for_loops
    | Noop -> Low_level.Noop
    | Block_comment (s, c) -> Low_level.Seq (Comment s, loop c)
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

let compile_proc ~name ?(verbose = false) static_indices proc =
  if verbose then Stdio.printf "Assignments.compile_proc: generating the initial low-level code\n%!";
  if Utils.settings.with_debug && Utils.settings.keep_files_in_run_directory then (
    let fname = name ^ ".hlc" in
    let f = Stdio.Out_channel.create fname in
    let ppf = Stdlib.Format.formatter_of_out_channel f in
    Stdlib.Format.pp_set_margin ppf !Low_level.code_sexp_margin;
    Stdlib.Format.fprintf ppf "%a%!" Sexp.pp_hum (sexp_of_t proc));
  let llc = to_low_level proc in
  Low_level.compile_proc ~name ~verbose static_indices llc

let flatten c =
  let rec loop = function
    | Noop -> []
    | Seq (c1, c2) -> loop c1 @ loop c2
    | Block_comment (s, c) -> Block_comment (s, Noop) :: loop c
    | (Accum_binop _ | Accum_unop _ | Fetch _) as c -> [ c ]
  in
  loop c

let to_string_hum c =
  let b = Buffer.create 16 in
  let out = Buffer.add_string b in
  let sp () = out " " in
  let ident la = out @@ LA.name la in
  let out_fetch_op (op : fetch_op) =
    match op with
    | Constant f -> out @@ Float.to_string f
    | Imported (Ops.C_function c) ->
        out c;
        out "()"
    | Imported (Ops.External_unsafe { ptr; prec; dims = _ }) -> out @@ Ops.ptr_to_string ptr prec
    | Slice { batch_idx; sliced } ->
        ident sliced;
        out " @| ";
        out @@ Indexing.symbol_ident batch_idx.static_symbol
    | Embed_symbol { static_symbol; static_range = _ } ->
        out "!@";
        out @@ Indexing.symbol_ident static_symbol
  in
  let rec loop = function
    | Noop -> ()
    | Seq (c1, c2) ->
        loop c1;
        out ";\n";
        loop c2
    | Block_comment (s, Noop) ->
        out "# \"";
        out s;
        out "\""
    | Block_comment (s, c) ->
        out "# \"";
        out s;
        out "\";\n";
        loop c
    | Accum_binop { zero_out; accum; op; lhs; rhs1; rhs2; projections } ->
        let proj_spec =
          if Lazy.is_val projections then (Lazy.force projections).debug_info.spec else "<not-in-yet>"
        in
        ident lhs;
        sp ();
        out @@ Ops.assign_op_cd_syntax ~zero_out accum;
        sp ();
        ident rhs1;
        sp ();
        out @@ Ops.binop_cd_syntax op;
        sp ();
        ident rhs2;
        if (not (String.equal proj_spec ".")) || List.mem ~equal:Ops.equal_binop Ops.[ Mul; Div ] op then (
          out " ~logic:\"";
          out proj_spec;
          out "\"")
    | Accum_unop { zero_out; accum; op; lhs; rhs; projections } ->
        let proj_spec =
          if Lazy.is_val projections then (Lazy.force projections).debug_info.spec else "<not-in-yet>"
        in
        ident lhs;
        sp ();
        out @@ Ops.assign_op_cd_syntax ~zero_out accum;
        sp ();
        out @@ Ops.unop_cd_syntax op;
        if String.equal proj_spec "." then out ".";
        sp ();
        ident rhs;
        if not (String.equal proj_spec ".") then (
          out " ~logic:\"";
          out proj_spec;
          out "\"")
    | Fetch { array; fetch_op; dims = _ } ->
        ident array;
        out " := ";
        out_fetch_op fetch_op
  in
  loop c;
  Buffer.contents b

let code_margin = ref 80

let fprint_code ppf c =
  let flat = flatten c |> List.map ~f:to_string_hum in
  let open Stdlib.Format in
  pp_set_margin ppf !code_margin;
  pp_print_list
    ~pp_sep:(fun f () ->
      pp_print_string f ";";
      pp_print_newline f ())
    pp_print_string ppf flat
