open Base
(** The code for operating on n-dimensional arrays. *)

module LA = Lazy_array

(** Resets a array by performing the specified computation or data fetching. *)
type fetch_op =
  | Constant of float
  | Imported of Ops.global_identifier
  | Slice of { batch_idx : Indexing.static_symbol; sliced : LA.t }
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
      if String.is_empty n1 || String.is_empty n2 then n1 ^ n2 else n1 ^ "_then_" ^ n2
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
          let lhs_ll = Get (lhs, lhs_idcs) in
          let rhs1_ll = Get (rhs1, rhs1_idcs) in
          let rhs2_ll = Get (rhs2, rhs2_idcs) in
          Set (lhs, lhs_idcs, binop ~op:accum ~rhs1:lhs_ll ~rhs2:(binop ~op ~rhs1:rhs1_ll ~rhs2:rhs2_ll))
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
        let s = Low_level.Comment ("Computing array " ^ LA.name lhs) in
        (* Note: it might be invalid to replicate computation across tasks. *)
        if zero_out then
          let dims = lazy projections.lhs_dims in
          Low_level.unflat_lines [ s; loop (Fetch { array = lhs; fetch_op = Constant 0.; dims }); for_loops ]
        else Low_level.Seq (s, for_loops)
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
          let lhs_ll = Get (lhs, lhs_idcs) in
          let rhs_ll = Get (rhs, rhs_idx ~product) in
          Set (lhs, lhs_idcs, binop ~op:accum ~rhs1:lhs_ll ~rhs2:(unop ~op ~rhs:rhs_ll))
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
        let s = Low_level.Comment ("Computing node " ^ LA.name lhs) in
        if zero_out then
          let dims = lazy projections.lhs_dims in
          Low_level.unflat_lines [ s; loop (Fetch { array = lhs; fetch_op = Constant 0.; dims }); for_loops ]
        else Seq (s, for_loops)
    | Noop -> Low_level.Noop
    | Block_comment (s, c) -> Low_level.Seq (Comment s, loop c)
    | Seq (c1, c2) -> Seq (loop c1, loop c2)
    | Fetch { array; fetch_op = Constant 0.0; dims = _ } -> Zero_out array
    | Fetch { array; fetch_op = Constant c; dims } ->
        Low_level.loop_over_dims (Lazy.force dims) ~body:(fun idcs -> Set (array, idcs, Constant c))
    | Fetch { array; fetch_op = Slice { batch_idx = { static_symbol = idx; _ }; sliced }; dims } ->
        (* TODO: doublecheck this always gets optimized away. *)
        Low_level.loop_over_dims (Lazy.force dims) ~body:(fun idcs ->
            Set (array, idcs, Get (sliced, Array.append [| Iterator idx |] idcs)))
    | Fetch { array = _; fetch_op = Imported _; dims = _ } ->
        failwith "to_low_level: Imported NOT IMPLEMENTED YET"
  in

  loop code

let compile_proc ~name ?(verbose = false) proc =
  if verbose then Stdio.printf "Assignments.compile_proc: generating the initial low-level code\n%!";
  if !Low_level.with_debug && !Low_level.keep_files_in_run_directory then (
    let fname = name ^ ".hlc" in
    let f = Stdio.Out_channel.create fname in
    let ppf = Caml.Format.formatter_of_out_channel f in
    Caml.Format.pp_set_margin ppf !Low_level.code_sexp_margin;
    Caml.Format.fprintf ppf "%a%!" Sexp.pp_hum (sexp_of_t proc));
  let llc = to_low_level proc in
  Low_level.compile_proc ~name ~verbose llc

let fprint_code ppf c =
  (* TODO: something nicely concise. *)
  Caml.Format.pp_set_margin ppf !Low_level.code_sexp_margin;
  Caml.Format.fprintf ppf "%s" @@ Sexp.to_string_hum @@ sexp_of_t c
