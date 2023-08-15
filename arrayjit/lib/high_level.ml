open Base
(** The code for operating on n-dimensional arrays. *)

type ndarray = Low_level.ndarray [@@deriving sexp_of, equal]

(** Resets a tensor by performing the specified computation or data fetching. *)
type fetch_op = Constant of float | Synthetic of t | Imported of Low_level.global_identifier
[@@deriving sexp_of]

and t =
  | Seq of t * t
      (** These tasks can only benefit from mutual parallelism via operator fusion / loop fusion. *)
  | Accum_binop of {
      zero_out : bool;
      accum : Low_level.binop;
      op : Low_level.binop;
      lhs : ndarray;
      rhs1 : ndarray;
      rhs2 : ndarray;
      projections : unit -> Indexing.projections;
    }
  | Accum_unop of {
      zero_out : bool;
      accum : Low_level.binop;
      op : Low_level.unop;
      lhs : ndarray;
      rhs : ndarray;
      projections : unit -> Indexing.projections;
    }
  | Fetch of { tensor : ndarray; fetch_op : fetch_op; dims : unit -> Indexing.dim array }
  | Block_comment of string * t
  | Noop
[@@deriving sexp_of]

(** If a backend does not support detection of when [ParHint (c1, c2)] is safe to parallelize,
    one can try setting [force_unsafe_parhint] to always parallelize if the particular code
    does not have a form of computation sharing that would get broken. *)
let force_unsafe_parhint = ref false

module Nd = Ndarray

let remove_updates tensor c =
  let rec rm check = function
    | ( Seq ((Accum_binop { lhs; _ } | Accum_unop { lhs; _ }), t)
      | Seq (t, (Accum_binop { lhs; _ } | Accum_unop { lhs; _ })) ) as c
      when check ->
        if equal_ndarray tensor lhs then rm true t else rm false c
    | Seq (t1, t2) -> Seq (rm true t1, rm true t2)
    | (Accum_binop { lhs; _ } | Accum_unop { lhs; _ }) when equal_ndarray tensor lhs -> Noop
    | c -> c
  in
  rm true c

let sequential = List.fold_right ~init:Noop ~f:(fun st sts -> Seq (st, sts))

let to_low_level (code : t) : Low_level.t =
  let rec loop code =
    match code with
    | Accum_binop { zero_out; accum; op; lhs; rhs1; rhs2; projections } ->
        let projections = projections () in
        let lhs_idx =
          Indexing.(
            derive_index ~product_syms:projections.product_iterators ~projection:projections.project_lhs)
        in
        let rhs1_idx =
          Indexing.(
            derive_index ~product_syms:projections.product_iterators ~projection:projections.project_rhs1)
        in
        let rhs2_idx =
          match projections.project_rhs2 with
          | None -> invalid_arg "accum_binop: projections missing project_rhs2"
          | Some rhs2 -> Indexing.(derive_index ~product_syms:projections.product_iterators ~projection:rhs2)
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
              let index = Indexing.get_sym_for_axis d.Indexing.special in
              For_loop
                {
                  index;
                  from_ = 0;
                  to_ = d.dim - 1;
                  body = for_loop (index :: rev_iters) product;
                  trace_it = true;
                }
        in
        let for_loops = for_loop [] (Array.to_list projections.product_space) in
        let s = Low_level.Comment ("Computing tensor " ^ Ndarray.get_name lhs) in
        (* Note: it might be invalid to replicate computation across tasks. *)
        if zero_out then
          let dims () = projections.lhs_dims in
          Low_level.unflat_lines [ s; loop (Fetch { tensor = lhs; fetch_op = Constant 0.; dims }); for_loops ]
        else Low_level.Seq (s, for_loops)
    | Accum_unop { zero_out; accum; op; lhs; rhs; projections } ->
        let projections = projections () in
        let lhs_idx =
          Indexing.(
            derive_index ~product_syms:projections.product_iterators ~projection:projections.project_lhs)
        in
        let rhs_idx =
          Indexing.(
            derive_index ~product_syms:projections.product_iterators ~projection:projections.project_rhs1)
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
              let index = Indexing.get_sym_for_axis d.Indexing.special in
              For_loop
                {
                  index;
                  from_ = 0;
                  to_ = d.dim - 1;
                  body = for_loop (index :: rev_iters) product;
                  trace_it = true;
                }
        in
        let for_loops = for_loop [] (Array.to_list projections.product_space) in
        let s = Low_level.Comment ("Computing node " ^ Ndarray.get_name lhs) in
        (* Note: it might be invalid to replicate computation across tasks. *)
        if zero_out then
          let dims () = projections.lhs_dims in
          Low_level.unflat_lines [ s; loop (Fetch { tensor = lhs; fetch_op = Constant 0.; dims }); for_loops ]
        else Seq (s, for_loops)
    | Noop -> Low_level.Noop
    | Block_comment (s, c) -> Low_level.Seq (Comment s, loop c)
    | Seq (c1, c2) -> Seq (loop c1, loop c2)
    | Fetch { tensor; fetch_op = Constant 0.0; dims = _ } -> Zero_out tensor
    | Fetch { tensor; fetch_op = Constant c; dims } ->
        Low_level.loop_over_dims ~skip_frozen:false (dims ()) ~body:(fun idcs ->
            Set (tensor, idcs, Constant c))
        (* let rec loop rev_idcs = function
             | [] -> Set (tensor, Array.of_list_rev rev_idcs, Constant c)
             | d :: product when Indexing.dim_1 d -> loop (Fixed_idx 0 :: rev_idcs) product
             | d :: product ->
                 let index = Indexing.get_sym_for_axis d.Indexing.special in
                 For_loop
                   {
                     index;
                     from_ = 0;
                     to_ = d.dim - 1;
                     body = loop (Indexing.Iterator index :: rev_idcs) product;
                     trace_it = true;
                   }
           in
           loop [] (Array.to_list product_space) *)
    | Fetch { tensor = _; fetch_op = Synthetic gen; dims = _ } -> loop gen
    | Fetch { tensor = _; fetch_op = Imported _; dims = _ } ->
        failwith "to_low_level: Imported NOT IMPLEMENTED YET"
  in

  loop code

let compile_proc ~name ?(verbose = false) ~for_step_update proc =
  if verbose then Stdio.printf "Code.compile_proc: generating the initial low-level code\n%!";
  let llc = to_low_level proc in
  if !Low_level.with_debug && !Low_level.keep_files_in_run_directory then (
    let fname = name ^ ".hlc" in
    let f = Stdio.Out_channel.create fname in
    let ppf = Caml.Format.formatter_of_out_channel f in
    Caml.Format.pp_set_margin ppf !Low_level.code_sexp_margin;
    Caml.Format.fprintf ppf "%a%!" Sexp.pp_hum (sexp_of_t proc));
  Low_level.compile_proc ~name ~verbose ~for_step_update llc
