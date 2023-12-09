(** The row type, shape inference related types and constraint solving. *)

open Base
module Utils = Arrayjit.Utils

module Dim_var = struct
  type t = { id : int; label : string option [@compare.ignore] [@equal.ignore] [@hash.ignore] }
  [@@deriving equal, hash, compare, sexp]

  include Comparator.Make (struct
    type nonrec t = t

    let compare = compare
    let sexp_of_t = sexp_of_t
  end)
end

type dim_var = Dim_var.t [@@deriving equal, hash, compare, sexp]
type dim_cmp = Dim_var.comparator_witness
type 'a dim_map = 'a Map.M(Dim_var).t [@@deriving equal, sexp]

let dim_map_empty = Map.empty (module Dim_var)

(** A single axis in a shape. *)
type dim = Var of dim_var | Dim of { d : int; label : string option; proj_id : int option }
[@@deriving equal, hash, compare, sexp, variants]

let uid = ref 0

let get_var ?label () : dim_var =
  Int.incr uid;
  { id = !uid; label }

let get_dim ~d ?label () = Dim { d; label; proj_id = None }

type 'a dim_hashtbl = 'a Hashtbl.M(Dim_var).t [@@deriving sexp]

let dim_hashtbl () = Hashtbl.create (module Dim_var)

let dim_to_string style = function
  | Dim { label = None; _ } when phys_equal style `Only_labels -> "_"
  | Dim { label = Some l; _ } when phys_equal style `Only_labels -> l
  | Dim { d; label = None; _ } -> Int.to_string d
  | Dim { d; label = Some l; _ } -> [%string "%{l}=%{d#Int}"]
  | Var { id; label = Some l } -> [%string "$%{id#Int}:%{l}"]
  | Var { id; label = None } -> "$" ^ Int.to_string id

module Row_var = struct
  type t = Row_var of int [@@deriving equal, hash, compare, sexp]

  include Comparator.Make (struct
    type nonrec t = t

    let compare = compare
    let sexp_of_t = sexp_of_t
  end)

  let get () =
    Int.incr uid;
    Row_var !uid
end

type row_var = Row_var.t [@@deriving equal, hash, compare, sexp]

let get_row_var = Row_var.get

(** A bcast specifies how axes of a single kind in a shape (i.e. the row) can adapt to other shapes. *)
type bcast =
  | Row_var of row_var  (** The row can be inferred to have more axes. *)
  | Broadcastable  (** The shape does not have more axes of this kind, but is "polymorphic". *)
[@@deriving equal, hash, compare, sexp, variants]

type dims_constraint =
  | Unconstrained
  | Total_elems of int  (** The shape-kind, inclusive of the further row spec, has this many elements. *)
[@@deriving equal, hash, compare, sexp, variants]

type kind = [ `Batch | `Input | `Output ] [@@deriving equal, compare, sexp, hash, variants]

module Row_id = struct
  type t = { sh_id : int; kind : kind } [@@deriving sexp, compare, equal, hash]

  include Comparator.Make (struct
    type nonrec t = t

    let compare = compare
    let sexp_of_t = sexp_of_t
  end)
end

type row_id = Row_id.t [@@deriving sexp, compare, equal, hash]
type row_cmp = Row_id.comparator_witness

let row_id ~sh_id ~kind = Row_id.{ sh_id; kind }

(* let row_map_empty = Map.empty (module Row_id) *)

type t = { dims : dim list; constr : dims_constraint; bcast : bcast; id : row_id }
[@@deriving equal, hash, compare, sexp]

let dims_label_assoc dims =
  let f = function Var { label = Some l; _ } as d -> Some (l, d) | _ -> None in
  List.filter_map dims.dims ~f

(** An entry implements inequalities [cur >= v >= subr] and/or an equality [v = solved]. *)
type ('a, 'b) entry = Solved of 'b | Bounds of { cur : 'a list; subr : 'a list; lub : 'b option }
[@@deriving sexp]

type dim_env = (dim_var, dim) entry Map.M(Dim_var).t [@@deriving sexp]
type row_env = (row_var, t) entry Map.M(Row_var).t [@@deriving sexp]

type environment = {
  dim_env : dim_env;
  row_env : row_env;
  dim_rev_elim_order : dim_var list;
  row_rev_elim_order : row_var list;
}
[@@deriving sexp]
(** Note that while we build up the partial sets of inequalities, the environment is not in solved form.
    It is only in resolved wrt. variables that are solved: [v -> e where Option.is_some e.solved]
    do not appear elsewhere in the environment. But once [finish_inference] is called, it becomes in
    solved form: variables later in the elimination order do not appear in entries for variables
    earlier in the elimination order. *)

type inequality =
  | Dim_eq of { d1 : dim; d2 : dim }
  | Row_eq of { r1 : t; r2 : t }
  | Dim_ineq of { cur : dim; subr : dim }
  | Row_ineq of { cur : t; subr : t }
[@@deriving compare, equal, sexp]

type error_trace = ..

type error_trace +=
  | Row_mismatch of t list
  | Dim_mismatch of dim list
  | Index_mismatch of Arrayjit.Indexing.axis_index list

let sexp_of_error_trace = function
  | Row_mismatch rs -> Sexp.List (Sexp.Atom "Row_mismatch" :: List.map rs ~f:sexp_of_t)
  | Dim_mismatch ds -> Sexp.List (Sexp.Atom "Dim_mismatch" :: List.map ds ~f:sexp_of_dim)
  | Index_mismatch idcs ->
      Sexp.List (Sexp.Atom "Index_mismatch" :: List.map idcs ~f:Arrayjit.Indexing.sexp_of_axis_index)
  | _ -> Sexp.Atom "<outdated version of sexp_of_error_trace>"

exception Shape_error of string * error_trace list [@@deriving sexp_of]

let dim_to_int_exn = function Dim { d; _ } -> d | Var _ -> invalid_arg "dim_to_int: dim still unknown"

let meet more_constr constr =
  match (more_constr, constr) with
  | Unconstrained, c -> c
  | c, Unconstrained -> c
  | (Total_elems n1 as c), Total_elems n2 when n1 = n2 -> c
  | Total_elems _, Total_elems _ -> raise @@ Shape_error ("Incompatible Total_elems constraints", [])

let s_dim_one v ~value ~in_ = match in_ with Var v2 when equal_dim_var v v2 -> value | _ -> in_

let s_dim_one_in_entry v ~value in_ =
  match in_ with
  | Solved in_ -> ([], Solved (s_dim_one v ~value ~in_))
  | Bounds { cur; subr; lub } ->
      let find_v side = List.partition_tf side ~f:(equal_dim_var v) in
      let v_cur, cur = find_v cur in
      let v_subr, subr = find_v subr in
      let ineqs0 =
        match (v_subr, lub) with _ :: _, Some lub -> [ Dim_ineq { cur = lub; subr = value } ] | _ -> []
      in
      let ineqs1 =
        if List.is_empty v_subr then []
        else List.map cur ~f:(fun cur -> Dim_ineq { cur = Var cur; subr = value })
      in
      let ineqs2 =
        if List.is_empty v_cur then []
        else List.map subr ~f:(fun subr -> Dim_ineq { subr = Var subr; cur = value })
      in
      ( ineqs0 @ ineqs1 @ ineqs2,
        Bounds { cur; subr; lub = Option.map lub ~f:(fun in_ -> s_dim_one v ~value ~in_) } )

let s_dim_one_in_row v ~value in_ =
  { in_ with dims = List.map in_.dims ~f:(fun in_ -> s_dim_one v ~value ~in_) }

let s_dim_one_in_row_entry v ~value in_ =
  match in_ with
  | Solved in_ -> Solved (s_dim_one_in_row v ~value in_)
  | Bounds { cur; subr; lub } -> Bounds { cur; subr; lub = Option.map lub ~f:(s_dim_one_in_row v ~value) }

let subst_dim env = function
  | Dim _ as d -> d
  | Var v as default -> ( match Map.find env.dim_env v with Some (Solved d) -> d | _ -> default)

let s_row_one v ~value:{ dims = more_dims; constr = more_constr; bcast; id = _ } ~in_ =
  match in_ with
  | { dims; constr; bcast = Row_var v2; id } when equal_row_var v v2 ->
      let more_constr =
        match more_constr with
        | Unconstrained -> Unconstrained
        | Total_elems m ->
            if List.for_all dims ~f:is_dim then
              Total_elems (m * List.fold dims ~init:1 ~f:(fun n d -> n * dim_to_int_exn d))
            else Unconstrained (* Wait for more shape inference. *)
      in
      { dims = more_dims @ dims; constr = meet more_constr constr; bcast; id }
  | _ -> in_

let s_row_one_in_entry v ~value in_ =
  match in_ with
  | Solved in_ -> ([], Solved (s_row_one v ~value ~in_))
  | Bounds { cur; subr; lub } ->
      (* TODO: audit code to ensure we don't lose the constraints associated with the bounds variables. *)
      let row_of_var v = { dims = []; bcast = Row_var v; id = value.id; constr = Unconstrained } in
      let find_v side = List.partition_tf side ~f:(equal_row_var v) in
      let v_cur, cur = find_v cur in
      let v_subr, subr = find_v subr in
      let ineqs0 =
        match (v_subr, lub) with _ :: _, Some lub -> [ Row_ineq { cur = lub; subr = value } ] | _ -> []
      in
      let ineqs1 =
        if List.is_empty v_subr then []
        else List.map cur ~f:(fun cur -> Row_ineq { cur = row_of_var cur; subr = value })
      in
      let ineqs2 =
        if List.is_empty v_cur then []
        else List.map subr ~f:(fun subr -> Row_ineq { subr = row_of_var subr; cur = value })
      in
      ( ineqs0 @ ineqs1 @ ineqs2,
        Bounds { cur; subr; lub = Option.map lub ~f:(fun in_ -> s_row_one v ~value ~in_) } )

let subst_row env ({ dims; constr; bcast; id } as default) =
  let dims = List.map dims ~f:(subst_dim env) in
  match bcast with
  | Broadcastable -> { dims; constr; bcast; id }
  | Row_var v -> (
      match Map.find env.row_env v with
      | None | Some (Bounds _) -> default
      | Some (Solved { dims = more_dims; constr = Unconstrained; bcast; id = _ }) ->
          { dims = more_dims @ dims; constr; bcast; id }
      | Some (Solved { dims = more_dims; constr = Total_elems m; bcast; id = _ }) ->
          let more_constr =
            if List.for_all dims ~f:is_dim then
              Total_elems (m * List.fold dims ~init:1 ~f:(fun n d -> n * dim_to_int_exn d))
            else Unconstrained (* Wait for more shape inference. *)
          in
          { dims = more_dims @ dims; constr = meet more_constr constr; bcast; id })

(* let occurs_row v = function { bcast = Row_var v'; _ } -> v = v' | _ -> false *)

let rec unify_dim ((dim1, dim2) as eq) env =
  match eq with
  | Dim { label = Some l1; _ }, Dim { label = Some l2; _ } when not (String.equal l1 l2) ->
      raise @@ Shape_error ("solved dimensions for axis: different labels", [ Dim_mismatch [ dim1; dim2 ] ])
  | Dim { d = d1; _ }, Dim { d = d2; _ } when d1 = d2 -> ([], env)
  | Var v, d2 | d2, Var v ->
      let ineqs = ref [] in
      let f in_ =
        let more_ineqs, result = s_dim_one_in_entry v ~value:d2 in_ in
        ineqs := more_ineqs @ !ineqs;
        result
      in
      let env =
        match Map.find env.dim_env v with
        | None ->
            let dim_env = Map.map env.dim_env ~f in
            {
              dim_env = Map.add_exn dim_env ~key:v ~data:(Solved d2);
              row_env = Map.map env.row_env ~f:(s_dim_one_in_row_entry v ~value:d2);
              dim_rev_elim_order = v :: env.dim_rev_elim_order;
              row_rev_elim_order = env.row_rev_elim_order;
            }
        | Some (Solved d1) ->
            let more_ineqs, env = unify_dim (d1, d2) env in
            ineqs := more_ineqs @ !ineqs;
            env
        | Some (Bounds { cur; subr; lub }) ->
            let dim_env = Map.map env.dim_env ~f in
            List.iter cur ~f:(fun cur -> ineqs := Dim_ineq { cur = Var cur; subr = d2 } :: !ineqs);
            List.iter subr ~f:(fun subr -> ineqs := Dim_ineq { subr = Var subr; cur = d2 } :: !ineqs);
            Option.iter lub ~f:(fun lub -> ineqs := Dim_ineq { cur = lub; subr = d2 } :: !ineqs);
            {
              env with
              dim_env = Map.update dim_env v ~f:(fun _ -> Solved d2);
              row_env = Map.map env.row_env ~f:(s_dim_one_in_row_entry v ~value:d2);
            }
      in
      let dim_eqs, ineqs =
        List.partition_map !ineqs ~f:(function Dim_eq { d1; d2 } -> First (d1, d2) | ineq -> Second ineq)
      in
      let f (ineqs, env) ds =
        let more_ineqs, env = unify_dim ds env in
        (more_ineqs @ ineqs, env)
      in
      List.fold ~init:(ineqs, env) dim_eqs ~f
  | _ ->
      (* Note: at the unify_dim stage, it's strict equality (no broadcasting). *)
      raise @@ Shape_error ("solved dimensions for axis: mismatch", [ Dim_mismatch [ dim1; dim2 ] ])

let drop_from_end l n = List.rev @@ List.drop (List.rev l) n
let take_from_end (l : dim list) (n : int) : dim list = List.rev @@ List.take (List.rev l) n

let apply_constraint r env =
  let r = subst_row env r in
  match r.constr with
  | Unconstrained -> ([], env)
  | Total_elems n -> (
      match r.bcast with
      | Row_var _ -> ([], env (* Wait for more shape inference. *))
      | Broadcastable -> (
          let dims = List.map r.dims ~f:(subst_dim env) in
          let vars, nonvars = List.partition_tf dims ~f:is_var in
          if List.length vars > 1 then ([], env (* Wait for more shape inference. *))
          else
            let known = List.fold nonvars ~init:1 ~f:(fun n d -> n * dim_to_int_exn d) in
            match vars with
            | [] ->
                if n <> known then (
                  if Utils.settings.with_debug then
                    Stdlib.Format.printf "apply_constraint: shape error env=@ %a\n%!" Sexp.pp_hum
                      (sexp_of_environment env);
                  raise @@ Shape_error ("Total_elems constraint failed", [ Row_mismatch [ r ] ]))
                else ([], env)
            | [ Var v ] ->
                let rem = n / known in
                if rem = 0 then (
                  if Utils.settings.with_debug then
                    Stdlib.Format.printf "apply_constraint: shape error env=@ %a\n%!" Sexp.pp_hum
                      (sexp_of_environment env);
                  raise @@ Shape_error ("Total_elems constraint failed", [ Row_mismatch [ r ] ]))
                else unify_dim (Var v, get_dim ~d:rem ()) env
            | _ -> assert false))

let prefix_constraint ~drop row =
  match row.constr with
  | Unconstrained -> Unconstrained
  | Total_elems n ->
      let prefix = take_from_end row.dims drop in
      if List.exists prefix ~f:(Fn.non is_dim) then (* wait for more inference *) Unconstrained
      else
        let discarded = List.fold prefix ~init:1 ~f:(fun n d -> n * dim_to_int_exn d) in
        let result = n / discarded in
        if result <= 0 then raise @@ Shape_error ("Not enough elements", [ Row_mismatch [ row ] ])
        else Total_elems result

(* Equate two rows, no broadcasting. Does not resolve inequalities. *)
let rec unify_row ((row1, row2) as eq) env =
  let rec solve (ineqs, env) = function
    | Dim_eq { d1; d2 } ->
        let more_ineqs, env = unify_dim (d1, d2) env in
        List.fold ~init:(ineqs, env) more_ineqs ~f:solve
    | Row_eq { r1; r2 } ->
        let more_ineqs, env = unify_row (r1, r2) env in
        (more_ineqs @ ineqs, env)
    | (Dim_ineq _ | Row_ineq _) as ineq -> (ineq :: ineqs, env)
  in
  let unify_prefix len =
    let dims1 = take_from_end row1.dims len and dims2 = take_from_end row2.dims len in
    List.fold ~init:([], env) ~f:(fun acc (d1, d2) -> solve acc (Dim_eq { d1; d2 }))
    @@ List.zip_exn dims1 dims2
  in
  match eq with
  | { bcast = Row_var v; dims = r1_dims; id; constr = _ }, r2
  | r2, { bcast = Row_var v; dims = r1_dims; id; constr = _ } -> (
      let r1_len = List.length r1_dims and r2_len = List.length r2.dims in
      if r1_len > r2_len then
        if is_row_var r2.bcast then unify_row (row2, row1) env
        else raise @@ Shape_error ("Number of axes mismatch", [ Row_mismatch [ row1; row2 ] ])
      else
        let ineqs, env =
          try unify_prefix r1_len
          with Shape_error (s, trace) -> raise @@ Shape_error (s, Row_mismatch [ row1; row2 ] :: trace)
        in
        let constr = meet row1.constr @@ prefix_constraint ~drop:r1_len row2 in
        let value = { constr; bcast = r2.bcast; dims = drop_from_end r2.dims r1_len; id } in
        let ineqs = ref ineqs in
        let f in_ =
          let more_ineqs, result = s_row_one_in_entry v ~value:r2 in_ in
          ineqs := more_ineqs @ !ineqs;
          result
        in
        match Map.find env.row_env v with
        | None ->
            let row_env = Map.map env.row_env ~f in
            let env =
              {
                env with
                row_env = Map.add_exn row_env ~key:v ~data:(Solved value);
                row_rev_elim_order = v :: env.row_rev_elim_order;
              }
            in
            List.fold ~init:([], env) ~f:solve !ineqs
        | Some (Solved r1) -> unify_row (r1, value) env
        | Some (Bounds { cur; subr; lub }) ->
            (* TODO: audit code to ensure we don't lose the constraints associated with the bounds variables. *)
            let row_of_var v = { dims = []; bcast = Row_var v; id = value.id; constr = Unconstrained } in
            let row_env = Map.map env.row_env ~f in
            List.iter cur ~f:(fun cur -> ineqs := Row_ineq { cur = row_of_var cur; subr = r2 } :: !ineqs);
            List.iter subr ~f:(fun subr -> ineqs := Row_ineq { subr = row_of_var subr; cur = r2 } :: !ineqs);
            Option.iter lub ~f:(fun lub -> ineqs := Row_ineq { cur = lub; subr = r2 } :: !ineqs);
            let env = { env with row_env = Map.update row_env v ~f:(fun _ -> Solved value) } in
            List.fold ~init:([], env) ~f:solve !ineqs)
  | ( { bcast = Broadcastable; dims = dims1; constr = constr1; id = _ },
      { bcast = Broadcastable; dims = dims2; constr = constr2; id = _ } ) ->
      let ineqs, env =
        match List.zip dims1 dims2 with
        | Unequal_lengths ->
            raise @@ Shape_error ("Mismatching number of axes", [ Row_mismatch [ row1; row2 ] ])
        | Ok eqs -> List.fold ~init:([], env) ~f:(fun acc (d1, d2) -> solve acc (Dim_eq { d1; d2 })) eqs
      in
      let more_ineqs, env = apply_constraint { row1 with constr = meet constr1 constr2 } env in
      (more_ineqs @ ineqs, env)

(*
let update_row v ?cur ?subr env =
  (* This is the same as [update_dim] except dealing with more potential side equations. *)
  let no_v =
    List.filter ~f:(fun r ->
        if equal_bcast (Row_var v) r.bcast then
          if List.is_empty r.dims then false
          else raise @@ Shape_error ("Infinite row via self-reference", [ Row_mismatch [ r ] ])
        else true)
  in
  let cur = no_v @@ List.map ~f:(subst_row env) @@ Option.to_list cur in
  let subr = no_v @@ List.map ~f:(subst_row env) @@ Option.to_list subr in
  if List.is_empty cur && List.is_empty subr then env
  else
    (* let guessed_id : row_id = (List.hd_exn @@ cur @ subr).id in *)
    let value = { cur; subr; solved = None } in
    let elim = perhaps_eliminate_row_var v ~v_cur:cur ~v_subr:subr in
    match Map.find env.row_env v with
    | Some in_ ->
        (* Note: this is where the bulk of the work usually happens. *)
        let eq, data = elim ~self:true ~in_ in
        let eqs = eq @ Option.to_list extra_eq in
        let row_env = Map.update env.row_env v ~f:(fun _ -> data) in
        List.fold eqs ~init:{ env with row_env } ~f:(Fn.flip unify_row)
    | None ->
        let eqs = ref @@ Option.to_list extra_eq (* No fold_map in Map. *) in
        let row_env =
          Map.mapi env.row_env ~f:(fun ~key:v2 ~data:in_ ->
              let eq, entry = elim ~self:(v = v2) ~in_ in
              eqs := eq @ !eqs;
              entry)
        in
        let env = List.fold !eqs ~init:{ env with row_env } ~f:(Fn.flip unify_row) in
        {
          env with
          row_env = Map.add_exn env.row_env ~key:v ~data:value;
          row_rev_elim_order = v :: env.row_rev_elim_order;
        }

let add_dim_ineq ~cur ~subr env =
  match (cur, subr) with
  | Dim { label = Some l1; _ }, Dim { label = Some l2; _ } when not (String.equal l1 l2) ->
      raise @@ Shape_error ("dimension comparison for axis: different labels", [ Dim_mismatch [ cur; subr ] ])
  | Dim { d = d1; _ }, Dim { d = d2; _ } when d1 = d2 -> env
  | Dim { d = _; _ }, Dim { d = 1; _ } -> env
  | Var v, _ -> update_dim v ~subr env
  | _, Var v -> update_dim v ~cur env
  | _ -> raise @@ Shape_error ("dimension comparison for axis: mismatch", [ Dim_mismatch [ cur; subr ] ])

let add_row_ineq ~cur ~subr env =
  let prefix_ineqs len =
    let dims1 = take_from_end cur.dims len and dims2 = take_from_end subr.dims len in
    List.fold ~init:env ~f:(fun env (cur, subr) -> add_dim_ineq ~cur ~subr env)
    @@ List.zip_exn dims1 dims2
  in
  let r1_len = List.length cur.dims and r2_len = List.length subr.dims in
  let len = min r1_len r2_len in
  let env =
    try prefix_ineqs len
    with Shape_error (s, trace) -> raise @@ Shape_error (s, Row_mismatch [ cur; subr ] :: trace)
  in
  let reduced ({ bcast; dims; constr = _; id } as r) =
    { bcast; dims = drop_from_end dims len; constr = prefix_constraint ~drop:len r; id }
  in
  match (cur, subr) with
  | { bcast = Row_var v; _ }, _ when r1_len < r2_len ->
      apply_constraint cur env |> update_row v ~subr:(reduced subr)
  | { bcast = Broadcastable; _ }, _ when r1_len < r2_len ->
      raise @@ Shape_error ("Too many axes", [ Row_mismatch [ cur; subr ] ])
  | _, { bcast = Row_var v; _ } when r2_len <= r1_len ->
      apply_constraint subr env |> update_row v ~cur:(reduced cur)
  | _, { bcast = Broadcastable; _ } when r2_len <= r1_len -> apply_constraint cur env |> apply_constraint subr
  | { bcast = Row_var _ | Broadcastable; _ }, { bcast = Row_var _ | Broadcastable; _ } -> assert false
*)

let empty_env =
  {
    dim_env = Map.empty (module Dim_var);
    row_env = Map.empty (module Row_var);
    dim_rev_elim_order = [];
    row_rev_elim_order = [];
  }

let add_dim_ineq ~cur:_ ~subr:_ _env = failwith "NOT IMPLEMENTED YET"
let add_row_ineq ~cur:_ ~subr:_ _env = failwith "NOT IMPLEMENTED YET"

let solve_inequalities ~is_complete ineqs env =
  let rec solve ineqs env =
    let f (ineqs, env) = function
      | Dim_eq { d1; d2 } ->
          let more_ineqs, env = unify_dim (d1, d2) env in
          (more_ineqs @ ineqs, env)
      | Row_eq { r1; r2 } ->
          let more_ineqs, env = unify_row (r1, r2) env in
          (more_ineqs @ ineqs, env)
      | Dim_ineq { cur; subr } ->
          let more_ineqs, env = add_dim_ineq ~cur ~subr env in
          (more_ineqs @ ineqs, env)
      | Row_ineq { cur; subr } ->
          let more_ineqs, env = add_row_ineq ~cur ~subr env in
          (more_ineqs @ ineqs, env)
    in
    let ineqs, env = List.fold ineqs ~init:([], env) ~f in
    if List.is_empty ineqs then env else solve ineqs env
  in
  let env = solve ineqs env in
  if is_complete then env else env

let rec row_to_labels env =
  let rec f = function
    | Dim { label = Some l; _ } -> l
    | Dim { label = None; _ } -> ""
    | Var v -> (
        match Map.find env.dim_env v with
        | None | Some (Bounds _) -> Option.value v.label ~default:""
        | Some (Solved dim) -> f dim)
  in
  function
  | { dims; constr; bcast = Row_var v; id } -> (
      match Map.find env.row_env v with
      | None | Some (Bounds _) -> Array.of_list_map dims ~f
      | Some (Solved row2) -> row_to_labels env { dims = row2.dims @ dims; constr; bcast = row2.bcast; id })
  | { dims; constr = _; bcast = Broadcastable; id = _ } -> Array.of_list_map dims ~f

let fresh_proj =
  let uid = ref 0 in
  fun () ->
    Int.incr uid;
    !uid

let fresh_row_proj r =
  let fresh_dim = function
    | Dim { d; label; proj_id = _ } -> Dim { d; label; proj_id = Some (fresh_proj ()) }
    | Var _ as d -> d
  in
  { r with dims = List.map r.dims ~f:fresh_dim }

(** *** Projection inference *** *)

open Arrayjit.Indexing

type proj_classes = int Map.M(Int).t [@@deriving sexp]

(* let update_proj_classes pid1 pid2 proj_classes = Utils.union_add ~equal:Int.equal proj_classes pid1 pid2 *)

type proj = Var of dim_var | Proj of { proj_id : int; d : int } | Solved of axis_index
[@@deriving compare, equal, sexp]

type error_trace += Projection_mismatch of proj list

let sexp_of_error_trace = function
  | Projection_mismatch ps -> Sexp.List (Sexp.Atom "Projection_mismatch" :: List.map ps ~f:sexp_of_proj)
  | error_trace -> sexp_of_error_trace error_trace

type proj_to_index = Arrayjit.Indexing.axis_index Map.M(Int).t [@@deriving sexp]

type proj_env = {
  proj_to_index : proj_to_index;
  proj_classes : proj_classes;
  product_dim : int Map.M(Int).t;
  non_product : Set.M(Int).t;
}
[@@deriving sexp]

let get_proj_equations inequalities proj_axis_env env =
  let to_proj : dim -> proj = function
    | Var v when Map.mem proj_axis_env v -> Solved (Map.find_exn proj_axis_env v)
    | Dim { proj_id = Some proj_id; d; label = _ } -> Proj { proj_id; d }
    | d -> (
        match subst_dim env d with
        | Dim { proj_id = Some proj_id; d; label = _ } -> Proj { proj_id; d }
        | Dim { proj_id = None; d; _ } -> Proj { proj_id = fresh_proj (); d }
        | Var v when Map.mem proj_axis_env v -> Solved (Map.find_exn proj_axis_env v)
        | Var v -> Var v)
  in
  let expand_dims = function
    | { dims; bcast = Row_var v; _ } when Map.mem env.row_env v -> (
        match Map.find_exn env.row_env v with Solved { dims = more_dims; _ } -> more_dims @ dims | _ -> dims)
    | { dims; _ } -> dims
  in
  let match_rows r1 r2 =
    match List.zip (expand_dims r1) (expand_dims r2) with
    | Unequal_lengths -> raise @@ Shape_error ("Mismatching number of axes", [ Row_mismatch [ r1; r2 ] ])
    | Ok eqs -> List.map ~f:(fun (d1, d2) -> (to_proj d1, to_proj d2)) eqs
  in
  let f = function
    | Dim_eq { d1; d2 } | Dim_ineq { cur = d1; subr = d2 } -> [ (to_proj d1, to_proj d2) ]
    | Row_eq { r1; r2 } | Row_ineq { cur = r1; subr = r2 } -> match_rows r1 r2
  in
  List.concat_map inequalities ~f

let solve_proj_equations eqs : proj_env =
  let v_env = dim_hashtbl () in
  let p_solved = ref [] in
  let p_dims = ref [] in
  let proj_classes = ref @@ Map.empty (module Int) in
  let rec f = function
    | (Proj { proj_id = p1; d = d1 } as proj1), (Proj { proj_id = p2; d = d2 } as proj2) ->
        if d1 <> d2 then
          raise
          @@ Shape_error
               ("Conflicting dimensions for the same projection", [ Projection_mismatch [ proj1; proj2 ] ]);
        p_dims := (p1, d1) :: !p_dims;
        proj_classes := Utils.union_add ~equal:Int.equal !proj_classes p1 p2
    | Proj p, Solved idx | Solved idx, Proj p -> p_solved := (p.proj_id, idx) :: !p_solved
    | Solved idx1, Solved idx2 when equal_axis_index idx1 idx2 -> ()
    | Solved idx1, Solved idx2 ->
        raise
        @@ Shape_error ("Conflicting indices for the same axis/projection", [ Index_mismatch [ idx1; idx2 ] ])
    | Var v, p | p, Var v -> (
        match Hashtbl.find v_env v with None -> Hashtbl.add_exn v_env ~key:v ~data:p | Some p2 -> f (p, p2))
  in
  List.iter eqs ~f;
  let projs = ref @@ Map.empty (module Int) and non_product = ref @@ Set.empty (module Int) in
  List.iter !p_solved ~f:(fun (p, idx) ->
      let repr, _ = Utils.union_find ~equal:Int.equal !proj_classes ~key:p ~rank:0 in
      non_product := Set.add !non_product repr;
      Utils.mref_add projs ~key:repr ~data:idx ~or_:(fun idx2 ->
          if not @@ equal_axis_index idx idx2 then
            raise
            @@ Shape_error
                 ( "Multiple constraints on the same projection",
                   [ Index_mismatch [ idx; Map.find_exn !projs p ] ] )));
  let product_dim = ref @@ Map.empty (module Int) in
  List.iter !p_dims ~f:(fun (p, d) ->
      let repr, _ = Utils.union_find ~equal:Int.equal !proj_classes ~key:p ~rank:0 in
      if iterated d && (not @@ Map.mem !projs repr) then
        Utils.mref_add product_dim ~key:repr ~data:d ~or_:(fun d2 ->
            if d <> d2 then
              raise
              @@ Shape_error
                   ( "Conflicting dimensions for the same projection",
                     [ Projection_mismatch [ Proj { proj_id = p; d }; Proj { proj_id = p; d = d2 } ] ] )));
  Map.iteri !product_dim ~f:(fun ~key:p ~data:_ ->
      let repr, _ = Utils.union_find ~equal:Int.equal !proj_classes ~key:p ~rank:0 in
      Utils.mref_add_missing projs repr ~f:(fun () -> Iterator (get_symbol ())));
  {
    proj_classes = !proj_classes;
    proj_to_index = !projs;
    product_dim = !product_dim;
    non_product = !non_product;
  }

let get_proj_index proj_env = function
  | Dim { d; _ } when not @@ iterated d -> Fixed_idx 0
  | Dim { proj_id = None; _ } -> assert false
  | Var _ as v ->
      raise @@ Shape_error ("projection_of_solved_dims: still not fully inferred", [ Dim_mismatch [ v ] ])
  | Dim { proj_id = Some proj_id; d; _ } -> (
      let repr, _ = Utils.union_find ~equal:Int.equal proj_env.proj_classes ~key:proj_id ~rank:0 in
      match Map.find proj_env.proj_to_index repr with
      | Some i -> i
      | None ->
          raise
          @@ Shape_error
               ( "projection_of_solved_dims: unknown projection",
                 [ Projection_mismatch [ Proj { proj_id; d } ] ] ))

let proj_repr proj_env p = fst @@ Utils.union_find ~equal:Int.equal proj_env.proj_classes ~key:p ~rank:0

let get_product_proj proj_env dim =
  match dim with
  | Dim { d; _ } when not @@ iterated d -> None
  | Dim { proj_id = Some proj_id; d; _ } ->
      let repr = proj_repr proj_env proj_id in
      if Map.mem proj_env.proj_to_index repr && (not @@ Set.mem proj_env.non_product repr) then Some (repr, d)
      else None
  | Dim { proj_id = None; _ } -> None
  | Var _ as dim ->
      raise @@ Shape_error ("derive_projections: shape still not fully inferred", [ Dim_mismatch [ dim ] ])

let proj_to_iterator proj_env p =
  match Map.find_exn proj_env.proj_to_index (proj_repr proj_env p) with Iterator s -> s | _ -> assert false
