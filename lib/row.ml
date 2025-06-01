(** The row type, shape inference related types and constraint solving. *)

open Base

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 9]
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL"]

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
type dim_var_set = Set.M(Dim_var).t [@@deriving equal, sexp]
type 'a dim_map = 'a Map.M(Dim_var).t [@@deriving equal, sexp]

let dim_var_set_empty = Set.empty (module Dim_var)
let dim_map_empty = Map.empty (module Dim_var)

type solved_dim = { d : int; label : string option; proj_id : int option }
[@@deriving equal, hash, compare, sexp]

type dim =
  | Var of dim_var
  | Dim of solved_dim
  | Prod of dim list
      (** A dimension that is the product of other dimensions, e.g. result of concatenation along
          one or more axes. *)
[@@deriving equal, hash, compare, sexp, variants]

let uid = ref 0

let get_var ?label () : dim_var =
  Int.incr uid;
  { id = !uid; label }

let get_dim ~d ?label () = Dim { d; label; proj_id = None }

type 'a dim_hashtbl = 'a Hashtbl.M(Dim_var).t [@@deriving sexp]

let dim_hashtbl () = Hashtbl.create (module Dim_var)

let rec dim_to_string style = function
  | Dim { label = None; _ } when phys_equal style `Only_labels -> "_"
  | Dim { label = Some l; _ } when phys_equal style `Only_labels -> l
  | Dim { d; label = None; _ } -> Int.to_string d
  | Dim { d; label = Some l; _ } -> [%string "%{l}=%{d#Int}"]
  | Var { id; label = Some l } -> [%string "$%{id#Int}:%{l}"]
  | Var { id; label = None } -> "$" ^ Int.to_string id
  | Prod ds -> String.concat ~sep:"&" (List.map ds ~f:(dim_to_string style))

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

type bcast = Row_var of { v : row_var; beg_dims : dim list } | Broadcastable
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
let phantom_row_id = row_id ~sh_id:(-1) ~kind:`Output
(* let row_map_empty = Map.empty (module Row_id) *)

type t = { dims : dim list; bcast : bcast; id : row_id } [@@deriving equal, hash, compare, sexp]
type row = t [@@deriving equal, sexp]

let dims_label_assoc dims =
  let f = function Var { label = Some l; _ } as d -> Some (l, d) | _ -> None in
  List.filter_map dims.dims ~f

type dim_constraint = Unconstrained_dim | At_least_dim of int
[@@deriving equal, hash, compare, sexp, variants]

type row_constraint =
  | Unconstrained
  | Total_elems of { nominator : int; divided_by : Set.M(Dim_var).t }
      (** The row or remainder of a row, inclusive of the further row spec, has this many elements.
      *)
[@@deriving equal, hash, compare, sexp, variants]

(** An entry implements inequalities [cur >= v >= subr] and/or an equality [v = solved]. [cur] and
    [subr] must be sorted using the [@@deriving compare] comparison. *)
type dim_entry =
  | Solved_dim of dim
  | Bounds_dim of {
      cur : dim_var list;
      subr : dim_var list;
      lub : dim option;
      constr : dim_constraint;
    }
[@@deriving sexp]

type row_entry =
  | Solved_row of t
  | Bounds_row of {
      cur : row_var list;
      subr : row_var list;
      lub : t option;
      constr : row_constraint;
    }
[@@deriving sexp]

type dim_env = dim_entry Map.M(Dim_var).t [@@deriving sexp]
type row_env = row_entry Map.M(Row_var).t [@@deriving sexp]

type environment = { dim_env : dim_env; row_env : row_env } [@@deriving sexp]
(** The environment is only in resolved wrt. variables that are solved: [v -> Solved ...] do not
    appear elsewhere in the environment. In particular, per-dim and per-row constraints might not
    have been applied. *)

type constraint_ =
  | Dim_eq of { d1 : dim; d2 : dim }
  | Row_eq of { r1 : t; r2 : t }
  | Dim_ineq of { cur : dim; subr : dim }
  | Row_ineq of { cur : t; subr : t }
  | Dim_constr of { d : dim; constr : dim_constraint }
  | Row_constr of { r : t; constr : row_constraint }
  | Terminal_dim of dim
  | Terminal_row of t
[@@deriving compare, equal, sexp, variants]

type stage = Stage1 | Stage2 | Stage3 | Stage4 | Stage5 | Stage6 | Stage7
[@@deriving sexp, equal, compare]

let is_stage2_up = function Stage1 -> false | _ -> true
let is_stage3_up = function Stage1 | Stage2 -> false | _ -> true
let is_stage4_up = function Stage1 | Stage2 | Stage3 -> false | _ -> true
let is_stage5_up = function Stage5 | Stage6 | Stage7 -> true | _ -> false
let is_stage6_up = function Stage6 | Stage7 -> true | _ -> false
let is_stage7 = function Stage7 -> true | _ -> false

module Idx = Ir.Indexing

type error_trace = ..

type error_trace +=
  | Row_mismatch of t list
  | Dim_mismatch of dim list
  | Index_mismatch of Idx.axis_index list

let sexp_of_error_trace = function
  | Row_mismatch rs -> Sexp.List (Sexp.Atom "Row_mismatch" :: List.map rs ~f:sexp_of_t)
  | Dim_mismatch ds -> Sexp.List (Sexp.Atom "Dim_mismatch" :: List.map ds ~f:sexp_of_dim)
  | Index_mismatch idcs ->
      Sexp.List (Sexp.Atom "Index_mismatch" :: List.map idcs ~f:Idx.sexp_of_axis_index)
  | _ -> Sexp.Atom "<outdated version of sexp_of_error_trace>"

exception Shape_error of string * error_trace list [@@deriving sexp_of]

type source = Direct | Equation | Cur | Subr [@@deriving equal, sexp]

let rec dim_to_int_exn = function
  | Dim { d; _ } -> d
  | Var _ -> invalid_arg "dim_to_int: dim still unknown"
  | Prod dims -> List.fold dims ~init:1 ~f:(fun acc d -> acc * dim_to_int_exn d)

(* Helper functions for Prod *)
let rec dim_vars = function
  | Var v -> [ v ]
  | Dim _ -> []
  | Prod dims -> List.concat_map dims ~f:dim_vars

let rec extract_dims_and_vars = function
  | Dim { d; _ } -> ([ d ], [])
  | Var v -> ([], [ v ])
  | Prod dims ->
      List.fold dims ~init:([], []) ~f:(fun (ds, vs) dim ->
          let d', v' = extract_dims_and_vars dim in
          (ds @ d', vs @ v'))

let rec is_solved_dim = function
  | Var _ -> false
  | Dim _ -> true
  | Prod dims -> List.for_all dims ~f:is_solved_dim

let s_dim_one v ~value ~in_ =
  match in_ with
  | Var v2 when equal_dim_var v v2 -> value
  | Prod dims -> (
      let rec flatten_prods = function
        | Var v2 when equal_dim_var v v2 -> flatten_prods value
        | Prod nested_dims -> List.concat_map nested_dims ~f:flatten_prods
        | d -> [ d ]
      in
      match List.concat_map dims ~f:flatten_prods with
      | [] -> get_dim ~d:1 ()
      | [ d ] -> d
      | dims -> Prod dims)
  | Dim _ | Var _ -> in_

(* For future flexibility *)
let dim_conjunction constr1 constr2 =
  match (constr1, constr2) with
  | Unconstrained_dim, _ -> Some ([], constr2)
  | _, Unconstrained_dim -> Some ([], constr1)
  | At_least_dim d1, At_least_dim d2 -> Some ([], At_least_dim (Int.max d1 d2))

let row_conjunction ?(id = phantom_row_id) constr1 constr2 =
  let elems_mismatch n1 n2 =
    raise @@ Shape_error ([%string "Total_elems constraint conflict: %{n1#Int} vs. %{n2#Int}"], [])
  in
  match (constr1, constr2) with
  | Unconstrained, _ -> Some ([], constr2)
  | _, Unconstrained -> Some ([], constr1)
  | ( Total_elems { nominator = n1; divided_by = vars1 },
      Total_elems { nominator = n2; divided_by = vars2 } )
    when [%equal: Set.M(Dim_var).t] vars1 vars2 ->
      if n1 <> n2 then elems_mismatch n1 n2 else Some ([], constr2)
  | ( Total_elems { nominator = n1; divided_by = vars1 },
      Total_elems { nominator = n2; divided_by = vars2 } ) ->
      let shared = Set.inter vars1 vars2 |> Set.to_list in
      let extras ~keep_constr1 =
        (* If we keep constr1, then it has fewer divided_by, i.e. n1 > n2. *)
        let nominator = if keep_constr1 then n1 / n2 else n2 / n1 in
        if nominator <= 0 then elems_mismatch n1 n2
        else if nominator = 1 then
          List.map shared ~f:(fun v -> Dim_eq { d1 = Var v; d2 = get_dim ~d:1 () })
        else if List.is_empty shared then []
        else
          let r = { dims = List.map shared ~f:(fun v -> Var v); bcast = Broadcastable; id } in
          [
            Row_constr
              { r; constr = Total_elems { nominator; divided_by = Set.empty (module Dim_var) } };
          ]
      in
      let subsum = Set.symmetric_diff vars1 vars2 in
      if Sequence.for_all ~f:Either.is_first subsum then Some (extras ~keep_constr1:false, constr2)
      else if Sequence.for_all ~f:Either.is_second subsum then
        Some (extras ~keep_constr1:true, constr1)
      else None

let rec apply_dim_constraint ~(source : source) ~(stage : stage) (dim : dim)
    (constr : dim_constraint) (env : environment) : constraint_ list * dim_constraint =
  let extras, constr =
    match (dim, constr) with
    | Dim { d; _ }, At_least_dim d_min ->
        if d < d_min then
          raise
          @@ Shape_error
               ( "At_least_dim constraint failed, expected " ^ Int.to_string d_min,
                 [ Dim_mismatch [ dim ] ] )
        else ([], constr)
    | Prod dims, At_least_dim d_min -> (
        (* For a product, we check if the product of all known dimensions meets the constraint *)
        let product = ref 1 in
        let vars = ref [] in
        let rec f = function
          | Dim { d; _ } -> product := !product * d
          | Var v -> vars := v :: !vars
          | Prod dims -> List.iter dims ~f
        in
        List.iter dims ~f;
        match !vars with
        | [] ->
            if !product < d_min then
              raise
              @@ Shape_error
                   ( "At_least_dim constraint failed for product, expected at least "
                     ^ Int.to_string d_min,
                     [ Dim_mismatch [ dim ] ] )
            else ([], constr)
        | [ v ] -> apply_dim_constraint ~source ~stage (Var v) (At_least_dim (d_min / !product)) env
        | _ -> ([], constr))
    | Var v, _ -> (
        match Map.find env.dim_env v with
        | None -> ([], constr)
        | Some (Solved_dim _) -> assert false
        | Some (Bounds_dim bounds) -> (
            match (source, constr) with
            (* If source is [Cur], then [constr] (target) is [Subr]. *)
            | Cur, (Unconstrained_dim | At_least_dim 1) -> ([], constr)
            | _ -> Option.value ~default:([], constr) @@ dim_conjunction constr bounds.constr))
    | _, Unconstrained_dim -> ([], constr)
  in
  match (dim, constr, stage) with
  | Var _, At_least_dim d, Stage4 ->
      (Dim_eq { d1 = dim; d2 = get_dim ~d () } :: extras, Unconstrained_dim)
  | _ -> (extras, constr)

let reduce_row_constraint (constr : row_constraint) ~(beg_dims : dim list) ~(dims : dim list) :
    row_constraint =
  match constr with
  | Total_elems { nominator; divided_by } ->
      let ds, vars =
        List.fold (beg_dims @ dims) ~init:([], []) ~f:(fun (ds, vs) dim ->
            let d', v' = extract_dims_and_vars dim in
            (ds @ d', vs @ v'))
      in
      let vars = Set.of_list (module Dim_var) vars in
      if not @@ Set.(is_empty @@ inter vars divided_by) then Unconstrained
      else
        let d : int = List.fold ds ~init:1 ~f:( * ) in
        let nominator : int = nominator / d in
        if nominator = 0 then
          raise
          @@ Shape_error
               ( "reduce_row_constraint: Total_elems constraint failed, shape is too big",
                 [ Dim_mismatch (beg_dims @ dims) ] )
        else if d = 1 && Set.is_empty vars then constr
        else Total_elems { nominator; divided_by = Utils.Set_O.(divided_by + vars) }
  | Unconstrained -> Unconstrained

(* Inverts what [reduce_row_constraint] would do. *)
let _lift_row_constraint (constr : row_constraint) ~(beg_dims : dim list) ~(dims : dim list) :
    row_constraint =
  match constr with
  | Total_elems { nominator; divided_by } ->
      let ds, vars =
        List.fold (beg_dims @ dims) ~init:([], []) ~f:(fun (ds, vs) dim ->
            let d', v' = extract_dims_and_vars dim in
            (ds @ d', vs @ v'))
      in
      let vars = Set.of_list (module Dim_var) vars in
      if not @@ Set.is_subset vars ~of_:divided_by then Unconstrained
      else
        let d = List.fold ds ~init:1 ~f:( * ) in
        if d = 1 && Set.is_empty vars then constr
        else Total_elems { nominator = nominator * d; divided_by = Utils.Set_O.(divided_by - vars) }
  | Unconstrained -> Unconstrained

let apply_row_constraint ~stage:_ (r : row) (constr : row_constraint) env : constraint_ list * _ =
  if is_unconstrained constr then ([], env)
  else
    let reduce constr ~beg_dims ~dims =
      try reduce_row_constraint constr ~beg_dims ~dims
      with Shape_error (s, trace) -> raise @@ Shape_error (s, Row_mismatch [ r ] :: trace)
    in
    let extras, constr, env, stored, updated =
      match r with
      | { bcast = Broadcastable; _ } -> ([], constr, env, false, false)
      | { bcast = Row_var { v; beg_dims }; dims; _ } -> (
          match Map.find env.row_env v with
          | Some (Solved_row _) -> ([], constr, env, false, false)
          | None ->
              ( [],
                constr,
                {
                  env with
                  row_env =
                    Map.set env.row_env ~key:v
                      ~data:
                        (Bounds_row
                           {
                             constr = reduce constr ~beg_dims ~dims;
                             cur = [];
                             subr = [];
                             lub = None;
                           });
                },
                true,
                false )
          | Some (Bounds_row ({ constr = Unconstrained; _ } as bounds)) ->
              ( [],
                constr,
                {
                  env with
                  row_env =
                    Map.set env.row_env ~key:v
                      ~data:(Bounds_row { bounds with constr = reduce constr ~beg_dims ~dims });
                },
                true,
                false )
          | Some (Bounds_row bounds) -> (
              match row_conjunction ~id:r.id (reduce constr ~beg_dims ~dims) bounds.constr with
              | None -> ([], constr, env, false, false)
              | Some (extras, constr) ->
                  if phys_equal constr bounds.constr then (extras, constr, env, true, false)
                  else
                    ( extras,
                      constr,
                      {
                        env with
                        row_env =
                          Map.set env.row_env ~key:v ~data:(Bounds_row { bounds with constr });
                      },
                      true,
                      true )))
    in
    match (r, constr) with
    | _ when stored && not updated -> (extras, env)
    | _, Unconstrained -> assert false
    | { dims; bcast = Broadcastable; _ }, Total_elems { nominator; divided_by }
      when Set.length divided_by <= 1 -> (
        let ds, vars =
          List.fold dims ~init:([], []) ~f:(fun (ds, vs) dim ->
              let d', v' = extract_dims_and_vars dim in
              (ds @ d', vs @ v'))
        in
        let d : int = List.fold ds ~init:1 ~f:( * ) in
        let nominator : int = nominator / d in
        if nominator = 0 then
          raise
          @@ Shape_error
               ( "apply_row_constraint: Total_elems constraint failed, shape is too big",
                 [ Dim_mismatch dims ] );
        match (vars, Set.elements divided_by) with
        | [], [] ->
            if nominator = 1 then (extras, env)
            else
              raise
              @@ Shape_error
                   ( "apply_row_constraint: Total_elems constraint failed, shape is too small",
                     [ Row_mismatch [ r ] ] )
        | [ v ], [] | [], [ v ] ->
            (Dim_eq { d1 = Var v; d2 = get_dim ~d:nominator () } :: extras, env)
        | vs1, vs2 when nominator = 1 ->
            ( List.map ~f:(fun v -> Dim_eq { d1 = Var v; d2 = get_dim ~d:1 () }) (vs1 @ vs2) @ extras,
              env )
        (* TODO: we can work harder making assumptions here if necessary... *)
        (* | v :: _, [] | [], v :: _ when (is_stage4_up stage) -> (Dim_eq { d1 = Var v; d2 = get_dim
           ~d:nominator () } :: extras, env) *)
        | _ :: _, _ when stored -> (extras, env)
        | _, _ -> (Row_constr { r; constr } :: extras, env (* Wait for more shape inference. *)))
    | { bcast = Row_var _; _ }, _ | _, Total_elems { nominator = _; divided_by = _ } ->
        if stored then (extras, env)
        else (Row_constr { r; constr } :: extras, env (* Wait for more shape inference. *))

let s_dim_one_in_entry v ~value (in_ : dim_entry) : _ * dim_entry =
  match in_ with
  | Solved_dim in_ -> ([], Solved_dim (s_dim_one v ~value ~in_))
  | Bounds_dim { cur; subr; lub; constr } ->
      let find_v side = List.partition_tf side ~f:(equal_dim_var v) in
      let cur_v, cur = find_v cur in
      let subr_v, subr = find_v subr in
      let ineqs0 =
        match (subr_v, lub) with
        | _ :: _, Some lub -> [ Dim_ineq { cur = lub; subr = value } ]
        | _ -> []
      in
      let ineqs1 =
        if List.is_empty subr_v then []
        else List.map cur ~f:(fun cur -> Dim_ineq { cur = Var cur; subr = value })
      in
      let ineqs2 =
        if List.is_empty cur_v then []
        else List.map subr ~f:(fun subr -> Dim_ineq { subr = Var subr; cur = value })
      in
      ( ineqs0 @ ineqs1 @ ineqs2,
        Bounds_dim
          { cur; subr; lub = Option.map lub ~f:(fun in_ -> s_dim_one v ~value ~in_); constr } )

let s_dim_one_in_row v ~value in_ =
  { in_ with dims = List.map in_.dims ~f:(fun in_ -> s_dim_one v ~value ~in_) }

let s_dim_one_in_row_constr v ~value constr =
  match constr with
  | Total_elems { nominator; divided_by } when Set.mem divided_by v -> (
      let divided_by = Set.remove divided_by v in
      match value with
      | Var v' -> Total_elems { nominator; divided_by = Set.(add divided_by v') }
      | Dim { d; _ } ->
          let nominator = nominator / d in
          if nominator <= 0 then
            raise
            @@ Shape_error
                 ( "s_dim_one_in_row_constr: Total_elems constraint failed: shape is too big",
                   [ Dim_mismatch [ value ] ] )
          else Total_elems { nominator; divided_by }
      | Prod _ ->
          (* When substituting with a Prod, we need to calculate its total dimension and extract
             variables *)
          let ds, vars = extract_dims_and_vars value in
          let d = List.fold ds ~init:1 ~f:( * ) in
          let nominator = nominator / d in
          if nominator <= 0 then
            raise
            @@ Shape_error
                 ( "s_dim_one_in_row_constr: Total_elems constraint failed: shape is too big",
                   [ Dim_mismatch [ value ] ] )
          else
            (* Add any variables from the Prod to divided_by *)
            Total_elems
              { nominator; divided_by = Set.union divided_by (Set.of_list (module Dim_var) vars) })
  | _ -> constr

let s_dim_one_in_row_entry v ~value in_ =
  match in_ with
  | Solved_row in_ -> Solved_row (s_dim_one_in_row v ~value in_)
  | Bounds_row { cur; subr; lub; constr } ->
      let constr = s_dim_one_in_row_constr v ~value constr in
      Bounds_row { cur; subr; lub = Option.map lub ~f:(s_dim_one_in_row v ~value); constr }

let rec subst_dim env = function
  | Dim _ as d -> d
  | Var v as default -> (
      match Map.find env.dim_env v with
      | Some (Solved_dim (Var v2)) when equal_dim_var v v2 -> default
      | Some (Solved_dim d) -> subst_dim env d
      | _ -> default)
  | Prod dims -> (
      let rec f dim = match subst_dim env dim with Prod dims -> dims | dim -> [ dim ] in
      match List.concat_map dims ~f with
      | [] -> get_dim ~d:1 ()
      | [ dim ] -> dim
      | dims -> Prod dims)

let s_row_one v ~value:{ dims = more_dims; bcast; id = _ } ~in_ =
  match in_ with
  | { dims; bcast = Row_var { v = v2; beg_dims }; id } when equal_row_var v v2 -> (
      match bcast with
      | Broadcastable -> { dims = beg_dims @ more_dims @ dims; bcast; id }
      | Row_var { v = v3; beg_dims = more_beg_dims } ->
          {
            dims = more_dims @ dims;
            bcast = Row_var { v = v3; beg_dims = beg_dims @ more_beg_dims };
            id;
          })
  | _ -> in_

let s_row_one_in_row_constr _v ~value:_ ~in_ = match in_ with Unconstrained | Total_elems _ -> in_
let row_of_var v id = { dims = []; bcast = Row_var { v; beg_dims = [] }; id }

let s_row_one_in_entry (v : row_var) ~(value : row) ~(in_ : row_entry) :
    constraint_ list * row_entry =
  match in_ with
  | Solved_row in_ -> ([], Solved_row (s_row_one v ~value ~in_))
  | Bounds_row { cur; subr; lub; constr } ->
      (* TODO: audit code to ensure we don't lose the constraints associated with the bounds
         variables. *)
      let find_v side = List.partition_tf side ~f:(equal_row_var v) in
      let cur_v, cur = find_v cur in
      let subr_v, subr = find_v subr in
      let ineqs0 =
        match (subr_v, lub) with
        | _ :: _, Some lub -> [ Row_ineq { cur = lub; subr = value } ]
        | _ -> []
      in
      let ineqs1 =
        if List.is_empty subr_v then []
        else List.map cur ~f:(fun cur -> Row_ineq { cur = row_of_var cur value.id; subr = value })
      in
      let ineqs2 =
        if List.is_empty cur_v then []
        else
          List.map subr ~f:(fun subr -> Row_ineq { subr = row_of_var subr value.id; cur = value })
      in
      let constr = s_row_one_in_row_constr v ~value ~in_:constr in
      ( ineqs0 @ ineqs1 @ ineqs2,
        Bounds_row
          { cur; subr; lub = Option.map lub ~f:(fun in_ -> s_row_one v ~value ~in_); constr } )

let%debug6_sexp subst_row (env : environment) ({ dims; bcast; id } : t) : t =
  let s_dims = List.map ~f:(subst_dim env) in
  let dims = s_dims dims in
  let bcast =
    match bcast with
    | Row_var { v; beg_dims } -> Row_var { v; beg_dims = s_dims beg_dims }
    | Broadcastable -> Broadcastable
  in
  let default = { dims; bcast; id } in
  match bcast with
  | Broadcastable -> default
  | Row_var { v; beg_dims } -> (
      match Map.find env.row_env v with
      | None | Some (Bounds_row _) -> default
      | Some (Solved_row { dims = []; bcast = Row_var { v = v2; beg_dims = [] }; _ })
        when equal_row_var v v2 ->
          default
      | Some (Solved_row ({ bcast = Row_var { v = v2; _ }; _ } as r2)) when equal_row_var v v2 ->
          raise
          @@ Shape_error
               ("Infinite number of axes by self-reference", [ Row_mismatch [ default; r2 ] ])
      | Some (Solved_row { dims = more_dims; bcast; id = _ }) -> (
          (* Note: we assume env is idempotent (solved wrt. equalities). *)
          match bcast with
          | Broadcastable ->
              { dims = beg_dims @ s_dims more_dims @ dims; bcast = Broadcastable; id }
          | Row_var { v = v2; beg_dims = more_beg_dims } ->
              {
                dims = s_dims more_dims @ dims;
                bcast = Row_var { v = v2; beg_dims = beg_dims @ more_beg_dims };
                id;
              }))

let%debug5_sexp rec unify_dim ~stage (eq : dim * dim) (env : environment) :
    constraint_ list * environment =
  let dim1 : dim = subst_dim env @@ fst eq and dim2 : dim = subst_dim env @@ snd eq in
  match (dim1, dim2) with
  | Dim { label = Some l1; _ }, Dim { label = Some l2; _ } when not (String.equal l1 l2) ->
      raise
      @@ Shape_error
           ("solved dimensions for axis: different labels", [ Dim_mismatch [ dim1; dim2 ] ])
  | Dim { d = d1; _ }, Dim { d = d2; _ } when d1 = d2 -> ([], env)
  | Var v1, Var v2 when equal_dim_var v1 v2 -> ([], env)
  | Var v, dim2 | dim2, Var v ->
      let ineqs = ref [] in
      let f in_ =
        let more_ineqs, result = s_dim_one_in_entry v ~value:dim2 in_ in
        ineqs := more_ineqs @ !ineqs;
        result
      in
      let env =
        match Map.find env.dim_env v with
        | None ->
            let dim_env = Map.map env.dim_env ~f in
            {
              dim_env = Map.add_exn dim_env ~key:v ~data:(Solved_dim dim2);
              row_env = Map.map env.row_env ~f:(s_dim_one_in_row_entry v ~value:dim2);
            }
        | Some (Solved_dim _) -> assert false
        | Some (Bounds_dim { cur; subr; lub; constr }) ->
            let dim_env = Map.map env.dim_env ~f in
            List.iter cur ~f:(fun cur -> ineqs := Dim_ineq { cur = Var cur; subr = dim2 } :: !ineqs);
            List.iter subr ~f:(fun subr ->
                ineqs := Dim_ineq { subr = Var subr; cur = dim2 } :: !ineqs);
            Option.iter lub ~f:(fun lub -> ineqs := Dim_ineq { cur = lub; subr = dim2 } :: !ineqs);
            let extras, constr = apply_dim_constraint ~source:Equation ~stage dim2 constr env in
            let extras =
              if is_unconstrained_dim constr then extras
              else Dim_constr { d = dim2; constr } :: extras
            in
            ineqs := extras @ !ineqs;
            {
              dim_env = Map.set dim_env ~key:v ~data:(Solved_dim dim2);
              row_env = Map.map env.row_env ~f:(s_dim_one_in_row_entry v ~value:dim2);
            }
      in
      let dim_eqs, ineqs =
        List.partition_map !ineqs ~f:(function
          | Dim_eq { d1; d2 } -> Either.First (d1, d2)
          | ineq -> Either.Second ineq)
      in
      let f (ineqs, env) ds =
        let more_ineqs, env = unify_dim ~stage ds env in
        (more_ineqs @ ineqs, env)
      in
      List.fold ~init:(ineqs, env) dim_eqs ~f
  | dim1, dim2 ->
      (* Note: at the unify_dim phase, it's strict equality (no broadcasting). *)
      raise @@ Shape_error ("solved dimensions for axis: mismatch", [ Dim_mismatch [ dim1; dim2 ] ])

let drop_from_end l n = List.rev @@ List.drop (List.rev l) n
let take_from_end (l : dim list) (n : int) : dim list = List.rev @@ List.take (List.rev l) n

(* Equate two rows, no broadcasting. Does not resolve inequalities. *)
let%debug5_sexp rec unify_row ~stage (eq : t * t) (env : environment) :
    constraint_ list * environment =
  let rec solve ((ineqs : constraint_ list), env) : constraint_ -> constraint_ list * environment =
    function
    | Dim_eq { d1; d2 } ->
        let more_ineqs, env = unify_dim ~stage (d1, d2) env in
        List.fold ~init:(ineqs, env) more_ineqs ~f:solve
    | Row_eq { r1; r2 } ->
        let more_ineqs, env = unify_row ~stage (r1, r2) env in
        (more_ineqs @ ineqs, env)
    | (Dim_ineq _ | Row_ineq _ | Dim_constr _ | Row_constr _ | Terminal_dim _ | Terminal_row _) as
      ineq ->
        (ineq :: ineqs, env)
  in
  let unify_suffix init dims1 dims2 len =
    let dims1 = take_from_end dims1 len and dims2 = take_from_end dims2 len in
    List.fold ~init ~f:(fun acc (d1, d2) -> solve acc (Dim_eq { d1; d2 }))
    @@ List.zip_exn dims1 dims2
  in
  let r1 : t = subst_row env @@ fst eq and r2 : t = subst_row env @@ snd eq in
  let l = List.length in
  match (r1, r2) with
  | r1, r2 when equal_row r1 r2 -> ([], env)
  | ( { bcast = Row_var { v = v1; beg_dims = beg_dims1 }; dims = dims1; id = _ },
      { bcast = Row_var { v = v2; beg_dims = beg_dims2 }; dims = dims2; id = _ } )
    when equal_row_var v1 v2 ->
      let dims1_l = l dims1
      and dims2_l = l dims2
      and beg_dims1_l = l beg_dims1
      and beg_dims2_l = l beg_dims2 in
      if beg_dims1_l + dims1_l <> beg_dims2_l + dims2_l then
        raise
        @@ Shape_error ("Infinite number of axes by self-reference", [ Row_mismatch [ r1; r2 ] ]);
      let result = unify_suffix ([], env) dims1 dims2 @@ min dims1_l dims2_l in
      unify_suffix result (List.rev beg_dims1) (List.rev beg_dims2) @@ min beg_dims1_l beg_dims2_l
  | ({ bcast = Row_var { v; beg_dims = beg_dims1 }; dims = dims1; id } as r1), r2
  | r2, ({ bcast = Row_var { v; beg_dims = beg_dims1 }; dims = dims1; id } as r1) -> (
      let dims1_l : int = l dims1
      and dims2_l : int = l r2.dims
      and beg_dims1_l : int = l beg_dims1 in
      let beg_dims2_l : int =
        match r2.bcast with Row_var { beg_dims; _ } -> l beg_dims | Broadcastable -> 0
      in
      let beg_dims_l = min beg_dims1_l beg_dims2_l in
      if dims1_l > dims2_l || (dims1_l = dims2_l && beg_dims1_l > beg_dims2_l) then
        if is_row_var r2.bcast then unify_row ~stage (r2, r1) env
        else raise @@ Shape_error ("Number of axes mismatch", [ Row_mismatch [ r1; r2 ] ])
      else
        let orig_rows = [ r1; r2 ] in
        let (beg_handled : bool), (ineqs, env), (value : row) =
          match r2.bcast with
          | Row_var { v = v2; beg_dims = beg_dims2 } ->
              let result =
                try unify_suffix ([], env) dims1 r2.dims dims1_l
                with Shape_error (s, trace) ->
                  raise @@ Shape_error (s, Row_mismatch orig_rows :: trace)
              in
              let dims = drop_from_end r2.dims dims1_l in
              if equal_row_var v v2 then
                if List.is_empty dims && l beg_dims2 = l beg_dims1 then
                  let bcast = Row_var { v; beg_dims = [] } in
                  let value : row = { bcast; dims; id } in
                  ( true,
                    unify_suffix result (List.rev beg_dims1) (List.rev beg_dims2) @@ l beg_dims2,
                    value )
                else
                  raise
                  @@ Shape_error
                       ("Infinite number of axes by self-reference", [ Row_mismatch orig_rows ])
              else
                let result =
                  unify_suffix result (List.rev beg_dims1) (List.rev beg_dims2) beg_dims_l
                in
                let bcast = Row_var { v = v2; beg_dims = List.drop beg_dims2 beg_dims_l } in
                let value : row = { bcast; dims; id } in
                (beg_dims_l = l beg_dims1, result, value)
          | Broadcastable ->
              if dims1_l + beg_dims1_l > dims2_l then
                raise @@ Shape_error ("Number of axes mismatch", [ Row_mismatch [ r1; r2 ] ])
              else
                let dims = List.drop r2.dims beg_dims1_l |> Fn.flip drop_from_end dims1_l in
                let result =
                  List.zip_exn beg_dims1 (List.take r2.dims beg_dims1_l)
                  @ List.zip_exn dims1 (take_from_end r2.dims dims1_l)
                  |> List.fold ~init:([], env) ~f:(fun acc (d1, d2) ->
                         solve acc (Dim_eq { d1; d2 }))
                in
                let value : row = { bcast = Broadcastable; dims; id } in
                (true, result, value)
        in
        (* From now on, we have no use for un-reduced r2 since we deal with the row variable. *)
        let r2 : row = value in
        let ineqs : constraint_ list ref = ref ineqs in
        let f in_ =
          let more_ineqs, result = s_row_one_in_entry v ~value:(value : row) ~in_ in
          ineqs := more_ineqs @ !ineqs;
          result
        in
        let result env =
          let row_env = Map.map env.row_env ~f in
          let unsolved, env =
            if beg_handled then
              ([], { env with row_env = Map.set row_env ~key:v ~data:(Solved_row value) })
            else
              ( [
                  Row_eq
                    {
                      r1 =
                        {
                          dims = [];
                          bcast = Row_var { v; beg_dims = List.drop beg_dims1 beg_dims_l };
                          id;
                        };
                      r2;
                    };
                ],
                env )
          in
          List.fold ~init:(unsolved, env) ~f:solve !ineqs
        in
        match Map.find env.row_env v with
        | None -> result env
        | Some (Solved_row _) -> assert false
        | Some (Bounds_row { cur; subr; lub; constr }) ->
            let env =
              if beg_handled then (
                List.iter cur ~f:(fun cur ->
                    ineqs := Row_ineq { cur = row_of_var cur value.id; subr = r2 } :: !ineqs);
                List.iter subr ~f:(fun subr ->
                    ineqs := Row_ineq { subr = row_of_var subr value.id; cur = r2 } :: !ineqs);
                Option.iter lub ~f:(fun lub -> ineqs := Row_ineq { cur = lub; subr = r2 } :: !ineqs);
                let extras, env = apply_row_constraint ~stage value constr env in
                ineqs := extras @ !ineqs;
                env)
              else env
            in
            let _bound_elim_ineqs : constraint_ list = !ineqs in
            result env)
  | ( ({ bcast = Broadcastable; dims = dims1; id = _ } as r1),
      ({ bcast = Broadcastable; dims = dims2; id = _ } as r2) ) -> (
      match List.zip dims1 dims2 with
      | Unequal_lengths ->
          raise @@ Shape_error ("Mismatching number of axes", [ Row_mismatch [ r1; r2 ] ])
      | Ok eqs ->
          List.fold ~init:([], env) ~f:(fun acc (d1, d2) -> solve acc (Dim_eq { d1; d2 })) eqs)

let%debug5_sexp solve_dim_ineq ~(stage : stage) ~(cur : dim) ~(subr : dim) (env : environment) :
    constraint_ list * environment =
  let nonredundant ?(more = []) v vs =
    Utils.sorted_diff ~compare:compare_dim_var
      (List.dedup_and_sort ~compare:compare_dim_var (v :: vs))
      more
  in
  let rec cyclic ~subr_v ~curs =
    (* TODO: it's somewhat inefficient *)
    List.exists curs ~f:(fun cur_v ->
        equal_dim_var subr_v cur_v
        ||
        match Map.find env.dim_env cur_v with
        | None | Some (Solved_dim (Dim _)) -> false
        | Some (Solved_dim (Var v)) | Some (Solved_dim (Prod [ Var v ])) -> equal_dim_var subr_v v
        | Some (Solved_dim (Prod _)) -> false (* Prod doesn't contain variables directly *)
        | Some (Bounds_dim { cur = curs; _ }) -> cyclic ~subr_v ~curs)
  in
  match (cur, subr) with
  | cur, subr when equal_dim cur subr -> ([], env)
  | Dim { label = Some l1; _ }, Dim { label = Some l2; _ } when not (String.equal l1 l2) ->
      raise
      @@ Shape_error
           ("dimension comparison for axis: different labels", [ Dim_mismatch [ cur; subr ] ])
  | Dim { d = d1; _ }, Dim { d = d2; _ } when d1 = d2 -> ([], env)
  | _, Dim { d = 1; _ } -> ([], env)
  | (Dim { d = 1; _ } as cur), _ -> ([ Dim_eq { d1 = subr; d2 = cur } ], env)
  | Prod ds1, Prod ds2 when List.equal equal_dim ds1 ds2 -> ([], env)
  | Prod ds1, Prod ds2 ->
      (* Two different products - need to handle inequality between them *)
      if is_solved_dim (Prod ds1) && is_solved_dim (Prod ds2) then
        let d1 = dim_to_int_exn (Prod ds1) in
        let d2 = dim_to_int_exn (Prod ds2) in
        let expanded1 = get_dim ~d:d1 () in
        let expanded2 = get_dim ~d:d2 () in
        ([ Dim_ineq { cur = expanded1; subr = expanded2 } ], env)
      else
        raise
        @@ Shape_error
             ( "Cannot compare Prod dimensions with unresolved variables",
               [ Dim_mismatch [ cur; subr ] ] )
  | Prod ds, Dim _ | Dim _, Prod ds ->
      (* For now, we can't directly compare a Prod with a Dim in an inequality. We could potentially
         expand this to handle cases where the product is known. *)
      if is_solved_dim (Prod ds) then
        let prod_val = dim_to_int_exn (Prod ds) in
        let expanded = get_dim ~d:prod_val () in
        match (cur, subr) with
        | Prod _, _ -> ([ Dim_ineq { cur = expanded; subr } ], env)
        | _, Prod _ -> ([ Dim_ineq { cur; subr = expanded } ], env)
        | _ -> assert false
      else
        raise
        @@ Shape_error
             ( "Cannot compare Prod with unresolved variables in inequality",
               [ Dim_mismatch [ cur; subr ] ] )
  | Prod _, Var _ | Var _, Prod _ ->
      (* Similar to above - we need all dimensions resolved to compare *)
      raise
      @@ Shape_error
           ("Cannot compare Prod with variables in inequality", [ Dim_mismatch [ cur; subr ] ])
  | Var cur_v, Var subr_v -> (
      match (Map.find env.dim_env cur_v, Map.find env.dim_env subr_v) with
      | Some (Bounds_dim { cur = cur1; _ }), _ when List.mem ~equal:equal_dim_var cur1 subr_v ->
          ([ Dim_eq { d1 = cur; d2 = subr } ], env)
      | _, Some (Bounds_dim { subr = subr2; _ }) when List.mem ~equal:equal_dim_var subr2 cur_v ->
          ([ Dim_eq { d1 = cur; d2 = subr } ], env)
      | None, None ->
          ( [],
            {
              env with
              dim_env =
                env.dim_env
                |> Map.add_exn ~key:cur_v
                     ~data:
                       (Bounds_dim
                          { lub = None; cur = []; subr = [ subr_v ]; constr = Unconstrained_dim })
                |> Map.add_exn ~key:subr_v
                     ~data:
                       (Bounds_dim
                          { lub = None; cur = [ cur_v ]; subr = []; constr = Unconstrained_dim });
            } )
      | Some (Solved_dim _), _ | _, Some (Solved_dim _) -> assert false
      | Some (Bounds_dim { cur = cur1; subr = subr1; lub = lub1; constr = constr1 }), None ->
          let from_lub = Option.to_list lub1 |> List.map ~f:(fun cur -> Dim_ineq { cur; subr }) in
          let from_constr1, constr1 = apply_dim_constraint ~source:Subr ~stage subr constr1 env in
          let from_constr2, constr2 =
            apply_dim_constraint ~source:Cur ~stage cur Unconstrained_dim env
          in
          ( from_constr1 @ from_constr2 @ from_lub,
            {
              env with
              dim_env =
                env.dim_env
                |> Map.set ~key:cur_v
                     ~data:
                       (Bounds_dim
                          {
                            lub = lub1;
                            cur = cur1;
                            subr = nonredundant subr_v subr1;
                            constr = constr1;
                          })
                |> Map.add_exn ~key:subr_v
                     ~data:(Bounds_dim { lub = None; cur = [ cur_v ]; subr = []; constr = constr2 });
            } )
      | ( Some (Bounds_dim { cur = _; subr = [ subr1 ]; lub = None; constr = _ }),
          Some (Bounds_dim { cur = [ cur2 ]; subr = _; lub = None; constr = _ }) )
        when is_stage2_up stage && equal_dim_var subr_v subr1 && equal_dim_var cur_v cur2 ->
          (* A heuristic to reduce template variables coming from e.g. einsum notation expansion. *)
          ([ Dim_eq { d1 = subr; d2 = cur } ], env)
      | Some (Bounds_dim { cur = curs; subr = _; lub = _; constr = _ }), Some (Bounds_dim _)
        when cyclic ~subr_v ~curs ->
          ([ Dim_eq { d1 = subr; d2 = cur } ], env)
      | None, Some (Bounds_dim { cur = cur2; subr = subr2; lub = lub2; constr = constr2 }) ->
          let from_constr1, constr1 =
            apply_dim_constraint ~source:Subr ~stage subr Unconstrained_dim env
          in
          let from_constr2, constr2 = apply_dim_constraint ~source:Cur ~stage cur constr2 env in
          ( from_constr2 @ from_constr1,
            {
              env with
              dim_env =
                env.dim_env
                |> Map.add_exn ~key:cur_v
                     ~data:
                       (Bounds_dim { lub = None; cur = []; subr = [ subr_v ]; constr = constr1 })
                |> Map.set ~key:subr_v
                     ~data:
                       (Bounds_dim
                          {
                            lub = lub2;
                            cur = nonredundant cur_v cur2;
                            subr = subr2;
                            constr = constr2;
                          });
            } )
      | ( Some (Bounds_dim { cur = cur1; subr = subr1; lub = lub1; constr = constr1 }),
          Some (Bounds_dim { cur = cur2; subr = subr2; lub = lub2; constr = constr2 }) ) ->
          let from_lub = Option.to_list lub1 |> List.map ~f:(fun cur -> Dim_ineq { cur; subr }) in
          let from_constr1, constr1 = apply_dim_constraint ~source:Subr ~stage subr constr1 env in
          let from_constr2, constr2 = apply_dim_constraint ~source:Cur ~stage cur constr2 env in
          ( from_constr1 @ from_constr2 @ from_lub,
            {
              env with
              dim_env =
                env.dim_env
                |> Map.set ~key:cur_v
                     ~data:
                       (Bounds_dim
                          {
                            lub = lub1;
                            cur = cur1;
                            subr = nonredundant ~more:subr2 subr_v subr1;
                            constr = constr1;
                          })
                |> Map.set ~key:subr_v
                     ~data:
                       (Bounds_dim
                          {
                            lub = lub2;
                            cur = nonredundant ~more:cur1 cur_v cur2;
                            subr = subr2;
                            constr = constr2;
                          });
            } ))
  | _, Var subr_v -> (
      match Map.find env.dim_env subr_v with
      | None ->
          ( [],
            {
              env with
              dim_env =
                Map.add_exn env.dim_env ~key:subr_v
                  ~data:
                    (Bounds_dim { lub = Some cur; cur = []; subr = []; constr = Unconstrained_dim });
            } )
      | Some (Solved_dim _) -> assert false
      | Some (Bounds_dim { cur = cur2; subr = subr2; lub = Some lub2; constr = constr2 }) ->
          let lub, lub_forcing =
            match (cur, lub2) with
            | Dim { d = d1; _ }, Dim { d = d2; _ } when d1 = d2 -> (cur, [])
            | Dim _, Dim _ (* when d1 <> d2 *) ->
                let lub = get_dim ~d:1 () in
                (lub, [ Dim_eq { d1 = subr; d2 = lub } ])
                (* raise @@ Shape_error ( "dimension comparison for axis: upper bound mismatch", [
                   Dim_mismatch [ lub2; cur; subr ] ] ) *)
            | Prod _, Dim _ | Dim _, Prod _ ->
                (* For LUB between Prod and Dim, we need to expand if possible *)
                if is_solved_dim cur && is_solved_dim lub2 then
                  let d_cur = dim_to_int_exn cur in
                  let d_lub2 = dim_to_int_exn lub2 in
                  if d_cur = d_lub2 then (cur, [])
                  else
                    let lub = get_dim ~d:1 () in
                    (lub, [ Dim_eq { d1 = subr; d2 = lub } ])
                else
                  raise
                  @@ Shape_error
                       ( "Cannot compute LUB between Prod and unsolved dimensions",
                         [ Dim_mismatch [ cur; lub2 ] ] )
            | Prod _, Prod _ ->
                (* For LUB between two Prods, they need to match structurally or we force to
                   dim-1 *)
                if equal_dim cur lub2 then (cur, [])
                else if is_solved_dim cur && is_solved_dim lub2 then
                  let d_cur = dim_to_int_exn cur in
                  let d_lub2 = dim_to_int_exn lub2 in
                  if d_cur = d_lub2 then (cur, [])
                  else
                    let lub = get_dim ~d:1 () in
                    (lub, [ Dim_eq { d1 = subr; d2 = lub } ])
                else
                  raise
                  @@ Shape_error
                       ( "Cannot compute LUB between different Prod structures",
                         [ Dim_mismatch [ cur; lub2 ] ] )
            | Var _, _ | _, Var _ -> assert false
          in
          let from_constr, constr2 = apply_dim_constraint ~source:Cur ~stage cur constr2 env in
          ( from_constr @ lub_forcing,
            {
              env with
              dim_env =
                Map.set env.dim_env ~key:subr_v
                  ~data:(Bounds_dim { lub = Some lub; cur = cur2; subr = subr2; constr = constr2 });
            } )
      | Some (Bounds_dim { cur = cur2; subr = subr2; lub = None; constr = constr2 }) ->
          let from_constr, constr2 = apply_dim_constraint ~source:Cur ~stage cur constr2 env in
          ( from_constr @ List.map subr2 ~f:(fun subr_v -> Dim_ineq { cur; subr = Var subr_v }),
            {
              env with
              dim_env =
                Map.set env.dim_env ~key:subr_v
                  ~data:(Bounds_dim { lub = Some cur; cur = cur2; subr = subr2; constr = constr2 });
            } ))
  | Var _, Dim _ (* when d2 > 1 *) -> ([ Dim_eq { d1 = cur; d2 = subr } ], env)
  | Dim _, Dim _ ->
      raise
      @@ Shape_error ("dimension comparison for axis: mismatch", [ Dim_mismatch [ cur; subr ] ])

let global_template_cache = Hashtbl.Poly.create ()

let%debug5_sexp solve_row_ineq ~(stage : stage) ~(cur : t) ~(subr : t) (env : environment) :
    constraint_ list * environment =
  let nonredundant ?(more = []) v vs =
    Utils.sorted_diff ~compare:compare_row_var
      (List.dedup_and_sort ~compare:compare_row_var (v :: vs))
      more
  in
  let l = List.length in
  let cur_dims_l : int = l cur.dims and subr_dims_l : int = l subr.dims in
  let cur_beg_dims =
    match cur.bcast with Row_var { beg_dims; _ } -> beg_dims | Broadcastable -> []
  in
  let subr_beg_dims =
    match subr.bcast with Row_var { beg_dims; _ } -> beg_dims | Broadcastable -> []
  in
  let cur_beg_dims_l = l cur_beg_dims and subr_beg_dims_l = l subr_beg_dims in
  let beg_dims_l = min cur_beg_dims_l subr_beg_dims_l in
  let dims_l = min cur_dims_l subr_dims_l in
  let ineqs =
    List.map2_exn
      ~f:(fun cur subr -> Dim_ineq { cur; subr })
      (take_from_end cur_beg_dims beg_dims_l)
      (take_from_end subr_beg_dims beg_dims_l)
    @ List.map2_exn
        ~f:(fun cur subr -> Dim_ineq { cur; subr })
        (take_from_end cur.dims dims_l) (take_from_end subr.dims dims_l)
  in
  match (cur, subr) with
  | ({ dims = _; bcast = Row_var { v; _ }; id }, _ | _, { dims = _; bcast = Row_var { v; _ }; id })
    when is_stage6_up stage ->
      ( Row_ineq { cur; subr }
        :: Row_eq { r1 = row_of_var v id; r2 = { dims = []; bcast = Broadcastable; id } }
        :: ineqs,
        env )
  | cur, subr when equal_row cur subr -> ([], env)
  | { bcast = Row_var { v = cur_v; _ }; _ }, { bcast = Row_var { v = subr_v; _ }; _ }
    when equal_row_var cur_v subr_v ->
      if cur_dims_l + cur_beg_dims_l = subr_dims_l + subr_beg_dims_l then (ineqs, env)
      else
        raise
        @@ Shape_error ("Infinite number of axes by self-reference", [ Row_mismatch [ cur; subr ] ])
  | { bcast = Row_var { v = cur_v; _ }; _ }, { bcast = Row_var { v = subr_v; _ }; _ }
    when cur_dims_l = subr_dims_l && cur_beg_dims_l = subr_beg_dims_l -> (
      match (Map.find env.row_env cur_v, Map.find env.row_env subr_v) with
      | Some (Bounds_row { cur = cur1; _ }), _ when List.mem ~equal:equal_row_var cur1 subr_v ->
          (Row_eq { r1 = row_of_var subr_v subr.id; r2 = row_of_var cur_v cur.id } :: ineqs, env)
      | _, Some (Bounds_row { subr = subr2; _ }) when List.mem ~equal:equal_row_var subr2 cur_v ->
          (Row_eq { r1 = row_of_var subr_v subr.id; r2 = row_of_var cur_v cur.id } :: ineqs, env)
      | Some (Bounds_row { subr = [ subr1 ]; _ }), Some (Bounds_row { cur = [ cur2 ]; _ })
        when is_stage2_up stage && equal_row_var subr1 subr_v && equal_row_var cur2 cur_v ->
          (Row_eq { r1 = row_of_var subr_v subr.id; r2 = row_of_var cur_v cur.id } :: ineqs, env)
      | Some (Bounds_row { subr = subr1; _ }), _ when List.mem ~equal:equal_row_var subr1 subr_v ->
          (ineqs, env)
      | _, Some (Bounds_row { cur = cur2; _ }) when List.mem ~equal:equal_row_var cur2 cur_v ->
          (ineqs, env)
      | None, None ->
          ( ineqs,
            {
              env with
              row_env =
                env.row_env
                |> Map.add_exn ~key:cur_v
                     ~data:
                       (Bounds_row
                          { cur = []; subr = [ subr_v ]; lub = None; constr = Unconstrained })
                |> Map.add_exn ~key:subr_v
                     ~data:
                       (Bounds_row
                          { cur = [ cur_v ]; subr = []; lub = None; constr = Unconstrained });
            } )
      | Some (Bounds_row { cur = cur1; subr = subr1; lub = lub1; constr = constr1 }), None ->
          ( ineqs,
            {
              env with
              row_env =
                env.row_env
                |> Map.set ~key:cur_v
                     ~data:
                       (Bounds_row
                          {
                            cur = cur1;
                            subr = nonredundant subr_v subr1;
                            lub = lub1;
                            constr = constr1;
                          })
                |> Map.add_exn ~key:subr_v
                     ~data:
                       (Bounds_row
                          { cur = [ cur_v ]; subr = []; lub = None; constr = Unconstrained });
            } )
      | None, Some (Bounds_row { cur = cur2; subr = subr2; lub = lub2; constr = constr2 }) ->
          ( ineqs,
            {
              env with
              row_env =
                env.row_env
                |> Map.set ~key:subr_v
                     ~data:
                       (Bounds_row
                          {
                            cur = nonredundant cur_v cur2;
                            subr = subr2;
                            lub = lub2;
                            constr = constr2;
                          })
                |> Map.add_exn ~key:cur_v
                     ~data:
                       (Bounds_row
                          { cur = []; subr = [ subr_v ]; lub = None; constr = Unconstrained });
            } )
      | ( Some (Bounds_row { cur = cur1; subr = subr1; lub = lub1; constr = constr1 }),
          Some (Bounds_row { cur = cur2; subr = subr2; lub = lub2; constr = constr2 }) ) ->
          ( ineqs,
            {
              env with
              row_env =
                env.row_env
                |> Map.set ~key:cur_v
                     ~data:
                       (Bounds_row
                          {
                            cur = cur1;
                            subr = nonredundant subr_v subr1;
                            lub = lub1;
                            constr = constr1;
                          })
                |> Map.set ~key:subr_v
                     ~data:
                       (Bounds_row
                          {
                            cur = nonredundant cur_v cur2;
                            subr = subr2;
                            lub = lub2;
                            constr = constr2;
                          });
            } )
      | Some (Solved_row _), _ | _, Some (Solved_row _) -> assert false)
  | { bcast = Row_var { v = cur_v; _ }; dims; _ }, _
    when cur_dims_l + cur_beg_dims_l < subr_dims_l + subr_beg_dims_l ->
      let budget = subr_dims_l + subr_beg_dims_l - (cur_dims_l + cur_beg_dims_l) in
      let more_dims_l = min budget @@ max 0 (subr_dims_l - cur_dims_l) in
      let more_dims : dim list =
        Array.(to_list @@ init more_dims_l ~f:(fun _ -> Var (get_var ())))
      in
      let budget = budget - more_dims_l in
      let more_beg_dims_l = min budget @@ max 0 (subr_beg_dims_l - cur_beg_dims_l) in
      let more_beg_dims : dim list =
        Array.(to_list @@ init more_beg_dims_l ~f:(fun _ -> Var (get_var ())))
      in
      (* The key of the template cache reflects that cur_v will end up substituted by
         {dims=more_dims; bcast=Row_var templ_v}. TODO: should we cache more_dims also? *)
      let templ_v : row_var =
        Hashtbl.find_or_add global_template_cache
          (cur_v, subr_dims_l - cur_dims_l, subr_beg_dims_l - cur_beg_dims_l)
          ~default:get_row_var
      in
      let template : t =
        {
          dims = more_dims @ dims;
          bcast = Row_var { v = templ_v; beg_dims = cur_beg_dims @ more_beg_dims };
          id = cur.id;
        }
      in
      (* We don't need to add any dimension inequalities, because they'll be captured by the extra
         row inequalities. *)
      ([ Row_eq { r1 = cur; r2 = template }; Row_ineq { cur = template; subr } ], env)
  | { bcast = Broadcastable; _ }, _ when cur_dims_l + cur_beg_dims_l < subr_dims_l + subr_beg_dims_l
    ->
      raise
      @@ Shape_error
           ( "Too many axes in a subtensor; maybe using * instead of *.?",
             [ Row_mismatch [ cur; subr ] ] )
  | { bcast; dims; id }, { bcast = Row_var { v = subr_v; _ }; _ }
    when subr_dims_l <= cur_dims_l && subr_beg_dims_l <= cur_beg_dims_l -> (
      let bcast =
        match bcast with
        | Row_var { v; beg_dims } -> Row_var { v; beg_dims = List.drop beg_dims beg_dims_l }
        | Broadcastable -> Broadcastable
      in
      let r_cur = { bcast; dims = drop_from_end dims dims_l; id } in
      match Map.find env.row_env subr_v with
      | None ->
          ( ineqs,
            {
              env with
              row_env =
                Map.add_exn env.row_env ~key:subr_v
                  ~data:
                    (Bounds_row { cur = []; subr = []; lub = Some r_cur; constr = Unconstrained });
            } )
      | Some (Bounds_row { cur = cur2; subr = subr2; lub = None; constr = constr2 }) ->
          ( ineqs,
            {
              env with
              row_env =
                env.row_env
                |> Map.set ~key:subr_v
                     ~data:
                       (Bounds_row { cur = cur2; subr = subr2; lub = Some r_cur; constr = constr2 });
            } )
      | Some (Bounds_row { cur = cur2; subr = subr2; lub = Some lub2; constr = constr2 }) ->
          let len1 = List.length r_cur.dims and len2 = List.length lub2.dims in
          let lub_len = min len1 len2 in
          let lub_is_cur = len1 < len2 || (len1 = len2 && is_broadcastable cur.bcast) in
          let lub_id = if lub_is_cur then r_cur.id else lub2.id in
          (* TODO: we lose connection here with the other bound if both have row variables. *)
          let lub_bcast = if lub_is_cur then r_cur.bcast else lub2.bcast in
          let lub_dims =
            List.map2_exn (take_from_end r_cur.dims lub_len) (take_from_end lub2.dims lub_len)
              ~f:(fun d1 d2 ->
                match (d1, d2) with
                | Dim { d = 1; _ }, _ -> d1
                | _, Dim { d = 1; _ } -> d2
                | Dim { d = d1; _ }, Dim { d = d2; _ } when d1 <> d2 -> get_dim ~d:1 ()
                | Var _, _ -> d1
                | _, Var _ -> d2
                | Dim _, Dim _ -> d1)
          in
          let lub = { dims = lub_dims; bcast = lub_bcast; id = lub_id } in
          ( ineqs,
            {
              env with
              row_env =
                env.row_env
                |> Map.set ~key:subr_v
                     ~data:
                       (Bounds_row { cur = cur2; subr = subr2; lub = Some lub; constr = constr2 });
            } )
      | Some (Solved_row _) -> assert false)
  | _ when cur_beg_dims_l > beg_dims_l && not (is_stage7 stage) ->
      (Row_ineq { cur; subr } :: ineqs, env)
  | _, { bcast = Broadcastable; _ }
    when subr_dims_l + subr_beg_dims_l <= cur_dims_l + cur_beg_dims_l ->
      (ineqs, env)
  | { bcast = Row_var _ | Broadcastable; _ }, { bcast = Row_var _ | Broadcastable; _ } ->
      (Row_ineq { cur; subr } :: ineqs, env)

let%debug5_sexp close_dim_terminal ~(stage : stage) (env : environment) (dim : dim) :
    constraint_ list =
  match dim with
  | Dim _ -> []
  | Var v -> (
      match Map.find env.dim_env v with
      | Some (Solved_dim _) -> assert false
      | Some (Bounds_dim { lub = None; constr = Unconstrained_dim; _ }) when is_stage2_up stage ->
          [ Dim_eq { d1 = dim; d2 = get_dim ~d:1 () } ]
      | Some (Bounds_dim { lub = Some lub; _ }) when is_stage3_up stage ->
          [ Dim_eq { d1 = dim; d2 = lub } ]
      | _ when not (is_stage4_up stage) -> [ Terminal_dim dim ]
      | _ -> [])

let last_dim_is dims d2 = match List.last dims with Some (Dim { d; _ }) -> d = d2 | _ -> false

let%debug5_sexp rec eliminate_row_constraint ~lub (r : row) (constr : row_constraint) env :
    constraint_ list =
  match r with
  | { bcast = Broadcastable; _ } ->
      (* The environment is unchanged, as apply_row_constraint would update only the constr. *)
      let ineqs, _env = apply_row_constraint ~stage:Stage5 r constr env in
      List.concat_map ineqs ~f:(function
        | Row_constr { r = r'; constr } ->
            if not (phys_equal r r') then eliminate_row_constraint ~lub:None r constr env else []
        | ineq -> [ ineq ])
  | { bcast = Row_var { v; beg_dims }; dims; id } -> (
      let r1 = row_of_var v id in
      let no_further_axes = Row_eq { r1; r2 = { dims = []; bcast = Broadcastable; id } } in
      (* Note: the reduced constraint applies to just the row variable. *)
      match reduce_row_constraint constr ~beg_dims ~dims with
      | Total_elems { nominator = d; divided_by } -> (
          match (d, Set.elements divided_by, lub) with
          | 1, vs, _ ->
              no_further_axes
              :: List.map vs ~f:(fun v ->
                     let d2 = get_dim ~d:1 () in
                     Dim_eq { d1 = Var v; d2 })
          | _, [], None ->
              let dim = get_dim ~d () in
              [ Row_eq { r1; r2 = { dims = [ dim ]; bcast = Broadcastable; id } } ]
          | _, [], Some { dims; _ } when last_dim_is dims d ->
              let dim = get_dim ~d () in
              [ Row_eq { r1; r2 = { dims = [ dim ]; bcast = Broadcastable; id } } ]
          | _, [], Some lub ->
              let ineqs, _env = apply_row_constraint ~stage:Stage5 lub constr env in
              List.concat_map ineqs ~f:(function
                | Row_constr { r = r'; constr } ->
                    if not (phys_equal r r') then eliminate_row_constraint ~lub:None r constr env
                    else []
                | ineq -> [ ineq ])
          | _, [ v ], _ -> no_further_axes :: [ Dim_eq { d1 = Var v; d2 = get_dim ~d () } ]
          | _ -> [])
      | _ -> [])

let%debug5_sexp close_row_terminal ~(stage : stage) (env : environment)
    ({ dims; bcast; id } as _r : row) : constraint_ list =
  let suffix () = List.map dims ~f:(fun d -> Terminal_dim d) in
  match bcast with
  | Broadcastable -> if is_stage5_up stage then [] else suffix ()
  | Row_var { v; beg_dims } -> (
      let term_dims () = List.map beg_dims ~f:(fun d -> Terminal_dim d) @ suffix () in
      let r1 : row = row_of_var v id in
      let no_further_axes = Row_eq { r1; r2 = { dims = []; bcast = Broadcastable; id } } in
      match Map.find env.row_env v with
      | Some (Bounds_row { lub = None; constr = Unconstrained; _ }) when is_stage3_up stage ->
          [%log6 "terminal row: closing", (_r : row)];
          no_further_axes :: term_dims ()
      | Some (Bounds_row { lub = None; constr; _ })
        when is_stage2_up stage && not (equal_row_constraint constr Unconstrained) ->
          let ineqs =
            (* This is the constraint on the row variable, not on the original row. *)
            try eliminate_row_constraint r1 ~lub:None constr env
            with Shape_error (s, trace) -> raise @@ Shape_error (s, Row_mismatch [ r1 ] :: trace)
          in
          ineqs @ term_dims ()
      | Some (Solved_row _) -> assert false
      | Some (Bounds_row { lub = Some lub; _ }) when is_stage3_up stage ->
          Row_eq { r1; r2 = lub } :: term_dims ()
      | _ when is_stage5_up stage -> []
      | _ ->
          [%log6 "terminal row: keeping", (_r : row), "as", (r1 : row)];
          Terminal_row r1 :: term_dims ())

let%debug5_sexp eliminate_dim_entry v ~lub constr =
  match (lub, constr) with
  | _, Unconstrained_dim | _, At_least_dim 1 -> None
  | Some (Dim { d; _ } as lub), At_least_dim d2 when d2 > d ->
      raise
      @@ Shape_error
           ( [%string "dereferenced at dimension %{d2#Int}, higher than use site"],
             [ Dim_mismatch [ lub; Var v ] ] )
  | Some lub, At_least_dim _ -> Some (Dim_eq { d1 = Var v; d2 = lub })
  | None, At_least_dim d -> Some (Dim_eq { d1 = Var v; d2 = get_dim ~d () })

let%debug5_sexp eliminate_variables (env : environment) ({ dims; bcast; id } as _r : row) :
    constraint_ list =
  let f = function
    | Var v as d1 ->
        Some
          (match Map.find env.dim_env v with
          | Some (Bounds_dim { lub; constr; _ }) ->
              Option.value_or_thunk (eliminate_dim_entry v ~lub constr) ~default:(fun () ->
                  Dim_eq { d1; d2 = get_dim ~d:1 () })
          | Some (Solved_dim _) -> assert false
          | None -> Dim_eq { d1; d2 = get_dim ~d:1 () })
    | _ -> None
  in
  let suffix = List.filter_map dims ~f in
  match bcast with
  | Broadcastable -> suffix
  | Row_var { v; beg_dims } -> (
      let elim_dims = List.filter_map beg_dims ~f @ suffix in
      let r2 = { dims = []; bcast = Broadcastable; id } in
      let elim_var = Row_eq { r1 = row_of_var v id; r2 } in
      match Map.find env.row_env v with
      | Some (Bounds_row { constr = Total_elems _; _ }) -> assert false
      | _ -> elim_var :: elim_dims)

let empty_env = { dim_env = Map.empty (module Dim_var); row_env = Map.empty (module Row_var) }

let%debug4_sexp solve_inequalities ~(stage : stage) (ineqs : constraint_ list) (env : environment) :
    constraint_ list * environment =
  let rec solve (ineqs : constraint_ list) (env : environment) : constraint_ list * environment =
    let f (ineqs, env) (ineq : constraint_) =
      match ineq with
      | Dim_eq { d1; d2 } ->
          (* Substituted inside unify_dim. *)
          let more_ineqs, env = unify_dim ~stage (d1, d2) env in
          (more_ineqs @ ineqs, env)
      | Row_eq { r1; r2 } ->
          (* Substituted inside unify_row. *)
          let more_ineqs, env = unify_row ~stage (r1, r2) env in
          (more_ineqs @ ineqs, env)
      | Dim_ineq { cur; subr } ->
          let cur = subst_dim env cur and subr = subst_dim env subr in
          let more_ineqs, env = solve_dim_ineq ~stage ~cur ~subr env in
          (more_ineqs @ ineqs, env)
      | Row_ineq { cur; subr } ->
          let cur = subst_row env cur and subr = subst_row env subr in
          let more_ineqs, env = solve_row_ineq ~stage ~cur ~subr env in
          (more_ineqs @ ineqs, env)
      | Dim_constr { d; constr } ->
          let d = subst_dim env d in
          let extras, constr = apply_dim_constraint ~source:Direct ~stage d constr env in
          let env =
            match (constr, d) with
            | Unconstrained_dim, _ | _, Dim _ -> env
            | _, Var v ->
                {
                  env with
                  dim_env =
                    Map.update env.dim_env v ~f:(function
                      | Some (Solved_dim _) -> assert false
                      | Some (Bounds_dim bounds) -> Bounds_dim { bounds with constr }
                      | None -> Bounds_dim { constr; lub = None; cur = []; subr = [] });
                }
          in
          (extras @ ineqs, env)
      | Row_constr { r; constr } ->
          let r = subst_row env r in
          let more_ineqs, env = apply_row_constraint ~stage r constr env in
          (more_ineqs @ ineqs, env)
      | Terminal_dim d ->
          let more_ineqs = close_dim_terminal ~stage env @@ subst_dim env d in
          (more_ineqs @ ineqs, env)
      | Terminal_row r ->
          let more_ineqs = close_row_terminal ~stage env @@ subst_row env r in
          (more_ineqs @ ineqs, env)
    in
    let ineqs', env = List.fold ineqs ~init:([], env) ~f in
    let ineqs' = List.rev ineqs' in
    if
      List.is_empty ineqs'
      || (List.length ineqs' = List.length ineqs && [%equal: constraint_ list] ineqs' ineqs)
    then (ineqs', env)
    else solve ineqs' env
  in
  match stage with
  | Stage1 | Stage2 | Stage3 | Stage6 | Stage7 -> solve ineqs env
  | Stage4 ->
      let finalize_lower_bound (v : dim_var) = function
        | Bounds_dim { lub; constr; _ } -> Option.to_list @@ eliminate_dim_entry v ~lub constr
        | _ -> []
      in
      let finalizing_entries : constraint_ list =
        Map.fold env.dim_env ~init:[] ~f:(fun ~key ~data accu ->
            finalize_lower_bound key data @ accu)
      in
      solve (finalizing_entries @ ineqs) env
  | Stage5 ->
      let finalize_total_elems v = function
        | Bounds_row { lub; constr; _ } ->
            (* TODO: should we store the id somewhere? *)
            let id = phantom_row_id in
            eliminate_row_constraint (row_of_var v id) ~lub constr env
        | _ -> []
      in
      let finalizing_entries : constraint_ list =
        Map.fold env.row_env ~init:[] ~f:(fun ~key ~data accu ->
            finalize_total_elems key data @ accu)
      in
      solve (finalizing_entries @ ineqs) env

let rec row_to_labels env =
  let rec f = function
    | Dim { label = Some l; _ } -> l
    | Dim { label = None; _ } -> ""
    | Var v -> (
        match Map.find env.dim_env v with
        | None | Some (Bounds_dim _) -> Option.value v.label ~default:""
        | Some (Solved_dim dim) -> f dim)
  in
  function
  | { dims; bcast = Row_var { v; beg_dims }; id } -> (
      match Map.find env.row_env v with
      | None | Some (Bounds_row _) -> Array.of_list_map (beg_dims @ dims) ~f
      | Some (Solved_row { dims = dims2; bcast = Broadcastable; _ }) ->
          row_to_labels env { dims = beg_dims @ dims2 @ dims; bcast = Broadcastable; id }
      | Some (Solved_row { dims = dims2; bcast = Row_var { v = v2; beg_dims = beg_dims2 }; _ }) ->
          row_to_labels env
            { dims = dims2 @ dims; bcast = Row_var { v = v2; beg_dims = beg_dims @ beg_dims2 }; id }
      )
  | { dims; bcast = Broadcastable; id = _ } -> Array.of_list_map dims ~f

(** *** Projection inference *** *)

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

(* let update_proj_classes pid1 pid2 proj_classes = Utils.union_add ~equal:Int.equal proj_classes
   pid1 pid2 *)

type proj = Var of dim_var | Proj of { proj_id : int; d : int } | Solved of Idx.axis_index
[@@deriving compare, equal, sexp]

type error_trace += Projection_mismatch of proj list

let sexp_of_error_trace = function
  | Projection_mismatch ps ->
      Sexp.List (Sexp.Atom "Projection_mismatch" :: List.map ps ~f:sexp_of_proj)
  | error_trace -> sexp_of_error_trace error_trace

type proj_to_index = Idx.axis_index Map.M(Int).t [@@deriving sexp]
type proj_classes = int Map.M(Int).t [@@deriving sexp]

type proj_env = {
  proj_to_index : proj_to_index;
  proj_classes : proj_classes;
  product_dim : int Map.M(Int).t;
  non_product : Set.M(Int).t;
}
[@@deriving sexp]

type proj_equation =
  | Proj_eq of proj * proj
      (** Two projections are the same, e.g. two axes share the same iterator. *)
  | Iterated of proj
      (** The projection needs to be an iterator even if an axis is not matched with another axis,
          e.g. for broadcasted-to axes of a tensor assigned a constant. *)
[@@deriving compare, equal, sexp]

let%debug4_sexp get_proj_equations (inequalities : constraint_ list) proj_axis_env
    (env : environment) : proj_equation list =
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
  let rec expand_dims = function
    | { dims; bcast = Row_var { v; beg_dims }; _ } when Map.mem env.row_env v -> (
        match Map.find_exn env.row_env v with
        | Solved_row r ->
            let more_dims = expand_dims r in
            beg_dims @ more_dims @ dims
        | _ -> dims)
    | { dims; _ } -> dims
  in
  let match_rows ~(with_broadcasting : bool) (r1 : row) (r2 : row) : proj_equation list =
    let dims1 : dim list = expand_dims r1 in
    let dims2 : dim list = expand_dims r2 in
    let len1 = List.length dims1 in
    let len = min len1 (List.length dims2) in
    let extras =
      if with_broadcasting then
        List.map ~f:(fun d -> Iterated (to_proj d)) @@ List.take dims1 (len1 - len)
      else []
    in
    extras
    @ (List.zip_exn (take_from_end dims1 len) (take_from_end dims2 len)
      |> List.map ~f:(fun (d1, d2) -> Proj_eq (to_proj d1, to_proj d2)))
  in
  let f = function
    | Dim_ineq { cur = _; subr = Dim { d = 1; proj_id = Some proj_id; _ } } ->
        [ Proj_eq (Proj { proj_id; d = 1 }, Solved (Fixed_idx 0)) ]
    | Dim_eq { d1; d2 } | Dim_ineq { cur = d1; subr = d2 } -> [ Proj_eq (to_proj d1, to_proj d2) ]
    | Row_eq { r1; r2 } -> match_rows ~with_broadcasting:false r1 r2
    | Row_ineq { cur = r1; subr = r2 } ->
        match_rows ~with_broadcasting:true r1 r2
        |> List.concat_map ~f:(function
             | Proj_eq (proj1, (Proj { proj_id = _; d = 1 } as proj2)) ->
                 [ Iterated proj1; Proj_eq (proj2, Solved (Fixed_idx 0)) ]
             | eq -> [ eq ])
    | Dim_constr _ | Row_constr _ | Terminal_dim _ | Terminal_row _ -> []
  in
  List.concat_map inequalities ~f

let%debug4_sexp solve_proj_equations (eqs : proj_equation list) : proj_env =
  let v_env = dim_hashtbl () in
  let p_solved = ref [] in
  let p_dims = ref [] in
  let proj_classes = ref @@ Map.empty (module Int) in
  let rec loop = function
    | Proj_eq (Proj { proj_id = p1; d }, Proj { proj_id = p2; _ }) when p1 = p2 ->
        p_dims := (p1, d) :: !p_dims
    | Proj_eq (Var v1, Var v2) when equal_dim_var v1 v2 -> ()
    | Proj_eq ((Proj { proj_id = p1; d = d1 } as proj1), (Proj { proj_id = p2; d = d2 } as proj2))
      ->
        if d1 <> d2 then
          raise
          @@ Shape_error
               ( "Conflicting dimensions for the same projection",
                 [ Projection_mismatch [ proj1; proj2 ] ] );
        p_dims := (p1, d1) :: !p_dims;
        proj_classes := Utils.union_add ~equal:Int.equal !proj_classes p1 p2
    | Proj_eq (Proj p, Solved idx) | Proj_eq (Solved idx, Proj p) ->
        p_solved := (p.proj_id, idx) :: !p_solved
    | Proj_eq (Solved idx1, Solved idx2) when Idx.equal_axis_index idx1 idx2 -> ()
    | Proj_eq (Solved idx1, Solved idx2) ->
        raise
        @@ Shape_error
             ("Conflicting indices for the same axis/projection", [ Index_mismatch [ idx1; idx2 ] ])
    | Proj_eq (Var v, p) | Proj_eq (p, Var v) -> (
        match Hashtbl.find v_env v with
        | None -> Hashtbl.add_exn v_env ~key:v ~data:p
        | Some p2 -> loop (Proj_eq (p, p2)))
    | Iterated (Solved _) -> ()
    | Iterated (Proj { proj_id; d }) -> p_dims := (proj_id, d) :: !p_dims
    | Iterated (Var v) -> (
        match Hashtbl.find v_env v with
        | None ->
            let idx = Idx.(Iterator (get_symbol ())) in
            Hashtbl.add_exn v_env ~key:v ~data:(Solved idx)
        | Some (Var v2) -> loop (Iterated (Var v2))
        | Some (Solved _) -> ()
        | Some (Proj { proj_id; d }) -> p_dims := (proj_id, d) :: !p_dims)
  in
  List.iter eqs ~f:loop;
  let projs = ref @@ Map.empty (module Int) and non_product = ref @@ Set.empty (module Int) in
  List.iter !p_solved ~f:(fun (p, idx) ->
      let repr, _ = Utils.union_find ~equal:Int.equal !proj_classes ~key:p ~rank:0 in
      non_product := Set.add !non_product repr;
      Utils.mref_add projs ~key:repr ~data:idx ~or_:(fun idx2 ->
          if not @@ Idx.equal_axis_index idx idx2 then
            raise
            @@ Shape_error
                 ("Multiple constraints on the same projection", [ Index_mismatch [ idx; idx2 ] ])));
  let product_dim = ref @@ Map.empty (module Int) in
  List.iter !p_dims ~f:(fun (p, d) ->
      let repr, _ = Utils.union_find ~equal:Int.equal !proj_classes ~key:p ~rank:0 in
      if Idx.iterated d && (not @@ Map.mem !projs repr) then
        Utils.mref_add product_dim ~key:repr ~data:d ~or_:(fun d2 ->
            if d <> d2 then
              raise
              @@ Shape_error
                   ( "Conflicting dimensions for the same projection",
                     [
                       Projection_mismatch [ Proj { proj_id = p; d }; Proj { proj_id = p; d = d2 } ];
                     ] )));
  Map.iteri !product_dim ~f:(fun ~key:p ~data:_ ->
      let repr, _ = Utils.union_find ~equal:Int.equal !proj_classes ~key:p ~rank:0 in
      Utils.mref_add_missing projs repr ~f:(fun () -> Idx.(Iterator (get_symbol ()))));
  {
    proj_classes = !proj_classes;
    proj_to_index = !projs;
    product_dim = !product_dim;
    non_product = !non_product;
  }

let get_proj_index proj_env = function
  | Dim { d; _ } when not @@ Idx.iterated d -> Idx.Fixed_idx 0
  | Dim { proj_id = None; _ } -> assert false
  | Var v as dim ->
      raise
      @@ Shape_error
           ( "projection_of_solved_dims: still not fully inferred for variable "
             ^ Sexp.to_string_hum ([%sexp_of: dim_var] v),
             [ Dim_mismatch [ dim ] ] )
  | Dim { proj_id = Some proj_id; d; _ } -> (
      let repr, _ = Utils.union_find ~equal:Int.equal proj_env.proj_classes ~key:proj_id ~rank:0 in
      match Map.find proj_env.proj_to_index repr with
      | Some i -> i
      | None ->
          raise
          @@ Shape_error
               ( "projection_of_solved_dims: unknown projection",
                 [ Projection_mismatch [ Proj { proj_id; d } ] ] ))

let proj_repr proj_env p =
  fst @@ Utils.union_find ~equal:Int.equal proj_env.proj_classes ~key:p ~rank:0

let get_product_proj proj_env dim =
  match dim with
  | Dim { d; _ } when not @@ Idx.iterated d -> None
  | Dim { proj_id = Some proj_id; d; _ } ->
      let repr = proj_repr proj_env proj_id in
      if Map.mem proj_env.proj_to_index repr && (not @@ Set.mem proj_env.non_product repr) then
        Some (repr, d)
      else None
  | Dim { proj_id = None; _ } -> None
  | Var v ->
      raise
      @@ Shape_error
           ( "projection_of_solved_dims: still not fully inferred for variable "
             ^ Sexp.to_string_hum ([%sexp_of: dim_var] v),
             [ Dim_mismatch [ dim ] ] )

let proj_to_iterator proj_env p =
  match Map.find_exn proj_env.proj_to_index (proj_repr proj_env p) with
  | Iterator s -> s
  | _ -> assert false
