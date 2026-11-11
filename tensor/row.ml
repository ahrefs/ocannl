(** The row type, shape inference related types and constraint solving. *)

open Base

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_ROW=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_ROW"]

type axis_padding = Ir.Ops.axis_padding [@@deriving equal, sexp]

module Dim_var = struct
  type t = { id : int; name : string option [@compare.ignore] [@equal.ignore] [@hash.ignore] }
  [@@deriving equal, hash, compare, sexp]

  let to_string { id; name } =
    match name with None -> "$" ^ Int.to_string id | Some n -> [%string "$%{id#Int}:%{n}"]

  include Comparator.Make (struct
    type nonrec t = t

    let compare = compare
    let sexp_of_t = sexp_of_t
  end)
end

type dim_var = Dim_var.t [@@deriving equal, hash, compare, sexp]
type dim_cmp = Dim_var.comparator_witness
type dim_var_set = Set.M(Dim_var).t [@@deriving equal, hash, compare, sexp]
type 'a dim_map = 'a Map.M(Dim_var).t [@@deriving equal, sexp]

module Proj_id = struct
  type t = Proj_id of int [@@deriving equal, hash, compare, sexp]

  let to_string (Proj_id i) = Int.to_string i

  let fresh =
    let uid = ref 0 in
    fun () ->
      Int.incr uid;
      Proj_id !uid
  [@@mutation_serial.exempt
    "a fresh-id counter: ids are never reused, so a rewind need not restore it"]

  include Comparator.Make (struct
    type nonrec t = t

    let compare = compare
    let sexp_of_t = sexp_of_t
  end)
end

type proj_id = Proj_id.t [@@deriving equal, hash, compare, sexp]
type proj_cmp = Proj_id.comparator_witness
type proj_var_set = Set.M(Proj_id).t [@@deriving equal, sexp]
type 'a proj_map = 'a Map.M(Proj_id).t [@@deriving equal, sexp]

let proj_var_set_empty = Set.empty (module Proj_id)
let proj_map_empty = Map.empty (module Proj_id)
let dim_var_set_empty = Set.empty (module Dim_var)
let dim_map_empty = Map.empty (module Dim_var)

(* The reserved basis tag of the claim-free broadcast top: an axis with this tag broadcasts to any
   size while it remains size 1, and is an ordinary fixed axis otherwise. The top of the broadcast
   order is precisely [1_(bcast_if_1)]. Scalars and rank-broadening fill carry it. *)
let bcast_if_1 = "bcast_if_1"

(* The default basis tag the frontend supplies for any axis the user writes without naming a basis.
   It is an ordinary atom, incompatible with other named bases (including [bcast_if_1]). *)
let default_basis = "default"

type solved_dim = { d : int; basis : string; proj_id : proj_id option }
[@@deriving equal, hash, compare, sexp]

type convolution = { dilation : int; kernel : dim; use_padding : bool }
[@@deriving equal, hash, compare, sexp]

and dim =
  | Var of dim_var
  | Dim of solved_dim
  | Sym of sym_dim
  | Affine of { stride : int; over : dim; conv : convolution option; stride_offset : int }
  | Concat of dim list

and sym_dim = { sym : Ir.Indexing.static_symbol; sym_basis : string; sym_proj_id : proj_id option }
[@@deriving equal, hash, compare, sexp]

let equal_dim d1 d2 =
  match (d1, d2) with
  | Dim { d = d1; basis = b1; proj_id = _ }, Dim { d = d2; basis = b2; proj_id = _ } ->
      d1 = d2 && String.equal b1 b2
  | ( Sym { sym = s1; sym_basis = b1; sym_proj_id = _ },
      Sym { sym = s2; sym_basis = b2; sym_proj_id = _ } ) ->
      Ir.Indexing.equal_static_symbol s1 s2 && String.equal b1 b2
  | _ -> equal_dim d1 d2

let uid = ref 0
[@@mutation_serial.exempt
  "a fresh-id counter: ids are never reused, so a rewind need not restore it"]

(* Variables minted inside an id-neutral window (gh-ocannl-1751: a shape-scheme trace, a check) come
   from a reserved range with a counter of their own, never reused: the window's variables are all
   discarded, and the ids the session mints -- printed in shapes and errors -- are then the same
   whether or not the window ran. *)
let reserved_uid_base = 1 lsl 40

let reserved_uid = ref reserved_uid_base
[@@mutation_serial.exempt
  "a fresh-id counter of the id-neutral windows: never reused, so never restored"]

let with_reserved_vars f =
  if !uid >= reserved_uid_base then f ()
  else
    let saved = !uid in
    uid := !reserved_uid;
    Exn.protect ~f ~finally:(fun () ->
        reserved_uid := max !reserved_uid !uid;
        uid := saved)

(* gh-ocannl-1859: a successful closure retires every variable minted so far
   ([retire_closed_variables]): its shapes are ground and the environment empty, so nothing should
   ask the solver's per-variable tables about one of them again. The two floors, the newest retired
   id of each range, let [debug_check_retired_solver_vars] refuse a read that does, since the
   emptied table would quietly answer "no". *)
let retired_floor = ref 0
[@@mutation_serial.exempt
  "the newest retired variable id: closures only raise it, and a rewind reaches no older variable"]

let retired_reserved_floor = ref reserved_uid_base
[@@mutation_serial.exempt
  "the newest retired variable id: closures only raise it, and a rewind reaches no older variable"]

let check_retired_reads =
  lazy (Utils.get_global_flag ~default:false ~arg_name:"debug_check_retired_solver_vars")

let refuse_retired_read ~table id =
  if
    Lazy.force check_retired_reads
    && id <= if id >= reserved_uid_base then !retired_reserved_floor else !retired_floor
  then
    failwith
      [%string
        "debug_check_retired_solver_vars: %{table} was asked about variable %{id#Int}, minted \
         before the last closure, which retired it"]

let get_var ?name () : dim_var =
  Int.incr uid;
  { id = !uid; name }

(* [basis] is required (Option A totality): every minted dimension carries a tag so the compiler
   surfaces every construction site and forces a provenance decision. Use [get_bcast_dim] for the
   broadcast top (scalars, rank-broadening) and [get_default_dim] for unannotated user/derived
   atoms. *)
let get_dim ~d ~basis ?proj_id () =
  let proj_id = Option.map ~f:(fun p -> Proj_id.Proj_id p) proj_id in
  Dim { d; basis; proj_id }

(* Mint the claim-free broadcast top (or, at sizes > 1, an inert [bcast_if_1] atom). *)
let get_bcast_dim ~d ?proj_id () = get_dim ~d ~basis:bcast_if_1 ?proj_id ()

(* Mint an unannotated user/derived atom. *)
let get_default_dim ~d ?proj_id () = get_dim ~d ~basis:default_basis ?proj_id ()

(* The maximum extent of a symbolic dimension: the declared range of its static symbol. Symbolic
   dimensions are materialized at this extent (Tnode dims, projections product space) until
   launch-time extent binding lands (gh-490 backend tasks). *)
let sym_dim_extent_exn { sym; _ } =
  match sym.Ir.Indexing.static_range with
  | Some range -> range
  | None ->
      raise
      @@ Utils.User_error
           (Printf.sprintf
              "Row: static index %s used as a symbolic dimension has no declared range; symbolic \
               dimensions require ~static_range (it is the maximum extent, used for allocation \
               sizing)"
              (Ir.Indexing.symbol_ident sym.Ir.Indexing.static_symbol))

(* Materialize a symbolic dimension as a solved dimension at its maximum extent. Used at the
   projection / Tnode boundary: until launch-time extent binding lands, symbolic axes iterate and
   allocate at their declared range. *)
let solved_dim_of_sym ({ sym = _; sym_basis; sym_proj_id } as sdim) =
  { d = sym_dim_extent_exn sdim; basis = sym_basis; proj_id = sym_proj_id }

(* {2 Session bookkeeping (gh-ocannl-1301)}

   [mutation_serial] counts writes that a failed construction can leave behind in the inference
   session (constraints, solver state, delayed references, role claims, precision links, root
   changes). [Tensor.op] and the other construction entry points compare it on exit: a construction
   that raises after the serial moved taints the session, one that raises before it does not.
   Substituting the current environment into a shape is neutral and does not count. A write to any
   module-level mutable binding here or in [Shape] calls [note_mutation] or is classified in place
   by a [@@mutation_serial.*] attribute; test/operations/mutation_serial_scan enforces it
   (gh-ocannl-1506). *)
let serial = ref 0 [@@mutation_serial.exempt "the serial itself: note_mutation is its writer"]
let note_mutation () = Int.incr serial
let mutation_serial () = !serial

(* Static-symbol roles claimed since the last closure (gh-ocannl-1301). A role ([used_as_extent] /
   [used_as_slice]) is a flag on a caller-retained symbol, so a construction that is later discarded
   by [Tensor.rewind_to_last_closure] must not leave it set: claims are recorded here and committed
   to the symbol by a successful [Shape.finish_inference] (nothing reads the flags before lowering,
   which closes first), and dropped by a rewind. The conflict checks consult both. The flags are
   private to [Ir.Indexing] (gh-ocannl-1624): the commit writes them through
   [Ir.Indexing.commit_role], whose writer [Shape] holds and passes in. *)
let pending_roles : (Ir.Indexing.static_symbol * [ `Extent | `Slice ]) list ref = ref []

let pending_role sym role =
  List.exists !pending_roles ~f:(fun (s, r) ->
      phys_equal s sym && Poly.equal (r : [ `Extent | `Slice ]) role)

let is_used_as_extent sym = sym.Ir.Indexing.used_as_extent || pending_role sym `Extent
let is_used_as_slice sym = sym.Ir.Indexing.used_as_slice || pending_role sym `Slice

let claim_symbol_role sym role =
  let committed =
    match role with
    | `Extent -> sym.Ir.Indexing.used_as_extent
    | `Slice -> sym.Ir.Indexing.used_as_slice
  in
  if not (committed || pending_role sym role) then (
    note_mutation ();
    pending_roles := (sym, role) :: !pending_roles)

let commit_symbol_roles ~commit =
  List.iter !pending_roles ~f:(fun (sym, role) -> commit sym role);
  pending_roles := []
[@@mutation_serial.boundary "part of the commit: only a successful finish_inference calls it"]

let drop_symbol_roles () = pending_roles := []
[@@mutation_serial.boundary "part of a rewind or session reset, which clears the taint"]

(* Mint a symbolic dimension from a static-indexing symbol: a rigid constant equal only to itself
   during shape inference. Requires a declared [static_range >= 1] (checked eagerly). *)
let get_sym_dim ?(basis = default_basis) ?proj_id sym =
  let proj_id = Option.map ~f:(fun p -> Proj_id.Proj_id p) proj_id in
  let sdim = { sym; sym_basis = basis; sym_proj_id = proj_id } in
  let extent = sym_dim_extent_exn sdim in
  if extent < 1 then
    raise
    @@ Utils.User_error
         (Printf.sprintf
            "Row: static index %s used as a symbolic dimension has declared range %d; symbolic \
             dimensions require a positive maximum extent"
            (Ir.Indexing.symbol_ident sym.Ir.Indexing.static_symbol)
            extent);
  (* The bound value of an extent symbol is a size in [0, range] (inclusive), not an index in [0,
     range): switch the symbol's bind-time validation accordingly. A slice index validates strictly,
     so a binding cannot be used as both. *)
  if is_used_as_slice sym then
    raise
    @@ Utils.User_error
         (Printf.sprintf
            "Row: static index %s already slices a batch axis and cannot also be used as a \
             symbolic dimension extent"
            (Ir.Indexing.symbol_ident sym.Ir.Indexing.static_symbol));
  claim_symbol_role sym `Extent;
  Sym sdim

(* The reserved tags ([default], [bcast_if_1]) carry no naming claim. *)
let is_reserved_basis b = String.equal b default_basis || String.equal b bcast_if_1

(* Combine the bases of two components of a concatenated dimension. A named tag wins over a reserved
   (claim-free) tag; two distinct named tags are a conflict handled by [on_conflict]. *)
let merge_derived_basis ~on_conflict b1 b2 =
  if String.equal b1 b2 then b1
  else if is_reserved_basis b1 then b2
  else if is_reserved_basis b2 then b1
  else on_conflict ()

(* The basis a valid window derives for its input from its extent's and its kernel's
   (gh-ocannl-1968). [default] and the top [1_(bcast_if_1)] make no claim, and every other tag is an
   atom, which includes [bcast_if_1] at any size but 1 (the inert atom of docs/shape_inference.md).
   Two atoms must agree, an atom wins over a claim-free tag, and two claim-free tags stay [default]
   unless both are the top. [None] is a conflict. *)
let is_claim_free (s : solved_dim) =
  String.equal s.basis default_basis || (String.equal s.basis bcast_if_1 && s.d = 1)

let window_input_basis ~(extent : solved_dim) ~(kernel : solved_dim) =
  if String.equal extent.basis kernel.basis then Some extent.basis
  else
    match (is_claim_free extent, is_claim_free kernel) with
    | true, true -> Some default_basis
    | true, false -> Some kernel.basis
    | false, true -> Some extent.basis
    | false, false -> None

(* The rule is not injective, so the tags of an unsolved output or kernel are what the rule says
   they are: its candidates, filtered by the rule itself. Inverting it by hand is how the reverse
   arm came to commit tags the forward arm refused. Besides the two tags in play only the claim-free
   ones can map onto the input. Several survive only when a claim-free tag does, so [default] is
   always among them: [input = F(default, kernel)] then, whichever window asks, so settling an open
   window with it cannot contradict another one over the same output. *)
let window_bases ~size ~tags ~derives =
  List.dedup_and_sort ~compare:String.compare (default_basis :: bcast_if_1 :: tags)
  |> List.filter ~f:(fun basis -> derives { d = size; basis; proj_id = None })

let window_output_bases ~size ~(input : solved_dim) ~(kernel : solved_dim) =
  window_bases ~size ~tags:[ input.basis; kernel.basis ] ~derives:(fun extent ->
      Option.equal String.equal (window_input_basis ~extent ~kernel) (Some input.basis))

let effective_kernel_span ~dilation ~kernel_size =
  (* The effective kernel span with dilation is: 1 + (kernel_size - 1) * dilation This is the actual
     range of input indices covered by the kernel. *)
  1 + ((kernel_size - 1) * dilation)

let kernel_size_of_span ~dilation ~span =
  (* Inverse of effective_kernel_span. Given span, find kernel_size such that effective_kernel_span
     ~dilation ~kernel_size = span. Since effective_kernel_span = 1 + (kernel_size - 1) * dilation,
     we have: span - 1 = (kernel_size - 1) * dilation kernel_size = (span - 1) / dilation + 1 *)
  if (span - 1) % dilation <> 0 then None else Some (((span - 1) / dilation) + 1)

(* A stride-1 valid window of effective span 1 -- [input = 1 * (output - 1) + 1] -- is the identity
   on its output extent, as a stride-1 padded window or a plain stride-1 affine is (a valid 1x1
   conv, say). On a shape-scheme trace's unsolved extents it must collapse: kept, a residual add
   equating its [over] with it reads as an occurs-check cycle (gh-ocannl-1820, where [unify_dim]'s
   mixed padded/valid arm built one as a helper; [window_pair_equation] now equates the outputs
   directly instead, gh-ocannl-1823). Collapsed, nothing is left to constrain the extent's basis
   later, so the identity passes the basis through as it passes the size, over a solved extent too
   (gh-ocannl-1824): deriving [window_input_basis] there turned the other reserved tag into the
   kernel's, and acceptance depended on which side was solved first. A kernel carrying a named basis
   still converts the extent, so its window is no identity and is left to the solved arms. *)
let is_identity_window ~stride ~conv =
  stride = 1
  &&
  match conv with
  | None | Some { use_padding = true; _ } -> true
  | Some { dilation; kernel = Dim k; use_padding = false } ->
      effective_kernel_span ~dilation ~kernel_size:k.d = 1 && is_reserved_basis k.basis
  | Some { kernel = Var _ | Sym _ | Affine _ | Concat _; use_padding = false; _ } -> false

(* A kernel inferred from its window: a reserved tag can make the window the identity, whose input
   is its extent's whatever the rule says. *)
let window_kernel_bases ~stride ~dilation ~size ~(input : solved_dim) ~(extent : solved_dim) =
  window_bases ~size ~tags:[ input.basis; extent.basis ] ~derives:(fun kernel ->
      let conv = Some { dilation; kernel = Dim kernel; use_padding = false } in
      if is_identity_window ~stride ~conv then String.equal extent.basis input.basis
      else Option.equal String.equal (window_input_basis ~extent ~kernel) (Some input.basis))

type 'a dim_hashtbl = 'a Hashtbl.M(Dim_var).t [@@deriving sexp]

let dim_hashtbl () = Hashtbl.create (module Dim_var)

type print_style = Only_bases | Axis_size | Axis_number_and_size | Projection_and_size
[@@deriving equal, compare, sexp]

(* Basis is total now: [Only_bases] prints the actual tag (including [default] and [bcast_if_1]) so
   the provenance split is inspectable. In size-oriented styles the reserved (claim-free) tags
   ([default] and [bcast_if_1]) — the display analog of the old blank [None] — print bare to keep
   the common shape display readable; only user-meaningful named tags print their [tag=] prefix. *)
let basis_size_prefix basis = if is_reserved_basis basis then "" else basis ^ "="

let solved_dim_to_string style { d; basis; proj_id } =
  match style with
  | Only_bases -> basis
  | Axis_size | Axis_number_and_size -> basis_size_prefix basis ^ Int.to_string d
  | Projection_and_size ->
      let size_part = Int.to_string d in
      let proj_part = match proj_id with None -> "" | Some pid -> "p" ^ Proj_id.to_string pid in
      basis_size_prefix basis ^ size_part ^ proj_part

let sym_dim_to_string style { sym; sym_basis; sym_proj_id } =
  let ident = Ir.Indexing.symbol_ident sym.Ir.Indexing.static_symbol in
  let range = match sym.Ir.Indexing.static_range with Some r -> Int.to_string r | None -> "?" in
  match style with
  | Only_bases -> sym_basis
  | Axis_size | Axis_number_and_size -> [%string "%{basis_size_prefix sym_basis}%{ident}<=%{range}"]
  | Projection_and_size ->
      let proj_part =
        match sym_proj_id with None -> "" | Some pid -> "p" ^ Proj_id.to_string pid
      in
      [%string "%{basis_size_prefix sym_basis}%{ident}<=%{range}%{proj_part}"]

let rec dim_to_string ?var_to_string style = function
  | Dim solved_dim -> solved_dim_to_string style solved_dim
  | Sym sym_dim -> sym_dim_to_string style sym_dim
  | Var v when Option.is_some var_to_string -> (Option.value_exn var_to_string) v
  | Var { id; name = Some n } -> [%string "$%{id#Int}:%{n}"]
  | Var { id; name = None } -> "$" ^ Int.to_string id
  | Affine { stride; over; conv; stride_offset } -> (
      let over_str = dim_to_string ?var_to_string style over in
      let stride_str = if stride = 1 then over_str else Int.to_string stride ^ "*" ^ over_str in
      let offset_str =
        if stride_offset = 0 then stride_str else [%string "%{stride_str}+%{stride_offset#Int}"]
      in
      match conv with
      | None -> offset_str
      | Some { dilation; kernel; use_padding = _ } ->
          let kernel_str = dim_to_string ?var_to_string style kernel in
          let kernel_str =
            if dilation = 1 then kernel_str else Int.to_string dilation ^ "*" ^ kernel_str
          in
          [%string "conv(%{offset_str}+%{kernel_str})"])
  | Concat dims ->
      let dims_str =
        List.map dims ~f:(dim_to_string ?var_to_string style) |> String.concat ~sep:"^"
      in
      [%string "(%{dims_str})"]

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

(** A bcast specifies how axes of a single kind in a shape (i.e. the row) can adapt to other shapes.
    [Row_var v] absorbs only the middle gap between [t.beg_dims] and [t.dims]; [Broadcastable] means
    no rank-broadening slack. *)
type bcast = Row_var of row_var | Broadcastable [@@deriving equal, hash, compare, sexp, variants]

type kind = [ `Batch | `Input | `Output ] [@@deriving equal, compare, sexp, hash, variants]
type provenance_origin = { sh_id : int; kind : kind } [@@deriving sexp, compare, equal, hash]

(* List of origins, maintained as deduplicated and sorted *)
type provenance = provenance_origin list [@@deriving sexp, compare, equal, hash]

let empty_provenance = []
let provenance ~sh_id ~kind = [ { sh_id; kind } ]

(* Merge two provenances by combining and deduplicating their origins *)
let merge_provenance p1 p2 = List.dedup_and_sort ~compare:compare_provenance_origin (p1 @ p2)
(* let row_map_empty = Map.empty (module Provenance) *)

let provenance_shapes prov =
  List.map prov ~f:(fun origin -> origin.sh_id) |> List.dedup_and_sort ~compare:Int.compare

type t = { beg_dims : dim list; dims : dim list; bcast : bcast; prov : provenance }
[@@deriving equal, hash, compare, sexp]
(** The row of axes of a single kind. [beg_dims] is the outer-left anchored flank (the first axes),
    [dims] is the outer-right anchored flank (the last axes). Both flanks may be non-empty for
    closed ([Broadcastable]) and open ([Row_var]) rows alike. *)

type row = t [@@deriving equal, sexp]

let get_row_for_var prov v = { beg_dims = []; dims = []; bcast = Row_var v; prov }
let row_shapes row = provenance_shapes row.prov

let dims_basis_assoc dims =
  let f = function Var { name = Some n; _ } as d -> Some (n, d) | _ -> None in
  List.filter_map (dims.beg_dims @ dims.dims) ~f

type fixed_index_site = { spec : string; kind : kind } [@@deriving sexp, compare]
(** A fixed-index axis of an einsum spec, the only source of an [At_least_dim] bound
    ([Shape.einsum_slot_spec_to_dims_bio]). *)

type dim_constraint =
  | Unconstrained_dim
  | At_least_dim of {
      d_min : int;
      sites : fixed_index_site list;
          [@equal.ignore]
          [@compare.ignore]
          [@hash.ignore]
          [@sexp_drop_if fun _ -> true]
          [@default []]
          (** The fixed-index axes the bound came from, sorted and deduplicated. Carried on the
              bound itself, uncapped, so a diagnostic naming them does not depend on the capped
              origins (gh-ocannl-2044). They explain the bound and are no part of what it says, so
              equality, comparison, hashing and the sexp ignore them: the sexp too, since a
              [Dim_ineq]'s [from_] sexp takes part in the solver's fixpoint comparison. *)
    }
[@@deriving equal, hash, compare, sexp, variants]

type total_elems =
  | Num_elems of int
  | Range_elems of { lo : int; hi : int }
      (** The total is any value in the inclusive range [lo..hi]. Arises only from substituting the
          solved variable of a round-up [Strided_var]: the leftover slack window. When the total is
          otherwise unconstrained, late stages prefer [hi] (all blocks fully consumed). *)
  | Strided_var of { coeff : int Utils.safe_lazy; var : dim_var; denom : int; round_up : bool }
      (** The total is [(coeff * var) / denom]. With [round_up], solving for [var] from a known
          total rounds up instead of failing on non-divisibility:
          [coeff * (var - 1) < total * denom <= coeff * var]. Minted only by the [Uint4x32_to_prec]
          constraint (packed [uniform]), whose last 128-bit block may be consumed partially. *)
[@@deriving equal, hash, compare, sexp_of]

type row_constraint =
  | Unconstrained
  | Total_elems of { numerator : total_elems; divided_by : dim_var list; keep_axis : bool }
      (** The row's total number of elements is [numerator] divided by the product of the
          [divided_by] variables. [keep_axis] means the constraint comes from a data terminal whose
          preference, when the constrained rows receive no other shape information, is to keep a
          single dim-1 axis rather than close the leftover row variable to the empty row ("identity
          reshape", see gh-460); it is reset once the constraint is reduced by consuming axes or
          variables. *)
  | Exact of dim list
[@@deriving equal, hash, compare, sexp_of, variants]

type constraint_origin = {
  lhs_name : string;
  lhs_kind : kind;
  rhs_name : string;
  rhs_kind : kind;
  operation : string option;
}
[@@deriving sexp_of, compare, equal]

(** An entry implements inequalities [res ⊑ v ⊑ opnd] and/or an equality [v = solved]. [res] and
    [opnd] must be sorted using the [@@deriving compare] comparison. *)
type dim_entry =
  | Solved_dim of dim
  | Bounds_dim of {
      is_in_param : bool;
      has_uniq_constr_unless : dim_var_set option;
      res : dim_var list;
      opnd : dim_var list;
      glb : dim option;
      constr : dim_constraint;
      origin : constraint_origin list;
    }
[@@deriving sexp_of, equal]

type row_entry =
  | Solved_row of t
  | Bounds_row of {
      is_in_param : bool;
      res : row_var list;
      opnd : row_var list;
      glb : t option;
      constr : row_constraint;
      origin : constraint_origin list;
    }
[@@deriving sexp_of, equal]

type dim_env = (dim_var, dim_entry) Utils.Tree_map.t

let sexp_of_dim_env env = Utils.Tree_map.sexp_of_t sexp_of_dim_var sexp_of_dim_entry env
let find_dim env var = Utils.Tree_map.find ~compare:compare_dim_var ~key:var env
let add_dim_map env ~key ~data = Utils.Tree_map.add ~compare:compare_dim_var ~key ~data env

(** Drops the [origin] field from a [dim_entry] to avoid redundant information when storing as sexp.
*)
let drop_origin (entry : dim_entry) : dim_entry =
  match entry with
  | Solved_dim _ -> entry
  | Bounds_dim { is_in_param; has_uniq_constr_unless; res; opnd; glb; constr; origin = _ } ->
      Bounds_dim { is_in_param; has_uniq_constr_unless; res; opnd; glb; constr; origin = [] }

type row_env = (row_var, row_entry) Utils.Tree_map.t

let sexp_of_row_env env = Utils.Tree_map.sexp_of_t sexp_of_row_var sexp_of_row_entry env
let find_row env var = Utils.Tree_map.find ~compare:compare_row_var ~key:var env
let add_row_map env ~key ~data = Utils.Tree_map.add ~compare:compare_row_var ~key ~data env

type row_var_set = Set.M(Row_var).t [@@deriving sexp_of]

let row_var_set_empty : row_var_set = Set.empty (module Row_var)

let rec vars_of_dim = function
  | Dim _ | Sym _ -> Set.empty (module Dim_var)
  | Var v -> Set.singleton (module Dim_var) v
  | Affine { over; conv = None; _ } -> vars_of_dim over
  | Affine { over; conv = Some { kernel; _ }; _ } ->
      Set.union (vars_of_dim over) (vars_of_dim kernel)
  | Concat dims -> Set.union_list (module Dim_var) (List.map dims ~f:vars_of_dim)

let vars_of_dims dims = Set.union_list (module Dim_var) (List.map dims ~f:vars_of_dim)
let dim_vars_of_row (r : t) = Set.union (vars_of_dims r.beg_dims) (vars_of_dims r.dims)

(* What a solution's visit of an entry reads (gh-ocannl-1904): the dimension variables
   [s_dim_one_in_entry] / [s_dim_one_in_row_entry] substitute or react to, and the row variables
   [s_row_one_in_entry] does. *)
let dim_entry_mentions : dim_entry -> dim_var_set = function
  | Solved_dim d -> vars_of_dim d
  | Bounds_dim { res; opnd; glb; _ } ->
      Set.union_list
        (module Dim_var)
        [
          Set.of_list (module Dim_var) res;
          Set.of_list (module Dim_var) opnd;
          Option.value_map glb ~default:(Set.empty (module Dim_var)) ~f:vars_of_dim;
        ]

let row_constraint_dim_vars : row_constraint -> dim_var_set = function
  | Unconstrained -> Set.empty (module Dim_var)
  | Total_elems { numerator; divided_by; keep_axis = _ } -> (
      let vars = Set.of_list (module Dim_var) divided_by in
      match numerator with Strided_var { var; _ } -> Set.add vars var | _ -> vars)
  | Exact dims -> vars_of_dims dims

let row_entry_dim_mentions : row_entry -> dim_var_set = function
  | Solved_row r -> dim_vars_of_row r
  | Bounds_row { glb; constr; _ } ->
      Set.union (row_constraint_dim_vars constr)
        (Option.value_map glb ~default:(Set.empty (module Dim_var)) ~f:dim_vars_of_row)

let row_entry_row_mentions : row_entry -> row_var_set = function
  | Solved_row { bcast = Row_var v; _ } -> Set.singleton (module Row_var) v
  | Solved_row { bcast = Broadcastable; _ } -> row_var_set_empty
  | Bounds_row { res; opnd; glb; _ } -> (
      let vars = Set.union (Set.of_list (module Row_var) res) (Set.of_list (module Row_var) opnd) in
      match glb with Some { bcast = Row_var v; _ } -> Set.add vars v | _ -> vars)

type environment = {
  dim_env : dim_env;
  row_env : row_env;
  discardable_vars : dim_var_set;
  dims_mentioning : dim_var_set Map.M(Dim_var).t; [@sexp_drop_if fun _ -> true]
  rows_mentioning_dim : row_var_set Map.M(Dim_var).t; [@sexp_drop_if fun _ -> true]
  rows_mentioning_row : row_var_set Map.M(Row_var).t; [@sexp_drop_if fun _ -> true]
  pending_dims : dim_var_set; [@sexp_drop_if fun _ -> true]
  pending_rows : row_var_set; [@sexp_drop_if fun _ -> true]
}
[@@deriving sexp_of]
(** The environment is only in resolved wrt. variables that are solved: [v -> Solved ...] do not
    appear elsewhere in the environment. In particular, per-dim and per-row constraints might not
    have been applied.

    [discardable_vars] are here for convenience -- see {!get_inequalities}.

    The remaining fields index the entries a solution must visit (gh-ocannl-1904), so that a
    solution need not map over the whole environment: [dims_mentioning] / [rows_mentioning_dim] take
    a dimension variable to the dim / row entries mentioning it ({!dim_entry_mentions},
    {!row_entry_dim_mentions}), and [rows_mentioning_row] a row variable to the row entries
    mentioning it. They over-approximate: written on every write ({!set_dim}, {!set_row}) and never
    pruned but for a solved variable's own index, since visiting an entry that does not mention the
    solved variable leaves it unchanged -- unless it is pending. [pending_dims] / [pending_rows] are
    the entries a visit by an unrelated dimension solution can still change or make emit: every
    written entry, until a visit leaves it unchanged without emitting. That covers the
    normalizations [s_dim_one] applies whatever the variable ([Concat] sums, collapsing windows). *)

let add_mentions index ~vars ~key ~empty =
  Set.fold vars ~init:index ~f:(fun index w ->
      Map.update index w ~f:(fun s -> Set.add (Option.value s ~default:empty) key))

(** Every write of an entry goes through [set_dim] / [set_row], which keep the visit index. *)
let set_dim env ~key ~data =
  {
    env with
    dim_env = add_dim_map env.dim_env ~key ~data;
    dims_mentioning =
      add_mentions env.dims_mentioning ~vars:(dim_entry_mentions data) ~key
        ~empty:(Set.empty (module Dim_var));
    pending_dims = Set.add env.pending_dims key;
  }

let set_row env ~key ~data =
  {
    env with
    row_env = add_row_map env.row_env ~key ~data;
    rows_mentioning_dim =
      add_mentions env.rows_mentioning_dim ~vars:(row_entry_dim_mentions data) ~key
        ~empty:row_var_set_empty;
    rows_mentioning_row =
      add_mentions env.rows_mentioning_row ~vars:(row_entry_row_mentions data) ~key
        ~empty:row_var_set_empty;
    pending_rows = Set.add env.pending_rows key;
  }

let get_dim_val env var =
  match find_dim env.dim_env var with Some (Solved_dim (Dim { d; _ })) -> Some d | _ -> None

let get_dim_from_env env var =
  match find_dim env.dim_env var with Some (Solved_dim d) -> Some d | _ -> None

let get_row_from_env env var =
  match find_row env.row_env var with Some (Solved_row row) -> Some row | _ -> None

let find_dim_entry env var = find_dim env.dim_env var
let find_row_entry env var = find_row env.row_env var

type constraint_ =
  | Dim_eq of { d1 : dim; d2 : dim; origin : constraint_origin list }
  | Row_eq of { r1 : t; r2 : t; origin : constraint_origin list }
  | Dim_ineq of { res : dim; opnd : dim; from_ : Sexp.t; origin : constraint_origin list }
  | Row_ineq of { res : t; opnd : t; origin : constraint_origin list }
  | Dim_constr of { d : dim; constr : dim_constraint; origin : constraint_origin list }
  | Rows_constr of { r : t list; constr : row_constraint; origin : constraint_origin list }
  | Terminal_dim of bool * dim * constraint_origin list
  | Terminal_row of bool * t * constraint_origin list
  | Shape_row of t * constraint_origin list
[@@deriving compare, equal, sexp_of, variants]

(** Drops the [origin] field from a [constraint_] to avoid redundant information when storing as
    sexp. *)
let drop_constraint_origin (c : constraint_) : constraint_ =
  match c with
  | Dim_eq { d1; d2; origin = _ } -> Dim_eq { d1; d2; origin = [] }
  | Row_eq { r1; r2; origin = _ } -> Row_eq { r1; r2; origin = [] }
  | Dim_ineq { res; opnd; from_; origin = _ } -> Dim_ineq { res; opnd; from_; origin = [] }
  | Row_ineq { res; opnd; origin = _ } -> Row_ineq { res; opnd; origin = [] }
  | Dim_constr { d; constr; origin = _ } -> Dim_constr { d; constr; origin = [] }
  | Rows_constr { r; constr; origin = _ } -> Rows_constr { r; constr; origin = [] }
  | Terminal_dim (b, d, _) -> Terminal_dim (b, d, [])
  | Terminal_row (b, r, _) -> Terminal_row (b, r, [])
  | Shape_row (r, _) -> Shape_row (r, [])

(* Origins explain a constraint, they are no part of what it says: the solver's fixpoint compares by
   this, so that [merge_origins] and its [max_shape_error_origins] cap cannot decide how many passes
   a stage takes (gh-ocannl-2000). *)
let equal_constraint_up_to_origin (c1 : constraint_) (c2 : constraint_) =
  match (c1, c2) with
  | Dim_eq { d1; d2; origin = _ }, Dim_eq { d1 = d1'; d2 = d2'; origin = _ } ->
      equal_dim d1 d1' && equal_dim d2 d2'
  | Row_eq { r1; r2; origin = _ }, Row_eq { r1 = r1'; r2 = r2'; origin = _ } ->
      equal_row r1 r1' && equal_row r2 r2'
  | ( Dim_ineq { res; opnd; from_; origin = _ },
      Dim_ineq { res = res'; opnd = opnd'; from_ = from_'; origin = _ } ) ->
      equal_dim res res' && equal_dim opnd opnd' && Sexp.equal from_ from_'
  | Row_ineq { res; opnd; origin = _ }, Row_ineq { res = res'; opnd = opnd'; origin = _ } ->
      equal_row res res' && equal_row opnd opnd'
  | Dim_constr { d; constr; origin = _ }, Dim_constr { d = d'; constr = constr'; origin = _ } ->
      equal_dim d d' && equal_dim_constraint constr constr'
  | Rows_constr { r; constr; origin = _ }, Rows_constr { r = r'; constr = constr'; origin = _ } ->
      List.equal equal_row r r' && equal_row_constraint constr constr'
  | Terminal_dim (b, d, _), Terminal_dim (b', d', _) -> Bool.equal b b' && equal_dim d d'
  | Terminal_row (b, r, _), Terminal_row (b', r', _) -> Bool.equal b b' && equal_row r r'
  | Shape_row (r, _), Shape_row (r', _) -> equal_row r r'
  | ( ( Dim_eq _ | Row_eq _ | Dim_ineq _ | Row_ineq _ | Dim_constr _ | Rows_constr _
      | Terminal_dim _ | Terminal_row _ | Shape_row _ ),
      _ ) ->
      false

(* Stage 6 is two stages (gh-ocannl-2126). In [Stage6a], rows that resolve at their use sites take
   their bounds and the constraints settle, and no operation result closes. In [Stage6b] the results
   close, to what their arguments, closed by then, require. *)
type stage = Stage1 | Stage2 | Stage3 | Stage4 | Stage5 | Stage6a | Stage6b | Stage7
[@@deriving sexp, equal, compare]

let is_stage2_up = function Stage1 -> false | _ -> true
let is_stage3_up = function Stage1 | Stage2 -> false | _ -> true
let is_stage4_up = function Stage1 | Stage2 | Stage3 -> false | _ -> true
let is_stage5_up = function Stage5 | Stage6a | Stage6b | Stage7 -> true | _ -> false
let is_stage6_up = function Stage6a | Stage6b | Stage7 -> true | _ -> false
let is_stage6b_up = function Stage6b | Stage7 -> true | _ -> false
let is_stage7 = function Stage7 -> true | _ -> false

module Idx = Ir.Indexing

type error_trace = ..

type error_trace +=
  | Row_mismatch of t list
  | Dim_mismatch of dim list
  | Index_mismatch of Idx.axis_index list
  | Rows_constr_failed of { constr : row_constraint }
  | Constraint_failed of constraint_

let sexp_of_error_trace = function
  | Row_mismatch rs -> Sexp.List (Sexp.Atom "Row_mismatch" :: List.map rs ~f:sexp_of_t)
  | Dim_mismatch ds -> Sexp.List (Sexp.Atom "Dim_mismatch" :: List.map ds ~f:sexp_of_dim)
  | Index_mismatch idcs ->
      Sexp.List (Sexp.Atom "Index_mismatch" :: List.map idcs ~f:Idx.sexp_of_axis_index)
  | Rows_constr_failed { constr } ->
      Sexp.List (Sexp.Atom "Rows_constr_failed" :: [ sexp_of_row_constraint constr ])
  | Constraint_failed constr ->
      Sexp.List (Sexp.Atom "Constraint_failed" :: [ sexp_of_constraint_ constr ])
  | _ -> Sexp.Atom "<outdated version of sexp_of_error_trace>"

exception Shape_error of string * error_trace list [@@deriving sexp_of]

type source = Direct | Equation | Res | Opnd [@@deriving equal, sexp]

(* Which origins explain an error best (gh-ocannl-1903): those of the shape scheme being traced or
   applied, innermost scheme first -- [Shape] pushes one predicate per scheme in progress. *)
let origin_focus : (constraint_origin -> bool) list ref = ref []
[@@mutation_serial.exempt
  "the shape schemes in progress, restored by with_origin_focus's own finally"]

let with_origin_focus focus f =
  let outer = !origin_focus in
  origin_focus := focus :: outer;
  Exn.protect ~f ~finally:(fun () -> origin_focus := outer)

(* Stable: the origins of the innermost scheme in progress, then the next one's, ..., then the rest,
   each group in its given order. *)
let rank_origins origins =
  match !origin_focus with
  | [] -> origins
  | focus ->
      let unfocused = List.length focus in
      let rank o =
        Option.value_map ~default:unfocused ~f:fst (List.findi focus ~f:(fun _ p -> p o))
      in
      List.map origins ~f:(fun o -> (rank o, o))
      |> List.stable_sort ~compare:(fun (r1, _) (r2, _) -> Int.compare r1 r2)
      |> List.map ~f:snd

let rec interleave l1 l2 =
  match (l1, l2) with [], l | l, [] -> l | h1 :: t1, h2 :: t2 -> h1 :: h2 :: interleave t1 t2

(* Read once per process: [merge_origins] runs at every merge of two constraints' origins, and a
   lookup there shows in a deep graph's profile (gh-ocannl-1891). *)
let max_origins =
  lazy (Int.of_string @@ Utils.get_global_arg ~default:"20" ~arg_name:"max_shape_error_origins")

(* [o1] is the incoming constraint's origins, [o2] those an entry of the environment accumulated.
   Both are in arrival order, each one's most direct first: interleaved, so that neither crowds the
   other out, without duplicates, ranked by relevance, and capped -- the cap keeps the most relevant
   (gh-ocannl-1903; before, each was sorted by its names, so the cap kept the alphabetically
   first). *)
let merge_origins o1 o2 =
  let seen = ref [] in
  let deduplicated =
    List.filter (interleave o1 o2) ~f:(fun item ->
        if List.mem !seen item ~equal:equal_constraint_origin then false
        else (
          seen := item :: !seen;
          true))
  in
  List.take (rank_origins deduplicated) (Lazy.force max_origins)

let rec dim_to_int_exn = function
  | Dim { d; _ } -> d
  | Sym s -> sym_dim_extent_exn s
  | Var _ -> invalid_arg "dim_to_int: dim still unknown"
  | Affine _ -> invalid_arg "dim_to_int: affine dimension cannot be converted to single int"
  | Concat dims -> List.sum (module Int) dims ~f:dim_to_int_exn

(* The axis a concatenation of solved components is: their sum. *)
let sum_solved_components (solved_dims : solved_dim list) =
  let total_d = List.sum (module Int) solved_dims ~f:(fun s -> s.d) in
  (* Reserved (claim-free) tags don't constrain the concat basis; all named tags must agree, and the
     concatenated axis takes that common named tag (else [default], which a named size then refuses:
     the short-term rule of gh-ocannl-2093, explained at [unify_dim]'s fully solved [Concat]/[Dim]
     arm). *)
  let named_bases =
    List.filter_map solved_dims ~f:(fun s ->
        if is_reserved_basis s.basis then None else Some s.basis)
    |> List.dedup_and_sort ~compare:String.compare
  in
  let basis =
    match named_bases with
    | [] -> default_basis
    | [ b ] -> b
    | _ ->
        raise
        @@ Shape_error
             ( "concat: conflicting dimension bases",
               [ Dim_mismatch (List.map solved_dims ~f:(fun s -> Dim s)) ] )
  in
  { d = total_d; basis; proj_id = None }

let derived_window_basis (s : solved_dim) (k : solved_dim) =
  match window_input_basis ~extent:s ~kernel:k with
  | Some basis -> basis
  | None ->
      raise
      @@ Shape_error
           ( "convolution: conflicting dimension bases between stride and kernel",
             [ Dim_mismatch [ Dim s; Dim k ] ] )

let rec s_dim_one ?(keep_affine = false) v ~value ~in_ =
  match in_ with
  | Var v2 when equal_dim_var v v2 -> value
  | Affine { stride; over; conv; stride_offset } -> (
      let over = s_dim_one ~keep_affine v ~value ~in_:over in
      let conv =
        Option.map conv ~f:(fun { dilation; kernel; use_padding } ->
            { dilation; kernel = s_dim_one ~keep_affine v ~value ~in_:kernel; use_padding })
      in
      let result = Affine { stride; over; conv; stride_offset } in
      match result with
      | res when keep_affine -> res
      (* Before the solved arms: an identity window keeps its extent's basis (gh-ocannl-1824). *)
      | Affine { stride; over; conv; stride_offset = _ } when is_identity_window ~stride ~conv ->
          over
      | Affine
          {
            stride;
            over = Dim s;
            conv = Some { dilation; kernel = Dim k; use_padding = false };
            stride_offset = _;
          } ->
          (* input_size = stride * (output_size - 1) + effective_kernel_span *)
          let span = effective_kernel_span ~dilation ~kernel_size:k.d in
          let basis = derived_window_basis s k in
          let d = (stride * (s.d - 1)) + span in
          (* Only [window_pair_equation]'s helper has a span below 1, and what it computes is a real
             window's output: below 1, that window is wider than the extent the two share, which the
             solved arms refuse when they meet it (gh-ocannl-1972). *)
          if span < 1 && d < 1 then
            raise
            @@ Shape_error
                 ("solved dimensions for axis: incompatible stride", [ Dim_mismatch [ result ] ]);
          Dim { d; basis; proj_id = None }
      (* A padded extent is the stride's alone, whatever the kernel, as [unify_dim]'s padded arms
         and a stride-1 padded window have it (gh-ocannl-1968). *)
      | Affine
          { stride; over = Dim s; conv = None | Some { use_padding = true; _ }; stride_offset = _ }
        ->
          Dim { d = s.d * stride; basis = s.basis; proj_id = None }
      | res -> res)
  | Concat dims -> (
      let dims = List.map dims ~f:(fun d -> s_dim_one ~keep_affine v ~value ~in_:d) in
      (* Filter out zero-dimension components *)
      let dims = List.filter dims ~f:(function Dim { d = 0; _ } -> false | _ -> true) in
      match dims with
      | [] -> Dim { d = 0; basis = default_basis; proj_id = None }
      | [ single ] -> single
      | res when keep_affine -> Concat res
      | _ ->
          if List.for_all dims ~f:(function Dim _ -> true | _ -> false) then
            Dim
              (sum_solved_components
                 (List.filter_map dims ~f:(function Dim s -> Some s | _ -> None)))
          else Concat dims)
  | Dim _ | Sym _ | Var _ -> in_

(** The extent of an axis made of concrete sizes alone: a [Dim], or a [Concat] of them at any depth.
    Unlike [s_dim_one], it needs no variable to substitute: [subst_dim] sums such an axis with it.
*)
let rec concrete_extent = function
  | Dim s -> Some s
  | Concat dims ->
      Option.map (Option.all (List.map dims ~f:concrete_extent)) ~f:sum_solved_components
  | Var _ | Sym _ | Affine _ -> None

(* Helper functions for total_elems operations *)
let total_elems_to_string = function
  | Num_elems n -> Int.to_string n
  | Range_elems { lo; hi } -> [%string "%{lo#Int}..%{hi#Int}"]
  | Strided_var { coeff; var; denom; round_up } ->
      let coeff_string =
        if Utils.is_safe_val coeff then Int.to_string (Utils.safe_force coeff) else coeff.unique_id
      in
      let var_str = match var.name with Some n -> n | None -> "$" ^ Int.to_string var.id in
      let s =
        if denom = 1 then [%string "%{coeff_string}*%{var_str}"]
        else [%string "(%{coeff_string}*%{var_str})/%{denom#Int}"]
      in
      if round_up then s ^ " (round-up)" else s

let total_elems_divide t d =
  if d <= 0 then raise @@ Shape_error ([%string "Division by non-positive number: %{d#Int}"], [])
  else
    match t with
    | Num_elems n ->
        if n % d = 0 then Num_elems (n / d)
        else
          raise
          @@ Shape_error ([%string "Total_elems constraint: %{n#Int} not divisible by %{d#Int}"], [])
    | Range_elems { lo; hi } ->
        let lo' = (lo + d - 1) / d and hi' = hi / d in
        if lo' > hi' then
          raise
          @@ Shape_error
               ( [%string
                   "Total_elems range constraint: no total in %{lo#Int}..%{hi#Int} is a multiple \
                    of %{d#Int}"],
                 [] )
        else Range_elems { lo = lo'; hi = hi' }
    | Strided_var sv -> Strided_var { sv with denom = sv.denom * d }

(* Solve [coeff * var / denom = n] for [var]: exact division normally, ceiling division under
   [round_up]. Returns [None] when the exact division does not divide evenly (the caller reports the
   mismatch), or when the round-up solution would be non-positive. *)
let solve_strided_var ~coeff_val ~denom ~round_up n =
  let num = n * denom in
  if round_up then if num <= 0 then None else Some ((num + coeff_val - 1) / coeff_val)
  else if num % coeff_val = 0 then Some (num / coeff_val)
  else None

let total_elems_known_zero = function
  | Num_elems n -> n <= 0
  | Range_elems { hi; _ } -> hi <= 0
  | Strided_var { coeff; denom; _ } -> (
      denom <= 0 || match coeff.value with `Callback _ -> false | `Value v -> v <= 0)

(* Helper to remove a dimension variable from a list *)
let remove_var v vars = Utils.remove_elem ~equal:equal_dim_var v vars

(* The numerator of an exact [Strided_var] over a product that holds its own variable: [(coeff * v /
   denom) / (v * rest) = (coeff / denom) / rest]. The variable cancels, so it stays free, and what
   the constraint still states is that [rest]'s product is [coeff / denom] (a row tied by product to
   a dimension and holding that dimension: gh-ocannl-1960). [can_force] says whether the coefficient
   may be read yet. A round-up total does not cancel: it fixes [v] only up to the slack window. *)
let cancel_strided_var ~can_force ~error_trace numerator vars =
  match numerator with
  | Strided_var { coeff; var; denom; round_up = false }
    when can_force coeff && List.mem vars var ~equal:equal_dim_var ->
      let coeff = Utils.safe_force coeff in
      if coeff % denom = 0 then (Num_elems (coeff / denom), remove_var var vars)
      else
        raise
        @@ Shape_error
             ( [%string
                 "Total_elems constraint failed: the dimension cancels, leaving a fractional \
                  %{coeff#Int}/%{denom#Int} elements"],
               error_trace )
  | _ -> (numerator, vars)

(* For future flexibility *)
let dim_conjunction constr1 constr2 =
  match (constr1, constr2) with
  | Unconstrained_dim, _ -> Some ([], constr2)
  | _, Unconstrained_dim -> Some ([], constr1)
  | At_least_dim { d_min = d1; sites = s1 }, At_least_dim { d_min = d2; sites = s2 } ->
      (* Every site bounds the axis from below, so all of them stay named. *)
      Some
        ( [],
          At_least_dim
            {
              d_min = Int.max d1 d2;
              sites = List.dedup_and_sort ~compare:compare_fixed_index_site (s1 @ s2);
            } )

let rec collect_dim_factors (known, vars) = function
  | Dim { d; _ } -> Some (d * known, vars)
  | Var v -> Some (known, v :: vars)
  | Affine { stride; over; conv = Some { use_padding = true; _ } | None; stride_offset = _ } ->
      Option.map
        (collect_dim_factors (known, vars) over)
        ~f:(fun (known, vars) -> (known * stride, vars))
  | _ -> None

let collect_factors dims =
  let f acc d = Result.of_option ~error:() @@ collect_dim_factors acc d in
  Result.ok @@ List.fold_result dims ~init:(1, []) ~f

let known_dims_product dims = match collect_factors dims with Some (_, []) -> true | _ -> false

let rec row_conjunction ~prov ~origin stage constr1 constr2 =
  let elems_mismatch n1 n2 =
    raise
    @@ Shape_error
         ( [%string
             "Total_elems constraint conflict: %{total_elems_to_string n1} vs. \
              %{total_elems_to_string n2}"],
           [] )
  in
  let late = is_stage2_up stage in
  match (constr1, constr2) with
  | _ when [%equal: row_constraint] constr1 constr2 -> Some ([], constr2)
  | Unconstrained, _ -> Some ([], constr2)
  | _, Unconstrained -> Some ([], constr1)
  | ( Total_elems { numerator = n1; divided_by = vars1; keep_axis = ka1 },
      Total_elems { numerator = n2; divided_by = vars2; keep_axis = ka2 } )
    when [%equal: dim_var list]
           (List.sort ~compare:compare_dim_var vars1)
           (List.sort ~compare:compare_dim_var vars2) -> (
      match (n1, n2) with
      | n1, n2 when [%equal: total_elems] n1 n2 ->
          (* Only keep the identity-reshape preference if both sources agree on it. *)
          Some ([], Total_elems { numerator = n2; divided_by = vars2; keep_axis = ka1 && ka2 })
      | Num_elems _, Num_elems _ ->
          (* Both are solved and different - this is a mismatch *)
          elems_mismatch n1 n2
      | Num_elems n, Range_elems { lo; hi } | Range_elems { lo; hi }, Num_elems n ->
          (* The exact total must fall within the slack window; keep the exact side. *)
          if lo <= n && n <= hi then
            Some
              ( [],
                Total_elems { numerator = Num_elems n; divided_by = vars2; keep_axis = ka1 && ka2 }
              )
          else elems_mismatch n1 n2
      | Range_elems { lo = lo1; hi = hi1 }, Range_elems { lo = lo2; hi = hi2 } ->
          let lo = Int.max lo1 lo2 and hi = Int.min hi1 hi2 in
          if lo <= hi then
            Some
              ( [],
                Total_elems
                  { numerator = Range_elems { lo; hi }; divided_by = vars2; keep_axis = ka1 && ka2 }
              )
          else elems_mismatch n1 n2
      | Range_elems _, Strided_var _ | Strided_var _, Range_elems _ ->
          (* Cannot arise from user specs; wait for one side to close. *)
          None
      | Num_elems n, Strided_var { coeff; var; denom; round_up }
      | Strided_var { coeff; var; denom; round_up }, Num_elems n
        when late -> (
          (* One is solved, one is not - we can derive an equation *)
          let coeff_val = Utils.safe_force coeff in
          (* The actual value represented is coeff * var / denom = n *)
          (* So var = n * denom / coeff (rounded up for a round-up constraint) *)
          match solve_strided_var ~coeff_val ~denom ~round_up n with
          | Some d ->
              (* Under round-up the solved [var] no longer determines the total exactly, so keep the
                 [Num_elems] side; the exact case keeps [constr1] as before. *)
              let kept =
                if round_up then
                  Total_elems
                    { numerator = Num_elems n; divided_by = vars2; keep_axis = ka1 && ka2 }
                else constr1
              in
              Some ([ Dim_eq { d1 = Var var; d2 = get_default_dim ~d (); origin } ], kept)
          | None ->
              (* n * denom is not divisible by coeff - this is a mismatch *)
              elems_mismatch n1 n2)
      | Strided_var { round_up = true; _ }, Strided_var _
      | Strided_var _, Strided_var { round_up = true; _ } ->
          (* Round-up constraints only relate the variable to the total by an inequality window;
             deriving exact equations between two strided variables is unsound here. Cannot arise
             from user specs; stay conservative and wait for the totals to close. *)
          None
      | ( Strided_var { coeff = c1; var = v1; denom = d1; round_up = false },
          Strided_var { coeff = c2; var = v2; denom = d2; round_up = false } )
        when late ->
          if equal_dim_var v1 v2 then
            (* Same variable but different coefficients/denominators - check if they're equal *)
            let val1 = Utils.safe_force c1 * d2 and val2 = Utils.safe_force c2 * d1 in
            if val1 <> val2 then elems_mismatch n1 n2 else Some ([], constr2)
          else
            (* Different variables - try to derive equations *)
            (* c1 * v1 / d1 = c2 * v2 / d2 *)
            (* c1 * v1 * d2 = c2 * v2 * d1 *)
            let c1_val = Utils.safe_force c1 in
            let c2_val = Utils.safe_force c2 in
            let lhs = c1_val * d2 in
            let rhs = c2_val * d1 in
            if lhs = rhs then Some ([], constr2)
            else if lhs % rhs = 0 then
              (* lhs = k * rhs, so c1 * v1 * d2 = k * c2 * v2 * d1 *)
              (* v1 = (k * c2 * d1) / (c1 * d2) * v2 *)
              let k = lhs / rhs in
              Some
                ( [
                    Dim_eq
                      {
                        d1 = Var v1;
                        d2 = Affine { stride = k; over = Var v2; conv = None; stride_offset = 0 };
                        origin;
                      };
                  ],
                  constr2 )
            else if rhs % lhs = 0 then
              (* rhs = k * lhs, so c2 * v2 * d1 = k * c1 * v1 * d2 *)
              (* v2 = (k * c1 * d2) / (c2 * d1) * v1 *)
              let k = rhs / lhs in
              Some
                ( [
                    Dim_eq
                      {
                        d1 = Var v2;
                        d2 = Affine { stride = k; over = Var v1; conv = None; stride_offset = 0 };
                        origin;
                      };
                  ],
                  constr1 )
            else
              (* Neither divides the other - we can't make progress *)
              (* Keep both constraints - this will be resolved later *)
              None
      | _ ->
          (* We don't want to force delayed values - keep both constraints *)
          None)
  | ( Total_elems
        { numerator = Strided_var { var = v1; round_up = false; _ } as n1; divided_by = vars1; _ },
      constr2 )
    when List.mem vars1 v1 ~equal:equal_dim_var && late ->
      (* Variable appears in both numerator and denominator, they cancel out *)
      let numerator, divided_by =
        cancel_strided_var ~can_force:(fun _ -> true) ~error_trace:[] n1 vars1
      in
      row_conjunction ~prov ~origin stage
        (Total_elems { numerator; divided_by; keep_axis = false })
        constr2
  | ( constr2,
      Total_elems
        { numerator = Strided_var { var = v1; round_up = false; _ } as n1; divided_by = vars1; _ } )
    when List.mem vars1 v1 ~equal:equal_dim_var && late ->
      let numerator, divided_by =
        cancel_strided_var ~can_force:(fun _ -> true) ~error_trace:[] n1 vars1
      in
      row_conjunction ~prov ~origin stage
        (Total_elems { numerator; divided_by; keep_axis = false })
        constr2
  | ( Total_elems { numerator = n1; divided_by = vars1; _ },
      Total_elems { numerator = n2; divided_by = vars2; _ } ) ->
      (* Helper function to compute multiset difference *)
      let list_diff l1 l2 = List.fold l2 ~init:l1 ~f:(fun acc x -> remove_var x acc) in
      let vars1_only = list_diff vars1 vars2 in
      let vars2_only = list_diff vars2 vars1 in
      let extras ~keep_constr1 ?v1 ?v2 ~n1_val ~n2_val () =
        (* If we keep constr1, then it has fewer divided_by, i.e. vars1 ⊂ vars2. n1 / (product of
           vars1) = n2 / (product of vars2) Since vars1 ⊂ vars2, we have vars2 = vars1 ∪ vars2_only
           So: n1 / (product of vars1) = n2 / (product of vars1 × product of vars2_only) Thus: n1 =
           n2 / (product of vars2_only) Which means: product of vars2_only = n2 / n1 *)
        let extra_var = Option.to_list @@ if keep_constr1 then v1 else v2 in
        let num_var = if keep_constr1 then v2 else v1 in
        let diff_vars = extra_var @ if keep_constr1 then vars2_only else vars1_only in
        let n_big = if keep_constr1 then n2_val else n1_val in
        let n_small = if keep_constr1 then n1_val else n2_val in
        if n_small = 0 then
          raise @@ Shape_error ([%string "Division by zero in constraint solving"], [])
        else if n_big % n_small <> 0 then
          raise
          @@ Shape_error
               ([%string "Total_elems constraint: %{n_big#Int} not divisible by %{n_small#Int}"], [])
        else
          let quotient = n_big / n_small in
          if List.is_empty diff_vars then (
            (* No difference in variables but different numerators - this is a mismatch *)
            match num_var with
            | None -> elems_mismatch n1 n2
            | Some v ->
                if quotient <= 0 then elems_mismatch n1 n2;
                [ Dim_eq { d1 = Var v; d2 = get_default_dim ~d:quotient (); origin } ])
          else if quotient <= 0 && Option.is_none num_var then elems_mismatch n1 n2
          else if quotient = 1 && Option.is_none num_var then
            (* The difference variables must all be 1 *)
            List.map diff_vars ~f:(fun v ->
                Dim_eq { d1 = Var v; d2 = get_bcast_dim ~d:1 ~proj_id:42 (); origin })
          else
            (* The product of difference variables equals the quotient *)
            let r =
              {
                beg_dims = [];
                dims = List.map diff_vars ~f:(fun v -> Var v);
                bcast = Broadcastable;
                prov;
              }
            in
            let numerator =
              match num_var with
              | None -> Num_elems quotient
              | Some var ->
                  let coeff = Utils.{ value = `Value n_big; unique_id = Int.to_string n_big } in
                  Strided_var { coeff; var; denom = n_small; round_up = false }
            in
            [
              Rows_constr
                {
                  r = [ r ];
                  constr = Total_elems { numerator; divided_by = []; keep_axis = false };
                  origin;
                };
            ]
      in
      let lazy_extras ~keep_constr1 ~num_var ?(extra_var = []) ~coeff ~denom ~round_up () =
        (* If we keep constr1, then it has fewer divided_by, i.e. vars1 ⊂ vars2. n1 / (product of
           vars1) = n2 / (product of vars2) Since vars1 ⊂ vars2, we have vars2 = vars1 ∪ vars2_only
           So: n1 / (product of vars1) = n2 / (product of vars1 × product of vars2_only) Thus: n1 =
           n2 / (product of vars2_only) Which means: product of vars2_only = n2 / n1 *)
        let diff_vars = extra_var @ if keep_constr1 then vars2_only else vars1_only in
        (* The product of difference variables equals the quotient *)
        let r =
          {
            beg_dims = [];
            dims = List.map diff_vars ~f:(fun v -> Var v);
            bcast = Broadcastable;
            prov;
          }
        in
        let constr =
          Total_elems
            {
              (* Propagating [round_up] is sound: the derived relation [product(diff_vars) = coeff *
                 num_var / denom] inherits the same inequality window. *)
              numerator = Strided_var { coeff; var = num_var; denom; round_up };
              divided_by = [];
              keep_axis = false;
            }
        in
        [ Rows_constr { r = [ r ]; constr; origin } ]
      in
      let extras ~keep_constr1 : _ option =
        match (n1, n2) with
        | Num_elems n1_val, Num_elems n2_val -> Some (extras ~keep_constr1 ~n1_val ~n2_val ())
        | ( Strided_var { coeff = c1; var = v1; denom = d1; round_up = false },
            Strided_var { coeff = c2; var = v2; denom = d2; round_up = false } )
          when equal_dim_var v1 v2 ->
            (* c1*v1/d1 = c2*v2/d2, and v1 = v2, so c1/d1 = c2/d2 *)
            if late then
              Some
                (extras ~keep_constr1
                   ~n1_val:(Utils.safe_force c1 * d2)
                   ~n2_val:(Utils.safe_force c2 * d1)
                   ())
            else None
        | Strided_var { coeff = c1; var = v1; denom = d1; round_up }, Num_elems n2_val ->
            (* v1 from the numerator joins vars from the denominator. *)
            (* c1*v1/d1 = n2_val, so c1 = n2_val*d1/v1 *)
            if late && not round_up then
              Some (extras ~keep_constr1 ~v1 ~n1_val:(Utils.safe_force c1) ~n2_val:(n2_val * d1) ())
            else if not keep_constr1 then
              Some
                (lazy_extras ~keep_constr1 ~num_var:v1 ~coeff:c1 ~denom:(n2_val * d1) ~round_up ())
            else None
        | Num_elems n1_val, Strided_var { coeff = c2; var = v2; denom = d2; round_up } ->
            if late && not round_up then
              Some (extras ~keep_constr1 ~v2 ~n1_val:(n1_val * d2) ~n2_val:(Utils.safe_force c2) ())
            else if keep_constr1 then
              Some
                (lazy_extras ~keep_constr1 ~num_var:v2 ~coeff:c2 ~denom:(n1_val * d2) ~round_up ())
            else None
        | ( Strided_var { coeff = c1; var = v1; denom = d1; round_up = false },
            Strided_var { coeff = c2; var = v2; denom = d2; round_up = false } ) ->
            if late then
              Some
                (extras ~keep_constr1 ~v1 ~v2
                   ~n1_val:(Utils.safe_force c1 * d2)
                   ~n2_val:(Utils.safe_force c2 * d1)
                   ())
            else None
        | Strided_var { round_up = true; _ }, Strided_var _
        | Strided_var _, Strided_var { round_up = true; _ } ->
            (* Exact cross-variable derivations are unsound under a round-up window; cannot arise
               from user specs. Wait for the totals to close. *)
            None
        | Range_elems _, _ | _, Range_elems _ ->
            (* Slack windows with differing [divided_by]: conservative, wait. *)
            None
      in
      if List.is_empty vars2_only then
        Option.map ~f:(fun x -> (x, constr2)) (extras ~keep_constr1:false)
      else if List.is_empty vars1_only then
        Option.map ~f:(fun x -> (x, constr1)) (extras ~keep_constr1:true)
      else None
  | Exact dims1, Exact dims2 ->
      if List.length dims1 <> List.length dims2 then
        raise
        @@ Shape_error
             ( "Exact row constraint length mismatch",
               [
                 Row_mismatch
                   [
                     { beg_dims = []; dims = dims1; bcast = Broadcastable; prov };
                     { beg_dims = []; dims = dims2; bcast = Broadcastable; prov };
                   ];
               ] )
      else
        let eqs = List.map2_exn dims1 dims2 ~f:(fun d1 d2 -> Dim_eq { d1; d2; origin }) in
        Some (eqs, constr1)
  | Total_elems { numerator; divided_by; _ }, Exact dims
  | Exact dims, Total_elems { numerator; divided_by; _ } -> (
      match collect_factors dims with
      | None -> None (* Give up on complex cases *)
      | Some (known_product, vars) -> (
          match numerator with
          | Num_elems n ->
              if n <= 0 then
                raise @@ Shape_error ([%string "Invalid Total_elems numerator: %{n#Int}"], [])
              else if known_product = 0 then
                raise @@ Shape_error ("Exact constraint has zero dimension", [])
              else if n % known_product <> 0 then
                raise
                @@ Shape_error
                     ( [%string
                         "Total_elems numerator %{n#Int} not divisible by Exact dimensions product \
                          %{known_product#Int}"],
                       [] )
              else
                let reminder = n / known_product in
                if reminder = 1 then
                  (* reminder is 1: equate all variables on both sides to 1 *)
                  let divided_by_eqs =
                    List.map divided_by ~f:(fun v ->
                        Dim_eq { d1 = Var v; d2 = get_bcast_dim ~d:1 ~proj_id:43 (); origin })
                  in
                  let exact_vars_eqs =
                    List.map vars ~f:(fun v ->
                        Dim_eq { d1 = Var v; d2 = get_bcast_dim ~d:1 ~proj_id:44 (); origin })
                  in
                  Some (divided_by_eqs @ exact_vars_eqs, Exact dims)
                else if List.is_empty divided_by && List.length vars = 1 && reminder > 0 then
                  (* divided_by is empty and there is only one dim variable in Exact dims *)
                  let v = List.hd_exn vars in
                  Some
                    ( [ Dim_eq { d1 = Var v; d2 = get_default_dim ~d:reminder (); origin } ],
                      Exact dims )
                else if List.is_empty vars && List.length divided_by = 1 && reminder > 0 then
                  (* Exact dims contain only known dimensions and divided_by has exactly one
                     variable *)
                  let v = List.hd_exn divided_by in
                  Some
                    ( [ Dim_eq { d1 = Var v; d2 = get_default_dim ~d:reminder (); origin } ],
                      Exact dims )
                else None
          | Range_elems { lo; hi } ->
              if known_product = 0 then
                raise @@ Shape_error ("Exact constraint has zero dimension", [])
              else if List.is_empty vars && List.is_empty divided_by then
                if lo <= known_product && known_product <= hi then Some ([], Exact dims)
                else elems_mismatch numerator (Num_elems known_product)
              else None
          | Strided_var { coeff; var; denom; round_up } ->
              if known_product = 0 then
                raise @@ Shape_error ("Exact constraint has zero dimension", [])
              else if late && List.is_empty vars && List.is_empty divided_by then
                (* Exact dims contain only known dimensions and divided_by is empty *)
                (* coeff * var / denom = known_product, so var = known_product * denom / coeff
                   (rounded up for a round-up constraint) *)
                let coeff_val = Utils.safe_force coeff in
                match solve_strided_var ~coeff_val ~denom ~round_up known_product with
                | Some d ->
                    Some
                      ([ Dim_eq { d1 = Var var; d2 = get_default_dim ~d (); origin } ], Exact dims)
                | None -> elems_mismatch numerator (Num_elems known_product)
              else if round_up then
                (* Remaining derivations assume the exact relation; conservative under round-up. *)
                None
              else if
                late
                && List.mem vars var ~equal:equal_dim_var
                && List.is_empty divided_by
                && List.length vars = 1
              then
                (* Simple case: (coeff * var / denom) = known_product * var, trivially satisfied if
                   coeff = denom *)
                if Utils.safe_force coeff = denom * known_product then Some ([], Exact dims)
                else
                  raise
                  @@ Shape_error
                       ( [%string
                           "Total_elems vs. exact dims mismatch: %{Utils.safe_force coeff#Int} * \
                            %{var#Dim_var} / %{denom#Int} = %{known_product#Int} * %{var#Dim_var}"],
                         [ Dim_mismatch dims ] )
              else if late && List.is_empty divided_by && List.length vars = 1 then
                (* Handle case: length vars = 1 and it's not equal var from Strided_var (while
                   divided_by is empty), derive the coefficient between the two *)
                let single_var = List.hd_exn vars in
                let coeff_val = Utils.safe_force coeff in
                (* coeff * var / denom = known_product * single_var *)
                (* So: var = (known_product * denom / coeff) * single_var *)
                if known_product * denom % coeff_val = 0 then
                  let coefficient = known_product * denom / coeff_val in
                  if coefficient = 1 then
                    (* Simple equality *)
                    Some ([ Dim_eq { d1 = Var var; d2 = Var single_var; origin } ], Exact dims)
                  else if coefficient > 1 then
                    (* Use Affine with stride and no convolution *)
                    Some
                      ( [
                          Dim_eq
                            {
                              d1 = Var var;
                              d2 =
                                Affine
                                  {
                                    stride = coefficient;
                                    over = Var single_var;
                                    conv = None;
                                    stride_offset = 0;
                                  };
                              origin;
                            };
                        ],
                        Exact dims )
                  else
                    (* coefficient <= 0, which is invalid *)
                    elems_mismatch numerator (Num_elems known_product)
                else
                  (* Not divisible, mismatch *)
                  elems_mismatch numerator (Num_elems known_product)
              else None))

let%track5_sexp rec apply_dim_constraint ~(source : source) ~(stage : stage) (dim : dim)
    (constr : dim_constraint) env : constraint_ list * dim_constraint =
  let extras, constr =
    match (dim, constr) with
    | Dim { d; _ }, At_least_dim { d_min; _ } ->
        if d < d_min then
          raise
          @@ Shape_error
               ( "At_least_dim constraint failed, expected " ^ Int.to_string d_min,
                 [ Dim_mismatch [ dim ] ] )
        else ([], constr)
    | Affine { stride; over; conv; stride_offset = _ }, At_least_dim { d_min; sites } -> (
        if d_min <= 0 then ([], Unconstrained_dim)
        else
          let quotient = if d_min % stride = 0 then d_min / stride else (d_min / stride) + 1 in
          match conv with
          | Some { dilation; kernel = Dim { d = d_k; _ }; use_padding = false } ->
              let d_min = d_min - (dilation * d_k) in
              if d_min <= 0 then ([], Unconstrained_dim)
              else
                let quotient =
                  if d_min % stride = 0 then d_min / stride else (d_min / stride) + 1
                in
                apply_dim_constraint ~source ~stage over
                  (At_least_dim { d_min = quotient; sites })
                  env
          | _ ->
              apply_dim_constraint ~source ~stage over
                (At_least_dim { d_min = quotient; sites })
                env)
    | Var v, _ -> (
        match find_dim env.dim_env v with
        | None -> ([], constr)
        | Some (Solved_dim _) -> assert false
        | Some (Bounds_dim bounds) -> (
            match (source, constr) with
            (* If source is [Res], then [constr] (target) is [Opnd]. *)
            | Res, (Unconstrained_dim | At_least_dim { d_min = 1; _ }) -> ([], constr)
            | _ -> Option.value ~default:([], constr) @@ dim_conjunction constr bounds.constr))
    | Sym s, At_least_dim { d_min; _ } ->
        (* The materialized extent of a symbolic dimension is its declared range. *)
        if sym_dim_extent_exn s < d_min then
          raise
          @@ Shape_error
               ( "At_least_dim constraint failed, expected " ^ Int.to_string d_min,
                 [ Dim_mismatch [ dim ] ] )
        else ([], constr)
    | Concat _dims, At_least_dim _ ->
        (* FIXME: reconsider if we can make progress *)
        (* For concatenation, the constraint applies to the sum of component dimensions. We don't
           propagate constraints to components here; they'll be resolved during unification. *)
        ([], constr)
    | _, Unconstrained_dim -> ([], constr)
  in
  (extras, constr)

exception Given_up

(* Mark variables in Total_elems constraints to prevent premature guessing. When a Total_elems has
   Strided_var { var; _ } in numerator and divided_by list, we mark var with has_uniq_constr_unless
   = Some divided_by. This prevents var from being guessed to 1 unless at least one divided_by
   variable is also prevented. *)
let mark_total_elems_vars (constr : row_constraint) env : environment =
  match constr with
  | Total_elems { numerator = Strided_var { var; _ }; divided_by; _ } -> (
      let unless_set = Set.of_list (module Dim_var) divided_by in
      match find_dim env.dim_env var with
      | Some (Bounds_dim bounds) ->
          let has_uniq_constr_unless =
            match bounds.has_uniq_constr_unless with
            | None -> Some unless_set
            | Some existing -> Some (Set.union existing unless_set)
          in
          let bounds = Bounds_dim { bounds with has_uniq_constr_unless } in
          set_dim env ~key:var ~data:bounds
      | None ->
          let bounds =
            Bounds_dim
              {
                is_in_param = false;
                has_uniq_constr_unless = Some unless_set;
                res = [];
                opnd = [];
                glb = None;
                constr = Unconstrained_dim;
                origin = [];
              }
          in
          set_dim env ~key:var ~data:bounds
      | Some (Solved_dim _) -> env)
  | _ -> env

let rec dim_contains_sym = function
  | Sym _ -> true
  | Dim _ | Var _ -> false
  | Affine { over; conv = None; _ } -> dim_contains_sym over
  | Affine { over; conv = Some { kernel; _ }; _ } ->
      dim_contains_sym over || dim_contains_sym kernel
  | Concat dims -> List.exists dims ~f:dim_contains_sym

(* gh-490 v1: symbolic dimensions do not participate in total-elements arithmetic; fail fast instead
   of silently dropping (or forever deferring) the constraint. *)
let fail_if_total_elems_over_sym (dims : dim list) =
  match List.find dims ~f:dim_contains_sym with
  | Some d ->
      raise
      @@ Shape_error
           ( "Total_elems constraints are not supported with symbolic dimensions",
             [ Dim_mismatch [ d ] ] )
  | None -> ()

let reduce_row_constraint (constr : row_constraint) ~(beg_dims : dim list) ~(dims : dim list) :
    row_constraint =
  match constr with
  | Unconstrained -> Unconstrained
  | Total_elems { numerator; divided_by; keep_axis } -> (
      try
        fail_if_total_elems_over_sym (beg_dims @ dims);
        let d, vars =
          match collect_factors (beg_dims @ dims) with
          | Some (d, vars) -> (d, vars)
          | None -> raise Given_up
        in
        (* Consuming any axes, even dim-1 ones, means the constrained rows received other shape
           information: reset the identity-reshape preference. *)
        let keep_axis = keep_axis && List.is_empty beg_dims && List.is_empty dims in
        (* The consumed variables join [divided_by] even when they are already there: it is a
           multiset, so for [n / w] over [w, ..r..] the rest holds [n / (w * w)]
           (gh-ocannl-2034). *)
        let numerator = total_elems_divide numerator d in
        if total_elems_known_zero numerator then
          raise
          @@ Shape_error
               ( "reduce_row_constraint: Total_elems constraint failed, shape is too big",
                 [ Dim_mismatch (beg_dims @ dims) ] )
        else Total_elems { numerator; divided_by = divided_by @ vars; keep_axis }
      with Given_up -> Unconstrained)
  | Exact exact_dims ->
      let beg_len = List.length beg_dims in
      let dims_len = List.length dims in
      let exact_len = List.length exact_dims in
      if beg_len + dims_len > exact_len then
        raise
        @@ Shape_error
             ( "reduce_row_constraint: Exact constraint failed, shape is too long",
               [ Dim_mismatch (beg_dims @ dims) ] )
      else Exact (List.take (List.drop exact_dims beg_len) (exact_len - beg_len - dims_len))

(** Helper function to convert a list of rows to either a single row or information about multiple
    row variables. Returns Either.First with a single row if there are zero or one row variables.
    Returns Either.Second with (all_dims, row_vars) if there are multiple row variables, where
    all_dims is a concatenation of all dims and beg_dims in proper order, and row_vars is a list of
    (row_var * provenance) pairs. *)
let rows_to_row_or_vars (rows : row list) : (row, dim list * (row_var * provenance) list) Either.t =
  let rec collect_info before_dims row_vars rows =
    match rows with
    | [] -> (List.rev before_dims, List.rev row_vars)
    | row :: remaining_rows -> (
        match row.bcast with
        | Broadcastable ->
            (* Regular row: include both flanks. *)
            let new_before_dims =
              List.rev_append row.dims (List.rev_append row.beg_dims before_dims)
            in
            collect_info new_before_dims row_vars remaining_rows
        | Row_var v ->
            (* Row variable - collect it and continue *)
            let new_before_dims =
              List.rev_append row.dims (List.rev_append row.beg_dims before_dims)
            in
            let new_row_vars = (v, row.prov) :: row_vars in
            collect_info new_before_dims new_row_vars remaining_rows)
  in
  let all_dims, row_vars = collect_info [] [] rows in
  match row_vars with
  | [] ->
      (* No row variables found *)
      let first_prov =
        match
          List.find_map rows ~f:(function
            | { prov; _ }
              when List.exists prov ~f:(fun (origin : provenance_origin) ->
                       equal_kind origin.kind `Output) ->
                Some prov
            | _ -> None)
        with
        | None -> []
        | Some prov -> prov
      in
      Either.First { beg_dims = []; dims = all_dims; bcast = Broadcastable; prov = first_prov }
  | [ (v, prov) ] ->
      (* Exactly one row variable - reconstruct the proper row structure *)
      let rec reconstruct_single_var before_dims rows =
        match rows with
        | [] -> failwith "rows_to_row_or_vars: single row variable not found during reconstruction"
        | row :: remaining_rows -> (
            match row.bcast with
            | Broadcastable ->
                let acc = List.rev_append row.dims (List.rev_append row.beg_dims before_dims) in
                reconstruct_single_var acc remaining_rows
            | Row_var found_v when equal_row_var found_v v ->
                let new_beg_dims = List.rev_append before_dims row.beg_dims in
                let after_dims = List.concat_map remaining_rows ~f:(fun r -> r.beg_dims @ r.dims) in
                let new_dims = row.dims @ after_dims in
                { beg_dims = new_beg_dims; dims = new_dims; bcast = Row_var v; prov }
            | Row_var _ ->
                let acc = List.rev_append row.dims (List.rev_append row.beg_dims before_dims) in
                reconstruct_single_var acc remaining_rows)
      in
      Either.First (reconstruct_single_var [] rows)
  | _ ->
      (* Multiple row variables *)
      Either.Second (all_dims, row_vars)

let row_of_var v prov = { beg_dims = []; dims = []; bcast = Row_var v; prov }

let unsolved_constraints env =
  let dims = Utils.Tree_map.to_alist env.dim_env in
  let rows = Utils.Tree_map.to_alist env.row_env in
  let dims =
    List.filter_map dims ~f:(fun (var, entry) ->
        match entry with
        | Solved_dim _ -> None
        | Bounds_dim { constr = Unconstrained_dim; _ } -> None
        | Bounds_dim { constr; origin; _ } -> Some (Dim_constr { d = Var var; constr; origin }))
  in
  let rows =
    List.filter_map rows ~f:(fun (var, entry) ->
        match entry with
        | Solved_row _ -> None
        | Bounds_row { constr = Unconstrained; _ } -> None
        | Bounds_row { constr; origin; _ } ->
            Some (Rows_constr { r = [ row_of_var var [] ]; constr; origin }))
  in
  dims @ rows

let check_empty_row ~origin r =
  if not (List.is_empty r.dims) then
    raise @@ Shape_error ("check_empty_row: row is not empty", [ Row_mismatch [ r ] ]);
  match r.bcast with
  | Broadcastable ->
      if List.is_empty r.beg_dims then []
      else
        raise @@ Shape_error ("check_empty_row: row is not empty (beg_dims)", [ Row_mismatch [ r ] ])
  | Row_var v ->
      if List.is_empty r.beg_dims then
        [
          Row_eq
            {
              r1 = row_of_var v r.prov;
              r2 = { beg_dims = []; dims = []; bcast = Broadcastable; prov = r.prov };
              origin;
            };
        ]
      else raise @@ Shape_error ("check_empty_row: row is not empty", [ Row_mismatch [ r ] ])

let s_dim_one_in_entry v ~value (in_ : dim_entry) : _ * dim_entry =
  (* Every entry of the environment passes here at each dimension solution: build the provenance
     only for an entry that yields an inequality (gh-ocannl-1802). *)
  let from_ = lazy ([%sexp_of: dim_var * dim_entry] (v, drop_origin in_)) in
  match in_ with
  | Solved_dim in_ -> ([], Solved_dim (s_dim_one v ~value ~in_))
  | Bounds_dim { is_in_param; has_uniq_constr_unless; res; opnd; glb; constr; origin } ->
      let find_v side = List.partition_tf side ~f:(equal_dim_var v) in
      let res_v, res = find_v res in
      let opnd_v, opnd = find_v opnd in
      let ineqs0 =
        match (opnd_v, glb) with
        | _ :: _, Some glb ->
            [ Dim_ineq { res = glb; opnd = value; from_ = Lazy.force from_; origin } ]
        | _ -> []
      in
      let ineqs1 =
        if List.is_empty opnd_v then []
        else
          List.map res ~f:(fun res ->
              Dim_ineq { res = Var res; opnd = value; from_ = Lazy.force from_; origin })
      in
      let ineqs2 =
        if List.is_empty res_v then []
        else
          List.map opnd ~f:(fun opnd ->
              Dim_ineq { res = value; opnd = Var opnd; from_ = Lazy.force from_; origin })
      in
      ( ineqs0 @ ineqs1 @ ineqs2,
        Bounds_dim
          {
            is_in_param;
            has_uniq_constr_unless;
            res;
            opnd;
            glb = Option.map glb ~f:(fun in_ -> s_dim_one v ~value ~in_);
            constr;
            origin;
          } )

let s_dim_one_in_row v ~value in_ =
  {
    in_ with
    beg_dims = List.map in_.beg_dims ~f:(fun in_ -> s_dim_one v ~value ~in_);
    dims = List.map in_.dims ~f:(fun in_ -> s_dim_one v ~value ~in_);
  }

let reapply_rows_constr = ref false
[@@mutation_serial.exempt
  "a transient flag of one constraint substitution, cleared before it returns or raises"]

let subst_row_constraint_unprotected ~subst_in_dim ~get_dim_val stage constr =
  let subst_total_elems_divided_by numerator divided_by ~keep_axis =
    let substituted_divided_by = List.map divided_by ~f:(fun v -> subst_in_dim (Var v)) in
    fail_if_total_elems_over_sym substituted_divided_by;
    let untouched =
      List.for_all2_exn divided_by substituted_divided_by ~f:(fun v d -> equal_dim (Var v) d)
    in
    match collect_factors substituted_divided_by with
    (* A solution that reaches no divisor says nothing new: reapplying here made every such row
       re-emit at every dimension solution (gh-ocannl-1977). *)
    | _ when untouched -> Total_elems { numerator; divided_by; keep_axis }
    | Some (known_product, residual_vars) ->
        reapply_rows_constr := true;
        Total_elems
          {
            numerator = total_elems_divide numerator known_product;
            divided_by = List.rev residual_vars;
            keep_axis;
          }
    | None ->
        (* Fall back to preserving the original constraint *)
        Total_elems { numerator; divided_by; keep_axis }
  in
  match constr with
  | Total_elems { numerator = Strided_var { coeff; var; denom; round_up }; divided_by; keep_axis }
    when is_stage2_up stage && Option.is_some (get_dim_val var) ->
      let dim = Option.value_exn (get_dim_val var) in
      let coeff_val = Utils.safe_force coeff in
      let tot = coeff_val * dim in
      reapply_rows_constr := true;
      if round_up then
        (* The solved variable determines the total only up to the slack window [coeff*(var-1) <
           total*denom <= coeff*var]; an otherwise-unconstrained total defaults to [hi] (all blocks
           fully consumed) at late stages. *)
        let lo = (coeff_val * (dim - 1) / denom) + 1 and hi = tot / denom in
        subst_total_elems_divided_by (Range_elems { lo; hi }) divided_by ~keep_axis
      else if tot % denom = 0 then
        subst_total_elems_divided_by (Num_elems (tot / denom)) divided_by ~keep_axis
      else
        raise
        @@ Shape_error
             ( [%string
                 "Total_elems constraint: shape cannot be strided, %{tot#Int} not divisible by \
                  %{denom#Int}"],
               [ Rows_constr_failed { constr } ] )
  | Total_elems { numerator = Strided_var { var; _ } as numerator; divided_by; keep_axis }
    when match subst_in_dim (Var var) with Affine _ | Concat _ -> true | _ -> false ->
      (* The tie's variable is solved to a window or a concatenation whose extent the substitution
         left open (a part unsolved, or [keep_affine] keeping the parts): one variable times a
         coefficient cannot carry it. The numerator keeps the solved variable, as the stage-1 [Dim]
         arm below does, and only the divisors are substituted: every later application substitutes
         through the environment, which sums a concrete extent into a [Dim]; and a closed row's
         total reaches the variable as an equation, which its solution carries to the parts
         (gh-ocannl-2086). *)
      subst_total_elems_divided_by numerator divided_by ~keep_axis
  | Total_elems { numerator = Strided_var { coeff; var; denom; round_up }; divided_by; keep_axis }
    when not (equal_dim (Var var) (subst_in_dim (Var var))) -> (
      reapply_rows_constr := true;
      match subst_in_dim (Var var) with
      | Dim { d; _ } as value when is_stage2_up stage ->
          (* Stage 2+: Replace (coeff * v / denom) with (coeff * d / denom) *)
          let coeff_val = Utils.safe_force coeff in
          let new_num = coeff_val * d in
          if round_up then
            let lo = (coeff_val * (d - 1) / denom) + 1 and hi = new_num / denom in
            Total_elems { numerator = Range_elems { lo; hi }; divided_by; keep_axis }
          else if new_num % denom = 0 then
            Total_elems { numerator = Num_elems (new_num / denom); divided_by; keep_axis }
          else
            raise
            @@ Shape_error
                 ( "s_dim_one_in_row_constr: Total_elems constraint failed: dimension is not \
                    divisible",
                   [ Dim_mismatch [ value ] ] )
      | Dim _ ->
          (* Stage 1: Don't force coeff yet, keep the constraint as-is *)
          Total_elems
            { numerator = Strided_var { coeff; var; denom; round_up }; divided_by; keep_axis }
      | Var v' ->
          Total_elems
            { numerator = Strided_var { coeff; var = v'; denom; round_up }; divided_by; keep_axis }
      | Sym _ as value ->
          (* gh-490 v1: symbolic dimensions do not participate in total-elements arithmetic. *)
          raise
          @@ Shape_error
               ( "Total_elems constraints are not supported with symbolic dimensions",
                 [ Dim_mismatch [ value ] ] )
      | Affine _ | Concat _ -> (* The deferring arm above takes these. *) assert false)
  | Total_elems { numerator; divided_by; keep_axis } ->
      subst_total_elems_divided_by numerator divided_by ~keep_axis
  | Exact dims ->
      (* The constraint update does not affect its applicability, so we don't need to reapply it. *)
      Exact (List.map dims ~f:subst_in_dim)
  | Unconstrained -> constr

(* All the [reapply_rows_constr := true] sites live in [subst_row_constraint_unprotected]; an
   exception escaping after one of them (e.g. the Total_elems-over-symbolic-dimension rejection)
   must not leave the global flag dirty for a subsequent solver call (the [s_dim_one_in_row_entry]
   assert), notably when the caller catches the [Shape_error] and continues. *)
let subst_row_constraint_impl ~subst_in_dim ~get_dim_val stage constr =
  try subst_row_constraint_unprotected ~subst_in_dim ~get_dim_val stage constr
  with e ->
    reapply_rows_constr := false;
    raise e

let s_dim_one_in_row_constr stage v ~value constr =
  let get_dim_val v' =
    if equal_dim_var v v' then match value with Dim { d; _ } -> Some d | _ -> None else None
  in
  subst_row_constraint_impl
    ~subst_in_dim:(fun in_ -> s_dim_one ~keep_affine:true v ~value ~in_)
    ~get_dim_val stage constr

let ineqs_from_reapply_rows_constr = ref []
[@@mutation_serial.exempt
  "a transient carrier within one solver pass, drained by unify_dim; reset_transient clears it at \
   a rewind"]

let reset_transient () =
  reapply_rows_constr := false;
  ineqs_from_reapply_rows_constr := []

let s_dim_one_in_row_entry stage v ~value ~key ~data =
  assert (not !reapply_rows_constr);
  let result =
    match data with
    | Solved_row in_ -> Solved_row (s_dim_one_in_row v ~value in_)
    | Bounds_row { is_in_param; res; opnd; glb; constr; origin } ->
        let constr = s_dim_one_in_row_constr stage v ~value constr in
        if !reapply_rows_constr then
          ineqs_from_reapply_rows_constr :=
            Rows_constr { r = [ row_of_var key [] ]; constr; origin }
            :: !ineqs_from_reapply_rows_constr;
        reapply_rows_constr := false;
        let glb = Option.map glb ~f:(s_dim_one_in_row v ~value) in
        Bounds_row { is_in_param; res; opnd; glb; constr; origin }
  in
  result

(* A concatenated axis of concrete components is a plain axis of their sum whatever brought it
   there: substituting a variable, or a spec with none to substitute ([2^3 =>], gh-ocannl-2019).
   Under [keep_affine] the parts stay, for the projections to index. *)
let subst_dim ?(keep_affine = false) env dim =
  let vars = vars_of_dim dim in
  let dim =
    List.fold (Set.elements vars) ~init:dim ~f:(fun acc v ->
        match find_dim env.dim_env v with
        | Some (Solved_dim d) -> s_dim_one ~keep_affine v ~value:d ~in_:acc
        | _ -> acc)
  in
  match dim with
  | Concat _ when not keep_affine ->
      Option.value_map (concrete_extent dim) ~default:dim ~f:(fun s -> Dim s)
  | _ -> dim

(** Substitute [value] for row variable [v] inside [in_], uniformly composing both flanks.

    If [in_]'s middle row variable matches [v], the result is

    [{ beg_dims = in_.beg_dims @ value.beg_dims; dims = value.dims @ in_.dims; bcast = value.bcast
     }]

    which preserves [in_]'s pinned leading flank whether [value] is closed or open. *)
let s_row_one v ~value:{ beg_dims = value_beg; dims = more_dims; bcast; prov = _ } ~in_ =
  match in_ with
  | { beg_dims; dims; bcast = Row_var v2; prov } when equal_row_var v v2 ->
      { beg_dims = beg_dims @ value_beg; dims = more_dims @ dims; bcast; prov }
  | _ -> in_

let s_row_one_in_row_constr _v ~value:_ ~in_ =
  match in_ with Unconstrained | Total_elems _ | Exact _ -> in_

let transitive_row_emissions = ref false
[@@mutation_serial.exempt "a test's reference switch, set around one call"]

let with_transitive_row_emissions f =
  let saved = !transitive_row_emissions in
  transitive_row_emissions := true;
  Exn.protect ~f ~finally:(fun () -> transitive_row_emissions := saved)

let is_bare_row_var : row -> bool = function
  | { beg_dims = []; dims = []; bcast = Row_var _; prov = _ } -> true
  | _ -> false

(* The visited entry [e] is bounded by [v], and [v := value]. What this owes through [e], beyond
   [v]'s own bounds, which [unify_row] re-emits (gh-ocannl-1977):

   - [e]'s GLB reaches [value]. - A [value] with axes reaches the far neighbours: [value ⊑ opnd]
   carries the lower bound on to them, as [solve_row_ineq] forwards a recorded GLB along
   variable-to-variable bounds (gh-ocannl-2126). [res ⊑ value] follows, for [e ⊑ value] solves [e],
   but it costs little. - A bare variable has no axes to carry. [r ⊑ value] is the path [r ⊑ e ⊑
   value], which stays in the environment, so emitting it would only store the transitive closure:
   quadratic pairs in a residual stack, each re-emitted per solution. *)
let s_row_one_in_entry (v : row_var) ~(value : row) ~(in_ : row_entry) :
    constraint_ list * row_entry =
  match in_ with
  | Solved_row in_ -> ([], Solved_row (s_row_one v ~value ~in_))
  | Bounds_row { is_in_param; res; opnd; glb; constr; origin } ->
      let find_v side = List.partition_tf side ~f:(equal_row_var v) in
      let res_v, res = find_v res in
      let opnd_v, opnd = find_v opnd in
      let ineqs0 =
        match (opnd_v, glb) with
        | _ :: _, Some glb -> [ Row_ineq { res = glb; opnd = value; origin } ]
        | _ -> []
      in
      let past_the_entry = !transitive_row_emissions || not (is_bare_row_var value) in
      let ineqs1 =
        if List.is_empty opnd_v || not past_the_entry then []
        else
          List.map res ~f:(fun res ->
              Row_ineq { res = row_of_var res value.prov; opnd = value; origin })
      in
      let ineqs2 =
        if List.is_empty res_v || not past_the_entry then []
        else
          List.map opnd ~f:(fun opnd ->
              Row_ineq { opnd = row_of_var opnd value.prov; res = value; origin })
      in
      let constr = s_row_one_in_row_constr v ~value ~in_:constr in
      let glb = Option.map glb ~f:(fun in_ -> s_row_one v ~value ~in_) in
      (ineqs0 @ ineqs1 @ ineqs2, Bounds_row { is_in_param; res; opnd; glb; constr; origin })

(* The occurs check's refusal (gh-ocannl-2036): [what] names the relation that no finite value of
   the row variable both rows share satisfies, [surplus] by how many axes one side exceeds the
   other. Only an equation gets the einsum hint: an inequality whose result has the surplus is
   satisfiable, and [solve_row_ineq] defers it instead. *)
let self_reference_error ?(equation = true) ~rows ~surplus ~what () =
  let by =
    if surplus > 0 then Printf.sprintf " (%d more, around the same row variable)" surplus else ""
  in
  let hint =
    if equation then
      ". Einsum equates each operand with its slot and does not broadcast it, so one tensor \
       filling two slots that put different numbers of axes around the same ..r.. is refused; to \
       contract a tensor with its own broadcast, pass one side through a pointwise operation (e.g. \
       t *. x)"
    else ""
  in
  Shape_error
    ( "Infinite number of axes by self-reference: " ^ what ^ by
      ^ ", which no finite shape satisfies" ^ hint,
      [ Row_mismatch rows ] )

let subst_row env ({ beg_dims; dims; bcast; prov } : t) : t =
  let s_dims = List.map ~f:(subst_dim env) in
  let beg_dims = s_dims beg_dims in
  let dims = s_dims dims in
  let default = { beg_dims; dims; bcast; prov } in
  match bcast with
  | Broadcastable -> default
  | Row_var v -> (
      match find_row env.row_env v with
      | None | Some (Bounds_row _) -> default
      | Some (Solved_row { beg_dims = []; dims = []; bcast = Row_var v2; _ })
        when equal_row_var v v2 ->
          default
      | Some (Solved_row ({ bcast = Row_var v2; _ } as r2)) when equal_row_var v v2 ->
          raise
          @@ self_reference_error ~rows:[ default; r2 ]
               ~surplus:(List.length r2.beg_dims + List.length r2.dims)
               ~what:"a row would have to equal itself with more axes" ()
      | Some (Solved_row { beg_dims = more_beg; dims = more_dims; bcast; prov = _ }) -> (
          (* Note: we assume env is idempotent (solved wrt. equalities). *)
          let more_beg = s_dims more_beg in
          let more_dims = s_dims more_dims in
          match bcast with
          | Broadcastable ->
              { beg_dims = beg_dims @ more_beg; dims = more_dims @ dims; bcast; prov }
          | Row_var _ -> { beg_dims = beg_dims @ more_beg; dims = more_dims @ dims; bcast; prov }))

let subst_row_constraint stage env constr =
  subst_row_constraint_impl ~subst_in_dim:(subst_dim env) ~get_dim_val:(get_dim_val env) stage
    constr

let%track5_sexp rec apply_rows_constraint ~depth ~stage origin (rows : row list)
    (constr : row_constraint) env : constraint_ list * _ =
  if depth > 16 then ([], env)
  else
    (* Mark variables in Total_elems to prevent premature guessing *)
    let env = mark_total_elems_vars constr env in
    match rows_to_row_or_vars rows with
    | Either.First single_row -> apply_row_constraint ~depth stage origin single_row constr env
    | Either.Second (all_dims, row_vars) -> (
        match constr with
        | Exact dims when List.length dims < List.length all_dims ->
            (* Case 1: Exact dims has fewer axes than all_dims - raise mismatch *)
            raise
            @@ Shape_error
                 ("apply_rows_constraint: Exact constraint has too few axes", [ Row_mismatch rows ])
        | Exact dims when List.length dims = List.length all_dims ->
            (* Case 2: Exact dims has same length as all_dims - derive pairwise equations *)
            let dim_eqs = List.map2_exn dims all_dims ~f:(fun d1 d2 -> Dim_eq { d1; d2; origin }) in
            let row_eqs =
              List.map row_vars ~f:(fun (v, prov) ->
                  Row_eq
                    {
                      r1 = row_of_var v prov;
                      r2 = { beg_dims = []; dims = []; bcast = Broadcastable; prov };
                      origin;
                    })
            in
            (dim_eqs @ row_eqs, env)
        | Total_elems { numerator = Num_elems n; divided_by; keep_axis } -> (
            (* Case 3: Total_elems with known numerator *)
            fail_if_total_elems_over_sym all_dims;
            match collect_factors all_dims with
            | None ->
                ([ Rows_constr { r = rows; constr; origin } ], env) (* Give up on complex cases *)
            | Some (known_product, product_vars) ->
                (* Move divided_by variables to the other side by combining with product_vars *)
                let all_product_vars = product_vars @ divided_by in
                if n % known_product <> 0 then
                  raise
                  @@ Shape_error
                       ( [%string
                           "Total_elems constraint: %{n#Int} not divisible by known product \
                            %{known_product#Int}"],
                         [] )
                else if n = known_product then
                  (* Equate all product vars to d=1 and add Total_elems 1 for each row var *)
                  let var_eqs =
                    List.map all_product_vars ~f:(fun v ->
                        Dim_eq { d1 = Var v; d2 = get_bcast_dim ~d:1 ~proj_id:45 (); origin })
                  in
                  (* The identity-reshape preference survives only if no axes were consumed, and
                     goes to a single row variable: the sole one, or the last output-kind one
                     (mirroring the preference in [eliminate_rows_constraint]). *)
                  let keep_axis = keep_axis && List.is_empty all_dims in
                  let keep_var =
                    if not keep_axis then None
                    else
                      match row_vars with
                      | [ (v, _) ] -> Some v
                      | _ ->
                          Option.map ~f:(fun (_, (v, _)) -> v)
                          @@ List.findi (List.rev row_vars) ~f:(fun _ (_, prov) ->
                              List.exists prov ~f:(fun (o : provenance_origin) ->
                                  equal_kind o.kind `Output))
                  in
                  let row_constrs =
                    List.map row_vars ~f:(fun (v, id) ->
                        Rows_constr
                          {
                            r = [ row_of_var v id ];
                            constr =
                              Total_elems
                                {
                                  numerator = Num_elems 1;
                                  divided_by = [];
                                  keep_axis =
                                    Option.value_map keep_var ~default:false ~f:(equal_row_var v);
                                };
                            origin;
                          })
                  in
                  (var_eqs @ row_constrs, env)
                else
                  (* Cannot deduce no_further_axes, return unchanged *)
                  ([ Rows_constr { r = rows; constr; origin } ], env))
        | Exact [ single_dim ] -> (
            (* Handle exact single dimension constraint, preferring non-empty output rows. *)
            match List.rev rows with
            | { beg_dims = []; dims = []; bcast = Broadcastable; prov = _ } :: more_rows ->
                apply_rows_constraint ~depth:(depth + 1) ~stage origin (List.rev more_rows) constr
                  env
            | { beg_dims = []; dims = []; bcast = Row_var v; prov } :: more_rows
              when List.exists prov ~f:(fun (origin : provenance_origin) ->
                       equal_kind origin.kind `Input) ->
                let more_eqs, env =
                  apply_rows_constraint ~depth:(depth + 1) ~stage origin (List.rev more_rows) constr
                    env
                in
                ( Row_eq
                    {
                      r1 = row_of_var v prov;
                      r2 = { beg_dims = []; dims = []; bcast = Broadcastable; prov };
                      origin;
                    }
                  :: more_eqs,
                  env )
            | { beg_dims = []; dims = []; bcast = Row_var v; prov } :: more_rows
              when List.exists prov ~f:(fun (origin : provenance_origin) ->
                       equal_kind origin.kind `Output) ->
                ( Row_eq
                    {
                      r1 = row_of_var v prov;
                      r2 = { beg_dims = []; dims = [ single_dim ]; bcast = Broadcastable; prov };
                      origin;
                    }
                  :: List.concat_map ~f:(check_empty_row ~origin) more_rows,
                  env )
            | { dims = _; bcast = Row_var _; prov; _ } :: _
              when List.exists prov ~f:(fun (origin : provenance_origin) ->
                       equal_kind origin.kind `Output) ->
                raise
                @@ Shape_error
                     ( "apply_rows_constraint: shape too big (non-empty output leading flank)",
                       [ Row_mismatch rows ] )
            | _ ->
                raise @@ Shape_error ("apply_rows_constraint: shape too big", [ Row_mismatch rows ])
            )
        | _ -> ([ Rows_constr { r = rows; constr; origin } ], env))

and apply_row_constraint ~depth stage origin (r : row) (constr : row_constraint) env :
    constraint_ list * _ =
  if depth > 16 then ([], env)
  else if is_unconstrained constr then ([], env)
  else
    (* Mark variables in Total_elems to prevent premature guessing *)
    let env = mark_total_elems_vars constr env in
    let constr = subst_row_constraint stage env constr in
    reapply_rows_constr := false;
    let reduce constr ~beg_dims ~dims =
      try reduce_row_constraint constr ~beg_dims ~dims
      with Shape_error (s, trace) -> raise @@ Shape_error (s, Row_mismatch [ r ] :: trace)
    in
    let extras, constr, env, stored, updated =
      match r with
      | { bcast = Broadcastable; _ } -> ([], constr, env, false, false)
      | { bcast = Row_var v; beg_dims; dims; _ } -> (
          match find_row env.row_env v with
          | Some (Solved_row _) -> ([], constr, env, false, false)
          | None ->
              ( [],
                constr,
                (let constr = reduce constr ~beg_dims ~dims in
                 set_row env ~key:v
                   ~data:
                     (Bounds_row
                        { is_in_param = false; constr; res = []; opnd = []; glb = None; origin })),
                true,
                false )
          | Some (Bounds_row ({ constr = Unconstrained; _ } as bounds)) ->
              ( [],
                constr,
                set_row env ~key:v
                  ~data:(Bounds_row { bounds with constr = reduce constr ~beg_dims ~dims }),
                true,
                false )
          | Some (Bounds_row bounds) -> (
              let origin = merge_origins origin bounds.origin in
              match
                row_conjunction ~prov:r.prov ~origin stage (reduce constr ~beg_dims ~dims)
                  bounds.constr
              with
              | None -> ([], constr, env, false, false)
              | Some (extras, constr) ->
                  if phys_equal constr bounds.constr then (extras, constr, env, true, false)
                  else
                    ( extras,
                      constr,
                      set_row env ~key:v ~data:(Bounds_row { bounds with constr }),
                      true,
                      true )))
    in
    match (r, constr) with
    | _ when stored && not updated -> (extras, env)
    | _, Unconstrained -> assert false
    | _, Total_elems { numerator = Strided_var { coeff; var; denom; round_up }; divided_by = []; _ }
      when is_stage2_up stage && Option.is_some (get_dim_val env var) ->
        (* The variable is already solved: under round-up the total is only determined up to the
           slack window, otherwise it is exactly [coeff * var / denom]. *)
        let v = Option.value_exn (get_dim_val env var) in
        let coeff_val = Utils.safe_force coeff in
        let numerator =
          if round_up then
            Range_elems { lo = (coeff_val * (v - 1) / denom) + 1; hi = coeff_val * v / denom }
          else Num_elems (coeff_val * v / denom)
        in
        apply_row_constraint ~depth:(depth + 1) stage origin r
          (Total_elems { numerator; divided_by = []; keep_axis = false })
          env
    | ( { beg_dims; dims; bcast = Broadcastable; _ },
        Total_elems { numerator = Strided_var { coeff; var; denom; round_up }; divided_by = []; _ }
      )
      when is_stage2_up stage && known_dims_product (beg_dims @ dims) -> (
        (* [beg_dims] participate in the total: block/stack rows carry leading dims there. *)
        let (d : int), _ = Option.value_exn (collect_factors (beg_dims @ dims)) in
        let coeff : int = Utils.safe_force coeff in
        match solve_strided_var ~coeff_val:coeff ~denom ~round_up d with
        | Some v -> (Dim_eq { d1 = Var var; d2 = get_default_dim ~d:v (); origin } :: extras, env)
        | None ->
            raise
            @@ Shape_error
                 ( [%string
                     "apply_row_constraint: Total_elems constraint failed: %{denom*d#Int} not \
                      divisible by %{coeff#Int}"],
                   [ Row_mismatch [ r ] ] ))
    | { beg_dims; dims; bcast = Broadcastable; _ }, Total_elems { numerator; divided_by; _ } -> (
        try
          (* [beg_dims] participate in the total: block/stack rows carry leading dims there (gh-509:
             a slack-window re-check against a block row must divide by the full product). *)
          let all_dims = beg_dims @ dims in
          fail_if_total_elems_over_sym all_dims;
          let d, vars =
            match collect_factors all_dims with
            | Some (d, vars) -> (d, vars)
            | None -> raise Given_up
          in
          let numerator = total_elems_divide numerator d in
          if total_elems_known_zero numerator then
            raise
            @@ Shape_error
                 ( "apply_row_constraint: Total_elems constraint failed, shape is too big",
                   [ Dim_mismatch dims ] );
          (* The row's closed part divided the numerator, so the row's own variables and
             [divided_by] sit on the same side of the equation. *)
          let can_force coeff = Utils.is_safe_val coeff || is_stage2_up stage in
          let cancel = cancel_strided_var ~can_force ~error_trace:[ Row_mismatch [ r ] ] in
          let strided = numerator in
          let numerator, vars = cancel numerator vars in
          let numerator, divided_by = cancel numerator divided_by in
          let env =
            match (strided, numerator) with
            | Strided_var { var; _ }, Num_elems _ ->
                (* Its own factor cancelled, so this constraint no longer holds [var] back from a
                   guess. *)
                mark_total_elems_vars
                  (Total_elems { numerator = strided; divided_by = [ var ]; keep_axis = false })
                  env
            | _ -> env
          in
          match (numerator, vars, divided_by) with
          | Num_elems 1, [], [] -> (extras, env)
          | Num_elems _, [], [] ->
              raise
              @@ Shape_error
                   ( "apply_row_constraint: Total_elems constraint failed, shape is too small",
                     [ Row_mismatch [ r ] ] )
          | Range_elems { lo; hi }, [], [] ->
              (* Slack window: after dividing by the closed rows' product, the remaining total is 1
                 iff the actual total falls within the window. *)
              if lo <= 1 && 1 <= hi then (extras, env)
              else
                let size = if hi < 1 then "big" else "small" in
                raise
                @@ Shape_error
                     ( "apply_row_constraint: Total_elems slack window failed, shape is too " ^ size,
                       [ Row_mismatch [ r ] ] )
          | Num_elems n, [ v ], [] | Num_elems n, [], [ v ] ->
              (Dim_eq { d1 = Var v; d2 = get_default_dim ~d:n (); origin } :: extras, env)
          | Num_elems 1, vs1, vs2 ->
              ( List.map
                  ~f:(fun v ->
                    Dim_eq { d1 = Var v; d2 = get_bcast_dim ~d:1 ~proj_id:46 (); origin })
                  (vs1 @ vs2)
                @ extras,
                env )
          | _ ->
              if stored then (extras, env)
              else
                ( Rows_constr { r = [ r ]; constr; origin } :: extras,
                  env (* Wait for more shape inference. *) )
        with Given_up ->
          if stored then (extras, env)
          else
            ( Rows_constr { r = [ r ]; constr; origin } :: extras,
              env (* Wait for more shape inference. *) ))
    | { beg_dims; dims; bcast = Row_var v; _ }, Exact exact_dims
      when List.length beg_dims + List.length dims >= List.length exact_dims ->
        assert (not stored);
        if List.length dims + List.length beg_dims > List.length exact_dims then
          raise
          @@ Shape_error
               ( "apply_row_constraint: Exact constraint failed, shape is too long",
                 [ Row_mismatch [ r ] ] );
        ( Row_eq
            {
              r1 = row_of_var v r.prov;
              r2 = { beg_dims = []; dims = []; bcast = Broadcastable; prov = r.prov };
              origin;
            }
          :: List.map2_exn exact_dims (beg_dims @ dims) ~f:(fun d1 d2 -> Dim_eq { d1; d2; origin })
          @ extras,
          env )
    | ( { bcast = Row_var v; _ },
        Total_elems { numerator = Strided_var { coeff; var = _; denom; _ }; divided_by = []; _ } )
      when is_stage2_up stage -> (
        (* Check if we have a GLB and if it meets our conditions *)
        match find_row env.row_env v with
        | Some (Bounds_row { glb = Some ({ dims = glb_dims; bcast = Broadcastable; _ } as glb); _ })
          when Utils.is_safe_val coeff && Utils.safe_force coeff > denom -> (
            (* Check if all GLB dimensions are known *)
            match collect_factors glb_dims with
            | Some (_known_product, []) ->
                (* Check if GLB has at most one dimension greater than 1 *)
                let greater_than_one =
                  List.filter glb_dims ~f:(function Dim { d; _ } -> d > 1 | _ -> false)
                in
                if List.length greater_than_one <= 1 then
                  (Row_eq { r1 = row_of_var v r.prov; r2 = glb; origin } :: extras, env)
                else if stored then (extras, env)
                else (Rows_constr { r = [ r ]; constr; origin } :: extras, env)
            | _ ->
                if stored then (extras, env)
                else (Rows_constr { r = [ r ]; constr; origin } :: extras, env))
        | _ ->
            if stored then (extras, env)
            else (Rows_constr { r = [ r ]; constr; origin } :: extras, env))
    | { bcast = Row_var _; _ }, _ ->
        if stored then (extras, env)
        else
          ( Rows_constr { r = [ r ]; constr; origin } :: extras,
            env (* Wait for more shape inference. *) )
    | { beg_dims; dims; bcast = Broadcastable; _ }, Exact exact_dims ->
        assert (not stored);
        ( List.map2_exn exact_dims (beg_dims @ dims) ~f:(fun d1 d2 -> Dim_eq { d1; d2; origin })
          @ extras,
          env )

let rec dim_var_occurs_in_dim (v : dim_var) (d : dim) : bool =
  match d with
  | Var v' -> equal_dim_var v v'
  | Dim _ | Sym _ -> false
  | Affine { over; conv = None; _ } -> dim_var_occurs_in_dim v over
  | Affine { over; conv = Some { kernel; _ }; _ } ->
      dim_var_occurs_in_dim v over || dim_var_occurs_in_dim v kernel
  | Concat dims -> List.exists dims ~f:(dim_var_occurs_in_dim v)

(* --- Arithmetic normalization for [Concat] strict equality (consumed by [unify_dim] below). A
   concatenated axis denotes the SUM of its components, so two concats are equated arithmetically
   (cancel common components, reduce solved [Dim]s, bind/defer the residual) rather than by
   equal-arity structural matching. --- *)

(* Flatten nested [Concat] components: a concatenated axis is the SUM of its components, and
   concatenation is associative, so [Concat [Concat [a; b]; c]] denotes the same axis as [Concat [a;
   b; c]]. Flattening before cancellation / solved-partition / variable-pairing keeps those steps
   operating on a flat component multiset, so a nested concat cannot mis-align the oldest-variable
   pairing or hide a structurally-equal component from cancellation. (The proposal only considered
   [Dim] and [Var] components; this normalizes the nested-[Concat] case the user flagged. [Affine]
   components are left intact — they are opaque atoms here and are handled by the solved-size
   arithmetic / single-component peeling below.) *)
let rec flatten_concat (dims : dim list) : dim list =
  List.concat_map dims ~f:(function Concat inner -> flatten_concat inner | d -> [ d ])

(* Multiset cancellation of structurally-equal components from both sides. *)
let cancel_common_dims (xs : dim list) (ys : dim list) : dim list * dim list =
  let kept_rev, ys =
    List.fold xs ~init:([], ys) ~f:(fun (kept, ys) x ->
        match List.findi ys ~f:(fun _ y -> equal_dim x y) with
        | Some (i, _) -> (kept, List.filteri ys ~f:(fun j _ -> j <> i))
        | None -> (x :: kept, ys))
  in
  (List.rev kept_rev, ys)

(* Split components into solved [Dim]s and the rest (vars / affine / nested concat). *)
let partition_solved_dims (dims : dim list) : solved_dim list * dim list =
  List.partition_map dims ~f:(function Dim s -> Either.First s | d -> Either.Second d)

(* Reduce the basis tags of solved components: reserved (claim-free) tags ([default], [bcast_if_1])
   drop out; all named tags must agree (and the residual takes that named tag), otherwise [default].
   [None] flags a genuine named-tag conflict. *)
let reduce_solved_basis (solved : solved_dim list) : string option =
  match
    List.filter_map solved ~f:(fun s -> if is_reserved_basis s.basis then None else Some s.basis)
    |> List.dedup_and_sort ~compare:String.compare
  with
  | [] -> Some default_basis
  | [ b ] -> Some b
  | _ -> None

(* gh-ocannl-544, dim-level counterpart of [resolve_at_use] (defined below): marked dimension
   variables may close to the GLB of their use sites at stage 6+; unmarked, unconstrained ones are
   guessed minimal instead (e.g. a learning-rate expression's axis meeting a single parameter shape
   closes to 1 rather than widening to it). [At_least_dim] variables keep GLB closing regardless:
   direct indexing is a dim-carrying use. *)
let resolve_at_use_dim = ref (Set.empty (module Dim_var))

(* Writes to the module-level inference sets count as session mutations (gh-ocannl-1301): they steer
   later guessing and resolution of open variables, so a construction that failed after one must
   taint. *)
let add_resolve_at_use_dim v =
  if not (Set.mem !resolve_at_use_dim v) then (
    note_mutation ();
    resolve_at_use_dim := Set.add !resolve_at_use_dim v)

let is_resolve_at_use_dim v =
  refuse_retired_read ~table:"resolve_at_use_dim" v.Dim_var.id;
  Set.mem !resolve_at_use_dim v

(* Two strided windows over one input extent, both in the valid form [input = stride * (output - 1)
   + span]. A valid window's span is its effective kernel span; a padded window or a plain affine of
   stride [s] reads [input = s * output], so its span is [s], not 1 (gh-ocannl-1823). With [s1 = k *
   s2], [s1 * (o1 - 1) + span1 = s2 * (o2 - 1) + span2] holds exactly when [o2 = k * (o1 - 1) +
   (span1 - span2) / s2 + 1]: a valid helper window of stride [k] and span [(span1 - span2) / s2 +
   1] over [o1], the identity when [k] and that span are both 1. Unless [s2] divides [span1 -
   span2], no outputs satisfy it. The helper's span can be below 1 (a valid window wider than the
   other side's span at one stride, [o2 = o1 - 2]); the solved arms' arithmetic holds for it all the
   same, and [s_dim_one] refuses an output below 1, which no window has.

   Strides that do not divide each other (2 and 3) make neither output a function of the other: [2 *
   o1 = 3 * o2] holds for every [o1 = 3 * k]. Such a pair waits for one side to be solved, and the
   solved arms then decide it (gh-ocannl-1972). Solutions exist exactly when [gcd s1 s2] divides
   [span1 - span2], the divisibility above when one stride divides the other, so a pair without any
   is refused at once. Two symbolic outputs are rigid, so waiting would decide nothing: refused, as
   a concat against a symbolic extent is. *)
let window_pair_equation ~s1 ~span1 ~o1 ~s2 ~span2 ~o2 =
  let reduce ~s1 ~span1 ~o1 ~s2 ~span2 ~o2 =
    let stride = s1 / s2 and span = ((span1 - span2) / s2) + 1 in
    if stride = 1 && span = 1 then `Equate (o1, o2)
    else
      `Equate
        ( o2,
          Affine
            {
              stride;
              over = o1;
              conv = Some { dilation = 1; kernel = get_default_dim ~d:span (); use_padding = false };
              stride_offset = 0;
            } )
  in
  if (span1 - span2) % Utils.gcd s1 s2 <> 0 then
    `Refuse
      "solved dimensions for axis: window spans differ by a non-multiple of the strides' common \
       divisor"
  else if s1 % s2 = 0 then reduce ~s1 ~span1 ~o1 ~s2 ~span2 ~o2
  else if s2 % s1 = 0 then reduce ~s1:s2 ~span1:span2 ~o1:o2 ~s2:s1 ~span2:span1 ~o2:o1
  else
    match (o1, o2) with
    | Sym _, Sym _ -> `Refuse "solved dimensions for axis: unresolvable strides"
    | _ -> `Defer

(* A window of [unify_dim]'s pair arms, as [(stride, span, output)]: a padded window or a plain
   affine spans its stride, a valid window its effective kernel span. *)
let pair_window = function
  | Affine { stride; over; conv = None | Some { use_padding = true; _ }; _ } ->
      Some (stride, stride, over)
  | Affine { stride; over; conv = Some { dilation; kernel = Dim k; use_padding = false }; _ } ->
      Some (stride, effective_kernel_span ~dilation ~kernel_size:k.d, over)
  | _ -> None

(* gh-ocannl-2040: a pair [window_pair_equation] defers, with both outputs open, that fails once the
   closing stages run. Nothing sized the extent or either output, so closing sized the outputs on
   their own, and the plain arms' [mismatch] or [incompatible stride] names neither window nor the
   fix. Name both, and the widths that fit them: [w = s1 * (o1 - 1) + span1 = s2 * (o2 - 1) + span2]
   with both outputs at least 1, the smallest such [w] and every [lcm s1 s2] after it (the deferral
   means [gcd s1 s2] divides [span1 - span2], so one exists below [max span + lcm]). Only a failure
   of sizes: an output closing sized whose read width the other window cannot fit. [None] for
   anything else, whose own message stands. *)
let unsized_window_pair_message env (d1, d2) =
  match (pair_window d1, pair_window d2) with
  | Some (s1, span1, o1), Some (s2, span2, o2)
    when s1 % s2 <> 0
         && s2 % s1 <> 0
         && (not (Set.is_empty (vars_of_dim o1)))
         && not (Set.is_empty (vars_of_dim o2)) -> (
      let var_to_string (v : Dim_var.t) = Option.value v.name ~default:(Dim_var.to_string v) in
      let show d = dim_to_string ~var_to_string Axis_size d in
      let fits ~s ~span w = w >= span && (w - span) % s = 0 in
      let closed o ~s ~span =
        match subst_dim env o with
        | Dim { d; _ } -> Some ([%string "%{show o} at %{d#Int}"], (s * (d - 1)) + span)
        | _ -> None
      in
      let sized =
        match (closed o1 ~s:s1 ~span:span1, closed o2 ~s:s2 ~span:span2) with
        | Some (c1, w1), Some (c2, w2) when w1 <> w2 ->
            Some [%string "%{c1} and %{c2}, which read %{w1#Int} and %{w2#Int} cells"]
        | Some (c, w), None when not (fits ~s:s2 ~span:span2 w) ->
            Some [%string "%{c}, which reads %{w#Int} cells, a width %{show d2} cannot read"]
        | None, Some (c, w) when not (fits ~s:s1 ~span:span1 w) ->
            Some [%string "%{c}, which reads %{w#Int} cells, a width %{show d1} cannot read"]
        | _ -> None
      in
      match sized with
      | None -> None
      | Some sized ->
          let g = Utils.gcd s1 s2 in
          let lcm = s1 / g * s2 in
          (* The widths the first window reads from [max span] on, [s1] apart: [s2 / g] of them
             cover every residue modulo [s2] the first can reach, so no list of [lcm] widths. *)
          let lowest = Int.max span1 span2 in
          let first = lowest + ((span1 - lowest) % s1) in
          let rec search k =
            if k >= s2 / g then None
            else
              let w = first + (k * s1) in
              if fits ~s:s2 ~span:span2 w then Some w else search (k + 1)
          in
          let fit =
            Option.value_map (search 0) ~default:"" ~f:(fun w ->
                [%string " The widths both fit are %{w#Int} and every %{lcm#Int} after it."])
          in
          Some
            [%string
              "the windows %{show d1} and %{show d2} read one input extent with strides %{s1#Int} \
               and %{s2#Int}, which divide neither way, so neither output determines the other, \
               and nothing sized the extent or either output before shape inference closed. \
               Closing sized the outputs on their own: %{sized}. Give the input a width both \
               windows fit, or size one output (capture it with the spec's trailing label list and \
               Shape.set_dim it) for the other to follow.%{fit}"])
  | _ -> None

(* Substituting a solution (gh-ocannl-1904). A solution [v := value] visits the entries it can
   change -- those mentioning [v], and the pending ones (see {!environment}) for a dimension
   solution -- rather than mapping over the whole environment, which made inference quadratic in the
   number of variables. [Utils.Tree_map.sparse_mapi] keeps the tree's shape and the whole map's
   visiting order, so the environment and the emitted constraints, in their order, are those the
   whole map produced. [with_whole_env_substitution] runs the whole map, for tests; the
   [shape_substitution_cross_check] setting runs both at every solution and fails on any
   difference. *)
let subst_whole_env = ref false
[@@mutation_serial.exempt "a test switch, set only around a test's own run"]

let with_whole_env_substitution f =
  let saved = !subst_whole_env in
  subst_whole_env := true;
  Exn.protect ~f ~finally:(fun () -> subst_whole_env := saved)

(* The entries solutions visited since the last reset, reported by [Shape.registration_counts]. *)
let subst_visit_count = ref 0
[@@mutation_serial.exempt "a work counter of solution substitution, not inference state"]

let subst_visits () = !subst_visit_count
let reset_subst_visits () = subst_visit_count := 0

(* The inequalities row solutions' visits emitted since the last reset (gh-ocannl-1977), reported by
   [Shape.registration_counts]. *)
let row_visit_emission_count = ref 0
[@@mutation_serial.exempt "a work counter of solution substitution, not inference state"]

let row_visit_emissions () = !row_visit_emission_count
let reset_row_visit_emissions () = row_visit_emission_count := 0

let subst_cross_check =
  lazy (Utils.get_global_flag ~default:false ~arg_name:"shape_substitution_cross_check")

let subst_cross_check_forced = ref false
[@@mutation_serial.exempt "a test switch, set only around a test's own run"]

let substitution_cross_checked () = !subst_cross_check_forced || Lazy.force subst_cross_check

let with_substitution_cross_check f =
  let saved = !subst_cross_check_forced in
  subst_cross_check_forced := true;
  Exn.protect ~f ~finally:(fun () -> subst_cross_check_forced := saved)

(* Visits [targets] of [map] (every entry under [subst_whole_env]) with [f], which returns an
   entry's new value and the constraints it emits. Returns the new map, the emitted constraints --
   the later visit's first, as the whole map's fold accumulated them -- and, per visited entry, its
   new value, whether it changed, and whether it stays pending for a dimension solution: unless the
   visit left an entry that does not mention the solved variable unchanged and silent, the next
   unrelated visit can still change it. *)
let visit_entries ~compare ~equal ~mentions ~targets map ~f =
  let emitted = ref [] and visited = ref [] in
  let g ~key ~data =
    Int.incr subst_visit_count;
    let data', more = f ~key ~data in
    emitted := more @ !emitted;
    let changed = not (equal data data') in
    visited :=
      (key, data', changed, changed || (not (List.is_empty more)) || mentions data) :: !visited;
    data'
  in
  let map =
    if !subst_whole_env then Utils.Tree_map.mapi map ~f:g
    else Utils.Tree_map.sparse_mapi ~compare ~targets:(Lazy.force targets) map ~f:g
  in
  (map, !emitted, !visited)

let checked_visit ~what ~compare ~equal ~sexp_of_key ~sexp_of_data ~mentions ~targets map ~f =
  let visit () = visit_entries ~compare ~equal ~mentions ~targets map ~f in
  if !subst_whole_env || not (substitution_cross_checked ()) then visit ()
  else
    (* The reference visit is not substitution's work: the counter keeps the indexed one's. *)
    let run whole =
      let saved = !subst_whole_env and visits = !subst_visit_count in
      subst_whole_env := whole;
      Exn.protect
        ~f:(fun () -> Result.try_with visit)
        ~finally:(fun () ->
          subst_whole_env := saved;
          if whole then subst_visit_count := visits)
    in
    let reference = run true in
    let indexed = run false in
    let fail detail =
      failwith
        [%string "shape_substitution_cross_check: the indexed %{what} visit differs: %{detail}"]
    in
    match (reference, indexed) with
    | Ok (map1, emitted1, _), Ok ((map2, emitted2, _) as result) ->
        let alist m = Utils.Tree_map.to_alist m in
        let sexp_of_alist m =
          List.sexp_of_t (fun (k, d) -> Sexp.List [ sexp_of_key k; sexp_of_data d ]) (alist m)
        in
        let entry_equal (k1, d1) (k2, d2) = compare k1 k2 = 0 && equal d1 d2 in
        if not (List.equal entry_equal (alist map1) (alist map2)) then
          fail
            (Sexp.to_string_hum
               (Sexp.List
                  [ Sexp.Atom "whole"; sexp_of_alist map1; Sexp.Atom "indexed"; sexp_of_alist map2 ]));
        if not (List.equal equal_constraint_ emitted1 emitted2) then
          fail
            (Sexp.to_string_hum
               (Sexp.List
                  [
                    Sexp.Atom "whole";
                    [%sexp_of: constraint_ list] emitted1;
                    Sexp.Atom "indexed";
                    [%sexp_of: constraint_ list] emitted2;
                  ]));
        result
    | Error e1, Error e2 ->
        if not (Sexp.equal (Exn.sexp_of_t e1) (Exn.sexp_of_t e2)) then
          fail (Exn.to_string e1 ^ " vs. " ^ Exn.to_string e2);
        raise e2
    | Ok _, Error e -> fail ("only the indexed visit raised: " ^ Exn.to_string e)
    | Error e, Ok _ -> fail ("only the whole visit raised: " ^ Exn.to_string e)

let mentioning index v ~empty = Option.value (Map.find index v) ~default:empty

(* Records the visited entries' new mentions; [~dim_solution] says the visit applied a dimension
   solution, whose verdict on pending applies: a row solution never normalizes, so it can only add
   to the pending entries (by changing one). *)
let reindex_dims env visited =
  List.fold visited ~init:env ~f:(fun env (key, data, changed, pending) ->
      let dims_mentioning =
        if changed then
          add_mentions env.dims_mentioning ~vars:(dim_entry_mentions data) ~key
            ~empty:(Set.empty (module Dim_var))
        else env.dims_mentioning
      in
      let pending_dims =
        if pending then Set.add env.pending_dims key else Set.remove env.pending_dims key
      in
      { env with dims_mentioning; pending_dims })

let reindex_rows ~dim_solution env visited =
  List.fold visited ~init:env ~f:(fun env (key, data, changed, pending) ->
      let env =
        if changed then
          {
            env with
            rows_mentioning_dim =
              add_mentions env.rows_mentioning_dim ~vars:(row_entry_dim_mentions data) ~key
                ~empty:row_var_set_empty;
            rows_mentioning_row =
              add_mentions env.rows_mentioning_row ~vars:(row_entry_row_mentions data) ~key
                ~empty:row_var_set_empty;
          }
        else env
      in
      let pending_rows =
        if dim_solution then
          if pending then Set.add env.pending_rows key else Set.remove env.pending_rows key
        else if changed then Set.add env.pending_rows key
        else env.pending_rows
      in
      { env with pending_rows })

(* The dimension solution [v := value]'s visits: of the dim entries, then of the row entries. *)
let visit_dims_for_dim_solution env v ~value =
  checked_visit ~what:"dim-entry" ~compare:compare_dim_var ~equal:equal_dim_entry
    ~sexp_of_key:sexp_of_dim_var ~sexp_of_data:sexp_of_dim_entry
    ~mentions:(fun e -> Set.mem (dim_entry_mentions e) v)
    ~targets:
      (lazy
        (Set.to_list
           (Set.union
              (mentioning env.dims_mentioning v ~empty:(Set.empty (module Dim_var)))
              env.pending_dims)))
    env.dim_env
    ~f:(fun ~key:_ ~data ->
      let more, data = s_dim_one_in_entry v ~value data in
      (data, more))

let visit_rows_for_dim_solution ~stage env v ~value =
  checked_visit ~what:"row-entry (dimension solution)" ~compare:compare_row_var
    ~equal:equal_row_entry ~sexp_of_key:sexp_of_row_var ~sexp_of_data:sexp_of_row_entry
    ~mentions:(fun e -> Set.mem (row_entry_dim_mentions e) v)
    ~targets:
      (lazy
        (Set.to_list
           (Set.union
              (mentioning env.rows_mentioning_dim v ~empty:row_var_set_empty)
              env.pending_rows)))
    env.row_env
    ~f:(fun ~key ~data ->
      let saved = !ineqs_from_reapply_rows_constr in
      ineqs_from_reapply_rows_constr := [];
      match s_dim_one_in_row_entry stage v ~value ~key ~data with
      | data ->
          let more = !ineqs_from_reapply_rows_constr in
          ineqs_from_reapply_rows_constr := saved;
          (data, more)
      | exception e ->
          ineqs_from_reapply_rows_constr := saved;
          raise e)

(* The environment after the dimension solution [v := value], from the visits' results. *)
let record_dim_solution env v ~value (dim_env, dim_visited) (row_env, row_visited) =
  let env = reindex_dims { env with dim_env; row_env } dim_visited in
  let env = reindex_rows ~dim_solution:true env row_visited in
  let env =
    {
      env with
      dims_mentioning = Map.remove env.dims_mentioning v;
      rows_mentioning_dim = Map.remove env.rows_mentioning_dim v;
    }
  in
  set_dim env ~key:v ~data:(Solved_dim value)

let visit_rows_for_row_solution env v ~value =
  checked_visit ~what:"row-entry (row solution)" ~compare:compare_row_var ~equal:equal_row_entry
    ~sexp_of_key:sexp_of_row_var ~sexp_of_data:sexp_of_row_entry
    ~mentions:(fun e -> Set.mem (row_entry_row_mentions e) v)
    ~targets:(lazy (Set.to_list (mentioning env.rows_mentioning_row v ~empty:row_var_set_empty)))
    env.row_env
    ~f:(fun ~key:_ ~data ->
      let more, data = s_row_one_in_entry v ~value ~in_:data in
      (data, more))

(* From [solve_inequalities], at a fixpoint: an equation that leaves a window's output or kernel
   open takes the first tag whose consequences [fits] accepts, and [settled] says one did
   (gh-ocannl-1968). With [merge_pairs], a window pair with both outputs open applies its size
   equation instead (gh-ocannl-2038); that pass comes only after no open tag would settle, since a
   tag settled first can solve one output and leave the pair to the solved arms, which are exact. *)
type settling = {
  mutable settled : bool;
  merge_pairs : bool;
  fits : constraint_ list -> environment -> bool;
}

let%track6_sexp rec unify_dim ?settling ~stage origin (eq : dim * dim) env : constraint_ list * _ =
  let dim1 : dim = subst_dim env @@ fst eq and dim2 : dim = subst_dim env @@ snd eq in
  let held = Dim_eq { d1 = dim1; d2 = dim2; origin } in
  (* Two windows over one extent say their inputs coincide, and their outputs need not: the basis
     rule is not injective, so [1_(bcast_if_1)] and [1_default] under [default] kernels both derive
     a [default] input. The size equation is a whole-dim equation, which would equate the outputs'
     tags too, and its helper window's [default] kernel carries a tag the windows do not
     (gh-ocannl-2038: refused pair-first and accepted solved-first, and an inert [2_(bcast_if_1)]
     became the top at size 1, accepted pair-first and refused solved-first). So while an output is
     open the pair waits: once a side is solved, the solved arms decide sizes and tags exactly, the
     other output's tag through [window_bases]. Only at a closing fixpoint, where nothing else will
     solve either output and closure would size each one on its own, does the size equation apply;
     the pair stays with it, to check the tags once closure solves the outputs (gh-ocannl-1968's
     reason as well). Rigid outputs are decided at once, since nothing would ever solve them: their
     tags are fixed, so the equation is exact unless a kernel claims a tag, which keeps the pair. *)
  let window_pair ?(kernels = []) ~s1 ~span1 ~o1 ~s2 ~span2 ~o2 () =
    match window_pair_equation ~s1 ~span1 ~o1 ~s2 ~span2 ~o2 with
    | `Equate eq -> (
        let is_open = not (Set.is_empty (Set.union (vars_of_dim o1) (vars_of_dim o2))) in
        match settling with
        | _ when not is_open ->
            let constrs, env' = unify_dim ~stage origin eq env in
            if List.for_all kernels ~f:is_claim_free then (constrs, env')
            else if phys_equal env' env then ([ held ], env)
            else (held :: constrs, env')
        | None -> ([ held ], env)
        | Some { merge_pairs = false; _ } -> ([ held ], env)
        | Some settling ->
            (* The settling goes along: outputs that are windows themselves reduce to a nested pair,
               which would wait again without it. *)
            let constrs, env' = unify_dim ~settling ~stage origin eq env in
            if phys_equal env' env then ([ held ], env)
            else (
              settling.settled <- true;
              (held :: constrs, env')))
    | `Defer -> ([ held ], env)
    | `Refuse msg -> raise @@ Shape_error (msg, [ Dim_mismatch [ dim1; dim2 ] ])
  in
  (* Several tags satisfy the equation. [default] suits every window, but a bound on the variable or
     a nested window can rule it out, so it goes first and the rest follow. A candidate must bind
     something: one waiting on a nested window's kernel settles nothing, and this equation stays
     with it. The last candidate needs no vetting, since the solver will say what is wrong with it.
     A deferral leaves the environment as it was. *)
  let settle_open bases ~solve =
    match settling with
    | None -> ([ held ], env)
    | Some settling -> (
        let attempt ~vet basis =
          match solve basis with
          | constrs, env' when not (phys_equal env' env) ->
              if vet then Option.some_if (settling.fits constrs env') (constrs, env')
              else Some (constrs, env')
          | _ -> None
        in
        let rec first_fit = function
          | [] -> assert false
          | [ basis ] -> attempt ~vet:false basis
          | basis :: rest -> (
              match attempt ~vet:true basis with
              | Some _ as fit -> fit
              | None -> first_fit rest
              | exception Shape_error _ -> first_fit rest)
        in
        let others = List.filter bases ~f:(Fn.non (String.equal default_basis)) in
        let fit = first_fit (default_basis :: others) in
        settling.settled <- Option.is_some fit;
        match fit with None -> ([ held ], env) | Some result -> result)
  in
  match (dim1, dim2) with
  | Dim { basis = b1; _ }, Dim { basis = b2; _ } when not (String.equal b1 b2) ->
      raise
      @@ Shape_error ("solved dimensions for axis: different bases", [ Dim_mismatch [ dim1; dim2 ] ])
  | Dim { d = d1; basis = _; _ }, Dim { d = d2; basis = _; _ } when d1 = d2 ->
      (* Basis is total now: equality of two solved dims requires both equal size and equal tag (the
         preceding arm already rejected unequal tags). There is no unspecified ([None]) side to
         propagate a basis onto, so the old upgrade-var pass is gone — variable solving records the
         already-total [Dim] exactly. *)
      ([], env)
  | Sym { sym = s1; sym_basis = b1; _ }, Sym { sym = s2; sym_basis = b2; _ }
    when Ir.Indexing.equal_static_symbol s1 s2 ->
      (* A symbolic dimension is a rigid constant: equal only to itself (and, mirroring [Dim],
         requiring equal basis tags). Distinct symbols and [Sym] vs. concrete [Dim] fall through to
         the final mismatch arm. *)
      if String.equal b1 b2 then ([], env)
      else
        raise
        @@ Shape_error
             ("solved dimensions for axis: different bases", [ Dim_mismatch [ dim1; dim2 ] ])
  | Var v1, Var v2 when equal_dim_var v1 v2 -> ([], env)
  (* Two arms, not an or-pattern: a guard failing on the first alternative would skip the second. An
     arm that rewrites the equation into an equivalent one passes [settling] on, so a window or a
     window pair inside it settles as this equation would (gh-ocannl-2038). *)
  | Affine { stride; over; conv; stride_offset = _ }, dim when is_identity_window ~stride ~conv ->
      unify_dim ?settling ~stage origin (over, dim) env
  | dim, Affine { stride; over; conv; stride_offset = _ } when is_identity_window ~stride ~conv ->
      unify_dim ?settling ~stage origin (over, dim) env
  | ( Affine
        {
          stride = stride1;
          over = over1;
          conv = Some { use_padding = true; _ } | None;
          stride_offset = _;
        },
      Affine
        {
          stride = stride2;
          over = over2;
          conv = Some { use_padding = true; _ } | None;
          stride_offset = _;
        } )
    when stride1 % stride2 = 0 || stride2 % stride1 = 0 ->
      (* Both have use_padding=true, we can simplify by dividing strides *)
      unify_dim ?settling ~stage origin
        ( Affine
            {
              stride = (if stride1 > stride2 then stride1 / stride2 else stride2 / stride1);
              over = (if stride1 > stride2 then over1 else over2);
              conv = None;
              stride_offset = 0;
            },
          if stride1 > stride2 then over2 else over1 )
        env
  | Affine { stride; over = Dim s; conv = None | Some { use_padding = true; _ }; _ }, dim
  | dim, Affine { stride; over = Dim s; conv = None | Some { use_padding = true; _ }; _ } ->
      (* stride_offset doesn't contribute to shapes when conv = None or use_padding = true *)
      unify_dim ?settling ~stage origin (get_dim ~d:(stride * s.d) ~basis:s.basis (), dim) env
  | Affine { stride; over; conv = None | Some { use_padding = true; _ }; _ }, Dim s
  | Dim s, Affine { stride; over; conv = None | Some { use_padding = true; _ }; _ } ->
      (* stride_offset doesn't contribute to shapes when conv = None or use_padding = true *)
      if s.d >= 0 && s.d % stride = 0 then
        unify_dim ?settling ~stage origin (get_dim ~d:(s.d / stride) ~basis:s.basis (), over) env
      else
        raise
        @@ Shape_error
             ("solved dimensions for axis: incompatible stride", [ Dim_mismatch [ dim1; dim2 ] ])
  | ( Affine
        { stride; over = Dim s; conv = Some { dilation; kernel = Dim k; use_padding = false }; _ },
      dim )
  | ( dim,
      Affine
        { stride; over = Dim s; conv = Some { dilation; kernel = Dim k; use_padding = false }; _ } )
    ->
      (* input_size = stride * (output_size - 1) + effective_kernel_span *)
      let span = effective_kernel_span ~dilation ~kernel_size:k.d in
      let basis = derived_window_basis s k in
      unify_dim ?settling ~stage origin
        (get_dim ~d:((stride * (s.d - 1)) + span) ~basis (), dim)
        env
  | Affine { stride; over; conv = Some { dilation; kernel = Dim k; use_padding = false }; _ }, Dim s
  | Dim s, Affine { stride; over; conv = Some { dilation; kernel = Dim k; use_padding = false }; _ }
    ->
      (* Reverse: solve for output_size given input_size s. input_size = stride * (output_size - 1)
         + effective_kernel_span output_size = (input_size - effective_kernel_span) / stride + 1 *)
      let span : int = effective_kernel_span ~dilation ~kernel_size:k.d in
      let numerator : int = s.d - span in
      if numerator >= 0 && numerator % stride = 0 then
        let d = (numerator / stride) + 1 in
        let solve_over basis = unify_dim ?settling ~stage origin (get_dim ~d ~basis (), over) env in
        match window_output_bases ~size:d ~input:s ~kernel:k with
        | [] ->
            raise
            @@ Shape_error
                 ( "convolution: conflicting dimension bases between stride and kernel",
                   [ Dim_mismatch [ dim1; dim2 ] ] )
        | [ basis ] -> solve_over basis
        | bases ->
            settle_open bases ~solve:(fun basis ->
                let constrs, env = solve_over basis in
                let more, env = unify_dim ~stage origin (dim1, dim2) env in
                (constrs @ more, env))
      else
        raise
        @@ Shape_error
             ("solved dimensions for axis: incompatible stride", [ Dim_mismatch [ dim1; dim2 ] ])
  | ( Affine
        { stride; over = Dim s; conv = Some { dilation; kernel = Var v; use_padding = false }; _ },
      Dim i )
  | ( Dim i,
      Affine
        { stride; over = Dim s; conv = Some { dilation; kernel = Var v; use_padding = false }; _ } )
    -> (
      (* Infer kernel dimension from the relationship: input_size = stride * (output_size - 1) +
         effective_kernel_span effective_kernel_span = input_size - stride * (output_size - 1) *)
      let span = i.d - (stride * (s.d - 1)) in
      match kernel_size_of_span ~dilation ~span with
      | Some k -> (
          (* The sizes pick the kernel's size, and the rule its tag, as for an open output. The
             equation is re-unified with the kernel solved, for the identity's basis
             (gh-ocannl-1824: a kernel solved before the equation met the solved arms' check, one
             inferred here skipped it). *)
          let solve_kernel basis =
            let constrs, env = unify_dim ~stage origin (Var v, get_dim ~d:k ~basis ()) env in
            let more, env = unify_dim ~stage origin (dim1, dim2) env in
            (constrs @ more, env)
          in
          match window_kernel_bases ~stride ~dilation ~size:k ~input:i ~extent:s with
          | [] ->
              raise
              @@ Shape_error
                   ( "convolution: conflicting dimension bases between stride and kernel",
                     [ Dim_mismatch [ dim1; dim2 ] ] )
          | [ basis ] -> solve_kernel basis
          | bases -> settle_open bases ~solve:solve_kernel)
      | None ->
          raise
          @@ Shape_error
               ( "solved dimensions for axis: cannot infer kernel dimension",
                 [ Dim_mismatch [ dim1; dim2 ] ] ))
  | ( Affine { stride = s1; over = o1; conv = None | Some { use_padding = true; _ }; _ },
      Affine { stride = s2; over = o2; conv = None | Some { use_padding = true; _ }; _ } ) ->
      (* Strides that divide each other took the padded-pair arm above. A padded window or a plain
         affine reads [input = s * o = s * (o - 1) + s]: its span is its stride. *)
      window_pair ~s1 ~span1:s1 ~o1 ~s2 ~span2:s2 ~o2 ()
  | ( Affine
        {
          stride = s1;
          over = o1;
          conv = Some { dilation = d1; kernel = Dim k1; use_padding = false };
          _;
        },
      Affine
        {
          stride = s2;
          over = o2;
          conv = Some { dilation = d2; kernel = Dim k2; use_padding = false };
          _;
        } ) ->
      (* Both sides use: input = stride * (output - 1) + span *)
      let span1 = effective_kernel_span ~dilation:d1 ~kernel_size:k1.d
      and span2 = effective_kernel_span ~dilation:d2 ~kernel_size:k2.d in
      window_pair ~kernels:[ k1; k2 ] ~s1 ~span1 ~o1 ~s2 ~span2 ~o2 ()
  | ( Affine { stride = s1; over = o1; conv = None | Some { use_padding = true; _ }; _ },
      Affine
        {
          stride = s2;
          over = o2;
          conv = Some { dilation = d2; kernel = Dim k2; use_padding = false };
          _;
        } )
  | ( Affine
        {
          stride = s2;
          over = o2;
          conv = Some { dilation = d2; kernel = Dim k2; use_padding = false };
          _;
        },
      Affine { stride = s1; over = o1; conv = None | Some { use_padding = true; _ }; _ } ) ->
      (* A padded window or a plain affine reads [input = s1 * o1 = s1 * (o1 - 1) + s1]: its span is
         its stride (gh-ocannl-1823). *)
      let span2 = effective_kernel_span ~dilation:d2 ~kernel_size:k2.d in
      window_pair ~kernels:[ k2 ] ~s1 ~span1:s1 ~o1 ~s2 ~span2 ~o2 ()
  | Affine { conv = Some { kernel = Var _; use_padding = false; _ }; _ }, _
  | _, Affine { conv = Some { kernel = Var _; use_padding = false; _ }; _ } ->
      (* Can't compute offset until kernel dimension is resolved; defer *)
      ([ Dim_eq { d1 = dim1; d2 = dim2; origin } ], env)
  | Var v, dim2 | dim2, Var v ->
      if dim_var_occurs_in_dim v dim2 then
        raise
        @@ Shape_error
             ( "occurs check failed: dimension variable occurs in its own definition",
               [ Dim_mismatch [ Var v; dim2 ] ] );
      (* Unification merges the axes, so use-site resolvability is shared (gh-ocannl-544). *)
      (match dim2 with
      | Var v2 ->
          if is_resolve_at_use_dim v then add_resolve_at_use_dim v2;
          if is_resolve_at_use_dim v2 then add_resolve_at_use_dim v
      | _ -> ());
      let ineqs = ref [] in
      ineqs_from_reapply_rows_constr := [];
      (* The dim entries' visit, then the row entries': the order the whole maps ran in. *)
      let visit_dims () =
        let dim_env, more, visited = visit_dims_for_dim_solution env v ~value:dim2 in
        ineqs := more @ !ineqs;
        (dim_env, visited)
      in
      let visit_rows () =
        let row_env, more, visited = visit_rows_for_dim_solution ~stage env v ~value:dim2 in
        ineqs_from_reapply_rows_constr := more;
        (row_env, visited)
      in
      let env =
        match find_dim env.dim_env v with
        | None ->
            let dims = visit_dims () in
            record_dim_solution env v ~value:dim2 dims (visit_rows ())
        | Some (Solved_dim _) -> assert false
        | Some (Bounds_dim { is_in_param = _; res; opnd; glb; constr; origin = origin1; _ } as in_)
          ->
            let origin = merge_origins origin origin1 in
            let from_ = [%sexp_of: dim_var * dim_entry] (v, drop_origin in_) in
            let dims = visit_dims () in
            List.iter res ~f:(fun res ->
                ineqs := Dim_ineq { res = Var res; opnd = dim2; from_; origin } :: !ineqs);
            List.iter opnd ~f:(fun opnd ->
                ineqs := Dim_ineq { res = dim2; opnd = Var opnd; from_; origin } :: !ineqs);
            Option.iter glb ~f:(fun glb ->
                ineqs := Dim_ineq { res = glb; opnd = dim2; from_; origin } :: !ineqs);
            let extras, constr = apply_dim_constraint ~source:Equation ~stage dim2 constr env in
            let extras =
              if is_unconstrained_dim constr then extras
              else Dim_constr { d = dim2; constr; origin } :: extras
            in
            ineqs := extras @ !ineqs;
            record_dim_solution env v ~value:dim2 dims (visit_rows ())
      in
      ineqs := !ineqs_from_reapply_rows_constr @ !ineqs;
      ineqs_from_reapply_rows_constr := [];
      let dim_eqs, ineqs =
        List.partition_map !ineqs ~f:(function
          | Dim_eq { d1; d2; origin } -> Either.First ((d1, d2), origin)
          | ineq -> Either.Second ineq)
      in
      let f (ineqs, env) (ds, origin) =
        let more_ineqs, env = unify_dim ~stage origin ds env in
        (more_ineqs @ ineqs, env)
      in
      List.fold ~init:(ineqs, env) dim_eqs ~f
  | Concat [], d | d, Concat [] ->
      (* An empty concat is the size-0 axis. *)
      unify_dim ?settling ~stage origin (get_default_dim ~d:0 (), d) env
  | Concat [ x ], d | d, Concat [ x ] ->
      (* Single-component concat erases to its component. *)
      unify_dim ?settling ~stage origin (x, d) env
  | Concat xs, Concat ys -> (
      (* Arithmetic equality of two concatenated axes. Cancel structurally-equal components, then
         reconcile the residual sums: a fully-solved side reduces to a [Dim] comparison; mixed
         solved/unsolved subtracts the solved part; an all-unsolved equal-length residual (the pure
         nested-stacking case, where corresponding fresh stack axes match positionally) pairs
         element-wise; anything still ambiguous is deferred until substitution resolves more. *)
      let xs = flatten_concat xs
      and ys = flatten_concat ys in
      let xs, ys = cancel_common_dims xs ys in
      let solved_x, rest_x = partition_solved_dims xs in
      let solved_y, rest_y = partition_solved_dims ys in
      let sum_x = List.sum (module Int) solved_x ~f:(fun s -> s.d) in
      let sum_y = List.sum (module Int) solved_y ~f:(fun s -> s.d) in
      let basis_of solved =
        match reduce_solved_basis solved with
        | Some b -> b
        | None ->
            raise
            @@ Shape_error ("concat: conflicting dimension bases", [ Dim_mismatch [ dim1; dim2 ] ])
      in
      let basis_x = basis_of solved_x and basis_y = basis_of solved_y in
      let exceed () =
        raise
        @@ Shape_error
             ("concat: components exceed the concatenated size", [ Dim_mismatch [ dim1; dim2 ] ])
      in
      match (rest_x, rest_y) with
      | [], [] ->
          unify_dim ~stage origin
            (get_dim ~d:sum_x ~basis:basis_x (), get_dim ~d:sum_y ~basis:basis_y ())
            env
      | [], _ ->
          let residual = sum_x - sum_y in
          if residual < 0 then exceed ();
          unify_dim ~stage origin (Concat rest_y, get_dim ~d:residual ~basis:basis_x ()) env
      | _, [] ->
          let residual = sum_y - sum_x in
          if residual < 0 then exceed ();
          unify_dim ~stage origin (Concat rest_x, get_dim ~d:residual ~basis:basis_y ()) env
      | _, _
        when let is_var = function Var _ -> true | _ -> false in
             (* The pairing arm requires BOTH residuals to be variables-only. Solved [Dim]s are
                already partitioned into [solved_x]/[solved_y] (handled by the
                arithmetic-cancellation arm below), so "the other side carries a solved [Dim]" still
                reaches here with an all-[Var] residual. But a residual that still holds a
                NON-variable, NON-solved component (e.g. an unresolved [Affine] from a strided/conv
                axis) must NOT be paired: equating oldest variables and forcing the leftover to the
                affine over-constrains a sum equality that is genuinely underdetermined (Codex P2).
                Such cases fall through to the arithmetic-cancellation arm, which cancels the common
                solved size and defers. *)
             List.for_all rest_x ~f:is_var && List.for_all rest_y ~f:is_var
             &&
             (* Original pure-nested-stacking case (PR #64): equal arity, no solved component on
                either side. Fires at ANY stage — the nested-stacking concat components are linked
                ONLY through this equation, and the incremental Stage1 solver (propagate_shapes /
                derive_projections) relies on it resolving early; gating it to stage 4 makes
                inference loop (Test 5 [[x1;x2];[x1;x2]]). *)
             (List.is_empty solved_x && List.is_empty solved_y
              && List.length rest_x = List.length rest_y
             (* AC3 generalization: at stage >= 4 extend to UNEQUAL arity and to a solved [Dim] on
                either side (still variables-only residuals). The same-arity and [solved_*]-empty
                restrictions are dropped. *)
             || is_stage4_up stage) ->
          (* Link the two concat axes by equating their oldest (lowest-id) residual variables and
             re-running the unification on the two original concats under the resulting env.
             [cancel_common_dims] then sees the freshly-equated components as common, so unequal
             arity and any further progress fall out of the re-run — no leftover-variable
             bookkeeping or reduced-sides construction is needed. This takes precedence over the
             arithmetic- cancellation arm below. Both residuals are non-empty here (the empty cases
             are handled by the earlier arms), so each side has an oldest variable. Termination:
             post-cancellation no variable appears on both sides, so the two oldest variables are
             distinct, and each re-entry equates one more pair — strictly shrinking the
             free-variable count. *)
          let oldest rest =
            List.filter_map rest ~f:(function Var v -> Some v | _ -> None)
            |> List.min_elt ~compare:compare_dim_var
          in
          let vx = Option.value_exn ~here:[%here] (oldest rest_x) in
          let vy = Option.value_exn ~here:[%here] (oldest rest_y) in
          let eq_constr, env = unify_dim ~stage origin (Var vx, Var vy) env in
          let more, env = unify_dim ~stage origin (dim1, dim2) env in
          (eq_constr @ more, env)
      | _, _ ->
          (* Both sides still have unsolved residuals (and at least one side carries a solved size
             or a non-variable residual). Normalize the solved [Dim] components by cancelling the
             common solved size from both sides, so at most one side carries the remaining solved
             size (proposal: "normalize solved Dim components so that only one side carries the
             remaining solved size"). The surviving non-neutral basis carries onto the residual;
             distinct non-neutral bases conflict. No negative size can arise here since the
             cancelled amount is the smaller of the two sums. *)
          let common = Int.min sum_x sum_y in
          let residual_basis =
            merge_derived_basis basis_x basis_y ~on_conflict:(fun () ->
                raise
                @@ Shape_error
                     ("concat: conflicting dimension bases", [ Dim_mismatch [ dim1; dim2 ] ]))
          in
          let with_residual rest sum =
            let r = sum - common in
            if r = 0 then Concat rest else Concat (rest @ [ get_dim ~d:r ~basis:residual_basis () ])
          in
          ( [ Dim_eq { d1 = with_residual rest_x sum_x; d2 = with_residual rest_y sum_y; origin } ],
            env ))
  | Concat xs, (Dim n as d) | (Dim n as d), Concat xs -> (
      (* A concatenated axis equals a solved size: subtract the solved components and bind/defer the
         residual. *)
      let xs = flatten_concat xs in
      let solved, rest = partition_solved_dims xs in
      let sum = List.sum (module Int) solved ~f:(fun s -> s.d) in
      let solved_basis =
        match reduce_solved_basis solved with
        | Some b -> b
        | None ->
            raise
            @@ Shape_error ("concat: conflicting dimension bases", [ Dim_mismatch [ dim1; dim2 ] ])
      in
      let residual = n.d - sum in
      if residual < 0 then
        raise
        @@ Shape_error
             ("concat: components exceed the concatenated size", [ Dim_mismatch [ dim1; dim2 ] ]);
      let residual_basis =
        merge_derived_basis solved_basis n.basis ~on_conflict:(fun () ->
            raise
            @@ Shape_error ("concat: conflicting dimension bases", [ Dim_mismatch [ dim1; dim2 ] ]))
      in
      match rest with
      | [] ->
          (* Fully solved: the sum meets [d] under [solved_basis], not [residual_basis]. A residual
             component is a variable, and binding it records the merged tag, so a named size claims
             it once and for all. A solved sum records nothing: were its [default] claim-free here,
             an all-integer [2^3] would take [rgb 5] and later [hsv 5] alike, merging the two named
             bases silently. So [2^3] keeps [default] and is refused against a named size, while
             [2^a] is accepted with [a] taking the claim (gh-ocannl-2093). [subst_dim] usually sums
             such an axis before it gets here, by the same rule ([sum_solved_components]). The
             principled answer, an integer component taking a slice of its partner's basis, is
             gh-ocannl-2095. *)
          unify_dim ~stage origin (get_dim ~d:sum ~basis:solved_basis (), d) env
      | [ c ] -> unify_dim ~stage origin (c, get_dim ~d:residual ~basis:residual_basis ()) env
      | _ ->
          (* Multiple unsolved components remain: defer the NORMALIZED residual (solved components
             subtracted), not the original equation, so the arithmetic progress is not lost and the
             same unsolved form is not requeued. *)
          ( [
              Dim_eq { d1 = Concat rest; d2 = get_dim ~d:residual ~basis:residual_basis (); origin };
            ],
            env ))
  | Concat _, Sym _ | Sym _, Concat _ ->
      (* gh-490 v1: reconciling a concatenation's component sum with a symbolic extent requires
         symbolic arithmetic; fail fast rather than defer forever. (A concat whose components cancel
         structurally against another concat's [Sym] components is already handled above.) *)
      raise
      @@ Shape_error
           ( "concat axis cannot be equated with a symbolic dimension",
             [ Dim_mismatch [ dim1; dim2 ] ] )
  | Concat _, _ | _, Concat _ ->
      (* Concat against a not-yet-reducible dimension (e.g. an unresolved affine): defer rather than
         reject, so a later substitution can make progress. *)
      ([ Dim_eq { d1 = dim1; d2 = dim2; origin } ], env)
  | dim1, dim2 ->
      (* Note: at the unify_dim phase, it's strict equality (no broadcasting). *)
      raise @@ Shape_error ("solved dimensions for axis: mismatch", [ Dim_mismatch [ dim1; dim2 ] ])

let drop_from_end l n = List.rev @@ List.drop (List.rev l) n
let take_from_end (l : dim list) (n : int) : dim list = List.rev @@ List.take (List.rev l) n
let safe_to_guess = ref (Set.empty (module Row_var))

let add_safe_to_guess v =
  if not (Set.mem !safe_to_guess v) then (
    note_mutation ();
    safe_to_guess := Set.add !safe_to_guess v)

let used_in_spec_or_compose = ref (Set.empty (module Row_var))
let used_in_pointwise = ref (Set.empty (module Row_var))

let add_used_in_spec_or_compose v =
  if not (Set.mem !used_in_spec_or_compose v) then (
    note_mutation ();
    used_in_spec_or_compose := Set.add !used_in_spec_or_compose v)

let add_used_in_pointwise v =
  if not (Set.mem !used_in_pointwise v) then (
    note_mutation ();
    used_in_pointwise := Set.add !used_in_pointwise v)

(* gh-ocannl-1656: dimension variables naming a component of a concatenated einsum axis. Closure
   never guesses one to dimension 1: an unsized component is a real segment, not a broadcast, and a
   guess silently picks which elements an extraction ([p^o => o]) or replacement ([a => a^b])
   touches. A discardable component (a block spec's, guessed to 0) is exempt. *)
let concat_component_vars = ref (Set.empty (module Dim_var))

let add_concat_component v =
  if not (Set.mem !concat_component_vars v) then (
    note_mutation ();
    concat_component_vars := Set.add !concat_component_vars v)

let row_var_id (Row_var.Row_var id) = id

let refuse_concat_component_guess env v =
  refuse_retired_read ~table:"concat_component_vars" v.Dim_var.id;
  if Set.mem !concat_component_vars v && not (Set.mem env.discardable_vars v) then
    raise
    @@ Shape_error
         ( [%string
             "the size of the concatenation component %{Dim_var.to_string v} is not determined, \
              and closure does not guess one: it is a segment of the axis, not a broadcast. Size \
              it: capture the component (the spec's trailing label list) and Shape.set_dim it, \
              write its size as an integer component (e.g. 2^o => o), or pass the operation's \
              result dims explicitly"],
           [ Dim_mismatch [ Var v ] ] )

(* gh-ocannl-544: row variables marked here are resolved by their use sites — at the final stages
   they may close to the GLB of their use sites' rows, acquiring axes their own arguments do not
   have. Unmarked (operation-result) variables close to an empty remainder instead: a use site must
   not silently widen an operation's result row beyond its arguments'. Registered for terminal rows
   (leaf tensors and parameters, from the [Terminal_row] dispatch), for parameter-initialization
   cones ([Tensor.param] via [Shape.set_resolve_at_use]: the init expression's job is to fill the
   parameter's shape, so its intermediates widen to it), and for explicit [stretch] wrappers. *)
let resolve_at_use = ref (Set.empty (module Row_var))

let add_resolve_at_use v =
  if not (Set.mem !resolve_at_use v) then (
    note_mutation ();
    resolve_at_use := Set.add !resolve_at_use v)

let is_resolve_at_use v =
  refuse_retired_read ~table:"resolve_at_use" (row_var_id v);
  Set.mem !resolve_at_use v

(* Persistent rank-relation graph, global like [global_template_cache] (row variable ids are never
   reused, so entries from unrelated inference problems are unreachable). An edge [v -> (w, k)] with
   [k >= 0] records the entailed fact [rank v >= rank w + k]: from solving [v := <w> ++ k known
   dims], from equal-flank row inequalities [<v>+flanks ⊑ <w>+flanks] (k = 0), and from the deficit
   (template) rule (k = the rank deficit > 0). Such facts are never retracted, unlike the
   [Bounds_row] res/opnd adjacency lists which drop entries when a variable is substituted away —
   mutually-growing row variables ping-pong between the store and the in-flight constraint list, so
   divergence detection needs this side table. *)
let global_rank_edges : (row_var * int) list Map.M(Row_var).t ref = ref (Map.empty (module Row_var))

(* gh-ocannl-1891: a potential per variable of the rank graph, kept so that every edge [v -> w] of
   weight [k] has [p v >= p w + k]. Summed around a cycle, that bounds the cycle's total weight by
   0, so holding it is a proof that no cycle is positive: an edge that already satisfies it needs no
   search, however the variables tie, and one that does not moves only the variables that must move.
   A forward build places each new variable next to its operand, so that is the rare case. *)
let global_rank_potentials : int Map.M(Row_var).t ref = ref (Map.empty (module Row_var))

(* The rank-cycle check's work counter (gh-ocannl-1802): the variables whose potentials it moved,
   and the states a reference walk visited, since the last reset, reported by
   [Shape.registration_counts]. *)
let rank_search_visit_count = ref 0
[@@mutation_serial.exempt "a work counter of the rank-cycle check, not inference state"]

let rank_search_visits () = !rank_search_visit_count
let reset_rank_search_visits () = rank_search_visit_count := 0

(* The dimension-cycle search's work counter (gh-ocannl-1891): the variables it visited since the
   last reset, reported by [Shape.registration_counts]. *)
let dim_cycle_visit_count = ref 0
[@@mutation_serial.exempt "a work counter of the dimension-cycle search, not inference state"]

let dim_cycle_visits () = !dim_cycle_visit_count
let reset_dim_cycle_visits () = dim_cycle_visit_count := 0

let unmemoized_cycle_search = ref false
[@@mutation_serial.exempt "a test switch, set only around a test's own run"]

let with_unmemoized_cycle_search f =
  unmemoized_cycle_search := true;
  Exn.protect ~f ~finally:(fun () -> unmemoized_cycle_search := false)

let unbounded_rank_search = ref false
[@@mutation_serial.exempt "a test switch, set only around a test's own run"]

let with_unbounded_rank_search f =
  unbounded_rank_search := true;
  Exn.protect ~f ~finally:(fun () -> unbounded_rank_search := false)

(* The potentials of a new edge's endpoints, placing a variable new to the graph so that the edge
   holds. *)
let rank_potentials v w k =
  let potentials = !global_rank_potentials in
  let place v p =
    note_mutation ();
    global_rank_potentials := Map.set !global_rank_potentials ~key:v ~data:p
  in
  match (Map.find potentials v, Map.find potentials w) with
  | Some pv, Some pw -> (pv, pw)
  | Some pv, None ->
      place w (pv - k);
      (pv, pv - k)
  | None, Some pw ->
      place v (pw + k);
      (pw + k, pw)
  | None, None ->
      place v k;
      place w 0;
      (k, 0)

(* Restores [p v >= p w + k] for a new edge that breaks it by [delta], lowering [w] and what must
   follow it, and returns [false], committing nothing, when the edge closes a positive cycle. A path
   [w ~> v] of weight [s] holds [p w >= p v + s], so [delta = p w + k - p v] reaching [v] means [s +
   k > 0], and a cycle of positive weight means it reaches [v]. How much of [delta] crosses an edge
   [u -> x] is what its slack [p u - p x - k'] leaves: nonnegative, so taking the largest pending
   amount first settles each variable once, as in Dijkstra's algorithm. *)
let lower_rank_potentials ?(commit = true) ~src:v ~dst:w ~delta () =
  let potentials = ref !global_rank_potentials in
  let pending = Hashtbl.create (module Row_var) in
  let settled = Hash_set.create (module Row_var) in
  Hashtbl.set pending ~key:w ~data:delta;
  let rec settle queue =
    match Set.min_elt queue with
    | None -> true
    | Some ((neg_amount, u) as first) ->
        let queue = Set.remove queue first in
        let amount = -neg_amount in
        if Hash_set.mem settled u || Hashtbl.find_exn pending u <> amount then settle queue
        else if equal_row_var u v then false
        else (
          Hash_set.add settled u;
          Int.incr rank_search_visit_count;
          let pu = Map.find_exn !potentials u in
          potentials := Map.set !potentials ~key:u ~data:(pu - amount);
          settle
            (List.fold (Map.find_multi !global_rank_edges u) ~init:queue ~f:(fun queue (x, k') ->
                 let crossing = amount - (pu - Map.find_exn !potentials x - k') in
                 if
                   crossing > 0
                   && (not (Hash_set.mem settled x))
                   && crossing > Option.value (Hashtbl.find pending x) ~default:0
                 then (
                   Hashtbl.set pending ~key:x ~data:crossing;
                   Set.add queue (-crossing, x))
                 else queue)))
  in
  let holds = settle (Set.Poly.singleton (-delta, w)) in
  if holds && commit then (
    note_mutation ();
    global_rank_potentials := !potentials);
  holds

(* The reference walk, for tests: would adding [rank v >= rank w + k] close a cycle of positive
   total weight? A path [w ~> v] of total weight [s] entails [rank v >= rank v + s + k]:
   contradiction iff [s + k > 0]. Stored weights are nonnegative, so for [k > 0] reachability
   suffices, and for [k = 0] the path must contain a positive edge; DFS over (node,
   still-needs-positive) states. *)
let rank_cycle_walk ~src:v ~dst:w k =
  let visited_need = Hash_set.create (module Row_var) in
  let visited_done = Hash_set.create (module Row_var) in
  let rec dfs u need_positive =
    (equal_row_var u v && not need_positive)
    ||
    let visited = if need_positive then visited_need else visited_done in
    (not (Hash_set.mem visited u))
    &&
    (Hash_set.add visited u;
     Int.incr rank_search_visit_count;
     List.exists (Map.find_multi !global_rank_edges u) ~f:(fun (u', k') ->
         dfs u' (need_positive && k' = 0)))
  in
  dfs w (k = 0)

(* Record the entailed fact [rank v >= rank w + k]; raise if it makes the ranks unsatisfiable, which
   would otherwise diverge by minting ever-fresh template variables (rank-cycle check, transitive
   generalization of the one-step self-reference check). *)
let add_rank_edge ~rows v w k =
  refuse_retired_read ~table:"global_rank_edges" (row_var_id v);
  refuse_retired_read ~table:"global_rank_edges" (row_var_id w);
  if equal_row_var v w then (
    if k > 0 then
      raise @@ Shape_error ("Infinite number of axes by self-reference", [ Row_mismatch rows ]))
  else
    let edges = Map.find_multi !global_rank_edges v in
    (* A recorded edge passed the check when it was added, and repeating it adds no path. *)
    if not (List.mem edges (w, k) ~equal:(fun (w1, k1) (w2, k2) -> equal_row_var w1 w2 && k1 = k2))
    then (
      note_mutation ();
      let pv, pw = rank_potentials v w k in
      let holds = pv >= pw + k || lower_rank_potentials ~src:v ~dst:w ~delta:(pw + k - pv) () in
      if !unbounded_rank_search && Bool.(rank_cycle_walk ~src:v ~dst:w k = holds) then
        failwith "add_rank_edge: the potentials and the reference walk disagree on a rank cycle";
      if not holds then
        raise
        @@ Shape_error
             ("Infinite number of axes by rank cycle among row variables", [ Row_mismatch rows ]);
      global_rank_edges := Map.add_multi !global_rank_edges ~key:v ~data:(w, k))

(* Is [rank v >= rank w + k] already refuted by the recorded facts? Asks what {!add_rank_edge} would
   answer and records nothing, for a commitment that is a choice rather than an entailment. A
   variable new to the graph has no path to close. *)
let rank_edge_refuted v w k =
  refuse_retired_read ~table:"global_rank_potentials" (row_var_id v);
  refuse_retired_read ~table:"global_rank_potentials" (row_var_id w);
  let refuted =
    if equal_row_var v w then k > 0
    else
      match (Map.find !global_rank_potentials v, Map.find !global_rank_potentials w) with
      | Some pv, Some pw ->
          pv < pw + k
          && not (lower_rank_potentials ~commit:false ~src:v ~dst:w ~delta:(pw + k - pv) ())
      | _ -> false
  in
  if
    !unbounded_rank_search
    && (not (equal_row_var v w))
    && Bool.(rank_cycle_walk ~src:v ~dst:w k <> refuted)
  then failwith "rank_edge_refuted: the potentials and the reference walk disagree";
  refuted

(* Closing [v] to a GLB is a choice, the widest reading of its use sites, so it yields to what is
   known: a GLB over [..w..] with axes around it is out of reach of a [v] already bounded by [w],
   however true the bound (a statistic [..c..] used by a tensor over [h, w, ..c..], gh-ocannl-1955).
   Such a GLB says nothing about [v]'s rank, so closing [v] to it is no reading of it. Its trailing
   flank still bounds [v]'s last axes, which a closing that must give [v] axes reads
   ([eliminate_row_constraint], gh-ocannl-2035). *)
let rank_refutes_glb v ({ beg_dims; dims; bcast; prov = _ } : t) =
  match bcast with
  | Row_var w -> rank_edge_refuted v w (List.length beg_dims + List.length dims)
  | Broadcastable -> false

let global_template_cache : (row_var * int * int, row_var) Map.Poly.t ref = ref Map.Poly.empty

(* {2 Checkpoint capture (gh-ocannl-903)}

   Every module-level binding above whose writes count as session mutations is a persistent value,
   so a checkpoint captures it as a pointer and a rewind puts the pointer back. [capture_globals]
   must capture each such binding (test/operations/mutation_serial_scan holds it to that, unless the
   binding is [exempt] or [kept]); [restore_globals] destructures the record exhaustively, so a
   captured field it forgets to restore is an unused-variable error. *)
type globals = {
  g_pending_roles : (Ir.Indexing.static_symbol * [ `Extent | `Slice ]) list;
  g_resolve_at_use_dim : Set.M(Dim_var).t;
  g_safe_to_guess : Set.M(Row_var).t;
  g_used_in_spec_or_compose : Set.M(Row_var).t;
  g_used_in_pointwise : Set.M(Row_var).t;
  g_concat_component_vars : Set.M(Dim_var).t;
  g_resolve_at_use : Set.M(Row_var).t;
  g_rank_edges : (row_var * int) list Map.M(Row_var).t;
  g_rank_potentials : int Map.M(Row_var).t;
  g_template_cache : (row_var * int * int, row_var) Map.Poly.t;
}

let capture_globals () =
  {
    g_pending_roles = !pending_roles;
    g_resolve_at_use_dim = !resolve_at_use_dim;
    g_safe_to_guess = !safe_to_guess;
    g_used_in_spec_or_compose = !used_in_spec_or_compose;
    g_used_in_pointwise = !used_in_pointwise;
    g_concat_component_vars = !concat_component_vars;
    g_resolve_at_use = !resolve_at_use;
    g_rank_edges = !global_rank_edges;
    g_rank_potentials = !global_rank_potentials;
    g_template_cache = !global_template_cache;
  }

let restore_globals
    {
      g_pending_roles;
      g_resolve_at_use_dim;
      g_safe_to_guess;
      g_used_in_spec_or_compose;
      g_used_in_pointwise;
      g_concat_component_vars;
      g_resolve_at_use;
      g_rank_edges;
      g_rank_potentials;
      g_template_cache;
    } =
  pending_roles := g_pending_roles;
  resolve_at_use_dim := g_resolve_at_use_dim;
  safe_to_guess := g_safe_to_guess;
  used_in_spec_or_compose := g_used_in_spec_or_compose;
  used_in_pointwise := g_used_in_pointwise;
  concat_component_vars := g_concat_component_vars;
  resolve_at_use := g_resolve_at_use;
  global_rank_edges := g_rank_edges;
  global_rank_potentials := g_rank_potentials;
  global_template_cache := g_template_cache
[@@mutation_serial.boundary
  "part of a rewind: puts back the solver's inference sets as a checkpoint captured them"]

(* Equate two rows, no broadcasting. Does not resolve inequalities. *)
let%debug5_sexp rec unify_row ~stage origin (eq : t * t) env : constraint_ list * _ =
  let solve (ineqs, env) : constraint_ -> constraint_ list * _ = function
    | Dim_eq { d1; d2; origin = eq_origin } ->
        let origin = merge_origins eq_origin origin in
        let more_ineqs, env = unify_dim ~stage origin (d1, d2) env in
        (* [unify_dim] solves everything it can eagerly (recursing internally); any [Dim_eq] it
           RETURNS is a deferral that cannot progress in the current env (e.g. a conv axis whose
           kernel dimension is still a variable, or a concat residual). Deferrals are idempotent, so
           re-solving them here would loop forever (gh-496) — accumulate them for the driver's next
           pass instead. *)
        (more_ineqs @ ineqs, env)
    | Row_eq { r1; r2; origin = eq_origin } ->
        let origin = merge_origins eq_origin origin in
        let more_ineqs, env = unify_row ~stage origin (r1, r2) env in
        (more_ineqs @ ineqs, env)
    | ( Dim_ineq _ | Row_ineq _ | Dim_constr _ | Rows_constr _ | Terminal_dim _ | Terminal_row _
      | Shape_row _ ) as ineq ->
        (ineq :: ineqs, env)
  in
  let unify_suffix init dims1 dims2 len =
    let dims1 = take_from_end dims1 len and dims2 = take_from_end dims2 len in
    List.fold ~init ~f:(fun acc (d1, d2) ->
        let constr = Dim_eq { d1; d2; origin } in
        try solve acc constr
        with Shape_error (s, trace) -> raise @@ Shape_error (s, Constraint_failed constr :: trace))
    @@ List.zip_exn dims1 dims2
  in
  let r1 : t = subst_row env @@ fst eq and r2 : t = subst_row env @@ snd eq in
  let l = List.length in
  match (r1, r2) with
  | r1, r2 when equal_row r1 r2 -> ([], env)
  | ( { beg_dims = beg_dims1; dims = dims1; bcast = Row_var v1; prov = _ },
      { beg_dims = beg_dims2; dims = dims2; bcast = Row_var v2; prov = _ } )
    when equal_row_var v1 v2 ->
      let dims1_l = l dims1
      and dims2_l = l dims2
      and beg_dims1_l = l beg_dims1
      and beg_dims2_l = l beg_dims2 in
      if beg_dims1_l + dims1_l <> beg_dims2_l + dims2_l then
        raise
        @@ self_reference_error ~rows:[ r1; r2 ]
             ~surplus:(abs (beg_dims1_l + dims1_l - (beg_dims2_l + dims2_l)))
             ~what:"a row would have to equal itself with more axes" ()
      else if beg_dims1_l <> beg_dims2_l then
        (* Equal total flank lengths but shifted splits: the equation is l1.<v>.r1 = l2.<v>.r2 whose
           surplus flank words rotate through [v]'s value (the word equation x ++ t = s ++ x), so
           the residue depends on [v]'s eventual length. Resolve by deferral into the closing
           policy: keep the equation in flight; if [v] is solved by other constraints, the
           substituted closed-closed (flat) check is exact; otherwise stage 6 closes [v] upward --
           the least-material disjunct -- after which the re-emitted equation requires the two
           surplus words to be equal. Eagerly binding [v] to the empty row here would be unsound: a
           later [v = [3]] is jointly satisfiable with [3].<v> = <v>.[3]. (This case was once
           silently dropped, accepting the unsatisfiable [3].<v> = <v>.[5] with no dimension
           checked.) *)
        if is_stage6_up stage then
          (* Emission order matters: the driver reverses the accumulated list each round, so the
             closing binding must come SECOND here to be processed FIRST next round (same idiom as
             [solve_row_ineq]'s stage-6 branch); otherwise the re-emitted equation reproduces the
             pair against the still-unsolved variable and the fixpoint stalls. *)
          ( [
              Row_eq { r1; r2; origin };
              Row_eq
                {
                  r1 = row_of_var v1 r1.prov;
                  r2 = { beg_dims = []; dims = []; bcast = Broadcastable; prov = r1.prov };
                  origin;
                };
            ],
            env )
        else ([ Row_eq { r1; r2; origin } ], env)
      else
        let result = unify_suffix ([], env) dims1 dims2 @@ min dims1_l dims2_l in
        (* Leading flank: outer-anchor (the reversed-then-take_from_end pair below pairs the OUTER
           prefixes of each list elementwise; see [unify_suffix] which is symmetric and the
           double-reverse cancels). *)
        unify_suffix result (List.rev beg_dims1) (List.rev beg_dims2) @@ min beg_dims1_l beg_dims2_l
  | ({ beg_dims = beg_dims1; dims = dims1; bcast = Row_var v; prov = _ } as r1), r2
  | r2, ({ beg_dims = beg_dims1; dims = dims1; bcast = Row_var v; prov = _ } as r1) -> (
      let dims1_l : int = l dims1
      and dims2_l : int = l r2.dims
      and beg_dims1_l : int = l beg_dims1 in
      let beg_dims2_l : int = l r2.beg_dims in
      let prov = merge_provenance r1.prov r2.prov in
      let beg_dims_l = min beg_dims1_l beg_dims2_l in
      if
        is_row_var r2.bcast
        && (dims1_l > dims2_l || (dims1_l = dims2_l && beg_dims1_l > beg_dims2_l))
      then unify_row ~stage origin (r2, r1) env
      else if (not (is_row_var r2.bcast)) && dims1_l + beg_dims1_l > dims2_l + beg_dims2_l then
        (* A closed r2 is flat: only the total axis count can rule the equation out (the
           [Broadcastable] arm below aligns r1's flanks against the flat axis list, so e.g. r1.dims
           may exceed r2.dims by matching into r2.beg_dims). *)
        raise @@ Shape_error ("Number of axes mismatch", [ Row_mismatch [ r1; r2 ] ])
      else
        let orig_rows = [ r1; r2 ] in
        let (beg_handled : bool), (ineqs, env), (value : row) =
          match r2.bcast with
          | Row_var v2 ->
              let beg_dims2 = r2.beg_dims in
              refuse_retired_read ~table:"the row-variable marks" (row_var_id v);
              refuse_retired_read ~table:"the row-variable marks" (row_var_id v2);
              if Set.mem !safe_to_guess v2 then add_safe_to_guess v;
              if Set.mem !used_in_spec_or_compose v2 then add_used_in_spec_or_compose v;
              if Set.mem !used_in_pointwise v2 then add_used_in_pointwise v;
              if Set.mem !safe_to_guess v then add_safe_to_guess v2;
              if Set.mem !used_in_spec_or_compose v then add_used_in_spec_or_compose v2;
              if Set.mem !used_in_pointwise v then add_used_in_pointwise v2;
              (* Unification merges the rows, so use-site resolvability is shared. *)
              if is_resolve_at_use v2 then add_resolve_at_use v;
              if is_resolve_at_use v then add_resolve_at_use v2;
              let result =
                try unify_suffix ([], env) dims1 r2.dims dims1_l
                with Shape_error (s, trace) ->
                  raise @@ Shape_error (s, Row_mismatch orig_rows :: trace)
              in
              let dims = drop_from_end r2.dims dims1_l in
              if equal_row_var v v2 then
                if List.is_empty dims && l beg_dims2 = l beg_dims1 then
                  let value : row = { beg_dims = []; dims; bcast = Row_var v; prov } in
                  ( true,
                    unify_suffix result (List.rev beg_dims1) (List.rev beg_dims2) @@ l beg_dims2,
                    value )
                else
                  raise
                  @@ self_reference_error ~rows:orig_rows
                       ~surplus:(abs (l beg_dims2 + l r2.dims - (l beg_dims1 + dims1_l)))
                       ~what:"a row would have to equal itself with more axes" ()
              else
                let result =
                  unify_suffix result (List.rev beg_dims1) (List.rev beg_dims2) beg_dims_l
                in
                let value : row =
                  { beg_dims = List.drop beg_dims2 beg_dims_l; dims; bcast = Row_var v2; prov }
                in
                (beg_dims_l = l beg_dims1, result, value)
          | Broadcastable ->
              (* r2 is closed with structural flanks beg_dims2/dims2. We match r1.beg_dims1
                 outer-left and r1.dims1 outer-right against r2's flat axis list. The middle (axes
                 absorbed by v) preserves r2's structural split: anything remaining in r2.beg_dims
                 becomes value.beg_dims; anything remaining in r2.dims becomes value.dims. r1's
                 flanks may spill over into r2's other flank if r1's flank length exceeds r2's
                 matching flank length — handled below. *)
              let r2_flat = r2.beg_dims @ r2.dims in
              let r2_l = beg_dims2_l + dims2_l in
              if dims1_l + beg_dims1_l > r2_l then
                raise @@ Shape_error ("Number of axes mismatch", [ Row_mismatch [ r1; r2 ] ])
              else
                let beg_overlap = min beg_dims1_l beg_dims2_l in
                let end_overlap = min dims1_l dims2_l in
                let beg_spill = beg_dims1_l - beg_overlap in
                let end_spill = dims1_l - end_overlap in
                let value_beg_dims =
                  r2.beg_dims |> Fn.flip List.drop beg_overlap |> Fn.flip drop_from_end end_spill
                in
                let value_dims =
                  r2.dims |> Fn.flip List.drop beg_spill |> Fn.flip drop_from_end end_overlap
                in
                let result =
                  List.zip_exn beg_dims1 (List.take r2_flat beg_dims1_l)
                  @ List.zip_exn dims1 (take_from_end r2_flat dims1_l)
                  |> List.fold ~init:([], env) ~f:(fun acc (d1, d2) ->
                      solve acc (Dim_eq { d1; d2; origin }))
                in
                let value : row =
                  { beg_dims = value_beg_dims; dims = value_dims; bcast = Broadcastable; prov }
                in
                (true, result, value)
        in
        (* From now on, we have no use for un-reduced r2 since we deal with the row variable. *)
        let r2 : row = value in
        let ineqs : constraint_ list ref = ref ineqs in
        let result env =
          let unsolved, env =
            if beg_handled then (
              (* Substituting reads [v = value] into every entry bounded by [v], so it belongs to
                 this arm alone: below, [value] is what is left of the other side, and [v] is not it
                 (gh-ocannl-1955). *)
              let row_env, more, visited = visit_rows_for_row_solution env v ~value in
              row_visit_emission_count := !row_visit_emission_count + List.length more;
              ineqs := more @ !ineqs;
              (match value.bcast with
              | Row_var u ->
                  add_rank_edge ~rows:orig_rows v u
                    (List.length value.beg_dims + List.length value.dims)
              | Broadcastable -> ());
              let constr =
                match find_row env.row_env v with
                | Some (Bounds_row { constr; origin; _ }) ->
                    [ Rows_constr { r = [ value ]; constr; origin } ]
                | _ -> []
              in
              let env = reindex_rows ~dim_solution:false { env with row_env } visited in
              let env = { env with rows_mentioning_row = Map.remove env.rows_mentioning_row v } in
              (constr, set_row env ~key:v ~data:(Solved_row value)))
            else
              let residual =
                Row_eq
                  {
                    r1 =
                      {
                        beg_dims = List.drop beg_dims1 beg_dims_l;
                        dims = [];
                        bcast = Row_var v;
                        prov;
                      };
                    r2;
                    origin;
                  }
              in
              (* Mixed anchoring with a leading-flank surplus: begS.<v> = <v2>.dimsR (with s =
                 |begS| >= 1 and t = |dimsR|) is a word equation whose residue depends on the
                 eventual rank: s + |v| = |v2| + t. *)
              let s = beg_dims1_l - beg_dims_l and t = List.length r2.dims in
              if s <> t then
                (* Sound eager alignment, valid at any stage: when t > s, |v| = |v2| + t - s >= t -
                   s, so the last t - s axes of [v] provably lie within dimsR's coverage -- bind v
                   := <fresh>.tail(dimsR, t - s); symmetrically when s > t the first s - t axes of
                   [v2] come from begS. Solving the binding eagerly (rather than emitting it) lets
                   the re-solved residual reduce to the balanced s = t form in the same pass, so the
                   aligned dims reach e.g. param terminals before any stage-4 guessing closes their
                   row variables. *)
                let fresh = get_row_var () in
                let binding =
                  if t > s then
                    Row_eq
                      {
                        r1 = row_of_var v prov;
                        r2 =
                          {
                            beg_dims = [];
                            dims = take_from_end r2.dims (t - s);
                            bcast = Row_var fresh;
                            prov;
                          };
                        origin;
                      }
                  else
                    match r2.bcast with
                    | Row_var v2 ->
                        Row_eq
                          {
                            r1 = row_of_var v2 prov;
                            r2 =
                              {
                                beg_dims = List.take (List.drop beg_dims1 beg_dims_l) (s - t);
                                dims = [];
                                bcast = Row_var fresh;
                                prov;
                              };
                            origin;
                          }
                    | Broadcastable ->
                        (* Unreachable: a closed r2 is handled by the [Broadcastable] arm above with
                           [beg_handled = true]. *)
                        assert false
                in
                solve (solve ([], env) binding) residual
              else if is_stage6_up stage then
                (* Balanced flanks (s = t): the rank stays ambiguous, so earlier stages defer (the
                   residual reproduces itself verbatim). A solution exists at every overlap k in
                   0..s whose overlap equations begS[k+i] = dimsR[i] (i < s - k) hold: k = 0 is the
                   empty close (rank s), k = s the always-satisfiable principal family v2 =
                   begS.<m>, v = <m>.dimsR. Stage 6 commits the least-material disjunct the solver
                   does not refute: the smallest k whose commit raises nothing (an unconditional
                   empty close would reject satisfiable stores like [3].<v> = <v2>.[5], whose least
                   solution is v2 = [3], v = [5]). Binding [v2] and re-solving the residual derives
                   the overlap equations and [v]'s value through the closed-row arm in the same
                   pass, so the commit itself is the test: the overlap equations hold or fail
                   together, as [unify_dim] decides them, with [v]'s and [v2]'s bounds -- not pair
                   by pair, which missed two occurrences of one variable forced apart, as in [x;
                   x].<v> = <v2>.[3; 5] at k = 0 (gh-ocannl-2037). A refused trial leaves nothing
                   behind: its environment is dropped and the module-level inference state is put
                   back ([restore_globals]; fresh ids and the mutation serial only move forward).
                   "Not refused" is not "satisfiable": a constraint the commit defers can still
                   fail. *)
                let v2 = match r2.bcast with Row_var v2 -> v2 | Broadcastable -> assert false in
                let begS = List.drop beg_dims1 beg_dims_l in
                let commit k =
                  let value : row =
                    if k = s then
                      { beg_dims = begS; dims = []; bcast = Row_var (get_row_var ()); prov }
                    else { beg_dims = []; dims = List.take begS k; bcast = Broadcastable; prov }
                  in
                  let binding = Row_eq { r1 = row_of_var v2 prov; r2 = value; origin } in
                  solve (solve ([], env) binding) residual
                in
                let rec first_fit k =
                  if k >= s then commit s
                  else
                    let saved = capture_globals () in
                    match commit k with
                    | result -> result
                    | exception Shape_error _ ->
                        restore_globals saved;
                        first_fit (k + 1)
                in
                first_fit 0
              else ([ residual ], env)
          in
          List.fold ~init:(unsolved, env) ~f:solve !ineqs
        in
        match find_row env.row_env v with
        | None -> result env
        | Some (Solved_row _) -> assert false
        | Some (Bounds_row { is_in_param = _; res; opnd; glb; constr; origin = origin1 }) ->
            let origin = merge_origins origin origin1 in
            let env =
              if beg_handled then (
                List.iter res ~f:(fun res ->
                    ineqs :=
                      Row_ineq { res = row_of_var res value.prov; opnd = r2; origin } :: !ineqs);
                List.iter opnd ~f:(fun opnd ->
                    ineqs :=
                      Row_ineq { opnd = row_of_var opnd value.prov; res = r2; origin } :: !ineqs);
                Option.iter glb ~f:(fun glb ->
                    ineqs := Row_ineq { res = glb; opnd = r2; origin } :: !ineqs);
                let extras, env = apply_row_constraint ~depth:0 stage origin value constr env in
                ineqs := extras @ !ineqs;
                env)
              else env
            in
            let _bound_elim_ineqs : constraint_ list = !ineqs in
            result env)
  | ({ bcast = Broadcastable; _ } as r1), ({ bcast = Broadcastable; _ } as r2) -> (
      (* Two closed rows must equal axis-by-axis, including both flanks. *)
      let r1_flat = r1.beg_dims @ r1.dims in
      let r2_flat = r2.beg_dims @ r2.dims in
      match List.zip r1_flat r2_flat with
      | Unequal_lengths ->
          raise @@ Shape_error ("Mismatching number of axes", [ Row_mismatch [ r1; r2 ] ])
      | Ok eqs ->
          List.fold ~init:([], env)
            ~f:(fun acc (d1, d2) -> solve acc (Dim_eq { d1; d2; origin }))
            eqs)

let%track5_sexp solve_dim_ineq ~(stage : stage) origin ~(res : dim) ~(opnd : dim) env :
    constraint_ list * _ =
  let nonredundant ?(more = []) (v : dim_var) (vs : dim_var list) : dim_var list =
    let _more : dim_var list = more in
    Utils.sorted_diff ~compare:compare_dim_var
      (List.dedup_and_sort ~compare:compare_dim_var (v :: vs))
      more
  in
  (* Would [res ⊑ opnd] close a cycle: is [opnd_v] below some of [ress]? Nothing here depends on the
     path taken, so each variable is visited once: the walk this replaces enumerated paths, which
     every residual connection multiplies (gh-ocannl-1891). *)
  let cyclic ~opnd_v ~ress =
    let visited = Hash_set.create (module Dim_var) in
    let rec below ress =
      List.exists ress ~f:(fun res_v ->
          equal_dim_var opnd_v res_v
          || (!unmemoized_cycle_search || not (Hash_set.mem visited res_v))
             &&
             (Hash_set.add visited res_v;
              Int.incr dim_cycle_visit_count;
              match find_dim env.dim_env res_v with
              | None | Some (Solved_dim (Dim _)) | Some (Solved_dim (Sym _)) -> false
              | Some (Solved_dim (Var v)) -> equal_dim_var opnd_v v
              | Some (Solved_dim (Affine _)) -> false (* Affine dimensions can't be cyclic *)
              | Some (Solved_dim (Concat _)) -> false (* Concat dimensions can't be cyclic *)
              | Some (Bounds_dim { res = ress; _ }) -> below ress))
    in
    (not (List.is_empty ress)) && below ress
  in
  (* The relation enforced here is [res ⊑ opnd] (res the result side that refines, opnd the
     broadcastable operand side). The broadcast order is now a flat partial order: [res ⊑ opnd] iff
     they are equal as dims (same size AND same tag) or [opnd = 1_(bcast_if_1)] (the claim-free top,
     above everything). There is no longer a wildcard that matches any same-size dim. Inequality
     records no basis update by design: every dimension carries a concrete tag at construction time,
     so there is no unspecified ([None]) side to propagate onto — the leak that was latent under
     [None] is closed by construction. *)
  match (res, opnd) with
  | res, opnd when equal_dim res opnd -> ([], env)
  | _, Dim { d = 1; basis; _ } when String.equal basis bcast_if_1 ->
      (* opnd is the top: [res ⊑ top] for every res. Accept, record nothing. *)
      ([], env)
  | (Dim { d = 1; basis; _ } as res), _ when String.equal basis bcast_if_1 ->
      (* res is the top: [top ⊑ opnd] only by equality — the top refines only itself; force it. *)
      ([ Dim_eq { d1 = opnd; d2 = res; origin } ], env)
  | Dim { basis = b1; _ }, Dim { basis = b2; _ } when not (String.equal b1 b2) ->
      raise
      @@ Shape_error
           ("dimension comparison for axis: different bases", [ Dim_mismatch [ res; opnd ] ])
  | Affine _, _ | _, Affine _ -> ([ Dim_eq { d1 = opnd; d2 = res; origin } ], env)
  | Var res_v, Var opnd_v -> (
      match (find_dim env.dim_env res_v, find_dim env.dim_env opnd_v) with
      | Some (Bounds_dim { res = res1; _ }), _ when List.mem ~equal:equal_dim_var res1 opnd_v ->
          ([ Dim_eq { d1 = res; d2 = opnd; origin } ], env)
      | _, Some (Bounds_dim { opnd = opnd2; _ }) when List.mem ~equal:equal_dim_var opnd2 res_v ->
          ([ Dim_eq { d1 = res; d2 = opnd; origin } ], env)
      | None, None ->
          ( [],
            env
            |> set_dim ~key:res_v
                 ~data:
                   (Bounds_dim
                      {
                        is_in_param = false;
                        has_uniq_constr_unless = None;
                        glb = None;
                        res = [];
                        opnd = [ opnd_v ];
                        constr = Unconstrained_dim;
                        origin;
                      })
            |> set_dim ~key:opnd_v
                 ~data:
                   (Bounds_dim
                      {
                        is_in_param = false;
                        has_uniq_constr_unless = None;
                        glb = None;
                        res = [ res_v ];
                        opnd = [];
                        constr = Unconstrained_dim;
                        origin;
                      }) )
      | Some (Solved_dim _), _ | _, Some (Solved_dim _) -> assert false
      | ( Some
            (Bounds_dim
               {
                 is_in_param;
                 res = res1;
                 opnd = opnd1;
                 glb = glb1;
                 constr = constr1;
                 origin = origin1;
                 _;
               } as in_),
          None ) ->
          let origin = merge_origins origin origin1 in
          let from_ = [%sexp_of: dim_var * dim_entry] (res_v, drop_origin in_) in
          let from_glb =
            Option.to_list glb1 |> List.map ~f:(fun res -> Dim_ineq { res; opnd; from_; origin })
          in
          let from_constr1, constr1 = apply_dim_constraint ~source:Opnd ~stage opnd constr1 env in
          let from_constr2, constr2 =
            apply_dim_constraint ~source:Res ~stage res Unconstrained_dim env
          in
          ( from_constr1 @ from_constr2 @ from_glb,
            env
            |> set_dim ~key:res_v
                 ~data:
                   (Bounds_dim
                      {
                        is_in_param;
                        has_uniq_constr_unless = None;
                        glb = glb1;
                        res = res1;
                        opnd = nonredundant opnd_v opnd1;
                        constr = constr1;
                        origin;
                      })
            |> set_dim ~key:opnd_v
                 ~data:
                   (Bounds_dim
                      {
                        is_in_param = false;
                        has_uniq_constr_unless = None;
                        glb = None;
                        res = [ res_v ];
                        opnd = [];
                        constr = constr2;
                        origin;
                      }) )
      (* No arm equates two dimensions because each is the other's only bound (gh-ocannl-2141), as
         none does for rows (gh-ocannl-2126). That guess strengthened [res ⊑ opnd] into an equality
         a solution need not satisfy ([3 ⊑ 1_(bcast_if_1)] holds unequal), and since it matched only
         an edge already recorded, it fired only on a re-processed bound: whether a store solved
         depended on whether a bound was repeated. A lower bound travels the chain by itself. *)
      | ( Some (Bounds_dim { res = ress; origin = origin1; _ }),
          Some (Bounds_dim { res = recorded; _ }) )
      (* A recorded edge passed this search when it was added, and repeating it adds no path: a
         merge re-emits a solved variable's bounds as inequalities, so what a merge adds comes
         through here too. A trace's end validation repeats every edge. *)
        when (!unmemoized_cycle_search || not (List.mem recorded res_v ~equal:equal_dim_var))
             && cyclic ~opnd_v ~ress ->
          let origin = merge_origins origin origin1 in
          ([ Dim_eq { d1 = opnd; d2 = res; origin } ], env)
      | ( None,
          Some
            (Bounds_dim
               {
                 is_in_param;
                 res = res2;
                 opnd = opnd2;
                 glb = glb2;
                 constr = constr2;
                 origin = origin2;
                 _;
               }) ) ->
          let origin = merge_origins origin origin2 in
          let from_constr1, constr1 =
            apply_dim_constraint ~source:Opnd ~stage opnd Unconstrained_dim env
          in
          let from_constr2, constr2 = apply_dim_constraint ~source:Res ~stage res constr2 env in
          ( from_constr2 @ from_constr1,
            env
            |> set_dim ~key:res_v
                 ~data:
                   (Bounds_dim
                      {
                        is_in_param;
                        has_uniq_constr_unless = None;
                        glb = None;
                        res = [];
                        opnd = [ opnd_v ];
                        constr = constr1;
                        origin;
                      })
            |> set_dim ~key:opnd_v
                 ~data:
                   (Bounds_dim
                      {
                        is_in_param;
                        has_uniq_constr_unless = None;
                        glb = glb2;
                        res = nonredundant res_v res2;
                        opnd = opnd2;
                        constr = constr2;
                        origin;
                      }) )
      | ( Some
            (Bounds_dim
               {
                 is_in_param = iip1;
                 res = res1;
                 opnd = opnd1;
                 glb = glb1;
                 constr = constr1;
                 origin = origin1;
                 _;
               } as in_),
          Some
            (Bounds_dim
               {
                 is_in_param = iip2;
                 res = res2;
                 opnd = opnd2;
                 glb = glb2;
                 constr = constr2;
                 origin = origin2;
                 _;
               }) ) ->
          let origin = merge_origins origin (merge_origins origin1 origin2) in
          let from_ = [%sexp_of: dim_var * dim_var * dim_entry] (res_v, opnd_v, drop_origin in_) in
          let from_glb =
            Option.to_list glb1 |> List.map ~f:(fun res -> Dim_ineq { res; opnd; from_; origin })
          in
          let from_constr1, constr1 = apply_dim_constraint ~source:Opnd ~stage opnd constr1 env in
          let from_constr2, constr2 = apply_dim_constraint ~source:Res ~stage res constr2 env in
          ( from_constr1 @ from_constr2 @ from_glb,
            env
            |> set_dim ~key:res_v
                 ~data:
                   (Bounds_dim
                      {
                        is_in_param = iip1;
                        has_uniq_constr_unless = None;
                        glb = glb1;
                        res = res1;
                        opnd = nonredundant ~more:opnd2 opnd_v opnd1;
                        constr = constr1;
                        origin;
                      })
            |> set_dim ~key:opnd_v
                 ~data:
                   (Bounds_dim
                      {
                        is_in_param = iip2;
                        has_uniq_constr_unless = None;
                        glb = glb2;
                        res = nonredundant ~more:res1 res_v res2;
                        opnd = opnd2;
                        constr = constr2;
                        origin;
                      }) ))
  | _, Var opnd_v -> (
      match find_dim env.dim_env opnd_v with
      | None ->
          ( [],
            set_dim env ~key:opnd_v
              ~data:
                (Bounds_dim
                   {
                     is_in_param = false;
                     has_uniq_constr_unless = None;
                     glb = Some res;
                     res = [];
                     opnd = [];
                     constr = Unconstrained_dim;
                     origin;
                   }) )
      | Some (Solved_dim _) -> assert false
      | Some
          (Bounds_dim
             {
               is_in_param;
               res = res2;
               opnd = opnd2;
               glb = Some glb2;
               constr = constr2;
               origin = origin2;
               _;
             }) -> (
          let origin = merge_origins origin origin2 in
          (* Commit the merged GLB [glb] for [opnd_v], threading [extra] constraints (the
             GLB-forcing equation plus any constraints produced by an equality attempt). [env] is
             the environment to extend, so a successful [unify_dim] attempt can thread its updated
             env through here. *)
          let commit ~glb ~extra env =
            let from_constr, constr2 = apply_dim_constraint ~source:Res ~stage res constr2 env in
            ( from_constr @ extra,
              set_dim env ~key:opnd_v
                ~data:
                  (Bounds_dim
                     {
                       is_in_param;
                       has_uniq_constr_unless = None;
                       glb = Some glb;
                       res = res2;
                       opnd = opnd2;
                       constr = constr2;
                       origin;
                     }) )
          in
          (* Demote the bound to the broadcast top 1_(bcast_if_1): the two bounds are incompatible,
             so these axes broadcast. *)
          let demote () =
            let glb = get_bcast_dim ~d:1 ~proj_id:47 () in
            commit ~glb ~extra:[ Dim_eq { d1 = opnd; d2 = glb; origin } ] env
          in
          (* Equality-attempt-first GLB merge for non-[Dim] bounds (AC1): try to unify the incoming
             bound [res] with the existing GLB [glb2]. On failure the two are irreconcilable ->
             demote to broadcast-top (mirroring the differing-[Dim] arm). On success keep the
             possibility that they resolve equal-as-sizes: below stage 4 postpone (re-defer the
             inequality, leaving the bound un-demoted, using the same idiom as the
             [discardable_vars] concat case); at stage 4+ commit the unification (the GLB stays
             [glb2], i.e. [res = glb2]) and force [res = opnd], threading the unification's own
             constraints through alongside. *)
          let merge_by_equality () =
            match
              try `Ok (unify_dim ~stage origin (res, glb2) env) with Shape_error _ -> `Failed
            with
            | `Failed -> demote ()
            | `Ok (more, env') ->
                if is_stage4_up stage then
                  commit ~glb:glb2 ~extra:(Dim_eq { d1 = opnd; d2 = glb2; origin } :: more) env'
                else ([ Dim_ineq { res; opnd; from_ = Sexp.List []; origin } ], env)
          in
          match (res, glb2) with
          | Dim _, Dim _ when equal_dim res glb2 ->
              (* Same size and same tag: keep the existing bound. (Basis is total now, so there is
                 no wildcard "one side unspecified, prefer the other" case to handle here.) *)
              commit ~glb:res ~extra:[] env
          | Sym _, Sym _ when equal_dim res glb2 ->
              (* Same symbolic dimension: keep the existing bound. *)
              commit ~glb:res ~extra:[] env
          | Dim _, Dim _ (* different size or different basis *) ->
              (* Intentional broadcast semantics: conflicting bases (or different sizes) demote to
                 the broadcast top 1_(bcast_if_1), meaning these axes are incompatible and should be
                 broadcast. This is NOT a bug — do not tighten to raise Shape_error. *)
              demote ()
          | Sym _, Sym _ | Sym _, Dim _ | Dim _, Sym _ ->
              (* A symbolic dimension is a rigid atom: incompatible with a different rigid bound, so
                 these axes broadcast (mirroring the differing-[Dim] arm). *)
              demote ()
          | Var _, _ | _, Var _ -> assert false
          | Affine _, _ | _, Affine _ | Concat _, _ | _, Concat _ -> merge_by_equality ())
      | Some
          (Bounds_dim
             {
               is_in_param;
               res = res2;
               opnd = opnd2;
               glb = None;
               constr = constr2;
               origin = origin2;
               _;
             } as in_) ->
          let origin = merge_origins origin origin2 in
          let from_ = [%sexp_of: dim_var * dim_entry] (opnd_v, drop_origin in_) in
          let from_constr, constr2 = apply_dim_constraint ~source:Res ~stage res constr2 env in
          ( from_constr
            @ List.map opnd2 ~f:(fun opnd_v -> Dim_ineq { res; opnd = Var opnd_v; from_; origin }),
            set_dim env ~key:opnd_v
              ~data:
                (Bounds_dim
                   {
                     is_in_param;
                     has_uniq_constr_unless = None;
                     glb = Some res;
                     res = res2;
                     opnd = opnd2;
                     constr = constr2;
                     origin;
                   }) ))
  | Var _, (Dim _ | Sym _) (* when d2 > 1 or based *) ->
      ([ Dim_eq { d1 = res; d2 = opnd; origin } ], env)
  | Concat dims1, Concat dims2 when List.length dims1 = List.length dims2 ->
      (* AC2: do not force pointwise element equalities (which over-constrains positionally). Emit a
         single [Concat = Concat] equation so [unify_dim]'s [Concat = Concat] branch handles it via
         structural cancellation and the generalized variable pairing (AC3). *)
      ([ Dim_eq { d1 = Concat dims1; d2 = Concat dims2; origin } ], env)
  | _, Concat dims
    when List.count dims ~f:(function Var v -> Set.mem env.discardable_vars v | _ -> false)
         >= List.length dims - 1 ->
      (* Concat in opnd position with all-but-one (or all) components being discardable vars:
         preserve inequality so the solver can infer the variables *)
      ([ Dim_ineq { res; opnd; from_ = Sexp.List []; origin } ], env)
  | Concat _, _ | _, Concat _ ->
      (* Defer to dimension equality for concat with non-concat *)
      ([ Dim_eq { d1 = res; d2 = opnd; origin } ], env)
  | (Dim _ | Sym _), (Dim _ | Sym _) ->
      (* Includes symbolic dimensions vs. concrete ones, and two distinct symbolic dimensions: a
         [Sym] is a rigid atom equal only to itself. *)
      raise
      @@ Shape_error ("dimension comparison for axis: mismatch", [ Dim_mismatch [ res; opnd ] ])

let retire_closed_variables () =
  if !uid >= reserved_uid_base then retired_reserved_floor := !uid
  else (
    retired_floor := !uid;
    retired_reserved_floor := !reserved_uid);
  resolve_at_use_dim := dim_var_set_empty;
  safe_to_guess := row_var_set_empty;
  used_in_spec_or_compose := row_var_set_empty;
  used_in_pointwise := row_var_set_empty;
  concat_component_vars := dim_var_set_empty;
  resolve_at_use := row_var_set_empty;
  global_rank_edges := Map.empty (module Row_var);
  global_rank_potentials := Map.empty (module Row_var);
  global_template_cache := Map.Poly.empty
[@@mutation_serial.boundary
  "part of the commit: only a successful closure or a session reset calls it, past every reader"]

let solver_census () =
  [
    ("pending_roles", List.length !pending_roles);
    ("resolve_at_use_dim", Set.length !resolve_at_use_dim);
    ("safe_to_guess", Set.length !safe_to_guess);
    ("used_in_spec_or_compose", Set.length !used_in_spec_or_compose);
    ("used_in_pointwise", Set.length !used_in_pointwise);
    ("concat_component_vars", Set.length !concat_component_vars);
    ("resolve_at_use", Set.length !resolve_at_use);
    ("global_rank_edges", Map.sum (module Int) !global_rank_edges ~f:List.length);
    ("global_rank_potentials", Map.length !global_rank_potentials);
    ("global_template_cache", Map.length !global_template_cache);
  ]

(* The components of two captures that differ (physically: each is a persistent value). *)
let globals_differences
    {
      g_pending_roles = a1;
      g_resolve_at_use_dim = a2;
      g_safe_to_guess = a3;
      g_used_in_spec_or_compose = a4;
      g_used_in_pointwise = a5;
      g_concat_component_vars = a6;
      g_resolve_at_use = a7;
      g_rank_edges = a8;
      g_rank_potentials = a10;
      g_template_cache = a9;
    }
    {
      g_pending_roles = b1;
      g_resolve_at_use_dim = b2;
      g_safe_to_guess = b3;
      g_used_in_spec_or_compose = b4;
      g_used_in_pointwise = b5;
      g_concat_component_vars = b6;
      g_resolve_at_use = b7;
      g_rank_edges = b8;
      g_rank_potentials = b10;
      g_template_cache = b9;
    } =
  List.filter_map
    [
      ("Row.pending_roles", phys_equal a1 b1);
      ("Row.resolve_at_use_dim", phys_equal a2 b2);
      ("Row.safe_to_guess", phys_equal a3 b3);
      ("Row.used_in_spec_or_compose", phys_equal a4 b4);
      ("Row.used_in_pointwise", phys_equal a5 b5);
      ("Row.concat_component_vars", phys_equal a6 b6);
      ("Row.resolve_at_use", phys_equal a7 b7);
      ("Row.global_rank_edges", phys_equal a8 b8);
      ("Row.global_rank_potentials", phys_equal a10 b10);
      ("Row.global_template_cache", phys_equal a9 b9);
    ]
    ~f:(fun (name, same) -> if same then None else Some name)

(* {2 Shape-scheme support (gh-ocannl-404)}

   A shape scheme is a trace's constraints read off the session before its rewind, renamed per
   application. Variables of both kinds share one counter ([uid]), so "minted since the trace began"
   is an id comparison. *)

let var_mark () = !uid
let dim_var_id (v : dim_var) = v.id
let row_var_id (Row_var.Row_var id) = id
let discardable_vars env = env.discardable_vars

type renaming = { rn_dim : dim_var -> dim_var; rn_row : row_var -> row_var }

let rec rename_dim rn = function
  | Var v -> Var (rn.rn_dim v)
  | (Dim _ | Sym _) as d -> d
  | Affine { stride; over; conv; stride_offset } ->
      let conv = Option.map conv ~f:(fun c -> { c with kernel = rename_dim rn c.kernel }) in
      Affine { stride; over = rename_dim rn over; conv; stride_offset }
  | Concat dims -> Concat (List.map dims ~f:(rename_dim rn))

let rename_row rn { beg_dims; dims; bcast; prov } =
  {
    beg_dims = List.map beg_dims ~f:(rename_dim rn);
    dims = List.map dims ~f:(rename_dim rn);
    bcast = (match bcast with Row_var v -> Row_var (rn.rn_row v) | Broadcastable -> Broadcastable);
    prov;
  }

let rename_total_elems rn = function
  | (Num_elems _ | Range_elems _) as n -> n
  | Strided_var s -> Strided_var { s with var = rn.rn_dim s.var }

let rename_row_constraint rn = function
  | Unconstrained -> Unconstrained
  | Total_elems { numerator; divided_by; keep_axis } ->
      Total_elems
        {
          numerator = rename_total_elems rn numerator;
          divided_by = List.map divided_by ~f:rn.rn_dim;
          keep_axis;
        }
  | Exact dims -> Exact (List.map dims ~f:(rename_dim rn))

let rename_constraint rn = function
  | Dim_eq { d1; d2; origin } -> Dim_eq { d1 = rename_dim rn d1; d2 = rename_dim rn d2; origin }
  | Row_eq { r1; r2; origin } -> Row_eq { r1 = rename_row rn r1; r2 = rename_row rn r2; origin }
  | Dim_ineq { res; opnd; from_; origin } ->
      Dim_ineq { res = rename_dim rn res; opnd = rename_dim rn opnd; from_; origin }
  | Row_ineq { res; opnd; origin } ->
      Row_ineq { res = rename_row rn res; opnd = rename_row rn opnd; origin }
  | Dim_constr { d; constr; origin } -> Dim_constr { d = rename_dim rn d; constr; origin }
  | Rows_constr { r; constr; origin } ->
      Rows_constr
        { r = List.map r ~f:(rename_row rn); constr = rename_row_constraint rn constr; origin }
  | Terminal_dim (b, d, origin) -> Terminal_dim (b, rename_dim rn d, origin)
  | Terminal_row (b, r, origin) -> Terminal_row (b, rename_row rn r, origin)
  | Shape_row (r, origin) -> Shape_row (rename_row rn r, origin)

(* The variables of a value, collected through the renaming traversal: one owner for the
   positions. *)
let collect_vars ~f =
  let dims = ref [] and rows = ref [] in
  let rn =
    {
      rn_dim =
        (fun v ->
          dims := v :: !dims;
          v);
      rn_row =
        (fun v ->
          rows := v :: !rows;
          v);
    }
  in
  f rn;
  ( List.dedup_and_sort ~compare:compare_dim_var !dims,
    List.dedup_and_sort ~compare:compare_row_var !rows )

let constraint_vars c = collect_vars ~f:(fun rn -> ignore (rename_constraint rn c : constraint_))
let row_vars r = collect_vars ~f:(fun rn -> ignore (rename_row rn r : t))

let add_constraint_origin o = function
  | Dim_eq r -> Dim_eq { r with origin = o :: r.origin }
  | Row_eq r -> Row_eq { r with origin = o :: r.origin }
  | Dim_ineq r -> Dim_ineq { r with origin = o :: r.origin }
  | Row_ineq r -> Row_ineq { r with origin = o :: r.origin }
  | Dim_constr r -> Dim_constr { r with origin = o :: r.origin }
  | Rows_constr r -> Rows_constr { r with origin = o :: r.origin }
  | Terminal_dim (b, d, origin) -> Terminal_dim (b, d, o :: origin)
  | Terminal_row (b, r, origin) -> Terminal_row (b, r, o :: origin)
  | Shape_row (r, origin) -> Shape_row (r, o :: origin)

(* A packed conversion's draw: only [Uint4x32_to_prec] mints a round-up [Strided_var], whose
   coefficient is a lazy over its result node's precision. *)
let map_draw_row_constraint ~f = function
  | Total_elems ({ numerator = Strided_var ({ round_up = true; coeff; _ } as s); _ } as t) ->
      Total_elems { t with numerator = Strided_var { s with coeff = f coeff } }
  | (Total_elems _ | Unconstrained | Exact _) as c -> c

let map_draw_coefficients ~f = function
  | Rows_constr r -> Rows_constr { r with constr = map_draw_row_constraint ~f r.constr }
  | c -> c

let map_env_draw_coefficients ~f env =
  {
    env with
    row_env =
      Utils.Tree_map.map env.row_env ~f:(function
        | Bounds_row b -> Bounds_row { b with constr = map_draw_row_constraint ~f b.constr }
        | Solved_row _ as e -> e);
  }

(* Any other lazy coefficient is of unknown provenance: a scheme cannot tell what it reads. The
   product ties of capture references are constants (a [Utils.safe_value]). *)
let has_lazy_coefficient c =
  let lazy_constr = function
    | Total_elems { numerator = Strided_var { round_up = false; coeff; _ }; _ } ->
        not (Utils.is_safe_val coeff)
    | Total_elems _ | Unconstrained | Exact _ -> false
  in
  match c with Rows_constr { constr; _ } -> lazy_constr constr | _ -> false

(* The policy marks added since a capture: the solver's per-variable sets steer closure's guesses,
   and a rewind takes the trace's additions back, so a scheme carries them. *)
type marks = {
  m_pending_roles : (Ir.Indexing.static_symbol * [ `Extent | `Slice ]) list;
  m_resolve_at_use_dim : dim_var list;
  m_safe_to_guess : row_var list;
  m_used_in_spec_or_compose : row_var list;
  m_used_in_pointwise : row_var list;
  m_concat_component_vars : dim_var list;
  m_resolve_at_use : row_var list;
}

let marks_since (g : globals) =
  let diff now before = Set.to_list (Set.diff now before) in
  let rec roles_since acc = function
    | l when phys_equal l g.g_pending_roles -> List.rev acc
    | [] -> List.rev acc
    | r :: rest -> roles_since (r :: acc) rest
  in
  {
    m_pending_roles = roles_since [] !pending_roles;
    m_resolve_at_use_dim = diff !resolve_at_use_dim g.g_resolve_at_use_dim;
    m_safe_to_guess = diff !safe_to_guess g.g_safe_to_guess;
    m_used_in_spec_or_compose = diff !used_in_spec_or_compose g.g_used_in_spec_or_compose;
    m_used_in_pointwise = diff !used_in_pointwise g.g_used_in_pointwise;
    m_concat_component_vars = diff !concat_component_vars g.g_concat_component_vars;
    m_resolve_at_use = diff !resolve_at_use g.g_resolve_at_use;
  }

let add_marks rn m =
  List.iter m.m_pending_roles ~f:(fun (sym, role) -> claim_symbol_role sym role);
  List.iter m.m_resolve_at_use_dim ~f:(fun v -> add_resolve_at_use_dim (rn.rn_dim v));
  List.iter m.m_safe_to_guess ~f:(fun v -> add_safe_to_guess (rn.rn_row v));
  List.iter m.m_used_in_spec_or_compose ~f:(fun v -> add_used_in_spec_or_compose (rn.rn_row v));
  List.iter m.m_used_in_pointwise ~f:(fun v -> add_used_in_pointwise (rn.rn_row v));
  List.iter m.m_concat_component_vars ~f:(fun v -> add_concat_component (rn.rn_dim v));
  List.iter m.m_resolve_at_use ~f:(fun v -> add_resolve_at_use (rn.rn_row v))

let equal_modulo_proj (r1 : t) (r2 : t) =
  let rec erase = function
    | Dim d -> Dim { d with proj_id = None }
    | Sym s -> Sym { s with sym_proj_id = None }
    | Var _ as d -> d
    | Affine { stride; over; conv; stride_offset } ->
        let conv =
          Option.map conv ~f:(fun { dilation; kernel; use_padding } ->
              { dilation; kernel = erase kernel; use_padding })
        in
        Affine { stride; over = erase over; conv; stride_offset }
    | Concat dims -> Concat (List.map dims ~f:erase)
  in
  let erase_row r =
    { r with beg_dims = List.map r.beg_dims ~f:erase; dims = List.map r.dims ~f:erase }
  in
  equal_row (erase_row r1) (erase_row r2)

let same_glb_axes (r1 : t) (r2 : t) =
  let same_dim d1 d2 =
    match (d1, d2) with
    | Var _, Var _ -> true
    | _ ->
        equal_modulo_proj
          { beg_dims = []; dims = [ d1 ]; bcast = Broadcastable; prov = [] }
          { beg_dims = []; dims = [ d2 ]; bcast = Broadcastable; prov = [] }
  in
  List.equal same_dim r1.beg_dims r2.beg_dims
  && List.equal same_dim r1.dims r2.dims
  &&
  match (r1.bcast, r2.bcast) with
  | Broadcastable, Broadcastable | Row_var _, Row_var _ -> true
  | _ -> false

let%debug5_sexp solve_row_ineq ~(stage : stage) origin ~(res : t) ~(opnd : t) env :
    constraint_ list * _ =
  let nonredundant ?(more = []) v vs =
    Utils.sorted_diff ~compare:compare_row_var
      (List.dedup_and_sort ~compare:compare_row_var (v :: vs))
      more
  in
  let l = List.length in
  let res_dims_l : int = l res.dims and opnd_dims_l : int = l opnd.dims in
  let res_beg_dims = res.beg_dims and opnd_beg_dims = opnd.beg_dims in
  let res_beg_dims_l = l res_beg_dims and opnd_beg_dims_l = l opnd_beg_dims in
  let beg_dims_l = min res_beg_dims_l opnd_beg_dims_l in
  let dims_l = min res_dims_l opnd_dims_l in
  let from_ = sexp_of_constraint_ (drop_constraint_origin (Row_ineq { res; opnd; origin })) in
  (* Outer-anchor alignment on BOTH flanks: leading flank uses [List.take] (outer-left prefix);
     trailing flank uses [take_from_end] (outer-right suffix). This is symmetric with
     [unify_row]. *)
  let ineqs =
    List.map2_exn
      ~f:(fun res opnd -> Dim_ineq { res; opnd; from_; origin })
      (List.take res_beg_dims beg_dims_l)
      (List.take opnd_beg_dims beg_dims_l)
    @ List.map2_exn
        ~f:(fun res opnd -> Dim_ineq { res; opnd; from_; origin })
        (take_from_end res.dims dims_l) (take_from_end opnd.dims dims_l)
  in
  match (res, opnd) with
  | ({ bcast = Row_var v; prov; _ }, _ | _, { bcast = Row_var v; prov; _ }) when is_stage6b_up stage
    ->
      ( Row_ineq { res; opnd; origin }
        :: Row_eq
             {
               r1 = row_of_var v prov;
               r2 = { beg_dims = []; dims = []; bcast = Broadcastable; prov };
               origin;
             }
        :: ineqs,
        env )
  | res, opnd when equal_row res opnd -> ([], env)
  | { bcast = Row_var res_v; _ }, { bcast = Row_var opnd_v; _ } when equal_row_var res_v opnd_v ->
      let res_l = res_dims_l + res_beg_dims_l and opnd_l = opnd_dims_l + opnd_beg_dims_l in
      if res_l < opnd_l then
        (* The operand has more axes than its own result around the variable: [rank res >= rank
           opnd] fails for every value of it. *)
        raise
        @@ self_reference_error ~rows:[ res; opnd ] ~surplus:(opnd_l - res_l) ~equation:false
             ~what:"an operand would need more axes than its own broadcast result" ()
      else if res_l > opnd_l then
        (* The result has more axes than its operand around the variable (gh-ocannl-2036):
           satisfiable, e.g. [<v>.[2] >= <v>] at [v = []], where the operand broadcasts into the
           result's surplus. As in the shifted case below, the residue is a pointwise chain through
           the variable's value: defer, keeping the outer-anchored overlaps. *)
        (Row_ineq { res; opnd; origin } :: ineqs, env)
      else if res_dims_l <> opnd_dims_l then
        (* Equal totals but shifted splits around the shared variable: the residue is a pointwise
           chain through the variable's value, so it cannot be finalized here. Defer: if the
           variable is solved by other constraints the substituted check applies; otherwise the
           stage-6 branch above closes it upward (the least-material disjunct) and the constraint
           reduces to the closed-closed check below. (This case was once silently dropped, accepting
           the unsatisfiable [3].<v> <= <v>.[5].) *)
        (Row_ineq { res; opnd; origin } :: ineqs, env)
      else (ineqs, env)
  | { bcast = Row_var res_v; _ }, { bcast = Row_var opnd_v; _ }
    when res_dims_l = opnd_dims_l && res_beg_dims_l = opnd_beg_dims_l -> (
      add_rank_edge ~rows:[ res; opnd ] res_v opnd_v 0;
      (* [glb ⊑ res_v ⊑ opnd_v] gives [glb ⊑ opnd_v], as for dimensions (gh-ocannl-2126). *)
      let glb_forward glb1 =
        Option.to_list glb1
        |> List.map ~f:(fun glb ->
            Row_ineq { res = glb; opnd = row_of_var opnd_v opnd.prov; origin })
      in
      match (find_row env.row_env res_v, find_row env.row_env opnd_v) with
      | Some (Bounds_row { res = res1; origin = origin1; _ }), _
        when List.mem ~equal:equal_row_var res1 opnd_v ->
          let origin = merge_origins origin origin1 in
          ( Row_eq { r1 = row_of_var opnd_v opnd.prov; r2 = row_of_var res_v res.prov; origin }
            :: ineqs,
            env )
      | _, Some (Bounds_row { opnd = opnd2; origin = origin2; _ })
        when List.mem ~equal:equal_row_var opnd2 res_v ->
          let origin = merge_origins origin origin2 in
          ( Row_eq { r1 = row_of_var opnd_v opnd.prov; r2 = row_of_var res_v res.prov; origin }
            :: ineqs,
            env )
      | Some (Bounds_row { opnd = opnd1; _ }), _ when List.mem ~equal:equal_row_var opnd1 opnd_v ->
          (ineqs, env)
      | _, Some (Bounds_row { res = res2; _ }) when List.mem ~equal:equal_row_var res2 res_v ->
          (ineqs, env)
      | None, None ->
          ( ineqs,
            env
            |> set_row ~key:res_v
                 ~data:
                   (Bounds_row
                      {
                        is_in_param = false;
                        res = [];
                        opnd = [ opnd_v ];
                        glb = None;
                        constr = Unconstrained;
                        origin;
                      })
            |> set_row ~key:opnd_v
                 ~data:
                   (Bounds_row
                      {
                        is_in_param = false;
                        res = [ res_v ];
                        opnd = [];
                        glb = None;
                        constr = Unconstrained;
                        origin;
                      }) )
      | ( Some
            (Bounds_row
               {
                 is_in_param;
                 res = res1;
                 opnd = opnd1;
                 glb = glb1;
                 constr = constr1;
                 origin = origin1;
               }),
          None ) ->
          let origin = merge_origins origin origin1 in
          ( glb_forward glb1 @ ineqs,
            env
            |> set_row ~key:res_v
                 ~data:
                   (Bounds_row
                      {
                        is_in_param;
                        res = res1;
                        opnd = nonredundant opnd_v opnd1;
                        glb = glb1;
                        constr = constr1;
                        origin;
                      })
            |> set_row ~key:opnd_v
                 ~data:
                   (Bounds_row
                      {
                        is_in_param = false;
                        res = [ res_v ];
                        opnd = [];
                        glb = None;
                        constr = Unconstrained;
                        origin;
                      }) )
      | ( None,
          Some
            (Bounds_row
               {
                 is_in_param;
                 res = res2;
                 opnd = opnd2;
                 glb = glb2;
                 constr = constr2;
                 origin = origin2;
               }) ) ->
          let origin = merge_origins origin origin2 in
          ( ineqs,
            env
            |> set_row ~key:opnd_v
                 ~data:
                   (Bounds_row
                      {
                        is_in_param;
                        res = nonredundant res_v res2;
                        opnd = opnd2;
                        glb = glb2;
                        constr = constr2;
                        origin;
                      })
            |> set_row ~key:res_v
                 ~data:
                   (Bounds_row
                      {
                        (* The bound shouldn't collapse on the param below. *)
                        is_in_param;
                        res = [];
                        opnd = [ opnd_v ];
                        glb = None;
                        constr = Unconstrained;
                        origin;
                      }) )
      | ( Some
            (Bounds_row
               {
                 is_in_param = iip1;
                 res = res1;
                 opnd = opnd1;
                 glb = glb1;
                 constr = constr1;
                 origin = origin1;
               }),
          Some
            (Bounds_row
               {
                 is_in_param = iip2;
                 res = res2;
                 opnd = opnd2;
                 glb = glb2;
                 constr = constr2;
                 origin = origin2;
               }) ) ->
          let origin = merge_origins origin (merge_origins origin1 origin2) in
          ( glb_forward glb1 @ ineqs,
            env
            |> set_row ~key:res_v
                 ~data:
                   (Bounds_row
                      {
                        is_in_param = iip1 || iip2;
                        res = res1;
                        opnd = nonredundant opnd_v opnd1;
                        glb = glb1;
                        constr = constr1;
                        origin;
                      })
            |> set_row ~key:opnd_v
                 ~data:
                   (Bounds_row
                      {
                        is_in_param = iip2;
                        res = nonredundant res_v res2;
                        opnd = opnd2;
                        glb = glb2;
                        constr = constr2;
                        origin;
                      }) )
      | Some (Solved_row _), _ | _, Some (Solved_row _) -> assert false)
  | { bcast = Row_var res_v; dims; _ }, _
    when res_dims_l + res_beg_dims_l < opnd_dims_l + opnd_beg_dims_l ->
      (* The template below commits [rank res_v >= rank opnd_v + deficit] with [deficit > 0]: record
         it (and detect rank cycles) before minting fresh template variables. *)
      (match opnd.bcast with
      | Broadcastable -> ()
      | Row_var opnd_v ->
          add_rank_edge ~rows:[ res; opnd ] res_v opnd_v
            (opnd_dims_l + opnd_beg_dims_l - (res_dims_l + res_beg_dims_l)));
      let budget = opnd_dims_l + opnd_beg_dims_l - (res_dims_l + res_beg_dims_l) in
      let more_dims_l = min budget @@ max 0 (opnd_dims_l - res_dims_l) in
      let more_dims : dim list =
        Array.(to_list @@ init more_dims_l ~f:(fun _ -> Var (get_var ())))
      in
      let budget = budget - more_dims_l in
      let more_beg_dims_l = min budget @@ max 0 (opnd_beg_dims_l - res_beg_dims_l) in
      let more_beg_dims : dim list =
        Array.(to_list @@ init more_beg_dims_l ~f:(fun _ -> Var (get_var ())))
      in
      (* The key of the template cache reflects that res_v will end up substituted by
         {dims=more_dims; bcast=Row_var templ_v}. TODO: should we cache more_dims also? *)
      let templ_key : row_var * int * int =
        (res_v, opnd_dims_l - res_dims_l, opnd_beg_dims_l - res_beg_dims_l)
      in
      refuse_retired_read ~table:"global_template_cache" (row_var_id res_v);
      let templ_v : row_var =
        match Map.find !global_template_cache templ_key with
        | Some v -> v
        | None ->
            note_mutation ();
            let v = get_row_var () in
            global_template_cache := Map.set !global_template_cache ~key:templ_key ~data:v;
            v
      in
      if more_dims_l > 0 then add_safe_to_guess templ_v;
      let template : t =
        {
          beg_dims = res_beg_dims @ more_beg_dims;
          dims = more_dims @ dims;
          bcast = Row_var templ_v;
          prov = res.prov;
        }
      in
      (* We don't need to add any dimension inequalities, because they'll be captured by the extra
         row inequalities. *)
      ( [ Row_eq { r1 = res; r2 = template; origin }; Row_ineq { res = template; opnd; origin } ],
        env )
  | { bcast = Broadcastable; _ }, _ when res_dims_l + res_beg_dims_l < opnd_dims_l + opnd_beg_dims_l
    ->
      raise
      @@ Shape_error
           ( "Too many axes in an operand; maybe using * instead of *.?",
             [ Row_mismatch [ res; opnd ] ] )
  | { bcast; dims; prov = _; _ }, { bcast = Row_var opnd_v; _ }
    when opnd_dims_l <= res_dims_l && opnd_beg_dims_l <= res_beg_dims_l -> (
      (* GLB residue: drop the OUTER matched prefix from beg_dims and the OUTER matched suffix from
         dims — symmetric with the per-axis match. Note: no rank edge is recorded here — the
         entailed fact [rank res_v >= rank opnd_v - surplus] has a nonpositive weight, which the
         rank graph (nonnegative weights only) cannot represent. *)
      let r_res =
        {
          beg_dims = List.drop res.beg_dims beg_dims_l;
          dims = drop_from_end dims dims_l;
          bcast;
          prov = res.prov;
        }
      in
      (* [glb ⊑ opnd_v ⊑ w] gives [glb ⊑ w] (gh-ocannl-2126). *)
      let forward_glb_to_operands glb ws =
        List.map ws ~f:(fun w -> Row_ineq { res = glb; opnd = row_of_var w opnd.prov; origin })
      in
      match find_row env.row_env opnd_v with
      | None ->
          ( ineqs,
            set_row env ~key:opnd_v
              ~data:
                (Bounds_row
                   {
                     is_in_param = false;
                     res = [];
                     opnd = [];
                     glb = Some r_res;
                     constr = Unconstrained;
                     origin;
                   }) )
      | Some
          (Bounds_row
             {
               is_in_param;
               res = res2;
               opnd = opnd2;
               glb = None;
               constr = constr2;
               origin = origin2;
             }) ->
          let origin = merge_origins origin origin2 in
          ( forward_glb_to_operands r_res opnd2 @ ineqs,
            env
            |> set_row ~key:opnd_v
                 ~data:
                   (Bounds_row
                      {
                        is_in_param;
                        res = res2;
                        opnd = opnd2;
                        glb = Some r_res;
                        constr = constr2;
                        origin;
                      }) )
      | Some
          (Bounds_row
             {
               is_in_param;
               res = res2;
               opnd = opnd2;
               glb = Some glb2;
               constr = constr2;
               origin = origin2;
             }) -> (
          let origin = merge_origins origin origin2 in
          (* Row-level bound merge: prefer generality for broadcasting. The broadcast top
             [1_(bcast_if_1)] is most general; conflicting sizes or conflicting (named) bases
             generalize to that top — the deliberate-broadcast case (incompatible caps join to the
             top), not a basis-checking gap. The same join applies on both flanks. *)
          let join_dim d1 d2 =
            match (d1, d2) with
            | Dim { d = 1; basis; _ }, _ when String.equal basis bcast_if_1 -> d1
            | _, Dim { d = 1; basis; _ } when String.equal basis bcast_if_1 -> d2
            | Dim { d = n1; _ }, Dim { d = n2; _ } when n1 <> n2 ->
                get_bcast_dim ~d:1 ~proj_id:48 ()
            | Dim { basis = b1; _ }, Dim { basis = b2; _ } when not (String.equal b1 b2) ->
                get_bcast_dim ~d:1 ~proj_id:63 ()
            | ( Affine
                  {
                    stride;
                    over = Dim s;
                    conv = Some { use_padding = true; _ } | None;
                    stride_offset = _;
                  },
                Dim s' )
            | ( Dim s',
                Affine
                  {
                    stride;
                    over = Dim s;
                    conv = Some { use_padding = true; _ } | None;
                    stride_offset = _;
                  } )
              when stride * s.d <> s'.d ->
                get_bcast_dim ~d:1 ~proj_id:49 ()
            | ( Affine
                  {
                    stride;
                    over = Dim s;
                    conv = Some { kernel = Dim k; dilation; use_padding = false };
                    stride_offset = _;
                  },
                Dim s' )
              when (stride * (s.d - 1)) + effective_kernel_span ~dilation ~kernel_size:k.d <> s'.d
              ->
                get_bcast_dim ~d:1 ~proj_id:50 ()
            | ( Dim s',
                Affine
                  {
                    stride;
                    over = Dim s;
                    conv = Some { kernel = Dim k; dilation; use_padding = false };
                    stride_offset = _;
                  } )
              when (stride * (s.d - 1)) + effective_kernel_span ~dilation ~kernel_size:k.d <> s'.d
              ->
                get_bcast_dim ~d:1 ~proj_id:50 ()
            | ( Affine
                  {
                    stride = stride1;
                    over = Dim s1;
                    conv = Some { use_padding = true; _ } | None;
                    stride_offset = _;
                  },
                Affine
                  {
                    stride = stride2;
                    over = Dim s2;
                    conv = Some { use_padding = true; _ } | None;
                    stride_offset = _;
                  } )
              when stride1 * s1.d <> stride2 * s2.d ->
                get_bcast_dim ~d:1 ~proj_id:51 ()
            | ( Affine
                  {
                    stride = stride1;
                    over = Dim s1;
                    conv = Some { kernel = Dim k1; dilation = dilation1; use_padding = false };
                    stride_offset = _;
                  },
                Affine
                  {
                    stride = stride2;
                    over = Dim s2;
                    conv = Some { kernel = Dim k2; dilation = dilation2; use_padding = false };
                    stride_offset = _;
                  } )
              when (stride1 * (s1.d - 1))
                   + effective_kernel_span ~dilation:dilation1 ~kernel_size:k1.d
                   <> (stride2 * (s2.d - 1))
                      + effective_kernel_span ~dilation:dilation2 ~kernel_size:k2.d ->
                get_bcast_dim ~d:1 ~proj_id:52 ()
            | Sym _, Sym _ when equal_dim d1 d2 -> d1
            | Sym _, Sym _ | Sym _, Dim _ | Dim _, Sym _ ->
                (* A symbolic dimension is a rigid atom: incompatible with a different rigid bound,
                   so the join generalizes to the broadcast top (mirroring the differing-[Dim]
                   arms). [Sym] vs. [Affine]/[Concat] falls to the deferring wildcards below. *)
                get_bcast_dim ~d:1 ~proj_id:64 ()
            | Var _, _ -> d1
            | _, Var _ -> d2
            | _, Dim _ -> d2
            | _ -> d1
          in
          (* Symmetric merge across both flanks: shorter on each side ("prefer generality"). *)
          let len1 = List.length r_res.dims and len2 = List.length glb2.dims in
          let glb_len = min len1 len2 in
          let beg_len1 = List.length r_res.beg_dims and beg_len2 = List.length glb2.beg_dims in
          let glb_beg_len = min beg_len1 beg_len2 in
          let glb_is_res =
            len1 + beg_len1 < len2 + beg_len2
            || (len1 + beg_len1 = len2 + beg_len2 && is_broadcastable res.bcast)
          in
          let glb_prov = if glb_is_res then r_res.prov else glb2.prov in
          (* Note: row-variable provenance is still kept on the shorter side; both flanks now merge
             correctly so the "we lose connection" caveat for the leading flank is gone. *)
          let glb_bcast = if glb_is_res then r_res.bcast else glb2.bcast in
          let glb_beg_dims =
            List.map2_exn
              (List.take r_res.beg_dims glb_beg_len)
              (List.take glb2.beg_dims glb_beg_len)
              ~f:join_dim
          in
          let glb_dims =
            List.map2_exn
              (take_from_end r_res.dims glb_len)
              (take_from_end glb2.dims glb_len) ~f:join_dim
          in
          let glb =
            { beg_dims = glb_beg_dims; dims = glb_dims; bcast = glb_bcast; prov = glb_prov }
          in
          let env =
            set_row env ~key:opnd_v
              ~data:
                (Bounds_row
                   {
                     is_in_param;
                     res = res2;
                     opnd = opnd2;
                     glb = Some glb;
                     constr = constr2;
                     origin;
                   })
          in
          let ineqs =
            (* Only a change of axes counts: the join mints a fresh broadcast top each time, and
               keeps the incoming side's variable where both sides have one, so two GLBs differing
               only in their variables would otherwise trade places forever along a diamond. *)
            if same_glb_axes glb glb2 then ineqs else forward_glb_to_operands glb opnd2 @ ineqs
          in
          match glb with
          | { beg_dims = []; dims = [] | [ Dim { d = 1; _ } ]; bcast = Broadcastable; prov = _ } ->
              (Row_eq { r1 = row_of_var opnd_v opnd.prov; r2 = glb; origin } :: ineqs, env)
          | _ -> (ineqs, env))
      | Some (Solved_row _) -> assert false)
  | _ when res_beg_dims_l > beg_dims_l && not (is_stage7 stage) ->
      (Row_ineq { res; opnd; origin } :: ineqs, env)
  | { bcast = Broadcastable; _ }, { bcast = Broadcastable; _ }
    when opnd_dims_l + opnd_beg_dims_l <= res_dims_l + res_beg_dims_l ->
      (* Both closed: the operand broadcasts only by inserting claim-free padding at its marker, so
         its EXPLICIT material always pins the corresponding result positions, aligned on the
         result's flat axis list from the outer edges; only the inserted middle positions are
         unconstrained. The structural beg-to-beg/dims-to-dims overlap pairs (in [ineqs]) are a
         subset of these flat pairs, so we emit the flat pairs alone. This replaces a vacuous accept
         that never compared cross-flank material, accepting the unsatisfiable [3].<closed> <=
         <closed>.[5] at equal ranks where no padding is inserted at all. *)
      let res_flat = res.beg_dims @ res.dims in
      ( List.map2_exn
          ~f:(fun res opnd -> Dim_ineq { res; opnd; from_; origin })
          (List.take res_flat opnd_beg_dims_l)
          opnd_beg_dims
        @ List.map2_exn
            ~f:(fun res opnd -> Dim_ineq { res; opnd; from_; origin })
            (take_from_end res_flat opnd_dims_l)
            opnd.dims,
        env )
  | { bcast = Row_var _; _ }, { bcast = Broadcastable; _ }
    when opnd_dims_l + opnd_beg_dims_l <= res_dims_l + res_beg_dims_l
         && (opnd_beg_dims_l > beg_dims_l || opnd_dims_l > dims_l) ->
      (* Open result, closed operand with explicit material outside the structural-flank overlap:
         the residual pairings depend on the variable's value. Defer -- at stage 6 the result row is
         closed upward and the constraint reduces to the closed-closed check above. *)
      (Row_ineq { res; opnd; origin } :: ineqs, env)
  | _, { bcast = Broadcastable; _ }
    when opnd_dims_l + opnd_beg_dims_l <= res_dims_l + res_beg_dims_l ->
      (ineqs, env)
  | { bcast = Row_var _ | Broadcastable; _ }, { bcast = Row_var _ | Broadcastable; _ } ->
      (Row_ineq { res; opnd; origin } :: ineqs, env)

(** Check if a dimension variable can be guessed to 1. A variable with has_uniq_constr_unless can
    only be guessed if at least one of the "unless" variables is also prevented from guessing (to
    break cycles). *)
let can_guess_dim_to_one env has_uniq_constr_unless =
  match has_uniq_constr_unless with
  | None -> true (* No restriction *)
  | Some unless_vars ->
      (* Can guess if at least one "unless" variable is also prevented from guessing *)
      Set.exists unless_vars ~f:(fun unless_v ->
          match find_dim env.dim_env unless_v with
          | Some (Bounds_dim { has_uniq_constr_unless = Some _; _ }) -> true
          | _ -> false)

let%debug5_sexp rec close_dim_terminal ?(deferred_dims = dim_var_set_empty) ~(stage : stage)
    ~is_param origin env (dim : dim) : constraint_ list =
  match dim with
  | Dim _ | Sym _ -> []
  | Var v -> (
      match find_dim env.dim_env v with
      | Some (Solved_dim _) -> assert false
      | Some (Bounds_dim { glb = Some glb; _ }) when is_stage4_up stage ->
          [ Dim_eq { d1 = dim; d2 = glb; origin } ]
      | Some
          (Bounds_dim
             { is_in_param; has_uniq_constr_unless; glb = None; constr = Unconstrained_dim; _ })
        when is_stage5_up stage ->
          (* Check if we can guess this variable to 1 (or 0 for discardable_vars) *)
          if not (can_guess_dim_to_one env has_uniq_constr_unless) then
            [ Terminal_dim (is_param, dim, origin) ]
          else if is_param || is_in_param then
            raise
            @@ Shape_error
                 ("You forgot to specify the hidden dimension(s) 1", [ Dim_mismatch [ dim ] ])
          else if Set.mem deferred_dims v then [ Terminal_dim (is_param, dim, origin) ]
          else (
            refuse_concat_component_guess env v;
            let guess_d = if Set.mem env.discardable_vars v then 0 else 1 in
            [ Dim_eq { d1 = dim; d2 = get_bcast_dim ~d:guess_d ~proj_id:53 (); origin } ])
      | _ when not (is_stage6_up stage) -> [ Terminal_dim (is_param, dim, origin) ]
      | _ -> [])
  | Affine _ ->
      (* The input dimension itself cannot be dim-1, and the over dimension doesn't become
         transitively terminal. *)
      []
  | Concat dims -> (
      (* For concatenation, filter out dimension-0 components and if one remains, close it. TODO:
         Consider guessing discardable_vars components to 0 at stage 6, but wait until needed. *)
      let non_zero_dims = List.filter dims ~f:(function Dim { d = 0; _ } -> false | _ -> true) in
      match non_zero_dims with
      | [ single ] -> close_dim_terminal ~deferred_dims ~stage ~is_param origin env single
      | _ -> [])

let last_dim_is dims p = match List.last dims with Some (Dim { d; _ }) -> p d | _ -> false
let _r_dims r = r.beg_dims @ r.dims

let row_var_is_in_param v env =
  match find_row env.row_env v with
  | Some (Bounds_row { is_in_param; _ }) -> is_in_param
  | _ -> false

let dim_var_is_in_param v env =
  match find_dim env.dim_env v with
  | Some (Bounds_dim { is_in_param; _ }) -> is_in_param
  | _ -> false

let is_safe_to_guess v =
  refuse_retired_read ~table:"safe_to_guess" (row_var_id v);
  Set.mem !safe_to_guess v
  || (Set.mem !used_in_pointwise v && not (Set.mem !used_in_spec_or_compose v))

let%track5_sexp rec eliminate_rows_constraint ~depth stage origin ~glb (rows : row list)
    (constr : row_constraint) env : constraint_ list * environment =
  if depth > 4 then ([ Rows_constr { r = rows; constr; origin } ], env)
  else
    match rows_to_row_or_vars rows with
    | Either.First single_row ->
        eliminate_row_constraint ~depth:(depth + 1) stage origin ~terminal:false ~glb single_row
          constr env
    | Either.Second (all_dims, row_vars) -> (
        let rev_row_vars = List.rev row_vars in
        match
          ( constr,
            List.findi rev_row_vars ~f:(fun _ (_, prov) ->
                List.exists prov ~f:(fun (origin : provenance_origin) ->
                    equal_kind origin.kind `Output)) )
        with
        | Total_elems _, Some (idx, (v, v_prov)) when is_stage3_up stage -> (
            (* TODO: in stage 3, consider restricting to a strided dimension variable case. *)
            (* gh-509 task 5: a round-up total (packed uniform) whose result shape is inferred
               gets its rows' bounds only once the surrounding shapes settle. Closing sibling rows
               empty on stage progression alone would discard axes that are still arriving (the
               result would silently repeat random values along them), so for round-up numerators
               a sibling with a pending GLB keeps the constraint deferred until stage 7, and only
               one without closes earlier. Not before stage 5 (gh-ocannl-1961): a variable shared
               with another row's [Total_elems] shows no bound at all until that stage sizes the
               row, and from then on [solve_inequalities] runs draws at quiescence. *)
            let round_up_numerator =
              match constr with
              | Total_elems { numerator = Strided_var { round_up = true; _ }; _ } -> true
              | _ -> false
            in
            let unconditional_close =
              if round_up_numerator then is_stage7 stage else is_stage5_up stage
            in
            let other_vars : (row_var * provenance) list =
              List.filteri rev_row_vars ~f:(fun i _ -> i <> idx)
            in
            let closeable (v, _prov) =
              match find_row env.row_env v with
              | None
              | Some
                  (Bounds_row
                     { glb = Some { beg_dims = []; dims = []; bcast = Broadcastable; _ }; _ }) ->
                  true
              | Some (Bounds_row { is_in_param = false; glb = None; _ }) -> true
              | _ -> false
            in
            (* gh-ocannl-2025: a fixed total whose kept row is bounded by a closed use site holding
               fewer elements than the total cannot close its one sibling empty: the kept row's
               product divides its bound's, so that would refuse a satisfiable store ([c * a] with
               [c : 3 <- 4] and [a] a 12-element reshape). The kept row takes the bound, the widest
               reading of its use sites, which its own [Shape_row] would read at stage 6, and the
               sibling the rest. *)
            let kept_bounded_short =
              match (constr, other_vars) with
              | Total_elems { numerator = Num_elems n; divided_by = []; _ }, [ _ ]
                when is_resolve_at_use v || row_var_is_in_param v env -> (
                  match (collect_factors all_dims, find_row env.row_env v) with
                  | Some (closed, []), Some (Bounds_row { glb = Some glb; _ })
                    when closed > 0 && n % closed = 0 -> (
                      let rest = n / closed in
                      match subst_row env glb with
                      | { bcast = Broadcastable; beg_dims; dims; _ } as glb -> (
                          match collect_factors (beg_dims @ dims) with
                          | Some (bound, []) when bound > 0 && bound < rest && rest % bound = 0 ->
                              Some glb
                          | _ -> None)
                      | _ -> None)
                  | _ -> None)
              | _ -> None
            in
            match kept_bounded_short with
            | Some glb ->
                ( [
                    Row_eq { r1 = row_of_var v v_prov; r2 = glb; origin };
                    Rows_constr { r = rows; constr; origin };
                  ],
                  env )
            | None ->
                let may_close_early = (not round_up_numerator) || is_stage5_up stage in
                let other_eqs : constraint_ list =
                  List.concat_map other_vars ~f:(fun ((v, prov) as vp) ->
                      if unconditional_close || (may_close_early && closeable vp) then
                        let r1 = row_of_var v prov in
                        [
                          Row_eq
                            {
                              r1;
                              r2 = { beg_dims = []; dims = []; bcast = Broadcastable; prov };
                              origin;
                            };
                        ]
                      else [])
                in
                let flatten_now =
                  if round_up_numerator then
                    is_stage7 stage || (is_stage5_up stage && List.for_all other_vars ~f:closeable)
                  else is_stage5_up stage
                in
                if flatten_now then
                  let rows =
                    List.map rows ~f:(function
                      | { bcast = Row_var v'; _ } as r when equal_row_var v' v -> r
                      | r ->
                          {
                            r with
                            beg_dims = [];
                            dims = r.beg_dims @ r.dims;
                            bcast = Broadcastable;
                          })
                  in
                  let ineqs, env =
                    eliminate_rows_constraint ~depth:(depth + 1) stage origin ~glb rows constr env
                  in
                  (other_eqs @ ineqs, env)
                else (other_eqs @ [ Rows_constr { r = rows; constr; origin } ], env))
        | _ -> ([ Rows_constr { r = rows; constr; origin } ], env))

and eliminate_row_constraint ~depth stage origin ~terminal ~(glb : row option) (r : row)
    (constr : row_constraint) env : constraint_ list * environment =
  let keep_constr () =
    let ineqs, env = apply_row_constraint ~depth stage origin r constr env in
    List.fold ineqs ~init:([], env) ~f:(fun (ineqs, env) ineq ->
        match ineq with
        | Rows_constr { r = rows; constr; origin } ->
            let ineqs', env =
              eliminate_rows_constraint ~depth:(depth + 1) stage origin ~glb:None rows constr env
            in
            (ineqs @ ineqs', env)
        | ineq -> ([ ineq ], env))
  in
  match r with
  | { bcast = Broadcastable; _ } -> keep_constr ()
  | { beg_dims; dims; bcast = Row_var v; prov } -> (
      let r1 = row_of_var v prov in
      (* If glb is not provided from context, try to get it from the row environment. This is
         critical for non-terminal shapes where GLBs are populated through inequalities but wouldn't
         otherwise be available until Stage 6. However, we only use the environment GLB if it has
         fully resolved dimensions (no dimension variables), as partially resolved GLBs can prevent
         proper constraint resolution. *)
      let flanks_known (glb : row) = known_dims_product (glb.beg_dims @ glb.dims) in
      let unless_tail_open (glb : row) = Option.some_if (not (is_broadcastable glb.bcast)) glb in
      (* gh-ocannl-2035: a GLB the rank facts refute says nothing about [v]'s rank, but its trailing
         flank still bounds [v]'s last axes, as [v]'s alignment against it does at any rank. A row
         that must take axes reads that flank, or the axes it takes can contradict it: a draw fills
         it, and a fixed total takes it when it holds the total. A row free to close empty has no
         use for it. *)
      let refuted_flank (glb : row) =
        let flank : row =
          { beg_dims = []; dims = glb.dims; bcast = Broadcastable; prov = glb.prov }
        in
        match
          ( glb.dims,
            try Some (reduce_row_constraint constr ~beg_dims ~dims) with Shape_error _ -> None )
        with
        | [], _ -> None
        | ( _,
            Some
              (Total_elems { numerator = Strided_var { round_up = true; _ }; divided_by = []; _ }) )
          ->
            Some flank
        | _, Some (Total_elems { numerator = Num_elems n; divided_by = []; _ }) -> (
            match collect_factors glb.dims with Some (d, []) when d = n -> Some flank | _ -> None)
        | _ -> None
      in
      (* [pending_glb]: the bound the round-up arms must wait for. An open tail is as unresolved as
         a variable in a flank: axes may still arrive there (gh-ocannl-1961). [from_refuted]: [glb]
         is a refuted GLB's flank. *)
      let glb, pending_glb, from_refuted =
        match glb with
        | Some ctx_glb ->
            (* gh-ocannl-1868: a GLB handed down by the caller can be as unresolved as one read from
               the environment below. It stays available to the arms that only inspect its shape,
               but the round-up arms must not take it for settled. *)
            let ctx_glb = subst_row env ctx_glb in
            ( Some ctx_glb,
              (if flanks_known ctx_glb then unless_tail_open ctx_glb else Some ctx_glb),
              false )
        | None -> (
            match find_row env.row_env v with
            | Some (Bounds_row { glb = Some env_glb; _ }) ->
                (* We need to substitute environment dimensions into the GLB to see if it's
                   resolved *)
                let env_glb = subst_row env env_glb in
                if rank_refutes_glb v env_glb then
                  match refuted_flank env_glb with
                  | Some flank when flanks_known flank -> (Some flank, None, true)
                  | Some flank -> (None, Some flank, false)
                  | None -> (None, None, false)
                else if flanks_known env_glb then (Some env_glb, unless_tail_open env_glb, false)
                else
                  (* GLB has unresolved dimension variables or collect_factors failed: bounds exist
                     but are still pending (gh-509 task 5) -- round-up guesses below must wait for
                     them to resolve rather than committing prematurely. *)
                  (None, Some env_glb, false)
            | _ -> (None, None, false))
      in
      let glb_pending = Option.is_some pending_glb in
      let glb_has_axes =
        Option.exists glb ~f:(fun glb -> not (List.is_empty (glb.beg_dims @ glb.dims)))
      in
      let opt_row_error () =
        if row_var_is_in_param v env && not (is_safe_to_guess v) then
          raise
          @@ Shape_error ("You forgot to specify the hidden dimension(s) 2", [ Row_mismatch [ r ] ])
      in
      let no_further_axes ~guess () =
        if guess then opt_row_error ();
        (* Close the bare row variable to empty Broadcastable; the original row's beg_dims is
           preserved at substitution time by s_row_one's uniform composition. *)
        Row_eq { r1; r2 = { beg_dims = []; dims = []; bcast = Broadcastable; prov }; origin }
      in
      (* Note: the reduced constraint applies to just the row variable. *)
      match reduce_row_constraint constr ~beg_dims ~dims with
      | Total_elems { numerator; divided_by; keep_axis } -> (
          let _divided_by : dim_var list = divided_by in
          (* Row-variable elimination runs at late stages only: an open total under a round-up slack
             window resolves to the window's top (all counter blocks fully consumed), matching the
             exact behavior of divisible shapes. The window's tolerance is only needed when
             re-checking against already-closed rows (in [apply_row_constraint]). *)
          let numerator = match numerator with Range_elems { hi; _ } -> Num_elems hi | n -> n in
          let numerator, divided_by =
            cancel_strided_var
              ~can_force:(fun coeff -> Utils.is_safe_val coeff || is_stage2_up stage)
              ~error_trace:[ Row_mismatch [ r ] ] numerator divided_by
          in
          match (numerator, divided_by, glb) with
          | Num_elems 1, [], (None | Some { beg_dims = []; dims = []; _ })
            when keep_axis && is_stage5_up stage ->
              (* Identity-reshape preference (gh-460): a 1-element data terminal whose rows received
                 no other shape information keeps a single dim-1 axis instead of collapsing to a
                 scalar. *)
              ( [
                  Row_eq
                    {
                      r1;
                      r2 =
                        {
                          beg_dims = [];
                          dims = [ get_default_dim ~d:1 ~proj_id:60 () ];
                          bcast = Broadcastable;
                          prov;
                        };
                      origin;
                    };
                ],
                env )
          | Num_elems 1, vs, _ when is_stage5_up stage ->
              ( no_further_axes ~guess:false ()
                :: List.map vs ~f:(fun v ->
                    let d2 = get_bcast_dim ~d:1 ~proj_id:54 () in
                    Dim_eq { d1 = Var v; d2; origin }),
                env )
          | Num_elems d, [], None when d <> 1 && is_stage3_up stage ->
              let dim = get_default_dim ~d ~proj_id:55 () in
              ( [
                  Row_eq
                    {
                      r1;
                      r2 = { beg_dims = []; dims = [ dim ]; bcast = Broadcastable; prov };
                      origin;
                    };
                ],
                env )
          | Num_elems d, [], Some { dims; _ } when d <> 1 && last_dim_is dims (( = ) d) ->
              let dim = get_default_dim ~d ~proj_id:56 () in
              ( [
                  Row_eq
                    {
                      r1;
                      r2 = { beg_dims = []; dims = [ dim ]; bcast = Broadcastable; prov };
                      origin;
                    };
                ],
                env )
          | Num_elems _, [], Some flank when from_refuted && is_stage3_up stage ->
              (* gh-ocannl-2035: the flank of a refuted GLB holding exactly the total. *)
              ([ Row_eq { r1; r2 = flank; origin } ], env)
          | Num_elems _, [], Some glb ->
              let ineqs, env =
                apply_row_constraint ~depth:(depth + 1) stage origin
                  (if terminal then glb else r)
                  constr env
              in
              List.fold ineqs ~init:([], env) ~f:(fun (ineqs, env) ineq ->
                  match ineq with
                  | Rows_constr { r = rows; constr; origin } ->
                      let ineqs', env =
                        eliminate_rows_constraint ~depth stage origin ~glb:None rows constr env
                      in
                      (ineqs @ ineqs', env)
                  | ineq -> ([ ineq ], env))
          | Num_elems d, [ dv ], None when is_stage4_up stage ->
              ( no_further_axes ~guess:true ()
                :: [ Dim_eq { d1 = Var dv; d2 = get_default_dim ~d (); origin } ],
                env )
          | Num_elems d, [ v ], Some ({ dims; _ } as r2)
            when last_dim_is dims (fun d2 -> d % d2 = 0) ->
              let d2 = match List.last dims with Some (Dim { d; _ }) -> d | _ -> assert false in
              let row_eq =
                if d = d2 && is_stage5_up stage then no_further_axes ~guess:false ()
                else Row_eq { r1; r2; origin }
              in
              (row_eq :: [ Dim_eq { d1 = Var v; d2 = get_default_dim ~d:(d / d2) (); origin } ], env)
          | Strided_var { round_up = true; _ }, [], _ when glb_pending && not (is_stage7 stage) ->
              (* gh-509 task 5: the row variable has registered bounds that have not resolved yet
                 (e.g. a composite parameter initializer over an inferred shape, where the bounds
                 settle only once the surrounding shapes do). Guessing the counter now would commit
                 the result prematurely; defer through the stored-constraint path (which dedups, so
                 the per-stage solver loop still terminates). *)
              keep_constr ()
          | Strided_var { round_up = true; _ }, [], _ when glb_pending ->
              (* gh-ocannl-1961: the final stage does not guess either, or a fixed-index axis
                 closing in this same stage would repeat the draw along it. Close the bound by its
                 own rules and come back: the arm consuming a known GLB derives the rest. The stored
                 path re-fires on the constraint's own variables only, so this one is re-emitted,
                 and its value is stable: with no progress the list reaches the solver's fixpoint
                 and [Shape.refuse_unsolved] reports it. Nothing can size a symbolic extent. *)
              let pending = Option.value_exn pending_glb in
              fail_if_total_elems_over_sym (pending.beg_dims @ pending.dims);
              ([ Rows_constr { r = [ r ]; constr; origin }; Shape_row (pending, origin) ], env)
          | Strided_var { coeff; var; denom; round_up = true }, [], None
            when is_stage5_up stage && denom > 1 && denom % Utils.safe_force coeff <> 0 ->
              (* Round-up totality: the known-axes product [denom] is not a multiple of [coeff];
                 instead of inventing an extra axis (the legacy arm below), close the leftover row
                 variable empty and give the counter [ceil(denom / coeff)] blocks -- the last block
                 is consumed partially. *)
              let coeff = Utils.safe_force coeff in
              let d2 = get_default_dim ~d:((denom + coeff - 1) / coeff) () in
              ([ Dim_eq { d1 = Var var; d2; origin }; no_further_axes ~guess:false () ], env)
          | Strided_var { coeff; var; denom; _ }, [], None
            when is_stage5_up stage
                 && (Utils.safe_force coeff > denom || denom % Utils.safe_force coeff <> 0) ->
              let coeff = Utils.safe_force coeff in
              let gcd = Utils.gcd coeff denom in
              let d = denom / gcd in
              let d2 = get_default_dim ~d () in
              let d3 = get_default_dim ~d:(coeff / gcd) () in
              (* opt_row_error (); *)
              ( [
                  Dim_eq { d1 = Var var; d2; origin };
                  Row_eq
                    {
                      r1;
                      r2 = { beg_dims = []; dims = [ d3 ]; bcast = Broadcastable; prov };
                      origin;
                    };
                ],
                env )
          | Strided_var { coeff; var; denom; round_up }, [], _
            when is_stage6_up stage
                 && denom % Utils.safe_force coeff = 0
                 && not (round_up && glb_has_axes) -> (
              (* gh-ocannl-1868: a round-up total (packed uniform) whose use sites give the row
                 variable axes belongs to the GLB arm below, also when the already-closed axes'
                 product [denom] happens to be a multiple of [coeff]: closing the variable empty
                 here left a [28 -> 28] parameter's draw at 28 elements, broadcast along the other
                 axis. *)
              match subst_dim env (Var var) with
              | (Affine _ | Concat _) as dim
                when denom = Utils.safe_force coeff && not (Set.is_empty (vars_of_dim dim)) ->
                  (* gh-ocannl-2159: do not force a composite with open parts to 1. Larger totals
                     keep the exact-solving path below: known row axes can size the parts. Close
                     free affine parts here, before operation results close at stage 6b. Revisit the
                     total explicitly: its stored form mentions only [var], so a part's solution
                     alone need not visit it. *)
                  let part_eqs =
                    match dim with
                    | Affine _ ->
                        Set.to_list (vars_of_dim dim)
                        |> List.concat_map ~f:(fun v ->
                            match find_dim env.dim_env v with
                            | None ->
                                refuse_concat_component_guess env v;
                                let d = if Set.mem env.discardable_vars v then 0 else 1 in
                                [ Dim_eq { d1 = Var v; d2 = get_bcast_dim ~d (); origin } ]
                            | Some _ -> close_dim_terminal ~stage ~is_param:false origin env (Var v))
                        |> List.filter ~f:(function Dim_eq _ -> true | _ -> false)
                    | _ -> []
                  in
                  if not (List.is_empty part_eqs) then
                    (* Apply both moves before the next shape in this pass can close: merely
                       emitting the equations lets a consumer see the still-open row. *)
                    let apply_equations eqs env =
                      List.fold eqs ~init:([], env) ~f:(fun (pending, env) eq ->
                          let more, env =
                            match eq with
                            | Dim_eq { d1; d2; origin } -> unify_dim ~stage origin (d1, d2) env
                            | Row_eq { r1; r2; origin } -> unify_row ~stage origin (r1, r2) env
                            | _ -> ([ eq ], env)
                          in
                          (pending @ more, env))
                    in
                    let pending, env = apply_equations part_eqs env in
                    let constr = subst_row_constraint stage env constr in
                    let more, env =
                      eliminate_row_constraint ~depth:(depth + 1) stage origin ~terminal ~glb:None
                        (subst_row env r) constr env
                    in
                    let more, env = apply_equations more env in
                    (pending @ more, env)
                  else if is_stage7 stage then
                    ( [
                        Rows_constr { r = [ r ]; constr; origin };
                        Shape_row
                          ({ beg_dims = []; dims = [ dim ]; bcast = Broadcastable; prov }, origin);
                      ],
                      env )
                  else keep_constr ()
              | _ ->
                  let d2 = get_default_dim ~d:(denom / Utils.safe_force coeff) () in
                  if dim_var_is_in_param var env then
                    raise
                    @@ Shape_error
                         ( "You forgot to specify the hidden dimension(s) 3",
                           [ Dim_mismatch [ Var var ] ] )
                  else ([ Dim_eq { d1 = Var var; d2; origin }; no_further_axes ~guess:true () ], env)
              )
          | ( Strided_var { coeff; var; denom; round_up },
              [],
              Some ({ beg_dims = glb_beg; dims = glb_dims; bcast = _; prov = glb_prov } as glb) )
            when is_stage5_up stage && (Utils.safe_force coeff > denom || round_up) -> (
              (* gh-509 task 5: under round-up this arm must also fire with [coeff <= denom]:
                 already-closed axes fold into [denom] (e.g. denom 10 from a 10-row parameter with
                 coeff 4), and no other arm consumes a Strided_var against a resolved GLB, so the
                 constraint would sit stored forever, leaving the counter variable unsolved. *)
              (* Check if coeff > denom * product of known dimensions of the GLB. The constraint is:
                 coeff * var / denom = total_elements(row). So: var = total_elements * denom / coeff
                 (rounded up for a round-up constraint). *)
              match collect_factors (glb_beg @ glb_dims) with
              | Some (known_product, []) ->
                  let coeff_val = Utils.safe_force coeff in
                  if (not round_up) && coeff_val > denom * known_product then
                    ([ Row_eq { r1; r2 = glb; origin } ], env)
                  else
                    (* Equate the row variable to the dimensions of the GLB, and compute var from
                       the total elements *)
                    let var_value =
                      match solve_strided_var ~coeff_val ~denom ~round_up known_product with
                      | Some v -> v
                      | None -> known_product * denom / coeff_val
                    in
                    ( [
                        Row_eq
                          {
                            r1;
                            r2 =
                              {
                                beg_dims = glb_beg;
                                dims = glb_dims;
                                bcast = Broadcastable;
                                prov = glb_prov;
                              };
                            origin;
                          };
                        Dim_eq { d1 = Var var; d2 = get_default_dim ~d:var_value (); origin };
                      ],
                      env )
              | _ -> keep_constr ())
          | Strided_var { coeff; var; denom; _ }, _, _ when is_stage5_up stage ->
              let _var : dim_var = var in
              let _coeff : int = Utils.safe_force coeff in
              let _denom : int = denom in
              keep_constr ()
          | _ -> keep_constr ())
      | Exact dims ->
          ([ Row_eq { r1; r2 = { beg_dims = []; dims; bcast = Broadcastable; prov }; origin } ], env)
      | Unconstrained -> ([], env))

(* A row's entry as closing it may read it: a GLB {!rank_refutes_glb} puts out of the variable's
   reach is no target to close it to (gh-ocannl-1955). Its flank still bounds the variable's last
   axes, which [eliminate_row_constraint] reads from the environment where it must give the row axes
   (gh-ocannl-2035). *)
let entry_for_closing env v =
  match find_row env.row_env v with
  | Some (Bounds_row ({ glb = Some glb; _ } as bounds)) when rank_refutes_glb v (subst_row env glb)
    ->
      Some (Bounds_row { bounds with glb = None })
  | entry -> entry

let%track5_sexp close_row_terminal ~(stage : stage) ~is_param origin env
    ({ beg_dims; dims; bcast; prov } as _r : row) : constraint_ list =
  let suffix () = List.map dims ~f:(fun d -> Terminal_dim (is_param, d, origin)) in
  (* TODO: can this be simplified? Should we return the environment? *)
  match bcast with
  | Broadcastable ->
      if is_stage6_up stage then []
      else List.map beg_dims ~f:(fun d -> Terminal_dim (is_param, d, origin)) @ suffix ()
  | Row_var v -> (
      let term_dims () =
        List.map beg_dims ~f:(fun d -> Terminal_dim (is_param, d, origin)) @ suffix ()
      in
      let r1 : row = row_of_var v prov in
      (* Close the bare row variable to empty Broadcastable; the original row's beg_dims is
         preserved through substitution via s_row_one's uniform composition. *)
      let no_further_axes =
        Row_eq { r1; r2 = { beg_dims = []; dims = []; bcast = Broadcastable; prov }; origin }
      in
      match entry_for_closing env v with
      | Some (Bounds_row { is_in_param; glb = None; constr = Unconstrained; _ })
        when is_stage4_up stage ->
          if (is_param || is_in_param) && not (is_safe_to_guess v) then
            raise @@ Shape_error ("You forgot to specify the hidden dimension(s) 4", [])
          else (
            [%log6 "terminal row: closing", (_r : row)];
            no_further_axes :: term_dims ())
      | Some (Bounds_row { glb = None; constr; _ })
        when is_stage2_up stage && not (equal_row_constraint constr Unconstrained) ->
          let ineqs, _env =
            (* This is the constraint on the row variable, not on the original row. *)
            try
              eliminate_row_constraint ~depth:0 stage origin r1 ~terminal:true ~glb:None constr env
            with Shape_error (s, trace) -> raise @@ Shape_error (s, Row_mismatch [ r1 ] :: trace)
          in
          (* FIXME: at which stage should we drop the terminal row? *)
          let keep_terminal =
            if is_stage6_up stage then [] else [ Terminal_row (is_param, r1, origin) ]
          in
          ineqs @ term_dims () @ keep_terminal
      | Some (Solved_row _) -> assert false
      | Some (Bounds_row { glb = Some _; constr = Total_elems { numerator = Num_elems 1; _ }; _ })
        when is_stage3_up stage ->
          term_dims ()
      | Some (Bounds_row { glb = Some glb; origin = glb_origin; _ }) when is_stage3_up stage ->
          let origin = merge_origins origin glb_origin in
          Row_eq { r1; r2 = glb; origin } :: term_dims ()
      | _ when is_stage6_up stage -> []
      | _ ->
          [%log6 "terminal row: keeping", (_r : row), "as", (r1 : row)];
          Terminal_row (is_param, r1, origin) :: term_dims ())

let%debug5_sexp eliminate_dim_entry stage origin env v ~defer_guess ~glb constr =
  let guess_dim () =
    refuse_concat_component_guess env v;
    if Set.mem env.discardable_vars v then get_default_dim ~d:0 ~proj_id:56 ()
    else get_bcast_dim ~d:1 ~proj_id:59 ()
  in
  match (glb, constr) with
  | Some (Dim { d; _ } as glb), At_least_dim { d_min = d2; _ } when d2 > d ->
      raise
      @@ Shape_error
           ( [%string "dereferenced at dimension %{d2#Int}, higher than use site"],
             [ Dim_mismatch [ glb; Var v ] ] )
  | Some _, At_least_dim { d_min = 1; _ } ->
      (* Direct access at 0 is a strong heuristic for dimension 1 axis (e.g. result of a
         reduction). *)
      if is_stage7 stage && not defer_guess then
        Some (Dim_eq { d1 = Var v; d2 = get_bcast_dim ~d:1 ~proj_id:57 (); origin })
      else None
  | Some glb, At_least_dim _ when is_stage6_up stage ->
      (* Direct indexing is a dim-carrying use: meet the use sites regardless of marking. *)
      Some (Dim_eq { d1 = Var v; d2 = glb; origin })
  | Some glb, Unconstrained_dim when is_stage6_up stage && is_resolve_at_use_dim v ->
      Some (Dim_eq { d1 = Var v; d2 = glb; origin })
  | Some _, Unconstrained_dim when is_stage7 stage && not defer_guess ->
      (* gh-ocannl-544, dim-level: an unmarked, unconstrained operation-result axis closes minimal
         instead of widening to its use sites' GLB (e.g. a learning-rate expression's axis meeting a
         single parameter shape closes to 1 and broadcasts). Deferred to stage 7 — not stage 6 —
         because stage-6 row closings still propagate concrete dims through equality chains (einsum
         specs), and an eager guess-to-1 would conflict with a dim that arrives within the same
         stage (box_muller's interior axes). *)
      Some (Dim_eq { d1 = Var v; d2 = guess_dim (); origin })
  | Some _, Unconstrained_dim when is_stage6_up stage -> None
  | None, At_least_dim { d_min = d; _ } when is_stage7 stage && not defer_guess ->
      Some (Dim_eq { d1 = Var v; d2 = get_default_dim ~d ~proj_id:58 (); origin })
  | None, _ when is_stage7 stage && not defer_guess ->
      Some (Dim_eq { d1 = Var v; d2 = guess_dim (); origin })
  | _ -> None

(* gh-ocannl-1508: a parameter dimension still open at the final stage. When what kept it open is a
   fixed-index lower bound -- [At_least_dim] is minted only for the fixed-index axes of einsum specs
   ([Shape.einsum_slot_spec_to_dims_bio]) -- the generic "hidden dimension" message points nowhere
   near the cause, so name the spec(s) the bound came from. They travel on the bound itself
   ([At_least_dim]'s [sites]), not among the capped origins, so the message holds at any
   [max_shape_error_origins] and graph depth (gh-ocannl-2044). *)
let unresolved_param_dim_error env v trace =
  match find_dim env.dim_env v with
  | Some (Bounds_dim { constr = At_least_dim { d_min; sites }; _ }) ->
      let kind_name = function `Batch -> "batch" | `Input -> "input" | `Output -> "output" in
      let specs =
        List.map sites ~f:(fun { spec; kind } ->
            [%string "a fixed-index %{kind_name kind} axis of einsum spec \"%{spec}\""])
        |> List.dedup_and_sort ~compare:String.compare
      in
      let where =
        match specs with
        | [] -> "a fixed-index axis of an einsum spec"
        | _ -> String.concat ~sep:"; " specs
      in
      Shape_error
        ( [%string
            "Unresolved parameter dimension: a parameter axis is sized only by %{where}. A fixed \
             index bounds an axis from below (here: at least %{d_min#Int}) but does not determine \
             its size; reduce the axis away instead of fixing an index, or specify the parameter's \
             dimension explicitly"],
          Dim_mismatch [ Var v ] :: trace )
  | _ -> Shape_error ("You forgot to specify the hidden dimension(s) 5", trace)

let%track5_sexp process_shape_row ?(deferred_dims = dim_var_set_empty) ~(stage : stage) origin env
    ({ beg_dims; dims; bcast; prov } as r : row) : constraint_ list * _ =
  let final = is_stage7 stage in
  let rec finalize_dim_bounds = function
    | Dim _ | Sym _ -> []
    | Affine { over; conv = None; _ } -> finalize_dim_bounds over
    | Affine { over; conv = Some { kernel; _ }; _ } ->
        finalize_dim_bounds over @ finalize_dim_bounds kernel
    | Concat dims -> List.concat_map dims ~f:finalize_dim_bounds
    | Var v -> (
        let defer_guess = Set.mem deferred_dims v in
        let guess_dim () =
          refuse_concat_component_guess env v;
          if Set.mem env.discardable_vars v then get_default_dim ~d:0 ~proj_id:61 ()
          else get_bcast_dim ~d:1 ~proj_id:62 ()
        in
        match find_dim env.dim_env v with
        | Some (Bounds_dim { is_in_param = true; _ }) when final ->
            raise @@ unresolved_param_dim_error env v [ Row_mismatch [ r ] ]
        | Some (Bounds_dim { glb; constr; has_uniq_constr_unless; _ })
          when is_stage4_up stage && can_guess_dim_to_one env has_uniq_constr_unless ->
            Option.to_list @@ eliminate_dim_entry stage origin env v ~defer_guess ~glb constr
        | Some (Solved_dim _) -> assert false
        | Some (Bounds_dim { has_uniq_constr_unless; _ })
          when final && (not defer_guess) && can_guess_dim_to_one env has_uniq_constr_unless ->
            [ Dim_eq { d1 = Var v; d2 = guess_dim (); origin } ]
        | None when final && not defer_guess -> [ Dim_eq { d1 = Var v; d2 = guess_dim (); origin } ]
        | _ -> [])
  in
  let rec has_dim_var = function
    | Dim _ | Sym _ -> false
    | Affine { over; conv = None; _ } -> has_dim_var over
    | Affine { over; conv = Some { kernel; _ }; _ } -> has_dim_var over || has_dim_var kernel
    | Concat dims -> List.exists dims ~f:has_dim_var
    | Var _ -> true
  in
  let process_dims dims = List.concat_map dims ~f:finalize_dim_bounds in
  let pending_window = not (Set.is_empty (Set.inter deferred_dims (dim_vars_of_row r))) in
  match bcast with
  | Broadcastable ->
      let keep =
        if pending_window || ((not final) && List.exists (beg_dims @ dims) ~f:has_dim_var) then
          [ Shape_row (r, origin) ]
        else []
      in
      (keep @ process_dims beg_dims @ process_dims dims, env)
  | Row_var v -> (
      let dim_eqs = process_dims beg_dims @ process_dims dims in
      let r1 : row = row_of_var v prov in
      let empty_broadcastable : row = { beg_dims = []; dims = []; bcast = Broadcastable; prov } in
      (* When an arm closes the row variable but the row still has unresolved dim vars (whose
         elimination may be deferred to a later stage, e.g. the [At_least_dim 1] guess-to-1 at stage
         7), re-emit the [Shape_row] so those vars get revisited: consuming the constraint here
         would leave them unresolved forever. *)
      let keep = if final && not pending_window then [] else [ Shape_row (r, origin) ] in
      match entry_for_closing env v with
      | Some (Bounds_row { glb = Some _; constr = Unconstrained; is_in_param; _ })
        when equal_stage stage Stage6a && not (is_in_param || is_resolve_at_use v) ->
          (* An operation result closes after its arguments (gh-ocannl-2126). Closed here, while a
             row that resolves at its use sites under it is still taking its bound, the result would
             end with fewer axes than its own operand, refusing a satisfiable program ("Too many
             axes in an operand", [(u + v) *. c] with two draws). *)
          (Shape_row (r, origin) :: dim_eqs, env)
      | Some (Bounds_row { glb = Some glb; constr = Unconstrained; is_in_param; _ })
        when is_stage6_up stage ->
          (* gh-ocannl-544: only rows that resolve at use sites (terminal rows of leaves and
             parameters, parameter-initialization cones, explicit [stretch]) close to the GLB of
             their use sites' rows. Operation results close to an empty remainder: a use site must
             broadcast the result in, not silently widen it (the gh-ocannl-540 cast-twin trap). *)
          let r2 = if is_in_param || is_resolve_at_use v then glb else empty_broadcastable in
          (keep @ (Row_eq { r1; r2; origin } :: dim_eqs), env)
      | Some (Bounds_row { constr = Unconstrained; _ }) when not final ->
          (Shape_row (r, origin) :: dim_eqs, env)
      | Some (Bounds_row { constr = Unconstrained; _ }) when final ->
          (keep @ (Row_eq { r1; r2 = empty_broadcastable; origin } :: dim_eqs), env)
      | Some
          (Bounds_row
             {
               glb =
                 Some
                   ({
                      beg_dims = [];
                      dims = [] | [ Dim { d = 1; _ } ];
                      bcast = Broadcastable;
                      prov = _;
                    } as glb);
               _;
             }) ->
          (* That's a greatest-lower-bound (most specific) value for a row. *)
          (keep @ (Row_eq { r1; r2 = glb; origin } :: dim_eqs), env)
      | Some (Bounds_row { glb; constr; _ }) ->
          let ineqs, env =
            try eliminate_row_constraint ~depth:0 stage origin r1 ~terminal:false ~glb constr env
            with Shape_error (s, trace) -> raise @@ Shape_error (s, Row_mismatch [ r1 ] :: trace)
          in
          (keep @ ineqs @ dim_eqs, env)
      | Some (Solved_row _) -> assert false
      | _ when final -> (keep @ (Row_eq { r1; r2 = empty_broadcastable; origin } :: dim_eqs), env)
      | _ -> (Shape_row (r, origin) :: dim_eqs, env))

let empty_env =
  {
    dim_env = Utils.Tree_map.empty;
    row_env = Utils.Tree_map.empty;
    discardable_vars = dim_var_set_empty;
    dims_mentioning = Map.empty (module Dim_var);
    rows_mentioning_dim = Map.empty (module Dim_var);
    rows_mentioning_row = Map.empty (module Row_var);
    pending_dims = dim_var_set_empty;
    pending_rows = row_var_set_empty;
  }

let update_dim_is_param d is_p env =
  if not is_p then env
  else
    match d with
    | Var v -> (
        match find_dim env.dim_env v with
        | None ->
            set_dim env ~key:v
              ~data:
                (Bounds_dim
                   {
                     is_in_param = true;
                     has_uniq_constr_unless = None;
                     res = [];
                     opnd = [];
                     glb = None;
                     constr = Unconstrained_dim;
                     origin = [];
                   })
        | Some (Bounds_dim b) ->
            let b = Bounds_dim { b with is_in_param = true } in
            set_dim env ~key:v ~data:b
        | _ -> env)
    | _ -> env

let update_row_is_param r is_p env =
  if not is_p then env
  else
    match r with
    | { bcast = Row_var v; _ } -> (
        match find_row env.row_env v with
        | None ->
            set_row env ~key:v
              ~data:
                (Bounds_row
                   {
                     is_in_param = true;
                     res = [];
                     opnd = [];
                     glb = None;
                     constr = Unconstrained;
                     origin = [];
                   })
        | Some (Bounds_row b) ->
            let b = Bounds_row { b with is_in_param = true } in
            set_row env ~key:v ~data:b
        | _ -> env)
    | _ -> env

let is_draw_total = function
  | Rows_constr { constr = Total_elems { numerator = Strided_var { round_up; _ }; _ }; _ } ->
      round_up
  | _ -> false

let%debug4_sexp solve_inequalities ~(stage : stage)
    ?(discardable_vars : dim_var_set = dim_var_set_empty) (ineqs : constraint_ list) env :
    constraint_ list * _ =
  let env = { env with discardable_vars = Set.union discardable_vars env.discardable_vars } in
  let rec solve ?(unblocked_dims = dim_var_set_empty) ineqs (env : environment) :
      constraint_ list * _ =
    (* gh-ocannl-2156: a held window equation is a dependency for closing, too. Its plain extent
       waits for the window; against a fixed extent, an open kernel waits for the iterator so kernel
       inference can use it. Recompute after substitution each pass, retaining the closing
       constraints until those equations settle. *)
    let waiting_on_window (waiting, kernels) extent window =
      match (extent, window) with
      | Var v, Affine { conv = Some { kernel = Var _; use_padding = false; _ }; _ } ->
          (Set.add waiting v, kernels)
      | Dim _, Affine { over; conv = Some { kernel = Var k; use_padding = false; _ }; _ }
        when not (Set.is_empty (vars_of_dim over)) ->
          (Set.add waiting k, Set.add kernels k)
      | _ -> (waiting, kernels)
    in
    let deferred_dims, deferred_kernels =
      if not (is_stage4_up stage) then (dim_var_set_empty, dim_var_set_empty)
      else
        List.fold ineqs ~init:(dim_var_set_empty, dim_var_set_empty) ~f:(fun waiting -> function
          | Dim_eq { d1; d2; _ } ->
              let d1 = subst_dim env d1 and d2 = subst_dim env d2 in
              waiting_on_window (waiting_on_window waiting d1 d2) d2 d1
          | _ -> waiting)
    in
    let deferred_dims = Set.diff deferred_dims unblocked_dims in
    let deferred_kernels = Set.diff deferred_kernels unblocked_dims in
    (* Process a single constraint and return new constraints + updated env *)
    let process_constraint env ineq =
      match ineq with
      | Dim_eq { d1; d2; origin } ->
          let _ineq : constraint_ = ineq in
          (* Substituted inside unify_dim. *)
          unify_dim ~stage origin (d1, d2) env
      | Row_eq { r1; r2; origin } ->
          (* Substituted inside unify_row. *)
          unify_row ~stage origin (r1, r2) env
      | Dim_ineq { res; opnd; origin; _ } ->
          let _ineq : constraint_ = ineq in
          let res = subst_dim env res and opnd = subst_dim env opnd in
          solve_dim_ineq ~stage origin ~res ~opnd env
      | Row_ineq { res; opnd; origin } ->
          let _ineq : constraint_ = ineq in
          let res = subst_row env res and opnd = subst_row env opnd in
          solve_row_ineq ~stage origin ~res ~opnd env
      | Dim_constr { d; constr; origin } ->
          let d = subst_dim env d in
          let extras, constr = apply_dim_constraint ~source:Direct ~stage d constr env in
          let env =
            match (constr, d) with
            | Unconstrained_dim, _ | _, Dim _ | _, Sym _ | _, Affine _ | _, Concat _ -> env
            | _, Var v ->
                let data =
                  match find_dim env.dim_env v with
                  | Some (Solved_dim _) -> assert false
                  | Some (Bounds_dim bounds) ->
                      let origin = merge_origins origin bounds.origin in
                      Bounds_dim { bounds with constr; origin }
                  | None ->
                      Bounds_dim
                        {
                          is_in_param = false;
                          has_uniq_constr_unless = None;
                          constr;
                          glb = None;
                          res = [];
                          opnd = [];
                          origin;
                        }
                in
                set_dim env ~key:v ~data
          in
          (extras, env)
      | Rows_constr { r = rows; constr; origin } ->
          let constr : row_constraint = subst_row_constraint stage env constr in
          reapply_rows_constr := false;
          let substituted_rows = List.map rows ~f:(subst_row env) in
          let (more_ineqs : constraint_ list), env =
            if is_stage3_up stage then
              eliminate_rows_constraint ~depth:0 stage origin ~glb:None substituted_rows constr env
            else apply_rows_constraint ~depth:0 ~stage origin substituted_rows constr env
          in
          (more_ineqs, env)
      | Terminal_dim (is_param, d, origin) ->
          let env = update_dim_is_param d is_param env in
          let d = subst_dim env d in
          (* Terminal (leaf and parameter) axes resolve at their use sites — protect them from the
             minimal-guess default of [eliminate_dim_entry] (gh-ocannl-544). *)
          Set.iter (vars_of_dim d) ~f:add_resolve_at_use_dim;
          let more_ineqs = close_dim_terminal ~deferred_dims ~stage ~is_param origin env d in
          (more_ineqs, env)
      | Terminal_row (is_param, r, origin) ->
          let env = update_row_is_param r is_param env in
          let r = subst_row env r in
          (* Terminal (leaf and parameter) rows resolve at their use sites — protect them from the
             close-down default of [process_shape_row] (gh-ocannl-544). *)
          (match r.bcast with Row_var v -> add_resolve_at_use v | Broadcastable -> ());
          let more_ineqs = close_row_terminal ~stage ~is_param origin env r in
          (more_ineqs, env)
      | Shape_row (r, origin) ->
          process_shape_row ~deferred_dims ~stage origin env @@ subst_row env r
    in

    (* Fold function that processes constraint, propagates origin, and handles errors *)
    let f (ineqs, env) ineq =
      try
        let more_ineqs, env = process_constraint env ineq in
        (more_ineqs @ ineqs, env)
      with Shape_error (s, trace) ->
        (* let f_out = Out_channel.open_text "shape_error.txt" in
           Stdlib.Printf.fprintf f_out "Shape_error: %s\nenv=\n%s\n%!" s
             (Sexp.to_string_hum @@ [%sexp_of: environment] env);
           Out_channel.close f_out; *)
        (* Add the failing constraint to the error trace *)
        let s =
          match ineq with
          | Dim_eq { d1; d2; _ } when is_stage5_up stage ->
              Option.value (unsized_window_pair_message env (d1, d2)) ~default:s
          | _ -> s
        in
        raise @@ Shape_error (s, Constraint_failed ineq :: trace)
    in
    (* At a fixpoint, from Stage2 on, nothing more will solve what an open window left open, and the
       closing stages need its size. One at a time, in that a choice can solve another window
       (gh-ocannl-1968). Stage1 waits: its constraints are still arriving. A window pair with both
       outputs open relates their sizes only once no open tag settles (gh-ocannl-2038). *)
    let rec settle_window ~merge_pairs kept = function
      | [] -> None
      | (Dim_eq { d1; d2; origin } as held) :: rest ->
          let fits constrs env =
            match solve constrs env with _ -> true | exception Shape_error _ -> false
          in
          let settling = { settled = false; merge_pairs; fits } in
          let more, env =
            try unify_dim ~settling ~stage origin (d1, d2) env
            with Shape_error (s, trace) ->
              raise @@ Shape_error (s, Constraint_failed held :: trace)
          in
          if settling.settled then Some (List.rev_append kept (more @ rest), env)
          else settle_window ~merge_pairs (held :: kept) rest
      | held :: rest -> settle_window ~merge_pairs (held :: kept) rest
    in
    let ineqs', env = List.fold ineqs ~init:([], env) ~f in
    let ineqs' = List.rev ineqs' in
    let settle () =
      match settle_window ~merge_pairs:false [] ineqs' with
      | Some _ as settled -> settled
      | None -> settle_window ~merge_pairs:true [] ineqs'
    in
    let at_fixpoint () =
      match if is_stage2_up stage then settle () else None with
      | Some (ineqs', env) -> solve ~unblocked_dims ineqs' env
      | None -> (
          (* Only at the final fixpoint, when nothing else can close, let one waiting dimension use
             ordinary closing and retry. Prefer a kernel, but a cycle with no fixed extent has only
             first-arm dependencies. Keep the choice even if it emits no guess. *)
          let next_guess =
            if not (is_stage7 stage) then None
            else
              match Set.min_elt deferred_kernels with
              | Some _ as kernel -> kernel
              | None -> Set.min_elt deferred_dims
          in
          match next_guess with
          | Some dim -> solve ~unblocked_dims:(Set.add unblocked_dims dim) ineqs' env
          | None -> (ineqs', env))
    in
    if List.is_empty ineqs' || List.equal equal_constraint_up_to_origin ineqs' ineqs then
      at_fixpoint ()
    else solve ~unblocked_dims ineqs' env
  in
  (* gh-ocannl-1961: what closes a draw's rows reads the environment, which mid-pass lies: a solved
     bound has left its operands' [res] while the [Row_ineq] carrying it is still in the list. So
     from the stage the draw arms start, a draw runs only once everything else has settled. *)
  let rec draws_at_quiescence ineqs env =
    let draws, others = List.partition_tf ineqs ~f:is_draw_total in
    let others, env = solve others env in
    let draws', env = solve draws env in
    if List.equal equal_constraint_up_to_origin draws' draws then (others @ draws, env)
    else draws_at_quiescence (others @ draws') env
  in
  match stage with
  | Stage5 | Stage6a | Stage6b -> draws_at_quiescence ineqs env
  | _ -> solve ineqs env

let rec row_to_bases env =
  let rec f = function
    (* Basis is total: return the actual tag (including [default] and [bcast_if_1]) so the
       provenance split is visible to callers. *)
    | Dim { basis = b; _ } -> b
    | Sym { sym_basis = b; _ } -> b
    | Var v -> (
        match find_dim env.dim_env v with
        | None | Some (Bounds_dim _) -> Option.value v.name ~default:""
        | Some (Solved_dim dim) -> f dim)
    | Affine _ -> ""
    | Concat dims -> List.map dims ~f |> String.concat ~sep:"^"
  in
  function
  | { beg_dims; dims; bcast = Row_var v; prov } -> (
      match find_row env.row_env v with
      | None | Some (Bounds_row _) -> Array.of_list_map (beg_dims @ dims) ~f
      | Some (Solved_row { beg_dims = beg_dims2; dims = dims2; bcast; _ }) ->
          row_to_bases env { beg_dims = beg_dims @ beg_dims2; dims = dims2 @ dims; bcast; prov })
  | { beg_dims; dims; bcast = Broadcastable; prov = _ } -> Array.of_list_map (beg_dims @ dims) ~f

(** *** Projection inference *** *)

let fresh_row_proj r =
  let rec fresh_dim = function
    | Dim { d; basis; proj_id = _ } -> Dim { d; basis; proj_id = Some (Proj_id.fresh ()) }
    | Sym s -> Sym { s with sym_proj_id = Some (Proj_id.fresh ()) }
    | Var _ as d -> d
    | Affine { stride; over; conv; stride_offset } ->
        let conv =
          Option.map conv ~f:(fun { dilation; kernel; use_padding } ->
              { dilation; kernel = fresh_dim kernel; use_padding })
        in
        Affine { stride; over = fresh_dim over; conv; stride_offset }
    | Concat dims -> Concat (List.map dims ~f:fresh_dim)
  in
  { r with beg_dims = List.map r.beg_dims ~f:fresh_dim; dims = List.map r.dims ~f:fresh_dim }

let populate_dim_proj_in_solved env =
  let rec fresh_dim = function
    | Dim { d; basis; proj_id = None } -> Dim { d; basis; proj_id = Some (Proj_id.fresh ()) }
    | Sym ({ sym_proj_id = None; _ } as s) -> Sym { s with sym_proj_id = Some (Proj_id.fresh ()) }
    | (Dim _ | Sym _ | Var _) as d -> d
    | Affine { stride; over; conv; stride_offset } ->
        let conv =
          Option.map conv ~f:(fun { dilation; kernel; use_padding } ->
              { dilation; kernel = fresh_dim kernel; use_padding })
        in
        Affine { stride; over = fresh_dim over; conv; stride_offset }
    | Concat dims -> Concat (List.map dims ~f:fresh_dim)
  in
  let fresh_row ({ beg_dims; dims; bcast; prov } : row) : row =
    let beg_dims = List.map beg_dims ~f:fresh_dim in
    let dims = List.map dims ~f:fresh_dim in
    { beg_dims; dims; bcast; prov }
  in
  let f_dim = function Solved_dim dim -> Solved_dim (fresh_dim dim) | entry -> entry in
  let f_row = function Solved_row row -> Solved_row (fresh_row row) | entry -> entry in
  (* Proj ids change no entry's mentions, nor whether an unrelated visit changes it: the index
     carries over. *)
  {
    env with
    dim_env = Utils.Tree_map.map env.dim_env ~f:f_dim;
    row_env = Utils.Tree_map.map env.row_env ~f:f_row;
  }

(* let update_proj_classes pid1 pid2 proj_classes = Utils.union_add ~equal:Int.equal proj_classes
   pid1 pid2 *)

type convolution_proj = { dilation : int; kernel : proj; kernel_size : int; use_padding : bool }
[@@deriving compare, equal, sexp]

and proj =
  (* TODO: remove this variant Var to see if it breaks anything *)
  | Var of dim_var
  | Proj of proj_id * solved_dim
  | Solved of Idx.axis_index
  | Conv_input of {
      stride : int;
      over : proj;
      conv : convolution_proj option;
      stride_offset : int;
      mutable target_id : proj_id option;
    }
  | Concat of (proj_id * solved_dim) list
[@@deriving compare, equal, sexp]

type error_trace += Projection_mismatch of proj list

let sexp_of_error_trace = function
  | Projection_mismatch ps ->
      Sexp.List (Sexp.Atom "Projection_mismatch" :: List.map ps ~f:sexp_of_proj)
  | error_trace -> sexp_of_error_trace error_trace

type proj_to_index = Idx.axis_index Map.M(Proj_id).t [@@deriving sexp]
type proj_classes = Proj_id.t Map.M(Proj_id).t [@@deriving sexp]

type committed_padding = { padding : axis_padding; operand : string; at_creation : bool }
[@@deriving sexp_of]

type proj_env = {
  v_env : (dim_var, proj) Hashtbl.t;
  proj_to_index : proj_to_index;
  resolved_padding : committed_padding list Map.M(Proj_id).t;
      (** Every committed layout read through the class, one entry per locked operand axis. *)
  inferred_padding : axis_padding Hashtbl.M(Proj_id).t;
  proj_classes : proj_classes;
  product_dim : int Map.M(Proj_id).t;
  clamp_padded : bool;
      (** gh-504: the operation's accumulation identity is non-finite (max/tropical family), so its
          padded ([=]-mode) window projections lower with clamped window bounds instead of margin
          demands: [get_proj_index] skips padding registration (and the exceeds-resolved check) for
          them, and the lowering guards the accumulation to the operand's valid range — an
          out-of-range position contributes the accumulation identity, which is the same as not
          visiting it. *)
}
[@@deriving sexp_of]

type proj_equation = Proj_eq of proj * proj | Iterated of proj [@@deriving compare, equal, sexp]

(** The refusal of a padded window whose margins exceed a committed layout (gh-ocannl-2041): it
    names the operand and gives only the remedies that apply to how its layout was committed. *)
let exceeds_committed_padding_message ~(operation_padding : axis_padding)
    { padding = resolved_pad; operand; at_creation } =
  let need = [%string "left=%{operation_padding.left#Int}, right=%{operation_padding.right#Int}"] in
  let head =
    [%string
      "Operation padding (%{need}) exceeds resolved padding (left=%{resolved_pad.left#Int}, \
       right=%{resolved_pad.right#Int}) of the operand %{operand}: "]
  in
  let copy =
    "read through a materialized copy of the operand (for a padded max-pool, \
     Nn_blocks.max_pool2d_copy)"
  in
  if at_creation then
    [%string
      "%{head}its buffer layout was committed at creation, since it keeps the array it was given \
       (wrap, wrap_padded, wrap_param, a loaded checkpoint). Create the data through reshape or \
       init, which copy it into a padded buffer, or through wrap_padded with margins of at least \
       (%{need}); or %{copy}"]
  else
    [%string
      "%{head}its buffer layout was committed at its first compilation or host access, before this \
       padded consumer was composed. Compose the padded consumer before that first compilation or \
       host access, or %{copy}"]

(** The refusal of a window kernel nothing sized (gh-ocannl-1691): no operand axis sizes it, and a
    padded window's extent equation omits it. Raised at construction when the spec and captures
    alone decide it ([Shape.propagate_shapes], gh-ocannl-1803, gh-ocannl-1983), by
    [get_proj_equations] for the kernel its local solve left a variable or closed by guess, and by
    [dim_to_proj] for one the projection environment does not reach (gh-ocannl-1734). [spec] is the
    einsum spec, where the raiser has one. *)
let unsized_kernel_error ?spec ~use_padding (v : dim_var) (dim : dim) =
  let name = Option.value v.name ~default:(Dim_var.to_string v) in
  let window = if use_padding then "a padded (=+) window's" else "the window's" in
  let in_spec =
    Option.value_map spec ~default:"" ~f:(fun spec -> [%string " in einsum spec \"%{spec}\""])
  in
  Shape_error
    ( [%string
        "the window kernel %{name} has no size%{in_spec}: no operand axis sizes it, and %{window} \
         extents do not determine it. Capture the kernel label and set it (x ++ \"s, 2*o=+k => s, \
         o\" [\"k\"], then Shape.set_dim k 3), or give the window a kernel operand of known extent \
         (x +* \"s, 2*o=+k; k => s, o\" w)"],
      [ Dim_mismatch [ dim ] ] )

let symbolic_kernel_error dim =
  Shape_error
    ( "projection_of_solved_dims: symbolic dimension not supported as convolution kernel",
      [ Dim_mismatch [ dim ] ] )

(** gh-ocannl-1971: a component with a symbolic extent keeps a [Concat] in a closed row, there being
    no [Dim] to sum it to. Where the segments after it start depends on the launch-time extent,
    which no index of a [Concat] can say, and such an axis was never lowered. *)
let symbolic_concat_error dim =
  Shape_error
    ( "a concatenation component has a symbolic extent, which is not supported: where the segments \
       after it start would depend on the launch-time extent. Concatenate along an axis of fixed \
       size",
      [ Dim_mismatch [ dim ] ] )

let%track4_sexp get_proj_equations ?spec (inequalities : constraint_ list) proj_axis_env env :
    proj_equation list =
  (* The difference between to_proj and dim_to_proj is that here we do not have a projection
     environment. *)
  let rec to_proj : dim -> proj = function
    | Var v when Map.mem proj_axis_env v -> Solved (Map.find_exn proj_axis_env v)
    | Dim { d = 0; _ } -> Solved (Fixed_idx 0)
    | Dim ({ proj_id = Some proj_id; _ } as solved_dim) -> Proj (proj_id, solved_dim)
    | Dim { d = 1; basis = _; proj_id = None } ->
        (* d=1 dims created during constraint solving don't need iteration *)
        Solved (Fixed_idx 0)
    | Dim { d; basis; proj_id = None } ->
        raise
        @@ Shape_error
             ("to_proj: Dim without proj_id (d=" ^ Int.to_string d ^ ", basis=" ^ basis ^ ")", [])
    | Sym ({ sym_proj_id = Some proj_id; _ } as s) ->
        (* Symbolic axes iterate at their maximum extent until launch-time binding lands. *)
        Proj (proj_id, solved_dim_of_sym s)
    | Sym ({ sym_proj_id = None; _ } as s) ->
        raise @@ Shape_error ("to_proj: Sym without proj_id", [ Dim_mismatch [ Sym s ] ])
    | Affine { stride; over; conv = None; stride_offset } ->
        (* Strided iteration: no convolution *)
        Conv_input { stride; over = to_proj over; conv = None; stride_offset; target_id = None }
    | Affine { stride; over; conv = Some { dilation; kernel; use_padding }; stride_offset } ->
        let kernel_size =
          match (subst_dim ~keep_affine:true env kernel, kernel) with
          | (Var v as dim), _ ->
              (* gh-ocannl-1691: no operand axis carries the kernel, and a padded window's extent
                 equation omits it, so nothing sized it. A captured, set kernel reaches here solved
                 (the projection-local sizing constraints of [Shape.bind_delayed_vars_to_envs]). *)
              raise @@ unsized_kernel_error ?spec ~use_padding v dim
          | (Dim { d = 1; basis; _ } as dim), Var v
            when use_padding && String.equal basis bcast_if_1 ->
              (* gh-ocannl-1983: an axis of the result (or of an operand) carried the kernel, and
                 nothing sized that axis, so closure guessed it: for a padded window the guess would
                 silently change the taps, not just an extent. An explicit 1 is [default]. *)
              raise @@ unsized_kernel_error ?spec ~use_padding v dim
          | Dim { d; _ }, _ -> d
          | (Sym _ as dim), _ -> raise @@ symbolic_kernel_error dim
          | (Affine _ as dim), _ ->
              raise
              @@ Shape_error
                   ("projection_of_solved_dims: still not fully inferred", [ Dim_mismatch [ dim ] ])
          | (Concat _ as dim), _ ->
              raise
              @@ Shape_error
                   ( "projection_of_solved_dims: concat not supported as kernel",
                     [ Dim_mismatch [ dim ] ] )
        in
        Conv_input
          {
            stride;
            over = to_proj over;
            conv = Some { dilation; kernel = to_proj kernel; kernel_size; use_padding };
            stride_offset;
            target_id = None;
          }
    | d -> (
        match subst_dim ~keep_affine:true env d with
        | Dim { d = 0; _ } -> Solved (Fixed_idx 0)
        | Dim ({ proj_id = Some proj_id; _ } as solved_dim) -> Proj (proj_id, solved_dim)
        | Dim { d = 1; basis = _; proj_id = None } ->
            (* d=1 dims created during constraint solving don't need iteration *)
            Solved (Fixed_idx 0)
        | Dim { d; basis; proj_id = None } ->
            raise
            @@ Shape_error
                 ( "to_proj (subst): Dim without proj_id (d=" ^ Int.to_string d ^ ", basis=" ^ basis
                   ^ ")",
                   [] )
        | Sym ({ sym_proj_id = Some proj_id; _ } as s) -> Proj (proj_id, solved_dim_of_sym s)
        | Sym ({ sym_proj_id = None; _ } as s) ->
            raise @@ Shape_error ("to_proj (subst): Sym without proj_id", [ Dim_mismatch [ Sym s ] ])
        | Var v when Map.mem proj_axis_env v -> Solved (Map.find_exn proj_axis_env v)
        | Var v -> Var v
        | Affine _ as affine -> to_proj affine
        | Concat dims ->
            let proj_dims =
              List.map dims ~f:(fun d ->
                  match subst_dim ~keep_affine:true env d with
                  | Dim ({ proj_id = Some proj_id; _ } as solved_dim) -> (proj_id, solved_dim)
                  | Dim { d; basis; proj_id = None } ->
                      let proj_id = Proj_id.fresh () in
                      (proj_id, { d; basis; proj_id = Some proj_id })
                  | Sym _ as dim -> raise @@ symbolic_concat_error dim
                  | _ ->
                      raise
                      @@ Shape_error
                           ("to_proj: concat component not fully solved", [ Dim_mismatch [ d ] ]))
            in
            Concat proj_dims)
  in
  let rec expand_dims = function
    | { beg_dims; dims; bcast = Row_var v; _ }
      when Utils.Tree_map.mem ~compare:compare_row_var ~key:v env.row_env -> (
        match find_row env.row_env v with
        | Some (Solved_row r) ->
            let more_dims = expand_dims r in
            beg_dims @ more_dims @ dims
        | _ -> beg_dims @ dims)
    | { beg_dims; dims; _ } -> beg_dims @ dims
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
    | Dim_ineq { res = _; opnd = Dim ({ d = 1; proj_id = Some proj_id; _ } as solved_dim); _ } ->
        [ Proj_eq (Proj (proj_id, solved_dim), Solved (Fixed_idx 0)) ]
    | Dim_eq { d1; d2; origin = _ } | Dim_ineq { res = d1; opnd = d2; _ } ->
        [ Proj_eq (to_proj d1, to_proj d2) ]
    | Row_eq { r1; r2; origin = _ } -> match_rows ~with_broadcasting:false r1 r2
    | Row_ineq { res = r1; opnd = r2; origin = _ } ->
        match_rows ~with_broadcasting:true r1 r2
        |> List.concat_map ~f:(function
          | Proj_eq (proj1, (Proj (_, { d = 1; _ }) as proj2)) ->
              [ Iterated proj1; Proj_eq (proj2, Solved (Fixed_idx 0)) ]
          | eq -> [ eq ])
    | Terminal_dim (_is_param, d, _origin) -> [ Iterated (to_proj d) ]
    | Terminal_row (_is_param, r, _origin) ->
        List.map ~f:(fun d -> Iterated (to_proj d)) (expand_dims r)
    | Shape_row _ -> []
    | Rows_constr
        {
          r;
          constr = Total_elems { numerator = Strided_var { coeff; var; denom; _ }; divided_by; _ };
          origin = _;
        } -> (
        let divided_by =
          List.map divided_by ~f:(fun v -> subst_dim ~keep_affine:true env (Var v))
        in
        match collect_factors divided_by with
        | Some (known_product, residual_vars) ->
            assert (List.is_empty residual_vars);
            let coeff = Utils.safe_force coeff in
            let denom = known_product * denom in
            (* TODO: check if this is correct *)
            if coeff % denom = 0 then
              let stride = coeff / denom in
              match rows_to_row_or_vars @@ List.map ~f:(subst_row env) r with
              | Second _ -> assert false
              | Either.First { dims; _ } -> (
                  (* The strided projection [stride * counter] carries the whole flat offset: it
                     must pair with the innermost axis of dimension > 1, because dim-1 axes'
                     projections are pinned to [Fixed_idx 0] elsewhere -- pairing with a trailing
                     dim-1 axis (e.g. a conv kernel's single input channel) would silently collapse
                     the stride and make every block store at offset 0. All other axes map to
                     [Sub_axis]; in the row-major flat offset trailing dim-1 axes then multiply the
                     accumulated offset by 1, so the store offsets are unchanged. *)
                  let rec split_at_inner rev_dims =
                    match rev_dims with
                    | [] -> (None, [])
                    | (Dim { d = 1; _ } as d1) :: rest ->
                        let inner, others = split_at_inner rest in
                        (inner, d1 :: others)
                    | inner :: rest -> (Some inner, rest)
                  in
                  match split_at_inner (List.rev dims) with
                  | None, all_ones ->
                      let output = subst_dim ~keep_affine:true env (Var var) in
                      Iterated (to_proj output)
                      :: List.map all_ones ~f:(fun d -> Proj_eq (to_proj d, Solved Sub_axis))
                  | Some inner, other_dims ->
                      let output = subst_dim ~keep_affine:true env (Var var) in
                      let input = to_proj inner in
                      Iterated (to_proj output)
                      :: Proj_eq
                           ( to_proj
                               (Affine { stride; over = output; conv = None; stride_offset = 0 }),
                             input )
                      :: List.map other_dims ~f:(fun d -> Proj_eq (to_proj d, Solved Sub_axis)))
            else assert false
        | None -> [])
    | Dim_constr _ | Rows_constr _ -> []
  in
  List.concat_map inequalities ~f

let unknown_projection proj_id d =
  raise
  @@ Shape_error
       ([%string "projection_of_solved_dims: unknown projection: %{proj_id#Proj_id} %{d#Int}"], [])

let%track7_sexp get_proj_index (proj_env : proj_env) (proj : proj) : Idx.axis_index =
  let rec loop (proj : proj) : Idx.axis_index =
    match proj with
    | Proj (proj_id, { d; _ }) -> (
        let repr, _ =
          Utils.union_find ~equal:Proj_id.equal proj_env.proj_classes ~key:proj_id ~rank:0
        in
        match (d, Map.find proj_env.proj_to_index repr) with
        | _, Some i -> i
        | (0 | 1), None -> Idx.Fixed_idx 0
        | _ -> unknown_projection proj_id d)
    | Solved idx -> idx
    | Conv_input { stride; over; conv = None; stride_offset; target_id = _ } ->
        (* Strided iteration: skip kernel computation since no convolution *)
        let over_idx = loop over in
        let symbols = ref [] in
        let offset = ref stride_offset in

        (* Expand over index - multiply by stride *)
        (match over_idx with
        | Idx.Fixed_idx i -> offset := !offset + (stride * i)
        | Idx.Sub_axis -> ()
        | Idx.Iterator s -> symbols := (stride, s) :: !symbols
        | Idx.Affine { symbols = over_syms; offset = over_offset } ->
            symbols := List.map over_syms ~f:(fun (c, s) -> (stride * c, s)) @ !symbols;
            offset := !offset + (stride * over_offset)
        | Idx.Concat syms -> symbols := List.map syms ~f:(fun s -> (stride, s)) @ !symbols);

        Idx.affine ~symbols:!symbols ~offset:!offset
    | Conv_input
        {
          stride;
          over;
          conv = Some { dilation; kernel; kernel_size; use_padding };
          stride_offset;
          target_id;
        } ->
        let over_idx = loop over in
        let kernel_idx = loop kernel in
        let symbols = ref [] in
        let offset = ref stride_offset in

        (* Expand over index - multiply by stride *)
        (match over_idx with
        | Idx.Fixed_idx i -> offset := !offset + (stride * i)
        | Idx.Sub_axis -> ()
        | Idx.Iterator s -> symbols := (stride, s) :: !symbols
        | Idx.Affine { symbols = over_syms; offset = over_offset } ->
            symbols := List.map over_syms ~f:(fun (c, s) -> (stride * c, s)) @ !symbols;
            offset := !offset + (stride * over_offset)
        | Idx.Concat syms -> symbols := List.map syms ~f:(fun s -> (stride, s)) @ !symbols);

        (match kernel_idx with
        | Idx.Fixed_idx i -> offset := !offset + (dilation * i)
        | Idx.Sub_axis -> ()
        | Idx.Iterator s -> symbols := (dilation, s) :: !symbols
        | Idx.Affine { symbols = kernel_syms; offset = kernel_offset } ->
            symbols := List.map kernel_syms ~f:(fun (c, s) -> (dilation * c, s)) @ !symbols;
            offset := !offset + (dilation * kernel_offset)
        | Idx.Concat syms -> symbols := List.map syms ~f:(fun s -> (dilation, s)) @ !symbols);

        (* Subtract padding if use_padding is true *)
        let offset =
          if use_padding then (
            (* The margins follow the DILATED kernel extent, so a dilated window is centered like a
               dense one: [o=+2*k] over a kernel of 3 reads [o - 2], [o], [o + 2]. Sizing them by
               the undilated kernel left a dilated window off-center, its right taps past the
               margins (gh-ocannl-1657). [left] is PyTorch's [padding='same'], so an even span's
               extra tap goes on the right: [o=+k] over a kernel of 4 reads [o - 1] to [o + 2], and
               a 2/2 pool tiles its operand, [2*o], [2*o + 1] (gh-ocannl-1692). *)
            let span = effective_kernel_span ~dilation ~kernel_size in
            let left_padding = (span - 1) / 2 in
            let right_padding = (span + 1) / 2 in
            let operation_padding : axis_padding =
              Ir.Ops.{ left = left_padding; right = right_padding }
            in

            (* Check and update padding based on projection ID from over *)
            (let check_and_update_padding proj_id =
               let repr, _ =
                 Utils.union_find ~equal:Proj_id.equal proj_env.proj_classes ~key:proj_id ~rank:0
               in
               let exceeded =
                 List.find
                   (Option.value ~default:[] @@ Map.find proj_env.resolved_padding repr)
                   ~f:(fun { padding = resolved_pad; _ } ->
                     operation_padding.left > resolved_pad.left
                     || operation_padding.right > resolved_pad.right)
               in
               match exceeded with
               | Some committed ->
                   raise
                   @@ Shape_error
                        ( exceeds_committed_padding_message ~operation_padding committed,
                          [ Projection_mismatch [ proj ] ] )
               | _ -> (
                   (* Update inferred padding to be sufficient for this operation, even if resolved
                      padding is present. *)
                   match Hashtbl.find proj_env.inferred_padding repr with
                   | Some existing_pad
                     when operation_padding.left > existing_pad.left
                          || operation_padding.right > existing_pad.right ->
                       let updated_pad : axis_padding =
                         Ir.Ops.
                           {
                             left = Int.max operation_padding.left existing_pad.left;
                             right = Int.max operation_padding.right existing_pad.right;
                           }
                       in
                       Hashtbl.set proj_env.inferred_padding ~key:repr ~data:updated_pad
                   | None -> Hashtbl.set proj_env.inferred_padding ~key:repr ~data:operation_padding
                   | Some _ -> (* Existing inferred padding is sufficient *) ())
             in
             (* gh-504: a clamp-lowered window (max/tropical family) demands no margins — the
                lowering guards the window to the operand's valid range instead — so neither the
                registration nor the exceeds-resolved check applies. *)
             if proj_env.clamp_padded then ()
             else
               match target_id with
               | Some proj_id -> check_and_update_padding proj_id
               | None -> () (* No target projection ID available to check *));

            !offset - left_padding)
          else !offset
        in

        Idx.affine ~symbols:!symbols ~offset
    | Var v when Hashtbl.mem proj_env.v_env v -> loop (Hashtbl.find_exn proj_env.v_env v)
    | Var v ->
        raise
        @@ Shape_error
             ( "projection_of_solved_dims: still not fully inferred for variable "
               ^ Sexp.to_string_hum ([%sexp_of: dim_var] v),
               [ Projection_mismatch [ proj ] ] )
    | Concat proj_dims ->
        (* For concatenation, collect all component iterators *)
        let syms =
          List.concat_map proj_dims ~f:(fun (proj_id, { d; _ }) ->
              let repr, _ =
                Utils.union_find ~equal:Proj_id.equal proj_env.proj_classes ~key:proj_id ~rank:0
              in
              match (d, Map.find proj_env.proj_to_index repr) with
              | _, Some (Idx.Iterator s) -> [ s ]
              | (0 | 1), _ -> []
              | _, Some (Idx.Fixed_idx _) -> []
              | _, Some (Idx.Affine { symbols; _ }) -> List.map symbols ~f:snd
              | _, Some (Idx.Concat syms) -> syms
              | _, Some Idx.Sub_axis -> []
              | _, None -> [])
        in
        if List.is_empty syms then Idx.Fixed_idx 0 else Idx.Concat syms
  in
  loop proj

let rec dim_to_proj proj_env : dim -> proj = function
  | Var v -> Var v
  | Dim ({ proj_id = Some proj_id; _ } as solved_dim) -> Proj (proj_id, solved_dim)
  | Dim s -> Proj (Proj_id.fresh (), s)
  | Sym ({ sym_proj_id = Some proj_id; _ } as s) -> Proj (proj_id, solved_dim_of_sym s)
  | Sym s -> Proj (Proj_id.fresh (), solved_dim_of_sym s)
  | Affine { stride; over; conv = None; stride_offset } ->
      Conv_input
        { stride; over = dim_to_proj proj_env over; conv = None; stride_offset; target_id = None }
  | Affine { stride; over; conv = Some { dilation; kernel; use_padding }; stride_offset } ->
      (* gh-ocannl-1734: the kernel's extent sizes a padded window's margins. The kernel here is the
         dim as the row holds it, not substituted, so a variable is read through the projection
         environment, as [get_proj_index] reads it; a kernel that environment does not size is the
         same refusal as [get_proj_equations]'. *)
      let rec extent_of_proj : proj -> int option = function
        | Proj (_, { d; _ }) -> Some d
        | Var v -> Option.bind (Hashtbl.find proj_env.v_env v) ~f:extent_of_proj
        (* A kernel pinned to one position: [to_proj] maps every extent-1 axis there. *)
        | Solved (Fixed_idx _) -> Some 1
        | Solved _ | Conv_input _ | Concat _ -> None
      in
      let kernel_size =
        match kernel with
        | Dim { d; _ } -> d
        | Var v -> (
            match extent_of_proj (Var v) with
            | Some d -> d
            | None -> raise @@ unsized_kernel_error ~use_padding v kernel)
        | Sym _ -> raise @@ symbolic_kernel_error kernel
        | Affine _ | Concat _ ->
            raise
            @@ Shape_error
                 ( "projection_of_solved_dims: a window kernel must be a single axis",
                   [ Dim_mismatch [ kernel ] ] )
      in
      Conv_input
        {
          stride;
          over = dim_to_proj proj_env over;
          conv = Some { dilation; kernel = dim_to_proj proj_env kernel; kernel_size; use_padding };
          stride_offset;
          target_id = None;
        }
  | Concat dims ->
      let proj_dims =
        List.map dims ~f:(fun d ->
            match d with
            | Dim ({ proj_id = Some proj_id; _ } as solved_dim) -> (proj_id, solved_dim)
            | Dim { d; basis; proj_id = None } ->
                let proj_id = Proj_id.fresh () in
                (proj_id, { d; basis; proj_id = Some proj_id })
            | Sym ({ sym_proj_id = Some proj_id; _ } as s) -> (proj_id, solved_dim_of_sym s)
            | Sym ({ sym_proj_id = None; _ } as s) ->
                let proj_id = Proj_id.fresh () in
                (proj_id, { (solved_dim_of_sym s) with proj_id = Some proj_id })
            | d ->
                raise
                @@ Shape_error
                     ("dim_to_proj: concat component not fully solved", [ Dim_mismatch [ d ] ]))
      in
      Concat proj_dims

let get_dim_index proj_env =
  let rec loop = function
    | Sym s ->
        (* Symbolic axes index like their materialized (maximum-extent) dimension. *)
        loop (Dim (solved_dim_of_sym s))
    | Dim { d; _ } when not @@ Idx.iterated d -> Idx.Fixed_idx 0
    | Dim { d; basis; proj_id = None } as dim ->
        (* gh-ocannl-1790: every route through [Shape] freshens the projection ids of the step's
           shapes ([Shape.fresh_proj_ids]) before deriving the equations and indexing those dims, so
           only a direct caller indexing some other dim reaches this. *)
        raise
        @@ Shape_error
             ( [%string
                 "get_dim_index: an iterated axis (d=%{d#Int}, basis=%{basis}) has no projection \
                  id, so the projection environment never saw it: index the dims the projection \
                  equations were derived from (their ids come from Row.fresh_row_proj)"],
               [ Dim_mismatch [ dim ] ] )
    | Var v when Hashtbl.mem proj_env.v_env v ->
        get_proj_index proj_env @@ Hashtbl.find_exn proj_env.v_env v
    | Var v as dim ->
        raise
        @@ Shape_error
             ( "projection_of_solved_dims: still not fully inferred for variable "
               ^ Sexp.to_string_hum ([%sexp_of: dim_var] v),
               [ Dim_mismatch [ dim ] ] )
    | Dim { proj_id = Some proj_id; d; _ } -> (
        let repr, _ =
          Utils.union_find ~equal:Proj_id.equal proj_env.proj_classes ~key:proj_id ~rank:0
        in
        match (d, Map.find proj_env.proj_to_index repr) with
        | _, Some i -> i
        | (0 | 1), None -> Fixed_idx 0
        | _ -> unknown_projection proj_id d)
    | Affine _ as dim -> get_proj_index proj_env (dim_to_proj proj_env dim)
    | Concat _ as dim -> get_proj_index proj_env (dim_to_proj proj_env dim)
  in
  loop

let%debug4_sexp solve_proj_equations ?(clamp_padded = false) (eqs : proj_equation list)
    ~(resolved_padding : (proj_id, committed_padding) List.Assoc.t)
    ~(inferred_padding : (proj_id, axis_padding) List.Assoc.t) : proj_env =
  let v_env = dim_hashtbl () in
  let p_solved = ref [] in
  let p_conv_input = ref [] in
  let verify_when_solved1 = ref [] in
  let verify_when_solved2 = ref [] in
  let p_dims = ref [] in
  let iterated_vars = ref [] in
  let p_concat_targets = ref [] in
  (* (target_pid, component proj_dims) for deferred Concat handling *)
  let p_concat_components = ref @@ Set.empty (module Proj_id) in
  (* proj_ids that are Concat components *)
  let proj_classes = ref @@ Map.empty (module Proj_id) in
  (* gh-ocannl-1790: the bindings must stay acyclic, since every reader of [v_env] follows them
     without a visited set ([get_proj_index], [dim_to_proj], the deferred passes below). [resolve]
     follows the bindings at the head of a projection; [occurrence v p] is the chain of variables
     through which [v] occurs in [p], a window's [over] and [kernel] included, if it does. Both
     terminate because the binding arm of [loop] keeps the bindings acyclic. *)
  let rec resolve = function
    | Var v as p -> ( match Hashtbl.find v_env v with Some p -> resolve p | None -> p)
    | p -> p
  in
  let rec occurrence v = function
    | Var w when equal_dim_var v w -> Some []
    | Var w ->
        Option.bind (Hashtbl.find v_env w) ~f:(fun p ->
            Option.map (occurrence v p) ~f:(fun chain -> w :: chain))
    | Conv_input { over; conv; _ } -> (
        match occurrence v over with
        | Some _ as found -> found
        | None -> Option.bind conv ~f:(fun { kernel; _ } -> occurrence v kernel))
    | Proj _ | Solved _ | Concat _ -> None
  in
  let var_name (v : dim_var) = Option.value v.name ~default:(Dim_var.to_string v) in
  let rec loop (eq : proj_equation) : unit =
    match eq with
    | Proj_eq (Proj (p1, { d; _ }), Proj (p2, _)) when Proj_id.equal p1 p2 ->
        p_dims := (p1, d) :: !p_dims
    | Proj_eq (Var v1, Var v2) when equal_dim_var v1 v2 -> ()
    | Proj_eq ((Proj (p1, { d = d1; _ }) as proj1), (Proj (p2, { d = d2; _ }) as proj2)) ->
        if d1 <> d2 then
          raise
          @@ Shape_error
               ( "Conflicting dimensions for the same projection",
                 [ Projection_mismatch [ proj1; proj2 ] ] );
        p_dims := (p1, d1) :: !p_dims;
        proj_classes := Utils.union_add ~equal:Proj_id.equal !proj_classes p1 p2
    | Proj_eq (Proj (p, _), Solved idx) | Proj_eq (Solved idx, Proj (p, _)) ->
        p_solved := (p, idx) :: !p_solved
    | Proj_eq (Proj (p, _), (Conv_input c as conv_input))
    | Proj_eq ((Conv_input c as conv_input), Proj (p, _)) ->
        (match c.target_id with
        | Some pid when Proj_id.equal p pid -> ()
        | Some pid -> proj_classes := Utils.union_add ~equal:Proj_id.equal !proj_classes pid p
        | None -> c.target_id <- Some p);
        (* [get_proj_index] will need indices for the compound's nested projections when this
           equation is processed below. Register them here: in a unary einsum the convolution window
           may have no standalone occurrence that would otherwise make it iterated. *)
        loop (Iterated conv_input);
        (* We will substitute variables in conv_input later *)
        p_conv_input := (p, conv_input) :: !p_conv_input
    | Proj_eq (Solved idx, ((Conv_input _ | Concat _) as conv_input))
    | Proj_eq (((Conv_input _ | Concat _) as conv_input), Solved idx) ->
        verify_when_solved1 := (idx, conv_input) :: !verify_when_solved1
    | Proj_eq
        ( (Conv_input { stride = stride1; over = over1; _ } as conv_input1),
          (Conv_input { stride = stride2; over = over2; _ } as conv_input2) )
      when stride1 = stride2 ->
        loop (Proj_eq (over1, over2));
        if equal_proj conv_input1 conv_input2 then ()
        else verify_when_solved2 := (conv_input1, conv_input2) :: !verify_when_solved2
    | Proj_eq ((Conv_input _ as conv_input1), ((Conv_input _ | Concat _) as conv_input2))
    | Proj_eq ((Concat _ as conv_input1), (Conv_input _ as conv_input2)) ->
        (* Conv_input vs Conv_input/Concat - defer verification *)
        if equal_proj conv_input1 conv_input2 then ()
        else verify_when_solved2 := (conv_input1, conv_input2) :: !verify_when_solved2
    | Proj_eq (Solved idx1, Solved idx2) when Idx.equal_axis_index idx1 idx2 -> ()
    | Proj_eq (Solved idx1, Solved idx2) ->
        raise
        @@ Shape_error
             ("Conflicting indices for the same axis/projection", [ Index_mismatch [ idx1; idx2 ] ])
    | Proj_eq (Var v1, Var v2) when equal_dim_var v1 v2 -> ()
    | Proj_eq (Var v, p) | Proj_eq (p, Var v) -> (
        match Hashtbl.find v_env v with
        | Some p2 -> loop (Proj_eq (p, p2))
        | None -> (
            match (resolve p, occurrence v p) with
            | Var w, _ when equal_dim_var v w ->
                (* The equation already holds through the bindings ([b = a] after [a = b]): binding
                   [v] would close the chain into a cycle. *)
                ()
            | _, None -> Hashtbl.add_exn v_env ~key:v ~data:p
            | _, Some chain ->
                (* [v] occurs inside a window of its own projection ([v = 2*v]), directly or through
                   the variables of [chain]: no binding can index it. *)
                let through =
                  if List.is_empty chain then ""
                  else " (through " ^ String.concat ~sep:", " (List.map chain ~f:var_name) ^ ")"
                in
                raise
                @@ Shape_error
                     ( [%string
                         "solve_proj_equations: the projection variable %{var_name v} is equated \
                          to a window over itself%{through}: the projection equations are cyclic"],
                       [ Projection_mismatch [ Var v; p ] ] )))
    | Iterated (Solved _) -> ()
    | Iterated (Proj (pid, { d; _ })) -> p_dims := (pid, d) :: !p_dims
    | Iterated (Conv_input { over; conv = None; _ }) ->
        (* Strided iteration: only process over since no convolution *)
        loop (Iterated over)
    | Iterated (Conv_input { over; conv = Some { kernel; _ }; _ }) ->
        loop (Iterated over);
        loop (Iterated kernel)
    | Iterated (Var v) -> (
        match Hashtbl.find v_env v with
        | None ->
            (* Defer: record that v needs iteration, will resolve after all equations processed *)
            iterated_vars := v :: !iterated_vars
        | Some proj -> loop @@ Iterated proj)
    | Iterated (Concat proj_dims) ->
        (* Each component of a concat needs iteration, including d=1 *)
        List.iter proj_dims ~f:(fun (pid, { d; _ }) ->
            p_dims := (pid, d) :: !p_dims;
            p_concat_components := Set.add !p_concat_components pid)
    | Proj_eq (Concat proj_dims1, Concat proj_dims2)
      when List.length proj_dims1 = List.length proj_dims2 ->
        (* Pairwise unification - projections must match structurally *)
        List.iter2_exn proj_dims1 proj_dims2 ~f:(fun (p1, s1) (p2, s2) ->
            p_concat_components := Set.add !p_concat_components p1;
            p_concat_components := Set.add !p_concat_components p2;
            loop (Proj_eq (Proj (p1, s1), Proj (p2, s2))))
    | Proj_eq (Concat proj_dims1, Concat proj_dims2) ->
        (* Mismatched Concat lengths - record all components for iteration *)
        List.iter proj_dims1 ~f:(fun (pid, { d; _ }) ->
            p_dims := (pid, d) :: !p_dims;
            p_concat_components := Set.add !p_concat_components pid);
        List.iter proj_dims2 ~f:(fun (pid, { d; _ }) ->
            p_dims := (pid, d) :: !p_dims;
            p_concat_components := Set.add !p_concat_components pid)
    | Proj_eq (Concat proj_dims, Proj (target_pid, _))
    | Proj_eq (Proj (target_pid, _), Concat proj_dims) ->
        (* Record components for iteration, including d=1 *)
        List.iter proj_dims ~f:(fun (pid, { d; _ }) ->
            p_dims := (pid, d) :: !p_dims;
            p_concat_components := Set.add !p_concat_components pid);
        (* Defer: target will get Concat index after components have iterators *)
        p_concat_targets := (target_pid, proj_dims) :: !p_concat_targets
  in
  let no_proj_assigned v =
    raise
    @@ Shape_error
         ( "Iterated variable has no projection assigned: "
           ^ Sexp.to_string_hum ([%sexp_of: dim_var] v),
           [] )
  in
  List.iter eqs ~f:loop;
  (* Process deferred iterated variables: they should now have projections assigned *)
  let pending_vars = !iterated_vars in
  iterated_vars := [];
  List.iter pending_vars ~f:(fun v ->
      match Hashtbl.find v_env v with
      | None -> no_proj_assigned v
      | Some proj -> loop @@ Iterated proj);
  (* Any variables added during the above iteration have no valid projection chain *)
  List.iter !iterated_vars ~f:no_proj_assigned;
  let repr_of p = fst @@ Utils.union_find ~equal:Proj_id.equal !proj_classes ~key:p ~rank:0 in
  (* gh-ocannl-1656: a concatenated axis exactly one of whose components another access names is an
     extraction ([p^o^q => o]: the result names [o]) or a replacement ([b => a^b^c]: the operand
     names [b]). Its index is a SHIFT, not a [Concat]: the named component's index plus the extents
     of the components before it ([s_o + p]). The unnamed components are iterated by nothing and get
     no iterator -- as [Concat] segments they lowered to loops whose every block was empty, and a
     width-1 named component, indexed at [Fixed_idx 0] like any dimension-1 axis, then selected no
     segment at all, so every segment read into (wrote from) the one cell. A concatenation proper
     ([a; b => a^b]: every component named) keeps its [Concat] index. *)
  let rec named = function
    | Proj (pid, _) -> [ pid ]
    | Conv_input { over; conv; _ } -> (
        named over @ match conv with Some { kernel; _ } -> named kernel | None -> [])
    | Var v -> ( match Hashtbl.find v_env v with Some p -> named p | None -> [])
    | Solved _ | Concat _ -> []
  in
  let named_reprs =
    let outside =
      List.concat_map eqs ~f:(function
        | Proj_eq (p1, p2) -> named p1 @ named p2
        | Iterated p -> named p)
      |> List.map ~f:repr_of
    in
    (* A component another concatenated axis also has is named by that access too: in [1^a; b =>
       a^b] the operand's [1^a] is a slice, and [a] is what the result's [a^b] concatenates. One
       axis can reach several equations, so the concatenations are told apart by their components,
       not counted per equation. *)
    let concats =
      List.concat_map eqs ~f:(function Proj_eq (p1, p2) -> [ p1; p2 ] | Iterated p -> [ p ])
      |> List.filter_map ~f:(function
        | Concat proj_dims ->
            Some
              (List.map proj_dims ~f:(fun (pid, _) -> repr_of pid)
              |> List.sort ~compare:Proj_id.compare)
        | _ -> None)
      |> List.dedup_and_sort ~compare:[%compare: Proj_id.t list]
    in
    let shared =
      List.concat_map concats ~f:(List.dedup_and_sort ~compare:Proj_id.compare)
      |> List.sort_and_group ~compare:Proj_id.compare
      |> List.filter_map ~f:(function p :: _ :: _ -> Some p | _ -> None)
    in
    Set.of_list (module Proj_id) (outside @ shared)
  in
  let all_concat_targets = !p_concat_targets in
  let shift_targets, concat_targets =
    List.partition_map all_concat_targets ~f:(fun ((target_pid, proj_dims) as target) ->
        match List.filter proj_dims ~f:(fun (pid, _) -> Set.mem named_reprs (repr_of pid)) with
        | [ (named_pid, _) ] when List.length proj_dims > 1 ->
            let offset, _ =
              List.fold proj_dims ~init:(0, false) ~f:(fun (offset, passed) (pid, { d; _ }) ->
                  if passed || Proj_id.equal pid named_pid then (offset, true)
                  else (offset + d, false))
            in
            First (target_pid, named_pid, offset, proj_dims)
        | _ -> Second target)
  in
  p_concat_targets := concat_targets;
  let shifted_components =
    List.concat_map shift_targets ~f:(fun (_, _, _, proj_dims) -> List.map proj_dims ~f:fst)
    |> Set.of_list (module Proj_id)
  in
  let kept_components =
    List.concat_map concat_targets ~f:(fun (_, proj_dims) -> List.map proj_dims ~f:fst)
    |> Set.of_list (module Proj_id)
  in
  p_concat_components :=
    Set.union
      (Set.diff !p_concat_components shifted_components)
      (Set.inter !p_concat_components kept_components);
  p_dims :=
    List.filter !p_dims ~f:(fun (p, _) ->
        Set.mem named_reprs (repr_of p) || not (Set.mem shifted_components p));
  let projs = ref @@ Map.empty (module Proj_id) in
  let concat_reprs =
    Set.of_list
      (module Proj_id)
      (Set.to_list !p_concat_components
      |> List.map ~f:(fun p ->
          fst @@ Utils.union_find ~equal:Proj_id.equal !proj_classes ~key:p ~rank:0))
  in
  List.iter !p_solved ~f:(fun (p, idx) ->
      let repr, _ = Utils.union_find ~equal:Proj_id.equal !proj_classes ~key:p ~rank:0 in
      if Idx.equal_axis_index idx (Idx.Fixed_idx 0) && Set.mem concat_reprs repr then ()
      else
        Utils.mref_add projs ~key:repr ~data:idx ~or_:(fun idx2 ->
            if not @@ Idx.equal_axis_index idx idx2 then
              raise
              @@ Shape_error
                   ("Multiple constraints on the same projection", [ Index_mismatch [ idx; idx2 ] ])));
  let product_dim = ref @@ Map.empty (module Proj_id) in
  let concat_dims =
    List.filter_map !p_dims ~f:(fun (p, d) ->
        if Set.mem !p_concat_components p then Some (p, d) else None)
  in
  (* Collect projection IDs that will get their index from Conv_input (target_id projections). These
     should NOT get fresh iterators from product_dim processing. *)
  let conv_input_targets =
    Set.of_list (module Proj_id)
    @@ List.filter_map !p_conv_input ~f:(fun (p, _) ->
        let repr, _ = Utils.union_find ~equal:Proj_id.equal !proj_classes ~key:p ~rank:0 in
        Some repr)
  in
  (* Likewise, a concatenated axis gets its index -- [Concat] over its components' iterators -- from
     the deferred concat-target pass below, never an iterator of its own (gh-ocannl-1656). The
     target is the result axis for a concatenation ([a; b => a^b]) but an OPERAND axis for an
     extraction ([a^b => a]), and every operand axis is [Iterated]: without this exclusion it got a
     fresh product iterator first and the deferred pass refused the conflict. Representatives are
     taken after all unions; a target that is itself a concat component keeps its iterator. *)
  let concat_target_reprs =
    Set.of_list (module Proj_id)
    @@ List.filter_map all_concat_targets ~f:(fun (p, _) ->
        let repr, _ = Utils.union_find ~equal:Proj_id.equal !proj_classes ~key:p ~rank:0 in
        if Set.mem concat_reprs repr then None else Some repr)
  in
  List.iter !p_dims ~f:(fun (p, d) ->
      let repr, _ = Utils.union_find ~equal:Proj_id.equal !proj_classes ~key:p ~rank:0 in
      (* Include d=1 if it's a concat component - they need iterators for proper Concat indexing *)
      let needs_iterator = Idx.iterated d || (d = 1 && Set.mem !p_concat_components p) in
      if needs_iterator && (not @@ Map.mem !projs repr) && not (Set.mem concat_target_reprs repr)
      then
        Utils.mref_add product_dim ~key:repr ~data:d ~or_:(fun d2 ->
            (* TODO: consider updating padding *)
            if d <> d2 then
              raise
              @@ Shape_error
                   ( [%string
                       "Conflicting dimensions for the same projection: %{p#Proj_id} %{d#Int} \
                        %{d2#Int}"],
                     [] )));
  List.iter concat_dims ~f:(fun (p, d) ->
      let repr, _ = Utils.union_find ~equal:Proj_id.equal !proj_classes ~key:p ~rank:0 in
      if not @@ Map.mem !product_dim repr then product_dim := Map.set !product_dim ~key:repr ~data:d);
  (* Create fresh iterators for product dimensions, EXCEPT for those that will get their index from
     Conv_input (they will be processed later). *)
  Map.iteri !product_dim ~f:(fun ~key:p ~data:_ ->
      let repr, _ = Utils.union_find ~equal:Proj_id.equal !proj_classes ~key:p ~rank:0 in
      if not (Set.mem conv_input_targets repr) then
        Utils.mref_add_missing projs repr ~f:(fun () -> Idx.(Iterator (get_symbol ()))));
  Set.iter concat_reprs ~f:(fun repr ->
      if Set.mem conv_input_targets repr then ()
      else
        match Map.find !projs repr with
        | Some (Idx.Iterator _) -> ()
        | _ -> projs := Map.set !projs ~key:repr ~data:Idx.(Iterator (get_symbol ())));

  (* Process p_conv_input to populate projs and compute padding *)
  let resolved_padding =
    (* Since locked (layout-committed) tensors register resolved padding for every axis, several
       axes of a class can carry entries; a padding demand routed through the class must fit every
       locked buffer read through it, so every entry is kept, and the check names the first that the
       demand exceeds (gh-ocannl-2041). *)
    Map.of_alist_multi (module Proj_id)
    @@ List.map resolved_padding ~f:(fun (p, committed) ->
        (fst @@ Utils.union_find ~equal:Proj_id.equal !proj_classes ~key:p ~rank:0, committed))
  in
  let inferred_padding =
    Hashtbl.of_alist_exn (module Proj_id)
    @@ List.map inferred_padding ~f:(fun (p, pad) ->
        (fst @@ Utils.union_find ~equal:Proj_id.equal !proj_classes ~key:p ~rank:0, pad))
  in

  let proj_env_so_far () =
    {
      v_env;
      proj_classes = !proj_classes;
      proj_to_index = !projs;
      inferred_padding;
      resolved_padding;
      product_dim = !product_dim;
      clamp_padded;
    }
  in
  (* Process postponed Conv_input equations, in dependency order (gh-ocannl-1971): a window's index
     is built from the indices of what it ranges over, so a window over another window's target ([u
     = 2*t] with [t = 1*x]) waits for that target, and each window reads every index assigned so
     far. When no window is ready the rest are cyclic ([t = 2*u], [u = 2*t]): each reads what it
     can, which is an unknown projection for an iterated axis. *)
  let index_window (p, conv_input) =
    let idx = get_proj_index (proj_env_so_far ()) conv_input in
    Utils.mref_add projs ~key:(repr_of p) ~data:idx ~or_:(fun idx2 ->
        if not @@ Idx.equal_axis_index idx idx2 then
          raise
          @@ Shape_error
               ( "Multiple constraints on the same Conv_input projection",
                 [ Index_mismatch [ idx; idx2 ] ] ))
  in
  let rec index_windows pending =
    let waits_for_no_target (_, conv_input) =
      List.for_all (named conv_input) ~f:(fun p ->
          let repr = repr_of p in
          Map.mem !projs repr || not (Set.mem conv_input_targets repr))
    in
    match List.partition_tf pending ~f:waits_for_no_target with
    | [], cyclic -> List.iter cyclic ~f:index_window
    | ready, blocked ->
        List.iter ready ~f:index_window;
        index_windows blocked
  in
  index_windows !p_conv_input;

  (* Now create fresh iterators for product dimensions that still don't have an index. This is done
     after p_conv_input processing so Conv_input projections don't conflict. *)
  Map.iteri !product_dim ~f:(fun ~key:p ~data:_ ->
      let repr, _ = Utils.union_find ~equal:Proj_id.equal !proj_classes ~key:p ~rank:0 in
      Utils.mref_add_missing projs repr ~f:(fun () -> Idx.(Iterator (get_symbol ()))));

  (* Process deferred concat targets: build Concat indices from component iterators *)
  List.iter !p_concat_targets ~f:(fun (target_pid, proj_dims) ->
      let target_repr, _ =
        Utils.union_find ~equal:Proj_id.equal !proj_classes ~key:target_pid ~rank:0
      in
      let syms =
        List.filter_map proj_dims ~f:(fun (pid, { d; _ }) ->
            let repr, _ = Utils.union_find ~equal:Proj_id.equal !proj_classes ~key:pid ~rank:0 in
            match Map.find !projs repr with
            | Some (Idx.Iterator s) -> Some s
            | Some (Idx.Fixed_idx 0) when d = 0 -> None (* d=0 is invalid dimension, skip *)
            | _ when d = 0 -> None
            | _ ->
                raise
                @@ Shape_error
                     ( [%string
                         "Concat component projection %{pid#Proj_id} (d=%{d#Int}) has no iterator"],
                       [] ))
      in
      let expected_idx = if List.is_empty syms then Idx.Fixed_idx 0 else Idx.Concat syms in
      match Map.find !projs target_repr with
      | None -> projs := Map.set !projs ~key:target_repr ~data:expected_idx
      | Some existing_idx ->
          let ok =
            Idx.equal_axis_index existing_idx expected_idx
            ||
            match (existing_idx, expected_idx) with
            | Idx.Iterator s, Idx.Concat [ s' ] when Idx.equal_symbol s s' -> true
            | _ -> false
          in
          if not ok then
            raise
            @@ Shape_error
                 ( [%string
                     "Concat target projection %{target_pid#Proj_id} conflicts with existing index"],
                   [ Index_mismatch [ existing_idx; expected_idx ] ] ));
  List.iter shift_targets ~f:(fun (target_pid, named_pid, offset, _) ->
      let expected_idx =
        match Map.find !projs (repr_of named_pid) with
        | Some (Idx.Iterator s) -> Idx.affine ~symbols:[ (1, s) ] ~offset
        | Some (Idx.Fixed_idx k) -> Idx.Fixed_idx (k + offset)
        | Some (Idx.Affine { symbols; offset = o }) -> Idx.affine ~symbols ~offset:(o + offset)
        | None -> Idx.Fixed_idx offset (* a dimension-1 (or empty) named component *)
        | Some idx ->
            raise
            @@ Shape_error
                 ( [%string
                     "Concat target projection %{target_pid#Proj_id}: the named component's index \
                      cannot be shifted"],
                   [ Index_mismatch [ idx ] ] )
      in
      let target_repr = repr_of target_pid in
      match Map.find !projs target_repr with
      | None -> projs := Map.set !projs ~key:target_repr ~data:expected_idx
      | Some existing_idx when Idx.equal_axis_index existing_idx expected_idx -> ()
      | Some existing_idx ->
          raise
          @@ Shape_error
               ( [%string
                   "Concat target projection %{target_pid#Proj_id} conflicts with existing index"],
                 [ Index_mismatch [ existing_idx; expected_idx ] ] ));

  let proj_env = proj_env_so_far () in
  (* gh-ocannl-1889: the equations no binding could carry -- a window or concatenation equated to a
     solved index, or to another window -- are checked once every index is assigned, against the
     environment the caller gets. They were checked before the indices of the windows' own targets
     and of the concat targets existed (so a window over another window's target read as an unknown
     projection), under a catch-all: every error was swallowed, and a window equated to an index it
     does not reach passed silently. Checked last, nothing here fails on a consistent program, so
     nothing is caught. An [Iterator] equals the [Concat] of that one iterator, as in the
     concat-target pass. A [Concat] side comes from an axis that did not close to a plain [Dim]:
     hand-built equations, since a symbolic component is refused by [get_proj_equations]
     (gh-ocannl-1971). *)
  let same_index idx1 idx2 =
    Idx.equal_axis_index idx1 idx2
    ||
    match (idx1, idx2) with
    | Idx.Iterator s, Idx.Concat [ s' ] | Idx.Concat [ s' ], Idx.Iterator s -> Idx.equal_symbol s s'
    | _ -> false
  in
  List.iter (List.rev !verify_when_solved1) ~f:(fun (idx, proj) ->
      let proj_idx = get_proj_index proj_env proj in
      if not @@ same_index idx proj_idx then
        raise
        @@ Shape_error
             ( "solve_proj_equations: a window or concatenation is equated to a solved index it \
                does not reach",
               [ Index_mismatch [ idx; proj_idx ]; Projection_mismatch [ Solved idx; proj ] ] ));
  List.iter (List.rev !verify_when_solved2) ~f:(fun (proj1, proj2) ->
      let idx1 = get_proj_index proj_env proj1 in
      let idx2 = get_proj_index proj_env proj2 in
      if not @@ same_index idx1 idx2 then
        raise
        @@ Shape_error
             ( "solve_proj_equations: two equated windows (or a window and a concatenation) index \
                differently",
               [ Index_mismatch [ idx1; idx2 ]; Projection_mismatch [ proj1; proj2 ] ] ));
  proj_env

let proj_repr proj_env p =
  fst @@ Utils.union_find ~equal:Proj_id.equal proj_env.proj_classes ~key:p ~rank:0

let rec get_product_proj proj_env dim =
  match dim with
  | Sym s -> get_product_proj proj_env (Dim (solved_dim_of_sym s))
  | Dim { proj_id = Some proj_id; d; _ } -> (
      let repr = proj_repr proj_env proj_id in
      if not (Map.mem proj_env.product_dim repr) then None
      else
        match Map.find proj_env.proj_to_index repr with
        | Some (Iterator _) -> Some (repr, d)
        | _ -> None)
  | Dim { proj_id = None; _ } -> None
  | Var v ->
      raise
      @@ Shape_error
           ( "projection_of_solved_dims: still not fully inferred for variable "
             ^ Sexp.to_string_hum ([%sexp_of: dim_var] v),
             [ Dim_mismatch [ dim ] ] )
  | Affine _ -> None
  | Concat _ -> None

let%debug6_sexp get_dim_padding (proj_env : proj_env) (dim : dim) : axis_padding option =
  match dim with
  | Dim { proj_id = Some proj_id; _ } | Sym { sym_proj_id = Some proj_id; _ } ->
      let repr = proj_repr proj_env proj_id in
      Hashtbl.find proj_env.inferred_padding repr
  | _ -> assert false

let proj_to_iterator_exn proj_env p =
  match Map.find_exn proj_env.proj_to_index (proj_repr proj_env p) with
  | Iterator s -> s
  | _ -> invalid_arg "proj_to_iterator_exn"

let product_dim_iterators proj_env =
  Map.to_alist proj_env.product_dim
  |> List.filter_map ~f:(fun (p, d) ->
      match Map.find proj_env.proj_to_index p with
      | Some (Idx.Iterator s) -> Some (p, d, s)
      | _ -> None)
