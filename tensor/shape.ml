(** {1 Tensor shape types, shape inference, projection inference.} *)

open Base
module Lazy = Utils.Lazy
module Idx = Ir.Indexing
module Tn = Ir.Tnode

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_SHAPE=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_SHAPE"]

(** {2 Shape types and inference.} *)

(* Re-export types from Einsum_parser (which includes Einsum_types) *)
include Einsum_parser

type padding = Row.axis_padding array option [@@deriving sexp, equal]

(** The shape record and its only writers (gh-ocannl-1506). The record is [private] outside this
    submodule -- here in [shape.ml] as much as in every other module -- so a field assignment, a
    record literal or a [{ sh with … }] copy anywhere else is a compile error, and every write goes
    through a setter that enforces the closed-shape guard (gh-ocannl-1301): a closed shape's rows
    (modulo projection ids), paddings and [padding_elem] no longer change, and a write that would
    change them raises before writing anything. The privacy is shallow: a padding array read off a
    shape is still an array. None of the setters counts as a session mutation ([Row.note_mutation]):
    substituting the environment into a shape is neutral, and paddings are written only by
    [finish_inference], whose failure taints the session anyway. *)
module Shape_record : sig
  type t = private {
    mutable batch : Row.t;
    mutable input : Row.t;
    mutable output : Row.t;
    mutable batch_padding : padding;
    mutable input_padding : padding;
    mutable output_padding : padding;
    mutable padding_elem : float option option;
        (** The padding element for this shape's tensors. [None] means "unknown" (not yet
            determined), [Some (Some v)] means all operations use neutral element [v], [Some None]
            means different operations require different neutral elements (margin must be reset
            before each operation). *)
    mutable closed : bool; [@sexp_drop_if fun _ -> true] [@default false] [@equal.ignore]
        (** Set by a successful [finish_inference] on every shape of the steps it commits
            (gh-ocannl-1301). Not part of the shape's value: dropped from sexps, ignored by [equal].
        *)
    mutable discarded_by : string option;
        [@sexp_drop_if fun _ -> true] [@default None] [@equal.ignore]
        (** Set by [Tensor.rewind_to_last_closure] on every open shape it discards, to the rewind's
            provenance: a step over a discarded shape is refused before it is registered. Not part
            of the shape's value. *)
    id : int;  (** A node that has the same shape as this shape. *)
    debug_name : string;
  }
  [@@deriving equal, sexp_of]

  type Row.error_trace += Shape_mismatch of t list

  val of_rows : batch:Row.t -> input:Row.t -> output:Row.t -> id:int -> debug_name:string -> t
  (** A fresh open shape with unknown paddings and neutral element. *)

  val copy : t -> t
  (** A shallow copy with its own row slots: same [id], [closed] and content ({!dealias_operands}).
  *)

  val set_rows : t -> batch:Row.t -> input:Row.t -> output:Row.t -> unit
  (** Guarded: on a closed shape, all three rows are checked equal modulo projection ids before any
      is written. *)

  val set_paddings : t -> batch:padding -> input:padding -> output:padding -> unit
  (** Guarded: on a closed shape, all three paddings are checked equal before any is written. *)

  val set_padding_elem : t -> float option option -> unit
  (** Guarded like the other setters. *)

  val close : t -> unit
  (** The commit of a successful [finish_inference]. *)

  val mark_discarded : t -> provenance:string -> unit
  (** Marks an open shape discarded by a rewind; a closed shape is left alone. *)

  type snapshot
  (** Every mutable field of a shape at a checkpoint, bound to the shape (gh-ocannl-903). The
      paddings' arrays are copied. *)

  val snapshot : t -> snapshot
  val snapshot_shape : snapshot -> t

  val restore : snapshot -> unit
  (** Puts the recorded fields back, bypassing the closed-shape guard: a rewind restores a shape
      that was open at the checkpoint (a closure since would have invalidated it). *)

  val snapshot_differences : snapshot -> string list
  (** The fields that differ from the snapshot -- rows and the neutral element physically, paddings
      by contents: a diagnostic. *)
end = struct
  type t = {
    mutable batch : Row.t;
    mutable input : Row.t;
    mutable output : Row.t;
    mutable batch_padding : padding;
    mutable input_padding : padding;
    mutable output_padding : padding;
    mutable padding_elem : float option option;
    mutable closed : bool; [@sexp_drop_if fun _ -> true] [@default false] [@equal.ignore]
    mutable discarded_by : string option;
        [@sexp_drop_if fun _ -> true] [@default None] [@equal.ignore]
    id : int;
    debug_name : string;
  }
  [@@deriving equal, sexp_of]

  type Row.error_trace += Shape_mismatch of t list

  let of_rows ~batch ~input ~output ~id ~debug_name =
    {
      batch;
      input;
      output;
      batch_padding = None;
      input_padding = None;
      output_padding = None;
      padding_elem = None;
      closed = false;
      discarded_by = None;
      id;
      debug_name;
    }

  let copy sh = { sh with batch = sh.batch }

  let closed_write_error (sh : t) ~field ~old_s ~new_s =
    raise
    @@ Row.Shape_error
         ( [%string
             "Shape of #%{sh.id#Int} %{sh.debug_name} is closed (committed by an earlier \
              finish_inference), but a later operation would change its %{field} from %{old_s} to \
              %{new_s}. A closed tensor's shape is committed: build such a consumer before closing \
              the tensor (before compiling it, Shape.to_dims or printing it), or consume a \
              materialized copy of it"],
           [ Shape_mismatch [ sh ] ] )

  let row_string r = Sexp.to_string_hum ([%sexp_of: Row.t] r)
  let padding_string p = Sexp.to_string_hum ([%sexp_of: padding] p)
  let padding_elem_string e = Sexp.to_string_hum ([%sexp_of: float option option] e)

  let set_rows (sh : t) ~batch ~input ~output =
    if sh.closed then
      List.iter
        [
          ("batch row", sh.batch, batch);
          ("input row", sh.input, input);
          ("output row", sh.output, output);
        ]
        ~f:(fun (field, old_r, new_r) ->
          if not (Row.equal_modulo_proj old_r new_r) then
            closed_write_error sh ~field ~old_s:(row_string old_r) ~new_s:(row_string new_r));
    sh.batch <- batch;
    sh.input <- input;
    sh.output <- output

  let set_paddings (sh : t) ~batch ~input ~output =
    if sh.closed then
      List.iter
        [
          ("batch padding", sh.batch_padding, batch);
          ("input padding", sh.input_padding, input);
          ("output padding", sh.output_padding, output);
        ]
        ~f:(fun (field, old_p, new_p) ->
          if not (equal_padding old_p new_p) then
            closed_write_error sh ~field ~old_s:(padding_string old_p) ~new_s:(padding_string new_p));
    sh.batch_padding <- batch;
    sh.input_padding <- input;
    sh.output_padding <- output

  let set_padding_elem (sh : t) v =
    if sh.closed && not ([%equal: float option option] sh.padding_elem v) then
      closed_write_error sh ~field:"padding neutral element"
        ~old_s:(padding_elem_string sh.padding_elem)
        ~new_s:(padding_elem_string v);
    sh.padding_elem <- v

  let close sh = sh.closed <- true
  let mark_discarded sh ~provenance = if not sh.closed then sh.discarded_by <- Some provenance

  type snapshot = {
    snap_shape : t;
    snap_batch : Row.t;
    snap_input : Row.t;
    snap_output : Row.t;
    snap_batch_padding : padding;
    snap_input_padding : padding;
    snap_output_padding : padding;
    snap_padding_elem : float option option;
    snap_closed : bool;
    snap_discarded_by : string option;
  }

  let copy_padding = Option.map ~f:Array.copy

  let snapshot
      ({
         batch;
         input;
         output;
         batch_padding;
         input_padding;
         output_padding;
         padding_elem;
         closed;
         discarded_by;
         id = _;
         debug_name = _;
       } as sh) =
    {
      snap_shape = sh;
      snap_batch = batch;
      snap_input = input;
      snap_output = output;
      snap_batch_padding = copy_padding batch_padding;
      snap_input_padding = copy_padding input_padding;
      snap_output_padding = copy_padding output_padding;
      snap_padding_elem = padding_elem;
      snap_closed = closed;
      snap_discarded_by = discarded_by;
    }

  let snapshot_shape s = s.snap_shape

  let restore
      {
        snap_shape = sh;
        snap_batch;
        snap_input;
        snap_output;
        snap_batch_padding;
        snap_input_padding;
        snap_output_padding;
        snap_padding_elem;
        snap_closed;
        snap_discarded_by;
      } =
    sh.batch <- snap_batch;
    sh.input <- snap_input;
    sh.output <- snap_output;
    sh.batch_padding <- copy_padding snap_batch_padding;
    sh.input_padding <- copy_padding snap_input_padding;
    sh.output_padding <- copy_padding snap_output_padding;
    sh.padding_elem <- snap_padding_elem;
    sh.closed <- snap_closed;
    sh.discarded_by <- snap_discarded_by

  let snapshot_differences
      {
        snap_shape = sh;
        snap_batch;
        snap_input;
        snap_output;
        snap_batch_padding;
        snap_input_padding;
        snap_output_padding;
        snap_padding_elem;
        snap_closed;
        snap_discarded_by;
      } =
    List.filter_map
      [
        ("batch row", phys_equal sh.batch snap_batch);
        ("input row", phys_equal sh.input snap_input);
        ("output row", phys_equal sh.output snap_output);
        ("batch padding", equal_padding sh.batch_padding snap_batch_padding);
        ("input padding", equal_padding sh.input_padding snap_input_padding);
        ("output padding", equal_padding sh.output_padding snap_output_padding);
        ("padding neutral element", phys_equal sh.padding_elem snap_padding_elem);
        ("closed", Bool.equal sh.closed snap_closed);
        ("discarded_by", phys_equal sh.discarded_by snap_discarded_by);
      ]
      ~f:(fun (field, same) ->
        if same then None else Some [%string "shape #%{sh.id#Int} %{sh.debug_name}: %{field}"])
end

include Shape_record

let row_of_kind kind (sh : t) =
  match kind with `Batch -> sh.batch | `Input -> sh.input | `Output -> sh.output

type deduce_within_shape = Not_constrained | Input_equals_output
[@@deriving compare, sexp, variants]

(* Capture references are caller-retained (gh-ocannl-1301): the ones open work bound or set since
   the last closure ([open_refs], by uid) are, after a rewind, stale handles like the work's tensors
   -- a reference carries what the discarded inference wrote into it (its binding, an inferred
   [solved_dim]), so reusing it is refused rather than letting it inject a discarded dimension. The
   tracking is keyed by uid so that it can live here, at the top level the mutation-serial scan
   reads, ahead of the sealed {!Delayed_ref} whose setters maintain it (gh-ocannl-1573). *)
let ref_uid = ref 0
[@@mutation_serial.exempt
  "a fresh-id counter: ids are never reused, so a rewind need not restore it"]

let open_refs : int list ref = ref []

(* The references touched while a shape-scheme trace runs (gh-ocannl-404), when one does: [None]
   otherwise. [open_refs] lists a reference once per inference, so it cannot tell a trace's touch of
   a reference touched before it. *)
let trace_touches : int list option ref = ref None
[@@mutation_serial.exempt "the shape-scheme trace's own record, saved and restored by the trace"]

let touch_ref uid =
  Row.note_mutation ();
  Option.iter !trace_touches ~f:(fun l -> trace_touches := Some (uid :: l));
  if not (List.mem !open_refs uid ~equal:Int.equal) then open_refs := uid :: !open_refs

(* Unbound references to which {!set_equal} handed a number across a row reference -- the row's
   product, given to or taken from it -- with that row reference's label (gh-ocannl-1655). Such a
   reference is a number to [set_equal], but bound to a row later it would leave the two rows tied
   by product only, where binding it first makes the same call a [Row_eq]: the binding is refused.
   The entry outlives closures (the tie is no better after one); the write touches the reference, so
   a rewind of the call discards the reference itself. *)
let row_numbered_refs : (int * string) list ref = ref []

let note_row_numbered uid ~partner =
  Row.note_mutation ();
  touch_ref uid;
  if not (List.Assoc.mem !row_numbered_refs uid ~equal:Int.equal) then
    row_numbered_refs := (uid, partner) :: !row_numbered_refs

(* The process's one [Ir.Indexing.shape_writer], claimed as [Shape] initializes (gh-ocannl-1624):
   solving an [Ir.Indexing.variable_ref] and committing a static symbol's role take it, so they are
   [Shape]'s alone -- [Delayed_ref]'s setters below and the commit of [finish_inference]. Not
   exported. *)
let shape_writer = Ir.Indexing.claim_shape_writer ()

(** A capture reference: the [Ir.Indexing.variable_ref] an operation's [~capture_dims] fills in, and
    the variable inference bound it to. Private (gh-ocannl-1573): read and matched freely, written
    only through the setters here, each of which refuses a reference a rewind discarded and records
    the write as a session mutation of open work ({!touch_ref}). Rebinding [var] is legitimate --
    every operation the reference is passed to binds it to its own variable -- whereas a solution is
    reached once ({!Ir.Indexing.set_solved_dim}, which only [shape_writer]'s holder can call).
    Projection derivation does not rebind it (gh-ocannl-1969): [var] is the latest binding a shape
    inference made, so after a closure its kind is the kind of the binding at the closure. *)
module Delayed_ref : sig
  type t = private {
    var_ref : Ir.Indexing.variable_ref;
    mutable var : [ `Row of Row.row_var | `Dim of Row.dim_var | `Not_set_yet ];
    mutable carried_row : Row.dim list option; [@sexp_drop_if Option.is_none]
        (** The axes of the row the reference was bound to when its inference closed, if it was
            bound to a row that resolved (gh-ocannl-1689): concrete [Dim]s and symbolic [Sym]s,
            outer to inner, without projection ids. Across the closure it stands in for that row,
            whose variable belongs to the reset environment. [None] when the binding at the closure
            was a dimension (a dimension and a row share only the product), or the row held a
            concatenated or affine axis over a symbolic extent. *)
    mutable solved_basis : string; [@sexp_drop_if String.equal Row.default_basis]
        (** The basis tag of the size [var_ref.solved_dim] holds (gh-ocannl-2033): the tag of the
            solution the reference's dimension variable reached when it was published, such as the
            broadcast top [1_(bcast_if_1)] a free dimension closes to, or a named tag an operand
            gave it. A dimension binding re-emits the size with it, so a capture after the closure
            holds the dimension itself, as [carried_row] does a row. [default] for a value [set_dim]
            gave and for a row's product. *)
    uid : int; [@equal.ignore] [@sexp_drop_if fun _ -> true]
        (** The reference's identity for the rewind's tracking; not part of its value. *)
    mutable discarded_by : string option; [@equal.ignore] [@sexp_drop_if fun _ -> true]
        (** The provenance of the rewind that discarded the reference (gh-ocannl-1789): the mark
            lives on the reference, so it lasts exactly as long as someone can present it. Not part
            of its value. *)
  }
  [@@deriving equal, sexp_of]

  val make : string -> t
  (** A fresh, unbound and unsolved reference. *)

  val check_live : t -> unit
  (** Raises if a rewind discarded the reference. *)

  val discard : uids:int list -> keep:Set.M(Int).t -> provenance:string -> unit
  (** Marks every live reference among [uids] and not in [keep] discarded, by [provenance], unless
      an earlier rewind did. *)

  val count_discarded : unit -> int
  (** The live references a rewind discarded: a diagnostic. *)

  val set_var : t -> [ `Row of Row.row_var | `Dim of Row.dim_var | `Not_set_yet ] -> unit

  val set_solved_dim : ?basis:string -> t -> int -> unit
  (** Solves the reference to a size, of the basis tag [basis] (by default [default]). *)

  val set_solved_sym : t -> Ir.Indexing.static_symbol -> unit

  val set_carried_row : t -> Row.dim list option -> unit
  (** Sets {!field-carried_row} at a closure, from the binding the reference has then; writing the
      value already held is not a mutation. *)

  type snapshot
  (** A reference's binding and solution at a checkpoint, bound to the reference (gh-ocannl-903). *)

  val snapshot_live : unit -> snapshot list
  (** A snapshot of every reference still alive: a caller may hold one that no step reaches -- made
      by [get_variable_ref], or solved by [set_dim] while unbound -- and the window may bind or
      solve it. *)

  val snapshot_uid : snapshot -> int

  val restore : snapshot -> unit
  (** Puts the binding and solution back, unsolving the reference if it was unsolved. *)

  val snapshot_changed : snapshot -> bool
  (** Whether the reference differs from the snapshot: a diagnostic. *)
end = struct
  type t = {
    var_ref : Ir.Indexing.variable_ref;
    mutable var : [ `Row of Row.row_var | `Dim of Row.dim_var | `Not_set_yet ];
    mutable carried_row : Row.dim list option; [@sexp_drop_if Option.is_none]
    mutable solved_basis : string; [@sexp_drop_if String.equal Row.default_basis]
    uid : int; [@equal.ignore] [@sexp_drop_if fun _ -> true]
    mutable discarded_by : string option; [@equal.ignore] [@sexp_drop_if fun _ -> true]
  }
  [@@deriving equal, sexp_of]

  (* Every reference made and still alive, weakly: a checkpoint snapshots them all, since a caller
     may hold one no step reaches, and a rewind marks the ones it discards. *)
  module Live = Stdlib.Weak.Make (struct
    type nonrec t = t

    let equal r1 r2 = r1.uid = r2.uid
    let hash r = r.uid
  end)

  let live = Live.create 64
  [@@mutation_serial.exempt
    "a weak registry of the capture references alive, which checkpoints read: not inference state"]

  let make ref_label =
    Int.incr ref_uid;
    let r =
      {
        var_ref = Ir.Indexing.variable_ref ref_label;
        var = `Not_set_yet;
        carried_row = None;
        solved_basis = Row.default_basis;
        uid = !ref_uid;
        discarded_by = None;
      }
    in
    Live.add live r;
    r

  let check_live r =
    Option.iter r.discarded_by ~f:(fun provenance ->
        let label = r.var_ref.ref_label in
        raise
        @@ Row.Shape_error
             ( [%string
                 "the variable reference %{label} was bound or set by work discarded by the rewind \
                  after %{provenance}; capture it afresh"],
               [] ))

  (* A reference nobody holds needs no mark: it is gone from [live], and no one can present it. *)
  let discard ~uids ~keep ~provenance =
    let uids = Set.diff (Set.of_list (module Int) uids) keep in
    if not (Set.is_empty uids) then
      Live.iter
        (fun r ->
          if Set.mem uids r.uid && Option.is_none r.discarded_by then
            r.discarded_by <- Some provenance)
        live

  let count_discarded () =
    Live.fold (fun r n -> if Option.is_some r.discarded_by then n + 1 else n) live 0

  let set_var r v =
    check_live r;
    touch_ref r.uid;
    r.var <- v

  (* A solution already held is no change, so it is not a mutation; a conflicting one raises before
     anything is written or counted. *)
  let set_solved_dim ?(basis = Row.default_basis) r d =
    check_live r;
    if not ([%equal: int option] r.var_ref.solved_dim (Some d)) then (
      Ir.Indexing.set_solved_dim shape_writer r.var_ref d;
      r.solved_basis <- basis;
      touch_ref r.uid)

  let set_solved_sym r sym =
    check_live r;
    if not ([%equal: Ir.Indexing.static_symbol option] r.var_ref.solved_sym (Some sym)) then (
      Ir.Indexing.set_solved_sym shape_writer r.var_ref sym;
      touch_ref r.uid)

  let set_carried_row r row =
    check_live r;
    if not ([%equal: Row.dim list option] r.carried_row row) then (
      touch_ref r.uid;
      r.carried_row <- row)

  type snapshot = {
    snap_ref : t;
    snap_var : [ `Row of Row.row_var | `Dim of Row.dim_var | `Not_set_yet ];
    snap_carried_row : Row.dim list option;
    snap_solved_basis : string;
    snap_var_ref : Ir.Indexing.variable_ref_snapshot;
  }

  let snapshot ({ var_ref; var; carried_row; solved_basis; uid = _; discarded_by = _ } as r) =
    {
      snap_ref = r;
      snap_var = var;
      snap_carried_row = carried_row;
      snap_solved_basis = solved_basis;
      snap_var_ref = Ir.Indexing.snapshot_variable_ref var_ref;
    }

  let snapshot_live () = Live.fold (fun r acc -> snapshot r :: acc) live []
  let snapshot_uid s = s.snap_ref.uid

  let restore { snap_ref; snap_var; snap_carried_row; snap_solved_basis; snap_var_ref } =
    snap_ref.var <- snap_var;
    snap_ref.carried_row <- snap_carried_row;
    snap_ref.solved_basis <- snap_solved_basis;
    Ir.Indexing.restore_variable_ref shape_writer snap_var_ref

  let snapshot_changed { snap_ref; snap_var; snap_carried_row; snap_solved_basis; snap_var_ref } =
    (not (phys_equal snap_ref.var snap_var))
    || (not (phys_equal snap_ref.carried_row snap_carried_row))
    || (not (String.equal snap_ref.solved_basis snap_solved_basis))
    || Ir.Indexing.variable_ref_snapshot_changed snap_var_ref
end

type delayed_var_ref = Delayed_ref.t = private {
  var_ref : Ir.Indexing.variable_ref;
  mutable var : [ `Row of Row.row_var | `Dim of Row.dim_var | `Not_set_yet ];
  mutable carried_row : Row.dim list option; [@sexp_drop_if Option.is_none]
  mutable solved_basis : string; [@sexp_drop_if String.equal Row.default_basis]
  uid : int; [@equal.ignore] [@sexp_drop_if fun _ -> true]
  mutable discarded_by : string option; [@equal.ignore] [@sexp_drop_if fun _ -> true]
}
[@@deriving equal, sexp_of]

let get_variable_ref = Delayed_ref.make
let check_ref_live = Delayed_ref.check_live

(* A trace refused by design (gh-ocannl-1751): what the function does is valid, but a scheme cannot
   carry it. Distinct from [Row.Shape_error], a contradiction, so that an automatic scheme can
   degrade to an unschemed function on this one and stay loud on the other. *)
exception Untraceable_scheme of string

type ref_binding = [ `Dim of Row.dim_var | `Row of Row.row_var ]

(* An external capture reference of a shape-scheme trace (gh-ocannl-1791): one made before the trace
   began and bound by it, with the variable of its first binding there -- the application's
   boundary, related to the reference's state at each application -- and of its last, which it is
   left bound to. *)
type ref_input = { input_ref : delayed_var_ref; first : ref_binding; mutable last : ref_binding }

(* The innermost running trace's reference frame: references with a uid up to [frame_refs_mark] were
   made before it, and [frame_inputs] (newest first) are the ones it bound. While a trace runs, such
   a reference reads as unbound and unsolved: its state outside belongs to the application, not to
   the scheme. *)
type ref_frame = { frame_refs_mark : int; mutable frame_inputs : ref_input list }

let trace_ref_frame : ref_frame option ref = ref None
[@@mutation_serial.exempt "the shape-scheme trace's own record, saved and restored by the trace"]

let trace_input_frame (r : delayed_var_ref) =
  match !trace_ref_frame with Some f when r.uid <= f.frame_refs_mark -> Some f | _ -> None

let is_trace_input r = Option.is_some (trace_input_frame r)

(* What a scheme input cannot carry: an operation that reads or writes the reference's solution, or
   relates it to another reference, rather than binding it to an operation's variable. Refused
   before anything is read, so a no-op arm (an already held solution) is refused too. *)
let refuse_external_in_trace ~operation (r : delayed_var_ref) =
  if is_trace_input r then
    let label = r.var_ref.ref_label in
    raise
    @@ Untraceable_scheme
         [%string
           "the function calls %{operation} on the capture reference %{label}, made outside it -- \
            a scheme carries such a reference only as an input its operations bind (through \
            ~capture_dims), not one it sets or relates; make the reference inside the function, or \
            apply the function inline"]

(* Whether a bound reference was bound before the last closure (or session reset): its uid is no
   longer in [open_refs]. [open_refs] also records solution writes, but in every reachable state it
   agrees with "bound in the open work": a closed reference holding no solution is written only
   through paths that call {!refuse_closed_unsolved_ref} before they touch it, and a closed one
   holding a solution (or, for a row use, a carried row) is touched again only by a binding, which
   checks before it binds. The closure's own writes ([carried_row] included) precede its emptying of
   [open_refs]. *)
let bound_before_closure (r : delayed_var_ref) =
  (match r.var with `Not_set_yet -> false | `Dim _ | `Row _ -> true)
  && not (List.mem !open_refs r.uid ~equal:Int.equal)

(* A reference bound before the last closure with no solution recorded has lost its constraint: its
   [var] names a variable of the reset environment, which nothing in the open work constrains
   (gh-ocannl-1655). Relating it is refused rather than letting the relation bind a dead variable. A
   row holding a symbolic extent has no numeric product, but its closure carries the row itself
   ([carried_row], gh-ocannl-1689): that is a value for [as_row] uses -- a row binding, a Row-Row
   [set_equal] -- and none for the uses needing a number. *)
let refuse_closed_unsolved_ref ?(as_row = false) ~operation (r : delayed_var_ref) =
  if
    bound_before_closure r
    && Option.is_none r.var_ref.solved_dim
    && Option.is_none r.var_ref.solved_sym
    && not (as_row && Option.is_some r.carried_row)
  then
    let label = r.var_ref.ref_label in
    let kind = match r.var with `Row _ -> "row" | `Dim _ | `Not_set_yet -> "dimension" in
    raise
    @@ Row.Shape_error
         ( [%string
             "%{operation}: the %{kind} variable reference %{label} was bound by shape inference \
              that has since closed (or by a session since reset) without a solved value -- a row \
              holding a symbolic extent keeps its axes but has no numeric product -- so the \
              closure kept nothing this use can relate, and relating it now would constrain a dead \
              variable. Capture it afresh in an operation of the open work."],
           [] )

type compose_type =
  | Pointwise_bin
  | Compose
  | Einsum of string * delayed_var_ref list
  | Defined_by_cd_logic
[@@deriving sexp_of, equal]

type transpose_type =
  | Transpose
  | Pointwise_un
  | Permute of string * delayed_var_ref list
  | Batch_slice of Idx.static_symbol
  | Uint4x32_to_prec of Ir.Ops.prec Lazy.t
  | Defined_by_cd_logic
[@@deriving equal, sexp_of]

type terminal_type = Data of Ir.Assignments.init_data | Fetch of Ir.Assignments.fetch_op
[@@deriving equal, sexp_of]

type ternary_type =
  | Pointwise_tern
  | Compose_accumulate
  | Defined_by_cd_logic
  | Einsum_tern of string * delayed_var_ref list
[@@deriving sexp_of, equal]

let einsum_of_spec spec =
  try Einsum_parser.einsum_of_spec spec
  with Einsum_parser.Parse_error msg ->
    raise @@ Utils.User_error ("Shape.einsum_of_spec: while parsing: " ^ spec ^ " error: " ^ msg)

type logic =
  | Broadcast of compose_type * t * t
  | Transpose of transpose_type * t
  | Broadcast_tern of ternary_type * t * t * t
  | Terminal of { is_param : bool; logic : terminal_type }
  | Block of { spec : string; delayed_vars : delayed_var_ref list; rhses : t list }
[@@deriving equal, sexp_of]

let logic_to_spec = function
  | Broadcast (Pointwise_bin, _, _)
  | Transpose (Pointwise_un, _)
  | Broadcast_tern (Pointwise_tern, _, _, _) ->
      "."
  | Broadcast (Compose, _, _) | Broadcast_tern (Compose_accumulate, _, _, _) -> "@"
  | Broadcast (Einsum (spec, _), _, _)
  | Transpose (Permute (spec, _), _)
  | Block { spec; _ }
  | Broadcast_tern (Einsum_tern (spec, _), _, _, _) ->
      spec
  | Transpose (Transpose, _) -> "T"
  | Transpose (Batch_slice _, _) -> "@|"
  | Transpose (Uint4x32_to_prec _, _) -> "U4x32"
  | Broadcast (Defined_by_cd_logic, _, _)
  | Transpose (Defined_by_cd_logic, _)
  | Broadcast_tern (Defined_by_cd_logic, _, _, _) ->
      "<cd_logic>"
  | Terminal _ -> "<terminal>"

module Update_id = struct
  module T = struct
    type t = Update_id of int [@@deriving equal, compare, hash, sexp]
  end

  include T
  include Comparator.Make (T)
end

type update_id = Update_id.t [@@deriving equal, compare, hash, sexp]

(* A negative control's switch (gh-ocannl-1890): while set, every packed conversion reuses the first
   coefficient registered -- one coefficient memoized across call sites, and for a scheme's
   applications owned by its trace. *)
let memoized_draw_coefficient : int Utils.safe_lazy option ref option ref = ref None
[@@mutation_serial.exempt "a negative control's switch, set by tests only"]

let with_memoized_draw_coefficient f =
  memoized_draw_coefficient := Some (ref None);
  Exn.protect ~f ~finally:(fun () -> memoized_draw_coefficient := None)

let update_uid = ref 0
[@@mutation_serial.exempt
  "a fresh-id counter: ids are never reused, so a rewind need not restore it"]

let get_update_id () =
  Int.incr update_uid;
  Update_id.Update_id !update_uid

(** Data required for a shape inference update step. Ideally, an update should be performed at least
    twice, the second time after all the other relevant updates have been performed for the first
    time. In OCANNL, this is achieved by performing updates both as the tensors are constructed, and
    via lazy callbacks as the corresponding [Ir.Indexing] dimensions and projections are first
    accessed.

    Private (gh-ocannl-1573): read and matched freely; built only by [make] and [with_logic], and
    its projections published only by [set_projections], which refuses to replace a result. *)
module Update_step_record : sig
  type update_step = private {
    shape : t;
    logic : logic;
    id : update_id;
    mutable unsafe_projections : Idx.projections option;
    neutral_elem : float option;
        (** The neutral element of the operation's accumulation, fixed at creation: [Some v] when
            all assignment ops in the update step use the same neutral element [v], [None] when
            different operations have different neutral elements or when there are no accumulator
            operations. [derive_projections] reads it for the clamped-window decision (gh-504) and
            to commit the margins' neutral ([padding_elem]), so a step is only created once its
            operation's assignments are known -- see [Tensor.op]. *)
    mutable discarded_by : string option; [@sexp_drop_if fun _ -> true]
        (** The provenance of the rewind that discarded the step (gh-ocannl-1789): the mark lives on
            the step, so it lasts exactly as long as someone can present it. *)
  }
  [@@deriving sexp_of]

  val make : shape:t -> logic:logic -> neutral_elem:float option -> update_step
  (** A fresh step, with a new id and no projections yet. *)

  val discard_since : update_mark:int -> provenance:string -> unit
  (** Marks every live step made since the last closure with an id above [update_mark] discarded, by
      [provenance], unless an earlier rewind did. *)

  val forget_made : unit -> unit
  (** Called at a closure (and a session reset): no rewind can discard a step made before it. *)

  val count_discarded : unit -> int
  (** The live steps made since the last closure that a rewind discarded: a diagnostic. *)

  val with_logic : update_step -> logic -> update_step
  (** The same step (same id, same projection state) over another logic: [dealias_operands]'s
      temporary derivation copy, never registered. *)

  val set_projections : update_step -> Idx.projections -> unit
  (** Publishes the step's projections; raises if they were already computed. *)

  type snapshot
  (** The step's mutable fields at a checkpoint, bound to the step (gh-ocannl-903). *)

  val snapshot : update_step -> snapshot
  val restore : snapshot -> unit

  val snapshot_changed : snapshot -> bool
  (** Whether the step differs from the snapshot: a diagnostic. *)

  val snapshot_is : update_step -> snapshot -> bool
  (** Whether the snapshot is of this step. *)
end = struct
  type update_step = {
    shape : t;
    logic : logic;
    id : update_id;
    mutable unsafe_projections : Idx.projections option;
    neutral_elem : float option;
    mutable discarded_by : string option; [@sexp_drop_if fun _ -> true]
  }
  [@@deriving sexp_of]

  (* The steps made since the last closure and still alive, weakly: the steps a rewind can discard
     -- a rewind cannot cross a closure -- registered or not (a construction that failed before
     registering its step, or [make] called directly, still made one a caller may hold). *)
  module Made = Stdlib.Weak.Make (struct
    type t = update_step

    let equal = phys_equal
    let hash step = Update_id.hash step.id
  end)

  let made = Made.create 64
  [@@mutation_serial.exempt
    "a weak registry of the steps made since the last closure, which rewinds read: not inference \
     state"]

  let make ~shape ~logic ~neutral_elem =
    let step =
      {
        shape;
        logic;
        id = get_update_id ();
        unsafe_projections = None;
        neutral_elem;
        discarded_by = None;
      }
    in
    Made.add made step;
    step

  let discard_since ~update_mark ~provenance =
    Made.iter
      (fun step ->
        let (Update_id.Update_id n) = step.id in
        if n > update_mark && Option.is_none step.discarded_by then
          step.discarded_by <- Some provenance)
      made

  let forget_made () = Made.clear made

  let count_discarded () =
    Made.fold (fun step n -> if Option.is_some step.discarded_by then n + 1 else n) made 0

  let with_logic step logic = { step with logic }

  let set_projections step projections =
    if Option.is_some step.unsafe_projections then
      raise
      @@ Utils.User_error "derive_projections: projections already computed for this update_step";
    step.unsafe_projections <- Some projections

  type snapshot = { snap_step : update_step; snap_projections : Idx.projections option }

  let snapshot
      ({ shape = _; logic = _; id = _; unsafe_projections; neutral_elem = _; discarded_by = _ } as
       step) =
    { snap_step = step; snap_projections = unsafe_projections }

  let restore { snap_step; snap_projections } = snap_step.unsafe_projections <- snap_projections

  let snapshot_changed { snap_step; snap_projections } =
    not (phys_equal snap_step.unsafe_projections snap_projections)

  let snapshot_is step { snap_step; snap_projections = _ } = phys_equal step snap_step
end

type update_step = Update_step_record.update_step = private {
  shape : t;
  logic : logic;
  id : update_id;
  mutable unsafe_projections : Idx.projections option;
  neutral_elem : float option;
  mutable discarded_by : string option; [@sexp_drop_if fun _ -> true]
}
[@@deriving sexp_of]

let make_update_step = Update_step_record.make

let with_error_trace = ref true
[@@mutation_serial.kept "user configuration of error messages, not inference state"]

(** Converts an axes-keyed map into three arrays of values: batch axes, input axes, output axes. If
    the map is incomplete, the result will likely be invalid: gaps in the array are filled with an
    arbitrary one of the provided values. *)
let axis_map_to_dims_bio (type a) ?(default : a option) (idcs : a axis_map) =
  if Map.is_empty idcs then (([||], [||], [||]), ([||], [||], [||]))
  else
    let witness =
      match default with Some witness -> witness | None -> snd @@ Map.min_elt_exn idcs
    in
    let bch_axes, other =
      Map.partition_mapi idcs ~f:(fun ~key:{ in_axes; _ } ~data ->
          if Row.is_batch in_axes then Either.First data else Either.Second data)
    in
    let inp_axes, out_axes =
      Map.partition_mapi other ~f:(fun ~key:{ in_axes; _ } ~data ->
          if Row.is_input in_axes then Either.First data else Either.Second data)
    in
    let make_row axes =
      let back_axes, front_axes =
        Map.to_alist axes
        |> List.partition_map ~f:(fun ({ AxisKey.from_end; pos = i; _ }, v) ->
            if from_end then Either.First (i, v) else Second (i, v))
      in
      let back_size = List.fold back_axes ~init:0 ~f:(fun accu (i, _) -> max i accu) in
      let front_size = List.fold front_axes ~init:0 ~f:(fun accu (i, _) -> max i accu) in
      let back = Array.create ~len:back_size witness in
      let front = Array.create ~len:front_size witness in
      List.iter back_axes ~f:(fun (i, v) -> back.(back_size - i) <- v);
      List.iter front_axes ~f:(fun (i, v) -> front.(i - 1) <- v);
      (back, front)
    in
    let bch, beg_bch = make_row bch_axes in
    let inp, beg_inp = make_row inp_axes in
    let out, beg_out = make_row out_axes in
    ((bch, inp, out), (beg_bch, beg_inp, beg_out))

(** Converts an axes-keyed map into an array of values using the [force_to_dims] semantics of axes.
    If the map is incomplete and the [~default] is not given, the result might be invalid: gaps in
    the array are filled with an arbitrary one of the provided values. *)
let axis_map_to_dims_index (type a) ?(default : a option) (idcs : a axis_map) : a array =
  let (bch, inp, out), (beg_bch, beg_inp, beg_out) = axis_map_to_dims_bio ?default idcs in
  Array.concat [ beg_bch; bch; beg_out; out; beg_inp; inp ]

let axes_spec_to_dims_bio ~sh_id ~row_var_env ~dim_var_env:_ ~f labels =
  let (b_dims, i_dims, o_dims), (beg_b_dims, beg_i_dims, beg_o_dims) =
    axis_map_to_dims_bio labels.labels
  in
  let to_dim kind = Array.(Fn.compose to_list @@ map ~f:(f kind)) in
  let to_bcast kind ~implicit v beg_dims =
    let beg_dims = to_dim kind beg_dims in
    Option.value_map v ~default:(Row.Broadcastable, beg_dims) ~f:(fun vname ->
        let v = Hashtbl.find_or_add row_var_env vname ~default:(fun () -> Row.get_row_var ()) in
        Row.add_used_in_spec_or_compose v;
        (* A row omitted together with its kind separator reads as the context ellipsis, but unlike
           an explicit ellipsis it may be silently closed to an empty row (even on parameters) when
           nothing else constrains it. *)
        if implicit then Row.add_safe_to_guess v;
        (Row.Row_var v, beg_dims))
  in
  let to_row kind ~implicit v dims beg_dims =
    let bcast, beg_dims = to_bcast kind ~implicit v beg_dims in
    { Row.beg_dims; dims = to_dim kind dims; bcast; prov = Row.provenance ~sh_id ~kind }
  in
  let batch = to_row `Batch ~implicit:labels.implicit_batch labels.bcast_batch b_dims beg_b_dims in
  let input = to_row `Input ~implicit:labels.implicit_input labels.bcast_input i_dims beg_i_dims in
  let output = to_row `Output ~implicit:false labels.bcast_output o_dims beg_o_dims in
  (batch, input, output)

let concat_component_text = function Concat_label l -> l | Concat_size n -> Int.to_string n

(** The dimension of one component of a concatenated axis: a label's variable (shared with the
    spec's other slots through [dim_var_env]), or a fixed size no other slot names. A fixed size
    carries no projection id: projection derivation gives it a fresh one, so the component is a
    segment with its own iterator that no other access mentions -- skipped by an extraction, left to
    the initialization by a replacement (gh-ocannl-1656). *)
let concat_component_dim ~spec ~dim_var_env = function
  | Concat_label label ->
      let v =
        Hashtbl.find_or_add dim_var_env label ~default:(fun () -> Row.get_var ~name:label ())
      in
      Row.add_concat_component v;
      Row.Var v
  | Concat_size n when n <= 0 ->
      raise
      @@ Row.Shape_error
           ( [%string
               "einsum spec %{spec}: a concatenation component of size %{n#Int} is empty; drop it \
                from the concatenation"],
             [] )
  | Concat_size n -> Row.get_default_dim ~d:n ()

(** Renders an affine axis spec as the einsum grammar reads it back ([[s*]o[=|<][+off]+[d*]k], or
    [s*o[+off]] without a window), for refusal messages. *)
let render_affine_axis = function
  | Affine_spec { stride; over_label; conv; stride_offset } -> (
      let coeff c = if String.equal c "1" then "" else c ^ "*" in
      let offset = if stride_offset = 0 then "" else "+" ^ Int.to_string stride_offset in
      match conv with
      | None -> [%string "%{stride}*%{over_label}%{offset}"]
      | Some { dilation; kernel_label; use_padding } ->
          let marker = match use_padding with `True -> "=" | `False -> "<" | `Unspecified -> "" in
          [%string
            "%{coeff stride}%{over_label}%{marker}%{offset}+%{coeff dilation}%{kernel_label}"])
  | Label s -> s
  | Fixed_index i -> Int.to_string i
  | Concat_spec components -> String.concat ~sep:"^" (List.map components ~f:concat_component_text)

(** Refuses an affine axis whose constant offset reaches past the axis's extent (gh-ocannl-1618), in
    any slot of an einsum spec. Shape inference sizes an affine axis without its offset
    ([stride * n] without a window, [stride * (n - 1) +] the dilated kernel extent for an unpadded
    window), so the offset is in bounds only below the stride and without an unpadded window. Before
    this, such a spec ran: on the result side the out-of-extent cells were silently dropped, on an
    operand they were read past the extent, and a stride-1 offset died with an internal projection
    error. A padded ([=+]) operand window keeps its offset: the padding absorbs it. *)
let refuse_out_of_extent_offset ~spec ~stride ~stride_offset ~(conv : bool option) ~axis =
  let refuse reason =
    raise
    @@ Row.Shape_error
         ( [%string
             "affine axis `%{Lazy.force axis}` in einsum spec \"%{spec}\" has offset \
              %{stride_offset#Int}, which %{reason}"],
           [] )
  in
  match conv with
  | None when stride_offset >= stride ->
      refuse
        [%string
          "is not smaller than its stride %{stride#Int}: the axis's extent is stride * n for an \
           extent n of the strided label, so the last positions would fall outside it. Write an \
           offset below the stride; to shift by whole strides, slice or pad the tensor separately."]
  | Some false when stride_offset > 0 ->
      refuse
        "an unpadded (<+) window cannot take: its extent, stride * (n - 1) + the dilated kernel \
         extent, has no room for an offset, so the last kernel positions would fall outside it. \
         Drop the offset, and slice or pad the tensor separately to shift it."
  | None | Some _ -> ()

(** Refuses a stride or a dilation below 1 (gh-ocannl-1657), invariants [Row.Affine] documents.
    Before this, a zero dilation ran ([o=+0*k] multiplied one cell by every kernel tap) or died in
    the dimension solver ([o<+0*k], an [Invalid_argument] dividing by zero). Integer literals in the
    grammar are nonnegative, but a stride or a dilation spelled by an identifier under [%op] and
    [%cd] takes its runtime value. *)
let refuse_nonpositive_coefficient ~spec ~axis ~what value =
  if value < 1 then
    raise
    @@ Row.Shape_error
         ( [%string
             "affine axis `%{Lazy.force axis}` in einsum spec \"%{spec}\" has %{what} \
              %{value#Int}, which must be at least 1: at 0 every position of its label would map \
              to the same cell"],
           [] )

(** The dimension of an affine axis spec. Refuses the coefficients and the offsets the dimension
    formulas cannot size ([refuse_nonpositive_coefficient], [refuse_out_of_extent_offset]); every
    route converting a parsed spec to dimensions goes through here. *)
let affine_dim_of_spec ~spec ~dim_var_env axis_spec =
  match axis_spec with
  | Affine_spec { stride; over_label; conv; stride_offset } ->
      let axis = lazy (render_affine_axis axis_spec) in
      let stride_int =
        try Int.of_string stride
        with _ -> failwith ("Invalid stride value (expected integer): " ^ stride)
      in
      refuse_nonpositive_coefficient ~spec ~axis ~what:"stride" stride_int;
      let over_dim =
        Row.Var
          (Hashtbl.find_or_add dim_var_env over_label ~default:(fun () ->
               Row.get_var ~name:over_label ()))
      in
      let conv =
        Option.map conv ~f:(fun { dilation; kernel_label; use_padding } ->
            let dilation_int =
              try Int.of_string dilation
              with _ -> failwith ("Invalid dilation value (expected integer): " ^ dilation)
            in
            refuse_nonpositive_coefficient ~spec ~axis ~what:"dilation" dilation_int;
            let use_padding_bool =
              match use_padding with
              | `True -> true
              | `False -> false
              | `Unspecified ->
                  failwith
                    "use_padding must be specified in convolution spec (use = for true, < for \
                     false)"
            in
            let kernel =
              Row.Var
                (Hashtbl.find_or_add dim_var_env kernel_label ~default:(fun () ->
                     Row.get_var ~name:kernel_label ()))
            in
            { Row.dilation = dilation_int; kernel; use_padding = use_padding_bool })
      in
      refuse_out_of_extent_offset ~spec ~stride:stride_int ~stride_offset
        ~conv:(Option.map conv ~f:(fun c -> c.use_padding))
        ~axis;
      Row.Affine { stride = stride_int; over = over_dim; conv; stride_offset }
  | Label _ | Fixed_index _ | Concat_spec _ -> invalid_arg "affine_dim_of_spec: not an affine axis"

let einsum_slot_spec_to_dims_bio ?(fixed_index = `Pin_var) ~original_spec ~sh_id ~row_var_env
    ~dim_var_env labels =
  let proj_env_update = ref @@ Row.dim_map_empty in
  let extras = ref [] in
  let f kind = function
    | Label name ->
        Row.Var (Hashtbl.find_or_add dim_var_env name ~default:(fun () -> Row.get_var ~name ()))
    | Fixed_index _ when Poly.equal fixed_index `Unit_dim ->
        (* Product-space proxies (gh-512): a fixed-index axis has a pinned projection, so it is
           never a product axis regardless of its extent — the proxy pins it to dimension 1. *)
        Row.get_default_dim ~d:1 ()
    | Fixed_index i ->
        let var = Row.get_var () in
        let d = Row.Var var in
        proj_env_update := Map.add_exn !proj_env_update ~key:var ~data:(Idx.Fixed_idx i);
        extras :=
          Row.Dim_constr
            {
              d;
              constr = At_least_dim (i + 1);
              origin =
                [
                  {
                    lhs_name = original_spec;
                    lhs_kind = kind;
                    rhs_name = "einsum_slot_spec_to_dims_bio";
                    rhs_kind = kind;
                    operation = Some "At_least_dim";
                  };
                ];
            }
          :: !extras;
        d
    | Affine_spec _ as axis_spec -> affine_dim_of_spec ~spec:original_spec ~dim_var_env axis_spec
    | Concat_spec components ->
        Row.Concat (List.map components ~f:(concat_component_dim ~spec:original_spec ~dim_var_env))
  in
  let result = axes_spec_to_dims_bio ~sh_id ~row_var_env ~dim_var_env ~f labels in
  (!extras, !proj_env_update, result)

type proj_axis_env = Idx.axis_index Row.dim_map [@@deriving sexp]

let check_dim_row_var_name_clash ~spec ~error_trace dim_var_env row_var_env =
  Hashtbl.iter_keys dim_var_env ~f:(fun name ->
      if Hashtbl.mem row_var_env name then
        raise
        @@ Row.Shape_error
             ( "Name '" ^ name
               ^ "' is used as both a dimension variable and a row variable in spec: " ^ spec,
               error_trace ))

let add_var_used_in_spec_or_compose row =
  match row with Row.Row_var v -> Row.add_used_in_spec_or_compose v | _ -> ()

let add_var_used_in_pointwise row =
  match row with Row.Row_var v -> Row.add_used_in_pointwise v | _ -> ()

(** Refuses constraints a fully staged solve left unsolved (gh-ocannl-1656). The one known source is
    a concatenated axis whose components the spec does not size: [p^o^q => o] fixes [p + q], not [p]
    and [q], so name it with the fix; anything else keeps the plain report of the leftover. *)
let refuse_unsolved ~where (unsolved : Row.constraint_ list) =
  if not (List.is_empty unsolved) then
    let concat =
      List.find_map unsolved ~f:(function
        | Row.Dim_eq { d1 = Row.Concat _ as c; _ } | Row.Dim_eq { d2 = Row.Concat _ as c; _ } ->
            Some c
        | _ -> None)
    in
    match concat with
    | Some c ->
        let concat_text = Row.dim_to_string Axis_size c in
        raise
        @@ Row.Shape_error
             ( [%string
                 "%{where}: the components of the concatenated axis %{concat_text} are not \
                  determined -- only their sum is. Size all but one of them: capture the component \
                  (the spec's trailing label list) and Shape.set_dim it, or write its size as an \
                  integer component (e.g. 2^o^1 => o)"],
               [ Row.Dim_mismatch [ c ] ] )
    | None ->
        raise
        @@ Row.Shape_error
             ( where ^ ": constraints left unsolved: "
               ^ Sexp.to_string_hum ([%sexp_of: Row.constraint_ list] unsolved),
               [] )

(** The origin operation of the size-only constraints [bind_delayed_vars_to_envs] emits for
    projection derivation; see there. *)
let projection_sizing_operation = "set_dim (projection-local sizing)"

(* Read off origins, which [Row.merge_origins] caps ([max_shape_error_origins]) -- sound only on
   constraints no solve has touched: [derive_projections] applies it to [get_inequalities]' own
   list, where the tagged constraint still carries the one origin [binding_constraints] built. A
   caller filtering a solver's output needs a marker of its own instead (gh-ocannl-2000). *)
let is_projection_sizing (c : Row.constraint_) =
  match c with
  | Dim_eq { origin; _ } ->
      List.exists origin ~f:(fun (o : Row.constraint_origin) ->
          Option.equal String.equal o.operation (Some projection_sizing_operation))
  | _ -> false

(* The refusals ahead of binding [delayed_ref] (gh-ocannl-1655): binding touches the reference,
   which would make one bound before the closure read as open work. *)
let check_before_binding ~where ~error_shapes ~is_row (delayed_ref : delayed_var_ref) =
  check_ref_live delayed_ref;
  let label = delayed_ref.var_ref.ref_label in
  refuse_closed_unsolved_ref ~as_row:is_row ~operation:[%string "capturing it %{where}"] delayed_ref;
  if is_row then
    Option.iter (List.Assoc.find !row_numbered_refs delayed_ref.uid ~equal:Int.equal)
      ~f:(fun partner ->
        raise
        @@ Row.Shape_error
             ( [%string
                 "Variable reference %{label} is captured as a row variable %{where}, but \
                  Shape.set_equal related it to the row reference %{partner} while it was unbound, \
                  which equated only a number (the row's product): bound to a row now, it would \
                  leave the two rows tied by product alone, not as rows. Pass %{label} to an \
                  operation's ~capture_dims before calling set_equal on it."],
               [ Shape_mismatch error_shapes ] ))

(* The binding a new binding of [r] ties to, and the row the closure carried for it
   (gh-ocannl-1689), read before the new binding touches [r] into the open work. A binding from
   before the closure names a dead variable, so it ties to nothing: its row, if it was one, travels
   in [carried_row], standing in for it. *)
let previous_of (r : delayed_var_ref) =
  if bound_before_closure r then (`Not_set_yet, r.carried_row) else (r.var, None)

(* A row's product equals a dimension: a dimension/row tie between capture references. Its
   coefficient is a constant, which a shape-scheme trace can carry (gh-ocannl-1890). *)
let product_tie ~origin row_var dim_var =
  Row.Rows_constr
    {
      r = [ Row.get_row_for_var Row.empty_provenance row_var ];
      constr =
        Total_elems
          {
            numerator =
              Strided_var
                {
                  coeff = Utils.safe_value "capture_product_tie" 1;
                  var = dim_var;
                  denom = 1;
                  round_up = false;
                };
            divided_by = [];
            keep_axis = false;
          };
      origin;
    }

(* The constraints binding [delayed_ref] to [binding] implies, given [previous_binding] and the
   reference's solution: equality with a value already solved for the reference, and a tie to the
   variable another operation bound the reference to (gh-ocannl-1545). [carried_row]: the row the
   last closure carried in place of a dead previous row binding (see {!previous_of}). [as_input]:
   the reference is an input of the running shape-scheme trace (gh-ocannl-1791), whose solution
   outside belongs to the application -- it reads as unsolved, in every branch. *)
let binding_constraints ~for_projections ~as_input ~error_shapes ~previous_binding ~carried_row
    (delayed_ref : delayed_var_ref) (binding : ref_binding) =
  let label = delayed_ref.var_ref.ref_label in
  let solved_dim, solved_sym, carried_row =
    if as_input then (None, None, None)
    else (delayed_ref.var_ref.solved_dim, delayed_ref.var_ref.solved_sym, carried_row)
  in
  let origin operation =
    [
      {
        Row.lhs_name = label;
        lhs_kind = `Output;
        rhs_name = delayed_ref.var_ref.ref_label;
        rhs_kind = `Output;
        operation = Some operation;
      };
    ]
  in
  let shared_origin = origin "shared capture reference" in
  let row_of var = Row.get_row_for_var Row.empty_provenance var in
  let product_is row_var dim_var = product_tie ~origin:shared_origin row_var dim_var in
  let unsolved = Option.is_none solved_dim && Option.is_none solved_sym in
  match binding with
  | `Dim var ->
      let solved_origin =
        origin (if for_projections then projection_sizing_operation else "set_dim")
      in
      let solved =
        match (solved_sym, solved_dim) with
        | Some sym, _ ->
            [ Row.Dim_eq { d1 = Row.Var var; d2 = Row.get_sym_dim sym; origin = solved_origin } ]
        | None, Some solved_dim ->
            [
              Row.Dim_eq
                {
                  d1 = Row.Var var;
                  d2 = Row.get_dim ~d:solved_dim ~basis:delayed_ref.solved_basis ();
                  origin = solved_origin;
                };
            ]
        | None, None -> []
      in
      let tie =
        match previous_binding with
        | `Dim v0 when not (Row.equal_dim_var v0 var) ->
            [ Row.Dim_eq { d1 = Row.Var var; d2 = Row.Var v0; origin = shared_origin } ]
        | `Row _ when Option.is_some solved_sym ->
            (* Unreachable (gh-ocannl-1655): [set_sym_dim] refuses a row reference, a row's numeric
               publication is a product only, and the row arm below refuses a symbolic reference
               before binding it -- and since gh-ocannl-1624 nothing but [Shape] can solve a
               reference. Kept as the row arm's refusal, not a silent drop of the row tie. *)
            raise
            @@ Row.Shape_error
                 ( "Variable " ^ label
                   ^ " was bound to a row variable but is solved to a symbolic dimension (symbolic \
                      dimensions take no part in row products)",
                   [ Shape_mismatch error_shapes ] )
        | `Row r0 when unsolved -> [ product_is r0 var ]
        | `Dim _ | `Row _ | `Not_set_yet -> []
      in
      solved @ tie
  | `Row var ->
      if Option.is_some solved_sym then
        raise
        @@ Row.Shape_error
             ( "Variable " ^ label
               ^ " is bound to a row variable but is solved to a symbolic dimension (symbolic \
                  dimensions take no part in row products)",
               [ Shape_mismatch error_shapes ] );
      let solved =
        match (carried_row, solved_dim) with
        | _ when for_projections -> []
        | Some dims, _ ->
            (* The row itself, not its product: [[2; 3]] and [[3; 2]] are different rows. It also
               carries a row holding a symbolic extent, which has no product. *)
            [
              Row.Row_eq
                {
                  r1 = row_of var;
                  r2 = { beg_dims = []; dims; bcast = Broadcastable; prov = Row.empty_provenance };
                  origin = origin "row carried across a closure";
                };
            ]
        | None, Some solved_dim ->
            [
              Row.Rows_constr
                {
                  r = [ row_of var ];
                  constr =
                    Row.Total_elems
                      { numerator = Num_elems solved_dim; divided_by = []; keep_axis = false };
                  origin = origin "set_dim";
                };
            ]
        | None, None -> []
      in
      let tie =
        match previous_binding with
        | `Row r0 when not (Row.equal_row_var r0 var) ->
            [ Row.Row_eq { r1 = row_of var; r2 = row_of r0; origin = shared_origin } ]
        | `Dim v0 when unsolved -> [ product_is var v0 ]
        | `Dim _ | `Row _ | `Not_set_yet -> []
      in
      solved @ tie

(* The binding of [delayed_ref] an operation (or an explicit scheme instance) makes: its
   constraints, then the binding itself. A previous binding from before the closure names a dead
   variable: the value it reached travels in [solved_dim]/[solved_sym], a row's also in
   [carried_row] (the pre-pass refused a reference holding none), and a tie to the dead variable
   would constrain nothing ({!previous_of}). Inside a shape-scheme trace, a reference made before it
   is the scheme's input (gh-ocannl-1791): its first binding there ties to nothing outside, and the
   trace's frame records it. Projection derivation reads [binding] and leaves [var] alone
   (gh-ocannl-1969): its variables are local to one step, and [var] has to go on saying what the
   closure saw. *)
let bind_ref ~for_projections ~error_shapes (delayed_ref : delayed_var_ref) (binding : ref_binding)
    =
  check_ref_live delayed_ref;
  let frame = if for_projections then None else trace_input_frame delayed_ref in
  let recorded =
    Option.bind frame ~f:(fun f ->
        List.find f.frame_inputs ~f:(fun i -> phys_equal i.input_ref delayed_ref))
  in
  let previous_binding, carried_row =
    match (frame, recorded) with
    | Some _, None -> (`Not_set_yet, None)
    | _ when for_projections -> (`Not_set_yet, None)
    | _ -> previous_of delayed_ref
  in
  if not for_projections then
    Delayed_ref.set_var delayed_ref
      (binding :> [ `Row of Row.row_var | `Dim of Row.dim_var | `Not_set_yet ]);
  let constraints =
    binding_constraints ~for_projections ~as_input:(Option.is_some frame) ~error_shapes
      ~previous_binding ~carried_row delayed_ref binding
  in
  (match (frame, recorded) with
  | Some f, None ->
      f.frame_inputs <-
        { input_ref = delayed_ref; first = binding; last = binding } :: f.frame_inputs
  | _, Some input -> input.last <- binding
  | None, None -> ());
  constraints

(** The hint for a capture label a single-char spec split (gh-ocannl-2092): there every character is
    an axis of its own -- a label, or a digit's fixed index -- so a multi-letter label is never one
    of the spec's labels. The test is on the spec's labels, not on its mode: the ppx renders every
    literal spec in comma-separated form, so the spec reaching here reads as multichar whatever the
    user wrote. A label with a character the spec does not hold gets no hint. *)
let split_capture_label_hint ~spec ~dim_var_env label =
  let is_axis c =
    if Char.is_digit c then String.mem spec c else Hashtbl.mem dim_var_env (String.of_char c)
  in
  if
    String.length label > 1
    && String.for_all label ~f:is_axis
    && String.exists label ~f:(fun c -> not (Char.is_digit c))
  then
    "; hint: the spec has an axis for each character of `" ^ label ^ "`"
    ^ Einsum_parser.multichar_note ~what:"capture label" ~written:(label ^ ",")
  else ""

(** Shared helper: bind each [delayed_var_ref] to the dim/row variable resolved for its label in
    [dim_var_env]/[row_var_env]. Returns the constraints the binding implies
    ({!binding_constraints}). Used by the unary-permute, binary-einsum, block, and ternary-einsum
    branches of [get_inequalities].

    A reference passed to several operations names one dimension. Within one inference, two dim
    bindings are equated and two row bindings are [Row_eq]-tied, whether or not the reference is
    already solved, so the outcome does not depend on the order inference runs in. A dim and a row
    share only a number, so a mixed pair is tied by the row's product (as {!set_equal} does; with a
    symbolic extent in the row this is refused, symbolic dimensions taking no part in total-elements
    arithmetic). Across a closure the reference carries its value in [solved_dim]/[solved_sym], a
    row's as its product and as its axes ([carried_row], gh-ocannl-1689): [var] then names a
    variable of the reset environment, free in the open work, so the tie to it constrains nothing,
    and a row binding is [Row_eq]-tied to the carried row instead.

    Projection derivation emits no tie and no row constraint, but it does emit a dim binding's
    solved value, tagged {!projection_sizing_operation}: the closed shapes carry every axis's extent
    except the split of a concatenated axis into components, so a middle extraction ([p^o^q => o])
    could not re-solve without it (gh-ocannl-1656). [derive_projections] gives these constraints to
    its projection-local solve only, never to [Row.get_proj_equations] -- a size equates no index
    (and a dimension-1 value there would pin the axis to [Fixed_idx 0]). *)
let bind_delayed_vars_to_envs ~for_projections ~spec ~dim_var_env ~row_var_env ~error_shapes
    dim_refs =
  (* Every refusal before any binding (gh-ocannl-1655). Projection derivation binds nothing and
     emits no ties, so it is exempt. *)
  if not for_projections then
    List.iter dim_refs ~f:(fun delayed_ref ->
        let label = delayed_ref.var_ref.ref_label in
        check_before_binding ~where:[%string "in spec \"%{spec}\""] ~error_shapes
          ~is_row:((not (Hashtbl.mem dim_var_env label)) && Hashtbl.mem row_var_env label)
          delayed_ref);
  List.concat_map dim_refs ~f:(fun delayed_ref ->
      check_ref_live delayed_ref;
      let label = delayed_ref.var_ref.ref_label in
      match Hashtbl.find dim_var_env label with
      | Some var -> bind_ref ~for_projections ~error_shapes delayed_ref (`Dim var)
      | None -> (
          match Hashtbl.find row_var_env label with
          | Some var -> bind_ref ~for_projections ~error_shapes delayed_ref (`Row var)
          | None ->
              raise
              @@ Row.Shape_error
                   ( "Variable " ^ label ^ " not found in environments for spec: " ^ spec
                     ^ split_capture_label_hint ~spec ~dim_var_env label,
                     [ Shape_mismatch error_shapes ] )))

(** Refuses a padded ([=+]) window in the result slot of an einsum spec (gh-ocannl-1550). A padded
    window on the result would be a padded transposed convolution: the result's index is the affine
    combination, so its projection is never iterated, and projection inference has no semantics for
    it. The check is structural over the parsed result slot (every kind and position of its [labels]
    map) and runs before any elaboration. Operand-side padded windows, unpadded ([<+]) and
    stride-only result axes are unaffected; the backward of an operand-side padded window reuses the
    forward's projections and never comes here with a reversed spec. The message gives the two-step
    recipe (gh-ocannl-1619): the full transposed convolution, then a crop by a slice of a
    concatenated axis (gh-ocannl-1656), executed against PyTorch's [ConvTranspose1d] semantics in
    [test/einsum/result_side_affine_forms.ml]. *)
let refuse_result_side_padded_window ~spec ~(error_shapes : t list) (ls_lhs : parsed_axis_labels) =
  Map.iter ls_lhs.labels ~f:(function
    | Affine_spec { conv = Some { use_padding = `True; _ }; _ } as axis ->
        let window = render_affine_axis axis in
        raise
        @@ Row.Shape_error
             ( [%string
                 "result-side padded window `%{window}` in einsum spec \"%{spec}\" is not \
                  supported: a padded (=+) window on the result is a padded transposed \
                  convolution, which OCANNL does not implement in one step. Build it in two: the \
                  full transposed convolution, with the result-side window unpadded (<+; its \
                  result extent is stride*(n-1) + the dilated kernel extent for an operand extent \
                  n), then a crop of `padding` cells at its start and `padding - output_padding` \
                  at its end (PyTorch's ConvTranspose arguments), written as a slice of a \
                  concatenated axis: \"s, p^o^q => s, o\" with the components p = padding and q = \
                  padding - output_padding captured and set (Shape.set_dim), or written as integer \
                  components (\"s, 2^o^1 => s, o\"); when output_padding > padding, slice only the \
                  start and pad the end with the output_padding - padding extra zero cells by a \
                  replacement (\"s, a => s, a^r\"); see docs/syntax_extensions.md. With stride 1, \
                  dilation 1 and an odd kernel size, the padded window on an operand with the \
                  kernel reversed (e.g. \"s, o=+k; k => s, o\") is the same map in one step."],
               [ Shape_mismatch error_shapes ] )
    | Label _ | Fixed_index _ | Affine_spec _ | Concat_spec _ -> ())

(** The kernels of [spec] that projection derivation can never size (gh-ocannl-1803): those that
    occur only as the kernel of a padded window and are bound by none of [dim_refs]. Each derivation
    re-reads the spec with fresh variables ([get_inequalities]), and [Row.unify_dim] drops a padded
    window's kernel, so such a kernel has a size only if the label occurs some other way (a plain
    axis, an [over], a concatenation component, an unpadded window's kernel) or a capture
    reference's solved value reaches it ({!bind_delayed_vars_to_envs}). Any capture counts, set or
    not: it may be set later. The answer reads no shape, so it holds for every application. Only
    operand windows count: a result-side padded window is {!refuse_result_side_padded_window}'s to
    refuse, whatever its kernel. *)
let unsized_padded_kernels ~spec ~(dim_refs : delayed_var_ref list) =
  let ls_rhs_list, ls_lhs = einsum_of_spec spec in
  let operand_axes = List.concat_map ls_rhs_list ~f:(fun ls -> Map.data ls.labels) in
  let axes = Map.data ls_lhs.labels @ operand_axes in
  let sizing =
    List.concat_map axes ~f:(function
      | Label l -> [ l ]
      | Fixed_index _ -> []
      | Affine_spec { over_label; conv = None | Some { use_padding = `True; _ }; _ } ->
          [ over_label ]
      | Affine_spec { over_label; conv = Some { kernel_label; _ }; _ } ->
          [ over_label; kernel_label ]
      | Concat_spec components ->
          List.filter_map components ~f:(function
            | Concat_label l -> Some l
            | Concat_size _ -> None))
    @ List.map dim_refs ~f:(fun r -> r.var_ref.ref_label)
  in
  List.filter_map operand_axes ~f:(function
    | Affine_spec { conv = Some { kernel_label; use_padding = `True; _ }; _ }
      when not (List.mem sizing kernel_label ~equal:String.equal) ->
        Some kernel_label
    | _ -> None)
  |> List.dedup_and_sort ~compare:String.compare

(** The spec and captures of an einsum-family logic. *)
let einsum_of_logic = function
  | Transpose (Permute (spec, dim_refs), _)
  | Broadcast (Einsum (spec, dim_refs), _, _)
  | Broadcast_tern (Einsum_tern (spec, dim_refs), _, _, _)
  | Block { spec; delayed_vars = dim_refs; _ } ->
      Some (spec, dim_refs)
  | Transpose _ | Broadcast _ | Broadcast_tern _ | Terminal _ -> None

(** Shared n-ary einsum constraint builder used by [Broadcast (Einsum ...)],
    [Broadcast_tern (Einsum_tern ...)], and [Block]. Handles per-shape
    [einsum_slot_spec_to_dims_bio] calls, name-clash check, delayed-var binding, proj-env merge, and
    [Row_eq] emission.

    [rhs_slots] is a list of [(parsed_axis_labels, shape)] pairs, one per RHS. [result_label] is the
    operation string for LHS [Row_eq] origins (e.g. ["Broadcast RESULT"] or ["Block RESULT"]).
    [arg_label_prefix] is the prefix for RHS [Row_eq] origins; the index (1-based) is appended,
    yielding e.g. ["Broadcast ARGUMENT 1"]. [error_shapes] is the shape list for error context.

    Returns [(proj_env, bio_lhs, bio_rhs_list, inequalities)] where [bio_lhs] and [bio_rhs_list] are
    the batch/input/output row triples for the LHS and each RHS — needed by [Block] to compute
    discardable vars. *)
let einsum_n_constraints ~for_projections ~spec ~(rhs_slots : (parsed_axis_labels * t) list)
    ~(cur_sh : t) ~ls_lhs ~dim_refs ~result_label ~arg_label_prefix ~(error_shapes : t list)
    ?(lhs_constraints_first = false) () =
  refuse_result_side_padded_window ~spec ~error_shapes ls_lhs;
  let row_var_env = Hashtbl.create (module String) in
  let dim_var_env = Hashtbl.create (module String) in
  let rhs_results =
    List.mapi rhs_slots ~f:(fun idx (ls_rhs, (sh : t)) ->
        let extras, proj_env, bio =
          einsum_slot_spec_to_dims_bio ~original_spec:spec ~sh_id:sh.id ~row_var_env ~dim_var_env
            ls_rhs
        in
        (idx, sh, extras, proj_env, bio))
  in
  let extras_lhs, proj_env_lhs, ((b_lhs, i_lhs, o_lhs) as bio_lhs) =
    einsum_slot_spec_to_dims_bio ~original_spec:spec ~sh_id:(cur_sh : t).id ~row_var_env
      ~dim_var_env ls_lhs
  in
  check_dim_row_var_name_clash ~spec ~error_trace:[ Shape_mismatch error_shapes ] dim_var_env
    row_var_env;
  let extras_dim_refs =
    bind_delayed_vars_to_envs ~for_projections ~spec ~dim_var_env ~row_var_env ~error_shapes
      dim_refs
  in
  let proj_env =
    let combine ~key:_ _ _ = assert false in
    List.fold rhs_results ~init:proj_env_lhs ~f:(fun acc (_, _, _, proj_env_rhs, _) ->
        Map.merge_skewed ~combine acc proj_env_rhs)
  in
  let rhs_constraints =
    List.concat_map rhs_results ~f:(fun (idx, sh, extras, _, (b_rhs, i_rhs, o_rhs)) ->
        let arg_label = Printf.sprintf "%s %d" arg_label_prefix (idx + 1) in
        extras
        @ [
            Row.Row_eq
              {
                r1 = b_rhs;
                r2 = sh.batch;
                origin =
                  [
                    {
                      Row.lhs_name = spec;
                      lhs_kind = `Batch;
                      rhs_name = sh.debug_name;
                      rhs_kind = `Batch;
                      operation = Some arg_label;
                    };
                  ];
              };
            Row.Row_eq
              {
                r1 = i_rhs;
                r2 = sh.input;
                origin =
                  [
                    {
                      Row.lhs_name = spec;
                      lhs_kind = `Input;
                      rhs_name = sh.debug_name;
                      rhs_kind = `Input;
                      operation = Some arg_label;
                    };
                  ];
              };
            Row.Row_eq
              {
                r1 = o_rhs;
                r2 = sh.output;
                origin =
                  [
                    {
                      Row.lhs_name = spec;
                      lhs_kind = `Output;
                      rhs_name = sh.debug_name;
                      rhs_kind = `Output;
                      operation = Some arg_label;
                    };
                  ];
              };
          ])
  in
  let lhs_constraints =
    [
      Row.Row_eq
        {
          r1 = cur_sh.batch;
          r2 = b_lhs;
          origin =
            [
              {
                Row.lhs_name = cur_sh.debug_name;
                lhs_kind = `Batch;
                rhs_name = spec;
                rhs_kind = `Batch;
                operation = Some result_label;
              };
            ];
        };
      Row.Row_eq
        {
          r1 = cur_sh.input;
          r2 = i_lhs;
          origin =
            [
              {
                Row.lhs_name = cur_sh.debug_name;
                lhs_kind = `Input;
                rhs_name = spec;
                rhs_kind = `Input;
                operation = Some result_label;
              };
            ];
        };
      Row.Row_eq
        {
          r1 = cur_sh.output;
          r2 = o_lhs;
          origin =
            [
              {
                Row.lhs_name = cur_sh.debug_name;
                lhs_kind = `Output;
                rhs_name = spec;
                rhs_kind = `Output;
                operation = Some result_label;
              };
            ];
        };
    ]
  in
  let bio_rhs_list = List.map rhs_results ~f:(fun (_, _, _, _, bio) -> bio) in
  let ordered_constraints =
    if lhs_constraints_first then lhs_constraints @ rhs_constraints
    else rhs_constraints @ lhs_constraints
  in
  (proj_env, bio_lhs, bio_rhs_list, extras_dim_refs @ extras_lhs @ ordered_constraints)

(* For Block specs, compute discardable_vars: variables that are allowed to be 0. A variable v is
   discardable if: 1. v appears in a component of a Concat dimension on one side 2. For ALL shapes
   on the other side, there EXISTS an axis such that for ALL components of that axis, the complement
   of v's component has non-empty intersection. This is a four-quantifier condition: ∀shapes ∃axis
   ∀components: complement ∩ component ≠ ∅ *)
let compute_block_discardable_vars ~(this_side_rows : Row.t list)
    ~(other_side_shapes : Row.t list list) : Row.dim_var_set =
  (* Extract all dims from rows (both flanks; bcast no longer carries beg_dims). *)
  let dims_of_rows rows = List.concat_map rows ~f:(fun (row : Row.t) -> row.beg_dims @ row.dims) in
  (* Get component var sets for each axis. For non-Concat dims, there's one component with all vars
     of that dim. For Concat dims, each element is a separate component. *)
  let axis_components_of_dim (dim : Row.dim) : Row.dim_var_set list =
    match dim with
    | Row.Concat dims -> List.map dims ~f:Row.vars_of_dim
    | _ -> [ Row.vars_of_dim dim ]
  in
  (* For each shape on the other side, get its axes (each axis is a list of component var sets) *)
  let other_side_shape_axes : Row.dim_var_set list list list =
    List.map other_side_shapes ~f:(fun shape_rows ->
        List.map (dims_of_rows shape_rows) ~f:axis_components_of_dim)
  in
  (* For each Concat on this side, find discardable vars. Non-Concat dims cannot contribute
     discardable vars since their complement is empty. *)
  let this_side_dims = dims_of_rows this_side_rows in
  List.fold this_side_dims ~init:Row.dim_var_set_empty ~f:(fun acc dim ->
      match dim with
      | Row.Concat components ->
          let component_var_sets = List.map components ~f:Row.vars_of_dim in
          let all_concat_vars =
            List.fold component_var_sets ~init:Row.dim_var_set_empty ~f:Set.union
          in
          List.fold component_var_sets ~init:acc ~f:(fun acc2 component_vars ->
              let complement = Set.diff all_concat_vars component_vars in
              (* Check: for ALL shapes, EXISTS an axis where ALL components intersect complement *)
              let complement_covers_all_shapes =
                List.for_all other_side_shape_axes ~f:(fun shape_axes ->
                    List.exists shape_axes ~f:(fun axis_components ->
                        List.for_all axis_components ~f:(fun axis_comp_vars ->
                            not (Set.is_empty (Set.inter complement axis_comp_vars)))))
              in
              if complement_covers_all_shapes then Set.union acc2 component_vars else acc2)
      | _ -> acc)

let unused_shapes = ref (Set.empty (module Int))

(* Writes to [unused_shapes] count as session mutations (gh-ocannl-1301): it filters which rows
   [finish_inference] closes. *)
let unmark_unused id =
  if Set.mem !unused_shapes id then (
    Row.note_mutation ();
    unused_shapes := Set.remove !unused_shapes id)

let mark_unused id =
  if not (Set.mem !unused_shapes id) then (
    Row.note_mutation ();
    unused_shapes := Set.add !unused_shapes id)

let%debug4_sexp get_inequalities ?(for_projections = false)
    ({ shape = cur_sh; logic; _ } as _upd : update_step) :
    proj_axis_env * Row.dim_var_set * Row.constraint_ list =
  unmark_unused cur_sh.id;
  let _debug_cur_sh : t = cur_sh in
  let _debug_logic : logic = logic in
  let open Row in
  let defaults ineqs = (dim_map_empty, dim_var_set_empty, ineqs) in
  let get_origin lhs_kind sh rhs_kind operation =
    Row.
      {
        lhs_name = cur_sh.debug_name;
        lhs_kind;
        rhs_name = sh.debug_name;
        rhs_kind;
        operation = Some operation;
      }
  in
  let mark_terminal ~is_param =
    [
      Terminal_row (is_param, cur_sh.batch, [ get_origin `Batch cur_sh `Batch "terminal" ]);
      Terminal_row (is_param, cur_sh.input, [ get_origin `Input cur_sh `Input "terminal" ]);
      Terminal_row (is_param, cur_sh.output, [ get_origin `Output cur_sh `Output "terminal" ]);
    ]
  in
  match logic with
  | Terminal { is_param; logic = Fetch Range_over_offsets } -> defaults @@ mark_terminal ~is_param
  | Terminal { is_param; logic = Fetch (Constant _) | Data (Broadcast _) } ->
      defaults @@ mark_terminal ~is_param
  | Terminal { is_param; logic = Fetch (Constant_bits _) } -> defaults @@ mark_terminal ~is_param
  | Terminal { is_param; logic = Data (Reshape nd) } ->
      ( dim_map_empty,
        dim_var_set_empty,
        Rows_constr
          {
            r = [ cur_sh.batch; cur_sh.output; cur_sh.input ];
            constr =
              (* [keep_axis]: a 1-element reshape prefers keeping a single dim-1 axis over
                 collapsing to a scalar when nothing else constrains the rows (gh-460). *)
              Total_elems
                {
                  numerator = Num_elems (Array.fold (Ir.Ndarray.dims nd) ~init:1 ~f:( * ));
                  divided_by = [];
                  keep_axis = true;
                };
            origin =
              [
                {
                  lhs_name = cur_sh.debug_name;
                  lhs_kind = `Batch;
                  rhs_name = "Reshape";
                  rhs_kind = `Output;
                  operation = Some "Total_elems";
                };
              ];
          }
        :: mark_terminal ~is_param )
  | Terminal { is_param; logic = Data (Keep_shape_no_padding nd) } ->
      (* FIXME: constrain padding to "not padded". *)
      ( dim_map_empty,
        dim_var_set_empty,
        (if for_projections then []
         else
           [
             Rows_constr
               {
                 r = [ cur_sh.batch; cur_sh.output; cur_sh.input ];
                 constr =
                   Exact
                     (Ir.Ndarray.dims nd
                     |> Array.map ~f:(fun d -> get_default_dim ~d ())
                     |> Array.to_list);
                 origin =
                   [
                     {
                       lhs_name = cur_sh.debug_name;
                       lhs_kind = `Batch;
                       rhs_name = "Keep_shape_no_padding";
                       rhs_kind = `Output;
                       operation = Some "Exact";
                     };
                   ];
               };
           ])
        @ mark_terminal ~is_param )
  | Terminal { is_param; logic = Data (Padded { data; padding; padded_value }) } ->
      ignore padded_value;
      ( dim_map_empty,
        dim_var_set_empty,
        (if for_projections then []
         else
           [
             Rows_constr
               {
                 r = [ cur_sh.batch; cur_sh.output; cur_sh.input ];
                 constr =
                   Exact
                     (* The ndarray is padded as given: the shape's (logical) dims exclude the
                        margins. The committed widths and neutral element themselves are enforced
                        against later demands via the tnode's forced padding lazy (see
                        [fresh_proj_ids]). *)
                     (Ir.Ndarray.dims data
                     |> Array.mapi ~f:(fun i d ->
                         let d =
                           if i < Array.length padding then
                             d - padding.(i).Ir.Ops.left - padding.(i).Ir.Ops.right
                           else d
                         in
                         get_default_dim ~d ())
                     |> Array.to_list);
                 origin =
                   [
                     {
                       lhs_name = cur_sh.debug_name;
                       lhs_kind = `Batch;
                       rhs_name = "Padded";
                       rhs_kind = `Output;
                       operation = Some "Exact";
                     };
                   ];
               };
           ])
        @ mark_terminal ~is_param )
  | Terminal { is_param; logic = Fetch (Constant_fill values) } ->
      let len = Array.length values in
      ( dim_map_empty,
        dim_var_set_empty,
        Rows_constr
          {
            r = [ cur_sh.batch; cur_sh.output; cur_sh.input ];
            constr = Total_elems { numerator = Num_elems len; divided_by = []; keep_axis = false };
            origin =
              [
                {
                  lhs_name = cur_sh.debug_name;
                  lhs_kind = `Batch;
                  rhs_name = "Constant_fill";
                  rhs_kind = `Output;
                  operation = Some "Total_elems";
                };
              ];
          }
        :: mark_terminal ~is_param )
  | Terminal { is_param; logic = Fetch (Slice { sliced = tn; batch_idx = _ }) } ->
      if Lazy.is_val tn.dims then
        ( dim_map_empty,
          dim_var_set_empty,
          (if for_projections then []
           else
             [
               Rows_constr
                 {
                   r = [ cur_sh.batch; cur_sh.output; cur_sh.input ];
                   constr =
                     Exact
                       (Lazy.force tn.dims |> Array.to_list |> List.tl_exn
                       |> List.map ~f:(fun d -> get_default_dim ~d ()));
                   origin =
                     [
                       {
                         lhs_name = cur_sh.debug_name;
                         lhs_kind = `Batch;
                         rhs_name = Tn.label_name tn;
                         rhs_kind = `Output;
                         operation = Some "Slice";
                       };
                     ];
                 };
             ])
          @ mark_terminal ~is_param )
      else defaults @@ mark_terminal ~is_param
  | Terminal { is_param; logic = Fetch (Embed_symbol _) } -> defaults @@ mark_terminal ~is_param
  | Terminal { is_param; logic = Fetch (Embed_dim _) } -> defaults @@ mark_terminal ~is_param
  | Terminal { is_param; logic = Fetch Embed_self_id } -> defaults @@ mark_terminal ~is_param
  | Transpose (Transpose, sh) ->
      unmark_unused sh.id;
      defaults
      @@ [
           Row_ineq
             {
               res = cur_sh.batch;
               opnd = sh.batch;
               origin = [ get_origin `Batch sh `Batch "transpose" ];
             };
           Row_ineq
             {
               res = cur_sh.input;
               opnd = sh.output;
               origin = [ get_origin `Input sh `Output "transpose" ];
             };
           Row_ineq
             {
               res = cur_sh.output;
               opnd = sh.input;
               origin = [ get_origin `Output sh `Input "transpose" ];
             };
         ]
  | Transpose (Pointwise_un, sh) ->
      unmark_unused sh.id;
      add_var_used_in_pointwise cur_sh.input.bcast;
      add_var_used_in_pointwise sh.input.bcast;
      defaults
      @@ [
           Row_ineq
             {
               res = cur_sh.batch;
               opnd = sh.batch;
               origin = [ get_origin `Batch sh `Batch "pointwise_unary" ];
             };
           Row_ineq
             {
               res = cur_sh.input;
               opnd = sh.input;
               origin = [ get_origin `Input sh `Input "pointwise_unary" ];
             };
           Row_ineq
             {
               res = cur_sh.output;
               opnd = sh.output;
               origin = [ get_origin `Output sh `Output "pointwise_unary" ];
             };
         ]
  | Broadcast (Compose, sh1, sh2) ->
      unmark_unused sh1.id;
      unmark_unused sh2.id;
      add_var_used_in_spec_or_compose sh1.input.bcast;
      defaults
      @@ [
           Row_ineq
             {
               origin =
                 [
                   {
                     lhs_name = sh1.debug_name;
                     lhs_kind = `Input;
                     rhs_name = sh2.debug_name;
                     rhs_kind = `Output;
                     operation = Some "compose";
                   };
                 ];
               res = sh1.input;
               opnd = sh2.output;
             };
           Row_ineq
             {
               origin = [ get_origin `Batch sh1 `Batch "compose" ];
               res = cur_sh.batch;
               opnd = sh1.batch;
             };
           Row_ineq
             {
               origin = [ get_origin `Batch sh2 `Batch "compose" ];
               res = cur_sh.batch;
               opnd = sh2.batch;
             };
           Row_ineq
             {
               origin = [ get_origin `Input sh2 `Input "compose" ];
               res = cur_sh.input;
               opnd = sh2.input;
             };
           Row_ineq
             {
               origin = [ get_origin `Output sh1 `Output "compose" ];
               res = cur_sh.output;
               opnd = sh1.output;
             };
         ]
  | Broadcast (Pointwise_bin, sh1, sh2) ->
      unmark_unused sh1.id;
      unmark_unused sh2.id;
      add_var_used_in_pointwise cur_sh.input.bcast;
      add_var_used_in_pointwise sh1.input.bcast;
      add_var_used_in_pointwise sh2.input.bcast;
      defaults
      @@ [
           Row_ineq
             {
               origin = [ get_origin `Batch sh1 `Batch "pointwise_binary" ];
               res = cur_sh.batch;
               opnd = sh1.batch;
             };
           Row_ineq
             {
               origin = [ get_origin `Batch sh2 `Batch "pointwise_binary" ];
               res = cur_sh.batch;
               opnd = sh2.batch;
             };
           Row_ineq
             {
               origin = [ get_origin `Input sh1 `Input "pointwise_binary" ];
               res = cur_sh.input;
               opnd = sh1.input;
             };
           Row_ineq
             {
               origin = [ get_origin `Input sh2 `Input "pointwise_binary" ];
               res = cur_sh.input;
               opnd = sh2.input;
             };
           Row_ineq
             {
               origin = [ get_origin `Output sh1 `Output "pointwise_binary" ];
               res = cur_sh.output;
               opnd = sh1.output;
             };
           Row_ineq
             {
               origin = [ get_origin `Output sh2 `Output "pointwise_binary" ];
               res = cur_sh.output;
               opnd = sh2.output;
             };
         ]
  | Broadcast_tern (Compose_accumulate, sh1, sh2, sh3) ->
      unmark_unused sh1.id;
      unmark_unused sh2.id;
      unmark_unused sh3.id;
      add_var_used_in_spec_or_compose sh1.input.bcast;
      defaults
      @@ [
           Row_ineq
             {
               origin =
                 [
                   {
                     lhs_name = sh1.debug_name;
                     lhs_kind = `Input;
                     rhs_name = sh2.debug_name;
                     rhs_kind = `Output;
                     operation = Some "compose_accumulate";
                   };
                 ];
               res = sh1.input;
               opnd = sh2.output;
             };
           Row_ineq
             {
               origin = [ get_origin `Batch sh1 `Batch "compose_accumulate" ];
               res = cur_sh.batch;
               opnd = sh1.batch;
             };
           Row_ineq
             {
               origin = [ get_origin `Batch sh2 `Batch "compose_accumulate" ];
               res = cur_sh.batch;
               opnd = sh2.batch;
             };
           Row_ineq
             {
               origin = [ get_origin `Input sh2 `Input "compose_accumulate" ];
               res = cur_sh.input;
               opnd = sh2.input;
             };
           Row_ineq
             {
               origin = [ get_origin `Output sh1 `Output "compose_accumulate" ];
               res = cur_sh.output;
               opnd = sh1.output;
             };
           Row_ineq
             {
               origin = [ get_origin `Batch sh3 `Batch "compose_accumulate" ];
               res = cur_sh.batch;
               opnd = sh3.batch;
             };
           Row_ineq
             {
               origin = [ get_origin `Input sh3 `Input "compose_accumulate" ];
               res = cur_sh.input;
               opnd = sh3.input;
             };
           Row_ineq
             {
               origin = [ get_origin `Output sh3 `Output "compose_accumulate" ];
               res = cur_sh.output;
               opnd = sh3.output;
             };
         ]
  | Broadcast_tern (Pointwise_tern, sh1, sh2, sh3) ->
      unmark_unused sh1.id;
      unmark_unused sh2.id;
      unmark_unused sh3.id;
      add_var_used_in_pointwise cur_sh.input.bcast;
      add_var_used_in_pointwise sh1.input.bcast;
      add_var_used_in_pointwise sh2.input.bcast;
      add_var_used_in_pointwise sh3.input.bcast;
      defaults
      @@ [
           Row_ineq
             {
               origin = [ get_origin `Batch sh1 `Batch "pointwise_ternary" ];
               res = cur_sh.batch;
               opnd = sh1.batch;
             };
           Row_ineq
             {
               origin = [ get_origin `Batch sh2 `Batch "pointwise_ternary" ];
               res = cur_sh.batch;
               opnd = sh2.batch;
             };
           Row_ineq
             {
               origin = [ get_origin `Batch sh3 `Batch "pointwise_ternary" ];
               res = cur_sh.batch;
               opnd = sh3.batch;
             };
           Row_ineq
             {
               origin = [ get_origin `Input sh1 `Input "pointwise_ternary" ];
               res = cur_sh.input;
               opnd = sh1.input;
             };
           Row_ineq
             {
               origin = [ get_origin `Input sh2 `Input "pointwise_ternary" ];
               res = cur_sh.input;
               opnd = sh2.input;
             };
           Row_ineq
             {
               origin = [ get_origin `Input sh3 `Input "pointwise_ternary" ];
               res = cur_sh.input;
               opnd = sh3.input;
             };
           Row_ineq
             {
               origin = [ get_origin `Output sh1 `Output "pointwise_ternary" ];
               res = cur_sh.output;
               opnd = sh1.output;
             };
           Row_ineq
             {
               origin = [ get_origin `Output sh2 `Output "pointwise_ternary" ];
               res = cur_sh.output;
               opnd = sh2.output;
             };
           Row_ineq
             {
               origin = [ get_origin `Output sh3 `Output "pointwise_ternary" ];
               res = cur_sh.output;
               opnd = sh3.output;
             };
         ]
  | Broadcast (Defined_by_cd_logic, _, _)
  | Transpose (Defined_by_cd_logic, _)
  | Broadcast_tern (Defined_by_cd_logic, _, _, _) ->
      defaults @@ []
  | Transpose (Batch_slice ({ static_range; static_symbol; _ } as slice_sym), sh) ->
      (* A slice index and a symbolic extent (gh-490) validate incompatibly (strict [0, range) vs.
         inclusive [0, range]): reject a binding used as both. *)
      if Row.is_used_as_extent slice_sym then
        raise
        @@ Row.Shape_error
             ( "Static index " ^ Idx.symbol_ident static_symbol
               ^ " is used as a symbolic dimension extent and cannot also slice a batch axis",
               [] );
      Row.claim_symbol_role slice_sym `Slice;
      unmark_unused sh.id;
      let slice_v = get_var () in
      let slice_var = Var slice_v in
      let proj_axis_env =
        Map.add_exn Row.dim_map_empty ~key:slice_v ~data:(Idx.Iterator static_symbol)
      in
      (* Expand a batch row instead of reducing one because even if the dimensions are known, the
         equations are also needed for projection inference. *)
      let expanded_batch =
        (* Slice axis is outer-anchored; prepend to beg_dims for both closed and open rows. *)
        {
          beg_dims = slice_var :: cur_sh.batch.beg_dims;
          dims = cur_sh.batch.dims;
          bcast = cur_sh.batch.bcast;
          prov = Row.provenance ~sh_id:cur_sh.id ~kind:`Batch;
        }
      in
      let get_origin kind =
        [
          {
            lhs_name = cur_sh.debug_name;
            lhs_kind = kind;
            rhs_name = sh.debug_name;
            rhs_kind = kind;
            operation = Some "Batch_slice";
          };
        ]
      in
      ( proj_axis_env,
        dim_var_set_empty,
        (Option.to_list (if for_projections then None else static_range)
        |> List.map ~f:(fun range ->
            Dim_eq
              {
                d1 = get_default_dim ~d:range ();
                d2 = slice_var;
                origin =
                  [
                    {
                      lhs_name = sh.debug_name;
                      lhs_kind = `Batch;
                      rhs_name = Idx.symbol_ident static_symbol;
                      rhs_kind = `Batch;
                      operation = Some "Slice";
                    };
                  ];
              }))
        @ [
            Row_eq { r1 = expanded_batch; r2 = sh.batch; origin = get_origin `Batch };
            Row_eq { r1 = cur_sh.input; r2 = sh.input; origin = get_origin `Input };
            Row_eq { r1 = cur_sh.output; r2 = sh.output; origin = get_origin `Output };
          ] )
  | Transpose (Permute (spec, dim_refs), sh) ->
      unmark_unused sh.id;
      add_var_used_in_spec_or_compose cur_sh.input.bcast;
      add_var_used_in_spec_or_compose sh.input.bcast;
      let ls_rhs, ls_lhs =
        match einsum_of_spec spec with
        | [ ls_rhs ], ls_lhs -> (ls_rhs, ls_lhs)
        | _ ->
            raise
            @@ Shape_error
                 ( "Invalid permutation spec (expected one argument): " ^ spec,
                   [ Shape_mismatch [ cur_sh; sh ] ] )
      in
      refuse_result_side_padded_window ~spec ~error_shapes:[ cur_sh; sh ] ls_lhs;
      let row_var_env = Hashtbl.create (module String) in
      let dim_var_env = Hashtbl.create (module String) in

      let extras_rhs, proj_env_rhs, (b_rhs, i_rhs, o_rhs) =
        einsum_slot_spec_to_dims_bio ~original_spec:spec ~sh_id:sh.id ~row_var_env ~dim_var_env
          ls_rhs
      in
      let extras_lhs, proj_env_lhs, (b_lhs, i_lhs, o_lhs) =
        einsum_slot_spec_to_dims_bio ~original_spec:spec ~sh_id:cur_sh.id ~row_var_env ~dim_var_env
          ls_lhs
      in
      check_dim_row_var_name_clash ~spec
        ~error_trace:[ Shape_mismatch [ cur_sh; sh ] ]
        dim_var_env row_var_env;
      let extras_dim_refs =
        bind_delayed_vars_to_envs ~for_projections ~spec ~dim_var_env ~row_var_env
          ~error_shapes:[ cur_sh; sh ] dim_refs
      in
      let proj_env =
        let combine ~key:_ _ _ = assert false in
        Map.merge_skewed ~combine proj_env_rhs proj_env_lhs
      in
      ( proj_env,
        dim_var_set_empty,
        extras_dim_refs @ extras_rhs @ extras_lhs
        @ [
            Row_eq
              {
                r1 = cur_sh.batch;
                r2 = b_lhs;
                origin =
                  [
                    {
                      lhs_name = cur_sh.debug_name;
                      lhs_kind = `Batch;
                      rhs_name = spec;
                      rhs_kind = `Batch;
                      operation = Some "Permute RESULT";
                    };
                  ];
              };
            Row_eq
              {
                r1 = b_rhs;
                r2 = sh.batch;
                origin =
                  [
                    {
                      lhs_name = spec;
                      lhs_kind = `Batch;
                      rhs_name = sh.debug_name;
                      rhs_kind = `Batch;
                      operation = Some "Permute ARGUMENT";
                    };
                  ];
              };
            Row_eq
              {
                r1 = cur_sh.input;
                r2 = i_lhs;
                origin =
                  [
                    {
                      lhs_name = cur_sh.debug_name;
                      lhs_kind = `Input;
                      rhs_name = spec;
                      rhs_kind = `Input;
                      operation = Some "Permute RESULT";
                    };
                  ];
              };
            Row_eq
              {
                r1 = i_rhs;
                r2 = sh.input;
                origin =
                  [
                    {
                      lhs_name = spec;
                      lhs_kind = `Input;
                      rhs_name = sh.debug_name;
                      rhs_kind = `Input;
                      operation = Some "Permute ARGUMENT";
                    };
                  ];
              };
            Row_eq
              {
                r1 = cur_sh.output;
                r2 = o_lhs;
                origin =
                  [
                    {
                      lhs_name = cur_sh.debug_name;
                      lhs_kind = `Output;
                      rhs_name = spec;
                      rhs_kind = `Output;
                      operation = Some "Permute RESULT";
                    };
                  ];
              };
            Row_eq
              {
                r1 = o_rhs;
                r2 = sh.output;
                origin =
                  [
                    {
                      lhs_name = spec;
                      lhs_kind = `Output;
                      rhs_name = sh.debug_name;
                      rhs_kind = `Output;
                      operation = Some "Permute ARGUMENT";
                    };
                  ];
              };
          ] )
  | Transpose (Uint4x32_to_prec target_prec, sh) ->
      let var = get_var () in
      let own_coeff () =
        Utils.safe_lazy [%string "Uint4x32 %{sh.id#Int} to_prec_of %{cur_sh.id#Int}"] (fun () ->
            16 / Ir.Ops.prec_in_bytes (Lazy.force target_prec))
      in
      let coeff =
        match !memoized_draw_coefficient with
        | None -> own_coeff ()
        | Some memo -> (
            match !memo with
            | Some c -> c
            | None ->
                let c = own_coeff () in
                memo := Some c;
                c)
      in
      defaults
      @@ [
           Rows_constr
             {
               r = [ sh.batch; sh.output; sh.input ];
               constr = Row.Exact [ Var var ];
               origin =
                 [
                   {
                     lhs_name = sh.debug_name;
                     lhs_kind = `Batch;
                     rhs_name = cur_sh.debug_name;
                     rhs_kind = `Batch;
                     operation = Some "Uint4x32_to_prec ARGUMENT axes exact";
                   };
                 ];
             };
           Rows_constr
             {
               r = [ cur_sh.batch; cur_sh.output; cur_sh.input ];
               constr =
                 Total_elems
                   {
                     (* [round_up]: the result total need not be a multiple of [coeff]; the counter
                        gets [ceil(total / coeff)] blocks and the last block is consumed partially
                        (tail-peeled in [Assignments] lowering). *)
                     numerator = Row.Strided_var { coeff; var; denom = 1; round_up = true };
                     divided_by = [];
                     keep_axis = false;
                   };
               origin =
                 [
                   {
                     lhs_name = cur_sh.debug_name;
                     lhs_kind = `Batch;
                     rhs_name = sh.debug_name;
                     rhs_kind = `Output;
                     operation = Some "Uint4x32_to_prec RESULT total elements";
                   };
                 ];
             };
         ]
  | Broadcast (Einsum (spec, dim_refs), sh1, sh2) ->
      unmark_unused sh1.id;
      unmark_unused sh2.id;
      add_var_used_in_spec_or_compose cur_sh.input.bcast;
      add_var_used_in_spec_or_compose sh1.input.bcast;
      add_var_used_in_spec_or_compose sh2.input.bcast;
      let ls_rhs1, ls_rhs2, ls_lhs =
        match einsum_of_spec spec with
        | [ ls_rhs1; ls_rhs2 ], ls_lhs -> (ls_rhs1, ls_rhs2, ls_lhs)
        | _ ->
            raise
            @@ Shape_error
                 ( "Invalid einsum spec (expected two arguments): " ^ spec,
                   [ Shape_mismatch [ cur_sh; sh1; sh2 ] ] )
      in
      let proj_env, _, _, inequalities =
        einsum_n_constraints ~for_projections ~spec
          ~rhs_slots:[ (ls_rhs1, sh1); (ls_rhs2, sh2) ]
          ~cur_sh ~ls_lhs ~dim_refs ~result_label:"Broadcast RESULT"
          ~arg_label_prefix:"Broadcast ARGUMENT" ~error_shapes:[ cur_sh; sh1; sh2 ]
          ~lhs_constraints_first:true ()
      in
      (proj_env, dim_var_set_empty, inequalities)
  | Block { spec; delayed_vars; rhses } ->
      List.iter rhses ~f:(fun sh -> unmark_unused sh.id);
      add_var_used_in_spec_or_compose cur_sh.input.bcast;
      List.iter rhses ~f:(fun sh -> add_var_used_in_spec_or_compose sh.input.bcast);
      let ls_rhses, ls_lhs =
        match einsum_of_spec spec with
        | ls_rhses, ls_lhs when List.length ls_rhses = List.length rhses -> (ls_rhses, ls_lhs)
        | _ ->
            raise
            @@ Shape_error
                 ( Printf.sprintf "Invalid block spec (expected %d arguments): %s"
                     (List.length rhses) spec,
                   [ Shape_mismatch (cur_sh :: rhses) ] )
      in
      let proj_env, (b_lhs, i_lhs, o_lhs), bio_rhs_list, inequalities =
        einsum_n_constraints ~for_projections ~spec ~rhs_slots:(List.zip_exn ls_rhses rhses) ~cur_sh
          ~ls_lhs ~dim_refs:delayed_vars ~result_label:"Block RESULT"
          ~arg_label_prefix:"Block ARGUMENT" ~error_shapes:(cur_sh :: rhses) ()
      in
      (* Compute discardable_vars for concat semantics: variables allowed to be 0. *)
      let rhs_shapes : Row.t list list =
        List.map bio_rhs_list ~f:(fun (b, inp, o) -> [ b; inp; o ])
      in
      let lhs_rows = [ b_lhs; i_lhs; o_lhs ] in
      let discardable_from_rhs =
        compute_block_discardable_vars ~this_side_rows:(List.concat rhs_shapes)
          ~other_side_shapes:[ lhs_rows ]
      in
      let discardable_from_lhs =
        compute_block_discardable_vars ~this_side_rows:lhs_rows ~other_side_shapes:rhs_shapes
      in
      let discardable_vars = Set.union discardable_from_rhs discardable_from_lhs in
      (proj_env, discardable_vars, inequalities)
  | Broadcast_tern (Einsum_tern (spec, dim_refs), sh1, sh2, sh3) ->
      unmark_unused sh1.id;
      unmark_unused sh2.id;
      unmark_unused sh3.id;
      add_var_used_in_spec_or_compose cur_sh.input.bcast;
      add_var_used_in_spec_or_compose sh1.input.bcast;
      add_var_used_in_spec_or_compose sh2.input.bcast;
      add_var_used_in_spec_or_compose sh3.input.bcast;
      let ls_rhs1, ls_rhs2, ls_rhs3, ls_lhs =
        match einsum_of_spec spec with
        | [ ls_rhs1; ls_rhs2; ls_rhs3 ], ls_lhs -> (ls_rhs1, ls_rhs2, ls_rhs3, ls_lhs)
        | _ ->
            raise
            @@ Shape_error
                 ( "Invalid einsum spec (expected three arguments): " ^ spec,
                   [ Shape_mismatch [ cur_sh; sh1; sh2; sh3 ] ] )
      in
      let proj_env, _, _, inequalities =
        einsum_n_constraints ~for_projections ~spec
          ~rhs_slots:[ (ls_rhs1, sh1); (ls_rhs2, sh2); (ls_rhs3, sh3) ]
          ~cur_sh ~ls_lhs ~dim_refs ~result_label:"Broadcast RESULT"
          ~arg_label_prefix:"Broadcast ARGUMENT" ~error_shapes:[ cur_sh; sh1; sh2; sh3 ] ()
      in
      (proj_env, dim_var_set_empty, inequalities)

let state = ref Row.empty_env [@@mutation_serial.fingerprinted]

(* Depth of tensor-operation constructions in progress: [Tensor.op] raises it around its [op_asn]
   (see [with_construction_window]). While it is positive, finalizing inference is refused at the
   finalizer: the operation whose assignments are being built has not registered its constraints
   yet, so a [finish_inference] here would close its operands without them and commit that to the
   shapes, which no later step can undo (gh-ocannl-830). *)
let construction_depth = ref 0
[@@mutation_serial.exempt "the construction window's depth, restored by the window's own finally"]

let active_update_steps = ref [] [@@mutation_serial.fingerprinted]
let active_constraints = ref [] [@@mutation_serial.fingerprinted]

(* gh-ocannl-404: an applied shape scheme registers no update step -- there is no computation to
   project -- so the closure learns of it here: its boundary (and anchored) shapes get the solution
   applied and are closed, and their rows, with the scheme's renamed internal rows, join the update
   steps' rows at the [Shape_row] stage. Session state like the steps: captured by checkpoints,
   cleared by closures and resets. *)
type scheme_instance = {
  inst_shapes : t list;
  inst_rows : (Row.t * Row.constraint_origin) list;
  inst_refs : delayed_var_ref list;
      (** The capture references the instance bound (gh-ocannl-1791): the closure publishes their
          solutions, as it does a step's. *)
}

let active_instances : scheme_instance list ref = ref [] [@@mutation_serial.fingerprinted]

(* The labels of the shape-scheme traces in progress, innermost first: [finish_inference] refuses
   while one runs, before it commits anything (a closure would end the trace's checkpoint). *)
let tracing : string list ref = ref []
[@@mutation_serial.exempt
  "the shape-scheme traces in progress, restored by the trace's own finally"]

let tracing_in_progress () = not (List.is_empty !tracing)

let as_tracing ~label f =
  tracing := label :: !tracing;
  Exn.protect ~f ~finally:(fun () -> match !tracing with _ :: rest -> tracing := rest | [] -> ())

(* The tensors of each shape-scheme trace in progress, by debug name, innermost first
   (gh-ocannl-1903): the shapes made while it runs, its arguments, and the tensors of the schemes it
   applies. A shape error's origins naming them rank first ([Row.with_origin_focus]). *)
let traced_names : string Hash_set.t list ref = ref []
[@@mutation_serial.exempt
  "the tensor names of the shape-scheme traces in progress, restored by the trace's own finally"]

let note_traced_name name = List.iter !traced_names ~f:(fun names -> Hash_set.add names name)

let names_origin (names : string -> bool) (o : Row.constraint_origin) =
  names o.lhs_name || names o.rhs_name

(* Runs [f] as a trace whose tensors [names] collects: the solver ranks the origins naming them
   first. *)
let with_traced_names names f =
  let outer = !traced_names in
  traced_names := names :: outer;
  Exn.protect
    ~f:(fun () -> Row.with_origin_focus (names_origin (Hash_set.mem names)) f)
    ~finally:(fun () -> traced_names := outer)

(** {3 Session consistency (gh-ocannl-1301)}

    [finish_inference] is the session's only commit point. A construction that fails after mutating
    the inference session taints it: every later construction, finalization and compile is refused
    until [Tensor.rewind_to_last_closure] discards everything opened since the last successful
    closure (or [Tensor.unsafe_reinitialize] resets the session). *)

let iter_shapes update_step ~f =
  f update_step.shape;
  match update_step.logic with
  | Terminal _ -> ()
  | Transpose (_, sh1) -> f sh1
  | Broadcast (_, sh1, sh2) ->
      f sh1;
      f sh2
  | Broadcast_tern (_, sh1, sh2, sh3) ->
      f sh1;
      f sh2;
      f sh3
  | Block { rhses; _ } -> List.iter rhses ~f

let taint : string option ref = ref None
[@@mutation_serial.exempt "the taint itself: a record of the first failure, not inference state"]

let tainted_message provenance =
  [%string
    "the session is inconsistent after an earlier shape error at %{provenance}; call \
     Tensor.rewind_to_last_closure () or Tensor.unsafe_reinitialize ()"]

let check_untainted () =
  match !taint with
  | None -> ()
  | Some provenance -> raise @@ Row.Shape_error (tainted_message provenance, [])

let taint_session ~provenance = if Option.is_none !taint then taint := Some provenance
let taint_provenance () = !taint

(* What a construction can leave behind: the solver globals (compared physically -- they are
   persistent values, replaced on every write) and the serial of the other writes. *)
type fingerprint = {
  fp_serial : int;
  fp_state : Row.environment;
  fp_steps : update_step list;
  fp_constraints : Row.constraint_ list;
  fp_instances : scheme_instance list;
}

let fingerprint () =
  {
    fp_serial = Row.mutation_serial ();
    fp_state = !state;
    fp_steps = !active_update_steps;
    fp_constraints = !active_constraints;
    fp_instances = !active_instances;
  }

let mutated_since fp =
  fp.fp_serial <> Row.mutation_serial ()
  || (not (phys_equal fp.fp_state !state))
  || (not (phys_equal fp.fp_steps !active_update_steps))
  || (not (phys_equal fp.fp_constraints !active_constraints))
  || not (phys_equal fp.fp_instances !active_instances)

let exn_summary = function
  | Row.Shape_error (msg, _) -> msg
  | Untraceable_scheme msg -> msg
  | Utils.User_error msg -> msg
  | exn -> Exn.to_string exn

let with_taint_on_failure ~describe f =
  let fp = fingerprint () in
  try f ()
  with exn ->
    let bt = Stdlib.Printexc.get_raw_backtrace () in
    if mutated_since fp then taint_session ~provenance:(describe exn);
    Stdlib.Printexc.raise_with_backtrace exn bt

(* Called after every successful closure; [Tensor] records its checkpoint here. *)
let on_closure : (unit -> unit) ref = ref (fun () -> ())
[@@mutation_serial.exempt
  "a hook Tensor installs once at module initialization, not inference state"]

let set_on_closure f = on_closure := f

let check_step_live (step : update_step) =
  Option.iter step.discarded_by ~f:(fun provenance ->
      raise
      @@ Row.Shape_error
           ( [%string
               "the shape update step of #%{step.shape.id#Int} %{step.shape.debug_name} was \
                discarded by the rewind after %{provenance}; re-run its definition"],
             [] ))

let check_can_rewind ?(what = "Tensor.rewind_to_last_closure") () =
  if !construction_depth > 0 then
    raise
    @@ Row.Shape_error
         ( what
           ^ ": called from within [op_asn], mid-construction of a tensor operation -- not \
              supported; rewind after the construction returns or raises",
           [] )

let can_checkpoint () = !construction_depth = 0 && Option.is_none !taint

(** {3 Checkpoints (gh-ocannl-903)}

    A checkpoint snapshots the open frontier: the session's module-level state here and in [Row]
    (persistent values, captured as pointers by [capture_globals]), and the mutable fields of every
    open object -- the shapes and steps of the active update steps, and every live capture reference
    -- each through its sealed owner's exhaustive snapshot. A rewind puts them back and discards
    what the window created. The last closure is the base checkpoint: its frontier is empty. *)

(* Module-level state whose writes count as session mutations, captured as pointers. Every such
   binding of this file is captured here, or classified [kept] or [exempt]
   (test/operations/mutation_serial_scan holds the relationship). *)
type globals = {
  g_state : Row.environment;
  g_active_update_steps : update_step list;
  g_active_constraints : Row.constraint_ list;
  g_active_instances : scheme_instance list;
  g_unused_shapes : Set.M(Int).t;
  g_open_refs : int list;
  g_row_numbered_refs : (int * string) list;
}

let capture_globals () =
  {
    g_state = !state;
    g_active_update_steps = !active_update_steps;
    g_active_constraints = !active_constraints;
    g_active_instances = !active_instances;
    g_unused_shapes = !unused_shapes;
    g_open_refs = !open_refs;
    g_row_numbered_refs = !row_numbered_refs;
  }

let restore_globals
    {
      g_state;
      g_active_update_steps;
      g_active_constraints;
      g_active_instances;
      g_unused_shapes;
      g_open_refs;
      g_row_numbered_refs;
    } =
  state := g_state;
  active_update_steps := g_active_update_steps;
  active_constraints := g_active_constraints;
  active_instances := g_active_instances;
  unused_shapes := g_unused_shapes;
  open_refs := g_open_refs;
  row_numbered_refs := g_row_numbered_refs
[@@mutation_serial.boundary "part of a rewind: puts back the globals a checkpoint captured"]

let globals_differences
    {
      g_state = a1;
      g_active_update_steps = a2;
      g_active_constraints = a3;
      g_active_instances = a7;
      g_unused_shapes = a4;
      g_open_refs = a5;
      g_row_numbered_refs = a6;
    }
    {
      g_state = b1;
      g_active_update_steps = b2;
      g_active_constraints = b3;
      g_active_instances = b7;
      g_unused_shapes = b4;
      g_open_refs = b5;
      g_row_numbered_refs = b6;
    } =
  List.filter_map
    [
      ("Shape.state", phys_equal a1 b1);
      ("Shape.active_update_steps", phys_equal a2 b2);
      ("Shape.active_constraints", phys_equal a3 b3);
      ("Shape.active_instances", phys_equal a7 b7);
      ("Shape.unused_shapes", phys_equal a4 b4);
      ("Shape.open_refs", phys_equal a5 b5);
      ("Shape.row_numbered_refs", phys_equal a6 b6);
    ]
    ~f:(fun (name, same) -> if same then None else Some name)

(* Successful closures and session resets so far: a checkpoint taken before the latest one is no
   longer valid (a closure commits). *)
let closure_count = ref 0
[@@mutation_serial.exempt "a count of commits, which only ends checkpoints: never restored"]

(* The explicit checkpoints still valid, newest first, by sequence number: a rewind to [ck] resets
   it to the list as it stood when [ck] was taken (so [ck] stays valid, and every checkpoint taken
   after it does not); a closure, a reset or a rewind to the last closure empties it. *)
let checkpoint_lineage : int list ref = ref []
[@@mutation_serial.exempt
  "the lineage of valid checkpoints: bookkeeping of checkpoints, not inference state"]

let checkpoint_seq = ref 0
[@@mutation_serial.exempt
  "a fresh-id counter: ids are never reused, so a rewind need not restore it"]

(* Committed-state reads that raised since the last closure, newest first: of an open shape
   ([to_dims], [to_padding]), or of a step's projections ([get_projections] -- a step can be pending
   over a closed shape, as a [%cd] assignment to a closed tensor is). A lazy over such a read (a
   node's dims or padding, a projections handle) memoizes the exception, which no rewind can reset:
   a checkpoint whose frontier holds one of them, read after it was taken, can no longer be rewound
   to. *)
let failed_reads : [ `Shape of t | `Step of update_step ] list ref = ref []
[@@mutation_serial.exempt
  "a record of reads that raised, consulted by a rewind's validity check: never restored"]

let note_failed_read read = failed_reads := read :: !failed_reads

module Checkpoint = struct
  type nonrec t = {
    seq : int;  (** 0 for a closure's checkpoint. *)
    label : string;
    closure : int;  (** [closure_count] when taken. *)
    lineage : int list;  (** [checkpoint_lineage] after taking it. *)
    id_mark : int;  (** The next tensor id when taken. *)
    update_mark : int;  (** [update_uid] when taken. *)
    globals : globals;
    row_globals : Row.globals;
    shapes : Shape_record.snapshot list;
    shape_index : (int, Shape_record.t list) Hashtbl.t Lazy.t;
        (** [shapes] by id (several physical shapes can share one), for [in_frontier]: a rewind asks
            it of every open shape, so a linear search made it quadratic in the frontier. *)
    steps : Update_step_record.snapshot list;
    refs : Delayed_ref.snapshot list;
    ref_uids : Set.M(Int).t;
    failed_reads_mark : [ `Shape of t | `Step of update_step ] list;
  }

  let label ck = ck.label
  let is_closure ck = ck.seq = 0

  (* The open shapes, each once (physically): [dealias_operands]' copies share an id, but are never
     registered. *)
  let open_shapes ?(instances = []) steps =
    let seen = Hashtbl.create (module Int) in
    let acc = ref [] in
    let visit (sh : Shape_record.t) =
      let same = Hashtbl.find_multi seen sh.id in
      if not (List.mem same sh ~equal:phys_equal) then (
        Hashtbl.add_multi seen ~key:sh.id ~data:sh;
        acc := sh :: !acc)
    in
    List.iter steps ~f:(fun step -> iter_shapes step ~f:visit);
    (* gh-ocannl-404: an applied scheme's boundary shapes are open objects no step may hold. *)
    List.iter instances ~f:(fun inst -> List.iter inst.inst_shapes ~f:visit);
    List.rev !acc

  let take ~closure ~label ~id_mark =
    let seq =
      if closure then 0
      else (
        Int.incr checkpoint_seq;
        checkpoint_lineage := !checkpoint_seq :: !checkpoint_lineage;
        !checkpoint_seq)
    in
    let steps = !active_update_steps in
    let shapes =
      List.map (open_shapes ~instances:!active_instances steps) ~f:Shape_record.snapshot
    in
    (* A closure's checkpoint snapshots no reference: rewinding to the last closure discards every
       reference the discarded work bound or set, as gh-ocannl-1301 specifies. *)
    let refs = if closure then [] else Delayed_ref.snapshot_live () in
    {
      seq;
      label;
      closure = !closure_count;
      lineage = !checkpoint_lineage;
      id_mark;
      update_mark = !update_uid;
      globals = capture_globals ();
      row_globals = Row.capture_globals ();
      shapes;
      shape_index =
        lazy
          (let index = Hashtbl.create (module Int) in
           List.iter shapes ~f:(fun snap ->
               let (sh : Shape_record.t) = Shape_record.snapshot_shape snap in
               Hashtbl.add_multi index ~key:sh.id ~data:sh);
           index);
      steps = List.map steps ~f:Update_step_record.snapshot;
      refs;
      ref_uids = Set.of_list (module Int) (List.map refs ~f:Delayed_ref.snapshot_uid);
      failed_reads_mark = !failed_reads;
    }

  let in_frontier ck (sh : Shape_record.t) =
    List.mem (Hashtbl.find_multi (Lazy.force ck.shape_index) sh.id) sh ~equal:phys_equal

  (* Why [ck] can no longer be rewound to, if it cannot. Checked before a rewind writes anything. *)
  let refusal ck =
    if ck.closure <> !closure_count then
      Some
        "inference closed (a finish_inference committed the open work) after it was taken; a \
         rewind cannot undo a closure"
    else if (not (is_closure ck)) && not (List.mem !checkpoint_lineage ck.seq ~equal:Int.equal) then
      Some
        "it was abandoned: a rewind to an earlier checkpoint, Tensor.rewind_to_last_closure or \
         Tensor.unsafe_reinitialize ran since"
    else
      let rec since acc = function
        | l when phys_equal l ck.failed_reads_mark -> acc
        | [] -> acc
        | sh :: rest -> since (sh :: acc) rest
      in
      let poisoned =
        List.filter_map (since [] !failed_reads) ~f:(function
          | `Shape sh -> if in_frontier ck sh then Some ("shape", sh) else None
          | `Step step ->
              if List.exists ck.steps ~f:(Update_step_record.snapshot_is step) then
                Some ("projections", step.shape)
              else None)
      in
      match poisoned with
      | (what, sh) :: _ when not (is_closure ck) ->
          Some
            [%string
              "a read of the committed %{what} of #%{sh.id#Int} %{sh.debug_name}, open at the \
               checkpoint, raised after it (forcing its dims, padding or projections while \
               inference could not close); the lazy that read it keeps the exception, and no \
               rewind can reset it -- rewind to the last closure instead"]
      | _ -> (
          (* Every shape a step of the window holds is the window's own or was in the frontier: each
             tensor's shape is registered by its own step, and shapes are written only through the
             steps that hold them. A shape older than the checkpoint that is neither would have been
             written without a snapshot to restore. *)
          let stray =
            List.find_map !active_update_steps ~f:(fun step ->
                let found = ref None in
                iter_shapes step ~f:(fun sh ->
                    if
                      Option.is_none !found && (not sh.closed) && sh.id < ck.id_mark
                      && not (in_frontier ck sh)
                    then found := Some sh);
                !found)
          in
          match stray with
          | Some sh when not (is_closure ck) ->
              Some
                [%string
                  "the open shape of #%{sh.id#Int} %{sh.debug_name} predates the checkpoint but \
                   was not in its frontier (no registered update step held it) -- please report \
                   this as a bug"]
          | _ -> None)

  (* Restore categories a test may skip one at a time (negative controls, [Tensor.For_testing]). *)
  let categories = [ "shapes"; "steps"; "refs"; "shape globals"; "row globals"; "stale marking" ]

  let rewind ?omit ck ~provenance =
    let skip c = Option.exists omit ~f:(String.equal c) in
    if not (skip "stale marking") then (
      (* The window's own shapes -- in some active step, open, not in the frontier -- become stale
         handles, like its tensors. *)
      List.iter (open_shapes ~instances:!active_instances !active_update_steps) ~f:(fun sh ->
          if not (in_frontier ck sh) then mark_discarded sh ~provenance);
      (* The steps the window made, registered or not, and the references it bound or set that the
         checkpoint holds no snapshot of: made in the window (or, for the closure's checkpoint,
         every one touched since). Each is marked on the object itself (gh-ocannl-1789): a rewind
         keeps no record of its own, so a long session's rewinds accumulate nothing beyond what its
         caller still holds. *)
      if !update_uid > ck.update_mark then
        Update_step_record.discard_since ~update_mark:ck.update_mark ~provenance;
      Delayed_ref.discard ~uids:!open_refs ~keep:ck.ref_uids ~provenance);
    if not (skip "shapes") then List.iter ck.shapes ~f:Shape_record.restore;
    if not (skip "steps") then List.iter ck.steps ~f:Update_step_record.restore;
    if not (skip "refs") then List.iter ck.refs ~f:Delayed_ref.restore;
    if not (skip "shape globals") then restore_globals ck.globals;
    if not (skip "row globals") then Row.restore_globals ck.row_globals;
    Row.reset_transient ();
    checkpoint_lineage := ck.lineage;
    taint := None

  (* What differs between the session now and [ck], by component: a diagnostic. *)
  let differences ck =
    globals_differences ck.globals (capture_globals ())
    @ Row.globals_differences ck.row_globals (Row.capture_globals ())
    @ List.concat_map ck.shapes ~f:Shape_record.snapshot_differences
    @ List.filter_map ck.steps ~f:(fun s ->
        if Update_step_record.snapshot_changed s then Some "an update step's projections" else None)
    @ List.filter_map ck.refs ~f:(fun s ->
        if Delayed_ref.snapshot_changed s then
          Some [%string "capture reference uid %{Delayed_ref.snapshot_uid s#Int}"]
        else None)

  let sizes ck =
    [
      ("active update steps", List.length ck.globals.g_active_update_steps);
      ("active constraints", List.length ck.globals.g_active_constraints);
      ("open shapes", List.length ck.shapes);
      ("open references", List.length ck.globals.g_open_refs);
    ]
end

let checkpoint ~label ~id_mark = Checkpoint.take ~closure:false ~label ~id_mark
let closure_checkpoint ~id_mark = Checkpoint.take ~closure:true ~label:"the last closure" ~id_mark

let discarded_registry_sizes () =
  [
    ("discarded capture references", Delayed_ref.count_discarded ());
    ("discarded update steps", Update_step_record.count_discarded ());
  ]

let infer_equal (sh1 : t) (sh2 : t) =
  unmark_unused sh1.id;
  unmark_unused sh2.id;
  let get_origin kind =
    [
      Row.
        {
          lhs_name = sh1.debug_name;
          lhs_kind = kind;
          rhs_name = sh2.debug_name;
          rhs_kind = kind;
          operation = Some "shape_equals";
        };
    ]
  in
  active_constraints :=
    Row.Row_eq { r1 = sh1.batch; r2 = sh2.batch; origin = get_origin `Batch }
    :: Row.Row_eq { r1 = sh1.input; r2 = sh2.input; origin = get_origin `Input }
    :: Row.Row_eq { r1 = sh1.output; r2 = sh2.output; origin = get_origin `Output }
    :: !active_constraints

(** The ties of [product_space_shape]'s fresh re-elaboration to the operation's captures
    (gh-ocannl-1691). The operand equalities tie every label an operand row carries to the
    operation's own solution, but a label only an affine axis names -- the kernel of a one-operand
    window, [x @^^ "s, 2*o=+1+k => s, o" ["k"]] -- has no operand axis, so without a tie its fresh
    variable stays free and closes to 1, and the proxy disagrees with the projections at lowering.
    Each captured label's fresh variable is equated to the variable the operation bound the
    reference to (which a later [Shape.set_dim] constrains) and to any value it already holds.
    Unlike {!bind_delayed_vars_to_envs}, the reference is NOT rebound: the proxy is a derived view
    of the operation, not a new capture. A binding from before the last closure names a dead
    variable, so only the held value is used then. *)
let capture_ties_for_proxy ~spec ~dim_var_env ~row_var_env (dim_refs : delayed_var_ref list) :
    Row.constraint_ list =
  List.concat_map dim_refs ~f:(fun delayed_ref ->
      let label = delayed_ref.var_ref.ref_label in
      let origin =
        [
          {
            Row.lhs_name = label;
            lhs_kind = `Output;
            rhs_name = spec;
            rhs_kind = `Output;
            operation = Some "product-space capture tie";
          };
        ]
      in
      let binding = if bound_before_closure delayed_ref then `Not_set_yet else delayed_ref.var in
      let row_of var = Row.get_row_for_var Row.empty_provenance var in
      match (Hashtbl.find dim_var_env label, Hashtbl.find row_var_env label) with
      | Some var, _ ->
          let held =
            match (delayed_ref.var_ref.solved_sym, delayed_ref.var_ref.solved_dim) with
            | Some sym, _ -> [ Row.Dim_eq { d1 = Row.Var var; d2 = Row.get_sym_dim sym; origin } ]
            | None, Some d ->
                [
                  Row.Dim_eq
                    {
                      d1 = Row.Var var;
                      d2 = Row.get_dim ~d ~basis:delayed_ref.solved_basis ();
                      origin;
                    };
                ]
            | None, None -> []
          in
          let tie =
            match binding with
            | `Dim v0 -> [ Row.Dim_eq { d1 = Row.Var var; d2 = Row.Var v0; origin } ]
            | `Row _ | `Not_set_yet -> []
          in
          held @ tie
      | None, Some var -> (
          match binding with
          | `Row r0 -> [ Row.Row_eq { r1 = row_of var; r2 = row_of r0; origin } ]
          | `Dim _ | `Not_set_yet -> [])
      | None, None -> [])

(** The product-space proxy shape of an einsum-family operation (gh-512): the result's axes followed
    by the reduced-over (contracted) axes, matching the product-space order of [derive_projections]
    — result rows in [to_dims] order, then contracted axes by first occurrence over the RHS slots
    (each walked batch, output, input; [beg_dims], row variable, [dims]). Built by re-elaborating
    the einsum spec with fresh variables — the re-emitted operand equalities tie them to the
    operation's own solution — and appending the contracted axes to the result's input row (the last
    row in [to_dims] order). A [%cd] [*_pspace] intermediate's shape is [infer_equal]-ed to this
    proxy; the identity projection over the product space is derived from the projections by
    [Ir.Indexing.prod_project_for], which cross-checks the resulting dimensions against the product
    space at lowering.

    Raises [Row.Shape_error] when the product space is not expressible as a shape: a non-einsum
    operation, more than one reduced-over row variable, or a reduced-over row variable combined with
    an open result input row. *)
let product_space_shape ~(shape : t) ~(logic : logic) : t =
  let cur_sh = shape in
  let invalid msg error_shapes =
    raise @@ Row.Shape_error ("product_space_shape: " ^ msg, [ Shape_mismatch error_shapes ])
  in
  let spec, dim_refs, ls_rhs_list, ls_lhs, rhs_shapes =
    match logic with
    | Broadcast (Einsum (spec, dim_refs), sh1, sh2) -> (
        match einsum_of_spec spec with
        | [ ls1; ls2 ], ls_lhs -> (spec, dim_refs, [ ls1; ls2 ], ls_lhs, [ sh1; sh2 ])
        | _ -> invalid ("invalid binary einsum spec: " ^ spec) [ cur_sh; sh1; sh2 ])
    | Transpose (Permute (spec, dim_refs), sh1) -> (
        match einsum_of_spec spec with
        | [ ls1 ], ls_lhs -> (spec, dim_refs, [ ls1 ], ls_lhs, [ sh1 ])
        | _ -> invalid ("invalid permutation spec: " ^ spec) [ cur_sh; sh1 ])
    | Broadcast_tern (Einsum_tern (spec, dim_refs), sh1, sh2, sh3) -> (
        match einsum_of_spec spec with
        | [ ls1; ls2; ls3 ], ls_lhs -> (spec, dim_refs, [ ls1; ls2; ls3 ], ls_lhs, [ sh1; sh2; sh3 ])
        | _ -> invalid ("invalid ternary einsum spec: " ^ spec) [ cur_sh; sh1; sh2; sh3 ])
    | _ ->
        invalid
          "product-space intermediates (*_pspace) require an einsum-family operation (an einsum or \
           permutation spec)"
          [ cur_sh ]
  in
  let error_shapes = cur_sh :: rhs_shapes in
  let cannot msg =
    invalid ("cannot express the product space of " ^ spec ^ " as a shape: " ^ msg) error_shapes
  in
  (* Re-elaborate the spec with fresh (name-shared) variables and re-emit the operand equalities:
     the fresh variables converge to the operation's own solution. A label no operand row carries (a
     one-operand window's captured kernel, gh-ocannl-1691) is reached by no equality; the capture
     ties below tie it to the operation's own binding, which [Shape.set_dim] constrains. *)
  let row_var_env = Hashtbl.create (module String) in
  let dim_var_env = Hashtbl.create (module String) in
  let constraints = ref [] in
  let elaborate ~label ls (sh : t) =
    let extras, _proj_env, ((b, i, o) as bio) =
      einsum_slot_spec_to_dims_bio ~original_spec:spec ~sh_id:sh.id ~row_var_env ~dim_var_env ls
    in
    let row_eq kind r1 r2 =
      Row.Row_eq
        {
          r1;
          r2;
          origin =
            [
              {
                Row.lhs_name = spec;
                lhs_kind = kind;
                rhs_name = sh.debug_name;
                rhs_kind = kind;
                operation = Some label;
              };
            ];
        }
    in
    constraints :=
      extras
      @ [ row_eq `Batch b sh.batch; row_eq `Input i sh.input; row_eq `Output o sh.output ]
      @ !constraints;
    bio
  in
  List.iteri (List.zip_exn ls_rhs_list rhs_shapes) ~f:(fun idx (ls, sh) ->
      ignore (elaborate ~label:(Printf.sprintf "Product-space ARGUMENT %d" (idx + 1)) ls sh));
  ignore (elaborate ~label:"Product-space RESULT" ls_lhs cur_sh);
  constraints := capture_ties_for_proxy ~spec ~dim_var_env ~row_var_env dim_refs @ !constraints;
  active_constraints := !constraints @ !active_constraints;
  (* The proxy's result rows: a SEPARATE, constraint-free elaboration (shared variable environments,
     so named variables coincide with the constrained ones) in which fixed-index axes are pinned to
     dimension 1 ([`Unit_dim]) — their projections are pinned, so they are never product axes
     regardless of their extent. The constrained elaboration above must not use [`Unit_dim]: it
     would pin the operation's actual fixed-index result axes. *)
  let _, _, (b_lhs, i_lhs, o_lhs) =
    einsum_slot_spec_to_dims_bio ~fixed_index:`Unit_dim ~original_spec:spec ~sh_id:cur_sh.id
      ~row_var_env ~dim_var_env ls_lhs
  in
  (* Contracted axes, from the parsed spec: labels (including convolution constituents and
     concatenation elements) and row variables of the RHS slots absent from the result, in
     first-occurrence order with each slot walked in the to_dims kind order (batch, output, input;
     leading axes, then the row variable, then trailing axes) — the order in which
     [derive_projections] first encounters them. Fixed-index axes are never product axes. *)
  let items_of_slot (ls : parsed_axis_labels) =
    let axis_items = function
      | Label l -> [ `Lbl l ]
      | Fixed_index _ -> []
      | Affine_spec { over_label; conv; _ } ->
          `Lbl over_label
          :: (match conv with Some { kernel_label; _ } -> [ `Lbl kernel_label ] | None -> [])
      | Concat_spec cs -> [ `Cnc cs ]
    in
    let row_items beg_specs bcast specs =
      List.concat_map beg_specs ~f:axis_items
      @ (match bcast with Some name -> [ `Rv name ] | None -> [])
      @ List.concat_map specs ~f:axis_items
    in
    row_items ls.given_beg_batch ls.bcast_batch ls.given_batch
    @ row_items ls.given_beg_output ls.bcast_output ls.given_output
    @ row_items ls.given_beg_input ls.bcast_input ls.given_input
  in
  let lhs_items = items_of_slot ls_lhs in
  let lhs_labels =
    Set.of_list
      (module String)
      (List.concat_map lhs_items ~f:(function
        | `Lbl l -> [ l ]
        | `Cnc cs ->
            List.filter_map cs ~f:(function Concat_label l -> Some l | Concat_size _ -> None)
        | `Rv _ -> []))
  in
  let lhs_rvars =
    Set.of_list
      (module String)
      (List.filter_map lhs_items ~f:(function `Rv name -> Some name | _ -> None))
  in
  let lookup_var name =
    Row.Var (Hashtbl.find_or_add dim_var_env name ~default:(fun () -> Row.get_var ~name ()))
  in
  let seen_labels = ref (Set.empty (module String)) in
  let seen_rvars = ref (Set.empty (module String)) in
  let contracted =
    List.concat_map ls_rhs_list ~f:(fun ls ->
        List.concat_map (items_of_slot ls) ~f:(function
          | `Lbl l ->
              if Set.mem lhs_labels l || Set.mem !seen_labels l then []
              else (
                seen_labels := Set.add !seen_labels l;
                [ `Dim (lookup_var l) ])
          | `Cnc cs ->
              (* A sized component ([3^o]) is named by no other slot, so it counts as unknown: [3^o
                 => o] is an extraction, which has no product-space shape here, like [p^o => o]. *)
              let is_known = function
                | Concat_label l -> Set.mem lhs_labels l || Set.mem !seen_labels l
                | Concat_size _ -> false
              in
              if List.for_all cs ~f:is_known then []
              else if List.exists cs ~f:is_known then
                cannot
                  ("a concatenation axis is partially reduced over: "
                  ^ String.concat ~sep:"^" (List.map cs ~f:concat_component_text))
              else (
                seen_labels :=
                  List.fold cs ~init:!seen_labels ~f:(fun acc -> function
                    | Concat_label l -> Set.add acc l
                    | Concat_size _ -> acc);
                [
                  `Dim
                    (Row.Concat
                       (List.map cs ~f:(function
                         | Concat_label l -> lookup_var l
                         | Concat_size n -> Row.get_default_dim ~d:n ())));
                ])
          | `Rv name ->
              if Set.mem lhs_rvars name || Set.mem !seen_rvars name then []
              else (
                seen_rvars := Set.add !seen_rvars name;
                let v =
                  Hashtbl.find_or_add row_var_env name ~default:(fun () -> Row.get_row_var ())
                in
                [ `Rvar v ])))
  in
  let contracted_dims_around idx =
    let dim_of = function `Dim d -> Some d | `Rvar _ -> None in
    ( List.filter_map (List.take contracted idx) ~f:dim_of,
      List.filter_map (List.drop contracted (idx + 1)) ~f:dim_of )
  in
  let input_row, output_row =
    match
      List.findi contracted ~f:(fun _ item -> match item with `Rvar _ -> true | _ -> false)
    with
    | None ->
        let extra = List.map contracted ~f:(function `Dim d -> d | `Rvar _ -> assert false) in
        (* Appending to [dims] places the contracted axes after an open row variable's expansion, so
           an open result input row needs no special case here. *)
        ({ i_lhs with dims = i_lhs.dims @ extra }, o_lhs)
    | Some (idx, `Rvar v) -> (
        if
          List.exists
            (List.drop contracted (idx + 1))
            ~f:(function `Rvar _ -> true | `Dim _ -> false)
        then
          cannot "the operation reduces over more than one row variable (unknown-rank axis groups)"
        else
          let before, after = contracted_dims_around idx in
          match (i_lhs.bcast, o_lhs.bcast) with
          | Row.Broadcastable, _ ->
              ( {
                  Row.beg_dims = i_lhs.beg_dims @ i_lhs.dims @ before;
                  bcast = Row.Row_var v;
                  dims = after;
                  prov = i_lhs.prov;
                },
                o_lhs )
          | Row.Row_var _, Row.Broadcastable when List.is_empty before && List.is_empty after ->
              (* Lone reduced-over row variable with an open result input row: place it in the
                 (closed) output row. The product order puts the result's input axes before the
                 reduced axes while this layout puts them after; a consistent layout is still
                 correct — every access goes through the same identity projection
                 ([Indexing.prod_project_for]), which pairs axes with product components by extent
                 (first-fit), and rejects a leftover on either side at lowering. *)
              ( i_lhs,
                {
                  Row.beg_dims = o_lhs.beg_dims @ o_lhs.dims;
                  bcast = Row.Row_var v;
                  dims = [];
                  prov = o_lhs.prov;
                } )
          | Row.Row_var _, _ ->
              cannot
                "the operation reduces over a row variable while the result's input row is open \
                 (and the reduced axes cannot go into the result's output row instead)")
    | Some (_, `Dim _) -> assert false
  in
  of_rows ~batch:b_lhs ~output:output_row ~input:input_row ~id:cur_sh.id
    ~debug_name:("pspace:" ^ cur_sh.debug_name)

(** Sets the dimension/total-elements for a delayed variable reference. For row variables, this
    creates a [Total_elems] constraint that will be reconciled during [finish_inference]. *)
let%track7_sexp set_dim (delayed_var_ref : delayed_var_ref) (dim : int) : unit =
  check_ref_live delayed_var_ref;
  refuse_external_in_trace ~operation:"Shape.set_dim" delayed_var_ref;
  refuse_closed_unsolved_ref ~operation:"Shape.set_dim" delayed_var_ref;
  match delayed_var_ref with
  | { var_ref = { solved_sym = Some _; ref_label; _ }; _ } ->
      raise
      @@ Row.Shape_error
           ( "Cannot set a concrete dimension for variable reference with label " ^ ref_label
             ^ ": already set to a symbolic dimension",
             [] )
  | { var_ref = { solved_dim = Some dim2; _ }; _ } when dim2 = dim -> ()
  | { var_ref = { solved_dim = Some dim2; ref_label; _ }; _ } ->
      raise
      @@ Row.Shape_error
           ( "Cannot set dimension for variable reference with label " ^ ref_label,
             [ Row.Dim_mismatch [ Row.get_default_dim ~d:dim2 (); Row.get_default_dim ~d:dim () ] ]
           )
  | { var_ref = { solved_dim = None; _ }; var = `Not_set_yet; _ } ->
      Delayed_ref.set_solved_dim delayed_var_ref dim
  | { var_ref = { solved_dim = None; _ }; var = `Dim dim_var; _ } ->
      Delayed_ref.set_solved_dim delayed_var_ref dim;
      active_constraints :=
        Row.Dim_eq
          {
            d1 = Row.Var dim_var;
            d2 = Row.get_default_dim ~d:dim ();
            origin =
              [
                {
                  lhs_name = delayed_var_ref.var_ref.ref_label;
                  lhs_kind = `Output;
                  rhs_name = Int.to_string dim;
                  rhs_kind = `Output;
                  operation = Some "Shape.set_dim Dim";
                };
              ];
          }
        :: !active_constraints
  | { var_ref = { solved_dim = None; _ }; var = `Row row_var; _ } ->
      Delayed_ref.set_solved_dim delayed_var_ref dim;
      active_constraints :=
        Row.Rows_constr
          {
            (* TODO: actually, the Row.provenance should be the one of the shape that the row
               variable is in, should be stored in `Row and in env_row_var. *)
            r = [ Row.get_row_for_var Row.empty_provenance row_var ];
            constr = Total_elems { numerator = Num_elems dim; divided_by = []; keep_axis = false };
            origin =
              [
                {
                  lhs_name = delayed_var_ref.var_ref.ref_label;
                  lhs_kind = `Output;
                  rhs_name = Int.to_string dim;
                  rhs_kind = `Output;
                  operation = Some "Shape.set_dim Row";
                };
              ];
          }
        :: !active_constraints

(** Sets a delayed variable reference to a symbolic dimension (gh-490): the referenced dim variable
    is equated with the rigid symbolic extent [Sym sym]. Mirrors {!set_dim}; row variables are not
    supported (a symbolic total-elements constraint would require symbolic arithmetic). *)
let%track7_sexp set_sym_dim (delayed_var_ref : delayed_var_ref) (sym : Ir.Indexing.static_symbol) :
    unit =
  check_ref_live delayed_var_ref;
  refuse_external_in_trace ~operation:"Shape.set_sym_dim" delayed_var_ref;
  refuse_closed_unsolved_ref ~operation:"Shape.set_sym_dim" delayed_var_ref;
  (* Eagerly validate the declared range (raises [Utils.User_error] if absent or non-positive). *)
  let sym_dim = Row.get_sym_dim sym in
  let origin =
    [
      {
        Row.lhs_name = delayed_var_ref.var_ref.ref_label;
        lhs_kind = `Output;
        rhs_name = Ir.Indexing.symbol_ident sym.Ir.Indexing.static_symbol;
        rhs_kind = `Output;
        operation = Some "Shape.set_sym_dim";
      };
    ]
  in
  match delayed_var_ref with
  | { var_ref = { solved_sym = Some sym2; _ }; _ } when Ir.Indexing.equal_static_symbol sym2 sym ->
      ()
  | { var_ref = { solved_sym = Some sym2; ref_label; _ }; _ } ->
      raise
      @@ Row.Shape_error
           ( "Cannot set symbolic dimension for variable reference with label " ^ ref_label
             ^ ": already set to a different symbolic dimension",
             [ Row.Dim_mismatch [ Row.get_sym_dim sym2; sym_dim ] ] )
  | { var_ref = { solved_dim = Some dim2; ref_label; _ }; _ } ->
      raise
      @@ Row.Shape_error
           ( "Cannot set symbolic dimension for variable reference with label " ^ ref_label
             ^ ": already set to a concrete dimension",
             [ Row.Dim_mismatch [ Row.get_default_dim ~d:dim2 (); sym_dim ] ] )
  | { var_ref = { solved_dim = None; _ }; var = `Not_set_yet; _ } ->
      Delayed_ref.set_solved_sym delayed_var_ref sym
  | { var_ref = { solved_dim = None; _ }; var = `Dim dim_var; _ } ->
      Delayed_ref.set_solved_sym delayed_var_ref sym;
      active_constraints :=
        Row.Dim_eq { d1 = Row.Var dim_var; d2 = sym_dim; origin } :: !active_constraints
  | { var_ref = { solved_dim = None; ref_label; _ }; var = `Row _; _ } ->
      raise
      @@ Row.Shape_error
           ( "Cannot set symbolic dimension for variable reference with label " ^ ref_label
             ^ ": it is bound to a row variable (set_sym_dim does not support row variables)",
             [] )

(* The row a row reference names, for a Row-Row [set_equal]. One bound before the last closure (or
   session reset) no longer names it by [var], a variable of the reset environment: it names the row
   its closure carried ([carried_row], gh-ocannl-1689). Its product is not a substitute
   (gh-ocannl-1617), so a closed reference that carries no row is refused. Binding a reference in
   the open work records it in [open_refs] ({!touch_ref}); a closure empties that list. *)
let row_for_set_equal ~other (delayed_ref : delayed_var_ref) =
  match delayed_ref.var with
  | `Row row_var when not (bound_before_closure delayed_ref) ->
      Row.get_row_for_var Row.empty_provenance row_var
  | _ -> (
      match delayed_ref.carried_row with
      | Some dims -> { Row.beg_dims = []; dims; bcast = Broadcastable; prov = Row.empty_provenance }
      | None ->
          let label = delayed_ref.var_ref.ref_label in
          let held =
            match delayed_ref.var_ref.solved_dim with
            | Some product -> [%string ", which keeps only its row's product %{product#Int}"]
            | None -> ""
          in
          raise
          @@ Row.Shape_error
               ( [%string
                   "Shape.set_equal: the row variable reference %{label} was bound by shape \
                    inference that has since closed%{held}; equating it with the row reference \
                    %{other} needs the row itself, which the closure could not carry (an axis of \
                    it is neither a concrete nor a symbolic extent). Call set_equal before the \
                    closure, or capture the row afresh in an operation of the open work."],
                 [] ))

let set_equal delayed_ref1 delayed_ref2 =
  check_ref_live delayed_ref1;
  check_ref_live delayed_ref2;
  refuse_external_in_trace ~operation:"Shape.set_equal" delayed_ref1;
  refuse_external_in_trace ~operation:"Shape.set_equal" delayed_ref2;
  (* Before the solved-value dispatch: its arms forward to [set_dim], which would otherwise reach a
     dead variable first (gh-ocannl-1655). A Row-Row pair relates the rows, which a closure carries
     even without a product. *)
  let as_row =
    match (delayed_ref1.var, delayed_ref2.var) with `Row _, `Row _ -> true | _ -> false
  in
  refuse_closed_unsolved_ref ~as_row ~operation:"Shape.set_equal" delayed_ref1;
  refuse_closed_unsolved_ref ~as_row ~operation:"Shape.set_equal" delayed_ref2;
  (* An unbound reference paired with a row reference is a number here; see [row_numbered_refs]. *)
  let note_numbered () =
    match (delayed_ref1.var, delayed_ref2.var) with
    | `Not_set_yet, `Row _ ->
        note_row_numbered delayed_ref1.uid ~partner:delayed_ref2.var_ref.ref_label
    | `Row _, `Not_set_yet ->
        note_row_numbered delayed_ref2.uid ~partner:delayed_ref1.var_ref.ref_label
    | _ -> ()
  in
  (* TODO: use provenance from the row variables once we have it there. *)
  let row_row () =
    (* Two row references are equal rows, whatever either one's solved state (gh-ocannl-1617): a row
       reference's [solved_dim] holds only its product, so deciding on it first tied the rows by
       product, or compared two products, depending on what inference had solved by the time of the
       call. A solution a reference holds already has its own constraint on [var] ([set_dim] pushed
       it, or the solver reached it), so nothing else is needed here. A closed reference contributes
       the row its closure carried. With both closed the two rows hold no variable, so the verdict
       is available now, and it is the solver's: the closed-rows arm of [Row.unify_row] writes
       nothing, and raising here leaves the session as it was (gh-ocannl-1969). Otherwise the
       mismatch would surface only at the next closure, far from this call. *)
    let label1 = delayed_ref1.var_ref.ref_label and label2 = delayed_ref2.var_ref.ref_label in
    let r1 = row_for_set_equal ~other:label2 delayed_ref1 in
    let r2 = row_for_set_equal ~other:label1 delayed_ref2 in
    let equation =
      Row.Row_eq
        {
          r1;
          r2;
          origin =
            [
              {
                lhs_name = label1;
                lhs_kind = `Output;
                rhs_name = label2;
                rhs_kind = `Output;
                operation = Some "Shape.set_equal Row-Row";
              };
            ];
        }
    in
    (if bound_before_closure delayed_ref1 && bound_before_closure delayed_ref2 then
       try
         ignore
           (Row.solve_inequalities ~stage:Row.Stage1 [ equation ] Row.empty_env
             : Row.constraint_ list * Row.environment)
       with Row.Shape_error (msg, trace) ->
         raise
         @@ Row.Shape_error
              ( [%string
                  "Shape.set_equal: the closed row references %{label1} and %{label2} are \
                   different rows: %{msg}"],
                trace ));
    active_constraints := equation :: !active_constraints
  in
  match (delayed_ref1, delayed_ref2) with
  | { var = `Row _; _ }, { var = `Row _; _ } -> row_row ()
  | { var_ref = { solved_sym = Some sym1; _ }; _ }, { var_ref = { solved_sym = Some sym2; _ }; _ }
    ->
      if Ir.Indexing.equal_static_symbol sym1 sym2 then ()
      else
        raise
        @@ Row.Shape_error
             ( "Cannot set equal dimensions for variable references bound to different symbolic \
                dimensions",
               [ Row.Dim_mismatch [ Row.get_sym_dim sym1; Row.get_sym_dim sym2 ] ] )
  | { var_ref = { solved_sym = Some sym; _ }; _ }, delayed_ref2 ->
      (* First is symbolic, second is not - set the second to match the first (raises if the second
         is already solved to a concrete dimension). *)
      set_sym_dim delayed_ref2 sym
  | delayed_ref1, { var_ref = { solved_sym = Some sym; _ }; _ } -> set_sym_dim delayed_ref1 sym
  | { var_ref = { solved_dim = Some dim1; _ }; _ }, { var_ref = { solved_dim = Some dim2; _ }; _ }
    ->
      if dim1 = dim2 then note_numbered ()
      else
        raise
        @@ Row.Shape_error
             ( "Cannot set equal dimensions for variable references with different values",
               [
                 Row.Dim_mismatch [ Row.get_default_dim ~d:dim1 (); Row.get_default_dim ~d:dim2 () ];
               ] )
  | { var_ref = { solved_dim = Some dim; _ }; _ }, delayed_ref2 ->
      (* First is solved, second is not - set the second to match the first *)
      set_dim delayed_ref2 dim;
      note_numbered ()
  | delayed_ref1, { var_ref = { solved_dim = Some dim; _ }; _ } ->
      (* Second is solved, first is not - set the first to match the second *)
      set_dim delayed_ref1 dim;
      note_numbered ()
  | ( { var_ref = { solved_dim = None; ref_label = ref_label1; _ }; var = _; _ },
      { var_ref = { solved_dim = None; ref_label = ref_label2; _ }; var = `Not_set_yet; _ } )
  | ( { var_ref = { solved_dim = None; ref_label = ref_label1; _ }; var = `Not_set_yet; _ },
      { var_ref = { solved_dim = None; ref_label = ref_label2; _ }; var = _; _ } ) ->
      raise
      @@ Row.Shape_error
           ( "set_equal: insufficient information between labels " ^ ref_label1 ^ " and "
             ^ ref_label2,
             [] )
  | { var = `Dim dim_var1; _ }, { var = `Dim dim_var2; _ } ->
      (* Both are dimension variables - create equality constraint *)
      active_constraints :=
        Row.Dim_eq
          {
            d1 = Row.Var dim_var1;
            d2 = Row.Var dim_var2;
            origin =
              [
                {
                  lhs_name = delayed_ref1.var_ref.ref_label;
                  lhs_kind = `Output;
                  rhs_name = delayed_ref2.var_ref.ref_label;
                  rhs_kind = `Output;
                  operation = Some "Shape.set_equal Dim-Dim";
                };
              ];
          }
        :: !active_constraints
  | { var = `Dim dim_var; _ }, { var = `Row row_var; _ }
  | { var = `Row row_var; _ }, { var = `Dim dim_var; _ } ->
      (* One is dim var, one is row var - equality via Total_elems constraint *)
      active_constraints :=
        product_tie row_var dim_var
          ~origin:
            [
              {
                lhs_name = delayed_ref1.var_ref.ref_label;
                lhs_kind = `Output;
                rhs_name = delayed_ref2.var_ref.ref_label;
                rhs_kind = `Output;
                operation = Some "Shape.set_equal Dim-Row or Row-Dim";
              };
            ]
        :: !active_constraints

let set_scale ~factor delayed_ref_large delayed_ref_small =
  check_ref_live delayed_ref_large;
  check_ref_live delayed_ref_small;
  refuse_external_in_trace ~operation:"Shape.set_scale" delayed_ref_large;
  refuse_external_in_trace ~operation:"Shape.set_scale" delayed_ref_small;
  refuse_closed_unsolved_ref ~operation:"Shape.set_scale" delayed_ref_large;
  refuse_closed_unsolved_ref ~operation:"Shape.set_scale" delayed_ref_small;
  if factor < 1 then
    raise
    @@ Row.Shape_error ("Shape.set_scale: factor must be >= 1, got " ^ Int.to_string factor, []);
  (* Row-variable guard: reject any ref whose var is `Row _, regardless of which side is solved.
     Without this, the solved-side branches below would forward to set_dim on a Row ref, which
     silently emits a Rows_constr Total_elems constraint -- out of scope for this API. *)
  (match (delayed_ref_large.var, delayed_ref_small.var) with
  | `Row _, _ | _, `Row _ ->
      raise
      @@ Row.Shape_error
           ( "Shape.set_scale: requires two dimension variables; got at least one row variable \
              (labels " ^ delayed_ref_large.var_ref.ref_label ^ ", "
             ^ delayed_ref_small.var_ref.ref_label ^ ")",
             [] )
  | _ -> ());
  (* gh-490 v1: scaling relations over symbolic dimensions would require symbolic arithmetic. *)
  if
    Option.is_some delayed_ref_large.var_ref.solved_sym
    || Option.is_some delayed_ref_small.var_ref.solved_sym
  then
    raise
    @@ Row.Shape_error
         ( "Shape.set_scale: symbolic dimensions are not supported (labels "
           ^ delayed_ref_large.var_ref.ref_label ^ ", " ^ delayed_ref_small.var_ref.ref_label ^ ")",
           [] );
  if factor = 1 then set_equal delayed_ref_large delayed_ref_small
  else
    match (delayed_ref_large, delayed_ref_small) with
    | ( { var_ref = { solved_dim = Some d_large; _ }; _ },
        { var_ref = { solved_dim = Some d_small; _ }; _ } ) ->
        if d_large = factor * d_small then ()
        else
          raise
          @@ Row.Shape_error
               ( "Shape.set_scale: dimensions do not satisfy d_large = factor * d_small",
                 [
                   Row.Dim_mismatch
                     [
                       Row.get_default_dim ~d:d_large ();
                       Row.get_default_dim ~d:(factor * d_small) ();
                     ];
                 ] )
    | { var_ref = { solved_dim = Some d_large; ref_label; _ }; _ }, delayed_ref_small ->
        if d_large % factor <> 0 then
          raise
          @@ Row.Shape_error
               ( "Shape.set_scale: solved dimension " ^ Int.to_string d_large ^ " for " ^ ref_label
                 ^ " is not divisible by factor " ^ Int.to_string factor,
                 [] );
        set_dim delayed_ref_small (d_large / factor)
    | delayed_ref_large, { var_ref = { solved_dim = Some d_small; _ }; _ } ->
        set_dim delayed_ref_large (factor * d_small)
    | ( { var_ref = { solved_dim = None; ref_label = label_l; _ }; var = var_l; _ },
        { var_ref = { solved_dim = None; ref_label = label_s; _ }; var = var_s; _ } ) -> (
        match (var_l, var_s) with
        | `Dim dim_var_large, `Dim dim_var_small ->
            active_constraints :=
              Row.Dim_eq
                {
                  d1 = Row.Var dim_var_large;
                  d2 =
                    Row.Affine
                      {
                        stride = factor;
                        over = Row.Var dim_var_small;
                        conv = None;
                        stride_offset = 0;
                      };
                  origin =
                    [
                      {
                        lhs_name = label_l;
                        lhs_kind = `Output;
                        rhs_name = Int.to_string factor ^ "*" ^ label_s;
                        rhs_kind = `Output;
                        operation = Some "Shape.set_scale Dim-Dim";
                      };
                    ];
                }
              :: !active_constraints
        | `Not_set_yet, _ | _, `Not_set_yet ->
            raise
            @@ Row.Shape_error
                 ( "Shape.set_scale: insufficient information between labels " ^ label_l ^ " and "
                   ^ label_s,
                   [] )
        | `Row _, _ | _, `Row _ ->
            (* Defensive: already caught by the top-of-function row guard. *)
            assert false)

let set_terminal ~is_param (sh : t) =
  mark_unused sh.id;
  let get_origin kind =
    Row.
      {
        lhs_name = sh.debug_name;
        lhs_kind = kind;
        rhs_name = "(parameter)";
        rhs_kind = kind;
        operation = Some "set_terminal";
      }
  in
  active_constraints :=
    Row.Terminal_row (is_param, sh.batch, [ get_origin `Batch ])
    :: Row.Terminal_row (is_param, sh.input, [ get_origin `Input ])
    :: Row.Terminal_row (is_param, sh.output, [ get_origin `Output ])
    :: !active_constraints

(* Marks the shape's rows as resolved by their use sites (gh-ocannl-544): their row variables may
   close to the GLB of their use sites' rows at the final solver stages, where operation results
   close to an empty remainder by default. Substitutes through the current environment so that a
   variable already unified away transfers the mark to its representative; later unifications
   propagate it further inside [Row.unify_row]. *)
let set_resolve_at_use (sh : t) =
  Row.note_mutation ();
  let mark row =
    let row = Row.subst_row !state row in
    (match row.bcast with Row.Row_var v -> Row.add_resolve_at_use v | Row.Broadcastable -> ());
    List.iter (row.beg_dims @ row.dims) ~f:(fun d ->
        Set.iter (Row.vars_of_dim d) ~f:Row.add_resolve_at_use_dim)
  in
  mark sh.batch;
  mark sh.input;
  mark sh.output

let unsafe_reinitialize () =
  update_uid := 0;
  state := Row.empty_env;
  active_update_steps := [];
  active_constraints := [];
  active_instances := [];
  tracing := [];
  construction_depth := 0;
  taint := None;
  open_refs := [];
  Update_step_record.forget_made ();
  row_numbered_refs := [];
  Int.incr closure_count;
  checkpoint_lineage := [];
  failed_reads := [];
  Row.reset_transient ();
  Row.drop_symbol_roles ();
  unused_shapes := Set.empty (module Int);
  Row.retire_closed_variables ()
[@@mutation_serial.boundary "the session reset: it discards all state and clears the taint"]

let all_rows_w_origin update_step =
  let get_origin sh kind =
    Row.
      {
        lhs_name = sh.debug_name;
        lhs_kind = kind;
        rhs_name = "";
        rhs_kind = kind;
        operation = Some "remaining rows";
      }
  in
  let rows_sh sh =
    [
      (sh.batch, get_origin sh `Batch);
      (sh.input, get_origin sh `Input);
      (sh.output, get_origin sh `Output);
    ]
  in
  rows_sh update_step.shape
  @
  match update_step.logic with
  | Terminal _ -> []
  | Transpose (_, sh1) -> rows_sh sh1
  | Broadcast (_, sh1, sh2) -> rows_sh sh1 @ rows_sh sh2
  | Broadcast_tern (_, sh1, sh2, sh3) -> rows_sh sh1 @ rows_sh sh2 @ rows_sh sh3
  | Block { rhses; _ } -> List.concat_map rhses ~f:rows_sh

(** {3 Projection inference} *)

let fresh_proj_ids update =
  let resolved_padding = ref [] in
  let inferred_padding = ref [] in
  let no_padding = Ir.Ops.{ left = 0; right = 0 } in
  (* [locked = None]: the tensor node's layout is not committed yet — register the shape's padding
     (if any) as inferred. [locked = Some arr_opt]: the tnode's padding lazy is forced, so the
     buffer layout is committed — register the TNODE's padding (zeros when unpadded) as resolved for
     every axis, so that [Row.solve_proj_equations] rejects later demands exceeding it (a
     forced-at-[None] node used to register nothing, silently losing the halo; see
     [check_and_update_padding]). [offset] is the axis position within the tnode's padding array
     (axis order batch;output;input, cf. {!to_padding}). Returns the offset advanced past this
     row. *)
  let fetch_padding ~locked row row_padding ~offset =
    let dims = Array.of_list (row.Row.beg_dims @ row.Row.dims) in
    (match locked with
    | None ->
        Option.iter row_padding ~f:(fun padding ->
            Array.iter2_exn dims padding ~f:(fun d p ->
                match d with
                | Row.Dim { proj_id = Some proj_id; _ } | Row.Sym { sym_proj_id = Some proj_id; _ }
                  ->
                    inferred_padding := (proj_id, p) :: !inferred_padding
                | _ -> ()))
    | Some locked_arr ->
        Array.iteri dims ~f:(fun i d ->
            let p =
              match locked_arr with
              | Some arr when offset + i < Array.length arr -> arr.(offset + i)
              | _ -> no_padding
            in
            match d with
            | Row.Dim { proj_id = Some proj_id; _ } | Row.Sym { sym_proj_id = Some proj_id; _ } ->
                resolved_padding := (proj_id, p) :: !resolved_padding
            | _ -> ()));
    offset + Array.length dims
  in
  let fresh_shape (sh : t) =
    (* Mint in batch, input, output order (labelled arguments evaluate right to left). *)
    let batch = Row.fresh_row_proj sh.batch in
    let input = Row.fresh_row_proj sh.input in
    let output = Row.fresh_row_proj sh.output in
    set_rows sh ~batch ~input ~output;
    let locked =
      match Ir.Tnode.find ~id:sh.id with
      | Some tn when Lazy.is_val tn.Ir.Tnode.padding -> (
          (* [Lazy.is_val] can be true for a lazy currently being forced (forcing it raises
             [Lazy.Undefined]) — mid-force means this very derivation is part of committing the
             tnode's layout, so the padding is not settled yet: treat as not locked. *)
          try
            let p = Lazy.force tn.Ir.Tnode.padding in
            (* Reconcile the committed neutral element into the shape: nodes created with an
               already-padded layout (e.g. [wrap_padded]) commit their [padded_value] at creation
               while the shape's [padding_elem] starts unknown — without this, a later
               margin-reading consumer with a different neutral would commit its own neutral at the
               shape level unchecked and silently read the buffer's halo values (Codex P1 on PR
               #173). [update_padding_elem] then rejects conflicting consumers. *)
            (match p with
            | Some (_, v) -> (
                match sh.padding_elem with
                | None -> set_padding_elem sh (Some (Some v))
                | Some (Some v1) when Float.( = ) v1 v -> ()
                | Some _ ->
                    raise
                    @@ Row.Shape_error
                         ( [%string
                             "Conflicting padding neutral elements: the tensor's buffer committed \
                              margins holding %{v#Float}, but a different neutral was inferred for \
                              its shape. Materialize a separate copy of the tensor for the \
                              conflicting padded consumers (for a padded max-pool, \
                              Nn_blocks.max_pool2d_copy)"],
                           [ Shape_mismatch [ sh ] ] ))
            | None -> ());
            Some (Option.map ~f:fst p)
          with Lazy.Undefined -> None)
      | _ -> None
    in
    let offset = fetch_padding ~locked sh.batch sh.batch_padding ~offset:0 in
    let offset = fetch_padding ~locked sh.output sh.output_padding ~offset in
    ignore (fetch_padding ~locked sh.input sh.input_padding ~offset : int)
  in
  fresh_shape update.shape;
  (match update.logic with
  | Broadcast (Defined_by_cd_logic, _, _)
  | Transpose (Defined_by_cd_logic, _)
  | Broadcast_tern (Defined_by_cd_logic, _, _, _) ->
      ()
  | Terminal _ -> ()
  | Transpose (_, sh) -> fresh_shape sh
  | Broadcast (_, sh1, sh2) ->
      fresh_shape sh1;
      fresh_shape sh2
  | Broadcast_tern (_, sh1, sh2, sh3) ->
      fresh_shape sh1;
      fresh_shape sh2;
      fresh_shape sh3
  | Block { rhses; _ } -> List.iter rhses ~f:fresh_shape);
  (!resolved_padding, !inferred_padding)

let%debug4_sexp row_to_dims (row : Row.t) : int array =
  let open Row in
  let rec f = function
    | Dim { d; _ } -> d
    | Sym s ->
        (* Reconciliation with Tnode dims: a symbolic axis materializes at its maximum extent (the
           declared range of its static symbol), which sizes allocations and host transfers. *)
        sym_dim_extent_exn s
    | Var v ->
        raise
        @@ Row.Shape_error
             ( "Not enough shape information: unresolved variable "
               ^ Sexp.to_string_hum ([%sexp_of: dim_var] v),
               [ Row_mismatch [ row ] ] )
    | Affine _ ->
        (* FIXME: reconsider this, we could return the input dimension of the convolution. *)
        raise
        @@ Row.Shape_error
             ( "Not enough shape information: affine dimension cannot be converted to single int",
               [ Row_mismatch [ row ] ] )
    | Concat dims -> List.sum (module Int) dims ~f
  in
  match row with
  | { bcast = Row_var v; _ } ->
      raise
      @@ Row.Shape_error
           ( "Not enough shape information: unresolved row variable "
             ^ Sexp.to_string_hum ([%sexp_of: row_var] v),
             [ Row_mismatch [ row ] ] )
  | { beg_dims; dims; bcast = Broadcastable; prov = _ } -> Array.of_list_map (beg_dims @ dims) ~f

let to_dims_impl (sh : t) : int array =
  try Array.concat_map ~f:row_to_dims [| sh.batch; sh.output; sh.input |]
  with Row.Shape_error (s, trace) -> raise @@ Row.Shape_error (s, Shape_mismatch [ sh ] :: trace)

(** One tensor can be several operands of one step -- [(w, w) ++^ "s, c; s, d => s, c^d"],
    [m +* "a, b; b, c => a, c" m] -- and then the step's logic names one shape object several times.
    Projection ids live on the [Dim]s of a shape's rows and [fresh_proj_ids] freshens a shape IN
    PLACE, so every occurrence would end up with the same ids: the solver would equate the axes the
    spec keeps apart (the concatenated segments, the contracted and the free axis of a
    self-product), collapsing them onto one iterator -- a self-product silently computed its
    diagonal, and a self-concatenation lowered one segment twice (gh-ocannl-1476). So every
    occurrence after the first (the result's own shape counts as the first) is replaced by a shallow
    copy: the copy has its own rows to freshen, hence its own projection ids, and keeps [id], so
    tnode-keyed lookups such as committed padding still find the node. Returns the de-aliased step
    and the [(copy, original)] pairs, for {!merge_dealiased_padding}. *)
let dealias_operands (update_step : update_step) : update_step * (t * t) list =
  let seen = ref [ update_step.shape ] and copies = ref [] in
  let occurrence sh =
    if List.mem !seen sh ~equal:phys_equal then (
      let copy = copy sh in
      copies := (copy, sh) :: !copies;
      copy)
    else (
      seen := sh :: !seen;
      sh)
  in
  let logic =
    match update_step.logic with
    | Terminal _ as logic -> logic
    | Transpose (ty, sh) -> Transpose (ty, occurrence sh)
    | Broadcast (ty, sh1, sh2) ->
        let sh1 = occurrence sh1 in
        let sh2 = occurrence sh2 in
        Broadcast (ty, sh1, sh2)
    | Broadcast_tern (ty, sh1, sh2, sh3) ->
        let sh1 = occurrence sh1 in
        let sh2 = occurrence sh2 in
        let sh3 = occurrence sh3 in
        Broadcast_tern (ty, sh1, sh2, sh3)
    | Block { spec; delayed_vars; rhses } ->
        Block { spec; delayed_vars; rhses = List.map rhses ~f:occurrence }
  in
  if List.is_empty !copies then (update_step, [])
  else (Update_step_record.with_logic update_step logic, List.rev !copies)

(** A copy made by {!dealias_operands} stands for its original during one derivation, so the padding
    that derivation infers for the copy belongs to the original: merge it in, axis by axis, taking
    the larger margin -- exactly how one shape's padding accumulates across steps. Both started from
    the original's padding and both commit the step's one neutral element, so their [padding_elem]s
    cannot conflict; the copy's only adds a value the original's occurrence left unknown. *)
let merge_dealiased_padding copies =
  let merge (a : padding) (b : padding) : padding =
    match (a, b) with
    | None, p | p, None -> p
    | Some a, Some b ->
        Some
          (Array.map2_exn a b ~f:(fun a b ->
               Ir.Ops.{ left = max a.left b.left; right = max a.right b.right }))
  in
  List.iter copies ~f:(fun (copy, orig) ->
      let batch = merge orig.batch_padding copy.batch_padding in
      let input = merge orig.input_padding copy.input_padding in
      let output = merge orig.output_padding copy.output_padding in
      (* The closed-shape guard (gh-ocannl-1301): the copy carries [closed] and was guarded itself,
         so on a closed original these merges are equal; the setter checks anyway, before
         writing. *)
      set_paddings orig ~batch ~input ~output;
      if Option.is_none orig.padding_elem then set_padding_elem orig copy.padding_elem)

(** The operand shapes a step's projections index, in [rhs_dims] order, and whether the step derives
    no projections at all (its [%cd] code indexes for itself). *)
let projected_operands = function
  | Broadcast (Defined_by_cd_logic, _, _)
  | Transpose (Defined_by_cd_logic, _)
  | Broadcast_tern (Defined_by_cd_logic, _, _, _) ->
      ([], true)
  | Terminal _ -> ([], false)
  | Transpose (_, sh) -> ([ sh ], false)
  | Broadcast (_, sh1, sh2) -> ([ sh1; sh2 ], false)
  | Broadcast_tern (_, sh1, sh2, sh3) -> ([ sh1; sh2; sh3 ], false)
  | Block { rhses; _ } -> (rhses, false)

(* The Stage1 solve of projection derivation, its capture sizes last (gh-ocannl-2033). A size
   ({!is_projection_sizing}) stands in for what the operation's closed axes leave open, the split of
   a concatenated axis or a padded window's kernel, and makes no basis claim: where the axes already
   fix the variable at that size it is dropped, so the tag they carry stands -- the broadcast top a
   free capture closed to, a named tag an operand gave it. A different size still meets them, and is
   refused there. *)
let solve_with_sizing ~discardable_vars ineqs =
  let sizing, rest = List.partition_tf ineqs ~f:is_projection_sizing in
  let unsolved, env = Row.solve_inequalities ~stage:Stage1 ~discardable_vars rest Row.empty_env in
  let open_sizing =
    List.filter sizing ~f:(function
      | Row.Dim_eq { d1 = Var v; d2 = Dim { d; _ }; _ } -> (
          match Row.get_dim_from_env env v with Some (Dim { d = d'; _ }) -> d <> d' | _ -> true)
      | _ -> true)
  in
  Row.solve_inequalities ~stage:Stage1 (open_sizing @ unsolved) env

(** Computes the indexing into subtensors given the shape information of a tensor.
    [derive_projections] should only be invoked when the shapes are fully inferred already! Sets
    [unsafe_projections] as a side effect. Raises if projections were already computed. *)
let%debug4_sexp derive_projections (update_step : update_step) : unit =
  if Option.is_some update_step.unsafe_projections then
    raise
    @@ Utils.User_error "derive_projections: projections already computed for this update_step";
  let original_step = update_step in
  let update_step, dealiased = dealias_operands update_step in
  let resolved_padding, _old_inferred_padding = fresh_proj_ids update_step in
  (* We will not use the old inferred padding so that we can derive precisely the padding
     contributed by this step. *)
  let _debug_update_step : update_step = update_step in
  let (proj_axis_env, discardable_vars, ineqs) :
      proj_axis_env * Row.dim_var_set * Row.constraint_ list =
    get_inequalities ~for_projections:true update_step
  in
  (* We need to solve the equations/inequalities one last time because of fresh row variables
     potentially generated by [get_inequalities]. Since the variables in the shapes must be
     substituted-out at this point, the global state is already an empty env, but in principle we
     want to only find a local solution to not contaminate projections across operations. *)
  let unsolved, local_env = solve_with_sizing ~discardable_vars ineqs in
  let unsolved, local_env = Row.solve_inequalities ~stage:Stage2 unsolved local_env in
  let unsolved, local_env = Row.solve_inequalities ~stage:Stage3 unsolved local_env in
  let unsolved, local_env = Row.solve_inequalities ~stage:Stage4 unsolved local_env in
  let unsolved, local_env = Row.solve_inequalities ~stage:Stage5 unsolved local_env in
  let unsolved, local_env = Row.solve_inequalities ~stage:Stage6 unsolved local_env in
  let unsolved, local_env = Row.solve_inequalities ~stage:Stage7 unsolved local_env in
  refuse_unsolved ~where:"derive_projections" unsolved;
  let local_env = Row.populate_dim_proj_in_solved local_env in
  let rhs, skip_deriving = projected_operands update_step.logic in
  let terminal_rows_for_inputs =
    List.concat_map rhs ~f:(fun sh ->
        [
          Row.Terminal_row (false, sh.batch, []);
          Row.Terminal_row (false, sh.input, []);
          Row.Terminal_row (false, sh.output, []);
        ])
  in
  (* Important: ineqs must not be substituted / solved before getting proj_equations, because
     get_inequalities provides indexing information that is lost after substitution. *)
  let proj_eqs : Row.proj_equation list =
    Row.get_proj_equations
      ?spec:(Option.map (einsum_of_logic update_step.logic) ~f:fst)
      (terminal_rows_for_inputs @ List.filter ineqs ~f:(fun c -> not (is_projection_sizing c)))
      proj_axis_env local_env
  in
  (* resolved_padding is passed for verification only - to check that operation padding doesn't
     exceed already-locked padding. It won't be used to set padding on shapes (get_dim_padding only
     returns inferred_padding of proj_env, which is for the current operation only). *)
  (* gh-504: a max/tropical-family accumulation (non-finite identity) lowers its padded ([=]-mode)
     window projections with clamped window bounds instead of demanding margins on the operand —
     skipping an out-of-range window position contributes the accumulation identity exactly. With
     no demand registered, [step_touches_margins] stays false for this step, so no neutral element
     is committed either: the operand composes freely with 0-neutral margin-touching consumers
     (padded convs — the Inception-block pattern). *)
  let clamp_padded =
    match update_step.neutral_elem with Some v -> not (Float.is_finite v) | None -> false
  in
  let proj_env : Row.proj_env =
    Row.solve_proj_equations ~clamp_padded ~resolved_padding ~inferred_padding:[] proj_eqs
  in
  let dims_of (sh : t) =
    sh.batch.beg_dims @ sh.batch.dims @ sh.output.beg_dims @ sh.output.dims @ sh.input.beg_dims
    @ sh.input.dims
  in
  let lhs = update_step.shape in
  let lhs_dims = to_dims_impl lhs in
  let rhs_dims = Array.of_list_map ~f:to_dims_impl rhs in
  let all_dims : Row.dim list = List.concat_map ~f:dims_of @@ (lhs :: rhs) in
  (* Note: the ordering will affect performance of naive backends. *)
  let all_product_projs : (Row.proj_id * int) list =
    Utils.unique_keep_first ~equal:(fun (p, _) (q, _) -> Row.equal_proj_id p q)
    @@ List.filter_map all_dims ~f:(Row.get_product_proj proj_env)
  in
  (* Deduplicate by iterator symbol. When Conv_input with d=1 over dimension is used, multiple
     proj_ids can share the same iterator (e.g., input height and kernel height). We keep the first
     occurrence for each unique iterator symbol. *)
  let all_product_projs_with_iters =
    List.map all_product_projs ~f:(fun (p, d) -> (p, d, Row.proj_to_iterator_exn proj_env p))
  in
  let unique_by_iterator =
    Utils.unique_keep_first
      ~equal:(fun (_, _, s1) (_, _, s2) -> Idx.equal_symbol s1 s2)
      all_product_projs_with_iters
  in
  (* gh-490 symbolic extents: associate each symbolic axis's product iterator with the axis's static
     symbol. The product-space entry is the maximum extent; lowering guards loop bodies with
     [iterator < value] when the symbol is bound. A maximum-one axis has no iterator, but retain its
     symbol under [None] so a consumer such as [Set_vec_unop] can reject the dynamic extent rather
     than silently treating it as the maximum (gh-ocannl-817). Distinct symbols sharing an iterator
     cannot arise (a [Sym] unifies only with itself), but check defensively. *)
  let extent_syms : (Idx.symbol option * Idx.static_symbol) list =
    let all =
      List.filter_map all_dims ~f:(fun dim ->
          match dim with
          | Row.Sym { sym; _ } ->
              let iter =
                Option.bind (Row.get_product_proj proj_env dim) ~f:(fun (p, _) ->
                    try Some (Row.proj_to_iterator_exn proj_env p) with Invalid_argument _ -> None)
              in
              Some (iter, sym)
          | _ -> None)
      |> List.dedup_and_sort ~compare:[%compare: Idx.symbol option * Idx.static_symbol]
    in
    List.iter all ~f:(fun (it, sym) ->
        List.iter all ~f:(fun (it2, sym2) ->
            match (it, it2) with
            | Some it, Some it2
              when Idx.equal_symbol it it2 && not (Idx.equal_static_symbol sym sym2) ->
                raise
                @@ Row.Shape_error
                     ( "derive_projections: distinct symbolic extents share a product iterator",
                       [ Shape_mismatch (lhs :: rhs) ] )
            | Some _, Some _ | Some _, None | None, Some _ | None, None -> ()));
    all
  in
  (* Most product projections occur as top-level dimensions of the result or operands. Constituents
     that occur only inside a compound projection do not: notably a unary convolution's kernel, and
     concat components. Include those solver-recorded iterators after the shape-derived axes. *)
  let all_product_dim_entries =
    Row.product_dim_iterators proj_env
    |> Utils.unique_keep_first ~equal:(fun (_, _, s1) (_, _, s2) -> Idx.equal_symbol s1 s2)
  in
  let unique_by_iterator =
    let seen =
      Set.of_list (module Idx.Symbol) (List.map unique_by_iterator ~f:(fun (_, _, s) -> s))
    in
    unique_by_iterator
    @ List.filter all_product_dim_entries ~f:(fun (_, _, s) -> not (Set.mem seen s))
  in
  let symbol_to_proj =
    Map.of_alist_exn
      (module Idx.Symbol)
      (List.map all_product_dim_entries ~f:(fun (p, d, s) -> (s, (p, d))))
  in
  (* Build connected components from Concat indices. Symbols that appear together in a Concat must
     be iterated together. We use union-find to group symbols into connected components. Include
     both product dimensions and Concat dimensions. Note: Concat can appear either as Row.Concat dim
     or as a regular Dim whose proj_id maps to Idx.Concat. *)
  let product_indices : Idx.axis_index list =
    List.filter_map all_dims ~f:(fun dim ->
        match dim with
        | Row.Concat _ -> Some (Row.get_dim_index proj_env dim)
        | Dim { proj_id = Some _; _ } -> (
            match Row.get_product_proj proj_env dim with
            | Some _ -> Some (Row.get_dim_index proj_env dim)
            | None -> (
                (* Also check if dim's projection maps to Idx.Concat *)
                try
                  match Row.get_dim_index proj_env dim with
                  | Idx.Concat _ as idx -> Some idx
                  | _ -> None
                with _ -> None))
        | _ -> (
            match Row.get_product_proj proj_env dim with
            | Some _ -> Some (Row.get_dim_index proj_env dim)
            | None -> None))
  in
  let concat_groups : Idx.symbol list list =
    List.filter_map product_indices ~f:(function Idx.Concat syms -> Some syms | _ -> None)
  in
  let unique_by_iterator =
    let seen =
      Set.of_list (module Idx.Symbol) (List.map unique_by_iterator ~f:(fun (_, _, s) -> s))
    in
    let missing_symbols =
      concat_groups |> List.concat
      |> Utils.unique_keep_first ~equal:Idx.equal_symbol
      |> List.filter ~f:(fun s -> not (Set.mem seen s))
    in
    let missing_entries =
      List.filter_map missing_symbols ~f:(fun s ->
          Map.find symbol_to_proj s |> Option.map ~f:(fun (p, d) -> (p, d, s)))
    in
    unique_by_iterator @ missing_entries
  in
  (* Union-find to group symbols that appear in the same Concat. Symbols within the same Concat must
     be in the same iteration group so that they are iterated SEQUENTIALLY (via unflat_lines), not
     NESTED. For (a, b) ++^ "a; b => a^b", i2 and i3 should be in the same group: - First iterate i2
     (RHS[0] active) - Then iterate i3 (RHS[1] active) This is achieved by making them part of the
     same product component. *)
  let symbol_classes : (Idx.symbol, Idx.symbol) Hashtbl.t = Hashtbl.create (module Idx.Symbol) in
  let find_repr sym =
    let rec loop s =
      match Hashtbl.find symbol_classes s with
      | None -> s
      | Some parent when Idx.equal_symbol parent s -> s
      | Some parent ->
          let repr = loop parent in
          Hashtbl.set symbol_classes ~key:s ~data:repr;
          repr
    in
    loop sym
  in
  let union sym1 sym2 =
    let r1 = find_repr sym1 and r2 = find_repr sym2 in
    if not (Idx.equal_symbol r1 r2) then Hashtbl.set symbol_classes ~key:r1 ~data:r2
  in
  (* Union all symbols within each Concat group to make them iterate sequentially *)
  List.iter concat_groups ~f:(fun syms ->
      match syms with [] -> () | first :: rest -> List.iter rest ~f:(fun s -> union first s));
  (* Group unique_by_iterator entries by their representative symbol *)
  let components : (Idx.symbol, (Row.proj_id * int * Idx.symbol) list) Hashtbl.t =
    Hashtbl.create (module Idx.Symbol)
  in
  List.iter unique_by_iterator ~f:(fun ((_, _, sym) as entry) ->
      let repr = find_repr sym in
      Hashtbl.update components repr ~f:(function None -> [ entry ] | Some l -> entry :: l));
  (* Convert to arrays, preserving order by first occurrence *)
  let seen_reprs = Hash_set.create (module Idx.Symbol) in
  let ordered_components =
    List.filter_map unique_by_iterator ~f:(fun (_, _, sym) ->
        let repr = find_repr sym in
        if Hash_set.mem seen_reprs repr then None
        else (
          Hash_set.add seen_reprs repr;
          Hashtbl.find components repr))
  in
  let product_components : Idx.component array =
    Array.of_list_map ordered_components ~f:(fun entries ->
        List.map entries ~f:(fun (_, d, s) -> (d, s)))
  in
  let indices_of_sh (sh : t) =
    Array.of_list_map ~f:(Row.get_dim_index proj_env)
    @@ List.concat
         [
           sh.batch.beg_dims;
           sh.batch.dims;
           sh.output.beg_dims;
           sh.output.dims;
           sh.input.beg_dims;
           sh.input.dims;
         ]
  in
  (* Extract padding from proj_env and set the padding fields on shapes *)
  let update_padding old_padding new_padding =
    let default = Ir.Ops.{ left = 0; right = 0 } in
    Option.value ~default
    @@ Option.merge old_padding new_padding ~f:(fun old new_p ->
        Ir.Ops.{ left = max old.left new_p.left; right = max old.right new_p.right })
  in
  let padding_of_row old_padding (row : Row.t) : padding =
    let paddings =
      (* Guard against insufficient shape information error. *)
      ignore (row_to_dims row);
      List.mapi (row.beg_dims @ row.dims) ~f:(fun i d ->
          update_padding
            (Option.map old_padding ~f:(fun p -> p.(i)))
            (Row.get_dim_padding proj_env d))
    in
    (* Only return Some if at least one dimension has non-zero padding *)
    if List.for_all paddings ~f:(fun p -> p.left = 0 && p.right = 0) then None
    else Some (Array.of_list paddings)
  in
  let set_padding (sh : t) : unit =
    let batch = padding_of_row sh.batch_padding sh.batch in
    let output = padding_of_row sh.output_padding sh.output in
    let input = padding_of_row sh.input_padding sh.input in
    (* An axis-free result keeps the padding already there; the setter validates all three before
       writing any (gh-ocannl-1301). *)
    set_paddings sh
      ~batch:(Option.first_some batch sh.batch_padding)
      ~input:(Option.first_some input sh.input_padding)
      ~output:(Option.first_some output sh.output_padding)
  in
  (* Whether THIS step demands nonzero padding on [sh] — i.e. actually reads (rhs) or writes (lhs)
     the margins. [Row.get_dim_padding] only reports the current operation's inferred padding. *)
  let step_touches_margins (sh : t) : bool =
    List.exists [ sh.batch; sh.output; sh.input ] ~f:(fun row ->
        List.exists (row.Row.beg_dims @ row.Row.dims) ~f:(fun d ->
            match Row.get_dim_padding proj_env d with
            | Some Ir.Ops.{ left; right } -> left > 0 || right > 0
            | None -> false))
  in
  let update_padding_elem (sh : t) : unit =
    (* The padding neutral element commits like the rest of the tensor's identity: the margins of a
       padded buffer permanently hold a single neutral value. Only operations that actually read the
       margins (demand nonzero padding on [sh] in this step) participate — a valid-window reader
       never sees the margins, so its accumulation neutral is irrelevant. A genuine conflict is
       rejected: the remedy is a materialized copy per neutral. *)
    if step_touches_margins sh then
      set_padding_elem sh
        (match (sh.padding_elem, update_step.neutral_elem) with
        | None, None -> None (* Both unknown *)
        | None, Some v -> Some (Some v) (* First operation sets the value *)
        | Some (Some v1), Some v2 when Float.( = ) v1 v2 -> sh.padding_elem (* Consistent *)
        | Some (Some v1), Some v2 ->
            raise
            @@ Row.Shape_error
                 ( [%string
                     "Conflicting padding neutral elements: an operation touches the tensor's \
                      margins expecting %{v2#Float}, but they are committed to %{v1#Float} \
                      (demanded by an earlier operation). Materialize a separate copy of the \
                      tensor for one of the padded consumers (for a padded max-pool, \
                      Nn_blocks.max_pool2d_copy)"],
                   [ Shape_mismatch [ sh ] ] )
        | Some None, _ -> Some None (* Unreachable: conflicts raise instead. *)
        | Some _, None -> sh.padding_elem)
    (* Operation has no neutral elem, keep current *)
  in
  if skip_deriving then ()
  else (
    set_padding lhs;
    List.iter rhs ~f:set_padding;
    (* Update padding_elem based on the operation's neutral element: for rhs shapes the operation
       reads the margins; for the lhs, a padded write projection (scatter-style [=] output specs)
       commits the accumulation's neutral just the same — this keeps "padded implies a known neutral
       element" an invariant (cf. [to_padding]). *)
    update_padding_elem lhs;
    List.iter rhs ~f:update_padding_elem;
    merge_dealiased_padding dealiased;
    let project_lhs = indices_of_sh lhs in
    let project_rhs = Array.of_list_map ~f:indices_of_sh rhs in
    let projections =
      try
        Idx.
          {
            components = product_components;
            lhs_dims;
            rhs_dims;
            project_lhs;
            project_rhs;
            extent_syms;
            debug_info =
              {
                spec = logic_to_spec update_step.logic;
                derived_for = sexp_of_update_step update_step;
                trace = [ ("derive_projections", Idx.unique_debug_id ()) ];
              };
          }
      with Row.Shape_error (s, trace) ->
        raise @@ Row.Shape_error (s, Shape_mismatch (lhs :: rhs) :: trace)
    in
    Update_step_record.set_projections original_step projections)

(** {3 Shape inference} *)

let apply_env_t env sh =
  set_rows sh ~batch:(Row.subst_row env sh.batch) ~input:(Row.subst_row env sh.input)
    ~output:(Row.subst_row env sh.output)

(** Computes the product of dimensions in a row (both flanks). Returns [None] if any dimension is
    not yet resolved (still a variable or unresolved affine), or if the row ends in a row variable
    that is not yet resolved -- the row could still gain axes, so its product is unknown. *)
let compute_row_product env (row : Row.t) : int option =
  let rec product = function
    | [] -> Some 1
    | dim :: rest -> (
        let dim_val =
          match dim with
          | Row.Dim { d; _ } -> Some d
          | Row.Sym _ -> None (* Symbolic dims don't participate in row-product arithmetic. *)
          | Row.Var v -> Row.get_dim_val env v
          | Row.Affine _ -> None (* TODO: handle affine/convolution input dimensions *)
          | Row.Concat dims ->
              List.fold dims ~init:(Some 0) ~f:(fun acc d ->
                  match acc with
                  | None -> None
                  | Some sum -> (
                      match d with
                      | Row.Dim { d; _ } -> Some (sum + d)
                      | Row.Var v -> Option.map (Row.get_dim_val env v) ~f:(fun d -> sum + d)
                      | _ -> None))
        in
        match (dim_val, product rest) with
        | Some d, Some rest_product -> Some (d * rest_product)
        | _ -> None)
  in
  (* Fuel guards against reference cycles among row variables. *)
  let rec row_product fuel (row : Row.t) =
    match (product (row.beg_dims @ row.dims), row.bcast) with
    | None, _ -> None
    | Some p, Row.Broadcastable -> Some p
    | Some p, Row.Row_var v when fuel > 0 -> (
        match Row.get_row_from_env env v with
        | Some tail -> Option.map (row_product (fuel - 1) tail) ~f:(fun q -> p * q)
        | None -> None)
    | Some _, Row.Row_var _ -> None
  in
  row_product 100 row

(** Publishes into each capture reference of [update_step] the solution [env] has reached for the
    variable the reference is bound to: a dimension variable's solution of either kind -- a concrete
    [Dim] as [solved_dim], a symbolic [Sym] as [solved_sym] (gh-ocannl-1616) -- and a row variable's
    numeric product as [solved_dim]. A row holding a symbolic extent has no product and records
    nothing: symbolic dimensions take no part in row-product arithmetic.

    Only an unsolved reference (neither field set) is written. A solution the reference already
    holds -- set by [set_dim]/[set_sym_dim], or published by an earlier step or inference -- is left
    as it is and never adjudicated here: every held solution has a constraint relating it to the
    reference's current variable ([set_dim]/[set_sym_dim] push one into [active_constraints], which
    [finish_inference] solves before it publishes; a later binding emits it from
    [bind_delayed_vars_to_envs]), so a disagreement is the solver's [Shape_error] before it could
    reach this point. {!Ir.Indexing.set_solved_dim}'s refusal of a second kind of solution is a
    backstop no inference path reaches, and since gh-ocannl-1624 no other path either: its
    [shape_writer] is [Shape]'s.

    At the [closure] (only [finish_inference] passes it), a reference also gets [carried_row]
    (gh-ocannl-1689): the axes of its row when it is bound to a row whose solution is closed --
    [Broadcastable], every axis a [Dim] or a [Sym] -- whether or not it holds a number already (a
    [set_dim] on a row reference sets the product early), and [None] when it is bound to a
    dimension. Substitution leaves a [Concat] only for a symbolic component, which stays [None], as
    does an [Affine] over a symbolic extent: carried into the next inference, it would reach
    projections there. The binding at the closure decides, not the first row to resolve: within one
    inference a row captured after a dimension binding is tied to it by product, whatever row an
    earlier binding had, and publishing a row as it resolved would make the carried value depend on
    whether it resolved at a registration's Stage1 or at the closure. *)
let carried_row_of env row_var =
  let row = Row.subst_row env (Row.get_row_for_var Row.empty_provenance row_var) in
  let carried = function
    | Row.Dim d -> Some (Row.Dim { d with proj_id = None })
    | Row.Sym s -> Some (Row.Sym { s with sym_proj_id = None })
    | Row.Var _ | Row.Affine _ | Row.Concat _ -> None
  in
  match row.bcast with
  | Row.Row_var _ -> None
  | Row.Broadcastable -> Option.all (List.map (row.beg_dims @ row.dims) ~f:carried)

let publish_ref_solutions ~closure env var_refs =
  List.iter var_refs ~f:(fun (delayed_ref : delayed_var_ref) ->
      let unsolved =
        Option.is_none delayed_ref.var_ref.solved_dim
        && Option.is_none delayed_ref.var_ref.solved_sym
      in
      (if closure && not (is_trace_input delayed_ref) then
         match delayed_ref.var with
         | `Row row_var -> Delayed_ref.set_carried_row delayed_ref (carried_row_of env row_var)
         | `Dim _ -> Delayed_ref.set_carried_row delayed_ref None
         | `Not_set_yet -> ());
      match delayed_ref.var with
      | _ when not unsolved -> ()
      (* An input of the running shape-scheme trace (gh-ocannl-1791): its solution belongs to the
         application, and the trace's constraints already say what this one would. *)
      | _ when is_trace_input delayed_ref -> ()
      | `Not_set_yet -> () (* Variable not bound yet, will be set later *)
      | `Dim dim_var -> (
          match Row.get_dim_from_env env dim_var with
          | Some (Row.Dim { d; basis; _ }) -> Delayed_ref.set_solved_dim ~basis delayed_ref d
          | Some (Row.Sym { sym; _ }) -> Delayed_ref.set_solved_sym delayed_ref sym
          | Some (Row.Var _ | Row.Affine _ | Row.Concat _) | None -> () (* Not yet resolved *))
      | `Row row_var -> (
          match Row.get_row_from_env env row_var with
          | Some row -> (
              match compute_row_product env row with
              | Some product -> Delayed_ref.set_solved_dim delayed_ref product
              | None -> () (* Row has unresolved or symbolic dimensions *))
          | None -> () (* Row variable not yet resolved *)))

let update_delayed_var_refs ~closure env update_step =
  match update_step.logic with
  | Transpose (Permute (_, var_refs), _) -> publish_ref_solutions ~closure env var_refs
  | Broadcast (Einsum (_, var_refs), _, _) -> publish_ref_solutions ~closure env var_refs
  | Broadcast_tern (Einsum_tern (_, var_refs), _, _, _) ->
      publish_ref_solutions ~closure env var_refs
  | Block { delayed_vars; _ } -> publish_ref_solutions ~closure env delayed_vars
  | _ -> ()

let apply_env_step ~closure env update_step =
  iter_shapes update_step ~f:(apply_env_t env);
  update_delayed_var_refs ~closure env update_step

(* Registration statistics, for attributing construction cost (gh-ocannl-1752). *)
type registration_counts = {
  checked_applications : int;
  inline_registrations : int;
  instantiated_constraints : int;
  instantiated_solutions : int;
  instantiated_boundary : int;
  inline_constraints : int;
  closures : int;
  closure_constraints : int;
  rank_search_visits : int;
  dim_cycle_visits : int;
  subst_visits : int;
  row_visit_emissions : int;
}
[@@deriving sexp_of]

let zero_counts =
  {
    checked_applications = 0;
    inline_registrations = 0;
    instantiated_constraints = 0;
    instantiated_solutions = 0;
    instantiated_boundary = 0;
    inline_constraints = 0;
    closures = 0;
    closure_constraints = 0;
    rank_search_visits = 0;
    dim_cycle_visits = 0;
    subst_visits = 0;
    row_visit_emissions = 0;
  }

let counts = ref zero_counts
[@@mutation_serial.exempt "registration statistics, not inference state"]

let registration_counts () =
  {
    !counts with
    rank_search_visits = Row.rank_search_visits ();
    dim_cycle_visits = Row.dim_cycle_visits ();
    subst_visits = Row.subst_visits ();
    row_visit_emissions = Row.row_visit_emissions ();
  }

let reset_registration_counts () =
  counts := zero_counts;
  Row.reset_rank_search_visits ();
  Row.reset_dim_cycle_visits ();
  Row.reset_subst_visits ();
  Row.reset_row_visit_emissions ()

(* Where shape-registration time goes (gh-ocannl-1752), process-wide since the last reset. *)
type registration_times = {
  tracing_s : float;
  instantiation_s : float;
  inline_steps_s : float;
  closure_s : float;
}
[@@deriving sexp_of]

let zero_times = { tracing_s = 0.; instantiation_s = 0.; inline_steps_s = 0.; closure_s = 0. }
let times = ref zero_times [@@mutation_serial.exempt "registration timings, not inference state"]
let registration_times () = !times
let reset_registration_times () = times := zero_times

type phase = [ `Tracing | `Instantiation | `Inline_step | `Closure ]

let add_time (phase : phase) dt =
  let t = !times in
  times :=
    match phase with
    | `Tracing -> { t with tracing_s = t.tracing_s +. dt }
    | `Instantiation -> { t with instantiation_s = t.instantiation_s +. dt }
    | `Inline_step -> { t with inline_steps_s = t.inline_steps_s +. dt }
    | `Closure -> { t with closure_s = t.closure_s +. dt }

(* Nested timed phases (an instantiation inside a closure built mid-construction, say) are charged
   to the innermost only: the outer phase's clock pauses meanwhile. *)
let timing_paused = ref 0. [@@mutation_serial.exempt "registration timings, not inference state"]

let timed (phase : phase) f =
  let started = Mtime_clock.counter () in
  let paused0 = !timing_paused in
  Exn.protect ~f ~finally:(fun () ->
      let total = Mtime.Span.to_float_ns (Mtime_clock.count started) /. 1e9 in
      let inner = !timing_paused -. paused0 in
      add_time phase (total -. inner);
      timing_paused := paused0 +. total)

let note_checked_application () =
  counts := { !counts with checked_applications = !counts.checked_applications + 1 }

(* Before anything is written: a step over a shape a rewind discarded (an operand, a [%cd] source or
   target retained across it) is refused, and taints nothing. *)
let refuse_discarded_operands update_step =
  iter_shapes update_step ~f:(fun sh ->
      Option.iter sh.discarded_by ~f:(fun provenance ->
          raise
          @@ Row.Shape_error
               ( [%string
                   "the shape of #%{sh.id#Int} %{sh.debug_name} was discarded by the rewind after \
                    %{provenance}; re-run its definition"],
                 [] )))

(* The registration proper: the step's own inequalities, derived and solved. *)
let propagate_inline (update_step : update_step) : unit =
  (* The step and its inequalities are registered before the solver runs: a solver failure leaves
     them registered against the old environment, so it taints the session (gh-ocannl-1301). *)
  with_taint_on_failure
    ~describe:(fun exn ->
      [%string
        "shape propagation for #%{update_step.shape.id#Int} %{update_step.shape.debug_name}: \
         %{exn_summary exn}"])
    (fun () ->
      if not (tracing_in_progress ()) then
        counts := { !counts with inline_registrations = !counts.inline_registrations + 1 };
      (* Allow the derivation of constraints to depend on the shapes (currently, only Batch_slice
         does). *)
      iter_shapes update_step ~f:(apply_env_t !state);
      let _, discardable_vars, ineqs = get_inequalities update_step in
      active_update_steps := update_step :: !active_update_steps;
      active_constraints := ineqs @ !active_constraints;
      if not (tracing_in_progress ()) then
        counts :=
          { !counts with inline_constraints = !counts.inline_constraints + List.length ineqs };
      let ineqs', env = Row.solve_inequalities ~stage:Row.Stage1 ~discardable_vars ineqs !state in
      let _debug_remaining_constraints : Row.constraint_ list = ineqs' in
      apply_env_step ~closure:false env update_step;
      state := env)

(* Before anything is written too, so a refused construction taints nothing: [get_inequalities] has
   already marked operands used by the time an einsum branch runs. This refuses what projection
   derivation would at every closure (gh-ocannl-1803, gh-ocannl-1983). The step's spec and captures
   decide it, not its shapes, so it needs no reasoning about what an application or a consumer of
   the result can supply. A trace runs the body, so a shape scheme reports it at the definition. *)
let refuse_unsized_kernels (update_step : update_step) =
  Option.iter (einsum_of_logic update_step.logic) ~f:(fun (spec, dim_refs) ->
      Option.iter
        (List.hd (unsized_padded_kernels ~spec ~dim_refs))
        ~f:(fun name ->
          let v = Row.get_var ~name () in
          raise @@ Row.unsized_kernel_error ~spec ~use_padding:true v (Row.Var v)))

(** The concatenations of a spec that have no meaning, whatever the shapes (gh-ocannl-2017). A
    concatenated axis is one axis of alternatives: a position lies in one segment, and an integer
    component is a segment the spec skips. Refused before anything is written, like
    [refuse_unsized_kernels], and from the spec, not the rows: closure sums a concatenated axis
    away, so a check on them would see a different program for a closed operand than for an open
    one.

    - One tensor's axes needing two components of a concatenation: no position has both, so
      lowering's segment blocks reach that access for none of them, at any extent but 1.
    - A result concatenation no component of which an operand names: nothing says what its segments
      hold. It was iterated like a plain axis, every cell taking the reduced operand, against what
      the components say everywhere else ([2^b => 2^b] skips the [2]). An operand concatenation no
      component of which the result names is a reduction over every cell, and stays.
    - Block operands out of the order of the segments they fill: an operand of a width-1 segment
      indexes at [Fixed_idx 0] and names no segment, so the lowering of a block selects its operand
      by position (in [Assignments]), and two swapped width-1 operands would swap silently. Other
      einsums multiply their operands, which may share a segment. *)
let refuse_meaningless_concats (update_step : update_step) =
  Option.iter (einsum_of_logic update_step.logic) ~f:(fun (spec, _) ->
      let ls_rhs_list, ls_lhs = einsum_of_spec spec in
      let refuse msg =
        raise
        @@ Row.Shape_error
             ([%string "einsum spec \"%{spec}\": %{msg}"], [ Shape_mismatch [ update_step.shape ] ])
      in
      let axes_of (ls : parsed_axis_labels) = Map.data ls.labels in
      let outside_concats axes =
        List.concat_map axes ~f:(function
          | Label l -> [ l ]
          | Affine_spec { over_label; conv; _ } ->
              over_label
              :: Option.value_map conv ~default:[] ~f:(fun { kernel_label; _ } -> [ kernel_label ])
          | Fixed_index _ | Concat_spec _ -> [])
        |> List.dedup_and_sort ~compare:String.compare
      in
      let in_concats axes =
        List.concat_map axes ~f:(function
          | Concat_spec components ->
              List.filter_map components ~f:(function
                | Concat_label l -> Some l
                | Concat_size _ -> None)
          | Label _ | Fixed_index _ | Affine_spec _ -> [])
      in
      let concats axes =
        List.filter_map axes ~f:(function Concat_spec components -> Some components | _ -> None)
      in
      let concat_text components =
        String.concat ~sep:"^" (List.map components ~f:concat_component_text)
      in
      let slots = List.map (ls_lhs :: ls_rhs_list) ~f:axes_of in
      let result_axes = axes_of ls_lhs in
      let operands = List.map ls_rhs_list ~f:axes_of in
      List.iter (List.concat_map slots ~f:concats) ~f:(fun components ->
          let labels = in_concats [ Concat_spec components ] in
          List.iter slots ~f:(fun axes ->
              match List.filter (outside_concats axes) ~f:(List.mem labels ~equal:String.equal) with
              | first :: second :: _ ->
                  refuse
                    [%string
                      "one tensor's axes need both %{first} and %{second}, components of the \
                       concatenated axis %{concat_text components}, but a position of that axis \
                       lies in one segment, so no position has both and the operation would \
                       compute nothing. Name one component per tensor (x ++ \"a^b => a\" takes a \
                       segment; (x, y) ++^ \"a; b => a^b\" joins two)"]
              | _ -> ()));
      let operand_labels =
        List.concat_map operands ~f:(fun axes -> outside_concats axes @ in_concats axes)
      in
      let result_concats = concats result_axes in
      List.iter result_concats ~f:(fun components ->
          if
            not
              (List.exists
                 (in_concats [ Concat_spec components ])
                 ~f:(List.mem operand_labels ~equal:String.equal))
          then
            let text = concat_text components in
            refuse
              [%string
                "no operand names a component of the concatenated result axis %{text}, so nothing \
                 says what its segments hold. To write part of the axis, name the segment (a => \
                 2^a^3 writes a between two skipped segments); to repeat a reduced operand in \
                 every cell, use a plain axis and size it (x ++ \"a => b\" [\"b\"], then \
                 Shape.set_dim b 5)"]);
      match (update_step.logic, result_concats) with
      | Block _, [ components ] when List.length components = List.length operands ->
          let position label =
            List.findi components ~f:(fun _ c ->
                match c with Concat_label l -> String.equal l label | Concat_size _ -> false)
            |> Option.map ~f:fst
          in
          let filled =
            List.filter_map operands ~f:(fun axes ->
                match
                  List.filter_map
                    (List.dedup_and_sort ~compare:String.compare
                       (outside_concats axes @ in_concats axes))
                    ~f:position
                with
                | [ position ] -> Some position
                | _ -> None)
          in
          if not (List.is_sorted filled ~compare:Int.compare) then
            let text = concat_text components in
            refuse
              [%string
                "the operands are not in the order of the segments of the concatenated result axis \
                 %{text}. List them in that order: (y, x) ++^ \"a; b => a^b\" rather than (x, y) \
                 ++^ \"b; a => a^b\""]
      | _ -> ())

let%debug4_sexp propagate_shapes (update_step : update_step) : unit =
  check_untainted ();
  refuse_discarded_operands update_step;
  refuse_unsized_kernels update_step;
  refuse_meaningless_concats update_step;
  if tracing_in_progress () then propagate_inline update_step
  else timed `Inline_step (fun () -> propagate_inline update_step)

let with_construction_window f =
  Int.incr construction_depth;
  Exn.protect ~f ~finally:(fun () -> Int.decr construction_depth)

let%debug4_sexp finish_inference_body (() : unit) : unit =
  counts :=
    {
      !counts with
      closures = !counts.closures + 1;
      closure_constraints = !counts.closure_constraints + List.length !active_constraints;
    };
  let unsolved =
    List.filter !active_constraints ~f:(function
      | Shape_row (r, _) | Terminal_row (_, r, _) ->
          not (List.exists (Row.row_shapes r) ~f:(Set.mem !unused_shapes))
      | _ -> true)
  in
  (* TODO: optimize to keep all needed information in unsolved, rather than starting with all
     constraints. *)
  let unsolved, env = Row.solve_inequalities ~stage:Stage2 unsolved !state in
  let unsolved, env = Row.solve_inequalities ~stage:Stage3 unsolved env in
  let all_update_rows =
    List.concat_map ~f:all_rows_w_origin !active_update_steps
    @ List.concat_map !active_instances ~f:(fun inst -> inst.inst_rows)
    |> List.map ~f:(fun (r, o) -> (Row.subst_row env r, o))
    |> List.dedup_and_sort ~compare:(fun (r1, _) (r2, _) -> Row.compare r1 r2)
  in
  let unsolved =
    List.map ~f:(fun (ro, o) -> Row.Shape_row (ro, [ o ])) all_update_rows @ unsolved
  in
  let unsolved, env = Row.solve_inequalities ~stage:Stage4 unsolved env in
  let unsolved, env = Row.solve_inequalities ~stage:Stage5 unsolved env in
  let unsolved, env = Row.solve_inequalities ~stage:Stage6 unsolved env in
  let unsolved, env = Row.solve_inequalities ~stage:Stage7 unsolved env in
  refuse_unsolved ~where:"finish_inference" unsolved;
  let _active_update_steps : update_step list = !active_update_steps in
  List.iter ~f:(apply_env_step ~closure:true env) !active_update_steps;
  List.iter !active_instances ~f:(fun inst ->
      List.iter inst.inst_shapes ~f:(apply_env_t env);
      publish_ref_solutions ~closure:true env inst.inst_refs);
  let _applied_update_steps : update_step list = !active_update_steps in
  (* Derive projections for all active update_steps, setting their unsafe_projections field. This
     must happen before clearing active_update_steps. *)
  List.iter !active_update_steps ~f:(fun update_step ->
      if Option.is_none update_step.unsafe_projections then derive_projections update_step);
  (* The commit: every shape of the applied steps is closed, and the role claims of the open work
     reach their symbols. *)
  List.iter !active_update_steps ~f:(fun update_step -> iter_shapes update_step ~f:close);
  List.iter !active_instances ~f:(fun inst -> List.iter inst.inst_shapes ~f:close);
  Row.commit_symbol_roles ~commit:(Ir.Indexing.commit_role shape_writer);
  open_refs := [];
  Update_step_record.forget_made ();
  Int.incr closure_count;
  checkpoint_lineage := [];
  failed_reads := [];
  active_constraints := [];
  active_update_steps := [];
  active_instances := [];
  (* There should not be any shape variables remaining in any inference-undergoing update steps. *)
  state := Row.empty_env;
  (* gh-ocannl-1859: so nothing the solver keeps per variable or per shape outlives them. Ahead of
     [on_closure], whose checkpoint would otherwise pin these tables. *)
  unused_shapes := Set.empty (module Int);
  Row.retire_closed_variables ();
  !on_closure ()
[@@mutation_serial.boundary
  "the commit point: finish_inference taints the session on any failure of this body"]

let finish_inference (() : unit) : unit =
  (match !tracing with
  | [] -> ()
  | label :: _ ->
      raise
      @@ Untraceable_scheme
           [%string
             "finish_inference: shape inference was finalized while tracing the shape scheme \
              %{label} (via Shape.to_dims, Shape.to_padding or a projections handle) -- not \
              supported: a closure would commit the trace's work, which the trace must leave \
              undone. Do not force dims or projections inside a traced function."]);
  if !construction_depth > 0 then
    raise
    @@ Row.Shape_error
         ( "finish_inference: shape inference was finalized from within [op_asn], mid-construction \
            of a tensor operation (via Shape.to_dims, Shape.to_padding or the projections handle) \
            -- not supported: the operation's own constraints are not registered yet, so its \
            operands would close without them. Fix the [op_asn], then rewind the session \
            (Tensor.rewind_to_last_closure).",
           [] );
  check_untainted ();
  (* A failed closure leaves the open state unsatisfiable (or partially applied): it always taints,
     and it marks nothing closed and moves no checkpoint (gh-ocannl-1301). *)
  try timed `Closure finish_inference_body
  with exn ->
    let bt = Stdlib.Printexc.get_raw_backtrace () in
    taint_session ~provenance:("finish_inference: " ^ exn_summary exn);
    Stdlib.Printexc.raise_with_backtrace exn bt

(* A closed shape's dims and padding are committed: reading them needs no finalization, so a lazy
   over a closed tensor (its node's dims, padding, or a committed step's projections) never runs the
   session-global finalizer, and a failed closure of OTHER open work cannot poison it
   (gh-ocannl-1301). *)
(* A committed-state read of an open shape finalizes; a lazy over it memoizes a failure, so a
   failure is recorded against the shape for the checkpoints' validity (gh-ocannl-903). *)
let finish_for_read ?step (sh : t) =
  if Option.is_some step || not sh.closed then (
    try finish_inference ()
    with exn ->
      let bt = Stdlib.Printexc.get_raw_backtrace () in
      note_failed_read (match step with Some step -> `Step step | None -> `Shape sh);
      Stdlib.Printexc.raise_with_backtrace exn bt)

let to_dims (sh : t) =
  finish_for_read sh;
  to_dims_impl sh

let%track4_sexp to_padding (sh : t) : (Ir.Ops.axis_padding array * float) option =
  finish_for_read sh;
  try
    (* If any row has padding, we need to return padding for all dimensions. Use zero padding for
       rows without explicit padding. *)
    let no_padding = Ir.Ops.{ left = 0; right = 0 } in
    let get_padding_array row_opt row =
      match row_opt with
      | Some padding -> padding
      | None -> Array.create ~len:(List.length (row.Row.beg_dims @ row.Row.dims)) no_padding
    in
    let has_any_padding : bool =
      Option.is_some sh.batch_padding || Option.is_some sh.output_padding
      || Option.is_some sh.input_padding
    in
    if not has_any_padding then None
    else
      let batch : Row.axis_padding array = get_padding_array sh.batch_padding sh.batch in
      let output : Row.axis_padding array = get_padding_array sh.output_padding sh.output in
      let input : Row.axis_padding array = get_padding_array sh.input_padding sh.input in
      (* Padded implies a known neutral element: padding only arises from margin-touching
         accumulation projections, and [update_padding_elem] commits their neutral (conflicts are
         rejected there). *)
      let padded_value =
        match sh.padding_elem with
        | Some (Some v) -> v
        | Some None | None ->
            raise
            @@ Row.Shape_error
                 ( "Internal invariant violation: shape has committed padding but no neutral \
                    element; padding should only arise from accumulation projections that carry \
                    one",
                   [ Shape_mismatch [ sh ] ] )
      in
      Some (Array.concat [ batch; output; input ], padded_value)
  with Row.Shape_error (s, trace) -> raise @@ Row.Shape_error (s, Shape_mismatch [ sh ] :: trace)

let to_bases (sh : t) : string array =
  Array.concat_map ~f:(Row.row_to_bases !state) [| sh.batch; sh.output; sh.input |]

(* Per-kind dimension bases, so callers can check which row/axis a basis landed on (not just that a
   tag appears somewhere in the flattened list). Returns [(batch, input, output)]. *)
let to_bases_bio (sh : t) : string array * string array * string array =
  ( Row.row_to_bases !state sh.batch,
    Row.row_to_bases !state sh.input,
    Row.row_to_bases !state sh.output )

let sexp_of_error_trace = function
  | Shape_mismatch ts -> Sexp.List (Sexp.Atom "Shape_mismatch" :: List.map ts ~f:sexp_of_t)
  | error_trace -> Row.sexp_of_error_trace error_trace

(** {3 Readable shape errors (gh-ocannl-1799)} *)

let reserved_id_base = 1 lsl 40

(* The name of the [n]th variable in order of first appearance: [a] .. [z], then [a1] .. [z1], ...
   Shared by the scheme printer and the error printer. *)
let letter_name n =
  let letter = String.make 1 (Char.of_int_exn (Char.to_int 'a' + (n % 26))) in
  if n < 26 then letter else letter ^ Int.to_string (n / 26)

let kind_to_string = function `Batch -> "batch" | `Input -> "input" | `Output -> "output"

let error_to_string_hum msg trace =
  let names = Hashtbl.create (module String) in
  let n_vars = ref 0 and n_tensors = ref 0 in
  let fresh counter ~mk key =
    Hashtbl.find_or_add names key ~default:(fun () ->
        let n = !counter in
        Int.incr counter;
        mk n)
  in
  let var_id id = fresh n_vars ~mk:(fun n -> "'" ^ letter_name n) ("d" ^ Int.to_string id) in
  let tensor_id id =
    fresh n_tensors ~mk:(fun n -> Int.to_string (n + 1)) ("t" ^ Int.to_string id)
  in
  (* The ids a text spells: a dimension variable [$<id>] (its name, if any, follows as [:<name>]),
     and a tensor of a trace or check, [n<id>] (an unlabeled tensor's name) or [#<id>], whose id
     comes from the reserved range. A session tensor's id stays: it is the user's to look up. *)
  let rename_text s =
    let len = String.length s in
    let buf = Buffer.create len in
    let is_word c = Char.is_alphanum c || Char.equal c '_' || Char.equal c '\'' in
    let rec digits_end j = if j < len && Char.is_digit s.[j] then digits_end (j + 1) else j in
    let rec go i =
      if i < len then
        let c = s.[i] in
        let starts =
          match c with '$' | '#' -> true | 'n' -> i = 0 || not (is_word s.[i - 1]) | _ -> false
        in
        if starts && i + 1 < len && Char.is_digit s.[i + 1] then (
          let j = digits_end (i + 1) in
          let ends_word = j >= len || not (is_word s.[j]) in
          (match Int.of_string_opt (String.sub s ~pos:(i + 1) ~len:(j - i - 1)) with
          | Some id when Char.equal c '$' -> Buffer.add_string buf (var_id id)
          | Some id when id >= reserved_id_base && ends_word ->
              Buffer.add_char buf c;
              Buffer.add_char buf '\'';
              Buffer.add_string buf (tensor_id id)
          | _ -> Buffer.add_substring buf s ~pos:i ~len:(j - i));
          go j)
        else (
          Buffer.add_char buf c;
          go (i + 1))
    in
    go 0;
    Buffer.contents buf
  in
  (* Names are given in reading order, so everything below is sequenced: [List.rev_map] applies [f]
     front to back, while OCaml evaluates a list literal's (and [%string]'s) parts back to front. *)
  let map_seq l ~f = List.rev (List.rev_map l ~f) in
  let concat_seq ~sep l ~f = String.concat ~sep (map_seq l ~f) in
  let pair f a b =
    let a = f a in
    (a, f b)
  in
  let dim_var v = rename_text (Row.dim_to_string Row.Axis_size (Var v)) in
  let dim = Row.dim_to_string ~var_to_string:dim_var Row.Axis_size in
  let row_var v =
    fresh n_vars ~mk:(fun n -> "'" ^ letter_name n) ("r" ^ Int.to_string (Row.row_var_id v))
  in
  let row (r : Row.t) =
    let beg = map_seq r.beg_dims ~f:dim in
    let middle =
      match r.bcast with Row.Row_var v -> [ ".." ^ row_var v ^ ".." ] | Broadcastable -> []
    in
    match beg @ middle @ map_seq r.dims ~f:dim with
    | [] -> "()"
    | items -> String.concat ~sep:"," items
  in
  let total = function
    | Row.Num_elems n -> Int.to_string n
    | Range_elems { lo; hi } -> [%string "%{lo#Int}..%{hi#Int}"]
    | Strided_var { coeff; var; denom; round_up } ->
        let coeff =
          if Utils.is_safe_val coeff then Int.to_string (Utils.safe_force coeff) else "?"
        in
        let div = if round_up then "/^" else "/" in
        [%string "%{coeff}*%{dim_var var}%{div}%{denom#Int}"]
  in
  let row_constraint = function
    | Row.Unconstrained -> "unconstrained"
    | Total_elems { numerator; divided_by; keep_axis = _ } ->
        let numerator = total numerator in
        let divisor = concat_seq ~sep:"" divided_by ~f:(fun v -> "/" ^ dim_var v) in
        [%string "%{numerator}%{divisor} elements"]
    | Exact dims -> "exactly " ^ concat_seq ~sep:"," dims ~f:dim
  in
  let relation f a op b origin =
    let a, b = pair f a b in
    ([%string "%{a} %{op} %{b}"], origin)
  in
  let constraint_ (c : Row.constraint_) =
    match c with
    | Dim_eq { d1; d2; origin } -> relation dim d1 "=" d2 origin
    | Row_eq { r1; r2; origin } -> relation row r1 "=" r2 origin
    | Dim_ineq { res; opnd; from_ = _; origin } -> relation dim res "⊑" opnd origin
    | Row_ineq { res; opnd; origin } -> relation row res "⊑" opnd origin
    | Dim_constr { d; constr; origin } ->
        let constr =
          match constr with
          | Unconstrained_dim -> "unconstrained"
          | At_least_dim n -> [%string ">= %{n#Int}"]
        in
        (dim d ^ " " ^ constr, origin)
    | Rows_constr { r; constr; origin } ->
        let rows = concat_seq ~sep:" ^ " r ~f:row in
        ([%string "%{rows} has %{row_constraint constr}"], origin)
    | Terminal_dim (_, d, origin) -> ("terminal axis " ^ dim d, origin)
    | Terminal_row (_, r, origin) -> ("terminal row " ^ row r, origin)
    | Shape_row (r, origin) -> ("row " ^ row r, origin)
  in
  let origin (o : Row.constraint_origin) =
    let name n = if String.is_empty n then "_" else rename_text n in
    let lhs, rhs = pair name o.lhs_name o.rhs_name in
    let op = Option.value_map o.operation ~default:"" ~f:(fun op -> ": " ^ rename_text op) in
    [%string
      "%{lhs} (%{kind_to_string o.lhs_kind}) with %{rhs} (%{kind_to_string o.rhs_kind})%{op}"]
  in
  let shape (sh : t) =
    let name = rename_text sh.debug_name in
    let b = row sh.batch in
    let i = row sh.input in
    let o = row sh.output in
    [%string "%{name}: %{b}|%{i}->%{o}"]
  in
  (* The sizes in conflict first, then the shapes, then the constraint and where it came from. *)
  let rank = function
    | Row.Dim_mismatch _ | Row.Row_mismatch _ | Row.Index_mismatch _ -> 0
    | Shape_mismatch _ -> 1
    | Row.Constraint_failed _ | Row.Rows_constr_failed _ -> 2
    | _ -> 3
  in
  let ordered = List.stable_sort trace ~compare:(fun a b -> Int.compare (rank a) (rank b)) in
  let header = rename_text msg in
  let lines =
    map_seq ordered ~f:(function
      | Row.Dim_mismatch [ d ] -> [ "the axis: " ^ dim d ]
      | Row.Dim_mismatch ds -> [ "conflicting sizes: " ^ concat_seq ~sep:" vs " ds ~f:dim ]
      | Row.Row_mismatch [ r ] -> [ "the row: " ^ row r ]
      | Row.Row_mismatch rs -> [ "conflicting rows: " ^ concat_seq ~sep:" vs " rs ~f:row ]
      | Row.Index_mismatch idcs ->
          [
            "conflicting indices: "
            ^ concat_seq ~sep:" vs " idcs ~f:(fun i ->
                rename_text (Sexp.to_string (Idx.sexp_of_axis_index i)));
          ]
      | Shape_mismatch shapes -> map_seq shapes ~f:(fun sh -> "shape of " ^ shape sh)
      | Row.Constraint_failed c ->
          let text, origins = constraint_ c in
          ("failed constraint: " ^ text) :: map_seq origins ~f:(fun o -> "  from " ^ origin o)
      | Row.Rows_constr_failed { constr } ->
          [ "failed row constraint: the rows have " ^ row_constraint constr ]
      | other -> [ rename_text (Sexp.to_string_hum (sexp_of_error_trace other)) ])
    |> List.concat
  in
  String.concat ~sep:"\n" (header :: List.map lines ~f:(fun l -> "  " ^ l))

let shape_error_full_sexp () =
  Utils.get_global_flag ~default:false ~arg_name:"shape_error_full_sexp"

let () =
  Sexplib0.Sexp_conv.Exn_converter.add [%extension_constructor Row.Shape_error] (function
    | Row.Shape_error (msg, trace) -> (
        let full () =
          Sexp.List
            [
              Sexp.Atom "lib/shape.ml.Shape_error";
              sexp_of_string msg;
              sexp_of_list sexp_of_error_trace trace;
            ]
        in
        match if shape_error_full_sexp () then None else Some (error_to_string_hum msg trace) with
        | None -> full ()
        | Some text ->
            Sexp.List
              (Sexp.Atom "Shape_error"
              :: List.map (String.split_lines text) ~f:(fun l -> Sexp.Atom (String.strip l)))
        | exception _ -> full ())
    | _ -> assert false);
  (* What the toplevel prints for an uncaught exception, and [Printexc.to_string]: registered after
     Base's sexp printer, so tried first. *)
  Stdlib.Printexc.register_printer (function
    | Row.Shape_error (msg, trace) -> (
        try
          if shape_error_full_sexp () then None
          else Some ("Shape_error: " ^ error_to_string_hum msg trace)
        with _ -> None)
    | _ -> None)

(** {2 Shape builders.} *)

(* gh-ocannl-1970: a closure derives no projections for a step that has them, so projections that
   survived a change of the shapes they were derived over would reach a kernel iterating dims the
   buffers do not have. Only reached on a step that is committed by now, which no checkpoint's
   frontier holds, so the refusal is no read for [failed_reads]. *)
let refuse_stale_projections (step : update_step) (projections : Idx.projections) =
  let check ~what (sh : t) dims =
    let closed = to_dims_impl sh in
    if not (Array.equal Int.equal dims closed) then
      let text dims = Sexp.to_string ([%sexp_of: int array] dims) in
      raise
      @@ Row.Shape_error
           ( [%string
               "get_projections: the projections of the update step of #%{step.shape.id#Int} \
                %{step.shape.debug_name} are stale: they iterate dims %{text dims} of %{what}, \
                which closed to dims %{text closed}"],
             [ Shape_mismatch [ sh ] ] )
  in
  check ~what:"its result" step.shape projections.lhs_dims;
  List.iteri
    (fst (projected_operands step.logic))
    ~f:(fun i sh -> check ~what:[%string "operand %{i#Int}"] sh projections.rhs_dims.(i))

let get_projections (update_step : update_step) : Idx.projections =
  check_step_live update_step;
  let projections =
    match update_step.unsafe_projections with
    | Some projections when not (List.mem !active_update_steps update_step ~equal:phys_equal) ->
        (* Committed by an earlier closure. *)
        projections
    | _ -> (
        finish_for_read ~step:update_step update_step.shape;
        match update_step.unsafe_projections with
        | Some projections -> projections
        | None ->
            (* This shouldn't happen if finish_inference is working correctly, but derive
               projections just in case *)
            derive_projections update_step;
            Option.value_exn update_step.unsafe_projections)
  in
  refuse_stale_projections update_step projections;
  projections

(** [Input_equals_output] equates a fresh shape's input row with its output row. What [unify_row]
    cannot decide on the spot it returns rather than solves: an unpadded window whose kernel is
    unsolved, or a window pair held for the Stage2+ fixpoint (gh-ocannl-2038). Those join the active
    constraints, so the closure solves them with the rest -- or refuses them there, naming this
    equation as their origin (gh-ocannl-2030). *)
let deduce_within ~context deduced (result : t) =
  match deduced with
  | Not_constrained -> ()
  | Input_equals_output -> (
      try
        let origin =
          [
            Row.
              {
                lhs_name = result.debug_name;
                lhs_kind = `Input;
                rhs_name = result.debug_name;
                rhs_kind = `Output;
                operation = Some "input_equals_output";
              };
          ]
        in
        let deferred, env =
          Row.unify_row ~stage:Stage2 origin (result.input, result.output) !state
        in
        active_constraints := deferred @ !active_constraints;
        state := env
      with Row.Shape_error (s, trace) when !with_error_trace ->
        raise @@ Row.Shape_error (context ^ " / " ^ s, Shape_mismatch [ result ] :: trace))

let make ?batch_dims ?input_dims ?output_dims ?batch_axes ?input_axes ?output_axes
    ?(deduced = Not_constrained) ~debug_name ~id () =
  let open Row in
  let f _kind d =
    (* Provenance split at construction (brief §Technical-issue-1, AC#4): an unannotated user axis
       is a [default] atom at every size, INCLUDING size 1 — an explicit user [1] does not stretch
       (it flags a forgotten/hidden dimension rather than silently broadcasting). Only scalar
       helpers, internal broadcast fill (e.g. [embed_self_id]), and rank-broadening synthesizers
       mint the claim-free broadcast top [1_(bcast_if_1)], and they do so explicitly via [~*_axes] /
       [get_bcast_dim], not through this default path. *)
    get_default_dim ~d ()
  in
  let make_dims kind ds =
    {
      beg_dims = [];
      dims = List.map ~f:(f kind) ds;
      bcast = Broadcastable;
      prov = provenance ~sh_id:id ~kind;
    }
  in
  let make_axes kind ds =
    {
      beg_dims = [];
      dims = List.map ~f:(fun (basis, d) -> get_dim ~d ~basis ()) ds;
      bcast = Broadcastable;
      prov = provenance ~sh_id:id ~kind;
    }
  in
  let make_unknown kind =
    {
      dims = [];
      beg_dims = [];
      bcast = Row_var (get_row_var ());
      prov = provenance ~sh_id:id ~kind;
    }
  in
  let batch =
    match (batch_dims, batch_axes) with
    | Some batch_dims, None -> make_dims `Batch batch_dims
    | None, Some batch_axes -> make_axes `Batch batch_axes
    | None, None -> make_unknown `Batch
    | Some _, Some _ -> invalid_arg "Shape.make: do not provide both batch_dims, batch_axes"
  in
  let input =
    match (input_dims, input_axes) with
    | Some input_dims, None -> make_dims `Input input_dims
    | None, Some input_axes -> make_axes `Input input_axes
    | None, None -> make_unknown `Input
    | Some _, Some _ -> invalid_arg "Shape.make: do not provide both input_dims, input_axes"
  in
  let output =
    match (output_dims, output_axes) with
    | Some output_dims, None -> make_dims `Output output_dims
    | None, Some output_axes -> make_axes `Output output_axes
    | None, None -> make_unknown `Output
    | Some _, Some _ -> invalid_arg "Shape.make: do not provide both output_dims, output_axes"
  in
  let result = of_rows ~batch ~input ~output ~id ~debug_name in
  note_traced_name debug_name;
  deduce_within ~context:"Input_equals_output" deduced result;
  result

let shape_spec_to_dims_bio ~spec ~sh_id labels =
  let dim_var_env = Hashtbl.create (module String) in
  let f _kind = function
    | Label s when String.contains s '=' -> (
        let _label, dim =
          match String.split s ~on:'=' with
          | [ l; d ] -> (l, d)
          | _ -> invalid_arg "shape_spec_to_dims_bio: too many '='"
        in
        (* This is not a dimension basis i.e. unit! *)
        try Row.get_default_dim ~d:(Int.of_string dim) ()
        with _ -> invalid_arg "shape_spec_to_dims_bio: int expected after '='")
    | Label name ->
        Var (Hashtbl.find_or_add dim_var_env name ~default:(fun () -> Row.get_var ~name ()))
    | Fixed_index d -> Row.get_default_dim ~d ()
    | Affine_spec _ as axis_spec -> affine_dim_of_spec ~spec ~dim_var_env axis_spec
    | Concat_spec components ->
        Row.Concat (List.map components ~f:(concat_component_dim ~spec ~dim_var_env))
  in
  let row_var_env = Hashtbl.create (module String) in
  let result = axes_spec_to_dims_bio ~sh_id ~row_var_env ~dim_var_env ~f labels in
  check_dim_row_var_name_clash ~spec ~error_trace:[] dim_var_env row_var_env;
  result

let of_spec ?(deduced = Not_constrained) ~debug_name ~id spec =
  let batch, input, output = shape_spec_to_dims_bio ~spec ~sh_id:id @@ axis_labels_of_spec spec in
  let result = of_rows ~batch ~input ~output ~id ~debug_name in
  note_traced_name debug_name;
  deduce_within ~context:"of spec" deduced result;
  result

let to_string_hum ?(style = Row.Axis_size) (sh : t) =
  let n_outputs = List.length (sh.output.beg_dims @ sh.output.dims) in
  let n_batch = List.length (sh.batch.beg_dims @ sh.batch.dims) in
  let dims_to_string kind =
    let row = row_of_kind kind sh in
    let dims = row.beg_dims @ row.dims in
    String.concat ~sep:","
    @@ List.mapi dims ~f:(fun i d ->
        let num =
          match kind with `Input -> n_batch + n_outputs + i | `Output -> n_batch + i | `Batch -> i
        in
        match style with
        | Row.Only_bases | Axis_size | Projection_and_size -> Row.dim_to_string style d
        | Axis_number_and_size -> Int.to_string num ^ ":" ^ Row.dim_to_string style d)
  in
  let batch_dims = dims_to_string `Batch in
  let input_dims = dims_to_string `Input in
  let output_dims = dims_to_string `Output in
  if String.is_empty batch_dims && String.is_empty input_dims then output_dims
  else if String.is_empty batch_dims then input_dims ^ "->" ^ output_dims
  else if String.is_empty input_dims then batch_dims ^ "|" ^ output_dims
  else batch_dims ^ "|" ^ input_dims ^ "->" ^ output_dims

(** Given a fully-inferred shape, maps axes to their corresponding positions in an index using the
    [force_to_dims] semantics. *)
let axis_keys_to_idcs (sh : t) : int axis_map =
  if
    not
      (Row.is_broadcastable sh.input.bcast
      && Row.is_broadcastable sh.output.bcast
      && Row.is_broadcastable sh.batch.bcast)
  then raise @@ Utils.User_error "Shape.axis_keys_to_idcs: shape not fully inferred";
  let b_dims =
    (* Enumerate axes backwards across both flanks. *)
    Array.of_list_rev_mapi (sh.batch.beg_dims @ sh.batch.dims) ~f:(fun i _ ->
        AxisKey.{ in_axes = `Batch; pos = i + 1; from_end = true })
  in
  let i_dims =
    Array.of_list_rev_mapi (sh.input.beg_dims @ sh.input.dims) ~f:(fun i _ ->
        AxisKey.{ in_axes = `Input; pos = i + 1; from_end = true })
  in
  let o_dims =
    Array.of_list_rev_mapi (sh.output.beg_dims @ sh.output.dims) ~f:(fun i _ ->
        AxisKey.{ in_axes = `Output; pos = i + 1; from_end = true })
  in
  let idcs = Array.concat [ i_dims; o_dims; b_dims ] in
  Array.rev_inplace idcs;
  Map.of_alist_exn (module AxisKey) @@ Array.to_list @@ Array.mapi idcs ~f:(fun i key -> (key, i))

let%debug5_sexp default_display_indices (sh : t) : int array =
  let axes = axis_keys_to_idcs sh |> Map.map ~f:(fun _ -> 0) in
  let occupied = Array.create ~len:5 false in
  let set_occu prio =
    occupied.(prio + 5) <- true;
    prio
  in
  let occu prio = occupied.(prio + 5) in
  let num_input_axes = List.length (sh.input.beg_dims @ sh.input.dims) in
  let remaining =
    Stack.of_list
    @@ List.filter ~f:(Map.mem axes)
    @@ AxisKey.
         [
           { in_axes = `Input; from_end = true; pos = 1 };
           { in_axes = `Output; from_end = true; pos = 1 };
           { in_axes = `Input; from_end = true; pos = 2 };
           { in_axes = `Output; from_end = true; pos = 2 };
           (if num_input_axes > 1 then { in_axes = `Batch; from_end = true; pos = 1 }
            else { in_axes = `Output; from_end = true; pos = 3 });
           { in_axes = `Batch; from_end = true; pos = 1 };
           { in_axes = `Batch; from_end = true; pos = 2 };
           { in_axes = `Input; from_end = true; pos = 3 };
           { in_axes = `Output; from_end = true; pos = 3 };
           { in_axes = `Input; from_end = true; pos = 4 };
           { in_axes = `Output; from_end = true; pos = 4 };
           { in_axes = `Input; from_end = true; pos = 5 };
           { in_axes = `Output; from_end = true; pos = 5 };
         ]
  in
  let rec loop offset axes =
    if Stack.is_empty remaining || offset > 5 then axes
    else if Fn.non occu ~-offset then
      loop (offset + 1)
      @@ Map.change axes (Stack.pop_exn remaining) ~f:(Option.map ~f:(fun _ -> set_occu ~-offset))
    else loop (offset + 1) axes
  in
  let axes = loop 1 axes in
  axis_map_to_dims_index axes

let parse_n5_layout priorities =
  let f : Einsum_parser.axis_spec -> int = function
    | Fixed_index i -> i
    | Label _ -> invalid_arg "parse_n5_layout requires integer-only labels"
    | Affine_spec _ -> invalid_arg "parse_n5_layout does not support affine expressions"
    | Concat_spec _ -> invalid_arg "parse_n5_layout does not support concatenation expressions"
  in
  let p_labels = Einsum_parser.(axis_labels @@ axis_labels_of_spec priorities) in
  axis_map_to_dims_index p_labels |> Array.map ~f

(** {2 Shape schemes (gh-ocannl-404)}

    A scheme is read off the open session at the end of a trace (inside [Tensor.with_rewind]): the
    constraints the traced function registered, in registration order, and the solution equations of
    their variables under the trace's final environment. The equations matter: the solver mints
    variables while solving and writes them into shapes, and later constraints mention them, so a
    renamed replay of the constraints alone would come apart. Everything else -- the boundary rows,
    the internal rows closure finalizes, the rows of the pre-existing shapes the trace used -- is
    exported substituted under that environment, so it holds only unsolved variables, which the
    equations relate. Variables minted since the trace began are renamed per application
    (generalized); older ones stay (fixed) until a closure retires them, and the anchors re-tie them
    to their shapes' current rows, which keeps them valid across it: a closed anchor pins a fresh
    variable as it did the retired one. *)
module Scheme = struct
  type shape = t
  type rows = { r_batch : Row.t; r_input : Row.t; r_output : Row.t }

  type sabotage =
    | No_generalization
    | Sever_shared
    | No_marks
    | No_internal_rows
    | No_end_validation
    | No_ref_inputs
    | No_draw_holes
    | No_draw_refusal

  type nonrec t = {
    label : string;
    mark : int;  (** Variables with larger ids were minted by the trace: generalized. *)
    constraints : Row.constraint_ list;
        (** The trace's constraints, oldest first, then the solution equations. *)
    num_solutions : int;  (** How many of [constraints] are solution equations: the last ones. *)
    arg_rows : rows list;
    result_rows : rows list;
    internal_rows : (Row.t * Row.constraint_origin) list;
        (** The rows of every shape the trace's steps (and the schemes it applied) held: closure's
            [Shape_row] stage finalizes non-terminal variables, an intermediate's too. *)
    anchors : (shape * rows) list;
        (** Pre-existing open shapes the trace used, with their rows at the end of the trace. *)
    marks : Row.marks;
    discardable : Row.dim_var list;
    unmarked_unused : int list;
    closure_bound : string option;
        (** Fixed variables no anchor covers: the scheme cannot be instantiated after a closure. *)
    closure_at_trace : int;
    env : Row.environment;  (** The trace's final environment: for printing the scheme. *)
    ref_inputs : ref_input list;
        (** The capture references made before the trace that it bound (gh-ocannl-1791), in first
            binding order: inputs, related at each application to their state then. *)
    tensor_names : Set.M(String).t;
        (** The scheme's own tensors, by debug name (gh-ocannl-1903): its label, its arguments, the
            shapes its trace made and the tensors of the schemes the trace applied. An error at an
            application ranks the origins naming them first. *)
    draws : int;
        (** The random draws (packed uint4x32 conversions) the trace registered (gh-ocannl-1890):
            their coefficients are holes, which an explicit instance cannot fill. *)
  }

  let label s = s.label
  let arity s = List.length s.arg_rows
  let num_results s = List.length s.result_rows

  (* A boundary (argument or result) or fixed variable, as the relations relate them; the
     intermediates between them are eliminated. *)
  type var = D of Row.dim_var | R of Row.row_var

  (* Variables are named by first appearance, so the text does not depend on the session's id
     counters: generalized ones ['a], ['b], ...; fixed ones -- shared with what the function closes
     over, e.g. a lifted parameter -- ['_a], ['_b], ..., as OCaml prints weak type variables.
     Dimension and row variables share the alphabet. The indented lines are the broadcasting
     inequalities among the boundary's variables and the fixed ones under the trace's final
     environment, with the function's intermediate variables eliminated along monotone chains:
     grouped by left side, [v ⊑ w1, w2, ...] for every boundary variable [v] and for every fixed
     variable refining one, then [_ ⊑ w1, w2, ...] for the variables an eliminated intermediate
     refines together when no printed variable does (gh-ocannl-1988: arguments consumed together by
     a reduction), then the lower bounds [b ⊑ v]. The order is canonical and name-independent -- by
     first appearance in the header for the boundary's variables, and by their relations to those
     for the fixed ones -- so that the text does not follow the solver's traversal either, and
     naming follows the order. *)
  let to_string s =
    let names = Hashtbl.create (module String) in
    let counts = [| 0; 0 |] in
    let appearance = ref 0 in
    (* [names] holds (fixed, the letter index, the appearance order). *)
    let name ~fixed key =
      let _, n, _ =
        Hashtbl.find_or_add names key ~default:(fun () ->
            let k = if fixed then 1 else 0 in
            let n = counts.(k) in
            counts.(k) <- n + 1;
            let a = !appearance in
            Int.incr appearance;
            (fixed, n, a))
      in
      (if fixed then "'_" else "'") ^ letter_name n
    in
    let dim_key v = "d" ^ Int.to_string (Row.dim_var_id v) in
    let row_key v = "r" ^ Int.to_string (Row.row_var_id v) in
    let dim_fixed v = Row.dim_var_id v <= s.mark in
    let row_fixed v = Row.row_var_id v <= s.mark in
    let dim_name v = name ~fixed:(dim_fixed v) (dim_key v) in
    let row_name v = ".." ^ name ~fixed:(row_fixed v) (row_key v) ^ ".." in
    let dim_str = Row.dim_to_string ~var_to_string:dim_name Row.Axis_size in
    (* Sequenced: names are given in reading order. *)
    let row_str (r : Row.t) =
      let beg = List.map r.beg_dims ~f:dim_str in
      let middle = match r.bcast with Row.Row_var v -> [ row_name v ] | Broadcastable -> [] in
      let ends = List.map r.dims ~f:dim_str in
      match beg @ middle @ ends with [] -> "()" | items -> String.concat ~sep:"," items
    in
    let shape_str { r_batch; r_input; r_output } =
      let b = row_str r_batch in
      let i = row_str r_input in
      let o = row_str r_output in
      [%string "%{b}|%{i}->%{o}"]
    in
    let side rows = String.concat ~sep:", " (List.map rows ~f:shape_str) in
    let args = side s.arg_rows in
    (* [_]: the scheme constrains no result. *)
    let results = match s.result_rows with [] -> "_" | rows -> side rows in
    let header = [%string "%{s.label} : %{args} => %{results}"] in
    (* The capture inputs (gh-ocannl-1791): what each first binding is under the final
       environment. *)
    let captures =
      List.map s.ref_inputs ~f:(fun i ->
          let label = i.input_ref.var_ref.ref_label in
          let bound =
            match i.first with
            | `Dim v -> dim_str (Option.value (Row.get_dim_from_env s.env v) ~default:(Row.Var v))
            | `Row v -> row_str (Row.subst_row s.env (Row.get_row_for_var Row.empty_provenance v))
          in
          [%string "capture %{label} = %{bound}"])
    in
    let key = function D v -> dim_key v | R v -> row_key v in
    let fixed = function D v -> dim_fixed v | R v -> row_fixed v in
    let pr = function D v -> dim_name v | R v -> row_name v in
    let id = function D v -> Row.dim_var_id v | R v -> Row.row_var_id v in
    let same v w = String.equal (key v) (key w) in
    let bounds = function
      | D v -> (
          match Row.find_dim_entry s.env v with
          | Some (Row.Bounds_dim { res; opnd; _ }) ->
              (List.map res ~f:(fun v -> D v), List.map opnd ~f:(fun v -> D v))
          | _ -> ([], []))
      | R v -> (
          match Row.find_row_entry s.env v with
          | Some (Row.Bounds_row { res; opnd; _ }) ->
              (List.map res ~f:(fun v -> R v), List.map opnd ~f:(fun v -> R v))
          | _ -> ([], []))
    in
    let dedup vars =
      List.fold vars ~init:[] ~f:(fun acc v ->
          if List.exists acc ~f:(same v) then acc else v :: acc)
      |> List.rev
    in
    (* In reading order ([Row.row_vars] sorts by id, which a lower bound's variables must not
       follow). *)
    let rec vars_of_dim = function
      | Row.Var v -> [ D v ]
      | Dim _ | Sym _ -> []
      | Affine { over; conv; _ } ->
          vars_of_dim over
          @ Option.value_map conv ~default:[] ~f:(fun c -> vars_of_dim c.Row.kernel)
      | Concat dims -> List.concat_map dims ~f:vars_of_dim
    in
    let vars_of_row (r : Row.t) =
      List.concat_map r.beg_dims ~f:vars_of_dim
      @ (match r.bcast with Row.Row_var v -> [ R v ] | Broadcastable -> [])
      @ List.concat_map r.dims ~f:vars_of_dim
    in
    let boundary =
      dedup
        (List.concat_map (s.arg_rows @ s.result_rows) ~f:(fun { r_batch; r_input; r_output } ->
             List.concat_map [ r_batch; r_input; r_output ] ~f:vars_of_row))
    in
    (* The lower bounds of the boundary's variables, with the variables each names. *)
    let glbs =
      List.filter_map boundary ~f:(fun v ->
          match v with
          | D d -> (
              match Row.find_dim_entry s.env d with
              | Some (Row.Bounds_dim { glb = Some b; _ }) ->
                  Some ((fun () -> dim_str b), vars_of_dim b, v)
              | _ -> None)
          | R r -> (
              match Row.find_row_entry s.env r with
              | Some (Row.Bounds_row { glb = Some b; _ }) ->
                  Some ((fun () -> row_str b), vars_of_row b, v)
              | _ -> None))
    in
    let mem vars v = List.exists vars ~f:(same v) in
    (* The printed variables: the boundary's, and the generalized ones a lower bound names
       (gh-ocannl-1988: the class dimension of a reduced axis, below one argument's row as its lower
       bound while refining the other's axis). The chains stop at them and at the fixed ones; the
       variables in between are the eliminated intermediates. *)
    let glb_vars =
      List.concat_map glbs ~f:(fun (_, vars, _) -> vars)
      |> dedup
      |> List.filter ~f:(fun v -> not (mem boundary v || fixed v))
    in
    let printed = boundary @ glb_vars in
    let endpoint v = mem printed v || fixed v in
    (* The intermediates the searches pass through. *)
    let intermediates = Hashtbl.create (module String) in
    (* Monotone chains from [start], through the intermediates: upward to every variable it refines,
       downward to the ones refining it; the endpoints found, deduplicated. *)
    let search ~up start =
      let visited = Hash_set.create (module String) in
      let found = ref [] in
      let rec go v =
        if not (Hash_set.mem visited (key v)) then (
          Hash_set.add visited (key v);
          let res, opnd = bounds v in
          List.iter
            (if up then opnd else res)
            ~f:(fun w ->
              if same w start then ()
              else if endpoint w then found := w :: !found
              else (
                Hashtbl.set intermediates ~key:(key w) ~data:w;
                go w)))
      in
      go start;
      dedup (List.rev !found)
    in
    (* [v ⊑ w]: a printed variable's upward endpoints; of its downward endpoints, the fixed ones (a
       printed one is found by its own upward search). *)
    let edges =
      List.concat_map printed ~f:(fun v ->
          List.map (search ~up:true v) ~f:(fun w -> (v, w))
          @ List.filter_map (search ~up:false v) ~f:(fun w ->
              if mem printed w then None else Some (w, v)))
    in
    (* Grouped by left side: [(left, rights)]. *)
    let groups =
      List.fold edges ~init:[] ~f:(fun acc (l, r) ->
          match List.partition_tf acc ~f:(fun (l', _) -> same l l') with
          | [ (_, rs) ], rest -> (l, rs @ [ r ]) :: rest
          | _ -> (l, [ r ]) :: acc)
      |> List.rev_map ~f:(fun (l, rs) -> (l, dedup rs))
    in
    let subset vs ws = List.for_all vs ~f:(mem ws) in
    (* What an eliminated intermediate refines together, when no printed variable does: a group's
       left side witnesses its own rights. The intermediates are those the searches passed through,
       so each refines a printed variable or is refined by one. *)
    let commons =
      Hashtbl.data intermediates
      |> List.map ~f:(search ~up:true)
      |> List.filter ~f:(fun ws -> List.length ws >= 2)
      |> List.filter ~f:(fun ws -> not (List.exists groups ~f:(fun (l, rs) -> subset ws (l :: rs))))
      |> List.fold ~init:[] ~f:(fun acc ws ->
          if List.exists acc ~f:(fun ws' -> subset ws ws') then acc
          else ws :: List.filter acc ~f:(fun ws' -> not (subset ws' ws)))
    in
    (* The canonical order. A variable the header or a capture named sorts by its appearance; one a
       lower bound names after every other; a fixed one not yet named by its relations to the named
       ones, then by id (fixed ids precede the trace, so they follow the program, not the
       solver). *)
    let named v = Option.map (Hashtbl.find names (key v)) ~f:(fun (_, _, a) -> a) in
    let signature v =
      let tag t w = Option.map (named w) ~f:(fun a -> (t, a)) in
      List.concat_map groups ~f:(fun (l, rs) ->
          if same l v then List.filter_map rs ~f:(tag 0)
          else if mem rs v then Option.to_list (tag 1 l)
          else [])
      @ List.concat_map commons ~f:(fun ws ->
          if mem ws v then List.filter_map ws ~f:(tag 2) else [])
      |> List.sort ~compare:[%compare: int * int]
    in
    let unnamed =
      List.concat_map groups ~f:(fun (l, rs) -> l :: rs) @ List.concat commons
      |> dedup
      |> List.filter ~f:(fun v -> Option.is_none (named v) && not (mem glb_vars v))
      |> List.map ~f:(fun v -> (signature v, id v, v))
      |> List.sort ~compare:(fun (s1, i1, _) (s2, i2, _) ->
          [%compare: (int * int) list * int] (s1, i1) (s2, i2))
      |> List.mapi ~f:(fun i (_, _, v) -> (key v, i))
      |> Hashtbl.of_alist_exn (module String)
    in
    let order v =
      match named v with
      | Some a -> (0, a)
      | None -> (
          match List.findi glb_vars ~f:(fun _ w -> same v w) with
          | Some (i, _) -> (2, i)
          | None -> (1, Hashtbl.find_exn unnamed (key v)))
    in
    let by_order = List.sort ~compare:(fun v w -> [%compare: int * int] (order v) (order w)) in
    let rights ws = String.concat ~sep:", " (List.map (by_order ws) ~f:pr) in
    let lines =
      List.sort groups ~compare:(fun (l, _) (l', _) -> [%compare: int * int] (order l) (order l'))
      |> List.map ~f:(fun (l, rs) -> fun () -> [%string "%{pr l} ⊑ %{rights rs}"])
    in
    let commons =
      List.map commons ~f:(fun ws -> (List.map (by_order ws) ~f:order, ws))
      |> List.sort ~compare:(fun (o, _) (o', _) -> [%compare: (int * int) list] o o')
      |> List.map ~f:(fun (_, ws) -> fun () -> [%string "_ ⊑ %{rights ws}"])
    in
    let glbs =
      List.sort glbs ~compare:(fun (_, _, v) (_, _, w) -> [%compare: int * int] (order v) (order w))
      |> List.map ~f:(fun (b, _, v) -> fun () -> [%string "%{b ()} ⊑ %{pr v}"])
    in
    (* Rendered in order: names follow the reading order. *)
    String.concat ~sep:"\n"
      ((header :: List.map captures ~f:(fun r -> "  " ^ r))
      @ List.map (lines @ commons @ glbs) ~f:(fun r -> "  " ^ r ()))

  let rows_of env (sh : shape) =
    {
      r_batch = Row.subst_row env sh.batch;
      r_input = Row.subst_row env sh.input;
      r_output = Row.subst_row env sh.output;
    }

  let rows_vars { r_batch; r_input; r_output } =
    List.map [ r_batch; r_input; r_output ] ~f:Row.row_vars

  (* The elements of [l] in front of its physical suffix [mark], in [l]'s (newest-first) order. *)
  let window ~mark l =
    let rec go acc = function
      | l when phys_equal l mark -> List.rev acc
      | [] -> List.rev acc
      | x :: rest -> go (x :: acc) rest
    in
    go [] l

  (* The first occurrence of each shape, physically: an id can name several, as in [open_shapes]. *)
  let dedup_shapes shapes =
    let seen = Hashtbl.create (module Int) in
    List.filter shapes ~f:(fun (sh : shape) ->
        (not (List.mem (Hashtbl.find_multi seen sh.id) sh ~equal:phys_equal))
        &&
        (Hashtbl.add_multi seen ~key:sh.id ~data:sh;
         true))

  let trace ?sabotage ~var_mark ~label ~(args : shape list) (f : unit -> shape list) =
    let sabotaged x = Option.exists sabotage ~f:(fun y -> Poly.equal x y) in
    let constraints_mark = !active_constraints in
    let steps_mark = !active_update_steps in
    let instances_mark = !active_instances in
    let row_globals = Row.capture_globals () in
    let env_start = !state in
    let unused_start = !unused_shapes in
    let refs_mark = !ref_uid in
    let outer_touches = !trace_touches in
    let frontier = Hashtbl.create (module Int) in
    List.iter (Checkpoint.open_shapes ~instances:!active_instances !active_update_steps)
      ~f:(fun (sh : shape) -> Hashtbl.add_multi frontier ~key:sh.id ~data:sh);
    let pre_existing (sh : shape) =
      (not sh.closed)
      && List.mem (Hashtbl.find_multi frontier sh.id) sh ~equal:phys_equal
      && not (List.mem args sh ~equal:phys_equal)
    in
    let refuse msg = raise @@ Untraceable_scheme msg in
    let frame = { frame_refs_mark = refs_mark; frame_inputs = [] } in
    let outer_frame = !trace_ref_frame in
    let names =
      Hash_set.of_list (module String) (label :: List.map args ~f:(fun sh -> sh.debug_name))
    in
    try
      with_traced_names names @@ fun () ->
      trace_touches := Some [];
      trace_ref_frame := Some frame;
      let results, touched =
        Exn.protect
          ~f:(fun () ->
            as_tracing ~label (fun () ->
                let results = f () in
                (results, Option.value !trace_touches ~default:[])))
          ~finally:(fun () ->
            trace_touches := outer_touches;
            trace_ref_frame := outer_frame)
      in
      let ref_inputs = List.rev frame.frame_inputs in
      let constraints = List.rev (window ~mark:constraints_mark !active_constraints) in
      if List.exists constraints ~f:Row.has_lazy_coefficient then
        refuse
          "the function registers a total-elements constraint whose coefficient is a lazy of \
           unknown provenance, which a scheme cannot carry past the trace -- apply the function \
           inline";
      (* Random draws (gh-ocannl-1890): a packed conversion's coefficient is a lazy over its result
         node's precision, which the node's consumers set as they register and closure forces. The
         scheme never reads it -- a trace, a check and an instance solve Stage1 only, which reads no
         lazy coefficient -- so it carries a hole in its place, holding none of the trace's nodes;
         an application's own conversion node, registered by the body, is what fills it. *)
      let holes = Hashtbl.create (module String) in
      let hole (coeff : int Utils.safe_lazy) =
        if sabotaged No_draw_holes then coeff
        else
          (* Only the id is kept: the hole must not close over [coeff], a lazy over the trace's
             node. *)
          let id = coeff.unique_id in
          Hashtbl.find_or_add holes id ~default:(fun () ->
              Utils.safe_lazy ("draw hole of " ^ id) (fun () ->
                  raise
                  @@ Row.Shape_error
                       ( [%string
                           "the shape scheme %{label} read the coefficient of one of its random \
                            draws (%{id}): a scheme carries a hole in its place, which only the \
                            conversion node an application's body registers fills"],
                         [] )))
      in
      let constraints = List.map constraints ~f:(Row.map_draw_coefficients ~f:hole) in
      let draws = Hashtbl.length holes in
      (* A backstop: the setters refuse an outside reference themselves, and a binding is an
         input. *)
      if
        List.exists touched ~f:(fun uid ->
            uid <= refs_mark && not (List.exists ref_inputs ~f:(fun i -> i.input_ref.uid = uid)))
      then
        refuse
          "the function writes a capture reference made outside it other than by binding it in an \
           operation, which a scheme cannot carry per application -- make the reference inside the \
           function, or apply the function inline";
      let marks = Row.marks_since row_globals in
      (* Validation: the trailing setters ([set_dim], [infer_equal], [set_terminal]) push
         constraints without solving them, so the whole window is solved once more -- Stage1 only:
         the closing stages would default the generic arguments. *)
      let env_end =
        if sabotaged No_end_validation then !state
        else snd (Row.solve_inequalities ~stage:Row.Stage1 constraints !state)
      in
      let unused_end = !unused_shapes in
      (* As [finish_inference] filters the terminal rows of shapes still marked unused: the trace's
         own such shapes can never be used afterwards. *)
      let own_unused = Set.diff unused_end unused_start in
      let constraints =
        List.filter constraints ~f:(function
          | Row.Shape_row (r, _) | Terminal_row (_, r, _) ->
              not (List.exists (Row.row_shapes r) ~f:(Set.mem own_unused))
          | _ -> true)
      in
      let solution_origin =
        Row.
          {
            lhs_name = label;
            lhs_kind = `Output;
            rhs_name = "(its trace's solution)";
            rhs_kind = `Output;
            operation = Some [%string "the trace of the shape scheme %{label}"];
          }
      in
      let dim_vars, row_vars =
        List.fold constraints
          ~init:(Set.empty (module Int), [], Set.empty (module Int), [])
          ~f:(fun acc c ->
            let ds, rs = Row.constraint_vars c in
            let add_d (seen, l, rseen, rl) v =
              let id = Row.dim_var_id v in
              if Set.mem seen id then (seen, l, rseen, rl) else (Set.add seen id, v :: l, rseen, rl)
            in
            let add_r (seen, l, rseen, rl) v =
              let id = Row.row_var_id v in
              if Set.mem rseen id then (seen, l, rseen, rl) else (seen, l, Set.add rseen id, v :: rl)
            in
            List.fold rs ~init:(List.fold ds ~init:acc ~f:add_d) ~f:add_r)
        |> fun (_, dl, _, rl) -> (List.rev dl, List.rev rl)
      in
      let solutions =
        List.filter_map dim_vars ~f:(fun v ->
            Option.map (Row.get_dim_from_env env_end v) ~f:(fun d ->
                Row.Dim_eq { d1 = Row.Var v; d2 = d; origin = [ solution_origin ] }))
        @ List.filter_map row_vars ~f:(fun v ->
            Option.map (Row.get_row_from_env env_end v) ~f:(fun r ->
                Row.Row_eq
                  {
                    r1 = Row.get_row_for_var Row.empty_provenance v;
                    r2 = r;
                    origin = [ solution_origin ];
                  }))
      in
      let steps = List.rev (window ~mark:steps_mark !active_update_steps) in
      (* A scheme the function applied registered no step: its instance holds its boundary and
         anchored shapes, and its rows. *)
      let instances = List.rev (window ~mark:instances_mark !active_instances) in
      let used_shapes =
        List.concat_map steps ~f:(fun step ->
            let acc = ref [] in
            iter_shapes step ~f:(fun sh -> acc := sh :: !acc);
            List.rev !acc)
        @ List.concat_map instances ~f:(fun inst -> inst.inst_shapes)
        @ results
      in
      let internal_rows =
        List.concat_map steps ~f:all_rows_w_origin
        @ List.concat_map instances ~f:(fun inst -> inst.inst_rows)
        |> List.map ~f:(fun (r, o) -> (Row.subst_row env_end r, o))
        |> List.dedup_and_sort ~compare:(fun (r1, _) (r2, _) -> Row.compare r1 r2)
      in
      let anchors =
        List.filter (dedup_shapes used_shapes) ~f:pre_existing
        |> List.map ~f:(fun sh -> (sh, rows_of env_end sh))
      in
      let arg_rows = List.map args ~f:(rows_of env_end) in
      let result_rows = List.map results ~f:(rows_of env_end) in
      (* Coverage: a fixed variable left unsolved dies at the next closure; an anchor re-ties it to
         its shape's closed rows then. One no anchor covers makes the scheme closure-bound. *)
      let fixed_unsolved =
        List.concat_map
          (List.map (constraints @ solutions) ~f:Row.constraint_vars
          @ List.concat_map (arg_rows @ result_rows) ~f:rows_vars
          @ List.map internal_rows ~f:(fun (r, _) -> Row.row_vars r))
          ~f:(fun (ds, rs) ->
            List.filter_map ds ~f:(fun v ->
                if Row.dim_var_id v <= var_mark && Option.is_none (Row.get_dim_from_env env_end v)
                then Some (`Dim (Row.dim_var_id v))
                else None)
            @ List.filter_map rs ~f:(fun v ->
                if Row.row_var_id v <= var_mark && Option.is_none (Row.get_row_from_env env_end v)
                then Some (`Row (Row.row_var_id v))
                else None))
      in
      let covered =
        Hash_set.Poly.of_list
          (List.concat_map anchors ~f:(fun (_, rows) ->
               List.concat_map (rows_vars rows) ~f:(fun (ds, rs) ->
                   List.map ds ~f:(fun v -> `Dim (Row.dim_var_id v))
                   @ List.map rs ~f:(fun v -> `Row (Row.row_var_id v)))))
      in
      let uncovered =
        List.filter fixed_unsolved ~f:(fun v -> not (Hash_set.mem covered v))
        |> List.dedup_and_sort ~compare:Poly.compare
      in
      let closure_bound =
        if List.is_empty uncovered then None
        else
          Some
            (String.concat ~sep:", "
               (List.map uncovered ~f:(function
                 | `Dim id -> "dimension variable $" ^ Int.to_string id
                 | `Row id -> "row variable $" ^ Int.to_string id)))
      in
      {
        label;
        mark = var_mark;
        constraints = constraints @ solutions;
        num_solutions = List.length solutions;
        arg_rows;
        result_rows;
        internal_rows;
        anchors;
        marks;
        discardable =
          Set.to_list (Set.diff (Row.discardable_vars env_end) (Row.discardable_vars env_start));
        unmarked_unused = Set.to_list (Set.diff unused_start unused_end);
        closure_bound;
        closure_at_trace = !closure_count;
        env = Row.map_env_draw_coefficients ~f:hole env_end;
        ref_inputs;
        tensor_names = Set.of_list (module String) (Hash_set.to_list names);
        draws;
      }
    with
    | Row.Shape_error (msg, trace) ->
        let bt = Stdlib.Printexc.get_raw_backtrace () in
        Stdlib.Printexc.raise_with_backtrace
          (Row.Shape_error ([%string "tracing the shape scheme %{label}: %{msg}"], trace))
          bt
    | Untraceable_scheme msg ->
        let bt = Stdlib.Printexc.get_raw_backtrace () in
        Stdlib.Printexc.raise_with_backtrace
          (Untraceable_scheme [%string "tracing the shape scheme %{label}: %{msg}"])
          bt

  let stale s = !closure_count <> s.closure_at_trace && Option.is_some s.closure_bound

  let same_binding (b1 : ref_binding) (b2 : ref_binding) =
    match (b1, b2) with
    | `Dim v1, `Dim v2 -> Row.equal_dim_var v1 v2
    | `Row v1, `Row v2 -> Row.equal_row_var v1 v2
    | _ -> false

  (* [results]: [None] relates none of the scheme's results (a check, which has only the
     arguments). *)
  let instantiate_core ?sabotage ~instance ~bind_refs s ~application ~(args : shape list)
      ~(results : shape list option) =
    check_untainted ();
    let sabotaged x = Option.exists sabotage ~f:(fun y -> Poly.equal x y) in
    let prefix = [%string "applying the shape scheme %{s.label} to %{application}: "] in
    let fail msg = raise @@ Row.Shape_error (prefix ^ msg, []) in
    let check_count what expected got =
      if expected <> got then
        fail [%string "the scheme has %{expected#Int} %{what}, the application gives %{got#Int}"]
    in
    check_count "arguments" (arity s) (List.length args);
    let results, result_rows =
      match results with
      | None -> ([], [])
      | Some results ->
          check_count "results" (num_results s) (List.length results);
          (results, s.result_rows)
    in
    (* An explicit instance stands for the body until closure, which forces every draw's
       coefficient; it has no conversion node to fill the scheme's holes with. A check ([bind_refs]
       false) solves Stage1 only and is put back before the body registers its own. *)
    (if bind_refs && s.draws > 0 && not (sabotaged No_draw_refusal) then
       let msg =
         [%string
           "the scheme draws random numbers (%{s.draws#Int} packed conversion(s)), whose counter \
            size is the precision the conversion node of the body's own application settles; an \
            explicit instance has no such node -- apply the function inline"]
       in
       if tracing_in_progress () then raise @@ Untraceable_scheme (prefix ^ msg) else fail msg);
    List.iter
      (args @ results @ List.map s.anchors ~f:fst)
      ~f:(fun (sh : shape) ->
        Option.iter sh.discarded_by ~f:(fun provenance ->
            fail
              [%string
                "the shape of #%{sh.id#Int} %{sh.debug_name} was discarded by the rewind after \
                 %{provenance}"]));
    if !closure_count <> s.closure_at_trace then
      Option.iter s.closure_bound ~f:(fun vars ->
          fail
            [%string
              "inference closed since the trace, and the scheme depends on open state no shape it \
               used holds (%{vars}) -- trace it again after the closure"]);
    (* A variable from before the trace is shared by the applications while it is open. Once a
       closure retired it, the check above leaves only variables of the anchors' shapes, now ground,
       which the anchor equations below pin whatever the variable is called: each application gets a
       fresh one, so a retired variable never re-enters the solver (gh-ocannl-1859). *)
    let closed_since_trace = !closure_count <> s.closure_at_trace in
    let generalize id =
      if sabotaged No_generalization then false
      else sabotaged Sever_shared || closed_since_trace || id > s.mark
    in
    let dim_tbl = Hashtbl.create (module Int) and row_tbl = Hashtbl.create (module Int) in
    let rn =
      Row.
        {
          rn_dim =
            (fun v ->
              let id = Row.dim_var_id v in
              if generalize id then
                Hashtbl.find_or_add dim_tbl id ~default:(fun () -> Row.get_var ())
              else v);
          rn_row =
            (fun v ->
              let id = Row.row_var_id v in
              if generalize id then Hashtbl.find_or_add row_tbl id ~default:Row.get_row_var else v);
        }
    in
    let origin =
      Row.
        {
          lhs_name = s.label;
          lhs_kind = `Output;
          rhs_name = application;
          rhs_kind = `Output;
          operation = Some [%string "the shape scheme %{s.label} applied to %{application}"];
        }
    in
    let equate rows (sh : shape) =
      List.map
        [
          (`Batch, rows.r_batch, sh.batch);
          (`Input, rows.r_input, sh.input);
          (`Output, rows.r_output, sh.output);
        ]
        ~f:(fun (kind, r, actual) ->
          Row.Row_eq
            {
              r1 = Row.rename_row rn r;
              r2 = actual;
              origin =
                [ { origin with lhs_kind = kind; rhs_kind = kind; rhs_name = sh.debug_name } ];
            })
    in
    let constraints =
      List.map s.constraints ~f:(fun c ->
          Row.add_constraint_origin origin (Row.rename_constraint rn c))
    in
    let boundary =
      List.concat (List.map2_exn s.arg_rows args ~f:equate)
      @ List.concat (List.map2_exn result_rows results ~f:equate)
    in
    let anchors =
      if sabotaged Sever_shared then []
      else List.concat_map s.anchors ~f:(fun (sh, r) -> equate r sh)
    in
    let all = constraints @ boundary @ anchors in
    (* The anchored shapes too: a trace that applies this scheme must see them as used. *)
    let inst_shapes = dedup_shapes (args @ results @ List.map s.anchors ~f:fst) in
    let inst_rows =
      (if sabotaged No_internal_rows then []
       else List.map s.internal_rows ~f:(fun (r, o) -> (Row.rename_row rn r, o)))
      @ List.concat_map inst_shapes ~f:(fun (sh : shape) ->
          List.map
            [ (`Batch, sh.batch); (`Input, sh.input); (`Output, sh.output) ]
            ~f:(fun (kind, r) -> (r, { origin with lhs_kind = kind; rhs_kind = kind })))
    in
    let discardable_vars =
      List.fold s.discardable ~init:Row.dim_var_set_empty ~f:(fun acc v ->
          Set.add acc (rn.rn_dim v))
    in
    let ref_inputs = if sabotaged No_ref_inputs then [] else s.ref_inputs in
    let rename_binding : ref_binding -> ref_binding = function
      | `Dim v -> `Dim (rn.rn_dim v)
      | `Row v -> `Row (rn.rn_row v)
    in
    (* The scheme's tensors are those of a trace applying it too. *)
    if not (List.is_empty !traced_names) then Set.iter s.tensor_names ~f:note_traced_name;
    try
      Row.with_origin_focus (names_origin (Set.mem s.tensor_names)) @@ fun () ->
      with_taint_on_failure
        ~describe:(fun exn -> prefix ^ exn_summary exn)
        (fun () ->
          (* The capture inputs (gh-ocannl-1791): the first binding of each, renamed, related to the
             reference's state at this application, as the body's first binding of it would be. An
             instance also binds it, to the last one, so that the closure publishes its solution; a
             check writes no reference. *)
          let ref_constraints =
            List.concat_map ref_inputs ~f:(fun i ->
                let r = i.input_ref in
                let first = rename_binding i.first in
                check_before_binding ~where:[%string "in the shape scheme %{s.label}"]
                  ~error_shapes:args
                  ~is_row:(match first with `Row _ -> true | `Dim _ -> false)
                  r;
                if bind_refs then
                  bind_ref ~for_projections:false ~error_shapes:args r first
                  @
                  if same_binding i.first i.last then []
                  else bind_ref ~for_projections:false ~error_shapes:args r (rename_binding i.last)
                else
                  let previous_binding, carried_row = previous_of r in
                  binding_constraints ~for_projections:false ~as_input:(is_trace_input r)
                    ~error_shapes:args ~previous_binding ~carried_row r first)
            |> List.map ~f:(Row.add_constraint_origin origin)
          in
          let all = all @ ref_constraints in
          let inst_refs = if bind_refs then List.map ref_inputs ~f:(fun i -> i.input_ref) else [] in
          List.iter s.unmarked_unused ~f:unmark_unused;
          if not (sabotaged No_marks) then Row.add_marks rn s.marks;
          active_constraints := List.rev_append all !active_constraints;
          (let c = !counts and n = List.length constraints in
           counts :=
             {
               c with
               instantiated_constraints = c.instantiated_constraints + n - s.num_solutions;
               instantiated_solutions = c.instantiated_solutions + s.num_solutions;
               instantiated_boundary =
                 c.instantiated_boundary + List.length boundary + List.length anchors
                 + List.length ref_constraints;
             });
          if instance then
            active_instances := { inst_shapes; inst_rows; inst_refs } :: !active_instances;
          let _, env = Row.solve_inequalities ~stage:Row.Stage1 ~discardable_vars all !state in
          state := env;
          List.iter inst_shapes ~f:(apply_env_t env);
          (* As [propagate_inline] publishes a step's references after its Stage1 solve. *)
          publish_ref_solutions ~closure:false env inst_refs);
      (rn, origin)
    with Row.Shape_error (msg, trace) ->
      let bt = Stdlib.Printexc.get_raw_backtrace () in
      Stdlib.Printexc.raise_with_backtrace (Row.Shape_error (prefix ^ msg, trace)) bt

  let instantiate ?sabotage s ~application ~args ~results =
    ignore
      (instantiate_core ?sabotage ~instance:true ~bind_refs:true s ~application ~args
         ~results:(Some results)
        : Row.renaming * Row.constraint_origin)

  (* What [instantiate] writes, put back: the session globals, the solver's, the taint, and the rows
     of the shapes it substitutes the solution into (the arguments and the anchored shapes). *)
  let restore_after_check ~globals ~row_globals ~taint:taint0 rows =
    restore_globals globals;
    Row.restore_globals row_globals;
    taint := taint0;
    List.iter rows ~f:(fun (sh, batch, input, output) -> set_rows sh ~batch ~input ~output)

  let check ?sabotage s ~application ~args =
    let globals = capture_globals () and row_globals = Row.capture_globals () and taint0 = !taint in
    let rows =
      List.map
        (args @ List.map s.anchors ~f:fst)
        ~f:(fun (sh : shape) -> (sh, sh.batch, sh.input, sh.output))
    in
    timed `Instantiation (fun () ->
        Exn.protect
          ~f:(fun () ->
            ignore
              (instantiate_core ?sabotage ~instance:true ~bind_refs:false s ~application ~args
                 ~results:None
                : Row.renaming * Row.constraint_origin))
          ~finally:(fun () -> restore_after_check ~globals ~row_globals ~taint:taint0 rows))
end
