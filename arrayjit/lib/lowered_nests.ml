(* The nest view of raw lowered code shared by the rewrite tier's recognizers; the contract is in
   the interface. *)

open Base
module LL = Low_level
module Tn = Tnode
module Idx = Indexing

let ( let* ) x f = Option.bind x ~f

(* {1 Nests}

   Every [Accum_op] lowers to one top-level nest: serial loops from the outside in and a single
   [Set] at the bottom. The recognizers read nests, never raw statements, so a guard, a scan or a
   hardware-annotated loop anywhere in a statement makes it invisible to them. *)

type loop = { index : Idx.symbol; from_ : int; to_ : int }
type nest = { loops : loop list; tn : Tn.t; idcs : Idx.axis_index array; llsc : LL.scalar_t }

let nest_of (stmt : LL.t) : nest option =
  let rec go loops = function
    | LL.For_loop { index; from_; to_; body; axis = LL.Serial } ->
        go ({ index; from_; to_ } :: loops) body
    | LL.Set { tn; idcs; llsc; _ } -> Some { loops = List.rev loops; tn; idcs; llsc }
    | _ -> None
  in
  go [] stmt

(* Whether a statement holds code the census cannot see, or that the rewrite must not move a write
   across: a query over the statement's effect rows (gh-ocannl-1016), the one view of what runs
   beside its accesses. *)
let opaque_effect (e : Tn.t Affine.statement_effect) =
  match e.e_kind with
  | Affine.Staged | Affine.Barrier -> true
  | Affine.Mma ->
      (* A code-motion barrier as a construct, whatever its scalar fallback spells (gh-1001). *)
      true
  | Affine.Scope_body ->
      (* The census is the same relations, which descend into a scope body at its use site: what a
         body reads or writes -- an impure one included, whose purity only the optimizer checks,
         downstream of the tier -- is counted at the top-level statement holding it, where it runs
         (gh-ocannl-1050). A body's staged code or barrier is a row of its own. *)
      false
  | Affine.Local_write _ | Affine.Local_read _ | Affine.Local_declare | Affine.Merge_read _ -> false

let wrap loops body =
  List.fold_right loops ~init:body ~f:(fun { index; from_; to_ } body ->
      LL.For_loop { index; from_; to_; body; axis = LL.Serial })

let mentions sym (idcs : Idx.axis_index array) =
  Array.exists idcs ~f:(function Idx.Iterator s -> Idx.equal_symbol s sym | _ -> false)

let is_loop (n : nest) sym = List.exists n.loops ~f:(fun lp -> Idx.equal_symbol lp.index sym)

(* [reduction n] is the accumulation operator and the accumulated operand of a nest of the shape
   [lhs[i] := lhs[i] op rhs] (either operand order for a commutative [op]) where [rhs] does not read
   [lhs]. *)
let reduction (n : nest) : (Ops.binop * LL.scalar_t) option =
  let self = function
    | LL.Get (tn, idcs) -> Tn.equal tn n.tn && [%equal: Idx.axis_index array] idcs n.idcs
    | _ -> false
  in
  let plain op rhs = if LL.scalar_mentions_tn n.tn rhs then None else Some (op, rhs) in
  match n.llsc with
  | LL.Binop (op, (acc, _), (rhs, _)) when self acc -> plain op rhs
  | LL.Binop (((Ops.Add | Ops.Max) as op), (rhs, _), (acc, _)) when self acc -> plain op rhs
  | _ -> None

(* An elementwise definition: the value reads no cell of its own target. *)
let pointwise (n : nest) = not (LL.scalar_mentions_tn n.tn n.llsc)

(* [s] with every index rewritten by [idx] and every read [get] answers replaced by its answer, and
   whether any read was replaced. Only plain arithmetic moves: a scope, a local, a dynamic or a
   merge-buffer read declines. *)
let rec map_scalar ~(idx : Idx.axis_index -> Idx.axis_index option)
    ~(get : Tn.t -> Idx.axis_index array -> LL.scalar_t option) (s : LL.scalar_t) :
    (LL.scalar_t * bool) option =
  let idcs a = Array.to_list a |> List.map ~f:idx |> Option.all |> Option.map ~f:Array.of_list in
  (* An argument whose subtree was substituted takes the substitute's precision. *)
  let arg (s, prec) =
    let* s', changed = map_scalar ~idx ~get s in
    Some ((s', if changed then LL.scalar_precision s' else prec), changed)
  in
  match s with
  | LL.Get (tn, i) -> (
      match get tn i with
      | Some s' -> Some (s', true)
      | None ->
          let* i = idcs i in
          Some (LL.Get (tn, i), false))
  | LL.Embed_index i ->
      let* i = idx i in
      Some (LL.Embed_index i, false)
  | LL.Constant _ | LL.Constant_bits _ -> Some (s, false)
  | LL.Unop (op, a) ->
      let* a, ca = arg a in
      Some (LL.Unop (op, a), ca)
  | LL.Binop (op, a, b) ->
      let* a, ca = arg a in
      let* b, cb = arg b in
      Some (LL.Binop (op, a, b), ca || cb)
  | LL.Ternop (op, a, b, c) ->
      let* a, ca = arg a in
      let* b, cb = arg b in
      let* c, cc = arg c in
      Some (LL.Ternop (op, a, b, c), ca || cb || cc)
  | LL.Local_scope _ | LL.Get_local _ | LL.Get_dynamic _ | LL.Get_merge_buffer _ -> None

(* The rewrite's view of a routine: its top-level statements, their nests, and the census -- who
   writes what, what each statement reads, and which statements hold code the census cannot see --
   read off each statement's relations (gh-ocannl-1050), which enter loop, scan, guard and scope
   bodies alike. A merge-buffer read counts as a read of its source node, conservatively. *)
type routine = {
  stmts : LL.t array;
  nests : nest option array;
  writers : (Tn.t, int list) Hashtbl.t;
  reads : Set.M(Tn).t array;
  opaque : bool array;
}

let routine_of (llc : LL.t) : routine =
  let stmts = Array.of_list (LL.flat_lines [ llc ]) in
  let writers = Hashtbl.create (module Tn) in
  let census =
    Array.mapi stmts ~f:(fun pos stmt ->
        let accs, effs = LL.affine_relations stmt in
        let nodes l = Set.of_list (module Tn) l in
        let writes, reads = List.partition_tf accs ~f:(fun (a : Tn.t Affine.access) -> a.a_write) in
        Set.iter
          (nodes (List.map writes ~f:(fun a -> a.a_tn)))
          ~f:(fun tn -> Hashtbl.add_multi writers ~key:tn ~data:pos);
        let merge_reads =
          List.filter_map effs ~f:(fun (e : Tn.t Affine.statement_effect) ->
              match e.e_kind with Affine.Merge_read tn -> Some tn | _ -> None)
        in
        ( nodes (merge_reads @ List.map reads ~f:(fun a -> a.a_tn)),
          List.exists effs ~f:opaque_effect ))
  in
  {
    stmts;
    nests = Array.map stmts ~f:nest_of;
    writers;
    reads = Array.map census ~f:fst;
    opaque = Array.map census ~f:snd;
  }

let writers r tn = Hashtbl.find_multi r.writers tn |> List.sort ~compare:Int.compare
let reads_at r pos = r.reads.(pos)

(* The one pointwise nest defining [tn], if it is defined exactly once and elementwise. *)
let definition r tn =
  match writers r tn with
  | [ pos ] -> Option.bind r.nests.(pos) ~f:(fun n -> Option.some_if (pointwise n) (pos, n))
  | _ -> None

(* Whether a nest writes every cell of its node, once: each axis a plain iterator of its own loop
   over the axis's whole extent, or the fixed index of a unit axis. Instantiating a definition at
   arbitrary subscripts is sound only then -- a partial writer's node keeps whatever else it held in
   the cells it skips. *)
let covers_whole (n : nest) =
  let dims = Lazy.force n.tn.Tn.dims in
  Array.length dims = Array.length n.idcs
  && List.length n.loops = Array.count n.idcs ~f:(function Idx.Iterator _ -> true | _ -> false)
  && Array.for_alli n.idcs ~f:(fun a -> function
    | Idx.Fixed_idx k -> k = 0 && dims.(a) = 1
    | Idx.Iterator s -> (
        match List.find n.loops ~f:(fun lp -> Idx.equal_symbol lp.index s) with
        | Some lp -> lp.from_ = 0 && lp.to_ = dims.(a) - 1
        | None -> false)
    | _ -> false)
  && not
       (List.contains_dup ~compare:Idx.compare_symbol
          (Array.to_list n.idcs
          |> List.filter_map ~f:(function Idx.Iterator s -> Some s | _ -> None)))

(* The expression a read stands for, unfolded through single-writer elementwise definitions and
   constant fills (short of the [stop] nodes) that write their whole node at a position before
   [before] -- where the reads being replaced ran -- with the positions of the definitions it went
   through. A definition's target symbols map positionally to the read's indices. *)
let rec unfold r ~stop ~before ~depth (s : LL.scalar_t) : (LL.scalar_t * int list) option =
  let found = ref [] in
  let get tn idcs =
    if depth <= 0 || List.mem stop tn ~equal:Tn.equal then None
    else
      match writers r tn with
      | [ pos ] when pos < before -> (
          match (r.stmts.(pos), r.nests.(pos)) with
          | LL.Zero_out _, _ ->
              found := [ pos ] :: !found;
              Some (LL.Constant 0.)
          | _, Some ({ llsc = LL.Constant c; _ } as n) when covers_whole n ->
              found := [ pos ] :: !found;
              Some (LL.Constant c)
          | _, Some n when pointwise n && covers_whole n ->
              let pairs = Array.zip_exn n.idcs idcs |> Array.to_list in
              let subst =
                List.filter_map pairs ~f:(function Idx.Iterator s, i -> Some (s, i) | _ -> None)
              in
              let plain =
                Array.length n.idcs = Array.length idcs
                && List.for_all pairs ~f:(function
                  | Idx.Iterator _, (Idx.Iterator _ | Idx.Fixed_idx _) -> true
                  | Idx.Fixed_idx k, Idx.Fixed_idx k' -> k = k'
                  | _ -> false)
                && List.length n.loops = List.length subst
                && List.for_all n.loops ~f:(fun lp ->
                    List.Assoc.mem subst ~equal:Idx.equal_symbol lp.index)
              in
              if not plain then None
              else
                let idx = function
                  | Idx.Iterator s -> List.Assoc.find subst ~equal:Idx.equal_symbol s
                  | Idx.Fixed_idx _ as i -> Some i
                  | _ -> None
                in
                let* body, _ = map_scalar ~idx ~get:(fun _ _ -> None) n.llsc in
                let* body, ps = unfold r ~stop ~before ~depth:(depth - 1) body in
                found := (pos :: ps) :: !found;
                Some body
          | _ -> None)
      | _ -> None
  in
  let* s, _ = map_scalar ~idx:Option.some ~get s in
  Some (s, List.concat !found)

(* {2 Index arithmetic} *)

let terms_of = function
  | Idx.Fixed_idx k -> Some ([], k)
  | Idx.Iterator s -> Some ([ (1, s) ], 0)
  | Idx.Affine { symbols; offset } -> Some (symbols, offset)
  | Idx.Sub_axis | Idx.Concat _ -> None

(* [i] with each symbol [sub] maps replaced by the index it maps to, affinely. *)
let subst_idx sub (i : Idx.axis_index) : Idx.axis_index option =
  let* terms, offset = terms_of i in
  let* parts =
    List.map terms ~f:(fun (c, s) ->
        match sub s with
        | None -> Some ([ (c, s) ], 0)
        | Some i' ->
            let* ts, o = terms_of i' in
            Some (List.map ts ~f:(fun (c', s') -> (c * c', s')), c * o))
    |> Option.all
  in
  Some
    (Idx.affine ~symbols:(List.concat_map parts ~f:fst)
       ~offset:(offset + List.sum (module Int) parts ~f:snd))

let subst_scalar ?(get = fun _ _ -> None) sub s =
  Option.map (map_scalar ~idx:(subst_idx sub) ~get s) ~f:fst
