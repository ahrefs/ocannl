(* The online-softmax attention rewrite (gh-ocannl-483); the contract is in the interface. *)

open Base
module LL = Low_level
module Tn = Tnode
module Idx = Indexing

let override : bool option ref = ref None
let set_enabled b = override := b

let enabled () =
  match !override with
  | Some b -> b
  | None -> Utils.get_global_flag ~default:false ~arg_name:"online_softmax"

(* {1 Minted nodes}

   The rewrite mints scalar nodes for its scope locals -- the carried state of the scan, the score
   it reads once per step, the hoisted probability -- in a namespace of its own, like the schedule's
   tile nodes, so their session ids are independent of tensor-land ids. Each is declared [Virtual]
   at creation: a scope local's node names and types the local and is never a buffer. *)

let namespace = "rewrite"
let provenance = 483

let fresh_id =
  let c = ref (-1) in
  fun () ->
    Int.incr c;
    !c

let scalar_node ~label ~(like : Tn.t) prec =
  let tn =
    Tn.create ~namespace (Tn.Specified prec) ~id:(fresh_id ()) ~label:(label :: like.Tn.label)
      ~unpadded_dims:(lazy [| 1 |])
      ~padding:(lazy None)
      ()
  in
  Tn.update_memory_mode tn Tn.Virtual provenance;
  tn

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

(* {1 Axis roles}

   The recognizer relates nests through the axes of the nodes they share, never through loop
   symbols, which every nest mints afresh. The max-reduction fixes the vocabulary: each of its loops
   is a row of the max -- [Row a], named by the axis [a] of the max it indexes -- or the one
   [Reduced] axis. A node's signature gives each of its axes a role or a fixed index; reading a
   signed node in another nest binds that nest's loop symbols to roles, and a write through bound
   symbols signs the written node. *)

type role = Row of int | Reduced [@@deriving equal, compare, sexp_of]
type slot = Role of role | Fixed of int
type signature = slot array
type env = (Idx.symbol * role) list
type vocabulary = { env : env; extents : (role * int) list  (** Each role's inclusive bound. *) }

let role_of (env : env) s = List.Assoc.find env ~equal:Idx.equal_symbol s

(* [bind env sg idcs] extends [env] with what a read at [idcs] of a node signed [sg] imposes: a role
   slot wants a loop symbol carrying that role (fresh, or already bound to it) and a fixed slot
   wants that fixed index. Roles are injective over symbols. *)
let bind (env : env) (sg : signature) (idcs : Idx.axis_index array) : env option =
  if Array.length sg <> Array.length idcs then None
  else
    Array.fold2_exn sg idcs ~init:(Some env) ~f:(fun env slot idx ->
        Option.bind env ~f:(fun env ->
            match (slot, idx) with
            | Fixed k, Idx.Fixed_idx k' when k = k' -> Some env
            | Role r, Idx.Iterator s -> (
                match role_of env s with
                | Some r' -> Option.some_if (equal_role r r') env
                | None ->
                    Option.some_if
                      (not (List.exists env ~f:(fun (_, r') -> equal_role r r')))
                      ((s, r) :: env))
            | _ -> None))

(* [sign env idcs] is the signature a write at [idcs] gives its node under [env]: every iterator
   bound, no symbol indexing two axes, nothing but iterators and fixed indices. *)
let sign (env : env) (idcs : Idx.axis_index array) : signature option =
  let seen = ref [] in
  Array.to_list idcs
  |> List.map ~f:(function
    | Idx.Fixed_idx k -> Some (Fixed k)
    | Idx.Iterator s when not (List.mem !seen s ~equal:Idx.equal_symbol) ->
        seen := s :: !seen;
        Option.map (role_of env s) ~f:(fun r -> Role r)
    | _ -> None)
  |> Option.all |> Option.map ~f:Array.of_list

let roles_of (sg : signature) =
  Array.to_list sg |> List.filter_map ~f:(function Role r -> Some r | Fixed _ -> None)

let same_roles a b =
  let sort = List.sort ~compare:compare_role in
  List.equal equal_role (sort (roles_of a)) (sort (roles_of b))

(* [consistent voc env loops] holds when [env] binds exactly the loops of a nest, each over the full
   extent its role has in the vocabulary. *)
let consistent (voc : vocabulary) (env : env) (loops : loop list) =
  List.length env = List.length loops
  && List.for_all loops ~f:(fun lp ->
      match role_of env lp.index with
      | None -> false
      | Some r ->
          lp.from_ = 0
          && Option.equal Int.equal (List.Assoc.find voc.extents ~equal:equal_role r) (Some lp.to_))

let ( let* ) x f = Option.bind x ~f

(* The reads of signed nodes in a fresh nest: the bindings they impose together, if consistent. *)
let reads_in voc (n : nest) (reads : (signature * Idx.axis_index array) list) =
  List.fold reads ~init:(Some []) ~f:(fun env (sg, idcs) ->
      let* env = env in
      bind env sg idcs)
  |> Option.filter ~f:(fun env -> consistent voc env n.loops)

(* {1 The online normalizer}

   The four nests the composed softmax lowers to,

   [m := max over t of x; n := x - m; e := exp n; l := sum over t of e]

   (with [m]'s neutral-element initialization and [l]'s zeroing), become one scan per row carrying
   the running max and the rescaled running sum, writing both trajectories to [m] and [l]: [m'] =
   max(m, x); [l'] = l * exp(m - m') + exp(x - m'). The pointwise nests stay: they define [n] and
   [e] for whoever reads them, and the virtualizer inlines them per cell. *)

type normalizer = {
  a_init : int;  (** The position of [m]'s initialization. *)
  a : int;  (** The max-reduction. *)
  c_init : int;  (** [l]'s zeroing. *)
  c : int;  (** The sum-reduction. *)
  m : Tn.t;
  l : Tn.t;
  x : Tn.t;
  x_idcs : Idx.axis_index array;
  rows : loop list;  (** The max-reduction's loops other than [t], in their order. *)
  t : loop;
  m_idcs : Idx.axis_index array;
  l_idcs : Idx.axis_index array;  (** [l]'s cell in the max-reduction's symbols. *)
}

(* The vocabulary: a max-reduction over exactly one loop of a plain read. *)
let max_reduce (n : nest) =
  match reduction n with
  | Some (Ops.Max, LL.Get (x, x_idcs)) -> (
      let rows_env =
        Array.to_list n.idcs
        |> List.filter_mapi ~f:(fun a -> function Idx.Iterator s -> Some (s, Row a) | _ -> None)
      in
      let reduced = List.filter n.loops ~f:(fun lp -> Option.is_none (role_of rows_env lp.index)) in
      let distinct =
        not (List.contains_dup (List.map rows_env ~f:fst) ~compare:Idx.compare_symbol)
      in
      match reduced with
      | [ t ] when distinct && t.from_ <= t.to_ ->
          let env = (t.index, Reduced) :: rows_env in
          let extents =
            List.filter_map n.loops ~f:(fun lp ->
                Option.map (role_of env lp.index) ~f:(fun r -> (r, lp.to_)))
          in
          let voc = { env; extents } in
          if consistent voc env n.loops then
            Option.bind (sign env x_idcs) ~f:(fun sig_x ->
                Option.bind (sign env n.idcs) ~f:(fun sig_m ->
                    Option.some_if
                      (List.mem (roles_of sig_x) Reduced ~equal:equal_role)
                      (voc, x, x_idcs, sig_x, sig_m, t)))
          else None
      | _ -> None)
  | _ -> None

(* The rewrite's view of a routine: its top-level statements, their nests, and who writes what. *)
type routine = {
  stmts : LL.t array;
  nests : nest option array;
  writers : (Tn.t, int list) Hashtbl.t;
}

let routine_of (llc : LL.t) : routine =
  let stmts = Array.of_list (LL.flat_lines [ llc ]) in
  let writers = Hashtbl.create (module Tn) in
  Array.iteri stmts ~f:(fun pos stmt ->
      Set.iter (LL.writes_of_stmt stmt) ~f:(fun tn -> Hashtbl.add_multi writers ~key:tn ~data:pos));
  { stmts; nests = Array.map stmts ~f:nest_of; writers }

let writers r tn = Hashtbl.find_multi r.writers tn |> List.sort ~compare:Int.compare
let reads_at r pos = LL.reads_of_body r.stmts.(pos)

(* The one pointwise nest defining [tn], if it is defined exactly once and elementwise. *)
let definition r tn =
  match writers r tn with
  | [ pos ] -> Option.bind r.nests.(pos) ~f:(fun n -> Option.some_if (pointwise n) (pos, n))
  | _ -> None

let find_normalizer r (a : int) : normalizer option =
  let* an = r.nests.(a) in
  let* voc, x, x_idcs, sig_x, sig_m, t = max_reduce an in
  let m = an.tn in
  (* The one elementwise definition whose reads are [reads], signing its node. *)
  let defined_by reads (n : nest) =
    let* env = reads_in voc n reads in
    let* _ = definition r n.tn in
    let* sg = sign env n.idcs in
    Some (sg, n.tn)
  in
  (* [n := x - m], elementwise over the max's rows and reduced axis. *)
  let* sig_n, n_tn =
    Array.find_map r.nests ~f:(function
      | Some ({ llsc = LL.Binop (Ops.Sub, (LL.Get (x', xi), _), (LL.Get (m', mi), _)); _ } as nn)
        when Tn.equal x' x && Tn.equal m' m ->
          defined_by [ (sig_x, xi); (sig_m, mi) ] nn
      | _ -> None)
  in
  (* [e := exp n]. *)
  let* sig_e, e_tn =
    Array.find_map r.nests ~f:(function
      | Some ({ llsc = LL.Unop (Ops.Exp, (LL.Get (n', ni), _)); _ } as en) when Tn.equal n' n_tn ->
          defined_by [ (sig_n, ni) ] en
      | _ -> None)
  in
  (* [l := sum over t of e], reducing the same axis into the same rows. *)
  let* c, l, l_sig =
    Array.find_mapi r.nests ~f:(fun c nest ->
        let* cn = nest in
        match reduction cn with
        | Some (Ops.Add, LL.Get (e', ei)) when Tn.equal e' e_tn ->
            let* env = reads_in voc cn [ (sig_e, ei) ] in
            let* sig_l = sign env cn.idcs in
            Option.some_if (same_roles sig_l sig_m) (c, cn.tn, sig_l)
        | _ -> None)
  in
  (* [m]'s neutral-element fill covering all of [m]'s rows, and [l]'s zeroing. *)
  let covers sg (n : nest) = Option.is_some (reads_in voc n [ (sg, n.idcs) ]) in
  let filled tn sg value pos =
    match r.nests.(pos) with
    | Some ({ llsc = LL.Constant v; _ } as n) ->
        Tn.equal n.tn tn && Float.equal v value && covers sg n
    | _ -> false
  in
  let zeroed tn pos = match r.stmts.(pos) with LL.Zero_out tn' -> Tn.equal tn' tn | _ -> false in
  let* a_init, c_init =
    match (writers r m, writers r l) with
    | [ a_init; a' ], [ c_init; c' ]
      when a' = a && c' = c
           && filled m sig_m Float.neg_infinity a_init
           && (zeroed l c_init || filled l l_sig 0. c_init) ->
        Some (a_init, c_init)
    | _ -> None
  in
  (* [m] and [l] are consumed only once each is complete: the scan writes them at [a], earlier than
     [l]'s original definition, so nothing between [a] and [c] may read [l] -- and nothing may
     redefine [x] once the max has read it. *)
  let untouched =
    a < c
    && List.for_all (writers r x) ~f:(fun w -> w < a)
    && (not (List.exists (List.range (a + 1) c) ~f:(fun pos -> Set.mem (reads_at r pos) l)))
    && not (List.exists (List.range (a_init + 1) a) ~f:(fun pos -> Set.mem (reads_at r pos) m))
  in
  let* () = Option.some_if untouched () in
  let rows = List.filter an.loops ~f:(fun lp -> not (Idx.equal_symbol lp.index t.index)) in
  let symbol_of role =
    List.find_map_exn voc.env ~f:(fun (s, r) -> Option.some_if (equal_role r role) s)
  in
  let l_idcs =
    Array.map l_sig ~f:(function
      | Fixed k -> Idx.Fixed_idx k
      | Role r -> Idx.Iterator (symbol_of r))
  in
  Some { a_init; a; c_init; c; m; l; x; x_idcs; rows; t; m_idcs = an.idcs; l_idcs }

let state_prec (tn : Tn.t) =
  match Lazy.force tn.Tn.storage_prec with Ops.Double_prec _ as p -> p | _ -> Ops.single

let emit_normalizer (nz : normalizer) : LL.t =
  let open LL in
  let prec = state_prec nz.m in
  let m_st = scalar_node ~label:"online_max" ~like:nz.m prec in
  let l_st = scalar_node ~label:"online_sum" ~like:nz.l prec in
  let x_st = scalar_node ~label:"online_score" ~like:nz.x prec in
  let m = { prev = get_scope m_st; next = get_scope m_st; init = Constant Float.neg_infinity } in
  let l = { prev = get_scope l_st; next = get_scope l_st; init = Constant 0. } in
  let x = get_scope x_st in
  let binop op a b = apply_op (Ops.Binop op) [| a; b |] in
  let exp_ a = apply_op (Ops.Unop Ops.Exp) [| a |] in
  let m_prev = Get_local m.prev and m_next = Get_local m.next and l_prev = Get_local l.prev in
  (* A row whose prefix is entirely masked ([-inf] scores) keeps [m' = -inf], where the rescaling
     factor would be [exp (-inf - -inf) = nan]; its normalizer stays 0 until the first live score,
     whose rescaling of the empty prefix is [exp (-inf - x) = 0]. A fully masked row ends with [l =
     0], the composed form's NaN in a different coat. *)
  let l_next =
    apply_op (Ops.Ternop Ops.Where)
      [|
        binop Ops.Cmpeq m_next (Constant Float.neg_infinity);
        Constant 0.;
        binop Ops.Add
          (binop Ops.Mul l_prev (exp_ (binop Ops.Sub m_prev m_next)))
          (exp_ (binop Ops.Sub (Get_local x) m_next));
      |]
  in
  let body =
    unflat_lines
      [
        Declare_local { id = x; needs_init = false };
        Set_local (x, Get (nz.x, nz.x_idcs));
        Set_local (m.next, binop Ops.Max m_prev (Get_local x));
        Set_local (l.next, l_next);
        Set { tn = nz.m; idcs = nz.m_idcs; llsc = m_next; debug = "" };
        Set { tn = nz.l; idcs = nz.l_idcs; llsc = Get_local l.next; debug = "" };
      ]
  in
  wrap nz.rows
    (Scan_loop
       {
         index = nz.t.index;
         from_ = nz.t.from_;
         to_ = nz.t.to_;
         direction = Forward;
         carried = [ m; l ];
         body;
       })

(* {1 Hoisting the probabilities}

   A reduction [o[.., e] += w[rows, t] * v[..]] consuming the normalized probabilities reads each
   cell of [w] once per iteration of the loops [w] does not index -- the value width of the
   attention -- which is exactly the multiplicity that materializes [w] as a [seq^2] buffer. Moving
   those loops innermost and reading [w] once into a local ahead of them leaves the per-cell
   summation orders untouched (every hoisted-over loop indexes the target, so it reduces nothing)
   while [w] is read once per cell: the whole probability chain then inlines into that one read. *)

(* Whether [tn] is defined elementwise from one of the [ls], through elementwise nests only. *)
let rec normalized r ~ls ~depth tn =
  depth < 8
  &&
  match definition r tn with
  | None -> false
  | Some (pos, _) ->
      let reads = reads_at r pos in
      List.exists ls ~f:(Set.mem reads)
      || Set.exists reads ~f:(fun d ->
          (not (Tn.equal d tn)) && normalized r ~ls ~depth:(depth + 1) d)

let hoist r ~ls (n : nest) : LL.t option =
  let* w, wi, v, vi, w_first =
    match reduction n with
    | Some (Ops.Add, LL.Binop (Ops.Mul, (LL.Get (w, wi), _), (LL.Get (v, vi), _))) ->
        if normalized r ~ls ~depth:0 w then Some (w, wi, v, vi, true)
        else if normalized r ~ls ~depth:0 v then Some (v, vi, w, wi, false)
        else None
    | _ -> None
  in
  let outer, inner = List.partition_tf n.loops ~f:(fun lp -> mentions lp.index wi) in
  let plain_loops idcs =
    Array.for_all idcs ~f:(function
      | Idx.Iterator s -> is_loop n s
      | Idx.Fixed_idx _ -> true
      | _ -> false)
  in
  let sound =
    (not (List.is_empty inner))
    && plain_loops wi && plain_loops vi
    && List.for_all inner ~f:(fun lp -> mentions lp.index n.idcs)
  in
  let* () = Option.some_if sound () in
  let p_st = scalar_node ~label:"probability" ~like:w (Lazy.force w.Tn.storage_prec) in
  let p = LL.get_scope p_st in
  let product =
    let w' = LL.Get_local p and v' = LL.Get (v, vi) in
    LL.apply_op (Ops.Binop Ops.Mul) (if w_first then [| w'; v' |] else [| v'; w' |])
  in
  let accumulate =
    LL.Set
      {
        tn = n.tn;
        idcs = n.idcs;
        llsc = LL.apply_op (Ops.Binop Ops.Add) [| LL.Get (n.tn, n.idcs); product |];
        debug = "";
      }
  in
  Some
    (wrap outer
       (LL.unflat_lines
          [
            LL.Declare_local { id = p; needs_init = false };
            LL.Set_local (p, LL.Get (w, wi));
            wrap inner accumulate;
          ]))

(* {1 The pass} *)

let rewrite (llc : LL.t) : LL.t =
  let r = routine_of llc in
  let normalizers = List.filter_map (List.range 0 (Array.length r.stmts)) ~f:(find_normalizer r) in
  if List.is_empty normalizers then llc
  else
    let stmts = Array.copy r.stmts in
    List.iter normalizers ~f:(fun nz ->
        stmts.(nz.a_init) <- LL.Noop;
        stmts.(nz.c_init) <- LL.Noop;
        stmts.(nz.c) <- LL.Noop;
        stmts.(nz.a) <- emit_normalizer nz);
    let ls = List.map normalizers ~f:(fun nz -> nz.l) in
    Array.iteri r.nests ~f:(fun pos -> function
      | Some n when not (List.exists normalizers ~f:(fun nz -> pos = nz.a || pos = nz.c)) -> (
          match hoist r ~ls n with Some replacement -> stmts.(pos) <- replacement | None -> ())
      | _ -> ());
    LL.unflat_lines (Array.to_list stmts)
