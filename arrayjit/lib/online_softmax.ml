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
let provenance = Tn.Site "483:online-softmax-state"

let fresh_id =
  let c = ref (-1) in
  fun () ->
    Int.incr c;
    !c

(* Sibling lowerings of one program -- placement arms, autotune candidates -- must mint the SAME
   nodes, or the analysis cache, which keys nodes by identity, misses on every one of them: a local
   is memoized by the node it stands for and its role, and the memo is cleared with the other
   node-retaining caches: ahead of an accessibility snapshot, and at a session reset through the
   tier's [Rewrites.reset]. *)
module Minted_key = struct
  type t = int * string [@@deriving compare, hash, sexp_of]
end

let minted : (Minted_key.t, Tn.t) Hashtbl.t = Hashtbl.create (module Minted_key)
let reset () = Hashtbl.clear minted
let () = Tn.before_accessibility_snapshot := reset :: !Tn.before_accessibility_snapshot

let scalar_node ~label ~(like : Tn.t) prec =
  Hashtbl.find_or_add minted (like.Tn.uid, label) ~default:(fun () ->
      let tn =
        Tn.create ~namespace (Tn.Specified prec) ~id:(fresh_id ()) ~label:(label :: like.Tn.label)
          ~unpadded_dims:(lazy [| 1 |])
          ~padding:(lazy None)
          ()
      in
      Tn.update_memory_mode tn Tn.Virtual provenance;
      tn)

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

(* {1 Axis roles}

   The recognizer relates nests through the axes of the nodes they share, never through loop
   symbols, which every nest mints afresh. The max-reduction fixes the vocabulary: each of its loops
   is a row of the max -- [Row a], named by the axis [a] of the max it indexes -- or the one
   [Reduced] axis. A node's signature gives each of its axes a role or a fixed index; reading a
   signed node in another nest binds that nest's loop symbols to roles, and a write through bound
   symbols signs the written node. *)

type role =
  | Row of int
  | Reduced
  | Chan of int
      (** A loop the max-reduction does not have: a contraction's channel (the fused backward's
          value width is [Chan 0], its head width [Chan 1]). *)
[@@deriving equal, compare, sexp_of]

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
  voc : vocabulary;  (** The max-reduction's roles and their extents. *)
  sig_x : signature;
  sig_m : signature;
  sig_l : signature;
  n : Tn.t;  (** The shifted scores [x - m]. *)
  e : Tn.t;  (** The exponential [exp n]. *)
  sig_e : signature;
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

let find_normalizer r (a : int) : normalizer option =
  let* an = r.nests.(a) in
  let* voc, x, x_idcs, sig_x, sig_m, t = max_reduce an in
  let m = an.tn in
  (* The one elementwise definition whose reads are [reads], signing its node. *)
  let defined_by reads (n : nest) =
    let* env = reads_in voc n reads in
    let* pos, _ = definition r n.tn in
    let* sg = sign env n.idcs in
    Some (sg, n.tn, pos)
  in
  (* The three links of the chain, each as the list of its candidates: a max may feed more than one
     subtraction (an auxiliary chain that never reaches a sum), so the first candidate at a link can
     dead-end while a later one is the normalizer -- the search below tries every subtraction, every
     exponential of it and every sum of that before giving up. *)
  let candidates f = Array.to_list r.nests |> List.filter_map ~f in
  (* [n := x - m], elementwise over the max's rows and reduced axis. *)
  let subtractions =
    candidates (function
      | Some ({ llsc = LL.Binop (Ops.Sub, (LL.Get (x', xi), _), (LL.Get (m', mi), _)); _ } as nn)
        when Tn.equal x' x && Tn.equal m' m ->
          defined_by [ (sig_x, xi); (sig_m, mi) ] nn
      | _ -> None)
  in
  (* [e := exp n]. *)
  let exponentials (sig_n, n_tn, _) =
    candidates (function
      | Some ({ llsc = LL.Unop (Ops.Exp, (LL.Get (n', ni), _)); _ } as en) when Tn.equal n' n_tn ->
          defined_by [ (sig_n, ni) ] en
      | _ -> None)
  in
  (* [l := sum over t of e], reducing the same axis into the same rows. *)
  let sums (sig_e, e_tn, _) =
    Array.to_list r.nests
    |> List.filter_mapi ~f:(fun c nest ->
        let* cn = nest in
        match reduction cn with
        | Some (Ops.Add, LL.Get (e', ei)) when Tn.equal e' e_tn ->
            let* env = reads_in voc cn [ (sig_e, ei) ] in
            let* sig_l = sign env cn.idcs in
            Option.some_if (same_roles sig_l sig_m) (c, cn.tn, sig_l)
        | _ -> None)
  in
  (* The rest of the contract, for one complete chain. *)
  let complete (_, n_tn, n_pos) (sig_e, e_tn, e_pos) (c, l, l_sig) : normalizer option =
    (* The state accumulates fractional exponentials at f32 or f64; an integer node's own reduction
       truncated after every step, which the scan would not reproduce. *)
    let float_node (tn : Tn.t) = Ops.is_float (Lazy.force tn.Tn.storage_prec) in
    let* () = Option.some_if (List.for_all [ m; l; x; n_tn; e_tn ] ~f:float_node) () in
    (* The chain -- max, subtraction, exponential, normalizer -- is at one precision, the scores'.
       The composed form rounds at each materialized node, so a node narrower than the scores rounds
       where the carried state does not (an out-of-range score to [-inf], a small exponential to
       zero), and a node wider than the others accumulates terms the composed form had rounded
       through the narrower ones. At one precision the two forms differ by the state's widening to
       f32 alone, which is the contract: the running pair lives at f32 under narrow scores
       (gh-ocannl-483), never per-step at the storage width. *)
    let prec (tn : Tn.t) = Lazy.force tn.Tn.storage_prec in
    let* () =
      Option.some_if
        (List.for_all [ m; n_tn; e_tn; l ] ~f:(fun tn -> Ops.equal_prec (prec tn) (prec x)))
        ()
    in
    (* [m]'s neutral-element fill covering all of [m]'s rows, and [l]'s zeroing. *)
    let covers sg (n : nest) = Option.is_some (reads_in voc n [ (sg, n.idcs) ]) in
    let filled tn sg value pos =
      match r.nests.(pos) with
      | Some ({ llsc = LL.Constant v; _ } as n) ->
          Tn.equal n.tn tn && Float.equal v value && covers sg n
      | _ -> false
    in
    (* A whole-node zeroing clears every cell; the scan writes the reduction's cells, so it may only
       replace the zeroing when those are all of them. *)
    let covers_node (tn : Tn.t) (sg : signature) =
      let dims = Lazy.force tn.Tn.dims in
      Array.length dims = Array.length sg
      && Array.for_alli sg ~f:(fun a slot ->
          match slot with
          | Fixed k -> k = 0 && dims.(a) = 1
          | Role role ->
              Option.equal Int.equal
                (List.Assoc.find voc.extents ~equal:equal_role role)
                (Some (dims.(a) - 1)))
    in
    let zeroed tn sg pos =
      match r.stmts.(pos) with
      | LL.Zero_out tn' -> Tn.equal tn' tn && covers_node tn sg
      | _ -> false
    in
    let* a_init, c_init =
      match (writers r m, writers r l) with
      | [ a_init; a' ], [ c_init; c' ]
        when a' = a && c' = c
             && filled m sig_m Float.neg_infinity a_init
             && (zeroed l l_sig c_init || filled l l_sig 0. c_init) ->
          Some (a_init, c_init)
      | _ -> None
    in
    let reads_between lo hi tn =
      List.exists (List.range (lo + 1) hi) ~f:(fun pos -> Set.mem (reads_at r pos) tn)
    in
    (* The chain runs in program order, max first and sum last, with its pointwise definitions in
       between: a definition outside that span consumed a stale [m] or a stale [n] in the original,
       which the scan would not reproduce. Between its initialization and the max, [l] holds its
       zero and [m] its neutral fill, and the scan deletes both fills -- so nothing may read either
       there, and nothing may redefine [x] once the max has read it. Staged code is invisible to the
       census, and barriers and tensor-core statements are fences code motion must not cross, so
       none may sit in the span the rewrite reorders. *)
    let untouched =
      a < n_pos && n_pos < e_pos && e_pos < c
      && List.for_all (writers r x) ~f:(fun w -> w < a)
      && (not (reads_between (Int.min a c_init) c l))
      && (not (reads_between a_init a m))
      && not
           (List.exists (List.range (Int.min a_init c_init) (c + 1)) ~f:(fun pos -> r.opaque.(pos)))
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
    Some
      {
        a_init;
        a;
        c_init;
        c;
        m;
        l;
        x;
        x_idcs;
        rows;
        t;
        m_idcs = an.idcs;
        l_idcs;
        voc;
        sig_x;
        sig_m;
        sig_l = l_sig;
        n = n_tn;
        e = e_tn;
        sig_e;
      }
  in
  List.find_map subtractions ~f:(fun n ->
      List.find_map (exponentials n) ~f:(fun e -> List.find_map (sums e) ~f:(complete n e)))

(* A local's precision: its node's, widened to f32 -- the state must not round per step under narrow
   storage, and a double node keeps its double. *)
let state_prec (tn : Tn.t) =
  match Lazy.force tn.Tn.storage_prec with Ops.Double_prec _ as p -> p | _ -> Ops.single

(* The most negative finite value of a state precision: the floor the rescaling reads the running
   max through, so that [-inf] never meets itself in a subtraction (see the recurrence below). Not
   [Float.max_value], which Base defines as [infinity]: that floor is [-inf] itself, and a masked
   prefix at f64 made every such row NaN. *)
let lowest_finite = function
  | Ops.Double_prec _ -> -.Float.max_finite_value
  | _ -> -3.4028234663852886e38

let emit_normalizer (nz : normalizer) : LL.t =
  let open LL in
  let m_st = scalar_node ~label:"online_max" ~like:nz.m (state_prec nz.m) in
  let l_st = scalar_node ~label:"online_sum" ~like:nz.l (state_prec nz.l) in
  let x_st = scalar_node ~label:"online_score" ~like:nz.x (state_prec nz.x) in
  let m = { prev = get_scope m_st; next = get_scope m_st; init = Constant Float.neg_infinity } in
  let l = { prev = get_scope l_st; next = get_scope l_st; init = Constant 0. } in
  let x = get_scope x_st in
  let binop op a b = apply_op (Ops.Binop op) [| a; b |] in
  let exp_ a = apply_op (Ops.Unop Ops.Exp) [| a |] in
  let m_prev = Get_local m.prev and m_next = Get_local m.next and l_prev = Get_local l.prev in
  (* No comparison in the update: a masked score is [-inf], and the arithmetic is arranged so that
     [-inf] never meets itself. The rescaling reads both maxima FLOORED at the format's lowest
     finite value. While a row's prefix is entirely masked the floored max is that value on both
     sides, so the rescale factor is [exp 0 = 1] against a zero normalizer and each masked score
     contributes [exp (-inf - lowest) = 0]; the first live score [x] rescales the empty prefix by
     [exp (lowest - x) = 0] and contributes [exp 0 = 1]; a score exactly at the lowest finite value
     is what the composed form makes of it, a finite maximum contributing [exp 0] per occurrence. A
     NaN score reaches [exp (nan - m')] and poisons the normalizer, and the poison rides every later
     step ([nan * alpha + p]), as in the composed form. Having no [-inf] comparison in the update
     leaves a C compiler's finite-math licence ([cc_backend_fast_math], on in the same [approximate]
     profile) nothing to fold that a finite result depends on. *)
  let floor_at prec v = Binop (Ops.Max, (v, prec), (Constant (lowest_finite prec), prec)) in
  let m_prec = state_prec nz.m in
  let m_floor_prev = floor_at m_prec m_prev and m_floor_next = floor_at m_prec m_next in
  let l_next =
    binop Ops.Add
      (binop Ops.Mul l_prev (exp_ (binop Ops.Sub m_floor_prev m_floor_next)))
      (exp_ (binop Ops.Sub (Get_local x) m_floor_next))
  in
  (* The stored trajectory: the composed form's normalizer for a row whose max is still [-inf] is
     NaN (a sum of [exp (-inf - -inf)]). This is the recurrence's one comparison against [-inf], and
     it decides only what a row with no live score yet stores; a row's final write has a finite max
     and takes [l'] whatever a finite-math licence makes of the test. (Not the exact [l' + (m' -
     m')]: the simplifier reassociates it into [(l' + m') - m'], which cancels catastrophically --
     gh-ocannl-998.) *)
  let l_stored =
    apply_op (Ops.Ternop Ops.Where)
      [|
        Binop (Ops.Cmpeq, (m_next, m_prec), (Constant Float.neg_infinity, m_prec));
        Constant Float.nan;
        Get_local l.next;
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
        Set { tn = nz.l; idcs = nz.l_idcs; llsc = l_stored; debug = "" };
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
let normalized r ~ls tn =
  let visited = Hash_set.create (module Tn) in
  let rec go tn =
    (not (Hash_set.mem visited tn))
    &&
    (Hash_set.add visited tn;
     match definition r tn with
     | None -> false
     | Some (pos, _) ->
         let reads = reads_at r pos in
         List.exists ls ~f:(Set.mem reads) || Set.exists reads ~f:go)
  in
  go tn

let hoist r ~ls (n : nest) : LL.t option =
  let* w, wi, v, vi, w_first =
    match reduction n with
    | Some (Ops.Add, LL.Binop (Ops.Mul, (LL.Get (w, wi), _), (LL.Get (v, vi), _))) ->
        if normalized r ~ls w then Some (w, wi, v, vi, true)
        else if normalized r ~ls v then Some (v, vi, w, wi, false)
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
  (* Sound when the moved loops address distinct cells: every inner loop occurs in the target's
     indices as a PLAIN iterator ([mentions] never looks inside an affine index), so for fixed outer
     values two inner-loop tuples that differ in any symbol differ in that symbol's plain entry, and
     no two contributions moved past each other land in one cell. A loop that reaches the target
     only through an affine index, [o[i + j]], counts as inner and fails this test -- it would fold
     distinct pairs onto one cell and reorder their sum. The moved loops must not be reduction loops
     either, which the same test says: a loop absent from the target reduces. *)
  let sound =
    (not (List.is_empty inner))
    && plain_loops wi && plain_loops vi
    && List.for_all inner ~f:(fun lp -> lp.from_ <= lp.to_ && mentions lp.index n.idcs)
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

(* {1 The fused backward (gh-ocannl-1002)}

   The composed backward of [O = P v] with [P = e / l], in the training step that also holds the
   forward: A, [dP += dO * v] over the value width; B, [v.grad += P * dO]; C1, D and C2, the
   gradient of [e / l] into [e.grad] (with the per-row [l.grad] a reduction of [dP * (-1 * e / l **
   2)]); E, [n.grad = e.grad * e]; an elementwise chain from [n.grad] (the subtraction's
   pass-through, the mask, the score scale) to [dS]; K and L, [q.grad += dS * k] and [k.grad += q *
   dS]. [dP], [dS] and the chain between them are [seq, seq] buffers written and read once each.

   With [D = sum over the value width of dO * O] per row, [n.grad = P * (dP - D)] (the
   FlashAttention backward), so every pair's gradient is a function of [x], [m], [l], [dO], [v] and
   [D] alone: a per-row reduction into a minted node, then a nest over the query rows recomputing
   [p], [dp] and [ds] per key into scope locals and accumulating [q.grad], one over the keys doing
   the same and accumulating [k.grad], and one over the keys recomputing [p] alone and accumulating
   [v.grad]. Each nest's outer loops index what it writes, so none races, and each has one channel
   loop after its scalars (the shape the default GPU annotator gives lanes); the per-cell summation
   orders of the three gradients are the composed ones. *)

let backward_override : bool option ref = ref None
let set_backward_enabled b = backward_override := b

let backward_enabled () =
  match !backward_override with
  | Some b -> b
  | None -> Utils.get_global_flag ~default:false ~arg_name:"online_softmax_backward"

let backward_provenance = Tn.Site "1002:fused-backward-row-dot"

(* The per-row [D]: a node of [m]'s shape two of the backward's nests read, so it is stored -- never
   virtual, which would replay its reduction at every pair; whether it is routine scratch or a
   buffer is the placements' call. *)
let row_node ~label ~(like : Tn.t) prec =
  Hashtbl.find_or_add minted (like.Tn.uid, label) ~default:(fun () ->
      let tn =
        Tn.create ~namespace (Tn.Specified prec) ~id:(fresh_id ()) ~label:(label :: like.Tn.label)
          ~unpadded_dims:(lazy (Lazy.force like.Tn.dims))
          ~padding:(lazy None)
          ()
      in
      Tn.update_memory_mode tn Tn.Never_virtual backward_provenance;
      tn)

let one = function [ x ] -> Some x | _ -> None
let pos_of (pos, _, _) = pos

let readers r tn =
  List.filter (List.range 0 (Array.length r.stmts)) ~f:(fun pos -> Set.mem r.reads.(pos) tn)

(* The positions reading [tn] other than the statements writing it (an accumulation reads its own
   target). *)
let outside_readers r tn =
  let ws = writers r tn in
  List.filter (readers r tn) ~f:(fun pos -> not (List.mem ws pos ~equal:Int.equal))

(* [tn]'s writers are a zeroing followed by accumulating nests: its value is the sum of their
   contributions, [(zeroing, [(position, nest, contribution)])]. *)
let accumulated r tn =
  match writers r tn with
  | z :: accs when match r.stmts.(z) with LL.Zero_out tn' -> Tn.equal tn tn' | _ -> false ->
      List.map accs ~f:(fun pos ->
          let* n = r.nests.(pos) in
          match reduction n with Some (Ops.Add, rhs) -> Some (pos, n, rhs) | _ -> None)
      |> Option.all
      |> Option.map ~f:(fun accs -> (z, accs))
  | _ -> None

(* The two plain reads of a product, in its operand order. *)
let product = function
  | LL.Binop (Ops.Mul, (LL.Get (a, ai), _), (LL.Get (b, bi), _)) -> Some ((a, ai), (b, bi))
  | _ -> None

(* The read of [tn] in a product of two plain reads, and the other one. *)
let factor rhs tn =
  let* (a, ai), (b, bi) = product rhs in
  if Tn.equal a tn && not (Tn.equal b tn) then Some (ai, (b, bi))
  else if Tn.equal b tn && not (Tn.equal a tn) then Some (bi, (a, ai))
  else None

(* [reads_in] for a nest with one loop more than the reads bind: that loop is the channel [Chan c],
   of the extent the vocabulary already gives it or, the first time, of its own. *)
let with_channel (voc : vocabulary) env (n : nest) reads c =
  let* env =
    List.fold reads ~init:(Some env) ~f:(fun env (sg, idcs) ->
        let* env = env in
        bind env sg idcs)
  in
  let* lp = one (List.filter n.loops ~f:(fun lp -> Option.is_none (role_of env lp.index))) in
  let* () = Option.some_if (lp.from_ = 0 && lp.from_ <= lp.to_) () in
  let known = List.Assoc.find voc.extents ~equal:equal_role (Chan c) in
  let* () = Option.some_if (Option.for_all known ~f:(Int.equal lp.to_)) () in
  let voc =
    if Option.is_some known then voc else { voc with extents = (Chan c, lp.to_) :: voc.extents }
  in
  let env = (lp.index, Chan c) :: env in
  Option.some_if (consistent voc env n.loops) (voc, env)

let has_role sg r = List.mem (roles_of sg) r ~equal:equal_role

(* {2 Moving scalar expressions between nests}

   The fused nests re-instantiate the composed nests' own expressions -- the contributions, the
   elementwise chain, the probability's definition -- over their own loop symbols: a symbol maps to
   the one the new nest gives its role. Only plain iterators and fixed indices move; anything else
   declines. *)

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

(* Moving a nest's symbols to another nest's, through the roles [env] gives them. *)
let transplanting env (sym : role -> Idx.symbol) = function
  | Idx.Fixed_idx _ as i -> Some i
  | Idx.Iterator s -> Option.map (role_of env s) ~f:(fun r -> Idx.Iterator (sym r))
  | _ -> None

let transplant env sym ?(get = fun _ _ -> None) s =
  Option.map (map_scalar ~idx:(transplanting env sym) ~get s) ~f:fst

let transplant_idcs env sym idcs =
  Array.to_list idcs
  |> List.map ~f:(transplanting env sym)
  |> Option.all |> Option.map ~f:Array.of_list

(* The expression a read stands for, unfolded through single-writer elementwise definitions and
   constant fills (short of the [stop] nodes), with the positions of the definitions it went
   through. A definition's target symbols map positionally to the read's indices. *)
let rec unfold r ~stop ~depth (s : LL.scalar_t) : (LL.scalar_t * int list) option =
  let found = ref [] in
  let get tn idcs =
    if depth <= 0 || List.mem stop tn ~equal:Tn.equal then None
    else
      match writers r tn with
      | [ pos ] -> (
          match (r.stmts.(pos), r.nests.(pos)) with
          | LL.Zero_out _, _ ->
              found := [ pos ] :: !found;
              Some (LL.Constant 0.)
          | _, Some { llsc = LL.Constant c; _ } ->
              found := [ pos ] :: !found;
              Some (LL.Constant c)
          | _, Some n when pointwise n ->
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
                let* body, ps = unfold r ~stop ~depth:(depth - 1) body in
                found := (pos :: ps) :: !found;
                Some body
          | _ -> None)
      | _ -> None
  in
  let* s, _ = map_scalar ~idx:Option.some ~get s in
  Some (s, List.concat !found)

type backward = {
  consumed : int list;  (** The statements the fused nests replace, zeroings included. *)
  emit : int;  (** Where the fused nests go: the first of the replaced nests. *)
  code : LL.t;
}

let find_backward r (nz : normalizer) : backward option =
  let all_pos = List.range 0 (Array.length r.stmts) in
  let nests_where f =
    List.filter_map all_pos ~f:(fun pos -> Option.bind r.nests.(pos) ~f:(f pos))
  in
  let x_prec = Lazy.force nz.x.Tn.storage_prec in
  let rows_roles = roles_of nz.sig_m in
  let all_rows sg = List.for_all rows_roles ~f:(has_role sg) in
  (* 1. The probabilities [P := e / l], elementwise over the scores' roles. *)
  let* p_tn, sig_p =
    one
      (nests_where (fun pos n ->
           match n.llsc with
           | LL.Binop (Ops.Div, (LL.Get (e', ei), _), (LL.Get (l', li), _))
             when Tn.equal e' nz.e && Tn.equal l' nz.l ->
               let* env = reads_in nz.voc n [ (nz.sig_e, ei); (nz.sig_l, li) ] in
               let* def_pos, _ = definition r n.tn in
               let* sg = sign env n.idcs in
               Option.some_if (def_pos = pos && same_roles sg nz.sig_x) (n.tn, sg)
           | _ -> None))
  in
  (* 2. The value pass [O += P * v]: its target names [O], its other operand [v], and its extra loop
     the value width [Chan 0]. *)
  let* v_pos, voc, o_tn, sig_o, v_tn, sig_v =
    one
      (nests_where (fun pos n ->
           let* op, rhs = reduction n in
           let* () = Option.some_if (Ops.equal_binop op Ops.Add) () in
           let* pi, (v, vi) = factor rhs p_tn in
           let* voc, env = with_channel nz.voc [] n [ (sig_p, pi) ] 0 in
           let* sig_o = sign env n.idcs in
           let* sig_v = sign env vi in
           Option.some_if
             (all_rows sig_o && has_role sig_o (Chan 0)
             && (not (has_role sig_o Reduced))
             && has_role sig_v Reduced && has_role sig_v (Chan 0))
             (pos, voc, n.tn, sig_o, v, sig_v)))
  in
  (* 3. A: [dP += dO * v] over the value width, into a node of the scores' roles; its other operand,
     read with [O]'s roles, names [dO]. *)
  let* a_pos, a_env, a_rhs, dp_tn, sig_dp, do_tn =
    one
      (nests_where (fun pos n ->
           let* op, rhs = reduction n in
           let* () = Option.some_if (Ops.equal_binop op Ops.Add) () in
           let* vi, (d_o, doi) = factor rhs v_tn in
           let* () = Option.some_if (not (Tn.equal d_o o_tn)) () in
           let* env = reads_in voc n [ (sig_v, vi); (sig_o, doi) ] in
           let* sg = sign env n.idcs in
           Option.some_if (same_roles sg nz.sig_x) (pos, env, rhs, n.tn, sg, d_o)))
  in
  let* z_dp, dp_accs = accumulated r dp_tn in
  let* () = Option.some_if (List.equal Int.equal (List.map dp_accs ~f:pos_of) [ a_pos ]) () in
  (* 4. C1: [e.grad += dP / l]. *)
  let c1 =
    List.filter_map (outside_readers r dp_tn) ~f:(fun pos ->
        let* n = r.nests.(pos) in
        match reduction n with
        | Some (Ops.Add, LL.Binop (Ops.Div, (LL.Get (d', di), _), (LL.Get (l', li), _)))
          when Tn.equal d' dp_tn && Tn.equal l' nz.l ->
            let* env = reads_in voc n [ (sig_dp, di); (nz.sig_l, li) ] in
            let* sg = sign env n.idcs in
            Option.some_if (same_roles sg nz.sig_x) (pos, n.tn, sg)
        | _ -> None)
  in
  let* c1_pos, eg_tn, sig_eg = one c1 in
  (* D: [l.grad += dP * u] reducing the key axis, with [u] unfolding to [-1 * e / l ** 2] -- the
     division's gradient with respect to its denominator. *)
  let dn =
    List.filter_map (outside_readers r dp_tn) ~f:(fun pos ->
        let* n = r.nests.(pos) in
        match reduction n with
        | Some (Ops.Add, (LL.Binop (Ops.Mul, (a, _), (b, _)) as rhs)) ->
            let* di, _ = factor rhs dp_tn in
            let u = match a with LL.Get (d', _) when Tn.equal d' dp_tn -> b | _ -> a in
            let* env = reads_in voc n [ (sig_dp, di) ] in
            let* sg = sign env n.idcs in
            let* () = Option.some_if (same_roles sg nz.sig_m) () in
            let* u, aux = unfold r ~stop:[ nz.e; nz.l; dp_tn ] ~depth:8 u in
            let e_read = function
              | LL.Get (e', ei) -> Tn.equal e' nz.e && Option.is_some (bind env nz.sig_e ei)
              | _ -> false
            in
            let minus_e = function
              | LL.Binop (Ops.Mul, (LL.Constant c, _), (e, _))
              | LL.Binop (Ops.Mul, (e, _), (LL.Constant c, _)) ->
                  Float.equal c (-1.) && e_read e
              | _ -> false
            in
            let matches =
              match u with
              | LL.Binop
                  ( Ops.Div,
                    (num, _),
                    (LL.Binop (Ops.ToPowOf, (LL.Get (l', li), _), (LL.Constant 2., _)), _) ) ->
                  minus_e num && Tn.equal l' nz.l && Option.is_some (bind env nz.sig_l li)
              | _ -> false
            in
            Option.some_if matches (pos, n.tn, sg, aux)
        | _ -> None)
  in
  let* d_pos, lg_tn, sig_lg, aux = one dn in
  let* () =
    Option.some_if
      (List.equal Int.equal
         (List.sort (outside_readers r dp_tn) ~compare:Int.compare)
         (List.sort [ c1_pos; d_pos ] ~compare:Int.compare))
      ()
  in
  (* C2: [e.grad += l.grad], broadcast over the key axis. *)
  let* z_lg, lg_accs = accumulated r lg_tn in
  let* () = Option.some_if (List.equal Int.equal (List.map lg_accs ~f:pos_of) [ d_pos ]) () in
  let* z_eg, eg_accs = accumulated r eg_tn in
  let* c2_pos =
    match List.map eg_accs ~f:pos_of with
    | [ p1; p2 ] when p1 = c1_pos || p2 = c1_pos -> (
        let c2 = if p1 = c1_pos then p2 else p1 in
        let* n = r.nests.(c2) in
        match reduction n with
        | Some (Ops.Add, LL.Get (lg', lgi)) when Tn.equal lg' lg_tn ->
            let* _ = reads_in voc n [ (sig_eg, n.idcs); (sig_lg, lgi) ] in
            Some c2
        | _ -> None)
    | _ -> None
  in
  let* () =
    Option.some_if (d_pos < c2_pos && List.equal Int.equal (outside_readers r lg_tn) [ c2_pos ]) ()
  in
  (* E: [n.grad += e.grad * e]. *)
  let* e_pos = one (outside_readers r eg_tn) in
  let* e_nest = r.nests.(e_pos) in
  let* ng_tn, sig_ng =
    match reduction e_nest with
    | Some (Ops.Add, rhs) ->
        let* egi, (e', ei) = factor rhs eg_tn in
        let* () = Option.some_if (Tn.equal e' nz.e) () in
        let* env = reads_in voc e_nest [ (sig_eg, egi); (nz.sig_e, ei) ] in
        let* sg = sign env e_nest.idcs in
        Option.some_if (same_roles sg nz.sig_x && c1_pos < e_pos && c2_pos < e_pos) (e_nest.tn, sg)
    | _ -> None
  in
  let* z_ng, ng_accs = accumulated r ng_tn in
  let* () = Option.some_if (List.equal Int.equal (List.map ng_accs ~f:pos_of) [ e_pos ]) () in
  (* 5. The elementwise chain from [n.grad]: each link the one reader of the previous gradient, an
     accumulation onto a zeroed node of the scores' roles; it ends at the node two contractions
     read, [dS]. A composed max gradient reads [n.grad] too and adds a second writer to the scores'
     gradient, so it declines here. *)
  let rec chain cur sig_cur links depth =
    match outside_readers r cur with
    | [ pos ] when depth > 0 ->
        let* n = r.nests.(pos) in
        let* z, accs = accumulated r n.tn in
        let* _, _, rhs = one accs in
        let gets = ref [] in
        let* _ =
          map_scalar ~idx:Option.some
            ~get:(fun tn i ->
              if Tn.equal tn cur then gets := i :: !gets;
              None)
            rhs
        in
        let* () = Option.some_if (not (List.is_empty !gets)) () in
        let* env = reads_in voc n (List.map !gets ~f:(fun i -> (sig_cur, i))) in
        let* sg = sign env n.idcs in
        let* () = Option.some_if (same_roles sg nz.sig_x) () in
        chain n.tn sg ((pos, z, env, rhs, cur, n.tn) :: links) (depth - 1)
    | [ p1; p2 ] -> Some (cur, sig_cur, List.rev links, p1, p2)
    | _ -> None
  in
  let* ds_tn, sig_ds, links, p1, p2 = chain ng_tn sig_ng [] 16 in
  (* 6. K and L: [q.grad += dS * k] (its target has every row role) and [k.grad += q * dS] (the key
     role, not every row role), each over the head width [Chan 1]. *)
  let contraction pos =
    let* n = r.nests.(pos) in
    let* op, rhs = reduction n in
    let* () = Option.some_if (Ops.equal_binop op Ops.Add) () in
    let* dsi, (w, wi) = factor rhs ds_tn in
    let* voc1, env = with_channel voc [] n [ (sig_ds, dsi) ] 1 in
    let* sg = sign env n.idcs in
    let* sig_w = sign env wi in
    Some (pos, n, (voc1, env), sg, w, sig_w)
  in
  let* k1 = contraction p1 in
  let* k2 = contraction p2 in
  let is_dq (_, _, _, sg, _, sig_w) =
    all_rows sg && has_role sg (Chan 1) && (not (has_role sg Reduced)) && has_role sig_w Reduced
  in
  let is_dk (_, _, _, sg, _, sig_w) =
    has_role sg Reduced && has_role sg (Chan 1)
    && (not (all_rows sg))
    && all_rows sig_w
    && not (has_role sig_w Reduced)
  in
  let* kq, kk =
    if is_dq k1 && is_dk k2 then Some (k1, k2)
    else if is_dq k2 && is_dk k1 then Some (k2, k1)
    else None
  in
  let kq_pos, kq_nest, (kq_voc, kq_env), _, k_op, _ = kq in
  let kk_pos, kk_nest, (kk_voc, kk_env), sig_kg, q_op, _ = kk in
  (* The two operands are the score reduction's, reached from [x] through elementwise
     definitions. *)
  let rec score_operands visited tn =
    if Set.mem visited tn then (visited, [])
    else
      let visited = Set.add visited tn in
      match definition r tn with
      | Some (pos, _) ->
          Set.fold (reads_at r pos) ~init:(visited, []) ~f:(fun (visited, acc) tn ->
              let visited, found = score_operands visited tn in
              (visited, found @ acc))
      | None -> (
          match accumulated r tn with
          | Some (_, [ (_, _, rhs) ]) -> (
              match product rhs with
              | Some ((a, _), (b, _)) -> (visited, [ (a, b) ])
              | None -> (visited, []))
          | _ -> (visited, []))
  in
  let _, operand_pairs = score_operands (Set.empty (module Tn)) nz.x in
  let* () =
    Option.some_if
      (List.exists operand_pairs ~f:(fun (a, b) ->
           (Tn.equal a q_op && Tn.equal b k_op) || (Tn.equal a k_op && Tn.equal b q_op)))
      ()
  in
  (* 7. B: [v.grad += P * dO], owned like [k.grad] by the key role. *)
  let key_roles sg =
    List.filter (roles_of sg) ~f:(function Chan _ -> false | _ -> true)
    |> List.sort ~compare:compare_role
  in
  let* b_pos, b_nest, b_env, sig_vg =
    one
      (List.filter_map (outside_readers r p_tn) ~f:(fun pos ->
           let* n = Option.some_if (pos <> v_pos) () |> Option.bind ~f:(fun () -> r.nests.(pos)) in
           let* op, rhs = reduction n in
           let* () = Option.some_if (Ops.equal_binop op Ops.Add) () in
           let* pi, (d_o, doi) = factor rhs p_tn in
           let* () = Option.some_if (Tn.equal d_o do_tn) () in
           let* env = reads_in voc n [ (sig_p, pi); (sig_o, doi) ] in
           let* sg = sign env n.idcs in
           Option.some_if
             (has_role sg (Chan 0) && List.equal equal_role (key_roles sg) (key_roles sig_kg))
             (pos, n, env, sg)))
  in
  (* {2 The contract} *)
  let link_pos (pos, _, _, _, _, _) = pos in
  let intermediates =
    [ dp_tn; eg_tn; lg_tn; ng_tn ] @ List.map links ~f:(fun (_, _, _, _, _, tn) -> tn)
  in
  let nests =
    [ a_pos; b_pos; c1_pos; d_pos; c2_pos; e_pos ] @ List.map links ~f:link_pos @ [ kq_pos; kk_pos ]
  in
  let zeros = [ z_dp; z_eg; z_lg; z_ng ] @ List.map links ~f:(fun (_, z, _, _, _, _) -> z) in
  let emit = List.fold nests ~init:Int.max_value ~f:Int.min in
  let last = List.fold nests ~init:Int.min_value ~f:Int.max in
  let consumed_so_far = Set.of_list (module Int) (nests @ zeros) in
  let outputs = [ (kq_nest.tn, kq_pos); (kk_nest.tn, kk_pos); (b_nest.tn, b_pos) ] in
  let is_output tn = List.exists outputs ~f:(fun (g, _) -> Tn.equal g tn) in
  let before_emit tn = List.for_all (writers r tn) ~f:(fun w -> w < emit) in
  (* A scalar node the chain reads that is (re)filled after the fused nests' position -- the zero of
     a masked branch, say -- enters them as its constant. *)
  let constant_of tn =
    match writers r tn with
    | [ pos ] -> (
        match (r.stmts.(pos), r.nests.(pos)) with
        | LL.Zero_out _, _ -> Some (0., pos)
        | _, Some { llsc = LL.Constant c; _ } -> Some (c, pos)
        | _ -> None)
    | _ -> None
  in
  let side_nodes =
    List.concat_map links ~f:(fun (_, _, _, rhs, cur, _) ->
        let sides = ref [] in
        ignore
          (map_scalar ~idx:Option.some
             ~get:(fun tn _ ->
               if not (Tn.equal tn cur) then sides := tn :: !sides;
               None)
             rhs);
        !sides)
    |> List.dedup_and_sort ~compare:Tn.compare
  in
  let substituted = List.filter side_nodes ~f:(fun tn -> not (before_emit tn)) in
  let side_get cur expr tn _ =
    if Tn.equal tn cur then Some expr
    else if List.mem substituted tn ~equal:Tn.equal then
      Option.map (constant_of tn) ~f:(fun (c, _) -> LL.Constant c)
    else None
  in
  let inputs = [ nz.x; nz.m; nz.l; v_tn; do_tn; o_tn; q_op; k_op ] @ side_nodes in
  let read_inputs =
    List.filter inputs ~f:(fun tn -> not (List.mem substituted tn ~equal:Tn.equal))
  in
  let grad_prec = Lazy.force dp_tn.Tn.storage_prec in
  let float_at prec tn =
    let prec' = Lazy.force tn.Tn.storage_prec in
    Ops.is_float prec' && Ops.equal_prec prec' prec
  in
  let untouched_between lo hi tn =
    List.for_all
      (List.range (lo + 1) hi)
      ~f:(fun pos ->
        Set.mem consumed_so_far pos
        || not (Set.mem (reads_at r pos) tn || List.mem (writers r tn) pos ~equal:Int.equal))
  in
  let contract =
    (* One precision per side: the probabilities at the scores' (the forward's chain), the gradient
       chain from dP to dS at one of its own -- a node narrower or wider than its neighbours rounds
       where the fused locals would not. *)
    float_at x_prec p_tn
    && List.for_all intermediates ~f:(float_at grad_prec)
    (* A requested intermediate keeps its composed definition: the fusion never writes it. *)
    && List.for_all intermediates ~f:(fun tn -> not (Tn.known_non_virtual tn))
    (* [O] is the value pass alone, so that [D] is the row sum of [P * dP]. *)
    && Option.equal (List.equal Int.equal)
         (Option.map (accumulated r o_tn) ~f:(fun (_, accs) -> List.map accs ~f:pos_of))
         (Some [ v_pos ])
    (* Everything the fused nests read is final where they run... *)
    && List.for_all inputs ~f:(fun tn ->
        before_emit tn
        || (List.mem substituted tn ~equal:Tn.equal && Option.is_some (constant_of tn)))
    && (not (List.exists inputs ~f:is_output))
    && (not (List.exists inputs ~f:(List.mem intermediates ~equal:Tn.equal)))
    (* ...and nothing else touches a gradient they accumulate into between there and the composed
       contraction that did. *)
    && List.for_all outputs ~f:(fun (g, c) -> untouched_between emit c g)
    (* No code the census cannot see, and no fence, in the span. *)
    && not (List.exists (List.range emit (last + 1)) ~f:(fun pos -> r.opaque.(pos)))
  in
  let* () = Option.some_if contract () in
  (* The definitions only the replaced nests read -- [l ** 2], [-1 * e] and their constants, the
     masked branch's zero -- go with them. *)
  let aux_candidates =
    aux @ List.filter_map substituted ~f:(fun tn -> Option.map (constant_of tn) ~f:snd)
  in
  let target_of pos =
    match (r.stmts.(pos), r.nests.(pos)) with
    | LL.Zero_out tn, _ -> Some tn
    | _, Some n -> Some n.tn
    | _ -> None
  in
  let rec closure consumed =
    let added =
      List.filter aux_candidates ~f:(fun pos ->
          (not (Set.mem consumed pos))
          &&
          match target_of pos with
          | Some tn ->
              List.equal Int.equal (writers r tn) [ pos ]
              && List.for_all (readers r tn) ~f:(Set.mem consumed)
              && (not (List.mem read_inputs tn ~equal:Tn.equal))
              && not (Tn.known_non_virtual tn)
          | None -> false)
    in
    if List.is_empty added then consumed
    else closure (Set.union consumed (Set.of_list (module Int) added))
  in
  let consumed = closure consumed_so_far in
  (* {2 Emission} *)
  let open LL in
  (* The locals at the wider of the two sides' state precisions: f32 under narrow storage, f64 where
     either side is double. *)
  let prec =
    match (state_prec nz.x, state_prec dp_tn) with
    | (Ops.Double_prec _ as p), _ | _, (Ops.Double_prec _ as p) -> p
    | p, _ -> p
  in
  let p_node = scalar_node ~label:"bwd_probability" ~like:p_tn prec in
  let dp_node = scalar_node ~label:"bwd_dprob" ~like:dp_tn prec in
  let ds_node = scalar_node ~label:"bwd_dscore" ~like:ds_tn prec in
  let acc_node = scalar_node ~label:"bwd_rowdot_acc" ~like:do_tn prec in
  let d_node = row_node ~label:"bwd_rowdot" ~like:nz.m prec in
  let extent (voc : vocabulary) role = List.Assoc.find_exn voc.extents ~equal:equal_role role in
  let fresh roles = List.map roles ~f:(fun r -> (r, Idx.get_symbol ())) in
  let symbols () = fresh (rows_roles @ [ Reduced; Chan 0; Chan 1 ]) in
  let sym_of syms r = List.Assoc.find_exn syms ~equal:equal_role r in
  let idcs_of sym sg =
    Array.map sg ~f:(function Fixed k -> Idx.Fixed_idx k | Role r -> Idx.Iterator (sym r))
  in
  let loop sym to_ body = For_loop { index = sym; from_ = 0; to_; body; axis = Serial } in
  let row_role (lp : loop) = Option.value_exn (role_of nz.voc.env lp.index) in
  let loops_over sym roles_extents body =
    List.fold_right roles_extents ~init:body ~f:(fun (r, to_) body -> loop (sym r) to_ body)
  in
  let rows_extents = List.map nz.rows ~f:(fun lp -> (row_role lp, lp.to_)) in
  let add a b = apply_op (Ops.Binop Ops.Add) [| a; b |] in
  let mul a b = apply_op (Ops.Binop Ops.Mul) [| a; b |] in
  let sub a b = apply_op (Ops.Binop Ops.Sub) [| a; b |] in
  let ext_v = extent voc (Chan 0) in
  (* [D] per row. *)
  let d_code =
    let syms = symbols () in
    let sym = sym_of syms in
    let acc = get_scope acc_node in
    loops_over sym rows_extents
      (unflat_lines
         [
           Declare_local { id = acc; needs_init = false };
           Set_local (acc, Constant 0.);
           loop (sym (Chan 0)) ext_v
             (Set_local
                ( acc,
                  add (Get_local acc)
                    (mul (Get (do_tn, idcs_of sym sig_o)) (Get (o_tn, idcs_of sym sig_o))) ));
           Set { tn = d_node; idcs = idcs_of sym nz.sig_m; llsc = Get_local acc; debug = "" };
         ])
  in
  (* One pair's scalars, outside the channel loop: [p] as the forward defines it, instantiated at
     this nest's subscripts down to [stop]; with [~grad], [dp] as A sums it and [ds] through the
     recovered chain. Each nest stops the instantiation at a different node of the probability's
     chain -- [e], [n], and past [x] to the scores' reduction -- because the visit cap counts a read
     in a [Set_local] (and exempts one in a [Set] at its own write position): the scan already reads
     [x], and a second counted read of any one node of the chain would store it as a [seq, seq]
     buffer. *)
  let pair ~stop ~grad sym =
    let p = get_scope p_node and dp = get_scope dp_node and ds = get_scope ds_node in
    let* p_val, _ =
      unfold r ~stop:(stop @ [ nz.m; nz.l ]) ~depth:8 (Get (p_tn, idcs_of sym sig_p))
    in
    (* Read where the fused nests run: only nodes final there. *)
    let p_reads = ref [] in
    let* _ =
      map_scalar ~idx:Option.some
        ~get:(fun tn _ ->
          p_reads := tn :: !p_reads;
          None)
        p_val
    in
    let* () = Option.some_if (List.for_all !p_reads ~f:before_emit) () in
    let p_code = [ Declare_local { id = p; needs_init = false }; Set_local (p, p_val) ] in
    if not grad then Some (p_code, p, ds)
    else
      (* [dp]'s value-width loop binds a symbol of its own: one bound by two sibling loops makes the
         routine uncacheable. It sits in the lane-uniform preamble, which is why the dQ and dK nests
         get no lane geometry from the default GPU annotator (a preamble must be loop-free, since
         every lane would recompute it); see the gh-1002/1003 record. *)
      let e_sym = Idx.get_symbol () in
      let dp_sym = function Chan 0 -> e_sym | r -> sym r in
      let* dp_term = transplant a_env dp_sym a_rhs in
      let ds0 = mul (Get_local p) (sub (Get_local dp) (Get (d_node, idcs_of sym nz.sig_m))) in
      let* ds_val =
        List.fold links ~init:(Some ds0) ~f:(fun expr (_, _, env, rhs, cur, _) ->
            let* expr = expr in
            transplant env sym ~get:(side_get cur expr) rhs)
      in
      Some
        ( p_code
          @ [
              Declare_local { id = dp; needs_init = false };
              Set_local (dp, Constant 0.);
              loop e_sym ext_v (Set_local (dp, add (Get_local dp) dp_term));
              Declare_local { id = ds; needs_init = false };
              Set_local (ds, ds_val);
            ],
          p,
          ds )
  in
  let accumulate env sym (n : nest) ~replaced ~by =
    let* idcs = transplant_idcs env sym n.idcs in
    let* llsc =
      transplant env sym
        ~get:(fun tn _ -> Option.some_if (Tn.equal tn replaced) (Get_local by))
        n.llsc
    in
    Some (Set { tn = n.tn; idcs; llsc; debug = "" })
  in
  (* Three nests, each owning what it writes, each with ONE channel loop after its scalars -- the
     shape the default GPU annotator's lane geometry reads (gh-ocannl-1003 stage 1). dQ: the query
     rows outer, the keys serial. *)
  let* dq_code =
    let sym = sym_of (symbols ()) in
    let* pair_code, _, ds = pair ~stop:[ nz.e ] ~grad:true sym in
    let* dq = accumulate kq_env sym kq_nest ~replaced:ds_tn ~by:ds in
    Some
      (loops_over sym rows_extents
         (loop (sym Reduced) (extent nz.voc Reduced)
            (unflat_lines (pair_code @ [ loop (sym (Chan 1)) (extent kq_voc (Chan 1)) dq ]))))
  in
  (* dK and dV: the key and the rows the target keeps outer, the other query rows serial. *)
  let key_nest ~target_env ~(target : nest) ~sg body_of =
    let sym = sym_of (symbols ()) in
    let outer =
      List.filter_map target.loops ~f:(fun lp ->
          match role_of target_env lp.index with
          | Some (Chan _) | None -> None
          | Some r -> Option.some_if (has_role sg r) (r, lp.to_))
    in
    let inner = List.filter rows_extents ~f:(fun (r, _) -> not (has_role sg r)) in
    let* body = body_of sym in
    Some (loops_over sym outer (loops_over sym inner (unflat_lines body)))
  in
  let* dk_code =
    key_nest ~target_env:kk_env ~target:kk_nest ~sg:sig_kg (fun sym ->
        let* pair_code, _, ds = pair ~stop:[ nz.n ] ~grad:true sym in
        let* dk = accumulate kk_env sym kk_nest ~replaced:ds_tn ~by:ds in
        Some (pair_code @ [ loop (sym (Chan 1)) (extent kk_voc (Chan 1)) dk ]))
  in
  (* dV needs [p] alone: a loop-free preamble wherever the scores' placement leaves [p]'s
     instantiation loop-free (the scores stored), hence lane-eligible. Its instantiation runs past
     [x] to the scores' reduction. *)
  let* dv_code =
    key_nest ~target_env:b_env ~target:b_nest ~sg:sig_vg (fun sym ->
        let* pair_code, p, _ = pair ~stop:[] ~grad:false sym in
        let* dv = accumulate b_env sym b_nest ~replaced:p_tn ~by:p in
        Some (pair_code @ [ loop (sym (Chan 0)) ext_v dv ]))
  in
  Some
    {
      consumed = Set.to_list consumed;
      emit;
      code = unflat_lines [ d_code; dq_code; dk_code; dv_code ];
    }

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
    (* The fused backward consumes its nests before the hoist sees them: B reads the probabilities
       and would otherwise be hoisted in place. Two attention blocks never share a nest, but a match
       overlapping an earlier one is dropped rather than trusted. *)
    let consumed =
      if not (backward_enabled ()) then Set.empty (module Int)
      else
        List.fold normalizers
          ~init:(Set.empty (module Int))
          ~f:(fun consumed nz ->
            match find_backward r nz with
            | Some b when not (List.exists b.consumed ~f:(Set.mem consumed)) ->
                List.iter b.consumed ~f:(fun pos -> stmts.(pos) <- LL.Noop);
                stmts.(b.emit) <- b.code;
                Set.union consumed (Set.of_list (module Int) b.consumed)
            | _ -> consumed)
    in
    let ls = List.map normalizers ~f:(fun nz -> nz.l) in
    Array.iteri r.nests ~f:(fun pos -> function
      | Some n
        when (not (Set.mem consumed pos))
             && not (List.exists normalizers ~f:(fun nz -> pos = nz.a || pos = nz.c)) -> (
          match hoist r ~ls n with Some replacement -> stmts.(pos) <- replacement | None -> ())
      | _ -> ());
    LL.unflat_lines (Array.to_list stmts)
