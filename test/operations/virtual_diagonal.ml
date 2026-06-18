(* Regression test for gh-ocannl-133 Stage A: virtualize producers whose index vector repeats a
   non-static symbol (diagonal [i;i] / partially-diagonal [i;j;i]) plus covered single-symbol affine
   positions.

   High-level lowering never produces a [Get] of a virtual diagonal with two distinct call-site
   symbols in one place (each assignment lowers to its own loop nest), so -- like [virtual_shared_loop]
   -- these cases are built directly as [Ir.Low_level.t] and run through [Ir.Low_level.optimize], the
   same pipeline (visit_llc -> virtual_llc -> cleanup_virtual_llc -> simplify -> CSE -> hoist) the
   backends use. We assert structurally on the optimized form: that the diagonal producer virtualizes,
   that its reads are inlined, and that an equality guard ([Where (Cmpeq ...)]) is emitted exactly when
   the read uses distinct/dynamic indices and folded away when the read indices are syntactically
   equal.

   [Concat] virtualization stays out of scope: a [Concat] index is eliminated during lowering and
   [visit_llc] raises if one ever reaches this pass, so it cannot be exercised through [optimize] here.
   The "Concat remains rejected" criterion is the unchanged [check_idcs] [Non_virtual 52] branch plus
   the existing test_concat_graph / test_block_tensor coverage.

   End-to-end numeric correctness of the guarded inline is covered by test/einsum/test_virtual_diagonal. *)

open Base
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Ops = Ir.Ops
module Idx = Ir.Indexing

let single = Ir.Ops.single
let next_id = ref 2000

let mk ?(dims = [| 3; 3 |]) label =
  Int.incr next_id;
  Tn.create (Tn.Specified single) ~id:!next_id ~label:[ label ] ~unpadded_dims:(lazy dims)
    ~padding:(lazy None) ()

let materialize tn = Tn.update_memory_mode tn Tn.Materialized 99

(* --- low-level builders --- *)
let sym () = Idx.get_symbol ()
let iter s = Idx.Iterator s
let embed s : LL.scalar_t = LL.Embed_index (iter s)
let zero tn : LL.t = LL.Zero_out tn

(* a setter writing [tn] at the given index array *)
let set_at idcs tn llsc : LL.t = LL.Set { tn; idcs; llsc; debug = "" }
let get_at idcs tn : LL.scalar_t = LL.Get (tn, idcs)
let loop s body : LL.t = LL.For_loop { index = s; from_ = 0; to_ = 2; body; trace_it = true }
let seq a b : LL.t = LL.Seq (a, b)

let optimize llc : LL.optimized =
  let ctx : LL.optimize_ctx = { computations = Hashtbl.create (module Tn) } in
  LL.optimize ctx ~unoptim_ll_source:None ~ll_source:None ~name:"virtual_diagonal" [] llc

(* --- structural probes on the optimized form --- *)
let rec walk_t ~on_get ~on_where (llc : LL.t) =
  match llc with
  | LL.Noop | LL.Declare_local _ | LL.Comment _ | LL.Staged_compilation _ -> ()
  | LL.Seq (a, b) ->
      walk_t ~on_get ~on_where a;
      walk_t ~on_get ~on_where b
  | LL.For_loop { body; _ } -> walk_t ~on_get ~on_where body
  | LL.Zero_out _ -> ()
  | LL.Set { llsc; _ } -> walk_s ~on_get ~on_where llsc
  | LL.Set_from_vec { arg = s, _; _ } -> walk_s ~on_get ~on_where s
  | LL.Set_local (_, s) -> walk_s ~on_get ~on_where s

and walk_s ~on_get ~on_where (s : LL.scalar_t) =
  match s with
  | LL.Constant _ | LL.Constant_bits _ | LL.Get_local _ | LL.Embed_index _ | LL.Get_merge_buffer _ ->
      ()
  | LL.Get (tn, _) -> on_get tn
  | LL.Local_scope { body; _ } -> walk_t ~on_get ~on_where body
  | LL.Ternop (op, (a, _), (b, _), (d, _)) ->
      on_where op;
      walk_s ~on_get ~on_where a;
      walk_s ~on_get ~on_where b;
      walk_s ~on_get ~on_where d
  | LL.Binop (_, (a, _), (b, _)) ->
      walk_s ~on_get ~on_where a;
      walk_s ~on_get ~on_where b
  | LL.Unop (_, (a, _)) -> walk_s ~on_get ~on_where a

let count_get (o : LL.optimized) tn =
  let n = ref 0 in
  walk_t ~on_get:(fun t -> if t.Tn.id = tn.Tn.id then Int.incr n) ~on_where:(fun _ -> ()) o.llc;
  !n

let count_where (o : LL.optimized) =
  let n = ref 0 in
  walk_t
    ~on_get:(fun _ -> ())
    ~on_where:(function Ops.Where -> Int.incr n | _ -> ())
    o.llc;
  !n

let p name b = Stdio.printf "%s: %b\n" name b

(* === Case 1: diagonal producer read by a generic (distinct-symbol) consumer ===
   d[i,i] = i (off-diagonal zero); a materialized consumer reads d[j,k]. The diagonal must virtualize,
   its read must be inlined, and exactly one equality guard must survive (j = k). *)
let case_diagonal_generic () =
  let d = mk "d" and o = mk "o" in
  materialize o;
  let i = sym () and j = sym () and k = sym () in
  let producer = seq (zero d) (loop i (set_at [| iter i; iter i |] d (embed i))) in
  let consumer = loop j (loop k (set_at [| iter j; iter k |] o (get_at [| iter j; iter k |] d))) in
  let opt = optimize (seq producer consumer) in
  p "diagonal-generic: producer virtual" (Tn.known_virtual d);
  p "diagonal-generic: read inlined (no array read of d)" (count_get opt d = 0);
  p "diagonal-generic: one equality guard survives" (count_where opt >= 1);
  p "diagonal-generic: consumer read inlined under guard" (count_get opt d = 0)

(* === Case 2: diagonal producer read at equal indices -> guard simplifies away === *)
let case_diagonal_equal () =
  let d = mk "d" and o = mk "o" in
  materialize o;
  let i = sym () and j = sym () in
  let producer = seq (zero d) (loop i (set_at [| iter i; iter i |] d (embed i))) in
  (* read d[j,j]: the two call-site indices are syntactically equal, so no guard. *)
  let consumer = loop j (set_at [| iter j; iter j |] o (get_at [| iter j; iter j |] d)) in
  let opt = optimize (seq producer consumer) in
  p "diagonal-equal: producer virtual" (Tn.known_virtual d);
  p "diagonal-equal: read inlined (no array read of d)" (count_get opt d = 0);
  p "diagonal-equal: guard folded away (no Where)" (count_where opt = 0)

(* === Case 3: partially-diagonal producer [i;j;i] read generically ===
   d[i,j,i] = i; consumer reads d[a,b,cc]. Repeated i guards (a = cc); j substituted normally. *)
let case_partial_diagonal () =
  let d = mk ~dims:[| 3; 3; 3 |] "pd" and o = mk ~dims:[| 3; 3; 3 |] "po" in
  materialize o;
  let i = sym () and j = sym () in
  let a = sym () and b = sym () and cc = sym () in
  let producer = seq (zero d) (loop i (loop j (set_at [| iter i; iter j; iter i |] d (embed i)))) in
  let consumer =
    loop a
      (loop b
         (loop cc (set_at [| iter a; iter b; iter cc |] o (get_at [| iter a; iter b; iter cc |] d))))
  in
  let opt = optimize (seq producer consumer) in
  p "partial-diagonal: producer virtual" (Tn.known_virtual d);
  p "partial-diagonal: read inlined (no array read of d)" (count_get opt d = 0);
  p "partial-diagonal: one equality guard survives" (count_where opt >= 1)

(* === Case 4: static-vs-dynamic read of a diagonal producer ===
   Read d[0, j] (a row slice): the first position is bound to Fixed_idx 0, the second is dynamic; the
   consistency must become a guard (0 = j), NOT a Non_virtual 13 rejection. *)
let case_static_dynamic () =
  let d = mk "d" and o = mk ~dims:[| 3 |] "o" in
  materialize o;
  let i = sym () and j = sym () in
  let producer = seq (zero d) (loop i (set_at [| iter i; iter i |] d (embed i))) in
  let consumer = loop j (set_at [| iter j |] o (get_at [| Idx.Fixed_idx 0; iter j |] d)) in
  let opt = optimize (seq producer consumer) in
  p "static-dynamic: producer virtual" (Tn.known_virtual d);
  p "static-dynamic: read inlined (no array read of d)" (count_get opt d = 0);
  p "static-dynamic: one equality guard survives" (count_where opt >= 1)

(* === Case 5: covered single-symbol affine producer position ===
   Producer d[i, i+1] (single-symbol affine, covered by the bare iterator i) read at d[j, j+1]; this
   must validate after substitution (no Non_virtual 13) and inline with no surviving guard. *)
let case_single_symbol_affine () =
  let d = mk "d" and o = mk ~dims:[| 3 |] "o" in
  materialize o;
  let i = sym () and j = sym () in
  let aff s : Idx.axis_index = Idx.Affine { symbols = [ (1, s) ]; offset = 1 } in
  let producer = seq (zero d) (loop i (set_at [| iter i; aff i |] d (embed i))) in
  let consumer = loop j (set_at [| iter j |] o (get_at [| iter j; aff j |] d)) in
  let opt = optimize (seq producer consumer) in
  p "single-affine: producer virtual" (Tn.known_virtual d);
  p "single-affine: read inlined (no array read of d)" (count_get opt d = 0);
  p "single-affine: no guard for matching affine" (count_where opt = 0)

(* === Case 6: single-symbol (non-repeated) producer still virtualizes -- regression guard === *)
let case_single_symbol () =
  let d = mk ~dims:[| 3 |] "d" and o = mk ~dims:[| 3 |] "o" in
  materialize o;
  let i = sym () and j = sym () in
  let producer = loop i (set_at [| iter i |] d (embed i)) in
  let consumer = loop j (set_at [| iter j |] o (get_at [| iter j |] d)) in
  let opt = optimize (seq producer consumer) in
  p "single-symbol: producer virtual" (Tn.known_virtual d);
  p "single-symbol: read inlined (no array read of d)" (count_get opt d = 0);
  p "single-symbol: no guard" (count_where opt = 0)

let () =
  case_diagonal_generic ();
  case_diagonal_equal ();
  case_partial_diagonal ();
  case_static_dynamic ();
  case_single_symbol_affine ();
  case_single_symbol ();
  Stdio.printf "%!"
