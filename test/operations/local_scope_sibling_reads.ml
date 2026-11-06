(* Regression test for gh-ocannl-1895: a scope body reading a sibling scope's local.

   The scalar CSE keeps one copy of a repeated scope and replaces the others with [Get_local], also
   inside the body of a sibling scope. Such a body has to run after the definition it reads. Two
   places broke that, and in both the kernel failed to compile:

   - [C_syntax.pp_local_defs] emitted the definitions by ascending [scope_id] alone, so a body with
   the lower id read its sibling's local ahead of the declaration ([v27_rw_x] in
   [rewind_to_last_closure], with the literal-table exclusion of [instantiate_computations] lifted);
   - [hoist_cross_statement_cse] remapped the [Get_local]s of a hoisted scope in the statement's
   expression, but not inside scope bodies, so a body kept reading a local nothing declares.

   Each row runs the same program twice, with its intermediates virtual and materialized, and
   compares both against a closed form. A structural claim pins that the row reaches the shape, so
   no row can go vacuous if a pass stops producing it. *)

open Base
open Ll_test
open Verdict.Claims
module Sched = Ir.Schedule

let mk = node_factory ~first_id:1_895_000 ~dims:[| 3 |] ()
let n = 3

(* Discriminating: every cell differs from the others, from the zero init and from the prefill. *)
let weights = [| 2.; 3.; 5. |]
let table = [| 5.; 7.; 11. |]

(* A literal in the in-kernel spelling of [Tensor.ndarray], one constant store per cell. The
   [Zero_out] is what still lets a read at a symbolic index replay it on master: the union proof
   leaves literal tables out, a zeroed node does not need it. *)
let literal x =
  LL.unflat_lines (zero x :: List.init n ~f:(fun k -> set_at x (fixed k) (LL.Constant table.(k))))

(* The same per-cell shape with computed values, served by the union proof. *)
let computed x w =
  LL.unflat_lines (List.init n ~f:(fun k -> set_at x (fixed k) (add (get w [| fixed k |]) (c 1.))))

(* [d[i] = x[i] * w[i] + x[i]], in two accumulations over a zeroed node, so that its scope has a
   body too long to collapse and reads [x] from inside it, as a gradient does. *)
let reader d ~x ~w =
  let s = sym () and t = sym () in
  LL.unflat_lines
    [
      zero d;
      loop_n s n
        (set_at d (iter s)
           (add (get d [| iter s |]) (mul (get x [| iter s |]) (get w [| iter s |]))));
      loop_n t n (set_at d (iter t) (add (get d [| iter t |]) (get x [| iter t |])));
    ]

(* [r[i] = d[i]]. A scope's id is minted when its read is walked, so a read in a stored setter gets
   its id ahead of every read in a later statement. Inlined, this relay's single assignment
   collapses and leaves [d]'s older scope standing beside whatever the reading statement mints, the
   order a gradient chain produces: [v25_d.grad] beside [v27_rw_x]. *)
let relay r d =
  let s = sym () in
  loop_n s n (set_at r (iter s) (get d [| iter s |]))

(* The reads of [llc] that reach a scope bound outside the body they sit in, as [(body's id, read
   id)] pairs of the outermost enclosing scope. Reads of a binder inside the body do not count. *)
let outward_reads (llc : LL.t) =
  let pairs = ref [] in
  let visit_scope (id : LL.scope_id) body =
    let bound = ref [ id.scope_id ] in
    walk body
      ~on_scope:(fun (b : LL.scope_id) -> bound := b.scope_id :: !bound)
      ~on_stmt:(function
        | LL.Declare_local { id = b; _ } -> bound := b.scope_id :: !bound | _ -> ());
    walk body ~on_scalar:(function
      | LL.Get_local r when not (List.mem !bound r.scope_id ~equal:Int.equal) ->
          pairs := (id.scope_id, r.scope_id) :: !pairs
      | _ -> ())
  in
  walk llc ~in_scopes:false ~on_scalar:(function
    | LL.Local_scope { id; body; _ } -> visit_scope id body
    | _ -> ());
  !pairs

(* Claims that every [Get_local] of [llc] names a binder of [llc], whatever depth either sits at:
   the compile gate's own predicate (gh-ocannl-1973), asked of code that does read a local. *)
let every_read_bound name (llc : LL.t) =
  p
    (name ^ ": every local a body reads has a binder")
    (count_scalar llc ~f:(function LL.Get_local _ -> true | _ -> false) >= 1
    && Option.is_none (LL.unbound_local_violation llc))

(* Runs [llc] with [virtuals] virtual, then again with them materialized, seeding [w] and reading
   [outs]. Materialized, a virtual's prefill is the harness's: off the zero init and per cell. *)
let both_arms ~name ~virtuals ~w ~outs llc =
  List.iter (w :: outs) ~f:materialize;
  let seed = (w, weights) :: List.map outs ~f:(fun o -> (o, blank n)) in
  let o = optimize ~name llc in
  List.iter virtuals ~f:(fun v ->
      p (name ^ ": " ^ Ir.Tnode.debug_name v ^ " virtualizes") (known_virtual o v));
  let got = execute ~name o ~seed ~read:outs in
  let mat_name = name ^ "_mat" in
  let mat = optimize ~materialized:virtuals ~name:mat_name llc in
  List.iter virtuals ~f:(fun v ->
      p (mat_name ^ ": " ^ Ir.Tnode.debug_name v ^ " stays a buffer") (known_non_virtual mat v));
  let want = execute ~name:mat_name mat ~seed ~read:outs in
  (o, got, want)

(* [x * x * (w + 1)], what [out[i] = d[i] * x[i]] holds over {!reader}. *)
let squared_rows ~x_at =
  Array.init n ~f:(fun i ->
      let x = x_at i in
      x *. x *. (weights.(i) +. 1.))

(* The issue's row: a virtual literal, read in a sibling's body and beside it. *)
let row name ~make_x =
  let w = mk (name ^ "_w") and x = mk (name ^ "_x") in
  let d = mk (name ^ "_d") and r = mk (name ^ "_r") and out = mk (name ^ "_out") in
  let t = sym () in
  let llc =
    LL.unflat_lines
      [
        make_x x w;
        reader d ~x ~w;
        relay r d;
        loop_n t n (set_at out (iter t) (mul (get r [| iter t |]) (get x [| iter t |])));
      ]
  in
  let o, got, want = both_arms ~name ~virtuals:[ x; d; r ] ~w ~outs:[ out ] llc in
  p
    (name ^ ": a body reads a sibling's local of a higher id")
    (List.exists (outward_reads o.LL.llc) ~f:(fun (body, read) -> body < read));
  (o, got, want)

let row_literal () =
  let name = "lsr_literal" in
  let _, got, want = row name ~make_x:(fun x _ -> literal x) in
  let closed = squared_rows ~x_at:(fun i -> table.(i)) in
  p (name ^ ": the virtual run gives the closed form") (same got [ closed ]);
  p (name ^ ": it matches the materialized run") (same got want)

let row_computed () =
  let name = "lsr_computed" in
  let _, got, want = row name ~make_x:computed in
  let closed = squared_rows ~x_at:(fun i -> weights.(i) +. 1.) in
  p (name ^ ": the virtual run gives the closed form") (same got [ closed ]);
  p (name ^ ": it matches the materialized run") (same got want)

(* One level further: [d]'s body reads [e], whose body reads [x], which sits beside [d]. The read is
   two bodies down from the definition it depends on. *)
let row_nested () =
  let name = "lsr_nested" in
  let w = mk (name ^ "_w") and x = mk (name ^ "_x") and e = mk (name ^ "_e") in
  let d = mk (name ^ "_d") and r = mk (name ^ "_r") and out = mk (name ^ "_out") in
  let t = sym () in
  let llc =
    LL.unflat_lines
      [
        literal x;
        reader e ~x ~w;
        reader d ~x:e ~w;
        relay r d;
        loop_n t n (set_at out (iter t) (mul (get r [| iter t |]) (get x [| iter t |])));
      ]
  in
  let o, got, want = both_arms ~name ~virtuals:[ x; e; d; r ] ~w ~outs:[ out ] llc in
  p
    (name ^ ": a body reads a sibling's local of a higher id")
    (List.exists (outward_reads o.LL.llc) ~f:(fun (body, read) -> body < read));
  let closed =
    Array.init n ~f:(fun i ->
        let x = table.(i) and w = weights.(i) in
        x *. x *. (w +. 1.) *. (w +. 1.))
  in
  p (name ^ ": the virtual run gives the closed form") (same got [ closed ]);
  p (name ^ ": it matches the materialized run") (same got want)

(* Two sibling statements, each with the shape of [row]. The literal's scope is shared by the two
   and hoisted, so one statement's [Get_local] inside its reader's body names a scope that is gone
   and has to follow the hoisted copy. *)
let hoisted_program name =
  let w = mk (name ^ "_w") and x = mk (name ^ "_x") in
  let d1 = mk (name ^ "_d1") and d2 = mk (name ^ "_d2") in
  let o1 = mk (name ^ "_o1") and o2 = mk (name ^ "_o2") in
  let t = sym () in
  let llc =
    LL.unflat_lines
      [
        literal x;
        reader d1 ~x ~w;
        reader d2 ~x:w ~w:x;
        loop_n t n
          (seq
             (set_at o1 (iter t) (mul (get d1 [| iter t |]) (get x [| iter t |])))
             (set_at o2 (iter t) (mul (get d2 [| iter t |]) (get x [| iter t |]))));
      ]
  in
  (llc, w, [ x; d1; d2 ], [ o1; o2 ])

let hoisted_closed_form =
  [
    squared_rows ~x_at:(fun i -> table.(i));
    Array.init n ~f:(fun i ->
        let x = table.(i) and w = weights.(i) in
        ((w *. x) +. w) *. x);
  ]

let row_hoisted () =
  let name = "lsr_hoisted" in
  let llc, w, virtuals, outs = hoisted_program name in
  let o, got, want = both_arms ~name ~virtuals ~w ~outs llc in
  p
    (name ^ ": the shared scope is hoisted to a declared local")
    (count_stmt o.LL.llc ~in_scopes:false ~f:(function LL.Declare_local _ -> true | _ -> false)
    >= 1);
  every_read_bound name o.LL.llc;
  p (name ^ ": the virtual run gives the closed form") (same got hoisted_closed_form);
  p (name ^ ": it matches the materialized run") (same got want)

(* A materializing [Unroll] copies the loop body and renames the binders of each copy, so a read in
   a body has to follow its binder, statement-level or sibling. *)
let row_unrolled () =
  let name = "lsr_unrolled" in
  let llc, w, _, outs = hoisted_program name in
  List.iter (w :: outs) ~f:materialize;
  let o = optimize ~name llc in
  let axis = List.hd_exn (List.hd_exn (nest_paths o.LL.llc)) in
  let unrolled = Sched.apply [ Sched.Unroll { axis; materialize = true } ] o in
  p (name ^ ": the loop is gone") (not (binds_loop axis unrolled.LL.llc));
  every_read_bound name unrolled.LL.llc;
  let linked =
    Context.compile ~name ~prelowered:o
      ~lowered_transform:(fun o -> [ Sched.apply [ Sched.Unroll { axis; materialize = true } ] o ])
      (Lazy.force base_ctx) Ir.Assignments.empty_comp Ir.Indexing.Empty
  in
  let ctx = run_linked linked ~seed:((w, weights) :: List.map outs ~f:(fun o -> (o, blank n))) in
  let got = List.map outs ~f:(Context.get_values ctx) in
  p (name ^ ": the unrolled run gives the closed form") (same got hoisted_closed_form)

(* A projection's discarded operand is never rendered, so a read there orders nothing. [a] mentions
   [b] only in one, while [b] really reads [a]: counted, the mention makes a cycle and walks [b]
   ahead of the definition it needs. Hand-built and prelowered, since [optimize] folds a projection
   away. *)
let row_discarded () =
  let name = "lsr_discarded" in
  let a = mk ~dims:[| 1 |] (name ^ "_a") and b = mk ~dims:[| 1 |] (name ^ "_b") in
  let out = mk ~dims:[| 1 |] (name ^ "_out") in
  List.iter [ a; b ] ~f:virtualize;
  materialize out;
  let cell = [| fixed 0 |] in
  let raw =
    LL.unflat_lines
      [
        set a cell (c 5.);
        set b cell (add (get a cell) (c 1.));
        set out cell (add (get a cell) (get b cell));
      ]
  in
  let ia = LL.get_scope a and ib = LL.get_scope b in
  let scoped =
    set out cell
      (add
         (scope ~orig_indices:cell ~id:ia a
            (LL.Binop (Ops.Arg1, (c 5., single), (LL.Get_local ib, single))))
         (scope ~orig_indices:cell ~id:ib b (add (LL.Get_local ia) (c 1.))))
  in
  let o = optimize_scoped ~name ~raw scoped in
  let got = execute ~name o ~seed:[ (out, blank 1) ] ~read:[ out ] in
  p (name ^ ": the discarded read leaves the order alone") (same got [ [| 11. |] ])

let () =
  case "literal" row_literal;
  case "computed" row_computed;
  case "nested" row_nested;
  case "hoisted" row_hoisted;
  case "unrolled" row_unrolled;
  case "discarded" row_discarded;
  Stdio.printf "%!"
