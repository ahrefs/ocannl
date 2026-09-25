(* Partition / index-set splitting (gh-ocannl-508): [Sched.Partition] splits a Serial loop's
   iteration range at static affine breakpoints into separate, individually specialized segment
   nests. The narrowed per-segment ranges let [Sched.apply]'s trailing simplify interval-fold the
   in-loop guards each segment decides, replacing the two guard workarounds:

   - [Split]'s construct-then-fold remainder guard: partitioning at the last tile-multiple first and
   then splitting the dividing main segment yields a guard-free main nest plus a serial epilogue (no
   [If] anywhere).

   - the virtualizer's per-component [Where] range guards of an inlined concatenation: partitioning
   the consumer loop at the component boundaries (derived by [Sched.partition_breakpoints] from the
   guards themselves) folds every [Where], converging the inlined-concat rendering with the segment
   nests that materialized concatenation already lowers to.

   The fresh segment symbols returned by [Sched.partition] make each segment individually
   addressable by subsequent ops (per-segment scheduling), demonstrated by unrolling just the tail
   segment. Section 6 pins that this addressability survives an accumulation mint: a loop moved into
   a [Local_scope] by a materializing [Unroll] is still located by [partition_breakpoints] and still
   rewritten by the [Partition] its breakpoints feed (gh-ocannl-668). Structural checks pair with
   executed-output parity against untransformed references on every backend. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Idx = Ir.Indexing
module Asgns = Ir.Assignments
open Verdict.Claims

(* Zeros compare equal to zeros. A fragment mapping that reads outside the staged block, a kernel
   that never ran, or a reference whose own setup silently collapsed all yield all-zeros, and a
   parity check between two zero arrays passes while covering nothing (gh-ocannl-481 item 3). Every
   reference array is pinned nonzero where it is produced, so the parity claims below have
   content. *)
let nonzero name (a : float array) =
  if not (Array.exists a ~f:(fun x -> Float.(x <> 0.))) then
    failwith (name ^ ": the reference is all zeros — the parity checks against it are vacuous");
  a

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* The IR queries come from [Ll_test] (gh-ocannl-608), which decides the [Local_scope] descent once
   for all of them: this file had grown six bespoke traversals of the same 14 statement and 9 scalar
   constructors, each answering a one-line question and each getting that descent right (or not) on
   its own — while the thing under test in section 6 is precisely where a loop lives. Where a query
   below pins the statement-position-only reading, it says so with [~in_scopes:false]; that is the
   walk [Schedule.find_loop] had before gh-ocannl-668, not a default. *)

(* The first [For_loop] (in preorder) with the given iteration count, through statement positions.
   Prefer [find_nest] wherever an initialization loop can share the extent of the reduction nest:
   this answers the weaker question, and answered it with the wrong loop until gh-ocannl-608. *)
let find_loop_with_extent ~n (llc : LL.t) : Idx.symbol option =
  Ll_test.find_loop_with_extent ~in_scopes:false ~n llc

(* Is a [For_loop] binding [sym] reachable through statement positions alone (the walk
   [Schedule.find_loop] had before gh-ocannl-668), and is it reachable at all (descending into
   [Local_scope] bodies, the way [rewrite_loop] reaches loops the accumulation mints wrapped in a
   scope)? Section 6 uses both to pin that its construction really is the scope-nested one. *)
let binds_loop ~in_scopes sym (llc : LL.t) : bool = Ll_test.binds_loop ~in_scopes sym llc

(* The first [For_loop] of extent [outer_n] enclosing one of extent [inner_n], as a symbol pair —
   the reduction nest of a pooling forward, past the initialization loop over the same extent. *)
let find_nest ~outer_n ~inner_n (llc : LL.t) : (Idx.symbol * Idx.symbol) option =
  Ll_test.find_nest ~in_scopes:false ~outer_n ~inner_n llc

(* How many [For_loop]s bind [sym] — the copies a materializing [Unroll] leaves behind, all of which
   [Sched.apply] rewrites. *)
let count_loops sym (llc : LL.t) : int = Ll_test.count_loops sym llc

(* The first [For_loop] binding [sym] in preorder, as a standalone routine — the slice of the code a
   first-match probe used to speak for. *)
let first_binding sym (llc : LL.t) : LL.t =
  Option.value_exn ~here:[%here] (Ll_test.first_binding sym llc)

(* Structural census: statement [If] guards, scalar [Where] guards, and [For_loop]s. *)
let census (llc : LL.t) : int * int * int = Ll_test.census llc

let run_with name transform (t : Tensor.t) =
  let ctx = Context.auto () in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun o -> [ transform o ])
      ctx
      (named name (Train.forward t))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  Context.get_values ctx t.Tensor.value

(* Non-empty by construction (gh-ocannl-746): [Array.for_all2_exn] answers [true] on two empty
   arrays, and a readback and the reference it is compared against go empty TOGETHER -- the
   reference went through the same path. {!Verdict.p_all2} is the claim-shaped form; the sites
   reached through this helper keep a boolean, so the guard lives here. *)
let close a b =
  (not (Array.is_empty a)) && Array.for_all2_exn a b ~f:(fun x y -> Float.(abs (x - y) < 1e-5))

let () =
  Tensor.unsafe_reinitialize ();

  (* === 1: Split-remainder replacement — partition at the last tile-multiple, then split the
     dividing main segment: guard-free main nest + serial epilogue. === *)
  let xv = Array.init 10 ~f:(fun i -> Float.of_int (i - 4)) in
  let make_graph1 () =
    let x = TDSL.ndarray xv ~label:[ "sp_x" ] ~output_dims:[ 10 ] () in
    let%op y = relu x in
    y
  in
  let want1 = nonzero "sp_ref1" (run_with "sp_ref1" (fun opt -> opt) (make_graph1 ())) in

  (* Reference discipline: a plain non-dividing Split leaves its remainder guard in-loop. *)
  let split_ifs = ref (-1) in
  let transform_split (opt : LL.optimized) =
    let axis = Option.value_exn ~here:[%here] (find_loop_with_extent ~n:10 opt.LL.llc) in
    let op, _, _ = Sched.split ~axis ~factor:4 ~outer:LL.Serial ~inner:LL.Serial in
    let opt = Sched.apply [ op ] opt in
    let ifs, _, _ = census opt.LL.llc in
    split_ifs := ifs;
    opt
  in
  let got_split = run_with "sp_split_guarded" transform_split (make_graph1 ()) in
  p "plain non-dividing split keeps a remainder guard" (!split_ifs > 0);
  p "guarded split matches reference" (close got_split want1);

  let part_census = ref (-1, -1, -1) in
  let transform_part (opt : LL.optimized) =
    let axis = Option.value_exn ~here:[%here] (find_loop_with_extent ~n:10 opt.LL.llc) in
    let op1, segs = Sched.partition ~axis ~breakpoints:[ 8 ] in
    let main = List.hd_exn segs in
    let op2, _, _ = Sched.split ~axis:main ~factor:4 ~outer:LL.Serial ~inner:LL.Serial in
    let opt = Sched.apply [ op1; op2 ] opt in
    part_census := census opt.LL.llc;
    opt
  in
  let got_part = run_with "sp_partition_split" transform_part (make_graph1 ()) in
  let ifs, _, loops = !part_census in
  p "partition+split is guard-free" (ifs = 0);
  p "partition+split has main outer+inner and tail loops" (loops = 3);
  p "partition+split matches reference" (close got_part want1);

  (* === 2: Inlined-concat consumer — partition at the component boundaries derived from the
     virtualizer's [Where] range guards; every guard folds. === *)
  let make_graph2 () =
    let a = TDSL.ndarray [| 1.0; 2.0; 3.0 |] ~label:[ "sp_ca" ] ~output_dims:[ 3 ] () in
    let b = TDSL.ndarray [| 4.0; 5.0 |] ~label:[ "sp_cb" ] ~output_dims:[ 2 ] () in
    let%op r = sin ((a, b) ++^ "a; b => a^b") in
    r
  in
  let ref_wheres = ref (-1) in
  let observe (opt : LL.optimized) =
    let _, wheres, _ = census opt.LL.llc in
    ref_wheres := wheres;
    opt
  in
  let want2 = nonzero "sp_ref2" (run_with "sp_ref2" observe (make_graph2 ())) in
  p "inlined concat consumer carries Where range guards" (!ref_wheres > 0);
  p "concat values are sin of the concatenation"
    (close want2 (Array.map [| 1.0; 2.0; 3.0; 4.0; 5.0 |] ~f:Float.sin));

  let bps = ref [] in
  let part2_census = ref (-1, -1, -1) in
  let transform_concat (opt : LL.optimized) =
    let axis = Option.value_exn ~here:[%here] (find_loop_with_extent ~n:5 opt.LL.llc) in
    bps := Sched.partition_breakpoints ~axis opt.LL.llc;
    let op, _segs = Sched.partition ~axis ~breakpoints:!bps in
    let opt = Sched.apply [ op ] opt in
    part2_census := census opt.LL.llc;
    opt
  in
  let got2 = run_with "sp_partition_concat" transform_concat (make_graph2 ()) in
  p "breakpoints derived from the guards are the component boundary"
    (List.equal Int.equal !bps [ 3 ]);
  let ifs2, wheres2, loops2 = !part2_census in
  p "partitioned concat consumer is guard-free" (ifs2 = 0 && wheres2 = 0);
  p "partitioned concat consumer has one nest per component" (loops2 = 2);
  p "partitioned concat matches reference" (close got2 want2);

  (* === 3: Per-segment scheduling — the fresh segment symbols are individually addressable: unroll
     just the tail segment. === *)
  let part3_census = ref (-1, -1, -1) in
  let transform_tail_unrolled (opt : LL.optimized) =
    let axis = Option.value_exn ~here:[%here] (find_loop_with_extent ~n:5 opt.LL.llc) in
    let op, segs = Sched.partition ~axis ~breakpoints:[ 3 ] in
    let tail = List.last_exn segs in
    let opt = Sched.apply [ op; Sched.Unroll { axis = tail; materialize = true } ] opt in
    part3_census := census opt.LL.llc;
    opt
  in
  let got3 = run_with "sp_partition_tail_unroll" transform_tail_unrolled (make_graph2 ()) in
  let ifs3, wheres3, loops3 = !part3_census in
  p "tail-unrolled partition is guard-free" (ifs3 = 0 && wheres3 = 0);
  p "only the main segment remains a loop" (loops3 = 1);
  p "tail-unrolled partition matches reference" (close got3 want2);

  (* === 4: Pattern discipline — targeted errors. === *)
  let expect_error name transform t =
    match
      try
        ignore
          (Context.compile
             ~lowered_transform:(fun o -> [ transform o ])
             (Context.auto ())
             (named name (Train.forward t))
             Ir.Indexing.Empty
            : Context.t * Context.routine);
        None
      with Invalid_argument msg -> Some msg
    with
    | Some msg ->
        p
          (name ^ " rejected with a targeted error")
          (String.is_substring msg ~substring:"Schedule.")
    | None -> p (name ^ " rejected with a targeted error") false
  in
  expect_error "sp_err_out_of_range"
    (fun (opt : LL.optimized) ->
      let axis = Option.value_exn ~here:[%here] (find_loop_with_extent ~n:10 opt.LL.llc) in
      let op, _ = Sched.partition ~axis ~breakpoints:[ 12 ] in
      Sched.apply [ op ] opt)
    (make_graph1 ());
  expect_error "sp_err_not_increasing"
    (fun (opt : LL.optimized) ->
      let axis = Option.value_exn ~here:[%here] (find_loop_with_extent ~n:10 opt.LL.llc) in
      let op, _ = Sched.partition ~axis ~breakpoints:[ 3; 3 ] in
      Sched.apply [ op ] opt)
    (make_graph1 ());
  expect_error "sp_err_no_such_loop"
    (fun (opt : LL.optimized) ->
      let op, _ = Sched.partition ~axis:(Idx.get_symbol ()) ~breakpoints:[ 3 ] in
      Sched.apply [ op ] opt)
    (make_graph1 ());

  (* === 5: Guard-shape canonicity — index guards use one canonical shape per role (upper bounds
     strict [Cmplt], lower bounds direct [Cmple]). [partition_breakpoints] must derive the same
     transition points from a [Cmple] guard as from the strict encoding it replaced; without a
     [Cmple] arm the guard silently contributes nothing. === *)
  let iprec = Ir.Ops.index_prec () in
  let bps_of ~to_ mk =
    let axis = Idx.get_symbol () in
    let llc =
      LL.For_loop
        {
          index = axis;
          from_ = 0;
          to_;
          body = LL.If { cond = (mk axis, iprec); body = LL.Noop };
          axis = LL.Serial;
        }
    in
    Sched.partition_breakpoints ~axis llc
  in
  let ivar i = (LL.Embed_index (Idx.Iterator i), iprec) in
  let fixed n = (LL.Embed_index (Idx.Fixed_idx n), iprec) in
  (* [3 <= i] and [2 < i] both flip at 3; [i <= 6] and [i < 7] both flip at 7. *)
  let le_lower = bps_of ~to_:9 (fun i -> LL.Binop (Ir.Ops.Cmple, fixed 3, ivar i)) in
  let lt_lower = bps_of ~to_:9 (fun i -> LL.Binop (Ir.Ops.Cmplt, fixed 2, ivar i)) in
  let le_upper = bps_of ~to_:9 (fun i -> LL.Binop (Ir.Ops.Cmple, ivar i, fixed 6)) in
  let lt_upper = bps_of ~to_:9 (fun i -> LL.Binop (Ir.Ops.Cmplt, ivar i, fixed 7)) in
  let is bps want = List.equal Int.equal bps want in
  p "Cmple lower bound breaks where its Cmplt encoding did" (is le_lower [ 3 ] && is lt_lower [ 3 ]);
  p "Cmple upper bound breaks where its Cmplt encoding did" (is le_upper [ 7 ] && is lt_upper [ 7 ]);
  (* A non-unit coefficient exercises the rounding: [2i <= 5] flips at [i = 3], as does [2i < 6]. *)
  let coef2 n = (LL.Embed_index (Idx.affine ~symbols:[ (2, n) ] ~offset:0), iprec) in
  let le_scaled = bps_of ~to_:9 (fun i -> LL.Binop (Ir.Ops.Cmple, coef2 i, fixed 5)) in
  let lt_scaled = bps_of ~to_:9 (fun i -> LL.Binop (Ir.Ops.Cmplt, coef2 i, fixed 6)) in
  p "Cmple with a scaled axis rounds like its Cmplt encoding"
    (is le_scaled [ 3 ] && is lt_scaled [ 3 ]);

  (* === 6: Loops inside a [Local_scope] (gh-ocannl-668) — the accumulation mint of a materializing
     [Unroll] wraps the inner reduction loop in the accumulator's scope, and [Sched.apply] keeps
     rewriting loops there. Every probe that LOCATES a loop must reach the same place, or it reports
     absent a loop the very next op rewrites: here [partition_breakpoints] derives the pad guard's
     flip point from the scope-nested [s] loop, and the [Partition] it feeds applies. === *)
  let ni, nr, ns = (4, 6, 5) in
  let xv6 =
    Array.init
      (ni * nr * ns)
      ~f:(Ll_test.cycle_flat ~dims:[| ni; nr; ns |] ~modulus:7 ~offset:1. ~stride:0.25)
  in
  let make_graph6 () =
    let x = TDSL.ndarray xv6 ~label:[ "sp_rx" ] ~output_dims:[ ni; nr; ns ] () in
    let%op out = x ++ "irs => i" in
    out
  in
  let want6 = nonzero "sp_ref6" (run_with "sp_ref6" (fun opt -> opt) (make_graph6 ())) in
  (* [Pad] gives the inner loop a guard with a known flip point ([s < 5] over the padded range [0,
     8)); the materializing unroll of the OUTER reduction axis then moves the whole guarded inner
     loop into the accumulator scope. *)
  let stmt_level = ref true and in_scope = ref false in
  let bps6 = ref [] in
  let census6 = ref (-1, -1, -1) in
  let transform_scope_nested (opt : LL.optimized) =
    let r = Option.value_exn ~here:[%here] (find_loop_with_extent ~n:nr opt.LL.llc) in
    let s = Option.value_exn ~here:[%here] (find_loop_with_extent ~n:ns opt.LL.llc) in
    let opt =
      Sched.apply
        [
          Sched.Pad { axis = s; to_multiple_of = 4 }; Sched.Unroll { axis = r; materialize = true };
        ]
        opt
    in
    stmt_level := binds_loop ~in_scopes:false s opt.LL.llc;
    in_scope := binds_loop ~in_scopes:true s opt.LL.llc;
    bps6 := Sched.partition_breakpoints ~axis:s opt.LL.llc;
    let op, _segs = Sched.partition ~axis:s ~breakpoints:!bps6 in
    let opt = Sched.apply [ op ] opt in
    census6 := census opt.LL.llc;
    opt
  in
  let got6 = run_with "sp_scope_nested_partition" transform_scope_nested (make_graph6 ()) in
  p "the materializing unroll left the inner loop only inside a Local_scope"
    ((not !stmt_level) && !in_scope);
  p "breakpoints of a scope-nested loop are the pad guard's flip point"
    (List.equal Int.equal !bps6 [ ns ]);
  let ifs6, wheres6, _ = !census6 in
  p "partitioning the scope-nested loop folds the pad guard" (ifs6 = 0 && wheres6 = 0);
  p "scope-nested partition matches reference" (close got6 want6);

  (* === 7: MANY bindings of one symbol (gh-ocannl-668, review round 1) — a materializing [Unroll]
     leaves one copy of the inner loop per unrolled step, each carrying the same guard with a
     different constant substituted for the unrolled index, and [Partition] rewrites every copy. So
     the breakpoints are the UNION over the copies: stopping at the first one folds that copy's
     guard and leaves its siblings mixed (or reports no breakpoint at all when the first copy's
     guard happens to be already decided).

     The clamped max-pool of gh-ocannl-504 is the natural shape: N=8, stride 2, window 5, so the
     window guard [0 <= 2o + w - 2 < 8] mentions both loops. Unrolling the OUTPUT loop [o] (whose
     index the accumulator carries, so no scope is minted — the copies are siblings) leaves four
     copies of the [w] loop guarded [w >= 2], [w >= 0], [w >= -2], [w < 4]: the first copy alone
     yields [2], the copies together [2; 4]. === *)
  Tensor.unsafe_reinitialize ();
  let make_pool () =
    let x =
      TDSL.ndarray
        (Array.init 8 ~f:(fun i -> Float.of_int i -. 16.))
        ~label:[ "sp_px" ] ~output_dims:[ 8 ] ()
    in
    let%op y = x @^+ "2*o=+w; w => o" [ "w" ] (stretch 0.0) in
    Shape.set_dim w 5;
    y
  in
  let want7 = nonzero "sp_ref7" (run_with "sp_ref7" (fun opt -> opt) (make_pool ())) in
  let first_only = ref [] and all_copies = ref [] in
  let copies = ref 0 in
  let census7 = ref (-1, -1, -1) in
  let transform_copies (opt : LL.optimized) =
    let o, w = Option.value_exn ~here:[%here] (find_nest ~outer_n:4 ~inner_n:5 opt.LL.llc) in
    let opt = Sched.apply [ Sched.Unroll { axis = o; materialize = true } ] opt in
    copies := count_loops w opt.LL.llc;
    (* What the first copy alone would have contributed, isolated the way the pre-fix walk saw it:
       the breakpoints of the subtree cut off after the first binding. *)
    first_only := Sched.partition_breakpoints ~axis:w (first_binding w opt.LL.llc);
    all_copies := Sched.partition_breakpoints ~axis:w opt.LL.llc;
    let op, _segs = Sched.partition ~axis:w ~breakpoints:!all_copies in
    let opt = Sched.apply [ op ] opt in
    census7 := census opt.LL.llc;
    opt
  in
  let got7 = run_with "sp_copies_partition" transform_copies (make_pool ()) in
  p "the materializing unroll left one copy of the inner loop per step" (!copies = 4);
  p "the copies' guards flip at different points, and the union is what the copies need"
    (List.equal Int.equal !first_only [ 2 ] && List.equal Int.equal !all_copies [ 2; 4 ]);
  (* Every copy's guard is decided within every segment: the clamp [Where]s are gone. *)
  let ifs7, wheres7, _ = !census7 in
  p "partitioning at the union folds every copy's guard" (ifs7 = 0 && wheres7 = 0);
  p "partition over many bindings matches reference" (close got7 want7);

  (* The two dimensions meet when the copies live inside an accumulator scope — the shape a
     materializing [Unroll] of an outer reduction axis mints. Hand-built so the copies carry visibly
     different guards ([i < 3] and [i < 6]) whatever a lowering happens to produce. *)
  let scoped_copies =
    let axis = Idx.get_symbol () in
    let tn =
      Ir.Tnode.create (Ir.Tnode.Specified Ir.Ops.single) ~id:9701 ~label:[ "sp_agg" ]
        ~unpadded_dims:(lazy [| 1 |])
        ~padding:(lazy None)
        ()
    in
    let guarded k =
      LL.For_loop
        {
          index = axis;
          from_ = 0;
          to_ = 9;
          body =
            LL.If { cond = (LL.Binop (Ir.Ops.Cmplt, ivar axis, fixed k), iprec); body = LL.Noop };
          axis = LL.Serial;
        }
    in
    let idcs = [| Idx.Fixed_idx 0 |] in
    let llc =
      LL.Set
        {
          tn;
          idcs;
          llsc =
            LL.Local_scope
              {
                id = LL.get_scope tn;
                body = LL.Seq (guarded 3, guarded 6);
                orig_indices = idcs;
                mint = LL.Schedule_minted;
              };
          debug = "";
        }
    in
    Sched.partition_breakpoints ~axis llc
  in
  p "copies inside a Local_scope contribute their breakpoints too"
    (List.equal Int.equal scoped_copies [ 3; 6 ]);

  Stdio.printf "\nDone.\n%!"
