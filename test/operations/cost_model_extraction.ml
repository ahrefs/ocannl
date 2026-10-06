(* gh-ocannl-491 task 1: [Ir.Cost_model] — footprint/FLOPs extraction and the roofline bound over
   hand-built programs where every number is checkable by hand (single precision, 4 bytes/cell):

   - elementwise map: bytes = the three operand footprints, one op per cell; - matmul: FLOPs =
   2*M*N*K, bytes = (MK + KN + 2*MN) * 4 (the rmw accumulator charged once per direction); -
   strided/gapped access: the image cardinality counts the touched cells, not the node size; - rmw
   reduction: the accumulator's read/write/rmw split; - guarded write: guards-taken op count
   ([flops_approx]) and a never-definite write; - dynamic gathers: the known-image-times-rows bound,
   capped by the loop box (gh-ocannl-1174), each count checked against the worst case over all
   runtime rows; - overlapping writes: the union bound capped by the node's size; - multi-read
   exactness (gh-ocannl-578): pairwise provably-disjoint exact reads sum exactly, overlapping ones
   stay a flagged union bound, and conditionally-evaluated reads (Where arms) stay a flagged bound
   even when disjoint — with the op count flagged too when the arms' costs differ, unless an arm's
   cost lives entirely in a hoisted scope body, which executes unconditionally (gh-ocannl-637); -
   vectorized runs (gh-ocannl-578): bases spaced by at least the run length (or on distinct
   in-bounds rows) count exactly, close-spaced or row-spilling bases stay a flagged upper bound, and
   two stores on distinct rows are provably disjoint, so they sum exactly, while two binding one
   loop symbol with different bounds stay a bound in either order; - a [Concat] coordinate:
   the box-over-fiber bound, flagged, never the whole node.

   The tail asserts the roofline bound is monotone in the envelope constants. *)

open Base
module LL = Ir.Low_level
module Idx = Ir.Indexing
module Tn = Ir.Tnode
module Ops = Ir.Ops
module CM = Ir.Cost_model
open Verdict.Claims

let fresh_tn =
  let make = Ll_test.node_factory ~first_id:970_000_000 ~dims:[||] () in
  fun label dims -> make ~dims label

let sp = Ops.single
let get = Ll_test.get
let it = Ll_test.iter

let show_summary name (s : CM.summary) =
  Stdio.printf "== %s ==\n" name;
  List.iter s.CM.per_node ~f:(fun (tn, fp) ->
      Stdio.printf "  %-3s rd=%-4d wr=%-4d rmw=%-4d %s\n" (Tn.debug_name tn) fp.CM.fp_read_bytes
        fp.CM.fp_write_bytes fp.CM.fp_rmw_bytes
        (if fp.CM.fp_approx then "approx" else "exact"));
  Stdio.printf "  flops=%d%s read=%d write=%d intensity=%.3f%s\n" s.CM.flops
    (if s.CM.flops_approx then "~" else "")
    s.CM.read_bytes s.CM.write_bytes (CM.arithmetic_intensity s)
    (if s.CM.opaque then " OPAQUE" else "")

(* The most distinct cells a gather can read over all runtime data. The cost model may assume
   nothing of a data-dependent row, so each loop-box point picks any of the [rows] independently
   ([instances] maps a row to the point's flat cell), and the worst case over those choices is
   searched exhaustively. *)
let worst_case_cells ~instances ~rows =
  let best = ref 0 in
  let rec go seen remaining = function
    | [] -> best := max !best (Set.length seen)
    | cell :: rest ->
        if Set.length seen + remaining > !best then
          (* Rows reaching an unseen cell first: the greedy path comes first, and the bound prunes
             most of the rest once it has been met. *)
          let fresh, stale =
            List.partition_tf (List.init rows ~f:Fn.id) ~f:(fun row ->
                not (Set.mem seen (cell row)))
          in
          List.iter (fresh @ stale) ~f:(fun row ->
              go (Set.add seen (cell row)) (remaining - 1) rest)
  in
  go (Set.empty (module Int)) (List.length instances) instances;
  !best

(* The table's read count must bound the worst case: an upper bound that holds for every row the
   data could name. *)
let gather_bound_sound ~name (s : CM.summary) table ~instances ~rows =
  let fp = List.Assoc.find_exn s.CM.per_node table ~equal:Tn.equal in
  let counted = fp.CM.fp_read_bytes / 4 and worst = worst_case_cells ~instances ~rows in
  Stdio.printf "  table: %d cells counted, worst case over the data %d\n" counted worst;
  claimf "%s: the table count bounds every runtime choice of rows" name
    (fp.CM.fp_approx && counted >= worst)

let () =
  let i = Idx.get_symbol () and j = Idx.get_symbol () and k = Idx.get_symbol () in
  (* Elementwise map, 4x5: for i: for j: C[i][j] = A[i][j] + B[j]. A rd 20 cells = 80 B, B rd 5
     cells = 20 B, C wr 20 cells = 80 B; 1 add x 20 iters. *)
  let a = fresh_tn "A" [| 4; 5 |] in
  let b = fresh_tn "B" [| 5 |] in
  let c = fresh_tn "C" [| 4; 5 |] in
  let pointwise =
    Ll_test.loop_n i 4
      (Ll_test.loop_n j 5
         (Ll_test.set c
            [| it i; it j |]
            (LL.Binop (Ops.Add, (get a [| it i; it j |], sp), (get b [| it j |], sp)))))
  in
  show_summary "elementwise map 4x5" (CM.analyze pointwise);

  (* Matmul M=4, N=3, K=5: for i: for j: for k: D[i][j] = D[i][j] + A[i][k] * B2[k][j]. FLOPs =
     2*M*N*K = 120; A rd MK=20 cells, B2 rd KN=15, D rd MN=12 + wr 12 (rmw). *)
  let a_mk = fresh_tn "Am" [| 4; 5 |] in
  let b_kn = fresh_tn "Bm" [| 5; 3 |] in
  let d_mn = fresh_tn "Dm" [| 4; 3 |] in
  let matmul =
    Ll_test.loop_n i 4
      (Ll_test.loop_n j 3
         (Ll_test.loop_n k 5
            (Ll_test.set d_mn
               [| it i; it j |]
               (LL.Binop
                  ( Ops.Add,
                    (get d_mn [| it i; it j |], sp),
                    ( LL.Binop
                        (Ops.Mul, (get a_mk [| it i; it k |], sp), (get b_kn [| it k; it j |], sp)),
                      sp ) )))))
  in
  let mm = CM.analyze matmul in
  show_summary "matmul 4x3x5" mm;

  (* The same matmul in FMA form: D[i][j] = FMA(A[i][k], B2[k][j], D[i][j]) — an FMA counts as two
     operations, so the score is identical to the mul+add form. *)
  let matmul_fma =
    Ll_test.loop_n i 4
      (Ll_test.loop_n j 3
         (Ll_test.loop_n k 5
            (Ll_test.set d_mn
               [| it i; it j |]
               (LL.Ternop
                  ( Ops.FMA,
                    (get a_mk [| it i; it k |], sp),
                    (get b_kn [| it k; it j |], sp),
                    (get d_mn [| it i; it j |], sp) )))))
  in
  show_summary "matmul 4x3x5, FMA form" (CM.analyze matmul_fma);

  (* Strided/gapped, size-8 nodes: for i: R[2*i] = A8[2*i] * 2 — touches 4 of 8 cells each. *)
  let a8 = fresh_tn "A8" [| 8 |] in
  let r8 = fresh_tn "R8" [| 8 |] in
  let stride2 s = Idx.affine ~symbols:[ (2, s) ] ~offset:0 in
  let strided =
    Ll_test.loop_n i 4
      (Ll_test.set r8
         [| stride2 i |]
         (LL.Binop (Ops.Mul, (get a8 [| stride2 i |], sp), (LL.Constant 2., sp))))
  in
  show_summary "strided copy-scale over half the cells" (CM.analyze strided);

  (* Rmw reduction: for i: for k: S[i] = S[i] + A[i][k] — S rd 16 B, wr 16 B all rmw. *)
  let s = fresh_tn "S" [| 4 |] in
  let reduction =
    Ll_test.loop_n i 4
      (Ll_test.loop_n k 5
         (Ll_test.set s
            [| it i |]
            (LL.Binop (Ops.Add, (get s [| it i |], sp), (get a [| it i; it k |], sp)))))
  in
  show_summary "rmw reduction 4x5" (CM.analyze reduction);

  (* Guarded write: if G[0] then D1[0] = G[0] * 3 — guards-taken op count, approximate write. *)
  let g = fresh_tn "G" [| 1 |] in
  let d1 = fresh_tn "D1" [| 1 |] in
  let guarded =
    Ll_test.if_ (get g [| Idx.Fixed_idx 0 |])
      (Ll_test.set d1 [| Idx.Fixed_idx 0 |]
         (LL.Binop (Ops.Mul, (get g [| Idx.Fixed_idx 0 |], sp), (LL.Constant 3., sp))))
  in
  show_summary "guarded write" (CM.analyze guarded);

  (* Dynamic gathers (gh-ocannl-1174): one cell per loop-box point, at a row the data picks — at
     most the known coordinates' image times the dynamic axis's extent, and at most the box. For i:
     E[i] = A[I[i]][0] reads column 0 of A, so at most 4 rows x 1 column (16 B, not A's 80). *)
  let e = fresh_tn "E" [| 4 |] in
  let ids = fresh_tn "I" [| 4 |] in
  let gather =
    Ll_test.loop_n i 4
      (Ll_test.set e
         [| it i |]
         (Ll_test.gather ~tn:a
            ~idcs:[| Idx.Fixed_idx 0; Idx.Fixed_idx 0 |]
            ~dyn_axis:0
            ~dyn_value:(get ids [| it i |], sp)))
  in
  let s_gather = CM.analyze gather in
  show_summary "dynamic gather (known column, any row)" s_gather;
  gather_bound_sound ~name:"dynamic gather (known column, any row)" s_gather a
    ~instances:(List.init 4 ~f:(fun _ row -> row * 5))
    ~rows:4;
  (* Box-limited: for b < 2: for c < 3: O[b][c] = T[I[b]][c] — 2 rows x 3 columns at most (24 B),
     where the known image times the extent (3 x 5) is T's whole 60 B. *)
  let t = fresh_tn "T" [| 5; 3 |] in
  let o = fresh_tn "O" [| 2; 3 |] in
  let ids2 = fresh_tn "I2" [| 2 |] in
  let box_limited =
    Ll_test.loop_n i 2
      (Ll_test.loop_n j 3
         (Ll_test.set o
            [| it i; it j |]
            (Ll_test.gather ~tn:t
               ~idcs:[| Idx.Fixed_idx 0; it j |]
               ~dyn_axis:0
               ~dyn_value:(get ids2 [| it i |], sp))))
  in
  let s_box = CM.analyze box_limited in
  show_summary "dynamic gather (box-limited)" s_box;
  gather_bound_sound ~name:"dynamic gather (box-limited)" s_box t
    ~instances:
      (List.concat_map (List.init 2 ~f:Fn.id) ~f:(fun _ ->
           List.init 3 ~f:(fun c row -> (row * 3) + c)))
    ~rows:5;
  (* Image-limited: for r < 4: for c < 2: P[r][c] = U[I[r]][c] over a 3x4 table — the box (8) is
     larger than the 3 rows x 2 gathered columns (24 B) the known image allows. *)
  let u = fresh_tn "U" [| 3; 4 |] in
  let pr = fresh_tn "P" [| 4; 2 |] in
  let ids3 = fresh_tn "I3" [| 4 |] in
  let image_limited =
    Ll_test.loop_n i 4
      (Ll_test.loop_n j 2
         (Ll_test.set pr
            [| it i; it j |]
            (Ll_test.gather ~tn:u
               ~idcs:[| Idx.Fixed_idx 0; it j |]
               ~dyn_axis:0
               ~dyn_value:(get ids3 [| it i |], sp))))
  in
  let s_image = CM.analyze image_limited in
  show_summary "dynamic gather (image-limited)" s_image;
  gather_bound_sound ~name:"dynamic gather (image-limited)" s_image u
    ~instances:
      (List.concat_map (List.init 4 ~f:Fn.id) ~f:(fun _ ->
           List.init 2 ~f:(fun c row -> (row * 4) + c)))
    ~rows:3;
  (* Flattened: for i < 8: F[i] = V[Sub_axis; I[i]] over a 2x4 table — a component after a
     [Sub_axis] run indexes the whole run, so the data picks any of 8 cells, not of the dynamic
     axis's own 4 (32 B; counting by the axis alone under-counted at 16 B). *)
  let v = fresh_tn "V" [| 2; 4 |] in
  let f8 = fresh_tn "F" [| 8 |] in
  let ids4 = fresh_tn "I4" [| 8 |] in
  let flattened =
    Ll_test.loop_n i 8
      (Ll_test.set f8
         [| it i |]
         (Ll_test.gather ~tn:v
            ~idcs:[| Idx.Sub_axis; Idx.Fixed_idx 0 |]
            ~dyn_axis:1
            ~dyn_value:(get ids4 [| it i |], sp)))
  in
  let s_flat = CM.analyze flattened in
  show_summary "dynamic gather (flattened over a Sub_axis run)" s_flat;
  gather_bound_sound ~name:"dynamic gather (flattened over a Sub_axis run)" s_flat v
    ~instances:(List.init 8 ~f:(fun _ row -> row))
    ~rows:8;

  (* Overlapping writes: Zero_out S2 then a covering pointwise write — the per-direction sum (16 +
     16 B) is a union bound, capped by the node's 16 bytes and flagged approximate. *)
  let s2 = fresh_tn "S2" [| 4 |] in
  let overlap =
    LL.unflat_lines
      [ LL.Zero_out s2; Ll_test.loop_n i 4 (Ll_test.set s2 [| it i |] (LL.Constant 0.5)) ]
  in
  show_summary "zero-out then overwrite (union bound capped)" (CM.analyze overlap);

  (* Disjoint multi-read exactness (gh-ocannl-578): C2[i] = A16[i] + A16[i+8] reads two disjoint
     4-cell slices — the union is their sum, 8 cells = 32 rd bytes, exact. *)
  let a16 = fresh_tn "A16" [| 16 |] in
  let c2 = fresh_tn "C2" [| 4 |] in
  let shift8 s = Idx.affine ~symbols:[ (1, s) ] ~offset:8 in
  let disjoint_slices =
    Ll_test.loop_n i 4
      (Ll_test.set c2
         [| it i |]
         (LL.Binop (Ops.Add, (get a16 [| it i |], sp), (get a16 [| shift8 i |], sp))))
  in
  show_summary "disjoint slice reads (exact union)" (CM.analyze disjoint_slices);

  (* Overlapping shifted reads: C2[i] = A16[i] + A16[i+1] — images {0..3} and {1..4} can share a
     cell, so the sum stays a flagged union bound. *)
  let shift1 s = Idx.affine ~symbols:[ (1, s) ] ~offset:1 in
  let overlapping_slices =
    Ll_test.loop_n i 4
      (Ll_test.set c2
         [| it i |]
         (LL.Binop (Ops.Add, (get a16 [| it i |], sp), (get a16 [| shift1 i |], sp))))
  in
  show_summary "overlapping slice reads (union bound)" (CM.analyze overlapping_slices);

  (* Parity-disjoint reads: C2[i] = A16[2i] * A16[2i+1] — evens and odds never collide (the gcd
     argument), 8 cells = 32 rd bytes, exact. *)
  let even s = Idx.affine ~symbols:[ (2, s) ] ~offset:0 in
  let odd s = Idx.affine ~symbols:[ (2, s) ] ~offset:1 in
  let parity =
    Ll_test.loop_n i 4
      (Ll_test.set c2
         [| it i |]
         (LL.Binop (Ops.Mul, (get a16 [| even i |], sp), (get a16 [| odd i |], sp))))
  in
  show_summary "parity-disjoint reads (exact union)" (CM.analyze parity);

  (* Conditionally-evaluated reads stay approximate even when disjoint (gh-ocannl-578 round 1):
     C2[i] = where(K4[i], A16[i] * 2, A16[i+8]) — only one arm executes per iteration, so A16's
     8-cell union is an upper bound, and charging both arms (unequal costs: 1 vs 0) makes the op
     count an upper bound too. *)
  let k4 = fresh_tn "K4" [| 4 |] in
  let where_arms =
    Ll_test.loop_n i 4
      (Ll_test.set c2
         [| it i |]
         (LL.Ternop
            ( Ops.Where,
              (get k4 [| it i |], sp),
              (LL.Binop (Ops.Mul, (get a16 [| it i |], sp), (LL.Constant 2., sp)), sp),
              (get a16 [| shift8 i |], sp) )))
  in
  show_summary "where-arm reads (conditional, stays a bound)" (CM.analyze where_arms);

  (* Equal-cost arms are charged in full but execute singly, so the op count is still a flagged
     bound (gh-ocannl-578 round 2): C2[i] = where(K4[i], A16[i] * 2, A16[i+8] * 3) charges both
     multiplications while one runs. *)
  let where_equal_arms =
    Ll_test.loop_n i 4
      (Ll_test.set c2
         [| it i |]
         (LL.Ternop
            ( Ops.Where,
              (get k4 [| it i |], sp),
              (LL.Binop (Ops.Mul, (get a16 [| it i |], sp), (LL.Constant 2., sp)), sp),
              (LL.Binop (Ops.Mul, (get a16 [| shift8 i |], sp), (LL.Constant 3., sp)), sp) )))
  in
  show_summary "where equal-cost arms (op count stays a bound)" (CM.analyze where_equal_arms);

  (* gh-ocannl-637: an arm whose whole cost is a hoisted [Local_scope] body executes unconditionally
     — every renderer emits the scope's definition before the statement — so charging it is exact:
     C2[i] = where(K4[i], { lv := A16[i] * 2 }, 0) counts 2 ops per cell with no flag, and A16's
     read inside the body is certain. *)
  let lv = fresh_tn "lv" [||] in
  Ll_test.virtualize lv;
  let hoisted body =
    LL.Local_scope
      { id = LL.get_scope lv; body; orig_indices = [| it i |]; mint = LL.Inlined_computation }
  in
  let scoped_mul idx k =
    hoisted
      (LL.Set_local
         (LL.get_scope lv, LL.Binop (Ops.Mul, (get a16 [| idx |], sp), (LL.Constant k, sp))))
  in
  let where_hoisted =
    Ll_test.loop_n i 4
      (Ll_test.set c2
         [| it i |]
         (LL.Ternop
            (Ops.Where, (get k4 [| it i |], sp), (scoped_mul (it i) 2., sp), (LL.Constant 0., sp))))
  in
  show_summary "where arm cost hoisted into a scope body (exact)" (CM.analyze where_hoisted);
  (* The same arm with one inline op on top of the hoisted body: only that op is conditional, and it
     is what flags the count — the scope's read stays certain. *)
  let where_hoisted_inline =
    Ll_test.loop_n i 4
      (Ll_test.set c2
         [| it i |]
         (LL.Ternop
            ( Ops.Where,
              (get k4 [| it i |], sp),
              (LL.Binop (Ops.Add, (scoped_mul (it i) 2., sp), (LL.Constant 1., sp)), sp),
              (LL.Constant 0., sp) )))
  in
  show_summary "where arm: hoisted body plus one inline op (bound)"
    (CM.analyze where_hoisted_inline);
  (* A gate's right operand hoisted likewise: C2[i] = relu_gate(K4[i], { lv := A16[i+8] * 3 }). *)
  let gate_hoisted =
    Ll_test.loop_n i 4
      (Ll_test.set c2
         [| it i |]
         (LL.Binop (Ops.Relu_gate, (get k4 [| it i |], sp), (scoped_mul (shift8 i) 3., sp))))
  in
  show_summary "gated operand cost hoisted into a scope body (exact)" (CM.analyze gate_hoisted);

  (* Vectorized runs (gh-ocannl-578), strip-mined: setv4 V16[4*i] — bases 4 apart tile the node, 16
     cells written exactly. The random-bits source is read once per run. *)
  let v16 = fresh_tn "V16" [| 16 |] in
  let src = fresh_tn "U" [| 4 |] in
  let base4 s = Idx.affine ~symbols:[ (4, s) ] ~offset:0 in
  (* for i < n: setv4 tn[idcs] from the random bits U[i]. *)
  let vec_store ?(n = 4) tn idcs =
    Ll_test.loop_n i n
      (LL.Set_from_vec
         {
           tn;
           idcs;
           length = 4;
           vec_unop = Ops.Uint4x32_to_prec_uniform;
           arg = (get src [| it i |], sp);
           debug = "";
         })
  in
  let vec_of = vec_store v16 in
  show_summary "vectorized writes, disjoint runs (exact)" (CM.analyze (vec_of [| base4 i |]));

  (* Close-spaced vec bases: setv4 V16[i] — runs from bases 1 apart may overlap, so the product
     stays a flagged upper bound (claims 16 cells where 7 distinct are touched). *)
  show_summary "vectorized writes, overlapping runs (bound)" (CM.analyze (vec_of [| it i |]));

  (* Row-spilling vec runs: setv4 W46[i][4] on a 4x6 node — each run crosses into the next row,
     where it could meet that row's base, so exactness is declined. *)
  let w46 = fresh_tn "W46" [| 4; 6 |] in
  let vec_spill = vec_store w46 [| it i; Idx.Fixed_idx 4 |] in
  show_summary "vectorized writes, row-spilling runs (bound)" (CM.analyze vec_spill);

  (* Constant minor base on distinct rows: setv4 W44[i][0] on a 4x4 node — one in-bounds run per
     row, disjoint by rows, 16 cells exact. *)
  let w44 = fresh_tn "W44" [| 4; 4 |] in
  let vec_rows = vec_store w44 [| it i; Idx.Fixed_idx 0 |] in
  show_summary "vectorized writes, one run per row (exact)" (CM.analyze vec_rows);

  (* Two vectorized stores on distinct rows: setv4 W28[0][4*i] then setv4 W28[1][4*i] on a 2x8 node.
     Each touches 8 cells exactly, and the pair query views each store as a run that its own loop
     bounds keep inside its row, so the stores are provably disjoint and the direction sums exactly
     to all 16 cells (a vectorized pair used to count as overlapping: a flagged bound). *)
  let w28 = fresh_tn "W28" [| 2; 8 |] in
  let vec_row r = vec_store ~n:2 w28 [| Idx.Fixed_idx r; base4 i |] in
  let s_vec_pair = CM.analyze (LL.unflat_lines [ vec_row 0; vec_row 1 ]) in
  show_summary "vectorized writes on distinct rows, two stores (exact)" s_vec_pair;
  let w28_fp = List.Assoc.find_exn s_vec_pair.CM.per_node w28 ~equal:Tn.equal in
  claim "two vectorized stores on distinct rows sum exactly to the whole node"
    ((not w28_fp.CM.fp_approx) && w28_fp.CM.fp_write_bytes = 16 * 4);

  (* Two vectorized stores binding the same loop symbol with different bounds: setv4 N34[1][0] for i
     < 1, and setv4 N34[i][0] for i < 2, on a 3x4 node. Together they write rows 0-1 (8 cells, 32 B)
     and read U[0..1] (8 B), so the union is no exact sum, and no floor may exceed 40 B. In both
     statement orders: the pair query must not read one store's bounds for the other's symbol. *)
  let n34 = fresh_tn "N34" [| 3; 4 |] in
  let one_row = vec_store ~n:1 n34 [| Idx.Fixed_idx 1; Idx.Fixed_idx 0 |]
  and two_rows = vec_store ~n:2 n34 [| it i; Idx.Fixed_idx 0 |] in
  List.iter
    [ ("one row first", [ one_row; two_rows ]); ("two rows first", [ two_rows; one_row ]) ]
    ~f:(fun (order, stmts) ->
      let code = LL.unflat_lines stmts in
      let s = CM.analyze code and floor = CM.completion_floor code in
      let name = "vectorized stores with unequal bounds, " ^ order in
      show_summary (name ^ " (bound)") s;
      Stdio.printf "  floor bytes=%d\n" floor.CM.fr_bytes;
      let fp = List.Assoc.find_exn s.CM.per_node n34 ~equal:Tn.equal in
      claimf "%s: the overlapping stores stay a flagged bound covering the union" name
        (fp.CM.fp_approx && fp.CM.fp_write_bytes >= 8 * 4);
      claimf "%s: the floor stays within the 40 B the stores and their source touch" name
        (floor.CM.fr_bytes <= (8 + 2) * 4));

  (* A [Concat] coordinate is uninterpretable to the view, yet every loop-box point still names one
     cell, which depends only on the symbols the map mentions: for a < 2: for b < 3: for k < 4:
     X16[a^b] counts at most box / k's width = 6 cells (24 B), flagged, where the whole-node
     fallback charged all of X16's 64 B. A concatenation of a 2- and a 3-cell segment can name 5
     cells, which the count must still cover. *)
  let x16 = fresh_tn "X16" [| 16 |] in
  let concat =
    Ll_test.loop_n i 2
      (Ll_test.loop_n j 3
         (Ll_test.loop_n k 4 (Ll_test.set x16 [| Idx.Concat [ i; j ] |] (LL.Constant 1.))))
  in
  let s_concat = CM.analyze concat in
  show_summary "concatenated coordinate (box over fiber bound)" s_concat;
  let x16_fp = List.Assoc.find_exn s_concat.CM.per_node x16 ~equal:Tn.equal in
  claim "a concatenated coordinate's count covers every cell the segments can name, flagged"
    (x16_fp.CM.fp_approx && x16_fp.CM.fp_write_bytes >= 5 * 4);
  claim "a concatenated coordinate's count is tighter than the whole node"
    (x16_fp.CM.fp_write_bytes < Tn.num_elems x16 * 4);

  (* Roofline: monotone in the envelope constants, bandwidth- vs. compute-bound flips. *)
  Stdio.printf "\n== roofline over the matmul (flops=%d, bytes=%d) ==\n" mm.CM.flops
    (CM.total_bytes mm);
  let bound ?peak_flops ?peak_memory_bandwidth () =
    CM.roofline_seconds ?peak_flops ?peak_memory_bandwidth ~flops:mm.CM.flops
      ~bytes:(CM.total_bytes mm) ()
  in
  let show_bound name t =
    Stdio.printf "  %-28s %s\n" name
      (match t with None -> "no bound" | Some t -> Printf.sprintf "%.1f ns" (t *. 1e9))
  in
  show_bound "no envelope" (bound ());
  show_bound "1 GFLOP/s only" (bound ~peak_flops:1e9 ());
  show_bound "1 GB/s only" (bound ~peak_memory_bandwidth:1e9 ());
  show_bound "1 GFLOP/s, 1 GB/s" (bound ~peak_flops:1e9 ~peak_memory_bandwidth:1e9 ());
  show_bound "2 GFLOP/s, 1 GB/s" (bound ~peak_flops:2e9 ~peak_memory_bandwidth:1e9 ());
  show_bound "1 GFLOP/s, 2 GB/s" (bound ~peak_flops:1e9 ~peak_memory_bandwidth:2e9 ());
  show_bound "2 GFLOP/s, 2 GB/s" (bound ~peak_flops:2e9 ~peak_memory_bandwidth:2e9 ());
  let peaks = [ 1e9; 2e9; 4e9 ] in
  Verdict.p_all "  monotone in the envelope constants" peaks ~f:(fun pf ->
      List.for_all peaks ~f:(fun bw ->
          let t0 = Option.value_exn (bound ~peak_flops:pf ~peak_memory_bandwidth:bw ()) in
          List.for_all
            [
              bound ~peak_flops:(2. *. pf) ~peak_memory_bandwidth:bw ();
              bound ~peak_flops:pf ~peak_memory_bandwidth:(2. *. bw) ();
            ]
            ~f:(fun t -> Float.(Option.value_exn t <= t0))))
