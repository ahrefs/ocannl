(* gh-ocannl-491 task 1: [Ir.Cost_model] — footprint/FLOPs extraction and the roofline bound over
   hand-built programs where every number is checkable by hand (single precision, 4 bytes/cell):

   - elementwise map: bytes = the three operand footprints, one op per cell; - matmul: FLOPs =
   2*M*N*K, bytes = (MK + KN + 2*MN) * 4 (the rmw accumulator charged once per direction); -
   strided/gapped access: the image cardinality counts the touched cells, not the node size; - rmw
   reduction: the accumulator's read/write/rmw split; - guarded write: guards-taken op count
   ([flops_approx]) and a never-definite write; - dynamic gather: the uninterpretable-component
   fallback to whole-node bytes; - overlapping writes: the union bound capped by the node's size; -
   multi-read exactness (gh-ocannl-578): pairwise provably-disjoint exact reads sum exactly,
   overlapping ones stay a flagged union bound, and conditionally-evaluated reads (Where arms) stay
   a flagged bound even when disjoint — with the op count flagged too when the arms' costs differ; -
   vectorized runs (gh-ocannl-578): bases spaced by at least the run length (or on distinct
   in-bounds rows) count exactly, close-spaced or row-spilling bases stay a flagged upper bound.

   The tail asserts the roofline bound is monotone in the envelope constants. *)

open Base
module LL = Ir.Low_level
module Idx = Ir.Indexing
module Tn = Ir.Tnode
module Ops = Ir.Ops
module CM = Ir.Cost_model

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
  let stride2 s = Idx.Affine { symbols = [ (2, s) ]; offset = 0 } in
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

  (* Dynamic gather: for i: E[i] = A[I[i]][0] — the table read falls back to whole-node bytes. *)
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
  show_summary "dynamic gather (whole-node fallback)" (CM.analyze gather);

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
  let shift8 s = Idx.Affine { symbols = [ (1, s) ]; offset = 8 } in
  let disjoint_slices =
    Ll_test.loop_n i 4
      (Ll_test.set c2
         [| it i |]
         (LL.Binop (Ops.Add, (get a16 [| it i |], sp), (get a16 [| shift8 i |], sp))))
  in
  show_summary "disjoint slice reads (exact union)" (CM.analyze disjoint_slices);

  (* Overlapping shifted reads: C2[i] = A16[i] + A16[i+1] — images {0..3} and {1..4} can share a
     cell, so the sum stays a flagged union bound. *)
  let shift1 s = Idx.Affine { symbols = [ (1, s) ]; offset = 1 } in
  let overlapping_slices =
    Ll_test.loop_n i 4
      (Ll_test.set c2
         [| it i |]
         (LL.Binop (Ops.Add, (get a16 [| it i |], sp), (get a16 [| shift1 i |], sp))))
  in
  show_summary "overlapping slice reads (union bound)" (CM.analyze overlapping_slices);

  (* Parity-disjoint reads: C2[i] = A16[2i] * A16[2i+1] — evens and odds never collide (the gcd
     argument), 8 cells = 32 rd bytes, exact. *)
  let even s = Idx.Affine { symbols = [ (2, s) ]; offset = 0 } in
  let odd s = Idx.Affine { symbols = [ (2, s) ]; offset = 1 } in
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

  (* Vectorized runs (gh-ocannl-578), strip-mined: setv4 V16[4*i] — bases 4 apart tile the node, 16
     cells written exactly. The random-bits source is read once per run. *)
  let v16 = fresh_tn "V16" [| 16 |] in
  let src = fresh_tn "U" [| 4 |] in
  let base4 s = Idx.Affine { symbols = [ (4, s) ]; offset = 0 } in
  let vec_of idcs =
    Ll_test.loop_n i 4
      (LL.Set_from_vec
         {
           tn = v16;
           idcs;
           length = 4;
           vec_unop = Ops.Uint4x32_to_prec_uniform;
           arg = (get src [| it i |], sp);
           debug = "";
         })
  in
  show_summary "vectorized writes, disjoint runs (exact)" (CM.analyze (vec_of [| base4 i |]));

  (* Close-spaced vec bases: setv4 V16[i] — runs from bases 1 apart may overlap, so the product
     stays a flagged upper bound (claims 16 cells where 7 distinct are touched). *)
  show_summary "vectorized writes, overlapping runs (bound)" (CM.analyze (vec_of [| it i |]));

  (* Row-spilling vec runs: setv4 W46[i][4] on a 4x6 node — each run crosses into the next row,
     where it could meet that row's base, so exactness is declined. *)
  let w46 = fresh_tn "W46" [| 4; 6 |] in
  let vec_spill =
    Ll_test.loop_n i 4
      (LL.Set_from_vec
         {
           tn = w46;
           idcs = [| it i; Idx.Fixed_idx 4 |];
           length = 4;
           vec_unop = Ops.Uint4x32_to_prec_uniform;
           arg = (get src [| it i |], sp);
           debug = "";
         })
  in
  show_summary "vectorized writes, row-spilling runs (bound)" (CM.analyze vec_spill);

  (* Constant minor base on distinct rows: setv4 W44[i][0] on a 4x4 node — one in-bounds run per
     row, disjoint by rows, 16 cells exact. *)
  let w44 = fresh_tn "W44" [| 4; 4 |] in
  let vec_rows =
    Ll_test.loop_n i 4
      (LL.Set_from_vec
         {
           tn = w44;
           idcs = [| it i; Idx.Fixed_idx 0 |];
           length = 4;
           vec_unop = Ops.Uint4x32_to_prec_uniform;
           arg = (get src [| it i |], sp);
           debug = "";
         })
  in
  show_summary "vectorized writes, one run per row (exact)" (CM.analyze vec_rows);

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
