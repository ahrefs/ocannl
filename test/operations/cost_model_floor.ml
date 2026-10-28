(* gh-ocannl-514 phase 3: [Ir.Cost_model.completion_floor] — the dual (lower-bound) extraction over
   hand-built programs where every number is checkable by hand (single precision, 4 bytes/cell).
   Each case also asserts the duality invariant on the same code: the floor never exceeds the upper
   extraction ([analyze]).

   - pointwise map: all accesses exact, floor = upper on both legs; - matmul with an rmw
   accumulator: the floor leaves out the accumulator's read of its own cell, which a privatizing
   completion never performs (gh-ocannl-1601); opening the accumulator's placement attributes the
   whole producer — its ops and operand reads — to the open level, and committing (re-evaluation
   with the narrowed open set) recovers the closed floor, monotone in refinement; - guarded write:
   the floor counts guards-never-taken (condition ops only, no write traffic) where the upper counts
   guards-taken — [fr_exact] false; - two reads of one node: pairwise provably-disjoint exact images
   sum in both extractions (gh-ocannl-578); possibly-overlapping ones keep the asymmetry — the floor
   takes the larger exact image (a union is at least its largest member, flagged loose) where the
   upper takes the capped sum; - short-circuiting forms: Where arms, And/Or right operands
   (conditional — cheaper-arm / left-operand floors, conditional reads zeroed — but an arm's hoisted
   scope body is certain work in both extractions, gh-ocannl-637) and Arg1/Arg2 discarded operands
   (never rendered at all, so absent from both extractions); - the over-producing open producer,
   dead loops, and the dynamic gather, whose floor counts the projection onto its known coordinates
   (gh-ocannl-1601) where the upper falls back to the whole node.

   The last section pins the classifier those short-circuiting cases now read from
   ([Ops.binop_conditionality] / [Ops.ternop_conditionality], gh-ocannl-582) and its agreement with
   the plain-C renderings. *)

open Base
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Idx = Ir.Indexing
module Tn = Ir.Tnode
module Ops = Ir.Ops
module CM = Ir.Cost_model
module Affine = Ir.Affine

let fresh_tn =
  let make = Ll_test.node_factory ~first_id:980_000_000 ~dims:[||] () in
  fun label dims -> make ~dims label

let sp = Ops.single
let for_over ?(extent = 4) sym body = Ll_test.loop_n sym extent body
let set = Ll_test.set
let get = Ll_test.get
let it = Ll_test.iter

let show name ?open_placement code =
  let s = CM.analyze code in
  let f = CM.completion_floor ?open_placement code in
  Stdio.printf "== %s ==\n  floor: flops=%d bytes=%d %s\n  upper: flops=%d bytes=%d\n" name
    f.CM.fr_flops f.CM.fr_bytes
    (if f.CM.fr_exact then "exact" else "inexact")
    s.CM.flops (CM.total_bytes s);
  Verdict.p "  floor <= upper" (f.CM.fr_flops <= s.CM.flops && f.CM.fr_bytes <= CM.total_bytes s);
  f

let () =
  let i = Idx.get_symbol () and j = Idx.get_symbol () and k = Idx.get_symbol () in
  (* Pointwise 4x5 map: C[i][j] = A[i][j] + B[j]. All accesses exact: floor = upper. A rd 80 B, B rd
     20 B, C wr 80 B; 20 adds. *)
  let a = fresh_tn "A" [| 4; 5 |] in
  let b = fresh_tn "B" [| 5 |] in
  let c = fresh_tn "C" [| 4; 5 |] in
  let pointwise =
    for_over i
      (for_over ~extent:5 j
         (set c
            [| it i; it j |]
            (LL.Binop (Ops.Add, (get a [| it i; it j |], sp), (get b [| it j |], sp)))))
  in
  let _ = show "pointwise map" pointwise in
  (* 4x5x6 matmul, rmw accumulator: D[i][j] += A2[i][k] * B2[k][j]. The upper extraction is exact:
     flops = 2*4*5*6 = 240; bytes = A2 96 + B2 120 + D rd 80 + D wr 80 = 376. The floor leaves out
     D's read of its own cell — a privatizing schedule accumulates in a register and never performs
     it (gh-ocannl-1601) — so 296, inexact. Opening D's placement drops its write too; committing D
     restores it. *)
  let a2 = fresh_tn "A2" [| 4; 6 |] in
  let b2 = fresh_tn "B2" [| 6; 5 |] in
  let d = fresh_tn "D" [| 4; 5 |] in
  let matmul =
    for_over i
      (for_over ~extent:5 j
         (for_over ~extent:6 k
            (set d
               [| it i; it j |]
               (LL.Binop
                  ( Ops.Add,
                    (get d [| it i; it j |], sp),
                    ( LL.Binop
                        (Ops.Mul, (get a2 [| it i; it k |], sp), (get b2 [| it k; it j |], sp)),
                      sp ) )))))
  in
  let closed = show "matmul (rmw accumulator)" matmul in
  let opened = show "matmul, D's placement open" ~open_placement:(fun tn -> Tn.equal tn d) matmul in
  (* Committing a placement is re-evaluation with the narrowed open set: suppression only shrinks,
     so the floor is monotone in refinement — here, committing D (the only open node) recovers the
     closed floor exactly, and both legs are monotone. *)
  let committed = CM.completion_floor ~open_placement:(fun _ -> false) matmul in
  Verdict.p "  commitment (narrowed open set) recovers the closed floor"
    (committed.CM.fr_bytes = closed.CM.fr_bytes && committed.CM.fr_flops = closed.CM.fr_flops);
  Verdict.p "  floors are monotone in refinement"
    (opened.CM.fr_bytes <= committed.CM.fr_bytes && opened.CM.fr_flops <= committed.CM.fr_flops);
  (* Guarded write: if (E[i] > 0) F[i] = E[i] * 2. Upper counts guards-taken (cmp + mul per
     iteration, F written); the floor counts only the certain condition ops and no F traffic. *)
  let e = fresh_tn "E" [| 4 |] in
  let fq = fresh_tn "F" [| 4 |] in
  let guarded =
    for_over i
      (LL.If
         {
           cond = (LL.Binop (Ops.Cmplt, (LL.Constant 0., sp), (get e [| it i |], sp)), sp);
           body =
             set fq [| it i |] (LL.Binop (Ops.Mul, (get e [| it i |], sp), (LL.Constant 2., sp)));
         })
  in
  let _ = show "guarded write" guarded in
  (* Two exact reads of one node with provably disjoint images: G[i][j] = H[0][j] + H[1][j]. Both
     extractions agree the union is the sum (gh-ocannl-578): 10 cells = 40 rd bytes, G writes 80
     bytes both ways — floor = upper = 120, exact. *)
  let g = fresh_tn "G" [| 4; 5 |] in
  let h = fresh_tn "H" [| 4; 5 |] in
  let two_reads =
    for_over i
      (for_over ~extent:5 j
         (set g
            [| it i; it j |]
            (LL.Binop
               ( Ops.Add,
                 (get h [| Idx.Fixed_idx 0; it j |], sp),
                 (get h [| Idx.Fixed_idx 1; it j |], sp) ))))
  in
  let _ = show "two disjoint reads, both extractions sum" two_reads in
  (* Two exact reads that may overlap: G2[i] = H8[i] + H8[i+1]. The upper sums (8 cells = 32 rd
     bytes, a union bound); the floor takes only the larger exact image (4 cells = 16 bytes) — a
     union is at least its largest member, no more is certain. G2 writes 16 both ways: floor 32 vs
     upper 48. *)
  let g2 = fresh_tn "G2" [| 4 |] in
  let h8 = fresh_tn "H8" [| 8 |] in
  let shift1 s = Idx.affine ~symbols:[ (1, s) ] ~offset:1 in
  let overlapping_reads =
    for_over i
      (set g2 [| it i |] (LL.Binop (Ops.Add, (get h8 [| it i |], sp), (get h8 [| shift1 i |], sp))))
  in
  let _ = show "two overlapping reads, union floor = larger image" overlapping_reads in
  (* Dynamic gather: P[i] = Q[R[i]]. Q's access is uninterpretable — the upper falls back to the
     whole node; the floor counts the projection onto Q's known coordinates (gh-ocannl-1601), here
     none, so one cell (inexact). *)
  let p = fresh_tn "P" [| 4 |] in
  let q = fresh_tn "Q" [| 8 |] in
  let r = fresh_tn "R" [| 4 |] in
  let gather =
    for_over i
      (set p
         [| it i |]
         (LL.Get_dynamic
            {
              tn = q;
              idcs = [| Idx.Fixed_idx 0 |];
              dyn_axis = 0;
              dyn_value = (get r [| it i |], sp);
            }))
  in
  let _ = show "dynamic gather" gather in
  (* Where short-circuits (?: in every renderer): K[i] = where(E>0, E*2, E+M[i]+1). The floor counts
     cond + select + the cheaper arm (1+1+1 = 3/iter vs the upper's both-arms 5/iter), and M — read
     only inside an arm — floors its read to zero; E is also read in an arm, so its read floors too
     (node-granular certainty, conservative). *)
  let kq = fresh_tn "K" [| 4 |] in
  let e2 = fresh_tn "E2" [| 4 |] in
  let m = fresh_tn "M" [| 4 |] in
  let where_case =
    for_over i
      (set kq
         [| it i |]
         (LL.Ternop
            ( Ops.Where,
              (LL.Binop (Ops.Cmplt, (LL.Constant 0., sp), (get e2 [| it i |], sp)), sp),
              (LL.Binop (Ops.Mul, (get e2 [| it i |], sp), (LL.Constant 2., sp)), sp),
              ( LL.Binop
                  ( Ops.Add,
                    (get e2 [| it i |], sp),
                    (LL.Binop (Ops.Add, (get m [| it i |], sp), (LL.Constant 1., sp)), sp) ),
                sp ) )))
  in
  let _ = show "where short-circuit" where_case in
  (* gh-ocannl-637: an arm whose cost is a hoisted scope body executes whichever arm is selected
     (the renderers emit scope definitions before the statement), so both extractions charge it:
     K[i] = where(0 < M[i], { lv := E2[i] * 2 }, 0) — cond + select + the body's multiply, 3/iter,
     floor = upper = 12, and E2's read inside the body is certain in both. *)
  let lv = fresh_tn "lv" [||] in
  Ll_test.virtualize lv;
  let where_hoisted =
    for_over i
      (set kq
         [| it i |]
         (LL.Ternop
            ( Ops.Where,
              (LL.Binop (Ops.Cmplt, (LL.Constant 0., sp), (get m [| it i |], sp)), sp),
              ( LL.Local_scope
                  {
                    id = LL.get_scope lv;
                    body =
                      LL.Set_local
                        ( LL.get_scope lv,
                          LL.Binop (Ops.Mul, (get e2 [| it i |], sp), (LL.Constant 2., sp)) );
                    orig_indices = [| it i |];
                    mint = LL.Inlined_computation;
                  },
                sp ),
              (LL.Constant 0., sp) )))
  in
  let hoisted = show "where arm hoisted into a scope body" where_hoisted in
  let upper = CM.analyze where_hoisted in
  Verdict.p "  hoisted arm: floor = upper on both legs, both exact"
    (hoisted.CM.fr_exact
    && (not (CM.approximate upper))
    && hoisted.CM.fr_flops = upper.CM.flops
    && hoisted.CM.fr_bytes = CM.total_bytes upper);
  (* An open producer computing a larger domain than consumed: P2[0..7] = A3[0..7] * 2, then C3[0] =
     P2[0]. The inline completion instantiates one multiply and one A3 read and drops the setter
     loop, so with P2 open the floor keeps only C3's certain write — the producer's ops AND its A3
     reads attribute to the open placement. *)
  let p2 = fresh_tn "P2" [| 8 |] in
  let a3 = fresh_tn "A3" [| 8 |] in
  let c3 = fresh_tn "C3" [| 1 |] in
  let over_produce =
    LL.Seq
      ( for_over ~extent:8 j
          (set p2 [| it j |] (LL.Binop (Ops.Mul, (get a3 [| it j |], sp), (LL.Constant 2., sp)))),
        set c3 [| Idx.Fixed_idx 0 |] (get p2 [| Idx.Fixed_idx 0 |]) )
  in
  let _ = show "over-producing closed" over_produce in
  let _ =
    show "over-producing, P2's placement open"
      ~open_placement:(fun tn -> Tn.equal tn p2)
      over_produce
  in
  (* And renders as the short-circuiting &&: the certain floor is the op plus the left operand; the
     right comparison (reading M) is conditional — its op floors away and M's read floors to zero.
     Arg1 renders only its selected operand: the discarded expensive right operand contributes to
     neither extraction — not its ops, not its M read. *)
  let l = fresh_tn "L" [| 4 |] in
  let and_case =
    for_over i
      (set l
         [| it i |]
         (LL.Binop
            ( Ops.And,
              (LL.Binop (Ops.Cmplt, (LL.Constant 0., sp), (get e2 [| it i |], sp)), sp),
              (LL.Binop (Ops.Cmplt, (get m [| it i |], sp), (LL.Constant 1., sp)), sp) )))
  in
  let _ = show "and short-circuit" and_case in
  let arg1_case =
    for_over i
      (set l
         [| it i |]
         (LL.Binop
            ( Ops.Arg1,
              (get e2 [| it i |], sp),
              (LL.Binop (Ops.Mul, (get m [| it i |], sp), (LL.Constant 3., sp)), sp) )))
  in
  let _ = show "arg1 discarded operand" arg1_case in
  (* Relu_gate also renders with ?: — its right operand (the expensive product reading M) is
     conditional: floor = gate + left operand per iteration, M's read zeroed. *)
  let gate_case =
    for_over i
      (set l
         [| it i |]
         (LL.Binop
            ( Ops.Relu_gate,
              (get e2 [| it i |], sp),
              (LL.Binop (Ops.Mul, (get m [| it i |], sp), (LL.Constant 3., sp)), sp) )))
  in
  let _ = show "relu-gate short-circuit" gate_case in
  (* A dead loop's body never executes: its whole-node access must not reach the floor. *)
  let d2 = fresh_tn "D2" [| 4 |] in
  let dead =
    LL.For_loop { index = k; from_ = 0; to_ = -1; body = LL.Zero_out d2; axis = LL.Serial }
  in
  let _ = show "dead loop" dead in
  ()

(* gh-ocannl-1601: the image lower bound behind the floor of a non-exact access
   ([Affine.image_cardinality_lb]), judged against the image ENUMERATED cell by cell. Every case
   builds an index map over a small loop box (origins shifted, widths 1-4), sizes the node so the
   map stays in bounds, enumerates the box and counts the distinct cells, and claims the bound never
   exceeds that count. The shapes: one-coordinate sums of one to three terms over every coefficient
   and width combination in a range (unit and larger strides, dilations with holes, negative
   coefficients, a symbol repeated so its terms cancel); random multi-coordinate maps, some sharing
   a symbol across coordinates; flattened [Sub_axis] pairs; dynamic gathers, whose data-chosen
   coordinate the enumeration picks both adversarially (a constant) and at random; vector runs. Then
   the tightness pins on windows, and the floor of hand-built window programs. *)
let () =
  Stdio.printf "\n== image lower bound vs enumerated image ==\n";
  let rng = Random.State.make [| 1601 |] in
  let s = Array.init 3 ~f:(fun _ -> Idx.get_symbol ()) in
  let eval env (idx : Idx.axis_index) =
    match idx with
    | Idx.Fixed_idx c -> c
    | Idx.Iterator sy -> env sy
    | Idx.Affine { symbols; offset } ->
        List.fold symbols ~init:offset ~f:(fun acc (c, sy) -> acc + (c * env sy))
    | Idx.Sub_axis | Idx.Concat _ -> assert false
  in
  (* The loop box: per symbol, an origin and a width. *)
  let points (box : (Idx.symbol * (int * int)) list) =
    List.fold box ~init:[ [] ] ~f:(fun acc (sy, (lo, hi)) ->
        List.concat_map acc ~f:(fun env ->
            List.map (List.range lo (hi + 1)) ~f:(fun v -> (sy, v) :: env)))
    |> List.map ~f:(fun env sy ->
        List.Assoc.find env sy ~equal:Idx.equal_symbol |> Option.value ~default:0)
  in
  (* One coordinate's terms shifted to start at 0: the map component and its extent. *)
  let component box terms ~slack =
    let lo, hi =
      List.fold terms ~init:(0, 0) ~f:(fun (lo, hi) (c, sy) ->
          let a, b = List.Assoc.find_exn box sy ~equal:Idx.equal_symbol in
          (lo + min (c * a) (c * b), hi + max (c * a) (c * b)))
    in
    (Idx.affine ~symbols:terms ~offset:(-lo + slack), hi - lo + 1 + slack)
  in
  let domain box = List.map box ~f:(fun (sy, (lo, hi)) -> (sy, hi - lo + 1)) in
  let cases = ref [] in
  let record name bound image = cases := (name, bound, image) :: !cases in
  (* gh-ocannl-1676: the upper side. [analyze]'s count of one access, run over the same box as a
     program, against the true count — for a dynamic gather the WORST case over the data, which may
     pick the dynamic coordinate per point independently. [exact] is [analyze]'s own verdict. *)
  let upper = ref [] in
  let record_upper name ~count ~exact ~truth = upper := (name, count, exact, truth) :: !upper in
  let in_box box body =
    List.fold_right box ~init:body ~f:(fun (sy, (lo, hi)) body ->
        Ll_test.loop ~from_:lo ~upto:hi sy body)
  in
  let analyzed tn code =
    let s = CM.analyze code in
    let fp = List.Assoc.find_exn s.CM.per_node tn ~equal:Tn.equal in
    ((fp.CM.fp_read_bytes + fp.CM.fp_write_bytes) / 4, not fp.CM.fp_approx)
  in
  (* A static map: [cell] enumerates each point's cell tuple. *)
  let static_case name box idcs dims =
    let v = Affine.view ~dims idcs in
    let bound = Affine.image_cardinality_lb ~domain:(domain box) v in
    let cells = List.map (points box) ~f:(fun env -> Array.map idcs ~f:(eval env)) in
    let image = List.length (List.dedup_and_sort cells ~compare:(Array.compare Int.compare)) in
    record name bound image
  in
  let random_box () =
    Array.to_list s
    |> List.map ~f:(fun sy ->
        let lo = Random.State.int rng 2 and w = 1 + Random.State.int rng 4 in
        (sy, (lo, lo + w - 1)))
  in
  (* One coordinate, every combination of up to three terms. *)
  let coeffs = [ -3; -2; -1; 1; 2; 3; 4; 6 ] and widths = [ 1; 2; 3; 4 ] in
  List.iter coeffs ~f:(fun c1 ->
      List.iter widths ~f:(fun w1 ->
          List.iter coeffs ~f:(fun c2 ->
              List.iter widths ~f:(fun w2 ->
                  let box = [ (s.(0), (1, w1)); (s.(1), (0, w2 - 1)) ] in
                  let idx, size = component box [ (c1, s.(0)); (c2, s.(1)) ] ~slack:0 in
                  static_case "1 coord, 2 terms" box [| idx |] [| size |];
                  List.iter [ 1; 5 ] ~f:(fun c3 ->
                      let box = box @ [ (s.(2), (0, 2)) ] in
                      let idx, size =
                        component box [ (c1, s.(0)); (c2, s.(1)); (c3, s.(2)) ] ~slack:1
                      in
                      static_case "1 coord, 3 terms" box [| idx |] [| size |])))));
  (* Cancellation: one symbol twice, its coefficients summing to zero or not. *)
  List.iter
    [ (1, -1); (2, -1); (3, -3) ]
    ~f:(fun (a, b) ->
      let box = [ (s.(0), (0, 3)); (s.(1), (0, 2)) ] in
      let idx, size = component box [ (a, s.(0)); (b, s.(0)); (1, s.(1)) ] ~slack:0 in
      static_case "cancelling terms" box [| idx |] [| size |]);
  (* Random multi-coordinate maps; the symbol pool is small, so coordinates often share one. *)
  for _ = 1 to 600 do
    let box = random_box () in
    let rank = 1 + Random.State.int rng 3 in
    let comps =
      Array.init rank ~f:(fun _ ->
          let nterms = Random.State.int rng 3 in
          let terms =
            List.init nterms ~f:(fun _ ->
                ( List.nth_exn coeffs (Random.State.int rng (List.length coeffs)),
                  s.(Random.State.int rng 3) ))
          in
          component box terms ~slack:(Random.State.int rng 2))
    in
    static_case "random map" box (Array.map comps ~f:fst) (Array.map comps ~f:snd)
  done;
  (* Flattened [Sub_axis] pairs: the component after the run indexes the whole run's extent. *)
  for _ = 1 to 200 do
    let box = random_box () in
    let terms =
      List.init
        (1 + Random.State.int rng 3)
        ~f:(fun _ -> (1 + Random.State.int rng 4, s.(Random.State.int rng 3)))
    in
    let idx, size = component box terms ~slack:0 in
    let inner = (size + 1) / 2 in
    let v_idcs = [| Idx.Sub_axis; idx |] in
    let v = Affine.view ~dims:[| 2; inner |] v_idcs in
    let bound = Affine.image_cardinality_lb ~domain:(domain box) v in
    let image =
      List.length
        (List.dedup_and_sort ~compare:Int.compare
           (List.map (points box) ~f:(fun env -> eval env idx)))
    in
    record "flattened Sub_axis" bound image
  done;
  (* Dynamic gathers over a rank-2 node: the data picks the dynamic axis's coordinate per point. *)
  for _ = 1 to 200 do
    let box = random_box () in
    let pick () =
      List.init (Random.State.int rng 3) ~f:(fun _ ->
          (1 + Random.State.int rng 3, s.(Random.State.int rng 3)))
    in
    let known, ksize = component box (pick ()) ~slack:0 in
    let dyn_axis = Random.State.int rng 2 and dsize = 1 + Random.State.int rng 4 in
    let idcs, dims =
      if dyn_axis = 0 then ([| Idx.Fixed_idx 0; known |], [| dsize; ksize |])
      else ([| known; Idx.Fixed_idx 0 |], [| ksize; dsize |])
    in
    let v = Affine.view ~dyn_axis ~dims idcs in
    let bound = Affine.image_cardinality_lb ~domain:(domain box) v in
    List.iter
      [ (fun _ -> 0); (fun _ -> Random.State.int rng dsize) ]
      ~f:(fun choose ->
        let cells =
          List.map (points box) ~f:(fun env ->
              let k = eval env known and d = choose env in
              if dyn_axis = 0 then (d, k) else (k, d))
        in
        record "dynamic gather" bound
          (List.length
             (List.dedup_and_sort cells ~compare:(fun (a1, b1) (a2, b2) ->
                  match Int.compare a1 a2 with 0 -> Int.compare b1 b2 | c -> c))));
    (* The worst case: each known value reached by [f] points pairs with at most [min f dsize]
       distinct values of the data-chosen coordinate. *)
    let fibers =
      List.map (points box) ~f:(fun env -> eval env known)
      |> List.sort_and_group ~compare:Int.compare
      |> List.map ~f:List.length
    in
    let worst = List.sum (module Int) fibers ~f:(fun f -> min f dsize) in
    let tn = fresh_tn "DG" dims and out = fresh_tn "DO" [||] in
    let count, exact =
      analyzed tn
        (in_box box
           (set out [||] (Ll_test.gather ~tn ~idcs ~dyn_axis ~dyn_value:(LL.Constant 0., sp))))
    in
    record_upper "dynamic gather" ~count ~exact ~truth:worst
  done;
  (* Vector runs of [len] consecutive flat cells from each base, over a rank-2 node. *)
  for _ = 1 to 200 do
    let box = random_box () in
    let len = 2 + Random.State.int rng 3 in
    let row, rsize = component box [ (1 + Random.State.int rng 2, s.(0)) ] ~slack:0 in
    let col, csize =
      component box [ (len * (1 + Random.State.int rng 2), s.(1)); (1, s.(2)) ] ~slack:0
    in
    let csize = csize + len in
    let idcs = [| row; col |] and dims = [| rsize; csize |] in
    let range sy = List.Assoc.find box sy ~equal:Idx.equal_symbol in
    let v = Affine.view ~dims idcs in
    let bound = Affine.image_cardinality_lb ~range ~run:len ~domain:(domain box) v in
    let cells =
      List.concat_map (points box) ~f:(fun env ->
          let base = (eval env row * csize) + eval env col in
          List.init len ~f:(fun i -> base + i))
    in
    let truth = List.length (List.dedup_and_sort cells ~compare:Int.compare) in
    record "vector run" bound truth;
    let tn = fresh_tn "VR" dims in
    let count, exact =
      analyzed tn
        (in_box box
           (Ll_test.set_from_vec ~length:len ~vec_unop:Ops.Uint4x32_to_prec_uniform tn idcs
              (LL.Constant 0., sp)))
    in
    record_upper "vector run" ~count ~exact ~truth
  done;
  (* Vector runs over random one- and two-coordinate bases (gh-ocannl-1676), windowed and strided,
     some spilling out of their row: the cells are the flat offsets [base + t], [t < len]. *)
  for _ = 1 to 200 do
    let box = random_box () in
    let len = 2 + Random.State.int rng 3 in
    let pick () =
      List.init
        (1 + Random.State.int rng 2)
        ~f:(fun _ -> (1 + Random.State.int rng 4, s.(Random.State.int rng 3)))
    in
    let col, csize = component box (pick ()) ~slack:0 in
    let idcs, dims, flat =
      if Random.State.bool rng then ([| col |], [| csize + len |], fun env -> eval env col)
      else
        let row, rsize = component box (pick ()) ~slack:0 in
        (* A spilling row: the minor extent is the base range alone, or one short of the run. *)
        let width = csize + Random.State.int rng (len + 1) in
        ([| row; col |], [| rsize + len; width |], fun env -> (eval env row * width) + eval env col)
    in
    let cells =
      List.concat_map (points box) ~f:(fun env -> List.init len ~f:(fun i -> flat env + i))
    in
    let truth = List.length (List.dedup_and_sort cells ~compare:Int.compare) in
    let range sy = List.Assoc.find box sy ~equal:Idx.equal_symbol in
    record "vector run (windowed)"
      (Affine.image_cardinality_lb ~range ~run:len ~domain:(domain box) (Affine.view ~dims idcs))
      truth;
    let tn = fresh_tn "VW" dims in
    let count, exact =
      analyzed tn
        (in_box box
           (Ll_test.set_from_vec ~length:len ~vec_unop:Ops.Uint4x32_to_prec_uniform tn idcs
              (LL.Constant 0., sp)))
    in
    record_upper "vector run (windowed)" ~count ~exact ~truth
  done;
  let cases = List.rev !cases in
  let kinds =
    List.dedup_and_sort ~compare:String.compare (List.map cases ~f:(fun (n, _, _) -> n))
  in
  List.iter kinds ~f:(fun kind ->
      let of_kind = List.filter cases ~f:(fun (n, _, _) -> String.equal n kind) in
      Stdio.printf "  %-20s %4d cases, %4d bounded exactly\n" kind (List.length of_kind)
        (List.count of_kind ~f:(fun (_, b, i) -> b = i)));
  Verdict.p_all ~min:2000 "  the bound never exceeds the enumerated image" cases
    ~f:(fun (_, bound, image) -> bound <= image);
  (* The upper side's census counts [analyze]'s count meeting the true (worst-case) count — a
     different tightness from the lower bound's above. *)
  let upper = List.rev !upper in
  Stdio.printf "  -- analyze's count vs the true count (worst case over the data) --\n";
  List.iter
    (List.dedup_and_sort ~compare:String.compare (List.map upper ~f:(fun (n, _, _, _) -> n)))
    ~f:(fun kind ->
      let of_kind = List.filter upper ~f:(fun (n, _, _, _) -> String.equal n kind) in
      Stdio.printf "  %-21s %4d cases, %4d counted exactly, %4d claimed exact\n" kind
        (List.length of_kind)
        (List.count of_kind ~f:(fun (_, c, _, t) -> c = t))
        (List.count of_kind ~f:(fun (_, _, e, _) -> e)));
  Verdict.p_all ~min:600 "  analyze's count never falls below the true count" upper
    ~f:(fun (_, count, _, truth) -> count >= truth);
  Verdict.p_all ~min:600 "  analyze's exact verdicts are the true count" upper
    ~f:(fun (_, count, exact, truth) -> (not exact) || count = truth);
  Verdict.p_none "  no dynamic gather is claimed exact" upper ~f:(fun (n, _, e, _) ->
      e && String.equal n "dynamic gather");
  (* Tightness: overlapping and strided windows count their whole image. *)
  let window ~stride ~o ~k =
    let box = [ (s.(0), (0, o - 1)); (s.(1), (0, k - 1)) ] in
    let idx, size = component box [ (stride, s.(0)); (1, s.(1)) ] ~slack:0 in
    Affine.image_cardinality_lb ~domain:(domain box) (Affine.view ~dims:[| size |] [| idx |])
  in
  Verdict.p "  a unit-stride window counts O + K - 1 (8 + 3 - 1)" (window ~stride:1 ~o:8 ~k:3 = 10);
  Verdict.p "  a stride-2 window counts 2*(O-1) + K (2*7 + 3)" (window ~stride:2 ~o:8 ~k:3 = 17);
  Verdict.p "  a non-overlapping stride-4 window counts O*K (8*3)" (window ~stride:4 ~o:8 ~k:3 = 24);
  let gcd_case =
    let box = [ (s.(0), (0, 2)); (s.(1), (0, 2)) ] in
    let idx, size = component box [ (2, s.(0)); (4, s.(1)) ] ~slack:0 in
    Affine.image_cardinality_lb ~domain:(domain box) (Affine.view ~dims:[| size |] [| idx |])
  in
  Verdict.p "  2*x + 4*y over 3x3 counts its 7 values (the gcd normalization)" (gcd_case = 7);
  (* The floor of window programs: B[o] += A[s*o + k] reads A's whole image. *)
  let window_program ~stride ~o ~k =
    let oi = Idx.get_symbol () and ki = Idx.get_symbol () in
    let a = fresh_tn "WA" [| (stride * (o - 1)) + k |] in
    let b = fresh_tn "WB" [| o |] in
    let code =
      for_over ~extent:o oi
        (for_over ~extent:k ki
           (set b
              [| it oi |]
              (LL.Binop
                 ( Ops.Add,
                   (get b [| it oi |], sp),
                   (get a [| Idx.affine ~symbols:[ (stride, oi); (1, ki) ] ~offset:0 |], sp) ))))
    in
    show (Printf.sprintf "window read, stride %d, O=%d, K=%d" stride o k) code
  in
  let f1 = window_program ~stride:1 ~o:8 ~k:3 in
  (* A 40 B, B wr 32 B; B's accumulation self-read is not certain (a privatizing completion skips
     it). *)
  Verdict.p "  the unit-stride window's input read floors at its whole image" (f1.CM.fr_bytes = 72);
  let f2 = window_program ~stride:2 ~o:8 ~k:3 in
  (* A 68 B, B wr 32 B. *)
  Verdict.p "  the stride-2 window's input read floors at its whole image" (f2.CM.fr_bytes = 100)

(* gh-ocannl-582: the classifier the cases above read from. The table is printed over [Ops]' derived
   enumeration, so adding an operator is a visible promotion diff here whether or not anyone
   remembers this test; the second half checks the plain-C renderings against the classifier — the
   same check every backend runs on itself at [C_syntax] functor application, where the GPU
   spellings (which shadow these) get covered on their own hardware. *)
module C_config = Ir.C_syntax.Pure_C_config (struct
  let procs = [||]
  let full_printf_support = true
end)

let () =
  Stdio.printf "\n== operand conditionality ==\n";
  List.iter Ops.all_of_binop ~f:(fun op ->
      Stdio.printf "  %-28s %s\n" (Ops.binop_cd_fallback_syntax op)
        (Sexp.to_string @@ Ops.sexp_of_binop_conditionality @@ Ops.binop_conditionality op));
  List.iter Ops.all_of_ternop ~f:(fun op ->
      Stdio.printf "  %-28s %s\n" (Ops.ternop_cd_syntax op)
        (Sexp.to_string @@ Ops.sexp_of_ternop_conditionality @@ Ops.ternop_conditionality op));
  let violations =
    Ir.C_syntax.operand_conditionality_violations ~ternop_syntax:C_config.ternop_syntax
      ~binop_syntax:C_config.binop_syntax ~select_blend_syntax:C_config.select_blend_syntax
  in
  (* Over the operators the table above enumerates, which is the sweep the violations came from. *)
  let swept =
    List.map Ops.all_of_binop ~f:Ops.binop_cd_fallback_syntax
    @ List.map Ops.all_of_ternop ~f:Ops.ternop_cd_syntax
  in
  Verdict.p_empty "  plain-C renderings agree" ~over:swept violations;
  List.iter violations ~f:(fun v -> Stdio.printf "  VIOLATION %s\n" v)
