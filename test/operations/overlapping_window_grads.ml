(* Exact tropical/einmax1 gradients via the product-space gate (gh-ocannl-512).

   The gradient gate of the max-family einsums lives in the product space of the operation — one bit
   per (result position, contracted position) pair ([{ cond_pspace }] in [Operation.tropical] /
   [Operation.einmax1]) — instead of a last-write-wins bit per input position. Pinned here:

   - Overlapping reduction windows (stride < window): positions receive gradient exactly from the
   windows they won — the issue's minimal repro.

   - General specs where RHS2 has indices independent of RHS1 (tropical matmul): the g2 gradient is
   exact — the previously documented limitation is gone.

   - Ties: the gradient goes in full to every achieving pair, and window contributions accumulate.

   - Clamped padded windows (gh-ocannl-504) with overlap: out-of-range pairs hold the max-neutral
   -inf in the product-space intermediate ([=:@^]), so they gate neither gradient.

   Both binary [@^+] and unary [@^^] overlapping windows are covered. The unary case also pins
   projection inference for convolution indices that occur only inside the convolution compound
   (gh-ocannl-515). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Stdio
open Verdict.Claims

(* Readbacks against their closed forms, to an absolute 1e-5: single pairs go through
   {!Verdict.p_close}, conjunctions through {!Verdict.within}, both refusing an empty pair
   (gh-ocannl-746) and naming the worst cell on a failure (gh-ocannl-1470). *)
let tol _ = 1e-5
let fa a = String.concat ~sep:" " (Array.to_list a |> List.map ~f:(fun v -> Printf.sprintf "%g" v))
let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad

let () =
  (* === 1: The issue's minimal repro — 1-D max-slide, window 2, stride 1, zero kernel, sum loss.
     Windows: (x0,x1) (x1,x2) (x2,x3) (x3,x4); argmaxes i=0, i=1, i=3, i=4. The old input-space gate
     gave [1;2;0;0;1] (i=1 over-credited with window 0, window 3's gradient lost). === *)
  Tensor.unsafe_reinitialize ();
  let xv = [| 5.; 3.; 1.; 2.; 5. |] in
  let x =
    Operation.init ~l:"owg_x" ~prec:Ir.Ops.single ~b:[] ~o:[ 5 ]
      ~f:(fun idcs -> xv.(idcs.(0)))
      ~grad_spec:Tensor.Require_grad ()
  in
  let%op y = x @^+ "o<+k; k => o" [ "k" ] (stretch 0.0) in
  Shape.set_dim k 2;
  let%op loss = y ++ "o => 0" in
  let ctx = Context.auto () in
  Train.set_materialized y.Tensor.value;
  Train.set_materialized loss.Tensor.value;
  Train.set_materialized (grad_of x);
  let ctx = Train.update_once ~output_cd_file:false ctx loss in
  let yv = Context.get_values ctx y.Tensor.value in
  let lv = Context.get_values ctx loss.Tensor.value in
  let gx = Context.get_values ctx (grad_of x) in
  printf "tropical overlap: y = [%s], loss = %s, gx = [%s]\n%!" (fa yv) (fa lv) (fa gx);
  p "tropical overlap forward"
    (Verdict.within "tropical overlap forward" yv [| 5.; 3.; 2.; 5. |] ~tol
    && Verdict.within "tropical overlap forward" lv [| 15. |] ~tol);
  p_close "tropical overlap gradient exact" gx [| 1.; 1.; 0.; 1.; 1. |] ~tol;

  (* === 1b: Unary conv specs — [k] occurs only inside [o<+k], so projection inference must register
     it while processing the compound. Pin both the issue's [++] repro and the [@^^] overlap
     gradient that motivated it. === *)
  Tensor.unsafe_reinitialize ();
  let x1b = NTDSL.ndarray xv ~label:[ "owg_x1b" ] ~output_dims:[ 5 ] () in
  let%op y1b = x1b ++ "o<+k => o" [ "k" ] in
  Shape.set_dim k 2;
  let ctx = Context.auto () in
  Train.set_materialized y1b.Tensor.value;
  let ctx = Train.forward_once ctx y1b in
  let yv1b = Context.get_values ctx y1b.Tensor.value in
  printf "unary conv sum: y = [%s]\n%!" (fa yv1b);
  p_close "unary conv sum forward" yv1b [| 8.; 4.; 3.; 7. |] ~tol;

  Tensor.unsafe_reinitialize ();
  let x1c =
    Operation.init ~l:"owg_x1c" ~prec:Ir.Ops.single ~b:[] ~o:[ 5 ]
      ~f:(fun idcs -> xv.(idcs.(0)))
      ~grad_spec:Tensor.Require_grad ()
  in
  let%op y1c = x1c @^^ "o<+k => o" [ "k" ] in
  Shape.set_dim k 2;
  let%op loss1c = y1c ++ "o => 0" in
  let ctx = Context.auto () in
  Train.set_materialized y1c.Tensor.value;
  Train.set_materialized (grad_of x1c);
  let ctx = Train.update_once ~output_cd_file:false ctx loss1c in
  let yv1c = Context.get_values ctx y1c.Tensor.value in
  let gx1c = Context.get_values ctx (grad_of x1c) in
  printf "unary einmax1 overlap: y = [%s], gx = [%s]\n%!" (fa yv1c) (fa gx1c);
  p "unary einmax1 overlap forward and gradient exact"
    (Verdict.within "unary einmax1 overlap forward" yv1c [| 5.; 3.; 2.; 5. |] ~tol
    && Verdict.within "unary einmax1 overlap gradient" gx1c [| 1.; 1.; 0.; 1.; 1. |] ~tol);

  (* === 2: einmax1 through the product-space gate — full reduction with ties: y = max x = 7, the
     gradient goes in full to every tying position. === *)
  Tensor.unsafe_reinitialize ();
  let x2 =
    Operation.init ~l:"owg_x2" ~prec:Ir.Ops.single ~b:[] ~o:[ 3 ]
      ~f:(fun idcs -> if idcs.(0) = 1 then 3. else 7.)
      ~grad_spec:Tensor.Require_grad ()
  in
  let%op loss2 = x2 @^^ "i => 0" in
  let ctx = Context.auto () in
  Train.set_materialized loss2.Tensor.value;
  Train.set_materialized (grad_of x2);
  let ctx = Train.update_once ~output_cd_file:false ctx loss2 in
  let lv2 = Context.get_values ctx loss2.Tensor.value in
  let gx2 = Context.get_values ctx (grad_of x2) in
  printf "einmax1 ties: loss = %s, gx = [%s]\n%!" (fa lv2) (fa gx2);
  p "einmax1 tie gradient gates every achieving position"
    (Verdict.within "einmax1 tie loss" lv2 [| 7. |] ~tol
    && Verdict.within "einmax1 tie gradient" gx2 [| 1.; 0.; 1. |] ~tol);

  (* === 3: Tropical matmul — RHS2 has an index (j) independent of RHS1, the formerly documented g2
     limitation. t[i,j] = max_k (a[i,k] + b[k,j]) with a = [[1,9],[5,1]], b = [[10,20],[3,4]]: t =
     [[12,21],[15,25]] (winners k=1,0,0,0), loss 73, ga = [[1,1],[2,0]], gb = [[1,2],[1,0]]. === *)
  Tensor.unsafe_reinitialize ();
  let av = [| [| 1.; 9. |]; [| 5.; 1. |] |] in
  let bv = [| [| 10.; 20. |]; [| 3.; 4. |] |] in
  let a =
    Operation.init ~l:"owg_a" ~prec:Ir.Ops.single ~b:[] ~o:[ 2; 2 ]
      ~f:(fun idcs -> av.(idcs.(0)).(idcs.(1)))
      ~grad_spec:Tensor.Require_grad ()
  in
  let b =
    Operation.init ~l:"owg_b" ~prec:Ir.Ops.single ~b:[] ~o:[ 2; 2 ]
      ~f:(fun idcs -> bv.(idcs.(0)).(idcs.(1)))
      ~grad_spec:Tensor.Require_grad ()
  in
  let%op t = a @^+ "ik; kj => ij" b in
  let%op loss3 = t ++ "ij => 0" in
  let ctx = Context.auto () in
  Train.set_materialized t.Tensor.value;
  Train.set_materialized loss3.Tensor.value;
  Train.set_materialized (grad_of a);
  Train.set_materialized (grad_of b);
  let ctx = Train.update_once ~output_cd_file:false ctx loss3 in
  let tv = Context.get_values ctx t.Tensor.value in
  let lv3 = Context.get_values ctx loss3.Tensor.value in
  let ga = Context.get_values ctx (grad_of a) in
  let gb = Context.get_values ctx (grad_of b) in
  printf "tropical matmul: t = [%s], loss = %s\n%!" (fa tv) (fa lv3);
  printf "tropical matmul: ga = [%s], gb = [%s]\n%!" (fa ga) (fa gb);
  p "tropical matmul forward"
    (Verdict.within "tropical matmul forward" tv [| 12.; 21.; 15.; 25. |] ~tol
    && Verdict.within "tropical matmul loss" lv3 [| 73. |] ~tol);
  p_close "tropical matmul g1 exact" ga [| 1.; 1.; 2.; 0. |] ~tol;
  p_close "tropical matmul g2 exact (formerly the documented limitation)" gb [| 1.; 2.; 1.; 0. |]
    ~tol;

  (* === 4: Ties with overlap — x = [7;7;7], window 2, stride 1. Both windows tie everywhere: each
     achieving pair gets the full gradient, and the shared middle position accumulates from both
     windows: gx = [1;2;1]. === *)
  Tensor.unsafe_reinitialize ();
  let x4 =
    Operation.init ~l:"owg_x4" ~prec:Ir.Ops.single ~b:[] ~o:[ 3 ]
      ~f:(fun _ -> 7.)
      ~grad_spec:Tensor.Require_grad ()
  in
  let%op y4 = x4 @^+ "o<+k; k => o" [ "k" ] (stretch 0.0) in
  Shape.set_dim k 2;
  let%op loss4 = y4 ++ "o => 0" in
  let ctx = Context.auto () in
  Train.set_materialized (grad_of x4);
  let ctx = Train.update_once ~output_cd_file:false ctx loss4 in
  let gx4 = Context.get_values ctx (grad_of x4) in
  printf "ties with overlap: gx = [%s]\n%!" (fa gx4);
  p_close "ties gate every achieving pair" gx4 [| 1.; 2.; 1. |] ~tol;

  (* === 5: Clamped padded windows (gh-ocannl-504) with overlap — stride 1, window 3, all-negative
     input [-4;-3;-2;-1]: y_o = max x[o-1..o+1] clamped to the valid range, y = [-3;-2;-1;-1];
     argmaxes x1, x2, x3, x3 give gx = [0;1;1;2]. Out-of-range pairs hold -inf in the product-space
     intermediate and gate nothing. === *)
  Tensor.unsafe_reinitialize ();
  let x5 =
    Operation.init ~l:"owg_x5" ~prec:Ir.Ops.single ~b:[] ~o:[ 4 ]
      ~f:(fun idcs -> Float.of_int idcs.(0) -. 4.)
      ~grad_spec:Tensor.Require_grad ()
  in
  let%op y5 = x5 @^+ "o=+w; w => o" [ "w" ] (stretch 0.0) in
  Shape.set_dim w 3;
  let%op loss5 = y5 ++ "o => 0" in
  let ctx = Context.auto () in
  Train.set_materialized y5.Tensor.value;
  Train.set_materialized (grad_of x5);
  let ctx = Train.update_once ~output_cd_file:false ctx loss5 in
  let yv5 = Context.get_values ctx y5.Tensor.value in
  let gx5 = Context.get_values ctx (grad_of x5) in
  printf "clamped overlap: y = [%s], gx = [%s]\n%!" (fa yv5) (fa gx5);
  p_close "clamped overlap forward" yv5 [| -3.; -2.; -1.; -1. |] ~tol;
  p_close "clamped overlap gradient exact" gx5 [| 0.; 1.; 1.; 2. |] ~tol;

  (* === 6: AlexNet-style overlapping max_pool2d (window 3, stride 2) on a 5x5 range input. Position
     (2,2)=12 lies in all four windows and wins only the first: the old input-space gate let the
     three losing windows overwrite its bit. Winners: 12@(2,2), 14@(2,4), 22@(4,2), 24@(4,4). === *)
  Tensor.unsafe_reinitialize ();
  let x6 =
    Operation.init ~l:"owg_x6" ~prec:Ir.Ops.single ~b:[] ~o:[ 5; 5; 1 ]
      ~f:(fun idcs -> Float.of_int ((5 * idcs.(0)) + idcs.(1)))
      ~grad_spec:Tensor.Require_grad ()
  in
  let pool = Nn_blocks.max_pool2d ~stride:2 ~window_size:3 () in
  let%op y6 = pool x6 in
  let%op loss6 = y6 ++ "...|... => |->0" in
  let ctx = Context.auto () in
  Train.set_materialized y6.Tensor.value;
  Train.set_materialized (grad_of x6);
  let ctx = Train.update_once ~output_cd_file:false ctx loss6 in
  let yv6 = Context.get_values ctx y6.Tensor.value in
  let gx6 = Context.get_values ctx (grad_of x6) in
  printf "overlapping max_pool2d: y = [%s]\n%!" (fa yv6);
  printf "overlapping max_pool2d: gx = [%s]\n%!" (fa gx6);
  p_close "overlapping max_pool2d forward" yv6 [| 12.; 14.; 22.; 24. |] ~tol;
  p_close "overlapping max_pool2d gradient exact" gx6
    (Array.init 25 ~f:(fun i -> if i = 12 || i = 14 || i = 22 || i = 24 then 1. else 0.))
    ~tol;

  (* === 7: All-[-inf] input under clamped padded windows with a learnable kernel — the validity
     mask. Every output is -inf, so out-of-range pairs' -inf neutral EQUALS t; without the
     [valid_pspace] conjunction the g2 scatter (whose kernel projection is valid for those pairs)
     would credit them. Valid pairs per kernel position w: w=0 misses o=0, w=2 misses o=3, so gk =
     [3;4;3]; per input position: gx = [2;3;3;2] (all valid pairs tie at -inf and each gates the
     full gradient). === *)
  Tensor.unsafe_reinitialize ();
  let x7 =
    Operation.init ~l:"owg_x7" ~prec:Ir.Ops.single ~b:[] ~o:[ 4 ]
      ~f:(fun _ -> Float.neg_infinity)
      ~grad_spec:Tensor.Require_grad ()
  in
  let k7 =
    Operation.init ~l:"owg_k7" ~prec:Ir.Ops.single ~b:[] ~o:[ 3 ]
      ~f:(fun _ -> 0.)
      ~grad_spec:Tensor.Require_grad ()
  in
  let%op y7 = x7 @^+ "o=+w; w => o" [ "w" ] k7 in
  Shape.set_dim w 3;
  let%op loss7 = y7 ++ "o => 0" in
  let ctx = Context.auto () in
  Train.set_materialized (grad_of x7);
  Train.set_materialized (grad_of k7);
  let ctx = Train.update_once ~output_cd_file:false ctx loss7 in
  let gx7 = Context.get_values ctx (grad_of x7) in
  let gk7 = Context.get_values ctx (grad_of k7) in
  printf "all -inf clamped: gx = [%s], gk = [%s]\n%!" (fa gx7) (fa gk7);
  p_close "clamped-out pairs gate no input gradient" gx7 [| 2.; 3.; 3.; 2. |] ~tol;
  p_close "clamped-out pairs gate no kernel gradient (validity mask)" gk7 [| 3.; 4.; 3. |] ~tol;

  (* === 8: Row-variable reduction keeping nonempty input axes — einmax1 @^^ "...|... => ...|0" on a
     2->3 matrix-shaped tensor. The reduced output row (extent 3) lands after the kept input row
     (extent 2) in the proxy layout while the product order has them the other way around;
     [Indexing.prod_project_for]'s extent-based (first-fit) pairing makes the layouts agree. p8 =
     [[1,5,3],[9,2,4]]: per-row maxima 5 (b=1) and 9 (b=0), loss 14, gp = [[0,1,0],[1,0,0]]. === *)
  Tensor.unsafe_reinitialize ();
  let p8v = [| [| 1.; 5.; 3. |]; [| 9.; 2.; 4. |] |] in
  let p8 =
    Operation.init ~l:"owg_p8" ~prec:Ir.Ops.single ~b:[] ~o:[ 2; 3 ]
      ~f:(fun idcs -> p8v.(idcs.(0)).(idcs.(1)))
      ~grad_spec:Tensor.Require_grad ()
  in
  let%op x8 = p8 ++ "ab => a->b" in
  let%op y8 = x8 @^^ "...|... => ...|0" in
  let%op loss8 = y8 ++ "...|...->... => |->0" in
  let ctx = Context.auto () in
  Train.set_materialized loss8.Tensor.value;
  Train.set_materialized (grad_of p8);
  let ctx = Train.update_once ~output_cd_file:false ctx loss8 in
  let lv8 = Context.get_values ctx loss8.Tensor.value in
  let gp8 = Context.get_values ctx (grad_of p8) in
  printf "row-var reduce with kept input axes: loss = %s, gp = [%s]\n%!" (fa lv8) (fa gp8);
  p "row-var reduction keeping input axes has exact gradients"
    (Verdict.within "row-var reduction loss" lv8 [| 14. |] ~tol
    && Verdict.within "row-var reduction gradient" gp8 [| 0.; 1.; 0.; 1.; 0.; 0. |] ~tol);

  printf "\nDone.\n%!"
