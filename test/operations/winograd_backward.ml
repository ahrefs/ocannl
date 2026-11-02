(* gh-ocannl-1548: the Winograd F(2x2, 3x3) rewrite of a conv's DATA GRADIENT ([Ir.Winograd]). The
   backward of [y = conv(x, k)] lowers to the scatter [dx[.., o + k, ..] += gy[.., o, ..] * k[..]];
   the rewrite computes it as its gather -- a 3x3 conv of [gy] extended by zero (guarded reads) with
   the kernel flipped and the channel roles exchanged -- over the same transforms as the forward,
   with the original scatter restricted to disjoint boxes on an odd extent's last row and column.

   Every executed leg compares against the SAME program with the gate off, the session reinitialized
   before each build so both mint the same nodes. The upstream gradient is fixed: [loss = sum (y *.
   r)], so [gy = r] exactly.

   Legs: 1. Exact parity per shape: padded and valid convs, even and odd extents on either axis (the
   remainder boxes, one or two on the last column), a non-square image, a batch of one (a fixed
   index), [n_o = 3], and two chained sites. Integer fixtures vary with every axis, every tap and
   both channels, the kernel asymmetric; every transform coefficient is dyadic and every partial sum
   of either form a multiple of 1/4 below 2^20, so both forms -- and a host transposed convolution
   in double precision -- agree BITWISE. Structurally: the coordinates (with multiplicity) the
   direct form writes, enumerated, are the full transposed image with the gate off and exactly its
   last odd row and column rewritten; the gradient is written from a minted product buffer; all
   three transforms are written; the pass is idempotent; a second lowering mints the same nodes. 2.
   Tolerance from Winograd's error bound on non-dyadic values (derived at the leg), with a negative
   control showing the bound would catch a missed kernel flip. 3. Hand-built scatters: a destination
   holding prior values keeps them through an odd-extent site; the swapped operand order and the
   [fma] form are rewritten; an extent-one source loop (the empty remainder box) executes; the
   rewritten contraction is a matmul site; non-separable, 5x5, f16 and aliased scatters are left
   alone. 4. A spatial extent of one is lowered as a fixed index, so the [n_o = 1] conv declines --
   and stays correct. 5. Padded margins: a padded conv's input gradient, read through a padded
   window of its own (gh-ocannl-1496's pattern), and its physical buffer margins included, match the
   direct form bitwise. 6. Staleness: one compiled step, run, then the kernel AND the upstream
   gradient replaced through [Context.set_values], run again. 7. The tuned path: [Autotune.tune]
   over a rewritten step seeds sketches and matches the direct form within the bound. *)

(* cc-fp-contract: exact-operands -- bounded integer fixtures with dyadic transforms, every sum
   exact in f32; the tolerance leg's bound holds under any contraction (an FMA only removes
   roundings) *)

open Base
open Stdio
open Ocannl
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Idx = Ir.Indexing
module Winograd = Ir.Winograd

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad

(* {1 Fixtures} *)

let ints ~weights ~m ~off idcs =
  Ll_test.weighted ~weights ~modulus:m ~offset:(Float.of_int (-off)) ~stride:1. idcs

(* Non-dyadic values in [-0.7, 0.7286]: [((sum_i w_i * idx_i) mod 11 - 4.9) / 7]. *)
let reals ~weights idcs =
  Ll_test.weighted ~weights ~modulus:11 ~offset:(-4.9) ~stride:(1. /. 7.) idcs

let x_f = ints ~weights:[| 7; 1; 3; 5 |] ~m:9 ~off:4
let r_ints = ints ~weights:[| 5; 3; 1; 2 |] ~m:9 ~off:4

(* Asymmetric in both taps (weights 3 and 2 differ, neither divides 7), so an unflipped or
   transposed kernel changes some cell. *)
let k_ints = ints ~weights:[| 1; 3; 2; 5 |] ~m:7 ~off:3

type shape = {
  name : string;
  b : int;  (** The batch extent: one batch axis. *)
  h : int;  (** Input extent. *)
  w : int;
  ic : int;
  oc : int;
  padded : bool;
}

let out_extent s n = if s.padded then n else n - 2

let spec ~padded =
  let m = if padded then "=+" else "<+" in
  Printf.sprintf "...| oh%skh, ow%skw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." m m

let const ~l ~b ~o ~f =
  let dims = Array.of_list (b @ o) in
  let nd = Ir.Ndarray.init_array ~debug:l Ir.Ops.single ~dims ~padding:None ~f in
  NTDSL.reshape ~l ~b ~o nd ()

let kernel ~l ~ic ~oc ~f () =
  let nd = Ir.Ndarray.init_array ~debug:l Ir.Ops.single ~dims:[| oc; 3; 3; ic |] ~padding:None ~f in
  NTDSL.reshape ~l ~i:[ 3; 3; ic ] ~o:[ oc ] nd ()

(* The leaf [x0] carries the gradient; a padded conv reads a computed copy, whose layout is not
   committed at creation, so the conv can grant it margins (padding_lifecycle). *)
let leaf ~l (s : shape) =
  let x0 =
    Operation.init ~l ~prec:Ir.Ops.single ~b:[ s.b ] ~o:[ s.h; s.w; s.ic ] ~f:x_f
      ~grad_spec:Tensor.Require_grad ()
  in
  let x =
    if s.padded then TDSL.O.einsum1 ~label:[ l ^ "c" ] "...|h, w, c => ...|h, w, c" x0 else x0
  in
  (x0, x)

let conv ~padded ~label x k = TDSL.O.einsum ~label:[ label ] (spec ~padded) x k

(* {1 Host oracle} *)

(* The transposed convolution [g[b, i, j, c] = sum over (oh, ow, o, kh, kw) with oh + kh - pad = i,
   ow + kw - pad = j of r[b, oh, ow, o] * k[o, kh, kw, c]], in double precision. *)
let host_grad ?(flip = false) (s : shape) ~f_r ~f_k =
  let pad = if s.padded then 1 else 0 in
  let oh_n = out_extent s s.h and ow_n = out_extent s s.w in
  let g = Array.create ~len:(s.b * s.h * s.w * s.ic) 0. in
  for b = 0 to s.b - 1 do
    for oh = 0 to oh_n - 1 do
      for ow = 0 to ow_n - 1 do
        for o = 0 to s.oc - 1 do
          let rv = f_r [| b; oh; ow; o |] in
          for kh = 0 to 2 do
            for kw = 0 to 2 do
              let i = oh + kh - pad and j = ow + kw - pad in
              if i >= 0 && i < s.h && j >= 0 && j < s.w then
                for c = 0 to s.ic - 1 do
                  let kv =
                    if flip then f_k [| o; 2 - kh; 2 - kw; c |] else f_k [| o; kh; kw; c |]
                  in
                  let at = (((((b * s.h) + i) * s.w) + j) * s.ic) + c in
                  g.(at) <- g.(at) +. (rv *. kv)
                done
            done
          done
        done
      done
    done
  done;
  g

(* {1 Structure} *)

let written (llc : LL.t) =
  LL.affine_accesses llc
  |> List.filter_map ~f:(fun (a : Tn.t Ir.Affine.access) -> Option.some_if a.a_write a.a_tn)

let minted_of llc =
  List.filter (written llc) ~f:(fun tn -> Option.is_some (Winograd.minted_role tn))
  |> List.dedup_and_sort ~compare:Tn.compare

let roles_written llc =
  List.filter_map (written llc) ~f:Winograd.minted_role |> List.dedup_and_sort ~compare:Poly.compare

let all_roles = List.dedup_and_sort ~compare:Poly.compare [ `Kernel; `Input; `Product ]

let rec reads ~pred (sc : LL.scalar_t) =
  match sc with
  | LL.Get (tn, _) -> pred tn
  | LL.Binop (_, (a, _), (b, _)) -> reads ~pred a || reads ~pred b
  | LL.Ternop (_, (a, _), (b, _), (c, _)) -> reads ~pred a || reads ~pred b || reads ~pred c
  | LL.Unop (_, (a, _)) -> reads ~pred a
  | _ -> false

(* The coordinates of [d] the DIRECT form writes, with multiplicity: every perfect nest whose [Set]
   writes [d] reading [kern], enumerated over its loop ranges (the fixtures are small). *)
let direct_writes (llc : LL.t) ~(d : Tn.t) ~(kern : Tn.t) =
  let out = Hashtbl.Poly.create () in
  let eval env (idx : Idx.axis_index) =
    match idx with
    | Idx.Fixed_idx i -> i
    | Idx.Iterator s -> List.Assoc.find_exn env s ~equal:Idx.equal_symbol
    | Idx.Affine { symbols; offset } ->
        List.fold symbols ~init:offset ~f:(fun acc (c, s) ->
            acc + (c * List.Assoc.find_exn env s ~equal:Idx.equal_symbol))
    | _ -> failwith "direct_writes: unexpected index"
  in
  let rec go env (llc : LL.t) =
    match llc with
    | LL.Seq (a, b) ->
        go env a;
        go env b
    | LL.For_loop { index; from_; to_; body; _ } ->
        for v = from_ to to_ do
          go ((index, v) :: env) body
        done
    | LL.Set { tn; idcs; llsc; _ } when Tn.equal tn d && reads ~pred:(Tn.equal kern) llsc ->
        Hashtbl.incr out (Array.to_list (Array.map idcs ~f:(eval env)))
    | _ -> ()
  in
  go [] llc;
  Hashtbl.to_alist out |> List.sort ~compare:Poly.compare

(* Whether [d] is written by a statement reading a minted product buffer: the inverse transform. *)
let written_from_product (llc : LL.t) ~(d : Tn.t) =
  let rec go (llc : LL.t) =
    match llc with
    | LL.Seq (a, b) -> go a || go b
    | LL.For_loop { body; _ } -> go body
    | LL.Set { tn; llsc; _ } ->
        Tn.equal tn d
        && reads ~pred:(fun tn -> Poly.equal (Winograd.minted_role tn) (Some `Product)) llsc
    | _ -> false
  in
  go llc

let lower name (comp : Ir.Assignments.comp) =
  Ir.Assignments.lower (LL.empty_optimize_ctx ()) ~unoptim_ll_source:None ~ll_source:None
    ~cd_source:None ~name [] comp.Ir.Assignments.asgns

(* {1 Runs} *)

type run = {
  grad : float array;  (** [x0]'s gradient. *)
  raw : LL.t;  (** The step's raw lowering, ahead of the rewrite tier. *)
  opt : LL.optimized;
  dx : Tn.t;  (** The gradient the conv's backward writes (the conv input's). *)
  kern : Tn.t;
  minted_again : bool;
}

(* Build [build ()] -- returning the loss, the leaf, the conv's input and its kernel -- and run one
   gradient step with the rewrite [on] or off. *)
let step ~on build =
  Tensor.unsafe_reinitialize ();
  Winograd.set_enabled (Some on);
  let loss, x0, x, k = build () in
  Train.set_materialized (grad_of x0);
  let comp = Train.grad_update loss in
  let ctx = Train.run_once ~f:(fun _ -> comp) (Context.auto ()) loss in
  let grad = Context.get_values ctx (grad_of x0) in
  let raw = Ir.Assignments.to_low_level comp.Ir.Assignments.asgns in
  let opt = lower "winob_probe" comp in
  let minted_again =
    List.equal Tn.equal (minted_of opt.LL.llc) (minted_of (lower "winob_probe" comp).LL.llc)
  in
  Winograd.set_enabled None;
  { grad; raw; opt; dx = grad_of x; kern = k.Tensor.value; minted_again }

let report label (got : float array) =
  eprintf "%s: %s (not part of the golden)\n%!" label
    (String.concat ~sep:" "
       (Array.to_list
          (Array.map
             (Array.sub got ~pos:0 ~len:(Int.min 8 (Array.length got)))
             ~f:(fun v -> Printf.sprintf "%.9g" v))))

let single ~f_r ~f_k (s : shape) () =
  let x0, x = leaf ~l:(s.name ^ "x") s in
  let k = kernel ~l:(s.name ^ "k") ~ic:s.ic ~oc:s.oc ~f:f_k () in
  let y = conv ~padded:s.padded ~label:(s.name ^ "y") x k in
  let r =
    const ~l:(s.name ^ "r") ~b:[ s.b ] ~o:[ out_extent s s.h; out_extent s s.w; s.oc ] ~f:f_r
  in
  let%op loss = (y *. r) ++ "...|... => |->0" in
  (loss, x0, x, k)

(* {1 Leg 1: exact parity and structure, per shape} *)

let leg1 (s : shape) =
  let oh = out_extent s s.h and ow = out_extent s s.w in
  printf "--- leg 1 %s: %s %dx%d (gather extent %dx%d), batch %d, %d -> %d channels ---\n" s.name
    (if s.padded then "padded" else "valid")
    s.h s.w (oh + 2) (ow + 2) s.b s.ic s.oc;
  let build = single ~f_r:r_ints ~f_k:k_ints s in
  let direct = step ~on:false build in
  let wino = step ~on:true build in
  report (s.name ^ " direct") direct.grad;
  report (s.name ^ " winograd") wino.grad;
  let host = host_grad s ~f_r:r_ints ~f_k:k_ints in
  let nh = oh + 2 and nw = ow + 2 in
  let full = direct_writes direct.raw ~d:direct.dx ~kern:direct.kern in
  let rewritten = Winograd.rewrite direct.raw in
  (* The spatial axes of [dx] follow its one batch axis. *)
  let on_remainder coord =
    match coord with
    | _ :: i :: j :: _ -> (nh % 2 = 1 && i = nh - 1) || (nw % 2 = 1 && j = nw - 1)
    | _ -> false
  in
  p
    (s.name ^ ": the direct scatter writes every gather row and column")
    (List.length
       (List.dedup_and_sort ~compare:Poly.compare
          (List.map full ~f:(fun (c, _) ->
               match c with _ :: i :: j :: _ -> (i, j) | _ -> (-1, -1))))
    = nh * nw);
  p_all (s.name ^ ": each cell the transposed image's count of times") full ~f:(fun (c, n) ->
      match c with
      | _ :: i :: j :: _ ->
          let taps l n_o =
            (* The (o, k) pairs with o + k = l, o in [0, n_o), k in [0, 3). *)
            List.count (List.range 0 3) ~f:(fun k -> l - k >= 0 && l - k < n_o)
          in
          n = taps i oh * taps j ow * s.oc
      | _ -> false);
  p
    (s.name ^ ": rewritten, the direct form writes exactly the scatter's last odd row and column")
    (List.equal Poly.equal
       (direct_writes rewritten ~d:direct.dx ~kern:direct.kern)
       (List.filter full ~f:(fun (c, _) -> on_remainder c)));
  p (s.name ^ ": the pass changes the raw lowering") (not (LL.equal rewritten direct.raw));
  p (s.name ^ ": the pass is idempotent") (LL.equal (Winograd.rewrite rewritten) rewritten);
  p (s.name ^ ": a second lowering mints the same nodes") wino.minted_again;
  p_none (s.name ^ ": the gate off mints nothing") (written direct.opt.LL.llc) ~f:(fun tn ->
      Option.is_some (Winograd.minted_role tn));
  p
    (s.name ^ ": the gate on writes the kernel, input and product transforms")
    (List.equal Poly.equal (roles_written wino.opt.LL.llc) all_roles);
  p
    (s.name ^ ": the conv input's gradient is written from a minted product buffer")
    (written_from_product wino.opt.LL.llc ~d:wino.dx);
  p_all2
    (s.name ^ ": the direct gradient is the host transposed convolution, bitwise")
    direct.grad host ~f:Float.equal;
  p_all2
    (s.name ^ ": the Winograd gradient matches the direct form bitwise")
    wino.grad direct.grad ~f:Float.equal

let shapes =
  [
    { name = "bpad"; b = 2; h = 6; w = 6; ic = 3; oc = 4; padded = true };
    { name = "bval"; b = 2; h = 8; w = 10; ic = 2; oc = 3; padded = false };
    { name = "bodd"; b = 2; h = 7; w = 9; ic = 3; oc = 2; padded = false };
    { name = "boddw"; b = 2; h = 8; w = 7; ic = 2; oc = 3; padded = false };
    { name = "brect"; b = 3; h = 5; w = 8; ic = 5; oc = 2; padded = true };
    { name = "bb1"; b = 1; h = 6; w = 4; ic = 2; oc = 3; padded = true };
    { name = "bn3"; b = 2; h = 5; w = 5; ic = 2; oc = 2; padded = false };
  ]

let () =
  eprintf "backend: %s (not part of the golden)\n%!" backend_name;
  List.iter shapes ~f:leg1;
  printf "--- leg 1 bchain: two chained conv sites ---\n";
  let build () =
    let s = { name = "bch"; b = 2; h = 6; w = 6; ic = 2; oc = 3; padded = true } in
    let x0, x = leaf ~l:"bchx" s in
    let k1 = kernel ~l:"bch1k" ~ic:2 ~oc:3 ~f:k_ints () in
    let k2 = kernel ~l:"bch2k" ~ic:3 ~oc:2 ~f:(ints ~weights:[| 2; 1; 3; 1 |] ~m:5 ~off:2) () in
    let y1 = conv ~padded:true ~label:"bch1y" x k1 in
    let y2 = conv ~padded:false ~label:"bch2y" y1 k2 in
    let r = const ~l:"bchr" ~b:[ 2 ] ~o:[ 4; 4; 2 ] ~f:r_ints in
    let%op loss = (y2 *. r) ++ "...|... => |->0" in
    (loss, x0, x, k1)
  in
  let direct = step ~on:false build in
  let wino = step ~on:true build in
  p_empty "bchain: the first conv's data-gradient scatter is gone"
    ~over:(direct_writes direct.raw ~d:direct.dx ~kern:direct.kern)
    (direct_writes wino.opt.LL.llc ~d:wino.dx ~kern:wino.kern);
  p "bchain: four sites' worth of transforms are written (12 distinct minted buffers)"
    (List.length (minted_of wino.opt.LL.llc) = 12);
  p_all2 "bchain: the Winograd gradient matches the direct form bitwise" wino.grad direct.grad
    ~f:Float.equal

(* {1 Leg 2: tolerance from the error bound}

   Both forms compute a sum of products of the inputs. Expanded into monomials, each carries at most
   [N] rounding factors [(1 + delta)], [|delta| <= u = 2^-24], so the computed value is within
   [gamma_N * S] of the exact one, where [gamma_N = N u / (1 - N u)] and [S] is the same computation
   over absolute values (Higham, Accuracy and Stability of Numerical Algorithms, ch. 3). An FMA
   removes a rounding, and any order of a [C]-long sum has depth at most [C], so the bound holds
   under contraction and reassociation alike; it assumes finite values clear of underflow.

   Direct: [9C] products accumulated, [N <= 9C + 1], [S <= 9 C mg mk]. Winograd: the kernel
   transform sums at most 9 terms (scalings by 1/2 and 1/4 counted too), the input transform 4, one
   multiply, the [C]-long contraction, the inverse transform 9 terms into the destination: [N <= C +
   22]; and [S <= 64 C mg mk], since per axis [sum_xi |A^T[o][xi]| * |G[xi]|_1 * |B^T[xi]|_1 = 8]
   for both output positions (G's row sums 1, 3/2, 3/2, 1; every row of B^T sums to 2). So the two
   computed gradients differ by at most [(gamma_{C+22} 64 + gamma_{9C+1} 9) C mg mk] per cell. *)

let gamma n =
  let nu = Float.of_int n *. Float.ldexp 1. (-24) in
  nu /. (1. -. nu)

(* The greatest magnitude [f] takes over the index box [dims]. *)
let max_abs ~dims f =
  let rec go prefix = function
    | [] -> Float.abs (f (Array.of_list (List.rev prefix)))
    | d :: rest ->
        List.fold (List.range 0 d) ~init:0. ~f:(fun acc i -> Float.max acc (go (i :: prefix) rest))
  in
  go [] dims

(* The per-cell bound on the two forms' difference, from the fixture's own maxima. *)
let bound (s : shape) ~f_r ~f_k =
  let c = s.oc in
  let mg = max_abs ~dims:[ s.b; out_extent s s.h; out_extent s s.w; s.oc ] f_r in
  let mk = max_abs ~dims:[ s.oc; 3; 3; s.ic ] f_k in
  ((gamma (c + 22) *. 64.) +. (gamma ((9 * c) + 1) *. 9.)) *. Float.of_int c *. mg *. mk

let () =
  let s = { name = "breal"; b = 2; h = 9; w = 8; ic = 6; oc = 5; padded = true } in
  let f_r = reals ~weights:[| 3; 1; 4; 7 |] and f_k = reals ~weights:[| 2; 5; 1; 3 |] in
  let tol = bound s ~f_r ~f_k in
  printf "--- leg 2: non-dyadic inputs, within the error bound ---\n";
  let direct = step ~on:false (single ~f_r ~f_k s) in
  let wino = step ~on:true (single ~f_r ~f_k s) in
  report "breal direct" direct.grad;
  report "breal winograd" wino.grad;
  let worst =
    Array.fold2_exn wino.grad direct.grad ~init:0. ~f:(fun acc a b ->
        Float.max acc (Float.abs (a -. b)))
  in
  eprintf "breal: worst absolute difference %.3g, bound %.3g (not part of the golden)\n%!" worst tol;
  p "breal: the rewrite fired" (List.equal Poly.equal (roles_written wino.opt.LL.llc) all_roles);
  p_close "breal: the Winograd gradient is within the error bound of the direct form" wino.grad
    direct.grad ~tol:(fun _ -> tol);
  let host = host_grad s ~f_r ~f_k and unflipped = host_grad ~flip:true s ~f_r ~f_k in
  p_exists "breal: negative control -- an unflipped kernel's gradient leaves the bound by 10x"
    (List.init (Array.length host) ~f:Fn.id)
    ~f:(fun i -> Float.(abs (host.(i) -. unflipped.(i)) > 10. *. tol))

(* {1 Leg 3: hand-built scatters} *)

let node id dims label = Ll_test.node_factory ~first_id:id ~dims () label

(* [d[b, oh + kh, ow + kw, ic] += g[b, oh, ow, oc] * k[oc, kh, kw, ic]], in the given form. *)
let scatter ?(form = `Gk) ?(window = 3) ?(extra = fun ~kh:_ -> [||]) ~n_oh ~n_ow ~n_b ~n_ic ~n_oc ~g
    ~k ~d () =
  let module B = Ll_test in
  let b = B.sym () and oh = B.sym () and ow = B.sym () and oc = B.sym () and ic = B.sym () in
  let kh = B.sym () and kw = B.sym () in
  let di = [| B.iter b; B.aff [ (1, oh); (1, kh) ] 0; B.aff [ (1, ow); (1, kw) ] 0; B.iter ic |] in
  let gv = B.get g (Array.append [| B.iter b; B.iter oh; B.iter ow; B.iter oc |] (extra ~kh)) in
  let kv = B.get k [| B.iter oc; B.iter kh; B.iter kw; B.iter ic |] in
  let rhs =
    match form with
    | `Gk -> B.add (B.get d di) (B.mul gv kv)
    | `Kg_first -> B.add (B.mul kv gv) (B.get d di)
    | `Fma -> LL.Ternop (Ir.Ops.FMA, (gv, B.single), (kv, B.single), (B.get d di, B.single))
  in
  B.loop_n b n_b
    (B.loop_n oh n_oh
       (B.loop_n ow n_ow
          (B.loop_n oc n_oc
             (B.loop_n ic n_ic (B.loop_n kh window (B.loop_n kw window (B.set d di rhs)))))))

let () =
  printf "--- leg 3: hand-built scatters ---\n";
  let module B = Ll_test in
  let n_b = 2 and n_ic = 2 and n_oc = 3 in
  let case ~id ~n_oh ~n_ow ~form ~what =
    let g = node id [| n_b; n_oh; n_ow; n_oc |] (what ^ "g")
    and k = node (id + 1) [| n_oc; 3; 3; n_ic |] (what ^ "k")
    and d = node (id + 2) [| n_b; n_oh + 2; n_ow + 2; n_ic |] (what ^ "d") in
    List.iter [ g; k; d ] ~f:B.materialize;
    let llc = scatter ~form ~n_oh ~n_ow ~n_b ~n_ic ~n_oc ~g ~k ~d () in
    let rewritten = Winograd.rewrite llc in
    p (what ^ ": rewritten") (not (LL.equal rewritten llc));
    let seed =
      [
        ( g,
          Array.init
            (n_b * n_oh * n_ow * n_oc)
            ~f:
              (Ll_test.cycle_flat ~radix:4 ~dims:[| n_b; n_oh; n_ow; n_oc |] ~modulus:9
                 ~offset:(-4.) ~stride:1.) );
        ( k,
          Array.init
            (n_oc * 9 * n_ic)
            ~f:
              (Ll_test.cycle_flat ~radix:3 ~dims:[| n_oc; 3; 3; n_ic |] ~modulus:7 ~offset:(-3.)
                 ~stride:1.) );
        (* The destination's prior contents: nonzero and varying, so an overwrite shows. *)
        ( d,
          Array.init
            (n_b * (n_oh + 2) * (n_ow + 2) * n_ic)
            ~f:
              (Ll_test.cycle_flat ~radix:2
                 ~dims:[| n_b; n_oh + 2; n_ow + 2; n_ic |]
                 ~modulus:5 ~offset:10. ~stride:1.) );
      ]
    in
    let run name llc =
      let _o, got = B.optimize_and_execute ~name llc ~seed ~read:[ d ] in
      List.hd_exn got
    in
    let want = run (what ^ "_direct") llc in
    let got = run (what ^ "_rewritten") rewritten in
    p_all2 (what ^ ": adds onto the destination's prior contents, bitwise") got want ~f:Float.equal;
    rewritten
  in
  let r = case ~id:51400 ~n_oh:3 ~n_ow:5 ~form:`Gk ~what:"wbodd" in
  p "wbodd: the rewritten contraction is a matmul site accumulating the product buffer"
    (match Autotune.detect_matmul r with
    | Some m -> Option.is_some (Winograd.minted_role m.Autotune.m_d)
    | None -> false);
  ignore (case ~id:51410 ~n_oh:4 ~n_ow:4 ~form:`Kg_first ~what:"wbswap" : LL.t);
  ignore (case ~id:51420 ~n_oh:4 ~n_ow:3 ~form:`Fma ~what:"wbfma" : LL.t);
  (* An extent-one source loop on an odd gather extent on both axes: the last column's box over the
     rows above the last row, [oh <= n_oh - 2], is empty. *)
  ignore (case ~id:51430 ~n_oh:1 ~n_ow:3 ~form:`Gk ~what:"wbone" : LL.t);
  (* Declines. *)
  let g = node 51440 [| n_b; 4; 4; n_oc |] "wbdg"
  and g5 = node 51441 [| n_b; 4; 4; n_oc; 3 |] "wbdg5"
  and k = node 51442 [| n_oc; 3; 3; n_ic |] "wbdk"
  and k5 = node 51443 [| n_oc; 5; 5; n_ic |] "wbdk5"
  and k16 =
    Ll_test.node_factory ~prec:Ir.Ops.half ~first_id:51444 ~dims:[| n_oc; 3; 3; n_ic |] () "wbdkh"
  and d = node 51445 [| n_b; 8; 8; n_ic |] "wbdd" in
  let declined what llc = p (what ^ ": left alone") (LL.equal (Winograd.rewrite llc) llc) in
  declined "a scatter whose upstream gradient also reads the window symbol (not separable)"
    (scatter ~extra:(fun ~kh -> [| B.iter kh |]) ~n_oh:4 ~n_ow:4 ~n_b ~n_ic ~n_oc ~g:g5 ~k ~d ());
  declined "a 5x5 scatter" (scatter ~window:5 ~n_oh:4 ~n_ow:4 ~n_b ~n_ic ~n_oc ~g ~k:k5 ~d ());
  declined "an f16 kernel" (scatter ~n_oh:4 ~n_ow:4 ~n_b ~n_ic ~n_oc ~g ~k:k16 ~d ());
  declined "a scatter reading its own destination as the upstream gradient"
    (scatter ~n_oh:4 ~n_ow:4 ~n_b ~n_ic ~n_oc ~g:d ~k ~d ())

(* {1 Leg 4: a spatial extent of one} *)

let () =
  printf "--- leg 4: a 3x3 valid input (n_o = 1) declines ---\n";
  let s = { name = "bone"; b = 2; h = 3; w = 3; ic = 2; oc = 3; padded = false } in
  let build = single ~f_r:r_ints ~f_k:k_ints s in
  let direct = step ~on:false build in
  let wino = step ~on:true build in
  p_empty "bone: nothing minted -- the singleton source axis is a fixed index, so no o + k axis"
    ~over:(written wino.opt.LL.llc) (minted_of wino.opt.LL.llc);
  p_all2 "bone: the gradient is the host transposed convolution, bitwise" wino.grad
    (host_grad s ~f_r:r_ints ~f_k:k_ints)
    ~f:Float.equal;
  p_all2 "bone: and matches the gate off bitwise" wino.grad direct.grad ~f:Float.equal

(* {1 Leg 5: padded margins} *)

let () =
  printf "--- leg 5: a padded reader of the gradient, and its physical margins ---\n";
  let s = { name = "bmar"; b = 2; h = 6; w = 5; ic = 2; oc = 3; padded = true } in
  let run ~on =
    Tensor.unsafe_reinitialize ();
    Winograd.set_enabled (Some on);
    let loss, x0, x, k = single ~f_r:r_ints ~f_k:k_ints s () in
    let gx = grad_of x in
    let%cd smooth =
      { sm }
      =:+ x.grad
          * k ~logic:"...| oh=+kh, ow=+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.."
    in
    Train.set_materialized (grad_of x0);
    Train.set_materialized gx;
    Train.set_materialized sm.Tensor.value;
    let ctx = Train.update_once (Context.auto ()) loss in
    let ctx, routine = Context.compile ~name:"winob_smooth" ctx smooth Idx.Empty in
    let ctx = Context.run ctx routine in
    let padded = Option.is_some (Tn.get_padding gx) in
    let physical = Ir.Ndarray.retrieve_flat_values (Context.to_host ctx gx) in
    let smoothed = Context.get_values ctx sm.Tensor.value in
    let minted = List.length (minted_of (lower "winob_mar" (Train.grad_update loss)).LL.llc) in
    Winograd.set_enabled None;
    (padded, physical, smoothed, minted)
  in
  let pd, phys_d, sm_d, _ = run ~on:false in
  let pw, phys_w, sm_w, minted = run ~on:true in
  p "the conv input's gradient carries margins in both runs" (pd && pw);
  p "the gate on rewrites the step's conv (forward and data gradient: 6 minted buffers)" (minted = 6);
  p_all2 "the gradient's physical buffer, margins included, matches the direct form bitwise" phys_w
    phys_d ~f:Float.equal;
  p_all2 "a padded window reading the gradient matches the direct form bitwise" sm_w sm_d
    ~f:Float.equal

(* {1 Leg 6: staleness} *)

let () =
  printf "--- leg 6: a changed kernel and upstream gradient are seen by the next call ---\n";
  let s = { name = "bst"; b = 2; h = 6; w = 6; ic = 2; oc = 3; padded = true } in
  let run_twice ~on =
    Tensor.unsafe_reinitialize ();
    Winograd.set_enabled (Some on);
    let x0, x = leaf ~l:"bstx" s in
    let k = kernel ~l:"bstk" ~ic:2 ~oc:3 ~f:k_ints () in
    let y = conv ~padded:true ~label:"bsty" x k in
    let r = const ~l:"bstr" ~b:[ 2 ] ~o:[ 6; 6; 3 ] ~f:r_ints in
    let%op loss = (y *. r) ++ "...|... => |->0" in
    Train.set_materialized (grad_of x0);
    let ctx = Context.auto () in
    let ctx, routine = Context.compile ~name:"winob_stale" ctx (Train.grad_update loss) Idx.Empty in
    let ctx = Context.run ctx routine in
    let first = Context.get_values ctx (grad_of x0) in
    let k2 =
      Array.init
        (3 * 3 * 3 * 2)
        ~f:
          (Ll_test.cycle_flat ~radix:3 ~dims:[| 3; 3; 3; 2 |] ~modulus:7 ~offset:(-3.) ~stride:(-1.))
    in
    let r2 =
      Array.init
        (2 * 6 * 6 * 3)
        ~f:(Ll_test.cycle_flat ~radix:2 ~dims:[| 2; 6; 6; 3 |] ~modulus:5 ~offset:(-2.) ~stride:1.)
    in
    let ctx = Context.set_values ctx k.Tensor.value k2 in
    let ctx = Context.set_values ctx r.Tensor.value r2 in
    let ctx = Context.run ctx routine in
    let second = Context.get_values ctx (grad_of x0) in
    Winograd.set_enabled None;
    (first, second)
  in
  let d1, d2 = run_twice ~on:false in
  let w1, w2 = run_twice ~on:true in
  p "the replaced kernel and upstream gradient change the direct gradient"
    (not (Array.equal Float.equal d1 d2));
  p_all2 "first call: the Winograd gradient matches the direct form bitwise" w1 d1 ~f:Float.equal;
  p_all2 "second call, new kernel and upstream gradient: the Winograd gradient matches bitwise" w2
    d2 ~f:Float.equal

(* {1 Leg 7: the tuned path} *)

let () =
  printf "--- leg 7: the tuner schedules a rewritten step ---\n";
  (* A valid 16x16 conv: the data gradient's gather extent is 16, 8 tiles a side, and every
     contraction extent a multiple of 8, so a GPU with 8x8x8 matrix tiles seeds its sketches. *)
  let s = { name = "btu"; b = 2; h = 16; w = 16; ic = 8; oc = 8; padded = false } in
  let f_r = reals ~weights:[| 3; 1; 4; 7 |] and f_k = reals ~weights:[| 2; 5; 1; 3 |] in
  let tol = bound s ~f_r ~f_k in
  let want = (step ~on:false (single ~f_r ~f_k s)).grad in
  Tensor.unsafe_reinitialize ();
  Winograd.set_enabled (Some true);
  let loss, x0, _x, _k = single ~f_r ~f_k s () in
  Train.set_materialized (grad_of x0);
  let cache_dir = "autotune_cache_winograd_backward" in
  if Stdlib.Sys.file_exists cache_dir && Stdlib.Sys.is_directory cache_dir then
    Array.iter (Stdlib.Sys.readdir cache_dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat cache_dir f));
  let reports = ref [] in
  let ctx, routine =
    Autotune.tune ~name:"winob_tuned" ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir
      ~report:(fun r -> reports := r :: !reports)
      (Context.auto ()) (Train.grad_update loss) Idx.Empty
  in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx (grad_of x0) in
  Winograd.set_enabled None;
  p_exists "the search seeded sketches (whole routine or per fission segment)" !reports ~f:(fun r ->
      r.Autotune.sketch_candidates + r.Autotune.fiss_sketch_candidates > 0);
  p_close "the tuned Winograd gradient is within the error bound of the direct form" got want
    ~tol:(fun _ -> tol)
