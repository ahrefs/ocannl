(* The prefix-scan rewrite (gh-ocannl-1758): a reduction whose operand is masked by a triangle along
   one axis -- the cumulative sum written as a lower-triangular selector contraction, tensor puzzle
   7 -- becomes one [Scan_loop] per row. It changes how a value is computed, so every leg compares
   the program with the member ON against the same program with it OFF (the materialized-mask
   reduction), executed, and against a host oracle where the values are exact:

   1. puzzle 7's form under a training step: the forward is a [Forward] scan, the cumsum's backward
   (a suffix sum, with a second contribution into the same gradient) a [Backward] scan; the [n, n]
   selector is no longer written; sibling lowerings share the analysis cache; 2. frontend variants:
   rows, a strict and a reversed triangle, the [where]-selected sum, cummax through [@^^]
   (negative-only, repeated maxima, an empty selected prefix) and the tropical [@^+]; 3. hand-built
   raw lowerings: [Min], the complemented [where], an extent of two, both loop orders, rows; 4. what
   the recognizer leaves alone, each pinned by structural equality of the raw lowering; 5. the tier
   contract: idempotence, and the documented outcome of an infinite payload; 6. the cummax backward
   (gh-ocannl-1858) under a training step -- the gradient gate as a reverse scan, against the
   composed form and the tie rule written out on the host, over repeated, separated, equal,
   monotone, strict-prefix and suffix inputs; the tropical gate left composed; 7. hand-built gates
   over rows: cummin, a strict and a suffix gate, and the near misses, executed; 8. masks in host
   constants: [ndarray] literals in four orientations proven from their data, a near-triangle left
   alone, and an [NTDSL.init] mask left alone, whose data a host write changes after compiling. *)

open Base
open Stdio
module Train = Ocannl.Train
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Idx = Ir.Indexing
module Prefix_scan = Ir.Prefix_scan

let backend_name = Utils.get_global_arg ~arg_name:"backend" ~default:""
let fp_licences = Ir.Low_level.Fp_licence.(fields (of_config ()))

(* Above the constant-fill limit (16), so inputs are host-initialized data, and distinct from every
   other extent here: a node with two axes of this extent is the selector or a masked operand. *)
let n = 21
let rows = 3

(* {1 Probes} *)

let scans_where ~f (llc : LL.t) =
  Ll_test.count_stmt llc ~f:(function LL.Scan_loop { direction; _ } -> f direction | _ -> false)

let scans llc = scans_where llc ~f:(fun _ -> true)
let forward_scans llc = scans_where llc ~f:(function LL.Forward -> true | LL.Backward -> false)
let backward_scans llc = scans_where llc ~f:(function LL.Backward -> true | LL.Forward -> false)

(* The nodes the code writes that have two axes of extent [n]. *)
let square_writes (llc : LL.t) =
  LL.affine_accesses llc
  |> List.filter_map ~f:(fun (a : Tn.t Ir.Affine.access) -> Option.some_if a.a_write a.a_tn)
  |> Set.of_list (module Tn)
  |> Set.filter ~f:(fun tn -> Array.count (Lazy.force tn.Tn.dims) ~f:(fun d -> d = n) >= 2)

let report label (got : float array) =
  eprintf "%s: %s (not part of the golden)\n%!" label
    (String.concat ~sep:" " (Array.to_list (Array.map got ~f:(Printf.sprintf "%.9g"))))

(* Equality of two results, for the pairwise-distinctness claim of leg 0 (never a claim itself). *)
let same_floats a b = Array.length a = Array.length b && Array.for_all2_exn a b ~f:Float.equal
let close_tol ~tol w = Float.(tol *. max 1. (abs w))

(* {1 Host oracles} *)

let prefix ?(inclusive = true) op fill xs =
  let s = ref fill in
  Array.map xs ~f:(fun x ->
      if inclusive then (
        s := op !s x;
        !s)
      else
        let before = !s in
        s := op !s x;
        before)

let suffix ?inclusive op fill xs = Array.rev (prefix ?inclusive op fill (Array.rev xs))
let cumsum ?inclusive xs = prefix ?inclusive ( +. ) 0. xs
let cummax ?inclusive xs = prefix ?inclusive Float.max Float.neg_infinity xs

(* Small integers that vary with the position, change sign and repeat values; leg 0 derives what the
   bitwise claims rest on. *)
let ints = Array.init n ~f:(fun k -> Float.of_int ((k * 7 % 11) - 4))

(* Non-dyadic values, for the tolerance legs. *)
let reals = Array.init n ~f:(fun k -> Float.(sin (of_int k *. 1.3) *. 3.7))
let weights = Array.init n ~f:(fun o -> Float.of_int (1 + (o * 3 % 7)))
let weights2 = Array.init n ~f:(fun i -> Float.of_int (100 + i))

(* {1 Frontend tensors} *)

let input ?(grad_spec = Tensor.Prohibit_grad) label xs =
  Tensor.term_init xs ~label:[ label ] ~grad_spec ~output_dims:[ Array.length xs ] ()

(* The selector of puzzle 7, [L[i, o] = (i <= o)], from index grids. *)
let lower_triangle () =
  let ri = TDSL.range n and co = TDSL.range n in
  let%op tri = not (co ++ "j=>0j" < ri ++ "i=>i0") in
  tri

(* [L[i, o] = (i < o)]: a strict prefix. *)
let strict_triangle () =
  let ri = TDSL.range n and co = TDSL.range n in
  let%op tri = ri ++ "i=>i0" < co ++ "j=>0j" in
  tri

(* [L[i, o] = (i >= o)]: a suffix. *)
let upper_triangle () =
  let ri = TDSL.range n and co = TDSL.range n in
  let%op tri = not (ri ++ "i=>i0" < co ++ "j=>0j") in
  tri

let inspect (t : Tensor.t) : LL.optimized =
  Ir.Assignments.lower (LL.empty_optimize_ctx ()) ~unoptim_ll_source:None ~ll_source:None
    ~cd_source:None ~name:"probe" [] t.Tensor.forward.Ir.Assignments.asgns

type fwd = { values : float array; optimized : LL.optimized }

(* Run the forward of [build ()] with the rewrite [on] or off. *)
let forward ~on build =
  Tensor.unsafe_reinitialize ();
  Prefix_scan.set_enabled (Some on);
  let t = build () in
  Train.set_materialized t.Tensor.value;
  let ctx = Train.forward_once (Context.auto ()) t in
  let values = Context.get_values ctx t.Tensor.value in
  let optimized = inspect t in
  Prefix_scan.set_enabled None;
  { values; optimized }

(* {1 Leg 1: puzzle 7 under a training step} *)

type step = {
  y : float array;
  x_grad : float array;
  optimized : LL.optimized;
  raw : LL.t;  (** The raw lowering, ahead of the rewrite tier. *)
  cache : int * int;  (** Analysis-cache hits and misses of two sibling lowerings of the step. *)
}

(* Puzzle 7's cumulative sum, the default forward of a training step. *)
let puzzle7 x =
  let tri = lower_triangle () in
  let%op y = x +* "i; i o => o" tri in
  y

let training ?(fwd = puzzle7) ?(w = weights) ~on xs =
  Tensor.unsafe_reinitialize ();
  Prefix_scan.set_enabled (Some on);
  let x = input ~grad_spec:Tensor.Require_grad "x" xs in
  let w = input "w" w and w2 = input "w2" weights2 in
  let y = fwd x in
  (* A second contribution into [x.grad]: a scan that overwrote the gradient would lose it. *)
  let%op loss = (y *. w) ++ "... | ... => 0" + ((x *. w2) ++ "... | ... => 0") in
  let x_grad = (Option.value_exn x.Tensor.diff).Tensor.grad in
  Train.set_materialized y.Tensor.value;
  Train.set_materialized x_grad;
  let update = Train.grad_update loss in
  let ctx = Train.init_params (Context.auto ()) Idx.Empty loss in
  let captured = ref None in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun o ->
        captured := Some o;
        [ o ])
      ctx update Idx.Empty
  in
  let ctx = Context.run ctx routine in
  let y = Context.get_values ctx y.Tensor.value and x_grad = Context.get_values ctx x_grad in
  let lower () =
    ignore
      (Ir.Assignments.lower (LL.empty_optimize_ctx ()) ~unoptim_ll_source:None ~ll_source:None
         ~cd_source:None ~name:"probe_step" [] update.Ir.Assignments.asgns
        : LL.optimized)
  in
  lower ();
  let h1, m1 = LL.analysis_cache_stats () in
  lower ();
  let h2, m2 = LL.analysis_cache_stats () in
  let raw = Ir.Assignments.to_low_level update.Ir.Assignments.asgns in
  Prefix_scan.set_enabled None;
  { y; x_grad; optimized = Option.value_exn !captured; raw; cache = (h2 - h1, m2 - m1) }

let () =
  eprintf "backend: %s (not part of the golden)\n%!" backend_name;
  printf "--- with the floating-point licences: %s ---\n"
    (String.concat ~sep:", "
       (List.map fp_licences ~f:(fun (key, on) -> key ^ if on then " on" else " off")));
  printf "--- leg 0: the oracles discriminate ---\n";
  let variants =
    [
      ("inclusive prefix", cumsum ints);
      ("exclusive prefix", cumsum ~inclusive:false ints);
      ("inclusive suffix", suffix ( +. ) 0. ints);
      ("exclusive suffix", suffix ~inclusive:false ( +. ) 0. ints);
    ]
  in
  p_pairwise_distinct "the four cumulative sums of the integer input differ pairwise"
    (List.map variants ~f:snd) ~equal:same_floats ~to_string:(fun _ -> "a cumulative sum");
  (* What the bitwise claims below rest on: every partial sum of every reduction the legs run on
     integer data -- in either form, in any association -- is exact in f32. *)
  (* cc-fp-contract: exact-operands -- every product multiplies by an exact 0/1 selector, and the
     integer data's sums are derived exact below *)
  let exact xs =
    Ll_test.exact_partials ~any_sign:true
      ~sig_bits:(Ir.Ops.significand_bits Ir.Ops.single)
      (Array.to_list xs)
  in
  p "the integer input's partial sums are exact in f32" (exact ints);
  p "the gradient's partial sums are exact in f32" (exact (Array.append weights weights2));
  p_pairwise_distinct "the suffix sums of the weights differ from their prefix sums"
    [ cumsum weights; suffix ( +. ) 0. weights ]
    ~equal:same_floats
    ~to_string:(fun _ -> "a cumulative sum")

let () =
  printf "--- leg 1: puzzle 7 under a training step ---\n";
  let composed = training ~on:false ints in
  let scanned = training ~on:true ints in
  let want_y = cumsum ints in
  let want_grad = Array.map2_exn (suffix ( +. ) 0. weights) weights2 ~f:( +. ) in
  report "rewritten y" scanned.y;
  report "rewritten x.grad" scanned.x_grad;
  p "composed: no scan in the step" (scans composed.optimized.LL.llc = 0);
  p "composed: the step writes the n x n selector"
    (not (Set.is_empty (square_writes composed.optimized.LL.llc)));
  p "rewritten: one forward scan (the cumsum) in the step"
    (forward_scans scanned.optimized.LL.llc = 1);
  p "rewritten: one backward scan (its adjoint, a suffix sum) in the step"
    (backward_scans scanned.optimized.LL.llc = 1);
  p "rewritten: the step writes no n x n node"
    (Set.is_empty (square_writes scanned.optimized.LL.llc));
  p "the raw step, ahead of the tier, holds no scan" (scans scanned.raw = 0);
  p_all2 "rewritten y equals the composed y bitwise" scanned.y composed.y ~f:Float.equal;
  p_all2 "rewritten y equals the host prefix sums" scanned.y want_y ~f:Float.equal;
  p_all2 "rewritten x.grad equals the composed x.grad bitwise" scanned.x_grad composed.x_grad
    ~f:Float.equal;
  p_all2 "rewritten x.grad equals the host suffix sums plus the second contribution" scanned.x_grad
    want_grad ~f:Float.equal;
  let hits, misses = scanned.cache in
  eprintf "sibling lowerings: %d hits, %d misses (not part of the golden)\n%!" hits misses;
  p "a sibling lowering of the rewritten step hits the analysis cache" (hits >= 1 && misses = 0);
  let composed_r = training ~on:false reals and scanned_r = training ~on:true reals in
  p_close "non-dyadic: rewritten y matches the composed y within 1e-6 relative" scanned_r.y
    composed_r.y ~tol:(close_tol ~tol:1e-6);
  p_close "non-dyadic: rewritten x.grad matches the composed x.grad within 1e-5 relative"
    scanned_r.x_grad composed_r.x_grad ~tol:(close_tol ~tol:1e-5)

(* {1 Leg 2: frontend variants} *)

let check_variant label ~scans_want ~exact build =
  let composed = forward ~on:false build and scanned = forward ~on:true build in
  report (label ^ ", rewritten") scanned.values;
  pf "%s: composed has no scan" label (scans composed.optimized.LL.llc = 0);
  pf "%s: rewritten has the expected scan" label (scans_want scanned.optimized.LL.llc);
  pf "%s: rewritten writes no n x n node" label
    (Set.is_empty (square_writes scanned.optimized.LL.llc));
  pf_all2 "%s: rewritten equals the composed result bitwise" label scanned.values composed.values
    ~f:Float.equal;
  pf_all2 "%s: rewritten equals the host oracle" label scanned.values exact ~f:Float.equal

let one_forward llc = forward_scans llc = 1 && backward_scans llc = 0
let one_backward llc = backward_scans llc = 1 && forward_scans llc = 0
let neg_inf = Float.neg_infinity

(* Negative only, with repeated maxima. *)
let negatives = Array.init n ~f:(fun k -> -.Float.of_int (10 + (k * 5 % 7) - (k / 3)))

let () =
  printf "--- leg 2: frontend variants ---\n";
  let batched =
    Array.init (rows * n)
      ~f:(Ll_test.cycle_flat ~dims:[| rows; n |] ~modulus:17 ~offset:(-8.) ~stride:1.)
  in
  check_variant "rows: cumsum along the last axis of a [3, n] input" ~scans_want:one_forward
    ~exact:
      (Array.concat (List.init rows ~f:(fun b -> cumsum (Array.sub batched ~pos:(b * n) ~len:n))))
    (fun () ->
      let x =
        NTDSL.init ~l:"xb" ~prec:Ir.Ops.single ~b:[] ~i:[] ~o:[ rows; n ]
          ~f:(function [| b; k |] -> batched.((b * n) + k) | _ -> assert false)
          ()
      in
      let tri = lower_triangle () in
      let%op y = x +* "b i; i o => b o" tri in
      y);
  check_variant "strict: an exclusive prefix sum (i < o)" ~scans_want:one_forward
    ~exact:(cumsum ~inclusive:false ints) (fun () ->
      let x = input "x" ints and tri = strict_triangle () in
      let%op y = x +* "i; i o => o" tri in
      y);
  check_variant "reversed: a suffix sum (i >= o)" ~scans_want:one_backward
    ~exact:(suffix ( +. ) 0. ints) (fun () ->
      let x = input "x" ints and tri = upper_triangle () in
      let%op y = x +* "i; i o => o" tri in
      y);
  check_variant "where: the selected sum" ~scans_want:one_forward ~exact:(cumsum ints) (fun () ->
      let x = input "x" ints and tri = lower_triangle () in
      let%op y = where tri (x ++ "i=>i0") 0. ++ "i o => o" in
      y);
  check_variant "cummax: where and @^^ over negative values with repeated maxima"
    ~scans_want:one_forward ~exact:(cummax negatives) (fun () ->
      let x = input "x" negatives and tri = lower_triangle () in
      let%op y = where tri (x ++ "i=>i0") !.neg_inf @^^ "i o => o" in
      y);
  check_variant "cummax, strict: the first prefix is empty" ~scans_want:one_forward
    ~exact:(cummax ~inclusive:false negatives) (fun () ->
      let x = input "x" negatives and tri = strict_triangle () in
      let%op y = where tri (x ++ "i=>i0") !.neg_inf @^^ "i o => o" in
      y);
  check_variant "tropical: @^+ against a 0 / -inf log-mask" ~scans_want:one_forward
    ~exact:(cummax negatives) (fun () ->
      let x = input "x" negatives and tri = lower_triangle () in
      let%op logm = where tri 0. !.neg_inf in
      let%op y = x @^+ "i; i o => o" logm in
      y)

(* {1 Hand-built raw lowerings} *)

module B = Ll_test

let next_base = ref 175800

(* A raw lowering of a triangular reduction: range grids [ri] (over [r]) and [rq] (over [q]), a
   materialized mask [m[a, b] = mask (ri[a]) (rq[b])], the target's fill, and the reduction nest
   [y[row, q] := y[row, q] op operand (m[r, q]) (v[row, r])] in loop order [order]. Every optional
   is a way to build a near miss. *)
let build ?(nr = n) ?(nq = n) ?(rws = 2) ?(range_prec = Ir.Ops.single) ?(order = `Row_q_r)
    ?(mask_reads_row = false) ?(payload_reads_q = false) ?(two_reduced = false)
    ?(second_mask_writer = false) ?(y_prec = Ir.Ops.single) ?(payload_shift = 0) ?(prelude = [])
    ~mask ~op ~fill ~operand () =
  let base = !next_base in
  next_base := base + 20;
  let mk = B.node_factory ~first_id:base ~dims:[| 1 |] () in
  let mkr = B.node_factory ~prec:range_prec ~first_id:(base + 10) ~dims:[| nr |] () in
  let ri = mkr "ri" in
  let rq = mkr ~dims:[| nq |] "rq" in
  let m = mk ~dims:(if mask_reads_row then [| rws; nr; nq |] else [| nr; nq |]) "m" in
  (* Room for a shifted payload's reads: the composed form's mask is a value, which guards no range
     (gh-ocannl-2258), so its reads stay in the buffer as lowering would size it. *)
  let v = mk ~dims:[| rws; Int.max nr nq + payload_shift |] "v" in
  let y = B.node_factory ~prec:y_prec ~first_id:(base + 15) ~dims:[| rws; nq |] () "y" in
  List.iter [ v; y ] ~f:B.materialize;
  let k = B.sym () and k' = B.sym () in
  let a = B.sym () and b = B.sym () and w = B.sym () in
  let mask_idcs =
    if mask_reads_row then [| B.iter w; B.iter a; B.iter b |] else [| B.iter a; B.iter b |]
  in
  let mask_def =
    let body =
      B.set m mask_idcs
        (mask
           ~row:(if mask_reads_row then Some (B.embed w) else None)
           (B.get ri [| B.iter a |])
           (B.get rq [| B.iter b |]))
    in
    let inner = B.loop_n a nr (B.loop_n b nq body) in
    if mask_reads_row then B.loop_n w rws inner else inner
  in
  let fy = B.sym () and fq = B.sym () in
  let fill_y = B.loop_n fy rws (B.loop_n fq nq (B.set y [| B.iter fy; B.iter fq |] (B.c fill))) in
  let row = B.sym () and q = B.sym () and r = B.sym () and e = B.sym () in
  let y_cell = [| B.iter row; B.iter q |] in
  let m_read =
    B.get m
      (if mask_reads_row then [| B.iter row; B.iter r; B.iter q |] else [| B.iter r; B.iter q |])
  in
  let v_read =
    let at = if payload_reads_q then q else r in
    B.get v
      [| B.iter row; (if payload_shift = 0 then B.iter at else B.aff [ (1, at) ] payload_shift) |]
  in
  let red = B.set y y_cell (B.binop_own op (B.get y y_cell) (operand m_read v_read)) in
  let red = if two_reduced then B.loop_n e 2 red else red in
  let nest =
    match order with
    | `Row_q_r -> B.loop_n row rws (B.loop_n q nq (B.loop_n r nr red))
    | `R_row_q -> B.loop_n r nr (B.loop_n row rws (B.loop_n q nq red))
  in
  let extra = if second_mask_writer then [ B.set m [| B.fixed 0; B.fixed 0 |] (B.c 0.) ] else [] in
  let llc =
    LL.unflat_lines
      ([
         B.loop_n k nr (B.set_at ri (B.iter k) (B.embed k));
         B.loop_n k' nq (B.set_at rq (B.iter k') (B.embed k'));
         mask_def;
       ]
      @ extra @ prelude @ [ fill_y; nest ])
  in
  (llc, v, y)

let le ~row:_ a b = B.binop_own Ir.Ops.Cmple a b
let gt ~row:_ a b = B.binop_own Ir.Ops.Cmplt b a
let where_v mask v = B.where_own mask v (B.c Float.infinity)
let product mask v = B.mul_own v mask

(* Discriminating payload: varies with the row and the position, changes direction, repeats. *)
let payload ~rws ~len =
  Array.init (rws * len) ~f:(fun cell ->
      Ll_test.weighted ~weights:[| 5; 7 |] ~modulus:13 ~offset:(-6.) ~stride:1.
        [| cell / len; cell % len |])

let guarded_steps llc =
  Ll_test.count_scalar llc ~f:(function
    | LL.Ternop (Ir.Ops.Where, (LL.Binop (Ir.Ops.Cmplt, _, _), _), _, _) -> true
    | _ -> false)

let hand_built ~name ~op_fn ~fill ?(inclusive = true) ?(direction = `Prefix) ?nr ?rws ?order
    ?(payload_shift = 0) ~mask ~op ~operand () =
  let rws = Option.value rws ~default:2 and len = Option.value nr ~default:n in
  let llc, v, y = build ?nr ?nq:nr ~rws ?order ~payload_shift ~mask ~op ~fill ~operand () in
  let rewritten = Prefix_scan.rewrite llc in
  let width = len + payload_shift in
  let seed = [ (v, payload ~rws ~len:width); (y, B.blank (rws * len)) ] in
  let _, composed = B.optimize_and_execute ~name:(name ^ "_comp") llc ~seed ~read:[ y ] in
  let o, scanned = B.optimize_and_execute ~name:(name ^ "_scan") rewritten ~seed ~read:[ y ] in
  let composed = List.hd_exn composed and scanned = List.hd_exn scanned in
  let pay = payload ~rws ~len:width in
  let want =
    Array.concat
      (List.init rws ~f:(fun row ->
           let xs = Array.sub pay ~pos:(row * width) ~len:width in
           let xs =
             Array.init len ~f:(fun r ->
                 if r + payload_shift < len then xs.(r + payload_shift) else 0.)
           in
           match direction with
           | `Prefix -> prefix ~inclusive op_fn fill xs
           | `Suffix -> suffix ~inclusive op_fn fill xs))
  in
  report (name ^ ", rewritten") scanned;
  pf "%s: the rewrite emits one scan" name (scans rewritten = 1);
  (* An exclusive scan's last step must not read the payload the composed form never selected. *)
  pf "%s: %s" name
    (if inclusive then "no step is guarded" else "the step is guarded at the scan's end")
    (guarded_steps rewritten = if inclusive then 0 else 1);
  pf "%s: the scan survives optimization" name (scans o.LL.llc = 1);
  pf_all2 "%s: rewritten equals the composed execution bitwise" name scanned composed ~f:Float.equal;
  pf_all2 "%s: rewritten equals the host oracle" name scanned want ~f:Float.equal;
  pf "%s: the rewrite is idempotent" name (LL.equal (Prefix_scan.rewrite rewritten) rewritten)

let () =
  printf "--- leg 3: hand-built raw lowerings, executed ---\n";
  p "the hand-built payloads' partial sums are exact in f32"
    (Ll_test.exact_partials ~any_sign:true
       ~sig_bits:(Ir.Ops.significand_bits Ir.Ops.single)
       (Array.to_list (payload ~rws:2 ~len:n)));
  hand_built ~name:"ps_min" ~op_fn:Float.min ~fill:Float.infinity ~mask:le ~op:Ir.Ops.Min
    ~operand:where_v ();
  hand_built ~name:"ps_min_order" ~op_fn:Float.min ~fill:Float.infinity ~order:`R_row_q ~mask:le
    ~op:Ir.Ops.Min ~operand:where_v ();
  (* [where (r > q) 0 v]: the complement selects [r <= q]. *)
  hand_built ~name:"ps_complement" ~op_fn:( +. ) ~fill:0. ~mask:gt ~op:Ir.Ops.Add
    ~operand:(fun mask v -> B.where_own mask (B.c 0.) v)
    ();
  (* [r >= q]: a suffix. *)
  hand_built ~name:"ps_suffix" ~op_fn:( +. ) ~fill:0. ~direction:`Suffix
    ~mask:(fun ~row:_ a b -> B.binop_own Ir.Ops.Cmple b a)
    ~op:Ir.Ops.Add ~operand:product ();
  (* [where (r < q) v[r + 1] 0]: the composed form never reads [v[n]]; the scan's last step
     would. *)
  hand_built ~name:"ps_exclusive_shifted" ~op_fn:( +. ) ~fill:0. ~inclusive:false ~payload_shift:1
    ~mask:(fun ~row:_ a b -> B.binop_own Ir.Ops.Cmplt a b)
    ~op:Ir.Ops.Add
    ~operand:(fun mask v -> B.where_own mask v (B.c 0.))
    ();
  (* A stored payload factor [range * 1e20] is affine with a coefficient no [int] holds: the
     recognizer must decline it as a mask factor, not raise, and scan with it as payload. *)
  (let big = B.node_factory ~first_id:179100 ~dims:[| n |] () "big" in
   let k = B.sym () in
   let llc, v, y =
     build
       ~prelude:[ B.loop_n k n (B.set_at big (B.iter k) (B.mul_own (B.embed k) (B.c 1e20))) ]
       ~mask:le ~op:Ir.Ops.Add ~fill:0.
       ~operand:(fun mask v ->
         match mask with
         | LL.Get (_, idcs) -> B.mul_own (product mask v) (B.get big [| idcs.(0) |])
         | _ -> assert false)
       ()
   in
   ignore (v, y);
   p "a payload factor with a huge affine coefficient is scanned, not refused"
     (scans (Prefix_scan.rewrite llc) = 1));
  (* A selector computed through a comparison of infinities: [inf <= inf] holds, so the mask is [r
     <= q]; read as [inf - inf <= 0] it would be the opposite triangle. *)
  hand_built ~name:"ps_infinite_cond" ~op_fn:( +. ) ~fill:0.
    ~mask:(fun ~row:_ a b ->
      B.where_own
        (B.binop_own Ir.Ops.Cmple (B.c Float.infinity) (B.c Float.infinity))
        (B.binop_own Ir.Ops.Cmple a b) (B.binop_own Ir.Ops.Cmplt b a))
    ~op:Ir.Ops.Add ~operand:product ();
  hand_built ~name:"ps_pair" ~op_fn:( +. ) ~fill:0. ~nr:2 ~rws:1 ~mask:le ~op:Ir.Ops.Add
    ~operand:product ()

(* {1 Leg 4: what the recognizer leaves alone} *)

let declined label (llc, _, _) = pf "%s: left alone" label (LL.equal (Prefix_scan.rewrite llc) llc)

let () =
  printf "--- leg 4: near misses are left alone ---\n";
  let add = Ir.Ops.Add and fill = 0. in
  declined "a shifted triangle (r <= q + 1)"
    (build
       ~mask:(fun ~row:_ a b -> B.binop_own Ir.Ops.Cmple a (B.add_own b (B.c 1.)))
       ~op:add ~fill ~operand:product ());
  declined "a band (r <= q and q - r < 3)"
    (build
       ~mask:(fun ~row:_ a b ->
         B.mul_own (B.binop_own Ir.Ops.Cmple a b)
           (B.binop_own Ir.Ops.Cmplt (B.binop_own Ir.Ops.Sub b a) (B.c 3.)))
       ~op:add ~fill ~operand:product ());
  declined "an anti-triangle (r + q <= n - 1)"
    (build
       ~mask:(fun ~row:_ a b ->
         B.binop_own Ir.Ops.Cmple (B.add_own a b) (B.c (Float.of_int (n - 1))))
       ~op:add ~fill ~operand:product ());
  declined "a mask varying with the row"
    (build ~mask_reads_row:true
       ~mask:(fun ~row a b -> B.binop_own Ir.Ops.Cmple a (B.add_own b (Option.value_exn row)))
       ~op:add ~fill ~operand:product ());
  (* Over one cell the triangle is the constant 1, indistinguishable from a constant factor. *)
  declined "a singleton extent"
    (build ~nr:1 ~nq:1 ~rws:1 ~mask:le ~op:add ~fill ~operand:product ());
  declined "mismatched extents" (build ~nq:(n - 1) ~mask:le ~op:add ~fill ~operand:product ());
  declined "a payload that reads the output position"
    (build ~payload_reads_q:true ~mask:le ~op:add ~fill ~operand:product ());
  declined "two reduced loops" (build ~two_reduced:true ~mask:le ~op:add ~fill ~operand:product ());
  declined "a selector with a second writer"
    (build ~second_mask_writer:true ~mask:le ~op:add ~fill ~operand:product ());
  declined "an f16 index grid past 2048"
    (build ~nr:2050 ~nq:2050 ~rws:1 ~range_prec:Ir.Ops.half ~mask:le ~op:add ~fill ~operand:product
       ());
  declined "a max over a multiplicative 0/1 mask (its zeros are not neutral)"
    (build ~mask:le ~op:Ir.Ops.Max ~fill:Float.neg_infinity ~operand:product ());
  declined "an integer target (an f32 state would round its sums)"
    (build ~y_prec:Ir.Ops.int32 ~mask:le ~op:add ~fill ~operand:product ());
  (* [z = (16777217 - 16777216) - 1] is 0 over the integers, -1 through f32 storage, which rounds
     16777217 to 16777216: not the neutral element the rejected arm must be. *)
  (let mk = B.node_factory ~first_id:179000 ~dims:[| 1 |] () in
   let ca = mk "ca" and cb = mk "cb" and cz = mk "cz" in
   let at = [| B.fixed 0 |] in
   let prelude =
     [
       B.set ca at (B.c 16777217.);
       B.set cb at (B.binop_own Ir.Ops.Sub (B.get ca at) (B.c 16777216.));
       B.set cz at (B.binop_own Ir.Ops.Sub (B.get cb at) (B.c 1.));
     ]
   in
   declined "a rejected arm that is zero only before f32 rounding"
     (build ~prelude ~mask:le ~op:add ~fill
        ~operand:(fun mask v -> B.where_own mask v (B.get cz at))
        ()));
  declined "staged code ahead of the reduction (it may write what the proof read)"
    (build
       ~prelude:[ LL.Staged_compilation (fun () -> PPrint.string "/* staged */") ]
       ~mask:le ~op:add ~fill ~operand:product ());
  (* [saved] copies [old] BEFORE [old] is written: unfolding [saved] through [old]'s later
     definition would scan values [saved] never held. *)
  (let mk = B.node_factory ~first_id:179200 ~dims:[| n; n |] () in
   let saved = mk "saved" and old = mk "old" in
   let a = B.sym () and b = B.sym () and a' = B.sym () and b' = B.sym () in
   let cell a b = [| B.iter a; B.iter b |] in
   let prelude_of ~m ~v =
     [
       B.loop_n a n (B.loop_n b n (B.set saved (cell a b) (B.get old (cell a b))));
       B.loop_n a' n
         (B.loop_n b' n
            (B.set old (cell a' b')
               (B.where_own (B.get m (cell a' b')) (B.get v [| B.fixed 0; B.iter a' |]) (B.c 0.))));
     ]
   in
   (* [build] makes [m] and [v] itself: take them back out of its operand's reads. *)
   let m_and_v = ref None in
   let llc, _, _ =
     build ~rws:1 ~mask:le ~op:add ~fill
       ~operand:(fun mask v ->
         (match (mask, v) with
         | LL.Get (m, _), LL.Get (v, _) -> m_and_v := Some (m, v)
         | _ -> assert false);
         match mask with LL.Get (_, idcs) -> B.get saved idcs | _ -> assert false)
       ()
   in
   let m, v = Option.value_exn !m_and_v in
   let stmts = LL.flat_lines [ llc ] in
   (* The prelude goes right after the mask's definition (the third statement). *)
   let llc = LL.unflat_lines (List.take stmts 3 @ prelude_of ~m ~v @ List.drop stmts 3) in
   declined "a copy unfolded through a definition written after the copy" (llc, (), ()));
  declined "a min whose rejected cells are 0, not +inf"
    (build ~mask:le ~op:Ir.Ops.Min ~fill:Float.infinity
       ~operand:(fun mask v -> B.where_own mask v (B.c 0.))
       ())

(* {1 Leg 5: an infinite payload} *)

let () =
  printf "--- leg 5: an infinite payload ---\n";
  let at = 5 in
  let xs = Array.mapi ints ~f:(fun k x -> if k = at then Float.infinity else x) in
  let build () =
    let x = input "x" xs and tri = lower_triangle () in
    let%op y = x +* "i; i o => o" tri in
    y
  in
  let composed = forward ~on:false build and scanned = forward ~on:true build in
  report "infinite payload, composed" composed.values;
  report "infinite payload, rewritten" scanned.values;
  let want = cumsum xs in
  p_all2 "rewritten: the prefix before the infinity is the finite cumulative sum"
    (Array.sub scanned.values ~pos:0 ~len:at)
    (Array.sub want ~pos:0 ~len:at) ~f:Float.equal;
  p_all "rewritten: every sum from the infinity on is +inf"
    (Array.to_list (Array.sub scanned.values ~pos:at ~len:(n - at)))
    ~f:(fun v -> Float.(is_inf v && v > 0.))

(* {1 Leg 6: the cummax backward (gh-ocannl-1858)} *)

(* The scans carrying the extremum's gradient gate: two carried values, the run's sum and the
   previous extremum (every other scan here carries one). *)
let gate_scans llc =
  Ll_test.count_stmt llc ~f:(function
    | LL.Scan_loop { carried; _ } -> List.length carried = 2
    | _ -> false)

(* The extremum over each output's selection: [sel i o] says output [o] selects input [i]. *)
let selected_extremum ~op ~fill ~sel xs =
  Array.init (Array.length xs) ~f:(fun o ->
      Array.foldi xs ~init:fill ~f:(fun i acc x -> if sel i o then op acc x else acc))

(* The composed form's tie rule, written out: input [i] takes the upstream gradient of every output
   selecting it whose extremum it attains -- all of it, so tied positions each take it. *)
let gate_oracle ~ys ~sel ~g xs =
  Array.mapi xs ~f:(fun i x ->
      Array.foldi ys ~init:0. ~f:(fun o acc y ->
          if sel i o && Float.equal y x then acc +. g.(o) else acc))

let cummax_of tri x =
  let tri = tri () in
  let%op y = where tri (x ++ "i=>i0") !.neg_inf @^^ "i o => o" in
  y

(* A repeated maximum separated by losing values, then a higher one, repeated too. *)
let separated =
  Array.init n ~f:(fun k ->
      if k % 4 = 0 then if k < 10 then 5. else 7. else Float.of_int ((k % 3) - 2))

let increasing = Array.init n ~f:(fun k -> Float.of_int (k - 30))
let decreasing = Array.init n ~f:(fun k -> Float.of_int (-k))
let all_equal = Array.create ~len:n (-3.)

let () =
  printf "--- leg 6: the cummax backward ---\n";
  let prefix_sel i o = i <= o and strict_sel i o = i < o and suffix_sel i o = i >= o in
  let case label ~tri ~sel ~gate_dir xs =
    let composed = training ~fwd:(cummax_of tri) ~on:false xs in
    let scanned = training ~fwd:(cummax_of tri) ~on:true xs in
    let ys = selected_extremum ~op:Float.max ~fill:neg_inf ~sel xs in
    let want = Array.map2_exn (gate_oracle ~ys ~sel ~g:weights xs) weights2 ~f:( +. ) in
    report (label ^ ", rewritten x.grad") scanned.x_grad;
    let llc = scanned.optimized.LL.llc in
    pf "%s: composed has no scan" label (scans composed.optimized.LL.llc = 0);
    pf "%s: rewritten scans the cummax and its gradient gate (%s), and writes no n x n node" label
      (match gate_dir with LL.Backward -> "backward" | LL.Forward -> "forward")
      (gate_scans llc = 1
      && scans_where llc ~f:(fun d -> LL.equal_scan_direction d gate_dir) = 1
      && scans llc = 2
      && Set.is_empty (square_writes llc));
    pf_all2 "%s: rewritten y and x.grad equal the composed bitwise" label
      (Array.append scanned.y scanned.x_grad)
      (Array.append composed.y composed.x_grad)
      ~f:Float.equal;
    pf_all2 "%s: rewritten x.grad equals the tie-rule oracle" label scanned.x_grad want
      ~f:Float.equal;
    scanned
  in
  let first =
    case "negatives with repeated maxima" ~tri:lower_triangle ~sel:prefix_sel ~gate_dir:LL.Backward
      negatives
  in
  ignore
    (case "a repeated maximum separated by losing values" ~tri:lower_triangle ~sel:prefix_sel
       ~gate_dir:LL.Backward separated
      : step);
  ignore
    (case "all equal" ~tri:lower_triangle ~sel:prefix_sel ~gate_dir:LL.Backward all_equal : step);
  ignore
    (case "strictly increasing" ~tri:lower_triangle ~sel:prefix_sel ~gate_dir:LL.Backward increasing
      : step);
  ignore
    (case "strictly decreasing" ~tri:lower_triangle ~sel:prefix_sel ~gate_dir:LL.Backward decreasing
      : step);
  ignore
    (case "strict, the first prefix empty" ~tri:strict_triangle ~sel:strict_sel
       ~gate_dir:LL.Backward negatives
      : step);
  ignore
    (case "strict, separated ties" ~tri:strict_triangle ~sel:strict_sel ~gate_dir:LL.Backward
       separated
      : step);
  ignore
    (case "suffix (a reversed cummax)" ~tri:upper_triangle ~sel:suffix_sel ~gate_dir:LL.Forward
       negatives
      : step);
  ignore
    (case "suffix, separated ties" ~tri:upper_triangle ~sel:suffix_sel ~gate_dir:LL.Forward
       separated
      : step);
  let hits, misses = first.cache in
  eprintf "cummax step, sibling lowerings: %d hits, %d misses (not part of the golden)\n%!" hits
    misses;
  p "a sibling lowering of the rewritten cummax step hits the analysis cache"
    (hits >= 1 && misses = 0);
  (* Non-dyadic upstream gradients: the scan sums a run from its end, the composed form from its
     start. *)
  let w = Array.init n ~f:(fun o -> 1.1 +. Float.sin (Float.of_int o)) in
  let ties = Array.map separated ~f:(fun v -> v *. 0.37) in
  let composed = training ~fwd:(cummax_of lower_triangle) ~w ~on:false ties in
  let scanned = training ~fwd:(cummax_of lower_triangle) ~w ~on:true ties in
  p "non-dyadic: the gate is rewritten" (gate_scans scanned.optimized.LL.llc = 1);
  p_close "non-dyadic: rewritten x.grad matches the composed within 1e-5 relative" scanned.x_grad
    composed.x_grad ~tol:(close_tol ~tol:1e-5);
  (* The tropical form gates differently -- with no [where] filter, an output whose extremum is -inf
     gates its rejected positions too -- so it is left composed. *)
  let tropical x =
    let tri = lower_triangle () in
    let%op logm = where tri 0. !.neg_inf in
    let%op y = x @^+ "i; i o => o" logm in
    y
  in
  let composed = training ~fwd:tropical ~on:false negatives in
  let scanned = training ~fwd:tropical ~on:true negatives in
  p "tropical @^+: its gradient gate is left composed" (gate_scans scanned.optimized.LL.llc = 0);
  p_all2 "tropical @^+: rewritten x.grad equals the composed bitwise" scanned.x_grad composed.x_grad
    ~f:Float.equal

(* {1 Leg 7: hand-built gradient gates}

   The raw lowering of the cummax backward, built by hand over rows: [P[w, i, o] = where(M(i, o),
   v[w, i], fill)], [Y[w, o]] its neutral-filled reduction, [C[w, o, i] = (Y[w, o] = P[w, i, o])],
   [A[w, i, o] = 0 + where(C[w, o, i], g[w, o], 0)], and the target [T[w, i] += where(M'(i, o), A[w,
   i, o], 0)] over [o]. Every optional builds a near miss. *)

let gate_base = ref 185800

let build_gate ?(rws = 2) ?(op = Ir.Ops.Max) ?(fwd_mask = le) ?(t_mask = le)
    ?(chain_prec = Ir.Ops.single) ?(v_prec = Ir.Ops.single) ?a_prec ?c_prec ?g_prec ?g_read_prec
    ?(second_a_writer = false) ?(compare_copy = false) ?(row_from_i = false)
    ?(materialize_product = false) ?y_fill () =
  let base = !gate_base in
  gate_base := base + 20;
  let fill = if Ir.Ops.equal_binop op Ir.Ops.Max then neg_inf else Float.infinity in
  let y_fill = Option.value y_fill ~default:fill in
  let a_prec = Option.value a_prec ~default:chain_prec in
  let c_prec = Option.value c_prec ~default:chain_prec in
  let g_prec = Option.value g_prec ~default:chain_prec in
  let g_read_prec = Option.value g_read_prec ~default:g_prec in
  let mk = B.node_factory ~first_id:base ~dims:[| 1 |] () in
  (* [chain_prec]: the forward's intermediates, the gate's and the target's precision. *)
  let mkc = B.node_factory ~prec:chain_prec ~first_id:(base + 11) ~dims:[| 1 |] () in
  let ri = mk ~dims:[| n |] "gri" and rq = mk ~dims:[| n |] "grq" in
  let m = mk ~dims:[| n; n |] "gm" and mt = mk ~dims:[| n; n |] "gmt" in
  let v = B.node_factory ~prec:v_prec ~first_id:(base + 18) ~dims:[| rws; n |] () "gv" in
  let g = B.node_factory ~prec:g_prec ~first_id:(base + 10) ~dims:[| rws; n |] () "gg" in
  let t = mkc ~dims:[| rws; n |] "gt" in
  List.iter [ v; g; t ] ~f:B.materialize;
  let pn = mkc ~dims:[| rws; n; n |] "gp" and y = mkc ~dims:[| rws; n |] "gy" in
  let y_copy = mkc ~dims:[| rws; n |] "gy_copy" in
  let cn = B.node_factory ~prec:c_prec ~first_id:(base + 17) ~dims:[| rws; n; n |] () "gc" in
  let an = B.node_factory ~prec:a_prec ~first_id:(base + 15) ~dims:[| rws; n; n |] () "ga" in
  if materialize_product then B.materialize an;
  let s () = B.sym () in
  let nest3 f =
    let w = s () and a = s () and b = s () in
    B.loop_n w rws (B.loop_n a n (B.loop_n b n (f (B.iter w) (B.iter a) (B.iter b))))
  in
  let k = s () and k' = s () and a = s () and b = s () and a' = s () and b' = s () in
  let mask_def tn mask a b =
    B.loop_n a n
      (B.loop_n b n
         (B.set tn
            [| B.iter a; B.iter b |]
            (mask ~row:None (B.get ri [| B.iter a |]) (B.get rq [| B.iter b |]))))
  in
  let w1 = s () and o1 = s () in
  let w2 = s () and o2 = s () and i2 = s () in
  let w3 = s () and o3 = s () in
  let reduce_y =
    B.loop_n w2 rws
      (B.loop_n o2 n
         (B.loop_n i2 n
            (B.set y
               [| B.iter w2; B.iter o2 |]
               (B.binop_own op
                  (B.get y [| B.iter w2; B.iter o2 |])
                  (B.get pn [| B.iter w2; B.iter i2; B.iter o2 |])))))
  in
  let cmp_y = if compare_copy then y_copy else y in
  let accumulate_a () =
    nest3 (fun w i o ->
        B.set an [| w; i; o |]
          (B.add_own
             (B.get an [| w; i; o |])
             (LL.Ternop
                ( Ir.Ops.Where,
                  (B.get cn [| w; o; i |], c_prec),
                  (B.get g [| w; o |], g_read_prec),
                  (B.c 0., Ir.Ops.single) ))))
  in
  let llc =
    LL.unflat_lines
      ([
         B.loop_n k n (B.set_at ri (B.iter k) (B.embed k));
         B.loop_n k' n (B.set_at rq (B.iter k') (B.embed k'));
         mask_def m fwd_mask a b;
         mask_def mt t_mask a' b';
         nest3 (fun w i o ->
             B.set pn [| w; i; o |]
               (B.where_own (B.get m [| i; o |]) (B.get v [| w; i |]) (B.c fill)));
         B.loop_n w1 rws (B.loop_n o1 n (B.set y [| B.iter w1; B.iter o1 |] (B.c y_fill)));
         reduce_y;
       ]
      @ (if compare_copy then
           [
             B.loop_n w3 rws
               (B.loop_n o3 n
                  (B.set y_copy [| B.iter w3; B.iter o3 |] (B.get y [| B.iter w3; B.iter o3 |])));
           ]
         else [])
      @ [
          nest3 (fun w o i ->
              (* [row_from_i]: the comparison reads the forward's row at the input position. *)
              let row = if row_from_i then i else w in
              B.set cn [| w; o; i |]
                (B.binop_own Ir.Ops.Cmpeq (B.get cmp_y [| row; o |]) (B.get pn [| row; i; o |])));
          B.zero an;
          accumulate_a ();
        ]
      @ (if second_a_writer then [ accumulate_a () ] else [])
      @ [
          nest3 (fun w i o ->
              B.set t [| w; i |]
                (B.add_own
                   (B.get t [| w; i |])
                   (B.where_own (B.get mt [| i; o |]) (B.get an [| w; i; o |]) (B.c 0.))));
        ])
  in
  (llc, v, g, t)

let gate_seed ?(gradient_offset = 1.) ~rws () =
  let v = payload ~rws ~len:n in
  let dims = [| rws; n |] in
  (* Upstream gradients clear of zero, and a target holding earlier contributions. *)
  let g =
    Array.init (rws * n)
      ~f:(Ll_test.cycle_flat ~dims ~modulus:11 ~offset:gradient_offset ~stride:1.)
  in
  let t = Array.init (rws * n) ~f:(Ll_test.cycle_flat ~dims ~modulus:9 ~offset:(-4.) ~stride:1.) in
  (v, g, t)

let run_gate ~name ?(rws = 2) ?gradient_offset ?(prepare_rewritten = Fn.id) llc (v, g, t) =
  let vs, gs, ts = gate_seed ?gradient_offset ~rws () in
  let rewritten = Prefix_scan.rewrite llc in
  let seed = [ (v, vs); (g, gs); (t, ts) ] in
  let _, composed = B.optimize_and_execute ~name:(name ^ "_comp") llc ~seed ~read:[ t ] in
  let o, scanned =
    B.optimize_and_execute ~name:(name ^ "_scan") (prepare_rewritten rewritten) ~seed ~read:[ t ]
  in
  (rewritten, o, List.hd_exn composed, List.hd_exn scanned)

let () =
  printf "--- leg 7: hand-built gradient gates, executed ---\n";
  let rws = 2 in
  let recognized name ~op_fn ~fill ~sel ?op ?fwd_mask ?t_mask () =
    let llc, v, g, t = build_gate ~rws ?op ?fwd_mask ?t_mask () in
    let rewritten, o, composed, scanned = run_gate ~name llc (v, g, t) in
    let vs, gs, ts = gate_seed ~rws () in
    let want =
      Array.concat
        (List.init rws ~f:(fun row ->
             let xs = Array.sub vs ~pos:(row * n) ~len:n
             and gr = Array.sub gs ~pos:(row * n) ~len:n in
             let ys = selected_extremum ~op:op_fn ~fill ~sel xs in
             Array.map2_exn
               (Array.sub ts ~pos:(row * n) ~len:n)
               (gate_oracle ~ys ~sel ~g:gr xs) ~f:( +. )))
    in
    report (name ^ ", rewritten") scanned;
    pf "%s: the rewrite emits the forward scan and its gate scan" name
      (gate_scans rewritten = 1 && scans rewritten = 2);
    pf "%s: the gate scan survives optimization" name (gate_scans o.LL.llc = 1);
    pf_all2 "%s: rewritten equals the composed execution bitwise" name scanned composed
      ~f:Float.equal;
    pf_all2 "%s: rewritten equals the tie-rule oracle" name scanned want ~f:Float.equal;
    pf "%s: the rewrite is idempotent" name (LL.equal (Prefix_scan.rewrite rewritten) rewritten)
  in
  recognized "gg_max" ~op_fn:Float.max ~fill:neg_inf ~sel:(fun i o -> i <= o) ();
  recognized "gg_min (a cummin)" ~op:Ir.Ops.Min ~op_fn:Float.min ~fill:Float.infinity
    ~sel:(fun i o -> i <= o)
    ();
  recognized "gg_min_strict" ~op:Ir.Ops.Min ~op_fn:Float.min ~fill:Float.infinity
    ~fwd_mask:(fun ~row:_ a b -> B.binop_own Ir.Ops.Cmplt a b)
    ~t_mask:(fun ~row:_ a b -> B.binop_own Ir.Ops.Cmplt a b)
    ~sel:(fun i o -> i < o)
    ();
  recognized "gg_max_suffix" ~op_fn:Float.max ~fill:neg_inf
    ~fwd_mask:(fun ~row:_ a b -> B.binop_own Ir.Ops.Cmple b a)
    ~t_mask:(fun ~row:_ a b -> B.binop_own Ir.Ops.Cmple b a)
    ~sel:(fun i o -> i >= o)
    ();
  let left_composed ?(rws = rws) ?gradient_offset ?prepare_rewritten label (llc, v, g, t) =
    let rewritten, _, composed, scanned =
      run_gate ~name:label ~rws ?gradient_offset ?prepare_rewritten llc (v, g, t)
    in
    report (label ^ ", composed") composed;
    report (label ^ ", rewritten") scanned;
    pf "%s: the gate is left composed" label (gate_scans rewritten = 0);
    pf_all2 "%s: rewritten equals the composed execution bitwise" label scanned composed
      ~f:Float.equal
  in
  left_composed "gg_mismatched (the gate's triangle is strict, the forward's is not)"
    (build_gate ~rws ~t_mask:(fun ~row:_ a b -> B.binop_own Ir.Ops.Cmplt a b) ());
  left_composed "gg_second_writer (a second contribution into the gate's product)"
    (build_gate ~rws ~second_a_writer:true ());
  left_composed "gg_copy (the gate compares a copy of the extremum)"
    (build_gate ~rws ~compare_copy:true ());
  left_composed "gg_fill (the extremum's fill is 0, not -inf)" (build_gate ~rws ~y_fill:0. ());
  left_composed "gg_prec (the gate's product is f16, the target f32)"
    (build_gate ~rws ~a_prec:Ir.Ops.half ());
  (* The gate's equality stored at f16 beside an f32 extremum: it computes at f16, where 2049 and
     2048 compare equal. *)
  left_composed "gg_narrow (the gate's equality is f16, the extremum f32)"
    (build_gate ~rws ~c_prec:Ir.Ops.half ());
  (* An f16 chain over an f32 payload: [C] compares the stored, rounded [P], and a wide-compute
     backend would compare the unrounded payload. *)
  left_composed "gg_wide_payload (the payload is f32, the forward and its gate f16)"
    (build_gate ~rws ~chain_prec:Ir.Ops.half ~v_prec:Ir.Ops.single ());
  (* The forward's row read at the input position: [Y[i, o]] and [P[i, i, o]], a diagonal walk the
     scan would not follow. *)
  left_composed ~rws:n "gg_row_from_i (the gate reads the forward's row at the input position)"
    (build_gate ~rws:n ~row_from_i:true ());
  (* The f16 read annotation passes the read-precision check, leaving the f32 storage check to
     reject the gate. Keep the product stored so the composed execution rounds odd gradients above
     2048 before summing them. If the rewrite moves the read into an f32 sum, re-describe its
     operand annotation for execution so the parity claim can see the arithmetic. *)
  left_composed ~gradient_offset:2049. ~prepare_rewritten:LL.describe_operand_precisions
    "gg_wide_grad (the upstream gradient is f32, the gate's product f16)"
    (build_gate ~rws ~chain_prec:Ir.Ops.half ~v_prec:Ir.Ops.half ~g_prec:Ir.Ops.single
       ~g_read_prec:Ir.Ops.half ~materialize_product:true ())

(* {1 Leg 8: constant masks (gh-ocannl-1858)}

   A mask held in a host constant -- an [ndarray] literal, whose values are declared forever equal
   to its registered data -- is proven a triangle from that data. A mask built with [NTDSL.init] is
   mutable host data by contract, so it is left alone. *)

let mask_values sel = Array.init (n * n) ~f:(fun c -> if sel (c / n) (c % n) then 1. else 0.)
let literal values = NTDSL.ndarray values ~label:[ "tri_lit" ] ~output_dims:[ n; n ] ()

let matvec values xs =
  Array.init n ~f:(fun o ->
      Array.foldi xs ~init:0. ~f:(fun i acc x -> acc +. (x *. values.((i * n) + o))))

let () =
  printf "--- leg 8: constant masks ---\n";
  let lit label ~scans_want ~exact sel =
    check_variant label ~scans_want ~exact (fun () ->
        let x = input "x" ints and tri = literal (mask_values sel) in
        let%op y = x +* "i; i o => o" tri in
        y)
  in
  lit "literal (i <= o): an inclusive prefix" ~scans_want:one_forward ~exact:(cumsum ints)
    (fun i o -> i <= o);
  lit "literal (i < o): an exclusive prefix" ~scans_want:one_forward
    ~exact:(cumsum ~inclusive:false ints) (fun i o -> i < o);
  lit "literal (i >= o): an inclusive suffix" ~scans_want:one_backward
    ~exact:(suffix ( +. ) 0. ints) (fun i o -> i >= o);
  lit "literal (i > o): an exclusive suffix" ~scans_want:one_backward
    ~exact:(suffix ~inclusive:false ( +. ) 0. ints) (fun i o -> i > o);
  (* One interior cell off the triangle. *)
  let near = mask_values (fun i o -> i <= o) in
  near.((3 * n) + 10) <- 0.;
  let build_near () =
    let x = input "x" ints and tri = literal near in
    let%op y = x +* "i; i o => o" tri in
    y
  in
  let composed = forward ~on:false build_near and scanned = forward ~on:true build_near in
  p "a literal one cell off the triangle is left alone" (scans scanned.optimized.LL.llc = 0);
  p_all2 "the near-triangle literal: rewritten equals the composed bitwise" scanned.values
    composed.values ~f:Float.equal;
  p_all2 "the near-triangle literal: rewritten equals the host product" scanned.values
    (matvec near ints) ~f:Float.equal;
  (* [NTDSL.init]: the same triangle as mutable data. A host write after compiling changes what the
     routine computes, so no proof from the data at lowering would hold. *)
  Tensor.unsafe_reinitialize ();
  Prefix_scan.set_enabled (Some true);
  let x = input "x" ints in
  let tri =
    NTDSL.init ~l:"tri_init" ~prec:Ir.Ops.single ~o:[ n; n ]
      ~f:(function [| i; o |] -> if i <= o then 1. else 0. | _ -> assert false)
      ()
  in
  let%op y = x +* "i; i o => o" tri in
  Train.set_materialized y.Tensor.value;
  let captured = ref None in
  let ctx = Train.init_params (Context.auto ()) Idx.Empty y in
  let ctx, routine =
    Context.compile ~name:"init_mask"
      ~lowered_transform:(fun o ->
        captured := Some o;
        [ o ])
      ctx y.Tensor.forward Idx.Empty
  in
  let ctx = Context.run ctx routine in
  let before = Context.get_values ctx y.Tensor.value in
  let ctx = Context.set_values ctx tri.Tensor.value (mask_values (fun i o -> i >= o)) in
  let ctx = Context.run ctx routine in
  let after = Context.get_values ctx y.Tensor.value in
  Prefix_scan.set_enabled None;
  p "an NTDSL.init triangle (mutable host data) is left alone"
    (scans (Option.value_exn !captured).LL.llc = 0);
  p_all2 "the init triangle: the routine computes the prefix sums" before (cumsum ints)
    ~f:Float.equal;
  p_all2 "the init triangle, rewritten by the host: the same routine computes the suffix sums" after
    (suffix ( +. ) 0. ints) ~f:Float.equal
