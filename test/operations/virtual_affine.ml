(* Regression test for gh-ocannl-133 Stage B: virtualize injective affine producers (multi-symbol
   affine LHS indices).

   High-level lowering never produces these shapes directly in a controllable way, so -- like
   [virtual_shared_loop.ml] -- the cases are built directly as [Ir.Low_level.t] and run through the
   [Ll_test] harness (gh-ocannl-600), i.e. the same trace_node_facts -> virtual_llc -> cleanup ->
   simplify pipeline the backends use. We assert structurally on the optimized form (which producers
   virtualize, that no intermediate array read/setter survives, and which stay materialized).

   Cases: - structural affine match: producer [2*oh+wh] consumed at the same affine structure
   inlines with no intermediate buffer; - unit-coefficient solving at a plain iterator: producer
   [2*oh+wh] consumed at [t] inlines (the residual [oh] loop is kept and range-guarded); -
   triangular [(s1, s1+s2)]: unit-coefficient solving after pinning s1; - non-injective [i+j] (both
   ranges > 1) stays non-virtual, preserving a producer array read; - Stage A diagonal [i;i] still
   virtualizes (no regression).

   Structural pins say what the optimizer BUILT; virtualization rewrites what value a cell holds, so
   every case also has an EXECUTED leg (gh-ocannl-589): the very same [optimized] record is compiled
   through the [?prelowered] seam (gh-ocannl-562, worked out in [prelowered_seam.ml]), seeded, run,
   and its outputs checked against an OCaml reference, plus — where the producer virtualizes —
   against a second arm that re-specializes the same code with the producer's placement pre-decided
   [On_device]. That second arm is what pins the solved index and its range guard to the
   materialized reading of the same program: exactly the cells the scatter would have written must
   carry the scattered value, and the rest the init value. *)

open Base
open Ll_test
open Verdict.Claims

let mk = node_factory ~first_id:2000 ~dims:[| 6 |] ()
let optimize ?materialized llc = optimize ?materialized ~name:"virtual_affine" llc

(* The materialized arm's prefill of a zero-initialized producer: per cell, negative, so off the
   zero init and every producer value. A fresh device buffer is already zero, so without it a
   dropped [Zero_out] of the materialized node agrees with the virtual arm anyway
   (gh-ocannl-1625). *)
let prefill tn = (tn, Array.init (Tn.num_elems tn) ~f:(fun k -> -.(100. +. Float.of_int k)))

(* === Case 1: structural affine match === *)
let case_structural_match () =
  let tgt = mk ~dims:[| 6 |] "smatch" and out = mk ~dims:[| 6 |] "out1" in
  materialize out;
  let oh = sym () and wh = sym () and a = sym () and b = sym () in
  (* Injective + surjective scatter over [0, 6): no zero-init (lowering payoff). *)
  let prod = loop_n oh 3 (loop_n wh 2 (set tgt [| aff [ (2, oh); (1, wh) ] 0 |] (tag oh wh))) in
  let cons =
    loop_n a 3
      (loop_n b 2 (set out [| aff [ (2, a); (1, b) ] 0 |] (get tgt [| aff [ (2, a); (1, b) ] 0 |])))
  in
  let llc = seq prod cons in
  let o = optimize llc in
  p "structural-match producer virtual" (known_virtual o tgt);
  p "structural-match producer inlined (no array reads survive)" (count_get o tgt = 0);
  p "structural-match producer setter dropped" (count_set o tgt = 0);
  p "structural-match consumer setter kept" (count_set o out = 1);
  (* Same affine structure on both sides, the read's loops within the producer's (case 1b): bound
     pairwise, no range/equality guard. *)
  let wh, le, lt = count_guard_ops o in
  p "structural-match has no guard ops" (wh = 0 && le = 0 && lt = 0);
  (* The scatter is surjective onto [0, 6), so every cell carries the value of the producer
     iteration that pairwise-bound to it; the sentinel survives anywhere the inlined nest lost a
     cell the producer covered. *)
  let scattered = Array.init 6 ~f:(fun n -> Float.of_int (1 + (10 * (n / 2)) + (n % 2))) in
  let seed = [ (out, blank 6) ] and read = [ out ] in
  let virt = execute ~name:"va_structural" o ~seed ~read in
  let mat = execute ~name:"va_structural_mat" (optimize ~materialized:[ tgt ] llc) ~seed ~read in
  p "structural-match: every cell holds the value of the iteration bound to it"
    (same virt [ scattered ]);
  p "structural-match: virtual and materialized arms agree" (same virt mat)

(* === Case 1b: the structural match within the producer's bounds (gh-ocannl-1896) === The pairwise
   binding of case 1 evaluates the producer at the iteration [(u, v)] the read's symbols name. The
   producer map is injective (store time refuses it otherwise), so that iteration writes the read
   cell exactly when it lies in the producer's loops. A read whose inner symbol ranges wider --
   [x[2u + v]] for [v] in [lo, lo + 3], against [t] in [lo, lo + 1] -- reads cells the producer
   writes at [(u + 1, v - 2)], and the pairwise binding replayed [tag u v], a value no iteration
   wrote. Such a read must take unit-coefficient solving (a kept [s] loop and a range guard); a read
   inside the bounds keeps case 1's guard-free form. A single setter nest here: a second component
   would take the multi-component path ([block_components_two_terms] in
   [virtual_rejection_boundary]). [lo = 1] checks both the fallback's guard and the containment test
   against the loop's [from_], not [0]; [zero_init] checks that a [Zero_out], which is not a setter
   component, does not hide the read. *)
let case_structural_bounds ~name ~zero_init ~lo ~u_n ~v_n ~contained () =
  let tgt = mk ~dims:[| 6 |] (name ^ "_x") and out = mk ~dims:[| u_n; v_n |] (name ^ "_out") in
  materialize out;
  let s = sym () and t = sym () and u = sym () and v = sym () in
  let scatter =
    loop_n s 3
      (loop ~from_:lo ~upto:(lo + 1) t (set tgt [| aff [ (2, s); (1, t) ] (-lo) |] (tag s t)))
  in
  let prod = if zero_init then seq (zero tgt) scatter else scatter in
  let cons =
    loop_n u u_n
      (loop ~from_:lo
         ~upto:(lo + v_n - 1)
         v
         (set out [| iter u; aff [ (1, v) ] (-lo) |] (get tgt [| aff [ (2, u); (1, v) ] (-lo) |])))
  in
  let llc = seq prod cons in
  let o = optimize llc in
  p (name ^ " producer virtual") (known_virtual o tgt);
  p (name ^ " producer inlined (no array reads survive)") (count_get o tgt = 0);
  let wh, le, lt = count_guard_ops o in
  if contained then p (name ^ " keeps the guard-free pairwise binding") (wh = 0 && le = 0 && lt = 0)
  else
    p
      (name ^ " solves the inner symbol under a range guard (Where + Cmple lower + Cmplt upper)")
      (wh >= 1 && le >= 1 && lt >= 1);
  (* The reference replays the producer's writes, then reads them the consumer's way: [out[u, v -
     lo]] is cell [2u + v - lo]. *)
  let cells = Array.create ~len:6 Float.nan in
  for s = 0 to 2 do
    for t = lo to lo + 1 do
      cells.((2 * s) + t - lo) <- Float.of_int (1 + (10 * s) + t)
    done
  done;
  let expected =
    Array.concat (List.init u_n ~f:(fun u -> Array.init v_n ~f:(fun j -> cells.((2 * u) + j))))
  in
  let seed = [ (out, blank (u_n * v_n)) ] and read = [ out ] in
  let virt = execute ~name:("va_" ^ name) o ~seed ~read in
  let mat =
    execute
      ~name:("va_" ^ name ^ "_mat")
      (optimize ~materialized:[ tgt ] llc)
      ~seed:(if zero_init then prefill tgt :: seed else seed)
      ~read
  in
  p (name ^ ": every cell holds the value of the iteration that wrote it") (same virt [ expected ]);
  p (name ^ ": virtual and materialized arms agree") (same virt mat)

(* === Case 1c: the containment rule on normalized coordinates (gh-ocannl-1898) === A per-axis
   producer [x[h; e]] over dims [3; 2], read through a flattened [x[Sub_axis; 2u + v]]: the layouts
   differ, so the pair is re-expressed in its common frame, the producer position becoming [2h + e]
   -- the same coefficients as the read's. Normalization only composes coordinates, never splits
   one, so this structural match faces case 1b's containment rule like any other: with [v] wider
   than [e]'s loop the cell [2u + v] is written at [(u + 1, v - 2)], and binding [h := u], [e := v]
   would replay [tag u v], a value no iteration wrote. The wide read must take unit solving (a kept
   [h] loop, a range guard); the contained one keeps the guard-free pairwise form. *)
let case_normalized_bounds ~name ~u_n ~v_n ~contained () =
  let tgt = mk ~dims:[| 3; 2 |] (name ^ "_x") and out = mk ~dims:[| u_n; v_n |] (name ^ "_out") in
  materialize out;
  let h = sym () and e = sym () and u = sym () and v = sym () in
  let prod = loop_n h 3 (loop_n e 2 (set tgt [| iter h; iter e |] (tag h e))) in
  let cons =
    loop_n u u_n
      (loop_n v v_n
         (set out [| iter u; iter v |] (get tgt [| Idx.Sub_axis; aff [ (2, u); (1, v) ] 0 |])))
  in
  let llc = seq prod cons in
  let o = optimize llc in
  p (name ^ " producer virtual") (known_virtual o tgt);
  p (name ^ " producer inlined (no array reads survive)") (count_get o tgt = 0);
  let wh, le, lt = count_guard_ops o in
  if contained then p (name ^ " keeps the guard-free pairwise binding") (wh = 0 && le = 0 && lt = 0)
  else
    p
      (name ^ " solves the inner symbol under a range guard (Where + Cmple lower + Cmplt upper)")
      (wh >= 1 && le >= 1 && lt >= 1);
  (* Flat cell [2h + e] holds [tag h e]; [out[u, v]] reads flat cell [2u + v]. *)
  let cells = Array.init 6 ~f:(fun n -> Float.of_int (1 + (10 * (n / 2)) + (n % 2))) in
  let expected =
    Array.concat (List.init u_n ~f:(fun u -> Array.init v_n ~f:(fun v -> cells.((2 * u) + v))))
  in
  let seed = [ (out, blank (u_n * v_n)) ] and read = [ out ] in
  let virt = execute ~name:("va_" ^ name) o ~seed ~read in
  let mat =
    execute ~name:("va_" ^ name ^ "_mat") (optimize ~materialized:[ tgt ] llc) ~seed ~read
  in
  p (name ^ ": every cell holds the value of the iteration that wrote it") (same virt [ expected ]);
  p (name ^ ": virtual and materialized arms agree") (same virt mat)

(* === Case 1d: a zeroed, partially written node (gh-ocannl-1953) === A [Zero_out] satisfies the
   coverage rule at cells no iteration writes, so with one component nothing else stops a binding
   from replaying the producer outside its loop: [x[s]] for [s] in 0..1, read at [x[t]], computed
   the producer's expression at [t = 2] where the zeros are. Each shape takes a range guard and
   reads the zero; [guarded:false] is the other edge, a read inside the loop, which ends up
   guard-free (no guard is built, and the simplifier would fold one against the read's loop). Every
   producer varies with every symbol, and the materialized arm is prefilled so a dropped [Zero_out]
   cannot agree by accident. *)
let case_zeroed ~name ~dims ~prod ~read ~guarded ~expected () =
  let tgt = mk ~dims (name ^ "_x") and out = mk ~dims:[| Array.length expected |] (name ^ "_out") in
  materialize out;
  let llc = seq (zero tgt) (seq (prod tgt) (read tgt out)) in
  let o = optimize llc in
  p (name ^ " producer virtual") (known_virtual o tgt);
  p (name ^ " producer inlined (no array reads survive)") (count_get o tgt = 0);
  let wh, le, lt = count_guard_ops o in
  if guarded then
    (* Only the [Where]: the simplifier folds a bound the read's own loop proves, [0 <= t]. *)
    p (name ^ " reads under a guard") (wh >= 1)
  else p (name ^ " takes no guard: the read stays in the loop") (wh = 0 && le = 0 && lt = 0);
  let seed = [ (out, blank (Array.length expected)) ] and read = [ out ] in
  let virt = execute ~name:("va_" ^ name) o ~seed ~read in
  let mat =
    execute
      ~name:("va_" ^ name ^ "_mat")
      (optimize ~materialized:[ tgt ] llc)
      ~seed:(prefill tgt :: seed) ~read
  in
  p (name ^ ": each cell holds its write or the zero") (same virt [ expected ]);
  p (name ^ ": virtual and materialized arms agree") (same virt mat)

let zeroed_cases () =
  let flat_read n =
   fun tgt out ->
    let t = sym () in
    loop_n t n (set out [| iter t |] (get tgt [| iter t |]))
  in
  case "zeroed-partial"
    (case_zeroed ~name:"zeroed-partial" ~dims:[| 4 |]
       ~prod:(fun x ->
         let s = sym () in
         loop_n s 2 (set x [| iter s |] (tick s)))
       ~read:(flat_read 4) ~guarded:true ~expected:[| 1.; 2.; 0.; 0. |]);
  (* The loop need not start at 0: [t] in 0..2 lies within a width-3 loop only if the bound is taken
     from [0], which would drop the guard and compute [tick 0] at cell 0. *)
  case "zeroed-partial-offset"
    (case_zeroed ~name:"zeroed-partial-offset" ~dims:[| 4 |]
       ~prod:(fun x ->
         let s = sym () in
         loop ~from_:1 ~upto:3 s (set x [| iter s |] (tick s)))
       ~read:(flat_read 3) ~guarded:true ~expected:[| 0.; 2.; 3. |]);
  (* One axis partial: [k] wide of its loop takes the guard, [s] stays in its own. *)
  case "zeroed-partial-axis"
    (case_zeroed ~name:"zeroed-partial-axis" ~dims:[| 2; 3 |]
       ~prod:(fun x ->
         let s = sym () and k = sym () in
         loop_n s 2 (loop_n k 2 (set x [| iter s; iter k |] (tag s k))))
       ~read:(fun x out ->
         let a = sym () and b = sym () in
         loop_n a 2
           (loop_n b 3 (set out [| aff [ (3, a); (1, b) ] 0 |] (get x [| iter a; iter b |]))))
       ~guarded:true ~expected:[| 1.; 2.; 0.; 11.; 12.; 0. |]);
  (* A binding beside a fixed axis: the fixed position's own guard does not stop [s] past its
     loop. *)
  case "zeroed-fixed-axis"
    (case_zeroed ~name:"zeroed-fixed-axis" ~dims:[| 3; 2 |]
       ~prod:(fun x ->
         let s = sym () in
         loop_n s 2 (set x [| iter s; fixed 1 |] (tick s)))
       ~read:(fun x out ->
         let a = sym () and b = sym () in
         loop_n a 3
           (loop_n b 2 (set out [| aff [ (2, a); (1, b) ] 0 |] (get x [| iter a; iter b |]))))
       ~guarded:true ~expected:[| 0.; 1.; 0.; 2.; 0.; 0. |]);
  (* The zeroed reduction, the shape a [Zero_out] is usually for: an unbound [k] keeps its loop, and
     past [i]'s loop the accumulation must leave the zero. *)
  case "zeroed-reduction"
    (case_zeroed ~name:"zeroed-reduction" ~dims:[| 4 |]
       ~prod:(fun x ->
         let i = sym () and k = sym () in
         loop_n i 2 (loop_n k 3 (set x [| iter i |] (add (get x [| iter i |]) (tag i k)))))
       ~read:(flat_read 4) ~guarded:true ~expected:[| 6.; 36.; 0.; 0. |]);
  (* The single-term structural match: [x[2s]] read at [x[2t]] binds [s := t] past the loop. *)
  case "zeroed-structural"
    (case_zeroed ~name:"zeroed-structural" ~dims:[| 6 |]
       ~prod:(fun x ->
         let s = sym () in
         loop_n s 2 (set x [| aff [ (2, s) ] 0 |] (tick s)))
       ~read:(fun x out ->
         let t = sym () in
         loop_n t 3 (set out [| iter t |] (get x [| aff [ (2, t) ] 0 |])))
       ~guarded:true ~expected:[| 1.; 2.; 0. |]);
  (* The edge the guard must leave alone: the whole node written, and only the written part read. *)
  case "zeroed-whole"
    (case_zeroed ~name:"zeroed-whole" ~dims:[| 4 |]
       ~prod:(fun x ->
         let s = sym () in
         loop_n s 4 (set x [| iter s |] (tick s)))
       ~read:(flat_read 4) ~guarded:false ~expected:[| 1.; 2.; 3.; 4. |]);
  case "zeroed-read-inside"
    (case_zeroed ~name:"zeroed-read-inside" ~dims:[| 4 |]
       ~prod:(fun x ->
         let s = sym () in
         loop_n s 3 (set x [| iter s |] (tick s)))
       ~read:(flat_read 2) ~guarded:false ~expected:[| 1.; 2. |]);
  (* Pricing replays the same bindings at a synthetic read sweeping the node, with no call range:
     the guard is built, and what is priced must still be the walk's verdict once the pipeline folds
     it against the reader's static ranges -- a whole-node write prices no guard, a partial one
     does. *)
  let synthetic_guards ~dims ~write =
    let x = mk ~dims "synth_x" in
    let s = sym () in
    let computations =
      [ (Some (write s), loop_n s 4 (set x (write s) (tick s))); (None, zero x) ]
    in
    match
      LL.instantiate_at_synthetic_read ~placements:(LL.empty_optimize_ctx ()).LL.placements
        ~static_indices:[] x computations
    with
    | Error code -> failwith code
    | Ok body ->
        let wh = ref 0 in
        walk body ~on_ternop:(function Ops.Where -> Int.incr wh | _ -> ());
        !wh
  in
  case "zeroed-synthetic" (fun () ->
      p "zeroed-synthetic: a whole-node write prices no guard"
        (synthetic_guards ~dims:[| 4 |] ~write:(fun s -> [| iter s |]) = 0);
      p "zeroed-synthetic: a partial write prices one"
        (synthetic_guards ~dims:[| 6 |] ~write:(fun s -> [| iter s |]) >= 1);
      p "zeroed-synthetic: a flattened whole-node write prices no guard"
        (synthetic_guards ~dims:[| 2; 2 |] ~write:(fun s -> [| Idx.Sub_axis; iter s |]) = 0))

(* === Case 2: unit-coefficient solving at a plain iterator === *)
let case_unit_solve_plain () =
  let tgt = mk ~dims:[| 6 |] "usolve" and out = mk ~dims:[| 6 |] "out2" in
  materialize out;
  let oh = sym () and wh = sym () and t = sym () in
  let prod = loop_n oh 3 (loop_n wh 2 (set tgt [| aff [ (2, oh); (1, wh) ] 0 |] (tag oh wh))) in
  let cons = loop_n t 6 (set out [| iter t |] (get tgt [| iter t |])) in
  let llc = seq prod cons in
  let o = optimize llc in
  p "unit-solve(plain) producer virtual" (known_virtual o tgt);
  p "unit-solve(plain) producer inlined (no array reads survive)" (count_get o tgt = 0);
  p "unit-solve(plain) producer setter dropped" (count_set o tgt = 0);
  p "unit-solve(plain) consumer setter kept" (count_set o out = 1);
  (* Solving [wh = t - 2*oh] keeps the [oh] loop and range-guards [0 <= t-2*oh < 2]: a [Where] over
     an eager 0/1 product of a [Cmple] lower bound and a [Cmplt] upper bound. *)
  let wh, le, lt = count_guard_ops o in
  p "unit-solve(plain) emits a range guard (Where + Cmple lower + Cmplt upper)"
    (wh >= 1 && le >= 1 && lt >= 1);
  (* The residual [oh] loop folds over the producer's fiber, so the range guard is what keeps each
     [t] to the single [oh] that scattered it. Because the producer value identifies its iteration,
     a bound admitting a neighbouring [oh] lands that neighbour's value here instead. *)
  let scattered = Array.init 6 ~f:(fun n -> Float.of_int (1 + (10 * (n / 2)) + (n % 2))) in
  let seed = [ (out, blank 6) ] and read = [ out ] in
  let virt = execute ~name:"va_unit_solve" o ~seed ~read in
  let mat = execute ~name:"va_unit_solve_mat" (optimize ~materialized:[ tgt ] llc) ~seed ~read in
  p "unit-solve(plain): every cell holds the value of the one iteration that scattered it"
    (same virt [ scattered ]);
  p "unit-solve(plain): virtual and materialized arms agree" (same virt mat)

(* === Case 3: triangular (s1, s1 + s2), unit-coefficient solving after pinning s1 === *)
let case_triangular () =
  let tgt = mk ~dims:[| 3; 4 |] "tri" and out = mk ~dims:[| 3; 4 |] "out3" in
  materialize out;
  let s1 = sym () and s2 = sym () and a = sym () and b = sym () in
  (* Triangular map is injective but not surjective, so it carries a zero-init. *)
  let prod =
    seq (zero tgt)
      (loop_n s1 3 (loop_n s2 2 (set tgt [| iter s1; aff [ (1, s1); (1, s2) ] 0 |] (tag s1 s2))))
  in
  let cons =
    loop_n a 3 (loop_n b 4 (set out [| iter a; iter b |] (get tgt [| iter a; iter b |])))
  in
  let llc = seq prod cons in
  let o = optimize llc in
  p "triangular producer virtual" (known_virtual o tgt);
  p "triangular producer inlined (no array reads survive)" (count_get o tgt = 0);
  p "triangular consumer setter kept" (count_set o out = 1);
  (* s2 = b - a is solved and range-guarded [0 <= b-a < 2]. *)
  let wh, le, lt = count_guard_ops o in
  p "triangular emits a range guard (Where + Cmple lower + Cmplt upper)"
    (wh >= 1 && le >= 1 && lt >= 1);
  (* The scatter is injective but not surjective, so the guard has to separate the band it covers
     from the cells that keep the zero-init, and within the band the solved [s2 = b - a] has to pick
     the right producer iteration: [out.(a, b)] is [1 + 10a + (b - a)] exactly on [0 <= b - a < 2],
     and 0 elsewhere. *)
  let expected =
    Array.init 12 ~f:(fun n ->
        let a = n / 4 and b = n % 4 in
        if b - a >= 0 && b - a < 2 then Float.of_int (1 + (10 * a) + (b - a)) else 0.)
  in
  let seed = [ (out, blank 12) ] and read = [ out ] in
  let virt = execute ~name:"va_triangular" o ~seed ~read in
  let mat =
    execute ~name:"va_triangular_mat"
      (optimize ~materialized:[ tgt ] llc)
      ~seed:(prefill tgt :: seed) ~read
  in
  p "triangular: executed values are the scattered band over the zero-init" (same virt [ expected ]);
  p "triangular: virtual and materialized arms agree" (same virt mat)

(* === Case 4: non-injective i+j (both ranges > 1) stays non-virtual === *)
let case_noninjective () =
  let tgt = mk ~dims:[| 5 |] "ni" and out = mk ~dims:[| 5 |] "out4" in
  materialize out;
  let i = sym () and j = sym () and a = sym () and b = sym () in
  let prod = loop_n i 3 (loop_n j 3 (set tgt [| aff [ (1, i); (1, j) ] 0 |] (c 1.))) in
  let cons =
    loop_n a 3
      (loop_n b 3 (set out [| aff [ (1, a); (1, b) ] 0 |] (get tgt [| aff [ (1, a); (1, b) ] 0 |])))
  in
  let o = optimize (seq prod cons) in
  (* i+j with both ranges > 1 is not injective: the dropped producer loops fold over a fiber, so the
     producer must stay materialized (the reason is the injectivity soundness line). *)
  p "non-injective producer stays non-virtual" (known_non_virtual o tgt);
  p "non-injective producer array read preserved" (count_get o tgt >= 1);
  (* Staying materialized is already the safe reading, so this case has no second arm - the executed
     leg pins that the fiber the producer folds over reaches the consumer through a buffer. The
     producer value stays constant here, unlike the guarded cases: nothing selects an iteration, so
     a per-iteration value would only pin which write of the fiber lands last, which is the loop
     order rather than the property under test. Only [out] is read back: [tgt] is non-virtual but
     unobservable, so the pipeline places it [Local] (routine-scoped scratch with no context
     buffer). The second fact pins that reading it back raises (gh-ocannl-599) rather than answering
     with whatever a host write left behind — which is how this case reported an array of sentinels
     when its executed leg was first written. *)
  let ctx = run ~name:"va_noninjective" o ~seed:[ (out, blank 5) ] in
  p "non-injective: the scattered value reached the consumer through the buffer"
    (close (Context.get_values ctx out) (Array.create ~len:5 1.));
  p "non-injective: the Local producer refuses host readback"
    (refused_as_local (fun () -> Context.get_values ctx tgt))

(* === Case 5: Stage A diagonal [i;i] still virtualizes (no regression) === *)
let case_stage_a_diagonal () =
  let d = mk ~dims:[| 3; 3 |] "diag" and out = mk ~dims:[| 3; 3 |] "out5" in
  materialize out;
  let i = sym () and a = sym () and b = sym () in
  let prod = seq (zero d) (loop_n i 3 (set d [| iter i; iter i |] (add (c 4.) (embed i)))) in
  let cons = loop_n a 3 (loop_n b 3 (set out [| iter a; iter b |] (get d [| iter a; iter b |]))) in
  let llc = seq prod cons in
  let o = optimize llc in
  p "stage-a diagonal producer virtual" (known_virtual o d);
  p "stage-a diagonal inlined (no array reads survive)" (count_get o d = 0);
  let expected =
    Array.init 9 ~f:(fun n -> if n / 3 = n % 3 then Float.of_int (4 + (n / 3)) else 0.)
  in
  let seed = [ (out, blank 9) ] and read = [ out ] in
  let virt = execute ~name:"va_stage_a" o ~seed ~read in
  let mat =
    execute ~name:"va_stage_a_mat"
      (optimize ~materialized:[ d ] llc)
      ~seed:(prefill d :: seed) ~read
  in
  p "stage-a diagonal: executed values are the diagonal over the zero-init" (same virt [ expected ]);
  p "stage-a diagonal: virtual and materialized arms agree" (same virt mat)

(* Each row runs through [case], so a row that raises is a failed claim naming it, and the rows
   after it still run (gh-ocannl-1084; [footprint_materialization] is the exemplar). *)
let () =
  case "structural-match" case_structural_match;
  case "structural-wide"
    (case_structural_bounds ~name:"structural-wide" ~zero_init:false ~lo:0 ~u_n:2 ~v_n:4
       ~contained:false);
  case "structural-wide-zero"
    (case_structural_bounds ~name:"structural-wide-zero" ~zero_init:true ~lo:0 ~u_n:2 ~v_n:4
       ~contained:false);
  case "structural-wide-offset"
    (case_structural_bounds ~name:"structural-wide-offset" ~zero_init:false ~lo:1 ~u_n:2 ~v_n:4
       ~contained:false);
  case "structural-contained-offset"
    (case_structural_bounds ~name:"structural-contained-offset" ~zero_init:false ~lo:1 ~u_n:3 ~v_n:2
       ~contained:true);
  case "normalized-wide"
    (case_normalized_bounds ~name:"normalized-wide" ~u_n:2 ~v_n:4 ~contained:false);
  case "normalized-contained"
    (case_normalized_bounds ~name:"normalized-contained" ~u_n:3 ~v_n:2 ~contained:true);
  zeroed_cases ();
  case "unit-solve(plain)" case_unit_solve_plain;
  case "triangular" case_triangular;
  case "non-injective" case_noninjective;
  case "stage-a diagonal" case_stage_a_diagonal;
  Stdio.printf "%!"
