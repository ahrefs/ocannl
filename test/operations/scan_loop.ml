(* gh-ocannl-696: the loop-carried recurrence construct [Ir.Low_level.Scan_loop], pinned on
   hand-built IR through [Ll_test] -- the shape no [Assignments] lowering emits yet.

   What a scan is for: a loop whose iteration [i] needs state the previous iteration produced (a
   running sum, an online-softmax [(max, normalizer)] pair, a top-k frontier) and whose per-cell
   value therefore depends on the WHOLE prefix, not on the index alone. A plain [For_loop] cannot
   say that: its iterator-on-the-left-hand-side certifies both an affine address map and a
   self-contained body instance, and every analysis that parallelizes, reorders or recomputes
   iterations leans on the second half. The construct declares the recurrence instead -- carried
   scalars as rotated [prev]/[next] scope locals over a virtual state node, a direction, and a body
   that writes each [next] exactly once -- so the declaration, not a comment, is what keeps the
   schedule ops and the virtualizer honest.

   Legs, each an executed leg where a value can discriminate (AGENTS.md: a value-rewriting construct
   needs executed parity, not only structural pins):

   1. Inclusive forward cumsum over a discriminating input; the construct survives [optimize] as one
   [Scan_loop], and the state node reaches the routine as neither an input nor an output. 2. The
   same scan [Backward]: suffix sums. 3. Two carried scalars with the online-softmax rescaling
   recurrence -- #483's shape -- checked against a host reference within a tolerance (device floats
   stay off the golden). 4. A per-row scan under an outer loop, Serial and hand-annotated [Grid]:
   the enclosing loop keeps its hardware mapping, since the carried state is per-iteration scratch
   of that loop. 5. The placement contract: a scan output the program never declared materialized is
   refused as a virtualization candidate ([Non_virtual 148]) rather than recomputed per cell at a
   downstream read, and the downstream consumer computes from the materialized trajectory. 6. The
   well-formedness contract, at both pipeline gates: a guarded, missing or repeated [next] write, a
   [next] read before its write, an empty range, a written [prev], a materialized or undeclared
   state node, a shared id, an init reading carried state or the scan index, and a pair over two
   nodes are each refused by name -- by [optimize] on the way in, and by backend codegen when the
   malformed scan is handed straight through the [?prelowered] seam. 7. Schedule opacity: an op
   naming the scan's own index, or a loop nested inside its body, declines with the usual
   no-such-loop refusal. 8. Digest identity: the canonical rendering tells [Forward] from [Backward]
   and a swapped carried pair from the original, while a fresh lowering of the same program renders
   identically -- the schedule cache's replay key sees the recurrence and only the recurrence. *)

open Base
open Stdio
open Ll_test
open Verdict.Claims
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Ops = Ir.Ops
module Idx = Ir.Indexing
module CR = LL.Canonical_render
module Schedule = Ir.Schedule

let n = 7
let mk = node_factory ~first_id:10700 ~dims:[| n |] ()

(* The state node of a carried scalar: a virtual node the pair of scope ids is minted over -- its
   only roles are the state's name and precision. *)
let state label =
  let st = mk ~dims:[| 1 |] label in
  virtualize st;
  st

let sub a b = binop Ops.Sub a b
let maxf a b = binop Ops.Max a b
let exp_ a : LL.scalar_t = LL.Unop (Ops.Exp, (a, single))
let is_scan = function LL.Scan_loop _ -> true | _ -> false

(* Discriminating input: distinct, off the zero-init, and off the sentinel. *)
let xs = Array.init n ~f:(fun k -> Float.of_int (k + 1))

let prefix_sums =
  Array.init n ~f:(fun i -> Array.fold (Array.sub xs ~pos:0 ~len:(i + 1)) ~init:0. ~f:( +. ))

let suffix_sums =
  Array.init n ~f:(fun i -> Array.fold (Array.sub xs ~pos:i ~len:(n - i)) ~init:0. ~f:( +. ))

(* Whether [f] is refused by the scan contract specifically. *)
let rejected f =
  try
    ignore (f ());
    false
  with Invalid_argument msg -> String.is_substring msg ~substring:"validate_scan_loops"

(* Whether a schedule op declined for want of a [For_loop] to target. *)
let declines f =
  try
    ignore (f ());
    false
  with Invalid_argument msg -> String.is_substring msg ~substring:"no For_loop with index"

(* --- Legs 1 and 2: cumulative sums, both directions. --- *)

(* [out[i] = sum of x over the prefix (or suffix) ending at i], as one scan carrying the running
   sum. Returns the nodes and the scan index alongside the optimized record. *)
let build_cumsum ~direction ~label =
  let x = mk (label ^ "_x") and out = mk (label ^ "_out") and st = state (label ^ "_s") in
  materialize x;
  materialize out;
  let i = sym () in
  let s = carry ~init:(c 0.) st in
  let llc =
    scan ~direction ~upto:(n - 1) i ~carried:[ s ]
      (seq (set_next s (add (prev s) (get x [| iter i |]))) (set_at out (iter i) (next s)))
  in
  (x, out, st, i, llc)

let cumsum_leg ~direction ~label ~want =
  let x, out, st, i, llc = build_cumsum ~direction ~label in
  let o = optimize ~name:label llc in
  let ctx, routine = link ~name:label o in
  let ctx = run_linked (ctx, routine) ~seed:[ (x, xs); (out, Array.create ~len:n sentinel) ] in
  let got = Context.get_values ctx out in
  p_all2 (label ^ ": every cell equals the host cumulative sum") got want ~f:Float.equal;
  p
    (label ^ ": the scan survives optimize as exactly one Scan_loop")
    (count_stmt ~f:is_scan o.LL.llc = 1);
  p
    (label ^ ": the state node is neither an input nor an output of the linked routine")
    ((not (Set.mem routine.Context.inputs st)) && not (Set.mem routine.Context.outputs st));
  p
    (label ^ ": the input and the trajectory are the routine's only input and output")
    (Set.equal routine.Context.inputs (Set.singleton (module Tn) x)
    && Set.equal routine.Context.outputs (Set.singleton (module Tn) out));
  p
    (label ^ ": the state node stays virtual after optimize")
    (not (Tn.Placements.known_non_virtual o.LL.optimize_ctx.placements st));
  (o, i)

let () =
  eprintf "backend: %s (not part of the golden)\n%!"
    (Utils.get_global_arg ~arg_name:"backend" ~default:"cc");
  printf "--- leg 1: inclusive forward cumsum ---\n";
  let o, _ = cumsum_leg ~direction:LL.Forward ~label:"sl_fwd" ~want:prefix_sums in
  printf "optimized IR:\n";
  PPrint.ToChannel.pretty 1.0 100 stdout (LL.to_doc ~name:"sl_fwd" () o.LL.llc);
  printf "\n";
  printf "--- leg 2: backward cumsum (suffix sums) ---\n";
  ignore (cumsum_leg ~direction:LL.Backward ~label:"sl_bwd" ~want:suffix_sums : LL.optimized * _)

(* --- Leg 3: the online-softmax recurrence, two carried scalars. --- *)

let () =
  printf "--- leg 3: online softmax (running max, rescaled normalizer) ---\n";
  let scores = [| 0.5; 2.0; -1.0; 3.0; 1.5; -2.0; 0.0 |] in
  let x = mk "sl_sm_x" and out = mk "sl_sm_out" in
  let m_st = state "sl_sm_m" and l_st = state "sl_sm_l" in
  materialize x;
  materialize out;
  let i = sym () in
  let m = carry ~init:(c (-1e30)) m_st and l = carry ~init:(c 0.) l_st in
  let xi = get x [| iter i |] in
  let llc =
    scan ~upto:(n - 1) i ~carried:[ m; l ]
      (seq
         (set_next m (maxf (prev m) xi))
         (seq
            (set_next l
               (add (mul (prev l) (exp_ (sub (prev m) (next m)))) (exp_ (sub xi (next m)))))
            (set_at out (iter i) (next l))))
  in
  let o = optimize ~name:"sl_softmax" llc in
  let got =
    List.hd_exn
      (execute ~name:"sl_softmax" o
         ~seed:[ (x, scores); (out, Array.create ~len:n sentinel) ]
         ~read:[ out ])
  in
  (* The host reference runs the same recurrence in double precision. *)
  let want =
    let m = ref (-1e30) and l = ref 0. in
    Array.map scores ~f:(fun s ->
        let m' = Float.max !m s in
        l := (!l *. Float.exp (!m -. m')) +. Float.exp (s -. m');
        m := m';
        !l)
  in
  let final_direct =
    let mx = Array.fold scores ~init:Float.neg_infinity ~f:Float.max in
    Array.fold scores ~init:0. ~f:(fun acc s -> acc +. Float.exp (s -. mx))
  in
  eprintf "running normalizers: %s (not part of the golden)\n%!"
    (String.concat ~sep:" " (Array.to_list (Array.map got ~f:(Printf.sprintf "%.9g"))));
  let close g w = Float.(abs (g -. w) <= 1e-5 *. max 1. (abs w)) in
  p_all2 "every running normalizer is within 1e-5 relative of the host recurrence" got want ~f:close;
  p "the final normalizer equals the direct sum of exp(x - max x) within 1e-5 relative"
    (close got.(n - 1) final_direct);
  p "the trajectory is strictly increasing, as a sum of positive terms rescaled to a growing max"
    (Array.for_alli got ~f:(fun k v -> k = 0 || Float.( > ) v got.(k - 1)));
  p "both carried scalars are counted, and no other scan is in the routine"
    (count_stmt ~f:is_scan o.LL.llc = 1
    && count_stmt ~f:(function LL.Set_local _ -> true | _ -> false) o.LL.llc = 2)

(* --- Leg 4: a per-row scan under an enclosing loop, Serial and Grid. --- *)

let () =
  printf "--- leg 4: per-row scan under an outer loop, Serial vs Grid ---\n";
  let rows = 3 and cols = 5 in
  let x = mk ~dims:[| rows; cols |] "sl_rows_x" and st = state "sl_rows_s" in
  materialize x;
  let seed_x =
    Array.init (rows * cols) ~f:(fun k -> Float.of_int (1 + (10 * (k / cols)) + (k % cols)))
  in
  let want =
    Array.init (rows * cols) ~f:(fun k ->
        let r = k / cols and i = k % cols in
        let acc = ref 0. in
        for j = 0 to i do
          acc := !acc +. seed_x.((r * cols) + j)
        done;
        !acc)
  in
  let build ~axis ~label =
    let out = mk ~dims:[| rows; cols |] (label ^ "_out") in
    materialize out;
    let r = sym () and i = sym () in
    let s = carry ~init:(c 0.) st in
    let llc =
      LL.For_loop
        {
          index = r;
          from_ = 0;
          to_ = rows - 1;
          axis;
          body =
            scan ~upto:(cols - 1) i ~carried:[ s ]
              (seq
                 (set_next s (add (prev s) (get x [| iter r; iter i |])))
                 (set out [| iter r; iter i |] (next s)));
        }
    in
    let o = optimize ~name:label llc in
    List.hd_exn
      (execute ~name:label o
         ~seed:[ (x, seed_x); (out, Array.create ~len:(rows * cols) sentinel) ]
         ~read:[ out ])
  in
  let serial = build ~axis:LL.Serial ~label:"sl_rows_serial" in
  let grid = build ~axis:LL.Grid ~label:"sl_rows_grid" in
  p_all2 "serial outer loop: every cell equals the host per-row prefix sum" serial want
    ~f:Float.equal;
  p_all2 "Grid-annotated outer loop: every cell agrees with the serial twin" grid serial
    ~f:Float.equal

(* --- Leg 5: the placement contract for a scan's output. --- *)

let () =
  printf "--- leg 5: an undeclared scan output materializes (Non_virtual 148) ---\n";
  let x = mk "sl_pl_x" and mid = mk "sl_pl_mid" and y = mk "sl_pl_y" and st = state "sl_pl_s" in
  materialize x;
  materialize y;
  let i = sym () and j = sym () in
  let s = carry ~init:(c 0.) st in
  let llc =
    seq
      (scan ~upto:(n - 1) i ~carried:[ s ]
         (seq (set_next s (add (prev s) (get x [| iter i |]))) (set_at mid (iter i) (next s))))
      (loop_n j n (set_at y (iter j) (mul (get mid [| iter j |]) (c 2.))))
  in
  let o, got =
    optimize_and_execute ~name:"sl_placement" llc
      ~seed:[ (x, xs); (y, Array.create ~len:n sentinel) ]
      ~read:[ y ]
  in
  p "the scan output was refused as a virtualization candidate with provenance 148"
    (Option.equal Int.equal (rejection_code o mid) (Some 148));
  p "the scan output is decided non-virtual, so the consumer reads its buffer"
    (Tn.Placements.known_non_virtual o.LL.optimize_ctx.placements mid);
  p "the consumer's read of the trajectory survives as a buffer read, not an inlined scope"
    (count_scalar ~f:(function LL.Get (tn, _) -> Tn.equal tn mid | _ -> false) o.LL.llc = 1
    && count_scalar ~f:(function LL.Local_scope _ -> true | _ -> false) o.LL.llc = 0);
  p_all2 "the consumer computes twice the prefix sums" (List.hd_exn got)
    (Array.map prefix_sums ~f:(fun v -> 2. *. v))
    ~f:Float.equal

(* --- Leg 5b: a scan feeding a candidate's value through a scope local. --- *)

let () =
  printf "--- leg 5b: a candidate whose computation contains a scan materializes (148) ---\n";
  (* [x[i]] is computed through an inlined scope whose body runs a scan over [k] and leaves the
     row's prefix total in the scope local; [x] is left undecided and read by [y]. Replaying the
     scope at [y]'s read would have to replay the scan, which the inline filter cannot do, so the
     candidate is refused where its computation is captured and [y] reads the buffer. *)
  let rows = 3 and cols = 4 in
  let w = mk ~dims:[| rows; cols |] "sl_vs_w" and x = mk ~dims:[| rows |] "sl_vs_x" in
  let y = mk ~dims:[| rows |] "sl_vs_y" and v = state "sl_vs_v" and st = state "sl_vs_s" in
  materialize w;
  materialize y;
  let i = sym () and k = sym () and j = sym () in
  let id = LL.get_scope v in
  let s = carry ~init:(c 0.) st in
  let scope =
    LL.Local_scope
      {
        id;
        orig_indices = [| iter i |];
        mint = LL.Inlined_computation;
        body =
          seq
            (LL.Set_local (id, c 0.))
            (scan ~upto:(cols - 1) k ~carried:[ s ]
               (seq
                  (set_next s (add (prev s) (get w [| iter i; iter k |])))
                  (LL.Set_local (id, next s))));
      }
  in
  let llc =
    seq
      (loop_n i rows (set_at x (iter i) scope))
      (loop_n j rows (set_at y (iter j) (get x [| iter j |])))
  in
  let seed_w =
    Array.init (rows * cols) ~f:(fun q -> Float.of_int (1 + (10 * (q / cols)) + (q % cols)))
  in
  let o, got =
    optimize_and_execute ~name:"sl_value_scan" llc
      ~seed:[ (w, seed_w); (y, Array.create ~len:rows sentinel) ]
      ~read:[ y ]
  in
  p "the candidate computed through a scan was refused with provenance 148"
    (Option.equal Int.equal (rejection_code o x) (Some 148));
  p "the scan survives as the candidate's materialized producer" (count_stmt ~f:is_scan o.LL.llc = 1);
  p_all2 "the consumer reads each row's total from the buffer" (List.hd_exn got)
    (Array.init rows ~f:(fun r ->
         Array.fold (Array.sub seed_w ~pos:(r * cols) ~len:cols) ~init:0. ~f:( +. )))
    ~f:Float.equal

(* --- Leg 6: the well-formedness contract at both gates. --- *)

let () =
  printf "--- leg 6: malformed scans are refused by name, at both pipeline gates ---\n";
  let fresh label =
    let x = mk (label ^ "_x") and out = mk (label ^ "_out") in
    materialize x;
    materialize out;
    (x, out, sym ())
  in
  let entry label mk_llc = rejected (fun () -> optimize ~name:label (mk_llc (fresh label))) in
  p "a next written only under a guard is refused"
    (entry "sl_bad_guarded" (fun (x, out, i) ->
         let s = carry ~init:(c 0.) (state "sl_bad_guarded_s") in
         scan ~upto:(n - 1) i ~carried:[ s ]
           (seq
              (if_idx (lt (embed i) (ic 3)) (set_next s (add (prev s) (get x [| iter i |]))))
              (set_at out (iter i) (prev s)))));
  p "a next never written is refused"
    (entry "sl_bad_unwritten" (fun (x, out, i) ->
         let s = carry ~init:(c 0.) (state "sl_bad_unwritten_s") in
         scan ~upto:(n - 1) i ~carried:[ s ]
           (set_at out (iter i) (add (prev s) (get x [| iter i |])))));
  p "a next written twice is refused"
    (entry "sl_bad_twice" (fun (x, out, i) ->
         let s = carry ~init:(c 0.) (state "sl_bad_twice_s") in
         scan ~upto:(n - 1) i ~carried:[ s ]
           (seq
              (set_next s (add (prev s) (get x [| iter i |])))
              (seq (set_next s (next s)) (set_at out (iter i) (next s))))));
  p "a prev written in the body is refused"
    (entry "sl_bad_prev" (fun (x, out, i) ->
         let s = carry ~init:(c 0.) (state "sl_bad_prev_s") in
         scan ~upto:(n - 1) i ~carried:[ s ]
           (seq
              (set_next s (add (prev s) (get x [| iter i |])))
              (seq (LL.Set_local (s.LL.prev, c 0.)) (set_at out (iter i) (next s))))));
  p "a materialized state node is refused"
    (entry "sl_bad_mat" (fun (x, out, i) ->
         let st = mk ~dims:[| 1 |] "sl_bad_mat_s" in
         materialize st;
         let s = carry ~init:(c 0.) st in
         scan ~upto:(n - 1) i ~carried:[ s ]
           (seq (set_next s (add (prev s) (get x [| iter i |]))) (set_at out (iter i) (next s)))));
  p "an init reading carried state is refused"
    (entry "sl_bad_init" (fun (x, out, i) ->
         let a = carry ~init:(c 0.) (state "sl_bad_init_a") in
         let b = carry ~init:(prev a) (state "sl_bad_init_b") in
         scan ~upto:(n - 1) i ~carried:[ a; b ]
           (seq
              (set_next a (add (prev a) (get x [| iter i |])))
              (seq (set_next b (add (prev b) (next a))) (set_at out (iter i) (next b))))));
  p "a next read by a statement before the one that writes it is refused"
    (entry "sl_bad_order" (fun (x, out, i) ->
         let a = carry ~init:(c 0.) (state "sl_bad_order_a") in
         let b = carry ~init:(c 0.) (state "sl_bad_order_b") in
         scan ~upto:(n - 1) i ~carried:[ a; b ]
           (seq
              (set_next b (add (prev b) (next a)))
              (seq (set_next a (add (prev a) (get x [| iter i |]))) (set_at out (iter i) (next b))))));
  p "a next read by its own defining statement is refused"
    (entry "sl_bad_self" (fun (x, out, i) ->
         let s = carry ~init:(c 0.) (state "sl_bad_self_s") in
         scan ~upto:(n - 1) i ~carried:[ s ]
           (seq (set_next s (add (next s) (get x [| iter i |]))) (set_at out (iter i) (next s)))));
  p "an id shared by two carried pairs is refused"
    (entry "sl_bad_dup" (fun (x, out, i) ->
         let s = carry ~init:(c 0.) (state "sl_bad_dup_s") in
         scan ~upto:(n - 1) i ~carried:[ s; s ]
           (seq (set_next s (add (prev s) (get x [| iter i |]))) (set_at out (iter i) (next s)))));
  p "an init mentioning the scan index is refused"
    (entry "sl_bad_idx" (fun (x, out, i) ->
         let s = carry ~init:(embed i) (state "sl_bad_idx_s") in
         scan ~upto:(n - 1) i ~carried:[ s ]
           (seq (set_next s (add (prev s) (get x [| iter i |]))) (set_at out (iter i) (next s)))));
  p "a state node left undeclared (neither virtual nor materialized) is refused"
    (entry "sl_bad_undecl" (fun (x, out, i) ->
         let s = carry ~init:(c 0.) (mk ~dims:[| 1 |] "sl_bad_undecl_s") in
         scan ~upto:(n - 1) i ~carried:[ s ]
           (seq (set_next s (add (prev s) (get x [| iter i |]))) (set_at out (iter i) (next s)))));
  p "a Declare_local of a carried id inside the body is refused"
    (entry "sl_bad_shadow" (fun (x, out, i) ->
         let s = carry ~init:(c 0.) (state "sl_bad_shadow_s") in
         scan ~upto:(n - 1) i ~carried:[ s ]
           (seq
              (LL.Declare_local { id = s.LL.prev; needs_init = true })
              (seq (set_next s (add (prev s) (get x [| iter i |]))) (set_at out (iter i) (next s))))));
  p "a Local_scope binder reusing a carried id inside the body is refused"
    (entry "sl_bad_scope" (fun (x, out, i) ->
         let s = carry ~init:(c 0.) (state "sl_bad_scope_s") in
         let shadow =
           LL.Local_scope
             {
               id = s.LL.next;
               body = LL.Set_local (s.LL.next, get x [| iter i |]);
               orig_indices = [| iter i |];
               mint = LL.Inlined_computation;
             }
         in
         scan ~upto:(n - 1) i ~carried:[ s ]
           (seq (set_next s (add (prev s) shadow)) (set_at out (iter i) (next s)))));
  p "a tensor-buffer access to the state node elsewhere in the routine is refused"
    (entry "sl_bad_buffer" (fun (x, out, i) ->
         let st = state "sl_bad_buffer_s" in
         let s = carry ~init:(c 0.) st in
         let j = sym () in
         seq
           (scan ~upto:(n - 1) i ~carried:[ s ]
              (seq (set_next s (add (prev s) (get x [| iter i |]))) (set_at out (iter i) (next s))))
           (loop_n j n (set_at out (iter j) (get st [| fixed 0 |])))));
  p "a read of a carried local after the scan is refused"
    (entry "sl_bad_after" (fun (x, out, i) ->
         let s = carry ~init:(c 0.) (state "sl_bad_after_s") in
         seq
           (scan ~upto:(n - 1) i ~carried:[ s ]
              (seq (set_next s (add (prev s) (get x [| iter i |]))) (set_at out (iter i) (next s))))
           (set_at out (fixed 0) (next s))));
  p "a next read early through a Tile_mma fallback is refused"
    (entry "sl_bad_mma" (fun (x, out, i) ->
         let s = carry ~init:(c 0.) (state "sl_bad_mma_s") in
         let d = mk ~dims:[| 2; 2 |] "sl_bad_mma_d" in
         materialize d;
         let tile =
           LL.Tile_mma
             {
               d = (d, [| fixed 0; fixed 0 |]);
               a = (d, [| fixed 0; fixed 0 |]);
               b = (d, [| fixed 0; fixed 0 |]);
               ta = false;
               tb = false;
               m = 2;
               n = 2;
               k = 2;
               ldd = 2;
               lda = 2;
               ldb = 2;
               lane = sym ();
               tile = None;
               fallback = set d [| fixed 0; fixed 0 |] (next s);
             }
         in
         scan ~upto:(n - 1) i ~carried:[ s ]
           (seq tile
              (seq (set_next s (add (prev s) (get x [| iter i |]))) (set_at out (iter i) (next s))))));
  p "a tensor-buffer access to the state node inside a Tile_mma fallback is refused"
    (entry "sl_bad_mma_st" (fun (x, out, i) ->
         let st = state "sl_bad_mma_st_s" in
         let s = carry ~init:(c 0.) st in
         let d = mk ~dims:[| 2; 2 |] "sl_bad_mma_st_d" in
         materialize d;
         let tile =
           LL.Tile_mma
             {
               d = (d, [| fixed 0; fixed 0 |]);
               a = (d, [| fixed 0; fixed 0 |]);
               b = (d, [| fixed 0; fixed 0 |]);
               ta = false;
               tb = false;
               m = 2;
               n = 2;
               k = 2;
               ldd = 2;
               lda = 2;
               ldb = 2;
               lane = sym ();
               tile = None;
               fallback = set d [| fixed 0; fixed 0 |] (get st [| fixed 0 |]);
             }
         in
         scan ~upto:(n - 1) i ~carried:[ s ]
           (seq
              (set_next s (add (prev s) (get x [| iter i |])))
              (seq (set_at out (iter i) (next s)) tile))));
  p "a Staged_compilation inside the scan body is refused"
    (entry "sl_bad_staged" (fun (x, out, i) ->
         let s = carry ~init:(c 0.) (state "sl_bad_staged_s") in
         scan ~upto:(n - 1) i ~carried:[ s ]
           (seq
              (LL.Staged_compilation (fun () -> PPrint.empty))
              (seq (set_next s (add (prev s) (get x [| iter i |]))) (set_at out (iter i) (next s))))));
  p "an unrelated local sharing a carried id's integer over another node is refused"
    (entry "sl_bad_int" (fun (x, out, i) ->
         let s = carry ~init:(c 0.) (state "sl_bad_int_s") in
         let other = state "sl_bad_int_o" in
         (* Hand-minted: the same integer as [prev], over a different node -- what the pipeline's
            counter never produces and the integer-keyed censuses assume away. *)
         let clash = { LL.tn = other; scope_id = s.LL.prev.LL.scope_id } in
         seq
           (seq (LL.Declare_local { id = clash; needs_init = true }) (LL.Set_local (clash, c 1.)))
           (scan ~upto:(n - 1) i ~carried:[ s ]
              (seq (set_next s (add (prev s) (get x [| iter i |]))) (set_at out (iter i) (next s))))));
  p "a carried pair over two different nodes is refused"
    (entry "sl_bad_pair" (fun (x, out, i) ->
         let s =
           {
             LL.prev = LL.get_scope (state "sl_bad_pair_s1");
             next = LL.get_scope (state "sl_bad_pair_s2");
             init = c 0.;
           }
         in
         scan ~upto:(n - 1) i ~carried:[ s ]
           (seq (set_next s (add (prev s) (get x [| iter i |]))) (set_at out (iter i) (next s)))));
  (* The exit gate: a well-formed twin builds the record, and the malformed scan replaces its code
     on the way to the backend, where codegen's validation is the only gate left. *)
  let x, out, i = fresh "sl_exit" in
  let st = state "sl_exit_s" in
  let good =
    let s = carry ~init:(c 0.) st in
    scan ~upto:(n - 1) i ~carried:[ s ]
      (seq (set_next s (add (prev s) (get x [| iter i |]))) (set_at out (iter i) (next s)))
  in
  let bad =
    let s = carry ~init:(c 0.) st in
    scan ~upto:(n - 1) i ~carried:[ s ] (set_at out (iter i) (add (prev s) (get x [| iter i |])))
  in
  let o = optimize ~name:"sl_exit" good in
  p "the backend refuses a malformed scan handed through the prelowered seam"
    (rejected (fun () -> link ~name:"sl_exit" { o with LL.llc = bad }))

(* --- Leg 6a: a dead range is malformed. --- *)

let () =
  printf "--- leg 6a: an empty range is refused, at both gates ---\n";
  let x = mk "sl_dead_x" and out = mk "sl_dead_out" in
  materialize x;
  materialize out;
  let dead () =
    let i = sym () in
    let s = carry ~init:(c 0.) (state "sl_dead_s") in
    scan ~upto:(-1) i ~carried:[ s ]
      (seq (set_next s (add (prev s) (get x [| iter i |]))) (set_at out (iter i) (next s)))
  in
  p "a scan over an empty range is refused by optimize"
    (rejected (fun () -> optimize ~name:"sl_dead" (dead ())));
  (* The exit gate, through a well-formed twin's record. *)
  let live =
    let i = sym () in
    let s = carry ~init:(c 0.) (state "sl_dead_live_s") in
    scan ~upto:(n - 1) i ~carried:[ s ]
      (seq (set_next s (add (prev s) (get x [| iter i |]))) (set_at out (iter i) (next s)))
  in
  let o = optimize ~name:"sl_dead_live" live in
  p "a scan over an empty range is refused by the backend through the prelowered seam"
    (rejected (fun () -> link ~name:"sl_dead_live" { o with LL.llc = dead () }))

(* --- Leg 6b: carried state resides at its node's precision. --- *)

let () =
  printf "--- leg 6b: a half-precision state rounds the running sum every step ---\n";
  (* 2048 + 1 is not representable in fp16 (the spacing above 2048 is 2), so a running sum carried
     at half stays at 2048 through six increments, while one carried at single reaches 2054. Both
     values are exact by construction, so the golden may hold them. The input and output nodes are
     single either way: only the state node's precision changes between the twins. *)
  let steps = [| 2048.; 1.; 1.; 1.; 1.; 1.; 1. |] in
  let run ~prec ~first_id ~label =
    let x = mk (label ^ "_x") and out = mk (label ^ "_out") in
    let st = node_factory ~prec ~first_id ~dims:[| 1 |] () (label ^ "_s") in
    virtualize st;
    materialize x;
    materialize out;
    let i = sym () in
    let s = carry ~init:(c 0.) st in
    let llc =
      scan ~upto:(n - 1) i ~carried:[ s ]
        (seq (set_next s (add (prev s) (get x [| iter i |]))) (set_at out (iter i) (next s)))
    in
    let o = optimize ~name:label llc in
    (List.hd_exn
       (execute ~name:label o
          ~seed:[ (x, steps); (out, Array.create ~len:n sentinel) ]
          ~read:[ out ])).(n - 1)
  in
  let half = run ~prec:Ops.half ~first_id:10990 ~label:"sl_half" in
  let single = run ~prec:Ops.single ~first_id:10995 ~label:"sl_single" in
  p "a single-precision state accumulates all six increments (2054)" (Float.equal single 2054.);
  p "a half-precision state rounds each step back to 2048, so the sum stays at 2048"
    (Float.equal half 2048.)

(* --- Leg 7: schedule opacity. --- *)

let () =
  printf "--- leg 7: schedule ops decline the scan's index and the loops inside it ---\n";
  let x, out, _st, i, llc = build_cumsum ~direction:LL.Forward ~label:"sl_sched" in
  let o = optimize ~name:"sl_sched" llc in
  p "Retype of the scan's own index declines for want of a For_loop"
    (declines (fun () -> Schedule.apply [ Schedule.Retype { axis = i; ty = LL.Grid } ] o));
  p "Split of the scan's own index declines for want of a For_loop"
    (declines (fun () ->
         Schedule.apply
           [
             Schedule.Split
               {
                 axis = i;
                 factor = 2;
                 outer = LL.Serial;
                 inner = LL.Serial;
                 outer_index = sym ();
                 inner_index = sym ();
               };
           ]
           o));
  (* A loop nested inside the scan body is out of the ops' reach too: the scan is opaque in both
     directions, locate and rewrite alike (gh-ocannl-668's law). *)
  let z = mk ~dims:[| n; 3 |] "sl_sched_z" in
  materialize z;
  let k = sym () and i2 = sym () in
  let s = carry ~init:(c 0.) (state "sl_sched_s2") in
  let nested =
    scan ~upto:(n - 1) i2 ~carried:[ s ]
      (seq
         (set_next s (add (prev s) (get x [| iter i2 |])))
         (loop_n k 3 (set z [| iter i2; iter k |] (mul (next s) (tick k)))))
  in
  let o2 = optimize ~name:"sl_sched_nested" nested in
  p
    "Unroll of a loop nested inside the scan body declines: the scan is opaque to locate and \
     rewrite"
    (declines (fun () -> Schedule.apply [ Schedule.Unroll { axis = k; materialize = false } ] o2));
  p "the harness traversal reaches the loop inside the scan body exactly once"
    (count_loops k o2.LL.llc = 1);
  (* Executed, so the nested shape is a real program and not only a refusal fixture. *)
  let got =
    List.hd_exn
      (execute ~name:"sl_sched_nested" o2
         ~seed:[ (x, xs); (z, Array.create ~len:(n * 3) sentinel) ]
         ~read:[ z ])
  in
  let want =
    Array.init (n * 3) ~f:(fun idx -> prefix_sums.(idx / 3) *. Float.of_int (1 + (idx % 3)))
  in
  p_all2 "the nested loop computes prefix sum times (1 + k) in every cell" got want ~f:Float.equal;
  ignore (out : Tn.t)

(* --- Leg 8: digest identity. --- *)

let render llc =
  let buf = Buffer.create 256 in
  let add = Buffer.add_string buf in
  let policy =
    {
      CR.emit_tn = (fun tn -> add ("<" ^ List.hd_exn tn.Tn.label ^ ">"));
      emit_free_sym = (fun _ -> add "?");
      on_bind_loop = (fun _ ~id:_ ~shadowed:_ -> ());
      mark_incomplete = (fun () -> ());
      mma = CR.Structural_mma;
      initial_tokens = [];
    }
  in
  CR.emit ~buf policy llc;
  Buffer.contents buf

let () =
  printf "--- leg 8: the canonical rendering sees the recurrence and only the recurrence ---\n";
  let x = mk "sl_dg_x" and out = mk "sl_dg_out" and st = state "sl_dg_s" in
  (* [~swap] exchanges the two ids in the HEADER only, keeping the body's reads and writes: the
     program that reads what the header calls [next] and writes what it calls [prev]. (Swapping the
     record the body is built from as well would only relabel an isomorphic program, which the
     alpha-renaming rendering rightly digests identically.) *)
  let build ~direction ~swap =
    let i = sym () in
    let s = carry ~init:(c 0.) st in
    let header = if swap then { s with LL.prev = s.LL.next; next = s.LL.prev } else s in
    scan ~direction ~upto:(n - 1) i ~carried:[ header ]
      (seq (set_next s (add (prev s) (get x [| iter i |]))) (set_at out (iter i) (next s)))
  in
  let fwd = render (build ~direction:LL.Forward ~swap:false) in
  let fwd' = render (build ~direction:LL.Forward ~swap:false) in
  let bwd = render (build ~direction:LL.Backward ~swap:false) in
  let swapped = render (build ~direction:LL.Forward ~swap:true) in
  p "a fresh lowering of the same scan renders identically (alpha-renamed symbols and ids)"
    (String.equal fwd fwd');
  p "Forward and Backward render differently" (not (String.equal fwd bwd));
  p "swapping a carried pair's prev and next renders differently" (not (String.equal fwd swapped));
  printf "rendering: %s\n" fwd
