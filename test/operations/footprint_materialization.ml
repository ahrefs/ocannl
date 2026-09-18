(* gh-ocannl-616: footprint-scoped materialization — the middle ground between recomputing a virtual
   node at every read site and materializing its whole domain.

   When a heuristic cap (the reduction cap here, the visit cap, the fan-in cap) refuses to inline a
   node, the virtualizer used to have exactly one landing spot: [Never_virtual], a full-domain
   buffer plus a full pass to fill it. This test pins the cheaper one. For a node every read of
   which is an affine sub-image read — the motivating diagonal reader [o[i] = a[i, i]] of an [n×n]
   reduction — the node stays virtual, and the reader gets a fresh routine-private scratch shaped
   like ITS iteration box ([n] cells), filled right after the producer's last write by a prologue
   instantiating the stored template over that box: [n] reduction instances instead of [n×n]
   (materialized) or [n × multiplicity] (recomputed).

   Every case has an executed leg, per the structural-vs-executable rule: the footprint reading, the
   same program with the producer pre-decided [On_device], and (where the cap is the only thing
   standing between the program and inlining) the inlined reading with the cap disabled must agree
   cell for cell, and with an OCaml oracle. The producer discriminates (varies with every loop
   symbol and stays off the zero-init), and unwritten cells carry the sentinel. Scratch nodes are
   told from the program's own by their namespace ([Low_level.footprint_namespace]).

   The rows: the diagonal reader under the reduction cap (the headline); the visit cap as trigger; a
   full reader (unprofitable: the cap materializes as before, and no [`Footprint] flip is offered —
   though the explicit preference still takes the form); a guarded reader (ineligible); the explicit
   preference on a node no cap refuses; a consumer routine footprint-scoping a node an earlier
   routine of the lineage left virtual (the gh-573 corner) and the same shape declining back to
   inlining; a single-cell footprint (fixed indices only); two readers, two scratches; a consumer
   that is itself a virtualization candidate (the read reaches a template, so the decision retracts
   to the cap's materialization); a shared-loop consumer (ineligible); and the configuration key
   off. *)

open Base
open Ll_test
open Verdict.Claims

let n = 4

(* The reduction extent, above [virtualize_max_inline_reduction]'s default. *)
let kk = 32
let mk = node_factory ~first_id:3000 ~dims:[| n; n |] ()
let loop s body = loop_n s n body
let () = assert (LL.virtualize_settings.max_inline_reduction = 16)
let () = assert (LL.virtualize_settings.max_visits = 1)
let () = assert LL.virtualize_settings.footprint_materialization

let scratches (o : LL.optimized) =
  Hashtbl.keys o.LL.traced_store
  |> List.filter ~f:(fun tn -> String.equal tn.Tn.namespace LL.footprint_namespace)
  |> List.sort ~compare:Tn.compare

let scratch_dims (o : LL.optimized) =
  List.map (scratches o) ~f:(fun tn -> Array.to_list (Lazy.force tn.Tn.dims))

let flips_of (o : LL.optimized) tn =
  List.filter_map o.LL.flip_candidates ~f:(fun fc ->
      Option.some_if (Tn.equal fc.LL.fc_tn tn) fc.LL.fc_flip)

let offers o tn flip = List.mem (flips_of o tn) flip ~equal:Poly.equal

let with_setting ~set ~restore f =
  set ();
  Exn.protect ~f ~finally:restore

let reduction_cap_off f =
  with_setting
    ~set:(fun () -> LL.virtualize_settings.max_inline_reduction <- -1)
    ~restore:(fun () -> LL.virtualize_settings.max_inline_reduction <- 16)
    f

let footprint_off f =
  with_setting
    ~set:(fun () -> LL.virtualize_settings.footprint_materialization <- false)
    ~restore:(fun () -> LL.virtualize_settings.footprint_materialization <- true)
    f

let is_cap o tn cap = Option.exists (rejection_code o tn) ~f:(Tn.equal_provenance cap)

(* The reduction producer: [a[i, j] = Σ_{k < kk} (1 + 10 i + j + 100 k)]. Varies with every symbol
   of every loop, and the zero-init is off every partial sum. *)
let big_reduction a =
  let i = sym () and j = sym () and k = sym () in
  seq (zero a)
    (loop i
       (loop j
          (loop_n k kk
             (set a
                [| iter i; iter j |]
                (add (get a [| iter i; iter j |]) (add (tag i j) (mul (c 100.) (embed k))))))))

let reduced i j = Float.of_int ((kk * (1 + (10 * i) + j)) + (100 * (kk * (kk - 1) / 2)))
let diagonal = Array.init n ~f:(fun i -> reduced i i)

(* === The headline: a diagonal reader under the reduction cap. === *)
let case_diagonal_reduction () =
  let a = mk "a" and o = mk ~dims:[| n |] "o" in
  materialize o;
  let i = sym () in
  let llc = seq (big_reduction a) (loop i (set o [| iter i |] (get a [| iter i; iter i |]))) in
  let opt = optimize ~name:"fp_diag" llc in
  p "diagonal: the producer stays virtual" (known_virtual opt a);
  p "diagonal: no buffer read of the producer survives" (count_get opt a = 0);
  p "diagonal: exactly one scratch, 1-D of the reader's extent"
    (List.equal (List.equal Int.equal) (scratch_dims opt) [ [ n ] ]);
  p "diagonal: the scratch is written by the prologue and read by the consumer"
    (match scratches opt with [ d ] -> count_set opt d = 1 && count_get opt d = 1 | _ -> false);
  p "diagonal: the prologue instantiates the reduction (a loop of the reduction extent survives)"
    (Option.is_some (find_loop_with_extent ~in_scopes:true ~n:kk opt.LL.llc));
  p "diagonal: both lattice neighbours are offered as flips"
    (offers opt a `Materialize && offers opt a `Inline);
  p "diagonal: the footprint form is not offered where it is already chosen"
    (not (offers opt a `Footprint));
  let seed = [ (o, blank n) ] and read = [ o ] in
  let fp = execute ~name:"fp_diag" opt ~seed ~read in
  let mat =
    execute ~name:"fp_diag_mat" (optimize ~materialized:[ a ] ~name:"fp_diag" llc) ~seed ~read
  in
  let inl = reduction_cap_off (fun () -> optimize ~name:"fp_diag_inl" llc) in
  p_empty "diagonal: the inlined arm really inlines (no scratch)"
    ~over:(Hashtbl.keys inl.LL.traced_store)
    (scratches inl);
  let inl = execute ~name:"fp_diag_inl" inl ~seed ~read in
  p "diagonal: executed values are the diagonal of the reduction" (same fp [ diagonal ]);
  p "diagonal: footprint and materialized arms agree" (same fp mat);
  p "diagonal: footprint and inlined arms agree" (same fp inl)

(* === The visit cap as trigger: a cheap producer read [n] times per diagonal cell. === *)
let case_visit_cap () =
  let a = mk "av" and x = mk ~dims:[| n |] "xv" and o = mk "ov" in
  materialize o;
  materialize x;
  let i = sym () and j = sym () and i' = sym () and j' = sym () in
  (* Reading [x] makes the producer complex, so the visit cap applies. *)
  let producer =
    loop i (loop j (set a [| iter i; iter j |] (add (get x [| iter i |]) (tag i j))))
  in
  let consumer =
    loop i' (loop j' (set o [| iter i'; iter j' |] (add (get a [| iter i'; iter i' |]) (embed j'))))
  in
  let llc = seq producer consumer in
  let opt = optimize ~name:"fp_visit" llc in
  p "visit-cap: the producer stays virtual" (known_virtual opt a);
  p "visit-cap: one 1-D scratch over the reader's outer loop"
    (List.equal (List.equal Int.equal) (scratch_dims opt) [ [ n ] ]);
  let xs = Array.init n ~f:(fun i -> Float.of_int (5 + (3 * i))) in
  let expected =
    Array.init (n * n) ~f:(fun c ->
        let i = c / n and j = c % n in
        xs.(i) +. Float.of_int (1 + (10 * i) + i + j))
  in
  let seed = [ (x, xs); (o, blank (n * n)) ] and read = [ o ] in
  let fp = execute ~name:"fp_visit" opt ~seed ~read in
  let mat =
    execute ~name:"fp_visit_mat" (optimize ~materialized:[ a ] ~name:"fp_visit" llc) ~seed ~read
  in
  p "visit-cap: executed values are the diagonal broadcast plus the column" (same fp [ expected ]);
  p "visit-cap: footprint and materialized arms agree" (same fp mat)

(* === A full reader: as many scratch cells as the node has, so the cap materializes as before, and
   the search is not offered a form that cannot beat it; the explicit preference still takes it,
   yielding a 2-D scratch with the same values. === *)
let case_full_reader () =
  let a = mk "af" and o = mk "of" in
  materialize o;
  let i = sym () and j = sym () in
  let llc =
    seq (big_reduction a)
      (loop i (loop j (set o [| iter i; iter j |] (get a [| iter i; iter j |]))))
  in
  let opt = optimize ~name:"fp_full" llc in
  p "full: the reduction cap materializes the producer" (is_cap opt a Tn.Inline_reduction_cap);
  p_empty "full: no scratch" ~over:(Hashtbl.keys opt.LL.traced_store) (scratches opt);
  p "full: the inline flip is offered, the footprint flip is not (as many cells as the node)"
    (offers opt a `Inline && not (offers opt a `Footprint));
  let expected = Array.init (n * n) ~f:(fun c -> reduced (c / n) (c % n)) in
  let seed = [ (o, blank (n * n)) ] and read = [ o ] in
  let mat = execute ~name:"fp_full" opt ~seed ~read in
  p "full: executed values are the reduction" (same mat [ expected ]);
  let ctx = LL.empty_optimize_ctx () in
  Hash_set.add ctx.LL.footprint_preferences a;
  let pref = optimize_in ctx ~name:"fp_full_pref" llc in
  p "full-preferred: the producer stays virtual" (known_virtual pref a);
  p "full-preferred: one 2-D scratch over the reader's box"
    (List.equal (List.equal Int.equal) (scratch_dims pref) [ [ n; n ] ]);
  let fp = execute ~name:"fp_full_pref" pref ~seed ~read in
  p "full-preferred: footprint and materialized arms agree" (same fp mat)

(* === A guarded reader is ineligible: the prologue would replay the instantiation over the whole
   box, unconditionally. The cap materializes; the guard keeps its cells. === *)
let case_guarded_reader () =
  let a = mk "ag" and o = mk ~dims:[| n |] "og" in
  materialize o;
  let i = sym () in
  let consumer =
    loop i (if_idx (lt (embed i) (ic (n - 1))) (set o [| iter i |] (get a [| iter i; iter i |])))
  in
  let llc = seq (big_reduction a) consumer in
  let opt = optimize ~name:"fp_guarded" llc in
  p "guarded: the reduction cap materializes the producer" (is_cap opt a Tn.Inline_reduction_cap);
  p_empty "guarded: no scratch" ~over:(Hashtbl.keys opt.LL.traced_store) (scratches opt);
  p "guarded: the footprint flip is not offered" (not (offers opt a `Footprint));
  let expected = Array.init n ~f:(fun i -> if i < n - 1 then reduced i i else sentinel) in
  let got = execute ~name:"fp_guarded" opt ~seed:[ (o, blank n) ] ~read:[ o ] in
  p "guarded: executed values respect the guard" (same got [ expected ])

(* === The explicit preference on a node no cap refuses: the default inlines. === *)
let case_preference () =
  let a = mk "ap" and x = mk ~dims:[| n |] "xp" and o = mk ~dims:[| n |] "op" in
  materialize o;
  materialize x;
  let i = sym () and j = sym () and i' = sym () in
  let producer =
    loop i (loop j (set a [| iter i; iter j |] (add (get x [| iter i |]) (tag i j))))
  in
  let consumer = loop i' (set o [| iter i' |] (get a [| iter i'; iter i' |])) in
  let llc = seq producer consumer in
  let dflt = optimize ~name:"fp_pref_default" llc in
  p "preference: the default inlines the producer (no buffer read)"
    (known_virtual dflt a && count_get dflt a = 0);
  p_empty "preference: the default mints no scratch"
    ~over:(Hashtbl.keys dflt.LL.traced_store)
    (scratches dflt);
  let ctx = LL.empty_optimize_ctx () in
  Hash_set.add ctx.LL.footprint_preferences a;
  let pref = optimize_in ctx ~name:"fp_pref" llc in
  p "preference: the preference lands the producer on the footprint form"
    (known_virtual pref a && List.equal (List.equal Int.equal) (scratch_dims pref) [ [ n ] ]);
  let xs = Array.init n ~f:(fun i -> Float.of_int (7 + i)) in
  let expected = Array.init n ~f:(fun i -> xs.(i) +. Float.of_int (1 + (11 * i))) in
  let seed = [ (x, xs); (o, blank n) ] and read = [ o ] in
  let fp = execute ~name:"fp_pref" pref ~seed ~read in
  let inl = execute ~name:"fp_pref_default" dflt ~seed ~read in
  p "preference: executed values are the diagonal" (same fp [ expected ]);
  p "preference: footprint and inlined arms agree" (same fp inl)

(* === The gh-573 corner: an earlier routine of the lineage leaves the producer virtual (an
   all-virtual routine, gh-611), and no cap can refuse its recompute in a consumer routine — the
   consumer footprint-scopes it on the template's own reduction extent. A consumer the form cannot
   serve (a guarded reader) declines to the inlining it would get anyway. === *)
let case_inherited () =
  let a = mk "ai" and o = mk ~dims:[| n |] "oi" and og = mk ~dims:[| n |] "oig" in
  materialize o;
  materialize og;
  let ctx = LL.empty_optimize_ctx () in
  let producer = optimize_in ctx ~name:"fp_inh_producer" (big_reduction a) in
  p "inherited: the producer routine commits the node virtual with an empty schedule"
    (known_virtual producer a && match producer.LL.llc with LL.Noop -> true | _ -> false);
  let i = sym () in
  let consumer =
    optimize_in ctx ~name:"fp_inh_consumer"
      (loop i (set o [| iter i |] (get a [| iter i; iter i |])))
  in
  p "inherited: the consumer footprint-scopes the inherited node"
    (List.equal (List.equal Int.equal) (scratch_dims consumer) [ [ n ] ] && count_get consumer a = 0);
  let got = execute ~name:"fp_inh_consumer" consumer ~seed:[ (o, blank n) ] ~read:[ o ] in
  p "inherited: executed values are the diagonal of the inherited reduction" (same got [ diagonal ]);
  let i = sym () in
  let guarded =
    optimize_in ctx ~name:"fp_inh_guarded"
      (loop i
         (if_idx (lt (embed i) (ic (n - 1))) (set og [| iter i |] (get a [| iter i; iter i |]))))
  in
  p "inherited-guarded: the read declines to inlining (no buffer read)" (count_get guarded a = 0);
  p_empty "inherited-guarded: no scratch"
    ~over:(Hashtbl.keys guarded.LL.traced_store)
    (scratches guarded);
  let expected = Array.init n ~f:(fun i -> if i < n - 1 then reduced i i else sentinel) in
  let got = execute ~name:"fp_inh_guarded" guarded ~seed:[ (og, blank n) ] ~read:[ og ] in
  p "inherited-guarded: executed values respect the guard" (same got [ expected ])

(* === Fixed indices only: a single-cell footprint under a loop the read does not mention. === *)
let case_single_cell () =
  let a = mk "as" and o = mk ~dims:[| n |] "os" in
  materialize o;
  let j = sym () in
  let llc =
    seq (big_reduction a)
      (loop j (set o [| iter j |] (add (get a [| fixed 0; fixed 0 |]) (embed j))))
  in
  let opt = optimize ~name:"fp_cell" llc in
  p "single-cell: the producer stays virtual" (known_virtual opt a);
  p "single-cell: one single-cell scratch"
    (List.equal (List.equal Int.equal) (scratch_dims opt) [ [ 1 ] ]);
  let expected = Array.init n ~f:(fun j -> reduced 0 0 +. Float.of_int j) in
  let seed = [ (o, blank n) ] and read = [ o ] in
  let fp = execute ~name:"fp_cell" opt ~seed ~read in
  let mat =
    execute ~name:"fp_cell_mat" (optimize ~materialized:[ a ] ~name:"fp_cell" llc) ~seed ~read
  in
  p "single-cell: executed values are the corner cell plus the column" (same fp [ expected ]);
  p "single-cell: footprint and materialized arms agree" (same fp mat)

(* === Two readers, two scratches: the diagonal and the first column. === *)
let case_two_readers () =
  let a = mk "at" and o1 = mk ~dims:[| n |] "ot1" and o2 = mk ~dims:[| n |] "ot2" in
  materialize o1;
  materialize o2;
  let i = sym () and i' = sym () in
  let llc =
    seq (big_reduction a)
      (seq
         (loop i (set o1 [| iter i |] (get a [| iter i; iter i |])))
         (loop i' (set o2 [| iter i' |] (get a [| iter i'; fixed 0 |]))))
  in
  let opt = optimize ~name:"fp_two" llc in
  p "two-readers: the producer stays virtual" (known_virtual opt a);
  p "two-readers: one scratch per reader"
    (List.equal (List.equal Int.equal) (scratch_dims opt) [ [ n ]; [ n ] ]);
  let column = Array.init n ~f:(fun i -> reduced i 0) in
  let seed = [ (o1, blank n); (o2, blank n) ] and read = [ o1; o2 ] in
  let fp = execute ~name:"fp_two" opt ~seed ~read in
  let mat =
    execute ~name:"fp_two_mat" (optimize ~materialized:[ a ] ~name:"fp_two" llc) ~seed ~read
  in
  p "two-readers: executed values are the diagonal and the column" (same fp [ diagonal; column ]);
  p "two-readers: footprint and materialized arms agree" (same fp mat)

(* === A consumer that is itself a virtualization candidate: the read would land in ITS stored
   template, which a later routine may replay where the scratch does not exist, so the decision
   retracts to the cap's own materialization. === *)
let case_candidate_consumer () =
  let a = mk "ac" and cc = mk ~dims:[| n |] "cc" and o = mk ~dims:[| n |] "oc" in
  materialize o;
  let i = sym () and i' = sym () in
  let llc =
    seq (big_reduction a)
      (seq
         (loop i (set cc [| iter i |] (get a [| iter i; iter i |])))
         (loop i' (set o [| iter i' |] (get cc [| iter i' |]))))
  in
  let opt = optimize ~name:"fp_candidate" llc in
  p "candidate-consumer: the decision retracts to the reduction cap"
    (is_cap opt a Tn.Inline_reduction_cap);
  p_empty "candidate-consumer: no scratch" ~over:(Hashtbl.keys opt.LL.traced_store) (scratches opt);
  p "candidate-consumer: the intermediate consumer inlines" (known_virtual opt cc);
  p "candidate-consumer: the retracted node is not offered the footprint flip"
    (offers opt a `Inline && not (offers opt a `Footprint));
  let got = execute ~name:"fp_candidate" opt ~seed:[ (o, blank n) ] ~read:[ o ] in
  p "candidate-consumer: executed values are the diagonal" (same got [ diagonal ])

(* === A shared-loop consumer: the reader's statement writes a second node, so the prologue could
   not be hoisted ahead of it soundly; ineligible, the cap materializes. === *)
let case_shared_loop () =
  let a = mk "ash" and x = mk ~dims:[| n |] "xsh" in
  let o1 = mk ~dims:[| n |] "osh1" and o2 = mk ~dims:[| n |] "osh2" in
  materialize o1;
  materialize o2;
  materialize x;
  let i = sym () in
  let llc =
    seq (big_reduction a)
      (loop i
         (seq
            (set o1 [| iter i |] (get a [| iter i; iter i |]))
            (set o2 [| iter i |] (get x [| iter i |]))))
  in
  let opt = optimize ~name:"fp_shared" llc in
  p "shared-loop: the reduction cap materializes the producer"
    (is_cap opt a Tn.Inline_reduction_cap);
  p_empty "shared-loop: no scratch" ~over:(Hashtbl.keys opt.LL.traced_store) (scratches opt);
  let xs = Array.init n ~f:(fun i -> Float.of_int (2 + i)) in
  let got =
    execute ~name:"fp_shared" opt ~seed:[ (x, xs); (o1, blank n); (o2, blank n) ] ~read:[ o1; o2 ]
  in
  p "shared-loop: executed values are the diagonal and the copy" (same got [ diagonal; xs ])

(* === The key off restores the caps' full materialization. === *)
let case_key_off () =
  let a = mk "ak" and o = mk ~dims:[| n |] "ok" in
  materialize o;
  let i = sym () in
  let llc = seq (big_reduction a) (loop i (set o [| iter i |] (get a [| iter i; iter i |]))) in
  let opt = footprint_off (fun () -> optimize ~name:"fp_off" llc) in
  p "key-off: the reduction cap materializes the producer" (is_cap opt a Tn.Inline_reduction_cap);
  p_empty "key-off: no scratch" ~over:(Hashtbl.keys opt.LL.traced_store) (scratches opt);
  p "key-off: no footprint flip" (not (offers opt a `Footprint));
  let got = execute ~name:"fp_off" opt ~seed:[ (o, blank n) ] ~read:[ o ] in
  p "key-off: executed values are the diagonal" (same got [ diagonal ])

(* A reduction producer reading a materialized operand: [a[i, j] = Σ_{k < kk} (x[i] + 1 + 10 i + j +
   100 k)]. *)
let big_reduction_over x a =
  let i = sym () and j = sym () and k = sym () in
  seq (zero a)
    (loop i
       (loop j
          (loop_n k kk
             (set a
                [| iter i; iter j |]
                (add
                   (get a [| iter i; iter j |])
                   (add (get x [| iter i |]) (add (tag i j) (mul (c 100.) (embed k)))))))))

let reduced_plus b i j = reduced i j +. Float.of_int (kk * b)

(* The flat index of the first top-level statement writing [tn] in the optimized code. *)
let stmt_writing (o : LL.optimized) tn =
  List.findi (LL.flat_lines [ o.LL.llc ]) ~f:(fun _ st ->
      count_stmt ~in_scopes:true st ~f:(function
        | LL.Set { tn = t; _ } -> Tn.equal t tn
        | _ -> false)
      > 0)
  |> Option.map ~f:fst

(* === An intervening write to a template input between the producer and the reader: the prologue
   runs at the producer's position, so the scratch snapshots the input exactly as the materialized
   producer would have read it, not as the reader finds it. === *)
let case_intervening_write () =
  let a = mk "aw" and x = mk ~dims:[| n |] "xw" and o = mk ~dims:[| n |] "ow" in
  materialize o;
  materialize x;
  let t = sym () and i' = sym () in
  let rewrite = loop t (set x [| iter t |] (mul (c 2.) (get x [| iter t |]))) in
  let consumer = loop i' (set o [| iter i' |] (get a [| iter i'; iter i' |])) in
  let llc = seq (big_reduction_over x a) (seq rewrite consumer) in
  let opt = optimize ~name:"fp_intervening" llc in
  p "intervening: the producer stays virtual" (known_virtual opt a);
  p "intervening: one 1-D scratch" (List.equal (List.equal Int.equal) (scratch_dims opt) [ [ n ] ]);
  p "intervening: the prologue runs at the producer's position, ahead of the rewrite"
    (match scratches opt with
    | [ d ] -> (
        match (stmt_writing opt d, stmt_writing opt x) with
        | Some pro, Some rw -> pro < rw
        | _ -> false)
    | _ -> false);
  let xs = Array.init n ~f:(fun i -> Float.of_int (1 + i)) in
  let expected = Array.init n ~f:(fun i -> reduced_plus (1 + i) i i) in
  let doubled = Array.map xs ~f:(fun v -> 2. *. v) in
  let seed = [ (x, xs); (o, blank n) ] and read = [ o; x ] in
  let fp = execute ~name:"fp_intervening" opt ~seed ~read in
  let mat =
    execute ~name:"fp_intervening_mat"
      (optimize ~materialized:[ a ] ~name:"fp_intervening" llc)
      ~seed ~read
  in
  p "intervening: the scratch holds the reduction over the input as it was at the producer"
    (same fp [ expected; doubled ]);
  p "intervening: footprint and materialized arms agree" (same fp mat)

(* === A reader placed between the producer's zero-initialization and its accumulating nest reads
   the zeros; the prologue would run after the nest, so the site is ineligible and the cap
   materializes. === *)
let case_reader_between_setters () =
  let a = mk "ab" and o = mk ~dims:[| n |] "ob" in
  materialize o;
  let i = sym () and j = sym () and k = sym () and i' = sym () in
  let fill =
    loop i
      (loop j
         (loop_n k kk
            (set a
               [| iter i; iter j |]
               (add (get a [| iter i; iter j |]) (add (tag i j) (mul (c 100.) (embed k)))))))
  in
  let consumer = loop i' (set o [| iter i' |] (get a [| iter i'; iter i' |])) in
  let llc = seq (zero a) (seq consumer fill) in
  let opt = optimize ~name:"fp_between" llc in
  p "between-setters: the reduction cap materializes the producer"
    (is_cap opt a Tn.Inline_reduction_cap);
  p_empty "between-setters: no scratch" ~over:(Hashtbl.keys opt.LL.traced_store) (scratches opt);
  let seed = [ (o, blank n) ] and read = [ o ] in
  let got = execute ~name:"fp_between" opt ~seed ~read in
  p "between-setters: executed values are the zero-init the reader finds"
    (same got [ Array.create ~len:n 0. ])

(* === The gh-573 corner with a recurrence: the inherited template reads the very node the reader
   writes, so a prologue ahead of the reader's statement would not read what the inlined read reads;
   both the automatic trigger and the explicit preference decline to inlining. === *)
let case_inherited_recurrence () =
  let a = mk "ar" and o = mk ~dims:[| n |] "or" in
  materialize o;
  let ctx = LL.empty_optimize_ctx () in
  let producer = optimize_in ctx ~name:"fp_rec_producer" (big_reduction_over o a) in
  p "recurrence: the producer routine leaves the node virtual" (known_virtual producer a);
  Hash_set.add ctx.LL.footprint_preferences a;
  let i' = sym () in
  let consumer =
    optimize_in ctx ~name:"fp_rec_consumer"
      (loop i' (set o [| iter i' |] (add (get a [| iter i'; iter i' |]) (c 1.))))
  in
  p_empty "recurrence: the reader writes a template leaf, so no scratch"
    ~over:(Hashtbl.keys consumer.LL.traced_store)
    (scratches consumer);
  p "recurrence: the read is inlined" (count_get consumer a = 0);
  let os = Array.init n ~f:(fun i -> Float.of_int (3 + (2 * i))) in
  let expected = Array.init n ~f:(fun i -> reduced_plus (3 + (2 * i)) i i +. 1.) in
  let got = execute ~name:"fp_rec_consumer" consumer ~seed:[ (o, os) ] ~read:[ o ] in
  p "recurrence: executed values read the pre-write cell, as inlining does" (same got [ expected ])

(* === An inherited template whose operand the consumer routine never mentions: the consumer's
   traced store learns of it while the footprint decision is made, which must not disturb the
   decision pass; the operand is then an input of the consumer. === *)
let case_inherited_operand () =
  let a = mk "aio" and x = mk ~dims:[| n |] "xio" and o = mk ~dims:[| n |] "oio" in
  materialize o;
  materialize x;
  let ctx = LL.empty_optimize_ctx () in
  let producer = optimize_in ctx ~name:"fp_operand_producer" (big_reduction_over x a) in
  p "inherited-operand: the producer routine leaves the node virtual" (known_virtual producer a);
  let i' = sym () in
  let consumer =
    optimize_in ctx ~name:"fp_operand_consumer"
      (loop i' (set o [| iter i' |] (get a [| iter i'; iter i' |])))
  in
  p "inherited-operand: the consumer footprint-scopes the inherited node"
    (List.equal (List.equal Int.equal) (scratch_dims consumer) [ [ n ] ] && count_get consumer a = 0);
  let xs = Array.init n ~f:(fun i -> Float.of_int (4 + i)) in
  let expected = Array.init n ~f:(fun i -> reduced_plus (4 + i) i i) in
  let got =
    execute ~name:"fp_operand_consumer" consumer ~seed:[ (x, xs); (o, blank n) ] ~read:[ o ]
  in
  p "inherited-operand: executed values are the diagonal over the operand" (same got [ expected ])

(* === The producer's own statement writes one of the template's inputs after the producer (a shared
   loop): the prologue would run after the whole statement and read the updated input, so the site
   is ineligible and the cap materializes. === *)
let case_producer_statement_writes_input () =
  let a = mk "ap2" and x = mk ~dims:[| n |] "xp2" and o = mk ~dims:[| n |] "op2" in
  materialize o;
  materialize x;
  let i = sym () and j = sym () and k = sym () and i' = sym () in
  let producer =
    seq (zero a)
      (loop i
         (seq
            (loop j
               (loop_n k kk
                  (set a
                     [| iter i; iter j |]
                     (add
                        (get a [| iter i; iter j |])
                        (add (get x [| iter i |]) (add (tag i j) (mul (c 100.) (embed k))))))))
            (set x [| iter i |] (mul (c 2.) (get x [| iter i |])))))
  in
  let consumer = loop i' (set o [| iter i' |] (get a [| iter i'; iter i' |])) in
  let llc = seq producer consumer in
  let opt = optimize ~name:"fp_shared_producer" llc in
  p "shared-producer: the reduction cap materializes the producer"
    (is_cap opt a Tn.Inline_reduction_cap);
  p_empty "shared-producer: no scratch" ~over:(Hashtbl.keys opt.LL.traced_store) (scratches opt);
  let xs = Array.init n ~f:(fun i -> Float.of_int (1 + i)) in
  let expected = Array.init n ~f:(fun i -> reduced_plus (1 + i) i i) in
  let doubled = Array.map xs ~f:(fun v -> 2. *. v) in
  let got = execute ~name:"fp_shared_producer" opt ~seed:[ (x, xs); (o, blank n) ] ~read:[ o; x ] in
  p "shared-producer: executed values read the input as the producer did"
    (same got [ expected; doubled ])

(* === A template input rewritten between two accumulating components of the producer: the
   materialized execution folds the old input into the first component and the new one into the
   second, which a prologue replaying both after the last write cannot reproduce — ineligible, the
   cap materializes. === *)
let case_input_written_between_setters () =
  let a = mk "ab2" and x = mk ~dims:[| n |] "xb2" and o = mk ~dims:[| n |] "ob2" in
  materialize o;
  materialize x;
  let accumulate () =
    let i = sym () and j = sym () and k = sym () in
    loop i
      (loop j
         (loop_n k kk
            (set a
               [| iter i; iter j |]
               (add
                  (get a [| iter i; iter j |])
                  (add (get x [| iter i |]) (add (tag i j) (mul (c 100.) (embed k))))))))
  in
  let t = sym () and i' = sym () in
  let rewrite = loop t (set x [| iter t |] (mul (c 2.) (get x [| iter t |]))) in
  let consumer = loop i' (set o [| iter i' |] (get a [| iter i'; iter i' |])) in
  let llc = seq (zero a) (seq (accumulate ()) (seq rewrite (seq (accumulate ()) consumer))) in
  let opt = optimize ~name:"fp_between_components" llc in
  p "between-components: the reduction cap materializes the producer"
    (is_cap opt a Tn.Inline_reduction_cap);
  p_empty "between-components: no scratch" ~over:(Hashtbl.keys opt.LL.traced_store) (scratches opt);
  let xs = Array.init n ~f:(fun i -> Float.of_int (1 + i)) in
  let expected =
    Array.init n ~f:(fun i -> reduced_plus (1 + i) i i +. reduced_plus (2 * (1 + i)) i i)
  in
  let doubled = Array.map xs ~f:(fun v -> 2. *. v) in
  let got =
    execute ~name:"fp_between_components" opt ~seed:[ (x, xs); (o, blank n) ] ~read:[ o; x ]
  in
  p "between-components: executed values fold the old input, then the new"
    (same got [ expected; doubled ])

(* === A consumption-time rejection AFTER a scratch was minted: a fixed-index producer read once at
   the matching column (footprinted) and once at another (rejected, [Non_virtual 13]). The node
   materializes; the stranded prologue is a scope over a now-materialized node, which cleanup's
   scope-target retraction (gh-ocannl-681) rewrites into a read of the buffer — an [n]-cell gather,
   not a recomputation — so the outcome is the materialized placement plus that copy. === *)
let case_rejection_after_footprint () =
  let a = mk "arj" and o1 = mk ~dims:[| n |] "orj1" and o2 = mk ~dims:[| n |] "orj2" in
  materialize o1;
  materialize o2;
  let i = sym () and k = sym () and i1 = sym () and i2 = sym () in
  let producer =
    seq (zero a)
      (loop i
         (loop_n k kk
            (set a
               [| iter i; fixed 0 |]
               (add (get a [| iter i; fixed 0 |]) (add (tick i) (mul (c 100.) (embed k)))))))
  in
  let llc =
    seq producer
      (seq
         (loop i1 (set o1 [| iter i1 |] (get a [| iter i1; fixed 0 |])))
         (loop i2 (set o2 [| iter i2 |] (get a [| iter i2; fixed 1 |]))))
  in
  let opt = optimize ~name:"fp_rejection" llc in
  p "rejection: the mismatched read commits the producer materialized"
    (known_non_virtual opt a && not (known_virtual opt a));
  (* Three buffer reads of the producer: its own accumulating self-read, the gather, the rejected
     read; and no scope anywhere — the prologue's was retracted. *)
  p "rejection: the stranded prologue is a gather of the buffer (three buffer reads, no scope)"
    (List.equal (List.equal Int.equal) (scratch_dims opt) [ [ n ] ]
    && count_get opt a = 3
    && count_scopes opt.LL.llc = 0);
  let column =
    Array.init n ~f:(fun i -> Float.of_int ((kk * (1 + i)) + (100 * (kk * (kk - 1) / 2))))
  in
  let seed = [ (o1, blank n); (o2, blank n) ] and read = [ o1; o2 ] in
  let got = execute ~name:"fp_rejection" opt ~seed ~read in
  let mat =
    execute ~name:"fp_rejection_mat"
      (optimize ~materialized:[ a ] ~name:"fp_rejection" llc)
      ~seed ~read
  in
  p "rejection: executed values are the column and the untouched zeros"
    (same got [ column; Array.create ~len:n 0. ]);
  p "rejection: footprint and materialized arms agree" (same got mat)

(* === A LOCAL written between two accumulating components: the same hazard as a tensor written
   there, invisible to the access relations — ineligible, the cap materializes. === *)
let case_local_written_between_setters () =
  let a = mk "al" and o = mk ~dims:[| n |] "ol" and lt = mk ~dims:[| 1 |] "lt" in
  materialize o;
  virtualize lt;
  let l = LL.get_scope lt in
  let accumulate () =
    let i = sym () and j = sym () and k = sym () in
    loop i
      (loop j
         (loop_n k kk
            (set a
               [| iter i; iter j |]
               (add
                  (get a [| iter i; iter j |])
                  (add (LL.Get_local l) (add (tag i j) (mul (c 100.) (embed k))))))))
  in
  let i' = sym () in
  let consumer = loop i' (set o [| iter i' |] (get a [| iter i'; iter i' |])) in
  let llc =
    seq
      (LL.Declare_local { id = l; needs_init = false })
      (seq (zero a)
         (seq
            (LL.Set_local (l, c 1.))
            (seq (accumulate ()) (seq (LL.Set_local (l, c 2.)) (seq (accumulate ()) consumer)))))
  in
  let opt = optimize ~name:"fp_local_between" llc in
  p "local-between: the reduction cap materializes the producer"
    (is_cap opt a Tn.Inline_reduction_cap);
  p_empty "local-between: no scratch" ~over:(Hashtbl.keys opt.LL.traced_store) (scratches opt);
  let expected = Array.init n ~f:(fun i -> reduced_plus 1 i i +. reduced_plus 2 i i) in
  let seed = [ (o, blank n) ] and read = [ o ] in
  let got = execute ~name:"fp_local_between" opt ~seed ~read in
  let mat =
    execute ~name:"fp_local_between_mat"
      (optimize ~materialized:[ a ] ~name:"fp_local_between" llc)
      ~seed ~read
  in
  p "local-between: executed values fold the local's first value, then its second"
    (same got [ expected ]);
  p "local-between: capped and materialized arms agree" (same got mat)

(* === An explicit preference on a node the footprint form cannot serve (a guarded reader): the
   preference still exempts the node from the caps, and the read falls through to inlining. === *)
let case_preference_ineligible () =
  let a = mk "api" and o = mk ~dims:[| n |] "opi" in
  materialize o;
  let i = sym () in
  let consumer =
    loop i (if_idx (lt (embed i) (ic (n - 1))) (set o [| iter i |] (get a [| iter i; iter i |])))
  in
  let llc = seq (big_reduction a) consumer in
  let ctx = LL.empty_optimize_ctx () in
  Hash_set.add ctx.LL.footprint_preferences a;
  let opt = optimize_in ctx ~name:"fp_pref_ineligible" llc in
  p "preference-ineligible: the node stays virtual and its read is inlined"
    (known_virtual opt a && count_get opt a = 0);
  p_empty "preference-ineligible: no scratch"
    ~over:(Hashtbl.keys opt.LL.traced_store)
    (scratches opt);
  let expected = Array.init n ~f:(fun i -> if i < n - 1 then reduced i i else sentinel) in
  let got = execute ~name:"fp_pref_ineligible" opt ~seed:[ (o, blank n) ] ~read:[ o ] in
  p "preference-ineligible: executed values respect the guard" (same got [ expected ])

(* === A dead loop "writing" the producer after its real write, with a template input rewritten in
   between: a dead loop is no writer, so the prologue follows the real producer and snapshots the
   input as it was there. === *)
let case_dead_loop_writer () =
  let a = mk "ad" and x = mk ~dims:[| n |] "xd" and o = mk ~dims:[| n |] "od" in
  materialize o;
  materialize x;
  let t = sym () and dz = sym () and i' = sym () in
  let rewrite = loop t (set x [| iter t |] (mul (c 2.) (get x [| iter t |]))) in
  let dead = Ll_builders.loop ~from_:1 ~upto:0 dz (set a [| fixed 0; fixed 0 |] (c 7.)) in
  let consumer = loop i' (set o [| iter i' |] (get a [| iter i'; iter i' |])) in
  let llc = seq (big_reduction_over x a) (seq rewrite (seq dead consumer)) in
  let opt = optimize ~name:"fp_dead_writer" llc in
  p "dead-writer: the producer stays virtual, footprint-scoped"
    (known_virtual opt a && List.equal (List.equal Int.equal) (scratch_dims opt) [ [ n ] ]);
  p "dead-writer: the prologue precedes the rewrite"
    (match scratches opt with
    | [ d ] -> (
        match (stmt_writing opt d, stmt_writing opt x) with
        | Some pro, Some rw -> pro < rw
        | _ -> false)
    | _ -> false);
  let xs = Array.init n ~f:(fun i -> Float.of_int (1 + i)) in
  let expected = Array.init n ~f:(fun i -> reduced_plus (1 + i) i i) in
  let doubled = Array.map xs ~f:(fun v -> 2. *. v) in
  let seed = [ (x, xs); (o, blank n) ] and read = [ o; x ] in
  let fp = execute ~name:"fp_dead_writer" opt ~seed ~read in
  let mat =
    execute ~name:"fp_dead_writer_mat"
      (optimize ~materialized:[ a ] ~name:"fp_dead_writer" llc)
      ~seed ~read
  in
  p "dead-writer: the scratch holds the reduction over the input as it was at the producer"
    (same fp [ expected; doubled ]);
  p "dead-writer: footprint and materialized arms agree" (same fp mat)

(* === A local mutated inside the producer's own statement, read by the producer: the template would
   replay it from the local's state at the prologue rather than from the value each iteration saw —
   ineligible, the cap materializes. === *)
let case_local_in_producer_statement () =
  let a = mk "als" and o = mk ~dims:[| n |] "ols" and lt = mk ~dims:[| 1 |] "lts" in
  materialize o;
  virtualize lt;
  let l = LL.get_scope lt in
  let i = sym () and j = sym () and k = sym () and i' = sym () in
  let producer =
    seq (zero a)
      (loop i
         (seq
            (LL.Set_local (l, add (LL.Get_local l) (c 1.)))
            (loop j
               (loop_n k kk
                  (set a
                     [| iter i; iter j |]
                     (add
                        (get a [| iter i; iter j |])
                        (add (LL.Get_local l) (add (tag i j) (mul (c 100.) (embed k))))))))))
  in
  let consumer = loop i' (set o [| iter i' |] (get a [| iter i'; iter i' |])) in
  let llc =
    seq
      (LL.Declare_local { id = l; needs_init = false })
      (seq (LL.Set_local (l, c 0.)) (seq producer consumer))
  in
  let opt = optimize ~name:"fp_local_in_producer" llc in
  p "local-in-producer: the reduction cap materializes the producer"
    (is_cap opt a Tn.Inline_reduction_cap);
  p_empty "local-in-producer: no scratch" ~over:(Hashtbl.keys opt.LL.traced_store) (scratches opt);
  let expected = Array.init n ~f:(fun i -> reduced_plus (i + 1) i i) in
  let got = execute ~name:"fp_local_in_producer" opt ~seed:[ (o, blank n) ] ~read:[ o ] in
  p "local-in-producer: executed values see the local as each iteration left it"
    (same got [ expected ])

(* === A read under a scalar gate — the true arm of a [Where] whose condition keeps the shifted
   index in range: not an [If] guard, so the access relations do not mark it, but the prologue would
   instantiate the template unconditionally over the whole box and read the input out of range at
   the last row. Ineligible; the cap materializes and the gate short-circuits. === *)
let case_gated_read () =
  let a = mk "agt" and x = mk ~dims:[| n |] "xgt" and o = mk ~dims:[| n |] "ogt" in
  materialize o;
  materialize x;
  let i = sym () in
  let shifted = aff [ (1, i) ] 1 in
  let consumer =
    loop i
      (set o
         [| iter i |]
         (where_ (lt (embed i) (ic (n - 1))) (get a [| shifted; shifted |]) (c 0.)))
  in
  let llc = seq (big_reduction_over x a) consumer in
  let opt = optimize ~name:"fp_gated" llc in
  p "gated-read: the reduction cap materializes the producer" (is_cap opt a Tn.Inline_reduction_cap);
  p_empty "gated-read: no scratch" ~over:(Hashtbl.keys opt.LL.traced_store) (scratches opt);
  let xs = Array.init n ~f:(fun i -> Float.of_int (1 + i)) in
  let expected =
    Array.init n ~f:(fun i -> if i < n - 1 then reduced_plus (2 + i) (i + 1) (i + 1) else 0.)
  in
  let got = execute ~name:"fp_gated" opt ~seed:[ (x, xs); (o, blank n) ] ~read:[ o ] in
  p "gated-read: executed values are the shifted diagonal under the gate" (same got [ expected ])

let () =
  case_diagonal_reduction ();
  case_visit_cap ();
  case_full_reader ();
  case_guarded_reader ();
  case_preference ();
  case_inherited ();
  case_single_cell ();
  case_two_readers ();
  case_candidate_consumer ();
  case_shared_loop ();
  case_key_off ();
  case_intervening_write ();
  case_reader_between_setters ();
  case_inherited_recurrence ();
  case_inherited_operand ();
  case_producer_statement_writes_input ();
  case_input_written_between_setters ();
  case_rejection_after_footprint ();
  case_local_written_between_setters ();
  case_preference_ineligible ();
  case_dead_loop_writer ();
  case_local_in_producer_statement ();
  case_gated_read ();
  Stdio.printf "%!"
