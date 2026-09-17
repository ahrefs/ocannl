(* gh-ocannl-616: footprint-scoped materialization — the middle ground between recomputing a virtual
   node at every read site and materializing its whole domain.

   When a heuristic cap (the reduction cap here, the visit cap, the fan-in cap) refuses to inline a
   node, the virtualizer used to have exactly one landing spot: [Never_virtual], a full-domain
   buffer plus a full pass to fill it. This test pins the cheaper one. For a node every read of
   which is an affine sub-image read — the motivating diagonal reader [o[i] = a[i, i]] of an [n×n]
   reduction — the node stays virtual, and the reader gets a fresh routine-private scratch shaped
   like ITS iteration box ([n] cells), filled just ahead of the reader's statement by a prologue
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
  Stdio.printf "%!"
