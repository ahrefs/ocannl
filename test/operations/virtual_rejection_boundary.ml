(* Characterization test for gh-ocannl-658: WHICH candidate shapes the virtualizer refuses to
   inline, and WHERE each refusal is decided.

   Four phases record into the same placements table, and nothing but the recorded provenance says
   which one spoke:

   - [decide_placements] applies the heuristic caps BEFORE any legality question is asked
   (provenances [1:visit-cap] / uncovered read, [39:inline-reduction-cap], [41:inline-fanin-cap]).
   These are flippable policy: a shape capped here may be perfectly inlineable. -
   [check_and_store_virtual] rejects at STORE time, when the candidate's computation is captured. -
   [inline_computation] rejects at CONSUMPTION time, once a read site's indices are known — so a
   shape that stores fine still materializes if no read site can be served. - [cleanup_virtual_llc]
   commits a surviving read as provenance [17:surviving-read]; that is not a rejection, it is the
   absence of one.

   Before this test the boundary was documented only by scattered [Non_virtual] code comments and,
   in one case, wrongly: building "a node that becomes non-virtual after decide_placements" for the
   gh-ocannl-618 review took three failed constructions before one worked. Every row below was
   MEASURED against the live optimizer, not read off those comments.

   Each row carries an executed leg whose expected values are stated independently of the placement,
   plus the differential arm (the same program re-specialized with the candidate pre-decided
   [On_device]). That is deliberate: when a shape here becomes inlineable, the only edit the row
   needs is its [~verdict] — [~expected] and the differential arm carry over unchanged, and go on to
   pin that inlining PRESERVED the semantics the materialized reading has today. A row is therefore
   a boundary marker and a semantics test at once, and moving the boundary cannot quietly move the
   values.

   The shapes are hand-built [Ir.Low_level.t] through the [Ll_test] harness (gh-ocannl-600): most of
   them are unreachable through [Assignments], which gives each assignment its own loop nest.

   Codes with no minimal shape here, and why — each of these was tried, not assumed. They are cited
   by tag, which is a spelling test/operations/provenance_tag_inventory reads, so a change to where
   one can fire finds this list on its checklist.

   - [5:index-not-groundable] (a symbol no call site can ground) is preempted: a single-symbol
   affine position is injective unless its coefficient is zero, so a non-injective map arrives as
   [51:affine-not-injective] (multi-symbol) or [52:concat-index]. - [52:concat-index] (a [Concat]
   LHS position) is unreachable from this side. [trace_node_facts] runs first and raises
   [invalid_arg] on a [Concat] index outright, so the virtualizer's arm never sees one. -
   [11:already-non-virtual] (already decided materialized) fires, but records nothing of its own:
   the placement it finds is the one it keeps, so the provenance a test would read is the earlier
   decision's. That is the [~materialized] arm every row below already runs. - [12:no-setter] (no
   setter in the captured subtree) cannot fire: every call site is a setter arm, or a candidate
   drawn from the assignment-index map, which is where its setters put it. - [8:staged-compilation],
   [141:workgroup-barrier], [143:tile-mma] and [144:dynamic-write] guard constructors no
   pre-virtualization pass emits (staged compilation, barriers, cooperative tiles, dynamic
   scatters). These will never become inlineable, so a row would pin nothing that could move. -
   [14:empty-inlined-body], [140:vector-setter-unsupported], [145:lane-extract-layout] and
   [146:lane-extract-counter] belong to the vector-store (packed-uniform) consumption path,
   exercised through the uniform tests rather than by hand. - [149:scan-at-inline] is the
   consumption-time backstop behind [148:scan-recurrence], unreachable while storage refuses a scan
   first. - [154:capture-reads-merge], [155:capture-reads-local] and [156:capture-read-uninlinable]
   need a template or a placement left by an EARLIER routine of the lineage, which a single-routine
   row cannot build; test/operations/stored_template_elision.ml pins them with executed parity.

   [19:declare-local] was in that list and is not: gh-ocannl-483 put the algebraic rewrite tier
   AHEAD of [Low_level.optimize], and [Online_softmax.hoist] declares its cached probability cell as
   a [Declare_local], so the arm fires on the ordinary path with [online_softmax] on. It is rowed
   below ([row_hoisted_local]) — what the walk refuses is the CONSTRUCTOR, so a plain hoisted local
   reaches it without reconstructing the rewrite. *)

open Base
open Ll_test
open Verdict.Claims

let mk = node_factory ~first_id:2700 ~dims:[| 4 |] ()
let n = 4

type phase = Cap | Coverage | Store | Consumption

let equal_phase a b =
  match (a, b) with
  | Cap, Cap | Coverage, Coverage | Store, Store | Consumption, Consumption -> true
  | (Cap | Coverage | Store | Consumption), _ -> false

type verdict = Rejected of phase * Ir.Tnode.provenance | Accepted

(* The phase table: where each tag a row exercises is decided. Written down here because the phase
   has no exported owner a test could ask -- the store- and consumption-time tags are [Site]
   literals at their raise sites, in two functions of [low_level.ml] -- and held to those raise
   sites from outside: test/operations/provenance_tag_inventory reads this binding and refuses an
   entry whose phase's function does not mint its tag (Store is [check_and_store_virtual],
   Consumption is [instantiate_computations], [inline_computation]'s core), as it refuses a tag
   minted in both. Keep it a list of literal pairs, which is what that reader accepts.

   An entry no row exercises is refused below, so the table cannot double as documentation of the
   codes this test does not reach: the header lists those, and the inventory's golden lists every
   code by the function minting it. A new row adds its entry here; a row removed takes its entry.

   The Cap arm is not in the table: [Low_level] owns the cap set and answers for it exhaustively
   ([is_cap_provenance]), so a fourth cap is classified here without an edit -- and cannot be
   classified as anything else. Since gh-ocannl-609 each tag also carries its own reason, so what is
   left to pin is the PHASE that minted it. Nor is the Coverage arm: [36:read-before-write] is a
   constructor, not a [Site], and [decide_placements] alone commits it -- a node whose reads the
   routine's own earlier writes do not cover owns a buffer with its entry values, so no candidate
   question is asked of it at all. *)
let phase_table =
  [
    ("4:lhs-idcs-differ", Store);
    ("7:sibling-escaping-write-index", Store);
    ("9:sibling-escaping-read-index", Store);
    ("10:escaping-index-symbol", Store);
    ("19:declare-local", Store);
    ("51:affine-not-injective", Store);
    ("142:guarded-computation", Store);
    ("147:enclosing-repetition-loop", Store);
    ("148:scan-recurrence", Store);
    ("13:call-site-index-mismatch", Consumption);
  ]

let phase_of_code tag =
  if Ir.Low_level.is_cap_provenance tag then Some Cap
  else
    match tag with
    | Ir.Tnode.Read_before_write -> Some Coverage
    | Ir.Tnode.Site site -> List.Assoc.find phase_table ~equal:String.equal site
    | _ -> None

(* The tags some row's verdict named, for the exercised-only rule. *)
let exercised = Hash_set.create (module String)

let phase_name = function
  | Cap -> "by a heuristic cap"
  | Coverage -> "by the coverage rule"
  | Store -> "at store time"
  | Consumption -> "at consumption time"

let claim = function
  | Accepted -> "inlined"
  | Rejected (ph, code) ->
      Printf.sprintf "rejected %s as %s" (phase_name ph) (Ir.Tnode.provenance_to_string code)

(* One row: assert where the verdict was decided, then execute both readings of the same program. *)
let row ~label ~llc ~cand ~out ~seed ~expected ~verdict =
  let o = optimize ~name:label llc in
  (match verdict with
  | Rejected (_, Ir.Tnode.Site site) -> Hash_set.add exercised site
  | Rejected _ | Accepted -> ());
  let placed =
    match verdict with
    | Accepted -> known_virtual o cand && count_get o cand = 0
    | Rejected (ph, code) ->
        known_non_virtual o cand
        && count_get o cand >= 1
        && Option.equal Ir.Tnode.equal_provenance (rejection_code o cand) (Some code)
        && Option.equal equal_phase (phase_of_code code) (Some ph)
  in
  p (label ^ ": " ^ claim verdict) placed;
  let len = Array.length expected in
  let seed = (out, blank len) :: seed in
  let got = execute ~name:label o ~seed ~read:[ out ] in
  let mat =
    execute ~name:(label ^ "_mat")
      (optimize ~materialized:[ cand ] ~name:(label ^ "_mat") llc)
      ~seed ~read:[ out ]
  in
  p (label ^ ": executed values are the reference") (same got [ expected ]);
  p (label ^ ": the two placements agree") (same got mat)

(* === Heuristic caps, ahead of every legality question === *)

(* A scalar accumulator read once per consumer cell: the reads are [max_visits] apart, so the visit
   cap speaks before the store phase can look at the shape at all. *)
let row_visit_cap () =
  let a = mk "vcap_a" and x = mk ~dims:[| 1 |] "vcap_x" and out = mk "vcap_out" in
  materialize a;
  materialize out;
  let s = sym () and t = sym () in
  let llc =
    seq (zero x)
      (seq
         (loop_n s n (set x [| fixed 0 |] (add (get x [| fixed 0 |]) (get a [| iter s |]))))
         (loop_n t n (set out [| iter t |] (get x [| fixed 0 |]))))
  in
  row ~label:"visit_cap" ~llc ~cand:x ~out
    ~seed:[ (a, [| 1.; 2.; 3.; 4. |]) ]
    ~expected:(Array.create ~len:n 10.)
    ~verdict:(Rejected (Cap, Ir.Tnode.Visit_cap))

(* === Store-time rejections === *)

(* gh-ocannl-651: the guard ENCLOSES the candidate's nest, so it is outside the captured subtree —
   [virtual_llc] reports it rather than the walk finding it. [virtual_guarded_setter.ml] runs this
   shape at both flag values; here it is one row of the boundary. *)
let row_guarded_enclosing () =
  let flag = mk ~dims:[| 1 |] "genc_flag" and x = mk "genc_x" and out = mk "genc_out" in
  materialize flag;
  materialize out;
  let s = sym () and t = sym () in
  let llc =
    seq (zero x)
      (seq
         (if_ (get flag [| fixed 0 |]) (loop_n s n (set x [| iter s |] (tick s))))
         (loop_n t n (set out [| iter t |] (get x [| iter t |]))))
  in
  row ~label:"guard_enclosing" ~llc ~cand:x ~out
    ~seed:[ (flag, [| 0. |]) ]
    ~expected:(Array.create ~len:n 0.)
    ~verdict:(Rejected (Store, Ir.Tnode.Site "142:guarded-computation"))

(* The guard is INTERIOR to the captured subtree, so the walk's own [If] arm finds it. Same code,
   different arm. *)
let row_guarded_interior () =
  let mask = mk "gint_mask" and x = mk "gint_x" and out = mk "gint_out" in
  materialize mask;
  materialize out;
  let s = sym () and t = sym () in
  let llc =
    seq (zero x)
      (seq
         (loop_n s n (if_ (get mask [| iter s |]) (set x [| iter s |] (tick s))))
         (loop_n t n (set out [| iter t |] (get x [| iter t |]))))
  in
  row ~label:"guard_interior" ~llc ~cand:x ~out
    ~seed:[ (mask, Array.init n ~f:(fun i -> Float.of_int (i % 2))) ]
    ~expected:(Array.init n ~f:(fun i -> if i % 2 = 1 then 1. +. Float.of_int i else 0.))
    ~verdict:(Rejected (Store, Ir.Tnode.Site "142:guarded-computation"))

(* An enclosing loop whose symbol the index map does not mention replays the nest into the same
   cells. Capture happens at the outermost loop whose symbol DOES occur, so this repetition loop
   sits above the captured subtree and inlining would apply the update once instead of twice. *)
let row_repetition_above () =
  let x = mk "rept_x" and out = mk "rept_out" in
  materialize out;
  let k = sym () and t = sym () and u = sym () in
  let llc =
    seq (zero x)
      (seq
         (loop_n k 2 (loop_n t n (set x [| iter t |] (add (get x [| iter t |]) (c 1.)))))
         (loop_n u n (set out [| iter u |] (get x [| iter u |]))))
  in
  row ~label:"repetition_above_capture" ~llc ~cand:x ~out ~seed:[]
    ~expected:(Array.create ~len:n 2.)
    ~verdict:(Rejected (Store, Ir.Tnode.Site "147:enclosing-repetition-loop"))

(* gh-ocannl-2149: a symbol-free index map now captures at the outermost enclosing loop, retaining
   the whole reduction. The producer varies with its iterator, clear of the zero init. *)
let row_reduction_symbol_free () =
  let x = mk ~dims:[| 1 |] "rsf_x" and out = mk ~dims:[| 1 |] "rsf_out" in
  materialize out;
  let s = sym () in
  let llc =
    seq (zero x)
      (seq
         (loop_n s n (set x [| fixed 0 |] (add (get x [| fixed 0 |]) (tick s))))
         (set out [| fixed 0 |] (get x [| fixed 0 |])))
  in
  row ~label:"reduction_symbol_free" ~llc ~cand:x ~out ~seed:[] ~expected:[| 10. |]
    ~verdict:Accepted

(* Capturing a scalar's whole loop must not make a sibling observe the final sum at each iteration:
   the original program exposes a different prefix after each update. *)
let row_symbol_free_observer () =
  let x = mk ~dims:[| 1 |] "sfo_x" and out = mk "sfo_out" in
  materialize out;
  let k = sym () in
  let llc =
    seq (zero x)
      (loop_n k n
         (seq
            (set x [| fixed 0 |] (add (get x [| fixed 0 |]) (tick k)))
            (set out [| iter k |] (get x [| fixed 0 |]))))
  in
  (* Reach legality even though this accumulator is read once per reduction iteration. *)
  let settings = Ir.Low_level.virtualize_settings in
  let saved = settings.max_visits in
  settings.max_visits <- Int.max_value;
  Exn.protect
    ~finally:(fun () -> settings.max_visits <- saved)
    ~f:(fun () ->
      row ~label:"symbol_free_intermediate_observer" ~llc ~cand:x ~out ~seed:[]
        ~expected:[| 1.; 3.; 6.; 10. |]
        ~verdict:(Rejected (Store, Ir.Tnode.Site "10:escaping-index-symbol"));
      (* A statement-local accumulator is also an observer: sum the prefixes in a register, then
         expose that sum after the loop. A census of tensor-buffer writers alone misses it. *)
      let local_tn = mk ~dims:[| 1 |] "sfo_local"
      and local_out = mk ~dims:[| 1 |] "sfo_local_out" in
      virtualize local_tn;
      materialize local_out;
      let id = LL.get_scope local_tn in
      let llc =
        seq (declare_local id)
          (seq
             (LL.Set_local (id, c 0.))
             (seq (zero x)
                (seq
                   (loop_n k n
                      (seq
                         (set x [| fixed 0 |] (add (get x [| fixed 0 |]) (tick k)))
                         (LL.Set_local (id, add (LL.Get_local id) (get x [| fixed 0 |])))))
                   (set local_out [| fixed 0 |] (LL.Get_local id)))))
      in
      row ~label:"symbol_free_local_observer" ~llc ~cand:x ~out:local_out ~seed:[]
        ~expected:[| 20. |]
        ~verdict:(Rejected (Store, Ir.Tnode.Site "10:escaping-index-symbol")))

(* Two setters at different index maps WITHIN one captured subtree. The per-invocation index map is
   what [Non_virtual 4] compares, so this rejects while the same two setters as separate statements
   (below) do not. *)
let row_two_setters_one_subtree () =
  let x = mk "two_x" and out = mk "two_out" in
  materialize out;
  let s = sym () and t = sym () in
  let llc =
    seq
      (loop_n s n (seq (set x [| iter s |] (tick s)) (set x [| fixed 0 |] (c 9.))))
      (loop_n t n (set out [| iter t |] (get x [| iter t |])))
  in
  row ~label:"two_setters_one_subtree" ~llc ~cand:x ~out ~seed:[] ~expected:[| 9.; 2.; 3.; 4. |]
    ~verdict:(Rejected (Store, Ir.Tnode.Site "4:lhs-idcs-differ"))

(* A sibling READ at a symbol bound outside the captured subtree: inlining would move the read to a
   site where that symbol does not exist. Note this arm fires BEFORE the enclosing-loop check, which
   is why an array reduction reports 9 and the constant reduction above reports 147. *)
let row_escaping_read () =
  let b = mk "esc_b" and x = mk "esc_x" and out = mk "esc_out" in
  materialize b;
  materialize out;
  let k = sym () and s = sym () and t = sym () in
  let llc =
    seq (zero x)
      (seq
         (loop_n k 2
            (loop_n s n (set x [| iter s |] (add (get x [| iter s |]) (get b [| iter k |])))))
         (loop_n t n (set out [| iter t |] (get x [| iter t |]))))
  in
  row ~label:"escaping_sibling_read" ~llc ~cand:x ~out
    ~seed:[ (b, [| 1.; 10.; 0.; 0. |]) ]
    ~expected:(Array.create ~len:n 11.)
    ~verdict:(Rejected (Store, Ir.Tnode.Site "9:sibling-escaping-read-index"))

(* The same escape through a sibling SETTER rather than a sibling read: a distinct arm, and the
   reason it exists is the same — the write would move to a site where its symbol does not exist. *)
let row_escaping_setter () =
  let x = mk "eset_x" and side = mk "eset_side" and out = mk "eset_out" in
  materialize side;
  materialize out;
  let k = sym () and s = sym () and t = sym () in
  let llc =
    seq
      (loop_n k 2
         (loop_n s n (seq (set x [| iter s |] (tick s)) (set side [| iter k |] (ramp 5. k)))))
      (loop_n t n (set out [| iter t |] (get x [| iter t |])))
  in
  row ~label:"escaping_sibling_setter" ~llc ~cand:x ~out
    ~seed:[ (side, blank 2) ]
    ~expected:(Array.init n ~f:(fun i -> 1. +. Float.of_int i))
    ~verdict:(Rejected (Store, Ir.Tnode.Site "7:sibling-escaping-write-index"))

(* The same escape through an [Embed_index] rather than an array read. *)
let row_escaping_embed () =
  let x = mk "eemb_x" and out = mk "eemb_out" in
  materialize out;
  let k = sym () and s = sym () and t = sym () in
  let llc =
    seq (zero x)
      (seq
         (loop_n k 2
            (loop_n s n (set x [| iter s |] (add (get x [| iter s |]) (add (c 1.) (embed k))))))
         (loop_n t n (set out [| iter t |] (get x [| iter t |]))))
  in
  row ~label:"escaping_embed_index" ~llc ~cand:x ~out ~seed:[] ~expected:(Array.create ~len:n 3.)
    ~verdict:(Rejected (Store, Ir.Tnode.Site "10:escaping-index-symbol"))

(* A multi-symbol affine LHS position that is not injective: dropping the producer loops would fold
   a fiber away. The injective siblings of this shape are [virtual_affine.ml]'s accepted cases. *)
let row_noninjective () =
  let x = mk ~dims:[| 5 |] "nij_x" and out = mk ~dims:[| 5 |] "nij_out" in
  materialize out;
  let i = sym () and j = sym () and a = sym () and b = sym () in
  let llc =
    seq (zero x)
      (seq
         (loop_n i 3 (loop_n j 3 (set x [| aff [ (1, i); (1, j) ] 0 |] (tag i j))))
         (loop_n a 3
            (loop_n b 3
               (set out [| aff [ (1, a); (1, b) ] 0 |] (get x [| aff [ (1, a); (1, b) ] 0 |])))))
  in
  row ~label:"noninjective_multiaffine" ~llc ~cand:x ~out ~seed:[]
    ~expected:[| 1.; 11.; 21.; 22.; 23. |]
    ~verdict:(Rejected (Store, Ir.Tnode.Site "51:affine-not-injective"))

(* === Fixed-index producer components (gh-ocannl-1735) === *)

(* Two setters at different index maps as SEPARATE statements store fine, as two components -- the
   index-map comparison is per invocation -- and the verdict waits for a read site. The fixed-index
   component is served at the symbolic read by an equality guard ([0 == t]): its fallback is the
   covering component's value, which the components' union (here the loop alone) proves defined at
   every cell. *)
let row_fixed_component () =
  let x = mk "fcmp_x" and out = mk "fcmp_out" in
  materialize out;
  let s = sym () and t = sym () in
  let llc =
    seq
      (loop_n s n (set x [| iter s |] (tick s)))
      (seq (set x [| fixed 0 |] (c 9.)) (loop_n t n (set out [| iter t |] (get x [| iter t |]))))
  in
  row ~label:"fixed_component_symbolic_read" ~llc ~cand:x ~out ~seed:[]
    ~expected:[| 9.; 2.; 3.; 4. |] ~verdict:Accepted

(* The guard compares the producer's coordinate, not 0: a component at [Fixed_idx 2] lands on cell
   2, so a guard spelled against the first cell fails the oracle. *)
let row_fixed_component_offset () =
  let x = mk "fcof_x" and out = mk "fcof_out" in
  materialize out;
  let s = sym () and t = sym () in
  let llc =
    seq
      (loop_n s n (set x [| iter s |] (tick s)))
      (seq (set x [| fixed 2 |] (c 9.)) (loop_n t n (set out [| iter t |] (get x [| iter t |]))))
  in
  row ~label:"fixed_component_offset" ~llc ~cand:x ~out ~seed:[] ~expected:[| 1.; 2.; 9.; 4. |]
    ~verdict:Accepted

(* An affine reader: [out[t] = x[t + 1]] meets the fixed component at [t = 1]. *)
let row_fixed_component_affine_read () =
  let x = mk "fcaf_x" and out = mk ~dims:[| n - 1 |] "fcaf_out" in
  materialize out;
  let s = sym () and t = sym () in
  let llc =
    seq
      (loop_n s n (set x [| iter s |] (tick s)))
      (seq
         (set x [| fixed 2 |] (c 9.))
         (loop_n t (n - 1) (set out [| iter t |] (get x [| aff [ (1, t) ] 1 |]))))
  in
  row ~label:"fixed_component_affine_read" ~llc ~cand:x ~out ~seed:[] ~expected:[| 2.; 9.; 4. |]
    ~verdict:Accepted

(* The covering writer AFTER the fixed one overwrites every cell, the fixed one included, so the
   fixed write is dead -- in both readings. Order among components does not matter to the coverage
   proof, and this row executes the order the other rows do not. *)
let row_fixed_component_first () =
  let x = mk "fcfi_x" and out = mk "fcfi_out" in
  materialize out;
  let s = sym () and t = sym () in
  let llc =
    seq
      (set x [| fixed 1 |] (c 9.))
      (seq
         (loop_n s n (set x [| iter s |] (tick s)))
         (loop_n t n (set out [| iter t |] (get x [| iter t |]))))
  in
  row ~label:"fixed_component_first" ~llc ~cand:x ~out ~seed:[] ~expected:[| 1.; 2.; 3.; 4. |]
    ~verdict:Accepted

(* An update rather than a store: the fixed component reads the cell's earlier value, which is the
   guard's own fallback local, so the guarded arm adds to the covering writer's value at cell 0. *)
let row_fixed_component_accumulating () =
  let x = mk "fcac_x" and out = mk "fcac_out" in
  materialize out;
  let s = sym () and t = sym () in
  let llc =
    seq
      (loop_n s n (set x [| iter s |] (tick s)))
      (seq
         (set x [| fixed 0 |] (add (get x [| fixed 0 |]) (c 9.)))
         (loop_n t n (set out [| iter t |] (get x [| iter t |]))))
  in
  row ~label:"fixed_component_accumulating" ~llc ~cand:x ~out ~seed:[]
    ~expected:[| 10.; 2.; 3.; 4. |] ~verdict:Accepted

(* The union of a node [decide_placements] decides counts in full: the coverage rule already made
   any read of an entry value an input, so a setter reading the node ends nothing here. Cell 3 is
   written after the component accumulating cell 2, so the plain prefix a coverage-exempt node is
   held to (gh-ocannl-1954) would not cover this node, and the fixed guards would refuse. *)
let row_fixed_component_accumulating_inside () =
  let x = mk "fcai_x" and out = mk "fcai_out" in
  materialize out;
  let t = sym () in
  let llc =
    List.reduce_exn ~f:seq
      [
        set x [| fixed 0 |] (c 31.);
        set x [| fixed 1 |] (c 32.);
        set x [| fixed 2 |] (c 33.);
        set x [| fixed 2 |] (add (get x [| fixed 2 |]) (c 5.));
        set x [| fixed 3 |] (c 34.);
        loop_n t n (set out [| iter t |] (add (get x [| iter t |]) (tick t)));
      ]
  in
  row ~label:"fixed_component_accumulating_inside" ~llc ~cand:x ~out ~seed:[]
    ~expected:[| 32.; 34.; 41.; 38. |] ~verdict:Accepted

(* No covering writer, but a [Zero_out]: every other cell is the zero the materialized buffer holds
   too. The consumer adds [tick t], so the zero cells stay distinct from one another. *)
let row_fixed_component_zero_init () =
  let x = mk "fczi_x" and out = mk "fczi_out" in
  materialize out;
  let t = sym () in
  let llc =
    seq (zero x)
      (seq
         (set x [| fixed 2 |] (c 9.))
         (loop_n t n (set out [| iter t |] (add (get x [| iter t |]) (tick t)))))
  in
  row ~label:"fixed_component_zero_init" ~llc ~cand:x ~out ~seed:[] ~expected:[| 1.; 2.; 12.; 4. |]
    ~verdict:Accepted

(* Neither: the producer writes cell 2 only, so every other cell the reader reaches holds the
   buffer's ENTRY value. A guard falling back on an invented zero would print [0] there; the
   coverage rule instead places the node as an input, and the seeded entries survive. *)
let row_fixed_component_uninitialized () =
  let x = mk "fcun_x" and out = mk "fcun_out" in
  materialize out;
  let t = sym () in
  let llc =
    seq (set x [| fixed 2 |] (c 9.)) (loop_n t n (set out [| iter t |] (get x [| iter t |])))
  in
  row ~label:"fixed_component_uninitialized" ~llc ~cand:x ~out
    ~seed:[ (x, [| 101.; 102.; 103.; 104. |]) ]
    ~expected:[| 101.; 102.; 9.; 104. |]
    ~verdict:(Rejected (Coverage, Ir.Tnode.Read_before_write))

(* The proof is what a DECLARED-virtual node leans on: it has no materialization to fall back on and
   skips the coverage rule, so a guard over an uncovered cell would print a zero no write put there.
   With a [Zero_out] the guard is sound and the node inlines; without one, the fixed component still
   refuses -- loudly, as a declared-virtual node refuses any unservable read. *)
let declared_virtual_fixed_component () =
  let build label ~zeroed =
    let x = mk (label ^ "_x") and out = mk (label ^ "_out") in
    virtualize x;
    materialize out;
    let t = sym () in
    let producer = set x [| fixed 2 |] (c 9.) in
    let llc =
      seq
        (if zeroed then seq (zero x) producer else producer)
        (loop_n t n (set out [| iter t |] (add (get x [| iter t |]) (tick t))))
    in
    (x, out, llc)
  in
  let x, out, llc = build "dvfc_zeroed" ~zeroed:true in
  let o = optimize ~name:"dvfc_zeroed" llc in
  p "declared_virtual_zeroed: inlined" (known_virtual o x && count_get o x = 0);
  p "declared_virtual_zeroed: executed values are the reference"
    (same
       (execute ~name:"dvfc_zeroed" o ~seed:[ (out, blank n) ] ~read:[ out ])
       [ [| 1.; 2.; 12.; 4. |] ]);
  let _, _, llc = build "dvfc_uninit" ~zeroed:false in
  p "declared_virtual_uninitialized: the fixed component refuses rather than reading a zero"
    (match optimize ~name:"dvfc_uninit" llc with
    | (_ : Ir.Low_level.optimized) -> false
    | exception Utils.User_error msg ->
        String.is_substring msg ~substring:"157:guard-fallback-unproven")

(* === Disjoint components and unions of partial covers (gh-ocannl-1795) === *)

(* A fixed component read at a DIFFERENT fixed cell never writes it: the component is omitted at
   that read site, and the covering loop alone serves it. Before gh-ocannl-1795 this static mismatch
   refused the whole node as 13. *)
let row_fixed_component_fixed_read () =
  let x = mk "fcfr_x" and out = mk ~dims:[| 1 |] "fcfr_out" in
  materialize out;
  let s = sym () in
  let llc =
    seq
      (loop_n s n (set x [| iter s |] (tick s)))
      (seq (set x [| fixed 0 |] (c 9.)) (set out [| fixed 0 |] (get x [| fixed 2 |])))
  in
  row ~label:"fixed_component_fixed_read" ~llc ~cand:x ~out ~seed:[] ~expected:[| 3. |]
    ~verdict:Accepted

(* Components that cover the node only TOGETHER -- a loop over cells 0..2 and a fixed write at 3:
   their union defines every cell, so the fixed component's guard [3 == t] has a fallback, and the
   loop's range guard [0 <= t < 3] keeps it off cell 3. *)
let row_fixed_component_partial_cover () =
  let x = mk "fcpc_x" and out = mk "fcpc_out" in
  materialize out;
  let s = sym () and t = sym () in
  let llc =
    seq
      (loop_n s (n - 1) (set x [| iter s |] (tick s)))
      (seq (set x [| fixed 3 |] (c 9.)) (loop_n t n (set out [| iter t |] (get x [| iter t |]))))
  in
  row ~label:"fixed_component_partial_cover" ~llc ~cand:x ~out ~seed:[]
    ~expected:[| 1.; 2.; 3.; 9. |] ~verdict:Accepted

(* Cell-by-cell writers and nothing else: no loop, no [Zero_out], every component a guard, and the
   union of four single cells the whole node. Each cell reads a materialized [src], so the node is
   computed, not a literal table (next row). *)
let row_fixed_component_cells () =
  let x = mk "fcce_x" and src = mk ~dims:[| 1 |] "fcce_src" and out = mk "fcce_out" in
  materialize src;
  materialize out;
  let t = sym () in
  let cells =
    List.init n ~f:(fun k ->
        set x [| fixed k |] (add (c (31. +. Float.of_int k)) (get src [| fixed 0 |])))
  in
  let llc =
    List.reduce_exn ~f:seq (cells @ [ loop_n t n (set out [| iter t |] (get x [| iter t |])) ])
  in
  row ~label:"fixed_component_cells" ~llc ~cand:x ~out
    ~seed:[ (src, [| 100. |]) ]
    ~expected:[| 131.; 132.; 133.; 134. |] ~verdict:Accepted

(* The same cells as constants: a literal table, the in-kernel spelling of a [Tensor.ndarray] or
   [Tensor.term_init] literal. Its symbolic read stays a buffer read -- a guard per cell would be
   the cost, and a parameter's initial values would become code -- refused as 13 as before. *)
let row_fixed_component_literal_table () =
  let x = mk "fclt_x" and out = mk "fclt_out" in
  materialize out;
  let t = sym () in
  let cells = List.init n ~f:(fun k -> set x [| fixed k |] (c (31. +. Float.of_int k))) in
  let llc =
    List.reduce_exn ~f:seq (cells @ [ loop_n t n (set out [| iter t |] (get x [| iter t |])) ])
  in
  row ~label:"fixed_component_literal_table" ~llc ~cand:x ~out ~seed:[]
    ~expected:[| 31.; 32.; 33.; 34. |]
    ~verdict:(Rejected (Consumption, Ir.Tnode.Site "13:call-site-index-mismatch"))

(* Overlaps in both orders over a piecewise cover: [x[1] := 9] is overwritten by the later loop over
   cells 0..1; [x[0] += 100] after the loop accumulates onto its value; cell 3 is stored, then
   accumulated. The replay applies the components oldest first over one shared local, so the last
   writer wins and each accumulation reads the value before it. *)
let row_fixed_component_overlaps () =
  let x = mk "fcov_x" and out = mk "fcov_out" in
  materialize out;
  let s = sym () and t = sym () in
  let llc =
    List.reduce_exn ~f:seq
      [
        set x [| fixed 1 |] (c 9.);
        loop_n s 2 (set x [| iter s |] (tick s));
        set x [| fixed 2 |] (c 50.);
        set x [| fixed 3 |] (c 60.);
        set x [| fixed 0 |] (add (get x [| fixed 0 |]) (c 100.));
        set x [| fixed 3 |] (add (get x [| fixed 3 |]) (c 5.));
        loop_n t n (set out [| iter t |] (get x [| iter t |]));
      ]
  in
  row ~label:"fixed_component_overlaps" ~llc ~cand:x ~out ~seed:[]
    ~expected:[| 101.; 2.; 50.; 65. |] ~verdict:Accepted

(* Two dimensions: rows written by separate loops, and a fixed cell of row 1. Read along row 1 the
   cell is guarded ([2 == t]) and row 0's loop is omitted; read along row 0, both of row 1's
   components are omitted. No single component covers the node; the row loops do together. *)
let row_fixed_component_rows ~read_row =
  let label = Printf.sprintf "fixed_component_rows_read%d" read_row in
  let x = mk ~dims:[| 2; n |] (label ^ "_x") and out = mk (label ^ "_out") in
  materialize out;
  let s = sym () and s2 = sym () and t = sym () in
  let llc =
    List.reduce_exn ~f:seq
      [
        loop_n s n (set x [| fixed 0; iter s |] (ramp 10. s));
        loop_n s2 n (set x [| fixed 1; iter s2 |] (ramp 20. s2));
        set x [| fixed 1; fixed 2 |] (c 9.);
        loop_n t n (set out [| iter t |] (get x [| fixed read_row; iter t |]));
      ]
  in
  row ~label ~llc ~cand:x ~out ~seed:[]
    ~expected:(if read_row = 0 then [| 10.; 11.; 12.; 13. |] else [| 20.; 21.; 9.; 23. |])
    ~verdict:Accepted

(* Omission is decided for the whole component before any of its positions is guarded: [x[1; 2]]
   read along row 0 is disjoint at its first position, while its second ([2 == t]) would need a
   fallback this node cannot prove -- row 1 is otherwise never written. Processed position by
   position, the second would refuse the read site as 13. *)
let row_fixed_component_disjoint_first () =
  let x = mk ~dims:[| 2; n |] "fcdf_x" and out = mk "fcdf_out" in
  materialize out;
  let s = sym () and t = sym () in
  let llc =
    List.reduce_exn ~f:seq
      [
        loop_n s n (set x [| fixed 0; iter s |] (ramp 10. s));
        set x [| fixed 1; fixed 2 |] (c 9.);
        loop_n t n (set out [| iter t |] (get x [| fixed 0; iter t |]));
      ]
  in
  row ~label:"fixed_component_disjoint_first" ~llc ~cand:x ~out ~seed:[]
    ~expected:[| 10.; 11.; 12.; 13. |] ~verdict:Accepted

(* === Consumption-time rejection === *)

(* The proof's conservative edge: the components cover every cell the READ reaches (0..2), so the
   coverage rule admits the node, but not the node -- cell 3 is never written. The proof is about
   the node, since a read site's range is not known where the template is instantiated, so the fixed
   component's guard has no fallback and the read site refuses; the node materializes, the correct
   (if costlier) reading. *)
let row_fixed_component_read_covered_only () =
  let x = mk "fcro_x" and out = mk ~dims:[| n - 1 |] "fcro_out" in
  materialize out;
  let s = sym () and t = sym () in
  let llc =
    seq
      (loop_n s 2 (set x [| iter s |] (tick s)))
      (seq
         (set x [| fixed 2 |] (c 9.))
         (loop_n t (n - 1) (set out [| iter t |] (get x [| iter t |]))))
  in
  row ~label:"fixed_component_read_covered_only" ~llc ~cand:x ~out ~seed:[]
    ~expected:[| 1.; 2.; 9. |]
    ~verdict:(Rejected (Consumption, Ir.Tnode.Site "13:call-site-index-mismatch"))

(* Declared-virtual nodes skip the coverage rule and have no materialization to fall back on, so
   they are where the union proof alone decides: a complete union inlines, and every way of not
   covering the node refuses -- loudly -- rather than reading a zero no write put there. *)
let declared_virtual_union () =
  (* Cells read a materialized [src] (seeded 100), so no node here is a literal table. *)
  let src = mk ~dims:[| 1 |] "dvu_src" in
  materialize src;
  let case ?(width = n) ~label ~dims ~writes ~read_at expect =
    let x = mk ~dims (label ^ "_x") and out = mk (label ^ "_out") in
    virtualize x;
    materialize out;
    let t = sym () in
    let llc =
      List.reduce_exn ~f:seq
        (writes x @ [ loop_n t width (set out [| iter t |] (add (get x (read_at t)) (tick t))) ])
    in
    match expect with
    | `Inlined expected -> (
        (* A declared node's refusal raises: a failed claim, not the end of every row after it. *)
        match optimize ~name:label llc with
        | exception Utils.User_error _ -> p (label ^ ": inlined") false
        | o ->
            p (label ^ ": inlined") (known_virtual o x && count_get o x = 0);
            p
              (label ^ ": executed values are the reference")
              (same
                 (execute ~name:label o ~seed:[ (src, [| 100. |]); (out, blank n) ] ~read:[ out ])
                 [ expected ]))
    | `Refused code ->
        p
          (label ^ ": refused as " ^ code)
          (match optimize ~name:label llc with
          | (_ : Ir.Low_level.optimized) -> false
          | exception Utils.User_error msg -> String.is_substring msg ~substring:code)
  in
  let cell x k v = set x [| fixed k |] (add (c v) (get src [| fixed 0 |])) in
  let flat t = [| iter t |] in
  case ~label:"dvu_cells" ~dims:[| n |]
    ~writes:(fun x -> List.init n ~f:(fun k -> cell x k (31. +. Float.of_int k)))
    ~read_at:flat
    (`Inlined [| 132.; 134.; 136.; 138. |]);
  (* A duplicated cell instead of a missing one: four writes, three cells. *)
  case ~label:"dvu_duplicate" ~dims:[| n |]
    ~writes:(fun x -> [ cell x 0 31.; cell x 0 41.; cell x 2 33.; cell x 3 34. ])
    ~read_at:flat (`Refused "157:guard-fallback-unproven");
  (* A loop over cell 0 that never runs proves nothing about cell 0. *)
  case ~label:"dvu_dead_loop" ~dims:[| n |]
    ~writes:(fun x ->
      let s = sym () in
      loop ~upto:(-1) s (set x [| iter s |] (tick s))
      :: List.init (n - 1) ~f:(fun k -> cell x (k + 1) (32. +. Float.of_int k)))
    ~read_at:flat (`Refused "157:guard-fallback-unproven");
  (* A diagonal's bounding box is the whole [2; 2] node, but cell [1; 0] is a hole. Read along row
     1, the fixed component [x[1; 1]]'s guard [1 == t] falls back on that hole at [t = 0] (so does
     the diagonal's own [1 == t]); taking the diagonal for a box would admit it and print a zero. *)
  case ~label:"dvu_diagonal" ~width:2 ~dims:[| 2; 2 |]
    ~writes:(fun x ->
      let s = sym () in
      [ loop_n s 2 (set x [| iter s; iter s |] (tick s)); set x [| fixed 1; fixed 1 |] (c 9.) ])
    ~read_at:(fun t -> [| fixed 1; iter t |])
    (`Refused "157:guard-fallback-unproven");
  (* Every component disjoint from the fixed read: nothing is left to replay. *)
  case ~label:"dvu_all_disjoint" ~dims:[| n |]
    ~writes:(fun x -> [ cell x 0 31.; cell x 1 32. ])
    ~read_at:(fun _ -> [| fixed 3 |])
    (`Refused "13:call-site-index-mismatch");
  (* gh-ocannl-1898: a flattened read of a partial write. Normalization folds the producer's fixed
     row into the composed index ([x[1; k]] becomes [2 + k]) and a diagonal into [3s], so neither
     meets [fixed_guardable]; a partial flattened producer read per axis binds its symbol under a
     range guard. Each guard asks the fallback proof of a coverage-exempt node, and nothing covers
     the node, so each must refuse rather than read a zero no write put there. *)
  let flat_read t = [| Idx.Sub_axis; iter t |] in
  case ~label:"dvu_flat_fixed_row" ~dims:[| 2; 2 |]
    ~writes:(fun x ->
      let k = sym () in
      [ loop_n k 2 (set x [| fixed 1; iter k |] (tick k)) ])
    ~read_at:flat_read (`Refused "157:guard-fallback-unproven");
  case ~label:"dvu_flat_diagonal" ~dims:[| 2; 2 |]
    ~writes:(fun x ->
      let s = sym () in
      [ loop_n s 2 (set x [| iter s; iter s |] (tick s)) ])
    ~read_at:flat_read (`Refused "157:guard-fallback-unproven");
  case ~label:"dvu_flat_fixed_cell" ~dims:[| 2; 2 |]
    ~writes:(fun x -> [ set x [| fixed 1; fixed 1 |] (add (c 9.) (get src [| fixed 0 |])) ])
    ~read_at:flat_read (`Refused "157:guard-fallback-unproven");
  case ~label:"dvu_flat_producer_partial" ~width:2 ~dims:[| 2; 2 |]
    ~writes:(fun x ->
      let f = sym () in
      [ loop_n f 2 (set x [| Idx.Sub_axis; iter f |] (tick f)) ])
    ~read_at:(fun t -> [| fixed 1; iter t |])
    (`Refused "157:guard-fallback-unproven");
  (* gh-ocannl-1953: a declared node skips the coverage rule, and a one-component binding built no
     guard for [157] to inspect, so a read past the loop replayed the producer there. With no
     [Zero_out] the guard now has no fallback and refuses; with one it reads the zero; and a read
     inside the loop needs no guard, so it needs no proof either -- a partial declared node stays
     servable where it is written. [same] compares every cell of [out], so the cells this read
     leaves are the sentinel. *)
  let partial x =
    let s = sym () in
    loop_n s 2 (set x [| iter s |] (tick s))
  in
  case ~label:"dvu_partial_past_loop" ~dims:[| n |]
    ~writes:(fun x -> [ partial x ])
    ~read_at:flat (`Refused "157:guard-fallback-unproven");
  case ~label:"dvu_zeroed_past_loop" ~dims:[| n |]
    ~writes:(fun x -> [ zero x; partial x ])
    ~read_at:flat
    (`Inlined [| 2.; 4.; 3.; 4. |]);
  case ~label:"dvu_partial_in_loop" ~width:2 ~dims:[| n |]
    ~writes:(fun x -> [ partial x ])
    ~read_at:flat
    (`Inlined [| 2.; 4.; sentinel; sentinel |]);
  (* The union does cover the node: the proof holds, and the flattened read inlines. *)
  case ~label:"dvu_flat_covered" ~dims:[| 2; 2 |]
    ~writes:(fun x ->
      let i = sym () and k = sym () in
      [ loop_n i 2 (loop_n k 2 (set x [| iter i; iter k |] (tag i k))) ])
    ~read_at:flat_read
    (`Inlined [| 2.; 4.; 14.; 16. |])

(* gh-ocannl-1897: the same proof for EVERY guard on a node that skipped the coverage rule, not only
   a fixed-index one -- an equality guard from a repeated symbol ([x[s; s]]) or from an unsolved
   affine position with its loop kept ([x[2s]]), and a range guard from a unit solve ([x[2i + k]])
   or from the bindings of a multi-component node. Without a [Zero_out] or a covering union, each
   used to read the replay's invented [0.] at a cell no write reaches.

   The reference is a twin program over fresh nodes with [x] MATERIALIZED instead of declared
   virtual (re-specializing the declared node itself would contradict its declaration), its entries
   seeded [101 + cell], so a hole shows the entry value where the base printed a zero. Each row
   states its values in closed form, and the twin is held to them first. A refused row claims
   "refused as 157, or executed equal to the twin": red on the base, which inlined; and still a
   semantics pin should a sharper proof admit the shape. An accepted row is one whose guard fails on
   some iterations under a proved fallback, executed against the same twin. *)
let guard_fallback_code = "157:guard-fallback-unproven"

let declared_virtual_guards () =
  let numel dims = Array.fold dims ~init:1 ~f:( * ) in
  let case ~label ~dims ?(out_dims = dims) ~(build : Ir.Tnode.t -> Ir.Tnode.t -> Ir.Low_level.t)
      ~expected expect =
    let program ~declare suffix =
      let x = mk ~dims (label ^ suffix ^ "_x")
      and out = mk ~dims:out_dims (label ^ suffix ^ "_out") in
      declare x;
      materialize out;
      (x, out, build x out)
    in
    let entries = Array.init (numel dims) ~f:(fun i -> 101. +. Float.of_int i) in
    let blank_out = blank (numel out_dims) in
    let xm, outm, llc_m = program ~declare:materialize "_mat" in
    let reference =
      execute ~name:(label ^ "_mat")
        (optimize ~name:(label ^ "_mat") llc_m)
        ~seed:[ (xm, entries); (outm, blank_out) ]
        ~read:[ outm ]
    in
    p (label ^ ": the materialized twin computes the stated values") (same reference [ expected ]);
    let x, out, llc = program ~declare:virtualize "" in
    let run_virtual o = execute ~name:label o ~seed:[ (out, blank_out) ] ~read:[ out ] in
    match expect with
    | `Inlined ->
        let o = optimize ~name:label llc in
        p (label ^ ": inlined") (known_virtual o x && count_get o x = 0);
        p (label ^ ": executed values are the materialized twin's") (same (run_virtual o) reference)
    | `Refused code ->
        p
          (label ^ ": refused as " ^ code ^ ", or executed equal to the materialized twin")
          (match optimize ~name:label llc with
          | o -> same (run_virtual o) reference
          | exception Utils.User_error msg -> String.is_substring msg ~substring:code)
    | `Refused_though_sound code ->
        p
          (label ^ ": refused as " ^ code ^ ", the node-level proof's price")
          (match optimize ~name:label llc with
          | (_ : Ir.Low_level.optimized) -> false
          | exception Utils.User_error msg -> String.is_substring msg ~substring:code)
  in
  let fl = Float.of_int in
  (* Consumers add [tick]/[tag] of their own iteration, so equal producer values stay distinct. *)
  let grid_read x out =
    let j = sym () and k = sym () in
    loop_n j 3
      (loop_n k 3 (set out [| iter j; iter k |] (add (get x [| iter j; iter k |]) (tag j k))))
  in
  let flat_read ~width x out =
    let u = sym () in
    loop_n u width (set out [| iter u |] (add (get x [| iter u |]) (tick u)))
  in
  let diagonal x =
    let s = sym () in
    loop_n s 3 (set x [| iter s; iter s |] (tick s))
  in
  let grid_expected ~hole =
    Array.init 9 ~f:(fun cell ->
        let j = cell / 3 and k = cell % 3 in
        (if j = k then 1. +. fl j else hole cell) +. 1. +. (10. *. fl j) +. fl k)
  in
  (* Equality guard from a repeated symbol: [s := j], then [s == k]. *)
  case ~label:"dvg_diagonal_uninit" ~dims:[| 3; 3 |]
    ~build:(fun x out -> seq (diagonal x) (grid_read x out))
    ~expected:(grid_expected ~hole:(fun cell -> 101. +. fl cell))
    (`Refused guard_fallback_code);
  case ~label:"dvg_diagonal_zeroed" ~dims:[| 3; 3 |]
    ~build:(fun x out -> seq (zero x) (seq (diagonal x) (grid_read x out)))
    ~expected:(grid_expected ~hole:(fun _ -> 0.))
    `Inlined;
  (* Equality guard from an unsolved affine position: no unit coefficient, so the loop over [s] is
     kept and guarded by [2s == u]; the odd cells are holes. *)
  case ~label:"dvg_strided_hole" ~dims:[| n |]
    ~build:(fun x out ->
      let s = sym () in
      seq (loop_n s 2 (set x [| aff [ (2, s) ] 0 |] (tick s))) (flat_read ~width:n x out))
    ~expected:
      (Array.init n ~f:(fun u ->
           (if u % 2 = 0 then 1. +. fl (u / 2) else 101. +. fl u) +. 1. +. fl u))
    (`Refused guard_fallback_code);
  (* Range guard from a unit solve: [k := u - 2i] must fall in [0, 2), the loop over [i] kept. The
     writes cover cells 0..3, densely, and the node has [cells] of them. *)
  let affine_writes x =
    let i = sym () and k = sym () in
    loop_n i 2 (loop_n k 2 (set x [| aff [ (2, i); (1, k) ] 0 |] (tag i k)))
  in
  let affine_expected ~cells ~hole =
    Array.init cells ~f:(fun u ->
        (if u < 4 then 1. +. (10. *. fl (u / 2)) +. fl (u % 2) else hole u) +. 1. +. fl u)
  in
  case ~label:"dvg_affine_beyond_uninit" ~dims:[| 6 |]
    ~build:(fun x out -> seq (affine_writes x) (flat_read ~width:6 x out))
    ~expected:(affine_expected ~cells:6 ~hole:(fun u -> 101. +. fl u))
    (`Refused guard_fallback_code);
  case ~label:"dvg_affine_beyond_zeroed" ~dims:[| 6 |]
    ~build:(fun x out -> seq (zero x) (seq (affine_writes x) (flat_read ~width:6 x out)))
    ~expected:(affine_expected ~cells:6 ~hole:(fun _ -> 0.))
    `Inlined;
  case ~label:"dvg_affine_dense" ~dims:[| n |]
    ~build:(fun x out -> seq (affine_writes x) (flat_read ~width:n x out))
    ~expected:(affine_expected ~cells:n ~hole:(fun _ -> assert false))
    `Inlined;
  (* Range guards from the bindings of a multi-component node: cells 0..1 and [from_]..[from_ + 1].
     With [from_ = 3] on five cells, cell 2 is a hole; with [from_ = 2] on four, the union
     covers. *)
  let block ~from_ ~width x out =
    let s = sym () and s2 = sym () in
    seq
      (loop_n s 2 (set x [| iter s |] (tick s)))
      (seq
         (loop ~from_ ~upto:(from_ + 1) s2 (set x [| iter s2 |] (ramp 20. s2)))
         (flat_read ~width x out))
  in
  let block_expected ~from_ ~cells =
    Array.init cells ~f:(fun u ->
        (if u < 2 then 1. +. fl u else if u >= from_ then 20. +. fl u else 101. +. fl u)
        +. 1. +. fl u)
  in
  case ~label:"dvg_block_hole" ~dims:[| 5 |] ~build:(block ~from_:3 ~width:5)
    ~expected:(block_expected ~from_:3 ~cells:5)
    (`Refused guard_fallback_code);
  case ~label:"dvg_block_cover" ~dims:[| n |] ~build:(block ~from_:2 ~width:n)
    ~expected:(block_expected ~from_:2 ~cells:n)
    `Inlined;
  (* No guard at all: nothing to prove, so nothing refuses. *)
  case ~label:"dvg_guard_free" ~dims:[| n |]
    ~build:(fun x out ->
      let s = sym () in
      seq (loop_n s n (set x [| iter s |] (tick s))) (flat_read ~width:n x out))
    ~expected:(Array.init n ~f:(fun u -> 2. +. (2. *. fl u)))
    `Inlined;
  (* gh-ocannl-1954: a union proves only through the components ahead of the first one that reads
     the node. A write there or later is present, but the materialized cell builds on its entry
     value and the replay on the init, so a guard needs a [Zero_out] or that plain prefix covering
     the node; with no guard no init exists at all, and an opening read refuses on its own.
     [accumulate ~lo ~hi] is [x[s] += 1 + s] over [lo..hi]. *)
  let accumulate ~lo ~hi x =
    let s = sym () in
    loop ~from_:lo ~upto:hi s (set x [| iter s |] (add (get x [| iter s |]) (tick s)))
  in
  let store ~lo ~hi x =
    let s = sym () in
    loop ~from_:lo ~upto:hi s (set x [| iter s |] (ramp 10. s))
  in
  let entry u = 101. +. fl u and bump u = 1. +. fl u in
  let values f = Array.init n ~f:(fun u -> f u +. bump u) in
  let accumulation_code = "158:accumulation-without-init" in
  case ~label:"dva_block" ~dims:[| n |]
    ~build:(fun x out ->
      seq (accumulate ~lo:0 ~hi:1 x) (seq (accumulate ~lo:2 ~hi:3 x) (flat_read ~width:n x out)))
    ~expected:(values (fun u -> entry u +. bump u))
    (`Refused guard_fallback_code);
  (* A store under part of the node does not define the cells the accumulation opens on. *)
  case ~label:"dva_partial_store" ~dims:[| n |]
    ~build:(fun x out ->
      seq (store ~lo:0 ~hi:1 x) (seq (accumulate ~lo:2 ~hi:3 x) (flat_read ~width:n x out)))
    ~expected:(values (fun u -> if u < 2 then 10. +. fl u else entry u +. bump u))
    (`Refused guard_fallback_code);
  case ~label:"dva_one_component" ~dims:[| n |]
    ~build:(fun x out -> seq (accumulate ~lo:0 ~hi:3 x) (flat_read ~width:n x out))
    ~expected:(values (fun u -> entry u +. bump u))
    (`Refused accumulation_code);
  (* A plain store covering the node ahead of the accumulation defines every cell before any read:
     the update of an inherited virtual, and a non-zero neutral's fill, have this shape. *)
  case ~label:"dva_covering_store" ~dims:[| n |]
    ~build:(fun x out ->
      seq (store ~lo:0 ~hi:3 x) (seq (accumulate ~lo:0 ~hi:3 x) (flat_read ~width:n x out)))
    ~expected:(values (fun u -> 10. +. fl u +. bump u))
    `Inlined;
  (* Sound, and refused: each component stores before it reads, but the prefix stops at the first
     component that reads the node, so only a guard-free replay gets past it. *)
  case ~label:"dva_mixed_components" ~dims:[| n |]
    ~build:(fun x out ->
      let mixed ~lo ~hi =
        let s = sym () in
        loop ~from_:lo ~upto:hi s
          (seq
             (set x [| iter s |] (ramp 10. s))
             (set x [| iter s |] (add (get x [| iter s |]) (tick s))))
      in
      seq (mixed ~lo:0 ~hi:1) (seq (mixed ~lo:2 ~hi:3) (flat_read ~width:n x out)))
    ~expected:(values (fun u -> 10. +. fl u +. bump u))
    (`Refused_though_sound guard_fallback_code);
  case ~label:"dva_zeroed" ~dims:[| n |]
    ~build:(fun x out ->
      seq (zero x)
        (seq (accumulate ~lo:0 ~hi:1 x) (seq (accumulate ~lo:2 ~hi:3 x) (flat_read ~width:n x out))))
    ~expected:(values bump) `Inlined;
  case ~label:"dva_zeroed_one_component" ~dims:[| n |]
    ~build:(fun x out -> seq (zero x) (seq (accumulate ~lo:0 ~hi:3 x) (flat_read ~width:n x out)))
    ~expected:(values bump) `Inlined;
  (* A store opening the component: its accumulation reads what the store just wrote. *)
  case ~label:"dva_store_opens" ~dims:[| n |]
    ~build:(fun x out ->
      let s = sym () in
      seq
        (loop_n s n
           (seq
              (set x [| iter s |] (ramp 10. s))
              (set x [| iter s |] (add (get x [| iter s |]) (tick s)))))
        (flat_read ~width:n x out))
    ~expected:(values (fun u -> 10. +. fl u +. bump u))
    `Inlined;
  (* An earlier unguarded store of the read site: at a fixed cell no guard is built, and the second
     component's read is of the first one's store. *)
  case ~label:"dva_fixed_store" ~dims:[| n |]
    ~build:(fun x out ->
      let u = sym () in
      seq
        (set x [| fixed 2 |] (c 9.))
        (seq
           (set x [| fixed 2 |] (add (get x [| fixed 2 |]) (c 5.)))
           (loop_n u n (set out [| iter u |] (add (get x [| fixed 2 |]) (tick u))))))
    ~expected:(values (fun _ -> 14.))
    `Inlined

(* A merge-buffer read is no read of the node's previous cell: the replay keeps it, so an opening
   store from the node's own merge buffer accumulates nothing and is not refused as one (local Codex
   review of gh-ocannl-1954). Structural: the read needs a transfer to execute, and what this pins
   is the classification. *)
let declared_virtual_merge_opening () =
  let x = mk "dvm_x" and out = mk "dvm_out" in
  virtualize x;
  materialize out;
  let s = sym () and t = sym () in
  let llc =
    seq
      (loop_n s n (set x [| iter s |] (Ir.Low_level.Get_merge_buffer (x, [| iter s |]))))
      (loop_n t n (set out [| iter t |] (add (get x [| iter t |]) (tick t))))
  in
  let o = optimize ~name:"dvm" llc in
  p "dvm_opening: a store from the node's merge buffer inlines"
    (known_virtual o x && count_get o x = 0)

(* The inherited case: routine A computes [x] and does not read it, so it defers [x] -- the lineage
   commits it virtual and stores its template -- and routine B reads it over the whole grid. B
   skipped the coverage rule for [x] as a declared node does, so the diagonal's equality guard is
   held to the same proof. The reference runs A and B in a fresh lineage with [x] materialized, its
   entries seeded [101 + cell] before A. *)
let inherited_guard_fallback () =
  let fl = Float.of_int in
  let case ~label ~zeroed =
    let program suffix =
      let x = mk ~dims:[| 3; 3 |] (label ^ suffix ^ "_x")
      and out = mk ~dims:[| 3; 3 |] (label ^ suffix ^ "_out") in
      materialize out;
      let s = sym () and j = sym () and k = sym () in
      let diagonal = loop_n s 3 (set x [| iter s; iter s |] (tick s)) in
      let llc_a = if zeroed then seq (zero x) diagonal else diagonal in
      let llc_b =
        loop_n j 3
          (loop_n k 3 (set out [| iter j; iter k |] (add (get x [| iter j; iter k |]) (tag j k))))
      in
      (x, out, llc_a, llc_b)
    in
    let expected =
      Array.init 9 ~f:(fun cell ->
          let j = cell / 3 and k = cell % 3 in
          (if j = k then 1. +. fl j else if zeroed then 0. else 101. +. fl cell)
          +. 1.
          +. (10. *. fl j)
          +. fl k)
    in
    let blank_out = blank 9 in
    let reference =
      let xm, outm, llc_a, llc_b = program "_mat" in
      materialize xm;
      let ctx = Ir.Low_level.empty_optimize_ctx () in
      let o_a = optimize_in ctx ~name:(label ^ "_ma") llc_a in
      let o_b = optimize_in ctx ~name:(label ^ "_mb") llc_b in
      let after_a =
        run ~name:(label ^ "_ma") o_a ~seed:[ (xm, Array.init 9 ~f:(fun i -> 101. +. fl i)) ]
      in
      execute ~ctx:after_a ~name:(label ^ "_mb") o_b ~seed:[ (outm, blank_out) ] ~read:[ outm ]
    in
    p (label ^ ": the materialized twin computes the stated values") (same reference [ expected ]);
    let x, out, llc_a, llc_b = program "" in
    let ctx = Ir.Low_level.empty_optimize_ctx () in
    let o_a = optimize_in ctx ~name:(label ^ "_a") llc_a in
    p (label ^ ": routine A defers the node") (known_virtual o_a x);
    let run_b o_b =
      let after_a = run ~name:(label ^ "_a") o_a ~seed:[] in
      execute ~ctx:after_a ~name:(label ^ "_b") o_b ~seed:[ (out, blank_out) ] ~read:[ out ]
    in
    match optimize_in ctx ~name:(label ^ "_b") llc_b with
    | o_b when zeroed ->
        p (label ^ ": routine B inlines the template") (count_get o_b x = 0);
        p (label ^ ": executed values are the materialized twin's") (same (run_b o_b) reference)
    | o_b ->
        p
          (label ^ ": routine B is refused as " ^ guard_fallback_code
         ^ ", or executes equal to the materialized twin")
          (same (run_b o_b) reference)
    | exception Utils.User_error msg ->
        p
          (label ^ ": routine B is refused as " ^ guard_fallback_code
         ^ ", or executes equal to the materialized twin")
          ((not zeroed) && String.is_substring msg ~substring:guard_fallback_code)
  in
  case ~label:"inh_diagonal_uninit" ~zeroed:false;
  case ~label:"inh_diagonal_zeroed" ~zeroed:true

(* === Accepted === *)

(* Two components covering disjoint affine ranges of one node: each binds at the read site (the
   second by unit-coefficient solving), and the guards select between them over a shared init. *)
let row_block_components () =
  let x = mk "blk_x" and out = mk "blk_out" in
  materialize out;
  let s = sym () and s2 = sym () and t = sym () in
  let llc =
    seq
      (loop_n s 2 (set x [| iter s |] (tick s)))
      (seq
         (loop_n s2 2 (set x [| aff [ (1, s2) ] 2 |] (add (c 20.) (embed s2))))
         (loop_n t n (set out [| iter t |] (get x [| iter t |]))))
  in
  row ~label:"block_components" ~llc ~cand:x ~out ~seed:[] ~expected:[| 1.; 2.; 20.; 21. |]
    ~verdict:Accepted

(* The same selection when a component's loop does not start at 0 (gh-ocannl-1795): the second
   component writes cells 1..2 only, so its range guard is [1 <= t < 3]. A guard over the loop's
   WIDTH from 0 ([0 <= t < 2]) let it overwrite cell 0 with [20] and left cell 2 at [3]. *)
let row_block_components_offset_loop () =
  let x = mk "blko_x" and out = mk "blko_out" in
  materialize out;
  let s = sym () and s2 = sym () and t = sym () in
  let llc =
    seq
      (loop_n s n (set x [| iter s |] (tick s)))
      (seq
         (loop ~from_:1 ~upto:2 s2 (set x [| iter s2 |] (ramp 20. s2)))
         (loop_n t n (set out [| iter t |] (get x [| iter t |]))))
  in
  row ~label:"block_components_offset_loop" ~llc ~cand:x ~out ~seed:[]
    ~expected:[| 1.; 21.; 22.; 4. |] ~verdict:Accepted

(* The unit-coefficient solve's range guard, over the same offset loop: [x[s2 + 1]] for [s2] in 1..2
   writes cells 2..3, so [s2 = t - 1] must fall in [1, 3). *)
let row_block_components_offset_affine () =
  let x = mk "blka_x" and out = mk "blka_out" in
  materialize out;
  let s = sym () and s2 = sym () and t = sym () in
  let llc =
    seq
      (loop_n s n (set x [| iter s |] (tick s)))
      (seq
         (loop ~from_:1 ~upto:2 s2 (set x [| aff [ (1, s2) ] 1 |] (ramp 20. s2)))
         (loop_n t n (set out [| iter t |] (get x [| iter t |]))))
  in
  row ~label:"block_components_offset_affine" ~llc ~cand:x ~out ~seed:[]
    ~expected:[| 1.; 2.; 21.; 22. |] ~verdict:Accepted

(* A structural affine match binds producer symbols pairwise ([s + 1] read at [t + 1] binds [s :=
   t]), which in a multi-component replay needs the same range guard a pass-1 binding gets: the
   partial component writes cells 1..2 only, so at [t = 2] (cell 3) it must not replay. Without the
   guard it overwrote cell 3 with [22]. Two shapes: a covering loop under the partial one, and
   cell-by-cell ends whose union with it covers the node (Codex review of PR #1630). *)
let row_block_components_structural ~cells =
  let label = "block_components_structural" ^ if cells then "_cells" else "_loop" in
  let x = mk (label ^ "_x") and out = mk ~dims:[| n - 1 |] (label ^ "_out") in
  materialize out;
  let s = sym () and s2 = sym () and t = sym () in
  let base =
    if cells then [ set x [| fixed 0 |] (c 10.); set x [| fixed 3 |] (c 30.) ]
    else [ loop_n s n (set x [| iter s |] (tick s)) ]
  in
  let llc =
    List.reduce_exn ~f:seq
      (base
      @ [
          loop_n s2 2 (set x [| aff [ (1, s2) ] 1 |] (ramp 20. s2));
          loop_n t (n - 1) (set out [| iter t |] (get x [| aff [ (1, t) ] 1 |]));
        ])
  in
  row ~label ~llc ~cand:x ~out ~seed:[]
    ~expected:(if cells then [| 20.; 21.; 30. |] else [| 20.; 21.; 4. |])
    ~verdict:Accepted

(* With several terms the read's symbols need not decompose a cell the way the producer's loops do:
   [x[2u + v]] for [v] in 0..3 reads cells the producer writes as [x[2s + t]] for [t] in 0..1, so a
   pairwise binding [t := v] with its range guard would drop cells the component writes. In a
   multi-component replay the match is left to unit-coefficient solving, which keeps [s]'s loop and
   guards the decomposition that holds. [tag s t] separates the two decompositions (Codex
   review). *)
let row_block_components_two_terms () =
  let x = mk ~dims:[| 6 |] "blk2_x" and out = mk ~dims:[| 2; n |] "blk2_out" in
  materialize out;
  let s = sym () and t = sym () and u = sym () and v = sym () in
  let llc =
    List.reduce_exn ~f:seq
      [
        set x [| fixed 0 |] (c 10.);
        loop_n s 3 (loop_n t 2 (set x [| aff [ (2, s); (1, t) ] 0 |] (tag s t)));
        loop_n u 2
          (loop_n v n (set out [| iter u; iter v |] (get x [| aff [ (2, u); (1, v) ] 0 |])));
      ]
  in
  row ~label:"block_components_two_terms" ~llc ~cand:x ~out ~seed:[]
    ~expected:[| 1.; 2.; 11.; 12.; 11.; 12.; 21.; 22. |]
    ~verdict:Accepted

(* The same two components read INSIDE the producer's bounds (gh-ocannl-1896): [x[2u + v]] for [u]
   in 0..2, [v] in 0..1 names the very iteration that writes each cell, so the pairwise binding is
   exact and its range guards always hold. The match binds pairwise, as with one component, rather
   than replaying the [s] loop under a solved guard: the replay keeps no producer loop, only the
   consumer's own two. *)
let row_block_components_two_terms_contained () =
  let label = "block_components_two_terms_contained" in
  let x = mk ~dims:[| 6 |] "blk2c_x" and out = mk ~dims:[| 3; 2 |] "blk2c_out" in
  materialize out;
  let s = sym () and t = sym () and u = sym () and v = sym () in
  let llc =
    List.reduce_exn ~f:seq
      [
        set x [| fixed 0 |] (c 10.);
        loop_n s 3 (loop_n t 2 (set x [| aff [ (2, s); (1, t) ] 0 |] (tag s t)));
        loop_n u 3
          (loop_n v 2 (set out [| iter u; iter v |] (get x [| aff [ (2, u); (1, v) ] 0 |])));
      ]
  in
  let o = optimize ~name:(label ^ "_shape") llc in
  p
    (label ^ ": the replay keeps no producer loop")
    (List.length (loop_sites ~in_scopes:true o.Ir.Low_level.llc) = 2);
  row ~label ~llc ~cand:x ~out ~seed:[] ~expected:[| 1.; 2.; 11.; 12.; 21.; 22. |] ~verdict:Accepted

(* gh-ocannl-1690: one cell, two spellings. On an axis of extent 1 a producer position [Fixed_idx 0]
   and a read through a one-iteration loop address the same cell, so the read site is served with no
   guard and no refusal. The consumer reads the candidate both ways -- [x[u]] under [for u = 0 to 0]
   for cells [0..n-1], [x[0]] for cell [n] -- and adds [tick t], so every output cell names both the
   producer's value and the consumer iteration that read it. [~committed] rows are constant
   producers, committed virtual as [40:scalar-constexpr]: before gh-ocannl-1690 the loop read's
   [13:call-site-index-mismatch] tried to retract that commitment and raised in
   [Tnode.update_memory_mode] instead of compiling. Since gh-ocannl-1735 the commitment lands only
   after every read site was asked, so a refusal declines instead ([row_constexpr_*] below). *)
let unit_axis_row ~label ~x ~producer ~seed ~value ~committed =
  let out = mk ~dims:[| n + 1 |] (label ^ "_out") in
  materialize out;
  let t = sym () and u = sym () in
  let llc =
    seq producer
      (seq
         (loop_n t n (loop ~upto:0 u (set out [| iter t |] (add (get x [| iter u |]) (tick t)))))
         (set out [| fixed n |] (mul (c 2.) (get x [| fixed 0 |]))))
  in
  if committed then
    p
      (label ^ ": the candidate is committed virtual as a scalar constexpr")
      (Option.equal Ir.Tnode.equal_provenance
         (rejection_code (optimize ~name:(label ^ "_probe") llc) x)
         (Some (Ir.Tnode.Site "40:scalar-constexpr")));
  row ~label ~llc ~cand:x ~out ~seed
    ~expected:
      (Array.init (n + 1) ~f:(fun i -> if i < n then value +. 1. +. Float.of_int i else 2. *. value))
    ~verdict:Accepted

(* The issue's shape: a constant written at [Fixed_idx 0]. *)
let row_unit_axis_fixed_producer () =
  let x = mk ~dims:[| 1 |] "ua_fixed_x" in
  unit_axis_row ~label:"unit_axis_fixed_producer" ~x
    ~producer:(set x [| fixed 0 |] (c 7.))
    ~seed:[] ~value:7. ~committed:true

(* The same producer position with a value read from an input: nothing commits it ahead of the
   reads, so before the fix the loop read declined virtualization (13) rather than raising. The
   value is a bare read so the node stays simple: a computed one ([2 + inp[0]]) is complex, and its
   five reads trip the visit cap before any read site is asked. *)
let row_unit_axis_computed_producer () =
  let x = mk ~dims:[| 1 |] "ua_comp_x" and inp = mk ~dims:[| 1 |] "ua_comp_in" in
  materialize inp;
  unit_axis_row ~label:"unit_axis_computed_producer" ~x
    ~producer:(set x [| fixed 0 |] (get inp [| fixed 0 |]))
    ~seed:[ (inp, [| 7. |]) ]
    ~value:7. ~committed:false

(* The mirror spelling: the producer writes through the one-iteration loop, and the [Fixed_idx 0]
   read binds its symbol. *)
let row_unit_axis_loop_producer () =
  let x = mk ~dims:[| 1 |] "ua_loop_x" in
  let i = sym () in
  unit_axis_row ~label:"unit_axis_loop_producer" ~x
    ~producer:(loop ~upto:0 i (set x [| iter i |] (c 7.)))
    ~seed:[] ~value:7. ~committed:true

(* The boundary of the unit-cell rule: what counts is the extent of the COORDINATE a component
   indexes, not its axis's dim. After a [Sub_axis] run the component is a flattened index over the
   run's whole extent product ([Affine.axis_extents]), so on dims [4; 1] the read [x[Sub_axis; t]]
   addresses four cells and the [Fixed_idx 0] component's write is the first of them only. Not a
   unit cell, then, but a fixed component on a flattened coordinate: it is guarded ([0 == t] over
   the flat index) exactly as in [row_fixed_component]. *)
let row_unit_axis_flattened () =
  let x = mk ~dims:[| n; 1 |] "ua_flat_x" and out = mk "ua_flat_out" in
  materialize out;
  let s = sym () and t = sym () in
  let llc =
    seq
      (loop_n s n (set x [| Idx.Sub_axis; iter s |] (tick s)))
      (seq
         (set x [| Idx.Sub_axis; fixed 0 |] (c 9.))
         (loop_n t n (set out [| iter t |] (get x [| Idx.Sub_axis; iter t |]))))
  in
  row ~label:"unit_axis_flattened" ~llc ~cand:x ~out ~seed:[] ~expected:[| 9.; 2.; 3.; 4. |]
    ~verdict:Accepted

(* === Mismatched coordinate layouts (gh-ocannl-1795, gh-ocannl-1898) ===

   The mirror of [row_unit_axis_flattened]: there the producer and the read both flatten the node
   through [Sub_axis], so position by position they name the same coordinate. Here only one side
   does. Solved raw, the inliner bound the producer's first symbol to the run placeholder and
   carried it on: each of the first three rows inlined and miscomputed that way (the values in each
   comment), and gh-ocannl-1795 refused them as 13. Now the pair is first re-expressed in its common
   frame ([Affine.common_layout]: the coarser grouping, each coordinate a [Sub_axis] run and its
   composed row-major index), so a flattened read of a per-axis producer [x[i; k]] meets the
   producer position [K·i + k], and a per-axis read of a flattened producer binds the flat symbol to
   [K·h + e]. Every row inlines and agrees with the materialized run. The rows' producers vary with
   every symbol of their iteration, so a coordinate composed with the wrong stride reads a different
   value. *)

(* Was, into a guard: the diagonal's repeated symbol became [Sub_axis == t] ([Embed_index Sub_axis]
   renders as nothing), which printed [1; 0; 0; 0]. Normalized, the producer position is [3s]
   against [t]: no unit coefficient to solve for, so it is an equality guard [3s == t] over the kept
   [s] loop, falling back on the [Zero_out] off the diagonal. *)
let row_flattened_read_diagonal () =
  let x = mk ~dims:[| 2; 2 |] "frd_x" and out = mk "frd_out" in
  materialize out;
  let s = sym () and t = sym () in
  let llc =
    seq (zero x)
      (seq
         (loop_n s 2 (set x [| iter s; iter s |] (tick s)))
         (loop_n t n (set out [| iter t |] (get x [| Idx.Sub_axis; iter t |]))))
  in
  row ~label:"flattened_read_diagonal" ~llc ~cand:x ~out ~seed:[] ~expected:[| 1.; 0.; 0.; 2. |]
    ~verdict:Accepted

(* Was, into a fixed guard and the range guards: [2 == t] compared the producer's column with the
   flat index, and the covering component's guards bounded the flat index by the row count, which
   printed [1; 2; 90; 4; 0; 0; 0; 0]. Normalized, the covering component is [4i + k] (unit-solved
   for [k] under a range guard, the [i] loop kept) and the column component [4s + 2] (an equality
   guard over the kept [s] loop), replayed in order over one local. *)
let row_flattened_read_fixed_component () =
  let x = mk ~dims:[| 2; 4 |] "frf_x" and out = mk ~dims:[| 8 |] "frf_out" in
  materialize out;
  let i = sym () and k = sym () and s = sym () and t = sym () in
  let llc =
    seq
      (loop_n i 2 (loop_n k 4 (set x [| iter i; iter k |] (tag i k))))
      (seq
         (loop_n s 2 (set x [| iter s; fixed 2 |] (ramp 90. s)))
         (loop_n t 8 (set out [| iter t |] (get x [| Idx.Sub_axis; iter t |]))))
  in
  row ~label:"flattened_read_fixed_component" ~llc ~cand:x ~out ~seed:[]
    ~expected:[| 1.; 2.; 90.; 4.; 11.; 12.; 91.; 14. |]
    ~verdict:Accepted

(* Was, into the replay's other reads, with no guard at all: a transposing producer [x[i; k] := a[k;
   i]] replayed as [a[t; Sub_axis]], which printed [1; 3; 5; 0; 0; 0]. Normalized, [k := t - 3i]
   under a range guard, the [i] loop kept: the operand read is [a[t - 3i; i]]. *)
let row_flattened_read_transpose () =
  let a = mk ~dims:[| 3; 2 |] "frt_a" and x = mk ~dims:[| 2; 3 |] "frt_x" in
  let out = mk ~dims:[| 6 |] "frt_out" in
  materialize a;
  materialize out;
  let i = sym () and k = sym () and t = sym () in
  let llc =
    seq
      (loop_n i 2 (loop_n k 3 (set x [| iter i; iter k |] (get a [| iter k; iter i |]))))
      (loop_n t 6 (set out [| iter t |] (get x [| Idx.Sub_axis; iter t |])))
  in
  row ~label:"flattened_read_transpose" ~llc ~cand:x ~out
    ~seed:[ (a, [| 1.; 2.; 3.; 4.; 5.; 6. |]) ]
    ~expected:[| 1.; 3.; 5.; 2.; 4.; 6. |] ~verdict:Accepted

(* The other direction: a flattened producer [x[Sub_axis; f]] read per axis. The read is normalized
   to [x[Sub_axis; 3i + k]] and pass 1 binds [f := 3i + k] -- no guard, no loop. The transposed read
   [out[k; i] := x[i; k]] checks the stride is the producer's, not the reader's. *)
let row_flattened_producer_axis_read ~transposed =
  let label =
    if transposed then "flattened_producer_transposed_read" else "flattened_producer_axis_read"
  in
  let x = mk ~dims:[| 2; 3 |] (label ^ "_x") in
  let out = mk ~dims:(if transposed then [| 3; 2 |] else [| 2; 3 |]) (label ^ "_out") in
  materialize out;
  let f = sym () and i = sym () and k = sym () in
  let llc =
    seq
      (loop_n f 6 (set x [| Idx.Sub_axis; iter f |] (tick f)))
      (loop_n i 2
         (loop_n k 3
            (set out
               (if transposed then [| iter k; iter i |] else [| iter i; iter k |])
               (get x [| iter i; iter k |]))))
  in
  (* [x]'s flat cell [3i + k] holds [tick (3i + k) = 1 + 3i + k]; [out[k; i]] is flat cell [2k +
     i]. *)
  let expected =
    if transposed then [| 1.; 4.; 2.; 5.; 3.; 6. |] else [| 1.; 2.; 3.; 4.; 5.; 6. |]
  in
  row ~label ~llc ~cand:x ~out ~seed:[] ~expected ~verdict:Accepted

(* A flattened producer of part of a zeroed node, read per axis past its loop. The [Zero_out] admits
   the read, so the coverage rule has nothing to say, and with one component pass 1 binds [f := 2h +
   e] with no guard of its own: unguarded, the replay computes [tick f] at cells 2 and 3, where the
   materialized run reads the zeros. A normalized binding takes the range guard whatever the
   component count. *)
let row_flattened_producer_zero_partial () =
  let label = "flattened_producer_zero_partial" in
  let x = mk ~dims:[| 2; 2 |] (label ^ "_x") and out = mk (label ^ "_out") in
  materialize out;
  let f = sym () and h = sym () and e = sym () in
  let llc =
    seq (zero x)
      (seq
         (loop_n f 2 (set x [| Idx.Sub_axis; iter f |] (tick f)))
         (loop_n h 2
            (loop_n e 2 (set out [| aff [ (2, h); (1, e) ] 0 |] (get x [| iter h; iter e |])))))
  in
  row ~label ~llc ~cand:x ~out ~seed:[] ~expected:[| 1.; 2.; 0.; 0. |] ~verdict:Accepted

(* The same past a single-term structural match: normalized, the producer position [k + 1] meets the
   read's [t + 1], coefficients and offsets equal, so [k := t] binds pairwise, and it too takes the
   range guard. Unguarded, the replay computes [tick t] at cells 3 to 5, where the zeros are. *)
let row_flattened_read_zero_structural () =
  let label = "flattened_read_zero_structural" in
  let x = mk ~dims:[| 2; 3 |] (label ^ "_x") and out = mk ~dims:[| 5 |] (label ^ "_out") in
  materialize out;
  let k = sym () and t = sym () in
  let llc =
    seq (zero x)
      (seq
         (loop_n k 2 (set x [| fixed 0; aff [ (1, k) ] 1 |] (tick k)))
         (loop_n t 5 (set out [| iter t |] (get x [| Idx.Sub_axis; aff [ (1, t) ] 1 |]))))
  in
  row ~label ~llc ~cand:x ~out ~seed:[] ~expected:[| 1.; 2.; 0.; 0.; 0. |] ~verdict:Accepted

(* A partial write read through a flattened index, ordinary node, no [Zero_out]. A normalized pair's
   guards rest on the coverage rule here, as any guard's do, so that rule must see through the
   flattened read: read past the write the node owns a buffer with its entry values; read only where
   it is written, [x[1; k]] folded into [2 + k] is unit-solved under a range guard that always
   holds, and the node inlines -- the proof is the coverage rule's, about the cells read. *)
let flattened_read_partial_write ~label ~read_from ~verdict ~expected ~entry =
  let x = mk ~dims:[| 2; 2 |] (label ^ "_x")
  and out = mk ~dims:[| 4 - read_from |] (label ^ "_out") in
  materialize out;
  let k = sym () and t = sym () in
  let llc =
    seq
      (loop_n k 2 (set x [| fixed 1; iter k |] (tick k)))
      (loop ~from_:read_from ~upto:3 t
         (set out [| aff [ (1, t) ] (-read_from) |] (get x [| Idx.Sub_axis; iter t |])))
  in
  row ~label ~llc ~cand:x ~out ~seed:(entry x) ~expected ~verdict

let row_flattened_read_past_partial () =
  flattened_read_partial_write ~label:"flattened_read_past_partial" ~read_from:0
    ~verdict:(Rejected (Coverage, Ir.Tnode.Read_before_write))
    ~expected:[| 50.; 51.; 1.; 2. |]
    ~entry:(fun x -> [ (x, [| 50.; 51.; 52.; 53. |]) ])

let row_flattened_read_written_cells () =
  flattened_read_partial_write ~label:"flattened_read_written_cells" ~read_from:2 ~verdict:Accepted
    ~expected:[| 1.; 2. |] ~entry:(fun _ -> [])

(* gh-ocannl-1896: single-component pass-1 and single-term structural bindings carry no range guard,
   and the write-index ranges those templates trust are sound only because a read beyond a
   single-component producer's loops never reaches the inliner: the coverage rule gives the node a
   buffer holding its entry values first. The shapes the rule has to stop, each with the contained
   read beside it to show the shape does inline where the rule holds. [~span] is how many cells the
   read covers; the producer writes the first two. *)
let uncovered_read ~label ~dims ~span ~write_idx ~read_idx =
  let x = mk ~dims:[| dims |] (label ^ "_x") and out = mk ~dims:[| span |] (label ^ "_out") in
  materialize out;
  let s = sym () and t = sym () in
  let llc =
    seq
      (loop_n s 2 (set x [| write_idx s |] (tick s)))
      (loop_n t span (set out [| iter t |] (add (get x [| read_idx t |]) (tick t))))
  in
  (x, out, llc)

let uncovered_entries ~dims = Array.init dims ~f:(fun i -> 101. +. Float.of_int i)

let row_uncovered_iterator ~span () =
  let label = Printf.sprintf "uncovered_iterator_%d" span in
  let x, out, llc = uncovered_read ~label ~dims:4 ~span ~write_idx:iter ~read_idx:iter in
  let written t = if t < 2 then Float.of_int (t + 1) else 101. +. Float.of_int t in
  row ~label ~llc ~cand:x ~out
    ~seed:(if span > 2 then [ (x, uncovered_entries ~dims:4) ] else [])
    ~expected:(Array.init span ~f:(fun t -> written t +. Float.of_int (t + 1)))
    ~verdict:(if span > 2 then Rejected (Coverage, Ir.Tnode.Read_before_write) else Accepted)

let row_uncovered_single_term ~span () =
  let label = Printf.sprintf "uncovered_single_term_%d" span in
  let doubled s = aff [ (2, s) ] 0 in
  let x, out, llc = uncovered_read ~label ~dims:6 ~span ~write_idx:doubled ~read_idx:doubled in
  let written t = if t < 2 then Float.of_int (t + 1) else 101. +. Float.of_int (2 * t) in
  row ~label ~llc ~cand:x ~out
    ~seed:(if span > 2 then [ (x, uncovered_entries ~dims:6) ] else [])
    ~expected:(Array.init span ~f:(fun t -> written t +. Float.of_int (t + 1)))
    ~verdict:(if span > 2 then Rejected (Coverage, Ir.Tnode.Read_before_write) else Accepted)

(* The nested template: the wide read of [x] sits in the stored computation of an intermediate [y],
   replayed at [y]'s own read, so the coverage rule has to judge [x] by a read that reaches no read
   site of its own. *)
let row_uncovered_nested_template ~span () =
  let label = Printf.sprintf "uncovered_nested_template_%d" span in
  let x = mk ~dims:[| 4 |] (label ^ "_x")
  and y = mk ~dims:[| 4 |] (label ^ "_y")
  and out = mk ~dims:[| span |] (label ^ "_out") in
  materialize out;
  let s = sym () and t = sym () and u = sym () in
  let llc =
    seq
      (loop_n s 2 (set x [| iter s |] (tick s)))
      (seq
         (loop_n t span (set y [| iter t |] (add (get x [| iter t |]) (tick t))))
         (loop_n u span (set out [| iter u |] (add (get y [| iter u |]) (tick u)))))
  in
  let written t = if t < 2 then Float.of_int (t + 1) else 101. +. Float.of_int t in
  row ~label ~llc ~cand:x ~out
    ~seed:(if span > 2 then [ (x, uncovered_entries ~dims:4) ] else [])
    ~expected:(Array.init span ~f:(fun t -> written t +. (2. *. Float.of_int (t + 1))))
    ~verdict:(if span > 2 then Rejected (Coverage, Ir.Tnode.Read_before_write) else Accepted)

(* A pair normalized through a trailing [Sub_axis] comes back in the node's axis order, which is
   what gives the unit-extent test the coordinate it asks about (gh-ocannl-1964, found by
   [virtual_inliner_oracle]). The write [x[0; Sub_axis]] is cell 0 of dims [1; 2]; read at [x[0; u]]
   its normalized position meets [u] on the coordinate of extent 2, where a [Fixed_idx 0] takes a
   guard. Written back in reverse, that position stood on the axis of extent 1, read as the same
   cell, and the replay wrote cell 1 too. *)
let row_trailing_sub_axis_write () =
  let label = "trailing_sub_axis_write" in
  let x = mk ~dims:[| 1; 2 |] (label ^ "_x") and out = mk ~dims:[| 2 |] (label ^ "_out") in
  materialize out;
  let u = sym () in
  p
    (label ^ ": the common layout keeps the node's axis order")
    (match
       Ir.Affine.common_layout ~dims:[| 1; 2 |] [| fixed 0; Idx.Sub_axis |] [| fixed 0; iter u |]
     with
    | Some (write, read) ->
        Array.equal Idx.equal_axis_index write [| fixed 0; fixed 0 |]
        && Array.equal Idx.equal_axis_index read [| fixed 0; iter u |]
    | None -> false);
  let llc =
    seq (zero x)
      (seq
         (set x [| fixed 0; Idx.Sub_axis |] (c 7.))
         (loop_n u 2 (set out [| iter u |] (add (get x [| fixed 0; iter u |]) (tick u)))))
  in
  row ~label ~llc ~cand:x ~out ~seed:[] ~expected:[| 8.; 2. |] ~verdict:Accepted

(* A partial frame: on dims [2; 3; 2] the producer flattens only the first two axes, [x[Sub_axis; f;
   g]], and the read is per axis. The common frame coarsens just those two -- the read becomes
   [x[Sub_axis; 3a + b; c]] -- and keeps the last axis its own coordinate, so pass 1 binds [f := 3a
   + b] and [g := c]. *)
let row_partial_frame () =
  let label = "partial_frame" in
  let x = mk ~dims:[| 2; 3; 2 |] (label ^ "_x") and out = mk ~dims:[| 2; 3; 2 |] (label ^ "_out") in
  materialize out;
  let f = sym () and g = sym () and a = sym () and b = sym () and c' = sym () in
  let llc =
    seq
      (loop_n f 6 (loop_n g 2 (set x [| Idx.Sub_axis; iter f; iter g |] (tag f g))))
      (loop_n a 2
         (loop_n b 3
            (loop_n c' 2
               (set out [| iter a; iter b; iter c' |] (get x [| iter a; iter b; iter c' |])))))
  in
  (* Cell [(a, b, c)] is written at [f = 3a + b], [g = c]: [tag f g = 1 + 10f + g]. *)
  let expected = Array.init 12 ~f:(fun n -> Float.of_int (1 + (10 * (n / 2)) + (n % 2))) in
  row ~label ~llc ~cand:x ~out ~seed:[] ~expected ~verdict:Accepted

(* A trailing [Sub_axis] is index 0 of its own axis: [x[t; Sub_axis]] reads column 0, normalized to
   [x[t; 0]], which binds the producer's [i := t] and [k := 0]. *)
let row_trailing_sub_axis_read () =
  let label = "trailing_sub_axis_read" in
  let x = mk ~dims:[| 3; 2 |] (label ^ "_x") and out = mk ~dims:[| 3 |] (label ^ "_out") in
  materialize out;
  let i = sym () and k = sym () and t = sym () in
  let llc =
    seq
      (loop_n i 3 (loop_n k 2 (set x [| iter i; iter k |] (tag i k))))
      (loop_n t 3 (set out [| iter t |] (get x [| iter t; Idx.Sub_axis |])))
  in
  row ~label ~llc ~cand:x ~out ~seed:[] ~expected:[| 1.; 11.; 21. |] ~verdict:Accepted

(* === A refused scalar constexpr declines (gh-ocannl-1735) ===

   A constant scalar is exempt from the heuristic caps, but its [40:scalar-constexpr] commitment now
   waits for the walk to ask every store and read site. Before, it was committed in
   [decide_placements], ahead of both, and a refusal at either phase raised "already virtual" in
   [Tnode.update_memory_mode] instead of materializing the node. Each row first pins that the
   candidate IS a scalar constexpr, so it cannot pass by never having been one. *)
let is_scalar_constexpr (o : Ir.Low_level.optimized) tn =
  (Hashtbl.find_exn o.Ir.Low_level.traced_store tn).Ir.Low_level.is_scalar_constexpr

(* Store time: the setter sits under a runtime guard, refused as [142]. Zero-initialized, so both
   flag values have a defined reading (0 or 7), and a replay of the setter without its guard would
   print 7 at flag 0. *)
let row_constexpr_guarded_setter ~flag_value =
  let label = Printf.sprintf "constexpr_guarded_setter_flag%.0f" flag_value in
  let flag = mk ~dims:[| 1 |] (label ^ "_flag")
  and x = mk ~dims:[| 1 |] (label ^ "_x")
  and out = mk (label ^ "_out") in
  materialize flag;
  materialize out;
  let t = sym () in
  let llc =
    seq (zero x)
      (seq
         (if_ (get flag [| fixed 0 |]) (set x [| fixed 0 |] (c 7.)))
         (loop_n t n (set out [| iter t |] (add (get x [| fixed 0 |]) (tick t)))))
  in
  p
    (label ^ ": the candidate is a scalar constexpr")
    (is_scalar_constexpr (optimize ~name:(label ^ "_probe") llc) x);
  row ~label ~llc ~cand:x ~out
    ~seed:[ (flag, [| flag_value |]) ]
    ~expected:(Array.init n ~f:(fun i -> (7. *. flag_value) +. 1. +. Float.of_int i))
    ~verdict:(Rejected (Store, Ir.Tnode.Site "142:guarded-computation"))

(* The one cell of a [1; 1] scalar read through a flattened [Sub_axis] run. Until gh-ocannl-1898 a
   [Fixed_idx] producer position had no value to compare with a run placeholder, and the read site
   was refused as [13] -- this row's consumption-time refusal. Normalized, the producer is
   [x[Sub_axis; 0]] against [x[Sub_axis; u]] on a coordinate of extent 1: the same cell, served with
   no guard. No consumption-time refusal of a scalar constexpr remains to pin: the node has one
   cell, so every in-bounds read is that cell, and the other consumption-time codes belong to vector
   setters and scans, which a scalar constexpr does not have. The store-time leg
   ([row_constexpr_guarded_setter]) still pins that a refusal declines rather than raising. *)
let row_constexpr_flattened_read () =
  let label = "constexpr_flattened_read" in
  let x = mk ~dims:[| 1; 1 |] (label ^ "_x") and out = mk (label ^ "_out") in
  materialize out;
  let t = sym () and u = sym () in
  let llc =
    seq
      (set x [| fixed 0; fixed 0 |] (c 7.))
      (loop_n t n
         (loop ~upto:0 u (set out [| iter t |] (add (get x [| Idx.Sub_axis; iter u |]) (tick t)))))
  in
  p
    (label ^ ": the candidate is a scalar constexpr")
    (is_scalar_constexpr (optimize ~name:(label ^ "_probe") llc) x);
  row ~label ~llc ~cand:x ~out ~seed:[]
    ~expected:(Array.init n ~f:(fun i -> 8. +. Float.of_int i))
    ~verdict:Accepted

(* The exemption the deferral keeps: with simple computations no longer inlined unconditionally, a
   constant read [n] times is past the visit cap -- which a scalar constexpr is exempt from, and
   still is now that its commitment waits for the walk. *)
let row_constexpr_cap_exempt () =
  let label = "constexpr_cap_exempt" in
  let x = mk ~dims:[| 1 |] (label ^ "_x") and out = mk (label ^ "_out") in
  materialize out;
  let t = sym () in
  let llc =
    seq
      (set x [| fixed 0 |] (c 7.))
      (loop_n t n (set out [| iter t |] (add (get x [| fixed 0 |]) (tick t))))
  in
  let settings = Ir.Low_level.virtualize_settings in
  let saved = settings.Ir.Low_level.inline_simple_computations in
  settings.Ir.Low_level.inline_simple_computations <- false;
  Exn.protect
    ~finally:(fun () -> settings.Ir.Low_level.inline_simple_computations <- saved)
    ~f:(fun () ->
      p
        (label ^ ": the candidate is committed virtual as a scalar constexpr")
        (Option.equal Ir.Tnode.equal_provenance
           (rejection_code (optimize ~name:(label ^ "_probe") llc) x)
           (Some (Ir.Tnode.Site "40:scalar-constexpr")));
      row ~label ~llc ~cand:x ~out ~seed:[]
        ~expected:(Array.init n ~f:(fun i -> 8. +. Float.of_int i))
        ~verdict:Accepted)

(* gh-ocannl-2077: a guard over a scope's read -- an acceptance row, both sides of where the guard
   is carried into a scope. [b] is zeroed, then written as [b[i] = c[i] + 1] for [i < 3] only,
   though it spans [extent]; [c[i] = sum_k x[i, k, 0]] is a virtual reduction, so [b]'s template
   holds [c]'s scope, which cannot collapse. Inlining [b] at [out[j] = b[j] + 7 + 300 j] guards the
   replay with [j < 3], and the replay reads [x] at row [j] inside that scope. Codegen emits a
   scope's body ahead of the statement, outside the [?:], so before the fix the body read [x] past
   its end where the guard fails -- megabytes past at a wide [x], a SIGBUS on cc at [-O0], volatile
   reads on Metal, the values right throughout. The replay now carries the guard into the body
   ([Low_level.guard_hoisted]): [b] stays inlined, and the strict interpreter, which probes a
   discarded arm's scopes, holds what the optimizer left to [x]'s bounds. [`Constant] reads row 0
   and adds [10 i], so no read can leave [x] and the body keeps no guard of its own. [`Init] starts
   the reduction from a read of [x] rather than a zero: that initializer goes under an [If], its
   value untouched, where the update takes a [Where]. *)
let row_guarded_scope_read ~read () =
  let stride = 4 and extent = 8 in
  let tag =
    match read with
    | `Row -> "gsr_row"
    | `Constant -> "gsr_const"
    | `Init -> "gsr_init"
    | `Init_chain -> "gsr_chain"
  in
  let e = mk ~dims:[| extent |] (tag ^ "_e") in
  let x = mk ~dims:[| 3; 2; stride |] (tag ^ "_x") and cc = mk ~dims:[| 3 |] (tag ^ "_c") in
  let b = mk ~dims:[| extent |] (tag ^ "_b") and out = mk ~dims:[| extent |] (tag ^ "_out") in
  materialize x;
  materialize out;
  let i = sym () and k = sym () and i2 = sym () and j = sym () in
  let reduction =
    match read with
    | `Row ->
        seq (zero cc)
          (loop_n i 3
             (loop_n k 2
                (set cc
                   [| iter i |]
                   (add (get cc [| iter i |]) (get x [| iter i; iter k; fixed 0 |])))))
    | `Constant ->
        seq (zero cc)
          (loop_n i 3
             (loop_n k 2
                (set cc
                   [| iter i |]
                   (add
                      (get cc [| iter i |])
                      (add (get x [| fixed 0; iter k; fixed 0 |]) (mul (c 10.) (embed i)))))))
    | `Init | `Init_chain ->
        loop_n i 3
          (seq
             (set cc [| iter i |] (get x [| iter i; fixed 0; fixed 0 |]))
             (set cc
                [| iter i |]
                (add (get cc [| iter i |]) (get x [| iter i; fixed 1; fixed 0 |]))))
  in
  let consumers =
    match read with
    | `Init_chain ->
        let j0 = sym () in
        [
          loop_n j0 extent (set e [| iter j0 |] (mul (c 2.) (get b [| iter j0 |])));
          loop_n j extent
            (set out [| iter j |] (add (get e [| iter j |]) (add (c 7.) (mul (c 300.) (embed j)))));
        ]
    | `Row | `Constant | `Init ->
        [
          loop_n j extent
            (set out [| iter j |] (add (get b [| iter j |]) (add (c 7.) (mul (c 300.) (embed j)))));
        ]
  in
  let llc =
    List.reduce_exn ~f:seq
      ([ reduction; zero b; loop_n i2 3 (set b [| iter i2 |] (add (get cc [| iter i2 |]) (c 1.))) ]
      @ consumers)
  in
  let x_cells = Array.init (Tn.num_elems x) ~f:(fun n -> Float.of_int (1 + n)) in
  let at i k = x_cells.(((i * 2) + k) * stride) in
  let c_of i =
    match read with
    | `Row | `Init | `Init_chain -> at i 0 +. at i 1
    | `Constant -> at 0 0 +. at 0 1 +. (20. *. Float.of_int i)
  in
  let expected =
    Array.init extent ~f:(fun j ->
        let b_j = if j < 3 then c_of j +. 1. else 0. in
        (match read with `Init_chain -> 2. *. b_j | `Row | `Constant | `Init -> b_j)
        +. 7.
        +. (300. *. Float.of_int j))
  in
  let cand = match read with `Init_chain -> e | `Row | `Constant | `Init -> b in
  row ~label:tag ~llc ~cand ~out ~seed:[ (x, x_cells) ] ~expected ~verdict:Accepted;
  let o = optimize ~name:(tag ^ "_strict") llc in
  let ifs, wheres, _ = census o.LL.llc in
  (match read with
  | `Row ->
      p (tag ^ ": the guard reaches the reduction's update, two Wheres") (wheres = 2 && ifs = 0)
  | `Constant -> p (tag ^ ": the replay alone is guarded, one Where") (wheres = 1 && ifs = 0)
  | `Init ->
      (* The initializer goes under an [If], its value untouched, so it renders as its source store
         would: a [Where] around it would hide a narrowing read from the init's precision rule. *)
      p (tag ^ ": the update takes a Where, the initializer an If") (wheres = 2 && ifs = 1)
  | `Init_chain ->
      (* The guarded scope is captured in [e]'s template and replayed at [out]: an [If] inside a
         scope body assigns a local alone, so the template keeps it (Codex review round 2). *)
      p
        (tag ^ ": b stays virtual inside e's template, its initializer under an If")
        (known_virtual o b && ifs = 1));
  p
    (tag ^ ": the strict interpreter evaluates what the optimizer left without a fault")
    (match
       interpret ~strict:true (interp_memory [ (x, x_cells); (out, blank extent) ]) o.LL.llc
     with
    | () -> true
    | exception Interp_fault msg ->
        Stdio.eprintf "%s: strict fault: %s (not part of the golden)\n%!" tag msg;
        false)

(* gh-ocannl-483 made this arm live: the algebraic rewrite tier runs on the raw lowered code BEFORE
   [Low_level.optimize], and [Online_softmax.hoist] declares its cached probability cell as a
   [Declare_local] beside the statement that consumes it. What the walk refuses is the CONSTRUCTOR,
   not the rewrite, so the boundary is reachable with a plain hoisted local: declare a scope, write
   it, and read it from the candidate's setter in the same captured body. (The rewrite's other
   [Declare_local], inside [emit_normalizer]'s scan body, never reaches this arm -- an enclosing
   [Scan_loop] is refused as 148 first.) The local is pure scalar plumbing, so the materialized arm
   computes the same values: the row pins that inlining is refused AND that the refusal costs
   nothing. *)
let row_hoisted_local () =
  let p_st = mk ~dims:[| 1 |] "hl_p" and x = mk "hl_x" and out = mk "hl_out" in
  virtualize p_st;
  materialize out;
  let p = LL.get_scope p_st in
  let s = sym () and t = sym () in
  let llc =
    seq (zero x)
      (seq
         (loop_n s n
            (seq (declare_local p)
               (seq (LL.Set_local (p, tick s)) (set x [| iter s |] (LL.Get_local p)))))
         (loop_n t n (set out [| iter t |] (get x [| iter t |]))))
  in
  row ~label:"hoisted_local" ~llc ~cand:x ~out ~seed:[]
    ~expected:(Array.init n ~f:(fun i -> 1. +. Float.of_int i))
    ~verdict:(Rejected (Store, Ir.Tnode.Site "19:declare-local"))

(* 148, the remaining store-time constructor arm that fires on ordinary code. This row exists for
   the PHASE: [scan_loop.ml] pins the provenance and the executed values of this shape already, but
   it asserts the tag directly and so says nothing about which pass minted it. Here the setter sits
   INSIDE the scan, so the arm is the caller's [~in_scan] report rather than the walk's own
   [Scan_loop] arm -- the enclosing route of the same pair [row_guarded_enclosing] shows for 142.
   The candidate's cells are prefix sums, which is what makes the differential arm worth running: a
   recurrence is exactly what replaying one instance at a read site cannot reproduce. *)
let row_scan_recurrence () =
  let a = mk "scan_a" and x = mk "scan_x" and out = mk "scan_out" in
  let st = mk ~dims:[| 1 |] "scan_st" in
  virtualize st;
  materialize a;
  materialize out;
  let s = sym () and t = sym () in
  let cr = carry ~init:(c 0.) st in
  let llc =
    seq
      (scan ~upto:(n - 1) s ~carried:[ cr ]
         (seq (set_next cr (add (prev cr) (get a [| iter s |]))) (set x [| iter s |] (next cr))))
      (loop_n t n (set out [| iter t |] (get x [| iter t |])))
  in
  row ~label:"scan_recurrence" ~llc ~cand:x ~out
    ~seed:[ (a, [| 1.; 2.; 3.; 4. |]) ]
    ~expected:[| 1.; 3.; 6.; 10. |]
    ~verdict:(Rejected (Store, Ir.Tnode.Site "148:scan-recurrence"))

(* The outer repetition loop is above the output-index capture. Its embedded index escapes, so
   storage must refuse before a read site is considered. *)
let virtual_store_refusal () =
  let check ~committed =
    let label = if committed then "committed" else "declared" in
    let x = mk (label ^ "_store_x") in
    let ctx = Ir.Low_level.empty_optimize_ctx () in
    if committed then
      (* Seed a lineage decision without declaring the node itself virtual. *)
      Ir.Tnode.Placements.update ctx.placements x Virtual (Site "40:scalar-constexpr")
    else virtualize x;
    let decided_by = if committed then "40:scalar-constexpr" else "99:test-setup" in
    let k = sym () and j = sym () in
    let llc =
      seq (zero x)
        (loop_n k 2 (loop_n j n (set x [| iter j |] (add (get x [| iter j |]) (tick k)))))
    in
    match optimize_in ctx ~name:(label ^ "_store_refusal") llc with
    | _ -> p (label ^ " store: refuses an escaping reduction") false
    | exception Utils.User_error msg ->
        Stdio.printf "%s\n" msg;
        p
          (label ^ " store: names the virtual decision and store-time reason")
          (String.is_substring msg ~substring:("decision " ^ decided_by)
          && String.is_substring msg ~substring:"could not be stored for inlining"
          && String.is_substring msg ~substring:"rejection 10:escaping-index-symbol:"
          && String.is_substring msg ~substring:"Train.set_materialized"
          && (not (String.is_substring msg ~substring:"Initialize every cell"))
          && (not (String.is_substring msg ~substring:"is declared virtual"))
          && not (String.is_substring msg ~substring:"is already virtual"))
  in
  check ~committed:false;
  check ~committed:true

(* Each row runs through [case], so a row that raises is a failed claim naming it, and the rows
   after it still run (gh-ocannl-1084). [--raise-every-row] puts that to the test: every row raises
   before it compiles anything, and [virtual_rejection_boundary_raise.expected] pins that each one
   is named and reached. A row called bare would run for real there, and move that golden. *)
let raise_every_row = Array.mem Stdlib.Sys.argv "--raise-every-row" ~equal:String.equal

let case label row =
  case label (if raise_every_row then fun () -> failwith "injected by --raise-every-row" else row)

let () =
  case "row_visit_cap" row_visit_cap;
  case "row_guarded_enclosing" row_guarded_enclosing;
  case "row_guarded_interior" row_guarded_interior;
  case "row_repetition_above" row_repetition_above;
  case "row_reduction_symbol_free" row_reduction_symbol_free;
  case "row_symbol_free_observer" row_symbol_free_observer;
  case "row_two_setters_one_subtree" row_two_setters_one_subtree;
  case "row_escaping_read" row_escaping_read;
  case "row_escaping_setter" row_escaping_setter;
  case "row_escaping_embed" row_escaping_embed;
  case "row_noninjective" row_noninjective;
  case "row_fixed_component" row_fixed_component;
  case "row_fixed_component_offset" row_fixed_component_offset;
  case "row_fixed_component_affine_read" row_fixed_component_affine_read;
  case "row_fixed_component_first" row_fixed_component_first;
  case "row_fixed_component_accumulating" row_fixed_component_accumulating;
  case "row_fixed_component_accumulating_inside" row_fixed_component_accumulating_inside;
  case "row_fixed_component_zero_init" row_fixed_component_zero_init;
  case "row_fixed_component_uninitialized" row_fixed_component_uninitialized;
  case "declared_virtual_fixed_component" declared_virtual_fixed_component;
  case "row_fixed_component_fixed_read" row_fixed_component_fixed_read;
  case "row_fixed_component_partial_cover" row_fixed_component_partial_cover;
  case "row_fixed_component_cells" row_fixed_component_cells;
  case "row_fixed_component_literal_table" row_fixed_component_literal_table;
  case "row_fixed_component_overlaps" row_fixed_component_overlaps;
  case "row_fixed_component_rows ~read_row:0" (fun () -> row_fixed_component_rows ~read_row:0);
  case "row_fixed_component_rows ~read_row:1" (fun () -> row_fixed_component_rows ~read_row:1);
  case "row_fixed_component_disjoint_first" row_fixed_component_disjoint_first;
  case "row_fixed_component_read_covered_only" row_fixed_component_read_covered_only;
  case "declared_virtual_union" declared_virtual_union;
  case "declared_virtual_guards" declared_virtual_guards;
  case "declared_virtual_merge_opening" declared_virtual_merge_opening;
  case "inherited_guard_fallback" inherited_guard_fallback;
  case "row_block_components" row_block_components;
  case "row_block_components_offset_loop" row_block_components_offset_loop;
  case "row_block_components_offset_affine" row_block_components_offset_affine;
  case "row_block_components_structural ~cells:false" (fun () ->
      row_block_components_structural ~cells:false);
  case "row_block_components_structural ~cells:true" (fun () ->
      row_block_components_structural ~cells:true);
  case "row_block_components_two_terms" row_block_components_two_terms;
  case "row_block_components_two_terms_contained" row_block_components_two_terms_contained;
  case "row_unit_axis_fixed_producer" row_unit_axis_fixed_producer;
  case "row_unit_axis_computed_producer" row_unit_axis_computed_producer;
  case "row_unit_axis_loop_producer" row_unit_axis_loop_producer;
  case "row_unit_axis_flattened" row_unit_axis_flattened;
  case "row_flattened_read_diagonal" row_flattened_read_diagonal;
  case "row_flattened_read_fixed_component" row_flattened_read_fixed_component;
  case "row_flattened_read_transpose" row_flattened_read_transpose;
  case "row_flattened_producer_axis_read ~transposed:false" (fun () ->
      row_flattened_producer_axis_read ~transposed:false);
  case "row_flattened_producer_axis_read ~transposed:true" (fun () ->
      row_flattened_producer_axis_read ~transposed:true);
  case "row_flattened_producer_zero_partial" row_flattened_producer_zero_partial;
  case "row_flattened_read_zero_structural" row_flattened_read_zero_structural;
  case "row_flattened_read_past_partial" row_flattened_read_past_partial;
  case "row_flattened_read_written_cells" row_flattened_read_written_cells;
  case "row_uncovered_iterator ~span:3" (row_uncovered_iterator ~span:3);
  case "row_uncovered_iterator ~span:2" (row_uncovered_iterator ~span:2);
  case "row_uncovered_single_term ~span:3" (row_uncovered_single_term ~span:3);
  case "row_uncovered_single_term ~span:2" (row_uncovered_single_term ~span:2);
  case "row_uncovered_nested_template ~span:3" (row_uncovered_nested_template ~span:3);
  case "row_uncovered_nested_template ~span:2" (row_uncovered_nested_template ~span:2);
  case "row_trailing_sub_axis_write" row_trailing_sub_axis_write;
  case "row_partial_frame" row_partial_frame;
  case "row_trailing_sub_axis_read" row_trailing_sub_axis_read;
  case "row_guarded_scope_read ~read:`Row" (row_guarded_scope_read ~read:`Row);
  case "row_guarded_scope_read ~read:`Constant" (row_guarded_scope_read ~read:`Constant);
  case "row_guarded_scope_read ~read:`Init" (row_guarded_scope_read ~read:`Init);
  case "row_guarded_scope_read ~read:`Init_chain" (row_guarded_scope_read ~read:`Init_chain);
  case "row_hoisted_local" row_hoisted_local;
  case "row_scan_recurrence" row_scan_recurrence;
  case "row_constexpr_cap_exempt" row_constexpr_cap_exempt;
  case "row_constexpr_flattened_read" row_constexpr_flattened_read;
  case "row_constexpr_guarded_setter ~flag_value:0." (fun () ->
      row_constexpr_guarded_setter ~flag_value:0.);
  case "row_constexpr_guarded_setter ~flag_value:1." (fun () ->
      row_constexpr_guarded_setter ~flag_value:1.);
  case "virtual_store_refusal" virtual_store_refusal;
  p_all "every phase-table entry is exercised by a row" phase_table ~f:(fun (site, _) ->
      Hash_set.mem exercised site);
  Stdio.printf "%!"
