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

(* Same defect with no capture point at all: a symbol-free index map means no loop is ever a capture
   site, so the whole reduction is outside the stored statement. The sibling-read arm ([Non_virtual
   9]) covers this for a reduction over an ARRAY; a reduction over a constant reaches here. *)
let row_reduction_symbol_free () =
  let x = mk ~dims:[| 1 |] "rsf_x" and out = mk ~dims:[| 1 |] "rsf_out" in
  materialize out;
  let s = sym () in
  let llc =
    seq (zero x)
      (seq
         (loop_n s n (set x [| fixed 0 |] (add (get x [| fixed 0 |]) (c 1.))))
         (set out [| fixed 0 |] (get x [| fixed 0 |])))
  in
  row ~label:"reduction_symbol_free" ~llc ~cand:x ~out ~seed:[] ~expected:[| 4. |]
    ~verdict:(Rejected (Store, Ir.Tnode.Site "147:enclosing-repetition-loop"))

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
   covering component's value, which this routine's covering loop proves defined at every cell. *)
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
        String.is_substring msg ~substring:"13:call-site-index-mismatch")

(* === Consumption-time rejection === *)

(* A fixed component read at a DIFFERENT fixed cell is a static mismatch, not a guard: the read site
   cannot be served, and there is no partial answer, so the whole node materializes. *)
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
    ~verdict:(Rejected (Consumption, Ir.Tnode.Site "13:call-site-index-mismatch"))

(* The proof's conservative edge: three components that cover the node only TOGETHER -- a loop over
   cells 0..2 and fixed writes at 3 -- prove nothing on their own, so the fixed component refuses
   and the node materializes, the correct (if costlier) reading. Unions of partial covers are not
   proved. *)
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
    ~expected:[| 1.; 2.; 3.; 9. |]
    ~verdict:(Rejected (Consumption, Ir.Tnode.Site "13:call-site-index-mismatch"))

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

(* Consumption time: the one cell of a [1; 1] scalar read through a flattened [Sub_axis] run -- the
   same cell, but a [Fixed_idx] producer position has no value to compare with a run placeholder, so
   the read site is refused as [13]. *)
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
    ~verdict:(Rejected (Consumption, Ir.Tnode.Site "13:call-site-index-mismatch"))

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

let () =
  row_visit_cap ();
  row_guarded_enclosing ();
  row_guarded_interior ();
  row_repetition_above ();
  row_reduction_symbol_free ();
  row_two_setters_one_subtree ();
  row_escaping_read ();
  row_escaping_setter ();
  row_escaping_embed ();
  row_noninjective ();
  row_fixed_component ();
  row_fixed_component_offset ();
  row_fixed_component_affine_read ();
  row_fixed_component_first ();
  row_fixed_component_accumulating ();
  row_fixed_component_zero_init ();
  row_fixed_component_uninitialized ();
  row_fixed_component_fixed_read ();
  row_fixed_component_partial_cover ();
  declared_virtual_fixed_component ();
  row_block_components ();
  row_block_components_offset_loop ();
  row_block_components_offset_affine ();
  row_unit_axis_fixed_producer ();
  row_unit_axis_computed_producer ();
  row_unit_axis_loop_producer ();
  row_unit_axis_flattened ();
  row_hoisted_local ();
  row_scan_recurrence ();
  row_constexpr_cap_exempt ();
  row_constexpr_flattened_read ();
  row_constexpr_guarded_setter ~flag_value:0.;
  row_constexpr_guarded_setter ~flag_value:1.;
  p_all "every phase-table entry is exercised by a row" phase_table ~f:(fun (site, _) ->
      Hash_set.mem exercised site);
  Stdio.printf "%!"
