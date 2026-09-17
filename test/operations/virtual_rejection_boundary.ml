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

   Codes with no minimal shape here, and why — each of these was tried, not assumed:

   - 5 (a symbol no call site can ground) is preempted: a single-symbol affine position is injective
   unless its coefficient is zero, so a non-injective map arrives as 51 (multi-symbol) or 52. - 52
   (a [Concat] LHS position) is unreachable from this side. [trace_node_facts] runs first and raises
   [invalid_arg] on a [Concat] index outright, so the virtualizer's arm never sees one. - 11
   (already decided materialized) fires, but records nothing of its own: the placement it finds is
   the one it keeps, so the provenance a test would read is the earlier decision's. That is the
   [~materialized] arm every row below already runs. - 12 (no setter in the captured subtree) cannot
   fire: every call site is a setter arm, or a candidate drawn from the assignment-index map, which
   is where its setters put it. - 8, 141, 143, 144 guard constructors no pre-virtualization pass
   emits (staged compilation, barriers, cooperative tiles, dynamic scatters). These will never
   become inlineable, so a row would pin nothing that could move. - 19 (a [Declare_local] in the
   captured nest) guarded the same way until gh-ocannl-483 put the algebraic rewrite tier AHEAD of
   [Low_level.optimize]: [Online_softmax.hoist] emits one, so the arm is reachable whenever
   [online_softmax] is on. Building that shape by hand needs the rewrite's whole normalizer pattern,
   so it stays unrowed here rather than unreachable. - 14, 140, 145, 146 belong to the vector-store
   (packed-uniform) consumption path, exercised through the uniform tests rather than by hand. *)

open Base
open Ll_test
open Verdict.Claims

let mk = node_factory ~first_id:2700 ~dims:[| 4 |] ()
let n = 4

type phase = Cap | Store | Consumption

let equal_phase a b =
  match (a, b) with
  | Cap, Cap | Store, Store | Consumption, Consumption -> true
  | (Cap | Store | Consumption), _ -> false

type verdict = Rejected of phase * Ir.Tnode.provenance | Accepted

(* The phase table. Not derivable from any one place in the source: the store- and consumption-time
   tags are [Site] literals at their raise sites, spread over three functions in [low_level.ml] plus
   the cleanup pass, with no exported owner to ask. Those arms are exemplars, and a row naming the
   wrong phase for its tag fails, so they are under test rather than beside the source.

   The Cap arm is the exception, and it is DERIVED rather than restated: [Low_level] owns the cap
   set and answers for it exhaustively ([is_cap_provenance]), so a fourth cap is classified here
   without an edit — and cannot be classified here as anything else. Since gh-ocannl-609 each tag
   also carries its own reason, so what is left to pin is the PHASE that minted it. *)
let phase_of_code tag =
  if Ir.Low_level.is_cap_provenance tag then Some Cap
  else
    match tag with
    | Ir.Tnode.Site
        ( "4:lhs-idcs-differ" | "5:index-not-groundable" | "7:sibling-escaping-write-index"
        | "8:staged-compilation" | "9:sibling-escaping-read-index" | "10:escaping-index-symbol"
        | "11:already-non-virtual" | "12:no-setter" | "19:declare-local" | "51:affine-not-injective"
        | "52:concat-index" | "141:workgroup-barrier" | "142:guarded-computation" | "143:tile-mma"
        | "144:dynamic-write" | "147:enclosing-repetition-loop" ) ->
        Some Store
    | Ir.Tnode.Site
        ( "13:call-site-index-mismatch" | "14:empty-inlined-body" | "140:vector-setter-unsupported"
        | "145:lane-extract-layout" | "146:lane-extract-counter" ) ->
        Some Consumption
    | _ -> None

let phase_name = function
  | Cap -> "by a heuristic cap"
  | Store -> "at store time"
  | Consumption -> "at consumption time"

let claim = function
  | Accepted -> "inlined"
  | Rejected (ph, code) ->
      Printf.sprintf "rejected %s as Non_virtual %s" (phase_name ph)
        (Ir.Tnode.provenance_to_string code)

(* One row: assert where the verdict was decided, then execute both readings of the same program. *)
let row ~label ~llc ~cand ~out ~seed ~expected ~verdict =
  let o = optimize ~name:label llc in
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

(* === Consumption-time rejection === *)

(* Two setters at different index maps as SEPARATE statements store fine, as two components — the
   index-map comparison is per invocation. The verdict waits for a read site: the fixed-index
   component cannot be served at a symbolic one, and there is no partial answer, so the whole node
   materializes. *)
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
    ~expected:[| 9.; 2.; 3.; 4. |]
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
  row_block_components ();
  Stdio.printf "%!"
