(* gh-ocannl-1174: the containment query reads a dynamic gather through its coordinate view.

   [Affine.access] carries the gather's data-dependent axis ([a_dyn_axis]) instead of a bare
   "dynamic" flag, so [Affine.read_covered_before] no longer declines a gather wholesale: the
   dynamic axis is a universal coordinate over its whole extent, the other axes stay known. A gather
   is then covered exactly when the prior writes fill every row at the cells its known coordinates
   name. That flips a DECISION: a table written in full before the gather is no longer
   read-before-write, so it is no longer an input of the routine whose entry values must be kept —
   it can become routine scratch. The raw analysis sees a gather only in IR handed to [optimize]
   already carrying one, as here ([Ll_test]'s supported input class); in the pipeline, gathers are
   minted after it, and the same coverage reaches [reconcile_traced_store]'s re-judgement of spliced
   reads instead — leg 12 executes that path (gh-ocannl-1245).

   Coverage changes which buffer a cell's value comes from, so each leg is executed, with the table
   writes discriminating ([Ll_test.tag]/[tick] values, clear of the zero-init) and the runtime row
   picked by an index node:

   - legs 1-2 are the newly admitted cases — a fully written table, and a table written only in the
   column the gather reads (the known coordinate is what proves it) — run beside a twin with the
   table forced materialized, which must read the same values; - legs 3-4 are the negative controls,
   where the writes miss part of what the gather can touch — a row (the dynamic axis), then a column
   (a known axis). The table must stay read-before-write, an input, and the gather must read the
   ENTRY values seeded into the cells no write covers: a coverage claim that were wrong here would
   have placed the table as scratch, and the seed would either be refused or never reach the read.

   gh-ocannl-1244 adds the read-modify-write exemption's side of the same rule: a dynamic slot holds
   a [Fixed_idx 0] placeholder, so comparing a read's map with its statement's write map
   ([rmw_exempt]) must not take the placeholder for row 0. Legs 5-10 ask the coverage verdict itself
   ([LL.reads_covered_query]) of single statements with no prior write, where an exempt read reads
   [`Covered_rmw_exempt] and any other uncovered read [`Unknown]: the gather of another row under a
   fixed-row [Set] (the issue's shape), a static read under a scatter, and gathers under a scatter
   of another runtime row or another axis are NOT exempt, while the same-position reads — static
   under a static write, and a gather of the scatter's own row along its own axis — stay exempt. The
   verdict is asserted directly because no placement outcome shows it yet: a gathered table is
   committed non-virtual at the read, and [reconcile_traced_store] promotes every
   exemption-dependent writer to read-before-write over the settled placements anyway. Leg 11
   executes the issue's shape, so the table's classification is pinned by the values the gather
   reads. *)

open Base
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Ops = Ir.Ops
open Verdict.Claims

let node = Ll_test.node_factory ~first_id:11740 ~dims:[| 3; 2 |] ()

(* [tbl[dyn; col]] with the runtime row read out of [idx]. *)
let gather ~tbl ~idx col : LL.scalar_t =
  Ll_test.gather ~tn:tbl
    ~idcs:[| Ll_test.fixed 0; col |]
    ~dyn_axis:0
    ~dyn_value:(Ll_test.get idx [| Ll_test.fixed 0 |], Ops.single)

(* Writes [tag k e = 1 + 10k + e] to every cell [tbl[k; e]] with [k < rows], both columns. *)
let fill_rows tbl ~rows =
  let k = Ll_test.sym () and e = Ll_test.sym () in
  Ll_test.loop_n k rows
    (Ll_test.loop_n e 2 (Ll_test.set tbl [| Ll_test.iter k; Ll_test.iter e |] (Ll_test.tag k e)))

(* Writes [tick k = 1 + k] to column 0 of every row, leaving column 1 unwritten. *)
let fill_column0 tbl =
  let k = Ll_test.sym () in
  Ll_test.loop_n k 3 (Ll_test.set tbl [| Ll_test.iter k; Ll_test.fixed 0 |] (Ll_test.tick k))

(* [out[e] = tbl[idx; e]] for both columns. *)
let gather_row ~tbl ~idx ~out =
  let e = Ll_test.sym () in
  Ll_test.loop_n e 2 (Ll_test.set out [| Ll_test.iter e |] (gather ~tbl ~idx (Ll_test.iter e)))

let read_before_write (o : LL.optimized) tn =
  match Hashtbl.find o.LL.traced_store tn with
  | Some traced -> traced.LL.read_before_write
  | None -> failwith ("untraced node " ^ Tn.debug_name tn)

let is_input (o : LL.optimized) tn =
  let (inputs, _), _ = LL.input_and_output_nodes o in
  Set.mem inputs tn

(* The entry value seeded into flat cell [i] of a table: off every value the writes produce. *)
let entry i = 100. +. Float.of_int i
let entries = Array.init 6 ~f:entry

let covered_leg ~name ~fill ~gather_into ~row ~expected =
  let tbl = node (name ^ "_tbl") in
  let idx = node ~dims:[| 1 |] (name ^ "_idx") in
  let out = node ~dims:[| Array.length expected |] (name ^ "_out") in
  Ll_test.materialize idx;
  Ll_test.materialize out;
  let prog = Ll_test.seq (fill tbl) (gather_into ~tbl ~idx ~out) in
  let o = Ll_test.optimize ~materialized:[ idx; out ] ~name prog in
  pf "%s: the gather is covered, so the table is not read-before-write" name
    (not (read_before_write o tbl));
  pf "%s: the table is not an input of the routine" name (not (is_input o tbl));
  let seed = [ (idx, [| Float.of_int row |]); (out, Ll_test.blank (Array.length expected)) ] in
  let got = Ll_test.execute ~name o ~seed ~read:[ out ] in
  pf "%s: the gather reads the row the writes put there" name (Ll_test.same got [ expected ]);
  let name_mat = name ^ "_mat" in
  let o_mat = Ll_test.optimize ~materialized:[ idx; out; tbl ] ~name:name_mat prog in
  let got_mat = Ll_test.execute ~name:name_mat o_mat ~seed ~read:[ out ] in
  pf "%s: the materialized-table twin reads the same values" name (Ll_test.same got got_mat)

let uncovered_leg ~name ~fill ~row ~expected =
  let tbl = node (name ^ "_tbl") in
  let idx = node ~dims:[| 1 |] (name ^ "_idx") in
  let out = node ~dims:[| 2 |] (name ^ "_out") in
  Ll_test.materialize idx;
  Ll_test.materialize out;
  let prog = Ll_test.seq (fill tbl) (gather_row ~tbl ~idx ~out) in
  let o = Ll_test.optimize ~materialized:[ idx; out ] ~name prog in
  pf "%s: the gather is not covered, so the table stays read-before-write" name
    (read_before_write o tbl);
  pf "%s: the table stays an input of the routine" name (is_input o tbl);
  let got =
    Ll_test.execute ~name o
      ~seed:[ (tbl, entries); (idx, [| Float.of_int row |]); (out, Ll_test.blank 2) ]
      ~read:[ out ]
  in
  pf "%s: the gather reads the entry values no write replaced" name (Ll_test.same got [ expected ])

let () =
  (* Leg 1: every row written, both columns gathered; row 2 holds [21; 22]. *)
  covered_leg ~name:"dac_full" ~fill:(fill_rows ~rows:3) ~gather_into:gather_row ~row:2
    ~expected:[| 21.; 22. |];
  (* Leg 2: only column 0 written, and only column 0 gathered: covered through the known coordinate;
     row 1 holds [tick 1 = 2]. *)
  covered_leg ~name:"dac_col0" ~fill:fill_column0
    ~gather_into:(fun ~tbl ~idx ~out ->
      Ll_test.set out [| Ll_test.fixed 0 |] (gather ~tbl ~idx (Ll_test.fixed 0)))
    ~row:1 ~expected:[| 2. |];
  (* Leg 3: rows 0-1 written, the gather can name row 2 — a miss on the dynamic axis. Row 2 is flat
     cells 4-5, which keep their entry values. *)
  uncovered_leg ~name:"dac_rows" ~fill:(fill_rows ~rows:2) ~row:2 ~expected:[| entry 4; entry 5 |];
  (* Leg 4: only column 0 written, both columns gathered — a miss on a known axis. Row 1 reads the
     write in column 0 and its entry value (flat cell 3) in column 1. *)
  uncovered_leg ~name:"dac_cols" ~fill:fill_column0 ~row:1 ~expected:[| 2.; entry 3 |]

(* gh-ocannl-1244: the exemption legs. [idx] holds two runtime rows, so a gather can name the
   scatter's own row ([idx[0]] for both) or another one ([idx[1]]). *)
let verdict_name = function
  | `Covered -> "covered"
  | `Covered_rmw_exempt _ -> "covered only by the rmw exemption"
  | `Unknown _ -> "uncovered"

let row_of idx k : LL.scalar_arg = (Ll_test.get idx [| Ll_test.fixed k |], Ops.single)

let verdict_leg ~name ?(dims = [| 3; 2 |]) ~expected stmt =
  let tbl = node ~dims (name ^ "_tbl") in
  let idx = node ~dims:[| 2 |] (name ^ "_idx") in
  let e = Ll_test.sym () in
  let prog = Ll_test.loop_n e 2 (stmt ~tbl ~idx (Ll_test.iter e) (Ll_test.tick e)) in
  let got = LL.reads_covered_query [] (LL.affine_accesses prog) tbl in
  pf "%s: the table's reads are %s" name (verdict_name expected)
    (String.equal (verdict_name got) (verdict_name expected))

let exempt = `Covered_rmw_exempt ""
let uncovered = `Unknown ""

let gather_leg ~name ~fill ~row ~expected =
  let tbl = node (name ^ "_tbl") in
  let idx = node ~dims:[| 1 |] (name ^ "_idx") in
  let out = node ~dims:[| 2 |] (name ^ "_out") in
  Ll_test.materialize idx;
  Ll_test.materialize out;
  let e = Ll_test.sym () in
  let prog =
    Ll_test.seq (fill ~tbl ~idx)
      (Ll_test.loop_n e 2
         (Ll_test.set out
            [| Ll_test.iter e |]
            (Ll_test.get tbl [| Ll_test.fixed 0; Ll_test.iter e |])))
  in
  let o = Ll_test.optimize ~materialized:[ idx; out ] ~name prog in
  pf "%s: the gather is not covered, so the table stays read-before-write" name
    (read_before_write o tbl);
  pf "%s: the table stays an input of the routine" name (is_input o tbl);
  let seed = [ (tbl, entries); (idx, [| Float.of_int row |]); (out, Ll_test.blank 2) ] in
  let got = Ll_test.execute ~name o ~seed ~read:[ out ] in
  pf "%s: the gather reads the entry values of the row it names" name
    (Ll_test.same got [ expected ]);
  let name_mat = name ^ "_mat" in
  let o_mat = Ll_test.optimize ~materialized:[ idx; out; tbl ] ~name:name_mat prog in
  let got_mat = Ll_test.execute ~name:name_mat o_mat ~seed ~read:[ out ] in
  pf "%s: the materialized-table twin reads the same values" name (Ll_test.same got got_mat)

let () =
  (* Leg 5, the issue's shape: [tbl[0; e] := tbl[?; e] + tick e] — the gather can name rows 1-2,
     which no write covers; the placeholder row 0 of its map is not the write's row 0. *)
  verdict_leg ~name:"rmw_gather_in_set" ~expected:uncovered (fun ~tbl ~idx e v ->
      Ll_test.set tbl
        [| Ll_test.fixed 0; e |]
        (Ll_test.add
           (Ll_test.gather ~tn:tbl
              ~idcs:[| Ll_test.fixed 0; e |]
              ~dyn_axis:0 ~dyn_value:(row_of idx 0))
           v));
  (* Leg 6, the control: [tbl[0; e] := tbl[0; e] + tick e] reads the write's own cell. *)
  verdict_leg ~name:"rmw_same_cell" ~expected:exempt (fun ~tbl ~idx:_ e v ->
      Ll_test.set tbl
        [| Ll_test.fixed 0; e |]
        (Ll_test.add (Ll_test.get tbl [| Ll_test.fixed 0; e |]) v));
  (* Leg 7: [tbl[?; e] := tbl[0; e] + tick e] — the scattered row need not be row 0. *)
  verdict_leg ~name:"rmw_static_in_scatter" ~expected:uncovered (fun ~tbl ~idx e v ->
      Ll_test.scatter ~tn:tbl
        ~idcs:[| Ll_test.fixed 0; e |]
        ~dyn_axis:0 ~dyn_value:(row_of idx 0)
        (Ll_test.add (Ll_test.get tbl [| Ll_test.fixed 0; e |]) v));
  (* Leg 8, the dynamic control: the scatter's own row, [tbl[?(idx[0]); e] := tbl[?(idx[0]); e] +
     tick e] — the embedding-gradient accumulation. *)
  verdict_leg ~name:"rmw_same_row_gather" ~expected:exempt (fun ~tbl ~idx e v ->
      Ll_test.scatter ~tn:tbl
        ~idcs:[| Ll_test.fixed 0; e |]
        ~dyn_axis:0 ~dyn_value:(row_of idx 0)
        (Ll_test.add
           (Ll_test.gather ~tn:tbl
              ~idcs:[| Ll_test.fixed 0; e |]
              ~dyn_axis:0 ~dyn_value:(row_of idx 0))
           v));
  (* Leg 9: another runtime row, [tbl[?(idx[0]); e] := tbl[?(idx[1]); e] + tick e]. *)
  verdict_leg ~name:"rmw_other_row_gather" ~expected:uncovered (fun ~tbl ~idx e v ->
      Ll_test.scatter ~tn:tbl
        ~idcs:[| Ll_test.fixed 0; e |]
        ~dyn_axis:0 ~dyn_value:(row_of idx 0)
        (Ll_test.add
           (Ll_test.gather ~tn:tbl
              ~idcs:[| Ll_test.fixed 0; e |]
              ~dyn_axis:0 ~dyn_value:(row_of idx 1))
           v));
  (* Leg 10: the same runtime value along another axis of a square table, [tbl[?(idx[0]); 0] :=
     tbl[0; ?(idx[0])] + tick e] — both maps read [0; 0] once the placeholders are planted, yet the
     cells are transposed. *)
  verdict_leg ~name:"rmw_other_axis_gather" ~dims:[| 3; 3 |] ~expected:uncovered
    (fun ~tbl ~idx _e v ->
      Ll_test.scatter ~tn:tbl
        ~idcs:[| Ll_test.fixed 0; Ll_test.fixed 0 |]
        ~dyn_axis:0 ~dyn_value:(row_of idx 0)
        (Ll_test.add
           (Ll_test.gather ~tn:tbl
              ~idcs:[| Ll_test.fixed 0; Ll_test.fixed 0 |]
              ~dyn_axis:1 ~dyn_value:(row_of idx 0))
           v));
  (* Leg 11, the issue's shape executed: row 2 is flat cells 4-5, which only their entry values
     fill, so [tbl[0; e]] becomes [entry (4 + e) + tick e]. *)
  gather_leg ~name:"rmw_gather_exec"
    ~fill:(fun ~tbl ~idx ->
      let e = Ll_test.sym () in
      Ll_test.loop_n e 2
        (Ll_test.set tbl
           [| Ll_test.fixed 0; Ll_test.iter e |]
           (Ll_test.add (gather ~tbl ~idx (Ll_test.iter e)) (Ll_test.tick e))))
    ~row:2
    ~expected:[| entry 4 +. 1.; entry 5 +. 2. |]

(* Leg 12 (gh-ocannl-1245): the decision the coverage change moves in production —
   [reconcile_traced_store]'s re-judgement of SPLICED reads. A packed-uniform producer [u :=
   uniform(ctr)] ([Set_from_vec], four single lanes per [uint4x32] counter block) is deferred by
   routine A. Routine B first writes every block of the counter (threefry bits of a range, so each
   block and each lane differ), then reads [u]. The inherited computation is spliced at B's read in
   the lane-extract form [uniform_lane(ctr[flat / 4], flat mod 4)]: the counter read becomes a
   GATHER that splicing put after the counter's write. The strict re-check applies only to reads
   spliced from an INHERITED computation (a uniform deferred and read within one routine splices
   locally and keeps the raw verdicts, hence the separate routine A), and [reads_covered_query]
   answers it over B's final code; the gather's row is a universal coordinate the full write covers,
   so the counter must stay out of read-before-write — no flip, no input of B. The structural claims
   pin that the case reaches that path. B then executes, with the counter never seeded, against a
   twin that evaluates the same text with [u] materialized in B itself, right after the counter's
   write: recompute-at-read is that reading (gh-ocannl-617). The random bits vary with every block
   and lane, so a read of the wrong block or lane, or of a counter buffer B never filled, cannot
   reproduce the twin's values. *)
let () =
  Verdict.case "dac_lane" @@ fun () ->
  let name = "dac_lane" in
  let blocks = 3 and lanes = Ops.vec_unop_lanes Ops.single in
  let cells = blocks * lanes in
  let u4 = Ll_test.node_factory ~prec:Ops.uint4x32 ~first_id:12450 ~dims:[| blocks |] () in
  let ctr = u4 (name ^ "_ctr") and rng = u4 (name ^ "_rng") in
  let u = node ~dims:[| cells |] (name ^ "_u") in
  let out = node ~dims:[| cells |] (name ^ "_out") in
  Ll_test.materialize ctr;
  Ll_test.materialize out;
  (* [u[lanes * b .. lanes * b + lanes - 1] := uniform(ctr[b])], the lowering's packed form. *)
  let store_u () =
    let b = Ll_test.sym () in
    Ll_test.loop_n b blocks
      (LL.Set_from_vec
         {
           tn = u;
           idcs = [| Ll_test.aff [ (lanes, b) ] 0 |];
           length = lanes;
           vec_unop = Ops.Uint4x32_to_prec_uniform;
           arg = (Ll_test.get ctr [| Ll_test.iter b |], Ops.uint4x32);
           debug = "";
         })
  in
  (* [rng[b] := b; ctr[b] := threefry(rng[b], rng[b])]: the counter in full, before the read. *)
  let write_ctr () =
    let b = Ll_test.sym () and b' = Ll_test.sym () in
    let r = (Ll_test.get rng [| Ll_test.iter b' |], Ops.uint4x32) in
    Ll_test.seq
      (Ll_test.loop_n b blocks (Ll_test.set rng [| Ll_test.iter b |] (Ll_test.embed b)))
      (Ll_test.loop_n b' blocks
         (Ll_test.set ctr [| Ll_test.iter b' |] (LL.Binop (Ops.Threefry4x32_light, r, r))))
  in
  let read_u () =
    let i = Ll_test.sym () in
    Ll_test.loop_n i cells
      (Ll_test.set out [| Ll_test.iter i |] (Ll_test.get u [| Ll_test.iter i |]))
  in
  let ctx = LL.empty_optimize_ctx () in
  let o_a = Ll_test.optimize_in ctx ~name:(name ^ "_a") (store_u ()) in
  pf "%s: routine A defers the uniform" name (Ll_test.known_virtual o_a u);
  let o = Ll_test.optimize_in ctx ~name (Ll_test.seq (write_ctr ()) (read_u ())) in
  let gathers_ctr =
    Ll_test.count_scalar o.LL.llc ~f:(function
      | LL.Get_dynamic { tn; _ } -> Tn.equal tn ctr
      | _ -> false)
  in
  pf "%s: routine B writes the counter" name (Ll_test.count_set o ctr > 0);
  pf "%s: the splice reads the counter through a lane-extract gather" name (gathers_ctr > 0);
  pf "%s: the gather is covered, so the counter is not read-before-write" name
    (not (read_before_write o ctr));
  pf "%s: the splice did not flip the counter to read-before-write" name
    (not (Set.mem o.LL.spliced_rbw ctr));
  pf "%s: the counter is not an input of routine B" name (not (is_input o ctr));
  let got = Ll_test.execute ~name o ~seed:[ (out, Ll_test.blank cells) ] ~read:[ out ] in
  Stdio.eprintf "%s: backend=%s (not part of the golden)\n%!" name
    (Context.backend_name (Lazy.force Ll_test.base_ctx));
  let name_mat = name ^ "_mat" in
  let o_mat =
    Ll_test.optimize ~materialized:[ u ] ~name:name_mat
      (Ll_test.seq (write_ctr ()) (Ll_test.seq (store_u ()) (read_u ())))
  in
  let got_mat =
    Ll_test.execute ~name:name_mat o_mat ~seed:[ (out, Ll_test.blank cells) ] ~read:[ out ]
  in
  pf "%s: the twin stores the uniform" name (Ll_test.count_set o_mat u > 0);
  let values = List.hd_exn got in
  pf "%s: the random stream holds %d pairwise distinct values" name cells
    (Array.length values = cells
    && List.length (List.dedup_and_sort (Array.to_list values) ~compare:Float.compare) = cells);
  p_all (name ^ ": every value lies in [0, 1)") (Array.to_list values) ~f:(fun v ->
      Float.(v >= 0. && v < 1.));
  pf "%s: the twin with the uniform materialized in B reads the same values" name
    (Ll_test.same got got_mat)
