(* gh-ocannl-1174: the containment query reads a dynamic gather through its coordinate view.

   [Affine.access] carries the gather's data-dependent axis ([a_dyn_axis]) instead of a bare
   "dynamic" flag, so [Affine.read_covered_before] no longer declines a gather wholesale: the
   dynamic axis is a universal coordinate over its whole extent, the other axes stay known. A gather
   is then covered exactly when the prior writes fill every row at the cells its known coordinates
   name. That flips a DECISION: a table written in full before the gather is no longer
   read-before-write, so it is no longer an input of the routine whose entry values must be kept —
   it can become routine scratch.

   Coverage changes which buffer a cell's value comes from, so each leg is executed, with the table
   writes discriminating ([Ll_test.tag]/[tick] values, clear of the zero-init) and the runtime row
   picked by an index node:

   - legs 1-2 are the newly admitted cases — a fully written table, and a table written only in the
   column the gather reads (the known coordinate is what proves it) — run beside a twin with the
   table forced materialized, which must read the same values; - legs 3-4 are the negative controls,
   where the writes miss part of what the gather can touch — a row (the dynamic axis), then a column
   (a known axis). The table must stay read-before-write, an input, and the gather must read the
   ENTRY values seeded into the cells no write covers: a coverage claim that were wrong here would
   have placed the table as scratch, and the seed would either be refused or never reach the
   read. *)

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
