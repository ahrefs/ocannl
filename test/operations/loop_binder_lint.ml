(* gh-ocannl-1402: the loop-binder lint at [Ir.Low_level.optimize]'s entry
   ([Low_level.validate_loop_binders]) -- each loop symbol is bound by at most one loop per routine,
   and every symbol an index mentions is bound by an enclosing loop or is a static index.

   Why a lint: two lowering generators broke sibling-binder uniqueness independently -- concat
   segments (gh-ocannl-765) and padding-margin strips (gh-ocannl-1384) -- and each time the flat
   symbol-keyed consumers misread it SILENTLY (extent tables kept one loop's width, the canonical
   render declined the routine for both digest caches). The lowering generators themselves are held
   to the rule by every test that compiles through [optimize]; [concat_loop_symbols] and
   [padded_conv_cache] are the ones that pin the two historical offenders. This test pins the lint:
   each refusal by name on hand-built IR, and the well-formed shapes it must keep accepting (fresh
   sibling binders, a static index used free, a scan beside a loop). *)

open Base
open Ll_test
open Verdict.Claims
module LL = Ir.Low_level
module Idx = Ir.Indexing

let n = 4
let mk = node_factory ~first_id:14020 ~dims:[| n |] ()

(* The refusal message when [optimize] refuses [llc] through the binder lint, [None] when it
   accepts. Any other exception propagates: a different refusal is not this lint's verdict. *)
let lint_refusal ?static_indices ~name llc =
  match optimize ?static_indices ~name llc with
  | (_ : LL.optimized) -> None
  | exception Invalid_argument msg
    when String.is_prefix msg ~prefix:"Low_level.validate_loop_binders:" ->
      Some msg

(* [llc] is refused by the lint, and the message names [s] and contains [phrase]. *)
let refused ?static_indices ~label ~s ~phrase llc =
  match lint_refusal ?static_indices ~name:label llc with
  | None -> p (label ^ ": optimize refuses it through the binder lint") false
  | Some msg ->
      p (label ^ ": optimize refuses it through the binder lint") true;
      p
        (label ^ ": the refusal names the symbol and the defect")
        (String.is_substring msg ~substring:(Idx.symbol_ident s)
        && String.is_substring msg ~substring:phrase)

let accepted ?static_indices ~label llc =
  p (label ^ ": optimize accepts it")
    (Option.is_none (lint_refusal ?static_indices ~name:label llc))

let static_of s : Idx.static_symbol =
  { static_symbol = s; static_range = Some n; used_as_extent = false; used_as_slice = false }

(* Fresh nodes per leg: the analysis cache would otherwise hand an alpha-equivalent build the first
   build's symbols (#1151's trap), and a refusal leg must not depend on an earlier leg's
   lowering. *)
let out label =
  let o = mk (label ^ "_out") in
  materialize o;
  o

let () =
  (* --- Well-formed shapes the lint must keep accepting. --- *)
  case "fresh sibling binders" (fun () ->
      let o = out "lbl_ok" in
      let i = sym () and j = sym () in
      accepted ~label:"two sibling loops, each with its own symbol"
        (seq (loop_n i n (set_at o (iter i) (tick i))) (loop_n j n (set_at o (iter j) (tick j)))));
  case "static index used free" (fun () ->
      let o = out "lbl_static" in
      let s = sym () and i = sym () in
      accepted
        ~static_indices:[ static_of s ]
        ~label:"a static index read under a loop that does not bind it"
        (loop_n i n (set_at o (iter i) (add (tick i) (embed s)))));
  case "scan beside a loop" (fun () ->
      let o = out "lbl_scan" and st = mk ~dims:[| 1 |] "lbl_scan_state" in
      virtualize st;
      let i = sym () and j = sym () in
      let cr = carry ~init:(c 0.) st in
      accepted ~label:"a scan and a sibling loop with distinct symbols"
        (seq
           (scan ~upto:(n - 1) i ~carried:[ cr ]
              (seq (set_next cr (add (prev cr) (tick i))) (set_at o (iter i) (next cr))))
           (loop_n j n (set_at o (iter j) (add (get o [| iter j |]) (c 1.))))));
  (* --- Refusals: one per defect, by name. --- *)
  case "sibling binders share a symbol" (fun () ->
      (* The gh-ocannl-765 / gh-ocannl-1384 shape: two loops over different ranges, one symbol. *)
      let o = out "lbl_sib" in
      let i = sym () in
      refused ~label:"sibling loops sharing a binder" ~s:i ~phrase:"rebinds a symbol already bound"
        (seq
           (loop ~upto:0 i (set_at o (iter i) (tick i)))
           (loop ~from_:1 ~upto:(n - 1) i (set_at o (iter i) (tick i)))));
  case "nested loops share a symbol" (fun () ->
      let o = mk ~dims:[| n; n |] "lbl_nest_out" in
      materialize o;
      let i = sym () in
      refused ~label:"a loop shadowing its enclosing loop's binder" ~s:i
        ~phrase:"rebinds a symbol already bound"
        (loop_n i n (loop_n i n (set o [| iter i; iter i |] (tick i)))));
  case "scan rebinds a loop's symbol" (fun () ->
      let o = out "lbl_scanrb" and st = mk ~dims:[| 1 |] "lbl_scanrb_state" in
      virtualize st;
      let i = sym () in
      let cr = carry ~init:(c 0.) st in
      refused ~label:"a scan rebinding a sibling loop's symbol" ~s:i
        ~phrase:"rebinds a symbol already bound"
        (seq
           (loop_n i n (set_at o (iter i) (tick i)))
           (scan ~upto:(n - 1) i ~carried:[ cr ]
              (seq (set_next cr (add (prev cr) (tick i))) (set_at o (iter i) (next cr))))));
  case "loop binds a static index" (fun () ->
      let o = out "lbl_stbind" in
      let s = sym () in
      refused
        ~static_indices:[ static_of s ]
        ~label:"a loop binding a static index" ~s ~phrase:"binds a static index"
        (loop_n s n (set_at o (iter s) (tick s))));
  case "free symbol in a write index" (fun () ->
      let o = out "lbl_free_set" in
      let i = sym () and f = sym () in
      refused ~label:"a Set index naming an unbound symbol" ~s:f ~phrase:"no enclosing loop binds"
        (loop_n i n (set_at o (iter f) (tick i))));
  case "free symbol in an affine read" (fun () ->
      let o = out "lbl_free_aff" and x = mk ~dims:[| 2 * n |] "lbl_free_aff_x" in
      materialize x;
      let i = sym () and f = sym () in
      refused ~label:"an Affine read index naming an unbound symbol" ~s:f
        ~phrase:"no enclosing loop binds"
        (loop_n i n (set_at o (iter i) (get x [| aff [ (1, i); (1, f) ] 0 |]))));
  case "free symbol in an Embed_index" (fun () ->
      let o = out "lbl_free_emb" in
      let i = sym () and f = sym () in
      refused ~label:"an Embed_index naming an unbound symbol" ~s:f
        ~phrase:"no enclosing loop binds"
        (loop_n i n (set_at o (iter i) (add (tick i) (embed f)))));
  case "symbol used after its loop" (fun () ->
      (* Bound in the routine, but not in scope where it is read: scope is lexical, not
         routine-wide. *)
      let o = out "lbl_escape" in
      let i = sym () in
      refused ~label:"a symbol read after the loop that bound it" ~s:i
        ~phrase:"no enclosing loop binds"
        (seq (loop_n i n (set_at o (iter i) (tick i))) (set_at o (fixed 0) (embed i))))
