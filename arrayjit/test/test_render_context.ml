(* gh-ocannl-769: a backend syntax callback may render another procedure before the outer renderer
   resumes. The inner compile must not replace the outer traversal's state. *)
open Base
open Verdict.Claims
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Cs = Ir.C_syntax
module B = Ll_builders

let node = B.node_factory ~first_id:76900 ~dims:[| 2 |] ()
let scratch = node "reentrant_scratch"
let output = node "reentrant_output"
let source = node "reentrant_source"

let () =
  Tn.update_memory_mode scratch Tn.Local (Site "769:test-setup");
  B.materialize output;
  B.materialize source

let index = B.sym ()

let outer =
  B.seq (B.zero scratch)
    (B.seq
       (B.loop ~upto:1 index
          (B.seq
             (B.set output
                [| B.fixed 0 |]
                (B.add (B.get output [| B.fixed 0 |]) (B.get source [| B.iter index |])))
             (B.set scratch [| B.iter index |] (LL.Constant 7.))))
       (B.zero scratch))

let optimized llc nodes =
  let traced_store = Hashtbl.create (module Tn) in
  List.iter nodes ~f:(fun tn ->
      let n = LL.get_node traced_store tn in
      n.LL.zero_initialized_by_code <- Tn.equal tn scratch);
  {
    LL.traced_store;
    llc;
    optimize_ctx = LL.empty_optimize_ctx ();
    merge_node = None;
    workgroup_shared = Set.empty (module Tn);
    simdgroup_fragments = Set.empty (module Tn);
    swizzled = Map.empty (module Tn);
    pipelined = Map.empty (module Tn);
    zero_fringe = Set.empty (module Tn);
    flip_candidates = [];
    spliced_rbw = Set.empty (module Tn);
  }

let outer_proc = optimized outer [ scratch; output; source ]
let inner_proc = optimized LL.Noop []
let on_binop = ref (fun () -> ())

module Config = struct
  include Cs.Pure_C_config (struct
    let procs = [| outer; LL.Noop |]
    let full_printf_support = true
  end)

  let volatile_serial_accumulation = true

  let binop_syntax prec op a b =
    !on_binop ();
    Cs.default_binop_syntax prec op a b
end

module Syntax = Cs.C_syntax (Config)

let render name proc =
  let _, doc, _ = Syntax.compile_proc ~name [] proc in
  Syntax.doc_to_string doc

let zero_stores text =
  List.count (String.split_lines text) ~f:(fun line ->
      String.is_substring line ~substring:"reentrant_scratch["
      && String.is_substring line ~substring:"] = (float)(0.0)")

let () =
  let baseline, baseline_census = Cs.with_volatility_census (fun () -> render "outer" outer_proc) in
  p "baseline retains the genuine second whole-node zero" (zero_stores baseline = 1);
  p "baseline exercises volatile device RMW rendering" (baseline_census.volatile_rmw_reads = 1);
  let nested_calls = ref 0 in
  (on_binop :=
     fun () ->
       (on_binop := fun () -> ());
       Int.incr nested_calls;
       ignore (render "inner" inner_proc : string));
  let nested, census = Cs.with_volatility_census (fun () -> render "outer" outer_proc) in
  p "the nested compile ran while rendering the outer expression" (!nested_calls = 1);
  p "nested rendering retains the outer genuine second whole-node zero" (zero_stores nested = 1);
  p "nested rendering preserves the outer volatility decision" (census.volatile_rmw_reads = 1);
  p_all "nested rendering attributes every outer volatility site to its own kernel" census.entries
    ~f:(fun (name, _) -> String.equal name "outer");
  (* A callback fails after the inner compile has finished: a fresh compile still starts clean. *)
  (on_binop :=
     fun () ->
       (on_binop := fun () -> ());
       ignore (render "inner_failure" inner_proc : string);
       raise Stdlib.Exit);
  let failed =
    try
      ignore (render "interrupted" outer_proc : string);
      false
    with Stdlib.Exit -> true
  in
  p "the exception control interrupts emission" failed;
  let recovered, recovered_census =
    Cs.with_volatility_census (fun () -> render "recovered" outer_proc)
  in
  p "rendering after an interrupted compile starts with fresh traversal state"
    (zero_stores recovered = 1 && recovered_census.volatile_rmw_reads = 1);
  p_all "recovery records only the recovered kernel" recovered_census.entries ~f:(fun (name, _) ->
      String.equal name "recovered")

(* Negative control: reproduce the old inner-compile reset against an explicitly shared outer cell.
   The same emitted-code observation must see the missing re-zero. *)
let () =
  let ctx = Syntax.create_render_ctx ~name:"shared_control" outer_proc in
  (on_binop :=
     fun () ->
       (on_binop := fun () -> ());
       Hash_set.clear ctx.Syntax.zero_out_seen);
  let broken = Syntax.doc_to_string (Syntax.compile_main ctx outer) in
  p "the negative control detects the old shared zero-state reset" (zero_stores broken = 0)

(* No compiler/global fresh-id allocation is needed to overlap collection brackets. The main domain
   holds its bracket open while a child completes its own, deterministically exposing a
   process-global collector without relying on scheduling sleeps. *)
let () =
  let (), outer =
    Cs.with_census (fun () ->
        Cs.mma_census () := [ ("main", Cs.Mma_scalar_fallback) ];
        let inner =
          Stdlib.Domain.spawn (fun () ->
              let (), summary =
                Cs.with_census (fun () -> Cs.mma_census () := [ ("child", Cs.Mma_register_tiled) ])
              in
              summary)
        in
        let child = Stdlib.Domain.join inner in
        p "a child domain summarizes only its own renderings"
          (child.statements = 1 && child.scalar_fallbacks = 0))
  in
  p "a main-domain census excludes a child domain's renderings"
    (outer.statements = 1 && outer.scalar_fallbacks = 1);
  p_all "main-domain attribution survives overlapping collection brackets" outer.renderings
    ~f:(fun (name, _) -> String.equal name "main")

let () =
  let (), _ =
    Cs.with_peel_census (fun () ->
        let outer = Syntax.create_render_ctx ~name:"suspended" outer_proc in
        outer.Syntax.peel_census_enabled := false;
        let inner = Syntax.create_render_ctx ~name:"inner_collecting" inner_proc in
        p "suspending the outer peel census does not suspend a nested compile"
          (!(inner.Syntax.peel_census_enabled) && not !(outer.Syntax.peel_census_enabled)))
  in
  ()
