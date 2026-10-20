(* gh-ocannl-687: the action menu enumerates loops inside SCHEDULE-minted [Local_scope] bodies only.

   gh-ocannl-666 taught [Autotune.collect_loops] to descend scope bodies, because since
   gh-ocannl-639 the accumulation mints of [Sched.Unroll { materialize = true }] and [Partition]
   move the per-step / per-segment loops inside the accumulator's scope, where
   [Schedule.rewrite_loop] still reaches them. The IR carried no provenance, so the descent could
   not be aimed: virtualization mints the same construct at every inlined read, and the menu began
   proposing splits, swaps, unrolls AND vectorize retypes for inlined interpolation and reduction
   loops — up to eight descriptors per loop, each costing a candidate compile and, under the
   per-unit cap, displacing a proposal for the main nest.

   The aim is by action CATEGORY, not by loop. An inlined loop keeps its [Vectorized] retype: that
   is one descriptor and the only one with a renderer built for the shape — [try_vectorize_reduce]
   recognizes an inlined reduction's [Set_local] accumulation, while the ENCLOSING loop cannot be
   explicitly vectorized at all, its body holding a [Local_scope] the elementwise vectorizer bails
   on. Dropping the inner loop wholesale destroys that candidate instead of moving it outward, which
   is what the first attempt here did and what PR #424's review round 2 caught.

   [Low_level.scope_mint] is the fact that separates them. The probe is a controlled pair: ONE
   program shape — [out[i] = scope{ acc := out[i]; for k: acc := acc + x[i,k] }] — built twice, the
   two copies differing in nothing but the scope's [mint] field. Both copies' [k] must keep the
   vectorize retype; only the schedule-minted copy's [k] may draw Splits, Swaps and Unrolls. The
   control that makes the negative half discriminating: [k] is registry-nameable in the inlined copy
   too, so what keeps those categories away from it is the provenance decision and not an
   unresolvable binder.

   The last claims are the ones the exclusion has to earn. The retype must stay on the loop that has
   a renderer — [k] in BOTH copies — and must not migrate to [i], which cannot be explicitly
   vectorized whoever minted the scope below it. *)

open Base
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Idx = Ir.Indexing
module SC = Ir.Schedule_cache
open Verdict.Claims

let node = Ll_test.node_factory ~first_id:7100 ~dims:[| 4 |] ()
let ni, nk = (4, 4)

(* Synthetic CPU limits: a vector width, so [Vectorized] retypes are proposed at all. *)
let limits =
  {
    Ir.Backend_intf.no_hardware_limits with
    simd_vector = { Ir.Vector_file.bytes = 32; registers = 16 };
  }

let () =
  let out = node ~dims:[| ni |] "asp_out" in
  let x = node ~dims:[| ni; nk |] "asp_x" in
  Ll_test.materialize out;
  Ll_test.materialize x;
  let i = Ll_test.sym () and k = Ll_test.sym () in
  let acc = LL.get_scope out in
  let scope_body =
    Ll_test.seq
      (LL.Set_local (acc, Ll_test.get out [| Ll_test.iter i |]))
      (Ll_test.loop_n k nk
         (LL.Set_local
            ( acc,
              Ll_test.add (LL.Get_local acc) (Ll_test.get x [| Ll_test.iter i; Ll_test.iter k |]) )))
  in
  (* The one variable of this experiment. *)
  let program mint =
    Ll_test.loop_n i ni
      (Ll_test.set out
         [| Ll_test.iter i |]
         (LL.Local_scope { id = acc; body = scope_body; orig_indices = [| Ll_test.iter i |]; mint }))
  in
  (* The scope-free twin the post-optimize seam needs (gh-ocannl-681): same nodes, same reads and
     writes, no scope — it supplies the traced store and placements. *)
  let raw =
    Ll_test.loop_n i ni
      (Ll_test.loop_n k nk
         (Ll_test.set out
            [| Ll_test.iter i |]
            (Ll_test.add
               (Ll_test.get out [| Ll_test.iter i |])
               (Ll_test.get x [| Ll_test.iter i; Ll_test.iter k |]))))
  in
  let menu_of mint =
    let o =
      Ll_test.optimize_scoped ~materialized:[ out; x ] ~name:"asp_probe" ~raw (program mint)
    in
    let canon = SC.canonicalize ~static_indices:[] ~with_placements:false o in
    let registry = SC.base_registry canon in
    (registry, Autotune.menu ~is_cpu:true ~is_gpu:false ~limits ~registry o)
  in
  let reg_mint, menu_mint = menu_of LL.Schedule_minted in
  let reg_inline, menu_inline = menu_of LL.Inlined_computation in
  let targets rf op =
    match op with
    | SC.Split { axis; _ } | SC.Unroll { axis; _ } | SC.Retype { axis; _ } ->
        SC.equal_sym_ref axis rf
    | SC.Swap { outer; inner } -> SC.equal_sym_ref outer rf || SC.equal_sym_ref inner rf
    | _ -> false
  in
  let count reg menu sym =
    match SC.resolve reg sym with None -> 0 | Some rf -> List.count menu ~f:(targets rf)
  in
  (* The control: the inlined copy's inner binder IS nameable by a persisted schedule, so its
     absence from the menu is the provenance decision and nothing else. *)
  p "control: the inlined scope's inner loop is registry-nameable"
    (Option.is_some (SC.resolve reg_inline k));
  let reshaping menu rf =
    List.count menu ~f:(function
      | SC.Split { axis; _ } | SC.Unroll { axis; _ } -> SC.equal_sym_ref axis rf
      | SC.Swap { outer; inner } -> SC.equal_sym_ref outer rf || SC.equal_sym_ref inner rf
      | _ -> false)
  in
  let reshaping_of reg menu sym =
    match SC.resolve reg sym with None -> 0 | Some rf -> reshaping menu rf
  in
  p "the schedule mint's scope-nested loop draws proposals" (count reg_mint menu_mint k > 0);
  p "the schedule mint's scope-nested loop draws the reshaping categories"
    (reshaping_of reg_mint menu_mint k > 0);
  p "the virtualization inline's scope-nested loop draws none of them"
    (reshaping_of reg_inline menu_inline k = 0);
  (* Only the scope-nested descriptors differ: the statement-level nest is untouched by the
     provenance filter. *)
  p "the statement-level loop draws the same proposals under either mint"
    (count reg_mint menu_mint i = count reg_inline menu_inline i && count reg_mint menu_mint i > 0);
  p "the inlined copy's menu is the schedule mint's minus exactly its scope-nested reshaping part"
    (List.length menu_inline + reshaping_of reg_mint menu_mint k = List.length menu_mint);
  (* The retype stays on the loop a renderer can serve. [C_syntax]'s elementwise vectorizer bails on
     a body holding a [Local_scope], and an accumulating bailout falls back to a plain serial loop,
     so a retype of [i] would render like the baseline whoever minted the scope below it;
     [try_vectorize_reduce] serves [k]'s [Set_local] accumulation in both copies. Hence: same answer
     under either mint, and it is the inner loop. *)
  let vectorizes reg menu sym =
    match SC.resolve reg sym with
    | None -> 0
    | Some rf ->
        List.count menu ~f:(function
          | SC.Retype { axis; ty = LL.Vectorized } -> SC.equal_sym_ref axis rf
          | _ -> false)
  in
  p "the inlined copy keeps the Vectorized retype on its scope-nested loop"
    (vectorizes reg_inline menu_inline k = 1);
  p "the schedule mint keeps it there too — the mint decides categories, not this one"
    (vectorizes reg_mint menu_mint k = 1);
  p "neither copy proposes vectorizing the enclosing loop, which no renderer would serve"
    (vectorizes reg_mint menu_mint i = 0 && vectorizes reg_inline menu_inline i = 0);
  Stdio.printf "\nDone.\n%!"
