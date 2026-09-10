(* gh-ocannl-666: the autotuner's menu enumeration descends accumulation mints.

   Since gh-ocannl-639, [Sched.Unroll { materialize = true }] (and [Partition]) of a recognized
   reduction nest mint a [Local_scope] holding the accumulator, and the inner loops move inside it —
   sharing their binders across the per-step copies. [Schedule.rewrite_loop] descends there, so
   those loops remain schedulable; the menu's own enumeration ([Autotune.collect_loops]) must reach
   them too, or the moment such an op joins a beam incumbent every inner loop vanishes from the rest
   of the search.

   The probe: out[i] = sum_{q,r,s,t} x[i,q,r,s,t], materializing-unrolled over q so the scope holds
   two copies of the r->s->t nest. Structural claims pin the menu's proposals over the transformed
   form: the scope-nested loops are enumerated (Split/Swap/Unroll/Vectorized-Retype proposals name
   them) and binder-sharing copies collapse to one decision. The untransformed nest is the control:
   its statement-level innermost loop draws the Vectorized retype the transformed form must move to
   the scope-nested innermost loop. ([Tensorize] needs no executable claim here: the loops inside an
   accumulation mint reduce into one loop-invariant cell, so no valid matmul micro-kernel triple can
   sit there — see [Autotune.collect_serial_triples].)

   The executed leg (cc, where every proposal family renders): every menu proposal applied on top of
   the materializing unroll must compile — the restriction to scope-surviving ops is exactly what
   makes the menu emit no candidate that cannot — and must reproduce the serial result. Cell values
   vary with every axis symbol through distinct coefficients and stay integer-valued, so any sum
   reordering the proposals introduce is exact in f32 and the parity is bitwise, while a dropped or
   wrongly-substituted iteration shifts the result. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Idx = Ir.Indexing
module Sched = Ir.Schedule
module SC = Ir.Schedule_cache
module Asgns = Ir.Assignments
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let skipped = Verdict.skipped ~backend:backend_name
let on_cpu = Sched.backend_is_cpu backend_name
let cc_only claim leg = if on_cpu then leg () else skipped claim

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let ni, nq, nr, ns, nt = (4, 2, 2, 2, 8)

(* Discriminating producer: varies with every axis symbol (distinct coefficients), never 0. *)
let fx idcs =
  Float.of_int (1 + (2 * idcs.(0)) + idcs.(1) + (3 * idcs.(2)) + (5 * idcs.(3)) + (7 * idcs.(4)))

let x = NTDSL.init ~l:"smn_x" ~prec:Ir.Ops.single ~o:[ ni; nq; nr; ns; nt ] ~f:fx ()
let%op out = x ++ "iqrst => i"
let comp = named "smn" (Train.forward out)

(* The single-child chain of loops of the reduction nest (accum_width's helper). *)
let nest_path (llc : LL.t) : Idx.symbol list =
  let strip stmts = List.filter stmts ~f:(function LL.Noop | LL.Comment _ -> false | _ -> true) in
  let rec path (llc : LL.t) : Idx.symbol list =
    match llc with
    | LL.For_loop { index; body; _ } ->
        index :: (match strip (LL.flat_lines [ body ]) with [ single ] -> path single | _ -> [])
    | LL.If { body; _ } -> path body
    | _ -> []
  in
  List.find_map_exn (LL.flat_lines [ llc ]) ~f:(fun stmt ->
      match path stmt with p when List.length p = 5 -> Some p | _ -> None)

let hermetic (o : LL.optimized) =
  {
    o with
    LL.traced_store = Hashtbl.copy o.LL.traced_store;
    optimize_ctx = LL.copy_optimize_ctx o.LL.optimize_ctx;
  }

(* Synthetic CPU limits: a vector width so Vectorized retypes are proposed. *)
let limits = { Ir.Backend_intf.no_hardware_limits with simd_vector_bytes = 32 }

let () =
  (* One base compile captures the lowering the way the tuner does; the run supplies the serial
     reference values. *)
  let captured = ref None in
  let ctx = Context.auto () in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun o ->
        captured := Some o;
        [ o ])
      ctx comp Idx.Empty
  in
  let ctx = Context.run ctx routine in
  let reference = Context.get_values ctx out.Tensor.value in
  let wide =
    Array.init ni ~f:(fun i ->
        let acc = ref 0.0 in
        for q = 0 to nq - 1 do
          for r = 0 to nr - 1 do
            for s = 0 to ns - 1 do
              for t = 0 to nt - 1 do
                acc := !acc +. fx [| i; q; r; s; t |]
              done
            done
          done
        done;
        !acc)
  in
  p_all2 "the serial reference matches the host-side wide sum" reference wide ~f:Float.equal;
  let pre = Option.value_exn !captured in
  let i, q, r, s, t =
    match nest_path pre.LL.llc with [ i; q; r; s; t ] -> (i, q, r, s, t) | _ -> assert false
  in
  let pre_canon = SC.canonicalize ~static_indices:[] ~with_placements:false pre in
  let ref_of sym = Option.value_exn (SC.resolve (SC.base_registry pre_canon) sym) in
  let ri, rr, rs, rt = (ref_of i, ref_of r, ref_of s, ref_of t) in
  (* The control: the untransformed nest's statement-level innermost loop draws the Vectorized
     retype. *)
  let pre_menu =
    Autotune.menu ~is_cpu:true ~is_gpu:false ~limits ~registry:(SC.base_registry pre_canon)
      (hermetic pre)
  in
  p "control: the untransformed nest proposes vectorizing its innermost loop"
    (List.exists pre_menu ~f:(function
      | SC.Retype { axis; ty = LL.Vectorized } -> SC.equal_sym_ref axis rt
      | _ -> false));
  (* The transformed unit, built the way the search builds one: the materializing unroll of q mints
     the accumulator scope and moves the r->s->t copies inside it. *)
  let unroll_q = [ Sched.Unroll { axis = q; materialize = true } ] in
  let saved_unroll, registry = SC.to_saved (SC.base_registry pre_canon) unroll_q in
  let post = Sched.apply unroll_q (hermetic pre) in
  let menu = Autotune.menu ~is_cpu:true ~is_gpu:false ~limits ~registry post in
  let split_axis = function SC.Split { axis; _ } -> Some axis | _ -> None in
  let targets rf op =
    match op with
    | SC.Split { axis; _ } | SC.Unroll { axis; _ } | SC.Retype { axis; _ } ->
        SC.equal_sym_ref axis rf
    | SC.Swap { outer; inner } -> SC.equal_sym_ref outer rf || SC.equal_sym_ref inner rf
    | _ -> false
  in
  p_all "scope-nested loops are enumerated (each inner reduction loop draws a proposal)"
    [ rr; rs; rt ] ~f:(fun rf -> List.exists menu ~f:(targets rf));
  p "a dividing Split of the scope-nested innermost loop is proposed"
    (List.exists menu ~f:(fun op ->
         match split_axis op with Some axis -> SC.equal_sym_ref axis rt | None -> false));
  p "binder-sharing mint copies are one decision (one Split-by-2 of the copied loop)"
    (1
    = List.count menu ~f:(function
      | SC.Split { axis; factor = 2; _ } -> SC.equal_sym_ref axis rt
      | _ -> false));
  p "a Swap of a perfect serial pair inside the scope is proposed"
    (List.exists menu ~f:(function
      | SC.Swap { outer; inner } ->
          (SC.equal_sym_ref outer rr && SC.equal_sym_ref inner rs)
          || (SC.equal_sym_ref outer rs && SC.equal_sym_ref inner rt)
      | _ -> false));
  p "an Unroll of a scope-nested loop is proposed"
    (List.exists menu ~f:(function SC.Unroll { axis; _ } -> SC.equal_sym_ref axis rs | _ -> false));
  p "the Vectorized retype moves to the scope-nested innermost accumulating loop"
    (List.exists menu ~f:(function
      | SC.Retype { axis; ty = LL.Vectorized } -> SC.equal_sym_ref axis rt
      | _ -> false));
  p_none
    "the statement-level loop over the scope no longer reads as innermost (no Vectorized retype)"
    menu ~f:(function
    | SC.Retype { axis; ty = LL.Vectorized } -> SC.equal_sym_ref axis ri
    | _ -> false);
  (* The executed leg: every proposal, replayed through the cache's saved form the way a beam
     candidate is, must compile and reproduce the serial result. *)
  let parity_leg ~claim ~saved_prefix menu_ops =
    cc_only claim (fun () ->
        let all_ok =
          (not (List.is_empty menu_ops))
          && (not (Array.is_empty reference))
          && List.for_all menu_ops ~f:(fun op ->
              let saved = saved_prefix @ [ op ] in
              let cctx = Context.auto () in
              let cctx, croutine =
                Context.compile
                  ~lowered_transform:(fun o ->
                    [
                      (let canon = SC.canonicalize ~static_indices:[] ~with_placements:false o in
                       let sched, _reg = SC.of_saved canon saved in
                       Sched.apply sched o);
                    ])
                  cctx comp Idx.Empty
              in
              let cctx = Context.run cctx croutine in
              let got = Context.get_values cctx out.Tensor.value in
              let ok = Array.for_all2_exn got reference ~f:Float.equal in
              if not ok then
                Stdio.eprintf "menu proposal diverges: %s\n"
                  (Sexp.to_string_hum (SC.sexp_of_saved_optop op));
              ok)
        in
        p claim all_ok)
  in
  parity_leg
    ~claim:"every menu proposal on the minted form compiles and matches the serial result"
    ~saved_prefix:saved_unroll menu;
  (* === Partition: segments after the first start at their breakpoint (absolute ranges), and only
     [Split]'s index arithmetic needs a zero-origin loop — the other families must keep proposing
     for them (Codex P2 on PR #403). Partitioning the innermost reduction axis mints one scope
     spanning both segment loops. *)
  let pt, segment_syms = Sched.partition ~axis:t ~breakpoints:[ nt / 2 ] in
  let saved_pt, pt_registry = SC.to_saved (SC.base_registry pre_canon) [ pt ] in
  let pt_post = Sched.apply [ pt ] (hermetic pre) in
  let pt_menu = Autotune.menu ~is_cpu:true ~is_gpu:false ~limits ~registry:pt_registry pt_post in
  let seg1, seg2 = match segment_syms with [ a; b ] -> (a, b) | _ -> assert false in
  let rseg1 = Option.value_exn (SC.resolve pt_registry seg1) in
  let rseg2 = Option.value_exn (SC.resolve pt_registry seg2) in
  p "the nonzero-origin partition segment draws an Unroll proposal"
    (List.exists pt_menu ~f:(function
      | SC.Unroll { axis; _ } -> SC.equal_sym_ref axis rseg2
      | _ -> false));
  p "the nonzero-origin partition segment draws the Vectorized retype"
    (List.exists pt_menu ~f:(function
      | SC.Retype { axis; ty = LL.Vectorized } -> SC.equal_sym_ref axis rseg2
      | _ -> false));
  p_none "no Split targets the nonzero-origin segment (Split alone requires a zero origin)" pt_menu
    ~f:(function
    | SC.Split { axis; _ } -> SC.equal_sym_ref axis rseg2
    | _ -> false);
  p "the zero-origin first segment still draws a dividing Split"
    (List.exists pt_menu ~f:(function
      | SC.Split { axis; _ } -> SC.equal_sym_ref axis rseg1
      | _ -> false));
  parity_leg
    ~claim:"every menu proposal on the partitioned form compiles and matches the serial result"
    ~saved_prefix:saved_pt pt_menu
