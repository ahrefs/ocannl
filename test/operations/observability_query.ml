(* gh-ocannl-777: [Context.observability] answers, without compiling, whether a host read of a node
   can be served from a context, and if not, why.

   The headline pair is one program with two virtual nodes reading the same [Local] scratch [h]
   (placed [Local] by the ordinary pipeline: its reduction extent, 32, exceeds
   [virtualize_max_inline_reduction]): - [v] reads it through the materialized [m]: shielded, so
   recomputable; - [u] reads it directly: unobservable, and the answer names the path [u -> h] and
   [h]'s placement provenance.

   Around it, every other answer the query gives -- [Stored], [Host_init], [Needs_context] (with
   several leaves, deduplicated over a diamond), [Launch_bound], [Slice_view], [Not_computed] -- and
   the replay filter the walk reads through ([Ir.Low_level.replay_facts]): the sibling setters of a
   shared-loop body contribute neither a read nor a launch symbol, a dead loop replays nothing, and
   a merge-buffer read is reported. The merge answer is also pinned through real contexts, one cc
   device and a multidev_cc pair, filled by [Context.copy ~into_merge_buffer] (gh-ocannl-1741), and
   the walk's internal error -- a stored computation reading an undecided node -- on a hand-built
   lineage whose stored component is replaced after its link.

   Each placement is asserted before the claim that depends on it, so no claim passes vacuously, and
   the query is pinned as pure: it decides no placement, stores no computation and compiles
   nothing. *)

open Base
open Stdio
open Ocannl
open Ocannl.Operation.DSL_modules
module IDX = Train.IDX
module Tn = Ir.Tnode
module Ll = Ir.Low_level
open Verdict.Claims

let rows = 4
let inner = 32
let a_values = Array.init (rows * inner) ~f:(fun n -> Float.of_int ((n / inner) + 1))
let x_values = Array.init inner ~f:Float.of_int
let mode ctx (t : Tensor.t) = Option.map ~f:fst (Tn.Placements.get (Context.placements ctx) t.value)
let is_mode m ctx t = Option.equal Tn.equal_memory_mode (mode ctx t) (Some m)
let obs ctx (t : Tensor.t) = Context.observability ctx t.value
let tns ts = List.map ts ~f:(fun (t : Tensor.t) -> t.value)
let same_path got want = List.equal Tn.equal got (tns want)

let show ctx name (t : Tensor.t) =
  let o = obs ctx t in
  printf "%s: %s\n" name (Option.value (Context.observability_tag o) ~default:"<readable>");
  Option.iter (Context.explain_observability ctx t.value o) ~f:(printf "  %s\n")

let snapshot ctx =
  ( Sexp.to_string (Tn.Placements.sexp_of_t (Context.placements ctx)),
    Context.stored_component_total ctx,
    Context.backend_compiles () )

let () =
  (* --- Shielded vs unshielded --- *)
  let x = TDSL.ndarray x_values ~label:[ "oq_x" ] ~output_dims:[ inner ] () in
  let a = TDSL.ndarray a_values ~label:[ "oq_a" ] ~input_dims:[ inner ] ~output_dims:[ rows ] () in
  let%op h = a * x in
  let%op m = h *. 2 in
  let%op v = m + 1 in
  let%op u = h + 3 in
  let%op out = v *. u in
  Train.set_materialized m.Tensor.value;
  let ctx = Train.forward_once (Context.cpu ()) out in
  p "h is placed Local by the pipeline" (is_mode Tn.Local ctx h);
  p "m is materialized" (is_mode Tn.On_device ctx m);
  p "v and u are virtual" (is_mode Tn.Virtual ctx v && is_mode Tn.Virtual ctx u);
  let before = snapshot ctx in
  p "v, shielded from h by the materialized m, is recomputable"
    (match obs ctx v with Recomputable -> true | _ -> false);
  p "u, reading h directly, is unobservable along the path u -> h"
    (match obs ctx u with Unobservable_local path -> same_path path [ u; h ] | _ -> false);
  p "h itself is unobservable"
    (match obs ctx h with Unobservable_local path -> same_path path [ h ] | _ -> false);
  p "the stored output and an input are stored"
    (match (obs ctx out, obs ctx m, obs ctx x) with Stored, Stored, Stored -> true | _ -> false);
  let u_reason =
    Option.value_exn ~here:[%here] (Context.explain_observability ctx u.value (obs ctx u))
  in
  p "the reason names h's placement provenance"
    (String.is_substring u_reason ~substring:"39:inline-reduction-cap");
  show ctx "u" u;
  (* --- Host_init and Not_computed --- *)
  let lit = TDSL.ndarray [| 1.; 2. |] ~label:[ "oq_lit" ] ~output_dims:[ 2 ] () in
  p "an ndarray literal outside every routine is host-initialized"
    (match obs ctx lit with Host_init -> true | _ -> false);
  let fresh = TDSL.range_of_shape ~label:[ "oq_fresh" ] ~output_dims:[ 3 ] () in
  p "a tensor no routine of the lineage computes is not computed"
    (match obs ctx fresh with Not_computed -> true | _ -> false);
  let sibling = Context.cpu () in
  p "v read from a context not descending from its compile is not computed"
    (match obs sibling v with Not_computed -> true | _ -> false);
  show sibling "v from an unrelated context" v;
  p "the queries decided no placement, stored nothing and compiled nothing"
    (let (p1, s1, c1), (p2, s2, c2) = (before, snapshot ctx) in
     String.equal p1 p2 && s1 = s2 && c1 = c2);
  (* --- A diamond over materialized leaves, and Needs_context --- [d] reads [w1] twice. A
     materialized node with no buffer here needs another context; a VIRTUAL node cannot need one in
     practice, since linking a routine that stores its computation requires every leaf it reads to
     be present ([verify_prior_context]) and descendants inherit buffers -- the leaf form of the
     answer is the defensive one. *)
  let w1 = TDSL.param ~value:1. "oq_w1" ~output_dims:[ 2 ] () in
  let w2 = TDSL.param ~value:2. "oq_w2" ~output_dims:[ 2 ] () in
  let%op d = (w1 *. w2) + w1 in
  let%op dy = d *. 3 in
  let dctx = Train.forward_once (Context.cpu ()) dy in
  p "the diamond's d is virtual" (is_mode Tn.Virtual dctx d);
  p "d over its two present leaves is recomputable"
    (match obs dctx d with Recomputable -> true | _ -> false);
  let w3 = TDSL.param ~value:3. "oq_w3" ~output_dims:[ 2 ] () in
  Train.set_materialized w3.Tensor.value;
  p "a materialized node absent from the context needs a context"
    (match obs dctx w3 with Needs_context [ l ] -> Tn.equal l w3.value | _ -> false);
  show dctx "w3" w3;
  (* --- Launch_bound and Slice_view --- [e] embeds a static (launch) index; [s] is a slice view at
     another one. *)
  let k, bindings = IDX.get_static_symbol IDX.empty in
  let%op e = !@k + 0.5 in
  let%op ey = e *. 2 in
  let lctx = Train.forward_once ~bindings ~launch_values:[ (k, 3) ] (Context.cpu ()) ey in
  p "e is virtual" (is_mode Tn.Virtual lctx e);
  p "e depends on the launch index k"
    (match obs lctx e with
    | Launch_bound (path, syms) ->
        same_path path [ e ]
        && List.equal Ir.Indexing.equal_symbol syms [ k.Ir.Indexing.static_symbol ]
    | _ -> false);
  let i, bindings = IDX.get_static_symbol ~static_range:2 IDX.empty in
  let base =
    TDSL.ndarray [| 1.; 2.; 3.; 4. |] ~label:[ "oq_base" ] ~batch_dims:[ 2 ] ~output_dims:[ 2 ] ()
  in
  let%op s = base @| i in
  let%op sy = s *. 3 in
  let sctx = Train.forward_once ~bindings ~launch_values:[ (i, 1) ] (Context.cpu ()) sy in
  p "a slice view is read through its parent"
    (match obs sctx s with Slice_view parent -> Tn.equal parent base.value | _ -> false);
  (* --- A released context is refused, not answered --- *)
  let rctx =
    Train.forward_once (Context.cpu ())
      (TDSL.range_of_shape ~label:[ "oq_rel" ] ~output_dims:[ 2 ] ())
  in
  Context.release rctx;
  p "a released context is refused"
    (try
       ignore (Context.observability rctx out.value);
       false
     with Utils.User_error _ -> true)

(* --- The replay filter, on hand-built code --- *)
let () =
  let open Ll_test in
  let mk = node_factory ~first_id:9000 ~dims:[| 4 |] () in
  let self = mk "rf_self" and sib = mk "rf_sibling" and src = mk "rf_src" in
  let sib_src = mk "rf_sib_src" in
  let i = sym () and launch = sym () in
  let body extra =
    loop_n i 4
      (seq
         (set self [| iter i |] (add (get src [| iter i |]) extra))
         (* The sibling setter reads its own source at a launch index. *)
         (set sib [| iter i |] (get sib_src [| iter launch |])))
  in
  let facts = Ll.replay_facts ~self (body (c 1.)) in
  p "the replay reads self's source" (Set.equal facts.replay_reads (Set.singleton (module Tn) src));
  p "a sibling setter's read and launch symbol are not the replay's"
    (Set.is_empty facts.replay_free_symbols);
  let facts = Ll.replay_facts ~self (body (embed launch)) in
  p "self's own launch symbol is free"
    (Set.equal facts.replay_free_symbols (Set.singleton (module Ir.Indexing.Symbol) launch));
  let facts = Ll.replay_facts ~self (body (Ll.Get_merge_buffer (sib_src, [| iter i |]))) in
  p "a merge-buffer read in self's setter is reported" facts.replay_merge_reads;
  let dead = loop ~from_:1 ~upto:0 i (set self [| iter i |] (get src [| iter i |])) in
  p "a dead loop replays nothing" (Set.is_empty (Ll.replay_facts ~self dead).replay_reads)

(* --- A stored computation reading a merge buffer, through a real context (gh-ocannl-1741) --- [mv]
   is written from [mt]'s merge buffer, filled by a [Context.copy ~into_merge_buffer], and read
   within the same routine by the materialized [mo], so the lineage stores [mv]'s merge-reading
   computation. A merge buffer holds one transfer, so no other context can serve the read. The leg
   runs on one cc device and on a multidev_cc pair. *)
let () =
  let merge_leg ~tag make_ctx =
    let%op mt = [ 1.; 2. ] + [ 10.; 20. ] in
    let%op mv = [ 0.; 0. ] + [ 0.; 0. ] in
    let%op mo = [ 0.; 0. ] + [ 0.; 0. ] in
    Train.set_materialized mo.Tensor.value;
    let src = Train.forward_once (make_ctx ()) mt in
    let dst = Train.forward_once (make_ctx ()) mt in
    let dst = Context.copy ~into_merge_buffer:Copy ~src ~dst mt.Tensor.value in
    let consumer =
      [%cd
        mv =: mt.merge;
        mo =: mv + mv]
    in
    let consumer =
      { consumer with asgns = Ir.Assignments.Block_comment ("oq_merge_" ^ tag, consumer.asgns) }
    in
    let ctx, routine = Context.compile dst consumer IDX.empty in
    let ctx = Context.run ctx routine in
    p (tag ^ ": the merge reader mv is virtual") (is_mode Tn.Virtual ctx mv);
    p
      (tag ^ ": the consumer read the transferred values")
      (Array.equal Float.equal (Context.get_values ctx mo.Tensor.value) [| 22.; 44. |]);
    p
      (tag ^ ": mv, whose stored computation reads a merge buffer, is unobservable")
      (match obs ctx mv with Unobservable_merge path -> same_path path [ mv ] | _ -> false);
    show ctx (tag ^ " mv") mv
  in
  merge_leg ~tag:"cc" (fun () -> Context.cpu ());
  merge_leg ~tag:"multidev_cc" (fun () -> Context.cpu ~threads:2 ())

(* --- The internal-error branch, on a hand-built lineage (gh-ocannl-1741) --- A compile that stores
   a computation decides every node it reads, so no pipeline reaches the branch; a stored component
   replaced after the link, reading a node no placement was decided for, does. Before the
   replacement the same query answers. *)
let () =
  let open Ll_test in
  let mk = node_factory ~first_id:9100 ~dims:[| 4 |] () in
  let src = mk "ie_src" and v = mk "ie_v" and out = mk "ie_out" and ghost = mk "ie_ghost" in
  let i = sym () and j = sym () and k = sym () in
  let code =
    seq
      (loop_n i 4 (set v [| iter i |] (add (get src [| iter i |]) (c 1.))))
      (loop_n j 4 (set out [| iter j |] (mul (get v [| iter j |]) (c 2.))))
  in
  let o =
    optimize_in ~materialized:[ src; out ] (Ll.empty_optimize_ctx ()) ~name:"oq_internal" code
  in
  p "the hand-built v is virtual" (known_virtual o v);
  let ctx, _ = link ~ctx:(Context.cpu ()) ~name:"oq_internal" o in
  let internal_error () =
    match Context.observability ctx v with
    | exception Failure msg -> String.is_substring msg ~substring:"internal error"
    | _ -> false
  in
  p "before the replacement, v's query answers" (not (internal_error ()));
  let comps = o.Ll.optimize_ctx.Ll.computations in
  Hashtbl.update comps v ~f:(function
    | Some ((idx, _) :: _) -> [ (idx, loop_n k 4 (set v [| iter k |] (get ghost [| iter k |]))) ]
    | _ -> []);
  p "a stored computation reading an undecided node is an internal error" (internal_error ());
  Hashtbl.set comps ~key:v ~data:[];
  p "a virtual node with an empty stored computation is an internal error" (internal_error ())
