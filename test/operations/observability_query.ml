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
   a merge-buffer read is reported. The merge answer is pinned at that level only: building a
   lineage whose stored computation reads a merge buffer needs a multi-device transfer setup.

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
let mk label =
  Tn.create (Tn.Specified Ir.Ops.single)
    ~id:(9000 + String.length label)
    ~label:[ label ]
    ~unpadded_dims:(lazy [| 4 |])
    ~padding:(lazy None)
    ()

let () =
  let self = mk "rf_self"
  and sib = mk "rf_sibling"
  and src = mk "rf_src"
  and sib_src = mk "rf_sib_src" in
  let i = Ir.Indexing.get_symbol () and launch = Ir.Indexing.get_symbol () in
  let it = Ir.Indexing.Iterator i in
  let get tn idx = Ll.Get (tn, [| idx |]) in
  let set tn llsc = Ll.Set { tn; idcs = [| it |]; llsc; debug = "" } in
  let body extra =
    Ll.For_loop
      {
        index = i;
        from_ = 0;
        to_ = 3;
        axis = Ll.Serial;
        body =
          Ll.Seq
            ( set self (Ll.Binop (Ir.Ops.Add, (get src it, Ir.Ops.single), (extra, Ir.Ops.single))),
              (* The sibling setter reads its own source at a launch index. *)
              set sib (get sib_src (Ir.Indexing.Iterator launch)) );
      }
  in
  let facts = Ll.replay_facts ~self (body (Ll.Constant 1.)) in
  p "the replay reads self's source" (Set.equal facts.replay_reads (Set.singleton (module Tn) src));
  p "a sibling setter's read and launch symbol are not the replay's"
    (Set.is_empty facts.replay_free_symbols);
  let facts = Ll.replay_facts ~self (body (Ll.Embed_index (Ir.Indexing.Iterator launch))) in
  p "self's own launch symbol is free"
    (Set.equal facts.replay_free_symbols (Set.singleton (module Ir.Indexing.Symbol) launch));
  let facts = Ll.replay_facts ~self (body (Ll.Get_merge_buffer (sib_src, [| it |]))) in
  p "a merge-buffer read in self's setter is reported" facts.replay_merge_reads;
  let dead =
    Ll.For_loop { index = i; from_ = 1; to_ = 0; axis = Ll.Serial; body = set self (get src it) }
  in
  p "a dead loop replays nothing" (Set.is_empty (Ll.replay_facts ~self dead).replay_reads)
