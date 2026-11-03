(* gh-ocannl-1745, gh-ocannl-1796: the session's root maps do not own their graphs. A differentiable
   graph whose forward code left the forward-root map (a forward-only evaluation) stays a backprop
   root -- a loss built over it later trains through it (gh-ocannl-1724's leg B) -- but only while
   the caller can still name it: once dropped, the GC reclaims it, the closure checkpoint and
   explicit checkpoints included. Since gh-ocannl-1796 the forward-root map is weak too: a graph
   dropped unevaluated is reclaimed, and so is a root only the last closure captured.
   [consume_forward_code]'s conflicting-roots refusal reads the owner cells, not the map, so it does
   not depend on GC timing: a held owner and a dropped one are both refused, and the cell verdict
   equals the old live-root scan wherever every root is held.

   Witnesses ([Gc_witness]): each dropped graph is built in an [@inline never] function that returns
   a [Weak] observer on the root record and only what the case keeps. A "reclaimed" claim retries
   the collection ([Gc.full_major] twice and [Gc.compact]) up to three times; a "kept" claim
   requires the record to survive three such rounds, so neither rests on a single collection. Leaves
   are deterministic [term_init] constants, so no seed state is involved. *)
open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims
module LC = Lineage_claims
module IDX = Train.IDX
module Asgns = Ir.Assignments

let leaf name =
  Tensor.term_init [| 1.; 2.; 3. |] ~label:[ name ] ~output_dims:[ 3 ] ~grad_spec:Require_grad ()

let not_handed_out (t : Tensor.t) =
  match t.consumption.bprop with Not_handed_out -> true | Taken -> false

let witness (t : Tensor.t) = Gc_witness.witness t

let unpropagated msg tn =
  let name = Ir.Tnode.label_name tn in
  (LC.has msg "no routine of its lineage carries" && LC.has msg ("propagating from " ^ name))
  || LC.has msg ("no routine in this lineage propagates from " ^ name)

(* [t = w *. w] over a fresh [Require_grad] leaf, its forward run once. *)
let[@inline never] evaluated prefix =
  let w = leaf (prefix ^ "_w") in
  let%op t = w *. w in
  Train.set_materialized (LC.grad_of w);
  let ctx = Train.forward_once (Context.auto ()) t in
  (w, t, ctx)

let[@inline never] forward_once_dropped () =
  let _w, t, ctx = evaluated "dg_fo" in
  p "forward_once: before the drop, t is a backprop root and no longer a forward root"
    (Tensor.is_bprop_root t && not (Tensor.is_fwd_root t));
  (ctx, witness t)

let[@inline never] reusable_forward_dropped () =
  let w = leaf "dg_rf_w" in
  let%op t = w *. w in
  let ctx, routine = Train.to_routine (Context.auto ()) IDX.empty (Train.forward t) in
  let ctx = Context.run ctx routine in
  p "Train.forward: before the drop, t is a backprop root and no longer a forward root"
    (Tensor.is_bprop_root t && not (Tensor.is_fwd_root t));
  ignore (Stdlib.Sys.opaque_identity ctx : Context.t);
  witness t

(* The checkpoint is taken after the closure, while [t] is a backprop root, and outlives [t]. *)
let[@inline never] checkpointed_dropped () =
  let _w, t, _ctx = evaluated "dg_ck" in
  let ck = Tensor.checkpoint ~label:"dg_ck" () in
  p "checkpoint: t is a backprop root when the checkpoint is taken" (Tensor.is_bprop_root t);
  (ck, witness t)

(* Only an ownership-preserving copy of [t] is returned: a record that shares [t]'s consumption cell
   but is not [t]. The registered record is then reachable only through the root map's entry, so a
   map anchored on the record rather than on the cell would lose the root here. *)
let[@inline never] copy_kept () =
  let w, t, ctx = evaluated "dg_cp" in
  let copy = { t with Tensor.children = t.Tensor.children } in
  p "copy: the copy is a distinct record" (not (phys_equal copy t));
  (w, copy, ctx)

(* A shared trunk [h] (as in lineage_fragments): [l1] embeds [h]'s backprop, so [l2] requires
   [(h.grad, Bprop)]. [h] is computed and its gradient zeroed first; the losses are built after that
   closure, and [l1] is evaluated forward-only (its compile is the next closure) and dropped. *)
let[@inline never] trunk_owner_dropped () =
  let w = TDSL.param ~values:[| 1.; 2.; 3. |] "dg_tr_w" ~output_dims:[ 3 ] () in
  let%op h = w *. w in
  let h_fwd = Train.forward h in
  let h_zero =
    let z = (Option.value_exn ~here:[%here] h.diff).zero_grads in
    { z with asgns = Asgns.Block_comment ("dg_tr_h_zero", z.asgns) }
  in
  let ctx = Train.init_params (Context.auto ()) IDX.empty h in
  let ctx, r = Train.to_routine ctx IDX.empty h_fwd in
  let ctx = Context.run ctx r in
  let ctx, r = Train.to_routine ctx IDX.empty h_zero in
  let ctx = Context.run ctx r in
  let%op l1 = h *. 2. in
  let%op l2 = h *. 3. in
  let ctx = Train.forward_once ~skip_init:true ctx l1 in
  p "trunk: before the drop, l1 is a backprop root and no longer a forward root"
    (Tensor.is_bprop_root l1 && not (Tensor.is_fwd_root l1));
  (LC.grad_of h, l2, ctx, witness l1)

(* The consumer [u] embeds the child [c]'s forward and backprop; [u] is evaluated and dropped, [c]
   is kept. *)
let[@inline never] consumer_dropped () =
  let w = leaf "dg_cn_w" in
  let%op c = w *. w in
  let%op u = c *. 2. in
  let ctx = Train.forward_once (Context.auto ()) u in
  ignore (Stdlib.Sys.opaque_identity ctx : Context.t);
  (c, witness u)

(* [t] is built, the parameters' initialization closes its shapes while it is still an unconsumed
   forward root, and only then is its forward run and [t] dropped: the last closure's checkpoint
   captured it as a forward root, and holds it only weakly. *)
let[@inline never] closed_before_evaluation () =
  let w = TDSL.param ~values:[| 1.; 2.; 3. |] "dg_cl_w" ~output_dims:[ 3 ] () in
  let%op t = w *. 2. in
  let ctx = Train.init_params (Context.auto ()) IDX.empty t in
  let ctx = Train.forward_once ~skip_init:true ctx t in
  ignore (Stdlib.Sys.opaque_identity ctx : Context.t);
  witness t

(* A closure of an unrelated graph. *)
let[@inline never] a_closure () =
  let x = leaf "dg_nc_x" in
  let%op y = x *. 2. in
  ignore (Train.forward_once (Context.auto ()) y : Context.t)

let[@inline never] unevaluated_dropped () =
  let w = leaf "dg_ue_w" in
  let%op t = w *. w in
  p "unevaluated: t is a forward and a backprop root"
    (Tensor.is_fwd_root t && Tensor.is_bprop_root t);
  witness t

(* {2 The conflicting-roots refusal (gh-ocannl-1796)}

   [shared] is a computed trunk whose code its first-constructed consumer [owner] embeds; [sibling]
   reads [shared] without embedding it, so consuming [sibling] is refused while [owner]'s code is
   unconsumed -- and stays refused once [owner] is dropped, since its code can then never run. *)

let nleaf name =
  Tensor.term_init [| 1.; 2.; 3. |] ~label:[ name ] ~output_dims:[ 3 ] ~grad_spec:Prohibit_grad ()

let name_of (t : Tensor.t) = Ir.Tnode.label_name t.value

let fwd_refusal name f =
  match f () with
  | _ ->
      fail (name ^ ": was accepted");
      ""
  | exception Tensor.Session_error (msg, _) ->
      Stdio.eprintf "%s: %s (not part of the golden)\n%!" name msg;
      msg

let consumed name t =
  match Tensor.consume_forward_code t with
  | _ -> true
  | exception Tensor.Session_error (msg, _) ->
      Stdio.eprintf "%s: refused: %s (not part of the golden)\n%!" name msg;
      false

let conflicting msg ~owner = LC.has msg "found conflicting roots" && LC.has msg owner

(* The cell verdict, pinned to the live-root scan it replaced while every root is held. *)
let verdicts_agree name t ~expected =
  let by_cells = Tensor.For_testing.forward_conflicts_by_cells t in
  let by_scan = Tensor.For_testing.forward_conflicts_by_live_roots t in
  p
    (name ^ ": the owner cells and the live-root scan agree")
    (List.equal String.equal by_cells by_scan);
  p (name ^ ": they name the expected roots") (List.equal String.equal by_cells expected)

let trio prefix =
  let x = nleaf (prefix ^ "_x") in
  let shared = NTDSL.O.relu ~label:[ prefix ^ "_shared" ] x in
  let owner = NTDSL.O.relu ~label:[ prefix ^ "_owner" ] shared in
  let sibling = NTDSL.O.relu ~label:[ prefix ^ "_sibling" ] shared in
  (shared, owner, sibling)

let[@inline never] dropped_owner prefix =
  let _shared, owner, sibling = trio prefix in
  (sibling, name_of owner, witness owner)

(* [c]'s forward code replayed into a new root over [c]: the root's own [op_asn] embeds it. *)
let replay ~label (code : Asgns.comp) c =
  let%cd op_asn ~t ~t1 ~projections = v =:+ relu v1 in
  let op_asn ~t ~t1 ~projections = Asgns.sequence [ code; op_asn ~t ~t1 ~projections ] in
  Tensor.unop ~op_label:"replay" ~op_asn
    ~grad_asn:(fun ~t:_ ~g:_ ~t1:_ ~projections:_ -> Asgns.empty_comp)
    ~grad_spec:Prohibit_grad c ~label:[ label ] ()

(* A handed-out computation [c], to replay. *)
let handed_out prefix =
  let c = NTDSL.O.relu ~label:[ prefix ^ "_c" ] (nleaf (prefix ^ "_x")) in
  let code = Tensor.consume_forward_code c in
  (c, code)

let[@inline never] scoped_consumed_owner () =
  let _shared, owner, sibling = trio "fo_sc" in
  Tensor.with_unchanged_roots ~f:(fun () -> ignore (Tensor.consume_forward_code owner : Asgns.comp));
  (sibling, name_of owner, witness owner)

let[@inline never] transient_replay ~label c code = witness (replay ~label code c)

let[@inline never] rewound_owner () =
  let _shared, owner, sibling = trio "fo_rw" in
  let ck = Tensor.checkpoint ~label:"fo_rw" () in
  let u = NTDSL.O.relu ~label:[ "fo_rw_u" ] owner in
  ignore (Stdlib.Sys.opaque_identity u : Tensor.t);
  (sibling, name_of owner, ck, witness owner)

(* [wrapped]: the replaying root is embedded by a consumer, which is what gets handed out, so the
   held computation's owner is resolved only through its own owner. *)
let[@inline never] replay_round ~wrapped ~label c code =
  let r = replay ~label code c in
  let live = Tensor.For_testing.owner_count c in
  let out = if wrapped then NTDSL.O.relu ~label:[ label ^ "_s" ] r else r in
  ignore (Tensor.consume_forward_code out : Asgns.comp);
  live

let () =
  case "a graph dropped after forward_once is reclaimed" (fun () ->
      let ctx, w = forward_once_dropped () in
      p "forward_once: the dropped graph is reclaimed" (Gc_witness.reclaimed w);
      ignore (Stdlib.Sys.opaque_identity ctx : Context.t));
  case "a graph dropped after a reusable forward routine is reclaimed" (fun () ->
      let w = reusable_forward_dropped () in
      p "Train.forward: the dropped graph is reclaimed" (Gc_witness.reclaimed w));
  case "an explicit checkpoint does not keep a dropped graph" (fun () ->
      let ck, w = checkpointed_dropped () in
      p "checkpoint: the dropped graph is reclaimed while the checkpoint is held"
        (Gc_witness.reclaimed w);
      ignore (Stdlib.Sys.opaque_identity ck : Tensor.checkpoint));
  case "leg B under GC pressure" (fun () ->
      let w, t, ctx = evaluated "dg_b" in
      Gc_witness.settle ();
      p "leg B: the held t is still a backprop root after collections" (Tensor.is_bprop_root t);
      let%op loss = t *. 3. in
      let ctx = Train.update_once ctx loss in
      LC.exact "leg B: w.grad = 6 w" (Context.get_values ctx (LC.grad_of w)) [| 6.; 12.; 18. |]);
  case "an ownership-preserving copy keeps the root" (fun () ->
      let w, copy, ctx = copy_kept () in
      Gc_witness.settle ();
      p "copy: the copy is still a backprop root after collections" (Tensor.is_bprop_root copy);
      let%op loss = copy *. 3. in
      let ctx = Train.update_once ctx loss in
      LC.exact "copy: w.grad = 6 w through the copy"
        (Context.get_values ctx (LC.grad_of w))
        [| 6.; 12.; 18. |]);
  case "the lineage check survives its owner's reclamation" (fun () ->
      let h_grad, l2, ctx, l1 = trunk_owner_dropped () in
      p "trunk: the dropped owner l1 is reclaimed" (Gc_witness.reclaimed l1);
      let msg = LC.refusal "trunk" (fun () -> Train.update_once ~skip_init:true ctx l2) in
      p "trunk: l2 is refused at run naming h's backprop fragment"
        (unpropagated msg h_grad && LC.has msg "accumulates into gradients"));
  case "a held root survives scoped restoration and rewind" (fun () ->
      let _w, t, _ctx = evaluated "dg_rs" in
      Tensor.with_unchanged_roots ~f:(fun () ->
          ignore (Tensor.consume_backprop_code t : Asgns.comp);
          Gc_witness.settle ());
      p "restore: with_unchanged_roots puts the held root back"
        (Tensor.is_bprop_root t && not_handed_out t);
      let ck = Tensor.checkpoint ~label:"dg_rs" () in
      let%op u = t *. 2. in
      ignore (Stdlib.Sys.opaque_identity u : Tensor.t);
      p "rewind: the consumer embedded t's backprop" (not (Tensor.is_bprop_root t));
      Gc_witness.settle ();
      Tensor.rewind ck;
      p "rewind: the held root is a root again, its marker reset"
        (Tensor.is_bprop_root t && not_handed_out t);
      let ck = Tensor.checkpoint ~label:"dg_rs_out" () in
      ignore (Tensor.consume_backprop_code t : Asgns.comp);
      Gc_witness.settle ();
      Tensor.rewind ck;
      p "rewind: an external handout stays excluded" (not (Tensor.is_bprop_root t)));
  case "a reclaimed consumer gives no ownership back" (fun () ->
      let c, u = consumer_dropped () in
      p "consumer: the dropped consumer is reclaimed" (Gc_witness.reclaimed u);
      p "consumer: its embedded child is neither root"
        ((not (Tensor.is_fwd_root c)) && not (Tensor.is_bprop_root c));
      let refused =
        match Tensor.consume_forward_code c with
        | _ -> false
        | exception Tensor.Session_error (msg, _) -> LC.has msg "is not a forward root"
      in
      p "consumer: consuming the child's forward code is still refused" refused);
  case "the last closure's checkpoint does not hold a forward root it captured" (fun () ->
      let w = closed_before_evaluation () in
      p "closure: the dropped root is reclaimed before any next closure" (Gc_witness.reclaimed w));
  case "a graph dropped unevaluated is reclaimed" (fun () ->
      let w = unevaluated_dropped () in
      p "unevaluated: the dropped graph is reclaimed" (Gc_witness.reclaimed w));
  case "a held owner refuses its sibling until it is handed out" (fun () ->
      let _shared, owner, sibling = trio "fo_h" in
      let w = witness owner in
      p "held: the owner is kept while held" (Gc_witness.kept w);
      verdicts_agree "held" sibling ~expected:[ name_of owner ];
      let msg = fwd_refusal "held" (fun () -> Tensor.consume_forward_code sibling) in
      p "held: the sibling is refused, naming the owner" (conflicting msg ~owner:(name_of owner));
      p "held: the owner can be handed out" (consumed "held owner" owner);
      verdicts_agree "held, owner handed out" sibling ~expected:[];
      p "held: then the sibling is accepted" (consumed "held sibling" sibling));
  case "a dropped owner still refuses its sibling" (fun () ->
      let sibling, owner, w = dropped_owner "fo_d" in
      p "dropped: the owner is reclaimed" (Gc_witness.reclaimed w);
      p "dropped: the owner cells still name it, the live-root scan no longer does"
        (List.equal String.equal (Tensor.For_testing.forward_conflicts_by_cells sibling) [ owner ]
        && not
             (List.mem
                (Tensor.For_testing.forward_conflicts_by_live_roots sibling)
                owner ~equal:String.equal));
      let msg = fwd_refusal "dropped" (fun () -> Tensor.consume_forward_code sibling) in
      p "dropped: the sibling is refused, naming the owner" (conflicting msg ~owner));
  case "a dropped owner refuses its sibling whatever the GC has done" (fun () ->
      let sibling, owner, _w = dropped_owner "fo_n" in
      let msg = fwd_refusal "uncollected" (fun () -> Tensor.consume_forward_code sibling) in
      p "uncollected: the sibling is refused, naming the owner" (conflicting msg ~owner));
  case "an owner embedded into a later consumer is released with it" (fun () ->
      let _shared, owner, sibling = trio "fo_e" in
      let later = NTDSL.O.relu ~label:[ "fo_e_later" ] owner in
      verdicts_agree "embedded" sibling ~expected:[ name_of later ];
      p "embedded: the later consumer can be handed out" (consumed "embedded later" later);
      verdicts_agree "embedded, consumer handed out" sibling ~expected:[];
      p "embedded: then the sibling is accepted" (consumed "embedded sibling" sibling));
  case "code replayed into two roots is owned by both, and by nothing else" (fun () ->
      let c, code = handed_out "fo_rp" in
      let a = replay ~label:"fo_rp_a" code c in
      let b = replay ~label:"fo_rp_b" code c in
      p "replay: a can be handed out" (consumed "replay a" a);
      let reads_a = NTDSL.O.relu ~label:[ "fo_rp_ra" ] a in
      verdicts_agree "replay, reader of a" reads_a ~expected:[];
      p "replay: a reader of the handed-out root's result is accepted"
        (consumed "replay reader of a" reads_a);
      let reads_c = NTDSL.O.relu ~label:[ "fo_rp_rc" ] c in
      verdicts_agree "replay, reader of c" reads_c ~expected:[ name_of b ];
      let msg = fwd_refusal "replay" (fun () -> Tensor.consume_forward_code reads_c) in
      p "replay: a reader of the replayed code is refused, naming the unconsumed root"
        (conflicting msg ~owner:(name_of b)));
  case "with_unchanged_roots restores a dropped owner's claim" (fun () ->
      let sibling, owner, w = scoped_consumed_owner () in
      p "scope: the owner is reclaimed" (Gc_witness.reclaimed w);
      let msg = fwd_refusal "scope" (fun () -> Tensor.consume_forward_code sibling) in
      p "scope: the sibling is refused, naming the owner" (conflicting msg ~owner));
  case "a transient owner inside with_unchanged_roots leaves no claim, closure or not" (fun () ->
      List.iter [ false; true ] ~f:(fun close ->
          let tag = if close then "closed" else "open" in
          let c, code = handed_out ("fo_tr_" ^ tag) in
          let w =
            Tensor.with_unchanged_roots ~f:(fun () ->
                let w = transient_replay ~label:("fo_tr_t_" ^ tag) c code in
                if close then (
                  Gc_witness.settle ();
                  a_closure ());
                w)
          in
          p ("transient (" ^ tag ^ "): the transient root is reclaimed") (Gc_witness.reclaimed w);
          let reader = NTDSL.O.relu ~label:[ "fo_tr_r_" ^ tag ] c in
          p
            ("transient (" ^ tag ^ "): a reader of the replayed code is accepted")
            (consumed ("transient " ^ tag) reader)));
  case "a rewind restores a dropped owner's claim" (fun () ->
      let sibling, owner, ck, w = rewound_owner () in
      Tensor.rewind ck;
      p "rewind: the owner is reclaimed" (Gc_witness.reclaimed w);
      let msg = fwd_refusal "rewind" (fun () -> Tensor.consume_forward_code sibling) in
      p "rewind: the sibling is refused, naming the owner" (conflicting msg ~owner));
  case "a held computation's owners do not accumulate across closures" (fun () ->
      List.iter [ false; true ] ~f:(fun wrapped ->
          let tag = if wrapped then "wrapped" else "direct" in
          let c, code = handed_out ("fo_bd_" ^ tag) in
          let live =
            List.map (List.range 0 4) ~f:(fun i ->
                let label = "fo_bd_" ^ tag ^ "_" ^ Int.to_string i in
                let live = replay_round ~wrapped ~label c code in
                a_closure ();
                live)
          in
          p_all
            ("bounded (" ^ tag ^ "): each replay finds the consumed earlier owners compacted away")
            live
            ~f:(fun n -> n = 1)))
