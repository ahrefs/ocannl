(* gh-ocannl-1777: the flip surface and the memory-budget planner's choice are a function of the
   program, not of how many nodes the session minted before it or of the order its nodes were built
   in.

   Three parts:

   1. A whole training step compiled as one routine (the memory_budget_planner subject), built after
   minting a different number of unrelated nodes each time. Every build then has the same session
   ids and the same code, but node uids shifted by a different amount -- so every Tnode-keyed hash
   table iterates in a different order (the control claim). The surface (each candidate's default,
   alternatives, recompute costs and refusals) and the [Minimize] plan (its flips, their reliefs and
   costs, the final footprint) must be identical across the builds. On master they were not: the
   transitive inline-fanin pass entered parameters, cutting the step's parameter -> gradient ->
   activation -> parameter cycles wherever its hash-ordered traversal started, so the fan-in a
   proxy-priced candidate's cost multiplies changed with the uids, and the planner's minimum with it
   (13 flips or 12, as gh-ocannl-1751 saw).

   2. Hand-built IR: one program, with its two equal-cost candidates' nodes created in either order.
   The surface lists them in program order both times (master: in uid order, so the creation order
   decided which came first).

   3. [Autotune.rank_flip_candidates] keeps its input's order among equal ranks, rather than
   re-sorting ties by uid. *)

open Base
open Stdio
open Ocannl
module IDX = Train.IDX
open Nn_blocks.DSL_modules
module Tn = Ir.Tnode
module Asgns = Ir.Assignments
module LL = Ir.Low_level
open Verdict.Claims

(* The planner's precondition, and no ambient budget. *)
let () = Unix.putenv "OCANNL_MEMORY_BUDGET" "0"
let () = Unix.putenv "OCANNL_BUFFER_ALIASING" "true"

(* === Part 1: unrelated minting === *)

(* The memory_budget_planner step, after [mint] unrelated nodes. They are minted BEFORE the
   reinitialization, so session ids restart at the same place and only uids move. *)
let build ~mint () =
  for _ = 1 to mint do
    ignore (NTDSL.O.(!.1.0) : Tensor.t)
  done;
  Utils.settings.fixed_state_for_init <- Some 3;
  Tensor.unsafe_reinitialize ();
  TDSL.default_param_init := NTDSL.xavier ~scale_sq:2.0 TDSL.O.uniform1;
  let ctx = Context.auto () in
  let batch = 32 and d_in = 64 and d_hid = 256 in
  let xs =
    NTDSL.init ~l:"xs" ~prec:Ir.Ops.single ~b:[ batch ] ~o:[ d_in ]
      ~f:(fun idcs -> Float.sin (Float.of_int ((idcs.(0) * d_in) + idcs.(1))))
      ()
  in
  let ys =
    NTDSL.init ~l:"ys" ~prec:Ir.Ops.single ~b:[ batch ] ~o:[ 1 ]
      ~f:(fun idcs -> Float.cos (Float.of_int idcs.(0)))
      ()
  in
  let%op mlp x =
    { w4 }
    * relu
        ({ b3; o = [ d_hid ] }
        + { w3 }
          * relu ({ b2; o = [ d_hid ] } + ({ w2 } * relu ({ b1; o = [ d_hid ] } + ({ w1 } * x)))))
  in
  let%op err = mlp xs - ys in
  let%op scalar_loss = ((err *. err) ++ "...|... => |->0") /. !..batch in
  let update = Train.grad_update scalar_loss in
  let sgd = Train.sgd_update ~learning_rate:(TDSL.O.( !. ) 1e-6) scalar_loss in
  let comp = Asgns.sequence [ update; sgd ] in
  let ctx = Train.init_params ctx IDX.empty scalar_loss in
  (ctx, comp)

(* Everything a build decides, keyed by session id (equal across the builds by construction). *)
type outcome = {
  hash_order : int list;  (** The traced store's iteration order: the control. *)
  surface : string list;
  plan : string list;
  final : int;
}

let observe ~mint =
  let ctx, comp = build ~mint () in
  let lowered = Context.lowered_for_decisions ctx comp IDX.empty in
  let surface =
    List.map lowered.LL.flip_candidates ~f:(fun fc ->
        Printf.sprintf "%d %s -> %s" fc.LL.fc_tn.Tn.id
          (LL.reading_to_string fc.LL.fc_default)
          (String.concat ~sep:" "
             (List.map fc.LL.fc_alternatives ~f:(fun fa ->
                  Printf.sprintf "%s=%d%s%s"
                    (LL.reading_to_string fa.LL.fa_flip)
                    fa.LL.fa_recompute_cost
                    (if fa.LL.fa_modeled then "" else "~")
                    (Option.value_map fa.LL.fa_refused ~default:"" ~f:(fun c -> "!" ^ c))))))
  in
  let _ctx, plan = Memory_budget.fit ~budget:Memory_budget.Minimize ctx comp IDX.empty in
  {
    hash_order = List.map (Hashtbl.keys lowered.LL.traced_store) ~f:(fun tn -> tn.Tn.id);
    surface;
    plan =
      List.map plan.bp_flips ~f:(fun (tn, relief, cost) ->
          Printf.sprintf "%d relief=%d cost=%d" tn.Tn.id relief cost);
    final = plan.bp_final.fp_total;
  }

let () =
  let mints = [ 0; 1; 2; 3; 7 ] in
  let outcomes = List.map mints ~f:(fun mint -> (mint, observe ~mint)) in
  let _, first = List.hd_exn outcomes in
  let rest = List.tl_exn outcomes in
  List.iter outcomes ~f:(fun (mint, o) ->
      eprintf "mint %d (not part of the golden): %d candidates, minimum %d bytes with %d flips\n%!"
        mint (List.length o.surface) o.final (List.length o.plan));
  (* The control: without it, equal outcomes could mean the minting moved nothing. *)
  p_exists "control: the minting changes a node-keyed table's iteration order" rest
    ~f:(fun (_, o) -> not (List.equal Int.equal o.hash_order first.hash_order));
  p "the step offers flips and the planner takes some"
    ((not (List.is_empty first.surface)) && not (List.is_empty first.plan));
  p_all "the flip surface is the same whatever was minted before" rest ~f:(fun (_, o) ->
      List.equal String.equal o.surface first.surface);
  p_all "the planner's minimum takes the same flips whatever was minted before" rest
    ~f:(fun (_, o) -> List.equal String.equal o.plan first.plan && o.final = first.final)

(* === Part 2: construction order, hand-built IR === *)

(* [a = x + 1; b = y + 1; out = a + b]: [a] and [b] are left virtual, each offering a [`Materialize]
   flip at the same cost. [b_first] creates [b]'s node before [a]'s, with unrelated nodes minted in
   between; the program is the same either way. *)
let tie_surface ~b_first =
  let mk = Ll_test.node_factory ~first_id:17770 ~dims:[| 8 |] () in
  let x = mk "x" and y = mk "y" and out = mk "out" in
  let a, b =
    if b_first then (
      let b = mk "b" in
      for _ = 1 to 5 do
        ignore (mk "unrelated" : Tn.t)
      done;
      let a = mk "a" in
      (a, b))
    else
      let a = mk "a" in
      let b = mk "b" in
      (a, b)
  in
  List.iter [ x; y; out ] ~f:Ll_test.materialize;
  let pointwise f =
    let s = Ll_test.sym () in
    Ll_test.loop_n s 8 (f (Ll_test.iter s))
  in
  let open Ll_test in
  let llc =
    List.reduce_exn ~f:seq
      [
        pointwise (fun i -> set_at a i (add (get x [| i |]) (c 1.)));
        pointwise (fun i -> set_at b i (add (get y [| i |]) (c 1.)));
        pointwise (fun i -> set_at out i (add (get a [| i |]) (get b [| i |])));
      ]
  in
  let o = optimize ~name:(if b_first then "fo_tie_b_first" else "fo_tie_a_first") llc in
  ( (a, b),
    List.filter_map o.LL.flip_candidates ~f:(fun fc ->
        if Tn.equal fc.LL.fc_tn a || Tn.equal fc.LL.fc_tn b then
          Some
            ( String.concat fc.LL.fc_tn.Tn.label,
              (List.hd_exn fc.LL.fc_alternatives).fa_recompute_cost )
        else None) )

let () =
  let (a, b), a_first = tie_surface ~b_first:false in
  let (a', b'), b_first = tie_surface ~b_first:true in
  p "control: the creation orders differ" (Tn.compare a b < 0 && Tn.compare b' a' < 0);
  let costs l = List.map l ~f:snd in
  p "hand-built: both nodes are candidates, at one cost"
    (match costs a_first @ costs b_first with
    | [ c1; c2; c3; c4 ] -> c1 = c2 && c2 = c3 && c3 = c4
    | _ -> false);
  let labels l = List.map l ~f:fst in
  p "hand-built: equal-cost candidates come in program order, built a first"
    (List.equal String.equal (labels a_first) [ "a"; "b" ]);
  p "hand-built: equal-cost candidates come in program order, built b first"
    (List.equal String.equal (labels b_first) [ "a"; "b" ])

(* === Part 3: the autotuner's ranking keeps the surface's order among ties === *)

let () =
  let mk = Ll_test.node_factory ~first_id:17790 ~dims:[| 8 |] () in
  (* [late] is minted first, so a uid tie-break would put it ahead. *)
  let late = mk "late" and early = mk "early" in
  let candidate tn =
    {
      LL.fc_tn = tn;
      fc_default = `Inline;
      fc_alternatives =
        [
          {
            LL.fa_flip = `Materialize;
            fa_recompute_cost = 64;
            fa_modeled = true;
            fa_refused = None;
          };
        ];
    }
  in
  let ranked input =
    Autotune.rank_flip_candidates ~ordering:`Cost
      ~enablement:(Set.empty (module Tn))
      ~disablement:(Set.empty (module Tn))
      (List.map input ~f:candidate)
    |> List.map ~f:(fun fc -> String.concat fc.LL.fc_tn.Tn.label)
  in
  p "ranking: equal ranks keep the input order"
    (List.equal String.equal (ranked [ early; late ]) [ "early"; "late" ]
    && List.equal String.equal (ranked [ late; early ]) [ "late"; "early" ])
