(* gh-ocannl-1777, gh-ocannl-1877: what the compiler decides is a function of the program, not of
   how many nodes the session minted before it or of the order its nodes were built in.

   Four parts:

   1. A whole training step compiled as one routine (the memory_budget_planner subject), built after
   minting a different number of unrelated nodes each time. Every build then has the same session
   ids and the same code, but node uids shifted by a different amount -- so every Tnode-keyed hash
   table iterates in a different order (the control claim). Identical across the builds must be:

   - the surface (each candidate's default, alternatives, recompute costs and refusals) and the
   [Minimize] plan (its flips, their reliefs and costs, the final footprint). On master before
   gh-ocannl-1777 they were not: the transitive inline-fanin pass entered parameters, cutting the
   step's parameter -> gradient -> activation -> parameter cycles wherever its hash-ordered
   traversal started, so the fan-in a proxy-priced candidate's cost multiplies changed with the
   uids, and the planner's minimum with it (13 flips or 12, as gh-ocannl-1751 saw); - the scored
   baseline footprint and the plan for a mid-range [Bytes] budget, halfway between the baseline and
   the minimum (gh-ocannl-1877): a byte budget stops at the first prefix of the ranking that fits,
   so it reads the ranking's order where [Minimize] reads only its set; - what two compiles ship
   (gh-ocannl-1877) -- the recipes' untuned compile, and the compile from the mid-budget plan's
   context: the emitted source up to minted identifiers (the final schedule, the kernel's parameter
   and declaration order, the planned layout's offsets) and every program node's placement with its
   provenance, keyed by session id.

   2. Hand-built IR: one program, with its two equal-cost candidates' nodes created in either order.
   The surface lists them in program order both times (master: in uid order, so the creation order
   decided which came first).

   2b. Hand-built IR (gh-ocannl-1877): a cycle of covered reads through two nodes headed for
   inlining, after minting a different number of unrelated nodes each time. The transitive
   inline-fanin pass cuts the cycle where its traversal enters it; entered in program order, both
   nodes' fan-in and placement are the same every time (master: in the traced store's order, so one
   node was materialized by the fan-in cap or left virtual depending on the uids).

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
module Generated = Test_utils.Generated
open Verdict.Claims

(* The compiles' sources are read back. *)
let () = Utils.settings.output_debug_files_in_build_directory <- true

let () =
  Generated.init
    ~backend_name:(String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:""))

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
  baseline : LL.footprint;
  mid_plan : string list;  (** The plan for the budget halfway from [baseline] to [final]. *)
  mid_final : LL.footprint;
  untuned : string * string list;  (** The untuned compile: its source, its placements. *)
  budgeted : string * string list;  (** The compile from the mid-budget plan's context. *)
}

let flips_to_strings flips =
  List.map flips ~f:(fun (tn, relief, cost) ->
      Printf.sprintf "%d relief=%d cost=%d" tn.Tn.id relief cost)

(* What a compile shipped: the routine's emitted source up to the identifiers the compile minted,
   and the placement, with its provenance, the compiled context's lineage settled for each of
   [nodes]. *)
let shipped ~name ~nodes ((compiled : Context.t), (routine : Context.routine)) =
  (* A fissioned routine emits its segments into one [<name>__seg] source. *)
  let source = if List.length routine.segments > 1 then name ^ "__seg" else name in
  (* The nodes the schedule minted (staged tiles, moved accumulators): their ids come from a
     process-wide counter, so they differ from build to build and are compared by position in the
     source only. *)
  let minted =
    List.concat_map routine.segments ~f:(fun seg -> Hashtbl.keys seg.LL.traced_store)
    |> List.filter ~f:(fun tn -> not (List.mem nodes tn ~equal:Tn.equal))
    |> List.dedup_and_sort ~compare:Tn.compare
  in
  eprintf "%s (not part of the golden): %d segment(s), schedule-minted nodes: %s\n%!" name
    (List.length routine.segments)
    (String.concat ~sep:" " (List.map minted ~f:Tn.debug_name));
  ( Generated.normalize_minted ~facts:(Context.backend_facts compiled) (Generated.read source),
    List.map nodes ~f:(fun tn ->
        Printf.sprintf "%d %s" tn.Tn.id
          (Tn.debug_memory_mode (Tn.Placements.get (Context.placements compiled) tn))) )

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
  let hash_order = List.map (Hashtbl.keys lowered.LL.traced_store) ~f:(fun tn -> tn.Tn.id) in
  (* The program's nodes in session-id order: the same list in every build. *)
  let nodes =
    List.sort (Hashtbl.keys lowered.LL.traced_store) ~compare:(fun a b ->
        Int.compare a.Tn.id b.Tn.id)
  in
  let _ctx, plan = Memory_budget.fit ~budget:Memory_budget.Minimize ctx comp IDX.empty in
  let baseline = Memory_budget.footprint ctx comp IDX.empty in
  let mid = (baseline.fp_total + plan.bp_final.fp_total) / 2 in
  let planned, mid_plan = Memory_budget.fit ~budget:(Memory_budget.Bytes mid) ctx comp IDX.empty in
  (* As [Train.to_routine] compiles under a budget: the default pipeline, from the plan's
     context. *)
  let ship ~name compile =
    Generated.arm name;
    Generated.arm (name ^ "__seg");
    let ((compiled, _) as result) = compile () in
    let shipped = shipped ~name ~nodes result in
    Context.release compiled;
    shipped
  in
  let budgeted =
    ship ~name:"fom_budgeted" (fun () ->
        Context.compile ~name:"fom_budgeted" planned comp IDX.empty)
  in
  (* As [Train.to_routine] compiles without one. *)
  let untuned =
    ship ~name:"fom_untuned" (fun () ->
        Autotune.untuned_compile ~name:"fom_untuned" ctx comp IDX.empty)
  in
  {
    hash_order;
    surface;
    plan = flips_to_strings plan.bp_flips;
    final = plan.bp_final.fp_total;
    baseline;
    mid_plan = flips_to_strings mid_plan.bp_flips;
    mid_final = mid_plan.bp_final;
    untuned;
    budgeted;
  }

let () =
  let mints = [ 0; 1; 2; 3; 7 ] in
  let outcomes = List.map mints ~f:(fun mint -> (mint, observe ~mint)) in
  let _, first = List.hd_exn outcomes in
  let rest = List.tl_exn outcomes in
  List.iter outcomes ~f:(fun (mint, o) ->
      eprintf
        "mint %d (not part of the golden): %d candidates, baseline %d bytes, minimum %d bytes with \
         %d flips, mid-budget %d bytes with %d flips\n\
         %!"
        mint (List.length o.surface) o.baseline.fp_total o.final (List.length o.plan)
        o.mid_final.fp_total (List.length o.mid_plan));
  (* The control: without it, equal outcomes could mean the minting moved nothing. *)
  p_exists "control: the minting changes a node-keyed table's iteration order" rest
    ~f:(fun (_, o) -> not (List.equal Int.equal o.hash_order first.hash_order));
  p "the step offers flips and the planner takes some"
    ((not (List.is_empty first.surface)) && not (List.is_empty first.plan));
  p_all "the flip surface is the same whatever was minted before" rest ~f:(fun (_, o) ->
      List.equal String.equal o.surface first.surface);
  p_all "the planner's minimum takes the same flips whatever was minted before" rest
    ~f:(fun (_, o) -> List.equal String.equal o.plan first.plan && o.final = first.final);
  p_all "the scored baseline footprint is the same whatever was minted before" rest
    ~f:(fun (_, o) -> LL.equal_footprint o.baseline first.baseline);
  (* Strictly between the ends and short of the minimum's flips, so the plan is a proper prefix of a
     ranking: which candidates come first decides it. *)
  p "control: the mid-range budget takes some of the minimum's flips, not all"
    ((not (List.is_empty first.mid_plan))
    && List.length first.mid_plan < List.length first.plan
    && first.mid_final.fp_total > first.final
    && first.mid_final.fp_total < first.baseline.fp_total);
  p_all "the mid-range budget takes the same flips whatever was minted before" rest
    ~f:(fun (_, o) ->
      List.equal String.equal o.mid_plan first.mid_plan
      && LL.equal_footprint o.mid_final first.mid_final);
  let same_shipped what (pick : outcome -> string * string list) =
    let text0, placements0 = pick first in
    List.iter rest ~f:(fun (mint, o) ->
        let text, placements = pick o in
        if not (String.equal text text0) then
          eprintf
            "the %s compile's source after minting %d differs from minting 0's:\n%s\n----\n%s\n"
            what mint text0 text;
        List.iter2_exn placements0 placements ~f:(fun p0 p ->
            if not (String.equal p0 p) then
              eprintf "the %s compile's placement after minting %d: %s, minting 0's: %s\n" what mint
                p p0));
    p_all (Printf.sprintf "the %s compile emits the same source whatever was minted before" what)
      rest ~f:(fun (_, o) -> String.equal (fst (pick o)) text0);
    p_all
      (Printf.sprintf "the %s compile places every node the same whatever was minted before" what)
      rest ~f:(fun (_, o) -> List.equal String.equal (snd (pick o)) placements0)
  in
  same_shipped "untuned" (fun o -> o.untuned);
  same_shipped "budgeted" (fun o -> o.budgeted);
  (* The two compiles are not one decision observed twice. *)
  p "control: the budgeted compile's placements differ from the untuned compile's"
    (not (List.equal String.equal (snd first.untuned) (snd first.budgeted)))

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

(* === Part 2b: a covered cycle through nodes headed for inlining (gh-ocannl-1877) === *)

(* [a]'s first half sums [n_x] inputs, [b] adds [n_y] more to it, and [a]'s second half reads [b]:
   every read is covered, so neither node is materialized by the coverage rule, and the setter-reads
   graph has the cycle a -> b -> a. The fan-in pass cuts it where its traversal enters it. Entered
   at [a], the node the code touches first, [b]'s set is [a] itself and the [n_y] inputs: within the
   cap. Entered at [b] it is all [n_x + n_y] inputs: over the cap, and [b] is materialized. On
   master the traversal followed the traced store's iteration order, so the unrelated nodes minted
   before the program -- [mint] of them -- decided which. *)
let cycle ~mint =
  let cap = LL.virtualize_settings.max_inline_fanin in
  let n_x = (cap / 2) + 1 in
  let n_y = cap - n_x + 1 in
  let mk = Ll_test.node_factory ~first_id:(18770 + (100 * mint)) ~dims:[| 8 |] () in
  for _ = 1 to mint do
    ignore (mk "unrelated" : Tn.t)
  done;
  let inputs prefix n = List.init n ~f:(fun k -> mk (prefix ^ Int.to_string k)) in
  let xs = inputs "x" n_x and ys = inputs "y" n_y in
  let out = mk "out" and out2 = mk ~dims:[| 4 |] "out2" in
  let a = mk "a" in
  let b = mk ~dims:[| 4 |] "b" in
  List.iter (xs @ ys @ [ out; out2 ]) ~f:Ll_test.materialize;
  let open Ll_test in
  let over n f =
    let s = sym () in
    loop_n s n (f s)
  in
  let sum_at i tns = List.map tns ~f:(fun tn -> get tn [| iter i |]) |> List.reduce_exn ~f:add in
  let llc =
    List.reduce_exn ~f:seq
      [
        over 4 (fun i -> set_at a (iter i) (sum_at i xs));
        over 4 (fun i -> set_at b (iter i) (add (get a [| iter i |]) (sum_at i ys)));
        over 4 (fun i -> set_at a (aff [ (1, i) ] 4) (add (get b [| iter i |]) (c 1.)));
        over 8 (fun i -> set_at out (iter i) (get a [| iter i |]));
        over 4 (fun i -> set_at out2 (iter i) (get b [| iter i |]));
      ]
  in
  let o = optimize ~name:(Printf.sprintf "fo_cycle_%d" mint) llc in
  let decided tn =
    ( (Hashtbl.find_exn o.LL.traced_store tn).LL.inline_fanin,
      Tn.debug_memory_mode (Tn.Placements.get o.LL.optimize_ctx.LL.placements tn) )
  in
  let a_ahead =
    List.find_map_exn (Hashtbl.keys o.LL.traced_store) ~f:(fun tn ->
        if Tn.equal tn a then Some true else if Tn.equal tn b then Some false else None)
  in
  eprintf "cycle, mint %d (not part of the golden): a %d %s, b %d %s, the table has %s first\n%!"
    mint
    (fst (decided a))
    (snd (decided a))
    (fst (decided b))
    (snd (decided b))
    (if a_ahead then "a" else "b");
  (a_ahead, decided a, decided b, (n_x, n_y))

let () =
  let builds = List.map [ 0; 1; 2; 3; 4; 5; 6; 7 ] ~f:(fun mint -> cycle ~mint) in
  let _, a0, b0, (n_x, n_y) = List.hd_exn builds in
  (* The minting does reorder a table over the two nodes. The lowered store is not the table the
     pass walks (that is the analysis' own copy), so this shows the shift reaching node-keyed
     tables, not which node the pass would have entered first. *)
  p "control: the minting changes which of a and b the lowered store iterates first"
    (List.exists builds ~f:(fun (a_ahead, _, _, _) -> a_ahead)
    && List.exists builds ~f:(fun (a_ahead, _, _, _) -> not a_ahead));
  let same (f1, m1) (f2, m2) = f1 = f2 && String.equal m1 m2 in
  p_all "cycle: both nodes' fan-in and placement are the same whatever was minted before"
    (List.tl_exn builds) ~f:(fun (_, a, b, _) -> same a a0 && same b b0);
  p "cycle: it is entered at the node the code touches first, so b counts a once"
    (fst a0 = n_x && fst b0 = 1 + n_y)

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
