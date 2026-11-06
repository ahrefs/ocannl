(* gh-ocannl-1859: what a long session keeps beside its contexts. A loop that builds a fresh graph,
   runs it and releases what it ran gets its buffers back (dropped_graph_buffers); this test follows
   the OCaml heap and the session's own tables across such a loop.

   The solver's per-variable tables ([Row.solver_census]) are read on either side of a closure: each
   must have held something, so that "empty" is a claim about these graphs and not about tables
   nothing wrote, and a closure must leave them all empty. They were about 96% of the live words a
   cycle added before they were retired. A failed closure commits nothing, so it keeps them, and a
   rewind to the closure gives the empty tables back. A graph built over a closed one still infers
   and computes. The retired-read guard is shown refusing a read, or not, by the setting that
   enables it.

   The loop's heap claim is a slope in live words, after the settling [Gc_witness.settle] does, over
   the last window of cycles, so the first compiles' one-time structures are not in it. The digits
   go to stderr. Neither [Tensor.unsafe_reinitialize] nor [Rewrites.reset] runs between cycles:
   either would hide what the loop is about.

   The rewrite tier keeps what it knows about a node keyed by that node ([Rewrites.retained]). Each
   member is first shown to fire on its graph: its tables grow while the graph is held. Fresh graphs
   then plateau with the bounded analysis cache's window of nodes, and drain to the baseline once
   that cache is cleared. A graph held across the loop keeps its minted nodes' identity through
   collections, its markers through a memo reset, and computes as before. *)
open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims
module IDX = Train.IDX
module Census = Ir.Alloc_census
module LL = Ir.Low_level
module Tn = Ir.Tnode

let settle () =
  (* The bounded analysis cache keeps the last eight compiles' nodes by design. *)
  Ir.Low_level.clear_analysis_cache ();
  Gc_witness.settle ()

let live_words () = (Stdlib.Gc.stat ()).live_words

let leaf () =
  Tensor.term_init [| 1.; 2.; 3.; 4. |] ~label:[ "sg_x" ] ~output_dims:[ 4 ]
    ~grad_spec:Prohibit_grad ()

(* A graph writing every table: a parameter (terminal rows), einsum and pointwise row relations, a
   concatenated axis with a captured component. *)
let[@inline never] build () =
  let x = leaf () in
  let%op part = x ++ "2^o => o" in
  let%op h = ({ sg_w; o = [ 8 ] } * part) + { sg_b; o = [ 8 ] } in
  let%op y = relu h in
  y

let[@inline never] cycle ~update () =
  let y = build () in
  let open_census = Row.solver_census () in
  let root = Context.auto () in
  let ctx =
    if update then
      let%op loss = y ++ "...|... => 0" in
      Train.update_once root loss
    else Train.forward_once root y
  in
  let closed_census = Row.solver_census () in
  Context.release ctx;
  Context.release root;
  (open_census, closed_census)

let sizes census = List.map census ~f:snd
let cycles_warm = 30
let cycles_read = 40

let () =
  (* The tables a cycle wrote, summed by name, and what the closures left behind them. *)
  let written = Hashtbl.create (module String) in
  let left = ref [] in
  let run n =
    for i = 1 to n do
      let opened, closed = cycle ~update:(i % 2 = 0) () in
      List.iter opened ~f:(fun (name, size) -> Hashtbl.incr written name ~by:size);
      left := sizes closed @ !left
    done
  in
  run cycles_warm;
  settle ();
  let before = live_words () in
  run cycles_read;
  settle ();
  let after = live_words () in
  let slope = (after - before) / cycles_read in
  Stdio.eprintf "live words per cycle: %d, %d to %d (not part of the golden)\n%!" slope before after;
  (* Before retirement a cycle added 1640 to 2000 words. The bound is a fifth of that, well above
     what a settled cycle leaves. *)
  p "a cycle leaves under 400 live words" (slope < 400);
  p "the live-word slope is not an artifact of a shrinking heap" (slope > -400);
  p_all "the open work wrote each solver table but the role claims"
    (List.filter (Row.solver_census ()) ~f:(fun (name, _) ->
         not (String.equal name "pending_roles")))
    ~f:(fun (name, _) -> Hashtbl.find written name |> Option.exists ~f:(fun n -> n > 0));
  p_all "every closure left every solver table empty" !left ~f:(fun n -> n = 0);
  (* A failed closure is not a commit. *)
  let x = leaf () in
  let%op unsized = x ++ "a^o => o" in
  ignore unsized;
  let open_tables = Row.solver_census () in
  let failed =
    match Train.forward_once (Context.auto ()) unsized with
    | exception Row.Shape_error _ -> true
    | exception _ -> false
    | _ -> false
  in
  p "an unsized concatenation component fails its closure" failed;
  p "a failed closure keeps the open work's tables"
    (List.exists (sizes open_tables) ~f:(fun n -> n > 0)
    && List.equal Int.equal (sizes (Row.solver_census ())) (sizes open_tables));
  Tensor.rewind_to_last_closure ();
  p_all "a rewind to the closure restores the empty tables"
    (sizes (Row.solver_census ()))
    ~f:(fun n -> n = 0);
  (* A closed graph stays usable as an operand of later open work. *)
  let root = Context.auto () in
  let held = build () in
  let ctx = Train.forward_once root held in
  let held_values = Context.get_values ctx held.value in
  let%op over = held *. 2. in
  let ctx = Train.forward_once ctx over in
  p_all "a graph over a closed one computes"
    (List.zip_exn (Array.to_list held_values) (Array.to_list (Context.get_values ctx over.value)))
    ~f:(fun (h, o) -> Float.equal o (2. *. h));
  (* The guard: a variable from before a closure, read after it. *)
  let stale = Row.get_row_var () in
  let trip =
    Tensor.term_init [| 1. |] ~label:[ "sg_trip" ] ~output_dims:[ 1 ] ~grad_spec:Prohibit_grad ()
  in
  let ctx = Train.forward_once ctx trip in
  let refused = match Row.is_resolve_at_use stale with _ -> false | exception Failure _ -> true in
  p "the retired-read guard refuses exactly when it is enabled"
    (Bool.equal refused
       (Utils.get_global_flag ~default:false ~arg_name:"debug_check_retired_solver_vars"));
  Context.release ctx;
  Context.release root

(* {1 The rewrite tier} *)

type member = {
  name : string;  (** The tier's name for the member. *)
  set_enabled : bool option -> unit;
  build : unit -> Tensor.t;
  role : (Tn.t -> bool) option;
      (** Whether the member marked the node as one it minted, where a forward lowering mints one
          (the softmax's marker is the fused backward's: its own claims below). *)
  candidates : LL.t -> Tn.t list;  (** Where the rewritten code holds such nodes. *)
}

let written llc =
  LL.affine_accesses llc
  |> List.filter_map ~f:(fun (a : Tn.t Ir.Affine.access) -> Option.some_if a.a_write a.a_tn)

let scan_states llc =
  let states = ref [] in
  Ll_test.walk llc ~on_stmt:(function
    | LL.Scan_loop { carried; _ } ->
        List.iter carried ~f:(fun c -> states := c.LL.prev.tn :: !states)
    | _ -> ());
  !states

let image () =
  TDSL.range_of_shape ~label:[ "sg_img" ] ~batch_dims:[ 1 ] ~input_dims:[] ~output_dims:[ 6; 6; 2 ]
    ()

let scan_input () =
  Tensor.term_init
    (Array.init 8 ~f:(fun k -> Float.of_int ((k * 7 % 11) - 4)))
    ~label:[ "sg_scan" ] ~grad_spec:Prohibit_grad ~output_dims:[ 8 ] ()

let triangle () =
  let ri = TDSL.range 8 and co = TDSL.range 8 in
  let%op tri = not (co ++ "j=>0j" < ri ++ "i=>i0") in
  tri

let attention () =
  let x =
    TDSL.range_of_shape ~label:[ "sg_seq" ] ~batch_dims:[ 1; 5 ] ~input_dims:[] ~output_dims:[ 8 ]
      ()
  in
  let%op x = x /. 224. in
  let mask =
    NTDSL.init ~l:"sg_mask" ~prec:Ir.Ops.single ~b:[ 5 ] ~i:[ 5 ] ~o:[]
      ~f:(function [| s; t |] -> if s >= t then 1. else 0. | _ -> assert false)
      ()
  in
  let block = Nn_blocks.multi_head_attention ~label:[ "sg_attn" ] ~num_heads:2 ~d_k:4 ~d_v:4 () in
  let%op y = x + block ~train_step:None ~mask x in
  y

let members =
  [
    {
      name = "online_softmax";
      set_enabled = Ir.Online_softmax.set_enabled;
      build = attention;
      role = None;
      candidates = (fun _ -> []);
    };
    {
      name = "winograd_conv";
      set_enabled = Ir.Winograd.set_enabled;
      build =
        (fun () ->
          let x = image () in
          let%op y =
            relu (Nn_blocks.conv2d ~label:[ "sg_conv" ] ~use_padding:true ~out_channels:3 () x)
          in
          y);
      role = Some (fun tn -> Option.is_some (Ir.Winograd.minted_role tn));
      candidates = written;
    };
    {
      name = "prefix_scan";
      set_enabled = Ir.Prefix_scan.set_enabled;
      build =
        (fun () ->
          let x = scan_input () and tri = triangle () in
          let%op y = x +* "i; i o => o" tri in
          y);
      role = Some (fun tn -> Option.is_some (Ir.Prefix_scan.state_owner tn));
      candidates = scan_states;
    };
  ]

let retained (m : member) = List.Assoc.find_exn (Ir.Rewrites.retained ()) m.name ~equal:String.equal

(* The nodes the enabled tier's rewrite of [t]'s raw code holds and marks: the raw form, since the
   optimizer inlines a virtual state away. *)
let minted_in (m : member) role (t : Tensor.t) =
  Ir.Assignments.to_low_level t.Tensor.forward.Ir.Assignments.asgns
  |> Ir.Rewrites.apply |> m.candidates |> List.filter ~f:role
  |> List.dedup_and_sort ~compare:Tn.compare

let[@inline never] rewrite_cycle (m : member) =
  let y = m.build () in
  let root = Context.auto () in
  let ctx = Train.forward_once root y in
  Context.release ctx;
  Context.release root

type held = {
  graph : Tensor.t;
  held_ctx : Context.t;
  routine : Context.routine;
  values : float array;
}

let hold (m : member) =
  let graph = m.build () in
  Train.set_materialized graph.value;
  let root = Context.auto () in
  let c1 = Train.init_params root IDX.empty graph in
  let held_ctx, routine = Train.to_routine c1 IDX.empty (Train.forward graph) in
  let held_ctx = Context.run held_ctx routine in
  { graph; held_ctx; routine; values = Context.get_values held_ctx graph.value }

let cycles_natural = 12

let () =
  List.iter members ~f:(fun m ->
      m.set_enabled (Some true);
      settle ();
      let baseline = retained m in
      let holding = hold m in
      let minted = Option.map m.role ~f:(fun role -> (role, minted_in m role holding.graph)) in
      let with_held = retained m in
      p (m.name ^ ": it fired on its graph") (with_held > baseline);
      let run n =
        for _ = 1 to n do
          rewrite_cycle m
        done
      in
      (* The natural steady state: no cache is cleared between these two readings. *)
      run cycles_natural;
      Gc_witness.settle ();
      let plateau = retained m in
      run 4;
      Gc_witness.settle ();
      Stdio.eprintf "%s: retained %d, %d held, plateau %d then %d (not part of the golden)\n%!"
        m.name baseline with_held plateau (retained m);
      p (m.name ^ ": fresh graphs plateau with the analysis cache") (retained m = plateau);
      (* One more graph's writes prune the last one's writer events, which hold its nodes. *)
      m.set_enabled (Some false);
      rewrite_cycle m;
      m.set_enabled (Some true);
      settle ();
      Stdio.eprintf "%s: retained %d with the cache cleared (not part of the golden)\n%!" m.name
        (retained m);
      p (m.name ^ ": dropped graphs' entries are gone once that cache is") (retained m = with_held);
      Option.iter minted ~f:(fun (role, nodes) ->
          p
            (m.name ^ ": a held graph keeps its minted nodes' identity")
            (List.equal Tn.equal (minted_in m role holding.graph) nodes);
          Ir.Rewrites.reset ();
          settle ();
          p_all
            (m.name ^ ": a held graph's nodes keep their markers through a memo reset")
            nodes ~f:role);
      let ctx = Context.run holding.held_ctx holding.routine in
      p_all2
        (m.name ^ ": the held graph still computes")
        (Context.get_values ctx holding.graph.value)
        holding.values ~f:Float.equal;
      m.set_enabled None)

(* A marker no graph above reaches: the fused backward's licence. It stays on a held node, and goes
   with a dropped one. *)
let[@inline never] dprob ~like = Ir.Online_softmax.dprob_local ~like Ir.Ops.single

let () =
  Ir.Online_softmax.set_backward_enabled (Some true);
  let held_like = (leaf ()).value in
  let held = dprob ~like:held_like in
  let witness =
    let[@inline never] drop () = Gc_witness.witness (dprob ~like:(leaf ()).value) in
    drop ()
  in
  Ir.Rewrites.reset ();
  settle ();
  p "a held node keeps its reassociation licence through a memo reset"
    (Ir.Online_softmax.reassociable_local held);
  p "a dropped node's marker goes with it" (Gc_witness.reclaimed witness);
  Ir.Online_softmax.set_backward_enabled None
