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
   either would hide what the loop is about. *)
open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims
module IDX = Train.IDX
module Census = Ir.Alloc_census

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
