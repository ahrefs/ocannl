(* gh-ocannl-1526: one lowering of a routine, reused by the many compiles of an autotuning search.
   The contract is in lowering_reuse.mli. *)

open Base

type owner = {
  lineage : Low_level.optimize_ctx;
  asgns : Assignments.t;
  bindings : Indexing.unit_bindings;
  name : string option;
}

type t = { mutable filled : (owner * string * Low_level.optimized) option }

let created : (t -> unit) option Stdlib.Domain.DLS.key = Stdlib.Domain.DLS.new_key (fun () -> None)

let create () =
  let t = { filled = None } in
  Option.iter (Stdlib.Domain.DLS.get created) ~f:(fun observe -> observe t);
  t

let private_copy (o : Low_level.optimized) : Low_level.optimized =
  {
    o with
    traced_store = Low_level.copy_traced_store o.traced_store;
    optimize_ctx = Low_level.copy_optimize_ctx o.optimize_ctx;
  }

let reused_count = Atomic.make 0
let lowered_count = Atomic.make 0
let stats () = (Atomic.get reused_count, Atomic.get lowered_count)

let audit :
    (reused:Low_level.optimized -> fresh:Low_level.optimized -> unit) option Stdlib.Domain.DLS.key =
  Stdlib.Domain.DLS.new_key (fun () -> None)

let disabled = Stdlib.Domain.DLS.new_key (fun () -> false)

let with_dls key v f =
  let prior = Stdlib.Domain.DLS.get key in
  Stdlib.Domain.DLS.set key v;
  Exn.protect ~f ~finally:(fun () -> Stdlib.Domain.DLS.set key prior)

let with_audit check f = with_dls audit (Some check) f
let with_created_observer observe f = with_dls created (Some observe) f
let with_disabled f = with_dls disabled true f

let same_owner a b =
  phys_equal a.lineage b.lineage && phys_equal a.asgns b.asgns && phys_equal a.bindings b.bindings
  && Option.equal String.equal a.name b.name

let lower_or_reuse t ~lineage ~asgns ~bindings ~name ~(lower : unit -> string * Low_level.optimized)
    : string * Low_level.optimized =
  if Stdlib.Domain.DLS.get disabled then lower ()
  else
    let owner = { lineage; asgns; bindings; name } in
    match t.filled with
    | None ->
        let ((resolved, lowered) as result) = lower () in
        t.filled <- Some (owner, resolved, private_copy lowered);
        Atomic.incr lowered_count;
        result
    | Some (o, resolved, snapshot) ->
        if not (same_owner o owner) then
          invalid_arg
            "Lowering_reuse: a compile handed a reuse cell from another context, computation, \
             bindings or name (the cell belongs to the search that filled it)";
        Atomic.incr reused_count;
        let reused = private_copy snapshot in
        Option.iter (Stdlib.Domain.DLS.get audit) ~f:(fun check ->
            let _, fresh = lower () in
            check ~reused:(private_copy reused) ~fresh);
        (resolved, reused)

let reseed t (lowered : Low_level.optimized) =
  match t.filled with
  | None -> invalid_arg "Lowering_reuse.reseed: the cell was never filled"
  | Some (o, resolved, _) -> t.filled <- Some (o, resolved, private_copy lowered)

let snapshot t = Option.map t.filled ~f:(fun (_, _, s) -> s)
