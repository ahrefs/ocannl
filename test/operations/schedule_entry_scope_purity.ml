(* gh-ocannl-2314: a schedule's entry program is checked for scope purity once, before any op runs.

   Scope purity (gh-ocannl-584) bans tensor writes in a [Local_scope] body, and several schedule
   ops' scanners rely on it: they read writes at statement level and never look inside a scalar.
   [Low_level.optimize] validates purity on the way in, and every op is refused when it leaves an
   impure body its input did not hold. A hand-built [?prelowered] routine skips the first gate, and
   the per-op check exempts a violation already present before the op, so such a routine used to
   reach every op of its schedule with the hidden write intact, to be refused only at codegen's exit
   gate after the ops had misread it. [Schedule.apply_typed] and [Schedule.schedule_legality] now
   refuse an impure entry, naming it, before the first op runs.

   The gap: [r { i { sink[i] = scope { acc[i] += x[r, i]; local := 1 } } }], scheduled [Privatize
   acc over r; Retype i Grid]. [Privatize]'s scan descends the scope body and remaps the write to
   its tile; the per-thread-lifetime check of the later [Retype] reads tile writes at statement
   level only, so it does not see that the [Grid] loop splits the tile's accumulation across
   threads. Its statement-level twin is refused at the [Retype]. The base applied the impure chain.

   The claims. The impure entry is refused by [Schedule.apply], naming the entry and the [Set] of
   [acc], with no op dispatched (the [with_transform_fault] seam counts dispatches), and by
   [Schedule.schedule_legality], which returns no verdict. The refusal is untyped -- the routine's
   fault, not a candidate's: under [Schedule_outcome.protect] it is fatal when strict and an
   [Unclassified] decline when permissive, never an [Illegal_schedule]. The controls: the
   statement-level twin reaches its ops and is refused at the [Retype], naming the tile's lifetime;
   a pure entry holding a pure scope body runs every op and computes the unscheduled values; the
   empty schedule is the identity, even on the impure entry; and a pure entry holding a [Grid] loop
   inside a scope body (the other half of the scope grammar, which an op can repair) is admitted,
   its [Retype] to [Serial] applied. *)

open Base
open Ocannl.Operation.DSL_modules
open Ll_test
open Verdict.Claims
module Sched = Ir.Schedule
module Outcome = Ir.Schedule_outcome

let node = node_factory ~first_id:2314000 ~dims:[| 2 |] ()

type case = {
  opt : LL.optimized;
  acc : Tn.t;
  x : Tn.t;
  r : Ir.Indexing.symbol;
  i : Ir.Indexing.symbol;
}

(* [~hidden]: the update sits in the scope body; otherwise it is a statement of its own, beside a
   pure scope body. *)
let program tag ~hidden =
  let acc = node ("acc" ^ tag) and x = node ~dims:[| 2; 2 |] ("x" ^ tag) in
  let sink = node ("sink" ^ tag) and v = node ("v" ^ tag) in
  virtualize v;
  List.iter [ acc; x; sink ] ~f:materialize;
  let r = sym () and i = sym () in
  let id = LL.get_scope v in
  let update = set acc [| iter i |] (add (get acc [| iter i |]) (get x [| iter r; iter i |])) in
  let local = LL.Set_local (id, c 1.) in
  let body ~scoped =
    match (hidden, scoped) with
    | true, true -> set sink [| iter i |] (scope_of id (seq update local))
    | false, true -> seq update (set sink [| iter i |] (scope_of id local))
    | _, false -> seq update (set sink [| iter i |] (c 1.))
  in
  let build ~scoped = loop_n r 2 (loop_n i 2 (body ~scoped)) in
  let opt =
    optimize_scoped ~name:("se2314_" ^ tag) ~raw:(build ~scoped:false) (build ~scoped:true)
  in
  { opt; acc; x; r; i }

let pz c = Sched.privatize ~accum_prec:Fn.id ~target:c.acc ~over:c.r
let gap c = [ pz c; Sched.Retype { axis = c.i; ty = LL.Grid } ]

(* [f ()] with the number of ops it dispatched. *)
let counting f =
  let n = ref 0 in
  let result = Sched.with_transform_fault (fun (_ : Sched.optop) -> Int.incr n) f in
  (result, !n)

let names_entry ~what ~acc msg =
  Stdio.eprintf "%s: %s (not part of the golden)\n%!" what msg;
  String.is_prefix msg ~prefix:"Schedule.apply: the program entering the schedule"
  && String.is_substring msg ~substring:"impure scope body"
  && String.is_substring msg ~substring:("Set writes the tensor node " ^ Tn.debug_name acc)

let entry_refusal ~what ~acc f =
  match counting (fun () -> Result.try_with f) with
  | Error (Invalid_argument msg), 0 -> names_entry ~what ~acc msg
  | Error exn, n ->
      Stdio.eprintf "%s: %d ops ran, then %s (not part of the golden)\n%!" what n
        (Exn.to_string exn);
      false
  | Ok _, n ->
      Stdio.eprintf "%s: applied, %d ops ran (not part of the golden)\n%!" what n;
      false

let classify ~strict f =
  Outcome.protect ~strict
    ~classify_backend:(fun _ _ -> None)
    ~provenance:Outcome.Candidate ~phase:Outcome.Transform f

let values c (o : LL.optimized) =
  let memory =
    interp_memory
      [ (c.acc, [| 0.5; 0.25 |]); (c.x, Array.init 4 ~f:(fun n -> 0.125 *. Float.of_int (n + 1))) ]
  in
  interpret memory o.LL.llc;
  Array.copy (Hashtbl.find_exn memory c.acc)

let () =
  let hidden = program "h" ~hidden:true in
  p
    "an impure entry is refused by Schedule.apply, naming the entry and the hidden write, before \
     any op runs"
    (entry_refusal ~what:"apply" ~acc:hidden.acc (fun () -> Sched.apply (gap hidden) hidden.opt));
  (* [schedule_legality] probes on a hermetic copy and calls no fault seam, so only the refusal
     itself is observable here. *)
  p "schedule_legality refuses it too, naming the entry and the hidden write, with no verdict"
    (match Sched.schedule_legality hidden.opt (gap hidden) with
    | verdicts ->
        Stdio.eprintf "legality: %d verdicts (not part of the golden)\n%!" (List.length verdicts);
        false
    | exception Invalid_argument msg -> names_entry ~what:"legality" ~acc:hidden.acc msg);
  p "strict: the entry refusal is fatal, untyped (Invalid_argument, no cause)"
    (match classify ~strict:true (fun () -> Sched.apply_classified (gap hidden) hidden.opt) with
    | Error (Outcome.Fatal { exn = Invalid_argument msg; cause = None; _ }) ->
        names_entry ~what:"strict" ~acc:hidden.acc msg
    | Ok _ | Error _ -> false);
  p "permissive: it is contained as Unclassified, never as Illegal_schedule"
    (match classify ~strict:false (fun () -> Sched.apply_classified (gap hidden) hidden.opt) with
    | Error
        (Outcome.Classified
           { cause = Outcome.Unclassified { exn_constructor = "Invalid_argument"; _ }; _ }) ->
        true
    | Ok _ | Error _ -> false);
  p "the empty schedule is the identity on the impure entry, dispatching nothing"
    (match counting (fun () -> Sched.apply [] hidden.opt) with
    | o, 0 -> phys_equal o hidden.opt
    | _ -> false);
  (* The statement-level twin: the op reads the write and refuses. *)
  let twin = program "t" ~hidden:false in
  p
    "the statement-level twin reaches its ops and is refused at the Retype, naming the tile's \
     lifetime"
    (match counting (fun () -> Result.try_with (fun () -> Sched.apply (gap twin) twin.opt)) with
    | Error (Invalid_argument msg), 2 ->
        Stdio.eprintf "twin: %s (not part of the golden)\n%!" msg;
        String.is_prefix msg ~prefix:"Schedule.Retype:"
        && String.is_substring msg ~substring:"per-thread lifetime of the Privatize tile"
    | _ -> false);
  (* A pure entry with a pure scope body runs every op. *)
  let pure = program "p" ~hidden:false in
  let sched = [ pz pure; Sched.Retype { axis = pure.r; ty = LL.Serial } ] in
  let applied, dispatched = counting (fun () -> Sched.apply sched pure.opt) in
  p "a pure entry holding a pure scope body runs every op" (dispatched = List.length sched);
  p_all2 "and computes the unscheduled program's values" (values pure applied)
    (values pure pure.opt) ~f:Float.equal;
  (* The other half of the scope grammar is left to the ops, which can repair it. *)
  let s = node "sink_g" and v = node "v_g" in
  virtualize v;
  materialize s;
  let i = sym () and j = sym () in
  let id = LL.get_scope v in
  let build axis =
    loop_n i 2
      (set s
         [| iter i |]
         (scope_of id (loop_n ~axis j 2 (LL.Set_local (id, add (c 1.) (embed j))))))
  in
  let grid =
    optimize_scoped ~name:"se2314_grid"
      ~raw:(loop_n i 2 (set s [| iter i |] (c 1.)))
      (build LL.Grid)
  in
  p
    "a pure entry with a Grid loop inside a scope body is admitted, and its Retype to Serial \
     repairs it"
    (match Sched.apply [ Sched.Retype { axis = j; ty = LL.Serial } ] grid with
    | o -> Option.is_none (LL.scope_hardware_loop o.LL.llc)
    | exception Invalid_argument msg ->
        Stdio.eprintf "grid: %s (not part of the golden)\n%!" msg;
        false);
  Stdio.printf "schedule_entry_scope_purity: done\n"
