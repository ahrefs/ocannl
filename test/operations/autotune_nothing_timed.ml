(* gh-ocannl-2310: a search that timed nothing says so in its outcome, and the routine it returns is
   the untuned default compile.

   The single-output shape is test/operations/metal_wait_bound_real_hang's reduction at a finite
   extent, summed to one cell: [y = sum over i, j, o, k of x[o, k]], two product components of
   extent 2 added to [x ++ "ok=>"] that neither side reads. On a GPU backend the search has nothing
   to dispatch there: the serial baseline binds no hardware dimension, and neither does any seed
   (the added components keep the split-reduce family off it), so every one of them is refused
   before codegen (gh-ocannl-532) and the census holds only [Not_dispatched_key]s. On cc the serial
   baseline is dispatched and timed, so the same declines are forced: the test's driver wraps
   [Autotune.Context_driver] and refuses every candidate compile, the baseline's included, inside
   its transform, with the typed cause the dispatch gate raises there. The untuned default compile,
   which carries no candidate label, passes untouched; so does every compile on a GPU backend, where
   the search runs as in production.

   Asserted, on every backend: the search timed nothing, its outcome is [Searched_untimed] (not
   [Searched] over zero timings, the state before gh-ocannl-2310), the crown line spells it, the
   census holds only not-dispatched refusals with the baseline's among them, and the returned
   routine is the untuned default's own compile, computing the untuned values. On a GPU backend the
   routine it returned binds no hardware dimension: it is the one-work-item form the search refused
   to time, shipped because it is what the caller would have compiled without the tuner (running it,
   like running any routine [tune] returns, is a wait no bound covers).

   The negative control is a search that times exactly one candidate: an [m]-output reduction, whose
   outputs give the GPU seeds a hardware dimension to bind, through the same driver letting the
   first label past the dispatch gate through (on cc the baseline, on a GPU backend the first seed)
   and refusing the rest. Its outcome stays [Searched], and it crowns that candidate. The timings
   are scripted ([Scripted_search]) so no window is refused for the host's load. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Asgns = Ir.Assignments
module Idx = Ir.Indexing
module SO = Ir.Schedule_outcome
open Verdict.Claims

let m = 16
let inner = 16

(* [x[o, k]] is [o mod 8] plus 0, 0.25, 0.5 or 0.75 by [k]: dyadic, so every sum is exact. *)
let x_value o k = (Float.of_int (k % 4) *. 0.25) +. Float.of_int (o % 8)

let x_tensor label =
  TDSL.ndarray
    (Array.init (m * inner) ~f:(fun f -> x_value (f / inner) (f % inner)))
    ~label:[ label ^ "_x" ]
    ~output_dims:[ m; inner ] ()

(* The single-output reduction: [x ++ "ok=>"] with two more product components of extent 2. *)
let single_output label =
  let x = x_tensor label in
  let%op y = x ++ "ok=>" in
  let fwd = Train.forward y in
  let leaves = Asgns.fold_leaves fwd.Asgns.asgns ~init:[] ~f:(fun acc leaf -> leaf :: acc) in
  let patched =
    match leaves with
    | [ Asgns.Accum_op a ] ->
        let p = Lazy.force a.projections in
        let extra = Array.init 2 ~f:(fun _ -> [ (2, Idx.get_symbol ()) ]) in
        let p = { p with Idx.components = Array.append extra p.Idx.components } in
        Asgns.Accum_op { a with projections = Lazy.from_val p }
    | _ -> failwith "expected the reduction's forward code to be one Accum_op"
  in
  (Asgns.comp ~embedded:fwd.Asgns.embedded_nodes (Asgns.Block_comment (label, patched)), y)

(* The control's [m]-output reduction. *)
let m_outputs label =
  let x = x_tensor label in
  let%op y = x ++ "ok=>o" in
  (Ll_test.unowned_copy (Train.forward y), y)

(* How many distinct candidate labels the driver lets past the gate: [None] lets every compile
   through (the production search), [Some k] the first [k] labels and every later compile of one of
   them (the winner's replay carries the label it was timed under). *)
let allow : int option ref = ref None
let allowed = Hash_set.create (module String)

(* Every compile that returned a routine: its id, provenance and candidate label. *)
let compiles : (int * SO.provenance * string option) list ref = ref []

module Refusing_driver = struct
  include Autotune.Context_driver

  let refuse c =
    raise
      (SO.Cause_at
         ( SO.Transform,
           SO.Not_dispatched
             {
               origin = (if String.equal c "baseline" then "baseline" else "candidate");
               detail = c ^ " refused by the test's driver (gh-ocannl-2310)";
             } ))

  let compile_outcome ?name ?lowered_transform ?prelowered ?reuse ~provenance ?candidate ?retain ctx
      comp bindings =
    let lowered_transform =
      match (lowered_transform, candidate, !allow) with
      | Some transform, Some c, Some k ->
          (* After the inner transform, so the search's own gate (and the base compile's unwind)
             decides first, as on a GPU backend. *)
          Some
            (fun opt ->
              let forms = transform opt in
              if Hash_set.mem allowed c then forms
              else if Hash_set.length allowed < k then (
                Hash_set.add allowed c;
                forms)
              else refuse c)
      | transform, _, _ -> transform
    in
    let outcome =
      Autotune.Context_driver.compile_outcome ?name ?lowered_transform ?prelowered ?reuse
        ~provenance ?candidate ?retain ctx comp bindings
    in
    (match outcome with
    | Ok (_, routine) -> compiles := (routine_id routine, provenance, candidate) :: !compiles
    | Error _ -> ());
    outcome
end

module Refusing = Autotune.Make (Refusing_driver)

type run = {
  report : Autotune.report;
  values : float array;
  expected : float array;
  shipped_id : int;
  shipped_hardware : bool;  (** Whether a kernel of the returned routine binds a hardware axis. *)
  run_compiles : (int * SO.provenance * string option) list;
}

let binds_hardware (opt : Ir.Low_level.optimized) =
  not (List.is_empty (Ir.Low_level.hardware_axes opt.Ir.Low_level.llc))

let search ~allow:a (comp, (y : Tensor.t)) =
  allow := a;
  Hash_set.clear allowed;
  compiles := [];
  let expected = Scripted_search.untuned ~ctx:(Context.auto ()) y comp in
  let report = ref None in
  let (values, shipped_id, shipped_hardware), _record =
    Scripted_search.scripted
      ~measure:(fun w -> Float.of_int (10 + w.Scripted_search.ordinal))
      (fun () ->
        let ctx, routine =
          Refusing.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:""
            ~report:(fun r -> report := Some r)
            (Context.auto ()) comp Idx.Empty
        in
        let shipped_id = Autotune.Context_driver.routine_id routine in
        let hardware =
          List.exists (Autotune.Context_driver.routine_segments routine) ~f:binds_hardware
        in
        let ctx = Context.run ctx routine in
        let values = Context.get_values ctx y.Tensor.value in
        Context.release ctx;
        (values, shipped_id, hardware))
  in
  let report = Option.value_exn ~message:"the search delivered no report" !report in
  Stdio.eprintf "%s (not part of the golden)\n%!" (Autotune.crown_fields report);
  { report; values; expected; shipped_id; shipped_hardware; run_compiles = List.rev !compiles }

let not_dispatched (d : Autotune.decline_summary) =
  match d.Autotune.key with SO.Not_dispatched_key _ -> true | _ -> false

let () =
  let ctx = Context.auto () in
  let backend = Context.backend_name ctx in
  let on_gpu = Ir.Backend_intf.binds_hardware_grid (Context.backend_facts ctx) in
  Context.release ctx;
  (* --- The search that times nothing. *)
  let u = search ~allow:(if on_gpu then None else Some 0) (single_output "nt") in
  let r = u.report in
  p "the single-output search timed nothing"
    (r.Autotune.candidates_timed = 0 && Float.is_inf r.Autotune.best_ms
    && String.is_empty r.Autotune.best_label);
  p "its outcome says nothing was timed"
    (match r.Autotune.outcome with Autotune.Searched_untimed -> true | _ -> false);
  p "its crown line spells the state"
    (String.is_prefix (Autotune.crown_fields r) ~prefix:"outcome=searched-untimed crowned=\"\" ");
  p_all "every decline in its census is a not-dispatched refusal" r.Autotune.declines
    ~f:not_dispatched;
  p "the serial baseline is among them"
    (List.exists r.Autotune.declines ~f:(fun d ->
         SO.equal_rejection_key d.Autotune.key (SO.Not_dispatched_key "baseline")));
  (* The untuned default is the one compile with the user's provenance; a seeded plan the search
     refused would degenerate there too, so the plain default is what ships on every backend. *)
  let untuned_ids =
    List.filter_map u.run_compiles ~f:(fun (id, provenance, _) ->
        match provenance with SO.User_schedule -> Some id | _ -> None)
  in
  p "the returned routine is the plain untuned default's compile"
    (List.equal Int.equal untuned_ids [ u.shipped_id ]);
  p_all2 "the returned routine computes the untuned values" u.values u.expected ~f:Float.equal;
  Verdict.in_leg ~leg:(Verdict.leg ~when_:on_gpu ~on:backend ())
    "on a GPU backend, the returned routine binds no hardware dimension" (fun claim ->
      p claim (not u.shipped_hardware));
  (* --- The negative control: one candidate timed. *)
  let c = search ~allow:(Some 1) (m_outputs "ntc") in
  let r = c.report in
  let let_through = Hash_set.to_list allowed in
  p "the control timed exactly the one candidate let through"
    (r.Autotune.candidates_timed = 1 && List.length let_through = 1);
  p "the control's outcome stays Searched"
    (match r.Autotune.outcome with Autotune.Searched -> true | _ -> false);
  p "the control crowned that candidate"
    (Float.is_finite r.Autotune.best_ms
    && List.equal String.equal [ r.Autotune.best_label ] let_through);
  p_all2 "the control's returned routine computes the untuned values" c.values c.expected
    ~f:Float.equal
