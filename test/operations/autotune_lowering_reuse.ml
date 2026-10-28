(* gh-ocannl-1526: a search lowers its routine once, and every candidate compile reuses that
   lowering.

   Every candidate of [Autotune.tune] builds its code from the one base lowering the search
   captured; the fresh lowering each candidate compile used to make contributed only a fork of the
   lineage state, at the price of a whole lowering per candidate. The search now hands its compiles
   one [Ir.Lowering_reuse] cell: the base compile fills it, every later compile from the search's
   context takes a private copy. What this test holds that to:

   1. Equivalence: under [Lowering_reuse.with_audit], every reuse also lowers fresh, and the two
   agree in every section of [Lowering_fingerprint] -- the code, the traced store, every placement,
   the stored computations, the preference and alias sets, the companions. The fingerprint is
   checked to be non-vacuous (the lineage carries a virtual node, an inherited computation and an
   inline preference) and discriminating (a flipped placement, a dropped preference and a toggled
   traced flag each change it). 2. Isolation: the cell's snapshot fingerprints the same at the end
   of the search as right after the base compile, codegen, links, refusals and duplicates of every
   candidate in between. 3. The saving: a search makes fewer analysis-cache lookups (one per fresh
   lowering) than it has compiled candidates; the same search with reuse disabled makes at least as
   many, so the counter discriminates. 4. Executed parity: scripted measurements crown, in turn,
   each of several compiled candidates; the routine each such search ships takes the reference step.

   The step is [autotune_timing_state_restore]'s SGD step with momentum over a matmul, warmed by one
   step so that its lineage holds stored computations, then given an inline preference. Every value
   is exact by construction (small integers, eighths, power-of-two rates), so a step run by any
   schedule agrees with the untuned reference bitwise. The measurements are scripted; what the
   claims read is never a time. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module IDX = Train.IDX
module LL = Ir.Low_level
module LR = Ir.Lowering_reuse
module FP = Lowering_fingerprint

let n = 16

(* One observer for the process (observers accumulate): the records of the search in progress. *)
let current_records : Autotune.compile_record list ref option ref = ref None
let on_base_compile : (unit -> unit) ref = ref (fun () -> ())

let () =
  Autotune.install_compile_observer (fun r ->
      Option.iter !current_records ~f:(fun acc -> acc := r :: !acc);
      if String.equal r.Autotune.compile_stage "base_compile" then !on_base_compile ())

let compiled_candidates records =
  List.filter records ~f:(fun (r : Autotune.compile_record) ->
      String.equal r.compile_stage "candidate" && String.equal r.compile_verdict "compiled")

let lookups () =
  let hits, misses = LL.analysis_cache_stats () in
  hits + misses

let () =
  Tensor.unsafe_reinitialize ();
  let x =
    TDSL.ndarray
      (Array.init (n * n)
         ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:7 ~offset:(-3.) ~stride:1.))
      ~input_dims:[ n ] ~output_dims:[ n ] ()
  in
  let w_init =
    Array.init (n * n)
      ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:9 ~offset:(-4.) ~stride:0.125)
  in
  let w = TDSL.param ~values:w_init "w" ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op y = w * x in
  let%op loss = y ++ "...|...->... => 0" in
  let%op learning_rate = 0.0078125 in
  Train.set_materialized w.Tensor.value;
  Train.set_materialized learning_rate.Tensor.value;
  let comp =
    Ir.Assignments.sequence
      [ Train.grad_update loss; Train.sgd_update ~learning_rate ~momentum:0.5 loss ]
  in
  (* A lineage holding stored computations (the warm-up step's) and an inline preference. *)
  let lineage () =
    let ctx = Train.init_params (Context.auto ()) IDX.empty loss in
    let ctx, routine = Context.compile ctx comp IDX.empty in
    Context.decide_inline (Context.run ctx routine) [ y.Tensor.value ]
  in
  let read ctx = Context.get_values ctx w.Tensor.value in
  let stepped_once =
    let ctx, routine = Context.compile (lineage ()) comp IDX.empty in
    read (Context.run ctx routine)
  in
  (* The routine's own nodes, as a fresh lowering holds them: the shipped segments of two compiles
     agree on these by identity, and on the nodes their schedule minted by first occurrence. *)
  let known =
    let lowered = Context.lowered_for_decisions (lineage ()) comp IDX.empty in
    let set = Hash_set.create (module Ir.Tnode) in
    Hashtbl.iter_keys lowered.LL.traced_store ~f:(Hash_set.add set);
    Hashtbl.iter_keys lowered.LL.optimize_ctx.LL.placements.Ir.Tnode.Placements.table
      ~f:(Hash_set.add set);
    Hash_set.mem set
  in
  Stdio.eprintf "backend: %s (not part of the golden)\n%!" (Context.backend_name (Context.auto ()));
  (* One search on a fresh lineage, its windows scripted by [measure]; what it did and shipped. *)
  let search ?(audit = false) ?(disabled = false) ?timing_ctx ~measure tag =
    let records = ref [] and cells = ref [] and audits = ref [] and report = ref None in
    let at_base = ref None in
    current_records := Some records;
    (on_base_compile :=
       fun () ->
         Option.iter (List.hd !cells) ~f:(fun cell ->
             at_base := Option.map (LR.snapshot cell) ~f:FP.sections));
    let ctx = lineage () in
    let lookups0 = lookups () and reused0, lowered0 = LR.stats () in
    let tune () =
      Autotune.tune ~search:true ~beam_width:2 ~rounds:0 ~repeats:1 ~timing:Autotune.Isolated
        ~cache_dir:""
        ~report:(fun r -> report := Some r)
        ?timing_ctx ctx comp IDX.empty
    in
    let tune () =
      if audit then
        LR.with_audit
          (fun ~reused ~fresh -> audits := (FP.sections reused, FP.sections fresh) :: !audits)
          tune
      else tune ()
    in
    let (shipped_ctx, routine), _ =
      Scripted_search.scripted ~measure (fun () ->
          LR.with_created_observer
            (fun cell -> cells := cell :: !cells)
            (fun () -> if disabled then LR.with_disabled tune else tune ()))
    in
    let lookups = lookups () - lookups0 in
    let reused1, lowered1 = LR.stats () in
    current_records := None;
    (on_base_compile := fun () -> ());
    let step = read (Context.run shipped_ctx routine) in
    let shipped =
      List.map routine.Context.segments ~f:(fun (seg : LL.optimized) -> FP.sections ~known seg)
    in
    let records = List.rev !records in
    Stdio.eprintf "%s stages: %s (not part of the golden)\n%!" tag
      (String.concat ~sep:" "
         (List.map records ~f:(fun (r : Autotune.compile_record) ->
              r.compile_stage ^ "/" ^ r.compile_verdict)));
    Stdio.eprintf
      "%s: %d compile records, %d compiled candidates, %d analysis-cache lookups, %d reused, %d \
       lowered into a cell (not part of the golden)\n\
       %!"
      tag (List.length records)
      (List.length (compiled_candidates records))
      lookups (reused1 - reused0) (lowered1 - lowered0);
    ( records,
      List.rev !cells,
      List.rev !audits,
      !at_base,
      Option.value_exn ~here:[%here] !report,
      step,
      lookups,
      (reused1 - reused0, lowered1 - lowered0),
      shipped )
  in
  let by_ordinal (w : Scripted_search.window) = Float.of_int w.ordinal in
  (* 1, 2. The audited search. *)
  let records, cells, audits, at_base, _, step, _, (reused, lowered), _ =
    search ~audit:true ~measure:by_ordinal "audited"
  in
  let candidates = compiled_candidates records in
  p "audited: the search compiled at least three candidates" (List.length candidates >= 3);
  p "audited: the search made one cell and filled it once" (List.length cells = 1 && lowered = 1);
  p "audited: every candidate compile reused the lowering, and the audit saw every reuse"
    (reused >= List.length candidates && List.length audits = reused);
  p_all "audited: every reuse equals a fresh lowering of the routine, section by section" audits
    ~f:(fun (r, f) ->
      let differing = FP.differing r f in
      if not (List.is_empty differing) then
        Stdio.eprintf "reuse differs from a fresh lowering in: %s\n%!"
          (String.concat ~sep:", " differing);
      List.is_empty differing);
  let snapshot = Option.bind (List.hd cells) ~f:LR.snapshot in
  p "audited: the cell holds a snapshot" (Option.is_some snapshot);
  Option.iter snapshot ~f:(fun snapshot ->
      let plc = snapshot.LL.optimize_ctx.LL.placements.Ir.Tnode.Placements.table in
      p "the fingerprint is non-vacuous: the lineage places a node virtual"
        (Hashtbl.exists plc ~f:(fun (mode, _) -> Poly.equal mode Ir.Tnode.Virtual));
      p "the fingerprint is non-vacuous: the lineage stores a computation"
        (Hashtbl.length snapshot.LL.optimize_ctx.LL.computations > 0);
      p "the fingerprint is non-vacuous: the lineage carries an inline preference"
        (Hash_set.length snapshot.LL.optimize_ctx.LL.inline_preferences > 0);
      let base = FP.sections snapshot in
      let perturbed section mutate =
        let copy = LR.private_copy snapshot in
        mutate copy;
        List.equal String.equal (FP.differing base (FP.sections copy)) [ section ]
      in
      p "the fingerprint discriminates a flipped placement"
        (perturbed "placements" (fun copy ->
             let table = copy.LL.optimize_ctx.LL.placements.Ir.Tnode.Placements.table in
             match Hashtbl.to_alist table with
             | [] -> ()
             | (tn, (mode, prov)) :: _ ->
                 let other =
                   if Poly.equal mode Ir.Tnode.Virtual then Ir.Tnode.Local else Ir.Tnode.Virtual
                 in
                 Hashtbl.set table ~key:tn ~data:(other, prov)));
      p "the fingerprint discriminates a dropped inline preference"
        (perturbed "inline_preferences" (fun copy ->
             Hash_set.clear copy.LL.optimize_ctx.LL.inline_preferences));
      p "the fingerprint discriminates a toggled traced flag"
        (perturbed "traced_store" (fun copy ->
             match Hashtbl.data copy.LL.traced_store with
             | [] -> ()
             | t :: _ -> t.LL.read_only <- not t.LL.read_only));
      p "isolation: the snapshot is unchanged from the base compile to the end of the search"
        (Option.exists at_base ~f:(fun at_base -> List.is_empty (FP.differing at_base base))));
  p "audited: the shipped routine takes the reference step" (Poly.equal step stepped_once);
  (* 3. The saving, and its control. *)
  let records, _, _, _, _, step, lookups, (reused, _), _ = search ~measure:by_ordinal "reusing" in
  let candidates = compiled_candidates records in
  let has_winner_compile records =
    List.exists records ~f:(fun (r : Autotune.compile_record) ->
        String.equal r.compile_stage "winner_compile")
  in
  p
    "reusing: every compile but the base's reused the lowering, the winner's into the search's own \
     context included"
    (has_winner_compile records && reused = List.length records - 1);
  p "reusing: fewer fresh lowerings (analysis-cache lookups) than compiled candidates"
    (List.length candidates >= 3 && lookups < List.length candidates);
  p "reusing: the shipped routine takes the reference step" (Poly.equal step stepped_once);
  let control_records, _, _, _, _, control_step, control_lookups, (control_reused, _), _ =
    search ~disabled:true ~measure:by_ordinal "disabled"
  in
  p "control: with reuse disabled, a fresh lowering per compiled candidate"
    (control_reused = 0
    && control_lookups >= List.length (compiled_candidates control_records)
    && not (List.is_empty (compiled_candidates control_records)));
  p "control: the shipped routine takes the reference step" (Poly.equal control_step stepped_once);
  (* 4. Executed parity, candidate by candidate: each crowned in turn. *)
  let labels =
    List.dedup_and_sort ~compare:String.compare
      (List.map candidates ~f:(fun (r : Autotune.compile_record) -> r.compile_attempt))
  in
  let crowned = List.take labels 4 in
  Stdio.eprintf "crowning in turn: %s (not part of the golden)\n%!" (String.concat ~sep:" " crowned);
  p "parity: at least three distinct candidates to crown" (List.length crowned >= 3);
  (* Each crowned twice, with reuse and without: the kernels shipped must be the same segments --
     covering-zero elision included on a GPU backend, which a compile through the [?prelowered] test
     seam would skip. *)
  p_all
    "parity: each crowned candidate ships the segments reuse-disabled ships, and takes the \
     reference step"
    crowned ~f:(fun label ->
      let crown ~disabled =
        let _, _, _, _, report, step, _, _, shipped =
          search ~disabled
            ~measure:(fun (w : Scripted_search.window) ->
              if String.equal w.label label then 0.5 else 10. +. Float.of_int w.ordinal)
            ((if disabled then "crown (disabled) " else "crown ") ^ label)
        in
        (String.equal report.Autotune.best_label label, step, shipped)
      in
      let crowned, step, shipped = crown ~disabled:false in
      let control_crowned, control_step, control_shipped = crown ~disabled:true in
      let same_segments =
        List.length shipped = List.length control_shipped
        && List.for_all2_exn shipped control_shipped ~f:(fun a b ->
            List.is_empty (FP.differing a b))
      in
      if not same_segments then
        Stdio.eprintf "%s: %d shipped segment(s) with reuse, %d without; differing: %s\n%!" label
          (List.length shipped) (List.length control_shipped)
          (String.concat ~sep:" | "
             (match List.zip shipped control_shipped with
             | Ok pairs ->
                 List.map pairs ~f:(fun (a, b) -> String.concat ~sep:"," (FP.differing a b))
             | Unequal_lengths -> [ "the segment counts" ]));
      crowned && control_crowned && same_segments && Poly.equal step stepped_once
      && Poly.equal control_step stepped_once);

  (* 5. Ownership: a cell serves the context, computation and bindings that filled it; a compile
     handing it another context is refused, and a search on a [timing_ctx] compiles its winner into
     the caller's lineage without the cell (that lineage lowers afresh, behind
     [Lineage_refused]). *)
  let cell = LR.create () and mine = lineage () and other = lineage () in
  let refused ctx =
    match Context.compile ~reuse:cell ctx comp IDX.empty with
    | _ -> false
    | exception exn -> String.is_substring (Exn.to_string exn) ~substring:"Lowering_reuse"
  in
  p "ownership: the cell serves the context that filled it"
    ((not (refused mine)) && not (refused mine));
  p "ownership: a compile from another context is refused" (refused other);
  let records, _, _, _, _, step, _, (reused, _), _ =
    search ~timing_ctx:(lineage ()) ~measure:by_ordinal "timing_ctx"
  in
  p
    "timing_ctx: the candidates reused the lowering, the winner compiled into the caller's lineage \
     did not"
    (has_winner_compile records && reused = List.length records - 2);
  p "timing_ctx: the shipped routine takes the reference step" (Poly.equal step stepped_once)
