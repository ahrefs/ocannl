(* gh-ocannl-1526: under node-bounds drift, a search's candidates get the lineage state of the code
   they are rebased on -- the base's -- not that of a lowering taken after the drift.

   A search lowers its routine once ([Ir.Lowering_reuse]) and every candidate compile takes a copy.
   The one way the inputs of that lowering change during a search is a link proposing node value
   bounds: [Host_inits] uploads an index tensor and records its scanned range, and a lowering taken
   after that folds the gather's in-range guard the base kept (gh-ocannl-1614's drift). Every
   candidate's CODE is the base's either way; before gh-ocannl-1526 its lineage fork came from a
   fresh lowering taken after the drift -- decided under other code. Reuse keeps the base's.

   The fixture is [tune_warm_cache_key]'s embedding gather, tuned first in the process so that
   nothing has uploaded the ids before the cold search. The cold leg runs under
   [Lowering_reuse.with_audit], so every reuse is also compared with the fresh lowering the compile
   would have made: some such comparison differing is the precondition that makes the leg
   non-vacuous (the drift happened mid-search). The cold search's snapshot is the routine as lowered
   before the search (its base's inputs), not the program a lowering after the drift gives, and the
   routine it ships computes the host reference. A second, warm search then lowers its base AFTER
   the drift; a winner is stored for the blind lowering, so the call rebases on that
   (gh-ocannl-1614), and its cell must be reseeded with it: the warm cell's snapshot is the blind
   lowering too, and the replayed routine computes the reference. Every value is a multiple of 1/4
   small enough that any summation order computes it exactly, so outputs are compared bit for
   bit. *)

open Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module LR = Ir.Lowering_reuse
module FP = Lowering_fingerprint

let tokens = 2048
let vocab = 16
let embed = 32
let id_of t = ((t * 7) + (t / 5)) % vocab
let table_value c = Float.of_int ((c * 5 % 9) - 4) /. 4.

let reference =
  Array.init embed ~f:(fun o ->
      let acc = ref 0. in
      for t = 0 to tokens - 1 do
        acc := !acc +. table_value ((o * vocab) + id_of t)
      done;
      !acc)

let ids = Nn_blocks.class_ids_of_int_list ~label:"lrd_ids" (List.init tokens ~f:id_of)

let table =
  TDSL.ndarray
    (Array.init (embed * vocab) ~f:table_value)
    ~label:[ "lrd_table" ] ~input_dims:[ vocab ] ~output_dims:[ embed ] ()

let%op lrd_emb = table * Nn_blocks.one_hot_of_ids ~num_classes:vocab ids
let%op lrd_out = lrd_emb ++ "b|o => o"
let gather = Train.forward lrd_out
let cache_dir = "autotune_cache_lowering_reuse_drift"
let () = Scripted_search.clear_cache_dir cache_dir

(* A fissioned candidate wins, as in [tune_warm_cache_key]: the warm call's replay then checks each
   segment's digest against the code it cuts. *)
let measure (w : Scripted_search.window) =
  if String.is_prefix w.label ~prefix:"F_" then 1. +. (Float.of_int w.ordinal /. 64.) else 50.

(* One search, observing the cells it makes and (with [~audit]) each reuse beside the fresh lowering
   it replaced. *)
let search ?(audit = false) tag =
  let cells = ref [] and audits = ref [] in
  let run () =
    Scripted_search.search ~ctx:(Context.auto ()) ~output:lrd_out ~cache_dir ~beam_width:1 ~tag
      ~measure gather
  in
  let run () =
    if audit then
      LR.with_audit
        (fun ~reused ~fresh ->
          audits := FP.differing (FP.sections reused) (FP.sections fresh) :: !audits)
        run
    else run ()
  in
  let o = LR.with_created_observer (fun c -> cells := c :: !cells) run in
  Stdio.eprintf "%s (not part of the golden): %s, crowned %S, %d cell(s), %d audited reuse(s)\n%!"
    tag
    (Autotune.outcome_name o.report.Autotune.outcome)
    o.report.Autotune.best_label (List.length !cells) (List.length !audits);
  (o, List.rev !cells, List.rev !audits)

(* The routine as a fresh compile would lower it now, with the node bounds read or not. *)
let lowered_now ~blind =
  let lower () = Context.lowered_for_decisions (Context.auto ()) gather Ir.Indexing.Empty in
  FP.sections (if blind then Ir.Low_level.without_node_bounds lower else lower ())

let snapshot_sections cells =
  match cells with [ cell ] -> Option.map (LR.snapshot cell) ~f:FP.sections | _ -> None

let () =
  Stdio.eprintf "backend: %s (not part of the golden)\n%!" (Context.backend_name (Context.auto ()));
  p "setup: nothing had proposed the ids' bounds before the cold search"
    (Option.is_none (Ir.Tnode.bounds_candidate ids.Tensor.value));
  (* The routine lowered under the base's inputs: before anything of the search linked. *)
  let before = lowered_now ~blind:false in
  let cold, cells, audits = search ~audit:true "cold" in
  p "cold: the call searched"
    (match cold.report.Autotune.outcome with Autotune.Searched -> true | _ -> false);
  p "cold: a link proposed the ids' bounds during the search"
    (Option.is_some (Ir.Tnode.bounds_candidate ids.Tensor.value));
  p_exists
    "cold: the drift happened mid-search -- a reuse differs from the fresh lowering it replaced"
    audits ~f:(fun differing -> not (List.is_empty differing));
  let blind = lowered_now ~blind:true and bounded = lowered_now ~blind:false in
  Option.iter (snapshot_sections cells) ~f:(fun snap ->
      Stdio.eprintf
        "cold snapshot vs the lowering before the search: [%s]; vs the bounded one now: [%s] (not \
         part of the golden)\n\
         %!"
        (String.concat ~sep:"," (FP.differing snap before))
        (String.concat ~sep:"," (FP.differing snap bounded)));
  p "precondition: the routine lowers to two programs, blind and bounded"
    (not (List.is_empty (FP.differing blind bounded)));
  (* Against a lowering taken BEFORE the search, not a blind one taken now: a link also changes
     inputs of the flip pricing ([Host_inits] membership among them), so a lowering taken after the
     search differs from the base in [flip_candidates] even blind to bounds (seen on HIP). *)
  p "cold: the candidates' lowering is the base's, the routine as lowered before the search"
    (Option.exists (snapshot_sections cells) ~f:(fun s -> List.is_empty (FP.differing s before)));
  p "cold: ... and not the program a lowering after the drift gives"
    (Option.exists (snapshot_sections cells) ~f:(fun s ->
         List.mem (FP.differing s bounded) "llc" ~equal:String.equal));
  p "cold: the shipped routine computes the host reference bit for bit"
    (Array.equal Float.equal cold.output reference);
  let warm, cells, _ = search "warm" in
  p "warm: the call replayed the cold winner"
    (match warm.report.Autotune.outcome with Autotune.Cache_replay -> true | _ -> false);
  p "warm: its base was lowered after the drift, yet its cell holds the blind lowering (reseeded)"
    (Option.exists (snapshot_sections cells) ~f:(fun s -> List.is_empty (FP.differing s blind)));
  p "warm: the shipped routine computes the host reference bit for bit"
    (Array.equal Float.equal warm.output reference)
