(* gh-ocannl-1614: a tune after a link replays the winner its cold twin stored.

   The base lowering reads node value bounds: a [Host_inits] upload of an index tensor proposes its
   scanned range when a routine first links, and a lowering taken after that folds the gather's
   in-range guard that a lowering taken before it kept. The base's canonical digest is the schedule
   cache's key, so a second [Autotune.tune] of the same routine, after the first had linked and run,
   missed the entry the first stored and searched again (gh-ocannl-1560's audit, verified on cc).
   [tune] now rebases such a call on the routine's bounds-blind lowering when a winner is stored
   under that lowering's key and none under the bounded one's.

   The fixture is [tune_strict_replay]'s embedding gather, [table * one_hot(ids)] summed over 8192
   tokens into 64 cells, tuned first in the process so that nothing has uploaded the ids before the
   cold search. The sequence: a cold call searches and stores; the link it made has proposed the
   ids' bounds, so the routine now lowers to two programs, blind and bounded (the precondition that
   makes the rest non-vacuous); a warm call replays the cold winner, as does a third. Over-keying is
   the other defect, so the last leg shows the key still separates a genuinely different base: the
   same gather with a [relu] before the sum, over the same ids -- its bounded lowering also drifted,
   so the blind tier is consulted -- searches rather than replaying the gather's winner. Every value
   is a multiple of 1/4 small enough that any summation order computes it exactly, so the outputs
   are compared with a host reference bit for bit. *)

open Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module SC = Ir.Schedule_cache

let tokens = 8192
let vocab = 16
let embed = 64
let id_of t = ((t * 7) + (t / 5)) % vocab
let table_value c = Float.of_int ((c * 5 % 9) - 4) /. 4.

(* [out.(o) = sum_t f (table.(o, ids.(t)))]. *)
let reference f =
  Array.init embed ~f:(fun o ->
      let acc = ref 0. in
      for t = 0 to tokens - 1 do
        acc := !acc +. f (table_value ((o * vocab) + id_of t))
      done;
      !acc)

let ids = Nn_blocks.class_ids_of_int_list ~label:"wck_ids" (List.init tokens ~f:id_of)

let table =
  TDSL.ndarray
    (Array.init (embed * vocab) ~f:table_value)
    ~label:[ "wck_table" ] ~input_dims:[ vocab ] ~output_dims:[ embed ] ()

let%op wck_emb = table * Nn_blocks.one_hot_of_ids ~num_classes:vocab ids
let%op wck_out = wck_emb ++ "b|o => o"
let gather = Train.forward wck_out

(* Its own gather: a node one routine's forward computes is the other's input, not code. *)
let%op wck_relu_emb = table * Nn_blocks.one_hot_of_ids ~num_classes:vocab ids
let%op wck_relu_out = relu wck_relu_emb ++ "b|o => o"
let relu_gather = Train.forward wck_relu_out
let cache_dir = "autotune_cache_warm_cache_key"

(* An earlier run's entry would turn the cold search into a replay. *)
let () = Scripted_search.clear_cache_dir cache_dir

(* A fissioned candidate wins where the search times one: its replay checks each segment's digest
   against the code it cuts, the replay a drifted base would be refused by. *)
let measure (w : Scripted_search.window) =
  if String.is_prefix w.label ~prefix:"F_" then 1. +. (Float.of_int w.ordinal /. 64.) else 50.

let search ~tag ~output comp =
  let (o : Scripted_search.observed) =
    Scripted_search.search ~output ~cache_dir ~beam_width:1 ~tag ~measure comp
  in
  Stdio.eprintf "%s (not part of the golden): %s, digest %s, crowned %S\n%!" tag
    (Autotune.outcome_name o.report.Autotune.outcome)
    o.report.Autotune.source_digest o.report.Autotune.best_label;
  o

let searched (o : Scripted_search.observed) =
  match o.report.Autotune.outcome with Autotune.Searched -> true | _ -> false

let replayed (o : Scripted_search.observed) =
  match o.report.Autotune.outcome with Autotune.Cache_replay -> true | _ -> false

(* A store the filesystem committed. *)
let is_store (io : SC.cache_io) =
  match (io.op, io.refusal) with SC.Store, None -> true | _ -> false

(* The routine's canonical digest as a fresh compile would lower it now, with the node bounds read
   ([blind = false]) or not. *)
let digest_now ~blind comp =
  let lower () = Context.lowered_for_decisions (Context.cpu ()) comp Ir.Indexing.Empty in
  let opt = if blind then Ir.Low_level.without_node_bounds lower else lower () in
  SC.digest (SC.canonicalize opt)

let () =
  p "setup: nothing had proposed the ids' bounds before the cold search"
    (Option.is_none (Ir.Tnode.bounds_candidate ids.Tensor.value));
  let gather_ref = reference Fn.id in
  let cold = search ~tag:"cold" ~output:wck_out gather in
  p "cold: the call searched" (searched cold);
  p_exists "cold: the search stored its winner" cold.ios ~f:is_store;
  p "cold: a fissioned candidate won, so a replay checks each segment's digest"
    (String.is_prefix cold.report.Autotune.best_label ~prefix:"F_");
  p "cold: the shipped routine computes the host reference bit for bit"
    (Array.equal Float.equal cold.output gather_ref);
  let cold_digest = cold.report.Autotune.source_digest in
  (* The precondition: the link drifted the lowering, and the cold base was the blind one. *)
  p "after cold: a link proposed the ids' bounds"
    (Option.is_some (Ir.Tnode.bounds_candidate ids.Tensor.value));
  p "after cold: the routine lowered blind to node bounds is the cold base"
    (String.equal (digest_now ~blind:true gather) cold_digest);
  p "after cold: the routine lowered with the node bounds is another program"
    (not (String.equal (digest_now ~blind:false gather) cold_digest));
  let replay_leg tag =
    let o = search ~tag ~output:wck_out gather in
    p (tag ^ ": the call replayed the cold winner rather than searching") (replayed o);
    p
      (tag ^ ": the replay names the code the winner was searched on")
      (String.equal o.report.Autotune.source_digest cold_digest);
    p_none (tag ^ ": nothing was stored") o.ios ~f:is_store;
    p
      (tag ^ ": the shipped routine computes the host reference bit for bit")
      (Array.equal Float.equal o.output gather_ref)
  in
  replay_leg "warm";
  replay_leg "third";
  (* Over-keying's mirror: the blind tier must not conflate two genuinely different routines. *)
  let relu_blind = digest_now ~blind:true relu_gather in
  p "relu: the routine lowers to two programs too, so the blind tier is consulted"
    (not (String.equal relu_blind (digest_now ~blind:false relu_gather)));
  p "relu: its blind lowering is not the gather's" (not (String.equal relu_blind cold_digest));
  let relu = search ~tag:"relu" ~output:wck_relu_out relu_gather in
  p "relu: the call searched rather than replaying the gather's winner" (searched relu);
  p "relu: the shipped routine computes the host reference bit for bit"
    (Array.equal Float.equal relu.output (reference (fun v -> Float.max 0. v)))

let () = Stdio.printf "tune_warm_cache_key: done\n"
