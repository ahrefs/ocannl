(* gh-ocannl-1384: a padded conv's training step is keyed by the schedule cache, and a tuned run of
   it stores in pass 1 and replays in pass 2.

   Every padded node computed in a routine gets its margins reset by
   [Low_level.loop_over_padding_region], and its left-margin and middle strips used to share one
   loop symbol. The canonical render calls a re-bound binder shadowed, so [Schedule_cache.complete]
   was false for every routine computing a padded conv operand or its gradient, and [Autotune.tune]
   silently neither looked up nor stored: each later call searched again.

   Three legs: - the margin walk itself, on hand-sized dims: every loop binds its own symbol, and
   the cells it visits are exactly the margin, each once; - the lowered step of a small two-conv
   padded network (the shape of the benchmark's [cifar_resnet]: a batch slice feeding a padded conv,
   a relu feeding a second one): its padded nodes are really padded, every loop binder in it is
   distinct, and its canonical form is complete; - executed: [Autotune.tune] over that step searches
   in pass 1 and replays in pass 2 (a fresh context, the same cache directory), and both passes'
   gradients match the untuned compile's. *)

open Base
open Ocannl
open Nn_blocks.DSL_modules
module IDX = Train.IDX
module LL = Ir.Low_level
module SC = Ir.Schedule_cache
module Cache_plant = Test_utils.Cache_plant
open Verdict.Claims

let distinct_binders label llc =
  p_pairwise_distinct label
    (List.map (Ll_test.loop_sites llc) ~f:(fun s -> s.Ll_test.ls_index))
    ~equal:Ir.Indexing.equal_symbol ~to_string:Ir.Indexing.symbol_ident

(* --- Leg 1: the margin walk, executed. Its body increments the cell it visits, so after one run
   each cell holds its visit count: 1 on the margin, 0 in the data region. --- *)
let () =
  let dims = [| 2; 5; 6 |] in
  let padding =
    Ir.Ops.[| { left = 0; right = 0 }; { left = 1; right = 2 }; { left = 2; right = 1 } |]
  in
  let tn = Ll_test.node_factory ~first_id:91384 ~dims () "margin_probe" in
  Ll_test.materialize tn;
  let llc =
    LL.loop_over_padding_region ~dims ~padding ~body:(fun idcs ->
        Ll_test.set tn idcs (Ll_test.add (Ll_test.get tn idcs) (Ll_test.c 1.)))
  in
  distinct_binders "every margin loop binds its own symbol" llc;
  let n = Array.fold dims ~init:1 ~f:( * ) in
  let _o, got =
    Ll_test.optimize_and_execute ~name:"pcc_margin_walk" llc
      ~seed:[ (tn, Array.create ~len:n 0.) ]
      ~read:[ tn ]
  in
  let got = List.hd_exn got in
  let expected =
    Array.init n ~f:(fun i ->
        let h = i / 6 % 5 and w = i % 6 in
        if h < 1 || h >= 5 - 2 || w < 2 || w >= 6 - 1 then 1. else 0.)
  in
  p_all2 "the margin walk visits every margin cell once and no data cell" got expected
    ~f:Float.equal

(* --- Legs 2 and 3: a small padded two-conv network's training step. --- *)
let () =
  let n_batches = 2 and batch_size = 2 in
  let x_nd =
    Ir.Ndarray.init_array ~debug:"xs" Ir.Ops.single
      ~dims:[| n_batches * batch_size; 6; 6; 2 |]
      ~padding:None
      ~f:(fun idx ->
        Float.of_int ((Array.fold idx ~init:7 ~f:(fun acc i -> (acc * 31) + i) % 13) - 6) /. 4.)
  in
  let xs = TDSL.rebatch ~l:"xs" x_nd () in
  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let%op batch_x = xs @| batch_n in
  let%op h1 =
    relu (Nn_blocks.conv2d ~label:[ "c1" ] ~use_padding:true ~out_channels:3 () batch_x)
  in
  let%op h2 = Nn_blocks.conv2d ~label:[ "c2" ] ~use_padding:true ~out_channels:2 () h1 in
  let%op loss = (h2 *. h2) ++ "...|... => |->0" in
  let step = Train.grad_update loss in
  let comp = { step with asgns = Ir.Assignments.Block_comment ("padded_conv_step", step.asgns) } in
  let kernels =
    Set.to_list loss.Tensor.params
    |> List.filter ~f:(fun t ->
        String.is_substring (Ir.Tnode.label_name t.Tensor.value) ~substring:"kernel")
  in
  let grads_of ctx =
    List.concat_map kernels ~f:(fun t ->
        Array.to_list (Context.get_values ctx (Option.value_exn ~here:[%here] t.Tensor.diff).grad))
    |> Array.of_list
  in
  let fresh () =
    let ctx = Context.auto () in
    Train.init_params ctx bindings loss
  in
  let run_with ctx routine =
    let routine_ref = IDX.find_exn routine.Context.bindings batch_n in
    routine_ref := 1;
    Context.run ctx routine
  in
  (* Leg 2: the lowered step, as the tuner's baseline sees it. *)
  let captured = ref None in
  let ctx = fresh () in
  let _ctx, _routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        captured := Some opt;
        [ opt ])
      ctx comp bindings
  in
  let opt = Option.value_exn ~here:[%here] !captured in
  let padded t = Option.is_some (Ir.Tnode.get_padding t.Tensor.value) in
  p "the conv operands are padded nodes" (padded batch_x && padded h1);
  distinct_binders "every loop binder of the padded step is distinct" opt.LL.llc;
  let canon = SC.canonicalize ~static_indices:[ batch_n ] opt in
  p "the padded step's canonical form is complete" (SC.complete canon);
  p "a complete canonical form records no incompleteness reason"
    (Option.is_none (SC.incomplete_reason canon));
  (* Leg 3: tune twice against one cache directory. *)
  let cache_dir = "autotune_cache_padded_conv" in
  if Stdlib.Sys.file_exists cache_dir && Stdlib.Sys.is_directory cache_dir then
    Array.iter (Stdlib.Sys.readdir cache_dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat cache_dir f));
  let untuned_grads () =
    let ctx = fresh () in
    let ctx, routine = Context.compile ctx comp bindings in
    grads_of (run_with ctx routine)
  in
  let reports = ref [] in
  (* Each pass's own context's timing identity (what its cache key rests on) and the cache I/O it
     made, for the waivers below. Pass 1 also tunes the forward-only step, a second padded-conv
     routine into the same directory, so the claim no refusal waives has two schedule stores to
     judge. *)
  let fwd_comp =
    let fwd = Train.forward loss in
    { fwd with asgns = Ir.Assignments.Block_comment ("padded_conv_forward", fwd.asgns) }
  in
  let tune_once ~with_forward () =
    let ctx = fresh () in
    let identity = Option.is_some (Context.timing_identity ctx) in
    let (grads, fwd_report), ios =
      SC.recording_cache_io (fun () ->
          let ctx, routine =
            Autotune.tune ~beam_width:1 ~rounds:1 ~repeats:1 ~cache_dir
              ~report:(fun r -> reports := r :: !reports)
              ctx comp bindings
          in
          let grads = grads_of (run_with ctx routine) in
          let fwd_report = ref None in
          if with_forward then
            ignore
              (Autotune.tune ~beam_width:1 ~rounds:0 ~repeats:1 ~cache_dir
                 ~report:(fun r -> fwd_report := Some r)
                 (fresh ()) fwd_comp bindings
                : Context.t * Context.routine);
          (grads, !fwd_report))
    in
    (grads, identity, ios, fwd_report)
  in
  let reference = untuned_grads () in
  (* An all-zero reference (a relu that killed everything) would let any routine match it. *)
  p_exists "the untuned gradients are not all zero" (Array.to_list reference) ~f:(fun g ->
      Float.(abs g > 1e-3));
  let tuned1, identity1, ios1, fwd_report = tune_once ~with_forward:true () in
  let tuned2, _identity2, ios2, _ = tune_once ~with_forward:false () in
  let r2, r1 =
    match !reports with [ r2; r1 ] -> (r2, r1) | _ -> failwith "expected two reports"
  in
  let approx_tol b = Float.(1e-4 *. (1. + abs b)) in
  p_close "pass 1's tuned gradients match the untuned compile" tuned1 reference ~tol:approx_tol;
  p_close "pass 2's gradients match the untuned compile" tuned2 reference ~tol:approx_tol;
  p "pass 1 searched" (match r1.Autotune.outcome with Autotune.Searched -> true | _ -> false);
  (* A search whose timings were refused for contention is not stored, by design
     ([Autotune.search_measurements_cacheable]); that is not this issue's question, so it closes the
     gate rather than passing. The device-identity gate and the refusal waiver are
     [Cache_plant]'s. *)
  let clean_search (r : Autotune.report) =
    (match r.Autotune.outcome with Autotune.Searched -> true | _ -> false)
    && r.Autotune.timings_contended = 0 && r.Autotune.candidates_timed > 0
  in
  let clean = clean_search r1 in
  Stdio.eprintf
    "pass 1 (not part of the golden): %d candidate(s) timed, %d timing(s) refused, %d cache \
     operation(s); pass 2: %d\n\
     %!"
    r1.Autotune.candidates_timed r1.Autotune.timings_contended (List.length ios1) (List.length ios2);
  let schedule_stores ios =
    List.filter ios ~f:(fun (io : SC.cache_io) ->
        (match io.op with SC.Store -> true | SC.Lookup -> false)
        && String.equal (Cache_plant.store_kind io) "schedule")
  in
  let cache_claim ios label b =
    if clean then Cache_plant.unless_unkeyed_or_refused ~identity:identity1 ios label b
    else gated ~aggregation:`Environment ~when_:false ~on:"a contended pass 1" label b
  in
  cache_claim ios1 "pass 1 committed a schedule-cache entry"
    (List.exists (schedule_stores ios1) ~f:(Fn.non Cache_plant.refused));
  cache_claim (ios1 @ ios2) "pass 2 replayed the cached winner"
    (match r2.Autotune.outcome with Autotune.Cache_replay -> true | _ -> false);
  (* The claim no refusal waives: pass 1's two clean, keyed searches were each bound to attempt a
     schedule store, and a cache that refuses every store fails here instead of skipping above. *)
  Cache_plant.not_every_store_refused
    ~expected:(identity1 && clean && Option.exists fwd_report ~f:clean_search)
    "not every schedule store pass 1 attempted was refused" ios1
