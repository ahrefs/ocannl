(* Diagnostic for the conv workloads' training step's default schedule: prints the fission segment
   census (kinds, launch geometry, per-nest loop extents and written nodes) on the configured
   backend, then times a few individual steps. Builds bench_conv.ml's model (Bench_conv_model); not
   part of the benchmark protocol. *)

open Base
open Ocannl
module IDX = Train.IDX
open Nn_blocks.DSL_modules
module Asgns = Ir.Assignments
module St = Safetensors
module H = Bench_harness

let cross_entropy_loss = Nn_blocks.cross_entropy_loss

let () =
  let fixture = Stdlib.Sys.getenv "BENCH_FIXTURE" in
  let materialize = H.env_flag "BENCH_MATERIALIZE" in
  let st = St.read fixture in
  let batch_size = H.meta_int st "batch_size" in
  let lr = Float.of_string (H.get_meta st "lr") in
  let x_nd = St.to_ndarray st "x" in
  let y_nd = St.to_ndarray st "y" in
  let total = (Ir.Ndarray.dims x_nd).(0) in
  let n_batches = total / batch_size in
  let xs = TDSL.rebatch ~l:"xs" x_nd () in
  let ys = TDSL.rebatch ~l:"ys" y_nd () in
  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let%op batch_x = xs @| batch_n in
  let%op batch_y = ys @| batch_n in
  let { Bench_conv_model.model; mapping } = Bench_conv_model.build st in
  let logits = model batch_x in
  let%op batch_loss =
    cross_entropy_loss ~spec:"...|v" ~normalize_by:!..batch_size () ~logits ~targets:batch_y
  in
  if materialize then Train.every_non_literal_materialized batch_loss;
  let step_comp =
    if H.env_flag "BENCH_FWD" then Train.forward batch_loss
    else
      let update = Train.grad_update batch_loss in
      let learning_rate = TDSL.O.( !. ) lr in
      let sgd = Train.sgd_update ~learning_rate batch_loss in
      Asgns.sequence [ update; sgd ]
  in
  let ctx = Context.auto () in
  let backend = Context.backend_name ctx in
  let limits = Context.hardware_limits ctx in
  let ctx = Train.init_params ctx bindings batch_loss in
  (* The first conv's kernel: [conv1] in the two-conv models, [stem] in the ResNet. *)
  let kc1 () =
    match
      Base.Set.find batch_loss.Tensor.params ~f:(fun p ->
          let dn = Ir.Tnode.debug_name p.Tensor.value in
          String.is_substring dn ~substring:"kernel_conv1"
          || String.is_substring dn ~substring:"kernel_stem")
    with
    | Some p -> p.Tensor.value
    | None -> assert false
  in
  let trace = H.env_flag "BENCH_TRACE" in
  let show ctx stage =
    if trace then
      let vs = Context.get_values ctx (kc1 ()) in
      Stdio.printf "kc1 %-14s sum: %.6g first: %.6g\n" stage
        (Array.fold vs ~init:0. ~f:( +. ))
        vs.(0)
  in
  show ctx "after-init";
  let late_inject = H.env_flag "BENCH_LATE_INJECT" in
  let ctx = if late_inject then ctx else H.inject ctx st batch_loss mapping in
  show ctx "after-inject";
  (* Only the lowering is wanted here (an explicit transform replaces the default pipeline, so the
     routine is discarded); the timed routine is compiled with the regular default pipeline
     below. *)
  let opt = H.capture_lowering ctx step_comp bindings in
  if H.env_flag "BENCH_CANON" then (
    let module LL = Ir.Low_level in
    let canon = Ir.Schedule_cache.canonicalize ~static_indices:[ batch_n ] opt in
    Stdio.printf "schedule-cache canonical form complete: %b\n" (Ir.Schedule_cache.complete canon);
    let free = Hash_set.create (module String) and shadowed = Hash_set.create (module String) in
    let opaque = ref 0 in
    (* The first node each re-bound loop's body references, to say whose loop nest it is. *)
    let pending = ref None in
    LL.Canonical_render.emit ~buf:(Buffer.create 4096)
      {
        LL.Canonical_render.emit_tn =
          (fun tn ->
            Option.iter !pending ~f:(fun s ->
                Hash_set.add shadowed (s ^ "@" ^ Ir.Tnode.debug_name tn);
                pending := None));
        emit_free_sym = (fun s -> Hash_set.add free (Ir.Indexing.symbol_ident s));
        on_bind_loop =
          (fun s ~id:_ ~shadowed:sh -> if sh then pending := Some (Ir.Indexing.symbol_ident s));
        mark_incomplete = (fun () -> Int.incr opaque);
        mma = LL.Canonical_render.Structural_mma;
        initial_tokens = [ (batch_n.Ir.Indexing.static_symbol, "s0") ];
      }
      opt.LL.llc;
    let show name set =
      Stdio.printf "  %s: %d [%s]\n" name (Hash_set.length set)
        (String.concat ~sep:" " (List.sort ~compare:String.compare (Hash_set.to_list set)))
    in
    show "free symbols" free;
    show "shadowed loop binders" shadowed;
    Stdio.printf "  opaque constructs: %d\n%!" !opaque);
  let promote_locals =
    match Stdlib.Sys.getenv_opt "BENCH_PROMOTE" with Some "0" -> Some false | _ -> None
  in
  H.print_census ?promote_locals ~backend ~limits ~static_indices:[ batch_n ] opt;
  (* Which accumulations the gh-ocannl-484 task-3 detector actually proposes on this graph — the
     seeded family can only reach the segments listed here, so this is what says whether the
     dominant segment of the census above is even a candidate. *)
  if H.env_flag "BENCH_SR_SITES" then (
    let sites = Autotune.split_reduce_sites opt in
    Stdio.printf "split-reduce sites detected: %d\n" (List.length sites);
    List.iter sites ~f:(fun s ->
        Stdio.printf "  %s: reduction extent %d, target cells %d, est. segment cost %d%s%s\n"
          (Ir.Tnode.debug_name s.Autotune.sr_target)
          s.Autotune.sr_red s.Autotune.sr_out s.Autotune.sr_cost
          (if s.Autotune.sr_dynamic then " (scatter)" else "")
          (* gh-ocannl-537: the conv-gradient sites are reachable only through an enabling
             interchange, so name it — a site listed with no swaps was splittable as lowered. *)
          (match s.Autotune.sr_swaps with
          | [] -> ""
          | swaps ->
              Printf.sprintf " (via %d swap%s: %s)" (List.length swaps)
                (if List.length swaps = 1 then "" else "s")
                (String.concat ~sep:" "
                   (List.map swaps ~f:(fun (o, i) ->
                        Ir.Indexing.symbol_ident i ^ "^" ^ Ir.Indexing.symbol_ident o))))));
  (* ...and, for the ones it does not propose, which rule rejected them. *)
  if H.env_flag "BENCH_SR_SITES" then H.print_split_reduce_verdicts opt;
  let t0 = Unix.gettimeofday () in
  let ctx, routine = Context.compile ctx step_comp bindings in
  let compile_s = Unix.gettimeofday () -. t0 in
  let ctx = if late_inject then H.inject ctx st batch_loss mapping else ctx in
  show ctx "after-compile";
  Stdio.printf "backend: %s  compile_s: %.3f\n" backend compile_s;
  (* Per-segment (per-layer) times: populate intermediates with one full step, then time each
     fission segment as its own routine. *)
  if H.env_flag "BENCH_SEG_TIMES" then (
    let batch_ref = IDX.find_exn routine.Context.bindings batch_n in
    batch_ref := 0;
    Train.run ctx routine;
    Context.sync ctx;
    H.time_segments ?promote_locals ~backend ~limits ~static_indices:[ batch_n ] ~ctx
      ~comp:step_comp ~bindings
      ~bind:(fun r -> IDX.find_exn r.Context.bindings batch_n := 0)
      opt);
  (if H.env_flag "BENCH_STEPS" then
     let batch_ref = IDX.find_exn routine.Context.bindings batch_n in
     for step = 0 to 2 do
       batch_ref := step % n_batches;
       let t0 = Unix.gettimeofday () in
       Train.run ctx routine;
       Context.sync ctx;
       Stdio.printf "step %d: %.1f ms\n" step ((Unix.gettimeofday () -. t0) *. 1000.);
       Stdio.Out_channel.flush Stdio.stdout
     done);
  if H.env_flag "BENCH_PROBE" then (
    let batch_ref = IDX.find_exn routine.Context.bindings batch_n in
    batch_ref := 0;
    Train.run ctx routine;
    Context.sync ctx;
    let dump name tn n =
      let vs = Context.get_values ctx tn in
      Stdio.printf "%s[0..%d]: %s\n" name (n - 1)
        (String.concat ~sep:" "
           (Array.to_list (Array.map (Array.sub vs ~pos:0 ~len:n) ~f:(Printf.sprintf "%.5g"))));
      Stdio.printf "%s sum: %.6g\n" name (Array.fold vs ~init:0. ~f:( +. ))
    in
    ignore logits;
    dump "loss" batch_loss.Tensor.value 1;
    Hashtbl.iter_keys opt.Ir.Low_level.traced_store ~f:(fun tn ->
        match Context.get_values ctx tn with
        | vs ->
            Stdio.printf "node %s sum: %.6g first: %.6g\n" (Ir.Tnode.debug_name tn)
              (Array.fold vs ~init:0. ~f:( +. ))
              vs.(0);
            if H.env_flag "BENCH_DUMP" then (
              let f =
                Stdio.Out_channel.create
                  (Printf.sprintf "/tmp/probe_%s.txt" (Ir.Tnode.debug_name tn))
              in
              Array.iter vs ~f:(fun v -> Stdio.Out_channel.fprintf f "%.9g\n" v);
              Stdio.Out_channel.close f)
        | exception _ -> ());
    Base.Set.iter batch_loss.Tensor.params ~f:(fun p ->
        let tn = p.Tensor.value in
        let vs = Context.get_values ctx tn in
        Stdio.printf "param %s sum: %.6g first: %.6g\n" (Ir.Tnode.debug_name tn)
          (Array.fold vs ~init:0. ~f:( +. ))
          vs.(0)))
