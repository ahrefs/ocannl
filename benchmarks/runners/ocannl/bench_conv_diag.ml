(* Diagnostic for the conv workloads' training step's default schedule: prints the fission segment
   census (kinds, launch geometry, per-nest loop extents and written nodes) on the configured
   backend, then times a few individual steps. Builds bench_conv.ml's model (Bench_conv_model); not
   part of the benchmark protocol.

   Two modes replace the rest of the run (gh-ocannl-1490, Bench_seed_parity): - BENCH_CENSUS_ONLY=1
   prints the autotuner's seeding census for the step -- every lowering a search seeds, the whole
   routine and each keyed fission segment -- and exits, no search started. - BENCH_PARITY=1 executes
   each conv seed (BENCH_PARITY_SEEDS=all: every seed) of each keyed segment, compiled as the
   search's per-segment candidate, for one step against the default pipeline's step, and prints a
   verdict per seed. BENCH_PARITY_TOL (default 1e-3) bounds the worst normalized difference. A seed
   computing on a tf32 arm is compared with the default pipeline whose matching accumulations round
   their multiplicands to tf32 (gh-ocannl-1602); BENCH_PARITY_ROUND=0 keeps the plain reference, the
   control showing the rounding is load-bearing. A seed whose decisions (max-pool's selections, relu
   gates) flipped against its reference is judged by a reference replaying them, within a bound on
   the flipped cells; BENCH_PARITY_REPLAY=0 turns that off. BENCH_PARITY_PERTURB=1 is the negative
   control: every seeded segment's code is perturbed, and each perturbed seed must be reported a
   mismatch. Exits 4 when a seed fails (a mismatch, an unjudged seed, or under the control a
   perturbation not detected), 5 when no seed was executed, 0 otherwise. *)

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
  (* gh-ocannl-1490: the search's seeding census, no search started. *)
  if H.env_flag "BENCH_CENSUS_ONLY" then (
    List.iter
      (Bench_seed_parity.census_lines ctx ~static_indices:[ batch_n ] opt)
      ~f:Stdio.print_endline;
    Stdlib.exit 0);
  (* gh-ocannl-1490: every selected seed of every keyed segment the search seeds, executed against
     the default pipeline's step; exit 4 when any seed fails. *)
  if H.env_flag "BENCH_PARITY" then (
    let perturb = H.env_flag "BENCH_PARITY_PERTURB" in
    let on_off name =
      match Stdlib.Sys.getenv_opt name with
      | None | Some ("" | "1") -> true
      | Some "0" -> false
      | Some other -> failwith (name ^ " must be 0 or 1; found: " ^ other)
    in
    let round = on_off "BENCH_PARITY_ROUND" and replay = on_off "BENCH_PARITY_REPLAY" in
    let select =
      match Stdlib.Sys.getenv_opt "BENCH_PARITY_SEEDS" with
      | None | Some ("" | "conv") -> Bench_seed_parity.conv_seed
      | Some "all" -> Bench_seed_parity.any_seed
      | Some other -> failwith ("BENCH_PARITY_SEEDS must be conv or all; found: " ^ other)
    in
    let tol =
      match Stdlib.Sys.getenv_opt "BENCH_PARITY_TOL" with
      | Some t -> Float.of_string t
      | None -> Bench_seed_parity.default_tol ()
    in
    let n = ref 0 in
    let reference, results =
      Bench_seed_parity.parity ~tol ~perturb ~round ~replay ~select
        ?detail:
          (if H.env_flag "BENCH_PARITY_DETAIL" then Some (fun l -> Stdio.printf "%s\n%!" l)
           else None)
        ~on_result:(fun r ->
          Int.incr n;
          Stdio.printf "parity seed %d %s\n%!" !n (Bench_seed_parity.verdict_line r))
        ctx step_comp bindings
        ~bind:(fun r -> IDX.find_exn r.Context.bindings batch_n := 0)
        ~static_indices:[ batch_n ] opt
    in
    let failed = List.count results ~f:(fun r -> not (Bench_seed_parity.passed ~perturb r)) in
    let rounded =
      List.count results ~f:(fun r -> not (List.is_empty r.Bench_seed_parity.rounded))
    in
    Stdio.printf
      "parity (%s): backend %s, %d seeds, %d failed at tolerance %g; reference: the default \
       pipeline%s, %d nodes compared, %d restored, %d poisoned%s; %s\n"
      (if perturb then "negative control, every seed perturbed" else "seeds as proposed")
      backend (List.length results) failed tol
      (if not round then " (tf32 rounding off)"
       else
         Printf.sprintf ", tf32-rounded for %d seeds (%d rounded compiles)" rounded
           reference.ref_rounded_compiles)
      reference.ref_compared reference.ref_restored reference.ref_poisoned
      (match reference.ref_poison_unsafe with
      | [] -> ""
      | l -> ", not poisoned: " ^ String.concat ~sep:" " l)
      (let replayed =
         List.filter_map results ~f:(fun r ->
             if r.Bench_seed_parity.replayed then Some r.Bench_seed_parity.flipped else None)
       in
       Printf.sprintf
         "decision nodes %d (%d cells)%s; replay %s: %d seeds replayed %d flipped cells, at most \
          %d per seed (bound %d), %d replay compiles; masked cells: 0"
         (List.length reference.ref_decisions)
         reference.ref_decision_cells
         (match reference.ref_unreplayable with
         | [] -> ""
         | l -> ", unreplayable: " ^ String.concat ~sep:"; " l)
         (if replay then "on" else "off")
         (List.length replayed)
         (List.sum (module Int) replayed ~f:Fn.id)
         (List.fold replayed ~init:0 ~f:Int.max)
         (Bench_seed_parity.replay_bound ~decision_cells:reference.ref_decision_cells ())
         reference.ref_replay_compiles);
    let executed =
      List.count results ~f:(fun r ->
          match r.Bench_seed_parity.verdict with Match _ | Mismatch _ -> true | _ -> false)
    in
    if executed = 0 then Stdio.printf "parity: no seed was executed, so nothing was checked\n";
    Stdlib.exit (if failed > 0 then 4 else if executed = 0 then 5 else 0));
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
        Stdio.printf "  %s: reduction extent %d, target cells %d, est. segment cost %d%s%s%s\n"
          (Ir.Tnode.debug_name s.Autotune.sr_target)
          s.Autotune.sr_red s.Autotune.sr_out s.Autotune.sr_cost
          (if s.Autotune.sr_dynamic then " (scatter)" else "")
          (* gh-ocannl-1414: a site split over two reduction loops merged into one. *)
          (match s.Autotune.sr_coalesce with
          | None -> ""
          | Some (o, i) ->
              Printf.sprintf " (coalesced %s*%s)" (Ir.Indexing.symbol_ident o)
                (Ir.Indexing.symbol_ident i))
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
