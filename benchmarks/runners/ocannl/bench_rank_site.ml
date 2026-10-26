(* gh-ocannl-1442: ranking agreement at one site -- what the cost model's ranking score says of each
   whole-routine candidate beside what the autotuner measures of it.

   [BENCH_SITE] picks the computation: - [wgrad] (default): gpt2_mini_train's [l*_ffn_w2] weight
   gradient, the layer bench_wgrad times, at [BENCH_SITE_DIMS=b,s,o,i] (default [8,128,256,1024]); -
   [fp8mm]: the gh-ocannl-1190 site, an fp8 [m x k] by [k x n] matmul with an f32 result, at
   [BENCH_SITE_DIMS=m,n,k] (default [8192,32,4096]); - [qkv]: gpt2_mini's q/k/v projection
   [d[b,s,h,e] += w[h,e,k] * x[b,s,k]], whose column tile the coalesced branch's [coal-n] layout
   merges across heads (gh-ocannl-1165), at [BENCH_SITE_DIMS=b,s,h,e,k] (default [8,128,8,32,256]);
   - [outproj]: its out projection [d[b,s,j] += w[j,h,e] * x[b,s,h,e]], whose contraction the
   [coal-k] layout merges (gh-ocannl-1279), at [BENCH_SITE_DIMS=b,s,j,h,e] (default
   [8,128,256,8,32]).

   Always: one [rank-site: explain] line per candidate [Autotune.model_explain ~coalesced:true]
   lists -- the incumbent, then every whole-routine sketch leaf, then the coalesced branch's leaves
   the tuner seeds beside them (gh-ocannl-1278; [admitted] there says whether the leaf's uncoalesced
   form is) -- with its ranking score and the score's two legs under the configuration in force
   (pass the box's fitted [model_*] constants, or [none] for the ceilings), and the accepted
   tile-MMA work and device gathers the compute leg divides by their rates (gh-ocannl-1495). With
   [BENCH_TUNE=1], the same computation then goes through [Autotune.tune] with seed candidates only
   ([~rounds:0]) and no schedule cache: every timed candidate appends a calibration row (label and
   measured time) to config [autotune_calibration_file], which the driver joins with the explain
   lines by label (benchmarks/gh1442_agreement.py). The labels are the ones [autotune_log] prints,
   which [model_explain] uses too. [anchor=true] marks the leaf [model_default] anchors its walk at
   under seeding, the seeded plan's own sketch (gh-ocannl-1477): a leaf must rank strictly below it.
   Nothing here is a value check: the tuner's own validation and the timings are the point. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules

let () =
  let site = Option.value (Stdlib.Sys.getenv_opt "BENCH_SITE") ~default:"wgrad" in
  let dims default =
    Option.value (Stdlib.Sys.getenv_opt "BENCH_SITE_DIMS") ~default
    |> String.split ~on:',' |> List.map ~f:Int.of_string
  in
  let tune = Option.equal String.equal (Stdlib.Sys.getenv_opt "BENCH_TUNE") (Some "1") in
  let cyc n m stride = Array.init n ~f:(fun c -> Float.of_int (1 + (c % m)) *. stride) in
  let cell idx = Float.of_int (1 + (Array.fold idx ~init:0 ~f:( + ) % 7)) *. 0.125 in
  let name, comp =
    match site with
    | "wgrad" ->
        let nb, ns, no, ni =
          match dims "8,128,256,1024" with
          | [ b; s; o; i ] -> (b, s, o, i)
          | _ -> failwith "BENCH_SITE_DIMS must be b,s,o,i for the wgrad site"
        in
        (* The contraction alone, as one routine: inside a training step's backprop the GPU search
           times per-segment candidates only, which no whole-routine leaf of [model_explain]
           names. *)
        let k = nb * ns in
        let dy = NTDSL.ndarray (cyc (k * no) 7 0.25) ~label:[ "rs_dy" ] ~output_dims:[ k; no ] () in
        let x = NTDSL.ndarray (cyc (k * ni) 5 0.125) ~label:[ "rs_x" ] ~output_dims:[ k; ni ] () in
        let%op dw = dy +* "ko; ki => oi" x in
        Train.set_materialized dw.Tensor.value;
        (Printf.sprintf "rs_wgrad_b%d" nb, Train.forward dw)
    | "fp8mm" ->
        let m, n, k =
          match dims "8192,32,4096" with
          | [ m; n; k ] -> (m, n, k)
          | _ -> failwith "BENCH_SITE_DIMS must be m,n,k for the fp8mm site"
        in
        let fa idx = Float.of_int (1 + (((idx.(0) * 17) + (idx.(1) * 13)) % 7 / 3 % 2)) in
        let fb idx = Float.of_int (1 + (((idx.(0) * 13) + (idx.(1) * 19)) % 11 / 5 % 2)) in
        let a = NTDSL.init ~l:"rs_a" ~prec:Ir.Ops.fp8 ~i:[ k ] ~o:[ m ] ~f:fa () in
        let b = NTDSL.init ~l:"rs_b" ~prec:Ir.Ops.fp8 ~i:[ n ] ~o:[ k ] ~f:fb () in
        let%op t = a * b in
        Ir.Tnode.update_prec t.Tensor.value Ir.Ops.single;
        ("rs_fp8mm", Train.forward t)
    | "qkv" ->
        let nb, ns, nh, ne, nk =
          match dims "8,128,8,32,256" with
          | [ b; s; h; e; k ] -> (b, s, h, e, k)
          | _ -> failwith "BENCH_SITE_DIMS must be b,s,h,e,k for the qkv site"
        in
        let w = NTDSL.init ~l:"rs_wqkv" ~prec:Ir.Ops.single ~o:[ nh; ne ] ~i:[ nk ] ~f:cell () in
        let x = NTDSL.init ~l:"rs_xqkv" ~prec:Ir.Ops.single ~b:[ nb; ns ] ~o:[ nk ] ~f:cell () in
        let%op d = w * x in
        Train.set_materialized d.Tensor.value;
        (Printf.sprintf "rs_qkv_b%d" nb, Train.forward d)
    | "outproj" ->
        let nb, ns, nj, nh, ne =
          match dims "8,128,256,8,32" with
          | [ b; s; j; h; e ] -> (b, s, j, h, e)
          | _ -> failwith "BENCH_SITE_DIMS must be b,s,j,h,e for the outproj site"
        in
        let w = NTDSL.init ~l:"rs_wout" ~prec:Ir.Ops.single ~o:[ nj ] ~i:[ nh; ne ] ~f:cell () in
        let x =
          NTDSL.init ~l:"rs_xout" ~prec:Ir.Ops.single ~b:[ nb; ns ] ~o:[ nh; ne ] ~f:cell ()
        in
        let%op d = w * x in
        Train.set_materialized d.Tensor.value;
        (Printf.sprintf "rs_outproj_b%d" nb, Train.forward d)
    | other -> failwith ("BENCH_SITE must be wgrad, fp8mm, qkv or outproj, got " ^ other)
  in
  let ctx = Context.auto () in
  let ms = Option.value_map ~default:"n/a" ~f:(Printf.sprintf "%.6f") in
  let sum =
    Option.value_map ~default:"n/a" ~f:(fun xs ->
        Printf.sprintf "%.6e" (List.fold xs ~init:0. ~f:Float.( + )))
  in
  List.iter
    (Autotune.model_explain ~name ~coalesced:true ctx comp Ir.Indexing.Empty).Autotune.x_routine
    ~f:(fun (label, ex) ->
      Stdio.printf
        "rank-site: explain site=%s label=%S rank_ms=%s compute_ms=%s memory_ms=%s floor_ms=%s \
         admitted=%b anchor=%b fills=%s mma_work=%s gathers=%s\n\
         %!"
        site label (ms ex.Autotune.ex_rank_ms) (ms ex.ex_compute_ms) (ms ex.ex_memory_ms)
        (ms ex.ex_floor_ms) ex.ex_admitted ex.ex_anchor
        (Option.value_map ex.ex_fills ~default:"n/a" ~f:(fun fills ->
             String.concat ~sep:"," (List.map fills ~f:(Printf.sprintf "%.4f"))))
        (* gh-ocannl-1495: summed over kernels, as the compute leg divides them by their rates --
           what [gh1442_agreement.py --fit-mma] fits the tile-MMA and device-gather rates from. *)
        (sum ex.ex_mma_work)
        (sum ex.ex_device_gathers);
      (* gh-ocannl-1600: per kernel, its two legs, its fill-weighted excess memory transactions
         (32-, 64- and 128-byte sectors), its uncovered accesses, workgroup bytes and block threads
         -- what [gh1442_agreement.py --fit-txn] re-scores offline (the score sums each kernel's
         larger leg, so the routine's summed legs cannot rebuild it). *)
      Option.iter ex.ex_kernels ~f:(fun kernels ->
          Stdio.printf "rank-site: kernels site=%s label=%S kernels=%s\n%!" site label
            (String.concat ~sep:";"
               (List.map kernels ~f:(fun (k : Autotune.explain_kernel) ->
                    Printf.sprintf
                      "c=%s,m=%s,txn=%.6e,txn64=%s,txn128=%s,unc=%d,wg=%d,thr=%d,occ=%.6f"
                      (ms k.ek_compute_ms) (ms k.ek_memory_ms) k.ek_transactions
                      (Option.value_map ~default:"n/a" ~f:(Printf.sprintf "%.6e")
                         (List.Assoc.find k.ek_transactions_by_sector 64 ~equal:Int.equal))
                      (Option.value_map ~default:"n/a" ~f:(Printf.sprintf "%.6e")
                         (List.Assoc.find k.ek_transactions_by_sector 128 ~equal:Int.equal))
                      k.ek_transactions_uncovered k.ek_workgroup_bytes k.ek_block_threads
                      k.ek_occupancy)))));
  (* [BENCH_PICK=1] (gh-ocannl-1278): what [Autotune.model_default] actually picks here under the
     configuration in force, read off the compile itself. Nothing is timed. [seeded_label] names the
     seeded plan under the same configuration as the tuner labels its seed ([F_seeded[...]], the
     plan's content digest included, gh-ocannl-1477): a pick of ["seeded"] is joined to a timing
     only where the timed seed is this very plan -- a profile's [auto] gate can seed a different
     plan than the timing cell's pinned threshold. *)
  (if Option.equal String.equal (Stdlib.Sys.getenv_opt "BENCH_PICK") (Some "1") then
     let seeded_label =
       let limits = Context.hardware_limits ctx in
       let caps = Context.codegen_capabilities ctx in
       match
         Autotune.seeded_plan ~capabilities:caps ~backend:(Context.backend_name ctx) ~limits
           ~accum_prec:caps.Ir.Backend_intf.accum_prec ~static_indices:[]
           (Context.lowered_for_decisions ~name:(name ^ "_pick") ctx comp Ir.Indexing.Empty)
       with
       | Some plan -> Autotune.seeded_plan_label plan
       | None -> "none"
     in
     let pick = ref None in
     let pick_ctx, pick_routine =
       Autotune.model_default ~name:(name ^ "_pick")
         ~report:(fun c -> pick := Some c)
         ctx comp Ir.Indexing.Empty
     in
     (* gh-ocannl-1479: the pick timed by itself, beside the configuration's untuned plan
        ([Autotune.seeded_default]: the seeded plan, or the default where nothing seeds) timed the
        same way -- a pick the timing cell never timed still gets a measured comparison. Best of
        [BENCH_PICK_REPEATS] (default 20) synced runs after a warm-up. *)
     let repeats =
       Option.value_map (Stdlib.Sys.getenv_opt "BENCH_PICK_REPEATS") ~default:20 ~f:Int.of_string
     in
     let time run_ctx routine =
       Context.sync (Context.run run_ctx routine);
       let best = ref Float.infinity in
       for _ = 1 to repeats do
         let t0 = Mtime_clock.counter () in
         Context.sync (Context.run run_ctx routine);
         best := Float.min !best (Mtime.Span.to_float_ns (Mtime_clock.count t0) /. 1e6)
       done;
       !best
     in
     let pick_ms = time pick_ctx pick_routine in
     let untuned_ctx, untuned_routine =
       Autotune.seeded_default ~name:(name ^ "_untuned") ctx comp Ir.Indexing.Empty
     in
     let untuned_ms = time untuned_ctx untuned_routine in
     match !pick with
     | None -> Stdio.printf "rank-site: pick site=%s no report\n%!" site
     | Some c ->
         Stdio.printf
           "rank-site: pick site=%s label=%S rank_ms=%s seeded_label=%S pick_ms=%.6f \
            untuned_ms=%.6f pick_over_untuned=%.3f\n\
            %!"
           site c.Autotune.mc_label (ms c.mc_rank_ms) seeded_label pick_ms untuned_ms
           (pick_ms /. untuned_ms));
  if tune then
    let report = ref None in
    let _ctx, _routine =
      Autotune.tune ~search:true ~rounds:0 ~cache_dir:"" ~name
        ~report:(fun r -> report := Some r)
        ctx comp Ir.Indexing.Empty
    in
    match !report with
    | None -> Stdio.printf "rank-site: tune site=%s no report\n" site
    | Some r ->
        Stdio.printf "rank-site: tune site=%s crowned=%S best_ms=%.6f timed=%d\n%!" site
          r.Autotune.best_label r.Autotune.best_ms r.Autotune.candidates_timed
