(* The analytic cost model's selection half (gh-ocannl-491 tasks 3-4): candidate scoring
   ([Autotune.model_score]), the order-preserving keep-fraction pre-filter
   ([Autotune.model_prefilter]), the seed pre-filter inside [Autotune.tune] (report fields
   [model_scored] / [model_pruned]), and the model-picked untuned default
   ([Autotune.model_default]).

   Run by a dune rule (not a test stanza) with --ocannl_backend=cc and the model_peak_* envelope
   overrides pinned on the command line: the C backends carry no advisory envelope constants, so the
   overrides are what make roofline scoring live here — deterministically, on every entry of the
   backend test matrix. The override values are chosen bandwidth-dominant (peak_flops huge,
   peak_memory_bandwidth tiny), so candidate ranking is decided by the byte counts, which differ
   across tile geometries. All printed values are booleans that hold regardless of the machine's
   SIMD width (which changes which sketch seeds exist, but not the properties asserted). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
open Verdict.Claims

let approx_tol _ = 1e-4
let approx a b = Float.(abs (a - b) < approx_tol b)

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let rec first_loop (llc : LL.t) =
  match llc with
  | LL.Seq (a, b) -> ( match first_loop a with Some r -> Some r | None -> first_loop b)
  | LL.For_loop { index; body; _ } -> Some (index, body)
  | _ -> None

let first_loop_exn llc = Option.value_exn ~here:[%here] (first_loop llc)

let capture comp =
  let captured = ref None in
  let ctx = Context.auto () in
  let _ctx, _routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        captured := Some opt;
        [ opt ])
      ctx comp Ir.Indexing.Empty
  in
  Option.value_exn ~here:[%here] !captured

(* 16x16 x 16x16 matmul values, for correctness checks below. *)
let n = 16

let mav =
  Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:7 ~offset:0. ~stride:0.5)

let mbv =
  Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:11 ~offset:(-4.) ~stride:1.)

let mm_expected =
  Array.init (n * n) ~f:(fun idx ->
      let i = idx / n and j = idx % n in
      let acc = ref 0. in
      for k = 0 to n - 1 do
        acc := !acc +. (mav.((i * n) + k) *. mbv.((k * n) + j))
      done;
      !acc)

let () =
  (* --- The pre-filter over fabricated scores: pure, envelope-independent --- *)
  let items = [ ("a", Some 3.); ("b", None); ("c", Some 1.); ("d", Some 2.); ("e", None) ] in
  let kept = Autotune.model_prefilter ~keep_fraction:0.5 items in
  let names = List.map kept ~f:fst in
  let mem x = List.mem names x ~equal:String.equal in
  p "prefilter keeps unscored candidates (no-coverage exemption)" (mem "b" && mem "e");
  p "prefilter keeps ceil(0.5 * 3) = 2 scored candidates, dropping the worst"
    (mem "c" && mem "d" && not (mem "a"));
  p "prefilter preserves candidate order" (List.equal String.equal names [ "b"; "c"; "d"; "e" ]);
  p "keep_fraction 1 keeps everything"
    (List.length (Autotune.model_prefilter ~keep_fraction:1. items) = 5);
  let ties = [ (1, Some 2.); (2, Some 1.); (3, Some 1.) ] in
  p "ties at the cutoff are all kept (order-independence)"
    (List.equal Int.equal
       (List.map (Autotune.model_prefilter ~keep_fraction:0.34 ties) ~f:fst)
       [ 2; 3 ]);
  p "at least one scored candidate always survives"
    (List.length (Autotune.model_prefilter ~keep_fraction:0.0001 ties) >= 1)

let () =
  (* --- model_score on a real lowering (envelope from the command-line overrides) --- *)
  let ma = TDSL.ndarray mav ~label:[ "sc_a" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "sc_b" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op mc = ma * mb in
  let opt = capture (named "score_mm" (Train.forward mc)) in
  let limits = Ir.Backend_intf.no_hardware_limits in
  let s_empty = Autotune.model_score ~static_indices:[] ~limits opt [] in
  p "empty schedule scoreable under the override envelope"
    (match s_empty with Some s -> Float.(s > 0.) | None -> false);
  let sym_i, body = first_loop_exn opt.LL.llc in
  let split_op, _o, _i = Sched.split ~axis:sym_i ~factor:4 ~outer:LL.Serial ~inner:LL.Serial in
  let s_split = Autotune.model_score ~static_indices:[] ~limits opt [ split_op ] in
  p "pure loop restructuring (Split) leaves the score unchanged"
    (match (s_empty, s_split) with Some a, Some b -> approx a b | _ -> false);
  let sym_j, _ = first_loop_exn body in
  (* Swapping a pair in the wrong nesting order fails to apply: no coverage, not an error. *)
  let bad = [ Sched.Swap { outer = sym_j; inner = sym_i } ] in
  p "an inapplicable schedule has no score (kept, only measured)"
    (Option.is_none (Autotune.model_score ~static_indices:[] ~limits opt bad))

let () =
  (* --- The seed pre-filter inside tune --- *)
  let ta = TDSL.ndarray mav ~label:[ "tn_a" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let tb = TDSL.ndarray mbv ~label:[ "tn_b" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op tc = ta * tb in
  let comp = named "prefilter_mm" (Train.forward tc) in
  let reports = ref [] in
  let tune keep_fraction =
    let ctx = Context.auto () in
    let ctx, routine =
      Autotune.tune ~beam_width:1 ~rounds:0 ~repeats:1 ~cache_dir:"" ~keep_fraction
        ~report:(fun r -> reports := r :: !reports)
        ctx comp Ir.Indexing.Empty
    in
    let ctx = Context.run ctx routine in
    Context.get_values ctx tc.Tensor.value
  in
  let got_all = tune 1.0 in
  let got_cut = tune 0.01 in
  let r_cut, r_all =
    match !reports with [ b; a ] -> (b, a) | _ -> failwith "expected two reports"
  in
  p "keep_fraction 1 turns the pre-filter off"
    (r_all.Autotune.model_scored = 0 && r_all.Autotune.model_pruned = 0);
  p "pre-filter scored the sketch candidates" (r_cut.Autotune.model_scored >= 2);
  p "pre-filter pruned the worse-scored sketches before timing"
    (r_cut.Autotune.model_pruned >= 1 && r_cut.Autotune.model_pruned < r_cut.Autotune.model_scored);
  p "pruned search timed fewer candidates than the full search"
    (r_cut.Autotune.candidates_timed <= r_all.Autotune.candidates_timed);
  p "baseline still timed under aggressive pruning" (r_cut.Autotune.candidates_timed >= 1);
  p_close "tuned values correct without the pre-filter" got_all mm_expected ~tol:approx_tol;
  p_close "tuned values correct with the pre-filter" got_cut mm_expected ~tol:approx_tol

let () =
  (* --- A builder refusal inside the seed pre-filter (gh-ocannl-1437) ---

     Below [keep_fraction] 1 the pre-filter builds every sketch to score it, before any candidate's
     failure boundary, so a typed builder refusal ([Sketch_families.refuse], gh-ocannl-1390) used to
     escape it and end the search. Injected through the builder's fault seam: the first parameter
     set the builder sees is refused on every build. Under [keep_fraction] 0.5 the search completes,
     the refusal is a counted decline under the builder's own cause, and no candidate carrying the
     refused parameters is compiled; under 1 (the control) the same parameters do reach a candidate
     compile and are declined there -- so the label match below can find them. *)
  let ra = TDSL.ndarray mav ~label:[ "rf_a" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let rb = TDSL.ndarray mbv ~label:[ "rf_b" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op rc = ra * rb in
  let comp = named "prefilter_refusal_mm" (Train.forward rc) in
  let injected = "gh-ocannl-1437: injected sketch builder refusal" in
  let run keep_fraction =
    let target = ref None and attempts = ref [] and report = ref None in
    let fault p =
      let refused =
        match !target with
        | Some q -> q
        | None ->
            target := Some p;
            p
      in
      if Poly.equal p refused then
        raise
          (Ir.Schedule_outcome.Cause_at
             ( Ir.Schedule_outcome.Transform,
               Ir.Schedule_outcome.Illegal_schedule { check = "Autotune.sketch"; detail = injected }
             ))
    in
    let result =
      Autotune.Seam.with_value Autotune.on_candidate_attempt (fun label ->
          attempts := label :: !attempts)
      @@ fun () ->
      Autotune.with_sketch_builder_fault fault @@ fun () ->
      match
        Autotune.tune ~beam_width:1 ~rounds:0 ~repeats:1 ~cache_dir:"" ~keep_fraction
          ~report:(fun r -> report := Some r)
          (Context.auto ()) comp Ir.Indexing.Empty
      with
      | ctx, routine -> Ok (Context.get_values (Context.run ctx routine) rc.Tensor.value)
      | exception exn -> Error (Exn.to_string exn)
    in
    Result.iter_error result ~f:(fun e ->
        Stdio.eprintf "keep %g: the search raised %s (not part of the golden)\n%!" keep_fraction e);
    (result, !report, Option.map !target ~f:Autotune.sketch_params_label, List.rev !attempts)
  in
  let carries refused label =
    match refused with Some s -> String.is_substring label ~substring:s | None -> false
  in
  let counted (r : Autotune.report option) =
    match r with
    | None -> []
    | Some r ->
        List.filter r.Autotune.declines ~f:(fun d ->
            Ir.Schedule_outcome.equal_rejection_key d.Autotune.key
              (Ir.Schedule_outcome.Illegal_schedule_key "Autotune.sketch")
            && List.mem d.Autotune.sample_details injected ~equal:String.equal)
  in
  let cut, cut_report, cut_refused, cut_attempts = run 0.5 in
  p "a typed builder refusal in the pre-filter does not end the search"
    (Result.is_ok cut
    &&
    match cut_report with
    | Some r -> Poly.equal r.Autotune.outcome Autotune.Searched && r.Autotune.model_scored >= 1
    | None -> false);
  p_exists "the pre-filter's refusal is a counted decline under the builder's cause"
    (counted cut_report) ~f:(fun d -> d.Autotune.count >= 1);
  p_none "no candidate carrying the refused parameters is compiled" cut_attempts
    ~f:(carries cut_refused);
  p_close "the search with a refused candidate computes correct values"
    (Result.ok cut |> Option.value ~default:[||])
    mm_expected ~tol:approx_tol;
  let all, all_report, all_refused, all_attempts = run 1.0 in
  p "control: without the pre-filter the same refusal is declined at candidate compile"
    (Result.is_ok all && not (List.is_empty (counted all_report)));
  p_exists "control: there, a candidate carrying the refused parameters is compiled" all_attempts
    ~f:(carries all_refused)

let () =
  (* --- The model-picked untuned default --- *)
  let da = TDSL.ndarray mav ~label:[ "df_a" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let db = TDSL.ndarray mbv ~label:[ "df_b" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op dc = da * db in
  let comp = named "model_default_mm" (Train.forward dc) in
  let choice = ref None in
  let ctx = Context.auto () in
  let ctx, routine =
    Autotune.model_default ~report:(fun r -> choice := Some r) ctx comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx dc.Tensor.value in
  p_close "model_default returns a working routine with correct values" got mm_expected
    ~tol:approx_tol;
  let r = Option.value_exn ~here:[%here] !choice in
  p "model_default scored candidates (default pipeline included)" (r.Autotune.mc_scored >= 1);
  p "model_default reports a choice with its model score"
    ((not (String.is_empty r.Autotune.mc_label))
    && match r.Autotune.mc_model_ms with Some m -> Float.(m > 0.) | None -> false)

let () =
  (* --- Unbuildable candidates are excluded from the ranking (gh-ocannl-522) ---

     The roofline cannot see that a schedule will not compile, and it rates the tensorized families
     best, so it used to crown one and — once gh-ocannl-519 stopped that from killing the process —
     degrade to the default, making the gate a no-op exactly where it looked most promising. With
     [validate_parallel] applied to each candidate's scheduled form during scoring, the dead ones
     drop out and the argmin is taken over what can actually be built.

     A 128x128 matmul is the smallest square size at which the C backend's packed mma sketches both
     get seeded and beat the default under this rule's bandwidth-dominant envelope.

     The dead candidates here were the CPU tensorized flavors binding a hardware dimension (the
     whole-triple form's [Workgroup] lane, the packed Grid shapes' row-block loop) beside the relu
     tail, a companion they left uncovered. Since gh-ocannl-1561 the family tree refutes them at
     seeding, so the filter's population on this routine is empty: the claim is now that the search
     never meets an unbuildable candidate here, and the argmin claim below is the property the
     filter existed for. *)
  let m = 128 in
  let av =
    Array.init (m * m) ~f:(Ll_test.cycle_flat ~dims:[| m; m |] ~modulus:7 ~offset:0. ~stride:0.5)
  in
  let bv =
    Array.init (m * m) ~f:(Ll_test.cycle_flat ~dims:[| m; m |] ~modulus:11 ~offset:(-4.) ~stride:1.)
  in
  let expected =
    Array.init (m * m) ~f:(fun idx ->
        let i = idx / m and j = idx % m in
        let acc = ref 0. in
        for k = 0 to m - 1 do
          acc := !acc +. (av.((i * m) + k) *. bv.((k * m) + j))
        done;
        !acc)
  in
  let va = TDSL.ndarray av ~label:[ "vd_a" ] ~input_dims:[ m ] ~output_dims:[ m ] () in
  let vb = TDSL.ndarray bv ~label:[ "vd_b" ] ~input_dims:[ m ] ~output_dims:[ m ] () in
  let%op vc = relu (va * vb) in
  let comp = named "model_default_valid_mm" (Train.forward vc) in
  let choice = ref None in
  let ctx = Context.auto () in
  let ctx, routine =
    Autotune.model_default ~report:(fun r -> choice := Some r) ctx comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx vc.Tensor.value in
  let r = Option.value_exn ~here:[%here] !choice in
  p "no candidate is unbuildable: the seeding refutes the flavors the relu tail would fail"
    (r.Autotune.mc_rejected = 0);
  p "the surviving argmin is a schedule, not the default fallback"
    (not (String.equal r.Autotune.mc_label "default"));
  p_close "the picked schedule computes correct values" got expected ~tol:(fun _ -> 1e-2)

let () =
  (* --- The ranking score past and below capacity (gh-ocannl-1253) ---

     A dense layer's weight gradient, [w.grad[o,i] += dy[b,s,o] * x[b,s,i]], as backprop lowers it,
     at two sizes under ONE cache capacity (the rule's [--ocannl_model_cache_capacity]): the large
     one's working set exceeds it, so its re-reads are charged and a reuse-tiled sketch ranks below
     the default; the small one's fits, so the scores are the distinct-cell roofline, on which the
     default wins here, and the pick stays the default. Both claims have teeth, checked by hand at
     these sizes: with the charge off ([none]) the large leg picks the default too, and at a quarter
     of the capacity (2^18) the small leg picks a sketch too -- the working sets sit in (2^18, 2^20]
     and (2^20, 2^22) bytes. *)
  let leg ~tag (nb, ns, no, ni) =
    let cyc n m stride = Array.init n ~f:(fun c -> Float.of_int (1 + (c % m)) *. stride) in
    let w =
      Operation.init ~l:(tag ^ "_w") ~prec:Ir.Ops.single ~i:[ ni ] ~o:[ no ]
        ~f:(Ll_test.weighted ~weights:[| 1; 3 |] ~modulus:5 ~offset:0.125 ~stride:0.125)
        ~grad_spec:Tensor.Require_grad ()
    in
    let x =
      NTDSL.ndarray
        (cyc (nb * ns * ni) 5 0.125)
        ~label:[ tag ^ "_x" ]
        ~batch_dims:[ nb; ns ] ~output_dims:[ ni ] ()
    in
    let dy =
      NTDSL.ndarray
        (cyc (nb * ns * no) 7 0.25)
        ~label:[ tag ^ "_dy" ]
        ~batch_dims:[ nb; ns ] ~output_dims:[ no ] ()
    in
    let%op y = w * x in
    let%op loss = (y *. dy) ++ "... => 0" in
    let wgrad = (Option.value_exn ~here:[%here] w.Tensor.diff).Tensor.grad in
    Train.set_materialized wgrad;
    let comp = named (tag ^ "_wgrad") (Train.grad_update loss) in
    let choice = ref None in
    let ctx = Context.auto () in
    (* Before the compile, from the same context: [model_explain] lowers for decisions only. *)
    let explained = Autotune.model_explain ctx comp Ir.Indexing.Empty in
    let _ctx, _routine =
      Autotune.model_default ~report:(fun r -> choice := Some r) ctx comp Ir.Indexing.Empty
    in
    let r = Option.value_exn ~here:[%here] !choice in
    Stdio.eprintf "%s: picked %s, rank %s ms, bound %s ms (not part of the golden)\n" tag
      r.Autotune.mc_label
      (Option.value_map r.Autotune.mc_rank_ms ~default:"n/a" ~f:Float.to_string)
      (Option.value_map r.Autotune.mc_model_ms ~default:"n/a" ~f:Float.to_string);
    List.iter explained.Autotune.x_segments ~f:(fun xs ->
        Stdio.eprintf "%s: segment %s explains %d candidates (not part of the golden)\n" tag
          xs.Autotune.xs_segment
          (List.length xs.Autotune.xs_candidates));
    (r, explained)
  in
  (* Both legs rank against a SERIAL untuned default: the claims are about the capacity charge. The
     CPU preset would parallelize these gradients' output loop (gh-ocannl-1458), and
     [Cost_model.reuse_traffic] reads each iteration of a [Grid] loop as a resident block that
     re-requests its cells, which moves the default's own working set past the capacity. *)
  Hashtbl.set Utils.config_file_args ~key:"cpu_schedule_min_work" ~data:"1000000000000";
  let large, large_explained = leg ~tag:"rk_large" (16, 32, 64, 128) in
  let small, small_explained = leg ~tag:"rk_small" (4, 32, 64, 128) in
  Hashtbl.remove Utils.config_file_args "cpu_schedule_min_work";
  p "past capacity the weight gradient's pick is a sketch, not the default"
    (not (String.equal large.Autotune.mc_label "default"));
  p "below capacity the weight gradient's pick stays the default"
    (String.equal small.Autotune.mc_label "default");
  p "the pick's rank is at least its bound"
    (match (large.Autotune.mc_rank_ms, large.Autotune.mc_model_ms) with
    | Some r, Some b -> Float.(r >= b)
    | _ -> false);
  (* --- [model_explain] per segment (gh-ocannl-1438): what [model_default] saw of each segment of
     its per-segment substitution, against what it picked. The weight-gradient steps above are
     segmented too since gh-ocannl-1512 (their small nests no longer coalesce), and keep their
     picks. The two-matmul chain the autotune tests search, with its materialized sums cutting it
     into keyed matmul segments, at 128x128: every nest's per-dispatch work clears cc's default
     threshold ([cpu_schedule_min_work], 16384), where the configured and the search's presets
     agree. The leg below takes a chain under it. --- *)
  let chain_comp =
    let q = 128 in
    let mat ~label ~modulus =
      TDSL.ndarray
        (Array.init (q * q)
           ~f:(Ll_test.cycle_flat ~dims:[| q; q |] ~modulus ~offset:0. ~stride:0.5))
        ~label:[ label ] ~input_dims:[ q ] ~output_dims:[ q ] ()
    in
    let qa = mat ~label:"xc_a" ~modulus:11 and qb = mat ~label:"xc_b" ~modulus:7 in
    let qc = mat ~label:"xc_c" ~modulus:5 and qf = mat ~label:"xc_f" ~modulus:9 in
    let%op qd = qa + qb in
    Train.set_materialized qd.Tensor.value;
    let%op qe = qd * qc in
    Train.set_materialized qe.Tensor.value;
    let%op qg = qf * qe in
    named "explain_chain" (Train.forward qg)
  in
  let chain_choice = ref None in
  let chain_ctx = Context.auto () in
  let chain_explained = Autotune.model_explain chain_ctx chain_comp Ir.Indexing.Empty in
  let _ctx, _routine =
    Autotune.model_default
      ~report:(fun r -> chain_choice := Some r)
      chain_ctx chain_comp Ir.Indexing.Empty
  in
  let chain = Option.value_exn ~here:[%here] !chain_choice in
  Stdio.eprintf "explain_chain: picked %s; segments [%s] (not part of the golden)\n"
    chain.Autotune.mc_label
    (String.concat ~sep:"; "
       (List.map chain_explained.Autotune.x_segments ~f:(fun xs ->
            Printf.sprintf "%s: %d candidates" xs.Autotune.xs_segment
              (List.length xs.Autotune.xs_candidates))));
  p "a fissioned chain's explanation lists its matmul segments, each with sketch leaves"
    (List.count chain_explained.Autotune.x_segments ~f:(fun xs ->
         List.length xs.Autotune.xs_candidates > 1)
    >= 2);
  (* --- Below the parallel threshold the model ranks on the search's segmentation (gh-ocannl-1512)
     ---

     A chain under the configured threshold: a 64x64 product over a 4096-long contraction, a second
     32x64 product of it, and a pointwise tail, each materialized. The threshold counts work per
     dispatch ([cpu_schedule_min_work], gh-ocannl-1458), and a product whose operands exceed the
     cache capacity cannot sit under its default, so the leg raises it past the chain in-process (as
     the rank legs above do): the configured CPU preset then leaves every nest unannotated, and
     fission coalesces adjacent unannotated segments back into one kernel, so the configured
     segmentation is the single kernel and [model_default] used to make no per-segment substitution
     at all. The search's ungated presets keep the chain cut, and its first product's operands (2
     MiB) exceed the rule's 1 MiB cache capacity, so a reuse-tiled sketch of that segment strictly
     beats its preset. The inputs are dyadic and small, so every sum is exact in f32 and the values
     are checked exactly against the host. *)
  let bm = 64 and bk = 4096 and bn = 64 and bp = 32 in
  let cyc ~dims ~modulus = Ll_test.cycle_flat ~dims ~modulus ~offset:0. ~stride:0.5 in
  let bav = Array.init (bm * bk) ~f:(cyc ~dims:[| bm; bk |] ~modulus:3) in
  let bcv = Array.init (bk * bn) ~f:(cyc ~dims:[| bk; bn |] ~modulus:5) in
  let bfv = Array.init (bp * bm) ~f:(cyc ~dims:[| bp; bm |] ~modulus:5) in
  let btv = Array.init (bp * bn) ~f:(cyc ~dims:[| bp; bn |] ~modulus:3) in
  let host_mm a b ~rows ~inner ~cols =
    Array.init (rows * cols) ~f:(fun idx ->
        let i = idx / cols and j = idx % cols in
        let acc = ref 0. in
        for l = 0 to inner - 1 do
          acc := !acc +. (a.((i * inner) + l) *. b.((l * cols) + j))
        done;
        !acc)
  in
  let bh_expected =
    let be = host_mm bav bcv ~rows:bm ~inner:bk ~cols:bn in
    let bg = host_mm bfv be ~rows:bp ~inner:bm ~cols:bn in
    Array.mapi bg ~f:(fun i g -> g +. btv.(i))
  in
  let below_comp, bh =
    let mat ~label values ~o ~i =
      TDSL.ndarray values ~label:[ label ] ~input_dims:[ i ] ~output_dims:[ o ] ()
    in
    let ba = mat ~label:"bt_a" bav ~o:bm ~i:bk and bc = mat ~label:"bt_c" bcv ~o:bk ~i:bn in
    let bf = mat ~label:"bt_f" bfv ~o:bp ~i:bm and bt = mat ~label:"bt_t" btv ~o:bp ~i:bn in
    let%op be = ba * bc in
    Train.set_materialized be.Tensor.value;
    let%op bg = bf * be in
    Train.set_materialized bg.Tensor.value;
    let%op bh = bg + bt in
    (named "below_threshold_chain" (Train.forward bh), bh)
  in
  Hashtbl.set Utils.config_file_args ~key:"cpu_schedule_min_work" ~data:"1000000000000";
  let below_choice = ref None in
  let below_ctx = Context.auto () in
  (* The premise: the configured segmentation is the single kernel, the search's cuts the chain. *)
  let premise_segments presets =
    let target = Ll_test.host_target () in
    List.length
      (fst
         (Autotune.sketch_segmentation ~presets ~is_gpu:target.Ll_test.is_gpu
            ~is_cpu:target.Ll_test.is_cpu ~limits:target.Ll_test.limits ~static_indices:[]
            (Context.lowered_for_decisions below_ctx below_comp Ir.Indexing.Empty)))
  in
  let config_segments = premise_segments `Config and search_segments = premise_segments `Search in
  Stdio.eprintf
    "below_threshold_chain: %d configured segment(s), %d search segment(s) (not part of the golden)\n"
    config_segments search_segments;
  p "below the threshold the configured segmentation is the single kernel" (config_segments = 1);
  p "below the threshold the search's segmentation cuts the chain" (search_segments >= 3);
  let below_explained = Autotune.model_explain below_ctx below_comp Ir.Indexing.Empty in
  let below_ctx, below_routine =
    Autotune.model_default
      ~report:(fun r -> below_choice := Some r)
      below_ctx below_comp Ir.Indexing.Empty
  in
  let below = Option.value_exn ~here:[%here] !below_choice in
  Stdio.eprintf "below_threshold_chain: picked %s; segments [%s] (not part of the golden)\n"
    below.Autotune.mc_label
    (String.concat ~sep:"; "
       (List.map below_explained.Autotune.x_segments ~f:(fun xs ->
            Printf.sprintf "%s: %d candidates" xs.Autotune.xs_segment
              (List.length xs.Autotune.xs_candidates))));
  p "below the threshold the explanation lists the chain's matmul segments, each with sketch leaves"
    (List.count below_explained.Autotune.x_segments ~f:(fun xs ->
         List.length xs.Autotune.xs_candidates > 1)
    >= 2);
  (* The condition for [model_default] to form a per-segment substitution at all: a segment whose
     family search finds a leaf strictly below the segment's preset. Under the search's preset the
     first product's re-reads past capacity are charged; under the configured (unannotated) one they
     would not be, its sketches could at best tie it, and no substitution would form. *)
  p_exists "below the threshold some segment's sketch strictly beats the segment's preset"
    below_explained.Autotune.x_segments ~f:(fun xs ->
      match xs.Autotune.xs_candidates with
      | ("preset", { Autotune.ex_rank_ms = Some preset; _ }) :: leaves ->
          List.exists leaves ~f:(fun (_, ex) ->
              match ex.Autotune.ex_rank_ms with Some r -> Float.(r < preset) | None -> false)
      | _ -> false);
  Hashtbl.remove Utils.config_file_args "cpu_schedule_min_work";
  let below_ctx = Context.run below_ctx below_routine in
  let below_got = Context.get_values below_ctx bh.Tensor.value in
  p_all2 "the below-threshold pick computes the host's values exactly" below_got bh_expected
    ~f:Float.equal;
  let legs =
    [
      (large, large_explained);
      (small, small_explained);
      (chain, chain_explained);
      (below, below_explained);
    ]
  in
  p_all "each explained segment is listed once" legs ~f:(fun (_, x) ->
      not
        (List.contains_dup x.Autotune.x_segments ~compare:(fun a b ->
             String.compare a.Autotune.xs_segment b.Autotune.xs_segment)));
  p_all "each explained segment lists its preset first, admitted" legs ~f:(fun (_, x) ->
      List.for_all x.Autotune.x_segments ~f:(fun xs ->
          match xs.Autotune.xs_candidates with
          | ("preset", ex) :: _ -> ex.Autotune.ex_admitted
          | _ -> false));
  p_all "every explained per-segment ranking score is at least its bound" legs ~f:(fun (_, x) ->
      List.for_all x.Autotune.x_segments ~f:(fun xs ->
          List.for_all xs.Autotune.xs_candidates ~f:(fun (_, ex) ->
              match (ex.Autotune.ex_rank_ms, ex.Autotune.ex_floor_ms) with
              | Some r, Some f -> Float.(r >= f)
              | None, None -> true
              | _ -> false)));
  (* The pick is a candidate the explanation lists: the incumbent or a whole-routine leaf by its
     label, or a per-segment substitution whose every entry is a leaf listed under its own segment,
     in segment order -- which is the per-segment explanation's whole claim to describe what
     [model_default] ranks. *)
  let listed_pick (r, (x : Autotune.explanation)) =
    let label = r.Autotune.mc_label in
    List.exists x.Autotune.x_routine ~f:(fun (l, _) -> String.equal l label)
    ||
    match String.chop_prefix label ~prefix:"F_sketch[" with
    | None -> false
    | Some rest -> (
        match String.chop_suffix rest ~suffix:"]" with
        | None -> false
        | Some entries ->
            (* Each entry is a segment's leaf label, segments in order, separated by commas; tried
               against every remaining segment's leaves, so no label's own text is parsed. *)
            let rec matches text = function
              | [] -> String.is_empty text
              | (xs : Autotune.segment_explain) :: later ->
                  matches text later
                  || List.exists (List.tl_exn xs.Autotune.xs_candidates) ~f:(fun (leaf, _) ->
                      match String.chop_prefix text ~prefix:leaf with
                      | Some "" -> true
                      | Some after -> (
                          match String.chop_prefix after ~prefix:"," with
                          | Some next -> (not (String.is_empty next)) && matches next later
                          | None -> false)
                      | None -> false)
            in
            (not (String.is_empty entries)) && matches entries x.Autotune.x_segments)
  in
  p_all "model_default's pick is a candidate model_explain lists" legs ~f:listed_pick
