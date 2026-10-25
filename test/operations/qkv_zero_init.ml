(* gh-ocannl-1175: GPU fission folds a reduction's covering zero, expanded per cell, into the
   reduction's own kernel, and changes the segmentation in nothing else. The sketches keep the
   folded projection tiled and tensorized; Privatize forwards the zero directly into its private
   accumulator with the serial localizer's proof, rather than loading the output. A tensorized
   accumulator still loads the output the folded zero just wrote, lane-partitioned: every GPU
   renderer opens that load with a workgroup barrier, and the executed GPU leg below runs every
   seed, tensorized ones included. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module Sched = Ir.Schedule

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let () = Stdio.eprintf "qkv_zero_init backend=%s\n" backend_name
let x_value ~offset ix = Float.of_int (offset + 1 + (3 * ix.(0)) + (5 * ix.(1)) + ix.(2)) /. 16.
let w_value ~offset ix = Float.of_int (offset + 1 + (7 * ix.(0)) + (3 * ix.(1)) + ix.(2)) /. 32.

(* [prec] is the operands' storage; the output stays f32 whatever it is, so a narrow-operand
   projection exercises the mixed (narrow, narrow, f32) triple, never a narrow accumulator. [offset]
   lifts every operand numerator, bounding each contraction term away from zero relative to its
   cell's sum. *)
let projection ?(prec = Ir.Ops.single) ?(offset = 0) ~b ~s ~h ~j ~k () =
  let x = NTDSL.init ~l:"zi_x" ~prec ~o:[ b; s; k ] ~f:(x_value ~offset) () in
  let w = NTDSL.init ~l:"zi_w" ~prec ~o:[ h; j; k ] ~f:(w_value ~offset) () in
  let%op out = x +* "bsk;hjk=>bshj" w in
  Ir.Tnode.update_prec out.Tensor.value Ir.Ops.single;
  Train.set_materialized out.Tensor.value;
  out

let capture ~name out =
  let captured = ref None in
  let _, _ =
    Context.compile ~name
      ~lowered_transform:(fun opt ->
        captured := Some opt;
        [ opt ])
      (Context.auto ()) (Train.forward out) Ir.Indexing.Empty
  in
  Option.value_exn !captured

(* The sketch candidates' segmentation folds the zero; the untuned default pipeline does not. Both
   are the tuner's own pipeline (gh-ocannl-1483), under a synthetic GPU target so the GPU zero
   policy applies on every host. *)
let synthetic = Ll_test.synthetic_gpu ()
let fission opt = snd (Ll_test.tuner_segments ~target:synthetic opt)
let unfolded opt = snd (Ll_test.tuner_segments_unfolded ~target:synthetic opt)

(* A synthetic matrix-unit capability (as [launch_predicate_parity]'s), so the tensorized seeds are
   proposed and validated whatever the host backend. *)
let mma_limits =
  let module BI = Ir.Backend_intf in
  {
    BI.no_hardware_limits with
    mma =
      Some
        {
          BI.minimal_mma_capability with
          mma_tile = (8, 8, 8);
          mma_format_tiles = [ ((BI.Mma_f32, BI.Mma_f32, BI.Mma_f32), (8, 8, 8)) ];
        };
  }

let gpu_seeds ~limits pre =
  Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits pre
  |> List.filter ~f:(fun q -> q.Autotune.sk_gpu)

(* Schedule application registers scratch nodes and placements. Each prototype owns a copy, so
   another prototype's scratch cannot leak into its interface or declarations. *)
let apply schedule opt =
  Sched.apply schedule
    {
      opt with
      LL.traced_store = Hashtbl.copy opt.LL.traced_store;
      optimize_ctx = LL.copy_optimize_ctx opt.LL.optimize_ctx;
    }

(* Every sketch the host GPU would seed for [pre], executed against the materialized run [want]
   under [matches]. Seeding reads the backend's tensor-unit formats and the numerics policy, so the
   population is the host's: tf32 seeds join CUDA's f32 site only under the approximate profile, and
   HIP's rocWMMA seeds 16-bit operands only. No gate reads the population under test
   (gh-ocannl-1115): a GPU that seeds nothing fails the first claim, and the tensorized claim is
   gated on the CAPABILITY the backend advertises for the site's actual operand and destination
   precisions, so where a tensor tile is advertised a vanished tensorized seed fails rather than
   skipping. *)
let folded_seed_claims ~tag ~what ~pre ~out ~seed ~want ~matches =
  let on_gpu = Sched.backend_is_gpu backend_name in
  let limits = Context.hardware_limits (Context.auto ()) in
  let site = Option.value_exn (Autotune.detect_matmul pre.LL.llc) in
  let capability =
    if on_gpu then
      Ll_test.tensorized_matmul_capability ~is_gpu:true ~is_cpu:false ~limits ~a:site.Autotune.m_a
        ~b:site.m_b ~d:site.m_d
    else `Withheld `Backend
  in
  let real = if on_gpu then gpu_seeds ~limits pre else [] in
  (* Each result keeps its routine's own [mma] census: a tensorized seed whose [Tile_mma] declined
     to the scalar fallback at codegen still matches, but it did not run the tensorized path. *)
  let results =
    List.mapi real ~f:(fun i q ->
        let o = apply (Autotune.sketch_schedule ~accum_prec:Fn.id ~p:q pre) pre in
        let ctx, routine = Ll_test.link ~name:(tag ^ Int.to_string i) o in
        let got = Context.get_values (Ll_test.run_linked (ctx, routine) ~seed) out.Tensor.value in
        let ok = matches got want in
        if not ok then
          Stdio.eprintf "%s seed %d (mma=%b) differs from the materialized run\n" what i
            q.Autotune.sk_mma;
        let rendered =
          Ir.C_syntax.equal_tensorization routine.Context.mma.tensorization Ir.C_syntax.Tensorized
        in
        (q, ok, rendered))
  in
  Stdio.eprintf
    "%s seeds executed (not part of the golden): %d, %d tensorized, %d of those rendered tensorized\n"
    what (List.length real)
    (List.count real ~f:(fun q -> q.Autotune.sk_mma))
    (List.count results ~f:(fun (q, _, rendered) -> q.Autotune.sk_mma && rendered));
  gated_all ~when_:on_gpu ~on:backend_name
    ("every GPU sketch of the " ^ what ^ " matches the materialized run")
    results
    ~f:(fun (_, ok, _) -> ok);
  let label =
    "every tensorized " ^ what ^ " sketch renders tensorized and matches the materialized run"
  in
  match capability with
  | `Advertised ->
      (* Every tensorized seed, not some (gh-ocannl-1283): one seed declining to the scalar fallback
         fails the claim even while its siblings render the intrinsics. *)
      p_all label
        (List.filter results ~f:(fun (q, _, _) -> q.Autotune.sk_mma))
        ~f:(fun (_, ok, rendered) -> rendered && ok)
  | `Withheld aggregation -> Verdict.skipped ~aggregation ~backend:backend_name label

let () =
  let out = projection ~b:32 ~s:32 ~h:4 ~j:32 ~k:128 () in
  let opt = capture ~name:"zi_shape" out in
  let parts = fission opt in
  p "gpt2 qkv projection has one kernel including initialization" (List.length parts = 1);
  p "the untuned default keeps the projection's zero in its own kernel"
    (List.map (unfolded opt) ~f:(fun (kind, _, _, _) -> kind)
    |> List.equal Poly.equal [ `Zeros; `Normal ]);
  p_all "qkv kernel retains GPU parallelism" parts ~f:(fun (_, _, _, o) ->
      not (List.is_empty (LL.hardware_axes o.LL.llc)));
  let _, pre, _, _ = List.hd_exn parts in
  let seeds = gpu_seeds ~limits:mma_limits pre in
  p "qkv init companion preserves tiled sketch eligibility" (not (List.is_empty seeds));
  p_exists "qkv init companion preserves tensorized sketch eligibility" seeds ~f:(fun q ->
      q.Autotune.sk_mma);
  p_all "qkv tiled and tensorized sketches construct and validate" seeds ~f:(fun seed ->
      match apply (Autotune.sketch_schedule ~accum_prec:Fn.id ~p:seed pre) pre with
      | o -> (
          match LL.validate_parallel o.LL.optimize_ctx.placements o.LL.llc with
          | () -> true
          | exception exn ->
              Stdio.eprintf "qkv sketch validation FAILED: %s\n" (Exn.to_string exn);
              false)
      | exception exn ->
          Stdio.eprintf "qkv sketch construction FAILED: %s\n" (Exn.to_string exn);
          false);
  p_exists "a tiled qkv sketch forwards zero into the private accumulator" seeds ~f:(fun seed ->
      let o = apply (Autotune.sketch_schedule ~accum_prec:Fn.id ~p:seed pre) pre in
      Ll_test.count_get o out.Tensor.value = 0 && Ll_test.count_set o out.Tensor.value = 1)

let () =
  let out = projection ~b:2 ~s:32 ~h:2 ~j:32 ~k:64 () in
  let opt = capture ~name:"zi_parity_capture" out in
  let seed = [ (out.Tensor.value, Array.create ~len:(2 * 32 * 2 * 32) (-999.)) ] in
  let want =
    List.hd_exn (Ll_test.execute ~name:"zi_materialized" opt ~seed ~read:[ out.Tensor.value ])
  in
  p_all "qkv reference differs from zero and the entry sentinel" (Array.to_list want) ~f:(fun v ->
      Float.(v > 0.));
  let parts = fission opt in
  let _, pre, _, _ = List.hd_exn parts in
  let site = Option.value_exn (Autotune.detect_matmul pre.LL.llc) in
  let tiled =
    apply [ Sched.privatize ~accum_prec:Fn.id ~target:site.Autotune.m_d ~over:site.m_k ] pre
  in
  p "private qkv accumulator opens from zero without output reads"
    (Ll_test.count_get tiled out.Tensor.value = 0);
  p "private qkv accumulator drops the covering init store"
    (Ll_test.count_set tiled out.Tensor.value = 1);
  let got =
    List.hd_exn (Ll_test.execute ~name:"zi_private" tiled ~seed ~read:[ out.Tensor.value ])
  in
  p_all2 "private qkv matches the materialized run with discriminating operands" got want
    ~f:Float.equal;
  let staged_label = "staged tiled qkv with zero forwarded matches the materialized run" in
  if Sched.backend_is_gpu backend_name then begin
    let staged =
      Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false
        ~limits:Ir.Backend_intf.no_hardware_limits pre
      |> List.find_map ~f:(fun p ->
          let o = apply (Autotune.sketch_schedule ~accum_prec:Fn.id ~p pre) pre in
          Option.some_if (Ll_test.count_get o out.Tensor.value = 0) o)
      |> Option.value_exn
    in
    let got =
      List.hd_exn (Ll_test.execute ~name:"zi_staged" staged ~seed ~read:[ out.Tensor.value ])
    in
    p_all2 staged_label got want ~f:Float.equal
  end
  else Verdict.skipped ~backend:backend_name staged_label;
  (* Every sketch the host GPU would seed for the folded segment, executed: the tensorized ones load
     their accumulator fragment from cells the folded zero nest wrote earlier in the same kernel.
     Bitwise, under tf32 too (the [-approximate] run): every operand numerator is below 2^11, inside
     tf32's significand, and every partial sum is a multiple of 2^-9 below 2^15, so a correct
     schedule rounds nowhere in any format the f32 site seeds. *)
  folded_seed_claims ~tag:"zi_seed_" ~what:"folded qkv" ~pre ~out ~seed ~want
    ~matches:(Array.equal Float.equal);
  let ctx, routine =
    Context.compile ~name:"zi_default" ~prelowered:opt
      ~lowered_transform:(Ll_test.tuner_transform ~target:synthetic)
      (Context.auto ()) Ir.Assignments.empty_comp Ir.Indexing.Empty
  in
  let ctx = Ll_test.run_linked (ctx, routine) ~seed in
  let got = Context.get_values ctx out.Tensor.value in
  p_all2 "fission qkv matches the materialized run with discriminating operands" got want
    ~f:Float.equal;
  let ctx = Context.run ctx routine in
  p_all2 "fission qkv resets the accumulator on every call"
    (Context.get_values ctx out.Tensor.value)
    want ~f:Float.equal

(* The folded projection with f16 operands and its f32 accumulator: the one tensor-unit triple CUDA,
   HIP and Metal all advertise. HIP's rocWMMA has no f32 tile, tf32 included, so the f32 legs above
   execute no tensorized seed there under any profile. The operands are still exact (the numerators
   are below 2^11), but the serial reference may form each product at f16 where the tensor unit
   forms it exactly, so the claim is a tolerance: |got - want| <= 2^-8 want, four times the bound of
   one f16 rounding per product (2^-11 on each side, every product positive). The operand offset
   makes every single contraction term exceed that tolerance against its cell's sum -- claimed below
   from the operand formulas -- so losing any one term fails the parity claim. *)
let () =
  let b = 2 and s = 32 and h = 2 and j = 32 and k = 64 and offset = 255 in
  let tolerance = 0x1p-8 in
  let cells =
    List.concat_map (List.range 0 b) ~f:(fun bi ->
        List.concat_map (List.range 0 s) ~f:(fun si ->
            List.concat_map (List.range 0 h) ~f:(fun hi ->
                List.map (List.range 0 j) ~f:(fun ji ->
                    List.map (List.range 0 k) ~f:(fun ki ->
                        x_value ~offset [| bi; si; ki |] *. w_value ~offset [| hi; ji; ki |])))))
  in
  p_all "every f16-operand qkv contraction term exceeds the tolerance against its cell's sum" cells
    ~f:(fun terms ->
      let sum = List.sum (module Float) terms ~f:Fn.id in
      List.for_all terms ~f:(fun t -> Float.(t > tolerance * sum)));
  let out = projection ~prec:Ir.Ops.half ~offset ~b ~s ~h ~j ~k () in
  let opt = capture ~name:"zh_parity_capture" out in
  let seed = [ (out.Tensor.value, Array.create ~len:(b * s * h * j) (-999.)) ] in
  let want =
    List.hd_exn (Ll_test.execute ~name:"zh_materialized" opt ~seed ~read:[ out.Tensor.value ])
  in
  p_all "f16-operand qkv reference differs from zero and the entry sentinel" (Array.to_list want)
    ~f:(fun v -> Float.(v > 0.));
  let _, pre, _, _ = List.hd_exn (fission opt) in
  let worst = ref 0. in
  folded_seed_claims ~tag:"zh_seed_" ~what:"folded f16-operand qkv" ~pre ~out ~seed ~want
    ~matches:(fun got want ->
      Array.iter2_exn got want ~f:(fun g w -> worst := Float.max !worst (Float.abs (g -. w) /. w));
      Array.for_all2_exn got want ~f:(fun g w -> Float.(abs (g - w) <= w * tolerance)));
  Stdio.eprintf "f16-operand qkv seeds' max relative deviation (not part of the golden): %h\n"
    !worst

(* Sibling projections (the q and k of one attention layer): each zero folds into its OWN
   accumulation's kernel and the segmentation changes in nothing else. An expanded zero joining the
   preceding segment bridged the two projections into one kernel no matmul sketch reaches, doubling
   the tuned CUDA step (staging#934, reverted). The expected segmentation is derived from the
   untuned default's ([fold_zeros:false], the same zero policy): every whole-node zero segment
   merged into the segment that follows it. *)
let () =
  let b = 2 and s = 32 and h = 2 and j = 32 and k = 64 in
  let init ~l ~o ~f = NTDSL.init ~l ~prec:Ir.Ops.single ~o ~f () in
  let x =
    init ~l:"zs_x" ~o:[ b; s; k ] ~f:(fun ix ->
        Float.of_int (1 + (3 * ix.(0)) + (5 * ix.(1)) + ix.(2)) /. 64.)
  in
  let wq =
    init ~l:"zs_wq" ~o:[ h; j; k ] ~f:(fun ix ->
        Float.of_int (1 + (7 * ix.(0)) + (3 * ix.(1)) + ix.(2)) /. 128.)
  in
  let wk =
    init ~l:"zs_wk" ~o:[ h; j; k ] ~f:(fun ix ->
        Float.of_int (2 + (5 * ix.(0)) + ix.(1) + (2 * ix.(2))) /. 128.)
  in
  let%op q = x +* "bsk;hjk=>bshj" wq in
  let%op kp = x +* "bsk;hjk=>bshj" wk in
  let%op scores = q +* "bshj;bthj=>bhst" kp in
  List.iter [ q; kp; scores ] ~f:(fun t -> Train.set_materialized t.Tensor.value);
  let opt = capture ~name:"zs_capture" scores in
  let writes (pre : LL.optimized) =
    LL.affine_accesses pre.LL.llc
    |> List.filter_map ~f:(fun a -> Option.some_if a.Ir.Affine.a_write a.Ir.Affine.a_tn)
    |> Set.of_list (module Ir.Tnode)
  in
  let segments parts = List.map parts ~f:(fun (kind, pre, _, _) -> (kind, writes pre)) in
  let unexpanded = segments (unfolded opt) in
  let folded = segments (fission opt) in
  let rec fold_zeros = function
    | (`Zeros, zs) :: (_, ws) :: rest when Set.is_subset zs ~of_:ws -> ws :: fold_zeros rest
    | (_, ws) :: rest -> ws :: fold_zeros rest
    | [] -> []
  in
  let expected = fold_zeros unexpanded in
  p "the untuned default separates every projection zero"
    (List.count unexpanded ~f:(fun (kind, _) -> Poly.equal kind `Zeros) = 3);
  p "folding moves each zero into its own accumulation and changes nothing else"
    (List.equal Set.equal (List.map folded ~f:snd) expected);
  p_none "no kernel accumulates both sibling projections" folded ~f:(fun (_, ws) ->
      Set.mem ws q.Tensor.value && Set.mem ws kp.Tensor.value);
  let projection_segments =
    List.filter_map (fission opt) ~f:(fun (_, pre, _, _) ->
        match Autotune.detect_matmul pre.LL.llc with
        | Some site
          when Ir.Tnode.equal site.Autotune.m_d q.Tensor.value
               || Ir.Tnode.equal site.m_d kp.Tensor.value ->
            Some pre
        | _ -> None)
  in
  p "both projections keep a detectable matmul site of their own"
    (List.length projection_segments = 2);
  p_all "every sibling projection keeps tiled and tensorized sketches that construct and validate"
    projection_segments ~f:(fun pre ->
      let seeds = gpu_seeds ~limits:mma_limits pre in
      List.exists seeds ~f:(fun q -> q.Autotune.sk_mma)
      && List.for_all seeds ~f:(fun seed ->
          match apply (Autotune.sketch_schedule ~accum_prec:Fn.id ~p:seed pre) pre with
          | o -> (
              match LL.validate_parallel o.LL.optimize_ctx.placements o.LL.llc with
              | () -> true
              | exception exn ->
                  Stdio.eprintf "sibling sketch validation FAILED: %s\n" (Exn.to_string exn);
                  false)
          | exception exn ->
              Stdio.eprintf "sibling sketch construction FAILED: %s\n" (Exn.to_string exn);
              false));
  let read = [ q.Tensor.value; kp.Tensor.value; scores.Tensor.value ] in
  let numel t = Array.fold (Lazy.force t.Ir.Tnode.dims) ~init:1 ~f:( * ) in
  let seed = List.map read ~f:(fun t -> (t, Array.create ~len:(numel t) (-999.))) in
  let want = Ll_test.execute ~name:"zs_materialized" opt ~seed ~read in
  p_all "sibling references differ from zero and the entry sentinel"
    (List.concat_map want ~f:Array.to_list) ~f:(fun v -> Float.(v > 0.));
  let ctx, routine =
    Context.compile ~name:"zs_fission" ~prelowered:opt
      ~lowered_transform:(Ll_test.tuner_transform ~target:synthetic)
      (Context.auto ()) Ir.Assignments.empty_comp Ir.Indexing.Empty
  in
  let ctx = Ll_test.run_linked (ctx, routine) ~seed in
  let got = List.map read ~f:(Context.get_values ctx) in
  p_all2 "folded sibling projections match the materialized run" (Array.concat got)
    (Array.concat want) ~f:Float.equal

(* An enclosing reduction repeats each cell. Its inner private tile must load the previous partial
   sum, and the whole-node zero remains live. *)
let () =
  let open Ll_test in
  let node = node_factory ~first_id:117500 ~dims:[| 4 |] () in
  let out = node "zi_repeated" in
  materialize out;
  let outer = sym () and i = sym () and k = sym () in
  let update = set_at out (iter i) (add (get out [| iter i |]) (add (tag outer i) (tick k))) in
  let code = seq (zero out) (loop_n outer 3 (loop_n i 4 (loop_n k 5 update))) in
  let opt = optimize ~name:"zi_repeat_capture" code in
  let priv = apply [ Sched.privatize ~accum_prec:Fn.id ~target:out ~over:k ] opt in
  p "an enclosing repeated-cell loop retains initialization and opening reads"
    (count_set priv out = 2 && count_get priv out = 1);
  let seed = [ (out, Array.create ~len:4 (-999.)) ] in
  let want = List.hd_exn (execute ~name:"zi_repeat_materialized" opt ~seed ~read:[ out ]) in
  let got = List.hd_exn (execute ~name:"zi_repeat_private" priv ~seed ~read:[ out ]) in
  p_all2 "repeated-cell private reduction preserves earlier contributions" got want ~f:Float.equal

let () =
  let open Ll_test in
  let node = node_factory ~first_id:117600 ~dims:[| 4 |] () in
  let out = node "zi_refusal" in
  let i = sym () in
  let full = loop_n i 4 (set_at out (iter i) (c 0.)) in
  p "a full per-cell zero is recognized" (Option.is_some (LL.zero_initializer_target full));
  let refused =
    [
      loop_n i 3 (set_at out (iter i) (c 0.));
      if_ (tick i) full;
      loop_n i 0 (set_at out (iter i) (c 0.));
      loop_n i 4 (set_at out (iter i) (c (-0.)));
    ]
  in
  p_all "partial, guarded, dead and negative-zero initializers are retained" refused ~f:(fun init ->
      Option.is_none (LL.zero_initializer_target init))

(* Reusing an immutable statement at two positions must not make zero DSE remove both. *)
let () =
  let open Ll_test in
  let node = node_factory ~first_id:117700 ~dims:[| 4 |] () in
  let out = node "zi_reused_zero" in
  materialize out;
  let i = sym () and k = sym () in
  let cell = [| iter i |] in
  let shared_zero = zero out in
  let update = set out cell (add (get out cell) (add (tick i) (tick k))) in
  let program = seq shared_zero (seq (loop_n i 4 (loop_n k 5 update)) shared_zero) in
  let opt = optimize_scoped ~materialized:[ out ] ~name:"zi_reused_capture" ~raw:program program in
  let priv = apply [ Sched.privatize ~accum_prec:Fn.id ~target:out ~over:k ] opt in
  p "private zero forwarding removes only the proven initializer occurrence"
    (List.count (LL.flat_lines [ priv.LL.llc ]) ~f:(function
       | LL.Zero_out tn -> Ir.Tnode.equal tn out
       | _ -> false)
    = 1);
  let seed = [ (out, Array.create ~len:4 (-999.)) ] in
  let want = List.hd_exn (execute ~name:"zi_reused_materialized" opt ~seed ~read:[ out ]) in
  let got = List.hd_exn (execute ~name:"zi_reused_private" priv ~seed ~read:[ out ]) in
  p_all2 "reused final zero preserves executed parity with the materialized run" got want
    ~f:Float.equal

(* A matmul with an elementwise tail (bias + relu): the folded zero collapses the routine to ONE
   segment, so fission's single-kernel fallback applies the sketch candidate's schedule, fused
   epilogue twins included. A sketch whose preconditions the segment violates must decline as a
   classified schedule outcome, exactly as it does in a multi-segment routine -- an uncaught
   [Invalid_argument] escaped the autotuner (master red at a34634b62, tuf HIP). Structural, so it
   runs on every backend, cc included. *)
let () =
  let n = 32 in
  let init ~l ~o ~f = NTDSL.init ~l ~prec:Ir.Ops.single ~o ~f () in
  let ma =
    init ~l:"ze_a" ~o:[ n; n ] ~f:(fun ix -> Float.of_int (1 + (3 * ix.(0)) + ix.(1)) /. 64.)
  in
  let mb =
    init ~l:"ze_b" ~o:[ n; n ] ~f:(fun ix -> Float.of_int (2 + ix.(0) + (5 * ix.(1))) /. 64.)
  in
  let bias = init ~l:"ze_bias" ~o:[ n ] ~f:(fun ix -> Float.of_int (ix.(0) - 7) /. 8.) in
  let%op prod = ma +* "ik;kj=>ij" mb in
  Train.set_materialized prod.Tensor.value;
  let%op out = relu (prod + bias) in
  Train.set_materialized out.Tensor.value;
  let opt = capture ~name:"ze_capture" out in
  let limits = mma_limits in
  let target = Ll_test.synthetic_gpu ~limits () in
  let plan, folded = Ll_test.tuner_segments ~target opt in
  p "the folded matmul and its tail collapse to one segment opening with the zero"
    (match folded with
    | [ (`Normal, pre, _, _) ] ->
        List.exists (LL.flat_lines [ pre.LL.llc ]) ~f:(fun stmt ->
            Option.exists (LL.zero_initializer_target stmt) ~f:(Ir.Tnode.equal prod.Tensor.value))
    | _ -> false);
  let pre = match folded with (_, pre, _, _) :: _ -> pre | [] -> opt in
  let seeds = gpu_seeds ~limits pre in
  p_exists "the folded segment seeds fused-epilogue twins" seeds ~f:(fun q ->
      q.Autotune.sk_epilogue);
  let outcome q =
    (* Replayed as a per-segment sketch candidate compiles: the derived plan, every segment
       sketched. *)
    let schedule _ seg = Some (Autotune.sketch_schedule ~accum_prec:Fn.id ~p:q seg) in
    match
      Ll_test.tuner_replay ~target ~plan ~schedule
        {
          opt with
          LL.traced_store = Hashtbl.copy opt.LL.traced_store;
          optimize_ctx = LL.copy_optimize_ctx opt.LL.optimize_ctx;
        }
    with
    | _ -> `Applied
    | exception Ir.Schedule_outcome.Cause_at _ -> `Declined
    | exception exn ->
        Stdio.eprintf "folded sketch (epilogue=%b mma=%b) escaped unclassified: %s\n"
          q.Autotune.sk_epilogue q.Autotune.sk_mma (Exn.to_string exn);
        `Escaped
  in
  let outcomes = List.map seeds ~f:(fun q -> (q, outcome q)) in
  Stdio.eprintf "folded single-segment sketches (not part of the golden): %d applied, %d declined\n"
    (List.count outcomes ~f:(fun (_, o) -> Poly.equal o `Applied))
    (List.count outcomes ~f:(fun (_, o) -> Poly.equal o `Declined));
  p_none "no sketch of the folded single segment escapes as an unclassified exception" outcomes
    ~f:(fun (_, o) -> Poly.equal o `Escaped)

(* The same matmul+tail with the product NOT materialized (gh-ocannl-1453), segmented as the tuner's
   per-segment sketch candidates segment it ([Ll_test.tuner_segments]: promoted locals, the default
   GPU mapping probe, the zero folded and expanded per cell). The fused tail is the product's only
   consumer, so the tensorized fused twins' [Fuse_epilogue] places it in workgroup-shared storage,
   and the folded per-cell zero nest beside the site takes the site's geometry from the sketch
   itself -- companion coverage counts only materialized writes, and the zero used to stay bare
   (every thread storing every cell), which gh-ocannl-959's binding legality refused at render.
   (This file's earlier hand-rolled segmentation promoted no locals, so it left the folded zero
   whole-node, which the pipelines always expanded with the site's geometry, and the leg passed on
   the unfixed code, gh-ocannl-1483.) Every GPU seed of the folded segment executes against the
   materialized run, bitwise: every operand numerator is below 2^11 over a power-of-two denominator
   and every partial sum is a multiple of 2^-12 below 2^8, so a correct schedule rounds nowhere,
   tf32 included; the integral bias drives part of the output below zero, so the relu both passes
   and clamps cells. GPU-only: the C backends fold no per-cell zero. *)
let () =
  let n = 32 in
  let init ~l ~o ~f = NTDSL.init ~l ~prec:Ir.Ops.single ~o ~f () in
  let ma =
    init ~l:"zl_a" ~o:[ n; n ] ~f:(fun ix -> Float.of_int (1 + (3 * ix.(0)) + ix.(1)) /. 64.)
  in
  let mb =
    init ~l:"zl_b" ~o:[ n; n ] ~f:(fun ix -> Float.of_int (2 + ix.(0) + (5 * ix.(1))) /. 64.)
  in
  let bias = init ~l:"zl_bias" ~o:[ n ] ~f:(fun ix -> Float.of_int (5 * (7 - ix.(0)))) in
  let%op prod = ma +* "ik;kj=>ij" mb in
  let%op out = relu (prod + bias) in
  Train.set_materialized out.Tensor.value;
  let opt = capture ~name:"zl_capture" out in
  let seed = [ (out.Tensor.value, Array.create ~len:(n * n) (-999.)) ] in
  let want =
    List.hd_exn (Ll_test.execute ~name:"zl_materialized" opt ~seed ~read:[ out.Tensor.value ])
  in
  p_exists "the routine-local product's reference has positive cells" (Array.to_list want)
    ~f:(fun v -> Float.(v > 0.));
  p_exists "the routine-local product's reference has relu-clamped cells" (Array.to_list want)
    ~f:(fun v -> Float.(v = 0.));
  let structural =
    [
      "the tuner's segmentation folds the routine-local product's zero per cell into one segment";
      "the product is not materialized in the folded segment";
      "the routine-local product's folded segment seeds tensorized fused-epilogue twins";
    ]
  in
  (* The one segment opening with the product's PER-CELL zero (a whole-node [Zero_out] is the form
     the bug hid behind). *)
  let folded_per_cell segments =
    match segments with
    | [ (`Normal, pre, _, _) ] ->
        Option.some_if
          (List.exists (LL.flat_lines [ pre.LL.llc ]) ~f:(function
            | LL.Zero_out _ -> false
            | stmt ->
                Option.exists (LL.zero_initializer_target stmt)
                  ~f:(Ir.Tnode.equal prod.Tensor.value)))
          pre
    | _ -> None
  in
  (* gh-ocannl-1483: the tuner's segmentation reaches the folded per-cell path on every host, cc
     included, under a synthetic GPU target -- the fold is GPU zero policy. The hand-rolled
     segmentation this file used before promoted no locals, so the routine-local product's zero
     never reached fission as a materialized whole-node zero and stayed whole in the matmul's
     segment: this claim is false under it, and under a pipeline that drops the fold. *)
  p "the tuner's segmentation under a synthetic GPU folds the routine-local product's zero per cell"
    (Option.is_some (folded_per_cell (fission opt)));
  let pre =
    if not (Sched.backend_is_gpu backend_name) then (
      List.iter structural ~f:(Verdict.skipped ~backend:backend_name);
      opt)
    else
      let limits = Context.hardware_limits (Context.auto ()) in
      let pre = folded_per_cell (snd (Ll_test.tuner_segments opt)) in
      p (List.nth_exn structural 0) (Option.is_some pre);
      let pre = Option.value pre ~default:opt in
      p (List.nth_exn structural 1)
        (not
           (Ir.Tnode.Placements.is_materialized_peek pre.LL.optimize_ctx.LL.placements
              prod.Tensor.value));
      (* Gated, like [folded_seed_claims]' tensorized claim, on the tile the backend advertises for
         the site's precisions: no f32-input tile, no tensorized seed (HIP; CUDA without tf32). *)
      (match
         Option.map (Autotune.detect_matmul pre.LL.llc) ~f:(fun site ->
             Ll_test.tensorized_matmul_capability ~is_gpu:true ~is_cpu:false ~limits
               ~a:site.Autotune.m_a ~b:site.m_b ~d:site.m_d)
       with
      | Some (`Withheld aggregation) ->
          Verdict.skipped ~aggregation ~backend:backend_name (List.nth_exn structural 2)
      | Some `Advertised | None ->
          p_exists (List.nth_exn structural 2) (gpu_seeds ~limits pre) ~f:(fun q ->
              q.Autotune.sk_mma && q.Autotune.sk_epilogue));
      pre
  in
  (* Gated in itself: on the C backends it executes no seed and reports both claims skipped. *)
  folded_seed_claims ~tag:"zl_seed_" ~what:"folded routine-local product with its tail" ~pre ~out
    ~seed ~want ~matches:(Array.equal Float.equal)
