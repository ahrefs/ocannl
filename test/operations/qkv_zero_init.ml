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

let projection ~b ~s ~h ~j ~k =
  let x =
    NTDSL.init ~l:"zi_x" ~prec:Ir.Ops.single ~o:[ b; s; k ]
      ~f:(fun ix -> Float.of_int (1 + (3 * ix.(0)) + (5 * ix.(1)) + ix.(2)) /. 16.)
      ()
  in
  let w =
    NTDSL.init ~l:"zi_w" ~prec:Ir.Ops.single ~o:[ h; j; k ]
      ~f:(fun ix -> Float.of_int (1 + (7 * ix.(0)) + (3 * ix.(1)) + ix.(2)) /. 32.)
      ()
  in
  let%op out = x +* "bsk;hjk=>bshj" w in
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

(* The sketch candidates' segmentation folds the zero ([fold_zeros:true], what the autotuner and the
   model selector pass); the untuned default pipeline does not. *)
let fission ?(fold_zeros = true) opt =
  let limits = Ir.Backend_intf.no_hardware_limits in
  Sched.fission_scheduled ~keep_mapping:(Sched.default_gpu ~limits) ~fold_zeros
    ~preset:(Sched.default_gpu ~limits)
    ~zero_sched:(fun tns -> Sched.zero_expansion ~limits tns)
    ~static_indices:[] opt

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

let () =
  let out = projection ~b:32 ~s:32 ~h:4 ~j:32 ~k:128 in
  let opt = capture ~name:"zi_shape" out in
  let parts = fission opt in
  p "gpt2 qkv projection has one kernel including initialization" (List.length parts = 1);
  p "the untuned default keeps the projection's zero in its own kernel"
    (List.map (fission ~fold_zeros:false opt) ~f:(fun (kind, _, _, _) -> kind)
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
  let out = projection ~b:2 ~s:32 ~h:2 ~j:32 ~k:64 in
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
     their accumulator fragment from cells the folded zero nest wrote earlier in the same kernel. *)
  let every_label = "every GPU sketch of the folded qkv matches the materialized run" in
  let tensor_label = "a tensorized folded qkv sketch matches the materialized run" in
  let on_gpu = Sched.backend_is_gpu backend_name in
  let real =
    if on_gpu then gpu_seeds ~limits:(Context.hardware_limits (Context.auto ())) pre else []
  in
  let results =
    List.mapi real ~f:(fun i q ->
        let o = apply (Autotune.sketch_schedule ~accum_prec:Fn.id ~p:q pre) pre in
        let got =
          List.hd_exn
            (Ll_test.execute
               ~name:("zi_seed_" ^ Int.to_string i)
               o ~seed ~read:[ out.Tensor.value ])
        in
        let ok = Array.equal Float.equal got want in
        if not ok then
          Stdio.eprintf "folded qkv seed %d (mma=%b) differs from the materialized run\n" i
            q.Autotune.sk_mma;
        (q, ok))
  in
  (* The gates read the backend and the seeding (hardware-capability facts), never the executed
     values: a GPU that seeds nothing fails the first claim rather than skipping it. *)
  gated_all ~when_:on_gpu ~on:backend_name every_label results ~f:snd;
  gated_exists
    ~when_:(List.exists real ~f:(fun q -> q.Autotune.sk_mma))
    ~on:backend_name tensor_label results
    ~f:(fun (q, ok) -> q.Autotune.sk_mma && ok);
  let ctx, routine =
    Context.compile ~name:"zi_default" ~prelowered:opt
      ~lowered_transform:(fun o -> List.map (fission o) ~f:(fun (_, _, _, scheduled) -> scheduled))
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
  let unexpanded = segments (fission ~fold_zeros:false opt) in
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
      ~lowered_transform:(fun o -> List.map (fission o) ~f:(fun (_, _, _, scheduled) -> scheduled))
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
  let folded = fission opt in
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
    let preset seg = Autotune.sketch_schedule ~accum_prec:Fn.id ~p:q seg in
    match
      Sched.fission_scheduled ~fold_zeros:true ~keep_mapping:(Sched.default_gpu ~limits) ~preset
        ~zero_sched:(fun tns -> Sched.zero_expansion ~limits tns)
        ~static_indices:[]
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
