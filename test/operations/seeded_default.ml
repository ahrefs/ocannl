(* gh-ocannl-1324: the seeded untuned default gives large GPU contractions operand reuse without
   changing a result.

   One dense layer with a bias, [y = w * x + b], backpropagated through [loss = sum (y *. dy)], so
   its weight gradient lowers as backprop lowers every weight gradient of a training step -- [for b,
   s, o, i: w.grad[o,i] += dy[b,s,o] * x[b,s,i]], the contraction loops outermost -- with a
   reduction extent of [nb * ns] = 256, while the forward product reduces over [ni] = 96. The bias
   gradient [b.grad[o] += dy[b,s,o]] reduces over the same rows, which is what lets the default
   fission keep it in the weight gradient's kernel as a companion no sketch covers: the seeding has
   to cut that segment finer to reach the site.

   The routine is compiled twice from fresh contexts, on the default pipeline and through
   [Autotune.seeded_default] at a threshold of 128 (between the two reduction extents), and both
   gradients are compared bitwise. The inputs are NOT dyadic: every partial sum rounds, so equality
   holds only if the seeded schedule keeps each output's summation order -- the property that keeps
   a training run's losses unchanged. On CUDA and HIP the seeding must seed, refine and not fall
   back; elsewhere it must leave the default alone. *)

(* cc-fp-contract: same-shape -- cc seeds no untuned default, so both sides are the plain compile *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")

(* gh-ocannl-998: the bitwise claims are about summation order, so every product accumulation is an
   explicit FMA -- the contraction licence, pinned on here. With the licence off (the default) the
   update is a plain [acc + a * b], and whether it fuses is the GPU compiler's call, kernel by
   kernel: on HIP (gfx1102) clang's [-ffast-math] contracted the seeded tiled kernel and the default
   kernel differently, which is a different rounding, not a different order, and failed the
   claims. *)
let () = Unix.putenv "OCANNL_FP_CONTRACT" "true"
let seeds_here = (Context.backend_facts_of_name backend_name).Ir.Backend_intf.seeds_untuned_default
let with_mma_seeding f = Utils.with_stated_config [ ("gpu_default_sketch_min_reduction", "1") ] f

let mma_standin =
  let target = Ll_test.cuda_model_target ~limits:Ll_test.f16_mma_limits () in
  (* The CUDA syntax stand-in has no MMA hook. Give the model a matrix unit that accepts the
     scheduled tile so the advertised matrix rate is used; the device leg reads its own hook. *)
  let capabilities =
    {
      target.mt_capabilities with
      mma_accepts =
        (fun ~d_prec:_
          ~a_prec:_
          ~b_prec:_
          ~ta:_
          ~tb:_
          ~m:_
          ~n:_
          ~k:_
          ~fragment_target:_
          ~d:_
          ~a:_
          ~b:_
        -> true);
    }
  in
  { target with mt_capabilities = capabilities }

type mma_ranking = { unrestricted_mma : bool; seeded_scalar : bool; seeding_active : bool }

(* This aligned f16 contraction offers an f32-accumulating tile on CUDA and HIP. The gradient
   fixture is smaller and serves the bitwise parity checks. *)
let mma_picks ~device =
  let half = Ir.Ops.half and single = Ir.Ops.single in
  let n = 256 and k = 512 in
  let init label ~i ~o stride =
    NTDSL.init ~l:label ~prec:half ~i:[ i ] ~o:[ o ]
      ~f:
        (Ll_test.weighted ~weights:[| stride; 31 |] ~modulus:1009 ~offset:(-504.5)
           ~stride:(1. /. 1009.))
      ()
  in
  let a = init "sd_mma_a" ~i:k ~o:n 17 and b = init "sd_mma_b" ~i:n ~o:k 43 in
  let%op out = a * b in
  Ir.Tnode.update_prec out.Tensor.value single;
  let comp = Train.forward out in
  let ctx = Context.auto () in
  let bindings = Ir.Indexing.Empty in
  let is_mma r = String.is_substring r.Autotune.mc_label ~substring:"mma-" in
  let pick () =
    Autotune.model_pick ~name:"sd_mma_pick"
      ?target:(if device then None else Some mma_standin)
      ctx comp bindings
  in
  let control =
    Utils.with_stated_config
      [
        ("gpu_default_sketch_min_reduction", "0");
        ("gpu_default_split_reduce_min_serial", "0");
        ("gpu_default_reduction_sink", "false");
      ]
      pick
  in
  let seeded, seeding_active =
    with_mma_seeding (fun () ->
        let facts, limits =
          if device then (Context.backend_facts ctx, Context.hardware_limits ctx)
          else (mma_standin.Autotune.mt_facts, mma_standin.Autotune.mt_limits)
        in
        let seeding_active = Option.is_some (Autotune.model_seeding ~facts ~limits) in
        let report =
          if device then (
            let report = ref None in
            let compiled, _ =
              Autotune.model_default ~report:(fun r -> report := Some r) ctx comp bindings
            in
            Context.release compiled;
            Option.value_exn ~here:[%here] !report)
          else pick ()
        in
        (report, seeding_active))
  in
  Stdio.eprintf "%s matmul: unrestricted %s, seeded %s (not part of the golden)\n"
    (if device then "device" else "structural")
    control.Autotune.mc_label seeded.Autotune.mc_label;
  {
    unrestricted_mma = is_mma control;
    seeded_scalar =
      Option.is_some seeded.Autotune.mc_rank_ms
      && not (is_mma seeded || String.is_substring seeded.Autotune.mc_label ~substring:"conv-");
    seeding_active;
  }

let host_mma_child = "--host-mma-ranking"
let device_mma_child = "--device-mma-ranking"

let () =
  let requested flag = Array.exists Stdlib.Sys.argv ~f:(String.equal flag) in
  let device = requested device_mma_child in
  if device || requested host_mma_child then (
    let { unrestricted_mma; seeded_scalar; seeding_active } = mma_picks ~device in
    Stdio.printf "%b\n%b\n%b\n" unrestricted_mma seeded_scalar seeding_active;
    Stdlib.exit 0)

let nb = 4
and ns = 64
and no = 64
and ni = 96

(* Deterministic values with full mantissas: [k / 1009 - 1/2] over a stride coprime to 1009. *)
let values ~n ~stride =
  Array.init n ~f:(fun c -> (Float.of_int (c * stride % 1009) /. 1009.) -. 0.5)

let w =
  Operation.init ~l:"sd_w" ~prec:Ir.Ops.single ~i:[ ni ] ~o:[ no ]
    ~f:(fun idx ->
      Ll_test.weighted ~weights:[| 31; 17 |] ~modulus:1013 ~offset:(-506.5) ~stride:(1. /. 1013.)
        idx)
    ~grad_spec:Tensor.Require_grad ()

let b =
  Operation.init ~l:"sd_b" ~prec:Ir.Ops.single ~o:[ no ]
    ~f:(fun idx -> Float.of_int (idx.(0) * 13 % 101) /. 101.)
    ~grad_spec:Tensor.Require_grad ()

let x_values = values ~n:(nb * ns * ni) ~stride:389
let dy_values = values ~n:(nb * ns * no) ~stride:577
let x = NTDSL.ndarray x_values ~label:[ "sd_x" ] ~batch_dims:[ nb; ns ] ~output_dims:[ ni ] ()
let dy = NTDSL.ndarray dy_values ~label:[ "sd_dy" ] ~batch_dims:[ nb; ns ] ~output_dims:[ no ] ()

(* The fixture's order sensitivity, the control that gives the bitwise claims below their teeth:
   [w.grad] reduced on the host in f32, over the rows in REVERSE order. *)
let f32 v = Int32.float_of_bits (Int32.bits_of_float v)

let wgrad_reversed =
  Array.init (no * ni) ~f:(fun c ->
      let o = c / ni and i = c % ni in
      let acc = ref 0. in
      for r = (nb * ns) - 1 downto 0 do
        acc := f32 (!acc +. f32 (f32 dy_values.((r * no) + o) *. f32 x_values.((r * ni) + i)))
      done;
      !acc)

let%op y = (w * x) + b
let%op loss = (y *. dy) ++ "... => 0"
let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad
let wgrad = grad_of w
let bgrad = grad_of b

let () =
  Train.set_materialized wgrad;
  Train.set_materialized bgrad

let comp = Train.grad_update loss
let bitwise a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

(* One compile and run from a fresh context: the two gradients' values, and the seeding report. *)
let run compile =
  let ctx, routine = compile (Context.auto ()) comp Ir.Indexing.Empty in
  let ctx = Context.run ctx routine in
  Context.sync ctx;
  (Context.get_values ctx wgrad, Context.get_values ctx bgrad)

let seeded_gated seed_gate =
  let report = ref None in
  let values =
    run (fun ctx comp bindings ->
        Autotune.seeded_default ~seed_gate ~report:(fun r -> report := Some r) ctx comp bindings)
  in
  (values, Option.value_exn ~here:[%here] !report)

let seeded ~min_reduction = seeded_gated (Autotune.Min_reduction min_reduction)

(* The structural leg (gh-ocannl-1626): per-PR CI runs cc, where the CUDA/HIP claims below are
   skipped. What the seeded default seeds and refines is its plan, derived on the host
   ([Autotune.seeded_plan]), so it is derived here for "cuda" whatever the host, on the lowering
   this host captures, at the executed leg's thresholds and gates. The controls: the same derivation
   for cc (where the seeded default leaves the default alone) derives no plan, and above every
   reduction extent the plan seeds nothing, so the census is the seeding's, not the fixture's. *)
let () =
  let opt = Context.lowered_for_decisions (Context.auto ()) comp Ir.Indexing.Empty in
  let derive ?(limits = Ir.Backend_intf.no_hardware_limits) ?min_reduction ?seed_gate backend =
    Autotune.seeded_plan ?min_reduction ?seed_gate
      ~facts:(Context.backend_facts_of_name backend)
      ~limits ~accum_prec:Fn.id ~static_indices:[] opt
  in
  let rated = Ll_test.rated_limits in
  let plan ~backend ~min_reduction = Ll_test.seeded_census (derive ~min_reduction backend) in
  let cuda = plan ~backend:"cuda" ~min_reduction:128 in
  p "structural: the plan derived for cuda seeds a segment at the threshold of 128"
    (Option.exists cuda ~f:(fun c -> c.Ll_test.sketched >= 1));
  p "structural: it cuts the bias gradient's companion away to seed the weight gradient"
    (Option.exists cuda ~f:(fun c -> c.Ll_test.refined >= 1));
  p "structural control: the same derivation for cc derives no plan"
    (Option.is_none (plan ~backend:"cc" ~min_reduction:128));
  p "structural control: above every reduction extent the plan derived for cuda seeds nothing"
    (Option.for_all
       (plan ~backend:"cuda" ~min_reduction:(nb * ns * 2))
       ~f:(fun c -> c.Ll_test.sketched = 0 && c.refined = 0));
  let ranked capacity =
    Autotune.Ranked
      {
        capacity;
        cache_rate = None;
        scalar_rate = None;
        dram_rate = None;
        matrix_rate = None;
        gather_rate = None;
        fill = None;
        sector_rate = None;
        wg_capacity = None;
        copy_rate = None;
      }
  in
  let decisions plan =
    match plan with
    | Some { Autotune.sp_segments = Some (_, ds); _ } ->
        List.map ds ~f:(function
          | Autotune.Seeded_sketch { params; _ } -> Autotune.sketch_params_label params
          | Autotune.Seeded_sunk { swaps; _ } -> Printf.sprintf "sunk %d" swaps
          | Autotune.Seeded_preset -> "preset")
    | _ -> []
  in
  let d32 = decisions (derive ~min_reduction:32 "cuda") in
  let d_auto = decisions (derive ~limits:rated ~seed_gate:(ranked (Some 0)) "cuda") in
  let d_cached = decisions (derive ~limits:rated ~seed_gate:(ranked (Some Int.max_value)) "cuda") in
  let sketched ds =
    List.count ds ~f:(fun d -> not (String.equal d "preset" || String.is_prefix d ~prefix:"sunk"))
  in
  Stdio.eprintf "structural: t32 [%s] auto [%s] cached [%s] (not part of the golden)\n"
    (String.concat ~sep:"; " d32) (String.concat ~sep:"; " d_auto)
    (String.concat ~sep:"; " d_cached);
  (* The [auto] gate prices a sketch against the preset at the device's rated ceilings, which
     [no_hardware_limits] leaves unset: stand-in rated limits, as cost_model_reuse states them. *)
  p
    "structural: on rated limits, past capacity the auto gate decides for cuda what threshold 32 \
     decides"
    (sketched d32 >= 1 && List.equal String.equal d_auto d32);
  p
    "structural: on rated limits, below capacity the auto gate seeds fewer segments than threshold \
     32"
    (sketched d_cached < sketched d32)

let () =
  let w_default, b_default = run Context.compile in
  let (w_seeded, b_seeded), r = seeded ~min_reduction:128 in
  List.iter r.Autotune.sd_seeded ~f:(fun (at, p) ->
      Stdio.eprintf "seeded: unit %d %s (not part of the golden)\n" at
        (Autotune.sketch_params_label p));
  Verdict.gated ~when_:seeds_here ~on:backend_name
    "on CUDA/HIP the seeded default seeds a segment at a threshold below the weight gradient's \
     reduction"
    (not (List.is_empty r.sd_seeded));
  Verdict.gated ~when_:seeds_here ~on:backend_name
    "on CUDA/HIP it cuts the bias gradient's companion away from the weight gradient to seed it"
    (r.sd_refined >= 1);
  Verdict.gated ~when_:seeds_here ~on:backend_name
    "on CUDA/HIP the seeded routine compiles without falling back to the default"
    (not r.sd_fell_back);
  Verdict.gated ~when_:(not seeds_here) ~on:backend_name
    "off CUDA/HIP the seeded default seeds nothing"
    (List.is_empty r.sd_seeded && r.sd_refined = 0);
  p_exists "the weight gradient is not trivially zero" (Array.to_list w_default) ~f:(fun v ->
      Float.(abs v > 1e-3));
  p_exists "the fixture is order-sensitive: a reversed f32 reduction differs from it in some cell"
    (List.zip_exn (Array.to_list wgrad_reversed) (Array.to_list w_default))
    ~f:(fun (a, b) -> not (bitwise a b));
  p_all2 "the seeded weight gradient is bitwise the default pipeline's" w_seeded w_default
    ~f:bitwise;
  p_all2 "the seeded bias gradient is bitwise the default pipeline's" b_seeded b_default ~f:bitwise;
  (* The threshold: above every reduction extent nothing qualifies, and 0 disables the seeding. *)
  let (w_high, _), r_high = seeded ~min_reduction:(nb * ns * 2) in
  p "a threshold above every reduction extent seeds nothing"
    (List.is_empty r_high.sd_seeded && r_high.sd_refined = 0);
  p_all2 "that routine computes the default pipeline's weight gradient" w_high w_default ~f:bitwise;
  let _, r_off = seeded ~min_reduction:0 in
  p "a threshold of 0 disables the seeding" (List.is_empty r_off.sd_seeded);
  (* gh-ocannl-1253, the [auto] gate: seed where the ranking score puts the sketch below the preset.
     With a capacity every working set exceeds, the charge is on everywhere, and the gate must seed
     exactly what threshold 32 seeds here (both matmul sites, refinement included); with a capacity
     none reaches, the scores are distinct-cell rooflines and it must seed less. *)
  let census r =
    List.map r.Autotune.sd_seeded ~f:(fun (at, p) -> (at, Autotune.sketch_params_label p))
  in
  let _, r32 = seeded ~min_reduction:32 in
  let (w_auto, b_auto), r_auto =
    seeded_gated
      (Autotune.Ranked
         {
           capacity = Some 0;
           cache_rate = None;
           scalar_rate = None;
           dram_rate = None;
           matrix_rate = None;
           gather_rate = None;
           fill = None;
           sector_rate = None;
           wg_capacity = None;
           copy_rate = None;
         })
  in
  let _, r_auto_cached =
    seeded_gated
      (Autotune.Ranked
         {
           capacity = Some Int.max_value;
           cache_rate = None;
           scalar_rate = None;
           dram_rate = None;
           matrix_rate = None;
           gather_rate = None;
           fill = None;
           sector_rate = None;
           wg_capacity = None;
           copy_rate = None;
         })
  in
  List.iter (census r_auto) ~f:(fun (at, label) ->
      Stdio.eprintf "auto seeded: unit %d %s (not part of the golden)\n" at label);
  Verdict.gated ~when_:seeds_here ~on:backend_name
    "on CUDA/HIP, past capacity the auto gate seeds the census threshold 32 seeds"
    ((not (List.is_empty (census r32)))
    && List.equal (fun (a, l) (b, m) -> a = b && String.equal l m) (census r_auto) (census r32));
  Verdict.gated ~when_:seeds_here ~on:backend_name
    "on CUDA/HIP, below capacity the auto gate seeds fewer segments than threshold 32"
    (List.length r_auto_cached.sd_seeded < List.length r32.sd_seeded);
  p_all2 "the auto-gated weight gradient is bitwise the default pipeline's" w_auto w_default
    ~f:bitwise;
  p_all2 "the auto-gated bias gradient is bitwise the default pipeline's" b_auto b_default
    ~f:bitwise

(* gh-ocannl-1253: the analytic model starts from the seeded plan. On CUDA/HIP [model_default]'s
   incumbent is the seeded untuned plan under the configured gate ([auto] unless the test config
   says otherwise), and its sketch candidates are the seeding's numerics-preserving forms, so its
   pick -- whichever it is -- computes both gradients bitwise as the default pipeline does, on the
   same order-sensitive inputs. Elsewhere the model starts from the default pipeline, and its pick
   may round differently (the C backends' packed mma sketches), so only the report is checked. *)
let () =
  let report = ref None in
  let w_model, b_model =
    run (fun ctx comp bindings ->
        Autotune.model_default ~report:(fun r -> report := Some r) ctx comp bindings)
  in
  let w_default, b_default = run Context.compile in
  let r = Option.value_exn ~here:[%here] !report in
  Stdio.eprintf
    "model_default picked: %s, %d candidates not numerics-preserving (not part of the golden)\n"
    r.Autotune.mc_label r.Autotune.mc_excluded;
  p "the model reports a pick" (not (String.is_empty r.Autotune.mc_label));
  Verdict.gated ~when_:seeds_here ~on:backend_name
    "on CUDA/HIP the model starts from the seeded plan: its pick is the seeded plan or beats it, \
     never the plain default pipeline"
    (not (String.equal r.Autotune.mc_label "default"));
  Verdict.gated_all ~when_:seeds_here ~on:backend_name
    "on CUDA/HIP the model's weight gradient is bitwise the default pipeline's"
    (List.zip_exn (Array.to_list w_model) (Array.to_list w_default))
    ~f:(fun (m, d) -> bitwise m d);
  Verdict.gated_all ~when_:seeds_here ~on:backend_name
    "on CUDA/HIP the model's bias gradient is bitwise the default pipeline's"
    (List.zip_exn (Array.to_list b_model) (Array.to_list b_default))
    ~f:(fun (m, d) -> bitwise m d);
  (* gh-ocannl-1780: [model_pick] is [model_default]'s decision without the compile, the link the
     structural legs below rest on. Where this backend rates an envelope (the GPUs), the model ranks
     and the two must name the same pick; elsewhere both report the default pipeline unranked. *)
  let limits = Context.hardware_limits (Context.auto ()) in
  Verdict.gated
    ~when_:(Option.is_some limits.peak_flops || Option.is_some limits.peak_memory_bandwidth)
    ~on:backend_name
    "where the backend rates an envelope, model_pick names model_default's pick without compiling"
    (String.equal
       (Autotune.model_pick ~name:"sd_pick_here" (Context.auto ()) comp Ir.Indexing.Empty)
         .Autotune.mc_label
       r.Autotune.mc_label)

(* gh-ocannl-1780: the model claims' premise on every backend. [Autotune.model_pick] decides what
   [model_default] would compile, for a CUDA stand-in target ([Ll_test.cuda_model_target]: CUDA's
   facts, the CUDA dialect's capabilities, rated ceilings), on this host's hermetic lowering; the
   bitwise claims stay the device's. The leg is the configuration's: where the untuned seeding is
   off for that target, [model_default] starts from the default pipeline there too. The control
   takes the envelope away: without rated ceilings the model ranks nothing and picks the default. *)
let () =
  let target = Ll_test.cuda_model_target () in
  let seeds_for_target =
    Option.is_some
      (Autotune.model_seeding ~facts:target.Autotune.mt_facts ~limits:target.Autotune.mt_limits)
  in
  let leg =
    Verdict.leg ~aggregation:`Environment ~when_:seeds_for_target
      ~on:(backend_name ^ ", untuned seeding off for the cuda stand-in")
      ()
  in
  let pick =
    lazy (Autotune.model_pick ~name:"sd_pick" ~target (Context.auto ()) comp Ir.Indexing.Empty)
  in
  Verdict.in_leg ~leg
    "structural: for the cuda stand-in the model starts from the seeded plan: its pick is never \
     the plain default pipeline" (fun label ->
      let r = Lazy.force pick in
      Stdio.eprintf "structural: model_pick for cuda: %s (not part of the golden)\n"
        r.Autotune.mc_label;
      p label (not (String.equal r.Autotune.mc_label "default")));
  p "structural control: without rated ceilings the cuda stand-in's pick is the default pipeline"
    (String.equal
       (Autotune.model_pick ~name:"sd_pick_unrated"
          ~target:{ target with mt_limits = Ir.Backend_intf.no_hardware_limits }
          (Context.auto ()) comp Ir.Indexing.Empty)
         .Autotune.mc_label
       "default")

(* Run the ranking fixture in a fresh process so its price parameters reach the model's
   process-lifetime readers without changing the gradient checks. The device supplies its real tile
   and hook, with the same rates as the stand-in. The child's stdout protocol is exactly three
   boolean lines: the unrestricted pick, the restricted pick and whether seeding was active. *)
let () =
  let choices mode =
    let ((status, stdout, stderr) as child) =
      Fresh_process.run
        [
          mode;
          "--ocannl_backend=" ^ backend_name;
          "--ocannl_automatic_gpu_schedule=true";
          "--ocannl_schedule_fission=true";
          "--ocannl_debug_log_from_routines=false";
          "--ocannl_model_peak_flops=1e14";
          "--ocannl_model_peak_memory_bandwidth=1e15";
          "--ocannl_model_issue_rate=1e12";
          "--ocannl_model_mma_rate=1e14";
          "--ocannl_model_dram_bandwidth=1e15";
          "--ocannl_model_cache_capacity=none";
          "--ocannl_model_cache_bandwidth=none";
          "--ocannl_model_fill_threads=none";
          "--ocannl_model_workgroup_bytes_per_multiprocessor=none";
          "--ocannl_model_device_gather_rate=none";
          "--ocannl_model_transaction_rate=none";
          "--ocannl_model_staged_copy_rate=none";
          "--ocannl_model_shared_load_rate=none";
        ]
    in
    match (status, String.split_lines stdout) with
    | Unix.WEXITED 0, [ unrestricted; restricted; active ] ->
        Stdio.eprintf "%s%!" stderr;
        {
          unrestricted_mma = Stdlib.bool_of_string unrestricted;
          seeded_scalar = Stdlib.bool_of_string restricted;
          seeding_active = Stdlib.bool_of_string active;
        }
    | _ ->
        Fresh_process.report ~label:mode child;
        failwith "seeded model ranking child failed"
  in
  let host = choices host_mma_child in
  p "structural: without seeding, the model picks a tensor-core leaf" host.unrestricted_mma;
  p "structural: with seeding, the ranked pick is no tensor-core or conv form" host.seeded_scalar;
  p "structural: seeding is active under the fixture's seeded settings" host.seeding_active;
  let available, aggregation =
    if not seeds_here then (false, `Backend)
    else
      let ctx = Context.auto () in
      let limits = Context.hardware_limits ctx in
      match
        Ll_test.tensorized_capability ~is_gpu:true ~is_cpu:false ~limits ~a_prec:Ir.Ops.half
          ~b_prec:Ir.Ops.half ~d_prec:Ir.Ops.single
      with
      | `Advertised -> (true, `Backend)
      | `Withheld aggregation -> (false, aggregation)
  in
  let gpu_leg = Verdict.leg ~when_:available ~aggregation ~on:backend_name () in
  let gpu = lazy (choices device_mma_child) in
  Verdict.in_leg ~leg:gpu_leg "on CUDA/HIP without seeding, the model picks a tensor-core leaf"
    (fun label -> p label (Lazy.force gpu).unrestricted_mma);
  Verdict.in_leg ~leg:gpu_leg
    "on CUDA/HIP with seeding, the ranked pick is no tensor-core or conv form" (fun label ->
      p label (Lazy.force gpu).seeded_scalar);
  Verdict.in_leg ~leg:gpu_leg "on CUDA/HIP seeding is active under the fixture's seeded settings"
    (fun label -> p label (Lazy.force gpu).seeding_active)
