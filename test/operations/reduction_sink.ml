(* gh-ocannl-1348: the untuned GPU compile sinks a batch reduction's loops below its write loops.

   Backprop lowers a bias gradient as [for b, s, o: db[o] += dy[b,s,o]] -- the reduction loops
   outermost -- so the default GPU preset maps [o] to hardware BELOW the serial [b] and [s] loops,
   and every thread reads and writes its device cell once per reduction step. NVRTC promotes that
   cell to a register on its own; HIP's compiler does not where the preset's guard sits inside the
   loops (gpt2_mini's layer-norm gamma/beta gradients). [Autotune.seeded_default] therefore gives
   each segment it does not seed [Autotune.reduction_sink]'s adjacent-[Swap] chain first, so the
   preset maps the write loops and code generation localizes the accumulator.

   Two parts: - Which statements qualify, on hand-built nests (every backend; the chain is a pure
   function of the code): a bias-shaped reduction gets exactly the two swaps sinking [s] then [b]
   below [o]; the forward order, an accumulator a reduction loop's [Affine] index also moves, an
   overwrite, and a narrow-storage accumulator its backend widens get none. - One dense layer with a
   bias, backpropagated, compiled on the default pipeline and through [Autotune.seeded_default] with
   the sketch seeding OFF (threshold 0), so every change is the sink's. Both gradients must be
   BITWISE the default's: the inputs are not dyadic, and a control claim shows a reversed f32
   reduction of the bias gradient differs from it in some cell, so the equality holds only if the
   sink keeps each cell's summation order. On capable backends the sink must apply, leave fewer
   accumulations rendered against their cells (the peel census), and be turned off by
   [~reduction_sink:false]; elsewhere it must sink nothing. *)

(* cc-fp-contract: same-shape -- on cc seeded_default sinks nothing, so both sides compile the same
   kernel *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Idx = Ir.Indexing
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let facts = Context.backend_facts_of_name backend_name
let sinks_here = facts.Ir.Backend_intf.sinks_untuned_reductions

(* {2 Which statements qualify} *)

let nr = 4
and nq = 8
and no = 16

let node = Ll_test.node_factory ~first_id:134800 ~dims:[| 1 |] ()
let node_bf16 = Ll_test.node_factory ~prec:Ir.Ops.bfloat16 ~first_id:134900 ~dims:[| 1 |] ()
let it = Ll_test.iter

(* The chain, with the optimized code's top-level statements: the population an empty chain was
   derived from, which must exist. *)
let chain ?(accum_prec = Fn.id) ~name llc =
  let opt = Ll_test.optimize ~name llc in
  (Autotune.reduction_sink ~accum_prec opt, LL.flat_lines [ opt.LL.llc ])

let as_swap = function Sched.Swap s -> Some (s.outer, s.inner) | _ -> None

let swaps_are expected ops =
  List.equal
    (Option.equal (fun (o1, i1) (o2, i2) -> Idx.equal_symbol o1 o2 && Idx.equal_symbol i1 i2))
    (List.map expected ~f:Option.some)
    (List.map ops ~f:as_swap)

(* [for r, q, o: d[o] = f(d[o], a[r,q,o])], the bias gradient's order, with [f] the accumulation
   [acc] builds; [idx] is the accumulator's index map over [(r, q, o)]. *)
let reduce3 ~tag ?(narrow = false) ?(acc = fun d a -> Ll_test.add_own d a) ~idx ~d_dims () =
  let a = node ~dims:[| nr; nq; no |] ("rs_" ^ tag ^ "_a") in
  let d = (if narrow then node_bf16 else node) ~dims:d_dims ("rs_" ^ tag ^ "_d") in
  List.iter [ a; d ] ~f:Ll_test.materialize;
  let r = Ll_test.sym () and q = Ll_test.sym () and o = Ll_test.sym () in
  let di = idx r q o in
  ( (r, q, o),
    Ll_test.seq (Ll_test.zero d)
      (Ll_test.loop_n r nr
         (Ll_test.loop_n q nq
            (Ll_test.loop_n o no
               (Ll_test.set d di (acc (Ll_test.get d di) (Ll_test.get a [| it r; it q; it o |]))))))
  )

let () =
  let (r, q, o), bias = reduce3 ~tag:"bias" ~idx:(fun _ _ o -> [| it o |]) ~d_dims:[| no |] () in
  p "a bias-shaped reduction sinks q then r below o, and nothing else"
    (swaps_are [ (q, o); (r, o) ] (fst (chain ~name:"rs_bias" bias)));
  (* A single batch axis, [for b, o: db[o] += dy[b,o]]: the shortest nest the sink takes. *)
  let a2 = node ~dims:[| nr; no |] "rs_two_a" and d2 = node ~dims:[| no |] "rs_two_d" in
  List.iter [ a2; d2 ] ~f:Ll_test.materialize;
  let r2 = Ll_test.sym () and o2 = Ll_test.sym () in
  let two =
    Ll_test.seq (Ll_test.zero d2)
      (Ll_test.loop_n r2 nr
         (Ll_test.loop_n o2 no
            (Ll_test.set d2
               [| it o2 |]
               (Ll_test.add (Ll_test.get d2 [| it o2 |]) (Ll_test.get a2 [| it r2; it o2 |])))))
  in
  p "a two-loop bias reduction sinks its one reduction loop below o"
    (swaps_are [ (r2, o2) ] (fst (chain ~name:"rs_two" two)));
  let a = node ~dims:[| no; nr |] "rs_fwd_a" and d = node ~dims:[| no |] "rs_fwd_d" in
  List.iter [ a; d ] ~f:Ll_test.materialize;
  let o' = Ll_test.sym () and r' = Ll_test.sym () in
  let fwd =
    Ll_test.seq (Ll_test.zero d)
      (Ll_test.loop_n o' no
         (Ll_test.loop_n r' nr
            (Ll_test.set d
               [| it o' |]
               (Ll_test.add (Ll_test.get d [| it o' |]) (Ll_test.get a [| it o'; it r' |])))))
  in
  let ops, over = chain ~name:"rs_fwd" fwd in
  p_empty "a reduction already innermost gets no interchange" ~over ops;
  let _, shifted =
    reduce3 ~tag:"aff"
      ~idx:(fun r _ o -> [| Ll_test.aff [ (1, o); (1, r) ] 0 |])
      ~d_dims:[| no + nr |]
      ()
  in
  let ops, over = chain ~name:"rs_aff" shifted in
  p_empty "an accumulator a reduction loop's affine index also moves is left alone" ~over ops;
  p "a plain-cell gradient sinks beside an overlapping-window input gradient"
    (swaps_are [ (q, o); (r, o) ] (fst (chain ~name:"rs_mixed" (Ll_test.seq bias shifted))));
  let _, overwrite =
    reduce3 ~tag:"ovw" ~acc:(fun _ a -> a) ~idx:(fun _ _ o -> [| it o |]) ~d_dims:[| no |] ()
  in
  let ops, over = chain ~name:"rs_ovw" overwrite in
  p_empty "an overwrite (no self-read) is not an accumulation and is left alone" ~over ops;
  let _, narrow =
    reduce3 ~tag:"bf16" ~narrow:true ~idx:(fun _ _ o -> [| it o |]) ~d_dims:[| no |] ()
  in
  (* gh-ocannl-2362: the optimizer sinks a narrow plain-cell accumulation itself
     ([LL.sink_narrow_reduction_loops]), so on optimized code the chain has nothing to do whichever
     the accumulator; the sink's narrow rule is pinned on the nest as built
     ([Ll_test.optimize_scoped]), the order a nest the optimizer's narrower allowlist declines
     keeps. *)
  let sunk = Ll_test.optimize ~name:"rs_bf16_sunk" narrow in
  p_empty "the optimizer has already sunk the bf16 accumulator: no chain, even at storage precision"
    ~over:(LL.flat_lines [ sunk.LL.llc ])
    (Autotune.reduction_sink ~accum_prec:Fn.id sunk);
  let opt = Ll_test.optimize_scoped ~name:"rs_bf16" ~raw:narrow narrow in
  p_empty "a bf16 accumulator its backend widens to f32 is left alone"
    ~over:(LL.flat_lines [ opt.LL.llc ])
    (Autotune.reduction_sink ~accum_prec:(fun _ -> Ir.Ops.single) opt);
  p "the same bf16 accumulator kept at its storage precision is sunk (the control)"
    (not (List.is_empty (Autotune.reduction_sink ~accum_prec:Fn.id opt)));
  (* gh-ocannl-1405: the narrow-storage admission takes it under the widening accumulator too, with
     the chain it takes at storage-width accumulation -- the admission changes which statements
     qualify, not the interchange. *)
  p "admitted, the widened bf16 accumulator is sunk by the chain it gets at its storage precision"
    (let admitted =
       Autotune.reduction_sink ~admit_narrow:true ~accum_prec:(fun _ -> Ir.Ops.single) opt
     in
     let render sched = Sexp.to_string (Sched.sexp_of_schedule sched) in
     (not (List.is_empty admitted))
     && String.equal (render admitted) (render (Autotune.reduction_sink ~accum_prec:Fn.id opt)));
  (* gh-ocannl-1741: [Ll_test.schedule_op_lines], the debugging print of an op list -- the admitted
     chain and a [Privatize] targeting a node, which prints as its label and id rather than the
     node's record. The lines go to stderr: their fresh symbols' numbering is no part of the
     claim. *)
  let target = node "rs_ops_target" in
  let sched =
    Autotune.reduction_sink ~admit_narrow:true ~accum_prec:(fun _ -> Ir.Ops.single) opt
    @ [ Sched.Privatize { target; over = Ll_test.sym (); acc_prec = Ir.Ops.single } ]
  in
  let shown = Ll_test.schedule_op_lines sched in
  List.iter shown ~f:(Stdio.eprintf "schedule_op_lines (not part of the golden): %s\n");
  p "schedule_op_lines prints one line per op, a node as its label#id and never its record"
    (List.length shown = List.length sched
    && List.exists shown
         ~f:(String.is_substring ~substring:(Printf.sprintf "rs_ops_target#%d" target.Ir.Tnode.id))
    && not (List.exists shown ~f:(String.is_substring ~substring:"storage_prec")))

(* {2 Executed: a backpropagated dense layer} *)

let nb = 4
and ns = 64
and nout = 64
and nin = 32

(* Deterministic values with full mantissas: [k / 1009 - 1/2] over a stride coprime to 1009. *)
let values ~n ~stride =
  Array.init n ~f:(fun c -> (Float.of_int (c * stride % 1009) /. 1009.) -. 0.5)

let w =
  Operation.init ~l:"rs_w" ~prec:Ir.Ops.single ~i:[ nin ] ~o:[ nout ]
    ~f:(fun idx ->
      Ll_test.weighted ~weights:[| 31; 17 |] ~modulus:1013 ~offset:(-506.5) ~stride:(1. /. 1013.)
        idx)
    ~grad_spec:Tensor.Require_grad ()

let b =
  Operation.init ~l:"rs_b" ~prec:Ir.Ops.single ~o:[ nout ]
    ~f:(fun idx -> Float.of_int (idx.(0) * 13 % 101) /. 101.)
    ~grad_spec:Tensor.Require_grad ()

let x_values = values ~n:(nb * ns * nin) ~stride:389
let dy_values = values ~n:(nb * ns * nout) ~stride:577
let x = NTDSL.ndarray x_values ~label:[ "rs_x" ] ~batch_dims:[ nb; ns ] ~output_dims:[ nin ] ()
let dy = NTDSL.ndarray dy_values ~label:[ "rs_dy" ] ~batch_dims:[ nb; ns ] ~output_dims:[ nout ] ()

(* The control giving the bitwise claims their teeth: [b.grad] reduced on the host in f32 over the
   rows in REVERSE order. *)
let f32 v = Int32.float_of_bits (Int32.bits_of_float v)

let bgrad_reversed =
  Array.init nout ~f:(fun o ->
      let acc = ref 0. in
      for r = (nb * ns) - 1 downto 0 do
        acc := f32 (!acc +. f32 dy_values.((r * nout) + o))
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

(* One compile and run from a fresh context: the gradients and the routine's peel census. *)
let run compile =
  let ctx, routine = compile (Context.auto ()) comp Ir.Indexing.Empty in
  let ctx = Context.run ctx routine in
  Context.sync ctx;
  (Context.get_values ctx wgrad, Context.get_values ctx bgrad, routine.Context.peel)

let sunk ?reduction_sink () =
  let report = ref None in
  let values =
    run (fun ctx comp bindings ->
        Autotune.seeded_default ~min_reduction:0 ?reduction_sink
          ~report:(fun r -> report := Some r)
          ctx comp bindings)
  in
  (values, Option.value_exn ~here:[%here] !report)

(* The structural leg (gh-ocannl-1626): per-PR CI runs cc, where the claims gated on [sinks_here]
   below are skipped. Which segments the untuned compile sinks is the seeded plan's decision,
   derived on the host ([Autotune.seeded_plan]), so it is derived here for "cuda" whatever the host,
   on the lowering this host captures, with the seeding off as below. The controls: with the sink
   off, and for cc, nothing is sunk. *)
let () =
  let opt = Context.lowered_for_decisions (Context.auto ()) comp Ir.Indexing.Empty in
  let open Ir.Backend_intf in
  let plan ?(min_reduction = 0) ?min_serial ?seed_narrow ~facts reduction_sink =
    Autotune.seeded_plan ~min_reduction ?min_serial ?seed_narrow ~reduction_sink ~facts
      ~limits:no_hardware_limits ~accum_prec:Fn.id ~static_indices:[] opt
  in
  let census ?(facts = cuda_facts) ?min_serial reduction_sink =
    Ll_test.seeded_census (plan ~facts ?min_serial reduction_sink)
  in
  p "structural: for cuda the untuned plan sinks a segment with the seeding off, seeding none"
    (Option.exists (census true) ~f:(fun c -> c.Ll_test.sunk >= 1 && c.sketched = 0));
  p "structural: the untuned Metal plan sinks a segment without sketching or splitting"
    (Option.exists (census ~facts:metal_facts true) ~f:(fun c ->
         c.Ll_test.sunk >= 1 && c.sketched = 0 && List.is_empty c.split));
  p "structural: Metal still only sinks with sketching and narrow admission requested"
    (let digest = Option.map ~f:Autotune.seeded_plan_digest in
     let ordinary = digest (plan ~facts:metal_facts true) in
     Option.exists ordinary ~f:(fun d ->
         Option.equal String.equal (Some d)
           (digest (plan ~facts:metal_facts ~min_reduction:1 ~min_serial:0 ~seed_narrow:true true))));
  p "structural control: Metal with the sink and split off derives no plan"
    (Option.is_none (census ~facts:metal_facts ~min_serial:0 false));
  p "structural control: for cuda with the sink off nothing is sunk"
    (Option.for_all (census false) ~f:(fun c -> c.Ll_test.sunk = 0));
  p "structural control: for cc no plan is derived" (Option.is_none (census ~facts:cc_facts true));
  p "structural: an inactive Metal schedule ignores a malformed sink flag"
    (let saved_log = Utils.settings.debug_log_from_routines
     and saved_level = Utils.settings.log_level in
     Exn.protect
       ~finally:(fun () ->
         Utils.settings.debug_log_from_routines <- saved_log;
         Utils.settings.log_level <- saved_level)
       ~f:(fun () ->
         Utils.settings.debug_log_from_routines <- true;
         Utils.settings.log_level <- 2;
         match
           Utils.with_stated_config
             [ ("gpu_default_reduction_sink", "invalid") ]
             (fun () ->
               Autotune.seeded_plan ~facts:metal_facts ~limits:no_hardware_limits ~accum_prec:Fn.id
                 ~static_indices:[] opt)
         with
         | None -> true
         | Some _ -> false
         | exception Utils.Invalid_config_value _ -> false));
  let target =
    {
      Autotune.mt_facts = metal_facts;
      mt_limits = { Ll_test.f32_mma_limits with peak_flops = Some 5e12; simdgroup_width = Some 32 };
      mt_capabilities = Ll_test.gpu_like_capabilities ();
    }
  in
  (* A convolution gives the model candidates outside the scalar-blocktile restriction. *)
  let conv_input =
    NTDSL.init ~l:"rs_model_input" ~prec:Ir.Ops.single ~b:[ 2 ] ~o:[ 10; 10; 8 ]
      ~f:(Ll_test.weighted ~weights:[| 1; 1; 2; 3 |] ~modulus:7 ~offset:0. ~stride:1.)
      ()
  and conv_kernel =
    NTDSL.init ~l:"rs_model_kernel" ~prec:Ir.Ops.single ~i:[ 3; 3; 8 ] ~o:[ 16 ]
      ~f:(Ll_test.weighted ~weights:[| 2; 1; 1; 3 |] ~modulus:5 ~offset:(-2.) ~stride:1.)
      ()
  in
  let%op conv =
    conv_input
    +* "...| 1*oh<+kh, 1*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.."
         conv_kernel
  in
  Train.set_materialized conv.Tensor.value;
  let model_comp = Ir.Assignments.sequence [ comp; Train.forward conv ] in
  let explain ?(facts = metal_facts) sink =
    Utils.with_stated_config
      [
        ("gpu_default_reduction_sink", Bool.to_string sink);
        ("gpu_default_sketch_min_reduction", "0");
        ("gpu_default_split_reduce_min_serial", "0");
      ]
      (fun () ->
        Autotune.model_explain ~target:{ target with mt_facts = facts } (Context.auto ()) model_comp
          Ir.Indexing.Empty)
  in
  let on = explain true and off = explain false in
  p "structural: the Metal model compares sketches against the sunk plan when enabled"
    (match (on.Autotune.x_routine, off.Autotune.x_routine) with
    | ("seeded", _) :: _, ("default", _) :: _ -> true
    | _ -> false);
  let candidates (x : Autotune.explanation) =
    List.drop x.x_routine 1
    @ List.concat_map x.x_segments ~f:(fun seg -> List.drop seg.xs_candidates 1)
  in
  let admitted x =
    List.filter_map (candidates x) ~f:(fun (label, row) -> Option.some_if row.ex_admitted label)
    |> Array.of_list
  in
  p_all2 "structural: the Metal model admits the same sketches with the sink on and off"
    (admitted on) (admitted off) ~f:String.equal;
  let restricted = explain ~facts:{ metal_facts with seeds_untuned_default = true } true in
  let ordinary = candidates off in
  p_exists "structural control: the sketch-seeding restriction excludes an ordinary Metal sketch"
    (candidates restricted) ~f:(fun (label, row) ->
      (not row.ex_admitted)
      && Option.exists (List.Assoc.find ordinary label ~equal:String.equal) ~f:(fun x ->
          x.ex_admitted))

let () =
  let w_default, b_default, peel_default = run Context.compile in
  let (w_sunk, b_sunk, peel_sunk), r = sunk ~reduction_sink:true () in
  Stdio.eprintf
    "backend %s: %d segment(s) sunk; peel default: %s; sunk: %s (not part of the golden)\n"
    backend_name (List.length r.Autotune.sd_sunk)
    (Ir.C_syntax.peel_summary_string peel_default)
    (Ir.C_syntax.peel_summary_string peel_sunk);
  let gpu_leg = Verdict.leg ~when_:sinks_here ~on:backend_name () in
  Verdict.in_leg ~leg:gpu_leg "the untuned GPU compile sinks a segment with the seeding off"
    (fun label -> p label (not (List.is_empty r.sd_sunk)));
  Verdict.in_leg ~leg:gpu_leg "the sunk routine compiles without falling back to the default"
    (fun label -> p label (not r.sd_fell_back));
  (* Declined sites are the accumulations rendered against their cells. A sunk one that localizes
     may not be censused at all: the covered-zero elision can open it in the localized scope form
     itself, a scope minted upstream of the peel. *)
  Verdict.in_leg ~leg:gpu_leg "the default renders some accumulation against its cell" (fun label ->
      p label (peel_default.declined > 0));
  Verdict.in_leg ~leg:gpu_leg "the sunk routine renders fewer accumulations against their cells"
    (fun label -> p label (peel_sunk.declined < peel_default.declined));
  let cc_leg = Verdict.leg ~when_:(not sinks_here) ~on:backend_name () in
  Verdict.in_leg ~leg:cc_leg "the backend without the sink compiles without sinking" (fun label ->
      p_empty label ~over:[ r ] r.sd_sunk);
  p "with the seeding off nothing is seeded" (List.is_empty r.sd_seeded);
  p_exists "the bias gradient is not trivially zero" (Array.to_list b_default) ~f:(fun v ->
      Float.(abs v > 1e-3));
  p_exists "the fixture is order-sensitive: a reversed f32 reduction differs from it in some cell"
    (List.zip_exn (Array.to_list bgrad_reversed) (Array.to_list b_default))
    ~f:(fun (a, b) -> not (bitwise a b));
  p_all2 "the sunk bias gradient is bitwise the default pipeline's" b_sunk b_default ~f:bitwise;
  p_all2 "the sunk weight gradient is bitwise the default pipeline's" w_sunk w_default ~f:bitwise;
  let (_, b_off, _), r_off = sunk ~reduction_sink:false () in
  p "reduction_sink:false sinks nothing" (List.is_empty r_off.sd_sunk);
  p_all2 "that routine computes the default pipeline's bias gradient" b_off b_default ~f:bitwise
