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
   sink keeps each cell's summation order. On CUDA and HIP the sink must apply, leave fewer
   accumulations rendered against their cells (the peel census), and be turned off by
   [~reduction_sink:false]; elsewhere it must sink nothing. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Idx = Ir.Indexing
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

let sinks_here =
  String.is_substring backend_name ~substring:"cuda"
  || String.is_substring backend_name ~substring:"hip"

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
  let _, overwrite =
    reduce3 ~tag:"ovw" ~acc:(fun _ a -> a) ~idx:(fun _ _ o -> [| it o |]) ~d_dims:[| no |] ()
  in
  let ops, over = chain ~name:"rs_ovw" overwrite in
  p_empty "an overwrite (no self-read) is not an accumulation and is left alone" ~over ops;
  let _, narrow =
    reduce3 ~tag:"bf16" ~narrow:true ~idx:(fun _ _ o -> [| it o |]) ~d_dims:[| no |] ()
  in
  let opt = Ll_test.optimize ~name:"rs_bf16" narrow in
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
     && String.equal (render admitted) (render (Autotune.reduction_sink ~accum_prec:Fn.id opt)))

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

let () =
  let w_default, b_default, peel_default = run Context.compile in
  let (w_sunk, b_sunk, peel_sunk), r = sunk ~reduction_sink:true () in
  Stdio.eprintf
    "backend %s: %d segment(s) sunk; peel default: %s; sunk: %s (not part of the golden)\n"
    backend_name (List.length r.Autotune.sd_sunk)
    (Ir.C_syntax.peel_summary_string peel_default)
    (Ir.C_syntax.peel_summary_string peel_sunk);
  Verdict.gated ~when_:sinks_here ~on:backend_name
    "on CUDA/HIP the untuned compile sinks a segment with the seeding off"
    (not (List.is_empty r.sd_sunk));
  Verdict.gated ~when_:sinks_here ~on:backend_name
    "on CUDA/HIP the sunk routine compiles without falling back to the default" (not r.sd_fell_back);
  (* Declined sites are the accumulations rendered against their cells. A sunk one that localizes
     may not be censused at all: the covered-zero elision can open it in the localized scope form
     itself, a scope minted upstream of the peel. *)
  Verdict.gated ~when_:sinks_here ~on:backend_name
    "on CUDA/HIP the default renders some accumulation against its cell" (peel_default.declined > 0);
  Verdict.gated ~when_:sinks_here ~on:backend_name
    "on CUDA/HIP the sunk routine renders fewer accumulations against their cells"
    (peel_sunk.declined < peel_default.declined);
  Verdict.gated ~when_:(not sinks_here) ~on:backend_name "off CUDA/HIP nothing is sunk"
    (List.is_empty r.sd_sunk);
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
