(* gh-ocannl-2079 investigation, before implementing host rounding. Compare tuned inference from a
   frozen f32 master under a virtual half twin with the same weights frozen directly at half
   precision. The control exposes hoisted candidates without changing the compiler.
   BENCH_FROZEN_SIZE selects a square extent (default 64). Both arms compile before eight
   interleaved Bench_retime calls; search times are diagnostics, never the comparison. This is a
   CPU-only probe. Measurements require the fleet's exclusive reservation and
   benchmarks/check-box.sh; ordinary runs establish structure and exact values only. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module S = Ir.Schedule
module Tn = Ir.Tnode

let n =
  match Stdlib.Sys.getenv_opt "BENCH_FROZEN_SIZE" with
  | None -> 64
  | Some s ->
      let n = Int.of_string s in
      if not (List.mem [ 64; 128; 256 ] n ~equal:Int.equal) then
        invalid_arg "BENCH_FROZEN_SIZE must be 64, 128, or 256";
      n

let dims = [| n; n |]
let input_cell = Ll_test.cycle ~dims ~modulus:7 ~offset:1. ~stride:0.125
let weight_cell = Ll_test.cycle ~dims ~modulus:13 ~offset:8. ~stride:0.125

let make ?(host_rounded = false) () =
  let a = NTDSL.init ~l:"input" ~prec:Ir.Ops.single ~i:[ n ] ~o:[ n ] ~f:input_cell () in
  let values =
    Array.init (n * n) ~f:(fun flat -> weight_cell (Ll_test.unflat ~dims flat) +. (2. **. -12.))
  in
  Tensor.default_value_prec := if host_rounded then Ir.Ops.half else Ir.Ops.single;
  let master = NTDSL.ndarray values ~label:[ "frozen" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  Tensor.default_value_prec := Ir.Ops.single;
  Tn.update_prec master.Tensor.value (if host_rounded then Ir.Ops.half else Ir.Ops.single);
  let b =
    if host_rounded then master else Operation.cast ~grad_spec:Tensor.Prohibit_grad master ()
  in
  Tn.update_prec b.Tensor.value Ir.Ops.half;
  if not host_rounded then Train.set_virtual b.Tensor.value;
  let%op out = a * b in
  Tn.update_prec out.Tensor.value Ir.Ops.single;
  (a, master, out)

let schedule ~hoisted a b opt =
  let i, j, k =
    match List.find_exn (Ll_test.nest_paths opt.LL.llc) ~f:(fun p -> List.length p = 3) with
    | [ i; j; k ] -> (i, j, k)
    | _ -> assert false
  in
  let si, io, ii = S.split ~axis:i ~factor:16 ~outer:LL.Serial ~inner:LL.Serial in
  let sk, ko, ki = S.split ~axis:k ~factor:16 ~outer:LL.Serial ~inner:LL.Serial in
  let stage source tile_loops hoisted =
    S.Stage
      {
        source;
        tile_loops;
        hoisted;
        shared = false;
        cooperative = None;
        swizzle = None;
        pad_stride = None;
        pipeline_depth = 1;
        tile_prec = Some Ir.Ops.single;
      }
  in
  [
    si;
    sk;
    S.Swap { outer = j; inner = ko };
    S.Swap { outer = ii; inner = ko };
    S.Swap { outer = io; inner = ko };
    stage b [ ki; j ] hoisted;
    stage a [ ii; ki ] false;
  ]

let () =
  let a, b, out = make () in
  let refused = ref false in
  let comp = Train.forward out in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        let site = Option.value_exn (Autotune.detect_matmul opt.LL.llc) in
        p "the frozen master has host-init-backed constant data"
          (S.hoistable_constant b.Tensor.value);
        p "the virtual half twin lowers to a rounded master read"
          (Option.is_some site.Autotune.m_b_round);
        (try ignore (S.apply (schedule ~hoisted:true a.Tensor.value b.Tensor.value opt) opt)
         with Invalid_argument why ->
           refused := String.is_substring why ~substring:"no load nest";
           Stdio.eprintf "hoist refusal: %s\n" why);
        [ S.apply (schedule ~hoisted:false a.Tensor.value b.Tensor.value opt) opt ])
      (Context.cpu ()) comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx out.Tensor.value in
  p_exists "frozen rounded inference executes with nonzero output" (Array.to_list got) ~f:(fun x ->
      Float.(x > 0.));
  let want =
    Array.init (n * n) ~f:(fun flat ->
        let i = flat / n and j = flat % n in
        List.sum
          (module Float)
          (List.range 0 n)
          ~f:(fun k -> input_cell [| i; k |] *. weight_cell [| k; j |]))
  in
  p_all2 "runtime packing computes the rounded host oracle" got want ~f:Float.equal;
  let arms =
    List.map [ false; true ] ~f:(fun host_rounded ->
        let _, _, tuned = make ~host_rounded () in
        let reports = ref [] in
        let tc, tr =
          Train.tune_placements ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:""
            ~report:(fun r -> reports := r :: !reports)
            (Context.cpu ()) tuned (Train.forward tuned) Ir.Indexing.Empty
        in
        let tc = Context.run tc tr in
        p_all2 "tuned frozen inference matches the packed reference" got
          (Context.get_values tc tuned.Tensor.value)
          ~f:Float.equal;
        List.iter !reports ~f:(fun r ->
            Stdio.eprintf
              "host_rounded=%b tuned candidates=%d winner=%s, shared-box search ms=%f (not a \
               performance claim)\n"
              host_rounded r.Autotune.mma_candidates r.best_label r.best_ms);
        (tc, tr, tuned))
  in
  let _, readings = Bench_retime.retime ~calls:8 (List.map arms ~f:(fun (c, r, _) -> (c, r))) in
  List.iter2_exn
    (List.zip_exn [ "twin"; "host" ] arms)
    readings
    ~f:(fun (label, (c, r, out)) readings ->
      let c = Context.run c r in
      p_all2 "retimed inference still computes the rounded reference" got
        (Context.get_values c out.Tensor.value)
        ~f:Float.equal;
      let spread = Bench_retime.spread_of readings in
      Stdio.eprintf "frozen-pack n=%d arm=%s %s (not part of the golden)\n%!" n label
        (Bench_retime.spread_fields ~name:"ms" spread);
      Context.release c);
  (match readings with
  | [ twin; host ] ->
      Stdio.eprintf "frozen-pack n=%d %s (not part of the golden)\n%!" n
        (Bench_retime.ratio_fields ~name:"host_over_twin"
           (Bench_retime.paired_ratio ~num:host ~den:twin));
      p "both arms admit enough timing calls to state a spread"
        ((Bench_retime.spread_of twin).admitted >= Bench_retime.min_readings
        && (Bench_retime.spread_of host).admitted >= Bench_retime.min_readings)
  | _ -> assert false);
  Context.release ctx;
  p "base refuses rounded hoisting" !refused
