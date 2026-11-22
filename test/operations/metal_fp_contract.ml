(* gh-ocannl-2406: Metal's floating-point contraction gate ([metal_fp_contract=off] -> [#pragma STDC
   FP_CONTRACT OFF] in the kernel header).

   MTLCompileOptions exposes no contraction switch, so under the default [metal_fp_contract=auto]
   Metal's MathMode.Safe permits contracting a single-expression [(a * b) + c] into a fused
   multiply-add. Under [metal_fp_contract=off] (pinned by [profile=reproducible]), the kernel header
   emits [#pragma STDC FP_CONTRACT OFF]:

   - [compiler_contraction_off] flips from [false] to [true]; - the emitted [.metal] header omits
   the pragma under [auto] and carries it under [off]; - compiling the same-named probe kernel twice
   under [auto] hits the process-local kernel code cache, and flipping to [off] under the same name
   misses the cache and compiles afresh; - under [auto], [(a * b) + c] returns a legal rounding
   ([2^-24] fused or [+0.0] unfused; the observed values and chain difference are reported to stderr
   as in [tools/metal_contract_probe.swift]), while under [off] [(a * b) + c] returns [+0.0] (two
   roundings, bitwise) and an explicit [Ir.Ops.FMA (a, b, c)] still returns [2^-24] (fused); - an
   FMA-sensitive 37-term multiply-add chain over 16 rows computes the exact host two-rounding f32
   chain bitwise under [off]. *)

(* cc-fp-contract: not-cc -- every device claim sits in the Metal leg; cc skips them all *)

open Base
open Verdict.Claims
open Ll_test
module LL = Ir.Low_level
module Ops = Ir.Ops

let () = Utils.state_config ~key:"fp_contract" ~data:"false"
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let on_metal = String.equal backend_name "metal"
let metal_leg = Verdict.leg ~when_:on_metal ~on:backend_name ()
let p2 n = Float.ldexp 1. n
let r32 x = Int32.float_of_bits (Int32.bits_of_float x)
let bits x = Int64.bits_of_float x
let bitwise_scalar a b = Int64.equal (bits a) (bits b)
let bitwise_array a b = Array.equal bitwise_scalar a b

(* Single-expression probe inputs, exact in f32: [a * b = 1 + 2^-11 + 2^-24] is a tie at f32 and
   rounds to even ([1 + 2^-11]), so [(a * b) + c] gives [+0.0] with two roundings and [2^-24]
   fused. *)
let near_one = 1. +. p2 (-12)
let probe_inputs = [| near_one; near_one; -.(1. +. p2 (-11)) |]
let fused_expected = p2 (-24)

(* 16 independent 37-term non-dyadic multiply-add chains, cycled over prime moduli so every product
   and partial sum rounds at f32. *)
let chain_rows = 16
let chain_k = 37

let chain_a =
  Array.init (chain_rows * chain_k)
    ~f:(cycle_flat ~dims:[| chain_rows; chain_k |] ~modulus:97 ~offset:(-1.3) ~stride:0.1)

let chain_b =
  Array.init (chain_rows * chain_k)
    ~f:(cycle_flat ~dims:[| chain_rows; chain_k |] ~modulus:89 ~offset:(-0.7) ~stride:(1. /. 3.))

let host_chain ~step =
  Array.init chain_rows ~f:(fun i ->
      let acc = ref 0. in
      for k = 0 to chain_k - 1 do
        let idx = (i * chain_k) + k in
        acc := step !acc (r32 chain_a.(idx)) (r32 chain_b.(idx))
      done;
      !acc)

let chain_unfused = host_chain ~step:(fun acc a b -> r32 (acc +. r32 (a *. b)))
let chain_fused = host_chain ~step:(fun acc a b -> r32 (Stdlib.Float.fma a b acc))

type metal_results = {
  cap_auto : bool;
  cap_off : bool;
  src_auto : string;
  src_off : string;
  cost_auto_hit : Ir.Compile_cost.t;
  cost_off_miss : Ir.Compile_cost.t;
  probe_auto : float;
  probe_off : float;
  explicit_fma_off : float;
  chain_off : float array;
}

let results =
  lazy
    (Utils.settings.output_debug_files_in_build_directory <- true;
     Test_utils.Generated.init ~backend_name;
     let run_probe ~first_id ~name ~explicit_fma =
       let mk = node_factory ~first_id ~dims:[| 3 |] () in
       let inp = mk (name ^ "_in") in
       let omk = node_factory ~first_id:(first_id + 1) ~dims:[| 1 |] () in
       let out = omk (name ^ "_out") in
       materialize inp;
       materialize out;
       let a = get inp [| fixed 0 |] and b = get inp [| fixed 1 |] and c = get inp [| fixed 2 |] in
       let rhs =
         if explicit_fma then LL.Ternop (Ops.FMA, (a, Ops.single), (b, Ops.single), (c, Ops.single))
         else add (mul a b) c
       in
       let o = optimize ~materialized:[ inp; out ] ~name (set out [| fixed 0 |] rhs) in
       let got, cost =
         Ir.Compile_cost.with_census (fun () ->
             List.hd_exn
               (execute ~name o ~seed:[ (inp, probe_inputs); (out, blank 1) ] ~read:[ out ]))
       in
       (got.(0), Test_utils.Generated.read name, cost)
     in
     let run_chain ~first_id ~name =
       let mk_in = node_factory ~first_id ~dims:[| chain_rows; chain_k |] () in
       let ma = mk_in (name ^ "_a") in
       let mb = mk_in (name ^ "_b") in
       let mk_out = node_factory ~first_id:(first_id + 2) ~dims:[| chain_rows |] () in
       let out = mk_out (name ^ "_out") in
       materialize ma;
       materialize mb;
       materialize out;
       let i = sym () and k = sym () in
       let body =
         seq (zero out)
           (loop_n i chain_rows
              (loop_n k chain_k
                 (set out
                    [| iter i |]
                    (add
                       (get out [| iter i |])
                       (mul (get ma [| iter i; iter k |]) (get mb [| iter i; iter k |]))))))
       in
       let o = optimize ~materialized:[ ma; mb; out ] ~name body in
       List.hd_exn
         (execute ~name o
            ~seed:[ (ma, chain_a); (mb, chain_b); (out, blank chain_rows) ]
            ~read:[ out ])
     in
     (* Compile the same-named [mfc_probe] twice under [auto] (second compile must hit the
        process-local kernel code cache), then flip to [off] under the same name (must miss and
        recompile with the pragma in the prelude). *)
     let cap_auto, probe_auto, src_auto, cost_auto_hit, chain_auto =
       Utils.with_stated_config
         [ ("metal_fp_contract", "auto") ]
         (fun () ->
           let cap = backend_contraction_off () in
           let p_val, src, _cost_cold =
             run_probe ~first_id:240600 ~name:"mfc_probe" ~explicit_fma:false
           in
           Test_utils.Generated.arm "mfc_probe";
           let _p_val2, _src2, cost_hit =
             run_probe ~first_id:240605 ~name:"mfc_probe" ~explicit_fma:false
           in
           let c_val = run_chain ~first_id:240610 ~name:"mfc_chain_auto" in
           (cap, p_val, src, cost_hit, c_val))
     in
     let cap_off, probe_off, src_off, cost_off_miss, explicit_fma_off, chain_off =
       Utils.with_stated_config
         [ ("metal_fp_contract", "off") ]
         (fun () ->
           let cap = backend_contraction_off () in
           Test_utils.Generated.arm "mfc_probe";
           let p_val, src, cost_miss =
             run_probe ~first_id:240620 ~name:"mfc_probe" ~explicit_fma:false
           in
           let f_val, _, _ =
             run_probe ~first_id:240630 ~name:"mfc_explicit_fma_off" ~explicit_fma:true
           in
           let c_val = run_chain ~first_id:240640 ~name:"mfc_chain_off" in
           (cap, p_val, src, cost_miss, f_val, c_val))
     in
     let max_diff a b =
       Array.foldi a ~init:0. ~f:(fun i acc x -> Float.max acc (Float.abs (x -. b.(i))))
     in
     Stdio.eprintf
       "metal_fp_contract (not part of the golden): probe auto=%h off=%h explicit_fma_off=%h; \
        chain max|auto-unfused|=%h max|off-unfused|=%h; cache hit=(compiled=%d reused=%d) \
        flip_miss=(compiled=%d reused=%d)\n\
        %!"
       probe_auto probe_off explicit_fma_off
       (max_diff chain_auto chain_unfused)
       (max_diff chain_off chain_unfused)
       cost_auto_hit.kernels_compiled cost_auto_hit.kernels_reused cost_off_miss.kernels_compiled
       cost_off_miss.kernels_reused;
     {
       cap_auto;
       cap_off;
       src_auto;
       src_off;
       cost_auto_hit;
       cost_off_miss;
       probe_auto;
       probe_off;
       explicit_fma_off;
       chain_off;
     })

let () =
  p "the host emulation separates one rounding per step from two on the chain"
    (not (bitwise_array chain_unfused chain_fused));
  in_leg ~leg:metal_leg
    "compiler_contraction_off is false under metal_fp_contract=auto and true under off"
    (fun label ->
      let r = Lazy.force results in
      p label ((not r.cap_auto) && r.cap_off));
  in_leg ~leg:metal_leg
    "the generated Metal header omits #pragma STDC FP_CONTRACT OFF under auto and emits it under \
     off" (fun label ->
      let r = Lazy.force results in
      let pragma = Ir.Compiler_options.metal_fp_contract_pragma in
      p label
        ((not (String.is_substring r.src_auto ~substring:pragma))
        && String.is_substring r.src_off ~substring:pragma));
  in_leg ~leg:metal_leg
    "recompiling the same-named probe hits the kernel cache under auto and misses after flipping \
     to off" (fun label ->
      let r = Lazy.force results in
      p label
        (r.cost_auto_hit.kernels_reused = 1
        && r.cost_auto_hit.kernels_compiled = 0
        && r.cost_off_miss.kernels_reused = 0
        && r.cost_off_miss.kernels_compiled = 1));
  in_leg ~leg:metal_leg
    "under metal_fp_contract=auto, (a * b) + c computes a legal rounding (fused 2^-24 or serial \
     +0.0)" (fun label ->
      let r = Lazy.force results in
      p label (bitwise_scalar r.probe_auto fused_expected || bitwise_scalar r.probe_auto 0.));
  in_leg ~leg:metal_leg "under metal_fp_contract=off, (a * b) + c rounds twice to +0.0 bitwise"
    (fun label ->
      let r = Lazy.force results in
      p label (bitwise_scalar r.probe_off 0.));
  in_leg ~leg:metal_leg
    "under metal_fp_contract=off, explicit fma(a, b, c) still computes the fused 2^-24 bitwise"
    (fun label ->
      let r = Lazy.force results in
      p label (bitwise_scalar r.explicit_fma_off fused_expected));
  in_leg ~leg:metal_leg
    "under metal_fp_contract=off, the FMA-sensitive chain computes the two-rounding f32 chain \
     bitwise" (fun label ->
      let r = Lazy.force results in
      p label (bitwise_array r.chain_off chain_unfused))
