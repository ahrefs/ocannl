(* gh-ocannl-2344: diagnostic extra HIP compile options ([hip_extra_compile_options]).

   Appending whitespace-separated flags at the end of hiprtc's option vector lets compiler-vector
   A/B experiments differ only on the command line without patching [compiler_options.ml]. This test
   verifies on HIP hardware that: - [compiler_contraction_off] follows left-to-right overrides in
   [hip_extra_compile_options]; - HIP's [hardware_limits.codegen_tag] and
   [Schedule_cache.cache_key_components] key on the canonical parsed options (leaving empty and
   whitespace-only on the default key) and reflect [/mma-fallback-unfused] when extra options pin
   contraction off; - the process-local kernel code cache hits on identical or whitespace-equivalent
   option strings and misses when the parsed option vector changes; - hiprtc actually honors
   appended flags at execution: overriding [hip_fast_math=true]'s [-ffp-contract=on] with
   [-ffp-contract=off] makes [(a * b) + c] and a 37-term multiply-add chain match the host
   two-rounding f32 chain bitwise while preserving explicit [FMA]; - multi-token overrides
   ([-freciprocal-math -fapprox-func]) compile and execute cleanly; - an unrecognized option fails
   loudly with the appended flag reported in [hiprtc options:]. *)

(* cc-fp-contract: not-cc -- every device claim sits in the HIP leg; cc skips them all *)

open Base
open Verdict.Claims
open Ll_test
module LL = Ir.Low_level
module Ops = Ir.Ops
module SC = Ir.Schedule_cache

let () = Utils.state_config ~key:"fp_contract" ~data:"false"
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let on_hip = String.equal backend_name "hip"
let hip_leg = Verdict.leg ~when_:on_hip ~on:backend_name ()
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

type hip_results = {
  cap_default : bool;
  cap_contract_off : bool;
  cap_repro_contract_fast : bool;
  tag_default : string;
  tag_ws : string;
  tag_contract_off : string;
  tag_approx : string;
  tag_approx_ws : string;
  comp_default : (string * string) list;
  comp_ws : (string * string) list;
  comp_contract_off : (string * string) list;
  comp_approx : (string * string) list;
  comp_approx_ws : (string * string) list;
  cost_default_cold : Ir.Compile_cost.t;
  cost_default_hit : Ir.Compile_cost.t;
  cost_contract_off_miss : Ir.Compile_cost.t;
  cost_approx_miss : Ir.Compile_cost.t;
  cost_approx_ws_hit : Ir.Compile_cost.t;
  probe_default : float;
  probe_contract_off : float;
  probe_approx : float;
  explicit_fma_off : float;
  chain_contract_off : float array;
  invalid_error : string option;
}

let timing_identity =
  Some
    {
      Ir.Backend_intf.device_signature = "hip-extra-options-test";
      toolchain_signature = Some "hiprtc";
    }

let results =
  lazy
    (let build_probe ~first_id ~name ~explicit_fma =
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
       (inp, out, o)
     in
     let run_probe ~first_id ~name ~explicit_fma =
       let inp, out, o = build_probe ~first_id ~name ~explicit_fma in
       let got, cost =
         Ir.Compile_cost.with_census (fun () ->
             List.hd_exn
               (execute ~name o ~seed:[ (inp, probe_inputs); (out, blank 1) ] ~read:[ out ]))
       in
       (got.(0), cost)
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
     let ctx = Context.auto () in
     let _, _, sample_opt = build_probe ~first_id:234490 ~name:"heo_canon" ~explicit_fma:false in
     let canon = SC.canonicalize ~static_indices:[] sample_opt in
     let snapshot () =
       let limits = Context.hardware_limits ctx in
       let tag = Option.value limits.codegen_tag ~default:"" in
       let comps =
         Option.value_exn
           (SC.cache_key_components ~facts:(Context.backend_facts ctx) ~timing_identity ~limits
              ~capabilities:(Context.codegen_capabilities ctx)
              canon ~backend:backend_name)
       in
       (backend_contraction_off (), tag, comps)
     in
     let ( cap_default,
           tag_default,
           comp_default,
           tag_ws,
           comp_ws,
           cap_contract_off,
           tag_contract_off,
           comp_contract_off,
           cap_repro_contract_fast,
           tag_approx,
           comp_approx,
           tag_approx_ws,
           comp_approx_ws,
           cost_default_cold,
           cost_default_hit,
           cost_contract_off_miss,
           cost_approx_miss,
           cost_approx_ws_hit,
           probe_default,
           probe_contract_off,
           probe_approx,
           explicit_fma_off,
           chain_contract_off,
           invalid_error ) =
       Ir.Kernel_code_cache.with_cold_caches (fun () ->
           let ( cap_default,
                 tag_default,
                 comp_default,
                 probe_default,
                 cost_default_cold,
                 cost_default_hit ) =
             Utils.with_stated_config
               [ ("hip_fast_math", "true"); ("hip_extra_compile_options", "") ]
               (fun () ->
                 let cap, tag, comps = snapshot () in
                 let p1, c1 = run_probe ~first_id:234400 ~name:"heo_probe" ~explicit_fma:false in
                 let _p2, c2 = run_probe ~first_id:234405 ~name:"heo_probe" ~explicit_fma:false in
                 (cap, tag, comps, p1, c1, c2))
           in
           let _cap_ws, tag_ws, comp_ws =
             Utils.with_stated_config
               [ ("hip_fast_math", "true"); ("hip_extra_compile_options", "   \t  ") ]
               snapshot
           in
           let ( cap_contract_off,
                 tag_contract_off,
                 comp_contract_off,
                 probe_contract_off,
                 cost_contract_off_miss,
                 explicit_fma_off,
                 chain_contract_off ) =
             Utils.with_stated_config
               [ ("hip_fast_math", "true"); ("hip_extra_compile_options", "-ffp-contract=off") ]
               (fun () ->
                 let cap, tag, comps = snapshot () in
                 let p_val, c_miss =
                   run_probe ~first_id:234410 ~name:"heo_probe" ~explicit_fma:false
                 in
                 let f_val, _ =
                   run_probe ~first_id:234415 ~name:"heo_explicit_fma" ~explicit_fma:true
                 in
                 let chain = run_chain ~first_id:234420 ~name:"heo_chain_off" in
                 (cap, tag, comps, p_val, c_miss, f_val, chain))
           in
           let cap_repro_contract_fast, _, _ =
             Utils.with_stated_config
               [ ("hip_fast_math", "false"); ("hip_extra_compile_options", "-ffp-contract=fast") ]
               snapshot
           in
           let _cap_approx, tag_approx, comp_approx, probe_approx, cost_approx_miss =
             Utils.with_stated_config
               [
                 ("hip_fast_math", "true");
                 ("hip_extra_compile_options", "-freciprocal-math -fapprox-func");
               ]
               (fun () ->
                 let cap, tag, comps = snapshot () in
                 let p_val, c_miss =
                   run_probe ~first_id:234430 ~name:"heo_probe" ~explicit_fma:false
                 in
                 (cap, tag, comps, p_val, c_miss))
           in
           let _cap_approx_ws, tag_approx_ws, comp_approx_ws, cost_approx_ws_hit =
             Utils.with_stated_config
               [
                 ("hip_fast_math", "true");
                 ("hip_extra_compile_options", "  -freciprocal-math\t -fapprox-func  ");
               ]
               (fun () ->
                 let cap, tag, comps = snapshot () in
                 let _p_val, c_hit =
                   run_probe ~first_id:234435 ~name:"heo_probe" ~explicit_fma:false
                 in
                 (cap, tag, comps, c_hit))
           in
           let invalid_error =
             Utils.with_stated_config
               [
                 ("hip_fast_math", "true");
                 ("hip_extra_compile_options", "-mocannl-no-such-hiprtc-flag");
               ]
               (fun () ->
                 match run_probe ~first_id:234440 ~name:"heo_invalid_opt" ~explicit_fma:false with
                 | _ -> None
                 | exception exn -> Some (Exn.to_string exn))
           in
           ( cap_default,
             tag_default,
             comp_default,
             tag_ws,
             comp_ws,
             cap_contract_off,
             tag_contract_off,
             comp_contract_off,
             cap_repro_contract_fast,
             tag_approx,
             comp_approx,
             tag_approx_ws,
             comp_approx_ws,
             cost_default_cold,
             cost_default_hit,
             cost_contract_off_miss,
             cost_approx_miss,
             cost_approx_ws_hit,
             probe_default,
             probe_contract_off,
             probe_approx,
             explicit_fma_off,
             chain_contract_off,
             invalid_error ))
     in
     let max_diff a b =
       Array.foldi a ~init:0. ~f:(fun i acc x -> Float.max acc (Float.abs (x -. b.(i))))
     in
     Stdio.eprintf
       "hip_extra_compile_options (not part of the golden): probe default=%h contract_off=%h \
        approx=%h explicit_fma_off=%h; chain max|contract_off-unfused|=%h; tags default=%S \
        contract_off=%S approx=%S; cache cold=(%d/%d) hit=(%d/%d) off_miss=(%d/%d) \
        approx_miss=(%d/%d) approx_ws_hit=(%d/%d); invalid_error=%s\n\
        %!"
       probe_default probe_contract_off probe_approx explicit_fma_off
       (max_diff chain_contract_off chain_unfused)
       tag_default tag_contract_off tag_approx cost_default_cold.kernels_compiled
       cost_default_cold.kernels_reused cost_default_hit.kernels_compiled
       cost_default_hit.kernels_reused cost_contract_off_miss.kernels_compiled
       cost_contract_off_miss.kernels_reused cost_approx_miss.kernels_compiled
       cost_approx_miss.kernels_reused cost_approx_ws_hit.kernels_compiled
       cost_approx_ws_hit.kernels_reused
       (Option.value invalid_error ~default:"<none>");
     {
       cap_default;
       cap_contract_off;
       cap_repro_contract_fast;
       tag_default;
       tag_ws;
       tag_contract_off;
       tag_approx;
       tag_approx_ws;
       comp_default;
       comp_ws;
       comp_contract_off;
       comp_approx;
       comp_approx_ws;
       cost_default_cold;
       cost_default_hit;
       cost_contract_off_miss;
       cost_approx_miss;
       cost_approx_ws_hit;
       probe_default;
       probe_contract_off;
       probe_approx;
       explicit_fma_off;
       chain_contract_off;
       invalid_error;
     })

let equal_components =
  List.equal (fun (k1, v1) (k2, v2) -> String.equal k1 k2 && String.equal v1 v2)

let moved a b =
  List.filter_map (List.zip_exn a b) ~f:(fun ((name, x), (_, y)) ->
      Option.some_if (not (String.equal x y)) name)

let () =
  p "the host emulation separates one rounding per step from two on the chain"
    (not (bitwise_array chain_unfused chain_fused));
  in_leg ~leg:hip_leg
    "compiler_contraction_off tracks left-to-right contraction overrides in \
     hip_extra_compile_options" (fun label ->
      let r = Lazy.force results in
      p label ((not r.cap_default) && r.cap_contract_off && not r.cap_repro_contract_fast));
  in_leg ~leg:hip_leg
    "HIP codegen_tag omits /hiprtc-extra: when empty or whitespace-only, appends canonical \
     /hiprtc-extra: when set, and reflects /mma-fallback-unfused" (fun label ->
      let r = Lazy.force results in
      p label
        ((not (String.is_substring r.tag_default ~substring:"/hiprtc-extra:"))
        && String.equal r.tag_default r.tag_ws
        && String.is_suffix r.tag_contract_off
             ~suffix:"/mma-fallback-unfused/hiprtc-extra:-ffp-contract=off"
        && String.is_suffix r.tag_approx ~suffix:"/hiprtc-extra:-freciprocal-math -fapprox-func"
        && String.equal r.tag_approx r.tag_approx_ws));
  in_leg ~leg:hip_leg
    "schedule cache key moves only codegen across distinct extra options and ignores whitespace \
     normalization" (fun label ->
      let r = Lazy.force results in
      p label
        (equal_components r.comp_default r.comp_ws
        && List.equal String.equal (moved r.comp_default r.comp_contract_off) [ "codegen" ]
        && List.equal String.equal (moved r.comp_default r.comp_approx) [ "codegen" ]
        && List.equal String.equal (moved r.comp_contract_off r.comp_approx) [ "codegen" ]
        && equal_components r.comp_approx r.comp_approx_ws));
  in_leg ~leg:hip_leg
    "kernel code cache hits on matching or whitespace-equivalent extra options and misses on \
     changes" (fun label ->
      let r = Lazy.force results in
      p label
        (r.cost_default_cold.kernels_compiled = 1
        && r.cost_default_cold.kernels_reused = 0
        && r.cost_default_hit.kernels_compiled = 0
        && r.cost_default_hit.kernels_reused = 1
        && r.cost_contract_off_miss.kernels_compiled = 1
        && r.cost_contract_off_miss.kernels_reused = 0
        && r.cost_approx_miss.kernels_compiled = 1
        && r.cost_approx_miss.kernels_reused = 0
        && r.cost_approx_ws_hit.kernels_compiled = 0
        && r.cost_approx_ws_hit.kernels_reused = 1));
  in_leg ~leg:hip_leg
    "under hip_fast_math=true with hip_extra_compile_options=-ffp-contract=off, (a * b) + c rounds \
     twice to +0.0 bitwise while default computes a legal rounding" (fun label ->
      let r = Lazy.force results in
      p label
        ((bitwise_scalar r.probe_default fused_expected || bitwise_scalar r.probe_default 0.)
        && bitwise_scalar r.probe_contract_off 0.));
  in_leg ~leg:hip_leg
    "under hip_fast_math=true with hip_extra_compile_options=-ffp-contract=off, explicit fma(a, b, \
     c) still computes the fused 2^-24 bitwise" (fun label ->
      let r = Lazy.force results in
      p label (bitwise_scalar r.explicit_fma_off fused_expected));
  in_leg ~leg:hip_leg
    "under hip_fast_math=true with hip_extra_compile_options=-ffp-contract=off, the FMA-sensitive \
     chain computes the two-rounding f32 chain bitwise" (fun label ->
      let r = Lazy.force results in
      p label (bitwise_array r.chain_contract_off chain_unfused));
  in_leg ~leg:hip_leg
    "multi-token hip_extra_compile_options (-freciprocal-math -fapprox-func) compiles and executes \
     cleanly on HIP" (fun label ->
      let r = Lazy.force results in
      p label (bitwise_scalar r.probe_approx fused_expected || bitwise_scalar r.probe_approx 0.));
  in_leg ~leg:hip_leg
    "an unrecognized flag in hip_extra_compile_options fails loudly and reports the flag in hiprtc \
     options" (fun label ->
      let r = Lazy.force results in
      p label
        (Option.exists r.invalid_error ~f:(fun msg ->
             String.is_substring msg ~substring:"hiprtc options:"
             && String.is_substring msg ~substring:"-mocannl-no-such-hiprtc-flag")))
