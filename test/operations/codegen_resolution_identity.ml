(* gh-ocannl-1117: the schedule-cache identity tracks what a numerics mode RESOLVES to per backend.

   The numerics tag fingerprints the configured MODE, which is not the same fact as what the mode
   means on a backend: gh-ocannl-1051 made HIP's [Bf16_auto] resolve wide with the mode unchanged,
   and what kept winners tuned under the old narrow resolution from replaying was a [/bf16-acc-wide]
   component added to HIP's own codegen tag by hand. The codegen tag now tabulates the backend's
   resolution functions themselves ([Backend_intf.codegen_capabilities_fingerprint] over
   [Context.codegen_capabilities]), so the component is derived and that hand-added one is gone.

   The controls, all printed as backend-uniform booleans:

   - On synthetic capability records (no backend involved): a resolution that changes — the
   accumulator's or the compute precision's — moves the codegen identity; an extensionally equal
   one, a different closure computing the same table, does not. - On the backend this runs on, over
   every fp16 x bf16 x narrow-compute policy: two policies get the same codegen identity exactly
   when the backend resolves them the same. Both sides of that equivalence are populated on every
   backend (each has a pair it resolves apart — [Fp16_wide] widens f16 everywhere — and a distinct
   pair it resolves alike), so neither half is vacuous. On HIP this is where [Bf16_auto] against
   [Bf16_narrow] moves the identity with no hand-named component left to do it.

   The per-policy resolution rows go to stderr, tagged, because they are the backend's own table and
   differ between backends. *)

open Base
open Ocannl.Operation.DSL_modules
module BI = Ir.Backend_intf
module SC = Ir.Schedule_cache
module Numerics = Ir.Numerics
open Verdict.Claims

let widen_bf16 = function Ir.Ops.Bfloat16_prec _ -> Ir.Ops.single | prec -> prec

let () =
  (* --- Synthetic records: the derivation itself, independent of any backend --- *)
  let tag capabilities = SC.codegen_tag ~limits:BI.no_hardware_limits ~capabilities () in
  let base = BI.no_codegen_capabilities in
  p "a changed accumulator resolution moves the codegen identity"
    (not (String.equal (tag base) (tag { base with BI.accum_prec = widen_bf16 })));
  p "a changed compute resolution moves the codegen identity"
    (not (String.equal (tag base) (tag { base with BI.compute_prec = widen_bf16 })));
  p "an extensionally equal resolution leaves the codegen identity alone"
    (String.equal
       (tag { base with BI.accum_prec = widen_bf16 })
       (tag
          {
            base with
            BI.accum_prec =
              (function
              | Ir.Ops.Void_prec -> Ir.Ops.Void_prec
              | Ir.Ops.Bfloat16_prec _ -> Ir.Ops.single
              | prec -> prec);
          }));

  (* --- The live backend, over the policy grid --- *)
  let ctx = Context.auto () in
  Stdio.eprintf "backend: %s (not part of the golden)\n%!" (Context.backend_name ctx);
  let saved = Numerics.get () in
  let policies =
    List.concat_map [ Numerics.Fp16_auto; Fp16_narrow; Fp16_wide ] ~f:(fun fp16_arithmetic ->
        List.concat_map [ Numerics.Bf16_auto; Bf16_narrow; Bf16_wide ] ~f:(fun bf16_arithmetic ->
            List.map [ true; false ] ~f:(fun narrow_compute_f32 ->
                { saved with Numerics.fp16_arithmetic; bf16_arithmetic; narrow_compute_f32 })))
  in
  let rows =
    List.map policies ~f:(fun policy ->
        Numerics.set_policy policy;
        let capabilities = Context.codegen_capabilities ctx in
        let resolution = BI.codegen_capabilities_fingerprint capabilities in
        let identity = SC.codegen_tag ~limits:(Context.hardware_limits ctx) ~capabilities () in
        Stdio.eprintf "resolution (not part of the golden): %s ->%s\n" (Numerics.fingerprint policy)
          (String.concat
             (List.map [ Ir.Ops.half; Ir.Ops.bfloat16; Ir.Ops.fp8 ] ~f:(fun prec ->
                  Printf.sprintf " %s:compute=%s,accum=%s" (Ir.Ops.prec_string prec)
                    (Ir.Ops.prec_string (capabilities.BI.compute_prec prec))
                    (Ir.Ops.prec_string (capabilities.BI.accum_prec prec)))));
        (policy, resolution, identity))
  in
  Numerics.set_policy saved;
  let pairs =
    List.concat_mapi rows ~f:(fun i a ->
        List.filteri rows ~f:(fun j _ -> j > i) |> List.map ~f:(fun b -> (a, b)))
  in
  let same_resolution ((_, r1, _), (_, r2, _)) = String.equal r1 r2 in
  let same_identity ((_, _, t1), (_, _, t2)) = String.equal t1 t2 in
  p_all "across policies, the codegen identity moves exactly when the backend's resolution does"
    pairs ~f:(fun pair -> Bool.equal (same_resolution pair) (same_identity pair));
  p_exists "some policy pair resolves apart on this backend" pairs ~f:(fun pair ->
      not (same_resolution pair));
  p_exists "some distinct policy pair resolves alike on this backend" pairs ~f:same_resolution;
  let first_policy, _, first_identity = List.hd_exn rows in
  Numerics.set_policy first_policy;
  let again =
    SC.codegen_tag ~limits:(Context.hardware_limits ctx)
      ~capabilities:(Context.codegen_capabilities ctx)
      ()
  in
  Numerics.set_policy saved;
  p "the codegen identity is stable within one policy" (String.equal first_identity again)
