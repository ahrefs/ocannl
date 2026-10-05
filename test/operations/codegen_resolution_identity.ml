(* gh-ocannl-1117: the schedule-cache identity tracks what a numerics mode RESOLVES to per backend.

   The numerics tag fingerprints the configured MODE, which is not the same fact as what the mode
   means on a backend: gh-ocannl-1051 made HIP's [Bf16_auto] resolve wide with the mode unchanged,
   and what kept winners tuned under the old narrow resolution from replaying was a [/bf16-acc-wide]
   component added to HIP's own codegen tag by hand. The codegen tag now tabulates the backend's
   resolution functions themselves ([Backend_intf.codegen_capabilities_fingerprint] over
   [Context.codegen_capabilities]), so the component is derived and that hand-added one is gone.

   gh-ocannl-1153 extends the derivation to the one numerics decision those two functions do not
   carry: which tensor-unit arm a mode selects. CUDA's tf32 gate in its wmma combination table puts
   f32 storage on a tf32 arm, while [accum_prec] says f32 accumulates at f32 with the gate open or
   shut; a code change to that gate, under an unchanged mode, left the identity where it was. The
   capability record now carries the backend's arm table ([mma_arm], derived from the table its mma
   hooks dispatch on), and the fingerprint tabulates it over every storage triple and scope.

   The controls, all printed as backend-uniform booleans:

   - On synthetic capability records (no backend involved): a resolution that changes — the
   accumulator's, the compute precision's, or the arm table's (an arm appearing, another arm, its
   accumulator, its floor) — moves the codegen identity; an extensionally equal one, a different
   closure computing the same table, does not. Neutralizing the arm row of the fingerprint fails the
   four arm claims on every backend. - On the backend this runs on, over every fp16 x bf16 x
   narrow-compute x tf32 policy: two policies get the same codegen identity exactly when the backend
   resolves them the same. The resolution here is the test's own tabulation of the capability
   record's functions, not the fingerprint, so the equivalence can fail: on CUDA a fingerprint
   without the arm row gives the tf32 pair one identity while the arm table tells them apart. Both
   sides are populated on every backend (each has a pair it resolves apart — [Fp16_wide] widens f16
   everywhere — and a distinct pair it resolves alike), so neither half is vacuous. On HIP this is
   where [Bf16_auto] against [Bf16_narrow] moves the identity with no hand-named component left to
   do it. - Every arm a uniform storage triple resolves to accumulates at the backend's [accum_prec]
   of that storage: gh-ocannl-663's width uniformity, read off the arm table. The population is the
   policy x scope x storage grid, so a backend without arms (cc) passes on rows with no arm to
   disagree, not on an empty collection.

   The per-policy resolution rows go to stderr, tagged, because they are the backend's own table and
   differ between backends. *)

open Base
open Ocannl.Operation.DSL_modules
module BI = Ir.Backend_intf
module SC = Ir.Schedule_cache
module Numerics = Ir.Numerics
open Verdict.Claims

let widen_bf16 = function Ir.Ops.Bfloat16_prec _ -> Ir.Ops.single | prec -> prec
let precs = Ir.Ops.all_precs
let scopes = BI.all_of_mma_emission_scope

(* A table with one arm, for uniform f32 storage, in every scope. *)
let uniform_f32 arm ~a_prec ~b_prec ~d_prec ~scope:_ =
  match (a_prec, b_prec, d_prec) with
  | Ir.Ops.Single_prec _, Ir.Ops.Single_prec _, Ir.Ops.Single_prec _ -> Some arm
  | _ -> None

let tf32_arm = { BI.arm_name = "wmma-tf32"; arm_accumulator = Ir.Ops.single; arm_floor = Some 80 }

let arm_string { BI.arm_name; arm_accumulator; arm_floor } =
  Printf.sprintf "%s/acc=%s/floor=%s" arm_name
    (Ir.Ops.prec_string arm_accumulator)
    (Option.value_map arm_floor ~default:"none" ~f:Int.to_string)

(* The oracle: the capability record's functions tabulated by the test itself, compared structurally
   — deliberately not [BI.codegen_capabilities_fingerprint], whose rendering is the thing under
   test. *)
let resolution (c : BI.codegen_capabilities) =
  ( (c.supports_f64, c.asynchronous_staging_copy),
    (List.map precs ~f:c.compute_prec, List.map precs ~f:c.accum_prec),
    List.concat_map scopes ~f:(fun scope ->
        List.concat_map precs ~f:(fun a_prec ->
            List.concat_map precs ~f:(fun b_prec ->
                List.map precs ~f:(fun d_prec -> c.mma_arm ~a_prec ~b_prec ~d_prec ~scope)))) )

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
  let with_arm arm = { base with BI.mma_arm = uniform_f32 arm } in
  p "an arm the policy enables moves the codegen identity"
    (not (String.equal (tag base) (tag (with_arm tf32_arm))));
  p "another arm for the same storage moves the codegen identity"
    (not
       (String.equal (tag (with_arm tf32_arm)) (tag (with_arm { tf32_arm with arm_name = "wmma" }))));
  p "a changed arm accumulator moves the codegen identity"
    (not
       (String.equal
          (tag (with_arm tf32_arm))
          (tag (with_arm { tf32_arm with arm_accumulator = Ir.Ops.half }))));
  p "a changed arm floor moves the codegen identity"
    (not
       (String.equal
          (tag (with_arm tf32_arm))
          (tag (with_arm { tf32_arm with arm_floor = Some 89 }))));
  p "an extensionally equal arm table leaves the codegen identity alone"
    (String.equal
       (tag (with_arm tf32_arm))
       (tag
          {
            base with
            BI.mma_arm =
              (fun ~a_prec ~b_prec ~d_prec ~scope:_ ->
                let f32 = Ir.Ops.equal_prec Ir.Ops.single in
                Option.some_if
                  (f32 a_prec && f32 b_prec && f32 d_prec)
                  {
                    BI.arm_name = "wmma-tf32";
                    arm_accumulator = Ir.Ops.single;
                    arm_floor = Some 80;
                  });
          }));

  (* --- The live backend, over the policy grid --- *)
  let ctx = Context.auto () in
  Stdio.eprintf "backend: %s (not part of the golden)\n%!" (Context.backend_name ctx);
  let saved = Numerics.get () in
  let policies =
    List.concat_map [ Numerics.Fp16_auto; Fp16_narrow; Fp16_wide ] ~f:(fun fp16_arithmetic ->
        List.concat_map [ Numerics.Bf16_auto; Bf16_narrow; Bf16_wide ] ~f:(fun bf16_arithmetic ->
            List.concat_map [ true; false ] ~f:(fun narrow_compute_f32 ->
                List.map [ false; true ] ~f:(fun tf32_matmuls ->
                    { Numerics.fp16_arithmetic; bf16_arithmetic; narrow_compute_f32; tf32_matmuls }))))
  in
  let rows =
    List.map policies ~f:(fun policy ->
        Numerics.set_policy policy;
        let capabilities = Context.codegen_capabilities ctx in
        let identity = SC.codegen_tag ~limits:(Context.hardware_limits ctx) ~capabilities () in
        let uniform =
          List.concat_map scopes ~f:(fun scope ->
              List.map [ Ir.Ops.single; Ir.Ops.half; Ir.Ops.bfloat16 ] ~f:(fun prec ->
                  ( scope,
                    prec,
                    capabilities.BI.mma_arm ~a_prec:prec ~b_prec:prec ~d_prec:prec ~scope,
                    capabilities.BI.accum_prec prec )))
        in
        Stdio.eprintf "resolution (not part of the golden): %s ->%s;%s\n"
          (Numerics.fingerprint policy)
          (String.concat
             (List.map [ Ir.Ops.half; Ir.Ops.bfloat16; Ir.Ops.fp8 ] ~f:(fun prec ->
                  Printf.sprintf " %s:compute=%s,accum=%s" (Ir.Ops.prec_string prec)
                    (Ir.Ops.prec_string (capabilities.BI.compute_prec prec))
                    (Ir.Ops.prec_string (capabilities.BI.accum_prec prec)))))
          (match
             List.filter_map uniform ~f:(fun (scope, prec, arm, _) ->
                 Option.map arm ~f:(fun arm ->
                     Printf.sprintf " %s:%s=%s"
                       (Sexp.to_string (BI.sexp_of_mma_emission_scope scope))
                       (Ir.Ops.prec_string prec) (arm_string arm)))
           with
          | [] -> " no uniform-storage arms"
          | arms -> String.concat arms);
        (policy, (resolution capabilities, uniform), identity))
  in
  Numerics.set_policy saved;
  let pairs =
    List.concat_mapi rows ~f:(fun i a ->
        List.filteri rows ~f:(fun j _ -> j > i) |> List.map ~f:(fun b -> (a, b)))
  in
  let same_resolution ((_, (r1, _), _), (_, (r2, _), _)) = Poly.equal r1 r2 in
  let same_identity ((_, _, t1), (_, _, t2)) = String.equal t1 t2 in
  p_all "across policies, the codegen identity moves exactly when the backend's resolution does"
    pairs ~f:(fun pair -> Bool.equal (same_resolution pair) (same_identity pair));
  p_exists "some policy pair resolves apart on this backend" pairs ~f:(fun pair ->
      not (same_resolution pair));
  p_exists "some distinct policy pair resolves alike on this backend" pairs ~f:same_resolution;
  p_all "every uniform-storage arm accumulates at the backend's accum_prec"
    (List.concat_map rows ~f:(fun (_, (_, uniform), _) -> uniform))
    ~f:(fun (_, _, arm, accum) ->
      match arm with None -> true | Some arm -> Ir.Ops.equal_prec arm.BI.arm_accumulator accum);
  let first_policy, _, first_identity = List.hd_exn rows in
  Numerics.set_policy first_policy;
  let again =
    SC.codegen_tag ~limits:(Context.hardware_limits ctx)
      ~capabilities:(Context.codegen_capabilities ctx)
      ()
  in
  Numerics.set_policy saved;
  p "the codegen identity is stable within one policy" (String.equal first_identity again)
