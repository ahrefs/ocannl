(* gh-ocannl-2326: [Ir.Backend_intf.with_select_blend], the seam [bench_cuda_select] times a whole
   training step through, reaches both readers of a backend's select blend policy: the lowering's
   inline fan-in guard and the renderer. A step timed under it is then placed as it renders, which
   an override of the renderer alone would not give: the guard would charge one arm of a select the
   renderer blends.

   Pinned here, on the running backend:

   - The facts: inside the extent [Context.backend_facts] reports the stated fact, both ways, and
   the backend's own record afterwards. - [y = where (m1 = m2) (a1 + a2 + a3) (b1 + ... + b5)] is
   charged 7 as a short-circuiting select and 10 as a blend against the fan-in cap of 8. Stated
   short-circuiting, [y] stays virtual and the routine renders no blend; stated blending, [y] is
   materialized and its select blended. The two sides of one program, so the override, not the
   program, moved both. Values against a reference. - The realistic composition, ReLU into a 2x2 max
   pool, backward: blended and short-circuiting, the same gradient, bitwise, with the blend emitted
   only where stated.

   Emission claims run in a leg open where the renderer has a blend (cc, CUDA, HIP, Metal under
   [Blend_over_locals] with [`Stmt_expr], gh-ocannl-2400), while the placement and value claims run
   on every backend. *)

(* cc-fp-contract: exact-operands -- sums of small integers, and a pool gradient selecting dyadic
   weights *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module Tn = Ir.Tnode
module BI = Ir.Backend_intf
module Generated = Test_utils.Generated

let () = Utils.settings.output_debug_files_in_build_directory <- true
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let blend_marker = Ir.C_syntax.blend_local_a ^ " ="
let () = Generated.init ~backend_name
let arm_name blend = if blend then "blend" else "short"

(* Runs [f] with the running backend's selects stated as [blend], as its lowering and renderer read
   them. *)
let under ~blend f =
  let ctx = Context.auto () in
  BI.with_select_blend ~facts:(Context.backend_facts ctx) ~blend (fun () -> f ctx)

(* Open where the renderer has a blend to emit: derived from the backend's
   [C_syntax_config.select_blend_syntax] via {!Ir.Backend_intf.codegen_capabilities.select_blend}
   under {!Ir.Backend_intf.with_select_blend} [~blend:true] (gh-ocannl-2386). *)
let emission_leg =
  let has_blend =
    under ~blend:true (fun ctx ->
        Option.is_some ((Context.codegen_capabilities ctx).BI.select_blend Ir.Ops.single))
  in
  Verdict.leg ~when_:has_blend ~on:backend_name ()

let () =
  let ctx = Context.auto () in
  let own = Context.backend_facts ctx in
  p_all "inside the extent, the backend's facts state the blend asked for, either way"
    [ false; true ] ~f:(fun blend ->
      BI.with_select_blend ~facts:own ~blend (fun () ->
          BI.equal_select_blend_policy (Context.backend_facts ctx).BI.select_blend_policy
            (if blend then Blend_speculable else No_blend)));
  p "after the extent, the backend's own facts again"
    (BI.equal_backend_facts (Context.backend_facts ctx) own)

let emission_claim what ~name ~blended =
  let claim =
    Printf.sprintf "%s: the routine %s the blend" what
      (if blended then "emits" else "does not emit")
  in
  Verdict.in_leg ~leg:emission_leg claim (fun claim ->
      p claim
        (Bool.equal blended
           (String.is_substring
              (Generated.main_logic (Generated.read name))
              ~substring:blend_marker)))

(* === The fan-in guard and the renderer read the stated fact === *)

let dim = 4
let weight k i = Float.of_int ((100 * k) + (10 * i) + 1)
let then_ks = [ 1; 2; 3 ]
let else_ks = [ 4; 5; 6; 7; 8 ]

let expected_loss =
  let cellv ks i = List.fold ks ~init:0. ~f:(fun acc k -> acc +. weight k i) in
  List.init dim ~f:(fun i -> if i % 2 = i / 2 % 2 then cellv then_ks i else cellv else_ks i)
  |> List.fold ~init:0. ~f:( +. )

let run_program ~name ~compile =
  Tensor.unsafe_reinitialize ();
  let init l k =
    NTDSL.init ~l ~prec:Ir.Ops.single ~o:[ dim ] ~f:(fun idcs -> weight k idcs.(0)) ()
  in
  let pattern l f = NTDSL.init ~l ~prec:Ir.Ops.single ~o:[ dim ] ~f:(fun idcs -> f idcs.(0)) () in
  let m1 = pattern "sbo_m1" (fun i -> Float.of_int (i % 2)) in
  let m2 = pattern "sbo_m2" (fun i -> Float.of_int (i / 2 % 2)) in
  let sum ks =
    List.map ks ~f:(fun k -> init (Printf.sprintf "sbo_w%d" k) k) |> List.reduce_exn ~f:TDSL.O.( + )
  in
  let y = TDSL.O.where (TDSL.O.( = ) m1 m2) (sum then_ks) (sum else_ks) in
  let%op loss = y ++ "i=>0" in
  Train.set_materialized loss.Tensor.value;
  Tn.set_observable loss.Tensor.value;
  let ctx, routine =
    compile (fun ctx -> Context.compile ~name ctx (Train.forward loss) Ir.Indexing.Empty)
  in
  let ctx = Context.run ctx routine in
  let plc = Context.placements ctx in
  ( Tn.Placements.known_non_virtual plc y.Tensor.value,
    Tn.Placements.known_virtual plc y.Tensor.value,
    Context.get_values ctx loss.Tensor.value )

let program ~blend =
  let detail = arm_name blend in
  let name = "sbo_eq_" ^ detail in
  let what = "the equality select stated " ^ detail in
  let materialized, virtual_, lv = run_program ~name ~compile:(under ~blend) in
  p
    (Printf.sprintf "%s: the producer is %s by the fan-in cap" what
       (if blend then "materialized" else "kept virtual"))
    (Bool.equal blend materialized && Bool.equal (not blend) virtual_);
  p
    (what ^ ": the loss matches the reference (exact: small integers)")
    (Array.length lv = 1 && Float.equal lv.(0) expected_loss);
  emission_claim what ~name ~blended:blend

let () =
  program ~blend:false;
  program ~blend:true;
  let ctx = Context.auto () in
  let load_free_leg =
    Verdict.leg ~when_:(String.equal (Context.backend_name ctx) "hip") ~on:backend_name ()
  in
  Verdict.in_leg ~leg:load_free_leg "HIP defaults to load-free blending" (fun claim ->
      p claim
        (BI.equal_select_blend_policy (Context.backend_facts ctx).BI.select_blend_policy
           Blend_load_free));
  Verdict.in_leg ~leg:load_free_leg
    "HIP's default policy leaves a two-loaded-arm select within the cap" (fun claim ->
      let name = "sbo_eq_load_free" in
      let materialized, virtual_, lv = run_program ~name ~compile:(fun f -> f ctx) in
      p claim
        ((not materialized) && virtual_
        && Array.length lv = 1
        && Float.equal lv.(0) expected_loss
        && not
             (String.is_substring
                (Generated.main_logic (Generated.read name))
                ~substring:blend_marker)));
  let serial_loop_leg =
    Verdict.leg ~when_:(String.equal (Context.backend_name ctx) "metal") ~on:backend_name ()
  in
  Verdict.in_leg ~leg:serial_loop_leg "Metal defaults to serial-loop blending" (fun claim ->
      p claim
        (BI.equal_select_blend_policy (Context.backend_facts ctx).BI.select_blend_policy
           Blend_serial_loop));
  Verdict.in_leg ~leg:serial_loop_leg
    "Metal's default policy leaves a two-loaded-arm select within the cap and blends in a serial \
     loop" (fun claim ->
      let name = "sbo_eq_serial_loop" in
      let materialized, virtual_, lv = run_program ~name ~compile:(fun f -> f ctx) in
      p claim
        ((not materialized) && virtual_
        && Array.length lv = 1
        && Float.equal lv.(0) expected_loss
        && String.is_substring (Generated.main_logic (Generated.read name)) ~substring:blend_marker
        ))

(* === The realistic composition: ReLU into a 2x2 max pool, backward === *)

let grad_of t = (Option.value_exn ~here:[%here] t.Tensor.diff).Tensor.grad

let pool ~blend =
  Tensor.unsafe_reinitialize ();
  let x =
    Operation.init ~l:"sbo_px" ~prec:Ir.Ops.single ~b:[] ~o:[ 4; 4 ]
      ~f:(Ll_test.weighted ~weights:[| 3; 5 |] ~modulus:7 ~offset:(-3.) ~stride:1.)
      ~grad_spec:Tensor.Require_grad ()
  in
  let w =
    Operation.init ~l:"sbo_pw" ~prec:Ir.Ops.single ~b:[] ~o:[ 2; 2 ]
      ~f:(fun idcs -> 0.25 *. Float.of_int (1 + idcs.(0) + (2 * idcs.(1))))
      ~grad_spec:Tensor.Prohibit_grad ()
  in
  let%op y = Nn_blocks.max_pool2d () (relu x) in
  let%op loss = (y *. w) ++ "... | ... => 0" in
  Train.set_materialized (grad_of x);
  (* The pool's gradient a buffer argument, so the max-pool gate's arm is a load. *)
  Train.set_materialized (grad_of y);
  let name = "sbo_pool_" ^ arm_name blend in
  let ctx, routine =
    under ~blend (fun ctx -> Context.compile ~name ctx (Train.grad_update loss) Ir.Indexing.Empty)
  in
  let ctx = Context.run ctx routine in
  emission_claim ("pool backward stated " ^ arm_name blend) ~name ~blended:blend;
  Context.get_values ctx (grad_of x)

let () =
  let short = pool ~blend:false in
  let blended = pool ~blend:true in
  let same_bits a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b) in
  p_all2 "pool backward: the blended gradient equals the short-circuiting one, bitwise" blended
    short ~f:same_bits;
  p "pool backward: the gradient is discriminating (zero and nonzero cells)"
    (Array.exists short ~f:(fun v -> Float.(v = 0.))
    && Array.exists short ~f:(fun v -> Float.(v > 0.)))
