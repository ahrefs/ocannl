(* gh-ocannl-1457, item 1: a sum's accumulator starts at +0.0, so a sum whose every term is a zero
   returns +0.0 on every schedule -- a recorded decision, pinned here.

   IEEE round-to-nearest gives [x + y = -0] for two zeros only when both are [-0], so a sum of zeros
   is [+0] as soon as one term is. Starting at [+0] makes that true of every schedule at once:
   zero-fringe padding and tensor-core K-padding add [(+0) * x] terms, a split reduction's partials
   start at [+0], and vector chains fold into the zeroed target. Starting at [-0] (the additive
   identity, which a serial fold from the first element agrees with) would return [-0] only on
   schedules that add no [+0] term -- a padded seed and the serial loop would then disagree in the
   sign, and no padding value fixes that ([(-0) * x = +0] for [x < 0]). The decision and its site
   inventory: docs/agent-notes/backend-precision-and-simd.md, ahrefs/ocannl#1457.

   What is pinned, on the bits ([Float.equal (-0.) 0.] is [true]): - a clearing reduction and a
   clearing matmul whose terms are all [-0] return [+0], on every backend's default schedule and, on
   cc, under every packed sketch seed of an awkward-extent matmul (zero-fringe padded tiles); - the
   control: the same products accumulated with [=+] into a target preloaded with [-0] return [-0],
   so the [+0] is the start value's, not a backend normalizing the sign; and the host's serial fold
   from the first element returns [-0], the reference the decision departs from.

   HIP's default [-ffast-math] implies [-fno-signed-zeros], so the sign of an exact zero is the
   compiler's call there (backend-precision-and-simd.md, "Fast math and device finiteness"): the
   sign claims open on HIP only under [hip_fast_math=false]. Operands are signed zeros and eighths,
   so contraction cannot move a value. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let on_cpu = Ir.Backend_intf.renders_on_host_pool (Context.backend_facts_of_name backend_name)
let accum_prec = Ll_test.backend_accum_prec

(* Where the compiler keeps the sign of a zero: everywhere but HIP's fast-math umbrella. *)
let signs_kept = not (String.equal backend_name "hip" && Ir.Cuda_like_config.hip_fast_math ())
let sign_leg = Verdict.leg ~when_:signs_kept ~on:"hip under its fast-math umbrella" ()
let cpu_leg = Verdict.leg ~when_:(on_cpu && signs_kept) ~on:backend_name ()
let is_pos_zero x = Float.equal x 0. && not (Float.ieee_negative x)
let is_neg_zero x = Float.equal x 0. && Float.ieee_negative x

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let run ?(transform = Fn.id) ?ctx ~name comp =
  let ctx = match ctx with Some ctx -> ctx | None -> Context.auto () in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun o -> [ transform o ])
      ctx (named name comp) Ir.Indexing.Empty
  in
  Context.run ctx routine

let () =
  p "the host's serial fold from the first element of all-negative-zero terms is -0"
    (is_neg_zero (List.reduce_exn (List.init 5 ~f:(fun _ -> -0.)) ~f:( +. )))

(* A clearing reduction over all-[-0] terms. *)
let () =
  let rows = 3 and cols = 37 in
  let x =
    TDSL.ndarray
      (Array.create ~len:(rows * cols) (-0.))
      ~label:[ "ssz_x" ] ~output_dims:[ rows; cols ] ()
  in
  let%op s = x ++ "ij => i" in
  let ctx = run ~name:"ssz_reduce" (Train.forward s) in
  in_leg ~leg:sign_leg "a clearing sum of all-negative-zero terms returns +0" (fun label ->
      p_all ~min:rows label (Array.to_list (Context.get_values ctx s.Tensor.value)) ~f:is_pos_zero)

(* Matmuls whose row 0 products are all [-0]: row 0 of A is [-0], B strictly positive; the other
   rows are ordinary eighths, so the results are not all zeros. Awkward extents, so the packed seeds
   pad every axis. *)
let m_ext, n_ext, k_ext = (33, 40, 37)

let a_vals =
  Array.init (m_ext * k_ext) ~f:(fun c ->
      if c < k_ext then -0.
      else Ll_test.cycle_flat ~dims:[| m_ext; k_ext |] ~modulus:7 ~offset:(-1.) ~stride:0.5 c)

let b_vals =
  Array.init (k_ext * n_ext)
    ~f:(Ll_test.cycle_flat ~dims:[| k_ext; n_ext |] ~modulus:13 ~offset:0.125 ~stride:0.25)

let operands name =
  ( TDSL.ndarray a_vals ~label:[ name ^ "a" ] ~input_dims:[ k_ext ] ~output_dims:[ m_ext ] (),
    TDSL.ndarray b_vals ~label:[ name ^ "b" ] ~input_dims:[ n_ext ] ~output_dims:[ k_ext ] () )

let row0 (v : float array) = Array.to_list (Array.sub v ~pos:0 ~len:n_ext)

let nonvacuous (v : float array) =
  Array.exists (Array.sub v ~pos:n_ext ~len:(Array.length v - n_ext)) ~f:(fun x -> Float.(x <> 0.))

let () =
  let seeds = ref [] in
  let ma, mb = operands "ssz_m0" in
  let%op c = ma * mb in
  let ctx =
    run ~name:"ssz_matmul"
      ~transform:(fun opt ->
        if on_cpu then
          seeds :=
            List.filter
              (Autotune.sketch_seed_params ~is_gpu:false ~is_cpu:true
                 ~limits:
                   {
                     Ir.Backend_intf.no_hardware_limits with
                     simd_vector = { Ir.Vector_file.bytes = 32; registers = 16 };
                   }
                 opt)
              ~f:(fun p ->
                p.Autotune.sk_mma && (not p.Autotune.sk_gpu) && (not p.Autotune.sk_epilogue)
                && Option.is_none p.Autotune.sk_tile);
        opt)
      (Train.forward c)
  in
  let got = Context.get_values ctx c.Tensor.value in
  in_leg ~leg:sign_leg "a clearing matmul whose row-0 products are all -0 returns +0 there"
    (fun label -> p_all ~min:n_ext label (row0 got) ~f:is_pos_zero);
  p "the clearing matmul's other rows are not all zeros" (nonvacuous got);
  (* The control: the same products, accumulated into a target preloaded with [-0]. *)
  let ma, mb = operands "ssz_m1" in
  let d =
    TDSL.ndarray
      (Array.create ~len:(m_ext * n_ext) (-0.))
      ~label:[ "ssz_d" ] ~input_dims:[ n_ext ] ~output_dims:[ m_ext ] ()
  in
  let ctx = run ~name:"ssz_accumulate" [%cd d =+ ma * mb ~logic:"@"] in
  in_leg ~leg:sign_leg
    "the same products accumulated into a -0 target return -0 (the +0 is the start value's)"
    (fun label ->
      p_all ~min:n_ext label (row0 (Context.get_values ctx d.Tensor.value)) ~f:is_neg_zero);
  (* Every packed seed: zero-fringe padded tiles add [(+0) * b] terms, which a [+0] start makes
     harmless to the sign. *)
  let runs =
    List.mapi (List.take !seeds 3) ~f:(fun i (p : Autotune.sketch_params) ->
        let ma, mb = operands (Printf.sprintf "ssz_s%d" i) in
        let%op c = ma * mb in
        let ctx =
          run ~name:(Printf.sprintf "ssz_seed_%d" i)
            ~transform:(fun opt -> Sched.apply (Autotune.sketch_schedule ~accum_prec ~p opt) opt)
            (Train.forward c)
        in
        Context.get_values ctx c.Tensor.value)
  in
  in_leg ~leg:cpu_leg "every packed seed's clearing matmul returns +0 on the all -0 row"
    (fun label ->
      p_all ~min:2 label runs ~f:(fun v ->
          Array.length v = m_ext * n_ext && List.for_all (row0 v) ~f:is_pos_zero))
