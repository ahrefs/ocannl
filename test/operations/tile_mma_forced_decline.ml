(* gh-ocannl-1308: the test-only forced Tile_mma decline ([Utils.force_tile_mma_decline], set from
   OCANNL_TOOL_TEST_FORCE_TILE_MMA_DECLINE) is the negative control of every census claim, so it has
   to decline the way a real decline does -- and keep compiling while it does.

   One f32 matmul, scheduled two ways, and an f16->f32 staged matmul, compiled under both regimes in
   one process (the override, not the variable, so this test reads the same whatever the ambient
   environment says):

   - the plain tensorized f32 schedule, on every backend: the C backends' register tiling, Metal's
   simdgroup intrinsics, CUDA's wmma; - on a GPU, the staged f32 schedule, whose split-k accumulator
   the backend's fragment hook holds in registers across the whole reduction (a persistent-fragment
   scope) where uniform f32/tf32 is advertised (Metal, CUDA with tf32_matmuls=true); - on a GPU, the
   staged f16->f32 schedule (gh-ocannl-2402): every MMA-capable GPU (HIP RDNA3+ with rocWMMA, CUDA
   sm_70+, Metal Apple7+) advertises (Mma_f16, Mma_f16, Mma_f32) in Mma_fragment_scope regardless of
   the numerics policy, so HIP exercises the persistent-fragment forced-decline hook rather than
   skipping it. This is the leg the hook exists for: forcing the decline only per statement, inside
   a scope the hook already rendered, emits scalar code over an intrinsic fragment with no local
   array -- a kernel Metal and hiprtc refuse.

   Unforced, each renders tensorized (the staged ones with a fragment scope where advertised);
   forced, each renders every statement as the lane-0 scalar fallback with no fragment scope, and
   the values stay bitwise the serial twin's (quarters times small integers: every product and every
   sum over k = 32 gives one exact f32 value in any summation order; the f16->f32 operands are
   strictly positive so gfx11 WMMA cannot leave a cancellation residue, gh-ocannl-1420). The forced
   regime also moves the schedule cache's codegen identity, so a forced routine is never stored or
   replayed under the normal key. *)

(* cc-fp-contract: exact-operands -- quarters times small integers, every sum exact in f32 *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
module BI = Ir.Backend_intf

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let on_gpu = Ir.Backend_intf.binds_hardware_grid (Context.backend_facts_of_name backend_name)
let n = 32
let bm = 16
let simd_width = 32

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let with_forced forced f =
  let saved = !Utils.force_tile_mma_decline in
  Utils.force_tile_mma_decline := Some forced;
  Exn.protect ~f ~finally:(fun () -> Utils.force_tile_mma_decline := saved)

(* The two nests of the lowered matmul: its zeroing and its [i, j, k] accumulation. *)
let matmul_nest (opt : LL.optimized) =
  match List.find_exn (Ll_test.nest_paths opt.LL.llc) ~f:(fun p -> List.length p = 3) with
  | [ i; j; k ] -> (i, j, k)
  | _ -> assert false

let zeroing ~out =
  let ez, zsyms = Sched.expand_zero ~tn:out in
  let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
  let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
  [ ez; sp_zi; Sched.Retype { axis = zj; ty = LL.Workgroup } ]

let plain_schedule ~out (opt : LL.optimized) : Sched.schedule =
  let i, j, k = matmul_nest opt in
  let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
  let tz, _lane = Sched.tensorize ~i:i_i ~j ~k ~simd_width () in
  zeroing ~out @ [ sp_i; tz ]

(* Staged operand tiles and a serial outer-k loop: the shape whose accumulator becomes a persistent
   fragment. *)
let staged_schedule ~out ~src_a ~src_b (opt : LL.optimized) : Sched.schedule =
  let i, j, k = matmul_nest opt in
  let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
  let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bm ~outer:LL.Serial ~inner:LL.Serial in
  let tz, _lane = Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width () in
  let stage source tile_loops =
    Sched.Stage
      {
        source;
        tile_loops;
        shared = true;
        cooperative = Some simd_width;
        hoisted = false;
        swizzle = None;
        pad_stride = None;
        pipeline_depth = 1;
        tile_prec = None;
      }
  in
  zeroing ~out
  @ [
      sp_i;
      sp_k;
      Sched.Swap { outer = j; inner = k_o };
      Sched.Swap { outer = i_i; inner = k_o };
      stage src_a [ i_i; k_i ];
      stage src_b [ k_i; j ];
      tz;
    ]

let () =
  let mav =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:13 ~offset:0. ~stride:0.25)
  in
  let mbv =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:17 ~offset:(-8.) ~stride:1.)
  in
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let run_matmul ~a ~b ?out_prec ~name ?schedule () =
    let%op c = a * b in
    Option.iter out_prec ~f:(Tn.update_prec c.Tensor.value);
    let transform opt =
      match schedule with
      | None -> opt
      | Some sched -> Sched.apply (sched ~out:c.Tensor.value opt) opt
    in
    let ctx, routine =
      Context.compile
        ~lowered_transform:(fun o -> [ transform o ])
        (Context.auto ())
        (named name (Train.forward c))
        Ir.Indexing.Empty
    in
    (* The target starts off every product these operands sum to, so a cell the fallback leaves
       unwritten fails parity (gh-ocannl-1625). *)
    let tn = c.Tensor.value in
    let numel = Array.fold (Tn.dims_without_padding tn) ~init:1 ~f:( * ) in
    let ctx = Context.set_values ctx tn (Array.init numel ~f:(fun i -> Float.of_int (1000 + i))) in
    (Context.get_values (Context.run ctx routine) tn, routine.Context.mma)
  in
  let run ~name ?schedule () = run_matmul ~a:ma ~b:mb ~name ?schedule () in
  let want, _ = with_forced false (fun () -> run ~name:"fd_serial" ()) in
  p "the serial twin computes a nonzero product" (Array.exists want ~f:(fun x -> Float.(x <> 0.)));
  let legs ~what ~schedule =
    let got_unforced, mma_unforced =
      with_forced false (fun () -> run ~name:(what ^ "_unforced") ~schedule ())
    in
    let got_forced, mma_forced =
      with_forced true (fun () -> run ~name:(what ^ "_forced") ~schedule ())
    in
    Stdio.eprintf "%s (not part of the golden): unforced %s, %d fragment scopes; forced %s, %d\n%!"
      what
      (Ir.C_syntax.mma_summary_string mma_unforced)
      mma_unforced.Ir.C_syntax.fragment_scopes
      (Ir.C_syntax.mma_summary_string mma_forced)
      mma_forced.Ir.C_syntax.fragment_scopes;
    (got_unforced, mma_unforced, got_forced, mma_forced)
  in
  let plain_unforced, plain_mma_unforced, plain_forced, plain_mma_forced =
    legs ~what:"fd_plain" ~schedule:plain_schedule
  in
  let single = Ir.Ops.single in
  p "unforced, the plain tensorized schedule renders tensorized where uniform f32 is advertised"
    (Ll_test.census_agrees_with_capability ~backend_name ~what:"fd_plain unforced" ~a_prec:single
       ~b_prec:single ~d_prec:single
       (Ll_test.renderings plain_mma_unforced));
  p "forced, every Tile_mma of the plain schedule renders the scalar fallback"
    (Ll_test.all_declined ~what:"fd_plain forced" (Ll_test.renderings plain_mma_forced));
  p_all2 "unforced, the plain tensorized matmul matches the serial twin bitwise" plain_unforced want
    ~f:Float.equal;
  p_all2 "forced, the plain tensorized matmul's fallback matches the serial twin bitwise"
    plain_forced want ~f:Float.equal;
  let gpu = Verdict.leg ~when_:on_gpu ~on:backend_name () in
  (* The descriptor's judgment that this GPU holds a uniform-f32 accumulator in a persistent
     fragment: Metal's genuine f32 tile, or CUDA's tf32 one where the policy enables it. *)
  let fragment_scope =
    Verdict.sub_leg gpu ~on:backend_name
      ~when_:
        (lazy
          (Ll_test.advertised ~backend_name ~a_prec:single ~b_prec:single ~d_prec:single
          && List.exists [ BI.Mma_f32; BI.Mma_tf32 ] ~f:(fun f ->
              BI.advertises_mma_format_in_scope
                (Context.hardware_limits (Context.auto ()))
                ~policy:(Ir.Numerics.get ()) ~scope:BI.Mma_fragment_scope ~a:f ~b:f ~d:BI.Mma_f32)))
  in
  let staged = staged_schedule ~src_a:ma.Tensor.value ~src_b:mb.Tensor.value in
  let staged_unforced =
    lazy (with_forced false (fun () -> run ~name:"fd_staged_unforced" ~schedule:staged ()))
  in
  (* The forced compile is the one the hook can break: a decline taken only per statement, inside
     the scope the fragment hook rendered, is a kernel Metal does not compile. Its exception is a
     false claim here rather than the end of the run. *)
  let staged_forced =
    lazy
      (match with_forced true (fun () -> run ~name:"fd_staged_forced" ~schedule:staged ()) with
      | result -> Some result
      | exception exn ->
          Stdio.eprintf "fd_staged forced: %s\n%!" (Exn.to_string exn);
          None)
  in
  let census what (mma : Ir.C_syntax.mma_summary) =
    Stdio.eprintf "%s (not part of the golden): %s, %d fragment scopes\n%!" what
      (Ir.C_syntax.mma_summary_string mma)
      mma.Ir.C_syntax.fragment_scopes
  in
  in_leg ~leg:fragment_scope
    "unforced, the staged schedule renders tensorized in a persistent-fragment scope" (fun label ->
      let _, mma = Lazy.force staged_unforced in
      census "fd_staged unforced" mma;
      p label
        (Ll_test.all_tensorized ~what:"fd_staged unforced" (Ll_test.renderings mma)
        && mma.Ir.C_syntax.fragment_scopes > 0));
  in_leg ~leg:gpu "forced, the staged schedule compiles and runs" (fun label ->
      p label (Option.is_some (Lazy.force staged_forced)));
  in_leg ~leg:gpu
    "forced, the staged schedule declines at the fragment scope: every Tile_mma falls back, no \
     scope renders" (fun label ->
      p label
        (match Lazy.force staged_forced with
        | None -> false
        | Some (_, mma) ->
            census "fd_staged forced" mma;
            Ll_test.all_declined ~what:"fd_staged forced" (Ll_test.renderings mma)
            && mma.Ir.C_syntax.fragment_scopes = 0));
  in_leg ~leg:gpu "both staged tensorized matmuls match the serial twin bitwise" (fun label ->
      let got_unforced, _ = Lazy.force staged_unforced in
      p label
        (Array.equal Float.equal got_unforced want
        &&
        match Lazy.force staged_forced with
        | None -> false
        | Some (got_forced, _) -> Array.equal Float.equal got_forced want));
  (* gh-ocannl-2402: f16 -> f32 staged leg. Every MMA-capable GPU (HIP RDNA3+ with rocWMMA, CUDA
     sm_70+, Metal Apple7+) advertises (Mma_f16, Mma_f16, Mma_f32) in Mma_fragment_scope regardless
     of the numerics policy, so gating on [limits.mma] keeps the descriptor claim non-vacuous and
     ensures HIP exercises its persistent-fragment forced-decline path. Strictly positive quarters
     times small positive integers keep every 32-term sum in [8, 728]: exact in f32, free of gfx11
     WMMA cancellation residue (gh-ocannl-1420), and below the [1000, 2023] prefill. *)
  let half = Ir.Ops.half in
  let ha_fn = Ll_test.cycle ~dims:[| n; n |] ~modulus:13 ~offset:1. ~stride:0.25 in
  let hb_fn = Ll_test.cycle ~dims:[| n; n |] ~modulus:7 ~offset:1. ~stride:1. in
  let ha = NTDSL.init ~l:"ha" ~prec:half ~i:[ n ] ~o:[ n ] ~f:ha_fn () in
  let hb = NTDSL.init ~l:"hb" ~prec:half ~i:[ n ] ~o:[ n ] ~f:hb_fn () in
  let want_f16, _ =
    with_forced false (fun () -> run_matmul ~a:ha ~b:hb ~out_prec:single ~name:"fd_f16_serial" ())
  in
  p_all "the f16->f32 serial twin is positive and below the prefill" (Array.to_list want_f16)
    ~f:(fun x -> Float.(x > 0. && x < 1000.));
  let gpu_mma =
    Verdict.sub_leg gpu ~on:backend_name
      ~when_:(lazy (Option.is_some (Context.hardware_limits (Context.auto ())).BI.mma))
  in
  let f16_staged = staged_schedule ~src_a:ha.Tensor.value ~src_b:hb.Tensor.value in
  let f16_staged_unforced =
    lazy
      (with_forced false (fun () ->
           run_matmul ~a:ha ~b:hb ~out_prec:single ~name:"fd_f16_staged_unforced"
             ~schedule:f16_staged ()))
  in
  let f16_staged_forced =
    lazy
      (match
         with_forced true (fun () ->
             run_matmul ~a:ha ~b:hb ~out_prec:single ~name:"fd_f16_staged_forced"
               ~schedule:f16_staged ())
       with
      | result -> Some result
      | exception exn ->
          Stdio.eprintf "fd_f16_staged forced: %s\n%!" (Exn.to_string exn);
          None)
  in
  in_leg ~leg:gpu_mma "an MMA-capable GPU advertises f16->f32 in a persistent-fragment scope"
    (fun label ->
      p label
        (BI.advertises_mma_format_in_scope
           (Context.hardware_limits (Context.auto ()))
           ~policy:(Ir.Numerics.get ()) ~scope:BI.Mma_fragment_scope ~a:BI.Mma_f16 ~b:BI.Mma_f16
           ~d:BI.Mma_f32));
  in_leg ~leg:gpu
    "unforced, the f16->f32 staged schedule renders tensorized in a persistent-fragment scope \
     where advertised, else the scalar fallback" (fun label ->
      let _, mma = Lazy.force f16_staged_unforced in
      census "fd_f16_staged unforced" mma;
      let expect_fragment =
        Ll_test.advertised ~backend_name ~a_prec:half ~b_prec:half ~d_prec:single
        && BI.advertises_mma_format_in_scope
             (Context.hardware_limits (Context.auto ()))
             ~policy:(Ir.Numerics.get ()) ~scope:BI.Mma_fragment_scope ~a:BI.Mma_f16 ~b:BI.Mma_f16
             ~d:BI.Mma_f32
      in
      p label
        (if expect_fragment then
           Ll_test.all_tensorized ~what:"fd_f16_staged unforced" (Ll_test.renderings mma)
           && mma.Ir.C_syntax.fragment_scopes > 0
         else
           Ll_test.all_declined ~what:"fd_f16_staged unforced" (Ll_test.renderings mma)
           && mma.Ir.C_syntax.fragment_scopes = 0));
  in_leg ~leg:gpu "forced, the f16->f32 staged schedule compiles and runs" (fun label ->
      p label (Option.is_some (Lazy.force f16_staged_forced)));
  in_leg ~leg:gpu
    "forced, the f16->f32 staged schedule declines at the fragment scope: every Tile_mma falls \
     back, no scope renders" (fun label ->
      p label
        (match Lazy.force f16_staged_forced with
        | None -> false
        | Some (_, mma) ->
            census "fd_f16_staged forced" mma;
            Ll_test.all_declined ~what:"fd_f16_staged forced" (Ll_test.renderings mma)
            && mma.Ir.C_syntax.fragment_scopes = 0));
  in_leg ~leg:gpu "unforced, the f16->f32 staged tensorized matmul matches the serial twin bitwise"
    (fun label ->
      let got_unforced, _ = Lazy.force f16_staged_unforced in
      p_all2 label got_unforced want_f16 ~f:Float.equal);
  in_leg ~leg:gpu
    "forced, the f16->f32 staged tensorized matmul's fallback matches the serial twin bitwise"
    (fun label ->
      match Lazy.force f16_staged_forced with
      | None -> p label false
      | Some (got_forced, _) -> p_all2 label got_forced want_f16 ~f:Float.equal);
  (* The schedule cache's codegen identity: the forced regime has its own, so neither regime's
     crowns replay into the other. *)
  let ctx = Context.auto () in
  let tag () =
    Ir.Schedule_cache.codegen_tag ~facts:(Context.backend_facts ctx)
      ~limits:(Context.hardware_limits ctx)
      ~capabilities:(Context.codegen_capabilities ctx)
      ()
  in
  p "the forced decline moves the schedule cache's codegen identity"
    (not (String.equal (with_forced false tag) (with_forced true tag)))
