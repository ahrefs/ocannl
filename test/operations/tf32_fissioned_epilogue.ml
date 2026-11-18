(* gh-ocannl-2347: the two-segment winner of autotune_device_memory's matmul + ReLU fixture. Pin the
   segmentation and transforms so the failing kernel does not depend on search timing. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Tn = Ir.Tnode

let m = 192
and n = 32
and k = 32

let simd_width = 32
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let prec = Ir.Ops.single

let a_values =
  Ll_test.operand ~b:[] ~o:[ m ] ~i:[ k ]
    (Ll_test.weighted ~weights:[| 3; 1 |] ~modulus:7 ~offset:(-3.) ~stride:1.)

let b_values =
  Ll_test.operand ~b:[] ~o:[ k ] ~i:[ n ]
    (Ll_test.weighted ~weights:[| 1; 2 |] ~modulus:5 ~offset:(-2.) ~stride:1.)

let make_graph () =
  let a = NTDSL.init ~l:"fl_fa" ~prec ~o:[ m ] ~i:[ k ] ~f:a_values () in
  let b = NTDSL.init ~l:"fl_fb" ~prec ~o:[ k ] ~i:[ n ] ~f:b_values () in
  let%op prod = a * b in
  let%op out = relu prod in
  (a.Tensor.value, b.Tensor.value, prod.Tensor.value, out)

let segmentation = [ (`Zeros, 1); (`Normal, 2) ]

let transform ?target ~pipeline_depth ~a ~b ~prod opt =
  let schedule i (segment : LL.optimized) =
    let ops =
      match i with
      | 0 ->
          let expand, axes = Sched.expand_zero ~tn:prod in
          let row, col = match axes with [ row; col ] -> (row, col) | _ -> assert false in
          [
            expand;
            Sched.Retype { axis = row; ty = LL.Grid };
            Sched.Retype { axis = col; ty = LL.Workgroup };
          ]
      | 1 ->
          let row, col, reduce =
            match Ll_test.nest_paths segment.llc with
            | [ [ row; col; reduce ]; [ _; _ ] ] -> (row, col, reduce)
            | _ -> failwith "expected the matmul nest followed by the ReLU nest"
          in
          let split axis factor outer = Sched.split ~axis ~factor ~outer ~inner:LL.Serial in
          let split_i, _, ii = split row 16 LL.Grid in
          let split_j, jo, ji = split col 32 LL.Grid in
          let split_k, ko, ki = split reduce 32 LL.Serial in
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
                pipeline_depth;
                tile_prec = None;
              }
          in
          let mma, _ = Sched.tensorize ~i:ii ~j:ji ~k:ki ~simd_width () in
          [
            split_i;
            split_j;
            split_k;
            Sched.Swap { outer = ii; inner = jo };
            Sched.Swap { outer = ji; inner = ko };
            Sched.Swap { outer = ii; inner = ko };
            stage a [ ii; ki ];
            stage b [ ki; ji ];
            mma;
            Sched.Fuse_epilogue { target = prod; shared = true };
          ]
      | _ -> assert false
    in
    Some ops
  in
  let _, segments = Ll_test.tuner_replay ?target ~plan:segmentation ~schedule opt in
  List.map segments ~f:(fun (_, _, _, post) -> post)

let () =
  Unix.putenv "OCANNL_BUFFER_ALIASING" "false";
  let a, b, prod, out = make_graph () in
  let opt = Ll_test.captured_lowering ~name:"tf32_ep_capture" (Train.forward out) in
  let segments = transform ~target:(Ll_test.synthetic_gpu ()) ~pipeline_depth:2 ~a ~b ~prod opt in
  p "the two scheduled kernels share the device accumulator"
    (match segments with
    | [ zero; fused ] ->
        LL.code_touches_tn prod zero.llc && LL.code_touches_tn prod fused.llc
        && (match Tn.Placements.get fused.optimize_ctx.placements prod with
          | Some (Tn.On_device, _) -> true
          | _ -> false)
        && not (Set.mem fused.workgroup_shared prod)
    | _ -> false)

let compile ?pipeline_depth name =
  let a, b, prod, out = make_graph () in
  let ctx = Context.auto () in
  let ctx, lowered_transform =
    match pipeline_depth with
    | None -> (Context.decide_materialized ctx [ prod ], fun opt -> [ opt ])
    | Some pipeline_depth -> (ctx, fun opt -> transform ~pipeline_depth ~a ~b ~prod opt)
  in
  let ctx, routine =
    Context.compile ~name ~lowered_transform ctx (Train.forward out) Ir.Indexing.Empty
  in
  Stdio.eprintf "%s: shipped %d segments, mma %s (not part of the golden)\n%!" name
    (List.length routine.Context.segments)
    (Ll_test.census_string (Ll_test.renderings routine.mma));
  (ctx, routine, prod, out.Tensor.value)

let () =
  let policy = Ir.Numerics.get () in
  let set_tf32 tf32_matmuls = Ir.Numerics.set_policy { policy with tf32_matmuls } in
  set_tf32 true;
  let cuda = String.equal backend_name "cuda" in
  let leg =
    Verdict.leg ~on:backend_name
      ~when_:(cuda && Ll_test.advertised ~backend_name ~a_prec:prec ~b_prec:prec ~d_prec:prec)
      ()
  in
  let cells dims read =
    List.init (Array.fold dims ~init:1 ~f:( * )) ~f:(fun i -> read (Ll_test.unflat ~dims i))
  in
  p_all "the fixture operands fit TF32's significand"
    (cells [| m; k |] a_values @ cells [| k; n |] b_values)
    ~f:(fun x ->
      let mantissa, _ = Float.frexp x in
      Float.is_integer (Float.ldexp mantissa 11));
  let products = Ll_test.matmul_products ~m ~n ~k a_values b_values in
  p_all "the fixture products have exact partial sums at f32" products
    ~f:(Ll_test.exact_partials ~any_sign:true ~sig_bits:(Ir.Ops.significand_bits prec));
  let reference =
    lazy
      (let ctx, routine, prod, out = compile "tf32_ep_reference" in
       let ran = Context.run ctx routine in
       let values = (Context.get_values ran prod, Context.get_values ran out) in
       Context.release ctx;
       values)
  in
  Verdict.in_leg ~leg "the unscheduled product matches the host reduction" (fun label ->
      p_all2 label
        (fst (Lazy.force reference))
        (Array.of_list (List.map products ~f:(List.sum (module Float) ~f:Fn.id)))
        ~f:Float.equal);
  Verdict.in_leg ~leg "the unscheduled reference has positive cells" (fun label ->
      p_exists label (Array.to_list (snd (Lazy.force reference))) ~f:(fun x -> Float.(x > 0.)));
  let check variant name ~pipeline_depth ~tf32 =
    let result =
      lazy
        (set_tf32 tf32;
         let ctx, routine, _, out = compile ~pipeline_depth name in
         let got = Context.get_values (Ll_test.run_linked (ctx, routine) ~seed:[]) out in
         Context.release ctx;
         (got, Ll_test.renderings routine.mma))
    in
    Verdict.in_leg ~leg (variant ^ " renders the requested MMA path") (fun label ->
        let _, census = Lazy.force result in
        p label
          (if tf32 then Ll_test.all_rendered ~what:name Ir.C_syntax.Mma_intrinsics census
           else Ll_test.all_declined ~what:name census));
    Verdict.in_leg ~leg (variant ^ " matches the unscheduled result") (fun label ->
        let got, _ = Lazy.force result in
        p_close ~min:(m * n) label got (snd (Lazy.force reference)) ~tol:(fun _ -> 0.))
  in
  check "the depth-one TF32 replay" "tf32_ep_pd1" ~pipeline_depth:1 ~tf32:true;
  check "the full-f32 replay" "tf32_ep_f32" ~pipeline_depth:2 ~tf32:false;
  check "the depth-two TF32 replay" "tf32_ep_pd2" ~pipeline_depth:2 ~tf32:true
