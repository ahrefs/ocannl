(* gh-ocannl-2078: a conv pack holds rounded reads at compute precision. All conv flavors stage both
   operands; unlike matmul there is no in-place flavor. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module LL = Ir.Low_level
module Sched = Ir.Schedule

let input_values =
  Ll_test.weighted ~weights:[| 1; 2; 1 |] ~modulus:7 ~offset:1.000244140625 ~stride:0.125

let kernel_values =
  Ll_test.weighted ~weights:[| 1; 3; 2 |] ~modulus:5 ~offset:1.000244140625 ~stride:0.125

let product ~materialized ~raw ~round_a ~round_b =
  let x = NTDSL.init ~l:"cvtx" ~prec:Ir.Ops.single ~b:[ 2 ] ~o:[ 6; 4 ] ~f:input_values () in
  let k = NTDSL.init ~l:"cvtk" ~prec:Ir.Ops.single ~i:[ 3; 4 ] ~o:[ 8 ] ~f:kernel_values () in
  let twin enabled t =
    if raw || not enabled then t
    else
      let c = Operation.cast ~grad_spec:Tensor.Prohibit_grad t () in
      Ir.Tnode.update_prec c.Tensor.value Ir.Ops.half;
      (if materialized then Train.set_materialized else Train.set_virtual) c.Tensor.value;
      c
  in
  let a = twin round_a x and b = twin round_b k in
  let%op y = a +* "...|oh<+kh, ic; |kh, ic -> oc => ...|oh, oc" b in
  Ir.Tnode.update_prec y.Tensor.value Ir.Ops.single;
  y

let compile ?(transform = Fn.id) name out =
  let ctx, routine =
    Context.compile ~name
      ~lowered_transform:(fun opt -> [ transform opt ])
      (Context.cpu ()) (Train.forward out) Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  (Context.get_values ctx out.Tensor.value, routine)

(* cc-fp-contract: exact-operands -- Ll_test.exact_partials derives the f32 reduction premise from
   the initialized operands after their required rounding. *)
let () =
  List.iter
    [ (true, false, "input"); (false, true, "kernel"); (true, true, "both") ]
    ~f:(fun (round_a, round_b, label) ->
      let rounded enabled init idcs =
        let x = init idcs in
        if enabled then Ir.Ops.half_to_single (Ir.Ops.single_to_half x) else x
      in
      (* [NTDSL.init] indexes batch, then output, then input axes: [cvtx] is [batch; h; ic], [cvtk]
         is [oc; kh; ic]. *)
      let products =
        Ll_test.reduction_cells ~out:[| 2; 4; 8 |] ~over:[| 3; 4 |] (fun o r ->
            rounded round_a input_values [| o.(0); o.(1) + r.(0); r.(1) |]
            *. rounded round_b kernel_values [| o.(2); r.(0); r.(1) |])
      in
      p_all
        (label ^ ": rounded products have exact partial sums at f32")
        products
        ~f:(Ll_test.exact_partials ~sig_bits:(Ir.Ops.significand_bits Ir.Ops.single));
      let product = product ~round_a ~round_b in
      let reference = ref None in
      let want, _ =
        compile ("cvt_ref_" ^ label)
          ~transform:(fun o ->
            reference := Some o;
            o)
          (product ~materialized:true ~raw:false)
      in
      (* The premise above holds for the products the conv forms only if they are these: their exact
         sums must be the executed reference's bits. *)
      p_all2
        (label ^ ": the materialized twin equals the premise's products summed on the host")
        want
        (Array.of_list (List.map products ~f:(List.sum (module Float) ~f:Fn.id)))
        ~f:Float.equal;
      let raw, _ = compile ("cvt_raw_" ^ label) (product ~materialized:false ~raw:true) in
      p
        (label ^ ": an unrounded read differs from the materialized twin")
        (not (Array.equal Float.equal want raw));
      let captured = ref None in
      let _, _ =
        compile ("cvt_capture_" ^ label)
          ~transform:(fun opt ->
            captured := Some opt;
            opt)
          (product ~materialized:false ~raw:false)
      in
      let opt = Option.value_exn !captured in
      let limits =
        {
          (Context.hardware_limits (Context.cpu ())) with
          Ir.Backend_intf.native_fp16_arithmetic = false;
        }
      in
      let seeds =
        Autotune.sketch_seed_params ~is_gpu:false ~is_cpu:true ~limits opt
        |> List.filter ~f:(fun q -> q.Autotune.sk_conv)
      in
      p_all (label ^ ": conv seeds request compute-precision packs") seeds ~f:(fun q ->
          Option.equal Ir.Ops.equal_prec q.Autotune.sk_pack_prec (Some Ir.Ops.single));
      let serial =
        List.filter seeds ~f:(fun q ->
            (not q.Autotune.sk_grid) && q.sk_bm = 0 && Option.is_none q.sk_tile)
      in
      p_exists (label ^ ": a serial packed seed executes") serial ~f:(fun _ -> true);
      let ap = if round_a then Ir.Backend_intf.Mma_f16 else Ir.Backend_intf.Mma_f32 in
      let bp = if round_b then Ir.Backend_intf.Mma_f16 else Ir.Backend_intf.Mma_f32 in
      let gpu_limits =
        {
          Ir.Backend_intf.no_hardware_limits with
          mma =
            Some
              {
                Ir.Backend_intf.minimal_mma_capability with
                mma_tile = (2, 4, 4);
                mma_format_tiles = [ ((ap, bp, Ir.Backend_intf.Mma_f32), (2, 4, 4)) ];
              };
        }
      in
      let gpu_seeds =
        Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits:gpu_limits opt
      in
      p_all (label ^ ": GPU seeds judge the rounded operand formats and stage both operands")
        gpu_seeds ~f:(fun q ->
          let sched = Autotune.sketch_schedule ~accum_prec:Fn.id ~p:q opt in
          q.Autotune.sk_conv
          && List.count sched ~f:(function Sched.Stage _ -> true | _ -> false) = 2);
      let backing_only =
        {
          gpu_limits with
          mma =
            Some
              {
                (Option.value_exn gpu_limits.Ir.Backend_intf.mma) with
                mma_format_tiles =
                  [
                    ( (Ir.Backend_intf.Mma_f32, Ir.Backend_intf.Mma_f32, Ir.Backend_intf.Mma_f32),
                      (2, 4, 4) );
                  ];
              };
        }
      in
      let refused =
        Option.value_exn
          (Autotune.conv_sketch_tree ~is_gpu:true ~is_cpu:false ~limits:backing_only opt)
      in
      p_empty
        (label ^ ": backing-storage formats alone are refuted at seeding")
        ~over:(Ir.Schedule_space.refutations refused)
        (Ir.Schedule_space.leaves refused);
      List.iter
        [ (false, limits); (true, gpu_limits) ]
        ~f:(fun (is_gpu, limits) ->
          let tree =
            Option.value_exn (Autotune.conv_sketch_tree ~is_gpu ~is_cpu:(not is_gpu) ~limits opt)
          in
          let floor = Autotune.sketch_path_traffic_floor ~limits opt in
          let reference_floor =
            Autotune.sketch_path_traffic_floor ~limits (Option.value_exn !reference)
          in
          p_all
            (label
            ^ (if is_gpu then ": GPU" else ": CPU")
            ^ " traffic prices the same tiles as a materialized twin")
            (Ir.Schedule_space.enumerate tree)
            ~f:(fun (path, _) -> floor path = reference_floor path));
      (* Recompile a fresh graph for each seed: node identity belongs to that lowering. *)
      List.iteri serial ~f:(fun i q ->
          let got, routine =
            compile
              (Printf.sprintf "cvt_pack_%s_%d" label i)
              ~transform:(fun o ->
                let sched = Autotune.sketch_schedule ~accum_prec:Fn.id ~p:q o in
                p_all
                  (label ^ ": stages use compute precision for rounded reads")
                  (List.filter_map sched ~f:(function
                    | Sched.Stage s -> Some (s.source, s.tile_prec)
                    | _ -> None))
                  ~f:(fun (source, tile_prec) ->
                    let expected =
                      Option.map (LL.rounded_read_prec o.LL.llc source) ~f:(fun _ -> Ir.Ops.single)
                    in
                    Option.equal Ir.Ops.equal_prec tile_prec expected);
                Sched.apply sched o)
              (product ~materialized:false ~raw:false)
          in
          let packed_operands = ref [] in
          List.iter routine.Context.segments ~f:(fun seg ->
              Ll_test.walk seg.LL.llc ~on_stmt:(function
                | LL.Tile_mma { a = a, _; b = b, _; _ } ->
                    packed_operands := a :: b :: !packed_operands
                | _ -> ()));
          p_all (label ^ ": the compiled kernel reads f32 packed operands") !packed_operands
            ~f:(fun tn -> Ir.Ops.equal_prec (Lazy.force tn.Ir.Tnode.storage_prec) Ir.Ops.single);
          p_all2
            (label ^ ": packed execution equals the materialized twin bitwise")
            got want ~f:Float.equal;
          p
            (label ^ ": the packed conv renders a register tile")
            (Ll_test.all_rendered ~what:label Ir.C_syntax.Mma_register_tiled
               (Ll_test.renderings routine.Context.mma))))
