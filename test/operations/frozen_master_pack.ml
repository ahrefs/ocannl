(* gh-ocannl-2079: link-time panels preserve a virtual half twin's f32 read rounding. The preceding
   investigation commit records the pre-change timing probe. Ordinary paired runs justified
   implementing this conversion; no timing assertion belongs in this test. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims
module Tn = Ir.Tnode
module LL = Ir.Low_level
module S = Ir.Schedule
module G = Test_utils.Generated

let backend = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let () = Utils.settings.output_debug_files_in_build_directory <- true
let () = G.init ~backend_name:backend

let named name (comp : Ir.Assignments.comp) =
  { comp with asgns = Ir.Assignments.Block_comment (name, comp.asgns) }

let half_twin master =
  let t = Operation.cast ~grad_spec:Tensor.Prohibit_grad master () in
  Tn.update_prec t.Tensor.value Ir.Ops.half;
  Train.set_virtual t.Tensor.value;
  t

let constant ~label ~dims values =
  let m = NTDSL.ndarray values ~label:[ label ] ~output_dims:dims ~top_down_prec:false () in
  Tn.update_prec m.Tensor.value Ir.Ops.single;
  Train.set_materialized m.Tensor.value;
  m

let bits_equal a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

let stage ~source ~tile_loops ~hoisted ~tile_prec =
  S.Stage
    {
      source;
      tile_loops;
      hoisted;
      tile_prec;
      shared = false;
      cooperative = None;
      swizzle = None;
      pad_stride = None;
      pipeline_depth = 1;
    }

let run ~name ?(schedule = fun _ -> []) ?(check = fun _ -> ()) out =
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        let opt = S.apply (schedule opt) opt in
        check opt;
        [ opt ])
      (Context.auto ())
      (named name (Train.forward out))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let values = Context.get_values ctx out.Tensor.value in
  Context.release ctx;
  values

(* Every finite positive half interval, its midpoint and the adjacent f32 values, both signs. This
   covers ties, carry into the next binade, and the entire subnormal interval. *)
let cases =
  let midpoints =
    List.concat_map (List.range 0 0x7bff) ~f:(fun h ->
        let lo = Ir.Ops.half_to_single h and hi = Ir.Ops.half_to_single (h + 1) in
        let mid = (lo +. hi) /. 2. in
        List.concat_map [ -1l; 0l; 1l ] ~f:(fun delta ->
            let f =
              Stdlib.Int32.float_of_bits (Stdlib.Int32.add (Stdlib.Int32.bits_of_float mid) delta)
            in
            [ f; -.f ]))
  in
  Array.of_list
    (midpoints
    @ [
        0.;
        -0.;
        65504.;
        65519.;
        65520.;
        65521.;
        -65519.;
        -65520.;
        -65521.;
        Float.infinity;
        Float.neg_infinity;
        Stdlib.Int32.float_of_bits 0x00000001l;
        Stdlib.Int32.float_of_bits 0x80000001l;
        Stdlib.Int32.float_of_bits 0x00800000l;
        Stdlib.Int32.float_of_bits 0x80800000l;
        Stdlib.Int32.float_of_bits 0x7f7fffffl;
        Stdlib.Int32.float_of_bits 0xff7fffffl;
      ])

let () =
  let make () =
    let m = constant ~label:"conversion_master" ~dims:[ Array.length cases ] cases in
    let t = half_twin m in
    let out = Operation.cast ~grad_spec:Tensor.Prohibit_grad t () in
    Tn.update_prec out.Tensor.value Ir.Ops.single;
    Train.set_materialized out.Tensor.value;
    (m, out)
  in
  let _, direct_out = make () in
  let direct = run ~name:"frozen_device_conversion" direct_out in
  G.assert_emits ~routine:"frozen_device_conversion"
    ~contains:(G.node_ident ~facts:(Context.backend_facts_of_name backend) "conversion_master" ^ "[")
    "the conversion oracle reads the materialized f32 master on device";
  List.iter [ None; Some Ir.Ops.single ] ~f:(fun tile_prec ->
      let name = if Option.is_none tile_prec then "frozen_half_panel" else "frozen_wide_panel" in
      let m, out = make () in
      let got =
        run ~name
          ~check:(fun opt ->
            p
              (name ^ ": no per-invocation master access remains")
              (not (LL.code_touches_tn m.Tensor.value opt.LL.llc)))
          ~schedule:(fun opt ->
            let axis =
              match
                List.find_exn (Ll_test.nest_paths opt.LL.llc) ~f:(fun p -> List.length p = 1)
              with
              | [ i ] -> i
              | _ -> assert false
            in
            let split, _, inner = S.split ~axis ~factor:16 ~outer:LL.Serial ~inner:LL.Serial in
            [ split; stage ~source:m.Tensor.value ~tile_loops:[ inner ] ~hoisted:true ~tile_prec ])
          out
      in
      p_all2
        (name ^ ": host packing matches the device conversion bitwise")
        got direct ~f:bits_equal)

(* A matmul carries the rounding through a tile into Tensorize, rather than leaving a wrapper over
   its operand. Both dimensions discriminate; the independent dyadic oracle is f32 exact. *)
(* cc-fp-contract: exact-operands -- conversion legs only copy; every dyadic matmul partial sum is f32-exact. *)
let () =
  let n = 128 and dims = [| 128; 128 |] in
  let a_cell = Ll_test.cycle ~dims ~modulus:7 ~offset:1. ~stride:0.125 in
  let b_cell = Ll_test.cycle ~dims ~modulus:13 ~offset:8. ~stride:0.125 in
  let output () =
    let a = NTDSL.init ~l:"inference_input" ~prec:Ir.Ops.single ~i:[ n ] ~o:[ n ] ~f:a_cell () in
    let m =
      NTDSL.ndarray
        (Array.init (n * n) ~f:(fun i -> b_cell (Ll_test.unflat ~dims i) +. (2. **. -12.)))
        ~label:[ "inference_master" ] ~input_dims:[ n ] ~output_dims:[ n ] ~top_down_prec:false ()
    in
    Tn.update_prec m.Tensor.value Ir.Ops.single;
    let b = half_twin m in
    let%op out = a * b in
    Tn.update_prec out.Tensor.value Ir.Ops.single;
    out
  in
  let expected =
    Array.init (n * n) ~f:(fun flat ->
        let idcs = Ll_test.unflat ~dims flat in
        let products =
          List.init n ~f:(fun k -> a_cell [| idcs.(0); k |] *. b_cell [| k; idcs.(1) |])
        in
        ( List.sum (module Float) products ~f:Fn.id,
          Ll_test.exact_partials ~sig_bits:(Ir.Ops.significand_bits Ir.Ops.single) products ))
  in
  p_all "inference products have f32-exact partial sums" (Array.to_list expected) ~f:snd;
  let want = Array.map expected ~f:fst in
  let ctx = Context.auto () in
  let cpu = Ir.Backend_intf.renders_on_host_pool (Context.backend_facts_of_name backend) in
  let limits = Context.hardware_limits ctx and caps = Context.codegen_capabilities ctx in
  Context.release ctx;
  let seeds = ref [] in
  let plain =
    run ~name:"frozen_inference_reference"
      ~schedule:(fun opt ->
        seeds := Autotune.sketch_seed_params ~is_cpu:cpu ~is_gpu:(not cpu) ~limits opt;
        [])
      (output ())
  in
  p_all2 "the frozen twin's serial inference matches the rounded oracle" plain want ~f:bits_equal;
  let leg = Verdict.leg ~when_:cpu ~on:backend () in
  List.iter [ false; true ] ~f:(fun grid ->
      let label =
        if grid then "frozen twins seed hoisted grid packing"
        else "frozen twins seed hoisted matmul packing"
      in
      in_leg ~leg label (fun label ->
          let hoisted =
            List.filter !seeds ~f:(fun p ->
                p.Autotune.sk_mma && p.sk_hoist && Bool.equal p.sk_grid grid && not p.sk_pack_rest)
          in
          let runs =
            List.mapi (List.take hoisted 1) ~f:(fun i p ->
                run
                  ~name:
                    (Printf.sprintf "frozen_inference_%s_%d" (if grid then "grid" else "serial") i)
                  ~schedule:(fun opt ->
                    Autotune.sketch_schedule ~accum_prec:caps.Ir.Backend_intf.accum_prec ~p opt)
                  (output ()))
          in
          p_all label runs ~f:(fun got -> Array.equal bits_equal got want)))

(* NaN payload conversion is backend-specific. Refuse those hoists and separate the cache identity,
   while ordinary finite data values still share one schedule identity. *)
let () =
  let compile name values =
    let m = constant ~label:"nan_guard_master" ~dims:[ 4 ] values in
    let t = half_twin m in
    let out = Operation.cast ~grad_spec:Tensor.Prohibit_grad t () in
    Tn.update_prec out.Tensor.value Ir.Ops.single;
    Train.set_materialized out.Tensor.value;
    let digest = ref "" and refusal = ref None in
    let ctx, _ =
      Context.compile
        ~lowered_transform:(fun opt ->
          digest := Ir.Schedule_cache.digest (Ir.Schedule_cache.canonicalize opt);
          let axis =
            match List.find_exn (Ll_test.nest_paths opt.LL.llc) ~f:(fun p -> List.length p = 1) with
            | [ i ] -> i
            | _ -> assert false
          in
          (match
             S.apply
               [
                 stage ~source:m.Tensor.value ~tile_loops:[ axis ] ~hoisted:true
                   ~tile_prec:(Some Ir.Ops.single);
               ]
               opt
           with
          | _ -> ()
          | exception Invalid_argument why -> refusal := Some why);
          [ opt ])
        (Context.auto ())
        (named name (Train.forward out))
        Ir.Indexing.Empty
    in
    Context.release ctx;
    (!digest, !refusal)
  in
  let finite = [| 1.000244140625; 2.000244140625; 3.000244140625; 4.000244140625 |] in
  let other = Array.map finite ~f:(fun x -> x +. 1.) in
  let nan = Array.copy finite in
  nan.(0) <- Float.nan;
  let a, ra = compile "nan_guard_finite" finite in
  let b, rb = compile "nan_guard_other" other in
  let c, rc = compile "nan_guard_nan" nan in
  p_all "NaN-free frozen masters admit host rounding" [ ra; rb ] ~f:Option.is_none;
  p "a NaN-bearing master refuses host rounding"
    (Option.value_map rc ~default:false ~f:(fun why ->
         String.is_substring why ~substring:"no verified host rounding"));
  p "ordinary finite values share a cache identity, NaN eligibility separates it"
    ((not (String.is_empty a)) && String.equal a b && not (String.equal a c));
  let m = constant ~label:"replaced_host_init" ~dims:[ 4 ] finite in
  let admitted () =
    Option.is_some (S.hoisted_read_rounding ~source:m.Tensor.value ~round_prec:Ir.Ops.half)
  in
  let first = admitted () in
  let replace values =
    Ir.Host_inits.register m.Tensor.value
      (lazy
        (Ir.Ndarray.init_array ~debug:"nan-domain" Ir.Ops.single ~dims:[| 4 |] ~padding:None
           ~f:(fun idcs -> values.(idcs.(0)))))
  in
  replace nan;
  let rejected = not (admitted ()) in
  replace finite;
  p_all "replacement host data refreshes cached NaN eligibility"
    [ first; rejected; admitted () ]
    ~f:Fn.id
