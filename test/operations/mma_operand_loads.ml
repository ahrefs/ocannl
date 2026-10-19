(* gh-ocannl-1254: the [mma_operand_loads] capability answers with the emission's own load rule.

   [Backend_intf.codegen_capabilities.mma_operand_loads] says which of a tile-MMA's [a] and [b]
   operands the backend loads with a warp-cooperative matrix load ([ldmatrix] on CUDA) rather than
   per-lane gathers. The cost model's load tie-break reads it, so it must be the decision the kernel
   makes, not a second copy of it: CUDA's emission, its rendering census and the capability all
   reach [Cuda_mma.arm_ldmatrix_operands].

   The legs compile staged + tensorized matmuls of the shapes [schedule_ldmatrix_matmul] pins, each
   operand tile plain or [Swizzle_b128], stored in its role's orientation or transposed, and compare
   what the capability answers for the leg's operand places with what was emitted: A comes in
   through [ldmatrix ... .x4] and B through [... .x2] in every inline-PTX arm. The legs cover all
   four answers on CUDA (bf16 both operands, fp8 A-only and B-only, gathers only), plus the wmma f16
   arm, which never loads through [ldmatrix] and declines a swizzled operand.

   Where it is evaluated: every GPU backend can compile the legs, and on HIP and Metal (no such
   load) the claim is that neither the capability nor the kernel uses one. A leg whose capability
   promises an [ldmatrix] is evaluated only where the device's tile-MMA descriptor advertises the
   leg's storage triple — the arm's floor gates the emission, which the capability (an arm-level
   answer) does not — and is skipped elsewhere. The C backends cannot place a tile in shared memory,
   so their emission legs are skipped; the capability's exhaustive claims run everywhere. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
module BI = Ir.Backend_intf
open Verdict.Claims

let () = Utils.settings.output_debug_files_in_build_directory <- true
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_gpu = Sched.backend_is_gpu backend_name

(* Intentional dialect identity: [ldmatrix] is a CUDA instruction, and the capability's (false,
   false) contract elsewhere is what the exhaustive claim below pins. *)
let on_cuda = String.is_substring backend_name ~substring:"cuda"

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name
let caps = Context.codegen_capabilities (Context.auto ())
let limits = Context.hardware_limits (Context.auto ())

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let n = 32
let simd_width = 32

(* [schedule_ldmatrix_matmul]'s staged leg: i_o(Grid) { k_o { i_i { j { k_i } } } }, both operands
   staged through cooperative shared tiles anchored at k_o, the micro-kernel tensorized. The
   [tile_loops] order decides each staged tile's orientation. *)
let staged_schedule ~bm ~bk ~out ~src_a ~src_b ~swz_a ~swz_b ~ta ~tb (opt : LL.optimized) =
  let i, j, k =
    match List.find_exn (Ll_test.nest_paths opt.LL.llc) ~f:(fun p -> List.length p = 3) with
    | [ i; j; k ] -> (i, j, k)
    | _ -> assert false
  in
  let ez, zsyms = Sched.expand_zero ~tn:out in
  let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
  let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
  let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
  let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
  let tz, _lane = Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width () in
  let stage source tile_loops swizzle =
    Sched.Stage
      {
        source;
        tile_loops;
        shared = true;
        cooperative = Some simd_width;
        hoisted = false;
        swizzle;
        pad_stride = None;
        pipeline_depth = 1;
        tile_prec = None;
      }
  in
  Sched.apply
    [
      ez;
      sp_zi;
      Sched.Retype { axis = zj; ty = LL.Workgroup };
      sp_i;
      sp_k;
      Sched.Swap { outer = j; inner = k_o };
      Sched.Swap { outer = i_i; inner = k_o };
      stage src_a (if ta then [ k_i; i_i ] else [ i_i; k_i ]) swz_a;
      stage src_b (if tb then [ j; k_i ] else [ k_i; j ]) swz_b;
      tz;
    ]
    opt

let layout = function Some LL.Swizzle_b128 -> `Swizzled_b128 | _ -> `Plain

let format_of = function
  | Ir.Ops.Fp8_prec _ -> BI.Mma_fp8_e5m2
  | Ir.Ops.Bfloat16_prec _ -> BI.Mma_bf16
  | Ir.Ops.Half_prec _ -> BI.Mma_f16
  | _ -> BI.Mma_f32

let has src s = String.is_substring src ~substring:s

(* One leg: what the capability answers for the staged operands' places, against the [ldmatrix]
   forms and census the compiled kernel carries. *)
let leg ~tag ~a ~b ~d_prec ?(bm = 16) ?(bk = 16) ?(ta = false) ?(tb = false) ~swz_a ~swz_b () =
  let a_prec = Lazy.force a.Tensor.value.Tn.storage_prec
  and b_prec = Lazy.force b.Tensor.value.Tn.storage_prec in
  let promised =
    caps.BI.mma_operand_loads ~a_prec ~b_prec ~d_prec ~scope:BI.Mma_per_statement ~ta ~tb
      ~a:(`Shared, layout swz_a)
      ~b:(`Shared, layout swz_b)
  in
  let label =
    Printf.sprintf "%s: the kernel loads through ldmatrix exactly the operands the capability names"
      tag
  in
  let advertised =
    BI.advertises_mma_format limits ~a:(format_of a_prec) ~b:(format_of b_prec)
      ~d:(format_of d_prec)
  in
  let evaluable = on_gpu && (advertised || Poly.equal promised (false, false)) in
  (if not evaluable then skipped ~backend:backend_name label
   else
     let t =
       let%op t = a * b in
       t
     in
     Tn.update_prec t.Tensor.value d_prec;
     let name = "mol_" ^ tag in
     let _ctx, routine =
       Context.compile
         ~lowered_transform:(fun o ->
           [
             staged_schedule ~bm ~bk ~out:t.Tensor.value ~src_a:a.Tensor.value ~src_b:b.Tensor.value
               ~swz_a ~swz_b ~ta ~tb o;
           ])
         (Context.auto ())
         (named name (Train.forward t))
         Ir.Indexing.Empty
     in
     let src = Generated.read name in
     let emitted =
       (has src "ldmatrix.sync.aligned.m8n8.x4", has src "ldmatrix.sync.aligned.m8n8.x2")
     in
     (* The census reads the same hook: [Mma_intrinsics_ldmatrix] exactly when an operand does. *)
     let census = List.map routine.Context.mma.Ir.C_syntax.renderings ~f:snd in
     let ldm_census =
       List.for_all census ~f:(Ir.C_syntax.equal_mma_rendering Ir.C_syntax.Mma_intrinsics_ldmatrix)
     in
     p label
       (Poly.equal emitted promised
       && (not (List.is_empty census))
       && Bool.equal ldm_census (fst promised || snd promised)));
  promised

let cycle = Ll_test.cycle ~dims:[| n; n |]

let () =
  let mk ~l ~prec ~f = NTDSL.init ~l ~prec ~i:[ n ] ~o:[ n ] ~f () in
  let bf_a = mk ~l:"mol_bfa" ~prec:Ir.Ops.bfloat16 ~f:(cycle ~modulus:3 ~offset:0. ~stride:0.25) in
  let bf_b =
    mk ~l:"mol_bfb" ~prec:Ir.Ops.bfloat16 ~f:(cycle ~modulus:5 ~offset:(-2.) ~stride:0.5)
  in
  let f8_a = mk ~l:"mol_f8a" ~prec:Ir.Ops.fp8 ~f:(cycle ~modulus:5 ~offset:(-2.) ~stride:0.5) in
  let f8_b = mk ~l:"mol_f8b" ~prec:Ir.Ops.fp8 ~f:(cycle ~modulus:7 ~offset:(-3.) ~stride:0.5) in
  let f16_a = mk ~l:"mol_f16a" ~prec:Ir.Ops.half ~f:(cycle ~modulus:3 ~offset:0. ~stride:0.25) in
  let f16_b = mk ~l:"mol_f16b" ~prec:Ir.Ops.half ~f:(cycle ~modulus:5 ~offset:(-2.) ~stride:0.5) in
  let b128 = Some LL.Swizzle_b128 in
  (* Thunks, run in order: a list literal's elements evaluate right to left. *)
  let answers =
    List.map
      ~f:(fun leg -> leg ())
      [
        (fun () ->
          leg ~tag:"bf16_ab" ~a:bf_a ~b:bf_b ~d_prec:Ir.Ops.bfloat16 ~swz_a:b128 ~swz_b:b128 ());
        (fun () ->
          leg ~tag:"bf16_ab_trans" ~a:bf_a ~b:bf_b ~d_prec:Ir.Ops.bfloat16 ~ta:true ~tb:true
            ~swz_a:b128 ~swz_b:b128 ());
        (* fp8 needs [bk = 32]: m16n8k32's reduction tile, and two whole 16-byte units per row. *)
        (fun () ->
          leg ~tag:"fp8_a" ~a:f8_a ~b:f8_b ~d_prec:Ir.Ops.single ~bk:32 ~swz_a:b128 ~swz_b:None ());
        (fun () ->
          leg ~tag:"fp8_b_trans" ~a:f8_a ~b:f8_b ~d_prec:Ir.Ops.single ~bk:32 ~tb:true ~swz_a:None
            ~swz_b:b128 ());
        (* The autotuner's row-major swizzled twin: A on ldmatrix, B's strided bytes gathered. *)
        (fun () ->
          leg ~tag:"fp8_twin_swizzled" ~a:f8_a ~b:f8_b ~d_prec:Ir.Ops.single ~bk:32 ~swz_a:b128
            ~swz_b:b128 ());
        (fun () ->
          leg ~tag:"fp8_twin_plain" ~a:f8_a ~b:f8_b ~d_prec:Ir.Ops.single ~bk:32 ~swz_a:None
            ~swz_b:None ());
        (fun () ->
          leg ~tag:"fp8_a_trans_gather" ~a:f8_a ~b:f8_b ~d_prec:Ir.Ops.single ~bm:32 ~bk:32 ~ta:true
            ~swz_a:b128 ~swz_b:None ());
        (fun () ->
          leg ~tag:"f16_wmma" ~a:f16_a ~b:f16_b ~d_prec:Ir.Ops.single ~swz_a:b128 ~swz_b:b128 ());
      ]
  in
  (* The legs discriminate: on CUDA they reach all four answers, so a capability stuck at one value
     (or a test whose legs all share one) fails here rather than agreeing with itself. *)
  List.iter
    [ (true, true); (true, false); (false, true); (false, false) ]
    ~f:(fun want ->
      gated_exists ~when_:on_cuda ~on:backend_name
        (Printf.sprintf "on CUDA some leg's capability answers (%b, %b)" (fst want) (snd want))
        answers ~f:(Poly.equal want));
  (* The capability over every storage triple, emission scope, orientation and operand place: an
     operand that is not a swizzled shared tile never loads through ldmatrix on any backend, and off
     CUDA none does. *)
  let places = List.cartesian_product [ `Device; `Shared; `Thread ] [ `Plain; `Swizzled_b128 ] in
  let bools = [ false; true ] in
  let all_cases =
    List.concat_map Ir.Ops.all_precs ~f:(fun a_prec ->
        List.concat_map Ir.Ops.all_precs ~f:(fun b_prec ->
            List.concat_map Ir.Ops.all_precs ~f:(fun d_prec ->
                List.concat_map BI.all_of_mma_emission_scope ~f:(fun scope ->
                    List.concat_map (List.cartesian_product bools bools) ~f:(fun (ta, tb) ->
                        List.concat_map places ~f:(fun a ->
                            List.map places ~f:(fun b ->
                                ( (a, b),
                                  caps.BI.mma_operand_loads ~a_prec ~b_prec ~d_prec ~scope ~ta ~tb
                                    ~a ~b ))))))))
  in
  let eligible (place : BI.mma_operand_place) = Poly.equal place (`Shared, `Swizzled_b128) in
  p_all "only a swizzled shared operand is ever loaded through ldmatrix" all_cases
    ~f:(fun ((a, b), (a_ldm, b_ldm)) -> ((not a_ldm) || eligible a) && ((not b_ldm) || eligible b));
  gated_all ~when_:(not on_cuda) ~on:backend_name
    "off CUDA the capability answers (false, false) everywhere" all_cases ~f:(fun (_, answer) ->
      Poly.equal answer (false, false))
