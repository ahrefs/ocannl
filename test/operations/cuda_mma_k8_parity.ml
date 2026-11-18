(* gh-ocannl-2284: CUDA's Turing arm for uniform f16 per statement, executed.

   Under the wide f16 resolution (the default [Fp16_auto] on CUDA since gh-ocannl-2195) a
   per-statement uniform-f16 [Tile_mma] renders through [mma.sync] m16n8k16 where the device meets
   that arm's floor (sm_80) and through m16n8k8 below it, down to sm_75 ([Cuda_mma.f16_wide_arm]).
   No fleet device sits between the two floors, so the m16n8k8 rendering is forced on the attached
   device through [Cuda_mma.with_resolution_ceiling]: inside the extent the hooks resolve arms as a
   sm_75 device would, while acceptance, the descriptor and the nvrtc target keep the device's own
   capability. The instruction exists on every architecture from sm_75 up, so the rendering a Turing
   device gets runs here as it is, its PTX targeting the arm's floor (read from the compile's own
   [.target] directive through [Cuda_mma.with_ptx_target]).

   The fixture is one hand-scheduled [Tile_mma] statement per 16-row block with the whole k extent
   inside it (mma_bf16_seed_parity's schedule), over k = 32: four m16n8k8 steps, two m16n8k16 ones.
   Three renderings of the same matmul are compared: the m16n8k8 tile, the m16n8k16 tile and the
   serial loop with no tile at all; each tile runs over a prefilled destination, so every cell has
   to be written. The premise claims below derive from the operands why the three, and the host's
   sum, agree bitwise. The operands vary with every index, and a contraction that pairs the wrong k
   (a register of the wrong k half) is exhibited to differ on them, so a fragment layout error
   cannot agree by accident.

   Which arm rendered is read from the routine's census (tensorized) and its generated text: the
   arm's marker and instruction, the one text pin here, stating which dialect instruction ran. The
   serial reference runs on every backend; every tile claim is a CUDA leg, and elsewhere no tile
   compiles. Each arm's claims also skip on a device below that arm's floor: on sm_75-79 the
   device's own resolution is m16n8k8 itself, so the m16n8k16 control has nothing to run. *)

(* cc-fp-contract: exact-operands -- the premise claims below bound every reduction of these
   quarter-valued operands within f16, so contraction moves no value *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Sched = Ir.Schedule
module M = Ir.Cuda_mma
module G = Test_utils.Generated
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let on_cuda = String.is_substring backend_name ~substring:"cuda"
let leg = Verdict.leg ~when_:on_cuda ~on:backend_name ()

(* The attached devices' minimum compute capability, the one the CUDA hooks render at, from the
   backend's own static properties; [None] where none reads. *)
let device_cc =
  lazy
    (match
       Ir.Backend_intf.parse_static_properties (Context.static_properties (Context.auto ()))
     with
    | Some { devices; _ } ->
        List.filter_map devices ~f:(fun fields ->
            let number key =
              match List.Assoc.find fields key ~equal:String.equal with
              | Some (Sexp.Atom v) -> Int.of_string_opt v
              | _ -> None
            in
            Option.map2 (number "compute_capability_major") (number "compute_capability_minor")
              ~f:(fun major minor -> (major * 10) + minor))
        |> List.min_elt ~compare:Int.compare
    | None -> None)

(* A CUDA device that meets [arm]'s floor: below it the hooks decline the arm, so its claims skip on
   the device's capability. *)
let arm_leg arm =
  Verdict.sub_leg ~aggregation:`Environment leg
    ~when_:
      (lazy
        (Option.value_map (Lazy.force device_cc) ~default:false ~f:(fun cc -> cc >= M.arm_floor arm)))
    ~on:(Printf.sprintf "a CUDA device below %s's floor" (M.arm_marker arm))

let n = 32
let k = 32
let bm = 16
let simd_width = 32

(* Quarters from 1/4 to 7/4, so every operand is exact in f16 and every product is positive; the
   weights differ, so neither operand is a function of its index sum. *)
let fa =
  Ll_test.operand ~b:[] ~o:[ n ] ~i:[ k ]
    (Ll_test.weighted ~weights:[| 3; 5 |] ~modulus:7 ~offset:1. ~stride:0.25)

let fb =
  Ll_test.operand ~b:[] ~o:[ k ] ~i:[ n ]
    (Ll_test.weighted ~weights:[| 2; 3 |] ~modulus:7 ~offset:1. ~stride:0.25)

(* The same contraction pairing [a]'s k with [b]'s k + 8 (mod k): what a register loaded from the
   other k half computes. *)
let misaligned =
  Array.init (n * n) ~f:(fun t ->
      let i = t / n and j = t % n in
      List.sum
        (module Float)
        (List.range 0 k)
        ~f:(fun kk -> fa [| i; kk |] *. fb [| (kk + 8) % k; j |]))

let matmul ~tag =
  let a = NTDSL.init ~l:(tag ^ "_a") ~prec:Ir.Ops.half ~i:[ k ] ~o:[ n ] ~f:fa () in
  let b = NTDSL.init ~l:(tag ^ "_b") ~prec:Ir.Ops.half ~i:[ n ] ~o:[ k ] ~f:fb () in
  let%op c = a * b in
  Tn.update_prec c.Tensor.value Ir.Ops.half;
  c

(* The zero expanded ahead of one [Tile_mma] statement per 16-row grid block, the whole k extent
   inside the statement: the per-statement scope. *)
let mma_schedule ~out (opt : LL.optimized) : Sched.schedule =
  let i, j, kx =
    match List.find_exn (Ll_test.nest_paths opt.LL.llc) ~f:(fun p -> List.length p = 3) with
    | [ i; j; kx ] -> (i, j, kx)
    | _ -> assert false
  in
  let ez, zsyms = Sched.expand_zero ~tn:out in
  let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
  let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
  let rz = Sched.Retype { axis = zj; ty = LL.Workgroup } in
  let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
  let tz, _lane = Sched.tensorize ~i:i_i ~j ~k:kx ~simd_width () in
  [ ez; sp_zi; rz; sp_i; tz ]

let named name (comp : Ir.Assignments.comp) : Ir.Assignments.comp =
  { comp with asgns = Ir.Assignments.Block_comment (name, comp.asgns) }

(* One compile and run of [name]: the destination's values, the routine's mma census, and the
   [.target] of every PTX program the compile produced. [tile] applies the tile schedule and
   prefills the destination below every value the fixture reaches. *)
let run ~tile ~name =
  let t = matmul ~tag:name in
  let transform opt =
    if tile then [ Sched.apply (mma_schedule ~out:t.Tensor.value opt) opt ] else [ opt ]
  in
  let (ctx, routine), compiles =
    M.with_ptx_target M.Floor (fun () ->
        Context.compile ~lowered_transform:transform (Context.auto ())
          (named name (Train.forward t))
          Ir.Indexing.Empty)
  in
  let ctx =
    if tile then
      Context.set_values ctx t.Tensor.value
        (Array.init (n * n) ~f:(fun c -> Float.of_int (-4096 - c)))
    else ctx
  in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx t.Tensor.value in
  (got, Ll_test.renderings routine.Context.mma, List.filter_map compiles ~f:(fun c -> c.M.target))

let has logic substring = String.is_substring logic ~substring

let count_mismatches a b =
  Array.foldi a ~init:0 ~f:(fun c acc x -> if Float.equal x b.(c) then acc else acc + 1)

let () =
  Utils.settings.output_debug_files_in_build_directory <- true;
  G.init ~backend_name;
  Ir.Numerics.set_policy { (Ir.Numerics.get ()) with fp16_arithmetic = Ir.Numerics.Fp16_auto };
  p_all "every operand is a quarter in 1/4 .. 7/4, exact in f16"
    (List.concat_map (List.range 0 n) ~f:(fun r ->
         List.concat_map (List.range 0 k) ~f:(fun c -> [ fa [| r; c |]; fb [| c; r |] ])))
    ~f:(fun v -> Float.(v >= 0.25 && v <= 1.75 && Float.equal (Float.round (v * 4.)) (v * 4.)));
  (* Every partial sum of the matmul, in any association, is exact at f16's significand, so neither
     an f32 accumulation order nor the one narrowing to f16 at the store can move a value: every
     rendering returns the host's sum. *)
  let products = Ll_test.matmul_products ~m:n ~n ~k fa fb in
  p_all "every partial sum of these operands is exact in f16, with one-sign products" products
    ~f:(Ll_test.exact_partials ~sig_bits:(Ir.Ops.significand_bits Ir.Ops.half));
  let host = Array.of_list (List.map products ~f:(List.sum (module Float) ~f:Fn.id)) in
  p "pairing the wrong k half changes some cell of the product"
    (count_mismatches host misaligned > 0);
  let serial, serial_census, _ = run ~tile:false ~name:"k8p_serial" in
  p_all2 "the serial rendering equals the premise's host sums bitwise" serial
    (Array.of_list (List.map products ~f:(List.sum (module Float) ~f:Fn.id)))
    ~f:Float.equal;
  p "the serial rendering renders no tile"
    (Ll_test.none_emitted ~what:"the serial rendering" serial_census);
  let k16_arm = M.Inline_ptx (M.M16n8k16 M.M16n8k16_f16) and k8_arm = M.Inline_ptx M.M16n8k8_f16 in
  (* Each tile runs only where its leg is open: on sm_75-79 the device's own resolution is m16n8k8,
     so the m16n8k16 control and the cross-arm claim skip there. *)
  let k16_leg = arm_leg k16_arm and k8_leg = arm_leg k8_arm in
  let report tag (got, _, targets) =
    Stdio.eprintf
      "cuda_mma_k8_parity: %s: %d of %d cells differ from the serial rendering; PTX targets %s \
       (not part of the golden)\n\
       %!"
      tag (count_mismatches got serial) (n * n) (String.concat ~sep:"," targets)
  in
  let k16_run =
    lazy
      (let r = run ~tile:true ~name:"k8p_m16n8k16" in
       report "m16n8k16" r;
       r)
  in
  let k8_run =
    lazy
      (let r =
         M.with_resolution_ceiling ~cc:(M.arm_floor k8_arm) (fun () ->
             run ~tile:true ~name:"k8p_m16n8k8")
       in
       report "m16n8k8" r;
       r)
  in
  let k16 () = Lazy.force k16_run and k8 () = Lazy.force k8_run in
  let logic name = G.main_logic (G.read name) in
  let marked arm logic = has logic ("(" ^ M.arm_marker arm ^ ")") in
  in_leg ~leg:k16_leg "at the device's own resolution the tile renders the m16n8k16 arm, tensorized"
    (fun label ->
      let _, census, _ = k16 () in
      let logic = logic "k8p_m16n8k16" in
      p label
        (Ll_test.all_tensorized ~what:label census
        && marked k16_arm logic
        && has logic "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"));
  in_leg ~leg:k16_leg "the m16n8k16 tile equals the serial rendering bitwise" (fun label ->
      let got, _, _ = k16 () in
      p_all2 label got serial ~f:Float.equal);
  in_leg ~leg:k8_leg
    "at sm_75's resolution the tile renders the m16n8k8 arm, tensorized, and no m16n8k16"
    (fun label ->
      let _, census, _ = k8 () in
      let logic = logic "k8p_m16n8k8" in
      p label
        (Ll_test.all_tensorized ~what:label census
        && marked k8_arm logic
        && has logic "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
        && (not (has logic "m16n8k16"))
        && has logic "__half2float" && has logic "__float2half"));
  in_leg ~leg:k8_leg "the m16n8k8 kernel's PTX targets the arm's floor, sm_75" (fun label ->
      let _, _, targets = k8 () in
      p_all ~min:1 label targets ~f:(String.equal "sm_75"));
  in_leg ~leg:k8_leg "the m16n8k8 tile equals the serial rendering bitwise" (fun label ->
      let got, _, _ = k8 () in
      p_all2 label got serial ~f:Float.equal);
  in_leg ~leg:k16_leg "the m16n8k8 tile equals the m16n8k16 tile bitwise" (fun label ->
      let got8, _, _ = k8 () and got16, _, _ = k16 () in
      p_all2 label got8 got16 ~f:Float.equal)
