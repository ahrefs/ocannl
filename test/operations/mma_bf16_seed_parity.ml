(* gh-ocannl-2029: Metal's uniform-bf16 [simdgroup_matrix] tile and the serial loop narrow at
   different points, so a [tune] that crowned the tile changed the values (mlp_small bf16: 23 of 24
   losses off by 1.3e-4, where the f16 crown was bitwise equal on all 24).

   The root cause, measured on an M4 Max and characterized here: [simdgroup_multiply_accumulate]
   over a [simdgroup_bfloat8x8] accumulator sums each 8-deep k block in f32 and rounds to bf16 once
   per block -- the intrinsic is a k-blocked schedule of its own -- while the serial bfloat loop
   narrows at every k step. The f32-accumulator arm ([simdgroup_float8x8] over bfloat operand
   fragments) is an f32 chain whose partials equal the serial f32 accumulation, so since
   gh-ocannl-2029 Metal resolves [Bf16_auto] wide ([Metal_mma.bf16_accum_wide]), as HIP has since
   gh-ocannl-1051, and only [Bf16_narrow] keeps the narrow pair. The original three parts are Metal
   legs only; the narrow characterization and f16 seed control are M4 Max legs. The f16 census below
   also runs on CUDA and HIP.

   (i) The discriminating fixture: one hand-scheduled [Tile_mma] statement against its serial twin,
   over a single intrinsic block (k = 8) and four (k = 32). The inputs make the three narrowing
   models -- per k step, per 8-deep block, once -- exact predictions computed here (the premise
   claims below derive why, and that the models differ). Under the default policy and [Bf16_wide]
   tile and serial must equal the once-narrowed prediction bitwise. Under [Bf16_narrow] the claims
   characterize the M4-class hardware behaviour the documentation states (the tile narrows per
   block, the serial loop per step, and they differ): they are evidence for the docs, not the
   contract.

   (ii) The issue's own claim: every tensorized seed the autotuner proposes for a bf16 matmul on
   this device, built as the tuner builds it and executed over a prefilled destination, equals the
   serial rendering bitwise under the default policy -- so [tune] crowning any of them changes no
   value. Some seed must render the intrinsics, by the routine's census. The f16 twin is the
   positive control, which needed no change (both of its legs are per-step half FMAs).

   (iii) The cache identity: Metal's capabilities fingerprint under [Bf16_auto] equals [Bf16_wide]'s
   and differs from [Bf16_narrow]'s.

   Mismatch counts go to stderr; the golden holds the claims. *)

(* cc-fp-contract: not-cc -- every device bitwise claim is a GPU leg; on cc it skips and nothing
   executes *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Numerics = Ir.Numerics
open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"")
let on_metal = String.is_substring backend_name ~substring:"metal"
let on_cuda = String.is_substring backend_name ~substring:"cuda"
let on_hip = String.is_substring backend_name ~substring:"hip"
let on_gpu = on_metal || on_cuda || on_hip
let leg = Verdict.leg ~when_:on_metal ~on:backend_name ()

type f16_model = Per_step | Per_native_block | Hip_ordered_pairs

let measured_metal_family = "Apple M4 Max"

let measured_f16_models =
  [
    (("cuda", "sm_120"), Per_native_block);
    (("hip", "gfx1151"), Hip_ordered_pairs);
    (("metal", measured_metal_family), Per_step);
  ]

(* Decode ordinal zero, the device Context.auto executes on, through the backend's shared reader.
   Keep the decoder pure so missing fields cannot silently erase all measured-device coverage. *)
let family_of_fields ~backend fields =
  let atom key =
    Option.bind (List.Assoc.find fields key ~equal:String.equal) ~f:(function
      | Sexp.Atom value -> Some value
      | Sexp.List _ -> None)
  in
  match backend with
  | "cuda" ->
      let number key = Option.bind (atom key) ~f:Int.of_string_opt in
      Option.both (number "compute_capability_major") (number "compute_capability_minor")
      |> Option.map ~f:(fun (major, minor) -> Printf.sprintf "sm_%d%d" major minor)
  | "hip" ->
      Option.map (atom "gcn_arch_name") ~f:(fun arch -> List.hd_exn (String.split arch ~on:':'))
  | "metal" -> atom "device_name"
  | _ -> None

let select_f16_model ~backend ~family ~mma_capable =
  if not mma_capable then None
  else
    Option.bind family ~f:(fun family ->
        List.Assoc.find measured_f16_models (backend, family) ~equal:Poly.equal)

let executing_family =
  lazy
    (if not on_gpu then None
     else
       match
         Ir.Backend_intf.parse_static_properties (Context.static_properties (Context.auto ()))
       with
       | Some { devices = first :: _; _ } -> family_of_fields ~backend:backend_name first
       | _ -> None)

let family_label () =
  backend_name ^ "/" ^ Option.value (Lazy.force executing_family) ~default:"unidentified-family"

let measured_metal_leg () =
  Verdict.leg
    ~when_:
      (on_metal
      && Option.value_map (Lazy.force executing_family) ~default:false
           ~f:(String.equal measured_metal_family))
    ~on:(family_label ()) ()

let f16_mma_capable =
  lazy
    (on_gpu
    &&
    match (Context.hardware_limits (Context.auto ())).Ir.Backend_intf.mma with
    | None -> false
    | Some mma ->
        List.Assoc.mem mma.Ir.Backend_intf.mma_format_tiles
          (Ir.Backend_intf.Mma_f16, Ir.Backend_intf.Mma_f16, Ir.Backend_intf.Mma_f16)
          ~equal:Ir.Backend_intf.equal_mma_format_triple)

let () =
  p_all "device properties select measured models, including missing fields"
    [
      ( "cuda",
        [
          ("compute_capability_major", Sexp.Atom "12"); ("compute_capability_minor", Sexp.Atom "0");
        ],
        Some "sm_120",
        Some Per_native_block );
      ( "hip",
        [ ("gcn_arch_name", Sexp.Atom "gfx1151:xnack-") ],
        Some "gfx1151",
        Some Hip_ordered_pairs );
      ( "metal",
        [ ("device_name", Sexp.Atom "Apple M4 Max") ],
        Some measured_metal_family,
        Some Per_step );
      ("cuda", [], None, None);
    ]
    ~f:(fun (backend, fields, want_family, want_model) ->
      let family = family_of_fields ~backend fields in
      Option.equal String.equal family want_family
      && Poly.equal (select_f16_model ~backend ~family ~mma_capable:true) want_model);
  p_all ~min:3 "measured families require f16 tensor-unit capability before selecting their model"
    measured_f16_models ~f:(fun ((backend, family), model) ->
      Poly.equal (select_f16_model ~backend ~family:(Some family) ~mma_capable:true) (Some model)
      && Option.is_none (select_f16_model ~backend ~family:(Some family) ~mma_capable:false));
  p_all "unmeasured families select no model even when tensor units are available"
    [
      ("cuda", "sm_90");
      ("hip", "gfx1102");
      ("metal", "Apple M4 Pro");
      ("metal", "Apple M1");
      ("metal", "Apple M2");
      ("metal", "Apple M3");
    ]
    ~f:(fun (backend, family) ->
      Option.is_none (select_f16_model ~backend ~family:(Some family) ~mma_capable:true))

let n = 32
let bm = 16
let simd_width = 32

(* The seed-parity reduction depth: deep enough that every seed's k blocking (whole-k, 32 and 16
   deep on an M4 Max) contracts over several intrinsic blocks; the population is the seeder's,
   printed to stderr. *)
let k_seeds = 64

(* bf16 round-to-nearest-even of a value exact in f32 (every value rounded here is one, by the
   premise claims below). *)
let round_bf16 x =
  let bits = Int32.bits_of_float x in
  let lsb = Int32.(bit_and (shift_right_logical bits 16) one) in
  let rounded = Int32.(bit_and (bits + of_int_exn 0x7FFF + lsb) (of_int_exn (-65536))) in
  Int32.float_of_bits rounded

(* Multiples of 1/32 with |m| <= 127: at most 7 significant bits, exact in bf16 and f16 (the premise
   claims below derive what follows for the sums). Both vary with both indices. *)
let fa = Ll_test.weighted ~weights:[| 37; 11 |] ~modulus:255 ~offset:(-127.) ~stride:(1. /. 32.)
let fb = Ll_test.weighted ~weights:[| 29; 53 |] ~modulus:251 ~offset:(-125.) ~stride:(1. /. 32.)

(* The three narrowing models of cell (i, j), parameterized by format and intrinsic k depth. *)
let models ?(fa = fa) ?(fb = fb) ?(round = round_bf16) ?(k_index = fun _ kk -> kk) ?(block = 8) ~k
    () =
  let cell f =
    Array.init (n * n) ~f:(fun t ->
        let i = t / n and j = t % n in
        f (fun kk ->
            let kk = k_index i kk in
            fa [| i; kk |] *. fb [| kk; j |]))
  in
  let per_step =
    cell (fun prod ->
        let acc = ref 0. in
        for kk = 0 to k - 1 do
          acc := round (!acc +. prod kk)
        done;
        !acc)
  in
  let per_block =
    cell (fun prod ->
        let acc = ref 0. in
        for b = 0 to (k / block) - 1 do
          let partial = ref !acc in
          for kk = block * b to (block * (b + 1)) - 1 do
            partial := !partial +. prod kk
          done;
          acc := round !partial
        done;
        !acc)
  in
  let once =
    cell (fun prod ->
        let acc = ref 0. in
        for kk = 0 to k - 1 do
          acc := !acc +. prod kk
        done;
        round !acc)
  in
  (per_step, per_block, once)

let differ a b = not (Array.equal Float.equal a b)

let count_mismatches a b =
  Array.foldi a ~init:0 ~f:(fun c acc x -> if Float.equal x b.(c) then acc else acc + 1)

let named name (comp : Ir.Assignments.comp) : Ir.Assignments.comp =
  { comp with asgns = Ir.Assignments.Block_comment (name, comp.asgns) }

(* Every tensorized run starts its destination at [-4096 - c] per cell, below every value either
   fixture reaches, so a parity claim needs every cell overwritten (gh-ocannl-1625). *)
let run_prefilled ctx routine (t : Tensor.t) =
  let tn = t.Tensor.value in
  let numel = Array.fold (Tn.dims_without_padding tn) ~init:1 ~f:( * ) in
  Context.run
    (Context.set_values ctx tn (Array.init numel ~f:(fun c -> Float.of_int (-4096 - c))))
    routine

let matmul_with_inputs ~fa ~fb ~prec ~k ~tag =
  let a = NTDSL.init ~l:(tag ^ "_a") ~prec ~i:[ k ] ~o:[ n ] ~f:fa () in
  let b = NTDSL.init ~l:(tag ^ "_b") ~prec ~i:[ n ] ~o:[ k ] ~f:fb () in
  let%op c = a * b in
  Tn.update_prec c.Tensor.value prec;
  c

let matmul = matmul_with_inputs ~fa ~fb

(* The readback and the mma census of one compile; [transform] rewrites the lowering. *)
let run ?(prefill = false) ?(transform = fun o -> o) ~name (t : Tensor.t) =
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun o -> [ transform o ])
      (Context.auto ())
      (named name (Train.forward t))
      Ir.Indexing.Empty
  in
  let ctx = if prefill then run_prefilled ctx routine t else Context.run ctx routine in
  (Context.get_values ctx t.Tensor.value, Ll_test.renderings routine.Context.mma)

(* schedule_mma_matmul's hand schedule: the zero expanded ahead of one [Tile_mma] statement per
   16-row grid block, the whole k extent inside the statement. *)
let mma_schedule ~out (opt : LL.optimized) : Sched.schedule =
  let i, j, k =
    match List.find_exn (Ll_test.nest_paths opt.LL.llc) ~f:(fun p -> List.length p = 3) with
    | [ i; j; k ] -> (i, j, k)
    | _ -> assert false
  in
  let ez, zsyms = Sched.expand_zero ~tn:out in
  let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
  let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
  let rz = Sched.Retype { axis = zj; ty = LL.Workgroup } in
  let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
  let tz, _lane = Sched.tensorize ~i:i_i ~j ~k ~simd_width () in
  [ ez; sp_zi; rz; sp_i; tz ]

let bf16_policy mode p = { p with Numerics.bf16_arithmetic = mode }

let mode_name = function
  | Numerics.Bf16_auto -> "Bf16_auto"
  | Numerics.Bf16_wide -> "Bf16_wide"
  | Numerics.Bf16_narrow -> "Bf16_narrow"

(* One tile/serial pair; shared by the bf16 policies and the f16 intrinsic census below. *)
let fixture_pair_at ?(fa = fa) ?(fb = fb) ~prec ~k ~tag () =
  let matmul = matmul_with_inputs ~fa ~fb in
  let serial, _ = run ~name:(tag ^ "_serial") (matmul ~prec ~k ~tag:(tag ^ "s")) in
  let t = matmul ~prec ~k ~tag:(tag ^ "t") in
  let tile, census =
    run ~prefill:true ~name:(tag ^ "_tile")
      ~transform:(fun opt -> Sched.apply (mma_schedule ~out:t.Tensor.value opt) opt)
      t
  in
  (serial, tile, census)

let fixture_pair ~mode ~k =
  Test_utils.with_policy (bf16_policy mode) (fun () ->
      fixture_pair_at ~prec:Ir.Ops.bfloat16 ~k
        ~tag:(Printf.sprintf "bsp_%s_k%d" (String.lowercase (mode_name mode)) k)
        ())

let () =
  let per_step8, _, once8 = models ~k:8 () in
  let per_step32, per_block32, once32 = models ~k:32 () in
  (* The premises, from the inputs alone. Every product and every f32 partial sum of these operands
     is exact (derived by [Ll_test.exact_partials] at f32's 24 bits, over every k this file
     contracts), so the host's double arithmetic in [models] reproduces the device's f32 chains and
     the three models are exact predictions. At k = 8 one block is the whole reduction, so per-block
     and once coincide and only the per-step model differs; at k = 32 all three differ. *)
  p_all "every partial sum of these operands is exact in f32, at every k this file contracts"
    (List.concat_map [ 8; 32; k_seeds ] ~f:(fun k -> Ll_test.matmul_products ~m:n ~n ~k fa fb))
    ~f:(Ll_test.exact_partials ~any_sign:true ~sig_bits:(Ir.Ops.significand_bits Ir.Ops.single));
  p "k = 8: the per-step model differs from the once-narrowed one on these inputs"
    (differ per_step8 once8);
  p "k = 32: the per-step, per-block and once-narrowed models pairwise differ on these inputs"
    (differ per_step32 per_block32 && differ per_step32 once32 && differ per_block32 once32);
  let pairs =
    lazy
      (List.map [ Numerics.Bf16_auto; Numerics.Bf16_wide; Numerics.Bf16_narrow ] ~f:(fun mode ->
           (mode, [ (8, fixture_pair ~mode ~k:8); (32, fixture_pair ~mode ~k:32) ])))
  in
  let get mode k =
    List.Assoc.find_exn ~equal:Int.equal
      (List.Assoc.find_exn ~equal:Poly.equal (Lazy.force pairs) mode)
      k
  in
  let report mode k =
    let serial, tile, _ = get mode k in
    let per_step, per_block, once = models ~k () in
    Stdio.eprintf
      "mma_bf16_seed_parity: %s k=%d mismatches (of %d) -- tile vs serial %d; tile vs per-step %d, \
       per-block %d, once %d; serial vs per-step %d, once %d (not part of the golden)\n\
       %!"
      (mode_name mode) k (n * n) (count_mismatches tile serial) (count_mismatches tile per_step)
      (count_mismatches tile per_block) (count_mismatches tile once)
      (count_mismatches serial per_step)
      (count_mismatches serial once)
  in
  if on_metal then
    List.iter [ Numerics.Bf16_auto; Numerics.Bf16_wide; Numerics.Bf16_narrow ] ~f:(fun mode ->
        List.iter [ 8; 32 ] ~f:(fun k -> report mode k));
  List.iter [ Numerics.Bf16_auto; Numerics.Bf16_wide ] ~f:(fun mode ->
      List.iter
        [ (8, once8); (32, once32) ]
        ~f:(fun (k, once) ->
          in_legf ~leg "%s k=%d: every tile statement renders the simdgroup intrinsics"
            (mode_name mode) k (fun label ->
              let _, _, census = get mode k in
              p label (Ll_test.all_rendered ~what:label Ir.C_syntax.Mma_intrinsics census));
          in_legf ~leg "%s k=%d: the tile equals the serial twin bitwise" (mode_name mode) k
            (fun label ->
              let serial, tile, _ = get mode k in
              p_all2 label tile serial ~f:Float.equal);
          in_legf ~leg "%s k=%d: the serial twin equals the once-narrowed model bitwise"
            (mode_name mode) k (fun label ->
              let serial, _, _ = get mode k in
              p_all2 label serial once ~f:Float.equal)));
  (* The narrow pair is measured on M4 Max, rather than a policy-wide contract. *)
  let measured_leg = measured_metal_leg () in
  in_leg ~leg:measured_leg
    "Bf16_narrow k=32: the tile narrows once per 8-deep block (the per-block model)" (fun label ->
      let _, tile, _ = get Numerics.Bf16_narrow 32 in
      p_all2 label tile per_block32 ~f:Float.equal);
  in_leg ~leg:measured_leg
    "Bf16_narrow k=32: the serial loop narrows at every k step (the per-step model)" (fun label ->
      let serial, _, _ = get Numerics.Bf16_narrow 32 in
      p_all2 label serial per_step32 ~f:Float.equal);
  in_leg ~leg:measured_leg
    "Bf16_narrow k=8: one block -- the tile narrows once where the serial loop narrows per step"
    (fun label ->
      let serial, tile, _ = get Numerics.Bf16_narrow 8 in
      p label (Array.equal Float.equal tile once8 && Array.equal Float.equal serial per_step8))

(* === Every tensorized seed equals the serial rendering under the default policy === *)

let seed_parity ~prec ~tag =
  let limits = Context.hardware_limits (Context.auto ()) in
  let accum_prec = (Context.codegen_capabilities (Context.auto ())).Ir.Backend_intf.accum_prec in
  let seeds_of opt =
    List.filter (Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits opt) ~f:(fun p ->
        p.Autotune.sk_mma)
  in
  let serial, _ = run ~name:(tag ^ "_serial") (matmul ~prec ~k:k_seeds ~tag:(tag ^ "s")) in
  let labels = ref [] in
  let _ =
    run ~name:(tag ^ "_enum")
      ~transform:(fun opt ->
        labels := List.map (seeds_of opt) ~f:Autotune.sketch_params_label;
        opt)
      (matmul ~prec ~k:k_seeds ~tag:(tag ^ "e"))
  in
  let results =
    List.filter_mapi !labels ~f:(fun idx label ->
        let t = matmul ~prec ~k:k_seeds ~tag:(Printf.sprintf "%s%d" tag idx) in
        let transform opt =
          let p =
            List.find_exn (seeds_of opt) ~f:(fun p ->
                String.equal (Autotune.sketch_params_label p) label)
          in
          Sched.apply (Autotune.sketch_schedule ~accum_prec ~p opt) opt
        in
        match run ~prefill:true ~transform ~name:(Printf.sprintf "%s_seed%d" tag idx) t with
        | got, census ->
            let mism = count_mismatches got serial in
            Stdio.eprintf
              "mma_bf16_seed_parity: %s seed %d %s: %d of %d cells differ from serial, census %s \
               (not part of the golden)\n\
               %!"
              tag idx label mism (n * n) (Ll_test.census_string census);
            Some (label, got, census)
        | exception e ->
            Stdio.eprintf
              "mma_bf16_seed_parity: %s seed %d %s did not run: %s (not part of the golden)\n%!" tag
              idx label (Exn.to_string e);
            None)
  in
  Stdio.eprintf
    "mma_bf16_seed_parity: %s: %d tensorized seeds proposed, %d ran (not part of the golden)\n%!"
    tag (List.length !labels) (List.length results);
  (serial, results)

let () =
  let parity =
    lazy
      (Test_utils.with_policy
         (fun p ->
           {
             p with
             Numerics.bf16_arithmetic = Numerics.Bf16_auto;
             fp16_arithmetic = Numerics.Fp16_auto;
           })
         (fun () ->
           let bf16 = seed_parity ~prec:Ir.Ops.bfloat16 ~tag:"bsp_bf16" in
           let f16 = seed_parity ~prec:Ir.Ops.half ~tag:"bsp_f16" in
           (bf16, f16)))
  in
  List.iter
    [ ("bf16", fst); ("f16", snd) ]
    ~f:(fun (what, pick) ->
      let parity_leg = if String.equal what "f16" then measured_metal_leg () else leg in
      in_legf ~leg:parity_leg
        "default policy, %s: some seeded tile renders the simdgroup intrinsics" what (fun label ->
          let _, results = pick (Lazy.force parity) in
          p_exists label results ~f:(fun (_, _, census) ->
              List.exists census ~f:(Ir.C_syntax.equal_mma_rendering Ir.C_syntax.Mma_intrinsics)));
      in_legf ~leg:parity_leg
        "default policy, %s: every seeded tile that ran equals the serial rendering bitwise" what
        (fun label ->
          let serial, results = pick (Lazy.force parity) in
          p_all label results ~f:(fun (_, got, _) -> Array.equal Float.equal got serial)))

(* === Cache identity follows the resolution by derivation (gh-ocannl-1117/1153) === *)
(* The schedule cache hashes [Backend_intf.codegen_capabilities_fingerprint], which tabulates the
   backend's [accum_prec] and [mma_arm] under the current policy: Metal's [Bf16_auto] must now key
   as [Bf16_wide] does, and apart from [Bf16_narrow], so a crown timed with the narrow arm cannot
   replay under the default. *)
let () =
  in_leg ~leg
    "Metal's capabilities fingerprint under Bf16_auto equals Bf16_wide's and differs from \
     Bf16_narrow's" (fun label ->
      let fingerprint mode =
        Test_utils.with_policy (bf16_policy mode) (fun () ->
            Ir.Backend_intf.codegen_capabilities_fingerprint
              (Context.codegen_capabilities (Context.auto ())))
      in
      let auto = fingerprint Numerics.Bf16_auto in
      p label
        (String.equal auto (fingerprint Numerics.Bf16_wide)
        && not (String.equal auto (fingerprint Numerics.Bf16_narrow))))

(* === f16 tensor-unit narrowing census (gh-ocannl-2151) === *)
(* Keep the pre-existing bf16 Metal legs above. These fixtures deliberately select the narrow
   f16 arm, independent of the profile, and compare with exact host predictions rather than
   inferring narrowing from the fragment's declared type. The model table is measured evidence,
   not a vendor guarantee: unregistered families report measurements without a model verdict.
   CUDA sm_120 and HIP gfx1151 are probed at
   k=16/64; Metal M4 Max at k=8/32. Earlier Apple families remain unmeasured. *)
let () =
  let capable = Lazy.force f16_mma_capable in
  let selected =
    select_f16_model ~backend:backend_name ~family:(Lazy.force executing_family)
      ~mma_capable:capable
  in
  (* Intrinsic rendering and the explicit serial FMA are contracts on every capable GPU. Tile
     narrowing is a characterization only of families in measured_f16_models. *)
  let gpu_leg = Verdict.leg ~when_:capable ~on:(family_label ()) () in
  let model_leg = Verdict.leg ~when_:(Option.is_some selected) ~on:(family_label ()) () in
  (* One-sign operands isolate narrowing from gfx11 WMMA's signed-cancellation residue
     (gh-ocannl-1420/1509). Both indices still discriminate. The Dune action also pins
     fp_contract=true: the serial twin must use explicit FMAs rather than clang's per-kernel
     contraction choice (the unpinned k=16 HIP serial twin missed the FMA model in 49 cells). *)
  let fa indices = fa indices +. 4. in
  let fb indices = fb indices +. 4. in
  let block = if on_metal then 8 else 16 in
  let host_pairs_leg =
    Verdict.leg ~when_:(block = 16) ~on:"8-deep fixture (ordered pairs need 16-deep blocks)" ()
  in
  List.iter
    [ ("one-block", block); ("four-block", 4 * block) ]
    ~f:(fun (depth, k) ->
      let step, blocked, once = models ~fa ~fb ~round:Ir.Ops.round_to_half ~block ~k () in
      (* gfx1151's f16 WMMA narrows per pair, in a row-dependent order within each native 16-deep
         block: odd rows visit its upper eight terms first. [lxor 8] preserves the block number at
         both k depths. This is measured behaviour, not a claim about other AMD GPU families. *)
      let hip_pairs =
        lazy
          (let _, paired, _ =
             models ~fa ~fb ~round:Ir.Ops.round_to_half ~block:2
               ~k_index:(fun i kk -> if i % 2 = 0 then kk else kk lxor 8)
               ~k ()
           in
           paired)
      in
      p_all
        (Printf.sprintf "f16 %s: every product and partial is exact in f32, with one-sign products"
           depth)
        (Ll_test.matmul_products ~m:n ~n ~k fa fb)
        ~f:(Ll_test.exact_partials ~sig_bits:24);
      pf "f16 %s: per-step differs from both wider models" depth
        (differ step once && differ step blocked);
      pf "f16 %s: per-block equals once only for a single block" depth
        (Bool.equal (not (differ blocked once)) (k = block));
      in_legf ~leg:host_pairs_leg "f16 %s: HIP ordered pairs differ from all three baseline models"
        depth (fun label ->
          let paired = Lazy.force hip_pairs in
          p label (differ paired step && differ paired blocked && differ paired once));
      let pair =
        lazy
          (Test_utils.with_policy
             (fun p -> { p with Numerics.fp16_arithmetic = Numerics.Fp16_narrow })
             (fun () ->
               let serial, tile, census =
                 fixture_pair_at ~fa ~fb ~prec:Ir.Ops.half ~k
                   ~tag:(Printf.sprintf "bsp_f16_narrow_k%d" k)
                   ()
               in
               Stdio.eprintf
                 "f16 narrowing: device=%s k=%d block=%d tile mismatches: step=%d block=%d \
                  once=%d; serial: step=%d block=%d once=%d (not part of the golden)\n\
                  %!"
                 (family_label ()) k block (count_mismatches tile step)
                 (count_mismatches tile blocked) (count_mismatches tile once)
                 (count_mismatches serial step) (count_mismatches serial blocked)
                 (count_mismatches serial once);
               if block = 16 then
                 Stdio.eprintf "HIP ordered pairs: k=%d mismatches=%d (not part of the golden)\n%!"
                   k
                   (count_mismatches tile (Lazy.force hip_pairs));
               (serial, tile, census)))
      in
      (* Also measure unregistered families, but assign no pass/fail model to their results. *)
      if capable then ignore (Lazy.force pair);
      in_legf ~leg:gpu_leg "f16 %s: the narrow tile renders tensor-unit intrinsics" depth
        (fun label ->
          let _, _, census = Lazy.force pair in
          p label (Ll_test.all_rendered ~what:label Ir.C_syntax.Mma_intrinsics census));
      in_legf ~leg:gpu_leg "f16 %s: the serial loop equals the per-step model" depth (fun label ->
          let serial, _, _ = Lazy.force pair in
          p_all2 label serial step ~f:Float.equal);
      in_legf ~leg:model_leg "f16 %s: the tensor unit equals the measured family model" depth
        (fun label ->
          let _, tile, _ = Lazy.force pair in
          p_all2 label tile
            (match selected with
            | Some Per_step -> step
            | Some Per_native_block -> blocked
            | Some Hip_ordered_pairs -> Lazy.force hip_pairs
            | None -> assert false)
            ~f:Float.equal))
