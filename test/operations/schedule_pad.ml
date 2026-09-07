(* Pad-to-tile scheduling, gh-ocannl-485 (PADTO): masked edges so tensorized paths cover arbitrary
   extents. An awkward 33 x 65 x 70 matmul — every extent indivisible by every tile in play — runs
   the tensorized pipelines that previously required divisibility:

   - [Pad { axis; to_multiple_of }] extends each loop to the next tile multiple and guards each
   effectful leaf statement with [If (axis < N)], so the downstream [Split]s divide cleanly. -
   [Stage] tiles minted over the padded loops zero-fill their fringe (the [Where]-form edge guards),
   so the micro-kernel reads the add-reduce identity where the source ends. - [Tensorize] recognizes
   the guards: row/column masks move to the accumulator contraction's transfers (0-filled init-load,
   [If]-guarded store-back — the "padded scratch, only the valid region written out" discipline),
   and reduction-axis guards are discharged against the zero-fringe staged operands (padded
   contributions are exact zeros) — so the [Tile_mma] block covers the full padded extents
   guard-free and the intrinsic/register-tiled renderings fire.

   The CPU leg is the packed GEBP composition (pad i and k; the 65-wide column panel needs no pad —
   the register tiling peels columns natively): values must match the serial twin BITWISE (per
   output element the k-chain keeps serial order and fused rounding; padded k-chunks contribute
   exact FMA identities). The GPU leg is the shared cooperative composition with all three extents
   padded (j too: the [Tile_mma] n extent must be a tile multiple for the intrinsics); the masked
   contraction places the accumulator fragment in workgroup-shared memory — guarded transfers are
   not a fragment-resident scope, and thread-local arrays are not simdgroup-loadable — so the
   intrinsic path still fires, within f32 tolerance (the tile reduction reassociates).

   The negative pin: a pad guard over an operand read in place (no zero-fringe staged tile) must be
   rejected — the intrinsic path would read out of bounds. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments

let () = Utils.settings.output_debug_files_in_build_directory <- true

open Verdict.Claims

(* Zeros compare equal to zeros. A fragment mapping that reads outside the staged block, a kernel
   that never ran, or a reference whose own setup silently collapsed all yield all-zeros, and a
   parity check between two zero arrays passes while covering nothing (gh-ocannl-481 item 3). Every
   reference array is pinned nonzero where it is produced, so the parity claims below have
   content. *)
let nonzero name (a : float array) =
  if not (Array.exists a ~f:(fun x -> Float.(x <> 0.))) then
    failwith (name ^ ": the reference is all zeros — the parity checks against it are vacuous");
  a

let approx a b = Float.(abs (a - b) < 1e-2)
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let skipped = Verdict.skipped ~backend:backend_name
let on_cpu = Sched.backend_is_cpu backend_name
let on_gpu = Sched.backend_is_gpu backend_name

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name

(* The maximal single-child chains of statement-level loops: one symbol list per top-level nest. *)
let nest_paths (llc : LL.t) : Ir.Indexing.symbol list list =
  let strip stmts = List.filter stmts ~f:(function LL.Noop | LL.Comment _ -> false | _ -> true) in
  let rec path (llc : LL.t) : Ir.Indexing.symbol list =
    match llc with
    | LL.For_loop { index; body; _ } ->
        index :: (match strip (LL.flat_lines [ body ]) with [ single ] -> path single | _ -> [])
    | LL.If { body; _ } -> path body
    | _ -> []
  in
  List.filter_map (LL.flat_lines [ llc ]) ~f:(fun stmt ->
      match path stmt with [] -> None | p -> Some p)

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner })
let triple paths = List.find_exn paths ~f:(fun p -> List.length p = 3)

(* Awkward extents: indivisible by 8 and 16 (and by each other's tiles). *)
let m_ext, n_ext, k_ext = (33, 65, 70)
let bm, bk = (16, 16)
let simd_width = 32

let padded_gpu_shape_accepts_mma =
  match (Context.hardware_limits (Context.auto ())).Ir.Backend_intf.mma with
  | None -> false
  | Some mma ->
      let _, tile_n, _ = mma.Ir.Backend_intf.mma_tile in
      let padded_n = (n_ext + 7) / 8 * 8 in
      padded_n % tile_n = 0

let run_serial ~name (out : Tensor.t) =
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun opt -> [ opt ])
      (Context.auto ())
      (named name (Train.forward out))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  nonzero name (Context.get_values ctx out.Tensor.value)

let mav =
  Array.init (m_ext * k_ext)
    ~f:(Ll_test.cycle_flat ~dims:[| m_ext; k_ext |] ~modulus:13 ~offset:0. ~stride:0.25)

let mbv =
  Array.init (k_ext * n_ext)
    ~f:(Ll_test.cycle_flat ~dims:[| k_ext; n_ext |] ~modulus:17 ~offset:(-8.) ~stride:1.)

(* === CPU leg: padded packed GEBP (mirrors autotune's [cpu_mma_pack_sketch_schedule] with [bn = 0],
   plus the pads). Pad i to [bm], k to [bk]; split; sink to [k_o { i_o { i_i { j { k_i }}}}]; pack
   B~ [bk x n] at [k_o] and A~ [bm x bk] at [i_o]; tensorize with a unit lane. The pad guards
   survive as a row mask (transfers) and a discharged reduction mask; the [Tile_mma] runs 16 x 65 x
   16 blocks through the register tiling. === *)
let () =
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ k_ext ] ~output_dims:[ m_ext ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ n_ext ] ~output_dims:[ k_ext ] () in
  let%op pc0 = ma * mb in
  let want = run_serial ~name:"pad_serial" pc0 in
  let%op pc1 = ma * mb in
  let padded_schedule (opt : LL.optimized) : Sched.schedule =
    let i, j, k =
      match triple (nest_paths opt.LL.llc) with [ i; j; k ] -> (i, j, k) | _ -> assert false
    in
    let sp_i, i_o, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Serial ~inner:LL.Serial in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    let stage source tile_loops =
      Sched.Stage
        {
          source;
          tile_loops;
          shared = false;
          cooperative = None;
          hoisted = false;
          swizzle = None;
          pad_stride = None;
          pipeline_depth = 1;
          tile_prec = None;
        }
    in
    [ Sched.Pad { axis = i; to_multiple_of = bm }; Sched.Pad { axis = k; to_multiple_of = bk } ]
    @ [ sp_i; sp_k ] @ sink j [ k_o ] @ sink i_i [ k_o ] @ sink i_o [ k_o ]
    @ [ stage mb.Tensor.value [ k_i; j ]; stage ma.Tensor.value [ i_i; k_i ] ]
    @ if on_cpu then [ fst (Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width:1 ()) ] else []
  in
  let transform opt = Sched.apply (padded_schedule opt) opt in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun o -> [ transform o ])
      (Context.auto ())
      (named "pad_packed" (Train.forward pc1))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx pc1.Tensor.value in
  p_all2 "padded packed matmul matches the serial twin bitwise" got want ~f:Float.equal;
  let src = Generated.read "pad_packed" in
  let has s = String.is_substring src ~substring:s in
  p "padded 33x65x70 runs the register-tiled micro-kernel"
    (if on_cpu then
       has "Tile_mma register tiling" && not (has "Tile_mma renders the lane-0 scalar fallback")
     else not (has "tmma_"));
  (* The zero-filled fringe of the staged tiles: the load nests keep Where-form edge guards
     (rendered as conditional expressions storing 0 past the source extents), and the masked
     store-back writes only the valid rows. *)
  p "staged tiles zero-fill their fringe and the store-back is masked"
    ((not on_cpu)
     (* The bounds are float constants cast to the index type, so they carry the radix point every
        emitted float constant now does (gh-ocannl-623): [(int)(33.0)], same value. *)
    || (has "!= 0.0 ? " && has "(float)(0.0)" && has (Printf.sprintf "< (int)(%d.0))) {" m_ext)))

(* === GPU leg: padded shared cooperative composition (mirrors autotune's [gpu_mma_sketch_schedule],
   plus the pads — including the unsplit column panel, which the intrinsic n-extent needs). The
   masked contraction places the fragment in threadgroup memory and brackets the reduction loop with
   barriers; the store-back writes only the valid 33 x 65 region. === *)
let () =
  if on_gpu then (
    let ma = TDSL.ndarray mav ~label:[ "gma" ] ~input_dims:[ k_ext ] ~output_dims:[ m_ext ] () in
    let mb = TDSL.ndarray mbv ~label:[ "gmb" ] ~input_dims:[ n_ext ] ~output_dims:[ k_ext ] () in
    let%op gc0 = ma * mb in
    let want = run_serial ~name:"pad_gpu_serial" gc0 in
    let%op gc1 = ma * mb in
    let staged_schedule (opt : LL.optimized) : Sched.schedule =
      let i, j, k =
        match triple (nest_paths opt.LL.llc) with [ i; j; k ] -> (i, j, k) | _ -> assert false
      in
      let ez, zsyms = Sched.expand_zero ~tn:gc1.Tensor.value in
      let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
      (* Zeroing: Grid rows aligned with the accumulation's grid blocks (ceil(33/16) = 3 on both
         sides); the column loop splits into Serial x Workgroup(32) to satisfy barrier-strength
         extent uniformity with the lane loops. *)
      let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
      let sp_zj, _, _ =
        Sched.split ~axis:zj ~factor:simd_width ~outer:LL.Serial ~inner:LL.Workgroup
      in
      let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
      let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
      let tz, _lane = Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width () in
      [
        Sched.Pad { axis = i; to_multiple_of = bm };
        Sched.Pad { axis = j; to_multiple_of = 8 };
        Sched.Pad { axis = k; to_multiple_of = bk };
        ez;
        sp_zi;
        sp_zj;
        sp_i;
        sp_k;
        Sched.Swap { outer = j; inner = k_o };
        Sched.Swap { outer = i_i; inner = k_o };
        Sched.Stage
          {
            source = ma.Tensor.value;
            tile_loops = [ i_i; k_i ];
            shared = true;
            cooperative = Some simd_width;
            hoisted = false;
            swizzle = None;
            pad_stride = None;
            pipeline_depth = 1;
            tile_prec = None;
          };
        Sched.Stage
          {
            source = mb.Tensor.value;
            tile_loops = [ k_i; j ];
            shared = true;
            cooperative = Some simd_width;
            hoisted = false;
            swizzle = None;
            pad_stride = None;
            pipeline_depth = 1;
            tile_prec = None;
          };
        tz;
      ]
    in
    let transform opt = Sched.apply (staged_schedule opt) opt in
    let ctx, routine =
      Context.compile
        ~lowered_transform:(fun o -> [ transform o ])
        (Context.auto ())
        (named "pad_gpu_mma" (Train.forward gc1))
        Ir.Indexing.Empty
    in
    let ctx = Context.run ctx routine in
    let got = Context.get_values ctx gc1.Tensor.value in
    p_all2 "padded staged+tensorized matmul parity (GPU)" got want ~f:approx;
    (* The schedule's tensorization pass pads n to 8 (72 here). Whether that shape reaches a real
       intrinsic is a tile-capability question; the routine census is the authoritative outcome. *)
    p "padded GPU intrinsics follow the advertised tile divisibility"
      (Ir.C_syntax.equal_tensorization routine.mma.Ir.C_syntax.tensorization
         (if padded_gpu_shape_accepts_mma then Ir.C_syntax.Tensorized
          else Ir.C_syntax.Scalar_fallback)))
  else (
    (* Keep the printed transcript identical across backends. *)
    skipped "padded staged+tensorized matmul parity (GPU)";
    skipped "padded GPU intrinsics follow the advertised tile divisibility")

(* === Autotune pad-composition seeding (issue task 3): the divisibility pre-filters are gone for
   fully staged tensorized pipelines — the awkward site seeds staged candidates on both the GPU MMA
   and the CPU packed families, and a seeded CPU packed candidate executes with bitwise parity.
   In-place pipelines keep their gates: the unstaged GPU flavor ([sk_bk = 0]) must not be seeded
   here. === *)
let () =
  let ma = TDSL.ndarray mav ~label:[ "sma" ] ~input_dims:[ k_ext ] ~output_dims:[ m_ext ] () in
  let mb = TDSL.ndarray mbv ~label:[ "smb" ] ~input_dims:[ n_ext ] ~output_dims:[ k_ext ] () in
  let%op sc0 = ma * mb in
  let want = run_serial ~name:"pad_seed_serial" sc0 in
  let%op sc1 = ma * mb in
  let gpu_limits =
    {
      Ir.Backend_intf.no_hardware_limits with
      mma =
        Some
          {
            Ir.Backend_intf.mma_simd_width = 32;
            mma_tile = (8, 8, 8);
            mma_format_tiles =
              [
                ( (Ir.Backend_intf.Mma_f32, Ir.Backend_intf.Mma_f32, Ir.Backend_intf.Mma_f32),
                  (8, 8, 8) );
              ];
            mma_f16_wide_acc_scopes = [];
            mma_staged_layouts = [];
            mma_pipeline_depths = [];
          };
    }
  in
  let cpu_limits = { Ir.Backend_intf.no_hardware_limits with simd_vector_bytes = 32 } in
  let seeded = ref None in
  let transform opt =
    let gpu_seeds = Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits:gpu_limits opt in
    let cpu_seeds = Autotune.sketch_seed_params ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt in
    let staged_gpu =
      List.exists gpu_seeds ~f:(fun p -> p.Autotune.sk_mma && p.Autotune.sk_bk > 0)
    in
    let unstaged_gpu =
      List.exists gpu_seeds ~f:(fun p ->
          p.Autotune.sk_mma && p.Autotune.sk_gpu && p.Autotune.sk_bk = 0)
    in
    let packed_cpu =
      List.find cpu_seeds ~f:(fun p ->
          p.Autotune.sk_mma && p.Autotune.sk_bm = 16 && p.Autotune.sk_bk = 16
          && (not p.Autotune.sk_hoist) && (not p.Autotune.sk_grid) && (not p.Autotune.sk_pack_rest)
          && not p.Autotune.sk_epilogue)
    in
    seeded := Some (staged_gpu, unstaged_gpu, packed_cpu);
    match packed_cpu with
    | Some p when on_cpu -> Sched.apply (Autotune.sketch_schedule ~p opt) opt
    | _ -> opt
  in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun o -> [ transform o ])
      (Context.auto ())
      (named "pad_seeded" (Train.forward sc1))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx sc1.Tensor.value in
  (match !seeded with
  | Some (staged_gpu, unstaged_gpu, packed_cpu) ->
      p "awkward site seeds staged GPU MMA pad-compositions" staged_gpu;
      p "awkward site does not seed the unstaged (in-place) GPU flavor" (not unstaged_gpu);
      p "awkward site seeds the CPU packed pad-composition" (Option.is_some packed_cpu)
  | None ->
      p "awkward site seeds staged GPU MMA pad-compositions" false;
      p "awkward site does not seed the unstaged (in-place) GPU flavor" false;
      p "awkward site seeds the CPU packed pad-composition" false);
  p "seeded packed pad-composition matches the serial twin bitwise"
    ((not on_cpu) || Array.for_all2_exn got want ~f:Float.equal)

(* === Negative pin: a pad guard over an in-place operand (no zero-fringe staged tile) must be
   rejected — the intrinsic path would read past the operand's extent. === *)
let () =
  let ma = TDSL.ndarray mav ~label:[ "nma" ] ~input_dims:[ k_ext ] ~output_dims:[ m_ext ] () in
  let mb = TDSL.ndarray mbv ~label:[ "nmb" ] ~input_dims:[ n_ext ] ~output_dims:[ k_ext ] () in
  let%op nc = ma * mb in
  let unstaged_padded (opt : LL.optimized) : Sched.schedule =
    let i, j, k =
      match triple (nest_paths opt.LL.llc) with [ i; j; k ] -> (i, j, k) | _ -> assert false
    in
    [ Sched.Pad { axis = k; to_multiple_of = bk }; fst (Sched.tensorize ~i ~j ~k ~simd_width:1 ()) ]
  in
  let transform opt = Sched.apply (unstaged_padded opt) opt in
  let rejected =
    try
      let _ctx, _routine =
        Context.compile
          ~lowered_transform:(fun o -> [ transform o ])
          (Context.auto ())
          (named "pad_unstaged" (Train.forward nc))
          Ir.Indexing.Empty
      in
      false
    with Invalid_argument msg -> String.is_substring msg ~substring:"zero-fringe"
  in
  p "pad guard over an unstaged operand is rejected" rejected
