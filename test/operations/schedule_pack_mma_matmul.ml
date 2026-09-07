(* Tile_mma composed with cache tiling / operand packing on the CPU backends (the remaining piece of
   gh-ocannl-469): packed Stage tiles feeding the register-tiled Tile_mma micro-kernel, the
   compiler-BLAS (GEBP) structure.

   The composed pipeline, all-Serial: split i by [bm] and k by [bk] (j stays the full column panel),
   sink to [k_o { i_o { i_i { j { k_i }}}}], pack the B panel [bk x n] at [k_o] (once per k-block,
   reused across every row block) and the A tile [bm x bk] at [i_o] via non-shared [Stage]s with
   [tile_loops] in micro-kernel order, then [Tensorize { i = i_i; j; k = k_i }] with a unit lane
   width — the C backends render the lane loop serially, and a unit lane keeps the kernel
   single-threaded so the whole-node [Zero_out] needs no expansion. The [Tile_mma] block statement
   then streams the contiguous, cache-resident tiles ([lda = bk], [ldb = n]) through the
   register-tiled rendering; per output element the k-chain keeps the serial order and fused
   rounding, so values match the serial twin BITWISE on the C backends.

   The transposed-B case ([d[i,j] += a[i,k] * bt[j,k]], the gradient-GEMM layout the register tiling
   declines whole-triple) is the pack-normalization headline: [Stage] orders the tile's axes by
   [tile_loops] ([k_i; j]), so the packed B~ tile is k-major regardless of the source layout and
   [Tensorize] sees [tb = false] — the register tiling fires on a layout it could not handle in
   place.

   The Grid-blocked whole-triple form checks the pool-parallel composition: [parallel_grid_safe]
   analyzes [Tile_mma] through its scalar fallback (the statement's true access footprint), so a
   Grid row-block loop over a Tile_mma with materialized operands renders as chunked parallel loops
   on the native pool ([dispatch_apply] / OpenMP).

   The parallel packed form is the fully parallel packed GEMM (in-kernel packing under Grid row
   blocks): the per-row-block A~ tile is privatized to per-chunk block-scope storage by the renderer
   ([parallel_grid_safe]'s privatization rule) while the B~ panel packed at [k_o] outside the Grid
   is read-only inside (behind a pointer alias under the blocks extension).

   The Grid + hoisted-pack form pins autotune's [sk_grid && sk_hoist] seed variant (the typical
   inference GEMM: activations x constant weights): pool-parallel Grid row blocks over the
   cache-blocked packed pipeline, with ONLY the hoistable B operand packed — at link time, into the
   constant pool (gh-ocannl-470) — and A read in place (standard layout, [ta = false]). The kernel
   body then has no in-kernel tile writes at all, so the Grid split stays outermost. In both Grid
   forms the whole-node [Zero_out] is expanded with matching Grid coverage (the unit-lane Workgroup
   axis stays inactive).

   The mixed grid-outermost form pins autotune's [sk_pack_rest] seed variant (gh-ocannl-473): same
   loop structure as the hoisted-pack form, hoisted B~ panel and all, but the non-hoistable A
   operand gets an in-kernel packing Stage instead of being read in place — its [bm x bk] tile lands
   inside the Grid body and the renderer privatizes it to per-chunk block-scope storage
   ([parallel_grid_safe]'s privatization rule), recovering the A~ pack the hoisted-only shape
   forfeits while keeping the single outermost dispatch.

   On GPU backends the packing schedules apply without the [Tensorize] step (all-serial packing is
   legal everywhere; a unit-lane [Tile_mma] must not reach hardware simdgroup intrinsics) and the
   Grid case skips its transform; every printed boolean holds on every backend. *)

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

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_cpu = Sched.backend_is_cpu backend_name

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name

let has_parallel_construct src =
  String.is_substring src ~substring:"dispatch_apply"
  || String.is_substring src ~substring:"#pragma omp parallel for"

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* The single-child chain of loops from the top of each top-level nest. *)
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

let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner })
let n = 64
let bm, bk = (16, 16)

(* The composed pipeline (mirrors autotune's [cpu_mma_pack_sketch_schedule] with [bn = 0]). On GPU
   backends the trailing [Tensorize] is dropped: the packing is legal everywhere, while a unit-lane
   [Tile_mma] must not reach hardware simdgroup intrinsics. *)
let composed_schedule ~a ~b (opt : LL.optimized) : Sched.schedule =
  let paths = nest_paths opt.LL.llc in
  let i, j, k =
    match List.find_exn paths ~f:(fun p -> List.length p = 3) with
    | [ i; j; k ] -> (i, j, k)
    | _ -> assert false
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
  [ sp_i; sp_k ]
  (* i_o i_i j k_o k_i -> k_o i_o i_i j k_i (B~ packs at k_o, outside the row blocks). *)
  @ sink j [ k_o ]
  @ sink i_i [ k_o ] @ sink i_o [ k_o ]
  @ [ stage b [ k_i; j ]; stage a [ i_i; k_i ] ]
  @ if on_cpu then [ fst (Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width:1 ()) ] else []

let run_composed ~name ~a ~b (out : Tensor.t) =
  let comp = named name (Train.forward out) in
  let transform opt = Sched.apply (composed_schedule ~a ~b opt) opt in
  let ctx = Context.auto () in
  let ctx, routine =
    Context.compile ~lowered_transform:(fun o -> [ transform o ]) ctx comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  Context.get_values ctx out.Tensor.value

let run_serial ~name (out : Tensor.t) =
  let comp = named name (Train.forward out) in
  let ctx = Context.auto () in
  let ctx, routine =
    Context.compile ~lowered_transform:(fun opt -> [ opt ]) ctx comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  nonzero name (Context.get_values ctx out.Tensor.value)

let () =
  let mav = Array.init (n * n) ~f:(fun x -> Float.of_int (x % 13) *. 0.25) in
  let mbv = Array.init (n * n) ~f:(fun x -> Float.of_int (x % 17) -. 8.) in
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in

  (* === Standard layout: packed tiles feed the register-tiled micro-kernel === *)
  let%op mc0 = ma * mb in
  let want = run_serial ~name:"pmm_serial" mc0 in
  let%op mc1 = ma * mb in
  let got = run_composed ~name:"pmm_packed" ~a:ma.Tensor.value ~b:mb.Tensor.value mc1 in
  p_all2 "packed+tensorized matmul matches the serial twin bitwise" got want ~f:Float.equal;
  (let src = Generated.read "pmm_packed" in
   let has s = String.is_substring src ~substring:s in
   let count_sub sub = String.substr_index_all src ~may_overlap:false ~pattern:sub |> List.length in
   (* Two plain local tile arrays; on the C backends both operand pointers of the register tiling
      point into them (the accumulator streams from the real output). *)
   p "packed tiles feed the register-tiled micro-kernel"
     (count_sub "float tile_" = 2
     && (not (has "threadgroup float tile_"))
     && (not (has "__shared__"))
     &&
     if on_cpu then
       has "Tile_mma register tiling"
       && count_sub "__ = (tile_" = 2
       && not (has "Tile_mma register tiling: 0x0")
     else not (has "tmma_")));

  (* === Transposed B (the layout the register tiling declines whole-triple): the packing normalizes
     it — [tile_loops = [k_i; j]] packs B~ k-major, so [Tensorize] sees [tb = false] and the
     register tiling fires. === *)
  let mtav = Array.init (n * n) ~f:(fun x -> Float.of_int (x % 7) *. 0.5) in
  let mtbv = Array.init (n * n) ~f:(fun x -> Float.of_int (x % 11) -. 5.) in
  let mta = TDSL.ndarray mtav ~label:[ "mta" ] ~output_dims:[ n; n ] () in
  let mtb = TDSL.ndarray mtbv ~label:[ "mtb" ] ~output_dims:[ n; n ] () in
  let%op td0 = mta +* "ik;jk=>ij" mtb in
  let want_tb = run_serial ~name:"pmm_tb_serial" td0 in
  let%op td1 = mta +* "ik;jk=>ij" mtb in
  let got_tb = run_composed ~name:"pmm_tb_packed" ~a:mta.Tensor.value ~b:mtb.Tensor.value td1 in
  p_all2 "packed transposed-B matmul matches the serial twin bitwise" got_tb want_tb ~f:Float.equal;
  (let src = Generated.read "pmm_tb_packed" in
   let has s = String.is_substring src ~substring:s in
   p "packing normalizes the transposed-B layout for the register tiling"
     (if on_cpu then has "Tile_mma register tiling" else not (has "tmma_")));

  (* === Grid-blocked whole-triple Tile_mma pool-parallelizes (materialized operands only: the
     analysis reads the statement through its scalar fallback). CPU-only schedule; identity
     transform elsewhere. === *)
  let%op gc0 = ma * mb in
  let want_g = run_serial ~name:"pmm_grid_serial" gc0 in
  let%op gc1 = ma * mb in
  let grid_schedule (opt : LL.optimized) : Sched.schedule =
    let paths = nest_paths opt.LL.llc in
    let i, j, k =
      match List.find_exn paths ~f:(fun p -> List.length p = 3) with
      | [ i; j; k ] -> (i, j, k)
      | _ -> assert false
    in
    let ez, zsyms = Sched.expand_zero ~tn:gc1.Tensor.value in
    let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
    let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let tz, _ = Sched.tensorize ~i:i_i ~j ~k ~simd_width:n () in
    [ ez; sp_zi; Sched.Retype { axis = zj; ty = LL.Workgroup }; sp_i; tz ]
  in
  let transform opt = if on_cpu then Sched.apply (grid_schedule opt) opt else opt in
  let ctx = Context.auto () in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun o -> [ transform o ])
      ctx
      (named "pmm_grid_mma" (Train.forward gc1))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got_g = Context.get_values ctx gc1.Tensor.value in
  p_all2 "grid-blocked tensorized matmul matches the serial twin bitwise" got_g want_g
    ~f:Float.equal;
  (let src = Generated.read "pmm_grid_mma" in
   p "grid-blocked Tile_mma renders pool-parallel"
     (if on_cpu then
        has_parallel_construct src && String.is_substring src ~substring:"Tile_mma register tiling"
      else not (String.is_substring src ~substring:"tmma_")));

  (* === Parallel packed composition (the fully parallel packed GEMM, gh-ocannl-469): the row-block
     loop is Grid-typed and pool-parallelizes. The per-row-block A~ packing Stage sits inside the
     Grid body — its tile has all accesses in the body and the pack nest fully rewrites it before
     the micro-kernel reads it, so the renderer privatizes it to per-chunk block-scope storage
     ([C_syntax.parallel_grid_safe]'s privatization rule). The B~ panel packs at [k_o] outside and
     is read-only inside (function-scope; behind a pointer alias under the blocks extension, which
     cannot capture arrays). The whole-node [Zero_out] is no longer legal beside a
     hardware-annotated loop, so it expands into a nest whose row loop Grid-splits with the same
     [bm] geometry. CPU-only schedule; identity transform elsewhere. === *)
  let%op pp0 = ma * mb in
  let want_pp = run_serial ~name:"pmm_par_serial" pp0 in
  let%op pp1 = ma * mb in
  let par_packed_schedule (opt : LL.optimized) : Sched.schedule =
    let paths = nest_paths opt.LL.llc in
    let i, j, k =
      match List.find_exn paths ~f:(fun p -> List.length p = 3) with
      | [ i; j; k ] -> (i, j, k)
      | _ -> assert false
    in
    let ez, zsyms = Sched.expand_zero ~tn:pp1.Tensor.value in
    let zi = match zsyms with [ zi; _ ] -> zi | _ -> assert false in
    let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_i, i_o, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
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
    let tz, _lane = Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width:1 () in
    [ ez; sp_zi; sp_i; sp_k ] @ sink j [ k_o ] @ sink i_i [ k_o ] @ sink i_o [ k_o ]
    @ [ stage mb.Tensor.value [ k_i; j ]; stage ma.Tensor.value [ i_i; k_i ]; tz ]
  in
  let transform opt = if on_cpu then Sched.apply (par_packed_schedule opt) opt else opt in
  let ctx = Context.auto () in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun o -> [ transform o ])
      ctx
      (named "pmm_par_packed" (Train.forward pp1))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got_pp = Context.get_values ctx pp1.Tensor.value in
  p_all2 "parallel packed matmul matches the serial twin bitwise" got_pp want_pp ~f:Float.equal;
  let src = Generated.read "pmm_par_packed" in
  let has s = String.is_substring src ~substring:s in
  p "parallel packed composition renders pool-parallel with per-chunk tiles"
    (if on_cpu then
       has_parallel_construct src && has "Pool-backed Grid rendering"
       && has "Tile_mma register tiling"
       &&
       (* The A~ tile is declared per chunk, inside the parallel construct. *)
       let par_pos =
         Option.value_exn
           (Option.first_some
              (String.substr_index src ~pattern:"dispatch_apply")
              (String.substr_index src ~pattern:"#pragma omp parallel for"))
       in
       List.exists (String.substr_index_all src ~may_overlap:false ~pattern:"float tile_")
         ~f:(fun pos -> pos > par_pos)
     else not (has "tmma_"))

(* === Grid + hoisted-pack composition (autotune's [sk_grid && sk_hoist] seeds, [bn = 0]): Grid row
   blocks over the packed pipeline with only the hoistable B operand packed (link-time constant-pool
   panel) and A read in place — no in-kernel tile writes, so the Grid split pool-parallelizes with
   the Grid loop outermost. CPU-only schedule; identity transform elsewhere. === *)
let () =
  let gav = Array.init (n * n) ~f:(fun x -> Float.of_int (x % 19) *. 0.125) in
  let gbv = Array.init (n * n) ~f:(fun x -> Float.of_int (x % 23) -. 11.) in
  let ga = TDSL.ndarray gav ~label:[ "ga" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let gb = TDSL.ndarray gbv ~label:[ "gb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op hc0 = ga * gb in
  let want = run_serial ~name:"pmm_gridpack_serial" hc0 in
  let%op hc1 = ga * gb in
  let gridpack_schedule (opt : LL.optimized) : Sched.schedule =
    let paths = nest_paths opt.LL.llc in
    let i, j, k =
      match List.find_exn paths ~f:(fun p -> List.length p = 3) with
      | [ i; j; k ] -> (i, j, k)
      | _ -> assert false
    in
    let ez, zsyms = Sched.expand_zero ~tn:hc1.Tensor.value in
    let zi = match zsyms with [ zi; _zj ] -> zi | _ -> assert false in
    let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    let tz, _ = Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width:1 () in
    [ ez; sp_zi; sp_i; sp_k ] @ sink j [ k_o ] @ sink i_i [ k_o ]
    @ [
        Sched.Stage
          {
            source = gb.Tensor.value;
            tile_loops = [ k_i; j ];
            shared = false;
            cooperative = None;
            hoisted = true;
            swizzle = None;
            pad_stride = None;
            pipeline_depth = 1;
            tile_prec = None;
          };
        tz;
      ]
  in
  let transform opt = if on_cpu then Sched.apply (gridpack_schedule opt) opt else opt in
  let ctx = Context.auto () in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun o -> [ transform o ])
      ctx
      (named "pmm_gridpack" (Train.forward hc1))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx hc1.Tensor.value in
  p_all2 "grid + hoisted-pack matmul matches the serial twin bitwise" got want ~f:Float.equal;
  let src = Generated.read "pmm_gridpack" in
  let has s = String.is_substring src ~substring:s in
  p "grid + hoisted-pack renders pool-parallel with no in-kernel tile writes"
    (if on_cpu then
       has_parallel_construct src && has "Tile_mma register tiling" && has "packed_gb"
       && not (has "float tile_")
     else not (has "tmma_"))

(* === Mixed grid-outermost composition (autotune's [sk_pack_rest] seeds, gh-ocannl-473): the
   hoisted-pack shape with the non-hoistable A operand packed in-kernel — Grid row blocks stay
   outermost (one dispatch spanning the GEBP triple), the B~ panel packs at link time into the
   constant pool, and the per-row-block A~ tile packs inside the Grid body where the renderer
   privatizes it per chunk. A is a values-backed trainable param, so it is genuinely non-hoistable —
   the inference-GEMM case the seed targets (activations x constant weights). CPU-only schedule;
   identity transform elsewhere. === *)
let () =
  let gav = Array.init (n * n) ~f:(fun x -> (Float.of_int (x % 19) *. 0.125) -. 1.) in
  let gbv = Array.init (n * n) ~f:(fun x -> Float.of_int (x % 23) -. 11.) in
  let gb = TDSL.ndarray gbv ~label:[ "mgb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let xa0 = TDSL.param ~values:gav "xa0" ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op mx0 = xa0 * gb in
  let want = run_serial ~name:"pmm_mixed_serial" mx0 in
  let xa1 = TDSL.param ~values:gav "xa1" ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op mx1 = xa1 * gb in
  let mixed_schedule (opt : LL.optimized) : Sched.schedule =
    let paths = nest_paths opt.LL.llc in
    let i, j, k =
      match List.find_exn paths ~f:(fun p -> List.length p = 3) with
      | [ i; j; k ] -> (i, j, k)
      | _ -> assert false
    in
    let ez, zsyms = Sched.expand_zero ~tn:mx1.Tensor.value in
    let zi = match zsyms with [ zi; _zj ] -> zi | _ -> assert false in
    let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    let tz, _ = Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width:1 () in
    [ ez; sp_zi; sp_i; sp_k ] @ sink j [ k_o ] @ sink i_i [ k_o ]
    @ [
        Sched.Stage
          {
            source = gb.Tensor.value;
            tile_loops = [ k_i; j ];
            shared = false;
            cooperative = None;
            hoisted = true;
            swizzle = None;
            pad_stride = None;
            pipeline_depth = 1;
            tile_prec = None;
          };
        Sched.Stage
          {
            source = xa1.Tensor.value;
            tile_loops = [ i_i; k_i ];
            shared = false;
            cooperative = None;
            hoisted = false;
            swizzle = None;
            pad_stride = None;
            pipeline_depth = 1;
            tile_prec = None;
          };
        tz;
      ]
  in
  let transform opt = if on_cpu then Sched.apply (mixed_schedule opt) opt else opt in
  let ctx = Context.auto () in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun o -> [ transform o ])
      ctx
      (named "pmm_mixed" (Train.forward mx1))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx mx1.Tensor.value in
  p_all2 "mixed grid-outermost matmul matches the serial twin bitwise" got want ~f:Float.equal;
  let src = Generated.read "pmm_mixed" in
  let has s = String.is_substring src ~substring:s in
  p "mixed shape: hoisted B~ panel plus per-chunk in-kernel A~ tile"
    (if on_cpu then
       has_parallel_construct src && has "Tile_mma register tiling" && has "packed_mgb"
       &&
       (* Exactly one in-kernel tile (A~), declared per chunk inside the parallel construct. *)
       let tile_positions = String.substr_index_all src ~may_overlap:false ~pattern:"float tile_" in
       let par_pos =
         Option.value_exn
           (Option.first_some
              (String.substr_index src ~pattern:"dispatch_apply")
              (String.substr_index src ~pattern:"#pragma omp parallel for"))
       in
       List.length tile_positions = 1 && List.for_all tile_positions ~f:(fun pos -> pos > par_pos)
     else not (has "tmma_"))
