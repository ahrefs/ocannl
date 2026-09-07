(* Swizzled shared-memory staging ([Stage ~swizzle], the bank-conflict follow-up of
   docs/proposals/tensorize-mma.md): the S2 shared-memory matmul schedule with both operand tiles
   stored XOR-swizzled, executed against the serial twin.

   [mc = ma * mb] (32x32 times 32x32) with the same schedule as schedule_smem_matmul.ml — Grid x
   Workgroup splits, [Stage ~shared ~swizzle] of both operands at 8x8 tiles, Privatize — except the
   tiles are marked in [optimized.swizzled]: codegen remaps each tile access [P*8 + col] to [P*8 +
   (col ^ (P & 7))], a per-row bijection of the minor axis, so the values are unchanged while
   same-column reads (the classic strided access of the staged A tile) spread across shared-memory
   banks. GPU backends execute and must match the serial twin; the C backends cannot express shared
   placement and must reject cleanly (same pinning as the SMEM matmul test).

   The staged+tensorized composition (lane-aware Stage feeding Tensorize) with swizzled tiles pins
   the decline path: the tile-MMA intrinsic and fragment renderings assume row-major pointer+stride
   operands, so swizzled operands must fall back to the lane-0 scalar micro-kernel — which reads
   elementwise through the swizzle-aware offsets and stays correct.

   The same schedule runs again under [Swizzle_b128] (gh-ocannl-481 item 3, D1): the coarser
   16-byte-unit XOR — the layout [ldmatrix] wants — is still a per-row bijection read by the same
   scalar accesses, so parity holds identically and the two flavors differ only in the emitted
   offset arithmetic.

   A third run adds [Stage ~pad_stride] (gh-ocannl-481 item 4) with no swizzle at all: the tiles'
   minor dim rounds up, which changes the row STRIDE while the iterated index space stays the same.
   That is the whole mechanism — bank conflicts and every layout rule stated on the leading
   dimension depend on the stride, not on the data — so parity is unchanged and the evidence is the
   emitted arithmetic.

   The error legs pin [Schedule.Stage]'s layout validation on every backend: swizzle requires shared
   staging and a tile with at least two axes; the element flavor needs a power-of-two minor tile
   dim, the b128 flavor a power-of-two count > 1 of whole 16-byte units — two rules that neither
   imply nor are implied by each other; [pad_stride] needs a stride to pad and a multiple above 1.
   The last leg is where items 3 and 4 touch (D5): the b128 rule is checked on the PADDED dims, so a
   tile that misses it is lifted over it by [pad_stride] rather than being declined. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
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

let approx a b = Float.(abs (a - b) < 1e-3)
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let skipped = Verdict.skipped ~backend:backend_name

(* Intentional dialect identity: these branches pin literal MSL versus CUDA/HIP shared-memory,
   barrier, and intrinsic spellings around swizzled tiles; schedule availability uses family and
   hardware-limit queries. *)
let on_metal = String.is_substring backend_name ~substring:"metal"
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

let accum_syms opt =
  let paths = nest_paths opt.LL.llc in
  match List.find_exn paths ~f:(fun p -> List.length p = 3) with
  | [ i; j; k ] -> (i, j, k)
  | _ -> assert false

let n = 32
let bm, bn, bk = (8, 8, 8)
let simd_width = 32

let () =
  let mav = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 13) *. 0.25) in
  let mbv = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 17) -. 8.) in
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in

  (* --- Serial twin --- *)
  let%op mc0 = ma * mb in
  let serial_comp = named "swz_mm_serial" (Train.forward mc0) in
  let ctx_s = Context.auto () in
  let ctx_s, routine_s =
    Context.compile ~lowered_transform:(fun opt -> [ opt ]) ctx_s serial_comp Ir.Indexing.Empty
  in
  let ctx_s = Context.run ctx_s routine_s in
  let got_serial = nonzero "swz_serial" (Context.get_values ctx_s mc0.Tensor.value) in

  (* --- The swizzled SMEM schedule (element-granularity XOR) --- *)
  let%op mc1 = ma * mb in
  let smem_schedule ?(swizzle = Some LL.Swizzle_elem) ?pad_stride (mc : Tensor.t)
      (opt : LL.optimized) : Sched.schedule =
    let i, j, k = accum_syms opt in
    let ez, zsyms = Sched.expand_zero ~tn:mc.Tensor.value in
    let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
    let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_zj, _, _ = Sched.split ~axis:zj ~factor:bn ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_j, _, j_i = Sched.split ~axis:j ~factor:bn ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    [
      ez;
      sp_zi;
      sp_zj;
      sp_i;
      sp_j;
      sp_k;
      Sched.Stage
        {
          source = ma.Tensor.value;
          tile_loops = [ i_i; k_i ];
          shared = true;
          cooperative = None;
          hoisted = false;
          swizzle;
          pad_stride;
          pipeline_depth = 1;
          tile_prec = None;
        };
      Sched.Stage
        {
          source = mb.Tensor.value;
          tile_loops = [ k_i; j_i ];
          shared = true;
          cooperative = None;
          hoisted = false;
          swizzle;
          pad_stride;
          pipeline_depth = 1;
          tile_prec = None;
        };
      Sched.Privatize { target = mc.Tensor.value; over = k_o };
    ]
  in
  let smem_comp = named "mm_swizzled_smem" (Train.forward mc1) in
  let transform opt = Sched.apply (smem_schedule mc1 opt) opt in
  let ctx_a = Context.auto () in
  if on_gpu then (
    let ctx_a, routine_a =
      Context.compile
        ~lowered_transform:(fun o -> [ transform o ])
        ctx_a smem_comp Ir.Indexing.Empty
    in
    let ctx_a = Context.run ctx_a routine_a in
    let got_smem = Context.get_values ctx_a mc1.Tensor.value in
    p_all2 "swizzled SMEM matmul parity (GPU) or clean rejection (CPU)" got_smem got_serial
      ~f:approx;
    let src = Generated.read "mm_swizzled_smem" in
    let has sub = String.is_substring src ~substring:sub in
    let count_sub sub =
      String.substr_index_all src ~may_overlap:false ~pattern:sub |> List.length
    in
    let shared_ok =
      if on_metal then count_sub "threadgroup float tile_" = 2 && has "threadgroup_barrier"
      else count_sub "__shared__ float tile_" = 2 && has "__syncthreads()"
    in
    (* Each tile is written by its cooperative load and read by the micro-kernel, all through the
       swizzled offset [P*8 + (col ^ (P & 7))]: at least 4 masked-XOR sites. *)
    p "tile accesses XOR-swizzled (GPU) or rejected (CPU)"
      (shared_ok && count_sub " & 7)" >= 4 && count_sub " ^ " >= 4))
  else (
    (match
       try
         ignore
           (Context.compile
              ~lowered_transform:(fun o -> [ transform o ])
              ctx_a smem_comp Ir.Indexing.Empty
             : Context.t * Context.routine);
         None
       with Invalid_argument msg -> Some msg
     with
    | Some msg ->
        p "swizzled SMEM matmul parity (GPU) or clean rejection (CPU)"
          (String.is_substring msg ~substring:"not supported")
    | None -> p "swizzled SMEM matmul parity (GPU) or clean rejection (CPU)" false);
    skipped "tile accesses XOR-swizzled (GPU) or rejected (CPU)");

  (* --- The same schedule under the 16-byte-unit flavor ([Swizzle_b128], gh-ocannl-481 item 3 D1):
     the XOR permutes whole 16-byte units of the row instead of single elements, leaving the offset
     within a unit alone. On these f32 8x8 tiles a row is 32 bytes = 2 units of 4 elements, so the
     offset renders as [P*8 + ((((col >> 2) ^ (P & 1)) << 2) + (col & 3))]. Still a per-row
     bijection, so the values are unchanged — the scalar reads below go through it exactly like the
     element flavor, which is what keeps the b128 layout usable by every rendering that does not
     know about [ldmatrix]. --- *)
  let%op mc1b = ma * mb in
  let smem_b128_comp = named "mm_swz128_smem" (Train.forward mc1b) in
  let transform_b128 opt =
    Sched.apply (smem_schedule ~swizzle:(Some LL.Swizzle_b128) mc1b opt) opt
  in
  let ctx_b = Context.auto () in
  if on_gpu then (
    let ctx_b, routine_b =
      Context.compile
        ~lowered_transform:(fun o -> [ transform_b128 o ])
        ctx_b smem_b128_comp Ir.Indexing.Empty
    in
    let ctx_b = Context.run ctx_b routine_b in
    let got = Context.get_values ctx_b mc1b.Tensor.value in
    p_all2 "b128-swizzled SMEM matmul parity (GPU) or clean rejection (CPU)" got got_serial
      ~f:approx;
    let src = Generated.read "mm_swz128_smem" in
    let count_sub sub =
      String.substr_index_all src ~may_overlap:false ~pattern:sub |> List.length
    in
    (* Four access sites (two cooperative loads, two micro-kernel reads), each rendering the unit
       index, its XOR mask, the shift back and the within-unit remainder. The element flavor's [& 7]
       mask must NOT appear: that would be the wrong granularity. *)
    p "tile accesses b128-swizzled (GPU) or rejected (CPU)"
      (count_sub " >> 2)" >= 4
      && count_sub " & 1)" >= 4
      && count_sub " << 2)" >= 4
      && count_sub " & 3)" >= 4
      && count_sub " & 7)" = 0))
  else (
    (match
       try
         ignore
           (Context.compile
              ~lowered_transform:(fun o -> [ transform_b128 o ])
              ctx_b smem_b128_comp Ir.Indexing.Empty
             : Context.t * Context.routine);
         None
       with Invalid_argument msg -> Some msg
     with
    | Some msg ->
        p "b128-swizzled SMEM matmul parity (GPU) or clean rejection (CPU)"
          (String.is_substring msg ~substring:"not supported")
    | None -> p "b128-swizzled SMEM matmul parity (GPU) or clean rejection (CPU)" false);
    skipped "tile accesses b128-swizzled (GPU) or rejected (CPU)");

  (* --- [Stage ~pad_stride] (gh-ocannl-481 item 4): the same schedule with the tiles' minor dim
     rounded up from 8 to a multiple of 12. Nothing about the DATA changes — the load nest and the
     micro-kernel iterate the same 8x8 index space — only the stride between rows, which is what a
     bank-conflict fix and every layout rule stated on the leading dimension actually depend on.
     Parity is therefore the same as the unpadded schedule's, and the emitted evidence is the
     stride: [* 12 +] row arithmetic over 96-element shared arrays instead of [* 8 +] over 64.
     --- *)
  let%op mc1p = ma * mb in
  let smem_pad_comp = named "mm_padstride_smem" (Train.forward mc1p) in
  let transform_pad opt = Sched.apply (smem_schedule ~swizzle:None ~pad_stride:12 mc1p opt) opt in
  let ctx_p = Context.auto () in
  if on_gpu then (
    let ctx_p, routine_p =
      Context.compile
        ~lowered_transform:(fun o -> [ transform_pad o ])
        ctx_p smem_pad_comp Ir.Indexing.Empty
    in
    let ctx_p = Context.run ctx_p routine_p in
    let got = Context.get_values ctx_p mc1p.Tensor.value in
    p_all2 "pad_stride SMEM matmul parity (GPU) or clean rejection (CPU)" got got_serial ~f:approx;
    let src = Generated.read "mm_padstride_smem" in
    let has sub = String.is_substring src ~substring:sub in
    let count_sub sub =
      String.substr_index_all src ~may_overlap:false ~pattern:sub |> List.length
    in
    let decl = if on_metal then "threadgroup float tile_" else "__shared__ float tile_" in
    p "pad_stride widens the tile stride (GPU) or rejected (CPU)"
      (count_sub decl = 2
      && count_sub "[96]" = 2
      && count_sub " * 12 + " >= 4
      && (not (has " * 8 + "))
      (* Padding alone is not a swizzle: the offsets stay plain row-major. *)
      && not (has " ^ ")))
  else (
    (match
       try
         ignore
           (Context.compile
              ~lowered_transform:(fun o -> [ transform_pad o ])
              ctx_p smem_pad_comp Ir.Indexing.Empty
             : Context.t * Context.routine);
         None
       with Invalid_argument msg -> Some msg
     with
    | Some msg ->
        p "pad_stride SMEM matmul parity (GPU) or clean rejection (CPU)"
          (String.is_substring msg ~substring:"not supported")
    | None -> p "pad_stride SMEM matmul parity (GPU) or clean rejection (CPU)" false);
    skipped "pad_stride widens the tile stride (GPU) or rejected (CPU)");

  (* --- Swizzled tiles feeding Tensorize: the intrinsic/fragment renderings assume row-major
     pointer+stride operands, so they must decline and the lane-0 scalar fallback must run — and
     stay correct, reading elementwise through the swizzled offsets. Same pipeline as the staged leg
     of schedule_mma_matmul.ml, with [swizzle = Some LL.Swizzle_elem] on both stages ([bm = 16]
     keeps the block extents intrinsic-sized, so a surviving intrinsic would fire — its absence
     below is the decline, not a shape accident). --- *)
  let%op mc2 = ma * mb in
  let bt = 16 in
  let staged_schedule (opt : LL.optimized) : Sched.schedule =
    let i, j, k = accum_syms opt in
    let ez, zsyms = Sched.expand_zero ~tn:mc2.Tensor.value in
    let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
    let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bt ~outer:LL.Grid ~inner:LL.Serial in
    let rz = Sched.Retype { axis = zj; ty = LL.Workgroup } in
    let sp_i, _, i_i = Sched.split ~axis:i ~factor:bt ~outer:LL.Grid ~inner:LL.Serial in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bt ~outer:LL.Serial ~inner:LL.Serial in
    let tz, _lane = Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width () in
    [
      ez;
      sp_zi;
      rz;
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
          swizzle = Some LL.Swizzle_elem;
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
          swizzle = Some LL.Swizzle_elem;
          pad_stride = None;
          pipeline_depth = 1;
          tile_prec = None;
        };
      tz;
    ]
  in
  let staged_comp = named "mm_swizzled_mma" (Train.forward mc2) in
  let staged_transform opt = Sched.apply (staged_schedule opt) opt in
  let ctx_c = Context.auto () in
  if on_gpu then (
    let ctx_c, routine_c =
      Context.compile
        ~lowered_transform:(fun o -> [ staged_transform o ])
        ctx_c staged_comp Ir.Indexing.Empty
    in
    let ctx_c = Context.run ctx_c routine_c in
    let got_staged = Context.get_values ctx_c mc2.Tensor.value in
    p_all2 "swizzled staged+tensorized parity (GPU) or clean rejection (CPU)" got_staged got_serial
      ~f:approx;
    let src = Generated.read "mm_swizzled_mma" in
    let has sub = String.is_substring src ~substring:sub in
    let count_sub sub =
      String.substr_index_all src ~may_overlap:false ~pattern:sub |> List.length
    in
    let shared_ok =
      if on_metal then count_sub "threadgroup float tile_" = 2
      else count_sub "__shared__ float tile_" = 2
    in
    let no_intrinsics =
      if on_metal then (not (has "simdgroup_multiply_accumulate")) && not (has "simdgroup_load")
      else not (has "wmma")
    in
    (* The ma tile is [16 x 16] (mask 15), the mb tile [16 x 32] (mask 31); each is written by its
       cooperative load and read by the fallback micro-kernel: >= 4 XOR sites total. *)
    p "swizzled operands decline the MMA intrinsics to the lane-0 fallback"
      (shared_ok && no_intrinsics && has "== 0)"
      && count_sub " ^ " >= 4
      && has " & 15)" && has " & 31)"))
  else (
    (match
       try
         ignore
           (Context.compile
              ~lowered_transform:(fun o -> [ staged_transform o ])
              ctx_c staged_comp Ir.Indexing.Empty
             : Context.t * Context.routine);
         None
       with Invalid_argument msg -> Some msg
     with
    | Some msg ->
        p "swizzled staged+tensorized parity (GPU) or clean rejection (CPU)"
          (String.is_substring msg ~substring:"not supported")
    | None -> p "swizzled staged+tensorized parity (GPU) or clean rejection (CPU)" false);
    skipped "swizzled operands decline the MMA intrinsics to the lane-0 fallback");

  (* --- Validation errors, uniform on every backend (raised by [Schedule.apply] before any
     backend-specific compilation) --- *)
  let expect_error name ~substring (mc : Tensor.t) schedule_of =
    let comp = named name (Train.forward mc) in
    let transform opt = Sched.apply (schedule_of opt) opt in
    let ctx = Context.auto () in
    match
      try
        ignore
          (Context.compile ~lowered_transform:(fun o -> [ transform o ]) ctx comp Ir.Indexing.Empty
            : Context.t * Context.routine);
        None
      with Invalid_argument msg -> Some msg
    with
    | Some msg -> p name (String.is_substring msg ~substring)
    | None -> p name false
  in
  let%op mc3 = ma * mb in
  expect_error "swizzle requires shared staging" ~substring:"requires shared" mc3 (fun opt ->
      let i, _j, k = accum_syms opt in
      let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Serial ~inner:LL.Serial in
      let sp_k, _, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
      [
        sp_i;
        sp_k;
        Sched.Stage
          {
            source = ma.Tensor.value;
            tile_loops = [ i_i; k_i ];
            shared = false;
            cooperative = None;
            hoisted = false;
            swizzle = Some LL.Swizzle_elem;
            pad_stride = None;
            pipeline_depth = 1;
            tile_prec = None;
          };
      ]);
  let%op mc4 = ma * mb in
  expect_error "swizzle requires at least two tile axes" ~substring:"at least two axes" mc4
    (fun opt ->
      let _i, _j, k = accum_syms opt in
      let sp_k, _, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
      [
        sp_k;
        Sched.Stage
          {
            source = ma.Tensor.value;
            tile_loops = [ k_i ];
            shared = true;
            cooperative = None;
            hoisted = false;
            swizzle = Some LL.Swizzle_elem;
            pad_stride = None;
            pipeline_depth = 1;
            tile_prec = None;
          };
      ]);
  (* A 24-wide matmul: splitting k by 12 gives a [8 x 12] tile whose minor dim is not a power of two
     (every divisor of 32 is, so the main tensors cannot produce this case). *)
  let n2 = 24 in
  let ma2v = Array.init (n2 * n2) ~f:(fun i -> Float.of_int (i % 7) *. 0.5) in
  let mb2v = Array.init (n2 * n2) ~f:(fun i -> Float.of_int (i % 11) -. 5.) in
  let ma2 = TDSL.ndarray ma2v ~label:[ "ma2" ] ~input_dims:[ n2 ] ~output_dims:[ n2 ] () in
  let mb2 = TDSL.ndarray mb2v ~label:[ "mb2" ] ~input_dims:[ n2 ] ~output_dims:[ n2 ] () in
  let%op mc5 = ma2 * mb2 in
  expect_error "swizzle requires a power-of-two minor tile dim" ~substring:"power-of-two" mc5
    (fun opt ->
      let i, _j, k = accum_syms opt in
      let sp_i, _, i_i = Sched.split ~axis:i ~factor:8 ~outer:LL.Serial ~inner:LL.Serial in
      let sp_k, _, k_i = Sched.split ~axis:k ~factor:12 ~outer:LL.Serial ~inner:LL.Serial in
      [
        sp_i;
        sp_k;
        Sched.Stage
          {
            source = ma2.Tensor.value;
            tile_loops = [ i_i; k_i ];
            shared = true;
            cooperative = None;
            hoisted = false;
            swizzle = Some LL.Swizzle_elem;
            pad_stride = None;
            pipeline_depth = 1;
            tile_prec = None;
          };
      ]);
  (* The b128 flavor's minor-extent rule is a different one, neither implying nor implied by the
     element flavor's (gh-ocannl-481 item 3, D1): it counts 16-byte units, not elements. A 4-wide
     f32 tile row is a legal element-flavor extent (power of two) but only 16 bytes = 1 unit, so
     there is nothing to permute; a 12-wide f32 row is 3 units — a legal unit count in size but not
     a power of two, so its XOR would leave the row. *)
  let%op mc6 = ma * mb in
  expect_error "Swizzle_b128 requires a power-of-two unit count > 1"
    ~substring:"power-of-two count > 1 of 16-byte units" mc6 (fun opt ->
      let i, _j, k = accum_syms opt in
      let sp_i, _, i_i = Sched.split ~axis:i ~factor:8 ~outer:LL.Serial ~inner:LL.Serial in
      let sp_k, _, k_i = Sched.split ~axis:k ~factor:4 ~outer:LL.Serial ~inner:LL.Serial in
      [
        sp_i;
        sp_k;
        Sched.Stage
          {
            source = ma.Tensor.value;
            tile_loops = [ i_i; k_i ];
            shared = true;
            cooperative = None;
            hoisted = false;
            swizzle = Some LL.Swizzle_b128;
            pad_stride = None;
            pipeline_depth = 1;
            tile_prec = None;
          };
      ]);
  let%op mc7 = ma2 * mb2 in
  expect_error "Swizzle_b128 requires a 16-byte-multiple minor tile dim"
    ~substring:"multiple of 16 bytes" mc7 (fun opt ->
      let i, _j, k = accum_syms opt in
      let sp_i, _, i_i = Sched.split ~axis:i ~factor:8 ~outer:LL.Serial ~inner:LL.Serial in
      (* A 3-wide f32 row is 12 bytes: not a whole number of 16-byte units. *)
      let sp_k, _, k_i = Sched.split ~axis:k ~factor:3 ~outer:LL.Serial ~inner:LL.Serial in
      [
        sp_i;
        sp_k;
        Sched.Stage
          {
            source = ma2.Tensor.value;
            tile_loops = [ i_i; k_i ];
            shared = true;
            cooperative = None;
            hoisted = false;
            swizzle = Some LL.Swizzle_b128;
            pad_stride = None;
            pipeline_depth = 1;
            tile_prec = None;
          };
      ]);
  (* pad_stride's own validation: it pads against the row stride, so a one-axis tile has no stride
     to pad, and a multiple of 1 is a no-op request rather than an intent. *)
  let%op mc8 = ma * mb in
  expect_error "pad_stride requires at least two tile axes" ~substring:"at least two axes" mc8
    (fun opt ->
      let _i, _j, k = accum_syms opt in
      let sp_k, _, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
      [
        sp_k;
        Sched.Stage
          {
            source = ma.Tensor.value;
            tile_loops = [ k_i ];
            shared = true;
            cooperative = None;
            hoisted = false;
            swizzle = None;
            pad_stride = Some 16;
            pipeline_depth = 1;
            tile_prec = None;
          };
      ]);
  let%op mc9 = ma * mb in
  expect_error "pad_stride must exceed 1" ~substring:"pad_stride must be > 1" mc9 (fun opt ->
      let i, _j, k = accum_syms opt in
      let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Serial ~inner:LL.Serial in
      let sp_k, _, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
      [
        sp_i;
        sp_k;
        Sched.Stage
          {
            source = ma.Tensor.value;
            tile_loops = [ i_i; k_i ];
            shared = true;
            cooperative = None;
            hoisted = false;
            swizzle = None;
            pad_stride = Some 1;
            pipeline_depth = 1;
            tile_prec = None;
          };
      ]);
  (* D5, the one place items 3 and 4 touch: [Swizzle_b128]'s extent rule is checked on the PADDED
     dims, so [pad_stride] is the knob for a tile that misses it. The 12-wide f32 row rejected two
     legs above (48 bytes = 3 units) becomes a 16-wide one (64 bytes = 4 units) and passes — the
     coverage half of item 4, converting a layout decline into an emission rather than accepting a
     bank-conflicted tile. Only the b128 rule is at stake here, so the check is that the schedule
     does NOT fail for that reason; the C backends still reject the shared placement itself. *)
  let expect_not_error name ~substring (mc : Tensor.t) schedule_of =
    let comp = named name (Train.forward mc) in
    let transform opt = Sched.apply (schedule_of opt) opt in
    let ctx = Context.auto () in
    match
      try
        ignore
          (Context.compile ~lowered_transform:(fun o -> [ transform o ]) ctx comp Ir.Indexing.Empty
            : Context.t * Context.routine);
        None
      with Invalid_argument msg -> Some msg
    with
    | Some msg -> p name (not (String.is_substring msg ~substring))
    | None -> p name true
  in
  let%op mc10 = ma2 * mb2 in
  expect_not_error "pad_stride lifts a b128 tile over its unit-count rule"
    ~substring:"16-byte units" mc10 (fun opt ->
      let i, _j, k = accum_syms opt in
      let sp_i, _, i_i = Sched.split ~axis:i ~factor:8 ~outer:LL.Serial ~inner:LL.Serial in
      let sp_k, _, k_i = Sched.split ~axis:k ~factor:12 ~outer:LL.Serial ~inner:LL.Serial in
      [
        sp_i;
        sp_k;
        Sched.Stage
          {
            source = ma2.Tensor.value;
            tile_loops = [ i_i; k_i ];
            shared = true;
            cooperative = None;
            hoisted = false;
            swizzle = Some LL.Swizzle_b128;
            pad_stride = Some 16;
            pipeline_depth = 1;
            tile_prec = None;
          };
      ])
