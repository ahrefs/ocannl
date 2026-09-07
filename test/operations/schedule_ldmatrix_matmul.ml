(* [ldmatrix] over b128-swizzled staged tiles (gh-ocannl-481 item 3, D2; the design resolution is
   docs/proposals/gh-ocannl-481-item3-ldmatrix.md).

   The staged + tensorized composition of schedule_mma_matmul.ml, with both operand tiles minted
   [Stage ~swizzle:(Some Swizzle_b128)]. On CUDA the inline-PTX [mma.sync] arms then build their
   fragment registers with [ldmatrix.sync.aligned.m8n8.xN[.trans].shared.b16] — one warp-cooperative
   instruction per fragment instead of 4 (resp. 2) per-lane gather chains — reading the 16-byte-unit
   XOR layout that keeps the 8 row addresses of each phase in distinct banks.

   What each leg pins:

   - Bitwise parity against the serial twin. The inputs are exact in bf16 (resp. e5m2) and every
   partial sum is exactly representable, so ANY correct accumulation order reproduces them exactly —
   which makes parity a test of the per-lane element mapping, not of tolerance. A mis-derived
   [ldmatrix] address or a wrong [.trans] gives a different product. - The emitted load form.
   [ldmatrix]'s four bf16 shapes are selected by the operands' storage orientation: [.trans]
   transposes each 8x8 tile on distribution, which is exactly the difference between an operand
   stored in its role's orientation and its transpose. - The rendering census:
   [Mma_intrinsics_ldmatrix], distinct from [Mma_intrinsics], so a sweep can tell which of the two
   load paths a timing measured.

   fp8's eligibility is one-sided per operand and the sides are opposite: [ldmatrix.b16] moves
   16-bit units, so it can build a register holding 4 fp8 bytes only when those bytes are contiguous
   — 4 consecutive [k] at fixed [m] for A (row-major storage, [ta = false]) and 4 consecutive [k] at
   fixed [n] for B (transposed storage, [tb = true]). The other side keeps its byte gathers, so the
   fp8 legs swizzle one operand and leave the other plain.

   The wmma path cannot participate at all: its fragments are opaque and there is no supported way
   to feed one from [ldmatrix] destination registers. The f16 leg pins that decline — the statement
   falls back to the swizzle-aware lane-0 micro-kernel and stays correct.

   Non-CUDA backends: Metal and HIP have no [ldmatrix] analogue, so their intrinsics decline the
   swizzled operands and the scalar fallback runs (still correct — it reads elementwise through the
   swizzled offsets); the C backends cannot express shared placement and reject cleanly. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments

let () = Utils.settings.output_debug_files_in_build_directory <- true

open Verdict.Claims

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

(* Intentional dialect identity: this test exists to pin CUDA inline-PTX [ldmatrix] instruction
   forms and the other dialects' explicit decline source, cross-checked with the MMA census. *)
let on_cuda = String.is_substring backend_name ~substring:"cuda"
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

let n = 32
let bm = 16
let simd_width = 32

(* The staged leg of schedule_mma_matmul.ml: loop order i_o(Grid) { k_o { i_i { j { k_i } } } },
   both operands staged through cooperative shared tiles anchored at k_o, then the i_i x j x k_i
   micro-kernel tensorized.

   A cooperative [Stage] mints the tile's axis order from the order of its [tile_loops], so the
   staged operand's orientation is the SCHEDULE's choice, not the source's: [tile_loops = [i_i;
   k_i]] gives the tile the A role's own orientation ([ta = false]) and [[k_i; i_i]] its transpose.
   That is what lets these legs cover all four [ldmatrix] shapes with one pair of source tensors —
   and it is also why a staged fp8 training GEMM does not automatically hit the transposed arms.

   Tile extents (the [Swizzle_b128] rule is a power-of-two count > 1 of whole 16-byte units per
   row): with [bk = 16] the 16-bit tiles are [16 x 16] / [16 x 32] / [32 x 16] — 32 or 64 bytes per
   row, 2 or 4 units. fp8 needs [bk = 32] both for [m16n8k32]'s reduction tile and to put 2 units in
   a row of single-byte elements. *)
let staged_schedule ~out ~src_a ~src_b ~swz_a ~swz_b ~bk ~ta ~tb (opt : LL.optimized) :
    Sched.schedule =
  let paths = nest_paths opt.LL.llc in
  let i, j, k =
    match List.find_exn paths ~f:(fun p -> List.length p = 3) with
    | [ i; j; k ] -> (i, j, k)
    | _ -> assert false
  in
  let ez, zsyms = Sched.expand_zero ~tn:out in
  let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
  let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
  let rz = Sched.Retype { axis = zj; ty = LL.Workgroup } in
  let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
  let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
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
        source = src_a;
        tile_loops = (if ta then [ k_i; i_i ] else [ i_i; k_i ]);
        shared = true;
        cooperative = Some simd_width;
        hoisted = false;
        swizzle = swz_a;
        pad_stride = None;
        pipeline_depth = 1;
        tile_prec = None;
      };
    Sched.Stage
      {
        source = src_b;
        tile_loops = (if tb then [ j; k_i ] else [ k_i; j ]);
        shared = true;
        cooperative = Some simd_width;
        hoisted = false;
        swizzle = swz_b;
        pad_stride = None;
        pipeline_depth = 1;
        tile_prec = None;
      };
    tz;
  ]

(* One leg: the serial twin, then the swizzled staged+tensorized form under census collection. On
   the C backends the shared placement is rejected before any of that. *)
let leg ?schedule_of ~tag ~build ~src_a ~src_b ~swz_a ~swz_b ~check ~acc_prec ?(bk = bm)
    ?(ta = false) ?(tb = false) () =
  let name = "ldm_" ^ tag in
  let parity_label =
    Printf.sprintf "%s staged ldmatrix parity (GPU) or clean rejection (CPU)" tag
  in
  let struct_label = Printf.sprintf "%s ldmatrix structure and census as expected" tag in
  (* Each [build ()] mints a fresh root over the shared leaves, so a root must be fully consumed
     before the next is built ([Tensor.consume_forward_code] rejects overlapping live roots). *)
  let with_target f =
    let t0 = build () in
    Tn.update_prec t0.Tensor.value acc_prec;
    let schedule_of =
      Option.value schedule_of
        ~default:(staged_schedule ~out:t0.Tensor.value ~src_a ~src_b ~swz_a ~swz_b ~bk ~ta ~tb)
    in
    let transform opt = Sched.apply (schedule_of opt) opt in
    f t0 transform
  in
  if not on_gpu then (
    with_target (fun t0 transform ->
        let ctx = Context.auto () in
        match
          try
            ignore
              (Context.compile
                 ~lowered_transform:(fun o -> [ transform o ])
                 ctx
                 (named (name ^ "_mma") (Train.forward t0))
                 Ir.Indexing.Empty
                : Context.t * Context.routine);
            None
          with Invalid_argument msg -> Some msg
        with
        | Some msg -> p parity_label (String.is_substring msg ~substring:"not supported")
        | None -> p parity_label false);
    p struct_label true)
  else begin
    let want =
      let serial = build () in
      Tn.update_prec serial.Tensor.value acc_prec;
      let ctx_s = Context.auto () in
      let ctx_s, routine_s =
        Context.compile
          ~lowered_transform:(fun opt -> [ opt ])
          ctx_s
          (named (name ^ "_serial") (Train.forward serial))
          Ir.Indexing.Empty
      in
      let ctx_s = Context.run ctx_s routine_s in
      Context.get_values ctx_s serial.Tensor.value
    in
    with_target @@ fun t0 transform ->
    let ctx, routine =
      Context.compile
        ~lowered_transform:(fun o -> [ transform o ])
        (Context.auto ())
        (named (name ^ "_mma") (Train.forward t0))
        Ir.Indexing.Empty
    in
    (* The census is a field of the compiled routine since gh-ocannl-626. *)
    let census = List.map routine.Context.mma.Ir.C_syntax.renderings ~f:snd in
    let ctx = Context.run ctx routine in
    let got = Context.get_values ctx t0.Tensor.value in
    (* The nonzero guard is not ceremony: a fragment mapping that reads outside the staged block
       would plausibly produce all zeros, and zeros compare equal to zeros. *)
    p parity_label
      (Array.exists want ~f:(fun x -> Float.(x <> 0.)) && Array.for_all2_exn got want ~f:Float.equal);
    let src = Generated.read (name ^ "_mma") in
    p struct_label (check ~src ~census)
  end

(* The bf16 inputs are multiples of 1/4 and 1/2 with 32-term sums bounded by 16, so every product is
   a multiple of 1/8 and every partial sum is exactly representable in bf16's 8 mantissa bits: the
   result is exact regardless of accumulation order or accumulator width. *)
let bf_a idcs = Float.of_int (((idcs.(0) * n) + idcs.(1)) % 3) *. 0.25
let bf_b idcs = (Float.of_int (((idcs.(0) * n) + idcs.(1)) % 5) -. 2.) *. 0.5

(* e5m2 has 2 mantissa bits: inputs from {-1,-0.5,0,0.5,1} and {-1.5..1.5 step 0.5} are exact, every
   product is a multiple of 0.25 bounded by 1.5, and a 32-term f32 sum of such products is exact. *)
let f8_a idcs = (Float.of_int (((idcs.(0) * n) + idcs.(1)) % 5) *. 0.5) -. 1.
let f8_b idcs = (Float.of_int (((idcs.(0) * n) + idcs.(1)) % 7) -. 3.) *. 0.5

let () =
  let mk ~l ~prec ~f = NTDSL.init ~l ~prec ~i:[ n ] ~o:[ n ] ~f () in
  let mab = mk ~l:"lab" ~prec:Ir.Ops.bfloat16 ~f:bf_a in
  let mbb = mk ~l:"lbb" ~prec:Ir.Ops.bfloat16 ~f:bf_b in
  let maf = mk ~l:"laf" ~prec:Ir.Ops.fp8 ~f:f8_a in
  let mbf = mk ~l:"lbf" ~prec:Ir.Ops.fp8 ~f:f8_b in
  let mah = mk ~l:"lah" ~prec:Ir.Ops.half ~f:bf_a in
  let mbh = mk ~l:"lbh" ~prec:Ir.Ops.half ~f:bf_b in
  let b128 = Some LL.Swizzle_b128 in
  let has src s = String.is_substring src ~substring:s in
  let ldm_census census =
    (not (List.is_empty census))
    && List.for_all census ~f:(Ir.C_syntax.equal_mma_rendering Ir.C_syntax.Mma_intrinsics_ldmatrix)
  in
  (* Every non-CUDA GPU backend declines the swizzled operands: the statement renders the lane-0
     scalar micro-kernel (correct through the swizzle-aware offsets) and the census says so. *)
  let declined ~src ~census =
    has src "== 0)"
    && (not (List.is_empty census))
    && List.for_all census ~f:(Ir.C_syntax.equal_mma_rendering Ir.C_syntax.Mma_scalar_fallback)
  in
  let bf16_check ~a_trans ~b_trans ~src ~census =
    if not on_cuda then declined ~src ~census
    else
      let a_form =
        if a_trans then "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
        else "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      in
      let b_form =
        if b_trans then "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16"
        else "ldmatrix.sync.aligned.m8n8.x2.shared.b16"
      in
      has src a_form && has src b_form
      && has src "mma.sync.aligned.m16n8k16"
      && has src "(mma-bf16) ldmatrix a,b"
      (* No gather chain survives for either operand. *)
      && (not (has src "__bfloat16_as_ushort"))
      && ldm_census census
  in
  (* A staged in the A role's own orientation loads with no [.trans]; B staged in the B role's own
     orientation (k x n) needs [.trans], because its fragment is column-major. Staging an operand
     transposed flips its flag. *)
  leg ~tag:"bf_ab" ~acc_prec:Ir.Ops.bfloat16 ~src_a:mab.Tensor.value ~src_b:mbb.Tensor.value
    ~swz_a:b128 ~swz_b:b128
    ~build:(fun () ->
      let%op t = mab * mbb in
      t)
    ~check:(bf16_check ~a_trans:false ~b_trans:true)
    ();
  leg ~tag:"bf_ta" ~acc_prec:Ir.Ops.bfloat16 ~src_a:mab.Tensor.value ~src_b:mbb.Tensor.value
    ~swz_a:b128 ~swz_b:b128 ~ta:true
    ~build:(fun () ->
      let%op t = mab * mbb in
      t)
    ~check:(bf16_check ~a_trans:true ~b_trans:true)
    ();
  leg ~tag:"bf_tb" ~acc_prec:Ir.Ops.bfloat16 ~src_a:mab.Tensor.value ~src_b:mbb.Tensor.value
    ~swz_a:b128 ~swz_b:b128 ~tb:true
    ~build:(fun () ->
      let%op t = mab * mbb in
      t)
    ~check:(bf16_check ~a_trans:false ~b_trans:false)
    ();
  (* The swizzled twin as AUTOTUNE builds it (gh-ocannl-481 item 3, D3), not as this file hand-wires
     it: [Autotune.sketch_schedule] with [sk_swizzle] set is the pipeline the tuner will actually
     time on CUDA, pads and companion geometry included. Its staged tiles are [16 x 32] and [32 x
     32] bf16 — 64-byte rows, 4 whole 16-byte units. *)
  leg ~tag:"bf_sketch" ~acc_prec:Ir.Ops.bfloat16 ~src_a:mab.Tensor.value ~src_b:mbb.Tensor.value
    ~swz_a:b128 ~swz_b:b128
    ~build:(fun () ->
      let%op t = mab * mbb in
      t)
    ~schedule_of:(fun opt ->
      Autotune.sketch_schedule
        ~p:
          {
            Autotune.sk_gpu = true;
            sk_mma = true;
            sk_simd = simd_width;
            sk_bm = bm;
            sk_bn = n;
            sk_bk = n;
            sk_tm = 0;
            sk_tn = 0;
            sk_hoist = false;
            sk_grid = false;
            sk_pack_rest = false;
            sk_conv = false;
            sk_epilogue = false;
            sk_swizzle = Some LL.Swizzle_b128;
            sk_depth = 1;
            sk_batch_grid = false;
            sk_pack_prec = None;
            sk_tile = None;
          }
        opt)
    ~check:(bf16_check ~a_trans:false ~b_trans:true)
    ();
  (* fp8, A-side only: staged in the A role's orientation, A's 4 fp8 bytes per register are
     contiguous and come in through [ldmatrix], while B keeps its strided byte gathers from a plain
     shared tile. [bk = 32] is [m16n8k32]'s reduction tile, and also what puts two whole 16-byte
     units in a row of single-byte elements. *)
  leg ~tag:"f8_a" ~acc_prec:Ir.Ops.single ~src_a:maf.Tensor.value ~src_b:mbf.Tensor.value
    ~swz_a:b128 ~swz_b:None ~bk:32
    ~build:(fun () ->
      let%op t = maf * mbf in
      t)
    ~check:(fun ~src ~census ->
      if not on_cuda then declined ~src ~census
      else
        has src "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
        && has src "mma.sync.aligned.m16n8k32"
        && has src "(mma-fp8) e5m2 ldmatrix a"
        && (not (has src "ldmatrix.sync.aligned.m8n8.x2"))
        && ldm_census census)
    ();
  (* fp8, B-side only: staging B transposed makes ITS 4 bytes per register contiguous instead. *)
  leg ~tag:"f8_b" ~acc_prec:Ir.Ops.single ~src_a:maf.Tensor.value ~src_b:mbf.Tensor.value
    ~swz_a:None ~swz_b:b128 ~bk:32 ~tb:true
    ~build:(fun () ->
      let%op t = maf * mbf in
      t)
    ~check:(fun ~src ~census ->
      if not on_cuda then declined ~src ~census
      else
        has src "ldmatrix.sync.aligned.m8n8.x2.shared.b16"
        && has src "mma.sync.aligned.m16n8k32"
        && has src "(mma-fp8) e5m2 ldmatrix b"
        && (not (has src "ldmatrix.sync.aligned.m8n8.x4"))
        && ldm_census census)
    ();
  (* The other fp8 side of each operand has no 16-bit-granularity load form: a b128-swizzled B
     staged in the B role's own orientation needs 4 bytes strided by the tile's leading dimension,
     which no [ldmatrix.b16] shape produces. The arm declines the whole call rather than reading the
     swizzled tile as if it were row-major. *)
  leg ~tag:"f8_b_decline" ~acc_prec:Ir.Ops.single ~src_a:maf.Tensor.value ~src_b:mbf.Tensor.value
    ~swz_a:None ~swz_b:b128 ~bk:32
    ~build:(fun () ->
      let%op t = maf * mbf in
      t)
    ~check:(fun ~src ~census -> (not (has src "ldmatrix")) && declined ~src ~census)
    ();
  (* The wmma combinations cannot consume [ldmatrix] destination registers, so a swizzled staged f16
     leg declines the template path AND the accumulator-residency scope built on it, landing on the
     swizzle-aware lane-0 micro-kernel. Correct, and visibly not tensorized. *)
  leg ~tag:"f16_decline" ~acc_prec:Ir.Ops.single ~src_a:mah.Tensor.value ~src_b:mbh.Tensor.value
    ~swz_a:b128 ~swz_b:b128
    ~build:(fun () ->
      let%op t = mah * mbh in
      t)
    ~check:(fun ~src ~census ->
      (not (has src "ldmatrix")) && (not (has src "wmma::mma_sync")) && declined ~src ~census)
    ()
