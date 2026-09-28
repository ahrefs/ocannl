(* Schedule IR, Phase S2 (docs/proposals/schedule-ir-optops.md §5): the hand-written shared-memory
   matmul schedule, built from Split/Retype/Stage optops and executed against the serial twin.

   [mc = ma * mb] (32x32 times 32x32) lowers to a zeroing nest (2 loops) plus the naive triple loop.
   The schedule: split the zeroing nest's loops into Grid x Workgroup pairs (whole-node Zero_out is
   not distributed across threads, so the zeroing writes must cover the same hardware dimensions as
   the accumulation); split i and j of the accumulation by 8 into Grid x Workgroup, split k by 8
   into Serial x Serial; then [Stage ma [i_i; k_i] ~shared] and [Stage mb [k_i; j_i] ~shared]. Stage
   inserts cooperative loads (reusing the Workgroup axis of each tile, iterating the serial k_i
   under a fresh symbol, restricting the redundant workgroup axis to thread 0) and barriers inside
   k_o; the 8x8 tiles become [threadgroup] / [__shared__] declarations via
   [optimized.workgroup_shared]. Tile sizes divide the extents, so Split's remainder guards and
   Stage's edge guards all fold away -- required: surviving whole-body guards would place the
   barriers under divergent control flow ([validate_parallel] rejects).

   On GPU backends (Metal locally, CUDA in CI) the kernel executes with a 4x4 grid of 8x8 workgroups
   and must match the serial twin. The C backends cannot express shared placement; they must reject
   cleanly -- that rejection is what this test pins there. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments

(* The backend's accumulator residency, which a [Privatize] tile is minted at (gh-ocannl-1116). *)
let accum_prec =
  let caps = lazy (Context.codegen_capabilities (Context.auto ())) in
  fun p -> (Lazy.force caps).Ir.Backend_intf.accum_prec p

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
let has_shared = Sched.backend_is_gpu backend_name

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
let bm, bn, bk = (8, 8, 8)

let () =
  let mav =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:13 ~offset:0. ~stride:0.25)
  in
  let mbv =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:17 ~offset:(-8.) ~stride:1.)
  in
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in

  (* --- Serial twin --- *)
  let%op mc0 = ma * mb in
  let serial_comp = named "smem_mm_serial" (Train.forward mc0) in
  let ctx_s = Context.auto () in
  let ctx_s, routine_s =
    Context.compile ~lowered_transform:(fun opt -> [ opt ]) ctx_s serial_comp Ir.Indexing.Empty
  in
  let ctx_s = Context.run ctx_s routine_s in
  let got_serial = nonzero "smem_serial" (Context.get_values ctx_s mc0.Tensor.value) in

  (* --- The SMEM schedule --- *)
  let%op mc1 = ma * mb in
  let smem_schedule (opt : LL.optimized) : Sched.schedule =
    let paths = nest_paths opt.LL.llc in
    (* The lowered code is [Zero_out mc1] (an opaque statement until Expand_zero) plus the naive
       triple loop -- the only loop nest. *)
    let accum = List.find_exn paths ~f:(fun p -> List.length p = 3) in
    let i, j, k = match accum with [ i; j; k ] -> (i, j, k) | _ -> assert false in
    let ez, zsyms = Sched.expand_zero ~tn:mc1.Tensor.value in
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
          swizzle = None;
          pad_stride = None;
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
          swizzle = None;
          pad_stride = None;
          pipeline_depth = 1;
          tile_prec = None;
        };
      Sched.privatize ~accum_prec ~target:mc1.Tensor.value ~over:k_o;
    ]
  in
  let smem_comp = named "mm_smem" (Train.forward mc1) in
  let transform opt = Sched.apply (smem_schedule opt) opt in
  let ctx_a = Context.auto () in
  if has_shared then (
    let ctx_a, routine_a =
      Context.compile
        ~lowered_transform:(fun o -> [ transform o ])
        ctx_a smem_comp Ir.Indexing.Empty
    in
    let ctx_a = Context.run ctx_a routine_a in
    let got_smem = Context.get_values ctx_a mc1.Tensor.value in
    p_all2 "SMEM matmul parity (GPU) or clean rejection (CPU)" got_smem got_serial ~f:approx;
    let src = Generated.read "mm_smem" in
    let has sub = String.is_substring src ~substring:sub in
    let count_sub sub =
      String.substr_index_all src ~may_overlap:false ~pattern:sub |> List.length
    in
    (* This is intentionally dialect identity: [shared_ok] pins each language's shared-memory
       declaration and barrier spellings in emitted source. *)
    let shared_ok =
      if String.is_substring backend_name ~substring:"metal" then
        count_sub "threadgroup float tile_" = 2 && has "threadgroup_barrier"
      else count_sub "__shared__ float tile_" = 2 && has "__syncthreads()"
    in
    (* All tile sizes divide the extents: Split remainder guards and Stage/Privatize edge guards
       fold; the only Ifs left are the two thread-0 load restrictions. The k loop accumulates into
       the privatized scalar [acc_mc1] instead of round-tripping mc1 through global memory. *)
    p "shared tiles, barriers, no edge guards (GPU) or rejected (CPU)"
      (shared_ok && count_sub "if (" = 2 && has "acc_mc1"))
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
        p "SMEM matmul parity (GPU) or clean rejection (CPU)"
          (String.is_substring msg ~substring:"not supported")
    | None -> p "SMEM matmul parity (GPU) or clean rejection (CPU)" false);
    skipped "shared tiles, barriers, no edge guards (GPU) or rejected (CPU)");

  (* --- Broadcast-vector staging (PR #90 review regression): a shared Stage with no anchor — the
     source access has no outer-part symbols and no reused workgroup axis. The load nest must NOT go
     to the routine top level (every hardware thread executes top-level statements, and no workgroup
     index symbol is bound there to restrict the writers): Stage wraps the outermost tile loop
     instead, where the enclosing Workgroup axis guards the cooperative load down to one
     representative thread. --- *)
  let a2v =
    Array.init (n * 32) ~f:(Ll_test.cycle_flat ~dims:[| n; 32 |] ~modulus:19 ~offset:0. ~stride:0.5)
  in
  let vv = Array.init 32 ~f:(fun idx -> Float.of_int idx -. 15.5) in
  let a2 = TDSL.ndarray a2v ~label:[ "a2" ] ~output_dims:[ n; 32 ] () in
  let v = TDSL.ndarray vv ~label:[ "v" ] ~output_dims:[ 32 ] () in
  let bcast_expected = Array.init (n * 32) ~f:(fun idx -> a2v.(idx) +. vv.(idx % 32)) in
  let%op cb = a2 + v in
  let bcast_comp = named "bcast_staged" (Train.forward cb) in
  let bcast_transform (opt : LL.optimized) : LL.optimized =
    let paths = nest_paths opt.LL.llc in
    let i, j =
      match List.find_exn paths ~f:(fun p -> List.length p = 2) with
      | [ i; j ] -> (i, j)
      | _ -> assert false
    in
    let sp_i, _, _ = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Workgroup in
    Sched.apply
      [
        sp_i;
        Sched.Stage
          {
            source = v.Tensor.value;
            tile_loops = [ j ];
            shared = true;
            cooperative = None;
            hoisted = false;
            swizzle = None;
            pad_stride = None;
            pipeline_depth = 1;
            tile_prec = None;
          };
      ]
      opt
  in
  let ctx_b = Context.auto () in
  if has_shared then (
    let ctx_b, routine_b =
      Context.compile
        ~lowered_transform:(fun o -> [ bcast_transform o ])
        ctx_b bcast_comp Ir.Indexing.Empty
    in
    let ctx_b = Context.run ctx_b routine_b in
    let got_b = Context.get_values ctx_b cb.Tensor.value in
    p_all2 "broadcast staging values correct (GPU) or clean rejection (CPU)" got_b bcast_expected
      ~f:approx;
    let src = Generated.read "bcast_staged" in
    let count_sub sub =
      String.substr_index_all src ~may_overlap:false ~pattern:sub |> List.length
    in
    (* One shared tile; the single If is the thread-0 restriction of the cooperative load — its
       presence is exactly what the review fix guarantees (previously the load sat unguarded at
       routine top level). *)
    p "broadcast load is thread-0 guarded under the workgroup axis"
      (count_sub "if (" = 1
      &&
      (* This is intentionally dialect identity: the assertion counts the literal shared-memory
         declaration token emitted by each language. *)
      if String.is_substring backend_name ~substring:"metal" then
        count_sub "threadgroup float tile_" = 1
      else count_sub "__shared__ float tile_" = 1))
  else (
    (match
       try
         ignore
           (Context.compile
              ~lowered_transform:(fun o -> [ bcast_transform o ])
              ctx_b bcast_comp Ir.Indexing.Empty
             : Context.t * Context.routine);
         None
       with Invalid_argument msg -> Some msg
     with
    | Some msg ->
        p "broadcast staging values correct (GPU) or clean rejection (CPU)"
          (String.is_substring msg ~substring:"not supported")
    | None -> p "broadcast staging values correct (GPU) or clean rejection (CPU)" false);
    skipped "broadcast load is thread-0 guarded under the workgroup axis")
