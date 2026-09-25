(* Schedule IR, Phase S4 (docs/proposals/schedule-ir-optops.md §5, watch-ocannl-README-md-347818d3
   scope): cache-sized tiling + operand packing for the C backends, built from Split / Swap /
   non-shared Stage optops and executed against the naive twin.

   [mc = ma * mb] (32x32 times 32x32, 8x8x8 tiles): split each of i, j, k by 8 (all Serial — nothing
   hardware-bound, so this schedule is legal on every backend and the Zero_out needs no expansion),
   sink the intra-tile loops into the classic tiled i_o j_o k_o k_i i_i j_i order (contiguous
   row-major access to mb and mc in the innermost loop), then pack both operand tiles into
   contiguous stack scratch via [Stage ~shared:false] — Boehm's operand packing: a serial copy nest
   at k_o, no barriers, tiles declared as plain per-routine local arrays. Tile sizes divide the
   extents so the packing edge guards all fold away.

   Runs on every backend (on GPUs it is just a 1x1 serial kernel); the structural expectations below
   hold everywhere: two packed tile declarations without any shared-memory prefix, no barriers, no
   guards. *)

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

let approx a b = Float.(abs (a - b) < 1e-3)
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name

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

  (* --- Naive twin --- *)
  let%op mc0 = ma * mb in
  let serial_comp = named "mmp_naive" (Train.forward mc0) in
  let ctx_s = Context.auto () in
  let ctx_s, routine_s =
    Context.compile ~lowered_transform:(fun opt -> [ opt ]) ctx_s serial_comp Ir.Indexing.Empty
  in
  let ctx_s = Context.run ctx_s routine_s in
  let got_naive = nonzero "cpupack_naive" (Context.get_values ctx_s mc0.Tensor.value) in

  (* --- Tiled + packed --- *)
  let%op mc1 = ma * mb in
  let pack_schedule (opt : LL.optimized) : Sched.schedule =
    let paths = nest_paths opt.LL.llc in
    let accum = List.find_exn paths ~f:(fun p -> List.length p = 3) in
    let i, j, k = match accum with [ i; j; k ] -> (i, j, k) | _ -> assert false in
    let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Serial ~inner:LL.Serial in
    let sp_j, j_o, j_i = Sched.split ~axis:j ~factor:bn ~outer:LL.Serial ~inner:LL.Serial in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner }) in
    [ sp_i; sp_j; sp_k ]
    (* i_o i_i j_o j_i k_o k_i -> i_o j_o k_o k_i i_i j_i (contiguous mb/mc rows innermost). *)
    @ sink i_i [ j_o; j_i; k_o; k_i ]
    @ sink j_i [ k_o; k_i; i_i ]
    @ [
        Sched.Stage
          {
            source = ma.Tensor.value;
            tile_loops = [ i_i; k_i ];
            shared = false;
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
            shared = false;
            cooperative = None;
            hoisted = false;
            swizzle = None;
            pad_stride = None;
            pipeline_depth = 1;
            tile_prec = None;
          };
        Sched.Privatize { target = mc1.Tensor.value; over = k_o };
      ]
  in
  let pack_comp = named "mmp_packed" (Train.forward mc1) in
  let transform opt = Sched.apply (pack_schedule opt) opt in
  let ctx_a = Context.auto () in
  let ctx_a, routine_a =
    Context.compile ~lowered_transform:(fun o -> [ transform o ]) ctx_a pack_comp Ir.Indexing.Empty
  in
  let ctx_a = Context.run ctx_a routine_a in
  let got_packed = Context.get_values ctx_a mc1.Tensor.value in
  p_all2 "packed tiled matmul matches the naive twin" got_packed got_naive ~f:approx;
  let src = Generated.read "mmp_packed" in
  let has sub = String.is_substring src ~substring:sub in
  let count_sub sub = String.substr_index_all src ~may_overlap:false ~pattern:sub |> List.length in
  p "packed tiles are plain local arrays, no barriers, no guards"
    (count_sub "float tile_" = 2
    && has "acc_mc1"
    && (not (has "threadgroup float tile_"))
    && (not (has "__shared__"))
    && (not (has "barrier"))
    && not (has "if ("))
