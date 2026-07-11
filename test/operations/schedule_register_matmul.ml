(* Schedule IR, Phase S3 (docs/proposals/schedule-ir-optops.md §4): the register-blocktiled
   matmul schedule -- Split + Swap + Stage + materializing Unroll -- executed against the serial
   twin. "Register blocktiling is not a bespoke transform": each output dimension is split twice
   (block tile BM/BN = Grid, then register tile TM/TN), the register loops are sunk innermost by
   perfect-nesting Swaps, the operands are staged through workgroup-shared tiles at k_o, and the
   register loops are unrolled in the IR (materialize = true) so the trailing simplify folds the
   Affine indices of the TM x TN copies into constants.

   Kernel geometry: 64x64 times 64x64, BM = BN = 16, BK = 8, TM = TN = 4 -- a 4x4 grid of 4x4
   workgroups, each thread computing a 4x4 register tile of outputs; the unrolled body carries
   TM * TN = 16 fma statements. All tile sizes divide the extents so every constructed guard
   folds; the only surviving Ifs are the two thread-0 load restrictions.

   On the C backends shared placement is rejected cleanly (as in schedule_smem_matmul). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments

let () = Utils.settings.output_debug_files_in_build_directory <- true
let p name b = Stdio.printf "%s: %b\n" name b
let approx a b = Float.(abs (a - b) < 1e-2)

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

let has_shared =
  String.is_substring backend_name ~substring:"metal"
  || String.is_substring backend_name ~substring:"cuda"

let read_generated base_name =
  let ext = if String.is_substring backend_name ~substring:"metal" then ".metal" else ".cu" in
  let path = Stdlib.Filename.concat "build_files" (base_name ^ ext) in
  if Stdlib.Sys.file_exists path then Some (Stdio.In_channel.read_all path) else None

let nest_paths (llc : LL.t) : Ir.Indexing.symbol list list =
  let strip stmts =
    List.filter stmts ~f:(function LL.Noop | LL.Comment _ -> false | _ -> true)
  in
  let rec path (llc : LL.t) : Ir.Indexing.symbol list =
    match llc with
    | LL.For_loop { index; body; _ } -> (
        index :: (match strip (LL.flat_lines [ body ]) with [ single ] -> path single | _ -> []))
    | LL.If { body; _ } -> path body
    | _ -> []
  in
  List.filter_map (LL.flat_lines [ llc ]) ~f:(fun stmt ->
      match path stmt with [] -> None | p -> Some p)

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let n = 64
let bm, bn, bk = (16, 16, 8)
let tm, tn = (4, 4)

let () =
  let mav = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 13) *. 0.25) in
  let mbv = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 17) -. 8.) in
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in

  (* --- Serial twin --- *)
  let%op mc0 = ma * mb in
  let serial_comp = named "mmr_serial" (Train.forward mc0) in
  let ctx_s = Context.auto () in
  let ctx_s, routine_s =
    Context.compile ~lowered_transform:(fun opt -> opt) ctx_s serial_comp Ir.Indexing.Empty
  in
  let ctx_s = Context.run ctx_s routine_s in
  let got_serial = Context.get_values ctx_s mc0.Tensor.value in

  (* --- The register-blocktiled schedule --- *)
  let%op mc1 = ma * mb in
  let register_schedule (opt : LL.optimized) : Sched.schedule =
    let paths = nest_paths opt.LL.llc in
    let accum = List.find_exn paths ~f:(fun p -> List.length p = 3) in
    let i, j, k = match accum with [ i; j; k ] -> (i, j, k) | _ -> assert false in
    let ez, zsyms = Sched.expand_zero ~tn:mc1.Tensor.value in
    let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
    (* Zero nest: same hardware geometry as the accumulation (barriers require slot-uniform
       workgroup extents). *)
    let sp_zi, _, zi_i = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_zi2, _, _ = Sched.split ~axis:zi_i ~factor:tm ~outer:LL.Workgroup ~inner:LL.Serial in
    let sp_zj, _, zj_i = Sched.split ~axis:zj ~factor:bn ~outer:LL.Grid ~inner:LL.Serial in
    let sp_zj2, _, _ = Sched.split ~axis:zj_i ~factor:tn ~outer:LL.Workgroup ~inner:LL.Serial in
    (* Accumulation nest: block tiles to Grid, register-tile strides to Workgroup. *)
    let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_i2, i_w, i_t = Sched.split ~axis:i_i ~factor:tm ~outer:LL.Workgroup ~inner:LL.Serial in
    let sp_j, j_o, j_i = Sched.split ~axis:j ~factor:bn ~outer:LL.Grid ~inner:LL.Serial in
    let sp_j2, j_w, j_t = Sched.split ~axis:j_i ~factor:tn ~outer:LL.Workgroup ~inner:LL.Serial in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    (* Sink the register loops innermost: i_t past the j-splits and k-splits (they sit between the
       i-splits and k in the original nesting), then j_t past the k-splits. *)
    let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner }) in
    let swaps = sink i_t [ j_o; j_w; j_t; k_o; k_i ] @ sink j_t [ k_o; k_i ] in
    [ ez; sp_zi; sp_zi2; sp_zj; sp_zj2; sp_i; sp_i2; sp_j; sp_j2; sp_k ]
    @ swaps
    @ [
        Sched.Stage { source = ma.Tensor.value; tile_loops = [ i_w; i_t; k_i ]; shared = true; cooperative = None; hoisted = false };
        Sched.Stage { source = mb.Tensor.value; tile_loops = [ k_i; j_w; j_t ]; shared = true; cooperative = None; hoisted = false };
        Sched.Privatize { target = mc1.Tensor.value; over = k_o };
        Sched.Unroll { axis = i_t; materialize = true };
        Sched.Unroll { axis = j_t; materialize = true };
      ]
  in
  let reg_comp = named "mmr_tiled" (Train.forward mc1) in
  let transform opt = Sched.apply (register_schedule opt) opt in
  let ctx_a = Context.auto () in
  if has_shared then (
    let ctx_a, routine_a =
      Context.compile ~lowered_transform:transform ctx_a reg_comp Ir.Indexing.Empty
    in
    let ctx_a = Context.run ctx_a routine_a in
    let got_reg = Context.get_values ctx_a mc1.Tensor.value in
    p "register-tiled matmul parity (GPU) or clean rejection (CPU)"
      (Array.for_all2_exn got_reg got_serial ~f:approx);
    match read_generated "mmr_tiled" with
    | None -> p "unrolled register tile structure (GPU) or rejected (CPU)" false
    | Some src ->
        let count_sub sub =
          String.substr_index_all src ~may_overlap:false ~pattern:sub |> List.length
        in
        (* TM * TN fma statements from the materialized unrolls, two shared tiles, and only the
           two thread-0 load guards survive. Metal spells single-precision FMA [fma(], CUDA
           [fmaf(]; the patterns are disjoint so the counts add. *)
        p "unrolled register tile structure (GPU) or rejected (CPU)"
          (count_sub "fma(" + count_sub "fmaf(" = tm * tn
          && count_sub "if (" = 2
          && String.is_substring src ~substring:"acc_mc1"
          &&
          if String.is_substring backend_name ~substring:"metal" then
            count_sub "threadgroup float tile_" = 2
          else count_sub "__shared__ float tile_" = 2))
  else (
    (match
       try
         ignore
           (Context.compile ~lowered_transform:transform ctx_a reg_comp Ir.Indexing.Empty
             : Context.t * Context.routine);
         None
       with Invalid_argument msg -> Some msg
     with
    | Some msg ->
        p "register-tiled matmul parity (GPU) or clean rejection (CPU)"
          (String.is_substring msg ~substring:"not supported")
    | None -> p "register-tiled matmul parity (GPU) or clean rejection (CPU)" false);
    p "unrolled register tile structure (GPU) or rejected (CPU)" true)
