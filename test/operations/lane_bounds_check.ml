(* gh-ocannl-1867: [debug_check_lane_bounds] turns an out-of-bounds lane of a packed-uniform lane
   extract into an abort. The [uint4x32_to_<prec>_uniform_lane] builtins index the converted block
   unchecked, and the block's lane count belongs to the precision the call RENDERS at: a lane index
   sized for another precision reads past it. That is the form gh-ocannl-1769's miscompile left -- a
   half draw's [flat mod 8] lane index beneath a store rendered at single, a 4-lane block -- and
   what it read was platform-dependent garbage (zeros and 1.4e27 on x86/gcc, a finite wrong loss on
   arm64/clang), not a crash.

   The kernel is hand-built in that form, since the lowering no longer produces it: a single-
   precision store of the lane extract at lane [i mod lanes], for [lanes] 4 (every lane inside the
   block) and 8 (lanes 4..7 outside it). An abort ends the process, so each leg runs in a child (the
   [child] mode) and the parent reads its fate:

   - in range, key on: the lane operand is wrapped, the child exits normally; - in range, key off:
   the kernel carries no check, and draws the same bits as the checked one; - out of range, key on:
   the child is killed by its own [abort], having named the lane, the lane count and the precision
   on stderr. Lane 4 is the first one outside the block, so that is the one named.

   Out of range with the key off is the undefined behavior the check exists for, and is not run. The
   lowering's own lane extracts under the key are [test_uniform_virtual_lane]'s [-lane_checked]
   run. *)

open Base
module LL = Ir.Low_level
module Ops = Ir.Ops
module Idx = Ir.Indexing
open Verdict.Claims

let u4 = Ops.uint4x32
let iprec = Ops.index_prec ()
let n = 8
let routine = "lane_bounds"
let extract = "uint4x32_to_single_uniform_lane("
let wrapper = "OCANNL_CHECKED_LANE("

let child ~lanes =
  Utils.settings.output_debug_files_in_build_directory <- true;
  Test_utils.Generated.init ~backend_name:"cc";
  let mk = Ll_test.node_factory ~first_id:1867000 ~dims:[| n |] () in
  let out = mk "lane_out" in
  Ll_test.materialize out;
  let i = Ll_test.sym () in
  let bits =
    LL.Binop
      (Ops.Threefry4x32_light, (LL.Constant_bits 7L, u4), (LL.Embed_index (Idx.Fixed_idx 3), u4))
  in
  let lane =
    LL.Binop
      ( Ops.Mod,
        (LL.Embed_index (Ll_test.iter i), iprec),
        (LL.Embed_index (Idx.Fixed_idx lanes), iprec) )
  in
  let llc =
    Ll_test.loop_n i n
      (Ll_test.set_at out (Ll_test.iter i)
         (LL.Binop (Ops.Uint4x32_to_prec_uniform_lane, (bits, u4), (lane, iprec))))
  in
  let optimized = Ll_test.optimize ~materialized:[ out ] ~name:routine llc in
  Test_utils.Generated.arm routine;
  let values =
    match Ll_test.execute ~name:routine optimized ~seed:[] ~read:[ out ] with
    | [ values ] -> values
    | _ -> assert false
  in
  (* The statements extracting a lane, for the parent to read the lane operand off. *)
  String.split_lines (Test_utils.Generated.main_logic (Test_utils.Generated.read routine))
  |> List.filter ~f:(String.is_substring ~substring:extract)
  |> List.iter ~f:(fun line -> Stdio.printf "kernel: %s\n" (String.strip line));
  Array.iter values ~f:(fun v -> Stdio.printf "draw: %Lx\n" (Int64.bits_of_float v))

(* An [abort]: SIGABRT, or on Windows, which has no signals, the C runtime's exit code 3. *)
let aborted = function
  | Unix.WSIGNALED s -> s = Stdlib.Sys.sigabrt
  | Unix.WEXITED 3 -> Stdlib.Sys.win32
  | _ -> false

let parent () =
  let run ~lanes ~checked =
    let key = "debug_check_lane_bounds=" ^ Bool.to_string checked in
    let ((status, stdout, stderr) as result) =
      Fresh_process.run [ "child"; Int.to_string lanes; "--ocannl_backend=cc"; "--ocannl_" ^ key ]
    in
    Fresh_process.report ~label:(Int.to_string lanes ^ " lanes, " ^ key) result;
    let tagged tag =
      List.filter_map (String.split_lines stdout) ~f:(String.chop_prefix ~prefix:(tag ^ ": "))
    in
    (status, tagged "kernel", tagged "draw", stderr)
  in
  let exited_normally status = Poly.equal status (Unix.WEXITED 0) in
  let wraps = String.is_substring ~substring:wrapper in
  let on_status, on_kernel, on_draws, _ = run ~lanes:4 ~checked:true in
  let off_status, off_kernel, off_draws, _ = run ~lanes:4 ~checked:false in
  p "in range, key on: the child exits normally" (exited_normally on_status);
  p_all ~min:1 "in range, key on: every lane extract's lane operand is wrapped in the check"
    on_kernel ~f:wraps;
  p "in range, key off: the child exits normally" (exited_normally off_status);
  p_none ~min:1 "in range, key off: no lane extract's lane operand is wrapped" off_kernel ~f:wraps;
  p "in range: the checked kernel stores one draw per cell" (List.length on_draws = n);
  p "in range: the draws are not all one value"
    (List.length (List.dedup_and_sort on_draws ~compare:String.compare) > 1);
  p "in range: the checked and unchecked kernels draw the same bits"
    (List.equal String.equal on_draws off_draws);
  let out_status, _, _, out_stderr = run ~lanes:8 ~checked:true in
  p "out of range, key on: the child is killed by abort" (aborted out_status);
  p "out of range, key on: stderr names the lane, the lane count and the rendered precision"
    (String.is_substring out_stderr
       ~substring:
         "ocannl debug_check_lane_bounds: lane 4 is outside the 4 lanes of a single uniform block")

let () =
  match
    List.filter
      (List.tl_exn (Array.to_list Stdlib.Sys.argv))
      ~f:(fun arg -> not (String.is_prefix arg ~prefix:"--"))
  with
  | [] -> parent ()
  | [ "child"; lanes ] -> child ~lanes:(Int.of_string lanes)
  | _ -> failwith "usage: lane_bounds_check.exe [child <lanes>] [--ocannl_<key>=<value>...]"
