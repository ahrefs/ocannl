(* Hoisted (out-of-routine) operand packing for compile-time constants (gh-ocannl-470): a [Stage
   ~shared:false ~hoisted:true] of a known-constant, host-init-backed operand emits no in-kernel
   load nest at all — the packed layout covers the whole source and is materialized once into the
   per-device constant pool at link time (the compiler-native analog of ggml's [CPU_REPACK]
   [set_tensor] hook).

   Three checks against [mc = ma * mb] with constant [ma], [mb]:

   1. Both operands hoisted, tile sizes dividing the extents (32x32, 8x8x8 tiles): values match the
   naive twin, and the generated code has no scratch tile arrays and no packing copy nests — the
   kernel reads the packed constants directly (they appear as ordinary buffer parameters).

   2. Mixed flavors on non-dividing extents (20x20, 8x8x8 tiles): [ma] hoisted, [mb] packed
   in-kernel. Values still match: hoisted edge tiles are zero-filled on the host and the consumer
   keeps its own remainder guards.

   3. A non-constant source (the output of an upstream matmul) is rejected with
   [Invalid_argument]. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
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

let bm, bn, bk = (8, 8, 8)

(* The classic tiled i_o j_o k_o k_i i_i j_i order of schedule_cpu_pack_matmul.ml, with per-operand
   staging flavor. [reorder = false] skips the Swaps and the Privatize (Swap needs perfect nesting
   and Privatize an iteration-invariant guard chain — both broken by the remainder guards of
   non-dividing splits), leaving splits + stages only. *)
let tiled_schedule ?(reorder = true) ~ma ~mb ~mc ~hoist_a ~hoist_b (opt : LL.optimized) :
    Sched.schedule =
  let paths = nest_paths opt.LL.llc in
  let accum = List.find_exn paths ~f:(fun p -> List.length p = 3) in
  let i, j, k = match accum with [ i; j; k ] -> (i, j, k) | _ -> assert false in
  let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Serial ~inner:LL.Serial in
  let sp_j, j_o, j_i = Sched.split ~axis:j ~factor:bn ~outer:LL.Serial ~inner:LL.Serial in
  let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
  let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner }) in
  let stages =
    [
      Sched.Stage
        {
          source = ma;
          tile_loops = [ i_i; k_i ];
          shared = false;
          cooperative = None;
          hoisted = hoist_a;
          swizzle = None;
          pad_stride = None;
          pipeline_depth = 1;
          tile_prec = None;
        };
      Sched.Stage
        {
          source = mb;
          tile_loops = [ k_i; j_i ];
          shared = false;
          cooperative = None;
          hoisted = hoist_b;
          swizzle = None;
          pad_stride = None;
          pipeline_depth = 1;
          tile_prec = None;
        };
    ]
  in
  [ sp_i; sp_j; sp_k ]
  @ (if reorder then sink i_i [ j_o; j_i; k_o; k_i ] @ sink j_i [ k_o; k_i; i_i ] else [])
  @ stages
  @ if reorder then [ Sched.privatize ~accum_prec ~target:mc ~over:k_o ] else []

let make_pair n =
  let mav =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:13 ~offset:0. ~stride:0.25)
  in
  let mbv =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:17 ~offset:(-8.) ~stride:1.)
  in
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  (ma, mb)

let run_forward ?(transform = fun opt -> opt) comp =
  let ctx = Context.auto () in
  let ctx, routine =
    Context.compile ~lowered_transform:(fun o -> [ transform o ]) ctx comp Ir.Indexing.Empty
  in
  Context.run ctx routine

(* --- 1. Both operands hoisted, dividing extents --- *)
let () =
  let ma, mb = make_pair 32 in
  let%op mc0 = ma * mb in
  let ctx = run_forward (named "mmh_naive" (Train.forward mc0)) in
  let got_naive = nonzero "mmh_naive" (Context.get_values ctx mc0.Tensor.value) in
  let%op mc1 = ma * mb in
  let transform opt =
    Sched.apply
      (tiled_schedule ~ma:ma.Tensor.value ~mb:mb.Tensor.value ~mc:mc1.Tensor.value ~hoist_a:true
         ~hoist_b:true opt)
      opt
  in
  let ctx = run_forward ~transform (named "mmh_hoisted" (Train.forward mc1)) in
  let got_hoisted = Context.get_values ctx mc1.Tensor.value in
  p_all2 "hoisted-packed matmul matches the naive twin" got_hoisted got_naive ~f:approx;
  let src = Generated.read "mmh_hoisted" in
  let has sub = String.is_substring src ~substring:sub in
  p "no in-kernel scratch or copy nests; packed constants are buffer params"
    (has "packed_ma" && has "packed_mb"
    && (not (has "float tile_"))
    (* Shared-tile declarations — not the [[..._in_threadgroup]] parameter attributes every Metal
       kernel carries. *)
    && (not (has "threadgroup float"))
    && (not (has "__shared__"))
    && not (has "barrier("))

(* --- 2. Mixed flavors, non-dividing extents (edge tiles) --- *)
let () =
  let ma, mb = make_pair 20 in
  let%op mc0 = ma * mb in
  let ctx = run_forward (named "mme_naive" (Train.forward mc0)) in
  let got_naive = nonzero "mmh_naive" (Context.get_values ctx mc0.Tensor.value) in
  let%op mc1 = ma * mb in
  let transform opt =
    Sched.apply
      (tiled_schedule ~reorder:false ~ma:ma.Tensor.value ~mb:mb.Tensor.value ~mc:mc1.Tensor.value
         ~hoist_a:true ~hoist_b:false opt)
      opt
  in
  let ctx = run_forward ~transform (named "mme_mixed" (Train.forward mc1)) in
  let got_mixed = Context.get_values ctx mc1.Tensor.value in
  p_all2 "mixed hoisted + in-kernel packing on edge tiles matches the naive twin" got_mixed
    got_naive ~f:approx

(* --- 3. Non-constant sources are rejected --- *)
let () =
  let n = 32 in
  let ma, _mb = make_pair n in
  (* A values-backed parameter: host-init data is registered (so packing input exists), but the node
     is trainable, not a compile-time constant — hoisting must refuse it. *)
  let wv =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:7 ~offset:0. ~stride:0.5)
  in
  let w = TDSL.param ~values:wv "w" ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op mc1 = ma * w in
  let transform opt =
    Sched.apply
      (tiled_schedule ~ma:ma.Tensor.value ~mb:w.Tensor.value ~mc:mc1.Tensor.value ~hoist_a:false
         ~hoist_b:true opt)
      opt
  in
  match run_forward ~transform (named "mmr_rejected" (Train.forward mc1)) with
  | _ -> p "hoisting a non-constant source is rejected" false
  | exception Invalid_argument msg ->
      p "hoisting a non-constant source is rejected"
        (String.is_substring msg ~substring:"hoisted staging requires a known-constant source")

(* --- 4. Operand constancy enters the canonical digest --- *)
(* Same-shape matmuls over constant vs trainable operands must not share schedule-cache keys: a
   cached non-hoisted winner for the trainable program would otherwise mask the constant
   program's hoisted candidates, and a hoisted winner would replay against a site it cannot
   apply to (Codex P2 on PR #123). *)
let () =
  let n = 32 in
  let ma, mb = make_pair n in
  let%op mc0 = ma * mb in
  let wv =
    Array.init (n * n) ~f:(Ll_test.cycle_flat ~dims:[| n; n |] ~modulus:7 ~offset:0. ~stride:0.5)
  in
  let w = TDSL.param ~values:wv "w" ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op mc1 = ma * w in
  let digest comp =
    let d = ref "" in
    let transform opt =
      d := Ir.Schedule_cache.digest (Ir.Schedule_cache.canonicalize opt);
      opt
    in
    let (_ : Context.t * _) =
      Context.compile
        ~lowered_transform:(fun o -> [ transform o ])
        (Context.auto ()) comp Ir.Indexing.Empty
    in
    !d
  in
  let d_const = digest (named "mmd_const" (Train.forward mc0)) in
  let d_param = digest (named "mmd_param" (Train.forward mc1)) in
  p "operand constancy enters the schedule-cache digest"
    ((not (String.is_empty d_const)) && not (String.equal d_const d_param))
