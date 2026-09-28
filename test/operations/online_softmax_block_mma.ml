(* gh-ocannl-1003: the block fold's two contractions on matrix units ([Schedule.Fold_mma], which the
   default GPU preset emits for a kernel holding the fold alone).

   The fold ([online_softmax_block], test/operations/online_softmax_block.ml) runs one scan per
   query row; its score slice [q . k^T] and value update [P . V] are rows of block matmuls only the
   query rows together form. On a backend whose MMA units take f32, the preset splits the query loop
   by the SIMD width into lanes -- one lane per query row, each running its row's scan in lockstep
   -- and replaces each contraction by one [Tile_mma] over the block between barriers, the tiles in
   workgroup-shared memory.

   Whether the backend can is read off its advertised capability (the f32 triple in
   [hardware_limits.mma]), never off its name: where it can, both contractions must render as
   intrinsics -- a nonzero aggregate count is not enough, so the census must count exactly the two
   statements with no fallback among them, and the generated source is checked; where it cannot (cc;
   CUDA without tf32), the scalar fold ships and no [Tile_mma] is emitted. Every leg is executed
   parity against the composed form, on the same backend; device floats stay off the golden. *)

open Base
open Stdio
module Train = Ocannl.Train
module Nn_blocks = Ocannl.Nn_blocks
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module Online_softmax = Ir.Online_softmax
module Generated = Test_utils.Generated

let backend_name = Utils.get_global_arg ~arg_name:"backend" ~default:"cc"
let () = Generated.init ~backend_name
let () = Utils.settings.output_debug_files_in_build_directory <- true
let batch = 2
let d_model = 16
let heads = 2

let f32_mma =
  lazy
    (match (Context.hardware_limits (Context.auto ())).Ir.Backend_intf.mma with
    | Some mma ->
        List.Assoc.mem mma.Ir.Backend_intf.mma_format_tiles
          (Ir.Backend_intf.Mma_f32, Ir.Backend_intf.Mma_f32, Ir.Backend_intf.Mma_f32)
          ~equal:Ir.Backend_intf.equal_mma_format_triple
    | None -> false)

let width =
  lazy
    (Option.value_map (Context.hardware_limits (Context.auto ())).Ir.Backend_intf.mma ~default:0
       ~f:(fun m -> m.Ir.Backend_intf.mma_simd_width))

let mask ~seq =
  NTDSL.init ~l:"mask" ~prec:Ir.Ops.single ~b:[ seq ] ~i:[ seq ] ~o:[]
    ~f:(function [| s; t |] -> if s >= t then 1. else 0. | _ -> assert false)
    ()

let model ~seq ~d_k () =
  let x =
    TDSL.range_of_shape ~label:[ "x" ] ~batch_dims:[ batch; seq ] ~input_dims:[]
      ~output_dims:[ d_model ] ()
  in
  let scale = Float.of_int (batch * seq * d_model) in
  let%op x = x /. !.scale in
  let mask = mask ~seq in
  let block = Nn_blocks.multi_head_attention ~label:[ "attn" ] ~num_heads:heads ~d_k ~d_v:d_k () in
  let%op y = x + block ~train_step:None ~mask x in
  y

type run = { values : float array; mma : Ir.C_syntax.mma_summary }

(* The forward under the default pipeline (the preset is what emits [Fold_mma]). *)
let forward ~name ~on ~block ~seq ~d_k () =
  Tensor.unsafe_reinitialize ();
  Online_softmax.set_enabled (Some on);
  Online_softmax.set_block (Some block);
  Exn.protect
    ~finally:(fun () ->
      Online_softmax.set_enabled None;
      Online_softmax.set_block None)
    ~f:(fun () ->
      let t = model ~seq ~d_k () in
      Train.set_materialized t.Tensor.value;
      let ctx = Train.init_params (Context.auto ()) Ir.Indexing.Empty t in
      Generated.arm (name ^ "__seg");
      let ctx, routine = Context.compile ~name ctx t.Tensor.forward Ir.Indexing.Empty in
      let ctx = Context.run ctx routine in
      { values = Context.get_values ctx t.Tensor.value; mma = routine.Context.mma })

let close ~tol g w = Float.(abs (g -. w) <= tol *. max 1. (abs w))

let report label (got : float array) =
  eprintf "%s: %s (not part of the golden)\n%!" label
    (String.concat ~sep:" "
       (Array.to_list (Array.map (Array.sub got ~pos:0 ~len:6) ~f:(Printf.sprintf "%.9g"))))

let () =
  let capable = Lazy.force f32_mma in
  eprintf "backend: %s, f32 MMA advertised: %b, SIMD width %d (not part of the golden)\n%!"
    backend_name capable (Lazy.force width);
  printf "--- leg 1: query counts a multiple of the SIMD width, key blocks the tile divides ---\n";
  List.iter
    [ (64, 32, 8); (64, 32, 16); (64, 32, 32); (96, 16, 32) ]
    ~f:(fun (seq, d_k, block) ->
      let what = Printf.sprintf "seq %d, head width %d, block %d" seq d_k block in
      let composed = forward ~name:"osbm_composed" ~on:false ~block:0 ~seq ~d_k () in
      let fold = forward ~name:"osbm_fold" ~on:true ~block ~seq ~d_k () in
      report (what ^ " composed") composed.values;
      report (what ^ " fold") fold.values;
      eprintf "%s: mma %s (not part of the golden)\n%!" what
        (Sexp.to_string_hum (Ir.C_syntax.sexp_of_mma_summary fold.mma));
      gated ~when_:capable ~on:backend_name
        (what ^ ": two Tile_mma statements, the score and the value contraction, both intrinsics")
        (fold.mma.Ir.C_syntax.statements = 2 && fold.mma.Ir.C_syntax.scalar_fallbacks = 0);
      gated ~when_:capable ~on:backend_name
        (what ^ ": the generated kernel emits the matrix-unit intrinsic")
        (capable
        && String.is_substring (Generated.read "osbm_fold__seg")
             ~substring:"simdgroup_multiply_accumulate");
      gated ~when_:(not capable) ~on:backend_name
        (what ^ ": without f32 MMA units the scalar fold ships, no Tile_mma emitted")
        (fold.mma.Ir.C_syntax.statements = 0);
      p_all2
        (what ^ ": matches the composed output within 1e-4 relative")
        fold.values composed.values ~f:(close ~tol:1e-4));
  printf "--- leg 2: what keeps the scalar fold ---\n";
  List.iter
    [
      ("a query count not a multiple of the SIMD width", 40, 32, 8);
      ("a key tail (the block does not divide the key count)", 64, 32, 24);
    ]
    ~f:(fun (why, seq, d_k, block) ->
      let composed = forward ~name:"osbm_composed" ~on:false ~block:0 ~seq ~d_k () in
      let fold = forward ~name:"osbm_fold" ~on:true ~block ~seq ~d_k () in
      p (why ^ ": no Tile_mma emitted") (fold.mma.Ir.C_syntax.statements = 0);
      p_all2
        (why ^ ": matches the composed output within 1e-4 relative")
        fold.values composed.values ~f:(close ~tol:1e-4))

(* --- Leg 3: the training step (the fold forward under the fused backward, treatment F). --- *)

let training ~on ~block ~bwd ~seq ~d_k =
  Tensor.unsafe_reinitialize ();
  Online_softmax.set_enabled (Some on);
  Online_softmax.set_block (Some block);
  Online_softmax.set_backward_enabled (Some bwd);
  Exn.protect
    ~finally:(fun () ->
      Online_softmax.set_enabled None;
      Online_softmax.set_block None;
      Online_softmax.set_backward_enabled None)
    ~f:(fun () ->
      let y = model ~seq ~d_k () in
      let%op loss = (y *. y) ++ "... | ... => 0" in
      let params =
        Set.to_list y.Tensor.params
        |> List.sort ~compare:(fun a b ->
            Int.compare a.Tensor.value.Ir.Tnode.id b.Tensor.value.Ir.Tnode.id)
      in
      List.iter params ~f:(fun p ->
          Train.set_materialized (Option.value_exn p.Tensor.diff).Tensor.grad);
      let update = Train.grad_update loss in
      let ctx = Train.init_params (Context.auto ()) Ir.Indexing.Empty loss in
      let ctx, routine = Context.compile ~name:"osbm_step" ctx update Ir.Indexing.Empty in
      let ctx = Context.run ctx routine in
      ( List.map params ~f:(fun p ->
            Context.get_values ctx (Option.value_exn p.Tensor.diff).Tensor.grad),
        routine.Context.mma ))

let () =
  printf "--- leg 3: the training step, the fold forward under the fused backward ---\n";
  let capable = Lazy.force f32_mma in
  let seq = 64 and d_k = 32 in
  let composed, _ = training ~on:false ~block:0 ~bwd:false ~seq ~d_k in
  let fused, mma = training ~on:true ~block:16 ~bwd:true ~seq ~d_k in
  eprintf "training step: mma %s (not part of the golden)\n%!"
    (Sexp.to_string_hum (Ir.C_syntax.sexp_of_mma_summary mma));
  gated ~when_:capable ~on:backend_name "the step's fold renders both contractions as intrinsics"
    (mma.Ir.C_syntax.statements = 2 && mma.Ir.C_syntax.scalar_fallbacks = 0);
  List.iter2_exn fused composed ~f:(fun gf gc ->
      p_all2 "a parameter gradient agrees with the composed step within 1e-4 relative" gf gc
        ~f:(close ~tol:1e-4))
