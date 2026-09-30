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
   parity against the composed form, on the same backend; device floats stay off the golden.

   The preset's verdict is the transform's own dry run, not a prediction of it: a fold the scalar
   form accepts but the block rendering cannot take -- a value tensor indexed by the query, whose
   micro-kernel [tensorize_llc] refuses -- and a fold whose [Grid] launch the device's caps refuse
   both keep the scalar fold (leg 2), rather than failing the compile. *)

open Base
open Stdio
module Train = Ocannl.Train
module Nn_blocks = Ocannl.Nn_blocks
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module Online_softmax = Ir.Online_softmax
module Generated = Test_utils.Generated
module LL = Ir.Low_level
module Sched = Ir.Schedule
module BI = Ir.Backend_intf

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

(* Attention whose value tensor is indexed by the query as well as the key, [V[s, t, h, e]]: the
   scalar fold takes it (its value pass reads [V] per row), but the block rendering's value
   micro-kernel would make the [Tile_mma]'s B operand lane-dependent. *)
let model_query_indexed_v ~seq ~d_k () =
  let wave salt idcs =
    Float.sin
      (salt *. Float.of_int (Array.foldi idcs ~init:0 ~f:(fun a acc i -> acc + ((a + 1) * i))))
  in
  let leaf name salt ~b ~o = NTDSL.init ~l:name ~prec:Ir.Ops.single ~b ~o ~f:(wave salt) () in
  let q = leaf "q" 0.1 ~b:[ seq ] ~o:[ heads; d_k ] in
  let k = leaf "k" 0.2 ~b:[ seq ] ~o:[ heads; d_k ] in
  let v = leaf "v" 0.3 ~b:[ seq; seq ] ~o:[ heads; d_k ] in
  let mask = mask ~seq in
  let%op scores =
    (q +* k " ... s | h d; ... t | h d => ... s | t -> h" [ "h"; "d" ]) /. sqrt (dim d)
  in
  let%op masked = where mask scores !.Float.neg_infinity in
  let weights = Nn_blocks.softmax ~spec:" ... | t -> ..." () masked in
  let%op o = weights +* v " ... s | t -> h; ... s t | h e => ... s | h e" [ "e" ] in
  o

(* The scans a kernel carries: the fold's own survives both renderings (the block one keeps the scan
   and rewrites its body). *)
let scans (llc : LL.t) =
  let n = ref 0 in
  Ll_test.walk ~on_stmt:(function LL.Scan_loop _ -> Int.incr n | _ -> ()) llc;
  !n

type run = { values : float array; mma : Ir.C_syntax.mma_summary; scans : int }

(* The forward under the default pipeline (the preset is what emits [Fold_mma]), or under
   [lowered_transform] where a leg supplies its own pipeline. *)
let forward ?(model = model) ?lowered_transform ~name ~on ~block ~seq ~d_k () =
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
      let ctx, routine =
        Context.compile ?lowered_transform ~name ctx t.Tensor.forward Ir.Indexing.Empty
      in
      let ctx = Context.run ctx routine in
      {
        values = Context.get_values ctx t.Tensor.value;
        mma = routine.Context.mma;
        scans = List.sum (module Int) routine.Context.segments ~f:(fun o -> scans o.LL.llc);
      })

(* The default pipeline under [limits], as [Schedule.maybe_default_schedules] runs it (the fission
   seam, locals promoted on a GPU, the GPU preset and the schedule-aware merge rule read against
   those limits), and whether any segment's schedule carries a [Fold_mma]. *)
let pipeline ~limits =
  let emitted = ref false in
  let gpu = Sched.backend_is_gpu backend_name in
  let transform (o : LL.optimized) =
    let preset seg = if gpu then Sched.default_gpu ~limits seg else Sched.default_cpu seg in
    let zero_sched tns = if gpu then Sched.zero_expansion ~limits tns else [] in
    let segments =
      Sched.fission_scheduled ~promote_locals:gpu
        ?keep_mapping:(Sched.fission_keep_mapping ~is_gpu:gpu ~limits)
        ~preset ~zero_sched ~static_indices:[] o
    in
    List.iter segments ~f:(fun (_, _, sched, _) ->
        if List.exists sched ~f:(function Sched.Fold_mma _ -> true | _ -> false) then
          emitted := true);
    List.map segments ~f:(fun (_, _, _, post) -> post)
  in
  (transform, emitted)

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
        fold.values composed.values ~f:(close ~tol:1e-4));
  (* The transform's own refusal: a fold whose value tensor is indexed by the query. The scalar fold
     ships (the scan is there), no [Tile_mma], and the compile does not fail. *)
  (let why = "a value tensor indexed by the query (the block micro-kernel does not tensorize)" in
   let seq = 64 and d_k = 32 and block = 16 in
   let model = model_query_indexed_v in
   let composed = forward ~model ~name:"osbm_qv_composed" ~on:false ~block:0 ~seq ~d_k () in
   let fold = forward ~model ~name:"osbm_qv_fold" ~on:true ~block ~seq ~d_k () in
   p (why ^ ": the scalar fold ships, one scan") (fold.scans = 1);
   p (why ^ ": no Tile_mma emitted") (fold.mma.Ir.C_syntax.statements = 0);
   p_all2
     (why ^ ": matches the composed output within 1e-4 relative")
     fold.values composed.values ~f:(close ~tol:1e-4));
  (* The device's refusal: the same fold under a grid cap the folded launch (every row loop a [Grid]
     axis) exceeds. Read through the pipeline's own seam with the limits substituted: with the
     device's limits the preset emits the op where the units exist, under the cap never. *)
  let seq = 64 and d_k = 32 and block = 16 in
  let capable = Lazy.force f32_mma in
  let limits = Context.hardware_limits (Context.auto ()) in
  let composed = forward ~name:"osbm_cap_composed" ~on:false ~block:0 ~seq ~d_k () in
  let device_transform, device_emitted = pipeline ~limits in
  let device =
    forward ~lowered_transform:device_transform ~name:"osbm_cap_device" ~on:true ~block ~seq ~d_k ()
  in
  let capped_transform, capped_emitted = pipeline ~limits:{ limits with BI.max_grid_yz = Some 1 } in
  let capped =
    forward ~lowered_transform:capped_transform ~name:"osbm_cap_capped" ~on:true ~block ~seq ~d_k ()
  in
  gated ~when_:capable ~on:backend_name
    "under the device's limits the preset emits Fold_mma for the fold" !device_emitted;
  gated ~when_:(not capable) ~on:backend_name "without f32 MMA units the preset emits no Fold_mma"
    (not !device_emitted);
  p "a grid cap of 1 keeps the scalar fold: no Fold_mma emitted, no Tile_mma"
    ((not !capped_emitted) && capped.mma.Ir.C_syntax.statements = 0 && capped.scans = 1);
  p_all2 "under the grid cap the output matches the composed output within 1e-4 relative"
    capped.values composed.values ~f:(close ~tol:1e-4);
  p_all2 "under the device's limits the output matches the composed output within 1e-4 relative"
    device.values composed.values ~f:(close ~tol:1e-4)

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
