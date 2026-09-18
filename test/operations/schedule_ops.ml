(* Schedule IR, Phase S1 (docs/proposals/schedule-ir-optops.md): executed parity of scheduled
   kernels against their unscheduled twins, plus structural checks on the generated source read from
   [build_files/].

   Covered here: [Split] with a dividing factor (the remainder guard folds away), [Split] with a
   non-dividing factor (the guard survives on every backend -- it is part of the loop structure, not
   launch guarding), [Swap] of a perfectly nested pair, and the default GPU annotator preset
   ([Schedule.default_gpu]) on a two-nest elementwise kernel (unequal Grid extents => launch-extent
   guard on GPU backends) and on a matmul (reduction; whatever the conservative analysis decides,
   the values must match the unscheduled twin).

   On GPU backends (Metal locally, CUDA in CI) annotated kernels execute with real grid /
   threadgroup dimensions; on the C backends annotated loops legally fall back to serial loops.
   Every printed boolean holds on every backend -- structural expectations are dispatched on the
   configured backend. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments

let () = Utils.settings.output_debug_files_in_build_directory <- true

open Verdict.Claims

let approx a b = Float.(abs (a - b) < 1e-4)
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_gpu = Sched.backend_is_gpu backend_name

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name

let has_hardware_regs src =
  String.is_substring src ~substring:"gid." || String.is_substring src ~substring:"blockIdx."

(* The outermost statement-level loop of the optimized code: its index symbol and body. *)
let rec first_loop (llc : LL.t) =
  match llc with
  | LL.Seq (a, b) -> ( match first_loop a with Some r -> Some r | None -> first_loop b)
  | LL.For_loop { index; body; _ } -> Some (index, body)
  | _ -> None

let first_loop_exn llc = Option.value_exn ~here:[%here] (first_loop llc)

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let () =
  let av = Array.init 32 ~f:(fun i -> Float.of_int i *. 0.5) in
  let bv =
    Array.init 32 ~f:(Ll_test.cycle_flat ~dims:[| 4; 8 |] ~modulus:7 ~offset:(-3.) ~stride:1.)
  in
  let expected_c = Array.init 32 ~f:(fun i -> av.(i) +. bv.(i)) in
  let a = TDSL.ndarray av ~label:[ "a" ] ~output_dims:[ 4; 8 ] () in
  let b = TDSL.ndarray bv ~label:[ "b" ] ~output_dims:[ 4; 8 ] () in
  let run_variant ~name ~transform =
    let%op c = a + b in
    let comp = named name (Train.forward c) in
    let ctx = Context.auto () in
    let ctx, routine =
      Context.compile ~lowered_transform:(fun o -> [ transform o ]) ctx comp Ir.Indexing.Empty
    in
    let ctx = Context.run ctx routine in
    Context.get_values ctx c.Tensor.value
  in

  (* --- Split with a dividing factor: the remainder guard must fold away --- *)
  let got_div =
    run_variant ~name:"split_div" ~transform:(fun opt ->
        let sym, _ = first_loop_exn opt.LL.llc in
        let op, _, _ = Sched.split ~axis:sym ~factor:2 ~outer:LL.Grid ~inner:LL.Workgroup in
        Sched.apply [ op ] opt)
  in
  p_all2 "split (dividing factor) values correct" got_div expected_c ~f:approx;
  (let src = Generated.read "split_div" in
   let has s = String.is_substring src ~substring:s in
   let ok =
     if on_gpu then
       (* Hardware bindings for the split pair, the inner elementwise loop stays serial, and no
          guard: 2 divides 4 (fold) and extents are slot-uniform (no launch guard). *)
       has_hardware_regs src && has "for (" && not (has "if (")
     else (not (has_hardware_regs src)) && has "for (" && not (has "if (")
   in
   p "split (dividing factor) structure as expected" ok);

  (* --- Split with a non-dividing factor: the remainder guard survives on every backend --- *)
  let got_rem =
    run_variant ~name:"split_rem" ~transform:(fun opt ->
        let sym, _ = first_loop_exn opt.LL.llc in
        let op, _, _ = Sched.split ~axis:sym ~factor:3 ~outer:LL.Grid ~inner:LL.Workgroup in
        Sched.apply [ op ] opt)
  in
  p_all2 "split (remainder) values correct" got_rem expected_c ~f:approx;
  Generated.assert_emits ~routine:"split_rem" ~contains:"if (" "split (remainder) guard survives";

  (* --- Swap of the perfectly nested elementwise pair --- *)
  let got_swap =
    run_variant ~name:"swap_ij" ~transform:(fun opt ->
        let i, body = first_loop_exn opt.LL.llc in
        let j, _ = first_loop_exn body in
        Sched.apply [ Sched.Swap { outer = i; inner = j } ] opt)
  in
  p_all2 "swap values correct" got_swap expected_c ~f:approx;
  (let src = Generated.read "swap_ij" in
   (* After the interchange the extent-8 loop is outermost: [<= 7] appears before [<= 3]. *)
   let idx7 = String.substr_index src ~pattern:"<= 7" in
   let idx3 = String.substr_index src ~pattern:"<= 3" in
   p "swap reorders the loop bounds"
     (match (idx7, idx3) with Some i7, Some i3 -> i7 < i3 | _ -> false));

  (* --- Retype to Vectorized (gh-ocannl-164): a pragma-annotated serial loop on the C backends, a
     plain serial loop on GPU backends (empty [vectorize_pragma]); values must match either way, and
     on cc the pragma-annotated source must actually compile under the real C compiler. --- *)
  let got_vec =
    run_variant ~name:"vec_inner" ~transform:(fun opt ->
        let _, body = first_loop_exn opt.LL.llc in
        let j, _ = first_loop_exn body in
        Sched.apply [ Sched.Retype { axis = j; ty = LL.Vectorized } ] opt)
  in
  p_all2 "vectorized retype values correct" got_vec expected_c ~f:approx;
  (let src = Generated.read "vec_inner" in
   let has s = String.is_substring src ~substring:s in
   let ok =
     if on_gpu then has "for (" && not (has "#pragma")
     else
       (* Explicit SIMD emission (GCC/Clang vector extensions) supersedes the pragma hints for
          eligible bodies; the serial remainder loop follows the vector main loop. *)
       has "vector_size" && has "__builtin_memcpy" && not (has "#pragma")
   in
   p "vectorized retype structure as expected" ok);

  (* --- The default GPU annotator on a two-nest elementwise kernel (4x8 and 6x8: unequal Grid
     extents => launch-extent guard on GPU backends) --- *)
  let ev = Array.init 48 ~f:(fun i -> Float.of_int i *. 0.25) in
  let fv =
    Array.init 48 ~f:(Ll_test.cycle_flat ~dims:[| 6; 8 |] ~modulus:5 ~offset:1. ~stride:1.)
  in
  let expected_c2 = Array.init 48 ~f:(fun i -> ev.(i) *. fv.(i)) in
  let e = TDSL.ndarray ev ~label:[ "e" ] ~output_dims:[ 6; 8 ] () in
  let f = TDSL.ndarray fv ~label:[ "f" ] ~output_dims:[ 6; 8 ] () in
  let%op c1 = a + b in
  let%op c2 = e *. f in
  let combo = named "combo_default" (Asgns.sequence [ Train.forward c1; Train.forward c2 ]) in
  let sched_len = ref (-1) in
  let ctx = Context.auto () in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        let sched = Sched.default_gpu ~min_parallel:1 opt in
        sched_len := List.length sched;
        [ Sched.apply sched opt ])
      ctx combo Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got_c1 = Context.get_values ctx c1.Tensor.value in
  let got_c2 = Context.get_values ctx c2.Tensor.value in
  p_all2 "default annotator add-nest values correct" got_c1 expected_c ~f:approx;
  p_all2 "default annotator multiply-nest values correct" got_c2 expected_c2 ~f:approx;
  p "default annotator schedules both nests" (!sched_len = 4);
  (let src = Generated.read "combo_default" in
   let has s = String.is_substring src ~substring:s in
   let ok =
     if on_gpu then
       (* Both nests fully hardware-bound (no loops left) and the smaller Grid nest guarded. *)
       has_hardware_regs src && has "if (" && not (has "for (")
     else (not (has_hardware_regs src)) && has "for ("
   in
   p "default annotator structure as expected" ok);

  (* --- The default GPU annotator on a matmul: values must match the unscheduled twin whatever the
     conservative analysis decides (reduction loops stay serial in the preset) --- *)
  let mav =
    Array.init 20 ~f:(Ll_test.cycle_flat ~dims:[| 4; 5 |] ~modulus:7 ~offset:0. ~stride:0.5)
  in
  let mbv =
    Array.init 30 ~f:(Ll_test.cycle_flat ~dims:[| 5; 6 |] ~modulus:11 ~offset:(-4.) ~stride:1.)
  in
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ 5 ] ~output_dims:[ 4 ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ 6 ] ~output_dims:[ 5 ] () in
  let run_mm ~name ~transform =
    let%op mc = ma * mb in
    let comp = named name (Train.forward mc) in
    let ctx = Context.auto () in
    let ctx, routine =
      Context.compile ~lowered_transform:(fun o -> [ transform o ]) ctx comp Ir.Indexing.Empty
    in
    let ctx = Context.run ctx routine in
    Context.get_values ctx mc.Tensor.value
  in
  let mm_serial = run_mm ~name:"ops_mm_serial" ~transform:(fun opt -> opt) in
  let mm_sched =
    run_mm ~name:"mm_default" ~transform:(fun opt ->
        Sched.apply (Sched.default_gpu ~min_parallel:1 opt) opt)
  in
  p_all2 "default annotator matmul values match the serial twin" mm_sched mm_serial ~f:approx;

  (* --- Privatize under If guards (PR #91 review; gh-ocannl-730): a lane-guarded accumulation [if
     (w == 0) c += va[k]] must carry the guard onto the synthesized init-load and store-back (stale
     lanes would otherwise clobber the result). A guard that VARIES across the accumulation is
     classified instead of rejected: one free of hardware-typed loop symbols masks iterations of one
     thread's own accumulator, so it stays on the update and off the transfers (this is what a
     reduction-axis [Pad] leaves behind, and what lets the GPU blocktile family pad); one that mixes
     a thread-selecting symbol into a comparison that is not the target's own index is still
     rejected. Structural checks on hand-built IR, backend-independent. --- *)
  let cacc = TDSL.ndarray [| 0. |] ~label:[ "cacc" ] ~output_dims:[ 1 ] () in
  let va = TDSL.ndarray [| 1.; 2.; 3.; 4. |] ~label:[ "va" ] ~output_dims:[ 4 ] () in
  let single = Ir.Ops.single in
  let iprec = Ir.Ops.index_prec () in
  let fake llc : LL.optimized =
    {
      LL.traced_store = Hashtbl.create (module Ir.Tnode);
      optimize_ctx = LL.empty_optimize_ctx ();
      llc;
      merge_node = None;
      workgroup_shared = Set.empty (module Ir.Tnode);
      simdgroup_fragments = Set.empty (module Ir.Tnode);
      swizzled = Map.empty (module Ir.Tnode);
      pipelined = Map.empty (module Ir.Tnode);
      zero_fringe = Set.empty (module Ir.Tnode);
      flip_candidates = [];
      spliced_rbw = Base.Set.empty (module Ir.Tnode);
      source = llc;
    }
  in
  let guarded_accum ~cond_of =
    let w = Ir.Indexing.get_symbol () and k = Ir.Indexing.get_symbol () in
    let accum =
      LL.Set
        {
          tn = cacc.Tensor.value;
          idcs = [| Ir.Indexing.Fixed_idx 0 |];
          llsc =
            LL.Binop
              ( Ir.Ops.Add,
                (LL.Get (cacc.Tensor.value, [| Ir.Indexing.Fixed_idx 0 |]), single),
                (LL.Get (va.Tensor.value, [| Ir.Indexing.Iterator k |]), single) );
          debug = "";
        }
    in
    let llc =
      LL.For_loop
        {
          index = w;
          from_ = 0;
          to_ = 3;
          axis = LL.Workgroup;
          body =
            LL.For_loop
              {
                index = k;
                from_ = 0;
                to_ = 3;
                axis = LL.Serial;
                body = LL.If { cond = (cond_of ~w ~k, iprec); body = accum };
              };
        }
    in
    (llc, k)
  in
  let doc_to_str doc =
    let b = Buffer.create 256 in
    PPrint.ToBuffer.pretty 0.7 110 b doc;
    Buffer.contents b
  in
  let lane_llc, lane_k =
    guarded_accum ~cond_of:(fun ~w ~k:_ ->
        LL.Binop
          (Ir.Ops.Cmpeq, (LL.Embed_index (Ir.Indexing.Iterator w), iprec), (LL.Constant 0., iprec)))
  in
  let lane_res =
    Sched.apply [ Sched.Privatize { target = cacc.Tensor.value; over = lane_k } ] (fake lane_llc)
  in
  let lane_src = doc_to_str (LL.to_doc () lane_res.LL.llc) in
  let count_sub sub =
    String.substr_index_all lane_src ~may_overlap:false ~pattern:sub |> List.length
  in
  p "lane-guarded privatize gates init and store-back"
    (count_sub "if " = 3 && String.is_substring lane_src ~substring:"acc_");
  let iter_llc, iter_k =
    guarded_accum ~cond_of:(fun ~w:_ ~k ->
        LL.Binop
          (Ir.Ops.Cmplt, (LL.Embed_index (Ir.Indexing.Iterator k), iprec), (LL.Constant 3., iprec)))
  in
  let iter_res =
    Sched.apply [ Sched.Privatize { target = cacc.Tensor.value; over = iter_k } ] (fake iter_llc)
  in
  let iter_src = doc_to_str (LL.to_doc () iter_res.LL.llc) in
  (* The mask fires within one thread's own accumulation: it stays on the update (one [if]) and the
     init-load and store-back run unguarded, so the slots it skips round-trip unchanged. *)
  p "iteration mask stays on the update and off privatize's transfers"
    (String.substr_index_all iter_src ~may_overlap:false ~pattern:"if " |> List.length |> ( = ) 1);
  (* Mixing the thread-selecting symbol into the same varying condition is still rejected: it would
     leave non-updating lanes storing back a stale accumulator. *)
  let mixed_llc, mixed_k =
    guarded_accum ~cond_of:(fun ~w ~k ->
        LL.Binop
          ( Ir.Ops.Mul,
            ( LL.Binop
                ( Ir.Ops.Cmplt,
                  (LL.Embed_index (Ir.Indexing.Iterator k), iprec),
                  (LL.Constant 3., iprec) ),
              iprec ),
            ( LL.Binop
                ( Ir.Ops.Cmpeq,
                  (LL.Embed_index (Ir.Indexing.Iterator w), iprec),
                  (LL.Constant 0., iprec) ),
              iprec ) ))
  in
  p "thread-selecting varying guard rejected by privatize"
    (match
       try
         ignore
           (Sched.apply
              [ Sched.Privatize { target = cacc.Tensor.value; over = mixed_k } ]
              (fake mixed_llc)
             : LL.optimized);
         None
       with Invalid_argument msg -> Some msg
     with
    | Some msg -> String.is_substring msg ~substring:"mask on one thread's own iterations"
    | None -> false);

  (* --- Device-limit-aware tile sizes: the annotator clamps the (configured) block size to the
     backend's max-threads-per-workgroup limit, and [check_hardware_limits] (called by backend
     [compile] after any transform) rejects kernels exceeding the workgroup-size or shared-memory
     capacity. Backend-independent: limits are passed explicitly. --- *)
  let tight_limits =
    { Ir.Backend_intf.no_hardware_limits with Ir.Backend_intf.max_threads_per_workgroup = Some 4 }
  in
  let clamp_sched = ref [] in
  let got_clamp =
    run_variant ~name:"limit_clamp" ~transform:(fun opt ->
        (* Without the clamp, block_size 8 covers the extent-8 inner loop => two Retypes and no
           Split; clamped to 4, the inner loop must be split by 4. *)
        let sched = Sched.default_gpu ~min_parallel:1 ~block_size:8 ~limits:tight_limits opt in
        clamp_sched := sched;
        Sched.apply sched opt)
  in
  p_all2 "device-limit clamp values correct" got_clamp expected_c ~f:approx;
  p "device-limit clamp splits by the max-threads limit"
    (List.exists !clamp_sched ~f:(function
      | Sched.Split { factor; inner = LL.Workgroup; _ } -> factor = 4
      | _ -> false));
  p "workgroup size over the limit rejected"
    (match
       try
         ignore
           (Sched.check_hardware_limits ~name:"limit_threads"
              ~limits:
                {
                  Ir.Backend_intf.no_hardware_limits with
                  Ir.Backend_intf.max_threads_per_workgroup = Some 2;
                }
              (fake lane_llc)
             : unit);
         None
       with Utils.User_error msg -> Some msg
     with
    | Some msg -> String.is_substring msg ~substring:"threads per workgroup"
    | None -> false);
  (* [a] is 4x8 single precision = 128 shared bytes when staged. *)
  let shared_opt =
    { (fake LL.Noop) with LL.workgroup_shared = Set.singleton (module Ir.Tnode) a.Tensor.value }
  in
  let smem_limits bytes =
    {
      Ir.Backend_intf.no_hardware_limits with
      Ir.Backend_intf.max_workgroup_memory_bytes = Some bytes;
    }
  in
  p "shared tiles within the memory limit accepted"
    (try
       Sched.check_hardware_limits ~name:"limit_smem_ok" ~limits:(smem_limits 128) shared_opt;
       true
     with Utils.User_error _ -> false);
  p "shared tiles over the memory limit rejected"
    (match
       try
         Sched.check_hardware_limits ~name:"limit_smem" ~limits:(smem_limits 64) shared_opt;
         None
       with Utils.User_error msg -> Some msg
     with
    | Some msg -> String.is_substring msg ~substring:"workgroup-shared"
    | None -> false)
