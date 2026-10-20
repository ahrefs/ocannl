(* Axis-type annotations, Phase B (docs/proposals/axis-types-for-loops.md): executed parity of
   hardware-annotated kernels against their all-Serial twins, plus structural checks on the
   generated source read from [build_files/] (as-compiled structure; re-lowering reflects mutated
   global state).

   The same computation -- two elementwise nests, [c1 = a + b] over 4x8 and [c2 = e *. f] over 6x8,
   combined into one routine -- is compiled twice: once as lowered (all-Serial), and once with the
   outer loops annotated [Grid] and the inner loops [Workgroup] via the [?lowered_transform] seam of
   [Context.compile]. The Grid extents differ (4 vs 6), so the launch takes the per-slot max (6) and
   the smaller nest's body gets an [If (i < 4)] launch-extent guard (construct-then-fold; here it
   must survive, the extents genuinely differ).

   On GPU backends (Metal locally, CUDA in CI) the annotated kernel executes with real grid /
   threadgroup dimensions; on the C backends the annotated loops legally fall back to serial [for]
   loops (no barriers involved). Every printed boolean holds on every backend -- the structural
   expectations are dispatched on the configured backend. A third variant exercises [Unrolled]: the
   inner loop of a 4x8 nest emits 8 repeated blocks with the index bound as a per-block constant
   (backend-independent rendering). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Asgns = Ir.Assignments

let () = Utils.settings.output_debug_files_in_build_directory <- true

open Verdict.Claims

let approx_tol _ = 1e-5
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name

(* Annotate every top-level loop nest of the optimized code: the outermost loop gets [outer_axis],
   the immediately nested loop [inner_axis]; deeper loops stay [Serial]. *)
let annotate ~outer_axis ~inner_axis (opt : LL.optimized) : LL.optimized =
  let rec map_inner (llc : LL.t) : LL.t =
    match llc with
    | LL.Seq (x, y) -> LL.Seq (map_inner x, map_inner y)
    | LL.For_loop fc -> LL.For_loop { fc with axis = inner_axis }
    | other -> other
  in
  let rec map_outer (llc : LL.t) : LL.t =
    match llc with
    | LL.Seq (x, y) -> LL.Seq (map_outer x, map_outer y)
    | LL.For_loop fc -> LL.For_loop { fc with axis = outer_axis; body = map_inner fc.body }
    | other -> other
  in
  { opt with llc = map_outer opt.llc }

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let () =
  let av = Array.init 32 ~f:(fun i -> Float.of_int i *. 0.5) in
  let bv = Array.init 32 ~f:(fun i -> Float.of_int (i % 7) -. 3.) in
  let ev = Array.init 48 ~f:(fun i -> Float.of_int i *. 0.25) in
  let fv = Array.init 48 ~f:(fun i -> Float.of_int ((i % 5) + 1)) in
  let expected_c1 = Array.init 32 ~f:(fun i -> av.(i) +. bv.(i)) in
  let expected_c2 = Array.init 48 ~f:(fun i -> ev.(i) *. fv.(i)) in
  let a = TDSL.ndarray av ~label:[ "a" ] ~output_dims:[ 4; 8 ] () in
  let b = TDSL.ndarray bv ~label:[ "b" ] ~output_dims:[ 4; 8 ] () in
  let e = TDSL.ndarray ev ~label:[ "e" ] ~output_dims:[ 6; 8 ] () in
  let f = TDSL.ndarray fv ~label:[ "f" ] ~output_dims:[ 6; 8 ] () in

  (* --- Serial twin --- *)
  let%op c1s = a + b in
  let%op c2s = e *. f in
  let serial_comp =
    named "combo_serial" (Asgns.sequence [ Train.forward c1s; Train.forward c2s ])
  in
  let ctx_s = Context.auto () in
  let ctx_s, routine_s = Context.compile ctx_s serial_comp Ir.Indexing.Empty in
  let ctx_s = Context.run ctx_s routine_s in
  let got_c1s = Context.get_values ctx_s c1s.Tensor.value in
  let got_c2s = Context.get_values ctx_s c2s.Tensor.value in
  p_close "serial add-nest values correct" got_c1s expected_c1 ~tol:approx_tol;
  p_close "serial multiply-nest values correct" got_c2s expected_c2 ~tol:approx_tol;

  (* --- Grid+Workgroup annotated twin (unequal Grid extents: 4 vs 6 => guard on the 4-nest) --- *)
  let%op c1a = a + b in
  let%op c2a = e *. f in
  let annot_comp = named "combo_annot" (Asgns.sequence [ Train.forward c1a; Train.forward c2a ]) in
  let ctx_a = Context.auto () in
  let ctx_a, routine_a =
    Context.compile
      ~lowered_transform:(fun o -> [ (annotate ~outer_axis:LL.Grid ~inner_axis:LL.Workgroup) o ])
      ctx_a annot_comp Ir.Indexing.Empty
  in
  let ctx_a = Context.run ctx_a routine_a in
  let got_c1a = Context.get_values ctx_a c1a.Tensor.value in
  let got_c2a = Context.get_values ctx_a c2a.Tensor.value in
  p_close "annotated add-nest values match the serial twin" got_c1a got_c1s ~tol:approx_tol;
  p_close "annotated multiply-nest values match the serial twin" got_c2a got_c2s ~tol:approx_tol;
  (let src = Generated.read "combo_annot" in
   let has s = String.is_substring src ~substring:s in
   (* This is intentionally dialect identity: the assertion names the MSL, CUDA/HIP, and C spellings
      of the hardware-axis registers and serial-loop fallback. *)
   let structure_ok =
     if String.is_substring backend_name ~substring:"metal" then
       (* Hardware bindings from the gid/lid extra args, no loops left, and the launch-extent guard
          on the smaller Grid nest. *)
       has "gid.x" && has "lid.x" && has "if (" && not (has "for (")
     else if
       String.is_substring backend_name ~substring:"cuda"
       || String.is_substring backend_name ~substring:"hip"
     then has "blockIdx.x" && has "threadIdx.x" && has "if (" && not (has "for (")
     else
       (* C backends: legal serial fallback, no hardware registers, no guard (the serial loop
          iterates the true extent). *)
       has "for (" && (not (has "gid")) && not (has "blockIdx")
   in
   p "annotated kernel structure as expected" structure_ok);

  (* --- Unrolled inner loop (backend-independent rendering) --- *)
  let%op c3u = a + b in
  let unroll_comp = named "unroll_annot" (Train.forward c3u) in
  let ctx_u = Context.auto () in
  let ctx_u, routine_u =
    Context.compile
      ~lowered_transform:(fun o -> [ (annotate ~outer_axis:LL.Serial ~inner_axis:LL.Unrolled) o ])
      ctx_u unroll_comp Ir.Indexing.Empty
  in
  let ctx_u = Context.run ctx_u routine_u in
  let got_c3u = Context.get_values ctx_u c3u.Tensor.value in
  p_close "unrolled variant values correct" got_c3u expected_c1 ~tol:approx_tol;
  (let src = Generated.read "unroll_annot" in
   (* The inner extent-8 loop unrolls into blocks binding the index to 0..7. *)
   p "unrolled kernel repeats the body with constant bindings"
     (String.is_substring src ~substring:" = 0;" && String.is_substring src ~substring:" = 7;"));

  (* --- Negative: partial hardware coverage is rejected (PR #89 review) --- *)
  (* Launch dimensions are global to the kernel: with the first nest annotated Grid+Workgroup
     (block.x = 8), a sibling Grid-only nest would execute its materialized write once per thread
     of the workgroup — racing or repeating the update. [validate_parallel] must reject this, on
     every backend (the check is backend-independent). *)
  let annotate_mixed (opt : LL.optimized) : LL.optimized =
    let first = ref true in
    let rec map_inner (llc : LL.t) : LL.t =
      match llc with
      | LL.Seq (x, y) -> LL.Seq (map_inner x, map_inner y)
      | LL.For_loop fc -> LL.For_loop { fc with axis = LL.Workgroup }
      | other -> other
    in
    let rec map_outer (llc : LL.t) : LL.t =
      match llc with
      | LL.Seq (x, y) -> LL.Seq (map_outer x, map_outer y)
      | LL.For_loop fc ->
          let body = if !first then map_inner fc.body else fc.body in
          first := false;
          LL.For_loop { fc with axis = LL.Grid; body }
      | other -> other
    in
    { opt with llc = map_outer opt.llc }
  in
  let%op c1m = a + b in
  let%op c2m = e *. f in
  let mixed_comp = named "combo_mixed" (Asgns.sequence [ Train.forward c1m; Train.forward c2m ]) in
  let ctx_m = Context.auto () in
  match
    try
      ignore
        (Context.compile
           ~lowered_transform:(fun o -> [ annotate_mixed o ])
           ctx_m mixed_comp Ir.Indexing.Empty
          : Context.t * Context.routine);
      None
    with Invalid_argument msg -> Some msg
  with
  | Some msg ->
      p "partial hardware coverage rejected"
        (String.is_substring msg ~substring:"all active hardware dimensions")
  | None -> p "partial hardware coverage rejected" false
